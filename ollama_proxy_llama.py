#!/usr/bin/env python3
# Copyright 2026 Satisfanly Ltd
#
# OKO OS is a product of Satisfanly Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Ollama API -> dynamically managed llama-server compatibility shim.

Front-end API (for Ollama CLI/clients):
    http://127.0.0.1:11435

Managed back-end API (llama-server):
    http://127.0.0.1:8080

The proxy loads model definitions from a YAML file. Model names are the
top-level keys directly. Before YAML parsing, `%max_free_ram%` is expanded
anywhere in the file to Linux MemAvailable minus a 5 GiB reserve, expressed in
MiB (suitable for llama-server `--cache-ram`). Configuration is strict about
field names: unknown fields are rejected. `cmd` is required. Omitted generation controls stay omitted
at the llama-server request boundary unless the YAML explicitly configures them;
the shim does not silently replace backend/model sampler defaults. `external-chain` is an optional list of shell
command strings. All configured chain programs are started at proxy startup and
kept alive for the proxy lifetime. The proxy appends `--listen-host`,
`--listen-port`, `--target-host`, and `--target-port` to every command and wires
each model as public gateway -> chain -> private model-bound Ollama ingress. If
any chain process crashes the complete chain is restarted with a warning. Each
external-chain program writes both stdout and stderr to its own append-only
log at `/var/log/oko-ai-external-chain/<proxy_name>.log`; `proxy_name` is derived
from the command's executable or interpreter script name. Chain restarts append
to the existing log and add a startup marker instead of truncating it. After
the last public user request for a model disconnects, stale internal requests get
a 3 second grace period before the chain is forcibly restarted. `alias` may be a
string or list of strings and adds alternative Ollama-visible names for the
model. `prio` is a non-negative integer request priority; smaller values have
higher priority (default: 100). Scheduler activity follows public user requests,
not the internal requests produced by the chain. A higher-priority public request
therefore starts runner preemption/loading immediately: an active generation is
checkpointed exactly, while a runner between internal requests saves reusable
idle prompt/KV state before it is unloaded. `sysf` defaults to no
system prompt. A plain `sysf` is a fallback and is replaced by a client-supplied
system prompt;
`sysf (ro)` allows a client system prompt only when it exactly matches the
configured prompt, while `sysf (force)` intentionally replaces any client system
prompt with the configured value. `think` accepts `on`/`off`; when omitted from
YAML it imposes no shim default. When configured, a client-supplied value
overrides it unless configured as `think (ro)`, in which case an equal value is
accepted and only a differing value is rejected, or `think (force)`, in which
case the configured value intentionally wins. `reasoning-effort` is an optional template-specific
reasoning level (for example `low`, `medium`, `high`, or `xhigh`); a client
`reasoning_effort` overrides a plain configured value, while `(ro)` and
`(force)` provide the same match-only/forced policies. `reasoning-preserve`
controls whether prior assistant reasoning traces are retained in full chat history;
it accepts `on`/`off` (or true/false), and supports `(ro)` / `(force)` as well.
For compatibility the proxy sets both llama.cpp's `preserve_reasoning` capability
control and the `preserve_thinking` template kwarg used by Qwen/Gemma templates.
`think: off` always disables reasoning for the current turn regardless of effort.
The same `(ro)` and `(force)` policies
apply to generation parameters. Managed
commands must listen on 127.0.0.1:8080. Discovery endpoints `/api/tags`
and `/api/show` never start or switch a backend; `/api/show` enriches
metadata only when the requested model is already running.

Transparency contract:
- The public gateway forwards generation request bytes through `external-chain`
  unchanged apart from normal HTTP hop-by-hop headers.
- The private Ollama<->llama-server adapter performs only representation changes
  required by the two APIs. It never invents tool-call IDs, guesses tool/result
  associations, repairs malformed tool arguments, weakens JSON schemas, or
  silently discards unsupported request controls/backend stream events.
- Explicit YAML values are policy and may intentionally change inference.
- Client `options` also retain the shim's explicit llama.cpp extensions
  (`dry_*`, `mirostat*`, and `reasoning_budget`). They are never injected when
  omitted; when supplied, they are forwarded as deliberate caller policy rather
  than pretending to be stock Ollama options.
- Normal Ollama controls such as explicit `keep_alive`, `truncate`/`shift`, and
  runner-only options such as `num_ctx` are accepted for client compatibility.
  When this managed architecture cannot enact one of those controls per request,
  the limitation is reported on stderr instead of turning an otherwise valid
  Ollama request into a shim-only HTTP error. Unknown Ollama fields/options are
  likewise ignored with a stderr warning, matching Ollama's permissive decoding.
- Behaviors that would require fabricating a different prompt or transcript still
  fail explicitly rather than being silently approximated. Legacy `/api/generate`
  Go-template/context overrides are examples. Normal
  non-raw `/api/generate` uses llama-server's chat template and therefore cannot
  reproduce Ollama's deprecated returned `context` token array exactly.
- Backend generation is always streamed internally to make checkpoint suspend/
  resume possible; `stream:false` is reconstructed from that stream without
  changing the final semantic message. Tool-call deltas are assembled by index
  because llama-server exposes them as OpenAI-style incremental fragments.

Example models.yml:

    qwen3:latest:
      alias:
        - qwen3
        - main
      prio: 10
      cmd: >-
        llama-server -m /models/qwen3.gguf
        --host 127.0.0.1 --port 8080
      external-chain:
        - "/usr/local/bin/prog0 --mode normalize"
        - "/usr/local/bin/prog1 --mode filter"
      sysf (ro): /etc/ollama-proxy/qwen3-system.txt
      think (ro): on
      reasoning-effort: medium
      reasoning-preserve: on
      presence-penalty: 0.0
      repeat-penalty: 1.0
      min-p: 0.0
      top-k: 20
      top-p: 0.8
      temperature (force): 0.2
      dry-multiplier: 0.0
      dry-base: 1.75
      dry-allowed-length: 2
      dry-penalty-last-n: 8192
      reasoning-budget: -1

Command-line options:
    -l ADDR:PORT, --listen ADDR:PORT
        Listen address and port (default: 127.0.0.1:11435).

    -c FILE, --config FILE
        YAML model configuration file (required).

    --full-llama-responses-log [FILE]
        Append raw llama-server generation response-body bytes to FILE before
        any SSE/UTF-8/JSON parsing or response translation. If FILE is omitted,
        use ./full-llama-responses.log.

Environment variables (capability advertisement only; they do not inject
generation defaults):
    OLLAMA_PROXY_THINKING=1
    OLLAMA_PROXY_TOOLS=0

Dependencies:
    Python 3.11+
    pip install fastapi 'uvicorn[standard]' httpx pyyaml
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import contextlib
import hashlib
import json
import math
import os
import re
import signal
import shlex
import socket
import sys
import time
import yaml
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse


LLAMA_SERVER_URL = "http://127.0.0.1:8080"
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 11435
FULL_LLAMA_RESPONSES_LOG: Path | None = None
FULL_LLAMA_RESPONSES_LOG_FD: int | None = None
_FULL_LLAMA_RESPONSES_LOG_LOCK = asyncio.Lock()
BACKEND_START_TIMEOUT = 300.0
ADVERTISE_THINKING = os.getenv("OLLAMA_PROXY_THINKING", "1").lower() in {"1", "true", "yes", "on"}
ADVERTISE_TOOLS = os.getenv("OLLAMA_PROXY_TOOLS", "0").lower() in {"1", "true", "yes", "on"}

MODEL_CONFIG_FIELDS = {
    "cmd",
    "external-chain",
    "alias",
    "prio",
    "sysf",
    "think",
    "reasoning-effort",
    "reasoning-preserve",
    "presence-penalty",
    "repeat-penalty",
    "min-p",
    "top-k",
    "top-p",
    "temperature",
    "dry-multiplier",
    "dry-base",
    "dry-allowed-length",
    "dry-penalty-last-n",
    "reasoning-budget",
}

MODEL_CONFIG_DEFAULTS: dict[str, Any] = {
    "external-chain": [],
    "prio": 100,
    "sysf": None,
    # Transparency rule: omitted generation controls remain omitted at the
    # llama-server boundary. Only values explicitly configured here or supplied
    # by the client are sent. This avoids silently replacing backend/model
    # defaults with shim defaults.
    "think": None,
    "reasoning-effort": None,
    "reasoning-preserve": None,
    "presence-penalty": None,
    "repeat-penalty": None,
    "min-p": None,
    "top-k": None,
    "top-p": None,
    "temperature": None,
    "dry-multiplier": None,
    "dry-base": None,
    "dry-allowed-length": None,
    "dry-penalty-last-n": None,
    "reasoning-budget": None,
}

GENERATION_CONFIG_TO_OPTION = {
    "presence-penalty": "presence_penalty",
    "repeat-penalty": "repeat_penalty",
    "min-p": "min_p",
    "top-k": "top_k",
    "top-p": "top_p",
    "temperature": "temperature",
    "dry-multiplier": "dry_multiplier",
    "dry-base": "dry_base",
    "dry-allowed-length": "dry_allowed_length",
    "dry-penalty-last-n": "dry_penalty_last_n",
    "reasoning-budget": "reasoning_budget",
}

# Keep Ollama/config names separate from llama-server request names. Most
# happen to match, but treating them as the same namespace caused `(ro)` and
# `(force)` checks to break when a backend field was renamed. In particular,
# current llama-server builds accept `thinking_budget_tokens` per request;
# `reasoning_budget` is not the request field.
OLLAMA_OPTION_TO_LLAMA = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "typical_p": "typical_p",
    "seed": "seed",
    "repeat_penalty": "repeat_penalty",
    "repeat_last_n": "repeat_last_n",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "dry_multiplier": "dry_multiplier",
    "dry_base": "dry_base",
    "dry_allowed_length": "dry_allowed_length",
    "dry_penalty_last_n": "dry_penalty_last_n",
    "reasoning_budget": "thinking_budget_tokens",
    "mirostat": "mirostat",
    "mirostat_tau": "mirostat_tau",
    "mirostat_eta": "mirostat_eta",
    "num_keep": "n_keep",
}


class StrictConfigLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate keys and uses YAML 1.2 booleans."""


# PyYAML's default YAML 1.1 resolver treats unquoted `on`/`off` as booleans.
# This config deliberately uses those exact strings for the `think` field, so
# restrict implicit booleans to true/false as YAML 1.2 does. Copy the resolver
# table first so this does not mutate yaml.SafeLoader globally.
StrictConfigLoader.yaml_implicit_resolvers = {
    ch: [
        (tag, regexp)
        for tag, regexp in resolvers
        if tag != "tag:yaml.org,2002:bool"
    ]
    for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
StrictConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


def _construct_unique_mapping(loader: StrictConfigLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as e:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from e
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictConfigLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    aliases: tuple[str, ...]
    priority: int
    cmd: str
    external_chain: tuple[str, ...]
    system_prompt: str | None
    generation_defaults: dict[str, Any]
    read_only_options: frozenset[str]
    force_options: frozenset[str]
    system_prompt_read_only: bool
    system_prompt_force: bool
    think_default: bool | None
    think_read_only: bool
    think_force: bool
    reasoning_effort_default: str | None
    reasoning_effort_read_only: bool
    reasoning_effort_force: bool
    reasoning_preserve_default: bool | None
    reasoning_preserve_read_only: bool
    reasoning_preserve_force: bool


def _require_number(model_name: str, field: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"model {model_name!r}: {field!r} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"model {model_name!r}: {field!r} must be finite")
    return number


MAX_FREE_RAM_RESERVE_MIB = 5 * 1024


def _max_free_ram_mib() -> int:
    """Return Linux MemAvailable minus a 5 GiB safety reserve, in MiB."""
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                if not line.startswith("MemAvailable:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    break
                available_kib = int(parts[1])
                available_mib = available_kib // 1024
                return max(0, available_mib - MAX_FREE_RAM_RESERVE_MIB)
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot determine available RAM from /proc/meminfo: {e}") from e

    raise ValueError("cannot determine available RAM: /proc/meminfo has no MemAvailable entry")


CONFIG_VARIABLE_RESOLVERS = {
    "%max_free_ram%": _max_free_ram_mib,
}


def _expand_config_variables(text: str) -> str:
    """Expand proxy-owned variables in raw YAML before YAML parsing.

    Expansion happens on the raw file so a variable can appear anywhere, including
    inside command strings or as a complete unquoted scalar. RAM variables are
    rendered in MiB because llama-server --cache-ram expects MiB.
    """
    for variable, resolver in CONFIG_VARIABLE_RESOLVERS.items():
        if variable in text:
            text = text.replace(variable, str(resolver()))
    return text


def load_models_config(path: str | Path) -> dict[str, ModelConfig]:
    config_path = Path(path).expanduser().resolve()
    try:
        config_text = config_path.read_text(encoding="utf-8")
        config_text = _expand_config_variables(config_text)
        raw = yaml.load(config_text, Loader=StrictConfigLoader)
    except (OSError, yaml.YAMLError, ValueError) as e:
        raise ValueError(f"cannot read models config {str(config_path)!r}: {e}") from e

    if not isinstance(raw, dict) or not raw:
        raise ValueError("models config must be a non-empty YAML mapping")

    models: dict[str, ModelConfig] = {}
    for name, table in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("model names must be non-empty strings")
        if name != name.strip():
            raise ValueError(f"model name {name!r} must not have leading/trailing whitespace")
        if not isinstance(table, dict):
            raise ValueError(f"model {name!r} must be a YAML mapping")

        normalized_table: dict[str, Any] = {}
        read_only_config_keys: set[str] = set()
        force_config_keys: set[str] = set()
        unexpected: list[str] = []
        for raw_key, value in table.items():
            if not isinstance(raw_key, str):
                unexpected.append(repr(raw_key))
                continue

            key = raw_key
            modifier: str | None = None
            if raw_key.endswith(" (ro)"):
                key = raw_key[:-5]
                modifier = "ro"
            elif raw_key.endswith(" (force)"):
                key = raw_key[:-8]
                modifier = "force"

            if key not in MODEL_CONFIG_FIELDS:
                unexpected.append(raw_key)
                continue
            if modifier is not None and key not in GENERATION_CONFIG_TO_OPTION and key not in {"sysf", "think", "reasoning-effort", "reasoning-preserve"}:
                raise ValueError(
                    f"model {name!r}: '({modifier})' is only valid for generation parameters, "
                    f"'sysf', 'think', 'reasoning-effort', or 'reasoning-preserve'; cannot use it on {key!r}"
                )
            if key in normalized_table:
                raise ValueError(
                    f"model {name!r}: field {key!r} is configured more than once "
                    f"(possibly with different modifiers)"
                )

            normalized_table[key] = value
            if modifier == "ro":
                read_only_config_keys.add(key)
            elif modifier == "force":
                force_config_keys.add(key)

        if unexpected:
            raise ValueError(
                f"model {name!r}: unexpected field(s): " + ", ".join(sorted(unexpected))
            )
        if "cmd" not in normalized_table:
            raise ValueError(f"model {name!r}: missing required field: cmd")

        effective = {**MODEL_CONFIG_DEFAULTS, **normalized_table}

        cmd = effective["cmd"]
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError(f"model {name!r}: 'cmd' must be a non-empty string")

        external_chain_value = effective["external-chain"]
        if external_chain_value is None:
            external_chain_items: list[Any] = []
        elif isinstance(external_chain_value, list):
            external_chain_items = external_chain_value
        else:
            raise ValueError(
                f"model {name!r}: 'external-chain' must be a list of shell command strings or null"
            )
        external_chain_list: list[str] = []
        for index, chain_cmd in enumerate(external_chain_items):
            if not isinstance(chain_cmd, str) or not chain_cmd.strip():
                raise ValueError(
                    f"model {name!r}: external-chain[{index}] must be a non-empty string"
                )
            external_chain_list.append(chain_cmd.strip())
        external_chain = tuple(external_chain_list)

        priority = effective["prio"]
        if not isinstance(priority, int) or isinstance(priority, bool) or priority < 0:
            raise ValueError(f"model {name!r}: 'prio' must be a non-negative integer")

        alias_value = normalized_table.get("alias", [])
        if alias_value is None:
            alias_items: list[Any] = []
        elif isinstance(alias_value, str):
            alias_items = [alias_value]
        elif isinstance(alias_value, list):
            alias_items = alias_value
        else:
            raise ValueError(
                f"model {name!r}: 'alias' must be a string, a list of strings, or null"
            )

        aliases_list: list[str] = []
        seen_aliases: set[str] = set()
        for alias in alias_items:
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError(
                    f"model {name!r}: every alias must be a non-empty string"
                )
            if alias != alias.strip():
                raise ValueError(
                    f"model {name!r}: alias {alias!r} must not have leading/trailing whitespace"
                )
            if alias == name:
                raise ValueError(
                    f"model {name!r}: alias {alias!r} duplicates the canonical model name"
                )
            if alias in seen_aliases:
                raise ValueError(f"model {name!r}: duplicate alias {alias!r}")
            seen_aliases.add(alias)
            aliases_list.append(alias)
        aliases = tuple(aliases_list)

        system_file = effective["sysf"]
        system_prompt: str | None = None
        if system_file is not None:
            if not isinstance(system_file, str) or not system_file.strip():
                raise ValueError(
                    f"model {name!r}: 'sysf' must be a non-empty string or null"
                )
            system_path = Path(system_file).expanduser()
            if not system_path.is_absolute():
                system_path = config_path.parent / system_path
            try:
                system_prompt = system_path.read_text(encoding="utf-8")
            except OSError as e:
                raise ValueError(
                    f"model {name!r}: cannot read sysf {str(system_path)!r}: {e}"
                ) from e
            if not system_prompt.strip():
                raise ValueError(f"model {name!r}: sysf {str(system_path)!r} is empty")

        think_value = effective["think"]
        if think_value is None:
            think_default: bool | None = None
        elif isinstance(think_value, str) and think_value in {"on", "off"}:
            think_default = think_value == "on"
        else:
            raise ValueError(
                f"model {name!r}: 'think' must be exactly 'on', 'off', or null"
            )
        if think_default is None and (
            "think" in read_only_config_keys or "think" in force_config_keys
        ):
            raise ValueError(
                f"model {name!r}: 'think (ro)'/'think (force)' requires an explicit on/off value"
            )

        reasoning_effort_value = effective["reasoning-effort"]
        if reasoning_effort_value is None:
            reasoning_effort_default: str | None = None
        elif not isinstance(reasoning_effort_value, str) or not reasoning_effort_value.strip():
            raise ValueError(
                f"model {name!r}: 'reasoning-effort' must be a non-empty string or null"
            )
        elif reasoning_effort_value != reasoning_effort_value.strip():
            raise ValueError(
                f"model {name!r}: 'reasoning-effort' must not have leading/trailing whitespace"
            )
        else:
            # Do not hard-code an enum here. Reasoning-effort levels are model/chat-template
            # specific (for example some templates use xhigh). Pass the configured string
            # through exactly after validation.
            reasoning_effort_default = reasoning_effort_value

        reasoning_preserve_value = effective["reasoning-preserve"]
        if reasoning_preserve_value is None:
            reasoning_preserve_default: bool | None = None
        elif isinstance(reasoning_preserve_value, bool):
            reasoning_preserve_default = reasoning_preserve_value
        elif isinstance(reasoning_preserve_value, str) and reasoning_preserve_value in {"on", "off"}:
            reasoning_preserve_default = reasoning_preserve_value == "on"
        else:
            raise ValueError(
                f"model {name!r}: 'reasoning-preserve' must be on, off, true, false, or null"
            )

        # Reject contradictory configured reasoning policy rather than relying on
        # request-time precedence between separate controls.
        if think_default is False and (
            reasoning_effort_default is not None
            and reasoning_effort_default.lower() != "none"
        ):
            raise ValueError(
                f"model {name!r}: think=off conflicts with reasoning-effort "
                f"{reasoning_effort_default!r}"
            )
        if think_default is True and (
            isinstance(reasoning_effort_default, str)
            and reasoning_effort_default.lower() == "none"
        ):
            raise ValueError(
                f"model {name!r}: think=on conflicts with reasoning-effort='none'"
            )

        defaults: dict[str, Any] = {}
        for config_key, option_key in GENERATION_CONFIG_TO_OPTION.items():
            value = effective[config_key]
            if value is None:
                if config_key in read_only_config_keys or config_key in force_config_keys:
                    raise ValueError(
                        f"model {name!r}: {config_key!r} with (ro)/(force) requires an explicit value"
                    )
                continue
            if config_key in {"top-k", "dry-allowed-length", "dry-penalty-last-n", "reasoning-budget"}:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f"model {name!r}: {config_key!r} must be an integer")
                minimum = -1 if config_key in {"reasoning-budget", "dry-penalty-last-n"} else 0
                if value < minimum:
                    raise ValueError(f"model {name!r}: {config_key!r} must be >= {minimum}")
                defaults[option_key] = value
                continue

            number = _require_number(name, config_key, value)
            if config_key in {"min-p", "top-p"} and not 0.0 <= number <= 1.0:
                raise ValueError(f"model {name!r}: {config_key!r} must be between 0 and 1")
            if config_key == "dry-base" and number < 1.0:
                raise ValueError(f"model {name!r}: 'dry-base' must be >= 1.0")
            if config_key in {"repeat-penalty", "temperature", "dry-multiplier"} and number < 0.0:
                raise ValueError(f"model {name!r}: {config_key!r} must be >= 0")
            defaults[option_key] = number

        models[name] = ModelConfig(
            name=name,
            aliases=aliases,
            priority=priority,
            cmd=cmd.strip(),
            external_chain=external_chain,
            system_prompt=system_prompt,
            generation_defaults=defaults,
            read_only_options=frozenset(
                GENERATION_CONFIG_TO_OPTION[key]
                for key in read_only_config_keys
                if key in GENERATION_CONFIG_TO_OPTION
            ),
            force_options=frozenset(
                GENERATION_CONFIG_TO_OPTION[key]
                for key in force_config_keys
                if key in GENERATION_CONFIG_TO_OPTION
            ),
            system_prompt_read_only="sysf" in read_only_config_keys,
            system_prompt_force="sysf" in force_config_keys,
            think_default=think_default,
            think_read_only="think" in read_only_config_keys,
            think_force="think" in force_config_keys,
            reasoning_effort_default=reasoning_effort_default,
            reasoning_effort_read_only="reasoning-effort" in read_only_config_keys,
            reasoning_effort_force="reasoning-effort" in force_config_keys,
            reasoning_preserve_default=reasoning_preserve_default,
            reasoning_preserve_read_only="reasoning-preserve" in read_only_config_keys,
            reasoning_preserve_force="reasoning-preserve" in force_config_keys,
        )

    alias_owners: dict[str, str] = {}
    for model_name, cfg in models.items():
        for alias in cfg.aliases:
            if alias in models:
                raise ValueError(
                    f"model {model_name!r}: alias {alias!r} conflicts with canonical "
                    f"model name {alias!r}"
                )
            previous = alias_owners.get(alias)
            if previous is not None:
                raise ValueError(
                    f"model {model_name!r}: alias {alias!r} is already used by "
                    f"model {previous!r}"
                )
            alias_owners[alias] = model_name

    return models

app = FastAPI(title="Ollama -> llama-server internal shim")
public_app = FastAPI(title="Ollama -> managed llama-server gateway")


async def _write_full_llama_response_bytes(data: bytes) -> None:
    """Append raw llama-server response-body bytes without altering them.

    This is deliberately below ``httpx.Response.aiter_lines()``: callers see
    exactly the same body bytes after this write, while SSE framing, UTF-8
    decoding, JSON parsing, checkpoint-sentinel handling, and Ollama translation
    all happen later. HTTP headers and HTTP/1 chunk framing are not part of this
    body log.
    """
    if not data:
        return
    fd = FULL_LLAMA_RESPONSES_LOG_FD
    if fd is None:
        return

    async with _FULL_LLAMA_RESPONSES_LOG_LOCK:
        view = memoryview(data)
        try:
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("os.write returned no progress")
                view = view[written:]
        except OSError as e:
            raise RuntimeError(
                f"failed to write raw llama response log "
                f"{str(FULL_LLAMA_RESPONSES_LOG)!r}: {e}"
            ) from e


class _FullLlamaResponseLogStream(httpx.AsyncByteStream):
    """Transparent tee for one llama-server response body."""

    def __init__(self, inner: httpx.AsyncByteStream) -> None:
        self._inner = inner

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._inner:
            # Log before yielding the chunk to httpx's line/content decoders.
            await _write_full_llama_response_bytes(chunk)
            yield chunk

    async def aclose(self) -> None:
        await self._inner.aclose()


def _enable_full_llama_response_logging(response: httpx.Response) -> httpx.Response:
    """Wrap a generation response before any body consumer sees it."""
    if FULL_LLAMA_RESPONSES_LOG_FD is not None:
        response.stream = _FullLlamaResponseLogStream(response.stream)
    return response


def _open_full_llama_responses_log(path: str | None) -> None:
    """Enable append-only raw generation-response logging for this process."""
    global FULL_LLAMA_RESPONSES_LOG, FULL_LLAMA_RESPONSES_LOG_FD

    if path is None:
        return
    if FULL_LLAMA_RESPONSES_LOG_FD is not None:
        raise RuntimeError("full llama response log is already open")

    log_path = Path(path).expanduser()
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(log_path, flags, 0o600)
    except OSError as e:
        raise RuntimeError(
            f"cannot open full llama response log {str(log_path)!r}: {e}"
        ) from e

    FULL_LLAMA_RESPONSES_LOG = log_path
    FULL_LLAMA_RESPONSES_LOG_FD = fd


def _close_full_llama_responses_log() -> None:
    global FULL_LLAMA_RESPONSES_LOG_FD
    fd = FULL_LLAMA_RESPONSES_LOG_FD
    FULL_LLAMA_RESPONSES_LOG_FD = None
    if fd is not None:
        with contextlib.suppress(OSError):
            os.close(fd)


# No read timeout: model generation can legitimately take a long time.
client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=60.0),
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
)


@dataclass
class ModelJob:
    id: int
    model_name: str
    priority: int
    sequence: int
    preempt_event: asyncio.Event
    cancel_event: asyncio.Event
    finished_event: asyncio.Event
    preemptible: bool = True
    checkpoint: bytes | None = None
    slot_id: int | None = None
    preempt_disabled_reason: str | None = None
    active: bool = False
    waiting: bool = True
    finished: bool = False


class ModelManager:
    """Own llama-server residency and checkpoint-aware runner preemption.

    Scheduling demand is deliberately user-bound: a model is active while at
    least one public request for it is still open. Internal requests emitted by
    external-chain programs do *not* make a model active or idle; they only tell
    us whether the resident runner is GENERATING (exact generation checkpoint)
    or BETWEEN_REQUESTS (idle prompt/KV snapshot only).
    """

    def __init__(self) -> None:
        self.models: dict[str, ModelConfig] = {}
        self.aliases: dict[str, str] = {}
        self.active_name: str | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.active_requests = 0
        self.switching = False
        self.condition = asyncio.Condition()
        self.metadata_cache: dict[str, dict[str, Any]] = {}
        self.jobs: dict[int, ModelJob] = {}
        self.active_jobs: dict[int, ModelJob] = {}
        self.active_users: dict[str, int] = {}
        # Opaque llama-server idle prompt/KV checkpoints, keyed by model/slot.
        # These are runner-level warm-cache state, not ModelJob continuation state.
        self.idle_checkpoints: dict[str, dict[int, bytes]] = {}
        self.model_total_slots: dict[str, int] = {}
        self._user_activation_sequence: dict[str, int] = {}
        self._next_user_sequence = 1
        self._next_job_id = 1
        self._next_sequence = 1
        self._scheduler_task: asyncio.Task[None] | None = None
        self._stopping = False

    def configure(self, models: dict[str, ModelConfig]) -> None:
        self.models = models
        self.aliases = {
            alias: name
            for name, cfg in models.items()
            for alias in cfg.aliases
        }
        self.active_users = {name: 0 for name in models}
        self.idle_checkpoints = {name: {} for name in models}
        self.model_total_slots = {}
        self._user_activation_sequence = {}

    def start_scheduler(self) -> None:
        if self._scheduler_task is None:
            self._scheduler_task = asyncio.create_task(
                self._scheduler_loop(), name="llama-model-scheduler"
            )

    def _resolve_exact_name(self, name: str) -> str | None:
        if name in self.models:
            return name
        return self.aliases.get(name)

    def resolve_name(self, requested: Any = None) -> str:
        if requested is None or requested == "":
            if self.active_name and self.active_name in self.models:
                return self.active_name
            if len(self.models) == 1:
                return next(iter(self.models))
            raise KeyError("model is required when multiple models are configured")

        name = str(requested)
        resolved = self._resolve_exact_name(name)
        if resolved is not None:
            return resolved

        if name.endswith(":latest"):
            resolved = self._resolve_exact_name(name[:-7])
            if resolved is not None:
                return resolved

        if ":" not in name:
            resolved = self._resolve_exact_name(name + ":latest")
            if resolved is not None:
                return resolved

        raise KeyError(f"model {name!r} not found")

    def running_name(self) -> str | None:
        if self.process is None or self.process.returncode is not None:
            return None
        return self.active_name

    def active_user_count(self, name: str) -> int:
        return self.active_users.get(name, 0)

    async def user_request_started(self, name: str) -> None:
        """Register public demand before the request enters external-chain."""
        async with self.condition:
            previous = self.active_users.get(name, 0)
            self.active_users[name] = previous + 1
            if previous == 0:
                self._user_activation_sequence[name] = self._next_user_sequence
                self._next_user_sequence += 1
                print(
                    f"Model {name!r} became active from user request "
                    f"(prio={self.models[name].priority})",
                    flush=True,
                )
            self.condition.notify_all()

    async def user_request_finished(self, name: str) -> None:
        """Drop public demand when the outer response ends or client disconnects."""
        async with self.condition:
            current = self.active_users.get(name, 0)
            if current <= 0:
                return
            current -= 1
            self.active_users[name] = current
            if current == 0:
                self._user_activation_sequence.pop(name, None)
                print(f"Model {name!r} became idle (no user requests)", flush=True)
            self.condition.notify_all()

    def _desired_model_locked(self) -> str | None:
        active = [
            name for name, count in self.active_users.items()
            if count > 0
        ]
        if not active:
            return None

        best_priority = min(self.models[name].priority for name in active)
        running = self.running_name()

        # Residency is sticky between internal requests. For equal priority,
        # keep the currently loaded active model instead of flapping runners.
        if (
            running in active
            and self.models[running].priority == best_priority
        ):
            return running

        candidates = [
            name for name in active
            if self.models[name].priority == best_priority
        ]
        return min(
            candidates,
            key=lambda name: self._user_activation_sequence.get(name, 2**63 - 1),
        )

    def _current_generation_jobs_locked(self) -> list[ModelJob]:
        running = self.running_name()
        if running is None:
            return []
        return [
            job
            for job in self.active_jobs.values()
            if job.model_name == running and not job.finished
        ]

    def _request_generation_preemption_locked(self, target_name: str) -> bool:
        """Ask all active generations to checkpoint before a residency switch.

        Returns False when at least one active generation is known to be
        uncheckpointable; in that case the scheduler must wait for it to finish.
        """
        running = self.running_name()
        current = self._current_generation_jobs_locked()
        if running is None or not current:
            return True
        if any(not job.preemptible for job in current):
            return False

        signalled = False
        for job in current:
            if not job.preempt_event.is_set():
                job.preempt_event.set()
                signalled = True
        if signalled:
            print(
                f"Preempting model {running!r} (prio={self.models[running].priority}) "
                f"for {target_name!r} (prio={self.models[target_name].priority}) "
                f"during generation; checkpoint required",
                flush=True,
            )
            self.condition.notify_all()
        return True

    def _generation_checkpoint_slots_locked(self, model_name: str) -> set[int]:
        """Slots whose exact in-flight state is already owned by a ModelJob."""
        return {
            job.slot_id
            for job in self.jobs.values()
            if job.model_name == model_name
            and not job.finished
            and job.checkpoint is not None
            and job.slot_id is not None
        }

    async def remember_idle_checkpoint(
        self, model_name: str, slot_id: int, blob: bytes
    ) -> None:
        """Remember a performance-only idle KV checkpoint in RAM."""
        async with self.condition:
            self.idle_checkpoints.setdefault(model_name, {})[slot_id] = blob
            self.condition.notify_all()

    async def _capture_idle_checkpoints(
        self, model_name: str, excluded_slots: set[int]
    ) -> None:
        """Snapshot reusable idle slot KV before unloading a resident model.

        Generation-checkpoint slots are excluded because their authoritative
        state lives in ModelJob.checkpoint. Failure here only loses warm-cache
        performance; it must not block a priority switch.
        """
        total_slots = self.model_total_slots.get(model_name, 0)
        if total_slots <= 0:
            return

        # Usually empty because restored blobs are consumed immediately. A
        # generation->idle race may already have produced a valid current-runner
        # idle blob, so retain it unless a fresh snapshot replaces it.
        captured: dict[int, bytes] = dict(
            self.idle_checkpoints.get(model_name, {})
        )
        for slot_id in range(total_slots):
            if slot_id in excluded_slots:
                continue
            try:
                response = await _request_checkpoint_suspend(slot_id)
            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                print(
                    f"WARNING: failed to save idle KV for model {model_name!r} "
                    f"slot {slot_id}: {e}",
                    flush=True,
                )
                continue

            if response.status_code >= 400:
                # Empty/fresh slots legitimately have nothing to preserve.
                continue

            kind = _checkpoint_response_kind(response)
            if kind != "idle":
                print(
                    f"WARNING: expected idle checkpoint for model {model_name!r} "
                    f"slot {slot_id}, got kind={kind!r}; discarding it",
                    flush=True,
                )
                continue

            blob = bytes(response.content)
            if not blob:
                print(
                    f"WARNING: llama-server returned an empty idle checkpoint for "
                    f"model {model_name!r} slot {slot_id}",
                    flush=True,
                )
                continue
            version = response.headers.get("X-Llama-Checkpoint-Version")
            if version not in (None, "1", "2"):
                print(
                    f"WARNING: unsupported idle checkpoint version {version!r} for "
                    f"model {model_name!r} slot {slot_id}",
                    flush=True,
                )
                continue

            response_slot = _slot_id_from_headers(response)
            if response_slot is not None and response_slot != slot_id:
                print(
                    f"WARNING: idle checkpoint slot mismatch for model {model_name!r}: "
                    f"requested {slot_id}, server reported {response_slot}",
                    flush=True,
                )
                continue

            captured[slot_id] = blob
            print(
                f"Saved idle KV for model {model_name!r} slot {slot_id}: "
                f"{len(blob)} checkpoint byte(s) held in RAM",
                flush=True,
            )

        async with self.condition:
            # A previous bundle is stale once this incarnation has run.
            self.idle_checkpoints[model_name] = captured

    async def _restore_idle_checkpoints(self, model_name: str) -> None:
        """Restore saved warm KV into fresh slots of a newly started model."""
        async with self.condition:
            bundle = dict(self.idle_checkpoints.get(model_name, {}))

        if not bundle:
            return

        for slot_id, blob in sorted(bundle.items()):
            try:
                response = await client.post(
                    f"{LLAMA_SERVER_URL}/v1/checkpoint/restore",
                    content=blob,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Accept": "application/json",
                    },
                    timeout=httpx.Timeout(
                        connect=10.0, read=60.0, write=60.0, pool=60.0
                    ),
                )
            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                print(
                    f"WARNING: failed to restore idle KV for model {model_name!r} "
                    f"slot {slot_id}: {e}; cache discarded",
                    flush=True,
                )
            else:
                if response.status_code >= 400:
                    try:
                        message = await _response_error_text(response)
                    except Exception:
                        message = response.text
                    print(
                        f"WARNING: idle KV restore for model {model_name!r} slot "
                        f"{slot_id} returned HTTP {response.status_code}: {message}; "
                        f"cache discarded",
                        flush=True,
                    )
                elif _checkpoint_response_kind(response) != "idle":
                    print(
                        f"WARNING: idle KV restore for model {model_name!r} slot "
                        f"{slot_id} returned unexpected checkpoint kind; cache discarded",
                        flush=True,
                    )
                else:
                    restored_slot = _slot_id_from_headers(response)
                    if restored_slot is not None and restored_slot != slot_id:
                        print(
                            f"WARNING: idle KV restore slot mismatch for model "
                            f"{model_name!r}: expected {slot_id}, got {restored_slot}",
                            flush=True,
                        )
                    else:
                        print(
                            f"Restored idle KV for model {model_name!r} slot {slot_id}: "
                            f"{len(blob)} checkpoint byte(s)",
                            flush=True,
                        )

            # Idle blobs describe one fresh-server restore opportunity. Never
            # retry them later against a slot that may already have been used.
            async with self.condition:
                self.idle_checkpoints.setdefault(model_name, {}).pop(slot_id, None)

    async def _scheduler_loop(self) -> None:
        while True:
            switch_to: ModelConfig | None = None
            old_name: str | None = None
            excluded_idle_slots: set[int] = set()
            async with self.condition:
                while True:
                    if self._stopping:
                        return
                    if self.switching:
                        await self.condition.wait()
                        continue

                    desired = self._desired_model_locked()
                    running = self.running_name()
                    if desired is None or desired == running:
                        await self.condition.wait()
                        continue

                    if running is not None:
                        generation_jobs = self._current_generation_jobs_locked()
                        if generation_jobs:
                            if not self._request_generation_preemption_locked(desired):
                                # An uncheckpointable generation remains authoritative.
                                await self.condition.wait()
                                continue
                            # The generation coroutine will export its checkpoint,
                            # pause its ModelJob, and wake us once no generation is
                            # using the runner.
                            await self.condition.wait()
                            continue

                        # Metadata probes may briefly hold the backend without being
                        # generation. They are short and are not a third preemption
                        # state: simply let them finish before the BETWEEN_REQUESTS
                        # fast-path switch.
                        if self.active_requests > 0:
                            await self.condition.wait()
                            continue

                        # Residency/activity is still determined solely by public
                        # user requests, but cache retention is independent from
                        # activity. Any reusable idle KV is worth preserving before
                        # unloading the runner, even when this model currently has no
                        # users. It can make a future request warm after the model is
                        # loaded again.
                        excluded_idle_slots = self._generation_checkpoint_slots_locked(running)
                        print(
                            f"Preempting model {running!r} for {desired!r} "
                            f"between internal requests; saving reusable idle KV",
                            flush=True,
                        )

                    self.switching = True
                    old_name = running
                    switch_to = self.models[desired]
                    break

            assert switch_to is not None
            try:
                if old_name is not None:
                    await self._capture_idle_checkpoints(
                        old_name, excluded_idle_slots
                    )
                await self._stop_current()
                proc = await self._start(switch_to)
                # Restore warm prompt/KV before publishing this runner as
                # resident, so waiting internal requests cannot race fresh slots.
                await self._restore_idle_checkpoints(switch_to.name)
            except asyncio.CancelledError:
                async with self.condition:
                    self.switching = False
                    self.condition.notify_all()
                raise
            except BaseException as e:
                print(
                    f"WARNING: failed to switch llama-server from {old_name!r} "
                    f"to {switch_to.name!r}: {e}",
                    flush=True,
                )
                async with self.condition:
                    self.switching = False
                    self.condition.notify_all()
                await asyncio.sleep(1.0)
                continue

            async with self.condition:
                self.active_name = switch_to.name
                self.process = proc
                self.switching = False
                self.condition.notify_all()

    def _activate_job_locked(self, job: ModelJob) -> None:
        job.waiting = False
        job.active = True
        job.preempt_event.clear()
        self.active_jobs[job.id] = job
        self.active_requests += 1

    async def _stop_current(self) -> None:
        proc = self.process
        old_name = self.active_name
        self.process = None
        self.active_name = None
        if proc is None or proc.returncode is not None:
            return

        print(f"Stopping model {old_name!r} (pid {proc.pid})", flush=True)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            print(f"Model {old_name!r} did not stop after SIGTERM; sending SIGKILL", flush=True)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()

    async def _start(self, cfg: ModelConfig) -> asyncio.subprocess.Process:
        # Refuse to start while the managed address is already serving
        # something. Otherwise a stale or unrelated process can satisfy the
        # readiness probe and be mistaken for the model we just launched.
        try:
            occupied = await asyncio.wait_for(
                client.get(f"{LLAMA_SERVER_URL}/health"), timeout=1.0
            )
        except (httpx.HTTPError, asyncio.TimeoutError):
            pass
        else:
            raise RuntimeError(
                f"cannot start model {cfg.name!r}: {LLAMA_SERVER_URL} is already "
                f"responding (HTTP {occupied.status_code})"
            )

        print(f"Starting model {cfg.name!r}: {cfg.cmd}", flush=True)
        proc = await asyncio.create_subprocess_shell(
            cfg.cmd,
            start_new_session=True,
        )

        # Give the shell/launcher a moment to fail before accepting a health
        # response that could belong to some unrelated process on port 8080.
        await asyncio.sleep(0.15)
        if proc.returncode is not None:
            raise RuntimeError(
                f"model {cfg.name!r} command exited with status {proc.returncode}: {cfg.cmd}"
            )

        deadline = asyncio.get_running_loop().time() + BACKEND_START_TIMEOUT
        last_error = "llama-server is not ready"
        while True:
            if proc.returncode is not None:
                raise RuntimeError(
                    f"model {cfg.name!r} command exited with status {proc.returncode}: {cfg.cmd}"
                )
            try:
                r = await asyncio.wait_for(
                    client.get(f"{LLAMA_SERVER_URL}/health"), timeout=2.0
                )
                if r.status_code < 400:
                    print(f"Model {cfg.name!r} is ready on {LLAMA_SERVER_URL}", flush=True)
                    return proc
                last_error = f"HTTP {r.status_code}: {r.text}"
            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                last_error = str(e) or type(e).__name__

            if asyncio.get_running_loop().time() >= deadline:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await proc.wait()
                raise RuntimeError(
                    f"model {cfg.name!r} did not become ready within "
                    f"{BACKEND_START_TIMEOUT:g}s: {last_error}"
                )
            await asyncio.sleep(0.25)

    async def _wait_for_job(self, job: ModelJob) -> None:
        while True:
            async with self.condition:
                while self.switching:
                    if job.cancel_event.is_set():
                        raise asyncio.CancelledError
                    await self.condition.wait()
                if job.finished or job.cancel_event.is_set():
                    raise asyncio.CancelledError

                # Internal chain traffic cannot make an idle model active. A
                # stale request emitted after the user disconnected therefore
                # waits until the chain cleanup closes it instead of reviving the
                # runner by itself.
                if (
                    self.active_users.get(job.model_name, 0) > 0
                    and self.running_name() == job.model_name
                    and self._desired_model_locked() == job.model_name
                ):
                    self._activate_job_locked(job)
                    return

                await self.condition.wait()

    async def acquire_job(
        self, requested: Any = None, *, preemptible: bool = True
    ) -> tuple[ModelJob, ModelConfig]:
        name = self.resolve_name(requested)
        cfg = self.models[name]
        async with self.condition:
            job = ModelJob(
                id=self._next_job_id,
                model_name=name,
                priority=cfg.priority,
                sequence=self._next_sequence,
                preempt_event=asyncio.Event(),
                cancel_event=asyncio.Event(),
                finished_event=asyncio.Event(),
                preemptible=preemptible,
            )
            self._next_job_id += 1
            self._next_sequence += 1
            self.jobs[job.id] = job
            self.condition.notify_all()

        try:
            await self._wait_for_job(job)
        except BaseException:
            await self.finish_job(job)
            raise
        return job, cfg

    async def claim_backend_slot(self, job: ModelJob, total_slots: int) -> int:
        """Reserve a stable llama-server slot id for a new generation.

        The reservation belongs to the logical job, not to a particular
        llama-server process. It is intentionally retained while a checkpoint
        is suspended so that restoring the original model can put the sequence
        back into the same slot id. Slot namespaces are per model process, so
        reservations for other models do not conflict.
        """
        total_slots = max(1, int(total_slots))
        async with self.condition:
            self.model_total_slots[job.model_name] = total_slots
        while True:
            if job.cancel_event.is_set():
                raise asyncio.CancelledError
            if job.preempt_event.is_set():
                await self.yield_if_preempted(job)
                continue

            async with self.condition:
                if job.finished or job.cancel_event.is_set():
                    raise asyncio.CancelledError
                if job.slot_id is not None:
                    return job.slot_id

                used = {
                    other.slot_id
                    for other in self.jobs.values()
                    if other.id != job.id
                    and other.model_name == job.model_name
                    and not other.finished
                    and other.slot_id is not None
                }
                for slot_id in range(total_slots):
                    if slot_id not in used:
                        job.slot_id = slot_id
                        self.condition.notify_all()
                        return slot_id

                await self.condition.wait()

    async def pause_job(self, job: ModelJob) -> None:
        async with self.condition:
            if job.finished:
                return
            if job.active:
                self.active_jobs.pop(job.id, None)
                job.active = False
                self.active_requests = max(0, self.active_requests - 1)
            job.waiting = True
            job.preempt_event.clear()
            self.condition.notify_all()

    async def resume_job(self, job: ModelJob) -> None:
        if job.finished:
            raise asyncio.CancelledError
        await self._wait_for_job(job)

    async def yield_if_preempted(self, job: ModelJob) -> None:
        if job.cancel_event.is_set():
            raise asyncio.CancelledError
        if not job.preempt_event.is_set():
            return
        await self.pause_job(job)
        await self.resume_job(job)
        if job.cancel_event.is_set():
            raise asyncio.CancelledError

    async def mark_unpreemptible(self, job: ModelJob, reason: str) -> None:
        async with self.condition:
            if job.finished:
                return
            job.preemptible = False
            job.preempt_disabled_reason = reason
            job.preempt_event.clear()
            print(
                f"Checkpoint preemption disabled for model {job.model_name!r}, "
                f"job {job.id}: {reason}",
                flush=True,
            )
            self.condition.notify_all()

    async def finish_job(self, job: ModelJob) -> None:
        async with self.condition:
            if job.finished:
                return
            job.finished = True
            job.waiting = False
            job.preempt_event.set()
            job.checkpoint = None
            job.slot_id = None
            if job.active:
                self.active_jobs.pop(job.id, None)
                job.active = False
                self.active_requests = max(0, self.active_requests - 1)
            self.jobs.pop(job.id, None)
            job.finished_event.set()
            self.condition.notify_all()

    async def cancel_job(self, job_id: int, *, timeout: float = 35.0) -> bool:
        """Cancel one internal generation and wait until its slot/job is released.

        Returns True when the job existed. A missing job is treated as already
        finished, which makes the cancellation endpoint race-safe.
        """
        async with self.condition:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            job.cancel_event.set()
            self.condition.notify_all()
            finished_event = job.finished_event

        try:
            await asyncio.wait_for(finished_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"job {job_id} did not finish within {timeout:g}s after cancellation"
            ) from e
        return True

    # Kept for short, non-disruptive metadata probes such as /api/ps and
    # /api/show. Generation requests use acquire_job() so they can be preempted.
    async def acquire_running(self) -> tuple[str, ModelConfig] | None:
        async with self.condition:
            if self.switching:
                return None
            name = self.running_name()
            if name is None:
                return None
            self.active_requests += 1
            return name, self.models[name]

    async def release(self, name: str) -> None:
        async with self.condition:
            if self.active_requests <= 0:
                return
            self.active_requests -= 1
            self.condition.notify_all()

    async def shutdown(self) -> None:
        self._stopping = True
        async with self.condition:
            self.condition.notify_all()
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
            self._scheduler_task = None
        await self._stop_current()
        self.idle_checkpoints.clear()


MODEL_MANAGER = ModelManager()


CHAIN_HOST = "127.0.0.1"
CHAIN_STALE_TIMEOUT = 3.0
CHAIN_START_TIMEOUT = 10.0
EXTERNAL_CHAIN_LOG_DIR = Path("/var/log/oko-ai-external-chain")
INTERNAL_PORT_MODELS: dict[int, str] = {}


def external_chain_proxy_name(command: str) -> str:
    """Derive a safe, readable log basename from an external-chain command."""
    try:
        argv = shlex.split(command)
    except ValueError:
        argv = []

    index = 0
    while index < len(argv) and re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*=.*", argv[index]
    ):
        index += 1

    if index < len(argv) and Path(argv[index]).name == "env":
        index += 1
        while index < len(argv):
            token = argv[index]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token):
                index += 1
                continue
            if token.startswith("-"):
                index += 1
                continue
            break

    candidate = argv[index] if index < len(argv) else ""
    executable = Path(candidate).name
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable) or executable in {
        "pypy",
        "pypy3",
        "bash",
        "sh",
    }:
        index += 1
        while index < len(argv) and argv[index].startswith("-"):
            if argv[index] == "-m" and index + 1 < len(argv):
                index += 1
                break
            index += 1
        if index < len(argv):
            candidate = argv[index]

    raw_name = Path(candidate).name
    for suffix in (".py", ".sh"):
        if raw_name.endswith(suffix):
            raw_name = raw_name[: -len(suffix)]
            break

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._-")
    if not safe_name:
        safe_name = f"proxy-{hashlib.sha256(command.encode()).hexdigest()[:12]}"
    return safe_name[:128]


@dataclass
class ModelChainRuntime:
    model_name: str
    listen_ports: tuple[int, ...]
    ingress_port: int
    processes: list[asyncio.subprocess.Process | None]
    monitor_tasks: list[asyncio.Task[None]]
    ingress_server: Any = None
    ingress_task: asyncio.Task[None] | None = None
    restart_lock: asyncio.Lock | None = None
    cleanup_task: asyncio.Task[None] | None = None
    internal_requests: int = 0
    generation: int = 0
    stopping: bool = False

    @property
    def entry_port(self) -> int:
        return self.listen_ports[0] if self.listen_ports else self.ingress_port


class InternalIngressApp:
    """Bind one local ingress listener to exactly one configured model."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.inner(scope, receive, send)
            return

        server = scope.get("server")
        port = int(server[1]) if server and len(server) > 1 else -1
        model_name = INTERNAL_PORT_MODELS.get(port)
        if model_name is None:
            _stderr("ERROR", f"unknown internal ingress on port {port}")
            response = PlainTextResponse("unknown internal ingress\n", status_code=404)
            await response(scope, receive, send)
            return

        state = scope.setdefault("state", {})
        state["bound_model"] = model_name

        track = scope.get("path") in {"/api/chat", "/api/generate"}
        if track:
            await CHAIN_MANAGER.internal_request_started(model_name)
        try:
            await self.inner(scope, receive, send)
        finally:
            if track:
                await CHAIN_MANAGER.internal_request_finished(model_name)


INTERNAL_APP = InternalIngressApp(app)


class ChainManager:
    """Persistent per-model external-chain processes and local ingress ports."""

    def __init__(self) -> None:
        self.runtimes: dict[str, ModelChainRuntime] = {}
        self._stopping = False

    @staticmethod
    def _allocate_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((CHAIN_HOST, 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _ensure_log_directory() -> None:
        try:
            EXTERNAL_CHAIN_LOG_DIR.mkdir(
                mode=0o755,
                parents=True,
                exist_ok=True,
            )
        except OSError as e:
            raise RuntimeError(
                f"cannot create external-chain log directory "
                f"{str(EXTERNAL_CHAIN_LOG_DIR)!r}: {e}"
            ) from e
        if not EXTERNAL_CHAIN_LOG_DIR.is_dir():
            raise RuntimeError(
                f"external-chain log path {str(EXTERNAL_CHAIN_LOG_DIR)!r} "
                "is not a directory"
            )

    @staticmethod
    def _log_path(runtime: ModelChainRuntime, index: int) -> Path:
        cfg = MODEL_MANAGER.models[runtime.model_name]
        proxy_name = external_chain_proxy_name(cfg.external_chain[index])
        return EXTERNAL_CHAIN_LOG_DIR / f"{proxy_name}.log"

    @staticmethod
    def _append_log_marker(log_path: Path, marker: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        data = f"\n[{timestamp}] {marker}\n".encode("utf-8", errors="replace")
        try:
            with log_path.open("ab", buffering=0) as log_file:
                log_file.write(data)
        except OSError as e:
            print(
                f"WARNING: cannot append external-chain log marker to "
                f"{str(log_path)!r}: {e}",
                flush=True,
            )

    async def _wait_port(self, port: int, *, timeout: float = CHAIN_START_TIMEOUT) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        last_error: BaseException | None = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                reader, writer = await asyncio.open_connection(CHAIN_HOST, port)
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
                return
            except OSError as e:
                last_error = e
                await asyncio.sleep(0.05)
        raise RuntimeError(
            f"listener {CHAIN_HOST}:{port} did not become ready within {timeout:g}s: "
            f"{last_error or 'connection refused'}"
        )

    async def start(self) -> None:
        import uvicorn

        self._stopping = False
        if any(cfg.external_chain for cfg in MODEL_MANAGER.models.values()):
            self._ensure_log_directory()
        for model_name, cfg in MODEL_MANAGER.models.items():
            listen_ports = tuple(self._allocate_port() for _ in cfg.external_chain)
            ingress_port = self._allocate_port()
            runtime = ModelChainRuntime(
                model_name=model_name,
                listen_ports=listen_ports,
                ingress_port=ingress_port,
                processes=[],
                monitor_tasks=[],
                restart_lock=asyncio.Lock(),
            )
            self.runtimes[model_name] = runtime
            INTERNAL_PORT_MODELS[ingress_port] = model_name

            config = uvicorn.Config(
                INTERNAL_APP,
                host=CHAIN_HOST,
                port=ingress_port,
                log_level="warning",
                access_log=False,
                lifespan="off",
            )
            runtime.ingress_server = uvicorn.Server(config)
            runtime.ingress_task = asyncio.create_task(
                runtime.ingress_server.serve(),
                name=f"internal-ingress-{model_name}",
            )
            await self._wait_port(ingress_port)
            await self._start_programs(runtime)

            if cfg.external_chain:
                path = " -> ".join(
                    [f"{CHAIN_HOST}:{port}" for port in runtime.listen_ports]
                    + [f"{CHAIN_HOST}:{runtime.ingress_port}"]
                )
                print(f"External chain {model_name!r}: {path}", flush=True)
            else:
                print(
                    f"Model {model_name!r}: no external-chain; internal ingress "
                    f"{CHAIN_HOST}:{runtime.ingress_port}",
                    flush=True,
                )

    def entry_url(self, model_name: str) -> str:
        runtime = self.runtimes[model_name]
        return f"http://{CHAIN_HOST}:{runtime.entry_port}"

    def _runtime_command(self, runtime: ModelChainRuntime, index: int) -> str:
        cfg = MODEL_MANAGER.models[runtime.model_name]
        listen_port = runtime.listen_ports[index]
        target_port = (
            runtime.listen_ports[index + 1]
            if index + 1 < len(runtime.listen_ports)
            else runtime.ingress_port
        )
        return (
            f"{cfg.external_chain[index]} "
            f"--listen-host {shlex.quote(CHAIN_HOST)} "
            f"--listen-port {listen_port} "
            f"--target-host {shlex.quote(CHAIN_HOST)} "
            f"--target-port {target_port}"
        )

    async def _start_programs(self, runtime: ModelChainRuntime) -> None:
        cfg = MODEL_MANAGER.models[runtime.model_name]
        runtime.generation += 1
        generation = runtime.generation
        runtime.processes = [None] * len(cfg.external_chain)
        runtime.monitor_tasks = []

        # Start downstream first so every program sees a live target when it
        # accepts its first request.
        for index in reversed(range(len(cfg.external_chain))):
            command = self._runtime_command(runtime, index)
            log_path = self._log_path(runtime, index)
            print(
                f"Starting external-chain {runtime.model_name!r}[{index}]: {command} "
                f"(log: {log_path})",
                flush=True,
            )
            try:
                log_file = log_path.open("ab", buffering=0)
            except OSError as e:
                raise RuntimeError(
                    f"cannot open external-chain log {str(log_path)!r}: {e}"
                ) from e

            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            marker = (
                f"\n[{timestamp}] START model={runtime.model_name!r} "
                f"chain_index={index} generation={generation}\n"
                f"command: {command}\n"
            ).encode("utf-8", errors="replace")
            try:
                log_file.write(marker)
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=log_file,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            except BaseException as e:
                with contextlib.suppress(OSError):
                    failed_at = datetime.now(timezone.utc).isoformat().replace(
                        "+00:00", "Z"
                    )
                    log_file.write(
                        f"[{failed_at}] SPAWN FAILED: {e}\n".encode(
                            "utf-8", errors="replace"
                        )
                    )
                raise
            finally:
                log_file.close()
            runtime.processes[index] = proc
            await asyncio.sleep(0.05)
            if proc.returncode is not None:
                raise RuntimeError(
                    f"external-chain {runtime.model_name!r}[{index}] exited with "
                    f"status {proc.returncode}: {command}"
                )
            await self._wait_port(runtime.listen_ports[index])
            runtime.monitor_tasks.append(
                asyncio.create_task(
                    self._watch_process(runtime, index, proc, generation),
                    name=f"external-chain-watch-{runtime.model_name}-{index}",
                )
            )

    async def _watch_process(
        self,
        runtime: ModelChainRuntime,
        index: int,
        proc: asyncio.subprocess.Process,
        generation: int,
    ) -> None:
        rc = await proc.wait()
        self._append_log_marker(
            self._log_path(runtime, index),
            f"EXIT model={runtime.model_name!r} chain_index={index} "
            f"generation={generation} pid={proc.pid} status={rc}",
        )
        if runtime.stopping or self._stopping or generation != runtime.generation:
            return
        print(
            f"WARNING: external-chain {runtime.model_name!r}[{index}] crashed "
            f"with status {rc}; restarting the complete chain",
            flush=True,
        )
        asyncio.create_task(
            self.restart_programs(runtime.model_name, reason="chain process crashed")
        )

    async def _terminate_process(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            print(
                f"WARNING: external-chain process pid {proc.pid} did not stop within "
                f"3 seconds; sending SIGKILL",
                flush=True,
            )
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()

    async def _stop_programs(self, runtime: ModelChainRuntime) -> None:
        runtime.generation += 1
        tasks = list(runtime.monitor_tasks)
        runtime.monitor_tasks = []
        processes = [p for p in runtime.processes if p is not None]
        runtime.processes = []
        for proc in processes:
            await self._terminate_process(proc)
        for task in tasks:
            if task is asyncio.current_task():
                continue
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def restart_programs(self, model_name: str, *, reason: str) -> None:
        runtime = self.runtimes.get(model_name)
        if runtime is None or runtime.stopping or self._stopping:
            return
        assert runtime.restart_lock is not None
        async with runtime.restart_lock:
            if runtime.stopping or self._stopping:
                return
            print(
                f"WARNING: restarting external-chain for model {model_name!r}: {reason}",
                flush=True,
            )
            await self._stop_programs(runtime)
            while not runtime.stopping and not self._stopping:
                try:
                    await self._start_programs(runtime)
                    return
                except Exception as e:
                    print(
                        f"WARNING: failed to restart external-chain for model {model_name!r}: "
                        f"{e}; retrying in 1 second",
                        flush=True,
                    )
                    await self._stop_programs(runtime)
                    await asyncio.sleep(1.0)

    async def user_request_started(self, model_name: str) -> None:
        runtime = self.runtimes[model_name]
        if runtime.cleanup_task is not None:
            runtime.cleanup_task.cancel()
            runtime.cleanup_task = None
        await MODEL_MANAGER.user_request_started(model_name)

    async def user_request_finished(self, model_name: str) -> None:
        await MODEL_MANAGER.user_request_finished(model_name)
        runtime = self.runtimes[model_name]
        if MODEL_MANAGER.active_user_count(model_name) != 0:
            return
        if runtime.cleanup_task is not None:
            runtime.cleanup_task.cancel()
        runtime.cleanup_task = asyncio.create_task(
            self._cleanup_stale_chain(runtime),
            name=f"external-chain-cleanup-{model_name}",
        )

    async def _cleanup_stale_chain(self, runtime: ModelChainRuntime) -> None:
        try:
            await asyncio.sleep(CHAIN_STALE_TIMEOUT)
            if runtime.stopping or self._stopping:
                return
            if MODEL_MANAGER.active_user_count(runtime.model_name) != 0:
                return
            if runtime.internal_requests <= 0:
                return
            if not MODEL_MANAGER.models[runtime.model_name].external_chain:
                return
            print(
                f"WARNING: model {runtime.model_name!r} has no user connection but "
                f"external-chain still holds {runtime.internal_requests} internal "
                f"request(s) after {CHAIN_STALE_TIMEOUT:g}s; forcing chain restart",
                flush=True,
            )
            await self.restart_programs(
                runtime.model_name,
                reason=f"stale internal requests after user disconnect ({CHAIN_STALE_TIMEOUT:g}s)",
            )
        except asyncio.CancelledError:
            raise
        finally:
            if runtime.cleanup_task is asyncio.current_task():
                runtime.cleanup_task = None

    async def internal_request_started(self, model_name: str) -> None:
        runtime = self.runtimes.get(model_name)
        if runtime is not None:
            runtime.internal_requests += 1

    async def internal_request_finished(self, model_name: str) -> None:
        runtime = self.runtimes.get(model_name)
        if runtime is not None:
            runtime.internal_requests = max(0, runtime.internal_requests - 1)

    async def shutdown(self) -> None:
        self._stopping = True
        for runtime in self.runtimes.values():
            runtime.stopping = True
            if runtime.cleanup_task is not None:
                runtime.cleanup_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await runtime.cleanup_task
                runtime.cleanup_task = None
            await self._stop_programs(runtime)

        for runtime in self.runtimes.values():
            if runtime.ingress_server is not None:
                runtime.ingress_server.should_exit = True
        for runtime in self.runtimes.values():
            if runtime.ingress_task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await runtime.ingress_task
            INTERNAL_PORT_MODELS.pop(runtime.ingress_port, None)
        self.runtimes.clear()


CHAIN_MANAGER = ChainManager()

# Separate client for the public -> external-chain hop. It deliberately has no
# read timeout because the outer Ollama stream may remain open through model
# checkpoint/preemption and through several internal requests made by filters.
gateway_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=60.0),
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
)


@public_app.on_event("startup")
async def _startup() -> None:
    MODEL_MANAGER.start_scheduler()
    await CHAIN_MANAGER.start()


@public_app.on_event("shutdown")
async def _shutdown() -> None:
    await CHAIN_MANAGER.shutdown()
    await MODEL_MANAGER.shutdown()
    await gateway_client.aclose()
    await client.aclose()
    _close_full_llama_responses_log()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def iso_from_epoch(epoch: int | float | None) -> str:
    if epoch is None:
        return iso_now()
    if (
        not isinstance(epoch, (int, float))
        or isinstance(epoch, bool)
        or not math.isfinite(float(epoch))
    ):
        raise RuntimeError(f"timestamp was not a finite number: {epoch!r}")
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def ndjson(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _ndjson_error(message: str) -> bytes:
    _stderr("ERROR", message)
    return ndjson({"error": message})


def human_params(n: int | None) -> str:
    if not n:
        return ""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def model_digest(
    model: dict[str, Any] | None = None,
    fallback_identity: str | None = None,
) -> str:
    """Return an Ollama-CLI-safe, stable model digest.

    Prefer a real backend-supplied digest when one is available. llama-server
    commonly exposes no Ollama-style manifest digest at all, but Ollama CLI
    currently assumes every /api/tags digest is at least 12 characters long
    (``digest[:12]``) and can panic on an empty/short value. In that case use a
    deterministic SHA-256 compatibility identifier derived from the canonical
    configured model identity. This value is metadata-only; it is never sent to
    llama-server and does not affect inference semantics.
    """
    if isinstance(model, dict):
        meta = model.get("meta")
        candidates = [model.get("digest"), model.get("sha256")]
        if isinstance(meta, dict):
            candidates.extend((meta.get("digest"), meta.get("sha256")))
        for value in candidates:
            if isinstance(value, str):
                digest = value.strip()
                if len(digest) >= 12:
                    return digest

    if isinstance(fallback_identity, str) and fallback_identity:
        return hashlib.sha256(fallback_identity.encode("utf-8")).hexdigest()

    # Internal callers should always provide the canonical model name as the
    # fallback. Keep this fail-closed rather than returning another short value.
    raise RuntimeError("model digest unavailable and no fallback identity was provided")


def model_modified_at(
    model: dict[str, Any] | None = None, props: dict[str, Any] | None = None
) -> str | None:
    """Return only an actual model modification timestamp when known.

    `/v1/models.created` and the shim config mtime are not the model's
    `modified_at`, so neither is substituted when model-file metadata is absent.
    """
    props = props or {}
    model_path = props.get("model_path")
    if isinstance(model_path, str) and model_path:
        try:
            return iso_from_epoch(Path(model_path).stat().st_mtime)
        except OSError:
            pass
    if isinstance(model, dict):
        value = model.get("modified_at")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def backend_model() -> dict[str, Any]:
    try:
        r = await client.get(f"{LLAMA_SERVER_URL}/v1/models")
        r.raise_for_status()
        root = r.json()
        if not isinstance(root, dict):
            raise RuntimeError("llama-server /v1/models returned a non-object response")
        data = root.get("data")
        if not isinstance(data, list):
            raise RuntimeError("llama-server /v1/models.data was not an array")
        if not data:
            raise RuntimeError("llama-server returned no loaded models")
        if len(data) != 1:
            raise RuntimeError(
                f"managed llama-server returned {len(data)} loaded models; expected exactly one"
            )
        if not isinstance(data[0], dict):
            raise RuntimeError("llama-server /v1/models entry was not an object")
        return data[0]
    except (httpx.HTTPError, ValueError, RuntimeError) as e:
        raise RuntimeError(f"cannot query llama-server at {LLAMA_SERVER_URL}: {e}") from e


def backend_model_id(model: dict[str, Any], fallback: str) -> str:
    value = model.get("id")
    if value is None or value == "":
        return fallback
    if not isinstance(value, str):
        raise RuntimeError(
            f"llama-server /v1/models returned a non-string model id: {value!r}"
        )
    return value


async def backend_health() -> None:
    try:
        r = await client.get(f"{LLAMA_SERVER_URL}/health")
        if r.status_code >= 400:
            raise RuntimeError(
                f"llama-server health check failed: {r.status_code} {r.text}"
            )
    except httpx.HTTPError as e:
        raise RuntimeError(
            f"cannot query llama-server health at {LLAMA_SERVER_URL}: {e}"
        ) from e


def infer_model_family(
    m: dict[str, Any], model_name: str, props: dict[str, Any] | None = None
) -> str:
    """Return only architecture metadata actually reported by the backend."""
    meta = m.get("meta") or {}
    if not isinstance(meta, dict):
        return "unknown"
    for key in ("architecture", "general.architecture", "family"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def infer_quantization(
    m: dict[str, Any], props: dict[str, Any] | None = None
) -> str:
    """Return only quantization metadata actually reported by the backend."""
    meta = m.get("meta") or {}
    if not isinstance(meta, dict):
        return "unknown"
    for key in ("quantization", "quantization_level", "general.file_type_name"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return ""


async def backend_props() -> dict[str, Any]:
    """Read llama-server runtime/model properties. GET /props is read-only by default."""
    try:
        r = await client.get(f"{LLAMA_SERVER_URL}/props")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else {}
    except (httpx.HTTPError, ValueError):
        # /props is richer metadata, but /api/show should still work against
        # older llama-server builds that don't expose it.
        return {}


def cache_backend_metadata(
    name: str, model: dict[str, Any], props: dict[str, Any] | None = None
) -> None:
    """Update metadata without erasing previously fetched runtime properties."""
    cached = MODEL_MANAGER.metadata_cache.get(name) or {}
    MODEL_MANAGER.metadata_cache[name] = {
        "model": model or cached.get("model") or {},
        "props": props if props is not None else (cached.get("props") or {}),
    }


def model_details(
    m: dict[str, Any], model_name: str, props: dict[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    props = props or {}
    meta = m.get("meta") or {}
    params = int(meta.get("n_params") or 0)
    size = int(meta.get("size") or 0)
    n_ctx_train = int(meta.get("n_ctx_train") or 0)
    n_embd = int(meta.get("n_embd") or 0)
    n_vocab = int(meta.get("n_vocab") or 0)
    vocab_type = meta.get("vocab_type")
    family = infer_model_family(m, model_name, props)
    quantization = infer_quantization(m, props)

    details: dict[str, Any] = {
        "parent_model": "",
        "format": "gguf",
        "family": family,
        "families": [] if not family else [family],
        "parameter_size": human_params(params),
        "quantization_level": quantization,
    }
    if n_ctx_train:
        details["context_length"] = n_ctx_train
    if n_embd:
        details["embedding_length"] = n_embd

    info: dict[str, Any] = {}
    if family:
        info["general.architecture"] = family
    reported_name = meta.get("general.name") or meta.get("name")
    if isinstance(reported_name, str) and reported_name.strip():
        info["general.name"] = reported_name.strip()
    if params:
        info["general.parameter_count"] = params
    if size:
        info["general.file_size"] = size
    if family:
        if n_ctx_train:
            info[f"{family}.context_length"] = n_ctx_train
        if n_embd:
            info[f"{family}.embedding_length"] = n_embd
        if n_vocab:
            info[f"{family}.vocab_size"] = n_vocab
    if vocab_type is not None:
        info["tokenizer.ggml.vocab_type"] = vocab_type

    generation = props.get("default_generation_settings") or {}
    runtime_ctx = generation.get("n_ctx")
    if isinstance(runtime_ctx, int) and runtime_ctx > 0:
        info["llama-server.context_length"] = runtime_ctx
    if props.get("model_path"):
        info["llama-server.model_path"] = str(props["model_path"])
    if props.get("build_info"):
        info["llama-server.build_info"] = str(props["build_info"])
    if props.get("total_slots") is not None:
        info["llama-server.total_slots"] = props["total_slots"]
    if props.get("is_sleeping") is not None:
        info["llama-server.is_sleeping"] = props["is_sleeping"]

    return details, info


def capabilities(props: dict[str, Any] | None = None) -> list[str]:
    caps = ["completion"]
    props = props or {}
    modalities = props.get("modalities") or {}
    if isinstance(modalities, dict) and modalities.get("vision"):
        caps.append("vision")
    if ADVERTISE_TOOLS:
        caps.append("tools")
    if ADVERTISE_THINKING:
        caps.append("thinking")
    return caps


def ollama_parameters_from_props(
    props: dict[str, Any], generation_defaults: dict[str, Any]
) -> list[tuple[str, Any]]:
    """Translate llama-server generation defaults to Ollama/Modelfile names."""
    generation = props.get("default_generation_settings") or {}
    params = generation.get("params") or {}
    result: list[tuple[str, Any]] = []

    n_ctx = generation.get("n_ctx")
    if isinstance(n_ctx, int) and n_ctx > 0:
        result.append(("num_ctx", n_ctx))

    mapping = (
        ("n_predict", "num_predict"),
        ("seed", "seed"),
        ("temperature", "temperature"),
        ("top_k", "top_k"),
        ("top_p", "top_p"),
        ("min_p", "min_p"),
        ("typical_p", "typical_p"),
        ("repeat_last_n", "repeat_last_n"),
        ("repeat_penalty", "repeat_penalty"),
        ("presence_penalty", "presence_penalty"),
        ("frequency_penalty", "frequency_penalty"),
        ("dry_multiplier", "dry_multiplier"),
        ("dry_base", "dry_base"),
        ("dry_allowed_length", "dry_allowed_length"),
        ("dry_penalty_last_n", "dry_penalty_last_n"),
        ("thinking_budget_tokens", "reasoning_budget"),
        ("mirostat", "mirostat"),
        ("mirostat_tau", "mirostat_tau"),
        ("mirostat_eta", "mirostat_eta"),
    )
    for llama_key, ollama_key in mapping:
        if llama_key in params:
            result.append((ollama_key, params[llama_key]))

    stop = params.get("stop")
    if isinstance(stop, list):
        result.extend(("stop", item) for item in stop)
    elif stop not in (None, ""):
        result.append(("stop", stop))

    # /api/show should describe the defaults this proxy actually applies,
    # not only llama-server's own startup defaults.
    effective_keys = {
        "presence_penalty",
        "repeat_penalty",
        "min_p",
        "top_k",
        "top_p",
        "temperature",
        "dry_multiplier",
        "dry_base",
        "dry_allowed_length",
        "dry_penalty_last_n",
        "reasoning_budget",
    }
    configured_keys = effective_keys.intersection(generation_defaults)
    result = [(k, v) for k, v in result if k not in configured_keys]
    for key in (
        "presence_penalty",
        "repeat_penalty",
        "min_p",
        "top_k",
        "top_p",
        "temperature",
        "dry_multiplier",
        "dry_base",
        "dry_allowed_length",
        "dry_penalty_last_n",
        "reasoning_budget",
    ):
        if key in generation_defaults:
            result.append((key, generation_defaults[key]))
    return result


def format_show_parameters(items: list[tuple[str, Any]]) -> str:
    lines: list[str] = []
    for key, value in items:
        encoded = json.dumps(value, ensure_ascii=False) if isinstance(value, (str, list, dict, bool)) else str(value)
        lines.append(f"{key:<30} {encoded}")
    return "\n".join(lines)


def _requested_think_enabled(value: Any) -> bool | None:
    """Normalize an Ollama think value to enabled/disabled without closing its level enum."""
    if value is False:
        return False
    if value is True:
        return True
    if isinstance(value, str):
        level = value.strip()
        if not level:
            return None
        if level.lower() == "off":
            return False
        # Current Ollama exposes model-defined string levels. Preserve the exact
        # level elsewhere; every non-empty non-"off" string enables thinking.
        return True
    return None


def _think_matches_config(value: Any, configured: bool) -> bool:
    """Whether a client think value repeats an on/off config without adding policy.

    A level such as ``high`` is not equivalent to configured ``on`` because it
    additionally selects a reasoning effort. Booleans and the literal on/off
    strings are equivalent representations of the same binary control.
    """
    if configured:
        return value is True or (isinstance(value, str) and value.strip().lower() == "on")
    return value is False or (isinstance(value, str) and value.strip().lower() == "off")


def _explicit_reasoning_effort(body: dict[str, Any]) -> str | None:
    """Return a client-supplied reasoning effort, including legacy think levels."""
    if "reasoning_effort" in body and body.get("reasoning_effort") is not None:
        value = body["reasoning_effort"]
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    think = body.get("think")
    if isinstance(think, str):
        value = think.strip()
        if value.lower() not in {"on", "off"} and value:
            return value
    return None


def _requested_reasoning_preserve(value: Any) -> bool | None:
    """Normalize the proxy's optional per-request reasoning preservation control."""
    if value is True or value is False:
        return value
    if isinstance(value, str):
        if value == "on":
            return True
        if value == "off":
            return False
    return None


# Request fields supported by the current Ollama API, plus the two local
# reasoning extensions exposed by this shim. Fields whose semantics cannot be
# reproduced by a fixed, managed llama-server are rejected explicitly below;
# they are listed here so the error can say *why* rather than looking unknown.
_CHAT_REQUEST_FIELDS = {
    "model", "messages", "stream", "format", "keep_alive", "tools", "options",
    "think", "truncate", "shift", "_debug_render_only", "logprobs",
    "top_logprobs", "reasoning_effort", "reasoning_preserve",
}
_GENERATE_REQUEST_FIELDS = {
    "model", "prompt", "suffix", "system", "template", "context", "stream",
    "raw", "format", "keep_alive", "images", "options", "think", "truncate",
    "shift", "_debug_render_only", "logprobs", "top_logprobs",
    "reasoning_effort", "reasoning_preserve",
}
_OLLAMA_RUNNER_OPTIONS = {
    "num_ctx", "num_batch", "num_gpu", "main_gpu", "use_mmap",
    "num_thread", "draft_num_predict",
}
_TRANSLATABLE_OLLAMA_OPTIONS = (
    set(OLLAMA_OPTION_TO_LLAMA) | {"num_predict", "stop"} | _OLLAMA_RUNNER_OPTIONS
)


def _stderr(level: str, message: str) -> None:
    print(f"{level}: {message}", file=sys.stderr, flush=True)


def _proxy_error_response(message: str, status_code: int = 400) -> JSONResponse:
    _stderr("ERROR", message)
    return JSONResponse({"error": message}, status_code=status_code)


def _compat_warning(message: str) -> None:
    _stderr("WARNING", message)


def _warn_non_native_controls(body: dict[str, Any]) -> None:
    """Report accepted Ollama controls that the managed runner cannot enact per-request.

    These controls are deliberately accepted because rejecting an explicitly supplied
    Ollama default (for example ``truncate: true``) makes the shim less compatible
    than simply omitting the field.  They are never silently ignored: any semantic
    limitation is visible on stderr.
    """
    keep_alive = body.get("keep_alive")
    if keep_alive is not None:
        _compat_warning(
            f"keep_alive={keep_alive!r} accepted as an Ollama compatibility hint; "
            "model residency is controlled by the managed scheduler"
        )

    # Ollama defaults truncate=true. llama-server currently errors when an already
    # rendered prompt itself is larger than the available context, so the common
    # explicit true value must not be rejected merely for spelling out the default.
    if body.get("truncate") is True:
        _compat_warning(
            "truncate=true accepted; normal requests are forwarded unchanged, but "
            "llama-server may still reject an initially over-context rendered prompt "
            "instead of trimming chat history exactly like Ollama"
        )

    # Context shifting is a llama-server startup policy rather than a request field.
    # Accept the Ollama control to avoid breaking clients, while making the limitation
    # explicit.  If the startup policy already matches, this warning is harmless.
    if body.get("shift") is not None:
        _compat_warning(
            f"shift={body.get('shift')!r} accepted; per-request context-shift toggling "
            "is not exposed by llama-server, so the server startup policy remains authoritative"
        )

    raw_or_infill = bool(body.get("raw")) or body.get("suffix") not in (None, "")
    if raw_or_infill:
        explicit_reasoning = [
            key for key in ("think", "reasoning_effort", "reasoning_preserve")
            if key in body and body.get(key) is not None
        ]
        if explicit_reasoning:
            _compat_warning(
                "raw/FIM generation bypasses chat-template reasoning controls; accepted field(s) "
                "do not alter the raw prompt path: " + ", ".join(explicit_reasoning)
            )

    opts = body.get("options")
    if isinstance(opts, dict):
        supplied = {k: v for k, v in opts.items() if k in _OLLAMA_RUNNER_OPTIONS and v is not None}
        if supplied:
            _compat_warning(
                "load-time Ollama runner option(s) accepted but managed by the configured "
                "llama-server command: " + ", ".join(f"{k}={v!r}" for k, v in sorted(supplied.items()))
            )


def _validate_tool_call(
    call: Any, path: str, *, expected_index: int | None = None
) -> str | None:
    if not isinstance(call, dict):
        return f"{path} must be an object"
    unexpected = set(call) - {"id", "function"}
    if unexpected:
        _compat_warning(
            f"{path} unknown field(s) ignored for Ollama compatibility: "
            + ", ".join(sorted(unexpected))
        )
    if "id" in call and call["id"] is not None and not isinstance(call["id"], str):
        return f"{path}.id must be a string"
    fn = call.get("function")
    if not isinstance(fn, dict):
        return f"{path}.function must be an object"
    unexpected_fn = set(fn) - {"index", "name", "arguments"}
    if unexpected_fn:
        _compat_warning(
            f"{path}.function unknown field(s) ignored for Ollama compatibility: "
            + ", ".join(sorted(unexpected_fn))
        )
    if not isinstance(fn.get("name"), str) or not fn.get("name"):
        return f"{path}.function.name must be a non-empty string"
    # Function.Index is stream metadata in Ollama. OpenAI-compatible historical
    # assistant tool calls do not carry an index field, but array order and call IDs
    # preserve the semantic association. Validate the value when present; do not
    # reject a perfectly valid history merely because its stored index is not equal
    # to its current array position.
    effective_index = fn.get("index")
    if effective_index is None:
        effective_index = 0
    if (
        not isinstance(effective_index, int)
        or isinstance(effective_index, bool)
        or effective_index < 0
    ):
        return f"{path}.function.index must be a non-negative integer"
    # Native Ollama ToolCallFunctionArguments is a JSON object. Do not silently
    # stringify strings or coerce arrays/scalars here.
    if "arguments" in fn and fn["arguments"] is not None and not isinstance(fn["arguments"], dict):
        return f"{path}.function.arguments must be a JSON object or null"
    return None


def _validate_tool_definition(tool: Any, path: str) -> str | None:
    if not isinstance(tool, dict):
        return f"{path} must be an object"
    if tool.get("type") != "function":
        return f"{path}.type must be exactly 'function'"
    # Current Ollama Tool also has an optional top-level `items` field. OpenAI
    # function tools do not, so preserve the function semantics and warn/drop any
    # other JSON fields exactly as Ollama's struct unmarshal would.
    unexpected = set(tool) - {"type", "function", "items"}
    if unexpected:
        _compat_warning(
            f"{path} unknown field(s) ignored for Ollama compatibility: "
            + ", ".join(sorted(unexpected))
        )
    if tool.get("items") is not None:
        _compat_warning(
            f"{path}.items is accepted by Ollama but has no OpenAI function-tool equivalent; "
            "it is ignored by this llama-server bridge"
        )
    fn = tool.get("function")
    if not isinstance(fn, dict):
        return f"{path}.function must be an object"
    unexpected_fn = set(fn) - {"name", "description", "parameters"}
    if unexpected_fn:
        _compat_warning(
            f"{path}.function unknown field(s) ignored for Ollama compatibility: "
            + ", ".join(sorted(unexpected_fn))
        )
    if not isinstance(fn.get("name"), str) or not fn.get("name"):
        return f"{path}.function.name must be a non-empty string"
    if "description" in fn and fn["description"] is not None and not isinstance(fn["description"], str):
        return f"{path}.function.description must be a string"
    if "parameters" in fn and fn["parameters"] is not None and not isinstance(fn["parameters"], dict):
        return f"{path}.function.parameters must be a JSON object"
    return None


def validate_read_only_overrides(
    body: dict[str, Any], cfg: ModelConfig, endpoint: str
) -> str | None:
    """Validate one Ollama request while preserving Ollama compatibility.

    Type/shape errors and contradictions remain errors. Controls that Ollama
    accepts but this managed runner cannot enact per request are accepted with a
    stderr warning; only cases that would require fabricating different prompt or
    transcript semantics are rejected explicitly.
    """
    allowed = _CHAT_REQUEST_FIELDS if endpoint == "chat" else _GENERATE_REQUEST_FIELDS
    unexpected_fields = sorted(set(body) - allowed)
    if unexpected_fields:
        _compat_warning(
            "unknown Ollama request field(s) ignored for compatibility: "
            + ", ".join(repr(k) for k in unexpected_fields)
        )

    if "stream" in body and body["stream"] is not None and not isinstance(body["stream"], bool):
        return "'stream' must be a boolean or null"
    if "raw" in body and body["raw"] is not None and not isinstance(body["raw"], bool):
        return "'raw' must be a boolean or null"

    # Current Ollama clients may explicitly send these controls even when their
    # value is the normal default. Rejecting them makes the compatibility surface
    # less faithful than omission. Validate the JSON shape here; the managed-runner
    # limitations are reported to stderr by _warn_non_native_controls().
    keep_alive = body.get("keep_alive")
    if keep_alive is not None and (
        isinstance(keep_alive, bool)
        or not isinstance(keep_alive, (str, int, float))
    ):
        return "'keep_alive' must be a duration string, number, or null"
    if isinstance(keep_alive, float) and not math.isfinite(keep_alive):
        return "'keep_alive' must be finite"

    for field in ("truncate", "shift", "_debug_render_only"):
        if field in body and body[field] is not None and not isinstance(body[field], bool):
            return f"{field!r} must be a boolean or null"

    if (
        "think" in body
        and body["think"] is not None
        and _requested_think_enabled(body["think"]) is None
    ):
        return "'think' must be a boolean, null, or a non-empty model-defined string level"

    if "reasoning_effort" in body and body["reasoning_effort"] is not None:
        value = body["reasoning_effort"]
        if not isinstance(value, str) or not value.strip():
            return "'reasoning_effort' must be a non-empty string or null"
        if value != value.strip():
            return "'reasoning_effort' must not have leading/trailing whitespace"

    if "reasoning_preserve" in body and body["reasoning_preserve"] is not None:
        if _requested_reasoning_preserve(body["reasoning_preserve"]) is None:
            return "'reasoning_preserve' must be a boolean, 'on', 'off', or null"

    # The local reasoning_effort extension and Ollama's think control must not
    # contradict each other. Silently choosing one would make the effective
    # request depend on shim precedence rather than the caller's stated intent.
    explicit_enabled = (
        _requested_think_enabled(body["think"])
        if "think" in body and body["think"] is not None
        else None
    )
    explicit_effort = _explicit_reasoning_effort(body)
    if explicit_enabled is False and explicit_effort not in (None, "none"):
        return "'think: off/false' conflicts with a non-'none' reasoning effort"
    if explicit_enabled is True and isinstance(explicit_effort, str) and explicit_effort.lower() == "none":
        return "enabled 'think' conflicts with reasoning_effort='none'"

    opts_value = body.get("options", {})
    if opts_value is None:
        opts: dict[str, Any] = {}
    elif not isinstance(opts_value, dict):
        return "'options' must be an object"
    else:
        opts = opts_value

    unsupported_options = sorted(set(opts) - _TRANSLATABLE_OLLAMA_OPTIONS)
    if unsupported_options:
        _compat_warning(
            "unknown Ollama option(s) ignored, matching Ollama Options.FromMap behavior: "
            + ", ".join(repr(k) for k in unsupported_options)
        )

    # Runner controls are part of the current Ollama Options schema, but this
    # managed shim starts llama-server from YAML and cannot resize/rebind it on a
    # single request. Accept and validate them, then leave the configured command
    # authoritative (with a stderr warning rather than an HTTP rejection).
    for key in ("num_ctx", "num_batch", "num_gpu", "main_gpu", "num_thread", "draft_num_predict"):
        if key not in opts or opts[key] is None:
            continue
        if not isinstance(opts[key], int) or isinstance(opts[key], bool):
            return f"option {key!r} must be an integer"
    if "use_mmap" in opts and opts["use_mmap"] is not None and not isinstance(opts["use_mmap"], bool):
        return "option 'use_mmap' must be a boolean or null"

    # Current Ollama rejects typical_p for new requests. Preserve that behavior
    # instead of forwarding a llama.cpp-only value through the Ollama surface.
    if opts.get("typical_p") is not None:
        return "option 'typical_p' is no longer supported by current Ollama"

    integer_minimums = {
        "top_k": 0,
        "repeat_last_n": -1,
        "dry_allowed_length": 0,
        "dry_penalty_last_n": -1,
        "reasoning_budget": -1,
        # Keep Ollama's historical -2 sentinel valid at the API boundary. It is
        # handled explicitly by translate_options() instead of being forwarded to
        # llama-server as an invalid max_tokens value.
        "num_predict": -2,
        "num_keep": -1,
        "mirostat": 0,
        "seed": -1,
    }
    for key, minimum in integer_minimums.items():
        if key not in opts:
            continue
        value = opts[key]
        # Ollama Options.FromMap treats JSON null as unset. Preserve omission
        # rather than rejecting it or converting it to a numeric zero.
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            return f"option {key!r} must be an integer"
        if value < minimum:
            return f"option {key!r} must be >= {minimum}"

    number_options = {
        "temperature", "top_p", "min_p", "repeat_penalty",
        "presence_penalty", "frequency_penalty", "dry_multiplier", "dry_base",
        "mirostat_tau", "mirostat_eta",
    }
    for key in sorted(number_options.intersection(opts)):
        value = opts[key]
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"option {key!r} must be a number"
        if not math.isfinite(float(value)):
            return f"option {key!r} must be finite"
        if key in {"top_p", "min_p"} and not 0 <= value <= 1:
            return f"option {key!r} must be between 0 and 1"
        if key == "dry_base" and value < 1:
            return "option 'dry_base' must be >= 1"
        if key in {
            "temperature", "repeat_penalty", "dry_multiplier",
            "mirostat_tau", "mirostat_eta",
        } and value < 0:
            return f"option {key!r} must be >= 0"

    if "mirostat" in opts and opts["mirostat"] is not None and opts["mirostat"] not in {0, 1, 2}:
        return "option 'mirostat' must be 0, 1, or 2"

    stop = opts.get("stop")
    if stop is not None and not (
        isinstance(stop, str)
        or (isinstance(stop, list) and all(isinstance(item, str) for item in stop))
    ):
        return "option 'stop' must be a string or an array of strings"

    fmt = body.get("format")
    if fmt not in (None, "", "json") and not isinstance(fmt, dict):
        return "'format' must be 'json' or a JSON Schema object"

    if (
        "logprobs" in body
        and body["logprobs"] is not None
        and not isinstance(body["logprobs"], bool)
    ):
        return "'logprobs' must be a boolean or null"
    if "top_logprobs" in body and body["top_logprobs"] is not None:
        top_logprobs = body["top_logprobs"]
        if (
            not isinstance(top_logprobs, int)
            or isinstance(top_logprobs, bool)
            or not 0 <= top_logprobs <= 20
        ):
            return "'top_logprobs' must be an integer between 0 and 20, or null"
        if top_logprobs and not body.get("logprobs"):
            _compat_warning(
                "top_logprobs was supplied while logprobs is false/unset; Ollama treats it "
                "as inactive, so the value is accepted but not forwarded"
            )

    if endpoint == "chat":
        messages_value = body.get("messages")
        messages = [] if messages_value is None else messages_value
        if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
            return "'messages' must be an array of objects or null"
        for i, message in enumerate(messages):
            unexpected_msg = set(message) - {
                "role", "content", "thinking", "images", "tool_calls", "tool_name", "tool_call_id"
            }
            if unexpected_msg:
                _compat_warning(
                    f"messages[{i}] unknown field(s) ignored for Ollama compatibility: "
                    + ", ".join(sorted(unexpected_msg))
                )
            if "role" not in message or not isinstance(message["role"], str):
                return f"messages[{i}].role must be a string"
            role = message["role"].lower()
            # encoding/json decodes an omitted or null Go string field to its zero
            # value. Match Ollama here so assistant tool-call messages do not need
            # to carry a synthetic explicit content string.
            if "content" in message and message["content"] is not None and not isinstance(message["content"], str):
                return f"messages[{i}].content must be a string or null"
            if "thinking" in message and message["thinking"] is not None and not isinstance(message["thinking"], str):
                return f"messages[{i}].thinking must be a string or null"
            images_value = message.get("images")
            images = [] if images_value is None else images_value
            if not isinstance(images, list) or not all(isinstance(image, str) for image in images):
                return f"messages[{i}].images must be an array of base64 strings"
            for j, image in enumerate(images):
                try:
                    base64.b64decode(image, validate=True)
                except (binascii.Error, ValueError):
                    return f"messages[{i}].images[{j}] is not valid standard base64"
            tool_calls = message.get("tool_calls")
            if tool_calls is not None:
                if not isinstance(tool_calls, list):
                    return f"messages[{i}].tool_calls must be an array"
                for j, call in enumerate(tool_calls):
                    error = _validate_tool_call(
                        call, f"messages[{i}].tool_calls[{j}]", expected_index=j
                    )
                    if error:
                        return error
            if "tool_name" in message and message["tool_name"] is not None and not isinstance(message["tool_name"], str):
                return f"messages[{i}].tool_name must be a string or null"
            if "tool_call_id" in message and message["tool_call_id"] is not None and not isinstance(message["tool_call_id"], str):
                return f"messages[{i}].tool_call_id must be a string or null"
    else:
        for key in ("prompt", "system", "suffix", "template"):
            if key in body and body[key] is not None and not isinstance(body[key], str):
                return f"{key!r} must be a string"
        images_value = body.get("images")
        images = [] if images_value is None else images_value
        if not isinstance(images, list) or not all(isinstance(image, str) for image in images):
            return "'images' must be an array of base64 strings"
        for i, image in enumerate(images):
            try:
                base64.b64decode(image, validate=True)
            except (binascii.Error, ValueError):
                return f"images[{i}] is not valid standard base64"

        # Match Ollama's load semantics: an empty prompt is a model-load request.
        # Non-generation fields are not interpreted in that path, except raw-mode
        # conflicts that Ollama itself rejects before generation.
        prompt_is_empty = body.get("prompt") in (None, "")
        if body.get("raw") and (
            body.get("template") not in (None, "")
            or body.get("system") not in (None, "")
            or body.get("context") not in (None, [])
        ):
            return "raw mode does not support template, system, or context"

        if not prompt_is_empty:
            has_suffix = body.get("suffix") not in (None, "")
            # Ollama allows raw+suffix: raw only disables ordinary prompt templating,
            # while suffix still requests insertion/FIM. The /infill backend path
            # already represents that combination directly.
            raw_or_infill = bool(body.get("raw")) or has_suffix
            if images and raw_or_infill:
                return "raw/suffix generate requests with images cannot be translated losslessly"
            # Raw/FIM bypass ordinary chat templating in native Ollama as well.
            # Do not reject requests merely because this managed model has a chat
            # reasoning/system default; those controls are irrelevant to the raw
            # prompt path. Explicit client controls are accepted with a stderr
            # compatibility warning in _warn_non_native_controls().

            if body.get("template") not in (None, ""):
                return (
                    "per-request Ollama Go templates cannot be translated losslessly to "
                    "llama-server chat templates"
                )
            if body.get("context") not in (None, []):
                return (
                    "the legacy generate 'context' token array cannot be translated "
                    "losslessly; send the conversation through /api/chat instead"
                )

    if "tools" in body and body["tools"] is not None:
        tools = body["tools"]
        if not isinstance(tools, list):
            return "'tools' must be an array of function tool objects or null"
        for i, tool in enumerate(tools):
            error = _validate_tool_definition(tool, f"tools[{i}]")
            if error:
                return error

    # Read-only means the client may repeat the configured value, but may not
    # change it.
    if cfg.system_prompt_read_only:
        if endpoint == "chat":
            messages = body.get("messages") or []
            system_messages = [
                message
                for message in messages
                if isinstance(message, dict)
                and isinstance(message.get("role"), str)
                and message["role"].lower() == "system"
            ]
            if system_messages:
                same = (
                    len(system_messages) == 1
                    and system_messages[0].get("content") == cfg.system_prompt
                )
                if not same:
                    return (
                        f"model {cfg.name!r}: system prompt is read-only; "
                        "client value differs from configured value"
                    )
        elif endpoint == "generate" and "system" in body:
            client_system = body.get("system")
            if client_system not in (None, "") and client_system != cfg.system_prompt:
                return (
                    f"model {cfg.name!r}: system prompt is read-only; "
                    "client value differs from configured value"
                )

    if cfg.think_read_only and "think" in body:
        assert cfg.think_default is not None
        if not _think_matches_config(body.get("think"), cfg.think_default):
            configured = "on" if cfg.think_default else "off"
            return (
                f"model {cfg.name!r}: think is read-only ({configured}); "
                f"client value {body.get('think')!r} differs"
            )

    if cfg.reasoning_effort_read_only:
        requested_effort = _explicit_reasoning_effort(body)
        if requested_effort is not None and requested_effort != cfg.reasoning_effort_default:
            return (
                f"model {cfg.name!r}: reasoning-effort is read-only "
                f"({cfg.reasoning_effort_default!r}); client value {requested_effort!r} differs"
            )

    if cfg.reasoning_preserve_read_only and "reasoning_preserve" in body:
        raw_preserve = body.get("reasoning_preserve")
        if raw_preserve is not None:
            requested_preserve = _requested_reasoning_preserve(raw_preserve)
            if requested_preserve != cfg.reasoning_preserve_default:
                return (
                    f"model {cfg.name!r}: reasoning-preserve is read-only "
                    f"({cfg.reasoning_preserve_default!r}); client value {raw_preserve!r} differs"
                )

    differing: list[str] = []
    for key in sorted(cfg.read_only_options):
        # A JSON null option is unset in Ollama rather than an override.
        if key in opts and opts[key] is not None and opts[key] != cfg.generation_defaults[key]:
            differing.append(key)
    if differing:
        quoted = ", ".join(repr(key) for key in differing)
        return f"model {cfg.name!r}: read-only option(s) {quoted} differ from configured value"

    return None


def translate_options(
    body: dict[str, Any],
    generation_defaults: dict[str, Any],
    think_default: bool | None,
    force_options: frozenset[str] = frozenset(),
    think_force: bool = False,
    reasoning_effort_default: str | None = None,
    reasoning_effort_force: bool = False,
    reasoning_preserve_default: bool | None = None,
    reasoning_preserve_force: bool = False,
    endpoint: str = "chat",
) -> dict[str, Any]:
    opts = body.get("options") or {}

    # Explicit model-config values are policy. Otherwise preserve omission and
    # let llama-server/model defaults remain authoritative.
    effective_options: dict[str, Any] = dict(generation_defaults)
    for ollama_key, value in opts.items():
        # Ollama treats JSON null options as unset. Do not let null erase an
        # explicit YAML policy or reach llama-server as a different value.
        if value is None:
            continue
        if ollama_key in OLLAMA_OPTION_TO_LLAMA and ollama_key not in force_options:
            effective_options[ollama_key] = value

    out: dict[str, Any] = {
        OLLAMA_OPTION_TO_LLAMA[ollama_key]: value
        for ollama_key, value in effective_options.items()
        if ollama_key in OLLAMA_OPTION_TO_LLAMA and value is not None
    }

    if "num_predict" in opts and opts["num_predict"] is not None:
        num_predict = opts["num_predict"]
        if num_predict == -2:
            _compat_warning(
                "options.num_predict=-2 (fill context) has no exact llama-server OpenAI "
                "request equivalent; leaving the generation limit unset so the backend "
                "context boundary remains authoritative"
            )
        else:
            out["n_predict" if endpoint == "native" else "max_tokens"] = num_predict

    if "stop" in opts and opts["stop"] is not None:
        out["stop"] = opts["stop"]

    # Preserve the schema byte-for-JSON-value. Do not weaken, rewrite, or remove
    # unsupported JSON-Schema keywords; llama-server must either honor it or
    # reject it.
    fmt = body.get("format")
    if endpoint in {"completion", "native"}:
        if fmt == "json":
            out["json_schema"] = {}
        elif isinstance(fmt, dict):
            out["json_schema"] = fmt
    elif fmt == "json":
        out["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        out["response_format"] = {"type": "json_schema", "schema": fmt}

    if body.get("logprobs"):
        if endpoint == "native":
            out["n_probs"] = max(1, body.get("top_logprobs", 0))
        elif endpoint == "completion":
            out["logprobs"] = body.get("top_logprobs", 1)
        else:
            out["logprobs"] = True
            if "top_logprobs" in body and body["top_logprobs"] is not None:
                out["top_logprobs"] = body["top_logprobs"]

    # `think` is sent only when the client supplied it or the model config
    # explicitly defines/forces a value. An omitted value no longer becomes an
    # implicit shim policy.
    if think_force:
        think: Any = think_default
    elif "think" in body:
        think = body["think"]
    else:
        think = think_default
    enabled = _requested_think_enabled(think) if think is not None else None

    if think_force and "reasoning_effort" not in body:
        client_effort = None
    else:
        client_effort = _explicit_reasoning_effort(body)
    if reasoning_effort_force:
        effort = reasoning_effort_default
    elif client_effort is not None:
        effort = client_effort
    else:
        effort = reasoning_effort_default

    if reasoning_preserve_force:
        preserve = reasoning_preserve_default
    elif "reasoning_preserve" in body and body.get("reasoning_preserve") is not None:
        preserve = _requested_reasoning_preserve(body["reasoning_preserve"])
    else:
        preserve = reasoning_preserve_default

    if endpoint == "chat":
        chat_template_kwargs: dict[str, Any] = {}
        if isinstance(effort, str) and effort.lower() == "none":
            # `none` is itself an explicit effort value; preserve it.
            chat_template_kwargs["enable_thinking"] = False
            out["reasoning_effort"] = effort
        elif enabled is False:
            # Do not manufacture a separate effort value from `think: false`.
            chat_template_kwargs["enable_thinking"] = False
        elif enabled is True:
            chat_template_kwargs["enable_thinking"] = True
            # Do not override llama-server's response reasoning format here. Its
            # default `auto` already exposes reasoning_content when appropriate,
            # and an explicit server setting is backend policy that the shim
            # should not silently replace.
            if effort is not None:
                out["reasoning_effort"] = effort
        elif effort is not None:
            # An explicit effort requests reasoning. Preserve only that control;
            # leave response-format selection to llama-server's configured/auto
            # reasoning format.
            out["reasoning_effort"] = effort

        if preserve is not None:
            chat_template_kwargs["preserve_reasoning"] = preserve
            chat_template_kwargs["preserve_thinking"] = preserve

        if chat_template_kwargs:
            out["chat_template_kwargs"] = chat_template_kwargs

    return out


def apply_system_prompt(
    messages: list[dict[str, Any]],
    system_prompt: str | None,
    system_prompt_force: bool = False,
) -> list[dict[str, Any]]:
    """Apply configured system-prompt fallback or force policy."""
    if system_prompt_force:
        # `(force)` means the configured value always wins. Remove every
        # client-supplied system message first; `null` therefore forces no
        # system prompt at all.
        result = [
            dict(message)
            for message in messages
            if message.get("role") != "system"
        ]
        if system_prompt:
            result.insert(0, {"role": "system", "content": system_prompt})
        return result

    if not system_prompt:
        return messages

    # A caller-provided system message completely overrides the configured
    # default. Never concatenate the two prompts.
    if any(message.get("role") == "system" for message in messages):
        return messages

    result = [dict(m) for m in messages]
    result.insert(0, {"role": "system", "content": system_prompt})
    return result


def _image_mime_from_bytes(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in {b"avif", b"avis"}:
            return "image/avif"
        if brand in {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}:
            return "image/heic"
    return None


def _ollama_image_to_data_url(image: str) -> str:
    # Native Ollama `images` are JSON base64 strings decoded into []byte. A data
    # URL is a different wire representation, so accepting it here would make
    # this shim more permissive than the API it claims to expose.
    if image.startswith("data:"):
        raise ValueError("Ollama images must be raw base64 strings, not data URLs")
    try:
        raw = base64.b64decode(image, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("image is not valid standard base64") from e
    mime = _image_mime_from_bytes(raw)
    if mime is None:
        raise ValueError(
            "cannot determine image MIME type from base64 payload; refusing to label it incorrectly"
        )
    return f"data:{mime};base64,{image}"


def translate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        # Ollama's Message.UnmarshalJSON lowercases roles. Reproduce that exact
        # normalization before crossing into llama-server.
        role = m["role"].lower()
        content = m.get("content")
        if content is None:
            content = ""
        item: dict[str, Any] = {"role": role, "content": content}

        images = m.get("images") or []
        if images:
            parts: list[dict[str, Any]] = []
            content = m.get("content")
            if isinstance(content, str) and content:
                parts.append({"type": "text", "text": content})
            for j, image in enumerate(images):
                try:
                    url = _ollama_image_to_data_url(image)
                except ValueError as e:
                    raise ValueError(f"messages[{i}].images[{j}]: {e}") from e
                parts.append({"type": "image_url", "image_url": {"url": url}})
            item["content"] = parts

        thinking = m.get("thinking")
        if isinstance(thinking, str) and thinking:
            # Preserve the field rather than enforcing an Ollama-side role rule.
            # llama-server/template validation remains authoritative.
            item["reasoning_content"] = thinking

        if m.get("tool_calls"):
            tc_out: list[dict[str, Any]] = []
            for tc in m["tool_calls"]:
                fn = tc["function"]
                call: dict[str, Any] = {
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        # llama.cpp accepts either a JSON string or a JSON value
                        # for historical tool-call arguments. Keep Ollama's
                        # object representation intact instead of serializing it.
                        "arguments": fn.get("arguments") or {},
                    },
                }
                tc_id = tc.get("id")
                if isinstance(tc_id, str) and tc_id:
                    call["id"] = tc_id
                tc_out.append(call)
            item["tool_calls"] = tc_out

        tool_name = m.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            item["name"] = tool_name
        tool_call_id = m.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            item["tool_call_id"] = tool_call_id
        # Never synthesize a call ID from tool_name or positional guesses.

        out.append(item)
    return out


def translate_tools(tools: Any) -> Any:
    """Translate Ollama function tools without inventing or weakening schemas.

    Go's JSON decoder drops unknown struct fields before Ollama's renderer sees
    them. Mirror that behavior here so compatibility-only fields do not leak into
    llama-server's stricter OpenAI tool schema. The function parameter schema
    itself is preserved byte-for-semantics as a JSON value; it is never sanitized.
    """
    if not isinstance(tools, list):
        return tools
    out: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool["function"]
        fn_out: dict[str, Any] = {"name": fn["name"]}
        if isinstance(fn.get("description"), str) and fn.get("description"):
            fn_out["description"] = fn["description"]
        if "parameters" in fn and fn["parameters"] is not None:
            fn_out["parameters"] = fn["parameters"]
        out.append({"type": "function", "function": fn_out})
    return out


def finish_reason(reason: str | None) -> str:
    # Ollama exposes only stop/length here. llama-server's OpenAI endpoint uses
    # `tool_calls` to signal a successful tool turn; Ollama carries that semantic
    # in message.tool_calls and reports the turn itself as `stop`. Unknown backend
    # reasons cannot be represented losslessly, so fail instead of inventing an
    # Ollama value.
    if reason is None:
        return ""
    if reason in {"length", "max_tokens"}:
        return "length"
    if reason in {"stop", "tool_calls", "function_call"}:
        return "stop"
    raise BackendStreamError(
        f"llama-server returned unsupported finish_reason {reason!r}"
    )


def native_finish_reason(stop_type: str | None) -> str:
    if stop_type in (None, "none"):
        return ""
    if stop_type == "limit":
        return "length"
    if stop_type in {"word", "eos"}:
        return "stop"
    raise BackendStreamError(
        f"llama-server returned unsupported stop_type {stop_type!r}"
    )


def metrics_from_response(obj: dict[str, Any], started_ns: int) -> dict[str, int]:
    """Translate only metrics with a defensible llama.cpp -> Ollama mapping.

    `total_duration` and `load_duration` are intentionally omitted: measuring the
    proxy wall clock or writing zero would claim native Ollama semantics that the
    managed scheduler cannot reproduce.
    """
    usage = obj.get("usage")
    timings = obj.get("timings")
    if usage is None:
        usage = {}
    elif not isinstance(usage, dict):
        raise BackendStreamError("llama-server usage metadata was not an object")
    if timings is None:
        timings = {}
    elif not isinstance(timings, dict):
        raise BackendStreamError("llama-server timings metadata was not an object")

    result: dict[str, int] = {}

    prompt_total = usage.get("prompt_tokens")
    if prompt_total is None:
        prompt_n = timings.get("prompt_n")
        cache_n = timings.get("cache_n")
        if prompt_n is not None or cache_n is not None:
            for label, value in (("prompt_n", prompt_n), ("cache_n", cache_n)):
                if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                    raise BackendStreamError(f"llama-server {label} was not an integer")
            prompt_total = (prompt_n or 0) + (cache_n or 0)
    if prompt_total is not None:
        if not isinstance(prompt_total, int) or isinstance(prompt_total, bool):
            raise BackendStreamError("llama-server prompt token count was not an integer")
        result["prompt_eval_count"] = prompt_total

    cached: Any = None
    details = usage.get("prompt_tokens_details")
    if details is not None:
        if not isinstance(details, dict):
            raise BackendStreamError("llama-server prompt_tokens_details was not an object")
        cached = details.get("cached_tokens")
    if cached is None:
        cached = timings.get("cache_n")
    if cached is not None:
        if not isinstance(cached, int) or isinstance(cached, bool):
            raise BackendStreamError("llama-server cached prompt token count was not an integer")
        result["prompt_eval_cached_count"] = cached

    prompt_ms = timings.get("prompt_ms")
    if prompt_ms is not None:
        if not isinstance(prompt_ms, (int, float)) or isinstance(prompt_ms, bool):
            raise BackendStreamError("llama-server prompt_ms was not numeric")
        result["prompt_eval_duration"] = int(float(prompt_ms) * 1_000_000)

    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = timings.get("predicted_n")
    if completion_tokens is not None:
        if not isinstance(completion_tokens, int) or isinstance(completion_tokens, bool):
            raise BackendStreamError("llama-server completion token count was not an integer")
        result["eval_count"] = completion_tokens

    predicted_ms = timings.get("predicted_ms")
    if predicted_ms is not None:
        if not isinstance(predicted_ms, (int, float)) or isinstance(predicted_ms, bool):
            raise BackendStreamError("llama-server predicted_ms was not numeric")
        result["eval_duration"] = int(float(predicted_ms) * 1_000_000)

    return result


def ollama_logprobs(choice: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate logprobs without inventing or silently dropping information."""
    if "logprobs" not in choice or choice["logprobs"] is None:
        return []
    raw = choice["logprobs"]
    if not isinstance(raw, dict):
        raise BackendStreamError("llama-server logprobs was not an object")

    content = raw.get("content")
    if content is not None:
        if not isinstance(content, list):
            raise BackendStreamError("llama-server chat logprobs.content was not an array")
        result: list[dict[str, Any]] = []
        for i, entry in enumerate(content):
            if not isinstance(entry, dict):
                raise BackendStreamError(f"llama-server logprobs.content[{i}] was not an object")
            token = entry.get("token")
            logprob = entry.get("logprob")
            if not isinstance(token, str):
                raise BackendStreamError(f"llama-server logprobs.content[{i}].token was not a string")
            if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
                raise BackendStreamError(f"llama-server logprobs.content[{i}].logprob was not numeric")
            item: dict[str, Any] = {"token": token, "logprob": float(logprob)}
            if "bytes" in entry and entry["bytes"] is not None:
                if not isinstance(entry["bytes"], list) or not all(
                    isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 255
                    for v in entry["bytes"]
                ):
                    raise BackendStreamError(f"llama-server logprobs.content[{i}].bytes was invalid")
                item["bytes"] = entry["bytes"]

            alternatives = entry.get("top_logprobs")
            if alternatives is not None:
                if not isinstance(alternatives, list):
                    raise BackendStreamError(
                        f"llama-server logprobs.content[{i}].top_logprobs was not an array"
                    )
                top: list[dict[str, Any]] = []
                for j, alternative in enumerate(alternatives):
                    if not isinstance(alternative, dict):
                        raise BackendStreamError(
                            f"llama-server logprobs.content[{i}].top_logprobs[{j}] was not an object"
                        )
                    alt_token = alternative.get("token")
                    alt_logprob = alternative.get("logprob")
                    if not isinstance(alt_token, str):
                        raise BackendStreamError(
                            f"llama-server top logprob token at [{i}][{j}] was not a string"
                        )
                    if not isinstance(alt_logprob, (int, float)) or isinstance(alt_logprob, bool):
                        raise BackendStreamError(
                            f"llama-server top logprob value at [{i}][{j}] was not numeric"
                        )
                    top_item: dict[str, Any] = {
                        "token": alt_token,
                        "logprob": float(alt_logprob),
                    }
                    if "bytes" in alternative and alternative["bytes"] is not None:
                        if not isinstance(alternative["bytes"], list) or not all(
                            isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 255
                            for v in alternative["bytes"]
                        ):
                            raise BackendStreamError(
                                f"llama-server top logprob bytes at [{i}][{j}] was invalid"
                            )
                        top_item["bytes"] = alternative["bytes"]
                    top.append(top_item)
                if top:
                    item["top_logprobs"] = top
            result.append(item)
        return result

    # Legacy /v1/completions logprob shape.
    tokens = raw.get("tokens")
    token_logprobs = raw.get("token_logprobs")
    top_rows = raw.get("top_logprobs")
    if tokens is None and token_logprobs is None and top_rows is None:
        return []
    if not isinstance(tokens, list) or not isinstance(token_logprobs, list):
        raise BackendStreamError("llama-server completion logprobs arrays were malformed")
    if len(tokens) != len(token_logprobs):
        raise BackendStreamError("llama-server completion logprobs arrays had different lengths")
    if top_rows is None:
        top_rows = [None] * len(tokens)
    elif not isinstance(top_rows, list) or len(top_rows) != len(tokens):
        raise BackendStreamError("llama-server completion top_logprobs array had an invalid length")

    result: list[dict[str, Any]] = []
    for i, (token, logprob, alternatives) in enumerate(zip(tokens, token_logprobs, top_rows)):
        if not isinstance(token, str):
            raise BackendStreamError(f"llama-server completion token {i} was not a string")
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
            raise BackendStreamError(f"llama-server completion logprob {i} was not numeric")
        item: dict[str, Any] = {"token": token, "logprob": float(logprob)}
        if alternatives is not None:
            if not isinstance(alternatives, dict):
                raise BackendStreamError(f"llama-server completion top_logprobs[{i}] was not an object")
            top: list[dict[str, Any]] = []
            for alt_token, alt_logprob in alternatives.items():
                if not isinstance(alt_token, str):
                    raise BackendStreamError(f"llama-server completion top token {i} was not a string")
                if not isinstance(alt_logprob, (int, float)) or isinstance(alt_logprob, bool):
                    raise BackendStreamError(f"llama-server completion top logprob {i} was not numeric")
                top.append({"token": alt_token, "logprob": float(alt_logprob)})
            if top:
                item["top_logprobs"] = top
        result.append(item)
    return result


def ollama_native_logprobs(probs: Any) -> list[dict[str, Any]]:
    """Translate llama-server native `/completion` probabilities losslessly."""
    if probs is None:
        return []
    if not isinstance(probs, list):
        raise BackendStreamError("llama-server native probabilities were not an array")
    result: list[dict[str, Any]] = []
    for i, entry in enumerate(probs):
        if not isinstance(entry, dict):
            raise BackendStreamError(f"llama-server native probs[{i}] was not an object")
        token = entry.get("token")
        logprob = entry.get("logprob")
        if not isinstance(token, str):
            raise BackendStreamError(f"llama-server native probs[{i}].token was not a string")
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
            raise BackendStreamError(f"llama-server native probs[{i}].logprob was not numeric")
        item: dict[str, Any] = {"token": token, "logprob": float(logprob)}
        if "bytes" in entry and entry["bytes"] is not None:
            if not isinstance(entry["bytes"], list) or not all(
                isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 255
                for v in entry["bytes"]
            ):
                raise BackendStreamError(f"llama-server native probs[{i}].bytes was invalid")
            item["bytes"] = entry["bytes"]
        alternatives = entry.get("top_logprobs")
        if alternatives is not None:
            if not isinstance(alternatives, list):
                raise BackendStreamError(f"llama-server native probs[{i}].top_logprobs was not an array")
            top: list[dict[str, Any]] = []
            for j, alt in enumerate(alternatives):
                if not isinstance(alt, dict):
                    raise BackendStreamError(f"llama-server native top_logprobs[{i}][{j}] was not an object")
                alt_token = alt.get("token")
                alt_logprob = alt.get("logprob")
                if not isinstance(alt_token, str):
                    raise BackendStreamError(f"llama-server native top token [{i}][{j}] was not a string")
                if not isinstance(alt_logprob, (int, float)) or isinstance(alt_logprob, bool):
                    raise BackendStreamError(f"llama-server native top logprob [{i}][{j}] was not numeric")
                top_item: dict[str, Any] = {"token": alt_token, "logprob": float(alt_logprob)}
                if "bytes" in alt and alt["bytes"] is not None:
                    if not isinstance(alt["bytes"], list) or not all(
                        isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 255
                        for v in alt["bytes"]
                    ):
                        raise BackendStreamError(f"llama-server native top bytes [{i}][{j}] was invalid")
                    top_item["bytes"] = alt["bytes"]
                top.append(top_item)
            if top:
                item["top_logprobs"] = top
        result.append(item)
    return result

def ollama_tool_calls(openai_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tc in openai_calls:
        if not isinstance(tc, dict):
            raise BackendStreamError("llama-server returned a non-object tool call")
        if tc.get("type") not in (None, "function"):
            raise BackendStreamError(
                f"llama-server returned unsupported tool-call type {tc.get('type')!r}"
            )
        fn = tc.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn.get("name"):
            raise BackendStreamError("llama-server returned a tool call without a function name")

        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                parsed_args: Any = json.loads(args)
            except json.JSONDecodeError as e:
                raise BackendStreamError(
                    f"llama-server returned invalid JSON tool arguments for {fn['name']!r}: {args!r}"
                ) from e
        else:
            parsed_args = args
        if not isinstance(parsed_args, dict):
            raise BackendStreamError(
                f"llama-server returned non-object tool arguments for {fn['name']!r}"
            )

        if "index" not in tc:
            raise BackendStreamError("llama-server returned a tool call without an index")
        index = tc["index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise BackendStreamError(f"llama-server returned invalid tool-call index {index!r}")

        item: dict[str, Any] = {
            "function": {
                "index": index,
                "name": fn["name"],
                "arguments": parsed_args,
            }
        }
        tc_id = tc.get("id")
        if tc_id is not None:
            if not isinstance(tc_id, str):
                raise BackendStreamError("llama-server returned a non-string tool-call id")
            if tc_id:
                item["id"] = tc_id
        result.append(item)
    return result


def merge_tool_delta(acc: dict[int, dict[str, Any]], calls: list[dict[str, Any]]) -> None:
    for tc in calls:
        if not isinstance(tc, dict):
            raise BackendStreamError("llama-server returned a non-object tool-call delta")
        idx = tc.get("index")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
            raise BackendStreamError(f"llama-server returned invalid tool-call delta index {idx!r}")
        if tc.get("type") not in (None, "function"):
            raise BackendStreamError(
                f"llama-server returned unsupported tool-call delta type {tc.get('type')!r}"
            )

        cur = acc.setdefault(
            idx,
            {
                "id": "",
                "index": idx,
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
        if "id" in tc and tc["id"] is not None:
            if not isinstance(tc["id"], str):
                raise BackendStreamError("llama-server returned a non-string tool-call id delta")
            incoming_id = tc["id"]
            if cur["id"] and incoming_id and incoming_id != cur["id"]:
                raise BackendStreamError(
                    f"llama-server changed tool-call id at index {idx} "
                    f"from {cur['id']!r} to {incoming_id!r}"
                )
            if incoming_id:
                cur["id"] = incoming_id

        if "function" in tc:
            fn = tc["function"]
            if not isinstance(fn, dict):
                raise BackendStreamError("llama-server returned a non-object tool-call function delta")
            if "name" in fn and fn["name"] is not None:
                if not isinstance(fn["name"], str):
                    raise BackendStreamError("llama-server returned a non-string tool name delta")
                cur["function"]["name"] += fn["name"]
            if "arguments" in fn and fn["arguments"] is not None:
                if not isinstance(fn["arguments"], str):
                    raise BackendStreamError("llama-server returned non-string streamed tool arguments")
                cur["function"]["arguments"] += fn["arguments"]


async def open_backend_stream(endpoint: str, payload: dict[str, Any]) -> httpx.Response:
    req = client.build_request("POST", f"{LLAMA_SERVER_URL}{endpoint}", json=payload)
    response = await client.send(req, stream=True)
    return _enable_full_llama_response_logging(response)


async def _apply_backend_chat_template(payload: dict[str, Any]) -> str:
    """Render the same llama-server chat template without running inference.

    `/apply-template` is the closest native llama-server equivalent to Ollama's
    `_debug_render_only` path. Keep only fields that participate in template
    rendering and preserve tool/reasoning template inputs where available.
    The response is routed through the raw-response tee before JSON decoding.
    """
    render_payload: dict[str, Any] = {
        "messages": payload["messages"],
        "add_generation_prompt": True,
    }
    for key in ("tools", "chat_template_kwargs", "reasoning_effort"):
        if key in payload:
            render_payload[key] = payload[key]

    req = client.build_request(
        "POST", f"{LLAMA_SERVER_URL}/apply-template", json=render_payload
    )
    response = _enable_full_llama_response_logging(
        await client.send(req, stream=True)
    )
    try:
        raw = await response.aread()
    finally:
        await response.aclose()

    if response.status_code >= 400:
        try:
            obj = json.loads(raw)
            message = backend_error_message(obj) if isinstance(obj, dict) else raw.decode("utf-8", errors="replace")
        except Exception:
            message = raw.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"llama-server /apply-template returned HTTP {response.status_code}: {message}"
        )

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"llama-server /apply-template returned malformed JSON: {e}"
        ) from e
    if not isinstance(obj, dict) or not isinstance(obj.get("prompt"), str):
        raise RuntimeError(
            "llama-server /apply-template response did not contain a string 'prompt'"
        )
    return obj["prompt"]


def _chat_image_count(body: dict[str, Any]) -> int:
    count = 0
    for message in body.get("messages") or []:
        if isinstance(message, dict) and isinstance(message.get("images"), list):
            count += len(message["images"])
    return count


def _debug_chat_response(model: str, rendered: str, image_count: int) -> JSONResponse:
    info: dict[str, Any] = {"rendered_template": rendered}
    if image_count:
        info["image_count"] = image_count
    # Match Ollama's zero-value ChatResponse fields in debug-render mode.
    return JSONResponse({
        "model": model,
        "created_at": iso_now(),
        "message": {"role": "", "content": ""},
        "done": False,
        "_debug_info": info,
    })


def _debug_generate_response(model: str, rendered: str, image_count: int) -> JSONResponse:
    info: dict[str, Any] = {"rendered_template": rendered}
    if image_count:
        info["image_count"] = image_count
    return JSONResponse({
        "model": model,
        "created_at": iso_now(),
        "response": "",
        "done": False,
        "_debug_info": info,
    })


class CheckpointError(RuntimeError):
    pass


class BackendStreamError(RuntimeError):
    pass


def _slot_id_from_headers(resp: httpx.Response) -> int | None:
    for key in (
        "X-Llama-Checkpoint-Slot",
        "X-Llama-Slot",
        "X-Llama-Id-Slot",
    ):
        value = resp.headers.get(key)
        if value is None:
            continue
        try:
            slot_id = int(value)
        except ValueError:
            continue
        if slot_id >= 0:
            return slot_id
    return None


def _slot_id_from_event(obj: Any) -> int | None:
    if not isinstance(obj, dict):
        return None

    candidates: list[Any] = [obj.get("id_slot")]
    verbose = obj.get("__verbose")
    if isinstance(verbose, dict):
        candidates.append(verbose.get("id_slot"))

    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _checkpoint_sentinel(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("__llama_checkpoint") == "suspended"


def _checkpoint_response_kind(resp: httpx.Response) -> str:
    # Older generation-only checkpoint servers did not send this header.
    value = resp.headers.get("X-Llama-Checkpoint-Kind")
    return value.strip().lower() if value else "generation"


async def _response_error_text(resp: httpx.Response) -> str:
    raw = await resp.aread()
    try:
        return backend_error_message(json.loads(raw))
    except Exception:
        return raw.decode("utf-8", errors="replace")


async def _open_initial_checkpoint_stream(
    job: ModelJob, endpoint: str, payload: dict[str, Any]
) -> httpx.Response:
    """Open a new generation while allowing cancellation/preemption before headers."""
    stable_payload = dict(payload)
    stable_payload["stream"] = True
    if job.slot_id is None:
        raise CheckpointError("initial generation has no reserved llama-server slot")
    stable_payload["id_slot"] = job.slot_id

    while True:
        if job.cancel_event.is_set():
            raise asyncio.CancelledError
        await MODEL_MANAGER.yield_if_preempted(job)
        open_task = asyncio.create_task(open_backend_stream(endpoint, stable_payload))
        cancel_task = asyncio.create_task(job.cancel_event.wait())
        preempt_task = (
            asyncio.create_task(job.preempt_event.wait()) if job.preemptible else None
        )
        try:
            wait_for: set[asyncio.Task[Any]] = {open_task, cancel_task}
            if preempt_task is not None:
                wait_for.add(preempt_task)
            done, _pending = await asyncio.wait(
                wait_for, return_when=asyncio.FIRST_COMPLETED
            )

            if cancel_task in done and job.cancel_event.is_set():
                open_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError):
                    await open_task
                raise asyncio.CancelledError

            if open_task in done:
                return open_task.result()

            # Preemption before response headers: abort and retry after resume.
            open_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError):
                await open_task
            await MODEL_MANAGER.pause_job(job)
            await MODEL_MANAGER.resume_job(job)
        finally:
            if not open_task.done():
                open_task.cancel()
            if not cancel_task.done():
                cancel_task.cancel()
            if preempt_task is not None and not preempt_task.done():
                preempt_task.cancel()


async def _request_checkpoint_suspend(slot_id: int) -> httpx.Response:
    # The global client intentionally has read=None because normal generation
    # may run indefinitely. A checkpoint suspend is different: after the
    # request reaches llama-server it should either hit a safe boundary or fail
    # quickly. Use an inactivity timeout so a backend control-flow bug cannot
    # make priority scheduling wait until the original generation finishes.
    print(f"Checkpoint suspend requested for slot {slot_id}", flush=True)
    try:
        response = await client.post(
            f"{LLAMA_SERVER_URL}/slots/{slot_id}?action=suspend",
            headers={"Accept": "application/octet-stream"},
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=60.0),
        )
    except httpx.HTTPError as e:
        print(f"Checkpoint suspend failed for slot {slot_id}: {e}", flush=True)
        raise
                                                                                        
    kind = _checkpoint_response_kind(response) if response.status_code < 400 else "error"
    print(
        f"Checkpoint suspend response for slot {slot_id}: HTTP {response.status_code}, "
        f"kind={kind}",
        flush=True,
    )
    return response


async def _open_checkpoint_resume_stream(job: ModelJob) -> httpx.Response:
    """Import the in-RAM checkpoint and return the continuation SSE response."""
    checkpoint = job.checkpoint
    if checkpoint is None:
        raise CheckpointError("resume requested without a checkpoint")

    while True:
        if job.cancel_event.is_set():
            raise asyncio.CancelledError
        await MODEL_MANAGER.yield_if_preempted(job)
        req = client.build_request(
            "POST",
            f"{LLAMA_SERVER_URL}/v1/checkpoint/resume",
            content=checkpoint,
            headers={
                "Content-Type": "application/octet-stream",
                "Accept": "text/event-stream",
            },
        )
        open_task = asyncio.create_task(client.send(req, stream=True))
        cancel_task = asyncio.create_task(job.cancel_event.wait())
        preempt_task = (
            asyncio.create_task(job.preempt_event.wait()) if job.preemptible else None
        )
        try:
            wait_for: set[asyncio.Task[Any]] = {open_task, cancel_task}
            if preempt_task is not None:
                wait_for.add(preempt_task)
            done, _pending = await asyncio.wait(
                wait_for, return_when=asyncio.FIRST_COMPLETED
            )

            if cancel_task in done and job.cancel_event.is_set():
                open_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError):
                    await open_task
                raise asyncio.CancelledError

            if open_task in done:
                return _enable_full_llama_response_logging(open_task.result())

            # No continuation output has reached the proxy. Keep the old opaque
            # checkpoint and retry it after the higher-priority model finishes.
            open_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError):
                await open_task
            await MODEL_MANAGER.pause_job(job)
            await MODEL_MANAGER.resume_job(job)
        finally:
            if not open_task.done():
                open_task.cancel()
            if not cancel_task.done():
                cancel_task.cancel()
            if preempt_task is not None and not preempt_task.done():
                preempt_task.cancel()


async def checkpoint_backend_lines(
    job: ModelJob,
    endpoint: str,
    payload: dict[str, Any],
    initial_response: httpx.Response | None = None,
) -> AsyncIterator[str]:
    """Expose one logical SSE stream across model unload/reload preemptions.

    The checkpoint blob is opaque. On a priority interrupt llama-server exports
    complete generation state at its ordered token/parser barrier. The bytes
    remain only in ``job.checkpoint`` while another model owns the backend.

    Important: never cancel an in-flight ``anext(response.aiter_lines())`` just
    because a preemption event fired. Cancelling that await can close httpx's
    response iterator and lose the private suspension sentinel. We keep one line
    read outstanding while the suspend request proceeds in parallel.
    """
    response: httpx.Response | None = initial_response

    try:
        while True:
            resuming = job.checkpoint is not None
            if response is None:
                response = (
                    await _open_checkpoint_resume_stream(job)
                    if resuming
                    else await _open_initial_checkpoint_stream(job, endpoint, payload)
                )

            if response.status_code >= 400:
                message = await _response_error_text(response)
                if resuming:
                    raise CheckpointError(
                        f"llama-server checkpoint resume failed with HTTP "
                        f"{response.status_code}: {message}"
                    )
                raise BackendStreamError(
                    f"llama-server returned HTTP {response.status_code}: {message}"
                )

            response_slot = _slot_id_from_headers(response)
            if response_slot is not None:
                if resuming and job.slot_id is not None and response_slot != job.slot_id:
                    print(
                        f"llama-server remapped checkpoint job {job.id} slot "
                        f"{job.slot_id} -> {response_slot}",
                        flush=True,
                    )
                job.slot_id = response_slot

            slot_id = job.slot_id
            if slot_id is None:
                raise CheckpointError(
                    "llama-server slot id is unknown; initial requests must use "
                    "an explicit id_slot and resumed responses must preserve or "
                    "report the checkpoint slot"
                )

            iterator = response.aiter_lines()
            suspend_task: asyncio.Task[httpx.Response] | None = None
            preempt_task: asyncio.Task[bool] | None = None
            cancel_task: asyncio.Task[bool] | None = asyncio.create_task(
                job.cancel_event.wait()
            )
            line_task: asyncio.Task[str] | None = asyncio.create_task(anext(iterator))
            # Preemption and cancellation use the same ordered checkpoint
            # suspension barrier. The only semantic difference is what happens
            # to the exported state: preemption retains it for resume, while
            # cancellation discards it and terminates the logical job.
            stop_reason: str | None = None  # None, "preempt", or "cancel"
            cancel_suspend_failed = False
            switch_after_checkpoint = False
            resumed_blob_is_stale = False
            idle_race_checkpoint_adopted = False

            try:
                while True:
                    # Cancellation dominates preemption, but both transition the
                    # backend through the same checkpoint suspend protocol. Never
                    # tear down line_task while a suspend is in flight: llama-server
                    # needs the original SSE reader alive so the private suspension
                    # sentinel can cross the ordered barrier.
                    if job.cancel_event.is_set():
                        stop_reason = "cancel"
                    elif (
                        stop_reason is None
                        and job.preemptible
                        and job.preempt_event.is_set()
                    ):
                        stop_reason = "preempt"

                    if (
                        stop_reason is not None
                        and job.preemptible
                        and suspend_task is None
                        and not (stop_reason == "cancel" and cancel_suspend_failed)
                    ):
                        suspend_task = asyncio.create_task(
                            _request_checkpoint_suspend(slot_id)
                        )

                    if (
                        stop_reason is None
                        and job.preemptible
                        and suspend_task is None
                        and preempt_task is None
                    ):
                        preempt_task = asyncio.create_task(job.preempt_event.wait())

                    assert line_task is not None
                    wait_for: set[asyncio.Task[Any]] = {line_task}
                    if cancel_task is not None:
                        wait_for.add(cancel_task)
                    if preempt_task is not None:
                        wait_for.add(preempt_task)
                    if suspend_task is not None and not suspend_task.done():
                        wait_for.add(suspend_task)

                    done, _pending = await asyncio.wait(
                        wait_for, return_when=asyncio.FIRST_COMPLETED
                    )

                    if cancel_task is not None and cancel_task in done:
                        cancel_task = None
                        if job.cancel_event.is_set():
                            stop_reason = "cancel"
                            if preempt_task is not None and not preempt_task.done():
                                preempt_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await preempt_task
                            preempt_task = None
                            if (
                                job.preemptible
                                and suspend_task is None
                                and not cancel_suspend_failed
                            ):
                                suspend_task = asyncio.create_task(
                                    _request_checkpoint_suspend(slot_id)
                                )

                    if preempt_task is not None and preempt_task in done:
                        preempt_task = None
                        if stop_reason is None:
                            stop_reason = "preempt"
                        if (
                            stop_reason == "preempt"
                            and job.preemptible
                            and suspend_task is None
                        ):
                            suspend_task = asyncio.create_task(
                                _request_checkpoint_suspend(slot_id)
                            )

                    # Export failure leaves the original generation viable by
                    # checkpoint-v1 contract. For preemption, disable further
                    # checkpoint attempts and keep serving the generation. For a
                    # terminal cancellation, however, the caller is already gone:
                    # abort this HTTP stream instead of draining generation to EOS.
                    # Closing the stream restores ordinary llama-server disconnect
                    # cancellation semantics and, critically, does not leave the
                    # proxy task pinned to a generation nobody can consume.
                    if suspend_task is not None and suspend_task.done():
                        try:
                            suspend_result = suspend_task.result()
                        except (httpx.HTTPError, asyncio.TimeoutError) as e:
                            if stop_reason == "cancel":
                                print(
                                    f"WARNING: checkpoint cancellation failed for model "
                                    f"{job.model_name!r}, job {job.id}: {e}; aborting "
                                    f"backend stream",
                                    flush=True,
                                )
                                cancel_suspend_failed = True
                                return
                            else:
                                await MODEL_MANAGER.mark_unpreemptible(
                                    job, f"suspend request failed: {e}"
                                )
                            suspend_task = None
                        else:
                            if suspend_result.status_code >= 400:
                                message = await _response_error_text(suspend_result)
                                if stop_reason == "cancel":
                                    print(
                                        f"WARNING: checkpoint cancellation for model "
                                        f"{job.model_name!r}, job {job.id} returned HTTP "
                                        f"{suspend_result.status_code}: {message}; aborting "
                                        f"backend stream",
                                        flush=True,
                                    )
                                    cancel_suspend_failed = True
                                    return
                                else:
                                    await MODEL_MANAGER.mark_unpreemptible(
                                        job,
                                        f"suspend HTTP {suspend_result.status_code}: {message}",
                                    )
                                suspend_task = None
                            elif _checkpoint_response_kind(suspend_result) == "idle":
                                # The generation finished between the stop request
                                # and action=suspend reaching llama-server. For
                                # cancellation that is already terminal success;
                                # for preemption preserve the reusable idle KV and
                                # let the normal stream finish.
                                if stop_reason == "cancel":
                                    print(
                                        f"Cancelled model {job.model_name!r}, job {job.id}: "
                                        f"slot {slot_id} was already idle",
                                        flush=True,
                                    )
                                    job.checkpoint = None
                                    return
                                if not idle_race_checkpoint_adopted:
                                    blob = bytes(suspend_result.content)
                                    version = suspend_result.headers.get(
                                        "X-Llama-Checkpoint-Version"
                                    )
                                    if blob and version in (None, "1", "2"):
                                        await MODEL_MANAGER.remember_idle_checkpoint(
                                            job.model_name, slot_id, blob
                                        )
                                        idle_race_checkpoint_adopted = True
                                        print(
                                            f"Generation for model {job.model_name!r}, "
                                            f"job {job.id} completed before suspend; saved "
                                            f"{len(blob)} byte idle KV checkpoint",
                                            flush=True,
                                        )

                    if line_task not in done:
                        continue

                    try:
                        line = line_task.result()
                    except StopAsyncIteration:
                        if (
                            suspend_task is not None
                            and suspend_task.done()
                            and suspend_task.result().status_code < 400
                            and _checkpoint_response_kind(suspend_task.result()) == "generation"
                        ):
                            raise CheckpointError(
                                "llama-server returned a generation checkpoint blob but the "
                                "original stream ended without the suspension sentinel"
                            )
                        if suspend_task is not None and not suspend_task.done():
                            suspend_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await suspend_task
                        return

                    # The current read has been consumed. Create the next read
                    # only after this event is fully processed.
                    line_task = None
                    stripped = line.strip()
                    if not stripped or stripped.startswith(":"):
                        line_task = asyncio.create_task(anext(iterator))
                        continue

                    data_text = (
                        stripped[5:].strip()
                        if stripped.startswith("data:")
                        else stripped
                    )
                    if data_text == "[DONE]":
                        if (
                            suspend_task is not None
                            and suspend_task.done()
                            and suspend_task.result().status_code < 400
                            and _checkpoint_response_kind(suspend_task.result()) == "generation"
                        ):
                            raise CheckpointError(
                                "llama-server completed normally after exporting a generation "
                                "checkpoint instead of emitting the suspension sentinel"
                            )
                        if suspend_task is not None and not suspend_task.done():
                            suspend_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await suspend_task
                        if stop_reason != "cancel":
                            yield line
                        return

                    try:
                        obj = json.loads(data_text)
                    except json.JSONDecodeError:
                        if stop_reason != "cancel":
                            yield line
                        line_task = asyncio.create_task(anext(iterator))
                        continue

                    discovered_slot = _slot_id_from_event(obj)
                    if discovered_slot is not None and discovered_slot != slot_id:
                        print(
                            f"llama-server reported slot {discovered_slot} for job {job.id} "
                            f"(reserved {slot_id})",
                            flush=True,
                        )
                        slot_id = discovered_slot
                        job.slot_id = discovered_slot

                    if _checkpoint_sentinel(obj):
                        if suspend_task is None:
                            raise CheckpointError(
                                "llama-server emitted a checkpoint suspension sentinel "
                                "without an active suspend request"
                            )
                        try:
                            suspended = await suspend_task
                        except (httpx.HTTPError, asyncio.TimeoutError) as e:
                            raise CheckpointError(
                                f"checkpoint sentinel arrived but export response failed: {e}"
                            ) from e
                        # Cancellation is terminal and dominates an in-flight
                        # scheduler preemption even if it arrived in the tiny
                        # race between asyncio.wait() and this sentinel.
                        if job.cancel_event.is_set():
                            stop_reason = "cancel"

                        if suspended.status_code >= 400:
                            message = await _response_error_text(suspended)
                            raise CheckpointError(
                                f"llama-server emitted suspension sentinel but export "
                                f"failed with HTTP {suspended.status_code}: {message}"
                            )
                        if _checkpoint_response_kind(suspended) != "generation":
                            raise CheckpointError(
                                "llama-server emitted a generation suspension sentinel but "
                                "returned a non-generation checkpoint blob"
                            )
                        blob = bytes(suspended.content)
                        if not blob:
                            raise CheckpointError(
                                "llama-server returned an empty checkpoint blob"
                            )
                        version = suspended.headers.get("X-Llama-Checkpoint-Version")
                        if version not in (None, "1", "2"):
                            raise CheckpointError(
                                f"unsupported llama checkpoint version {version!r}"
                            )

                        if stop_reason == "cancel":
                            # Same safe backend transition as preemption, but this
                            # logical job is terminal: deliberately discard the
                            # exported continuation state instead of resuming it.
                            job.checkpoint = None
                            print(
                                f"Cancelled model {job.model_name!r}, job {job.id} "
                                f"at checkpoint boundary; discarded {len(blob)} "
                                f"checkpoint byte(s)",
                                flush=True,
                            )
                            return

                        job.checkpoint = blob
                        # Keep the explicit slot reservation with the logical
                        # job. Resume v1 should restore the saved slot id; if a
                        # build remaps it, X-Llama-Checkpoint-Slot updates it.
                        print(
                            f"Suspended model {job.model_name!r}, job {job.id}: "
                            f"{len(blob)} checkpoint byte(s) held in RAM",
                            flush=True,
                        )
                        switch_after_checkpoint = True
                        break

                    # Once a resumed server has emitted its first ordinary event,
                    # the imported checkpoint blob is no longer a reusable
                    # frontier. Free RAM immediately.
                    if resuming and not resumed_blob_is_stale:
                        job.checkpoint = None
                        resumed_blob_is_stale = True

                    # Any __verbose metadata is scheduling-only. The translators
                    # construct fresh Ollama objects and do not expose it. Once
                    # cancellation is pending, continue consuming only to drive
                    # the checkpoint barrier (or safely drain an unsupported
                    # generation); do not enqueue more output for the dead client.
                    if stop_reason != "cancel":
                        yield line
                    line_task = asyncio.create_task(anext(iterator))
            finally:
                if cancel_task is not None and not cancel_task.done():
                    cancel_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await cancel_task
                if preempt_task is not None and not preempt_task.done():
                    preempt_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await preempt_task
                if line_task is not None and not line_task.done():
                    line_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                        await line_task
                if (
                    suspend_task is not None
                    and not suspend_task.done()
                    and not switch_after_checkpoint
                ):
                    suspend_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await suspend_task

            if not switch_after_checkpoint:
                return

            await response.aclose()
            response = None
            await MODEL_MANAGER.pause_job(job)
            await MODEL_MANAGER.resume_job(job)
            # Loop back: the opaque RAM blob is imported by the newly resident
            # llama-server through /v1/checkpoint/resume.
    finally:
        if response is not None and not response.is_closed:
            await response.aclose()


async def buffered_backend_lines(
    source: AsyncIterator[str], job: ModelJob
) -> AsyncIterator[str]:
    """Decouple backend generation/preemption from front-end backpressure.

    StreamingResponse pulls its iterator only as quickly as the Ollama client
    accepts bytes. If we consumed llama-server directly from that iterator, a
    slow client could prevent a low-priority job from noticing its preemption
    event. Pump backend lines into an unbounded in-memory queue instead: tokens
    already generated before the interrupt are preserved and may continue
    draining to the client while the model is switched, while the producer can
    checkpoint/suspend independently.
    """
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for line in source:
                await queue.put(("line", line))
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            await queue.put(("error", e))
        finally:
            # Backend execution is over even if the front-end client is still
            # slowly draining already-generated chunks from this queue. Release
            # the scheduler lease now so a waiting model is not blocked by
            # client-side backpressure. finish_job() is idempotent; the outer
            # stream translator also calls it as a disconnect/error safety net.
            await MODEL_MANAGER.finish_job(job)
            await queue.put(("eof", None))

    task = asyncio.create_task(pump())
    try:
        while True:
            kind, value = await queue.get()
            if kind == "line":
                yield value
                continue
            if kind == "error":
                raise value
            return
    finally:
        outer_cancelled = False
        if not task.done():
            # Front-end teardown is a logical cancellation request, not a reason
            # to kill the backend pump. checkpoint_backend_lines() must keep its
            # SSE reader alive until llama-server crosses the same ordered
            # suspension barrier used for scheduler preemption.
            job.cancel_event.set()
            while not task.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(task), timeout=35.0
                    )
                except asyncio.TimeoutError:
                    # _request_checkpoint_suspend() itself has a 30s inactivity
                    # timeout. Give checkpoint_backend_lines() a little longer to
                    # observe that failure and close the backend response. Do not
                    # cancel the pump here: it owns the response iterator teardown.
                    print(
                        f"WARNING: model {job.model_name!r}, job {job.id} is still "
                        f"finishing backend cancellation; continuing to wait",
                        flush=True,
                    )
                    continue
                except asyncio.CancelledError:
                    # Starlette may cancel the response task while this async
                    # generator is closing. Do not let that second cancellation
                    # tear down the checkpoint SSE reader. Temporarily consume
                    # the cancellation, finish backend cleanup, then re-raise it
                    # after the ModelJob is safe to release.
                    outer_cancelled = True
                    current = asyncio.current_task()
                    if current is not None:
                        current.uncancel()
                    continue

        with contextlib.suppress(asyncio.CancelledError):
            await task
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()
        if outer_cancelled:
            raise asyncio.CancelledError


def backend_error_message(data: Any) -> str:
    error = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(error, dict):
        return str(error.get("message") or json.dumps(error, ensure_ascii=False))
    return str(error)


async def backend_error_response(resp: httpx.Response) -> JSONResponse:
    raw = await resp.aread()
    await resp.aclose()
    try:
        data = json.loads(raw)
        msg = backend_error_message(data)
    except Exception:
        msg = raw.decode("utf-8", errors="replace")
    return _proxy_error_response(msg, resp.status_code)


async def read_json_object(
    request: Request,
) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    try:
        body = await request.json()
    except ValueError as e:
        return None, _proxy_error_response(
            f"request body must be valid JSON: {e}", 400
        )
    if not isinstance(body, dict):
        return None, _proxy_error_response("request body must be a JSON object", 400)
    return body, None


def resolve_internal_model(request: Request, requested: Any) -> str:
    """Resolve an internal request against the model-bound ingress port.

    An external-chain belongs to one configured model for its whole process
    lifetime. It may omit the model on internally generated calls, but it may
    never escape to another model through its private ingress.
    """
    bound = getattr(request.state, "bound_model", None)
    if bound is None:
        return MODEL_MANAGER.resolve_name(requested)
    if requested is None or requested == "":
        return bound
    resolved = MODEL_MANAGER.resolve_name(requested)
    if resolved != bound:
        raise KeyError(
            f"internal ingress for model {bound!r} rejected request for model {resolved!r}"
        )
    return bound


@app.api_route("/", methods=["GET", "HEAD"])
async def root() -> PlainTextResponse:
    return PlainTextResponse("Ollama is running (llama-server shim)\n")


@app.api_route("/api/version", methods=["GET", "HEAD"])
async def version() -> dict[str, str]:
    return {"version": "0.0.0-llama-server-shim"}


@app.api_route("/api/tags", methods=["GET", "HEAD"])
async def tags() -> JSONResponse:
    models: list[dict[str, Any]] = []
    for name, cfg in MODEL_MANAGER.models.items():
        cached = MODEL_MANAGER.metadata_cache.get(name) or {}
        m = cached.get("model") or {}
        props = cached.get("props") or {}
        if m:
            details, _ = model_details(m, name, props)
            meta = m.get("meta") or {}
            modified_at = model_modified_at(m, props) or "0001-01-01T00:00:00Z"
            size = int(meta.get("size") or 0)
            digest = model_digest(m, name)
        else:
            # Ollama's list schema has non-optional scalar fields. Use their
            # honest zero values rather than borrowing unrelated config metadata.
            details = {
                "parent_model": "",
                "format": "",
                "family": "",
                "families": [],
                "parameter_size": "",
                "quantization_level": "",
            }
            modified_at = "0001-01-01T00:00:00Z"
            size = 0
            digest = model_digest(None, name)

        for visible_name in (name, *cfg.aliases):
            models.append({
                "name": visible_name,
                "model": visible_name,
                "modified_at": modified_at,
                "size": size,
                "digest": digest,
                "details": details,
            })
    return JSONResponse({"models": models})


@app.get("/api/ps")
async def ps() -> JSONResponse:
    lease = await MODEL_MANAGER.acquire_running()
    if lease is None:
        return JSONResponse({"models": []})

    name, _cfg = lease
    try:
        try:
            m = await backend_model()
            props = await backend_props()
        except RuntimeError:
            return JSONResponse({"models": []})

        cache_backend_metadata(name, m, props)
        details, _ = model_details(m, name, props)
        meta = m.get("meta") or {}
        generation = props.get("default_generation_settings") or {}
        runtime_ctx = generation.get("n_ctx")
        # Training context length is not necessarily the active runtime context.
        # Report zero (unknown) instead of substituting a different quantity.
        context_length = (
            runtime_ctx
            if isinstance(runtime_ctx, int) and not isinstance(runtime_ctx, bool) and runtime_ctx > 0
            else 0
        )
        return JSONResponse({
            "models": [{
                "name": name,
                "model": name,
                "size": int(meta.get("size") or 0),
                "digest": model_digest(m, name),
                "details": details,
                # Managed residency has no Ollama TTL and llama-server does not
                # report a trustworthy per-model VRAM byte count here. Omit both
                # instead of inventing `expires_at`/`size_vram`.
                "context_length": context_length,
            }]
        })
    finally:
        await MODEL_MANAGER.release(name)


_SHOW_REQUEST_FIELDS = {"model", "name", "system", "template", "verbose", "options"}


def validate_show_request(body: dict[str, Any]) -> str | None:
    """Validate /api/show without silently ignoring request semantics."""
    unexpected = sorted(set(body) - _SHOW_REQUEST_FIELDS)
    if unexpected:
        _compat_warning(
            "unknown /api/show field(s) ignored for Ollama compatibility: "
            + ", ".join(repr(k) for k in unexpected)
        )

    for key in ("model", "name", "system", "template"):
        if key in body and body[key] is not None and not isinstance(body[key], str):
            return f"{key!r} must be a string"

    model = body.get("model") or ""
    legacy_name = body.get("name") or ""
    if not model and not legacy_name:
        return "model is required and must be a non-empty string"
    if model and legacy_name and model != legacy_name:
        return "'model' and deprecated 'name' refer to different models"

    # These are accepted by Ollama's ShowRequest. They only affect/decorate
    # metadata and should not make ordinary CLI introspection fail through the
    # bridge. Accept them, report the limitation on stderr, and return the managed
    # model metadata we actually know.
    if body.get("system") not in (None, ""):
        _compat_warning("/api/show system override accepted but cannot alter managed metadata")
    if body.get("template") not in (None, ""):
        _compat_warning("/api/show template override accepted but cannot alter managed metadata")
    if body.get("verbose") not in (None, False):
        if not isinstance(body.get("verbose"), bool):
            return "'/api/show verbose' must be a boolean or null"
        _compat_warning("/api/show verbose=true accepted; tensor metadata is unavailable and will be omitted")

    options = body.get("options")
    if options is not None:
        if not isinstance(options, dict):
            return "'/api/show options' must be an object or null"
        if options:
            _compat_warning("non-empty /api/show options accepted but do not change managed model metadata")

    return None


@app.post("/api/show")
async def show(request: Request) -> JSONResponse:
    """Return model metadata without starting or switching a backend."""
    body, body_error = await read_json_object(request)
    if body_error is not None:
        return body_error
    assert body is not None
    show_error = validate_show_request(body)
    if show_error is not None:
        return _proxy_error_response(show_error, 400)

    requested = body.get("model") or body.get("name")
    try:
        name = MODEL_MANAGER.resolve_name(requested)
    except KeyError as e:
        return _proxy_error_response(str(e).strip("'"), 404)

    cfg = MODEL_MANAGER.models[name]
    m: dict[str, Any] = {}
    props: dict[str, Any] = {}
    # If the requested model is already running, enrich /api/show from the live
    # backend. Crucially, never start/switch a model merely to answer metadata.
    lease = await MODEL_MANAGER.acquire_running()
    if lease is not None:
        running_name, _running_cfg = lease
        if running_name == name:
            try:
                try:
                    m = await backend_model()
                    props = await backend_props()
                    cache_backend_metadata(name, m, props)
                except RuntimeError:
                    # Metadata discovery must remain non-disruptive. Fall back
                    # to cached/config-only data if the live probe fails.
                    cached = MODEL_MANAGER.metadata_cache.get(name) or {}
                    m = cached.get("model") or {}
                    props = cached.get("props") or {}
            finally:
                await MODEL_MANAGER.release(running_name)
        else:
            # Another model is active; do not switch it just for /api/show.
            await MODEL_MANAGER.release(running_name)

    if not m:
        cached = MODEL_MANAGER.metadata_cache.get(name) or {}
        m = cached.get("model") or {}
        props = cached.get("props") or {}

    details, info = model_details(m, name, props)
    template_value = props.get("chat_template")
    template = template_value if isinstance(template_value, str) else ""
    system = cfg.system_prompt or ""
    param_items = ollama_parameters_from_props(props, cfg.generation_defaults)
    parameters = format_show_parameters(param_items)
    modified_at = model_modified_at(m, props)

    # Do not synthesize an Ollama Modelfile or license. llama-server does not
    # expose those source artifacts, and a plausible reconstruction would still
    # be invented metadata. Only return fields supported by actual backend/config
    # state; ShowResponse's optional fields may be omitted.
    response: dict[str, Any] = {
        "details": details,
        "model_info": info,
        "capabilities": capabilities(props),
    }
    if parameters:
        response["parameters"] = parameters
    if template:
        response["template"] = template
    if system:
        response["system"] = system
    if modified_at is not None:
        response["modified_at"] = modified_at
    return JSONResponse(response)

def checkpoint_request_supported_by_proxy(
    body: dict[str, Any], mode: str
) -> tuple[bool, str | None]:
    """Fast proxy-side gate for checkpoint-v1's deliberately narrow surface.

    llama-server remains authoritative and may still reject suspend with 409
    when startup/configured sampler features are not checkpointable. In that
    case the proxy leaves the original request alive and marks that job
    unpreemptible.
    """
    if mode == "preload":
        return False, "preload request has no generation state"
    if mode == "infill":
        return False, "checkpoint v1 does not support infill/FIM"
    if body.get("logprobs"):
        return False, "checkpoint v1 does not support logprob mode"
    options = body.get("options") or {}
    if isinstance(options, dict):
        if options.get("mirostat") not in (None, 0):
            return False, "checkpoint v1 does not support Mirostat"
    if body.get("images"):
        return False, "checkpoint v1 does not support multimodal generation"

    if mode == "chat":
        for message in body.get("messages") or []:
            if isinstance(message, dict) and message.get("images"):
                return False, "checkpoint v1 does not support multimodal chat"

    return True, None


def _backend_total_slots(props: dict[str, Any]) -> int:
    value = props.get("total_slots")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    # llama-server defaults to one parallel slot. Falling back to one preserves
    # correctness if an older build omits total_slots, at the cost of parallelism.
    return 1


async def _collect_chat_response(source: AsyncIterator[bytes]) -> JSONResponse:
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    logprobs: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None

    async for chunk in source:
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError as e:
            return _proxy_error_response(
                f"internal Ollama stream was not valid UTF-8: {e}", 502
            )
        for raw_line in text.splitlines():
            if not raw_line.strip():
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as e:
                return _proxy_error_response(
                    f"internal Ollama stream contained malformed JSON: {e}: {raw_line!r}", 502
                )
            if not isinstance(obj, dict):
                return _proxy_error_response(
                    "internal Ollama stream contained a non-object JSON value", 502
                )
            if obj.get("error") is not None:
                return _proxy_error_response(str(obj["error"]), 502)

            message_value = obj.get("message")
            if message_value is None:
                message: dict[str, Any] = {}
            elif isinstance(message_value, dict):
                message = message_value
            else:
                return _proxy_error_response("internal Ollama stream message was not an object", 502)

            content_value = message.get("content")
            if content_value is not None:
                if not isinstance(content_value, str):
                    return _proxy_error_response("internal Ollama message.content was not a string", 502)
                if content_value:
                    content_parts.append(content_value)
            thinking_value = message.get("thinking")
            if thinking_value is not None:
                if not isinstance(thinking_value, str):
                    return _proxy_error_response("internal Ollama message.thinking was not a string", 502)
                if thinking_value:
                    thinking_parts.append(thinking_value)
            if "tool_calls" in message:
                calls = message["tool_calls"]
                if not isinstance(calls, list):
                    return _proxy_error_response("internal Ollama tool_calls was not an array", 502)
                tool_calls.extend(calls)
            if "logprobs" in obj:
                internal_logprobs = obj["logprobs"]
                if not isinstance(internal_logprobs, list):
                    return _proxy_error_response("internal Ollama logprobs was not an array", 502)
                logprobs.extend(internal_logprobs)
            done_value = obj.get("done")
            if done_value is not None and not isinstance(done_value, bool):
                return _proxy_error_response("internal Ollama done was not a boolean", 502)
            if done_value is True:
                final = dict(obj)

    if final is None:
        return _proxy_error_response(
            "llama-server stream ended without a final Ollama response", 502
        )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if thinking_parts:
        message["thinking"] = "".join(thinking_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    final["message"] = message
    if logprobs:
        final["logprobs"] = logprobs
    else:
        final.pop("logprobs", None)
    return JSONResponse(final)


async def _collect_generate_response(source: AsyncIterator[bytes]) -> JSONResponse:
    response_parts: list[str] = []
    thinking_parts: list[str] = []
    logprobs: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None

    async for chunk in source:
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError as e:
            return _proxy_error_response(
                f"internal Ollama stream was not valid UTF-8: {e}", 502
            )
        for raw_line in text.splitlines():
            if not raw_line.strip():
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as e:
                return _proxy_error_response(
                    f"internal Ollama stream contained malformed JSON: {e}: {raw_line!r}", 502
                )
            if not isinstance(obj, dict):
                return _proxy_error_response(
                    "internal Ollama stream contained a non-object JSON value", 502
                )
            if obj.get("error") is not None:
                return _proxy_error_response(str(obj["error"]), 502)
            response_value = obj.get("response")
            if response_value is not None:
                if not isinstance(response_value, str):
                    return _proxy_error_response("internal Ollama response was not a string", 502)
                if response_value:
                    response_parts.append(response_value)
            thinking_value = obj.get("thinking")
            if thinking_value is not None:
                if not isinstance(thinking_value, str):
                    return _proxy_error_response("internal Ollama thinking was not a string", 502)
                if thinking_value:
                    thinking_parts.append(thinking_value)
            if "tool_calls" in obj:
                internal_calls = obj["tool_calls"]
                if not isinstance(internal_calls, list):
                    return _proxy_error_response("internal Ollama tool_calls was not an array", 502)
                tool_calls.extend(internal_calls)
            if "logprobs" in obj:
                internal_logprobs = obj["logprobs"]
                if not isinstance(internal_logprobs, list):
                    return _proxy_error_response("internal Ollama logprobs was not an array", 502)
                logprobs.extend(internal_logprobs)
            done_value = obj.get("done")
            if done_value is not None and not isinstance(done_value, bool):
                return _proxy_error_response("internal Ollama done was not a boolean", 502)
            if done_value is True:
                final = dict(obj)

    if final is None:
        return _proxy_error_response(
            "llama-server stream ended without a final Ollama response", 502
        )

    final["response"] = "".join(response_parts)
    if thinking_parts:
        final["thinking"] = "".join(thinking_parts)
    else:
        final.pop("thinking", None)
    if tool_calls:
        final["tool_calls"] = tool_calls
    else:
        final.pop("tool_calls", None)
    if logprobs:
        final["logprobs"] = logprobs
    else:
        final.pop("logprobs", None)
    return JSONResponse(final)


async def chat_stream(
    lines: AsyncIterator[str], started_ns: int, response_model: str, job: ModelJob
) -> AsyncIterator[bytes]:
    final_obj: dict[str, Any] = {}
    final_reason: str | None = None
    final_created_at = iso_now()
    tool_acc: dict[int, dict[str, Any]] = {}

    try:
        async for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise BackendStreamError(
                    f"llama-server emitted malformed JSON: {e}: {line!r}"
                ) from e
            if not isinstance(obj, dict):
                raise BackendStreamError("llama-server emitted a non-object JSON event")

            if obj.get("error") is not None:
                yield _ndjson_error(backend_error_message(obj))
                return

            if obj.get("created") is not None:
                final_created_at = iso_from_epoch(obj.get("created"))
            if obj.get("usage") is not None:
                final_obj["usage"] = obj["usage"]
            if obj.get("timings") is not None:
                final_obj["timings"] = obj["timings"]

            choices = obj.get("choices")
            if choices is None:
                # Some terminal/usage events contain metadata only.
                continue
            if not isinstance(choices, list):
                raise BackendStreamError("llama-server 'choices' was not an array")
            if len(choices) == 0:
                continue
            if len(choices) != 1:
                raise BackendStreamError(
                    f"llama-server returned {len(choices)} choices; Ollama chat supports one response"
                )
            choice = choices[0]
            if not isinstance(choice, dict):
                raise BackendStreamError("llama-server choice was not an object")
            choice_index = choice.get("index")
            if choice_index is not None and (
                not isinstance(choice_index, int)
                or isinstance(choice_index, bool)
                or choice_index != 0
            ):
                raise BackendStreamError(
                    f"llama-server returned unexpected choice index {choice_index!r}"
                )
            finish_value = choice.get("finish_reason")
            if finish_value is not None:
                if not isinstance(finish_value, str):
                    raise BackendStreamError("llama-server finish_reason was not a string")
                if final_reason is not None and finish_value != final_reason:
                    raise BackendStreamError(
                        f"llama-server changed finish_reason from {final_reason!r} to {finish_value!r}"
                    )
                final_reason = finish_value

            delta_value = choice.get("delta")
            if delta_value is None:
                delta: dict[str, Any] = {}
            elif isinstance(delta_value, dict):
                delta = delta_value
            else:
                raise BackendStreamError("llama-server choice.delta was not an object")

            content_value = delta.get("content")
            if content_value is None:
                content = ""
            elif isinstance(content_value, str):
                content = content_value
            else:
                raise BackendStreamError("llama-server delta.content was not a string")

            reasoning_content_value = delta.get("reasoning_content")
            reasoning_value = delta.get("reasoning")
            if (
                reasoning_content_value is not None
                and reasoning_value is not None
                and reasoning_content_value != reasoning_value
            ):
                raise BackendStreamError(
                    "llama-server emitted conflicting reasoning_content and reasoning deltas"
                )
            thinking_value = (
                reasoning_content_value
                if reasoning_content_value is not None
                else reasoning_value
            )
            if thinking_value is None:
                thinking = ""
            elif isinstance(thinking_value, str):
                thinking = thinking_value
            else:
                raise BackendStreamError("llama-server reasoning delta was not a string")

            if "tool_calls" in delta and delta["tool_calls"] is not None:
                calls = delta["tool_calls"]
                if not isinstance(calls, list):
                    raise BackendStreamError("llama-server tool_calls delta was not an array")
                merge_tool_delta(tool_acc, calls)

            logprobs = ollama_logprobs(choice)
            if content or thinking or logprobs:
                message: dict[str, Any] = {"role": "assistant", "content": content}
                if thinking:
                    message["thinking"] = thinking
                item: dict[str, Any] = {
                    "model": response_model,
                    "created_at": final_created_at,
                    "message": message,
                    "done": False,
                }
                if logprobs:
                    item["logprobs"] = logprobs
                yield ndjson(item)

        if final_reason is None:
            raise BackendStreamError(
                "llama-server chat stream ended without a terminal finish_reason"
            )
        if final_reason in {"tool_calls", "function_call"} and not tool_acc:
            raise BackendStreamError(
                f"llama-server finished with {final_reason!r} but emitted no tool call"
            )
        if tool_acc and final_reason not in {"tool_calls", "function_call"}:
            raise BackendStreamError(
                f"llama-server emitted tool calls but finished with {final_reason!r}"
            )

        if tool_acc:
            calls = [tool_acc[i] for i in sorted(tool_acc)]
            yield ndjson({
                "model": response_model,
                "created_at": final_created_at,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": ollama_tool_calls(calls),
                },
                "done": False,
            })

        final: dict[str, Any] = {
            "model": response_model,
            "created_at": final_created_at,
            "message": {"role": "assistant", "content": ""},
            "done": True,
        }
        done_reason = finish_reason(final_reason)
        if done_reason:
            final["done_reason"] = done_reason
        final.update(metrics_from_response(final_obj, started_ns))
        yield ndjson(final)
    except (RuntimeError, httpx.HTTPError) as e:
        yield _ndjson_error(str(e))
    finally:
        aclose = getattr(lines, "aclose", None)
        try:
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
        finally:
            await MODEL_MANAGER.finish_job(job)


@app.post("/internal/jobs/{job_id}/cancel")
async def cancel_internal_job(job_id: int):
    try:
        existed = await MODEL_MANAGER.cancel_job(job_id)
    except RuntimeError as e:
        return _proxy_error_response(str(e), 504)
    return JSONResponse(
        {
            "job_id": job_id,
            "cancelled": existed,
            "already_finished": not existed,
        }
    )


@app.post("/api/chat")
async def chat(request: Request):
    body, body_error = await read_json_object(request)
    if body_error is not None:
        return body_error
    assert body is not None
    started_ns = time.perf_counter_ns()
    if "model" in body and body["model"] is not None and not isinstance(body["model"], str):
        return _proxy_error_response("'model' must be a string", 400)

    try:
        requested_name = resolve_internal_model(request, body.get("model"))
        requested_cfg = MODEL_MANAGER.models[requested_name]
    except KeyError as e:
        return _proxy_error_response(str(e.args[0]), 404)
    ro_error = validate_read_only_overrides(body, requested_cfg, "chat")
    if ro_error:
        return _proxy_error_response(ro_error, 400)
    _warn_non_native_controls(body)

    preemptible, preempt_reason = checkpoint_request_supported_by_proxy(body, "chat")
    try:
        job, cfg = await MODEL_MANAGER.acquire_job(
            requested_name, preemptible=preemptible
        )
        job.preempt_disabled_reason = preempt_reason
        model_name = job.model_name
        response_model = (
            body["model"]
            if isinstance(body.get("model"), str) and body.get("model")
            else model_name
        )
    except KeyError as e:
        return _proxy_error_response(str(e.args[0]), 404)
    except RuntimeError as e:
        return _proxy_error_response(str(e), 502)

    streaming = True if body.get("stream") is None else body["stream"]
    stream_handed_off = False
    try:
        # Native Ollama treats an empty chat history as a model-load request.
        # The scheduler has already selected/started the runner; health confirms
        # readiness without sending a fake generation request.
        if not (body.get("messages") or []):
            try:
                await backend_health()
            except RuntimeError as e:
                return _proxy_error_response(str(e), 502)
            return JSONResponse({
                "model": response_model,
                "created_at": iso_now(),
                "message": {"role": "assistant", "content": ""},
                "done_reason": "load",
                "done": True,
            })

        try:
            m = await backend_model()
            props = await backend_props()
        except RuntimeError as e:
            return _proxy_error_response(str(e), 502)
        cache_backend_metadata(model_name, m, props)
        await MODEL_MANAGER.claim_backend_slot(job, _backend_total_slots(props))

        try:
            translated_messages = translate_messages(body.get("messages") or [])
            payload: dict[str, Any] = {
                "model": backend_model_id(m, model_name),
                "messages": apply_system_prompt(
                    translated_messages,
                    cfg.system_prompt,
                    cfg.system_prompt_force,
                ),
                # Backend streaming is mandatory internally so suspend can stop at
                # an ordered parser/token frontier even for Ollama stream:false.
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            payload.update(
                translate_options(
                    body,
                    cfg.generation_defaults,
                    cfg.think_default,
                    cfg.force_options,
                    cfg.think_force,
                    cfg.reasoning_effort_default,
                    cfg.reasoning_effort_force,
                    cfg.reasoning_preserve_default,
                    cfg.reasoning_preserve_force,
                    endpoint="chat",
                )
            )
            if "tools" in body and body["tools"] is not None:
                payload["tools"] = translate_tools(body["tools"])
        except ValueError as e:
            return _proxy_error_response(str(e), 400)

        if body.get("_debug_render_only") is True:
            try:
                rendered = await _apply_backend_chat_template(payload)
            except (httpx.HTTPError, RuntimeError) as e:
                return _proxy_error_response(str(e), 502)
            return _debug_chat_response(
                response_model, rendered, _chat_image_count(body)
            )

        if body.get("_debug_render_only") is True:
            try:
                rendered = await _apply_backend_chat_template(payload)
            except (httpx.HTTPError, RuntimeError) as e:
                return _proxy_error_response(str(e), 502)
            return _debug_generate_response(
                response_model, rendered, len(body.get("images") or [])
            )

        try:
            resp = await _open_initial_checkpoint_stream(
                job, "/v1/chat/completions", payload
            )
        except (httpx.HTTPError, RuntimeError) as e:
            return _proxy_error_response(f"llama-server: {e}", 502)
        if resp.status_code >= 400:
            return await backend_error_response(resp)

        lines = buffered_backend_lines(
            checkpoint_backend_lines(
                job,
                "/v1/chat/completions",
                payload,
                initial_response=resp,
            ),
            job,
        )
        source = chat_stream(lines, started_ns, response_model, job)
        if streaming:
            stream_handed_off = True
            return StreamingResponse(
                source,
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Ollama-Proxy-Job-ID": str(job.id),
                },
            )
        return await _collect_chat_response(source)
    finally:
        if not stream_handed_off:
            await MODEL_MANAGER.finish_job(job)


async def completion_stream(
    lines: AsyncIterator[str], started_ns: int, response_model: str, job: ModelJob
) -> AsyncIterator[bytes]:
    final_obj: dict[str, Any] = {}
    final_reason: str | None = None
    final_created_at = iso_now()
    try:
        async for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise BackendStreamError(
                    f"llama-server emitted malformed JSON: {e}: {line!r}"
                ) from e
            if not isinstance(obj, dict):
                raise BackendStreamError("llama-server emitted a non-object JSON event")

            if obj.get("error") is not None:
                yield _ndjson_error(backend_error_message(obj))
                return

            if obj.get("created") is not None:
                final_created_at = iso_from_epoch(obj.get("created"))
            if obj.get("usage") is not None:
                final_obj["usage"] = obj["usage"]
            if obj.get("timings") is not None:
                final_obj["timings"] = obj["timings"]

            choices = obj.get("choices")
            if choices is None:
                continue
            if not isinstance(choices, list):
                raise BackendStreamError("llama-server 'choices' was not an array")
            if not choices:
                continue
            if len(choices) != 1:
                raise BackendStreamError(
                    f"llama-server returned {len(choices)} choices; Ollama generate supports one response"
                )
            choice = choices[0]
            if not isinstance(choice, dict):
                raise BackendStreamError("llama-server choice was not an object")
            choice_index = choice.get("index")
            if choice_index is not None and (
                not isinstance(choice_index, int)
                or isinstance(choice_index, bool)
                or choice_index != 0
            ):
                raise BackendStreamError(
                    f"llama-server returned unexpected choice index {choice_index!r}"
                )
            finish_value = choice.get("finish_reason")
            if finish_value is not None:
                if not isinstance(finish_value, str):
                    raise BackendStreamError("llama-server finish_reason was not a string")
                if final_reason is not None and finish_value != final_reason:
                    raise BackendStreamError(
                        f"llama-server changed finish_reason from {final_reason!r} to {finish_value!r}"
                    )
                final_reason = finish_value

            text_value = choice.get("text")
            if text_value is None:
                text = ""
            elif isinstance(text_value, str):
                text = text_value
            else:
                raise BackendStreamError("llama-server completion text was not a string")

            logprobs = ollama_logprobs(choice)
            if text or logprobs:
                item: dict[str, Any] = {
                    "model": response_model,
                    "created_at": final_created_at,
                    "response": text,
                    "done": False,
                }
                if logprobs:
                    item["logprobs"] = logprobs
                yield ndjson(item)

        if final_reason is None:
            raise BackendStreamError(
                "llama-server completion stream ended without a terminal finish_reason"
            )

        final: dict[str, Any] = {
            "model": response_model,
            "created_at": final_created_at,
            "response": "",
            "done": True,
        }
        done_reason = finish_reason(final_reason)
        if done_reason:
            final["done_reason"] = done_reason
        final.update(metrics_from_response(final_obj, started_ns))
        yield ndjson(final)
    except (RuntimeError, httpx.HTTPError) as e:
        yield _ndjson_error(str(e))
    finally:
        aclose = getattr(lines, "aclose", None)
        try:
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
        finally:
            await MODEL_MANAGER.finish_job(job)


async def native_completion_stream(
    lines: AsyncIterator[str], started_ns: int, response_model: str, job: ModelJob
) -> AsyncIterator[bytes]:
    """Translate llama-server's native /completion or /infill SSE stream."""
    final_obj: dict[str, Any] = {}
    final_reason: str | None = None
    final_created_at = iso_now()
    saw_stop = False
    try:
        async for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise BackendStreamError(
                    f"llama-server emitted malformed JSON: {e}: {line!r}"
                ) from e
            if not isinstance(obj, dict):
                raise BackendStreamError("llama-server emitted a non-object JSON event")

            if obj.get("error") is not None:
                yield _ndjson_error(backend_error_message(obj))
                return

            if obj.get("created") is not None:
                final_created_at = iso_from_epoch(obj.get("created"))
            if obj.get("usage") is not None:
                final_obj["usage"] = obj["usage"]
            if obj.get("timings") is not None:
                final_obj["timings"] = obj["timings"]
            if obj.get("stop_type") is not None:
                if not isinstance(obj["stop_type"], str):
                    raise BackendStreamError("llama-server stop_type was not a string")
                stop_type = obj["stop_type"]
                if (
                    final_reason not in (None, "none")
                    and stop_type not in {"none", final_reason}
                ):
                    raise BackendStreamError(
                        f"llama-server changed stop_type from {final_reason!r} to {stop_type!r}"
                    )
                if stop_type != "none" or final_reason is None:
                    final_reason = stop_type

            text_value = obj.get("content")
            if text_value is None:
                text = ""
            elif isinstance(text_value, str):
                text = text_value
            else:
                raise BackendStreamError("llama-server native content was not a string")

            probs = obj.get("probs")
            if probs is None:
                probs = obj.get("completion_probabilities")
            if probs is None:
                probs = []
            if not isinstance(probs, list):
                raise BackendStreamError("llama-server native probabilities were not an array")
            logprobs = ollama_native_logprobs(probs)

            if text or logprobs:
                item: dict[str, Any] = {
                    "model": response_model,
                    "created_at": final_created_at,
                    "response": text,
                    "done": False,
                }
                if logprobs:
                    item["logprobs"] = logprobs
                yield ndjson(item)

            if "stop" in obj and not isinstance(obj["stop"], bool):
                raise BackendStreamError("llama-server native stop flag was not a boolean")
            if obj.get("stop") is True:
                saw_stop = True
                break

        if not saw_stop:
            raise BackendStreamError(
                "llama-server native stream ended without a terminal stop=true event"
            )
        if final_reason in (None, "none"):
            raise BackendStreamError(
                "llama-server native stream stopped without a terminal stop_type"
            )

        final: dict[str, Any] = {
            "model": response_model,
            "created_at": final_created_at,
            "response": "",
            "done": True,
        }
        done_reason = native_finish_reason(final_reason)
        if done_reason:
            final["done_reason"] = done_reason
        final.update(metrics_from_response(final_obj, started_ns))
        yield ndjson(final)
    except (RuntimeError, httpx.HTTPError) as e:
        yield _ndjson_error(str(e))
    finally:
        aclose = getattr(lines, "aclose", None)
        try:
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
        finally:
            await MODEL_MANAGER.finish_job(job)


async def generate_via_chat_stream(
    lines: AsyncIterator[str], started_ns: int, response_model: str, job: ModelJob
) -> AsyncIterator[bytes]:
    final_obj: dict[str, Any] = {}
    final_reason: str | None = None
    final_created_at = iso_now()
    tool_acc: dict[int, dict[str, Any]] = {}
    try:
        async for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise BackendStreamError(
                    f"llama-server emitted malformed JSON: {e}: {line!r}"
                ) from e
            if not isinstance(obj, dict):
                raise BackendStreamError("llama-server emitted a non-object JSON event")

            if obj.get("error") is not None:
                yield _ndjson_error(backend_error_message(obj))
                return

            if obj.get("created") is not None:
                final_created_at = iso_from_epoch(obj.get("created"))
            if obj.get("usage") is not None:
                final_obj["usage"] = obj["usage"]
            if obj.get("timings") is not None:
                final_obj["timings"] = obj["timings"]

            choices = obj.get("choices")
            if choices is None:
                continue
            if not isinstance(choices, list):
                raise BackendStreamError("llama-server 'choices' was not an array")
            if not choices:
                continue
            if len(choices) != 1:
                raise BackendStreamError(
                    f"llama-server returned {len(choices)} choices; Ollama generate supports one response"
                )
            choice = choices[0]
            if not isinstance(choice, dict):
                raise BackendStreamError("llama-server choice was not an object")
            choice_index = choice.get("index")
            if choice_index is not None and (
                not isinstance(choice_index, int)
                or isinstance(choice_index, bool)
                or choice_index != 0
            ):
                raise BackendStreamError(
                    f"llama-server returned unexpected choice index {choice_index!r}"
                )
            finish_value = choice.get("finish_reason")
            if finish_value is not None:
                if not isinstance(finish_value, str):
                    raise BackendStreamError("llama-server finish_reason was not a string")
                if final_reason is not None and finish_value != final_reason:
                    raise BackendStreamError(
                        f"llama-server changed finish_reason from {final_reason!r} to {finish_value!r}"
                    )
                final_reason = finish_value

            delta_value = choice.get("delta")
            if delta_value is None:
                delta: dict[str, Any] = {}
            elif isinstance(delta_value, dict):
                delta = delta_value
            else:
                raise BackendStreamError("llama-server choice.delta was not an object")

            text_value = delta.get("content")
            if text_value is None:
                text = ""
            elif isinstance(text_value, str):
                text = text_value
            else:
                raise BackendStreamError("llama-server delta.content was not a string")

            reasoning_content_value = delta.get("reasoning_content")
            reasoning_value = delta.get("reasoning")
            if (
                reasoning_content_value is not None
                and reasoning_value is not None
                and reasoning_content_value != reasoning_value
            ):
                raise BackendStreamError(
                    "llama-server emitted conflicting reasoning_content and reasoning deltas"
                )
            thinking_value = (
                reasoning_content_value
                if reasoning_content_value is not None
                else reasoning_value
            )
            if thinking_value is None:
                thinking = ""
            elif isinstance(thinking_value, str):
                thinking = thinking_value
            else:
                raise BackendStreamError("llama-server reasoning delta was not a string")

            if "tool_calls" in delta and delta["tool_calls"] is not None:
                calls = delta["tool_calls"]
                if not isinstance(calls, list):
                    raise BackendStreamError("llama-server tool_calls delta was not an array")
                merge_tool_delta(tool_acc, calls)

            logprobs = ollama_logprobs(choice)
            if text or thinking or logprobs:
                item: dict[str, Any] = {
                    "model": response_model,
                    "created_at": final_created_at,
                    "response": text,
                    "done": False,
                }
                if thinking:
                    item["thinking"] = thinking
                if logprobs:
                    item["logprobs"] = logprobs
                yield ndjson(item)

        if final_reason is None:
            raise BackendStreamError(
                "llama-server chat stream ended without a terminal finish_reason"
            )
        if final_reason in {"tool_calls", "function_call"} and not tool_acc:
            raise BackendStreamError(
                f"llama-server finished with {final_reason!r} but emitted no tool call"
            )
        if tool_acc and final_reason not in {"tool_calls", "function_call"}:
            raise BackendStreamError(
                f"llama-server emitted tool calls but finished with {final_reason!r}"
            )

        if tool_acc:
            calls = [tool_acc[i] for i in sorted(tool_acc)]
            yield ndjson({
                "model": response_model,
                "created_at": final_created_at,
                "response": "",
                "tool_calls": ollama_tool_calls(calls),
                "done": False,
            })

        final: dict[str, Any] = {
            "model": response_model,
            "created_at": final_created_at,
            "response": "",
            "done": True,
        }
        done_reason = finish_reason(final_reason)
        if done_reason:
            final["done_reason"] = done_reason
        final.update(metrics_from_response(final_obj, started_ns))
        yield ndjson(final)
    except (RuntimeError, httpx.HTTPError) as e:
        yield _ndjson_error(str(e))
    finally:
        aclose = getattr(lines, "aclose", None)
        try:
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
        finally:
            await MODEL_MANAGER.finish_job(job)


@app.post("/api/generate")
async def generate(request: Request):
    body, body_error = await read_json_object(request)
    if body_error is not None:
        return body_error
    assert body is not None
    started_ns = time.perf_counter_ns()
    if "model" in body and body["model"] is not None and not isinstance(body["model"], str):
        return _proxy_error_response("'model' must be a string", 400)
    prompt = body.get("prompt") or ""

    try:
        requested_name = resolve_internal_model(request, body.get("model"))
        requested_cfg = MODEL_MANAGER.models[requested_name]
    except KeyError as e:
        return _proxy_error_response(str(e.args[0]), 404)
    ro_error = validate_read_only_overrides(body, requested_cfg, "generate")
    if ro_error:
        return _proxy_error_response(ro_error, 400)
    _warn_non_native_controls(body)

    is_preload = not prompt
    is_infill = isinstance(body.get("suffix"), str) and body["suffix"] != ""
    if is_preload:
        checkpoint_mode = "preload"
    elif is_infill:
        checkpoint_mode = "infill"
    elif body.get("raw"):
        checkpoint_mode = "completion"
    else:
        checkpoint_mode = "chat"
    preemptible, preempt_reason = checkpoint_request_supported_by_proxy(
        body, checkpoint_mode
    )

    try:
        job, cfg = await MODEL_MANAGER.acquire_job(
            requested_name, preemptible=preemptible
        )
        job.preempt_disabled_reason = preempt_reason
        model_name = job.model_name
        response_model = (
            body["model"]
            if isinstance(body.get("model"), str) and body.get("model")
            else model_name
        )
    except KeyError as e:
        return _proxy_error_response(str(e.args[0]), 404)
    except RuntimeError as e:
        return _proxy_error_response(str(e), 502)

    streaming = True if body.get("stream") is None else body["stream"]
    stream_handed_off = False
    try:
        # Ollama uses an empty generate request for preload/unload operations.
        # This proxy intentionally keeps the selected model resident until a
        # different configured model is requested or the proxy exits.
        if is_preload:
            try:
                await backend_health()
            except RuntimeError as e:
                return _proxy_error_response(str(e), 502)
            return JSONResponse({
                "model": response_model,
                "created_at": iso_now(),
                "response": "",
                "done": True,
                "done_reason": "load",
            })

        try:
            m = await backend_model()
            props = await backend_props()
        except RuntimeError as e:
            return _proxy_error_response(str(e), 502)
        cache_backend_metadata(model_name, m, props)
        await MODEL_MANAGER.claim_backend_slot(job, _backend_total_slots(props))
        backend_id = backend_model_id(m, model_name)

        if body.get("raw") or is_infill:
            if body.get("_debug_render_only") is True:
                if is_infill:
                    return _proxy_error_response(
                        "_debug_render_only with suffix/FIM cannot be reproduced exactly: "
                        "Ollama renders its model Go template before FIM insertion, while "
                        "llama-server /infill exposes no equivalent render-only endpoint",
                        400,
                    )
                return _debug_generate_response(
                    response_model, prompt, len(body.get("images") or [])
                )

            if is_infill:
                payload: dict[str, Any] = {
                    "model": backend_id,
                    "input_prefix": prompt,
                    "input_suffix": body["suffix"],
                    # In llama-server this is a generated-middle primer, not
                    # Ollama's prefix prompt.
                    "prompt": "",
                    "stream": True,
                }
                backend_endpoint = "/infill"
                option_endpoint = "native"
            else:
                payload = {
                    "model": backend_id,
                    "prompt": prompt,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                backend_endpoint = "/v1/completions"
                option_endpoint = "completion"
            try:
                payload.update(
                    translate_options(
                        body,
                        cfg.generation_defaults,
                        cfg.think_default,
                        cfg.force_options,
                        cfg.think_force,
                        cfg.reasoning_effort_default,
                        cfg.reasoning_effort_force,
                        cfg.reasoning_preserve_default,
                        cfg.reasoning_preserve_force,
                        endpoint=option_endpoint,
                    )
                )
            except ValueError as e:
                return _proxy_error_response(str(e), 400)

            try:
                resp = await _open_initial_checkpoint_stream(
                    job, backend_endpoint, payload
                )
            except (httpx.HTTPError, RuntimeError) as e:
                return _proxy_error_response(f"llama-server: {e}", 502)
            if resp.status_code >= 400:
                return await backend_error_response(resp)

            lines = buffered_backend_lines(
                checkpoint_backend_lines(
                    job,
                    backend_endpoint,
                    payload,
                    initial_response=resp,
                ),
                job,
            )
            stream_handler = (
                native_completion_stream if is_infill else completion_stream
            )
            source = stream_handler(lines, started_ns, response_model, job)
            if streaming:
                stream_handed_off = True
                return StreamingResponse(
                    source,
                    media_type="application/x-ndjson",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "X-Ollama-Proxy-Job-ID": str(job.id),
                    },
                )
            return await _collect_generate_response(source)

        # Normal generate: let llama-server's model chat template do formatting.
        messages: list[dict[str, Any]] = []
        if body.get("system"):
            messages.append({"role": "system", "content": body["system"]})
        user_message: dict[str, Any] = {"role": "user", "content": prompt}
        if body.get("images"):
            user_message["images"] = body["images"]
        messages.append(user_message)
        try:
            messages = apply_system_prompt(
                translate_messages(messages),
                cfg.system_prompt,
                cfg.system_prompt_force,
            )

            payload = {
                "model": backend_id,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            payload.update(
                translate_options(
                    body,
                    cfg.generation_defaults,
                    cfg.think_default,
                    cfg.force_options,
                    cfg.think_force,
                    cfg.reasoning_effort_default,
                    cfg.reasoning_effort_force,
                    cfg.reasoning_preserve_default,
                    cfg.reasoning_preserve_force,
                    endpoint="chat",
                )
            )
        except ValueError as e:
            return _proxy_error_response(str(e), 400)

        try:
            resp = await _open_initial_checkpoint_stream(
                job, "/v1/chat/completions", payload
            )
        except (httpx.HTTPError, RuntimeError) as e:
            return _proxy_error_response(f"llama-server: {e}", 502)
        if resp.status_code >= 400:
            return await backend_error_response(resp)

        lines = buffered_backend_lines(
            checkpoint_backend_lines(
                job,
                "/v1/chat/completions",
                payload,
                initial_response=resp,
            ),
            job,
        )
        source = generate_via_chat_stream(lines, started_ns, response_model, job)
        if streaming:
            stream_handed_off = True
            return StreamingResponse(
                source,
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Ollama-Proxy-Job-ID": str(job.id),
                },
            )
        return await _collect_generate_response(source)
    finally:
        if not stream_handed_off:
            await MODEL_MANAGER.finish_job(job)


_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _filter_end_to_end_raw_headers(
    raw_headers: list[tuple[bytes, bytes]],
    *,
    drop_host: bool = False,
    drop_content_length: bool = False,
) -> list[tuple[bytes, bytes]]:
    """Preserve duplicate/end-to-end headers while removing hop-by-hop fields.

    RFC Connection tokens name additional hop-by-hop headers, so filtering only
    the fixed header names is insufficient for a transparent proxy hop.
    """
    connection_tokens: set[bytes] = set()
    for key, value in raw_headers:
        if key.lower() == b"connection":
            connection_tokens.update(
                token.strip().lower()
                for token in value.split(b",")
                if token.strip()
            )

    blocked = {name.encode("ascii") for name in _HOP_BY_HOP_HEADERS}
    blocked.update(connection_tokens)
    if drop_host:
        blocked.add(b"host")
    if drop_content_length:
        blocked.add(b"content-length")

    return [
        (key, value)
        for key, value in raw_headers
        if key.lower() not in blocked
    ]


async def _wait_for_public_disconnect(request: Request) -> None:
    while True:
        if await request.is_disconnected():
            return
        await asyncio.sleep(0.05)


async def proxy_user_generation_request(request: Request):
    """Public user-bound side for /api/chat and /api/generate.

    Register the user before entering external-chain so the scheduler can begin
    loading/preempting the desired model while filters are still processing the
    outer request. The response remains a byte-for-byte streaming HTTP hop; the
    external programs may issue any number of internal Ollama requests through
    their model-bound ingress during this one public request.
    """
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except ValueError as e:
        return _proxy_error_response(f"request body must be valid JSON: {e}", 400)
    if not isinstance(body, dict):
        return _proxy_error_response("request body must be a JSON object", 400)

    requested_model = body.get("model")
    if not isinstance(requested_model, str) or not requested_model:
        return _proxy_error_response("model is required and must be a non-empty string", 400)

    try:
        model_name = MODEL_MANAGER.resolve_name(requested_model)
    except KeyError as e:
        return _proxy_error_response(str(e.args[0]), 404)

    await CHAIN_MANAGER.user_request_started(model_name)
    released = False

    async def release_once() -> None:
        nonlocal released
        if released:
            return
        released = True
        await CHAIN_MANAGER.user_request_finished(model_name)

    target = CHAIN_MANAGER.entry_url(model_name) + request.url.path
    if request.url.query:
        target += "?" + request.url.query

    request_headers = _filter_end_to_end_raw_headers(
        list(request.scope.get("headers") or []),
        drop_host=True,
        drop_content_length=True,
    )

    upstream_request = gateway_client.build_request(
        request.method,
        target,
        content=raw_body,
        headers=request_headers,
    )
    open_task = asyncio.create_task(
        gateway_client.send(upstream_request, stream=True),
        name=f"external-chain-open-{model_name}",
    )
    disconnect_task = asyncio.create_task(
        _wait_for_public_disconnect(request),
        name=f"public-disconnect-{model_name}",
    )

    try:
        done, _pending = await asyncio.wait(
            {open_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if disconnect_task in done and open_task not in done:
            open_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError):
                await open_task
            await release_once()
            return PlainTextResponse("client disconnected\n", status_code=499)

        disconnect_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await disconnect_task
        upstream = open_task.result()
    except (httpx.HTTPError, RuntimeError) as e:
        if not open_task.done():
            open_task.cancel()
        if not disconnect_task.done():
            disconnect_task.cancel()
        await release_once()
        return _proxy_error_response(
            f"external-chain for model {model_name!r}: {e}", 502
        )
    except BaseException:
        if not open_task.done():
            open_task.cancel()
        if not disconnect_task.done():
            disconnect_task.cancel()
        await release_once()
        raise

    response_headers = _filter_end_to_end_raw_headers(list(upstream.headers.raw))

    async def stream_upstream() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        except httpx.HTTPError as e:
            # Response headers have already been sent at this point, so propagating
            # an upstream reset/read error through StreamingResponse can only tear
            # down the downstream transport. Treat it as an interrupted upstream
            # stream: log the failure and finish the client response with clean EOF.
            print(
                f"WARNING: external-chain stream for model {model_name!r} ended "
                f"abnormally: {e}; closing downstream stream cleanly",
                flush=True,
            )
            return
        finally:
            await upstream.aclose()
            await release_once()

    response = StreamingResponse(
        stream_upstream(),
        status_code=upstream.status_code,
        media_type=None,
    )
    # Starlette's mapping-style `headers=` path collapses duplicate headers.
    # Assign raw headers directly so Set-Cookie and other repeatable end-to-end
    # fields survive the external-chain hop unchanged.
    response.raw_headers = response_headers
    return response


async def public_chat(request: Request):
    return await proxy_user_generation_request(request)


async def public_generate(request: Request):
    return await proxy_user_generation_request(request)


# Public metadata stays outside external-chain. Only generation endpoints own a
# user lease and therefore participate in scheduler activity/residency.
public_app.add_api_route("/", root, methods=["GET", "HEAD"])
public_app.add_api_route("/api/version", version, methods=["GET", "HEAD"])
public_app.add_api_route("/api/tags", tags, methods=["GET", "HEAD"])
public_app.add_api_route("/api/ps", ps, methods=["GET"])
public_app.add_api_route("/api/show", show, methods=["POST"])
public_app.add_api_route("/api/chat", public_chat, methods=["POST"])
public_app.add_api_route("/api/generate", public_generate, methods=["POST"])


def parse_listen(value: str) -> tuple[str, int]:
    """Parse ADDR:PORT, including bracketed IPv6 such as [::1]:11435."""
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("listen address must not be empty")

    if value.startswith("["):
        end = value.find("]")
        if end < 0 or end + 1 >= len(value) or value[end + 1] != ":":
            raise argparse.ArgumentTypeError("IPv6 listen address must use [ADDR]:PORT")
        host = value[1:end]
        port_text = value[end + 2:]
    else:
        if ":" not in value:
            raise argparse.ArgumentTypeError("listen address must be ADDR:PORT")
        host, port_text = value.rsplit(":", 1)

    if not host:
        raise argparse.ArgumentTypeError("listen address must not be empty")
    try:
        port = int(port_text)
    except ValueError as e:
        raise argparse.ArgumentTypeError("listen port must be an integer") from e
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("listen port must be between 1 and 65535")
    return host, port


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Ollama API -> managed llama-server compatibility shim")
    parser.add_argument(
        "-l", "--listen",
        dest="listen",
        metavar="ADDR:PORT",
        type=parse_listen,
        default=("127.0.0.1", 11435),
        help="listen address and port (default: 127.0.0.1:11435)",
    )
    parser.add_argument(
        "-c", "--config",
        dest="config_file",
        metavar="FILE",
        required=True,
        help="YAML model configuration file",
    )
    parser.add_argument(
        "--full-llama-responses-log",
        dest="full_llama_responses_log",
        nargs="?",
        const="full-llama-responses.log",
        default=None,
        metavar="FILE",
        help=(
            "append raw llama-server generation response-body bytes before any "
            "SSE/UTF-8/JSON parsing; with no FILE use ./full-llama-responses.log"
        ),
    )
    args = parser.parse_args()

    PROXY_HOST, PROXY_PORT = args.listen
    try:
        models = load_models_config(args.config_file)
    except ValueError as e:
        parser.error(str(e))

    try:
        _open_full_llama_responses_log(args.full_llama_responses_log)
    except RuntimeError as e:
        parser.error(str(e))

    try:
        MODEL_MANAGER.configure(models)
        print(f"Loaded {len(models)} model(s) from {Path(args.config_file).expanduser()}", flush=True)
        for model_name, cfg in models.items():
            alias_text = f" (aliases: {', '.join(cfg.aliases)})" if cfg.aliases else ""
            print(f"  {model_name}{alias_text} prio={cfg.priority}", flush=True)
        if FULL_LLAMA_RESPONSES_LOG is not None:
            print(
                f"Raw llama generation response logging enabled: "
                f"{FULL_LLAMA_RESPONSES_LOG}",
                flush=True,
            )

        uvicorn.run(public_app, host=PROXY_HOST, port=PROXY_PORT, log_level="info")
    finally:
        _close_full_llama_responses_log()

