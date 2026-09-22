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
Ollama/OpenAI embedding API -> dynamically managed llama-server shim.

This is intentionally much smaller than ollama_proxy_llama.py.  Each model has
only two configuration fields:

    embeddinggemma:latest:
      alias:
        - embeddinggemma
        - embed
      cmd: >-
        llama-server
        -m /models/embeddinggemma-300M-Q8_0.gguf
        --embedding
        --host 127.0.0.1 --port 8080

    bge-m3:latest:
      alias: bge-m3
      cmd: >-
        llama-server
        -m /models/bge-m3-Q8_0.gguf
        --embedding
        --pooling cls
        --host 127.0.0.1 --port 8080

The YAML path is required and is passed with -c/--config.

The managed llama-server command for every model MUST listen on
127.0.0.1:8080.  Only one configured model process is owned by this proxy at a
time.  Requests for the currently loaded model can run concurrently.  A request
for another model waits for active requests to finish, terminates the current
process, and starts the requested model.

There is deliberately NO KV-cache save/restore, NO priority field, NO idle
stop, and NO per-model generation configuration.

Exposed embedding endpoints:

    POST /api/embed          Ollama current embedding API
    POST /api/embeddings     Ollama legacy embedding API
    POST /v1/embeddings      OpenAI-compatible embedding API

Small Ollama discovery endpoints are also provided:

    GET  /api/tags
    GET  /api/ps
    POST /api/show
    GET  /api/version

Notes:
  * llama-server must be started with --embedding/--embeddings for embedding
    models.
  * llama-server's /v1/embeddings returns L2-normalized vectors.
  * `dimensions` is implemented in the proxy by taking the requested prefix of
    each vector and L2-normalizing it again.  This is appropriate for models
    trained for Matryoshka/variable-dimensional embeddings; using it on a model
    not trained for that purpose is mathematically valid but may reduce quality.
  * Ollama's `keep_alive`, `options`, and `truncate` are accepted for client
    compatibility.  Process lifetime is owned by this proxy, so keep_alive is
    intentionally ignored.  llama-server decides whether an overlong input is
    accepted; this shim does not perform model-specific token truncation.

Dependencies:
    Python 3.11+
    pip install fastapi 'uvicorn[standard]' httpx pyyaml
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse


LLAMA_SERVER_URL = "http://127.0.0.1:8081"
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 11435
BACKEND_START_TIMEOUT = 300.0
CONFIG_MODIFIED_AT: str | None = None


class StrictConfigLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate keys and uses YAML 1.2 booleans."""


# Avoid PyYAML YAML-1.1 surprises such as parsing "on"/"off" as booleans.
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


def _construct_unique_mapping(
    loader: StrictConfigLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
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
    cmd: str


def load_models_config(path: str | Path) -> dict[str, ModelConfig]:
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.load(f, Loader=StrictConfigLoader)
    except (OSError, yaml.YAMLError) as e:
        raise ValueError(f"cannot read models config {str(config_path)!r}: {e}") from e

    if not isinstance(raw, dict) or not raw:
        raise ValueError("models config must be a non-empty YAML mapping")

    models: dict[str, ModelConfig] = {}
    allowed_fields = {"cmd", "alias"}

    for name, table in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("model names must be non-empty strings")
        if name != name.strip():
            raise ValueError(f"model name {name!r} must not have leading/trailing whitespace")
        if not isinstance(table, dict):
            raise ValueError(f"model {name!r} must be a YAML mapping")

        unexpected = [str(key) for key in table if key not in allowed_fields]
        if unexpected:
            raise ValueError(
                f"model {name!r}: unexpected field(s): " + ", ".join(sorted(unexpected))
            )

        cmd = table.get("cmd")
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError(f"model {name!r}: 'cmd' must be a non-empty string")
        cmd += " --port 8081"

        alias_value = table.get("alias", [])
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

        aliases: list[str] = []
        seen: set[str] = set()
        for alias in alias_items:
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError(f"model {name!r}: every alias must be a non-empty string")
            if alias != alias.strip():
                raise ValueError(
                    f"model {name!r}: alias {alias!r} must not have leading/trailing whitespace"
                )
            if alias == name:
                raise ValueError(
                    f"model {name!r}: alias {alias!r} duplicates the canonical model name"
                )
            if alias in seen:
                raise ValueError(f"model {name!r}: duplicate alias {alias!r}")
            seen.add(alias)
            aliases.append(alias)

        models[name] = ModelConfig(name=name, aliases=tuple(aliases), cmd=cmd.strip())

    alias_owners: dict[str, str] = {}
    for model_name, cfg in models.items():
        for alias in cfg.aliases:
            if alias in models:
                raise ValueError(
                    f"model {model_name!r}: alias {alias!r} conflicts with canonical model name"
                )
            previous = alias_owners.get(alias)
            if previous is not None:
                raise ValueError(
                    f"model {model_name!r}: alias {alias!r} is already used by model {previous!r}"
                )
            alias_owners[alias] = model_name

    return models


app = FastAPI(title="Ollama/OpenAI -> managed llama-server embeddings shim")

client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=60.0),
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
)


class ModelManager:
    def __init__(self) -> None:
        self.models: dict[str, ModelConfig] = {}
        self.aliases: dict[str, str] = {}
        self.active_name: str | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.active_requests = 0
        self.switching = False
        self.condition = asyncio.Condition()
        self.metadata_cache: dict[str, dict[str, Any]] = {}

    def configure(self, models: dict[str, ModelConfig]) -> None:
        self.models = models
        self.aliases = {
            alias: name
            for name, cfg in models.items()
            for alias in cfg.aliases
        }

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

        # Match Ollama's common :latest naming convenience in both directions.
        if name.endswith(":latest"):
            resolved = self._resolve_exact_name(name[:-7])
            if resolved is not None:
                return resolved
        elif ":" not in name:
            resolved = self._resolve_exact_name(name + ":latest")
            if resolved is not None:
                return resolved

        raise KeyError(f"model {name!r} not found")

    def running_name(self) -> str | None:
        if self.process is None or self.process.returncode is not None:
            return None
        return self.active_name

    async def _wait_backend_down(self, timeout: float = 10.0) -> None:
        """Wait until the managed listen address stops answering.

        The configured command is launched through /bin/sh.  A child server can
        remain alive briefly after the shell itself exits (for example while
        handling SIGTERM gracefully), so proc.wait() alone is not enough to
        make an immediate model switch reliable.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            try:
                await asyncio.wait_for(
                    client.get(f"{LLAMA_SERVER_URL}/health"), timeout=0.5
                )
            except (httpx.HTTPError, asyncio.TimeoutError):
                return
            if asyncio.get_running_loop().time() >= deadline:
                return
            await asyncio.sleep(0.1)

    async def _stop_current(self) -> None:
        proc = self.process
        old_name = self.active_name
        self.process = None
        self.active_name = None

        if proc is None or proc.returncode is not None:
            return

        print(f"Stopping embedding model {old_name!r} (pid {proc.pid})", flush=True)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            await self._wait_backend_down()
            return

        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            print(
                f"Embedding model {old_name!r} did not stop after SIGTERM; sending SIGKILL",
                flush=True,
            )
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()

        await self._wait_backend_down()

    async def _start(self, cfg: ModelConfig) -> asyncio.subprocess.Process:
        # Prevent accidentally treating an unrelated process on :8080 as the
        # model we just launched.
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

        print(f"Starting embedding model {cfg.name!r}: {cfg.cmd}", flush=True)
        proc = await asyncio.create_subprocess_shell(
            cfg.cmd,
            start_new_session=True,
        )

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
                    print(f"Embedding model {cfg.name!r} is ready", flush=True)
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

    async def acquire(self, requested: Any = None) -> tuple[str, ModelConfig]:
        name = self.resolve_name(requested)
        cfg = self.models[name]

        while True:
            async with self.condition:
                while self.switching:
                    await self.condition.wait()

                if (
                    self.active_name == name
                    and self.process is not None
                    and self.process.returncode is None
                ):
                    self.active_requests += 1
                    return name, cfg

                if self.active_requests > 0:
                    await self.condition.wait()
                    continue

                self.switching = True
                break

        try:
            await self._stop_current()
            proc = await self._start(cfg)
        except Exception:
            async with self.condition:
                self.switching = False
                self.condition.notify_all()
            raise

        async with self.condition:
            self.active_name = name
            self.process = proc
            self.active_requests = 1
            self.switching = False
            self.condition.notify_all()

        return name, cfg

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
            if self.active_requests == 0:
                self.condition.notify_all()

    async def shutdown(self) -> None:
        # Shutdown is allowed to run only when the application itself exits.
        async with self.condition:
            while self.switching or self.active_requests > 0:
                await self.condition.wait()
            self.switching = True
        try:
            await self._stop_current()
        finally:
            async with self.condition:
                self.switching = False
                self.condition.notify_all()


MODEL_MANAGER = ModelManager()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await MODEL_MANAGER.shutdown()
    await client.aclose()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def iso_from_epoch(epoch: int | float | None) -> str:
    if epoch is None:
        return iso_now()
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def model_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def read_json_object(
    request: Request,
) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    try:
        body = await request.json()
    except ValueError as e:
        return None, JSONResponse(
            {"error": f"request body must be valid JSON: {e}"}, status_code=400
        )
    if not isinstance(body, dict):
        return None, JSONResponse(
            {"error": "request body must be a JSON object"}, status_code=400
        )
    return body, None


def ollama_error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def openai_error(message: str, status_code: int, error_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
        status_code=status_code,
    )


def backend_error_message(data: Any) -> str:
    error = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(error, dict):
        return str(error.get("message") or json.dumps(error, ensure_ascii=False))
    return str(error)


async def backend_error(resp: httpx.Response) -> tuple[str, int]:
    try:
        data = resp.json()
        message = backend_error_message(data)
    except Exception:
        message = resp.text
    return f"llama-server: {message}", resp.status_code


async def backend_model() -> dict[str, Any]:
    try:
        r = await client.get(f"{LLAMA_SERVER_URL}/v1/models")
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            raise RuntimeError("llama-server returned no loaded models")
        model = data[0]
        if not isinstance(model, dict):
            raise RuntimeError("llama-server returned invalid model metadata")
        return model
    except (httpx.HTTPError, ValueError, RuntimeError) as e:
        raise RuntimeError(f"cannot query llama-server at {LLAMA_SERVER_URL}: {e}") from e


async def backend_props() -> dict[str, Any]:
    try:
        r = await client.get(f"{LLAMA_SERVER_URL}/props")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else {}
    except (httpx.HTTPError, ValueError):
        return {}


def cache_backend_metadata(name: str, model: dict[str, Any], props: dict[str, Any]) -> None:
    MODEL_MANAGER.metadata_cache[name] = {"model": model, "props": props}


def _embedding_length(model: dict[str, Any]) -> int:
    meta = model.get("meta") or {}
    for key in ("n_embd", "n_embd_out", "embedding_length"):
        value = meta.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return 0


def _basic_details(model: dict[str, Any]) -> dict[str, Any]:
    meta = model.get("meta") or {}
    family = str(meta.get("architecture") or meta.get("general.architecture") or "unknown")
    return {
        "parent_model": "",
        "format": "gguf",
        "family": family,
        "families": [] if family == "unknown" else [family],
        "parameter_size": "unknown",
        "quantization_level": str(meta.get("quantization") or "unknown"),
    }


def _validate_text_input(value: Any, field: str = "input") -> tuple[list[str] | None, str | None]:
    if isinstance(value, str):
        return [value], None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value, None
    return None, f"{field} must be a string or an array of strings"


def _normalize_dimension(value: Any) -> tuple[int | None, str | None]:
    if value is None:
        return None, None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None, "dimensions must be a positive integer"
    return value, None


def _resize_and_normalize(vector: list[Any], dimensions: int | None) -> list[float]:
    try:
        floats = [float(x) for x in vector]
    except (TypeError, ValueError) as e:
        raise ValueError("backend returned a non-numeric embedding") from e

    if dimensions is None:
        return floats
    if dimensions > len(floats):
        raise ValueError(
            f"requested dimensions ({dimensions}) exceed model embedding size ({len(floats)})"
        )

    result = floats[:dimensions]
    norm = math.sqrt(sum(value * value for value in result))
    if norm > 0.0:
        result = [value / norm for value in result]
    return result


async def _call_llama_embeddings(
    backend_model_id: str,
    inputs: Any,
    *,
    encoding_format: str = "float",
    extra: dict[str, Any] | None = None,
) -> httpx.Response:
    payload: dict[str, Any] = {
        "model": backend_model_id,
        "input": inputs,
        "encoding_format": encoding_format,
    }
    if extra:
        payload.update(extra)
    return await client.post(f"{LLAMA_SERVER_URL}/v1/embeddings", json=payload)


async def _acquire_and_identify_model(requested: Any) -> tuple[str, ModelConfig, dict[str, Any]]:
    model_name, cfg = await MODEL_MANAGER.acquire(requested)
    try:
        model = await backend_model()
        props = await backend_props()
        cache_backend_metadata(model_name, model, props)
        return model_name, cfg, model
    except Exception:
        await MODEL_MANAGER.release(model_name)
        raise


@app.api_route("/", methods=["GET", "HEAD"])
async def root() -> PlainTextResponse:
    return PlainTextResponse("Ollama is running (llama-server embeddings shim)\n")


@app.api_route("/api/version", methods=["GET", "HEAD"])
async def version() -> dict[str, str]:
    return {"version": "0.0.0-llama-server-embeddings-shim"}


@app.get("/v1/models")
async def v1_models() -> JSONResponse:
    data: list[dict[str, Any]] = []
    created = 0
    if CONFIG_MODIFIED_AT:
        try:
            created = int(
                datetime.fromisoformat(CONFIG_MODIFIED_AT.replace("Z", "+00:00")).timestamp()
            )
        except ValueError:
            created = 0

    for name, cfg in MODEL_MANAGER.models.items():
        for visible_name in (name, *cfg.aliases):
            data.append(
                {
                    "id": visible_name,
                    "object": "model",
                    "created": created,
                    "owned_by": "llama.cpp",
                }
            )
    return JSONResponse({"object": "list", "data": data})


@app.api_route("/api/tags", methods=["GET", "HEAD"])
async def tags() -> JSONResponse:
    models: list[dict[str, Any]] = []
    for name, cfg in MODEL_MANAGER.models.items():
        cached = MODEL_MANAGER.metadata_cache.get(name) or {}
        model = cached.get("model") or {}
        meta = model.get("meta") or {}
        size = int(meta.get("size") or 0)
        digest = model_digest(str(model.get("id", name)))
        modified_at = CONFIG_MODIFIED_AT or "1970-01-01T00:00:00Z"
        details = _basic_details(model)

        for visible_name in (name, *cfg.aliases):
            models.append(
                {
                    "name": visible_name,
                    "model": visible_name,
                    "modified_at": modified_at,
                    "size": size,
                    "digest": digest,
                    "details": details,
                }
            )

    return JSONResponse({"models": models})


@app.get("/api/ps")
async def ps() -> JSONResponse:
    lease = await MODEL_MANAGER.acquire_running()
    if lease is None:
        return JSONResponse({"models": []})

    name, _cfg = lease
    try:
        try:
            model = await backend_model()
            props = await backend_props()
            cache_backend_metadata(name, model, props)
        except RuntimeError:
            return JSONResponse({"models": []})

        meta = model.get("meta") or {}
        return JSONResponse(
            {
                "models": [
                    {
                        "name": name,
                        "model": name,
                        "size": int(meta.get("size") or 0),
                        "digest": model_digest(str(model.get("id", name))),
                        "details": _basic_details(model),
                        "expires_at": "9999-12-31T23:59:59Z",
                        "size_vram": int(meta.get("size") or 0),
                    }
                ]
            }
        )
    finally:
        await MODEL_MANAGER.release(name)


@app.post("/api/show")
async def show(request: Request) -> JSONResponse:
    body, body_error = await read_json_object(request)
    if body_error is not None:
        return body_error
    assert body is not None

    requested = body.get("model") or body.get("name")
    try:
        name = MODEL_MANAGER.resolve_name(requested)
    except KeyError as e:
        return ollama_error(str(e.args[0]), 404)

    model: dict[str, Any] = {}
    props: dict[str, Any] = {}

    # Metadata lookup never starts or switches a model.
    lease = await MODEL_MANAGER.acquire_running()
    if lease is not None:
        running_name, _cfg = lease
        try:
            if running_name == name:
                try:
                    model = await backend_model()
                    props = await backend_props()
                    cache_backend_metadata(name, model, props)
                except RuntimeError:
                    pass
        finally:
            await MODEL_MANAGER.release(running_name)

    if not model:
        cached = MODEL_MANAGER.metadata_cache.get(name) or {}
        model = cached.get("model") or {}
        props = cached.get("props") or {}

    info: dict[str, Any] = {}
    embd_len = _embedding_length(model)
    if embd_len:
        info["embedding_length"] = embd_len
    if props.get("model_path"):
        info["llama-server.model_path"] = str(props["model_path"])

    return JSONResponse(
        {
            "license": "",
            "modelfile": "",
            "parameters": "",
            "template": "",
            "system": "",
            "details": _basic_details(model),
            "model_info": info,
            "capabilities": ["embedding"],
            "modified_at": CONFIG_MODIFIED_AT or "1970-01-01T00:00:00Z",
        }
    )


@app.post("/api/embed")
async def api_embed(request: Request) -> JSONResponse:
    started_ns = time.perf_counter_ns()
    body, body_error = await read_json_object(request)
    if body_error is not None:
        return body_error
    assert body is not None

    inputs, input_error = _validate_text_input(body.get("input"))
    if input_error:
        return ollama_error(input_error, 400)
    assert inputs is not None

    dimensions, dimension_error = _normalize_dimension(body.get("dimensions"))
    if dimension_error:
        return ollama_error(dimension_error, 400)

    if "truncate" in body and not isinstance(body["truncate"], bool):
        return ollama_error("truncate must be a boolean", 400)

    # keep_alive/options are intentionally accepted but ignored.  The model
    # process lifecycle and all llama-server runtime parameters live in `cmd`.
    requested_model = body.get("model")
    try:
        model_name, _cfg, model = await _acquire_and_identify_model(requested_model)
    except KeyError as e:
        return ollama_error(str(e.args[0]), 404)
    except RuntimeError as e:
        return ollama_error(str(e), 502)

    try:
        backend_id = str(model.get("id") or model_name)
        try:
            resp = await _call_llama_embeddings(backend_id, inputs)
        except httpx.HTTPError as e:
            return ollama_error(f"cannot call llama-server: {e}", 502)

        if resp.status_code >= 400:
            message, status = await backend_error(resp)
            return ollama_error(message, status)

        try:
            data = resp.json()
            rows = data.get("data")
            if not isinstance(rows, list):
                raise ValueError("backend response has no data array")
            rows = sorted(rows, key=lambda item: int(item.get("index", 0)))
            embeddings = [
                _resize_and_normalize(item.get("embedding"), dimensions)
                for item in rows
            ]
        except ValueError as e:
            if str(e).startswith("requested dimensions"):
                return ollama_error(str(e), 400)
            return ollama_error(f"invalid llama-server embeddings response: {e}", 502)
        except (TypeError, AttributeError) as e:
            return ollama_error(f"invalid llama-server embeddings response: {e}", 502)

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        result: dict[str, Any] = {
            "model": str(requested_model or model_name),
            "embeddings": embeddings,
            "total_duration": time.perf_counter_ns() - started_ns,
        }
        if isinstance(prompt_tokens, int):
            result["prompt_eval_count"] = prompt_tokens
        return JSONResponse(result)
    finally:
        await MODEL_MANAGER.release(model_name)


@app.post("/api/embeddings")
async def api_embeddings_legacy(request: Request) -> JSONResponse:
    body, body_error = await read_json_object(request)
    if body_error is not None:
        return body_error
    assert body is not None

    prompt = body.get("prompt")
    if not isinstance(prompt, str):
        return ollama_error("prompt must be a string", 400)

    requested_model = body.get("model")
    try:
        model_name, _cfg, model = await _acquire_and_identify_model(requested_model)
    except KeyError as e:
        return ollama_error(str(e.args[0]), 404)
    except RuntimeError as e:
        return ollama_error(str(e), 502)

    try:
        backend_id = str(model.get("id") or model_name)
        try:
            resp = await _call_llama_embeddings(backend_id, prompt)
        except httpx.HTTPError as e:
            return ollama_error(f"cannot call llama-server: {e}", 502)

        if resp.status_code >= 400:
            message, status = await backend_error(resp)
            return ollama_error(message, status)

        try:
            data = resp.json()
            rows = data.get("data") or []
            if not rows:
                raise ValueError("backend response has no embeddings")
            vector = _resize_and_normalize(rows[0].get("embedding"), None)
        except (ValueError, TypeError, AttributeError) as e:
            return ollama_error(f"invalid llama-server embeddings response: {e}", 502)

        return JSONResponse({"embedding": vector})
    finally:
        await MODEL_MANAGER.release(model_name)


@app.post("/v1/embeddings")
async def v1_embeddings(request: Request) -> JSONResponse:
    body, body_error = await read_json_object(request)
    if body_error is not None:
        # Convert the generic Ollama-shaped JSON error into OpenAI shape.
        try:
            msg = json.loads(body_error.body).get("error", "invalid request")
        except Exception:
            msg = "invalid request"
        return openai_error(str(msg), body_error.status_code)
    assert body is not None

    if "input" not in body:
        return openai_error("input is required", 400)

    dimensions, dimension_error = _normalize_dimension(body.get("dimensions"))
    if dimension_error:
        return openai_error(dimension_error, 400)

    encoding_format = body.get("encoding_format", "float")
    if encoding_format not in (None, "float"):
        # Post-processing dimensions requires numeric vectors, and current
        # llama-server embedding output is expected as float arrays here.
        return openai_error("only encoding_format='float' is supported", 400)

    requested_model = body.get("model")
    try:
        model_name, _cfg, model = await _acquire_and_identify_model(requested_model)
    except KeyError as e:
        return openai_error(str(e.args[0]), 404)
    except RuntimeError as e:
        return openai_error(str(e), 502, "server_error")

    try:
        backend_id = str(model.get("id") or model_name)

        # Keep the request intentionally narrow.  We own model routing and
        # dimensions; the rest of the OpenAI-compatible request is just input.
        try:
            resp = await _call_llama_embeddings(
                backend_id,
                body.get("input"),
                encoding_format="float",
            )
        except httpx.HTTPError as e:
            return openai_error(f"cannot call llama-server: {e}", 502, "server_error")

        if resp.status_code >= 400:
            message, status = await backend_error(resp)
            error_type = "server_error" if status >= 500 else "invalid_request_error"
            return openai_error(message, status, error_type)

        try:
            data = resp.json()
            rows = data.get("data")
            if not isinstance(rows, list):
                raise ValueError("backend response has no data array")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("backend data item is not an object")
                row["embedding"] = _resize_and_normalize(row.get("embedding"), dimensions)
            data["model"] = str(requested_model or model_name)
            data.setdefault("object", "list")
            data.setdefault("usage", {"prompt_tokens": 0, "total_tokens": 0})
        except ValueError as e:
            if str(e).startswith("requested dimensions"):
                return openai_error(str(e), 400)
            return openai_error(
                f"invalid llama-server embeddings response: {e}", 502, "server_error"
            )
        except (TypeError, AttributeError) as e:
            return openai_error(
                f"invalid llama-server embeddings response: {e}", 502, "server_error"
            )

        return JSONResponse(data)
    finally:
        await MODEL_MANAGER.release(model_name)


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
        port_text = value[end + 2 :]
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

    parser = argparse.ArgumentParser(
        description="Ollama/OpenAI embeddings API -> managed llama-server shim"
    )
    parser.add_argument(
        "-l",
        "--listen",
        dest="listen",
        metavar="ADDR:PORT",
        type=parse_listen,
        default=("127.0.0.1", 11435),
        help="listen address and port (default: 127.0.0.1:11435)",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        metavar="FILE",
        required=True,
        help="YAML model configuration file",
    )
    args = parser.parse_args()

    PROXY_HOST, PROXY_PORT = args.listen
    try:
        models = load_models_config(args.config_file)
    except ValueError as e:
        parser.error(str(e))

    try:
        CONFIG_MODIFIED_AT = iso_from_epoch(
            Path(args.config_file).expanduser().resolve().stat().st_mtime
        )
    except OSError:
        CONFIG_MODIFIED_AT = "1970-01-01T00:00:00Z"

    MODEL_MANAGER.configure(models)
    print(
        f"Loaded {len(models)} embedding model(s) from {Path(args.config_file).expanduser()}",
        flush=True,
    )
    for model_name, cfg in models.items():
        alias_text = f" (aliases: {', '.join(cfg.aliases)})" if cfg.aliases else ""
        print(f"  {model_name}{alias_text}", flush=True)

    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="info")

