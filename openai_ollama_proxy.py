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
"""OpenAI-compatible HTTP API -> native Ollama API proxy.

Supported OpenAI-compatible endpoints:
  GET  /v1/models
  GET  /v1/models/{model}
  POST /v1/chat/completions
  POST /v1/completions
  POST /v1/embeddings
  POST /v1/responses

The same endpoints are also accepted without the /v1 prefix for clients whose
base URL already points at a versioned path incorrectly.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import struct
import time
import uuid
from datetime import datetime
from typing import Any, AsyncIterator, Iterable

from aiohttp import ClientConnectionError, ClientResponse, ClientSession, ClientTimeout, TCPConnector, web

LOG = logging.getLogger("openai-ollama-proxy")

JSON = dict[str, Any]


class ProxyError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.error_type = error_type
        self.param = param
        self.code = code


def openai_error_payload(
    message: str,
    *,
    error_type: str = "api_error",
    param: str | None = None,
    code: str | None = None,
) -> JSON:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def now_unix() -> int:
    return int(time.time())


def new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex}"


def parse_iso_timestamp(value: Any) -> int:
    if not isinstance(value, str) or not value:
        return 0
    try:
        v = value.replace("Z", "+00:00")
        return int(datetime.fromisoformat(v).timestamp())
    except (ValueError, OverflowError):
        return 0


def get_request_id(request: web.Request) -> str:
    rid = request.headers.get("x-request-id") or request.headers.get("x-correlation-id")
    return rid or new_id("req_")


def openai_usage(ollama: JSON) -> JSON:
    prompt = int(ollama.get("prompt_eval_count") or 0)
    completion = int(ollama.get("eval_count") or 0)
    usage: JSON = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    cached = int(ollama.get("prompt_eval_cached_count") or 0)
    if cached:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage


def completion_usage_sum(items: Iterable[JSON]) -> JSON:
    prompt = sum(int(x.get("prompt_eval_count") or 0) for x in items)
    completion = sum(int(x.get("eval_count") or 0) for x in items)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def finish_reason(ollama: JSON, has_tool_calls: bool = False) -> str:
    if has_tool_calls:
        return "tool_calls"
    reason = str(ollama.get("done_reason") or "stop").lower()
    if reason in {"length", "max_tokens", "limit"}:
        return "length"
    if reason == "tool_calls":
        return "tool_calls"
    return "stop"


def normalize_stop(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    raise ProxyError("'stop' must be a string or an array of strings", param="stop")


def parse_json_arguments(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass
        # Ollama documents tool arguments as an object. Keeping an invalid
        # OpenAI argument string would produce a malformed native request, so
        # preserve it in a single field instead of silently dropping it.
        return {"_raw": value}
    return {"_raw": value}


def stringify_tool_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "{}"
    return json_dumps(value)


def extract_text_content(content: Any, *, allow_images: bool = False) -> tuple[str, list[str]]:
    """Convert OpenAI message content to Ollama text + base64 image list."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        raise ProxyError("Message content must be a string, null, or an array of content parts", param="messages")

    text_parts: list[str] = []
    images: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            raise ProxyError("Invalid message content part", param="messages")
        typ = part.get("type")
        if typ in {"text", "input_text", "output_text"}:
            text = part.get("text", "")
            if not isinstance(text, str):
                raise ProxyError("Text content part must contain a string 'text'", param="messages")
            text_parts.append(text)
            continue
        if typ == "refusal":
            refusal = part.get("refusal", "")
            if isinstance(refusal, str):
                text_parts.append(refusal)
            continue
        if typ in {"image_url", "input_image"}:
            if not allow_images:
                raise ProxyError("Images are not supported in this content position", param="messages")
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
            elif isinstance(image_url, str):
                url = image_url
            else:
                url = part.get("image_url") or part.get("url")
            if not isinstance(url, str):
                raise ProxyError("Image content part is missing image_url.url", param="messages")
            if url.startswith("data:"):
                marker = ";base64,"
                pos = url.find(marker)
                if pos < 0:
                    raise ProxyError("Only base64 data URLs are supported for image input", param="messages")
                images.append(url[pos + len(marker) :])
            else:
                raise ProxyError(
                    "Remote image URLs are not supported by this proxy; send a base64 data URL instead",
                    param="messages",
                )
            continue
        raise ProxyError(f"Unsupported OpenAI content part type: {typ!r}", param="messages")
    return "".join(text_parts), images


def build_tool_id_map(messages: list[Any]) -> dict[str, str]:
    tool_ids: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            fn = tc.get("function") or {}
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(tc_id, str) and isinstance(name, str):
                tool_ids[tc_id] = name
    return tool_ids


def convert_messages(messages: Any) -> list[JSON]:
    if not isinstance(messages, list) or not messages:
        raise ProxyError("'messages' must be a non-empty array", param="messages")

    tool_id_map = build_tool_id_map(messages)
    out: list[JSON] = []
    for src in messages:
        if not isinstance(src, dict):
            raise ProxyError("Each message must be an object", param="messages")
        role = src.get("role")
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant", "tool", "function"}:
            raise ProxyError(f"Unsupported message role: {role!r}", param="messages")

        if role == "function":
            role = "tool"

        text, images = extract_text_content(src.get("content"), allow_images=(role == "user"))
        dst: JSON = {"role": role, "content": text}
        if images:
            dst["images"] = images

        if role == "assistant" and src.get("tool_calls"):
            calls: list[JSON] = []
            for index, tc in enumerate(src.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    raise ProxyError("Invalid assistant tool_calls entry", param="messages")
                call: JSON = {
                    "function": {
                        "index": index,
                        "name": fn["name"],
                        "arguments": parse_json_arguments(fn.get("arguments")),
                    }
                }
                if isinstance(tc.get("id"), str):
                    call["id"] = tc["id"]
                calls.append(call)
            if calls:
                dst["tool_calls"] = calls

        if role == "tool":
            tool_name = src.get("tool_name") or src.get("name")
            if not tool_name:
                call_id = src.get("tool_call_id")
                if isinstance(call_id, str):
                    tool_name = tool_id_map.get(call_id)
            if isinstance(tool_name, str) and tool_name:
                dst["tool_name"] = tool_name
            if isinstance(src.get("tool_call_id"), str):
                dst["tool_call_id"] = src["tool_call_id"]

        reasoning = src.get("reasoning")
        if reasoning is None:
            reasoning = src.get("reasoning_content")
        if role == "assistant" and isinstance(reasoning, str) and reasoning:
            dst["thinking"] = reasoning

        out.append(dst)
    return out


def convert_tools(tools: Any) -> list[JSON] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ProxyError("'tools' must be an array", param="tools")
    out: list[JSON] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ProxyError("Only function tools are supported", param="tools")
        fn = tool.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            raise ProxyError("Invalid function tool", param="tools")
        function: JSON = {"name": fn["name"]}
        if isinstance(fn.get("description"), str):
            function["description"] = fn["description"]
        parameters = fn.get("parameters")
        if isinstance(parameters, dict):
            function["parameters"] = parameters
        out.append({"type": "function", "function": function})
    return out


def apply_tool_choice(tools: list[JSON] | None, tool_choice: Any) -> list[JSON] | None:
    if tools is None or tool_choice is None or tool_choice == "auto":
        return tools
    if tool_choice == "none":
        return None
    if tool_choice == "required":
        # Ollama's native /api/chat accepts tools but currently has no separate
        # documented required-tool switch. Keep the tools available rather than
        # silently deleting them; the model is still free to answer directly.
        return tools
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise ProxyError("Only function tool_choice is supported", param="tool_choice")
        fn = tool_choice.get("function") or {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str):
            raise ProxyError("Invalid function tool_choice", param="tool_choice")
        selected = [t for t in tools if (t.get("function") or {}).get("name") == name]
        if not selected:
            raise ProxyError(f"tool_choice references unknown function {name!r}", param="tool_choice")
        return selected
    raise ProxyError("Unsupported tool_choice", param="tool_choice")


def response_format_to_ollama(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProxyError("'response_format' must be an object", param="response_format")
    typ = value.get("type")
    if typ in {None, "text"}:
        return None
    if typ == "json_object":
        return "json"
    if typ == "json_schema":
        wrapper = value.get("json_schema") or {}
        if not isinstance(wrapper, dict):
            raise ProxyError("Invalid json_schema response_format", param="response_format")
        schema = wrapper.get("schema")
        if not isinstance(schema, dict):
            raise ProxyError("response_format.json_schema.schema must be an object", param="response_format")
        return schema
    raise ProxyError(f"Unsupported response_format type: {typ!r}", param="response_format")


def openai_generation_options(body: JSON) -> JSON:
    options: JSON = {}
    direct = {
        "temperature": "temperature",
        "top_p": "top_p",
        "seed": "seed",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
    }
    for src, dst in direct.items():
        if body.get(src) is not None:
            options[dst] = body[src]

    # Match OpenAI-compatible defaults rather than Ollama model defaults.
    if "temperature" not in options:
        options["temperature"] = 1.0
    if "top_p" not in options:
        options["top_p"] = 1.0

    max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise ProxyError("max_tokens/max_completion_tokens must be an integer", param="max_tokens")
        options["num_predict"] = max_tokens

    stop = normalize_stop(body.get("stop"))
    if stop is not None:
        options["stop"] = stop

    # Explicit extension for callers that know Ollama options. It is merged last
    # so deployment-specific options can intentionally override translated ones.
    ext = body.get("ollama_options")
    if ext is not None:
        if not isinstance(ext, dict):
            raise ProxyError("ollama_options must be an object", param="ollama_options")
        options.update(ext)
    return options


def openai_think(body: JSON) -> Any:
    if "think" in body:  # explicit Ollama-compatible extension
        value = body["think"]
        if isinstance(value, bool) or value in {"low", "medium", "high", "max"}:
            return value
        raise ProxyError("think must be boolean or low/medium/high/max", param="think")

    effort = body.get("reasoning_effort")
    if effort is None and isinstance(body.get("reasoning"), dict):
        effort = body["reasoning"].get("effort")
    if effort is None:
        return None
    mapping = {
        "none": False,
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "max",
        "max": "max",
    }
    if effort not in mapping:
        raise ProxyError("Unsupported reasoning_effort", param="reasoning_effort")
    return mapping[effort]


def convert_ollama_tool_calls(tool_calls: Any) -> list[JSON]:
    if not isinstance(tool_calls, list):
        return []
    out: list[JSON] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        out.append(
            {
                "id": tc.get("id") if isinstance(tc.get("id"), str) and tc.get("id") else new_id("call_"),
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "arguments": stringify_tool_arguments(fn.get("arguments")),
                },
            }
        )
    return out


def chat_logprobs(ollama_logprobs: Any) -> JSON | None:
    if not isinstance(ollama_logprobs, list):
        return None
    content: list[JSON] = []
    for item in ollama_logprobs:
        if not isinstance(item, dict):
            continue
        top: list[JSON] = []
        for t in item.get("top_logprobs") or []:
            if isinstance(t, dict):
                top.append(
                    {
                        "token": str(t.get("token", "")),
                        "logprob": float(t.get("logprob", -9999.0)),
                        "bytes": t.get("bytes"),
                    }
                )
        content.append(
            {
                "token": str(item.get("token", "")),
                "logprob": float(item.get("logprob", -9999.0)),
                "bytes": item.get("bytes"),
                "top_logprobs": top,
            }
        )
    return {"content": content, "refusal": None}


def legacy_completion_logprobs(ollama_logprobs: Any) -> JSON | None:
    if not isinstance(ollama_logprobs, list):
        return None
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []
    text_offset: list[int] = []
    offset = 0
    for item in ollama_logprobs:
        if not isinstance(item, dict):
            continue
        token = str(item.get("token", ""))
        tokens.append(token)
        token_logprobs.append(float(item.get("logprob", -9999.0)))
        tops: dict[str, float] = {}
        for t in item.get("top_logprobs") or []:
            if isinstance(t, dict):
                tops[str(t.get("token", ""))] = float(t.get("logprob", -9999.0))
        top_logprobs.append(tops)
        text_offset.append(offset)
        offset += len(token)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    }


def parse_stream_flag(body: JSON) -> bool:
    value = body.get("stream", False)
    if not isinstance(value, bool):
        raise ProxyError("'stream' must be a boolean", param="stream")
    return value


def convert_responses_tools(tools: Any) -> list[JSON] | None:
    """Convert Responses API flat function tools to Ollama native tools.

    Also accepts Chat Completions' nested function shape as a compatibility
    convenience; several OpenAI-compatible clients reuse that representation.
    """
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ProxyError("'tools' must be an array", param="tools")
    out: list[JSON] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ProxyError("Only function tools are supported by /v1/responses", param="tools")
        source = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = source.get("name") if isinstance(source, dict) else None
        if not isinstance(name, str) or not name:
            raise ProxyError("Invalid function tool", param="tools")
        fn: JSON = {"name": name}
        if isinstance(source.get("description"), str):
            fn["description"] = source["description"]
        parameters = source.get("parameters")
        if isinstance(parameters, dict):
            fn["parameters"] = parameters
        out.append({"type": "function", "function": fn})
    return out


def responses_text_format_to_ollama(text: Any) -> Any:
    if text is None:
        return None
    if not isinstance(text, dict):
        raise ProxyError("'text' must be an object", param="text")
    fmt = text.get("format")
    if fmt is None:
        return None
    if not isinstance(fmt, dict):
        raise ProxyError("text.format must be an object", param="text")
    typ = fmt.get("type")
    if typ in {None, "text"}:
        return None
    if typ == "json_object":
        return "json"
    if typ == "json_schema":
        schema = fmt.get("schema")
        if not isinstance(schema, dict):
            raise ProxyError("text.format.schema must be an object", param="text")
        return schema
    raise ProxyError(f"Unsupported text.format type: {typ!r}", param="text")


def responses_input_to_messages(body: JSON) -> list[JSON]:
    messages: list[JSON] = []
    instructions = body.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise ProxyError("'instructions' must be a string", param="instructions")
        if instructions:
            messages.append({"role": "system", "content": instructions})

    value = body.get("input")
    if isinstance(value, str):
        messages.append({"role": "user", "content": value})
        return messages
    if not isinstance(value, list) or not value:
        raise ProxyError("'input' must be a string or a non-empty array", param="input")

    # Build an ID -> tool name map when both calls and outputs are supplied in
    # one request. Ollama can also correlate by tool_call_id on newer versions.
    call_names: dict[str, str] = {}
    for item in value:
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name")
            if isinstance(call_id, str) and isinstance(name, str):
                call_names[call_id] = name

    for item in value:
        if not isinstance(item, dict):
            raise ProxyError("Each Responses input item must be an object", param="input")
        typ = item.get("type")

        # Input-message items; tolerate omitted type because some clients send
        # Chat-style message objects through Responses-compatible frontends.
        if typ in {None, "message"} and isinstance(item.get("role"), str):
            role = item["role"]
            if role == "developer":
                role = "system"
            if role not in {"system", "user", "assistant", "tool"}:
                raise ProxyError(f"Unsupported Responses message role: {role!r}", param="input")
            text, images = extract_text_content(item.get("content"), allow_images=(role == "user"))
            msg: JSON = {"role": role, "content": text}
            if images:
                msg["images"] = images
            messages.append(msg)
            continue

        if typ == "function_call":
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise ProxyError("function_call input item is missing name", param="input")
            call_id = item.get("call_id") or item.get("id")
            tc: JSON = {
                "function": {
                    "index": 0,
                    "name": name,
                    "arguments": parse_json_arguments(item.get("arguments")),
                }
            }
            if isinstance(call_id, str) and call_id:
                tc["id"] = call_id
            messages.append({"role": "assistant", "content": "", "tool_calls": [tc]})
            continue

        if typ == "function_call_output":
            call_id = item.get("call_id")
            output = item.get("output", "")

            # Responses API function outputs may be either a string or a list of
            # input_text/input_image/input_file content parts. Programmer's
            # read_image tool uses the multimodal form. Preserve image bytes
            # instead of JSON-stringifying the whole content array.
            tool_images: list[str] = []
            if isinstance(output, str):
                output_text = output
            elif isinstance(output, list):
                output_text, tool_images = extract_text_content(output, allow_images=True)
            else:
                # Be liberal with non-standard compatible clients that return a
                # JSON value here, while keeping standards-compliant arrays
                # multimodal.
                output_text = json_dumps(output)

            msg: JSON = {"role": "tool", "content": output_text}
            tool_name: str | None = None
            if isinstance(call_id, str) and call_id:
                msg["tool_call_id"] = call_id
                if call_id in call_names:
                    tool_name = call_names[call_id]
                    msg["tool_name"] = tool_name
            if tool_name is None and isinstance(item.get("name"), str):
                tool_name = item["name"]
                msg["tool_name"] = tool_name
            messages.append(msg)

            if tool_images:
                # Ollama accepts `images` on Message, but as of 2026 vision
                # models do not reliably consume images attached to role=tool.
                # Inject a following user image message so multimodal tool
                # outputs (notably Programmer read_image) are actually visible
                # to the model while retaining the correlated tool result above.
                label = f"Image output from tool {tool_name}." if tool_name else "Image output from tool call."
                messages.append({"role": "user", "content": label, "images": tool_images})
            continue

        if typ == "reasoning":
            # Reasoning items are model-side context. Preserve textual summary
            # when present without pretending encrypted reasoning is portable.
            pieces: list[str] = []
            for part in item.get("summary") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    pieces.append(part["text"])
            if pieces:
                messages.append({"role": "assistant", "content": "", "thinking": "".join(pieces)})
            continue

        raise ProxyError(f"Unsupported Responses input item type: {typ!r}", param="input")

    if not messages:
        raise ProxyError("Responses input produced no messages", param="input")
    return messages


def responses_usage(ollama: JSON) -> JSON:
    prompt = int(ollama.get("prompt_eval_count") or 0)
    output = int(ollama.get("eval_count") or 0)
    cached = int(ollama.get("prompt_eval_cached_count") or 0)

    # OpenAI's current Responses schema requires both detail objects and, in
    # particular, cache_write_tokens + cached_tokens on input details and
    # reasoning_tokens on output details. Ollama currently reports cache hits
    # but not cache writes or reasoning-token accounting separately, so use 0
    # for fields for which there is no native counter. Keep forward-compatible
    # aliases in case Ollama adds a write counter later.
    cache_write = int(
        ollama.get("prompt_eval_cache_write_count")
        or ollama.get("cache_write_count")
        or 0
    )
    return {
        "input_tokens": prompt,
        "input_tokens_details": {
            "cache_write_tokens": cache_write,
            "cached_tokens": cached,
        },
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt + output,
    }


def build_responses_output(message: JSON, *, message_id: str | None = None) -> list[JSON]:
    output: list[JSON] = []
    thinking = message.get("thinking")
    if isinstance(thinking, str) and thinking:
        output.append(
            {
                "id": new_id("rs_"),
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": thinking}],
                "status": "completed",
            }
        )

    tool_calls = convert_ollama_tool_calls(message.get("tool_calls"))
    for tc in tool_calls:
        output.append(
            {
                "id": new_id("fc_"),
                "type": "function_call",
                "status": "completed",
                "call_id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            }
        )

    content = message.get("content")
    if isinstance(content, str) and content:
        output.append(
            {
                "id": message_id or new_id("msg_"),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        )
    return output


def make_response_object(
    *,
    response_id: str,
    model: str,
    created_at: int,
    body: JSON,
    output: list[JSON],
    usage: JSON | None,
    status: str = "completed",
) -> JSON:
    obj: JSON = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": body.get("reasoning"),
        "store": False,
        "temperature": body.get("temperature", 1.0),
        "text": body.get("text") or {"format": {"type": "text"}},
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools") or [],
        "top_p": body.get("top_p", 1.0),
        "truncation": body.get("truncation", "disabled"),
        "usage": usage,
        "metadata": body.get("metadata") or {},
    }
    return obj


class Proxy:
    def __init__(
        self,
        upstream: str,
        *,
        connect_timeout: float,
        max_connections: int,
        api_key: str | None,
        upstream_api_key: str | None,
    ) -> None:
        if "://" not in upstream:
            upstream = "http://" + upstream
        self.upstream = upstream.rstrip("/")
        self.connect_timeout = connect_timeout
        self.max_connections = max_connections
        self.api_key = api_key
        self.upstream_api_key = upstream_api_key
        self.session: ClientSession | None = None

    async def start(self) -> None:
        timeout = ClientTimeout(total=None, connect=self.connect_timeout, sock_connect=self.connect_timeout, sock_read=None)
        connector = TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        headers = {"User-Agent": "openai-ollama-proxy/1.0"}
        if self.upstream_api_key:
            headers["Authorization"] = f"Bearer {self.upstream_api_key}"
        self.session = ClientSession(timeout=timeout, connector=connector, headers=headers)

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None

    def check_auth(self, request: web.Request) -> None:
        if not self.api_key:
            return
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {self.api_key}":
            raise ProxyError(
                "Incorrect API key provided",
                status=401,
                error_type="invalid_request_error",
                code="invalid_api_key",
            )

    def _session(self) -> ClientSession:
        if self.session is None:
            raise RuntimeError("Proxy session is not started")
        return self.session

    async def upstream_json(self, method: str, path: str, payload: Any = None) -> tuple[int, JSON]:
        try:
            async with self._session().request(method, self.upstream + path, json=payload) as resp:
                raw = await resp.read()
                try:
                    data = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    text = raw.decode(errors="replace")
                    data = {"error": text or f"Ollama returned HTTP {resp.status}"}
                if not isinstance(data, dict):
                    data = {"data": data}
                return resp.status, data
        except (ClientConnectionError, asyncio.TimeoutError) as exc:
            raise ProxyError(
                f"Unable to connect to Ollama upstream at {self.upstream}: {exc}",
                status=502,
                error_type="api_error",
                code="upstream_unavailable",
            ) from exc

    async def open_upstream_stream(self, path: str, payload: JSON) -> ClientResponse:
        try:
            return await self._session().post(self.upstream + path, json=payload)
        except (ClientConnectionError, asyncio.TimeoutError) as exc:
            raise ProxyError(
                f"Unable to connect to Ollama upstream at {self.upstream}: {exc}",
                status=502,
                error_type="api_error",
                code="upstream_unavailable",
            ) from exc

    @staticmethod
    def raise_upstream(status: int, data: JSON) -> None:
        if status < 400:
            return
        message = data.get("error")
        if isinstance(message, dict):
            message = message.get("message")
        if not isinstance(message, str):
            message = f"Ollama upstream returned HTTP {status}"
        if status == 404:
            typ, code = "invalid_request_error", "model_not_found"
        elif status in {400, 422}:
            typ, code = "invalid_request_error", None
        else:
            typ, code = "api_error", "upstream_error"
        raise ProxyError(message, status=status, error_type=typ, code=code)

    async def models(self, request: web.Request) -> web.Response:
        self.check_auth(request)
        status, data = await self.upstream_json("GET", "/api/tags")
        self.raise_upstream(status, data)
        models = data.get("models") or []
        result: list[JSON] = []
        for model in models:
            if not isinstance(model, dict):
                continue
            model_id = model.get("model") or model.get("name")
            if not isinstance(model_id, str):
                continue
            result.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": parse_iso_timestamp(model.get("modified_at")),
                    "owned_by": "library",
                }
            )
        return web.json_response({"object": "list", "data": result})

    async def model(self, request: web.Request) -> web.Response:
        self.check_auth(request)
        wanted = request.match_info["model"]
        status, data = await self.upstream_json("GET", "/api/tags")
        self.raise_upstream(status, data)
        for model in data.get("models") or []:
            if not isinstance(model, dict):
                continue
            model_id = model.get("model") or model.get("name")
            if model_id == wanted:
                return web.json_response(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": parse_iso_timestamp(model.get("modified_at")),
                        "owned_by": "library",
                    }
                )
        raise ProxyError(
            f"The model '{wanted}' does not exist",
            status=404,
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
        )

    async def chat_completions(self, request: web.Request) -> web.StreamResponse:
        self.check_auth(request)
        body = await read_json_body(request)
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise ProxyError("'model' is required", param="model")
        n = body.get("n", 1)
        if n not in (None, 1):
            raise ProxyError("This proxy currently supports n=1 only", param="n")

        tools = apply_tool_choice(convert_tools(body.get("tools")), body.get("tool_choice"))
        payload: JSON = {
            "model": model,
            "messages": convert_messages(body.get("messages")),
            "stream": parse_stream_flag(body),
        }
        if tools:
            payload["tools"] = tools
        fmt = response_format_to_ollama(body.get("response_format"))
        if fmt is not None:
            payload["format"] = fmt
        options = openai_generation_options(body)
        if options:
            payload["options"] = options
        think = openai_think(body)
        if think is not None:
            payload["think"] = think
        if body.get("keep_alive") is not None:
            payload["keep_alive"] = body["keep_alive"]
        if body.get("logprobs") is not None:
            payload["logprobs"] = bool(body["logprobs"])
        if body.get("top_logprobs") is not None:
            payload["top_logprobs"] = body["top_logprobs"]

        if payload["stream"]:
            return await self.stream_chat(request, payload, body)

        status, result = await self.upstream_json("POST", "/api/chat", payload)
        self.raise_upstream(status, result)
        message = result.get("message") or {}
        if not isinstance(message, dict):
            message = {}
        tool_calls = convert_ollama_tool_calls(message.get("tool_calls"))
        content = message.get("content")
        if content is None:
            content = ""
        out_message: JSON = {"role": "assistant", "content": content}
        if tool_calls:
            out_message["tool_calls"] = tool_calls
            if not content:
                out_message["content"] = None
        thinking = message.get("thinking")
        if isinstance(thinking, str) and thinking:
            # Widely used OpenAI-compatible extension. Unknown response fields
            # are ignored by standard clients but reasoning-aware clients can
            # preserve Ollama's separate thinking channel.
            out_message["reasoning"] = thinking
            out_message["reasoning_content"] = thinking

        lp = chat_logprobs(result.get("logprobs"))
        response: JSON = {
            "id": new_id("chatcmpl-"),
            "object": "chat.completion",
            "created": parse_iso_timestamp(result.get("created_at")) or now_unix(),
            "model": result.get("model") or model,
            "system_fingerprint": "fp_ollama",
            "choices": [
                {
                    "index": 0,
                    "message": out_message,
                    "logprobs": lp,
                    "finish_reason": finish_reason(result, bool(tool_calls)),
                }
            ],
            "usage": openai_usage(result),
        }
        return web.json_response(response)

    async def stream_chat(self, request: web.Request, payload: JSON, original: JSON) -> web.StreamResponse:
        upstream = await self.open_upstream_stream("/api/chat", payload)
        try:
            if upstream.status >= 400:
                raw = await upstream.read()
                try:
                    data = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    data = {"error": raw.decode(errors="replace")}
                if not isinstance(data, dict):
                    data = {"error": str(data)}
                self.raise_upstream(upstream.status, data)

            request_id = new_id("chatcmpl-")
            created = now_unix()
            model = str(payload["model"])
            stream = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream; charset=utf-8",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "x-request-id": request.get("request_id", ""),
                },
            )
            await stream.prepare(request)

            async def send(obj: JSON) -> None:
                await stream.write(f"data: {json_dumps(obj)}\n\n".encode())

            await send(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "system_fingerprint": "fp_ollama",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}
                    ],
                }
            )

            include_usage = bool((original.get("stream_options") or {}).get("include_usage")) if isinstance(original.get("stream_options") or {}, dict) else False
            final_usage: JSON | None = None
            saw_done = False
            tool_ids: dict[int, str] = {}
            next_tool_index = 0
            saw_tool_calls = False

            async for raw_line in upstream.content:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                if item.get("error"):
                    await send(openai_error_payload(str(item["error"]), error_type="api_error", code="upstream_error"))
                    saw_done = True
                    break

                msg = item.get("message") or {}
                if not isinstance(msg, dict):
                    msg = {}
                delta: JSON = {}
                content = msg.get("content")
                if isinstance(content, str) and content:
                    delta["content"] = content
                thinking = msg.get("thinking")
                if isinstance(thinking, str) and thinking:
                    delta["reasoning"] = thinking
                    delta["reasoning_content"] = thinking

                raw_calls = msg.get("tool_calls")
                if isinstance(raw_calls, list) and raw_calls:
                    saw_tool_calls = True
                    delta_calls: list[JSON] = []
                    for pos, tc in enumerate(raw_calls):
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function") or {}
                        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                            continue
                        if isinstance(fn.get("index"), int):
                            idx = fn["index"]
                        else:
                            idx = next_tool_index
                            next_tool_index += 1
                        native_id = tc.get("id") if isinstance(tc.get("id"), str) and tc.get("id") else None
                        call_id = tool_ids.setdefault(idx, native_id or new_id("call_"))
                        delta_calls.append(
                            {
                                "index": idx,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": fn["name"],
                                    "arguments": stringify_tool_arguments(fn.get("arguments")),
                                },
                            }
                        )
                    if delta_calls:
                        delta["tool_calls"] = delta_calls

                if delta:
                    lp = chat_logprobs(item.get("logprobs"))
                    await send(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": item.get("model") or model,
                            "system_fingerprint": "fp_ollama",
                            "choices": [{"index": 0, "delta": delta, "logprobs": lp, "finish_reason": None}],
                        }
                    )

                if item.get("done"):
                    saw_done = True
                    final_usage = openai_usage(item)
                    await send(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": item.get("model") or model,
                            "system_fingerprint": "fp_ollama",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "logprobs": None,
                                    "finish_reason": finish_reason(item, saw_tool_calls),
                                }
                            ],
                        }
                    )
                    break

            if not saw_done:
                await send(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "system_fingerprint": "fp_ollama",
                        "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}],
                    }
                )

            if include_usage and final_usage is not None:
                await send(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "system_fingerprint": "fp_ollama",
                        "choices": [],
                        "usage": final_usage,
                    }
                )
            await stream.write(b"data: [DONE]\n\n")
            await stream.write_eof()
            return stream
        except (ConnectionResetError, asyncio.CancelledError):
            raise
        finally:
            upstream.close()

    async def responses(self, request: web.Request) -> web.StreamResponse:
        self.check_auth(request)
        body = await read_json_body(request)
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise ProxyError("'model' is required", param="model")
        if body.get("previous_response_id") is not None:
            raise ProxyError("Stateful Responses API (previous_response_id) is not supported", param="previous_response_id")
        if body.get("conversation") is not None:
            raise ProxyError("Stateful Responses API (conversation) is not supported", param="conversation")
        if body.get("background") not in (None, False):
            raise ProxyError("Background Responses are not supported", param="background")

        stream_requested = parse_stream_flag(body)
        tools = convert_responses_tools(body.get("tools"))
        # Responses tool_choice may be a string or {type:function,name:...}.
        choice = body.get("tool_choice")
        if choice == "none":
            tools = None
        elif isinstance(choice, dict) and choice.get("type") == "function":
            name = choice.get("name")
            if not isinstance(name, str):
                raise ProxyError("Invalid Responses function tool_choice", param="tool_choice")
            selected = [t for t in (tools or []) if (t.get("function") or {}).get("name") == name]
            if not selected:
                raise ProxyError(f"tool_choice references unknown function {name!r}", param="tool_choice")
            tools = selected
        elif choice not in (None, "auto", "required"):
            raise ProxyError("Unsupported Responses tool_choice", param="tool_choice")

        options_body = dict(body)
        if body.get("max_output_tokens") is not None:
            options_body["max_tokens"] = body["max_output_tokens"]
        payload: JSON = {
            "model": model,
            "messages": responses_input_to_messages(body),
            "stream": stream_requested,
        }
        if tools:
            payload["tools"] = tools
        options = openai_generation_options(options_body)
        if options:
            payload["options"] = options
        fmt = responses_text_format_to_ollama(body.get("text"))
        if fmt is not None:
            payload["format"] = fmt
        think = openai_think(body)
        if think is not None:
            payload["think"] = think
        if body.get("keep_alive") is not None:
            payload["keep_alive"] = body["keep_alive"]

        if stream_requested:
            return await self.stream_responses(request, payload, body)

        status, result = await self.upstream_json("POST", "/api/chat", payload)
        self.raise_upstream(status, result)
        message = result.get("message") or {}
        if not isinstance(message, dict):
            message = {}
        response_id = new_id("resp_")
        created = parse_iso_timestamp(result.get("created_at")) or now_unix()
        response = make_response_object(
            response_id=response_id,
            model=str(result.get("model") or model),
            created_at=created,
            body=body,
            output=build_responses_output(message),
            usage=responses_usage(result),
        )
        response["completed_at"] = created
        return web.json_response(response)

    async def stream_responses(self, request: web.Request, payload: JSON, body: JSON) -> web.StreamResponse:
        upstream = await self.open_upstream_stream("/api/chat", payload)
        try:
            if upstream.status >= 400:
                raw = await upstream.read()
                try:
                    data = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    data = {"error": raw.decode(errors="replace")}
                if not isinstance(data, dict):
                    data = {"error": str(data)}
                self.raise_upstream(upstream.status, data)

            response_id = new_id("resp_")
            created = now_unix()
            model = str(payload["model"])
            response_stream = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream; charset=utf-8",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "x-request-id": request.get("request_id", ""),
                },
            )
            await response_stream.prepare(request)
            seq = 0

            async def send(event_type: str, fields: JSON) -> None:
                nonlocal seq
                obj = {"type": event_type, **fields, "sequence_number": seq}
                seq += 1
                await response_stream.write(
                    f"event: {event_type}\ndata: {json_dumps(obj)}\n\n".encode()
                )

            initial = make_response_object(
                response_id=response_id,
                model=model,
                created_at=created,
                body=body,
                output=[],
                usage=None,
                status="in_progress",
            )
            await send("response.created", {"response": initial})
            await send("response.in_progress", {"response": initial})

            output_items: list[JSON] = []
            full_text = ""
            full_thinking = ""
            message_id: str | None = None
            message_index: int | None = None
            message_started = False
            native_done: JSON = {}

            async for raw_line in upstream.content:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                if item.get("error"):
                    err = {
                        "type": "server_error",
                        "code": "upstream_error",
                        "message": str(item["error"]),
                        "param": None,
                    }
                    await send("response.failed", {"response": {**initial, "status": "failed", "error": err}})
                    await response_stream.write_eof()
                    return response_stream

                msg = item.get("message") or {}
                if not isinstance(msg, dict):
                    msg = {}
                thinking = msg.get("thinking")
                if isinstance(thinking, str):
                    full_thinking += thinking

                text_delta = msg.get("content")
                if isinstance(text_delta, str) and text_delta:
                    if not message_started:
                        message_started = True
                        message_id = new_id("msg_")
                        message_index = len(output_items)
                        message_item: JSON = {
                            "id": message_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        }
                        output_items.append(message_item)
                        await send(
                            "response.output_item.added",
                            {"output_index": message_index, "item": message_item},
                        )
                        await send(
                            "response.content_part.added",
                            {
                                "item_id": message_id,
                                "output_index": message_index,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
                            },
                        )
                    full_text += text_delta
                    await send(
                        "response.output_text.delta",
                        {
                            "item_id": message_id,
                            "output_index": message_index,
                            "content_index": 0,
                            "delta": text_delta,
                            "logprobs": [],
                        },
                    )

                raw_calls = msg.get("tool_calls")
                if isinstance(raw_calls, list):
                    for native_call in raw_calls:
                        if not isinstance(native_call, dict):
                            continue
                        fn = native_call.get("function") or {}
                        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                            continue
                        call_id = native_call.get("id") if isinstance(native_call.get("id"), str) else None
                        call_id = call_id or new_id("call_")
                        args = stringify_tool_arguments(fn.get("arguments"))
                        fc_id = new_id("fc_")
                        idx = len(output_items)
                        call_item: JSON = {
                            "id": fc_id,
                            "type": "function_call",
                            "status": "in_progress",
                            "call_id": call_id,
                            "name": fn["name"],
                            "arguments": "",
                        }
                        output_items.append(call_item)
                        await send("response.output_item.added", {"output_index": idx, "item": call_item})
                        if args:
                            await send(
                                "response.function_call_arguments.delta",
                                {"item_id": fc_id, "output_index": idx, "delta": args},
                            )
                        await send(
                            "response.function_call_arguments.done",
                            {"item_id": fc_id, "output_index": idx, "arguments": args},
                        )
                        completed_call = {**call_item, "status": "completed", "arguments": args}
                        output_items[idx] = completed_call
                        await send("response.output_item.done", {"output_index": idx, "item": completed_call})

                if item.get("done"):
                    native_done = item
                    break

            if message_started and message_id is not None and message_index is not None:
                part = {"type": "output_text", "text": full_text, "annotations": [], "logprobs": []}
                await send(
                    "response.output_text.done",
                    {
                        "item_id": message_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "text": full_text,
                        "logprobs": [],
                    },
                )
                await send(
                    "response.content_part.done",
                    {"item_id": message_id, "output_index": message_index, "content_index": 0, "part": part},
                )
                completed_message: JSON = {
                    "id": message_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [part],
                }
                output_items[message_index] = completed_message
                await send("response.output_item.done", {"output_index": message_index, "item": completed_message})

            # Ollama exposes the model's thinking separately. Treat it as a
            # reasoning summary. It is emitted at completion because Ollama may
            # interleave thinking/content differently across model families.
            if full_thinking:
                rs_id = new_id("rs_")
                idx = len(output_items)
                reasoning_item: JSON = {
                    "id": rs_id,
                    "type": "reasoning",
                    "summary": [],
                    "status": "in_progress",
                }
                output_items.append(reasoning_item)
                await send("response.output_item.added", {"output_index": idx, "item": reasoning_item})
                summary_part = {"type": "summary_text", "text": ""}
                await send(
                    "response.reasoning_summary_part.added",
                    {"item_id": rs_id, "output_index": idx, "summary_index": 0, "part": summary_part},
                )
                await send(
                    "response.reasoning_summary_text.delta",
                    {"item_id": rs_id, "output_index": idx, "summary_index": 0, "delta": full_thinking},
                )
                final_part = {"type": "summary_text", "text": full_thinking}
                await send(
                    "response.reasoning_summary_text.done",
                    {"item_id": rs_id, "output_index": idx, "summary_index": 0, "text": full_thinking},
                )
                await send(
                    "response.reasoning_summary_part.done",
                    {"item_id": rs_id, "output_index": idx, "summary_index": 0, "part": final_part},
                )
                completed_reasoning = {
                    "id": rs_id,
                    "type": "reasoning",
                    "summary": [final_part],
                    "status": "completed",
                }
                output_items[idx] = completed_reasoning
                await send("response.output_item.done", {"output_index": idx, "item": completed_reasoning})

            final_model = str(native_done.get("model") or model)
            final = make_response_object(
                response_id=response_id,
                model=final_model,
                created_at=created,
                body=body,
                output=output_items,
                usage=responses_usage(native_done),
                status="completed",
            )
            final["completed_at"] = now_unix()
            await send("response.completed", {"response": final})
            await response_stream.write_eof()
            return response_stream
        except (ConnectionResetError, asyncio.CancelledError):
            raise
        finally:
            upstream.close()

    async def completions(self, request: web.Request) -> web.StreamResponse:
        self.check_auth(request)
        body = await read_json_body(request)
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise ProxyError("'model' is required", param="model")
        n = body.get("n", 1)
        if n not in (None, 1):
            raise ProxyError("This proxy currently supports n=1 only", param="n")
        if body.get("best_of") not in (None, 1):
            raise ProxyError("'best_of' values other than 1 are not supported", param="best_of")

        prompt = body.get("prompt")
        if isinstance(prompt, str):
            prompts = [prompt]
        elif isinstance(prompt, list) and prompt and all(isinstance(x, str) for x in prompt):
            prompts = prompt
        else:
            raise ProxyError("'prompt' must be a string or a non-empty array of strings", param="prompt")

        stream_requested = parse_stream_flag(body)
        if stream_requested:
            if len(prompts) != 1:
                raise ProxyError("Streaming legacy completions support a single prompt only", param="prompt")
            return await self.stream_completion(request, model, prompts[0], body)

        async def one(prompt_text: str) -> JSON:
            payload = self.build_generate_payload(model, prompt_text, body, stream=False)
            status, result = await self.upstream_json("POST", "/api/generate", payload)
            self.raise_upstream(status, result)
            return result

        results = await asyncio.gather(*(one(p) for p in prompts))
        completion_id = new_id("cmpl-")
        choices: list[JSON] = []
        for index, (prompt_text, result) in enumerate(zip(prompts, results)):
            text = str(result.get("response") or "")
            if body.get("echo"):
                text = prompt_text + text
            choices.append(
                {
                    "text": text,
                    "index": index,
                    "logprobs": legacy_completion_logprobs(result.get("logprobs")),
                    "finish_reason": finish_reason(result),
                }
            )
        return web.json_response(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": now_unix(),
                "model": model,
                "system_fingerprint": "fp_ollama",
                "choices": choices,
                "usage": completion_usage_sum(results),
            }
        )

    def build_generate_payload(self, model: str, prompt: str, body: JSON, *, stream: bool) -> JSON:
        payload: JSON = {"model": model, "prompt": prompt, "stream": stream}
        if isinstance(body.get("suffix"), str):
            payload["suffix"] = body["suffix"]
        options = openai_generation_options(body)
        if options:
            payload["options"] = options
        fmt = response_format_to_ollama(body.get("response_format"))
        if fmt is not None:
            payload["format"] = fmt
        think = openai_think(body)
        if think is not None:
            payload["think"] = think
        if body.get("keep_alive") is not None:
            payload["keep_alive"] = body["keep_alive"]
        if body.get("logprobs") is not None:
            payload["logprobs"] = True
            if isinstance(body.get("logprobs"), int) and not isinstance(body.get("logprobs"), bool):
                payload["top_logprobs"] = body["logprobs"]
        return payload

    async def stream_completion(self, request: web.Request, model: str, prompt: str, body: JSON) -> web.StreamResponse:
        payload = self.build_generate_payload(model, prompt, body, stream=True)
        upstream = await self.open_upstream_stream("/api/generate", payload)
        try:
            if upstream.status >= 400:
                raw = await upstream.read()
                try:
                    data = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    data = {"error": raw.decode(errors="replace")}
                if not isinstance(data, dict):
                    data = {"error": str(data)}
                self.raise_upstream(upstream.status, data)

            completion_id = new_id("cmpl-")
            created = now_unix()
            stream = web.StreamResponse(
                headers={
                    "Content-Type": "text/event-stream; charset=utf-8",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "x-request-id": request.get("request_id", ""),
                }
            )
            await stream.prepare(request)
            echo_sent = False
            saw_done = False
            final_usage: JSON | None = None

            async for raw_line in upstream.content:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                if item.get("error"):
                    await stream.write(f"data: {json_dumps(openai_error_payload(str(item['error'])))}\n\n".encode())
                    break
                text = str(item.get("response") or "")
                if body.get("echo") and not echo_sent:
                    text = prompt + text
                    echo_sent = True
                if text or item.get("done"):
                    chunk = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": item.get("model") or model,
                        "choices": [
                            {
                                "text": text,
                                "index": 0,
                                "logprobs": legacy_completion_logprobs(item.get("logprobs")),
                                "finish_reason": finish_reason(item) if item.get("done") else None,
                            }
                        ],
                    }
                    await stream.write(f"data: {json_dumps(chunk)}\n\n".encode())
                if item.get("done"):
                    saw_done = True
                    final_usage = openai_usage(item)
                    break
            if not saw_done:
                chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}],
                }
                await stream.write(f"data: {json_dumps(chunk)}\n\n".encode())
            stream_opts = body.get("stream_options") or {}
            include_usage = isinstance(stream_opts, dict) and bool(stream_opts.get("include_usage"))
            if include_usage and final_usage is not None:
                usage_chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": final_usage,
                }
                await stream.write(f"data: {json_dumps(usage_chunk)}\n\n".encode())
            await stream.write(b"data: [DONE]\n\n")
            await stream.write_eof()
            return stream
        finally:
            upstream.close()

    async def embeddings(self, request: web.Request) -> web.Response:
        self.check_auth(request)
        body = await read_json_body(request)
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise ProxyError("'model' is required", param="model")
        value = body.get("input")
        if isinstance(value, str):
            inputs: str | list[str] = value
        elif isinstance(value, list) and all(isinstance(x, str) for x in value):
            inputs = value
        else:
            raise ProxyError("Embedding input must be a string or an array of strings", param="input")
        payload: JSON = {"model": model, "input": inputs}
        if body.get("dimensions") is not None:
            payload["dimensions"] = body["dimensions"]
        if body.get("keep_alive") is not None:
            payload["keep_alive"] = body["keep_alive"]
        ext = body.get("ollama_options")
        if ext is not None:
            if not isinstance(ext, dict):
                raise ProxyError("ollama_options must be an object", param="ollama_options")
            payload["options"] = ext

        status, result = await self.upstream_json("POST", "/api/embed", payload)
        self.raise_upstream(status, result)
        vectors = result.get("embeddings") or []
        if not isinstance(vectors, list):
            raise ProxyError("Invalid embedding response from Ollama", status=502, error_type="api_error")
        encoding = body.get("encoding_format", "float")
        if encoding not in {"float", "base64"}:
            raise ProxyError("encoding_format must be 'float' or 'base64'", param="encoding_format")

        data: list[JSON] = []
        for index, vector in enumerate(vectors):
            if not isinstance(vector, list):
                continue
            embedding: Any = vector
            if encoding == "base64":
                try:
                    packed = struct.pack(f"<{len(vector)}f", *(float(x) for x in vector))
                except (TypeError, ValueError, struct.error) as exc:
                    raise ProxyError("Invalid embedding vector from Ollama", status=502, error_type="api_error") from exc
                embedding = base64.b64encode(packed).decode("ascii")
            data.append({"object": "embedding", "embedding": embedding, "index": index})

        tokens = int(result.get("prompt_eval_count") or 0)
        return web.json_response(
            {
                "object": "list",
                "data": data,
                "model": result.get("model") or model,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }
        )

    async def health(self, request: web.Request) -> web.Response:
        status, data = await self.upstream_json("GET", "/api/tags")
        if status >= 400:
            return web.json_response({"ok": False, "upstream_status": status}, status=503)
        return web.json_response({"ok": True, "upstream": self.upstream, "models": len(data.get("models") or [])})


async def read_json_body(request: web.Request) -> JSON:
    try:
        body = await request.json(loads=json.loads)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProxyError("Invalid JSON request body", status=400, error_type="invalid_request_error") from exc
    if not isinstance(body, dict):
        raise ProxyError("JSON request body must be an object", status=400, error_type="invalid_request_error")
    return body


@web.middleware
async def request_context_middleware(request: web.Request, handler):
    request_id = get_request_id(request)
    request["request_id"] = request_id
    start = time.monotonic()
    try:
        response = await handler(request)
    except ProxyError as exc:
        response = web.json_response(
            openai_error_payload(
                exc.message,
                error_type=exc.error_type,
                param=exc.param,
                code=exc.code,
            ),
            status=exc.status,
        )
    except web.HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except Exception:
        LOG.exception("Unhandled request error id=%s path=%s", request_id, request.path)
        response = web.json_response(
            openai_error_payload("Internal proxy error", error_type="api_error", code="internal_error"),
            status=500,
        )
    response.headers["x-request-id"] = request_id
    elapsed_ms = (time.monotonic() - start) * 1000
    LOG.info("%s %s -> %s %.1fms id=%s", request.method, request.path, response.status, elapsed_ms, request_id)
    return response


def make_app(args: argparse.Namespace) -> web.Application:
    proxy = Proxy(
        args.upstream,
        connect_timeout=args.connect_timeout,
        max_connections=args.max_connections,
        api_key=args.api_key,
        upstream_api_key=args.upstream_api_key,
    )
    app = web.Application(
        middlewares=[request_context_middleware],
        client_max_size=args.max_body_mb * 1024 * 1024,
    )
    app["proxy"] = proxy

    async def on_startup(_: web.Application) -> None:
        await proxy.start()

    async def on_cleanup(_: web.Application) -> None:
        await proxy.close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    # Canonical OpenAI-compatible paths.
    app.router.add_get("/v1/models", proxy.models)
    app.router.add_get(r"/v1/models/{model:.+}", proxy.model)
    app.router.add_post("/v1/chat/completions", proxy.chat_completions)
    app.router.add_post("/v1/completions", proxy.completions)
    app.router.add_post("/v1/embeddings", proxy.embeddings)
    app.router.add_post("/v1/responses", proxy.responses)

    # Useful aliases for clients configured with base_url=<proxy>/v1 but which
    # accidentally strip/preserve the version prefix differently.
    app.router.add_get("/models", proxy.models)
    app.router.add_get(r"/models/{model:.+}", proxy.model)
    app.router.add_post("/chat/completions", proxy.chat_completions)
    app.router.add_post("/completions", proxy.completions)
    app.router.add_post("/embeddings", proxy.embeddings)
    app.router.add_post("/responses", proxy.responses)

    app.router.add_get("/healthz", proxy.health)

    async def root(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "service": "openai-ollama-proxy",
                "openai_base_url": "/v1",
                "upstream": args.upstream,
            }
        )

    app.router.add_get("/", root)
    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible API to native Ollama API proxy")
    p.add_argument("--host", default="127.0.0.1", help="listen address (default: 127.0.0.1)")
    p.add_argument("--port", "-p", type=int, default=8000, help="listen port (default: 8000)")
    p.add_argument("--upstream", "-u", default="http://127.0.0.1:11434", help="Ollama base URL")
    p.add_argument("--connect-timeout", type=float, default=10.0, help="upstream connect timeout in seconds")
    p.add_argument(
        "--max-connections",
        type=int,
        default=0,
        help="max pooled upstream connections; 0 means unlimited (default: 0)",
    )
    p.add_argument("--max-body-mb", type=int, default=100, help="maximum request body size in MiB")
    p.add_argument("--api-key", default=None, help="optional Bearer token required from OpenAI clients")
    p.add_argument("--upstream-api-key", default=None, help="optional Bearer token sent to the Ollama upstream")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.port < 1 or args.port > 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.max_connections < 0:
        raise SystemExit("--max-connections must be >= 0")
    LOG.info("listening on %s:%s -> %s", args.host, args.port, args.upstream)
    web.run_app(make_app(args), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()

