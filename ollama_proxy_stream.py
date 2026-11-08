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
Reverse proxy for Ollama.

Incoming JSON requests containing ``"stream": false`` are sent upstream with
``"stream": true``. The streamed upstream response is accumulated and rebuilt
into the corresponding non-streaming JSON response.

Other requests are proxied unchanged at the HTTP payload level. If a top-level
``"stream": true`` appears early in a JSON upload, the upload is forwarded
immediately; otherwise it is spooled while mode is determined. Supplying
``X-Ollama-Proxy-Passthrough: true`` guarantees upload streaming and bypasses
flag inspection. Responses in passthrough mode are always streamed.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import tempfile
import gzip
import zlib
from contextlib import suppress
from collections.abc import Iterable, Mapping
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout, DummyCookieJar, TCPConnector, web
from multidict import CIMultiDict

LOG = logging.getLogger("ollama-stream-proxy")

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def connection_tokens(headers: Mapping[str, str]) -> set[str]:
    tokens: set[str] = set()
    for value in headers.getall("Connection", []):  # type: ignore[attr-defined]
        tokens.update(part.strip().lower() for part in value.split(",") if part.strip())
    return tokens


def filtered_request_headers(
    headers: Mapping[str, str], *, transformed: bool
) -> CIMultiDict[str]:
    blocked = HOP_BY_HOP_HEADERS | connection_tokens(headers) | {
        "host",
        "expect",
    }
    if transformed:
        blocked.add("content-length")  # JSON is reserialized; size changes.
        # The transformed response must be readable by the proxy. Avoid an
        # encoded NDJSON/SSE body whose Content-Encoding would then need to be
        # decoded and reconstructed.
        blocked |= {"accept-encoding", "content-encoding", "content-md5"}

    result: CIMultiDict[str] = CIMultiDict()
    for name, value in headers.items():
        if name.lower() not in blocked:
            result.add(name, value)

    if transformed:
        result["Content-Type"] = "application/json"
        result["Accept-Encoding"] = "identity"
    return result


def filtered_response_headers(
    headers: Mapping[str, str], *, transformed: bool
) -> CIMultiDict[str]:
    blocked = HOP_BY_HOP_HEADERS | connection_tokens(headers)
    if transformed:
        blocked |= {
            "content-length", "content-type", "content-encoding",
            "etag", "content-range", "content-md5", "digest",
        }

    result: CIMultiDict[str] = CIMultiDict()
    for name, value in headers.items():
        if name.lower() not in blocked:
            result.add(name, value)
    return result


def compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def parse_json_request(body: bytes) -> dict[str, Any] | None:
    if not body:
        return None
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def parse_ndjson(body: bytes) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(body.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid NDJSON on line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"NDJSON line {line_number} is not a JSON object")
        objects.append(value)
    if not objects:
        raise ValueError("upstream returned an empty NDJSON response")
    return objects


def parse_sse(body: bytes) -> list[tuple[str | None, Any]]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("upstream returned non-UTF-8 SSE") from exc

    events: list[tuple[str | None, Any]] = []
    event_name: str | None = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return
        data_text = "\n".join(data_lines)
        data_lines = []
        current_name = event_name
        event_name = None
        if data_text == "[DONE]":
            return
        try:
            data: Any = json.loads(data_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid SSE JSON payload: {exc}") from exc
        events.append((current_name, data))

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line == "":
            flush()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)

    flush()
    return events


def append_value(target: dict[str, Any], key: str, value: Any) -> None:
    """Merge a streamed delta into a reconstructed object."""
    if value is None:
        return
    if key not in target:
        target[key] = copy.deepcopy(value)
        return

    current = target[key]
    if isinstance(current, str) and isinstance(value, str):
        target[key] = current + value
    elif isinstance(current, list) and isinstance(value, list):
        current.extend(copy.deepcopy(value))
    elif isinstance(current, dict) and isinstance(value, dict):
        for child_key, child_value in value.items():
            append_value(current, child_key, child_value)
    else:
        target[key] = copy.deepcopy(value)


def merge_indexed_objects(
    destination: list[dict[str, Any]],
    incoming: Iterable[dict[str, Any]],
) -> None:
    """Merge OpenAI-style indexed tool-call/content-block deltas."""
    by_index: dict[int, dict[str, Any]] = {
        int(item.get("index", index)): item for index, item in enumerate(destination)
    }

    for fallback_index, fragment in enumerate(incoming):
        index_value = fragment.get("index", fallback_index)
        try:
            index = int(index_value)
        except (TypeError, ValueError):
            index = fallback_index

        item = by_index.get(index)
        if item is None:
            item = {"index": index}
            destination.append(item)
            by_index[index] = item

        for key, value in fragment.items():
            if key == "index":
                continue
            append_value(item, key, value)

    destination.sort(key=lambda item: int(item.get("index", 0)))


def aggregate_native_chat(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    message: dict[str, Any] = {}
    tool_calls: list[dict[str, Any]] = []
    saw_tool_calls = False

    for chunk in chunks:
        if "error" in chunk:
            raise RuntimeError(str(chunk["error"]))

        for key, value in chunk.items():
            if key != "message":
                result[key] = copy.deepcopy(value)

        part = chunk.get("message")
        if not isinstance(part, dict):
            continue

        for key, value in part.items():
            if key == "tool_calls" and isinstance(value, list):
                saw_tool_calls = True
                # Ollama documents tool calls as complete objects in individual
                # chunks. Some models/providers additionally emit an index, so
                # merge indexed fragments when possible and append otherwise.
                indexed = any(
                    isinstance(call, dict)
                    and (
                        "index" in call
                        or (
                            isinstance(call.get("function"), dict)
                            and "index" in call["function"]
                        )
                    )
                    for call in value
                )
                if indexed:
                    normalized: list[dict[str, Any]] = []
                    for fallback_index, call in enumerate(value):
                        if not isinstance(call, dict):
                            continue
                        call = copy.deepcopy(call)
                        function = call.get("function")
                        if "index" not in call and isinstance(function, dict):
                            if "index" in function:
                                call["index"] = function.pop("index")
                        call.setdefault("index", fallback_index)
                        normalized.append(call)
                    merge_indexed_objects(tool_calls, normalized)
                else:
                    tool_calls.extend(copy.deepcopy(value))
            elif key in {"content", "thinking"} and isinstance(value, str):
                append_value(message, key, value)
            elif key == "images" and isinstance(value, list):
                append_value(message, key, value)
            elif value not in (None, "", []):
                message[key] = copy.deepcopy(value)

    message.setdefault("role", "assistant")
    message.setdefault("content", "")
    if saw_tool_calls:
        for call in tool_calls:
            call.pop("index", None)
        message["tool_calls"] = tool_calls
    result["message"] = message
    return result


def aggregate_native_generate(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    response_parts: list[str] = []
    thinking_parts: list[str] = []
    logprobs: list[Any] = []
    saw_thinking = False
    saw_logprobs = False

    for chunk in chunks:
        if "error" in chunk:
            raise RuntimeError(str(chunk["error"]))

        for key, value in chunk.items():
            if key == "response" and isinstance(value, str):
                response_parts.append(value)
            elif key == "thinking" and isinstance(value, str):
                saw_thinking = True
                thinking_parts.append(value)
            elif key == "logprobs" and isinstance(value, list):
                saw_logprobs = True
                logprobs.extend(copy.deepcopy(value))
            else:
                result[key] = copy.deepcopy(value)

    result["response"] = "".join(response_parts)
    if saw_thinking:
        result["thinking"] = "".join(thinking_parts)
    if saw_logprobs:
        result["logprobs"] = logprobs
    return result


def aggregate_openai_chat(events: list[tuple[str | None, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"object": "chat.completion"}
    choices: dict[int, dict[str, Any]] = {}

    for _event_name, payload in events:
        if not isinstance(payload, dict):
            continue
        if "error" in payload:
            error = payload["error"]
            raise RuntimeError(error.get("message", str(error)) if isinstance(error, dict) else str(error))

        for key in ("id", "created", "model", "system_fingerprint", "service_tier"):
            if key in payload:
                result[key] = copy.deepcopy(payload[key])
        if payload.get("usage") is not None:
            result["usage"] = copy.deepcopy(payload["usage"])

        for raw_choice in payload.get("choices") or []:
            if not isinstance(raw_choice, dict):
                continue
            index = int(raw_choice.get("index", 0))
            choice = choices.setdefault(
                index,
                {
                    "index": index,
                    "message": {},
                    "finish_reason": None,
                },
            )
            message = choice["message"]
            delta = raw_choice.get("delta")
            if isinstance(delta, dict):
                for key, value in delta.items():
                    if key == "tool_calls" and isinstance(value, list):
                        calls = message.setdefault("tool_calls", [])
                        merge_indexed_objects(calls, [v for v in value if isinstance(v, dict)])
                    elif key == "function_call" and isinstance(value, dict):
                        function_call = message.setdefault("function_call", {})
                        for child_key, child_value in value.items():
                            append_value(function_call, child_key, child_value)
                    else:
                        append_value(message, key, value)

            if raw_choice.get("finish_reason") is not None:
                choice["finish_reason"] = raw_choice["finish_reason"]

            logprobs = raw_choice.get("logprobs")
            if isinstance(logprobs, dict):
                merged = choice.setdefault("logprobs", {})
                for key, value in logprobs.items():
                    append_value(merged, key, value)

    for choice in choices.values():
        message = choice["message"]
        message.setdefault("role", "assistant")
        if "content" not in message:
            message["content"] = ""
        for call in message.get("tool_calls", []):
            if isinstance(call, dict):
                call.pop("index", None)

    result["choices"] = [choices[index] for index in sorted(choices)]
    return result


def aggregate_openai_completion(events: list[tuple[str | None, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"object": "text_completion"}
    choices: dict[int, dict[str, Any]] = {}

    for _event_name, payload in events:
        if not isinstance(payload, dict):
            continue
        if "error" in payload:
            error = payload["error"]
            raise RuntimeError(error.get("message", str(error)) if isinstance(error, dict) else str(error))

        for key in ("id", "created", "model", "system_fingerprint"):
            if key in payload:
                result[key] = copy.deepcopy(payload[key])
        if payload.get("usage") is not None:
            result["usage"] = copy.deepcopy(payload["usage"])

        for raw_choice in payload.get("choices") or []:
            if not isinstance(raw_choice, dict):
                continue
            index = int(raw_choice.get("index", 0))
            choice = choices.setdefault(
                index,
                {"index": index, "text": "", "finish_reason": None},
            )
            if isinstance(raw_choice.get("text"), str):
                choice["text"] += raw_choice["text"]
            if raw_choice.get("finish_reason") is not None:
                choice["finish_reason"] = raw_choice["finish_reason"]
            if isinstance(raw_choice.get("logprobs"), dict):
                merged = choice.setdefault("logprobs", {})
                for key, value in raw_choice["logprobs"].items():
                    append_value(merged, key, value)

    result["choices"] = [choices[index] for index in sorted(choices)]
    return result


def aggregate_openai_responses(events: list[tuple[str | None, Any]]) -> dict[str, Any]:
    # The terminal Responses API event contains the complete response object,
    # which is already the object returned by a non-streaming call.
    for event_name, payload in reversed(events):
        if not isinstance(payload, dict):
            continue
        if "error" in payload:
            error = payload["error"]
            raise RuntimeError(error.get("message", str(error)) if isinstance(error, dict) else str(error))
        response = payload.get("response")
        if isinstance(response, dict) and event_name in {
            "response.completed",
            "response.failed",
            "response.incomplete",
        }:
            return response

    for _event_name, payload in reversed(events):
        if isinstance(payload, dict) and isinstance(payload.get("response"), dict):
            return payload["response"]
    raise ValueError("Responses API stream did not contain a terminal response object")


def aggregate_anthropic_messages(events: list[tuple[str | None, Any]]) -> dict[str, Any]:
    message: dict[str, Any] | None = None
    blocks: dict[int, dict[str, Any]] = {}

    for event_name, payload in events:
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "error" or event_name == "error":
            error = payload.get("error", payload)
            raise RuntimeError(error.get("message", str(error)) if isinstance(error, dict) else str(error))

        event_type = payload.get("type") or event_name
        if event_type == "message_start" and isinstance(payload.get("message"), dict):
            message = copy.deepcopy(payload["message"])
            message["content"] = []
        elif event_type == "content_block_start":
            index = int(payload.get("index", 0))
            block = payload.get("content_block")
            if isinstance(block, dict):
                blocks[index] = copy.deepcopy(block)
        elif event_type == "content_block_delta":
            index = int(payload.get("index", 0))
            block = blocks.setdefault(index, {})
            delta = payload.get("delta")
            if isinstance(delta, dict):
                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    append_value(block, "text", delta.get("text", ""))
                elif delta_type == "thinking_delta":
                    append_value(block, "thinking", delta.get("thinking", ""))
                elif delta_type == "signature_delta":
                    append_value(block, "signature", delta.get("signature", ""))
                elif delta_type == "input_json_delta":
                    append_value(block, "_partial_json", delta.get("partial_json", ""))
                else:
                    for key, value in delta.items():
                        if key != "type":
                            append_value(block, key, value)
        elif event_type == "message_delta":
            if message is None:
                message = {}
            delta = payload.get("delta")
            if isinstance(delta, dict):
                message.update(copy.deepcopy(delta))
            usage = payload.get("usage")
            if isinstance(usage, dict):
                merged_usage = message.setdefault("usage", {})
                merged_usage.update(copy.deepcopy(usage))

    if message is None:
        raise ValueError("Anthropic stream did not contain message_start")

    content: list[dict[str, Any]] = []
    for index in sorted(blocks):
        block = blocks[index]
        partial_json = block.pop("_partial_json", None)
        if isinstance(partial_json, str):
            try:
                block["input"] = json.loads(partial_json) if partial_json else {}
            except json.JSONDecodeError:
                block["input"] = partial_json
        content.append(block)
    message["content"] = content
    return message


def aggregate_streamed_response(
    path: str,
    content_type: str,
    body: bytes,
) -> dict[str, Any]:
    media_type = content_type.split(";", 1)[0].strip().lower()

    if media_type == "text/event-stream" or body.lstrip().startswith((b"data:", b"event:")):
        events = parse_sse(body)
        if path.endswith("/chat/completions"):
            return aggregate_openai_chat(events)
        if path.endswith("/completions"):
            return aggregate_openai_completion(events)
        if path.endswith("/responses"):
            return aggregate_openai_responses(events)
        if path.endswith("/messages"):
            return aggregate_anthropic_messages(events)

        # Best effort for a future/unknown SSE endpoint: use an explicitly
        # supplied complete response, otherwise the last JSON object.
        for _event_name, payload in reversed(events):
            if isinstance(payload, dict) and isinstance(payload.get("response"), dict):
                return payload["response"]
        for _event_name, payload in reversed(events):
            if isinstance(payload, dict):
                return payload
        raise ValueError("upstream returned no JSON SSE events")

    chunks = parse_ndjson(body)
    if path.endswith("/api/chat"):
        return aggregate_native_chat(chunks)
    if path.endswith("/api/generate"):
        return aggregate_native_generate(chunks)

    # Ollama's progress-stream endpoints (/api/pull, /api/push, /api/create,
    # etc.) reduce to their terminal object when streaming is disabled.
    terminal = chunks[-1]
    if "error" in terminal:
        raise RuntimeError(str(terminal["error"]))
    return terminal


# The request body is arbitrary JSON: a top-level "stream" key may appear late,
# after megabytes of model input. This small scanner recognizes a top-level flag
# without parsing or retaining the entire JSON document. It ignores string
# contents and nested objects. Early stream=true allows request-body passthrough.
class StreamFlagProbe:
    def __init__(self) -> None:
        self.depth = 0
        self.string = False
        self.escaped = False
        self.key_bytes: bytearray | None = None
        self.expect_key = False
        self.stream_key = False
        self.awaiting_literal = False
        self.literal = bytearray()
        self.result: bool | None = None

    def feed(self, data: bytes) -> bool | None:
        for ch in data:
            if self.awaiting_literal:
                if ch in b" \t\r\n" and not self.literal:
                    continue
                self.literal.append(ch)
                if self.literal == b"true":
                    self.result = True
                    return True
                if self.literal == b"false":
                    self.result = False
                    return False
                if not (b"true".startswith(self.literal) or b"false".startswith(self.literal)):
                    self.awaiting_literal = False
                else:
                    continue
                # Unexpected stream value: continue structural tracking below.

            if self.string:
                if self.escaped:
                    self.escaped = False
                    if self.key_bytes is not None:
                        self.key_bytes.extend((92, ch))
                elif ch == 92:  # backslash
                    self.escaped = True
                elif ch == 34:  # quote
                    self.string = False
                    if self.key_bytes is not None:
                        try:
                            self.stream_key = (
                                json.loads(b'"' + self.key_bytes + b'"') == "stream"
                            )
                        except (UnicodeDecodeError, ValueError):
                            self.stream_key = False
                        self.key_bytes = None
                        self.expect_key = False
                elif self.key_bytes is not None:
                    # Avoid allocating unbounded memory for a very long key.
                    if len(self.key_bytes) < 256:
                        self.key_bytes.append(ch)
                    else:
                        self.key_bytes = None
                continue

            if ch == 34:
                self.string = True
                if self.depth == 1 and self.expect_key:
                    self.key_bytes = bytearray()
                continue
            if ch in (123, 91):  # { [
                self.depth += 1
                if self.depth == 1 and ch == 123:
                    self.expect_key = True
            elif ch in (125, 93):  # } ]
                self.depth -= 1
            elif self.depth == 1 and ch == 44:  # comma
                self.expect_key = True
                self.stream_key = False
            elif self.depth == 1 and ch == 58 and self.stream_key:  # colon
                self.awaiting_literal = True
                self.literal.clear()
        return None


class RequestTooLarge(Exception):
    pass


class OllamaProxy:
    """Forward HTTP requests, internally stream/rebuild only stream:false."""

    def __init__(
        self,
        upstream: str,
        *,
        pool_limit: int = 0,
        connect_timeout: float = 30,
        max_buffered_request: int = 1024**3,
        spool_memory: int = 2 * 1024**2,
        max_aggregate_response: int = 128 * 1024**2,
        max_transform_request: int = 64 * 1024**2,
    ) -> None:
        self.upstream = upstream.rstrip("/")
        self.pool_limit = pool_limit
        self.connect_timeout = connect_timeout
        self.max_buffered_request = max_buffered_request
        self.spool_memory = spool_memory
        self.max_aggregate_response = max_aggregate_response
        self.max_transform_request = max_transform_request
        self.session: ClientSession | None = None

    async def start(self, _app: web.Application) -> None:
        self.session = ClientSession(
            connector=TCPConnector(limit=self.pool_limit),
            timeout=ClientTimeout(
                total=None,
                connect=None,        # Don't time out only because the pool is busy.
                sock_connect=self.connect_timeout,
                sock_read=None,      # LLM generation may take a long time.
            ),
            cookie_jar=DummyCookieJar(),
            auto_decompress=False,   # Raw compressed bytes stay raw in passthrough.
        )

    async def stop(self, _app: web.Application) -> None:
        if self.session:
            await self.session.close()
            self.session = None

    async def _request_body(self, request: web.Request) -> tuple[bool, Any, Any]:
        """Return (transform_response, payload, owned_spool).

        Try to discover stream:true while the upload is still arriving. Until
        the flag is known, spool the bytes to memory/disk instead of retaining
        unlimited RAM. If stream:true appears, relay the buffered prefix then
        the remaining upload progressively. Otherwise read the completed body
        to decide if stream:false requires conversion.

        For guaranteed request-body streaming even if the JSON stream flag is
        last, send X-Ollama-Proxy-Passthrough: true. This is a proxy-only header.
        """
        if request.headers.get("X-Ollama-Proxy-Passthrough", "").lower() in {"1", "true", "yes"}:
            return False, request.content.iter_chunked(64 * 1024), None

        spool = tempfile.SpooledTemporaryFile(max_size=self.spool_memory)
        size = 0
        probe = StreamFlagProbe()
        owned_by_handler = False

        async def forward_spool(*, remainder: bool):
            spool.seek(0)
            while (part := spool.read(64 * 1024)):
                yield part
            if remainder:
                async for part in request.content.iter_chunked(64 * 1024):
                    yield part

        try:
            async for chunk in request.content.iter_chunked(64 * 1024):
                size += len(chunk)
                if self.max_buffered_request and size > self.max_buffered_request:
                    raise RequestTooLarge(
                        f"Could not determine streaming mode within {self.max_buffered_request} bytes; "
                        "send X-Ollama-Proxy-Passthrough: true for large streaming uploads"
                    )
                spool.write(chunk)
                if probe.feed(chunk) is True:
                    owned_by_handler = True
                    return False, forward_spool(remainder=True), spool
                # On stream:false we need the complete body to modify the JSON.

            if probe.result is not False:
                # If no stream:false was found, no JSON mutation is needed.
                # Large bodies remain on disk; do not read them back into RAM.
                if size <= self.spool_memory:
                    spool.seek(0)
                    return False, spool.read(), None
                owned_by_handler = True
                return False, forward_spool(remainder=False), spool

            if self.max_transform_request and size > self.max_transform_request:
                raise RequestTooLarge(
                    f"stream:false JSON request exceeds transformation limit "
                    f"({self.max_transform_request} bytes)"
                )
            spool.seek(0)
            body = spool.read()
            request_json = parse_json_request(body)
            if request_json is None or request_json.get("stream") is not False:
                return False, body, None

            upstream_json = copy.deepcopy(request_json)
            upstream_json["stream"] = True
            if request.path.endswith(("/chat/completions", "/completions")):
                options = upstream_json.get("stream_options")
                if not isinstance(options, dict):
                    options = {}
                    upstream_json["stream_options"] = options
                options["include_usage"] = True
            return True, compact_json_bytes(upstream_json), None
        finally:
            if not owned_by_handler:
                spool.close()

    async def handle(self, request: web.Request) -> web.StreamResponse:
        assert self.session is not None
        try:
            transformed, upstream_body, owned_spool = await self._request_body(request)
        except RequestTooLarge as exc:
            return web.json_response({"error": str(exc)}, status=413)
        except (ConnectionError, asyncio.CancelledError):
            raise

        LOG.debug(
            "%s %s: %s, inbound Content-Length=%s",
            request.method, request.raw_path,
            "rebuild" if transformed else "passthrough",
            request.content_length,
        )
        headers = filtered_request_headers(request.headers, transformed=transformed)
        headers.popall("X-Ollama-Proxy-Passthrough", None)
        upstream_url = self.upstream + request.raw_path
        forwarding_response = False
        try:
            async with self.session.request(
                request.method,
                upstream_url,
                headers=headers,
                data=upstream_body if upstream_body else None,
                allow_redirects=False,
            ) as upstream_response:
                if transformed:
                    return await self._handle_transformed(request, upstream_response)
                forwarding_response = True
                return await self._handle_passthrough(request, upstream_response)
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            LOG.debug("client disconnected while proxying %s", request.raw_path)
            raise
        except (ClientError, asyncio.TimeoutError, OSError) as exc:
            LOG.warning("upstream failed on %s: %r", request.raw_path, exc)
            if forwarding_response:
                # Headers may already be on the wire. The only honest outcome
                # for a broken stream is to terminate the connection.
                raise
            return web.json_response(
                {"error": f"Ollama upstream request failed: {exc}"}, status=502
            )
        finally:
            # Async generator close releases the spool if upstream refused the
            # request before consuming the complete upload.
            if hasattr(upstream_body, "aclose"):
                with suppress(Exception):
                    await upstream_body.aclose()
            if owned_spool is not None:
                owned_spool.close()

    async def _handle_transformed(
        self, request: web.Request, upstream_response: Any
    ) -> web.Response:
        parts: list[bytes] = []
        body_size = 0
        async for chunk in upstream_response.content.iter_chunked(64 * 1024):
            body_size += len(chunk)
            if self.max_aggregate_response and body_size > self.max_aggregate_response:
                return web.json_response(
                    {"error": "Upstream streaming response exceeds aggregation limit"},
                    status=502,
                )
            parts.append(chunk)
        body = b"".join(parts)
        status = upstream_response.status
        content_type = upstream_response.headers.get("Content-Type", "")

        if status >= 400 or status in (204, 205, 304) or request.method == "HEAD":
            # An error (or no-body response) is already non-streaming: retain
            # its original headers, including content-encoding if any.
            return web.Response(
                status=status,
                reason=upstream_response.reason,
                headers=filtered_response_headers(upstream_response.headers, transformed=False),
                body=body,
            )

        encoding = upstream_response.headers.get("Content-Encoding", "identity").lower()
        try:
            if encoding == "gzip":
                body = gzip.decompress(body)
            elif encoding == "deflate":
                body = zlib.decompress(body)
            elif encoding not in ("identity", ""):
                raise ValueError(f"unsupported upstream Content-Encoding: {encoding}")
        except (OSError, ValueError, zlib.error) as exc:
            return web.json_response(
                {"error": f"Could not decompress upstream response: {exc}"},
                status=502,
            )

        headers = filtered_response_headers(upstream_response.headers, transformed=True)
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type == "application/json":
            # Upstream ignored stream:true and returned a complete JSON object.
            headers["Content-Type"] = content_type
            return web.Response(status=status, headers=headers, body=body)

        try:
            result = aggregate_streamed_response(request.path, content_type, body)
        except RuntimeError as exc:
            LOG.warning("upstream streamed an error: %s", exc)
            return web.json_response({"error": str(exc)}, status=500)
        except (ValueError, TypeError, KeyError) as exc:
            LOG.warning("cannot reconstruct %s: %s", request.path, exc)
            return web.json_response(
                {"error": f"Could not reconstruct Ollama stream: {exc}"},
                status=502,
            )

        return web.Response(
            status=status,
            reason=upstream_response.reason,
            headers=headers,
            body=compact_json_bytes(result),
            content_type="application/json",
        )

    async def _handle_passthrough(
        self, request: web.Request, upstream_response: Any
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            status=upstream_response.status,
            reason=upstream_response.reason,
            headers=filtered_response_headers(upstream_response.headers, transformed=False),
        )
        await response.prepare(request)
        try:
            if request.method != "HEAD":
                async for chunk in upstream_response.content.iter_any():
                    await response.write(chunk)
            await response.write_eof()
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            # Never try to send a new JSON error response after headers.
            LOG.debug("stream interrupted by disconnect: %s", request.raw_path)
            raise
        return response


def build_app(
    upstream: str,
    *,
    pool_limit: int = 0,
    connect_timeout: float = 30,
    max_buffered_request: int = 1024**3,
    max_aggregate_response: int = 128 * 1024**2,
    max_transform_request: int = 64 * 1024**2,
) -> web.Application:
    proxy = OllamaProxy(
        upstream,
        pool_limit=pool_limit,
        connect_timeout=connect_timeout,
        max_buffered_request=max_buffered_request,
        max_aggregate_response=max_aggregate_response,
        max_transform_request=max_transform_request,
    )
    app = web.Application(client_max_size=1024**3)
    app.on_startup.append(proxy.start)
    app.on_cleanup.append(proxy.stop)
    app.router.add_route("*", "/{tail:.*}", proxy.handle)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        default=os.environ.get("OLLAMA_UPSTREAM", "http://127.0.0.1:11434"),
        help="upstream Ollama base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("LISTEN_HOST", "127.0.0.1"),
        help="listen address (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LISTEN_PORT", "11435")),
        help="listen port (default: %(default)s)",
    )
    parser.add_argument(
        "--pool-limit", type=int,
        default=int(os.environ.get("UPSTREAM_POOL_LIMIT", "0")),
        help="maximum simultaneous upstream connections (0 = unlimited)",
    )
    parser.add_argument(
        "--connect-timeout", type=float,
        default=float(os.environ.get("UPSTREAM_CONNECT_TIMEOUT", "30")),
        help="timeout for establishing new upstream sockets, seconds",
    )
    parser.add_argument(
        "--max-buffered-request", type=int,
        default=int(os.environ.get("MAX_BUFFERED_REQUEST", str(1024**3))),
        help="max bytes to spool before stream mode is determined (0 = unlimited)",
    )
    parser.add_argument(
        "--max-aggregate-response", type=int,
        default=int(os.environ.get("MAX_AGGREGATE_RESPONSE", str(128*1024**2))),
        help="maximum upstream bytes for rebuilding a non-streaming response",
    )
    parser.add_argument(
        "--max-transform-request", type=int,
        default=int(os.environ.get("MAX_TRANSFORM_REQUEST", str(64*1024**2))),
        help="maximum size of stream:false request JSON eligible for transformation",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="logging level (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOG.info("proxying http://%s:%d -> %s", args.host, args.port, args.upstream)
    web.run_app(
        build_app(
            args.upstream,
            pool_limit=args.pool_limit,
            connect_timeout=args.connect_timeout,
            max_buffered_request=args.max_buffered_request,
            max_aggregate_response=args.max_aggregate_response,
            max_transform_request=args.max_transform_request,
        ),
        host=args.host,
        port=args.port,
        access_log=LOG,
    )


if __name__ == "__main__":
    main()

