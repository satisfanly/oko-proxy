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
"""Streaming Ollama proxy that recovers from repeated assistant messages.

For ``/api/chat``, the proxy compares each completed assistant response with
assistant messages already present in the request context.  If the current
response makes the number of identical occurrences exceed ``--max-repeats``,
the proxy keeps the client NDJSON connection open and starts another internal
chat turn.  That turn tells the model what it repeated, asks it to explain the
repetition briefly, and requires it to break the loop and complete the task.

Equality covers the complete logical assistant message: content, attached
reasoning/thinking fields, tool calls, and every other non-empty message field.
Tool-call chunks are held until classification so a repeated tool call is not
released to the client and executed before the corrective turn can replace it.
Non-chat endpoints are forwarded without special handling.

Example:

    python3 ollama_proxy_crossctx.py \
        --listen-host 127.0.0.1 --listen-port 45043 \
        --target-host 127.0.0.1 --target-port 45649 \
        --max-repeats 2

With ``--max-repeats 2``, two total occurrences are allowed; the third
identical assistant response triggers a corrective turn.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from copy import deepcopy
from collections.abc import AsyncIterator, Mapping
from typing import Any

import aiohttp
from aiohttp import web


CHAT_ENDPOINT = "/api/chat"
MAX_CORRECTION_ROUNDS = 100
REPEATED_TEXT_PREVIEW_CHARS = 4_000

# These fields arrive as text deltas in streaming Ollama-compatible responses.
# Supporting the common aliases keeps reasoning attached to the message used
# for equality rather than comparing only the user-visible answer.
TEXT_DELTA_FIELDS = {
    "content",
    "thinking",
    "reasoning",
    "reasoning_content",
    "analysis",
}

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ollama-message-repeat-proxy")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Streaming Ollama proxy that detects complete assistant messages "
            "repeated from the supplied chat context"
        )
    )
    parser.add_argument(
        "--listen-host",
        default="127.0.0.1",
        help="Host to listen on (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=45043,
        help="Port to listen on (default: 45043)",
    )
    parser.add_argument(
        "--target-host",
        default="127.0.0.1",
        help="Ollama target host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--target-port",
        type=int,
        default=45649,
        help="Ollama target port (default: 45649)",
    )
    parser.add_argument(
        "--max-repeats",
        type=positive_int,
        required=True,
        metavar="NUMBER",
        help=(
            "Maximum allowed total occurrences of an identical complete "
            "assistant message; the next occurrence triggers a corrective turn"
        ),
    )
    parser.add_argument(
        "--auto-retries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry target connection failures automatically (default: enabled)",
    )
    parser.add_argument(
        "--max-retries",
        type=positive_int,
        default=3,
        help="Maximum target connection attempts (default: 3)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Delay between target connection attempts in seconds (default: 1.0)",
    )
    return parser.parse_args()


def filtered_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Copy end-to-end request headers and let aiohttp set framing/Host."""
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
        and key.lower() not in {"content-length", "host"}
    }


def filtered_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def canonical_assistant_message(message: Any) -> dict[str, Any] | None:
    """Return the complete semantic assistant message used for equality.

    Empty top-level transport fields are omitted so a missing optional field
    and the same field emitted as ``""``/``[]`` by the stream are equivalent.
    Nested values, including tool arguments, are preserved exactly.
    """
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None

    canonical: dict[str, Any] = {"role": "assistant"}
    for key, value in message.items():
        if key == "role" or value is None or value == "":
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        canonical[key] = deepcopy(value)
    return canonical


def message_has_payload(message: dict[str, Any] | None) -> bool:
    return bool(message and any(key != "role" for key in message))


def count_prior_identical(
    messages: list[Any],
    current_message: dict[str, Any],
) -> int:
    """Count exact complete-message matches already present in context."""
    return sum(
        1
        for message in messages
        if canonical_assistant_message(message) == current_message
    )


def repeated_message_preview(message: dict[str, Any]) -> str:
    rendered = json.dumps(
        message,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    if len(rendered) <= REPEATED_TEXT_PREVIEW_CHARS:
        return rendered

    half = REPEATED_TEXT_PREVIEW_CHARS // 2
    omitted = len(rendered) - (half * 2)
    return (
        rendered[:half]
        + f"\n\n[... {omitted} characters omitted from this preview ...]\n\n"
        + rendered[-half:]
    )


def build_correction_prompt(
    message: dict[str, Any],
    *,
    prior_identical: int,
    total_occurrences: int,
    max_repeats: int,
    correction_round: int,
) -> str:
    preview = repeated_message_preview(message)
    return f"""Your immediately preceding assistant message repeated an earlier assistant message exactly, including its response content, attached reasoning, tool calls, and other message fields.

It matched {prior_identical} earlier assistant message(s) in this conversation and has now appeared {total_occurrences} times. The configured maximum is {max_repeats}. This is corrective attempt {correction_round}.

The repeated complete assistant message was:
<repeated_assistant_message>
{preview}
</repeated_assistant_message>

First, briefly explain why you repeated the same response instead of advancing the conversation. Then break out of the loop and directly continue the user's unresolved task with materially new content.

Do not repeat the message above again. Do not stop after apologizing or describing a plan: perform the unfinished work and provide the concrete result in this same response."""


class AssistantMessageAccumulator:
    """Reconstruct one complete assistant message from streaming deltas."""

    def __init__(self) -> None:
        self._message: dict[str, Any] = {"role": "assistant"}

    def add_chunk(self, chunk: dict[str, Any]) -> None:
        message = chunk.get("message")
        if not isinstance(message, dict):
            return

        for key, value in message.items():
            if key == "role" or value is None:
                continue

            if key in TEXT_DELTA_FIELDS and isinstance(value, str):
                existing = self._message.get(key, "")
                if not isinstance(existing, str):
                    existing = ""
                self._message[key] = existing + value
                continue

            if isinstance(value, list):
                existing = self._message.get(key)
                if existing == value:
                    # Some servers repeat a complete list in the final chunk.
                    continue
                if not isinstance(existing, list):
                    self._message[key] = deepcopy(value)
                else:
                    existing.extend(deepcopy(value))
                continue

            existing = self._message.get(key)
            if existing != value:
                self._message[key] = deepcopy(value)

    def complete_message(self) -> dict[str, Any]:
        canonical = canonical_assistant_message(self._message)
        return canonical or {"role": "assistant"}


def final_chunk_has_visible_message_data(chunk: dict[str, Any]) -> bool:
    message = chunk.get("message")
    return message_has_payload(canonical_assistant_message(message))


def nonfinal_message_chunk(
    chunk: dict[str, Any],
    *,
    include_tool_calls: bool = True,
) -> dict[str, Any] | None:
    """Retain final-chunk message data without ending the client stream."""
    message = chunk.get("message")
    if not isinstance(message, dict):
        return None

    forwarded_message = deepcopy(message)
    if not include_tool_calls:
        forwarded_message.pop("tool_calls", None)
    if not message_has_payload(canonical_assistant_message(forwarded_message)):
        return None

    forwarded: dict[str, Any] = {
        key: chunk[key]
        for key in ("model", "created_at")
        if key in chunk
    }
    forwarded["message"] = forwarded_message
    forwarded["done"] = False
    return forwarded


def split_tool_calls_from_chunk(
    chunk: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Separate tool calls so they can be withheld until repeat detection."""
    message = chunk.get("message")
    if not isinstance(message, dict) or not message.get("tool_calls"):
        return chunk, None

    public_chunk = deepcopy(chunk)
    public_message = public_chunk["message"]
    tool_calls = public_message.pop("tool_calls")

    if not message_has_payload(canonical_assistant_message(public_message)):
        public_chunk = None

    tool_chunk: dict[str, Any] = {
        key: chunk[key]
        for key in ("model", "created_at")
        if key in chunk
    }
    tool_chunk["message"] = {
        "role": "assistant",
        "tool_calls": deepcopy(tool_calls),
    }
    tool_chunk["done"] = False
    return public_chunk, tool_chunk


class OllamaMessageRepeatProxy:
    def __init__(
        self,
        *,
        target_host: str,
        target_port: int,
        max_repeats: int,
        auto_retries: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.target_url = f"http://{target_host}:{target_port}"
        self.max_repeats = max_repeats
        self.auto_retries = auto_retries
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _forward_request(
        self,
        method: str,
        path_qs: str,
        *,
        body: bytes | None,
        headers: Mapping[str, str],
    ) -> aiohttp.ClientResponse:
        url = f"{self.target_url}{path_qs}"
        attempts = self.max_retries if self.auto_retries else 1
        last_exception: BaseException | None = None

        for attempt in range(1, attempts + 1):
            try:
                return await self.session.request(
                    method,
                    url,
                    data=body,
                    headers=filtered_request_headers(headers),
                    timeout=aiohttp.ClientTimeout(total=None),
                    auto_decompress=False,
                )
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                ConnectionError,
                OSError,
            ) as exc:
                last_exception = exc
                if attempt < attempts:
                    logger.warning(
                        "Target connection attempt %d/%d failed: %s",
                        attempt,
                        attempts,
                        exc,
                    )
                    await asyncio.sleep(self.retry_delay)

        if last_exception is not None:
            raise last_exception
        raise ConnectionError(f"Target unreachable after {attempts} attempts")

    async def _cancel_target_generation(
        self,
        response: aiohttp.ClientResponse,
        *,
        timeout: float = 15.0,
    ) -> None:
        """Close the target response and acknowledge cancellation when supported."""
        job_id = response.headers.get("X-Ollama-Proxy-Job-ID")
        response.close()
        try:
            await response.wait_for_close()
        except Exception:
            pass

        if not job_id:
            return

        url = f"{self.target_url}/internal/jobs/{job_id}/cancel"
        try:
            async with self.session.post(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as cancel_response:
                response_text = await cancel_response.text()
                if cancel_response.status >= 400:
                    raise RuntimeError(
                        f"cancel job {job_id} failed with HTTP "
                        f"{cancel_response.status}: {response_text[:500]}"
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"failed waiting for target job {job_id} cancellation: {exc}"
            ) from exc

        logger.info("Target generation job %s fully cancelled", job_id)

    async def _stream_lines(
        self,
        target_response: aiohttp.ClientResponse,
    ) -> AsyncIterator[tuple[bytes, dict[str, Any] | None]]:
        async for line in target_response.content:
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                parsed = None
            yield line, parsed if isinstance(parsed, dict) else None

    async def _consume_generation(
        self,
        target_response: aiohttp.ClientResponse,
        client_response: web.StreamResponse,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any] | None,
        list[dict[str, Any]],
    ]:
        """Forward one generation while withholding done/tool-call chunks."""
        message_accumulator = AssistantMessageAccumulator()
        final_chunk: dict[str, Any] | None = None
        buffered_tool_call_chunks: list[dict[str, Any]] = []

        async for raw_line, chunk in self._stream_lines(target_response):
            if chunk is None:
                await client_response.write(raw_line)
                continue

            message_accumulator.add_chunk(chunk)

            if chunk.get("done") is True:
                final_chunk = chunk
                break

            public_chunk, tool_call_chunk = split_tool_calls_from_chunk(chunk)
            if tool_call_chunk is not None:
                buffered_tool_call_chunks.append(tool_call_chunk)
            if public_chunk is not None:
                await client_response.write(
                    json.dumps(public_chunk, ensure_ascii=False).encode("utf-8")
                    + b"\r\n"
                )

        return (
            message_accumulator.complete_message(),
            final_chunk,
            buffered_tool_call_chunks,
        )

    async def _write_chunk(
        self,
        response: web.StreamResponse,
        chunk: dict[str, Any],
    ) -> None:
        await response.write(
            json.dumps(chunk, ensure_ascii=False).encode("utf-8") + b"\r\n"
        )

    async def _write_stream_error(
        self,
        response: web.StreamResponse,
        message: str,
    ) -> None:
        try:
            await self._write_chunk(response, {"error": message, "done": True})
            await response.write_eof()
        except (ConnectionResetError, RuntimeError):
            pass

    async def _forward_chat_stream(
        self,
        request: web.Request,
        target_response: aiohttp.ClientResponse,
        *,
        path_qs: str,
        request_data: dict[str, Any],
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            status=target_response.status,
            headers={
                "Content-Type": "application/x-ndjson",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        messages = request_data["messages"]
        correction_round = 0

        try:
            while True:
                current_message, final_chunk, buffered_tool_call_chunks = (
                    await self._consume_generation(target_response, response)
                )

                prior_identical = (
                    count_prior_identical(messages, current_message)
                    if message_has_payload(current_message)
                    else 0
                )
                total_occurrences = prior_identical + 1
                repeated_too_many_times = (
                    message_has_payload(current_message)
                    and total_occurrences > self.max_repeats
                )

                if (
                    repeated_too_many_times
                    and correction_round < MAX_CORRECTION_ROUNDS
                ):
                    correction_round += 1
                    logger.warning(
                        "Repeated assistant response detected: "
                        "%d prior match(es), %d total occurrence(s), limit=%d; "
                        "starting corrective turn %d",
                        prior_identical,
                        total_occurrences,
                        self.max_repeats,
                        correction_round,
                    )

                    # A done chunk normally has empty message data.  If a target
                    # places visible data in it, forward that data without ending
                    # the public stream before starting the corrective turn.
                    if (
                        final_chunk is not None
                        and final_chunk_has_visible_message_data(final_chunk)
                    ):
                        visible_final_chunk = nonfinal_message_chunk(
                            final_chunk,
                            include_tool_calls=False,
                        )
                        if visible_final_chunk is not None:
                            await self._write_chunk(response, visible_final_chunk)

                    if buffered_tool_call_chunks or (
                        final_chunk is not None
                        and isinstance(final_chunk.get("message"), dict)
                        and final_chunk["message"].get("tool_calls")
                    ):
                        logger.warning(
                            "Suppressing tool calls from repeated assistant message"
                        )

                    messages.append(deepcopy(current_message))
                    messages.append(
                        {
                            "role": "user",
                            "content": build_correction_prompt(
                                current_message,
                                prior_identical=prior_identical,
                                total_occurrences=total_occurrences,
                                max_repeats=self.max_repeats,
                                correction_round=correction_round,
                            ),
                        }
                    )
                    request_data["messages"] = messages
                    request_data["stream"] = True

                    target_response.close()
                    try:
                        await target_response.wait_for_close()
                    except Exception:
                        pass

                    try:
                        target_response = await self._forward_request(
                            request.method,
                            path_qs,
                            body=json.dumps(
                                request_data,
                                ensure_ascii=False,
                            ).encode("utf-8"),
                            headers=request.headers,
                        )
                    except Exception as exc:
                        logger.exception("Corrective target request failed")
                        await self._write_stream_error(
                            response,
                            f"Corrective target request failed: {exc}",
                        )
                        return response

                    if target_response.status >= 400:
                        error_body = await target_response.read()
                        error_text = error_body.decode("utf-8", errors="replace")
                        target_response.close()
                        await self._write_stream_error(
                            response,
                            "Corrective target returned HTTP "
                            f"{target_response.status}: {error_text[:500]}",
                        )
                        return response

                    continue

                if repeated_too_many_times:
                    logger.error(
                        "Maximum corrective rounds reached; closing with the "
                        "target's final chunk"
                    )

                for tool_call_chunk in buffered_tool_call_chunks:
                    await self._write_chunk(response, tool_call_chunk)
                if final_chunk is not None:
                    await self._write_chunk(response, final_chunk)
                else:
                    await self._write_chunk(
                        response,
                        {
                            "error": (
                                "Target stream ended without a final done=true chunk"
                            ),
                            "done": True,
                        },
                    )
                await response.write_eof()
                return response

        except asyncio.CancelledError:
            try:
                await asyncio.shield(
                    self._cancel_target_generation(target_response)
                )
            except Exception as exc:
                logger.warning(
                    "Target cancellation during task cancellation failed: %s",
                    exc,
                )
            raise
        except (
            aiohttp.ClientConnectionResetError,
            ConnectionResetError,
            BrokenPipeError,
        ):
            logger.info("Client disconnected; cancelling target generation")
            try:
                await self._cancel_target_generation(target_response)
            except Exception as exc:
                logger.warning(
                    "Target cancellation after client disconnect failed: %s",
                    exc,
                )
            return response
        except Exception as exc:
            logger.exception("Streaming failed: %s", exc)
            try:
                await self._cancel_target_generation(target_response)
            except Exception as cancel_exc:
                logger.warning(
                    "Target cancellation after streaming failure failed: %s",
                    cancel_exc,
                )
            await self._write_stream_error(response, f"Proxy streaming failure: {exc}")
            return response
        finally:
            target_response.close()

    async def _transparent_forward(
        self,
        request: web.Request,
        body: bytes,
    ) -> web.StreamResponse:
        async with self.session.request(
            method=request.method,
            url=f"{self.target_url}{request.path_qs}",
            headers=filtered_request_headers(request.headers),
            data=body,
            allow_redirects=False,
            auto_decompress=False,
            timeout=aiohttp.ClientTimeout(total=None),
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status,
                reason=upstream.reason,
                headers=filtered_response_headers(upstream.headers),
            )
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(64 * 1024):
                await response.write(chunk)
            await response.write_eof()
            return response

    async def handle_request(self, request: web.Request) -> web.StreamResponse:
        body = await request.read()
        logger.info("%s %s", request.method, request.path_qs)

        if request.path != CHAT_ENDPOINT:
            return await self._transparent_forward(request, body)

        try:
            request_data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return web.json_response(
                {"error": "Invalid JSON request body"},
                status=400,
            )

        if not isinstance(request_data, dict):
            return web.json_response(
                {"error": "The /api/chat request body must be a JSON object"},
                status=400,
            )

        if request_data.get("stream") is False:
            return web.json_response(
                {
                    "error": (
                        "Non-streaming /api/chat requests are not supported; "
                        "use \"stream\": true"
                    )
                },
                status=400,
            )

        messages = request_data.get("messages")
        if not isinstance(messages, list):
            return web.json_response(
                {"error": "The /api/chat request must contain a messages array"},
                status=400,
            )

        request_data["stream"] = True
        upstream_body = json.dumps(
            request_data,
            ensure_ascii=False,
        ).encode("utf-8")

        try:
            target_response = await self._forward_request(
                request.method,
                request.path_qs,
                body=upstream_body,
                headers=request.headers,
            )
        except Exception as exc:
            logger.error("Target connection failed: %s", exc)
            return web.json_response(
                {"error": f"Bad Gateway: {exc}"},
                status=502,
            )

        if target_response.status >= 400:
            error_body = await target_response.read()
            response_headers = filtered_response_headers(target_response.headers)
            target_response.close()
            return web.Response(
                status=target_response.status,
                body=error_body,
                headers=response_headers,
            )

        return await self._forward_chat_stream(
            request,
            target_response,
            path_qs=request.path_qs,
            request_data=request_data,
        )


async def start_proxy(args: argparse.Namespace) -> None:
    proxy = OllamaMessageRepeatProxy(
        target_host=args.target_host,
        target_port=args.target_port,
        max_repeats=args.max_repeats,
        auto_retries=args.auto_retries,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    app = web.Application(client_max_size=50 * 1024 * 1024)
    app.router.add_route("*", "/{path:.*}", proxy.handle_request)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, args.listen_host, args.listen_port)
    await site.start()

    logger.info(
        "Listening on %s:%d; forwarding to %s; max_repeats=%d",
        args.listen_host,
        args.listen_port,
        proxy.target_url,
        args.max_repeats,
    )

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        await proxy.close()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(start_proxy(args))
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()

