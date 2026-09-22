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
Ollama HTTP proxy with automatic continuation support.

Forwards requests to an Ollama server. When a chat response ends with ":",
automatically sends "Continue" to extend the response seamlessly for the client.

Managed /api/chat requests are streaming-only (NDJSON). Requests that set
"stream": false are rejected explicitly.
Continuation is only applied to /api/chat — it appends assistant + user("Continue") messages.
Non-chat endpoints are forwarded transparently without any special handling.

python3 ollama_proxy.py --listen-port 8080 --target-host 192.168.1.50 --target-port 11434
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import Dict, Optional

import aiohttp
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ollama-proxy")

# ============================================================
# Embedded Loop Detector — detects repeating patterns in streaming text.
# Uses suffix pattern matching: finds all patterns that repeat
# continuously at the END of accumulated text, then picks the best match.
# ============================================================

def find_repeating_suffix_patterns(
    s: str,
    min_repeats: int = 2,
    min_pattern_len: int = 1,
    max_pattern_len: int | None = None,
):
    """
    Find all patterns that repeat continuously at the END of the string.

    Returns:
        list of dicts with keys: pattern, pattern_len, repeats, start, end, repeated_text
    """
    n = len(s)

    if max_pattern_len is None:
        max_pattern_len = n // min_repeats

    results = []

    for pattern_len in range(min_pattern_len, max_pattern_len + 1):
        pattern = s[n - pattern_len:n]

        repeats = 1
        pos = n - pattern_len

        while pos - pattern_len >= 0:
            prev_chunk = s[pos - pattern_len:pos]

            if prev_chunk != pattern:
                break

            repeats += 1
            pos -= pattern_len

        if repeats >= min_repeats:
            results.append({
                "pattern": pattern,
                "pattern_len": pattern_len,
                "repeats": repeats,
                "start": pos,
                "end": n,
                "repeated_text": s[pos:n],
            })

    return results




class LoopDetector:
    """Detects repeating patterns in accumulated streaming text."""

    def __init__(self, min_pattern_len: int = 200, max_pattern_len: int = 10000,
                 repeat_threshold: int = 4):
        self.min_pattern_len = min_pattern_len
        self.max_pattern_len = max(min_pattern_len, max_pattern_len)
        # Window is derived from pattern range and threshold — you need at least
        # max_pattern * repeats chars of text before detection is possible.
        self.window_size = self.max_pattern_len * repeat_threshold
        self.repeat_threshold = repeat_threshold

        self._text: str = ""
        self._detected: bool = False
        self._pattern: Optional[str] = None
        self._results: list[dict] = []

    @property
    def detected(self) -> bool:
        return self._detected

    @property
    def pattern(self) -> Optional[str]:
        return self._pattern

    @property
    def results(self) -> list[dict]:
        return self._results

    @property
    def text_length(self) -> int:
        return len(self._text)

    def add_chunk(self, chunk: str) -> bool:
        if self._detected:
            return True

        if not isinstance(chunk, str) or len(chunk.strip()) == 0:
            return False

        self._text += chunk

        if len(self._text) < self.min_pattern_len * self.repeat_threshold:
            return False

        detected, pattern = self._check_for_loop()
        if detected and pattern is not None:
            self._detected = True
            self._pattern = pattern
            return True

        return False

    def _get_window(self) -> str:
        if len(self._text) <= self.window_size:
            return self._text
        return self._text[-self.window_size:]

    def _check_for_loop(self) -> tuple[bool, Optional[str]]:
        window = self._get_window()

        stripped = window.rstrip()
        if len(stripped) < self.min_pattern_len * self.repeat_threshold:
            return False, None

        results = find_repeating_suffix_patterns(
            stripped,
            min_repeats=self.repeat_threshold,
            min_pattern_len=self.min_pattern_len,
            max_pattern_len=self.max_pattern_len,
        )

        if not results:
            normalized = " ".join(stripped.split())
            if len(normalized) >= self.min_pattern_len * self.repeat_threshold:
                results = find_repeating_suffix_patterns(
                    normalized,
                    min_repeats=self.repeat_threshold,
                    min_pattern_len=self.min_pattern_len,
                    max_pattern_len=self.max_pattern_len,
                )

        if not results:
            return False, None

        self._results = results
        best = max(results, key=lambda r: (r["repeats"], len(r["pattern"])))
        return True, best["pattern"]

    def reset(self):
        self._text = ""
        self._detected = False
        self._pattern = None
        self._results = []

# Endpoints that support chat-style continuation
CHAT_ENDPOINTS = {"/api/chat"}

# Max continuation rounds to prevent infinite loops
MAX_CONTINUATION_ROUNDS = 100

# TODO: instead of is_loop_content and is_loop_thinking just use single variable "continuation_reason"
def _get_continuation_message(is_loop_thinking, is_thinking_budget, is_loop_content, think, content, round_num, thinking_budget=32000):
    m = []

    if is_loop_thinking:
        if content != "":
            m.append({"role": "assistant", "content": content})
        m.append({"role": "user", "content": f"You're stuck in a thinking loop. This is continuation attempt {round_num}. I'm interrupting you. Break out of the repetition and continue with a new direction.\nYour last thoughts (I'm including them here because your thinking history is not preserved):\n{think}"})
        return m
    if is_loop_content:
        m.append({"role": "assistant", "content": content})
        m.append({"role": "user", "content": f"You're repeating yourself in your response. This is continuation attempt {round_num}. I'm interrupting you. Stop the repetition and continue with something new."})
        return m 

    if is_thinking_budget:
        if content != "":
            m.append({"role": "assistant", "content": content})
        m.append({"role": "user", "content": f"""You have already spent more than {thinking_budget:,} characters on reasoning. This is interruption attempt {round_num}.

Stop the current reasoning process and inspect it for loops, repetition, re-evaluating already settled decisions, or repeatedly planning changes without actually applying them.

Do the following now:

Briefly summarize:
what you have established so far;
what decisions you have already made;
what work has actually been completed;
what remains unfinished.
Identify any changes, fixes, commands, edits, patches, or other actions that you previously said you would perform but have not yet performed.
Immediately perform those unfinished actions in this same response. Do not stop after the summary, do not wait for my reply, and do not ask for confirmation unless proceeding is genuinely impossible without missing information.
Do not restart the analysis from scratch. Treat conclusions already reached as settled unless you have found a specific contradiction or error.
Avoid further repetitive reasoning. Spend only the reasoning necessary to complete the remaining work, then provide the actual result.

Your previous hidden reasoning is not preserved, so the most recent portion is included below. Use it as continuity context, not as material to endlessly re-analyze:

{think}"""})
        return m

    # Keywords ("let me" case)
    if round_num == 1:
        if content == "": # think "let me"
            m.append({"role": "user", "content": f"You stopped mid-thought. Now do what you were planning.\nYour last thoughts (included here because your thinking history is not preserved):\n{think}"})
        else:
            m.append({"role": "assistant", "content": content})
            m.append({"role": "user", "content": "You stopped before finishing. Continue and complete what you started."})
        return m
    else:
        if content == "": # think "let me"
            m.append({"role": "user", "content": f"You've stopped again. This is attempt {round_num}. You may be in a thinking loop. Break out of it and produce a concrete answer.\nYour last thoughts (included here because your thinking history is not preserved):\n{think}"})
        else:
            m.append({"role": "assistant", "content": content})
            m.append({"role": "user", "content": f"You've stopped again. This is attempt {round_num}. Stop repeating or stalling and finish your response."})
        return m

    return m

def _check_letme_words(text):
    # Get the last non-empty line (strip trailing whitespace from each line)
    lines = text.splitlines()
    last_line = ""
    for line in reversed(lines):
        stripped = line.rstrip(" \t\n\r")
        if stripped:
            last_line = stripped
            break

    lower = last_line.lower()
    if lower.endswith(":"):
        return True
    if "let me" in lower and "let me know" not in lower:
        return True
    if "now i need" in lower:
        return True
    if "i need to" in lower:
        return True
    if "now i'll" in lower:
        return True
    if "i'll create" in lower:
        return True
    if "i'll make" in lower:
        return True

    return False

def _needs_continuation(cont: str, think: str, has_tool_calls: bool = False, last_ctx_type = None) -> bool:
    """Determine whether the accumulated response text needs a continuation.

    If *has_tool_calls* is True, the response already contains tool calls and
    no continuation is needed.

    Otherwise continuation is triggered when the last non-empty line matches any of:
    - ends with ':' (possibly followed by spaces/newlines)
    - contains 'let me' (case insensitive)
    - contains 'now i need' (case insensitive)
    - contains 'i need to' (case insensitive)
    - contains "now i'll" (case insensitive)
    """
    if has_tool_calls:
        return False

    match last_ctx_type:
        case "think":
            return _check_letme_words(think)
        case "content":
            return _check_letme_words(cont)

    return (_check_letme_words(cont) or _check_letme_words(think))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ollama HTTP proxy with auto-continuation")
    parser.add_argument("--thinking-budget", type=int, default=32000, help="Thinking budget in characters (default: 32000)")
    parser.add_argument("--listen-host", default="0.0.0.0", help="Host to listen on (default: 0.0.0.0)")
    parser.add_argument("--listen-port", type=int, default=11434, help="Port to listen on (default: 11434)")
    parser.add_argument("--target-host", default="localhost", help="Ollama target host (default: localhost)")
    parser.add_argument("--target-port", type=int, default=11434, help="Ollama target port (default: 11434)")
    parser.add_argument(
        "--auto-retries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic retries on connection failures (default: enabled)",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts when auto-retries is enabled (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Delay between retries in seconds (default: 1.0)")
    return parser.parse_args()


def _get_content(chunk: dict, path: str) -> Optional[str]:
    """Extract content string from a response chunk (only /api/chat)."""
    msg = chunk.get("message")
    if isinstance(msg, dict):
        return msg.get("content", "")
    return None


def _get_thinking(chunk: dict, path: str) -> Optional[str]:
    """Extract thinking string from a response chunk (only /api/chat)."""
    if path == "/api/chat":
        msg = chunk.get("message")
        if isinstance(msg, dict):
            return msg.get("thinking", "") or None
    return None



class OllamaProxy:
    def __init__(
        self,
        target_host: str,
        target_port: int,
        auto_retries: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        thinking_budget: int = 32000,
    ):
        self.target_url = f"http://{target_host}:{target_port}"
        self.auto_retries = auto_retries
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.thinking_budget = thinking_budget
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _filter_chat_request_system_messages(self, body):
        j = json.loads(body.decode("utf-8", errors="replace"))
        messages = j['messages']
        if messages[0]['role'] != "system":
            return body
        
        #system_latch = True
        new_messages = []
        new_system = []
        for m in messages:
            if m['role'] == 'system':
                #assert(system_latch)
                new_system.append(m['content'])
            else:
                #system_latch = False
                new_messages.append(m)

        messages = []
        messages.append({"role": "system", "content": "\n".join(new_system)})
        messages.extend(new_messages)

        j['messages'] = messages

        return json.dumps(j).encode("utf-8")

    async def _forward_request(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> aiohttp.ClientResponse:
        """Forward a request to the Ollama target with optional retries."""
        url = f"{self.target_url}{path}"
        last_exc = None
        attempts = self.max_retries if self.auto_retries else 1

        logger.info(f"_forward_request ...{body}")

        for attempt in range(1, attempts + 1):
            try:
                resp = await self.session.request(
                    method,
                    url,
                    data=body,
                    headers={k: v for k, v in dict(headers).items() if k != "Content-Length"} if headers != None else None,
                    timeout=aiohttp.ClientTimeout(total=None),
                )
                return resp
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, OSError) as exc:
                last_exc = exc
                if attempt < attempts:
                    await asyncio.sleep(self.retry_delay)

        raise last_exc or ConnectionError(
            f"Target unreachable after {attempts} attempts"
        )

    async def _cancel_target_generation(
        self, response: aiohttp.ClientResponse, *, timeout: float = 15.0
    ) -> None:
        """Abort one managed internal generation and wait for slot release.

        New managed-proxy builds expose X-Ollama-Proxy-Job-ID and an internal
        cancellation endpoint. If that header is absent (for example when this
        proxy is used against stock Ollama), we still force-close the response
        but cannot obtain a server-side acknowledgement.
        """
        job_id = response.headers.get("X-Ollama-Proxy-Job-ID")
        response.close()
        try:
            await response.wait_for_close()
        except Exception:
            pass

        if not job_id:
            logger.warning(
                "Target did not provide X-Ollama-Proxy-Job-ID; "
                "generation cancellation cannot be acknowledged"
            )
            return

        url = f"{self.target_url}/internal/jobs/{job_id}/cancel"
        try:
            async with self.session.post(
                url, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as cancel_resp:
                body = await cancel_resp.text()
                if cancel_resp.status >= 400:
                    raise RuntimeError(
                        f"cancel job {job_id} failed with HTTP "
                        f"{cancel_resp.status}: {body[:500]}"
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"failed waiting for target job {job_id} cancellation: {exc}"
            ) from exc

        logger.info("Target generation job %s fully cancelled", job_id)

    def _is_chat_endpoint(self, path: str) -> bool:
        return path in CHAT_ENDPOINTS



    async def _stream_from_target(self, response: aiohttp.ClientResponse):
        """Yield (line_str, parsed_chunk_or_None) from a streaming target response."""
        async for line in response.content:
            line_str = line.decode("utf-8", errors="replace")
            if not line_str:
                continue
            try:
                chunk = json.loads(line_str)
            except json.JSONDecodeError:
                chunk = None
            yield line, chunk

    async def _consume_stream(
        self,
        target_resp: aiohttp.ClientResponse,
        client_resp: web.StreamResponse,
        path: str,
    ) -> tuple[str, str, bool, bool, bool, bool, Optional[dict]]:
        """
        Consume a streaming response from target, forward to client.
        Returns (accumulated_content, accumulated_thinking,
                 stream_completed, has_tool_calls, content_loop_detected, thinking_loop_detected, last_chunk).
        stream_completed is True only if we received an explicit done=True chunk.
        has_tool_calls is True if any chunk contained tool_calls in the message.
        """
        accumulated_content = ""
        accumulated_thinking = ""
        stream_completed = False
        has_tool_calls = False
        content_loop_detected = False
        thinking_loop_detected = False
        thinking_budget_exhausted = False
        content_detector = LoopDetector()
        thinking_detector = LoopDetector()
        last_chunk = None
        last_ctx_type = None

        try:
            async for line, chunk in self._stream_from_target(target_resp):
                if chunk is None:
                    await client_resp.write(line)
                    continue

                # Check for tool_calls in /api/chat message
                msg = chunk.get("message")
                if isinstance(msg, dict) and msg.get("tool_calls"):
                    has_tool_calls = True

                content = _get_content(chunk, path)
                if content is not None:
                    if content.rstrip(" \t\n\r"): last_ctx_type = "content"
                    accumulated_content += content
                    if content_detector.add_chunk(content):
                        logger.warning(f"Loop detected in reply: pattern {content_detector.pattern!r} repeated {len(content_detector._results)} times")
                        content_loop_detected = True

                thinking = _get_thinking(chunk, path)
                if thinking is not None:
                    if thinking.rstrip(" \t\n\r"): last_ctx_type = "think"
                    accumulated_thinking += thinking
                    if thinking_detector.add_chunk(thinking):
                        logger.warning(f"Loop detected in thinking: pattern {thinking_detector.pattern!r} repeated {len(thinking_detector._results)} times")
                        thinking_loop_detected = True
                    if len(accumulated_thinking) > self.thinking_budget:
                        thinking_budget_exhausted = True
                        logger.warning(f"Thinking budget exhausted ({self.thinking_budget} chars)")

                if chunk.get("done", False):
                    stream_completed = True
                    last_chunk = chunk
                    break

                await client_resp.write(json.dumps(chunk).encode("utf-8") + b"\r\n")
                if any([thinking_loop_detected, thinking_budget_exhausted, content_loop_detected, stream_completed]):
                    break

        except (aiohttp.ClientConnectionResetError, ConnectionResetError, BrokenPipeError):
            logger.warn( f"(client disconnected): ...{accumulated_content[-1000:]}")
            raise

        logger.info(f"(_consume_stream finished) "
                    f"thinking_loop_detected={thinking_loop_detected}, thinking_budget_exhausted={thinking_budget_exhausted}, content_loop_detected={content_loop_detected}, stream_completed={stream_completed}, last_ctx_type={last_ctx_type}: "
                    f"...{accumulated_thinking[-1000:].replace('\n', ' ')} ...{accumulated_content[-1000:].replace('\n', ' ')}")
        return accumulated_content, accumulated_thinking, stream_completed, has_tool_calls, content_loop_detected, thinking_loop_detected, thinking_budget_exhausted, last_chunk, last_ctx_type

    async def _forward_streaming_response(
        self,
        request: web.Request,
        target_response: aiohttp.ClientResponse,
        path: str,
        original_body: bytes,
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

        decoded_original_body = original_body.decode("utf-8", errors="replace")
        try:
            req_data = json.loads(decoded_original_body)
            messages = req_data["messages"]
            # for m in messages:
            #     for k in list(m):
            #         if k not in [
            #             "role", "content", "images", "tool_calls",
            #             "tool_name", "thinking",
            #         ]:
            #             raise ValueError(f"Unexpected message key: {k!r}")
        except Exception as exc:
            logger.error("Bad request: %s (%s)", decoded_original_body, exc)
            target_response.close()
            raise web.HTTPBadGateway(text="Bad Request")

        round_num = 0
        try:
            while True:
                accumulated, accumulated_thinking, \
                    stream_completed, \
                    has_tool_calls, content_loop_detected, \
                    thinking_loop_detected, thinking_budget_exhausted, \
                    last_chunk, last_ctx_type = await self._consume_stream(
                        target_response, response, path
                    )

                forced_interrupt = (
                    content_loop_detected
                    or thinking_loop_detected
                    or thinking_budget_exhausted
                )
                natural_continuation = (
                    stream_completed
                    and _needs_continuation(
                        accumulated, accumulated_thinking,
                        has_tool_calls, last_ctx_type
                    )
                )
                should_continue = (
                    (forced_interrupt or natural_continuation)
                    and round_num < MAX_CONTINUATION_ROUNDS
                )

                # A budget/loop detector can stop _consume_stream while the target
                # is still generating. Always cancel that live job, even if the
                # continuation limit has already been reached and no next round
                # will be started.
                target_was_cancelled = False
                if forced_interrupt and not stream_completed:
                    await self._cancel_target_generation(target_response)
                    target_was_cancelled = True

                if should_continue:
                    # Natural continuations only happen after done=True, so there
                    # is no live job to cancel in that case.
                    if not target_was_cancelled:
                        target_response.close()
                        try:
                            await target_response.wait_for_close()
                        except Exception:
                            pass

                    round_num += 1
                    logger.info(
                        "Continuation round "
                        "(content_loop_detected=%s thinking_loop_detected=%s "
                        "thinking_budget_exhausted=%s) %s",
                        content_loop_detected, thinking_loop_detected,
                        thinking_budget_exhausted, round_num,
                    )
                    messages.extend(
                        _get_continuation_message(
                            thinking_loop_detected,
                            thinking_budget_exhausted,
                            content_loop_detected,
                            accumulated_thinking,
                            accumulated,
                            round_num,
                            self.thinking_budget,
                        )
                    )
                    logger.info("Message appended: %s", messages[-1])
                    target_response = await self._forward_request(
                        "POST", path, body=json.dumps(req_data).encode()
                    )
                    if target_response.status >= 400:
                        err_body = await target_response.read()
                        target_response.close()
                        raise RuntimeError(
                            f"continuation target returned HTTP {target_response.status}: "
                            f"{err_body.decode('utf-8', errors='replace')[:500]}"
                        )
                    continue

                # No continuation. A natural completion may have a final done
                # chunk withheld by _consume_stream so that we could decide first.
                if last_chunk is not None:
                    await response.write(
                        json.dumps(last_chunk).encode("utf-8") + b"\r\n"
                    )
                await response.write_eof()
                return response

        except asyncio.CancelledError:
            # Server/request-task cancellation must propagate. First terminate the
            # target generation so no detached llama job survives this handler.
            try:
                await asyncio.shield(self._cancel_target_generation(target_response))
            except Exception as exc:
                logger.warning("target cancellation during task cancellation failed: %s", exc)
            raise
        except (
            aiohttp.ClientConnectionResetError,
            ConnectionResetError,
            BrokenPipeError,
        ):
            # The public client is already gone. Abort the target generation and
            # finish this handler quietly; re-raising only makes aiohttp log a
            # spurious server error for a normal disconnect.
            try:
                await self._cancel_target_generation(target_response)
            except Exception as exc:
                logger.warning("target cancellation after client disconnect failed: %s", exc)
            return response
        except Exception as exc:
            logger.exception("forwarding failed: %s", exc)
            try:
                await self._cancel_target_generation(target_response)
            except Exception as cancel_exc:
                logger.warning("target cancellation after forwarding error failed: %s", cancel_exc)
            try:
                await response.write_eof()
            except (ConnectionResetError, RuntimeError):
                pass
            return response
        finally:
            target_response.close()

    async def transparent_forward(self, request: web.Request, body, path: str) -> web.StreamResponse:
        async with self.session.request(
            method=request.method,
            url=f"{self.target_url}{path}",
            headers=request.headers,
            data=body,
            allow_redirects=False,
            auto_decompress=False,
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status,
                reason=upstream.reason,
                headers=upstream.headers,
            )

            await response.prepare(request)

            async for chunk in upstream.content.iter_chunked(32):
                await response.write(chunk)

            await response.write_eof()
            return response

    async def handle_request(self, request: web.Request):
        """Handle an incoming HTTP request and forward it to the Ollama target."""
        path = request.path
        body = await request.read()

        logger.info(f"handle_request entered. path: {path}")

        if not self._is_chat_endpoint(path):
            logger.info(f"Doing transparent")
            return await self.transparent_forward(request, body, path)

        # Continuation handling relies on Ollama's NDJSON stream. Keep the
        # contract explicit so an accidental non-streaming caller fails before
        # starting target generation instead of receiving a malformed response.
        try:
            request_data = json.loads(body.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return web.json_response({"error": "Invalid JSON request body"}, status=400)

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

        # Merge system messages (only for /api/chat which uses messages array)
        body = await self._filter_chat_request_system_messages(body)

        # Forward the request to the target
        try:
            target_response = await self._forward_request(request.method, path, body=body, headers=request.headers)
        except Exception as exc:
            logger.error(f"Target error: {exc}")
            return web.Response(status=502, text=f"Bad Gateway: {exc}")

        if target_response.status >= 400:
            err_body = await target_response.read()
            logger.error(f"Target error {target_response.status}: {err_body.decode('utf-8', errors='replace')}")
            return web.Response(status=target_response.status, body=err_body)

        return await self._forward_streaming_response(request, target_response, path, body)


async def start_proxy(args: argparse.Namespace):
    proxy = OllamaProxy(
        target_host=args.target_host,
        target_port=args.target_port,
        auto_retries=args.auto_retries,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        thinking_budget=args.thinking_budget,
    )

    # Allow request bodies up to 50 MB (default is 1 MB)
    app = web.Application(client_max_size=50 * 1024 * 1024)
    app.router.add_route("*", "/{path:.*}", proxy.handle_request)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, args.listen_host, args.listen_port)
    await site.start()

    logger.info("Ready")

    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await runner.cleanup()
        await proxy.close()


def main():
    args = parse_args()
    try:
        asyncio.run(start_proxy(args))
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()

