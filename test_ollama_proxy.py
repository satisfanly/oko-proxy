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
"""Tests for ollama_proxy."""

import asyncio
import collections
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from aiohttp import web
from aiohttp.http import HttpVersion

from ollama_proxy import (
    OllamaProxy,
    _get_content,
    _needs_continuation,
    _get_continuation_message,
    MAX_CONTINUATION_ROUNDS,
    CHAT_ENDPOINTS,
    parse_args,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _AsyncContent:
    """Fake aiohttp response.content that works with `async for`."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        val = self._chunks[self._idx]
        self._idx += 1
        return val


class _AsyncResponse:
    """Minimal fake aiohttp.ClientResponse for testing."""

    def __init__(self, chunks=None, body_bytes=None, status=200):
        self.chunks = chunks
        self.body_bytes = body_bytes
        self.status = status
        self.content = _AsyncContent(chunks) if chunks else _AsyncContent([])

    async def read(self):
        return self.body_bytes

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


class _FakeSession:
    """Fake aiohttp.ClientSession.

    aiohttp.session.request() returns an async context manager synchronously.
    We pre-register a list of responses (or Exceptions) that are returned/raised
    in order on each call to request().
    """

    def __init__(self, responses=None):
        self.closed = False
        self._queue = collections.deque(responses) if responses else collections.deque()

    async def request(self, *args, **kwargs):
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        return item


def _make_proxy(responses=None):
    """Create a proxy whose session.request returns items from *responses* in order."""
    proxy = OllamaProxy("localhost", 11434, auto_retries=True, max_retries=3, retry_delay=0.01)
    proxy._session = _FakeSession(responses)
    return proxy


class AsyncTestCase(unittest.TestCase):
    """Base class for async tests."""

    def async_test(self, coro):
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Synchronous unit tests
# ---------------------------------------------------------------------------

class TestGetContent(unittest.TestCase):

    def test_chat_message_dict(self):
        self.assertEqual(_get_content({"message": {"role": "assistant", "content": "Hi"}}, "/api/chat"), "Hi")

    def test_chat_empty(self):
        self.assertEqual(_get_content({"message": {"role": "assistant", "content": ""}}, "/api/chat"), "")

    def test_chat_none_message(self):
        self.assertIsNone(_get_content({"message": None}, "/api/chat"))

    def test_chat_missing_message(self):
        self.assertIsNone(_get_content({"done": True}, "/api/chat"))

    def test_unknown_path_returns_none(self):
        # /api/generate handling is deprecated; _get_content only handles chat messages now.
        self.assertIsNone(_get_content({"response": "foo"}, "/api/unknown"))


class TestOllamaProxyInit(unittest.TestCase):

    def test_default_url(self):
        self.assertEqual(OllamaProxy("localhost", 11434).target_url, "http://localhost:11434")

    def test_custom_url(self):
        self.assertEqual(OllamaProxy("10.0.0.1", 9000).target_url, "http://10.0.0.1:9000")

    def test_auto_retries_default(self):
        self.assertTrue(OllamaProxy("localhost", 11434).auto_retries)

    def test_auto_retries_disabled(self):
        self.assertFalse(OllamaProxy("localhost", 11434, auto_retries=False).auto_retries)

    def test_max_retries(self):
        self.assertEqual(OllamaProxy("localhost", 11434, max_retries=5).max_retries, 5)

    def test_retry_delay(self):
        self.assertEqual(OllamaProxy("localhost", 11434, retry_delay=2.5).retry_delay, 2.5)


class TestNeedsContinuation(unittest.TestCase):

    def test_ends_with_colon(self):
        self.assertTrue(_needs_continuation("Hello:"))

    def test_ends_with_colon_spaces(self):
        self.assertTrue(_needs_continuation("Hello:   "))

    def test_ends_with_colon_newlines(self):
        self.assertTrue(_needs_continuation("Hello:\n\n"))

    def test_ends_with_colon_spaces_and_newlines(self):
        self.assertTrue(_needs_continuation("Hello:  \n  \n"))

    def test_no_colon(self):
        self.assertFalse(_needs_continuation("Hello world"))

    def test_let_me_lowercase(self):
        self.assertTrue(_needs_continuation("let me think about this"))

    def test_let_me_uppercase(self):
        self.assertTrue(_needs_continuation("LET ME continue"))

    def test_let_me_mixed(self):
        self.assertTrue(_needs_continuation("Let Me explain"))

    def test_let_me_in_middle_no_match(self):
        self.assertFalse(_needs_continuation("let me think\nOkay done."))

    def test_let_me_know_no_continuation(self):
        self.assertFalse(_needs_continuation("let me know if you need anything else"))

    def test_let_me_know_uppercase_no_continuation(self):
        self.assertFalse(_needs_continuation("LET ME KNOW the result"))

    def test_let_me_know_in_middle_no_match(self):
        self.assertFalse(_needs_continuation("let me know\nGot it."))

    def test_now_i_need_lowercase(self):
        self.assertTrue(_needs_continuation("now i need more details"))

    def test_now_i_need_uppercase(self):
        self.assertTrue(_needs_continuation("NOW I NEED to elaborate"))

    def test_now_i_need_mixed(self):
        self.assertTrue(_needs_continuation("Now I Need to continue"))

    def test_now_i_need_in_middle_no_match(self):
        self.assertFalse(_needs_continuation("now i need help\nGot it."))

    def test_i_ll_lowercase(self):
        self.assertTrue(_needs_continuation("i'll explain further"))

    def test_i_ll_uppercase(self):
        self.assertTrue(_needs_continuation("I'LL elaborate on that"))

    def test_i_ll_mixed(self):
        self.assertTrue(_needs_continuation("I'll continue now"))

    def test_i_ll_in_middle_no_match(self):
        self.assertFalse(_needs_continuation("i'll think about it\nOkay done."))

    def test_last_non_empty_line_let_me(self):
        self.assertTrue(_needs_continuation("Some text\n\nlet me elaborate\n\n  \n"))

    def test_last_non_empty_line_now_i_need(self):
        self.assertTrue(_needs_continuation("Previous text\n\nnow i need to expand\n\n"))

    def test_last_non_empty_line_i_ll(self):
        self.assertTrue(_needs_continuation("Previous text\n\ni'll add more\n\n"))

    def test_last_non_empty_line_colon(self):
        self.assertTrue(_needs_continuation("Some text\n\nItems:\n\n"))

    def test_empty_string(self):
        self.assertFalse(_needs_continuation(""))

    def test_only_whitespace(self):
        self.assertFalse(_needs_continuation("   \n\n  "))

    def test_colon_in_middle_no_match(self):
        self.assertFalse(_needs_continuation("Hello: world"))

    def test_tool_calls_no_continuation(self):
        self.assertFalse(_needs_continuation("Hello:", has_tool_calls=True))

    def test_tool_calls_overrides_let_me(self):
        self.assertFalse(_needs_continuation("let me think", has_tool_calls=True))

    def test_tool_calls_overrides_i_ll(self):
        self.assertFalse(_needs_continuation("i'll explain", has_tool_calls=True))

    def test_tool_calls_false_still_continues(self):
        self.assertTrue(_needs_continuation("Hello:", has_tool_calls=False))

    def test_colon_on_last_line_with_trailing_spaces(self):
        self.assertTrue(_needs_continuation("Some text\nItems:   \n\n"))

    def test_colon_on_last_line_with_trailing_newlines(self):
        self.assertTrue(_needs_continuation("Some text\nItems:\n\n"))


class TestIsChatEndpoint(unittest.TestCase):

    def setUp(self):
        self.proxy = OllamaProxy("localhost", 11434)

    def test_chat(self):
        self.assertTrue(self.proxy._is_chat_endpoint("/api/chat"))

    def test_tags(self):
        self.assertFalse(self.proxy._is_chat_endpoint("/api/tags"))

    def test_pull(self):
        self.assertFalse(self.proxy._is_chat_endpoint("/api/pull"))


class TestDetectStreaming(unittest.TestCase):

    def setUp(self):
        self.proxy = OllamaProxy("localhost", 11434)

    def test_ndjson(self):
        resp = MagicMock()
        resp.headers = {"Content-Type": "application/x-ndjson"}
        self.assertTrue(self.proxy._detect_streaming(resp, "/api/chat", b"{}"))

    def test_stream_false(self):
        resp = MagicMock()
        resp.headers = {}
        body = json.dumps({"stream": False, "model": "llama3"}).encode()
        self.assertFalse(self.proxy._detect_streaming(resp, "/api/chat", body))

    def test_stream_true(self):
        resp = MagicMock()
        resp.headers = {}
        body = json.dumps({"stream": True, "model": "llama3"}).encode()
        self.assertTrue(self.proxy._detect_streaming(resp, "/api/chat", body))

    def test_default_streaming(self):
        resp = MagicMock()
        resp.headers = {}
        body = json.dumps({"model": "llama3"}).encode()
        self.assertTrue(self.proxy._detect_streaming(resp, "/api/chat", body))

    def test_non_chat_no_body(self):
        resp = MagicMock()
        resp.headers = {}
        self.assertFalse(self.proxy._detect_streaming(resp, "/api/tags", b""))


class TestGetContinuationMessage(unittest.TestCase):

    def test_loop_message(self):
        msg = _get_continuation_message(True, 1)
        self.assertEqual(msg, "You seem to be stuck in a loop.")

    def test_first_round_message(self):
        msg = _get_continuation_message(False, 1)
        self.assertEqual(msg, "Do it. Don't stop.")

    def test_subsequent_round_message(self):
        msg = _get_continuation_message(False, 3)
        self.assertIn("3", msg)


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------

class TestForwardRequestRetries(AsyncTestCase):

    def test_retries_then_succeeds(self):
        async def test():
            proxy = _make_proxy([
                ConnectionError("refused"),
                ConnectionError("refused"),
                _AsyncResponse(),
            ])
            result = await proxy._forward_request("POST", "/api/chat", body=b"{}")
            self.assertIsInstance(result, _AsyncResponse)

        self.async_test(test())

    def test_gives_up_after_max(self):
        async def test():
            proxy = _make_proxy([
                ConnectionError("refused"),
                ConnectionError("refused"),
                ConnectionError("refused"),
            ])
            with self.assertRaises(ConnectionError):
                await proxy._forward_request("POST", "/api/chat", body=b"{}")

        self.async_test(test())

    def test_no_retry_when_disabled(self):
        async def test():
            proxy = OllamaProxy("localhost", 11434, auto_retries=False)
            proxy._session = _FakeSession([ConnectionError("refused")])
            with self.assertRaises(ConnectionError):
                await proxy._forward_request("POST", "/api/chat", body=b"{}")

        self.async_test(test())


class TestConsumeStream(AsyncTestCase):

    def test_accumulates_chat(self):
        async def test():
            chunks = [
                b'{"message":{"role":"assistant","content":"Hello"},"done":false}\n',
                b'{"message":{"role":"assistant","content":" world"},"done":false}\n',
                b'{"message":{"role":"assistant","content":""},"done":true}\n',
            ]
            target = _AsyncResponse(chunks=chunks)
            client = AsyncMock()
            written = []
            client.write = AsyncMock(side_effect=lambda d: written.append(d))

            proxy = OllamaProxy("localhost", 11434)
            acc, _thinking, stream_completed, has_tool_calls, content_loop_detected, thinking_loop_detected, last_chunk = await proxy._consume_stream(target, client, "/api/chat")
            self.assertEqual(acc, "Hello world")
            self.assertTrue(stream_completed)
            self.assertFalse(has_tool_calls)
            self.assertFalse(content_loop_detected)
            self.assertFalse(thinking_loop_detected)
            # done=True chunk is not written to client (break before write), so only 2 chunks
            self.assertEqual(len(written), 2)

        self.async_test(test())

    def test_forwards_lines(self):
        async def test():
            chunks = [
                b'{"response":"A","done":false}\n',
                b'{"response":"B","done":true}\n',
            ]
            target = _AsyncResponse(chunks=chunks)
            client = AsyncMock()
            written = []
            client.write = AsyncMock(side_effect=lambda d: written.append(d))

            proxy = OllamaProxy("localhost", 11434)
            acc, _thinking, stream_completed, has_tool_calls, content_loop_detected, thinking_loop_detected, last_chunk = await proxy._consume_stream(target, client, "/api/generate")
            # json.dumps adds spaces and \r\n separator; done=True chunk is NOT written (break before write)
            self.assertEqual(len(written), 1)
            self.assertIn(b'"response"', written[0])
            self.assertIn(b'"done": false', written[0])

        self.async_test(test())

    def test_skips_empty_lines(self):
        async def test():
            chunks = [
                b'\n',
                b'{"response":"OK","done":true}\n',
                b'\n',
            ]
            target = _AsyncResponse(chunks=chunks)
            client = AsyncMock()
            written = []
            client.write = AsyncMock(side_effect=lambda d: written.append(d))

            proxy = OllamaProxy("localhost", 11434)
            acc, _thinking, stream_completed, has_tool_calls, content_loop_detected, thinking_loop_detected, last_chunk = await proxy._consume_stream(target, client, "/api/generate")
            # /api/generate chunks have no "message" key, so _get_content returns None and nothing is accumulated
            self.assertEqual(acc, "")
            # The raw line is forwarded since chunk has no message content to extract
            self.assertEqual(len(written), 1)

        self.async_test(test())

    def test_tool_calls_detected_in_stream(self):
        async def test():
            chunks = [
                b'{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","function":{"name":"cmd_runner","arguments":"{}"}}]},"done":false}\n',
                b'{"message":{"role":"assistant","content":""},"done":true}\n',
            ]
            target = _AsyncResponse(chunks=chunks)
            client = AsyncMock()
            client.write = AsyncMock()

            proxy = OllamaProxy("localhost", 11434)
            acc, _thinking, stream_completed, has_tool_calls, content_loop_detected, thinking_loop_detected, last_chunk = await proxy._consume_stream(target, client, "/api/chat")
            self.assertTrue(has_tool_calls)
            self.assertTrue(stream_completed)

        self.async_test(test())





class TestContinuationNonStreaming(AsyncTestCase):

    def test_merges_content(self):
        async def test():
            orig_data = {"model": "llama3", "messages": [{"role": "user", "content": "List"}]}
            response_data = {"message": {"role": "assistant", "content": "Item 1:"}}

            proxy = _make_proxy([_AsyncResponse(body_bytes=json.dumps({
                "message": {"role": "assistant", "content": " Item 2, Item 3."},
                "done": True,
            }).encode())])

            merged = await proxy._do_continuation_non_streaming(
                "/api/chat", orig_data, response_data, "Item 1:", None,
            )
            self.assertEqual(merged, "Item 1: Item 2, Item 3.")

        self.async_test(test())

    def test_recursive(self):
        async def test():
            orig_data = {"model": "llama3", "messages": [{"role": "user", "content": "List"}]}
            response_data = {}

            proxy = _make_proxy([
                _AsyncResponse(body_bytes=json.dumps({
                    "message": {"role": "assistant", "content": " more:"},
                    "done": True,
                }).encode()),
                _AsyncResponse(body_bytes=json.dumps({
                    "message": {"role": "assistant", "content": " done."},
                    "done": True,
                }).encode()),
            ])

            merged = await proxy._do_continuation_non_streaming(
                "/api/chat", orig_data, response_data, "start:", None,
            )
            self.assertEqual(merged, "start: more: done.")

        self.async_test(test())

    def test_continuation_history_preserved_non_streaming(self):
        """Verify that on the 2nd continuation round (non-streaming), the LLM
        sees the full conversation including the 1st continuation exchange."""
        async def test():
            orig_data = {"model": "llama3", "messages": [{"role": "user", "content": "Tell me"}]}
            response_data = {}

            # Capture the request body sent on each continuation round
            sent_bodies = []

            async def tracking_forward(self, method, path, body=None, headers=None):
                sent_bodies.append(json.loads(body))
                resp = self.session._queue.popleft()
                return resp

            responses = [
                # Round 1 response: triggers another continuation
                _AsyncResponse(body_bytes=json.dumps({
                    "message": {"role": "assistant", "content": " part1:"},
                    "done": True,
                }).encode()),
                # Round 2 response: final
                _AsyncResponse(body_bytes=json.dumps({
                    "message": {"role": "assistant", "content": " part2 done."},
                    "done": True,
                }).encode()),
            ]

            proxy = _make_proxy(responses)
            proxy._forward_request = lambda *a, **kw: tracking_forward(proxy, *a, **kw)

            merged = await proxy._do_continuation_non_streaming(
                "/api/chat", orig_data, response_data, "intro:", None,
            )

            # Two continuation requests should have been sent
            self.assertEqual(len(sent_bodies), 2)

            # Round 1: original messages + assistant("intro:") + user("Do it...")
            round1_msgs = sent_bodies[0]["messages"]
            self.assertEqual(round1_msgs[0], {"role": "user", "content": "Tell me"})
            self.assertEqual(round1_msgs[1], {"role": "assistant", "content": "intro:"})
            self.assertEqual(round1_msgs[2], {"role": "user", "content": "Do it. Don't stop."})

            # Round 2: original messages + round1 assistant + round1 user
            #                          + assistant(" part1:") + user("Do it...")
            round2_msgs = sent_bodies[1]["messages"]
            self.assertEqual(round2_msgs[0], {"role": "user", "content": "Tell me"})
            self.assertEqual(round2_msgs[1], {"role": "assistant", "content": "intro:"})
            self.assertEqual(round2_msgs[2], {"role": "user", "content": "Do it. Don't stop."})
            self.assertEqual(round2_msgs[3], {"role": "assistant", "content": " part1:"})
            self.assertEqual(round2_msgs[4], {"role": "user", "content": "You have stopped again. It was already 2 time. Please ensure that you are not in thinking loop."})

            self.assertEqual(merged, "intro: part1: part2 done.")

        self.async_test(test())

    def test_tool_calls_skips_continuation_non_streaming(self):
        async def test():
            orig_data = {"model": "llama3", "messages": [{"role": "user", "content": "Do something"}]}
            response_data = {}

            proxy = _make_proxy([])  # No continuation responses needed

            merged = await proxy._do_continuation_non_streaming(
                "/api/chat", orig_data, response_data, "I will use a tool:", has_tool_calls=True,
            )
            # Should not continue because has_tool_calls=True
            self.assertEqual(merged, "I will use a tool:")

        self.async_test(test())


class TestHandleNonStreaming(AsyncTestCase):

    def test_forwards_non_colon(self):
        async def test():
            resp = _AsyncResponse(body_bytes=json.dumps({
                "message": {"role": "assistant", "content": "Hello!"},
                "done": True,
            }).encode())

            proxy = OllamaProxy("localhost", 11434)
            result = await proxy._handle_non_streaming(MagicMock(), resp, "/api/chat", b'{}')
            self.assertEqual(result.status, 200)
            data = json.loads(result.body)
            self.assertEqual(data["message"]["content"], "Hello!")

        self.async_test(test())

    def test_continuation_merged(self):
        async def test():
            resp = _AsyncResponse(body_bytes=json.dumps({
                "message": {"role": "assistant", "content": "Items:"},
                "done": True,
            }).encode())

            proxy = _make_proxy([_AsyncResponse(body_bytes=json.dumps({
                "message": {"role": "assistant", "content": " A, B, C."},
                "done": True,
            }).encode())])

            orig_body = json.dumps({"model": "llama3", "messages": [{"role": "user", "content": "List"}]}).encode()
            result = await proxy._handle_non_streaming(MagicMock(), resp, "/api/chat", orig_body)
            self.assertEqual(result.status, 200)
            data = json.loads(result.body)
            self.assertEqual(data["message"]["content"], "Items: A, B, C.")

        self.async_test(test())

    def test_tool_calls_no_continuation(self):
        async def test():
            resp = _AsyncResponse(body_bytes=json.dumps({
                "message": {
                    "role": "assistant",
                    "content": "I will use a tool:",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "cmd_runner", "arguments": "{\"cmd\": \"ls\"}"}}
                    ],
                },
                "done": True,
            }).encode())

            proxy = _make_proxy([])  # No continuation responses should be consumed

            orig_body = json.dumps({"model": "llama3", "messages": [{"role": "user", "content": "Do something"}]}).encode()
            result = await proxy._handle_non_streaming(MagicMock(), resp, "/api/chat", orig_body)
            self.assertEqual(result.status, 200)
            data = json.loads(result.body)
            # Content should be unchanged - no continuation happened
            self.assertEqual(data["message"]["content"], "I will use a tool:")
            # No continuation requests were made
            self.assertEqual(len(proxy.session._queue), 0)

        self.async_test(test())


class TestStreamFromTarget(AsyncTestCase):

    def test_yields_parsed(self):
        async def test():
            chunks = [
                b'{"response":"A","done":false}\n',
                b'{"response":"B","done":true}\n',
            ]
            target = _AsyncResponse(chunks=chunks)
            proxy = OllamaProxy("localhost", 11434)

            results = []
            async for line_str, chunk in proxy._stream_from_target(target):
                results.append((line_str, chunk))

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0][1]["response"], "A")
            self.assertEqual(results[1][1]["response"], "B")

        self.async_test(test())

    def test_invalid_json_yields_none(self):
        async def test():
            chunks = [
                b'not json\n',
                b'{"response":"OK","done":true}\n',
            ]
            target = _AsyncResponse(chunks=chunks)
            proxy = OllamaProxy("localhost", 11434)

            results = []
            async for line_str, chunk in proxy._stream_from_target(target):
                results.append((line_str, chunk))

            self.assertEqual(len(results), 2)
            self.assertIsNone(results[0][1])
            self.assertIsNotNone(results[1][1])

        self.async_test(test())





if __name__ == "__main__":
    unittest.main()
