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
"""Passive plaintext Ollama HTTP/1.x sniffer.

The response parser can attach in the middle of an already-running chunked
response. It does not require seeing the initial ``HTTP/1.1 200`` headers.

Examples:
    sudo python3 -u ollama_wire_sniffer.py -i lo -p 11435
    sudo python3 -u ollama_wire_sniffer.py -i lo -p 11435 --requests
    python3 ollama_wire_sniffer.py --self-test
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable


HTTP_METHOD_RE = re.compile(
    rb"(?:^|\r\n)(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|CONNECT|TRACE) "
    rb"[^ \r\n]+ HTTP/1\.[01]\r\n"
)
CHUNK_LINE_RE = re.compile(rb"^([0-9A-Fa-f]{1,16})(?:;[^\r\n]*)?\r\n")
CHUNK_BOUNDARY_RE = re.compile(
    rb"(?:^|\r\n)([0-9A-Fa-f]{1,16})(?:;[^\r\n]*)?\r\n"
)


def eprint(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class StreamPrinter:
    def __init__(self, label: str, raw_json: bool = False) -> None:
        self.label = label
        self.raw_json = raw_json
        self.shown = False
        self.section_name: str | None = None

    def _show(self) -> None:
        if not self.shown:
            print(f"\n--- Ollama stream {self.label} ---", flush=True)
            self.shown = True

    def _section(self, name: str) -> None:
        self._show()
        if self.section_name != name:
            if self.section_name is not None:
                print(flush=True)
            print(f"[{name}]", flush=True)
            self.section_name = name

    def text(self, name: str, value: Any) -> bool:
        if not isinstance(value, str) or value == "":
            return False
        self._section(name)
        print(value, end="", flush=True)
        return True

    def value(self, name: str, value: Any) -> bool:
        if value in (None, "", [], {}):
            return False
        self._section(name)
        if isinstance(value, str):
            print(value, flush=True)
        else:
            print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
        return True

    def event(self, obj: dict[str, Any]) -> None:
        if self.raw_json:
            self._show()
            print(compact_json(obj), flush=True)
            if obj.get("done") is True:
                self.finish()
            return

        if "error" in obj:
            self.value("error", obj["error"])

        emitted = False

        # Ollama native /api/generate.
        emitted |= self.text("thinking", obj.get("thinking"))
        emitted |= self.text("answer", obj.get("response"))
        emitted |= self.value("tool_calls", obj.get("tool_calls"))

        # Ollama native /api/chat.
        message = obj.get("message")
        if isinstance(message, dict):
            emitted |= self.text("thinking", message.get("thinking"))
            emitted |= self.text("answer", message.get("content"))
            emitted |= self.value("tool_calls", message.get("tool_calls"))

        # OpenAI-compatible chat/completions.
        choices = obj.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                emitted |= self.text("answer", choice.get("text"))
                for key in ("delta", "message"):
                    delta = choice.get(key)
                    if not isinstance(delta, dict):
                        continue
                    for reasoning_key in (
                        "thinking",
                        "reasoning",
                        "reasoning_content",
                        "reasoning_text",
                    ):
                        emitted |= self.text("thinking", delta.get(reasoning_key))
                    content = delta.get("content")
                    if isinstance(content, str):
                        emitted |= self.text("answer", content)
                    elif isinstance(content, list):
                        for part in content:
                            if not isinstance(part, dict):
                                continue
                            part_type = part.get("type")
                            part_text = part.get("text") or part.get("content")
                            if part_type in ("text", "output_text"):
                                emitted |= self.text("answer", part_text)
                            elif part_type in ("thinking", "reasoning"):
                                emitted |= self.text("thinking", part_text)
                            else:
                                emitted |= self.value("content_part", part)
                    emitted |= self.value("tool_calls", delta.get("tool_calls"))
                    emitted |= self.value("function_call", delta.get("function_call"))

        # OpenAI Responses API.
        event_type = obj.get("type")
        if isinstance(event_type, str) and event_type.startswith("response."):
            if event_type in ("response.output_text.delta", "response.refusal.delta"):
                emitted |= self.text("answer", obj.get("delta"))
            elif event_type in (
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            ):
                emitted |= self.text("thinking", obj.get("delta"))
            elif "function_call" in event_type or "tool_call" in event_type:
                emitted |= self.value(event_type, obj)

        # Anthropic-compatible SSE.
        if event_type == "content_block_start":
            block = obj.get("content_block")
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type in ("tool_use", "server_tool_use"):
                    emitted |= self.value("tool_call", block)
                elif block_type in ("thinking", "redacted_thinking"):
                    emitted |= self.value("thinking", block)
        elif event_type == "content_block_delta":
            delta = obj.get("delta")
            if isinstance(delta, dict):
                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    emitted |= self.text("answer", delta.get("text"))
                elif delta_type in ("thinking_delta", "signature_delta"):
                    emitted |= self.text(
                        "thinking", delta.get("thinking") or delta.get("signature")
                    )
                elif delta_type == "input_json_delta":
                    emitted |= self.value("tool_call_arguments", delta)

        # Unknown compatible schemas: retain named fields.
        if not emitted:
            for key in ("thinking", "reasoning", "reasoning_content", "reasoning_text"):
                emitted |= self.text("thinking", obj.get(key))
            for key in ("tool_calls", "tool_call", "function_call"):
                emitted |= self.value(key, obj.get(key))

        if obj.get("done") is True:
            self.finish()

    def finish(self, reason: str = "end") -> None:
        if self.shown:
            if self.section_name is not None:
                print(flush=True)
            print(f"--- {reason} ---", flush=True)
        self.shown = False
        self.section_name = None


class EventLineDecoder:
    """Decode Ollama NDJSON and SSE data lines from decoded HTTP body bytes."""

    def __init__(self, printer: StreamPrinter, debug: Callable[[str], None]) -> None:
        self.printer = printer
        self.debug = debug
        self.buf = bytearray()

    def reset(self) -> None:
        self.buf.clear()

    def feed(self, data: bytes) -> None:
        self.buf.extend(data)
        while True:
            pos = self.buf.find(b"\n")
            if pos < 0:
                return
            raw = bytes(self.buf[:pos]).rstrip(b"\r")
            del self.buf[: pos + 1]
            line = raw.strip()
            if not line:
                continue

            # Defensive fallback when fed undecoded chunk framing.
            if re.fullmatch(rb"[0-9A-Fa-f]{1,16}(?:;[^\r\n]*)?", line):
                continue
            if line.startswith((b"HTTP/", b"Content-", b"Transfer-Encoding:")):
                continue
            if line.startswith((b":", b"event:")):
                continue
            if line.startswith(b"data:"):
                line = line[5:].strip()
                if line == b"[DONE]":
                    self.printer.finish()
                    continue

            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self.debug(f"ignored non-JSON body line: {line[:160]!r} ({exc})")
                continue
            if isinstance(obj, dict):
                self.printer.event(obj)
            elif self.printer.raw_json:
                print(compact_json(obj), flush=True)


class ResponseParser:
    """Incremental HTTP response parser that can synchronize mid-stream."""

    def __init__(self, label: str, raw_json: bool, debug: Callable[[str], None]) -> None:
        self.label = label
        self.debug = debug
        self.printer = StreamPrinter(label, raw_json)
        self.lines = EventLineDecoder(self.printer, debug)
        self.buf = bytearray()
        self.mode = "unknown"  # unknown, headers, chunked, raw
        self.chunk_remaining: int | None = None

    def gap(self) -> None:
        self.debug(f"{self.label}: TCP gap; resetting response framing")
        self.buf.clear()
        self.lines.reset()
        self.mode = "unknown"
        self.chunk_remaining = None

    def close(self) -> None:
        self.printer.finish("connection closed")

    @staticmethod
    def _find_chunk_boundary(data: bytes | bytearray) -> int | None:
        match = CHUNK_BOUNDARY_RE.search(bytes(data))
        if match is None:
            return None
        # If the regex consumed a leading CRLF, point after that CRLF.
        return match.start(1)

    def feed(self, data: bytes) -> None:
        self.buf.extend(data)
        while True:
            if self.mode == "unknown":
                http_at = self.buf.find(b"HTTP/")
                if http_at >= 0:
                    if http_at:
                        del self.buf[:http_at]
                    self.mode = "headers"
                    continue

                # Critical behavior: synchronize directly on a chunk-size line.
                # This handles captures that begin with e.g. b"38c\r\n{...}".
                if CHUNK_LINE_RE.match(bytes(self.buf)):
                    self.debug(f"{self.label}: attached at mid-stream HTTP chunk")
                    self.mode = "chunked"
                    continue

                boundary = self._find_chunk_boundary(self.buf)
                if boundary is not None and boundary > 0:
                    self.debug(
                        f"{self.label}: discarded {boundary} byte(s) before chunk boundary"
                    )
                    del self.buf[:boundary]
                    self.mode = "chunked"
                    continue

                # A non-chunked NDJSON/SSE response, or capture beginning inside
                # body data. Once a complete line exists, decode it directly.
                if b"\n" in self.buf:
                    self.mode = "raw"
                    continue
                return

            if self.mode == "headers":
                end = self.buf.find(b"\r\n\r\n")
                if end < 0:
                    return
                raw_headers = bytes(self.buf[:end])
                del self.buf[: end + 4]
                lower = raw_headers.lower()
                status = raw_headers.split(b"\r\n", 1)[0].decode("latin-1", "replace")
                self.debug(f"{self.label}: {status}")
                if b"transfer-encoding: chunked" in lower:
                    self.mode = "chunked"
                else:
                    self.mode = "raw"
                continue

            if self.mode == "raw":
                if not self.buf:
                    return
                data_now = bytes(self.buf)
                self.buf.clear()
                self.lines.feed(data_now)
                return

            if self.mode == "chunked":
                if self.chunk_remaining is None:
                    end = self.buf.find(b"\r\n")
                    if end < 0:
                        return
                    size_line = bytes(self.buf[:end]).split(b";", 1)[0].strip()
                    try:
                        size = int(size_line, 16)
                    except ValueError:
                        # Search for the next real chunk boundary instead of
                        # waiting forever or requiring an HTTP status line.
                        boundary = self._find_chunk_boundary(self.buf[1:])
                        if boundary is None:
                            if len(self.buf) > 1 << 20:
                                del self.buf[:-64]
                            return
                        boundary += 1
                        self.debug(
                            f"{self.label}: invalid chunk line {size_line!r}; "
                            f"resynchronized {boundary} byte(s) later"
                        )
                        del self.buf[:boundary]
                        continue

                    del self.buf[: end + 2]
                    if size == 0:
                        # Consume trailers if they are already available.
                        if self.buf.startswith(b"\r\n"):
                            del self.buf[:2]
                        else:
                            trailer_end = self.buf.find(b"\r\n\r\n")
                            if trailer_end >= 0:
                                del self.buf[: trailer_end + 4]
                        self.printer.finish()
                        self.mode = "unknown"
                        continue
                    self.chunk_remaining = size
                    self.debug(f"{self.label}: chunk size={size}")

                size = self.chunk_remaining
                assert size is not None
                if len(self.buf) < size + 2:
                    return
                payload = bytes(self.buf[:size])
                suffix = bytes(self.buf[size : size + 2])
                if suffix != b"\r\n":
                    self.debug(f"{self.label}: malformed chunk terminator; resynchronizing")
                    del self.buf[:1]
                    self.chunk_remaining = None
                    self.mode = "unknown"
                    continue
                del self.buf[: size + 2]
                self.chunk_remaining = None
                self.lines.feed(payload)
                continue


class RequestParser:
    """Incremental HTTP/1.x request parser; JSON schema is never filtered."""

    def __init__(self, label: str, debug: Callable[[str], None]) -> None:
        self.label = label
        self.debug = debug
        self.buf = bytearray()
        self.state = "headers"
        self.request_line = ""
        self.headers: dict[str, str] = {}
        self.body = bytearray()
        self.remaining = 0
        self.chunk_remaining: int | None = None

    def gap(self) -> None:
        self.debug(f"{self.label}: TCP gap; abandoning incomplete request")
        self.buf.clear()
        self._reset_message()

    def _reset_message(self) -> None:
        self.state = "headers"
        self.request_line = ""
        self.headers = {}
        self.body.clear()
        self.remaining = 0
        self.chunk_remaining = None

    @staticmethod
    def _decode_body(body: bytes, headers: dict[str, str]) -> bytes:
        encoding = headers.get("content-encoding", "").lower()
        try:
            if "gzip" in encoding:
                return gzip.decompress(body)
            if "deflate" in encoding:
                return zlib.decompress(body)
        except (OSError, zlib.error):
            pass
        return body

    def _print(self) -> None:
        body = self._decode_body(bytes(self.body), self.headers)
        print(f"\n--- Ollama request {self.label} ---", flush=True)
        print(self.request_line, flush=True)
        if not body:
            print("(no body)", flush=True)
        else:
            try:
                text = body.decode("utf-8")
                value = json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError):
                try:
                    print(body.decode("utf-8", "replace"), flush=True)
                except Exception:
                    print(repr(body), flush=True)
            else:
                # Entire object is printed. Thinking history, tool definitions,
                # assistant tool_calls, and role=tool responses are untouched.
                print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
        print("--- end request ---", flush=True)

    def _find_start(self) -> int | None:
        match = HTTP_METHOD_RE.search(bytes(self.buf))
        if match is None:
            return None
        # Skip a preceding CRLF consumed by the regex.
        return match.start(1)

    def feed(self, data: bytes) -> None:
        self.buf.extend(data)
        while True:
            if self.state == "headers":
                start = self._find_start()
                if start is None:
                    if len(self.buf) > 1 << 20:
                        del self.buf[:-32]
                    return
                if start:
                    del self.buf[:start]
                end = self.buf.find(b"\r\n\r\n")
                if end < 0:
                    return
                block = bytes(self.buf[:end])
                del self.buf[: end + 4]
                lines = block.split(b"\r\n")
                self.request_line = lines[0].decode("latin-1", "replace")
                headers: dict[str, str] = {}
                for line in lines[1:]:
                    if b":" not in line:
                        continue
                    name, value = line.split(b":", 1)
                    headers[name.decode("latin-1").strip().lower()] = (
                        value.decode("latin-1", "replace").strip()
                    )
                self.headers = headers
                transfer = headers.get("transfer-encoding", "").lower()
                if "chunked" in transfer:
                    self.state = "chunked"
                    continue
                try:
                    self.remaining = max(0, int(headers.get("content-length", "0")))
                except ValueError:
                    self.remaining = 0
                if self.remaining == 0:
                    self._print()
                    self._reset_message()
                    continue
                self.state = "fixed"
                continue

            if self.state == "fixed":
                if len(self.buf) < self.remaining:
                    return
                self.body.extend(self.buf[: self.remaining])
                del self.buf[: self.remaining]
                self._print()
                self._reset_message()
                continue

            if self.state == "chunked":
                if self.chunk_remaining is None:
                    end = self.buf.find(b"\r\n")
                    if end < 0:
                        return
                    size_line = bytes(self.buf[:end]).split(b";", 1)[0].strip()
                    try:
                        size = int(size_line, 16)
                    except ValueError:
                        self.debug(f"{self.label}: invalid request chunk size; resync")
                        del self.buf[:1]
                        self._reset_message()
                        continue
                    del self.buf[: end + 2]
                    if size == 0:
                        if self.buf.startswith(b"\r\n"):
                            del self.buf[:2]
                        else:
                            trailer_end = self.buf.find(b"\r\n\r\n")
                            if trailer_end < 0:
                                return
                            del self.buf[: trailer_end + 4]
                        self._print()
                        self._reset_message()
                        continue
                    self.chunk_remaining = size
                size = self.chunk_remaining
                assert size is not None
                if len(self.buf) < size + 2:
                    return
                if self.buf[size : size + 2] != b"\r\n":
                    self.debug(f"{self.label}: malformed request chunk; resync")
                    del self.buf[:1]
                    self._reset_message()
                    continue
                self.body.extend(self.buf[:size])
                del self.buf[: size + 2]
                self.chunk_remaining = None
                continue


@dataclass
class DirectionFlow:
    parser: Any
    last_end: int | None = None

    def feed(self, seq: int, payload: bytes) -> None:
        """Process every observed payload and never block forever on a gap."""
        if not payload:
            return
        if self.last_end is None:
            self.last_end = seq + len(payload)
            self.parser.feed(payload)
            return
        if seq < self.last_end:
            overlap = self.last_end - seq
            if overlap >= len(payload):
                return  # retransmission/duplicate
            payload = payload[overlap:]
            seq += overlap
        elif seq > self.last_end:
            self.parser.gap()
        self.last_end = seq + len(payload)
        self.parser.feed(payload)


def run_self_test() -> int:
    events: list[dict[str, Any]] = []

    class TestPrinter(StreamPrinter):
        def event(self, obj: dict[str, Any]) -> None:
            events.append(obj)

    debug = lambda _msg: None
    response = ResponseParser("self-test", False, debug)
    response.printer = TestPrinter("self-test", False)
    response.lines = EventLineDecoder(response.printer, debug)

    obj1 = {
        "model": "drts",
        "message": {"role": "assistant", "content": "", "thinking": " France"},
        "done": False,
    }
    obj2 = {
        "model": "drts",
        "message": {"role": "assistant", "content": "", "thinking": '?"'},
        "done": False,
    }
    body1 = compact_json(obj1).encode() + b"\n"
    body2 = compact_json(obj2).encode() + b"\n"
    wire = (
        f"{len(body1):x}\r\n".encode() + body1 + b"\r\n"
        + f"{len(body2):x}\r\n".encode() + body2 + b"\r\n"
    )
    # Feed in awkward splits and without HTTP headers, exactly the reported case.
    for cut in (3, 11, 47, 89, len(wire)):
        part, wire = wire[:cut], wire[cut:]
        if part:
            response.feed(part)
        if not wire:
            break
    if wire:
        response.feed(wire)

    if events != [obj1, obj2]:
        eprint(f"SELF-TEST FAILED: decoded {events!r}")
        return 1

    request_seen: list[dict[str, Any]] = []

    class TestRequest(RequestParser):
        def _print(self) -> None:
            value = json.loads(bytes(self.body))
            request_seen.append(value)

    request = TestRequest("self-test", debug)
    request_obj = {
        "model": "drts",
        "messages": [
            {"role": "assistant", "thinking": "x", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
        ],
        "stream": True,
    }
    request_body = compact_json(request_obj).encode()
    request_wire = (
        b"POST /api/chat HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        + f"Content-Length: {len(request_body)}\r\n".encode()
        + b"Content-Type: application/json\r\n\r\n"
        + request_body
    )
    request.feed(request_wire[:37])
    request.feed(request_wire[37:])
    if request_seen != [request_obj]:
        eprint(f"SELF-TEST FAILED: request decoded {request_seen!r}")
        return 1

    print("SELF-TEST PASS: mid-stream chunked responses and tool-result requests")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sniff plaintext Ollama requests and streaming responses"
    )
    ap.add_argument("-i", "--iface", default="lo", help="capture interface")
    ap.add_argument("-p", "--port", type=int, default=11436, help="Ollama TCP port")
    ap.add_argument("--requests", action="store_true", help="pretty-print HTTP requests")
    ap.add_argument("--json", action="store_true", help="print every response JSON event")
    ap.add_argument("--debug", action="store_true", help="print packet/parser diagnostics")
    ap.add_argument("--self-test", action="store_true", help="test parsers without Scapy/root")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    try:
        from scapy.all import IP, IPv6, TCP, Raw, sniff
    except ImportError:
        ap.error("Scapy is required: python3 -m pip install scapy")

    def debug(message: str) -> None:
        if args.debug:
            eprint(f"[debug] {message}")

    responses: dict[tuple[Any, ...], DirectionFlow] = {}
    requests: dict[tuple[Any, ...], DirectionFlow] = {}

    def endpoints(packet: Any) -> tuple[str, str] | None:
        if IP in packet:
            return packet[IP].src, packet[IP].dst
        if IPv6 in packet:
            return packet[IPv6].src, packet[IPv6].dst
        return None

    def handle(packet: Any) -> None:
        if TCP not in packet:
            return
        pair = endpoints(packet)
        if pair is None:
            return
        src, dst = pair
        tcp = packet[TCP]
        sport, dport = int(tcp.sport), int(tcp.dport)
        if sport != args.port and dport != args.port:
            return

        if sport == args.port:
            response_key = (src, sport, dst, dport)
            request_key = (dst, dport, src, sport)
        else:
            response_key = (dst, dport, src, sport)
            request_key = (src, sport, dst, dport)

        flags = int(tcp.flags)
        if flags & 0x02:  # SYN
            old = responses.pop(response_key, None)
            if old is not None:
                old.parser.close()
            requests.pop(request_key, None)

        payload = bytes(tcp[Raw].load) if Raw in tcp else bytes(tcp.payload)
        if payload:
            direction = "response" if sport == args.port else "request"
            debug(
                f"{direction} {src}:{sport} -> {dst}:{dport} "
                f"seq={int(tcp.seq)} len={len(payload)} prefix={payload[:48]!r}"
            )

            if sport == args.port:
                flow = responses.get(response_key)
                if flow is None:
                    label = f"{src}:{sport} -> {dst}:{dport}"
                    flow = DirectionFlow(ResponseParser(label, args.json, debug))
                    responses[response_key] = flow
                flow.feed(int(tcp.seq), payload)
            elif args.requests:
                flow = requests.get(request_key)
                if flow is None:
                    label = f"{src}:{sport} -> {dst}:{dport}"
                    flow = DirectionFlow(RequestParser(label, debug))
                    requests[request_key] = flow
                flow.feed(int(tcp.seq), payload)

        if flags & 0x05:  # FIN or RST
            flow = responses.pop(response_key, None)
            if flow is not None:
                flow.parser.close()
            requests.pop(request_key, None)

    eprint(
        f"Sniffing plaintext tcp/{args.port} on {args.iface}"
        f"{' with requests' if args.requests else ''}; Ctrl-C to stop"
    )
    sniff(
        iface=args.iface,
        filter=f"tcp port {args.port}",
        prn=handle,
        store=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
