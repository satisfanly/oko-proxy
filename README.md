# llm-proxy — Ollama HTTP Proxy with Auto-Continuation & Loop Detection

## Overview

HTTP proxy that forwards requests to an Ollama server. Two key features:

1. **Automatic continuation** — when a response ends mid-sentence (e.g., `Items:`), the proxy automatically sends "Continue" prompts until the model finishes or hits a tool call / max rounds.
2. **Loop detection** — monitors streaming text for repeating suffix patterns and injects a "stuck" message to break infinite loops.

## How Auto-Continuation Works

When the accumulated response text ends with one of these signals, the proxy sends a follow-up prompt:

| Signal | Example |
|--------|---------|
| Ends with `:` | `"Python is:"` |
| Contains `let me` (not "let me know") | `"let me think about this"` |
| Contains `now i need` | `"now i need more details"` |
| Contains `i need to` | `"I need to expand on that"` |
| Contains `i'll` | `"I'll explain further"` |

Continuation prompts:
- Round 1: `"Do it. Don't stop."`
- Round N (N>1): `"You have stopped again. It was already {N} time. Please ensure that you are not in thinking loop."`

For `/api/chat`, continuation appends assistant + user messages to the conversation history. For `/api/generate`, it uses the `context` token array from the done chunk. Tool calls (`tool_calls`) suppress further continuation — they indicate structured output, not incomplete text.

Max rounds: 100 (configurable via `MAX_CONTINUATION_ROUNDS`).

## How Loop Detection Works

The `LoopDetector` class monitors streaming chunks for repeating suffix patterns using suffix pattern matching:

- Scans the accumulated text for all substrings that repeat at the end
- Picks the longest/most-repeated match above configurable thresholds
- Default: min 50 chars per pattern, max 10,000 chars, requires ≥4 consecutive repeats
- Also normalizes whitespace (collapses runs of spaces) before re-checking

When a loop is detected in either content or thinking fields, the proxy sends a `"I seem to be stuck in a loop"` message and stops consuming from the target. The connection stays open for potential recovery.

## Usage

```bash
python3 ollama_proxy.py --listen-port 8080 --target-host 192.168.1.50 --target-port 11434
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--listen-host` | `0.0.0.0` | Host to listen on |
| `--listen-port` | `11434` | Port to listen on |
| `--target-host` | `localhost` | Ollama target host |
| `--target-port` | `11434` | Ollama target port |
| `--auto-retries/--no-auto-retries` | enabled | Retry on connection failures |
| `--max-retries` | `3` | Max retry attempts |
| `--retry-delay` | `1.0` | Seconds between retries |

## Architecture

```
Client → ollama_proxy.py (port 8080) → Ollama server (port 11434)
                                        │
                                        ├─ Auto-continuation loop
                                        ├─ Loop detector (suffix matching)
                                        └─ System message consolidation (/api/chat only)
```

### Key Components

- **`OllamaProxy`** — main proxy class; handles request forwarding, streaming, continuation logic
- **`LoopDetector`** — suffix-pattern-based loop detection on accumulated text
- **`_needs_continuation()`** — heuristic to detect incomplete responses
- **`_filter_chat_request_system_messages()`** — merges multiple system messages into one (only for `/api/chat`)

## Testing

```bash
python3 -m pytest test_ollama_proxy.py -v
# or: python3 test_ollama_proxy.py
```

Covers: content extraction, continuation triggers, streaming/non-streaming flows, retry logic, tool call handling, and history preservation across rounds.
