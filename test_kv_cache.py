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
Test Ollama KV cache slot save/restore mechanism.

Scenario:
1. Request model A with long context (should compute and cache)
2. Request model B with small context (should evict A's cache, save slot)
3. Request model A with same long context again (should restore cache, be fast)

If save/restore is broken, step 3 will be as slow as step 1.
"""

import sys
import time
import json
import urllib.request
import urllib.error


def generate(model, prompt, ollama_addr, stream=False):
    """Send a generate request to Ollama and return (response_text, elapsed_seconds)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "num_predict": 10,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{ollama_addr}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            if stream:
                # For streaming, just consume all chunks
                full = ""
                for line in resp:
                    chunk = json.loads(line.decode("utf-8"))
                    full += chunk.get("response", "")
                    if chunk.get("done"):
                        break
                elapsed = time.monotonic() - start
                return full, elapsed
            else:
                result = json.loads(resp.read().decode("utf-8"))
                elapsed = time.monotonic() - start
                return result.get("response", ""), elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.monotonic() - start
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        raise


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <modelA> <modelB> <ollama_addr>")
        print(f"Example: {sys.argv[0]} llama3.1:8b qwen2.5:0.5b http://localhost:11434")
        sys.exit(1)

    model_a = sys.argv[1]
    model_b = sys.argv[2]
    ollama_addr = sys.argv[3].rstrip("/")

    # Long context for model A (~2000 tokens worth of text)
    long_context = "The quick brown fox jumps over the lazy dog. " * 200
    long_prompt = long_context + "\n\nSummarize the above in one word."

    # Small context for model B
    small_prompt = "Say hello."

    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print(f"Ollama:  {ollama_addr}")
    print(f"Long context length: {len(long_context)} chars")
    print()

    # Step 1: Model A with long context (cold, computes KV cache)
    print("[1] Model A, long context (cold cache)...")
    _, t1 = generate(model_a, long_prompt, ollama_addr)
    print(f"    Time: {t1:.2f}s")
    print()

    # Step 2: Model A with same long context (should be fast, confirms KV cache works)
    print("[2] Model A, same long context (KV cache hit, no eviction)...")
    _, t2 = generate(model_a, long_prompt, ollama_addr)
    print(f"    Time: {t2:.2f}s")
    print()

    # Step 3: Model B with small context (evicts A's cache)
    print("[3] Model B, small context (evicts A)...")
    _, t3 = generate(model_b, small_prompt, ollama_addr)
    print(f"    Time: {t3:.2f}s")
    print()

    # Step 4: Model A with same long context (should restore cache)
    print("[4] Model A, same long context (should restore cache)...")
    _, t4 = generate(model_a, long_prompt, ollama_addr)
    print(f"    Time: {t4:.2f}s")
    print()

    # Analysis
    print("=" * 60)
    print("Results:")
    print(f"  Step 1 (A cold):           {t1:.2f}s")
    print(f"  Step 2 (A cache hit):      {t2:.2f}s")
    print(f"  Step 3 (B small, evicts):  {t3:.2f}s")
    print(f"  Step 4 (A restored):       {t4:.2f}s")
    print()

    # Check 1: Does KV cache work at all? (step 2 should be much faster than step 1)
    if t2 < t1 * 0.5:
        print(f"KV cache works: Step 2 is {t1/t2:.1f}x faster than Step 1.")
    else:
        print(f"WARNING: KV cache may not be working at all!")
        print(f"         Step 2 ({t2:.2f}s) is not significantly faster than Step 1 ({t1:.2f}s).")
    print()

    # Check 2: Does save/restore work? (step 4 should be close to step 2)
    if t4 < t1 * 0.5:
        print("PASS: Step 4 is significantly faster than Step 1.")
        print("       KV cache save/restore appears to be working.")
        print(f"       Speedup vs cold: {t1/t4:.1f}x")
    elif t4 < t1 * 0.8:
        print("PARTIAL: Step 4 is somewhat faster than Step 1.")
        print("         Cache may be partially restored.")
        print(f"         Speedup vs cold: {t1/t4:.1f}x")
    else:
        print("FAIL: Step 4 is about as slow as Step 1.")
        print("      KV cache was likely NOT restored (recomputed from scratch).")
        print(f"      Ratio vs cold: {t4/t1:.2f}x (close to 1.0 = no cache benefit)")


if __name__ == "__main__":
    main()
