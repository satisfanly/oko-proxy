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
Test Ollama KV cache eviction with same model, different prompts.

Scenario:
1. Request model with big prompt (cold cache, computes KV)
2. Request same model with same big prompt (should be fast, cache hit)
3. Request same model with small prompt (big cache preserved in RAM, not dropped)
4. Request same model with big prompt again (should be medium: slower than 2/5
   because cache needs to be reloaded/activated, but faster than 1 because it's
   preserved in RAM, not recomputed)
5. Request same model with same big prompt again (should be fast, cache hit)

Expected timing: t2 ≈ t5 < t4 < t1

If cache is dropped entirely, step 4 will be as slow as step 1.
If cache is not preserved at all, step 4 will be as fast as step 2.
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
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <model> <ollama_addr>")
        print(f"Example: {sys.argv[0]} llama3.1:8b http://localhost:11434")
        sys.exit(1)

    model = sys.argv[1]
    ollama_addr = sys.argv[2].rstrip("/")

    # Big prompt (~2000 tokens worth of text)
    big_context = "The quick brown fox jumps over the lazy dog. " * 200
    big_prompt = big_context + "\n\nSummarize the above in one word."

    # Small prompt to evict the big prompt's cache
    small_prompt = "Say hello."

    print(f"Model:  {model}")
    print(f"Ollama: {ollama_addr}")
    print(f"Big prompt length: {len(big_prompt)} chars")
    print(f"Small prompt length: {len(small_prompt)} chars")
    print()

    # Step 1: Big prompt (cold cache)
    print("[1] Big prompt (cold cache)...")
    _, t1 = generate(model, big_prompt, ollama_addr)
    print(f"    Time: {t1:.2f}s")
    print()

    # Step 2: Same big prompt (should be fast, cache hit)
    print("[2] Same big prompt (cache hit)...")
    _, t2 = generate(model, big_prompt, ollama_addr)
    print(f"    Time: {t2:.2f}s")
    print()

    # Step 3: Small prompt (big cache should be preserved in RAM)
    print("[3] Small prompt (big cache preserved in RAM)...")
    _, t3 = generate(model, small_prompt, ollama_addr)
    print(f"    Time: {t3:.2f}s")
    print()

    # Step 4: Big prompt again (should be medium: cache reloaded from RAM)
    print("[4] Big prompt again (cache reloaded from RAM)...")
    _, t4 = generate(model, big_prompt, ollama_addr)
    print(f"    Time: {t4:.2f}s")
    print()

    # Step 5: Same big prompt again (should be fast, cache hit)
    print("[5] Same big prompt again (cache hit)...")
    _, t5 = generate(model, big_prompt, ollama_addr)
    print(f"    Time: {t5:.2f}s")
    print()

    # Analysis
    print("=" * 60)
    print("Results:")
    print(f"  Step 1 (big, cold):        {t1:.2f}s")
    print(f"  Step 2 (big, cache hit):   {t2:.2f}s")
    print(f"  Step 3 (small, evict):     {t3:.2f}s")
    print(f"  Step 4 (big, evicted):     {t4:.2f}s")
    print(f"  Step 5 (big, rebuilt):     {t5:.2f}s")
    print()

    # Check 1: Does KV cache work at all? (step 2 should be much faster than step 1)
    if t2 < t1 * 0.5:
        print(f"KV cache works: Step 2 is {t1/t2:.1f}x faster than Step 1.")
    else:
        print(f"WARNING: KV cache may not be working at all!")
        print(f"         Step 2 ({t2:.2f}s) is not significantly faster than Step 1 ({t1:.2f}s).")
    print()

    # Check 2: Was the cache preserved in RAM? (step 4 should be medium)
    # Expected: t2 < t4 < t1 (slower than cache hit, faster than cold compute)
    if t4 > t2 * 1.2 and t4 < t1 * 0.9:
        print("PASS: Step 4 is medium (slower than Step 2, faster than Step 1).")
        print("       Cache was preserved in RAM and reloaded.")
        print(f"       Ratio vs cold: {t4/t1:.2f}x, vs cache hit: {t4/t2:.2f}x")
    elif t4 > t1 * 0.8:
        print("FAIL: Step 4 is slow (similar to Step 1).")
        print("      Cache was dropped entirely, not preserved in RAM.")
        print(f"      Ratio vs cold: {t4/t1:.2f}x (close to 1.0 = recomputed)")
    elif t4 < t2 * 1.2:
        print("FAIL: Step 4 is fast (similar to Step 2).")
        print("      Cache was never deactivated (no reload needed).")
        print(f"      Ratio vs cache hit: {t4/t2:.2f}x (close to 1.0 = still active)")
    else:
        print("PARTIAL: Step 4 timing is ambiguous.")
        print(f"         Ratio vs cold: {t4/t1:.2f}x, vs cache hit: {t4/t2:.2f}x")
    print()

    # Check 3: Cache hit again (step 5 should be fast, similar to step 2)
    if t5 < t4 * 0.8:
        print("PASS: Step 5 is fast (similar to Step 2).")
        print("       Cache is active again after reload.")
        print(f"       Speedup vs Step 4: {t4/t5:.1f}x")
    else:
        print("FAIL: Step 5 is slow (similar to Step 4).")
        print("      Cache is not active after reload.")
        print(f"      Ratio vs Step 4: {t5/t4:.2f}x (close to 1.0 = no cache benefit)")


if __name__ == "__main__":
    main()
