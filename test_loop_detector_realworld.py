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
"""Real-world LLM output loops for testing the loop detector."""

import sys, os, hashlib
sys.path.insert(0, os.path.dirname(__file__))

# Inline the loop detector code to avoid importing aiohttp from ollama_proxy
from typing import Optional

def find_repeating_suffix_patterns(
    s: str,
    min_repeats: int = 2,
    min_pattern_len: int = 1,
    max_pattern_len: int | None = None,
):
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
    def __init__(self, min_pattern_len: int = 50, max_pattern_len: int = 10000,
                 repeat_threshold: int = 4):
        self.min_pattern_len = min_pattern_len
        self.max_pattern_len = max(max(min_pattern_len, max_pattern_len), min_pattern_len)
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


def _make_pattern(length: int) -> str:
    """Generate a deterministic non-repeating pattern of exactly `length` chars."""
    import hashlib
    seed = f"pattern_{length}"
    hash_str = hashlib.md5(seed.encode()).hexdigest() * 100
    return hash_str[:length]


passed = 0
failed = 0
errors = []


def check(name, fn):
    global passed, failed
    try:
        result = fn()
        if not result:
            raise AssertionError("Test returned False")
        passed += 1
        print(f"  PASS: {name}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  FAIL: {name} — {e}")


def dl(detector):
    return detector.detected


# ============================================================
# Real-world LLM loop scenarios
# ============================================================

# Scenario 1: Sentence-level repetition (common with smaller models)
REAL_WORLD_1 = """I'm sorry, but I cannot provide that information as it would be inappropriate. 
I'm sorry, but I cannot provide that information as it would be inappropriate. 
I'm sorry, but I cannot provide that information as it would be inappropriate."""

# Scenario 2: Word-by-word echo (very common failure mode)
REAL_WORLD_2 = "the the the the the the the the the the the the the the the the"

# Scenario 3: Paragraph-level repetition (truly identical)
REAL_WORLD_3 = """Here is a detailed explanation of how machine learning works. Machine learning 
is a subset of artificial intelligence that enables systems to learn from data and improve 
their performance over time without being explicitly programmed. The process involves training 
algorithms on large datasets to identify patterns and make predictions or decisions based on 
those patterns. This technology has applications in many fields including healthcare, finance, 
and autonomous vehicles."""
REAL_WORLD_3 = REAL_WORLD_3 * 3

# Scenario 4: JSON/array repetition (common with code-generation models)
REAL_WORLD_4 = """[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]"""

# Scenario 5: Code block repetition
REAL_WORLD_5 = """def hello():
    print("Hello, World!")


def hello():
    print("Hello, World!")


def hello():
    print("Hello, World!")"""

# Scenario 6: Normal text that should NOT trigger (for negative testing)
NORMAL_TEXT = """The quick brown fox jumps over the lazy dog. This sentence contains every 
letter of the alphabet at least once. It's commonly used for typing practice and font 
demonstrations. The fox is quite agile and can leap over obstacles with ease. Many people 
use this phrase to test their keyboards or showcase new typefaces."""

# Scenario 7: Mixed content then loop (realistic — model starts normal, then gets stuck)
MIXED_THEN_LOOP = """Let me help you with that question. Based on my analysis, the answer is 
42. This is a well-known result in mathematics and computer science. Now let me provide some 
additional context about why this number is significant. It appears in various fields including 
physics, engineering, and even philosophy. The number 42 has become something of a cultural 
icon thanks to Douglas Adams' "The Hitchhiker's Guide to the Galaxy." Moving on to your next 
question, I'd be happy to assist with that as well. However, it seems like I might be repeating 
myself here. Let me try to provide a fresh perspective on this topic instead of going in circles.
I'm sorry, but I cannot provide that information as it would be inappropriate. 
I'm sorry, but I cannot provide that information as it would be inappropriate. 
I'm sorry, but I cannot provide that information as it would be inappropriate."""

# Scenario 8: Very long single-word repetition (extreme case)
REAL_WORLD_8 = "x" * 5000


def test_real_world_1_sentence_loop():
    """Sentence-level repetition should be detected."""
    d = LoopDetector(min_pattern_len=50, max_pattern_len=400, repeat_threshold=3)
    for chunk in REAL_WORLD_1.split():
        if d.add_chunk(chunk):
            break
    return dl(d) and len(d.pattern) >= 50


def test_real_world_2_word_echo():
    """Word-by-word echo should be detected."""
    # This is a short pattern, so we need lower min_pattern_len for this specific test
    d = LoopDetector(min_pattern_len=3, max_pattern_len=400, repeat_threshold=3)
    for chunk in REAL_WORLD_2.split():
        if d.add_chunk(chunk):
            break
    return dl(d)


def test_real_world_3_paragraph_loop():
    """Paragraph-level repetition should be detected."""
    d = LoopDetector(min_pattern_len=50, max_pattern_len=1000, repeat_threshold=3)
    for chunk in REAL_WORLD_3.split():
        if d.add_chunk(chunk):
            break
    return dl(d) and len(d.pattern) >= 100


def test_real_world_4_json_loop():
    """JSON array repetition should be detected."""
    d = LoopDetector(min_pattern_len=50, max_pattern_len=400, repeat_threshold=3)
    for chunk in REAL_WORLD_4.split():
        if d.add_chunk(chunk):
            break
    return dl(d)


def test_real_world_5_code_loop():
    """Code block repetition should be detected."""
    # Code blocks are shorter, use lower threshold
    d = LoopDetector(min_pattern_len=20, max_pattern_len=400, repeat_threshold=3)
    for chunk in REAL_WORLD_5.split("\n"):
        if d.add_chunk(chunk):
            break
    return dl(d)


def test_normal_text_no_loop():
    """Normal text should NOT trigger detection."""
    d = LoopDetector(min_pattern_len=50, max_pattern_len=400, repeat_threshold=3)
    for chunk in NORMAL_TEXT.split():
        d.add_chunk(chunk)
    return not dl(d)


def test_mixed_then_loop():
    """Normal text followed by loop should detect the loop."""
    d = LoopDetector(min_pattern_len=50, max_pattern_len=400, repeat_threshold=3)
    detected_at = None
    for i, chunk in enumerate(MIXED_THEN_LOOP.split()):
        if d.add_chunk(chunk):
            detected_at = i
            break
    return dl(d) and detected_at is not None


def test_extreme_single_char():
    """Extreme single-char repetition should be detected."""
    # This is a very long pattern of 'x' chars — should detect quickly
    d = LoopDetector(min_pattern_len=50, max_pattern_len=400, repeat_threshold=3)
    for chunk in REAL_WORLD_8:  # Feed char by char
        if d.add_chunk(chunk):
            break
    return dl(d)


def test_find_repeating_suffix_patterns_basic():
    """Test the standalone function with basic cases."""
    results = find_repeating_suffix_patterns("xxabcabcabc", min_pattern_len=1, max_pattern_len=20)
    patterns = [r["pattern"] for r in results]
    return "abc" in patterns


def test_find_repeating_suffix_patterns_overlapping():
    """Test overlapping patterns (aaaaaa -> a, aa, aaa)."""
    results = find_repeating_suffix_patterns("aaaaaa", min_pattern_len=1, max_pattern_len=6)
    pattern_lens = [r["pattern_len"] for r in results]
    return 1 in pattern_lens and 2 in pattern_lens and 3 in pattern_lens


def test_find_repeating_suffix_patterns_no_match():
    """Test with no repeating suffix."""
    results = find_repeating_suffix_patterns("abcdef", min_pattern_len=2, max_pattern_len=6)
    return len(results) == 0


# Run all tests
check("real_world_1_sentence_loop", test_real_world_1_sentence_loop)
check("real_world_2_word_echo", test_real_world_2_word_echo)
check("real_world_3_paragraph_loop", test_real_world_3_paragraph_loop)
check("real_world_4_json_loop", test_real_world_4_json_loop)
check("real_world_5_code_loop", test_real_world_5_code_loop)
check("normal_text_no_loop", test_normal_text_no_loop)
check("mixed_then_loop", test_mixed_then_loop)
check("extreme_single_char", test_extreme_single_char)
check("find_repeating_suffix_patterns_basic", test_find_repeating_suffix_patterns_basic)
check("find_repeating_suffix_patterns_overlapping", test_find_repeating_suffix_patterns_overlapping)
check("find_repeating_suffix_patterns_no_match", test_find_repeating_suffix_patterns_no_match)


print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
if errors:
    print("\nFailures:")
    for name, err in errors:
        print(f"  - {name}: {err}")
sys.exit(1 if failed > 0 else 0)
