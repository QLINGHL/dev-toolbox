"""Tests for toolbox.llmtools.estimate_tokens / token_breakdown.

Runs two ways:
    python tests/test_tokens.py     # plain assert runner, no dependencies
    pytest tests/                   # the same functions are pytest tests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from toolbox import cli, llmtools  # noqa: E402


def test_empty_string_is_zero():
    assert llmtools.estimate_tokens("") == 0
    assert llmtools.token_breakdown("")["total"] == 0


def test_whitespace_only_is_zero():
    assert llmtools.estimate_tokens("  \n\t ") == 0


def test_english_words_use_four_chars_per_token():
    # "hello" (5 chars) -> ceil(5/4) = 2, "world" -> 2, total 4.
    assert llmtools.estimate_tokens("hello world") == 4
    # Short words never round down to zero.
    assert llmtools.estimate_tokens("a") == 1
    assert llmtools.estimate_tokens("hi") == 1
    # Exactly 4 chars -> 1 token, 8 chars -> 2 tokens.
    assert llmtools.estimate_tokens("test") == 1
    assert llmtools.estimate_tokens("abcdefgh") == 2


def test_cjk_chars_count_one_each():
    assert llmtools.estimate_tokens("你好世界") == 4
    breakdown = llmtools.token_breakdown("你好世界")
    assert breakdown == {
        "cjk_chars": 4,
        "ascii_words": 0,
        "word_tokens": 0,
        "symbols": 0,
        "total": 4,
    }


def test_mixed_chinese_and_english():
    # "hello" -> 2 word tokens, "你好" -> 2 CJK tokens, total 4.
    assert llmtools.estimate_tokens("hello 你好") == 4
    breakdown = llmtools.token_breakdown("hello 你好")
    assert breakdown["cjk_chars"] == 2
    assert breakdown["ascii_words"] == 1
    assert breakdown["word_tokens"] == 2


def test_punctuation_counts_as_symbols():
    # "Hello, world!" -> 2+2 word tokens + 2 symbols = 6.
    breakdown = llmtools.token_breakdown("Hello, world!")
    assert breakdown["symbols"] == 2
    assert breakdown["total"] == 6


def test_numbers_and_underscore_are_words():
    # "abc123" (6 chars) -> ceil(6/4) = 2; "foo_bar" (7 chars) -> 2.
    assert llmtools.estimate_tokens("abc123") == 2
    assert llmtools.estimate_tokens("foo_bar") == 2


def test_breakdown_total_is_sum_of_parts():
    for text in ["hello 你好!", "Python 3.10 发布了", "a, b; c"]:
        breakdown = llmtools.token_breakdown(text)
        assert breakdown["total"] == (
            breakdown["cjk_chars"]
            + breakdown["word_tokens"]
            + breakdown["symbols"]
        )
        assert llmtools.estimate_tokens(text) == breakdown["total"]


def test_non_string_raises_type_error():
    for bad in [None, 123, ["hello"]]:
        try:
            llmtools.estimate_tokens(bad)
        except TypeError:
            pass
        else:
            raise AssertionError(f"expected TypeError for {bad!r}")
        try:
            llmtools.token_breakdown(bad)
        except TypeError:
            pass
        else:
            raise AssertionError(f"expected TypeError for {bad!r}")


def test_cli_tokens():
    assert cli.run(["tokens", "hello world"]) == "4"
    assert cli.run(["tokens", "你好世界"]) == "4"


def test_cli_tokens_detail():
    import json

    out = cli.run(["tokens", "hello 你好", "--detail"])
    breakdown = json.loads(out)
    assert breakdown["total"] == 4
    assert breakdown["cjk_chars"] == 2
    assert breakdown["ascii_words"] == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
