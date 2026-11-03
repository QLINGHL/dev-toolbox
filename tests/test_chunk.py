"""Tests for toolbox.llmtools.chunk_text and the `chunk` CLI command.

Runs two ways:
    python tests/test_chunk.py      # plain assert runner, no dependencies
    pytest tests/                   # the same functions are pytest tests
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from toolbox import cli, llmtools  # noqa: E402


def test_chars_exact_split_no_overlap():
    assert llmtools.chunk_text("abcdefgh", 3) == ["abc", "def", "gh"]
    assert llmtools.chunk_text("abc", 10) == ["abc"]


def test_chars_overlap_repeats_tail():
    # Step is size - overlap = 2: "abcd", "cdef", "efgh", "gh".
    assert llmtools.chunk_text("abcdefgh", 4, overlap=2) == [
        "abcd",
        "cdef",
        "efgh",
        "gh",
    ]


def test_chars_overlap_joins_back_to_original():
    text = "hello world, this is a test"
    chunks = llmtools.chunk_text(text, 10, overlap=3)
    # Removing the overlap from each chunk after the first rebuilds it.
    rebuilt = chunks[0] + "".join(c[3:] for c in chunks[1:])
    assert rebuilt == text


def test_empty_text_returns_empty_list():
    assert llmtools.chunk_text("", 10) == []
    assert llmtools.chunk_text("   ", 10, by="tokens") == []


def test_tokens_never_splits_words():
    text = "one two three four five"
    chunks = llmtools.chunk_text(text, 2, by="tokens")
    # "one"(1) + "two"(1) = 2, then "three"(2) alone, then "four"(1)+"five"(1).
    assert chunks == ["one two", "three", "four five"]
    for chunk in chunks:
        assert llmtools.estimate_tokens(chunk) <= 2


def test_tokens_overlap_carries_trailing_words():
    text = "alpha beta gamma delta"
    # Token counts: alpha=2, beta=1, gamma=2, delta=2; size 3, overlap 2.
    # "alpha beta" (3), overlap carries "beta" -> "beta gamma" (3),
    # overlap carries "gamma" -> "gamma" alone (delta would exceed 3),
    # then "delta".
    chunks = llmtools.chunk_text(text, 3, overlap=2, by="tokens")
    assert chunks == ["alpha beta", "beta gamma", "gamma", "delta"]
    # Original words appear in order; "beta" and "gamma" are the repeats.
    assert [w for c in chunks for w in c.split()] == [
        "alpha",
        "beta",
        "beta",
        "gamma",
        "gamma",
        "delta",
    ]


def test_tokens_single_oversized_word_is_its_own_chunk():
    # "abcdefghij" alone is ceil(10/4) = 3 tokens > size 2.
    chunks = llmtools.chunk_text("aa abcdefghij bb", 2, by="tokens")
    assert chunks == ["aa", "abcdefghij", "bb"]


def test_tokens_cjk_text_chunks_by_estimated_tokens():
    # Each CJK char is 1 token, but token mode splits on whitespace, so a
    # spaceless CJK run is one "word" and stays whole even if oversized.
    assert llmtools.chunk_text("你好世界", 2, by="tokens") == ["你好世界"]
    assert llmtools.chunk_text("你好 世界", 2, by="tokens") == ["你好", "世界"]


def test_invalid_arguments_raise():
    for kwargs in [
        {"size": 0},
        {"size": -1},
        {"size": 5, "overlap": -1},
        {"size": 5, "overlap": 5},
        {"size": 5, "overlap": 6},
        {"size": 5, "by": "words"},
    ]:
        try:
            llmtools.chunk_text("hello", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
    for bad_text in [None, 123]:
        try:
            llmtools.chunk_text(bad_text, 5)
        except TypeError:
            pass
        else:
            raise AssertionError(f"expected TypeError for {bad_text!r}")
    try:
        llmtools.chunk_text("hello", 2.5)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for float size")


def test_cli_chunk_chars():
    out = cli.run(["chunk", "abcdefgh", "--size", "3"])
    assert out == "abc\n---\ndef\n---\ngh"


def test_cli_chunk_tokens_json():
    out = cli.run(
        ["chunk", "one two three", "--size", "2", "--by", "tokens", "--json"]
    )
    assert json.loads(out) == ["one two", "three"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
