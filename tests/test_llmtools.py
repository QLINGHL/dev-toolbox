"""Tests for toolbox.llmtools.extract_json.

Runs two ways:
    python tests/test_llmtools.py   # plain assert runner, no dependencies
    pytest tests/                   # the same functions are pytest tests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from toolbox import cli, llmtools  # noqa: E402


def test_bare_object():
    assert llmtools.extract_json('{"a": 1, "b": [1, 2]}') == {"a": 1, "b": [1, 2]}


def test_bare_array():
    assert llmtools.extract_json('[1, "two", {"three": 3}]') == [
        1,
        "two",
        {"three": 3},
    ]


def test_fenced_json_block():
    text = '```json\n{"name": "Muse", "score": 9}\n```'
    assert llmtools.extract_json(text) == {"name": "Muse", "score": 9}


def test_fenced_block_without_language():
    text = '```\n{"a": 1}\n```'
    assert llmtools.extract_json(text) == {"a": 1}


def test_prose_around_json():
    text = 'Sure! Here is the result: {"ok": true} Hope that helps.'
    assert llmtools.extract_json(text) == {"ok": True}


def test_prose_then_fenced_block():
    text = 'Here you go:\n```json\n{"x": [1, 2, 3]}\n```\nDone.'
    assert llmtools.extract_json(text) == {"x": [1, 2, 3]}


def test_invalid_brace_before_valid_json_is_skipped():
    # A stray "{not json}" must not stop the scan for the real payload.
    text = 'Example format: {not json} but actual: {"a": 1}'
    assert llmtools.extract_json(text) == {"a": 1}


def test_braces_inside_json_strings():
    text = '{"text": "a {b} c", "n": 1}'
    assert llmtools.extract_json(text) == {"text": "a {b} c", "n": 1}


def test_nested_structure():
    text = 'Result:\n{"user": {"name": "Li", "tags": ["a", "b"]}, "count": 2}'
    assert llmtools.extract_json(text) == {
        "user": {"name": "Li", "tags": ["a", "b"]},
        "count": 2,
    }


def test_lead_in_inside_fence():
    text = '```json\nHere it is: {"a": 1}\n```'
    assert llmtools.extract_json(text) == {"a": 1}


def test_no_json_raises_value_error():
    for bad in ["", "hello world", "no braces at all", "{not json}", "```json\nnope\n```"]:
        try:
            llmtools.extract_json(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_non_string_raises_type_error():
    try:
        llmtools.extract_json({"a": 1})
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for non-str input")


def test_cli_extract_json():
    import json

    out = cli.run(["extract_json", 'Here: {"a": 1, "b": 2}'])
    assert json.loads(out) == {"a": 1, "b": 2}
    out = cli.run(["extract_json", '```json\n{"x": true}\n```'])
    assert json.loads(out) == {"x": True}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
