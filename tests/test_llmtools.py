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


def test_cosine_identical_vectors_is_one():
    assert llmtools.cosine_similarity([1, 2, 3], [1, 2, 3]) == 1.0


def test_cosine_scaled_vectors_is_one():
    # Magnitude does not matter -- embedding search ranks by direction.
    assert abs(llmtools.cosine_similarity([1, 2], [2, 4]) - 1.0) < 1e-12


def test_cosine_orthogonal_is_zero():
    assert llmtools.cosine_similarity([1, 0], [0, 1]) == 0.0


def test_cosine_opposite_is_minus_one():
    assert abs(llmtools.cosine_similarity([1, 2], [-1, -2]) - (-1.0)) < 1e-12


def test_cosine_known_value():
    # dot = 32, |a| = 5, |b| = sqrt(41) -> 32 / (5 * sqrt(41))
    import math

    score = llmtools.cosine_similarity([3, 4], [4, 5])
    assert abs(score - 32 / (5 * math.sqrt(41))) < 1e-12


def test_cosine_mismatched_lengths_raise():
    for a, b in [([1, 2], [1]), ([1], [1, 2, 3])]:
        try:
            llmtools.cosine_similarity(a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {a!r} vs {b!r}")


def test_cosine_empty_or_zero_vector_raises():
    for a, b in [([], [1]), ([1], []), ([0, 0], [1, 2]), ([1, 2], [0, 0])]:
        try:
            llmtools.cosine_similarity(a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {a!r} vs {b!r}")


def test_cosine_non_numeric_raises_type_error():
    for a, b in [([1, "x"], [1, 2]), ("ab", [1, 2]), ([1, True], [1, 2]), (5, [1, 2])]:
        try:
            llmtools.cosine_similarity(a, b)
        except TypeError:
            pass
        else:
            raise AssertionError(f"expected TypeError for {a!r} vs {b!r}")


def test_text_cosine_identical_is_one():
    assert abs(llmtools.text_cosine_similarity("hello world", "world hello") - 1.0) < 1e-12


def test_text_cosine_disjoint_is_zero():
    assert llmtools.text_cosine_similarity("hello world", "goodbye moon") == 0.0


def test_text_cosine_partial_overlap():
    # "hello world" vs "hello there": dot=1, |a|=sqrt(2), |b|=sqrt(2) -> 0.5
    assert abs(llmtools.text_cosine_similarity("hello world", "hello there") - 0.5) < 1e-12


def test_text_cosine_case_insensitive_and_ignores_punctuation():
    assert abs(llmtools.text_cosine_similarity("Hello, World!", "hello world") - 1.0) < 1e-12


def test_text_cosine_repeated_words_count():
    # "cat cat dog" vs "cat dog dog": dot=4, |a|=|b|=sqrt(5) -> 0.8
    assert abs(llmtools.text_cosine_similarity("cat cat dog", "cat dog dog") - 0.8) < 1e-12


def test_text_cosine_cjk_characters_are_tokens():
    # "你好世界" vs "你好": dot=2, |a|=2, |b|=sqrt(2) -> 1/sqrt(2)
    import math

    score = llmtools.text_cosine_similarity("你好世界", "你好")
    assert abs(score - 1 / math.sqrt(2)) < 1e-12


def test_text_cosine_empty_or_no_words_raises():
    for a, b in [("", "hello"), ("hello", ""), ("!!!", "hello"), ("   ", "x")]:
        try:
            llmtools.text_cosine_similarity(a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {a!r} vs {b!r}")


def test_text_cosine_non_string_raises_type_error():
    try:
        llmtools.text_cosine_similarity(["hello"], "hello")
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for non-str input")


def test_cli_cosine_vectors():
    out = cli.run(["cosine", "1 2 3", "1 2 3"])
    assert out == "1.0000"
    out = cli.run(["cosine", "1,0", "0,1"])
    assert out == "0.0000"


def test_cli_cosine_text_mode():
    out = cli.run(["cosine", "hello world", "hello there", "--text"])
    assert out == "0.5000"


def test_render_prompt_basic():
    assert (
        llmtools.render_prompt(
            "You are a {{role}}. Summarise this {{language}} text.",
            {"role": "translator", "language": "French"},
        )
        == "You are a translator. Summarise this French text."
    )


def test_render_prompt_whitespace_inside_braces():
    assert (
        llmtools.render_prompt("Hello {{ name }}!", {"name": "Li"})
        == "Hello Li!"
    )


def test_render_prompt_repeated_placeholder():
    assert (
        llmtools.render_prompt("{{x}} and {{x}}", {"x": "a"}) == "a and a"
    )


def test_render_prompt_no_placeholders_is_unchanged():
    assert llmtools.render_prompt("plain text", {"unused": 1}) == "plain text"
    assert llmtools.render_prompt("plain text") == "plain text"
    assert llmtools.render_prompt("") == ""


def test_render_prompt_missing_variable_names_it():
    try:
        llmtools.render_prompt("Hi {{name}}, you are {{role}}.", {"name": "Li"})
    except ValueError as exc:
        assert "role" in str(exc)
        assert "name" not in str(exc)  # supplied variables are not reported
    else:
        raise AssertionError("expected ValueError for missing variable")


def test_render_prompt_lists_every_missing_variable():
    try:
        llmtools.render_prompt("{{a}} {{b}} {{c}}", {})
    except ValueError as exc:
        assert str(exc) == "render_prompt missing variable(s): a, b, c"
    else:
        raise AssertionError("expected ValueError for missing variables")


def test_render_prompt_unmatched_braces_are_literal():
    # A stray "{{" or a non-identifier placeholder is not a variable.
    assert llmtools.render_prompt("use {{ for dicts", {}) == "use {{ for dicts"
    assert (
        llmtools.render_prompt("{{not-a-name}} stays", {})
        == "{{not-a-name}} stays"
    )


def test_render_prompt_non_string_value_is_converted():
    assert llmtools.render_prompt("count {{n}}", {"n": 42}) == "count 42"


def test_render_prompt_bad_types_raise():
    for template, variables in [
        (42, {}),
        ("{{x}}", ["x"]),
        ("{{x}}", "x"),
    ]:
        try:
            llmtools.render_prompt(template, variables)
        except TypeError:
            pass
        else:
            raise AssertionError(
                f"expected TypeError for {template!r}, {variables!r}"
            )


def test_list_prompt_variables_order_and_dedup():
    assert llmtools.list_prompt_variables(
        "{{role}}: {{text}} ({{role}} again, {{text}})"
    ) == ["role", "text"]
    assert llmtools.list_prompt_variables("no placeholders") == []
    assert llmtools.list_prompt_variables("{{ not-a-name }} {{ ok }}") == ["ok"]


def test_cli_prompt_renders_and_reports_missing():
    out = cli.run(
        ["prompt", "You are a {{role}}.", "--var", "role=translator"]
    )
    assert out == "You are a translator."
    try:
        cli.run(["prompt", "Hi {{name}}"])
    except ValueError as exc:
        assert "name" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing CLI variable")


def test_cli_prompt_list_vars():
    out = cli.run(
        ["prompt", "{{role}}: {{text}} ({{role}})", "--list-vars"]
    )
    assert out == "role\ntext"


def test_cli_prompt_var_must_be_name_equals_value():
    for bad in [["prompt", "{{x}}", "--var", "no-equals"], ["prompt", "{{x}}", "--var", "1bad=v"]]:
        try:
            cli.run(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
