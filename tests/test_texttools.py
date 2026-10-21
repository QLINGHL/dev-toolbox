"""Tests for toolbox.texttools and the CLI dispatch.

Runs two ways:
    python tests/test_texttools.py   # plain assert runner, no dependencies
    pytest tests/                    # the same functions are pytest tests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from toolbox import cli, texttools  # noqa: E402


def test_slugify_basic():
    assert texttools.slugify("Hello World!") == "hello-world"
    assert texttools.slugify("  My  First   Post  ") == "my-first-post"
    assert texttools.slugify("Python 3.12: What's New?") == "python-312-whats-new"


def test_slugify_keeps_unicode_letters():
    assert texttools.slugify("你好 世界") == "你好-世界"


def test_snake_case():
    assert texttools.to_snake_case("helloWorld") == "hello_world"
    assert texttools.to_snake_case("HelloWorld") == "hello_world"
    assert texttools.to_snake_case("hello-world foo") == "hello_world_foo"
    assert texttools.to_snake_case("HTTPResponse") == "http_response"


def test_camel_case():
    assert texttools.to_camel_case("hello_world") == "helloWorld"
    assert texttools.to_camel_case("hello-world foo") == "helloWorldFoo"
    assert texttools.to_camel_case("hello_world", upper_first=True) == "HelloWorld"
    assert texttools.to_camel_case("") == ""


def test_b64_round_trip():
    assert texttools.b64_encode("hello") == "aGVsbG8="
    assert texttools.b64_decode("aGVsbG8=") == "hello"
    assert texttools.b64_decode(texttools.b64_encode("你好，世界")) == "你好，世界"


def test_hash_text():
    assert (
        texttools.hash_text("hello")
        == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    assert texttools.hash_text("hello", "md5") == "5d41402abc4b2a76b9719d911017c592"
    try:
        texttools.hash_text("hello", "sha512")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unsupported algorithm")


def test_cli_dispatch():
    assert cli.run(["slugify", "Hello World!"]) == "hello-world"
    assert cli.run(["snake", "helloWorld"]) == "hello_world"
    assert cli.run(["camel", "hello_world", "--upper"]) == "HelloWorld"
    assert cli.run(["b64", "hello"]) == "aGVsbG8="
    assert cli.run(["b64", "aGVsbG8=", "--decode"]) == "hello"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
