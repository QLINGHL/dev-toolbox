"""Command-line interface: python -m toolbox <command> [args]

Commands:
    slugify  "Hello World!"        -> hello-world
    snake    "helloWorld-foo bar"  -> hello_world_foo_bar
    camel    "hello_world"         -> helloWorld   (--upper for PascalCase)
    b64      "hello"               -> aGVsbG8=     (--decode to reverse)
    hash     "hello"               -> sha256 hex digest (--algo md5|sha1|sha256)
    extract_json 'Here: {"a": 1}' -> {"a": 1} (pretty-printed)
"""

import argparse
import json
import sys

from toolbox import llmtools, texttools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolbox", description="Small, well-tested command-line utilities."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("slugify", help="turn a title into a URL slug")
    p.add_argument("text")

    p = sub.add_parser("snake", help="convert to snake_case")
    p.add_argument("text")

    p = sub.add_parser("camel", help="convert to camelCase")
    p.add_argument("text")
    p.add_argument("--upper", action="store_true", help="PascalCase")

    p = sub.add_parser("b64", help="base64 encode text (or --decode)")
    p.add_argument("text")
    p.add_argument("--decode", action="store_true")

    p = sub.add_parser("hash", help="hash text (default sha256)")
    p.add_argument("text")
    p.add_argument("--algo", default="sha256", choices=["md5", "sha1", "sha256"])

    p = sub.add_parser(
        "extract_json",
        help="extract the first valid JSON object/array from LLM output",
    )
    p.add_argument("text")

    return parser


def run(argv: list[str] | None = None) -> str:
    args = build_parser().parse_args(argv)
    if args.command == "slugify":
        return texttools.slugify(args.text)
    if args.command == "snake":
        return texttools.to_snake_case(args.text)
    if args.command == "camel":
        return texttools.to_camel_case(args.text, upper_first=args.upper)
    if args.command == "b64":
        return texttools.b64_decode(args.text) if args.decode else texttools.b64_encode(args.text)
    if args.command == "hash":
        return texttools.hash_text(args.text, args.algo)
    if args.command == "extract_json":
        return json.dumps(
            llmtools.extract_json(args.text), ensure_ascii=False, indent=2
        )
    raise AssertionError(f"unhandled command: {args.command}")


def main() -> None:
    print(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
