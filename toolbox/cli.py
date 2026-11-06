"""Command-line interface: python -m toolbox <command> [args]

Commands:
    slugify  "Hello World!"        -> hello-world
    snake    "helloWorld-foo bar"  -> hello_world_foo_bar
    camel    "hello_world"         -> helloWorld   (--upper for PascalCase)
    b64      "hello"               -> aGVsbG8=     (--decode to reverse)
    hash     "hello"               -> sha256 hex digest (--algo md5|sha1|sha256)
    extract_json 'Here: {"a": 1}' -> {"a": 1} (pretty-printed)
    tokens   "hello world"         -> 4 (offline estimate; --detail for breakdown)
    cosine   "1 2 3" "4 5 6"        -> 0.9746 (cosine similarity of two vectors)
    cosine   "hello world" "hello" --text -> 0.7071 (word-frequency similarity)
    chunk    "long text ..." --size 500 --overlap 50 --by tokens
                                        -> chunks separated by a --- line (--json for a JSON list)
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

    p = sub.add_parser(
        "tokens",
        help="offline token-count estimate for text (CJK 1/char, English ~4 chars/token)",
    )
    p.add_argument("text")
    p.add_argument("--detail", action="store_true", help="print the full breakdown")

    p = sub.add_parser(
        "cosine",
        help="cosine similarity of two vectors (space/comma-separated numbers), "
        "or of two texts with --text",
    )
    p.add_argument("a", help='first vector, e.g. "1 2 3", or first text with --text')
    p.add_argument("b", help='second vector, e.g. "4 5 6", or second text with --text')
    p.add_argument(
        "--text", action="store_true",
        help="compare texts by word frequency instead of numeric vectors",
    )

    p = sub.add_parser(
        "chunk",
        help="split long text into chunks for RAG preprocessing (with overlap)",
    )
    p.add_argument("text")
    p.add_argument("--size", type=int, default=500, help="max chunk size (default 500)")
    p.add_argument("--overlap", type=int, default=0, help="overlap between chunks (default 0)")
    p.add_argument(
        "--by", choices=["chars", "tokens"], default="chars",
        help="chunk unit: chars (exact) or tokens (estimated, word boundaries)",
    )
    p.add_argument("--json", action="store_true", help="print chunks as a JSON list")

    return parser


def _parse_vector(text: str) -> list:
    """Parse "1 2 3" / "1,2,3" / "1, 2, 3" into a list of floats."""
    parts = text.replace(",", " ").split()
    if not parts:
        raise ValueError("cosine expects a non-empty vector")
    try:
        return [float(part) for part in parts]
    except ValueError:
        raise ValueError(
            f"cosine expects numbers, got {text!r} "
            "(use --text to compare texts)"
        ) from None


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
    if args.command == "tokens":
        if args.detail:
            return json.dumps(
                llmtools.token_breakdown(args.text), ensure_ascii=False, indent=2
            )
        return str(llmtools.estimate_tokens(args.text))
    if args.command == "cosine":
        if args.text:
            score = llmtools.text_cosine_similarity(args.a, args.b)
        else:
            score = llmtools.cosine_similarity(
                _parse_vector(args.a), _parse_vector(args.b)
            )
        return f"{score:.4f}"
    if args.command == "chunk":
        chunks = llmtools.chunk_text(
            args.text, args.size, overlap=args.overlap, by=args.by
        )
        if args.json:
            return json.dumps(chunks, ensure_ascii=False, indent=2)
        return "\n---\n".join(chunks)
    raise AssertionError(f"unhandled command: {args.command}")


def main() -> None:
    print(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
