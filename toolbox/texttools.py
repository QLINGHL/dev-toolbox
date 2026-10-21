"""Small text utilities. Pure standard library, each function is pure."""

import base64
import hashlib
import re


def slugify(text: str, separator: str = "-") -> str:
    """Turn a title into a URL slug.

    Lowercases, drops punctuation, and collapses any run of whitespace or
    separators into one separator. Unicode letters (e.g. CJK) are kept.
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", separator, text)
    return text.strip(separator)


def to_snake_case(text: str) -> str:
    """Convert camelCase / PascalCase / kebab-case / spaces to snake_case."""
    text = re.sub(r"[\s-]+", "_", text.strip())
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def to_camel_case(text: str, upper_first: bool = False) -> str:
    """Convert snake_case / kebab-case / spaces to camelCase (or PascalCase)."""
    parts = [p for p in re.split(r"[_\s-]+", text.strip()) if p]
    if not parts:
        return ""
    first = parts[0].capitalize() if upper_first else parts[0].lower()
    return first + "".join(p.capitalize() for p in parts[1:])


def b64_encode(text: str) -> str:
    """Base64-encode a UTF-8 string."""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def b64_decode(data: str) -> str:
    """Decode a Base64 string back to UTF-8 text."""
    return base64.b64decode(data.encode("ascii")).decode("utf-8")


def hash_text(text: str, algorithm: str = "sha256") -> str:
    """Hex digest of a UTF-8 string (md5 / sha1 / sha256)."""
    if algorithm not in ("md5", "sha1", "sha256"):
        raise ValueError(f"unsupported algorithm: {algorithm}")
    return hashlib.new(algorithm, text.encode("utf-8")).hexdigest()
