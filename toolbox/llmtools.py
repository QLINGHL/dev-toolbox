"""LLM-oriented utilities. Pure standard library.

``extract_json`` is the workhorse for post-processing model output: models
rarely return bare JSON. They wrap it in prose, in a ```json fenced block,
or both, and they sometimes include a stray, invalid brace before the real
payload. This module returns the first *valid* JSON object or array found,
skipping anything that does not parse.

``estimate_tokens`` is an offline, tokenizer-free estimate for budgeting
prompt / context size before calling a model. It is a heuristic, not a
tokenizer: real tokenizers (BPE and friends) split words differently, so
treat the number as a planning estimate, not an exact count.

``chunk_text`` splits long text for RAG preprocessing: by characters
(exact sizes, for fixed-window splitting) or by estimated tokens
(word-boundary chunks that fit a context budget), with an overlap so a
sentence straddling a boundary is not lost from either side.
"""

import json
import math
import re

_FENCE_RE = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
_DECODER = json.JSONDecoder()
# CJK Unified Ideographs and common CJK-adjacent blocks.
_CJK_RE = re.compile(
    r"[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\U00020000-\U0002ffff]"
)
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _scan_text(text: str):
    """Return the first valid JSON object/array in ``text``, else None."""
    i = 0
    while i < len(text):
        if text[i] in "{[":
            try:
                value, _end = _DECODER.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(value, (dict, list)):
                return value
            i += 1
        else:
            i += 1
    return None


def _extract_from_fence(content: str):
    """Extract JSON from inside one fenced block, else None."""
    stripped = content.strip()
    if stripped and stripped[0] in "{[":
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(value, (dict, list)):
                return value
    # Fence may carry a lead-in ("Here you go: {...}") before the JSON.
    return _scan_text(content)


def extract_json(text: str):
    """Extract the first valid JSON object or array from LLM output.

    Handles, in document order:
      * bare JSON (``{"a": 1}`` or ``[1, 2]``),
      * a ```json (or plain ```) fenced block,
      * prose mixed around the JSON ("Sure! Here it is: {...} Hope that helps"),
      * an invalid brace earlier in the text (skipped, scanning continues),
      * braces inside JSON strings (handled by the real JSON parser).

    Raises:
        ValueError: if no valid JSON object or array is found.
        TypeError: if ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"extract_json expects str, got {type(text).__name__}")
    fences = list(_FENCE_RE.finditer(text))
    fence_by_start = {m.start(): m for m in fences}

    i = 0
    while i < len(text):
        if i in fence_by_start:
            match = fence_by_start[i]
            value = _extract_from_fence(match.group(1))
            if value is not None:
                return value
            i = match.end()
            continue
        if text[i] in "{[":
            try:
                value, end = _DECODER.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(value, (dict, list)):
                return value
            i = end
            continue
        i += 1

    raise ValueError("no valid JSON object or array found in text")


def token_breakdown(text: str) -> dict:
    """Break an offline token estimate down by character class.

    Model (documented, deliberately simple):
      * each CJK character counts as 1 token -- CJK text is dense and
        most tokenizers spend roughly one token per character;
      * each ASCII word (letters/digits/underscore) counts as
        ``ceil(len(word) / 4)`` tokens, at least 1 -- the common
        "about 4 characters per English token" rule of thumb;
      * every other non-whitespace character (punctuation, emoji,
        symbols) counts as 1 token;
      * whitespace counts as 0.

    Returns a dict with ``cjk_chars``, ``ascii_words``,
    ``word_tokens``, ``symbols``, and ``total``. ``total`` is
    ``cjk_chars + word_tokens + symbols``.

    Raises:
        TypeError: if ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"token_breakdown expects str, got {type(text).__name__}")
    cjk_chars = len(_CJK_RE.findall(text))
    words = _WORD_RE.findall(text)
    word_tokens = sum(max(1, math.ceil(len(word) / 4)) for word in words)
    # Characters that are neither whitespace, CJK, nor part of an ASCII word.
    symbols = sum(
        1
        for ch in text
        if not ch.isspace() and not _CJK_RE.match(ch) and not re.match(r"[A-Za-z0-9_]", ch)
    )
    return {
        "cjk_chars": cjk_chars,
        "ascii_words": len(words),
        "word_tokens": word_tokens,
        "symbols": symbols,
        "total": cjk_chars + word_tokens + symbols,
    }


def estimate_tokens(text: str) -> int:
    """Estimate how many tokens ``text`` would cost, offline.

    This is the ``total`` from :func:`token_breakdown`. It is a
    budgeting heuristic, not a real tokenizer count: use it to decide
    whether a prompt fits a context window or to compare two prompts,
    not to predict an exact API bill.

    Raises:
        TypeError: if ``text`` is not a string.
    """
    return token_breakdown(text)["total"]


def chunk_text(text: str, size: int, overlap: int = 0, by: str = "chars") -> list:
    """Split ``text`` into chunks of at most ``size`` units, with overlap.

    Args:
        text: the text to split.
        size: maximum chunk size, in characters (``by="chars"``) or in
            estimated tokens (``by="tokens"``, see :func:`estimate_tokens`).
        overlap: units shared between consecutive chunks (0 = no overlap).
            Must be smaller than ``size``.
        by: ``"chars"`` or ``"tokens"``.

    Behaviour:
      * ``by="chars"``: exact sliding window over characters; the last
        chunk may be shorter than ``size``.
      * ``by="tokens"``: chunks are built word by word (whitespace
        separated, rejoined with single spaces) so words are never split;
        a single word that alone exceeds ``size`` becomes its own chunk.
        The overlap carries trailing whole words worth at most
        ``overlap`` estimated tokens into the next chunk. Overlap never
        re-emits a whole chunk: progress is at least one word per chunk,
        so chunking always terminates.
      * Empty (or whitespace-only, for ``by="tokens"``) text returns ``[]``.

    Raises:
        TypeError: if ``text`` is not a string, or ``size``/``overlap``
            are not integers.
        ValueError: if ``size`` < 1, ``overlap`` < 0, ``overlap`` >=
            ``size``, or ``by`` is neither ``"chars"`` nor ``"tokens"``.
    """
    if not isinstance(text, str):
        raise TypeError(f"chunk_text expects str, got {type(text).__name__}")
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError(f"chunk_text size expects int, got {type(size).__name__}")
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise TypeError(
            f"chunk_text overlap expects int, got {type(overlap).__name__}"
        )
    if size < 1:
        raise ValueError(f"chunk_text size must be >= 1, got {size}")
    if overlap < 0:
        raise ValueError(f"chunk_text overlap must be >= 0, got {overlap}")
    if overlap >= size:
        raise ValueError(
            f"chunk_text overlap ({overlap}) must be smaller than size ({size})"
        )
    if by not in ("chars", "tokens"):
        raise ValueError(f"chunk_text by must be 'chars' or 'tokens', got {by!r}")

    if by == "chars":
        if not text:
            return []
        step = size - overlap
        return [text[i : i + size] for i in range(0, len(text), step)]

    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start
        total = 0
        while end < len(words):
            word_tokens = estimate_tokens(words[end])
            if total > 0 and total + word_tokens > size:
                break
            total += word_tokens
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        if overlap == 0:
            start = end
            continue
        back = end
        back_tokens = 0
        while back > start:
            word_tokens = estimate_tokens(words[back - 1])
            if back_tokens + word_tokens > overlap:
                break
            back_tokens += word_tokens
            back -= 1
        # Guarantee progress: never restart at the same word.
        start = back if back > start else start + 1
    return chunks
