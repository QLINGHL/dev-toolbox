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

``cosine_similarity`` compares two vectors -- the core operation behind
embedding search in RAG: rank candidate chunks by their cosine
similarity to the query embedding. ``text_cosine_similarity`` is the
same idea without embeddings: word-frequency vectors, so two texts can
be compared offline (e.g. to check whether a retrieved chunk is a
near-duplicate of the query, or to rank chunks by keyword overlap).

``render_prompt`` fills ``{{variable}}`` placeholders in a prompt
template -- the step that turns a stored template into the exact text
sent to a model. It fails loudly on a missing variable instead of
silently sending a half-filled prompt, and
``list_prompt_variables`` reports which variables a template needs.
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
# {{name}} placeholders, with optional whitespace inside the braces.
# Names are Python-identifier style so a template cannot smuggle in
# attribute access or expressions -- it is plain substitution, not eval.
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


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


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two numeric vectors.

    ``dot(a, b) / (|a| * |b|)`` -- 1.0 means the vectors point the same
    way, 0.0 means they are orthogonal, -1.0 means opposite. Magnitude
    does not matter: ``[1, 2]`` and ``[2, 4]`` score 1.0, which is why
    embedding search ranks by cosine rather than by distance.

    Args:
        a, b: sequences of numbers (lists, tuples, ...) of equal length.

    Raises:
        TypeError: if either input is not a sequence of numbers, or an
            element is not a real number (bool counts as non-numeric here).
        ValueError: if the vectors are empty, have different lengths,
            or either one is the zero vector (similarity is undefined --
            there is no direction to compare).
    """
    for name, vec in (("a", a), ("b", b)):
        if isinstance(vec, (str, bytes)) or not hasattr(vec, "__iter__"):
            raise TypeError(
                f"cosine_similarity {name} expects a sequence of numbers, "
                f"got {type(vec).__name__}"
            )
        for element in vec:
            if isinstance(element, bool) or not isinstance(
                element, (int, float)
            ):
                raise TypeError(
                    f"cosine_similarity {name} expects numbers, "
                    f"got {type(element).__name__}"
                )
    a = list(a)
    b = list(b)
    if not a or not b:
        raise ValueError("cosine_similarity expects non-empty vectors")
    if len(a) != len(b):
        raise ValueError(
            f"cosine_similarity vectors must have equal length, "
            f"got {len(a)} and {len(b)}"
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError(
            "cosine_similarity is undefined for the zero vector"
        )
    return dot / (norm_a * norm_b)


def _text_tokens(text: str) -> list:
    """Tokens for the word-frequency vector: lowercased ASCII words plus
    each CJK character as its own token (CJK has no word boundaries)."""
    tokens = [word.lower() for word in _WORD_RE.findall(text)]
    tokens.extend(_CJK_RE.findall(text))
    return tokens


def text_cosine_similarity(a: str, b: str) -> float:
    """Cosine similarity between two texts, via word-frequency vectors.

    Each text becomes a bag-of-words vector (ASCII words lowercased,
    each CJK character its own token; punctuation and whitespace are
    ignored), then :func:`cosine_similarity` compares the vectors. The
    result is in [0.0, 1.0]: 1.0 for identical word multisets (order
    does not matter), 0.0 when the texts share no word.

    Use it offline to compare a query with a candidate chunk, or to
    spot near-duplicate chunks before indexing them. For semantic
    similarity (same meaning, different words) use real embeddings --
    this only sees shared words.

    Raises:
        TypeError: if either input is not a string.
        ValueError: if either text has no tokens (empty, or only
            punctuation/whitespace) -- the zero vector has no direction.
    """
    if not isinstance(a, str):
        raise TypeError(f"text_cosine_similarity expects str, got {type(a).__name__}")
    if not isinstance(b, str):
        raise TypeError(f"text_cosine_similarity expects str, got {type(b).__name__}")
    freq_a: dict = {}
    for token in _text_tokens(a):
        freq_a[token] = freq_a.get(token, 0) + 1
    freq_b: dict = {}
    for token in _text_tokens(b):
        freq_b[token] = freq_b.get(token, 0) + 1
    if not freq_a or not freq_b:
        raise ValueError(
            "text_cosine_similarity expects texts with at least one word"
        )
    keys = set(freq_a) | set(freq_b)
    return cosine_similarity(
        [freq_a.get(k, 0) for k in keys], [freq_b.get(k, 0) for k in keys]
    )


def list_prompt_variables(template: str) -> list:
    """Return the placeholder names in ``template``, in first-appearance
    order, deduplicated.

    Only well-formed ``{{name}}`` placeholders count (names are
    identifier-style, optional whitespace inside the braces is allowed).
    Text that merely looks like a placeholder -- ``{{`` with no closing
    ``}}``, or ``{{not-a-name}}`` -- is literal text and is not listed.

    Raises:
        TypeError: if ``template`` is not a string.
    """
    if not isinstance(template, str):
        raise TypeError(
            f"list_prompt_variables expects str, got {type(template).__name__}"
        )
    seen = []
    for match in _PLACEHOLDER_RE.finditer(template):
        name = match.group(1)
        if name not in seen:
            seen.append(name)
    return seen


def render_prompt(template: str, variables=None) -> str:
    """Render a prompt template by filling ``{{variable}}`` placeholders.

    Args:
        template: the template text, e.g.
            ``"You are a {{role}}. Summarise this {{language}} text: {{text}}"``.
        variables: a mapping from placeholder name to value. Values are
            converted with ``str()``; a missing key is an error, not a
            silent empty string. ``None`` is treated as an empty mapping
            (so a template with no placeholders renders as-is).

    Behaviour:
      * ``{{ name }}`` (whitespace inside the braces) works the same as
        ``{{name}}``; the same placeholder may appear several times.
      * Text that is not a well-formed placeholder -- an unmatched
        ``{{``, or ``{{not-a-name}}`` -- is left untouched, so literal
        braces in the prompt survive rendering.

    Raises:
        TypeError: if ``template`` is not a string, or ``variables``
            is not a mapping.
        ValueError: if any placeholder has no value in ``variables``;
            the message names every missing variable, so the caller can
            fix them all in one pass instead of one at a time.
    """
    if not isinstance(template, str):
        raise TypeError(f"render_prompt expects str, got {type(template).__name__}")
    if variables is None:
        variables = {}
    if not isinstance(variables, dict) and not hasattr(variables, "keys"):
        raise TypeError(
            f"render_prompt variables expects a mapping, "
            f"got {type(variables).__name__}"
        )
    missing = [
        name for name in list_prompt_variables(template) if name not in variables
    ]
    if missing:
        raise ValueError(
            "render_prompt missing variable(s): " + ", ".join(missing)
        )
    return _PLACEHOLDER_RE.sub(
        lambda match: str(variables[match.group(1)]), template
    )
