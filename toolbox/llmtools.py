"""LLM-oriented utilities. Pure standard library.

``extract_json`` is the workhorse for post-processing model output: models
rarely return bare JSON. They wrap it in prose, in a ```json fenced block,
or both, and they sometimes include a stray, invalid brace before the real
payload. This module returns the first *valid* JSON object or array found,
skipping anything that does not parse.
"""

import json
import re

_FENCE_RE = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
_DECODER = json.JSONDecoder()


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
