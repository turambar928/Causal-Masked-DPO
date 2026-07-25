from __future__ import annotations

import re


_NUMBERED_STEP_RE = re.compile(
    r"(?=(?:^|\n|\s)(?:step\s*\d+[:.)-]|\d+[:.)-]|\(\d+\)))",
    flags=re.IGNORECASE,
)
_CONNECTOR_RE = re.compile(r"\b(Therefore|Thus|So|Hence|Then|Next)\b", flags=re.IGNORECASE)


def _clean_piece(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _merge_short_steps(steps: list[str], min_chars: int) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for step in steps:
        if not buffer:
            buffer = step
        elif len(buffer) < min_chars:
            buffer = f"{buffer} {step}".strip()
        else:
            merged.append(buffer)
            buffer = step
    if buffer:
        if merged and len(buffer) < min_chars:
            merged[-1] = f"{merged[-1]} {buffer}".strip()
        else:
            merged.append(buffer)
    return merged


def _split_long_step(step: str, max_chars: int) -> list[str]:
    if len(step) <= max_chars:
        return [step]
    pieces = re.split(r"(?<=[.;。；])\s+", step)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks or [step]


def segment_steps(text: str, min_chars: int = 8, max_chars: int = 500) -> list[str]:
    """Split a reasoning response into coarse semantic steps."""
    text = text.strip()
    if not text:
        return []

    raw_lines = [_clean_piece(line) for line in text.splitlines() if _clean_piece(line)]
    if len(raw_lines) >= 2:
        candidates = raw_lines
    else:
        numbered = [_clean_piece(x) for x in _NUMBERED_STEP_RE.split(text) if _clean_piece(x)]
        if len(numbered) >= 2:
            candidates = numbered
        else:
            marked = _CONNECTOR_RE.sub(r"@@@\1", text)
            connector_parts = [_clean_piece(x) for x in marked.split("@@@") if _clean_piece(x)]
            if len(connector_parts) >= 2:
                candidates = connector_parts
            else:
                candidates = [_clean_piece(x) for x in re.split(r"(?<=[.。;；])\s+", text) if _clean_piece(x)]

    split_steps: list[str] = []
    for candidate in candidates:
        split_steps.extend(_split_long_step(candidate, max_chars=max_chars))
    return _merge_short_steps(split_steps, min_chars=min_chars)


def find_step_char_spans(response: str, steps: list[str]) -> list[tuple[int, int]]:
    """Best-effort mapping from segmented step text back to response character spans."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for step in steps:
        normalized = step.strip()
        start = response.find(normalized, cursor)
        if start < 0:
            compact = re.sub(r"\s+", " ", response[cursor:])
            compact_start = compact.find(normalized)
            if compact_start < 0:
                start = cursor
                end = min(len(response), cursor + len(normalized))
            else:
                start = cursor + compact_start
                end = start + len(normalized)
        else:
            end = start + len(normalized)
        spans.append((start, end))
        cursor = max(cursor, end)
    return spans
