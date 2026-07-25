from __future__ import annotations

import math
import re
from fractions import Fraction


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_RE = re.compile(
    r"(?:####|answer\s*(?:is|=|:)|therefore\s*,?\s*the\s*answer\s*is)\s*([^\n]+)",
    flags=re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:,\d{3})*)?")


def extract_answer(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None

    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()

    answer_matches = _ANSWER_RE.findall(text)
    if answer_matches:
        return answer_matches[-1].strip()

    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].strip()
    return None


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", answer)
    answer = answer.strip().rstrip(".")
    return answer or None


def _to_float(answer: str | None) -> float | None:
    answer = normalize_answer(answer)
    if answer is None:
        return None
    try:
        if "/" in answer and re.fullmatch(r"[-+]?\d+/\d+", answer):
            return float(Fraction(answer))
        return float(answer)
    except ValueError:
        return None


def verify_answer(prediction: str, gold: str, atol: float = 1e-4) -> bool:
    pred = normalize_answer(extract_answer(prediction) or prediction)
    ref = normalize_answer(extract_answer(gold) or gold)
    if pred is None or ref is None:
        return False
    if pred == ref:
        return True

    pred_float = _to_float(pred)
    ref_float = _to_float(ref)
    if pred_float is not None and ref_float is not None:
        return math.isclose(pred_float, ref_float, rel_tol=0.0, abs_tol=atol)
    return False
