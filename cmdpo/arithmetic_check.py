from __future__ import annotations

import re


_PROBLEM_RE = re.compile(
    r"has\s+(\d+)\s+packs\.\s+Each pack contains\s+(\d+)\s+pencils\.\s+Then\s+(\d+)\s+more",
    flags=re.IGNORECASE,
)
_EQUATION_RE = re.compile(r"(-?\d+)\s*([+*×x])\s*(-?\d+)\s*=\s*(-?\d+)")


def _parse_problem(prompt: str) -> tuple[int, int, int] | None:
    match = _PROBLEM_RE.search(prompt)
    if not match:
        return None
    return tuple(int(x) for x in match.groups())


def _equation_is_correct(left: int, op: str, right: int, result: int) -> bool:
    if op == "+":
        return left + right == result
    if op in {"*", "×", "x"}:
        return left * right == result
    return True


def locate_arithmetic_error(prompt: str, steps: list[str]) -> int | None:
    """Locate simple arithmetic/variable-use errors for generated toy arithmetic data."""
    problem = _parse_problem(prompt)
    expected_product = expected_extra = expected_total = None
    if problem is not None:
        packs, per_pack, extra = problem
        expected_product = packs * per_pack
        expected_extra = extra
        expected_total = expected_product + expected_extra

    for idx, step in enumerate(steps):
        for match in _EQUATION_RE.finditer(step):
            left = int(match.group(1))
            op = match.group(2)
            right = int(match.group(3))
            result = int(match.group(4))
            if not _equation_is_correct(left, op, right, result):
                return idx

            if expected_product is not None and op in {"*", "×", "x"}:
                if result != expected_product:
                    return idx
            if expected_total is not None and op == "+" and left == expected_product:
                if right != expected_extra or result != expected_total:
                    return idx

    return None
