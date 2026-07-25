from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class LocalizationResult:
    first_error_step: int
    confidence: float
    prefix_success_rates: list[float]
    step_weights: list[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "first_error_step": self.first_error_step,
            "confidence": self.confidence,
            "prefix_success_rates": self.prefix_success_rates,
            "step_weights": self.step_weights,
        }


def build_cm_weights(num_steps: int, first_error: int, gamma: float) -> list[float]:
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    if num_steps == 0:
        return []
    first_error = max(0, min(first_error, num_steps - 1))
    weights: list[float] = []
    for k in range(num_steps):
        if k < first_error:
            weights.append(0.0)
        elif k == first_error:
            weights.append(1.0)
        else:
            weights.append(gamma ** (k - first_error))
    return weights


def localize_from_success_rates(
    success_rates: Sequence[float],
    gamma: float = 0.5,
    tau: float = 0.3,
    high_success: float = 0.8,
    low_success: float = 0.2,
) -> LocalizationResult:
    if not success_rates:
        return LocalizationResult(0, 0.0, [], [])

    rates = [float(max(0.0, min(1.0, q))) for q in success_rates]
    first_error = 0
    confidence = 0.0

    for idx in range(1, len(rates)):
        drop = rates[idx - 1] - rates[idx]
        if drop > tau:
            first_error = idx
            confidence = min(1.0, drop)
            break
    else:
        if all(q <= low_success for q in rates):
            first_error = 0
            confidence = 1.0 - max(rates)
        elif all(q >= high_success for q in rates):
            first_error = len(rates) - 1
            confidence = 0.0
        else:
            first_error = min(range(len(rates)), key=lambda i: rates[i])
            confidence = max(0.0, 1.0 - rates[first_error])

    return LocalizationResult(
        first_error_step=first_error,
        confidence=confidence,
        prefix_success_rates=rates,
        step_weights=build_cm_weights(len(rates), first_error, gamma),
    )
