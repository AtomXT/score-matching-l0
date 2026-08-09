"""Penalty-rate conventions for the primary Gaussian comparison."""

from __future__ import annotations

import math


PENALTY_RATE_LABELS = {
    "sm_l0": "log(p)/n",
    "sm_l0_core": "log(p)/n",
    "sm_l0_milp": "log(p)/n",
    "graphl0": "log(p)/n",
    "sm_l1": "sqrt(log(p)/n)",
    "glasso": "sqrt(log(p)/n)",
}


def penalty_value(
    method: str,
    constant: float,
    p: int,
    n: int,
) -> tuple[float, str]:
    """Return ``constant`` times the method's paper-specific penalty rate."""
    rate = PENALTY_RATE_LABELS[method]
    base = math.log(p) / n
    if rate == "sqrt(log(p)/n)":
        base = math.sqrt(base)
    return constant * base, rate
