"""Shared diagonal-LinUCB action scorer for the perception-budget and recovery policies: pure stdlib, no rl/ imports."""

from __future__ import annotations

import math
from collections.abc import Sequence


def score_linucb_action(
    *,
    A_diag: Sequence[float],
    b: Sequence[float],
    support: Sequence[int],
    onehot: Sequence[int],
    alpha: float,
) -> tuple[float, float, int]:
    """Diagonal-LinUCB closed-form score for one action over a one-hot context: ``expected = Sigma_active b_i/A_i``, ``ucb = expected + alpha * sqrt(Sigma_active 1/A_i)``, ``support`` = min active per-feature support (0 if none)."""
    expected = 0.0
    var_sum = 0.0
    min_sup = -1
    for i, x in enumerate(onehot):
        if x == 0:
            continue
        ai = A_diag[i]
        expected += b[i] / ai
        var_sum += 1.0 / ai
        si = support[i]
        if min_sup < 0 or si < min_sup:
            min_sup = si
    if min_sup < 0:
        min_sup = 0
    ucb = expected + alpha * math.sqrt(max(0.0, var_sum))
    return expected, ucb, min_sup
