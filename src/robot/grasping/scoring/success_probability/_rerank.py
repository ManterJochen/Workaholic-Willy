"""Bounded per-candidate uncertainty reranker for grasp candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import math
from ._shadow import (
    _DENSE_BLEND_MODES,
    UNCERTAINTY_RERANK_METADATA_KEY_ADJUSTED,
    UNCERTAINTY_RERANK_METADATA_KEY_GEOMETRIC,
    UNCERTAINTY_RERANK_METADATA_KEY_WEIGHT,
    ShadowSuccessContext,
)


from ...uncertainty import per_candidate_uncertainty


@dataclass(frozen=True, slots=True)
class UncertaintyRerankConfig:
    """Runtime carrier for the per-candidate uncertainty re-sort, which subtracts as ``adjusted = score - weight*uncertainty`` after the blend. It is dense-only and mirrors :class:`RankingBlendConfig`."""

    enabled: bool = False
    weight: float = 0.0
    modes: tuple[str, ...] = ("dense_clutter", "dense_autonomous")

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(
                f"UncertaintyRerankConfig.enabled must be bool, got {type(self.enabled).__name__}"
            )
        if not isinstance(self.weight, (int, float)) or isinstance(self.weight, bool):
            raise TypeError(
                f"UncertaintyRerankConfig.weight must be a real number, got "
                f"{type(self.weight).__name__}"
            )
        w = float(self.weight)
        if not math.isfinite(w) or w < 0.0 or w > 0.5:
            raise ValueError(
                "UncertaintyRerankConfig.weight must be in [0.0, 0.5] "
                f"(geometric-score floor of 50%); got {self.weight!r}"
            )
        if not isinstance(self.modes, tuple):
            raise TypeError(
                f"UncertaintyRerankConfig.modes must be a tuple, got {type(self.modes).__name__}"
            )
        bad = [m for m in self.modes if m not in _DENSE_BLEND_MODES]
        if bad:
            raise ValueError(
                "UncertaintyRerankConfig.modes is LOCKED to dense modes only "
                f"per the uncertainty-rerank design contract; got disallowed mode(s) {bad!r}; "
                f"valid: {sorted(_DENSE_BLEND_MODES)!r}"
            )
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("UncertaintyRerankConfig.modes must be unique")


@dataclass(frozen=True, slots=True)
class UncertaintyRerankTelemetry:
    """Audit payload for one :func:`maybe_uncertainty_rerank_candidates` call. ``applied`` and ``skip_reason`` say whether the rerank fired and why."""

    enabled: bool
    applied: bool
    skip_reason: str | None
    weight: float
    mode_label: str | None
    lifecycle_phase: str | None
    n_candidates: int
    top1_changed: bool
    winner_geometric_score: float | None
    winner_uncertainty: float | None
    winner_adjusted_score: float | None
    pre_rerank_top1_geometric_score: float | None


def _uncertainty_rerank_skip(
    *,
    reason: str,
    config: UncertaintyRerankConfig,
    ctx: ShadowSuccessContext | None,
    n_candidates: int,
    pre_rerank_top1_geometric_score: float | None = None,
) -> "UncertaintyRerankTelemetry":
    """Construct an ``UncertaintyRerankTelemetry`` describing a non-applied call."""
    return UncertaintyRerankTelemetry(
        enabled=True,
        applied=False,
        skip_reason=reason,
        weight=float(config.weight),
        mode_label=getattr(ctx, "mode_label", None) if ctx is not None else None,
        lifecycle_phase=(
            getattr(ctx, "lifecycle_phase", None) if ctx is not None else None
        ),
        n_candidates=n_candidates,
        top1_changed=False,
        winner_geometric_score=None,
        winner_uncertainty=None,
        winner_adjusted_score=None,
        pre_rerank_top1_geometric_score=pre_rerank_top1_geometric_score,
    )


def maybe_uncertainty_rerank_candidates(
    candidates: Sequence[Any],
    *,
    ctx: ShadowSuccessContext | None,
    config: UncertaintyRerankConfig | None,
) -> tuple[tuple[Any, ...], UncertaintyRerankTelemetry | None]:
    """Re-sort ``candidates`` by the subtractive penalty ``adjusted = score - weight*uncertainty``, after the blend, when every gate passes. It never mutates ``GraspPoint.score``, and it is a hard no-op if any candidate lacks an uncertainty of its own."""
    cand_tuple: tuple[Any, ...] = tuple(candidates)
    if config is None or not config.enabled:
        return cand_tuple, None

    n = len(cand_tuple)
    if ctx is None:
        return cand_tuple, _uncertainty_rerank_skip(
            reason="no_uncertainty_context", config=config, ctx=None, n_candidates=n,
        )
    if n == 0:
        return cand_tuple, _uncertainty_rerank_skip(
            reason="empty_candidates", config=config, ctx=ctx, n_candidates=0,
        )
    if ctx.lifecycle_phase == "shadow":
        return cand_tuple, _uncertainty_rerank_skip(
            reason="shadow_lifecycle", config=config, ctx=ctx, n_candidates=n,
        )
    if ctx.mode_label not in config.modes:
        return cand_tuple, _uncertainty_rerank_skip(
            reason="mode_not_in_rerank_modes", config=config, ctx=ctx, n_candidates=n,
        )

    w = float(config.weight)
    pre_top1 = cand_tuple[0]
    pre_geo = float(getattr(pre_top1, "score", 0.0))

    if w == 0.0:
        return cand_tuple, _uncertainty_rerank_skip(
            reason="weight_zero",
            config=config,
            ctx=ctx,
            n_candidates=n,
            pre_rerank_top1_geometric_score=pre_geo,
        )

    # Read the per-candidate uncertainty, which is the corridor risk. A missing value is a hard
    # no-op: the producer must have stamped every survivor, and a partial set is refused.
    scored: list[tuple[float, int, Any, float, float]] = []
    for i, point in enumerate(cand_tuple):
        u = per_candidate_uncertainty(point)
        if u is None:
            return cand_tuple, _uncertainty_rerank_skip(
                reason="missing_uncertainty",
                config=config,
                ctx=ctx,
                n_candidates=n,
                pre_rerank_top1_geometric_score=pre_geo,
            )
        g = float(getattr(point, "score", 0.0))
        adjusted = g - w * float(u)
        scored.append((adjusted, i, point, g, float(u)))

    # All gates passed, so the metadata is stamped additively and the order changes. ``.score``
    # is never touched.
    for adjusted, _i, point, g, _u in scored:
        md = getattr(point, "metadata", None)
        if isinstance(md, dict):
            md[UNCERTAINTY_RERANK_METADATA_KEY_ADJUSTED] = float(adjusted)
            md[UNCERTAINTY_RERANK_METADATA_KEY_GEOMETRIC] = float(g)
            md[UNCERTAINTY_RERANK_METADATA_KEY_WEIGHT] = w

    # Sort by descending adjusted score, breaking ties by the pre-rerank index so the order is
    # deterministic.
    scored.sort(key=lambda t: (-t[0], t[1]))
    new_tuple = tuple(t[2] for t in scored)
    winner_adj, _, new_winner, winner_geo, winner_u = scored[0]

    return new_tuple, UncertaintyRerankTelemetry(
        enabled=True,
        applied=True,
        skip_reason=None,
        weight=w,
        mode_label=ctx.mode_label,
        lifecycle_phase=ctx.lifecycle_phase,
        n_candidates=n,
        top1_changed=(new_winner is not pre_top1),
        winner_geometric_score=float(winner_geo),
        winner_uncertainty=float(winner_u),
        winner_adjusted_score=float(winner_adj),
        pre_rerank_top1_geometric_score=pre_geo,
    )
