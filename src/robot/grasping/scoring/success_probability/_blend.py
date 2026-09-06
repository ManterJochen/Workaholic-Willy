"""Bounded probability-blend reranker for grasp candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import math

from src.robot.grasping.constants import (
    RANKING_BLEND_LOG_FILE,
    create_grasping_logger,
)

from ._shadow import (
    _DENSE_BLEND_MODES,
    SHADOW_METADATA_KEY_BLEND_WEIGHT,
    SHADOW_METADATA_KEY_FINAL_SCORE,
    SHADOW_METADATA_KEY_GEOMETRIC_PRE_BLEND,
    SHADOW_METADATA_KEY_PROBABILITY,
    ShadowSuccessContext,
)

#: The telemetry record below is the audit trail only where something is recording.
#: The blend is off by default and record logging is opt-in, so on a live cell this
#: file is the only durable answer to a blend that was enabled and changed nothing.
logger = create_grasping_logger("RankingBlend", RANKING_BLEND_LOG_FILE)




# The design contract, locked. Do not relax any of it without a design review:
#
# * The blend is convex only: ``final = (1 - w) * geometric + w * predicted_p``.
# * The weight ``w`` is clamped to ``[0.0, 0.5]``, so the geometric score keeps
#   at least 50 % of the influence and a miscalibrated model cannot dominate.
# * The mode filter is locked to ``{"dense_clutter", "dense_autonomous"}``. The
#   easy, auto and closed-loop modes never take part in blend reranking.
# * The lifecycle gate makes ``"shadow"`` a hard no-op. Only ``"canary"`` and
#   ``"active"`` may rerank, and promotion has already gated the artifact load.
# * ``GraspPoint.score`` is never mutated per candidate. Only the ``metadata``
#   dict and the order of the returned tuple change.
# * A missing per-candidate shadow probability is a hard no-op: a partially
#   scored candidate set is refused rather than guessed at.
# * Every call returns either ``None``, when the config is disabled, which keeps
#   the path silent and unchanged, or a :class:`RankingBlendTelemetry` saying
#   exactly why the blend was or was not applied. That is the audit trail.


@dataclass(frozen=True, slots=True)
class RankingBlendConfig:
    """Runtime carrier for the probability-blend reranker. It mirrors the pydantic config validator, so an out-of-contract value is rejected here as well."""

    enabled: bool = False
    weight: float = 0.20
    modes: tuple[str, ...] = ("dense_clutter", "dense_autonomous")

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(
                f"RankingBlendConfig.enabled must be bool, got "
                f"{type(self.enabled).__name__}"
            )
        if not isinstance(self.weight, (int, float)) or isinstance(
            self.weight, bool
        ):
            raise TypeError(
                f"RankingBlendConfig.weight must be a real number, got "
                f"{type(self.weight).__name__}"
            )
        w = float(self.weight)
        if not math.isfinite(w) or w < 0.0 or w > 0.5:
            raise ValueError(
                "RankingBlendConfig.weight must be in [0.0, 0.5] "
                f"(geometric-score floor of 50%); got {self.weight!r}"
            )
        if not isinstance(self.modes, tuple):
            raise TypeError(
                f"RankingBlendConfig.modes must be a tuple, got "
                f"{type(self.modes).__name__}"
            )
        bad = [m for m in self.modes if m not in _DENSE_BLEND_MODES]
        if bad:
            raise ValueError(
                "RankingBlendConfig.modes is LOCKED to dense modes only "
                "per the ranking-blend design contract; got disallowed mode(s) "
                f"{bad!r}; valid: {sorted(_DENSE_BLEND_MODES)!r}"
            )
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("RankingBlendConfig.modes must be unique")


@dataclass(frozen=True, slots=True)
class RankingBlendTelemetry:
    """Audit payload for one :func:`maybe_blend_rerank_candidates` call. ``applied`` and ``skip_reason`` say whether the blend fired and why."""

    enabled: bool
    applied: bool
    skip_reason: str | None
    weight: float
    mode_label: str | None
    lifecycle_phase: str | None
    n_candidates: int
    top1_changed: bool
    winner_geometric_score: float | None
    winner_predicted_probability: float | None
    winner_final_score: float | None
    pre_blend_top1_geometric_score: float | None


def _blend_skip(
    *,
    reason: str,
    config: RankingBlendConfig,
    ctx: ShadowSuccessContext | None,
    n_candidates: int,
    pre_blend_top1_geometric_score: float | None = None,
) -> "RankingBlendTelemetry":
    """Construct a ``RankingBlendTelemetry`` describing a non-applied call."""
    # The level depends on why it skipped. ``shadow_lifecycle``,
    # ``mode_not_in_blend_modes`` and ``empty_candidates`` are the design contract
    # doing its job, since the blend is locked to dense modes and inert in shadow,
    # so they log at debug. The rest mean the operator asked for a rerank and the
    # order came back untouched, which is a misconfiguration and is visible here
    # and nowhere else.
    log = (
        logger.debug
        if reason
        in ("shadow_lifecycle", "mode_not_in_blend_modes", "empty_candidates")
        else logger.warning
    )
    log(
        "Blend SKIPPED (%s): %d candidate(s), mode %s, lifecycle %s, weight %.3f",
        reason,
        n_candidates,
        getattr(ctx, "mode_label", None) if ctx is not None else None,
        getattr(ctx, "lifecycle_phase", None) if ctx is not None else None,
        float(config.weight),
    )
    return RankingBlendTelemetry(
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
        winner_predicted_probability=None,
        winner_final_score=None,
        pre_blend_top1_geometric_score=pre_blend_top1_geometric_score,
    )


def maybe_blend_rerank_candidates(
    candidates: Sequence[Any],
    *,
    ctx: ShadowSuccessContext | None,
    config: RankingBlendConfig | None,
) -> tuple[tuple[Any, ...], RankingBlendTelemetry | None]:
    """Rerank ``candidates`` by the convex blend ``final = (1-w)*geometric + w*predicted_p`` when every gate passes. Returns the tuple, reordered or not, together with the telemetry, which is ``None`` when the blend is disabled."""
    # Materialise once so callers can pass any Sequence.
    cand_tuple: tuple[Any, ...] = tuple(candidates)

    # The silent no-op path, taken when the config is absent or disabled. A
    # caller that does not opt in sees no change at all.
    if config is None or not config.enabled:
        return cand_tuple, None

    n = len(cand_tuple)

    if ctx is None:
        return cand_tuple, _blend_skip(
            reason="no_shadow_context", config=config, ctx=None, n_candidates=n,
        )
    if n == 0:
        return cand_tuple, _blend_skip(
            reason="empty_candidates", config=config, ctx=ctx, n_candidates=0,
        )
    if ctx.lifecycle_phase == "shadow":
        return cand_tuple, _blend_skip(
            reason="shadow_lifecycle", config=config, ctx=ctx, n_candidates=n,
        )
    if ctx.mode_label not in config.modes:
        return cand_tuple, _blend_skip(
            reason="mode_not_in_blend_modes",
            config=config,
            ctx=ctx,
            n_candidates=n,
        )

    w = float(config.weight)
    pre_top1 = cand_tuple[0]
    pre_geo = float(getattr(pre_top1, "score", 0.0))

    if w == 0.0:
        # The schema allows 0.0, but at runtime it does nothing, so a telemetry
        # record goes out and an operator can see the dead switch.
        return cand_tuple, _blend_skip(
            reason="weight_zero",
            config=config,
            ctx=ctx,
            n_candidates=n,
            pre_blend_top1_geometric_score=pre_geo,
        )

    # Read the per-candidate shadow probability. A missing or non-finite value
    # is a hard no-op: a partially scored set is refused rather than blended.
    scored: list[tuple[float, int, Any, float, float]] = []
    for i, point in enumerate(cand_tuple):
        md = getattr(point, "metadata", None)
        prob_raw: Any = None
        if isinstance(md, dict):
            prob_raw = md.get(SHADOW_METADATA_KEY_PROBABILITY)
        if not isinstance(prob_raw, (int, float)) or isinstance(prob_raw, bool):
            return cand_tuple, _blend_skip(
                reason="missing_shadow_probability",
                config=config,
                ctx=ctx,
                n_candidates=n,
                pre_blend_top1_geometric_score=pre_geo,
            )
        p = float(prob_raw)
        if not math.isfinite(p):
            return cand_tuple, _blend_skip(
                reason="missing_shadow_probability",
                config=config,
                ctx=ctx,
                n_candidates=n,
                pre_blend_top1_geometric_score=pre_geo,
            )
        g = float(getattr(point, "score", 0.0))
        final = (1.0 - w) * g + w * p
        scored.append((final, i, point, g, p))

    # All gates passed, so the metadata is stamped additively and the order changes.
    for final, _i, point, g, _p in scored:
        md = getattr(point, "metadata", None)
        if isinstance(md, dict):
            md[SHADOW_METADATA_KEY_FINAL_SCORE] = float(final)
            md[SHADOW_METADATA_KEY_GEOMETRIC_PRE_BLEND] = float(g)
            md[SHADOW_METADATA_KEY_BLEND_WEIGHT] = w

    # Sort by descending final score and break ties by the original index, so
    # equally scored candidates keep their pre-blend order and the result is
    # deterministic.
    scored.sort(key=lambda t: (-t[0], t[1]))
    new_tuple = tuple(t[2] for t in scored)
    winner_final, _, new_winner, winner_geo, winner_prob = scored[0]
    top1_changed = new_winner is not pre_top1

    logger.info(
        "Blend applied over %d candidate(s) at w=%.2f (mode %s, lifecycle %s): "
        "top1 %s: winner geometric %.3f, p %.3f -> final %.3f (was %.3f)",
        n,
        w,
        ctx.mode_label,
        ctx.lifecycle_phase,
        "CHANGED" if top1_changed else "unchanged",
        float(winner_geo),
        float(winner_prob),
        float(winner_final),
        pre_geo,
    )
    return new_tuple, RankingBlendTelemetry(
        enabled=True,
        applied=True,
        skip_reason=None,
        weight=w,
        mode_label=ctx.mode_label,
        lifecycle_phase=ctx.lifecycle_phase,
        n_candidates=n,
        top1_changed=top1_changed,
        winner_geometric_score=float(winner_geo),
        winner_predicted_probability=float(winner_prob),
        winner_final_score=float(winner_final),
        pre_blend_top1_geometric_score=pre_geo,
    )
