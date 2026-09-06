"""Clutter-aware target selector.

Pure ranking module the bin-picking orchestrator calls after per-
segmentation grasps have been computed and before the executor
picks one to attempt. The selector decides which segmentation index
to execute first when several grasps are simultaneously viable, by
estimating how much each candidate would unblock others if removed.

The module is intentionally pure: no I/O, no logging, no robot
state. It takes a tuple of :class:`TargetCandidate` records (one per
successful per-segmentation grasp) and a
:class:`TargetOrderingConfig` and returns a single
:class:`OrderingDecision` carrying the chosen index, supporting
scores, the resolved mode, and a typed reason code.

Invariants
----------

* Defaults reduce to the legacy behaviour byte-identically: the master
  flag ``enabled`` is ``False`` and every blocker signal is also
  ``False``. When disabled the selector returns the legacy
  ``argmax(local_score)`` pick with reason
  :attr:`OrderingReason.DISABLED`.
* Easy mode is excluded from the default ``apply_modes`` at the
  schema layer; the selector itself never inspects the grasp mode.
* When enabled, the selector uses a two-stage hard guard:

    1. Compute the legacy local-score winner ``top_idx``.
    2. Compute ``priority[i] = local_score[i] + unlock_weight *
       unlock_score[i]`` and find candidate ``c`` maximising
       priority. Swap from ``top_idx`` to ``c`` only when
       ``local_score[c] >= local_score[top_idx] -
       max_local_score_drop``; otherwise keep ``top_idx`` with reason
       :attr:`OrderingReason.GUARD_BLOCKED_SWAP`.

* Unlock scores are normalised to ``[0, 1]`` so ``unlock_weight``
  has the same dimensional meaning as ``local_score``.

Not handled here: calibrated weights from replay, recovery
integration when no swap is feasible, and KPI scoring across the
dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "BlockerGraphConfig",
    "OrderingDecision",
    "OrderingMode",
    "OrderingReason",
    "TargetCandidate",
    "TargetOrderingConfig",
    "select_target",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrderingMode(str, Enum):
    """Which selection regime produced an :class:`OrderingDecision`."""

    SINGLE_BEST = "single_best"
    CLUTTER_AWARE = "clutter_aware"


class OrderingReason(str, Enum):
    """Why the selector picked the index it did."""

    LOCAL_MAX = "local_max"
    UNLOCK_SWAP = "unlock_swap"
    GUARD_BLOCKED_SWAP = "guard_blocked_swap"
    NO_CANDIDATES = "no_candidates"
    DISABLED = "disabled"


# ---------------------------------------------------------------------------
# Config records (immutable, slotted)
# ---------------------------------------------------------------------------


def _check_non_negative(name: str, value: float | int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")


def _check_unit_interval(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


@dataclass(frozen=True, slots=True)
class BlockerGraphConfig:
    """Per-signal flags for the blocker graph.

    Every flag defaults to ``False`` so the selector reduces to legacy
    behaviour even when the master flag is on. ``adjacency_radius_px``
    is the dilation radius used by the mask-adjacency signal; the
    depth checks use ``depth_tolerance_mm`` to decide whether a
    candidate is "in front" of another in camera space.
    """

    mask_adjacency_enabled: bool = False
    depth_only_enabled: bool = False
    corridor_overlap_enabled: bool = False
    adjacency_radius_px: int = 5
    depth_tolerance_mm: float = 10.0

    def __post_init__(self) -> None:
        _check_non_negative("adjacency_radius_px", self.adjacency_radius_px)
        _check_non_negative("depth_tolerance_mm", self.depth_tolerance_mm)


@dataclass(frozen=True, slots=True)
class TargetOrderingConfig:
    """Operator-facing configuration for clutter-aware ordering."""

    enabled: bool = False
    unlock_weight: float = 0.0
    max_local_score_drop: float = 0.1
    blocker_graph: BlockerGraphConfig = field(default_factory=BlockerGraphConfig)

    def __post_init__(self) -> None:
        _check_non_negative("unlock_weight", self.unlock_weight)
        _check_unit_interval("unlock_weight", float(self.unlock_weight))
        _check_unit_interval(
            "max_local_score_drop", float(self.max_local_score_drop)
        )


# ---------------------------------------------------------------------------
# Candidate + decision records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TargetCandidate:
    """A successful per-segmentation pick the selector may choose.

    Attributes
    ----------
    segmentation_index
        Index in the perception frame's ``segmentations`` tuple. This
        is what the orchestrator threads back into ``PickAttempt``.
    mask
        Boolean / uint8 HxW mask for the segmentation. Used for the
        mask-adjacency and corridor-overlap signals.
    centroid_depth_mm
        Per-object depth proxy in millimetres. Used by all signals to
        determine which candidate is closer to the camera.
    local_score
        The legacy per-segmentation grasp score the calculator
        returned. The selector reduces to picking ``argmax`` of this
        whenever it is disabled or unable to discriminate.
    bbox_px
        ``(x0, y0, x1, y1)`` inclusive pixel bounding box of ``mask``.
        Used by the depth-only signal to cheaply detect xy-overlap.
    """

    segmentation_index: int
    mask: np.ndarray
    centroid_depth_mm: float
    local_score: float
    bbox_px: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class OrderingDecision:
    """Telemetry record describing one selector invocation."""

    chosen_index: int | None
    local_scores: tuple[float, ...]
    unlock_scores: tuple[float, ...]
    priority_scores: tuple[float, ...]
    mode: OrderingMode
    reason: OrderingReason

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for log scrapers."""

        return {
            "chosen_index": self.chosen_index,
            "mode": self.mode.value,
            "reason": self.reason.value,
            "local_scores": list(self.local_scores),
            "unlock_scores": list(self.unlock_scores),
            "priority_scores": list(self.priority_scores),
        }


# ---------------------------------------------------------------------------
# Blocker-graph signals
# ---------------------------------------------------------------------------


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool, copy=False)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square-kernel dilation by ``radius`` using numpy-only shifts.

    Equivalent to a Chebyshev-distance binary dilation with kernel
    ``(2*radius+1) x (2*radius+1)``. Scipy is not used: the rest of
    the grasping stack stays numpy-only.
    """

    if radius <= 0:
        return _as_bool_mask(mask)
    src = _as_bool_mask(mask)
    out = src.copy()
    h, w = src.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            y0_src = max(0, -dy)
            y1_src = h - max(0, dy)
            x0_src = max(0, -dx)
            x1_src = w - max(0, dx)
            y0_dst = max(0, dy)
            y1_dst = h - max(0, -dy)
            x0_dst = max(0, dx)
            x1_dst = w - max(0, -dx)
            if y1_src <= y0_src or x1_src <= x0_src:
                continue
            out[y0_dst:y1_dst, x0_dst:x1_dst] |= src[y0_src:y1_src, x0_src:x1_src]
    return out


def _bbox_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1


def _a_in_front(a: TargetCandidate, b: TargetCandidate, tol_mm: float) -> bool:
    """``True`` iff ``a`` is at least ``tol_mm`` closer than ``b``."""

    return float(a.centroid_depth_mm) + float(tol_mm) <= float(b.centroid_depth_mm)


def _blocks(
    a: TargetCandidate,
    b: TargetCandidate,
    cfg: BlockerGraphConfig,
    *,
    dilated_a: np.ndarray | None = None,
) -> bool:
    """``True`` iff removing ``a`` is expected to unblock ``b``.

    The function is a logical OR over the enabled blocker signals.
    Corridor-overlap requires per-candidate corridor geometry that is
    not wired through pick_loop; when its flag is on but the inputs
    are unavailable the signal is skipped (kept conservative).
    """

    if a is b:
        return False
    tol = cfg.depth_tolerance_mm

    if cfg.mask_adjacency_enabled:
        if dilated_a is None:
            dilated_a = _dilate(a.mask, cfg.adjacency_radius_px)
        if _a_in_front(a, b, tol) and bool(
            np.any(dilated_a & _as_bool_mask(b.mask))
        ):
            return True

    if cfg.depth_only_enabled:
        if _a_in_front(a, b, tol) and _bbox_overlap(a.bbox_px, b.bbox_px):
            return True

    # corridor_overlap_enabled: the flag ships for forward-compatibility
    # but corridor geometry is not consumed here.
    return False


def _unlock_score(
    a_idx: int,
    candidates: tuple[TargetCandidate, ...],
    cfg: BlockerGraphConfig,
) -> float:
    """Fraction of other candidates that ``a`` blocks, in [0, 1]."""

    n = len(candidates)
    if n <= 1:
        return 0.0
    a = candidates[a_idx]
    any_signal_on = (
        cfg.mask_adjacency_enabled
        or cfg.depth_only_enabled
        or cfg.corridor_overlap_enabled
    )
    if not any_signal_on:
        return 0.0
    dilated_a: np.ndarray | None = None
    if cfg.mask_adjacency_enabled:
        dilated_a = _dilate(a.mask, cfg.adjacency_radius_px)
    blocked = 0
    for j, b in enumerate(candidates):
        if j == a_idx:
            continue
        if _blocks(a, b, cfg, dilated_a=dilated_a):
            blocked += 1
    raw = blocked / float(n - 1)
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        return 1.0
    return float(raw)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def select_target(
    *,
    candidates: tuple[TargetCandidate, ...],
    config: TargetOrderingConfig,
) -> OrderingDecision:
    """Choose which candidate the orchestrator should execute first.

    See module docstring for the two-stage hard-guard algorithm.
    """

    n = len(candidates)

    # --- empty fan-in ----------------------------------------------------
    if n == 0:
        return OrderingDecision(
            chosen_index=None,
            local_scores=(),
            unlock_scores=(),
            priority_scores=(),
            mode=OrderingMode.SINGLE_BEST,
            reason=OrderingReason.NO_CANDIDATES,
        )

    local_scores = tuple(float(c.local_score) for c in candidates)

    def _argmax_local() -> int:
        # Stable argmax: ties broken by the lower segmentation index
        # to keep behaviour deterministic across runs.
        best_i = 0
        best_v = local_scores[0]
        for i in range(1, n):
            if local_scores[i] > best_v:
                best_v = local_scores[i]
                best_i = i
        return best_i

    top_idx = _argmax_local()

    # --- disabled: byte-identical legacy path ----------------------------
    if not config.enabled:
        zeros = tuple(0.0 for _ in range(n))
        return OrderingDecision(
            chosen_index=top_idx,
            local_scores=local_scores,
            unlock_scores=zeros,
            priority_scores=local_scores,
            mode=OrderingMode.SINGLE_BEST,
            reason=OrderingReason.DISABLED,
        )

    # --- enabled: compute unlock + priority ------------------------------
    unlock_scores = tuple(
        _unlock_score(i, candidates, config.blocker_graph) for i in range(n)
    )
    weight = float(config.unlock_weight)
    priority_scores = tuple(
        local_scores[i] + weight * unlock_scores[i] for i in range(n)
    )

    # argmax priority (stable, ties resolve to the lower index)
    best_p = 0
    best_pv = priority_scores[0]
    for i in range(1, n):
        if priority_scores[i] > best_pv:
            best_pv = priority_scores[i]
            best_p = i

    # No swap proposed: already on local_max.
    if best_p == top_idx:
        return OrderingDecision(
            chosen_index=top_idx,
            local_scores=local_scores,
            unlock_scores=unlock_scores,
            priority_scores=priority_scores,
            mode=OrderingMode.CLUTTER_AWARE,
            reason=OrderingReason.LOCAL_MAX,
        )

    # Hard guard: only accept the swap if the candidate's local score
    # is within ``max_local_score_drop`` of the legacy winner.
    drop = local_scores[top_idx] - local_scores[best_p]
    if drop <= float(config.max_local_score_drop):
        return OrderingDecision(
            chosen_index=best_p,
            local_scores=local_scores,
            unlock_scores=unlock_scores,
            priority_scores=priority_scores,
            mode=OrderingMode.CLUTTER_AWARE,
            reason=OrderingReason.UNLOCK_SWAP,
        )

    return OrderingDecision(
        chosen_index=top_idx,
        local_scores=local_scores,
        unlock_scores=unlock_scores,
        priority_scores=priority_scores,
        mode=OrderingMode.CLUTTER_AWARE,
        reason=OrderingReason.GUARD_BLOCKED_SWAP,
    )
