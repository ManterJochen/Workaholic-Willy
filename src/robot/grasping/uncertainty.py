"""Uncertainty fusion carrier.

This module holds the typed surface of the production uncertainty
layer, which the owner has locked:

* :class:`UncertaintyChannel`, the string enum naming the seven
  signal channels: depth_confidence, mask_confidence,
  occlusion_corridor_risk, feasibility_margin,
  verification_residual, topology_risk and semantic_confidence.
* :class:`UncertaintyChannelValues`, the frozen carrier of
  per-channel values in ``[0, 1]``, where ``None`` means not
  measured.
* :class:`UncertaintyWeights`, the frozen carrier of per-channel
  non-negative weights.
* :class:`UncertaintyMonotoneMap`, a non-decreasing piecewise-linear
  remap of one channel.
* :class:`UncertaintyCalibration`, the bundle of weights and
  per-channel maps, where ``identity()`` gives a no-op map per
  channel and the default weights.
* :class:`UncertaintySnapshot`, the JSON-safe telemetry payload.
* :func:`fuse_uncertainty`, the pure fusion function: a convex
  combination of the remapped channel values weighted by
  :class:`UncertaintyWeights`. A missing channel is skipped, and so
  is a channel whose weight is zero.

The fusion is monotone by construction: each channel passes through a
non-decreasing map and is then multiplied by a non-negative weight
before it joins the convex combination, so raising any input value can
never lower the fused output.

Every type is frozen and slotted, which keeps it safe across worker
boundaries, and JSON-safe through :meth:`to_dict()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from src.robot.grasping.constants import (
    UNCERTAINTY_LOG_FILE,
    create_grasping_logger,
)

#: The fusion itself is pure, but its verdict terminates a pick as
#: ``AutonomousGraspOutcome.UNCERTAINTY_FAIL_CLOSED``, which leaves the operator
#: holding an outcome string. These lines carry the arithmetic behind it: which
#: channels contributed, what they fused to, and against which threshold.
logger = create_grasping_logger("UncertaintyFusion", UNCERTAINTY_LOG_FILE)

__all__ = [
    "UncertaintyChannel",
    "UncertaintyChannelValues",
    "UncertaintyWeights",
    "UncertaintyMonotoneMap",
    "UncertaintyCalibration",
    "UncertaintySnapshot",
    "fuse_uncertainty",
    "apply_uncertainty_ranking_penalty",
    "per_candidate_uncertainty",
    "should_bias_recovery_for_uncertainty",
]

# A sentinel default mirroring GraspingUncertaintyConfig. 1.01 is strictly
# greater than any valid disagreement, which is bounded above by 1.0 because
# remapped channel values live in ``[0, 1]``, so the disagreement gate never
# trips unless the operator lowers the threshold in robot.yaml. The schema knob
# and this default hold the same value, so the two move together.
_DISAGREEMENT_OFF_SENTINEL: float = 1.01


class UncertaintyChannel(str, Enum):
    """The seven typed uncertainty signals fused into one per-grasp uncertainty score.

    Each channel is a normalised value in ``[0, 1]`` coming from a different part of the pipeline
    that runs from perception to grasp. They are kept separate rather than collapsed into one opaque
    number, so the fusion layer can weight and calibrate each source independently. A grasp fails for
    distinct reasons, and these seven cover them:

    * ``DEPTH_CONFIDENCE``, trust in the stereo or RGB-D depth at the grasp, which is low on a
      reflective, thin or distant surface.
    * ``MASK_CONFIDENCE``, how clean the segmentation mask of the object is, which is low on a fuzzy
      or merged mask.
    * ``OCCLUSION_CORRIDOR_RISK``, the risk that the straight-line approach corridor to the grasp is
      occluded or blocked.
    * ``FEASIBILITY_MARGIN``, how far the grasp sits from the IK and reachability feasibility
      boundary.
    * ``VERIFICATION_RESIDUAL``, the disagreement left over after the post-grasp verification
      cross-check.
    * ``TOPOLOGY_RISK``, the risk from the local shape of the object, which covers thin, deformable
      and ambiguous parts.
    * ``SEMANTIC_CONFIDENCE``, confidence in the class the detector assigned to the object.

    The set is deliberately fixed at seven. Adding an eighth is a coordinated schema change across
    this enum, the fusion weights and the on-disk calibration artifact, not a drop-in.
    """

    DEPTH_CONFIDENCE = "depth_confidence"
    MASK_CONFIDENCE = "mask_confidence"
    OCCLUSION_CORRIDOR_RISK = "occlusion_corridor_risk"
    FEASIBILITY_MARGIN = "feasibility_margin"
    VERIFICATION_RESIDUAL = "verification_residual"
    TOPOLOGY_RISK = "topology_risk"
    SEMANTIC_CONFIDENCE = "semantic_confidence"


_CHANNEL_FIELDS: tuple[tuple[UncertaintyChannel, str], ...] = (
    (UncertaintyChannel.DEPTH_CONFIDENCE, "depth_confidence"),
    (UncertaintyChannel.MASK_CONFIDENCE, "mask_confidence"),
    (UncertaintyChannel.OCCLUSION_CORRIDOR_RISK, "occlusion_corridor_risk"),
    (UncertaintyChannel.FEASIBILITY_MARGIN, "feasibility_margin"),
    (UncertaintyChannel.VERIFICATION_RESIDUAL, "verification_residual"),
    (UncertaintyChannel.TOPOLOGY_RISK, "topology_risk"),
    (UncertaintyChannel.SEMANTIC_CONFIDENCE, "semantic_confidence"),
)


def _validate_unit_or_none(name: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real number or None; got {value!r}")
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if not 0.0 <= fv <= 1.0:
        raise ValueError(
            f"{name} must lie in [0, 1]; got {value!r}"
        )
    return fv


def _validate_non_negative(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real number; got {value!r}")
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if fv < 0.0:
        raise ValueError(f"{name} must be non-negative; got {value!r}")
    return fv


@dataclass(frozen=True, slots=True)
class UncertaintyChannelValues:
    """Per-channel raw signals in ``[0, 1]``.

    Each field defaults to :data:`None`, meaning not measured. A
    ``None`` channel does not contribute to the fused value.
    """

    depth_confidence: Optional[float] = None
    mask_confidence: Optional[float] = None
    occlusion_corridor_risk: Optional[float] = None
    feasibility_margin: Optional[float] = None
    verification_residual: Optional[float] = None
    topology_risk: Optional[float] = None
    semantic_confidence: Optional[float] = None

    def __post_init__(self) -> None:
        for _, name in _CHANNEL_FIELDS:
            object.__setattr__(
                self, name,
                _validate_unit_or_none(name, getattr(self, name)),
            )

    def get(self, channel: UncertaintyChannel) -> Optional[float]:
        return getattr(self, channel.value)

    def to_dict(self) -> dict[str, Optional[float]]:
        return {name: getattr(self, name) for _, name in _CHANNEL_FIELDS}


@dataclass(frozen=True, slots=True)
class UncertaintyWeights:
    """Per-channel non-negative weights.

    The default reflects the locked priorities: the five channels
    that are always produced carry weight ``1.0``, while the two
    optional ones, ``topology_risk`` and ``semantic_confidence``,
    default to ``0.0``. An operator opts one in by raising its weight
    in ``robot.yaml``.
    """

    depth_confidence: float = 1.0
    mask_confidence: float = 1.0
    occlusion_corridor_risk: float = 1.0
    feasibility_margin: float = 1.0
    verification_residual: float = 1.0
    topology_risk: float = 0.0
    semantic_confidence: float = 0.0

    def __post_init__(self) -> None:
        for _, name in _CHANNEL_FIELDS:
            object.__setattr__(
                self, name,
                _validate_non_negative(name, getattr(self, name)),
            )

    def get(self, channel: UncertaintyChannel) -> float:
        return getattr(self, channel.value)

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for _, name in _CHANNEL_FIELDS}


@dataclass(frozen=True, slots=True)
class UncertaintyMonotoneMap:
    """Non-decreasing piecewise-linear map :math:`[0, 1] \\to [0, 1]`.

    The map is sampled at ``breakpoints`` (strictly increasing) and
    takes the corresponding ``values`` (non-decreasing). Inputs
    outside the breakpoint range are clamped to the nearest endpoint
    value.
    """

    breakpoints: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        bp = tuple(float(x) for x in self.breakpoints)
        vs = tuple(float(x) for x in self.values)
        if len(bp) != len(vs):
            raise ValueError(
                "UncertaintyMonotoneMap: breakpoints and values must "
                f"have the same length (got {len(bp)} vs {len(vs)})"
            )
        if len(bp) < 2:
            raise ValueError(
                "UncertaintyMonotoneMap requires at least two breakpoints"
            )
        for i in range(1, len(bp)):
            if bp[i] <= bp[i - 1]:
                raise ValueError(
                    "UncertaintyMonotoneMap breakpoints must be strictly "
                    f"increasing; got {bp!r}"
                )
            if vs[i] < vs[i - 1]:
                raise ValueError(
                    "UncertaintyMonotoneMap values must be non-decreasing; "
                    f"got {vs!r}"
                )
        object.__setattr__(self, "breakpoints", bp)
        object.__setattr__(self, "values", vs)

    @classmethod
    def identity(cls) -> "UncertaintyMonotoneMap":
        return cls(breakpoints=(0.0, 1.0), values=(0.0, 1.0))

    def apply(self, x: float) -> float:
        bp = self.breakpoints
        vs = self.values
        if x <= bp[0]:
            return vs[0]
        if x >= bp[-1]:
            return vs[-1]
        # A linear scan: N is small enough that a bisection buys nothing.
        for i in range(1, len(bp)):
            if x <= bp[i]:
                lo_x, hi_x = bp[i - 1], bp[i]
                lo_y, hi_y = vs[i - 1], vs[i]
                if hi_x == lo_x:
                    return hi_y
                t = (x - lo_x) / (hi_x - lo_x)
                return lo_y + t * (hi_y - lo_y)
        return vs[-1]  # pragma: no cover

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "breakpoints": list(self.breakpoints),
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class UncertaintyCalibration:
    """Calibration bundle: weights + per-channel monotone maps."""

    weights: UncertaintyWeights
    maps: Mapping[UncertaintyChannel, UncertaintyMonotoneMap]
    calibration_id: Optional[str] = None

    def __post_init__(self) -> None:
        # Every channel must have a map, so a missing one is filled with the identity.
        normalised: dict[UncertaintyChannel, UncertaintyMonotoneMap] = {}
        for ch, _ in _CHANNEL_FIELDS:
            normalised[ch] = self.maps.get(ch, UncertaintyMonotoneMap.identity())
        object.__setattr__(self, "maps", normalised)

    @classmethod
    def identity(cls) -> "UncertaintyCalibration":
        return cls(
            weights=UncertaintyWeights(),
            maps={ch: UncertaintyMonotoneMap.identity() for ch, _ in _CHANNEL_FIELDS},
            calibration_id=None,
        )

    def to_artifact(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "weights": self.weights.to_dict(),
            "maps": {
                ch.value: self.maps[ch].to_dict()
                for ch, _ in _CHANNEL_FIELDS
            },
        }

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "UncertaintyCalibration":
        w = artifact.get("weights", {})
        weights = UncertaintyWeights(
            depth_confidence=float(w.get("depth_confidence", 1.0)),
            mask_confidence=float(w.get("mask_confidence", 1.0)),
            occlusion_corridor_risk=float(w.get("occlusion_corridor_risk", 1.0)),
            feasibility_margin=float(w.get("feasibility_margin", 1.0)),
            verification_residual=float(w.get("verification_residual", 1.0)),
            topology_risk=float(w.get("topology_risk", 0.0)),
            semantic_confidence=float(w.get("semantic_confidence", 0.0)),
        )
        maps_in = artifact.get("maps", {})
        maps: dict[UncertaintyChannel, UncertaintyMonotoneMap] = {}
        for ch, name in _CHANNEL_FIELDS:
            entry = maps_in.get(name)
            if entry is None:
                maps[ch] = UncertaintyMonotoneMap.identity()
                continue
            maps[ch] = UncertaintyMonotoneMap(
                breakpoints=tuple(float(x) for x in entry["breakpoints"]),
                values=tuple(float(x) for x in entry["values"]),
            )
        return cls(
            weights=weights, maps=maps,
            calibration_id=artifact.get("calibration_id"),
        )


@dataclass(frozen=True, slots=True)
class UncertaintySnapshot:
    """Telemetry payload (always emitted).

    Attributes
    ----------
    fused
        Convex-combination fused uncertainty in ``[0, 1]``. ``0.0``
        when no channels are available.
    fused_available
        :data:`False` when every channel was missing or carried zero
        weight, meaning there was no signal, and :data:`True` otherwise.
    fail_closed
        :data:`True` when ``fused >= fail_closed_threshold`` and
        ``fused_available`` is :data:`True`.
    fail_closed_threshold
        The threshold the snapshot was evaluated against.
    channels
        JSON-safe mapping of the raw input channels.
    calibration_id
        Opaque identifier of the calibration artifact, or :data:`None`
        when the identity calibration was used.
    """

    fused: float
    fused_available: bool
    fail_closed: bool
    fail_closed_threshold: float
    channels: Mapping[str, Optional[float]] = field(default_factory=dict)
    calibration_id: Optional[str] = None
    # Additional fields, off by default. ``disagreement`` is
    # ``max(remapped) - min(remapped)`` over the contributing channels after
    # the remap, meaning those that are not None and carry a positive weight.
    # It is ``0.0`` when fewer than two channels contribute, since no pair is
    # left to disagree.
    disagreement: float = 0.0
    # The threshold the snapshot was evaluated against. A value of ``1.01`` or
    # more is the off-sentinel, because a disagreement cannot exceed it while
    # remapped values live in [0, 1].
    disagreement_threshold: float = _DISAGREEMENT_OFF_SENTINEL
    # :data:`True` when ``disagreement >= disagreement_threshold`` and
    # ``fused_available`` is :data:`True`.
    disagreement_triggered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "fused": float(self.fused),
            "fused_available": bool(self.fused_available),
            "fail_closed": bool(self.fail_closed),
            "fail_closed_threshold": float(self.fail_closed_threshold),
            "channels": {k: (None if v is None else float(v)) for k, v in self.channels.items()},
            "calibration_id": self.calibration_id,
            "disagreement": float(self.disagreement),
            "disagreement_threshold": float(self.disagreement_threshold),
            "disagreement_triggered": bool(self.disagreement_triggered),
        }

    @classmethod
    def disabled(cls, *, fail_closed_threshold: float) -> "UncertaintySnapshot":
        return cls(
            fused=0.0,
            fused_available=False,
            fail_closed=False,
            fail_closed_threshold=float(fail_closed_threshold),
            channels={name: None for _, name in _CHANNEL_FIELDS},
            calibration_id=None,
            disagreement=0.0,
            disagreement_threshold=_DISAGREEMENT_OFF_SENTINEL,
            disagreement_triggered=False,
        )


def fuse_uncertainty(
    values: UncertaintyChannelValues,
    calibration: UncertaintyCalibration,
    *,
    fail_closed_threshold: float,
    channel_disagreement_threshold: float = _DISAGREEMENT_OFF_SENTINEL,
) -> UncertaintySnapshot:
    """Convex-combine remapped channel values into an :class:`UncertaintySnapshot`.

    A channel contributes if and only if its value is not :data:`None`
    and its weight is strictly positive.

    The ``channel_disagreement_threshold`` knob, which defaults to
    :data:`_DISAGREEMENT_OFF_SENTINEL` and therefore never trips,
    records the spread ``max(remapped) - min(remapped)`` across the
    contributing channels and sets
    :attr:`UncertaintySnapshot.disagreement_triggered` when the spread
    reaches it. It leaves the fusion arithmetic alone: disagreement is
    an additional telemetry and control signal.
    """

    num = 0.0
    den = 0.0
    contributing: list[float] = []
    for ch, name in _CHANNEL_FIELDS:
        raw = getattr(values, name)
        if raw is None:
            continue
        w = calibration.weights.get(ch)
        if w <= 0.0:
            continue
        remapped = calibration.maps[ch].apply(float(raw))
        num += w * remapped
        den += w
        contributing.append(remapped)
    disagreement_threshold = float(channel_disagreement_threshold)
    if den <= 0.0:
        # Not a failure, but worth a line: every channel was unmeasured or carried
        # zero weight, so the gate below cannot fire whatever the scene contains.
        logger.debug(
            "Uncertainty not available: no channel contributed (calibration %s)",
            calibration.calibration_id,
        )
        return UncertaintySnapshot(
            fused=0.0,
            fused_available=False,
            fail_closed=False,
            fail_closed_threshold=float(fail_closed_threshold),
            channels=values.to_dict(),
            calibration_id=calibration.calibration_id,
            disagreement=0.0,
            disagreement_threshold=disagreement_threshold,
            disagreement_triggered=False,
        )
    fused = num / den
    if len(contributing) >= 2:
        disagreement = max(contributing) - min(contributing)
    else:
        disagreement = 0.0
    disagreement_triggered = bool(disagreement >= disagreement_threshold)
    if fused >= float(fail_closed_threshold):
        logger.warning(
            "Uncertainty FAIL-CLOSED: fused %.3f >= threshold %.3f over %d channel(s) "
            "(calibration %s, channels %s)",
            fused,
            float(fail_closed_threshold),
            len(contributing),
            calibration.calibration_id,
            values.to_dict(),
        )
    elif disagreement_triggered:
        # The channels disagree by more than the operator allows. The fused number
        # is below the gate, and it is an average of signals that do not agree.
        logger.warning(
            "Uncertainty channel disagreement %.3f >= %.3f (fused %.3f, calibration %s)",
            disagreement,
            disagreement_threshold,
            fused,
            calibration.calibration_id,
        )
    else:
        logger.debug(
            "Uncertainty fused %.3f over %d channel(s) (threshold %.3f)",
            fused,
            len(contributing),
            float(fail_closed_threshold),
        )
    return UncertaintySnapshot(
        fused=float(fused),
        fused_available=True,
        fail_closed=bool(fused >= float(fail_closed_threshold)),
        fail_closed_threshold=float(fail_closed_threshold),
        channels=values.to_dict(),
        calibration_id=calibration.calibration_id,
        disagreement=float(disagreement),
        disagreement_threshold=disagreement_threshold,
        disagreement_triggered=disagreement_triggered,
    )


def apply_uncertainty_ranking_penalty(
    top_score: Optional[float],
    snapshot: UncertaintySnapshot,
    weight: float,
) -> tuple[Optional[float], bool]:
    """Compute a penalised top score (scene-level).

    Returns ``(penalised_top_score, applied)``. The penalty
    ``weight * snapshot.fused`` is subtracted from ``top_score`` and
    the result is clamped to ``[0.0, 1.0]``. The penalty applies
    exactly when:

    * ``top_score`` is not :data:`None`,
    * ``snapshot.fused_available`` is :data:`True`,
    * ``weight > 0.0``.

    Otherwise ``penalised_top_score == top_score`` and ``applied`` is
    False, which leaves the ranking exactly as it was.
    """

    if top_score is None or not snapshot.fused_available or float(weight) <= 0.0:
        return (top_score, False)
    penalised = float(top_score) - float(weight) * float(snapshot.fused)
    if penalised < 0.0:
        penalised = 0.0
    elif penalised > 1.0:
        penalised = 1.0
    return (penalised, True)


_CORRIDOR_RISK_KEY = "corridor_blockage_confidence"


def per_candidate_uncertainty(point: Any) -> Optional[float]:
    """Per-candidate corridor-risk uncertainty in ``[0, 1]``, or ``None``.

    Reads the ``corridor_blockage_confidence`` that the optional corridor-risk producer of the
    calculator stamps on each surviving candidate; that producer is off by default. The corridor
    analyzer already returns blockage in ``[0, 1]``, where higher means more blocked and so more
    uncertain, so this is a direct clamped read with no inversion and no re-derivation from the
    geometric score. It is physically independent of the antipodal, width and depth score, because it
    depends on the approach direction of the candidate, which that score ignores, so it cannot be
    circular.

    Returns :data:`None` when the candidate carries no corridor signal, either because the producer is
    disabled or because this candidate has none. The caller then stays additive and the re-ranker
    hard-no-ops rather than re-sorting on a fabricated value.
    """
    metadata = getattr(point, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get(_CORRIDOR_RISK_KEY)
    if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if math.isnan(value) or math.isinf(value):
        return None
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def should_bias_recovery_for_uncertainty(
    snapshot: Optional[UncertaintySnapshot],
    *,
    recovery_aggressive_threshold: float,
) -> bool:
    """Recovery-bias gate.

    Returns :data:`True` when the orchestrator should reorder the
    recovery action sequence so perception actions
    (``NEXT_VIEWPOINT`` / ``RESCAN``) come first.

    It triggers exactly when:

    * ``snapshot`` is not :data:`None`,
    * ``snapshot.fused_available`` is :data:`True`,
    * ``snapshot.fused >= recovery_aggressive_threshold``.

    The default threshold of ``1.01`` mirrors
    :class:`GraspingUncertaintyConfig.recovery_aggressive_threshold`,
    so the bias never trips unless the operator opts in.
    """

    if snapshot is None or not snapshot.fused_available:
        return False
    return float(snapshot.fused) >= float(recovery_aggressive_threshold)
