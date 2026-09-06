"""Scoring a pick's candidates without changing it: the shadow half of the learned ranker.

Shadow first. The ranker computes, the telemetry records, and the order the robot is handed is the
geometric one. That is the repo's pattern for anything learned (`success_probability` ships the same
way, with its rerankers at weight zero), and a measurement asks for it here: a ranker fitted under
`bootstrap_jaw_v1` beats the calculator's order on this repo's own candidates and still loses to
picking at random, so a version that acted would act wrongly.

Pure and never-throwing. Everything here takes arrays and returns a dataclass. It opens no cell,
reads no file, and a failure anywhere returns a telemetry object saying what stopped it: a scorer
that raised inside a pick would turn a ranking opinion into a dropped attempt.

What it needs from the caller is what the pick loop already has: the candidates, the target's point
cloud, the other objects' points, and the support plane. Nothing is estimated that the cell does not
measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.robot.grasping.deep.ranker.features import (
    FeatureSpec,
    environment_features,
    grasp_frame_offsets,
    object_centre_mm,
)
from src.robot.grasping.deep.ranker.runtime import GbtRanker

__all__ = ["ShadowRankerTelemetry", "score_candidates"]

#: Below this a cloud cannot support a centre estimate, let alone a local normal. Measured on the same
#: clouds the centre estimator was measured on: the estimate is quoted over objects with >= 200 points,
#: so claiming anything under that would be quoting a number outside the range it was measured in.
_MINIMUM_CLOUD_POINTS = 200


@dataclass(frozen=True, slots=True)
class ShadowRankerTelemetry:
    """What the ranker thought, and whether it would have changed anything.

    ``would_change_top1`` is the number that matters while the ranker is in shadow: it answers on
    every real pick whether turning it on would do anything.
    """

    #: False with a `reason` when the ranker could not be asked at all.
    scored: bool
    reason: str = ""
    model_sha256: str = ""
    spec: str = ""
    candidates: int = 0
    #: The ranker's score for each candidate, in the order the calculator produced them.
    scores: tuple[float, ...] = ()
    #: Index the ranker would have put first. -1 when it was not asked.
    ranker_top1: int = -1
    #: True when that is not the candidate the robot is about to execute.
    would_change_top1: bool = False
    #: The object centre the features were built on, so a later reader can tell a bad score from a
    #: bad centre; they look identical in the score alone.
    object_centre_mm: tuple[float, float, float] | None = None
    features: tuple[tuple[float, ...], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        """The flat shape telemetry consumers read. Prefixed, because a bare ``top1`` in a shared
        namespace is a key two subsystems will eventually both want."""
        payload: dict[str, Any] = {
            "deep_ranker_scored": self.scored,
            "deep_ranker_candidates": self.candidates,
            "deep_ranker_would_change_top1": self.would_change_top1,
        }
        if not self.scored:
            payload["deep_ranker_reason"] = self.reason
            return payload
        payload.update({
            "deep_ranker_spec": self.spec,
            "deep_ranker_sha256": self.model_sha256[:16],
            "deep_ranker_top1": self.ranker_top1,
            "deep_ranker_top_score": max(self.scores) if self.scores else 0.0,
            "deep_ranker_score_span": (max(self.scores) - min(self.scores)) if self.scores else 0.0,
        })
        if self.object_centre_mm is not None:
            payload["deep_ranker_object_centre_mm"] = [
                round(v, 2) for v in self.object_centre_mm]
        return payload


#: The two name families a candidate may carry, per field, runtime name first.
#:
#: The runtime candidate is a `GraspPoint`, whose fields are `position` / `axis` / `grip_width_mm`.
#: The training-side row is `datagen.grasps.labels.GraspLabel`, whose fields are `position_mm` /
#: `closing_axis` / `width_mm`. A family missing from this table makes `_candidate_vectors` return
#: None for every candidate of that shape, and `score_candidates` then reports
#: `deep_ranker_reason: 'a candidate is missing its pose'` on every pick, silently: whatever the
#: shipped `held_jaw_v1` artifact is worth, none of it then reaches a pick.
#:
#: A stand-in candidate carrying only the training names is not a `GraspPoint` and proves nothing
#: about the runtime path.
_CANDIDATE_FIELDS: "tuple[tuple[str, ...], ...]" = (
    ("position", "position_mm"),
    ("approach",),
    ("axis", "closing_axis"),
    ("grip_width_mm", "width_mm"),
)


def _first_attr(candidate: Any, names: "tuple[str, ...]") -> Any:
    for name in names:
        value = getattr(candidate, name, None)
        if value is not None:
            return value
    raise AttributeError(f"none of {names} on {type(candidate).__name__}")


def _candidate_vectors(candidate: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Pull (position, approach, closing axis, width) off a candidate, or ``None`` if it is not one.

    Duck-typed on purpose: this must work for a `GraspPoint`, for the dataclasses the sim runners
    pass around, and for any plain object carrying the same attributes, without `deep/` importing
    any of them.

    Both name families, and the runtime one first; see `_CANDIDATE_FIELDS`. Accepting both rather
    than renaming either is deliberate: the training rows are what the ranker was fitted on and
    their names are the feature contract, while `GraspPoint` is a frozen value object the pick path
    depends on. Renaming either side to satisfy this function would move a breakage rather than fix
    it.
    """
    try:
        position = np.asarray(_first_attr(candidate, _CANDIDATE_FIELDS[0]),
                              dtype=np.float64).reshape(3)
        approach = np.asarray(_first_attr(candidate, _CANDIDATE_FIELDS[1]),
                              dtype=np.float64).reshape(3)
        axis = np.asarray(_first_attr(candidate, _CANDIDATE_FIELDS[2]), dtype=np.float64).reshape(3)
        width = float(_first_attr(candidate, _CANDIDATE_FIELDS[3]))
    except (AttributeError, TypeError, ValueError):
        return None
    return position, approach, axis, width


def score_candidates(
    ranker: GbtRanker,
    spec: FeatureSpec,
    candidates: "list[Any] | tuple[Any, ...]",
    *,
    target_points_mm: np.ndarray | None,
    obstacle_points_mm: np.ndarray | None = None,
    support_z_mm: float = 0.0,
    observedness: float = 1.0,
    executed_index: int = 0,
) -> ShadowRankerTelemetry:
    """Score a pick's candidates and report what the ranker would have chosen. Never raises.

    ``executed_index`` is the candidate the robot is actually about to take: 0 in the default path,
    because the calculator sorts descending and the pick loop takes the first. It is a parameter
    rather than an assumption because a rerank upstream can already have moved it.
    """
    if not candidates:
        return ShadowRankerTelemetry(False, reason="no candidates")
    if target_points_mm is None:
        return ShadowRankerTelemetry(False, reason="no target cloud")
    points = np.atleast_2d(np.asarray(target_points_mm, dtype=np.float64))
    if len(points) < _MINIMUM_CLOUD_POINTS:
        return ShadowRankerTelemetry(
            False, reason=f"target cloud has {len(points)} points, under {_MINIMUM_CLOUD_POINTS}")

    try:
        centre = object_centre_mm(points, support_z_mm=support_z_mm)
    except ValueError as exc:
        return ShadowRankerTelemetry(False, reason=f"object centre: {exc}")

    rows: list[list[float]] = []
    for candidate in candidates:
        vectors = _candidate_vectors(candidate)
        if vectors is None:
            return ShadowRankerTelemetry(False, reason="a candidate is missing its pose")
        position, approach, axis, width = vectors
        try:
            along = grasp_frame_offsets(centre, position, approach, axis)
        except ValueError as exc:
            # The GraspPoint contract promises the axis perpendicular to the approach. One that
            # does not is a broken candidate, and scoring the rest while silently dropping it would
            # report a ranking over a different candidate set than the one the robot has.
            return ShadowRankerTelemetry(False, reason=f"candidate geometry: {exc}")
        values = {
            "width_mm": width,
            "centre_along_axis_mm": along[0],
            "centre_along_approach_mm": along[1],
            "centre_along_binormal_mm": along[2],
            **environment_features(
                position, approach, axis, width, points,
                obstacle_points_mm=obstacle_points_mm, support_z_mm=support_z_mm,
                observedness=observedness),
        }
        try:
            rows.append([float(values[name]) for name in spec.features])
        except KeyError as exc:
            return ShadowRankerTelemetry(
                False, reason=f"spec {spec.name!r} wants a feature this seam does not compute: {exc}")

    matrix = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(matrix).all():
        # `jaw_clearance_mm` is +inf when nothing is known to be in the way, which is honest as a
        # value and unusable as a model input. Clipped to a large finite number rather than refused:
        # "nothing near it" is the common case in a sparse scene, not an error.
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0e4, neginf=-1.0e4)

    try:
        scores = ranker.score(matrix)
    except ValueError as exc:
        return ShadowRankerTelemetry(False, reason=f"scoring: {exc}")

    top1 = int(np.argmax(scores))
    return ShadowRankerTelemetry(
        scored=True,
        model_sha256=ranker.sha256,
        spec=spec.name,
        candidates=len(candidates),
        scores=tuple(float(v) for v in scores),
        ranker_top1=top1,
        would_change_top1=top1 != int(executed_index),
        object_centre_mm=(float(centre[0]), float(centre[1]), float(centre[2])),
        features=tuple(tuple(float(v) for v in row) for row in matrix),
    )
