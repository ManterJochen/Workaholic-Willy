"""Active perception planner.

:class:`LateralOffsetViewpointPlanner` in
:mod:`src.robot.grasping.loop.pick_loop` sweeps four cardinal offsets
in a fixed order and never considers the scene. This module ships a
planner alongside it that:

* generates a small set of geometric viewpoint candidates around the
  current TCP: four lateral cardinal offsets plus two diagonals, and
  a vertical lift when ``vertical_offset_mm > 0``;
* keeps a typed history of viewpoints already observed (pose plus
  optional :class:`ViewpointSignals`) so the planner prefers diverse
  views and downweights ones close to a previously poor viewpoint;
* scores each candidate against a configurable
  :class:`ViewScoringPolicy` that combines history-diversity,
  workspace bounds, and optional scene-signal weights for occlusion,
  approach clearance, and reachability;
* filters candidates through a :class:`ViewpointSafetyCheck` Protocol,
  which an operator can wire to :class:`SafetyPreflight`;
* returns ``None`` once the policy budget is exhausted or every
  candidate fails the safety check.

Scope and non-goals
-------------------

* The planner is stateful but explicit: the caller records each
  observation via :meth:`record_observation`.
  :class:`BinPickingOrchestrator` does not call
  :meth:`record_observation`, so dropping this planner into the
  orchestrator gives a deterministic baseline. A higher-level
  service can feed scene signals without changing the Protocol.
* No vision model is invoked here. :class:`ViewpointSignals` is a
  small typed bag the caller fills in from the calculator's
  occlusion / approach-clearance telemetry.
* :class:`LateralOffsetViewpointPlanner` is not modified; it remains
  the deterministic fallback and this module ships alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from src.geometry import Frame, Pose

__all__ = [
    "AcceptAllViewpointSafetyCheck",
    "ScoringViewpointPlanner",
    "ViewScoringPolicy",
    "ViewpointCandidate",
    "ViewpointHistory",
    "ViewpointObservation",
    "ViewpointSafetyCheck",
    "ViewpointSignals",
    "WorkspaceBoxSafetyCheck",
]


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ViewpointSignals:
    """Scene measurements attached to a viewpoint after observation.

    Every field is optional; the caller supplies whichever signals
    the calculator/perception stack produced for that frame. Values
    that are ``None`` are treated as "no information" by the scoring
    function, i.e. they neither help nor hurt the candidate.

    Attributes
    ----------
    occlusion_ratio
        Fraction of the selected target's mask that was occluded in
        the captured frame. Range ``[0.0, 1.0]``.
    approach_clearance_mm
        Approach-cone clearance for the best candidate, in
        millimetres. Larger is better.
    reachability_score
        Optional ``[0.0, 1.0]`` score: how reachable the best
        candidate is from this viewpoint (IK headroom, joint margin).
    candidate_count
        Number of valid grasp candidates produced from this view.
        Zero is a strong negative signal; the planner penalises
        revisiting a region where the calculator produced nothing.
    """

    occlusion_ratio: Optional[float] = None
    approach_clearance_mm: Optional[float] = None
    reachability_score: Optional[float] = None
    candidate_count: Optional[int] = None

    def __post_init__(self) -> None:
        if self.occlusion_ratio is not None and not (
            0.0 <= self.occlusion_ratio <= 1.0
        ):
            raise ValueError(
                "occlusion_ratio must be in [0.0, 1.0]; "
                f"got {self.occlusion_ratio}"
            )
        if (
            self.approach_clearance_mm is not None
            and self.approach_clearance_mm < 0.0
        ):
            raise ValueError(
                "approach_clearance_mm must be non-negative; "
                f"got {self.approach_clearance_mm}"
            )
        if self.reachability_score is not None and not (
            0.0 <= self.reachability_score <= 1.0
        ):
            raise ValueError(
                "reachability_score must be in [0.0, 1.0]; "
                f"got {self.reachability_score}"
            )
        if self.candidate_count is not None and self.candidate_count < 0:
            raise ValueError(
                f"candidate_count must be non-negative; got {self.candidate_count}"
            )


@dataclass(frozen=True, slots=True)
class ViewpointObservation:
    """A pose that was visited, together with the signals captured there."""

    pose: Pose
    signals: ViewpointSignals = field(default_factory=ViewpointSignals)
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ViewpointCandidate:
    """A pose the planner is considering, scored against the policy."""

    pose: Pose
    score: float
    reason: str
    telemetry: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ViewpointHistory:
    """Mutable typed log of viewpoints already observed this session.

    Kept as a small list so the planner can compute diversity scores
    cheaply. The orchestrator/service is responsible for appending
    via :meth:`record`.
    """

    observations: list[ViewpointObservation] = field(default_factory=list)

    def record(self, observation: ViewpointObservation) -> None:
        self.observations.append(observation)

    @property
    def poses(self) -> tuple[Pose, ...]:
        return tuple(obs.pose for obs in self.observations)

    @property
    def best_observation(self) -> Optional[ViewpointObservation]:
        """Observation with the highest informational value.

        Informational value rewards low occlusion and at least one
        grasp candidate. Returns ``None`` when the log is empty.
        """

        if not self.observations:
            return None
        scored: list[tuple[float, ViewpointObservation]] = []
        for obs in self.observations:
            occ = obs.signals.occlusion_ratio
            n = obs.signals.candidate_count
            score = 0.0
            if occ is not None:
                score += 1.0 - occ
            if n is not None and n > 0:
                score += 1.0
            scored.append((score, obs))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]


# ---------------------------------------------------------------------------
# Safety check Protocol + built-ins
# ---------------------------------------------------------------------------


@runtime_checkable
class ViewpointSafetyCheck(Protocol):
    """Predicate over a candidate camera viewpoint.

    Returns whether ``pose`` is acceptable as the next viewpoint. Real
    deployments wire this to :class:`SafetyPreflight`. The Protocol is
    narrow so the planner does not depend on the full safety stack.
    """

    def is_safe(self, pose: Pose) -> bool:
        ...


@dataclass(frozen=True, slots=True)
class AcceptAllViewpointSafetyCheck:
    """No-op safety check. Accepts every viewpoint.

    The default, so the planner stays drop-in compatible with callers
    that do not wire :class:`SafetyPreflight`.
    """

    def is_safe(self, pose: Pose) -> bool:  # noqa: ARG002
        return True


@dataclass(frozen=True, slots=True)
class WorkspaceBoxSafetyCheck:
    """Axis-aligned workspace bound for camera viewpoints.

    Built-in convenience check that refuses any viewpoint outside the
    given box. Bounds are in :attr:`Frame.BASE` millimetres.
    """

    center_mm: tuple[float, float, float]
    half_extents_mm: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.center_mm) != 3:
            raise ValueError(
                f"center_mm must have 3 components; got {self.center_mm!r}"
            )
        if len(self.half_extents_mm) != 3 or any(
            h < 0.0 for h in self.half_extents_mm
        ):
            raise ValueError(
                "half_extents_mm must have 3 non-negative components; "
                f"got {self.half_extents_mm!r}"
            )

    def is_safe(self, pose: Pose) -> bool:
        pos = np.asarray(pose.position_mm, dtype=np.float64)
        for axis in range(3):
            lo = self.center_mm[axis] - self.half_extents_mm[axis]
            hi = self.center_mm[axis] + self.half_extents_mm[axis]
            if pos[axis] < lo or pos[axis] > hi:
                return False
        return True


# ---------------------------------------------------------------------------
# Scoring policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ViewScoringPolicy:
    """Bounded scoring weights and budgets for the planner.

    Attributes
    ----------
    max_viewpoints
        Hard cap on the number of viewpoints the planner will
        propose this session. Reaching the cap returns ``None``.
    lateral_offset_mm
        Magnitude of each cardinal lateral offset, in millimetres.
    vertical_offset_mm
        Magnitude of the lift candidate, in millimetres.
    diversity_weight
        Weight on "distance from existing history" in the candidate
        score. Larger values make the planner prefer viewpoints far
        from previously visited ones.
    occlusion_weight
        Weight applied to a candidate's expected occlusion
        improvement, estimated as the negative of the nearest
        history-occlusion (lower-is-better mirrored to higher-is-
        better). ``0`` disables.
    clearance_weight
        Weight applied to a candidate's expected approach
        clearance, estimated from the nearest history observation.
        ``0`` disables.
    reachability_weight
        Weight applied to a candidate's expected reachability,
        estimated from the nearest history observation. ``0``
        disables.
    min_score_to_act
        Floor on the top score. If the best candidate scores below
        this threshold the planner returns ``None``. Defaults to
        ``-inf`` so the planner never refuses on score alone: the
        safety check and the budget are the hard gates.
    """

    max_viewpoints: int = 4
    lateral_offset_mm: float = 50.0
    vertical_offset_mm: float = 30.0
    diversity_weight: float = 1.0
    occlusion_weight: float = 1.0
    clearance_weight: float = 0.5
    reachability_weight: float = 0.25
    min_score_to_act: float = float("-inf")

    def __post_init__(self) -> None:
        if self.max_viewpoints < 0:
            raise ValueError(
                f"max_viewpoints must be non-negative; got {self.max_viewpoints}"
            )
        for name in (
            "lateral_offset_mm",
            "vertical_offset_mm",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative; got {value}")
        for name in (
            "diversity_weight",
            "occlusion_weight",
            "clearance_weight",
            "reachability_weight",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative; got {value}")


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


_CARDINAL_OFFSETS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("+x", (+1.0, 0.0, 0.0)),
    ("-x", (-1.0, 0.0, 0.0)),
    ("+y", (0.0, +1.0, 0.0)),
    ("-y", (0.0, -1.0, 0.0)),
    ("+xy", (+0.70710678, +0.70710678, 0.0)),
    ("-xy", (-0.70710678, -0.70710678, 0.0)),
)


@dataclass(slots=True)
class ScoringViewpointPlanner:
    """Score-and-filter next-best-view planner.

    Drop-in compatible with the :class:`ViewpointPlanner` Protocol
    defined in :mod:`src.robot.grasping.loop.pick_loop`. The
    orchestrator calls :meth:`next_viewpoint` with the current TCP
    and the chronological tuple of viewpoints visited so far; a
    higher-level service may additionally call
    :meth:`record_observation` after each capture to feed scene
    signals into the score for subsequent calls.

    The planner is deterministic for a fixed input: candidate order,
    scoring, and tie-breaking are all stable.
    """

    policy: ViewScoringPolicy = field(default_factory=ViewScoringPolicy)
    safety_check: ViewpointSafetyCheck = field(
        default_factory=AcceptAllViewpointSafetyCheck
    )
    history: ViewpointHistory = field(default_factory=ViewpointHistory)
    # Count of candidates proposed; enforces ``max_viewpoints``
    # without depending on the orchestrator's ``history`` argument,
    # which counts visited poses, not proposed ones. The two normally
    # agree; the planner stays correct when a caller drops a
    # proposal.
    _proposed: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_observation(self, observation: ViewpointObservation) -> None:
        """Append ``observation`` to the typed history.

        The caller invokes this after each capture; ``next_viewpoint``
        does not.
        """

        self.history.record(observation)

    @property
    def last_selection_reason(self) -> Optional[str]:
        """Reason of the most recently recorded observation, if any."""

        if not self.history.observations:
            return None
        return self.history.observations[-1].reason or None

    def candidates(self, current_tcp: Pose) -> tuple[ViewpointCandidate, ...]:
        """Return the scored candidate list for diagnostics.

        Does not consume the proposal budget and does not mutate
        history. The caller can inspect the planner's ranking
        without committing to a move.
        """

        return self._score_candidates(current_tcp, tuple(self.history.poses))

    def next_viewpoint(
        self,
        *,
        current_tcp: Pose,
        history: Sequence[Pose],
    ) -> Optional[Pose]:
        """Propose the next viewpoint, or :data:`None` when exhausted.

        Parameters
        ----------
        current_tcp
            Live TCP pose in :attr:`Frame.BASE`.
        history
            Chronological viewpoints already executed by the
            orchestrator this session. The planner takes the maximum
            of this length and its own internal proposal counter to
            decide whether the budget is exhausted; either side
            running out exhausts the planner.
        """

        executed = len(history)
        if max(executed, self._proposed) >= self.policy.max_viewpoints:
            return None
        # Merge the executed history with any signals the caller
        # already pushed via record_observation. Scoring runs against
        # the union: the planner diversifies away from both visited
        # and proposed-but-not-yet-visited poses.
        history_poses: list[Pose] = list(history)
        for obs in self.history.observations:
            if not _pose_already_in(history_poses, obs.pose):
                history_poses.append(obs.pose)
        scored = self._score_candidates(current_tcp, tuple(history_poses))
        safe = [c for c in scored if self.safety_check.is_safe(c.pose)]
        if not safe:
            return None
        # Stable sort by descending score, then by the candidate name
        # held in ``reason``, for deterministic tie-breaking.
        safe.sort(key=lambda c: (-c.score, c.reason))
        top = safe[0]
        if top.score < self.policy.min_score_to_act:
            return None
        self._proposed += 1
        return top.pose

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        current_tcp: Pose,
        history_poses: Sequence[Pose],
    ) -> tuple[ViewpointCandidate, ...]:
        candidates: list[ViewpointCandidate] = []
        base = np.asarray(current_tcp.position_mm, dtype=np.float64)
        history_arr = (
            np.stack(
                [np.asarray(p.position_mm, dtype=np.float64) for p in history_poses]
            )
            if history_poses
            else None
        )
        for name, direction in _CARDINAL_OFFSETS:
            offset = np.asarray(direction, dtype=np.float64) * float(
                self.policy.lateral_offset_mm
            )
            pose = self._make_pose(current_tcp, base + offset, f"viewpoint_{name}")
            candidates.append(
                self._score_one(pose, name, history_arr, history_poses)
            )
        # Vertical lift candidate.
        if self.policy.vertical_offset_mm > 0.0:
            offset = np.array([0.0, 0.0, self.policy.vertical_offset_mm])
            pose = self._make_pose(current_tcp, base + offset, "viewpoint_+z")
            candidates.append(
                self._score_one(pose, "+z", history_arr, history_poses)
            )
        return tuple(candidates)

    def _score_one(
        self,
        pose: Pose,
        name: str,
        history_arr: Optional[np.ndarray],
        history_poses: Sequence[Pose],
    ) -> ViewpointCandidate:
        # Diversity term: minimum distance to any history pose, in mm.
        # With no history the policy lateral offset is the neutral
        # diversity baseline, which keeps the score bounded.
        if history_arr is None or history_arr.shape[0] == 0:
            diversity = float(self.policy.lateral_offset_mm)
        else:
            pos = np.asarray(pose.position_mm, dtype=np.float64)
            diffs = history_arr - pos[None, :]
            diversity = float(np.min(np.linalg.norm(diffs, axis=1)))
        # Scene-signal terms come from the nearest history
        # observation. occlusion_ratio is mirrored (lower is better)
        # so all three terms are higher-is-better.
        nearest = _nearest_observation(pose, history_poses, self.history.observations)
        occlusion_term = 0.0
        clearance_term = 0.0
        reachability_term = 0.0
        zero_candidate_penalty = 0.0
        if nearest is not None:
            signals = nearest.signals
            if signals.occlusion_ratio is not None:
                occlusion_term = 1.0 - float(signals.occlusion_ratio)
            if signals.approach_clearance_mm is not None:
                clearance_term = float(signals.approach_clearance_mm)
            if signals.reachability_score is not None:
                reachability_term = float(signals.reachability_score)
            if (
                signals.candidate_count is not None
                and signals.candidate_count == 0
            ):
                # Subtract one ``diversity_weight`` for proposing near a
                # viewpoint that produced zero candidates.
                zero_candidate_penalty = -float(self.policy.diversity_weight)
        score = (
            self.policy.diversity_weight * diversity / max(
                self.policy.lateral_offset_mm, 1e-6
            )
            + self.policy.occlusion_weight * occlusion_term
            + self.policy.clearance_weight * clearance_term
            + self.policy.reachability_weight * reachability_term
            + zero_candidate_penalty
        )
        telemetry = {
            "candidate": name,
            "diversity_mm": diversity,
            "occlusion_term": occlusion_term,
            "clearance_term": clearance_term,
            "reachability_term": reachability_term,
            "zero_candidate_penalty": zero_candidate_penalty,
        }
        return ViewpointCandidate(
            pose=pose, score=score, reason=name, telemetry=telemetry
        )

    @staticmethod
    def _make_pose(current_tcp: Pose, new_position: np.ndarray, label: str) -> Pose:
        return Pose(
            position_mm=new_position,
            quaternion_xyzw=np.asarray(
                current_tcp.quaternion_xyzw, dtype=np.float64
            ).copy(),
            frame=Frame.BASE,
            label=label,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pose_already_in(history: Sequence[Pose], pose: Pose) -> bool:
    target = np.asarray(pose.position_mm, dtype=np.float64)
    for existing in history:
        existing_pos = np.asarray(existing.position_mm, dtype=np.float64)
        if np.allclose(existing_pos, target, atol=1e-6):
            return True
    return False


def _nearest_observation(
    pose: Pose,
    history_poses: Sequence[Pose],
    observations: Sequence[ViewpointObservation],
) -> Optional[ViewpointObservation]:
    if not observations:
        return None
    target = np.asarray(pose.position_mm, dtype=np.float64)
    best: Optional[tuple[float, ViewpointObservation]] = None
    for obs in observations:
        if obs not in observations:  # pragma: no cover (defensive)
            continue
        obs_pos = np.asarray(obs.pose.position_mm, dtype=np.float64)
        dist = float(np.linalg.norm(obs_pos - target))
        if best is None or dist < best[0]:
            best = (dist, obs)
    return best[1] if best is not None else None
