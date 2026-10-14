"""GraspPoint, the immutable container for a candidate grasp.

A GraspPoint fully specifies how a parallel-jaw gripper approaches and closes on an
object:

* ``position`` is the 3D anchor point, in mm, in either the camera or the base frame.
* ``approach`` is the unit vector the gripper moves along toward the object before
  closing. For a top-down grasp in the robot base frame it is ``[0, 0, -1]``.
* ``axis`` is the unit vector along the line connecting the two finger pads,
  perpendicular to ``approach``. The gripper closes across this axis.
* ``grip_width_mm`` is the expected finger-pad separation at contact.
* ``score`` is the quality in ``[0, 1]``, where higher is better.
* ``frame`` is the coordinate frame the vectors live in.

Numerics contract
-----------------
Distances are millimetres. Direction vectors are unit-norm, which is enforced at
construction. The dataclass is frozen and the underlying numpy arrays are marked
read-only, so a consumer cannot silently mutate ``gp.position[0]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.robot.grasping.geometry import as_vec3
from src.robot.grasping.geometry.grasp_frame import pose_from_grasp_axes


class GraspFrame(StrEnum):
    """Coordinate frame a GraspPoint is expressed in."""
    CAMERA = "camera"
    BASE = "base"


# The tolerances for vector validation.
_UNIT_TOL = 1e-3        # acceptable deviation from |v| = 1
_ZERO_TOL = 1e-9        # below this the vector is considered the zero vector


def _normalise_unit(v: np.ndarray, name: str) -> np.ndarray:
    """Return a unit-length copy of ``v`` or raise if it is the zero vector."""
    n = float(np.linalg.norm(v))
    if n < _ZERO_TOL:
        raise ValueError(f"GraspPoint.{name} cannot be the zero vector")
    if abs(n - 1.0) <= _UNIT_TOL:
        return v
    return v / n


@dataclass(frozen=True, slots=True)
class GraspPoint:
    """A single grasp candidate. The module docstring carries the semantics."""

    position: np.ndarray
    approach: np.ndarray
    axis: np.ndarray
    grip_width_mm: float
    score: float
    frame: GraspFrame
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate and coerce the three vectors, normalising approach and axis.
        position = as_vec3(self.position, "position")
        approach = _normalise_unit(as_vec3(self.approach, "approach"), "approach")
        axis = _normalise_unit(as_vec3(self.axis, "axis"), "axis")

        if self.grip_width_mm < 0.0:
            raise ValueError(
                f"GraspPoint.grip_width_mm must be >= 0, got {self.grip_width_mm}",
            )
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"GraspPoint.score must be in [0, 1], got {self.score}",
            )

        # Make the arrays read-only so consumers cannot mutate them.
        position.flags.writeable = False
        approach.flags.writeable = False
        axis.flags.writeable = False

        # A frozen dataclass, so the setattr lock is bypassed to install the validated
        # values.
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "approach", approach)
        object.__setattr__(self, "axis", axis)

    def distance_to(self, other: GraspPoint) -> float:
        """The Euclidean distance in millimetres between two grasp positions."""
        return float(np.linalg.norm(self.position - other.position))

    def pose(self) -> Pose:
        """Where the tool goes for this grasp, in this grasp's own frame: +Z ``approach``,
        +X ``axis``.

        A BASE grasp gives the BASE pose ``Robot.pick`` takes, with ``grip_width_mm`` as
        its width. A CAMERA grasp gives a CAMERA pose, which ``Robot.pick`` refuses with
        nothing commanded until the pose is carried into BASE; the frame is never
        relabelled here. The columns are the ones ``grasp_point_to_pose`` builds for the
        collision check.
        """
        return pose_from_grasp_axes(self.position, approach=self.approach, closing_axis=self.axis,
                                    frame=Frame(self.frame.value))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "position": self.position.tolist(),
            "approach": self.approach.tolist(),
            "axis": self.axis.tolist(),
            "grip_width_mm": self.grip_width_mm,
            "score": self.score,
            "frame": self.frame.value,
            "label": self.label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraspPoint:
        """Deserialise from a dictionary produced by :meth:`to_dict`."""
        try:
            return cls(
                position=np.asarray(d["position"], dtype=np.float64),
                approach=np.asarray(d["approach"], dtype=np.float64),
                axis=np.asarray(d["axis"], dtype=np.float64),
                grip_width_mm=float(d["grip_width_mm"]),
                score=float(d["score"]),
                frame=GraspFrame(d["frame"]),
                label=d.get("label", ""),
                metadata=d.get("metadata", {}),
            )
        except KeyError as exc:
            raise ValueError(f"GraspPoint.from_dict: missing key {exc}") from exc

    def __repr__(self) -> str:
        p = self.position
        return (
            f"GraspPoint(pos=[{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}] "
            f"{self.frame.value}, width={self.grip_width_mm:.1f}mm, "
            f"score={self.score:.2f}, label={self.label!r})"
        )


#: The key under a grasp's ``score_components["support_footprint"]`` (a breakdown's ``components``) whose ``True`` says
#: the grasp grips its part from inside a bore (``generation.inner_grasps``, ``robot.grasping.inner_grasps``): its
#: fingers come down narrow into the bore, open onto its wall, and let the part go by closing.
INNER_GRASP_KEY = "inner"


@dataclass(frozen=True, slots=True)
class GripFromInside:
    """How an inner grasp grips its part, as the stage that found it says (``generation.inner_grasps``).

    Attributes:
        line_in_width_mm (float): How far apart the jaws stand on the way into the bore, between their inner faces: the
            hand comes down at it (lever G's ``open_width_mm``).
        part_width_mm (float): The part's extent about the grasp, the diameter of the circle about the hand's axis that
            holds it in the plane: what the carried part is attached as.
        depth_mm (float): How far under the part's top the fingertips stand at the grasp.
        bore_mm (float): The bore across, wall to wall, along the closing axis in the finger's band.
    """

    line_in_width_mm: float
    part_width_mm: float
    depth_mm: float
    bore_mm: float


def _footprint_components(grasp: object) -> "dict[str, Any] | None":
    """The ``support_footprint`` margins of ``grasp``: a breakdown's ``components``, or a grasp point's
    ``metadata["score_components"]``; ``None`` where it has none."""
    components = getattr(grasp, "components", None)
    if components is None:
        components = (getattr(grasp, "metadata", None) or {}).get("score_components")
    stage = components.get("support_footprint") if isinstance(components, dict) else None
    return stage if isinstance(stage, dict) else None


def grips_from_inside(grasp: object) -> bool:
    """Whether ``grasp`` grips its part from inside a bore: its ``support_footprint`` margins say so (:data:`INNER_GRASP_KEY`
    ``True``), read off a ``GraspPoint``'s ``metadata["score_components"]`` or a breakdown's ``components``.

    Never guessed from a width: a grasp no inner search found is an outer one.

    Args:
        grasp: A ``GraspPoint``, a ``GraspScoreBreakdown`` or anything else.

    Returns:
        bool: ``True`` only where the margins carry the key as ``True``.
    """
    stage = _footprint_components(grasp)
    return stage is not None and stage.get(INNER_GRASP_KEY) is True


def outer_first(grasp: GraspPoint) -> tuple[bool, float]:
    """The key that ranks grasps best first, every grasp from inside a bore after every outer grasp (the owner,
    2026-10-14: "Nur wo außen nichts geht"); a sort keeps grasps of equal keys in the order they came.

    Args:
        grasp: The grasp.

    Returns:
        tuple[bool, float]: Whether it grips from inside, and its score negated.
    """
    return grips_from_inside(grasp), -float(grasp.score)


def grip_from_inside_of(grasp: object) -> "GripFromInside | None":
    """How ``grasp`` grips from inside, where it does (:func:`grips_from_inside`) and its margins carry every number the
    execution reads: the jaws' width on the way in, the part's width, the fingertips' depth and the bore; else ``None``.

    Args:
        grasp: A ``GraspPoint``, a ``GraspScoreBreakdown`` or anything else.

    Returns:
        GripFromInside | None: The numbers, millimetres; ``None`` for an outer grasp, and for an inner one whose margins
            lack a number or hold one that is not a finite non-negative number.
    """
    if not grips_from_inside(grasp):
        return None
    stage = _footprint_components(grasp) or {}
    read: dict[str, float] = {}
    for name in ("line_in_width_mm", "part_width_mm", "depth_mm", "bore_mm"):
        value = stage.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)) \
                or float(value) < 0.0:
            return None
        read[name] = float(value)
    return GripFromInside(**read)


def fingers_held_at(grasp: object) -> float | None:
    """How wide the guard holds the fingers of ``grasp`` from the judgement ahead to the lift, millimetres: from inside
    a bore, where they meet its wall (its margins' ``grip_width_mm``), the widest they stand, wider than they come
    down; planned for jaws that open only as needed (lever G), at that opening (``open_width_mm``); ``None``, fully
    open, for every other grasp.

    Args:
        grasp: A ``GraspPoint``, a ``GraspScoreBreakdown`` or anything else.

    Returns:
        float | None: The opening, millimetres, before the guard's slack; ``None`` for fully open.
    """
    stage = _footprint_components(grasp)
    if stage is None:
        return None
    value = stage.get("grip_width_mm" if stage.get(INNER_GRASP_KEY) is True else "open_width_mm")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)) \
            or float(value) <= 0.0:
        return None
    return float(value)
