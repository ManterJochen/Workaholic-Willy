"""Local collision geometry for the gripper end-effector.

Two interchangeable envelope models share one frame convention, in which Z is the approach axis, the
object contact sits near the origin and the tool body extends back along -Z, and one
:class:`GripperGeometryStrategy` contract:

- :class:`ParallelJawGripperModel` is two fingers and a palm, sized by the commanded grip width.
- :class:`SuctionCupGripperModel` is a cup, a shaft and a wrist mount, and uses no grip width.

Both bound their parts with axis-aligned boxes, so the collision checker treats any gripper the
same way. Add a new end-effector by writing a third model that satisfies the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np

from src.robot.grasping.planning import GraspPose

__all__ = [
    "CollisionBox",
    "FingerStep",
    "GripperGeometryStrategy",
    "HandSteps",
    "HousingStep",
    "ParallelJawGripperModel",
    "SuctionCupGripperModel",
    "hand_steps_of",
    "points_to_grasp_frame",
]


def _vec3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    scalar = float(value)
    lower_ok = scalar >= 0.0 if allow_zero else scalar > 0.0
    if not np.isfinite(scalar) or not lower_ok:
        comparison = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {comparison}")
    return scalar


@dataclass(frozen=True, slots=True)
class CollisionBox:
    """Axis-aligned box in the local grasp frame."""

    label: str
    min_corner_mm: np.ndarray
    max_corner_mm: np.ndarray

    def __post_init__(self) -> None:
        min_corner = _vec3(self.min_corner_mm, "CollisionBox.min_corner_mm")
        max_corner = _vec3(self.max_corner_mm, "CollisionBox.max_corner_mm")
        if not np.all(min_corner < max_corner):
            raise ValueError("CollisionBox min_corner_mm must be strictly below max_corner_mm")
        if not self.label:
            raise ValueError("CollisionBox.label must not be empty")
        min_corner.setflags(write=False)
        max_corner.setflags(write=False)
        object.__setattr__(self, "min_corner_mm", min_corner)
        object.__setattr__(self, "max_corner_mm", max_corner)

    def contains_local_points(self, points_local_mm: np.ndarray, *, margin_mm: float = 0.0) -> np.ndarray:
        """Return a boolean mask for local points inside this box."""
        points = np.asarray(points_local_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_local_mm must be shape (N, 3), got {points.shape}")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_local_mm must contain only finite values")
        margin = _positive(margin_mm, "margin_mm", allow_zero=True)
        lower = self.min_corner_mm - margin
        upper = self.max_corner_mm + margin
        return np.all((points >= lower) & (points <= upper), axis=1)

    def corners_local_mm(self) -> np.ndarray:
        """Return the eight local box corners as a fresh ``(8, 3)`` array."""
        low = self.min_corner_mm
        high = self.max_corner_mm
        return np.array(
            [
                [low[0], low[1], low[2]],
                [low[0], low[1], high[2]],
                [low[0], high[1], low[2]],
                [low[0], high[1], high[2]],
                [high[0], low[1], low[2]],
                [high[0], low[1], high[2]],
                [high[0], high[1], low[2]],
                [high[0], high[1], high[2]],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class FingerStep:
    """A stretch of the finger on +X along the approach, in the grasp frame: from ``z_mm[0]`` to ``z_mm[1]`` (the tip
    positive), ``thickness_mm`` out from its jaw face along the closing axis, across the binormal from ``y_mm[0]`` to
    ``y_mm[1]``. The finger on -X is its half turn about the approach. Measured off the hand's guard bundle
    (:func:`hand_steps_of`)."""

    z_mm: tuple[float, float]
    thickness_mm: float
    y_mm: tuple[float, float]


@dataclass(frozen=True, slots=True)
class HousingStep:
    """A stretch of a hand's housing along the approach, in the grasp frame: from ``z_mm[0]`` to ``z_mm[1]``, across the
    closing axis from ``x_mm[0]`` to ``x_mm[1]`` and the binormal from ``y_mm[0]`` to ``y_mm[1]``."""

    z_mm: tuple[float, float]
    x_mm: tuple[float, float]
    y_mm: tuple[float, float]


@dataclass(frozen=True, slots=True)
class HandSteps:
    """A parallel jaw as its guard bundle holds it, in steps along the approach (:func:`hand_steps_of`)."""

    finger: tuple[FingerStep, ...]
    housing: tuple[HousingStep, ...]


def _stepped(vertices: np.ndarray, faces: np.ndarray, centre: float, z_from: float, z_to: float, slice_mm: float,
             most: int, x_from: float | None = None) -> list[list[float]]:
    """One mesh in at most ``most`` steps along the approach (bundle +Y) from ``z_from`` to ``z_to`` about ``centre``,
    each step ``[z0, z1, x0, x1, y0, y1]``: every ``slice_mm`` of it the box of the triangles crossing it, then the two
    neighbouring steps whose joint box adds the least volume joined until ``most`` are left, a step the box of its
    slices. ``x_from`` holds every step's near side along X there (a finger's jaw face), so only its far side counts."""
    corners = vertices[faces]                                     # (F, 3, 3): bundle x, approach, binormal
    z_lo, z_hi = corners[:, :, 1].min(axis=1) - centre, corners[:, :, 1].max(axis=1) - centre
    x_lo, x_hi = corners[:, :, 0].min(axis=1), corners[:, :, 0].max(axis=1)
    y_lo, y_hi = corners[:, :, 2].min(axis=1), corners[:, :, 2].max(axis=1)
    edges = np.arange(z_from, z_to + 1e-9, slice_mm)
    edges = np.append(edges[edges < z_to - 1e-9], z_to)
    steps: list[list[float]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        crossing = (z_hi >= low) & (z_lo <= high)
        if crossing.any():
            steps.append([low, high, float(x_lo[crossing].min()) if x_from is None else x_from,
                          float(x_hi[crossing].max()), float(y_lo[crossing].min()), float(y_hi[crossing].max())])

    def volume(step: list[float]) -> float:
        return (step[1] - step[0]) * (step[3] - step[2]) * (step[5] - step[4])

    def joined(a: list[float], b: list[float]) -> list[float]:
        return [a[0], b[1], min(a[2], b[2]), max(a[3], b[3]), min(a[4], b[4]), max(a[5], b[5])]

    while len(steps) > most:
        costs = [volume(joined(a, b)) - volume(a) - volume(b) for a, b in zip(steps[:-1], steps[1:])]
        n = int(np.argmin(costs))
        steps[n:n + 2] = [joined(steps[n], steps[n + 1])]
    return steps


def hand_steps_of(bundle_path: Any, *, aperture_mm: float, finger_ahead_mm: float, finger_length_mm: float,
                  slice_mm: float = 1.0, finger_steps: int = 3, housing_steps: int = 4) -> HandSteps:
    """A parallel jaw's fingers and housing in steps along the approach, as its guard bundle holds them.

    The bundle's frame is ``scripts/grippers/measure_jaw_from_bundle.py``'s: approach +Y, closing X, binormal Z, the
    fingers open to ``aperture_mm``; the grasp centre stands ``finger_ahead_mm`` behind the fingertip. The finger on +X
    is read from ``finger_length_mm`` behind the centre, where it leaves the housing, to its tip, its thickness out from
    the jaw face; the housing over the whole of its mesh behind that. Every ``slice_mm`` is the box of the triangles
    crossing it, so a slice holds every point of the surface in it, not only the vertices; the slices join into
    ``finger_steps`` steps of the finger and ``housing_steps`` of the housing, the neighbours whose joint box adds the
    least volume first, and a step is the box of its slices: every box the calculator holds of the hand holds the mesh
    the guard judges in it.

    Args:
        bundle_path: The hand's ``{hand}_hand_meshes.npz`` (``safety.planning.environment.hand_mesh_bundle``).
        aperture_mm: How far apart the jaw faces stand in the bundle.
        finger_ahead_mm: How far the fingertip stands past the grasp centre.
        finger_length_mm: How far behind the grasp centre the finger leaves the housing.
        slice_mm: How thick a slice is.
        finger_steps: How many steps the finger is held in.
        housing_steps: How many steps the housing is held in.

    Returns:
        HandSteps: The finger's steps and the housing's, nearest the housing's back first.
    """
    with np.load(bundle_path) as data:
        finger_v = np.asarray(data["rfinger__v"], dtype=np.float64)
        finger_f = np.asarray(data["rfinger__f"], dtype=np.int64)
        housing_v = np.asarray(data["gripper__v"], dtype=np.float64)
        housing_f = np.asarray(data["gripper__f"], dtype=np.int64)
    centre = float(finger_v[:, 1].max()) - float(finger_ahead_mm)
    face = 0.5 * float(aperture_mm)
    finger = _stepped(finger_v, finger_f, centre, -float(finger_length_mm), float(finger_ahead_mm), slice_mm,
                      finger_steps, x_from=face)
    housing = _stepped(housing_v, housing_f, centre, float(housing_v[:, 1].min()) - centre, -float(finger_length_mm),
                       slice_mm, housing_steps)
    return HandSteps(
        finger=tuple(FingerStep(z_mm=(z0, z1), thickness_mm=x1 - face, y_mm=(y0, y1))
                     for z0, z1, _x0, x1, y0, y1 in finger),
        housing=tuple(HousingStep(z_mm=(z0, z1), x_mm=(x0, x1), y_mm=(y0, y1))
                      for z0, z1, x0, x1, y0, y1 in housing))


@dataclass(frozen=True, slots=True)
class ParallelJawGripperModel:
    """Conservative box model in the local grasp frame, where X closes between the contacts, Y is the binormal and Z is the approach."""

    # Measured off the 2F-85 collision shapes, worst case over the aperture range: the finger swings
    # as the jaw moves, so its reach toward the table is largest at a narrow aperture. Every
    # millimetre understated here is finger hidden from the table-clearance check and pointing
    # straight at the support surface. See the schema for the sweep.
    finger_length_mm: float = 39.98
    finger_thickness_mm: float = 31.35
    finger_width_mm: float = 27.0
    finger_pad_overlap_mm: float = 2.0
    fingertip_depth_mm: float = 28.72
    #: The contact patch along the approach, 38.0 mm measured. It is modelled as its own box so a
    #: check can ask about the part that touches rather than about the housing: it runs 14.4 mm behind
    #: the grasp centre and 23.6 mm in front, so a grasp is not symmetric about its own contact. It
    #: lies inside the finger box, so adding it changes no collision outcome, only what a caller can
    #: ask.
    pad_length_mm: float = 38.0
    #: How far the patch reaches in front of the grasp centre, measured 23.61 mm. It is not
    #: ``fingertip_depth_mm``: the finger housing extends 5.11 mm past the rubber, so deriving the pad
    #: front from the fingertip places the whole patch 5.11 mm too far forward. That changes no
    #: collision outcome, since the pad lies inside the finger box either way, but it is the number a
    #: contact question is answered with, so it is measured rather than derived.
    pad_ahead_mm: float = 23.61
    palm_depth_mm: float = 35.0
    palm_width_mm: float = 70.0
    outer_margin_mm: float = 0.0
    #: The housing along the closing axis, millimetres, where the registry measured it (``palm_thickness_mm``): the palm
    #: box is at least this wide across the closing axis. ``None`` keeps it as wide as the open fingers' outer faces, as
    #: it always was. The Hand-E's housing is 75.0 mm across, its open fingers 71.6, and the guard held it 1.7 mm past the
    #: box on either side (the grasp bench, 2026-10-06).
    palm_thickness_mm: float | None = None
    #: The fingers and the housing in steps along the approach where the hand's guard bundle measured them
    #: (:func:`hand_steps_of`), in this model's frame; ``None`` holds each as one box, as it always was. A Hand-E's tip,
    #: where its pad sits, is 6.5 mm thick and 21 mm across, and its housing 62 mm across the binormal where it meets
    #: the fingers: the boxes stood 4.8 and 6.5 mm past them.
    hand_steps: "HandSteps | None" = None

    def __post_init__(self) -> None:
        if self.palm_thickness_mm is not None:
            object.__setattr__(self, "palm_thickness_mm", _positive(self.palm_thickness_mm, "palm_thickness_mm"))
        for field_name in (
            "finger_length_mm",
            "finger_thickness_mm",
            "finger_width_mm",
            "fingertip_depth_mm",
            "pad_length_mm",
            "pad_ahead_mm",
            "palm_depth_mm",
            "palm_width_mm",
        ):
            object.__setattr__(self, field_name, _positive(getattr(self, field_name), field_name))
        object.__setattr__(
            self,
            "finger_pad_overlap_mm",
            _positive(self.finger_pad_overlap_mm, "finger_pad_overlap_mm", allow_zero=True),
        )
        object.__setattr__(
            self,
            "outer_margin_mm",
            _positive(self.outer_margin_mm, "outer_margin_mm", allow_zero=True),
        )

    def collision_boxes(self, grip_width_mm: float) -> tuple[CollisionBox, ...]:
        """Return local boxes for the current commanded grip width."""
        width = _positive(grip_width_mm, "grip_width_mm", allow_zero=True)
        half_gap = 0.5 * width
        y_finger = 0.5 * self.finger_width_mm
        z_min = -self.finger_length_mm
        z_max = self.fingertip_depth_mm
        margin = self.outer_margin_mm

        steps = (self.hand_steps.finger if self.hand_steps is not None and self.hand_steps.finger
                 else (FingerStep(z_mm=(z_min, z_max), thickness_mm=self.finger_thickness_mm, y_mm=(-y_finger, y_finger)),))
        fingers = []
        overlap = self.finger_pad_overlap_mm
        for step in steps:
            (z0, z1), (y0, y1), thick = step.z_mm, step.y_mm, step.thickness_mm
            fingers.append(CollisionBox(
                "finger_negative_x",
                cast(np.ndarray, [-half_gap - thick - margin, -y1 - margin, z0 - margin]),
                cast(np.ndarray, [-half_gap + overlap + margin, -y0 + margin, z1 + margin]),
            ))
            fingers.append(CollisionBox(
                "finger_positive_x",
                cast(np.ndarray, [half_gap - overlap - margin, y0 - margin, z0 - margin]),
                cast(np.ndarray, [half_gap + thick + margin, y1 + margin, z1 + margin]),
            ))

        if self.hand_steps is not None and self.hand_steps.housing:
            # The housing as measured, every step at least as wide as the open fingers' outer faces, which the fingers'
            # carriages reach inside it.
            outer = half_gap + self.finger_thickness_mm
            palms = tuple(CollisionBox(
                "palm",
                cast(np.ndarray, [min(step.x_mm[0], -outer) - margin, step.y_mm[0] - margin, step.z_mm[0] - margin]),
                cast(np.ndarray, [max(step.x_mm[1], outer) + margin, step.y_mm[1] + margin, step.z_mm[1] + margin]),
            ) for step in self.hand_steps.housing)
            return (*fingers, *palms)
        outer_half_x = max(half_gap + self.finger_thickness_mm, 0.5 * float(self.palm_thickness_mm or 0.0))
        palm_half_y = 0.5 * self.palm_width_mm
        palm = CollisionBox(
            "palm",
            cast(
                np.ndarray,
                [-outer_half_x - margin, -palm_half_y - margin, -self.finger_length_mm - self.palm_depth_mm - margin],
            ),
            cast(np.ndarray, [outer_half_x + margin, palm_half_y + margin, -self.finger_length_mm + margin]),
        )
        return (*fingers, palm)

    def pad_boxes(self, grip_width_mm: float) -> tuple[CollisionBox, ...]:
        """The contact patches alone: where the gripper touches, not where its housing is.

        Deliberately not part of :meth:`collision_boxes`. Each pad box lies entirely inside its finger
        box, so adding them to the envelope cannot change a single collision outcome while costing 5
        box tests instead of 3 on every candidate.

        What the patch is worth is the questions the envelope cannot answer. Whether a grasp closes on
        two opposing faces, and whether the part that touches clears the table, are questions about the
        38 mm patch, which runs 14.4 mm behind the grasp centre and 23.6 mm in front of it, not about
        the 68 mm of housing around it. The independent reference measured what getting the patch wrong
        costs: modelling the jaw as a line through the anchor rejected 147 candidates for an anchor
        outside the object whose median miss was 0.3 mm, and modelling the patch at twice its real size
        halved precision.
        """
        width = _positive(grip_width_mm, "grip_width_mm", allow_zero=True)
        half_gap = 0.5 * width
        y_finger = 0.5 * self.finger_width_mm
        margin = self.outer_margin_mm
        front = self.pad_ahead_mm
        back = self.pad_ahead_mm - self.pad_length_mm
        return (
            CollisionBox(
                "pad_negative_x",
                cast(np.ndarray, [-half_gap - margin, -y_finger - margin, back - margin]),
                cast(np.ndarray, [-half_gap + self.finger_pad_overlap_mm + margin, y_finger + margin, front + margin]),
            ),
            CollisionBox(
                "pad_positive_x",
                cast(np.ndarray, [half_gap - self.finger_pad_overlap_mm - margin, -y_finger - margin, back - margin]),
                cast(np.ndarray, [half_gap + margin, y_finger + margin, front + margin]),
            ),
        )

    def local_corners_mm(self, grip_width_mm: float) -> np.ndarray:
        """Return all local collision-box corners as a fresh ``(24, 3)`` array."""
        return np.vstack([box.corners_local_mm() for box in self.collision_boxes(grip_width_mm)])


@dataclass(frozen=True, slots=True)
class SuctionCupGripperModel:
    """Conservative box model of a suction end-effector in the local grasp frame.

    The frame is the jaw's, with Z the approach axis: the cup contacts the object near the origin and
    the shaft and wrist mount extend back along -Z. There are no jaws, so the commanded grip width is
    ignored. The round cup, shaft and mount are each bounded by a square-section box, which
    circumscribes the circle, to stay conservative and to match the jaw's box representation. The
    defaults describe a 30 mm diameter cup on a slim shaft with a wrist coupler.
    """

    cup_radius_mm: float = 15.0
    cup_height_mm: float = 25.0
    contact_tip_depth_mm: float = 2.0
    shaft_radius_mm: float = 10.0
    shaft_length_mm: float = 40.0
    mount_radius_mm: float = 30.0
    mount_depth_mm: float = 20.0
    outer_margin_mm: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "cup_radius_mm",
            "cup_height_mm",
            "shaft_radius_mm",
            "shaft_length_mm",
            "mount_radius_mm",
            "mount_depth_mm",
        ):
            object.__setattr__(self, field_name, _positive(getattr(self, field_name), field_name))
        object.__setattr__(
            self, "contact_tip_depth_mm", _positive(self.contact_tip_depth_mm, "contact_tip_depth_mm", allow_zero=True)
        )
        object.__setattr__(self, "outer_margin_mm", _positive(self.outer_margin_mm, "outer_margin_mm", allow_zero=True))

    def collision_boxes(self, grip_width_mm: float) -> tuple[CollisionBox, ...]:
        """Return the local cup, shaft and mount boxes; the grip width is ignored, since a suction cup has no jaws."""
        margin = self.outer_margin_mm
        cup_back = -self.cup_height_mm
        shaft_back = cup_back - self.shaft_length_mm
        mount_back = shaft_back - self.mount_depth_mm

        def _cylinder_box(label: str, radius: float, z_low: float, z_high: float) -> CollisionBox:
            return CollisionBox(
                label,
                cast(np.ndarray, [-radius - margin, -radius - margin, z_low - margin]),
                cast(np.ndarray, [radius + margin, radius + margin, z_high + margin]),
            )

        return (
            _cylinder_box("suction_cup", self.cup_radius_mm, cup_back, self.contact_tip_depth_mm),
            _cylinder_box("suction_shaft", self.shaft_radius_mm, shaft_back, cup_back),
            _cylinder_box("suction_mount", self.mount_radius_mm, mount_back, shaft_back),
        )

    def local_corners_mm(self, grip_width_mm: float) -> np.ndarray:
        """Return all local collision-box corners as a fresh ``(24, 3)`` array."""
        return np.vstack([box.corners_local_mm() for box in self.collision_boxes(grip_width_mm)])


def points_to_grasp_frame(points_mm: np.ndarray, pose: GraspPose) -> np.ndarray:
    """Transform external-frame points into a ``GraspPose`` local frame."""
    if not isinstance(pose, GraspPose):
        raise TypeError("pose must be a GraspPose")
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_mm must be shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_mm must contain only finite values")
    return (points - pose.position_mm) @ pose.rotation_matrix


@runtime_checkable
class GripperGeometryStrategy(Protocol):
    """Collision-geometry contract letting callers swap the gripper model without changing the calculator."""

    def collision_boxes(
        self, grip_width_mm: float
    ) -> tuple[CollisionBox, ...]: ...

    def local_corners_mm(self, grip_width_mm: float) -> np.ndarray: ...
