"""Multi-finger grasp planning, for N contacts.

A separate planning surface for grippers with three or more fingers, or any bespoke
multi-contact topology, reached through ``GraspCalculator.plan_multifinger()``. It runs
independently of the two-finger parallel-jaw pipeline, which stays the default.

Scope
-----

* Both reference planners reason in the style of 2D Coulomb friction and assume a fixed
  approach axis, usually camera ``+Z``. The radial planner produces geometrically feasible
  N-contact candidates inside the kinematic envelope, but it does not prove dexterous force
  closure for ``N >= 3``. A force-closure planner can be plugged in through the same Protocol.
* Implement :class:`MultiContactGraspPlanner` and pass it to ``plan_multifinger(planner=...)``
  to use bespoke kinematics without touching core code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from src.robot.grasping.contacts.contact_point import as_unit_normal
from src.robot.grasping.geometry import as_vec3

if TYPE_CHECKING:
    from src.robot.grasping.geometry import CameraIntrinsics

__all__ = [
    "FingerKinematicSpec",
    "GripperKind",
    "MultiContactGrasp",
    "MultiContactGraspPlanner",
    "MultiContactPlanRequest",
    "ParallelJawContactPlanner",
    "RadialMultiFingerPlanner",
]


class GripperKind(StrEnum):
    """Stable telemetry-safe string enum for gripper topology.

    ``VACUUM`` is carried for telemetry completeness only. A suction grasp is a single-contact
    seal and lives in its own :class:`~src.robot.grasping.suction.synthesis.SuctionGrasp`,
    because :class:`MultiContactGrasp` requires ``finger_count >= 2`` and cannot represent it.
    """

    PARALLEL_JAW = "parallel_jaw"
    THREE_FINGER = "three_finger"
    FOUR_FINGER = "four_finger"
    VACUUM = "vacuum"


@dataclass(frozen=True, slots=True)
class FingerKinematicSpec:
    """Mechanical envelope shared across all multi-finger planners.

    Fields
    ------
    finger_count
        Number of contact fingers. Must be ``>= 2``.
    min_radius_mm, max_radius_mm
        Allowed signed radial distance from the palm centerline to the
        contact point (projected onto the plane perpendicular to the
        approach axis). ``min`` must be ``>= 0`` and strictly less than
        ``max``.
    min_angular_separation_deg
        Minimum angle between adjacent finger directions, measured in
        the plane perpendicular to the approach axis. Equal-spaced
        ``N`` fingers have separation ``360 / N`` degrees; this field
        lets the operator reject configurations that pack fingers
        too tightly to avoid collisions.
    palm_offset_mm
        Standoff distance from the palm plane to the deepest contact,
        used downstream to derive a pre-grasp pose. Must be ``>= 0``.
    finger_reach_mm
        Maximum axial finger reach along the approach axis. Used as a
        coarse depth-spread tolerance: contacts whose depth differs
        from the palm depth by more than ``finger_reach_mm`` are
        rejected.
    """

    finger_count: int
    min_radius_mm: float
    max_radius_mm: float
    min_angular_separation_deg: float = 30.0
    palm_offset_mm: float = 20.0
    finger_reach_mm: float = 80.0

    def __post_init__(self) -> None:
        if int(self.finger_count) < 2:
            raise ValueError(
                f"FingerKinematicSpec.finger_count must be >= 2; got {self.finger_count}"
            )
        for name in ("min_radius_mm", "max_radius_mm", "palm_offset_mm", "finger_reach_mm"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"FingerKinematicSpec.{name} must be finite and >= 0; got {value}"
                )
        if float(self.min_radius_mm) >= float(self.max_radius_mm):
            raise ValueError(
                "FingerKinematicSpec.min_radius_mm must be < max_radius_mm"
            )
        ang = float(self.min_angular_separation_deg)
        if not np.isfinite(ang) or ang <= 0.0 or ang > 180.0:
            raise ValueError(
                "FingerKinematicSpec.min_angular_separation_deg must be in (0, 180]"
            )
        equal_spacing_deg = 360.0 / float(self.finger_count)
        if equal_spacing_deg + 1e-9 < ang:
            raise ValueError(
                "FingerKinematicSpec inconsistent: equal-spaced "
                f"{self.finger_count} fingers give {equal_spacing_deg:.2f} deg "
                f"separation, which is below min_angular_separation_deg "
                f"= {ang:.2f}"
            )


@dataclass(frozen=True, slots=True)
class MultiContactGrasp:
    """Immutable N-contact grasp candidate (camera-frame).

    The grasp is described by:

    * a palm anchor point (``palm_center_mm``) along the approach axis,
    * the approach axis itself (unit vector in camera frame),
    * ``N`` contact points and outward surface normals (one per finger),
    * a normalised geometric score in ``[0, 1]``,
    * arbitrary metadata for downstream telemetry.

    Validation is strict: contact and normal tuples must agree in
    length and the score must lie in ``[0, 1]``. Each ``(contact point,
    normal)`` is validated by the shared
    :class:`~src.robot.grasping.contacts.ContactPoint` contract
    (finite 3D point; normal within 1% of unit length, stored normalised),
    so a multi-finger contact obeys exactly the same invariants a
    :class:`~src.robot.grasping.contacts.ContactPair` contact does.
    """

    palm_center_mm: np.ndarray
    approach_axis: np.ndarray
    contact_points_mm: tuple[np.ndarray, ...]
    contact_normals: tuple[np.ndarray, ...]
    finger_count: int
    gripper_kind: GripperKind
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        palm = as_vec3(self.palm_center_mm, "palm_center_mm")
        approach = as_vec3(self.approach_axis, "approach_axis")
        norm = float(np.linalg.norm(approach))
        if norm < 1e-9:
            raise ValueError("approach_axis must be non-zero")
        approach = approach / norm

        finger_count = int(self.finger_count)
        if finger_count < 2:
            raise ValueError("MultiContactGrasp.finger_count must be >= 2")
        if len(self.contact_points_mm) != finger_count:
            raise ValueError(
                f"MultiContactGrasp expects {finger_count} contact points; "
                f"got {len(self.contact_points_mm)}"
            )
        if len(self.contact_normals) != finger_count:
            raise ValueError(
                f"MultiContactGrasp expects {finger_count} contact normals; "
                f"got {len(self.contact_normals)}"
            )

        points: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        for i, (pt, nv) in enumerate(zip(self.contact_points_mm, self.contact_normals)):
            # Each (point, outward unit normal) is validated by the shared ContactPoint contract,
            # so a multi-finger contact obeys exactly the invariants a ContactPair contact does.
            points.append(as_vec3(pt, f"contact_points_mm[{i}]"))
            normals.append(as_unit_normal(nv, f"contact_normals[{i}]"))

        score = float(self.score)
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"MultiContactGrasp.score must be in [0, 1]; got {score}")

        gripper_kind = (
            self.gripper_kind
            if isinstance(self.gripper_kind, GripperKind)
            else GripperKind(str(self.gripper_kind))
        )

        # Freeze the stored copies so a frozen grasp is immutable in content and not only in
        # attribute binding, matching the ContactPoint and GraspPose discipline. as_vec3 and
        # as_unit_normal already copied the caller's arrays, so this freezes nothing the caller
        # still holds.
        palm.setflags(write=False)
        approach.setflags(write=False)
        for _frozen in (*points, *normals):
            _frozen.setflags(write=False)
        object.__setattr__(self, "palm_center_mm", palm)
        object.__setattr__(self, "approach_axis", approach)
        object.__setattr__(self, "contact_points_mm", tuple(points))
        object.__setattr__(self, "contact_normals", tuple(normals))
        object.__setattr__(self, "finger_count", finger_count)
        object.__setattr__(self, "gripper_kind", gripper_kind)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of the candidate."""
        return {
            "palm_center_mm": self.palm_center_mm.tolist(),
            "approach_axis": self.approach_axis.tolist(),
            "contact_points_mm": [pt.tolist() for pt in self.contact_points_mm],
            "contact_normals": [nv.tolist() for nv in self.contact_normals],
            "finger_count": int(self.finger_count),
            "gripper_kind": self.gripper_kind.value,
            "score": float(self.score),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MultiContactPlanRequest:
    """Input bundle for any :class:`MultiContactGraspPlanner`."""

    mask: np.ndarray
    depth_map: np.ndarray
    intrinsics: CameraIntrinsics
    kinematics: FingerKinematicSpec
    approach_axis_cam: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    scale_to_mm: float = 1.0
    max_results: int = 5

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError("mask must be 2D")
        if self.depth_map.shape != self.mask.shape:
            raise ValueError(
                "depth_map and mask must share the same shape; "
                f"got {self.depth_map.shape} vs {self.mask.shape}"
            )
        approach = as_vec3(self.approach_axis_cam, "approach_axis_cam")
        norm = float(np.linalg.norm(approach))
        if norm < 1e-9:
            raise ValueError("approach_axis_cam cannot be the zero vector")
        object.__setattr__(self, "approach_axis_cam", approach / norm)
        if not np.isfinite(self.scale_to_mm) or self.scale_to_mm <= 0.0:
            raise ValueError("scale_to_mm must be finite and > 0")
        if int(self.max_results) < 1:
            raise ValueError("max_results must be >= 1")


@runtime_checkable
class MultiContactGraspPlanner(Protocol):
    """Protocol implemented by every multi-finger planner.

    Implementations must:

    * Expose a :attr:`gripper_kind` attribute of type :class:`GripperKind`.
    * Implement :meth:`plan` returning a list of :class:`MultiContactGrasp`
      ordered by descending score.
    * Be idempotent: a second ``plan(request)`` call on the same inputs
      must return an equivalent result (modulo identity).

    Extending to production planners
    --------------------------------

    To add a model-based planner, a learned grasp proposal network for
    instance, implement this Protocol, mark ``gripper_kind`` on the
    instance, and feed the planner to
    ``GraspCalculator.plan_multifinger(planner=...)``. No core code
    needs to change.
    """

    gripper_kind: GripperKind

    def plan(self, request: MultiContactPlanRequest) -> list[MultiContactGrasp]:
        ...


def _back_project(
    pixel_xy: tuple[float, float],
    depth_mm: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Back-project a single ``(px, py)`` pixel and its depth into the 3D camera frame."""
    px, py = float(pixel_xy[0]), float(pixel_xy[1])
    z = float(depth_mm)
    x = (px - intrinsics.cx) * z / intrinsics.fx
    y = (py - intrinsics.cy) * z / intrinsics.fy
    return np.array([x, y, z], dtype=np.float64)


def _trace_to_boundary(
    mask: np.ndarray,
    start_xy: tuple[float, float],
    direction_xy: tuple[float, float],
    *,
    step_px: float = 1.0,
    max_steps: int = 2048,
) -> tuple[int, int] | None:
    """Walk along a ray in pixel space until the mask edge is crossed.

    Returns the last pixel that is still inside the mask (the boundary
    contact). ``None`` is returned when the ray exits the image before
    reaching a mask boundary, or when the starting pixel is outside
    the mask.
    """
    H, W = mask.shape
    x, y = float(start_xy[0]), float(start_xy[1])
    dx, dy = float(direction_xy[0]), float(direction_xy[1])
    if not 0 <= int(round(x)) < W or not 0 <= int(round(y)) < H:
        return None
    if not bool(mask[int(round(y)), int(round(x))]):
        return None
    last_inside: tuple[int, int] | None = (int(round(x)), int(round(y)))
    for _ in range(max_steps):
        x += dx * step_px
        y += dy * step_px
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < W and 0 <= iy < H):
            return last_inside
        if bool(mask[iy, ix]):
            last_inside = (ix, iy)
        else:
            return last_inside
    return last_inside


def _sample_depth_mm(
    depth_map: np.ndarray,
    pixel_xy: tuple[int, int],
    scale_to_mm: float,
) -> float | None:
    """Return the depth in millimetres at the given pixel, or ``None``."""
    px, py = int(pixel_xy[0]), int(pixel_xy[1])
    H, W = depth_map.shape
    if not (0 <= px < W and 0 <= py < H):
        return None
    raw = float(depth_map[py, px])
    if not np.isfinite(raw) or raw <= 0.0:
        return None
    return raw * float(scale_to_mm)


def _trace_radial_fingers(
    mask_bool: np.ndarray,
    depth_map: np.ndarray,
    centroid_xy: tuple[float, float],
    palm_center: np.ndarray,
    palm_depth_mm: float,
    offset: float,
    request: MultiContactPlanRequest,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]] | None:
    """Trace one N-finger candidate at a given rotational ``offset``.

    Casts a ray outward from the mask centroid for each of the ``N`` fingers, back-projects the
    boundary hit, and keeps the contact only if its depth spread and radial distance sit inside
    the kinematic envelope. Returns ``(contacts, outward_normals, radii)``, or ``None`` if any
    finger leaves the image, samples invalid depth, or falls outside the envelope.
    """
    spec = request.kinematics
    n_fingers = int(spec.finger_count)
    cx, cy = centroid_xy
    contacts_3d: list[np.ndarray] = []
    normals_3d: list[np.ndarray] = []
    radii: list[float] = []
    for i in range(n_fingers):
        theta = offset + 2.0 * np.pi * i / float(n_fingers)
        boundary = _trace_to_boundary(
            mask_bool, (cx, cy), (float(np.cos(theta)), float(np.sin(theta)))
        )
        if boundary is None:
            return None
        contact_depth_mm = _sample_depth_mm(depth_map, boundary, request.scale_to_mm)
        if contact_depth_mm is None:
            return None
        if abs(contact_depth_mm - palm_depth_mm) > float(spec.finger_reach_mm):
            return None
        contact_3d = _back_project(boundary, contact_depth_mm, request.intrinsics)
        # Radial distance from the palm axis (the line through palm_center along the approach axis).
        delta = contact_3d - palm_center
        axial = float(np.dot(delta, request.approach_axis_cam))
        radial_vec = delta - request.approach_axis_cam * axial
        radius = float(np.linalg.norm(radial_vec))
        if radius < float(spec.min_radius_mm) or radius > float(spec.max_radius_mm):
            return None
        if radius < 1e-6:  # no well-defined outward normal at the palm axis
            return None
        contacts_3d.append(contact_3d)
        normals_3d.append(radial_vec / radius)  # outward: from palm axis toward the contact
        radii.append(radius)
    return contacts_3d, normals_3d, radii


def _radial_uniformity_score(radii: np.ndarray, spec: FingerKinematicSpec) -> float:
    """Blend radial uniformity and envelope-centred coverage into a ``[0, 1]`` score.

    Uniformity rewards fingers packed at a single radius (low spread); coverage rewards a mean
    radius near the middle of the allowed envelope. Both radial extremes are penalised.
    """
    mean_r = float(radii.mean())
    std_r = float(radii.std())
    uniformity = float(np.clip(1.0 - (std_r / max(mean_r, 1e-6)), 0.0, 1.0))
    envelope_mid = 0.5 * (spec.min_radius_mm + spec.max_radius_mm)
    envelope_half = max(0.5 * (spec.max_radius_mm - spec.min_radius_mm), 1e-6)
    coverage = float(np.clip(1.0 - abs(mean_r - envelope_mid) / envelope_half, 0.0, 1.0))
    return float(np.clip(0.5 * (uniformity + coverage), 0.0, 1.0))


@dataclass
class RadialMultiFingerPlanner:
    """N-finger planner that places fingers radially around the mask centroid.

    Algorithm
    ---------

    1. Compute the mask centroid in pixel space and back-project it
       using the median in-mask depth to obtain the palm anchor.
    2. For each candidate rotational offset (``rotation_samples``
       evenly spaced angles in ``[0, 2pi / N)``) and for each of the
       ``N`` fingers, cast a ray outward from the centroid in pixel
       space and find the mask boundary contact.
    3. Back-project each boundary pixel to 3D, compute its radial
       distance from the palm axis, and reject candidates whose radii
       fall outside the kinematic envelope or whose depth differs from
       the palm depth by more than ``kinematics.finger_reach_mm``.
    4. Score the surviving candidates by radial uniformity, where a
       low variance scores high. Sort and return the top
       ``request.max_results``.

    The planner is deterministic for fixed inputs.
    """

    rotation_samples: int = 6
    gripper_kind: GripperKind = GripperKind.THREE_FINGER  # overridden at runtime

    def plan(self, request: MultiContactPlanRequest) -> list[MultiContactGrasp]:
        spec = request.kinematics
        n_fingers = int(spec.finger_count)
        # Snap gripper_kind to the matching enum for telemetry; operators can override it on the
        # instance for bespoke topologies.
        kind = _gripper_kind_for_count(n_fingers, default=self.gripper_kind)

        mask_bool = np.asarray(request.mask).astype(bool)
        depth_map = np.asarray(request.depth_map, dtype=np.float64)
        if not mask_bool.any():
            return []

        ys, xs = np.where(mask_bool)
        centroid_xy = (float(xs.mean()), float(ys.mean()))

        # The median in-mask depth is a robust palm-anchor depth.
        depths_inside = depth_map[mask_bool] * float(request.scale_to_mm)
        valid_depths = depths_inside[np.isfinite(depths_inside) & (depths_inside > 0.0)]
        if valid_depths.size == 0:
            return []
        palm_depth_mm = float(np.median(valid_depths))
        palm_contact_3d = _back_project(centroid_xy, palm_depth_mm, request.intrinsics)
        palm_center = palm_contact_3d - request.approach_axis_cam * float(spec.palm_offset_mm)

        candidates: list[MultiContactGrasp] = []
        rotation_samples = max(1, int(self.rotation_samples))
        for r in range(rotation_samples):
            offset = (2.0 * np.pi / float(n_fingers)) * (r / rotation_samples)
            traced = _trace_radial_fingers(
                mask_bool, depth_map, centroid_xy, palm_center, palm_depth_mm, offset, request
            )
            if traced is None:
                continue
            contacts_3d, normals_3d, radii = traced
            radii_arr = np.asarray(radii, dtype=np.float64)
            candidates.append(
                MultiContactGrasp(
                    palm_center_mm=palm_center,
                    approach_axis=request.approach_axis_cam,
                    contact_points_mm=tuple(contacts_3d),
                    contact_normals=tuple(normals_3d),
                    finger_count=n_fingers,
                    gripper_kind=kind,
                    score=_radial_uniformity_score(radii_arr, spec),
                    metadata={
                        "planner": "radial_multifinger",
                        "rotation_offset_rad": float(offset),
                        "mean_radius_mm": float(radii_arr.mean()),
                        "radius_std_mm": float(radii_arr.std()),
                        "radii_mm": radii_arr.tolist(),
                        "palm_depth_mm": palm_depth_mm,
                        "confidence_kind": "heuristic",
                    },
                )
            )

        candidates.sort(key=lambda g: g.score, reverse=True)
        return candidates[: int(request.max_results)]


@dataclass
class ParallelJawContactPlanner:
    """2-finger planner that delegates to the antipodal pipeline.

    Runs the standard
    :func:`src.robot.grasping.contacts.find_antipodal_pairs`
    flow on a freshly back-projected masked cloud and converts each
    resulting antipodal pair into a 2-contact :class:`MultiContactGrasp`,
    so a parallel-jaw grasp is expressed in the same multi-contact
    representation as the N-finger planners.
    """

    normal_radius_mm: float = 18.0
    normal_opposition_threshold: float = 0.75
    axis_alignment_threshold: float = 0.45
    voxel_size_mm: float | None = 8.0
    gripper_kind: GripperKind = GripperKind.PARALLEL_JAW

    def plan(self, request: MultiContactPlanRequest) -> list[MultiContactGrasp]:
        from src.robot.grasping.contacts import find_antipodal_pairs
        from src.robot.grasping.geometry import (
            estimate_surface_normals,
            masked_point_cloud,
        )

        spec = request.kinematics
        if int(spec.finger_count) != 2:
            raise ValueError(
                "ParallelJawContactPlanner requires finger_count == 2; "
                f"got {spec.finger_count}"
            )

        cloud = masked_point_cloud(
            request.mask,
            request.depth_map,
            request.intrinsics,
            voxel_size_mm=self.voxel_size_mm,
        )
        if cloud.points_mm.shape[0] < 4:
            return []

        normals = estimate_surface_normals(
            cloud.points_mm.astype(np.float64),
            radius_mm=float(self.normal_radius_mm),
            min_neighbors=6,
            max_neighbors=64,
        )
        if normals.valid_count < 2:
            return []

        pairs = find_antipodal_pairs(
            cloud.points_mm.astype(np.float64),
            normals,
            min_width_mm=float(spec.min_radius_mm * 2.0),
            max_width_mm=float(spec.max_radius_mm * 2.0),
            normal_opposition_threshold=float(self.normal_opposition_threshold),
            axis_alignment_threshold=float(self.axis_alignment_threshold),
            max_pairs=int(request.max_results),
        )
        results: list[MultiContactGrasp] = []
        for pair in pairs:
            palm = 0.5 * (np.asarray(pair.point_a) + np.asarray(pair.point_b))
            palm_center = palm - request.approach_axis_cam * float(
                spec.palm_offset_mm
            )
            results.append(
                MultiContactGrasp(
                    palm_center_mm=palm_center,
                    approach_axis=request.approach_axis_cam,
                    contact_points_mm=(
                        np.asarray(pair.point_a, dtype=np.float64),
                        np.asarray(pair.point_b, dtype=np.float64),
                    ),
                    contact_normals=(
                        np.asarray(pair.normal_a, dtype=np.float64),
                        np.asarray(pair.normal_b, dtype=np.float64),
                    ),
                    finger_count=2,
                    gripper_kind=self.gripper_kind,
                    score=float(pair.antipodal_score),
                    metadata={
                        "planner": "parallel_jaw_antipodal",
                        "antipodal_score": float(pair.antipodal_score),
                        "axis_alignment": float(pair.axis_alignment),
                        "normal_opposition": float(pair.normal_opposition),
                        "width_mm": float(pair.distance_mm),
                        "confidence_kind": "heuristic",
                    },
                )
            )
        return results


def _gripper_kind_for_count(
    finger_count: int, *, default: GripperKind
) -> GripperKind:
    """Map a finger count to the matching :class:`GripperKind`."""
    if finger_count == 2:
        return GripperKind.PARALLEL_JAW
    if finger_count == 3:
        return GripperKind.THREE_FINGER
    if finger_count == 4:
        return GripperKind.FOUR_FINGER
    return default
