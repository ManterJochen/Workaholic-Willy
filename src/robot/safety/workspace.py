"""WorkspaceGuard, the vendor-neutral safety gate for commanded robot poses.

The guard validates that every TCP target pose lies inside the Cartesian workspace of
``WorkspaceLimitsConfig``, and that successive accepted poses differ enough from one
another to keep calibration and sampling diverse.

Numerics contract
-----------------
* A public method accepts a vendor-neutral :class:`Pose`, which is preferred and is in
  :attr:`Frame.BASE` with millimetres and an XYZW quaternion, or a vendor
  :class:`URPose`, in millimetres and axis-angle radians. A URPose is coerced at the
  boundary, and the internal state is always :class:`Pose`.
* Translation distance is millimetres.
* Orientation distance is degrees, the geodesic angle between quaternions.

This module is the only safety gate a pipeline, a planner or a calibration routine
traverses before commanding a Cartesian move.
"""

from __future__ import annotations

import numpy as np

from src.config.schema.robot import WorkspaceLimitsConfig
from src.geometry import Frame, FrameMismatchError, Pose
from src.robot.constants import SAFETY_WORKSPACE_LOG_FILE, create_robot_logger

__all__ = ["WorkspaceGuard"]


PoseLike = Pose | object


def _coerce_pose(pose: PoseLike) -> Pose:
    """Normalise a :class:`Pose` or :class:`URPose` input to :class:`Pose`.

    A ``URPose`` is converted through
    :func:`src.geometry.conversions.urpose_to_pose`, which keeps the label and switches
    millimetres and axis-angle to millimetres and an XYZW quaternion.

    A :class:`Pose` whose frame is not :attr:`Frame.BASE` is rejected.
    """
    if isinstance(pose, Pose):
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"WorkspaceGuard requires Frame.BASE; got {pose.frame!r}."
            )
        return pose
    # A vendor adapter, such as the UR URPose, converts to ``Pose`` at the boundary,
    # because the safety guard speaks vendor-neutral types only. The canonical UR
    # adapter is ``src.robot.drivers.ur.pose_adapter``.
    try:
        from src.robot.drivers.ur.pose import URPose
        from src.robot.drivers.ur.pose_adapter import urpose_to_pose
    except ImportError:  # pragma: no cover (exercised when UR driver absent)
        URPose = None  # type: ignore[misc,assignment]
        urpose_to_pose = None  # type: ignore[assignment]
    if URPose is not None and isinstance(pose, URPose):
        return urpose_to_pose(pose, frame=Frame.BASE, label=pose.label or None)
    raise TypeError(
        f"WorkspaceGuard expects Pose or URPose; got {type(pose).__name__}."
    )


def _label_of(pose: Pose) -> str:
    return pose.label or "<unlabeled>"


class WorkspaceGuard:
    """Cartesian-box and orientation-diversity gate for robot poses.

    Parameters
    ----------
    limits
        Cartesian box in millimetres, from :class:`WorkspaceLimitsConfig`.
    min_distance_mm
        Smallest translation distance in millimetres between two accepted poses that
        the diversity check allows.
    min_angle_deg
        Smallest geodesic rotation angle in degrees between two accepted poses that
        the diversity check allows.
    """

    def __init__(
        self,
        limits: WorkspaceLimitsConfig,
        min_distance_mm: float = 30.0,
        min_angle_deg: float = 5.0,
    ):
        if min_distance_mm < 0.0:
            raise ValueError(f"min_distance_mm must be >= 0, got {min_distance_mm}")
        if min_angle_deg < 0.0:
            raise ValueError(f"min_angle_deg must be >= 0, got {min_angle_deg}")
        self.limits = limits
        self.min_distance_mm = float(min_distance_mm)
        self.min_angle_deg = float(min_angle_deg)
        self.logger = create_robot_logger("WorkspaceGuard", SAFETY_WORKSPACE_LOG_FILE)

        self._accepted: list[Pose] = []

    # ------------------------------------------------------------------
    # Workspace-box check
    # ------------------------------------------------------------------

    def is_inside_workspace(self, pose: PoseLike) -> bool:
        """Check whether the TCP position falls inside the allowed box."""
        p = _coerce_pose(pose)
        x, y, z = p.position_mm
        ok = (
            self.limits.x_min <= x <= self.limits.x_max
            and self.limits.y_min <= y <= self.limits.y_max
            and self.limits.z_min <= z <= self.limits.z_max
        )
        if not ok:
            self.logger.warning(
                "Pose '%s' outside workspace: (%.1f, %.1f, %.1f) not in "
                "[%.1f..%.1f, %.1f..%.1f, %.1f..%.1f]",
                _label_of(p), x, y, z,
                self.limits.x_min, self.limits.x_max,
                self.limits.y_min, self.limits.y_max,
                self.limits.z_min, self.limits.z_max,
            )
        return ok

    # ------------------------------------------------------------------
    # Diversity check
    # ------------------------------------------------------------------

    def is_diverse_enough(self, pose: PoseLike) -> bool:
        """Check that ``pose`` differs enough from the accepted poses.

        A pose is rejected where some previously accepted pose is within both
        ``min_distance_mm`` in translation and ``min_angle_deg`` in orientation.
        Similarity requires both to be close.
        """
        candidate = _coerce_pose(pose)
        for prev in self._accepted:
            # Skip the self-comparison. Where the candidate is the pose that was
            # accepted before, by identity or by equality, it must not reject itself
            # for being too similar to itself.
            if prev is candidate or prev == candidate:
                continue
            dist = float(np.linalg.norm(candidate.position_mm - prev.position_mm))
            angle = float(np.degrees(prev.angle_to(candidate)))
            if dist < self.min_distance_mm and angle < self.min_angle_deg:
                self.logger.warning(
                    "Pose '%s' too similar to '%s' "
                    "(dist=%.1f mm < %.1f, angle=%.1f%s < %.1f)",
                    _label_of(candidate), _label_of(prev),
                    dist, self.min_distance_mm,
                    angle, "deg", self.min_angle_deg,
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Combined validation + bookkeeping
    # ------------------------------------------------------------------

    def validate(self, pose: PoseLike) -> bool:
        """Full validation, meaning the workspace limits and the diversity check."""
        if not self.is_inside_workspace(pose):
            return False
        if not self.is_diverse_enough(pose):
            return False
        return True

    def accept(self, pose: PoseLike) -> None:
        """Register a pose as used for future diversity checks."""
        p = _coerce_pose(pose)
        self._accepted.append(p)
        self.logger.info(
            "Accepted pose '%s' (#%d).", _label_of(p), len(self._accepted),
        )

    def reset(self) -> None:
        """Clear the accepted-poses history."""
        self._accepted.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def accepted_poses(self) -> list[Pose]:
        """A snapshot copy of the accepted-pose history, in vendor-neutral types."""
        return list(self._accepted)

    @property
    def num_accepted(self) -> int:
        return len(self._accepted)
