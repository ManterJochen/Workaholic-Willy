"""
PoseProvider: vendor-neutral source of TCP target poses.

Loads calibration / sampling poses from JSON files or generates them
automatically inside the configured workspace. Every pose returned is a
:class:`src.geometry.Pose` in :attr:`Frame.BASE` (millimetres +
XYZW quaternion).

Accepted record layouts
-----------------------
Two on-disk shapes are read:

* URPose-shaped records (``{x, y, z, rx, ry, rz, label}``, mm +
  axis-angle rad), detected via the ``rx``/``ry``/``rz`` keys, and
* Pose-shaped records (via :func:`pose_from_dict`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import from_axis_angle
from src.geometry.serialization import pose_from_dict
from src.robot.constants import ROBOT_LOG_DIR, ROBOT_LOG_FILE
from src.robot.drivers.ur.pose_adapter import urpose_to_pose
from src.robot.safety.workspace import WorkspaceGuard
from src.utility.io import load_json
from src.utility.log_cfg import create_logger

__all__ = ["PoseProvider"]


# Default base orientation: TCP pointing straight down (Z = -base.Z).
# Equivalent to URPose default ``[0, pi, 0]`` axis-angle.
_DEFAULT_DOWN_AXIS_ANGLE_RAD: list[float] = [0.0, float(np.pi), 0.0]


def _axis_angle_to_quat(axis_angle_rad: Sequence[float]) -> np.ndarray:
    """Build a unit XYZW quaternion from an axis-angle 3-vector (rad)."""
    return from_axis_angle(np.asarray(axis_angle_rad, dtype=np.float64))


def _record_to_pose(record: dict, *, default_label: str = "") -> Pose:
    """Convert one JSON record into a :class:`Pose` (Frame.BASE)."""
    if isinstance(record, dict) and {"rx", "ry", "rz"} <= record.keys():
        # URPose-shaped record. Coerce via URPose to Pose to reuse the
        # canonical conversion path (axis-angle rad to quaternion).
        from src.robot.drivers.ur.pose import URPose

        urpose = URPose(
            x=float(record["x"]),
            y=float(record["y"]),
            z=float(record["z"]),
            rx=float(record["rx"]),
            ry=float(record["ry"]),
            rz=float(record["rz"]),
            label=str(record.get("label", default_label)),
        )
        return urpose_to_pose(urpose, frame=Frame.BASE, label=urpose.label or None)
    if isinstance(record, dict) and "position_mm" in record and "quaternion_xyzw" in record:
        return pose_from_dict(record)
    raise ValueError(
        "Unrecognised pose JSON record. Expected URPose-shaped "
        "{x,y,z,rx,ry,rz} or Pose-shaped {position_mm, quaternion_xyzw, ...}; "
        f"got keys: {sorted(record.keys()) if isinstance(record, dict) else type(record).__name__}"
    )


class PoseProvider:
    """Source of validated calibration / sampling poses.

    Parameters
    ----------
    guard
        :class:`WorkspaceGuard` used to validate every emitted pose.
    """

    def __init__(self, guard: WorkspaceGuard):
        self.guard = guard
        self.logger = create_logger("PoseProvider", ROBOT_LOG_FILE, log_dir=ROBOT_LOG_DIR)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_json(self, path: str | Path) -> list[Pose]:
        """Load poses from a JSON file and validate each one.

        Returns
        -------
        list of Pose
            Poses (Frame.BASE) that passed the workspace + diversity
            checks, in input order.

        Raises
        ------
        ValueError
            If the file is empty, malformed, or none of the loaded poses
            pass validation.
        """
        records = load_json(path)
        if not isinstance(records, list):
            raise ValueError(
                f"Pose JSON at '{path}' must be a top-level list; got {type(records).__name__}"
            )
        poses = [
            _record_to_pose(rec, default_label=f"pose_{i}")
            for i, rec in enumerate(records)
        ]
        self.logger.info("Loaded %d poses from %s.", len(poses), path)

        valid = self._filter_valid(poses)
        if not valid:
            raise ValueError(
                f"None of the {len(poses)} poses from '{path}' "
                "passed the workspace + diversity check."
            )
        return valid

    def generate(
        self,
        n: int,
        *,
        base_orientation: list[float] | None = None,
        orientation_spread_deg: float = 15.0,
        max_attempts_per_pose: int = 200,
        seed: int | None = None,
    ) -> list[Pose]:
        """Generate ``n`` diverse poses inside the workspace.

        Parameters
        ----------
        n
            Number of poses to generate.
        base_orientation
            Reference TCP orientation as an axis-angle 3-vector in
            radians (e.g. ``[0, pi, 0]`` for "tool pointing down").
            Per-pose perturbations are added on top. ``None`` uses
            ``_DEFAULT_DOWN_AXIS_ANGLE_RAD``, which points the tool down.
        orientation_spread_deg
            Half-amplitude (deg) of the per-axis uniform perturbation
            applied around ``base_orientation``.
        max_attempts_per_pose
            How many random samples to try per generated pose before
            giving up and raising :class:`RuntimeError`.
        seed
            RNG seed for reproducibility.

        Returns
        -------
        list of Pose
            ``n`` poses, all inside the workspace, sufficiently diverse
            from each other.
        """
        rng = np.random.default_rng(seed)
        limits = self.guard.limits

        base_axis_angle = (
            list(base_orientation)
            if base_orientation is not None
            else list(_DEFAULT_DOWN_AXIS_ANGLE_RAD)
        )

        spread_rad = np.radians(orientation_spread_deg)
        poses: list[Pose] = []
        local_guard = self._fresh_local_guard()

        for i in range(n):
            found = False
            for _ in range(max_attempts_per_pose):
                x = rng.uniform(limits.x_min, limits.x_max)
                y = rng.uniform(limits.y_min, limits.y_max)
                z = rng.uniform(limits.z_min, limits.z_max)

                rx = base_axis_angle[0] + rng.uniform(-spread_rad, spread_rad)
                ry = base_axis_angle[1] + rng.uniform(-spread_rad, spread_rad)
                rz = base_axis_angle[2] + rng.uniform(-spread_rad, spread_rad)

                quat = _axis_angle_to_quat([rx, ry, rz])
                candidate = Pose(
                    position_mm=np.array([x, y, z], dtype=np.float64),
                    quaternion_xyzw=quat,
                    frame=Frame.BASE,
                    label=f"auto_{i}",
                )

                if local_guard.validate(candidate):
                    local_guard.accept(candidate)
                    poses.append(candidate)
                    found = True
                    break

            if not found:
                raise RuntimeError(
                    f"Could not generate pose #{i + 1}/{n} after "
                    f"{max_attempts_per_pose} attempts. "
                    f"Try widening the workspace or lowering diversity thresholds."
                )

        self.logger.info("Generated %d diverse poses.", len(poses))
        return poses

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fresh_local_guard(self) -> WorkspaceGuard:
        """Return a guard with the same limits/thresholds but empty state."""
        return WorkspaceGuard(
            limits=self.guard.limits,
            min_distance_mm=self.guard.min_distance_mm,
            min_angle_deg=self.guard.min_angle_deg,
        )

    def _filter_valid(self, poses: list[Pose]) -> list[Pose]:
        """Return poses that pass workspace + diversity checks, in order."""
        valid: list[Pose] = []
        local_guard = self._fresh_local_guard()
        for pose in poses:
            if local_guard.validate(pose):
                local_guard.accept(pose)
                valid.append(pose)
            else:
                self.logger.warning(
                    "Pose '%s' rejected (workspace or diversity).",
                    pose.label or "<unlabeled>",
                )
        self.logger.info("%d / %d poses accepted.", len(valid), len(poses))
        return valid
