"""Vendor-neutral :class:`IKService` adapters for the grasping pipeline.

The grasping package is forbidden from importing any vendor driver, so
reachability checks reach the controller through this thin layer. The
adapters here are the only place where a real robot is touched on
behalf of grasp filtering.

Three adapters are provided:

* :class:`RobotArmIKService` wraps any object that satisfies the
  :class:`src.robot.core.RobotArm` protocol. UR, KUKA, the sim arm
  and the dummy driver qualify: those are the four registered in
  ``src.robot.drivers``. The Franka and ROS 2 packages are empty
  slots, and qualify the day a driver is registered in them. No
  vendor-specific code lives in this file.
* :class:`CachedIKService` is an LRU front-end that quantises the
  query pose so an active-perception loop does not hammer the
  controller with near-duplicate queries.
* :class:`URAnalyticIKService` is an offline solver seam wrapping
  ``ur_ikfast`` when it is installed. Construction fails with a clear
  :class:`ImportError` when the optional package is missing, rather
  than reporting offline reachability that is not available.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.robot.constants import IK_SERVICE_LOG_FILE, create_robot_logger
from src.robot.core import (
    JointPositions,
    RobotArm,
    RobotConnectionError,
    RobotKinematicsError,
)
from src.robot.grasping.planning import IKResult, IKService
from src.robot.grasping.planning.grasp_pose import GraspPose

__all__ = [
    "CachedIKService",
    "RobotArmIKService",
    "URAnalyticIKService",
]


# ``query`` runs once per candidate, hundreds of times per pick, so nothing in it logs. What is
# logged is the two things that happen once: an offline solver being loaded, and the cache reporting
# its hit rate when it is cleared. Same discipline as ``GraspCalculator``: aggregate, never per item.
logger = create_robot_logger("IKService", IK_SERVICE_LOG_FILE)


def _validate_arm_frame(pose: GraspPose) -> Pose | None:
    """Return ``pose`` as a base-frame :class:`Pose`, or ``None`` if not base.

    The grasping layer is responsible for transforming candidates into
    base frame before calling the IK service when a calibration is
    available; the service refuses to silently extrapolate.
    """
    if pose.frame is not Frame.BASE:
        return None
    return pose.as_pose(label="ik_query")


@dataclass(frozen=True, slots=True)
class RobotArmIKService:
    """IK service backed by any :class:`RobotArm` implementation.

    Parameters
    ----------
    arm:
        The vendor-neutral arm whose ``ik()`` method will be queried.
        Connection state is the caller's responsibility; if the arm is
        not connected the service maps :class:`RobotConnectionError`
        into ``IKResult(reachable=False, reason="controller_not_connected")``.
    seed_provider:
        Optional callable returning a :class:`JointPositions` to use as
        the IK seed. Defaults to the arm's current joints when
        connected, otherwise no seed. The provider is invoked per query
        so a moving robot supplies a fresh nearest-solution hint.
    """

    arm: RobotArm
    seed_provider: Any = None

    def _seed(self) -> JointPositions | None:
        if self.seed_provider is not None:
            seed = self.seed_provider()
            return seed if isinstance(seed, JointPositions) else None
        try:
            return self.arm.get_joint_positions()
        except RobotConnectionError:
            return None
        except Exception:  # noqa: BLE001 (fall back gracefully)
            return None

    def query(self, pose: GraspPose) -> IKResult:
        target = _validate_arm_frame(pose)
        if target is None:
            return IKResult(
                reachable=False,
                reason="non_base_frame",
                metadata={"frame": pose.frame.value},
            )
        try:
            joints = self.arm.ik(target, seed=self._seed())
        except RobotConnectionError as exc:
            return IKResult(
                reachable=False,
                reason="controller_not_connected",
                metadata={"detail": str(exc)},
            )
        except RobotKinematicsError as exc:
            return IKResult(
                reachable=False,
                reason="ik_no_solution",
                metadata={"detail": str(exc)},
            )
        return IKResult(
            reachable=True,
            joints=tuple(float(j) for j in joints.tolist()),
            metadata={"source": "robot_arm"},
        )


def _quantise_pose(pose: GraspPose, *, position_mm: float, angle_deg: float) -> tuple:
    """Return a hashable bucket key for a near-duplicate pose.

    Position is rounded to a ``position_mm`` grid and the rotation
    matrix entries to a step of ``deg2rad(angle_deg)``, an
    angle-equivalent grid rather than a rounded angle, so cached
    entries cover candidates that differ only by numerical noise.
    Both resolutions are required arguments here; the 1.0 defaults
    live on :class:`CachedIKService` as ``position_resolution_mm``
    and ``angle_resolution_deg``.
    """
    pos_bucket = tuple(np.round(pose.position_mm / position_mm).astype(np.int64).tolist())
    # Rotation matrix flattened then rounded to ``angle_deg``-equivalent grid.
    angle_step = np.deg2rad(angle_deg)
    rot_bucket = tuple(
        np.round(pose.rotation_matrix.ravel() / angle_step).astype(np.int64).tolist()
    )
    return (pos_bucket, rot_bucket, pose.frame.value)


@dataclass(slots=True)
class CachedIKService:
    """LRU cache in front of any :class:`IKService`.

    Repeated active-perception loops produce many near-duplicate query
    poses; this cache stops a real controller from being asked the same
    question twice in a cycle.
    """

    inner: IKService
    max_entries: int = 512
    position_resolution_mm: float = 1.0
    angle_resolution_deg: float = 1.0
    _cache: OrderedDict[tuple, IKResult] = field(default_factory=OrderedDict, init=False, repr=False)
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)

    def query(self, pose: GraspPose) -> IKResult:
        key = _quantise_pose(
            pose,
            position_mm=self.position_resolution_mm,
            angle_deg=self.angle_resolution_deg,
        )
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.hits += 1
            return cached
        self.misses += 1
        result = self.inner.query(pose)
        self._cache[key] = result
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return result

    def clear(self) -> None:
        # Logged before the reset, because after it the numbers are gone. A collapsing hit rate is how
        # a mis-tuned quantisation grid (or a controller being asked the same question hundreds of
        # times) shows up.
        logger.debug(
            "IK cache cleared: %d hit(s), %d miss(es), %d entr(ies) held",
            self.hits, self.misses, len(self._cache),
        )
        self._cache.clear()
        self.hits = 0
        self.misses = 0


class URAnalyticIKService:
    """Offline analytic IK for UR arms via ``ur_ikfast``.

    The class is an optional seam: ``ur_ikfast`` is not declared in
    ``requirements.txt`` because it is not on PyPI and its build is
    platform-sensitive (Cython + BLAS/LAPACK). Install it from source
    (``github.com/cambel/ur_ikfast``), or prefer the maintained,
    pip-installable ``ur-analytic-ik``. Only needed for offline reachability
    without a live controller; the controller's calibrated IK is otherwise
    more faithful to the physical arm than this ideal-model analytic IK.

    Parameters
    ----------
    model:
        UR model identifier accepted by ``ur_ikfast``
        (``"ur3"``, ``"ur5"``, ``"ur5e"``, ``"ur10"`` etc.).
    tool_translation_mm:
        Optional tool-tip translation (mm) baked into the wrist pose so
        the grasp-frame target matches the calibrated TCP.
    """

    def __init__(
        self,
        *,
        model: str = "ur5e",
        tool_translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        try:
            import ur_ikfast  # type: ignore
        except ImportError as exc:  # pragma: no cover (depends on optional dep)
            raise ImportError(
                "URAnalyticIKService requires the optional 'ur_ikfast' package "
                "(not on PyPI; build from source at github.com/cambel/ur_ikfast, "
                "or use the pip-installable 'ur-analytic-ik')."
            ) from exc
        try:
            # The class lives at ur_ikfast.ur_kinematics.URKinematics; some builds also
            # re-export it at the package root, so accept either.
            URKinematics = getattr(ur_ikfast, "URKinematics", None) or ur_ikfast.ur_kinematics.URKinematics
            self._solver = URKinematics(model)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover (optional dep specific)
            raise RuntimeError(f"ur_ikfast could not load model {model!r}: {exc}") from exc
        self._model = model
        self._tool_translation_mm = np.asarray(tool_translation_mm, dtype=np.float64)
        # Which offline model answered "is that reachable" matters: this is an ideal-kinematics solver,
        # not the controller's calibrated one, and a reachability verdict from it is weaker evidence.
        logger.info(
            "loaded offline analytic IK (ur_ikfast) for model %r, tool translation %s mm",
            model, tuple(float(v) for v in self._tool_translation_mm),
        )

    def query(self, pose: GraspPose) -> IKResult:  # pragma: no cover (depends on optional dep)
        if pose.frame is not Frame.BASE:
            return IKResult(reachable=False, reason="non_base_frame")
        # ur_ikfast expects metres + xyzw quaternion.
        position_m = (pose.position_mm + self._tool_translation_mm) / 1000.0
        quat = pose.quaternion_xyzw
        target = np.concatenate([position_m, quat])
        try:
            solution = self._solver.inverse(target, False)
        except Exception as exc:  # noqa: BLE001 (vendor lib raises various)
            return IKResult(
                reachable=False,
                reason="ik_no_solution",
                metadata={"detail": str(exc)},
            )
        if solution is None:
            return IKResult(reachable=False, reason="ik_no_solution")
        return IKResult(
            reachable=True,
            joints=tuple(float(j) for j in np.asarray(solution).ravel().tolist()),
            metadata={"source": "ur_ikfast", "model": self._model},
        )
