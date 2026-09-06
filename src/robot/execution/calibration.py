"""
CalibrationRoutine: end-to-end eye-to-hand / eye-in-hand calibration.

Orchestration
-------------
For each calibration pose:

1. Move the robot there via :meth:`RobotArm.move` (typed result).
2. Wait for mechanical settle (:meth:`RobotArm.wait_until_steady`,
   with a fallback to a fixed sleep when the driver does not
   implement it).
3. Read the actual TCP pose from forward kinematics. Do not trust
   the commanded pose; the controller may stop short.
4. Grab a rectified stereo frame and detect the ArUco marker.
5. Hand the (T_base_to_tool, T_cam_to_marker) pair to the
   selected :class:`EyeToHandCalibrator` / :class:`EyeInHandCalibrator`.

When all poses are processed the calibrator solves for ``T_cam_to_base``
(eye-to-hand) or ``T_cam_to_tool`` (eye-in-hand) and the result is
validated. Both metrics (RMSE, max error) are carried on
:class:`CalibrationResult`.

Vendor neutrality
-----------------
This module commands motion exclusively through the vendor-neutral
:class:`RobotArm` protocol and speaks in Willy :class:`Pose` types,
so the same routine can drive calibration on any registered driver,
real or simulated. It is not free of vendor imports:
:mod:`pose_provider` needs ``urpose_to_pose`` to read URPose-shaped
pose files, so importing this module pulls
``src.robot.drivers.ur.pose_adapter`` in with it. No vendor SDK is
loaded, and none may be.

Event surface
-------------
Progress events are emitted via the typed
:class:`RobotCalibrationEventListener` callback. The canonical event
names live in :class:`RobotCalibrationEvent` so in-process consumers
(loggers, tooling) cannot drift from the producer.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypeAlias

import numpy as np

from src.config.schema.camera import EyeHandRoutineConfig
from src.config.schema.robot import MotionLimitsConfig, WorkspaceLimitsConfig
from src.calibration import (
    Extrinsics,
    EyeInHandCalibrator,
    EyeToHandCalibrator,
    MountingMode,
)
from src.calibration.stereo.manager import StereoCam3D
from src.camera.orchestration import FrameProvider
from src.geometry import Pose, Transform
from src.geometry.adapters.matrix import pose_to_matrix
from src.robot.constants import ROBOT_LOG_DIR, ROBOT_LOG_FILE
from src.robot.core import MotionStatus, RobotArm
from src.robot.events import RobotCalibrationEvent, RobotCalibrationEventListener
from src.robot.execution.pose_provider import PoseProvider
from src.robot.safety.workspace import WorkspaceGuard
from src.utility.log_cfg import create_logger

CalibrationMode: TypeAlias = MountingMode
EyeHandCalibrator: TypeAlias = EyeToHandCalibrator | EyeInHandCalibrator
#: A pluggable marker-pose source: returns ``T_cam_to_marker`` (4x4 homogeneous, mm) for the
#: current camera view, or ``None`` when the marker is not detected. This is the one perception
#: seam of the routine: the default path uses a stereo ``FrameProvider`` + ArUco ``StereoCam3D``,
#: but injecting a ``marker_source`` (e.g. a sim Isaac-camera ArUco/ground-truth source) overrides
#: it so the same orchestration drives real-stereo, sim, or any other marker pipeline.
MarkerPoseProvider: TypeAlias = Callable[[], Optional[np.ndarray]]


@dataclass
class CalibrationResult:
    """Consolidated output of a calibration run.

    Attributes
    ----------
    T_cam_to_base : np.ndarray or None
        4x4 camera to base transform for fixed-camera eye-to-hand mode.
    T_cam_to_tool : np.ndarray or None
        4x4 camera to tool transform for eye-in-hand mode. Compose with the
        live TCP pose before using it for grasping.
    rmse_mm : float
        Validation RMSE in millimetres.
    max_error_mm : float
        Largest residual on the validation set, in millimetres.
    num_samples : int
        Number of (move, marker) samples that were actually accepted by
        the calibrator (after pose-rejection and marker-not-found
        skips).
    dataset_path : str or None
        On-disk path of the saved sample dataset, if a save path was
        requested.
    extrinsics : Extrinsics or None
        Typed, schema-versioned, frame-checked extrinsics from
        :meth:`EyeHandCalibrationResult.to_extrinsics`. ``None`` in
        eye-in-hand mode, and when the result is built directly from
        ndarrays.
    """

    T_cam_to_base: np.ndarray | None
    rmse_mm: float
    max_error_mm: float
    num_samples: int
    dataset_path: str | None = None
    extrinsics: Extrinsics | None = None
    T_cam_to_tool: np.ndarray | None = None
    transform: Transform | None = None
    mode: CalibrationMode = MountingMode.EYE_TO_HAND


class CalibrationRoutine:
    """Run an end-to-end eye-to-hand / eye-in-hand calibration.

    The routine is fully vendor-neutral: it commands motion exclusively
    through the :class:`RobotArm` protocol and reads ground-truth TCP
    poses through :meth:`RobotArm.get_tcp_pose`. Any driver that
    satisfies the protocol (UR, KUKA, sim, dummy, ...) can drive a
    calibration without any UR-specific shim.

    Parameters
    ----------
    arm : RobotArm
        Vendor-neutral robot arm. Required.
    provider : FrameProvider
        Open frame provider with at least one stereo rig configured.
    stereo : StereoCam3D
        Calibrated stereo engine for the chosen rig.
    eth_calibrator : EyeToHandCalibrator or EyeInHandCalibrator
        Sample collector / solver. Built from ``eth_settings.mode``
        when the settings object carries one: only the
        ``EyeToHandWorkflowConfig`` and ``EyeInHandWorkflowConfig``
        subclasses declare ``mode``. A plain
        :class:`EyeHandRoutineConfig` has none, and the mode then
        falls back to ``MountingMode.EYE_IN_HAND``.
    workspace_limits : WorkspaceLimitsConfig
        Cartesian workspace box (mm) used to gate every commanded move.
    eth_settings : EyeHandRoutineConfig
        Calibration-specific thresholds (``min_distance_mm``,
        ``min_angle``, ``min_samples``).
    rig_id : str
        Which rig to grab frames from (``"rig_0"`` by default).
    marker_id : int
        ArUco marker ID to look for in each frame.
    settle_time_s : float
        Maximum wait (seconds) for the robot to come to a steady stop
        after each move. Implementation prefers
        :meth:`RobotArm.wait_until_steady`; when the driver does not
        implement it, the routine falls back to a blind ``time.sleep``
        of the same duration.
    motion_limits : MotionLimitsConfig or None
        Commanded velocity / acceleration for the calibration moves: when
        supplied, ``max_velocity`` / ``max_acceleration`` are passed to
        ``RobotArm.move(vel=, acc=)``. ``None`` keeps the driver default speed.
    on_event : RobotCalibrationEventListener or None
        Progress callback. Listener exceptions are swallowed so a buggy
        subscriber cannot abort a calibration run.
    calibration_mode : CalibrationMode or str or None
        Override for ``eth_settings.mode``. Use to select
        ``MountingMode.EYE_TO_HAND`` vs ``MountingMode.EYE_IN_HAND``
        without rewriting the settings file. Pass it explicitly unless
        ``eth_settings`` is an ``EyeToHandWorkflowConfig`` or an
        ``EyeInHandWorkflowConfig``, because a plain
        :class:`EyeHandRoutineConfig` declares no ``mode`` and the
        fallback is ``MountingMode.EYE_IN_HAND``.
    """

    def __init__(
        self,
        arm: RobotArm,
        provider: FrameProvider | None = None,
        stereo: StereoCam3D | None = None,
        eth_calibrator: EyeHandCalibrator | None = None,
        workspace_limits: WorkspaceLimitsConfig | None = None,
        eth_settings: EyeHandRoutineConfig | None = None,
        rig_id: str = "rig_0",
        marker_id: int = 0,
        settle_time_s: float = 0.5,
        motion_limits: MotionLimitsConfig | None = None,
        on_event: RobotCalibrationEventListener | None = None,
        calibration_mode: CalibrationMode | str | None = None,
        marker_source: MarkerPoseProvider | None = None,
    ):
        if arm is None:
            raise ValueError("CalibrationRoutine requires arm=RobotArm.")
        # The marker comes either from a stereo provider+ArUco pipeline or from an injected
        # marker_source (e.g. the Isaac sim). Require the stereo pair only when no source is given.
        if marker_source is None and (provider is None or stereo is None):
            raise ValueError(
                "CalibrationRoutine requires provider and stereo (or an injected marker_source)."
            )
        if workspace_limits is None or eth_settings is None:
            raise ValueError(
                "CalibrationRoutine requires workspace_limits and eth_settings."
            )

        cfg_mode = getattr(eth_settings, "mode", MountingMode.EYE_IN_HAND)
        selected_mode = MountingMode(calibration_mode or cfg_mode)
        calibrator = eth_calibrator or self._build_calibrator(
            selected_mode,
            settings=eth_settings,
        )
        self._validate_calibrator_mode(calibrator, selected_mode)

        self.logger = create_logger(
            "CalibrationRoutine", ROBOT_LOG_FILE, log_dir=ROBOT_LOG_DIR,
        )
        self._on_event: RobotCalibrationEventListener | None = on_event

        self.arm = arm
        self.provider = provider
        self.stereo = stereo
        self._marker_source = marker_source
        self.calibrator = calibrator
        # ``motion_limits`` reaches the controller through :meth:`_move_to_pose`, which
        # passes max_velocity/max_acceleration to ``RobotArm.move(vel=, acc=)``.
        # ``None`` keeps the driver default speed.
        self._motion_limits = motion_limits

        self.guard = WorkspaceGuard(
            limits=workspace_limits,
            min_distance_mm=eth_settings.min_distance_mm,
            min_angle_deg=eth_settings.min_angle,
        )
        self.pose_provider = PoseProvider(self.guard)

        self.rig_id = rig_id
        self.rig_idx = provider.get_stereo_rig_index(rig_id) if provider is not None else 0
        self.marker_id = marker_id
        self.settle_time_s = max(0.0, float(settle_time_s))
        self.calibration_mode = selected_mode

    @staticmethod
    def _build_calibrator(
        mode: MountingMode,
        *,
        settings: EyeHandRoutineConfig,
    ) -> EyeHandCalibrator:
        if mode is MountingMode.EYE_TO_HAND:
            return EyeToHandCalibrator(settings=settings)
        if mode is MountingMode.EYE_IN_HAND:
            return EyeInHandCalibrator(settings=settings)
        raise ValueError(f"Unsupported calibration mode {mode!r}")

    @staticmethod
    def _validate_calibrator_mode(
        calibrator: EyeHandCalibrator,
        mode: MountingMode,
    ) -> None:
        expected_type = (
            EyeToHandCalibrator
            if mode is MountingMode.EYE_TO_HAND
            else EyeInHandCalibrator
        )
        if not isinstance(calibrator, expected_type):
            raise TypeError(
                f"{mode.value} requires {expected_type.__name__}; got "
                f"{type(calibrator).__name__}."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_from_json(
        self,
        poses_path: str | Path,
        T_tool_to_marker: np.ndarray | None = None,
        dataset_save_path: str | None = None,
    ) -> CalibrationResult:
        """Run a calibration using pre-planned poses from a JSON file."""
        poses = self.pose_provider.load_from_json(poses_path)
        return self._execute(poses, T_tool_to_marker, dataset_save_path)

    def run_auto(
        self,
        n_poses: int,
        T_tool_to_marker: np.ndarray | None = None,
        dataset_save_path: str | None = None,
        base_orientation: list[float] | None = None,
        orientation_spread_deg: float = 15.0,
        max_attempts_per_pose: int = 200,
        seed: int | None = None,
    ) -> CalibrationResult:
        """Run a calibration using auto-generated diverse poses."""
        poses = self.pose_provider.generate(
            n_poses,
            base_orientation=base_orientation,
            orientation_spread_deg=orientation_spread_deg,
            max_attempts_per_pose=max_attempts_per_pose,
            seed=seed,
        )
        return self._execute(poses, T_tool_to_marker, dataset_save_path)

    def run_with_poses(
        self,
        poses: list[Pose],
        T_tool_to_marker: np.ndarray | None = None,
        dataset_save_path: str | None = None,
    ) -> CalibrationResult:
        """Run a calibration using an explicit, caller-supplied list of TCP target poses.

        For viewpoint patterns that ``PoseProvider.generate`` cannot express, such as eye-in-hand
        look-at viewpoints that must aim a wrist camera at a fixed marker, the caller plans the
        poses itself and feeds them here. Identical collection and solve path to ``run_auto`` and
        ``run_from_json``: move, settle, read FK, capture marker, add_sample, then AX=XB.
        """
        return self._execute(list(poses), T_tool_to_marker, dataset_save_path)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _execute(
        self,
        poses: list[Pose],
        T_tool_to_marker: np.ndarray | None,
        dataset_save_path: str | None,
    ) -> CalibrationResult:
        self.logger.info(
            "Starting calibration with %d poses (marker_id=%d).",
            len(poses), self.marker_id,
        )

        accepted = 0
        total = len(poses)
        for i, pose in enumerate(poses):
            idx = i + 1
            self.logger.info("=== Pose %d/%d '%s' ===", idx, total, pose.label)
            self._emit_moving_to_pose(idx, total, pose, accepted)

            # Calibration hops between maximally-diverse viewpoints, so the motion-continuity guard
            # must not diff this move against the previous viewpoint: it would read the intentional
            # hop as a huge spurious joint/TCP step and reject it. The continuity reference is reset
            # per viewpoint. Only a driver wired with a SafetyPreflight implements
            # reset_safety_continuity(); it is a no-op otherwise, and every other guard still gates
            # each move.
            reset_continuity = getattr(self.arm, "reset_safety_continuity", None)
            if callable(reset_continuity):
                reset_continuity()

            try:
                if not self._move_to_pose(pose):
                    self._emit_pose_rejected(idx, total, reason="move_rejected")
                    continue

                self._wait_for_settle()

                T_base_to_tool = self._read_actual_tool_pose()

                T_cam_to_marker = self._capture_marker_pose()
                if T_cam_to_marker is None:
                    self._emit_pose_rejected(
                        idx, total,
                        reason="marker_not_found",
                        marker_id=self.marker_id,
                    )
                    continue

                self._emit(RobotCalibrationEvent.MARKER_DETECTED, {
                    "index": idx, "total": total, "marker_id": self.marker_id,
                })

                added = self.calibrator.add_sample(
                    T_base_to_tool, T_cam_to_marker, marker_id=self.marker_id,
                )
                if added:
                    accepted += 1
                    self.guard.accept(pose)
                    self.logger.info("Sample %d accepted.", accepted)
                    self._emit(RobotCalibrationEvent.POSE_ACCEPTED, {
                        "index": idx, "total": total, "accepted": accepted,
                    })
                else:
                    self.logger.warning("Sample rejected by calibrator.")
                    self._emit_pose_rejected(idx, total, reason="sample_rejected")

            except (RuntimeError, OSError, ValueError) as exc:
                # RuntimeError: ur_rtde controller errors / IK failures.
                # OSError:      network errors talking to controller / camera.
                # ValueError:   malformed marker / TCP / FK output.
                # Anything else (TypeError, KeyError) is a programming bug
                # and bubbles out with its stack trace.
                self.logger.error(
                    "Recoverable error at pose %d/%d '%s': %s; skipping.",
                    idx, total, pose.label, exc,
                )
                self._emit_pose_rejected(
                    idx, total, reason="exception", detail=str(exc),
                )
                continue

        self.logger.info(
            "Collection done: %d / %d samples accepted.", accepted, total,
        )

        ds_path: str | None = None
        if dataset_save_path:
            ds_path = str(self.calibrator.save_dataset(dataset_save_path))
            self.logger.info("Dataset saved to %s.", ds_path)

        result = self.calibrator.calibrate()
        if result.mode is not self.calibration_mode:
            raise RuntimeError(
                f"Calibrator returned {result.mode.value}, expected "
                f"{self.calibration_mode.value}."
            )
        transform = result.transform

        if self.calibration_mode is MountingMode.EYE_TO_HAND:
            T_cam_to_base = transform.to_matrix()
            self.logger.info(
                "Solved T_cam_to_base via AX=XB. RMSE = %.3f.",
                result.rmse_mm,
            )
            extrinsics = result.to_extrinsics(rig_id=self.rig_id)
            return CalibrationResult(
                T_cam_to_base=T_cam_to_base,
                T_cam_to_tool=None,
                rmse_mm=result.rmse_mm,
                max_error_mm=result.max_error_mm,
                num_samples=accepted,
                dataset_path=ds_path,
                extrinsics=extrinsics,
                transform=transform,
                mode=self.calibration_mode,
            )

        T_cam_to_tool = transform.to_matrix()
        self.logger.info(
            "Solved T_cam_to_tool via AX=XB. RMSE = %.3f.",
            result.rmse_mm,
        )
        return CalibrationResult(
            T_cam_to_base=None,
            T_cam_to_tool=T_cam_to_tool,
            rmse_mm=result.rmse_mm,
            max_error_mm=result.max_error_mm,
            num_samples=accepted,
            dataset_path=ds_path,
            extrinsics=None,
            transform=transform,
            mode=self.calibration_mode,
        )

    # ------------------------------------------------------------------
    # Per-pose helpers (each returns success/value or None on skip)
    # ------------------------------------------------------------------

    def _move_to_pose(self, pose: Pose) -> bool:
        """Command the move; return True on success, False on rejection.

        Uses the typed :meth:`RobotArm.move` surface so the categorical
        reason (workspace rejection / IK failure / connection error /
        controller refusal) is preserved in the routine log. Returns
        the bool the surrounding loop expects for control flow.

        When the routine was given ``motion_limits``, its ``max_velocity``
        and ``max_acceleration`` are commanded through
        ``RobotArm.move(vel=, acc=)``; without them the driver default
        speed applies.
        """
        if not isinstance(pose, Pose):
            raise TypeError(
                f"CalibrationRoutine requires Pose targets; got {type(pose).__name__}."
            )
        if self._motion_limits is not None:
            result = self.arm.move(
                pose,
                register=False,
                vel=self._motion_limits.max_velocity,
                acc=self._motion_limits.max_acceleration,
            )
        else:
            result = self.arm.move(pose, register=False)
        if result.status is MotionStatus.EXECUTED:
            return True
        self.logger.warning(
            "Move to '%s' rejected: status=%s message=%s",
            pose.label or "<unlabeled>",
            result.status.value,
            result.message or "<no detail>",
        )
        return False

    def _wait_for_settle(self) -> None:
        """Wait for the driver to report "steady".

        Tries :meth:`RobotArm.wait_until_steady` first; falls back to a
        blind ``time.sleep`` when the driver does not implement the
        method or raises a recoverable runtime fault.
        """
        if self.settle_time_s <= 0.0:
            return
        wait_fn = getattr(self.arm, "wait_until_steady", None)
        if callable(wait_fn):
            try:
                wait_fn(self.settle_time_s)
                return
            except (RuntimeError, OSError) as exc:
                self.logger.debug(
                    "wait_until_steady raised (%s); falling back to sleep.", exc,
                )
        time.sleep(self.settle_time_s)

    def _read_actual_tool_pose(self) -> np.ndarray:
        """Return the actual TCP pose as a 4x4 matrix (mm).

        Reads a typed :class:`Pose` in :attr:`Frame.BASE` from the
        driver and converts to a homogeneous matrix in millimetres.
        """
        tcp_pose = self.arm.get_tcp_pose()
        return pose_to_matrix(tcp_pose)

    def _capture_marker_pose(self) -> np.ndarray | None:
        """Return T_cam_to_marker (4x4) for the current view, or None if not detected.

        Uses the injected ``marker_source`` when one was supplied (the sim / non-stereo path);
        otherwise grabs a rectified stereo frame and runs ArUco detection (the default path).
        """
        if self._marker_source is not None:
            pose = self._marker_source()
            return None if pose is None else np.asarray(pose, dtype=np.float64)
        # The stereo path requires both a frame provider and a stereo camera (set together when no
        # marker_source is injected); narrow them, raising when either is absent.
        provider, stereo = self.provider, self.stereo
        if provider is None or stereo is None:
            raise RuntimeError("CalibrationRoutine stereo capture requires a provider + stereo camera.")
        rect_frame = provider.grab_rectified(self.rig_id)
        marker_poses = stereo.estimate_marker_pose_left(
            rect_frame.left, target_id=self.marker_id, rig=self.rig_idx,
        )
        return self._extract_marker_pose(marker_poses)

    def _extract_marker_pose(
        self,
        marker_poses: Any,
    ) -> np.ndarray | None:
        """Normalise the dict-or-array return shape to a single 4x4.

        ``StereoCam3D.estimate_marker_pose_left`` returns either a
        ``dict[id -> 4x4]`` (multi-marker mode) or a single ``(4, 4)``
        ndarray (when ``target_id`` is given). Both are accepted; in the
        dict case the configured ``marker_id`` is looked up.
        """
        if marker_poses is None:
            self.logger.warning("Marker %d not detected.", self.marker_id)
            return None
        if isinstance(marker_poses, dict):
            T = marker_poses.get(self.marker_id)
            if T is None:
                self.logger.warning(
                    "Marker %d not in detection results (have %s).",
                    self.marker_id, sorted(marker_poses.keys()),
                )
            return T
        # Assume it is already a 4x4 ndarray.
        return marker_poses

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit_moving_to_pose(
        self,
        idx: int,
        total: int,
        pose: Pose,
        accepted: int,
    ) -> None:
        # Build the event payload directly from the vendor-neutral
        # :class:`Pose` (millimetres + XYZW quaternion). The wire shape
        # matches the URPose-derived layout the event consumer reads:
        # ``x/y/z`` in millimetres, ``rx/ry/rz`` as the axis-angle
        # rotation vector in radians.
        axis_angle = pose.axis_angle_rad()
        position = pose.position_mm
        self._emit(RobotCalibrationEvent.MOVING_TO_POSE, {
            "index": idx, "total": total, "label": pose.label or "",
            "pose": {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "rx": float(axis_angle[0]),
                "ry": float(axis_angle[1]),
                "rz": float(axis_angle[2]),
            },
            "accepted": accepted,
        })

    def _emit_pose_rejected(
        self, idx: int, total: int, *, reason: str, **extra: Any,
    ) -> None:
        payload: dict[str, Any] = {"index": idx, "total": total, "reason": reason}
        payload.update(extra)
        self._emit(RobotCalibrationEvent.POSE_REJECTED, payload)

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Safe event emission: never lets a listener crash the routine."""
        if self._on_event is None:
            return
        try:
            self._on_event(event_type, data)
        except Exception:
            self.logger.debug("on_event listener raised; ignoring", exc_info=True)
