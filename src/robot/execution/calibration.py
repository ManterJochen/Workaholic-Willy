"""
CalibrationRoutine: end-to-end eye-to-hand / eye-in-hand calibration.

Orchestration
-------------
A station is a :class:`Pose` or a :class:`JointStation` (joint angles, from
a JSON file or a caller's list). For each station:

1. Screen it where it needs screening (:meth:`PoseProvider.screen`): every
   station of a JSON file, and every joint station, is skipped unless its
   grasp centre lies inside the workspace box and differs enough from the
   stations kept before it. For a joint station that grasp centre is the
   arm's own forward kinematics of those joints, times the tool frame
   (``RobotArm.fk``, as the UR's home gate reads it), because a joint move
   skips the Cartesian box. A skipped station says why and never moves.
2. Move the robot there: a pose through :meth:`RobotArm.move`, a joint
   station through :meth:`RobotArm.move_to_joints` with exactly the joints
   the file wrote, which the arm judges as one joint move (on a cuRobo UR,
   the straight joint line by the path guard and the planner, then one
   ``moveJ`` along that line).
3. Wait for mechanical settle (:meth:`RobotArm.wait_until_steady`,
   with a fallback to a fixed sleep when the driver does not
   implement it). A wait that answers ``False`` is waited once more, and
   a second ``False`` skips the capture as ``not_steady``.
4. Read the actual TCP pose from forward kinematics. Do not trust
   the commanded pose; the controller may stop short.
5. Grab a rectified stereo frame and detect the ArUco marker.
6. Hand the (T_base_to_tool, T_cam_to_marker) pair to the
   selected :class:`EyeToHandCalibrator` / :class:`EyeInHandCalibrator`.

When all poses are processed the calibrator solves for ``T_cam_to_base``
(eye-to-hand) or ``T_cam_to_tool`` (eye-in-hand) and the result is
validated. Both metrics (RMSE, max error) are carried on
:class:`CalibrationResult`.

A generated sweep runs in bearing order round the base
(:func:`order_by_bearing`), and a JSON file runs in the order it was
written, with a warning where a joint turns more than half a turn.

An aimed sweep (:meth:`CalibrationRoutine.run_aimed`, eye in hand) aims
the CAMERA at one marker lying flat at a known place: it first looks from
where the arm stands (:meth:`CalibrationRoutine.look`, nothing moved),
estimates where the camera sits on the tool from that one view, and
visits a cone of views round the marker (``camera_aim``), each rolling
the camera about its line of sight, under the aim's heading or, where
that admits fewer views than the solve needs, the heading that admits
the most. Every view the marker is seen in refines the estimate, and the
views still to come are aimed afresh from it. A view that cannot be aimed or reached is a
:class:`SkippedStation`: reported with its reason, never moved to. Each
station that is moved to is a :class:`Pose` the arm plans and judges
when it moves, as any other.

When the sweep stops
--------------------
A refused move skips its station and the sweep goes on, when nothing
moved: a planner or a gate said no. It stops, and raises
:class:`SweepStopped` once the dataset is saved, when the arm may be
somewhere nobody judged or cannot be commanded at all: a move that ended
``CONNECTION_ERROR``, a ``CONTROLLER_REJECTED`` whose message says the
command had been sent (or that says nothing), and any failed move after
which the arm reports a stopped state (``get_robot_status``, else
``raise_if_stopped``, read best effort). A protective stop therefore ends
the sweep at the pose it happened on, instead of every pose after it being
tried and refused.

Camera world
------------
A sweep produces the transform a camera world is built with, so its moves
cannot plan against one. Every move runs inside
:func:`~src.robot.core.camera_world.without_camera_world` for the
routine's arm, with one reason per mounting unless the caller passes
``camera_world=``, and :attr:`CalibrationResult.camera_worlds` carries the
stamp of each move.

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

Every event carries ``index``, ``total`` and the pose's ``label``. A
rejection carries its ``reason`` and a one-sentence ``detail``: the
move's status and message, why the marker source returned no pose (its
``last_observation.why_not`` when it keeps one), the calibrator's
``last_rejection``, or the exception. Each pose's verdict is also kept, in
order, on :attr:`CalibrationRoutine.pose_log` and on
:attr:`CalibrationResult.pose_log`. None of this changes which move is
commanded or which sample counts; it says why.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
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
from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, Pose, Transform
from src.geometry.adapters.matrix import pose_to_matrix
from src.robot.constants import ROBOT_LOG_DIR, ROBOT_LOG_FILE
from src.robot.core import (
    CameraWorldDecline,
    CameraWorldStamp,
    JointPositions,
    MotionStatus,
    RobotArm,
    RobotEmergencyStop,
)
from src.robot.core.camera_world import without_camera_world
from src.robot.events import RobotCalibrationEvent, RobotCalibrationEventListener
from src.robot.execution.camera_aim import (
    AimedSweep,
    ArmReach,
    HeadingChoice,
    Look,
    MarkerAim,
    MountEstimate,
    NotAimed,
    View,
    estimate_camera_in_tool,
    flange_in_tool_mm,
)
from src.robot.execution.pose_provider import (
    JointStation,
    PoseProvider,
    SkippedStation,
    Station,
    StationScreen,
    joint_hop,
    load_stations,
)
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

#: The camera-world decline a sweep runs under when its caller states none, one per mounting. A
#: hand-eye sweep produces the transform a camera world is built with, so no camera world can stand
#: behind the sweep's own motions, and each motion's result says so.
_SWEEP_DECLINES: dict[MountingMode, CameraWorldDecline] = {
    MountingMode.EYE_TO_HAND: CameraWorldDecline(
        "eye-to-hand calibration: this sweep produces the CAMERA to BASE transform a camera "
        "world needs"
    ),
    MountingMode.EYE_IN_HAND: CameraWorldDecline(
        "eye-in-hand calibration: the CAMERA to TOOL transform is being solved"
    ),
}
#: The decline of a routine built without ``__init__``, whose mounting nobody set.
_SWEEP_DECLINE_UNSET_MOUNTING = CameraWorldDecline(
    "hand-eye calibration: this sweep produces the transform a camera world is built with"
)

#: How a refused move's message says the command had reached the controller: the UR driver's refusals say
#: ``moveJ was sent``, ``moveJ may have been sent`` or ``the arm may have moved part of the way``, and a refusal
#: before sending says ``was not sent`` or ``nothing moved``, which this does not match.
_SENT = re.compile(r"\bwas sent\b|\bmay have been sent\b|\bmay have moved\b", re.IGNORECASE)


class SweepStopped(RuntimeError):
    """The sweep stopped at one pose and commanded nothing after it: the arm cannot be reached, or may stand
    somewhere nobody judged.

    Raised once the pose's verdict is on the pose log and the dataset is saved. ``index`` and ``label`` name the
    pose, ``why`` is the reason in one sentence, and ``status`` is the move's status value, empty when an
    exception stopped it.
    """

    def __init__(self, message: str, *, index: int, total: int, label: str, why: str, status: str = "") -> None:
        super().__init__(message)
        self.index = index
        self.total = total
        self.label = label
        self.why = why
        self.status = status


@dataclass(frozen=True, slots=True)
class PoseVerdict:
    """What one pose of a sweep came to.

    ``index`` is 1-based in the order the stations were given. ``counted`` is whether the calibrator stored the
    sample. For a pose that did not count, ``reason`` is one of

    * ``outside_workspace`` or ``too_similar``: the screen skipped it before anything moved (a station of a JSON
      file, or a joint station, whose grasp centre is outside the box or too close to a station kept before it);
    * ``not_boxed``: a joint station whose grasp centre the arm could not place, so the box never saw it and
      nothing moved;
    * ``outside_workspace``, ``out_of_reach``, ``outside_joint_window``, ``too_tilted`` or ``not_aimed``: a view of
      an aimed sweep that could not be aimed or reached (a :class:`SkippedStation`), so nothing moved;
    * ``move_rejected``: the arm refused or failed the move;
    * ``not_steady``: the arm did not report steady after two waits, so no frame was taken;
    * ``marker_not_found``, ``sample_rejected`` or ``exception``,

    and ``detail`` says why in one sentence. For a pose that counted, ``detail`` is what the target's pose rests
    on when the marker source says, such as ``24 corners, 0.31 px, 520 mm away``.
    """

    index: int
    label: str
    counted: bool
    reason: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "label": self.label, "counted": self.counted, "reason": self.reason,
                "detail": self.detail}


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
    camera_worlds : tuple of CameraWorldStamp
        What stood behind each move the sweep commanded: one stamp per
        move in command order, rejected moves included. A move that raised
        returned no result and has no stamp.
    pose_log : tuple of PoseVerdict
        Each pose the sweep visited, in order: counted, or why not.
    aim_estimates : tuple of MountEstimate
        For an aimed sweep, each estimate of where the camera sits on the
        tool that its stations were aimed from, in order: the first look's
        (or the mount stated for a look that saw nothing), then each
        refinement. Empty for any other sweep.
    aim_heading : HeadingChoice or None
        For an aimed sweep, the heading its stations kept: the aim's, or the
        one that admitted the most views where the aim's admitted fewer than
        the solve needs. ``None`` for any other sweep.
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
    camera_worlds: tuple[CameraWorldStamp, ...] = ()
    pose_log: tuple[PoseVerdict, ...] = ()
    aim_estimates: tuple[MountEstimate, ...] = ()
    aim_heading: HeadingChoice | None = None


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
        ArUco marker ID to look for in each frame, recorded on each sample.
        ``-1`` for a target that is not one marker, such as a ChArUco board.
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
    camera_world : CameraWorldDecline, keyword only
        The decline every move of the sweep runs under. Left unset, the
        mounting's own: the sweep produces the transform a camera world
        needs, so no camera world can stand behind its moves. On an arm
        that plans with cuRobo each move then says DECLINED with this
        reason, and an arm whose live camera world is wired refuses a
        declined planned move. A move no planner plans says UNPLANNED.
    """

    #: The decline the sweep's moves run under. ``__init__`` sets it from the caller or the
    #: mounting; this value serves a routine built without ``__init__``.
    _camera_world: CameraWorldDecline = _SWEEP_DECLINE_UNSET_MOUNTING
    #: One stamp per move commanded since the run began, in command order.
    _camera_worlds: tuple[CameraWorldStamp, ...] = ()
    #: The result of the last move commanded, kept so a rejection can say why. A class default, so a routine
    #: built without ``__init__`` has one.
    _last_move: Any = None
    #: No marker source until ``__init__`` names one; the stereo path is used then.
    _marker_source: MarkerPoseProvider | None = None
    #: The verdict of each pose visited since the run began, in order. Kept on the routine as well as on the
    #: result, so a caller still has it when the solve raises.
    pose_log: tuple[PoseVerdict, ...] = ()
    #: The aimed sweep of the last :meth:`run_aimed`, ``None`` before one; its ``estimates`` are what the stations
    #: were aimed from, kept here as well as on the result for a solve that raises.
    aimed: AimedSweep | None = None

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
        *,
        camera_world: Maybe[CameraWorldDecline] = UNSET,
    ):
        if arm is None:
            raise ValueError("CalibrationRoutine requires arm=RobotArm.")
        if chosen(camera_world) and not isinstance(camera_world, CameraWorldDecline):
            raise TypeError(
                f"camera_world is a CameraWorldDecline, not {camera_world!r}; pass "
                f"CameraWorldDecline('why this sweep needs no camera world'), or leave it unset "
                f"for the mounting's own"
            )
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
        self._camera_world = camera_world if chosen(camera_world) else _SWEEP_DECLINES[selected_mode]

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
        """Run a calibration using pre-planned stations from a JSON file, in the order it lists them.

        The records may be poses or joint stations (:mod:`pose_provider`), and every record is checked
        before anything moves. Each station is screened as the sweep reaches it: one whose grasp centre is
        outside the workspace box, or too close to a station kept before it, is reported with its label and
        reason (``outside_workspace``, ``too_similar``) and never moved to.
        """
        stations = load_stations(poses_path)
        self.logger.info("Loaded %d stations from %s.", len(stations), poses_path)
        return self._execute(stations, T_tool_to_marker, dataset_save_path, screen_all=True)

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
        """Run a calibration using auto-generated diverse poses, in bearing order round the base.

        The round starts at the end nearer where the tool stands, when the arm says where that is.
        """
        poses = self.pose_provider.generate(
            n_poses,
            base_orientation=base_orientation,
            orientation_spread_deg=orientation_spread_deg,
            max_attempts_per_pose=max_attempts_per_pose,
            seed=seed,
            start_mm=self._tool_position_mm(),
        )
        return self._execute(poses, T_tool_to_marker, dataset_save_path)

    def run_with_poses(
        self,
        poses: Sequence[Station],
        T_tool_to_marker: np.ndarray | None = None,
        dataset_save_path: str | None = None,
    ) -> CalibrationResult:
        """Run a calibration using an explicit, caller-supplied list of stations, in order.

        For viewpoint patterns that ``PoseProvider.generate`` cannot express, such as eye-in-hand
        look-at viewpoints that must aim a wrist camera at a fixed marker, the caller plans the
        poses itself and feeds them here. Identical collection and solve path to ``run_auto`` and
        ``run_from_json``: move, settle, read FK, capture marker, add_sample, then AX=XB.

        A station is a :class:`Pose`, which the arm's own gates box when it moves, or a
        :class:`JointStation`, which is screened first as a JSON file's stations are.
        """
        return self._execute(list(poses), T_tool_to_marker, dataset_save_path)

    def look(self) -> Look:
        """What the camera sees from where the arm stands. Nothing moves.

        Waits for the arm to report steady as after any move, reads the TCP, then takes one judged
        frame through the marker source, in the order a sweep station does, so the TCP and the frame
        belong together. ``marker_in_camera`` is ``None`` where the source found no marker, with its
        sentence in ``why_not``. Raises :class:`NotAimed` where the arm does not report steady, and
        lets a camera that fails raise as it would at any station.
        """
        if not self._wait_for_settle():
            raise NotAimed(
                f"the arm did not report steady within {self.settle_time_s:g} s, twice, where it stands, so no first "
                "look was taken and nothing moved")
        tcp_pose = self._read_actual_tcp_pose()
        joints = self._joints_now()
        T_cam_to_marker = self._capture_marker_pose()
        observation = self._last_observation()
        why_not = "" if T_cam_to_marker is not None else (
            str(getattr(observation, "why_not", "") or "") or "the marker source returned no pose")
        return Look(tool_in_base=tcp_pose, joints=joints, marker_in_camera=T_cam_to_marker, why_not=why_not)

    def run_aimed(
        self,
        aim: MarkerAim,
        dataset_save_path: str | None = None,
        *,
        reach: ArmReach | None = None,
        margin_mm: float = 0.0,
    ) -> CalibrationResult:
        """An eye-in-hand sweep aimed at one marker: look once, estimate the mount, visit a cone of views round it.

        The first look (:meth:`look`) is taken where the arm stands. Where it sees the marker, the
        camera's pose in the tool frame is estimated from that one view
        (:func:`~src.robot.execution.camera_aim.estimate_camera_in_tool`, the camera first taken to
        stand at the flange when ``reach`` or the arm says where that is); where it does not,
        ``aim.mount_if_unseen`` is aimed from, and without one :class:`NotAimed` is raised with
        nothing moved and a sentence saying what to do. The views of ``aim`` are then aimed, screened
        against the workspace box less ``margin_mm`` and, with ``reach``, against the arm's reach and
        joint window, and ordered by joint travel from where the arm stands. The sweep visits them in
        that order, and every view the marker is seen in refines the estimate and re-aims the views
        still to come (:class:`~src.robot.execution.camera_aim.AimedSweep`). A view that cannot be
        aimed or reached is reported with its reason and never moved to; every other is a ``Pose``
        the arm plans and judges when it moves, as any station. Each view also rolls the camera
        about its line of sight; a rolled station the arm refuses with nothing moved is tried once
        more with the tool's heading, as ``<label>_r0`` (``AimedSweep.unrolled``).

        Before any station, the heading: where ``aim.closing_axis`` admits fewer views than the
        solve's ``min_samples``, the one of ``-y``, ``y``, ``x``, ``-x`` that admits the most is
        kept instead (``AimedSweep.settle_heading``), the log says so, and
        :attr:`CalibrationResult.aim_heading` carries the choice. A camera tilted toward the tool's -X
        under a heading written for one tilted toward +X stands every view outside the box, and the
        opposite heading stands them inside.

        The first look is not a sample: every sample comes from a station the sweep moved to.
        """
        if self.calibration_mode is not MountingMode.EYE_IN_HAND:
            raise ValueError("run_aimed aims a camera on the tool, an eye_in_hand sweep; this routine is "
                             f"{self.calibration_mode.value}")
        prior = flange_in_tool_mm(reach.flange_to_tcp if reach is not None
                                  else getattr(self.arm, "active_tool_frame", None))
        look = self.look()
        views: tuple[View, ...] = ()
        if look.view is None:
            if aim.mount_if_unseen is None:
                raise NotAimed(
                    f"the first look, from where the arm stands, saw no marker ({look.why_not}), so no station could "
                    "be aimed and nothing moved. Jog the arm on the pendant until the marker is in the camera's view, "
                    f"about {aim.distance_mm:.0f} mm from it and seen from one side rather than from straight above, "
                    "then run this again; or state the camera's mount (MarkerAim.mount_if_unseen) to aim from. To "
                    "see the camera while you jog, build the cell in the operator console (python -m api --profile "
                    "<cell>) and watch its camera view, which shows a frame taken now (api/viewfinder.py); stop the "
                    "console before this runs, because both open the camera")
            mount = aim.mount_if_unseen
            self.logger.warning("First look: no marker (%s). Aiming from %s until a station sees it.",
                                look.why_not, mount.source)
        else:
            try:
                mount = estimate_camera_in_tool(look.tool_in_base, look.marker_in_camera, aim.marker_mm,
                                                camera_prior_in_tool_mm=prior)
            except ValueError as exc:
                raise NotAimed(
                    f"the first look saw the marker, and its view cannot be aimed from: {exc}. Nothing moved; jog the "
                    "arm and run this again") from None
            views = (look.view,)
            self.logger.info("First look: %s.", mount.line())
        sweep = AimedSweep(aim, mount, box=self.guard.limits, margin_mm=margin_mm, reach=reach,
                           camera_prior_in_tool_mm=prior, joints_now=self._joints_now,
                           tool_now=self._tool_position_mm)
        self.aimed = sweep
        start_tool = [float(v) for v in look.tool_in_base.position_mm]
        heading = sweep.settle_heading(self._min_samples() or 1, start_joints=look.joints, start_tool_mm=start_tool)
        if heading.changed or dict(heading.admitted)[heading.asked] < heading.need:
            self.logger.warning("Heading: %s.", heading.line())
        else:
            self.logger.info("Heading: %s.", heading.line())
        planned = sweep.stations(start_joints=look.joints, start_tool_mm=start_tool)
        moving = sum(1 for station in planned if station.pose is not None)
        self.logger.info("%s: %d of %d views aimed, %s.", sweep.aim.line(), moving, len(planned),
                         "ordered by joint travel" if reach is not None and look.joints is not None
                         else "ordered by the tool's travel")
        return self._execute([station.as_station() for station in planned], None, dataset_save_path,
                             replan=sweep.replan, views=views, retry=sweep.unrolled)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _execute(
        self,
        poses: Sequence[Station | SkippedStation],
        T_tool_to_marker: np.ndarray | None,
        dataset_save_path: str | None,
        *,
        screen_all: bool = False,
        replan: Callable[[tuple[View, ...], tuple[str, ...]], Sequence[Station | SkippedStation] | None] | None = None,
        views: Sequence[View] = (),
        retry: Callable[[str], Station | None] | None = None,
    ) -> CalibrationResult:
        """Visit each station in order, then solve. ``screen_all`` screens poses too, not only joint stations.

        A :class:`SkippedStation` is reported with its reason and never moved to. ``replan``, when given, is called
        after every frame in which the marker was posed, with every view so far (``views`` first) and the labels of
        the stations reached so far; a sequence it returns replaces the stations still to come, and ``None`` keeps
        them. ``retry``, when given, is called with a station's label after the arm refused it and the sweep goes on
        (nothing moved); a station it returns is visited next, and ``None`` visits none. Each station either returns is
        commanded, judged and reported as any other.
        """
        self.logger.info(
            "Starting calibration with %d poses (marker_id=%d).",
            len(poses), self.marker_id,
        )

        if replan is None:
            self.aimed = None  # the estimates of an earlier aimed run belong to that run's result
        accepted = 0
        stations: list[Station | SkippedStation] = list(poses)
        seen_views: list[View] = list(views)
        reached: list[str] = []
        need = self._min_samples()
        self._camera_worlds = ()
        self.pose_log = ()
        screen = self.pose_provider.screen()
        stopped: SweepStopped | None = None
        total = len(stations)
        i = 0
        while i < len(stations):
            station = stations[i]
            i += 1
            idx, total = i, len(stations)
            if isinstance(station, SkippedStation):
                reached.append(station.label)
                self.logger.warning("Pose %d/%d '%s' skipped, %s: %s", idx, total, station.label, station.reason,
                                    station.detail)
                self._reject(idx, total, station.label, station.reason, station.detail)
                continue
            if not isinstance(station, (Pose, JointStation)):
                raise TypeError(
                    f"CalibrationRoutine requires Pose or JointStation targets; got {type(station).__name__}."
                )
            label = station.label or ""
            reached.append(label)
            self.logger.info("=== Pose %d/%d '%s' ===", idx, total, station.label)
            target = self._screened(idx, total, station, screen, screen_all=screen_all)
            if target is None:
                continue
            self._emit_moving_to_pose(idx, total, target, accepted, need, station=station)

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
                self._last_move = None
                moved = (self._move_to_joint_station(station) if isinstance(station, JointStation)
                         else self._move_to_pose(station))
                if not moved:
                    move = self._last_move
                    status = getattr(getattr(move, "status", None), "value", "") or "rejected"
                    message = str(getattr(move, "message", "") or "")
                    detail = f"{status}: {message}" if message else status
                    why = self._why_the_sweep_stops(move)
                    if why is not None:
                        stopped = self._stop(idx, total, label, "move_rejected", detail, why, status=status,
                                             message=message)
                        break
                    self._reject(idx, total, label, "move_rejected", detail, status=status, message=message)
                    if retry is not None:
                        try:
                            instead = retry(label)
                        except Exception:  # noqa: BLE001 (the hook chooses a target only; the sweep goes on)
                            self.logger.exception("The retry hook raised; the sweep goes on to the next station.")
                            instead = None
                        if instead is not None:
                            self.logger.info("Pose %d/%d '%s' refused; its view is tried once more as '%s'.", idx,
                                             total, label, getattr(instead, "label", "") or "")
                            stations.insert(i, instead)
                    continue

                if not self._wait_for_settle():
                    self._reject(idx, total, label, "not_steady",
                                 f"the arm did not report steady within {self.settle_time_s:g} s, twice, after the "
                                 "move, so no frame was taken")
                    continue

                # The TCP the arm reports, not the station: the sample and the guard's history both rest on
                # where the arm is, and a joint station has no pose of its own to stand in for it.
                tcp_pose = self._read_actual_tcp_pose()
                T_base_to_tool = pose_to_matrix(tcp_pose)

                T_cam_to_marker = self._capture_marker_pose()
                observation = self._last_observation()
                if T_cam_to_marker is None:
                    why_not = str(getattr(observation, "why_not", "") or "") or (
                        "the marker source returned no pose")
                    # The one verdict that said nothing on the console: the log jumped to the next pose.
                    self.logger.warning("Pose %d/%d '%s': no marker pose: %s", idx, total, label, why_not)
                    self._reject(idx, total, label, "marker_not_found", why_not,
                                 marker_id=self.marker_id, why_not=why_not)
                    continue

                seen = self._seen(T_cam_to_marker, observation)
                if seen["hint"]:
                    self.logger.warning("Pose %d/%d '%s': %s", idx, total, label, seen["hint"])
                self._emit(RobotCalibrationEvent.MARKER_DETECTED, {
                    "index": idx, "total": total, "label": label, "marker_id": self.marker_id, **seen,
                })

                added = self.calibrator.add_sample(
                    T_base_to_tool, T_cam_to_marker, marker_id=self.marker_id,
                )
                if added:
                    accepted += 1
                    self.guard.accept(tcp_pose)
                    self.logger.info("Sample %d accepted.", accepted)
                    self.pose_log = (*self.pose_log, PoseVerdict(idx, label, True, detail=seen["summary"]))
                    self._emit(RobotCalibrationEvent.POSE_ACCEPTED, {
                        "index": idx, "total": total, "label": label, "accepted": accepted,
                        "min_samples": need, "detail": seen["summary"],
                    })
                else:
                    rejection = getattr(self.calibrator, "last_rejection", None)
                    detail = rejection.render() if rejection is not None else "the calibrator refused the sample"
                    self.logger.warning("Sample rejected by calibrator: %s.", detail)
                    extra = {"rejection": rejection.to_dict()} if rejection is not None else {}
                    self._reject(idx, total, label, "sample_rejected", detail, **extra)

                if replan is not None:
                    # Every view the marker was posed in, counted or not: what the camera saw is the same
                    # evidence of where it sits whether or not the solve keeps the sample.
                    seen_views.append(View.of(tcp_pose, T_cam_to_marker))
                    try:
                        upcoming = replan(tuple(seen_views), tuple(reached))
                    except Exception:  # noqa: BLE001 (the hook chooses targets only; the planned ones stay)
                        self.logger.exception("The re-plan hook raised; the stations still to come stay as planned.")
                        upcoming = None
                    if upcoming is not None:
                        stations = stations[:i] + list(upcoming)

            except (RuntimeError, OSError, ValueError) as exc:
                # RuntimeError: ur_rtde controller errors / IK failures.
                # OSError:      network errors talking to controller / camera.
                # ValueError:   malformed marker / TCP / FK output.
                # Anything else (TypeError, KeyError) is a programming bug
                # and bubbles out with its stack trace.
                self.logger.error(
                    "Recoverable error at pose %d/%d '%s': %s; skipping.",
                    idx, total, station.label, exc,
                )
                why = self._stopped_state()
                if why is not None:
                    stopped = self._stop(idx, total, label, "exception", str(exc), why)
                    break
                self._reject(idx, total, label, "exception", str(exc))
                continue

        self.logger.info(
            "Collection done: %d / %d samples accepted.", accepted, total,
        )

        ds_path: str | None = None
        if dataset_save_path:
            if stopped is not None:
                # The samples taken before the stop are kept, as a solve that fails keeps them; a save that fails
                # here must not hide why the sweep stopped.
                try:
                    ds_path = str(self.calibrator.save_dataset(dataset_save_path))
                except Exception:  # noqa: BLE001 (the stop is what the caller is told)
                    self.logger.exception("The dataset of the stopped sweep could not be saved.")
            else:
                ds_path = str(self.calibrator.save_dataset(dataset_save_path))
            if ds_path is not None:
                self.logger.info("Dataset saved to %s.", ds_path)
        if stopped is not None:
            raise stopped

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
                camera_worlds=self._camera_worlds,
                pose_log=self.pose_log,
                aim_estimates=self._aim_estimates(),
                aim_heading=self._aim_heading(),
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
            camera_worlds=self._camera_worlds,
            pose_log=self.pose_log,
            aim_estimates=self._aim_estimates(),
            aim_heading=self._aim_heading(),
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

        The move runs inside the sweep's camera-world decline, bound to this
        routine's arm, and its stamp is kept for ``CalibrationResult``. A
        result that carries no stamp counts as UNSTATED, because nothing
        said what stood behind it. A rejection is logged with the stamp, and
        the result is kept on ``_last_move`` so the sweep's event can say why.
        """
        if not isinstance(pose, Pose):
            raise TypeError(
                f"CalibrationRoutine requires Pose targets; got {type(pose).__name__}."
            )
        with without_camera_world(self.arm, self._camera_world.reason):
            if self._motion_limits is not None:
                result = self.arm.move(
                    pose,
                    register=False,
                    vel=self._motion_limits.max_velocity,
                    acc=self._motion_limits.max_acceleration,
                )
            else:
                result = self.arm.move(pose, register=False)
        return self._kept(result, pose.label)

    def _move_to_joint_station(self, station: JointStation) -> bool:
        """Command one joint move to exactly ``station.joints``; return True on success, False on rejection.

        :meth:`RobotArm.move_to_joints` with the one :class:`JointPositions` the file wrote and the
        screen boxed, never a copy or a rounding of it: the arm judges that configuration's move
        (on a cuRobo UR, the straight joint line by the path guard and by the planner) and sends
        the same one. ``motion_limits`` reach it as ``velocity`` and ``acceleration``. The move
        runs inside the sweep's camera-world decline, and its result is kept as
        :meth:`_move_to_pose` keeps a pose's.
        """
        with without_camera_world(self.arm, self._camera_world.reason):
            if self._motion_limits is not None:
                result = self.arm.move_to_joints(
                    station.joints,
                    velocity=self._motion_limits.max_velocity,
                    acceleration=self._motion_limits.max_acceleration,
                )
            else:
                result = self.arm.move_to_joints(station.joints)
        return self._kept(result, station.label)

    def _kept(self, result: Any, label: str | None) -> bool:
        """Keep a move's result and stamp; ``True`` when it executed. A rejection is logged with its stamp.

        A result that carries no stamp counts as UNSTATED, because nothing said what stood behind it.
        """
        self._last_move = result
        stamp = getattr(result, "camera_world", None)
        if not isinstance(stamp, CameraWorldStamp):
            stamp = CameraWorldStamp.unstated()
        self._camera_worlds = (*self._camera_worlds, stamp)
        if result.status is MotionStatus.EXECUTED:
            return True
        self.logger.warning(
            "Move to '%s' rejected: status=%s message=%s; %s",
            label or "<unlabeled>",
            result.status.value,
            result.message or "<no detail>",
            stamp.render(),
        )
        return False

    def _screened(
        self, idx: int, total: int, station: Station, screen: StationScreen, *, screen_all: bool,
    ) -> Pose | None:
        """The pose a station stands for, when it may be commanded; ``None`` when it was skipped and said so.

        A joint station is always screened, on the grasp centre the arm's forward kinematics puts
        at its joints (:meth:`_grasp_centre_at`), because a joint move skips the Cartesian box. A
        pose is screened when ``screen_all`` says so (a JSON file's), and otherwise is left to the
        arm's own gates when it moves, as a caller's list always was. Every station that is not
        skipped is kept on the screen, so the diversity of the next is judged against it.
        """
        label = station.label or ""
        if isinstance(station, JointStation):
            target, unread = self._grasp_centre_at(station)
            if target is None:
                self._reject(idx, total, label, "not_boxed", unread)
                return None
            refused = screen.refusal(target, what="the grasp centre at these joints")
        else:
            target = station
            refused = screen.refusal(target) if screen_all else None
        if refused is not None:
            reason, detail = refused
            self.logger.warning("Pose %d/%d '%s' skipped, %s: %s", idx, total, label, reason, detail)
            self._reject(idx, total, label, reason, detail)
            return None
        screen.keep(target)
        return target

    def _grasp_centre_at(self, station: JointStation) -> tuple[Pose | None, str]:
        """``(pose, "")``: where the grasp centre stands at ``station``'s joints, or ``(None, why)``.

        The arm's own forward kinematics first, ``RobotArm.fk``: on a UR that is the controller's FK
        with its TCP offset passed explicitly, times the declared tool frame where this code owns it,
        which is how the home gate boxes home. Where that raises, the arm's DH chain times the tool
        frame (``_tcp_at_joints_mm``) when the driver has one. A station neither can place is not
        moved to: a joint move skips the box, and a station the box never saw is what this stops.
        """
        failures: list[str] = []
        fk = getattr(self.arm, "fk", None)
        if callable(fk):
            try:
                pose = fk(station.joints)
                if isinstance(pose, Pose) and pose.frame is Frame.BASE:
                    return pose.with_label(station.label), ""
                failures.append(f"fk gave {type(pose).__name__}, not a BASE pose")
            except Exception as exc:  # noqa: BLE001 (an FK that cannot be read refuses the station below)
                failures.append(f"fk raised {type(exc).__name__}: {exc}")
        chain = getattr(self.arm, "_tcp_at_joints_mm", None)
        if callable(chain):
            try:
                matrix = chain(station.joints)
                if matrix is not None:
                    return Pose.from_matrix(np.asarray(matrix, dtype=np.float64), frame=Frame.BASE,
                                            label=station.label), ""
                failures.append("the arm's kinematic chain has no tool frame or no model to place it with")
            except Exception as exc:  # noqa: BLE001 (as above)
                failures.append(f"the kinematic chain raised {type(exc).__name__}: {exc}")
        said = "; ".join(failures) if failures else "the arm offers no forward kinematics"
        return None, (f"the grasp centre at these joints could not be read ({said}), and a joint station the "
                      "workspace box never saw is not moved to, so nothing moved")

    def _why_the_sweep_stops(self, move: Any) -> str | None:
        """Why a refused move ends the sweep, in one clause, or ``None`` when the sweep may go on.

        It stops where the controller cannot be reached (``CONNECTION_ERROR``), where the
        controller refused a command it had been sent (``CONTROLLER_REJECTED`` whose message says
        it was sent, or says nothing, which is how the UR driver's ik path reports a ``moveJ`` that
        returned ``False``), and where the arm reports a stopped state. Everything else is a planner
        or a gate refusing before anything moved, and the next station is tried.
        """
        status = getattr(move, "status", None)
        message = str(getattr(move, "message", "") or "")
        if status is MotionStatus.CONNECTION_ERROR:
            return "the move ended connection_error, so the controller cannot be commanded"
        if status is MotionStatus.CONTROLLER_REJECTED and (not message.strip() or _SENT.search(message)):
            return ("the controller refused the move after it was sent, so the arm may stand part of the way along "
                    "it" if message.strip() else
                    "the controller refused the move and the driver did not say whether it had been sent")
        return self._stopped_state()

    def _stopped_state(self) -> str | None:
        """Why the arm counts as stopped, read best effort, or ``None`` when it is not or cannot say.

        ``get_robot_status`` first (a protective or emergency stop, or a stop safety mode), then
        ``raise_if_stopped`` for an arm that offers only that. A read that fails is no answer, never
        a second fault.
        """
        read = getattr(self.arm, "get_robot_status", None)
        if callable(read):
            try:
                status = read()
            except Exception:  # noqa: BLE001 (best effort after a failure)
                self.logger.debug("get_robot_status raised after a failed move", exc_info=True)
            else:
                if getattr(status, "is_stopped", None) is True:
                    mode = getattr(getattr(status, "safety_mode", None), "value", None) or "a stop"
                    text = str(getattr(status, "message", "") or "").strip()
                    return f"the controller reports it is stopped (safety mode {mode}{': ' + text if text else ''})"
                return None
        check = getattr(self.arm, "raise_if_stopped", None)
        if callable(check):
            try:
                check()
            except RobotEmergencyStop as exc:
                return f"the controller reports it is stopped ({exc})"
            except Exception:  # noqa: BLE001 (best effort after a failure)
                self.logger.debug("raise_if_stopped raised after a failed move", exc_info=True)
        return None

    def _stop(self, idx: int, total: int, label: str, reason: str, detail: str, why: str, **extra: Any) -> SweepStopped:
        """Record the pose that stops the sweep, say so, and return the exception raised once the dataset is saved."""
        self.logger.error("Pose %d/%d '%s': %s. The sweep stops here and commands nothing more.", idx, total, label,
                          why)
        self._reject(idx, total, label, reason, f"{detail}; the sweep stops here: {why}", stops_sweep=why, **extra)
        counted = sum(1 for verdict in self.pose_log if verdict.counted)
        return SweepStopped(
            f"the sweep stopped at pose {idx}/{total} {label!r} with {counted} counted, and commanded nothing after "
            f"it: {why}. {detail}",
            index=idx, total=total, label=label, why=why, status=str(extra.get("status", "")),
        )

    def _wait_for_settle(self) -> bool:
        """Wait for the driver to report "steady"; ``False`` when it said it was not, twice.

        Tries :meth:`RobotArm.wait_until_steady` first; falls back to a
        blind ``time.sleep`` when the driver does not implement the
        method or raises a recoverable runtime fault. A wait that answers
        ``False`` (still moving at its timeout) is waited once more, and a
        second ``False``, or a raise after a first ``False``, is not steady:
        a frame taken while the tool still moves is placed where it no
        longer stands. Any other answer counts as steady, as before.
        """
        if self.settle_time_s <= 0.0:
            return True
        wait_fn = getattr(self.arm, "wait_until_steady", None)
        if callable(wait_fn):
            first = True
            try:
                if wait_fn(self.settle_time_s) is not False:
                    return True
                first = False
                self.logger.warning("The arm was not steady after %.2f s; waiting once more.", self.settle_time_s)
                return wait_fn(self.settle_time_s) is not False
            except (RuntimeError, OSError) as exc:
                if not first:
                    self.logger.warning("wait_until_steady raised (%s) after the arm was not steady.", exc)
                    return False
                self.logger.debug(
                    "wait_until_steady raised (%s); falling back to sleep.", exc,
                )
        time.sleep(self.settle_time_s)
        return True

    def _read_actual_tcp_pose(self) -> Pose:
        """The TCP the arm reports now, a typed :class:`Pose` in :attr:`Frame.BASE`, in millimetres."""
        return self.arm.get_tcp_pose()

    def _joints_now(self) -> tuple[float, ...] | None:
        """The joints the arm stands at, in radians, or ``None`` where it cannot say. Read best effort: only the
        order of an aimed sweep reads it."""
        try:
            joints = self.arm.get_joint_positions()
        except Exception:  # noqa: BLE001 (an order's input, never a gate's)
            return None
        values = getattr(joints, "values", joints)
        try:
            return tuple(float(v) for v in values)
        except (TypeError, ValueError):
            return None

    def _aim_estimates(self) -> tuple[MountEstimate, ...]:
        """What the last aimed sweep's stations were aimed from, in order; empty for any other sweep."""
        return tuple(self.aimed.estimates) if self.aimed is not None else ()

    def _aim_heading(self) -> HeadingChoice | None:
        """The heading the last aimed sweep's stations kept; ``None`` for any other sweep."""
        return self.aimed.heading if self.aimed is not None else None

    def _tool_position_mm(self) -> list[float] | None:
        """Where the tool stands now, in BASE millimetres, or ``None`` where the arm cannot say."""
        try:
            pose = self.arm.get_tcp_pose()
        except Exception:  # noqa: BLE001 (only the order of a generated sweep reads this)
            return None
        if not isinstance(pose, Pose) or pose.frame is not Frame.BASE:
            return None
        return [float(v) for v in pose.position_mm]

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

    def _min_samples(self) -> int | None:
        """How many samples the solve needs, as the calibrator holds it; ``None`` when it does not say."""
        settings = getattr(self.calibrator, "settings", None)
        value = getattr(settings, "min_samples", None)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    def _last_observation(self) -> Any:
        """What the marker source's last frame showed, when the source keeps it (``last_observation``)."""
        return getattr(self._marker_source, "last_observation", None) if self._marker_source is not None else None

    @staticmethod
    def _seen(T_cam_to_marker: np.ndarray, observation: Any) -> dict[str, Any]:
        """What a detection rests on, for the event and the pose log. Values a source does not give are ``None``."""
        found = observation is not None and getattr(observation, "T_cam_to_target", None) is not None
        distance = float(np.linalg.norm(np.asarray(T_cam_to_marker, dtype=np.float64)[:3, 3]))
        reprojection = getattr(observation, "reprojection_px", None) if found else None
        points = int(getattr(observation, "n_points", 0) or 0) if found else 0
        parts = [f"{points} corners"] if points else []
        if reprojection is not None:
            parts.append(f"{float(reprojection):.2f} px")
        parts.append(f"{distance:.0f} mm away")
        return {
            "distance_mm": distance,
            "reprojection_px": None if reprojection is None else float(reprojection),
            "n_points": points or None,
            "hint": str(getattr(observation, "hint", "") or "") if found else "",
            "summary": ", ".join(parts),
        }

    def _reject(self, idx: int, total: int, label: str, reason: str, detail: str, **extra: Any) -> None:
        """Record a pose that did not count, and emit it with its reason and one sentence of why."""
        self.pose_log = (*self.pose_log, PoseVerdict(idx, label, False, reason=reason, detail=detail))
        self._emit_pose_rejected(idx, total, reason=reason, label=label, detail=detail, **extra)

    def _emit_moving_to_pose(
        self,
        idx: int,
        total: int,
        pose: Pose,
        accepted: int,
        min_samples: int | None = None,
        *,
        station: Station | None = None,
    ) -> None:
        # Build the event payload directly from the vendor-neutral
        # :class:`Pose` (millimetres + XYZW quaternion). The wire shape
        # matches the URPose-derived layout the event consumer reads:
        # ``x/y/z`` in millimetres, ``rx/ry/rz`` as the axis-angle
        # rotation vector in radians. For a joint station ``pose`` is the
        # grasp centre its joints put the tool at, and the joints ride along
        # in degrees, with the joints that turn more than half a turn from
        # where the arm stands.
        axis_angle = pose.axis_angle_rad()
        position = pose.position_mm
        payload: dict[str, Any] = {
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
            "min_samples": min_samples,
        }
        if isinstance(station, JointStation):
            payload["joints_deg"] = [round(value, 3) for value in station.degrees]
            hop = self._hop_from_here(station)
            if hop:
                self.logger.warning("Pose %d/%d '%s': from where the arm stands, %s.", idx, total,
                                    station.label or "", hop)
                payload["hop"] = hop
        self._emit(RobotCalibrationEvent.MOVING_TO_POSE, payload)

    def _hop_from_here(self, station: JointStation) -> str:
        """The joints that turn more than half a turn from where the arm stands to ``station``; ``""`` for none or
        where the arm cannot say. A warning only: the station runs as written."""
        try:
            here = self.arm.get_joint_positions()
        except Exception:  # noqa: BLE001 (a warning's input; the move reads its own start)
            return ""
        if not isinstance(here, JointPositions) or here.dof != station.joints.dof:
            return ""
        hop = joint_hop(here, station.joints)
        return f"{hop}, more than half a turn; it runs as written, the long way round" if hop else ""

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
