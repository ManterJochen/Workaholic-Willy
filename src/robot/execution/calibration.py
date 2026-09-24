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

Stations run in the order they were given. Nothing here generates one:
every station is one somebody wrote down, taught on the pendant, or
guided the arm to by hand.

Hand guiding
------------
Two flows hand the arm to a person, on an arm that offers it
(:class:`~src.robot.core.freedrive.SupportsFreedrive`, the UR teach mode);
an arm without it has fixed stations only, and asking for either refuses
with :class:`~src.robot.execution.hand_guiding.HandGuidingRefused` before
anything moves.

* **Adjust** (``run_with_poses(adjust=True)``, ``run_from_json(adjust=True)``):
  at each fixed station the arm reached, it is freed so a person can
  fine-tune the view by hand; Enter (after the stillness gate) holds it
  where it stands and the station is judged there. Before the next
  automatic move the console asks for the hands off the arm and counts
  down three seconds.
* **Freedrive** (:meth:`CalibrationRoutine.run_freedrive`): nothing moves
  by itself. The person moves the arm to a pose where the camera sees the
  board and presses Enter; the arm is held, settled, its TCP read and one
  frame judged, and it is freed again, until ``samples`` counted or the
  person finishes. Given stations, each is a target the preview shows the
  way to, in millimetres and degrees along the tool's own axes.

Both show the controller payload and ask whether it is right before the
first free, sample the free arm at 50 Hz (which feeds the vendor's crash
watchdog), and show a sample outside the cable window or the workspace
box in red, where Enter does not capture; a boundary never holds or stops
the arm. Each station counted by hand is written, joints first, to a
stations file :func:`~src.robot.execution.pose_provider.load_stations`
reads back, so the sweep can be run again without hands
(``run_from_json``). The file a run reads its stations from is never the
file it writes (:func:`stations_file_to_write`): a replay with ``adjust``
that names its own file to save to writes ``<name>.adjusted.json`` beside
it, so a replay finished partway keeps every station it was given.

Once a person has guided the arm, the routine's guide asks for the hands
off it and counts down before the next move the routine commands, adjust
or not: a replay run on a routine that has just been guided by hand does
not drive before that.

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

import math
import os
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
    MotionStatus,
    RobotArm,
    RobotEmergencyStop,
)
from src.robot.core.camera_world import without_camera_world
from src.robot.core.freedrive import FreedriveSample, SupportsFreedrive
from src.robot.events import RobotCalibrationEvent, RobotCalibrationEventListener
from src.robot.execution.hand_guiding import (
    CAPTURE,
    FINISH,
    HandGuide,
    HandGuidingLimits,
    HandGuidingRefused,
    offset_in_tool,
    sample_pose,
)
from src.robot.execution.pose_provider import (
    JOINT_NAMES,
    JointStation,
    PoseProvider,
    Station,
    StationScreen,
    load_stations,
    station_record,
    write_stations,
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
    * ``move_rejected``: the arm refused or failed the move;
    * ``not_captured``: an adjusted station the person skipped (``s``) or finished the sweep at (``q``), so no
      frame was taken;
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
        Each pose the sweep visited, in order: counted, or why not. For a
        hand-guided run, each capture the person asked for.
    stations_path : str or None
        The stations file a run guided by hand wrote, one joint station per
        counted pose, which ``run_from_json`` replays without hands.
        ``None`` for a run that wrote none.
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
    stations_path: str | None = None


#: How many counted samples a hand-guided run collects unless its caller says otherwise
#: (``robot.calibration.freedrive_samples`` in a tree).
DEFAULT_FREEDRIVE_SAMPLES = 15


def stations_file_to_write(read_from: str | Path | None, save_to: str | Path | None) -> str | None:
    """Where a run writes the stations it counts by hand: ``save_to``, unless that is the file ``read_from`` names.

    A run begins its stations file afresh and writes it after every pose it counts, so a run that wrote into the
    file its own stations came from would leave, when finished partway (``q``), one station of the many it was
    given. That file is never written over: the stations go to ``<name>.adjusted<suffix>`` beside it instead
    (``eye_in_hand_wrist_stations.json`` gives ``eye_in_hand_wrist_stations.adjusted.json``). The same file is
    the same whichever way it is spelt: relative or absolute, through ``..``, a link, or in another case where
    the file system ignores case. ``None`` when there is nowhere to write.
    """
    if save_to is None:
        return None
    if read_from is None or not _same_file(read_from, save_to):
        return str(save_to)
    path = Path(save_to)
    beside = path.with_name(f"{path.stem}.adjusted{path.suffix}")
    while _same_file(read_from, beside):  # a link named like the one beside it back to the file read
        beside = beside.with_name(f"{beside.stem}.adjusted{beside.suffix}")
    return str(beside)


def _same_file(one: str | Path, other: str | Path) -> bool:
    """Whether two paths name one file: the file system's answer when both exist, else their absolute spellings."""
    try:
        return os.path.samefile(one, other)
    except OSError:
        return os.path.normcase(os.path.abspath(one)) == os.path.normcase(os.path.abspath(other))


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
    hand_guide : HandGuide, keyword only
        The console side of hand guiding (the adjust step and
        :meth:`run_freedrive`): where the person reads and types, the preview
        it shows in, and the clock it runs on. Left unset, the terminal and
        no window.
    images_dir : str or Path, keyword only
        Where each counted sample's judged frame is written as a PNG, when the
        marker source keeps it (``last_frame``). ``None`` writes none.
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
    #: The stations file the last run guided by hand wrote, ``None`` before one; kept here as well as on the result
    #: for a solve that raises, because the stations are what makes the run repeatable without hands.
    stations_path: str | None = None
    #: The console side of hand guiding; ``__init__`` sets it, and a routine built without ``__init__`` has none.
    hand_guide: HandGuide | None = None
    _images_dir: Path | None = None
    _stations: list[dict[str, Any]] = []  # noqa: RUF012 (rebound whole by every run, never mutated)

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
        hand_guide: HandGuide | None = None,
        images_dir: str | Path | None = None,
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
        self.hand_guide = hand_guide
        self._images_dir = Path(images_dir) if images_dir is not None else None

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
        *,
        adjust: bool = False,
        stations_save_path: str | Path | None = None,
    ) -> CalibrationResult:
        """Run a calibration using pre-planned stations from a JSON file, in the order it lists them.

        The records may be poses or joint stations (:mod:`pose_provider`), and every record is checked
        before anything moves. Each station is screened as the sweep reaches it: one whose grasp centre is
        outside the workspace box, or too close to a station kept before it, is reported with its label and
        reason (``outside_workspace``, ``too_similar``) and never moved to. ``adjust`` and
        ``stations_save_path`` are :meth:`run_with_poses`'s, except that ``poses_path`` is never written over: a
        ``stations_save_path`` naming that same file writes ``<name>.adjusted.json`` beside it instead, says so
        on the guide's console and in the log, and the result's ``stations_path`` names where the stations went.
        """
        stations = load_stations(poses_path)
        self.logger.info("Loaded %d stations from %s.", len(stations), poses_path)
        # The file read is never the file written: a run begins its stations file afresh and writes it after every
        # pose counted, so a replay finished partway would leave one station of the many it was given.
        save_to = stations_file_to_write(poses_path, stations_save_path)
        if stations_save_path is not None and save_to != str(stations_save_path):
            said = (f"{stations_save_path} is the file the stations were read from, so it is not written over: the "
                    f"stations counted by hand go to {save_to}")
            self.logger.warning("%s.", said)
            if self.hand_guide is not None:
                self.hand_guide.console.say(f"{said}.")
        return self._execute(stations, T_tool_to_marker, dataset_save_path, screen_all=True, adjust=adjust,
                             stations_save_path=save_to)

    def run_with_poses(
        self,
        poses: Sequence[Station],
        T_tool_to_marker: np.ndarray | None = None,
        dataset_save_path: str | None = None,
        *,
        adjust: bool = False,
        stations_save_path: str | Path | None = None,
    ) -> CalibrationResult:
        """Run a calibration using an explicit, caller-supplied list of stations, in order.

        Move, settle, read the TCP, capture the marker, add the sample, then AX=XB, as ``run_from_json``. A
        station is a :class:`Pose`, which the arm's own gates box when it moves, or a :class:`JointStation`,
        which is screened first as a JSON file's stations are.

        ``adjust`` frees the arm at each station it reached so a person can fine-tune the view by hand (see
        the module's *Hand guiding*): Enter holds it where it stands and the station is judged there, and
        before the next automatic move the console asks for the hands off the arm and counts down. It needs
        an arm that offers hand guiding and refuses, before anything moves, on one that does not. Each station
        counted by hand is written to ``stations_save_path`` when given, as a joint station.
        """
        return self._execute(list(poses), T_tool_to_marker, dataset_save_path, adjust=adjust,
                             stations_save_path=stations_save_path)

    def run_freedrive(
        self,
        stations: Sequence[Station] | None = None,
        samples: int = DEFAULT_FREEDRIVE_SAMPLES,
        dataset_save_path: str | None = None,
        *,
        stations_save_path: str | Path | None = None,
    ) -> CalibrationResult:
        """A calibration a person guides by hand: nothing moves by itself.

        Before the arm is freed, the controller payload is shown and has to be confirmed. Then, until
        ``samples`` counted or the person finishes (``q``, ESC or closing the preview), the console and the
        preview ask for a new pose where the camera sees the board; Enter from either, once the arm stood
        still for half a second, holds the arm, waits for it to settle, reads its TCP, judges one frame and
        frees it again. Each capture is on the events and the pose log like a station of a sweep. A pose
        outside the cable window or the workspace box is shown in red and not captured; the arm is never held
        for it.

        ``stations``, when given, are targets and are never moved to: the preview shows the way to the current
        one along the tool's axes (a joint station through the arm's forward kinematics, else joint by joint),
        and it moves on after each counted capture or on ``s``. Every counted pose is written to
        ``stations_save_path`` as it is taken, a stations file ``run_from_json`` replays. The dataset is saved
        after every counted pose as well, so a run that is stopped keeps what it counted.
        """
        arm = self._freedrive_arm("a hand-guided calibration")
        guide = self._guide()
        targets = list(stations or ())
        need = self._min_samples()
        wanted = max(1, int(samples))
        self._begin(stations_save_path)
        self.logger.info("Hand-guided calibration: %d samples wanted, %d target stations (marker_id=%d).", wanted,
                         len(targets), self.marker_id)
        guide.confirm_payload(arm)
        limits = HandGuidingLimits.of(arm, self.guard.limits)
        guide.console.say(f"Watched while you guide the arm: {limits.window}.")
        accepted = attempts = 0
        target = 0
        stopped: SweepStopped | None = None
        with arm.freedrive() as session:
            session.free()
            while accepted < wanted:
                aim = targets[target] if target < len(targets) else None
                where = (f"to target {aim.label or target + 1!r} ({target + 1} of {len(targets)})" if aim is not None
                         else "to a new pose where the camera sees the board")
                choice = guide.wait(session, limits, banner=f"MOVE THE ARM BY HAND {where}, then Enter",
                                    status=self._guidance(aim, accepted, wanted))
                if choice == FINISH:
                    break
                if choice != CAPTURE:
                    if aim is not None:
                        target += 1
                    continue
                attempts += 1
                label = aim.label if aim is not None and aim.label else f"hand_{attempts:02d}"
                session.hold()
                try:
                    tcp = self._judge_here(attempts, wanted, label, accepted, need)
                except (RuntimeError, OSError, ValueError) as exc:
                    self.logger.error("Capture %d '%s' raised: %s", attempts, label, exc)
                    why = self._stopped_state()
                    if why is not None:
                        stopped = self._stop(attempts, wanted, label, "exception", str(exc), why)
                        break
                    self._reject(attempts, wanted, label, "exception", str(exc))
                    tcp = None
                if tcp is not None:
                    accepted += 1
                    self._record_station(label, tcp)
                    if dataset_save_path:
                        self.calibrator.save_dataset(dataset_save_path)
                    if aim is not None:
                        target += 1
                guide.console.say(f"{accepted} of {wanted} counted.")
                session.free()
        return self._finish(accepted, attempts, dataset_save_path, stopped)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _execute(
        self,
        poses: Sequence[Station],
        T_tool_to_marker: np.ndarray | None,
        dataset_save_path: str | None,
        *,
        screen_all: bool = False,
        adjust: bool = False,
        stations_save_path: str | Path | None = None,
    ) -> CalibrationResult:
        """Visit each station in order, then solve. ``screen_all`` screens poses too, not only joint stations.

        With ``adjust`` the arm is refused before anything moves unless it offers hand guiding, the payload is
        confirmed before the first move, and each station the arm reached is adjusted by hand
        (:meth:`_adjusted`) before it is judged.

        Before every move, adjust or not, the routine's hand guide (when it has one) asks for the hands off the
        arm and counts down if a person has guided the arm since the last countdown (``HandGuide.touched``): a
        replay run on a routine that just ran :meth:`run_freedrive` drives only after that, and ``q`` there
        moves nothing. A guide nobody touched asks nothing.
        """
        self.logger.info(
            "Starting calibration with %d poses (marker_id=%d).",
            len(poses), self.marker_id,
        )

        guide: HandGuide | None = self.hand_guide
        if adjust:
            arm = self._freedrive_arm("--adjust, which frees the arm at each station,")
            guide = self._guide()
            guide.confirm_payload(arm)
        accepted = 0
        stations: list[Station] = list(poses)
        need = self._min_samples()
        self._begin(stations_save_path)
        screen = self.pose_provider.screen()
        stopped: SweepStopped | None = None
        total = len(stations)
        for idx, station in enumerate(stations, start=1):
            if not isinstance(station, (Pose, JointStation)):
                raise TypeError(
                    f"CalibrationRoutine requires Pose or JointStation targets; got {type(station).__name__}."
                )
            label = station.label or ""
            self.logger.info("=== Pose %d/%d '%s' ===", idx, total, station.label)
            target = self._screened(idx, total, station, screen, screen_all=screen_all)
            if target is None:
                continue
            if guide is not None and not guide.hands_off():
                self.logger.info("The person finished the sweep before pose %d/%d '%s'.", idx, total, label)
                break
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
                    continue

                finished = False
                if guide is None or not adjust:
                    counted = self._judge_here(idx, total, label, accepted, need)
                else:
                    finished, counted = self._adjusted(guide, idx, total, label, accepted, need)
                    if counted is not None:
                        self._record_station(label, counted)
                if counted is not None:
                    accepted += 1
                if finished:
                    break

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

        return self._finish(accepted, total, dataset_save_path, stopped)

    def _begin(self, stations_save_path: str | Path | None) -> None:
        """A run's fresh state: no stamps, no verdicts, no stations counted by hand yet."""
        self._camera_worlds = ()
        self.pose_log = ()
        self._stations = []
        self.stations_path = str(stations_save_path) if stations_save_path is not None else None

    @property
    def stations_written(self) -> str | None:
        """The stations file this run wrote, once it recorded at least one station there; ``None`` before.

        :attr:`stations_path` is where the run would write, set when it begins; a file already standing there may be
        an earlier run's. This names it only for stations of this run, so a run that failed before it counted
        anything never hands on an old file as its own.
        """
        return self.stations_path if self._stations else None

    def _finish(self, accepted: int, total: int, dataset_save_path: str | None,
                stopped: SweepStopped | None) -> CalibrationResult:
        """Save the dataset, raise a stop once it is saved, then solve and build the result."""
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
        if self.stations_written is not None:
            self.logger.info("%d stations counted by hand written to %s.", len(self._stations), self.stations_written)
        if stopped is not None:
            raise stopped

        result = self.calibrator.calibrate()
        if result.mode is not self.calibration_mode:
            raise RuntimeError(
                f"Calibrator returned {result.mode.value}, expected "
                f"{self.calibration_mode.value}."
            )
        transform = result.transform
        stations_path = self.stations_written

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
                stations_path=stations_path,
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
            stations_path=stations_path,
        )

    def _judge_here(self, idx: int, total: int, label: str, accepted: int, need: int | None) -> Pose | None:
        """Judge the station the arm stands at: settle, read the TCP, take one frame, hand the pair to the calibrator.

        Returns the TCP the sample was taken at when it counted, ``None`` when it did not, and every verdict is
        on the pose log and the events. Raises what the arm or the camera raise, for the caller's recovery.
        """
        if not self._wait_for_settle():
            self._reject(idx, total, label, "not_steady",
                         f"the arm did not report steady within {self.settle_time_s:g} s, twice, after the "
                         "move, so no frame was taken")
            return None

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
            return None

        seen = self._seen(T_cam_to_marker, observation)
        if seen["hint"]:
            self.logger.warning("Pose %d/%d '%s': %s", idx, total, label, seen["hint"])
        self._emit(RobotCalibrationEvent.MARKER_DETECTED, {
            "index": idx, "total": total, "label": label, "marker_id": self.marker_id, **seen,
        })

        added = self.calibrator.add_sample(
            T_base_to_tool, T_cam_to_marker, marker_id=self.marker_id,
        )
        if not added:
            rejection = getattr(self.calibrator, "last_rejection", None)
            detail = rejection.render() if rejection is not None else "the calibrator refused the sample"
            self.logger.warning("Sample rejected by calibrator: %s.", detail)
            extra = {"rejection": rejection.to_dict()} if rejection is not None else {}
            self._reject(idx, total, label, "sample_rejected", detail, **extra)
            return None
        self.guard.accept(tcp_pose)
        self.logger.info("Sample %d accepted.", accepted + 1)
        self.pose_log = (*self.pose_log, PoseVerdict(idx, label, True, detail=seen["summary"]))
        self._emit(RobotCalibrationEvent.POSE_ACCEPTED, {
            "index": idx, "total": total, "label": label, "accepted": accepted + 1,
            "min_samples": need, "detail": seen["summary"],
        })
        self._keep_image(idx, label)
        return tcp_pose

    def _adjusted(self, guide: HandGuide, idx: int, total: int, label: str, accepted: int,
                  need: int | None) -> tuple[bool, Pose | None]:
        """The adjust step at a station the arm reached: ``(finished, the TCP it counted at or None)``.

        The arm is freed for the person to fine-tune the view, and held on Enter once it stood still; the
        station is judged there, inside the session. ``s`` leaves the station uncounted and ``q`` finishes the
        sweep here, each as ``not_captured``. Leaving the session holds the arm and gives motion back.
        """
        arm = self._freedrive_arm("--adjust")
        limits = HandGuidingLimits.of(arm, self.guard.limits)
        with arm.freedrive() as session:
            session.free()
            choice = guide.wait(session, limits,
                                banner=f"ADJUST pose {idx}/{total} {label!r} BY HAND until the view is right, then Enter",
                                status=self._guidance(None, accepted, total))
            session.hold()
            if choice != CAPTURE:
                said = "finished the sweep here (q)" if choice == FINISH else "skipped this station (s)"
                self._reject(idx, total, label, "not_captured", f"the person {said}, so no frame was taken")
                return choice == FINISH, None
            return False, self._judge_here(idx, total, label, accepted, need)

    # ------------------------------------------------------------------
    # Hand guiding helpers
    # ------------------------------------------------------------------

    def _freedrive_arm(self, what: str) -> SupportsFreedrive:
        """The arm as one a person can guide, or :class:`HandGuidingRefused` before anything moves."""
        if not isinstance(self.arm, SupportsFreedrive):
            raise HandGuidingRefused(
                f"{what} needs an arm a person can guide by hand, and {type(self.arm).__name__} offers no hand "
                "guiding (SupportsFreedrive; the UR teach mode is one). Run the fixed stations without it; nothing "
                "moved")
        return self.arm

    def _guide(self) -> HandGuide:
        """The console side of hand guiding: the one given, else the terminal with no window."""
        if self.hand_guide is None:
            self.hand_guide = HandGuide()
        return self.hand_guide

    def _guidance(self, aim: Station | None, accepted: int, wanted: int
                  ) -> Callable[[FreedriveSample], tuple[list[str], float | None]]:
        """What the preview shows beside a free arm, from each sample: the way to ``aim``, the nearest counted pose
        and the count. The target's pose is placed once, not per sample."""
        target: Pose | None = None
        if isinstance(aim, JointStation):
            target, _ = self._grasp_centre_at(aim)
        elif isinstance(aim, Pose):
            target = aim
        need = self._min_samples()
        count = f"{accepted} of {wanted} counted" + (f", the solve needs {need}" if need else "")
        name = repr(aim.label) if aim is not None and aim.label else "the target"

        def lines(sample: FreedriveSample) -> tuple[list[str], float | None]:
            said: list[str] = []
            bar: float | None = None
            if target is not None:
                offset = offset_in_tool(sample, target)
                said.append(f"to {name}: {offset.line()}")
                bar = offset.closeness
            elif isinstance(aim, JointStation):
                turns = [f"{joint} {math.degrees(float(want) - float(have)):+.0f}" for joint, have, want
                         in zip(JOINT_NAMES, sample.joints_rad, aim.joints.values)]
                said.append(f"to {name}, turn (deg): {', '.join(turns)}")
            said += [self._nearest_counted(sample), count]
            return said, bar

        return lines

    def _nearest_counted(self, sample: FreedriveSample) -> str:
        """Whether the TCP stands far enough from every pose counted so far to count, by the calibrator's own rule
        (a stored pose within both ``min_distance_mm`` and ``min_angle_deg`` blocks it), and how far the nearest is."""
        now = sample_pose(sample).to_matrix()
        dataset = getattr(self.calibrator, "dataset", None)
        settings = getattr(self.calibrator, "settings", None)
        min_mm = float(getattr(settings, "min_distance_mm", 0.0) or 0.0)
        min_deg = float(getattr(settings, "min_angle_deg", 0.0) or 0.0)
        nearest: tuple[float, float] | None = None
        blocking: tuple[float, float] | None = None
        for stored in (dataset.iter_samples() if dataset is not None else ()):
            before = np.asarray(stored.T_base_to_tool, dtype=np.float64)
            distance = float(np.linalg.norm(before[:3, 3] - now[:3, 3]))
            cos = (float(np.trace(before[:3, :3].T @ now[:3, :3])) - 1.0) / 2.0
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            if nearest is None or distance < nearest[0]:
                nearest = (distance, angle)
            if distance <= min_mm and angle <= min_deg and (blocking is None or distance < blocking[0]):
                blocking = (distance, angle)
        if nearest is None:
            return "no pose counted yet: any view of the board is a new one"
        if blocking is not None:
            return (f"too close to a counted pose ({blocking[0]:.0f} mm, {blocking[1]:.0f} deg away): move "
                    f"{min_mm:g} mm or turn {min_deg:g} deg from it")
        return f"a new view: the nearest counted pose is {nearest[0]:.0f} mm, {nearest[1]:.0f} deg away"

    def _record_station(self, label: str, tcp: Pose) -> None:
        """Write a pose counted by hand to the stations file, joints first; a joint read that fails is logged only."""
        if self.stations_path is None:
            return
        try:
            joints = self.arm.get_joint_positions()
        except Exception:  # noqa: BLE001 (the sample counted; only its replay record is lost, and said so)
            self.logger.exception("The joints of '%s' could not be read; it is not in the stations file.", label)
            return
        self._stations = [*self._stations, station_record(label, joints, tcp)]
        write_stations(self.stations_path, self._stations)

    def _keep_image(self, idx: int, label: str) -> None:
        """Write the judged frame of a counted sample as a PNG, when there is a folder and the source kept it."""
        frame = getattr(self._marker_source, "last_frame", None)
        if self._images_dir is None or frame is None:
            return
        try:
            import cv2

            self._images_dir.mkdir(parents=True, exist_ok=True)
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", label) or "pose"
            path = self._images_dir / f"{idx:02d}_{name}.png"
            if not cv2.imwrite(str(path), np.asarray(frame)):
                raise OSError(f"cv2.imwrite refused {path}")
        except Exception:  # noqa: BLE001 (the sample counted; its picture is a record, not a gate)
            self.logger.exception("The frame of pose %d '%s' was not written.", idx, label)

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
        # in degrees.
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
        self._emit(RobotCalibrationEvent.MOVING_TO_POSE, payload)

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
