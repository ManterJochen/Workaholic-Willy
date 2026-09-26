"""Hand-eye calibration of one camera against the robot, as one noun with two verbs.

    from src.config import ConfigTree
    from src.robot.execution.hand_eye import HandEyeCalibration, SweepOptions

    loaded = ConfigTree.from_directory().load()
    calibration = HandEyeCalibration.from_tree(loaded, rig_id="overhead", mode="eye_to_hand",
                                               options=SweepOptions(freedrive=True, marker_length_mm=39.7))
    print(calibration.check().render())        # the config alone: builds nothing, opens nothing
    print(calibration.run(dry_run=True).render())   # builds the arm, opens the camera, moves nothing
    report = calibration.run()                 # this moves the robot
    print(report.render())                     # ends with the rig block to paste
    raise SystemExit(report.exit_code)

A correct sweep needs the arm built alone through ``Robot.from_config(robot_config, gripper=None)`` so the
readiness gate runs, one camera through its ``Camera`` owner, the cell lock through ``Robot.connected()``, every wrist
body the tree declares handed to the arm before it moves, whichever camera is swept, the flange to TCP record read
while the arm is connected, the tree's quality bands, and the rig block to paste. This class is that flow, so a
Python caller builds it rather than copying it, and ``python -m src.robot.execution.real_cell.calibrate`` is a caller
of it.

What happens, in order:

1. ``check()`` reads the config only: the rig (configured, switched on, RGB-D), for an eye in hand sweep the
   camera's body, for either mounting the body of every other wrist camera the tree declares (the arm carries it
   whichever camera is swept; a tree that declares none reads nothing, and a wrist rig declared without a body is
   named on one warning line and refuses nothing), where the stations come from (a file or a list of the caller's
   own, ``fixed_poses``, or a person guiding the arm, ``freedrive``; there is no generated sweep), and on a cuRobo UR
   cell that declares its planner margin, the committed planner evidence the sweep's planner starts on. A refusal is
   a report, never an exception.
2. ``run()`` checks again, then builds: the arm alone and one camera, opened. What the arm refuses (its safety
   attestation) and what world it plans against are on the build report, and ``on_built`` receives it before any
   motion, so a caller can show it while there is still time to stop. A run guided by hand (``freedrive``, or
   ``adjust`` at fixed stations) is refused here on an arm that offers no hand guiding (``SupportsFreedrive``).
   ``dry_run=True`` stops here and gives the camera back.
3. The sweep connects through ``Robot.connected()``: the cell lock first, then the arm, and no gripper. Every move
   declines the camera world for itself (``CalibrationRoutine``, one reason per mounting), because the sweep is what
   produces the transform a camera world needs. On the way out the arm comes down, the lock is given back, and then
   the camera. With ``SweepOptions(preview=...)`` a window beside the sweep shows the camera and each judged frame
   (``src/calibration/preview.py``); it opens once the arm is connected, closes before the camera is given back, and
   draws only. While a person guides the arm it also shows the way to the next target and the boundaries, and
   takes Enter, ``s`` and ``q`` as the console does (``hand_guiding``).
4. The solve's carrier is written into ``out_dir``: ``eth_<rig>.json`` (CAMERA to BASE) or ``eih_<rig>.json``
   (CAMERA to TOOL, with the flange to TCP record when the tool frame is declared ``willy`` or ``polyscope``). The
   dataset is written during the sweep, before the solve, so a solve that fails still leaves the samples, and each
   counted sample's frame lands in ``<mode>_<rig>_images``. A run guided by hand also writes
   ``<mode>_<rig>_stations.json``, one joint station per counted pose, which ``fixed_poses`` replays without hands;
   when that is the very file ``fixed_poses`` names, it is never written over, and the stations go to
   ``<mode>_<rig>_stations.adjusted.json`` beside it.

Writing is part of ``run()`` and not a separate ``save()``: an artifact a caller forgot to save is the silence the
CLI's exit code 2 exists to prevent, and the report says what was written and where.

The routine and the solve are exercised in simulation only, and nothing here has run against a physical
controller.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

from src.calibration.eye_hand import MountingMode
from src.contracts import UNSET, Maybe, chosen, resolve

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Sequence

    from src.config.tree import LoadedTree
    from src.calibration.serialization import FlangeToTcp
    from src.camera.orchestration.camera import Camera
    from src.geometry import Pose
    from src.robot.core.camera_world import CameraWorldStamp
    from src.robot.events import RobotCalibrationEventListener
    from src.robot.execution.calibration import PoseVerdict
    from src.robot.execution.hand_guiding import OperatorConsole
    from src.robot.execution.lifecycle import ConnectStage, TeardownReport
    from src.robot.execution.pose_provider import JointStation
    from src.robot.execution.robot import Robot
    from src.robot.safety import SafetyAttestation

__all__ = [
    "DEFAULT_OUT_DIR",
    "CalibrationBuild",
    "CalibrationCheck",
    "CalibrationOutcome",
    "CalibrationRunReport",
    "CalibrationStage",
    "HandEyeCalibration",
    "SweepOptions",
    "print_sweep_progress",
    "render_sweep_event",
]

logger = logging.getLogger(__name__)

#: Where the artifact and the dataset land unless the options say otherwise. The sim runners write the same layout,
#: so a cell brought up in sim and then on hardware keeps one place to look.
DEFAULT_OUT_DIR = "calibration/real"
#: The ArUco id a sweep poses unless the options or the mode's ``camera.hand_eye`` block say otherwise.
DEFAULT_MARKER_ID = 0

_BANNERS = {
    "config": "=== 1. CONFIG ===",
    "build": "=== 2. BUILD === the arm alone, and one camera",
    "sweep": "=== 3. SWEEP === the robot moves now, keep hands clear",
    "result": "=== 4. RESULT ===",
}
#: The sweep banner of a run nothing moves in by itself.
_BY_HAND_BANNER = "=== 3. SWEEP === guided by hand: the arm moves only when you move it"


class CalibrationStage(StrEnum):
    """The four stages of a run, in order. ``banner()`` is the one wording the CLI and ``render()`` both print."""

    CONFIG = "config"
    BUILD = "build"
    SWEEP = "sweep"
    RESULT = "result"

    def banner(self) -> str:
        """The stage's heading line, ASCII."""
        return _BANNERS[self.value]


class CalibrationOutcome(StrEnum):
    """How far a run got, and why it stopped there."""

    #: The config refused the sweep. Nothing was built.
    REFUSED = "refused"
    #: The arm, the camera or the wrist body refused. Nothing moved.
    BUILD_REFUSED = "build_refused"
    #: Built, attested and given back. Nothing moved.
    DRY_RUN = "dry_run"
    #: Another process holds the controller. Nothing was commanded.
    CELL_BUSY = "cell_busy"
    #: The connect refused and rolled itself back.
    CONNECT_FAILED = "connect_failed"
    #: The sweep raised. Reported once the arm is down. A run guided by hand whose payload was not confirmed ends here
    #: too, as ``HandGuidingRefused``, with nothing freed.
    SWEEP_FAILED = "sweep_failed"
    #: The solve produced no carrier. Nothing was written, and the camera keeps its previous calibration.
    NO_ARTIFACT = "no_artifact"
    #: The artifact is written and the rig block is ready to paste.
    WRITTEN = "written"


#: The CLI's four codes: 0 done, 1 refused before the sweep, 2 no artifact, 3 the sweep raised.
_EXIT_CODES = {
    CalibrationOutcome.REFUSED: 1,
    CalibrationOutcome.BUILD_REFUSED: 1,
    CalibrationOutcome.DRY_RUN: 0,
    CalibrationOutcome.CELL_BUSY: 1,
    CalibrationOutcome.CONNECT_FAILED: 1,
    CalibrationOutcome.SWEEP_FAILED: 3,
    CalibrationOutcome.NO_ARTIFACT: 2,
    CalibrationOutcome.WRITTEN: 0,
}
#: The outcomes a run reaches only after it announced the sweep.
_SWEPT = frozenset({
    CalibrationOutcome.CELL_BUSY,
    CalibrationOutcome.CONNECT_FAILED,
    CalibrationOutcome.SWEEP_FAILED,
    CalibrationOutcome.NO_ARTIFACT,
    CalibrationOutcome.WRITTEN,
})


@dataclass(frozen=True, slots=True)
class SweepOptions:
    """What a caller may choose about one sweep. A field left ``UNSET`` takes the tree's value or the code default.

    ``marker_length_mm``, ``marker_id`` and ``dict_name`` default to the mode's ``camera.hand_eye`` block (its
    ``target`` when that names one marker). A wrong marker length scales every sample uniformly: the solve converges
    and is uniformly wrong, so measure the printed board.
    ``target`` names the whole target instead: a board spec as ``--board`` takes it (``"aruco:ID:SIZE_MM[:DICT]"``,
    ``"charuco:XxY:SQUARE_MM:MARKER_MM[:DICT][:legacy]"``), a ``Path`` to a YAML or JSON file of one target mapping,
    a mapping, or an ``ArucoTargetConfig`` / ``CharucoTargetConfig``. It is validated by the config schema at
    ``check()``, and it cannot be combined with the three single-marker options. A dictionary it leaves out is the
    block's.
    ``unmodelled_wrist_body`` is the reason a sweep may run while a wrist camera's declared body cannot be placed yet
    (no calibration, one that does not load, no record or a stale one): the camera an eye in hand sweep calibrates,
    and any wrist camera the tree declares on the arm, for either mounting. It is printed and logged, and the sweep
    then runs with no body for that camera in the planner and the guard. It is read only for such a body: a wrist
    rig declared with no body at all, the one an eye in hand sweep calibrates included, needs none, and is named on
    a warning line instead.

    Where the stations come from, one of two ways; nothing is generated, and a sweep that names neither is refused at
    ``check()``:

    ``fixed_poses`` are the caller's own stations, run in the order given: a list of ``Pose`` (BASE) the caller
    already planned and ``JointStation``, or a path to a JSON file of them (URPose-shaped or ``Pose``-shaped records,
    and joint stations ``{"joints_deg": [...]}`` or ``{"joints_rad": [...]}``, six values from the base to the last
    wrist joint; see ``pose_provider``). ``check()`` reads and checks the file and counts its stations and its joint
    stations. A joint station runs as one judged joint move, and before it moves the grasp centre the arm's forward
    kinematics puts at its joints must lie inside the workspace box and differ enough from the stations before it;
    so must every pose of a file. A station that does not is reported with its reason and never moved to.
    ``adjust`` frees the arm at each fixed station it reached so a person can fine-tune the view by hand, captures on
    Enter, and asks for the hands off the arm and counts down before the next automatic move.

    ``freedrive`` lets a person guide the arm to each pose instead: nothing moves by itself. The console and the
    preview ask for a pose where the camera sees the board, Enter captures once the arm stands still, and the run ends
    at ``samples`` counted (the tree's ``robot.calibration.freedrive_samples``, 15) or when the person finishes. With
    ``fixed_poses`` beside it, those stations are targets the preview shows the way to and are never moved to.

    Both ways by hand need an arm that offers hand guiding and are refused at the build on one that does not; both
    show the controller payload and ask whether it is right before the arm is first freed; and both write each pose
    counted by hand to ``<mode>_<rig>_stations.json`` in ``out_dir``, which ``fixed_poses`` replays without hands.
    The file ``fixed_poses`` names is never written over: when it is that file, the stations go to
    ``<mode>_<rig>_stations.adjusted.json`` beside it, and the console says so before anything moves.

    ``preview`` opens a window beside the sweep: the camera's view while the arm moves, each judged frame with the
    target drawn on it, and whether its pose counted and why not. ``True`` opens it wherever a window can show, and
    the build says why when it cannot. ``"auto"`` opens it only where a window can show and stdout is a terminal, and
    says nothing otherwise, so a piped or captured run prints what it printed before. ``False`` or unset opens none.
    ``WILLY_NO_PREVIEW`` set keeps it off whatever this says. It draws only: closing it closes the window (and
    finishes a run guided by hand, which holds the arm), and the pendant stops the robot.
    """

    marker_length_mm: Maybe[float] = UNSET
    marker_id: Maybe[int] = UNSET
    dict_name: Maybe[str] = UNSET
    out_dir: "Maybe[str | Path]" = UNSET
    unmodelled_wrist_body: Maybe[str] = UNSET
    fixed_poses: "Maybe[Sequence[Pose | JointStation] | str | Path]" = UNSET
    target: Maybe[Any] = UNSET
    preview: "Maybe[bool | Literal['auto']]" = UNSET
    freedrive: Maybe[bool] = UNSET
    adjust: Maybe[bool] = UNSET
    samples: Maybe[int] = UNSET


@dataclass(frozen=True, slots=True)
class CalibrationCheck:
    """What the config says about a sweep, decided with nothing built and nothing opened."""

    rig_id: str
    mode: MountingMode
    #: Why the sweep is refused, verbatim. Empty when it may run.
    refusal: str = ""
    rig_source: str = ""
    vendor: str = ""
    marker_length_mm: float = 0.0
    marker_id: int = 0
    dict_name: str = ""
    #: How many stations the sweep visits, or for a run guided throughout by hand how many samples it collects.
    poses: int = 0
    #: Where the artifact will be written.
    artifact_path: str = ""
    #: A target that is not one marker, in one line (``charuco 7x5, square 30.0 mm, ...``). Empty for a marker, which
    #: the ``marker`` line describes.
    target: str = ""
    #: The file the caller's own stations were read from. Empty for a list and for none.
    poses_file: str = ""
    #: How many of the caller's own stations are joint stations.
    joint_stations: int = 0
    #: ``"freedrive"`` for a run guided throughout by hand, ``"adjust"`` for fixed stations adjusted by hand, else empty.
    by_hand: str = ""
    #: For a run guided throughout by hand, how many of the caller's stations it shows the way to.
    targets: int = 0
    #: The wrist cameras the arm carries during the sweep, as ``WristBodies.render()`` says it: carried, or moved
    #: without and why. Every wrist camera the tree declares, and beside them the one an eye in hand sweep calibrates.
    #: Empty where the tree declares no wrist camera other than that one, on a cell that reads no geometry, and for a
    #: camera handed in without its section.
    wrist: str = ""
    #: The warning for the wrist rigs the tree declares on the arm without a body, the one an eye in hand sweep
    #: calibrates included: nothing is carried for them, and the sentence names them. Empty when there are none,
    #: and on a cell that reads no geometry; a camera handed in without its section names only itself here.
    without_body: str = ""

    @property
    def ok(self) -> bool:
        return not self.refusal

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def sweep_banner(self) -> str:
        """The sweep stage's heading: the robot moves by itself, or only when a person moves it."""
        return _BY_HAND_BANNER if self.by_hand == "freedrive" else CalibrationStage.SWEEP.banner()

    def render(self) -> str:
        """The config stage as the CLI prints it. ASCII, no trailing newline, no arguments."""
        if self.refusal:
            return f"[config] REFUSED: {self.refusal}"
        return "\n".join((
            f"  rig        {self.rig_id!r} ({self.rig_source})",
            f"  mode       {self.mode.value}",
            f"  arm        {self.vendor}",
            *((f"  {self.wrist}",) if self.wrist else ()),
            *((f"  !! {self.without_body}",) if self.without_body else ()),
            f"  board      {self.target}" if self.target else
            f"  marker     {self.marker_length_mm:.1f} mm, id {self.marker_id}, {self.dict_name}",
            f"  poses      {self._poses_line()}",
            f"  artifact   {self.artifact_path}",
        ))

    def _poses_line(self) -> str:
        """How many stations the sweep visits and where they come from, or how many samples a person guides to."""
        where = f" of {self.poses_file}" if self.poses_file else ""
        if self.by_hand == "freedrive":
            toward = f", toward the {self.targets} stations{where}" if self.targets else ""
            return _ascii(f"{self.poses} guided by hand{toward}")
        line = f"{self.poses} from {self.poses_file}" if self.poses_file else str(self.poses)
        if self.joint_stations:
            line += f", {self.joint_stations} of them joint stations"
        if self.by_hand == "adjust":
            line += ", each adjusted by hand"
        return _ascii(line)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rig_id": self.rig_id,
            "mode": self.mode.value,
            "ok": self.ok,
            "refusal": self.refusal,
            "rig_source": self.rig_source,
            "vendor": self.vendor,
            "marker_length_mm": self.marker_length_mm,
            "marker_id": self.marker_id,
            "dict_name": self.dict_name,
            "poses": self.poses,
            "artifact_path": self.artifact_path,
            "target": self.target,
            "poses_file": self.poses_file,
            "joint_stations": self.joint_stations,
            "by_hand": self.by_hand,
            "targets": self.targets,
            "wrist": self.wrist,
            "without_body": self.without_body,
        }


@dataclass(frozen=True, slots=True)
class CalibrationBuild:
    """What was built for a sweep and what its arm will refuse, known before anything moves."""

    #: Why the build stopped, as ``Type: message``. Empty when it stands.
    refusal: str = ""
    #: The arm's class name. Empty when the build stopped before the arm and the camera stood.
    arm: str = ""
    gripper: str = ""
    lock_key: str | None = None
    #: The camera handle, as ``repr`` gives it.
    camera: str = ""
    #: Whether the camera reports a camera matrix. A marker pose needs one.
    intrinsics: bool = False
    safety: "SafetyAttestation | None" = None
    #: What world the arm plans against, as ``Robot.camera_world_line()`` says it.
    camera_world: str = ""
    #: The wrist bodies the arm carries, as ``WristBodies.line()`` says it: the camera an eye in hand sweep
    #: calibrates, and every other wrist camera the tree declares, for either mounting, beside any a robot handed in
    #: already held. Empty when none was read.
    wrist: str = ""
    #: The check's warning for the wrist rigs declared on the arm without a body, repeated where the arm is built and
    #: logged. Empty when none.
    without_body: str = ""
    #: The decline of a sweep that moves the arm without a wrist camera's body: the camera an eye in hand sweep
    #: calibrates, or a wrist camera on the arm whose body cannot be placed yet. Empty when none.
    unmodelled: str = ""
    #: What the sweep's preview window will do, or why it will not open when it was asked for by name. Empty when
    #: none was asked for, and for ``"auto"`` where no window can show.
    preview: str = ""

    @property
    def ok(self) -> bool:
        return not self.refusal

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The build stage as the CLI prints it. ASCII, no trailing newline, no arguments."""
        lines: list[str] = []
        if self.arm:
            lines += [
                f"  arm      {self.arm}",
                f"  gripper  {self.gripper}",
                f"  lock     {self.lock_key or 'none, this arm drives no controller of its own'}",
                f"  camera   {self.camera} intrinsics={'yes' if self.intrinsics else 'NO'}",
            ]
            if self.preview:
                lines.append(f"  preview  {self.preview}")
            if self.safety is not None:
                lines.append(self.safety.render())
            lines.append(f"  {self.camera_world}; this sweep declines for itself")
            if self.wrist:
                lines.append(f"  {self.wrist}")
            if self.without_body:
                lines.append(f"  !! {self.without_body}")
            if self.unmodelled:
                lines.append(f"  !! {self.unmodelled}")
        if self.refusal:
            lines.append(f"[build] REFUSED: {self.refusal}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "refusal": self.refusal,
            "arm": self.arm,
            "gripper": self.gripper,
            "lock_key": self.lock_key,
            "camera": self.camera,
            "intrinsics": self.intrinsics,
            "safety": self.safety.to_dict() if self.safety is not None else None,
            "camera_world": self.camera_world,
            "wrist": self.wrist,
            "without_body": self.without_body,
            "unmodelled": self.unmodelled,
            "preview": self.preview,
        }


@dataclass(frozen=True, slots=True)
class CalibrationRunReport:
    """What a run did: the config, the build, the connect, the sweep and what it wrote."""

    check: CalibrationCheck
    outcome: CalibrationOutcome
    build: CalibrationBuild | None = None
    #: The connect refusal verbatim for ``CELL_BUSY``; ``Type: message`` for ``CONNECT_FAILED`` and ``SWEEP_FAILED``.
    failure: str = ""
    #: How the arm came down. ``None`` when it never came up.
    teardown: "TeardownReport | None" = None
    accepted_samples: int = 0
    rmse_mm: float | None = None
    max_error_mm: float | None = None
    #: The RMSE's label under the tree's ``robot.calibration.quality_bands_mm``, the label the artifact carries.
    quality: str = ""
    #: The solved transform as a 4x4 in millimetres: CAMERA to BASE (eye to hand) or CAMERA to TOOL (eye in hand).
    transform_mm: tuple[tuple[float, ...], ...] | None = None
    #: The flange to TCP an eye in hand artifact records. ``None`` when it records none (it is then ``/1``).
    flange_to_tcp: "FlangeToTcp | None" = None
    artifact_path: str = ""
    dataset_path: str | None = None
    #: The block to paste under ``camera.cameras.rigs``, as YAML text ending in a newline. Empty when nothing was
    #: written. A wrist camera's shutter motion tolerances are comments to measure and fill in.
    rig_block: str = ""
    #: What stood behind each move the sweep commanded, in command order.
    camera_worlds: "tuple[CameraWorldStamp, ...]" = ()
    #: Each pose the sweep visited, in order: counted, or why not. Empty when the routine kept none, and the summary
    #: then prints no table.
    pose_log: "tuple[PoseVerdict, ...]" = ()
    #: The stations file a run guided by hand wrote, one joint station per counted pose, which ``fixed_poses``
    #: replays without hands. Empty when none was written.
    stations_path: str = ""

    @property
    def exit_code(self) -> int:
        """0 done, 1 refused before the sweep, 2 ran and wrote nothing, 3 the sweep raised."""
        return _EXIT_CODES[self.outcome]

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def swept(self) -> bool:
        """Whether the run went past the build and announced the sweep."""
        return self.outcome in _SWEPT

    def summary(self) -> str:
        """Everything after the build stage, as the CLI prints it after ``run()`` returns.

        A fragment, so it keeps the name ``summary``: the CLI prints the config and the build stages as they happen,
        and the sweep banner before the arm moves, and then this. The rig block is not in it, because the CLI prints
        that as the YAML text it is. Empty for a run that stopped at the config or the build.
        """
        outcome = self.outcome
        if outcome is CalibrationOutcome.DRY_RUN:
            return "\n--dry-run: built cleanly and the camera answered. Stopping before any motion."
        if outcome is CalibrationOutcome.CELL_BUSY:
            return f"[connect] REFUSED: {self.failure}"
        if outcome is CalibrationOutcome.CONNECT_FAILED:
            return f"[connect] FAILED: {self.failure}"
        if outcome not in _SWEPT:
            return ""
        lines = ["", "  down: the arm, then the lock, then the camera"]
        if self.teardown is not None:
            lines.append(self.teardown.render())
        if outcome is CalibrationOutcome.SWEEP_FAILED:
            lines.append(f"[sweep] FAILED: {self.failure}")
            lines += self._pose_lines()
            lines += self._stations_lines()
            return "\n".join(lines)
        lines += [
            "",
            CalibrationStage.RESULT.banner(),
            *self._pose_lines(),
            f"  accepted samples  {self._accepted()}",
            f"  AX=XB rmse        {self.rmse_mm or 0.0:.4f} mm (max {self.max_error_mm or 0.0:.4f}) -> {self.quality}",
        ]
        if outcome is CalibrationOutcome.NO_ARTIFACT:
            kind = "Extrinsics" if self.check.mode is MountingMode.EYE_TO_HAND else "Transform"
            lines += ["", f"[result] no {kind} carrier; nothing was written. This camera keeps its "
                          "previous calibration, if it had one."]
            return "\n".join(lines)
        if self.check.mode is MountingMode.EYE_IN_HAND:
            recorded = "not recorded" if self.flange_to_tcp is None else f"recorded on {self.flange_to_tcp.source}"
            lines.append(f"  flange to TCP     {recorded}")
        lines += [
            f"  written           {self.artifact_path}",
            f"  dataset           {self.dataset_path}",
            *self._stations_lines(),
            "",
            "Paste this into the camera section so the camera reaches the pick path. Until its rig",
            "declares it, the cell has no CAMERA->BASE for this camera:",
        ]
        return "\n".join(lines)

    def _accepted(self) -> str:
        """Accepted against the poses the sweep ran: the pose log's count, else the stations the check counted."""
        run = len(self.pose_log) if self.pose_log else self.check.poses
        return f"{self.accepted_samples}/{run}"

    def _stations_lines(self) -> list[str]:
        """Where the stations counted by hand went, and how to run them again without hands."""
        if not self.stations_path:
            return []
        return [_ascii(f"  stations          {self.stations_path}"),
                _ascii(f"                    replay without hands: --fixed-poses {self.stations_path}")]

    def _pose_lines(self) -> list[str]:
        """The per-pose table: each pose's index, label and verdict, and why it did not count."""
        if not self.pose_log:
            return []
        counted = sum(1 for verdict in self.pose_log if verdict.counted)
        names = [ascii(verdict.label) if verdict.label else "" for verdict in self.pose_log]
        width = max(len(name) for name in names)
        lines = [f"  per pose          {counted} of {len(self.pose_log)} counted"]
        for verdict, name in zip(self.pose_log, names):
            if verdict.counted:
                said = f"COUNTED   {verdict.detail}"
            else:
                said = f"REJECTED  {verdict.reason}" + (f": {verdict.detail}" if verdict.detail else "")
            lines.append(_ascii(f"    {verdict.index:>3}  {name:<{width}}  {said}").rstrip())
        return lines

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The whole run with its stage banners. ASCII, no trailing newline, no arguments."""
        lines = [CalibrationStage.CONFIG.banner(), self.check.render()]
        if self.build is not None:
            lines += ["", CalibrationStage.BUILD.banner(), self.build.render()]
        if self.swept:
            lines += ["", self.check.sweep_banner()]
        tail = self.summary()
        if tail:
            lines.append(tail)
        if self.rig_block:
            lines += ["", self.rig_block.rstrip("\n")]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe with no custom encoder."""
        return {
            "rig_id": self.check.rig_id,
            "mode": self.check.mode.value,
            "outcome": self.outcome.value,
            "exit_code": self.exit_code,
            "ok": self.ok,
            "check": self.check.to_dict(),
            "build": self.build.to_dict() if self.build is not None else None,
            "failure": self.failure,
            "teardown": self.teardown.to_dict() if self.teardown is not None else None,
            "accepted_samples": self.accepted_samples,
            "rmse_mm": self.rmse_mm,
            "max_error_mm": self.max_error_mm,
            "quality": self.quality,
            "transform_mm": [list(row) for row in self.transform_mm] if self.transform_mm is not None else None,
            "flange_to_tcp": self.flange_to_tcp.to_dict() if self.flange_to_tcp is not None else None,
            "artifact_path": self.artifact_path,
            "dataset_path": self.dataset_path,
            "rig_block": self.rig_block,
            "camera_worlds": [stamp.to_dict() for stamp in self.camera_worlds],
            "pose_log": [verdict.to_dict() for verdict in self.pose_log],
            "stations_path": self.stations_path,
        }


@dataclass(frozen=True, slots=True)
class _Sections:
    """The two halves of a tree a sweep reads. ``camera`` is ``UNSET`` for a camera handed in."""

    robot: Any
    camera: Any = UNSET


@dataclass(frozen=True, slots=True)
class _Staged:
    """The config stage: its report, and the rig and wrist bodies the build needs when it passed."""

    check: CalibrationCheck
    rig: Any = None
    #: The body of the camera an eye in hand sweep calibrates, ``None`` when none is read or it is swept without one.
    wrist: Any = None
    #: Every other wrist camera the tree declares on the arm, for either mounting; ``None`` when none is read.
    carried: Any = None
    #: The validated target: an ``ArucoTargetConfig`` or a ``CharucoTargetConfig``.
    target: Any = None


@dataclass(frozen=True, slots=True)
class _Parts:
    """What the build stood up, and whether the camera is this run's to give back."""

    robot: Any
    routine: Any
    handle: Any
    owned: bool
    #: The sweep's preview window, or ``None``. It grabs through ``handle``.
    preview: Any = None

    def give_back(self) -> None:
        """The preview first, then the camera: the preview grabs through the handle, and a grab after the release
        would meet a closed camera. The camera is given back whatever the close did."""
        try:
            if self.preview is not None:
                self.preview.close()
        finally:
            if self.owned:
                self.handle.release()


@dataclass(frozen=True, slots=True)
class HandEyeCalibration:
    """One camera of a cell, calibrated against its robot: ``check()`` at a desk, ``run()`` at the cell.

    Built by ``from_tree`` (a loaded tree), ``from_config`` (a validated config and the directory it came from) or
    ``from_parts`` (a built ``Robot`` and ``Camera``). Building touches nothing. Every setting is resolved at the
    factory, so ``check().render()`` prints the values the sweep will use.
    """

    rig_id: str
    mode: MountingMode
    #: The mode's ``camera.hand_eye`` block: the thresholds the sweep's poses and samples are held to.
    settings: Any
    #: How many samples a run guided throughout by hand collects (``SweepOptions.samples``).
    samples: int
    marker_length_mm: float
    marker_id: int
    dict_name: str
    out_dir: str
    sections: _Sections = field(repr=False)
    #: A target named in full, validated at ``check()``: the options' ``target``, or the block's ``target`` when it is
    #: a board. ``None`` for one marker, which the three fields above describe (a block ``target`` naming one marker
    #: is read into them).
    target: Any = field(default=None, repr=False)
    #: Where ``target`` came from, ``"options"`` or ``"tree"``; empty without one.
    target_from: str = ""
    #: The single-marker options a caller set (``marker_length_mm``, ``marker_id``, ``dict_name``). Refused beside a
    #: target named in full, which states them itself.
    overrides: tuple[str, ...] = ()
    unmodelled_wrist_body: str | None = None
    #: A caller's own stations: a list of Pose (BASE) and JointStation, or a path to a JSON file of them. For a run
    #: guided throughout by hand, the targets it shows the way to. None names none.
    fixed_poses: "Sequence[Pose | JointStation] | str | Path | None" = field(default=None, repr=False)
    #: The preview window: ``True``, ``"auto"`` or ``False`` (see ``SweepOptions.preview``). Off unless asked for.
    preview: "bool | Literal['auto']" = False
    #: A run guided throughout by hand (``SweepOptions.freedrive``).
    freedrive: bool = False
    #: Fixed stations, each adjusted by hand once the arm reached it (``SweepOptions.adjust``).
    adjust: bool = False
    #: Where a person guiding the arm reads and types. None is the terminal.
    console: "OperatorConsole | None" = field(default=None, repr=False)
    #: The config tree the sections came from. None is the repository's tree.
    data_dir: "str | Path | None" = None
    #: A robot built by the caller. ``UNSET``: the arm alone is built at ``run()`` from the robot section.
    robot: "Maybe[Robot]" = field(default=UNSET, repr=False)
    #: A camera owner built by the caller. ``UNSET``: the rig's owner is built at ``run()`` from the camera section.
    camera: "Maybe[Camera]" = field(default=UNSET, repr=False)
    #: Narration of the connect, forwarded to ``Robot.connected()``.
    announce: "Callable[[ConnectStage], None] | None" = None
    #: Progress of the sweep, one event per pose, forwarded to ``CalibrationRoutine``.
    on_event: "RobotCalibrationEventListener | None" = None
    #: Called with the build report after the build and before any motion, on a refused build too. A hook, because
    #: the build, the attestation and the sweep run inside one verb and an operator must read what the arm refuses
    #: before it moves, not after.
    on_built: "Callable[[CalibrationBuild], None] | None" = None

    # --- three doors -------------------------------------------------------------------------

    @classmethod
    def from_tree(
        cls, loaded: "LoadedTree", *, rig_id: str, mode: "MountingMode | str",
        options: Maybe[SweepOptions] = UNSET,
        announce: "Callable[[ConnectStage], None] | None" = None,
        on_event: "RobotCalibrationEventListener | None" = None,
        on_built: "Callable[[CalibrationBuild], None] | None" = None,
        console: "OperatorConsole | None" = None,
    ) -> "HandEyeCalibration":
        """The sweep a loaded tree describes for ``rig_id``: both sections and the directory come from one tree.

        A tree that did not load is refused with its own refusal, the ``ConfigError`` ``load_config`` raises.
        """
        if not loaded.ok:
            from src.config.loader import ConfigError

            raise ConfigError(loaded.error)
        # A tree that loaded with no robot block describes no sweep: the loader's own sentence, as a ConfigError.
        _ = loaded.robot
        return cls.from_config(loaded.app_config, rig_id=rig_id, mode=mode, options=options, data_dir=loaded.root,
                               announce=announce, on_event=on_event, on_built=on_built, console=console)

    @classmethod
    def from_config(
        cls, app_config: Any, *, rig_id: str, mode: "MountingMode | str",
        options: Maybe[SweepOptions] = UNSET, data_dir: "str | Path | None" = None,
        announce: "Callable[[ConnectStage], None] | None" = None,
        on_event: "RobotCalibrationEventListener | None" = None,
        on_built: "Callable[[CalibrationBuild], None] | None" = None,
        console: "OperatorConsole | None" = None,
    ) -> "HandEyeCalibration":
        """The sweep a validated config describes for ``rig_id``.

        ``app_config`` is what ``load_config`` returns (its ``robot`` and ``camera`` sections are read). ``data_dir`` is
        the directory it was loaded from, ``None`` for the repository's tree; a wrist camera's body is looked up in
        that tree's registry. ``mode`` has no default: a wrist camera swept eye to hand writes a CAMERA to BASE file for
        a camera that moves.
        """
        return cls.from_parts(robot_config=app_config.robot, camera_config=app_config.camera, rig_id=rig_id,
                              mode=mode, options=options, data_dir=data_dir,
                              announce=announce, on_event=on_event, on_built=on_built, console=console)

    @classmethod
    def from_parts(
        cls, *, mode: "MountingMode | str", robot: "Maybe[Robot]" = UNSET, camera: "Maybe[Camera]" = UNSET,
        robot_config: Maybe[Any] = UNSET, camera_config: Maybe[Any] = UNSET, rig_id: Maybe[str] = UNSET,
        settings: Maybe[Any] = UNSET, options: Maybe[SweepOptions] = UNSET, data_dir: "str | Path | None" = None,
        announce: "Callable[[ConnectStage], None] | None" = None,
        on_event: "RobotCalibrationEventListener | None" = None,
        on_built: "Callable[[CalibrationBuild], None] | None" = None,
        console: "OperatorConsole | None" = None,
    ) -> "HandEyeCalibration":
        """A sweep over parts the caller built, or over the sections to build them from at ``run()``.

        ``robot`` is a built ``Robot``; build it with ``gripper=None`` and no cameras, as the config door does, so no
        activation stroke runs beside the board and no live world refuses the sweep's declines. ``robot_config`` is
        the robot section the workspace box, the calibration block and the tool frame are read from; left unset it is
        the tree the arm keeps (``arm.config``). ``camera`` is an owner of the rig to calibrate; ``run()`` opens it
        when it is closed and gives back only what it opened. Without one, ``camera_config`` and ``rig_id`` name the
        rig. ``settings`` left unset is the mode's ``camera.hand_eye`` block, or the schema's default block when the
        camera section has none. ``console`` is where a person guiding the arm reads and types, the terminal when
        unset.

        The wrist bodies the arm carries are read from ``camera_config``, the tree's camera section. A ``camera``
        handed in without it reads no other wrist camera: an owner holds its own rig and no section, and a robot
        from ``Robot.from_config`` holds none either, so there is nothing to read them from, and guessing would read
        another tree. Pass ``camera_config=`` beside ``camera`` to carry every wrist body the tree declares; without
        it only the body of the camera an eye in hand sweep calibrates is read, from the owner's rig. A robot that
        ``Robot.from_tree`` built carries every declared body already, but without ``camera_config`` a wrist rig
        of its tree declared with no body gets no ``!!`` line here: pass the section to have it named.

        A ``robot`` that already holds bodies (``Robot.wrist_bodies``, and what its arm's guard holds, so a second
        sweep of the same robot counts the first one's) is handed only those it does not hold, equal field by field,
        and nothing at all when it holds them all or there is nothing to carry: an exact mesh guard that is built
        refuses any hand-over, an empty one included. A body it does not hold is handed beside the ones it does, and
        an arm that can no longer take one refuses the build.

        A missing part raises ``ValueError``: that is a call that cannot describe a sweep, not a refusal.
        """
        from src.robot.execution.calibration import DEFAULT_FREEDRIVE_SAMPLES

        selected = MountingMode(mode)
        tree: Any = robot_config if chosen(robot_config) else (
            getattr(robot.arm, "config", None) if chosen(robot) else None)
        if tree is None:
            raise ValueError(
                "a sweep reads the robot section (workspace box, calibration block, tool frame): pass robot_config=, "
                "or a robot whose arm keeps the tree it was built from (arm.config)")
        if chosen(camera):
            if chosen(rig_id) and rig_id != camera.rig_id:
                raise ValueError(f"rig_id={rig_id!r} names another rig than the camera handed in ({camera.rig_id!r})")
            rig = str(camera.rig_id)
        elif chosen(camera_config) and chosen(rig_id):
            rig = str(rig_id)
        else:
            raise ValueError("a sweep needs its camera: pass camera= (an owner of the rig), or camera_config= and "
                             "rig_id=")
        block = settings if chosen(settings) else _settings_for(selected, camera_config)
        chosen_options = options if chosen(options) else SweepOptions()
        reason = chosen_options.unmodelled_wrist_body
        # A marker the block's target names is read like the block's own three keys; a board is kept whole.
        tree_target = getattr(block, "target", None)
        if getattr(tree_target, "kind", None) not in ("aruco", "charuco"):
            tree_target = None
        marker = tree_target if tree_target is not None and tree_target.kind == "aruco" else block
        whole, whole_from = (chosen_options.target, "options") if chosen(chosen_options.target) else (
            (tree_target, "tree") if tree_target is not None and marker is block else (None, ""))
        block_id = getattr(marker, "marker_id", DEFAULT_MARKER_ID)
        tree_samples = getattr(getattr(tree, "calibration", None), "freedrive_samples", UNSET)
        return cls(
            rig_id=rig,
            mode=selected,
            settings=block,
            samples=int(resolve("samples", chosen_options.samples,
                                tree_samples if isinstance(tree_samples, int) else UNSET,
                                DEFAULT_FREEDRIVE_SAMPLES)),
            marker_length_mm=float(resolve("marker_length_mm", chosen_options.marker_length_mm,
                                           marker.marker_length_mm)),
            marker_id=int(resolve("marker_id", chosen_options.marker_id,
                                  block_id if isinstance(block_id, int) else DEFAULT_MARKER_ID)),
            dict_name=str(resolve("dict_name", chosen_options.dict_name, marker.aruco_dict_name)),
            out_dir=str(resolve("out_dir", chosen_options.out_dir, DEFAULT_OUT_DIR)),
            sections=_Sections(robot=tree, camera=camera_config),
            target=whole,
            target_from=whole_from,
            overrides=tuple(name for name in ("marker_length_mm", "marker_id", "dict_name")
                            if chosen(getattr(chosen_options, name))),
            unmodelled_wrist_body=reason if chosen(reason) else None,
            # `resolve` rather than `chosen`, because the field is a union: `Maybe[_T]` solved
            # against `Sequence[Pose] | str | Path | _Unset` leaves `_T` ambiguous, and the guard
            # then narrows to a type that still admits `UNSET`. Resolving against `None` states the
            # default once and types cleanly.
            fixed_poses=resolve("fixed_poses", chosen_options.fixed_poses, None),
            preview=_preview_setting(chosen_options.preview),
            freedrive=bool(resolve("freedrive", chosen_options.freedrive, False)),
            adjust=bool(resolve("adjust", chosen_options.adjust, False)),
            console=console,
            data_dir=data_dir,
            robot=robot,
            camera=camera,
            announce=announce,
            on_event=on_event,
            on_built=on_built,
        )

    # --- the verbs ---------------------------------------------------------------------------

    def check(self) -> CalibrationCheck:
        """What the config says about this sweep. Builds nothing, opens nothing, moves nothing."""
        return self._stage().check

    def run(self, *, dry_run: bool = False) -> CalibrationRunReport:
        """Check, build, and unless ``dry_run``, sweep and write. The robot moves unless ``dry_run``.

        Never raises for a refusal: the config, the build, a held cell, a refused connect and a sweep that raised are
        each an outcome on the report. A failing write of the artifact raises, with the arm down and the camera given
        back.
        """
        staged = self._stage()
        check = staged.check
        if not check.ok:
            return CalibrationRunReport(check=check, outcome=CalibrationOutcome.REFUSED)
        build, parts = self._build(staged)
        try:
            if self.on_built is not None:
                self.on_built(build)
        except BaseException:
            if parts is not None:
                parts.give_back()
            raise
        if parts is None or not build.ok:
            if parts is not None:
                parts.give_back()
            return CalibrationRunReport(check=check, outcome=CalibrationOutcome.BUILD_REFUSED, build=build)
        if dry_run:
            parts.give_back()
            return CalibrationRunReport(check=check, outcome=CalibrationOutcome.DRY_RUN, build=build)
        return self._sweep(check, build, parts)

    # --- the stages --------------------------------------------------------------------------

    def _stage(self) -> _Staged:
        """The config stage: the rig, the wrist body of an eye in hand sweep, every other wrist body the tree declares,
        and a robot that cannot sweep."""
        section = _one_rig(self.camera.rig) if chosen(self.camera) else self.sections.camera
        rig, refusal = _select_rig(section, self.rig_id)
        if refusal is not None:
            return _Staged(CalibrationCheck(rig_id=self.rig_id, mode=self.mode, refusal=refusal))
        wrist = None
        if self.mode is MountingMode.EYE_IN_HAND:
            wrist, refusal = _wrist_body_for_sweep(self.sections.robot, rig, data_dir=self.data_dir,
                                                   reason=self.unmodelled_wrist_body)
            if refusal is not None:
                return _Staged(CalibrationCheck(rig_id=self.rig_id, mode=self.mode, refusal=refusal))
        carried, bodiless, refusal = _declared_wrist_bodies(self.sections.robot, self.sections.camera, rig,
                                                            mode=self.mode, data_dir=self.data_dir,
                                                            reason=self.unmodelled_wrist_body)
        if refusal is not None:
            return _Staged(CalibrationCheck(rig_id=self.rig_id, mode=self.mode, refusal=refusal))
        refusal = self._robot_refusal()
        if refusal is not None:
            return _Staged(CalibrationCheck(rig_id=self.rig_id, mode=self.mode, refusal=refusal))
        target, refusal = self._resolved_target()
        if refusal is not None:
            return _Staged(CalibrationCheck(rig_id=self.rig_id, mode=self.mode, refusal=refusal))
        refusal = self._way_refusal()
        if refusal is not None:
            return _Staged(CalibrationCheck(rig_id=self.rig_id, mode=self.mode, refusal=refusal))
        stations, refusal = self._own_stations()
        if refusal is not None:
            return _Staged(CalibrationCheck(rig_id=self.rig_id, mode=self.mode, refusal=refusal))
        refusal = self._planner_evidence_refusal()
        if refusal is not None:
            return _Staged(CalibrationCheck(rig_id=self.rig_id, mode=self.mode, refusal=refusal))
        prefix = "eth" if self.mode is MountingMode.EYE_TO_HAND else "eih"
        from src.calibration.targets import describe_target
        from src.robot.execution.pose_provider import JointStation

        one_marker = target.kind == "aruco"
        listed = list(stations or ())
        return _Staged(CalibrationCheck(
            rig_id=str(rig.rig_id), mode=self.mode, rig_source=str(rig.source), vendor=f"{self.sections.robot.vendor}",
            marker_length_mm=float(target.marker_length_mm), marker_id=int(target.marker_id) if one_marker else -1,
            dict_name=str(target.aruco_dict_name), poses=self.samples if self.freedrive else len(listed),
            artifact_path=f"{self.out_dir}/{prefix}_{rig.rig_id}.json",
            target="" if one_marker else describe_target(target),
            poses_file=str(self.fixed_poses) if isinstance(self.fixed_poses, (str, Path)) else "",
            joint_stations=sum(isinstance(one, JointStation) for one in listed),
            by_hand="freedrive" if self.freedrive else "adjust" if self.adjust else "",
            targets=len(listed) if self.freedrive else 0,
            wrist=_together(wrist, carried).render() if carried is not None else "",
            without_body=_without_body(bodiless),
        ), rig=rig, wrist=wrist, carried=carried, target=target)

    def _way_refusal(self) -> str | None:
        """Why the stations cannot come from where the options say, or ``None``.

        There is no generated sweep: the stations are the caller's (``fixed_poses``) or a person guides the arm
        (``freedrive``). ``adjust`` works on fixed stations and is refused without them and beside ``freedrive``, which
        guides the arm throughout. A freedrive run needs at least one sample.
        """
        if self.freedrive and self.adjust:
            return ("freedrive and adjust both hand the arm to a person: freedrive guides it to every pose, adjust fine-"
                    "tunes each fixed station the arm drove to. Give one of them")
        if self.adjust and self.fixed_poses is None:
            return ("adjust fine-tunes fixed stations by hand, and none were given: name them with fixed_poses "
                    "(--fixed-poses PATH), or guide the arm to every pose with freedrive (--freedrive)")
        if not self.freedrive and self.fixed_poses is None:
            return ("a sweep needs its stations and none were named: fixed_poses (--fixed-poses PATH, a JSON file of "
                    "poses and joint stations) visits yours, and freedrive (--freedrive) lets you guide the arm to each "
                    "pose by hand. Nothing generates stations")
        if self.freedrive and self.samples < 1:
            return f"a run guided by hand collects at least one sample, not {self.samples}"
        return None

    def _planner_evidence_refusal(self) -> str | None:
        """Why the sweep's planner would refuse to start, read at the desk, or ``None``.

        Only for an arm this noun builds on a real UR cell that plans with cuRobo and declares its planner margin:
        there the planner starts on the committed evidence file for this cell's combination (arm, hand, coupling,
        margin and the carried part's attach slots) and refuses without one, after the arm is connected. The check
        makes the same lookup (``desk_evidence_refusal``, the one ``real_cell --check`` reads), so a combination
        nobody measured, a declared ``payload.length_mm`` whose attach slots have no file among them, is refused
        here instead. An undeclared margin is left to ``real_cell --check``, which blocks on it with its own fix.
        """
        robot_cfg = self.sections.robot
        if chosen(self.robot) or str(getattr(robot_cfg, "vendor", "")) != "ur":
            return None
        if str(getattr(getattr(robot_cfg, "ur", None), "motion_planner", "")) != "curobo":
            return None
        from src.robot.safety.planning.evidence import desk_evidence_refusal
        from src.robot.safety.planning.margin import declared_planner_margin

        self_collision = getattr(getattr(robot_cfg, "safety", None), "self_collision", None)
        if self_collision is None or not chosen(declared_planner_margin(self_collision)):
            return None
        refused, evidence = desk_evidence_refusal(robot_cfg, data_dir=self.data_dir)
        if refused is None and evidence is not None:
            return None
        return (f"the sweep's planner would refuse to start on this cell, after the arm is connected: "
                f"{refused or 'no evidence admits this cell'}")

    def _own_stations(self) -> "tuple[list[Any] | None, str | None]":
        """``(stations, None)`` for the caller's own stations, ``(None, None)`` for none, or ``(None, why)``.

        A file is read and every record checked here (``pose_provider.load_stations``), so ``--check`` refuses a
        file the sweep would refuse, and a list may hold only ``Pose`` and ``JointStation``. Neither is screened
        against the box here: a joint station's grasp centre needs the arm's forward kinematics, and the sweep
        screens each station as it reaches it.
        """
        fixed = self.fixed_poses
        if fixed is None:
            return None, None
        from src.geometry import Pose
        from src.robot.execution.pose_provider import JointStation, load_stations

        if isinstance(fixed, (str, Path)):
            try:
                return list(load_stations(fixed)), None
            except (OSError, ValueError) as exc:
                return None, f"fixed_poses: {exc}"
        stations = list(fixed)
        if not stations:
            return None, ("fixed_poses is an empty list, and a sweep of no stations solves nothing; name some, or "
                          "guide the arm by hand with freedrive")
        for index, station in enumerate(stations):
            if not isinstance(station, (Pose, JointStation)):
                return None, (f"fixed_poses[{index}] is a {type(station).__name__}; a station is a Pose (BASE) or a "
                              "JointStation")
        return stations, None

    def _resolved_target(self) -> "tuple[Any, str | None]":
        """``(target, None)`` validated by the config schema, or ``(None, why)``.

        One marker is the three fields; a target named in full is read from its spec, file, mapping or config. The
        dictionary name is normalised the way the estimator resolves it (case, and an optional ``DICT_``) before the
        schema's strict name check, so ``--dict 5x5_100`` means ``DICT_5X5_100``. The schema refuses a length that is
        not above zero, an unknown dictionary and an id the dictionary does not hold, which would otherwise fail at
        the build or, for the id, turn every pose into ``marker_not_found``. A target file is read here.
        """
        from pydantic import TypeAdapter, ValidationError

        from src.calibration.targets import canonical_aruco_dict_name
        from src.config.schema.camera.cam_schema import CalibrationTargetConfig

        where = ""
        if self.target is None:
            data: dict[str, Any] = {"kind": "aruco", "marker_id": self.marker_id,
                                    "marker_length_mm": self.marker_length_mm, "aruco_dict_name": self.dict_name}
        else:
            where = (f" camera.hand_eye.{self.mode.value}.target" if self.target_from == "tree"
                     else f" {_target_source(self.target)}")
            if self.overrides:
                flags = ", ".join(f"--{name.replace('_name', '').replace('_', '-')}" for name in self.overrides)
                return None, (f"calibration target{where} names the whole target, so the single-marker options "
                              f"({', '.join(self.overrides)}; {flags} on the CLI) cannot be given beside it")
            try:
                data = _target_mapping(self.target)
            except (OSError, ValueError) as exc:
                # A spec that does not parse is named by its own sentence.
                return None, f"calibration target: {exc}" if isinstance(self.target, str) else (
                    f"calibration target{where}: {exc}")
            data.setdefault("aruco_dict_name", self.dict_name)
        name = data.get("aruco_dict_name")
        if isinstance(name, str):
            try:
                data["aruco_dict_name"] = canonical_aruco_dict_name(name)
            except ValueError:
                pass  # the schema's own sentence refuses it below
        try:
            return TypeAdapter(CalibrationTargetConfig).validate_python(data), None
        except ValidationError as exc:
            return None, f"calibration target{where}: {_validation_sentence(exc)}"

    def _robot_refusal(self) -> str | None:
        """Why a robot handed in cannot sweep: its arm holds a live camera world, which refuses a declined move."""
        if not chosen(self.robot):
            return None
        wiring = self.robot.camera_world
        if wiring is None or getattr(wiring, "world", None) is None:
            return None
        return (f"the robot handed in holds a live camera world ({self.robot.camera_world_line()}). Every move of a "
                "sweep declines the camera world, and an arm whose world is wired refuses a declined move. Build the "
                "robot without cameras: Robot.from_config(robot_config, gripper=None)")

    def _build(self, staged: _Staged) -> "tuple[CalibrationBuild, _Parts | None]":
        """The arm alone and one camera, opened, and what the arm refuses. Commands no motion and takes no lock."""
        rig = staged.rig
        handle: Any = None
        owned = False
        preview: Any = None
        try:
            from src.calibration.rgbd_marker_source import RGBDArucoMarkerSource
            from src.calibration.targets import estimator_for
            from src.camera.orchestration.camera import Camera
            from src.robot.execution.calibration import CalibrationRoutine
            from src.robot.execution.hand_guiding import HandGuide
            from src.robot.execution.robot import Robot

            # The arm through the builder a pick run uses, so the arm-vendor readiness gate runs first. No gripper: a
            # sweep beside a board needs no activation stroke, and a gripper this tree cannot build is not a reason to
            # refuse calibrating a camera.
            robot = self.robot if chosen(self.robot) else Robot.from_config(self.sections.robot, gripper=None)
            # A run by hand needs an arm a person can guide: refused before the camera is opened.
            self._refuse_without_hand_guiding(robot.arm)
            # One camera, through the camera noun. A sweep that claimed every configured camera would fight the
            # console for devices it does not need, and a rig the console holds refuses here naming the holder.
            if chosen(self.camera):
                camera = self.camera
                owned = not camera.is_open
            else:
                camera = Camera.from_config(self.sections.camera, rig_id=rig.rig_id)
                owned = True
            handle = camera.handle()
            camera.open()
            # The target the check validated, posed by its own estimator: one marker, or a board.
            target = staged.target
            marker_source = RGBDArucoMarkerSource(streamer=handle, estimator=estimator_for(target))
            # Built here and started by the sweep, so a dry run opens no window and grabs no frame. It sees the
            # judged frame through the source's hook and the events through the routine's listener, and neither
            # can change a pose.
            preview, preview_line = self._preview(handle, target, rig.rig_id)
            if preview is not None:
                marker_source.on_observation = preview.observe
            # A person guiding the arm reads and types in the console and in the preview, which draws what the
            # routine hands it and passes keys back; neither calls the arm.
            routine = CalibrationRoutine(
                arm=robot.arm, marker_source=marker_source, eth_calibrator=self._calibrator(),
                workspace_limits=self.sections.robot.workspace_limits, eth_settings=self.settings,
                rig_id=rig.rig_id, marker_id=int(target.marker_id) if target.kind == "aruco" else -1,
                settle_time_s=self.sections.robot.calibration.settle_time_s,
                calibration_mode=self.mode, on_event=_fan_out(preview, self.on_event),
                hand_guide=HandGuide(self.console, preview), images_dir=self._stem(rig.rig_id, "images"),
            )
            intrinsics = handle.get_intrinsics() is not None
            safety = robot.safety()
            camera_world = robot.camera_world_line()
        except Exception as exc:  # noqa: BLE001 (a build refusal is the designed outcome)
            if preview is not None:
                preview.close()
            if handle is not None and owned:
                handle.release()
            return CalibrationBuild(refusal=f"{type(exc).__name__}: {exc}"), None
        parts = _Parts(robot=robot, routine=routine, handle=handle, owned=owned, preview=preview)
        gripper = ("none, the sweep connects the arm alone" if robot.gripper is None
                   else type(robot.gripper).__name__)
        wrist_line = refusal = ""
        # The cameras on the arm: the one an eye in hand sweep calibrates and every other one the tree declares, each
        # placed from its calibration and handed to the arm before it moves, or moved without a body because the
        # caller said why, which is logged like a decline. A tree that declares none hands nothing over, and a robot
        # handed in is handed only the bodies it does not hold yet.
        carried = _together(staged.wrist, staged.carried)
        if carried is not None:
            try:
                wrist_line = _hand_over(carried, robot)
            except Exception as exc:  # noqa: BLE001 (an arm that cannot hold the camera does not sweep)
                refusal = f"{type(exc).__name__}: {exc}"
        # A wrist rig declared without a body: carried by nobody, named once here as the check named it, and logged.
        without_body = staged.check.without_body
        if without_body:
            logger.warning(without_body)
        declines: list[str] = []
        # A swept camera that declares no body is already named on the warning line; a reason adds nothing to it.
        named = bool(without_body) and getattr(rig, "body", None) is None
        if (staged.wrist is None and self.mode is MountingMode.EYE_IN_HAND and self.unmodelled_wrist_body
                and not named):
            declines.append(f"wrist camera {rig.rig_id!r} swept without its body in the planner and the guard: "
                            f"{self.unmodelled_wrist_body.strip()}")
        if staged.carried is not None and staged.carried.unmodelled:
            names = ", ".join(repr(rig_id) for rig_id, _ in staged.carried.unmodelled)
            which = (f"wrist camera {names} rides on the arm without its body" if len(staged.carried.unmodelled) == 1
                     else f"wrist cameras {names} ride on the arm without their bodies")
            declines.append(f"{which} in the planner and the guard: {staged.carried.unmodelled_reason}")
        unmodelled = "; ".join(declines)
        if unmodelled:
            logger.warning(unmodelled)
        build = CalibrationBuild(
            refusal=refusal, arm=type(robot.arm).__name__, gripper=gripper, lock_key=robot.lock_key,
            camera=repr(handle), intrinsics=intrinsics, safety=safety, camera_world=camera_world,
            wrist=wrist_line, without_body=without_body, unmodelled=unmodelled, preview=preview_line,
        )
        return build, parts

    def _refuse_without_hand_guiding(self, arm: Any) -> None:
        """Raise ``HandGuidingRefused`` where a run by hand meets an arm a person cannot guide."""
        if not (self.freedrive or self.adjust):
            return
        from src.robot.core.freedrive import SupportsFreedrive
        from src.robot.execution.hand_guiding import HandGuidingRefused

        if isinstance(arm, SupportsFreedrive):
            return
        what = "freedrive guides the arm to every pose" if self.freedrive else "adjust frees the arm at each station"
        raise HandGuidingRefused(
            f"{what} by hand, and {type(arm).__name__} offers no hand guiding (SupportsFreedrive; the UR teach mode "
            "is one). This arm has fixed stations only: run them without adjust (--fixed-poses PATH)")

    def _stem(self, rig_id: str, what: str) -> str:
        """A file of this run in ``out_dir``: ``<mode>_<rig>_<what>``."""
        return f"{self.out_dir}/{self.mode.value}_{rig_id}_{what}"

    def _by_hand_stations(self, rig_id: str, routine: Any) -> str | None:
        """Where a run by hand writes the stations it counts, ``None`` for a run nobody guides.

        ``<mode>_<rig>_stations.json``, unless that is the file ``fixed_poses`` names: the file 09 and 10 suggest
        replaying with ``adjust`` is exactly the one this run writes by default, and a run begins its stations file
        afresh, so a replay finished partway would leave one station of the many it was given. The file read is
        never written over (``stations_file_to_write``): the stations go to ``<mode>_<rig>_stations.adjusted.json``
        beside it, and the guide's console says so before anything moves.
        """
        if not (self.freedrive or self.adjust):
            return None
        from src.robot.execution.calibration import stations_file_to_write

        wanted = self._stem(rig_id, "stations.json")
        read_from = self.fixed_poses if isinstance(self.fixed_poses, (str, Path)) else None
        save_to = stations_file_to_write(read_from, wanted)
        if save_to != wanted:
            said = (f"{read_from} is the file the stations were read from, so it is not written over: the stations "
                    f"counted by hand go to {save_to}")
            logger.warning("%s.", said)
            guide = getattr(routine, "hand_guide", None)
            if guide is not None:
                guide.console.say(f"{said}.")
        return save_to

    def _preview(self, handle: Any, target: Any, rig_id: str) -> "tuple[Any, str]":
        """The sweep's preview and the build's line about it: ``(None, "")`` when none was asked for.

        ``"auto"`` builds one only where a window can show (``preview_unavailable``) and stdout is a terminal, and
        says nothing otherwise; ``True`` says why it cannot. Building it opens no window and grabs nothing. Its live
        frames are posed by an estimator of its own, built from the same target, never by the sweep's. A preview that
        cannot be built is left out with its reason: it is never a reason to refuse the sweep.
        """
        if self.preview is False:
            return None, ""
        try:
            from src.calibration.preview import SweepPreview, preview_unavailable
            from src.calibration.targets import estimator_for

            reason = preview_unavailable()
            if reason is None and self.preview == "auto" and not _is_terminal(sys.stdout):
                reason = "stdout is not a terminal"
            if reason is not None:
                return None, "" if self.preview == "auto" else f"off: {reason}"
            preview = SweepPreview(handle, title=f"willy hand-eye sweep: {rig_id}", estimator=estimator_for(target),
                                   describe=render_sweep_event, axis_mm=_axis_mm(target))
        except Exception as exc:  # noqa: BLE001 (display only: the sweep runs without it)
            logger.warning("No sweep preview: %s: %s", type(exc).__name__, exc)
            return None, "" if self.preview == "auto" else f"off: {type(exc).__name__}: {exc}"
        closing = ("closing it finishes the run and holds the arm" if self.freedrive or self.adjust
                   else "closing it closes the window only")
        return preview, f"window {preview.title!r} opens with the sweep; {closing}, the pendant stops the robot"

    def _sweep(self, check: CalibrationCheck, build: CalibrationBuild, parts: _Parts) -> CalibrationRunReport:
        """Connect through ``Robot.connected()``, sweep, solve, and write. The camera is given back last."""
        from src.robot.execution.cell_lock import CellBusy

        failure: Exception | None = None
        result: Any = None
        record: Any = UNSET
        live: Any = None
        # The lock, the connect and the teardown are `Robot.connected()`, the enter and the exit a pick run uses, so
        # this class writes no connect or disconnect of its own. A refused connect is answered before the sweep
        # starts; a sweep that raises is reported once the arm is down.
        try:
            with parts.robot.connected(announce=self.announce) as live:
                if parts.preview is not None:
                    # Once the arm is connected, so a refused connect opens no window. It never raises.
                    parts.preview.start()
                try:
                    Path(self.out_dir).mkdir(parents=True, exist_ok=True)
                    dataset_path = self._stem(check.rig_id, "dataset.json")
                    by_hand = self._by_hand_stations(check.rig_id, parts.routine)
                    # The caller's own stations, in their order: a path is a JSON file of them, screened station by
                    # station; anything else a list of Pose and JointStation. For a run guided throughout by hand
                    # they are targets, never moved to.
                    stations = self._own_stations()[0]
                    if self.freedrive:
                        result = parts.routine.run_freedrive(stations, samples=self.samples,
                                                             dataset_save_path=dataset_path,
                                                             stations_save_path=by_hand)
                    elif isinstance(self.fixed_poses, (str, Path)):
                        result = parts.routine.run_from_json(self.fixed_poses, dataset_save_path=dataset_path,
                                                             adjust=self.adjust, stations_save_path=by_hand)
                    else:
                        result = parts.routine.run_with_poses(stations or [], dataset_save_path=dataset_path,
                                                              adjust=self.adjust, stations_save_path=by_hand)
                    # While the arm is still connected: on a polyscope cell the frame is known only then.
                    record = _flange_to_tcp_record(self.sections.robot, parts.robot.arm)
                except Exception as exc:  # noqa: BLE001 (reported once the arm is down)
                    failure = exc
        except CellBusy as exc:
            # Another process holds this controller, and nothing was commanded.
            return CalibrationRunReport(check=check, outcome=CalibrationOutcome.CELL_BUSY, build=build,
                                        failure=str(exc))
        except Exception as exc:  # noqa: BLE001 (connect is a transaction and has rolled itself back)
            return CalibrationRunReport(check=check, outcome=CalibrationOutcome.CONNECT_FAILED, build=build,
                                        failure=f"{type(exc).__name__}: {exc}")
        finally:
            # After the block, so the arm is down and the lock given back before the camera is. give_back closes the
            # preview before it releases the camera.
            parts.give_back()
        teardown = live.teardown
        if failure is not None:
            # The poses the sweep visited before it raised, a solve refused for too few samples included, and the
            # stations counted by hand so far, which a replay can run again: only stations this run recorded, never
            # a file an earlier run left at the same path.
            return CalibrationRunReport(check=check, outcome=CalibrationOutcome.SWEEP_FAILED, build=build,
                                        failure=f"{type(failure).__name__}: {failure}", teardown=teardown,
                                        pose_log=tuple(getattr(parts.routine, "pose_log", ())),
                                        stations_path=_written(getattr(parts.routine, "stations_written", None)))
        transform = result.transform
        solved = CalibrationRunReport(
            check=check, outcome=CalibrationOutcome.NO_ARTIFACT, build=build, teardown=teardown,
            accepted_samples=int(result.num_samples), rmse_mm=float(result.rmse_mm),
            max_error_mm=float(result.max_error_mm), quality=self._quality(float(result.rmse_mm)),
            transform_mm=_rows(transform.to_matrix()) if transform is not None else None,
            dataset_path=result.dataset_path, camera_worlds=tuple(result.camera_worlds),
            pose_log=tuple(getattr(result, "pose_log", ())),
            stations_path=str(getattr(result, "stations_path", None) or ""),
        )
        # Both modes: an eye in hand solve with no Transform must not reach a write of nothing. With no carrier
        # nothing is written and the camera keeps whatever calibration it had.
        carrier = result.extrinsics if self.mode is MountingMode.EYE_TO_HAND else transform
        if carrier is None:
            return solved
        written = self._write(result, check.rig_id, record)
        kept = record if chosen(record) and self.mode is MountingMode.EYE_IN_HAND else None
        # The block names the file absolute: pasted into a tree, a relative path is read against the config
        # folder (src.config.paths), not against the directory this sweep ran in.
        return dataclasses.replace(
            solved, outcome=CalibrationOutcome.WRITTEN, artifact_path=str(written), flange_to_tcp=kept,
            rig_block=_rig_block(check.rig_id, self.mode.value, str(Path(written).absolute())),
        )

    # --- helpers -----------------------------------------------------------------------------

    def _bands(self) -> Any:
        """The tree's quality bands as the calibration layer's type. The two hold the same three numbers, and the
        conversion is written out so that a field added on either side fails here rather than at a band boundary."""
        from src.calibration.quality import QualityBandsMm

        bands = self.sections.robot.calibration.quality_bands_mm
        return QualityBandsMm(excellent=float(bands.excellent), good=float(bands.good),
                              marginal=float(bands.marginal))

    def _quality(self, rmse_mm: float) -> str:
        from src.calibration.quality import classify_rmse

        return str(classify_rmse(rmse_mm, self._bands()))

    def _calibrator(self) -> Any:
        """The solver the routine would build, holding the tree's bands, so the label in the artifact and the one on
        the report agree."""
        from src.calibration import EyeInHandCalibrator, EyeToHandCalibrator

        solver = EyeToHandCalibrator if self.mode is MountingMode.EYE_TO_HAND else EyeInHandCalibrator
        return solver(settings=self.settings, bands=self._bands())

    def _write(self, result: Any, rig_id: str, record: Any) -> Path:
        """The artifact, keyed by the rig that declares it, the convention the sim runners write."""
        from src.calibration.serialization import save_cam_to_tool, save_extrinsics

        if self.mode is MountingMode.EYE_TO_HAND:
            extrinsics = dataclasses.replace(result.extrinsics, rig_id=rig_id)
            return save_extrinsics(f"{self.out_dir}/eth_{rig_id}.json", extrinsics)
        return save_cam_to_tool(f"{self.out_dir}/eih_{rig_id}.json", result.transform, rig_id=rig_id,
                                flange_to_tcp=record)


def render_sweep_event(event_type: str, data: Mapping[str, Any]) -> str:
    """One line for one event of a sweep, in the wording the calibrate CLI, the examples and the sim runners print.

    ASCII, no newline. ``data`` is the payload ``CalibrationRoutine`` emits: every event carries ``index``, ``total``
    and ``label``, a rejection its ``reason`` and ``detail``, a detection what it rests on. A key the payload lacks is
    left out of the line, so a producer that sends less still prints::

        pose  3/22 'look_2'  moving, 2 counted so far, need 6
        pose  3/22 'look_2'  target seen: 24 corners, 0.31 px, 520 mm away
        pose  3/22 'look_2'  COUNTED 3 (need 6)
        pose  4/22 'look_3'  REJECTED marker_not_found: no DICT_5X5_100 marker in view
    """
    from src.robot.events import RobotCalibrationEvent

    index, total = data.get("index"), data.get("total")
    head = f"pose {index:>2}/{total}" if isinstance(index, int) and isinstance(total, int) else "pose"
    label = data.get("label")
    if label:
        head += f" {ascii(str(label))}"
    need = data.get("min_samples")
    if event_type == RobotCalibrationEvent.MOVING_TO_POSE:
        needed = f", need {need}" if isinstance(need, int) else ""
        return _ascii(f"{head}  moving, {data.get('accepted', 0)} counted so far{needed}")
    if event_type == RobotCalibrationEvent.MARKER_DETECTED:
        parts: list[str] = []
        if data.get("n_points"):
            parts.append(f"{data['n_points']} corners")
        if isinstance(data.get("reprojection_px"), (int, float)):
            parts.append(f"{float(data['reprojection_px']):.2f} px")
        if isinstance(data.get("distance_mm"), (int, float)):
            parts.append(f"{float(data['distance_mm']):.0f} mm away")
        line = f"{head}  target seen" + (f": {', '.join(parts)}" if parts else "")
        hint = data.get("hint")
        return _ascii(f"{line}; !! {hint}" if hint else line)
    if event_type == RobotCalibrationEvent.POSE_ACCEPTED:
        needed = f" (need {need})" if isinstance(need, int) else ""
        return _ascii(f"{head}  COUNTED {data.get('accepted', '?')}{needed}")
    if event_type == RobotCalibrationEvent.POSE_REJECTED:
        detail = data.get("detail") or data.get("why_not") or ""
        return _ascii(f"{head}  REJECTED {data.get('reason') or 'rejected'}" + (f": {detail}" if detail else ""))
    return _ascii(f"{head}  {event_type}")


def print_sweep_progress(event_type: str, data: Mapping[str, Any]) -> None:
    """A sweep's ``on_event``: prints each event as :func:`render_sweep_event` words it, indented, and flushes.

        HandEyeCalibration.from_tree(tree, rig_id="overhead", mode="eye_to_hand", on_event=print_sweep_progress)
    """
    print(f"  {render_sweep_event(event_type, data)}", flush=True)


def _ascii(text: str) -> str:
    """``text`` with anything outside ASCII replaced, as every report line is."""
    return text.encode("ascii", "replace").decode("ascii")


def _preview_setting(value: Any) -> "bool | Literal['auto']":
    """``SweepOptions.preview`` as the noun keeps it. Unset is ``False``: the library opens no window unasked."""
    if not chosen(value):
        return False
    if value is True or value is False:
        return value
    if value == "auto":
        return "auto"
    raise ValueError(f"SweepOptions.preview is True, False or 'auto', not {value!r}")


def _written(path: Any) -> str:
    """``path`` where a file was written there, else ``""``."""
    return str(path) if path and Path(str(path)).is_file() else ""


def _is_terminal(stream: Any) -> bool:
    """Whether ``stream`` is a terminal. A stream that cannot say is not one."""
    isatty = getattr(stream, "isatty", None)
    try:
        return bool(isatty()) if callable(isatty) else False
    except Exception:  # noqa: BLE001 (a closed or odd stream is not a terminal)
        return False


def _fan_out(preview: Any, listener: "RobotCalibrationEventListener | None") -> Any:
    """The routine's one listener: the preview first, which only queues the event and never raises, then the
    caller's, whose exceptions the routine swallows as before."""
    if preview is None:
        return listener

    def both(event_type: str, data: dict[str, Any]) -> None:
        preview(event_type, data)
        if listener is not None:
            listener(event_type, data)

    return both


def _axis_mm(target: Any) -> float:
    """How long the preview draws the target's axes: a marker's edge, or two squares of a board."""
    if getattr(target, "kind", "") == "charuco":
        return 2.0 * float(target.square_length_mm)
    return float(target.marker_length_mm)


def _target_source(target: Any) -> str:
    """How a target named in full is referred to in a refusal: its spec, its file, or the option."""
    if isinstance(target, str):
        return repr(target)
    if isinstance(target, Path):
        return f"file {target}"
    return "SweepOptions.target"


def _target_mapping(target: Any) -> dict[str, Any]:
    """The mapping a target named in full holds: a board spec, a YAML or JSON file, a mapping, or a target config."""
    from src.calibration.targets import parse_board_spec

    if isinstance(target, str):
        return parse_board_spec(target)
    if isinstance(target, Path):
        import yaml

        try:
            loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"is not YAML or JSON: {exc}") from None
        if not isinstance(loaded, Mapping):
            raise ValueError("holds no mapping; write one target, its kind (aruco or charuco) and its keys")
        return dict(loaded)
    if isinstance(target, Mapping):
        return dict(target)
    dump = getattr(target, "model_dump", None)
    if callable(dump):
        return dict(dump())
    raise ValueError(f"a target is a board spec, a Path to a file, a mapping or a target config, not "
                     f"{type(target).__name__}")


def _validation_sentence(exc: Any) -> str:
    """A pydantic refusal as one line: each field and what it needs, without the error-page links."""
    parts: list[str] = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error.get("loc", ()) if part not in ("aruco", "charuco"))
        said = str(error.get("msg", "")).removeprefix("Value error, ")
        parts.append(f"{where}: {said}" if where else said)
    return "; ".join(parts)


def _settings_for(mode: MountingMode, camera_config: Any) -> Any:
    """The mode's ``camera.hand_eye`` block, or the schema's default block for a section that carries none.

    The block of the sweep's own mounting, so a wrist sweep reads ``camera.hand_eye.eye_in_hand`` for its marker and
    its sample thresholds. On every shipped tree the two blocks agree.
    """
    hand_eye = getattr(camera_config, "hand_eye", None) if chosen(camera_config) else None
    if hand_eye is None:
        from src.config.schema.camera import HandEyeConfig

        hand_eye = HandEyeConfig()
    return getattr(hand_eye, mode.value)


def _one_rig(rig: Any) -> Any:
    """A camera section holding one rig, for the rules that read a section."""
    return SimpleNamespace(cameras=SimpleNamespace(rigs=[rig], primary_rig_id=rig.rig_id))


def _rows(matrix: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _select_rig(camera_cfg: Any, rig_id: str) -> "tuple[Any, str | None]":
    """``(rig, None)``, or ``(None, why)`` for a rig that is not configured, switched off, or has no depth.

    The refusals are the camera noun's (``select_rig``), so a sweep and the cell say one thing about a rig. A stereo
    pair calibrates through the routine's own stereo path, not this one, and letting a webcam rig through would fail
    much later about ArUco rather than about the rig. A rig switched off is refused, because a check that called it
    usable would be followed by a sweep of every pose in front of a camera the cell will not open. The sentence about
    the artifact is added to that refusal, because it is the case an operator is most likely to meet here.
    """
    from src.camera.orchestration.camera import CameraRefused, select_rig

    try:
        return select_rig(camera_cfg, rig_id=rig_id), None
    except CameraRefused as refused:
        tail = (" A sweep against a camera the cell will not open produces an artifact nothing consumes."
                if refused.reason == "disabled" else "")
        return None, f"{refused}{tail}"


def _rig_block(camera_id: str, mode: str, path: str) -> str:
    """The rig block an operator pastes into the camera section so this camera's calibration is read.

    Writing the artifact is only half the job. Until the rig declares it, the cell has no CAMERA to BASE for this
    camera. A fixed camera's block is complete as printed. A wrist camera's shutter motion tolerances are a fact of the
    cell that is not measured here, so its block names them as comments to fill in, and the loader refuses the block
    until they are written.
    """
    tolerances = "" if mode == "eye_to_hand" else (
        "          # shutter_motion_tolerance_mm: <measure: how far the tool may travel while a frame is taken>\n"
        "          # shutter_motion_tolerance_deg: <measure: how far the tool may turn while a frame is taken>\n"
        "          # record_tolerance_mm: <measure, when the rig declares a body: how far the connect derives the tool frame apart>\n"
        "          # record_tolerance_deg: <measure, when the rig declares a body: the same in degrees>\n"
    )
    return (
        "camera:\n"
        "  cameras:\n"
        "    rigs:\n"
        f"      - rig_id: {camera_id}\n"
        "        extrinsics:\n"
        f"          mounting_mode: {mode}\n"
        f"          artifact_path: {path}\n"
        + tolerances
    )


def _flange_to_tcp_record(robot_cfg: Any, arm: Any) -> Any:
    """The flange to TCP the arm's tool pose applied during the sweep, as an eye in hand artifact keeps it.

    The solve is CAMERA to TOOL against ``get_tcp_pose``, which is the flange times this frame, so the body a wrist
    camera declares is placed from it and a later tool frame that differs refuses the rig. Read from the arm while it
    is still connected: the declared frame on a ``willy`` cell, the frame derived from the controller at connect on a
    ``polyscope`` cell. ``UNSET`` for an arm that does not say which frame it applies and for an ``undeclared`` cell,
    and the artifact is then written as ``/1``.
    """
    from src.calibration.serialization import FlangeToTcp

    source = robot_cfg.gripper.tool_frame.source
    matrix = getattr(arm, "active_tool_frame", None)
    if source not in ("willy", "polyscope") or matrix is None:
        return UNSET
    return FlangeToTcp.from_matrix(source, matrix)


def _wrist_body_for_sweep(robot_cfg: Any, rig: Any, *, data_dir: Any, reason: "str | None") -> Any:
    """``(bodies, refusal)`` for an eye in hand sweep of ``rig``.

    On a cell that reads geometry, a body the registry cannot stand for is refused, whatever the reason. A body that
    cannot be placed yet is refused unless ``reason`` says why the sweep may run without it, and the sweep then runs
    with no wrist body. A rig that declares no body at all refuses nothing (the owner's decision of 2026-09-25, for
    calibration only): nothing is carried for it, and :func:`_declared_wrist_bodies` names it on the sweep's one
    warning line. On any other cell nothing is read.
    """
    from src.calibration.rig_calibration import RigArtifactMissing
    from src.config.cameras import load_camera, tree_camera_refusal
    from src.config.loader import ConfigError
    from src.robot.execution.wrist_bodies import WristBodies, WristBodyRequired, WristBodyUnplaced
    from src.robot.safety.planning.hand import wrist_body_reader

    reader = wrist_body_reader(robot_cfg)
    key = f"camera.cameras.rigs[{rig.rig_id!r}]"
    if reader is None:
        return None, None
    if getattr(rig, "body", None) is None:
        # Swept without its housing in the planner and the guard; the sweep's `!!` line says so
        # (`_declared_wrist_bodies`), and declaring the body (docs/calibration-setup.md, section 4) carries it.
        return None, None
    registry = tree_camera_refusal(rig.body.model, data_dir=data_dir)
    if registry is None:
        try:
            load_camera(rig.body.model)
        except ConfigError as exc:
            registry = str(exc)
    if registry is not None:
        return None, f"{key}.body: {registry}"
    try:
        return WristBodies.from_config(robot_cfg, _one_rig(rig), data_dir=data_dir), None
    except WristBodyUnplaced as exc:
        if reason and reason.strip():
            return None, None
        # The observation alone when the artifact is missing: its remedy is to run this sweep, which is what
        # the operator reading this refusal just did.
        cause = exc.__cause__
        said = (f"{key}.body cannot be placed: {cause.observation}" if isinstance(cause, RigArtifactMissing)
                else str(exc))
        # rstrip: the refusal it wraps may end on a full stop of its own, and ".." reads as a typo.
        return None, (f"{said.rstrip('.')}. To sweep this camera before its body can be placed, say why: "
                      '--unmodelled-wrist-body "<reason>"')
    except WristBodyRequired as exc:
        # A body that is placed and does not cover its camera (E14): no reason stands in for a cover, so none is
        # offered (review of 2026-09-25; before, a reason swept this camera with no body at all).
        return None, str(exc)


def _declared_wrist_bodies(
    robot_cfg: Any, camera_cfg: Any, swept: Any, *, mode: MountingMode, data_dir: Any, reason: "str | None",
) -> "tuple[Any, tuple[str, ...], str | None]":
    """``(bodies, bodiless, refusal)`` for the wrist cameras the tree declares on the arm a sweep drives.

    A camera the arm carries hangs on it whichever camera is swept, so every sweep carries the bodies the tree's
    camera section declares, resolved as ``Robot.from_tree`` resolves them (``WristBodies.from_config``, one rig at a
    time, so a refusal names its camera). ``bodies`` is ``None`` where none is read. The camera an eye in hand sweep
    calibrates is left to :func:`_wrist_body_for_sweep` when it declares a body, and named in ``bodiless`` when it
    declares none. Nothing else is read, and the sweep is what it was, for a camera handed in without its section,
    on a cell that reads no geometry, and on a tree that declares no other camera on the arm (no rig with a ``body``
    or ``eye_in_hand`` extrinsics).

    ``bodiless`` names the wrist rigs (``eye_in_hand`` extrinsics) declared without a body, switched on or off, and
    the camera an eye in hand sweep calibrates when it declares none. The owner's decisions of 2026-09-25, for
    calibration only: such a rig refuses no sweep, its own eye in hand sweep included (a switched-off rig being
    calibrated is refused by ``_select_rig`` before this runs), nothing is carried for it, and the check and the
    build name it on one warning line; ``Robot.from_tree`` and the pick path still refuse an enabled one. A declared body that cannot be placed yet refuses the sweep unless ``reason`` says why the arm may
    move without it, and no reason excuses a camera the registry does not stand for.
    """
    from src.robot.execution.wrist_bodies import WristBodies, WristBodyRequired, WristBodyUnplaced
    from src.robot.safety.planning.hand import wrist_body_reader

    if wrist_body_reader(robot_cfg) is None:
        return None, (), None
    own = ((str(swept.rig_id),)
           if mode is MountingMode.EYE_IN_HAND and swept is not None and getattr(swept, "body", None) is None else ())
    if not chosen(camera_cfg):
        return None, own, None
    rigs = [rig for rig in getattr(getattr(camera_cfg, "cameras", None), "rigs", None) or ()
            if not (mode is MountingMode.EYE_IN_HAND and str(rig.rig_id) == str(swept.rig_id))]
    bodiless = own + tuple(str(rig.rig_id) for rig in rigs
                           if _on_the_arm(rig) and getattr(rig, "body", None) is None)
    declared = [rig for rig in rigs if getattr(rig, "body", None) is not None]
    if not declared:
        return None, bodiless, None
    resolved: list[Any] = []
    for rig in declared:
        try:
            resolved.append(WristBodies.from_config(robot_cfg, _one_rig(rig), data_dir=data_dir,
                                                    unmodelled_reason=reason if reason else UNSET))
        except WristBodyUnplaced as exc:
            return None, bodiless, _unplaced_refusal(exc, rig_id=str(rig.rig_id), swept=str(swept.rig_id), mode=mode)
        except WristBodyRequired as exc:
            return None, bodiless, str(exc)
    return WristBodies(
        bodies=tuple(body for one in resolved for body in one.bodies), reader=resolved[0].reader,
        unmodelled=tuple(row for one in resolved for row in one.unmodelled),
        unmodelled_reason=next((one.unmodelled_reason for one in resolved if one.unmodelled_reason), ""),
    ), bodiless, None


def _unplaced_refusal(exc: Exception, *, rig_id: str, swept: str, mode: MountingMode) -> str:
    """The refusal of a sweep that moves the arm carrying ``rig_id``, whose declared body cannot be placed yet.

    This sweep's own flag first, labelled with the rig it runs for, and then why the body cannot be placed, which for
    a camera with no calibration names the separate command that calibrates it, with a flag of its own.
    """
    return (f"this {mode.value} sweep of {swept!r} moves the arm that carries wrist camera {rig_id!r}, whose declared "
            f"body cannot be placed yet. For THIS sweep (--rig {swept}), say why the arm may move without that body: "
            f'--unmodelled-wrist-body "<reason>". Why the body of {rig_id!r} cannot be placed: {exc}')


def _without_body(rig_ids: "tuple[str, ...]") -> str:
    """The one warning for the wrist rigs declared on the arm without a body, or ``""`` for none. ASCII."""
    if not rig_ids:
        return ""
    names = ", ".join(repr(rig_id) for rig_id in rig_ids)
    said = (f"wrist camera {names} is declared on the arm without a body: the planner and the guard do not know it "
            "is there during this sweep" if len(rig_ids) == 1 else
            f"wrist cameras {names} are declared on the arm without a body: the planner and the guard do not know "
            "they are there during this sweep")
    return _ascii(said)


def _hand_over(carried: Any, robot: Any) -> str:
    """Hand ``carried`` to the robot's arm, less the bodies it already holds, and say what the arm carries.

    ``Robot.wrist_bodies`` is what the robot handed its arm when it was built: every body its tree declares for a
    robot ``Robot.from_tree`` built, ``None`` for the arm the sweep builds itself. A body equal field by field to one
    it holds is not handed again, because an arm whose exact mesh guard or planner is built refuses any hand-over, the
    same bodies included. Anything else is handed beside the bodies the arm holds, replacing one of the same camera,
    and an arm that can no longer take it raises: the build fails closed.
    """
    held = _held_by_the_arm(robot)
    links = {body.link_name for body in carried.bodies}
    kept = tuple(body for body in held if getattr(body, "link_name", None) not in links)
    on_the_arm = dataclasses.replace(carried, bodies=kept + tuple(carried.bodies))
    # Only a body the arm does not hold yet is handed. Handing nothing new, the empty tuple included (every declared
    # body excused by a reason), would still be a hand-over, and an arm whose guard is built refuses every one
    # (verification of 2026-09-25: a from_tree robot with its guard built, or one robot swept twice, was refused).
    if any(body not in held for body in carried.bodies):
        on_the_arm.hand_to(robot.arm)
    return on_the_arm.line()


def _held_by_the_arm(robot: Any) -> tuple[Any, ...]:
    """The wrist bodies the robot's arm holds now: what its guard holds, else the robot's record from its build.

    The arm's self-collision guard (``SafetyPreflight.wrist_bodies``) is what the planner and the guard read, so it is
    the answer wherever it can be read: a sweep's own hand-over is in it, and so is a body a sweep replaced.
    ``Robot.wrist_bodies`` is set once, when the robot is built, and stands in only for an arm whose guard cannot be
    read. Counting the record beside the guard (before the review of 2026-09-25) took a body the guard no longer
    held for one it held, so a sweep carrying it again handed nothing.
    """
    preflight = getattr(getattr(robot, "arm", None), "safety_preflight", None)
    reader = getattr(preflight, "wrist_bodies", None)
    if callable(reader):
        try:
            return tuple(reader(robot.arm))
        except Exception as exc:  # noqa: BLE001 (a guard that cannot be read says nothing; the record stands)
            logger.debug("hand-eye: the arm's guard gave no wrist bodies: %s", exc)
    held_by = getattr(robot, "wrist_bodies", None)
    return tuple(held_by.bodies) if held_by is not None else ()


def _on_the_arm(rig: Any) -> bool:
    """Whether the tree declares ``rig`` a camera the arm carries: it declares a body, or eye_in_hand extrinsics."""
    extrinsics = getattr(rig, "extrinsics", None)
    return getattr(rig, "body", None) is not None or getattr(extrinsics, "mounting_mode", None) == "eye_in_hand"


def _together(own: Any, others: Any) -> Any:
    """The bodies an eye in hand sweep's own camera and the tree's other wrist cameras make, as one ``WristBodies``;
    ``None`` when neither was read. The others say which cameras the arm moves without."""
    if own is None or others is None:
        return others if own is None else own
    return dataclasses.replace(others, bodies=tuple(own.bodies) + tuple(others.bodies))
