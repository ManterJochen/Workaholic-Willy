"""Hand-eye calibration of one camera against the robot, as one noun with two verbs.

    from src.config import ConfigTree
    from src.robot.execution.hand_eye import HandEyeCalibration, SweepOptions

    loaded = ConfigTree.from_directory().load()
    calibration = HandEyeCalibration.from_tree(loaded, rig_id="overhead", mode="eye_to_hand",
                                               options=SweepOptions(marker_length_mm=39.7))
    print(calibration.check().render())        # the config alone: builds nothing, opens nothing
    print(calibration.run(dry_run=True).render())   # builds the arm, opens the camera, moves nothing
    report = calibration.run()                 # this moves the robot
    print(report.render())                     # ends with the rig block to paste
    raise SystemExit(report.exit_code)

A correct sweep needs the arm built alone through ``Robot.from_config(robot_config, gripper=None)`` so the
readiness gate runs, one camera through its ``Camera`` owner, the cell lock through ``Robot.connected()``, the wrist
body handed to the arm before it moves, the flange to TCP record read while the arm is connected, the tree's quality
bands, and the rig block to paste. This class is that flow, so a Python caller builds it rather than copying it, and
``python -m src.robot.execution.real_cell.calibrate`` is a caller of it.

What happens, in order:

1. ``check()`` reads the config only: the rig (configured, switched on, RGB-D), and for an eye in hand sweep the
   camera's body. A refusal is a report, never an exception.
2. ``run()`` checks again, then builds: the arm alone and one camera, opened. What the arm refuses (its safety
   attestation) and what world it plans against are on the build report, and ``on_built`` receives it before any
   motion, so a caller can show it while there is still time to stop. ``dry_run=True`` stops here and gives the
   camera back.
3. The sweep connects through ``Robot.connected()``: the cell lock first, then the arm, and no gripper. Every move
   declines the camera world for itself (``CalibrationRoutine``, one reason per mounting), because the sweep is what
   produces the transform a camera world needs. On the way out the arm comes down, the lock is given back, and then
   the camera.
4. The solve's carrier is written into ``out_dir``: ``eth_<rig>.json`` (CAMERA to BASE) or ``eih_<rig>.json``
   (CAMERA to TOOL, with the flange to TCP record when the tool frame is declared ``willy`` or ``polyscope``). The
   dataset is written during the sweep, before the solve, so a solve that fails still leaves the samples.

Writing is part of ``run()`` and not a separate ``save()``: an artifact a caller forgot to save is the silence the
CLI's exit code 2 exists to prevent, and the report says what was written and where.

The routine and the solve are exercised in simulation only, and nothing here has run against a physical
controller.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable

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
    from src.robot.execution.lifecycle import ConnectStage, TeardownReport
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
]

logger = logging.getLogger(__name__)

#: Where the artifact and the dataset land unless the options say otherwise. The sim runners write the same layout,
#: so a cell brought up in sim and then on hardware keeps one place to look.
DEFAULT_OUT_DIR = "calibration/real"
#: How many generated TCP poses a sweep visits unless the options say otherwise, as in sim.
DEFAULT_POSES = 22
#: The ArUco id a sweep poses unless the options say otherwise.
DEFAULT_MARKER_ID = 0
#: The narrowest orientation spread a fiducial sweep runs with, 30 degrees as in the sim runner. A planar ArUco seen
#: near frontally has an IPPE flip ambiguity that ruins the AX=XB rotation.
_MIN_SPREAD_DEG = 30.0

_BANNERS = {
    "config": "=== 1. CONFIG ===",
    "build": "=== 2. BUILD === the arm alone, and one camera",
    "sweep": "=== 3. SWEEP === the robot moves now, keep hands clear",
    "result": "=== 4. RESULT ===",
}


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
    #: The sweep raised. Reported once the arm is down.
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

    ``marker_length_mm`` and ``dict_name`` default to the mode's ``camera.hand_eye`` block. A wrong marker length scales
    every sample uniformly: the solve converges and is uniformly wrong, so measure the printed board.
    ``unmodelled_wrist_body`` is the reason an eye in hand sweep may run while its camera's body cannot be placed
    yet; it is printed and logged, and the sweep then runs with no body in the planner and the guard.

    ``fixed_poses`` replaces the automatic sweep with a caller-chosen one: a list of ``Pose`` (BASE) the caller
    already planned, or a path to a JSON file of them (URPose-shaped or ``Pose``-shaped records). Set, ``poses``
    (the auto-generated count) is not read: nothing is generated when the caller supplies its own poses.
    """

    poses: Maybe[int] = UNSET
    marker_length_mm: Maybe[float] = UNSET
    marker_id: Maybe[int] = UNSET
    dict_name: Maybe[str] = UNSET
    out_dir: "Maybe[str | Path]" = UNSET
    unmodelled_wrist_body: Maybe[str] = UNSET
    fixed_poses: "Maybe[Sequence[Pose] | str | Path]" = UNSET


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
    poses: int = 0
    #: Where the artifact will be written.
    artifact_path: str = ""

    @property
    def ok(self) -> bool:
        return not self.refusal

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The config stage as the CLI prints it. ASCII, no trailing newline, no arguments."""
        if self.refusal:
            return f"[config] REFUSED: {self.refusal}"
        return "\n".join((
            f"  rig        {self.rig_id!r} ({self.rig_source})",
            f"  mode       {self.mode.value}",
            f"  arm        {self.vendor}",
            f"  marker     {self.marker_length_mm:.1f} mm, id {self.marker_id}, {self.dict_name}",
            f"  poses      {'custom, from a JSON file' if self.poses < 0 else self.poses}",
            f"  artifact   {self.artifact_path}",
        ))

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
    #: The wrist body handed to the arm, as ``WristBodies.line()`` says it. Empty when none.
    wrist: str = ""
    #: The decline of an eye in hand sweep that runs without its camera's body. Empty when none.
    unmodelled: str = ""

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
            if self.safety is not None:
                lines.append(self.safety.render())
            lines.append(f"  {self.camera_world}; this sweep declines for itself")
            if self.wrist:
                lines.append(f"  {self.wrist}")
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
            "unmodelled": self.unmodelled,
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
            return "\n".join(lines)
        lines += [
            "",
            CalibrationStage.RESULT.banner(),
            f"  accepted samples  {self.accepted_samples}/{self.check.poses}",
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
            "",
            "Paste this into the camera section so the camera reaches the pick path. Until its rig",
            "declares it, the cell has no CAMERA->BASE for this camera:",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The whole run with its stage banners. ASCII, no trailing newline, no arguments."""
        lines = [CalibrationStage.CONFIG.banner(), self.check.render()]
        if self.build is not None:
            lines += ["", CalibrationStage.BUILD.banner(), self.build.render()]
        if self.swept:
            lines += ["", CalibrationStage.SWEEP.banner()]
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
    wrist: Any = None


@dataclass(frozen=True, slots=True)
class _Parts:
    """What the build stood up, and whether the camera is this run's to give back."""

    robot: Any
    routine: Any
    handle: Any
    owned: bool

    def give_back(self) -> None:
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
    poses: int
    marker_length_mm: float
    marker_id: int
    dict_name: str
    out_dir: str
    sections: _Sections = field(repr=False)
    unmodelled_wrist_body: str | None = None
    #: A caller's own poses, replacing the automatic sweep: a list of Pose (BASE), or a path to a JSON file of
    #: them. None runs the automatic sweep, as every mode did before this existed.
    fixed_poses: "Sequence[Pose] | str | Path | None" = field(default=None, repr=False)
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
                               announce=announce, on_event=on_event, on_built=on_built)

    @classmethod
    def from_config(
        cls, app_config: Any, *, rig_id: str, mode: "MountingMode | str",
        options: Maybe[SweepOptions] = UNSET, data_dir: "str | Path | None" = None,
        announce: "Callable[[ConnectStage], None] | None" = None,
        on_event: "RobotCalibrationEventListener | None" = None,
        on_built: "Callable[[CalibrationBuild], None] | None" = None,
    ) -> "HandEyeCalibration":
        """The sweep a validated config describes for ``rig_id``.

        ``app_config`` is what ``load_config`` returns (its ``robot`` and ``camera`` sections are read). ``data_dir`` is
        the directory it was loaded from, ``None`` for the repository's tree; a wrist camera's body is looked up in
        that tree's registry. ``mode`` has no default: a wrist camera swept eye to hand writes a CAMERA to BASE file for
        a camera that moves.
        """
        return cls.from_parts(robot_config=app_config.robot, camera_config=app_config.camera, rig_id=rig_id,
                              mode=mode, options=options, data_dir=data_dir,
                              announce=announce, on_event=on_event, on_built=on_built)

    @classmethod
    def from_parts(
        cls, *, mode: "MountingMode | str", robot: "Maybe[Robot]" = UNSET, camera: "Maybe[Camera]" = UNSET,
        robot_config: Maybe[Any] = UNSET, camera_config: Maybe[Any] = UNSET, rig_id: Maybe[str] = UNSET,
        settings: Maybe[Any] = UNSET, options: Maybe[SweepOptions] = UNSET, data_dir: "str | Path | None" = None,
        announce: "Callable[[ConnectStage], None] | None" = None,
        on_event: "RobotCalibrationEventListener | None" = None,
        on_built: "Callable[[CalibrationBuild], None] | None" = None,
    ) -> "HandEyeCalibration":
        """A sweep over parts the caller built, or over the sections to build them from at ``run()``.

        ``robot`` is a built ``Robot``; build it with ``gripper=None`` and no cameras, as the config door does, so no
        activation stroke runs beside the board and no live world refuses the sweep's declines. ``robot_config`` is
        the robot section the workspace box, the calibration block and the tool frame are read from; left unset it is
        the tree the arm keeps (``arm.config``). ``camera`` is an owner of the rig to calibrate; ``run()`` opens it
        when it is closed and gives back only what it opened. Without one, ``camera_config`` and ``rig_id`` name the
        rig. ``settings`` left unset is the mode's ``camera.hand_eye`` block, or the schema's default block when the
        camera section has none.

        A missing part raises ``ValueError``: that is a call that cannot describe a sweep, not a refusal.
        """
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
        return cls(
            rig_id=rig,
            mode=selected,
            settings=block,
            poses=int(resolve("poses", chosen_options.poses, DEFAULT_POSES)),
            marker_length_mm=float(resolve("marker_length_mm", chosen_options.marker_length_mm,
                                           block.marker_length_mm)),
            marker_id=int(resolve("marker_id", chosen_options.marker_id, DEFAULT_MARKER_ID)),
            dict_name=str(resolve("dict_name", chosen_options.dict_name, block.aruco_dict_name)),
            out_dir=str(resolve("out_dir", chosen_options.out_dir, DEFAULT_OUT_DIR)),
            sections=_Sections(robot=tree, camera=camera_config),
            unmodelled_wrist_body=reason if chosen(reason) else None,
            # `resolve` rather than `chosen`, because the field is a union: `Maybe[_T]` solved
            # against `Sequence[Pose] | str | Path | _Unset` leaves `_T` ambiguous, and the guard
            # then narrows to a type that still admits `UNSET`. Resolving against `None` states the
            # default once and types cleanly.
            fixed_poses=resolve("fixed_poses", chosen_options.fixed_poses, None),
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
        """The config stage: the rig, the wrist body of an eye in hand sweep, and a robot that cannot sweep."""
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
        refusal = self._robot_refusal()
        if refusal is not None:
            return _Staged(CalibrationCheck(rig_id=self.rig_id, mode=self.mode, refusal=refusal))
        prefix = "eth" if self.mode is MountingMode.EYE_TO_HAND else "eih"
        # A caller's own poses report their count; a JSON path is not read here (check() opens nothing), so -1
        # says "not a generated count" without pretending to know how many the file holds.
        poses = (len(self.fixed_poses) if isinstance(self.fixed_poses, (list, tuple))
                 else -1 if self.fixed_poses is not None else self.poses)
        return _Staged(CalibrationCheck(
            rig_id=str(rig.rig_id), mode=self.mode, rig_source=str(rig.source), vendor=f"{self.sections.robot.vendor}",
            marker_length_mm=self.marker_length_mm, marker_id=self.marker_id, dict_name=self.dict_name,
            poses=poses, artifact_path=f"{self.out_dir}/{prefix}_{rig.rig_id}.json",
        ), rig=rig, wrist=wrist)

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
        try:
            from src.calibration.rgbd_marker_source import RGBDArucoMarkerSource
            from src.camera.orchestration.camera import Camera
            from src.robot.execution.calibration import CalibrationRoutine
            from src.robot.execution.robot import Robot

            # The arm through the builder a pick run uses, so the arm-vendor readiness gate runs first. No gripper: a
            # sweep beside a board needs no activation stroke, and a gripper this tree cannot build is not a reason to
            # refuse calibrating a camera.
            robot = self.robot if chosen(self.robot) else Robot.from_config(self.sections.robot, gripper=None)
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
            marker_source = RGBDArucoMarkerSource(streamer=handle, marker_length_mm=self.marker_length_mm,
                                                  dict_name=self.dict_name, target_id=self.marker_id)
            routine = CalibrationRoutine(
                arm=robot.arm, marker_source=marker_source, eth_calibrator=self._calibrator(),
                workspace_limits=self.sections.robot.workspace_limits, eth_settings=self.settings,
                rig_id=rig.rig_id, marker_id=self.marker_id,
                settle_time_s=self.sections.robot.calibration.settle_time_s,
                calibration_mode=self.mode, on_event=self.on_event,
            )
            intrinsics = handle.get_intrinsics() is not None
            safety = robot.safety()
            camera_world = robot.camera_world_line()
        except Exception as exc:  # noqa: BLE001 (a build refusal is the designed outcome)
            if handle is not None and owned:
                handle.release()
            return CalibrationBuild(refusal=f"{type(exc).__name__}: {exc}"), None
        parts = _Parts(robot=robot, routine=routine, handle=handle, owned=owned)
        gripper = ("none, the sweep connects the arm alone" if robot.gripper is None
                   else type(robot.gripper).__name__)
        wrist_line = unmodelled = refusal = ""
        # The camera on the arm: placed from its previous calibration and handed to the arm before it moves, or swept
        # without a body because the caller said why, which is logged like a decline.
        if staged.wrist is not None:
            try:
                staged.wrist.hand_to(robot.arm)
            except Exception as exc:  # noqa: BLE001 (an arm that cannot hold the camera does not sweep)
                refusal = f"{type(exc).__name__}: {exc}"
            else:
                wrist_line = staged.wrist.line()
        elif self.mode is MountingMode.EYE_IN_HAND and self.unmodelled_wrist_body:
            unmodelled = (f"wrist camera {rig.rig_id!r} swept without its body in the planner and the guard: "
                          f"{self.unmodelled_wrist_body.strip()}")
            logger.warning(unmodelled)
        build = CalibrationBuild(
            refusal=refusal, arm=type(robot.arm).__name__, gripper=gripper, lock_key=robot.lock_key,
            camera=repr(handle), intrinsics=intrinsics, safety=safety, camera_world=camera_world,
            wrist=wrist_line, unmodelled=unmodelled,
        )
        return build, parts

    def _sweep(self, check: CalibrationCheck, build: CalibrationBuild, parts: _Parts) -> CalibrationRunReport:
        """Connect through ``Robot.connected()``, sweep, solve, and write. The camera is given back last."""
        import numpy as np

        from src.robot.execution.cell_lock import CellBusy

        calibration = self.sections.robot.calibration
        failure: Exception | None = None
        result: Any = None
        record: Any = UNSET
        live: Any = None
        # The lock, the connect and the teardown are `Robot.connected()`, the enter and the exit a pick run uses, so
        # this class writes no connect or disconnect of its own. A refused connect is answered before the sweep
        # starts; a sweep that raises is reported once the arm is down.
        try:
            with parts.robot.connected(announce=self.announce) as live:
                try:
                    Path(self.out_dir).mkdir(parents=True, exist_ok=True)
                    dataset_path = f"{self.out_dir}/{self.mode.value}_{check.rig_id}_dataset.json"
                    if self.fixed_poses is not None:
                        # The caller's own poses: a path is a JSON file of them, anything else a list of Pose.
                        if isinstance(self.fixed_poses, (str, Path)):
                            result = parts.routine.run_from_json(self.fixed_poses, dataset_save_path=dataset_path)
                        else:
                            result = parts.routine.run_with_poses(
                                list(self.fixed_poses), dataset_save_path=dataset_path)
                    else:
                        result = parts.routine.run_auto(
                            self.poses,
                            # Tool down, so a board lying face up on the table faces a fixed camera.
                            base_orientation=[float(np.pi), 0.0, 0.0],
                            orientation_spread_deg=max(_MIN_SPREAD_DEG, float(calibration.orientation_spread_deg)),
                            max_attempts_per_pose=calibration.max_attempts_per_pose,
                            seed=0,
                            dataset_save_path=dataset_path,
                        )
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
            # After the block, so the arm is down and the lock given back before the camera is.
            parts.give_back()
        teardown = live.teardown
        if failure is not None:
            return CalibrationRunReport(check=check, outcome=CalibrationOutcome.SWEEP_FAILED, build=build,
                                        failure=f"{type(failure).__name__}: {failure}", teardown=teardown)
        transform = result.transform
        solved = CalibrationRunReport(
            check=check, outcome=CalibrationOutcome.NO_ARTIFACT, build=build, teardown=teardown,
            accepted_samples=int(result.num_samples), rmse_mm=float(result.rmse_mm),
            max_error_mm=float(result.max_error_mm), quality=self._quality(float(result.rmse_mm)),
            transform_mm=_rows(transform.to_matrix()) if transform is not None else None,
            dataset_path=result.dataset_path, camera_worlds=tuple(result.camera_worlds),
        )
        # Both modes: an eye in hand solve with no Transform must not reach a write of nothing. With no carrier
        # nothing is written and the camera keeps whatever calibration it had.
        carrier = result.extrinsics if self.mode is MountingMode.EYE_TO_HAND else transform
        if carrier is None:
            return solved
        written = self._write(result, check.rig_id, record)
        kept = record if chosen(record) and self.mode is MountingMode.EYE_IN_HAND else None
        return dataclasses.replace(
            solved, outcome=CalibrationOutcome.WRITTEN, artifact_path=str(written), flange_to_tcp=kept,
            rig_block=_rig_block(check.rig_id, self.mode.value, str(written)),
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

    On a cell that reads geometry, a rig without a body is refused, and so is a body the registry cannot stand for,
    whatever the reason. A body that cannot be placed yet is refused unless ``reason`` says why the sweep may run
    without it, and the sweep then runs with no wrist body. On any other cell nothing is read.
    """
    from src.calibration.rig_calibration import RigArtifactMissing
    from src.config.cameras import load_camera, tree_camera_refusal
    from src.config.loader import ConfigError
    from src.robot.execution.wrist_bodies import WristBodies, WristBodyRequired
    from src.robot.safety.planning.hand import wrist_body_reader

    reader = wrist_body_reader(robot_cfg)
    key = f"camera.cameras.rigs[{rig.rig_id!r}]"
    if reader is None:
        return None, None
    if getattr(rig, "body", None) is None:
        return None, (f"{key} is swept eye_in_hand and declares no body, and this cell reads geometry ({reader}): the "
                      f"camera's housing would be invisible to the planner and the guard during the sweep. Declare "
                      f"{key}.body")
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
    except WristBodyRequired as exc:
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
