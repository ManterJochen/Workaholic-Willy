"""What each operator function of a cell can do, asked of the arm the tree builds and never connected (P0i i2).

Plan 3.1: every function a cell offers declares what it needs, and a desk says which of them this cell can perform
before anything moves. The declarations are :data:`DECLARATIONS`, one per function the owner named (2026-09-26,
decision 1): the robot's motion verbs, its hand verbs, ``Robot.pick``/``place``, the pick service, hand-eye
calibration with fixed stations and by hand, teaching poses, the IO bench, the planner start and acknowledging a fault.
Each row reads AVAILABLE, REFUSED with the sentence the function itself refuses with, or BENCH for what only the cell
can show. The sentence IS the function's refusal: P0i i3 makes each function ask its row first, and the refusal tape
shows that a UR cell's refusals did not move.

    from src.config import ConfigTree
    from src.robot.execution.cell import Cell

    print(Cell.from_tree(ConfigTree.from_directory(profile="ursim").load()).abilities().render())

Asked of the arm the tree selects, constructed and never connected (``robot_parts.construct_arm``, owner decision 2),
through its capability Protocols, and of the tree for what a desk can read. Nothing connects, opens or starts. A tree
whose arm does not build refuses every row with that build's own refusal. ``abilities_of`` asks the same of handles
already built, which is what ``Robot.abilities()`` does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from src.contracts import UNSET, Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from src.config.schema import CameraConfig
    from src.config.schema.robot import RobotConfig

__all__ = ["DECLARATIONS", "Ability", "AbilityRow", "AbilityStatus", "CellAbilities", "Declaration", "abilities_of",
           "desk_abilities", "row_of"]


class Ability(StrEnum):
    """One operator function of a cell, as the abilities name it."""

    #: ``Robot.move``, ``move_joints`` and ``home``.
    MOVE = "move"
    #: ``Robot.grasp`` and ``release``.
    HAND = "hand"
    #: ``Robot.pick`` and ``place``: the hand verbs around planned motions.
    HANDLE = "handle"
    #: The pick service: ``Cell.build`` and a pick, ``real_cell``, the console's pick.
    PICK = "pick"
    #: Hand-eye calibration over fixed stations.
    CALIBRATE = "calibrate"
    #: Hand-eye calibration with the arm guided by hand (``--freedrive``, ``--adjust``).
    CALIBRATE_BY_HAND = "calibrate by hand"
    #: ``teach_poses``: joint poses taught by guiding the arm.
    TEACH = "teach"
    #: The digital I/O bench.
    IO_BENCH = "io bench"
    #: ``Cell.start_planner``, ``real_cell --start-planner``.
    PLANNER_START = "planner start"
    #: A person clearing a fault at the controller.
    ACKNOWLEDGE_FAULT = "acknowledge fault"


class AbilityStatus(StrEnum):
    """What a desk can say about one function."""

    #: What the function needs of the arm, the hand and the tree is there.
    AVAILABLE = "available"
    #: The function refuses on this cell, with the reason it gives.
    REFUSED = "refused"
    #: Nothing a desk reads refuses it, and what decides it is opened only at the cell.
    BENCH = "bench"


_MARKS = {AbilityStatus.AVAILABLE: "[ ok ]", AbilityStatus.REFUSED: "[ no ]", AbilityStatus.BENCH: "[bench]"}


@dataclass(frozen=True, slots=True)
class AbilityRow:
    """One function, what the desk says of it and why, and what to do about a refusal."""

    ability: Ability
    status: AbilityStatus
    #: The function's own refusal, verbatim, for a REFUSED row; what the answer rests on otherwise.
    reason: str
    #: What to do about it. Empty where the reason says it.
    fix: str = ""
    #: The exception the refusal was read from, where one was raised, so a function that raises in its turn keeps the
    #: chain (``raise ... from row.cause``). Not part of the row's data: left out of ``to_dict`` and of equality.
    cause: "BaseException | None" = field(default=None, compare=False, repr=False)

    @property
    def available(self) -> bool:
        return self.status is AbilityStatus.AVAILABLE

    def to_dict(self) -> dict[str, str]:
        """Plain data, ``json.dumps`` safe."""
        return {"ability": self.ability.value, "status": self.status.value, "reason": self.reason, "fix": self.fix}


@dataclass(frozen=True, slots=True)
class CellAbilities:
    """Every function a cell offers, one row each, in the order of :data:`DECLARATIONS`."""

    rows: tuple[AbilityRow, ...]

    def row(self, ability: Ability | str) -> AbilityRow:
        """The row of ``ability``; a name that is no ability is a ``ValueError``."""
        wanted = Ability(ability)
        return next(row for row in self.rows if row.ability is wanted)

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The rows as the checklist prints its own, then a count. ASCII, no trailing newline.

        A row that answers exactly as the row a run of them began with says so by name rather than printing the same
        sentence again: an arm that does not build refuses all ten rows with one refusal.
        """
        lines = []
        first: AbilityRow | None = None
        for row in self.rows:
            same = (first is not None and first.status is row.status and first.reason == row.reason
                    and first.fix == row.fix)
            if same and first is not None:
                lines.append(f"  {_MARKS[row.status]} {row.ability.value:<26} as {first.ability.value}")
                continue
            first = row
            lines.append(f"  {_MARKS[row.status]} {row.ability.value:<26} {_ascii(row.reason)}")
            if row.fix:
                lines.append(f"           -> {_ascii(row.fix)}")
        counts = {status: sum(1 for row in self.rows if row.status is status) for status in AbilityStatus}
        lines.append("")
        lines.append(f"  {counts[AbilityStatus.AVAILABLE]} available, {counts[AbilityStatus.REFUSED]} refused, "
                     f"{counts[AbilityStatus.BENCH]} decided at the bench.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The machine half of :meth:`render`: the same rows and counts."""
        return {
            "abilities": [row.to_dict() for row in self.rows],
            **{f"n_{status.value}": sum(1 for row in self.rows if row.status is status) for status in AbilityStatus},
        }


def _ascii(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


# ---- what each function needs --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Parts:
    """What a row is asked of: the arm and the hand as built, never connected, and the tree they came from."""

    arm: Any
    gripper: Any
    #: Why no gripper could be built at all, or ``None``.
    gripper_refusal: str | None
    robot_config: "Maybe[RobotConfig]"
    camera: "Maybe[CameraConfig]"
    data_dir: "str | Path | None"


@dataclass(frozen=True, slots=True)
class Declaration:
    """One function: its ability, the calls it covers, what it needs, and how a desk asks for it."""

    ability: Ability
    #: The calls an operator reaches it through.
    functions: tuple[str, ...]
    #: What it needs, in words: the capability Protocols of the arm and the hand, and the tree's facts.
    needs: str
    ask: Callable[[_Parts], AbilityRow]


def _hand_refusal(parts: _Parts) -> str | None:
    """Why the hand verbs refuse this hand, in their own words (``handling._refusal``), or ``None``."""
    if parts.gripper_refusal is not None:
        return parts.gripper_refusal
    if parts.gripper is None:
        return "this robot has no gripper"
    if bool(getattr(parts.gripper, "holds_nothing", False)):
        return f"the gripper in hand ({type(parts.gripper).__name__}) holds nothing: no end-effector was built"
    return None


def _move(parts: _Parts) -> AbilityRow:
    from src.robot.execution.motion import route_of

    route = route_of(parts.arm)
    if not route.runs:
        return AbilityRow(Ability.MOVE, AbilityStatus.REFUSED, route.reason)
    return AbilityRow(Ability.MOVE, AbilityStatus.AVAILABLE, route.render())


def _hand(parts: _Parts) -> AbilityRow:
    why = _hand_refusal(parts)
    if why is not None:
        return AbilityRow(Ability.HAND, AbilityStatus.REFUSED, why,
                          "name the hand this flange carries: robot.gripper.vendor and robot.gripper.model")
    return AbilityRow(Ability.HAND, AbilityStatus.AVAILABLE, f"{type(parts.gripper).__name__} on this arm's flange")


def _handle(parts: _Parts) -> AbilityRow:
    """The hand row, then the move row: ``Robot.pick`` and ``place`` ask exactly those two (P0i i3)."""
    from src.robot.execution.motion import route_of

    for row in (_hand(parts), _move(parts)):
        if not row.available:
            return AbilityRow(Ability.HANDLE, AbilityStatus.REFUSED, row.reason, row.fix)
    return AbilityRow(Ability.HANDLE, AbilityStatus.AVAILABLE,
                      f"{type(parts.gripper).__name__} on an arm whose route is {route_of(parts.arm).route.value.upper()}")


def _pick(parts: _Parts) -> AbilityRow:
    """The refusals the pick service's build and connect give that a desk can read, in the order they give them."""
    from src.config.grippers import tree_hand_refusal
    from src.robot.execution.lifecycle import no_real_gripper_reason

    config = parts.robot_config
    if not chosen(config):
        return AbilityRow(Ability.PICK, AbilityStatus.BENCH,
                          "this robot keeps no tree, so what a pick service would build for it cannot be read")
    # Read as the build read them before it asked this row (P0i i3), no further than each check needs: the tree's
    # hand only against a tree of its own, the world only where the section declares rigs and the selection plans with
    # cuRobo. A caller's partial tree answers here exactly as it answered there.
    refusal = None
    cause: BaseException | None = None
    if parts.data_dir is not None:
        model = config.gripper.model
        refusal = tree_hand_refusal(model, data_dir=parts.data_dir) if model else None
    if refusal is None and chosen(parts.camera):
        from src.robot.execution.camera_world_wiring import CameraWorldPlan
        from src.robot.execution.wrist_bodies import WristBodies, WristBodyRequired

        try:
            WristBodies.from_config(config, parts.camera, data_dir=parts.data_dir)
        except WristBodyRequired as exc:
            refusal, cause = str(exc), exc
        # A calibrated camera makes the world mandatory on a cell that plans with cuRobo (owner, Step 6 Q3 A).
        section = getattr(parts.camera, "cameras", None)
        rigs = getattr(section, "rigs", None)
        selected = getattr(config, "arm", None)  # the selection, bound first: a config tree's `arm` is not a robot arm
        if refusal is None and rigs is not None and selected is not None and selected.planner == "curobo":
            refusal = CameraWorldPlan.from_config(config, rigs,
                                                  primary_rig_id=str(getattr(section, "primary_rig_id", ""))).refusal()
    if refusal is None:
        substitution = getattr(parts.gripper, "substitution", None)
        if substitution is not None:
            refusal = no_real_gripper_reason(substitution)
    if refusal is not None:
        return AbilityRow(Ability.PICK, AbilityStatus.REFUSED, refusal, cause=cause)
    return AbilityRow(Ability.PICK, AbilityStatus.BENCH,
                      "nothing a desk reads refuses it: the cameras and the models open at the build, and the "
                      "controller answers at the connect")


def _sweepable_rigs(camera: "CameraConfig") -> tuple[tuple[str, ...], str]:
    """The rigs a sweep can take, and, where there are none, why each other rig cannot be (``select_rig``)."""
    from src.camera.orchestration.camera import CameraRefused, select_rig

    taken: list[str] = []
    refused: list[str] = []
    for rig in camera.cameras.rigs:
        try:
            select_rig(camera, rig_id=rig.rig_id)
        except CameraRefused as exc:
            refused.append(f"{rig.rig_id} ({exc.reason})")
            continue
        taken.append(rig.rig_id)
    why = "no camera rig a sweep can take: " + (", ".join(refused) if refused else "the camera section declares none")
    return tuple(taken), why


def _calibrate(parts: _Parts) -> AbilityRow:
    if not chosen(parts.camera):
        # Not a refusal: a robot built without its tree has rigs nobody read (review of i2).
        return AbilityRow(Ability.CALIBRATE, AbilityStatus.BENCH, "this robot keeps no tree, so which rig a sweep can take cannot be read")
    rigs, why = _sweepable_rigs(parts.camera)
    if not rigs:
        return AbilityRow(Ability.CALIBRATE, AbilityStatus.REFUSED, why,
                          "switch on the rig in camera.cameras.rigs the calibration is for: an RGB-D rig, or a stereo "
                          "pair once its stereo calibration is written")
    return AbilityRow(Ability.CALIBRATE, AbilityStatus.AVAILABLE,
                      f"fixed stations over {', '.join(rigs)}; each sweep's own check reads its rig, target and "
                      "stations")


def _calibrate_by_hand(parts: _Parts) -> AbilityRow:
    from src.robot.core import SupportsFreedrive

    if not isinstance(parts.arm, SupportsFreedrive):
        return AbilityRow(Ability.CALIBRATE_BY_HAND, AbilityStatus.REFUSED, (
            f"{type(parts.arm).__name__} offers no hand guiding (SupportsFreedrive; the UR teach mode is one), and a "
            "calibration by hand, --freedrive or --adjust, hands the arm to a person. This arm has fixed stations only: "
            "run them with --fixed-poses PATH, without --freedrive or --adjust"))
    if not chosen(parts.camera):
        return AbilityRow(Ability.CALIBRATE_BY_HAND, AbilityStatus.BENCH, "this robot keeps no tree, so which rig a sweep can take cannot be read")
    rigs, why = _sweepable_rigs(parts.camera)
    if not rigs:
        return AbilityRow(Ability.CALIBRATE_BY_HAND, AbilityStatus.REFUSED, why,
                          "switch on an RGB-D rig in camera.cameras.rigs, the one the calibration is for")
    return AbilityRow(Ability.CALIBRATE_BY_HAND, AbilityStatus.AVAILABLE,
                      f"{type(parts.arm).__name__} is guided by hand (SupportsFreedrive), over {', '.join(rigs)}")


def _named(arm: Any) -> str:
    """The arm's class and the vendor it reports, as the teach refusal names them (moved from ``teach``, P0i i3)."""
    from src.robot.core import RobotCapabilities

    capabilities = getattr(arm, "capabilities", None)
    if isinstance(capabilities, RobotCapabilities):
        return f"{type(arm).__name__} (vendor {capabilities.vendor!r})"
    return f"{type(arm).__name__} (which reports no vendor)"


def _teach(parts: _Parts) -> AbilityRow:
    from src.robot.core import SupportsFreedrive

    if not isinstance(parts.arm, SupportsFreedrive):
        return AbilityRow(Ability.TEACH, AbilityStatus.REFUSED, (
            f"{_named(parts.arm)} offers no hand guiding (SupportsFreedrive; the UR teach mode is one), so no pose can "
            "be taught by hand on it: read its joints off its own pendant instead. Nothing was asked and nothing was "
            "freed"))
    return AbilityRow(Ability.TEACH, AbilityStatus.AVAILABLE,
                      f"{type(parts.arm).__name__} is guided by hand (SupportsFreedrive)")


def _io_bench(parts: _Parts) -> AbilityRow:
    from src.robot.core import SupportsDigitalIO

    if not isinstance(parts.arm, SupportsDigitalIO):
        return AbilityRow(Ability.IO_BENCH, AbilityStatus.REFUSED, (
            f"{type(parts.arm).__name__} does not advertise SupportsDigitalIO: it offers no digital I/O for this bench "
            "to read or drive."))
    return AbilityRow(Ability.IO_BENCH, AbilityStatus.AVAILABLE,
                      f"{type(parts.arm).__name__} switches digital I/O (SupportsDigitalIO)")


def _planner_start(parts: _Parts) -> AbilityRow:
    from src.robot.core import StartsAPlanner

    if not isinstance(parts.arm, StartsAPlanner):
        return AbilityRow(Ability.PLANNER_START, AbilityStatus.REFUSED,
                          f"{type(parts.arm).__name__} has no planner a desk can start (StartsAPlanner)")
    return AbilityRow(Ability.PLANNER_START, AbilityStatus.BENCH, (
        f"{type(parts.arm).__name__} has a planner a desk can start (StartsAPlanner): whether it starts is what the "
        "planner start itself answers, with every refusal the first planned move meets"))


def _acknowledge_fault(parts: _Parts) -> AbilityRow:
    from src.robot.core import SupportsRobotStatus, SupportsWatching

    if not isinstance(parts.arm, SupportsRobotStatus):
        return AbilityRow(Ability.ACKNOWLEDGE_FAULT, AbilityStatus.REFUSED, (
            f"{type(parts.arm).__name__} reads no controller state and acknowledges no fault (SupportsRobotStatus): "
            "a stop on it is cleared on its own controller"))
    if not isinstance(parts.arm, SupportsWatching):
        # Measured on URSim, 2026-09-26: a full connect is refused by the very stop the door is for (P0i).
        return AbilityRow(Ability.ACKNOWLEDGE_FAULT, AbilityStatus.REFUSED, (
            f"{type(parts.arm).__name__} reads its controller's state only over a connection (SupportsWatching), and "
            "a controller in a protective stop refuses one: clear the stop at its own pendant"))
    return AbilityRow(Ability.ACKNOWLEDGE_FAULT, AbilityStatus.AVAILABLE, (
        f"{type(parts.arm).__name__} lets a person acknowledge a fault at the cell (SupportsRobotStatus); never "
        "automatically: python -m src.robot.faults acknowledge"))


#: Every operator function, in the order the rows are read (P0i decision 1).
DECLARATIONS: tuple[Declaration, ...] = (
    Declaration(Ability.MOVE, ("Robot.move", "Robot.move_joints", "Robot.home"),
                "a route that runs: the arm says what it keeps of a path (KeepsLines), and cuRobo with the exact mesh "
                "guard judges it (PlansMotions.plans_paths), or it is a desk arm; a connection", _move),
    Declaration(Ability.HAND, ("Robot.grasp", "Robot.release"),
                "a gripper that holds something (not a substitute); a connection", _hand),
    Declaration(Ability.HANDLE, ("Robot.pick", "Robot.place"),
                "what the hand verbs need, and a route that runs", _handle),
    Declaration(Ability.PICK, ("Cell.build", "AutonomousGraspService.pick", "real_cell", "console pick"),
                "the tree's own hand as the registry's, wrist bodies it can place, a live camera world where cuRobo "
                "plans and a camera is calibrated, a real gripper; the cameras and models at the build", _pick),
    Declaration(Ability.CALIBRATE, ("HandEyeCalibration.run", "real_cell.calibrate"),
                "an enabled RGB-D rig (select_rig)", _calibrate),
    Declaration(Ability.CALIBRATE_BY_HAND, ("real_cell.calibrate --freedrive", "real_cell.calibrate --adjust"),
                "an arm a person can guide (SupportsFreedrive), and an enabled RGB-D rig", _calibrate_by_hand),
    Declaration(Ability.TEACH, ("teach_poses",), "an arm a person can guide (SupportsFreedrive); a connection", _teach),
    Declaration(Ability.IO_BENCH, ("python -m src.robot.execution.io_bench",),
                "an arm that switches digital I/O (SupportsDigitalIO); the cell lock; a connection", _io_bench),
    Declaration(Ability.PLANNER_START, ("Cell.start_planner", "real_cell --start-planner"),
                "an arm with a planner a desk can start (StartsAPlanner)", _planner_start),
    Declaration(Ability.ACKNOWLEDGE_FAULT, ("python -m src.robot.faults acknowledge",
                                            "FaultAcknowledgement.run"),
                "an arm that reads its controller's state and lets a person acknowledge a fault (SupportsRobotStatus), "
                "over a link that starts no control program (SupportsWatching); a person at the cell", _acknowledge_fault),
)


def row_of(
    ability: Ability | str, *, arm: Any, gripper: Any = None, robot_config: "Maybe[RobotConfig]" = UNSET,
    camera: "Maybe[CameraConfig]" = UNSET, data_dir: "str | Path | None" = None,
) -> AbilityRow:
    """One function's row, asked of handles already built: what the function asks before its first command (P0i i3).

    The function refuses with the row's reason, so what the desk says of it and what it says itself are one sentence.
    """
    wanted = Ability(ability)
    parts = _Parts(arm=arm, gripper=gripper, gripper_refusal=None, robot_config=robot_config, camera=camera,
                   data_dir=data_dir)
    return next(declaration for declaration in DECLARATIONS if declaration.ability is wanted).ask(parts)


def abilities_of(
    arm: Any, gripper: Any, *, robot_config: "Maybe[RobotConfig]" = UNSET, camera: "Maybe[CameraConfig]" = UNSET,
    data_dir: "str | Path | None" = None,
) -> CellAbilities:
    """Every row, asked of handles already built (connected or not), and of the tree where one is handed in."""
    parts = _Parts(arm=arm, gripper=gripper, gripper_refusal=None, robot_config=robot_config, camera=camera,
                   data_dir=data_dir)
    return CellAbilities(rows=tuple(declaration.ask(parts) for declaration in DECLARATIONS))


def desk_abilities(
    robot_config: "RobotConfig", *, camera: "Maybe[CameraConfig]" = UNSET, data_dir: "str | Path | None" = None,
) -> CellAbilities:
    """Every row for the arm and the hand ``robot_config`` builds, constructed and never connected.

    An arm that does not build refuses every row with its own refusal; a gripper that does not build refuses the rows
    that need a hand with its own. Nothing connects, opens or starts.
    """
    from src.robot.execution.robot_parts import build_gripper, construct_arm

    try:
        arm = construct_arm(robot_config)
    except Exception as exc:  # noqa: BLE001 - an arm that does not build is every row's answer, not a crash
        why = f"the arm this tree selects does not build: {type(exc).__name__}: {exc}"
        return CellAbilities(rows=tuple(AbilityRow(d.ability, AbilityStatus.REFUSED, why) for d in DECLARATIONS))
    gripper: Any = None
    gripper_refusal: str | None = None
    try:
        gripper = build_gripper(robot_config, arm=arm)
    except Exception as exc:  # noqa: BLE001 - a hand that does not build is its rows' answer
        gripper_refusal = f"the gripper this tree names does not build: {type(exc).__name__}: {exc}"
    parts = _Parts(arm=arm, gripper=gripper, gripper_refusal=gripper_refusal, robot_config=robot_config,
                   camera=camera, data_dir=data_dir)
    return CellAbilities(rows=tuple(declaration.ask(parts) for declaration in DECLARATIONS))
