"""Everything about a real cell that can be checked before the arm is asked to move.

The "no surprises at the bench" half of the real-cell runner. Every check here answers a question
that would otherwise be answered by a physical robot doing something wrong, or is a fail-closed
refusal that would otherwise fire deep inside ``connect()`` with less context.

Nothing in this module touches the network, the SDK or the arm. It reads config and reports, so an
operator gets a list they can fix at a desk in one pass instead of discovering the same list one
crash at a time with a powered robot in front of them.

These are config assertions, not hardware measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from src.contracts import UNSET, Maybe, chosen
from src.robot.safety.planning import curobo_env_available

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema import CameraConfig
    from src.config.schema.robot import RobotConfig
    from src.robot.safety.planning._hand_placement import HandPlacement
    from src.robot.safety.planning.hand import PlannerHand

__all__ = ["CheckStatus", "PreflightCheck", "PreflightReport", "run_config_preflight"]


class CheckStatus(StrEnum):
    """Outcome of one preflight check."""

    #: Verified good.
    OK = "ok"
    #: Will stop the cell. The runner refuses to proceed.
    BLOCK = "block"
    #: Works, but an operator should know. Never blocks.
    WARN = "warn"
    #: Cannot be decided from config alone; it needs the hardware.
    BENCH = "bench"


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One question, its verdict, and what to do about it."""

    name: str
    status: CheckStatus
    detail: str
    #: The concrete fix. Empty for OK.
    fix: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status.value, "detail": self.detail, "fix": self.fix}


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """The full checklist."""

    checks: tuple[PreflightCheck, ...]

    @property
    def blocking(self) -> tuple[PreflightCheck, ...]:
        return tuple(c for c in self.checks if c.status is CheckStatus.BLOCK)

    @property
    def ok(self) -> bool:
        return not self.blocking

    @property
    def exit_code(self) -> int:
        """0 when nothing blocks, 1 when something does.

        1 is `_EXIT_CONFIG`, and this is the `--check` rule rather than the run rule. `main` in
        `real_cell/__main__.py` continues past a blocking checklist under `--rehearse`, because a
        rehearsal exists to exercise the path on a desk where blocking items are expected. That
        waiver is a policy the runner applies on top of this verdict, not a second opinion about
        the verdict.
        """
        return 0 if self.ok else 1

    def to_dict(self) -> dict[str, object]:
        """The machine half, `json.dumps`-safe with no custom encoder.

        It lives here rather than in an HTTP layer, so a library caller who wants a preflight as
        data does not have to import a web framework to get it. The library layer may not depend
        on one.

        `profile` and `vendor` are deliberately absent: neither is on this report and neither is
        derivable from it, because the profile chain is a property of how the config was loaded
        and the vendor of the config itself. A caller that has both should add them, because a
        verdict without its chain is unreadable.
        """
        return {
            "checks": [c.to_dict() for c in self.checks],
            "ok": self.ok,
            "exit_code": self.exit_code,
            "n_blocking": len(self.blocking),
            "n_warn": sum(1 for c in self.checks if c.status is CheckStatus.WARN),
            # Counted separately because `bench` is not a milder warning: it is a question no
            # software can answer (is the controller in Remote Control, is the vacuum's 24 V
            # actually on) and it stays on the list after every other row is green.
            "n_bench": sum(1 for c in self.checks if c.status is CheckStatus.BENCH),
        }

    def render(self) -> str:
        """A human-readable checklist. The bench copy of this is what an operator works through."""
        marks = {CheckStatus.OK: "[ ok ]", CheckStatus.BLOCK: "[BLOCK]",
                 CheckStatus.WARN: "[warn]", CheckStatus.BENCH: "[bench]"}
        lines = []
        for c in self.checks:
            lines.append(f"  {marks[c.status]} {c.name:<26} {c.detail}")
            if c.fix:
                lines.append(f"           -> {c.fix}")
        n_block = len(self.blocking)
        lines.append("")
        lines.append(
            f"  {n_block} blocking, "
            f"{sum(1 for c in self.checks if c.status is CheckStatus.WARN)} warnings, "
            f"{sum(1 for c in self.checks if c.status is CheckStatus.BENCH)} deferred to the bench."
        )
        return "\n".join(lines)


def _exact_mesh_engine_row(robot_cfg: "RobotConfig", collision_engine: "Maybe[str | None]") -> PreflightCheck:
    """Whether this cell could judge a path at all, in the path gate's own words.

    BLOCK on a cell that plans, because there every motion goes through a gate that would refuse.
    WARN on an ik cell, which plans no path: it loses the check rather than every move.
    """
    from src.robot.safety._fcl_self_collision import mesh_backend_status
    from src.robot.safety.planning.environment import import_collision_engine
    from src.robot.safety.preflight import exact_mesh_path_refusal, no_path_guard_refusal

    plans = str(getattr(robot_cfg.ur, "motion_planner", "ik")) == "curobo"
    refused = CheckStatus.BLOCK if plans else CheckStatus.WARN
    fix = (
        "set safety.self_collision.backend: fcl, install Coal (or python-fcl) and bake this arm's mesh bundle "
        "(scripts/isaac/bake_ur_collision_meshes.py). Until then no path on this cell is judged on the geometry it "
        "actually has"
    )

    self_collision = robot_cfg.safety.self_collision
    if not getattr(self_collision, "enforce", True):
        return PreflightCheck("exact mesh engine", refused, no_path_guard_refusal(),
                              "set safety.self_collision.enforce: true")
    if str(getattr(self_collision, "backend", "capsule")) != "fcl":
        return PreflightCheck("exact mesh engine", refused, exact_mesh_path_refusal(), fix)

    engine = collision_engine if chosen(collision_engine) else import_collision_engine()[1]
    model = getattr(self_collision, "kinematics_model", None) or getattr(robot_cfg.ur, "model", "")
    hand = getattr(getattr(robot_cfg, "gripper", None), "model", None)
    token = mesh_backend_status(str(model), getattr(self_collision, "mesh_dir", None), hand)
    if engine is None or token != "ok":
        # The fix follows the token, so a missing or refused hand bundle is not told to bake the arm.
        from src.robot.safety._fcl_self_collision import _STATUS_HINTS

        if token in ("no_hand_bundle", "hand_bundle_refused"):
            fix = _STATUS_HINTS[token].replace("{hand}", str(hand))
        elif token == "no_bundle":
            fix = ("bake this arm's bundle from Universal Robots' own collision STLs with "
                   "scripts/isaac/bake_ur_meshes_from_urdf.py. Until then no path on this cell is judged on the geometry "
                   "it actually has")
        return PreflightCheck("exact mesh engine", refused, exact_mesh_path_refusal(),
                              f"{fix} (engine {engine or 'none'}, bundle {token})")
    said = _hand_geometry_source(str(hand), getattr(self_collision, "mesh_dir", None)) if hand else None
    plates = _unmodelled_plates(robot_cfg)
    return PreflightCheck(
        "exact mesh engine", CheckStatus.OK,
        f"{engine} judges every path on the exact meshes, bundle {token} for {model} with {hand or 'the arm bundle'}"
        + (f"; {said}" if said else "")
        + (f"; {plates}" if plates else ""),
    )


def _unmodelled_plates(robot_cfg: "RobotConfig") -> str | None:
    """The plates that hold the hand out and are in nobody's collision model, by name.

    A plate declares a thickness, which places the hand, and may declare a cross section, which
    makes it a body. One without is not refused: naming it is chosen over inventing a width, because
    a body built from one measured number and one guessed one is indistinguishable downstream from a
    measured one. So it is said here, where an operator reads what the cell is made of.

    The cost is measured by ``scripts/curobo/probe_plate_body.py``: over the 1,483 judged poses,
    against the exact guard's 10 mm, a plate of UR flange radius turns 0 to 2 of about 950 clear
    poses per arm, and a quick change coupler 0 to 25. So this is a sentence rather than a WARN row,
    and a tool changer is the case to measure.
    """
    plates = getattr(getattr(robot_cfg, "gripper", None), "coupling_plates", None) or ()
    bare = [plate.name for plate in plates if getattr(plate, "cross_section_mm", None) is None]
    if not bare:
        return None
    return (f"{', '.join(bare)} holds the hand out and is in no collision model: declare cross_section_mm to make "
            f"it a body")


def _hand_geometry_source(hand: str, mesh_dir: "str | None" = None) -> str | None:
    """How this cell's hand is modelled, where that is worth a sentence, or ``None`` where it is not.

    A hand written from its registry dimensions is planned and guarded like any other, so this is
    never a refusal. It is said because it costs room: the two shipped hands that carry both a
    bundle and a full set of dimensions needed 5.73 mm and 9.50 mm of inflation before an envelope
    enclosed them. A scanned hand adds nothing to the row, and neither does a hand with no bundle of
    its own.
    """
    from src.robot.safety.planning.environment import hand_provenance

    provenance = hand_provenance(hand, mesh_dir)
    return provenance.render() if provenance is not None and provenance.from_dimensions else None


def _planner_margin_row(
    robot_cfg: "RobotConfig", evidence_dir: "Maybe[Path]", data_dir: "str | Path | None",
) -> PreflightCheck:
    """Whether this cell's planner may start, by the committed evidence for its combination.

    OK where a file measured this cell's combination at b1, and the row states the planner margin
    against the guard's rather than warning about it: every shipped cell plans at 4 mm against a
    10 mm guard, measured and decided, and a warning on every cell teaches an operator to read past
    warnings. BLOCK where no file admits it. WARN on an ik cell, which runs no planner and so has no
    model to hold against the guard: the exact mesh guard alone decides there.
    """
    from src.robot.safety.planning.evidence import desk_evidence_refusal
    from src.robot.safety.planning.margin import declared_planner_margin

    sc = robot_cfg.safety.self_collision
    if str(getattr(robot_cfg.ur, "motion_planner", "ik")) != "curobo":
        return PreflightCheck(
            "planner margin", CheckStatus.WARN,
            "no planner runs on this cell, so there is no planner model to hold against the guard: the exact "
            f"mesh guard alone decides every motion, keeping {float(sc.min_distance_mm):g} mm clear",
            "set robot.ur.motion_planner: curobo for a cell whose paths should be planned around what it "
            "carries, and measure its combination with scripts/curobo/matrix_gate.py",
        )
    declared = declared_planner_margin(sc)
    if not chosen(declared):
        return PreflightCheck(
            "planner margin", CheckStatus.BLOCK,
            "safety.self_collision.planner_margin_mm is undeclared, and a cuRobo cell refuses to start a "
            "planner without one: undeclared is not zero",
            "declare it in this cell's profile; the UR family was measured at 4 mm, where a retract is found "
            "for every arm, and at 8 and 10 mm the UR5 and the UR10 have no pose at all",
        )
    refused, evidence = desk_evidence_refusal(
        robot_cfg, evidence_dir=evidence_dir if chosen(evidence_dir) else None, data_dir=data_dir,
    )
    if refused is not None or evidence is None:
        return PreflightCheck(
            "planner margin", CheckStatus.BLOCK, refused or "no evidence admits this cell",
            "run the scripts/curobo/matrix_gate.py command the detail ends with, on a box with the cuRobo environment, and commit the file it "
            "writes; a cell does not start a planner nobody measured",
        )
    measured = evidence.measured
    return PreflightCheck(
        "planner margin", CheckStatus.OK,
        f"{evidence.path.name} admits {evidence.arm} with {evidence.hand} at {evidence.criterion}: the "
        f"planner keeps {evidence.planner_margin_mm:g} mm and the exact guard {evidence.guard_margin_mm:g} mm, "
        f"measured over {measured.poses:,} poses with {measured.false_clears:,} false clears. The hashes the "
        f"sidecar reports about what it loaded are checked when it starts",
    )


#: How far the declared grasp centre may sit from the hand's own before the row says so.
_GRASP_CENTRE_TOLERANCE_MM = 1.0


def _camera_world_row(robot_cfg: "RobotConfig", camera: "Maybe[CameraConfig]") -> PreflightCheck:
    """Whether the cameras of a UR cell give its planner a live world, read as the build wires it.

    The same ``CameraWorldPlan`` the build reads, so a BLOCK here is the refusal the build gives. It
    opens nothing.
    """
    from src.robot.execution.camera_world_wiring import CameraWorldPlan

    name = "camera world"
    planner = getattr(getattr(robot_cfg, "ur", None), "motion_planner", None)
    if not chosen(camera):
        return PreflightCheck(
            name, CheckStatus.BLOCK,
            "this checklist was not handed the camera section, so it cannot say whether the planner gets a live "
            "camera world, and on a cuRobo cell every motion needs one or a decline",
            "run it with the camera section of the same tree, as Cell.preflight and the operator console do",
        )
    if planner != "curobo":
        return PreflightCheck(
            name, CheckStatus.WARN,
            f"robot.ur.motion_planner is {planner!r}, so no planner reads a camera world; the cuRobo environment row "
            "says what such a cell gives up",
        )
    section = camera.cameras
    plan = CameraWorldPlan.from_config(robot_cfg, section.rigs, primary_rig_id=section.primary_rig_id)
    if plan.rig_ids:
        return PreflightCheck(name, CheckStatus.OK, plan.render())
    refusal = plan.refusal()
    head = refusal if refusal is not None else (
        f"this cell plans with cuRobo and its cameras give no live world: {plan.reason}")
    return PreflightCheck(
        name, CheckStatus.BLOCK,
        f"{head}. The pick service declines nothing, so every pick motion would be refused before it moves",
        "calibrate the primary camera and enable safety.planning_world with a measured support_plane and "
        "perceived.enabled; motions from your own code may decline with robot.without_camera_world(\"<why>\")",
    )


def _wrist_camera_body_row(
    robot_cfg: "RobotConfig", camera: "Maybe[CameraConfig]", data_dir: "str | Path | None", *, is_real: bool,
) -> PreflightCheck:
    """Which wrist cameras the arm carries, placed as the build places them, or why one cannot be.

    The same resolution the build, ``Robot`` and the planner start run (``execution.wrist_bodies``), so a
    BLOCK here is the refusal those doors give. Only a real cell blocks; a sim or a dummy tree says the
    same thing as a warning.
    """
    from src.robot.execution.wrist_bodies import WristBodies, WristBodyRequired

    name = "wrist camera body"
    stop = CheckStatus.BLOCK if is_real else CheckStatus.WARN
    if not chosen(camera):
        return PreflightCheck(
            name, stop, "this checklist was handed no camera section, so it cannot say whether the arm carries a camera",
            "pass the camera section of the same tree (Cell.preflight and the CLI do)",
        )
    try:
        wrist = WristBodies.from_config(robot_cfg, camera, data_dir=data_dir)
    except WristBodyRequired as exc:
        return PreflightCheck(name, stop, str(exc), "declare, calibrate or correct the rig this names, as it says")
    if wrist.reader is None:
        return PreflightCheck(name, CheckStatus.OK, wrist.line())
    if not wrist.bodies:
        return PreflightCheck(name, CheckStatus.OK,
                              "the arm carries no camera: no rig declares camera.cameras.rigs[<id>].body and no enabled "
                              "rig is eye_in_hand")
    detail = "; ".join(
        f"{body.render()}. Carried by the planner as {body.link_name}, by the exact guard against wrist_2 and every "
        "link further in and every fixture, and by the self filter"
        for body in wrist.bodies
    )
    return PreflightCheck(
        name, CheckStatus.OK,
        f"{detail}. Cover certified. The combination evidence the planner margin row names is keyed without the "
        "camera; the planner start proves the cover it loaded",
    )


def _grasp_centre_row(robot_cfg: "RobotConfig", hand: "PlannerHand", placement: "HandPlacement") -> PreflightCheck:
    """Whether the declared tool frame puts the grasp centre where the hand's registry number and plates do.

    Along the approach the declared offset should be ``jaw.grasp_centre_mm`` plus the plates, and
    across it zero: a body written from dimensions is built around that number, so a frame elsewhere
    commands a grasp centre the guard and the planner do not model. WARN beyond 1 mm, not BLOCK: the
    2F-85's registry number is an estimate every shipped 2F-85 profile disagrees with, and it is
    measured before this row may stop a cell.
    """
    import math

    approach = placement.approach_in_tool0
    offset = tuple(float(v) for v in robot_cfg.gripper.tool_frame.offset_mm)
    along = sum(o * a for o, a in zip(offset, approach))
    across = math.sqrt(max(0.0, sum(o * o for o in offset) - along * along))
    expected = float(hand.jaw.grasp_centre_mm) + float(hand.coupling_mm)
    plates = f" plus {float(hand.coupling_mm):g} mm of plates" if float(hand.coupling_mm) else ""
    registry = f"config/grippers/{hand.model}.yaml"
    detail = (
        f"robot.gripper.tool_frame.offset_mm puts the grasp centre {along:g} mm along the approach "
        f"{placement.approach} and {across:.3g} mm across it; {hand.model}'s grasp_centre_mm "
        f"{float(hand.jaw.grasp_centre_mm):g} mm{plates} puts it at {expected:g} mm ({registry})"
    )
    off = max(abs(along - expected), across)
    if off <= _GRASP_CENTRE_TOLERANCE_MM:
        return PreflightCheck("grasp centre", CheckStatus.OK, detail)
    return PreflightCheck(
        "grasp centre", CheckStatus.WARN,
        f"{detail}: {off:.3g} mm apart, beyond the {_GRASP_CENTRE_TOLERANCE_MM:g} mm this row admits",
        f"measure flange to grasp centre on the bench. If the bench agrees with {registry}, correct "
        f"robot.gripper.tool_frame.offset_mm; if it agrees with the declared frame, correct grasp_centre_mm in {registry} "
        f"and rewrite the hand's body from it, because a body written from dimensions is placed around that number",
    )


def _gripper_driver_row(robot_cfg: "RobotConfig") -> PreflightCheck:
    """The driver the build constructs for this gripper on the UR arm the config builds.

    Asked of :func:`~src.robot.execution.robot_parts.gripper_driver_verdict`, the function the build
    asks, with the facts of that arm: the UR driver reports vendor ``ur``, switches digital I/O and
    keeps its tree's address. So it blocks exactly where the build would substitute a
    ``NullGripper``, in the build's own words.
    """
    from src.robot.core import GripperVendor
    from src.robot.execution.robot_parts import gripper_driver_verdict

    verdict = gripper_driver_verdict(robot_cfg, arm_vendor="ur", arm_has_digital_io=True, arm_ip=robot_cfg.ur.ip)
    hand = robot_cfg.gripper.model
    named = f" for {hand}" if hand else ""
    if verdict.reason is not None:
        return PreflightCheck(
            "gripper driver", CheckStatus.BLOCK,
            f"the build would put a NullGripper on this arm{named}, and the connect refuses it: {verdict.detail}",
            verdict.fix,
        )
    if verdict.vendor in (GripperVendor.NONE, GripperVendor.DUMMY):
        what = "nothing" if verdict.vendor is GripperVendor.NONE else "a stand-in that actuates nothing"
        return PreflightCheck(
            "gripper driver", CheckStatus.WARN,
            f"robot.gripper.vendor is {verdict.vendor.value!r}{named}: this real arm is built with {what} on its flange, "
            f"so a pick closes no jaws",
            "set robot.gripper.vendor to the driver that actuates the hand on this flange, when one is fitted",
        )
    return PreflightCheck(
        "gripper driver", CheckStatus.OK,
        f"robot.gripper.vendor {verdict.vendor.value!r} builds the {verdict.vendor.value} driver{named}",
    )


def _end_effector_wiring_row(robot_cfg: "RobotConfig") -> PreflightCheck:
    """What to confirm physically for the configured gripper driver, which no API reports."""
    gripper = robot_cfg.gripper
    vendor = str(getattr(gripper.vendor, "value", gripper.vendor)).lower()
    if vendor == "robotiq":
        return PreflightCheck(
            "end-effector wiring", CheckStatus.BENCH,
            "the Robotiq URCap on the controller (it opens port 63352, not any pip package)",
            "confirm physically; a missing URCap looks like a gripper that never answers",
        )
    if vendor == "vacuum":
        vac = gripper.vacuum
        extra = "".join((
            f", blow off on output {vac.blow_off_output_pin}" if vac.blow_off_output_pin is not None else "",
            f", vacuum switch on input {vac.vacuum_ok_input_pin}" if vac.vacuum_ok_input_pin is not None else "",
        ))
        return PreflightCheck(
            "end-effector wiring", CheckStatus.BENCH,
            f"the vacuum ejector on {vac.io_port} output {vac.vacuum_output_pin}{extra}, and the solenoid's 24 V supply",
            "confirm physically, pin by pin; an ejector with no supply looks like a cup that never seals",
        )
    if vendor == "jaw_io":
        jaw = gripper.jaw_io
        opens = (f"open output {jaw.open_output_pin}" if jaw.open_output_pin is not None
                 else f"open by dropping close output {jaw.close_output_pin}")
        return PreflightCheck(
            "end-effector wiring", CheckStatus.BENCH,
            f"the jaw solenoid ({jaw.actuation}) on {jaw.io_port} close output {jaw.close_output_pin}, {opens}, "
            f"and its 24 V supply",
            "confirm physically, pin by pin; a swapped pair opens the jaws on a grip command",
        )
    if vendor == "onrobot":
        rg = gripper.onrobot
        return PreflightCheck(
            "end-effector wiring", CheckStatus.BENCH,
            f"the OnRobot Compute Box at {rg.host}:{rg.port}, unit {rg.unit_id}, a separate device on the network",
            "confirm the box answers on that address; it is not the arm's controller",
        )
    return PreflightCheck(
        "end-effector wiring", CheckStatus.BENCH,
        f"robot.gripper.vendor is {vendor!r}, so no end-effector is actuated through this cell",
        "confirm nothing on the flange needs actuating, or name its driver (the gripper driver row says which build)",
    )


def _carried_part_row(robot_cfg: "RobotConfig", *, is_real: bool) -> PreflightCheck:
    """Whether the planner and the self filter model the part a grasp carries, and why not.

    BLOCK on a real cuRobo UR with the payload enabled and no length: every carry would be planned as
    if the hand were empty. OK names the length and the attach slots the planner reserves. WARN when
    the payload is disabled, or the planner is not cuRobo, naming what then goes unmodelled.
    """
    from src.robot.safety.planning.reservation import PlannerReservation

    name = "carried part"
    planner = str(getattr(getattr(robot_cfg, "ur", None), "motion_planner", ""))
    payload = robot_cfg.safety.planning_world.payload
    if planner != "curobo":
        return PreflightCheck(name, CheckStatus.WARN, (
            f"robot.ur.motion_planner is {planner!r}: no planner carries a part, so a lift or a transit after a grasp "
            "is judged as if the hand were empty"),
            "set robot.ur.motion_planner to curobo and declare safety.planning_world.payload.length_mm, where a "
            "carried part should be modelled")
    if not payload.enabled:
        return PreflightCheck(name, CheckStatus.WARN, (
            "safety.planning_world.payload.enabled is false: neither the planner nor the self filter models a part in "
            "the gripper, so a carry is planned as if the hand were empty and the part stays in what the cameras see"),
            "enable it and declare safety.planning_world.payload.length_mm, unless this cell never carries a part")
    if payload.length_mm is None:
        return PreflightCheck(name, CheckStatus.BLOCK if is_real else CheckStatus.WARN, (
            "safety.planning_world.payload.length_mm is undeclared: the payload is modelled, and no length is "
            "implied, so every carry would be planned as if the hand were empty"),
            "declare safety.planning_world.payload.length_mm, how far the longest part this cell carries hangs past "
            "the fingertips along the approach, or set enabled: false if it never carries one")
    slots = PlannerReservation.from_config(robot_cfg=robot_cfg).sphere_slots
    return PreflightCheck(name, CheckStatus.OK, (
        f"a part {float(payload.length_mm):g} mm past the fingertips, {float(payload.lateral_margin_mm):g} mm lateral "
        f"margin, {slots} attach slot(s) the planner reserves and its evidence names"))


def _vendor(robot_cfg: "RobotConfig") -> str:
    v = getattr(robot_cfg, "vendor", "")
    return str(getattr(v, "value", v)).lower()


def run_config_preflight(
    robot_cfg: "RobotConfig",
    *,
    camera: "Maybe[CameraConfig]" = UNSET,
    curobo_available: Maybe[bool] = UNSET,
    collision_engine: "Maybe[str | None]" = UNSET,
    evidence_dir: "Maybe[Path]" = UNSET,
    data_dir: "str | Path | None" = None,
) -> PreflightReport:
    """Check a ``RobotConfig`` for everything that stops a real pick, without touching hardware.

    ``data_dir`` is the config tree ``robot_cfg`` came from, ``None`` for the repository's. The hand
    row and the planner margin row resolve the hand against it, so a tree that describes the named
    hand differently from the repository's registry is a BLOCK here, with both files named, rather
    than an OK the build then refuses.

    ``camera`` is the camera section of the same tree. The camera to base row reads the primary
    rig's declared calibration there and opens nothing; left unset, the row says it was not handed
    one.

    ``curobo_available`` left unset asks this box, through ``curobo_env_available()``. It is
    a keyword so a caller can state the answer instead: a check that must not depend on what
    is installed, and a report written for a cell other than the one it runs on.

    ``collision_engine`` is the same shape for the exact mesh engine: ``'coal'``, ``'fcl'`` or
    ``None`` for a box with neither, and left unset it asks this one through
    ``import_collision_engine()``. Only a real cell's row reads it, so a console serving a sim or
    dummy profile never imports an engine to answer it.
    """
    checks: list[PreflightCheck] = []
    vendor = _vendor(robot_cfg)
    is_real = vendor not in ("sim", "dummy")

    checks.append(PreflightCheck(
        "vendor", CheckStatus.OK if is_real else CheckStatus.WARN,
        f"robot.vendor = {vendor!r}",
        "" if is_real else "not a real arm; this is a desk rehearsal, not a cell bring-up",
    ))

    # ---- the tool frame ------------------------------------------------------------------------
    tf = robot_cfg.gripper.tool_frame
    if tf.source == "undeclared":
        checks.append(PreflightCheck(
            "tool frame", CheckStatus.BLOCK if is_real else CheckStatus.WARN,
            "gripper.tool_frame.source is 'undeclared'; nobody has said where the grasp centre "
            "sits on the flange",
            "measure the coupling + gripper and set gripper.tool_frame.offset_mm + "
            "rotation_quat_xyzw, then source: 'willy' (this driver composes it, controller runs a "
            "bare flange) or 'polyscope' (the pendant holds it and we only verify)",
        ))
    else:
        checks.append(PreflightCheck(
            "tool frame", CheckStatus.OK,
            f"source={tf.source!r}, offset={tuple(round(v, 1) for v in tf.offset_mm)} mm, "
            f"verified to {tf.verify_tolerance_mm:.0f} mm at connect()",
        ))

    # ---- payload -------------------------------------------------------------------------------
    pl = robot_cfg.safety.payload
    if not pl.enforce:
        checks.append(PreflightCheck(
            "payload", CheckStatus.WARN,
            "safety.payload.enforce is false; the controller's payload is left untouched",
            "correct only for a genuinely bare flange; a mounted tool needs enforce: true + a "
            "weighed mass",
        ))
    elif pl.mass_kg == 0.0:
        checks.append(PreflightCheck(
            "payload", CheckStatus.BLOCK if is_real else CheckStatus.WARN,
            "safety.payload has enforce: true but mass_kg: 0.0",
            "weigh the tool (+ any carried workpiece) on a bench scale and set "
            "safety.payload.mass_kg and cog_mm; connect() refuses this combination",
        ))
    elif all(v == 0.0 for v in pl.cog_mm):
        checks.append(PreflightCheck(
            "payload", CheckStatus.BLOCK if is_real else CheckStatus.WARN,
            f"mass_kg {pl.mass_kg} declared but cog_mm is still [0, 0, 0]",
            "that says a multi-kilogram tool is a point mass at the flange face; measure the CoG in "
            "the same bench session as the tool frame",
        ))
    else:
        checks.append(PreflightCheck(
            "payload", CheckStatus.OK,
            f"{pl.mass_kg} kg at {tuple(round(v, 1) for v in pl.cog_mm)} mm "
            f"(limit {pl.max_mass_kg} kg)",
        ))

    # ---- the carried part ------------------------------------------------------------------------
    # A cuRobo UR models the part it carries, and its length is the cell's to declare: none is
    # implied. Read off the tree, starting nothing.
    if vendor == "ur":
        checks.append(_carried_part_row(robot_cfg, is_real=is_real))

    # ---- the camera -> base frame ---------------------------------------------------------------
    # Declared on the primary rig, camera.cameras.rigs[<id>].extrinsics. Read, not opened: loading
    # the artifact is the build's job, and the build refuses a file that does not load.
    if not chosen(camera):
        checks.append(PreflightCheck(
            "camera -> base", CheckStatus.BLOCK if is_real else CheckStatus.WARN,
            "this checklist was not handed the camera section, so it cannot vouch for CAMERA->BASE; a cell's "
            "calibration is declared on its primary rig, camera.cameras.rigs[<id>].extrinsics",
            "run it with the camera section of the same tree, as Cell.preflight and the operator console do",
        ))
    else:
        primary = camera.cameras.primary_rig_id
        key = f"camera.cameras.rigs[{primary!r}].extrinsics"
        rig = next((r for r in camera.cameras.rigs if r.rig_id == primary), None)
        extrinsics = getattr(rig, "extrinsics", None)
        if extrinsics is not None:
            checks.append(PreflightCheck(
                "camera -> base", CheckStatus.OK,
                f"{key}: {extrinsics.mounting_mode} from {extrinsics.artifact_path}; the build loads it "
                "and refuses a file that does not load",
            ))
        else:
            checks.append(PreflightCheck(
                "camera -> base", CheckStatus.BLOCK if is_real else CheckStatus.WARN,
                f"{key} is not declared, so the primary camera has no CAMERA->BASE transform",
                "perception reports grasps in the CAMERA frame; without a resolver the driver refuses every "
                "motion as INVALID_TARGET, which looks like a broken cell. Calibrate the camera (python -m "
                f"src.robot.execution.real_cell.calibrate --rig {primary}) and paste the rig block it "
                "prints (a wrist camera prints one too), or pass frame_resolver= in code",
            ))

    # ---- the camera world ----------------------------------------------------------------------
    # On a cuRobo UR cell every motion needs a live camera world or a decline, and the pick service
    # declines nothing, so a cell whose cameras give no world refuses every pick motion. Read off the
    # plan, opening nothing. A UR row only, because the schema defaults motion_planner to curobo for
    # every vendor.
    if vendor == "ur":
        checks.append(_camera_world_row(robot_cfg, camera))

    # ---- the exact mesh engine -----------------------------------------------------------------
    # Every planned or sampled path on a real cell is re-judged on the exact meshes before the arm
    # moves, and a box with no engine refuses every one of them. The cell would come up, connect and
    # refuse the first move: the right refusal in the wrong place. The row states the gate's own
    # sentence by calling the same function, so the two cannot drift into a desk check that promises
    # what the arm then refuses.
    if is_real:
        checks.append(_exact_mesh_engine_row(robot_cfg, collision_engine))

    # ---- the planner margin, and what admits it --------------------------------------------------
    # A cuRobo cell starts a planner only on a combination a committed evidence file measured. The
    # desk checks the half of that it can compute from config; the sidecar's own hashes are checked
    # when it starts, and the row says so rather than reading as if it had checked both.
    if is_real:
        checks.append(_planner_margin_row(robot_cfg, evidence_dir, data_dir))

    # ---- the gripper driver the build constructs ----------------------------------------------------
    # `robot.gripper.model` selects no driver, so a cell that names a customer hand and leaves
    # `gripper.vendor` alone gets the Robotiq socket driver aimed at it, and a vendor with no driver
    # is otherwise refused only at connect.
    if vendor == "ur":
        checks.append(_gripper_driver_row(robot_cfg))

    # ---- the cuRobo environment -----------------------------------------------------------------
    # Only a real arm has this row. A sim cell reaches cuRobo through its own driver and a dummy
    # plans nothing, so a missing environment there is not a fact about this checklist.
    if is_real:
        planner = getattr(getattr(robot_cfg, "ur", None), "motion_planner", None)
        here = curobo_available if chosen(curobo_available) else curobo_env_available()
        if planner == "curobo" and not here:
            checks.append(PreflightCheck(
                "cuRobo environment", CheckStatus.BLOCK,
                "robot.ur.motion_planner is 'curobo' and the cuRobo environment is not on this box",
                "every motion on this cell is planned or path checked, and both go through the "
                "sidecar, so the cell would connect and then refuse the first move. Install the "
                "environment (see ext_deps/README.md) or set robot.ur.motion_planner: ik, and run "
                "`python -m src.robot.safety.planning --doctor` afterwards: a `--check` "
                "that exits 0 says the config is sound, not that cuRobo runs here",
            ))
        elif planner == "curobo":
            checks.append(PreflightCheck(
                "cuRobo environment", CheckStatus.OK,
                "the cuRobo environment is present, so every motion is planned and every path judged",
            ))
        else:
            checks.append(PreflightCheck(
                "cuRobo environment", CheckStatus.WARN,
                f"robot.ur.motion_planner is {planner!r}, so no planner plans this cell's motions",
                "an interpolated move has no path anybody can judge and no camera world behind it: "
                "the guards see where each move ends and nothing in between. Set "
                "robot.ur.motion_planner: curobo for a cell that should be protected the whole way",
            ))

    # ---- self-collision ------------------------------------------------------------------------
    sc = robot_cfg.safety.self_collision
    if getattr(sc, "kinematics_model", None) is None:
        checks.append(PreflightCheck(
            "self-collision", CheckStatus.WARN,
            "safety.self_collision.kinematics_model is unset; the guard falls back to the arm's "
            "capability model",
            "set it explicitly so a wrong ur.model cannot silently drive one arm's link lengths "
            "against another's",
        ))
    else:
        checks.append(PreflightCheck(
            "self-collision", CheckStatus.OK,
            f"kinematics_model={sc.kinematics_model!r}, backend={getattr(sc, 'backend', '?')!r}",
        ))

    # ---- the hand ------------------------------------------------------------------------------
    # The one name the guard takes its hand from. A cell whose guard reads hand geometry refuses to
    # build without it, so this row is that refusal met at a desk, with the same sentence. And where
    # the declared tool frame puts it: a frame that places no hand blocks, whatever the backend,
    # because the guard refuses to be built on it, the planner refuses at start and the self filter
    # describes no body. Every placement the derivation admits builds; the checklist says it is
    # declared, and warns on a real UR declaring the Isaac asset's +Y.
    from src.config.loader import ConfigError
    from src.robot.safety.planning._hand_placement import ADMITTED_APPROACHES
    from src.robot.safety.planning.hand import (
        hand_geometry_model,
        planner_hand,
        unset_hand_refusal,
    )

    reads = hand_geometry_model(sc, robot_cfg.ur.model if vendor == "ur" else UNSET)
    try:
        hand = planner_hand(robot_cfg, data_dir=data_dir)
    except ConfigError as exc:
        checks.append(PreflightCheck(
            "hand", CheckStatus.BLOCK, str(exc),
            "the cell refuses to build until robot.gripper.model resolves to a hand it can model",
        ))
    else:
        if chosen(hand) and not chosen(hand.placement):
            checks.append(PreflightCheck(
                "hand", CheckStatus.BLOCK, str(hand.placement_refusal),
                f"declare the tool frame this cell's flange really has: a hand is placed along flange "
                f"{' or '.join(ADMITTED_APPROACHES)}, clocked by a quarter turn about that axis, and a planner starts "
                f"on a placement only with its own committed evidence file",
            ))
        elif chosen(hand) and chosen(hand.placement):
            bundle = ("the arm's own" if hand.guard_variant is None
                      else f"{hand.guard_variant}_hand_meshes.npz composed onto the arm")
            placement = hand.placement
            declared = placement.source != "undeclared"
            if not declared:
                axis = "tool frame undeclared"
            else:
                axis = (f"approach {placement.approach}, closing {placement.closing}, as robot.gripper.tool_frame "
                        f"declares it, declared and not measured on this flange")
                if placement.residual_deg > 1e-3:
                    axis += f", {placement.residual_deg:.3g} degrees from the declared frame, which snaps to it"
            from src.robot.drivers.sim.robot_models import curobo_arm_descriptor

            descriptor = (
                f"cuRobo descriptor {curobo_arm_descriptor(str(robot_cfg.ur.model))} with the hand added as a body link"
                if vendor == "ur" else "no UR descriptor on this vendor"
            )
            detail = (
                f"{hand.model}: sphere map {hand.sphere_map.name}, origin {hand.origin}, coupling "
                f"{hand.coupling_mm:g} mm, guard bundle {bundle}, {descriptor}, {axis}"
            )
            if vendor == "ur" and declared and placement.approach == "+Y":
                checks.append(PreflightCheck(
                    "hand", CheckStatus.WARN,
                    f"{detail}. +Y is the axis the Isaac UR asset mounts its hand along, and a hand "
                    f"bolted to a real UR flange approaches along +Z by the UR convention",
                    "measure which way the hand approaches on this cell's flange and declare that; a +Y frame builds, "
                    "and its planner starts only with its own committed evidence file",
                ))
            else:
                checks.append(PreflightCheck("hand", CheckStatus.OK, detail))
            if declared:
                checks.append(_grasp_centre_row(robot_cfg, hand, placement))
        elif reads is not None:
            what, fix = unset_hand_refusal(reads)
            checks.append(PreflightCheck("hand", CheckStatus.BLOCK, what, fix))
        else:
            checks.append(PreflightCheck(
                "hand", CheckStatus.OK,
                "robot.gripper.model is unset, and this cell reads no hand geometry: no exact mesh guard "
                "places a hand on a known arm",
            ))

    # ---- the wrist camera the arm carries ------------------------------------------------------
    checks.append(_wrist_camera_body_row(robot_cfg, camera, data_dir, is_real=is_real))

    # ---- fixtures ------------------------------------------------------------------------------
    # The list lives under the self-collision guard, not on `robot`. Read from the wrong place it
    # reports "none declared" on every cell, which is how a warning stops being read.
    fixtures = tuple(getattr(sc, "fixtures", None) or ())
    checks.append(PreflightCheck(
        "fixtures", CheckStatus.OK if fixtures else CheckStatus.WARN,
        ", ".join(f.name for f in fixtures) if fixtures
        else "none declared; there is no table in the collision world",
        "" if fixtures else "the guards will not stop a motion that goes through the bench surface; "
        "declare the table (and any bin walls) under safety.self_collision.fixtures before running "
        "near it",
    ))

    # ---- what the planner knows ----------------------------------------------------------------
    # A different question from the one above, with a different answer. The guard checks one
    # commanded configuration; the planner shapes the whole path. Without this block the planner
    # knows its own robot model and a generic table, and routes through everything else.
    from src.robot.safety.planning.reservation import PlannerReservation
    from src.robot.safety.planning.world import PlanningWorldError, build_planner_cuboids

    world_cfg = getattr(robot_cfg.safety, "planning_world", None)
    planner_in_use = str(getattr(robot_cfg.ur, "motion_planner", "ik")) == "curobo"
    reservation: PlannerReservation | None = None
    try:
        # The slot count comes from the reservation the arm starts its planner with, so this row and
        # the cell cannot disagree about whether a world fits.
        reservation = PlannerReservation.from_config(robot_cfg=robot_cfg)
        cuboids = build_planner_cuboids(world_cfg, fixtures, max_cuboids=reservation.cuboid_slots)
    except PlanningWorldError as exc:
        checks.append(PreflightCheck(
            "planning world", CheckStatus.BLOCK, str(exc),
            "fix the declaration; the planner cannot be given this world as written",
        ))
    else:
        if cuboids:
            checks.append(PreflightCheck(
                "planning world", CheckStatus.OK,
                f"{len(cuboids)} box(es): " + ", ".join(str(c["name"]) for c in cuboids),
                "" if planner_in_use else "declared, but robot.ur.motion_planner is 'ik', so no "
                "planner reads it; set 'curobo' for these obstacles to shape a path",
            ))
        else:
            checks.append(PreflightCheck(
                "planning world", CheckStatus.WARN if planner_in_use else CheckStatus.OK,
                "not declared; the planner keeps its own generic world",
                "the planner will route through your bench, bin and fixtures; declare them under "
                "safety.planning_world" if planner_in_use else "",
            ))

    # ---- what the planner allocates ------------------------------------------------------------
    # Printed rather than budgeted: nothing measures planner VRAM yet, and a budget that counted the
    # grid and the slots alone would claim more than it checks.
    if planner_in_use and reservation is not None:
        checks.append(PreflightCheck("planner reservation", CheckStatus.OK, reservation.render()))

    # ---- record logging ------------------------------------------------------------------------
    path = getattr(robot_cfg.grasping, "record_log_path", None)
    checks.append(PreflightCheck(
        "record log", CheckStatus.OK if path else CheckStatus.WARN,
        f"-> {path}" if path else "grasping.record_log_path is unset; nothing will be recorded",
        "" if path else "set it; a bring-up with no telemetry cannot be diagnosed afterwards",
    ))

    # ---- what config genuinely cannot answer ----------------------------------------------------
    if is_real:
        checks.append(PreflightCheck(
            "controller state", CheckStatus.BENCH,
            "powered, brakes released, Remote Control active, no program running on the pendant",
            # The two cases behave differently and the line says so rather than bundling them under
            # "refuses". In local control the upload is refused, measured against URSim 5.26.0
            # (drivers/ur/connection.py). With a pendant program running in remote, ur_rtde is not
            # refused: it stops that program and takes the robot, which the UR driver README in this
            # same tree states as well ("A running program also stops the moment another is sent").
            # An operator told to expect a refusal waits for one that never comes.
            "ur_rtde uploads a control script. In LOCAL control the controller refuses it. In REMOTE "
            "with a pendant program running it is NOT refused: the upload STOPS that program and "
            "takes the robot. Confirm on the pendant; no API reports this",
        ))
        checks.append(_end_effector_wiring_row(robot_cfg))

    return PreflightReport(tuple(checks))
