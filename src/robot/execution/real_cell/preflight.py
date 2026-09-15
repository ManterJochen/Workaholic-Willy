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
from typing import TYPE_CHECKING

from src.contracts import UNSET, Maybe, chosen
from src.robot.safety.planning import curobo_env_available

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

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


def _vendor(robot_cfg: "RobotConfig") -> str:
    v = getattr(robot_cfg, "vendor", "")
    return str(getattr(v, "value", v)).lower()


def run_config_preflight(
    robot_cfg: "RobotConfig", *, curobo_available: Maybe[bool] = UNSET
) -> PreflightReport:
    """Check a ``RobotConfig`` for everything that stops a real pick, without touching hardware.

    ``curobo_available`` left unset asks this box, through ``curobo_env_available()``. It is
    a keyword so a caller can state the answer instead: a check that must not depend on what
    is installed, and a report written for a cell other than the one it runs on.
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

    # ---- the camera -> base frame ---------------------------------------------------------------
    fusion = getattr(robot_cfg.grasping, "fusion", None)
    artifact = getattr(fusion, "extrinsics_artifact_path", None) if fusion else None
    if artifact and getattr(fusion, "enabled", False):
        checks.append(PreflightCheck(
            "camera -> base", CheckStatus.OK, f"from the calibration artifact {artifact}",
        ))
    else:
        checks.append(PreflightCheck(
            "camera -> base", CheckStatus.BLOCK if is_real else CheckStatus.WARN,
            "no CAMERA->BASE resolver in config (grasping.fusion is off or has no artifact path)",
            "perception reports grasps in the CAMERA frame; without a resolver the driver refuses "
            "every motion as INVALID_TARGET, which looks like a broken cell. Run the eye-to-hand "
            "calibration and point grasping.fusion.extrinsics_artifact_path at what it wrote, or "
            "pass a resolver in code (the eye-in-hand route)",
        ))

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
                "`python -m src.robot.execution.real_cell --doctor` afterwards: a `--check` "
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
            f"kinematics_model={sc.kinematics_model!r}, backend={getattr(sc, 'backend', '?')!r}, "
            f"planner margin {getattr(sc, 'planner_margin_mm', '?')} mm",
        ))

    # ---- the hand ------------------------------------------------------------------------------
    # The one name the guard takes its hand from. A cell whose guard reads hand geometry refuses to
    # build without it, so this row is that refusal met at a desk, with the same sentence.
    from src.config.loader import ConfigError
    from src.robot.safety.planning.hand import (
        approach_refusal,
        hand_geometry_model,
        planner_hand,
        unset_hand_refusal,
    )

    reads = hand_geometry_model(sc, robot_cfg.ur.model if vendor == "ur" else UNSET)
    try:
        hand = planner_hand(robot_cfg)
    except ConfigError as exc:
        checks.append(PreflightCheck(
            "hand", CheckStatus.BLOCK, str(exc),
            "the cell refuses to build until robot.gripper.model resolves to a hand it can model",
        ))
    else:
        disagreement = approach_refusal(hand) if chosen(hand) else None
        if chosen(hand) and reads is not None and disagreement is not None:
            checks.append(PreflightCheck(
                "hand", CheckStatus.BLOCK, disagreement,
                "measure which way the hand approaches on this cell's flange, then bake and commit a hand model "
                "that holds it; declaring the model's axis instead would describe a hand this cell does not have",
            ))
        elif chosen(hand):
            bundle = "the arm's own" if hand.guard_variant is None else hand.guard_variant
            if hand.declared_approach is None:
                axis = "tool frame undeclared"
            elif disagreement is None:
                axis = "approach agrees with the declared tool frame"
            else:
                axis = (f"approach {hand.approach_disagreement_deg:.0f} degrees from the declared tool frame, "
                        f"which no exact mesh guard on this cell reads")
            from src.robot.drivers.sim.robot_models import curobo_robot_yml

            descriptor = (
                f"cuRobo descriptor {curobo_robot_yml(str(robot_cfg.ur.model), hand.model)}" if vendor == "ur"
                else "no UR descriptor on this vendor"
            )
            checks.append(PreflightCheck(
                "hand", CheckStatus.OK,
                f"{hand.model}: sphere map {hand.sphere_map.name}, origin {hand.origin}, coupling "
                f"{hand.coupling_mm:g} mm, guard bundle {bundle}, {descriptor}, {axis}",
            ))
        elif reads is not None:
            what, fix = unset_hand_refusal(reads)
            checks.append(PreflightCheck("hand", CheckStatus.BLOCK, what, fix))
        else:
            checks.append(PreflightCheck(
                "hand", CheckStatus.OK,
                "robot.gripper.model is unset, and this cell reads no hand geometry: no exact mesh guard "
                "places a hand on a known arm",
            ))

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
            # ⛔ THE TWO CASES BEHAVE DIFFERENTLY AND THIS LINE USED TO BUNDLE THEM UNDER "refuses".
            # Local control: the upload IS refused (measured against URSim 5.26.0, see
            # drivers/ur/connection.py). A pendant program in Remote: ur_rtde does NOT get refused,
            # it STOPS that program and takes the robot, which the UR driver README states in this
            # same tree ("A running program also stops the moment another is sent"). An operator
            # told to expect a refusal waits for one that never comes.
            "ur_rtde uploads a control script. In LOCAL control the controller refuses it. In REMOTE "
            "with a pendant program running it is NOT refused: the upload STOPS that program and "
            "takes the robot. Confirm on the pendant; no API reports this",
        ))
        checks.append(PreflightCheck(
            "end-effector wiring", CheckStatus.BENCH,
            "the Robotiq URCap (it opens port 63352, not any pip package) or the vacuum solenoid's "
            "24 V supply",
            "confirm physically; a missing URCap looks like a gripper that never answers",
        ))

    return PreflightReport(tuple(checks))
