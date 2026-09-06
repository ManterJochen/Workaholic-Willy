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


def run_config_preflight(robot_cfg: "RobotConfig") -> PreflightReport:
    """Check a ``RobotConfig`` for everything that stops a real pick, without touching hardware."""
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
    from src.robot.safety.planning.world import PlanningWorldError, build_planner_cuboids

    world_cfg = getattr(robot_cfg.safety, "planning_world", None)
    planner_in_use = str(getattr(robot_cfg.ur, "motion_planner", "ik")) == "curobo"
    try:
        cuboids = build_planner_cuboids(world_cfg, fixtures)
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
            "ur_rtde uploads a control script; the controller refuses it in Local control or with a "
            "pendant program owning the robot. Confirm on the pendant; no API reports this",
        ))
        checks.append(PreflightCheck(
            "end-effector wiring", CheckStatus.BENCH,
            "the Robotiq URCap (it opens port 63352, not any pip package) or the vacuum solenoid's "
            "24 V supply",
            "confirm physically; a missing URCap looks like a gripper that never answers",
        ))

    return PreflightReport(tuple(checks))
