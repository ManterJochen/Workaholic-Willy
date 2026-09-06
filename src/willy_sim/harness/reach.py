"""Reachability audit for a sim cell: does the configured scene fit the configured robot?

Every willy_sim scene position (object, marker, bins, the retreat ``safe_pose``, the workspace box)
was authored for a UR5e, whose reach is 850 mm. Setting ``sim.robot_model`` to a UR3e shrinks the
working sphere to 500 mm, which puts most of those positions out of reach without saying so. The
symptom is an opaque "IK_FAILED" or "no-plan" roughly 60 s into an Isaac boot, or a run that looks
like bad grasping when it is bad geometry.

This module answers the question in milliseconds and without Isaac: it measures each configured
point against the model's working sphere and reports which ones do not fit, and by how much.

The working sphere is centred on the shoulder, not on the base plane. A UR's datasheet reach is the
radius about the shoulder joint, which sits ``d1`` above the base (the first DH ``d``). Measuring
from the base origin would over-state the envelope by up to ``d1``, which is 152 mm on a UR3e, and
wave through poses the arm cannot touch.

Pure numpy plus the config and DH tables, with no ``isaacsim`` import.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.robot.drivers.sim.robot_models import ur_model_spec
from src.robot.safety._ur_kinematics import UR_DH_TABLES_M

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

_LOGGER = logging.getLogger(__name__)

__all__ = ["ReachFinding", "audit_reach", "format_reach_report", "shoulder_height_mm"]


@dataclass(frozen=True, slots=True)
class ReachFinding:
    """One configured point measured against the model's working sphere."""

    label: str
    position_mm: tuple[float, float, float]
    radius_mm: float          #: distance from the shoulder (the reach-sphere centre)
    max_reach_mm: float
    reachable: bool

    @property
    def overshoot_mm(self) -> float:
        return max(0.0, self.radius_mm - self.max_reach_mm)


def shoulder_height_mm(model: str) -> float:
    """``d1`` in mm: the shoulder height above the base plane, the centre of the reach sphere."""
    table = UR_DH_TABLES_M.get(model.lower())
    return float(table[0].d_m * 1000.0) if table else 0.0


def _finding(label: str, pos, model: str, max_reach_mm: float, shoulder_mm: float) -> ReachFinding:
    x, y, z = (float(pos[0]), float(pos[1]), float(pos[2]))
    r = math.sqrt(x * x + y * y + (z - shoulder_mm) ** 2)
    return ReachFinding(label, (x, y, z), r, max_reach_mm, r <= max_reach_mm)


def audit_reach(robot: "RobotConfig", *, margin_mm: float = 0.0) -> list[ReachFinding]:
    """Measure every reach-critical configured point against ``sim.robot_model``'s working sphere.

    ``margin_mm`` shrinks the allowed sphere (a safety/dexterity margin: the arm is near-singular and has
    almost no orientation freedom at the very edge of its reach, so a "just barely inside" target is not
    usefully reachable for a top-down grasp).
    """
    model = robot.sim.robot_model
    spec = ur_model_spec(model)
    limit = max(0.0, spec.max_reach_mm - margin_mm)
    shoulder = shoulder_height_mm(model)
    scene = robot.sim.scene_setup

    out: list[ReachFinding] = [
        _finding("scene_setup.object", scene.object.position_mm, model, limit, shoulder),
        _finding("scene_setup.marker", scene.marker.position_mm, model, limit, shoulder),
        _finding("safe_pose", (robot.safe_pose.x, robot.safe_pose.y, robot.safe_pose.z), model, limit, shoulder),
    ]
    # The eye-in-hand calibration does not drive the arm to the marker. It drives it around a
    # standoff sphere of camera viewpoints centred on the marker, and it is that sphere, not the
    # marker, that has to be reachable: the viewpoints sit up to ``max(radii_mm)`` further out. The
    # worst of them is the one pushed radially away from the base at the shallowest elevation, so
    # that is the one checked. Conservative by design: it bounds the camera standoff while the tool
    # rides closer in, so a "just out" verdict means expect rejected poses, not certainly zero.
    vp = scene.eih_viewpoints
    if vp.radii_mm and vp.elevations_deg:
        mx, my, mz = (float(v) for v in scene.marker.position_mm)
        horiz = math.hypot(mx, my)
        ux, uy = (mx / horiz, my / horiz) if horiz > 1e-9 else (1.0, 0.0)   # outward, away from the base
        r_max, el_min = max(vp.radii_mm), math.radians(min(vp.elevations_deg))
        reach_out = r_max * math.cos(el_min)
        out.append(_finding(
            "scene_setup.eih_viewpoints (worst)",
            (mx + ux * reach_out, my + uy * reach_out, mz + r_max * math.sin(el_min)),
            model, limit, shoulder,
        ))

    # The workspace box is a hard gate: the WorkspaceGuard rejects any target outside it, so its far
    # corners describe what the operator believes is usable. Only the worst corner is checked.
    wl = robot.workspace_limits
    corners = [(x, y, z) for x in (wl.x_min, wl.x_max) for y in (wl.y_min, wl.y_max) for z in (wl.z_min, wl.z_max)]
    worst = max(corners, key=lambda c: _finding("", c, model, limit, shoulder).radius_mm)
    out.append(_finding("workspace_limits (worst corner)", worst, model, limit, shoulder))
    return out


def format_reach_report(findings: list[ReachFinding], model: str) -> str:
    lines = [f"reach audit ({model}, sphere r<={findings[0].max_reach_mm:.0f} mm about the shoulder):"]
    for f in findings:
        mark = "ok " if f.reachable else "OUT"
        extra = "" if f.reachable else f"  (+{f.overshoot_mm:.0f} mm beyond reach)"
        lines.append(
            f"  [{mark}] {f.label:34s} {tuple(round(v, 1) for v in f.position_mm)!s:>28} "
            f"r={f.radius_mm:6.1f} mm{extra}"
        )
    return "\n".join(lines)


def warn_if_unreachable(robot: "RobotConfig", *, margin_mm: float = 0.0) -> list[ReachFinding]:
    """Audit and log every unreachable point. Returns the offenders; empty means the scene fits."""
    findings = audit_reach(robot, margin_mm=margin_mm)
    bad = [f for f in findings if not f.reachable]
    if bad:
        _LOGGER.warning(
            "%s\nThe scene above was authored for a longer arm. Re-anchor it (or pick another robot_model); "
            "these targets are physically untouchable and will surface as IK/plan failures, not as "
            "grasp-quality problems.",
            format_reach_report(findings, robot.sim.robot_model),
        )
    return bad
