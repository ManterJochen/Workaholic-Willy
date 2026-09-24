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
from pathlib import Path
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


def _furthest_station(model: str, shoulder_mm: float,
                      path: "str | Path | None") -> tuple[float, float, float] | None:
    """The TCP position of the declared eye-in-hand station furthest from the shoulder, or ``None`` for none.

    A joint station has no position before forward kinematics and is left out; a file that cannot be read is logged
    and has no row.
    """
    from src.robot.execution.pose_provider import load_stations
    from src.willy_sim.calibration.paths import STATIONS_DIR

    source = Path(path) if path is not None else STATIONS_DIR / f"eih_{model}.json"
    if not source.is_file():
        return None
    try:
        stations = load_stations(source)
    except (OSError, ValueError) as exc:
        _LOGGER.warning("reach audit: the stations in %s were not read: %s", source, exc)
        return None
    points = [tuple(float(v) for v in station.position_mm) for station in stations
              if hasattr(station, "position_mm")]
    if not points:
        return None
    x, y, z = max(points, key=lambda p: math.sqrt(p[0] ** 2 + p[1] ** 2 + (p[2] - shoulder_mm) ** 2))
    return (x, y, z)


def audit_reach(robot: "RobotConfig", *, margin_mm: float = 0.0,
                eih_stations: "str | Path | None" = None) -> list[ReachFinding]:
    """Measure every reach-critical configured point against ``sim.robot_model``'s working sphere.

    ``margin_mm`` shrinks the allowed sphere (a safety/dexterity margin: the arm is near-singular and has
    almost no orientation freedom at the very edge of its reach, so a "just barely inside" target is not
    usefully reachable for a top-down grasp). ``eih_stations`` is the eye-in-hand stations file to measure,
    the one declared for the model when unset; a model with none declared has no stations row.
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
    # The eye-in-hand calibration does not drive the arm to the marker. It drives it to the declared stations
    # round it, which stand up to a radius further out, so it is the stations that have to be reachable. The
    # one furthest from the shoulder is checked: a stations file declared for another arm, or for a scene whose
    # marker moved, shows here before an Isaac boot.
    worst = _furthest_station(model, shoulder, eih_stations)
    if worst is not None:
        out.append(_finding("eih stations (worst)", worst, model, limit, shoulder))

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
