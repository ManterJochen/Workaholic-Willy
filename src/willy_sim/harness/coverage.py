"""Camera-coverage audit for a sim cell: does the configured camera see the scene?

The companion to :mod:`src.willy_sim.harness.reach`. Reach asks whether the arm can touch a point;
coverage asks whether the camera can see it. Both are geometry questions, and without this audit
they are answered roughly 60 s into an Isaac boot by a run that fails for a reason that looks like
something else: a detector that finds nothing reads as a perception-quality problem long before an
out-of-frame object is suspected.

A camera hanging directly over the scene is pointed at the workspace by construction. A rig of
cameras mounted off to the side and tilted is not, and the config expresses the two halves as
independent layers: a robot layer anchors the scene, a camera layer places the optics, and nothing
in the config system forces them to agree about where the workspace is. This module checks that.

The check is conservative: a point counts as framed only if it falls inside the cone of the
narrower (vertical) field of view, so the verdict holds however the image is rolled.

Pure math and config, with no ``isaacsim`` import.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.willy_sim.scene.cameras import vfov_deg_from_hfov

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "CoverageFinding",
    "audit_camera_coverage",
    "format_coverage_report",
    "warn_if_out_of_frame",
]

#: Resolution assumed when a camera entry does not state one (matches the authored default).
_DEFAULT_RESOLUTION = (640, 480)


@dataclass(frozen=True, slots=True)
class CoverageFinding:
    """One scene point measured against one camera's view cone."""

    camera: str
    label: str
    position_mm: tuple[float, float, float]
    off_axis_deg: float          #: angle between the optical axis and the direction to the point
    half_fov_deg: float | None   #: conservative (vertical) half-FOV; ``None`` when the lens is unstated

    @property
    def in_frame(self) -> bool | None:
        """``True``/``False`` when the lens is known, ``None`` when it is not (unknown, not "fine")."""
        if self.half_fov_deg is None:
            return None
        return self.off_axis_deg <= self.half_fov_deg

    @property
    def overshoot_deg(self) -> float:
        return 0.0 if self.half_fov_deg is None else max(0.0, self.off_axis_deg - self.half_fov_deg)


def _vec3(values: Iterable[float]) -> tuple[float, float, float]:
    """Pydantic hands back a plain tuple; pin it to a 3-vector for the geometry helpers."""
    x, y, z = (float(v) for v in values)
    return (x, y, z)


def _resolution(values: Iterable[int] | None) -> tuple[int, int]:
    if values is None:
        return _DEFAULT_RESOLUTION
    width, height = (int(v) for v in values)
    return (width, height)


def _off_axis_deg(
    position_mm: tuple[float, float, float],
    aim_mm: tuple[float, float, float],
    point_mm: tuple[float, float, float],
) -> float:
    axis = np.asarray(aim_mm, dtype=np.float64) - np.asarray(position_mm, dtype=np.float64)
    to_point = np.asarray(point_mm, dtype=np.float64) - np.asarray(position_mm, dtype=np.float64)
    n_axis, n_point = float(np.linalg.norm(axis)), float(np.linalg.norm(to_point))
    if n_axis < 1e-9 or n_point < 1e-9:
        return 0.0  # degenerate placement; the reach/aim validators own that case
    cos = float(np.dot(axis, to_point)) / (n_axis * n_point)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _scene_points(robot: "RobotConfig") -> list[tuple[str, tuple[float, float, float]]]:
    """The points a camera is expected to frame: every authored object, the marker, and the
    workspace footprint corners at table height (what the operator believes is the usable area)."""
    scene = robot.sim.scene_setup
    objects = list(scene.objects) if scene.objects else [scene.object]
    out: list[tuple[str, tuple[float, float, float]]] = [
        (f"object[{i}] {spec.name}", _vec3(spec.position_mm)) for i, spec in enumerate(objects)
    ]
    if scene.marker.kind != "none":
        out.append(("marker", _vec3(scene.marker.position_mm)))
    wl = robot.workspace_limits
    for x in (wl.x_min, wl.x_max):
        for y in (wl.y_min, wl.y_max):
            out.append((f"workspace corner ({x:.0f},{y:.0f})", (float(x), float(y), float(wl.z_min))))
    return out


def audit_camera_coverage(robot: "RobotConfig") -> list[CoverageFinding]:
    """Measure the configured scene against every aimed fixed camera's view cone.

    Only cameras that state both a world ``position_mm`` and a ``mount_aim_mm`` are audited: the aim
    point is what marks a camera as deliberately placed and pointed. The nadir camera states none
    and is skipped.
    """
    points = _scene_points(robot)
    out: list[CoverageFinding] = []
    for name, cam in sorted(robot.sim.cameras.items()):
        if cam.mounting_mode != "eye_to_hand" or not cam.position_mm or not cam.mount_aim_mm:
            continue
        position, aim = _vec3(cam.position_mm), _vec3(cam.mount_aim_mm)
        half_fov = (
            None if cam.hfov_deg is None
            else vfov_deg_from_hfov(cam.hfov_deg, _resolution(cam.resolution)) / 2.0
        )
        out.extend(
            CoverageFinding(name, label, point, _off_axis_deg(position, aim, point), half_fov)
            for label, point in points
        )
    return out


def format_coverage_report(findings: list[CoverageFinding]) -> str:
    lines = ["camera coverage audit (conservative: the narrower vertical FOV cone):"]
    for f in findings:
        if f.half_fov_deg is None:
            mark, extra = "?  ", "  (lens unstated -> set hfov_deg to check)"
        elif f.in_frame:
            mark, extra = "ok ", ""
        else:
            mark, extra = "OUT", f"  (+{f.overshoot_deg:.1f} deg outside the frame)"
        lines.append(
            f"  [{mark}] {f.camera:12s} {f.label:28s} off-axis {f.off_axis_deg:5.1f} deg"
            f"{'' if f.half_fov_deg is None else f' / {f.half_fov_deg:.1f} deg half-FOV'}{extra}"
        )
    return "\n".join(lines)


def warn_if_out_of_frame(robot: "RobotConfig") -> list[CoverageFinding]:
    """Audit and log every scene point outside a camera's frame. Returns the offenders."""
    findings = audit_camera_coverage(robot)
    bad = [f for f in findings if f.in_frame is False]
    if bad:
        _LOGGER.warning(
            "%s\nThe camera rig above does not frame the whole configured scene. A detector cannot find "
            "what is not in the image, so this surfaces as poor perception rather than as bad geometry. "
            "Re-aim the camera, widen it, or move the scene.",
            format_coverage_report(findings),
        )
    return bad
