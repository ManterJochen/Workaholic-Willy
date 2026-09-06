"""Why a mesh earns no grasp, answerable from a shell.

The question a user with their own objects actually asks. They point `datagen` at a directory of
parts, run the screen, and half of them come back with nothing. "Why" is the whole difference
between a tool they can use and one they abandon.

The pooled rejection histogram refutes the obvious guess. Thin flat objects look as though they must
fail because the fingers would meet the table underneath them, and `below_table` is a small share of
the rejections. What refuses them is `too_wide`: the width along the line tried. That is why the two
levers that matter are the closing axis and the anchor grid, because both change which line gets
tried.

And the pose is half the answer. A screen places a mesh upright; a settled scene does not. A bottle
standing up presents a 70 mm neck over a 250 mm height, and the same bottle on its side presents its
70 mm diameter along the whole length, so lying an object down rescues meshes an upright screen
refused.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Final, Sequence

import numpy as np

__all__ = ["REST_ROTATIONS", "MeshVerdict", "why_no_jaw"]

#: Upright, then each of the two other faces down. A settled object rests on one of its faces, so
#: these three cover what a random flat orientation actually produces for a box-like solid.
REST_ROTATIONS: Final[tuple[tuple[str, tuple], ...]] = (
    ("upright", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("x down", ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))),
    ("y down", ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))),
)


@dataclass(frozen=True, slots=True)
class MeshVerdict:
    """What one mesh earned, in every rest pose tried, and why the upright one refused it."""

    asset_id: str
    path: str
    upright_labels: int
    best_labels: int
    best_pose: str
    reasons: dict[str, int]

    @property
    def rescued_by_lying_down(self) -> bool:
        return self.upright_labels == 0 and self.best_labels > 0


def _label_one(shape: Any, rotation: np.ndarray, density: Any) -> tuple[int, dict[str, int]]:
    from datagen.grasps.labels import SceneGeometry, Solid, label_jaw_grasps  # noqa: PLC0415

    extent = np.abs(rotation @ np.asarray(shape.extent_mm, dtype=float))
    solid = Solid(kind="mesh", half_extent_mm=np.asarray(shape.extent_mm, dtype=float) / 2.0,
                  rotation=rotation, centre_mm=np.array([0.0, 0.0, float(extent[2] / 2.0)]),
                  instance_id=0, asset_id="probe", mesh=shape)
    geometry = SceneGeometry(scene_id="probe", family="alone", objects={0: solid}, walls=())
    labels, rejected = label_jaw_grasps(geometry, 0, density=density)
    return len(labels), dict(rejected)


def why_no_jaw(meshes: Sequence[tuple[str, str]], *, density: Any = None,
               poses: Sequence[tuple[str, tuple]] = REST_ROTATIONS,
               report: Any = None) -> dict[str, Any]:
    """Label each mesh alone in every rest pose and pool the reasons the upright one refused.

    `meshes` is `(asset_id, path)` pairs. Returns the pooled histogram, the per-mesh verdicts and
    how many were rescued by a different rest pose.

    Every pose is tried, and that is not a detail. A screen that reports only the upright answer
    measures the worst case, and a user told "this part cannot be grasped" on the strength of one
    pose has been told something false about their part.
    """
    from datagen.assets.meshes import load_mesh_shape  # noqa: PLC0415
    from datagen.grasps.labels import DEFAULT_DENSITY  # noqa: PLC0415

    say = report if report is not None else (lambda _line: None)
    density = density if density is not None else DEFAULT_DENSITY
    reasons: collections.Counter[str] = collections.Counter()
    verdicts: list[MeshVerdict] = []
    unreadable = 0
    for asset_id, path in meshes:
        shape = load_mesh_shape(str(path))
        if shape is None:
            unreadable += 1
            continue
        upright, best, best_pose, upright_reasons = 0, 0, "", {}
        for name, rows in poses:
            rotation = np.asarray(rows, dtype=float)
            count, rejected = _label_one(shape, rotation, density)
            if name == poses[0][0]:
                upright, upright_reasons = count, rejected
                reasons.update(rejected)
            if count > best:
                best, best_pose = count, name
        verdicts.append(MeshVerdict(asset_id, str(path), upright, best, best_pose,
                                    upright_reasons))
        if len(verdicts) % 25 == 0:
            say(f"  {len(verdicts)} of {len(meshes)} probed")

    total = sum(reasons.values()) or 1
    rescued = [v for v in verdicts if v.rescued_by_lying_down]
    return {
        "probed": len(verdicts),
        "unreadable": unreadable,
        "with_no_label_in_any_pose": sum(1 for v in verdicts if v.best_labels == 0),
        "rescued_by_a_different_pose": len(rescued),
        "reasons": {name: {"count": count, "share": round(count / total, 4)}
                    for name, count in reasons.most_common()},
        "verdicts": verdicts,
    }
