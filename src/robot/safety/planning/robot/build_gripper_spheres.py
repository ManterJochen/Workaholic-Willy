"""Generate the Robotiq 2F-85 gripper collision-sphere map for cuRobo, without Isaac.

The collision geometry of the reference robot is the vertex-exact, version-controlled
``src/robot/safety/data/ur5e_collision_meshes.npz``, the same source that feeds the
Coal self-collision guard and the on-box cuRobo sphere fit. This script grid-fits the
gripper, meaning the ``gripper`` body with ``lfinger`` and ``rfinger``, all in the
tool0 frame, into a cuRobo-format collision-sphere map. The gripper therefore exists as
a labelled file in the repository with no Isaac or GPU dependency.

It writes ``ur5e_gripper_spheres.yml`` next to itself. Run it with the project venv:

    .venv/Scripts/python.exe src/robot/safety/planning/robot/build_gripper_spheres.py

The scope is the tool0 gripper spheres, which are frame-correct and directly usable.
The arm-link spheres and the Lula-tuned, schema-complete ``ur5e.yml`` are produced
on-box by ``scripts/curobo/build_ur_config.py``, which needs the Isaac Lula
description and the cuRobo environment; ``PROVENANCE.md`` records that. Planning
quality with either sphere set is an on-box measurement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

_MESH_NPZ = (
    Path(__file__).resolve().parents[3] / "safety" / "data" / "ur5e_collision_meshes.npz"
)
_OUT = Path(__file__).with_name("ur5e_gripper_spheres.yml")


def _grid_fit_spheres(verts_mm: np.ndarray, cell_mm: float, rmax_mm: float) -> list[dict]:
    """One tight sphere per occupied voxel.

    The centre is the voxel centroid and the radius is the largest vertex distance,
    capped. Their union covers the real mesh with minimal over-reach. The spheres come
    back in metres in the tool0 frame. It mirrors ``scripts/curobo/build_ur_config.py``,
    so the committed gripper spheres match the on-box fit.
    """
    v = np.asarray(verts_mm, dtype=np.float64)
    keys = np.floor(v / cell_mm).astype(np.int64)
    out: list[dict] = []
    for key in sorted({tuple(k) for k in keys}):
        pts = v[np.all(keys == np.asarray(key), axis=1)]
        c = pts.mean(0)
        r = min(float(np.max(np.linalg.norm(pts - c, axis=1))), rmax_mm)
        out.append(
            {
                "center": [round(float(x) / 1000.0, 4) for x in c],
                "radius": round(max(r, 6.0) / 1000.0, 4),
            }
        )
    return out


def build() -> dict:
    mesh = np.load(_MESH_NPZ, allow_pickle=True)
    tool0 = (
        _grid_fit_spheres(mesh["gripper__v"], cell_mm=44.0, rmax_mm=24.0)
        + _grid_fit_spheres(mesh["lfinger__v"], cell_mm=34.0, rmax_mm=17.0)
        + _grid_fit_spheres(mesh["rfinger__v"], cell_mm=34.0, rmax_mm=17.0)
    )
    return {
        "_provenance": {
            "robot": "Universal Robots UR5e",
            "gripper": "Robotiq 2F-85",
            "frame": "tool0 (Y=approach, X=closing, Z=depth), metres",
            "source": "src/robot/safety/data/ur5e_collision_meshes.npz (vertex-exact)",
            "generated_by": "src/robot/safety/planning/robot/build_gripper_spheres.py",
            "note": (
                "tool0 gripper spheres only (frame-correct, directly cuRobo-usable). Arm-link "
                "spheres + the Lula-tuned complete ur5e.yml are on-box via "
                "scripts/curobo/build_ur_config.py. See PROVENANCE.md."
            ),
        },
        "collision_spheres": {"tool0": tool0},
    }


def main() -> None:
    cfg = build()
    _OUT.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")
    print(f"wrote {_OUT}  ({len(cfg['collision_spheres']['tool0'])} tool0 spheres from {_MESH_NPZ.name})")


if __name__ == "__main__":
    main()
