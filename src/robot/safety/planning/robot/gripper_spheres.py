"""Fit a gripper's collision spheres, from a baked bundle or from a mesh file.

The planner models a robot as spheres and nothing else. A gripper's mesh is never used directly: it
is the source a sphere set is fitted from, once, and the spheres are what the planner ever sees. So
"give the planner the gripper in 3D" means exactly this file.

Why it exists as a module rather than as a script. The fit was written twice, in
``build_gripper_spheres.py`` here and in ``scripts/curobo/build_ur_config.py`` on the target box, one
saying it mirrored the other and nothing checking. The on-box copy also fitted the Robotiq 2F-85
whatever gripper the cell had, with a comment calling that model independent, so a cell running a
different end effector planned as though a Robotiq were bolted to the flange while its safety guard
used the right geometry. One function, two callers, and a test that the committed file is what this
produces.

Two sources are supported, and they are the two a customer actually has:

  * A baked bundle, ``{name}_collision_meshes.npz``, which is what this repository ships for its own
    grippers and what the safety guard reads. Vertex exact and already in the ``tool0`` frame.
  * A mesh file, anything trimesh loads. That is the case for a gripper nobody has baked yet, and it
    puts the burden where it belongs: on stating the frame and the units, which the loader cannot
    know and this module therefore demands.

The frame is ``tool0``, the units in the file are millimetres and the units in the output are metres,
because that is what cuRobo reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "GripperSpheres",
    "SphereFitError",
    "bundle_gripper_arrays",
    "fit_gripper_spheres",
    "fit_spheres_from_mesh",
    "grid_fit_spheres",
]

#: The arrays a baked bundle carries for the end effector, in the order they are fitted.
#:
#: The body first and the fingers after it, each with its own cell size: the palm is one large solid
#: and a finger is a thin blade, and one cell size for both either buries the fingers in a sphere the
#: size of the palm or covers the palm in dozens of tiny ones.
_BUNDLE_PARTS: tuple[tuple[str, float, float], ...] = (
    ("gripper__v", 44.0, 24.0),
    ("lfinger__v", 34.0, 17.0),
    ("rfinger__v", 34.0, 17.0),
)

#: No sphere smaller than this, millimetres. A sphere below it costs a collision check and covers
#: almost nothing, and a fit that produces hundreds of them is slower everywhere for no clearance.
_MIN_RADIUS_MM = 6.0


class SphereFitError(ValueError):
    """The gripper cannot be fitted as described."""


@dataclass(frozen=True, slots=True)
class GripperSpheres:
    """A fitted sphere set, and where it came from.

    The provenance is not decoration. A sphere map is geometry a planner trusts absolutely, it looks
    like any other list of numbers, and the question a reader will have in a year is which gripper
    and which file it was fitted from.
    """

    #: Spheres in the `tool0` frame, metres, in cuRobo's map format.
    spheres: tuple[dict[str, Any], ...]
    #: What was fitted: a bundle name or a mesh path.
    source: str
    #: The gripper this describes, as an operator would name it.
    gripper: str

    @property
    def count(self) -> int:
        return len(self.spheres)

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        radii = [float(s["radius"]) * 1000.0 for s in self.spheres]
        if not radii:
            return f"{self.gripper}: no spheres fitted from {self.source}"
        return (
            f"{self.gripper}: {self.count} sphere(s) from {self.source}, "
            f"radius {min(radii):.1f} to {max(radii):.1f} mm"
        )

    def to_dict(self) -> dict[str, Any]:
        """The wire view, in the shape the planner config expects under `collision_spheres`."""
        return {"tool0": [dict(s) for s in self.spheres]}


def grid_fit_spheres(
    verts_mm: np.ndarray, *, cell_mm: float, rmax_mm: float
) -> list[dict[str, Any]]:
    """One tight sphere per occupied voxel: centre at the voxel's centroid, radius to its farthest vertex.

    The union covers the real mesh with little over-reach, which is what a planner needs: spheres
    that are too big refuse reachable poses, and spheres that are too small let the gripper through
    a wall it would have hit.

    The radius is capped at `rmax_mm` because one stray vertex, and every mesh has some, otherwise
    inflates its whole voxel. It has a floor as well: a sphere smaller than a few millimetres costs a
    check and covers nothing.

    Returns spheres in METRES, in the frame the vertices were given in.
    """
    verts = np.asarray(verts_mm, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise SphereFitError(f"vertices must be (N, 3) millimetres, got {verts.shape}")
    if verts.shape[0] == 0:
        return []
    if cell_mm <= 0.0 or rmax_mm <= 0.0:
        raise SphereFitError("cell_mm and rmax_mm must both be positive")

    keys = np.floor(verts / cell_mm).astype(np.int64)
    out: list[dict[str, Any]] = []
    for key in sorted({tuple(k) for k in keys}):
        points = verts[np.all(keys == np.asarray(key), axis=1)]
        centre = points.mean(0)
        radius = min(float(np.max(np.linalg.norm(points - centre, axis=1))), rmax_mm)
        out.append(
            {
                "center": [round(float(x) / 1000.0, 4) for x in centre],
                "radius": round(max(radius, _MIN_RADIUS_MM) / 1000.0, 4),
            }
        )
    return out


def bundle_gripper_arrays(bundle: Path) -> dict[str, np.ndarray]:
    """The end-effector vertex arrays out of a baked collision-mesh bundle.

    A bundle carries the arm links as well, and those are not the gripper: fitting them into the
    `tool0` sphere set would hang the whole robot off the flange.
    """
    if not bundle.is_file():
        raise SphereFitError(
            f"no collision-mesh bundle at {bundle}. Bake one with "
            "scripts/isaac/bake_ur_collision_meshes.py, or fit from a mesh file instead."
        )
    with np.load(bundle, allow_pickle=True) as data:
        missing = [name for name, _, _ in _BUNDLE_PARTS if name not in data]
        if missing:
            raise SphereFitError(
                f"{bundle.name} has no {missing} array(s), so it carries no end effector this can "
                "fit. A bundle without a gripper is an arm bundle."
            )
        return {name: np.asarray(data[name], dtype=np.float64) for name, _, _ in _BUNDLE_PARTS}


def fit_gripper_spheres(
    bundle: Path, *, gripper: str = "", parts: Sequence[tuple[str, float, float]] = _BUNDLE_PARTS
) -> GripperSpheres:
    """Fit the end effector out of a baked bundle.

    This is the path for a gripper this repository has baked, and it is the one the committed sphere
    map is generated with. The same call is what the on-box planner config uses, so the two cannot
    describe different hands.
    """
    arrays = bundle_gripper_arrays(bundle)
    spheres: list[dict[str, Any]] = []
    for name, cell_mm, rmax_mm in parts:
        spheres.extend(grid_fit_spheres(arrays[name], cell_mm=cell_mm, rmax_mm=rmax_mm))
    return GripperSpheres(
        spheres=tuple(spheres),
        source=bundle.name,
        gripper=gripper or bundle.name.replace("_collision_meshes.npz", ""),
    )


def fit_spheres_from_mesh(
    mesh_path: Path,
    *,
    gripper: str,
    cell_mm: float = 34.0,
    rmax_mm: float = 24.0,
    scale_to_mm: float = 1.0,
) -> GripperSpheres:
    """Fit an end effector from a mesh file, for a gripper nobody has baked.

    ``scale_to_mm`` is asked for rather than guessed. A mesh file does not say what its numbers mean,
    the two common cases are metres and millimetres, and the wrong one produces a gripper a thousand
    times too big or too small. Too big refuses everything, which is survivable. Too small models a
    hand the size of a grain of rice, which plans straight through the bin it is reaching into.

    The mesh must already be in the ``tool0`` frame: its origin where the flange is, and its axes the
    tool's. Nothing here can check that, and nothing downstream can either, so it is stated in the
    provenance the caller writes beside the result.
    """
    try:
        import trimesh  # noqa: PLC0415 - an optional dependency, and only this path needs it
    except ImportError as exc:  # pragma: no cover - exercised only where trimesh is absent
        raise SphereFitError(
            "fitting from a mesh file needs trimesh, which requirements.txt pins. Fit from a baked "
            "bundle instead, or install it."
        ) from exc

    if not mesh_path.is_file():
        raise SphereFitError(f"no mesh at {mesh_path}")
    if scale_to_mm <= 0.0:
        raise SphereFitError("scale_to_mm must be positive: 1.0 for a mesh in millimetres, "
                             "1000.0 for one in metres")

    loaded = trimesh.load(str(mesh_path), force="mesh")
    vertices = np.asarray(getattr(loaded, "vertices", ()), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        raise SphereFitError(f"{mesh_path.name} has no vertices this could fit")

    spheres = grid_fit_spheres(vertices * float(scale_to_mm), cell_mm=cell_mm, rmax_mm=rmax_mm)
    return GripperSpheres(spheres=tuple(spheres), source=mesh_path.name, gripper=gripper)
