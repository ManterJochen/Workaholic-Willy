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
    "FLANGE",
    "GripperSpheres",
    "MOUNTING_FACE",
    "SphereFitError",
    "bundle_gripper_arrays",
    "bundle_origin",
    "fit_gripper_spheres",
    "fit_spheres_from_mesh",
    "grid_fit_spheres",
]

#: The sphere set already sits where the hand will be bolted, so nothing is added on the box.
#: True of a bundle baked out of a COMPOSED arm asset: the arm placed the gripper.
FLANGE = "flange"

#: The sphere set starts at the hand's own mounting face, and whatever plate sits between that
#: face and the flange has to be added before the planner sees it. True of a bundle read from a
#: standalone vendor asset, which has no idea what it will be bolted to. The plate's thickness is
#: a bench measurement, the same one `robot.gripper.tool_frame.offset_mm` needs.
MOUNTING_FACE = "mounting_face"

#: The npz key carrying one of the two. Absent in every bundle baked before this existed, and
#: those are all bundles from composed arm assets, so the absent case reads as :data:`FLANGE`.
_ORIGIN_KEY = "gripper__origin"

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
    #: Where these numbers start: :data:`FLANGE` or :data:`MOUNTING_FACE`. It decides whether the
    #: on-box builder may use them as they are, and it is carried rather than inferred because a
    #: sphere set placed one coupling plate too close to the flange looks entirely reasonable.
    origin: str = FLANGE

    @property
    def count(self) -> int:
        return len(self.spheres)

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        radii = [float(s["radius"]) * 1000.0 for s in self.spheres]
        if not radii:
            return f"{self.gripper}: no spheres fitted from {self.source}"
        origin = (
            "at the flange" if self.origin == FLANGE
            else "from the mounting face, so a coupling has still to be added"
        )
        return (
            f"{self.gripper}: {self.count} sphere(s) from {self.source}, "
            f"radius {min(radii):.1f} to {max(radii):.1f} mm, {origin}"
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


def bundle_origin(bundle: Path) -> str:
    """Whether a bundle's gripper arrays start at the flange or at the hand's mounting face.

    Bundles baked before this key existed all came out of composed arm assets, where the arm had
    already placed the hand, so their absent case is :data:`FLANGE` and reading it that way is a
    statement about those files rather than a lenient default.
    """
    if not bundle.is_file():
        raise SphereFitError(f"no collision-mesh bundle at {bundle}")
    with np.load(bundle, allow_pickle=True) as data:
        if _ORIGIN_KEY not in data:
            return FLANGE
        value = str(np.asarray(data[_ORIGIN_KEY]).reshape(-1)[0])
    if value not in (FLANGE, MOUNTING_FACE):
        raise SphereFitError(
            f"{bundle.name} declares origin {value!r}, which is neither {FLANGE!r} nor "
            f"{MOUNTING_FACE!r}. The on-box builder decides whether to add a coupling from this, "
            "so an unknown value is not something to guess past."
        )
    return value


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
        origin=bundle_origin(bundle),
    )


def fit_spheres_from_mesh(
    mesh_path: Path,
    *,
    gripper: str,
    cell_mm: float,
    rmax_mm: float,
    scale_to_mm: float,
    origin: str = MOUNTING_FACE,
) -> GripperSpheres:
    """Fit an end effector from a mesh file, for a gripper nobody has baked.

    ⚠ **NOTHING HERE HAS A DEFAULT, AND THAT IS THE POINT.** This is the path a customer with a
    new hand takes, from the one-liner in the module docstring, and every one of these is a fact
    the loader cannot recover from the file.

    ``scale_to_mm``: a mesh does not say what its numbers mean, the two common cases are metres
    and millimetres, and the wrong one produces a gripper a thousand times too big or too small.
    Too big refuses everything, which is survivable. Too small models a hand the size of a grain
    of rice, which plans straight through the bin it is reaching into.

    ``cell_mm`` and ``rmax_mm``: the voxel a sphere is fitted per, and the cap on its radius. They
    had defaults of 34.0 and 24.0, which are the 2F-85's FINGER cell size paired with its PALM
    radius cap: a combination describing no part of any gripper, including the one both halves
    were measured from. A palm is one large solid and a finger is a thin blade, and one setting
    for both either buries the fingers inside a sphere the size of the palm or covers the palm in
    dozens of tiny ones. :data:`_BUNDLE_PARTS` is what those numbers look like when they are
    stated per part; a single mesh has to be told.

    ``origin``: whether the mesh already sits at the flange or starts at the hand's own mounting
    face. It defaults to :data:`MOUNTING_FACE` because that is what a vendor mesh is, and getting
    it wrong puts every sphere one coupling plate too close to the flange, which is optimistic and
    looks entirely reasonable.

    The mesh must already be in the ``tool0`` AXES: closing on x, approach on y, binormal on z.
    Nothing here can check that, and nothing downstream can either, so it is stated in the
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

    if origin not in (FLANGE, MOUNTING_FACE):
        raise SphereFitError(
            f"origin must be {FLANGE!r} or {MOUNTING_FACE!r}, got {origin!r}"
        )

    spheres = grid_fit_spheres(vertices * float(scale_to_mm), cell_mm=cell_mm, rmax_mm=rmax_mm)
    return GripperSpheres(
        spheres=tuple(spheres), source=mesh_path.name, gripper=gripper, origin=origin
    )
