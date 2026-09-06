"""The grasp label set: every grasp the geometry admits, per scene.

Sampling here is not a heuristic search. For each primitive the antipodal closing axes are known in
closed form, so the candidate set is enumerated rather than hunted for:

* box: contacts with exactly antiparallel normals exist only on opposite faces, so there are exactly
  three closing axes, the three body axes. Anything else lands on faces whose normals are 90 deg
  apart and is not a grasp at any friction coefficient.
* cylinder: the barrel admits any closing axis perpendicular to the body axis and through it (the
  radial normals are antiparallel only there), plus the cap pair along the body axis.
* sphere: every line through the centre, and nothing else.

The anchor and the approach are then sampled, because those are continuous. The approach is not
pre-filtered by direction: an approach from below is refused by ``BELOW_TABLE`` or
``APPROACH_BLOCKED``, so the rejection is measured rather than assumed.

The verdict is split to keep the cost down: span and antipodality depend only on (anchor, axis), so
a line that cannot be gripped at all is refused once instead of once per approach direction.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, GRASP_LABELS_LOG_FILE
from datagen.assets.mesh_geometry import load_mesh_shape
from datagen.grasps.shapes import (
    Solid,
    solid_from_mesh,
    solid_from_scene,
    solids_from_parts,
    wall_solid,
)
from datagen.grasps.verdict import (
    SLIM_CUP,
    STANDARD_CUP,
    JawGrasp,
    PROCEDURAL_JAWS,
    JawModel,
    Rejection,
    SuctionCup,
    SuctionGrasp,
    Verdict,
    check_jaw_grasp,
    check_suction_grasp,
    suction_payload_ok,
)

__all__ = [
    "DEFAULT_DENSITY", "DENSE_DENSITY", "GRID_DENSITY", "DENSITIES", "GraspLabel", "LabelDensity",
    "SceneAssets", "SceneGeometry",
    "label_scene", "label_dataset", "scene_assets", "scene_geometry",
]

logger = create_logger("datagen.grasps.labels", GRASP_LABELS_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Approach directions sampled in the plane perpendicular to the closing axis. Twelve is 30 deg
#: apart: finer than the tolerance any downstream comparison uses, coarse enough to stay affordable.
_APPROACH_AZIMUTHS = 12
#: Anchor offsets along the remaining axes, as a fraction of that half-extent. The centre plus a pair
#: either side: enough that an elongated object has labels away from its middle.
_ANCHOR_FRACTIONS = (0.0, -0.35, 0.35)
#: Closing axes sampled around a cylinder barrel or through a sphere. Six covers 180 deg in 30 deg
#: steps; c and -c are the same grasp, so a full turn would only duplicate.
_RADIAL_AXES = 6
#: Closing axes sampled over a real mesh, where no closed form exists. This is where the reference
#: stops being complete: for a box the three axes are provably all of them, whereas a scanned
#: surface admits a continuum and eighteen directions are a sample of it. Every label produced is
#: still a real grasp; what does not hold for a mesh is that a missing label proves a missing grasp.
#:
#: Eighteen is a Fibonacci hemisphere, which spaces directions about 27 deg apart: the same order as
#: the 30 deg the cylinder branch uses, so a mesh is not sampled coarser than a primitive.
_MESH_CLOSING_AXES = 18
#: Cup landings sampled over a mesh surface, area-weighted. The same order as the landings a box
#: yields in `_suction_contacts`, so neither shape family dominates the suction labels.
_MESH_SUCTION_CONTACTS = 24

#: How close to an existing ring sample a guaranteed approach may land before it is dropped as a
#: duplicate. Two degrees: far below any tolerance a consumer compares directions at, so this only ever
#: removes a direction the ring already had.
_DUPLICATE_APPROACH_COS = 0.99939                                       # cos(2 deg)

#: How far from vertical a closing axis may be before "the approach closest to straight down" stops
#: meaning anything. The perpendicular ring's most-downward direction has length ``sin(angle between
#: the axis and the vertical)``, so 0.1 is an axis within 5.7 deg of upright: the whole ring is then
#: horizontal and there is no near-vertical approach. Such an axis is skipped rather than faked.
_VERTICAL_PROJECTION_FLOOR = 0.1


@dataclass(frozen=True, slots=True)
class LabelDensity:
    """How finely one object is sampled, and whether the vertical approach is guaranteed.

    A parameter rather than an edit. These numbers were module constants, so changing one changed
    every label set the package produces, silently. Sampling is a property of a dataset: two corpora
    sampled differently are not comparable, and the denser one is not thereby better. The defaults
    are the module constants above, so relabelling a dataset reproduces the same file.

    Why a mesh earns no label, and it is not the table. The dominant refusal is `TOO_WIDE`: the
    object spans more than the jaw opens along the line that was tried. Thin flat objects look as
    though they should fail on `BELOW_TABLE`, because the fingers meet the table under them, and
    they do not. That is why the two levers here are the closing axis and the anchor grid: both
    change which line gets tried.

    Density does not buy a direction balance, it costs one. `_approaches` samples the ring around
    the closing axis uniformly in azimuth and the vertical is one sample of it, so a finer ring
    makes the top-down grasp a smaller share of the set even while its absolute count rises. What
    density buys is coverage of objects.
    """

    #: Approach directions around the closing axis, evenly spaced in azimuth.
    approach_azimuths: int = _APPROACH_AZIMUTHS
    #: Anchor offsets along the free axis, as a fraction of its half-extent. With `anchor_grid` the
    #: grid is the product of these on both free axes, so five of them is 25 points and seven is 49.
    #:
    #: A finer grid does find grasps a coarser one steps over on meshes that earn no label in any
    #: rest pose at this setting, and it does not saturate over the range measured. Neither does the
    #: closing-axis count below.
    #:
    #: Neither is raised here, and the reason is consistency rather than cost. The screen that
    #: decides which meshes enter a corpus at all runs at this setting, so labelling more finely
    #: would label finely exactly the objects the coarse screen already refused. The screen and the
    #: labelling have to move together, which makes this a second pass rather than a flag flip.
    anchor_fractions: tuple[float, ...] = _ANCHOR_FRACTIONS
    #: Closing axes sampled over a real mesh, on top of its three principal axes.
    #:
    #: This count does not saturate cleanly against the meshes it rescues, and the samples behind
    #: that are too small to call one: doubling the axes finds meshes that earn no label at all, at
    #: close to double the time per mesh. What the numbers support is that the first doubling is
    #: worth its cost and that everything above it is undecided.
    #:
    #: Settling it costs no render time: labelling is a separate command from the build, so the same
    #: corpus can be labelled twice and the two label sets compared.
    mesh_closing_axes: int = _MESH_CLOSING_AXES
    #: Closing axes sampled around a cylinder barrel or through a sphere.
    radial_axes: int = _RADIAL_AXES
    #: Cup landings sampled over a mesh surface for suction.
    mesh_suction_contacts: int = _MESH_SUCTION_CONTACTS
    #: Add, per closing axis, the ring direction closest to straight down. Smaller than it sounds:
    #: the even ring already carries the exact vertical for most axes, and this fires only for the
    #: few whose ring is out of phase with it. Off by default, because it changes the label set
    #: rather than its resolution. `_approaches` states why the ring already carries it.
    vertical_approach: bool = False
    #: Spread anchors over the plane the closing axis leaves free, instead of along one line in it.
    #:
    #: `_anchors` walks the longest free axis and offers one point per anchor fraction along it. On
    #: a large object that line stays where the object is widest, which is exactly where the jaw
    #: does not fit, so the object earns no label at all however many approaches are tried. A grid
    #: also reaches the narrow parts, and the objects it gains are the large ones a line down the
    #: centroid could never fit.
    #:
    #: Anchors that fall outside the solid are not a problem to solve here: `check_jaw_grasp` asks
    #: whether the anchor lies between the two contacts and returns ANCHOR_OUTSIDE when it does not.
    #: A grid buys reach at the price of more cheap refusals.
    #:
    #: Default False, so a label set written with the defaults is reproducible from the same call.
    anchor_grid: bool = False

    def as_row(self) -> dict:
        """The stamp a dataset carries, so a label set can say how it was sampled."""
        return {
            "approach_azimuths": self.approach_azimuths,
            "anchor_fractions": list(self.anchor_fractions),
            "mesh_closing_axes": self.mesh_closing_axes,
            "radial_axes": self.radial_axes,
            "mesh_suction_contacts": self.mesh_suction_contacts,
            "vertical_approach": self.vertical_approach,
            "anchor_grid": self.anchor_grid,
        }


#: The default sampling, straight from the module constants: relabelling writes the same bytes.
DEFAULT_DENSITY = LabelDensity()

#: The wide sampling, for a corpus meant to teach a model rather than to score one: more approaches,
#: more anchors and more closing axes, with the top-down approach guaranteed rather than left to the
#: ring.
DENSE_DENSITY = LabelDensity(
    approach_azimuths=24,
    anchor_fractions=(0.0, -0.2, 0.2, -0.4, 0.4),
    mesh_closing_axes=30,
    radial_axes=8,
    vertical_approach=True,
)

#: The names the CLI accepts. A named preset rather than a flag per field: the fields only mean
#: anything together, and a dataset labelled with some of one preset and some of another is a corpus
#: nobody chose. This one is `dense` plus the silhouette grid, and it is a third preset rather than
#: a change to `dense`: a label report is stamped with the preset's fields, so folding the grid into
#: `dense` would make two different label sets answer to one name.
GRID_DENSITY = dataclasses.replace(DENSE_DENSITY, anchor_grid=True)

DENSITIES: dict[str, LabelDensity] = {
    "default": DEFAULT_DENSITY, "dense": DENSE_DENSITY, "grid": GRID_DENSITY,
}


@dataclass(frozen=True, slots=True)
class GraspLabel:
    """One grasp the geometry admits. ``kind`` is ``jaw`` or ``suction``."""

    scene_id: str
    instance_id: int
    kind: str
    position_mm: tuple[float, ...]
    approach: tuple[float, ...]
    closing_axis: tuple[float, ...]
    width_mm: float
    approach_tilt_deg: float
    contact_angle_deg: float
    cup: str = ""
    payload_ok: bool = True
    #: Which part of the object this grasp touches: ``handle``, ``body``, ``grip``, ``neck``,
    #: ``head``, or ``""`` for a single-primitive object where the distinction does not exist.
    #:
    #: This is the supervision a part-aware model needs: the same mug yields grasps on its handle
    #: and on its body, and only this field tells them apart. Empty by default, so a label from an
    #: object with no part structure does not claim to be a body grasp.
    part_role: str = ""

    def as_row(self) -> dict:
        return {
            "scene_id": self.scene_id, "instance_id": self.instance_id, "kind": self.kind,
            "position_mm": [round(v, 4) for v in self.position_mm],
            "approach": [round(v, 6) for v in self.approach],
            "closing_axis": [round(v, 6) for v in self.closing_axis],
            "width_mm": round(self.width_mm, 4),
            "approach_tilt_deg": round(self.approach_tilt_deg, 3),
            "part_role": self.part_role,
            "contact_angle_deg": round(self.contact_angle_deg, 3),
            "cup": self.cup, "payload_ok": self.payload_ok,
        }


@dataclass(frozen=True, slots=True)
class SceneGeometry:
    """A scene as exact solids: the settled objects plus everything that can be in the way."""

    scene_id: str
    family: str
    objects: dict[int, Solid]
    walls: tuple[Solid, ...]
    #: Extra solids belonging to an object, keyed by instance id: a mug's handle, a jug's neck.
    #: Absent for every single-primitive object, which is the overwhelming majority.
    #:
    #: The entry does not repeat the primary solid in `objects`; these are the additional parts, and
    #: `parts_of` returns the complete set.
    parts: dict[int, tuple[Solid, ...]] = field(default_factory=dict)
    #: The semantic role of each solid, keyed by `(instance_id, part index)` where index 0 is the
    #: primary solid in `objects`. Kept beside the geometry rather than on `Solid` so the analytic
    #: shape type stays purely geometric.
    part_roles: dict[tuple[int, int], str] = field(default_factory=dict)

    def parts_of(self, instance_id: int) -> list[Solid]:
        """Every solid making up one object: the primary one first, then any extra parts."""
        primary = self.objects.get(instance_id)
        extra = list(self.parts.get(instance_id, ()))
        return ([primary] if primary is not None else []) + extra

    def role_of(self, instance_id: int, part_index: int) -> str:
        """The semantic role of one part, or ``""`` when the object has no part structure."""
        return self.part_roles.get((instance_id, part_index), "")

    def obstacles_for(self, instance_id: int) -> list[Solid]:
        """Everything a grasp on this object must not hit: other objects, their parts, and walls.

        An object's own parts are absent here on purpose. `label_jaw_grasps` adds the siblings of
        whichever part it is targeting, which is the only correct pairing; including them
        unconditionally would make every body grasp collide with its own handle.
        """
        others: list[Solid] = []
        for index, solid in self.objects.items():
            if index == instance_id:
                continue
            others.append(solid)
            others.extend(self.parts.get(index, ()))
        return others + list(self.walls)


def scene_geometry(
    payload: dict, extents: Mapping[str, Sequence[float]], scene_id: str, *,
    parts: "Mapping[str, Sequence[Any]] | None" = None,
    mesh_paths: "Mapping[str, str] | None" = None,
) -> SceneGeometry:
    """Build the solids for one ``scene.json``. Dropped objects are absent; they were hidden.

    ``parts`` maps an asset id to the primitives a composite object is built from, as recorded in
    the manifest. An asset absent from it is a single primitive and takes the primitive path, which
    is almost everything in a dataset.

    ``mesh_paths`` does the same job for scanned objects, mapping an asset id to the file on disk.
    Both come from the manifest and both exist for one reason: the labeller must intersect the shape
    the renderer authored, and neither a part structure nor a mesh can be recovered from an asset's
    name.
    """
    dropped = set(payload.get("dropped_objects", []))
    settled = payload.get("settled_poses_mm_xyzw", {})
    part_index = dict(parts or {})
    objects: dict[int, Solid] = {}
    extra_parts: dict[int, tuple[Solid, ...]] = {}
    part_roles: dict[tuple[int, int], str] = {}
    for index, spec in enumerate(payload["spec"]["objects"]):
        if index in dropped or str(index) not in settled:
            continue
        extent = extents.get(spec["asset_id"])
        if extent is None:
            continue
        position, orientation = settled[str(index)]
        composite = part_index.get(spec["asset_id"])
        if composite:
            solids = solids_from_parts(
                composite, position, orientation, instance_id=index,
                asset_id=spec["asset_id"], mass_kg=float(spec.get("mass_kg", 0.0)),
            )
            objects[index] = solids[0]
            if len(solids) > 1:
                extra_parts[index] = tuple(solids[1:])
            for part_i, part in enumerate(composite):
                part_roles[(index, part_i)] = str(getattr(part, "role", ""))
            continue
        asset_id = str(spec["asset_id"])
        mesh_path = mesh_paths.get(asset_id) if mesh_paths else None
        if mesh_path is not None:
            # The mesh branch, and the lookup is by asset id rather than by name for one reason:
            # `primitive_for_kind` maps anything unlisted to "box", so a scanned object with no
            # entry here would be labelled as a cuboid it is not, and the renderer, deriving its
            # shape the same way, would agree. Two components consistent with each other and both
            # wrong about the object.
            shape = load_mesh_shape(mesh_path)
            if shape is None:
                # `load_mesh_shape` has logged which refusal this was. Raising rather than
                # skipping, and matching what the renderer does with the same mesh: labelling runs
                # on an already rendered dataset, so a mesh that cannot be closed here could not
                # have been drawn there either. Reaching this line means the asset bank and the
                # renderer disagree, and skipping would leave a scene whose picture holds an object
                # its labels do not.
                raise ValueError(
                    f"asset {asset_id!r} has a mesh at {mesh_path!r} that cannot be closed into a "
                    f"solid, yet the scene was rendered with it. The asset bank should have excluded "
                    f"it: see MeshAssetBank.load(require_closed=True)."
                )
            objects[index] = solid_from_mesh(
                shape, position, orientation,
                instance_id=index, asset_id=asset_id, mass_kg=float(spec.get("mass_kg", 0.0)),
            )
            continue
        if asset_id.startswith(("gso_", "ycb_", "objaverse_")):
            # A sourced asset that reached here without a mesh path: the manifest lost it, or the
            # caller did not pass one. Refusing is the only safe answer for the reason above.
            raise NotImplementedError(
                f"asset {asset_id!r} looks like a real mesh but no mesh path was supplied, so the "
                f"labeller would fall back to deriving a primitive from its name and label a box. "
                f"Pass `mesh_paths` from the manifest (see `_mesh_paths`)."
            )
        objects[index] = solid_from_scene(
            asset_id.rsplit("_", 1)[-1], extent, position, orientation,
            instance_id=index, asset_id=asset_id, mass_kg=float(spec.get("mass_kg", 0.0)),
        )
    walls = tuple(
        wall_solid(wall["center_mm"], wall["half_extents_mm"], name=wall.get("name", "wall"))
        for wall in payload["spec"].get("bin_walls", ())
    )
    return SceneGeometry(
        scene_id, str(payload["spec"].get("family", "")), objects, walls,
        parts=extra_parts, part_roles=part_roles,
    )


def _principal_axes(vertices_mm: np.ndarray) -> np.ndarray:
    """The mesh's three principal directions, in its own body frame, longest first.

    Eigenvectors of the vertex covariance. Signs are made deterministic, largest component positive,
    because an eigen-decomposition is free to flip one and a flipped closing axis is the same grasp
    with a different serialised label: two identical runs would produce different files.
    """
    centred = vertices_mm - vertices_mm.mean(axis=0)
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    axes = np.asarray(right, dtype=np.float64)
    for row in range(axes.shape[0]):
        if axes[row][int(np.argmax(np.abs(axes[row])))] < 0.0:
            axes[row] = -axes[row]
    return axes


def _fibonacci_hemisphere(count: int) -> np.ndarray:
    """``count`` directions spread evenly over a half-sphere, deterministically.

    A half-sphere because a closing axis and its negative are the same grasp. The golden-angle
    spiral is used rather than a random sample so that two runs of one dataset produce the same
    labels: a reference that is not reproducible is not a reference.
    """
    index = np.arange(count, dtype=np.float64) + 0.5
    z = index / count                                   # (0, 1]: the upper half only
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = np.pi * (1.0 + 5.0 ** 0.5) * index          # the golden angle
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])


def _closing_axes(solid: Solid, density: LabelDensity = DEFAULT_DENSITY) -> list[np.ndarray]:
    """The closing directions this primitive admits antipodal contacts on: enumerated, not searched."""
    R = solid.rotation
    if solid.mesh is not None:
        # Sampled, not enumerated; see `_MESH_CLOSING_AXES`. The object's three principal axes go
        # in first: an elongated scan is most often gripped across its long axis, and a spiral that
        # straddles that direction would miss exactly the grasps that matter most.
        principal = _principal_axes(solid.mesh.vertices_mm)
        sampled = _fibonacci_hemisphere(density.mesh_closing_axes)
        return [R @ axis for axis in np.vstack([principal, sampled])]
    if solid.kind == "box":
        return [R[:, i].copy() for i in range(3)]
    angles = np.linspace(0.0, np.pi, density.radial_axes, endpoint=False)
    if solid.kind == "cylinder":
        radial = [R[:, 0] * np.cos(a) + R[:, 1] * np.sin(a) for a in angles]
        return [*radial, R[:, 2].copy()]        # barrel pairs, plus the two end caps
    return [R[:, 0] * np.cos(a) + R[:, 1] * np.sin(a) for a in angles] + [R[:, 2].copy()]


def _anchors(solid: Solid, axis: np.ndarray,
             density: LabelDensity = DEFAULT_DENSITY) -> list[np.ndarray]:
    """Anchor points along the object, spread across the axes the closing axis leaves free.

    A mesh takes the general branch below with no special case, and that is deliberate: the branch
    reads only the axis-aligned half-extents and the body frame, which a mesh has like anything
    else. An anchor that lands in a hollow, as a mug's centre of bounds does, is not a problem to
    solve here. `check_jaw_grasp` asks whether the anchor lies between the two contacts, which it
    does, and rejects it with `ANCHOR_OUTSIDE` when it does not.
    """
    if solid.kind == "sphere":
        return [solid.centre_mm.copy()]
    R, h = solid.rotation, solid.half_extent_mm
    # The free body axis with the most room; a cylinder's barrel grasp slides along its length.
    free = [i for i in range(3) if abs(float(R[:, i] @ axis)) < 0.9]
    if not free:
        return [solid.centre_mm.copy()]
    if solid.kind == "cylinder" and abs(float(R[:, 2] @ axis)) < 0.9:
        longest = 2
    else:
        longest = max(free, key=lambda i: float(h[i]))
    if not density.anchor_grid or len(free) < 2:
        return [solid.centre_mm + fraction * float(h[longest]) * R[:, longest]
                for fraction in density.anchor_fractions]
    # The second free axis, which the line above never moves along. See `LabelDensity.anchor_grid`
    # for why a line down the longest axis alone leaves large objects unlabelled. The grid is the
    # product of the same fractions on both free axes, so it reduces to the line when the fractions
    # do, and its centre point is the same point.
    other = next(i for i in free if i != longest)
    return [solid.centre_mm
            + along * float(h[longest]) * R[:, longest]
            + across * float(h[other]) * R[:, other]
            for along in density.anchor_fractions
            for across in density.anchor_fractions]


def _approaches(axis: np.ndarray,
                density: LabelDensity = DEFAULT_DENSITY) -> Iterator[np.ndarray]:
    """Unit vectors perpendicular to ``axis``: the even ring, then the top-down one if asked for.

    The top-down grasp is not missing from the ring. ``u`` is ``axis x z_hat``, which is horizontal,
    so ``v = axis x u`` is the most-downward direction the ring admits, and it lands on a sample
    exactly whenever ``approach_azimuths`` is divisible by four.

    What a finer ring changes is dilution, not absence: the vertical is one direction in
    ``approach_azimuths``, so sampling more finely makes it a smaller share of the set while its
    absolute count rises. A direction balance is not something this switch can buy.

    It is kept for the closing axes whose ring is out of phase with the vertical, where this is the
    top-down grasp and nothing else produces it. It is skipped, not faked, when the axis is too
    close to upright for the question to have an answer.
    """
    helper = np.array([0.0, 0.0, 1.0]) if abs(float(axis[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, helper)
    u /= float(np.linalg.norm(u))
    v = np.cross(axis, u)
    ring = [np.cos(angle) * u + np.sin(angle) * v
            for angle in np.linspace(0.0, 2.0 * np.pi, density.approach_azimuths, endpoint=False)]
    yield from ring
    if not density.vertical_approach:
        return
    down = np.array([0.0, 0.0, -1.0])
    projected = down - float(down @ axis) * axis
    length = float(np.linalg.norm(projected))
    if length < _VERTICAL_PROJECTION_FLOOR:
        return
    projected = projected / length
    if any(float(projected @ sample) > _DUPLICATE_APPROACH_COS for sample in ring):
        return
    yield projected


def label_jaw_grasps(
    geometry: SceneGeometry, instance_id: int, *, model: JawModel | None = None,
    density: LabelDensity = DEFAULT_DENSITY,
) -> tuple[list[GraspLabel], dict[str, int]]:
    """Every jaw grasp the geometry admits for one object, plus why the rest were refused."""
    model = model or JawModel()
    solids = geometry.parts_of(instance_id)
    if not solids:
        return [], {}
    scene_obstacles = geometry.obstacles_for(instance_id)
    labels: list[GraspLabel] = []
    rejected: dict[str, int] = {}

    for part_index, target in enumerate(solids):
        # The object's other parts are obstacles for this one. That pairing is the whole correctness
        # argument for part labels: a grip on the handle that passes through the body is not a grasp,
        # and this is what rejects it.
        obstacles = scene_obstacles + [s for i, s in enumerate(solids) if i != part_index]
        role = geometry.role_of(instance_id, part_index)
        labels_here, rejected_here = _label_one_solid(
            geometry, instance_id, target, obstacles, model=model, part_role=role, density=density,
        )
        labels.extend(labels_here)
        for reason, count in rejected_here.items():
            rejected[reason] = rejected.get(reason, 0) + count
    return labels, rejected


def _label_one_solid(
    geometry: SceneGeometry, instance_id: int, target: Solid, obstacles: list[Solid], *,
    model: JawModel, part_role: str = "", density: LabelDensity = DEFAULT_DENSITY,
) -> tuple[list[GraspLabel], dict[str, int]]:
    """Every jaw grasp one solid admits, stamped with the part role, plus the refusal counts."""
    labels: list[GraspLabel] = []
    rejected: dict[str, int] = {}

    # The ring's size, needed by the stage-one shortcut below: a line refused outright costs the
    # approaches it would have had, and `vertical_approach` adds at most one more to that ring.
    approaches_per_axis = density.approach_azimuths + (1 if density.vertical_approach else 0)
    for axis in _closing_axes(target, density):
        axis = axis / float(np.linalg.norm(axis))
        for anchor in _anchors(target, axis, density):
            # Stage one: does the line admit a grip at all? Independent of the approach direction, so
            # a failure here costs one check instead of one per approach direction.
            probe = check_jaw_grasp(
                JawGrasp(anchor, _first_perpendicular(axis), axis, 0.0), target, (), model=model)
            if not probe.ok and probe.reason in (
                Rejection.LINE_MISSES_OBJECT, Rejection.ANCHOR_OUTSIDE,
                Rejection.TOO_WIDE, Rejection.TOO_THIN, Rejection.NOT_ANTIPODAL,
            ):
                rejected[probe.reason] = rejected.get(probe.reason, 0) + approaches_per_axis
                continue
            for approach in _approaches(axis, density):
                grasp = JawGrasp(anchor, approach, axis, probe.object_span_mm)
                verdict = check_jaw_grasp(grasp, target, obstacles, model=model)
                if verdict.ok:
                    labels.append(_label_from(
                        geometry.scene_id, instance_id, "jaw", grasp, verdict, part_role=part_role,
                    ))
                else:
                    rejected[verdict.reason] = rejected.get(verdict.reason, 0) + 1
    return labels, rejected


def _first_perpendicular(axis: np.ndarray) -> np.ndarray:
    helper = np.array([0.0, 0.0, 1.0]) if abs(float(axis[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    v = np.cross(axis, helper)
    return v / float(np.linalg.norm(v))


def _label_from(scene_id: str, instance_id: int, kind: str, grasp, verdict: Verdict,
                *, cup: str = "", payload_ok: bool = True, part_role: str = "") -> GraspLabel:
    axis = getattr(grasp, "closing_axis", np.zeros(3))
    return GraspLabel(
        scene_id=scene_id, instance_id=instance_id, kind=kind, part_role=part_role,
        position_mm=tuple(float(v) for v in np.asarray(grasp.position_mm)),
        approach=tuple(float(v) for v in np.asarray(grasp.approach)),
        closing_axis=tuple(float(v) for v in np.asarray(axis)),
        width_mm=float(verdict.object_span_mm),
        approach_tilt_deg=float(verdict.approach_tilt_deg),
        contact_angle_deg=float(verdict.contact_angle_deg),
        cup=cup, payload_ok=payload_ok,
    )


def _mesh_suction_contacts(
    solid: Solid, density: LabelDensity = DEFAULT_DENSITY,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Cup landings spread over a scanned surface, area-weighted and without an RNG.

    Systematic sampling of the cumulative area rather than a random draw: a big flat face, which is
    where a cup can seal, gets proportionally many landings, and the same mesh yields the same
    landings in every run. Each landing is a triangle's own centroid and its own normal, so the
    contact is on the surface, which is the first thing `check_suction_grasp` asks.
    """
    mesh = solid.mesh
    assert mesh is not None
    triangles = mesh.vertices_mm[mesh.faces]
    centroids = triangles.mean(axis=1)
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    total = float(areas.sum())
    if total <= 0.0:
        return
    cumulative = np.cumsum(areas)
    targets = (np.arange(density.mesh_suction_contacts, dtype=np.float64) + 0.5) * (
        total / density.mesh_suction_contacts)
    for face in np.unique(np.searchsorted(cumulative, targets)):
        index = int(min(face, len(centroids) - 1))
        yield (solid.centre_mm + solid.rotation @ centroids[index],
               solid.rotation @ mesh.face_normals[index])


def _suction_contacts(
    solid: Solid, density: LabelDensity = DEFAULT_DENSITY,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Candidate cup landings: face centres and a spread around them, with the outward normal."""
    if solid.mesh is not None:
        yield from _mesh_suction_contacts(solid, density)
        return
    R, h = solid.rotation, solid.half_extent_mm
    if solid.kind == "sphere":
        for axis in (0, 1, 2):
            for sign in (1.0, -1.0):
                normal = sign * R[:, axis]
                yield solid.centre_mm + normal * float(h[0]), normal
        return
    if solid.kind == "cylinder":
        for sign in (1.0, -1.0):                       # the two caps
            normal = sign * R[:, 2]
            centre = solid.centre_mm + normal * float(h[2])
            yield centre, normal
            for offset in (0.4, -0.4):
                yield centre + offset * float(h[0]) * R[:, 0], normal
        for angle in np.linspace(0.0, 2 * np.pi, 6, endpoint=False):   # the barrel, measured not assumed
            normal = R[:, 0] * np.cos(angle) + R[:, 1] * np.sin(angle)
            yield solid.centre_mm + normal * float(h[0]), normal
        return
    for axis in range(3):
        for sign in (1.0, -1.0):
            normal = sign * R[:, axis]
            centre = solid.centre_mm + normal * float(h[axis])
            others = [i for i in range(3) if i != axis]
            yield centre, normal
            for other in others:
                for fraction in (0.45, -0.45):
                    yield centre + fraction * float(h[other]) * R[:, other], normal


def label_suction_grasps(
    geometry: SceneGeometry, instance_id: int, *, cups: Sequence[SuctionCup] = (STANDARD_CUP, SLIM_CUP),
    density: LabelDensity = DEFAULT_DENSITY,
) -> tuple[list[GraspLabel], dict[str, int]]:
    """Every cup landing the geometry admits, per cup profile and per part.

    Parts are looped with the object's other parts as obstacles, the same pairing the jaw labeller
    uses. `geometry.objects[instance_id]` is only the primary solid and `obstacles_for` omits the
    object's own others, so without this loop a landing on a mug's body that passes through its
    handle counts as valid, a part that is not the primary solid earns no landing at all, and no
    suction label carries a `part_role`. One object must not have two rules.
    """
    solids = geometry.parts_of(instance_id)
    if not solids:
        return [], {}
    scene_obstacles = geometry.obstacles_for(instance_id)
    labels: list[GraspLabel] = []
    rejected: dict[str, int] = {}
    for part_index, target in enumerate(solids):
        # The object's other parts are obstacles for this one, which is the whole correctness argument:
        # a cup on the body that passes through the handle is not a landing.
        obstacles = scene_obstacles + [s for i, s in enumerate(solids) if i != part_index]
        role = geometry.role_of(instance_id, part_index)
        for cup in cups:
            for position, normal in _suction_contacts(target, density):
                grasp = SuctionGrasp(position - normal * 0.2, -normal)
                verdict = check_suction_grasp(grasp, target, obstacles, cup=cup)
                if verdict.ok:
                    labels.append(_label_from(
                        geometry.scene_id, instance_id, "suction", grasp, verdict,
                        cup=cup.name, payload_ok=suction_payload_ok(target.mass_kg, cup),
                        part_role=role))
                else:
                    key = f"{cup.name}:{verdict.reason}"
                    rejected[key] = rejected.get(key, 0) + 1
    return labels, rejected


def label_scene(
    payload: dict, extents: Mapping[str, Sequence[float]], scene_id: str, *,
    model: JawModel | None = None,
    parts: "Mapping[str, Sequence[Any]] | None" = None,
    mesh_paths: "Mapping[str, str] | None" = None,
    density: LabelDensity = DEFAULT_DENSITY,
) -> tuple[list[GraspLabel], dict[str, int]]:
    """All grasp labels for one scene, jaw and suction, with the pooled rejection histogram."""
    geometry = scene_geometry(payload, extents, scene_id, parts=parts, mesh_paths=mesh_paths)
    labels: list[GraspLabel] = []
    rejected: dict[str, int] = {}
    for instance_id in sorted(geometry.objects):
        for produced, refused in (
            label_jaw_grasps(geometry, instance_id, model=model, density=density),
            label_suction_grasps(geometry, instance_id, density=density),
        ):
            labels.extend(produced)
            for reason, count in refused.items():
                rejected[reason] = rejected.get(reason, 0) + count
    return labels, rejected


def _scene_order(directories: list[Path], *, balanced: bool) -> list[Path]:
    """Scene directories to label, in the order they will be labelled.

    A prefix is not a sample. Scene directories are named `<family>_<number>`, so `sorted()` groups
    them by family and a run that stops early takes every `bin` scene and no `sparse` one. A subset
    picked that way is a handful of asset groups, and a number measured on it describes them rather
    than the corpus.

    So a budgeted run walks the families round robin and a complete one keeps `sorted()` exactly.
    Without a budget the order is unchanged, byte for byte.
    """
    if not balanced:
        return directories
    families: dict[str, list[Path]] = {}
    for directory in directories:
        families.setdefault(directory.name.rsplit("_", 1)[0], []).append(directory)
    out: list[Path] = []
    for index in range(max(len(v) for v in families.values()) if families else 0):
        for family in sorted(families):
            if index < len(families[family]):
                out.append(families[family][index])
    return out


def label_dataset(root: Path, *, limit: int | None = None, label_budget: int | None = None,
                  density: LabelDensity = DEFAULT_DENSITY, jaw: str | None = None) -> dict:
    """Write ``root/grasps.jsonl``. Pure geometry, no GPU, no Isaac, re-runnable in minutes.

    ``jaw`` names an entry of :data:`PROCEDURAL_JAWS` and labels the scenes for that gripper instead
    of the 2F-85. It exists for one job: the learned generator takes a gripper description as an
    input, and with a single gripper that input is constant across every sample, carries no
    gradient, and leaves a seam that looks wired while doing nothing.

    A name, never a ``JawModel`` object, and the reason is the output file. Every layer below this
    one already accepts a model object; accepting one here too would be dangerous, because this
    function writes ``grasps.jsonl`` unconditionally and a run against a second gripper would
    overwrite the corpus labels with it. Taking a name makes the output file a function of the
    input, so a non-default gripper writes ``grasps_jaw_<name>.jsonl`` and cannot reach the corpus
    labels at all.
    """
    model = None
    if jaw is not None:
        if jaw not in PROCEDURAL_JAWS:
            raise ValueError(f"unknown jaw {jaw!r}; choose from "
                             f"{', '.join(sorted(PROCEDURAL_JAWS))}")
        model = PROCEDURAL_JAWS[jaw]
    stem = "grasps" if jaw is None else f"grasps_jaw_{jaw}"
    report_name = ("grasp_label_report.json" if jaw is None
                   else f"grasp_label_report_jaw_{jaw}.json")
    started = time.perf_counter()
    from datagen.prompts.build import manifest_for  # noqa: PLC0415 (shares the provenance check)

    root = Path(root)
    extents_longest, problems = manifest_for(root)
    del extents_longest      # that helper returns the longest axis; the solids need all three
    assets = scene_assets(root, density=density)

    # Written as they are produced, not hoarded. Accumulating every row and writing once at the end
    # costs two things. Memory: a shard then holds its entire output, hundreds of thousands of
    # labels, which is not a leak but the design. Observability: the pass then logs once at the end,
    # so a shard three hours in looks exactly like one that died in its first minute.
    #
    # The file is opened in "w", so an interrupted run leaves a truncated grasps.jsonl rather than
    # none at all. That is the trade for streaming, and the report file is what says a run finished:
    # no `grasp_label_report.json` means the labels beside it are partial.
    counts = {"rows": 0, "jaw": 0, "suction": 0}
    rejected: dict[str, int] = {}
    scenes = objects = 0
    per_family: dict[str, dict[str, int]] = {}
    handle = (root / f"{stem}.jsonl").open("w", encoding="utf-8")
    order = _scene_order(sorted((root / "scenes").iterdir()), balanced=label_budget is not None)
    for scene_dir in order:
        scene_json = scene_dir / "scene.json"
        if not scene_json.exists():
            continue
        if limit is not None and scenes >= limit:
            break
        # Checked between scenes, never inside one. A scene's labels are written together or not at
        # all: half an object's grasps is a corpus row that looks complete and is not, and every
        # count downstream is per object.
        if label_budget is not None and counts["rows"] >= label_budget:
            break
        payload = json.loads(scene_json.read_text(encoding="utf-8"))
        scenes += 1
        if scenes % 100 == 0:
            logger.info("labelling %s: %d scene(s), %d label(s) so far (%d jaw, %d suction) in "
                        "%.0f s", root.name, scenes, counts["rows"], counts["jaw"],
                        counts["suction"], time.perf_counter() - started)
        labels, refused = assets.label(payload, scene_dir.name, model=model)
        family = str(payload["spec"].get("family", ""))
        bucket = per_family.setdefault(family, {"scenes": 0, "objects": 0, "jaw": 0, "suction": 0,
                                                "objects_with_jaw": 0, "objects_with_suction": 0})
        bucket["scenes"] += 1
        geometry = assets.geometry(payload, scene_dir.name)
        objects += len(geometry.objects)
        bucket["objects"] += len(geometry.objects)
        with_jaw = {label.instance_id for label in labels if label.kind == "jaw"}
        with_suction = {label.instance_id for label in labels if label.kind == "suction"}
        bucket["objects_with_jaw"] += len(with_jaw)
        bucket["objects_with_suction"] += len(with_suction)
        for label in labels:
            bucket[label.kind] += 1
            handle.write(json.dumps(label.as_row(), sort_keys=True) + "\n")
            counts["rows"] += 1
            counts[label.kind] = counts.get(label.kind, 0) + 1
        for reason, count in refused.items():
            rejected[reason] = rejected.get(reason, 0) + count

    handle.close()
    report = {
        "scenes": scenes, "objects": objects, "labels": counts["rows"],
        "jaw": counts["jaw"],
        "suction": counts["suction"],
        "by_family": per_family, "rejected": dict(sorted(rejected.items())),
        "problems": problems,
        # The same reasoning as `density` below. A budgeted corpus and a complete one are different
        # statements about a dataset and the files look identical, so the report carries the budget,
        # whether it was reached, and how many scenes were left unvisited.
        "label_budget": label_budget,
        "scenes_available": len(order),
        "partial": bool(label_budget is not None and scenes < len(order)),
        # Without this the label count is uninterpretable. Two corpora sampled at different densities
        # differ in `labels` by a factor of five for no reason a reader can see, and "the dense corpus
        # has more grasps" would read as a fact about the objects. It is a fact about the sampler.
        "density": density.as_row(),
        # Same reason as `density` above. A label count from a 140 mm jaw and one from the 2F-85 are
        # not comparable, and without this the difference reads as a fact about the objects.
        "jaw_model": jaw or "2f85",
    }
    (root / report_name).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    # Once for the pass, never per object: the per-object work is the tight loop this package is
    # made of. The rejection histogram is in the report file; what is here is the shape of the
    # result plus the two things a stale label set is diagnosed from, how long it took and where it
    # landed.
    logger.info("labelled %d scene(s), %d object(s) -> %d label(s) (%d jaw, %d suction) in %.1f s; "
                "%s (%d bytes)",
                scenes, objects, counts["rows"], report["jaw"], report["suction"],
                time.perf_counter() - started, root / f"{stem}.jsonl",
                (root / f"{stem}.jsonl").stat().st_size)
    if problems:
        # Returned, not raised: the caller prints them and exits non-zero. Without this line the
        # strongest signal a dataset can give, "the assets moved under these labels", lives only in
        # stdout.
        logger.error("label pass has %d unresolved problem(s): %s",
                     len(problems), "; ".join(problems[:5]))
    return report


@dataclass(frozen=True, slots=True)
class SceneAssets:
    """Everything `scene_geometry` needs to know about a dataset's assets, as one object.

    One object rather than three arguments, and that is a fix rather than a preference.
    `scene_geometry` grew a ``parts`` channel so composite objects could be labelled per part, and
    the manifest grew the structure to fill it, while no production caller passed it: every
    composite in a generated dataset was labelled as the cuboid `primitive_for_kind` returns for the
    tail of its name.

    A bundle can be forgotten wholesale, loudly, because nothing works, but it cannot be
    half-passed, and a fourth channel added later reaches every call site without touching one of
    them.
    """

    #: Asset id to its full (x, y, z) extent in mm.
    extents: dict[str, tuple[float, ...]]
    #: Asset id to the primitives a composite is built from. Absent for single-primitive assets.
    parts: dict[str, tuple[Any, ...]]
    #: Asset id to the mesh file on disk, for scanned assets. Absent for everything authored.
    mesh_paths: dict[str, str]
    #: How PhysX may approximate a scanned mesh, straight from the config the dataset was built
    #: with. An asset fact, and it belongs here for a blunt reason: the shake has to collide the
    #: same approximation the render run used, or it is not testing the label it was given.
    mesh_collision: str = "sdf"
    #: How finely this dataset is sampled. On the bundle for the reason the class docstring gives:
    #: a separate argument is one a caller can leave out silently, and a bundle is not.
    density: LabelDensity = DEFAULT_DENSITY

    def geometry(self, payload: dict, scene_id: str) -> SceneGeometry:
        """The solids for one scene, with every channel this dataset has."""
        return scene_geometry(
            payload, self.extents, scene_id, parts=self.parts, mesh_paths=self.mesh_paths)

    def label(
        self, payload: dict, scene_id: str, *, model: JawModel | None = None,
    ) -> tuple[list[GraspLabel], dict[str, int]]:
        """Every label for one scene. The counterpart of :meth:`geometry`, same guarantee."""
        return label_scene(
            payload, self.extents, scene_id, model=model,
            parts=self.parts, mesh_paths=self.mesh_paths, density=self.density)


#: What the asset cache is written to, inside the dataset it describes.
_ASSET_CACHE = "scene_assets.cache.json"


def _library_stamp() -> str:
    """The mesh library's newest modification time, so a rescale invalidates the cache.

    `normalise_meshes` rewrites meshes in place, and a collection with no unit convention gets a
    size assigned on that pass. A cache keyed only on the config would then hand a run the extents
    of meshes that no longer exist at that size.
    """
    from datagen.assets.library import MESH_LIBRARY_DIR  # noqa: PLC0415

    root = Path(MESH_LIBRARY_DIR)
    if not root.is_dir():
        return "0"
    newest = max((p.stat().st_mtime for p in root.rglob("*") if p.is_file()), default=0.0)
    return f"{newest:.0f}"


def _cache_key(stamp: dict) -> str:
    """The dataset's own config hash plus the mesh library's state."""
    return f"{stamp.get('config_sha256', '')}:{_library_stamp()}"


def scene_assets(root: Path, *, density: LabelDensity = DEFAULT_DENSITY,
                 cache: bool = True) -> SceneAssets:
    """Rebuild a dataset's asset facts from its own provenance stamp.

    Not read from a side file: the stamp records the config, and re-running `build_manifest` on it
    is deterministic, so the assets a labelling run sees are exactly the ones the render run placed.

    Cached on disk, because that determinism is what makes caching safe. Closing every mesh in a
    dataset to learn facts a previous run already learned dominates the wall clock of a physics run
    that does far less work than that.

    The key is the config hash the dataset stamped on itself, not its name or its path. A cache
    keyed on the name would survive a config change and hand a run the assets of a different
    dataset, which is the shape of every stale-cache defect. Keyed on the hash, a changed config
    simply misses. The mesh library's newest modification time is in the key too, because
    `normalise_meshes` rewrites meshes in place.
    """
    from datagen.build import build_manifest  # noqa: PLC0415
    from datagen.config import DatagenConfig  # noqa: PLC0415

    root = Path(root)
    stamp = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    config = DatagenConfig.from_stamp(stamp["config"])
    key = _cache_key(stamp)
    path = root / _ASSET_CACHE
    if cache and path.is_file():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blob = None
        if blob is not None and blob.get("key") == key:
            from datagen.assets.composite import CompositePart  # noqa: PLC0415

            return SceneAssets(
                extents={k: tuple(v) for k, v in blob["extents"].items()},
                parts={k: tuple(CompositePart(role=row["role"], primitive=row["primitive"],
                                              extent_mm=tuple(row["extent_mm"]),
                                              offset_mm=tuple(row["offset_mm"]),
                                              yaw_deg=row["yaw_deg"])
                                for row in rows)
                       for k, rows in blob["parts"].items()},
                mesh_paths=dict(blob["mesh_paths"]),
                mesh_collision=blob["mesh_collision"],
                density=density,
            )

    records = list(build_manifest(config))
    built = SceneAssets(
        extents={record.asset_id: tuple(record.extent_mm) for record in records},
        parts={record.asset_id: tuple(record.parts) for record in records if record.parts},
        mesh_paths={record.asset_id: record.mesh_path
                    for record in records if record.mesh_path},
        mesh_collision=config.assets.mesh_collision,
        density=density,
    )
    if cache:
        # The parts serialise: `CompositePart` is a frozen dataclass of a role, a primitive name,
        # two millimetre triples and a yaw, every field a plain value. Skipping the cache whenever a
        # dataset has parts would skip it for exactly the datasets it exists for.
        #
        # A half-written bundle would be worse than none: `scene_geometry` falls back to deriving a
        # shape from the tail of an asset id, so a composite with missing parts is labelled as a
        # cuboid. That is why the round trip is checked rather than assumed, and why a cache that
        # fails to parse is discarded rather than patched up.
        try:
            path.write_text(json.dumps({
                "key": key,
                "extents": {k: list(v) for k, v in built.extents.items()},
                "parts": {k: [{"role": part.role, "primitive": part.primitive,
                               "extent_mm": list(part.extent_mm),
                               "offset_mm": list(part.offset_mm),
                               "yaw_deg": part.yaw_deg} for part in rows]
                          for k, rows in built.parts.items()},
                "mesh_paths": built.mesh_paths,
                "mesh_collision": built.mesh_collision}, sort_keys=True), encoding="utf-8")
        except (OSError, TypeError, AttributeError) as exc:
            # A part shape this writer does not know is a reason to have no cache, never a reason
            # to write half of one.
            logger.info("asset cache not written: %s", exc)
            path.unlink(missing_ok=True)
    return built

