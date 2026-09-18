"""A body somebody measured rather than scanned, as collision geometry. Stdlib plus numpy, on purpose.

Three things in this stack are described by numbers instead of by a mesh, and each has to reach the planner and the
exact-mesh guard as geometry:

* a gripper a customer describes in config and has never baked: jaws and palm out of ``gripper.jaw``;
* the plate and the tool changer between the flange and the hand;
* the wrist camera's housing and its bracket.

One writer serves all three, because a box becoming geometry in three places is three chances to disagree about what
a half extent means. What the three share is exactly what is here: a named box, its centre, its half extents, and the
frame it sits in.

A box is not the body, and the difference is measurable. The EGU-50's own config records what an envelope leaves out:
along the closing axis its housing reaches 2.25 mm past the finger's outer face, and a box built from the numbers does
not carry that corner. That is a false clear, the unsafe direction. So a body written from dimensions carries an
inflation, and the inflation is a measurement rather than a choice: the three shipped hands each have both a baked
bundle and a full set of declared dimensions, so the difference between the two can be read off.

Imported by the client side only: the boxes become spheres here, and the sidecar receives the spheres as a body link.
:func:`cover_refusal` is the other half of that trade for a wrist camera. The combination evidence is keyed without
the camera's body, so what stands in for a measurement is a proof, checked when the planner starts, that the spheres
the sidecar loaded hold every point of every box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Box", "box_mesh", "box_spheres", "boxes_to_parts", "cover_refusal", "inflated", "stacked_boxes"]

#: How far a sphere circumscribing a cube of side ``s`` reaches past that cube's face: ``s * (sqrt(3) - 1) / 2``.
#: Inverted below to pick the cell size from the reach the caller allows, which is the whole fit.
_CUBE_REACH_PER_SIDE = (3.0 ** 0.5 - 1.0) / 2.0

#: The eight corners of a unit box, as signs on each axis, and the twelve triangles over them wound outward.
_CORNERS = np.array([(sx, sy, sz) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
#: Index into ``_CORNERS``: bit 2 is x, bit 1 is y, bit 0 is z, so corner (+,-,+) is 0b101.
_FACES = np.array([
    # -x and +x
    (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
    # -y and +y
    (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
    # -z and +z
    (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
], dtype=np.int64)


@dataclass(frozen=True)
class Box:
    """One measured box: what it belongs to, where its centre sits, and half of each side.

    Millimetres, in the frame the caller names when the boxes become parts. Half extents rather than side lengths,
    because that is how a cell writes a fixture down and how the schema already asks for one.
    """

    name: str
    centre_mm: tuple[float, float, float]
    half_extents_mm: tuple[float, float, float]

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        sides = " x ".join(f"{2.0 * v:g}" for v in self.half_extents_mm)
        centre = ", ".join(f"{v:g}" for v in self.centre_mm)
        return f"{self.name}: {sides} mm centred at ({centre}) mm"

    def to_dict(self) -> dict[str, Any]:
        """The wire view."""
        return {"name": self.name, "centre_mm": list(self.centre_mm),
                "half_extents_mm": list(self.half_extents_mm)}


def inflated(box: Box, *, by_mm: float) -> Box:
    """``box`` with every half extent grown by ``by_mm``, which is how a measured under-model is carried.

    ``0.0`` is a real answer and means a cell measured its body and found nothing to add; a caller that writes a
    bundle refuses an inflation nobody stated rather than assuming one. A negative value shrinks a declared body,
    which is the unsafe direction and never what anybody meant.
    """
    grown = float(by_mm)
    if grown < 0.0:
        raise ValueError(
            f"an inflation of {grown:g} mm would shrink {box.name!r} below what was measured, and a body smaller "
            "than the part it stands for is a pose the guard clears and should refuse"
        )
    return Box(name=box.name, centre_mm=box.centre_mm,
               half_extents_mm=tuple(float(v) + grown for v in box.half_extents_mm))  # type: ignore[arg-type]


def box_mesh(box: Box) -> "tuple[np.ndarray, np.ndarray]":
    """``(vertices_mm, faces)`` for one box: eight corners and twelve triangles, wound outward.

    Outward matters. The cover fit's inside test is a generalised winding number, which reads an inverted surface
    with the wrong sign, and the exact-mesh guard builds its body from the faces it is handed.
    """
    half = np.asarray(box.half_extents_mm, dtype=np.float64)
    if half.shape != (3,) or not np.all(np.isfinite(half)):
        raise ValueError(f"{box.name!r} needs three finite half extents, got {box.half_extents_mm!r}")
    if np.any(half <= 0.0):
        raise ValueError(
            f"{box.name!r} has a half extent of {half.min():g} mm, so it encloses nothing. A body with no inside "
            "cannot be covered by spheres and cannot be checked for collision; measure it, or leave it out."
        )
    centre = np.asarray(box.centre_mm, dtype=np.float64)
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise ValueError(f"{box.name!r} needs three finite centre coordinates, got {box.centre_mm!r}")
    return _CORNERS * half + centre, _FACES.copy()


def stacked_boxes(
    slabs: "list[tuple[str, float, tuple[float, float] | None]] | tuple",
    *,
    axis: int,
    start_mm: float = 0.0,
) -> "tuple[list[Box], tuple[str, ...]]":
    """A stack of named slabs along one axis, as boxes, plus the names of the ones that declared no cross section.

    Each slab is ``(name, thickness_mm, cross_section_mm)``, where the cross section is the two half extents across
    ``axis``, in the order the remaining axes have. The stack grows from ``start_mm`` in the positive direction and
    every slab holds the next one out, whether or not it became a body.

    A thickness is not a body. A slab with no cross section is returned in the second half rather than turned into a
    box, because a body built from one measured number and one guessed one is indistinguishable downstream from a body
    somebody measured. The caller says those names out loud; nothing here refuses.

    What the missing body costs, measured by ``scripts/curobo/probe_plate_body.py``: the Hand-E's 20 mm plate over the
    1,483 judged poses, against the exact guard's 10 mm, turns 0 to 2 of the roughly 950 clear poses per arm at a UR
    flange radius, the worst grazing at 8.944 mm. At 45 mm, a quick change coupler's radius, it is 0 to 25. So the
    plate is completeness and the tool changer is the safety case, and both come through here.
    """
    boxes: list[Box] = []
    undeclared: list[str] = []
    if axis not in (0, 1, 2):
        raise ValueError(f"a stack grows along x, y or z, which is 0, 1 or 2, and not {axis!r}")

    at = float(start_mm)
    for slab in slabs:
        name, thickness_mm, cross_section = slab
        thickness = float(thickness_mm)
        if thickness <= 0.0:
            raise ValueError(
                f"{name!r} declares a thickness of {thickness:g} mm, and a slab of no thickness places nothing and "
                f"collides with nothing. Remove it, or write what it measures."
            )
        if cross_section is None:
            undeclared.append(str(name))
        else:
            across = [float(v) for v in cross_section]
            if len(across) != 2:
                raise ValueError(
                    f"{name!r} needs the two half extents across the stacking axis, and got {cross_section!r}. The "
                    f"third is the thickness, which is already declared, so it is never written twice."
                )
            half = [0.0, 0.0, 0.0]
            half[axis] = thickness / 2.0
            centre = [0.0, 0.0, 0.0]
            centre[axis] = at + thickness / 2.0
            for other, value in zip([i for i in (0, 1, 2) if i != axis], across):
                half[other] = value
            boxes.append(Box(name=str(name), centre_mm=(centre[0], centre[1], centre[2]),
                             half_extents_mm=(half[0], half[1], half[2])))
        at += thickness
    return boxes, tuple(undeclared)


def box_spheres(box: Box, *, reach_mm: float) -> "list[dict[str, Any]]":
    """A complete sphere cover of ``box``, in metres, in the shape a cuRobo body link reads.

    The same promise the cover fit makes for the arms, without the search. Its safe half is that every surface point
    of the body ends up inside a sphere, because a hole is a false clear and has no upper bound on what it costs. On a
    general mesh that takes marching; on a box it is arithmetic. Cut the box into cells no larger than a cube of side
    ``s`` and circumscribe each: every point of the box is in its own cell and so inside that cell's sphere, and the
    furthest such a sphere reaches past a face is ``s * (sqrt(3) - 1) / 2``.

    So the caller states the reach, the cell size follows, and the count follows from the cell size. That is the same
    trade the arms make, and it stays the caller's: a tighter cover refuses fewer poses and costs plan time.
    """
    reach = float(reach_mm)
    if reach <= 0.0:
        raise ValueError(
            f"a sphere cover needs a reach it is allowed to have; got {reach:g} mm. Zero reach would need "
            f"infinitely many spheres, because a sphere is not a corner."
        )
    half = np.asarray(box.half_extents_mm, dtype=np.float64)
    if half.shape != (3,) or not np.all(half > 0.0):
        raise ValueError(f"{box.name!r} has no volume to cover: half extents {box.half_extents_mm!r}")

    side = reach / _CUBE_REACH_PER_SIDE
    counts = np.maximum(np.ceil(2.0 * half / side).astype(int), 1)
    cell = 2.0 * half / counts
    radius = float(np.linalg.norm(cell) / 2.0)

    centre = np.asarray(box.centre_mm, dtype=np.float64)
    axes = [centre[i] - half[i] + cell[i] * (np.arange(counts[i]) + 0.5) for i in (0, 1, 2)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return [{"center": [float(v) / 1000.0 for v in point], "radius": radius / 1000.0} for point in grid]


def boxes_to_parts(boxes: "list[Box] | tuple[Box, ...]", *, frame: int) -> "dict[str, np.ndarray]":
    """The boxes as the arrays a collision bundle carries: ``<name>__v``, ``<name>__f`` and ``<name>__frame``.

    Boxes sharing a name become one part, because a camera is its housing and its bracket and both are the camera.
    The faces of the second box index its own vertices, so the parts stay separate solids inside one array rather
    than one tangled surface.
    """
    listed = list(boxes)
    if not listed:
        raise ValueError("a body needs at least one box; an empty declaration is not a body of zero size")

    grouped: dict[str, list[Box]] = {}
    for box in listed:
        grouped.setdefault(box.name, []).append(box)

    parts: dict[str, np.ndarray] = {}
    for name, members in grouped.items():
        vertices: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        offset = 0
        for box in members:
            corner, face = box_mesh(box)
            vertices.append(corner)
            faces.append(face + offset)
            offset += len(corner)
        parts[f"{name}__v"] = np.concatenate(vertices)
        parts[f"{name}__f"] = np.concatenate(faces)
        parts[f"{name}__frame"] = np.asarray([int(frame)], dtype=np.int32)
    return parts


def cover_refusal(box: Box, spheres: "list[dict[str, Any]] | tuple", *, tolerance_mm: float = 1e-6) -> "str | None":
    """Why ``spheres`` (metres, as :func:`box_spheres` writes them) do not hold every point of ``box``, or ``None``.

    A proof, not a sample. The spheres must be a grid over the box: on each axis their distinct centre coordinates cut
    the box into cells at the midpoints between them, each coordinate lies inside its own cell, there is exactly one
    sphere per grid point and no other, and every corner of every cell lies within that cell's sphere. The cells tile
    the box and a ball is convex, so a ball holding a cell's eight corners holds the cell, and the union holds the box,
    corners and interior. Nothing is sampled, so a hole smaller than any sample spacing is found too.

    The condition is stricter than coverage: a fill of another shape that also covers the box is refused, which costs
    nothing because :func:`box_spheres` is the one writer and writes exactly this grid, every cell's corners on its
    sphere. ``tolerance_mm`` absorbs the round trip through metres, far below the smallest real displacement measured
    in this stack, 0.5 mm. Numpy only, and :func:`box_spheres` is never called, so the writer cannot vouch for itself.
    """
    tolerance = float(tolerance_mm)
    listed = list(spheres)
    if not listed:
        return f"{box.name!r} has no spheres, so nothing of it is covered"
    try:
        centres = np.asarray([[float(v) for v in sphere["center"]] for sphere in listed], dtype=np.float64) * 1000.0
        radii = np.asarray([float(sphere["radius"]) for sphere in listed], dtype=np.float64) * 1000.0
    except (KeyError, TypeError, ValueError):
        return f"{box.name!r} is covered by something that is not a list of spheres with a center and a radius"
    if centres.ndim != 2 or centres.shape[1] != 3:
        return f"{box.name!r} is covered by spheres whose centres are not three coordinates"
    if not (np.all(np.isfinite(centres)) and np.all(np.isfinite(radii)) and np.all(radii > 0.0)):
        return f"{box.name!r} is covered by a sphere with a radius that is not above 0 or a number that is not finite"
    half = np.asarray(box.half_extents_mm, dtype=np.float64)
    middle = np.asarray(box.centre_mm, dtype=np.float64)
    low, high = middle - half, middle + half

    bounds: list[np.ndarray] = []
    index = np.zeros((len(listed), 3), dtype=np.int64)
    for axis in (0, 1, 2):
        values = np.sort(centres[:, axis])
        distinct = [float(values[0])]
        for value in values[1:]:
            if value - distinct[-1] > tolerance:
                distinct.append(float(value))
        coords = np.asarray(distinct)
        edges = np.concatenate(([low[axis]], (coords[:-1] + coords[1:]) / 2.0, [high[axis]]))
        outside = (coords < edges[:-1] - tolerance) | (coords > edges[1:] + tolerance)
        if np.any(outside):
            k = int(np.argmax(outside))
            return (f"{box.name!r}: the sphere centres along axis {'xyz'[axis]} at {coords[k]:.6f} mm lie outside the "
                    f"cell [{edges[k]:.6f}, {edges[k + 1]:.6f}] mm they would have to stand for, so the spheres are not "
                    "a grid over this box")
        bounds.append(edges)
        index[:, axis] = np.abs(centres[:, axis][:, None] - coords[None, :]).argmin(axis=1)
    counts = tuple(len(edges) - 1 for edges in bounds)
    flat = np.ravel_multi_index(index.T, counts)
    if len(np.unique(flat)) != len(listed) or len(listed) != counts[0] * counts[1] * counts[2]:
        return (f"{box.name!r} has {len(listed)} spheres over a {counts[0]} x {counts[1]} x {counts[2]} grid, and a "
                "cover proof needs exactly one sphere per grid point")
    reach = np.stack([
        np.maximum(np.abs(centres[:, axis] - bounds[axis][index[:, axis]]),
                   np.abs(bounds[axis][index[:, axis] + 1] - centres[:, axis]))
        for axis in (0, 1, 2)
    ], axis=1)
    farthest = np.linalg.norm(reach, axis=1)
    short = farthest - radii
    worst = int(np.argmax(short))
    if short[worst] > tolerance:
        i, j, k = (int(v) for v in index[worst])
        return (f"{box.name!r} has cell ({i}, {j}, {k}) with a corner {short[worst]:.6f} mm outside its sphere, so the "
                "spheres leave a hole in the box, which is a false clear")
    return None
