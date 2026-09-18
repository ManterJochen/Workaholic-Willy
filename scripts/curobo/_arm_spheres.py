"""Which arm spheres a cuRobo UR descriptor keeps, judged against the committed mesh of their own link.

``build_ur_config.py`` reads the committed cover fit, so this judgement runs only for an arm with no fit, on the Lula
fallback: the build starts from Isaac's Lula spheres and adds spheres fitted to the arm's committed collision bundle.
Isaac's vendor spheres reach well past the parts they stand for: wrist_1 spheres up to 55.8 mm past the link's
hull on ur10, a forearm sphere 79.6 mm on ur10e, while every surface fit sphere reaches 10 mm. A descriptor that keeps
them starts in self collision at wrist_1 where the exact meshes are clear, and then cannot plan at all: five of the 21
descriptors do exactly that.

So a sphere is judged by how far it reaches past the convex hull of its own link: the largest distance of its centre
from the hull's face planes, plus its radius. Off a face that is the exact reach; near an edge or a corner it is a
lower bound, so a sphere this keeps may reach a little further there, and one it drops certainly reaches past the
bound.

Loaded by path from the build, which runs in the cuRobo environment; numpy and scipy only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["ArmFitRecipe", "LinkHull", "arm_spheres_label", "cover_spheres_label", "filter_spheres",
           "link_frames", "link_spheres", "place_spheres"]

_RECIPE_GRAMMAR = "surface, or morphit:<density>:<protrusion weight> with both above 0"


@dataclass(frozen=True)
class ArmFitRecipe:
    """How a descriptor's arm spheres are fitted to the committed bundle, stated by the caller and recorded.

    ``surface`` places fixed counts of 10 mm spheres on the link's surface, beside Isaac's vendor spheres.
    ``morphit:<density>:<protrusion weight>`` is cuRobo's optimised fit, its sphere count scaled by the density and
    its reach past the mesh penalised by the weight (cuRobo's default weight is 10).
    """

    fit_type: str
    density: float | None = None
    protrusion_weight: float | None = None

    @classmethod
    def parse(cls, text: str) -> ArmFitRecipe:
        """The recipe ``text`` names. Anything else is refused with the grammar, never read as a default."""
        parts = str(text).strip().split(":")
        if parts == ["surface"]:
            return cls("surface")
        if len(parts) == 3 and parts[0] == "morphit":
            try:
                density, weight = float(parts[1]), float(parts[2])
            except ValueError:
                density = weight = 0.0
            if density > 0.0 and weight > 0.0:
                return cls("morphit", density, weight)
        raise ValueError(f"{text!r} is not an arm fit recipe; the recipes are {_RECIPE_GRAMMAR}")

    def render(self) -> str:
        """The recipe as it is written on the command line and in the descriptor's provenance."""
        if self.fit_type == "surface":
            return "surface"
        return f"morphit:{self.density:g}:{self.protrusion_weight:g}"


@dataclass(frozen=True)
class LinkHull:
    """The convex hull of one link's committed mesh: outward unit normals and offsets, ``normal . x <= offset`` inside."""

    normals: np.ndarray
    offsets: np.ndarray

    @classmethod
    def from_vertices(cls, vertices_m: Any) -> LinkHull:
        """The hull of ``vertices_m``, an (N, 3) array in metres, in the frame the spheres are given in."""
        from scipy.spatial import ConvexHull

        hull = ConvexHull(np.asarray(vertices_m, dtype=np.float64))
        # scipy writes each facet as `normal . x + c <= 0` inside, with the normal a unit vector pointing out.
        return cls(normals=hull.equations[:, :3].copy(), offsets=-hull.equations[:, 3].copy())

    def reach_m(self, centre_m: Sequence[float], radius_m: float) -> float:
        """How far a sphere reaches past the hull, metres; negative for a sphere inside it."""
        centre = np.asarray(centre_m, dtype=np.float64)
        return float(np.max(self.normals @ centre - self.offsets)) + float(radius_m)


def filter_spheres(
    spheres: Sequence[dict[str, Any]], hull: LinkHull, *, bound_m: float,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], float]]]:
    """``(kept, dropped)``: the spheres reaching past ``hull`` by at most ``bound_m``, and every other with its reach."""
    kept: list[dict[str, Any]] = []
    dropped: list[tuple[dict[str, Any], float]] = []
    for sphere in spheres:
        reach = hull.reach_m(sphere["center"], sphere["radius"])
        if reach > bound_m:
            dropped.append((sphere, reach))
        else:
            kept.append(sphere)
    return kept, dropped


def arm_spheres_label(*, bundle: bool, recipe: str, bound_mm: float | None) -> str:
    """What ``_provenance.arm_spheres`` says: the recipe the build ran, and the rule Isaac's vendor spheres were kept by.

    A fitted sphere is not held to the bound, so the label states the rule for the vendor spheres and nothing more.
    How far every kept sphere reaches is measured by ``probe_sphere_fidelity.py``, not claimed here.
    """
    if not bundle:
        return "lula"
    if bound_mm is None:
        return (f"lula + {recipe} fit to the committed bundle; vendor spheres dropped only when their centre sits "
                "outside the link's bounding box by more than their radius")
    return (f"lula + {recipe} fit to the committed bundle; vendor spheres reaching more than {bound_mm:g} mm past "
            "their link's hull dropped")

def cover_spheres_label(*, source: str, reach_mm: float, spheres: int) -> str:
    """What ``_provenance.arm_spheres`` says for a map this repository fitted rather than repaired.

    A different sentence from the one above, because it describes a different kind of map. The vendor's is a
    compromise measured afterwards, so its label states which spheres were kept and by what rule. A cover fit
    has no holes by construction and a reach it was given, so its label states those two numbers and names the
    source it was fitted from.
    """
    return (f"cover fit, {source}: {spheres} spheres, every surface sample of every link inside one, none "
            f"reaching more than {reach_mm:.1f} mm past its own link")

class CoverMapError(ValueError):
    """A committed cover fit that cannot be used as a sphere map, said by name rather than planned around."""


def link_spheres(document: dict, *, source: str) -> "dict[str, list[dict[str, Any]]]":
    """A committed cover fit map as a cuRobo sphere map, keyed by the link names a descriptor uses.

    The fit names bodies as the bundle does (``wrist_1``) because that is what it measured; a descriptor names
    links (``wrist_1_link``). The translation lives here rather than in the file, so the map stays a statement
    about geometry and the descriptor's vocabulary stays the descriptor's business.

    A map recording a hole is refused. A hole is the unsafe error: a surface point of that link lies outside
    every sphere, so the planner does not see the arm there and clears poses where the arm actually is. The
    fitter cannot produce one, so a file that claims one has been edited or truncated, and the right answer is
    to write no descriptor at all.
    """
    spheres = document.get("collision_spheres")
    if not isinstance(spheres, dict) or not spheres:
        raise CoverMapError(f"{source} holds no collision_spheres, so there is no map to read")

    rows = (document.get("_provenance") or {}).get("bodies") or []
    holed = sorted(str(row.get("body")) for row in rows if float(row.get("fresh_uncovered_max_mm", 0.0)) > 0.0)
    if holed:
        raise CoverMapError(
            f"{source} records a hole on {holed}: a surface point of that link lies outside every sphere, so a "
            f"planner reading this map would not see the arm there. Re-fit it; no descriptor is written."
        )
    missing = sorted(name for name in spheres if not (spheres[name] or {}).get("spheres"))
    if missing:
        raise CoverMapError(
            f"{source} carries {missing} with no spheres at all. A link with an empty list is not checked for "
            f"collision, and a present key is exactly what an emptiness check passes."
        )
    return {
        f"{body}_link": [{"center": [float(c) for c in sphere["center"]], "radius": float(sphere["radius"])}
                         for sphere in block["spheres"]]
        for body, block in spheres.items()
    }

def link_frames(urdf_text: str, dh_rows: Sequence[tuple[float, float, float]]):
    """``place(link, frame)``: the 4x4 that carries a point from a link's DH frame into its URDF link frame.

    The two frames are not the same, and the map between them carries a half turn. A committed collision bundle
    holds each link's mesh in that link's own DH frame; a cuRobo descriptor holds each link's spheres in the
    URDF link frame. UR's description mounts ``base_link_inertia`` yawed by pi, so the map between them is
    ``inv(urdf_link) @ Rz(pi) @ T_dh[frame]`` and nothing else.

    On the ur5e the difference is a displacement along the link rather than a turn: the upper arm's DH frame
    sits 425 mm from its URDF link frame, exactly the length of the link, so its spheres run x from -49 to
    +423 mm in one frame and -474 to -2 mm in the other. The shoulder's two frames differ by a quarter turn
    about x instead, which is the alpha of its own DH row. Every link differs somehow.

    A map handed over untransformed therefore puts the upper arm's spheres 425 mm from the upper arm, and
    nothing downstream measures where they landed: the descriptor loads, the planner plans, the ready gate
    says ready, and the robot the planner holds is not the robot on the bench.
    """
    import xml.etree.ElementTree as ET

    def axis_angle(axis: tuple[float, float, float], theta: float) -> np.ndarray:
        a = np.asarray(axis, dtype=np.float64)
        a = a / (np.linalg.norm(a) or 1.0)
        c, s = np.cos(theta), np.sin(theta)
        x, y, z = a
        return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                         [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                         [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])

    root = ET.fromstring(urdf_text)
    kids: dict[str, tuple[str, np.ndarray]] = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()] if origin is not None else [0.0] * 3
        xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()] if origin is not None else [0.0] * 3
        M = np.eye(4)
        M[:3, :3] = axis_angle((0, 0, 1), rpy[2]) @ axis_angle((0, 1, 0), rpy[1]) @ axis_angle((1, 0, 0), rpy[0])
        M[:3, 3] = xyz
        kids[joint.find("child").attrib["link"]] = (joint.find("parent").attrib["link"], M)
    if not kids:
        raise ValueError("this URDF declares no joints, so no link has a frame")
    base_link = next(iter({parent for parent, _ in kids.values()} - set(kids)))

    def urdf_frame(link: str) -> np.ndarray:
        if link == base_link:
            return np.eye(4)
        parent, M = kids[link]
        return urdf_frame(parent) @ M

    dh: list[np.ndarray] = [np.eye(4)]
    for a, d, alpha in dh_rows:
        ca, sa = np.cos(alpha), np.sin(alpha)
        step = np.array([[1.0, 0.0, 0.0, a], [0.0, ca, -sa, 0.0], [0.0, sa, ca, d], [0.0, 0.0, 0.0, 1.0]])
        dh.append(dh[-1] @ step)

    rz_pi = np.eye(4)
    rz_pi[:3, :3] = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])

    def place(link: str, frame: int) -> np.ndarray:
        return np.linalg.inv(urdf_frame(link)) @ rz_pi @ dh[frame]

    return place


def place_spheres(spheres: Sequence[Mapping[str, Any]], M: np.ndarray) -> "list[dict[str, Any]]":
    """The same spheres with their centres carried through ``M``. A rigid motion, so the radii do not move."""
    if not len(spheres):
        return []
    centres = np.asarray([sphere["center"] for sphere in spheres], dtype=np.float64)
    moved = (centres @ np.asarray(M)[:3, :3].T) + np.asarray(M)[:3, 3]
    return [{"center": [float(v) for v in centre], "radius": float(sphere["radius"])}
            for centre, sphere in zip(moved, spheres)]
