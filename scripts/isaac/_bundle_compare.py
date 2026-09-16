"""Whether two collision bundles are the same arm, order free and mirror aware. Stdlib plus numpy, no Isaac.

Two readers produce a bundle for one robot: Isaac's composed articulation, and Universal Robots' own collision STL
files through the rendered description. Whether they agree is what says a frame chain is right, and a bundle from a
wrong frame does not fail loudly: it guards empty space and passes every pose.

The comparison is honest about three things.

Vertex order is not geometry. Two readers list a mesh's vertices in their own order, so the arrays are compared by
where the surfaces are, not by index: the distance from each vertex of one to the nearest vertex of the other, in
both directions, plus the centroid and the axis aligned extent.

A mirror keeps every number a lazy check compares. Reflect a link across one of its own DH planes and the extent,
the centroid and the volume are all unchanged, while the arm is now left handed. The nearest point distances above
are what notices, which is the reason they are here rather than an extent diff.

A difference that is known is named, not absorbed. Isaac's URDFs carry a handful of mesh origins that differ from
UR's own description by a measured amount. Those live in :data:`ISAAC_MESH_ORIGIN_DIFFERENCES`, applied by name, so
a known difference stays visible instead of hiding inside a loosened ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = [
    "CEILINGS",
    "ISAAC_MESH_ORIGIN_DIFFERENCES",
    "LINKS",
    "BundleDifference",
    "Ceilings",
    "LinkDifference",
    "compare",
]

#: The six arm links every bundle carries, in the order a report reads best: base to flange.
LINKS = ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")


@dataclass(frozen=True)
class Ceilings:
    """How far two readers of one arm may differ before they are describing different geometry.

    Not a tolerance on measurement noise: both readers are exact. It is room for a mesh written at a different
    precision and for a decimation that moves a surface by a fraction of a millimetre. ``centroid_mm`` is the tight
    one, because a frame error moves the whole link and shows there first.
    """

    centroid_mm: float = 0.5
    p99_mm: float = 1.0
    max_mm: float = 3.0


CEILINGS = Ceilings()

#: (model, link) to the measured offset in millimetres, in the link's own DH frame, by which Isaac's mesh origin
#: sits away from the one UR's description declares. Each entry is a measurement, cited where it was taken, never a
#: fudge factor: a difference without a source stays a failure rather than becoming an entry here.
ISAAC_MESH_ORIGIN_DIFFERENCES: dict[tuple[str, str], tuple[float, float, float]] = {}


@dataclass(frozen=True)
class LinkDifference:
    """How far apart two readings of one link are, in millimetres."""

    link: str
    centroid_mm: float
    extent_mm: float
    p99_mm: float
    max_mm: float
    vertices: tuple[int, int]

    def agrees(self, ceilings: Ceilings) -> bool:
        return (self.centroid_mm <= ceilings.centroid_mm
                and self.p99_mm <= ceilings.p99_mm
                and self.max_mm <= ceilings.max_mm)

    def render(self) -> str:
        return (f"{self.link:10s} centroid {self.centroid_mm:7.3f} mm  extent {self.extent_mm:7.3f} mm  "
                f"p99 {self.p99_mm:7.3f} mm  max {self.max_mm:7.3f} mm  "
                f"v {self.vertices[0]} vs {self.vertices[1]}")

    def to_dict(self) -> dict[str, Any]:
        return {"link": self.link, "centroid_mm": self.centroid_mm, "extent_mm": self.extent_mm,
                "p99_mm": self.p99_mm, "max_mm": self.max_mm, "vertices": list(self.vertices)}


@dataclass(frozen=True)
class BundleDifference:
    """Every link of two bundles, side by side."""

    links: tuple[LinkDifference, ...]

    @property
    def worst(self) -> LinkDifference:
        """The link that differs most, by the distance a mirror shows up in."""
        return max(self.links, key=lambda row: (row.max_mm, row.centroid_mm))

    def agrees(self, ceilings: Ceilings = CEILINGS) -> bool:
        return all(row.agrees(ceilings) for row in self.links)

    def render(self) -> str:
        lines = [row.render() for row in self.links]
        lines.append(f"worst: {self.worst.link} at {self.worst.max_mm:.3f} mm")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"links": [row.to_dict() for row in self.links], "worst": self.worst.to_dict()}


def _nearest_mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For every point of ``a``, the distance to the nearest point of ``b``, in whatever unit they are in.

    Through a k-d tree where SciPy is importable, which it is in this repository's environment, where it is a
    pinned base dependency. The fallback is the same question asked the slow way, chunked so that two 5,000 vertex
    links never build the 200 MB pair matrix at once: a UR link pair is about 28 million distances, so the
    difference between the two paths is seconds against minutes, and this is called once per link per comparison.
    """
    try:
        from scipy.spatial import cKDTree  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 (outside this venv the arithmetic is the same, only slower)
        out = np.empty(len(a), dtype=np.float64)
        chunk = max(1, 2_000_000 // max(1, len(b)))
        for start in range(0, len(a), chunk):
            block = a[start:start + chunk]
            distances = np.linalg.norm(block[:, None, :] - b[None, :, :], axis=2)
            out[start:start + chunk] = distances.min(axis=1)
        return out
    return np.asarray(cKDTree(b).query(a, k=1)[0], dtype=np.float64)


def link_difference(a: np.ndarray, b: np.ndarray, *, link: str) -> LinkDifference:
    """One link, compared without ever lining up two vertex lists."""
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    nearest = np.concatenate([_nearest_mm(first, second), _nearest_mm(second, first)])
    return LinkDifference(
        link=link,
        centroid_mm=float(np.linalg.norm(first.mean(0) - second.mean(0))),
        extent_mm=float(np.max(np.abs((first.max(0) - first.min(0)) - (second.max(0) - second.min(0))))),
        p99_mm=float(np.percentile(nearest, 99.0)),
        max_mm=float(nearest.max()),
        vertices=(len(first), len(second)),
    )


def _vertices(bundle: "Mapping[str, Any]", link: str) -> np.ndarray:
    key = f"{link}__v"
    if key not in bundle:
        raise KeyError(f"this bundle has no {key}, so nothing here can say whether its {link} agrees with anything")
    return np.asarray(bundle[key], dtype=np.float64)


def compare(
    a: "Mapping[str, Any]",
    b: "Mapping[str, Any]",
    *,
    links: "tuple[str, ...]" = LINKS,
    model: "str | None" = None,
) -> BundleDifference:
    """Compare two bundles link by link, with any known mesh origin difference taken out by name.

    ``model`` opts into :data:`ISAAC_MESH_ORIGIN_DIFFERENCES`, which shifts ``b`` by the measured offset for that
    model and link before comparing. Left as ``None`` nothing is taken out, which is what a re-bake of one reader
    against itself wants.
    """
    rows = []
    for link in links:
        first = _vertices(a, link)
        second = _vertices(b, link)
        offset = ISAAC_MESH_ORIGIN_DIFFERENCES.get((str(model), link)) if model else None
        if offset is not None:
            second = second + np.asarray(offset, dtype=np.float64)
        rows.append(link_difference(first, second, link=link))
    return BundleDifference(links=tuple(rows))
