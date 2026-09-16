"""One body of a committed bundle as the two questions a sphere fit asks of it, and the ruler that judges any map.

The fitter in ``_cover_fit.py`` takes a body as an oracle: an unsigned distance and an inside test. This is that oracle
for a real mesh, plus ``measure``, which asks of any set of spheres the two questions that decide whether a collision
model is honest:

* how far outside the union any surface point lies (a hole, and a false clear: the planner believes part of the robot
  is not there), and
* how far any sphere reaches past the body (reach, a false collide: picks and workspace lost, nothing worse).

One implementation, because the vendor's map and the refit have to be compared with the same ruler or the comparison
says nothing. ``fit_cover_spheres.py`` fits with it and ``measure_sphere_map.py`` judges with it.

The inside test is the generalised winding number, not a ray cast. On these bundles 7 of 45 bodies are not watertight,
and there a ray test reads points up to 22 mm outside the body as inside it. The winding number sums solid angles over
every face and does not care how the shells are put together.

A group of meshes is one body for this purpose: the three parts of a hand sit in the same flange frame and a sphere
reaching out of the gripper into the finger has not left the hand. Arm links each sit in their own DH frame, so they
are measured one at a time.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cover_fit import unit_directions  # noqa: E402

__all__ = ["Fidelity", "MeshBody", "load_meshes", "winding"]

#: Surface samples per body when the caller does not say. Coverage is a claim at the samples, so this is the
#: resolution of the claim, and both the fit and the measurement re-ask it on a fresh draw with another seed.
DEFAULT_SAMPLES = 60_000

#: Directions per sphere when measuring reach. More is a tighter lower bound and a slower measurement.
DEFAULT_DIRECTIONS = 256


def winding(mesh: Any, points: np.ndarray, chunk: int = 64) -> np.ndarray:
    """The generalised winding number of ``mesh`` at each point: solid angles summed over every face, over 4 pi.

    About 1 inside a solid, about 0 outside, and right on a body whose shells do not close, which is what a ray test
    is not.
    """
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    out = np.zeros(len(points))
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk, None, :]
        a = triangles[None, :, 0, :] - block
        b = triangles[None, :, 1, :] - block
        c = triangles[None, :, 2, :] - block
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        det = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("ijk,ijk->ij", a, b) * lc
               + np.einsum("ijk,ijk->ij", a, c) * lb + np.einsum("ijk,ijk->ij", b, c) * la)
        out[start:start + chunk] = (2.0 * np.arctan2(det, den)).sum(axis=1) / (4.0 * np.pi)
    return out


@dataclass(frozen=True)
class Fidelity:
    """What a sphere map costs against the body it stands for, measured on a surface sample it did not choose."""

    #: The largest distance from a surface sample to the nearest sphere surface. Negative means every sample is inside
    #: a sphere by that much, which is the only safe sign: a positive number is a hole the planner cannot see.
    uncovered_max_mm: float
    #: The furthest any sphere's surface lies past the body. A lower bound: it is measured on a finite set of
    #: directions per sphere, so the true reach is at least this.
    reach_max_mm: float
    spheres: int
    samples: int

    @property
    def has_hole(self) -> bool:
        return self.uncovered_max_mm > 0.0

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        hole = (f"hole {self.uncovered_max_mm:.2f} mm" if self.has_hole
                else f"no hole (inside by {-self.uncovered_max_mm:.2f} mm)")
        return (f"{self.spheres} spheres over {self.samples} samples: {hole}, reaching {self.reach_max_mm:.2f} mm "
                f"past the body")

    def to_dict(self) -> dict[str, Any]:
        """The wire view."""
        return {"uncovered_max_mm": round(self.uncovered_max_mm, 3), "reach_max_mm": round(self.reach_max_mm, 3),
                "spheres": self.spheres, "samples": self.samples}


def load_meshes(bundle: Path, bodies: "list[str]") -> list:
    """The named bodies of a committed ``*_meshes.npz``, as trimesh meshes in metres."""
    import trimesh

    meshes = []
    with np.load(bundle, allow_pickle=True) as held:
        for body in bodies:
            if f"{body}__v" not in held:
                raise KeyError(f"{bundle.name} has no body {body!r}; it holds "
                               f"{sorted(name[:-3] for name in held.files if name.endswith('__v'))}")
            meshes.append(trimesh.Trimesh(
                vertices=np.asarray(held[f"{body}__v"], dtype=np.float64) / 1000.0,
                faces=np.asarray(held[f"{body}__f"], dtype=np.int64), process=False))
    return meshes


class MeshBody:
    """One or more meshes sharing a frame, as a surface sample plus the two questions asked of the geometry."""

    def __init__(self, meshes: "list | Any", *, samples: int = DEFAULT_SAMPLES, seed: int = 0) -> None:
        self.meshes = list(meshes) if isinstance(meshes, (list, tuple)) else [meshes]
        if not self.meshes:
            raise ValueError("a body needs at least one mesh")
        self.area = float(sum(mesh.area for mesh in self.meshes))
        self.points, self.normals = self._sample(samples, seed)
        self.spacing_m = float(np.sqrt(4.0 * self.area / max(len(self.points), 1)))

    def _sample(self, samples: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
        """Surface samples shared out by area, so a small part is not sampled finer than a large one."""
        import trimesh

        points, normals = [], []
        for index, mesh in enumerate(self.meshes):
            share = max(int(round(samples * mesh.area / self.area)), 1)
            drawn, face_index = trimesh.sample.sample_surface(mesh, share, seed=seed + index)
            points.append(np.asarray(drawn, dtype=np.float64))
            normals.append(np.asarray(mesh.face_normals[face_index], dtype=np.float64))
        return np.concatenate(points), np.concatenate(normals)

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Unsigned distance to the nearest surface of the group, metres.

        trimesh builds an R-tree for this and imports ``rtree`` at the call rather than at module import, so a
        box without it fails here, in the one query the whole ruler rests on. It is named rather than left as a
        ModuleNotFoundError, because the ruler going missing and the ruler reporting no hole look the same to a
        reader of a passing suite.
        """
        asked = np.asarray(points, dtype=np.float64)
        best = np.full(len(asked), np.inf)
        for mesh in self.meshes:
            try:
                nearest = mesh.nearest.on_surface(asked)[1]
            except ImportError as missing:
                raise RuntimeError(
                    f"this ruler cannot measure anything on this box: trimesh's nearest-surface query needs an "
                    f"R-tree and raised {missing}. It is pinned in requirements.txt; install the requirements "
                    f"into the interpreter running this. Until then no sphere map can be judged here, and a map "
                    f"nobody judged is not a map anybody may plan on."
                ) from missing
            best = np.minimum(best, np.asarray(nearest, dtype=np.float64))
        return best

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Inside any mesh of the group, by the generalised winding number."""
        asked = np.asarray(points, dtype=np.float64)
        held = np.zeros(len(asked), dtype=bool)
        for mesh in self.meshes:
            held |= np.abs(winding(mesh, asked)) > 0.5
        return held

    def fresh_points(self, seed: int = 12345) -> np.ndarray:
        """Another surface sample of the same size, which is the only thing that says the first was fine enough."""
        return self._sample(len(self.points), seed)[0]

    def measure(
        self,
        centres_m: np.ndarray,
        radii_m: np.ndarray,
        *,
        points: "np.ndarray | None" = None,
        directions: int = DEFAULT_DIRECTIONS,
    ) -> Fidelity:
        """Judge a sphere map against this body: the largest hole, and the furthest reach past the geometry."""
        centres = np.asarray(centres_m, dtype=np.float64).reshape(-1, 3)
        radii = np.asarray(radii_m, dtype=np.float64).reshape(-1)
        if len(centres) != len(radii):
            raise ValueError(f"{len(centres)} centres and {len(radii)} radii: that is not one sphere map")
        asked = self.points if points is None else np.asarray(points, dtype=np.float64)
        if not len(centres):
            raise ValueError("an empty sphere map covers nothing; there is no measurement to make")

        gaps = np.full(len(asked), np.inf)
        for start in range(0, len(centres), 64):
            block = slice(start, start + 64)
            near = (np.linalg.norm(asked[None, :, :] - centres[block][:, None, :], axis=2) - radii[block][:, None])
            gaps = np.minimum(gaps, near.min(axis=0))

        rays = unit_directions(directions)
        worst = 0.0
        for centre, radius in zip(centres, radii):
            surface = centre[None, :] + radius * rays
            outside = ~self.inside(surface)
            if outside.any():
                worst = max(worst, float(self.distance(surface[outside]).max()))
        return Fidelity(uncovered_max_mm=float(gaps.max()) * 1000.0, reach_max_mm=worst * 1000.0,
                        spheres=len(centres), samples=len(asked))
