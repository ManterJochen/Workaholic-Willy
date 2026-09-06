"""Convex decomposition, the thing that lets a physics engine collide the shape the scene contains.

Why this is a correctness gate and not an optimisation. MuJoCo resolves mesh contact against the
convex hull. That is exact for a box, a cylinder or a sphere, and for those it is what
`MujocoRenderer` does. It is wrong for anything with a pocket: a mug's hull is a solid cylinder, a
bucket's is a solid tub, a bracket's is the block it was milled from. Settling those against their
hulls does not fail loudly. It produces a plausible depth image over geometry the scene does not
contain, which is the same class as a scene whose colour frames are all black passing every check as
``ok`` and being found only by opening the folder.

So the engine refuses scanned and composite assets by name rather than settle them wrongly. This
module lifts that refusal: it splits a concave mesh into convex parts, each of which MuJoCo collides
exactly, and the union of them is the shape.

The parameters here are set against measurement rather than inherited. CoACD's own defaults are
unaffordable at bank scale, costing several times what these settings cost per mesh, and the result
is computed once and cached to disk. Decimation is free in time and cuts the part vertex count about
threefold, which is what a MuJoCo model actually has to carry.

The fidelity is a real limit. The sum of part volumes cannot answer whether the pocket survived,
because convex parts overlap and their sum over-counts by construction. The union is what a solver
collides, and it sits between the true volume and the convex hull: on a scanned mug most of the
concavity comes back, and on a thin shell such as a shoe rather less.

One cheaper setting is refused by that same measurement. A `preprocess_resolution` of 12 is two to
three times faster and produces a union larger than the convex hull, because the coarse remesh bulges
outside the original surface. That is worse than not decomposing at all, and the timing alone reads
like a bargain.

The cache is keyed on geometry, not on a file name. Two assets can name the same mesh at different
scales, and a customer can replace a part in place while keeping its id. Hashing the vertices and
faces the caller actually passes means a changed mesh gets a changed key, rather than a stale
decomposition of the shape that used to be there.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.utility.log_cfg import create_logger
from datagen.constants import CONVEX_DECOMPOSITION_LOG_FILE, DATAGEN_LOG_DIR

__all__ = [
    "CACHE_DIR",
    "DEFAULT_DECOMPOSITION",
    "DecompositionSettings",
    "DecompositionUnavailable",
    "convex_parts",
    "decomposition_available",
    "warm_asset",
]

logger = create_logger("ConvexDecomposition", CONVEX_DECOMPOSITION_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Beside the meshes it derives from rather than under `logs/`: this is a derived asset, it must
#: survive a log rotation, and `assets/` is already the untracked tree a clone fetches into.
CACHE_DIR: Path = Path("assets/cache/coacd")


class DecompositionUnavailable(RuntimeError):
    """Raised when a concave mesh needs decomposing and the library is not installed.

    Raised rather than falling back to the hull, because the hull is the wrong answer here and a
    silent wrong answer stays wrong for the life of the corpus.
    """


@dataclass(frozen=True, slots=True)
class DecompositionSettings:
    """What the decomposition may cost, and how closely it must follow the surface.

    Every field is a CoACD argument, gathered here so the cache key can cover all of them. A
    decomposition computed under different settings is a different decomposition, and reusing one
    under a new name is how a cache turns into a lie.
    """

    #: Concavity below which a part is accepted as convex enough. Higher is coarser and much faster.
    threshold: float = 0.08
    #: Hard cap on the parts one mesh may become. Bounds the MuJoCo model and bounds the worst case.
    #:
    #: 32 rather than 16, because a cap of 16 binds: meshes come back at exactly the cap, so the
    #: fidelity that results measures this constant rather than CoACD. At 32 the concavity threshold
    #: binds first, which is where the constraint belongs, and raising the cap further changes
    #: nothing.
    max_convex_hull: int = 32
    #: Voxel resolution of CoACD's own preprocessing remesh.
    preprocess_resolution: int = 30
    #: Sampling resolution of the concavity metric.
    resolution: int = 1000
    #: The Monte-Carlo tree search that chooses cut planes. The dominant cost, and the reason CoACD's
    #: defaults are unaffordable: 20/150/3 costs several times what 12/60/2 costs and buys only a
    #: handful more parts.
    mcts_nodes: int = 12
    mcts_iterations: int = 60
    mcts_max_depth: int = 2
    #: Simplify each part's hull. Free in time, roughly a third of the vertices.
    decimate: bool = True
    max_ch_vertex: int = 64
    #: Fixed, and the whole run rests on it. CoACD's search is stochastic: unseeded it returns a
    #: different decomposition every call, so two renders of one scene would settle differently and
    #: the corpus would not be reproducible. That failure only surfaces long after the data is made.
    seed: int = 0

    def as_kwargs(self) -> dict:
        return {
            "threshold": self.threshold,
            "max_convex_hull": self.max_convex_hull,
            "preprocess_resolution": self.preprocess_resolution,
            "resolution": self.resolution,
            "mcts_nodes": self.mcts_nodes,
            "mcts_iterations": self.mcts_iterations,
            "mcts_max_depth": self.mcts_max_depth,
            "decimate": self.decimate,
            "max_ch_vertex": self.max_ch_vertex,
            "seed": self.seed,
        }


#: The settings above. Named, so a caller that wants CoACD's own defaults has to say so.
DEFAULT_DECOMPOSITION = DecompositionSettings()


def decomposition_available() -> bool:
    """Is the library importable on this machine? Cheap enough to ask once per scene.

    `OSError` counts as "no". A probe whose whole job is to answer a yes/no question must not raise
    into the render loop, and a library that is installed but that the OS refuses to load raises
    `OSError`, not `ImportError`. Windows Smart App Control refuses an unsigned native library with
    `OSError: [WinError 4551]` days after it was installed, so this is not hypothetical.
    """
    return decomposition_refusal() is None


def decomposition_refusal() -> "str | None":
    """Why CoACD cannot be used here, or `None` when it can.

    Absent and unloadable are two faults with two remedies, and one sentence for both is worse than
    no sentence. Telling an operator "CoACD is not installed here: `pip install coacd`" for a library
    that is installed and that the machine refused to load sends them to reinstall a package they
    already have, and costs them the one thing the raw error gave them: the name of the policy that
    blocked it.

    Windows Smart App Control refuses an unsigned native library days after installation, and that
    import raises `OSError: [WinError 4551]`, not `ImportError`. The two branches below stay separate
    for that reason.
    """
    try:
        import coacd  # noqa: F401, PLC0415 - optional dependency, probed rather than used
    except ImportError:
        return ("CoACD is not installed: `pip install coacd` (MIT, about 1.5 MB, wheels for "
                "Windows, macOS and Linux).")
    except OSError as exc:
        return (f"CoACD IS installed, and its native library could not be LOADED ({exc}). This is "
                f"NOT a missing package and reinstalling will not help. On Windows this is usually "
                f"Smart App Control refusing an unsigned build whose reputation it re-evaluated; "
                f"read the CodeIntegrity event log before changing anything.")
    return None


def _cache_key(vertices_mm: np.ndarray, faces: np.ndarray,
               settings: DecompositionSettings) -> str:
    """A hash over the geometry and every setting that shaped the result."""
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(np.ascontiguousarray(vertices_mm, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(faces, dtype=np.int64).tobytes())
    digest.update(repr(sorted(settings.as_kwargs().items())).encode("utf-8"))
    return digest.hexdigest()


def _read_cache(path: Path) -> tuple[tuple[np.ndarray, np.ndarray], ...] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path) as blob:
            count = int(blob["count"])
            return tuple((blob[f"v{index}"], blob[f"f{index}"]) for index in range(count))
    except Exception:  # noqa: BLE001 (a corrupt entry costs one recompute, never the run)
        logger.warning("cache entry %s is unreadable; recomputing", path)
        return None


def _write_cache(path: Path, parts: tuple[tuple[np.ndarray, np.ndarray], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"count": np.asarray(len(parts))}
    for index, (vertices, faces) in enumerate(parts):
        payload[f"v{index}"] = np.asarray(vertices, dtype=np.float64)
        payload[f"f{index}"] = np.asarray(faces, dtype=np.int32)
    # Written beside the target and renamed, so a run killed mid-write leaves no half file that the
    # next run happily reads as a decomposition.
    temporary = path.with_suffix(".partial.npz")
    # `cast` because numpy's stub declares `savez(file, *args, allow_pickle=..., **kwds)`, so a
    # `**dict[str, ndarray]` is read as a candidate for `allow_pickle: bool`. The call is correct.
    np.savez(temporary, **cast("Any", payload))
    temporary.replace(path)


def convex_parts(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    *,
    settings: DecompositionSettings = DEFAULT_DECOMPOSITION,
    cache_dir: Path | str = CACHE_DIR,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """``((vertices_mm, faces), ...)`` convex pieces whose union is the mesh.

    Cached on disk by geometry and settings, because the decomposition costs seconds per mesh and a
    render draws from the same asset bank over and over.

    Raises :class:`DecompositionUnavailable` when CoACD is not installed. Deliberately: the caller
    asked for a decomposition because the hull is wrong for this mesh, and handing back the hull
    anyway would answer a question nobody asked.
    """
    vertices_mm = np.ascontiguousarray(vertices_mm, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)

    key = _cache_key(vertices_mm, faces, settings)
    path = Path(cache_dir) / f"{key}.npz"
    cached = _read_cache(path)
    if cached is not None:
        return cached

    try:
        import coacd  # noqa: PLC0415 (optional dependency, imported only on this path)
    except (ImportError, OSError) as exc:
        # The reason, not a fixed sentence. "not installed here: `pip install coacd`" is a false
        # instruction for a library the OS blocked, and sends the reader to redo something that
        # already worked.
        raise DecompositionUnavailable(
            "This mesh is concave, so MuJoCo's convex-hull contact would collide a shape the scene "
            f"does not contain, and convex decomposition is unavailable. {decomposition_refusal()} "
            "Alternatively set render.engine to `none`, which seats objects analytically and never "
            "asks a solver about contact at all."
        ) from exc

    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(vertices_mm, faces), **settings.as_kwargs())
    result = tuple((np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32))
                   for vertices, triangles in parts)
    _write_cache(path, result)
    logger.info("decomposed %d faces into %d part(s), %d vertices, cached as %s",
                len(faces), len(result), sum(len(v) for v, _ in result), path.name)
    return result


def warm_asset(record: Any) -> tuple[str, int, float]:
    """Decompose one asset into the cache. Returns ``(asset_id, parts, seconds)``.

    A separate pass because the cost lands in the wrong place otherwise. Decomposition is seconds per
    mesh and the cache is cold exactly once, so doing it inline during a render spends that time
    single-file inside the render loop. Run over the bank across processes first and the render finds
    every entry warm.

    It must build the same geometry the render will. The cache is keyed on the vertices and faces the
    caller passes, so this goes through `_mesh_for` at scale 1 exactly as `_collision_parts` does. A
    warm-up that loaded the mesh its own way would fill the cache with keys nothing ever asks for,
    and would look like it worked.
    """
    import time

    from datagen.render.noengine import _mesh_for, composite_part_meshes

    started = time.perf_counter()
    asset_id = str(getattr(record, "asset_id", ""))
    if composite_part_meshes(record, 1.0):
        # Authored parts are already convex; nothing to compute and nothing to cache.
        return (asset_id, 0, 0.0)
    vertices, faces = _mesh_for(record, 1.0)
    parts = convex_parts(vertices, faces)
    return (asset_id, len(parts), time.perf_counter() - started)
