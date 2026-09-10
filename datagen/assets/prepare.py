"""Everything a fetched collection needs before a build can place it, in the order it needs it.

The order is the point. Skipping the decomposition warm-up does not fail, it just makes the render
pay CoACD inside its own loop, one mesh at a time, which is slow enough to stall a build outright.
The chain lives in a command rather than in a README so that it cannot be skipped by accident.

    normalise  ->  screen  ->  decompose  ->  build  ->  label-grasps  ->  build-cloud-corpus
    \\_____________ prepare-assets ______________/         \\___ neither of these needs a re-render ___/

Three things this file exists to stop happening again.

Units. `measure_mesh` reads every mesh as metres and scales by 1000. A collection authored in
anything else lands wrong by that factor in every axis and by its cube in mass, and every check
downstream still passes. One collection authors in millimetres and lands a 127 mm bracket at
127,000 mm and 1,143,987 kg; another authors in decimetres. Both are caught by the units guard, not
by a reader.

Triangles. Scanned collections carry 3.5 to 10 million vertices per object. Convex decomposition
takes 5 to 30 seconds at 11k to 48k faces, so at seven million it is not slow, it is out of reach.

The pose. A graspability screen that places a mesh upright measures the worst case.
`datagen/config.py` records, on the very option that produces these poses, that a tipped object is
markedly more likely to admit a jaw grasp than an upright one. Screening one pose alone excludes
objects the corpus can label, which makes the asset list worse than no screen at all.
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.utility.log_cfg import create_logger
from datagen.assets.library import CUSTOM_SUFFIXES, MESH_LIBRARY_DIR, SUPPORTED_SOURCES
from datagen.constants import DATAGEN_LOG_DIR

#: What counts as a mesh file anywhere in this module.
#: Derived from `CUSTOM_SUFFIXES`, never restated. A second list drifts from the library's: a suffix
#: the library admits and this one does not is imported, counted, and then silently skipped by both
#: normalise and screen. One list cannot disagree with itself.
MESH_SUFFIXES: Final[tuple[str, ...]] = tuple(f".{s}" for s in CUSTOM_SUFFIXES)

#: The sentinel that means "assign a size" rather than "multiply by this". A sentinel rather than a
#: separate argument because the factor already travels through a process pool as a float, and adding
#: a parallel channel for one source is how two things that must agree stop agreeing.
_ASSIGN: Final[float] = -1.0

__all__ = [
    "MESH_SUFFIXES",
    "PLAUSIBLE_EXTENT_MM",
    "REST_POSES",
    "SCALES",
    "normalise_meshes",
    "prepare_assets",
    "screen_meshes",
    "screen_rows",
]

_LOG = create_logger("datagen.assets.prepare", "assets_prepare.log", log_dir=DATAGEN_LOG_DIR)

#: The band a normalised mesh's longest axis has to land in. This is the idempotency test as well as
#: a sanity one, and it is the geometry rather than a marker file on purpose: a marker can be lost or
#: copied and the geometry cannot.
#:
#: Scaling is not idempotent and this has to be. A killed run leaves more meshes written than its
#: log names, because `pool.map` reports in submission order, so a re-run without this check scales
#: those a second time: a 106 mm canister becomes 10.5 mm, and no later check questions 10 mm.
PLAUSIBLE_EXTENT_MM: tuple[float, float] = (50.0, 500.0)

#: `(scale to apply, why)` per source. A source absent from here keeps its authored scale, which is
#: the honest default: guessing a factor is how a wrongly sized object enters a corpus unnoticed.
SCALES: dict[str, tuple[float, str]] = {
    "asos": (0.1, "authored in decimetres; measured on all 50 meshes, 100 % land in a plausible "
                  "50-500 mm band at this factor and 0 % at either neighbouring power of ten"),
}

#: The rest poses a settled object actually takes, as rotations. Upright first so a reader can tell
#: from `best_pose` when upright was not the answer, which for many graspable meshes it is not.
REST_POSES: tuple[tuple[str, tuple[tuple[float, ...], ...]], ...] = (
    ("upright", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("x-down", ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))),
    ("y-down", ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))),
)


# ------------------------------------------------------------------------------------- normalise

#: Sources with no unit convention at all, whose meshes therefore get a size assigned rather than
#: converted. Artist-uploaded collections are the case: their longest sides span four orders of
#: magnitude, so no single factor fixes the set, because the numbers carry no unit to begin with.
#:
#: The assigned size is the largest lever measured on this pipeline: the same meshes labelled at
#: different assigned sizes yield several times as many grasp labels at the best of them. The curve
#: falls again at the large end because an 85 mm jaw cannot span a proportionally thick object, so
#: the peak is a property of this gripper and not of the meshes, which is the honest way to read it.
ASSIGN_SIZE: Final[frozenset[str]] = frozenset({"objaverse"})

#: Where an assigned object's longest side lands, in millimetres. A band rather than a single value,
#: drawn deterministically from the mesh's own name.
#:
#: A single value would be a lie of a different kind. Scaling every object to the size that gives
#: the most labels produces a corpus in which every object is the same size, which no cell sees and
#: which a model would learn as a prior. The band brackets the measured peak and keeps the variety a
#: scene needs.
#:
#: The size is assigned, not measured. The source states no real-world size, so any number here is
#: a choice. It is recorded per mesh so a reader can see it was chosen.
ASSIGNED_EXTENT_MM: Final[tuple[float, float]] = (80.0, 220.0)


def _implausible(extent_mm: float) -> str:
    """Why this mesh's size is not believable, and the `--scale` that would fix it.

    The guess is named, not applied. A factor of 1000 means a CAD export in millimetres read as
    metres, which is what a CAD package emits by default and therefore the overwhelmingly likely case
    on a customer rail. Saying so costs nothing; applying it silently would turn a visible error into
    an invisible one on the one input nobody here can check.
    """
    low, high = PLAUSIBLE_EXTENT_MM
    for factor, unit in ((0.001, "millimetres"), (0.01, "centimetres"), (0.1, "decimetres"),
                         (10.0, "tenths of a metre")):
        if low <= extent_mm * factor <= high:
            return (f"IMPLAUSIBLE SIZE {extent_mm:.0f} mm: this mesh is probably authored in {unit} "
                    f"and every mesh here is read as METRES. Pass --scale {factor} to convert it; "
                    f"left alone it lands {1 / factor:.0f}x too large in every axis and by its cube "
                    f"in mass, and nothing downstream will refuse it")
    return (f"IMPLAUSIBLE SIZE {extent_mm:.0f} mm, outside the plausible {low:.0f} to {high:.0f} mm, "
            f"and no simple unit factor lands it in range. Left alone; check the export")


def _assigned_factor(path: Path, extent_mm: float) -> tuple[float, float]:
    """`(factor, target_mm)` for a mesh whose collection has no unit convention.

    Deterministic in the file name, so the same mesh gets the same size in every run and on every
    machine, and two runs of a corpus are comparable.
    """
    import hashlib  # noqa: PLC0415 (only this path needs it)

    digest = hashlib.sha256(path.name.encode("utf-8")).digest()
    low, high = ASSIGNED_EXTENT_MM
    target = low + (high - low) * (int.from_bytes(digest[:4], "big") / 0xFFFFFFFF)
    return (target / extent_mm if extent_mm > 0 else 1.0), target


def _normalise_one(task: tuple[str, float, int]) -> str:
    path_text, scale, faces = task
    import open3d as o3d  # noqa: PLC0415 (heavy, and only this half needs it)

    path = Path(path_text)
    mesh = o3d.io.read_triangle_mesh(str(path))
    before = len(mesh.triangles)
    if before == 0:
        return f"{path.name}: no triangles, left alone"
    extent_mm = float(max(np.asarray(mesh.get_max_bound())
                          - np.asarray(mesh.get_min_bound()))) * 1000.0
    changed = False
    note = ""
    if scale == _ASSIGN and not (PLAUSIBLE_EXTENT_MM[0] <= extent_mm <= PLAUSIBLE_EXTENT_MM[1]):
        # A size is chosen, not converted, and the line says which. See `ASSIGN_SIZE`.
        factor, target = _assigned_factor(path, extent_mm)
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * factor)
        changed = True
        note = f", assigned {target:.0f} mm (was {extent_mm:.0f})"
    elif scale not in (1.0, _ASSIGN) and not (
            PLAUSIBLE_EXTENT_MM[0] <= extent_mm <= PLAUSIBLE_EXTENT_MM[1]):
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * scale)
        changed = True
    elif scale == 1.0 and not (PLAUSIBLE_EXTENT_MM[0] <= extent_mm <= PLAUSIBLE_EXTENT_MM[1]):
        # The gap a source at scale 1.0 falls through, and `custom` is the one that matters. The two
        # branches above cover a source that assigns a size and one with an authored conversion. A
        # source with neither keeps its authored scale, and nothing else checks that scale.
        #
        # A CAD export in millimetres then lands at tens of thousands of millimetres, screens
        # `status: ok`, earns no jaw label because nothing that size can be gripped, and the run
        # reports no problem. This module's own docstring records the same failure on a research
        # collection ("a 127 mm bracket at 127,000 mm and 1,143,987 kg"); the units guard that
        # catches it there does not cover this case.
        #
        # It warns rather than rescaling. A source with no authored convention has not stated its
        # units, and guessing would replace a visible 1000x error with an invisible one. The note
        # names the factor that would land it in range so a caller can pass `--scale`.
        _note = _implausible(extent_mm)
        note = f", {_note}"
    if before > faces:
        # Quadric only. Vertex clustering is much faster and it shatters the mesh, which is worse
        # than useless: a mesh in pieces cannot be closed into a solid, so it cannot be labelled or
        # collided, and the loader then refuses objects that look bad when what was bad was the
        # transform.
        #
        # The failure is structural rather than a tuning problem, and no cleanup step recovers it:
        # with and without cleanup the shattered piece count is identical. Quadric decimation
        # collapses edges, so connectivity survives by construction; clustering merges vertices
        # inside a voxel and tears the surface where it does.
        mesh = mesh.simplify_quadric_decimation(faces)
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        changed = True
    if not changed:
        # The note survives the early return. A mesh that needs no decimation and no rescale still
        # carries the implausible-size warning computed above. A warning that is calculated and then
        # thrown away is worse than one never written: the code reads as if the check exists.
        return f"{path.name}: {before} faces, already prepared{note}"
    # Normals first, and the return value is checked. Open3D refuses to write an STL without vertex
    # normals: it logs `Write STL failed: compute normals first`, returns False, and leaves a
    # zero-byte file behind. An unchecked return value therefore lets a rescale of an `.stl` destroy
    # the mesh, and the next stage reports it as `unclosable`, a geometry verdict on a file this
    # module emptied itself. `.stl` is what a CAD package emits by default, so this sits on the
    # single most likely customer path.
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False):
        # A refusal, not a warning. The file on disk is now whatever the failed write left, and
        # continuing would hand the next stage a mesh this module broke.
        raise OSError(
            f"open3d refused to write {path.name} after normalising it. The file may now be empty; "
            f"re-import it from the original export.")
    span = np.asarray(mesh.get_max_bound()) - np.asarray(mesh.get_min_bound())
    return (f"{path.name}: {before} -> {len(mesh.triangles)} faces, "
            f"extent {[round(float(v) * 1000.0, 1) for v in span]} mm{note}")


def normalise_meshes(source: str, *, library: Path | None = None, faces: int = 20000,
                     scale: float | None = None, jobs: int = 4,
                     report: Any = print) -> dict[str, int]:
    """Scale one collection into metres and decimate it into the face budget. Idempotent."""
    root = Path(library or MESH_LIBRARY_DIR) / source
    if not root.is_dir():
        report(f"  {source}: no directory at {root}")
        return {"meshes": 0, "changed": 0}
    meshes = sorted(p for p in root.iterdir() if p.suffix.lower() in MESH_SUFFIXES)
    if not meshes:
        return {"meshes": 0, "changed": 0}

    factor, why = SCALES.get(source, (1.0, "no factor recorded for this source"))
    if source in ASSIGN_SIZE:
        factor, why = _ASSIGN, (
            f"no unit convention at all; a size in {ASSIGNED_EXTENT_MM} mm is ASSIGNED per mesh, "
            f"deterministically from its name")
    if scale is not None:
        factor, why = scale, "given by the caller"
    report(f"  {source}: {len(meshes)} mesh(es), scale {factor} ({why}), face budget {faces}")

    changed = 0
    # Four workers by default, not cpu_count. Quadric decimation holds its error structure over
    # every face of the original mesh, so a worker on a multi-million-face scan is heavy and enough
    # of them will take a machine down. A killed run leaves meshes half-transformed with a log
    # naming fewer of them than were written, because `pool.map` reports in submission order.
    with futures.ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        tasks = [(str(p), factor, faces) for p in meshes]
        for line in pool.map(_normalise_one, tasks):
            changed += int("->" in line or "assigned" in line)
            _LOG.info("normalise %s", line)
    # The reader caches by path and this function rewrites the path. `load_mesh_shape` is
    # `lru_cache`d on the path, so any process that reads a mesh, normalises it and reads it again
    # gets the geometry from before the rescale, with no error and no warning: a report saying every
    # mesh changed, beside extents that have not moved.
    #
    # The commands are separate processes today, so the stale read cannot reach a corpus. `--jobs`
    # and a future in-process pipeline would both make it able to.
    from datagen.assets.meshes import load_mesh_shape  # noqa: PLC0415

    load_mesh_shape.cache_clear()
    report(f"  {source}: {changed} of {len(meshes)} mesh(es) changed")
    return {"meshes": len(meshes), "changed": changed}


# ---------------------------------------------------------------------------------------- screen

def _screen_one(task: tuple[str, str, str]) -> dict[str, Any]:
    source, path_text, density_name = task
    logging.disable(logging.INFO)
    from datagen.assets.meshes import load_mesh_shape  # noqa: PLC0415
    from datagen.grasps.labels import (  # noqa: PLC0415
        DENSITIES,
        SceneGeometry,
        label_jaw_grasps,
        label_suction_grasps,
    )
    from datagen.grasps.shapes import Solid  # noqa: PLC0415

    path = Path(path_text)
    row: dict[str, Any] = {"asset_id": f"{source}_{path.stem}", "source": source,
                           "file": path.name}
    shape = load_mesh_shape(str(path))
    if shape is None:
        # A status, not a count. A constant label count stamped on a mesh that could not be loaded
        # reads exactly like a measurement, and every selection built on it then draws from a pool
        # whose entries do not exist.
        row["status"] = "unclosable"
        return row

    extent = np.asarray(shape.extent_mm, dtype=float)
    density = DENSITIES[density_name]
    best_jaw = best_suction = best_pose = 0
    for index, (_name, rotation) in enumerate(REST_POSES):
        matrix = np.asarray(rotation, dtype=float)
        rested = np.abs(matrix @ extent)
        solid = Solid(kind="mesh", half_extent_mm=extent / 2.0, rotation=matrix,
                      centre_mm=np.array([0.0, 0.0, float(rested[2]) / 2.0]), instance_id=0,
                      asset_id=row["asset_id"], mesh=shape)
        geometry = SceneGeometry(scene_id="screen", family="alone", objects={0: solid}, walls=())
        try:
            jaw, _ = label_jaw_grasps(geometry, 0, density=density)
            suction, _ = label_suction_grasps(geometry, 0, density=density)
        except Exception as exc:                               # noqa: BLE001 (reported, not raised)
            row["status"] = f"labeller-error: {type(exc).__name__}"
            return row
        if len(jaw) > best_jaw:
            best_jaw, best_pose = len(jaw), index
        best_suction = max(best_suction, len(suction))
    row.update({
        "status": "ok",
        "extent_mm": [round(float(v), 2) for v in sorted(extent)],
        "jaw": best_jaw,
        "suction": best_suction,
        "best_pose": REST_POSES[best_pose][0],
        # Kept so the disagreement between the labeller and the bounding-box proxy stays queryable
        # without a re-run. The proxy calls meshes graspable that earn no label, and refuses meshes
        # that do have grasps.
        "box_screen_jaw_ok": bool(float(min(extent)) <= 85.0),
    })
    return row


def screen_meshes(sources: list[str] | None = None, *, library: Path | None = None,
                  out: Path, density: str = "grid", jobs: int = 10,
                  want_graspable: int | None = None,
                  report: Any = print) -> list[dict[str, Any]]:
    """Label every mesh alone, in every rest pose, and write what the labeller found.

    `want_graspable` answers the question a customer actually asks. Nobody wants every mesh in a
    library; they want enough objects to train on, and the honest unit is not meshes screened but
    meshes that turned out to be jaw-graspable. The screen stops as soon as it has that many and
    reports how many it had to look at to get them, which is the number that lets the next person
    budget the run.

    A stopped screen is stamped `partial`, and it has to be. The file it writes looks exactly like
    a complete one, every consumer treats a mesh with no row as unmeasured, and a screen that covered
    a fifth of a library would silently become "this library has few graspable objects". The stamp is
    in the file and the warning is in the log.
    """
    root = Path(library or MESH_LIBRARY_DIR)
    chosen = sources or [s for s in SUPPORTED_SOURCES if s != "custom"]
    tasks: list[tuple[str, str, str]] = []
    for source in chosen:
        suffix = SUPPORTED_SOURCES.get(source, "*")
        directory = root / source
        if not directory.is_dir():
            continue
        # Filtered by extension even when the source says "*". A collection may carry files that
        # are not meshes, and one of them is required rather than incidental: CC-BY obliges naming
        # the author, so the Objaverse fetch writes an ATTRIBUTION.tsv beside the meshes it credits.
        # Globbing "*" turns that file into a screening task, which fails harmlessly but also
        # inflates the reported mesh count by one. A count that is wrong by one is still a number
        # somebody will quote.
        found = sorted(p for p in directory.glob("*" if suffix == "*" else f"*.{suffix}")
                       if p.suffix.lower() in MESH_SUFFIXES)
        report(f"  {source}: {len(found)} mesh(es)")
        tasks += [(source, str(p), density) for p in found]
    if not tasks:
        report("  nothing to screen")
        return []

    target = f", stopping at {want_graspable} jaw-graspable" if want_graspable else ""
    report(f"  screening {len(tasks)} mesh(es) in {len(REST_POSES)} rest pose(s) "
           f"at density {density}, {jobs} job(s){target}")
    rows: list[dict[str, Any]] = []
    jaw = 0
    pool = futures.ProcessPoolExecutor(max_workers=max(1, jobs))
    try:
        # `map` yields in submission order, so a stop here is reproducible: the same library and the
        # same target screen the same prefix. A completion-ordered stream would give a different
        # subset on every machine.
        for done, row in enumerate(pool.map(_screen_one, tasks), start=1):
            rows.append(row)
            jaw += int(row.get("jaw", 0) > 0)
            if done % 100 == 0 or done == len(tasks):
                report(f"    [{done}/{len(tasks)}] jaw-graspable {jaw}")
            if want_graspable and jaw >= want_graspable:
                report(f"    stopped at {jaw} jaw-graspable after {done} of {len(tasks)} mesh(es) "
                       f"({100.0 * jaw / done:.1f} % yield)")
                break
    finally:
        # `cancel_futures`, or the stop is only a stop for the reader. Without it the pool drains
        # every task already queued, which for a 10-job pool over a large library is most of the work
        # the caller just asked to skip.
        pool.shutdown(wait=True, cancel_futures=True)
    partial = len(rows) < len(tasks)

    # The density travels with the verdict. A screen taken at `default` and one taken at `grid`
    # are different measurements of the same meshes and the files look identical, so a bare list of
    # rows cannot be read safely: the same object can score label counts a factor of twenty apart,
    # with nothing in either file to tell them apart. A label count means nothing without the
    # density.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "density": density,
        "rest_poses": [name for name, _ in REST_POSES],
        # The stamp, for the same reason the density is here. A partial screen and a complete one
        # are different statements about a library and the files are otherwise identical.
        "partial": partial,
        "screened": len(rows),
        "of": len(tasks),
        "want_graspable": want_graspable,
        "rows": rows,
    }, indent=1, sort_keys=True), encoding="utf-8")
    if partial:
        # ASCII in the printed line: this CLI's output is read on a Windows console at cp1252.
        report(f"  PARTIAL: {len(rows)} of {len(tasks)} mesh(es) screened. The file says so; a "
               f"reader that ignores it will read 'few graspable objects' where the truth is "
               f"'most were never looked at'")
    for source in chosen:
        mine = [r for r in rows if r.get("source") == source]
        if not mine:
            continue
        loaded = [r for r in mine if r.get("status") == "ok"]
        jaw_rows = [r for r in loaded if r.get("jaw", 0) > 0]
        suction_rows = [r for r in loaded if r.get("suction", 0) > 0]
        missed = [r for r in jaw_rows if not r.get("box_screen_jaw_ok")]
        report(f"  {source:10s} {len(mine):>4} meshes   loaded {len(loaded):>4}   "
               f"jaw {len(jaw_rows):>4}   suction {len(suction_rows):>4}   "
               f"the box proxy would have missed {len(missed):>4}")
    return rows


def screen_rows(parsed: Any) -> list[dict[str, Any]]:
    """The rows out of a parsed screen file, in either shape one has been written in.

    Measured 2026-09-10 against this repository's own committed
    `datagen/assets/screens/screen.json`: :func:`screen_meshes` wraps the rows in a dict that also
    carries the density and the partial stamp, and `why-no-jaw --from-screen` walked the parsed JSON
    as a bare list. On a dict that iterates the keys, so the reader called `.get` on the string
    "density", and the form the README documents could not run at all.

    A bare list is a screen written before the wrapper existed. It is still read, because refusing it
    would turn a file somebody measured last month into a lost measurement; it simply cannot say
    which density it was taken at. `scenes/layout.load_jaw_screen` carries the same rule for the
    other reader of this file.
    """
    if isinstance(parsed, dict):
        return list(parsed.get("rows", []))
    return list(parsed)


def asset_ids_from_screen(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """`{source: [asset_id]}` for every mesh graspable by either modality.

    Both, because a mesh with no jaw grasp is not waste: most of the meshes that earn no jaw label
    are suction-graspable, and a jaw-only list would drop them from a corpus that labels both.
    """
    out: dict[str, list[str]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("jaw", 0) <= 0 and row.get("suction", 0) <= 0:
            continue
        out.setdefault(str(row["source"]), []).append(str(row["asset_id"]))
    return {k: sorted(v) for k, v in sorted(out.items())}


# --------------------------------------------------------------------------------------- prepare

def prepare_assets(sources: list[str] | None = None, *, library: Path | None = None,
                   screen_out: Path, ids_out: Path | None = None, density: str = "grid",
                   faces: int = 20000, jobs: int = 6, want_graspable: int | None = None,
                   report: Any = print) -> dict[str, Any]:
    """normalise, then screen, then say what still has to happen. In that order, because it matters.

    Deliberately does not run the decomposition itself: that reads the manifest a config produces, so
    it belongs to a config rather than to a library, and running it here would warm whatever the
    default config places instead of what the caller is about to build.
    """
    chosen = sources or [s for s in SUPPORTED_SOURCES if s != "custom"]
    report("normalise")
    for source in chosen:
        normalise_meshes(source, library=library, faces=faces, jobs=jobs, report=report)

    report("\nscreen")
    rows = screen_meshes(chosen, library=library, out=screen_out, density=density,
                         jobs=max(jobs, 8), want_graspable=want_graspable, report=report)

    ids = asset_ids_from_screen(rows)
    if ids_out is not None:
        ids_out.parent.mkdir(parents=True, exist_ok=True)
        ids_out.write_text(json.dumps(ids, indent=1, sort_keys=True), encoding="utf-8")

    total = sum(len(v) for v in ids.values())
    jaw = sum(1 for r in rows if r.get("jaw", 0) > 0)
    suction = sum(1 for r in rows if r.get("suction", 0) > 0)
    report(f"\n{total} asset id(s): {jaw} jaw-graspable, {suction} suction-graspable")
    report(f"  screen  -> {screen_out}")
    if ids_out is not None:
        report(f"  ids     -> {ids_out}")
    report("\nNEXT, and the first one is not optional:")
    report("  1. point your config's assets.jaw_screen_path and assets.mesh_asset_ids_path at those")
    report("  2. python -m datagen decompose --config <config>     <- skipping this does not fail,")
    report("     it makes the render pay the decomposition one mesh at a time inside its own loop")
    report("  3. python -m datagen build --config <config>")
    report("  4. python -m datagen label-grasps --density grid     <- needs no re-render")
    report("  5. python -m datagen build-cloud-corpus")
    return {"assets": total, "jaw": jaw, "suction": suction, "rows": len(rows)}
