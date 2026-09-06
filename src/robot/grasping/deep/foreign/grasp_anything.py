"""Importing a public 6-DoF grasp corpus as scene files this pipeline already reads.

The import writes scene `.npz` files in the format `datagen` writes, rather than feeding samples
into the training loop through a second path. So the corpus index, the asset-disjoint folds,
`build_sample`, the seed mixture, every probe and every metric run on imported data with no change
and no flag: two producers, one consumer, and the consumer cannot tell them apart. A parallel loader
would instead leave every future measurement asking which path a sample came through.

What the source publishes:

    pc/<scene>.npy            (8192, 6) float64   XYZ in metres plus RGB in [0,1]
    grasp/<scene>_<i>         pickle of (Rts (G,4,4) float64, ws (G,) float64)   no extension
    pc_mask/<scene>_<i>.npy   the object's points
    grasp_prompt/<scene>.pkl  list[str], one instruction per object index

The missing extension is not cosmetic. The publisher's own training loader opens
`grasp/<scene>_<i>.pkl` inside a bare `try/except: continue`, so run against the archive as shipped
it loads zero grasps and raises nothing. The extensionless name is globbed here and a miss refuses.

The convention, which the files do not state and which is measured rather than assumed:

    approach       R[:, 2]      third column, pointing into the object
    closing axis   R[:, 0]      first column
    translation    the gripper mounting face, about 173 mm behind the annotated point

The 173 mm is re-derived from the data being imported and recorded, never hardcoded: the fit is
what the import writes into its provenance, so a revised source shows up as a changed offset rather
than as a silently wrong one.

Three things this source does not have, each handled rather than papered over.

No normals. The release ships none, and the sample contract requires unit normals in feature
channels 0 to 2. They are estimated here by local PCA and oriented toward the camera, which sits at
the origin by construction. An estimated normal is not a measured one and the scene stamps that.

No jaw contacts, and the stored width is not a contact separation. The stored opening is wider than
the object it holds, so a point at `centre +- w/2 * axis` is in free air by design. `grasp_width_mm`
means the separation at contact, so the width is re-derived from where the cloud actually is between
the jaws. See `_contact_and_width`.

No asset identity. Scenes are generated images, so there is no object that recurs across scenes and
asset-disjoint folds are not achievable. The scene id becomes the asset group, which makes folds
scene-disjoint and is also exactly what the source's own structure demands: grasps come in blocks of
twenty jittered variants of one 2D rectangle, so any split finer than the scene leaks near-duplicates
across it.
"""

from __future__ import annotations

import io
import json
from concurrent import futures
import pickle  # noqa: S403 (the source publishes pickles; see `_load_grasps` for the guard)
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Sequence

import numpy as np

from src.robot.grasping.deep.foreign.remote_zip import RemoteZip, ZipMember
from src.robot.grasping.deep.net.gripper import JAW_GEOMETRY

__all__ = ["DEFAULT_GRIPPER", "SOURCE", "SceneTriple", "estimate_normals", "fit_centre_offset",
           "import_scenes"]

#: Which jaw profile the imported labels belong to. The source names the 2F-140, and `wide_140` is
#: the profile carried for it.
#:
#: The source's own aperture is wrong and is not copied. Its code caps the opening at 0.202 m and
#: its paper attributes that to "the Robotiq 2F-140 specifications", but a 2F-140 opens 140 mm: the
#: 202 appears to come from a Franka-Panda gripper mesh rescaled by 1.75, whose lineage is visible in
#: the shared mantissa of its vertex constants. The aperture used for filtering is 140, and a label
#: needing more is refused rather than believed.
DEFAULT_GRIPPER: Final[str] = "wide_140"

_HOST: Final[str] = "https://huggingface.co/datasets/airvlab/Grasp-Anything-6D"
_FILES: Final[str] = f"{_HOST}/resolve/main"
_API: Final[str] = "https://huggingface.co/api/datasets/airvlab/Grasp-Anything-6D/tree/main"

#: The cloud archive, split across five raw byte ranges of one Zip64. Order is load-bearing: every
#: offset in its central directory is in concatenated coordinates.
_CLOUD_PARTS: Final[tuple[str, ...]] = tuple(f"pc.parta{letter}" for letter in "abcde")

#: Which rotation column means what. Not stated by the publisher; measured. See the module docstring.
_APPROACH_COLUMN: Final[int] = 2
_AXIS_COLUMN: Final[int] = 0

#: Candidate displacements of the stored translation from the annotated point, along the approach, in
#: metres. The window is centred on the four published estimates and wide enough that the fit cannot
#: sit on its own boundary, which is the failure a narrower grid produces in `prove_convention`.
_OFFSET_GRID_M: Final[tuple[float, ...]] = tuple(v / 1000.0 for v in range(100, 251, 5))

#: Neighbours used to estimate a normal. Small enough to follow a 10 mm feature, large enough that a
#: single noisy depth pixel does not define a plane.
_NORMAL_NEIGHBOURS: Final[int] = 24

#: How far from the closing-axis line a cloud point may sit and still count as between the jaws. The
#: fingers of the reference hand are about 20 mm across, so this is the finger, not a guess.
_JAW_CORRIDOR_MM: Final[float] = 12.0

#: Below this observed separation the grasp is refused rather than believed. On a single-view cloud a
#: tiny extent is the signature of having seen one face of the object, not of a 4 mm object, and a
#: width is a trained target so a wrong one is worse than a missing one.
_MIN_SEPARATION_MM: Final[float] = 10.0

#: Stamped into every imported scene, so a corpus that mixes sources says so and a KPI can split on
#: it. `datagen` writes its own engine name here.
_ENGINE: Final[str] = "imported"

#: Bumped only when the imported layout changes in a way a reader must notice.
_CORPUS_VERSION: Final[int] = 5


@dataclass(frozen=True, slots=True)
class SceneTriple:
    """One scene, and the members that describe it. Objects with no grasp file are a normal case."""

    scene: str
    cloud: ZipMember
    grasps: tuple[ZipMember, ...]
    masks: tuple[ZipMember, ...] = ()
    prompt: ZipMember | None = None

    @property
    def objects(self) -> int:
        return len(self.grasps)


@dataclass(frozen=True, slots=True)
class ForeignSource:
    """Everything needed to name the corpus in a provenance stamp."""

    key: str
    title: str
    licence: str
    url: str
    scenes: int


SOURCE: Final[ForeignSource] = ForeignSource(
    key="grasp_anything_6d",
    title="a public language-driven 6-DoF grasp corpus, 994,860 single-view scenes",
    licence="MIT",
    url=_HOST,
    scenes=994_860,
)


def _sizes() -> dict[str, int]:
    with urllib.request.urlopen(_API, timeout=60) as response:
        tree = json.loads(response.read().decode("utf-8"))
    return {entry["path"]: int((entry.get("lfs") or {}).get("size") or entry.get("size") or 0)
            for entry in tree}


def archives(*, timeout: float = 120.0) -> tuple[RemoteZip, RemoteZip, RemoteZip, RemoteZip]:
    """`(clouds, grasps, masks, prompts)`, addressed by range request rather than downloaded.

    203 GB, 23 GB, 1.9 GB and 0.36 GB. None is fetched; see `remote_zip` for why that works and for
    the two ways it silently would not.

    The masks and prompts matter more than their size suggests. A training unit is an object, so a
    scene whose points all carry instance zero collapses to one unit however many things are in it;
    the masks are `(8192,) uint8` per object, exactly the per-point instance id this format wants,
    and they multiply the units a scene yields. The prompts are one instruction per object, and this
    is a vision-language backend: they are the half of the problem the local corpus cannot generate
    at all.
    """
    sizes = _sizes()
    missing = [name for name in (*_CLOUD_PARTS, "grasp.zip", "pc_mask.zip", "grasp_prompt.zip")
               if name not in sizes]
    if missing:
        raise RuntimeError(f"the source is missing {missing}; its layout has changed and this "
                           f"importer needs a decision rather than a default")
    clouds = RemoteZip([f"{_FILES}/{name}" for name in _CLOUD_PARTS],
                       [sizes[name] for name in _CLOUD_PARTS], timeout=timeout)
    grasps = RemoteZip([f"{_FILES}/grasp.zip"], [sizes["grasp.zip"]], timeout=timeout)
    masks = RemoteZip([f"{_FILES}/pc_mask.zip"], [sizes["pc_mask.zip"]], timeout=timeout)
    prompts = RemoteZip([f"{_FILES}/grasp_prompt.zip"], [sizes["grasp_prompt.zip"]], timeout=timeout)
    return clouds, grasps, masks, prompts


def scene_index(clouds: RemoteZip, grasps: RemoteZip, masks: RemoteZip | None = None,
                prompts: RemoteZip | None = None, *, limit: int | None = None,
                cache_dir: Path | None = None, directory_span: int = 4_000_000,
                report: Any = None) -> list[SceneTriple]:
    """Scenes that have a cloud, at least one grasp file, and where possible masks and a prompt.

    Without `cache_dir` the masks and prompts arrive by coincidence. Each archive holds a separate
    central directory: 994,861 entries for the clouds, 1,385,211 for the grasps, 1,872,660 for the
    masks, 994,861 for the prompts. Reading a four-megabyte slice of each and intersecting them by
    scene id yields 12 scenes with clouds and grasps, of which exactly one has a mask and one a
    prompt: the slices do not overlap. An unmatched scene gets an all-zero instance map, so every
    point belongs to the background and the scene collapses to one training unit however many
    objects it holds, and nothing raises.

    So `cache_dir` reads each directory once, in full, and keeps it on disk. About 460 MB across the
    four, a real cost paid a single time against a 229 GB corpus, and it makes the index exact
    instead of accidental. The slice path stays for a quick look and reports its coverage rather
    than leaving a caller to assume.

    A slice is a sample here only because the names are hashes. Scene ids are 64 hex characters
    with no semantics, so directory order carries none either. That is the exception to this
    repository's "a prefix is not a sample" rule, and it is stated because the rule exists.
    """
    def head(archive: RemoteZip, name: str) -> list[ZipMember]:
        if cache_dir is not None:
            return archive.directory(cache=Path(cache_dir) / f"{name}.directory.json")
        offset, size, _ = archive._locate_directory()          # noqa: SLF001 (same package)
        blob = archive._span(offset, min(directory_span, size))  # noqa: SLF001
        start = blob.find(b"PK\x01\x02")
        if start < 0:
            raise ValueError("no central directory entry in the slice read")
        from src.robot.grasping.deep.foreign.remote_zip import (  # noqa: PLC0415
            _parse_directory)
        return [m for m in _parse_directory(blob[start:]) if m.uncompressed > 0]

    def by_object(archive: RemoteZip, name: str) -> dict[str, list[ZipMember]]:
        out: dict[str, list[ZipMember]] = {}
        for member in head(archive, name):
            name = member.name.rsplit("/", 1)[-1].removesuffix(".npy")
            if "_" not in name:
                continue
            out.setdefault(name.rsplit("_", 1)[0], []).append(member)
        return out

    by_scene = by_object(grasps, "grasp")
    mask_scene = by_object(masks, "pc_mask") if masks is not None else {}
    prompt_scene: dict[str, ZipMember] = {}
    if prompts is not None:
        for member in head(prompts, "grasp_prompt"):
            prompt_scene[member.name.rsplit("/", 1)[-1].removesuffix(".pkl")] = member
    triples: list[SceneTriple] = []
    for member in head(clouds, "pc"):
        scene = member.name.rsplit("/", 1)[-1].removesuffix(".npy")
        found = by_scene.get(scene)
        if not found:
            continue
        triples.append(SceneTriple(
            scene, member, tuple(sorted(found, key=lambda m: m.name)),
            tuple(sorted(mask_scene.get(scene, []), key=lambda m: m.name)),
            prompt_scene.get(scene)))
        if limit and len(triples) >= limit:
            break
    if report is not None:
        with_mask = sum(1 for t in triples if t.masks)
        with_prompt = sum(1 for t in triples if t.prompt is not None)
        report(f"  {len(triples)} scene(s) with a cloud and a grasp file")
        report(f"  {with_mask} with object masks, {with_prompt} with a prompt")
        if cache_dir is None and with_mask < len(triples):
            report("  sliced index: a scene without a mask collapses to one training "
                   "unit however many objects it holds. Pass a cache directory for an exact "
                   "index")
    return triples


def _load_grasps(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """The `(poses, widths)` pair the source pickles.

    The source publishes pickles and a pickle can execute code. There is no format-level defence:
    reading this corpus at all means trusting its host, the same trust a `pip install` from the same
    publisher requires. What is defended is the shape: anything that is not a pair of float arrays
    of the right dimensions is refused here rather than reaching the corpus.
    """
    payload = pickle.loads(blob)  # noqa: S301 (see the docstring; the shape is checked below)
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError(f"expected a 2-tuple of (poses, widths), got {type(payload).__name__}")
    poses = np.asarray(payload[0], dtype=np.float64)
    widths = np.asarray(payload[1], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (G,4,4), got {poses.shape}")
    if widths.shape != (len(poses),):
        raise ValueError(f"{widths.shape} widths against {len(poses)} poses")
    return poses, widths


def _load_prompts(blob: bytes) -> list[str]:
    """The instructions the source pickles, one per object index.

    Shape-checked like the grasps: a list of strings or nothing. A pickle that is anything else is a
    changed source, and guessing what it meant is how a corpus grows a column nobody chose.
    """
    payload = pickle.loads(blob)  # noqa: S301 (see `_load_grasps` for the trust boundary)
    if not isinstance(payload, (list, tuple)):
        raise ValueError(f"expected a list of prompts, got {type(payload).__name__}")
    return [str(item) for item in payload]


def estimate_normals(points_m: np.ndarray, *,
                     neighbours: int = _NORMAL_NEIGHBOURS) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals by local PCA, oriented toward the camera. Returns `(normals, valid)`.

    Orientation is not a detail. A plane's normal is defined up to sign, and the contract's normal
    is sign-meaningful: a head trained on the control learns `approach = -normal`, so a flipped
    normal teaches it to approach from inside the object. The camera sits at the origin by
    construction here, so the outward normal is the one with a negative component along the view
    ray.

    `valid` is False where the neighbourhood is degenerate, which the scene stamps rather than
    hiding: `normal_valid` already exists in the format for exactly this.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415 (heavy, and only this path needs it)

    points = np.asarray(points_m, dtype=np.float64)
    count = len(points)
    if count < neighbours:
        return np.zeros((count, 3), dtype=np.float32), np.zeros(count, dtype=bool)
    _, index = cKDTree(points).query(points, k=neighbours)
    patches = points[index]
    centred = patches - patches.mean(axis=1, keepdims=True)
    # The smallest eigenvector of each patch's covariance. `eigh` returns them ascending.
    covariance = np.einsum("nki,nkj->nij", centred, centred) / neighbours
    values, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    # A patch that is a line or a point has no plane. The ratio rather than an absolute threshold,
    # so the check does not depend on the scene's scale.
    valid = values[:, 0] < 0.5 * values[:, 1]
    view = points / np.linalg.norm(points, axis=-1, keepdims=True).clip(1e-9)
    flip = np.einsum("ij,ij->i", normals, view) > 0.0
    normals[flip] *= -1.0
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, length, out=np.zeros_like(normals), where=length > 1e-9)
    return normals.astype(np.float32), (valid & (length[:, 0] > 1e-9))


def fit_centre_offset(cloud_m: np.ndarray, poses: np.ndarray) -> tuple[float, float]:
    """`(offset_m, residual_mm)`: how far along the approach the annotated point sits from `t`.

    Measured per import rather than hardcoded. The published estimates agree on about 0.173 m, but
    a number nobody re-derives is a number that silently stops being true when the source is
    revised. The criterion is the one that works on a single-view cloud: the annotated point is a
    camera-facing surface point, so the right offset is the one that puts it on the cloud.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    if not len(poses):
        raise ValueError("no poses to fit an offset against")
    tree = cKDTree(np.asarray(cloud_m, dtype=np.float64))
    approach = poses[:, :3, _APPROACH_COLUMN]
    approach = approach / np.linalg.norm(approach, axis=-1, keepdims=True).clip(1e-9)
    translation = poses[:, :3, 3]
    best, best_residual = _OFFSET_GRID_M[0], float("inf")
    for candidate in _OFFSET_GRID_M:
        residual = float(np.median(tree.query(translation + candidate * approach)[0])) * 1000.0
        if residual < best_residual:
            best, best_residual = candidate, residual
    return best, best_residual


def _contact_and_width(cloud_mm: np.ndarray, annotated_mm: np.ndarray, axis: np.ndarray,
                       opening_mm: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(centre_mm, contact_mm, width_mm, usable)`: where the jaws actually close.

    The stored width is a commanded opening, not a contact separation, and the contract means the
    second. The stored opening is wider than the object it holds, so a point at
    `annotated +- w/2 * axis` sits in free air by design and every contact check against it fails;
    that is why `prove_convention` refuses this corpus outright, its claim being true of the labels
    generated here and false of these.

    The annotated point is not the midpoint between the jaws either. The source's own paper says
    what it is: the 2D rectangle's centre, lifted through a monocular depth map, so it is a
    camera-facing surface point. On a slab gripped across its width that point sits on the top face,
    not halfway along the closing axis. Measuring the extent as twice the distance from it to the
    furthest observed point is right only when that point happens to be centred: on a slab 80 mm
    across it gives a width of a few millimetres whenever the annotated point lands near a face, and
    a 55 mm hand then accepts grasps on an object it cannot close around. A wrong width is worse
    than a refused grasp, because the width is a trained target.

    So both extremes along the axis are taken, the separation is their difference, and the centre is
    corrected to their midpoint. On a 2.5D cloud one side may be unobserved, and the difference then
    understates the object: a separation below `_MIN_SEPARATION_MM` is refused for that reason
    rather than believed, since it is the signature of having seen one face only, and a 4 mm width
    on a real object is not a label.
    """
    centres = np.zeros_like(annotated_mm)
    contacts = np.zeros_like(annotated_mm)
    widths = np.zeros(len(annotated_mm), dtype=np.float64)
    usable = np.zeros(len(annotated_mm), dtype=bool)
    for row in range(len(annotated_mm)):
        delta = cloud_mm - annotated_mm[row]
        along = delta @ axis[row]
        perpendicular = np.linalg.norm(delta - along[:, None] * axis[row], axis=-1)
        inside = (perpendicular <= _JAW_CORRIDOR_MM) & (np.abs(along) <= opening_mm[row] * 0.5)
        if not inside.any():
            continue
        reach = along[inside]
        low, high = float(reach.min()), float(reach.max())
        separation = high - low
        if separation < _MIN_SEPARATION_MM:
            continue
        centres[row] = annotated_mm[row] + (0.5 * (low + high)) * axis[row]
        # The contact nearer the camera, which on a single view is the one actually observed.
        near, far = (high, low) if abs(high) <= abs(low) else (low, high)
        contacts[row] = annotated_mm[row] + near * axis[row]
        widths[row] = separation
        usable[row] = True
    return centres, contacts, widths, usable


def scene_arrays(scene: str, cloud: np.ndarray, per_object: Sequence[tuple[np.ndarray, np.ndarray]],
                 *, centre_offset_m: float, gripper: str = DEFAULT_GRIPPER,
                 masks: Sequence[np.ndarray] = (), prompts: Sequence[str] = ()
                 ) -> tuple[dict[str, np.ndarray] | None, int]:
    """One imported scene in the `.npz` layout `datagen` writes, and how many grasps were refused.

    `per_object` is one `(poses, widths)` pair per object index, in index order.

    The gripper is stamped and the width is filtered against its aperture. Without the stamp the
    corpus index falls back to `2f85`, so grasps labelled for a 140 mm hand would train a model that
    believes an 85 mm hand made them: a cloud that records no gripper makes a jaw-varied corpus
    indistinguishable from the 2F-85's at training time.

    The filter has a physical reason rather than a tidiness one. `_contact_and_width` re-derives the
    separation from the cloud, so a value of 190 mm is a statement about the object, not about the
    hand: it says the thing is 190 mm across. A hand that opens 140 cannot grasp it, so that row is
    not a label the model should be asked to reproduce. The count of refusals is returned rather
    than swallowed, because it is a property of the source worth knowing.

    The stamp also feeds the gripper-conditioning seam, which `eval/gripper_differential.py`
    measures on a trained artifact.
    """
    points_m = np.asarray(cloud[:, :3], dtype=np.float64)
    points_mm = points_m * 1000.0
    normals, normal_valid = estimate_normals(points_m)

    aperture_mm = float(JAW_GEOMETRY[gripper]["aperture_mm"])
    # Per-point instance ids from the masks, and this is worth more than its 1.9 GB suggests. A
    # training unit is an object, so a scene whose points all carry instance zero collapses to one
    # unit however many things are in it, and `target_instance` sampling has nothing to choose from.
    # Instance 0 stays the background so an unmasked point is not silently attributed to an object.
    instance_id = np.zeros(len(points_mm), dtype=np.int16)
    for index, mask in enumerate(masks):
        flat = np.asarray(mask).reshape(-1).astype(bool)
        if len(flat) == len(instance_id):
            instance_id[flat] = index + 1
    positions: list[np.ndarray] = []
    approaches: list[np.ndarray] = []
    axes: list[np.ndarray] = []
    widths: list[np.ndarray] = []
    instances: list[np.ndarray] = []
    contact_points: list[np.ndarray] = []
    contact_owner: list[np.ndarray] = []
    refused = 0
    for index, (poses, opening) in enumerate(per_object):
        if not len(poses):
            continue
        approach = poses[:, :3, _APPROACH_COLUMN]
        axis = poses[:, :3, _AXIS_COLUMN]
        approach = approach / np.linalg.norm(approach, axis=-1, keepdims=True).clip(1e-9)
        axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True).clip(1e-9)
        annotated_mm = (poses[:, :3, 3] + centre_offset_m * approach) * 1000.0
        centre_mm, contact, width, usable = _contact_and_width(points_mm, annotated_mm, axis,
                                                               opening * 1000.0)
        # A separation this hand cannot open to is a statement about the object, not a label.
        too_wide = usable & (width > aperture_mm)
        refused += int(too_wide.sum())
        usable = usable & ~too_wide
        if not usable.any():
            continue
        owner = np.arange(len(positions), len(positions) + int(usable.sum()))
        positions.append(centre_mm[usable])
        approaches.append(approach[usable])
        axes.append(axis[usable])
        widths.append(width[usable])
        instances.append(np.full(int(usable.sum()), index + 1, dtype=np.int16))
        contact_points.append(contact[usable])
        contact_owner.append(owner.astype(np.int32))
    if not positions:
        return None, refused

    position_mm = np.concatenate(positions).astype(np.float32)
    grasps = len(position_mm)
    present = sorted({int(v) for chunk in instances for v in chunk})
    # One asset group per scene. There is no object identity in this source, so asset-disjoint
    # folds are unavailable and scene-disjoint is the honest maximum. It is also what the source
    # demands: grasps arrive in blocks of twenty jittered variants of one 2D rectangle, so any split
    # finer than the scene puts near-duplicates on both sides of it.
    asset = f"{SOURCE.key}/{scene}"
    return {
        # The stamp. Absent, the corpus index falls back to `2f85` in silence.
        "gripper": np.array([gripper], dtype="<U32"),
        "points_mm": points_mm.astype(np.float32),
        "normals": normals,
        "normal_valid": normal_valid,
        "instance_id": instance_id,
        "view_count": np.ones(len(points_mm), dtype=np.uint8),
        "grasp_position_mm": position_mm,
        "grasp_approach": np.concatenate(approaches).astype(np.float32),
        "grasp_axis": np.concatenate(axes).astype(np.float32),
        "grasp_width_mm": np.concatenate(widths).astype(np.float32),
        "grasp_instance": np.concatenate(instances),
        "grasp_asset_id": np.full(grasps, asset, dtype="<U80"),
        "grasp_part_role": np.full(grasps, "body", dtype="<U8"),
        # No physics verdict exists for this source. `-1` is the format's "not measured", and it
        # matters: the scorer's target is the physics referee, so an imported corpus cannot train one
        # until a shake has run over it.
        "grasp_held": np.full(grasps, -1, dtype=np.int8),
        "grasp_approach_admissible": np.ones(grasps, dtype=bool),
        "contact_points_mm": np.concatenate(contact_points).astype(np.float32),
        "contact_grasp_index": np.concatenate(contact_owner),
        # One asset id for every object in the scene, and the instances differ. The fold key is
        # the asset group, so giving each object its own id would let two objects sharing one cloud
        # land on opposite sides of a split: the held-out half would then be scored on a point cloud
        # the trained half had already seen. Identical ids keep the split scene-disjoint while the
        # differing instances still yield one training unit per object.
        "object_asset_id": np.full(max(1, len(present)), asset, dtype="<U80"),
        "object_instance": np.asarray(present or [0], dtype=np.int16),
        # The language half, which the corpus generated here cannot produce at all. One instruction
        # per object, carried through so a later text-conditioned head has something to read.
        # Nothing consumes it yet and the key is inert by design; `load_scene` passes unknown keys
        # through.
        "prompt": np.asarray(list(prompts) or [""], dtype="<U160"),
        "corpus_version": np.array([_CORPUS_VERSION], dtype=np.int32),
        "engine": np.array([_ENGINE], dtype="<U32"),
        "source_dataset": np.array([SOURCE.key], dtype="<U64"),
    }, refused


def import_scenes(out_dir: str | Path, *, limit: int = 200, report: Any = None,
                  centre_offset_m: float | None = None, cache_dir: str | Path | None = None,
                  validate: int = 3, jobs: int = 8,
                  gripper: str = DEFAULT_GRIPPER) -> dict[str, Any]:
    """Fetch `limit` scenes and write them as `.npz` files the training loop reads.

    Returns a provenance block: the source, its licence, the fitted offset and what was refused.
    """
    say = report if report is not None else (lambda _line: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # The exact index is the default. The sliced index yields fewer training units per scene than
    # the cached one, because a scene whose mask was not in the slice gets an all-zero instance map
    # and collapses to a single unit however many objects it held. The run exits 0 either way, so a
    # sliced default silently reduces the yield.
    if cache_dir is None:
        cache_dir = out / ".directories"
    clouds, grasps, masks, prompts = archives()
    triples = scene_index(clouds, grasps, masks, prompts, limit=limit, report=say,
                          cache_dir=Path(cache_dir) if cache_dir is not None else None)

    if gripper not in JAW_GEOMETRY:
        raise ValueError(f"unknown gripper {gripper!r}; choose from {', '.join(JAW_GEOMETRY)}")
    counts = {"written": 0, "no_usable_grasp": 0, "failed": 0, "skipped_present": 0,
              "refused_too_wide": 0, "with_masks": 0, "with_prompt": 0,
              "training_units": 0}
    # Parallel over scenes, on a profile rather than a guess. Measured on three real scenes: 8.30 s
    # of an 8.53 s scene is the fetch, which is 97.3 %. The CPU half is 0.23 s in total, of which
    # the contact search is 0.11, so parallelising the Python loop reaches 1.3 % of the problem.
    #
    # Each scene needs one cloud member plus one grasp and one mask member per object plus a prompt,
    # so roughly six to ten range requests, and `urlopen` opens a fresh connection for every one of
    # them. The time is handshake latency, not bandwidth, which is exactly the shape a thread pool
    # fixes and a faster machine does not.
    #
    # Eight, not thirty-two. The host is somebody else's, this is a polite default, and the
    # measurement that matters is s/scene rather than requests/s.
    fitted: list[float] = []
    residuals: list[float] = []

    def one(triple: SceneTriple) -> dict[str, Any]:
        """Fetch and convert one scene. Returns a counts delta; raises nothing the caller must catch.

        The write happens here, per scene, and that is deliberate. Collecting arrays in memory and
        writing them at the end would mean an interrupted import loses everything it fetched, and a
        229 GB source is exactly where an import gets interrupted. Each scene lands as its own file,
        and `skipped_present` makes the next run continue rather than start over.
        """
        target = out / f"{triple.scene}.npz"
        if target.exists():
            return {"skipped_present": 1}
        try:
            cloud = np.load(io.BytesIO(clouds.read(triple.cloud)), allow_pickle=False)
            per_object = [_load_grasps(grasps.read(member)) for member in triple.grasps]
            per_mask = [np.load(io.BytesIO(masks.read(m)), allow_pickle=False)
                        for m in triple.masks]
            said = _load_prompts(prompts.read(triple.prompt)) if triple.prompt else []
        except (ValueError, OSError, pickle.UnpicklingError) as exc:
            say(f"  {triple.scene[:12]}: {type(exc).__name__}: {exc}")
            return {"failed": 1}
        every_pose = np.concatenate([p for p, _ in per_object if len(p)]) if per_object else None
        if every_pose is None or not len(every_pose):
            return {"no_usable_grasp": 1}
        offset = centre_offset_m
        residual = None
        if offset is None:
            offset, residual = fit_centre_offset(cloud[:, :3], every_pose)
        arrays, refused = scene_arrays(triple.scene, cloud, per_object, centre_offset_m=offset,
                                       gripper=gripper, masks=per_mask, prompts=said)
        if arrays is None:
            return {"no_usable_grasp": 1, "refused_too_wide": refused}
        # One dict and an ignore, the same shape `write_corpus` uses: `savez_compressed`'s stub
        # types its second positional as `allow_pickle`, so a `**` of columns reads as a bool.
        np.savez_compressed(target, **arrays)  # type: ignore[arg-type]
        return {"written": 1, "refused_too_wide": refused,
                "with_masks": int(bool(per_mask)), "with_prompt": int(bool(said)),
                "training_units": len(arrays["object_instance"]),
                "_offset": offset, "_residual": residual}

    with futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for done, delta in enumerate(pool.map(one, triples), start=1):
            for key, value in delta.items():
                if key == "_offset":
                    fitted.append(value)
                elif key == "_residual":
                    if value is not None:
                        residuals.append(value)
                else:
                    counts[key] = counts.get(key, 0) + value
            if counts["written"] and counts["written"] % 50 == 0 and delta.get("written"):
                say(f"  [{done}/{len(triples)}] {counts['written']} written")
    provenance: dict[str, Any] = {
        "source": SOURCE.key, "title": SOURCE.title, "licence": SOURCE.licence, "url": SOURCE.url,
        "gripper": gripper, "aperture_mm": float(JAW_GEOMETRY[gripper]["aperture_mm"]),
        **counts,
    }
    if fitted:
        provenance["centre_offset_m"] = {"median": round(float(np.median(fitted)), 4),
                                         "min": round(float(np.min(fitted)), 4),
                                         "max": round(float(np.max(fitted)), 4)}
    if residuals:
        provenance["centre_residual_mm"] = round(float(np.median(residuals)), 2)
    # The validator reads the written files back through the corpus loader, so it checks what a
    # training run will actually see rather than what this function believes it wrote. `contract.py`
    # calls checking before training "the cheapest place to catch the fifth" frame error.
    if validate and counts["written"]:
        provenance["contract"] = _validate_written(out, limit=validate, report=say)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True),
                                         encoding="utf-8")
    return provenance


def _validate_written(out: Path, *, limit: int, report: Any) -> dict[str, Any]:
    """Read imported scenes back through the corpus loader and put them through the contract."""
    import numpy as _np  # noqa: PLC0415 (already imported above; local alias keeps the import list flat)

    from src.robot.grasping.deep.corpus.sample import (  # noqa: PLC0415
        SampleSpec, build_sample, load_scene)
    from src.robot.grasping.deep.foreign.contract import (  # noqa: PLC0415
        describe_sample, validate_sample)

    checked, failures, described = 0, [], []
    for path in sorted(out.glob("*.npz"))[:limit]:
        try:
            sample = build_sample(load_scene(path), _np.random.default_rng(0),
                                  SampleSpec(grasp_set=True, points=8192, rotate_z=False),
                                  target_instance=None)
        except (ValueError, KeyError, IndexError) as exc:
            failures.append(f"{path.stem[:12]}: {type(exc).__name__}: {exc}")
            continue
        checked += 1
        fatal = [f"{p.key}: {p.problem}" for p in validate_sample(sample) if p.fatal]
        if fatal:
            failures.append(f"{path.stem[:12]}: {'; '.join(fatal)}")
        described.append(describe_sample(sample))
    outcome = {"checked": checked, "failures": failures}
    if described:
        outcome["labels_per_point"] = round(
            float(_np.mean([d.get("labels_per_point", {}).get("mean", 0.0) for d in described])), 3)
    report(f"  contract: {checked} scene(s) checked, {len(failures)} failure(s)"
           + (f" -- {failures[0]}" if failures else ""))
    return outcome


def scene_ids(out_dir: str | Path) -> Iterator[str]:
    """The scenes already imported into `out_dir`."""
    for path in sorted(Path(out_dir).glob("*.npz")):
        yield path.stem
