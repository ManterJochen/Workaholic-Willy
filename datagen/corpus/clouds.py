"""Turning a rendered scene into what a point-cloud grasp network eats.

The tabular corpus next door (`build.py`) hands a ranker eight numbers per candidate. A generator
cannot be trained on that: it has to propose grasps from geometry nobody has told it about, so its
input is the scene, and its supervision is every grasp the analytic reference found in it.

One file per scene, and the encoding is not done here. Bin counts, seed sampling, the graspability
radius and the target-object choice are all training-time decisions that get tuned; baking them into
an extraction that takes minutes per thousand scenes would mean re-extracting for every one of them.
What is written here is geometry and labels, the two things that are properties of the scene rather
than of a model.

The cloud is fused across every rendered view, in BASE millimetres. That is the input the runtime has
(`grasping.fusion.geometry`, every fixed camera against every object): a single depth view sees one
side of an object and an antipodal grasp needs two.

The environment is in the cloud. Unprojecting only masked object pixels would discard the background,
which is most of the pixels and carries valid depth throughout: the table, the bin walls and the
posed arm. A generator that never sees a wall proposes grasps into it. Table, walls and arm all go
in, because that is what `scene_points_mm` carries at runtime and because the arm is the dominant
occluder the two-oblique rig exists to work around.

`view_count` is why the fusion is not just `np.vstack`, which throws away which camera saw what.
Points are voxelised and the count is how many distinct views put a point in a voxel: a point both
obliques saw carries 2, a point only the wrist saw carries 1. A net that cannot tell those apart
cannot learn to distrust the second.

Normals are oriented towards the cameras that saw each point. Orienting them outward from the object
centroid is exact for a closed convex object, an approximation for a cup or a bracket, and
meaningless for a table. The camera that saw a surface is in free space relative to it, so a normal
pointing at that camera points out of the object, and this is the only rule that also works for
environment points, which have no centroid worth the name.

Contacts come from the reference, not from arithmetic. `jaw_contact_patch` walks the same nine pad
positions `check_jaw_grasp` does. `position_mm +- axis * width/2` is wrong because `position_mm` is
the anchor and can miss the surface, and the pad's own two extreme contacts sit at a corner of a
27 x 38 mm patch rather than in it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover (typing only)
    # Type-only. `grasps.labels` is the heavier module and this one is imported by
    # the corpus builder; a runtime import here would put the label machinery on that
    # path for an annotation.
    from datagen.grasps.labels import SceneGeometry

import collections
import json
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

from src.robot.grasping.geometry.normals import estimate_surface_normals
from src.utility.log_cfg import create_logger

from datagen.constants import CORPUS_GATE_LOG_FILE, DATAGEN_LOG_DIR

#: Every grasp kind the labeller writes. `jaw` is the default; see `scene_grasp_table` for what a
#: mixed corpus changes and what it does not.
_GRASP_KINDS: Final[tuple[str, ...]] = ("jaw", "suction")

__all__ = ["ARM_INSTANCE", "CORPUS_VERSION", "ENVIRONMENT_INSTANCE", "build_cloud_corpus",
           "scene_cloud", "scene_grasp_table"]

logger = create_logger("CloudCorpus", CORPUS_GATE_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Instance id for everything the segmentation calls background: table, bin walls, arm. Negative so
#: it can never collide with a real instance id, and a single id rather than three because the mask
#: does not tell them apart, and inventing a distinction the data does not carry is a lie the net
#: would then learn.
#: What shape a scene `.npz` is. Bumped whenever an array is added, renamed or given a new meaning,
#: so a consumer refuses an old file instead of quietly finding an empty array where a column should
#: be. Without the stamp the symptom lands downstream: a slice extracted before `grasp_asset_id`
#: reports one asset group, and one extracted before `object_asset_id` yields no training units.
#:
#:   1  points/normals/view_count/instance_id + the grasp table
#:   2  + arm channel, contact patches, part_role, approach admissibility
#:   3  + grasp_asset_id, object_instance, object_asset_id  (asset-disjoint splits)
#:   4  + engine
#:   5  + gripper, which jaw the labels were written for
#:
#: 4 and 5 are stamps, not arrays a sample is built from, so `sample._MINIMUM_CORPUS_VERSION` stays
#: at 3 for both. Raising the floor for a key the reader does not consume would invalidate every
#: corpus already extracted, over nothing.
CORPUS_VERSION = 5

#: What the engine stamp says for a corpus built before the stamp existed. Not "isaac": a corpus whose
#: engine nobody recorded is a corpus whose engine nobody knows, and guessing the majority case is
#: exactly how a mixed corpus slips through the check that exists to catch it.
_ENGINE_UNKNOWN = "unknown"


def _engine_of(root) -> str:
    """Which engine built this dataset, from its own provenance.

    ``provenance.json`` carries ``renderer`` as ``"<engine>:<mode>"``, written by `datagen/build.py`.
    Read from the dataset rather than passed in, so the stamp describes the corpus being extracted
    and not whatever engine happens to be configured in the shell doing the extracting, which is a
    different thing and wrong exactly when it would matter.
    """
    import json as _json

    path = Path(root) / "provenance.json"
    if not path.is_file():
        return _ENGINE_UNKNOWN
    try:
        renderer = str(_json.loads(path.read_text(encoding="utf-8")).get("renderer") or "")
    except Exception:                                            # pragma: no cover (unreadable file)
        return _ENGINE_UNKNOWN
    return renderer.split(":")[0] or _ENGINE_UNKNOWN


ENVIRONMENT_INSTANCE = -1

#: Instance id for the arm. Separate from the environment because it is the one thing in a scene whose
#: geometry changes completely between scenes: filing it under "background" hands a net a class with
#: no consistent shape. The z-rotation augmentation must also leave it behind, since a rotated table
#: is still a table while a rotated arm stands where no arm on a fixed base could.
ARM_INSTANCE = -2

#: Voxel edge for the fused cloud. 3 mm resolves the 2F-85's measured 24 mm contact face across
#: eight voxels, which is the resolution a contact patch needs. Deliberately finer than the ranker
#: corpus's 6 mm obstacle grid: that one only needs a distance, this one is the geometry a net
#: proposes grasps from.
_VOXEL_MM = 3.0

#: Voxel edge for environment points: table, walls, arm. Coarser than the object grid on purpose.
#: `build_ranker_corpus` thins obstacle clouds to the same 6 mm, which moves a nearest-obstacle
#: distance by at most ~5 mm, and an obstacle only has to be dense enough to be hit.
_ENVIRONMENT_VOXEL_MM = 6.0

#: How far beyond the objects' bounding box environment points are kept. The gripper's own reach is
#: `finger_behind_mm + approach_clearance_mm` = 113.37 mm, so geometry further away than this cannot
#: be touched by any grasp on these objects. Rounded up rather than exact, because the bounding box is
#: of the observed cloud and an unobserved sliver of an object could sit just outside it.
_ENVIRONMENT_MARGIN_MM = 150.0

#: Radius for local-PCA normals. 15 mm is `NormalEstimationConfig`'s own default and about five
#: voxels: enough neighbours for a stable plane fit without smoothing an edge into a ramp.
_NORMAL_RADIUS_MM = 15.0

#: `held` for a grasp physics never tried. Not 0: an unmeasured grasp and a measured failure are
#: different facts, and collapsing the first into the second reports harness faults as bad grasps.
_HELD_UNMEASURED = -1

#: How far back along -approach the gripper must stand before it moves in. `JawModel` measures
#: `finger_behind_mm` 33.37 and `approach_clearance_mm` 80.0; the sum is where the corridor starts,
#: and `check_jaw_grasp` refuses a grasp whose start is under the support plane. Duplicated here as a
#: number rather than imported as a model, because this is a filter over labels that already exist:
#: the authority is `verdict.py`, and if the two ever disagree that file wins.
_APPROACH_START_MM = 113.37


def _view_clouds(scene_dir: Path, payload: dict, instance_ids: list[int], *, mask_suffix: str,
                 with_environment: bool) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """``(view index, instance id, points in BASE mm, that view's camera position)`` per view."""
    from PIL import Image  # noqa: PLC0415

    from datagen.render.camera import unproject_to_base  # noqa: PLC0415

    out: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for index, view in enumerate(payload["views"]):
        if str(view.get("outcome")) not in ("rendered", "rendered_after_resample"):
            continue
        depth_path = scene_dir / f"{view['name']}_depth.png"
        mask_path = scene_dir / f"{view['name']}_{mask_suffix}.png"
        if not (depth_path.exists() and mask_path.exists()):
            continue
        depth = np.asarray(Image.open(depth_path), dtype=np.float64)
        instances = np.asarray(Image.open(mask_path))
        camera_to_base = np.asarray(view["camera_to_base_mm"], dtype=np.float64)
        intrinsics = np.asarray(view["intrinsics"], dtype=np.float64)
        camera = camera_to_base[:3, 3].copy()
        wanted = [(i, instances == (i + 1)) for i in instance_ids]
        if with_environment:
            background = instances == 0
            # The arm, split out of the background. `<view>_arm.png` is rendered by the same
            # depth-difference trick the object silhouettes use (hide the robot, read again, diff),
            # so it is exact. Written only when the arm is actually in frame, so a missing file means
            # "no arm pixels", which is why absence is not an error here.
            arm_path = scene_dir / f"{view['name']}_arm.png"
            arm = (np.asarray(Image.open(arm_path)) > 0 if arm_path.exists()
                   else np.zeros(background.shape, dtype=bool))
            wanted.append((ENVIRONMENT_INSTANCE, background & ~arm))
            if arm.any():
                wanted.append((ARM_INSTANCE, arm))
        for instance_id, mask in wanted:
            points = unproject_to_base(depth, mask, camera_to_base, intrinsics)
            if points.shape[0]:
                out.append((index, instance_id, points, camera))
    return out


def _empty_cloud() -> dict[str, np.ndarray]:
    return {
        "points_mm": np.zeros((0, 3), dtype=np.float32),
        "normals": np.zeros((0, 3), dtype=np.float32),
        "normal_valid": np.zeros(0, dtype=bool),
        "view_count": np.zeros(0, dtype=np.uint8),
        "instance_id": np.zeros(0, dtype=np.int16),
    }


def scene_cloud(scene_dir: Path, payload: dict, geometry: Any, *,
                mask_suffix: str = "instances", voxel_mm: float = _VOXEL_MM,
                environment_voxel_mm: float = _ENVIRONMENT_VOXEL_MM,
                environment_margin_mm: float = _ENVIRONMENT_MARGIN_MM,
                with_environment: bool = True) -> dict[str, np.ndarray]:
    """The fused, voxelised scene cloud with per-point instance, observedness and normal."""
    instance_ids = sorted(geometry.objects)
    parts = _view_clouds(scene_dir, payload, instance_ids,
                         mask_suffix=mask_suffix, with_environment=with_environment)
    if not parts:
        return _empty_cloud()

    points = np.concatenate([p for _v, _i, p, _c in parts])
    owners = np.concatenate([np.full(len(p), i, dtype=np.int64) for _v, i, p, _c in parts])
    views = np.concatenate([np.full(len(p), v, dtype=np.int64) for v, _i, p, _c in parts])
    cameras = np.concatenate([np.tile(c, (len(p), 1)) for _v, _i, p, c in parts])

    # The environment is cropped and coarser than the objects. At object resolution the table
    # dominates a scene's point count, and most of the extraction time goes to estimating normals on
    # flat floor half a metre from any grasp.
    #
    # 6 mm is the same thinning `build_ranker_corpus` applies to obstacle clouds, which moves a
    # nearest-obstacle distance by at most ~5 mm. 150 mm of margin is the gripper's own reach,
    # `finger_behind + approach_clearance` = 113 mm, rounded up: geometry further away than that
    # cannot be touched by a grasp on these objects.
    environment = owners == ENVIRONMENT_INSTANCE
    if with_environment and environment.any() and not (owners == ARM_INSTANCE).any():
        # No arm channel in this dataset, so fall back to the kinematic capsule chain. Its recall
        # against a rendered mask is good on an outstretched arm and worst when the arm is folded,
        # because `ur_link_origins_mm` returns joint origins and the upper arm's housing sits ~180 mm
        # off the line between two of them. Applied rather than skipped, so a dataset rendered
        # without the channel still carries an arm; `scenes_with_rendered_arm_channel` reports which
        # of the two produced a scene.
        arm_cfg = payload.get("arm") or {}
        joints, model = arm_cfg.get("joints_rad"), arm_cfg.get("robot_model")
        if joints and model:
            # Imported lazily for cost, not for a cycle: `datagen.render.arm` imports only
            # `datagen.render.kinematics`, so a top-level import would close no loop. What it would
            # cost is the module count and load time noted on the import line.
            from datagen.render.arm import arm_point_mask  # noqa: PLC0415 (136 modules, 130 ms)

            try:
                hit = arm_point_mask(str(model), np.asarray(joints, dtype=np.float64),
                                     points[environment])
            except (ValueError, KeyError):
                hit = np.zeros(int(environment.sum()), dtype=bool)
            if hit.any():
                index = np.flatnonzero(environment)[hit]
                owners[index] = ARM_INSTANCE
                environment = owners == ENVIRONMENT_INSTANCE

    # Every non-object point is cropped, arm included. Cropping only the environment would leave arm
    # points surviving a margin that had removed the table beside them, and an arm further from the
    # objects than the gripper can reach is exactly as irrelevant as far table.
    non_object = owners < 0
    if with_environment and non_object.any() and (~non_object).any():
        target_points = points[~non_object]
        low = target_points.min(axis=0) - environment_margin_mm
        high = target_points.max(axis=0) + environment_margin_mm
        far = non_object & ~np.all((points >= low) & (points <= high), axis=1)
        if far.any():
            keep = ~far
            points, owners, views, cameras = points[keep], owners[keep], views[keep], cameras[keep]

    # Vectorised: a Python loop over every point costs about a second per scene, which is hours over
    # a corpus. `np.unique` on the composite key does the grouping; `bincount` does the sums.
    # Deterministic because `np.unique` sorts.
    edge = np.where(owners < 0, environment_voxel_mm, voxel_mm)[:, None]
    keys = np.column_stack([owners, np.floor(points / edge).astype(np.int64)])
    _unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    groups = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=groups).astype(np.float64)
    centroid = np.column_stack([
        np.bincount(inverse, weights=points[:, axis], minlength=groups) / counts for axis in range(3)
    ])
    instance = np.zeros(groups, dtype=np.int16)
    instance[inverse] = owners.astype(np.int16)
    # Distinct views per voxel, which is the whole reason provenance is carried: `np.unique` over the
    # (voxel, view) pairs collapses repeats, then one count per voxel.
    pairs = np.unique(np.column_stack([inverse, views]), axis=0)
    view_count = np.bincount(pairs[:, 0], minlength=groups).astype(np.uint8)
    # Where to orient each voxel's normal: the mean position of the cameras that saw it.
    observer = np.column_stack([
        np.bincount(inverse, weights=cameras[:, axis], minlength=groups) / counts for axis in range(3)
    ])

    normals = np.zeros(centroid.shape, dtype=np.float32)
    valid = np.zeros(groups, dtype=bool)
    for instance_id in np.unique(instance):
        mask = instance == instance_id
        subset = centroid[mask]
        if len(subset) < 3:
            continue
        # `orient_towards_camera=False`, then flipped per point: the estimator takes one camera for a
        # whole call, and the whole point here is that different parts of a fused cloud were seen by
        # different cameras.
        estimated = estimate_surface_normals(subset, radius_mm=_NORMAL_RADIUS_MM,
                                             orient_towards_camera=False)
        raw = np.asarray(estimated.normals, dtype=np.float64)
        towards = observer[mask] - subset
        flip = np.einsum("ij,ij->i", raw, towards) < 0.0
        raw[flip] *= -1.0
        normals[mask] = raw.astype(np.float32)
        valid[mask] = np.asarray(estimated.valid_mask, dtype=bool)

    return {
        "points_mm": centroid.astype(np.float32),
        "normals": normals,
        "normal_valid": valid,
        "view_count": view_count,
        "instance_id": instance,
    }


def _empty_grasp_table() -> dict[str, np.ndarray]:
    empty3 = np.zeros((0, 3), dtype=np.float32)
    return {
        "grasp_position_mm": empty3, "grasp_approach": empty3, "grasp_axis": empty3,
        "grasp_width_mm": np.zeros(0, dtype=np.float32),
        "grasp_instance": np.zeros(0, dtype=np.int16),
        "grasp_held": np.zeros(0, dtype=np.int8),
        "grasp_approach_admissible": np.zeros(0, dtype=bool),
        "grasp_part_role": np.zeros(0, dtype="<U8"),
        "grasp_asset_id": np.zeros(0, dtype="<U80"),
        "object_instance": np.zeros(0, dtype=np.int16),
        "object_asset_id": np.zeros(0, dtype="<U80"),
        "contact_points_mm": empty3,
        "contact_grasp_index": np.zeros(0, dtype=np.int32),
        **_empty_suction_table(),
    }


def _empty_suction_table() -> dict[str, np.ndarray]:
    return {
        "suction_position_mm": np.zeros((0, 3), dtype=np.float32),
        "suction_approach": np.zeros((0, 3), dtype=np.float32),
        "suction_instance": np.zeros(0, dtype=np.int16),
        "suction_cup": np.zeros(0, dtype="<U16"),
    }


def _suction_table(rows: list[dict]) -> dict[str, np.ndarray]:
    """Suction labels as their own arrays, because a suction grasp is not a jaw grasp.

    A separate table rather than a `kind` column inside the jaw arrays. A suction row carries
    `closing_axis: [0, 0, 0]` and `width_mm: 0.0`, because a cup has neither, and the sample contract
    refuses exactly those two shapes under the names "zero-length direction" and "non-positive
    opening". Mixed into the jaw arrays they would produce samples that fail that validator, and any
    consumer that forgot to filter on the column would train a jaw head on a zero axis.

    A suction grasp is a position, an approach and a cup. Its own arrays mean `grasp_*` keeps meaning
    exactly what every existing reader believes it means, and a corpus built with both kinds is
    byte-identical in the jaw half.

    Worth carrying because far more objects earn a suction label than a jaw one, so a jaw-only corpus
    calls objects empty that a cell with a cup can pick.
    """
    if not rows:
        return _empty_suction_table()
    return {
        "suction_position_mm": np.asarray([r["position_mm"] for r in rows], dtype=np.float32),
        "suction_approach": np.asarray([r["approach"] for r in rows], dtype=np.float32),
        "suction_instance": np.asarray([r["instance_id"] for r in rows], dtype=np.int16),
        "suction_cup": np.asarray([str(r.get("cup", "")) for r in rows], dtype="<U16"),
    }


def scene_grasp_table(rows: list[dict], geometry: Any = None,
                      held: dict[tuple[str, int], bool] | None = None,
                      assets: dict[int, str] | None = None,
                      kinds: Sequence[str] = ("jaw",)) -> dict[str, np.ndarray]:
    """Every label of the requested kinds on one scene, with physics, part role and contacts.

    ``held`` is keyed ``(origin, row_index)`` exactly as the shake writes it, so a verdict reaches
    its label by file and line. A pose join is ambiguous by construction, because a label row and the
    `valid` row describing the same grasp carry identical poses.

    That key is what lets the shake's `label` trials land at all: they come from `grasps.jsonl`,
    which `build_ranker_corpus` never reads.

    ``kinds`` defaults to jaw alone, and what that discards is not small. Far more objects earn a
    suction label than a jaw one, for a small share of the total label count, so any count of objects
    with no label at all is a count over jaw labels unless suction is asked for.

    A mixed corpus is not a fix for the pose heads and must not be read as one. A jaw head cannot
    learn a pose from a suction label; they are different grasps. What it changes is the where stage,
    which learns from jaw labels alone and is therefore taught to call a suction-only object empty,
    which is false for a cell that carries both end effectors. It is also what a learned suction
    generator would need, there being no data path to one today.

    Suction labels go into their own `suction_*` arrays, so a consumer selects a kind by array name
    rather than by trusting the directory a file came from.
    """
    held = held or {}
    wanted = tuple(kinds)
    for kind in wanted:
        if kind not in _GRASP_KINDS:
            raise ValueError(f"unknown grasp kind {kind!r}; choose from {', '.join(_GRASP_KINDS)}")
    suction = _suction_table([r for r in rows if r.get("kind") == "suction"]
                             if "suction" in wanted else [])
    keep = [r for r in rows if r.get("kind") == "jaw"] if "jaw" in wanted else []
    if not keep:
        empty = _empty_grasp_table()
        empty.update(suction)
        # A scene with no jaw label still has objects, and they still have assets. Returning an empty
        # object table here would make those scenes ungroupable and silently unusable.
        empty["object_instance"] = np.asarray(sorted(assets or {}), dtype=np.int16)
        empty["object_asset_id"] = np.asarray(
            [str((assets or {})[i]) for i in sorted(assets or {})], dtype="<U80")
        return empty

    verdicts = []
    for row in keep:
        key = ("grasps", int(row["_row_index"]))
        verdicts.append(int(held[key]) if key in held else _HELD_UNMEASURED)
    approach = np.asarray([r["approach"] for r in keep], dtype=np.float64)
    position = np.asarray([r["position_mm"] for r in keep], dtype=np.float64)
    axis = np.asarray([r["closing_axis"] for r in keep], dtype=np.float64)

    contacts: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    if geometry is not None:
        from datagen.grasps.verdict import JawGrasp, jaw_contact_patch  # noqa: PLC0415

        for index, row in enumerate(keep):
            target = geometry.objects.get(int(row["instance_id"]))
            if target is None:
                continue
            patch = jaw_contact_patch(
                JawGrasp(position[index], approach[index], axis[index], float(row["width_mm"])),
                target)
            if len(patch):
                contacts.append(patch)
                owners.append(np.full(len(patch), index, dtype=np.int32))

    return {
        **suction,
        "grasp_position_mm": position.astype(np.float32),
        "grasp_approach": approach.astype(np.float32),
        "grasp_axis": axis.astype(np.float32),
        "grasp_width_mm": np.asarray([r["width_mm"] for r in keep], dtype=np.float32),
        "grasp_instance": np.asarray([r["instance_id"] for r in keep], dtype=np.int16),
        "grasp_held": np.asarray(verdicts, dtype=np.int8),
        # Would the gripper have to start under the table? Redundant for a corpus labelled since
        # `check_jaw_grasp` gained the corridor test, and carried anyway: an older label file still
        # holds grasps reached from underneath, and it cannot be re-labelled without shifting every
        # row index the shake's verdicts join by, which is (file, line). Training on those would
        # teach a generator to propose them.
        "grasp_approach_admissible": (position[:, 2] - approach[:, 2] * _APPROACH_START_MM > 0.0),
        # `handle`, `body`, `grip`, `neck`, `head`, or "" for a single-primitive object. The only
        # supervision an affordance head can have, and it is already in the label record.
        "grasp_part_role": np.asarray([str(r.get("part_role", "")) for r in keep], dtype="<U8"),
        # Which asset this grasp is on, and the only honest key for an object-disjoint split.
        # Grouping by `scene#instance`, which is what the tabular corpus uses, puts the same mesh in
        # train and test the moment it appears in two scenes, and a generalisation number measured
        # that way is measuring memorisation.
        #
        # Real meshes carry a globally unique name (`gso_Android_Lego`, `ycb_044_flat_screwdriver`).
        # Procedural and composite ids carry a per-scene index instead (`proc_00002_pouch`), so the
        # same name is different geometry in a different scene. That is why the grouping rule is left
        # to the trainer and only the fact is stored here.
        "grasp_asset_id": np.asarray(
            [str((assets or {}).get(int(r["instance_id"]), "")) for r in keep], dtype="<U80"),
        # Every object's asset, not only the ones a grasp landed on. The sampler needs it for the
        # object it conditions on, and an object with no labelled grasp still has to be groupable:
        # otherwise the only objects that can be held out are the ones that were already easy.
        "object_instance": np.asarray(sorted(assets or {}), dtype=np.int16),
        "object_asset_id": np.asarray(
            [str((assets or {})[i]) for i in sorted(assets or {})], dtype="<U80"),
        "contact_points_mm": (np.concatenate(contacts).astype(np.float32) if contacts
                              else np.zeros((0, 3), dtype=np.float32)),
        "contact_grasp_index": (np.concatenate(owners) if owners
                                else np.zeros(0, dtype=np.int32)),
    }


def physics_verdicts(path: Path | None) -> dict[tuple[str, int], bool]:
    """``(origin, row_index) -> held``, refusals excluded because a refusal is not a failure.

    Re-running refused trials in a fresh session leaves most of them refused, and few of the ones
    that do resolve hold. A refusal is a property of the grasp rather than of the moment, so folding
    one into `held=False` would record an unmeasured mostly-failure as a measured failure.
    """
    if path is None or not Path(path).exists():
        return {}
    out: dict[tuple[str, int], bool] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if "origin" not in row or str(row.get("note", "")).startswith("refused"):
            continue
        out[(str(row["origin"]), int(row["row_index"]))] = bool(row["held"])
    return out



def _gripper_of(root: Path, labels: str) -> str:
    """Which gripper a label file was written for, from its name, cross-checked against its report.

    The name is authoritative because `label_dataset` derives it from the same string: a file called
    `grasps_jaw_wide_140.jsonl` can only have come from `--jaw wide_140`. The report is read anyway
    and a disagreement refuses, because the one failure worth paying for here is a corpus stamped
    with a gripper it was not labelled for. A constant conditioning input teaches a head nothing; a
    wrong one teaches it a relationship that does not exist.
    """
    stem = Path(labels).stem
    named = stem[len("grasps_jaw_"):] if stem.startswith("grasps_jaw_") else ""
    report = (root / ("grasp_label_report.json" if not named
                      else f"grasp_label_report_jaw_{named}.json"))
    expected = named or "2f85"
    if report.is_file():
        stamped = str(json.loads(report.read_text(encoding="utf-8")).get("jaw_model", expected))
        if stamped != expected:
            raise ValueError(
                f"{labels} names gripper {expected!r} but {report.name} says {stamped!r}; refusing "
                f"to stamp a corpus with a gripper it may not have been labelled for")
    return expected


def _refuse_foreign_overwrite(target: Path, dataset: str) -> None:
    """Refuse to write over a scene that came from a different dataset.

    Re-extracting the same dataset is a normal thing to do and overwrites freely. Writing over
    another dataset's scene of the same name is data loss, and it is invisible: the directory simply
    ends up with fewer files than the runs that wrote into it reported between them.
    """
    if not target.exists():
        return
    try:
        with np.load(target, allow_pickle=False) as handle:
            previous = str(handle["source_dataset"][0]) if "source_dataset" in handle.files else ""
    except Exception:      # noqa: BLE001 (an unreadable file is not a reason to refuse a rewrite)
        return
    if previous and previous != dataset:
        raise ValueError(
            f"{target.name} already holds a scene from dataset {previous!r} and this run is "
            f"{dataset!r}. Scene ids repeat across shards, so writing here would DESTROY that scene. "
            f"Give each dataset its own --corpus-out directory.")


def build_cloud_corpus(root: str | Path, out_dir: str | Path, *, scenes: int | None = None,
                       mask_suffix: str = "instances", physics: str | Path | None = None,
                       voxel_mm: float = _VOXEL_MM,
                       environment_voxel_mm: float = _ENVIRONMENT_VOXEL_MM,
                       environment_margin_mm: float = _ENVIRONMENT_MARGIN_MM,
                       with_environment: bool = True,
                       kinds: Sequence[str] = ("jaw",),
                       labels: str = "grasps.jsonl") -> dict[str, Any]:
    """Write one ``.npz`` per scene, fused cloud plus grasp table, and return a report.

    ``labels`` names the grasp file inside ``root``. It is not `grasps.jsonl` when the dataset was
    labelled for a different gripper, which `label-grasps --jaw` writes as `grasps_jaw_<name>.jsonl`.

    A non-default label file changes the dataset identity, and it has to. Two extractions of the same
    scenes under two grippers produce different grasp tables under the same scene names, so a shared
    output directory would let the second silently replace the first. `_refuse_foreign_overwrite`
    guards that case by comparing the stamped dataset name, which without this would be identical for
    both.
    """
    from datagen.grasps.labels import scene_assets  # noqa: PLC0415 (heavy import chain)

    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verdicts = physics_verdicts(Path(physics) if physics else None)

    by_scene: dict[str, list[dict]] = collections.defaultdict(list)
    # The identity the overwrite guard compares. A gripper variant is a different dataset as far as
    # the extracted clouds are concerned, even though the scenes are the same files.
    identity = root.name if labels == "grasps.jsonl" else f"{root.name}::{Path(labels).stem}"
    gripper = _gripper_of(root, labels)
    label_path = root / labels
    # A dataset with no labels is a legitimate one (a freshly rendered corpus, or scenes extracted
    # for inference) and the clouds are exactly as valid without them. Refusing here would make the
    # extractor useless for the case it will be used in most often once a net exists.
    if label_path.exists():
        for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            if not line:
                continue
            row = json.loads(line)
            row["_row_index"] = index           # the join key, the convention the shake writes
            by_scene[str(row["scene_id"])].append(row)
    else:
        logger.warning("no %s under %s: writing clouds with EMPTY grasp tables", labels, root)

    assets = scene_assets(root)
    scene_dirs = [p for p in sorted((root / "scenes").iterdir()) if (p / "scene.json").exists()]
    if scenes is not None and scenes < len(scene_dirs):
        # An even stride, not a prefix. Scene directories sort by family name, so a prefix returns
        # one family and nothing else, and the families differ sharply in jaw labels per object. A
        # slice taken that way reports a property of the alphabet rather than of the data.
        step = len(scene_dirs) / scenes
        scene_dirs = [scene_dirs[int(i * step)] for i in range(scenes)]

    written = points_total = env_total = grasps_total = held_known = contacts_total = 0
    arm_total = arm_channel = 0
    empty: list[str] = []
    for scene_dir in scene_dirs:
        payload = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
        geometry = assets.geometry(payload, scene_dir.name)
        cloud = scene_cloud(scene_dir, payload, geometry, mask_suffix=mask_suffix,
                            voxel_mm=voxel_mm, environment_voxel_mm=environment_voxel_mm,
                            environment_margin_mm=environment_margin_mm,
                            with_environment=with_environment)
        if not len(cloud["points_mm"]):
            # Named, not silently skipped. A scene that renders but yields no cloud has the shape of
            # the blank-RGB defect: a scene whose frames are all black passes every check as `ok`.
            empty.append(scene_dir.name)
            continue
        spec_objects = (payload.get("spec") or {}).get("objects") or []
        assets_by_instance = {index: str(entry.get("asset_id") or "")
                              for index, entry in enumerate(spec_objects)}
        table = scene_grasp_table(by_scene.get(scene_dir.name, []), geometry, verdicts,
                                  assets_by_instance, kinds=kinds)
        # One dict and an ignore, the same shape `write_corpus` uses: `savez_compressed`'s stub types
        # its second positional as `allow_pickle`, so every array beyond the path trips the checker.
        engine = _engine_of(root)
        version = {"corpus_version": np.asarray([CORPUS_VERSION], dtype=np.int32),
                   # Which dataset this scene came from. A scene id is `<family>_<index>` where the
                   # index counts within one dataset, so two shards of the same corpus produce
                   # different scenes with identical names. Extracting both into one directory would
                   # overwrite silently; the stamp is what makes that loud instead.
                   "source_dataset": np.asarray([identity], dtype="<U64"),
                   # Which gripper these labels are for. Without it a corpus labelled for a
                   # 140 mm jaw is indistinguishable from the 2F-85's at training time, and the
                   # conditioning input would be fed from a guess. A wrong gripper vector is worse
                   # than a constant one: it teaches the head a relationship that is not there.
                   "gripper": np.asarray([gripper], dtype="<U32"),
                     # Which engine made it. Without this stamp `renderer` reaches
                     # `provenance.json` and stops there, so two corpora built by different engines
                     # are indistinguishable at file level, and a reader that pooled them would
                     # confound engine with scene family in every number drawn from the mix. Two
                     # engines never produce byte-identical settles; a corpus that cannot say which
                     # one made it is a corpus nobody can compare.
                     "engine": np.asarray([engine], dtype="<U32")}
        target = out_dir / f"{scene_dir.name}.npz"
        _refuse_foreign_overwrite(target, identity)
        np.savez_compressed(target, **{**cloud, **table, **version})  # type: ignore[arg-type]
        written += 1
        points_total += len(cloud["points_mm"])
        env_total += int((cloud["instance_id"] == ENVIRONMENT_INSTANCE).sum())
        arm_total += int((cloud["instance_id"] == ARM_INSTANCE).sum())
        # Any view, not the first: the first is the wrist camera, which sits on the arm and therefore
        # never sees it. Keyed on `views[0]` this counter reads 0 for a dataset that has the channel.
        arm_channel += int(any((scene_dir / f"{v['name']}_arm.png").exists()
                               for v in payload.get("views") or ()))
        grasps_total += len(table["grasp_width_mm"])
        held_known += int((table["grasp_held"] >= 0).sum())
        contacts_total += len(table["contact_points_mm"])

    report: dict[str, Any] = {
        "scenes_in": len(scene_dirs), "scenes_written": written,
        "scenes_with_no_cloud": empty[:20], "n_scenes_with_no_cloud": len(empty),
        "points": points_total, "environment_points": env_total, "arm_points": arm_total,
        # How the arm was found, per scene. The rendered channel is exact and the capsule fallback is
        # an approximation, so a corpus that mixed the two silently would be two datasets.
        "scenes_with_rendered_arm_channel": arm_channel,
        "grasps": grasps_total, "grasps_with_physics": held_known, "contacts": contacts_total,
        "voxel_mm": voxel_mm, "environment_voxel_mm": environment_voxel_mm,
        "environment_margin_mm": environment_margin_mm,
        "mask_suffix": mask_suffix, "with_environment": with_environment,
    }
    logger.info(
        "cloud corpus: %d/%d scene(s), %d point(s) (%d environment), %d grasp(s), "
        "%d with physics, %d contact(s)",
        written, len(scene_dirs), points_total, env_total, grasps_total, held_known, contacts_total)
    return report

# Public rather than private: `corpus/build.py` imports it across a package boundary, and a leading
# underscore there would claim that nothing outside uses it, which no export sweep can check and a
# rename would then break.
#
# It belongs in this module on behaviour rather than on tidiness: this module already reads a
# rendered scene's depth and masks and unprojects them with `unproject_to_base`, which is the whole
# of what this function does.


def fuse_instance_clouds(scene_dir: Path, payload: dict, geometry: "SceneGeometry",
                       *, mask_suffix: str = "instances") -> dict:
    """Every object's surface as seen from all rendered views at once, in BASE mm.

    The rig is a wrist view and two opposed obliques, so an object's near and far faces are between
    them. That is the whole point of fusing them: a top-down antipodal pair needs two opposing
    surfaces, and one depth image only ever contains one of them.
    """
    from PIL import Image  # noqa: PLC0415

    from datagen.render.camera import unproject_to_base  # noqa: PLC0415

    clouds: dict[int, list[np.ndarray]] = {}
    for view in payload["views"]:
        if str(view.get("outcome")) not in ("rendered", "rendered_after_resample"):
            continue
        depth = np.asarray(Image.open(scene_dir / f"{view['name']}_depth.png"), dtype=np.float64)
        instances = np.asarray(Image.open(scene_dir / f"{view['name']}_{mask_suffix}.png"))
        camera_to_base = np.asarray(view["camera_to_base_mm"], dtype=np.float64)
        intrinsics = np.asarray(view["intrinsics"], dtype=np.float64)
        for instance_id in geometry.objects:
            points = unproject_to_base(depth, instances == (instance_id + 1),
                                       camera_to_base, intrinsics)
            if points.shape[0]:
                clouds.setdefault(instance_id, []).append(points)
    return {instance_id: np.vstack(parts) for instance_id, parts in clouds.items()}
