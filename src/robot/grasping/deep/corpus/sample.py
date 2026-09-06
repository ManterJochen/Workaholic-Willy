"""One rendered scene turned into one training sample for the grasp generator.

The scene files this reads are written by `datagen.corpus.clouds`: geometry and labels, with none of
the choices below baked in. Everything here is a training-time decision, which is why it lives on
this side of the seam: bin counts, sampling budgets and the graspability radius are all tuned, and
re-extracting a corpus for each of them costs an afternoon apiece.

The channels are the ones a cell can actually provide. `BinPickingOrchestrator` hands its calculator
`geometry_points_base_mm` (this object's fused cloud), `scene_points_mm` (the neighbours') and
`rigid_obstacle_points_mm` (declared walls), so `is_target` and `is_object` are available at serve
time and go in. `is_arm` does not: nothing in the perception stack segments the arm, so a net that
leaned on that channel would be leaning on something that exists only in training. The corpus
carries the arm anyway, because it is needed to leave the arm out of a rotation and it is worth
being able to measure, but it never reaches the net as a channel.

Owner decisions:

  * the net eats the full perceived cloud, table included, and the runtime grows a path to supply
    one. Training on what the runtime has today would mean a generator that has never seen a table,
    and the support plane is analytic rather than a channel, so "this grasp bottoms out" would be
    unlearnable from geometry, which is the one thing a point-cloud model is for.
  * walls are perceived on both sides. A real cell sees its bin wall with gaps and noise; the
    declared, gapless `rigid_obstacle_points_mm` stays with the geometric calculator, where it
    measured a gain in precision.
  * a sample is a (scene, target object), and the loss is masked to the target plus the environment.
    Scenes share assets, and treating a scene as the unit chains nearly every asset group into one
    connected component, so an asset-disjoint split at scene level is impossible and every
    generalisation number would be part memorisation. Conditioning on one object makes the group
    unambiguous. The cost is real and is not hidden: bin-clearing is never trained, and leaving the
    mask empty at inference is train/serve skew for exactly the mode the investor demo runs.
  * z-rotation augmentation, with the arm dropped first: a rotated table is still a table, a rotated
    arm stands where no arm on a fixed base could.

Everything inside is metres, per section 4 of the arc plan: millimetres at the seam, metres in the
net. The corpus is millimetres, so this is where the conversion happens, once.

Heights are above the support plane, not absolute z. The table sits at z=0 in every datagen scene,
which is exactly what makes absolute z a trap: a net trained on it would learn "z < 5 mm is table"
and a cell with `robot.grasping.support.height_mm` set to anything else would break it silently. x
and y are centred per sample, because an object at x=300 poses the same problem as the same object
at x=600.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.robot.grasping.deep.corpus.grasp_encoding import (
    APPROACH_BINS,
    ROTATION_BINS,
    decode_grasp,
    encode_grasp,
    graspability_field,
)

__all__ = [
    "FEATURE_NAMES",
    "NO_TARGET",
    "SampleSpec",
    "build_sample",
    "load_scene",
]

_MM_TO_M: Final[float] = 1e-3

#: Instance ids the corpus uses for things that are not objects. Mirrored rather than imported:
#: `src` may never import `datagen` (a generator that becomes a runtime dependency of a robot is one
#: nobody can delete). The two definitions must agree.
_ENVIRONMENT_INSTANCE: Final[int] = -1
_ARM_INSTANCE: Final[int] = -2

#: The oldest corpus this reader will accept. Mirrored from `datagen.corpus.clouds.CORPUS_VERSION`
#: for the same reason the instance ids are, that `src` may never import `datagen`, and it must
#: agree with it.
_MINIMUM_CORPUS_VERSION: Final[int] = 3

#: Deliberately not raised to 4, and not to 5 either. Version 4 adds the `engine` stamp and 5 adds
#: `gripper`; every array a sample is built from is unchanged, so a v3 corpus still reads exactly
#: right and stays usable, as do v4 and v5 clouds. Raising the floor would invalidate them over
#: keys the reader does not consume.
#:
#: A consumer that wants the gripper reads the key and defaults, rather than gating on the version.
#: The default is safe only because every corpus written before the stamp was labelled with the
#: 2F-85, which is a fact about this repository's history and not a property of the format.

#: What `target_instance` means when a sample is not target-conditioned: bin-clearing, where any
#: object will do. A sentinel rather than `None` so the value can live in an array beside real ids.
NO_TARGET: Final[int] = -1

#: The per-point input channels, in order, after the three coordinates. Named because a silent
#: reordering between the corpus builder and the net is the kind of defect that trains perfectly.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "normal_x", "normal_y", "normal_z",
    #: This point belongs to the object the caller asked for.
    "is_target",
    #: This point belongs to some segmented object, as opposed to table, wall or arm, which the
    #: runtime cannot tell apart either.
    "is_object",
)

#: Observedness is not a channel, and section 4 asked for one. Its apparent signal is confounded by
#: the surface normal. Raw, graspability falls as view_count rises, but view_count correlates with
#: `normal_z`, and points every camera sees are the ones facing up. Split by normal, the normal
#: separates the two groups by a wide margin while view_count's marginal effect inside a band is
#: small and reverses direction.
#:
#: And the runtime cannot supply it honestly: `FusedSceneGeometry` carries `views_used` as a list of
#: camera names, not a count per point. A channel the net learnt a gradient on and the cell sets to
#: a constant is worse than no channel at all. The corpus still stores `view_count`, because it is a
#: fact about the scene, but it never reaches the net.
#:
#: The ranker holds the same rule: `features._REMOVED_NOT_COMPUTABLE_AT_RUNTIME` holds
#: `target_observedness`, dropped because the renderer's visibility label needs ground-truth
#: unoccluded pixels that no cell has.

#: Where a labelled contact stops counting as "this bit of surface is graspable". 10 mm is about
#: three voxels of the 3 mm object grid and half a 2F-85 pad.
_GRASPABILITY_RADIUS_MM: Final[float] = 10.0

#: A per-point target with no grasp. Integers get this, floats get NaN, and both are masks rather
#: than values: a loss that averaged over -1 approach bins would be training on nothing, loudly.
NO_GRASP: Final[int] = -1

#: Part role to class index, in the fixed order `SlotHeadConfig.part_roles` declares. An
#: unrecognised role maps to 0 ("") rather than raising: a corpus written with a role this net does
#: not model must degrade to "no part", never abort a run.
_PART_ROLE_INDEX: Final[dict[str, int]] = {
    "": 0, "body": 1, "handle": 2, "grip": 3, "neck": 4, "head": 5,
}

#: The widest a 2F-85 opens. A copy of `datagen.assets.procedural.MAX_JAW_APERTURE_MM`, because
#: `src` may never import `datagen`, and the two must agree. It is here for one reason: scale jitter
#: may not invent a grasp the gripper cannot close on.
_MAX_JAW_APERTURE_MM: Final[float] = 85.0


@dataclass(frozen=True, slots=True)
class SampleSpec:
    """Every training-time knob, in one place that can be written into a model card."""

    #: Emit the full offset from the seed to the grasp centre, not just its approach component.
    #: Default False, which is every corpus and model that exists.
    #:
    #: A position target of one scalar along the approach discards two of the three degrees of
    #: freedom at encode time. The encode-decode round trip loses the two components perpendicular
    #: to the approach, which is where the whole residual lands. No head and no decode-time repair
    #: recovers it.
    #:
    #: It changes the sample, so a corpus built with it off and a model trained expecting it on do
    #: not fit together. The artifact carries this whole `SampleSpec` under its `sample` key, so the
    #: two travel together.
    predict_offset: bool = False

    #: Emit the set of grasps reachable from each point, not one per point. Default False, which is
    #: every corpus and model that exists.
    #:
    #: `_assign_grasps` keeps the nearest contact per point, so a point that admits several
    #: different grasps gets one of them, and a head regressed against that lands between the modes
    #: on the approach, the offset and the width alike. A supervised point commonly admits more
    #: than one approach within the 20 mm first level.
    #:
    #: It decides an experiment, not only a model. Coverage is one of the four metrics, and a head
    #: supervised against one label per point can never report more than one hit, so the
    #: deterministic arm would lose to the generative one because of its supervision rather than
    #: its merits.
    #:
    #: What is emitted is sparse: the scene's grasp table plus a `(point, grasp)` pair list, never a
    #: padded per-point block. Most points reach no grasp at all, so padding 8,192 rows would be
    #: mostly zeros, and the head runs at a few dozen seeds rather than at every point.
    grasp_set: bool = False

    #: How many points reach the net. A stratified 8,192 recovers nearly as many labelled contacts
    #: as the whole cloud, at a fraction of the points. Uniform sampling at the same budget recovers
    #: fewer and keeps a much smaller share of the object points, so the stratification is what
    #: makes the budget affordable, not the budget itself.
    points: int = 8192

    #: What share of the budget goes to the target, and what share to the other objects. The rest is
    #: environment. Three ways rather than two, because the unit is a (scene, target): with a single
    #: "objects" share the target gets only its 1/N of it, and the positive signal drowns in true
    #: negatives.
    #:
    #: A third each, because the target is a small part of the cloud beside the other objects and
    #: the environment. Reserving a third means most units are limited by the object rather than by
    #: the budget, i.e. the target usually arrives whole.
    target_fraction: float = 1.0 / 3.0
    other_object_fraction: float = 1.0 / 3.0

    #: How often a sample names a target. 1.0, because an untargeted sample has to supervise every
    #: object in the scene, and those objects belong to assets that may be held out, which is the
    #: leak the whole (scene, target) unit exists to close. Lowering this reintroduces that leak,
    #: and nothing in the training loop measures it.
    target_probability: float = 1.0

    #: Rotate the scene about gravity. The arm is dropped when this fires; see the module docstring.
    rotate_z: bool = True

    #: Resize the whole scene by a factor drawn log-uniformly from this range. `(1.0, 1.0)` is off,
    #: and off is byte-identical: the draw is not taken at all, so the caller's generator is not
    #: advanced either.
    #:
    #: Within one object the grasp width barely varies, while across a corpus it varies widely.
    #: Width is very nearly an object property, which is exactly what a net can memorise per asset
    #: instead of reading, and it does: the width error on assets it has seen is far below the error
    #: on assets it has not, which is in turn worse than predicting a constant. Resizing the scene
    #: makes the shape and the answer independent, so the only way left to know the width is to
    #: measure the object in front of it.
    #:
    #: Physically honest, and that costs a clamp: a scaled grasp keeps its direction but changes its
    #: width, so the upper end is capped per scene at `_MAX_JAW_APERTURE_MM / widest label`. Without
    #: it, scaling up would mint labels the gripper cannot close on and teach the net to propose
    #: them, the same failure the `grasp_approach_admissible` filter above exists to prevent.
    scale_jitter: tuple[float, float] = (1.0, 1.0)

    #: Height of the support plane in the corpus's own millimetres. Subtracted from z so the net sees
    #: height above the table. Zero for every datagen scene, and a knob because a cell's is not.
    support_height_mm: float = 0.0

    graspability_radius_mm: float = _GRASPABILITY_RADIUS_MM
    approach_bins: int = APPROACH_BINS
    rotation_bins: int = ROTATION_BINS
    #: Treat an object point with no labelled grasp as unknown rather than as a true negative.
    #:
    #: Most objects carry no labelled grasp at all, and the decisive control is that the same assets
    #: are labelled far more often in `sparse` scenes than in `bin` scenes. A tennis ball is
    #: graspable on a table and out of reach at the bottom of a bin. So "no labelled grasp" is
    #: overwhelmingly a property of the scene, not of the surface, and training those points as
    #: negatives teaches the net that a surface is ungraspable when the truth is that nobody looked
    #: there.
    #:
    #: The rule is the one the published lineage uses: points that are not on an object are
    #: negative, object points near a labelled grasp are positive, and object points with no
    #: labelled grasp are ignored. The table is genuinely not graspable; an unlabelled patch of
    #: object is unknown.
    #:
    #: It is not obviously an improvement, which is why it is a flag and not a change. The
    #: annotation here is far sparser than the one that rule was written for: most objects carry no
    #: labelled grasp and a labelled one carries only a handful. Ignoring every unlabelled object
    #: point may remove nearly all of the object-side supervision and leave the net learning only
    #: "object versus table". Measure both arms before believing either.
    ignore_unlabelled_object_points: bool = False

    #: Emit `approach_set`, a multi-hot over every approach bin a labelled grasp reaches this point
    #: from, not only the nearest one's.
    #:
    #: A seed point carries several valid grasps: two supervised points a few millimetres apart can
    #: carry approach labels further apart than the bin grid's own resolution. The label at a point
    #: is genuinely multi-modal, and cross-entropy against the single nearest grasp demands an
    #: answer the input cannot determine. Under the K-slot head `oracle_ceiling` shows that a
    #: perfect head is not capped in top-1 at all; what multimodality caps is coverage. A multi-hot
    #: target asks "which approaches work here", which is both the question a cell has and one the
    #: geometry can answer.
    #:
    #: Default off and byte-identical: the extra array is simply absent, and every consumer branches
    #: on its presence. Both targets are meant to be run on the same folds and compared, not
    #: swapped.
    approach_multilabel: bool = False


def load_scene(path: str | Path) -> dict[str, np.ndarray]:
    """Read one scene ``.npz`` into memory, eagerly.

    `np.load` on an `.npz` returns a lazy handle, and every `d[key]` re-decompresses that whole
    column: a comprehension over a corpus does it once per key per scene and turns a read measured
    in seconds into one measured in hours. A full eager read is about 3.8 ms per scene.
    """
    with np.load(Path(path), allow_pickle=False) as handle:
        scene = {name: handle[name] for name in handle.files}
    # `.reshape(-1)[0]` rather than `[0]`: a writer that stamps the version as a scalar is the
    # natural thing to write and would otherwise raise `IndexError: too many indices` here, an
    # opaque crash instead of the loud, actionable message four lines down.
    stamped = scene.get("corpus_version")
    version = int(np.asarray(stamped).reshape(-1)[0]) if stamped is not None else 0
    if version < _MINIMUM_CORPUS_VERSION:
        # Loudly, because the quiet version of this failure is invisible: a slice extracted before
        # `object_asset_id` existed produces zero training units, and a slice before
        # `grasp_asset_id` produces exactly one asset group.
        raise ValueError(
            f"{Path(path).name} is corpus version {version}, this reader needs "
            f"{_MINIMUM_CORPUS_VERSION}. Re-extract it with a current `datagen.corpus.clouds`; the "
            f"arrays it is missing would otherwise read as empty and break a split silently.")
    return scene


def _rotation_z(angle: float) -> np.ndarray:
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _stratified_indices(instance: np.ndarray, target: int, count: int, spec: "SampleSpec",
                        rng: np.random.Generator) -> np.ndarray:
    """Indices of ``count`` points, split three ways: the target, the other objects, the environment.

    Three ways, because the unit is a (scene, target object). With a single "objects" share the
    target takes only its 1/N, so the thing the loss is actually about drowns in true negatives.

    Every stratum that runs short gives its budget back, in priority order target, then other
    objects, then environment, so a sample is always exactly ``count`` points when the cloud has
    that many. Sampling is without replacement: a duplicated point is not extra information and
    would weight one bit of surface twice in the per-point loss.

    A smaller cloud comes back whole, and padding it here would be wrong even though a batch needs a
    fixed width. This sampler also runs at inference, where nothing is stacked and where a real cell
    routinely supplies fewer points than the budget: padding would duplicate sensor returns and make
    the calculator's `sampled` telemetry report 8,192 where far fewer points were seen. The fixed
    width is a property of batching, so `corpus.index.collate` owns it.
    """
    if len(instance) <= count:
        return np.arange(len(instance))
    pools = [
        np.flatnonzero((instance == target) & (instance >= 0)),
        np.flatnonzero((instance >= 0) & (instance != target)),
        np.flatnonzero(instance < 0),
    ]
    shares = [spec.target_fraction, spec.other_object_fraction,
              max(0.0, 1.0 - spec.target_fraction - spec.other_object_fraction)]
    picked: list[np.ndarray] = []
    budget = count
    for index, (pool, share) in enumerate(zip(pools, shares, strict=True)):
        remaining_pools = sum(len(p) for p in pools[index:])
        wanted = min(len(pool), int(round(count * share)), budget, remaining_pools)
        take = rng.choice(pool, wanted, replace=False) if wanted else pool[:0]
        picked.append(take)
        budget -= len(take)
    # Whatever is left over goes back to whichever pool still has points, in the same priority order.
    for index, pool in enumerate(pools):
        if budget <= 0:
            break
        spare = np.setdiff1d(pool, picked[index], assume_unique=False)
        extra = rng.choice(spare, min(len(spare), budget), replace=False) if len(spare) else spare
        picked.append(extra)
        budget -= len(extra)
    # Sorted, so a sample is a deterministic function of its index set rather than of the order the
    # `rng.choice` calls happened to return.
    return np.sort(np.concatenate(picked))


def _draw_scale(rng: np.random.Generator, spec: "SampleSpec",
                grasp_width_mm: np.ndarray) -> float:
    """The scene's resize factor, capped so no label outgrows the jaw. ``1.0`` means untouched.

    Log-uniform rather than uniform: `(0.8, 1.25)` should shrink as often as it grows, and a uniform
    draw over that range grows 69 % of the time.

    Returns early without drawing when the jitter is off, and that is the byte-identity guarantee
    rather than an optimisation: a draw taken and discarded still advances the caller's generator,
    and every sample after it in the epoch would differ.
    """
    low, high = float(spec.scale_jitter[0]), float(spec.scale_jitter[1])
    if low == 1.0 and high == 1.0:
        return 1.0
    widest = float(np.max(grasp_width_mm)) if len(grasp_width_mm) else 0.0
    if widest > 0.0:
        high = min(high, _MAX_JAW_APERTURE_MM / widest)
    low = min(low, high)
    if low >= high:
        return float(high)
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _assign_grasps(points_mm: np.ndarray, contacts_mm: np.ndarray, owners: np.ndarray,
                   radius_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """``(grasp index per point, distance)``: the nearest labelled contact within ``radius_mm``.

    Nearest rather than best, because there is no "best": every row in `grasps.jsonl` is a grasp the
    analytic reference accepted, so they carry no quality ordering to choose by. Nearest is at least
    deterministic and geometrically meaningful, the grasp whose pad would actually land here.
    """
    from src.robot.grasping.geometry._spatial import RadiusIndex  # noqa: PLC0415

    assigned = np.full(len(points_mm), NO_GRASP, dtype=np.int64)
    best = np.full(len(points_mm), np.inf)
    if not len(points_mm) or not len(contacts_mm):
        return assigned, best
    index = RadiusIndex(np.asarray(points_mm, dtype=np.float64))
    for contact, owner in zip(contacts_mm, owners, strict=True):
        near = index.query_radius(np.asarray(contact, dtype=np.float64), radius_mm)
        if not near.size:
            continue
        distance = np.linalg.norm(points_mm[near] - contact, axis=1)
        closer = distance < best[near]
        assigned[near[closer]] = int(owner)
        best[near[closer]] = distance[closer]
    return assigned, best


def _reachable_grasps(points_mm: np.ndarray, contacts_mm: np.ndarray, owners: np.ndarray,
                      radius_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """``(point index, grasp index)`` for every labelled contact that reaches a point.

    `_assign_grasps` keeps only the nearest, which is the right answer when one grasp has to be named
    and the wrong one when the question is which grasps are possible. Deliberately a second pass over
    the same query rather than a flag on that function: the single-label path is a locked measurement
    and this must not be able to change it.
    """
    from src.robot.grasping.geometry._spatial import RadiusIndex  # noqa: PLC0415

    if not len(points_mm) or not len(contacts_mm):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    index = RadiusIndex(np.asarray(points_mm, dtype=np.float64))
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    for contact, owner in zip(contacts_mm, owners, strict=True):
        near = index.query_radius(np.asarray(contact, dtype=np.float64), radius_mm)
        if near.size:
            rows.append(near)
            columns.append(np.full(near.size, int(owner), dtype=np.int64))
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(rows).astype(np.int64), np.concatenate(columns)



def _grasp_set_arrays(pair_rows: np.ndarray, pair_owner: np.ndarray,
                      grasp_position: np.ndarray, grasp_approach: np.ndarray,
                      grasp_axis: np.ndarray, grasp_width: np.ndarray,
                      supervise: np.ndarray, centre_xy: np.ndarray,
                      support_height_mm: float,
                      grasp_part_role: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The scene's grasp table plus which points reach which of its rows.

    Sparse on purpose. A padded `(points, max_labels, 10)` block would be mostly zeros: most points
    reach no grasp, and the head runs at a few dozen seeds rather than at all 8,192 points. A pair
    list costs one row per reachable (point, grasp) and lets the consumer gather exactly the seeds
    it drew.

    Every array here is already augmented. The rotation and the scale jitter are applied to
    `grasp_position` / `grasp_approach` / `grasp_axis` before this is called, so re-deriving anything
    from the raw scene silently mixes an augmented cloud with unaugmented labels, which produces a
    target wrong by exactly the augmentation and looks like a hard problem.

    The offset is not precomputed. It is `grasp_position[g] - point[p]`, which the consumer forms
    for the pairs it actually uses. Storing it per pair would triple this block and, worse, would fix
    the frame at build time, and a frame fixed too early is how the single-scalar depth target came
    to encode one of three degrees of freedom.

    Pairs whose point is not supervised are dropped here rather than in the loss, so the two paths
    cannot disagree about which rows carry a target.

    The position is in the same local frame as `points_m`. `points_m` is the cloud with its xy mean
    removed and the support height subtracted from z, while the grasp table arrives in the raw scene
    frame; emitting it unshifted makes `set_grasp_position_m[g] - points_m[p]` wrong by exactly that
    offset, a constant per scene that looks like a systematic bias a head could almost learn around.
    """
    keep = supervise[pair_rows] if pair_rows.size else np.zeros(0, dtype=bool)
    rows, owners = pair_rows[keep], pair_owner[keep]
    local_position = grasp_position.copy()
    local_position[:, :2] -= centre_xy
    local_position[:, 2] -= support_height_mm
    return {
        "set_pair_point": rows.astype(np.int32),
        "set_pair_grasp": owners.astype(np.int32),
        "set_grasp_position_m": (local_position * _MM_TO_M).astype(np.float32),
        "set_grasp_approach": grasp_approach.astype(np.float32),
        "set_grasp_axis": grasp_axis.astype(np.float32),
        "set_grasp_width_m": (grasp_width * _MM_TO_M).astype(np.float32),
        # The part each grasp is on, per table row rather than per pair, exactly like the four
        # arrays above. `_PART_ROLE_INDEX` maps the corpus's strings onto the fixed vocabulary, and
        # anything unknown lands on 0, the empty role, which is also what an object with no parts to
        # tell apart carries.
        #
        # Zeros when the scene carries no roles at all, rather than an absent key. A missing key
        # would make the head's supervision depend on which scene a batch happened to draw, and a
        # batch that silently trained on some samples and not others is the sort of thing that reads
        # as a noisy loss.
        "set_grasp_part_index": (
            np.asarray([_PART_ROLE_INDEX.get(str(role), 0) for role in grasp_part_role],
                       dtype=np.int64)
            if grasp_part_role is not None and len(grasp_part_role) == len(grasp_width)
            else np.zeros(len(grasp_width), dtype=np.int64)),
    }


def build_sample(scene: dict[str, np.ndarray], rng: np.random.Generator,
                 spec: SampleSpec | None = None,
                 target_instance: int | None = None) -> dict[str, Any]:
    """One scene plus a random draw, turned into the tensors a training step consumes.

    ``target_instance`` forces the conditioning; leaving it ``None`` draws one with probability
    ``spec.target_probability``, which is what makes a single net cover both runtime modes.

    Every array is in metres and every angle is a bin. Points and features are per-point; so are the
    targets, with `NO_GRASP` / NaN standing in wherever no labelled grasp reaches, masks rather than
    values, so a loss that forgot to mask trains on nothing rather than on a plausible-looking zero.
    """
    spec = spec or SampleSpec()
    points_mm = np.asarray(scene["points_mm"], dtype=np.float64)
    instance = np.asarray(scene["instance_id"], dtype=np.int64)
    normals = np.asarray(scene["normals"], dtype=np.float64)

    grasp_position = np.asarray(scene["grasp_position_mm"], dtype=np.float64)
    grasp_approach = np.asarray(scene["grasp_approach"], dtype=np.float64)
    grasp_axis = np.asarray(scene["grasp_axis"], dtype=np.float64)
    grasp_width = np.asarray(scene["grasp_width_mm"], dtype=np.float64)
    grasp_instance = np.asarray(scene["grasp_instance"], dtype=np.int64)
    contacts_mm = np.asarray(scene["contact_points_mm"], dtype=np.float64)
    contact_owner = np.asarray(scene["contact_grasp_index"], dtype=np.int64)
    # `datagen` writes a part role per grasp, and this reads it. Filling the target with
    # `torch.zeros_like` instead spends a real share of the affordance head's 6-way cross-entropy on
    # learning to answer "" everywhere, while many of the labels say grip, head, body or handle.
    grasp_part_role = np.asarray(scene.get("grasp_part_role", np.zeros(0, dtype="<U8")))

    # First, so that everything downstream runs in the resized frame: the graspability field, the
    # nearest-contact assignment, the multi-hot's radius pass, the depth projection. The absolute
    # radii (10 mm here, 20/50/120 mm in the net) deliberately do not scale with it: they describe a
    # gripper pad and a receptive field, not the object, so a bigger object correctly covers fewer
    # of them, which is the point of the augmentation.
    scale = _draw_scale(rng, spec, grasp_width)
    if scale != 1.0:
        points_mm = points_mm * scale
        grasp_position = grasp_position * scale
        grasp_width = grasp_width * scale
        contacts_mm = contacts_mm * scale
        # Not the normals, and not `grasp_approach` / `grasp_axis`: a uniform resize leaves every
        # direction where it was. Depth needs no term either, being a projection of a difference of
        # positions, so it carries the factor already.

    admissible = np.asarray(
        scene.get("grasp_approach_admissible", np.ones(len(grasp_width), dtype=bool)), dtype=bool)
    if not admissible.all():
        # An inadmissible grasp has the gripper start under the table; `check_jaw_grasp` and its
        # approach-corridor check are what mark them. Dropped rather than trained on: a generator
        # that learnt them would propose them.
        keep_contact = admissible[contact_owner] if len(contact_owner) else np.zeros(0, dtype=bool)
        remap = np.cumsum(admissible) - 1
        contacts_mm = contacts_mm[keep_contact]
        contact_owner = remap[contact_owner[keep_contact]]
        grasp_position, grasp_approach = grasp_position[admissible], grasp_approach[admissible]
        grasp_axis, grasp_width = grasp_axis[admissible], grasp_width[admissible]
        grasp_instance = grasp_instance[admissible]
        if len(grasp_part_role) == len(admissible):
            grasp_part_role = grasp_part_role[admissible]

    # A point with no usable normal is dropped, not fed as a zero vector. `estimate_surface_normals`
    # returns zeros where it could not fit a plane (too few neighbours, usually a stray point), and
    # a zero normal reaching the net is indistinguishable from a normal that means something.
    #
    # Dropped rather than flagged with a seventh channel: such points are vanishingly rare, and a
    # channel that is constant almost everywhere spends capacity to teach nothing, which is the
    # zero-variance failure the dataset gate refuses on the tabular side.
    valid = np.asarray(scene.get("normal_valid", np.ones(len(points_mm), dtype=bool)), dtype=bool)
    if not valid.all():
        points_mm, instance = points_mm[valid], instance[valid]
        normals = normals[valid]

    rotated = bool(spec.rotate_z)
    if rotated:
        # The arm goes first. Rotating it would put it where no arm on a fixed base could stand, and
        # it is the only thing in the cloud with that property: a rotated table is still a table.
        keep = instance != _ARM_INSTANCE
        points_mm, instance = points_mm[keep], instance[keep]
        normals = normals[keep]

    # The target has to be known before sampling: its share of the budget depends on it.
    if target_instance is None:
        available = np.unique(instance[instance >= 0])
        target_instance = (int(rng.choice(available))
                           if len(available) and rng.random() < spec.target_probability
                           else NO_TARGET)
    picked = _stratified_indices(instance, int(target_instance), spec.points, spec, rng)
    points_mm, instance = points_mm[picked], instance[picked]
    normals = normals[picked]

    asset_of = {int(i): str(a) for i, a in zip(scene.get("object_instance", []),
                                               scene.get("object_asset_id", []), strict=False)}
    target_asset = asset_of.get(int(target_instance), "")

    # Only the target's contacts build the field. Built from every contact in the scene, a grasp on
    # a neighbour marks any point within the radius, including table points, which are supervised,
    # and the net is then taught that the table is graspable: more units show a positive than carry
    # a grasp of their own, and the gap is a neighbour's label bleeding across the mask.
    if target_instance != NO_TARGET and len(contact_owner):
        own = grasp_instance[contact_owner] == target_instance
        contacts_mm, contact_owner = contacts_mm[own], contact_owner[own]

    # The field and the per-point assignment are computed before any rotation, in the frame the
    # contacts were measured in. Rotating a field would be the same thing computed twice.
    field = graspability_field(points_mm, contacts_mm, radius_mm=spec.graspability_radius_mm)
    # The suction half, and it only reaches the where stage. Many more objects earn a suction label
    # than a jaw one, so the share of training units carrying zero labelled points, which the
    # architecture plan names as the thing starving the heads, is measured on jaw labels alone.
    # Counting suction, far fewer objects have nothing.
    #
    # It is not a fix for the pose heads, which is why it stops here. A jaw head cannot learn a pose
    # from a suction label: a cup has no closing axis and no opening, and the corpus stores exactly
    # that, `closing_axis [0,0,0]` and `width_mm 0.0`. What it changes is the where stage, which
    # today learns from jaw labels alone and is therefore taught to call a suction-only object
    # empty. That is false for any cell carrying both end effectors, and this repository has one.
    #
    # Absent unless the corpus was built with `--kinds both`, so every existing corpus and every
    # consumer that does not know about it is byte-identical.
    suction_mm = np.asarray(scene.get("suction_position_mm", np.zeros((0, 3))),
                            dtype=np.float64).reshape(-1, 3)
    if target_instance != NO_TARGET and len(suction_mm):
        owner = np.asarray(scene.get("suction_instance", np.zeros(len(suction_mm))),
                           dtype=np.int64).reshape(-1)
        if len(owner) == len(suction_mm):
            suction_mm = suction_mm[owner == target_instance]
    suction_field = (graspability_field(points_mm, suction_mm,
                                        radius_mm=spec.graspability_radius_mm)
                     if len(suction_mm) else np.zeros(len(points_mm), dtype=np.float32))
    assigned, _distance = _assign_grasps(points_mm, contacts_mm, contact_owner,
                                         spec.graspability_radius_mm)
    # Queried here, beside the assignment and before the rotation, for the same reason the comment
    # above gives for the field: the augmentation rotates `points_mm` and deliberately does not
    # rotate `contacts_mm`, so a radius query issued after it would match rotated points against
    # unrotated contacts. With `rotate_z=True`, the default, that silently scatters the multi-hot
    # onto the wrong rows.
    pair_rows, pair_owner = (
        _reachable_grasps(points_mm, contacts_mm, contact_owner, spec.graspability_radius_mm)
        # An `or`: `grasp_set` only adds a reason to take the same second pass, so the multi-hot
        # path is still reached under exactly the `approach_multilabel` condition.
        if spec.approach_multilabel or spec.grasp_set
        else (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)))
    if target_instance != NO_TARGET:
        # Graspability is a property of the object's surface. A contact low on an object sits within
        # the radius of the table beneath it, and without this environment points come back marked
        # graspable, teaching the net that the table is somewhere to grasp, which is the opposite
        # of what those true negatives are there for.
        elsewhere = instance != target_instance
        field[elsewhere] = 0.0
        assigned[elsewhere] = NO_GRASP

    if rotated:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        rotation = _rotation_z(angle)
        centre = points_mm.mean(axis=0)
        centre[2] = 0.0                      # the support plane is not something to rotate about
        points_mm = (points_mm - centre) @ rotation.T + centre
        normals = normals @ rotation.T
        grasp_position = (grasp_position - centre) @ rotation.T + centre
        grasp_approach = grasp_approach @ rotation.T
        grasp_axis = grasp_axis @ rotation.T

    has_grasp = assigned >= NO_GRASP + 1
    approach_bin = np.full(len(points_mm), NO_GRASP, dtype=np.int64)
    rotation_bin = np.full(len(points_mm), NO_GRASP, dtype=np.int64)
    depth_m = np.full(len(points_mm), np.nan)
    # The two components the depth target discards, in the grasp's own frame. NaN where unsupervised,
    # exactly like depth, so the loss masks them the same way. Emitted only when asked for.
    lateral_m = np.full(len(points_mm), np.nan)
    binormal_m = np.full(len(points_mm), np.nan)
    width_m = np.full(len(points_mm), np.nan)
    # 0 is the "" role, which is also what an unlabelled row keeps: the head is only ever graded on
    # `has_grasp` rows, so a row without a grasp never reaches the loss whatever it holds.
    part_index = np.zeros(len(points_mm), dtype=np.int64)
    if has_grasp.any():
        owner = assigned[has_grasp]
        # Encoded after the rotation, never remapped through it: the cap is symmetric about z, so a
        # rotated approach is still in the domain, but which bin it lands in is not something to
        # compute from the old index.
        bins, rotations, in_domain = encode_grasp(
            grasp_approach[owner], grasp_axis[owner],
            approach_count=spec.approach_bins, rotation_count=spec.rotation_bins)
        rows = np.flatnonzero(has_grasp)[in_domain]
        approach_bin[rows] = bins[in_domain]
        rotation_bin[rows] = rotations[in_domain]
        kept = owner[in_domain]
        # Depth is how far along the approach the grasp centre sits from this point, and it is one
        # component of a three-dimensional offset from a seed on the object's skin to that centre.
        # The other two are discarded here and cannot be recovered afterwards by any head or any
        # decode-time repair. The perpendicular part dominates the offset, so the small approach
        # component is no guide to the size of the whole.
        #
        # Encoding a real label and decoding it this way loses the two components perpendicular to
        # the approach, which is where the whole residual lands: one finger ends up inside the
        # object and the other in free space.
        offset = grasp_position[kept] - points_mm[rows]
        if spec.predict_offset:
            # In the frame the decoder rebuilds, not the label's own. `decode_grasp` folds the
            # rotation angle into [0, pi), which is right for an axis, since a parallel jaw is
            # symmetric under axis negation, and fatal for a signed offset along that axis: the
            # decoded axis is often the label axis negated, so a correctly learned +lateral is
            # applied as -lateral and the anchor lands on the far side of the object.
            #
            # Pushing a label through the decode and grading it with the analytic referee: depth
            # alone leaves a residual, all three components measured in the label frame leave a
            # larger one, worse than no target at all, and all three in the decoded frame close it
            # exactly. The middle case is not a small error, it is the offset applied backwards,
            # and its dominant rejection is `anchor_outside`.
            #
            # This is the rule the rotation target already follows (grasp_encoding.py: "measured
            # against the binned approach ... because that is the frame the net will decode in").
            #
            # Depth moves into this branch with the other two, because a frame is not a frame if one
            # of its three components is measured against a different basis. The `else` below keeps
            # every offset-off arm byte-identical.
            decoded_approach, decoded_axis = decode_grasp(
                bins[in_domain], rotations[in_domain],
                approach_count=spec.approach_bins, rotation_count=spec.rotation_bins)
            decoded_binormal = np.cross(decoded_approach, decoded_axis)
            depth_m[rows] = np.einsum("ij,ij->i", offset, decoded_approach) * _MM_TO_M
            lateral_m[rows] = np.einsum("ij,ij->i", offset, decoded_axis) * _MM_TO_M
            binormal_m[rows] = np.einsum("ij,ij->i", offset, decoded_binormal) * _MM_TO_M
        else:
            depth_m[rows] = np.einsum("ij,ij->i", offset, grasp_approach[kept]) * _MM_TO_M
        width_m[rows] = grasp_width[kept] * _MM_TO_M
        if len(grasp_part_role):
            # The same `kept` indexing the other per-seed targets use, so a point's part is the part
            # of the very grasp its width and approach came from.
            roles = np.asarray(grasp_part_role)[kept]
            part_index[rows] = np.asarray(
                [_PART_ROLE_INDEX.get(str(r), 0) for r in roles], dtype=np.int64)

    approach_set: np.ndarray | None = None
    if spec.approach_multilabel:
        # Every bin a labelled grasp reaches this point from. Built from a second radius pass rather
        # than from `assigned`, which by construction knows only the nearest.
        approach_set = np.zeros((len(points_mm), spec.approach_bins), dtype=np.uint8)
        if pair_rows.size and len(grasp_approach):
            every_bin, _every_rotation, every_ok = encode_grasp(
                grasp_approach, grasp_axis,
                approach_count=spec.approach_bins, rotation_count=spec.rotation_bins)
            # A grasp outside the admissible cap contributes no bin, the same rule the single-label
            # path applies through `in_domain`, and dropping it here keeps the two comparable.
            usable = every_ok[pair_owner]
            approach_set[pair_rows[usable], every_bin[pair_owner[usable]]] = 1
        # The same region the rest of the supervision is confined to. A point outside it carries no
        # target at all, and a multi-hot row of ones there would train the head on a held-out asset.
        if target_instance != NO_TARGET:
            approach_set[instance != target_instance] = 0
        # And never a positive where the single-label path has none, so the two paths agree about
        # which rows are supervised and differ only in what they are told.
        approach_set[approach_bin < 0] = 0

    centre_xy = points_mm[:, :2].mean(axis=0) if len(points_mm) else np.zeros(2)
    local = points_mm.copy()
    local[:, :2] -= centre_xy
    local[:, 2] -= spec.support_height_mm

    is_target = ((instance == target_instance) & (instance >= 0)).astype(np.float64)
    # Where the loss is allowed to look. The target's own points, plus environment and arm, which
    # carry no asset identity, so a true negative on a table leaks nothing. Every other object is
    # geometry the net may see and must not be graded on: its asset could be in the held-out fold,
    # and supervising it there is exactly the memorisation an asset-disjoint split is meant to
    # exclude.
    supervise = (instance == target_instance) | (instance < 0)
    if target_instance == NO_TARGET:
        # No target named: nothing object-shaped may be supervised, or the leak is back.
        supervise = instance < 0
    if spec.ignore_unlabelled_object_points:
        # An object point that carries no labelled grasp is unknown, not negative. Environment points
        # keep their negative label, because a table really is not somewhere to grasp. See the flag's
        # own comment for the measurement that motivates this and for why it may still be a bad trade.
        unlabelled_object = (instance >= 0) & (field <= 0.0)
        supervise = supervise & ~unlabelled_object
    features = np.column_stack([
        normals,
        is_target,
        (instance >= 0).astype(np.float64),
    ])
    return {
        "points_m": (local * _MM_TO_M).astype(np.float32),
        "features": features.astype(np.float32),
        "graspability": field.astype(np.float32),
        # The union, and it is a separate key rather than a wider `graspability`. Every probe reads
        # `graspability` as "somewhere a jaw touched", and silently widening it would change what
        # six existing readers mean without any of them saying so.
        "graspable_any": np.maximum(field, suction_field).astype(np.float32),
        "graspability_suction": suction_field.astype(np.float32),
        "approach_bin": approach_bin,
        "rotation_bin": rotation_bin,
        "part_index": part_index,
        # Absent unless asked for, so a consumer that does not know about it is byte-identical.
        **({"approach_set": approach_set} if approach_set is not None else {}),
        "depth_m": depth_m.astype(np.float32),
        # Absent unless asked for, exactly like `approach_set`, so a consumer that does not know
        # about the lateral terms is byte-identical.
        **({"lateral_m": lateral_m.astype(np.float32),
            "binormal_m": binormal_m.astype(np.float32)} if spec.predict_offset else {}),
        "width_m": width_m.astype(np.float32),
        # Absent unless asked for, exactly like the two above. See `SampleSpec.grasp_set`.
        **(_grasp_set_arrays(pair_rows, pair_owner, grasp_position, grasp_approach, grasp_axis,
                             grasp_width, supervise, centre_xy, spec.support_height_mm,
                             grasp_part_role)
           if spec.grasp_set else {}),
        "instance_id": instance.astype(np.int16),
        "supervise": supervise,
        "target_instance": int(target_instance),
        #: The asset the target is, so a fold can be grouped by it. Empty when no target is named or
        #: the corpus predates the map; `grouped_folds` refuses rather than pooling those.
        "target_asset_id": target_asset,
        # Carried so a prediction can be put back where it came from. Without these a sample is a
        # tensor nobody can turn into a grasp in the cell's own frame.
        "centre_xy_mm": centre_xy.astype(np.float32),
        "support_height_mm": float(spec.support_height_mm),
    }
