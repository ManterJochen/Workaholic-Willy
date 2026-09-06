"""What a training run needs to know about the corpus, independent of any architecture.

An asset, a group, a unit and a fold are facts about the corpus, and they mean the same thing
whichever net is being fitted, so they live here rather than beside any one trainer.

`asset_group` has a caller outside this package entirely: `datagen/heldout.py`, reached from
`python -m datagen heldout`, which the datagen example tells a user to run, so a deletion confined
to `src/` still breaks a documented command. `rebalance_units` fires on every default run, because
`labelled_unit_share` defaults to 0.5.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, NamedTuple, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover (typing only; `collate` imports torch at call time)
    import torch

__all__ = [
    "TrainingUnit",
    "asset_group",
    "collate",
    "grouped_folds",
    "rebalance_units",
    "scene_family",
    "training_units",
    "units_with_grasps",
]

#: Generated ids carry a per-scene index; the family is what makes two of them the same object class.
_GENERATED_ID = re.compile(r"^(proc|comp)_\d+_(?P<family>.+)$")


def asset_group(asset_id: str) -> str:
    """The group a grasp belongs to for an object-disjoint split.

    Real meshes group by themselves: `gso_Android_Lego` is one object wherever it appears. Generated
    ones group by family, because their ids carry a per-scene index and two `proc_00002_pouch` in
    two scenes are two different pouches. Grouping those by name would be grouping by an accident of
    ordering; grouping by family asks the harder question, "can it grasp a pouch it has never seen".
    """
    match = _GENERATED_ID.match(asset_id)
    return f"family:{match.group('family')}" if match else asset_id

def grouped_folds(groups: list[str], folds: int = 5,
                  seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(train, test)`` index pairs, disjoint by group and deterministic given ``seed``.

    Written here rather than taken from sklearn's `GroupKFold` because that one balances fold sizes and
    is not seeded: two runs of the same corpus produce the same split, which is what a comparison
    between two models needs, and the balancing it does instead would put the largest groups in
    predictable places.
    """
    unique = sorted(set(groups))
    if len(unique) < folds:
        raise ValueError(f"{len(unique)} group(s) cannot make {folds} disjoint folds")
    order = np.random.default_rng(seed).permutation(len(unique))
    assignment = {unique[int(g)]: index % folds for index, g in enumerate(order)}
    membership = np.asarray([assignment[g] for g in groups])
    return [(np.flatnonzero(membership != fold), np.flatnonzero(membership == fold))
            for fold in range(folds)]

class TrainingUnit(NamedTuple):
    """One trainable thing: an object in a scene, and the asset group it belongs to."""

    scene: int
    instance: int
    group: str
    #: Does this object carry any grasp label in this scene? Read straight from the corpus's
    #: `grasp_instance`, so it costs nothing and does not depend on the sampler.
    #:
    #: Most units answer no. Those units supervise the graspability head with true negatives and
    #: nothing else, and they are the reason most units are zero-positive; the 8,192-point budget is
    #: not, since it loses few of the labelled ones. Adding scenes does not move that share, so it
    #: is a property of the asset mix and cannot be rendered away.
    labelled: bool = False
    #: Is this object's asset ever jaw-labelled anywhere in the corpus? Distinct from `labelled`
    #: above, which asks only about this scene, and the distinction is the whole point.
    #:
    #: Only some of a corpus's assets are ever jaw-labelled, so an empty unit is one of two things:
    #: an object a parallel jaw cannot take at all, or a real negative, an object graspable
    #: somewhere else in the corpus but not here because of occlusion or pose. Computed per scene
    #: instead, almost every empty unit reads as out of domain, and a filter built on that discards
    #: the real negatives too, which is exactly the signal the head needs.
    in_domain: bool = True

    # New fields go at the end. A NamedTuple's field order is part of its contract: inserting one
    # before `labelled` shifts every positional `TrainingUnit(scene, instance, group, X)` by one, so
    # the flag meaning "this object has a label in this scene" starts answering "this asset is
    # graspable somewhere".

def training_units(scenes: list[dict[str, np.ndarray]]) -> list[TrainingUnit]:
    """Every (scene, object) that can be conditioned on, with its asset group.

    An object with no labelled grasp is still a unit: it supervises the graspability head with true
    negatives, and excluding it would mean the only objects ever held out are the ones that were
    already easy. An object whose asset is unknown, as in a corpus written before the map existed,
    is not a unit, because it cannot be grouped and pooling it would silently break the split.
    """
    units: list[TrainingUnit] = []
    # One pass first, because "is this object graspable at all" is a question about the corpus and
    # not about the scene at hand. An object occluded here but graspable elsewhere is a real
    # negative; an object no jaw can take is off-topic. Deciding that per scene conflates them.
    ever_labelled: set[str] = set()
    for scene in scenes:
        instances = scene.get("object_instance", np.zeros(0, dtype=np.int16))
        assets = scene.get("object_asset_id", np.zeros(0, dtype="<U1"))
        labelled = set(np.asarray(scene.get("grasp_instance", np.zeros(0, dtype=np.int16))).tolist())
        for instance, asset in zip(instances, assets, strict=False):
            if int(instance) in labelled and str(asset):
                ever_labelled.add(str(asset))

    for index, scene in enumerate(scenes):
        instances = scene.get("object_instance", np.zeros(0, dtype=np.int16))
        assets = scene.get("object_asset_id", np.zeros(0, dtype="<U1"))
        labelled = set(np.asarray(scene.get("grasp_instance", np.zeros(0, dtype=np.int16))).tolist())
        for instance, asset in zip(instances, assets, strict=False):
            name = str(asset)
            if name:
                units.append(TrainingUnit(index, int(instance), asset_group(name),
                                          int(instance) in labelled,
                                          name in ever_labelled))
    return units

def rebalance_units(order: np.ndarray, units: "list[TrainingUnit]", share: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Draw one epoch's units so ``share`` of them carry a grasp label, keeping the epoch's length.

    Same number of steps, different mix, so a run with this on is comparable to one without on
    everything except the mix. The minority side is drawn with replacement: a share above the natural
    rate repeats labelled units rather than inventing them, and the log says which it did.
    """
    labelled = np.array([i for i in order if units[int(i)].labelled], dtype=order.dtype)
    plain = np.array([i for i in order if not units[int(i)].labelled], dtype=order.dtype)
    if not len(labelled) or not len(plain):
        return order                      # nothing to trade against; leave the epoch untouched
    total = len(order)
    want_labelled = int(round(total * float(share)))
    want_labelled = max(1, min(total - 1, want_labelled))
    picked = np.concatenate([
        rng.choice(labelled, size=want_labelled, replace=want_labelled > len(labelled)),
        rng.choice(plain, size=total - want_labelled, replace=(total - want_labelled) > len(plain)),
    ])
    return rng.permutation(picked)


# ---------------------------------------------------------------------------
# Three more that a surviving command needs.
#
# `deep geometric-floor` imports all three. It is not a binned probe: it measures what a purely
# geometric rule achieves on the corpus, which is the baseline every learned number is read against,
# and it is architecture-independent.
# ---------------------------------------------------------------------------

def scene_family(name: str) -> str:
    """The family a scene belongs to, from its own file name: `bin_000004` gives `bin`.

    Read from the name rather than from a manifest because the corpus does not carry the family as a
    field, and the naming is stable across every shard. A name that does not split returns itself,
    so an unfamiliar corpus is visible rather than silently bucketed.
    """
    return name.split("_", 1)[0] if "_" in name else name


def units_with_grasps(scenes: "Sequence[dict[str, Any]]", limit: int) -> list[tuple[int, int]]:
    """`(scene index, target instance)` pairs whose target actually carries a labelled grasp.

    Not an arbitrary draw. Most objects carry no labelled grasp at all, so a uniform draw of sixteen
    units would very likely contain no positive point anywhere, and `deep geometric-floor` would
    then measure its training-free rules on units that carry no grasp to hit.
    """
    import numpy as np  # noqa: PLC0415

    found: list[tuple[int, int]] = []
    for index, scene in enumerate(scenes):
        grasp_instance = np.asarray(scene.get("grasp_instance", np.zeros(0)), dtype=np.int64)
        if not len(grasp_instance):
            continue
        for instance in np.unique(grasp_instance):
            found.append((index, int(instance)))
            if len(found) >= limit:
                return found
    return found


def collate(samples: list[dict[str, Any]], device: str = "cpu") -> "dict[str, torch.Tensor]":
    """Stack `corpus.sample.build_sample` samples into one batch, padding a short one if it appears.

    Samples do not all carry the same point count. `_stratified_indices` gives spare budget back so
    a sample is exactly the budget when the cloud has that many; a cloud that is smaller comes back
    whole, deliberately, because the same sampler runs at inference where a real cell routinely
    supplies fewer points and padding would duplicate sensor returns. A dense corpus never hits
    this, while a `sparse` scene of two randomly-oriented objects can fall under 8,192, so a trainer
    without this dies on the occasional scene.

    The fixed width is a property of batching, so it is enforced here: a short sample is padded to
    the batch's width by repeating points it already has, applied identically to every per-point
    array so they stay aligned. The duplicates weight a little surface twice in the per-point loss,
    which is the price of keeping those scenes; the alternatives are dropping them, which biases the
    split toward dense ones, or a mask threaded through every head's loss.
    """
    # Torch at call time, not at import. `corpus.index` is imported by `datagen.heldout`,
    # which has no business paying for torch to ask which group an asset belongs to.
    import torch  # noqa: PLC0415
    widths = {len(sample["points_m"]) for sample in samples}
    if len(widths) > 1:
        width = max(widths)
        padded = []
        for sample in samples:
            have = len(sample["points_m"])
            if have == width or not have:
                padded.append(sample)
                continue
            index = np.concatenate([np.arange(have),
                                    np.arange(width - have) % have])
            padded.append({
                key: (np.asarray(value)[index]
                      if isinstance(value, np.ndarray) and len(value) == have else value)
                for key, value in sample.items()
            })
        samples = padded

    def stack(key: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack([torch.as_tensor(s[key], dtype=dtype) for s in samples]).to(device)

    batch = {
        "points_m": stack("points_m", torch.float32),
        "features": stack("features", torch.float32),
        "graspability": stack("graspability", torch.float32),
        "approach_bin": stack("approach_bin", torch.long),
        "rotation_bin": stack("rotation_bin", torch.long),
        "depth_m": stack("depth_m", torch.float32),
        "width_m": stack("width_m", torch.float32),
    }
    # A fixed key list drops any target added later, silently. `build_sample` emits `lateral_m` and
    # `binormal_m` when `predict_offset` is on; a copy that omits them leaves the loss term, which
    # reads `"lateral_m" in targets`, never firing, and `balance_weights` reports the offset factor
    # as its "nothing to balance against" fallback of exactly 1.000 while every curve looks normal.
    for optional in ("lateral_m", "binormal_m"):
        if optional in samples[0]:
            batch[optional] = stack(optional, torch.float32)
    # NaN is the dataset's mask for "no grasp here", and it must not reach a loss even multiplied by
    # zero: `0 * nan` is nan, and one nan poisons every gradient in the batch.
    for key in ("depth_m", "width_m", "lateral_m", "binormal_m"):
        if key in batch:
            batch[key] = torch.nan_to_num(batch[key], nan=0.0)
    batch["supervise"] = torch.stack(
        [torch.as_tensor(s["supervise"], dtype=torch.bool) for s in samples]).to(device)
    # An all-zeros target for this 6-way cross-entropy spends a real share of the trunk's gradient
    # at initialisation on teaching the head to answer "" everywhere, while the corpus carries a
    # part role per grasp, many of them non-blank: grip, head, body, handle. So the role is read
    # from the sample.
    #
    # A sample written before `build_sample` emitted it still falls back to zeros, so an old corpus
    # or an old caller behaves exactly as before rather than raising.
    if all("part_index" in sample for sample in samples):
        batch["part_index"] = stack("part_index", torch.long)
    else:
        batch["part_index"] = torch.zeros_like(batch["approach_bin"])
    # Present only if every sample has it. A batch that mixed multilabel and single-label samples
    # would stack short, and half-supervising a head is worse than not.
    if all("approach_set" in sample for sample in samples):
        batch["approach_set"] = stack("approach_set", torch.uint8)
    return batch
