"""Is the approach direction a property of the object, or of the scene it happens to sit in?

The question that bounds the whole global-to-local build order, and it is answered from labels alone
with no network and no training.

`object_oracle` measures how much of a scene-instance's supervised points one direction covers. It
says nothing about whether that direction can be inferred from the object: it is computed per
instance, with hindsight, on the very instance it is scored on. If the same asset gets a different
best direction every time it appears, no amount of object-feature learning can predict it, and a
head that emits one answer per object is capped at whatever a single per-asset direction achieves.

The mechanism this asks about: a labelled grasp's approach is where the gripper could reach, and in
`pile` and `packed` scenes that is decided by neighbours, walls and the table, not by the object's
own shape. Two instances of the same asset in different surroundings would then carry different
labels with the perception stack working correctly. Many objects want something other than straight
down.

The three numbers, on one scale, all `approach_hit`:

* `global`: one bin for the whole corpus.
* `asset`: one bin per asset, shared across every scene it appears in.
* `instance`: one bin per scene-instance, the familiar `object_oracle`.

`asset` near `instance` means the direction is an object property and a per-object head is the right
shape. `asset` near `global` means the direction is scene-determined, the per-object head is capped
at the floor by construction, and the information a net would need lives in the neighbourhood.

All three are oracles: they read the labels of the points they are scored on and bound what is
achievable. None of them is a baseline and none may be compared against a trained arm as though it
were. What makes the comparison meaningful is the ratio between them, not any one value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Sequence
    from pathlib import Path

__all__ = ["AssetConsistency", "format_consistency", "measure_asset_consistency"]


@dataclass(frozen=True, slots=True)
class AssetConsistency:
    """How much of the per-instance approach ceiling survives being shared across an asset."""

    #: Assets appearing in at least `min_scenes` scenes, and the instances they account for.
    assets: int
    instances: int
    scored_points: int
    min_scenes: int
    #: `approach_hit` for one bin per corpus, per asset, and per scene-instance.
    global_hit: float
    asset_hit: float
    instance_hit: float
    #: Median angle in degrees between an instance's own best direction and its asset's best one.
    median_deviation_deg: float
    p90_deviation_deg: float
    #: Share of instances whose own best bin is their asset's best bin.
    share_agreeing: float
    #: Per-asset spread: median angle between the best directions of two instances of the same asset.
    median_within_asset_deg: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def recovered(self) -> float:
        """Of the headroom above a global constant, how much a per-asset constant recovers.

        1.0 means the direction is fully an asset property. 0.0 means it is entirely scene-determined
        and an object-level head cannot beat a global constant however well it is trained.
        """
        headroom = self.instance_hit - self.global_hit
        if headroom <= 0.0:
            return float("nan")
        return (self.asset_hit - self.global_hit) / headroom


def measure_asset_consistency(files: "Sequence[Path]", *,
                              approach_bins: int = 100,
                              max_tilt_deg: float = 135.0,
                              min_scenes: int = 3) -> AssetConsistency | None:
    """Read every scene's labels and compare a per-asset constant against a per-instance one.

    Scored over labelled grasps, not over cloud points. The trained arms score per supervised point,
    which needs the sampled cloud and therefore a full `build_sample` for every unit; here the
    population is the labelled grasps themselves, the population the direction is a property of, and
    it costs one small array per scene instead of a farthest-point sample. The two are not
    interchangeable and these numbers are not directly comparable to a trained arm's; what is
    comparable is the ratio of the three oracles.
    """
    import numpy as np  # noqa: PLC0415

    from src.robot.grasping.deep.corpus.grasp_encoding import (  # noqa: PLC0415
        approach_bin as encode_bin,
    )
    from src.robot.grasping.deep.corpus.grasp_encoding import (  # noqa: PLC0415
        approach_directions,
    )

    directions = np.asarray(approach_directions(approach_bins, max_tilt_deg), dtype=np.float64)
    per_asset: dict[str, list[np.ndarray]] = {}
    per_instance: list[tuple[str, np.ndarray]] = []
    notes: list[str] = []
    unusable = 0

    for path in files:
        with np.load(path, allow_pickle=False) as handle:
            if "grasp_approach" not in handle or "grasp_asset_id" not in handle:
                unusable += 1
                continue
            approaches = np.asarray(handle["grasp_approach"], dtype=np.float64)
            asset_ids = np.asarray(handle["grasp_asset_id"])
            instances = np.asarray(handle["grasp_instance"], dtype=np.int64)
        if not len(approaches):
            continue
        # The bin, not the raw vector. Everything downstream is scored in bins, and a direction
        # outside the 135 degree cap has no bin at all: `encode_bin` reports that separately and
        # those rows are dropped rather than snapped to a neighbour.
        bins, admissible = encode_bin(approaches, approach_bins, max_tilt_deg)
        for instance in np.unique(instances):
            rows = (instances == instance) & admissible
            if not rows.any():
                continue
            asset = str(asset_ids[rows][0])
            counts = np.bincount(bins[rows], minlength=approach_bins)
            per_instance.append((asset, counts))
            per_asset.setdefault(asset, []).append(counts)

    if unusable:
        notes.append(f"{unusable} scene(s) carry no grasp_approach/grasp_asset_id and were skipped")
    if not per_instance:
        return None

    # Only assets seen in enough scenes can say anything about consistency across scenes.
    repeated = {asset: rows for asset, rows in per_asset.items() if len(rows) >= min_scenes}
    if not repeated:
        notes.append(f"no asset appears in {min_scenes} or more scenes")
        return None
    kept = [(asset, counts) for asset, counts in per_instance if asset in repeated]

    total = float(sum(int(counts.sum()) for _, counts in kept))
    corpus_counts = np.zeros(approach_bins, dtype=np.int64)
    for _, counts in kept:
        corpus_counts += counts
    global_bin = int(corpus_counts.argmax())
    asset_bin = {asset: int(np.sum(rows, axis=0).argmax()) for asset, rows in repeated.items()}

    global_hits = sum(int(counts[global_bin]) for _, counts in kept)
    asset_hits = sum(int(counts[asset_bin[asset]]) for asset, counts in kept)
    instance_hits = sum(int(counts.max()) for _, counts in kept)

    def _angle(first: int, second: int) -> float:
        return float(np.degrees(np.arccos(
            np.clip(float(directions[first] @ directions[second]), -1.0, 1.0))))

    deviations = [_angle(int(counts.argmax()), asset_bin[asset]) for asset, counts in kept]
    agreeing = sum(1 for asset, counts in kept if int(counts.argmax()) == asset_bin[asset])
    within: list[float] = []
    for asset, rows in repeated.items():
        best = [int(counts.argmax()) for counts in rows]
        within.extend(_angle(a, b) for i, a in enumerate(best) for b in best[i + 1:])

    return AssetConsistency(
        assets=len(repeated), instances=len(kept), scored_points=int(total),
        min_scenes=min_scenes,
        global_hit=global_hits / total, asset_hit=asset_hits / total,
        instance_hit=instance_hits / total,
        median_deviation_deg=float(np.median(deviations)),
        p90_deviation_deg=float(np.percentile(deviations, 90)),
        share_agreeing=agreeing / len(kept),
        median_within_asset_deg=float(np.median(within)) if within else float("nan"),
        notes=tuple(notes))


def format_consistency(result: "AssetConsistency | None") -> str:
    """ASCII only, for the reason recorded throughout this package."""
    if result is None:
        return "no asset measured"
    lines: list[str] = []
    lines.append(f"{result.assets} asset(s) appearing in {result.min_scenes}+ scenes, "
                 f"{result.instances} instance(s), {result.scored_points} labelled grasp(s)")
    lines.append("")
    lines.append(f"  {'oracle':<12}{'hit':>9}   what one bin is shared across")
    lines.append(f"  {'global':<12}{result.global_hit:>9.4f}   the whole corpus")
    lines.append(f"  {'asset':<12}{result.asset_hit:>9.4f}   every appearance of one asset")
    lines.append(f"  {'instance':<12}{result.instance_hit:>9.4f}   one scene-instance (the familiar "
                 f"object_oracle)")
    lines.append("")
    lines.append(f"  a per-ASSET constant recovers {result.recovered * 100:.1f} % of the headroom "
                 f"a per-INSTANCE one has")
    lines.append("")
    lines.append(f"  instance vs its own asset: median {result.median_deviation_deg:.1f} deg, "
                 f"p90 {result.p90_deviation_deg:.1f} deg, "
                 f"{result.share_agreeing * 100:.1f} % share the same bin")
    lines.append(f"  two instances of the SAME asset: median "
                 f"{result.median_within_asset_deg:.1f} deg apart")
    lines.append("")
    # The reading, stated rather than left to the reader, because the two outcomes send the build
    # order to different places and neither is a failure.
    recovered = result.recovered
    if recovered >= 0.7:
        lines.append("  READING: the direction is largely an ASSET property. A per-object head is the")
        lines.append("  right shape, and Build 1's failure points at capacity or budget, not at form.")
    elif recovered <= 0.3:
        lines.append("  READING: the direction is largely SCENE-determined. A per-object head is capped")
        lines.append("  near the global constant BY CONSTRUCTION, which is exactly what Build 1")
        lines.append("  measured, and the information a net needs lives in the NEIGHBOURHOOD.")
    else:
        lines.append("  READING: mixed. Neither a per-object head nor the neighbourhood alone accounts")
        lines.append("  for the direction, and the fusion of the two is where the answer would be.")
    lines.append("")
    lines.append("  All three rows are ORACLES: they read the labels of the grasps they are scored on.")
    lines.append("  None is a baseline, and none may be set against a trained arm as though it were.")
    for note in result.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)
