"""How multimodal is a supervised point on the current corpus, and what does the set cost?

A multimodality figure belongs to the corpus it was measured on. A denser label set can only make
points more multimodal, so a figure carried over from an earlier corpus is at best a floor, and this
runs over the corpus named on the command line rather than quoting one.

Reports three things, because they answer three different questions:

    labels per point      how many grasps reach a point at all
    distinct approaches   how many of them point in genuinely different directions (>15 deg apart)
    distinct axes         the same, for the closing axis, folded antipodally

The second is the one that decides whether a single-output head can work. The third decides whether
the director representation is enough on its own.
"""
from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.robot.grasping.deep.corpus.sample import SampleSpec, build_sample, load_scene

#: The defaults `main()` falls back to when the caller names nothing. `main()` parses the arguments;
#: nothing here reads `sys.argv` at import. A module that did would take its value from the caller's
#: command line under pytest or inside a service process.
DEFAULT_ROOT = Path("logs/dl/clouds/v5")
DEFAULT_SCENES = 60
SEPARATION_DEG = 15.0


def _distinct(directions: np.ndarray, *, antipodal: bool) -> int:
    """Greedy count of directions no two of which lie within SEPARATION_DEG of each other.

    Greedy, so order-dependent and an estimate rather than a minimum cover. It is the right shape
    for the question anyway: a head fills its slots one at a time too, and what is being asked is how
    many genuinely different answers a point holds, not the smallest set that covers them.
    """
    threshold = float(np.cos(np.deg2rad(SEPARATION_DEG)))
    kept: list[np.ndarray] = []
    for row in directions:
        if any((abs(row @ k) if antipodal else row @ k) > threshold for k in kept):
            continue
        kept.append(row)
    return len(kept)


def main(argv: Sequence[str] | None = None) -> None:
    """`argv` is the corpus directory and the scene count, both optional."""
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else DEFAULT_ROOT
    scenes = int(args[1]) if len(args) > 1 else DEFAULT_SCENES

    files = sorted(root.rglob("*.npz"))
    if not files:
        raise SystemExit(f"no clouds under {root}")
    # An even stride, never a prefix: scene ids sort by family, so `[:60]` is one family.
    step = len(files) / scenes
    chosen = [files[int(i * step)] for i in range(min(scenes, len(files)))]
    print(f"{root}: {len(files)} clouds, reading {len(chosen)}\n", flush=True)

    spec = SampleSpec(grasp_set=True)
    rng = np.random.default_rng(0)
    per_point: Counter[int] = Counter()
    approaches: Counter[int] = Counter()
    axes: Counter[int] = Counter()
    points_with_any = 0
    for path in chosen:
        scene = load_scene(path)
        instances = [int(i) for i in np.unique(scene["grasp_instance"])] if "grasp_instance" in scene else []
        for instance in instances[:3]:
            sample = build_sample(scene, rng, spec, target_instance=instance)
            if "set_pair_point" not in sample:
                raise SystemExit("the sample carries no grasp set; the flag did not reach it")
            pairs, owners = sample["set_pair_point"], sample["set_pair_grasp"]
            if not pairs.size:
                continue
            order = np.argsort(pairs, kind="stable")
            pairs, owners = pairs[order], owners[order]
            edges = np.flatnonzero(np.diff(pairs)) + 1
            for group in np.split(owners, edges):
                points_with_any += 1
                per_point[min(len(group), 20)] += 1
                if len(group) > 1:
                    approaches[min(_distinct(sample["set_grasp_approach"][group],
                                             antipodal=False), 20)] += 1
                    axes[min(_distinct(sample["set_grasp_axis"][group], antipodal=True), 20)] += 1
                else:
                    approaches[1] += 1
                    axes[1] += 1

    if not points_with_any:
        raise SystemExit("no supervised point reached a grasp")
    multi = sum(n for k, n in per_point.items() if k > 1)
    multi_app = sum(n for k, n in approaches.items() if k > 1)
    multi_axis = sum(n for k, n in axes.items() if k > 1)
    print(f"  supervised points reaching >=1 grasp   {points_with_any:>9,}")
    print(f"  more than ONE label                    {multi / points_with_any:>8.1%}")
    print(f"  more than one DISTINCT approach        {multi_app / points_with_any:>8.1%}"
          f"   (>{SEPARATION_DEG:.0f} deg apart)")
    print(f"  more than one DISTINCT closing axis    {multi_axis / points_with_any:>8.1%}"
          f"   (antipodally folded)")
    counts = np.array([k for k, n in per_point.items() for _ in range(n)])
    print(f"\n  labels per point: median {np.median(counts):.0f}, "
          f"mean {counts.mean():.1f}, 90th pct {np.percentile(counts, 90):.0f}")
    app = np.array([k for k, n in approaches.items() for _ in range(n)])
    print(f"  distinct approaches per point: median {np.median(app):.0f}, "
          f"mean {app.mean():.2f}, 90th pct {np.percentile(app, 90):.0f}")
    print(f"\n  => K slots is worth up to {int(np.percentile(app, 90))} on this corpus; "
          f"K=1 discards {multi_app / points_with_any:.1%} of what a point knows.")


if __name__ == "__main__":
    main()
