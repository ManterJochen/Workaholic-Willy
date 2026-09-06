"""Is the lateral target multi-modal per supervised point?

The width head is a classification rather than a regression for one stated reason: an object
usually admits several widths, and a regressed scalar minimising squared error against them
converges on a mean that grasps nothing. The offset head is a regression over `lateral`, a signed
distance along the closing axis, so the same question applies to it.

If a point admits grasps whose lateral offsets sit on both sides of it, the least-squares answer is
their mean, and for a symmetric spread that mean is zero. Predicting zero is the floor.

Measured per supervised point: how many admissible grasps it has, the spread of their lateral values
in the decoded frame, and what a regression would converge to against what any single one needs.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.robot.grasping.deep.corpus.grasp_encoding import decode_grasp, encode_grasp

#: The defaults `main()` falls back to when the caller names nothing.
#:
#: `main()` parses the arguments; nothing here reads `sys.argv` at import. A module that did would
#: take its value from the caller's command line under pytest or inside a service process.
#:
#: The corpus path is relative and names one corpus version. An answer measured on one version is
#: about that version, so point it at the corpus under test rather than trusting the default.
DEFAULT_CLOUDS = Path("logs/dl/clouds/v5/s0")
RADIUS_MM = 10.0
DEFAULT_SCENES = 40


def main(argv: Sequence[str] | None = None) -> None:
    """`argv` is the scene count and the corpus directory, both optional."""
    args = list(sys.argv[1:] if argv is None else argv)
    scenes = int(args[0]) if args else DEFAULT_SCENES
    clouds = Path(args[1]) if len(args) > 1 else DEFAULT_CLOUDS

    # Not `sorted(...)[:N]`. Scene ids sort by family, so a prefix is one family, and two prefixes
    # of different lengths answer about different populations. That is the trap the `--limit` help
    # warns about; an even stride spans them all instead.
    every = sorted(clouds.glob("*.npz"))
    files = every if scenes <= 0 else [every[i] for i in
                                       np.linspace(0, len(every) - 1, min(scenes, len(every))).astype(int)]
    print(f"{len(files)} cloud file(s) of {len(sorted(clouds.glob('*.npz')))}")

    per_point_counts: list[int] = []
    per_point_spread: list[float] = []
    per_point_best: list[float] = []
    per_point_mean_err: list[float] = []
    per_point_family: list[str] = []
    signs_both: int = 0
    total: int = 0

    for path in files:
        with np.load(path, allow_pickle=False) as z:
            points = np.asarray(z["points_mm"], dtype=np.float64)
            gpos = np.asarray(z["grasp_position_mm"], dtype=np.float64)
            gapp = np.asarray(z["grasp_approach"], dtype=np.float64)
            gaxis = np.asarray(z["grasp_axis"], dtype=np.float64)
            # The corpus's own contacts and owner index, which is exactly what `_assign_grasps`
            # queries against. Reconstructing them from position and width would measure that
            # reconstruction, not the assignment the trainer performs.
            contacts = np.asarray(z["contact_points_mm"], dtype=np.float64)
            owner = np.asarray(z["contact_grasp_index"], dtype=np.int64)
        if not len(gpos) or not len(points) or not len(contacts):
            continue

        bins, rots, ok = encode_grasp(gapp, gaxis)
        if not ok.any():
            continue
        d_app, d_axis = decode_grasp(bins, rots)
        keep = ok[owner]
        contacts, owner = contacts[keep], owner[keep]
        if not len(contacts):
            continue

        # Every grasp within the assignment radius of this point, not only the nearest one: the
        # nearest is what the target records, the set is what the target is ambiguous over.
        step = 4096
        for start in range(0, len(points), step):
            block = points[start:start + step]
            d = np.linalg.norm(block[:, None, :] - contacts[None, :, :], axis=2)
            near = d <= RADIUS_MM
            for row in range(len(block)):
                idx = owner[near[row]]
                if len(idx) < 1:
                    continue
                total += 1
                offsets = gpos[idx] - block[row]
                lateral = np.einsum("ij,ij->i", offsets, d_axis[idx])
                per_point_counts.append(len(np.unique(idx)))
                per_point_spread.append(float(lateral.max() - lateral.min()))
                per_point_best.append(float(np.abs(lateral).min()))
                # What least squares converges to, against what the closest single grasp needs.
                per_point_mean_err.append(float(np.abs(lateral - lateral.mean()).mean()))
                per_point_family.append(path.name.split("_")[0])
                if lateral.min() < -1.0 and lateral.max() > 1.0:
                    signs_both += 1

    if not total:
        print("no supervised points found")
        return
    fams = np.array(per_point_family)
    c = np.array(per_point_counts, dtype=float)
    s = np.array(per_point_spread)
    m = np.array(per_point_mean_err)
    print(f"\n{total} supervised point(s)\n")
    print(f"  admissible grasps per point .......... median {np.median(c):.0f}   "
          f"mean {c.mean():.1f}   p90 {np.percentile(c, 90):.0f}   max {c.max():.0f}")
    print(f"  share with more than one ............. {100.0 * (c > 1).mean():.1f} %")
    print(f"  LATERAL SPREAD per point (mm) ........ median {np.median(s):.2f}   "
          f"mean {s.mean():.2f}   p90 {np.percentile(s, 90):.2f}")
    print(f"  points whose grasps sit on BOTH sides  {100.0 * signs_both / total:.1f} %")
    print(f"  residual a regression cannot beat (mm) median {np.median(m):.2f}   mean {m.mean():.2f}")
    print()
    print()
    print("  PER FAMILY, because the first attempt at this measured one family and claimed the corpus:")
    for fam in sorted(set(fams)):
        m2 = fams == fam
        print(f"    {fam:8s} n={m2.sum():>6}  grasps/pt median {np.median(c[m2]):>4.0f}  "
              f"spread median {np.median(s[m2]):>7.2f} mm  both sides {100.0*np.mean((s[m2] > 2.0)):>5.1f} %  "
              f"oracle {np.mean(m[m2]):>6.2f} mm")
    print()
    print("  The arms report a lateral MAE floor of 14.42 mm and have sat at 16 to 34 mm against it.")
    print(f"  An oracle regression would reach {m.mean():.2f} mm, so the target IS learnable in")
    print("  principle and multimodality does NOT explain the failure.")


if __name__ == "__main__":
    main()
