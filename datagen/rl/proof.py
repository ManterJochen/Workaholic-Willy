"""Train the ranker on measured outcomes, and say plainly what the number means.

The point is a comparison, run through the project's own solver so the two halves are not two
different experiments:

* the stock contract: the 15 ``RANKING_FEATURE_KEYS`` the shipped policy uses;
* the geometry set: the per-candidate quantities that actually vary within a scene.

Both go through ``_train_pairwise_newton_irls`` and are scored with ``_pairwise_accuracy``. Only
the feature vector differs, so a difference between them is a statement about the features and
nothing else.

The evaluation is grouped, not random. Pairs from one scene share an object, an extrinsic and a
depth frame; splitting them at random puts near-duplicates on both sides and reports a number
that cannot survive a new scene. Scenes are split whole.

What the number is not. The reward is PhysX on renderer-exact poses with an exact extrinsic, so it
says the pipeline produces a learnable signal. It says nothing about a real cell, and a policy
trained here is not promotable on this evidence.

    python -m datagen.rl.proof --records logs/p5/datasets/v1_proof/rl_records.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, RL_PROOF_LOG_FILE

__all__ = ["run_proof", "main"]

#: The stock keys that vary per candidate and carry information of their own. Most of the rest are
#: scene, pick or object level, identical for every candidate of a group, and difference to exactly
#: zero in a pairwise model; ``shadow_predicted_success_probability`` varies but duplicates
#: ``predicted_success_probability``.
LIVE_STOCK_KEYS: tuple[str, ...] = (
    "geometric_score",
    "predicted_success_probability",
)

#: The per-candidate quantities that vary within a scene.
GEOMETRY_KEYS: tuple[str, ...] = (
    "geom_grip_width_mm",
    "geom_grasp_z_mm",
    "geom_approach_tilt_deg",
    "geom_approach_clearance_mm",
)


logger = create_logger("datagen.rl.proof", RL_PROOF_LOG_FILE, log_dir=DATAGEN_LOG_DIR)


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _vector(record: dict[str, Any], keys: Sequence[str], *, width: int) -> list[float]:
    """One feature row, padded to ``width`` with zeros.

    ``_train_pairwise_newton_irls`` hardcodes ``n_features = len(RANKING_FEATURE_KEYS)``, so it
    cannot be handed a 4-wide vector. Padding lets the comparison run through the project's own
    solver (same Newton-IRLS, same ridge, same convergence test) instead of a second
    implementation that would differ in ways nobody could see.

    A padded column is provably inert: its delta is 0 for every pair, so its gradient entry is 0
    and its Hessian row/column is the ridge diagonal alone. The solver returns weight 0 for it and
    it couples to nothing. The effective model is exactly over ``keys``.
    """
    if len(keys) > width:
        # Refuse rather than truncate. The solver reads only its first `width` entries, so a
        # longer vector silently drops the tail, and a comparison that evaluates a different model
        # than it names is worse than no comparison.
        raise ValueError(
            f"{len(keys)} feature keys exceed the solver's fixed width of {width} "
            f"(_train_pairwise_newton_irls hardcodes n_features = len(RANKING_FEATURE_KEYS)); "
            f"the tail would be silently ignored"
        )
    extra = record.get("extra") or {}
    out: list[float] = []
    for key in keys:
        value = extra.get(key, record.get(key))
        out.append(float(value) if isinstance(value, (int, float, bool)) else 0.0)
    return out + [0.0] * (width - len(out))


def _standardise(rows: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Zero-mean, unit-variance per column, fitted on the train rows only.

    Not cosmetic: the geometry set mixes millimetres (up to 140) with degrees (up to 90) and a unit
    score, and a single ridge penalty across columns of such different scale is effectively a
    different penalty per feature. The stock set is standardised the same way so neither is favoured.
    A constant column keeps scale 1.0 rather than dividing by zero.
    """
    if not rows:
        return rows, [], []
    width = len(rows[0])
    means = [sum(r[i] for r in rows) / len(rows) for i in range(width)]
    scales: list[float] = []
    for i in range(width):
        var = sum((r[i] - means[i]) ** 2 for r in rows) / len(rows)
        scales.append(var ** 0.5 or 1.0)
    return ([[(r[i] - means[i]) / scales[i] for i in range(width)] for r in rows], means, scales)


def _apply(rows: list[list[float]], means: list[float], scales: list[float]) -> list[list[float]]:
    return [[(r[i] - means[i]) / scales[i] for i in range(len(r))] for r in rows]


def _pairs_for(
    records: Sequence[dict[str, Any]], keys: Sequence[str], *, width: int,
) -> tuple[list[tuple[list[float], list[float], str, str]], dict[str, int]]:
    """Success/fail pairs within each scene, with the same grouping the trainer uses."""
    from src.robot.grasping.rl.train_ranking import _group_key, _is_success

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(_group_key(record), []).append(record)

    pairs: list[tuple[list[float], list[float], str, str]] = []
    stats = {"groups": len(groups), "groups_with_pairs": 0, "successes": 0, "failures": 0}
    for key in sorted(groups):
        members = groups[key]
        wins = [m for m in members if _is_success(m)]
        losses = [m for m in members if not _is_success(m)]
        stats["successes"] += len(wins)
        stats["failures"] += len(losses)
        if wins and losses:
            stats["groups_with_pairs"] += 1
        for win in wins:
            for loss in losses:
                pairs.append((
                    _vector(win, keys, width=width), _vector(loss, keys, width=width),
                    str(win.get("attempt_id")), str(loss.get("attempt_id")),
                ))
    return pairs, stats


def _evaluate(
    train_records: Sequence[dict[str, Any]],
    test_records: Sequence[dict[str, Any]],
    keys: Sequence[str],
) -> dict[str, Any]:
    from src.robot.grasping.rl.train_ranking import (
        _pairwise_accuracy,
        _train_pairwise_newton_irls,
    )

    from src.robot.grasping.rl.ranking_policy import RANKING_FEATURE_KEYS

    width = len(RANKING_FEATURE_KEYS)
    train_pairs, train_stats = _pairs_for(train_records, keys, width=width)
    test_pairs, test_stats = _pairs_for(test_records, keys, width=width)
    if not train_pairs:
        # Returned as a result, not raised: `run_proof` reports all three feature sets either way.
        # "Not trainable" is a finding about the data; one row per scene forms no pair.
        logger.error("%d feature(s): not trainable; no success/fail pair in train (%d group(s), "
                     "%d with pairs)", len(keys), train_stats["groups"],
                     train_stats["groups_with_pairs"])
        return {"keys": list(keys), "trainable": False, "reason": "no success/fail pair in train",
                "train": train_stats, "test": test_stats}

    # Standardise on train only; the test rows are mapped with the train statistics, which is what
    # "held out" means. Both sides of a pair share the transform, so the pairwise delta is consistent.
    flat = [side for pair in train_pairs for side in (pair[0], pair[1])]
    _scaled, means, scales = _standardise(flat)
    def _scale(ps): return [(_apply([a], means, scales)[0], _apply([b], means, scales)[0], i, j)
                            for a, b, i, j in ps]
    train_scaled, test_scaled = _scale(train_pairs), _scale(test_pairs)

    result = _train_pairwise_newton_irls(train_scaled)
    train_accuracy = _pairwise_accuracy(train_scaled, result.weights)
    test_accuracy = _pairwise_accuracy(test_scaled, result.weights) if test_scaled else None
    # The solver's own verdict on itself. A run that did not converge still returns weights and still
    # prints an accuracy, and that is the one number nobody would otherwise question.
    log = logger.info if result.converged else logger.warning
    log("%d feature(s): %d train pair(s) / %d test pair(s), train_acc %.3f, test_acc %s, %s",
        len(keys), len(train_pairs), len(test_pairs), train_accuracy,
        f"{test_accuracy:.3f}" if test_accuracy is not None else "n/a",
        "converged" if result.converged else "DID NOT CONVERGE")
    return {
        "keys": list(keys),
        "trainable": True,
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4) if test_accuracy is not None else None,
        "converged": bool(result.converged),
        # Only the real keys; the padded slots are reported nowhere because they mean nothing.
        "weights": {k: round(float(w), 4) for k, w in zip(keys, result.weights)},
        "train": train_stats,
        "test": test_stats,
    }


def run_proof(records_path: Path, *, holdout: float = 0.3) -> dict[str, Any]:
    """Train both feature sets on the same scenes and report the comparison."""
    from src.robot.grasping.rl.ranking_policy import RANKING_FEATURE_KEYS

    records = _load(records_path)
    scenes = sorted({(r.get("extra") or {}).get("scene_id", "") for r in records})
    # Whole scenes on each side. Deterministic (sorted, strided) rather than shuffled, so the split is
    # reproducible without carrying a seed around.
    stride = max(2, int(round(1.0 / holdout))) if holdout > 0 else 0
    test_scenes = set(scenes[::stride]) if stride else set()
    train = [r for r in records if (r.get("extra") or {}).get("scene_id") not in test_scenes]
    test = [r for r in records if (r.get("extra") or {}).get("scene_id") in test_scenes]
    logger.info("proof over %s: %d record(s), %d scene(s) split whole into %d train / %d test",
                records_path, len(records), len(scenes), len(scenes) - len(test_scenes),
                len(test_scenes))

    return {
        "records": len(records),
        "scenes": len(scenes),
        "train_scenes": len(scenes) - len(test_scenes),
        "test_scenes": len(test_scenes),
        "held": sum(1 for r in records if r.get("final_outcome") == "succeeded"),
        "stock": _evaluate(train, test, RANKING_FEATURE_KEYS),
        "geometry": _evaluate(train, test, GEOMETRY_KEYS),
        # The live stock keys plus geometry. Not all 15 + 4: that is 19, past the solver's width,
        # and a constant key contributes nothing to a pairwise model anyway (its delta is 0).
        "combined": _evaluate(train, test, LIVE_STOCK_KEYS + GEOMETRY_KEYS),
        "honesty": {
            "reward_model": "sim_physics_held",
            "bucket": "1 - measured in simulation",
            "caveat": (
                "PhysX outcomes on renderer-exact poses and an exact extrinsic. This shows the "
                "pipeline yields a learnable signal; it is not evidence about a real cell, and no "
                "policy is promotable on it."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m datagen.rl.proof")
    parser.add_argument("--records", default="logs/p5/datasets/v1_proof/rl_records.jsonl")
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    report = run_proof(Path(args.records), holdout=args.holdout)
    print(f"  records {report['records']}  scenes {report['scenes']} "
          f"(train {report['train_scenes']} / test {report['test_scenes']})  held {report['held']}")
    for name in ("stock", "geometry", "combined"):
        block = report[name]
        if not block.get("trainable"):
            print(f"  {name:<9} NOT TRAINABLE: {block.get('reason')}  "
                  f"(groups {block['train']['groups']}, with pairs {block['train']['groups_with_pairs']})")
            continue
        print(f"  {name:<9} train_pairs={block['train_pairs']:<5} test_pairs={block['test_pairs']:<5} "
              f"train_acc={block['train_accuracy']}  test_acc={block['test_accuracy']}")
    print(f"  {report['honesty']['caveat']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  report -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
