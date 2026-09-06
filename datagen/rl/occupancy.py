"""The occupancy gate: does driving the stack over datagen scenes populate the RL features?

Run this before spending physics time. The RL trainers read fifteen feature keys and project a
missing one to ``0.0`` silently, so a run that produces mostly zeros trains on nothing and says
so nowhere.

Two questions, and the second is the one that gets forgotten:

* occupancy: how often is the key present and non-null?
* variance: does it move? A feature present on every row and always ``0.5`` carries exactly as
  much information as one that is absent. Both are worthless to a logistic model, and only the
  first looks fine on a dashboard.

It touches no robot: the arm is a dummy, the scenes are on disk, and no physics runs. What it
measures is whether the feature pipeline is wired, nothing about grasp quality.

    python -m datagen.rl.occupancy --dataset logs/p5/datasets/v1_proof --scenes 12
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, RL_OCCUPANCY_LOG_FILE

__all__ = ["measure_occupancy", "main"]

logger = create_logger("datagen.rl.occupancy", RL_OCCUPANCY_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Everything the two trainers read, in one list. Ranking is a superset of candidate.
_FEATURE_KEYS: tuple[str, ...] = ()


def _feature_keys() -> tuple[str, ...]:
    global _FEATURE_KEYS
    if not _FEATURE_KEYS:
        from src.robot.grasping.rl.ranking_policy import RANKING_FEATURE_KEYS

        _FEATURE_KEYS = tuple(RANKING_FEATURE_KEYS)
    return _FEATURE_KEYS


def build_service(robot_cfg: Any, rig: Any) -> Any:
    """The real composition root, with the multi-camera rig attached to the orchestrator.

    Public because `datagen/rl/collect.py` imports it. The repository's rule is that internals
    live in `_private.py`, and a seam two modules share is not an internal.

    ``from_robot_config`` is deliberately the same entry point a real cell uses, so the features
    come from the actual stack rather than from a reimplementation of it. The two orchestrator
    fields set afterwards are the documented seam for a rig that is not built from
    ``fusion.cameras``: datagen's extrinsics live per scene, and the wrist camera moves with the
    posed arm, so they are not a cell constant and cannot be a calibration artifact.
    """
    from src.robot.execution.autonomous_grasp import AutonomousGraspService
    from src.robot.execution.autonomous_grasp.config import GraspMode
    # Through the factory, not by name, so `robot.grasping.calculator: deep` reaches this sweep
    # instead of silently grading the analytic generator. The caller must have run
    # `preflight_calculator` first: this function is called per scene inside an `except Exception`,
    # which would turn one configuration error into N warnings.
    from src.robot.grasping.calculator_factory import build_calculator

    calculator = build_calculator(
        robot_cfg,
        camera_matrix=rig.primary.intrinsics_mm,
        max_grip_width_mm=robot_cfg.gripper.max_width_mm,
        min_grip_width_mm=robot_cfg.gripper.min_width_mm,
        support_footprint_geometry=(robot_cfg.grasping.geometry.stage == "support_footprint"),
        support_footprint_inflate_mm=robot_cfg.grasping.geometry.inflate_mm,
    )
    service = AutonomousGraspService.from_robot_config(
        robot_cfg,
        calculator=calculator,
        perception=rig.primary,
        frame_resolver=rig.resolvers[rig.primary_view],
        # Auto is not a preference. `watchdog` declares apply_modes ('auto',), so three of the
        # fifteen features (drift_severity, ood_flagged, degraded_mode_active) are unreachable in
        # every other mode. And the mode is not a config key: the loader rejects `grasping.mode`.
        mode=GraspMode.AUTO,
    )
    orchestrator = service.runtime.orchestrator
    orchestrator.multi_camera_perception = rig.rig
    orchestrator.camera_frame_resolvers = rig.resolvers
    return service


def measure_occupancy(
    dataset_dir: Path,
    *,
    scene_limit: int = 12,
    mask_source: str = "pred",
    depth_source: str = "noisy",
    record_path: Path | None = None,
) -> dict[str, Any]:
    """Drive the stack over up to ``scene_limit`` scenes and report what the features did."""
    from src.config.loader import load_robot_config  # noqa: PLC0415

    from datagen.rl.perception import load_scene_rig

    # One call, and no process-global state touched: `set_active_profile` plus `reload_config`
    # would change the active profile and invalidate every other holder's cached tree to read one
    # field.
    robot_cfg = load_robot_config(profile="rl_datagen")

    scenes = sorted(p for p in (dataset_dir / "scenes").iterdir() if p.is_dir())[:scene_limit]
    if not scenes:
        raise SystemExit(f"no scenes under {dataset_dir / 'scenes'}")

    record_path = record_path or (dataset_dir / "rl_occupancy_records.jsonl")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if record_path.exists():
        # The previous records are renamed, never unlinked: this sweep exists to detect a
        # regression, and deleting the last good run first makes the before-and-after comparison
        # impossible. One generation, not a history: the question is what changed since last time,
        # and a directory of timestamped records would be a different feature with a different
        # cost.
        previous = record_path.with_suffix(record_path.suffix + ".previous")
        previous.unlink(missing_ok=True)
        record_path.rename(previous)
        logger.info("kept the previous records as %s", previous.name)

    logger.info("occupancy sweep over %s: %d scene(s), masks=%s, depth=%s, records -> %s",
                dataset_dir, len(scenes), mask_source, depth_source, record_path)
    outcomes: dict[str, int] = defaultdict(int)
    shadow_seen = 0
    candidate_rows = 0
    failures: list[str] = []

    # A sweep with no robot config is a misconfiguration, not a default. Every feature below comes
    # from the actual stack, and the stack needs a robot; without one the loop would raise per scene
    # inside its own `except` and report zeros.
    if robot_cfg is None:
        raise SystemExit("the active profile carries no `robot` config, so no cell can be built")
    # Once, before the first scene. Everything below runs inside `except Exception: continue`, so a
    # selector pointing at a missing or wrong artifact would arrive as N warnings and a report of
    # zeros. Raised here it arrives as itself, before any scene is loaded.
    from src.robot.grasping.calculator_factory import preflight_calculator

    logger.info("grasp calculator: %s", preflight_calculator(robot_cfg))

    for scene_dir in scenes:
        try:
            rig = load_scene_rig(scene_dir, mask_source=mask_source, depth_source=depth_source)
        except Exception as exc:  # noqa: BLE001 (a bad scene must not end the sweep)
            logger.warning("%s: load_scene_rig failed: %s: %s",
                           scene_dir.name, type(exc).__name__, exc)
            failures.append(f"{scene_dir.name}: load_scene_rig: {type(exc).__name__}: {exc}")
            continue
        try:
            service = build_service(robot_cfg, rig)
            # Provenance on every record, because a corpus that cannot say which scene and which
            # fidelity produced a row is a corpus nobody can re-derive. `enable_record_logging`
            # already appends, so one path collects the whole sweep.
            service.enable_record_logging(record_path, provenance={
                "datagen_scene_id": rig.scene_id,
                "datagen_family": rig.family,
                "datagen_primary_view": rig.primary_view,
                "datagen_cameras": list(rig.camera_ids),
                "datagen_mask_source": mask_source,
                "datagen_depth_source": depth_source,
            })
            if service.shadow_router is not None:
                shadow_seen += 1
            report = service.pick()
            outcomes[str(getattr(report, "outcome", "unknown"))] += 1
        except Exception as exc:  # noqa: BLE001
            # The report keeps only the first twelve failures; every one of them is worth a line here,
            # because a sweep whose picks all threw still returns a report full of tidy zeros.
            logger.warning("%s: pick failed: %s: %s", scene_dir.name, type(exc).__name__, exc)
            failures.append(f"{scene_dir.name}: pick: {type(exc).__name__}: {exc}")

    records = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines()
               if line.strip()] if record_path.exists() else []

    stats = _feature_stats(records)
    candidate_rows = sum(len((r.get("extra") or {}).get("rl_candidate_features") or ())
                         for r in records)
    per_candidate = _per_candidate_stats(records)

    # The candidate rows are the number that matters: the pairwise ranker can only use those, and
    # a sweep that produced records but no per-candidate rows is the exact silent hole this module
    # exists to find.
    logger.info("%d record(s), %d candidate row(s) from %d scene(s); outcomes %s; "
                "shadow router built on %d; %d failure(s)",
                len(records), candidate_rows, len(scenes), dict(outcomes), shadow_seen,
                len(failures))
    return {
        "dataset": str(dataset_dir),
        "scenes_attempted": len(scenes),
        "records": len(records),
        "candidate_rows": candidate_rows,
        "shadow_router_built": shadow_seen,
        "outcomes": dict(outcomes),
        "mask_source": mask_source,
        "depth_source": depth_source,
        "features": stats,
        "per_candidate": per_candidate,
        "failures": failures[:12],
        "failure_count": len(failures),
    }


def _per_candidate_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Spread over the per-candidate rows, the only thing a pairwise ranker can use.

    The ranker differences features within a group, and a group is one attempt's candidate list,
    which all come from the same winning segmentation. So a scene-level, pick-level or even
    object-level quantity is identical for every row in the group and differences to exactly zero,
    however well populated it looks. Only a genuinely per-grasp quantity can move the model at
    all.

    Both the policy vector (`features`) and the geometry block are reported side by side, because
    the comparison between them answers what a ranker should train on.
    """
    columns: dict[str, list[float]] = {}
    rows = 0
    for record in records:
        for candidate in (record.get("extra") or {}).get("rl_candidate_features") or ():
            if "features" not in candidate:
                continue  # the tail-aggregate row
            rows += 1
            for block, prefix in (("features", ""), ("geometry", "geom.")):
                for key, value in (candidate.get(block) or {}).items():
                    if isinstance(value, (int, float, bool)):
                        columns.setdefault(prefix + key, []).append(float(value))

    out: dict[str, dict[str, Any]] = {"__rows__": {"rows": rows}}
    for key, values in columns.items():
        n = len(values)
        distinct = len({round(v, 9) for v in values})
        mean = sum(values) / n if n else 0.0
        stdev = math.sqrt(sum((v - mean) ** 2 for v in values) / n) if n else 0.0
        out[key] = {
            "rows": n, "distinct": distinct, "stdev": round(stdev, 6),
            "min": round(min(values), 4) if values else 0.0,
            "max": round(max(values), 4) if values else 0.0,
            "verdict": "constant" if distinct <= 1 else "live",
        }
    return out


def _feature_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-key occupancy and spread, read the way the trainer reads it.

    Deliberately reuses the trainer's own projection rather than reading the keys directly: the
    question is not "is the key somewhere in the record" but "what number does training see", and
    those differ exactly where the bugs live.
    """
    from src.robot.grasping.rl.train_ranking import _record_features

    keys = _feature_keys()
    columns: dict[str, list[float]] = {k: [] for k in keys}
    for record in records:
        row = _record_features(record)
        for key, value in zip(keys, row):
            columns[key].append(float(value))

    out: dict[str, dict[str, Any]] = {}
    for key, values in columns.items():
        n = len(values)
        nonzero = sum(1 for v in values if v != 0.0)
        distinct = len({round(v, 9) for v in values})
        mean = sum(values) / n if n else 0.0
        var = sum((v - mean) ** 2 for v in values) / n if n else 0.0
        out[key] = {
            "rows": n,
            "nonzero": nonzero,
            "nonzero_frac": round(nonzero / n, 4) if n else 0.0,
            "distinct": distinct,
            "mean": round(mean, 6),
            "stdev": round(math.sqrt(var), 6),
            # The verdict, and it is deliberately harsher than "is it present". A constant column is
            # dead weight in a logistic model however well-populated it looks.
            "verdict": (
                "absent" if nonzero == 0
                else "constant" if distinct <= 1
                else "near_constant" if math.sqrt(var) < 1e-9
                else "live"
            ),
        }
    return out


@dataclass(frozen=True, slots=True)
class SweepVerdict:
    """Did this sweep produce features a pairwise ranker could actually use?

    Both halves are counted. A record-level feature can be "live" while every candidate inside the
    record carries the same value, which is the exact state that makes a pairwise ranker
    unlearnable and the state this sweep exists to detect, so the per-candidate table gates the
    verdict too.

    Rows first, because zero rows is not a low score. With no candidate rows every per-candidate
    verdict is vacuous rather than bad, so it is refused by its own sentence rather than by a
    threshold.
    """

    ok: bool
    reason: str
    rows: int
    live_record: int
    features: int
    live_per_candidate: int
    per_candidate_features: int


def sweep_verdict(report: dict[str, Any]) -> SweepVerdict:
    """The gate, as one rule that both the CLI and a library caller read."""
    features = report["features"]
    live = sum(1 for s in features.values() if s["verdict"] == "live")
    per_candidate = dict(report["per_candidate"])
    rows = int(per_candidate.pop("__rows__", {}).get("rows", 0))
    live_pc = sum(1 for s in per_candidate.values() if s["verdict"] == "live")
    counts = {"rows": rows, "live_record": live, "features": len(features),
              "live_per_candidate": live_pc, "per_candidate_features": len(per_candidate)}
    if not rows:
        return SweepVerdict(ok=False, reason=(
            "no candidate row was produced, so the shadow router never populated one and every "
            "per-candidate verdict is vacuous rather than bad"), **counts)
    if live * 2 < len(features):
        return SweepVerdict(ok=False, reason=(
            f"only {live} of {len(features)} record-level feature(s) move"), **counts)
    if live_pc * 2 < len(per_candidate):
        return SweepVerdict(ok=False, reason=(
            f"only {live_pc} of {len(per_candidate)} PER-CANDIDATE feature(s) move, and those are "
            f"the only ones a pairwise ranker can use"), **counts)
    return SweepVerdict(ok=True, reason="", **counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m datagen.rl.occupancy",
        description="P1 gate: are the RL features actually populated by a datagen-driven pick?",
    )
    parser.add_argument("--dataset", default="logs/p5/datasets/v1_proof", help="datagen dataset root")
    parser.add_argument("--scenes", type=int, default=12, help="how many scenes to drive")
    parser.add_argument("--masks", default="pred", choices=("pred", "gt"))
    parser.add_argument("--depth", default="noisy", choices=("noisy", "clean"))
    parser.add_argument("--out", default=None, help="write the report JSON here")
    args = parser.parse_args(argv)

    report = measure_occupancy(
        Path(args.dataset), scene_limit=args.scenes,
        mask_source=args.masks, depth_source=args.depth,
    )

    print(f"  scenes {report['scenes_attempted']}  records {report['records']}  "
          f"candidate rows {report['candidate_rows']}  shadow built {report['shadow_router_built']}")
    print(f"  outcomes: {report['outcomes']}")
    if report["failure_count"]:
        print(f"  FAILURES ({report['failure_count']}):")
        for line in report["failures"]:
            print(f"    {line}")
    print(f"  {'feature':<38} {'verdict':<14} {'nonzero':>8} {'distinct':>9} {'stdev':>12}")
    live = 0
    for key, s in report["features"].items():
        live += s["verdict"] == "live"
        print(f"  {key:<38} {s['verdict']:<14} {s['nonzero_frac']:>8} "
              f"{s['distinct']:>9} {s['stdev']:>12}")
    print(f"  LIVE (record-level): {live}/{len(report['features'])}")

    pc = dict(report["per_candidate"])
    rows = pc.pop("__rows__", {}).get("rows", 0)
    print(f"  --- PER-CANDIDATE ({rows} rows); the only thing a pairwise ranker can use ---")
    print(f"  {'key':<38} {'verdict':<10} {'distinct':>9} {'stdev':>12}  range")
    live_pc = 0
    for key, st in sorted(pc.items(), key=lambda kv: (kv[1]["verdict"] != "live", kv[0])):
        live_pc += st["verdict"] == "live"
        print(f"  {key:<38} {st['verdict']:<10} {st['distinct']:>9} {st['stdev']:>12} "
              f" [{st['min']}, {st['max']}]")
    print(f"  LIVE (per-candidate): {live_pc}/{len(pc)}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  report -> {args.out}")
    verdict = sweep_verdict(report)
    if not verdict.ok:
        print(f"  REFUSED: {verdict.reason}")
    return 0 if verdict.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
