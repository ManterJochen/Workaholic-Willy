"""Top-k candidates per scene, executed in physics, written out as grasp records.

The occupancy sweep establishes that the features arrive; this attaches a measured outcome to
each of them.

Why top-k and not just the executed one. The pairwise ranker forms success/fail pairs within a
group, and a group is one scene. One execution per scene yields one row, no pair, and a training
set that is structurally empty. Executing candidates the deterministic policy would not have
chosen is exactly what off-policy data is, and it is the only way the pairs exist at all.

Why physics and not the calculator's own verdict. `valid`/`reason` in `grasp_eval.jsonl` is the
GraspCalculator's opinion; training the ranker on it teaches it to imitate the filter it is meant
to improve. `held` is a measurement. The harness that produces it carries its own controls
(`jaw_is_solid`, positive, negative, repeat) and hard-fails a run whose gripper cannot grip: keep
that, it is what makes the numbers a statement about the grasp rather than about the harness.

Frames are checked, not assumed. Datagen's analytic labels sit in the same world millimetres as
`settled_poses_mm_xyzw`, and the stack returns candidates in BASE, the same frame. A silent frame
mismatch here would produce physics trials that are individually plausible and collectively
meaningless.

    python -m datagen.rl.collect --dataset logs/p5/datasets/v1_proof --scenes 12 --top-k 5 --dry-run
    python -m datagen.rl.collect --dataset logs/p5/datasets/v1_proof --scenes 12 --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, RL_COLLECT_LOG_FILE

__all__ = ["collect_trials", "write_records", "main"]

logger = create_logger("datagen.rl.collect", RL_COLLECT_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Physics `source` prefix. The field is free text ("label" | "valid" | "rejected:<reason>"), so
#: prefixing with the candidate id makes the join back to the feature row exact rather than positional.
_SOURCE_PREFIX = "rl"


def _candidate_instance_id(frame: Any, target_index: int | None) -> int | None:
    """Which datagen object the winning segmentation is, or ``None`` when it cannot be established.

    ``None`` must drop the scene rather than default to 0: physics grades the trial against a specific
    instance (`check_ready`), so a wrong id produces a confidently wrong measurement.
    """
    if target_index is None:
        return None
    segmentations = getattr(frame, "segmentations", ())
    if not (0 <= target_index < len(segmentations)):
        return None
    instance_id = getattr(segmentations[target_index], "instance_id", None)
    return int(instance_id) if isinstance(instance_id, int) else None


def collect_trials(
    dataset_dir: Path,
    *,
    scene_limit: int = 12,
    top_k: int = 5,
    mask_source: str = "pred",
    depth_source: str = "noisy",
) -> tuple[list[Any], dict[str, dict[str, Any]], dict[str, int]]:
    """Drive the stack over scenes and return (physics trials, candidate index, counters)."""
    from src.config.loader import load_robot_config  # noqa: PLC0415

    from datagen.grasps.physics import PhysicsTrial
    from datagen.rl.occupancy import build_service
    from datagen.rl.perception import load_scene_rig

    # One call, and no process-global state touched: `set_active_profile` plus `reload_config`
    # would change the active profile and invalidate every other holder's cached tree to read one
    # field.
    robot_cfg = load_robot_config(profile="rl_datagen")

    scenes = sorted(p for p in (dataset_dir / "scenes").iterdir() if p.is_dir())[:scene_limit]
    trials: list[Any] = []
    index: dict[str, dict[str, Any]] = {}
    counters = {"scenes": 0, "scenes_with_candidates": 0, "candidates": 0, "dropped_no_instance": 0}
    # The fidelity pair is the difference between features that carry signal and features that
    # are constants, so it belongs in the record of the run rather than only in the argument list.
    logger.info("collecting over %s: %d scene(s), top-%d, masks=%s, depth=%s",
                dataset_dir, len(scenes), top_k, mask_source, depth_source)

    # A sweep with no robot config is a misconfiguration, not a default. Every feature below comes
    # from the actual stack, and the stack needs a robot; without one the loop would raise per scene
    # inside its own `except` and report zeros.
    if robot_cfg is None:
        raise SystemExit("the active profile carries no `robot` config, so no cell can be built")
    # The same preflight as the occupancy sweep, and for the same reason: `build_service` runs
    # inside `except Exception: continue`, so a misconfigured calculator would look like nine bad
    # scenes rather than one bad config line.
    from src.robot.grasping.calculator_factory import preflight_calculator

    logger.info("grasp calculator: %s", preflight_calculator(robot_cfg))

    for scene_dir in scenes:
        counters["scenes"] += 1
        try:
            rig = load_scene_rig(scene_dir, mask_source=mask_source, depth_source=depth_source)
            service = build_service(robot_cfg, rig)
            orchestrator = service.runtime.orchestrator
            frame = rig.primary.acquire()
            result, target_index, _decision = orchestrator._best_result_over_segmentations(frame)
        except Exception as exc:  # noqa: BLE001 (one bad scene must not end the sweep)
            # The sweep is meant to survive a bad scene; it is not meant to hide how many it survived.
            # Silent, a run over twelve scenes that skipped nine looks exactly like a small dataset.
            logger.warning("%s skipped: %s: %s", scene_dir.name, type(exc).__name__, exc)
            continue
        if result is None or not result.candidates:
            continue
        instance_id = _candidate_instance_id(frame, target_index)
        if instance_id is None:
            counters["dropped_no_instance"] += 1
            continue
        counters["scenes_with_candidates"] += 1

        for rank, candidate in enumerate(result.candidates[:top_k]):
            cid = f"{rig.scene_id}_c{rank}"
            metadata = dict(getattr(candidate, "metadata", None) or {})
            _shadow_raw = metadata.get("shadow")
            shadow: dict = _shadow_raw if isinstance(_shadow_raw, dict) else {}
            trials.append(PhysicsTrial(
                scene_id=rig.scene_id,
                instance_id=instance_id,
                source=f"{_SOURCE_PREFIX}:{cid}",
                position_mm=tuple(np.asarray(candidate.position, dtype=np.float64).tolist()),
                approach=tuple(np.asarray(candidate.approach, dtype=np.float64).tolist()),
                closing_axis=tuple(np.asarray(candidate.axis, dtype=np.float64).tolist()),
                width_mm=float(candidate.grip_width_mm),
                family=rig.family,
                config="rl_datagen",
            ))
            index[cid] = {
                "scene_id": rig.scene_id,
                "family": rig.family,
                "instance_id": instance_id,
                "rank": rank,
                "executed_by_policy": rank == 0,
                # The policy vector, exactly as the shadow would have logged it: flat, so
                # `_record_features` finds it in `extra` without a nested lookup.
                "features": _policy_features(metadata, shadow),
                "geometry": _geometry(candidate),
            }
            counters["candidates"] += 1
    logger.info("%d trial(s) from %d/%d scene(s); %d scene(s) produced no instance id",
                counters["candidates"], counters["scenes_with_candidates"], counters["scenes"],
                counters["dropped_no_instance"])
    return trials, index, counters


def _policy_features(metadata: dict, shadow: dict) -> dict[str, float]:
    """The 15 RANKING_FEATURE_KEYS for one candidate, same precedence the shadow aggregator uses."""
    from src.robot.grasping.rl.ranking_policy import RANKING_FEATURE_KEYS

    rl_md = metadata.get("rl_state_features")
    rl_md = rl_md if isinstance(rl_md, dict) else {}
    out: dict[str, float] = {}
    for key in RANKING_FEATURE_KEYS:
        for source in (rl_md, shadow, metadata):
            if key in source:
                value = source[key]
                out[key] = float(value) if isinstance(value, (int, float, bool)) else 0.0
                break
        else:
            out[key] = 0.0
    return out


def _geometry(candidate: Any) -> dict[str, float]:
    from src.robot.grasping.loop._shadow_aggregator import _candidate_geometry

    return _candidate_geometry(candidate)


def write_records(
    index: dict[str, dict[str, Any]],
    physics_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Join physics outcomes back onto the candidate rows and emit one grasp record per candidate.

    One record per candidate, not per pick, and that is the point: the ranker needs several rows
    in a group, each with its own features and its own outcome.

    ``extra.scene_id`` is set explicitly because ``_group_key`` looks for ``scene_family_id`` then
    ``scene_id``: a provenance key named anything else (``datagen_scene_id``) falls through to the
    attempt-id prefix and the pairs never form.

    A failure is left unclassified on purpose. The taxonomy tokens name causes
    (``empty_air_grasp``, ``slip_after_grasp``, ...) and physics measured "did not hold", not why.
    ``derive_outcome_class`` maps unclassified to not-success, so pairs still form correctly;
    inventing a cause to fill the field would put a guess into a contract that is read as a
    measurement.
    """
    from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord

    outcomes: dict[str, dict[str, Any]] = {}
    controls: dict[str, Any] = {}
    for line in physics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "controls" in row:
            controls = row["controls"]
            continue
        source = str(row.get("source") or "")
        if source.startswith(f"{_SOURCE_PREFIX}:"):
            outcomes[source.split(":", 1)[1]] = row

    written = 0
    held = 0
    refused = 0
    stamp = time.time()
    with out_path.open("w", encoding="utf-8") as handle:
        for cid, entry in sorted(index.items()):
            outcome = outcomes.get(cid)
            if outcome is None:
                continue
            note = str(outcome.get("note") or "")
            if note.startswith("refused"):
                # The harness refused to grade this trial (a drifted scene, a degenerate axis).
                # Kept out of the dataset entirely rather than recorded as a failure: a grasp
                # graded against a scene that moved is not evidence about the grasp, and the
                # harness says so.
                refused += 1
                continue
            is_held = bool(outcome.get("held"))
            held += int(is_held)
            extra: dict[str, Any] = {
                "scene_id": entry["scene_id"],
                "scene_family_id": f"{entry['family']}:{entry['instance_id']}",
                "datagen_family": entry["family"],
                "datagen_instance_id": entry["instance_id"],
                "candidate_rank": entry["rank"],
                "executed_by_policy": entry["executed_by_policy"],
                "physics_rise_mm": outcome.get("rise_mm"),
                "physics_note": note,
                # Honesty stamps: what produced the reward, and what it is not.
                "reward_model": "sim_physics_held",
                "reward_interpretation": (
                    "held = the object came up with the gripper in PhysX. Bucket (1): measured in "
                    "simulation, on renderer-exact poses and an exact extrinsic. It is not a real cell."
                ),
                "dataset_origin": "datagen_rl_collect",
                **entry["features"],
                **{f"geom_{k}": v for k, v in entry["geometry"].items()},
            }
            record = GraspAttemptRecord(
                timestamp=stamp,
                attempt_id=cid,
                mode="auto",
                final_outcome="succeeded" if is_held else "failed",
                extra=extra,
            )
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            written += 1

    # `refused` is not a failure count: the harness declined to grade those trials. The two are
    # the pair that has to be read together, which is exactly why they are logged together.
    logger.info("%d record(s) -> %s: %d held, %d refused (of %d indexed candidates, %d physics rows)",
                written, out_path, held, refused, len(index), len(outcomes))
    return {
        "records": written, "held": held, "refused": refused,
        "candidates_indexed": len(index), "physics_rows": len(outcomes),
        "harness_controls": controls,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m datagen.rl.collect",
        description="Execute top-k candidates per scene in physics and write grasp records.",
    )
    parser.add_argument("--dataset", default="logs/p5/datasets/v1_proof")
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--masks", default="pred", choices=("pred", "gt"))
    parser.add_argument("--depth", default="noisy", choices=("noisy", "clean"))
    # The referee is a choice, not an omission: MuJoCo exists so a customer without a 47 GB
    # NVIDIA-only install can still grade their own trials, so this path names the engine
    # rather than taking `run_physics_sample`'s default.
    parser.add_argument("--engine", default="isaac", choices=("isaac", "mujoco"),
                        help="which simulator grades the trials")
    parser.add_argument(
        "--join-only", action="store_true",
        help="skip physics and join an EXISTING rl_physics.jsonl. The candidate index is rebuilt "
             "deterministically from the same scenes, so a run whose simulator shut down before the "
             "join can be completed without paying for the trials again.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="build the trials and print them; start no simulator. Run this FIRST; a crashed Isaac "
             "hangs, and there is no reason to spend one on a malformed trial list.",
    )
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset)
    trials, index, counters = collect_trials(
        dataset_dir, scene_limit=args.scenes, top_k=args.top_k,
        mask_source=args.masks, depth_source=args.depth,
    )
    print(f"  scenes {counters['scenes']}  with candidates {counters['scenes_with_candidates']}  "
          f"dropped(no instance) {counters['dropped_no_instance']}  trials {len(trials)}")
    if not trials:
        print("  no trials to run")
        return 1

    for trial in trials[:3]:
        print(f"    {trial.scene_id} inst={trial.instance_id} src={trial.source} "
              f"pos={[round(v, 1) for v in trial.position_mm]} w={trial.width_mm:.1f}")
    scenes_covered = len({t.scene_id for t in trials})
    print(f"    ... {len(trials)} trials over {scenes_covered} scenes "
          f"({len(trials) / max(1, scenes_covered):.1f} per scene)")

    if args.dry_run:
        print("  --dry-run: no simulator started")
        return 0

    if args.join_only:
        summary = write_records(
            index, dataset_dir / "rl_physics.jsonl", dataset_dir / "rl_records.jsonl")
        print(f"  records {summary['records']}  held {summary['held']}  refused {summary['refused']}")
        print(f"  harness controls: {summary['harness_controls']}")
        return 0

    from datagen.grasps.physics import run_physics_sample

    # `--engine` is passed through, so "generate your own data" stays a claim a customer without a
    # 47 GB NVIDIA-only install can act on.
    print(f"  starting physics for {len(trials)} trials (~1 s each plus scene builds)", flush=True)
    run_physics_sample(dataset_dir, trials=trials, out_name="rl_physics.jsonl", engine=args.engine)
    summary = write_records(
        index, dataset_dir / "rl_physics.jsonl", dataset_dir / "rl_records.jsonl")
    print(f"  records {summary['records']}  held {summary['held']}  refused {summary['refused']}")
    print(f"  harness controls: {summary['harness_controls']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
