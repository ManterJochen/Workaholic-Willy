# Grasping telemetry: the frozen contract

The record the pick path writes, and the per-stage clock the latency gate reads. A pure recorder: no
numpy in the JSON, a lossless round trip, and nothing silently dropped.

| Module | Owns |
|---|---|
| [`outcome_logging.py`](outcome_logging.py) | `GraspAttemptRecord`, the `*_metadata_from` serialisers, and `append_jsonl` / `iter_jsonl` / `json_safe` |
| [`latency_tracker.py`](latency_tracker.py) | `LatencyTracker` and `LatencyStage`: per-stage wall-clock spans |

## `GraspAttemptRecord`

This is a frozen contract. Changing it means updating the telemetry catalog and the consumers that
read it in the same change: the KPI roll-up, the soak gate, the failure taxonomy, the offline
reinforcement-learning dataset builder and the operator console's history screen all read this shape.

Four required fields and twelve optional blocks:

```
timestamp, attempt_id, mode, final_outcome                      always present

profile, frame, target, initial_grasp, initial_telemetry,
refined_grasp, refinement, selected_grasp, execution,
verification, recovery_actions, extra
```

`extra` is the free-form bag, and it exists for a stated reason: nothing is silently dropped. A caller
that needs to carry something the schema does not name stashes it there instead of losing it, which
is how the per-candidate feature rows, the fusion telemetry and the record provenance travel without
re-freezing the contract.

Record logging is opt-in and off by default. Setting `robot.grasping.record_log_path` makes every
`pick()` append one record; that path is the source the soak, KPI and offline learning tooling read.

What the record deliberately does not store: images, depth maps and full point clouds. It keeps
summaries, positions, scores, telemetry counters and outcome strings. Replay tooling that needs
pixels references the perception system separately through the frame fingerprint.

It is a pure recorder. This module calls no perception, no robot and no verification; it only
serialises typed reports produced elsewhere. That is what makes replay-quality logs possible with no
hardware in the loop, and what keeps `to_dict()` and `from_dict()` a lossless round trip with no
`repr` strings and no numpy or dataclass leaks in the JSON payload.

## `LatencyTracker`

```
decision  ->  decision_latency_ms    the DecisionEngine.decide call
ranking   ->  ranking_latency_ms     the calculator and scoring blend
fusion    ->  fusion_latency_ms      the multi-view fusion update
```

Those key names are fixed in the telemetry catalog and gated at the 95th percentile by the offline
evaluator, at 60 ms, 80 ms and 220 ms respectively.

The tracker is deliberately passive: it never raises, never blocks the caller and performs no input
or output, and it uses `time.monotonic_ns`, so a clock adjustment does not affect it.

| Rule | Why |
|---|---|
| Stages are named by a small enum | A typo at the call site is a name error, not a silently missed metric |
| A stage never entered reports `None` | The latency gate skips nulls instead of counting them as a pass |
| Re-entering a stage replaces the previous span | Inside a retry loop the contract measures the final, decision-relevant span |
| `snapshot()` returns a fresh dictionary | A caller may mutate it without corrupting the tracker |
| A stage whose context is never entered is omitted entirely | The pipeline stays byte-identical when the performance block is off |

Do not read `fusion_latency_ms` as what fusion cost. It measures the span the caller wrapped, so an
attempt-level span recorded around a whole pick will appear under a stage name and be read as that
stage's cost. Wrap the stage, not the attempt.

## See also

- [`../README.md`](../README.md) for the tier that produces these records
- [`../replay/README.md`](../replay/README.md) for the KPI roll-up, the telemetry catalog and the
  soak gate that read them
- [`../rl/README.md`](../rl/README.md) for the offline trainers, and `check-dataset` for whether a
  given log is trainable at all
- [`../types/README.md`](../types/README.md) for `GraspResult` and `GraspFailureReason`, the reasons
  that end up in `final_outcome`
- [`../../../../api/README.md`](../../../../api/README.md) for the operator console history screen,
  which rolls these up with the same `compute_kpis`
