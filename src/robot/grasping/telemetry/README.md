# Grasp telemetry (`src.robot.grasping.telemetry`)

The record every pick writes, `GraspAttemptRecord`, and the per-stage clock the latency gate reads. It
records and never acts: no perception, no robot, no verification, only typed reports serialised from
elsewhere.

Recording is off by default. Turn it on for a run, or for a cell with `robot.grasping.record_log_path`,
and read the log back. This runs at a desk on the dummy arm of the `console_dummy` profile:

```python
from willy import Cell, PickRun, Recording, load_tree
from src.robot.grasping.telemetry.outcome_logging import iter_jsonl

cell = Cell.rehearsal(load_tree("console_dummy").robot)          # a dummy arm and a synthetic scene
print(PickRun.from_cell(cell, runs=2, recording=Recording.to_file("logs/attempts.jsonl")).execute())

for record in iter_jsonl("logs/attempts.jsonl"):                # one GraspAttemptRecord per pick
    print(record.attempt_id, record.mode, record.final_outcome)
```

The same log feeds the KPI roll-up and the gates. `--records` rolls it up and never fails; `--records-gate`
compares it against thresholds and can fail:

```bash
python -m src.robot.grasping.replay --records logs/attempts.jsonl
```

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `GraspAttemptRecord` | the pick service, one per `pick()` | `to_dict()`, `to_json()`, `from_dict()` | a lossless round trip |
| a record log | `append_jsonl(record, path)` | `iter_jsonl(path)` | the records, one per line |
| `LatencyTracker` | the pick service, one per pick | `with tracker.span(LatencyStage.RANKING):` then `snapshot()` | `{field name: milliseconds}` |

## `GraspAttemptRecord`, a frozen contract

```
always present   timestamp, attempt_id, mode, final_outcome
optional         profile, frame, target, initial_grasp, initial_telemetry, refined_grasp, refinement,
                 selected_grasp, execution, verification, recovery_actions, extra
```

Changing its shape means changing, in the same commit, every reader: the telemetry catalog, the KPI
roll-up, the soak gate, the failure taxonomy, the offline learning dataset builder and the operator
console's history screen, which rolls records up with the same `compute_kpis`.

`extra` is the free-form bag, so nothing is silently dropped: a caller with something the schema does
not name puts it there, which is how per-candidate feature rows, fusion telemetry and provenance travel
without re-freezing the contract. The record keeps summaries, positions, scores, counters and outcome
strings, never images, depth maps or full clouds. `to_dict()` and `from_dict()` round-trip with no
`repr` strings and no NumPy or dataclass objects in the JSON.

## `LatencyTracker`

| Stage | Field in `extra` | Measures | 95th percentile gate |
| --- | --- | --- | --- |
| `DECISION` | `decision_latency_ms` | the `DecisionEngine.decide` call | 60 ms |
| `RANKING` | `ranking_latency_ms` | the calculator and the scoring blend | 80 ms |
| `FUSION` | `fusion_latency_ms` | the multi-view fusion update | 220 ms |

The offline evaluator applies the gates. The tracker does no input or output and uses
`time.monotonic_ns`, so a clock adjustment does not affect it.

| Rule | Why |
| --- | --- |
| Stages are an enum | a typo at the call site is a name error, not a silently missed metric |
| A stage never entered is absent from `snapshot()`, and `get()` gives `None` | the gate skips it instead of counting a pass |
| Re-entering a stage replaces the span | inside a retry loop the final, decision-relevant span counts |
| A span that raises is still recorded | the tracker measures real cost, not successful paths alone |
| `snapshot()` returns a fresh dictionary | a caller may change it without corrupting the tracker |

A span measures what the caller wrapped. Wrap the stage, not the whole attempt, or the attempt's time
is read as that stage's cost.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ValueError` from `GraspAttemptRecord` | an empty `attempt_id`, `mode` or `final_outcome`, or a malformed field | fill the four required fields |
| `ValueError` from `iter_jsonl` | a malformed line; the message carries the line number | repair or drop that line |
| `TypeError` from `LatencyTracker.span` | anything but a `LatencyStage` | name the stage from the enum |

## Status

| Capability | Evidence |
| --- | --- |
| Records and the latency clock | measured in simulation: the Isaac runners write records with `--record-log` |
| Records from a physical cell | never touched hardware: no physical pick has run, so no record comes from one |

The soak gate over these records, `python -m src.robot.grasping.replay --soak-report`, is a synthetic
contract self-check: it proves the telemetry and KPI pipeline is consistent, not that grasps succeed.

## Files

| File | Holds |
| --- | --- |
| [`outcome_logging.py`](outcome_logging.py) | `GraspAttemptRecord`, the `*_metadata_from` serialisers, `append_jsonl`, `iter_jsonl`, `json_safe` |
| [`latency_tracker.py`](latency_tracker.py) | `LatencyTracker`, `LatencyStage`, `STAGE_FIELD_NAMES` |

## Details

- [`replay/`](../replay/README.md) for the KPI roll-up, the telemetry catalog and the soak gate, and
  [`rl/`](../rl/README.md) for the offline trainers and `check-dataset`.
- [`types/`](../types/README.md) for `GraspResult` and the typed reasons behind a failed attempt.
- [Guide 05, record logging](../../../../docs/guide/05-pick-loop.md) for what lands in a line, and
  [the console](../../../../api/README.md) for the history screen.
- Tests: `tests/test_grasp_outcome_logging.py`, `tests/test_k1_record_logging.py`,
  `tests/test_u0_telemetry_contract.py`, `tests/test_record_schema_evolution.py`,
  `tests/test_u10_runtime_slo_gate.py`.
