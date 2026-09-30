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

`target`, `refined_grasp` and `refinement` have had no writer since the two-scan refinement left on
2026-09-29, and `verification` only a simulator runner's ground-truth lift since the post-grasp
verification stage left the same day. They stay in the contract so a record logged before then still
reads.

Changing its shape means changing, in the same commit, every reader: the telemetry catalog, the KPI
roll-up, the soak gate, the failure taxonomy, the offline learning dataset builder and the operator
console's history screen, which rolls records up with the same `compute_kpis`.

`extra` is the free-form bag, so nothing is silently dropped: a caller with something the schema does
not name puts it there, which is how per-candidate feature rows, fusion telemetry and provenance travel
without re-freezing the contract. A pick handed looks adds what they came to: `looks_visited`, and on a
wrist camera `looks_fused`, `jaw_faces_seen`, `hand_eye_gap_mm` and `generated_view_deg`; `both_faces`
is written only when the pick asked for it. A pick a hand ended before its recovery picked again carries
`stage` `before_the_re_pick` and `re_pick_refused` `gripper_fault` (since 2026-09-30): its last
`recovery_actions` row names the `next_target` whose pick never ran, and its `final_outcome` is the hand's,
not that action's. The record keeps summaries, positions, scores, counters and outcome strings, never
images, depth maps or full clouds. `to_dict()` and `from_dict()` round-trip with no `repr` strings and no
NumPy or dataclass objects in the JSON.

### Strings that changed

A record keeps the strings it was written with, and the replay layer reads them all, so an old log still
audits and classifies. Where a string was renamed or retired, a record logged before the date carries
the old one:

| Field | Logged before | Carries | Since then |
| --- | --- | --- | --- |
| `extra.decision_reason_code` of a grasp less confident than the threshold | 2026-09-29 | `reobserve_planner_unavailable` | `low_confidence` |
| `extra.decision_action` and `extra.decision_reason_code` around a camera re-observation | 2026-09-29 | `move_camera` with `low_confidence`; `reobserve_budget_exhausted` once the camera moves were used up | never written: the gate no longer moves the camera |
| `extra.watchdog_enforced` beside a `reobserve` recommendation in canary or active mode | 2026-09-29 | `true`, meaning only eligible | `false`: the recommendation is advisory |
| `recovery_actions[].action` | 2026-09-29 | `next_viewpoint` | `rescan`, which it was merged into |
| `mode` | 2026-09-29 | `closed_loop`, `dense_autonomous` | refused: `auto`, `dense_clutter` |
| `final_outcome` of the two-scan refinement | 2026-09-29 | `refinement_failed`, `target_lost_during_refine`, `refinement_diverged` | never written |
| `final_outcome` of the multi-view commit gate | 2026-09-28 | `no_commit_insufficient_fusion` | never written |
| `final_outcome` after a recovery loop that ran an action and had none left | 2026-09-30 | the last pick's own, `no_valid_grasp` among them | `recovery_exhausted`, or `unsafe_recovery_refused` where a recovery motion stopped; a pick that found nothing keeps `no_target` |
| `recovery_actions[]` of a `dense_clutter` push | 2026-09-30 | a `nudge_target` step the recovery loop planned itself | one `nudge_target` row per push whose arm left the look: `outcome` the push's code, `failure_reason` its trigger (`all_collided`, `approach_path_blocked`), `plan_reason` `push_in_the_pick` |

The same `low_confidence` is thus two things by date: beside `move_camera` before 2026-09-29 it asked for a
camera move, and beside `fail_closed` or `grasp_now` since then it names the verdict. Since 2026-09-30 the
service writes `recovery_exhausted` and `unsafe_recovery_refused`, a stopped push among them, so such
records now reach rules that were already there: the KPI roll-up and the soak and records gates count
`recovery_exhausted` as a dead loop, `safety_rejection_rate` counts `unsafe_recovery_refused`, and the
offline recovery trainer reads `recovery_exhausted` as an unknown failure class.

## `LatencyTracker`

| Stage | Field in `extra` | Measures | 95th percentile gate |
| --- | --- | --- | --- |
| `DECISION` | `decision_latency_ms` | the `DecisionEngine.decide` call | 60 ms |
| `RANKING` | `ranking_latency_ms` | the calculator and the scoring blend | 80 ms |
| `FUSION` | `fusion_latency_ms` | the geometry fusion of a frame with the other views: the other cameras' frames in hand, or a wrist pick's earlier looks, to the fused scene | 220 ms |

The service records `DECISION` around `DecisionEngine.decide`. The orchestrator records `RANKING` and
`FUSION` on the call every pick path shares, the ranking of a frame, and the fusion span is taken inside
it, in `BinPickingOrchestrator._fused_scene`. That span leaves out the other cameras' own capture,
detection and segmentation (`acquire_all`), which are perception. It is recorded only on an attempt where
another view entered the fusion (a `fused_view_count` above 0): another camera's, or an earlier look of
a wrist pick. It is absent from a pick where none did: geometry fusion off or standing down, no other
camera delivering a segmented frame, or a wrist pick that stopped at its first look with no fixed camera
fused. Its presence is also the
perception-budget trainer's continue label (`fusion_latency_ms_presence`).

Records logged before 2026-09-28 carry a different quantity under the same name: the whole open-loop
attempt's wall time, stamped whenever the since-removed voxel grid had ingested a view.

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
| Records from a physical cell | run on a physical cell: example 12 writes one record per attempt and ran at a UR10 (CB3); none of its records is kept here |

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
