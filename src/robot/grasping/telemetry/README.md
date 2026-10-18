# Grasp telemetry (`src.robot.grasping.telemetry`)

The record every pick writes, `GraspAttemptRecord`. It records and never acts: no perception, no robot,
no verification, only typed reports serialised from elsewhere. The per-stage clock beside it,
`LatencyTracker`, left on 2026-10-16 with the latency gate it fed (`robot.grasping.performance`); a
record keeps its pick's wall time, `extra.attempt_wall_time_s`.

A cell built from config records every pick, to `robot.grasping.record_log_path` or else `grasp_records.jsonl` in
the log folder (since 2026-10-17); a campaign can name a file of its own, and so can a service built from its parts,
which records nothing otherwise. Read the log back. This runs at a desk on the dummy arm of the `console_dummy`
profile:

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
| a record log | the pick service: `log_record` in [`execution/autonomous_grasp/`](../../execution/autonomous_grasp/README.md) appends one line per pick | `iter_jsonl(path)` | the records, one per line |

`append_jsonl` and `frame_metadata_from` left on 2026-10-18 with the code nothing called; the pick
service writes its lines itself.

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
roll-up, the soak gate, the failure taxonomy and the operator console's history screen, which rolls
records up with the same `compute_kpis`.

`extra` is the free-form bag, so nothing is silently dropped: a caller with something the schema does
not name puts it there, which is how per-candidate feature rows, fusion telemetry and provenance travel
without re-freezing the contract. A pick handed looks adds what they came to: `looks_visited`, and on a
wrist camera `looks_fused`, `jaw_faces_seen`, `hand_eye_gap_mm` and `generated_view_deg`; `both_faces`
is written only when the pick asked for it. The record keeps summaries, positions, scores, counters and outcome strings, never
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
| `recovery_actions[]` of a `dense_clutter` push | 2026-09-30 | a `nudge_target` step the recovery loop planned itself | one `nudge_target` row per push whose arm left the look: `outcome` the push's code, `failure_reason` its trigger (`all_collided`; `approach_path_blocked` until 2026-10-16), `plan_reason` `push_in_the_pick` |
| `final_outcome` of the decision, uncertainty and watchdog gates | 2026-10-16 | `decision_fail_closed`, `decision_recover_pending`, `uncertainty_fail_closed`, `drift_blocked_auto`, `ood_blocked_auto` | never written: the gates left |
| `extra` of the learned, uncertainty, watchdog, latency, adaptation and RL layers | 2026-10-16 | `predicted_success_probability`, `uncertainty_score` and its channels, `drift_severity`, `ood_flagged`, `decision_latency_ms`, `ranking_latency_ms`, `fusion_latency_ms`, `adaptation_mode`, the `rl_*` fields | never written; a record that carries them still audits, unchecked on them |

The same `low_confidence` is thus two things by date: beside `move_camera` before 2026-09-29 it asked for a
camera move, and beside `fail_closed` or `grasp_now` since then it names the verdict. Since 2026-09-30 the
service writes `recovery_exhausted` and `unsafe_recovery_refused`, a stopped push among them, so such
records now reach rules that were already there: the KPI roll-up and the soak and records gates count
`recovery_exhausted` as a dead loop, `safety_rejection_rate` counts `unsafe_recovery_refused`, and the
offline recovery trainer read `recovery_exhausted` as an unknown failure class until it left with the RL
tier on 2026-10-16.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ValueError` from `GraspAttemptRecord` | an empty `attempt_id`, `mode` or `final_outcome`, or a malformed field | fill the four required fields |
| `ValueError` from `iter_jsonl` | a malformed line; the message carries the line number | repair or drop that line |

## Status

| Capability | Evidence |
| --- | --- |
| Records | measured in simulation: the Isaac runners write records with `--record-log` |
| Records from a physical cell | run on a physical cell: example 12 writes one record per attempt and ran at a UR10 (CB3); none of its records is kept here |

The gate over these records, `python -m src.robot.grasping.replay --records-gate`, judges a real log and
can fail; the synthetic soak that passed by construction left on 2026-10-16.

## Files

| File | Holds |
| --- | --- |
| [`outcome_logging.py`](outcome_logging.py) | `GraspAttemptRecord`, the `profile_`, `grasp_` and `execution_metadata_from` serialisers, `iter_jsonl`, `json_safe` |

## Details

- [`replay/`](../replay/README.md) for the KPI roll-up, the telemetry catalog and the soak gate.
- [`types/`](../types/README.md) for `GraspResult` and the typed reasons behind a failed attempt.
- [Guide 05, record logging](../../../../docs/guide/05-pick-loop.md) for what lands in a line, and
  [the console](../../../../api/README.md) for the history screen.
- Tests: `tests/robot/grasping/telemetry/test_grasp_outcome_logging.py`, `tests/robot/execution/autonomous_grasp/test_k1_record_logging.py`,
  `tests/robot/grasping/telemetry/test_u0_telemetry_contract.py`, `tests/robot/grasping/telemetry/test_record_schema_evolution.py`.
