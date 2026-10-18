# Record logs, KPIs and the soak gate (`src/robot/grasping/replay`)

Reads the record log a cell writes, one `GraspAttemptRecord` per JSONL line, rolls up the KPIs, runs
the soak gate over it and classifies its failures. It is offline: nothing in the pick path imports it,
and it needs the standard library and PyYAML only.

```python
from src.robot.grasping.replay import RecordLog, SoakGate

rollup = RecordLog.from_jsonl("picks.jsonl").kpis()
print(rollup.render())                  # the rates, and the ones these records cannot measure
verdict = SoakGate.over_records("picks.jsonl").evaluate()
print(verdict.render())                 # what it judged, and what it is evidence of
raise SystemExit(verdict.exit_code)     # 0 passes, 1 refused, 2 unreadable
```

A campaign writes such a log with `PickRun.from_cell(..., recording=Recording.to_file("picks.jsonl"))`
([12_pick_with_the_camera.py](../../../../examples/real_robot/12_pick_with_the_camera.py)), and every cell built
from config writes one, to `robot.grasping.record_log_path` or else `grasp_records.jsonl` in the log folder (since
2026-10-17). The same from a shell:

```bash
python -m src.robot.grasping.replay --records picks.jsonl         # the KPIs; exit 0 sound, 2 not
python -m src.robot.grasping.replay --records-gate picks.jsonl    # the gate over a real log, which can fail
```

The operator console rolls up its KPIs through the same `compute_kpis`, so the console and the command
line cannot disagree.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `RecordLog` | `from_jsonl(path)`, `from_records(records)` | `kpis()` | `KpiRollup`: the rates, `sound`, the telemetry offenders |
| `SoakGate` | `over_records`, `over_sim_records` | `evaluate()` | `SoakVerdict`: `passes`, the keys judged, `exit_code` |

The two `SoakGate` doors differ in what the verdict is evidence of, and the door is recorded on the
verdict as its `SoakSource`. `over_records` judges a real log and can fail. `over_sim_records` is real
physics but not representative of hardware, so the pick rate is reported and not gated.

Every gate key is judged from the records alone, and `GateKeyStatus` says whether it passed. A rate over
records that lack the fields it reads is unfounded rather than wrong, and `KpiRollup.sound` says which.
The free functions `compute_kpis`, `audit_records` and `unmeasurable_kpis` are exported beside the nouns.

## The command line

Modes are mutually exclusive, and every mode prints JSON to stdout.

| Flag | What it does | Exit |
| --- | --- | --- |
| `--records <log>` | the KPIs and the telemetry audit of a real log | 0, or 2 on offenders |
| `--records-gate <log>` | the soak gate over a real log | 0 or 1, 2 if unreadable |
| `--sim-soak-report <log>` | the same gate over a simulation log, its report written to `logs/u12/sim_soak_report.json` or `--out`; the floor is 300 attempts unless `--sim-min-attempts` says otherwise | 0 or 1, 2 if unreadable |
| `--failure-taxonomy <log...> --out <file>` | classifies failures and writes a JSON report | 0, or 2 if a log is missing |

The gate holds six keys: at least 2000 attempts, no untyped outcome, no pick with more than 8 recovery
actions, no record missing the telemetry its outcome requires, no `extra` field of the wrong type, and a
dead-loop rate of at most 0.005. The numbers are constants in [soak.py](soak.py).

`--records` prints an `unmeasurable` block beside the KPIs. `false_positive_grasp_rate` is always in
it, because nothing on this stack writes the field it divides by, and `first_attempt_success_rate`,
`dense_recovery_success_rate` and `median_cycle_time_s` join it when the log carries no recovery action
or no cycle time. A withheld rate is not printed as the 0.0 an empty denominator computes to.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| exit 2, `UNREADABLE` | the log is missing or does not parse | point at the log a run wrote |
| exit 1 with violations | the records break a gate threshold, `min_attempts` 2000 among them | read the violations the verdict prints |
| exit 2 on `--records` | records lack the telemetry fields their outcome requires | fix the producer the offenders name |
| an argument error | `--failure-taxonomy` without `--out` | pass the output path |

## Status

Everything here is batch over JSONL, with no streaming KPI. The synthetic soak (`--soak`,
`--soak-report`), the canonical packs, the baseline report, the latency and watchdog evaluations and the
guarded adaptation left on 2026-10-16 (the owner's cleanup): they judged invented picks or the layers
that left the same day, and said nothing about grasping.

## Traps

- A record logged before 2026-09-29 may carry strings renamed or retired since: a low-confidence decision
  reads `reobserve_planner_unavailable` where a current one reads `low_confidence`. The audits require a
  decision's keys, never its values, so both audit; the [telemetry README](../telemetry/README.md) lists
  every such string. A record of the layers that left on 2026-10-16 still audits by its outcome's string.
- The failure taxonomy is at version 3 (`TAXONOMY_VERSION`). Version 2 (2026-09-29) rewrote the
  recommendations that advised removed features; version 3 rewrote the `occlusion_misread` one for the
  fused wrist looks: declare a wrist camera's looks from the sides the clutter leaves open, the part's
  open side first, or fuse a second camera; a wrist pick fuses each look with the ones before, stops at
  the first valid and certain grasp, and takes one generated view only as the last resort. The
  classification is unchanged. A report of an older version keeps its old recommendation wording:
  re-run `--failure-taxonomy` over its logs for the current one.

## Files

| File | Holds |
| --- | --- |
| `runs.py` | `RecordLog`, `SoakGate` and their reports; the command-line modes call these |
| `kpi.py` | `compute_kpis`, `KpiSummary`, and the rates a record set cannot measure |
| `telemetry_catalog.py` | the fields each outcome requires, the extra-field type contract, the audits |
| `soak.py` | the soak gate's thresholds over a record log |
| `failure_taxonomy.py` | root-cause classification and its report |
| `soak_cli.py`, `__main__.py` | the command line |

## Details

- [grasping/](../README.md): the pick stack whose records this reads, and the KPI triage
- [telemetry/](../telemetry/README.md): the frozen record contract
- [api/](../../../../api/README.md): the operator console, which rolls up KPIs with `compute_kpis`
