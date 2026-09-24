# Record logs, KPIs and the soak gate (`src/robot/grasping/replay`)

Reads the record log a cell writes, one `GraspAttemptRecord` per JSONL line, rolls up the KPIs, runs
the soak gates, and scores latency, drift, out-of-distribution detection and the failure taxonomy. It
is offline: nothing in the pick path imports it, and it needs the standard library and PyYAML only.

```python
from src.robot.grasping.replay import RecordLog, SoakGate

rollup = RecordLog.from_jsonl("picks.jsonl").kpis()
print(rollup.render())                  # the rates, and the ones these records cannot measure
verdict = SoakGate.over_records("picks.jsonl").evaluate()
print(verdict.render())                 # what it judged, and what it is evidence of
raise SystemExit(verdict.exit_code)     # 0 passes, 1 refused, 2 unreadable
```

A campaign writes such a log with `PickRun.from_cell(..., recording=Recording.to_file("picks.jsonl"))`
([11_pick_with_the_camera.py](../../../../examples/real_robot/11_pick_with_the_camera.py)), and a cell writes one
when `robot.grasping.record_log_path` is set. The same from a shell:

```bash
python -m src.robot.grasping.replay --records picks.jsonl         # the KPIs; exit 0 sound, 2 not
python -m src.robot.grasping.replay --records-gate picks.jsonl    # the gate over a real log, which can fail
python -m src.robot.grasping.replay --soak-report                 # the synthetic self-check
```

The operator console rolls up its KPIs through the same `compute_kpis`, so the console and the command
line cannot disagree.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `RecordLog` | `from_jsonl(path)`, `from_records(records)` | `kpis()` | `KpiRollup`: the rates, `sound`, the telemetry offenders |
| `SoakGate` | `over_records`, `over_sim_records`, `over_synthetic`, `over_canonical_packs` | `evaluate()` | `SoakVerdict`: `passes`, `judged`, `exit_code` |
| `Baseline` | `canonical()` | `measure()` | `BaselineMeasurement`, with `write(path)` |

The four `SoakGate` doors differ in what the verdict is evidence of, and the door is recorded on the
verdict as its `SoakSource`. `over_records` judges a real log and can fail. `over_sim_records` is real
physics but not representative of hardware, so the pick rate is reported and not gated.
`over_synthetic` invented its own grasps. `over_canonical_packs` passes by construction: it is a
tripwire for a changed telemetry contract.

A gate key that could not be judged has not passed: `GateKeyStatus` has three states, and
`verdict.judged` counts the keys actually decided. A rate over records that lack the fields it reads is
unfounded rather than wrong, and `KpiRollup.sound` says which. The free functions `compute_kpis`,
`audit_records` and `unmeasurable_kpis` are exported beside the nouns.

## The command line

Modes are mutually exclusive, and every mode prints JSON to stdout.

| Flag | What it does | Exit |
| --- | --- | --- |
| `--records <log>` | the KPIs and the telemetry audit of a real log | 0, or 2 on offenders |
| `--records-gate <log>` | the record-intrinsic soak thresholds over a real log | 0 or 1, 2 if unreadable |
| `--sim-soak-report <log>` | the same gate over a simulation log, with a report | 0 or 1, 2 if unreadable |
| `--soak` | a cheap synthetic soak against `config/robot/kpi_thresholds.yaml` | 0 or 1 |
| `--soak-report` | the locked synthetic gate, written to `logs/u12/soak_report.json` | 0 or 1 |
| `--baseline-report` | the baseline report, written to `docs/baselines/u_plus_baseline_v1.json` | 0, or 2 on offenders |
| `--regenerate-canonical` | rewrites the canonical packs and their manifest in `tests/data/replay/` | 0 |
| `--failure-taxonomy <pack...> --out <file>` | classifies failures and writes a JSON report | 0, or 2 if a pack is missing |
| `--watchdog-eval`, `--slo-gate` | drift and OOD precision and recall; per-stage p95 against 60, 80 and 220 ms | 0 or 2 |
| `--adaptation-plan`, `--adaptation-verify`, `--adaptation-apply`, `--adaptation-rollback` | guarded adaptation | 0 or 2 |

`--out` names the output file of every mode that writes one; `--baseline-out` is an older spelling of
the same flag, and passing both with different paths is an error. The canonical packs and the baseline
report ship with the repository: `--regenerate-canonical` rewrites the committed packs, and
`--baseline-report` without `--out` rewrites the committed baseline.

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
| an argument error | `--out` and `--baseline-out` disagree, or `--failure-taxonomy` without `--out` | pass one output path |

## Status

`--soak-report` is a synthetic self-check, not a quality gate. It composes seed-locked scenarios whose
outcome distribution is authored, so a pass proves the telemetry, KPI, taxonomy, SLO and watchdog
pipeline is consistent and says nothing about grasp quality; the report's own `provenance` block says
so in its `does_not_measure` field. For a signal that can fail, run `--records-gate` or
`--sim-soak-report` over a real log. Everything here is batch over JSONL, with no streaming KPI.

## Traps

- `--soak` and `--soak-report` are different gates, with different scenarios and thresholds.
  `--soak-report` is the locked one.
- The synthetic soak stream stamps no evidence flag, so all 168 of its failures come back
  `unclassified` and the report's taxonomy coverage reads 0.0. The gate key
  `failure_taxonomy_classifier_pass` is graded against the committed labeled pack instead (agreement
  1.0, coverage 0.968).
- `--adaptation-apply` compares a baseline report from before the apply with one from after. Without
  `--post-apply-report PATH`, a report measured under the applied overlay, the two are equal, the
  deltas are zero and the rollback guardrail cannot fire. It logs a warning saying so.
- Wall-time p95 for `closed_loop` is `None`, because the canonical packs hold no `closed_loop` records.
  The `easy` wall-time budget is the only wall-time gate, and it is skipped rather than failed when its
  side is missing.

## Files

| File | Holds |
| --- | --- |
| `runs.py` | `RecordLog`, `SoakGate`, `Baseline` and their reports; the command-line modes call these |
| `kpi.py` | `compute_kpis`, `KpiSummary`, and the rates a record set cannot measure |
| `telemetry_catalog.py` | the fields each outcome requires, the extra-field type contract, the audits |
| `soak.py` | the seed-locked synthetic soak generator and the consolidated gate (`build_soak_report`) |
| `baseline_report.py` | per-pack KPI, SLO and telemetry blocks, and `compare_kpi_deltas` |
| `canonical_datasets.py` | the canonical JSONL packs and their SHA-256 manifest |
| `slo_eval.py`, `watchdog_eval.py` | per-stage latency against the locked budgets; drift and OOD precision and recall |
| `failure_taxonomy.py` | root-cause classification and its grading against the labeled pack |
| `adaptation.py`, `adaptation_io.py`, `adaptation_cli.py` | guarded adaptation: plan, verify, apply as an overlay with an audit log, roll back |
| `presets.py` | the grasping preset loader and its schema check |
| `soak_cli.py`, `__main__.py` | the command line |

## Details

- [grasping/](../README.md): the pick stack whose records this reads, and the KPI triage
- [telemetry/](../telemetry/README.md): the frozen record contract
- [rl/](../rl/README.md): the layer that reads the `rl_*` telemetry fields catalogued here
- [api/](../../../../api/README.md): the operator console, which rolls up KPIs with `compute_kpis`
