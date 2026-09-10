# Replay (`src.robot.grasping.replay`)

The offline analysis tail. It reads what the pick path logged, a JSONL stream of
`GraspAttemptRecord`s, computes KPIs, runs the soak gates, scores latency, drift, out-of-distribution
detection and the failure taxonomy, and builds and diffs the baseline report.

Standard library and PyYAML only. No numpy, no torch, no web surface. Nothing in the pick path
imports this package.

## What this package guarantees

A verdict says what population it was computed over. `SoakSource` records whether a gate ran over a
real log, a simulation log, a generated stream or the canonical packs, because the same
`dict[str, bool]` comes back either way and a green gate over the wrong population is how a stack
gets credited with work it did not do.

A gate key that could not be judged is not a key that passed. `GateKeyStatus` has three states, and
`verdict.judged` says how many keys were actually decided, because a gate whose keys are all
not-applicable passes loudest.

KPI numbers come with the audit that says whether the records behind them carry what those KPIs
read. A rate computed over records missing their required telemetry fields is unfounded rather than
wrong, and `KpiRollup.sound` is the difference.

The operator console rolls up KPIs through this package's `compute_kpis`, so the console and the
command line can never disagree.

## The public surface

| Module | Role |
| --- | --- |
| `runs.py` | The three nouns a Python caller reaches for: `RecordLog`, `SoakGate` and `Baseline`, plus `SoakSource`, `GateKeyStatus`, `KpiRollup`, `SoakVerdict`. The CLI modes are shims over these |
| `kpi.py` | `compute_kpis` and `KpiSummary`: success, first-attempt, dead-loop, safety-rejection, dense-recovery and false-positive rates, plus the median cycle time. `UNMEASURABLE_KPIS` and `unmeasurable_kpis` name the rates a given record set cannot measure |
| `telemetry_catalog.py` | The per-outcome required-field catalog, the additive extra-field type contract, the presence and type audits, and the `rl_*` telemetry contract |
| `soak.py` | The deterministic seed-locked synthetic soak generator and the consolidated thirteen-key gate (`build_soak_report`, `SOAK_DEFAULT_ATTEMPTS = 2400`) |
| `baseline_report.py` | Per-pack KPI, SLO and telemetry blocks, the runtime-SLO aggregate, per-mode wall-time p95, the adaptation block, and `compare_kpi_deltas` |
| `canonical_datasets.py` | Deterministic on-disk JSONL fixtures and a SHA-256 manifest |
| `slo_eval.py` | Per-stage decision, ranking and fusion p50/p95/p99 against the locked 60 / 80 / 220 ms budgets |
| `watchdog_eval.py` | Drift and out-of-distribution precision and recall, gated at drift 0.90 / 0.85 and OOD 0.90 / 0.80 |
| `failure_taxonomy.py` | Root-cause classification: six causes plus `unclassified`, with a coverage report, plus `evaluate_labeled_pack`, the classifier graded against the committed labeled pack (agreement 1.0, coverage 0.968) |
| `adaptation.py`, `adaptation_io.py`, `adaptation_cli.py` | Guarded adaptation: plan, verify, apply as an overlay sidecar plus an audit JSONL, roll back |
| `presets.py` | The operator-safe mode-preset overlay loader and its schema re-validation |
| `soak_cli.py`, `__main__.py` | The argparse dispatcher: one mutually exclusive mode per run, meaningful exit codes |

The package `__init__` re-exports `RecordLog`, `SoakGate`, `Baseline`, `BaselineMeasurement`,
`SoakSource`, `SoakVerdict`, `GateKeyStatus`, `KpiRollup`, `KpiSummary`, `TelemetryOffender`,
`TelemetryVerdict`, `SoakScenarioSpec`, `TELEMETRY_CATALOG`, `compute_kpis`,
`generate_soak_records`, `audit_record`, `audit_records`, `apply_preset`, `list_presets` and
`load_preset`. The rest is imported by module path.

## Usage

```python
from src.robot.grasping.replay import RecordLog, SoakGate

# Roll up a log: the KPIs, and whether the records behind them carry what those KPIs read.
rollup = RecordLog.from_jsonl("logs/backend/run.jsonl").kpis()
print(rollup.render())
if not rollup.sound:
    ...   # the numbers are unfounded, not wrong, and that is a different thing

# Gate it. Four named doors, one per population, never a boolean.
verdict = SoakGate.over_records("logs/backend/run.jsonl").evaluate()
print(verdict.render())        # states what it is, and is not, evidence of
raise SystemExit(verdict.exit_code)
```

The four doors differ in what their verdict is evidence of, not in a parameter. `over_records` can
fail and is meant to. `over_sim_records` is real physics but not hardware representative.
`over_synthetic` invented its own grasps. `over_canonical_packs` passes by construction.

The free functions still work and are still exported:

```python
from src.robot.grasping.replay import compute_kpis
from src.robot.grasping.telemetry.outcome_logging import iter_jsonl

kpis = compute_kpis(iter_jsonl("logs/backend/run.jsonl"))
print(kpis.pick_success_rate, kpis.median_cycle_time_s)
```

### The command line

Modes are mutually exclusive and every one prints JSON to stdout.

| Flag | Purpose | Exit |
| --- | --- | --- |
| `--records <path.jsonl>` | Roll up KPIs and audit telemetry over a real log | 0, or 2 if there are offenders |
| `--records-gate <path.jsonl>` | Record-intrinsic soak thresholds over a real log. This one can fail | 0 or 1, or 2 if the log is unreadable |
| `--soak` | Cheap synthetic soak against `config/robot/kpi_thresholds.yaml` | 0 or 1 |
| `--soak-report` | The locked thirteen-key synthetic gate, written to `logs/u12/soak_report.json` | 0 or 1 |
| `--sim-soak-report <path.jsonl>` | Simulation-records quality gate plus a persistent report | 0 or 1, or 2 if the log is unreadable |
| `--baseline-report` | Build `docs/baselines/u_plus_baseline_v1.json` | 0, or 2 if there are offenders |
| `--regenerate-canonical` | Rewrite the canonical packs and manifest under `tests/data/replay/` | 0 |
| `--failure-taxonomy <pack...> --out <p>` | Classify failures and write a JSON report | 0, or 2 if a pack is missing |
| `--watchdog-eval` | Drift and OOD precision and recall gate | 0 or 2 |
| `--slo-gate` | Per-stage decision, ranking and fusion p95 gate | 0 or 2 |
| `--adaptation-plan` / `-verify` / `-apply` / `-rollback` | The guarded adaptation flow | 0 or 2 |

`--out` names the output file for every mode that writes one: `--failure-taxonomy`, where it is
required, plus `--baseline-report`, `--soak-report` and `--sim-soak-report`. `--baseline-out` is the
older spelling of the same thing, and passing both with different paths is an error rather than a
preference. `--baseline-report --out mine.json` used to write `docs/baselines/u_plus_baseline_v1.json`
and report success, so an operator who named their own file overwrote a committed one.

`--records` prints an `unmeasurable` block beside `kpi`, naming every rate these records cannot
measure and why. `false_positive_grasp_rate` is always there, because nothing on this stack writes the
field it divides by; `first_attempt_success_rate`, `dense_recovery_success_rate` and
`median_cycle_time_s` join it whenever the log carries no recovery action or no cycle time. A named
rate is withheld from `kpi` rather than printed as the `0.0` an empty denominator computes to.

```bash
# First run in a fresh clone: neither the packs nor the baseline are shipped.
python -m src.robot.grasping.replay --regenerate-canonical
python -m src.robot.grasping.replay --baseline-report

# The contract self-check.
python -m src.robot.grasping.replay --soak-report

# A real quality signal, which can actually fail.
python -m src.robot.grasping.replay --records-gate logs/backend/run.jsonl
```

## Traps

- **`--soak-report` is a synthetic self-check, not a hardware quality gate.** It composes three
  seed-locked scenarios and the outcome distribution in them is authored, so a pass proves that the
  telemetry, KPI, taxonomy, SLO and watchdog pipeline is internally consistent. It says nothing
  about grasp quality, and the report's own `provenance` block says so in its `does_not_measure`
  field. For a signal that can fail, use `--records-gate` or `--sim-soak-report` over a real log.
- **The gate's inputs are generated, not shipped.** The canonical packs live under
  `tests/data/replay/` and the baseline report under `docs/baselines/`; neither is in the
  repository, and the soak gate reads both. Regenerate them in the order above before the first run.
  `logs/` is not shipped either, so `logs/u12/soak_report.json` names where the report is written,
  not a file that is already there.
- **`--soak` and `--soak-report` are different gates.** They use different scenario compositions and
  different thresholds. `--soak-report` is the locked one.
- **The taxonomy section of the soak report and the taxonomy gate key are two different things.** The
  synthetic soak stream stamps no `extra.*_evidence` flag anywhere, so all 168 of its failures come
  back `unclassified` and `failure_taxonomy.coverage_fraction` reads 0.0. A classifier deleted down to
  `return UNCLASSIFIED` produces that block byte for byte, which is why the gate key
  `failure_taxonomy_classifier_pass` is graded against the committed labeled pack instead, whose rows
  carry the cause they were authored to have.
- **The auto-rollback guardrail needs a re-measurement to have anything to compare.**
  `--adaptation-apply` snapshots the baseline report before the apply and compares it with
  `compare_kpi_deltas`. Without `--post-apply-report PATH`, a baseline-report JSON re-measured under
  the applied overlay, the after equals the before by construction, the deltas are zero and the
  guardrail cannot fire. It logs a warning saying exactly that, so a clean apply without that flag
  is not a passed gate.
- **Wall-time p95 for `closed_loop` is `None`,** because the canonical packs hold no `closed_loop`
  records. The `easy` end-to-end wall-time budget is the only wall-time gate, and it is skipped
  rather than failed when its side is missing: insufficient evidence, not a regression.
- Everything here is batch over JSONL. There is no streaming or incremental KPI.

## See also

- [`grasping/`](../README.md) for the pick stack this package analyses
- [`telemetry/`](../telemetry/README.md) for the frozen record contract it reads
- [`rl/`](../rl/README.md) for the layer that consumes the `rl_*` telemetry fields hosted in
  `telemetry_catalog.py`
- [`api/`](../../../../api/README.md) for the operator console, which rolls up KPIs with this same
  `compute_kpis`
