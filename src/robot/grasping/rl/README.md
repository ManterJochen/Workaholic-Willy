# RL optimisation layer (`src.robot.grasping.rl`)

An optional, offline-trained reinforcement-learning layer on top of the deterministic grasping
stack. It can re-rank and prune candidates, blend rankings, sequence recovery, budget perception
views and pick recovery actions, but only ever in shadow, meaning log-only, and it never overrides
safety, geometry or any deterministic decision.

Pure standard library on the runtime path. No numpy and no torch is imported for the shipping tier.

## What this package guarantees

**A cell that does not name an RL mode constructs no RL object.** `robot.rl.mode` defaults to
`hybrid_ml`, and `geometry_only` and `hybrid_ml` both return before anything here is built, so the
pick report is byte-identical to the deterministic stack's.

**Authority order, locked in `__init__.py`:** safety, then hardware and runtime constraints, then
the deterministic geometry pipeline, then deterministic recovery, and only then ML and RL. This
package imports no `drivers/`, no `safety/` and no vendor SDK, so it has no way to reach past the
layers above it.

**A mode with no shipped producer is rejected loudly rather than degrading to a silent no-op.**
`assert_rl_mode_supported` is the single typed gate and `maybe_build_shadow_router` calls it at boot.
`rl_active` and `rl_experimental` are schema-valid and raise `RLModeNotImplementedError`.

**No online weight updates on a policy that influences anything.** Online deltas accumulate outside
the frozen policy and stamp the artifact root with an `online_state` marker that the promotion gate
refuses, so activation requires a fresh promotion of a frozen artifact.

## The runtime seam, which is the only wired surface

```
candidates -> action_mask -> the wired policy slots -> ShadowRouter -> ShadowRouterTelemetry
```

`MASK_CHANNELS` is `(safety, feasibility, corridor, uncertainty, drift_ood, degraded_mode)`, in that
precedence order. `apply_action_mask` returns, per candidate, the ordered tuple of channels that
masked it, and the router re-checks on return that no policy re-admitted a masked id. The mask
reflects only what the deterministic stack already decided; it never invents a reason. Every router
method is exception-bounded, so a buggy policy degrades to a typed fallback rather than crashing a
pick.

| Module | Role |
| --- | --- |
| `__init__.py` | `assert_rl_mode_supported`, `RLModeNotImplementedError`, `RL_SUPPORTED_MODES` and the mode constants |
| `action_mask.py` | The inviolable first filter: `MASK_CHANNELS`, `ActionMaskContext`, `apply_action_mask`, `mask_summary` |
| `router.py` | `ShadowRouter`, the runtime inference seam, and `ShadowRouterTelemetry` |

### The five policies

Each ships a Protocol, a frozen deterministic implementation, request and selection records, and a
loader that also hashes the artifact.

| `policy_kind` | Module | Model |
| --- | --- | --- |
| `candidate_selection` | `candidate_policy.py` | logistic |
| `ranking` | `ranking_policy.py` | pairwise logistic |
| `sequencing` | `sequencing_policy.py` | lookup table plus an anti-loop gate |
| `perception_budget` | `perception_budget_policy.py` | LinUCB contextual bandit |
| `recovery` | `recovery_policy.py` | LinUCB contextual bandit plus an anti-loop gate |

Only `candidate_policy` is required on the router. The other four are typed slots defaulting to
`None`, which is the silent shadow-off path.

## Before you train on a log: `check-dataset`

Three failure modes make a corpus worthless. All three are silent, all three are cheap to detect up
front, and all three look identical to a healthy log until a training run reports that it converged
over nothing.

```bash
# Start here with a record log, your own cell's for instance.
# Trains nothing, opens no cell: occupancy, variance, and success/fail pair counts.
# Exit 0 trainable, 1 not trainable, 2 unreadable.
python -m src.robot.grasping.rl check-dataset --records logs/<your-cell>/grasp_records.jsonl
```

| Failure | What it is | Where it sends you |
| --- | --- | --- |
| No success/fail pairs | The pairwise ranker forms pairs within a group, keyed on `scene_family_id`, then `scene_id`, then `episode_id` or `session_id` or `bin_id`, and falling back to the attempt-id prefix. A log whose scene identity sits under a different name puts every record in its own group and yields zero pairs. | your logging schema |
| Dead features | `_project_feature` maps a missing key to `0.0` silently, so a feature that is never populated trains to exactly zero and reports as converged. | your feature producers |
| No per-candidate rows | `extra.rl_candidate_features` exists only when the ranking shadow ran, which needs `robot.rl.mode: rl_shadow` with `policy_id` and `artifact_path`. Without it a cell logs records that look perfectly healthy: fine for KPIs and a pointwise model, useless to a pairwise ranker. | one line of YAML |

A pairwise ranker differences features within a group, so a feature that is identical for every
member of a group contributes exactly zero however well populated it is. Scene-level, pick-level and
object-level quantities are all in that class, and no occupancy dashboard shows the difference.
`check-dataset` therefore reports occupancy at both levels and judges them differently.

The cell also tells you at boot. When `grasping.record_log_path` is set and `robot.rl.mode` is not
`rl_shadow`, the service warns once that this cell will log no per-candidate features. Deliberately a
warning and not a refusal: record logging also feeds the KPI roll-up, the failure taxonomy and every
post-mortem a bring-up depends on. The warning exists because the alternative is collecting for
months and finding out at the end.

## Usage

The runtime path is config-driven, not constructed by hand.
`execution/autonomous_grasp/shadow.py` builds a `ShadowRouter` only when
`robot.rl.mode == "rl_shadow"`; past the admission gate, any missing field, missing artifact or load
error returns no shadow rather than aborting startup. The typed gate is the one public call:

```python
from src.robot.grasping.rl import assert_rl_mode_supported

assert_rl_mode_supported("rl_shadow")   # admitted
assert_rl_mode_supported("rl_active")   # raises RLModeNotImplementedError
```

Everything else is the offline command line, which never runs on a hot path. Exit codes: 0 pass,
2 bad arguments or a missing manifest, 3 a leakage failure or malformed input.

```bash
# Dataset: build the splits and the leakage audit, and print the readiness summary
python -m src.robot.grasping.rl build-dataset     --dataset-id v1_bootstrap
python -m src.robot.grasping.rl audit-leakage     --manifest docs/baselines/rl_datasets/v1_bootstrap.json
python -m src.robot.grasping.rl replay-env-check  --manifest docs/baselines/rl_datasets/v1_bootstrap.json

# Train. Each writes a JSON artifact under docs/baselines/rl_policies/.
python -m src.robot.grasping.rl train-candidate-policy         --dataset-id v1_bootstrap
python -m src.robot.grasping.rl train-ranking-policy           --dataset-id v1_bootstrap
python -m src.robot.grasping.rl train-sequencing-policy        --dataset-id v1_bootstrap
python -m src.robot.grasping.rl train-perception-budget-policy --replay-pack <pack.jsonl>
python -m src.robot.grasping.rl train-recovery-policy          --replay-pack <pack.jsonl>

# Off-policy evaluation and the promotion gate, both report-only
python -m src.robot.grasping.rl ope
python -m src.robot.grasping.rl promote-policy --policy-family recovery \
    --policy-artifact <artifact.json>

# Hardening harnesses, non-zero exit on a parity or drill failure
python -m src.robot.grasping.rl paired-soak    ...
python -m src.robot.grasping.rl rollback-drill ...
```

### Offline tooling, online updates, canary harnesses

| Module | Role |
| --- | --- |
| `dataset.py`, `datasets.py` | `build_dataset` with leakage-safe splits, `DatasetManifest`, and the `LeakageAudit` noun |
| `readiness.py` | `assess_records` and `format_readiness`, the `check-dataset` engine |
| `leakage.py` | The embedded leakage audit |
| `sar.py`, `replay_env.py` | The state-action-reward extractor and the replay-env fingerprints |
| `train_*.py`, `policy_training.py` | The per-policy trainers, each writing a JSON artifact |
| `ope.py`, `evaluation.py` | Off-policy evaluation: weighted importance sampling, fitted Q evaluation, direct method |
| `promotion.py` | The offline promotion gate, writing a `promotion.json` with a SHA-256 attestation |
| `honesty.py` | The single source of truth for the artifact honesty and provenance stamps |
| `online_state.py`, `online_router.py`, `online_repromotion.py` | The mutable diagonal-LinUCB accumulator, the `OnlineUpdateCoordinator` that folds `(state, action, reward)` and stamps the marker the promotion gate refuses, and the freshness cadence that flags a stale `pass` |
| `recovery_dispatch.py` | The live recovery loop as a default-off adapter, applying the action through an injected executor |
| `canary_router.py`, `specialist_router.py` | `ActiveCanaryRouter` with a windowed regret and override auto-fallback plus a kill switch, gated on a passing promotion verdict, and `DeformableSpecialistRouter` |
| `paired_soak.py`, `rollback_drill.py` | The RL-on against RL-off paired comparator, and the kill-switch and auto-fallback drills |

The canary, specialist and hardening pieces are offline only: a command line and nothing else. They
have no runtime call site.

## Honest status

- **Five locked modes; the runtime admits three.** `geometry_only` and `hybrid_ml`, the default,
  build no RL object. `rl_shadow` builds the log-only router. `rl_active` and `rl_experimental` are
  rejected.
- **Shadow is the only runtime-wired surface,** log-only, with zero influence on the executed grasp.
  The candidate, ranking and sequencing sub-blocks are spliced into `report.telemetry`; the
  perception-budget and recovery sub-blocks are carrier-only and stay out of `record.extra`, so the
  replay JSONL and the telemetry catalog stay frozen.
- **No trained policy artifact ships in this repository.** The loaders and `shadow.py` default to
  paths under `docs/baselines/rl_policies/`, and that directory is where the trainers write. Until
  you train one, `rl_shadow` logs a warning that the artifact does not exist and the cell runs with
  no shadow router. Nothing is shadow-routed and nothing is annotated.
- **Every artifact carries an honesty and provenance stamp at its JSON root,** so a synthetic,
  off-policy or degenerate artifact can never pass for a hardware-validated one:
  `reward_model` and `reward_interpretation`, `dataset_provenance` with `fit_for` and `not_fit_for`,
  the meaning of the gate verdict, and a `degeneracy_note` emitted when a policy collapsed to a
  single action. Off-policy estimates are off-policy, not real lift, and the soak and rollback
  harnesses characterise mechanics only.
- **The promotion gate abstains rather than passing when it cannot judge.** Too few weighted
  records, too small an effective sample size, and a degenerate target policy proposing fewer than
  two distinct actions all abstain. A degenerate policy almost always proposes the action the
  behaviour policy took, so the importance weights collapse and the lift is arithmetically near
  zero: a do-nothing policy must not read as a pass. That is the correct fail-closed outcome, and it
  does not block a shadow ship, because the gate gates live control only.
- **Live control of the perception view budget must not ship without real-hardware data and a
  passing, not abstaining, gate.** In shadow the budget is driven only by the deterministic
  `ScoringViewpointPlanner`, and the perception shadow runs after the pick has committed.
- **The `torch` runtime tier is a declared not-built-yet seam.** `robot.rl.runtime_tier` defaults to
  `stdlib`; `numpy` and `torch` are opt-in and lazily imported, so the default path stays free of
  both.

## Traps

- **`LinUCBRecoveryPolicy` is a frozen dataclass, so `propose_recovery` cannot be monkey-patched.**
  A drill that tries gets an attribute error that reads like a typo. The way through is a duck-typed
  wrapper mimicking `policy_id`, `version` and `propose_recovery` closely enough for router
  validation to accept it.
- **The specialist's blocked-action mask outranks the policy, by design.**
  `DEFAULT_SPECIALIST_BLOCKED_ACTIONS` forbids `perturb_and_retry` on a deformable, because
  perturbing a deformable is a damage risk. When a policy's preferred action for a cell is the one
  the mask blocks, the applied action silently becomes the deterministic baseline. If a policy
  appears to have no effect, check this first.
- **An override drill has to use an action the mask does not block.** Otherwise the mask filters the
  override back to the baseline, the override never registers, and the drill reports that nothing
  happened rather than that it was blocked.
- **A canary override-rate gate can trip on the first overriding call** when `warmup_n` is 0 and
  `max_override_rate` is below 1.0, because there is no window to average over yet. The shipped
  defaults are `warmup_n = 20` and `max_override_rate = 0.95`.
- **`rl_shadow` is a schema-active mode**, so it requires `policy_id` and `artifact_path` to be set;
  the config refuses to load otherwise.

## See also

- [`execution/autonomous_grasp/`](../../execution/autonomous_grasp/README.md) for the code that
  builds the `ShadowRouter` and folds its telemetry onto the report
- [`grasping/replay/`](../replay/README.md) for the KPI and telemetry-catalog contract this layer
  consumes offline
- [`datagen/`](../../../../datagen/README.md) for the physics-reward collection path that produces
  trainable records
- [`safety/`](../../safety/README.md) for the fail-closed layer that outranks RL by construction
