# Offline reinforcement learning (`src/robot/grasping/rl`)

Policies trained offline on your record logs, which re-rank and prune grasp candidates, sequence
recovery and budget perception views. At pick time they run in shadow only: they log what they would
have done and change nothing. Safety, geometry and every deterministic decision outrank them, and the
package imports no driver, no safety module and no vendor SDK.

Start with the record log a cell writes when `robot.grasping.record_log_path` is set, and ask whether
it can train anything:

```python
from src.robot.grasping.rl.runs import RecordLogCheck

report = RecordLogCheck.from_records_log("logs/<your-cell>/grasp_records.jsonl").assess()
print(report.render())                  # occupancy, variance, success and fail pairs
raise SystemExit(report.exit_code)      # 0 trainable, 1 not trainable, 2 unreadable
```

The same check from a shell. It trains nothing and opens no cell:

```bash
python -m src.robot.grasping.rl check-dataset --records logs/<your-cell>/grasp_records.jsonl
```

To shadow a policy at pick time, name it in the cell's profile. The shipped `rl_datagen` profile sets
exactly this, with the baseline artifacts the repository carries:

```yaml
robot:
  rl:
    mode: rl_shadow
    policy_id: v2_candidate_baseline_v1
    artifact_path: "${WILLY_PROJECT_ROOT}/docs/baselines/rl_policies/v2_candidate_baseline_v1.json"
    ranking_artifact_path: "${WILLY_PROJECT_ROOT}/docs/baselines/rl_policies/v3_ranking_baseline_v1.json"
```

`${WILLY_PROJECT_ROOT}` is the repository; a relative path in the tree would be read against the
config folder ([`src/config/paths.py`](../../../config/paths.py)).

## The modes

| `robot.rl.mode` | What runs |
| --- | --- |
| `geometry_only`, `hybrid_ml` (the default) | no RL object is built, and the pick is the deterministic stack's |
| `rl_shadow` | the log-only `ShadowRouter`, whose telemetry lands on the pick report |
| `rl_active`, `rl_experimental` | refused at boot: no shipped producer |

The authority order is safety, then hardware and runtime constraints, then the deterministic geometry,
then deterministic recovery, and only then ML and RL. Inside the router the action mask filters first,
over `MASK_CHANNELS` in precedence order `(safety, feasibility, corridor, uncertainty, drift_ood,
degraded_mode)`. The mask only reflects what the deterministic stack decided, and the router re-checks
that no policy re-admitted a masked candidate. Every router method is exception-bounded, so a broken
policy degrades to a typed fallback rather than failing a pick.

| `policy_kind` | Module | Model |
| --- | --- | --- |
| `candidate_selection` | `candidate_policy.py` | logistic; the one policy the router requires |
| `ranking` | `ranking_policy.py` | pairwise logistic |
| `sequencing` | `sequencing_policy.py` | lookup table plus an anti-loop gate |
| `perception_budget` | `perception_budget_policy.py` | LinUCB contextual bandit |
| `recovery` | `recovery_policy.py` | LinUCB contextual bandit plus an anti-loop gate |

## What `check-dataset` looks for

Three failures make a log worthless for training. All three are silent, and all three look like a
healthy log until a training run converges over nothing.

| Failure | What it is | Where to fix it |
| --- | --- | --- |
| no success and fail pairs | pairs form within a group keyed on `scene_family_id`, then `scene_id`, then episode, session or bin | your logging schema |
| dead features | a feature that is never populated reads `0.0` and trains to exactly zero | your feature producers |
| no per-candidate rows | `extra.rl_candidate_features` is written only while the ranking shadow runs (`rl_shadow`) | the YAML above |

A pairwise ranker differences features within a group, so a feature constant across a group (scene,
pick or object level) contributes nothing however well it is populated; the check reports occupancy at
both levels. A cell with `record_log_path` set and no `rl_shadow` warns once at boot that it will log
no per-candidate features. It warns rather than refuses, because the same log feeds the KPIs and every
post-mortem.

## Training offline

The commands exit 0 on a pass, 2 on bad arguments or a missing manifest and 3 on a leakage failure or
malformed input; the two drills exit 1 on a failure. `--canonical-source` puts your log in place of the
bundled canonical packs, and `--dataset-origin` stamps where the records came from. A command given no
`--replay-pack` reads the canonical packs, and a trainer given no `--output` writes over the shipped
baseline it is named after, so name both.

```bash
LOG=logs/<your-cell>/grasp_records.jsonl
python -m src.robot.grasping.rl build-dataset --dataset-id my_cell_v1 --canonical-source $LOG --dataset-origin real_hardware
python -m src.robot.grasping.rl audit-leakage --manifest docs/baselines/rl_datasets/my_cell_v1.json
python -m src.robot.grasping.rl train-candidate-policy --dataset-id my_cell_v1 --output candidate.json
python -m src.robot.grasping.rl train-ranking-policy --dataset-id my_cell_v1 --output ranking.json
python -m src.robot.grasping.rl train-recovery-policy --replay-pack $LOG --output recovery.json
python -m src.robot.grasping.rl ope --replay-pack $LOG --output ope.json
python -m src.robot.grasping.rl promote-policy --policy-family v6_recovery --policy-artifact recovery.json \
    --replay-pack $LOG --output promotion.json
python -m src.robot.grasping.rl rollback-drill --policy-artifact recovery.json --promotion-report promotion.json
```

Run them from the repository root. `train-sequencing-policy`, `train-perception-budget-policy`,
`replay-env-check` and `paired-soak` complete the set, and `--help` on each lists its flags.
`promote-policy` takes `v4_sequencing`, `v5_perception_budget` or `v6_recovery`, and the two drills
need a promotion report whose verdict is `pass`. A candidate policy you train keeps the name
`v2_candidate_baseline_v1`, so a shadow profile keeps that `policy_id` and names your file in
`artifact_path`.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `RLModeNotImplementedError` at boot | `rl_active` or `rl_experimental` | use `rl_shadow` |
| `ConfigError` at load | `rl_shadow` without `policy_id` and `artifact_path`, or `online_update_enabled` without an RL mode | set both fields, or drop the flag |
| no shadow router, and a warning | `artifact_path` names no file, or the artifact is not the `policy_id` declared | point at a trained artifact with that id |
| the promotion gate refuses | the artifact carries an `online_state` stamp from online updates | promote a frozen artifact |
| the promotion gate abstains | too few weighted records, too small an effective sample, or one action only | collect more varied data |

An abstention is the fail-closed answer: a policy that always proposes what the logged behaviour did
collapses the importance weights, and a do-nothing policy must not read as a pass. The gate gates live
control only, so an abstention does not block a shadow run.

## Status

| Capability | Evidence |
| --- | --- |
| Dataset, training, off-policy evaluation and promotion | measured in simulation: the chain closes on measured physics outcomes |
| A policy on a physical cell | never touched hardware |

The baseline artifacts under `docs/baselines/rl_policies/` are trained on the synthetic canonical
bootstrap data. Their honesty stamps say so; the perception-budget and recovery baselines collapsed to a
single action, and every promotion report shipped beside them abstains. Train on your own logs before
you read anything into a shadow annotation. Off-policy estimates are estimates, not real lift. Live
control of the perception view budget waits for real-hardware data and a passing gate. The canary,
specialist and drill routers run from the command line only. The `torch` runtime tier is declared and
not built; `robot.rl.runtime_tier` defaults to `stdlib`, and `numpy` is opt-in.

## Traps

- `LinUCBRecoveryPolicy` is frozen, so `propose_recovery` cannot be monkey-patched. A drill wraps the
  policy instead, exposing `policy_id`, `version` and `propose_recovery`.
- The specialist router's blocked actions outrank the policy: `perturb_and_retry` is blocked on a
  deformable, and the applied action falls back to the deterministic baseline. If a policy seems to have
  no effect, check this first, and drill overrides with an action the mask does not block.
- A canary override-rate gate trips on the first overriding call when `warmup_n` is 0 and
  `max_override_rate` is below 1.0. The shipped defaults are 20 and 0.95.

## Files

| File | Holds |
| --- | --- |
| `__init__.py` | the mode constants, `assert_rl_mode_supported`, `RLModeNotImplementedError` |
| `action_mask.py`, `router.py` | the mask; `ShadowRouter`, the runtime seam, and its telemetry |
| `runs.py`, `readiness.py` | `RecordLogCheck`, the library face of `check-dataset`, and the assessment behind it |
| `dataset.py`, `datasets.py`, `leakage.py` | `build_dataset` with leakage-safe splits, `DatasetManifest`, the leakage audit |
| `sar.py`, `replay_env.py` | the state, action and reward extractor; the replay-environment fingerprints |
| `train_*.py`, `policy_training.py` | the per-policy trainers |
| `ope.py`, `evaluation.py`, `promotion.py` | off-policy evaluation; the promotion gate and its SHA-256 attested `promotion.json` |
| `honesty.py` | the honesty and provenance stamps every artifact carries |
| `online_state.py`, `online_router.py`, `online_repromotion.py` | online updates kept outside the frozen policy, and their freshness |
| `recovery_dispatch.py` | the live recovery loop as an adapter, off by default |
| `canary_router.py`, `specialist_router.py`, `paired_soak.py`, `rollback_drill.py` | the canary and specialist routers and their drills, offline only |

## Details

- [execution/autonomous_grasp/](../../execution/autonomous_grasp/README.md) builds the `ShadowRouter`
  and folds its telemetry onto the report
- [replay/](../replay/README.md) holds the telemetry catalog and the KPIs this layer reads offline
- [datagen/](../../../../datagen/README.md) collects records with physics rewards to train on
- [safety/](../../safety/README.md) is the fail-closed layer that outranks RL
