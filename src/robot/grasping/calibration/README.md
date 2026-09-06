# Grasping calibration (`src.robot.grasping.calibration`)

Offline, deterministic tooling that produces and gate-keeps the two learned artifacts the grasping
runtime consumes: the uncertainty-fusion remap and the success-probability model. The runtime only
ever loads what this package emits; nothing here runs on a pick path.

This is not the metric hand-eye calibration. That lives in `src.robot.execution.calibration` and
[`src/calibration/`](../../../calibration/README.md). This one fits behavioural mappings: channel to
label, and features to success.

## What this package guarantees

Determinism. No PRNG in the uncertainty fit, a stable sort, and byte-stable output across runs.

A trained model gains behavioural influence only through a verified promotion. The trust root is a
SHA-256 attestation chain over the artifact bytes; signing is an optional layer above it. The
runtime verifier checks that the verdict is `pass`, the bytes match the chain, the thresholds were
not weakened, the validation slice is the locked one, and the recorded metrics agree with the
recorded bounds.

scikit-learn is imported here, in the trainer, and nowhere on the hot path. The exported artifact is
pure JSON that the runtime predicts with numpy alone, and the export step validates that the numpy
runtime reproduces scikit-learn end to end.

## The public surface

| Module | Role |
| --- | --- |
| `fits.py` | `UncertaintyFit`, `UncertaintyFitReport`, `ChannelFit`, `FitVerdict`, `LabelPolicy`. The library twin of the uncertainty CLI, and the only place that says whether a fit learned anything |
| `uncertainty_calibration.py` | The monotone-remap fitter and its CLI: `fit_uncertainty_calibration`, `usable_samples` |
| `success_model_calibration.py` | Trainer and exporter, the synthetic and record-backed datasets, numpy metrics, and the `train` / `eval` / `promote` / `verify` CLI |
| `model_families.py` | The MLP family and the fail-closed A/B family-selection gate |
| `model_promotion.py` | `PromotionThresholds`, `PROMOTION_VALIDATION_SEED`, the promotion-gate evaluator, the `promotion.json` writer and loader, and the runtime tamper and threshold verifier |
| `signing.py` | The optional signing seam: `Signer` and `Verifier` Protocols, `Ed25519Signer` and `AwsKmsSigner` |
| `__main__.py` | `python -m src.robot.grasping.calibration` runs the uncertainty CLI |

The package `__init__` re-exports `ChannelFit`, `FitVerdict`, `LabelPolicy`, `UncertaintyFit`,
`UncertaintyFitReport`, `fit_uncertainty_calibration` and `usable_samples`. Everything else is
imported by module path.

## The uncertainty-fusion remap

The runtime fuses seven typed uncertainty signals into one per-grasp score. This tool fits a
deterministic, monotone per-channel remap from a labelled JSONL replay and exports it as a JSON
`UncertaintyCalibration`.

| Channel | What it measures |
| --- | --- |
| `depth_confidence` | trust in the stereo or RGB-D depth at the grasp |
| `mask_confidence` | how clean the object's segmentation mask is |
| `occlusion_corridor_risk` | risk that the straight-line approach corridor is occluded or blocked |
| `feasibility_margin` | how far the grasp sits from the IK and reachability boundary |
| `verification_residual` | disagreement left over after the post-grasp verification cross-check |
| `topology_risk` | risk from the object's local shape: thin, deformable, ambiguous |
| `semantic_confidence` | confidence in the class the detector assigned |

Channel weights are not learned here. They stay at the `UncertaintyWeights()` defaults, which ship
`topology_risk` and `semantic_confidence` at `0.0`, so those two fits are multiplied by zero and do
not reach the fused score. `report.inert_channels` names them and `render()` marks them. The set is
fixed at seven: adding an eighth is a coordinated change across the `UncertaintyChannel` enum in
[`grasping/uncertainty.py`](../uncertainty.py), the fusion weights and this artifact.

```python
from src.robot.grasping.calibration import LabelPolicy, UncertaintyFit

report = UncertaintyFit.from_jsonl("replay.jsonl", calibration_id="replay-v1").fit()
print(report.render())          # verdict, per-channel samples, span and weight, exit code
if report.artifact_available:
    report.write("artifact.json")
```

`fit_uncertainty_calibration(records)` is still the fit itself and is still exported. What it cannot
tell you is what came out, and that is what the report is for.

### Two rules the report exists to enforce

The CLI invents labels and the library does not. When a replay carries no `label`, the CLI derives
one from `feasibility_margin > 0.5`, and the two entry points therefore produce different artifacts
from the same file. That rule is also circular, because `feasibility_margin` is one of the seven
calibrated channels and is fitted against labels derived from itself; a missing margin reads as a
failure rather than as missing data. `UncertaintyFit` defaults to `LabelPolicy.REQUIRE` and refuses
instead. Pass `label_policy=LabelPolicy.FEASIBILITY_MARGIN_FIXTURE` to reproduce the CLI exactly, and
read the `FITTED_FROM_INVENTED_LABELS` verdict as what it says.

An identity map alone cannot tell a perfect fit from an empty channel. A channel that separates the
labels perfectly fits to `breakpoints=(0, 1), values=(0, 1)`, which is the identity map. The report
therefore carries `starved_channels`, meaning no usable sample, beside `identity_channels`, and the
`NOTHING_LEARNED` verdict is decided on starvation. Deciding it on identity would file the best
possible fit as the worst.

## The success-probability model

A calibrated `P(grasp succeeds | features)` over a locked 23-feature vector, exported as JSON plus a
manifest.

| Family | Runtime path | Best for |
| --- | --- | --- |
| `logistic_regression` | standardise, logistic, isotonic | a logistic-structured bootstrap |
| `gradient_boosted_trees` | serialized tree ensemble, sigmoid, isotonic | a real, nonlinear grasp-outcome corpus |
| `mlp` | standardise, ReLU MLP, sigmoid, isotonic | the most expressive family, if it wins the A/B gate |

A stronger challenger never ships just for being stronger. `model_families.py` runs a fail-closed
A/B gate on Brier score, ECE and AUROC with two admission paths, a calibration path requiring a
Brier gain with no AUROC regression, and a ranking path requiring an AUROC gain with no meaningful
Brier or ECE regression. A tie, a regression, and an AUROC that cannot be graded because a class is
missing from the eval split all keep the incumbent.

**Training data.** By default the trainer uses a deterministic synthetic bootstrap that exercises
the pipeline. To train on real grasp outcomes, point it at logged `GraspAttemptRecord`s: the runtime
stamps the executed grasp's feature vector into `record.extra["success_model_features"]`, and
`build_dataset_from_records` turns `(features, final_outcome)` into `(X, y)`. That feature logging
runs when the shadow success predictor is active, so collecting the corpus is an on-box step before
`train --records` is worth running.

### The promotion gate

Train, evaluate on a frozen validation slice seeded with `PROMOTION_VALIDATION_SEED`, write
`promotion.json` with metrics, thresholds and the attestation chain, then verify at load time.

A success model influences behaviour only in the `canary` and `active` lifecycle phases, and only
through a verified promotion against the locked `PromotionThresholds`, `brier_max = 0.09` and
`log_loss_max = 0.31`. The default `shadow` phase loads ungated because it has no influence.

`verify_promotion` takes a `thresholds` parameter, and it means "additionally demand", never "demand
less": the effective bar is the stricter of the caller's and the plan's, and the validation slice is
locked at verification rather than taken from the caller. Without that, a caller could pass its own
looser thresholds or a different seed and talk the gate out of its own exam.

**Signing is opt-in.** The SHA-256 chain is the trust root and needs no crypto. Above it, a
deployment with a key-management story injects a `Signer` and `Verifier`: `Ed25519Signer` with a
local PEM key, or `AwsKmsSigner` driving real KMS Sign and Verify through a lazily imported `boto3`.
GCP KMS and Azure Key Vault are the same Protocol with their own SDK. With no signer injected the
signature stays `"none"` and nothing is faked; the runtime verdict never depends on it.

## Usage

```bash
# Uncertainty monotone-remap fit
python -m src.robot.grasping.calibration \
    --replay <replay.jsonl> --out artifact.json [--calibration-id <id>]

# Success-probability model
python -m src.robot.grasping.calibration.success_model_calibration train \
    [--artifact-dir DIR] [--records LOG.jsonl] \
    [--model-family logistic_regression|gradient_boosted_trees]
python -m src.robot.grasping.calibration.success_model_calibration eval    [--artifact-dir DIR]
python -m src.robot.grasping.calibration.success_model_calibration promote [--artifact-dir DIR]
python -m src.robot.grasping.calibration.success_model_calibration verify  [--artifact-dir DIR]
```

Exit codes. The uncertainty CLI returns 2 for a replay it cannot read and 0 otherwise;
`UncertaintyFitReport.exit_code` additionally returns 3 for a refusal or a strict rejection, which
the CLI does not trigger because it does not pass `strict=True`. `promote` returns 1 on a failing
verdict and `verify` returns 2 on an unpromoted or tampered artifact.

## Traps

- The shipped success model under `assets/models/success_probability/v1` is a
  `logistic_regression` bootstrap trained on synthetic data. Its manifest says
  `"kind": "synthetic_bootstrap"` in as many words. It exercises the full train, gate and export
  pipeline and it passes the gate; it is not evidence about a physical cell.
- The gradient-boosted and MLP families are built and validated and do not help on the bootstrap: a
  logistic-structured synthetic dataset is exactly the case where the incumbent wins, which is why
  the shipped artifact stays logistic. They are for a real corpus.
- The uncertainty CLI writes an artifact for `NOTHING_LEARNED` too, and exits 0. That artifact is a
  seven-channel no-op the runtime will load and apply without complaint, so read the verdict rather
  than the exit code.
- No replay fixture ships in this repository. The uncertainty CLI needs a JSONL replay you supply.

## See also

- [`grasping/scoring/`](../scoring/README.md) for the scorer that consumes these artifacts
- [`grasping/uncertainty.py`](../uncertainty.py) for the fused-uncertainty types the remap targets
- [`datagen/rl/collect.py`](../../../../datagen/README.md) for where trainable records come from:
  top-k candidates per scene, executed in physics, written out as `GraspAttemptRecord`s
- [`rl/`](../rl/README.md) for the offline layer that shares this promotion and honesty discipline
