# Grasp score calibration (`src/robot/grasping/calibration`)

Offline tools that fit, export and gate the two learned files the grasp scorer can load: the
uncertainty remap and the success-probability model. Nothing here runs during a pick, and it has nothing
to do with where a camera is: that is [`src/calibration/`](../../../calibration/README.md). The scorer
loads the files from `robot.grasping.uncertainty.calibration_artifact_path` and
`robot.grasping.success_model.artifact_dir`; call this package to build or check them.

```python
from src.robot.grasping.calibration import LabelPolicy, UncertaintyFit

report = UncertaintyFit.from_jsonl("tests/data/uncertainty_replay.jsonl").fit()
print(report.render())     # verdict no_labels, exit code 3: the fixture carries no label, and none is invented

report = UncertaintyFit.from_jsonl("tests/data/uncertainty_replay.jsonl",
                                   label_policy=LabelPolicy.FEASIBILITY_MARGIN_FIXTURE).fit()
print(report.render())     # fitted_from_invented_labels: samples, span and weight per channel
if report.artifact_available:
    report.write("artifact.json")
```

The fixture ships with the tests and has no labels, so the first fit refuses and the second invents them
and says so. From a shell:

```bash
python -m src.robot.grasping.calibration --replay <replay.jsonl> --out artifact.json [--calibration-id <id>]
python -m src.robot.grasping.calibration.success_model_calibration train [--records LOG.jsonl]
python -m src.robot.grasping.calibration.success_model_calibration verify
```

The uncertainty command exits 2 for a replay it cannot read and 0 otherwise, and it applies the invented
label rule; read its log line for the verdict. `promote` exits 1 on a failing verdict, and `verify` exits
2 on an unpromoted or tampered model.

## The nouns

| Noun | Built by | Verb | Returns |
|---|---|---|---|
| `UncertaintyFit` | `from_jsonl(path)`, `from_records(records)`; `label_policy=`, `strict=` | `fit()` | `UncertaintyFitReport`: verdict, per-channel fits, `exit_code`, `write(path)` |
| `PromotionThresholds` | `PromotionThresholds(brier_max=, log_loss_max=)`, in `model_promotion.py` | `verify_promotion(dir, thresholds=)` | `(ok, reasons)`; it never raises |

The success model's `train`, `eval`, `promote` and `verify` commands work on one model directory, which
holds `model.json`, `manifest.json` and `promotion.json`.

The package exports `UncertaintyFit`, `UncertaintyFitReport`, `ChannelFit`, `FitVerdict`, `LabelPolicy`,
`fit_uncertainty_calibration` and `usable_samples`. The promotion half is imported by module path,
because it pulls scikit-learn and importing the package must stay cheap.

## The uncertainty remap

The scorer fuses seven typed uncertainty channels into one score per grasp: `depth_confidence`,
`mask_confidence`, `occlusion_corridor_risk`, `feasibility_margin`, `verification_residual`,
`topology_risk` and `semantic_confidence`. This fits a monotone remap per channel from a labelled JSONL
replay and exports it as an `UncertaintyCalibration`. The fit is deterministic: no random numbers, a
stable sort, and identical bytes on every run.

Channel weights are not learned here. They stay at the `UncertaintyWeights()` defaults, which ship
`topology_risk` and `semantic_confidence` at `0.0`, so those two fits never reach the fused score;
`report.inert_channels` names them. Adding a channel is a change to the `UncertaintyChannel` enum in
[`uncertainty.py`](../uncertainty.py), the fusion weights and this file together.

Two rules the report enforces:

- **The command invents labels and the library does not.** With no `label` in the replay, the command
  derives one from `feasibility_margin > 0.5`, which fits that channel against itself. `UncertaintyFit`
  defaults to `LabelPolicy.REQUIRE` and refuses instead; `FEASIBILITY_MARGIN_FIXTURE` reproduces the
  command, and its verdict says the labels were invented.
- **An identity map can mean a perfect fit or an empty channel.** The report names `starved_channels`
  (no usable sample) beside `identity_channels`, and decides `nothing_learned` on starvation.

`nothing_learned` still writes a file and exits 0: a no-op the scorer loads without complaint. Read the
verdict, not the exit code. `strict=True` turns that verdict and the invented labels into exit code 3.

## The success-probability model

A calibrated probability that a grasp succeeds, over a locked vector of 23 features, exported as JSON.
The scorer predicts with NumPy alone; scikit-learn is imported only by the trainer, and the export checks
that the NumPy runtime reproduces scikit-learn.

| Family | Runtime path | For |
|---|---|---|
| `logistic_regression` | standardise, logistic, isotonic | the synthetic bootstrap |
| `gradient_boosted_trees` | a serialised tree ensemble, sigmoid, isotonic | a real, nonlinear grasp-outcome corpus |
| `mlp` | standardise, ReLU network, sigmoid, isotonic | the most expressive, if it wins the A/B gate |

A stronger challenger ships only by winning the A/B gate in [`model_families.py`](model_families.py) on
Brier score, ECE and AUROC; a tie, a regression, or an AUROC that cannot be graded keeps the incumbent.

**Training on your cell.** `train --records LOG.jsonl` trains on logged `GraspAttemptRecord`s. A record
carries the feature vector in `extra["success_model_features"]` when the shadow success predictor was
active during the pick, so collecting a corpus is a step at the cell before training is worth running.

**The promotion gate.** A model influences the ranking only in the `canary` and `active` lifecycle
phases, and only through a verified promotion against `PromotionThresholds` (`brier_max = 0.09`,
`log_loss_max = 0.31`) on a validation slice seeded with `PROMOTION_VALIDATION_SEED`. The default
`shadow` phase has no influence and loads ungated. Thresholds passed to `verify_promotion` can only
demand more: the stricter of the caller's and the locked ones applies, and so does the locked
validation slice. The trust root is a SHA-256 chain over the files. Signing is an optional layer above it:
[`signing.py`](signing.py) holds `Ed25519Signer` and `AwsKmsSigner`, and with no signer the signature
stays `"none"` and the verdict never depends on it.

## Status

The legend is the root README's [Status and honest scope](../../../../README.md#status-and-honest-scope).

| Capability | Evidence |
|---|---|
| The shipped model, `assets/models/success_probability/v1` | never touched hardware; a `logistic_regression` bootstrap on synthetic data that passes `verify` |
| The gradient-boosted and MLP families | never touched hardware; built and validated, and no better on the bootstrap, which is why it stays logistic |
| The uncertainty remap | never touched hardware; the shipped replay has no labels |

The shipped model's manifest says `"kind": "synthetic_bootstrap"`: it exercises the whole train, gate
and export chain, and it is not evidence about a physical cell.

## Files

| File | Holds |
|---|---|
| [`fits.py`](fits.py) | `UncertaintyFit`, `UncertaintyFitReport`, `ChannelFit`, `FitVerdict`, `LabelPolicy` |
| [`uncertainty_calibration.py`](uncertainty_calibration.py) | the monotone remap fitter and its command |
| [`success_model_calibration.py`](success_model_calibration.py) | the trainer and exporter, the datasets, the metrics and the four commands |
| [`model_families.py`](model_families.py) | the MLP family and the A/B family gate |
| [`model_promotion.py`](model_promotion.py) | `PromotionThresholds`, the gate, the `promotion.json` writer and loader, the runtime verifier |
| [`signing.py`](signing.py) | the optional `Signer` and `Verifier` seam |
| [`__main__.py`](__main__.py) | runs the uncertainty command |

## Details

- The scorer that loads these files: [`grasping/scoring/`](../scoring/README.md).
- Where trainable records come from: [`datagen/`](../../../../datagen/README.md), whose `rl/collect.py`
  executes top candidates in physics and writes them as `GraspAttemptRecord`s.
- The offline layer that shares this promotion discipline: [`rl/`](../rl/README.md).
- Tests: [`test_calibration_fits.py`](../../../../tests/test_calibration_fits.py),
  [`test_promotion_gate_cannot_be_talked_down.py`](../../../../tests/test_promotion_gate_cannot_be_talked_down.py),
  [`test_success_probability_model.py`](../../../../tests/test_success_probability_model.py).
