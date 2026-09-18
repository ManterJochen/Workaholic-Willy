# The learned success predictor (`src/robot/grasping/scoring/success_probability`)

Estimates the probability that a scored grasp succeeds, from a locked vector of 23 features, and holds
the two rerankers that may reorder candidates with it. Numpy only at run time, with no scikit-learn and
no joblib. A cell reaches it through `robot.grasping.success_model`, which ships `enabled: false`; call
it directly to score candidates offline.

```python
from src.robot.grasping.scoring.success_probability import (
    extract_features, load_success_probability_model, predict_proba,
)

model = load_success_probability_model("assets/models/success_probability/v1")
features = extract_features(breakdown, mode="dense_clutter")   # breakdown: a GraspScoreBreakdown
print(predict_proba(model, features[None, :]))                 # a probability in [0, 1]
```

`breakdown` comes from `rank_grasp_poses` ([scoring/](../README.md)). The model under
`assets/models/success_probability/v1` is a synthetic bootstrap: it exercises the training and export
path and is not trained on real grasp outcomes. Train on your own cell's records before letting it
influence an order. A record carries the feature vector only when the predictor scored its grasp, so
collect with `robot.grasping.success_model.enabled: true` in the default `shadow` phase, which changes
no order; a log without the vectors is refused with a `ValueError`.

```bash
python -m src.robot.grasping.calibration.success_model_calibration train \
    --records logs/<your-cell>/grasp_records.jsonl --model-family gradient_boosted_trees \
    --artifact-dir assets/models/success_probability/my_cell
```

Without `--artifact-dir` the command writes over the shipped `v1`. Name your directory in
`robot.grasping.success_model.artifact_dir` as an absolute path: the pick service reads a relative one
from the working directory.

## Three families, one predictor

```
features -> a missing feature becomes its training mean -> standardise
         -> the family's model -> isotonic calibration -> probability
```

| Family | Artifact version | Model | For |
| --- | :--: | --- | --- |
| `logistic_regression` | 1 | linear, through a sigmoid | the synthetic bootstrap that ships |
| `gradient_boosted_trees` | 2 | a serialized tree ensemble, then a sigmoid | a real, nonlinear grasp-outcome corpus |
| `mlp` | 3 | standardised inputs, a ReLU network, one logit | the most expressive, if it wins the gate |

Every family ends in the same isotonic calibration, and a caller never sees which one it holds.
Calibration comes last because a raw model output is a ranking, not a probability, and every fence
below is stated in probability units. The feature vector is locked: `FEATURE_NAMES` in a fixed order
at `FEATURE_SCHEMA_VERSION` 1, and a change means a new schema version and a new model directory.

A challenger replaces the incumbent only through the gate in
[calibration/model_families.py](../../calibration/model_families.py): a Brier improvement of at least
0.005 with no AUROC regression, or an AUROC improvement of at least 0.02 with no Brier regression above
0.002 and no ECE regression above 0.005. Among admissible challengers the best ranker wins, and an AUROC
that cannot be graded, because a class is missing from the evaluation split, keeps the incumbent.

## The two rerankers

Both are off by default and fail safe, and neither changes `GraspPoint.score` or the input order except
through the tuple it returns.

| Fence on the ranking blend | Effect |
| --- | --- |
| `ranking_blend_weight` must lie in `[0.0, 0.5]`; a value outside is refused at load | geometry keeps at least half the influence |
| `ranking_blend_modes` is limited to the dense modes | it cannot fire in `easy` or `auto` |
| `lifecycle_phase: shadow`, the default, is a hard no-op | shadow observes and never reorders |
| a candidate without a shadow probability stops the blend | a partial reorder is worse than none |

The blend is convex: `(1 - w) * geometric + w * probability`. The uncertainty rerank subtracts weight
times the per-candidate uncertainty, under the same fences, after the blend and before the RL ranking
shadow, so that shadow sees the order that ran. It reads the corridor risk the calculator stamps on each
candidate; with that producer off it does nothing rather than reorder on an invented value. Its weight
defaults to 0.0 and runtime adaptation cannot change it, so only a person turns it on.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ArtifactSchemaError` from `load_success_probability_model` | a missing file, a schema or version mismatch | retrain with this feature schema |
| `None` from `try_load_shadow_success_context`, with a warning | the same, on the pick path | a pick never fails on a changed file; fix the artifact |
| the canary or active phase loads nothing | `promotion.json` fails `verify_promotion` | promote the artifact through the calibration gate |

`verify_promotion` checks the recorded verdict, that the metrics agree with their bounds, the locked
validation slice, thresholds no weaker than the locked ones, and the SHA-256 chain over the artifact
bytes. It stops an accident, a stray flag or a hurried operator; it is not proof against a forgery,
because whoever writes the file writes the fields it compares.

## Status

| Capability | Evidence |
| --- | --- |
| The predictor and the rerankers on a physical cell | never touched hardware |

No model trained on real grasp outcomes ships; the bootstrap proves the pipeline, not a prediction.

## Files

| Leaf | Holds |
| --- | --- |
| [_schema.py](_schema.py) | `FEATURE_SCHEMA_VERSION`, the artifact versions, `FEATURE_NAMES`, `MODE_BUCKETS`, `ArtifactSchemaError` |
| [_features.py](_features.py) | `extract_features`, `extract_features_from_grasp_point`: the only place the vector is built |
| [_model.py](_model.py) | `SuccessProbabilityModel`, `predict_proba`, the tree reader and the loaders |
| [_shadow.py](_shadow.py) | the shadow adapter, `try_load_shadow_success_context`, the metadata keys |
| [_blend.py](_blend.py), [_rerank.py](_rerank.py) | `maybe_blend_rerank_candidates`; `maybe_uncertainty_rerank_candidates` |

`__init__.py` re-exports the leaves, so one import path reaches them all.

## Details

- [scoring/](../README.md): the deterministic axes this sits above
- [calibration/](../../calibration/README.md): trains, gates and promotes the artifacts
- [rl/](../../rl/README.md): the offline ranking layer beside it
- [docs/grasping-math.md](../../../../../docs/grasping-math.md): the feature definitions and the
  within-group differencing property
