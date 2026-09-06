# The learned success predictor

A runtime-pure estimate of the probability that a scored grasp succeeds, over a locked 23-feature
vector, with two rerankers on top. Numpy only at runtime: no scikit-learn and no joblib.

This package is not re-exported by `scoring/__init__`. Import it by its own path. That is deliberate:
a learned probability arriving through the same door as the deterministic scorers would look like one
of them.

## Six leaves behind one facade

`__init__.py` re-exports the leaves, so the import path is a single name.

| Leaf | Owns |
|---|---|
| [`_schema.py`](_schema.py) | The frozen constants (`FEATURE_SCHEMA_VERSION`, the three family artifact versions, `FEATURE_NAMES`, `MODE_BUCKETS`) and `ArtifactSchemaError` |
| [`_features.py`](_features.py) | `extract_features` and `extract_features_from_grasp_point`, the only place the 23-vector is built |
| [`_model.py`](_model.py) | `SuccessProbabilityModel`, `predict_proba`, the tree-ensemble reader and the artifact loaders |
| [`_shadow.py`](_shadow.py) | The shadow-mode runtime adapter, `try_load_shadow_success_context`, and the metadata keys |
| [`_blend.py`](_blend.py) | `maybe_blend_rerank_candidates`, the bounded convex blend |
| [`_rerank.py`](_rerank.py) | `maybe_uncertainty_rerank_candidates`, the conservative last word on ordering |

## Three families, one schema, one predictor

```
x -> impute a missing feature with its training mean -> standardise
  -> family -> isotonic calibration -> probability
```

| Family | Artifact version | Runtime path | Intended for |
|---|:--:|---|---|
| `logistic_regression` | 1 | A linear model through a sigmoid | The synthetic bootstrap that ships |
| `gradient_boosted_trees` | 2 | A serialized ensemble evaluated by a vectorised numpy traversal, then a sigmoid | A real, nonlinear grasp-outcome corpus |
| `mlp` | 3 | Standardised inputs, a ReLU network, one raw logit | The most expressive family, if it wins the gate |

Every family finishes with the same isotonic calibration, and a caller never sees which one it holds.
Calibration comes last because a raw model output is a ranking, not a probability, and every fence
downstream (the blend weight, the promotion thresholds) is stated in probability units.

The bootstrap stays logistic. On a small, logistic-structured synthetic dataset the more expressive
families overfit, so they are trained explicitly, on real data. Family selection is a fail-closed
gate in [`../../calibration/model_families.py`](../../calibration/model_families.py) and not a
preference: a challenger is promoted either on a Brier improvement of at least 0.005 with no AUROC
regression, or on an AUROC improvement of at least 0.02 with no meaningful Brier regression (0.002)
or ECE regression (0.005). Among admissible challengers the best ranker wins. An AUROC that cannot be
graded, because a class is missing from the evaluation split, keeps the incumbent.

## Two rerankers, and every fence on them

Both are off by default, both are fail-safe, and neither touches `GraspPoint.score` or the input
order except through its own returned tuple.

The ranking blend is a bounded convex blend of the geometric score with the learned probability.

| Fence | Effect |
|---|---|
| The convex weight is clamped to `[0.0, 0.5]` | Geometry always keeps at least half the influence |
| Locked to the dense modes | It cannot fire in `easy` |
| A hard no-op in the shadow lifecycle phase | Shadow means observe, and that is enforced rather than requested |
| It refuses to rerank if any candidate is missing a shadow probability | A partial reorder is worse than none |

The uncertainty rerank is subtractive, score minus weight times the per-candidate uncertainty, under
the same fences. It runs after the blend and before the reinforcement-learning ranking shadow, so the
shadow observes the order that was actually executed.

It is honest only because it consumes a real per-candidate signal, the corridor risk the
`GraspCalculator` stamps, and not a relabelling of the geometric score. With that producer off it
hard no-ops rather than reordering on a fabricated value. Its weight is also not mutable by runtime
adaptation: enabling the block stays observe-only until a person sets the weight.

## No artifact is stored here

`model.json` and `manifest.json` are produced by [`../../calibration/README.md`](../../calibration/README.md)
and [`../../rl/README.md`](../../rl/README.md), and live under
`assets/models/success_probability/<version>/`. What the repository ships is a synthetic bootstrap of
the logistic family: it exercises the training and export path and is not trained on real grasp
outcomes. Train on your own cell's records before letting any of this influence an order.

| Loader | On a schema mismatch |
|---|---|
| `load_success_probability_model` | Raises `ArtifactSchemaError` |
| `try_load_shadow_success_context` | Fails closed to `ctx=None`, with a logged warning and never an exception into the runtime path |

The two exist separately on purpose. An offline tool wants the loud failure; a live pick must not
crash because a file on disk changed shape.

The predictor's only reach across tiers is a lazy call to `calibration.model_promotion.verify_promotion`
on the canary and active paths. That call checks the recorded verdict, that the recorded metrics agree
with the recorded bounds, that the validation slice is the locked one, that the recorded thresholds
are not weaker than the locked thresholds, and that the artifact bytes still match the SHA-256 chain.
It closes the accident, a stray flag or a hurried operator, and it is not proof against a forgery: the
fields it compares are written by whoever wrote the file.

## See also

- [`../README.md`](../README.md) for the four deterministic axes this sits above
- [`../../calibration/README.md`](../../calibration/README.md) trains, gates and promotes the artifacts
- [`../../rl/README.md`](../../rl/README.md) is the offline layer this is the runtime cousin of
- [`../../../../../docs/grasping-math.md`](../../../../../docs/grasping-math.md) for the feature
  definitions and the within-group differencing property
