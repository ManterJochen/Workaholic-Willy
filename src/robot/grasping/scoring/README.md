# Grasp scoring

Deterministic pure-function scorers that rate a `GraspPose` on four axes and fold them into one
ranked list of `GraspScoreBreakdown`, plus an offline-trained success predictor the runtime can run
in shadow, canary or active mode.

Hard rejection is a calculator and safety concern, not a scoring one. Every scorer here only ranks;
none of them can veto. A scorer that could veto would put a heuristic above the safety layer.

## The four axes

| File | Measures | Default weight |
| --- | --- | :--: |
| [`geometric_score.py`](geometric_score.py) | Width fit, approach quality, contact geometry | 0.50 |
| [`stability_score.py`](stability_score.py) | A confidence, contact and table-clearance heuristic | 0.35 |
| [`reachability_score.py`](reachability_score.py) | Workspace-box and approach-axis geometry, not an IK query | 0.15 |
| [`feasibility_score.py`](feasibility_score.py) | IK quality, joint margin, swept approach, corridor risk | 0.00 |

Each axis is a `*_grasp_score` and `*_score_components` pair driven by a frozen `*Config`.
`score_grasp_pose` blends them into a `GraspScoreBreakdown`, which carries the four scalars, a
per-group sub-score dictionary, an optional collision result and metadata, with a stable `to_dict()`
telemetry layout. `rank_grasp_poses` scores and sorts best first, breaking ties deterministically by
input index.

The feasibility axis is the additive fourth one, and it ships at weight `0.00` so `total_score` stays
identical to the three-axis blend. Every one of its sub-signals is off, and with none enabled it
returns the neutral value `0.5`, which leaves the weighted average unchanged. Its re-rank entry point
is `rerank_breakdowns_with_feasibility`, which runs after the reachability filter, because
per-candidate IK quality only exists once that filter has run.

| Other module | Role |
| --- | --- |
| [`force_closure.py`](force_closure.py) | The analytical Coulomb friction-cone certificate: `certify_contact_pair`, `friction_threshold` |
| [`occlusion.py`](occlusion.py) | `occlusion_ratio`, `approach_clearance_mm` |
| [`corridor.py`](corridor.py), [`topology_risk.py`](topology_risk.py), [`semantic_policy.py`](semantic_policy.py) | Flag-gated analyzers. Reach them by their module path; they are not in this package's public list |
| [`success_probability/`](success_probability/README.md) | The runtime-pure learned predictor and the two rerankers |

## The learned predictor

A runtime-pure model, numpy only with no scikit-learn and no joblib, that estimates the probability
a scored grasp succeeds:

```
23 features -> impute a missing feature with its training mean -> standardise
            -> model family -> isotonic calibration -> probability
```

The feature vector is locked: `FEATURE_NAMES` has 23 entries in a fixed order, and
`FEATURE_SCHEMA_VERSION` is 1. Changing an entry or its position means a new schema version and a new
model directory. Three families share that schema and one runtime predictor, distinguished by
artifact version: logistic regression at 1, gradient-boosted trees at 2 and a ReLU multi-layer
perceptron at 3. A caller never sees which one it holds.

The package is split into leaves (`_schema`, `_features`, `_model`, `_shadow`, `_blend`, `_rerank`)
behind a re-exporting `__init__`. Import it by its own path: it is deliberately not re-exported by
`scoring`, because a learned probability arriving through the same door as the deterministic scorers
would look like one of them.

Two runtime rerankers sit on top. Both are fail-safe, both are off by default, and neither ever
touches `GraspPoint.score` or the input order except through its own returned tuple.

- The ranking blend (`maybe_blend_rerank_candidates`) is a bounded convex blend of the geometric
  score with the learned probability. The weight is clamped to `[0.0, 0.5]`, so geometry keeps at
  least half the influence whatever a config says; it is locked to the dense modes, so it cannot fire
  in `easy`; it is a hard no-op in the shadow lifecycle phase; and it refuses to rerank at all if any
  candidate is missing a shadow probability, because a partial reorder is worse than none.
- The uncertainty rerank (`maybe_uncertainty_rerank_candidates`) is the conservative last word on
  ordering. It is subtractive, score minus weight times the per-candidate uncertainty, under the same
  fences, and it runs after the blend and before the reinforcement-learning ranking shadow so that
  the shadow observes the order that was actually executed.

The uncertainty rerank is honest only because it consumes a real per-candidate signal, the corridor
risk the calculator stamps, rather than a relabelling of the geometric score. With that producer off
it hard no-ops instead of reordering on a fabricated value. Its weight is also deliberately not
mutable by runtime adaptation, so enabling the block stays observe-only until a person sets a weight.

## Usage

```python
from src.robot.grasping.scoring import GraspScoreWeights, rank_grasp_poses

ranked = rank_grasp_poses(poses, weights=GraspScoreWeights())  # best first
best = ranked[0]                                               # GraspScoreBreakdown
print(best.total_score, best.to_dict())
```

```python
# The learned predictor is imported directly, not through `scoring`:
from src.robot.grasping.scoring.success_probability import (
    load_success_probability_model,   # raises ArtifactSchemaError on any mismatch
    try_load_shadow_success_context,  # fail-safe: returns ctx=None on a missing or bad artifact
)
```

## Traps

Defaults are conservative no-ops. The feasibility weight is `0.00` and both rerankers are disabled.
Turning any of them on is an explicit opt-in with bounded behaviour, and the uncertainty rerank needs
a weight set by hand on top of that.

These are heuristics, not ground truth. `reachability_score` is workspace-box and approach-axis
geometry rather than a real IK query. `stability_score` is a confidence, contact and clearance
heuristic; the analytical path is `force_closure.certify_contact_pair`, which proves force closure
under a contact model only and is no substitute for hardware force feedback. `corridor` casts a small
ray bundle, so a thin obstacle between two rays can be missed.

No trained artifact is produced here. `model.json` and `manifest.json` are written by
[`../calibration/README.md`](../calibration/README.md) and [`../rl/README.md`](../rl/README.md) and
live under `assets/models/success_probability/`. What ships with the repository is a synthetic
bootstrap of the logistic family, which exercises the pipeline and is not trained on real grasp
outcomes. A missing or malformed artifact fails closed to `ctx=None` with a logged warning and never
raises into the runtime path; the strict loader raises `ArtifactSchemaError` on any schema mismatch.

Determinism and telemetry. The axis scorers, `score_grasp_pose` and `rank_grasp_poses` are
deterministic pure functions, and shadow annotation never changes ordering or `GraspPoint.score`.
Units are millimetres, rotations are XYZW quaternions, and every pose is frame-tagged.

There is no `python -m` entry point here. The predictor's only reach across tiers is a lazy call to
`calibration.model_promotion.verify_promotion` on the canary and active paths.

## See also

- [`../README.md`](../README.md) for the tier this package lives in
- [`../../../../docs/grasping-math.md`](../../../../docs/grasping-math.md) for every scorer's formula
  and the friction-cone derivation
- [`../planning/README.md`](../planning/README.md) supplies `GraspPose` and the IK quality metrics
- [`../collision/README.md`](../collision/README.md) supplies `GraspCollisionResult`
- [`../rl/README.md`](../rl/README.md) is the offline ranking and candidate policy this predictor is
  the runtime cousin of
- [`../calibration/README.md`](../calibration/README.md) trains and promotes the model artifacts
