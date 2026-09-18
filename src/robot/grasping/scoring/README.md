# Grasp scores (`src/robot/grasping/scoring`)

Deterministic scorers that rate a candidate grasp on four axes and fold them into one ranked list, plus
the force-closure certificate and a learned success predictor. A cell reaches it through the
calculator, whose silhouette stage ranks with these scores; the default support-footprint stage keeps
its own score ([generation/](../generation/README.md)). Call it directly to rank poses of your own.
No scorer here can reject a grasp: refusal belongs to the calculator's filters and the safety layer, and
a scorer that could veto would put a heuristic above them.

```python
from src.robot.grasping import (
    GraspScoreWeights, estimate_surface_normals, find_antipodal_pairs,
    generate_grasp_poses, rank_grasp_poses,
)

normals = estimate_surface_normals(points_mm)          # points_mm: an (N, 3) cloud, millimetres
pairs = find_antipodal_pairs(points_mm, normals, max_width_mm=85.0)
ranked = rank_grasp_poses(generate_grasp_poses(pairs), weights=GraspScoreWeights())
if ranked:
    best = ranked[0]                                   # a GraspScoreBreakdown, best first
    print(best.total_score, best.to_dict())
```

`rank_grasp_poses` sorts best first and breaks ties by input order, so the same input always ranks the
same way. This package has no `python -m` entry.

## The four axes

| File | Measures | Default weight |
| --- | --- | :--: |
| [geometric_score.py](geometric_score.py) | width fit, approach quality, contact geometry | 0.50 |
| [stability_score.py](stability_score.py) | a confidence, contact and table-clearance heuristic | 0.35 |
| [reachability_score.py](reachability_score.py) | workspace-box and approach-axis geometry, not an IK query | 0.15 |
| [feasibility_score.py](feasibility_score.py) | IK quality, joint margin, swept approach, corridor risk | 0.00 |

Each axis is a `*_grasp_score` and `*_score_components` pair driven by a frozen `*Config`.
`score_grasp_pose` blends them into a `GraspScoreBreakdown`: the four scalars, the sub-scores per
group, an optional collision result and metadata, with a stable `to_dict()`. The feasibility axis
ships at weight 0.00, so `total_score` is the three-axis blend; with none of its sub-signals enabled it
returns the neutral 0.5. Its re-rank, `rerank_breakdowns_with_feasibility`, runs after the IK filter,
because the IK quality of a candidate exists only once that filter has run.

| Other module | Holds |
| --- | --- |
| [force_closure.py](force_closure.py) | the Coulomb friction-cone certificate: `certify_contact_pair`, `friction_threshold` |
| [occlusion.py](occlusion.py) | `occlusion_ratio`, `approach_clearance_mm` |
| [corridor.py](corridor.py), [topology_risk.py](topology_risk.py), [semantic_policy.py](semantic_policy.py) | analyzers behind flags; import them by module path |
| [success_probability/](success_probability/README.md) | the learned predictor and the two rerankers on top of it |

## The learned predictor

A numpy-only model estimates the probability that a scored grasp succeeds, from 23 locked features,
and two rerankers can reorder candidates with it. Both rerankers are off by default and fail safe, and
the blend keeps the geometric score at least half the weight. It is not re-exported here: import it as
`src.robot.grasping.scoring.success_probability`, so a learned probability never arrives through the
same door as the deterministic scores. The repository ships a synthetic bootstrap model, not one trained
on real outcomes. [success_probability/](success_probability/README.md) has the families, the fences
and the loaders.

## Status

| Capability | Evidence |
| --- | --- |
| The calculator these scores rank for, in the pick service | measured in simulation |
| The same at a physical cell | never touched hardware |

These are heuristics, not ground truth. `reachability_score` is box and axis geometry rather than an IK
query. `stability_score` is a heuristic; `force_closure.certify_contact_pair` proves force closure under
a contact model only, which is no substitute for force feedback. `corridor` casts a small bundle of
rays, so a thin obstacle between two rays can be missed. Units are millimetres, rotations are XYZW
quaternions, and every pose carries its frame.

## Files

| File | Holds |
| --- | --- |
| `__init__.py` | `GraspScoreWeights`, `GraspScoreBreakdown`, `score_grasp_pose`, `rank_grasp_poses` and the feasibility re-rank |
| the axis files above | one scorer each, with its config |
| the other modules above | force closure, occlusion, the flag-gated analyzers, the learned predictor |

## Details

- [grasping/](../README.md): the stack this ranks for
- [docs/grasping-math.md](../../../../docs/grasping-math.md): every scorer's formula and the friction-cone
  derivation
- [planning/](../planning/README.md) supplies `GraspPose`; [collision/](../collision/README.md) supplies
  `GraspCollisionResult`
- [calibration/](../calibration/README.md) trains and promotes the predictor's artifacts, and
  [rl/](../rl/README.md) is the offline ranking layer beside it
