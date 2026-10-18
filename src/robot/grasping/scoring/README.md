# Grasp scores (`src/robot/grasping/scoring`)

Deterministic scorers that rate a candidate grasp on three axes and fold them into one ranked list, plus
the force-closure certificate. The calculator's support-footprint stage keeps its own score
([generation/](../generation/README.md)); the silhouette stage, which ranked with these, left on 2026-10-16. Call
it directly to rank poses of your own.
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

## The three axes

| File | Measures | Default weight |
| --- | --- | :--: |
| [geometric_score.py](geometric_score.py) | width fit, approach quality, contact geometry | 0.50 |
| [stability_score.py](stability_score.py) | a confidence, contact and table-clearance heuristic | 0.35 |
| [reachability_score.py](reachability_score.py) | workspace-box and approach-axis geometry, not an IK query | 0.15 |

Each axis is a `*_grasp_score` and `*_score_components` pair driven by a frozen `*Config`.
`score_grasp_pose` blends them into a `GraspScoreBreakdown`: the three scalars, the sub-scores per
group, an optional collision result and metadata, with a stable `to_dict()`. A fourth axis, feasibility
(IK quality, joint margin, swept approach, corridor risk), shipped at weight 0.00 and left on 2026-10-16
with its re-rank and `robot.grasping.feasibility`: at that weight it moved no score.

| Other module | Holds |
| --- | --- |
| [force_closure.py](force_closure.py) | the Coulomb friction-cone certificate: `certify_contact_pair`, `friction_threshold` |
| [occlusion.py](occlusion.py) | `occlusion_ratio`, `approach_clearance_mm` |
| [topology_risk.py](topology_risk.py), [semantic_policy.py](semantic_policy.py) | analyzers behind flags; import them by module path |

The learned success predictor and its two rerankers (`success_probability/`), and the corridor analyzer
(`corridor.py`), left on 2026-10-16 with `robot.grasping.success_model` and `robot.grasping.occlusion`:
in the shadow the predictor changed nothing, and switched on the corridor changed no pick of the owner's
cell's replay bank.

## Status

| Capability | Evidence |
| --- | --- |
| The calculator these scores rank for, in the pick service | measured in simulation |
| The same at a physical cell | run on a physical cell: the camera picks of a UR10 (CB3) with a wrist D415; no score from them is kept here |

These are heuristics, not ground truth. `reachability_score` is box and axis geometry rather than an IK
query. `stability_score` is a heuristic; `force_closure.certify_contact_pair` proves force closure under
a contact model only, which is no substitute for force feedback. Units are millimetres, rotations are
XYZW quaternions, and every pose carries its frame.

## Files

| File | Holds |
| --- | --- |
| `__init__.py` | `GraspScoreWeights`, `GraspScoreBreakdown`, `score_grasp_pose`, `rank_grasp_poses` |
| the axis files above | one scorer each, with its config |
| the other modules above | force closure, occlusion, the flag-gated analyzers |

## Details

- [grasping/](../README.md): the stack this ranks for
- [docs/grasping-math.md](../../../../docs/grasping-math.md): every scorer's formula and the friction-cone
  derivation
- [planning/](../planning/README.md) supplies `GraspPose`; [collision/](../collision/README.md) supplies
  `GraspCollisionResult`
