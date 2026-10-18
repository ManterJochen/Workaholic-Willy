# Grasp scores (`src/robot/grasping/scoring`)

The record a candidate grasp's score travels in, the workspace box a grasp must stand in, and the signals
the calculator stamps beside its grasps: how much of the target its neighbours cover, how far the approach
is clear, how concave the target's outline is, and which labels a caller admits. The calculator's
support-footprint stage scores its own grasps ([generation/](../generation/README.md)) and writes each
score into a `GraspScoreBreakdown`.

The three scorers that rated a grasp on geometry, stability and reachability, their blend
(`GraspScoreWeights`, `score_grasp_pose`, `rank_grasp_poses`) and the force-closure certificate left on
2026-10-18 with the code nothing called. The silhouette stage, the last one that ranked with them, left on
2026-10-16, and so did the fourth axis, feasibility, with `robot.grasping.feasibility`: at its weight of
0.00 it moved no score.

```python
from src.robot.grasping.scoring import approach_clearance_mm, occlusion_ratio
from src.robot.grasping.scoring.topology_risk import topology_risk_from_mask

covered = occlusion_ratio(target_mask, [neighbour_mask])   # [0, 1]: how much of the target's hull is covered
risk = topology_risk_from_mask(target_mask)                 # 1 - solidity; None is no signal, not low risk
clear = approach_clearance_mm(point_mm, approach_axis, depth_mm, intrinsics)   # mm, at most 200 by default
print(covered, risk, clear)
```

The masks are HxW arrays of one shape. `point_mm` and `approach_axis` are in the camera frame, the axis
the direction the gripper travels toward the part. This package has no `python -m` entry.

## What the calculator reads

`GraspScoreBreakdown` carries `total_score`, the three category scalars `geometric_score`,
`stability_score` and `reachability_score`, the sub-scores per group, an optional collision result and
metadata, with a stable `to_dict()`. The support-footprint stage writes its score as `total_score` and
`geometric_score` and leaves the other two at 0.0, flagged `score_is_undecomposed` in `metadata`: its
score is one blend of measured physical margins, and those margins ride in
`components["support_footprint"]`.

`WorkspaceBox` is an axis-aligned box in the grasp's frame. The collision filter drops a candidate whose
position lies outside it and counts it as `rejected_workspace`.

The calculator stamps `occlusion_ratio` and `topology_risk_from_mask` into its telemetry on every call
where they can be computed. Each becomes a refusal only behind its own threshold
(`heavy_occlusion_threshold`, `topology_risk_threshold`), and a refusal names its reason code.
`SemanticPolicy` admits a candidate by an allow list, a deny list and a label-confidence floor; its
`evaluate` returns a `SemanticDecision` with a stable denial code.

The learned success predictor and its two rerankers (`success_probability/`), and the corridor analyzer
(`corridor.py`), left on 2026-10-16 with `robot.grasping.success_model` and `robot.grasping.occlusion`:
in the shadow the predictor changed nothing, and switched on the corridor changed no pick of the owner's
cell's replay bank.

## Status

| Capability | Evidence |
| --- | --- |
| The score record, the workspace box and the signals, in the pick service | measured in simulation |
| The same at a physical cell | run on a physical cell: the camera picks of a UR10 (CB3) with a wrist D415; no score from them is kept here |

These are heuristics, not ground truth. `topology_risk_from_mask` reads the outer outline only, so a
hole inside it does not register, and `approach_clearance_mm` reads one depth image. Units are
millimetres, rotations are XYZW quaternions, and every pose carries its frame.

## Files

| File | Holds |
| --- | --- |
| [`__init__.py`](__init__.py) | `GraspScoreBreakdown`, and the re-exports `WorkspaceBox`, `occlusion_ratio`, `approach_clearance_mm` |
| [`reachability_score.py`](reachability_score.py) | `WorkspaceBox` |
| [`occlusion.py`](occlusion.py) | `occlusion_ratio`, `approach_clearance_mm` |
| [`topology_risk.py`](topology_risk.py) | `topology_risk_from_mask`; import it by module path |
| [`semantic_policy.py`](semantic_policy.py) | `SemanticPolicy`, `SemanticDecision`; import them by module path |

## Details

- [grasping/](../README.md): the stack this scores for
- [docs/grasping-math.md](../../../../docs/grasping-math.md): the formula of the score the calculator
  ranks by
- [planning/](../planning/README.md) supplies `GraspPose`; [collision/](../collision/README.md) supplies
  `GraspCollisionResult` and the filter that holds a grasp to `WorkspaceBox`
