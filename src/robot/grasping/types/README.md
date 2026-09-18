# Grasp types (`src.robot.grasping.types`)

The words every grasping tier speaks: a grasp candidate, why no candidate came back, the sampling mode,
and what a camera hands the pick loop. Frozen, self-validating carriers with no behaviour of their own.

You meet these in the calculator's results and the pick service's reports; you build one yourself only
to hand a grasp to the robot or to plug in a perception source of your own. A BASE grasp becomes the
pose `Robot.pick` takes:

```python
import numpy as np
from src.robot.grasping import GraspFrame, GraspPoint

grasp = GraspPoint(
    position=np.array([450.0, 100.0, 40.0]),   # mm, in the robot's base frame
    approach=np.array([0.0, 0.0, -1.0]),       # straight down
    axis=np.array([1.0, 0.0, 0.0]),            # the jaws close along base X
    grip_width_mm=30.0,
    score=0.9,
    frame=GraspFrame.BASE,
)
print(grasp.pose())   # robot.pick(grasp.pose(), grasp.grip_width_mm) takes it; a CAMERA one is refused
```

Nothing here imports another grasping tier, so every tier can depend on these names without a cycle.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `GraspPoint` | a generator, or `GraspPoint(...)` | `pose()` | the tool `Pose` in the grasp's frame: +Z approach, +X closing axis |
| `GraspResult` | the calculator | read `candidates`, `reasons`, `is_success` | ranked candidates, or typed reasons for none |
| `GraspSamplingMode` | `resolve_grasp_sampling_mode(value)` | `mode_to_dense_sampling(mode)` | the calculator's `True`, `False` or `None` |
| `PerceptionSource` | your camera adapter | `acquire()` | `PerceptionFrame` |
| `MultiCameraPerceptionSource` | your rig, or `MappedCameraRig` | `acquire_all()` | a tuple of `CameraObservation` |

## `GraspPoint`, the candidate

| Field | Meaning |
| --- | --- |
| `position` | the anchor point in millimetres, in the camera or the base frame |
| `approach` | the unit vector the gripper moves along toward the part; top-down in BASE is `[0, 0, -1]` |
| `axis` | the unit vector joining the two finger pads, perpendicular to `approach`; the jaws close across it |
| `grip_width_mm` | the expected pad separation at contact |
| `score` | quality in `[0, 1]`, higher is better |
| `frame` | `GraspFrame.CAMERA` or `GraspFrame.BASE` |
| `label`, `metadata` | optional provenance and per-candidate telemetry |

The contract is enforced. `approach` and `axis` are normalised at construction and a zero vector
raises; a negative width or a score outside `[0, 1]` raises. The three arrays are read-only, so
`grasp.position[0] = x` raises instead of changing a candidate another tier holds.

## `GraspResult`, and why nothing came back

The calculator does not raise for an ordinary no-grasp case. A bare empty list would leave the caller
unable to choose between retry, rescan, escalation and abort, so the result carries typed
`GraspFailureReason`s, grouped here by what a caller does about them:

| Group | Reasons |
| --- | --- |
| Nothing to look at | `EMPTY_MASK`, `MASK_TOO_SMALL`, `NO_VALID_DEPTH`, `TARGET_LABEL_NOT_FOUND` |
| Perception too weak to trust | `LOW_DEPTH_CONFIDENCE`, `LOW_MASK_CONFIDENCE`, `HEAVY_OCCLUSION` |
| Every candidate was filtered | `NO_CANDIDATES_GENERATED`, `ALL_COLLIDED`, `ALL_OUT_OF_WORKSPACE`, `ALL_TABLE_CONFLICT`, `IK_FAILED`, `NO_VALID_GRASP` |
| A policy refused | `TOPOLOGY_RISK_REJECTED`, `SEMANTIC_REJECTED`, `DEFORMABLE_ROUTING_REQUIRED` |
| Try something different | `RESCAN_RECOMMENDED`, `TRY_NEXT_CANDIDATE`, `ACTIVE_PERCEPTION_RECOMMENDED` |
| The closed loop lost it | `TARGET_LOST_DURING_REFINE`, `REFINEMENT_DIVERGED` |
| The cell cannot move | `MOTION_PLAN_REFUSED`, `CONTROLLER_NOT_OPERATIONAL` |

The enum is coarse on purpose: enough for an execution layer to branch on. The full root-cause
classification lives offline in [`replay/failure_taxonomy.py`](../replay/failure_taxonomy.py). The last
two change what a cell does: a planner refusal is recovered by a rescan alone, and a controller that has
protective-stopped ends the loop instead of being retried.

## `GraspSamplingMode`

A typed name for the calculator's `dense_sampling` switch:

| Mode | `dense_sampling` | Strings accepted |
| --- | --- | --- |
| `AUTO` | `None`, decide per scene | `auto`, and `None` itself |
| `SINGLE_OBJECT` | `False` | `single`, `single_object`, and `False` |
| `DENSE_CLUTTER` | `True` | `dense`, `dense_clutter`, and `True` |

Strings are case-insensitive. Anything else, an integer included, raises `ValueError` naming the forms
that are accepted.

## What a camera hands the loop

`PerceptionSource.acquire()` returns a `PerceptionFrame`: a depth map, the lens matrix and the
segmentations, and optionally an RGB image, a timestamp, the tool pose at capture and the measured depth
before any grasp overwrite (`surface_depth_map`). A segmentation needs only a `.mask`
(`SegmentationLike`), so a simulated ground-truth source and a real detector are interchangeable.

`MultiCameraPerceptionSource.acquire_all()` returns one `CameraObservation` per camera that answered,
and the two Protocols are separate because fusing a bin needs the cameras at the same moment.
`camera_id` is a contract, not a label: it must name a camera the loop holds a resolver for, which on a
config-built cell is a rig id in `robot.grasping.fusion.cameras`. A frame whose id has no resolver is
dropped and named, never fused at a guessed position.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ValueError` from `GraspPoint` | a zero `approach` or `axis`, a negative width, a score outside `[0, 1]` | fix the value |
| `ValueError` from `resolve_grasp_sampling_mode` | a value outside the accepted forms | use a mode name or a bool |
| a camera absent from `acquire_all()` | it produced no frame; `MappedCameraRig.last_failures` says why | the loop applies `fusion.on_camera_unavailable` (`degrade` or `refuse`) |

## Status

| Capability | Evidence |
| --- | --- |
| The carriers and the reasons | measured in simulation: every Isaac pick carries them |
| `CONTROLLER_NOT_OPERATIONAL` | measured against real controller software: a protective stop in URSim |
| A real camera through `PerceptionSource` | never touched hardware: the RealSense source has never been fed a real frame |

## Files

| File | Holds |
| --- | --- |
| [`grasp_point.py`](grasp_point.py) | `GraspPoint`, `GraspFrame` |
| [`feedback.py`](feedback.py) | `GraspResult`, `GraspFailureReason` |
| [`modes.py`](modes.py) | `GraspSamplingMode`, `resolve_grasp_sampling_mode`, `mode_to_dense_sampling` |
| [`perception.py`](perception.py) | `PerceptionFrame`, `PerceptionSource`, `MultiCameraPerceptionSource`, `CameraObservation`, `MappedCameraRig` |

## Details

- [`planning/`](../planning/README.md) for `GraspPose`, the 6-DoF frame a candidate becomes, and
  [`geometry/`](../geometry/README.md) for the numbers it validates against.
- [`telemetry/`](../telemetry/README.md) for where a result ends up as a logged record.
- [The grasping maths](../../../../docs/grasping-math.md) for what the numbers mean.
- Tests: `tests/test_grasping_modes.py`, `tests/test_pick_loop_controller_state.py`,
  `tests/test_motion_plan_refusal_recovery.py`, `tests/test_pick_loop_target_label_gate.py`.
