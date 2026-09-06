# Grasping types: the vocabulary every tier speaks

Frozen, self-validating carriers with no behaviour of their own. Four value objects and the
perception Protocols.

Nothing here imports another grasping tier, so the calculator, the frame resolver, refinement,
verification, recovery and the orchestrator can all depend on these names without creating a cycle.

| Module | Owns |
|---|---|
| [`grasp_point.py`](grasp_point.py) | `GraspPoint`, one candidate grasp, and `GraspFrame` |
| [`feedback.py`](feedback.py) | `GraspResult` and the `GraspFailureReason` enum: why the list came back empty |
| [`modes.py`](modes.py) | `GraspSamplingMode` and the bridge to the calculator's internal tri-state |
| [`perception.py`](perception.py) | `SegmentationLike`, `PerceptionFrame`, `PerceptionSource`, `CameraObservation`, `MultiCameraPerceptionSource` |

## `GraspPoint`, the candidate

A `GraspPoint` fully specifies how a parallel jaw should approach and close.

| Field | Meaning |
|---|---|
| `position` | The 3D anchor point, in millimetres, in the camera or the base frame |
| `approach` | The unit vector the gripper moves along toward the object. Top-down in BASE is `[0, 0, -1]` |
| `axis` | The unit vector along the line joining the two finger pads, perpendicular to `approach`. The gripper closes across it |
| `grip_width_mm` | The expected finger-pad separation at contact |
| `score` | Quality in `[0, 1]`, higher is better |
| `frame` | Which frame the vectors live in: `GraspFrame.CAMERA` or `GraspFrame.BASE` |
| `label`, `metadata` | Optional provenance and per-candidate telemetry |

The numerics contract is enforced rather than described. Distances are millimetres; `approach` and
`axis` are normalised at construction and a zero vector raises. The dataclass is frozen and the three
numpy arrays are marked read-only, so `gp.position[0] = x` raises instead of silently mutating a
candidate another tier is holding.

## `GraspResult`, and why nothing came back

The calculator is deliberately non-throwing for the ordinary no-grasp cases: an empty mask, all
candidates collided, no valid depth under the mask. A bare empty list would leave the caller unable to
choose between retry, rescan, escalation and abort, so the result carries a typed reason.

`GraspFailureReason` has 23 members, grouped by what a caller would do about them.

| Group | Reasons |
|---|---|
| Nothing to look at | `EMPTY_MASK`, `MASK_TOO_SMALL`, `NO_VALID_DEPTH`, `TARGET_LABEL_NOT_FOUND` |
| Perception too weak to trust | `LOW_DEPTH_CONFIDENCE`, `LOW_MASK_CONFIDENCE`, `HEAVY_OCCLUSION` |
| Candidates existed and every one was filtered | `NO_CANDIDATES_GENERATED`, `ALL_COLLIDED`, `ALL_OUT_OF_WORKSPACE`, `ALL_TABLE_CONFLICT`, `IK_FAILED`, `NO_VALID_GRASP` |
| A policy refused | `TOPOLOGY_RISK_REJECTED`, `SEMANTIC_REJECTED`, `DEFORMABLE_ROUTING_REQUIRED` |
| Try something different | `RESCAN_RECOMMENDED`, `TRY_NEXT_CANDIDATE`, `ACTIVE_PERCEPTION_RECOMMENDED` |
| The closed loop lost it | `TARGET_LOST_DURING_REFINE`, `REFINEMENT_DIVERGED` |
| The cell cannot move | `MOTION_PLAN_REFUSED`, `CONTROLLER_NOT_OPERATIONAL` |

The enum is coarse-grained on purpose. It is not a diagnostic taxonomy, only enough resolution for an
execution layer to branch on; the full root-cause classification lives offline in
[`../replay/failure_taxonomy.py`](../replay/failure_taxonomy.py).

The last two change what a cell does. `MOTION_PLAN_REFUSED` is what makes a planner refusal
recoverable, by rescan alone. `CONTROLLER_NOT_OPERATIONAL` is what stops the loop retrying into a
controller that has protective-stopped.

## `GraspSamplingMode`, a typed name for a tri-state

The calculator's internal switch is a tri-state `dense_sampling` of `True`, `False` or `None`. A
caller should not have to remember which boolean means what, so it configures a typed enum and the
bridge maps it.

```
GraspSamplingMode.AUTO          -> dense_sampling = None    (decide per scene)
GraspSamplingMode.SINGLE_OBJECT -> dense_sampling = False
GraspSamplingMode.DENSE_CLUTTER -> dense_sampling = True
```

Boolean callers keep working (`True` becomes `DENSE_CLUTTER`, `False` becomes `SINGLE_OBJECT`).
Strings are accepted for YAML and command-line usability against a closed allow-list; anything else
raises `ValueError` with an explicit hint rather than silently picking a mode.

## The perception snapshot, and the one with a join key

`PerceptionSource.acquire()` returns a `PerceptionFrame`: a depth map, the camera intrinsics, the
segmentations, and optionally an RGB image, a timestamp and the tool pose the frame was taken at.
`MultiCameraPerceptionSource` returns a tuple of `CameraObservation`, each pairing a `camera_id` with
one such frame.

Segmentations need only satisfy `SegmentationLike`, which is a `.mask` attribute. The orchestrator
does not care which model produced them, which is what lets a simulated ground-truth source and a
real detector-plus-segmenter source be interchangeable.

`camera_id` is a contract, not a label. It must match an entry in `robot.grasping.fusion.cameras`,
which is where that camera's own CAMERA to BASE calibration artifact is declared. A frame whose id
has no calibration entry cannot be placed in BASE, so it cannot be fused; the orchestrator drops it
and names the id, rather than guessing at a default extrinsic and fusing a cloud into the wrong place.

The two Protocols are separate on purpose. The single-camera one answers what the camera sees now;
the multi-camera one answers what the cameras see at the same moment, and that simultaneity is what
makes fusing a bin of moving parts sound.

A camera that failed to produce a frame is simply absent from the returned tuple: the Protocol has no
error channel by design. Which cameras were expected is config, so comparing expected against
delivered belongs to the caller that holds the config, in one place and under one policy
(`fusion.on_camera_unavailable`, which is `degrade` or `refuse`).

## See also

- [`../README.md`](../README.md) for the tier these types are the vocabulary of
- [`../geometry/README.md`](../geometry/README.md) for the numeric layer `GraspPoint` validates against
- [`../planning/README.md`](../planning/README.md) for `GraspPose`, the richer 6-DoF frame a
  `GraspPoint` becomes
- [`../telemetry/README.md`](../telemetry/README.md) for where a `GraspResult` ends up as a logged
  record
- [`../../../../docs/grasping-math.md`](../../../../docs/grasping-math.md) for what the numbers in
  these carriers mean
