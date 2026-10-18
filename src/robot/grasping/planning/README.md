# Grasp planning (`src.robot.grasping.planning`)

The 6-DoF pose a parallel-jaw candidate is built as, the one rigid transform that carries it into another
frame, the `IKService` contract a real reachability solver fulfils, and the multi-finger planners. Pure
NumPy geometry; it plans poses, not trajectories.

You reach it through the grasp calculator: its support-footprint stage builds every candidate as a
`GraspPose`, and its IK filter hands each one, through `transform_grasp_pose`, to the `IKService` it was
given. The jaw poses built from contact pairs, the standoff, lift and waypoint helpers, the suction recipe
and the workspace-box IK stub left on 2026-10-18 with the code nothing called; the execution policy in
[`motion/`](../motion/README.md) builds its own standoff and lift. Call the package directly to carry a
pose of your own into the base frame and ask a solver about it:

```python
import numpy as np
from src.geometry import Frame
from src.robot.grasping.planning import GraspPose, IKResult, transform_grasp_pose

pose = GraspPose(
    position_mm=np.array([10.0, 0.0, 400.0]),
    rotation_matrix=np.eye(3),                     # columns: closing, binormal, approach
    grip_width_mm=40.0, score=0.8, confidence=0.8,
    contacts=(np.array([-10.0, 0.0, 400.0]), np.array([30.0, 0.0, 400.0])),
    frame=Frame.CAMERA,
)
in_base = transform_grasp_pose(pose, T_cam_to_base, target_frame=Frame.BASE)   # a validated 4x4


class ReachOf800mm:                                # any object with query(pose) -> IKResult is an IKService
    def query(self, pose: GraspPose) -> IKResult:
        reachable = bool(np.linalg.norm(pose.position_mm) <= 800.0)
        return IKResult(reachable=reachable, reason=None if reachable else "beyond_800_mm")


print(in_base.frame, ReachOf800mm().query(in_base))
```

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `GraspPose` | the calculator's support-footprint stage, or `GraspPose(...)` | `as_pose()`, `to_dict()` | a `Pose`, or a dict |
| `IKService` | an adapter in the execution layer, or any object with `query` | `query(pose)` | `IKResult` |
| `MultiContactGraspPlanner` | `ParallelJawContactPlanner`, `RadialMultiFingerPlanner` | through `GraspCalculator.plan_multifinger` | `MultiContactGrasp` candidates |

`GraspPose.rotation_matrix` has the columns `[closing, binormal, approach]` in the pose's frame; the
quaternion and the three axes are derived. Construction is strict: finite, orthonormal with determinant
+1, scores in `[0, 1]`, arrays read-only. Every pose is millimetres with an XYZW rotation and a tagged
frame.

## Three grasp value objects, on purpose

Suction and the jaw share the position half of a pick, not the orientation half. A `GraspPose` carries a
full frame built from a closing axis and an approach. A `SuctionGrasp` carries one contact and a press
direction, and the cup is round, so one wrist orientation serves. A `MultiContactGrasp` carries N
contacts and its own per-finger standoff. That is why there is no single shared pose type.

## What it deliberately leaves out

It imports [`src.geometry`](../../../geometry/README.md) and, from the grasping stack, only
[`contacts/`](../contacts/README.md), the [`geometry/`](../geometry/README.md) validators and the shared
constants: never a driver, a motion planner, a collision backend or a scorer. Obstacle-aware motion
happens below the driver, and self-collision is checked in the safety preflight.

Frame handling stays with the calculator. `transform_grasp_pose(pose, T, target_frame=...)` is the one
rigid transform offered; it re-orthonormalises the composed rotation, so a calibration matrix carrying
the slack `geometry.validate_transform` allows still yields a valid pose, and stamps
`metadata["source_frame"]`.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ValueError` from `GraspPose` | a non-finite position, a rotation that is not orthonormal, a score outside `[0, 1]`, a negative grip width | fix the input; the message names it |
| `TypeError` from the calculator's IK filter | an `IKService` whose `query` returns anything but an `IKResult` | return an `IKResult`, never a bare `bool` or `None` |

## Traps

- With no `IKService` the calculator runs no IK filter; a `WorkspaceBox` handed to it is then the only
  reach check, and it reads the position alone. Real reachability comes from the execution layer's
  `IKService` adapters, and there is no config key for one: a boot script passes it to the calculator
  ([guide 05](../../../../docs/guide/05-pick-loop.md)).
- The IK filter asks the service in the frame the calculator returns: BASE when it was handed a camera
  to base transform, CAMERA otherwise. An adapter must be configured for that frame. A pose that still
  fails `GraspPose` after the transform is dropped and counted as `rejected_ik`.
- Multi-finger planning is heuristic. It runs through `GraspCalculator.plan_multifinger`, apart from
  `compute()`, which stays parallel-jaw. `RadialMultiFingerPlanner` scores radial uniformity and
  coverage, does not prove force closure, and tags its results `confidence_kind="heuristic"`.
- `GripperKind.VACUUM` is telemetry only: a suction grasp is a single-contact `SuctionGrasp`.

## Status

| Capability | Evidence |
| --- | --- |
| `GraspPose` and `transform_grasp_pose` | measured in simulation: the grasp calculator builds its candidates with them in the Isaac picks |
| Multi-finger planners | never touched hardware: no multi-finger hand is driven by this code |

## Files

| File | Holds |
| --- | --- |
| `grasp_pose.py` | `GraspPose` |
| `reachability.py` | `IKService`, `IKResult`, `transform_grasp_pose` |
| `multifinger.py` | `FingerKinematicSpec`, `GripperKind`, `MultiContactGrasp`, `MultiContactPlanRequest`, the planners |

## Details

- [`contacts/`](../contacts/README.md) for the antipodal pairs the parallel-jaw contact planner builds
  on, and [`collision/`](../collision/README.md) for the filter the poses then pass.
- [`motion/`](../motion/README.md) for the policy that drives a grasp at the arm, and
  [`suction/`](../suction/README.md) for `SuctionGrasp`.
- [`safety/planning/`](../../safety/planning/README.md) for the trajectory planner this package does not
  import.
- [The grasping maths](../../../../docs/grasping-math.md) for the pose construction.
- Tests: `tests/robot/execution/test_ik_service_conformance.py`, `tests/robot/execution/test_execution_ik_service.py`,
  `tests/robot/grasping/test_multifinger_planner.py`, `tests/robot/grasping/test_grasping_boundaries.py`.
