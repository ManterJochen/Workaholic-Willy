# The second look and the proof of a hold (`src/robot/grasping/closed_loop`)

Looks again before the jaws close (refinement), checks after the grasp that something is held
(verification), and picks where a camera should look next. You reach it through the pick service in
the `closed_loop` and `dense_autonomous` grasp modes. It moves nothing itself: the service drives every
move, through the safety preflight.

```yaml
robot:
  grasping:
    default_mode: closed_loop   # a mode that asks for refinement and verification
    closed_loop:
      enabled: true             # builds the refiner
    verification:
      enabled: true             # builds the verifier
```

Both blocks ship `false`, and the `easy`, `auto` and `dense_clutter` modes keep refinement and
verification off whatever the blocks say. A mode that demands them on a cell without them refuses the
pick with a `MODE_NOT_AVAILABLE` outcome rather than running open loop. The `verification_heavy` preset
sets the mode ([grasping/](../README.md)); `load_tree().with_values({...})` checks the same keys in
memory before you edit a file.

## Refinement

The service takes a frame, picks the best grasp, drives to the pre-grasp standoff, takes a second frame
from there and hands it to `DefaultPreGraspRefiner.refine`, which re-identifies the target, recomputes
the grasp and accepts the result only inside the bounds. The bounds are
`robot.grasping.closed_loop.max_position_correction_mm` (20 mm by default),
`max_orientation_correction_deg` (15 degrees) and `max_grip_width_correction_mm` (20 mm). A large,
confident correction toward a misidentified object is worse than none, so a breach executes the
original grasp. `pregrasp_rescan: false` skips the standoff move, for a fixed camera whose view does
not change.

| `RefinementOutcome` | Means |
| --- | --- |
| `ACCEPTED` | the refined grasp replaces the first one |
| `TARGET_LOST` | no segmentation cleared `target_match_iou_threshold` (0.3 by default) |
| `DIVERGED` | the correction breached a bound; the original grasp runs |
| `NO_GRASP` | the recompute found no candidate, handled like any no-grasp outcome |
| `SKIPPED` | refinement is off |

Two trackers re-identify the target. `IoUCentroidTargetTracker` matches in the image, by mask overlap
and centroid distance, and loses the target when the camera moves far, as a wrist camera does.
`WorldSpacePoseTracker` matches the base-frame centroid, so a moving viewpoint does not matter, and
falls back to the image tracker when the 3D signal is missing. The refiner needs the same
`camera_to_base` as the first computation, or the two grasps it compares sit in different frames.

## Verification

Verification runs after the execution policy reports `EXECUTED`. A config-built cell verifies with a
`CompositeGraspVerifier` over `ObjectDetectingGripperVerifier` (does the gripper report an object) and
`WidthDeltaGripperVerifier` (did the jaws stop on something), plus `VisionTargetDisplacementVerifier`
(is the target gone from the scene) when `verification.post_lift_vision_check` is on.

| `VerificationOutcome` | Means |
| --- | --- |
| `PASSED` | evidence of a hold: object detected, a plausible jaw width, the target gone |
| `FAILED` | evidence of a miss: empty jaws, a full close, the target still in place |
| `INCONCLUSIVE` | nothing could be measured: no gripper feedback, no detection, no frame |
| `SKIPPED` | verification is off |

`INCONCLUSIVE` is a failure to observe, not a failed grasp. In a config-built cell
`verification.require_all_conclusive` decides it: `false`, the default, counts it as a pass and logs a
warning, because a cell without sensing would otherwise fail every pick; `true` demands a measurement.
`verification.fail_closed` decides the same question for a verifier a Python caller supplies.

The width check reads the gripper's physical closed width, `robot.gripper.closed_width_mm`, and not
the policy floor `min_width_mm`. With the shipped defaults the floor of 5.0 mm would put the empty-jaw
threshold at 7 mm, where a held 6 mm part reads as empty.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| outcome `MODE_NOT_AVAILABLE` | the mode demands refinement or verification the cell did not build | enable `closed_loop` and `verification` |
| `FAILED`, reason `no_end_effector_built` | the build substituted a stand-in gripper for the one configured | fix the arm and gripper pairing |
| `FAILED`, reason `no_end_effector_configured` | `robot.gripper.vendor: none` | fit a gripper, or turn verification off |
| `DIVERGED` | the refined grasp moved further than a bound allows | check the tracker and the calibration |

An empty flange is knowledge, not missing evidence, so both gripper refusals are `FAILED` and never
`INCONCLUSIVE`, and no setting turns them into a pass.

## Where to look next

`ScoringViewpointPlanner` proposes a small fixed set of viewpoints around the current tool position:
four lateral offsets, two diagonals and, when `vertical_offset_mm` is positive, a lift. It keeps a
history, prefers views unlike the poor ones already taken, scores each against a `ViewScoringPolicy`
and filters it through a `ViewpointSafetyCheck`, a Protocol a caller can back with the safety
preflight without this module importing the safety layer. It returns `None` when the budget is spent or
every candidate is refused. The `LateralOffsetViewpointPlanner` in [loop/](../loop/README.md) is the
simpler fallback: four fixed offsets in a fixed order.

## Status

| Capability | Evidence |
| --- | --- |
| Refinement, verification and next-best view in the pick service | measured in simulation, switched on per flag by the runners |
| The same on a physical arm | never touched hardware |

## Files

| File | Holds |
| --- | --- |
| `refinement.py` | `RefinementPolicy`, the `PreGraspRefiner` Protocol, `DefaultPreGraspRefiner`, `RefinementReport`, `RefinementOutcome` |
| `target_tracking.py` | `TargetIdentity`, the `TargetTracker` Protocol and the two trackers |
| `verification.py` | `GraspVerificationPolicy`, the `GraspVerifier` Protocol, the verifiers, `VerificationOutcome` |
| `active_perception.py` | `ScoringViewpointPlanner`, `ViewScoringPolicy`, `ViewpointHistory`, the viewpoint safety checks |

## Details

- [grasping/](../README.md): the grasp modes and the presets that switch this package on
- [motion/](../motion/README.md): the execution policy whose `EXECUTED` starts verification
- [recovery/](../recovery/README.md): what happens after `FAILED`
- [multiview/](../multiview/README.md): the other way to look again, with more cameras instead of more time
- [docs/grasping-config-reference.md](../../../../docs/grasping-config-reference.md): the `closed_loop`
  and `verification` blocks and the modes they fire in
