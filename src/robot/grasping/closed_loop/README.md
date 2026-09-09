# Closed loop (`src.robot.grasping.closed_loop`)

Two second chances around a grasp: look again before closing the jaws (refinement), and prove
something is actually held after (verification). Plus the planner that decides where to look next.

Everything here is off by default. `robot.grasping.closed_loop.enabled` and
`robot.grasping.verification.enabled` both ship `false`, and the `easy`, `auto` and `dense_clutter`
mode profiles set `refinement_enabled` and `verification_enabled` to false regardless. Only the
`closed_loop` and `dense_autonomous` profiles ask for either.

## What this package guarantees

It owns no motion. The refiner is a pure perception and geometry computation: a target identity, a
re-acquired `PerceptionFrame` and a `GraspCalculator` in, a `RefinementReport` out. The robot side
of the two-scan workflow lives in `AutonomousGraspService`, so there is one place where motion
intent is choreographed and gated by `SafetyPreflight`.

A correction is accepted only when it is bounded. An unbounded improvement derived from a
mis-identified target is worse than no refinement, so a breach of any configured bound returns
`DIVERGED` and the caller executes the original grasp instead of a large, confident correction
toward the wrong object.

A verifier that could not measure anything says so. `INCONCLUSIVE` is a distinct outcome from
`PASSED`, and the operator decides which way it falls.

## The public surface

| Module | Owns |
| --- | --- |
| `refinement.py` | `RefinementPolicy`, `PreGraspRefiner` Protocol, `DefaultPreGraspRefiner`, `RefinementReport`, `RefinementOutcome` |
| `target_tracking.py` | `TargetIdentity`, the `TargetTracker` Protocol and its two implementations |
| `verification.py` | `GraspVerificationPolicy`, `GraspVerifier` Protocol and five implementations, `GraspVerificationContext`, `GraspVerificationReport`, `VerificationOutcome` |
| `active_perception.py` | `ViewScoringPolicy`, `ScoringViewpointPlanner`, `ViewpointHistory`, `ViewpointSignals`, and the `ViewpointSafetyCheck` Protocol |

### Two-scan refinement

The workflow is: acquire a frame, pick the best grasp, drive to the pre-grasp standoff, acquire a
second frame from there, re-identify the target in it, recompute the grasp, and accept the result
only if the correction is inside the bounds. `DefaultPreGraspRefiner.refine` performs the steps from
re-identification onward.

`RefinementOutcome` has five values. `ACCEPTED` means the refined grasp replaces the initial one.
`TARGET_LOST` means the tracker found no segmentation clearing `target_match_iou_threshold`.
`DIVERGED` means the correction breached a bound. `NO_GRASP` means the recompute produced no
candidates, which the caller treats like an ordinary no-grasp outcome. `SKIPPED` means the policy is
disabled.

The bounds are operator config with schema defaults of 20 mm of position, 15 degrees of orientation
and 20 mm of grip width per refinement.

### Re-identifying the target across a moved camera

Between the first scan and the standoff scan the target shifts in the image, and it shifts a long
way for a wrist camera that moves with the arm.

| Tracker | Matches by | Cost |
| --- | --- | --- |
| `IoUCentroidTargetTracker` | image space: highest mask IoU, centroid distance as tie-break | cheap, but loses the target when the viewpoint moves |
| `WorldSpacePoseTracker` | 3D BASE-frame centroid distance, so viewpoint invariant | a strict superset; falls back to the image-space tracker when the 3D signal is missing |

Trackers live in their own module and are injected into the refiner rather than hard-wired, so the
refiner and the matching strategy stay independently testable.

### Post-grasp verification

Verification runs after `GraspExecutionPolicy.execute` has returned `EXECUTED`. The execution policy
keeps its own close-time check against an `ObjectDetectingGripper`; this is the second, richer
question.

| `VerificationOutcome` | Means |
| --- | --- |
| `PASSED` | positive evidence the grasp succeeded: object detected, jaw width plausible, target gone from the scene |
| `FAILED` | positive evidence the grasp failed: empty jaws, a full close, the target still visible |
| `INCONCLUSIVE` | the verifier could not collect the evidence it needs: no gripper, no detection capability, no perception frame |
| `SKIPPED` | the policy itself is disabled |

`GraspVerificationPolicy.fail_closed` maps `INCONCLUSIVE` either to a pass or to a
`VERIFICATION_FAILED` outcome. Collapsing the two would file a verifier that could not tell as one
that saw the grasp succeed, which is the silent success this layer exists to prevent. A second switch,
`require_all_conclusive`, decides whether a sensorless cell that can never measure counts as a pass;
it ships `false`, because the alternative is a cell that fails every pick including the good ones.

Five verifiers ship and compose: `NoOpVerifier`, `ObjectDetectingGripperVerifier` (the SDK register
read), `WidthDeltaGripperVerifier` (did the jaws collapse to the physical closed width?),
`VisionTargetDisplacementVerifier` (is the target still where it was?), and
`CompositeGraspVerifier`.

The width verifier is why `closed_width_mm` is a config key of its own rather than being read from
`min_width_mm`. "Did the jaws collapse on nothing?" is a question about the mechanism, and
`min_width_mm` is a policy floor. With the shipped defaults, reading the floor of 5.0 mm instead of
the physical 0.0 mm puts the empty-jaw threshold at 7 mm, where a genuinely held 6 mm part reports
as an empty grasp.

A jawless gripper is refused by name, before any width arithmetic. `from_robot_config` answers
four impossible gripper configurations with a working `NullGripper`, and that object takes
`set_width_mm(5.0)` and answers `get_width_mm() -> 85.0`, the configured maximum, forever. 85 mm
clears every threshold here, so the width verifier used to read it as jaws holding 85 mm of
something and the pick reported `SUCCEEDED` with nothing on the flange. `WidthDeltaGripperVerifier`
reads `gripper.substitution`, the record the build path already attaches and the same one the
operator console refuses to connect on, and returns `FAILED` with reason `no_end_effector_built`,
carrying the `SubstitutionReason` in its telemetry. `FAILED` and not `INCONCLUSIVE` on purpose: an
empty flange is knowledge, not missing evidence, so `fail_closed: false` cannot turn it back into a
success.

`gripper.vendor: none` is refused too, decided 2026-09-09. A cell configured with no end-effector is
the same jawless object carrying no substitution record, and it is caught by the object's
`holds_nothing` flag with its own reason, `no_end_effector_configured`. There are no jaws, so nothing
was held, whether or not anybody wanted a gripper. The reasons stay two strings because the repairs
differ: one operator fixes the arm/gripper pairing, the other fits a gripper or turns verification
off.

### Where to look next

`ScoringViewpointPlanner` generates a small fixed candidate set around the current TCP: four lateral
cardinals, two diagonals, and one vertical lift when `vertical_offset_mm` is positive. It keeps a
typed history of viewpoints already observed, so it prefers diverse views and downweights ones near
a previously poor viewpoint. Each candidate is scored against a `ViewScoringPolicy` combining
history diversity with optional occlusion, approach-clearance and reachability weights, then
filtered through a `ViewpointSafetyCheck` Protocol, so an operator can wire a `SafetyPreflight`-aware
acceptance without this module importing the safety layer. It returns `None` once the budget is
exhausted or every candidate is rejected: a bounded search that admits when it is done.

The older `LateralOffsetViewpointPlanner` in [`loop/pick_loop.py`](../loop/README.md) is a
deterministic stub. Four cardinal offsets in a fixed order, never considering the scene. It is a
usable fallback but it cannot say why a viewpoint was chosen.

## Traps

- Refinement and verification are both switched off in a stock config, and switching the config key
  on is not enough on its own: the active mode profile has to permit them. A service asked for the
  `closed_loop` mode without a wired refiner and verification policy refuses the pick with a
  `MODE_NOT_AVAILABLE` outcome rather than degrading to an open-loop attempt.
- `INCONCLUSIVE` is not a failure of the grasp; it is a failure to observe. Treat it as a signal
  about the cell's sensing, not about the pick.
- The refiner needs the same `camera_to_base` the initial computation used, or the refined candidate
  comes back in a different frame from the one it is compared against.

## See also

- [`grasping/`](../README.md) for the tier and which modes wire these layers on
- [`motion/`](../motion/README.md) for the execution policy whose `EXECUTED` starts verification
- [`recovery/`](../recovery/README.md) for what happens when verification says `FAILED`
- [`multiview/`](../multiview/README.md) for the other answer to "look again": more cameras rather
  than more time
- [`docs/grasping-config-reference.md`](../../../../docs/grasping-config-reference.md) for the
  `closed_loop` and `verification` blocks and the modes they can fire in
