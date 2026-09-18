# Execution

The one orchestration layer above the raw drivers and below any caller: calibration capture, pose
provisioning, the vendor-neutral IK seam, taking a cell up and down, and the grasp services that
run one attempt or a campaign of them.

## What it guarantees

Everything here drives a robot through the `RobotArm` and `Gripper` Protocols in
[`robot/core`](../core/README.md), so one body of code drives a UR arm, a KUKA arm, the Isaac
simulator backend or the dummy driver. No vendor SDK is imported at module top level; importing
this package loads none of them, because its 28 public names resolve lazily on first attribute
access.

Order is enforced rather than described. Preflight before build, because a blocking configuration
item is decidable at a desk. Attestation before motion, because an arm that gates nothing must say
so while there is still time to stop. The cross-process lock before connect, because a UR
controller accepts one control script. The arm before the gripper, because a vacuum cup's connect
drives digital I/O immediately, and the exact reverse on the way down, because a teardown that
skips the gripper leaves a vacuum line asserted. Asking for a step before the step it depends on
raises a typed `CellNotBuilt` rather than an `AttributeError` three frames down.

Units are millimetres, rotations are XYZW quaternions, and every pose carries its frame. The IK
seam refuses a query that is not in `Frame.BASE`.

## Contents

| File or subpackage | Role |
| --- | --- |
| `__init__.py` | The lazy re-export surface. Exactly 28 top-level names, listed below. |
| `cell.py` | `Cell`: a cell as one noun, with four steps in the order that makes them safe. `preflight()` needs no hardware, `build()` is idempotent, `safety()` needs a build but commands nothing, `connected()` is the only step that touches a cell. Narration is the caller's: the steps are separate methods. |
| `pick_run.py` | `PickRun`, `PickRunReport`, `PassRule`, `Recording`, `PickAttempt`, `PickOutcome`: N picks under one connect, one verdict over them, one frozen report. |
| `lifecycle.py` | `connect_cell` / `disconnect_cell` / `ConnectedCell` / `TeardownReport` / `NoRealGripper`. Bringing a cell up is a transaction: a gripper that refuses rolls the arm back, and a cell whose end-effector could not be built is refused before the arm is commanded. Teardown reports, never raises, and is never silent. |
| `calibration.py` | `CalibrationRoutine`, `CalibrationResult`, `MarkerPoseProvider`: move, settle, read FK, capture the marker, add the sample, solve `AX=XB`, for eye-to-hand and eye-in-hand alike. One pluggable perception seam (`marker_source`); entry points `run_from_json`, `run_auto`, `run_with_poses`. Every move of a sweep runs inside a camera-world decline for the routine's arm, one reason per mounting unless the caller passes `camera_world=`, and `CalibrationResult.camera_worlds` carries one stamp per commanded move. |
| `pose_provider.py` | `PoseProvider`: load or generate workspace-validated and diversity-validated TCP target poses. |
| `ik_service.py` | `RobotArmIKService` (the only place grasping reaches a controller for reachability), `CachedIKService` (an LRU quantiser), `URAnalyticIKService` (optional offline analytic IK). |
| `runtime_pick.py` | `RuntimePickService`, `PickSessionReport`, `PickTimings`. `from_robot_config` takes its arm and gripper from `robot_parts.py`. |
| `robot_parts.py` | `resolve_arm` and `build_gripper`: the arm and the gripper a `RobotConfig` describes, with the readiness gate and every `NullGripper` substitution, built without a pick service. Plus `build_sim_driver_config`, the single Pydantic-to-driver `SimRobotConfig` conversion that `resolve_arm` and the simulator runners share. It imports none of the grasping stack. |
| `robot.py` | `Robot`: the arm and the gripper as one noun, with no pick service. `from_config(robot_config, gripper=UNSET)` builds both through `robot_parts.py` (`gripper=None` builds the arm alone), `from_parts(arm=, gripper=, lock_key=UNSET)` wraps handles already built and refuses an arm with a controller of its own when no lock key derives, `connected()` takes the `CellLock` and runs the enter and exit `ConnectedCell` runs (`lifecycle.py`), `cameras=UNSET` on both factories takes open camera owners (primary first), builds the live planner world from them under the tree's planning block (`robot_config`, else `arm.config`), hands it to the arm's `set_live_planner_world` and keeps it as `camera_world`; the robot never opens or releases a camera, and `camera_world_line()` says what the arm holds. The wrist cameras among those cameras are resolved first (`wrist_bodies.py`) and handed to the arm, and `wrist_body_line()` names them. `safety()` reads the built arm, and `without_camera_world(reason)` declines the camera world for every motion of the robot's arm inside a `with` block, bound to that arm. `grasp(width_mm)`, `release()` and `is_holding()` are the hand verbs, and `pick` and `place` put them at the end of a straight line; all five delegate to `handling.py`. A sibling of `Cell`, which still builds through the pick service. The real-cell calibration command builds its arm this way, with `gripper=None`. |
| `handling.py` | The hand verbs `Robot` delegates to, and `pick` and `place` built on them (`HandlingReport`, `HandlingOutcome`). `grasp(robot, width_mm)` commands the width with no force, reads back the width, whether it was measured and the gripper's `HoldEvidence`, and hands a held or unmeasured hold to the arm's `CarriesPayload` model; a measured empty close attaches nothing. `release(robot)` opens to `max_width_mm` and detaches unless the gripper still measures a part (`RELEASE_NOT_CONFIRMED`). `is_holding(robot)` commands nothing. A gripper that raises is stopped once and reported as `GRIPPER_FAULT`; no gripper, a gripper that holds nothing or a closed link is `REFUSED` with nothing commanded. Each returns a frozen `HandReport` (`render()`, `to_dict()`). It imports nothing above `robot.core`. |
| `wrist_bodies.py` | `WristBodies.from_config(robot_cfg, camera_cfg, data_dir=None)` and `.from_owners(robot_cfg, owners)`: the wrist cameras a cell's arm carries, as `body_link.WristBody` values, resolved once for the real cell build, `Robot`, `PlannerStart`, the calibration command and the desk row. Only a cell whose planner is cuRobo or whose guard reads hand geometry resolves any. `WristBodyRequired` refuses an enabled eye_in_hand rig without a body, a body nothing places, a calibration without its flange to TCP record or with one the cell no longer holds, and a camera the repository's registry does not stand for. `hand_to(arm)` hands them over through `set_wrist_bodies`. |
| `planner_start.py` | `PlannerStart.from_robot_config(robot_config, data_dir=None, camera=UNSET).run()` builds a real UR arm alone and starts its cuRobo planner through the driver's own `start_planner()`, the path the first planned move takes, then stops it. The frozen `PlannerStartReport` says whether it started, what the sidecar loaded and which evidence file admits it, or the refusal verbatim (`exit_code` 0 or 1). No camera opened, no controller. Handed the camera section, it resolves the wrist cameras first and starts the planner with them, and the report's `wrist_bodies` line names them. `Cell.start_planner()` (with its tree's camera section) and `real_cell --start-planner` call it. |
| `cell_lock.py` | `CellLock`, `CellBusy`, `cell_lock_key`: one owner per cell, keyed on the controller address. The command-line runner and the operator console take the same lock. |
| `calibration_watchdog.py` | Pure-function drift and out-of-distribution evaluators. Stateless: the rolling history lives on the service. |
| [`autonomous_grasp/`](autonomous_grasp/README.md) | `AutonomousGraspService`, the composition root: mode selection, the fail-closed decision gate, uncertainty fusion, the drift watchdog, latency telemetry, the reinforcement-learning shadow router, the bounded recovery loop, and the two cell builders. |
| [`real_cell/`](real_cell/README.md) | `python -m src.robot.execution.real_cell`: the command-line surface over `Cell` and `PickRun`, plus the configuration preflight and the per-camera hand-eye calibration a real cell needs. |

The 28 lazy names: `Robot`, `HandReport`, `HandOutcome`, `PayloadState`, `HandlingReport`,
`HandlingOutcome`, `PassRule`, `PickAttempt`, `PickOutcome`, `PickRun`, `PickRunReport`,
`Recording`, `PoseProvider`, `CalibrationRoutine`, `CalibrationResult`, `MarkerPoseProvider`,
`RobotArmIKService`, `CachedIKService`, `URAnalyticIKService`, `PickSessionReport`, `PickTimings`,
`RuntimePickService`, `AutonomousGraspService`, `AutonomousGraspReport`, `AutonomousGraspOutcome`,
`GraspMode`, `GraspBehaviorProfile`, `resolve_grasp_mode`.

`Cell`, `ConnectedCell`, `CellLock` and the decision types are imported from their own modules
rather than from the package root. `EffectiveGraspingConfig` and its eight nested per-phase
sub-configs come from the `autonomous_grasp` subpackage.

## Usage

```python
from src.config import load_robot_config
from src.robot.execution.cell import Cell
from src.robot.execution.pick_run import PickRun, Recording

cell = Cell.from_robot_config(load_robot_config(), prompt="a red cube")
print(cell.preflight().render())        # decidable at a desk, no hardware
cell.build()                            # drivers, perception, the grasp stack
print(cell.safety().render())           # what this arm will refuse, before it moves

report = PickRun.from_cell(cell, runs=10, recording=Recording.off()).execute()
print(report.render())
raise SystemExit(report.exit_code)
```

`PickRun` owns the connect, takes the lock, and always takes the cell down again, which is the
only place a gripper that did not release is reported. `PickRun.from_service` is the other factory,
for a caller that has already connected and owns the teardown itself.

`GraspMode` is `easy`, `auto`, `dense_clutter`, `closed_loop` or `dense_autonomous`;
`resolve_grasp_mode` also accepts the aliases `single`, `dense` and `autonomous`.

An arm and its gripper with no pick service, connected in the order a cell connects:

```python
from src.config import load_robot_section
from src.robot.execution import Robot

robot = Robot.from_config(load_robot_section())   # gripper=None builds the arm alone
print(robot.safety().render())
with robot.connected() as live:                   # lock, arm, then gripper
    print(robot.grasp(40.0).render())             # close, read the hold, model the part
    print(robot.release().render())
    picked = robot.pick(grasp_pose, 40.0)         # standoff, a line in, grasp, a line out
    print(picked.render())
```

`pick(pose, width_mm, *, standoff_mm=80.0, squeeze_mm=1.0, pre_open_mm=UNSET, camera_world=UNSET,
keep_out=UNSET)` and `place(pose, *, standoff_mm=80.0, camera_world=UNSET)` refuse before any command a
robot with no usable gripper, a closed link, a pose not in BASE, a camera world the arm would refuse
(`ReadsCameraWorld`) and an arm that keeps no line or does not say (`KeepsLines`). Then they drive a
planned standoff, a line in, the hand verb and a line out, each motion after the arm's own steady gate
where its tree asks for one, and return a frozen `HandlingReport`.

## Traps

- **The default pick is open-loop.** Every advanced `robot.grasping` block ships `enabled: false`,
  and `robot.rl.mode` ships `hybrid_ml`, which builds no reinforcement-learning component. The
  decision gate, closed-loop refine and verify, fusion and commit, ordering and the learned success
  model are all built and switched off. A report's `layers` line answers `(none)` on the shipped
  tree, and that is the expected answer.
- **A pass rule is an argument, not a constant.** `PassRule` defaults to unanimity: every pick must
  succeed. The simulator gate configures 0.8 instead and additionally refuses to accept the
  service's own `SUCCEEDED` as evidence, requiring a physics-measured lift. A cell that built a
  `NullGripper` reports `SUCCEEDED` on every run, which is why the rule is stated per campaign and
  printed with the verdict.
- **`from_robot_config` substitutes a `NullGripper` rather than crashing.** Five configuration
  combinations cannot produce a real end-effector, one per `SubstitutionReason` member. Each logs a
  warning and records the typed reason on the gripper object. Since 2026-09-09 such a cell cannot be
  connected at all: `connect_cell` raises `NoRealGripper` before the arm is commanded, so
  `Cell.connected()` and the operator console give one answer. Until then it connected and reported
  every pick a success while holding nothing. A cell that declares `gripper.vendor: none` carries no
  substitution record and still connects.
- **Two ways to build the service are not equivalent.** `from_robot_config` reads the whole
  configuration tree and populates `effective_config`; `from_components`, which a caller uses when
  it holds live device handles config cannot describe, leaves it `None` and thereby silences the
  config-driven overlays. See [autonomous_grasp/](autonomous_grasp/README.md).
- **`URAnalyticIKService` needs `ur_ikfast`,** which is not in `requirements.txt` because it is not
  on PyPI and its build is platform-sensitive. Construction raises with the install hint if it is
  absent. The normal reachability path is `RobotArmIKService`, which queries the live controller,
  whose calibrated IK is more faithful to the physical arm than any ideal-model solver.
- **`PickTimings` has no `perception_s`.** Perception time is inside `orchestrator_s`, because the
  orchestrator owns perception acquisition. `PickSessionReport.candidate_count` is the number of
  candidates evaluated on the last perception frame the orchestrator saw, not a total over the
  attempt.
- **Empty and unvalidated driver slots.** The KUKA driver reachable through `from_robot_config` has
  not been validated against real hardware. `drivers/franka` and `drivers/ros2` hold nothing but a
  package marker.

## Status

`RuntimePickService` and `AutonomousGraspService` run end to end through the simulator driver in
mock mode and on an Isaac workstation. `CalibrationRoutine` solves both the eye-to-hand transform
(`T_cam_to_base`) and the eye-in-hand transform (`T_cam_to_tool`) against a real marker source.

`from_robot_config` has a live caller in [`real_cell`](real_cell/README.md) and in
`scripts/examples/api/01_first_cell/one_pick_end_to_end.py`, and both rehearse the whole path on a dummy arm under the
`console_dummy` profile, whose gripper a dummy arm can carry. A rehearsal of the base tree is
refused at the connect instead, because the vendor swap substitutes its Robotiq. Beyond that
rehearsal, nothing in this layer has executed against a physical controller.

The operator console in [`api/`](../../../api/README.md) consumes this layer over HTTP. The
dependency runs one way only: nothing under `src/` may import a web framework or the console.

## See also

- [Workaholic-Willy](../../../README.md), the repository overview
- [autonomous_grasp/](autonomous_grasp/README.md), modes, the decision layer, the cell builders
- [real_cell/](real_cell/README.md), the config-driven pick on a physical arm
- [robot/core](../core/README.md), the Protocols this layer drives
- [robot/safety](../safety/README.md), the fail-closed guards every motion passes through
- [willy_sim](../../willy_sim/README.md), the Isaac runners that consume this layer
- `scripts/examples/api/01_first_cell/one_pick_end_to_end.py`, one grasp end to end through `Cell` and `PickRun`
