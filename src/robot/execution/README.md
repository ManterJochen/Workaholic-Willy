# Execution

The one orchestration layer above the raw drivers and below any caller: calibration capture, pose
provisioning, the vendor-neutral IK seam, taking a cell up and down, and the grasp services that
run one attempt or a campaign of them.

## What it guarantees

Everything here drives a robot through the `RobotArm` and `Gripper` Protocols in
[`robot/core`](../core/README.md), so one body of code drives a UR arm, a KUKA arm, the Isaac
simulator backend or the dummy driver. No vendor SDK is imported at module top level; importing
this package loads none of them, because its 22 public names resolve lazily on first attribute
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
| `__init__.py` | The lazy re-export surface. Exactly 22 top-level names, listed below. |
| `cell.py` | `Cell`: a cell as one noun, with four steps in the order that makes them safe. `preflight()` needs no hardware, `build()` is idempotent, `safety()` needs a build but commands nothing, `connected()` is the only step that touches a cell. Narration is the caller's: the steps are separate methods. |
| `pick_run.py` | `PickRun`, `PickRunReport`, `PassRule`, `Recording`, `PickAttempt`, `PickOutcome`: N picks under one connect, one verdict over them, one frozen report. |
| `lifecycle.py` | `connect_cell` / `disconnect_cell` / `ConnectedCell` / `TeardownReport`. Bringing a cell up is a transaction: a gripper that refuses rolls the arm back. Teardown reports, never raises, and is never silent. |
| `calibration.py` | `CalibrationRoutine`, `CalibrationResult`, `MarkerPoseProvider`: move, settle, read FK, capture the marker, add the sample, solve `AX=XB`, for eye-to-hand and eye-in-hand alike. One pluggable perception seam (`marker_source`); entry points `run_from_json`, `run_auto`, `run_with_poses`. |
| `pose_provider.py` | `PoseProvider`: load or generate workspace-validated and diversity-validated TCP target poses. |
| `ik_service.py` | `RobotArmIKService` (the only place grasping reaches a controller for reachability), `CachedIKService` (an LRU quantiser), `URAnalyticIKService` (optional offline analytic IK). |
| `runtime_pick.py` | `RuntimePickService`, `PickSessionReport`, `PickTimings`, and `build_sim_driver_config`, the single Pydantic-to-driver `SimRobotConfig` conversion that `from_robot_config` and the simulator runners share. |
| `cell_lock.py` | `CellLock`, `CellBusy`, `cell_lock_key`: one owner per cell, keyed on the controller address. The command-line runner and the operator console take the same lock. |
| `calibration_watchdog.py` | Pure-function drift and out-of-distribution evaluators. Stateless: the rolling history lives on the service. |
| [`autonomous_grasp/`](autonomous_grasp/README.md) | `AutonomousGraspService`, the composition root: mode selection, the fail-closed decision gate, uncertainty fusion, the drift watchdog, latency telemetry, the reinforcement-learning shadow router, the bounded recovery loop, and the two cell builders. |
| [`real_cell/`](real_cell/README.md) | `python -m src.robot.execution.real_cell`: the command-line surface over `Cell` and `PickRun`, plus the configuration preflight and the per-camera hand-eye calibration a real cell needs. |

The 22 lazy names: `PassRule`, `PickAttempt`, `PickOutcome`, `PickRun`, `PickRunReport`,
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
  warning, records the typed reason on the gripper object, and lets the cell connect. The cell then
  reports every pick a success while holding nothing.
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
`scripts/examples/cell/03_first_pick.py`, and both rehearse the whole path on a dummy arm. Beyond that
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
- `scripts/examples/cell/03_first_pick.py`, one grasp end to end through `Cell` and `PickRun`
