# UR driver (RTDE)

The Universal Robots `RobotArm` implementation. It drives a UR controller over RTDE and speaks this
stack's vendor-neutral pose contract.

It talks to the controller through `ur_rtde` (`rtde_control`, `rtde_receive`, `rtde_io` and
`dashboard_client`), pinned at `ur_rtde==1.6.5` in `requirements.txt`. It satisfies the
vendor-neutral `RobotArm` Protocol, owns all UR-native unit conversion (metres to millimetres,
axis-angle to XYZW quaternion) inside `URPose` and `pose_adapter`, and exposes nothing UR-specific
upward. Build it through the registry: `create_arm(RobotVendor.UR, config=cfg.robot)`.

## Contents

| File | Role |
| --- | --- |
| `__init__.py` | Public exports. `URConnection`, `URPose` and the adapters are eager; `URRobotArm`, `UR_CAPABILITIES` and `MotionController` resolve through `__getattr__`, so the package imports without the SDK present. |
| `arm.py` | `URRobotArm`, `UR_CAPABILITIES` and `ur_capabilities(model)`. The Protocol implementation, the typed `move()` wired to `SafetyPreflight`, the connect-time refusals, the optional-capability mixins, and the `ik` against `curobo` planner switch. |
| `connection.py` | `URConnection`, the RTDE boundary and the only `ur_rtde` import site. |
| `motion.py` | `MotionController`: workspace-validated, velocity and acceleration clamped, singularity-aware point-to-point moves. |
| `curobo_motion.py` | `CuroboUrPlanner`: plan a global collision-free joint trajectory through `safety.planning`, then execute it waypoint by waypoint over `moveJ`. Fail-closed, with no blind-IK fallback. |
| `tool_frame.py` | Derive the controller's active tool frame from FK and the bundled DH table, and compare it with what config declares. `derive_active_tool_frame`, `tool_frame_matrix`, `compare_tool_frames`, `ToolFrameMismatch`. |
| `pose.py` | The `URPose` value object, millimetres and axis-angle radians internally, and its conversions. |
| `pose_adapter.py` | `pose_to_urpose` and `urpose_to_pose`, the only bridge between `URPose` and `Pose`. |
| `io_bench.py` | The digital-I/O primitives: `read_snapshot`, `watch_input`, `set_output`, `pulse_output`, `measure_transition`. Nothing here commands motion. |
| `bench.py` | `Bench`, the library twin of the I/O CLI: one `run(action)` verb over `Read`, `Watch`, `Set`, `Pulse` and `Measure`, returning a frozen `BenchReading`. |
| `__main__.py` | The bench CLI. It never moves the arm and gates every write behind `--yes`. |

## Public surface

`URRobotArm(config, home_joints=None)` implements the Protocol:

- `connect()` and `disconnect()`, also usable as a context manager. `connect()` pushes
  `set_payload` with mass and centre of gravity where `config.safety.payload.enforce` is true, and
  rolls the connection back if that push fails.
- `move(pose, *, linear, vel, acc, register) -> MotionResult`, the typed path. It pre-resolves IK,
  runs `SafetyPreflight`, and returns a precise `MotionStatus` on rejection. Where
  `robot.ur.motion_planner` is `curobo` it routes through `CuroboUrPlanner` instead of controller
  IK.
- `move_to(...) -> bool`, the boolean path through `MotionController`, plus `move_linear` and
  `move_joint` (which raise `RobotMotionRejected` where the preflight denies the move),
  `move_to_joints`, `move_home`, `stop`, `wait_until_steady`, and the asynchronous `amove_to` and
  `amove_home`.
- `fk(joints) -> Pose` and `ik(pose, *, seed) -> JointPositions`, both controller-native and both
  requiring an open connection; `get_tcp_pose`, `get_joint_positions`, `is_inside_workspace`,
  `capabilities`.
- `attach_payload(grip_width_mm)` and `detach_payload()` tell the cuRobo planner that the gripper is
  carrying a part. `attach_payload` returns `False` where nothing was attached, which includes the
  ordinary case of a cell that plans with `ik`.
- Optional capabilities, feature-checked with `isinstance(arm, SupportsForceTorque)`:
  `SupportsDigitalIO` for digital and analog I/O, `SupportsForceTorque` for `get_tcp_wrench` and
  `get_joint_torques`, and `SupportsRobotStatus` for `get_robot_status`,
  `recover_from_protective_stop` and an enriched `RobotEmergencyStop`.

Three value types complete the surface. `URConnection(ip, vel=1.0, acc=0.5, frequency=0.0)` is the
thin RTDE wrapper covering move, read, I/O, payload and safety status; its `rtde_io` and
`dashboard_client` paths are best-effort, so a slim build without them keeps move and read working.
`URPose(x, y, z, rx, ry, rz, label="")` is a frozen dataclass in millimetres and axis-angle radians,
with `to_ur_list` and `from_ur_list` doing the metre conversion at the boundary. And
`ur_capabilities(model)` returns the capability record for one model; `UR_CAPABILITIES` is its
`ur5e` instance, and everything except the model string is shared across the e-series.

## Usage

```python
import numpy as np
from src.robot.drivers import create_arm, RobotVendor
from src.geometry import Pose, Frame

arm = create_arm(RobotVendor.UR, config=cfg.robot)   # cfg.robot.vendor == "ur"

with arm:                                            # connect() and disconnect()
    target = Pose(
        position_mm=np.array([400.0, 0.0, 300.0]),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        frame=Frame.BASE,
    )
    result = arm.move(target)
    if not result.ok:
        print(result.status)                         # e.g. MotionStatus.WORKSPACE_REJECTED
```

Without `ur_rtde` installed, `connect()` raises with a message naming the install. The rest of the
package, meaning the poses, the adapters and capability introspection, imports and runs fine.

## What `connect()` refuses, and why

Four checks run before the arm is usable. Each fails closed, and three of them fire on the shipped
config, which is deliberate: a cell that has not been measured should not move.

1. **A contradictory payload.** `enforce: true` with `mass_kg: 0.0`, which is the shipped default,
   would push `setPayload(0.0)` and overwrite the controller's payload for a mounted tool.
   `mass_kg > 0` with the default `cog_mm: [0, 0, 0]` would declare that tool a point mass at the
   flange face. Both corrupt the controller's protective-stop model and its gravity compensation,
   which is what `get_tcp_wrench` reads. Weigh the whole assembly, meaning the coupling plate, every
   cable and hose riding on the wrist, and any workpiece the arm carries. A bare flange is expressed
   as `enforce: false`, which never touches the controller payload.
2. **An undeclared tool frame.** `gripper.tool_frame.source` ships as `undeclared`, so nobody has
   said where the grasp centre sits on the flange and the driver cannot know whether a commanded
   pose means the TCP or the flange. Set `offset_mm` and `rotation_quat_xyzw` from the bench, then
   choose `source: willy`, where this driver composes the transform and the controller runs a bare
   flange, or `source: polyscope`, where an operator set the TCP on the pendant and the driver only
   verifies it.
3. **The wrong robot.** The controller is asked which model it is through the dashboard, and a
   parsed, positive disagreement with `robot.ur.model` refuses the connection. The comparison is on
   the size class alone, because a URSim e-Series container reports `UR3` for a UR3e. It fails open
   on absent evidence: a dashboard that will not answer is logged, not refused, because a false
   refusal is a cell that will not start.
4. **The wrong tool frame.** The controller's active tool register is derived, never read, as
   `inv(base-to-flange from the bundled DH table) @ getForwardKinematics(q)`. Both halves are already
   on the safety-critical path, so this needs no SDK symbol that `move()` does not already require.
   The two `source` modes predict different answers, near identity for `willy` and the declared
   transform for `polyscope`, so a pass also proves which mode the cell is in.

The model check runs before the tool-frame check because where both would fire, the model one names
the cause and the tool-frame one only names a symptom.

## Traps

**Forward kinematics of an arbitrary configuration is corrupted by motion.**
`getForwardKinematics(q)` shares the controller's float registers with `moveJ` and `moveL`, so after
any commanded move it returns a pose wrong by hundreds of millimetres until those registers are
rewritten. Two consequences follow. `URConnection.fk()` passes the controller's own TCP offset
explicitly so it never reads a register something else wrote, and `URConnection.fk_current()`, which
uses the no-argument form, is the immune way to ask for FK of the joints the arm is at right now.
Use `fk_current()` wherever the current configuration is what is meant. `fk(q)` for an arbitrary `q`,
which the singularity guard and `MotionController` need, has no immune equivalent and remains open
against a real cell.

**External control requires Remote mode, and Remote mode costs the pendant.** A UR in Local control
mode never executes an externally-sent program, and the resulting transport error names nothing.
`URConnection.connect()` builds `RTDEControlInterface` first so the failure happens at
`arm.connect()`, and diagnoses the cause through a throwaway dashboard connection: Local mode, or a
protective stop nobody cleared. It claims a cause only where the dashboard proves it, because being
unable to ask is not an answer of no, and `URConnection.is_in_remote_control()` is three-valued for
the same reason. The price of Remote mode is a pendant locked for motion, with no jogging, no
freedrive and no manual program start, because a single source of control is what the standard
wants. A running program also stops the moment another is sent, so two `RTDEControlInterface`
instances kill each other. What a physical emergency stop does in Remote mode is not verified here:
it is a hardware safety circuit and should be mode-independent, but that is inference, so check the
robot manual before the first powered run.

**cuRobo-planned motion never degrades.** `robot.ur.motion_planner` defaults to `curobo`. That path
returns `CONTROLLER_REJECTED` where the planner is unavailable and `TIMEOUT` where no collision-free
plan exists, and it never falls back to blind IK, so a cell with no cuRobo environment does not move
at all. Check that environment with `python -m src.robot.safety.planning --doctor` before
commissioning. The alternative, `ik`, is the controller's calibrated IK and a straight `moveJ` or
`moveL` line, which knows nothing about the cell and will drive through anything in it.

**`robot.ur.model` is a safety key, not a label.** It selects the safety DH chain, the exact-mesh
collision bundle and the cuRobo robot config. Setting it wrong computes every self-collision verdict
against another robot's link lengths. The supported keys are `ur3e`, `ur5e` and `ur10e`. Joint limits
in the built-in table are factory-wide; site-specific limits go in `robot.safety.joint_limits`.

**Force and torque are read-only.** `SupportsForceTorque` exposes the TCP wrench and joint torques,
which are the hand-over signals. There is no force or impedance control, no compliant motion, no
spline or blended paths, and no tool-changer integration.

**`rtde_frequency: 0.0` means "let the controller choose".** That is this stack's sentinel, not
`ur_rtde`'s, which reads `0.0` as literally zero hertz and fails synchronisation. The UR connection
translates it at the driver boundary.

## The bench CLI, which never moves the arm

`python -m src.robot.drivers.ur` exercises the digital I/O a gripper is wired to. There is no path
from it to `arm.move`. Driving an output is still a physical action, so every write is gated behind
`--yes`, and in a non-interactive shell the gate refuses rather than prompts. `--read` and `--watch`
are read-only. One action per invocation: asking for two is refused rather than resolved silently.

```bash
python -m src.robot.drivers.ur --read                          # every pin on the bank, changes nothing
python -m src.robot.drivers.ur --watch 0 --for 15              # trip a sensor by hand and see it
python -m src.robot.drivers.ur --set 4=1 --yes                 # drive one output, then read it back
python -m src.robot.drivers.ur --pulse 4 --for 0.2 --yes       # the double-solenoid shape
python -m src.robot.drivers.ur --measure 4=1 --watch 0 --yes   # the number close_settle_s wants
```

`--port` selects the bank and defaults to `tool`, where a tool-mounted gripper usually sits.
`--measure` drives an output and times how long the watched input takes to answer; run it a few
times and configure the worst reading, because the driver waits that budget out and a typical value
turns into a dropped part on a slow stroke. `Bench.from_robot_config(...)` is the library twin, and
it resolves the bank from `gripper.jaw_io.io_port` or `gripper.vacuum.io_port` rather than
defaulting it. The `io_bench.py` functions underneath have no gate of their own: they energise a
pin the moment they are called.

## Bring-up checklist for a real UR

Ordered so each step de-risks the next. Nothing here moves the arm.

1. **Is the SDK present?** `python -m src.robot.drivers.doctor --require ur`. A miss reports
   `SDK not installed: rtde_control, rtde_receive`; install with `pip install -r requirements.txt`,
   or `requirements-cpu.txt` on a CPU-only box. Both pin `ur_rtde`.
2. **Does the config name the right arm?** `python -m src.config explain robot.ur.model`, and confirm
   it matches the arm in front of you. The driver cross-checks this against the controller at
   connect, but only where the dashboard answers, so the config is still the thing to get right.
3. **Are the payload and the tool frame measured?** One bench step gives both, and `connect()`
   refuses without either. See the four refusals above for what to set.
4. **Is Remote Control on?** External control requires it. Every URSim container start comes up in
   Local; expect the same discipline on a real pendant.
5. **Network and connection.** `python scripts/examples/cell/01_robot_setup.py --live` connects, reads
   the TCP pose back in millimetres and the controller's own robot and safety mode, then
   disconnects. No motion. The same script without `--live` runs the config preflight alone, which
   is the check `python -m src.robot.execution.real_cell --check` runs.

A green run at step 5 does not mean the cell is correct. The preflight reads your YAML and the
connect step reads the controller; neither knows what is bolted to the robot. Payload, tool frame
and workspace are declared values, and these checks confirm that the declaration is coherent and
that the controller accepts it, never that it matches the hardware in front of you.

## Where this driver has and has not run

It has been measured against URSim, the real UR controller software speaking real RTDE, so it is
exercised against a controller rather than against a mock: the dashboard lifecycle from power off
through brake release to running, `moveJ`, the RTDE reads, digital outputs that read back changed,
the tool-frame verification including a deliberate disagreement that must refuse, `setPayload` with
the connection rollback, and a genuine protective stop with the commanded motions each refused.

No physical UR has ever executed a line of it. URSim is controller software: it cannot prove torque,
payload dynamics, a real tool frame, or what a physical emergency stop does, and it validates the
call path rather than any wiring.

The container lifecycle and the probes behind those measurements live in
[`scripts/ursim/`](../../../../scripts/ursim/README.md), which documents the order to run them in.
The profile they run under is
[`config/robot/robot.ursim.yaml`](../../../../config/robot/robot.ursim.yaml), selected as
`WILLY_PROFILE=ursim`; a UR3e container layers `WILLY_PROFILE=ursim,ursim_ur3` on top.

## See also

- [drivers](../README.md), the registry, the factory and the vendor-neutral driver boundary
- [robot/core](../../core/README.md), the `RobotArm` Protocol, `MotionResult` and `MotionStatus`, and
  the optional-capability mixins
- [safety](../../safety/README.md), the `SafetyPreflight` every `move()` is gated through and the
  cuRobo planning the `curobo` path uses
- [sim driver](../sim/README.md), the Isaac driver that carries the validated pick paths
- [`scripts/ursim/`](../../../../scripts/ursim/README.md), the container lifecycle and the probes
