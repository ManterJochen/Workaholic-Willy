# `drivers/planned`: the planned arm

The judging and planning half every planned arm shares, cut out of the UR driver in P0e E8 without changing what a
method does. A vendor driver is `PlannedArm` plus its transport (`robot.core.transport`); `URRobotArm` is the first.

| File | Holds |
|---|---|
| `arm.py` | `PlannedArm`: the safety preflight on every commanded pose and path, the cuRobo route and its camera world, the tool-frame composition (TCP to the controller's frame and back), the payload it pushes at connect, the checked line, the judged joint move, the move home, the halt latch, and what Willy judges and holds ahead of a motion (the next leg, the line down, the grasp, the branch the arm keeps). It reaches the controller only through `self._transport`. |
| `goal_set.py` | `goal_set_for`, `GoalSet`, `Branch`: where a planned move's goal configuration comes from, the closed form the arm's description states and the branch each configuration holds. The UR chain is the only entry (`safety/_ur_ik.py`); an arm whose description states none is refused before anything is asked, and cuRobo never chooses a goal on any arm (the owner, 2026-09-25). |
| `curobo_motion.py` | `CuroboPlanner`: a collision-free plan to a joint goal through the cuRobo sidecar, from where the arm stands or an explicit `start_ur`, sent one waypoint at a time with the transport's `run_trajectory`, the halt read before each; `judge_joint_path`, every refused sample of a path with its terms and pairs; fail-closed, no blind-IK fallback. The name is the UR's until P0e E11 renames it. |
| `capabilities.py` | The optional capabilities a planned arm offers over its transport, one mixin each, which a vendor class lists only where its transport offers the matching transport Protocol, so an arm with no I/O is refused a vacuum or an I/O jaw by type (P0g). `PlannedDigitalIO`: `SupportsDigitalIO` and the end of a pulse a halt cut short over `ControllerIO`, a write the controller did not take raising `RobotConnectionError` (io1). Since slice 7 `PlannedForceTorque` over `ReadsForceTorque`, `PlannedRobotStatus` (the status and its quick read, each with the halt latch beside the controller's answer) over `ReportsControllerStatus`, `PlannedFaultAcknowledge` over `ReleasesProtectiveStop` (`SupportsRobotStatus` is these two together) and `PlannedControllerProvenance`, never raising, over `ReadsControllerProvenance`. `PlannedWatching`: `SupportsWatching` over `WatchesController` (P0i: `watch()` opens the link the fault door uses, on which the transport is not connected, so every motion verb refuses with nothing sent). Since port S4 `PlannedFreedrive`: `SupportsFreedrive` over `ControllerFreedrive`, `freedrive()` refused while not connected, in a session, halted (in the halt's words) or not operational, and `controller_payload()`. A mixin never subclasses the Protocol it implements. |
| `freedrive.py` | `PlannedFreedriveSession` (the UR's own session until port S4, now over the transport): `free()` arms the controller's watchdog and only then begins hand guiding; `sample()` kicks the watchdog while the arm is free, reads joints, speeds and the arm's TCP, and raises `ArmFault` for a stop and `RobotConnectionError` for a lost read or an ended program; `hold()`, and with it the with-block's end on any exception, ends hand guiding, restarts the control program and waits until the joints stand still. `raise_unless_operational` is the check before a session opens, a halted arm said in the halt's words. The order and the URSim measurements are in the module docstring. |
| `tool_frame.py` | `derive_active_tool_frame`, `described_chain`: the tool frame a controller applies, derived as `inv(the description's chain flange) @ the controller's TCP now`, which connect checks against the declared frame. The chain is the UR DH table bit for bit (`tests/robot/safety/test_serial_chain_is_the_ur_chain.py`); moved from `drivers/ur` in P0g slice 15. |
| `planner_frame.py` | `PlannerFrameClient`, `planner_pose`: every pose handed to the planner is turned from the controller's base into the planner's, half a turn about Z, and back. Applied where the arm's description says `base.controller_yaw_deg: 180`, every UR, and not where it says 0: the driver reads the key since P0f F9c. |

**What the subclass answers.** `PlannedArm` is built by keyword from what a vendor driver built from its tree
(`config`, `logger`, `capabilities`, `joint_names`, `controller_yaw_deg`, `transport`, `guard`, `preflight`,
`home_joints`, optionally `curobo_client_factory` and `tool`), with no positional form (P0g slice 4); `URRobotArm`
builds those in the order it always did and hands them over. `tool` is the `MountedTool` on the flange
(`safety/planning/mounted_tool.py`, P0i i5): the hand the guard and the planner model, the tool frame and the payload,
read from the tree by `MountedTool.from_config`, from which the UR builds its guard's hand too; left unset it is the
tree's, with the guard's hand. The arm reads its tool frame and its payload from `mounted_tool` and nothing else.
`mount_tool(tool)` is the one path a tool takes after the build: refused while connected (`ToolChangeRefused`, "a tool
change is not built yet"). On an arm not connected every reader of the tool follows it or the mount is refused with
nothing changed: the self collision guard models the new hand, refused as the build refuses it (the capsule proxy
takes its fingers from the new hand's registry file too, since 2026-10-17), and lets go of the opening the old hand was
told; the payload guard judges the new payload, refused where the schema would not declare it; a `willy` frame is held
to every wrist camera's flange to TCP record. Then the planner is stopped, a tool
frame the controller reported for the old tool is forgotten, and the next connect pushes the new payload. Nothing
calls it yet but its tests. `_VENDOR_HOOKS` names the only methods a subclass may answer differently, and each has a
neutral answer here: `is_inside_workspace` asks the box; `_boxed` judges a base-frame pose as it is and refuses
another frame (the UR answers through its own pose type, so its verdicts stay bit-identical); `_home_refusal` is the
home's destination guards and then the box on the grasp centre the controller reports, as the typed refusal
`move_to_home` returns, with the controller's FK read through `_home_fk` (the UR reads it again on a fresh control
program where the old one ended); `_controller_chain` answers no rows of the controller's own, and
`_controller_goal_m` the goal a line solved on them reads, a 4x4 in metres (the UR's through its axis-angle). Since
P0g slice 15 `connect` is shared, not a hook: before the transport opens it refuses a contradictory payload, a payload
guard over a transport with no payload register, an undeclared tool frame and a transport that reports no controller
TCP; it pushes the payload through `TakesPayload` and rolls the connection back on any failure, then checks the
controller and its tool frame, rolling back again. What a vendor answers of it are four hooks, each neutral here:
`_address_refusal` (the UR refuses a tree with no `robot.ur.ip`), `_verify_controller_model` (the UR compares size
classes with the dashboard's answer), `_refuse_if_another_model_fits_better` (the UR asks its series twin) and
`_vouched_flange_chain` (the UR reads its controller's calibrated rows and keeps them where they put the flange where
the controller says it stands, for `controller_flanges_mm`). `disconnect` retires the planner, forgets the tool frame
and the rows connect kept, and closes the transport.

**What a failure says it sent.** A failure these methods build says in a field whether anything had reached the
controller, `MotionResult.sent` (P0g R2): `NOTHING` before any command, `SENT` once one reached the controller whatever
the status (a range refusal at a later waypoint is `INVALID_TARGET` and `SENT`), and `UNSTATED` where the transport did
not classify its refusal and where the status or the sentence already says it, a guard's refusal of a goal before the
send or a drive that may have sent its `moveJ`. Every refusal of the cuRobo route and of the judged joint move says
`NOTHING`, a gate's or the planner's included, since neither sends a command (port S2). The calibration sweep reads the
field before the wording.

**What is still UR-shaped here, on purpose, until the slices after E8.** The methods read `self.config` (a
`RobotConfig`) where typed constructor values will stand; the transport is typed `PlannedArmTransport`, every verb the
UR transport has; the planner frame is the UR half turn; the branch and the nearest goal come from the goal set of
the arm's description (`goal_set.py`), whose one entry is the UR's closed form (`robot.safety._ur_ik`), and the end of a
plan is read on the UR's DH chain.

**Proof of the move.** `tests/robot/drivers/ur/test_the_planning_half_reaches_the_controller_through_its_transport.py` holds this
package free of vendor imports and `URRobotArm`'s own body to the UR side; the P0e command tape, every command the
arm sends for 56 verbs, is byte-identical before and after.
