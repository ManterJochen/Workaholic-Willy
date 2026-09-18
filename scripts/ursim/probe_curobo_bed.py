"""Drive URSim through the checked verbs: a real controller, a real planner, no hardware.

`probe_our_driver.py` proved our driver talks to a real controller. This proves the checked half does:
a joint move whose whole line went to the guards and to the cuRobo sidecar, a linear move whose every
sample was solved by the controller's own inverse kinematics, a planned free move, and a deliberate
refusal against a fixture declared across the line, with the joints asserted unchanged afterwards.

It also measures what the checking costs, because that is the number nobody has: sidecar start, plan
milliseconds, check milliseconds per sample count, inverse kinematics milliseconds per sample, and how
many commands actually reached the controller.

The sidecar runs on the host over pipes, with the cuRobo environment's own python, so the container's
missing GPU decides nothing. What decides it is whether that environment is installed here.

Run (URSim up, in remote control):
    python scripts/ursim/probe_curobo_bed.py
    WILLY_PROFILE=ursim,ursim_curobo,ursim_ur3 python scripts/ursim/probe_curobo_bed.py

A powered-on URSim stands wherever the container left it, which can put the tool under the workspace
box (82 mm above the base, measured), and then the checked line is refused before it begins.
WILLY_BED_START_JOINTS, six radians separated by commas, is a configuration the bed drives to first,
through the checked joint move, before it measures anything:
    WILLY_BED_START_JOINTS=-2.0,-1.9,1.9,-1.5708,-1.5708,0.0 python scripts/ursim/probe_curobo_bed.py
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

os.environ.setdefault("WILLY_PROFILE", "ursim,ursim_curobo")

import numpy as np  # noqa: E402

from src.config import load_robot_config  # noqa: E402
from src.config.schema.robot import FixtureBoxConfig  # noqa: E402
from src.geometry import Frame, Pose  # noqa: E402
from src.robot.core import JointPositions, RobotVendor  # noqa: E402
from src.robot.core.errors import RobotMotionRejected  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402

record: dict[str, object] = {"profile": os.environ["WILLY_PROFILE"]}


def _timed(label: str, call):
    started = time.perf_counter()
    try:
        value = call()
    finally:
        record[f"{label}_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    return value


config = load_robot_config()
record["planner"] = config.ur.motion_planner
record["model"] = config.ur.model
if config.ur.motion_planner != "curobo":
    raise SystemExit(
        f"this bed needs the planner on; {os.environ['WILLY_PROFILE']} resolves to "
        f"motion_planner={config.ur.motion_planner!r}. Use WILLY_PROFILE=ursim,ursim_curobo."
    )

# The factory answers the vendor neutral `RobotArm`; this bed is about a UR, and the two verbs
# below (the decline block and the preflight) are the UR driver's own rather than the protocol's.
arm: Any = create_arm(RobotVendor.from_string(config.vendor), config=config)
arm.connect()
# URSim has no camera and nothing is looking at the cell, which the driver has to be told rather than
# left to assume: without this every checked motion would be refused before planning, stamped MISSING.
with arm.without_camera_world(reason="URSim bed: the container has no camera"):
    start = os.environ.get("WILLY_BED_START_JOINTS", "").strip()
    if start:
        record["powered_on_joints"] = [round(float(v), 5) for v in arm.get_joint_positions().tolist()]
        moved = _timed("move_to_start", lambda: arm.move_to_joints(
            JointPositions([float(v) for v in start.split(",")])))
        record["move_to_start"] = f"{moved.status}: {str(moved.message or '')[:200]}"
    here = arm.get_joint_positions()
    record["start_joints"] = [round(float(v), 5) for v in here.tolist()]

    # 1. A checked joint move: a small wrist rotation, which is short enough that the sampling is
    #    cheap and long enough that there is a line to judge.
    goal = list(here.tolist())
    goal[4] += 0.3
    _timed("checked_move_joint", lambda: arm.move_joint(JointPositions(goal)))
    record["after_move_joint"] = [round(float(v), 5) for v in arm.get_joint_positions().tolist()]

    # 2. A checked linear move of 50 mm, which is where the seeded controller IK per sample is paid.
    tcp = arm.get_tcp_pose()
    target = Pose(
        position_mm=np.asarray(tcp.position_mm) + np.array([0.0, 0.0, 50.0]),
        quaternion_xyzw=np.asarray(tcp.quaternion_xyzw),
        frame=Frame.BASE,
    )
    try:
        _timed("checked_move_linear", lambda: arm.move_linear(target))
        record["checked_move_linear"] = "moved"
    except RobotMotionRejected as refusal:
        # Recorded rather than fatal: the planned moves below measure something else, and they still
        # need a start.
        record["checked_move_linear"] = f"refused: {str(refusal)[:300]}"
        up = _timed("planned_move_up", lambda: arm.move(target))
        record["planned_move_up_status"] = str(up.status)
        reached = arm.get_tcp_pose()
        record["planned_move_up_end_error_mm"] = round(
            float(np.linalg.norm(np.asarray(reached.position_mm) - np.asarray(target.position_mm))), 3)
    record["after_move_linear_mm"] = [round(float(v), 3) for v in arm.get_tcp_pose().position_mm]

    # 3. A free move, which the planner plans and whose path is judged sample by sample.
    back = Pose(
        position_mm=np.asarray(tcp.position_mm),
        quaternion_xyzw=np.asarray(tcp.quaternion_xyzw),
        frame=Frame.BASE,
    )
    result = _timed("planned_move", lambda: arm.move(back))
    record["planned_move_status"] = str(result.status)
    record["planned_move_camera_world"] = str(result.camera_world.use)
    record["planned_move_message"] = str(result.message or "")[:400]
    # Where the controller says the tool ended, against the goal the planner was handed. The planner
    # is rooted at the URDF base_link, half a turn from the controller's base, and a plan to the
    # mirrored goal would end about twice the flange's distance from the base axis away.
    arrived = arm.get_tcp_pose()
    record["planned_move_end_error_mm"] = round(
        float(np.linalg.norm(np.asarray(arrived.position_mm) - np.asarray(back.position_mm))), 3)
    alignment = abs(float(np.dot(np.asarray(arrived.quaternion_xyzw), np.asarray(back.quaternion_xyzw))))
    record["planned_move_end_error_deg"] = round(float(np.degrees(2.0 * np.arccos(min(1.0, alignment)))), 3)

    # 4. The refusal, and the control that goes with it: a fixture declared across the line, so the
    #    endpoints stay clear and the middle does not. Nothing may move.
    before = arm.get_joint_positions().tolist()
    mid = (np.asarray(tcp.position_mm) + np.asarray(target.position_mm)) / 2.0
    fixture = FixtureBoxConfig(
        name="bed_probe_wall", center_mm=(float(mid[0]), float(mid[1]), float(mid[2])),
        half_extents_mm=(60.0, 60.0, 5.0),
    )
    preflight = arm.safety_preflight
    assert preflight is not None
    for guard in preflight.guards:
        setter = getattr(guard, "set_perceived_fixtures", None)
        if callable(setter):
            setter((fixture,))
    try:
        arm.move_linear(target)
    except RobotMotionRejected as refusal:
        record["refusal"] = str(refusal)[:400]
    else:
        record["refusal"] = None
    after = arm.get_joint_positions().tolist()
    record["joints_unchanged_after_refusal"] = bool(
        np.allclose(np.asarray(before), np.asarray(after), atol=1e-6)
    )

    # 5. The middle of a line. The box in step 4 sits 25 mm above the grasp centre, where the hand
    #    already stands at the start, so its refusal is the start pose's. Here the line runs 200 mm
    #    along the base's +X and a 20 mm post stands at its middle at the height of the gripper's
    #    body, 100 mm from either end: the start and the end are clear, and only a configuration in
    #    between can meet it. Then the control, the same line with the post gone, which has to move.
    here_tcp = arm.get_tcp_pose()
    far = Pose(
        position_mm=np.asarray(here_tcp.position_mm) + np.array([200.0, 0.0, 0.0]),
        quaternion_xyzw=np.asarray(here_tcp.quaternion_xyzw),
        frame=Frame.BASE,
    )
    post_centre = np.asarray(here_tcp.position_mm) + np.array([100.0, 0.0, 80.0])
    post = FixtureBoxConfig(
        name="bed_probe_post", center_mm=(float(post_centre[0]), float(post_centre[1]), float(post_centre[2])),
        half_extents_mm=(10.0, 10.0, 30.0),
    )
    for guard in preflight.guards:
        setter = getattr(guard, "set_perceived_fixtures", None)
        if callable(setter):
            setter((post,))
    before = arm.get_joint_positions().tolist()
    try:
        arm.move_linear(far)
    except RobotMotionRejected as refusal:
        record["middle_refusal"] = str(refusal)[:400]
    else:
        record["middle_refusal"] = None
    record["joints_unchanged_after_middle_refusal"] = bool(
        np.allclose(np.asarray(before), np.asarray(arm.get_joint_positions().tolist()), atol=1e-6)
    )
    for guard in preflight.guards:
        setter = getattr(guard, "set_perceived_fixtures", None)
        if callable(setter):
            setter(())
    try:
        _timed("middle_control_line", lambda: arm.move_linear(far))
        reached = arm.get_tcp_pose()
        record["middle_control"] = "moved"
        record["middle_control_end_error_mm"] = round(
            float(np.linalg.norm(np.asarray(reached.position_mm) - np.asarray(far.position_mm))), 3)
    except RobotMotionRejected as refusal:
        record["middle_control"] = f"refused: {str(refusal)[:300]}"

arm.disconnect()
print(json.dumps(record, indent=2))
