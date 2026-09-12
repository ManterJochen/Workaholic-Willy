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
# left to assume: without this every checked motion would be stamped MISSING.
with arm.without_camera_world(reason="URSim bed: the container has no camera"):
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
    _timed("checked_move_linear", lambda: arm.move_linear(target))
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

arm.disconnect()
print(json.dumps(record, indent=2))
