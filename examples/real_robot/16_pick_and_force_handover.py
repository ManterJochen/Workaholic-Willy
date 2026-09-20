"""Move the arm, pick a known part, and hand it to a person: wait on the TCP wrench, then release.

This stack ships no active force control: no UR force mode, no admittance or compliant motion, no
freedrive from Python. ``SupportsForceTorque`` is read-only (``get_tcp_wrench`` / ``get_joint_torques``),
and the UR driver documents it as exactly one thing: the hand-over signal a person's tug produces on a
held part. That is what this example uses it for.

Run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/16_pick_and_force_handover.py
"""

import time

from willy import Pose, Robot, load_tree

robot = Robot.from_tree(load_tree())

# A known part and a pose held out to a person, in millimetres in the robot's base frame, tool down.
part = Pose.tool_down(450.0, 100.0, 120.0, yaw_deg=90.0)
handover = Pose.tool_down(300.0, 0.0, 400.0)
part_width_mm = 40.0

# The baseline TCP force this driver reports while it holds the part and nobody touches it, plus how
# far above that a person's pull must go before this counts it as a hand-over rather than noise. Read
# both off your own cell: weigh the part, watch get_tcp_wrench() idle, and set pull_threshold_n above
# what you see there.
pull_threshold_n = 15.0
poll_interval_s = 0.05
timeout_s = 30.0

bench = "a known part on a clear table, no camera"

with robot.connected():
    picked = robot.pick(part, part_width_mm, decline=bench)
    print(picked)
    if not picked.ok:
        raise SystemExit("pick failed, nothing to hand over")

    print(robot.move(handover))

    if not hasattr(robot.arm, "get_tcp_wrench"):
        raise SystemExit(
            f"{type(robot.arm).__name__} reports no TCP wrench: this driver has no hand-over signal, "
            "so the part stays held. Release it another way (a button, a timeout) instead."
        )

    print("holding the part out; pull it to release")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        wrench = robot.arm.get_tcp_wrench()
        if wrench.force_magnitude >= pull_threshold_n:
            break
        time.sleep(poll_interval_s)
    else:
        print(f"no pull above {pull_threshold_n} N within {timeout_s} s; releasing anyway")

    print(robot.release())
