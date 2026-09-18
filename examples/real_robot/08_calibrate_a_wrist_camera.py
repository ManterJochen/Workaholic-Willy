"""Calibrate a camera on the arm's wrist: its CAMERA->TOOL transform, from a sweep over a board.

Fix a printed ArUco board flat on the table within reach, then run it at the cell, under the
cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/08_calibrate_a_wrist_camera.py
"""

from willy import HandEyeCalibration, SweepOptions, load_tree

tree = load_tree()

# The solve is CAMERA->TOOL against the TCP the arm reports, which is the flange times the cell's
# tool frame. So the artifact records that flange to TCP, and the rig is refused later once the
# tool frame moves away from it: measure robot.gripper.tool_frame before you sweep. A tool frame
# left undeclared records nothing.
print("tool frame source:", tree.robot.gripper.tool_frame.source)

# The camera's own body is placed on the flange from its calibration, so a first sweep cannot
# place it, and on a cell that plans with geometry it is refused until you say why it may run
# without it. It then sweeps with no camera body in the planner and the guard. The reason is
# read only when the body cannot be placed; a recalibration places it from the previous solve.
options = SweepOptions(
    marker_length_mm=40.0,  # the printed edge, as you measured it
    unmodelled_wrist_body="first calibration: nothing can place the camera body before it",
)
calibration = HandEyeCalibration.from_tree(tree, rig_id="wrist", mode="eye_in_hand",
                                           options=options)

print(calibration.check())  # the config alone: the rig, its body and the marker; opens nothing
rehearsed = calibration.run(dry_run=True)  # built and attested, the body placed or declined
print(rehearsed)
if not rehearsed.ok:
    raise SystemExit(rehearsed.exit_code)

# The sweep takes the cell lock, connects the arm alone and carries the camera over the board.
# Every move declines the camera world, because a wrist camera's world needs the transform this
# sweep solves, so keep the cell clear while it runs.
report = calibration.run()  # this moves the robot
print(report)  # the flange to TCP it recorded, and the rig block with the tolerances to measure
raise SystemExit(report.exit_code)
