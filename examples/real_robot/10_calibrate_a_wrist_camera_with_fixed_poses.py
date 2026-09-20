"""Calibrate a WRIST camera (eye-in-hand) from your own poses, each one AIMED at the board.

09 sweeps SweepOptions(poses=<count>), auto-generated tool-down poses. That is the wrong shape for
a wrist camera: tool down aims the camera at whatever is under the flange, so from anywhere but
straight over the board it photographs the table beside it and the sweep collects nothing. The
solve then fails for want of samples, and nothing says the poses were the problem.

Pose.aimed_at(x, y, z, target_mm=...) is the fix: it points the tool's +Z at a point you name, so
the camera keeps the board in view from every station of a ring around it. 08 does the mirror image
of this for a fixed camera. What the aim does NOT do is promise the pose is reachable or the marker
visible: the arm's guards judge the first and the run reports the second per pose.

Fix a printed ArUco board flat on the table within reach, then run it at the cell, under the
cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/10_calibrate_a_wrist_camera_with_fixed_poses.py
"""

import math

from willy import HandEyeCalibration, Pose, SweepOptions, load_tree

# Where you put the board on the table, in BASE millimetres. Measure it roughly; the aim only has
# to bring the board into frame, and the solve is what measures anything precisely.
board_x_mm, board_y_mm, board_z_mm = 500.0, 0.0, 0.0

# A ring of stations around the board, each one looking down at it from a different side and
# height. A solve needs spread in BOTH position and orientation, and a ring of aimed poses gives
# both at once: the camera swings around the board rather than sliding over it.
fixed_poses = [
    Pose.aimed_at(
        board_x_mm + radius_mm * math.cos(math.radians(bearing_deg)),
        board_y_mm + radius_mm * math.sin(math.radians(bearing_deg)),
        board_z_mm + height_mm,
        target_mm=(board_x_mm, board_y_mm, board_z_mm),
        label=f"look_{index}",
    )
    for index, (bearing_deg, radius_mm, height_mm) in enumerate((
        (0.0, 0.0, 380.0),      # straight over the board: this one IS a tool-down pose
        (0.0, 120.0, 340.0),
        (90.0, 140.0, 300.0),
        (180.0, 120.0, 360.0),
        (270.0, 140.0, 320.0),
        (45.0, 170.0, 280.0),
    ))
]
# Or read them from disk instead of writing them here, as 08 shows:
#     SweepOptions(marker_length_mm=40.0, fixed_poses="examples/real_robot/eth_fixed_poses.json")

# The camera's own body is placed on the flange from its calibration, so a first sweep cannot place
# it, and this says why it may run without one; unmodelled_wrist_body="" (or leaving it unset)
# refuses instead, the way 09's automatic sweep does on a first calibration.
options = SweepOptions(marker_length_mm=40.0, fixed_poses=fixed_poses,
                       unmodelled_wrist_body="first calibration: nothing can place the camera body before it")
calibration = HandEyeCalibration.from_tree(load_tree(), rig_id="wrist", mode="eye_in_hand", options=options)

print(calibration.check())  # the config alone: the rig, its body and the marker; opens nothing
rehearsed = calibration.run(dry_run=True)  # built and attested, the body placed or declined
print(rehearsed)
if not rehearsed.ok:
    raise SystemExit(rehearsed.exit_code)

# The six poses above, in order, not a generated sweep pattern. The report says how many of them
# actually saw the marker: a station that saw nothing is a station to move, not a failed solve.
report = calibration.run()  # this moves the robot
print(report)  # the flange to TCP it recorded, and the rig block with the tolerances to measure
raise SystemExit(report.exit_code)
