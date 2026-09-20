"""Calibrate a FIXED camera (eye-to-hand) from your own poses, each one presenting the board to it.

07 sweeps SweepOptions(poses=<count>), auto-generated tool-down poses. Here the board rides on the
flange and the camera stands still, so tool down holds the board face-down at the table: from the
camera's side it is an edge, and edge-on the marker either fails to decode or decodes with a pose
nobody should believe. Yaw cannot fix that, because yaw about the vertical never tips the board.

Pose.aimed_at(x, y, z, target_mm=...) points the tool's +Z at a point you name, which turns the
board to face it IF the board's face is the tool's +Z, which is how a plate bolted flat to the
flange sits. Mounted on a bracket facing some other way, aim at a point offset from the camera by
as much as the mounting turns it. Two kinds of pose are below: a few aimed at the camera, which is
what a first calibration wants, and a few tool-down ones, which are fine directly under a camera
looking down. 10 does the mirror image of this for a wrist camera.

Bolt a printed ArUco board to the tool flange, then run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/08_calibrate_a_fixed_camera_with_fixed_poses.py
"""

from willy import HandEyeCalibration, Pose, SweepOptions, load_tree

# Roughly where the camera hangs, in BASE millimetres, off a tape measure. This is the chicken and
# egg of a first calibration: you are running this to find out exactly where the camera is, and
# aiming at it needs to know already. A hundred millimetres of error here costs a few degrees of
# aim and the board stays in frame; the solve is what measures the camera properly.
camera_mm = (250.0, -600.0, 900.0)

# Position spread AND orientation spread, which is what a solve needs and what a ring of poses at
# one height does not give. Each board pose faces the camera from a different station.
fixed_poses = [
    Pose.aimed_at(400.0, 0.0, 350.0, target_mm=camera_mm, label="face_0"),
    Pose.aimed_at(400.0, 150.0, 420.0, target_mm=camera_mm, roll_deg=25.0, label="face_1"),
    Pose.aimed_at(350.0, -150.0, 300.0, target_mm=camera_mm, roll_deg=-25.0, label="face_2"),
    Pose.aimed_at(500.0, 0.0, 450.0, target_mm=camera_mm, roll_deg=45.0, label="face_3"),
    # Under a camera that looks down, a tool-down board faces it well enough, and these add the
    # flat-on views that a purely aimed set leaves out.
    Pose.tool_down(400.0, 0.0, 500.0, yaw_deg=0.0, label="down_0"),
    Pose.tool_down(400.0, -100.0, 450.0, yaw_deg=60.0, label="down_1"),
]
# Or read them from disk instead of writing them here:
#     SweepOptions(marker_length_mm=40.0, fixed_poses="examples/real_robot/eth_fixed_poses.json")
options = SweepOptions(marker_length_mm=40.0, fixed_poses=fixed_poses)
calibration = HandEyeCalibration.from_tree(load_tree(), rig_id="overhead", mode="eye_to_hand", options=options)

# The config alone: the rig, the marker, and this run's poses. Opens nothing.
print(calibration.check())

# Builds the arm alone and opens the camera, says what the arm will refuse, and stops before any
# motion. A refusal here is a report, and the sweep below would be refused the same way.
rehearsed = calibration.run(dry_run=True)
print(rehearsed)
if not rehearsed.ok:
    raise SystemExit(rehearsed.exit_code)

# The poses above, in order, not a generated sweep pattern. The report says how many of them the
# camera actually decoded the marker in: a pose it saw nothing from is a pose to move.
report = calibration.run()  # this moves the robot
print(report)  # ends with the rig block to paste under camera.cameras.rigs
raise SystemExit(report.exit_code)
