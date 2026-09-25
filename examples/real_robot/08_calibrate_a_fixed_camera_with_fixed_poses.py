"""Calibrate a FIXED camera (eye-to-hand) from your own poses, each one presenting the board to it.

The board rides on the flange and the camera stands still, so tool down holds the board face-down at the table: from
the camera's side it is an edge, and edge-on the marker either fails to decode or decodes with a pose nobody should
believe. Yaw cannot fix that, because yaw about the vertical never tips the board. Pose.aimed_at(x, y, z,
target_mm=...) points the tool's +Z at a point you name, which turns the board to face it IF the board's face is the
tool's +Z, as a plate bolted flat to the flange sits. Mounted on a bracket facing some other way, aim at a point
offset from the camera by as much as the mounting turns it. Below: a few poses aimed at the camera, which is what a
first calibration wants, and a few tool-down ones, fine directly under a camera looking down. Nothing generates a
pose: these, a file of them, or the stations 07 wrote while you guided the arm are what the arm drives to.

Bolt a printed ArUco board to the tool flange, then run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/08_calibrate_a_fixed_camera_with_fixed_poses.py
"""

from willy import HandEyeCalibration, Pose, SweepOptions, load_tree, print_sweep_progress

# Roughly where the camera hangs, in BASE millimetres, off a tape measure: a hundred millimetres of
# error costs a few degrees of aim and the board stays in frame; the solve measures the camera.
camera_mm = (250.0, -600.0, 900.0)

# Position spread AND orientation spread, which is what a solve needs and what a ring of poses at
# one height does not give. Each board pose faces the camera from a different station.
fixed_poses = [
    Pose.aimed_at(400.0, 0.0, 350.0, target_mm=camera_mm, label="face_0"),
    Pose.aimed_at(400.0, 150.0, 420.0, target_mm=camera_mm, roll_deg=25.0, label="face_1"),
    Pose.aimed_at(350.0, -150.0, 300.0, target_mm=camera_mm, roll_deg=-25.0, label="face_2"),
    Pose.aimed_at(500.0, 0.0, 450.0, target_mm=camera_mm, roll_deg=45.0, label="face_3"),
    # Under a camera that looks down a tool-down board faces it well enough: the flat-on views.
    Pose.tool_down(400.0, 0.0, 500.0, yaw_deg=0.0, label="down_0"),
    Pose.tool_down(400.0, -100.0, 450.0, yaw_deg=60.0, label="down_1"),
]
# Or read them from disk: SweepOptions(fixed_poses="examples/real_robot/eth_fixed_poses.json"), a
# template of tool-down poses, or the file 07 wrote: "calibration/real/eye_to_hand_overhead_stations.json".
# adjust=True frees the arm at each station for you to fine-tune it by hand (10 shows it).
# The arm drives itself carrying every wrist camera body your tree declares; one not calibrated yet
# refuses the check until 09 calibrates it or SweepOptions(unmodelled_wrist_body="<why>") says why.
# One declared with no body yet refuses nothing: the check and the build name it on a !! line.
options = SweepOptions(fixed_poses=fixed_poses, preview="auto")
calibration = HandEyeCalibration.from_tree(load_tree(), rig_id="overhead", mode="eye_to_hand",
                                           options=options, on_event=print_sweep_progress)

print(calibration.check())  # the config alone: the rig, the marker, and this run's poses. Opens nothing.

# Builds the arm alone and opens the camera, says what the arm will refuse, and stops before any motion.
rehearsed = calibration.run(dry_run=True)
print(rehearsed)
if not rehearsed.ok:
    raise SystemExit(rehearsed.exit_code)

# The poses above, in order. Each prints as it runs, and the report lists every pose: counted, or why
# not. A pose the camera saw nothing from is a pose to move. Keep the cell clear: the arm moves itself.
report = calibration.run()  # this moves the robot
print(report)  # ends with the rig block to paste under camera.cameras.rigs
raise SystemExit(report.exit_code)
