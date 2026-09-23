"""Calibrate a camera fixed in the cell: its CAMERA->BASE transform, from a sweep of the arm.

Bolt a printed ArUco board to the tool flange, then run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/07_calibrate_a_fixed_camera.py
"""

from willy import HandEyeCalibration, SweepOptions, load_tree, print_sweep_progress

# The rig_id your camera section gives this camera; the artifact is keyed by it. The target is
# camera.hand_eye.eye_to_hand's: its marker_length_mm, aruco_dict_name and marker_id, or the
# ChArUco board its `target` names. Measure the printed edge: a wrong length scales every
# translation and the solve still converges, so no residual will show it.
# SweepOptions(marker_length_mm=...) overrides the YAML for one run. preview="auto" opens a window
# where one can show: the camera, and each judged frame with the target drawn on it. Closing it
# closes the window only; the pendant stops the robot.
calibration = HandEyeCalibration.from_tree(
    load_tree(), rig_id="overhead", mode="eye_to_hand", options=SweepOptions(preview="auto"),
    on_event=print_sweep_progress,  # a line per pose: where it goes, what it saw, counted or why not
)

# The config alone: the rig, the target, the poses and where the artifact goes. Opens nothing.
print(calibration.check())

# Builds the arm alone and opens the camera, says what the arm will refuse, and stops before any
# motion. A refusal here is a report, and the sweep below would be refused the same way.
rehearsed = calibration.run(dry_run=True)
print(rehearsed)
if not rehearsed.ok:
    raise SystemExit(rehearsed.exit_code)

# The sweep takes the cell lock, connects the arm alone and carries the board through the poses
# in view of the camera. Every move declines the camera world, because this sweep produces the
# CAMERA->BASE that world is built from, so keep the cell clear while it runs.
report = calibration.run()  # this moves the robot
print(report)  # every pose's verdict, then the rig block to paste under camera.cameras.rigs
raise SystemExit(report.exit_code)
