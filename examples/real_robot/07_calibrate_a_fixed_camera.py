"""Calibrate a camera fixed in the cell: its CAMERA->BASE transform, from a sweep of the arm.

Bolt a printed ArUco board to the tool flange, then run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/07_calibrate_a_fixed_camera.py
"""

from willy import HandEyeCalibration, SweepOptions, load_tree

# The rig_id your camera section gives this camera; the artifact is keyed by it. Measure the
# printed edge of the marker: a wrong length scales every translation and the solve still
# converges, so no residual will show it.
calibration = HandEyeCalibration.from_tree(
    load_tree(), rig_id="overhead", mode="eye_to_hand",
    options=SweepOptions(marker_length_mm=40.0),
)

# The config alone: the rig, the marker, the poses and where the artifact goes. Opens nothing.
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
print(report)  # ends with the rig block to paste under camera.cameras.rigs
raise SystemExit(report.exit_code)
