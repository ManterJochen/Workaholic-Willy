#!/usr/bin/env bash
# Calibrate a camera on the flange: the CAMERA->TOOL transform, from the command line.
#
# The Python twin of this file is scripts/examples/api/02_calibration/calibrate_wrist_camera.py and
# it drives the same CalibrationRoutine with the same MountingMode. `--rig` names the camera bolted
# to the flange; the shipped tree's only RGB-D rig is realsense_d435, so that is what is named here.
set -euo pipefail

# 1. Validate the config and the rig. Touches no hardware.
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --mode eye_in_hand --check

# 2. Build the arm driver and open that one camera, then stop before any motion. Needs the device.
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --mode eye_in_hand --dry-run

# 3. The sweep. It writes eih_<rig>.json, which fusion.extrinsics_artifact_path refuses by schema:
#    wire it under fusion.cameras.<rig> with mounting_mode eye_in_hand instead. This moves the arm.
#    python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --mode eye_in_hand --poses 22
