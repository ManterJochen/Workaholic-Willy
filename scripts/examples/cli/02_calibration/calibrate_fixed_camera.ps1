# Calibrate a camera bolted in the cell: the CAMERA->BASE transform, from the command line.
#
# The Python twin of this file is scripts/examples/api/02_calibration/calibrate_fixed_camera.py and
# it drives the same CalibrationRoutine with the same MountingMode. The runner is the only half that
# writes the artifact keyed by rig id, which is the id `grasping.fusion.cameras` is keyed by too.

# 1. Validate the config and the rig. Touches no hardware.
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --mode eye_to_hand --check

# 2. Build the arm driver and open that one camera, then stop before any motion. Needs the device.
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --mode eye_to_hand --dry-run

# 3. The sweep. This one moves the arm to 22 poses, so clear the cell first.
#    python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --mode eye_to_hand --poses 22 --marker-length-mm 39.7

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
