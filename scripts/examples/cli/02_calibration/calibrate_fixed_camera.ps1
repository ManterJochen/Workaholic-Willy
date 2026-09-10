# Calibrate a camera bolted in the cell: the CAMERA->BASE transform, from the command line.
#
# The Python twin of this file is scripts/examples/api/02_calibration/calibrate_fixed_camera.py and
# it drives the same CalibrationRoutine with the same MountingMode. The runner is the only half that
# writes the artifact keyed by rig id, which is the id `grasping.fusion.cameras` is keyed by too.

# 1. Validate the config and the rig. Touches no hardware. It refuses when the tree ships
#    this rig switched off, which the base tree does:
#    camera.cameras.rigs['realsense_d435'].enabled is false, and calibrating a camera the
#    cell will not open writes an artifact nothing reads. That refusal is this step's
#    answer, so it is reported rather than inherited.
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --mode eye_to_hand --check
if ($LASTEXITCODE -ne 0) {
    Write-Output "nothing to calibrate on this tree yet: the line above names the key to switch on."
    $global:LASTEXITCODE = 0
}

# 2. Build the arm driver and open that one camera, then stop before any motion. It needs the
#    device, so on a desk with no RealSense attached it refuses at the build and names it.
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --mode eye_to_hand --dry-run
if ($LASTEXITCODE -ne 0) {
    Write-Output "--dry-run got no further than the line above; it opens the camera, so it needs the device attached."
    $global:LASTEXITCODE = 0
}

# 3. The sweep. This one moves the arm to 22 poses, so clear the cell first.
#    python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --mode eye_to_hand --poses 22 --marker-length-mm 39.7

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
