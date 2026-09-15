"""Solve the CAMERA->BASE transform of a camera bolted in the cell. This moves the robot.

The arm carries an ArUco board to generated poses, the fixed camera watches it, and AX=XB solves the
transform. EYE_TO_HAND makes the answer CAMERA->BASE, which the camera's rig then declares.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 02_calibration, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import ConfigError, load_config  # noqa: E402
from src.calibration import MountingMode, classify_rmse, save_extrinsics  # noqa: E402
from src.calibration.rgbd_marker_source import RGBDArucoMarkerSource  # noqa: E402
from src.camera.orchestration.camera import Camera  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.execution.calibration import CalibrationRoutine  # noqa: E402

# 1. Calibrate the camera a cell grasps from: `camera.cameras.primary_rig_id`, the key the cell
#    itself reads. A marker pose needs live colour plus that camera's intrinsics, so the rig has to
#    be an RGB-D device that is switched on, and this says which rigs qualify when it is not.
try:
    app = load_config()
except ConfigError as broken:
    # A tree that does not parse is the state an operator is in five seconds after a bad edit,
    # and the four files under scripts/checks/ answer it with a sentence and exit 2. An example
    # that raises instead teaches a reader nothing about the thing it was written to show.
    print(f"this config tree does not load: {broken}")
    raise SystemExit
settings = app.camera.hand_eye.eye_to_hand
rig = next(r for r in app.camera.cameras.rigs if r.rig_id == app.camera.cameras.primary_rig_id)
if rig.source != "rgbd" or not rig.enabled:
    print(f"primary rig {rig.rig_id!r} is source={rig.source!r} enabled={rig.enabled}; the RGB-D "
          f"rigs here are {[r.rig_id for r in app.camera.cameras.rigs if r.source == 'rgbd']}")
    sys.exit(0)

# 2. The arm, and one camera held by its `Camera` owner, as on the pick path. Measure the
#    printed square: declared 50 mm and printed 48 mm scales every translation by 4 % and converges.
arm = create_arm(app.robot.vendor, config=app.robot)
camera = Camera.from_config(app.camera, rig_id=rig.rig_id)
camera.open()
marker = RGBDArucoMarkerSource(streamer=camera.handle(), target_id=0,
                               marker_length_mm=settings.marker_length_mm,
                               dict_name=settings.aruco_dict_name)

# 3. The sweep, tool down. 30 degrees of spread: a planar ArUco seen near-frontally flips (IPPE).
routine = CalibrationRoutine(arm=arm, marker_source=marker, rig_id=rig.rig_id,
                             workspace_limits=app.robot.workspace_limits, eth_settings=settings,
                             calibration_mode=MountingMode.EYE_TO_HAND)
arm.connect()
result = routine.run_auto(22, orientation_spread_deg=30.0, seed=0)
arm.disconnect()
camera.release()

# 4. The artifact is keyed by the `rig_id` the routine was given, the rig that declares it. The band
#    below and the file's `quality` are the library defaults; an RMSE only says the poses agree, and
#    a pick that lands proves the frame. `extrinsics` is filled only by the EYE_TO_HAND branch, so it
#    is Optional even where the mode makes it certain. Checked rather than asserted: the day someone
#    runs this routine in the other mode, a printed sentence beats an AttributeError.
if result.extrinsics is None:
    print(f"the solver returned no extrinsics for mode {result.mode}; "
          "eye-in-hand carries its answer in `transform` instead, not here")
else:
    path = save_extrinsics(f"calibration/real/eth_{rig.rig_id}.json", result.extrinsics)
    print(f"{result.num_samples} samples, rmse {result.rmse_mm:.3f} mm -> {classify_rmse(result.rmse_mm)}; "
          f"paste into the camera section:\ncamera:\n  cameras:\n    rigs:\n      - rig_id: {rig.rig_id}\n"
          f"        extrinsics:\n          mounting_mode: eye_to_hand\n          artifact_path: {path}")
