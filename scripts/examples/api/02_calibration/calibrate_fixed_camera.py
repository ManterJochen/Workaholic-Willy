"""Solve the CAMERA->BASE transform of a camera bolted in the cell. This moves the robot.

The arm carries an ArUco board to generated poses, the fixed camera watches it, and AX=XB solves the
transform. EYE_TO_HAND makes the answer CAMERA->BASE, the only kind a primary resolver can be.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 02_calibration, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import load_config  # noqa: E402
from src.calibration import MountingMode, classify_rmse, save_extrinsics  # noqa: E402
from src.calibration.rgbd_marker_source import RGBDArucoMarkerSource  # noqa: E402
from src.camera.orchestration.frame_provider import FrameProvider  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.execution.calibration import CalibrationRoutine  # noqa: E402

# 1. Calibrate the camera a cell grasps from: `camera.cameras.primary_rig_id`, the key the cell
#    itself reads. A marker pose needs live colour plus that camera's intrinsics, so the rig has to
#    be an RGB-D device that is switched on, and this says which rigs qualify when it is not.
app = load_config()
settings = app.camera.hand_eye.eye_to_hand
rig = next(r for r in app.camera.cameras.rigs if r.rig_id == app.camera.cameras.primary_rig_id)
if rig.source != "rgbd" or not rig.enabled:
    print(f"primary rig {rig.rig_id!r} is source={rig.source!r} enabled={rig.enabled}; the RGB-D "
          f"rigs here are {[r.rig_id for r in app.camera.cameras.rigs if r.source == 'rgbd']}")
    sys.exit(0)

# 2. The arm, and exactly one camera, opened through the provider the pick path uses. Measure the
#    printed square: declared 50 mm and printed 48 mm scales every translation by 4 % and converges.
arm = create_arm(app.robot.vendor, config=app.robot)
provider = FrameProvider(list(app.camera.cameras.rigs))
provider.open_rig(rig.rig_id)
marker = RGBDArucoMarkerSource(streamer=provider.rig(rig.rig_id), target_id=0,
                               marker_length_mm=settings.marker_length_mm,
                               dict_name=settings.aruco_dict_name)

# 3. The sweep, tool down. 30 degrees of spread: a planar ArUco seen near-frontally flips (IPPE).
routine = CalibrationRoutine(arm=arm, marker_source=marker, rig_id=rig.rig_id,
                             workspace_limits=app.robot.workspace_limits, eth_settings=settings,
                             calibration_mode=MountingMode.EYE_TO_HAND)
arm.connect()
result = routine.run_auto(22, orientation_spread_deg=30.0, seed=0)
arm.disconnect()
provider.release()

# 4. The artifact is keyed by the `rig_id` the routine was given, and `grasping.fusion.cameras` is
#    keyed by the same id. The band below is the library's default one, and so is the `quality`
#    stored in the file; an RMSE only says the poses agree. A pick that lands proves the frame.
#    `extrinsics` is filled only by the EYE_TO_HAND branch of the solver, so it is Optional on the
#    type even here where the mode makes it certain. Checked rather than asserted: the day someone
#    runs this routine in the other mode, a printed sentence beats an AttributeError.
if result.extrinsics is None:
    print(f"the solver returned no extrinsics for mode {result.mode}; "
          "eye-in-hand carries its answer in `transform` instead, not here")
else:
    path = save_extrinsics(f"calibration/real/eth_{rig.rig_id}.json", result.extrinsics)
    print(f"{result.num_samples} samples, rmse {result.rmse_mm:.3f} mm -> "
          f"{classify_rmse(result.rmse_mm)}; set grasping.fusion.extrinsics_artifact_path: {path}")
