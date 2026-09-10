"""Solve the CAMERA->TOOL transform of a camera on the flange. This moves the robot.

Same routine and same AX=XB solve as the fixed camera, with MountingMode.EYE_IN_HAND: the answer is
half a transform that EyeInHandFrameResolver completes with the live TCP, never a primary resolver.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 02_calibration, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import ConfigError, load_config  # noqa: E402
from src.calibration import MountingMode, load_cam_to_tool, save_cam_to_tool  # noqa: E402
from src.calibration.rgbd_marker_source import RGBDArucoMarkerSource  # noqa: E402
from src.camera.orchestration.frame_provider import FrameProvider  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.execution.calibration import CalibrationRoutine  # noqa: E402

# 1. Name the wrist rig. On this path the thresholds and the marker size come from
#    camera.hand_eye.EYE_TO_HAND; camera.hand_eye.eye_in_hand is read by the sim runner only.
try:
    app = load_config()
except ConfigError as broken:
    # A tree that does not parse is the state an operator is in five seconds after a bad edit,
    # and the four files under scripts/checks/ answer it with a sentence and exit 2. An example
    # that raises instead teaches a reader nothing about the thing it was written to show.
    print(f"this config tree does not load: {broken}")
    raise SystemExit
settings = app.camera.hand_eye.eye_to_hand
rig_id = "wrist"  # whatever rig_id names the camera on the flange
rig = next((r for r in app.camera.cameras.rigs if r.rig_id == rig_id), None)
if rig is None or rig.source != "rgbd" or not rig.enabled:
    print(f"no enabled RGB-D rig {rig_id!r} in camera.cameras.rigs; declared: "
          f"{[(r.rig_id, r.source, r.enabled) for r in app.camera.cameras.rigs]}")
    sys.exit(0)

# 2. The arm and the wrist camera, opened through the provider the pick path uses. Check the cable
#    has slack for every pose before this runs.
arm = create_arm(app.robot.vendor, config=app.robot)
provider = FrameProvider(list(app.camera.cameras.rigs))
provider.open_rig(rig_id)
marker = RGBDArucoMarkerSource(streamer=provider.rig(rig_id), target_id=0,
                               marker_length_mm=settings.marker_length_mm,
                               dict_name=settings.aruco_dict_name)

# 3. EYE_IN_HAND is the whole difference: here the board lies still and the camera moves over it.
routine = CalibrationRoutine(arm=arm, marker_source=marker, rig_id=rig_id,
                             workspace_limits=app.robot.workspace_limits, eth_settings=settings,
                             calibration_mode=MountingMode.EYE_IN_HAND)
arm.connect()
result = routine.run_auto(22, orientation_spread_deg=30.0, seed=0)
arm.disconnect()
provider.release()

# 4. save_cam_to_tool and result.transform: result.extrinsics is None in this mode, and the file
#    carries schema willy.calibration.cam_to_tool/1, which the single fusion key refuses by name.
if result.transform is None:
    print("the solver returned no transform, so there is nothing to persist")
else:
    path = save_cam_to_tool(f"calibration/real/eih_{rig_id}.json", result.transform, rig_id=rig_id)
    t = load_cam_to_tool(path)
    print(f"camera at {[round(float(v), 1) for v in t.translation_mm]} mm from the tool; wire it as "
          f"robot.grasping.fusion.cameras.{rig_id}.mounting_mode eye_in_hand -> {path}")
