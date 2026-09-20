"""Calibrate a FIXED camera (eye-to-hand) from your own fixed poses, instead of the automatic sweep.

07_calibrate_a_fixed_camera.py drives HandEyeCalibration, whose sweep always calls
CalibrationRoutine.run_auto(...) underneath: there is no SweepOptions switch for a caller's own
poses. CalibrationRoutine itself offers that path directly, run_with_poses(list[Pose]) or
run_from_json(path), so this example builds the same pieces HandEyeCalibration builds internally
and calls one of those instead.

Eye-to-hand: the camera is fixed in the cell and the ArUco board rides on the tool flange, so the
arm carries the board past a stationary camera. See 10_calibrate_a_wrist_camera_with_fixed_poses.py
for the other mounting, eye-in-hand, where it is the other way around.

Bolt a printed ArUco board to the tool flange, then run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/08_calibrate_a_fixed_camera_with_fixed_poses.py
"""

import numpy as np

from willy import Camera, Frame, Pose, load_tree
from src.calibration.rgbd_marker_source import RGBDArucoMarkerSource
from src.geometry.quaternion import from_axis_angle, multiply
from src.robot.execution.calibration import CalibrationRoutine
from src.robot.execution.robot import Robot

tree = load_tree()
rig_id = "overhead"  # the rig_id your camera section gives this fixed camera
marker_length_mm = 40.0  # measure the printed edge; a wrong length scales every translation

# Fixed TCP target poses (Frame.BASE), each carrying the marker board in view of the fixed camera.
# A solve needs both position spread and orientation spread; tool_down alone only varies yaw, so a
# few poses here also tilt the board a little about base X, composed onto the tool-down quaternion.
def _tilted(x_mm: float, y_mm: float, z_mm: float, *, yaw_deg: float, tilt_deg: float) -> Pose:
    base = Pose.tool_down(x_mm, y_mm, z_mm, yaw_deg=yaw_deg)
    tilt = from_axis_angle(np.array([np.radians(tilt_deg), 0.0, 0.0]))
    return Pose(
        position_mm=base.position_mm, quaternion_xyzw=multiply(tilt, base.quaternion_xyzw),
        frame=Frame.BASE, label=base.label,
    )


fixed_poses = [
    Pose.tool_down(400.0, 0.0, 350.0, yaw_deg=0.0, label="pose_0"),
    Pose.tool_down(400.0, 150.0, 350.0, yaw_deg=20.0, label="pose_1"),
    Pose.tool_down(400.0, -150.0, 350.0, yaw_deg=-20.0, label="pose_2"),
    _tilted(350.0, 0.0, 400.0, yaw_deg=0.0, tilt_deg=15.0),
    _tilted(350.0, 100.0, 400.0, yaw_deg=15.0, tilt_deg=-15.0),
    _tilted(350.0, -100.0, 300.0, yaw_deg=-15.0, tilt_deg=15.0),
]

# The arm alone, no gripper: a sweep beside a board needs no activation stroke, and a robot built
# with cameras would hold a live world that refuses every declined sweep move.
robot = Robot.from_config(tree.robot, gripper=None)
print(robot)

with Camera.from_tree(tree, rig_id=rig_id) as camera:
    handle = camera.handle()
    marker_source = RGBDArucoMarkerSource(
        streamer=handle, marker_length_mm=marker_length_mm, dict_name="DICT_4X4_50", target_id=0,
    )
    routine = CalibrationRoutine(
        arm=robot.arm, marker_source=marker_source,
        workspace_limits=tree.robot.workspace_limits, eth_settings=tree.robot.calibration,
        rig_id=rig_id, marker_id=0, settle_time_s=tree.robot.calibration.settle_time_s,
        calibration_mode="eye_to_hand",
    )

    # Two equivalent ways to hand the routine your poses: a Python list (above), built above as
    # `fixed_poses`, or a JSON file on disk (below), each record either URPose-shaped
    # ({x,y,z,rx,ry,rz,label}, mm + axis-angle rad) or Pose-shaped ({schema, frame, label,
    # position_mm, quaternion_xyzw}). eth_fixed_poses.json beside this file is URPose-shaped, the
    # simpler one to hand-write. Swap the call below for the JSON file instead of the Python list:
    #     result = routine.run_from_json(
    #         Path(__file__).with_name("eth_fixed_poses.json"),
    #         dataset_save_path=f"calibration/real/eth_{rig_id}_fixed_dataset.json",
    #     )
    with robot.connected():
        result = routine.run_with_poses(
            fixed_poses, dataset_save_path=f"calibration/real/eth_{rig_id}_fixed_dataset.json",
        )

print(f"accepted {result.num_samples}/{len(fixed_poses)} poses, "
      f"rmse {result.rmse_mm:.2f} mm, max error {result.max_error_mm:.2f} mm")
if result.transform is None:
    raise SystemExit("no transform solved: check the marker was in view of every fixed pose")
print(result.transform.to_matrix())  # paste under camera.cameras.rigs[<rig_id>].extrinsics
