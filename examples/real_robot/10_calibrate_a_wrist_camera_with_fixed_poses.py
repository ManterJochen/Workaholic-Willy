"""Calibrate a WRIST camera (eye-in-hand) from your own fixed, look-at poses.

Same idea as 08_calibrate_a_fixed_camera_with_fixed_poses.py (CalibrationRoutine.run_with_poses /
run_from_json instead of the automatic sweep), but the other mounting: the camera rides on the
wrist and a fixed ArUco board sits still on the table, so the poses cannot vary freely the way the
fixed-camera example's can. A wrist camera pointed anywhere but at the board never sees it, so each
pose must aim the camera at the board: a "look-at" viewpoint, not just a position and a yaw.

Rather than hand-deriving that aiming math, this reuses the same pure-geometry helper the Isaac
validation stack uses for the identical problem (no simulator import, plain numpy):
`generate_hemisphere_viewpoints` places the camera on a hemisphere around the board and aims its
optical axis at it, `look_at_world_matrix` underneath. See
`src/willy_sim/calibration/hand_eye.py` for both.

Fix a printed ArUco board flat on the table within reach, then run it at the cell, under the
cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/10_calibrate_a_wrist_camera_with_fixed_poses.py
"""

import numpy as np

from willy import Camera, Frame, load_tree
from src.calibration.rgbd_marker_source import RGBDArucoMarkerSource
from src.geometry import Transform
from src.robot.execution.calibration import CalibrationRoutine
from src.robot.execution.robot import Robot
from src.willy_sim.calibration.hand_eye import generate_hemisphere_viewpoints

tree = load_tree()
rig_id = "wrist"  # the rig_id your camera section gives this wrist camera
marker_length_mm = 40.0  # measure the printed edge
marker_pos_mm = np.array([500.0, 0.0, 0.0])  # where you place the board on the table, in BASE, roughly

# A rough camera-on-flange offset, from the mount's CAD or a tape measure: only used to plan poses
# that keep the board in frame, not the answer this calibration solves. 40 mm back along the tool's
# own +Z, pointing the same way the tool does, is a placeholder; put your own mount's rough numbers in.
t_cam_to_tool_rough = Transform(
    translation_mm=np.array([0.0, 0.0, -40.0]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    from_frame=Frame.CAMERA, to_frame=Frame.TOOL,
)

# Fewer, closer rings than the sim default: a real cell's reachable volume around a table-top board
# is smaller than the sim's free space. Every viewpoint is still just a planned target; a pose the
# arm cannot reach is skipped (logged as "move_rejected"), not a hard failure.
viewpoints = generate_hemisphere_viewpoints(
    marker_pos_mm, t_cam_to_tool_rough,
    radii_mm=(300.0, 380.0), elevations_deg=(20.0, 35.0), azimuths_deg=(0.0, 90.0, 180.0, 270.0),
)
fixed_poses = [v.tcp_pose for v in viewpoints]

# The arm alone, no gripper, and built without cameras: CalibrationRoutine declines the camera
# world for every move on its own, and a robot built with cameras would refuse a declined move.
robot = Robot.from_config(tree.robot, gripper=None)
print(robot)

with Camera.from_tree(tree, rig_id=rig_id) as camera:
    handle = camera.handle()
    marker_source = RGBDArucoMarkerSource(
        streamer=handle, marker_length_mm=marker_length_mm, dict_name="DICT_4X4_50", target_id=0,
    )
    routine = CalibrationRoutine(
        arm=robot.arm, marker_source=marker_source,
        workspace_limits=tree.robot.workspace_limits, eth_settings=tree.app_config.camera.hand_eye.eye_in_hand,
        rig_id=rig_id, marker_id=0, settle_time_s=tree.robot.calibration.settle_time_s,
        calibration_mode="eye_in_hand",
    )

    # As in 16: a Python list (above) or a JSON file. Orientation here comes from the look-at
    # geometry above rather than something to hand-write, so save the generated poses once and
    # replay them from disk on the next run instead of recomputing:
    #     from src.geometry.serialization import pose_to_dict
    #     import json
    #     json.dump([pose_to_dict(p) for p in fixed_poses], open("eih_fixed_poses.json", "w"))
    # then swap the call below for:
    #     result = routine.run_from_json(Path(__file__).with_name("eih_fixed_poses.json"), ...)
    with robot.connected():
        result = routine.run_with_poses(
            fixed_poses, dataset_save_path=f"calibration/real/eih_{rig_id}_fixed_dataset.json",
        )

print(f"accepted {result.num_samples}/{len(fixed_poses)} poses, "
      f"rmse {result.rmse_mm:.2f} mm, max error {result.max_error_mm:.2f} mm")
if result.transform is None:
    raise SystemExit("no transform solved: check the board was in view of every reachable pose")
print(result.transform.to_matrix())  # CAMERA -> TOOL; paste under camera.cameras.rigs[<rig_id>].extrinsics
