"""A parallel jaw and a suction cup ask two different questions of the same box.

The jaw wants an aperture, an antipodal pair and a closing axis; the cup wants one sealable patch
and a direction to press along. Sweeping one box from tall to flat says where each modality stops.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 05_grasping, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

from src.config import load_robot_config  # noqa: E402
from src.geometry import Frame, Transform  # noqa: E402
from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry  # noqa: E402
from src.robot.execution.autonomous_grasp.rehearsal import RehearsalPerceptionSource  # noqa: E402
from src.robot.grasping.collision import SupportPlane  # noqa: E402
from src.robot.grasping.generation.calculator import GraspCalculator  # noqa: E402
from src.robot.grasping.suction import SuctionConfig, synthesize_suction_grasps  # noqa: E402

# 1. The gripper this cell declares: the driver that actuates it, and the envelope jaw candidates
#    are filtered against. `gripper_geometry.kind` chooses that envelope, never the modality.
robot = load_robot_config()
jaw_model = build_gripper_geometry(robot.grasping.gripper_geometry)
support = SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=0.0, frame=Frame.BASE)
print(f"gripper {robot.gripper.vendor}, {robot.gripper.min_width_mm:.0f} to "
      f"{robot.gripper.max_width_mm:.0f} mm, envelope {robot.grasping.gripper_geometry.kind}")

# 2. The same synthetic box at falling heights, nothing else changed: no camera, no arm, and no
#    claim about grasp quality. What moves is the envelope and the table check.
for height_mm in (80.0, 50.0, 45.0, 40.0, 20.0):
    scene = RehearsalPerceptionSource(object_height_mm=height_mm)
    frame = scene.acquire()
    # 3. A nadir CAMERA->BASE transform built from the scene's own plane, so the support surface
    #    lands at z = 0. A real cell reads this from the calibration artifact instead.
    camera_to_base = Transform(
        translation_mm=np.array([400.0, 0.0, scene.plane_mm]),
        quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
        from_frame=Frame.CAMERA, to_frame=Frame.BASE)
    # 4. Jaw: an aperture, an antipodal pair, a closing axis, filtered against that envelope.
    jaw = GraspCalculator(
        camera_matrix=frame.intrinsics, max_candidates=8,
        min_grip_width_mm=robot.gripper.min_width_mm,
        max_grip_width_mm=robot.gripper.max_width_mm,
    ).compute(frame.segmentations[0], frame.depth_map, camera_to_base=camera_to_base,
              gripper_model=jaw_model, support_plane=support, unit="mm")
    # 5. Cup: one sealable contact and a press direction. The transform is how the wrench term
    #    knows which way gravity points, so dropping it makes every load read as compressive.
    cup = synthesize_suction_grasps(
        frame.segmentations[0], frame.depth_map, frame.intrinsics,
        camera_to_base=camera_to_base, payload_mass_g=250.0, config=SuctionConfig(max_results=5))
    print(f"{height_mm:5.0f} mm box   jaw {len(jaw)} candidate(s)   cup {len(cup)} candidate(s)")
