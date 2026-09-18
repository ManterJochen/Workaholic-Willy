"""Ranked grasps for a segmented point cloud, with no robot, no camera and no config tree.

A calibrated camera hands over an object's points in the robot's base frame, in millimetres, with
the height of the surface it rests on. This file draws such a cloud itself: the top face of a box.
"""

import numpy as np

from willy import Scene

# A box 40 by 30 mm and 80 mm tall, standing on a table at z = 0, centred at x 450, y 100. Its top
# face is sampled every 2 mm, which is what a camera straight above it sees.
x, y = np.meshgrid(np.arange(-20.0, 20.1, 2.0), np.arange(-15.0, 15.1, 2.0))
cloud = np.column_stack((x.ravel() + 450.0, y.ravel() + 100.0, np.full(x.size, 80.0)))

# Base frame and millimetres, unchecked: a camera-frame cloud gives confident nonsense, not an
# error. With no config there is no generator to select, so these come from the analytic
# generator, planned for the library's default jaw, which opens 85 mm; Scene.from_robot_config
# plans for the jaw a cell's tree names.
grasps = Scene.from_cloud(cloud, support_height_mm=0.0).grasps()
print(grasps)

# An empty result is an answer: the cloud was too sparse, or every grasp would go under the table.
best = grasps.best
if best is not None:
    # What Robot.pick takes, robot.pick(best.pose(), best.grip_width_mm): a base-frame pose whose
    # +Z is the approach and whose +X is the closing axis, and the width the object measures.
    print(best.pose())
    print(f"grip width {best.grip_width_mm:.1f} mm")
