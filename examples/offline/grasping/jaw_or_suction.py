"""A parallel jaw and a suction cup ask different questions of one box, so each gives up elsewhere.

The jaw needs two faces its fingers can reach and an opening wide enough; the cup needs one face it
seals on. Three boxes seen straight down by a camera 800 mm above the table show where each stops.
"""

import numpy as np

from willy import Scene, load_tree, synthesize_suction_grasps

# The base tree: the jaw opening and finger geometry every profile starts from.
tree = load_tree(None)
K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])  # intrinsics, pixels

for width_mm, height_mm in ((40.0, 60.0), (40.0, 20.0), (120.0, 60.0)):
    # The depth image of a square box on the table, and the mask a segmenter would cut for it.
    half_px = round(width_mm / 2.0 * K[0, 0] / (800.0 - height_mm))
    mask = np.zeros((480, 640), dtype=bool)
    mask[240 - half_px:240 + half_px, 320 - half_px:320 + half_px] = True
    depth = np.where(mask, 800.0 - height_mm, 800.0)

    # The jaw plans on the box's points in the base frame: the masked pixels, back-projected.
    v, u = np.nonzero(mask)
    z = depth[v, u]
    cloud = np.column_stack(((u - K[0, 2]) * z / K[0, 0], -(v - K[1, 2]) * z / K[1, 1], 800.0 - z))
    jaw = Scene.from_robot_config(tree.robot, cloud).grasps()

    # The cup is scored on the depth image itself: where its rim seals on the face it lands on.
    cup = synthesize_suction_grasps(mask, depth, K)
    print(f"box {width_mm:.0f} mm wide, {height_mm:.0f} mm tall: "
          f"jaw {len(jaw.candidates)} grasp(s), cup {len(cup)} grasp(s)")

# The low box leaves the fingers no room above the table, and the wide one does not fit the
# opening; both still offer the cup a flat face.
