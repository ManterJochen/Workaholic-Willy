"""Open the camera the cell runs on, take one frame, and read its lens and the calibration it declares.

This works before the camera is calibrated, and that is the state to run it in first. A rig that is
not calibrated yet declares no `extrinsics` block at all: leave it out of the rig, not in with an
empty path, which the tree refuses to load. 07 to 10 calibrate the rig and print the block to paste.

Run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/06_open_a_camera.py
"""

import numpy as np

from willy import Camera, RGBDFrame, RigCalibrationError, load_tree

# The rig camera.cameras.primary_rig_id names. The block opens its device and gives it back however
# the block ends, so a grab that raises cannot leave the camera claimed for the next program.
with Camera.from_tree(load_tree()) as camera:
    print(camera)
    frame = camera.grab()
    lens = camera.get_intrinsics()  # the device's own pinhole matrix, or None where it reports none

    # The calibration the rig declares under `extrinsics`, loaded from the artifact 07-10 wrote. Any
    # calibration that does not load is reported, not raised, and the refusal says which case it is:
    # no block yet, a block written before its artifact was, or a file that is there and does not
    # parse. Each is still a camera that can be opened and grabbed from.
    try:
        print(camera.calibration())
    except RigCalibrationError as unusable:
        print(f"{camera.rig_id}: no usable calibration. {unusable}")

assert isinstance(frame, RGBDFrame)  # a camera the cell runs on gives depth, or from_tree refuses it
# Depth in uint16 millimetres, where a zero is a pixel the camera could not measure. An empty array
# means the rig's rgbd_backend cannot reach this device's depth.
depth = frame.depth
print(f"colour {frame.color.shape}, depth {depth.shape}: {np.count_nonzero(depth)} pixels measured")
if lens is not None:
    print(f"fx {lens[0, 0]:.1f}  fy {lens[1, 1]:.1f}  cx {lens[0, 2]:.1f}  cy {lens[1, 2]:.1f} pixels")
