"""Calibrate a stereo pair, a webcam pair or a side by side stereo device, so that it gives depth as an RGB-D camera.

Print the ChArUco board camera.cameras.stereo_calibration describes, and measure it: its square sets the scale of the
baseline, and so of every depth. A window opens per eye: hold the board still where both cameras see it, from a new
distance and angle each time, and press s to keep the pair; ESC ends (the rig's min_pairs are needed, its max_pairs
end the session). The pairs are solved and written to the rig's calibration_paths.stereo_map_file, where its camera
reads them as it opens: from then on it hands out its left image rectified and the depth on it.

Nothing here places the pair in the cell. Calibrate it against the robot as any camera (07 to 10: the board is posed
on its left image rectified), paste the extrinsics block the sweep prints, and name the rig
camera.cameras.primary_rig_id; 20 picks with it. Run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/19_calibrate_a_stereo_pair.py
"""

import numpy as np

from willy import Camera, RGBDFrame, StereoCalibration, load_tree

RIG = "webcam_main"  # the rig_id your camera section gives the pair: source webcam_pair or single_device
RECORD = True  # False: solve the pairs already on disk

tree = load_tree()
calibration = StereoCalibration.from_tree(tree, rig_id=RIG)
print(calibration)  # the rig, the board, the pairs on disk and the file the calibration goes to

if RECORD:
    print(f"{calibration.record()} pairs on disk")  # s keeps the pair on screen, ESC ends

# Solved and written only where the pair's residual is 2 px or less. The report says what it rests on: the pairs
# solved with, the residual and its band, the baseline and focal length, and the depths the pair is searched at.
report = calibration.calibrate()
print(report)
if not report.ok:
    raise SystemExit(report.exit_code)

# The pair is a depth camera now, opened as the cell opens it: the colour its left image rectified, the depth in
# uint16 millimetres on it, 0 where the pair could not be matched (never a depth filled in from its neighbours).
with Camera.from_tree(tree, rig_id=RIG) as camera:
    frame = camera.grab()
    lens = camera.get_intrinsics()  # the rectified left lens, the one the depth lies on; its distortion is zero
assert isinstance(frame, RGBDFrame)
print(f"depth on {100.0 * np.count_nonzero(frame.depth) / frame.depth.size:.0f} % of the frame")
if lens is not None:
    print(f"fx {lens[0, 0]:.1f}  fy {lens[1, 1]:.1f}  cx {lens[0, 2]:.1f}  cy {lens[1, 2]:.1f} pixels")
