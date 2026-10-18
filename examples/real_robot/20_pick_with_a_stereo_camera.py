"""Pick with a stereo pair as the cell's camera: the pick path an RGB-D camera takes, on the depth the pair matches.

Once a webcam pair or a side by side stereo device is calibrated (19) and placed against the robot (07 to 10), naming
it camera.cameras.primary_rig_id makes it the camera every pick perceives with, plans its grasp in and judges every
motion against. Its left image rectified is the colour the detector reads and the depth is matched on it; a pixel the
pair could not match is a hole, never a depth filled in from its neighbours or an earlier frame. Fixed in the cell it
searches its depth from where the workspace box begins, on the wrist from 150 mm, and to infinity either way.

The camera is opened once first, to show what it gives, then a campaign as in 12 picks with it.
Run it at the cell, under the cell's profile, once the pair is calibrated (19) and placed (07-10):
    WILLY_PROFILE=<your cell> python examples/real_robot/20_pick_with_a_stereo_camera.py
"""

import numpy as np

from willy import Camera, Cell, JointPositions, PassRule, PickAttempt, PickRun, RGBDFrame, Recording, load_tree

tree = load_tree()
# The primary, opened through the camera noun as the cell opens it: a pair whose stereo calibration is not written
# is refused here, before anything moves, with what to run (19).
with Camera.from_tree(tree) as camera:
    print(camera)
    print(camera.calibration())  # where the pair stands against the robot: the artifact the hand-eye sweep wrote
    frame = camera.grab()
    lens = camera.get_intrinsics()  # the rectified left lens, the one the depth lies on
assert isinstance(frame, RGBDFrame)
print(f"colour {frame.color.shape}, depth on {100.0 * np.count_nonzero(frame.depth) / frame.depth.size:.0f} % of it")
if lens is not None:
    print(f"fx {lens[0, 0]:.1f}  fy {lens[1, 1]:.1f}  cx {lens[0, 2]:.1f}  cy {lens[1, 2]:.1f} pixels")

# Where a camera on the wrist looks from, or where the arm stands clear of a fixed camera's view of the parts. Degrees,
# one per joint, as the pendant shows them: FILL THIS IN from your own cell (11 teaches it by hand).
LOOK = JointPositions.deg(-90.0, -100.0, -110.0, -60.0, 90.0, 0.0)


def _each_attempt(attempt: PickAttempt) -> None:
    print(attempt)  # the outcome, the looks tried, where the object was seen and the tool closed


# One connect around every pick; the cell comes down whatever happened, and a refused build or connect is on the
# report, not raised.
report = PickRun.from_cell(
    Cell.from_tree(tree),
    runs=5,
    prompt="a red cube",  # what the camera grounds, for this campaign only
    look=LOOK,
    put_back=True,  # a part that does not go back stops the campaign: no pick starts with a part in the hand
    recording=Recording.to_file("logs/stereo_picks.jsonl"),  # one attempt record per line
    rule=PassRule(fraction=0.8),  # four of five; with no sensor in the hand a success is the close command's word
    on_attempt=_each_attempt,
).execute()
print(report)
raise SystemExit(report.exit_code)
