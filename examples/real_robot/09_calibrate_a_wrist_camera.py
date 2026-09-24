"""Calibrate a camera on the arm's wrist (its CAMERA->TOOL) by guiding the arm by hand over a board.

Fix a printed board flat on the table within reach. Then move the arm by hand, pose after pose, so the camera sees
the board from a new distance and angle each time, and press Enter (console, or Enter or Space in the preview): the
arm is held once it stands still, one frame is judged, and it is freed again. It works for a camera wherever it sits
on the tool, tilted or beside the hand, because you aim it by eye. s skips, q finishes. A pose outside the cable
window or the workspace box turns the window red and is not captured; the arm is never held for it. The arm has to
offer hand guiding; 10 runs fixed stations instead. Run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/09_calibrate_a_wrist_camera.py
"""

from willy import HandEyeCalibration, SweepOptions, load_tree, print_sweep_progress

tree = load_tree()

# The solve is CAMERA->TOOL against the TCP the arm reports, which is the flange times the cell's
# tool frame. So the artifact records that flange to TCP, and the rig is refused later once the
# tool frame moves away from it: measure robot.gripper.tool_frame before you start.
print("tool frame source:", tree.robot.gripper.tool_frame.source)

# The camera's own body is placed on the flange from its calibration, so a first run cannot place it,
# and on a cell that plans with geometry it is refused until you say why it may run without it. The
# board is camera.hand_eye.eye_in_hand's; SweepOptions(marker_length_mm=...) overrides it for one run.
# preview="auto": the camera, whether it sees the board, how far and how tilted, and the count.
options = SweepOptions(
    freedrive=True,
    unmodelled_wrist_body="first calibration: nothing can place the camera body before it",
    preview="auto",
)
calibration = HandEyeCalibration.from_tree(tree, rig_id="wrist", mode="eye_in_hand",
                                           options=options, on_event=print_sweep_progress)

print(calibration.check())  # the config alone: the rig, its body and the marker; opens nothing
rehearsed = calibration.run(dry_run=True)  # built and attested, the body placed or declined
print(rehearsed)
if not rehearsed.ok:
    raise SystemExit(rehearsed.exit_code)

# The payload the controller compensates for is shown and asked about before the arm is first freed.
# Every pose you count is written to calibration/real/eye_in_hand_wrist_stations.json: 10 replays it
# without hands, and with adjust you fine-tune each one again next time. That replay never writes over
# the file it reads: its adjusted stations go to eye_in_hand_wrist_stations.adjusted.json beside it.
report = calibration.run()
print(report)  # the flange to TCP it recorded, the stations file, and the rig block with its tolerances
raise SystemExit(report.exit_code)
