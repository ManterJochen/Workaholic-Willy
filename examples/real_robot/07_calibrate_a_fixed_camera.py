"""Calibrate a camera fixed in the cell (its CAMERA->BASE) by guiding the arm by hand: nothing moves by itself.

Bolt a printed board to the tool flange. The console and the preview window call you to move the arm by hand to a
pose where the camera sees the board; press Enter (in the console, or Enter or Space in the window), and once the
arm has stood still for half a second it is held, one frame is judged, and it is freed again. s skips, q finishes.
A pose outside the cable window or the workspace box turns the window red and Enter does not capture there; the arm
is never held or stopped for it. The arm has to offer hand guiding (teach mode); one that does not is refused at the
build, and 08 calibrates the same camera from fixed poses instead. Run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/07_calibrate_a_fixed_camera.py
"""

from willy import HandEyeCalibration, SweepOptions, load_tree, print_sweep_progress

# The rig_id your camera section gives this camera; the artifact is keyed by it. The target is
# camera.hand_eye.eye_to_hand's: its marker_length_mm, aruco_dict_name and marker_id, or the ChArUco
# board its `target` names. Measure the printed edge: a wrong length scales every translation and the
# solve still converges, so no residual will show it.
# samples= is how many counted poses end the run (robot.calibration.freedrive_samples, 15, unless you
# say); q finishes earlier. preview="auto" opens the window where one can show: the camera, whether it
# sees the board, how far and how tilted, whether the pose is new enough to count, and the count.
options = SweepOptions(freedrive=True, samples=15, preview="auto")
calibration = HandEyeCalibration.from_tree(
    load_tree(), rig_id="overhead", mode="eye_to_hand", options=options,
    on_event=print_sweep_progress,  # a line per capture: what the camera saw, counted or why not
)

# The config alone: the rig, the target and how many poses. Opens nothing.
print(calibration.check())

# Builds the arm alone and opens the camera, says what the arm will refuse, and stops before anything
# is freed. A refusal here is a report, and the run below would be refused the same way.
rehearsed = calibration.run(dry_run=True)
print(rehearsed)
if not rehearsed.ok:
    raise SystemExit(rehearsed.exit_code)

# Takes the cell lock and connects the arm alone. Before the arm is first freed the console shows the
# payload the controller compensates for and asks whether it is right (hand + camera + bracket): a
# wrong one makes the freed arm sink or rise in your hands. Every pose you count is also written to
# calibration/real/eye_to_hand_overhead_stations.json, which 08 replays without hands.
report = calibration.run()
print(report)  # every capture's verdict, the stations file, and the rig block to paste under camera.cameras.rigs
raise SystemExit(report.exit_code)
