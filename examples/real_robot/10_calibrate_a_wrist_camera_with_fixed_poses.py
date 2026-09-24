"""Calibrate a WRIST camera (eye-in-hand) from fixed stations, each one fine-tuned by hand when you ask.

The stations are yours, run in the order written: joint angles taught on the pendant, poses written down, or the
stations file 09 wrote while you guided the arm, which replays that run without hands. Nothing generates a station.
A joint station runs as one judged joint move; the arm takes each joint the full turn nearest where it stands,
inside its cable window. adjust=True stops at every station the arm reached and frees it: move it by hand until the
window shows the board well and press Enter; it is held once it stands still and judged where you left it. Before
the arm drives on, the console asks for your hands off it (only Enter goes on, q finishes) and counts down 3, 2, 1;
q there (console or window) or ESC finishes, and Ctrl-C in the console stops the program, before anything moves.
adjust needs an arm that offers hand guiding; without adjust any arm runs the file.

Before you run it: declare the camera's body on the rig (docs/calibration-setup.md, section 4, has a camera on a
bracket beside the gripper), and fix the board where every station sees it. Then, at the cell:
    WILLY_PROFILE=<your cell> python examples/real_robot/10_calibrate_a_wrist_camera_with_fixed_poses.py
"""

from willy import HandEyeCalibration, SweepOptions, load_tree, print_sweep_progress

# A template in the file's format, four joint stations and two poses, taught on no cell: teach your
# own, or name the file 09 wrote, "calibration/real/eye_in_hand_wrist_stations.json".
STATIONS = "examples/real_robot/eih_fixed_stations.json"
# The board, as --board spells it, so camera.hand_eye needs no edit for this run: id 50 of
# DICT_4X4_100, 150 mm edge. A wrong edge scales every sample; measure the print.
TARGET = "aruco:50:150:DICT_4X4_100"

# Or a list in code: a ring of Pose.aimed_at(x, y, z, target_mm=BOARD_MM, closing_axis="tangential")
# aims the TOOL at the board, right for a camera that looks along the tool axis; the roll follows the
# base, so wrist 3 does not wind round the ring. A tilted camera is aimed by hand: 09, or adjust here.
# The body the rig declares is placed on the flange from the camera's calibration, so a first run
# cannot place it, and this says why it may run without it. preview="auto" shows the camera.
options = SweepOptions(fixed_poses=STATIONS, adjust=True, target=TARGET, preview="auto",
                       unmodelled_wrist_body="first calibration: nothing can place the camera body before it")
calibration = HandEyeCalibration.from_tree(load_tree(), rig_id="wrist", mode="eye_in_hand",
                                           options=options, on_event=print_sweep_progress)

print(calibration.check())  # the config alone: the rig, its body, the marker and the stations; opens nothing
rehearsed = calibration.run(dry_run=True)  # built and attested; an arm nobody can guide is refused here
print(rehearsed)
if not rehearsed.ok:
    raise SystemExit(rehearsed.exit_code)

# The stations in order: the arm drives to each, you adjust and press Enter, hands off, 3-2-1, next.
# The adjusted stations are written to calibration/real/eye_in_hand_wrist_stations.json for next time;
# when that is the file STATIONS names (09's), it is never written over, and they go to
# eye_in_hand_wrist_stations.adjusted.json beside it. The report names the file either way.
report = calibration.run()  # this moves the robot
print(report)  # every station's verdict, the flange to TCP it recorded, and the rig block to paste
raise SystemExit(report.exit_code)
