"""Calibrate a WRIST camera (eye-in-hand) with every station aimed at one marker, the camera aimed and not the tool.

09 sweeps generated tool-down poses. A wrist camera tilted away from the tool axis, on a bracket beside the hand, does
not look where the tool points, so from most of them it photographs the table beside the marker and counts nothing.

This sweep aims the CAMERA. It looks once from where the arm stands, estimates from that one view where the camera
sits on the tool, and visits seventeen views round the marker from DISTANCE_MM, each tilting the tool as little as it
can from tool-down with CLOSING_AXIS and rolling the camera about its line of sight, so the solve sees it turn about
every axis; the order keeps joint travel short, and a view outside the workspace box, the arm's reach or its joint
window is reported and never moved to. Every view that sees the marker refines the estimate and re-aims the views
still to come. Each station is still a pose the arm plans and judges as it moves.

Before you run it: declare the camera's body on the rig, which a cuRobo cell refuses to sweep without
(docs/calibration-setup.md, section 4, has a D415 on a bracket beside the gripper). Lay the printed marker flat, face
up, and measure its centre in the base frame (a few centimetres is enough to aim; the solve measures nothing from it).
Jog the arm until the camera sees the marker, about half a metre away and from one side rather than from straight
above; to watch the camera while you jog, build the cell in the operator console (python -m api) and stop it before
this runs. A first look that sees no marker moves nothing and says so. Then, at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/10_calibrate_a_wrist_camera_with_fixed_poses.py
"""

from willy import HandEyeCalibration, MarkerAim, SweepOptions, load_tree, print_sweep_progress

# The marker, in BASE millimetres: its centre, lying flat and face up.
MARKER_MM = (-130.0, -700.0, 50.0)
# Its printed edge, dictionary and id, as --board spells them, so camera.hand_eye needs no edit for this run:
# id 50 of DICT_4X4_100, 150 mm edge. A wrong edge scales every sample; measure the print.
TARGET = "aruco:50:150:DICT_4X4_100"
# How far the camera stands from the marker at every view.
DISTANCE_MM = 500.0
# The heading the stations start from, as Pose.tool_down takes it: the tool's +X, the way the jaws close, along base
# -Y. Where it admits fewer views than the solve needs (a camera tilted the other way on the tool stands every view
# outside the box), the sweep takes the one of -y, y, x, -x that admits the most and its "heading" line says which.
CLOSING_AXIS = "-y"

aim = MarkerAim(marker_mm=MARKER_MM, distance_mm=DISTANCE_MM, closing_axis=CLOSING_AXIS)
# A first look that sees no marker is refused, because which way the camera is tilted on the tool is not guessed. To
# aim from a mount you state until a station sees the marker, add, for a camera tilted 45 deg toward the tool's +X
# and standing, by a rule, 60 mm out along it and 130 mm behind the TCP ("-x", "+y", "-y" for the other sides):
#     mount_if_unseen=nominal_camera_in_tool(45.0, toward="+x", offset_mm=(60.0, 0.0, -130.0))
# Or visit stations of your own instead of aim=: poses, or the joint angles you taught on the pendant,
#     SweepOptions(fixed_poses="examples/real_robot/eih_fixed_stations.json", ...)
# or a ring of Pose.aimed_at(x, y, z, target_mm=MARKER_MM, closing_axis="tangential"), which aims the TOOL: right
# for a camera that looks along the tool axis, and the roll follows the base so wrist 3 does not wind round the ring.
# The body the rig declares is placed on the flange from the camera's calibration, so a first sweep cannot place it,
# and this says why it may run without it; a rig that declares no body is refused on a cuRobo cell whatever this says.
# preview="auto" shows the camera and each judged frame in a window, where one can show.
options = SweepOptions(target=TARGET, aim=aim, preview="auto",
                       unmodelled_wrist_body="first calibration: nothing can place the camera body before it")
calibration = HandEyeCalibration.from_tree(load_tree(), rig_id="wrist", mode="eye_in_hand",
                                           options=options, on_event=print_sweep_progress)

print(calibration.check())  # the config alone: the rig, its body, the marker and the aim; opens nothing
rehearsed = calibration.run(dry_run=True)  # built and attested, the body placed or declined
print(rehearsed)
if not rehearsed.ok:
    raise SystemExit(rehearsed.exit_code)

# The first look, then the aimed views in order. Each prints as it runs, and the report lists every view: counted, or
# why not, the estimate the stations were aimed from and the heading they kept.
report = calibration.run()  # this moves the robot
print(report)  # the flange to TCP it recorded, and the rig block with the tolerances to measure
raise SystemExit(report.exit_code)
