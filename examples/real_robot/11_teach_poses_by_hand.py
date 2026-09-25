"""Teach joint poses by guiding the arm by hand, instead of copying numbers off the pendant.

Stand the arm by hand where a pose should be, a look pose of 12 for instance, and press Enter: one line to paste into
your program is printed,
    LOOK_1 = JointPositions.deg(-45.0, -100.2, -110.0, -60.0, 90.0, 0.0)  # rig 'wrist', eye in hand
and the pose, its TCP, its camera and the time are added to logs/taught_poses.json, beside every pose taught before.

A pose belongs to one camera: an eye-in-hand and an eye-to-hand camera are calibrated differently, so each gets poses
of its own. CAMERA names the rig these poses are taught for (None: the primary), and the line and the record say it.
For each pose the console asks for a name while the arm holds (Enter takes LOOK_1, LOOK_2 and so on, q finishes), then
frees the arm. Enter captures once the arm has stood still for half a second: the arm is held, and its joints and its
TCP are read. s skips a pose. Outside the cable window or the workspace box the console says so and Enter does not
capture there; the arm is never held or stopped for it. Nothing here moves the arm by itself: it moves only while you
move it, and q, Ctrl-C and an error alike leave it held.

With SHOW_CAMERA a window shows what that camera sees, the pose and how many are taught, red with the reason outside a
boundary: Enter or Space in it captures too, q or ESC finishes, and closing it finishes. A camera that cannot be
opened, or a desk that cannot show a window, is said in one line, and the poses are taught without it, still for that
rig. The arm has to offer hand guiding (teach mode). Run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/11_teach_poses_by_hand.py
"""

from contextlib import ExitStack

from willy import Camera, CameraRefused, HandGuidingRefused, Robot, load_tree, teach_poses

CAMERA = None  # the rig whose poses you teach, e.g. "wrist"; None: the primary
SHOW_CAMERA = True  # False: no window

tree = load_tree()
# The arm alone, as a calibration builds it: no hand is built, so no activation stroke and no question about where
# the jaws stand. The hand still hangs on the flange, and the payload question below counts it.
robot = Robot.from_config(tree.robot, gripper=None)
print(robot)
rig = CAMERA or tree.app_config.camera.cameras.primary_rig_id  # every pose taught here belongs to this rig

with ExitStack() as opened:  # gives the camera back as it ends, after teach_poses has closed its window
    camera: Camera | None = None
    if SHOW_CAMERA:
        try:  # looking through a camera needs no calibration: 06 opens one before its first
            camera = opened.enter_context(Camera.from_tree(tree, rig_id=rig))
        except (CameraRefused, OSError, RuntimeError) as unusable:  # not configured, switched off, or no device
            print(f"No camera window, the poses are still taught for rig {rig!r}: {unusable}")

    # Takes the cell's lock and connects the arm. Before the arm is first freed the console shows the payload the
    # controller compensates for and asks whether it is right (hand + camera + bracket): a wrong one makes the freed
    # arm sink or rise in your hands. Leaving the block gives the arm and the lock back.
    try:
        with robot.connected():
            taught = teach_poses(robot, tree=tree, for_rig=rig, camera=camera, store="logs/taught_poses.json")
    except HandGuidingRefused as refused:  # no hand guiding, a payload not confirmed, a file of other things
        raise SystemExit(f"REFUSED: {refused}")

# The lines of this run together, ready to paste above a list of look poses such as 12's.
for pose in taught:
    print(pose.line())
if taught:
    print(f"LOOK = [{', '.join(pose.name for pose in taught)}]  # the looks of rig {rig!r}")
