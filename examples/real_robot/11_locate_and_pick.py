"""See a part and pick it: the camera places it in the robot's base frame, and the arm plans
around what the camera sees.

A wrist camera sees what the arm points it at, so the arm first moves to a viewing pose: the calibrated camera
DISTANCE_MM from WORK_MM, looking straight at it, the tool down with CLOSING_AXIS and tilted no more than aiming needs.
A fixed camera sees the work from where it is mounted and needs none.

Run it at the cell, under the cell's profile, once its camera is calibrated (07-10):
    WILLY_PROFILE=<your cell> python examples/real_robot/11_locate_and_pick.py
"""

from willy import Camera, Locator, Robot, load_tree, viewing_pose

# Where the parts lie, in BASE millimetres: a point on the table to look at. Measure yours; these are the table
# under the marker example 10 calibrated with.
WORK_MM = (-130.0, -700.0, 50.0)
# How far the camera stands from it. The viewing pose refuses a distance nearer than the camera's depth measures
# from (Intel gives some models about 450 mm at a 1280 x 720 depth mode, 310 mm at 848 x 480).
DISTANCE_MM = 500.0
# The heading the tool keeps, as Pose.tool_down takes it: the one the camera was calibrated with.
CLOSING_AXIS = "-y"

tree = load_tree()

# The tree's primary camera, open for the block and given back at its end.
with Camera.from_tree(tree) as camera:
    # The robot on that camera, handed the live world the camera builds; the print says how the
    # arm's motions are planned and which world they are checked against.
    robot = Robot.from_tree(tree, cameras=[camera])
    print(robot)
    # Grounds a prompt in one frame and places every object in BASE; the models load here. A wrist
    # camera is placed from the tool pose the arm reports at the shutter; a fixed one ignores it.
    locator = Locator.from_tree(tree, camera=camera, tool_pose=robot.arm.get_tcp_pose)

    # No decline: the world is wired, and an arm whose world is wired refuses a declined motion.
    with robot.connected() as live:
        # This example picks and never places, so the last run may have ended holding its part. The hand opens where
        # that run left the arm, as 13 does, and the part falls back onto the work instead of riding to the view.
        released = robot.release()
        print(released)
        # Screened against the arm's box, reach and joint window; the arm judges the move itself.
        view = viewing_pose(camera, WORK_MM, arm=robot.arm, distance_mm=DISTANCE_MM, closing_axis=CLOSING_AXIS)
        print(view)
        moved = robot.move(view.pose) if released.ok and view.pose is not None else None
        if moved is not None:
            print(moved)
        # Nothing is located from wherever the arm happens to stand: not by a wrist camera with no viewing pose, nor
        # by one the arm did not take there, nor with a hand that did not open.
        there = released.ok and view.ok and (moved is None or moved.ok)
        located = locator.locate("a red cube") if there else None
        print(located if located is not None else "nothing located: the camera is not where it looks from")
        if located is not None and located.objects:
            # Object 0 as a grasp scene, every other object an obstacle; grasps best first.
            grasps = located.scene(0, tree.robot).grasps()
            print(grasps)
            best = grasps.best
            if best is not None:
                # keep_out leaves the part itself out of the world, so the fingers may reach it.
                print(robot.pick(best.pose(), best.grip_width_mm, keep_out=located.keep_out(0)))
    print(live.teardown)
