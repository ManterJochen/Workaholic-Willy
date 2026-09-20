"""See a part and pick it: the camera places it in the robot's base frame, and the arm plans
around what the camera sees.

Run it at the cell, under the cell's profile, once its camera is calibrated (07-10):
    WILLY_PROFILE=<your cell> python examples/real_robot/11_locate_and_pick.py
"""

from willy import Camera, Locator, Robot, load_tree

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
        located = locator.locate("a red cube")
        print(located)
        if located.objects:
            # Object 0 as a grasp scene, every other object an obstacle; grasps best first.
            grasps = located.scene(0, tree.robot).grasps()
            print(grasps)
            best = grasps.best
            if best is not None:
                # keep_out leaves the part itself out of the world, so the fingers may reach it.
                print(robot.pick(best.pose(), best.grip_width_mm, keep_out=located.keep_out(0)))
    print(live.teardown)
