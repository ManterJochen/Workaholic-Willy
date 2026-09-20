"""Move the arm against a live camera world and place a part the hand already holds: no pick.

The camera builds the arm's live planner world, so every move and the place below are checked against
what the camera actually sees, with no `decline` needed. This is the other half of `pick`: a part that
got into the hand some other way (handed in, grasped by hand as in 04, carried over from another step)
and now needs a planned, camera-checked line to a place pose.

Run it at the cell, under the cell's profile, once its camera is calibrated (07-10):
    WILLY_PROFILE=<your cell> python examples/real_robot/12_place_with_a_camera_world.py
"""

from willy import Camera, Pose, Robot, load_tree

tree = load_tree()

with Camera.from_tree(tree) as camera:
    # Handing the camera in is what gives the arm a live world; the print says so.
    robot = Robot.from_tree(tree, cameras=[camera])
    print(robot)

    part_width_mm = 40.0
    waypoint = Pose.tool_down(400.0, 0.0, 350.0)
    tray = Pose.tool_down(300.0, -250.0, 140.0)

    with robot.connected():
        # The part is already between the fingers; grasp is the same verb 04 uses on its own.
        # No pick: no planned descent onto a pose, just the close.
        held = robot.grasp(part_width_mm - 1.0)
        print(held)
        if not held.ok:
            raise SystemExit("nothing measured between the fingers; check the part is in place")

        # No `decline`: the camera world is wired, so this move is checked against what the camera
        # sees right now, and refuses rather than run blind if the world drops out.
        print(robot.move(waypoint))
        # A planned move to the standoff, a line in, release, a line back out, all camera-checked.
        print(robot.place(tray))
