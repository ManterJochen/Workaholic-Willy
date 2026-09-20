"""Connect to the cell's arm and hand, go home, and move: a move, a straight line down and a joint move.

Run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/03_connect_and_move.py
"""

from willy import Pose, Robot, load_tree

robot = Robot.from_tree(load_tree())
print(robot)             # the arm, the hand, the lock, the planner route and the camera world
print(robot.safety())    # what this arm refuses, asked of the arm that was built

# Poses in millimetres in the robot's base frame, joints in radians; choose both inside your
# cell's workspace.
above = Pose.tool_down(450.0, 100.0, 300.0)
lower = Pose.tool_down(450.0, 100.0, 200.0)

# Connecting takes the cell's lock, then the arm, then the hand; a hand may sweep its fingers as it
# activates. Every motion is planned and checked against the camera world. With no camera handed in
# there is no world, so this bench run says why it moves without one; 11_locate_and_pick.py hands
# the robot its cameras. Each verb returns a report, and a refused motion is a report too.
with robot.connected(), robot.without_camera_world("bench run, the table is clear"):
    print(robot.home())
    print(robot.move(above))
    print(robot.move(lower, linear=True))
    print(robot.move_joints([0.0, -1.57, 1.57, -1.57, -1.57, 0.0]))
