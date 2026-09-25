"""The real-robot calls on a dummy arm and hand: connect, move, close and open the hand, pick and place.

Nothing real moves. The same calls drive a real cell in examples/real_robot, under the profile of that cell.
"""

from willy import Pose, Robot, load_tree

robot = Robot.from_tree(load_tree("console_dummy"))
print(robot)             # the arm, the hand, the lock, the planner route and the camera world
print(robot.safety())    # a dummy arm gates nothing, and says so

# Poses in millimetres in the robot's base frame; at a cell, choose them inside its workspace.
above = Pose.tool_down(450.0, 100.0, 300.0)
part = Pose.tool_down(450.0, 100.0, 120.0)
tray = Pose.tool_down(300.0, -250.0, 140.0, yaw_deg=90.0)

# A desk has no camera, so every motion declines the camera world and says why. At a cell,
# examples/real_robot/15_speak_pick_and_hand_handover.py hands the robot its cameras instead.
with robot.connected() as live, robot.without_camera_world("desk run, a dummy arm and no camera"):
    print(robot.home())
    print(robot.move(above))

    print(robot.grasp(40.0))                 # close to 40 mm and read what the hand measured
    print("holding:", robot.is_holding())    # the dummy hand measures nothing, and says so
    print(robot.release())

    # A pick opens the hand, moves to a standoff above the part, goes straight in, closes to
    # 40 mm less a squeeze and backs out; a place is the same with a release.
    picked = robot.pick(part, 40.0)
    print(picked)
    if picked.ok:
        print(robot.place(tray))

print(live.teardown)     # the hand first, then the arm, then the lock
