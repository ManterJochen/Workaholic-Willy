"""Pick a part whose pose you know and place it elsewhere, each in one call that reports every motion.

Run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/05_pick_and_place_a_known_part.py
"""

from willy import Robot, load_tree

robot = Robot.from_tree(load_tree())

# Where the grasp centre between the fingers takes the part and where it lets it go, in millimetres
# in the robot's base frame with the tool pointing down; choose both inside your cell's workspace.
# The fingers close the way your cell's hand naturally stands (robot.natural_closing_axis, base x
# where it names none); yaw_deg turns them further about the vertical, counted from that direction, to
# meet the part across its width: where the hand stands along -y, yaw_deg=90.0 closes along base x.
part = robot.tool_down(450.0, 100.0, 120.0, yaw_deg=90.0)
tray = robot.tool_down(300.0, -250.0, 140.0)
part_width_mm = 40.0

# No camera is handed in, so there is no camera world and each verb says why it moves without one.
# 15_speak_pick_and_hand_handover.py hands the robot its camera instead, and holds the located part
# out of the world it plans against.
bench = "a known part on a clear table, no camera"

with robot.connected():
    # Open, a planned move to 80 mm above the part along its approach, a straight line in, a close
    # to 1 mm under the width, and the line back out. A hand that toggles on one output with no sensor
    # is never switched before the arm moves: it asked at connect where its jaws stand, and asks again
    # here where it believes them closed. It closes with one change of its output at the part.
    picked = robot.pick(part, part_width_mm, decline=bench)
    print(picked)
    # When the hand measured its fingers close on nothing, the pick reports NOTHING_HELD: it opened
    # the hand, backed out to where the line began, and left nothing to place.
    if picked.ok:
        # The same approach to the tray, an open, and the line back out. When the hand still measures
        # the part after opening, the place reports RELEASE_NOT_CONFIRMED and the arm stays put. A toggle
        # opens with one change, and says "already open" with none where its count says they stand open.
        print(robot.place(tray, decline=bench))
