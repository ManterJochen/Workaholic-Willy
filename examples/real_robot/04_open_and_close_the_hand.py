"""Open the hand, close it on a part you place between the fingers, and read what the hand measured.

Run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/04_open_and_close_the_hand.py
"""

from willy import HoldEvidence, Robot, load_tree

robot = Robot.from_tree(load_tree())
part_width_mm = 40.0  # measure the part you will hold; the close aims 1 mm under it

# Nothing here moves the arm, so no camera world is asked for. The fingers move: keep hands clear.
# Connecting takes the lock, the arm, then the hand, and a hand may sweep its fingers as it activates.
with robot.connected():
    print(robot.release())  # open to the hand's full width
    input("Place the part between the open fingers, step back, and press Enter: ")
    print(robot.grasp(part_width_mm - 1.0))
    hold = robot.is_holding()  # reads the hand again and commands nothing
    print(f"is_holding  {hold.value.upper()}")
    print(robot.release())

# A hold reads what the hand measured, never the command echoed back:
#   HELD        the fingers stopped on a part, or a part or vacuum sensor says one is there
#   EMPTY       the fingers reached the width they were sent to, so nothing is between them
#   UNMEASURED  nothing could say: no sensor, fingers still travelling, or a driver that cannot tell
# A close that measured EMPTY reports NOTHING_HELD. A release that still measures a part reports
# RELEASE_NOT_CONFIRMED, and the arm keeps modelling the part it carries.
if hold is HoldEvidence.UNMEASURED:
    print("this hand measures no hold: GRASPED above is what was commanded, not what was felt")
