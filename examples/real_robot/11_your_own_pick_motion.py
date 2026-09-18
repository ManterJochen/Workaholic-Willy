"""Choose how the cell's picks move (the approach, the close and the lift) and keep every guard the
pick service puts around the motion.

Run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/11_your_own_pick_motion.py
"""

from willy import Cell, GraspMotion, PickRun, Recording, load_tree

# Start 60 mm above the grasp, close 5 mm below the measured width, lift 100 mm after the close.
# A field left out keeps the service's own, and a value that cannot be a motion (a negative
# standoff, a close speed above 1) raises here, before any cell exists.
motion = GraspMotion(standoff_mm=60.0, close_squeeze_mm=5.0, retreat_mm=100.0)
print(motion.to_dict())

# The pick service builds the one policy it drives from this motion, on the arm and the hand it
# built, and adds its guards: a grasp not in the base frame is refused, the dwell gate the tree
# asks for holds each move until the arm stands still, and the jaws open before every approach.
# The service's older policy= door takes a policy built by hand, and a policy drives the arm and
# hand it was built with: one built around another arm or hand would move a robot the service
# never checked, with none of those guards, so the service refuses it. motion= is the way in.
cell = Cell.from_tree(load_tree(), motion=motion)

# The pick service hands the arm the world its calibrated cameras build and declines nothing. A
# motion this hand cannot do (jaws opened wider than they go) is refused at the build, on the report.
report = PickRun.from_cell(cell, runs=1, prompt="a red cube", recording=Recording.off()).execute()
print(report)
raise SystemExit(report.exit_code)
