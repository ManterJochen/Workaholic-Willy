"""Rehearse one pick at a desk, from the checklist to the teardown, on a dummy arm and a synthetic scene.

A rehearsal runs the steps a pick campaign runs at a cell: preflight, build, the safety attestation, the
connect, one pick and the teardown. Nothing real moves and nothing is seen, so a pass proves the wiring.
"""

from willy import Cell, PickRun, Recording, load_tree

# The desk profile: a dummy arm with a dummy hand. The rehearsal keeps this robot section and puts
# the dummy arm under it; the scene is one synthetic box, so no camera opens and no model loads.
tree = load_tree("console_dummy")
cell = Cell.rehearsal(tree.robot)

print(cell.preflight())    # everything decidable at a desk, each finding with its fix
cell.build()
# Asked of the arm that was built: a dummy arm gates nothing and says so, so nothing here shows
# what would refuse a bad command.
print(cell.safety())

# One pick, connect to teardown, judged by the rule the report prints, and nothing recorded.
run = PickRun.from_cell(cell, runs=1, recording=Recording.off()).execute()
print(run)
if run.last is not None:
    print(run.last)        # the pick service's own account of the attempt

# At a real cell the same campaign starts from the tree WILLY_PROFILE names and from what to pick:
#     cell = Cell.from_tree(load_tree(), prompt="<what to pick>")
# There the cell's own arm, its planner and guard, its cameras and the detector take the dummy's
# place, and connecting is itself motion. examples/real_robot/12_pick_with_the_camera.py runs it.
