"""Clear a bin: the dense modes, a blocker pushed aside when a pick has nowhere to go, and what ran.

13 runs a campaign of picks and reads the verdict. This drives the pick service directly, which is
what you do when the loop is yours: pick until the bin is empty, switch target without reopening a
camera, and read off each attempt which optional layers actually ran.

The mode is chosen at BUILD. `dense_clutter` samples a bin rather than one object;
`dense_autonomous` adds a second scan at the standoff and lets the scene recovery push a blocker
out of the way. A mode whose machinery the config never switched on refuses with
`MODE_NOT_AVAILABLE` rather than running a lesser pick under its name, so `simulation/06` is worth
running first: it lists every mode and what each one needs.

Every `robot.grasping` block that the dense modes read ships `enabled: false`
(`decision`, `closed_loop`, `verification`, `recovery`, `ordering`, ...). Switch on what this cell
should have in its profile; `docs/grasping-config-reference.md` is the list. The two lines below
turn them on in memory instead, which is for trying one out, not for running a cell.

Run it at the cell, under the cell's profile, once its camera is calibrated (07-10):
    WILLY_PROFILE=<your cell> python examples/real_robot/18_clear_a_bin_with_recovery.py
"""

from pathlib import Path

from willy import AutonomousGraspOutcome, Camera, Cell, load_tree, viewing_pose

attempts = 12
# Outcomes that mean the bin is done rather than that this attempt went wrong. Anything else is
# worth another try: a refused grasp on a cluttered bin is often the next scan's easy one.
empty = {AutonomousGraspOutcome.NO_TARGET, AutonomousGraspOutcome.EXECUTION_FAILED}

# In your profile these are YAML. Here they are set on the loaded tree, so this file shows what
# the dense modes actually need rather than leaving a reader to find out from a refusal.
tree = load_tree().with_values({
    "robot.grasping.recovery.enabled": True,      # push a blocker aside when a pick has nowhere to go
    "robot.grasping.closed_loop.enabled": True,   # re-scan at the standoff before the jaws close
    "robot.grasping.verification.enabled": True,  # check afterwards that something is really held
})

cell = Cell.from_tree(tree, prompt="a part in the bin", mode="dense_autonomous")
print(cell.preflight())  # everything decidable at a desk, each finding with its fix
service = cell.build()

# One record per attempt, appended: positions, scores, layers, outcome; `RecordLog` reads it back (simulation/05).
Path("logs").mkdir(exist_ok=True)
service.enable_record_logging("logs/bin_clearing.jsonl")

with cell.connected():
    # Every scan from the viewing pose over the bin (BASE mm), the hand emptied first, as in 13; a fixed camera: none.
    view = viewing_pose(Camera.from_tree(tree), (-130.0, -700.0, 50.0), arm=cell.arm, closing_axis="-y")
    print(view)
    for attempt in range(attempts):
        if not view.ok or (view.pose is not None and not (cell.robot.release().ok and cell.robot.move(view.pose).ok)):
            print("the camera is not at its viewing pose, so nothing more is picked")
            break
        report = service.pick()
        # `layers` is read off the attempt, not off the config: a block switched on in YAML but
        # never reached does not appear. That is how you tell a wired layer from a configured one.
        print(f"--- attempt {attempt}: {report.outcome.value}, "
              f"layers {', '.join(report.layers_that_ran()) or '(none)'}")
        if report.outcome in empty:
            print(report)
            break
    else:
        print(f"{attempts} attempts and the bin still has something in it")

    # What every camera grounds; changing it reopens no camera and reloads no model: one connect, two kinds of part.
    previous = service.set_prompt("a lid")
    print(f"now looking for a lid instead of {previous}")
    if view.ok and (view.pose is None or cell.robot.release().ok and cell.robot.move(view.pose).ok):
        print(service.pick())
