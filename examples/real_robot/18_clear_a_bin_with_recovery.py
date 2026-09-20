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

from willy import AutonomousGraspOutcome, Cell, load_tree

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

# One record per attempt, appended: positions, scores, the layers, the outcome. `RecordLog` reads
# the file back into KPIs afterwards, as simulation/05 shows.
Path("logs").mkdir(exist_ok=True)
service.enable_record_logging("logs/bin_clearing.jsonl")

with cell.connected():
    for attempt in range(attempts):
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

    # The prompt is what every camera grounds, and changing it reopens no camera and reloads no
    # model. This is how one connect clears a bin of one kind of part and then another.
    previous = service.set_prompt("a lid")
    print(f"now looking for a lid instead of {previous}")
    print(service.pick())
