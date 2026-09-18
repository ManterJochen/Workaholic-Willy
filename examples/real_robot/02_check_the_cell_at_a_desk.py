"""Check a cell at a desk: the checklist a first connect has to pass, and whether its planner starts.

Neither step opens a controller link or a camera, so it runs at a desk, on the computer that plans
the cell's motions:
    WILLY_PROFILE=<your cell> python examples/real_robot/02_check_the_cell_at_a_desk.py
"""

from willy import Cell, load_tree

cell = Cell.from_tree(load_tree())

# Every stop-the-cell condition the config can decide, each with its fix. BLOCK rows stop a first
# connect and WARN rows never do. BENCH rows are questions no software can answer, such as the
# controller's mode or a cable, and they stay until someone at the cell confirms them.
checklist = cell.preflight()
print(checklist)

# The planner every motion of this cell goes through, started on this machine from config and
# stopped again. It builds the arm alone and asks no controller, so a combination that cannot plan
# is refused here, saying why, rather than at the cell with the arm waiting.
planner = cell.start_planner()
print(planner)

# A script that gates a cell on this check exits with the reports' codes: 0 when nothing blocks.
raise SystemExit(checklist.exit_code or planner.exit_code)
