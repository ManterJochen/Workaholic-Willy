"""A campaign of picks at the cell: one connect around them all, and one verdict by a rule you
choose, with a record of every attempt.

Run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/10_pick_campaign.py
"""

from willy import Cell, PassRule, PickRun, Recording, load_tree

# The whole cell: its cameras, perception, the grasp stack, the arm and the hand. The pick service
# hands the arm the world its calibrated cameras build and declines nothing, so where the planner
# needs a world and the cameras give it none, the picks are refused before any motion.
cell = Cell.from_tree(load_tree())

# Four of five must succeed; the default rule is every one. Without confirm= the service's own
# word is the evidence of a success, and confirm= takes a check of your own on each attempt.
rule = PassRule(fraction=0.8)

campaign = PickRun.from_cell(
    cell,
    runs=5,
    prompt="a red cube",  # what every camera grounds, for this campaign only
    recording=Recording.to_file("picks.jsonl"),  # one attempt record per line
    rule=rule,
    on_attempt=print,  # each attempt as it ends, before the next one starts
)

# One connect around all five picks, because connecting is itself motion: a hand may sweep its
# fingers as it activates. The cell comes down at the end whatever happened, and a refused build
# or connect is on the report rather than raised.
report = campaign.execute()
print(report)
raise SystemExit(report.exit_code)
