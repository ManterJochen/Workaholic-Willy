"""A campaign of picks at the cell: one connect around them all, and one verdict by a rule you
choose, with a record of every attempt, and a picture of what the camera saw and gripped.

Run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/13_pick_campaign.py
"""

from pathlib import Path

from willy import Cell, PassRule, PickRun, Recording, load_tree

# The whole cell: its cameras, perception, the grasp stack, the arm and the hand. The pick service
# hands the arm the world its calibrated cameras build and declines nothing, so where the planner
# needs a world and the cameras give it none, the picks are refused before any motion.
cell = Cell.from_tree(load_tree())
cell.build()  # explicit and idempotent, so cell.service exists before the campaign below builds it again

# Off by default and costs no render time unless enabled: the mask the segmenter found, the
# gripper's projected wireframe, the contact arrows and the score bar, one overlay per attempt.
cell.service.enable_debug_image_rendering(True)

debug_dir = Path("logs/picks_debug")
debug_dir.mkdir(parents=True, exist_ok=True)


def _save_attempt(attempt) -> None:
    print(attempt)
    # The overlay the calculator rendered for this attempt's frame; None on an attempt that never
    # reached perception (a cancelled run, a refused camera world).
    png = cell.service.last_debug_image_png
    if png:
        (debug_dir / f"attempt_{attempt.index:03d}_{attempt.outcome.value}.png").write_bytes(png)


# Four of five must succeed; the default rule is every one. Without confirm= the service's own
# word is the evidence of a success, and confirm= takes a check of your own on each attempt.
rule = PassRule(fraction=0.8)

campaign = PickRun.from_cell(
    cell,
    runs=5,
    prompt="a red cube",  # what every camera grounds, for this campaign only
    recording=Recording.to_file("picks.jsonl"),  # one attempt record per line: positions, scores, outcome
    rule=rule,
    on_attempt=_save_attempt,  # each attempt as it ends: printed, and its overlay saved to debug_dir
)

# One connect around all five picks, because connecting is itself motion: a hand may sweep its
# fingers as it activates. The cell comes down at the end whatever happened, and a refused build
# or connect is on the report rather than raised.
report = campaign.execute()
print(report)
print(f"debug overlays written to {debug_dir}/")
raise SystemExit(report.exit_code)

