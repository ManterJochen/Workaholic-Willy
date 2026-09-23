"""A campaign of picks at the cell: one connect around them all, and one verdict by a rule you
choose, with a record of every attempt, and a picture of what the camera saw and gripped.

A wrist camera sees what the arm points it at, and a pick ends above its own grasp, so before every pick the arm
moves to a viewing pose over the work, as 11 does; a fixed camera needs none.

Run it at the cell, under the cell's profile:
    WILLY_PROFILE=<your cell> python examples/real_robot/13_pick_campaign.py
"""

from pathlib import Path

from willy import Camera, Cell, PassRule, PickRun, Recording, load_tree, viewing_pose

# Where the parts lie (BASE mm) and the heading the tool keeps, as in 11; the camera stands 500 mm off, the default.
WORK_MM, CLOSING_AXIS = (-130.0, -700.0, 50.0), "-y"

# The whole cell: its cameras, perception, the grasp stack, the arm and the hand. The pick service hands the arm the
# world its calibrated cameras build and declines nothing, so without that world the picks are refused unmoved.
tree = load_tree()
cell = Cell.from_tree(tree)
cell.build()  # explicit and idempotent, so cell.service exists before the campaign below builds it again

# Off by default, no render time unless on: the mask found, the gripper's wireframe, the contacts, the score.
cell.service.enable_debug_image_rendering(True)

debug_dir = Path("logs/picks_debug")
debug_dir.mkdir(parents=True, exist_ok=True)


def _save_attempt(attempt) -> None:
    print(attempt)
    # The overlay the calculator rendered for this attempt's frame; None where it never reached perception.
    png = cell.service.last_debug_image_png
    if png:
        (debug_dir / f"attempt_{attempt.index:03d}_{attempt.outcome.value}.png").write_bytes(png)


def _look_first() -> bool:
    """Before every pick; True stops the campaign, so no pick perceives from where the last one left the arm."""
    view = viewing_pose(Camera.from_tree(tree), WORK_MM, arm=cell.arm, closing_axis=CLOSING_AXIS)
    print(view)
    if view.pose is None:  # a fixed camera needs none; a wrist camera with none stops the campaign
        return not view.ok
    # The service places nothing: the hand opens where the last pick left it, as the next pick's pre-open would,
    # so the part falls back onto the work instead of riding to the viewing pose.
    released = cell.robot.release()
    moved = cell.robot.move(view.pose) if released.ok else None
    print(released, moved if moved is not None else "not moved: the hand did not open", sep="\n")
    return moved is None or not moved.ok


campaign = PickRun.from_cell(
    cell,
    runs=5,
    prompt="a red cube",  # what every camera grounds, for this campaign only
    recording=Recording.to_file("picks.jsonl"),  # one attempt record per line: positions, scores, outcome
    # Four of five must succeed; the default rule is every one. Without confirm= the service's own
    # word is the evidence of a success, and confirm= takes a check of your own on each attempt.
    rule=PassRule(fraction=0.8),
    should_cancel=_look_first,  # before each attempt: the viewing pose, or the campaign stops
    on_attempt=_save_attempt,  # each attempt as it ends: printed, and its overlay saved to debug_dir
)

# One connect around all five picks, because connecting is itself motion: a hand may sweep its fingers as it
# activates. The cell comes down whatever happened, and a refused build or connect is on the report, not raised.
report = campaign.execute()
print(report)
print(f"debug overlays written to {debug_dir}/")
raise SystemExit(report.exit_code)
