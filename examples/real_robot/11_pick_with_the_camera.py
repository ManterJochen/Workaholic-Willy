"""Pick a part with the camera, many times over: your own pick motion, where the camera looks from, and the part put
back where it was grasped after every lift, so one part serves the whole campaign. One verdict by a rule you choose,
a record of every attempt, and a picture of what the camera saw and gripped.

Each pick moves the arm to the first look below and perceives there, going on to the next look when it finds nothing.
Leave look= out and a wrist camera looks from the configured home, while a fixed camera does not move to look.

Run it at the cell, under the cell's profile, once its camera is calibrated (07-10):
    WILLY_PROFILE=<your cell> python examples/real_robot/11_pick_with_the_camera.py
"""

from pathlib import Path

from willy import Cell, GraspMotion, JointPositions, PassRule, PickAttempt, PickRun, Recording, load_tree

# Where the camera looks from, tried in order. Degrees, one per joint, as the pendant shows them: FILL THESE IN from
# your own cell. Stand the arm where the camera sees the work, then read the joints off with
#     python -m src.robot.drivers.ur --where
LOOK = [
    JointPositions.deg(-90.0, -100.0, -110.0, -60.0, 90.0, 0.0),
    JointPositions.deg(-70.0, -100.0, -110.0, -60.0, 90.0, 0.0),
]

# Start 60 mm above the grasp, close 5 mm below the measured width, lift 100 mm. A field left out keeps the service's
# own, and a value that cannot be a motion raises here, before any cell exists.
motion = GraspMotion(standoff_mm=60.0, close_squeeze_mm=5.0, retreat_mm=100.0)

# The whole cell: its cameras, perception, the grasp stack, the arm and the hand. Every motion is checked against the
# world its calibrated cameras build, and a pick with no such world is refused unmoved.
cell = Cell.from_tree(load_tree(), motion=motion)
cell.build()  # explicit and idempotent, so cell.service exists before the campaign builds it again
# Off by default, no render time unless on: the mask found, the gripper's wireframe, the contacts, the score.
cell.service.enable_debug_image_rendering(True)
debug_dir = Path("logs/picks_debug")
debug_dir.mkdir(parents=True, exist_ok=True)


def _each_attempt(attempt: PickAttempt) -> None:
    print(attempt)  # the outcome, the looks tried, where the object was seen and the tool closed, the put back
    # The same as values for a program of your own, None where the attempt did not get that far: the object's seen
    # centre in BASE millimetres, and the whole pose the tool closed at, its orientation included.
    print(f"    object_mm={attempt.object_mm}\n    grasp_pose={attempt.grasp_pose}")
    png = cell.service.last_debug_image_png  # None where the attempt never reached perception
    if png:
        (debug_dir / f"attempt_{attempt.index:03d}_{attempt.outcome.value}.png").write_bytes(png)


campaign = PickRun.from_cell(
    cell,
    runs=5,
    prompt="a red cube",  # what every camera grounds, for this campaign only
    look=LOOK,
    put_back=True,  # a part that does not go back stops the campaign: no pick starts with a part in the hand
    recording=Recording.to_file("logs/picks.jsonl"),  # one attempt record per line: positions, scores, outcome
    # Four of five must succeed; the default rule is every one. Without a sensor in the hand a success is the close
    # command's word, and the report says so; confirm= takes a check of your own on each attempt.
    rule=PassRule(fraction=0.8),
    on_attempt=_each_attempt,
)

# One connect around every pick, because connecting is itself motion. The cell comes down whatever happened, and a
# refused build or connect is on the report, not raised.
report = campaign.execute()
print(report)
print(f"debug overlays written to {debug_dir}/")
raise SystemExit(report.exit_code)
