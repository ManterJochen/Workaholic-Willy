"""Pick with two or more fixed cameras fused: every camera that can identify an object adds its surface to one cloud
in the robot's base frame, and the grasp is planned on that cloud instead of on the one side a single camera sees. A
campaign as in 12, and every attempt says which cameras its grasp was planned from. With SHOW_CAMERAS every camera
shows live in a window of its own, each attempt's grasp overlay pinned on the primary camera's.

Measured in simulation on the datagen reference: the best ranked grasp is graspable for 43.5 % of objects from one
view and 55.9 % fused. Never run on hardware: no physical multi-camera cell has been built with this code.

The tree is asked first, before a camera opens or a model loads. Fusing needs robot.grasping.fusion.enabled (the
other cameras' CAMERA to BASE are built under it) and fusion.geometry.enabled, two or more rigs in fusion.cameras with
the primary among them, and each an enabled RGB-D rig that declares its calibration. A cell short of any of it is
refused in one sentence; config/robot/robot.eth2.yaml with config/camera/cam.eth2.yaml is the reference cell. A
calibrated wrist camera counts as one of the views too, and then the look below is where it looks from.

Run it at the cell, under the cell's profile, once every camera is calibrated (07-10):
    WILLY_PROFILE=<your cell> python examples/real_robot/17_pick_with_fused_cameras.py
"""

from willy import CameraFusionPlan, Cell, JointPositions, LiveView, PassRule, PickAttempt, PickRun, Recording, load_tree

SHOW_CAMERAS = True  # False: no camera windows

tree = load_tree()
# Which rigs this cell fuses, or everything it is missing to fuse any: read from the tree, nothing opened.
plan = CameraFusionPlan.from_tree(tree)
if plan.refusal() is not None:
    raise SystemExit(plan.refusal())
print(plan)  # the rigs fused, the primary first: the grasp is synthesised in its view, the others fused onto it

# Where the arm stands while the cameras look: clear of every camera's view of the parts, or it hides the side it
# stands over. A fixed camera does not move, so a pick with no look perceives with the arm wherever it stopped, above
# the part it just put back. Degrees, one per joint, as the pendant shows them: FILL THESE IN from your own cell.
CLEAR = JointPositions.deg(-90.0, -100.0, -110.0, -60.0, 90.0, 0.0)


def _each_attempt(attempt: PickAttempt) -> None:
    print(attempt)  # the outcome, where the object was seen and the tool closed, and the cameras fused for it
    # The same as values for a program of your own: the rigs behind the cloud this grasp was planned on, empty where it
    # came from one camera, and how many objects of that frame a second camera added surface to.
    print(f"    fused_views={attempt.fused_views} fused_objects={attempt.fused_objects}")


# One connect around every pick; the cell comes down whatever happened, and a refused build is on the report. The
# camera windows close as the block ends.
with LiveView(show=SHOW_CAMERAS) as view:
    report = PickRun.from_cell(
        Cell.from_tree(tree),
        runs=5,
        prompt="a red cube",  # what every camera grounds, the primary and every fused one
        look=CLEAR,
        put_back=True,  # a part that does not go back stops the campaign: no pick starts with a part in the hand
        recording=Recording.to_file("logs/fused_picks.jsonl"),  # one attempt record per line
        rule=PassRule(fraction=0.8),  # four of five; with no sensor in the hand a success is the close command's word
        on_attempt=_each_attempt,
        view=view,  # every camera of the cell, each attempt's grasp overlay pinned on the primary camera's window
    ).execute()
print(report)
# A campaign that planned no grasp on a fused cloud picked from one view all along: a camera that delivered nothing,
# or one that never identified what the primary saw. The cell's log says which camera delivered what.
fused = sum(1 for attempt in report.attempts if attempt.fused_views)
print(f"{fused} of {report.attempted} attempt(s) planned on a fused cloud")
raise SystemExit(report.exit_code)
