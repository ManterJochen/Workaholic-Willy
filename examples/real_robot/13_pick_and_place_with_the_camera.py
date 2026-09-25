"""Pick what a camera finds and set it down on something else a camera finds: no coordinate in the program.

Two prompts name the part and where it goes. Every camera the cell's world is built from is opened, one owner each
(16), with a Locator each over one set of models (15), and the first camera that locates a prompt answers it. A camera
on the wrist sees the table only from a look pose, so the arm goes to each look below in turn before it locates; a
fixed one does not move. With SHOW_CAMERAS each camera shows live in a window of its own, its locates' masks on it.

The set-down is measured: the target's top is a high percentile of its seen surface, and the part hangs below the grasp
as far as the grasp stood above the table the cell declares, never above the part's lowest seen point, which a hidden
foot raises. So the part comes down to at least 5 mm of air over the middle of the target's top, never into it (one
taken off a raised block drops the block's height too), and the hand opens only once it is there. The target is held
out of the camera world for the place, as the part was for the pick; only a place whose line in arrived lets go.

Run it at the cell, under the cell's profile, once its cameras are calibrated (07-10):
    WILLY_PROFILE=<your cell> python examples/real_robot/13_pick_and_place_with_the_camera.py
"""

from contextlib import ExitStack

from willy import Camera, CameraWorldPlan, JointPositions, LiveView, Located, Locator, Robot, load_tree

SHOW_CAMERAS = True  # False: no camera windows
OBJECT, TARGET = "a red cube", "the blue plate"  # what to pick, and what to set it down on
# Where a wrist camera looks from, tried in order: degrees, one per joint, as the pendant shows them. FILL THESE IN from
# your own cell, with the lines 11 prints.
LOOK = [
    JointPositions.deg(-90.0, -100.0, -110.0, -60.0, 90.0, 0.0),
    JointPositions.deg(-70.0, -100.0, -110.0, -60.0, 90.0, 0.0),
]

tree = load_tree()
section = tree.app_config.camera.cameras
plan = CameraWorldPlan.from_config(tree.robot, list(section.rigs), primary_rig_id=section.primary_rig_id)
if plan.refusal() is not None or not plan.rig_ids:
    raise SystemExit(plan.refusal() or f"no camera feeds a world on this cell, and 13 finds by camera: {plan.reason}")

with ExitStack() as owners:  # one owner per rig, the primary first, each released however the program ends
    cameras = [owners.enter_context(Camera.from_tree(tree, rig_id=rig_id)) for rig_id in plan.rig_ids]
    robot = Robot.from_tree(tree, cameras=cameras)  # every motion checked against what all of them see
    view = owners.enter_context(LiveView(cameras, show=SHOW_CAMERAS))  # closed before the cameras are given back
    locators = Locator.for_cameras(tree, cameras, tool_pose=robot.arm.get_tcp_pose, view=view)  # one set of models

    def find(prompt: str, otherwise: str) -> Located:
        """The first camera that locates ``prompt``: one on the wrist from each look in turn, a fixed one as it is."""
        for locator in locators:
            for look in LOOK if locator.on_the_wrist else [None]:
                if look is not None and not (moved := robot.move_joints(look)).ok:
                    raise SystemExit(f"{moved}\nthe arm did not reach a look; {otherwise}")
                print(located := locator.locate(prompt))  # what it saw, or that it saw nothing
                if located.objects:
                    return located
        raise SystemExit(f"no camera located {prompt!r}; {otherwise}")

    with robot.connected():
        seen = find(OBJECT, "nothing was picked")
        scene = seen.scene(0, tree.robot)  # the part's grasps, planned on what it stands on
        if (best := scene.grasps().best) is None:
            raise SystemExit(f"no grasp on {OBJECT!r} as the camera saw it; nothing was picked")
        print(picked := robot.pick(best.pose(), best.grip_width_mm, keep_out=seen.keep_out(0)))
        if not picked.ok:
            raise SystemExit("the pick did not finish, so nothing is set down; the report above says what ran")
        onto = find(TARGET, "the part stays held, and nothing was released")
        # The grasp's own turn over the middle of the target's top, at the top + the hang from the table + 5 mm of air.
        print(set_down := onto.set_down(0, grasp=best.pose(), part_bottom_mm=scene.declared_support_height_mm))
        if set_down.pose is None:
            raise SystemExit("the target's top could not be read; the part stays held, and nothing was released")
        # The hand opens only once the line in has arrived; the target leaves the world for this place alone.
        print(placed := robot.place(set_down.pose, keep_out=onto.keep_out(0)))
        if not placed.ok:
            raise SystemExit("the place did not finish; the part stays held unless the report above says RELEASED")
