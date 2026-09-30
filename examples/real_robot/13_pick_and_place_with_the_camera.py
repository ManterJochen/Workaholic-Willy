"""Pick what a camera finds and set it down on something else a camera finds: no coordinate in the program.

Two prompts name the part and where it goes. Every camera the cell's world is built from is opened, one owner each
(16), with a Locator each over one set of models (15), and the first camera that locates a prompt answers it. A camera
on the wrist looks around for the part, its looks fused until the part's grasp is safe, and for the target from each
look in turn until one sees it; a fixed one does not move. With SHOW_CAMERAS each camera shows live in a window.

The set-down is measured: the target's top is a high percentile of its seen surface, and the part hangs below the grasp
at least as far as the grasp stood above the table the cell declares (further where the looks saw it stand lower), never
from its lowest seen point, which a hidden foot raises. So it comes down to 5 mm of air or more over the middle of the
target's top, never into it, and the target is held out of the camera world for the place as the part was for the pick.

Run it at the cell, under the cell's profile, once its cameras are calibrated (07-10):
    WILLY_PROFILE=<your cell> python examples/real_robot/13_pick_and_place_with_the_camera.py
"""

from contextlib import ExitStack

from willy import Camera, CameraWorldPlan, JointPositions, LiveView, Located, Locator, Robot, load_tree

SHOW_CAMERAS = True  # False: no camera windows
BOTH_FACES = False  # True: grip only once both jaw contact faces of the part's grasp were seen, else pick nothing
OBJECT, TARGET = "a red cube", "the blue plate"  # what to pick, and what to set it down on
# Where a wrist camera looks from, in order: degrees per joint, as the pendant shows them. FILL IN with 11's lines.
LOOK = [JointPositions.deg(-90.0, -100.0, -110.0, -60.0, 90.0, 0.0),
        JointPositions.deg(-70.0, -100.0, -110.0, -60.0, 90.0, 0.0)]

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
        for locator in locators:  # the part: the looks' frames stay in the camera world until the pick ends
            print(seen := locator.look_around(OBJECT, LOOK, robot=robot, both_faces=BOTH_FACES))
            if seen.refused or seen.objects:  # found, or a look not reached or a contact face no look showed
                break
        if seen.refused or not seen.objects:
            raise SystemExit((seen.refused or f"no camera located {OBJECT!r}") + "; nothing was picked")
        scene = seen.scene(0, tree.robot)  # the part's grasps, planned on what it stands on
        if (best := scene.grasps().best) is None:  # opt in: look_around(..., closing_axis="-y"); the scene takes it
            raise SystemExit(f"no grasp on {OBJECT!r} as the camera saw it; nothing was picked")
        print(picked := robot.pick(best.pose(), best.grip_width_mm, keep_out=seen.keep_out(0)))
        if not picked.ok:
            raise SystemExit("the pick did not finish, so nothing is set down; the report above says what ran")
        onto = find(TARGET, "the part stays held, and nothing was released")  # a target is not gripped: first sight
        print(set_down := onto.set_down(0, grasp=best.pose(), part_bottom_mm=scene.part_bottom_mm))
        if set_down.pose is None:
            raise SystemExit("the target's top could not be read; the part stays held, and nothing was released")
        print(placed := robot.place(set_down.pose, keep_out=onto.keep_out(0)))  # opens only once the line in arrived
        if not placed.ok:
            raise SystemExit("the place did not finish; the part stays held unless the report above says RELEASED")
