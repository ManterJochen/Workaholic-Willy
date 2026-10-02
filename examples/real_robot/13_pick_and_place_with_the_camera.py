"""Pick what a camera finds and set it down on something else a camera finds: no coordinate in the program.

Two prompts name the part and where it goes. Every camera of the cell's world is opened, one owner each (16), with a
Locator each over one set of models (15); the first that locates a prompt answers it, and SHOW_CAMERAS shows each live.
A wrist camera looks around for the part, its looks fused until the grasp is safe and kept (record_views), and for the
target from each look in turn; a fixed one does not move. A grasp refused before anything was sent gets a fresh look and
the next grasp. The part hangs at least as far below the grasp as the grasp stood above the declared table, so it comes
down 5 mm or more over the middle of the target's seen top, never into it, the target held out of the world.

Run it at the cell, under the cell's profile, once its cameras are calibrated (07-10):
    WILLY_PROFILE=<your cell> python examples/real_robot/13_pick_and_place_with_the_camera.py
"""

from contextlib import ExitStack

from willy import Camera, CameraWorldPlan, JointPositions, LiveView, Located, Locator, Robot, load_tree

SHOW_CAMERAS = True  # False: no camera windows
BOTH_FACES = False  # True: grip only once both jaw contact faces of the part's grasp were seen, else pick nothing
OBJECT, TARGET = "a red cube", "the blue plate"  # what to pick, and what to set it down on
LOOK = [JointPositions.deg(-90.0, -100.0, -110.0, -60.0, 90.0, 0.0),  # FILL IN: wrist looks, pendant degrees (11)
        JointPositions.deg(-70.0, -100.0, -110.0, -60.0, 90.0, 0.0)]

section = (tree := load_tree()).app_config.camera.cameras
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
        refused: list = []  # grasps a guard or the planner refused before anything was sent, the jaws left open
        for _ in range(1 if BOTH_FACES else 4):  # the best grasp, then up to 3 next ones, each on a fresh look
            for locator in locators:  # the part: the looks' frames stay in the camera world until the pick ends
                print(seen := locator.look_around(OBJECT, LOOK, robot=robot, both_faces=BOTH_FACES, record_views=True))
                if seen.refused or seen.objects:  # found, or a look not reached or a contact face no look showed
                    break
            if seen.refused or not seen.objects:
                raise SystemExit((seen.refused or f"no camera located {OBJECT!r}") + "; nothing was picked")
            print(grasps := (scene := seen.scene(0, tree.robot)).grasps().other_than(refused))  # or why none
            if (best := grasps.best) is None:  # opt in: look_around(..., closing_axis="-y"); the scene takes it
                raise SystemExit(f"no grasp on {OBJECT!r} as the camera saw it; nothing was picked")
            print(picked := robot.pick(best.pose(), best.grip_width_mm, keep_out=seen.keep_out(0)))  # standoff_mm=80
            if picked.ok or not picked.another_candidate_may_follow:
                break
            refused.append(best)  # nothing was sent and the jaws stand open: look again, then the next grasp
        if best is None or not picked.ok:
            raise SystemExit("the pick did not finish, so nothing is set down; the report above says what ran")
        onto = find(TARGET, "the part stays held, and nothing was released")  # a target is not gripped: first sight
        print(set_down := onto.set_down(0, grasp=best.pose(), part_bottom_mm=scene.part_bottom_mm))
        if set_down.pose is None:
            raise SystemExit("the target's top could not be read; the part stays held, and nothing was released")
        print(placed := robot.place(set_down.pose, keep_out=onto.keep_out(0)))  # opens only once the line in arrived
        if not placed.ok:
            raise SystemExit("the place did not finish; the part stays held unless the report above says RELEASED")
