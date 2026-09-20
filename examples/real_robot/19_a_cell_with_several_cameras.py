"""A cell with more than one camera: which rigs feed the planner world, and one owner per device.

Every earlier file opens the primary rig and says no more about it. A real cell often has several:
an overhead RGB-D, a second one across the bin, a wrist camera. Three rules decide what happens.

One owner per device. `Camera` is the only code that opens a rig, and a second owner on the same
rig is refused rather than given a second handle, so two parts of a program cannot both claim it.
Open one `Camera` per rig and hand them all to the robot.

Not every rig is a world camera. A rig feeds the live planner world only if it is enabled, is
RGB-D, and declares its calibration; `CameraWorldPlan` says which rigs do and, one line each, why
the others do not. It reads config and opens nothing, so this part answers at a desk.

A calibrated rig's world is mandatory. Once a rig declares its calibration on a cuRobo cell, a
build that produced no world is refused instead of planning blind (`CameraWorldRequired`).

Run it at the cell, under the cell's profile, once its cameras are calibrated (07-10):
    WILLY_PROFILE=<your cell> python examples/real_robot/19_a_cell_with_several_cameras.py
"""

from willy import Camera, CameraWorldPlan, Pose, Robot, load_tree

tree = load_tree()
cameras_section = tree.app_config.camera.cameras

# Every rig the tree declares, before a device opens.
for rig in cameras_section.rigs:
    primary = " (primary)" if rig.rig_id == cameras_section.primary_rig_id else ""
    print(f"  {rig.rig_id}{primary}: {rig.source}, enabled {rig.enabled}, "
          f"calibration {'declared' if rig.extrinsics is not None else 'NONE'}")

# Which of them the live planner world is built from, and why each of the others is not. Pure: no
# device is touched, so this is the call to run first when a cell refuses to build.
plan = CameraWorldPlan.from_config(tree.robot, list(cameras_section.rigs),
                                   primary_rig_id=cameras_section.primary_rig_id)
print(plan)
if plan.refusal() is not None:
    raise SystemExit(plan.refusal())
if not plan.rig_ids:
    raise SystemExit("no camera feeds a world on this cell; 19 is about the cells that have one")

# One owner per rig, opened in the plan's order: the primary first, then the rest. ExitStack is not
# needed for two, and a `with` per camera is what a reader can follow; for a variable number, nest
# them with contextlib.ExitStack instead.
with Camera.from_tree(tree, rig_id=plan.rig_ids[0]) as primary_camera:
    others = [Camera.from_tree(tree, rig_id=rig_id) for rig_id in plan.rig_ids[1:]]
    for camera in others:
        camera.open()
    try:
        # Every camera the world takes, handed over together. The print says how the arm's motions
        # are planned and which world they are checked against, and it names the rigs behind it.
        robot = Robot.from_tree(tree, cameras=[primary_camera, *others])
        print(robot)

        with robot.connected():
            # No `decline`: the world is wired from every rig above, so this move is checked
            # against what all of them see right now, and is refused if any of them drops out.
            print(robot.move(Pose.tool_down(400.0, 0.0, 350.0)))
    finally:
        # Given back in reverse, so the primary is the last device released, as it was the first
        # claimed. The `with` above releases it.
        for camera in reversed(others):
            camera.release()
