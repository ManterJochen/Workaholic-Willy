# Where objects are, from a live camera (`src/robot/perception`)

Grounds a prompt in one RGB-D frame from a cell's camera and places what it finds in the robot's base
frame: the detector boxes each object, the segmenter cuts it out, and the depth under its mask becomes
points in millimetres.

```python
from willy import Camera, Locator, Robot, load_tree

tree = load_tree()
with Camera.from_tree(tree) as camera:                   # the cell's primary camera, released at the end
    robot = Robot.from_tree(tree, cameras=[camera])      # the arm plans around what the camera sees
    locator = Locator.from_tree(tree, camera=camera, tool_pose=robot.arm.get_tcp_pose)   # loads the models
    with robot.connected():
        located = locator.locate("a red cube")           # one frame, every grounded object placed in BASE
        print(located)                                   # every object: label, score, centre in mm
        if located.objects:
            best = located.scene(0, tree.robot).grasps().best   # object 0; the others are obstacles
            if best is not None:
                print(robot.pick(best.pose(), best.grip_width_mm, keep_out=located.keep_out(0)))
```

A fixed camera ignores `tool_pose`; a wrist camera needs it, because its frame is placed by where the
tool stood at the shutter (`robot.core.shutter_motion.camera_to_base_at_shutter`, the composition the
Locator shares with the hand finder over a wrist camera). The same program at a cell is
[`examples/real_robot/15_speak_pick_and_hand_handover.py`](../../../examples/real_robot/15_speak_pick_and_hand_handover.py), with a spoken prompt. Prove
the camera and the models at a desk before a robot is involved:

```bash
python -m src.robot.perception --prompt "a red cube"
python -m src.robot.perception --prompt "cube ; screwdriver ; mug"   # several objects at once
python -m src.robot.perception --rig <rig id> --warmup 10
```

It opens the primary rig (or `--rig`, even one switched off), grabs one frame after the warm-ups, runs
both models and prints the lens matrix and the depth holes inside every mask. It needs the camera and the
weights, no robot, and exits 0 after the frame or 1 with one sentence for an unknown rig or one with no depth.

## The nouns

| Noun | Built by | Verb | Returns |
|---|---|---|---|
| `Locator` | `from_tree(tree, camera=, tool_pose=)`, `for_cameras(tree, cameras, tool_pose=)` (one per camera over one backend: the models load once for the cell), `from_config(robot, models, camera=)`, `from_parts(camera=, backend=)` | `locate(prompt)`; `look_around(prompt, looks=None, robot=, both_faces=False, record_views=False, closing_axis=UNSET)` | `Located` |
| `Located` | `locator.locate(prompt)`, or `locator.look_around(...)` for a part to grip | `scene(i, robot_config)`, `keep_out(i)`, `set_down(i, grasp=, part_bottom_mm=)` | object i as a grasp `Scene` with the others as obstacles; what the planner leaves out to reach it; where a held part is set down on it. A look around adds `looks`, `looks_fused`, `jaw_faces_seen`, `refused`, `hand_eye_gap_mm`, `generated_view_deg`, `views_file` and `closing_axis` |
| `SetDown` | `located.set_down(i, grasp=, part_bottom_mm=scene.part_bottom_mm, air_mm=5.0)` | `pose`, `reason` | the place pose over the middle of object i's top: its top (95th percentile of the seen surface), plus the part's hang below the grasp, plus the air; `pose` is `None`, with the `reason`, where it cannot be measured |
| `LocatedObject` | an entry of `located.objects` | | label, score, box, mask, points in BASE and their per-axis median centre |
| `RealSenseVisionPerceptionSource` | `RealSenseVisionPerceptionSource(streamer=, backend=, prompt=)` | `acquire()` | one `PerceptionFrame` for the pick loop |

The locator never opens or releases the camera the caller owns. `Locator`, `Located` and `LocatorRefused`
come from `willy`, the rest from `src.robot.perception`. No orientation is estimated (`LocatedOrientation`
says so), and an empty result reads the same as a failed detector, which `print(located)` says.

An object's points leave out the mask pixels that measure more than 10 mm behind a pixel of the same
mask within 3 px: depth-to-colour misregistration at a part's far edge puts the table behind the part
inside the mask, and on a camera tilted 45 degrees that table lies 40 mm behind the part. The mask itself
stays as segmented, and `keep_out(i)` holds all of it out of the planner world. `scene(i, robot_config)`
plans with the hand `grasping.gripper_geometry` describes and on the support the pick loop would use,
raised to the object's own lowest point when its surface reaches down to what it stands on.

A set-down measures the held part's hang below the grasp from `scene.part_bottom_mm`: the table (or
container floor) the cell declares, lowered only where two or more looks of a wrist camera measured the
part's foot below it (its 20th-lowest point, so one stray reading drops nothing), and never raised. Not
from `scene.support_height_mm`: the planned support is raised to the part's lowest *seen* point, which
stands above its real base wherever the camera missed the part's foot, and a hang measured from there
brings the part down into the target by that much (8 mm unseen pressed it 3 mm in, 12 mm pressed it 7 mm
in). A declared height set too high would press the part in the same way, which is what the looks'
measurement corrects. The declared support is a lower bound on where the part stood, so every error goes
to more air. One view, a fixed camera's above all, keeps the declared support:
`scene.declared_support_height_mm` is the same bound without the measurement. The cost is a larger drop
for a part taken off a raised block (a 30 mm block: 35 mm instead of 5). A caller that knows where the
part stood passes that as `part_bottom_mm`.

### Looking around for a part to grip

`locator.look_around(prompt, looks, robot=robot)` is how a wrist camera finds a part to grip. The arm goes
to each look through `robot.move_joints` (`robot.home` for `"home"`), and each look is fused with the ones
before, at the wrist tolerances and with grazing points thinned ([multiview/](../grasping/multiview/README.md)).

- **One part.** Object 0 of the first look that measured its surface (20 points or more) is the part,
  and association keeps it. A look that located it with no surface under its mask makes no part and is
  said; a look that misses it is left out and said. Parts only earlier looks saw stay as obstacles.
- **Early stop.** The grasp is computed again on the fused cloud after every look, the support included,
  and the looking stops at the first valid grasp the looks' labels agree on (a name the locator made up
  is no label). With `both_faces` it stops only once both jaw contact faces of the chosen grasp were seen,
  and otherwise the part is refused (`Located.refused`), after the generated view too.
- **One generated view** follows once every declared look is used up, reached or refused before
  anything was sent, and the grasp is still not safe: the same one a pick's looks generate, through
  `execution/generated_view.py`, on the straight joint line only. Where the part was last seen from
  another look, the arm then moves back there on that line, to the joints read off the arm at that look.
  An arm without `ChoosesConfigurations` or `DrivesJointLines` (the Isaac arm), or whose world does not
  hold the frames, generates none, and says why. A look around handed no looks generates nothing.
- **Refusals.** A look refused before anything was sent is skipped and said. The looking ends with
  `refused` set when no look is reachable, when a motion may have moved the arm, when a motion was
  refused before its command for a reason every look shares, when the controller is stopped, or when a
  camera cannot vouch, on the way to a look, to the view or back. A look around that reached no look
  answers with no objects, `refused` set and `captured_at_s` the time the looking ended. A hand that is
  not a parallel jaw, or a robot with no robot section, is refused before anything moves.
- **Held frames.** The frames of the looks are held in the arm's live planner world from the first look the
  arm stands at, never from where it started, until `Robot.pick` ends, the next `look_around`, or the
  disconnect (`Robot.connected()`, `Cell.connected()`). The next look around lets go of them before it moves
  (`LivePlannerWorld.holds_pick_views`), and a look around that raises lets them go itself. A retry after a
  failed pick looks around again first.
- **`record_views=True`** writes one `.npz` for the look around under `logs/robot/views`, named after
  `look_around-<prompt>` with the time; `views_file` says where. A file that cannot be written is said,
  and the look around answers all the same.
- **One line per look.** The per-look INFO account ends `label agreed in N of M looks`: M the looks that saw
  the part, N those that call it what this look calls it. Nothing acts on it.

A place's target is found look by look instead, the first look that sees it answering: a set-down has no
grasp to judge. On a fixed camera `look_around` locates once, where it stands, and keeps nothing;
`both_faces` judges that one frame's grasp the same way.

**Which way the jaws close.** Name the axis once, on the look around: `look_around(..., closing_axis="-y")`
judges only the grasps whose closing axis heads within 30 degrees of it, each turned the named way round,
for the early stop, `both_faces` and the generated view's aim. `Located.closing_axis` keeps it, and
`render()` and `to_dict()` say it. `located.scene(0, ...).grasps()` then takes it; the same axis named
another way (the owner's `Pose.tool_down(..., closing_axis="-y")` for `"-y"`) is accepted, and any other
raises `ValueError` naming `look_around`, as does any axis after a look around that named none. One
`locate()` judges nothing, so its scene takes any axis. A value that names no axis raises before anything
moves, and the per-look line reads `no grasp on the part (closing_axis ...)` where the axis left none.
Without a named axis, the cell's `robot.natural_closing_axis` turns every grasp the nearer way round, read
from the robot config handed in: `both_faces` judges the turned grasp, and its refusal names the jaw after
the turn. Hand `look_around` and `located.scene` the same tree.

### The source a pick perceives through

A real cell hands this source to its pick service; build one to read a `PerceptionFrame` yourself:

```python
from willy import Camera, PerceptionSpec, load_tree
from src.robot.perception import RealSenseVisionPerceptionSource

tree = load_tree()
with Camera.from_tree(tree) as camera:
    source = RealSenseVisionPerceptionSource(
        streamer=camera.handle(),
        backend=PerceptionSpec.from_config(tree.app_config.models).build(),
        prompt="a red cube",
    )
    frame = source.acquire()   # depth, masks, lens matrix, shutter time
```

What `acquire()` does with a frame:

| Step | Behaviour |
|---|---|
| Warm-up | discards `warmup_grabs` frames, 5 by default (`PICK_FRAME_WARMUP_GRABS`), so auto exposure settles |
| Lens | the device's own matrix; `intrinsics=` overrides it, and `source.intrinsics_source` says `factory` or `override` |
| Depth | published as the sensor measured it, holes (0) included; where a grasp is anchored is set by `robot.grasping.geometry` |
| Masks | used exactly as segmented; `mask_completion=` can fill them to a box (see below) |
| Labels | `object_labels=` maps the detector's free phrases onto your object names, so `label == target` works |
| Shutter time | `timestamp` is the camera owner's `captured_at_s`, else a clock read just before the grab |
| Wrist camera | the TCP is read before and after the grab; a frame taken while the tool moved is taken again (`ShutterStamp`, the stamp a hand finder over a wrist camera takes too) |

`set_prompt(prompt, object_labels=...)` changes what the source looks for from the next frame on, with
no reopen and no model reload. `stamp_tool_pose_with(reader, motion_tolerance=, attempts=)` binds a wrist
camera to the arm's TCP reader; a real cell binds its primary and every fused wrist camera this way.

## What it refuses

| Refusal | When | What to do |
|---|---|---|
| `LocatorRefused` | the rig has no depth of its own, a stereo pair | locate with an RGB-D rig |
| `RigNotCalibrated` | the rig declares no `extrinsics`; the message names the key | calibrate it (examples 07 and 08) and paste the block |
| `LocatorRefused` | a wrist rig and no `tool_pose=` | pass `tool_pose=robot.arm.get_tcp_pose` |
| `LocatorRefused` | a wrist rig solved against another flange to TCP than the cell has now | recalibrate, or restore the tool frame it was solved with |
| `LocatorRefused` | `look_around` with a robot that keeps no robot section, or a hand that is not a parallel jaw | build the robot with `Robot.from_tree`; use `locate()` for another hand |
| `LocatorRefused` from `scene(i)` or `set_down(i)` | the `Located` was refused: a look not reached, a stopped controller, or with `both_faces` a contact face no look saw | read `located.refused`; nothing is planned on it |
| `PerceptionFrameMoved` | a wrist camera could not take a still frame within its attempts | keep the arm still at the shutter; the rig's `shutter_motion_tolerance_*` sets how still |
| `ValueError` | `set_prompt("")`, or a source built with neither `backend=` nor `detector=` and `segmenter=` | pass a phrase, or a backend |
| `RuntimeError` | `acquire()` on a device that reports no lens matrix, with no `intrinsics=` | open the camera first, or pass a calibrated matrix |

The locator's refusals run before the models load. A wrist camera takes up to
`robot.safety.planning_world.perceived.fresh_frame_attempts` more frames, 3 by default, before it raises.

## Why masks are not filled to their box

The policies are `none` (the default), `axis_aligned_box` and `oriented_box`. The box rule replaces a
mask covering less than 0.8 of its detection box with the box. Against ground-truth masks it fires on
complete masks as often as on predicted ones, because it detects a shape that is not a rectangle (a
circle fills pi/4 = 0.785 of its box), and a filled mask turns the closing axis onto an image axis. In
paired grasp outcomes `axis_aligned_box` lost several times more grasps than it won, and `oriented_box`
could not be told apart from `none`. [`mask_completion.py`](mask_completion.py) carries the numbers.

## Peeking is not perceiving

A console that shows the camera must not call `acquire()`, which runs the models beside the pick (and
steps the simulator on the Isaac source). [`viewfinder.py`](viewfinder.py) declares `peek_color()`: one
BGR image with no models, no simulation and no change to what the next `acquire()` decides.
`peek_color_of(source)` is `None` for a source that cannot promise that, the Isaac sources among them.
`colour_source_kind` (`camera`, `synthetic` or `unknown`) is declared, never guessed from the image. A
peek takes one frame, which lands among the warm-ups the next `acquire()` discards, and waits its turn
at the rig's lock; [`api/viewfinder.py`](../../../api/viewfinder.py) leaves the camera alone during a run.

## Status

The legend is the guide's [evidence levels](../../../docs/guide/README.md#what-verified-means-on-these-pages).

| Capability | Evidence |
|---|---|
| The live source, the locator and the wrist shutter check | measured on a real camera setup |
| Prompt, detect, segment, masked cloud on the pick path | measured on a real camera setup |
| Mask completion default `none` | measured on a real camera setup |
| `look_around`: the fused looks, `both_faces`, the generated view, the held frames | pinned by tests with fake arms and cameras (`tests/test_a_located_part_is_looked_at_from_every_look.py`); never touched hardware |

How the stack behaves on real depth, which returns zeros inside a mask and leaks at its edges, is not
measured. It is the largest untested surface in the stack, and it can be tested without a robot: a
camera on a desk, pointed at the real parts, through the exerciser above.

## Files

| File | Holds |
|---|---|
| [`locator.py`](locator.py) | `Locator`, `Located`, `LocatedObject`, `LocatedOrientation`, `LocatorRefused`, `SetDown` |
| [`realsense_source.py`](realsense_source.py) | `RealSenseVisionPerceptionSource`, the live camera source of a real cell |
| [`mask_completion.py`](mask_completion.py) | the three mask fill policies and the measurement behind the default |
| [`viewfinder.py`](viewfinder.py) | `ColourPeekable`, `peek_color_of`, `colour_source_kind`, `COLOUR_SOURCE_KINDS` |
| [`__main__.py`](__main__.py) | the bench exerciser; the package itself imports neither `pyrealsense2` nor torch |

## Details

- The `PerceptionFrame` and `PerceptionSource` contract: [`grasping/types/perception.py`](../grasping/types/perception.py).
  This package needs it and a camera, and the camera package never imports the robot, so it lives here.
- The cameras and their owner: [`src/camera/`](../../camera/README.md). The models and `PerceptionSpec`:
  [`src/models/`](../../models/README.md) and [guide 02](../../../docs/guide/02-models.md).
- The simulated sources this mirrors: [`src/willy_sim/`](../../willy_sim/README.md). A physical cell:
  [`real_cell`](../execution/real_cell/README.md) and [the first pick](../../../docs/runbooks/real_cell_first_pick.md).
- Tests: [`test_locator.py`](../../../tests/test_locator.py), [`test_perception_shutter_stamp.py`](../../../tests/test_perception_shutter_stamp.py),
  [`test_realsense_perception_source.py`](../../../tests/test_realsense_perception_source.py),
  [`test_a_located_part_is_looked_at_from_every_look.py`](../../../tests/test_a_located_part_is_looked_at_from_every_look.py),
  [`test_a_part_is_placed_on_what_the_camera_found.py`](../../../tests/test_a_part_is_placed_on_what_the_camera_found.py),
  [`test_a_pick_closes_along_the_axis_its_program_names.py`](../../../tests/test_a_pick_closes_along_the_axis_its_program_names.py),
  [`test_every_grasp_closes_the_way_round_the_hand_naturally_stands.py`](../../../tests/test_every_grasp_closes_the_way_round_the_hand_naturally_stands.py),
  [`test_the_looks_say_how_many_agreed_on_the_label.py`](../../../tests/test_the_looks_say_how_many_agreed_on_the_label.py).
