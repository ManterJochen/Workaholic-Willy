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
| `Locator` | `from_tree(tree, camera=, tool_pose=)`, `for_cameras(tree, cameras, tool_pose=)` (one per camera over one backend: the models load once for the cell), `from_config(robot, models, camera=)`, `from_parts(camera=, backend=)` | `locate(prompt)`, a class list too (`"yellow bin \| blue bin"`: every description in one detector call, see below); `look_around(prompt, looks=None, robot=, both_faces=False, record_views=False, closing_axis=UNSET)`; `measure(points_base_mm)`, one new frame and no detector: the depth each BASE point should read, the depth measured at its pixel and that pixel's colour (`Measured`; a task checks its bin's rim with it, 2026-10-08), `located.measure(points, image_bgr)` the same on the frame a locate placed | `Located` |
| `Located` | `locator.locate(prompt)`, or `locator.look_around(...)` for a part to grip | `scene(i, robot_config)`, `keep_out(i)`, `set_down(i, grasp=, part_bottom_mm=)`, `split(prompt)` | object i as a grasp `Scene` with the others as obstacles; what the planner leaves out to reach it; where a held part is set down on it; which objects each description of a class list located, by index (`{"yellow bin": (0,), "blue bin": ()}`). A look around adds `looks`, `looks_fused`, `jaw_faces_seen`, `refused`, `hand_eye_gap_mm`, `generated_view_deg`, `views_file` and `closing_axis` |
| `SetDown` | `located.set_down(i, grasp=, part_bottom_mm=scene.part_bottom_mm, air_mm=5.0)` | `pose`, `reason` | the place pose over the middle of object i's top: its top (95th percentile of the seen surface), plus the part's hang below the grasp, plus the air; `pose` is `None`, with the `reason`, where it cannot be measured |
| `LocatedObject` | an entry of `located.objects` | | label, score, box, mask, points in BASE and their per-axis median centre |
| `RealSenseVisionPerceptionSource` | `RealSenseVisionPerceptionSource(streamer=, backend=, prompt=)` | `acquire()` | one `PerceptionFrame` for the pick loop |

The locator never opens or releases the camera the caller owns. `Locator`, `Located` and `LocatorRefused`
come from `willy`, the rest from `src.robot.perception`. No orientation is estimated (`LocatedOrientation`
says so), and an empty result reads the same as a failed detector, which `print(located)` says.

**A class list in one locate** (the sorting package, 2026-10-09). A task that places into several bins finds them
all in one detector call: `located = locator.locate(class_list_prompt(["yellow bin", "blue bin"]))`
(`src.models.vlm.qwen`). Each box's words are mapped onto the description they name, as a pick's camera source maps
them (below), and `located.split(prompt)` says which objects each description located, in the prompt's order:
`{"yellow bin": (0,), "blue bin": (2,)}`, `()` for a bin not seen. What no description clearly claims, a label of
another thing, one two descriptions fit equally, or the detector's `ambiguous` (an object it boxed under two of
them), belongs to none and stays an obstacle. A locate of one description reads as it always did, every object its
own. **A locate judges no colour**: a bin's mask holds whatever lies in it, and the colour check's thresholds were
measured on parts on the mat, so a bin keeps the description its box named whatever its pixels say.

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
from willy import Camera, load_tree
from src.models.perception_spec import PerceptionSpec
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
| Labels | `object_labels=` maps the detector's free phrases onto your object names, so `label == target` works: a phrase maps onto the name whose every word it holds (see below) |
| Colour | a part mapped onto a name whose words name a colour is judged on its pixels first (`colour_check=`, see below); a part of another colour keeps its mask under a label no object carries |
| Follow | `acquire(follow=...)` finds the parts a task kept again on the same frame, with SAM2 on their boxes and no detector, where every check passes, and grounds the frame as before where any fails (see below); `last_route` says which |
| Hide | `acquire(hide=...)` has the detector read a copy with the regions a task keeps out painted over (`kept_out_pixels`, see below); the segmenter and everything after it read the real frame; `last_hidden_px` says how many pixels were painted |
| Shutter time | `timestamp` is the camera owner's `captured_at_s`, else a clock read just before the grab |
| Wrist camera | the TCP is read before and after the grab; a frame taken while the tool moved is taken again (`ShutterStamp`, the stamp a hand finder over a wrist camera takes too) |

`set_prompt(prompt, object_labels=...)` changes what the source looks for from the next frame on, with
no reopen and no model reload. `stamp_tool_pose_with(reader, motion_tolerance=, attempts=)` binds a wrist
camera to the arm's TCP reader; a real cell binds its primary and every fused wrist camera this way.

**Which name a label maps onto** (2026-10-08). A detector's label maps onto the object name whose every word it holds:
"each separate gray cubes" is a "grey cube" (grey and gray are one word, a plural its singular). The most specific
name wins, two equally specific names map nothing, and a bare shortened label ("cube") maps only where there is one
name to shorten. Anything else passes through: "red cube" never becomes "grey cube", so it stays an obstacle and is
never a target. It used to map onto the name it shared one word with, and every box the detector drew was a target.
A sort's source maps onto every rule's kind at once (`object_labels=("green part", "red part")`, its prompt a class
list, `"each separate green part | each separate red part"`): the detector's `ambiguous`, "orange part" and "green or
red part" map onto none, and the pick loop's label gate, which takes any of the kinds (`target_labels`), passes them by.

**The colour check** ([`colour_check.py`](colour_check.py), `robot.grasping.colour_check`). Asked for "each separate
grey cube", Qwen3-VL boxed the green and the orange parts as well and labelled every box with the prompt's words
(the owner's cell, 2026-10-08). So a part mapped onto a name whose head names one colour (the words before a place
word, English or German: "grauer Würfel auf der schwarzen Matte" names grey) is judged on its own pixels before any
fusion reads it, in about 2 ms:

- **The pixels.** The mask eroded by 2 px, in CIELAB; pixels darker than L* 20 left out (the mat, a shadow, the
  edge's bleed) unless black is wanted; a pixel coloured from C* 12.
- **Clipped pixels** (`robot.grasping.colour_check_clipped`, `exclude` as shipped, the owner's choice of 2026-10-09:
  "wenn das schon so gut war, dann nutzen wir das doch instant"). A pixel with one or two channels at 250 or over and
  not all three is left out of every count, the 50 pixels' included (`CLIPPED_AT`, `ColourClipped`), and the
  verdict says how many were. The colour camera runs on auto exposure and clips an orange part's red channel where
  the mat around it is dark: its hue is then read off green alone and lands in the yellow band. On the 31 looks the
  cell recorded on 2026-10-07, through `judge_colour` itself: orange right 53 of 89 times with them counted, 89 of 89
  left out; the 69 grey, 30 white and 99 green sightings and the yellow bin judged exactly as before. No red part
  stood in those views. `keep` counts them, as before; glare, clipped in all three channels, is counted either way.
- **Grey, white or black.** Half the part coloured or more refuses it, more than a fifth leaves it unsure; otherwise
  the 80th percentile of its colourless pixels' L* says grey up to 74 and white from 80, unsure between, and black is
  a median L* up to 25. A part mostly darker than L* 20 is black.
- **A colour with a hue.** Lab hue bands: red 345-45, orange 45-75, yellow 75-115, green 115-190, cyan 190-250, blue
  250-320, purple 320-345 degrees. Half the part in its band agrees; 0.15 or less while another band holds half, or
  while the part is a fifth coloured or less, refuses it; anything else is unsure, and so is a part of fewer than 50
  pixels. Pink and brown are always unsure.
- **Unsure** asks the backend's `name_colour` for one colour word on a crop of the box (Qwen3-VL, about 0.3 to 1.5 s,
  unmeasured on the cell), read through the same words. No answer refuses the part, which stays on the mat; a question
  that raises is counted as the model's failure, so a task that then sees nothing ends `detector_failed`.

A refused part keeps its mask, its depth and its place in the planner world under a label no object carries ("green
object, not grey cube"): the label gate passes it by, and it stays a neighbour for the calculator, the push, the
blocker and the planner. Every verdict is logged with what the detector called the part (`perception_source.log`).
`colour_check="log"` judges and logs and changes nothing; `"off"` judges nothing. The thresholds come from 142 part
sightings on 9 of the cell's home views (2026-10-07/08): every part was half coloured or more, or a fifth or less, and
the grey parts read p80 L* 52-69, the white ones 80-100.

**The task's own places, hidden from the detector** (`robot.grasping.hide_own_places`, off by default; the owner,
2026-10-08 night). Every look that saw the task's bin had the detector box every part already placed in it, 2 to 4 s a
box on the cell, each then left out as standing in the bin. Where the cell turns it on, a pick hands `acquire` a
`hide` that answers which pixels to paint (`kept_out_pixels`): every pixel its own depth places in a region the task
keeps out of every label (the bin's footprint and 10 mm, the circle about a drop), and nothing of a part that reaches
out of one. What stands more than 5 mm over the declared support is painted whole or not at all: a patch of it with 25
or more pixels placed outside every region, by depths that agree with their neighbours' within 10 mm (no flying pixel
at an edge counts), keeps every pixel and the 3 round it. A hole in the depth is painted only where painted pixels
close round it. `painted_copy` paints them in the colour of the ring round them, what the parts stand on. The detector
reads that copy alone, through the backend's `perceive(image, prompt, detect_on=copy)`; a backend that does not say it
takes one (`detects_on_a_copy`) is asked as it always was, said once, and so is every frame whose `hide` raises or
answers no mask of its size. `hide` reads the depth through a view it cannot write.

### Following the parts a task kept

Every pick of a task used to ground its first look again: Qwen boxing every part on the mat, 7 to 17 s (13.1 s at the
median of 15 groundings on the cell, 2026-10-07/08), to find the parts where the last pick left them. The owner's rule
(2026-10-09): Qwen on the first pick and on every trigger, SAM2 on the known parts between, all or nothing per frame,
and any doubt grounds the frame as before. [`kept_scene.py`](kept_scene.py) holds the memory, `KeptScene`: the parts
of the target label one look grounded or followed, outside the task's keep-out regions, each its surface in BASE, its
colour, its top and its pixel count, with the look's depth, lens and camera placement. A task builds it from a pick's
first look less the part the pick gripped (`of_look`, `without`), and the next pick's first look hands the source a
follow (`acquire(follow=)`), which reads that look's own frame, no extra grab:

1. **Nothing new in depth.** The kept frame's points go into the new camera and the new frame's into the kept one, at
   every third pixel, inside the cell's workspace: changed where off by more than 10 mm, or new over the support where
   the kept frame read a hole or something farther (a part set down on the shiny profiles, where the last frame read
   holes, shows only this second way). Left out: the task's keep-out regions (its bin, the circle about its drop: an
   expected change), each kept part's footprint grown by `max_shift_mm` and 15 mm, and the taken part's, which must read
   at most 10 mm over the support. A changed cluster of 1000 px, or half the smallest kept part, grounds the frame.
2. **Each part where it stood**, z-buffered by the locator's rule: 80 % of it seen with depth, 85 % of that within
   8 mm and its colour within dE 10 stands where it stood, boxed by its pixels and 12 px; 60 % farther by 10 mm is gone
   and grounds the frame; anything between is boxed wider by `max_shift_mm` for its mask to measure the move.
3. **SAM2 on the boxes** (`backend.segment_boxes`), no detector asked.
4. **Each mask accepted** where its pixels are 0.6 to 1.5 of the kept part's, it moved at most `max_shift_mm` and crept
   at most `max_creep_mm` since its grounding, at most 2 % of its surface lies 15 mm past the kept footprint shifted by
   that move, its top stands within 6 mm and its colour within dE 10; then the colour check and the label mapping run
   on it as on every mask, and a refusal grounds the frame. Every mask is clipped to that shifted footprint and 15 mm.

The frame's segmentations are then the followed masks under their kept labels, and `last_route` is `("followed", ...)`;
a frame that fails any step is grounded by the detector, `last_route` `("grounded", "not followed: why")`, and a frame
handed no follow says no route, as before. What a pick leaves out of the planner world is its target's mask and surface
in this frame, never the kept cloud, and never past the part by more than 15 mm. Parts not kept (another label, a part
in the bin) are not segmented on a followed frame: they stay plain depth, which keeps the full distance. Following adds
or removes no depth point, and the kept frame never reaches the planner world.

A later look of the same pick (`KeptScene.projected`) segments the parts its first look saw by their surfaces projected
at its own stamped pose and padded 12 px, every mask held to its part's footprint and 15 mm (the extent guard), at least
0.6 of the projected top, at its top and of its colour, all or nothing. The numbers behind each threshold are measured
on the cell's recorded frames and noted in the module.

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
| The colour check and the label mapping | rule checked offline on 9 of the cell's recorded home views (142 sightings); pinned by tests on painted patches and a ray-cast mat (`tests/test_a_task_picks_the_grey_parts_and_ends_without_a_long_search.py`); not yet run on the cell (`colour_check: log` first) |
| The clipped pixels left out (`colour_check_clipped: exclude`) | measured offline through `judge_colour` on the cell's 31 recorded looks of 2026-10-07 (orange 53 to 89 of 89, every other part as before); pinned by tests on painted patches (`tests/test_the_colour_check_leaves_clipped_pixels_out.py`); no red part measured; not yet run on the cell |
| Following the parts a task kept (`kept_scene.py`) | thresholds measured offline on the cell's recorded frames of 2026-10-07/08 (CPU); pinned by tests on ray-cast frames (`tests/test_a_kept_scene_follows_only_what_stands_where_it_stood.py`, `tests/test_a_task_follows_its_parts_instead_of_grounding_them_again.py`); never run on the cell, off until a bench run proves it |
| The task's own places hidden from the detector (`kept_out_pixels`) | pinned by tests on ray-cast frames, flying pixels and holes included (`tests/test_the_tasks_own_places_are_hidden_from_the_detector.py`); needs a backend that takes `detect_on`; never run on real depth or the cell, off by default |

How the stack behaves on real depth, which returns zeros inside a mask and leaks at its edges, is not
measured. It is the largest untested surface in the stack, and it can be tested without a robot: a
camera on a desk, pointed at the real parts, through the exerciser above.

## Files

| File | Holds |
|---|---|
| [`locator.py`](locator.py) | `Locator`, `Located`, `LocatedObject`, `LocatedOrientation`, `LocatorRefused`, `SetDown`, `Measured` |
| [`realsense_source.py`](realsense_source.py) | `RealSenseVisionPerceptionSource`, the live camera source of a real cell; `kept_out_pixels` and `painted_copy`, the regions a task keeps out painted over the detector's copy of a frame |
| [`colour_check.py`](colour_check.py) | `colour_named`, `judge_colour`, `ColourVerdict`, `ColourCheck`, `ColourClipped`, `CLIPPED_AT`, `ask_colour`, `refused_label`: the colour a phrase names against a part's pixels |
| [`kept_scene.py`](kept_scene.py) | `KeptScene`, `KeptPart`, `Following`, `Accepted`: the parts a task kept, found again on a new frame with no detector, or why not |
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
  [`test_a_part_of_another_colour_is_no_target.py`](../../../tests/test_a_part_of_another_colour_is_no_target.py),
  [`test_a_kept_scene_follows_only_what_stands_where_it_stood.py`](../../../tests/test_a_kept_scene_follows_only_what_stands_where_it_stood.py),
  [`test_a_source_grounds_the_frame_whenever_following_fails.py`](../../../tests/test_a_source_grounds_the_frame_whenever_following_fails.py),
  [`test_a_followed_target_is_kept_out_by_its_new_mask.py`](../../../tests/test_a_followed_target_is_kept_out_by_its_new_mask.py),
  [`test_a_label_of_another_colour_is_never_mapped_onto_the_object.py`](../../../tests/test_a_label_of_another_colour_is_never_mapped_onto_the_object.py),
  [`test_a_located_part_is_looked_at_from_every_look.py`](../../../tests/test_a_located_part_is_looked_at_from_every_look.py),
  [`test_a_part_is_placed_on_what_the_camera_found.py`](../../../tests/test_a_part_is_placed_on_what_the_camera_found.py),
  [`test_a_pick_closes_along_the_axis_its_program_names.py`](../../../tests/test_a_pick_closes_along_the_axis_its_program_names.py),
  [`test_every_grasp_closes_the_way_round_the_hand_naturally_stands.py`](../../../tests/test_every_grasp_closes_the_way_round_the_hand_naturally_stands.py),
  [`test_the_looks_say_how_many_agreed_on_the_label.py`](../../../tests/test_the_looks_say_how_many_agreed_on_the_label.py).
