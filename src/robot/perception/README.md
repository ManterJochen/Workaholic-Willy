# Where objects are, from a live camera (`src/robot/perception`)

Grounds a prompt in one RGB-D frame from a cell's camera and places what it finds in the robot's base
frame: the detector boxes each object, the segmenter cuts it out, and the depth under its mask becomes
points in millimetres. `Locator` is the call for your own code. It reads its frame through the same
source a real cell's pick uses, so the warm-ups, the lens, the shutter stamp and the wrist check are the pick's.

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
tool stood at the shutter. The same program at a cell is
[`examples/real_robot/09_locate_and_pick.py`](../../../examples/real_robot/09_locate_and_pick.py). Prove
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
| `Locator` | `from_tree(tree, camera=, tool_pose=)`, `from_config(robot, models, camera=)`, `from_parts(camera=, backend=)` | `locate(prompt)` | `Located` |
| `Located` | `locator.locate(prompt)` | `scene(i, robot_config)`, `keep_out(i)` | object i as a grasp `Scene` with the others as obstacles; what the planner leaves out to reach it |
| `LocatedObject` | an entry of `located.objects` | | label, score, box, mask, points in BASE and their per-axis median centre |
| `RealSenseVisionPerceptionSource` | `RealSenseVisionPerceptionSource(streamer=, backend=, prompt=)` | `acquire()` | one `PerceptionFrame` for the pick loop |

The locator never opens or releases the camera the caller owns. `Locator`, `Located` and `LocatorRefused`
come from `willy`, the rest from `src.robot.perception`. No orientation is estimated (`LocatedOrientation`
says so), and an empty result reads the same as a failed detector, which `print(located)` says.

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
| Warm-up | discards `warmup_grabs` frames, 5 by default, so auto exposure settles |
| Lens | the device's own matrix; `intrinsics=` overrides it, and `source.intrinsics_source` says `factory` or `override` |
| Depth | published as the sensor measured it, holes (0) included; where a grasp is anchored is set by `robot.grasping.geometry` |
| Masks | used exactly as segmented; `mask_completion=` can fill them to a box (see below) |
| Labels | `object_labels=` maps the detector's free phrases onto your object names, so `label == target` works |
| Shutter time | `timestamp` is the camera owner's `captured_at_s`, else a clock read just before the grab |
| Wrist camera | the TCP is read before and after the grab; a frame taken while the tool moved is taken again |

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

The legend is the root README's [Status and honest scope](../../../README.md#status-and-honest-scope).

| Capability | Evidence |
|---|---|
| The live source, the locator and the wrist shutter check | never touched hardware; no physical camera has fed them, pinned against fakes in [`test_locator.py`](../../../tests/test_locator.py) |
| Prompt, detect, segment, masked cloud on the pick path | measured in simulation, through the Isaac vision sources this adapter mirrors |
| Mask completion default `none` | measured in simulation, over the reference corpus ([`mask_completion.py`](mask_completion.py)) |

How the stack behaves on real depth, which returns zeros inside a mask and leaks at its edges, is not
measured. It is the largest untested surface in the stack, and it can be tested without a robot: a
camera on a desk, pointed at the real parts, through the exerciser above.

## Files

| File | Holds |
|---|---|
| [`locator.py`](locator.py) | `Locator`, `Located`, `LocatedObject`, `LocatedOrientation`, `LocatorRefused` |
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
  [`test_realsense_perception_source.py`](../../../tests/test_realsense_perception_source.py).
