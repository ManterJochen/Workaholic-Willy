# Live-camera perception

One RGB-D frame, grounded and segmented, turned into one `PerceptionFrame`. This is the
real-hardware twin of the Isaac vision source, and the piece the whole vision-driven pick path waits
on.

No physical camera has ever fed it. Every line is exercised against a fake streamer and fake models.

## Why this package exists, and why it lives here

The pick loop consumes a [`PerceptionSource`](../grasping/types/perception.py), which is anything
with `acquire() -> PerceptionFrame`. The sim ships two, `IsaacVisionPerceptionSource` and its
multi-object twin in `willy_sim/perception/vision.py`, and both step a simulator. A real cell
needs an adapter that does not, and `RealSenseVisionPerceptionSource` is it.

It lives under `robot/` and not under `camera/` because the dependency edge only ever runs from
`robot` to `camera`. An adapter that needs both a camera streamer and `robot.grasping`'s
`PerceptionFrame` cannot sit in the camera package without inverting that edge.

Its streamer, detector and segmenter are injected and duck-typed, exactly like the sim source, so
this module imports with neither `pyrealsense2` nor torch present. In a real cell what gets injected
as the streamer is a `RigHandle` from the rig's `Camera` owner in the camera package: the handle
answers the same `grab()`, `get_intrinsics()` and `release()`, so the adapter did not have to change.
Every camera in the process has one owner: a second opener is refused naming the holder, every grab
runs under the rig's lock, and a consumer can give back the rig it was handed without touching any
other.

## Contents

| File | Role |
| --- | --- |
| [`locator.py`](locator.py) | `Locator.from_parts(camera=, backend=, tool_pose=, attempts=, tool_frame=)` and `Locator.from_config(robot_cfg, models_cfg, camera=)`, with one verb, `locate(prompt) -> Located`. It runs the pick frame's RealSense source over an open camera the caller owns and places every grounded object in BASE (`LocatedObject`: label, score, box, mask, points, per-axis median centre, and an orientation marked unmeasured), stamped at the shutter. `Located.keep_out(i)` is the `SegmentationOffer` a `keeping_out` block holds while user code reaches for object i, and `Located.scene(i, robot_config)` is object i as the grasp `Scene`, the other objects its obstacles, whose candidates' `pose()` and `grip_width_mm` are what `Robot.pick` takes. It refuses a rig with no depth, a rig with no calibration (naming its key), and a wrist rig with no TCP reader or whose calibration was not solved against the cell's flange to TCP, and it raises `PerceptionFrameMoved` when a wrist camera cannot take a still frame. An empty result says that it reads the same as a failed detector. It imports no driver, pick loop, camera package, models or torch. |
| [`realsense_source.py`](realsense_source.py) | `RealSenseVisionPerceptionSource`: grab one RGB-D frame, ground and segment it, emit a `PerceptionFrame` whose depth is the depth the sensor measured and whose `timestamp` is the shutter time (the camera owner's `captured_at_s`, else a clock read just before the grab). It carries the mask-completion lever, whose default is `none`. `stamp_tool_pose_with(reader, *, motion_tolerance, attempts)` binds a wrist camera: the TCP is read before and after the grab, a frame taken while the tool moved beyond the rig's shutter tolerance is grabbed again without the warm-ups, and after its attempts the acquire raises `PerceptionFrameMoved`. `build_real_cell` binds the primary and every fused wrist source, and refuses a rig whose calibration was not solved against the declared flange to TCP. `set_prompt(prompt, *, object_labels=())` changes the phrase and the label map between frames, with no reopen and no reload, and refuses an empty phrase; `prompt` and `object_labels` read them. |
| [`viewfinder.py`](viewfinder.py) | The peek capability: `ColourPeekable`, `peek_color_of`, `colour_source_kind`, `COLOUR_SOURCE_KINDS`. One colour image, with no models, no simulation and no pipeline state. |
| [`mask_completion.py`](mask_completion.py) | The three mask-fill policies, the shared threshold, and the reasoning behind the default. |
| [`__main__.py`](__main__.py) | The bench exerciser: open a real RGB-D rig, grab, detect, and print intrinsics and per-mask depth-hole statistics. No robot. |

## Usage

```bash
# Bench check. It needs a real RealSense device and the perception models on a GPU box.
python -m src.robot.perception --prompt "a red cube"
python -m src.robot.perception --prompt "cube ; screwdriver ; mug"   # multi-phrase clutter
python -m src.robot.perception --rig realsense_d435 --warmup 10
```

```python
# In a runner: build the injected pieces, then hand the source to the service.
from src.camera.orchestration.frame_provider import FrameProvider
from src.models.perception_spec import PerceptionSpec
from src.robot.perception import RealSenseVisionPerceptionSource

provider = FrameProvider(list(app_cfg.camera.cameras.rigs))
provider.open_rig(rig_id)
source = RealSenseVisionPerceptionSource(
    streamer=provider.rig(rig_id),
    backend=PerceptionSpec.from_config(app_cfg.models).build(),
    prompt="a red cube",
    # intrinsics=None uses the device's factory K; pass a calibrated K to override.
)
frame = source.acquire()
```

`build_real_components` in `robot/execution/autonomous_grasp/cells.py` builds exactly this for a
physical cell, and `build_real_cell` wraps it into a whole service. The `backend` argument is what
both construction sites pass; the alternative, a `detector` plus `segmenter` pair, stays supported
for callers that assemble models by hand.

The exerciser selects its rig on the same predicate the cell uses, `source == "rgbd"`, so a bench run
cannot open a different camera from the one the cell opens and still count as evidence about the
cell.

From user code, the locator is the way from seeing a part to picking it. The package exports
`Locator`, `Located`, `LocatedObject`, `LocatedOrientation` and `LocatorRefused`:

```python
from src.camera import Camera
from src.robot.execution.robot import Robot
from src.robot.perception import Locator

with Camera.from_config(app.camera) as camera:            # the primary rig, released however the block ends
    robot = Robot.from_config(app.robot, cameras=[camera])   # the camera world the arm plans in
    locator = Locator.from_config(app.robot, app.models, camera=camera)
    with robot.connected():
        located = locator.locate("a red cube")
        best = located.scene(0, app.robot).grasps().best      # BASE candidates, the other objects as obstacles
        if best is not None:                                  # +Z the approach, +X the closing axis
            print(robot.pick(best.pose(), best.grip_width_mm, keep_out=located.keep_out(0)).render())
```

A wrist rig also takes `tool_pose=robot.arm.get_tcp_pose` on the locator.

## What it does with a frame

| Step | Behaviour | Why not the obvious alternative |
| --- | --- | --- |
| Intrinsics | The streamer's factory K by default; a calibrated K overrides it. `source.intrinsics_source` records which was used, as `"factory"` or `"override"`. | Silently preferring one would make a calibration mistake invisible. |
| Top-referenced depth | The grasp depth over each mask is the nearest real surface, meaning the minimum over the mask while skipping zero-valued depth holes, plus `grasp_top_penetration_mm` (3.0 mm by default). A mask that is entirely holes keeps its raw depth. | Filling a hole-only mask would fabricate a plane and the failure would become invisible instead of visible downstream. |
| Mask completion | The default is `none`, so the segmenter's silhouette reaches the calculator exactly as segmented. | See below. |
| Label canonicalisation | Optional. Pass `object_labels` to map free-form detector phrases onto your known object names. | So an exact `seg.label == target` match works downstream. |

`acquire()` publishes the depth the sensor measured, under a mask as everywhere else. The bench
exerciser counts its depth-hole fraction on the depth the adapter consumed, tapped at the streamer,
rather than on `frame.depth_map`.

## Why filling a mask to its box is not the default

The rule replaced a mask with its detection box whenever the mask covered less than
`DEFAULT_MIN_FILL_RATIO` (0.8) of the box area. It was written for a real failure, since a
segmentation mask that drops one end of a long box shifts the centroid.

Measured against ground-truth masks as a control, it fires on complete masks at the same rate as on
predicted ones. Identical firing on complete masks means it was never detecting incompleteness; it
was detecting non-rectangularity. A circle fills pi/4 = 0.785 of its bounding box and is already
under the threshold. And a filled mask is an axis-aligned rectangle whose principal axis lies on an
image axis by construction, which is where the closing axis then comes from. In paired grasp-outcome
measurement, `axis_aligned_box` lost several times more grasps than it won, and `oriented_box` was
statistically indistinguishable from `none`.

So it is removed as a default and kept as a capability: pass `mask_completion=` to either
perception source. [`mask_completion.py`](mask_completion.py) carries the full reasoning, the bound
on the rescue that was given up, and the targeted fix worth building instead.

## Peeking is not perceiving

`acquire()` is the wrong way to answer "what is the camera seeing?", twice over. On a real cell it
discards warm-up frames and then runs a detector and a segmenter on the GPU, so a browser tab
polling it would run a detector several times a second beside the pick that needs it. On the Isaac
source it steps the simulator, so the same tab would advance the physics clock of the cell it is
watching.

So [`viewfinder.py`](viewfinder.py) declares a separate, deliberately narrow capability:

```python
peek_color() -> HxWx3 BGR uint8 | None      # no models, no simulation, no pipeline state
colour_source_kind = "camera" | "synthetic" | "unknown"
```

A source that cannot honour all three constraints does not implement it. `peek_color_of` then
returns `None`, and the caller says that this cell cannot show a live view and why, which is a true
sentence, instead of showing something stale or fabricated. The Isaac sources deliberately do not
implement it.

**The kind is declared, never inferred.** A rehearsal scene and a camera frame are both HxWx3 BGR
arrays. A console that guesses from the shape and prints "live from the camera" over a picture this
process drew is making a claim about a room with no camera in it. An undeclared source is `unknown`
and is described neutrally; only a source that says `camera` earns the word.

**The one allowed side effect, named.** On a real device, peeking consumes one frameset, because
there is no way to look at a stream without taking a frame from it. It is harmless here because
`acquire()` opens by discarding `warmup_grabs` frames, five by default, so a viewer's frame lands in
that discarded prefix. Two grabs never interleave on one device: every grab goes through the rig's
owner, `Camera`, under the rig's lock, so a peek during an `acquire()` waits its turn.
[`api/viewfinder.py`](../../../api/viewfinder.py) still stands down while a run is active: while a run
owns the cell it serves the pick's own overlay render instead of touching the camera, because that
overlay is the more informative picture then.

## Honesty

The behaviour under a real segmentation mask on real depth is the bench risk this package carries.
Real depth returns zeros inside a mask and leaks at mask edges. The two depth robustness rules here
are carried over from simulation debugging: they are the right starting point, not a proven answer
on a physical device.

This is the largest untested surface in the stack. The arm half has controller software to run
against; the camera half has no equivalent. It can be de-risked without a robot, with a camera on a
desk pointed at the real objects and run through the real models, which is exactly what the
exerciser above is for.

## See also

- [robot/grasping/types](../grasping/types/perception.py), the `PerceptionFrame` and
  `PerceptionSource` contract
- [camera setup and image taking](../../camera/setup/README.md), the streamer a `RigHandle` wraps
- [models](../../models/README.md), `PerceptionSpec` and the perception backends a cell builds
  through. This exerciser deliberately stays on the component doors, because it exists to prove that
  a camera and two models work before a cell exists.
- [willy_sim](../../willy_sim/README.md), the simulated source this mirrors
- [real_cell](../execution/real_cell/README.md), the runner that constructs this for a physical cell
