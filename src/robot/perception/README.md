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
as the streamer is a `RigHandle` from the camera package's `FrameProvider`: the handle answers the
same `grab()`, `get_intrinsics()` and `release()`, so the adapter did not have to change, and every
camera in the process has exactly one owner. A consumer can give back the rig it was handed without
touching any other.

## Contents

| File | Role |
| --- | --- |
| [`realsense_source.py`](realsense_source.py) | `RealSenseVisionPerceptionSource`: grab one RGB-D frame, ground and segment it, emit a `PerceptionFrame`. It carries the top-referenced depth rule and the mask-completion lever. |
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

## What it does with a frame

| Step | Behaviour | Why not the obvious alternative |
| --- | --- | --- |
| Intrinsics | The streamer's factory K by default; a calibrated K overrides it. `source.intrinsics_source` records which was used, as `"factory"` or `"override"`. | Silently preferring one would make a calibration mistake invisible. |
| Top-referenced depth | The grasp depth over each mask is the nearest real surface, meaning the minimum over the mask while skipping zero-valued depth holes, plus `grasp_top_penetration_mm` (3.0 mm by default). A mask that is entirely holes keeps its raw depth. | Filling a hole-only mask would fabricate a plane and the failure would become invisible instead of visible downstream. |
| Mask completion | The default is `none`, so the segmenter's silhouette reaches the calculator exactly as segmented. | See below. |
| Label canonicalisation | Optional. Pass `object_labels` to map free-form detector phrases onto your known object names. | So an exact `seg.label == target` match works downstream. |

A trap worth stating plainly: `acquire()` overwrites the depth inside every mask with the grasp
plane, so `PerceptionFrame.depth_map` is not a sensor reading there. The bench exerciser counts its
depth-hole fraction on the depth the adapter consumed, not on `frame.depth_map`, because the same
count taken on the output can only ever come out at 0 or 100 percent.

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
that discarded prefix. What is not safe is peeking while a pick is mid-acquire: both would call
`grab()` on one unsynchronised pipeline, and the camera package holds no lock.
[`api/viewfinder.py`](../../../api/viewfinder.py) enforces that exclusion on run state, and while a
run owns the cell it serves the pick's own overlay render instead of touching the camera.

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
