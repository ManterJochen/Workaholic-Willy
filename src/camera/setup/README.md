# Camera drivers and frames (`src/camera/setup`)

The device drivers underneath `Camera`, the two frame types they return, and the function that sets an
OpenCV device's capture quality. You reach them through [`Camera`](../README.md), which builds the
right driver from a rig's config; build one directly only for a bench test or a device double.

```python
import numpy as np
from willy import RGBDFrame

frame = RGBDFrame(color=np.zeros((480, 640, 3), np.uint8), depth=np.zeros((480, 640), np.uint16))
print(frame.captured_at_s)   # None: only a frame taken through a Camera owner carries its grab time
```

A frame that breaks its rules cannot be built, so nothing downstream checks it again. Building a driver
touches no device until `open()`, and no driver opens a window: frames come back to the caller.

## The frames

- `RGBDFrame`: `color` (BGR) and `depth` (millimetres), plus `captured_at_s`. The colour image must not
  be empty. Depth may be empty only when the driver cannot reach a depth stream, and otherwise matches
  the colour size. `research` is `None`, except from a RealSense rig that records for research
  (`realsense.record_for_research`): then a `ResearchCapture`, the frame's two infrared images, its
  depth as the sensor sent it (device units, before the filters and the alignment) and its colour
  metadata, copied, with the camera's `CameraFacts` (device, settings, lenses, extrinsics), read once
  when the camera opened.
- `StereoFrame`: `left` and `right`, both non-empty and of one size, plus `captured_at_s`.

`AnyFrame` is either. `src.camera` exports all three; `RGBDFrame` also comes from `willy`.
`ResearchCapture` and `CameraFacts` come from `src.camera.setup.image_taking`.

## The nouns

| Noun | Built by | Verb | Returns |
|---|---|---|---|
| `OpenCvRGBDStreamer` | a rig with `source: rgbd`, `rgbd_backend: opencv` | `open()`, `grab()`, `release()` | `RGBDFrame`; empty depth where the device exposes colour only |
| `RealSenseRGBDStreamer` | a rig with `source: rgbd`, `rgbd_backend: realsense` | `open()`, `grab()`, `release()` | `RGBDFrame`, depth in millimetres, aligned to colour by default |
| `WebcamPairStreamer` | a rig with `source: webcam_pair` | `open()`, `grab()`, `release()` | `StereoFrame` from two USB cameras |
| `SingleDeviceStreamer` | a rig with `source: single_device` | `open()`, `grab()`, `release()` | `StereoFrame` cut from one side-by-side or top-bottom image |
| `load_intrinsics` | `load_intrinsics(path)`, from `image_taking/intrinsics.py` | | `(K, dist)` read from a stored `intrinsics.json` |

Every driver takes its rig config and is a context manager. The two RGB-D drivers satisfy
`RGBDStreamerProtocol` (`open`, `release`, `is_opened`, `grab`), and `create_streamer` in
[`orchestration/camera.py`](../orchestration/camera.py) picks the driver a rig names.

`WebcamPairStreamer` grabs both cameras before it retrieves either, so the left and right images are
taken as close together in time as two devices allow; that ordering is why the class exists rather than
two independent captures.

`RealSenseRGBDStreamer` streams time-synced colour and depth. It runs the decimation, spatial,
temporal and hole-filling filters the rig switches on, in librealsense's recommended order (spatial and
temporal on disparity, between the SDK's two disparity transforms), then aligns depth to colour through
the SDK when `align_depth_to_color` is set (the default), and converts depth to uint16 millimetres with
the depth scale the device reads back. It writes the visual preset first and the emitter, the laser
power and the depth units over it, and refuses to open when the device reads back other depth units than
the rig configures. It imports `pyrealsense2` when it opens.

When it opens it logs, to `logs/camera/rgbd.log` and the console, the camera's name, serial, firmware
and USB link, the depth mode and that mode's Min-Z (`realsense_min_depth_mm`, Intel's figures: a D415
about 450 mm at 1280 x 720 and about 310 mm at 848 x 480), and warns for a D415 at 1280 x 720 and for a
USB 2 link. A request that does not start is raised with the RealSense cameras the SDK sees and the USB
link each is on.

It writes the rig's `realsense.color` block to the colour sensor (the owner, 2026-10-09): auto exposure,
exposure, gain, auto white balance, white balance, in that order, each auto mode before its value,
because librealsense writes its default exposure as auto exposure goes off. Every value is checked
against the range the sensor offers before the first is written, and a null key writes nothing. The
exposure is in the D400 colour sensor's own unit, 100 microseconds; on Windows librealsense's Media
Foundation backend holds it as the nearest power of two of a second (39, 78, 156, 312, 625). Once the
warm-up frames have run it reads the five back, logs them (`color_held`), and warns for a value the
sensor holds otherwise. A colour sensor that does not offer an option the block sets, a value outside
its range, and a camera whose colour comes off its depth imagers (a D405) refuse the open, naming the
option and the camera, before anything is written.

On a rig with `realsense.record_for_research` it asks for both infrared images beside colour and depth
(`rs.stream.infrared` 1 and 2, Y8, at the depth resolution and frame rate), reads the camera's facts once
(`CameraFacts`: every option of both sensors, the filters with their options, each stream's lens and
the extrinsics depth to colour, left infrared to colour and left to right infrared, in millimetres), and
hands every frame its `ResearchCapture`, taken off the frameset before any filter runs. A camera that
cannot stream the infrared images refuses the open with a sentence. With the key off, the streams asked
and the frames are what they were.

`camera_moved()` drops the temporal filter's history, so the next frame holds only depth seen from where
the camera is now. On a camera the arm carries (a rig with a `body`, or eye_in_hand extrinsics) the
driver also drops it after a pause between two grabs longer than a burst of back-to-back grabs takes,
0.25 s or three frame periods; its `clock` argument is the monotonic clock it reads.

## Capture quality

`configure_camera_for_quality()` in [`quality.py`](quality.py) sets an OpenCV device's pixel format,
resolution and frame rate, and optionally a manual exposure, gain and white balance. It returns the
values read back off the device, because a `cap.set(...)` that reports success does not mean the device
applied it. The webcam pair, the single device, the colour stream of the OpenCV RGB-D driver and the
two calibration capture classes all go through it; the RealSense driver sets its streams through the
SDK instead.

The calibration capture classes in [`devices/`](devices/) save raw images for stereo calibration, with
no preview overlays, and hold the rig's `min_pairs` and `max_pairs`.

## What it refuses

| Refusal | When | What to do |
|---|---|---|
| `ValueError` from a frame | an empty colour image or eye, or sizes that differ | fix the driver or the double that built it |
| `TypeError` from a frame | an array with a non-numeric dtype | hand it image arrays |
| `ImportError` naming `pyrealsense2` | a RealSense rig opens on a machine without the SDK | `pip install -r requirements.txt` |
| `RuntimeError` from `RealSenseRGBDStreamer.open()`, listing cameras | the SDK cannot start the requested streams; the message names each camera it sees and its USB link | connect the camera over USB 3, fix the serial, or ask for a mode it offers |
| `RuntimeError` from `RealSenseRGBDStreamer.open()`, naming `depth_units_m` | the device reads back other depth units than configured; the pipeline is stopped | remove the key, or write a value the device takes |
| `RuntimeError` from `RealSenseRGBDStreamer.open()`, naming `realsense.color.<key>` | the colour sensor does not offer the option, the value is outside its range, or the colour comes off the depth imagers; nothing was written and the pipeline is stopped | remove the key, or write a value in the range named |
| `RuntimeError` from `RealSenseRGBDStreamer.open()`, naming `record_for_research` | the camera does not stream both infrared images at this mode, or its link cannot carry them; the pipeline is stopped | switch `record_for_research` off, or connect over USB 3 |
| `OSError` from `WebcamPairStreamer.open()` | either camera does not open; both are released | check the two ids in the rig |
| `FileNotFoundError` from `load_intrinsics` | the file is absent; K is never defaulted | point at the file the lens calibration wrote |
| `ValueError` from `load_intrinsics` | unreadable JSON, a missing or non-physical focal length, non-finite distortion | recalibrate the lens |

A missing or empty `dist` in `intrinsics.json` reads as `zeros(5)`, the "no distortion known" value a
PnP solve expects. A device's own `get_intrinsics()` and `get_distortion()` answer after `open()`; a
stored file is used only where a caller reads it.

## Status

The legend is the guide's [evidence levels](../../../docs/guide/README.md#what-verified-means-on-these-pages).

| Capability | Evidence |
|---|---|
| `RealSenseRGBDStreamer`, its logic | run on a physical cell: a wrist D415 on a UR10 (CB3); in the suite it runs against an injected fake SDK ([`test_realsense_streamer.py`](../../../tests/test_realsense_streamer.py)) |
| `RealSenseRGBDStreamer`, its frame processing | run on a physical cell: the same D415; in the suite it runs through the real librealsense, no camera attached ([`test_realsense_sdk_contract.py`](../../../tests/test_realsense_sdk_contract.py), [`test_a_moved_wrist_camera_forgets_the_last_pose.py`](../../../tests/test_a_moved_wrist_camera_forgets_the_last_pose.py)) |
| `RealSenseRGBDStreamer`, what it says at open | never touched hardware; Min-Z is Intel's figure, and the messages run against a fake SDK ([`test_a_realsense_says_what_it_opened.py`](../../../tests/test_a_realsense_says_what_it_opened.py)) |
| `RealSenseRGBDStreamer`, the colour block and the research recording | never touched hardware; pinned against a fake SDK, the names and the extrinsics convention against the real librealsense ([`test_the_colour_sensor_holds_what_its_rig_asks.py`](../../../tests/test_the_colour_sensor_holds_what_its_rig_asks.py), [`test_a_rig_recording_for_research_keeps_its_infrared_images.py`](../../../tests/test_a_rig_recording_for_research_keeps_its_infrared_images.py)) |

## Files

| File | Holds |
|---|---|
| [`image_taking/frames.py`](image_taking/frames.py) | `RGBDFrame`, `StereoFrame`, `AnyFrame`, and what a frame recorded for research carries: `ResearchCapture`, `CameraFacts` |
| [`image_taking/rgbd.py`](image_taking/rgbd.py) | `OpenCvRGBDStreamer`, `RealSenseRGBDStreamer`, `RGBDStreamerProtocol` |
| [`image_taking/webcam.py`](image_taking/webcam.py) | `WebcamPairStreamer` |
| [`image_taking/single.py`](image_taking/single.py) | `SingleDeviceStreamer`, which owns the crop, split and resize of a combined stereo image |
| [`image_taking/intrinsics.py`](image_taking/intrinsics.py) | `load_intrinsics`; import it by this path, the package does not re-export it |
| [`quality.py`](quality.py) | `configure_camera_for_quality` |
| [`devices/webcam.py`](devices/webcam.py) | `StereoVisionCalibrationWebcams`, calibration images from two USB cameras |
| [`devices/stereocamera.py`](devices/stereocamera.py) | `StereoVisionCalibrationSingleDevice`, calibration images from one combined device |

## Details

- The owner above these drivers, the catalogue and the stereo pipeline: [`src/camera/`](../README.md).
- The rig definitions the drivers are built from: [`config/camera/cam.yaml`](../../../config/camera/cam.yaml).
- Stereo and hand-eye calibration: [`src/calibration/`](../../calibration/README.md).
- The adapter that turns these frames into a perception frame: [`src/robot/perception/`](../../robot/perception/README.md).
