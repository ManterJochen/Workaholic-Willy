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
  the colour size.
- `StereoFrame`: `left` and `right`, both non-empty and of one size, plus `captured_at_s`.

`AnyFrame` is either. `src.camera` exports all three; `RGBDFrame` also comes from `willy`.

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

`RealSenseRGBDStreamer` streams time-synced colour and depth, aligns depth to colour through the SDK
when `align_depth_to_color` is set (the default), converts the device's depth scale to uint16
millimetres, sets the emitter, the laser power and the visual preset, and runs the spatial, temporal,
hole-filling and decimation filters the rig switches on. It imports `pyrealsense2` when it opens.

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
| `OSError` from `WebcamPairStreamer.open()` | either camera does not open; both are released | check the two ids in the rig |
| `FileNotFoundError` from `load_intrinsics` | the file is absent; K is never defaulted | point at the file the lens calibration wrote |
| `ValueError` from `load_intrinsics` | unreadable JSON, a missing or non-physical focal length, non-finite distortion | recalibrate the lens |

A missing or empty `dist` in `intrinsics.json` reads as `zeros(5)`, the "no distortion known" value a
PnP solve expects. A device's own `get_intrinsics()` and `get_distortion()` answer after `open()`; a
stored file is used only where a caller reads it.

## Status

The legend is the root README's [Status and honest scope](../../../README.md#status-and-honest-scope).

| Capability | Evidence |
|---|---|
| `RealSenseRGBDStreamer`, its logic | never touched hardware; runs against an injected fake SDK ([`test_realsense_streamer.py`](../../../tests/test_realsense_streamer.py)) |
| `RealSenseRGBDStreamer`, its frame processing | never touched hardware; runs through the real librealsense, no camera attached ([`test_realsense_sdk_contract.py`](../../../tests/test_realsense_sdk_contract.py)) |

## Files

| File | Holds |
|---|---|
| [`image_taking/frames.py`](image_taking/frames.py) | `RGBDFrame`, `StereoFrame`, `AnyFrame` |
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
