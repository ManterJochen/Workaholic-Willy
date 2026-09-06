# Camera setup and image taking

Camera configuration and frame acquisition: the raw pixels every perception stage starts from.

This package sits at the bottom of the perception layer. It hands frames up to `FrameProvider` and
`StereoCapturePipeline`, and deliberately does not own stereo or hand-eye calibration math,
geometry transforms, model inference, or robot APIs.

## What this package guarantees

- **Every device goes through one quality path.** `configure_camera_for_quality()` is the single
  place that sets pixel format, resolution, frame rate and the optional manual exposure, gain and
  white balance. It returns the values read back off the device, because a `cap.set(...)` that
  reports success does not mean the backend applied it.
- **Validation happens in `__post_init__`.** A frame that does not satisfy its dataclass cannot be
  constructed, so nothing downstream has to re-check.
- **Streamers are context managers** and return frames programmatically, owning no GUI.

## Contents

| Path | Role |
| --- | --- |
| [`quality.py`](quality.py) | `configure_camera_for_quality()`: format, resolution, FPS, optional manual exposure, gain and white balance, auto-feature disabling, buffer size, warmup frames. Returns what OpenCV read back. |
| [`image_taking/frames.py`](image_taking/frames.py) | `StereoFrame` (`left` / `right`) and `RGBDFrame` (`color` / `depth`), the only frame dataclasses in the camera package, plus the `AnyFrame` union. |
| [`image_taking/webcam.py`](image_taking/webcam.py) | `WebcamPairStreamer`, two devices, returns a `StereoFrame`. |
| [`image_taking/single.py`](image_taking/single.py) | `SingleDeviceStreamer`, one side-by-side or top-bottom device; owns the crop, split and resize of a combined stereo frame. |
| [`image_taking/rgbd.py`](image_taking/rgbd.py) | `OpenCvRGBDStreamer`, `RealSenseRGBDStreamer` and the `RGBDStreamerProtocol` they both satisfy. |
| [`image_taking/intrinsics.py`](image_taking/intrinsics.py) | `load_intrinsics(path)`: read a stored `intrinsics.json` back into `(K, dist)`. Not re-exported from the package; import it by its full path. |
| [`devices/webcam.py`](devices/webcam.py) | `StereoVisionCalibrationWebcams`, interactive PNG capture from two USB cameras. |
| [`devices/stereocamera.py`](devices/stereocamera.py) | `StereoVisionCalibrationSingleDevice`, interactive PNG capture from one combined-frame device. |

## Usage

```python
from src.camera.setup.image_taking import RGBDFrame, StereoFrame, WebcamPairStreamer

with WebcamPairStreamer(config) as streamer:
    frame: StereoFrame = streamer.grab()   # .left / .right BGR arrays
```

| Streamer | Config | Returns |
| --- | --- | --- |
| `WebcamPairStreamer` | `WebcamPairRigConfig` | `StereoFrame` |
| `SingleDeviceStreamer` | `SingleDeviceRigConfig` | `StereoFrame` |
| `OpenCvRGBDStreamer` | `RGBDDeviceRigConfig` with `rgbd_backend: opencv` | `RGBDFrame` |
| `RealSenseRGBDStreamer` | `RGBDDeviceRigConfig` with `rgbd_backend: realsense` | `RGBDFrame` |

Both RGB-D streamers implement `RGBDStreamerProtocol` (`open` / `release` / `is_opened` / `grab`),
and `FrameProvider` picks between them from config. Build them through the provider rather than
directly: it is the one owner of rig identity and device lifecycle.

## Notes and traps

**Frame validation.** `StereoFrame` requires non-empty `left` and `right` with matching height and
width. `RGBDFrame` requires a non-empty `color`; `depth` may be empty only when the backend cannot
expose a depth stream, and otherwise must match the colour size.

**`WebcamPairStreamer` grabs both cameras before retrieving either** (`grab()`, `grab()`, then
`retrieve()`, `retrieve()`), to shrink the temporal offset between left and right. That ordering is
the whole reason the class exists rather than two independent captures.

**`OpenCvRGBDStreamer` uses no vendor SDK.** It reads colour and depth from a generic
`cv2.VideoCapture` through the OpenNI `CAP_OPENNI_*` channels, and returns empty depth explicitly
when the backend cannot supply it, rather than a plausible-looking zero array.

**`RealSenseRGBDStreamer` drives an Intel RealSense through `pyrealsense2`:** hardware-aligned
depth, device depth scale converted to uint16 millimetres, emitter, laser power and visual preset
control, a spatial, temporal, hole-filling and decimation post-processing chain, and a
colour-intrinsics accessor. The import is deferred, so the module stays importable without the SDK.
Its logic is covered with an injected fake `rs` module; **the real-device round trip has never been
exercised here.**

**Intrinsics have two sources and neither is imposed.** A device's own `get_intrinsics()` and
`get_distortion()` are live after `open()`; a bench-calibrated `intrinsics.json` is read by
`load_intrinsics`. An absent file raises `FileNotFoundError` rather than defaulting K, and a
malformed one raises `ValueError`; a missing or empty `dist` yields `zeros(5)`, the "no distortion
known" default a PnP solve expects.

**Calibration capture classes save raw images** with no preview overlays, enforce `min_pairs` and
`max_pairs`, and go through `configure_camera_for_quality()`, the same setup path the runtime
streamers use.

**Units.** Arrays keep the dtype OpenCV returns; RGB-D depth is millimetres wherever the backend
supplies it. No robotics transforms live here.

## See also

- [`src/camera/`](../README.md), the `FrameProvider` and `StereoCapturePipeline` above this layer
- [`src/calibration/`](../../calibration/README.md), for stereo and hand-eye calibration math
- [`src/geometry/`](../../geometry/README.md), for millimetre transforms and pose math
- [`src/robot/perception/`](../../robot/perception/README.md), the adapter that turns these frames
  into a perception frame
- [`config/camera/cam.yaml`](../../../config/camera/cam.yaml), the rig definitions these streamers
  are built from
