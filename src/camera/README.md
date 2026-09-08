# Camera

Open camera devices, capture frames, and front a rig-keyed runtime for the stereo and RGB-D
pipeline.

Two public entry points: **`FrameProvider`**, a rig-id-keyed facade that grabs raw or rectified
frames, and **`StereoCapturePipeline`**, which resolves the configured rigs and hands back a ready
`FrameProvider`.

## What this package guarantees

- **Frames are image arrays.** No robotics coordinates. This package does not own calibration math,
  hand-eye solving, extrinsics persistence, SE(3) geometry, robot drivers, model inference or UI
  workflows; those live in [`src/geometry/`](../geometry/README.md) and
  [`src/calibration/`](../calibration/README.md).
- **One place constructs a device.** No module outside this package builds a streamer. Every
  consumer takes a rig from a `FrameProvider`, so rig identity, open and release bookkeeping and
  intrinsics all have one owner.
- **A consumer can hold one rig.** `open_rig` / `release_rig`, and `rig(rig_id)` for a handle,
  exist because a cell holds one camera and must give back exactly the device it was handed, never
  acquiring or releasing one it was not.
- **Frame validation is fail closed.** `StereoFrame` requires non-empty numeric `left` and `right`
  with matching height and width; `RGBDFrame` requires a non-empty numeric colour image. Empty
  depth is allowed only as the documented fallback when an OpenCV backend exposes colour but not
  depth.

## Contents

| Path | Role |
| --- | --- |
| [`orchestration/frame_provider.py`](orchestration/frame_provider.py) | `FrameProvider`, the rig-keyed runtime facade over the low-level streamers, plus `RigHandle`. |
| [`pipeline/stereo_capture.py`](pipeline/stereo_capture.py) | `StereoCapturePipeline`: resolves rigs, ensures stereo calibration image sets exist, builds `StereoCam3D`, returns a `FrameProvider`. |
| [`setup/`](setup/README.md) | The streamers underneath the facade, the frame dataclasses, and the capture-quality configuration. |

## Usage

```python
from src.camera import FrameProvider, StereoCapturePipeline

# Build the provider straight from config: resolves rigs, ensures image sets, builds StereoCam3D.
provider, stereo = StereoCapturePipeline(camera_config, calibration, stereo_matcher).run()

with provider:                                  # open() / release() every streamer
    for rig_id in provider.rig_ids:
        if provider.is_stereo(rig_id):
            frame = provider.grab_rectified(rig_id)      # rectified StereoFrame
            rig = provider.get_stereo_rig_index(rig_id)  # StereoCam3D rig index
            depth_mm = stereo.compute_depth_map(frame.left, frame.right, unit="mm", rig=rig)
        else:                                    # RGB-D rig
            frame = provider.grab(rig_id)        # raw RGBDFrame
            color, depth_mm = frame.color, frame.depth
```

`run()` returns `(FrameProvider, StereoCam3D | None)`; the `StereoCam3D` is `None` only when every
target rig is RGB-D. Recording missing calibration images is interactive and blocks until the
operator ends the session.

`FrameProvider`'s surface: `open()` / `release()` (context-manager safe), `open_rig(rig_id)` /
`release_rig(rig_id)`, `grab(rig_id)`, `grab_rectified(rig_id)`, `get_intrinsics(rig_id)`,
`get_distortion(rig_id)`, `rig(rig_id)`, `rig_ids`, `open_rig_ids()`, `is_stereo(rig_id)`,
`is_rgbd(rig_id)`, `get_rig_config(rig_id)`, `get_stereo_rig_index(rig_id)`.

### One rig, for a consumer that should not know about rigs

```python
handle = provider.rig("overhead")   # a RigHandle
handle.grab()                       # -> RGBDFrame
handle.get_intrinsics()             # -> K, or None where a rig has no single pinhole matrix
handle.release()                    # gives THIS rig back; every other one keeps streaming
```

`RigHandle` answers exactly `grab()`, `get_intrinsics()`, `get_distortion()`, `release()` and
`is_open()`, which is the surface the real-camera perception adapter duck-types against. So nothing
under [`src/robot/perception/`](../robot/perception/README.md) has to learn about camera
orchestration, and a console teardown closes exactly the device it opened.

Constructing a streamer touches no device, so a provider is told about every configured rig and
opens only what it is asked to open: knowing a rig costs nothing, holding one is a deliberate act.

**Rig selection**: `camera.cameras.primary_rig_id` names the rig a CELL opens, and
`build_real_components` refuses a primary that is disabled or is not an RGB-D rig. The stereo
calibration pipeline is a tool over the whole catalogue rather than a cell: an explicit `rig_id`
argument wins there, and otherwise it works on every enabled rig that answers a probe.

## The two RGB-D backends

Selected per rig by `RGBDDeviceRigConfig.rgbd_backend`; `FrameProvider` builds the matching
streamer.

| `rgbd_backend` | Streamer | Depth source |
| --- | --- | --- |
| `opencv` (default) | `OpenCvRGBDStreamer` | A generic `cv2.VideoCapture` with OpenNI `CAP_OPENNI_*` channels, and no vendor SDK. Returns empty depth explicitly on a device that exposes colour only. |
| `realsense` | `RealSenseRGBDStreamer` | Intel RealSense through `pyrealsense2`: hardware-aligned depth, device depth scale to uint16 millimetres, emitter, laser power and visual preset, a post-processing filter chain, and a colour-intrinsics accessor. |

`pyrealsense2` is pinned in `requirements.txt` and its import is deferred, so the library runs on a
machine without it. **The RealSense driver has never been exercised against a physical device
here**: its logic is covered with an injected fake SDK, and the real round trip remains the same
gap [`src/robot/perception/`](../robot/perception/README.md) names as the largest one in the stack.

## Traps

**`FrameProvider` only routes frames.** It does not crop, split, resize or configure quality; the
streamers under [`setup/`](setup/README.md) do. An unknown rig id raises `UnknownCameraRigError`;
grabbing a rig that is not open raises `FrameProviderStateError`; `grab_rectified` on an RGB-D rig
raises `ValueError`, and on a stereo rig with no `StereoCam3D` raises `RuntimeError`;
`get_stereo_rig_index` on an RGB-D rig raises `KeyError`.

**A stereo rig has no single camera matrix**, so `get_intrinsics` answers `None` for one: its
geometry lives in `StereoCam3D`. Both accessors require the rig to be open.

**RGB-D rigs never reach stereo calibration.** Stereo rig config order is preserved and is what
`get_stereo_rig_index` returns. Each rig keeps isolated capture and calibration artefacts under its
own `calibration_paths.base_dir`.

## See also

- [`setup/`](setup/README.md), the streamers and the frame dataclasses
- [`src/calibration/`](../calibration/README.md), for stereo calibration, reconstruction, hand-eye
  and extrinsics
- [`src/geometry/`](../geometry/README.md), where robotics coordinates live
- [`src/config/`](../config/README.md), which validates the camera, rig, calibration and matcher
  config
- [`config/camera/cam.yaml`](../../config/camera/cam.yaml), the rigs themselves
