# Camera

Open camera devices, capture frames, and give every rig one owner, for the stereo and RGB-D
pipeline.

Three public entry points: **`Camera`**, the owner of one rig's device, which is its only opener,
serialises its grabs and stamps its frames; **`FrameProvider`**, a rig-id-keyed catalogue that opens
each of its rigs through a `Camera` and grabs raw or rectified frames; and
**`StereoCapturePipeline`**, which resolves the configured rigs and hands back a ready
`FrameProvider`.

## What this package guarantees

- **Frames are image arrays.** No robotics coordinates. This package does not own calibration math,
  hand-eye solving, extrinsics persistence, SE(3) geometry, robot drivers, model inference or UI
  workflows; those live in [`src/geometry/`](../geometry/README.md) and
  [`src/calibration/`](../calibration/README.md).
- **One place constructs a device.** No module outside
  [`orchestration/camera.py`](orchestration/camera.py) builds a streamer. Every consumer takes its
  rig from a `Camera` owner, directly or through a `FrameProvider`, so rig identity, open and release
  bookkeeping and intrinsics all have one owner.
- **One owner per device.** A second owner of a device that is already open in this process is
  refused with `CameraBusy`, naming the holder, before the device is touched.
- **A consumer can hold one rig.** `open_rig` / `release_rig`, and `rig(rig_id)` for a handle,
  exist because a cell holds one camera and must give back exactly the device it was handed, never
  acquiring or releasing one it was not.
- **Frame validation is fail closed.** `StereoFrame` requires non-empty numeric `left` and `right`
  with matching height and width; `RGBDFrame` requires a non-empty numeric colour image. Empty
  depth is allowed only as the documented fallback when an OpenCV backend exposes colour but not
  depth. `captured_at_s` defaults to None.

## Contents

| Path | Role |
| --- | --- |
| [`orchestration/camera.py`](orchestration/camera.py) | `Camera`: one owner per rig, the process registry that refuses a second opener (`CameraBusy`), the rig selection a cell uses (`select_rig`), and the one place a device streamer is constructed (`create_streamer`). A `with` block opens an owner and releases it. |
| [`orchestration/frame_provider.py`](orchestration/frame_provider.py) | `FrameProvider`, the rig-keyed catalogue over the owners, plus `RigHandle`. |
| [`pipeline/stereo_capture.py`](pipeline/stereo_capture.py) | `StereoCapturePipeline`: resolves rigs, ensures stereo calibration image sets exist, builds `StereoCam3D`, returns a `FrameProvider`. |
| [`setup/`](setup/README.md) | The streamers underneath the owners, the frame dataclasses, and the capture-quality configuration. |

## Usage

### `Camera`: one rig, one owner

```python
from src.camera import Camera, CameraRefused, RigNotCalibrated

with Camera.from_config(app_cfg.camera) as camera:   # opened here, released however the block ends
    frame = camera.grab()                           # RGBDFrame, stamped at the grab
    K = camera.get_intrinsics()                     # None where the backend reports no pinhole matrix
    calibration = camera.calibration()              # RigNotCalibrated names the key when the rig declares none
```

The same owner without a block, for a holder that outlives one:

```python
camera = Camera.from_config(app_cfg.camera)   # the primary rig; rig_id="wrist" names another
camera.open()                                  # CameraBusy if another owner in this process holds the device
handle = camera.handle()                       # a RigHandle, shaped like a streamer
frame = handle.grab()                          # RGBDFrame; frame.captured_at_s is the host time of the grab
handle.release()                               # gives the device back; never raises
```

`src.camera` exports `Camera` and its refusals: `CameraRefused`, `CameraBusy`, `CameraNotOpen`, and
the two raises of `camera.calibration()`, `RigNotCalibrated` and `RigCalibrationError` (defined in
[`calibration/rig_calibration.py`](../calibration/rig_calibration.py)). A `with` block releases the
owner it holds, also one that was open before the block began.

`Camera.from_config(camera_section, *, rig_id=UNSET, open_disabled=UNSET)` refuses with
`CameraRefused`, whose `reason` is `unknown`, `disabled` or `not_rgbd`: a rig the section does not
configure, a rig with `enabled: false`, and a rig with no depth channel. Only the bench exerciser
passes `open_disabled=True`, and it prints that it did. The cell (`build_real_components`), the
calibration command and the perception exerciser open their cameras this way, so the three say one
thing about a rig.

**Identity is the device, not the rig name.** A RealSense rig is its serial, and a RealSense with no
serial collides with every open RealSense, because the SDK then binds whichever camera it offers
first. An OpenCV RGB-D rig and a single-device stereo rig are their `device_index`. A webcam pair is
both of its ids, and an unset id collides with every open video device. A second owner of an open
device is refused before the device is touched, naming the holder. The registry holds live owners
only: an owner that no longer exists holds no claim, and releasing is still what gives the device
itself back.

**Every grab is serialised and stamped.** `grab`, `get_intrinsics` and `get_distortion` through any
handle of an owner run under that rig's lock, and a frame from the owner carries `captured_at_s`,
the host `time.time()` read just before the device grab. A streamer used directly returns
`captured_at_s=None`.

### `FrameProvider`: the catalogue

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
`get_distortion(rig_id)`, `rig(rig_id)`, `camera(rig_id)`, `rig_ids`, `open_rig_ids()`,
`is_stereo(rig_id)`, `is_rgbd(rig_id)`, `get_rig_config(rig_id)`, `get_stereo_rig_index(rig_id)`.
Stereo capture and hand detection use it. Every rig it opens is held by a `Camera`, so a catalogue
and a `Camera` elsewhere in the process never both hold one device.

### One rig, for a consumer that should not know about rigs

```python
handle = provider.rig("overhead")   # a RigHandle; camera.handle() hands out the same kind
handle.grab()                       # -> RGBDFrame
handle.get_intrinsics()             # -> K, or None where a rig has no single pinhole matrix
handle.release()                    # gives THIS rig back; every other one keeps streaming
```

`RigHandle` answers exactly `grab()`, `get_intrinsics()`, `get_distortion()`, `release()` and
`is_open`, and names its owner as `camera`. That is the surface the real-camera perception adapter
and hand-eye calibration duck-type against. So nothing under
[`src/robot/perception/`](../robot/perception/README.md) has to learn about camera orchestration,
and a console teardown closes exactly the device it opened.

Constructing a streamer touches no device, so a provider is told about every configured rig and
opens only what it is asked to open: knowing a rig costs nothing, holding one is a deliberate act.

**Rig selection**: `camera.cameras.primary_rig_id` names the rig a CELL opens, and `select_rig`
refuses a primary that is not configured, is disabled or is not an RGB-D rig. The stereo
calibration pipeline is a tool over the whole catalogue rather than a cell: an explicit `rig_id`
argument wins there, and otherwise it works on every enabled rig that answers a probe.

## The two RGB-D backends

Selected per rig by `RGBDDeviceRigConfig.rgbd_backend`; `create_streamer` in
`orchestration/camera.py` builds the matching streamer.

| `rgbd_backend` | Streamer | Depth source |
| --- | --- | --- |
| `opencv` (default) | `OpenCvRGBDStreamer` | A generic `cv2.VideoCapture` with OpenNI `CAP_OPENNI_*` channels, and no vendor SDK. Returns empty depth explicitly on a device that exposes colour only. |
| `realsense` | `RealSenseRGBDStreamer` | Intel RealSense through `pyrealsense2`: hardware-aligned depth, device depth scale to uint16 millimetres, emitter, laser power and visual preset, a post-processing filter chain, and a colour-intrinsics accessor. |

`pyrealsense2` is pinned in `requirements.txt` and its import is deferred, so the library runs on a
machine without it. **The RealSense driver has never been exercised against a physical device
here**: its logic is covered with an injected fake SDK, and the real round trip remains the same
gap [`src/robot/perception/`](../robot/perception/README.md) names as the largest one in the stack.
The device identity rules above have not met hardware either: how a D435 enumerates beside an
OpenCV video device has not been observed.

## Traps

**`FrameProvider` only routes frames.** It does not crop, split, resize or configure quality; the
streamers under [`setup/`](setup/README.md) do. An unknown rig id raises `UnknownCameraRigError`;
grabbing a rig that is not open raises `FrameProviderStateError`, and reading a `Camera` that is not
open raises `CameraNotOpen`; `grab_rectified` on an RGB-D rig raises `ValueError`, and on a stereo
rig with no `StereoCam3D` raises `RuntimeError`; `get_stereo_rig_index` on an RGB-D rig raises
`KeyError`.

**A stereo rig has no single camera matrix**, so `get_intrinsics` answers `None` for one: its
geometry lives in `StereoCam3D`. Both accessors require the rig to be open.

**RGB-D rigs never reach stereo calibration.** Stereo rig config order is preserved and is what
`get_stereo_rig_index` returns. Each rig keeps isolated capture and calibration artefacts under its
own `calibration_paths.base_dir`.

## See also

- [`setup/`](setup/README.md), the streamers underneath the owners and the frame dataclasses
- [`src/calibration/`](../calibration/README.md), for stereo calibration, reconstruction, hand-eye
  and extrinsics
- [`src/geometry/`](../geometry/README.md), where robotics coordinates live
- [`src/config/`](../config/README.md), which validates the camera, rig, calibration and matcher
  config
- [`config/camera/cam.yaml`](../../config/camera/cam.yaml), the rigs themselves
