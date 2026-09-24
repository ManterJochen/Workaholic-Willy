# Cameras (`src/camera`)

Opens the cameras of a cell and hands out their frames. `Camera` owns one rig's device: it is the only
code that opens it, it runs every grab under the rig's lock, and it stamps each frame with the time it
was taken. Where a camera sits relative to the robot is decided in
[`src/calibration/`](../calibration/README.md), not here.

```python
from willy import Camera, RGBDFrame, load_tree

with Camera.from_tree(load_tree()) as camera:   # the rig camera.cameras.primary_rig_id names
    print(camera)                               # its rig id, its device, and whether it is open
    frame = camera.grab()                       # frame.captured_at_s is the host time of the grab
    lens = camera.get_intrinsics()              # the device's 3x3 matrix, or None where it reports none
    print(camera.calibration() if camera.calibrated else f"{camera.rig_id}: not calibrated")

assert isinstance(frame, RGBDFrame)             # from_tree refuses a rig with no depth
print(frame.color.shape, frame.depth.shape)     # BGR colour; depth in millimetres, 0 where nothing was measured
```

The `with` block opens the device and gives it back however the block ends. The same program at a cell
is [`examples/real_robot/06_open_a_camera.py`](../../examples/real_robot/06_open_a_camera.py). From a
shell, see which rig a cell opens, then prove that rig on its own:

```bash
python -m src.config explain camera.cameras.primary_rig_id
python -m src.robot.perception --rig <rig id>
```

The first exits 0 once it has answered and 1 when the tree does not load. The second needs the camera
and the model weights: it grabs one frame, runs the detector and prints the depth holes inside every
mask ([`src/robot/perception/`](../robot/perception/README.md)).

## The nouns

| Noun | Built by | Verb | Returns |
|---|---|---|---|
| `Camera` | `from_tree(tree)`, `from_config(camera_section)`, `from_rig(rig)`; `rig_id=` names another rig | `grab()`, in a `with` block or between `open()` and `release()`; `camera_moved()` after the camera moved | an `RGBDFrame` (`color`, `depth`) or a `StereoFrame` (`left`, `right`) |
| `RigHandle` | `camera.handle()`, `provider.rig(rig_id)` | `grab()`, `get_intrinsics()`, `camera_moved()`, `release()` | the owner's frames; `release()` gives back that one rig |
| `FrameProvider` | `FrameProvider(rigs)` | `grab(rig_id)`, `grab_rectified(rig_id)` | frames keyed by rig id, over stereo and RGB-D rigs |
| `StereoCapturePipeline` | `StereoCapturePipeline(cameras, stereo_calibration, stereomatcher)` | `run()` | `(FrameProvider, StereoCam3D)`; the second is `None` when every rig is RGB-D |

`Camera` also answers `get_distortion()`, and `calibration()` loads the rig's declared calibration,
`camera.cameras.rigs[<id>].extrinsics`, through the one loader in
[`rig_calibration.py`](../calibration/rig_calibration.py). `Camera`, `CameraRefused`, `RGBDFrame` and
`RigNotCalibrated` come from `willy`; the other names on this page import from `src.camera`.

`camera_moved()` tells the owner its camera moved since the last grab. A RealSense's temporal filter
averages each depth pixel over the frames it was handed and fills a hole with a depth it saw in them, so
on a camera the arm carries, the first frame after a move would otherwise place the old pose's depth
along the new rays. The notice drops that history, under the rig's lock, and is a no-op on a device that
keeps none. The planning world calls it after a move; a camera the arm carries also drops the history by
itself after a pause between two grabs longer than a burst of back-to-back grabs takes.

### Which rigs feed the planner world

A cell with several cameras does not plan against all of them. A rig feeds the live planner world
only if it is enabled, is RGB-D and declares its calibration, and `CameraWorldPlan` says which rigs
do and, one line each, why the others do not. It reads config and opens nothing, which makes it the
first call when a cell refuses to build:

```python
from willy import CameraWorldPlan, load_tree

cameras = load_tree().app_config.camera.cameras
plan = CameraWorldPlan.from_config(load_tree().robot, list(cameras.rigs),
                                   primary_rig_id=cameras.primary_rig_id)
print(plan)                 # the rigs the world takes, and why each other one is left out
print(plan.refusal())       # why a cuRobo cell may not build on it, or None
```

Once a rig declares its calibration on a cuRobo cell, its world is mandatory: a build that produced
none is refused (`CameraWorldRequired`) rather than planning blind.
[`examples/real_robot/14`](../../examples/real_robot/14_a_cell_with_several_cameras.py) reads the
plan and then opens one `Camera` per world rig.

### Many rigs, one catalogue

`FrameProvider` knows every rig it is given and opens only what it is asked to: building one touches no
device. Each rig it opens is held by a `Camera` owner, so a catalogue and an owner elsewhere in the
process never both hold one device.

```python
from willy import load_tree
from src.camera import FrameProvider

provider = FrameProvider(list(load_tree().app_config.camera.cameras.rigs))
provider.open_rig("overhead")        # one rig; open() or a with block opens every rig
handle = provider.rig("overhead")    # a RigHandle, shaped like a device streamer
frame = handle.grab()
handle.release()                     # gives back this rig; every other rig keeps streaming
```

`StereoCapturePipeline(...).run()` builds the same catalogue from the camera section, together with
the `StereoCam3D` that rectifies its stereo rigs ([the stereo runtime](../calibration/README.md#the-stereo-runtime)).
When a stereo rig lacks calibration images it records them first, interactively, and blocks until the
operator ends that session. RGB-D rigs pass through to the catalogue and are never stereo calibrated.

## Choosing the RGB-D driver

Each RGB-D rig names its driver in `rgbd_backend`:

| `rgbd_backend` | Driver | Depth |
|---|---|---|
| `opencv` (the default) | `OpenCvRGBDStreamer`: `cv2.VideoCapture` over OpenNI, no vendor SDK | empty on a device that exposes colour only |
| `realsense` | `RealSenseRGBDStreamer`, through `pyrealsense2` | in millimetres, through the SDK's filter chain, then aligned to colour by default |

A RealSense on the `opencv` driver gives colour and an empty depth channel, so set
`rgbd_backend: realsense` for one. `pyrealsense2` is in `requirements.txt` and is imported only when a
RealSense rig opens, so the library runs on a machine without it.

The depth mode sets how near a RealSense measures. Intel gives a D415 a minimum depth (Min-Z) of about
450 mm at 1280 x 720 and about 310 mm at 848 x 480, and a D435 about 280 mm at 1280 x 720 (datasheet
figures, not measured here); anything nearer reads as a hole. A D415 on the wrist sets
`depth_resolution: [848, 480]` and keeps `color_resolution: [1280, 720]`: depth aligned to colour comes
back on the 1280 x 720 grid. When it opens, the driver logs the camera's name, serial, firmware and USB
link, the depth mode and its Min-Z, and warns for a D415 at 1280 x 720 and for a camera on a USB 2 link.

## What it refuses

| Refusal | When | What to do |
|---|---|---|
| `CameraRefused`, `reason="unknown"` | the section has no rig with that id | name a rig from `camera.cameras.rigs` |
| `CameraRefused`, `reason="disabled"` | the rig says `enabled: false` | switch it on, or name a rig that is on |
| `CameraRefused`, `reason="not_rgbd"` | the rig has no depth channel of its own, a stereo pair | name an RGB-D rig; the message lists them |
| `CameraBusy` | another owner in this process holds the same device; the message names it | release that owner, or give the rigs distinct device ids |
| `CameraNotOpen` | a grab or a lens read before `open()` | use a `with` block, or call `open()` first |
| `RigNotCalibrated` | `calibration()` on a rig with no `extrinsics` block | calibrate the camera and paste the block the sweep prints |
| `RigCalibrationError` | the declared artifact does not load | the message names the key and the path |
| `ConfigError` when the tree loads | two enabled RGB-D rigs where one has no `serial_number`, or two share one | set one serial per rig; `rs-enumerate-devices` prints them |
| `RuntimeError` from `open()`, naming the cameras the SDK sees | a RealSense request that does not start: no camera, another serial, or a mode the camera does not offer on its USB link | the message names each camera's USB link; connect it over USB 3, or ask for a mode it lists |
| `RuntimeError` from `open()`, naming `depth_units_m` | the device reads back other depth units than the rig configures | remove `depth_units_m`, or write a value the device takes |

A device is known by what identifies it, never by the rig name: a RealSense by its serial, an OpenCV
RGB-D rig or a single-device stereo rig by its `device_index`, a webcam pair by both ids. A RealSense
with no serial collides with every open RealSense, because the SDK then binds whichever camera it
offers first, and a webcam pair with an unset id collides with every open video device.

The catalogue raises `UnknownCameraRigError` for a rig id it does not hold, `FrameProviderStateError`
for a grab on a rig that is not open, `ValueError` for `grab_rectified` on an RGB-D rig, and `KeyError`
for `get_stereo_rig_index` on one. A stereo rig has no single camera matrix, so its `get_intrinsics()`
is `None`; its geometry lives in `StereoCam3D`.

## Status

The legend is the root README's [Status and honest scope](../../README.md#status-and-honest-scope).

| Capability | Evidence |
|---|---|
| One owner per device, serialised grabs, stamped frames | never touched hardware; pinned against device doubles in [`test_camera_noun.py`](../../tests/test_camera_noun.py) |
| The RealSense driver | never touched hardware; the real librealsense processes its frames, no camera attached ([test](../../tests/test_realsense_sdk_contract.py)), including its filter order and the dropped temporal history after a move ([test](../../tests/test_a_moved_wrist_camera_forgets_the_last_pose.py)) |
| Min-Z, USB link and depth units at open | never touched hardware; the Min-Z figures are Intel's, and the USB and depth-unit messages are pinned against an SDK double ([test](../../tests/test_a_realsense_says_what_it_opened.py)) |
| Device identity by serial and by index | never touched hardware; how a D435 enumerates beside an OpenCV video device is not observed |

## Files

| File | Holds |
|---|---|
| [`orchestration/camera.py`](orchestration/camera.py) | `Camera`, its refusals, the process registry, `select_rig`, and `create_streamer`, the one place a device streamer is built |
| [`orchestration/frame_provider.py`](orchestration/frame_provider.py) | `FrameProvider` and `RigHandle` |
| [`pipeline/stereo_capture.py`](pipeline/stereo_capture.py) | `StereoCapturePipeline` |
| [`setup/`](setup/README.md) | the device drivers, the two frame types and the capture quality settings |

## Details

- The rigs of the shipped tree: [`config/camera/cam.yaml`](../../config/camera/cam.yaml); every key of a
  rig as it validated: `python -m src.config explain camera.cameras.rigs`.
- Calibrating a camera: [`docs/calibration-setup.md`](../../docs/calibration-setup.md), and the examples
  [`07_calibrate_a_fixed_camera.py`](../../examples/real_robot/07_calibrate_a_fixed_camera.py) and
  [`09_calibrate_a_wrist_camera.py`](../../examples/real_robot/09_calibrate_a_wrist_camera.py).
- The adapter that turns a frame into what a pick perceives: [`src/robot/perception/`](../robot/perception/README.md).
- Every command: [`docs/cli.md`](../../docs/cli.md). The first cell:
  [`docs/runbooks/real_cell_first_pick.md`](../../docs/runbooks/real_cell_first_pick.md).
- Tests: [`test_camera_noun.py`](../../tests/test_camera_noun.py),
  [`test_frame_provider_seam.py`](../../tests/test_frame_provider_seam.py),
  [`test_realsense_streamer.py`](../../tests/test_realsense_streamer.py).
