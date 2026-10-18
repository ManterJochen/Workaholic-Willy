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

assert isinstance(frame, RGBDFrame)             # every camera gives depth; a stereo pair once it is calibrated
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
| `Camera` | `from_tree(tree)`, `from_config(camera_section, workspace=)`, `from_rig(rig, workspace=)`; `rig_id=` names another rig | `grab()`, in a `with` block or between `open()` and `release()`; `camera_moved()` after the camera moved | an `RGBDFrame` (`color`, `depth`), a stereo pair's too; a `StereoFrame` (`left`, `right`) only from the catalogue's raw pair |
| `RigHandle` | `camera.handle()`, `provider.rig(rig_id)` | `grab()`, `get_intrinsics()`, `camera_moved()`, `release()` | the owner's frames; `release()` gives back that one rig |
| `FrameProvider` | `FrameProvider(rigs)` | `grab(rig_id)`, `grab_rectified(rig_id)` | frames keyed by rig id, over stereo and RGB-D rigs |
| `StereoCapturePipeline` | `StereoCapturePipeline(cameras, stereo_calibration)` | `run()` | `(FrameProvider, StereoCam3D)`; the second is `None` when every rig is RGB-D |
| `StereoCalibration` | `from_tree(tree, rig_id=)` | `record()`, then `calibrate()` | a `StereoCalibrationReport`: the pair solved against its ChArUco board and its `stereo_map_file` written, or why not |

`Camera` also answers `get_distortion()`, and `calibration()` loads the rig's declared calibration,
`camera.cameras.rigs[<id>].extrinsics`, through the one loader in
[`rig_calibration.py`](../calibration/rig_calibration.py). `Camera`, `CameraRefused`, `RGBDFrame` and
`RigNotCalibrated` come from `willy`; the other names on this page import from `src.camera`.

`Camera.gives_depth` says whether a grab is an `RGBDFrame`: it is for every owner the three doors build,
and not for one a stereo catalogue built over a raw pair. `camera_moved()` tells the owner its camera moved
since the last grab. A RealSense's temporal filter
averages each depth pixel over the frames it was handed and fills a hole with a depth it saw in them, so
on a camera the arm carries, the first frame after a move would otherwise place the old pose's depth
along the new rays. The notice drops that history, under the rig's lock, and is a no-op on a device that
keeps none. The planning world calls it after a move; a camera the arm carries also drops the history by
itself after a pause between two grabs longer than a burst of back-to-back grabs takes.

### Which rigs feed the planner world

A cell with several cameras does not plan against all of them. A rig feeds the live planner world
only if it is enabled and declares its calibration, an RGB-D rig or a stereo pair alike (a pair's owner
refuses to open before its stereo calibration is written), and `CameraWorldPlan` says which rigs do
and, one line each, why the others do not. It reads config and opens nothing, which makes it the
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
[`examples/real_robot/16`](../../examples/real_robot/16_a_cell_with_several_cameras.py) reads the
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

## A stereo pair as the cell's camera

A webcam pair (`source: webcam_pair`) or one side by side stereo device (`source: single_device`) is a
camera like a RealSense once its stereo calibration is written (the owner, 2026-10-17). Its owner opens
it through `StereoDepthStreamer`: every grab rectifies the pair, matches it with OpenCV's semi-global
matcher and hands out an `RGBDFrame` whose colour is the left image rectified (BGR `uint8`) and whose
depth lies on its pixels in `uint16` millimetres, 0 where the pair could not be matched. Its lens is the
rectified left camera's, with no distortion (`get_intrinsics()`, `get_distortion()`). The pick
perception, the hand-eye calibration, the guard's perceived world and the planner read it as they read
a RealSense, fixed in the cell or on the wrist.

It takes three steps, each the one an RGB-D camera takes after its first:

1. **Calibrate the pair.** Print the ChArUco board `camera.cameras.stereo_calibration` describes and
   measure its squares: they set the scale of the baseline, and so of every depth. `StereoCalibration`
   records pairs of it in a window per eye (`s` keeps a pair, ESC ends) and solves them, writing the
   rig's `calibration_paths.stereo_map_file`; a residual above 2 px is refused and nothing is written.
   [`examples/real_robot/19_calibrate_a_stereo_pair.py`](../../examples/real_robot/19_calibrate_a_stereo_pair.py)
   is that program.
2. **Calibrate it against the robot**, as any camera:
   `python -m src.robot.execution.real_cell.calibrate --rig <rig id> --mode eye_to_hand --freedrive`
   (or `eye_in_hand` for a pair on the wrist), examples 07 to 10. The board is posed on the left image
   rectified with the rectified lens, the frame the depth lies in, so the solve places that depth. Paste
   the `extrinsics` block the sweep prints into the rig; a pair on the wrist declares its `body` too.
3. **Name it `camera.cameras.primary_rig_id`.** The cell perceives with it, plans in it and judges every
   motion against its depth;
   [`examples/real_robot/20_pick_with_a_stereo_camera.py`](../../examples/real_robot/20_pick_with_a_stereo_camera.py)
   opens it once and runs a campaign of picks with it.

Until step 1 the rig gives no depth, and every door that chooses or opens it says so in one sentence
that names the file and how to write it (`CameraRefused`, `reason="no_stereo_calibration"`). The stereo
calibration tools, `StereoCapturePipeline` and the catalogue, keep the raw pair.

**What the matcher searches.** Nothing of it is a config key: the stereo matcher block,
`camera/stereomatcher.yaml`, left the tree on 2026-10-17, and a tree that still carries it is refused
at load. The disparities searched are derived from the pair's own calibration, focal length times
baseline over depth, for the depths `working_depth_range_mm` gives: from where the work comes nearest
the camera (a fixed pair's workspace box, `robot.motion.workspace`, as its calibration places it; on the
wrist, before the pair is placed, or in a tool without the robot's section, 150 mm, where the owner's
wrist camera stands over the bench at the grasp) to infinity. Every other setting is fixed at the
values measured on the D415's own infrared pair (`src/calibration/stereo/sub_modules/disparity.py`),
and the matcher never fills a hole, from its neighbours or from an earlier frame: the guard judges
every motion against this depth, and a filled hole can hide an obstacle.

**Measured** on the owner's Thursday views (2026-10-15): 306 looks of the wrist D415 that kept its
own infrared pair, matched as a webcam pair is and set against the depth the chip matched on the same
images (`python scripts/stereo/measure_on_the_d415_ir_pair.py --views <folder> --alternatives`). The
first row is the matcher's settings, each row under it those settings with one thing changed:

| Settings | Depth on | Looks under half | Too far | Too near | Parts, mm | Mat, mm | Edge at no surface | ms |
|---|---|---|---|---|---|---|---|---|
| **the matcher's** | 56 % | 31 | 0.36 % | 0.72 % | 1.8 / 5.5 | 1.9 / 5.8 | 10.9 % | 185 |
| the full matcher, not 3-way | 55 % | 57 | 0.34 % | 0.88 % | 1.8 / 5.4 | 2.2 / 6.9 | 10.9 % | 680 |
| P1 and P2 three times (the block before) | 59 % | 0 | 0.60 % | 3.02 % | 2.1 / 7.0 | 2.5 / 9.1 | 17.0 % | 184 |
| a block of 3 | 28 % | 303 | 0.40 % | 0.35 % | 1.8 / 5.4 | 1.6 / 5.1 | 7.4 % | 177 |
| a block of 7 | 64 % | 0 | 0.45 % | 1.06 % | 1.8 / 5.5 | 1.9 / 6.2 | 12.6 % | 191 |
| uniqueness 10 | 57 % | 16 | 0.43 % | 0.80 % | 1.8 / 5.5 | 1.9 / 5.9 | 11.3 % | 187 |
| uniqueness 20 | 55 % | 54 | 0.29 % | 0.62 % | 1.8 / 5.3 | 1.9 / 5.7 | 10.4 % | 183 |
| no left padding | 34 % | 306 | 0.27 % | 0.63 % | 1.7 / 5.1 | 1.9 / 5.5 | 10.4 % | 133 |
| no left-right check | 60 % | 0 | 0.57 % | 0.90 % | 1.8 / 5.5 | 1.9 / 5.9 | 11.7 % | 94 |
| flat blocks kept | 62 % | 0 | 0.42 % | 0.89 % | 1.8 / 5.5 | 1.9 / 6.1 | 11.7 % | 184 |
| no margin about holes and steps | 73 % | 0 | 1.46 % | 1.87 % | 1.9 / 6.1 | 2.0 / 6.7 | 15.3 % | 181 |
| a margin of 3 | 49 % | 234 | 0.19 % | 0.46 % | 1.8 / 5.3 | 1.8 / 5.5 | 9.4 % | 183 |
| the search ending at the workspace's diagonal | 61 % | 0 | 0.65 % | 1.60 % | 1.8 / 5.5 | 1.9 / 5.8 | 11.1 % | 172 |
| `stereomatcher.yaml`, without its WLS filter | 70 % | 0 | 2.79 % | 7.93 % | 2.2 / 8.2 | 2.9 / 15.6 | 22.5 % | 63 |

*Depth on* is the share of the frame that holds a depth (the chip's: 79 %), and *looks under half*
the looks with less, which the guard would hold a fixed camera blind at. *Too far* and *too near* are
the pixels both measured that lie more than max(10 mm, 3 %) beyond the chip's depth, hiding what
stands in front, or before it, a surface where none stands. *Parts* and *mat* are the median and the
90th percentile of the difference on the parts and on the mat about them. *Edge at no surface* is the
share of the depths within 4 px of a part's edge that lie more than max(5 mm, 1 %) from every depth
the chip measured within 2 px. *ms* is the median frame, rectified and matched, on a 32-thread Ryzen 9
9950X3D shared with other work while it ran: a guide to the cell PC, not its number. A surface nearer
than the search is no better: with a search of 64 disparities, which reads nothing nearer than 778 mm,
an eighth of the pixels the chip measured nearer held a depth, nearly every one of them wrong.

The matcher's settings read the fewest pixels wrong of the rows that keep nine looks in ten over the
half. The median look holds a depth on 54 % of its frame, and the 31 under it, nearly all a pick's
first look, between 46 % and 50 %. A fixed pair that sees views like these sits near the guard's half;
a block of 7 is the measured way over it (no look under, the median at 62 %), at half again as many
pixels read too near. A pair on the wrist asked about a goal is judged about the goal.

**What it does not do.** A surface with nothing on it to match, a white table under plain light, holds
no depth. Every look above was lit by the D415's projector (`emitter_enabled`), whose dots give every
surface something to match; a webcam pair has none, so it reads less of a plain surface than the
table says, and a fixed camera with less than half its frame measured is blind to the guard
(`live_world.min_depth_coverage`). A surface nearer than the search, the arm between a fixed pair and
its box or a part held before a pair on the wrist, is a hole or a depth behind where it stands. And
every grab is matched, the warm-ups too: a pick look grabs 12 frames (5 warm-ups and a burst of 7)
and a wrist camera's world before each planned motion 6, so at 1280 x 720 with the wrist's search of
336 disparities a look costs about 2 s of matching on the machine above and a planned motion about
1 s (a frame takes 180 ms from 8 threads up, 290 ms on 4). A smaller frame costs less, and so does a
fixed pair, whose search starts at its workspace box.

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
link, the depth mode and its Min-Z, and warns for a D415 at 1280 x 720 and for a camera on a USB 2 link. Its lines go
to `logs/camera/rgbd.log` and the console.

### The colour sensor, and recording for research

A RealSense rig's `realsense.color` block sets the colour sensor's exposure, gain and white balance (the owner,
2026-10-09). Every key null, as shipped, is full auto and writes nothing. On full auto the same part comes out 1 to
1.8 stops brighter or darker with what else is in view at each look (measured on the owner's views), so a cell fixes
the values auto settles on at look 0 with the work lamp on:

```yaml
realsense:
  color:
    auto_exposure: false      # written first: librealsense writes its default exposure as auto exposure goes off
    exposure: 156             # in 100 microseconds, so 15.6 ms; Windows holds the nearest power of two of a second
    gain: 64
    auto_white_balance: false
    white_balance: 4600       # Kelvin
```

The driver writes the keys in that order to the sensor that streams colour, never to the depth sensor, once every
value has been checked against the range the sensor offers, and logs what the sensor holds when the warm-up frames
have run, also with every key null. A value the sensor holds otherwise is warned about, with the value it holds. The
camera keeps what is written until it is unplugged or power-cycled: a block put back to null leaves those values in
place, and `auto_exposure: true` and `auto_white_balance: true` put the camera back on auto.

`realsense.record_for_research: true` records for research, one switch, off by default. The camera then streams both
infrared images as well (infrared 1 and 2, Y8, at the depth mode), and every frame carries them, the depth as the
sensor sent it, before the filters and the alignment, and the camera's facts (`RGBDFrame.research`, a
`ResearchCapture`). The views file of every pick a program keeps holds them per look, with their lenses and
extrinsics, every segmentation's mask, box, label, score, SAM2's predicted IoUs and which one was the target, and once
the camera's name, serial, firmware, depth units, filters, preset, colour settings, the colour and depth lenses and the
depth to colour extrinsics (format 2, [`record_views.py`](../robot/execution/record_views.py)). Both infrared streams
add 442 Mbit/s to the 885 Mbit/s colour and depth take at 1280 x 720 and 30 fps, which a USB 3 link carries, and a
views file grows from about 2.1 MB to about 4.4 MB per look. Off, the streams, the frames and the views files are what
they were.

## What it refuses

| Refusal | When | What to do |
|---|---|---|
| `CameraRefused`, `reason="unknown"` | the section has no rig with that id | name a rig from `camera.cameras.rigs` |
| `CameraRefused`, `reason="disabled"` | the rig says `enabled: false` | switch it on, or name a rig that is on |
| `CameraRefused`, `reason="no_stereo_calibration"` | a stereo pair whose stereo calibration is not on disk at its `calibration_paths.stereo_map_file` | calibrate the pair (`StereoCalibration`, example 19); the message lists the rigs that give depth now |
| `StereoNotCalibrated` from `open()` | the same, met at a stereo rig's owner built with `from_rig` | as above |
| `StereoCalibrationError` from `open()` | the stereo calibration was solved at another size than the rig's eyes, or rectifies its cameras one above the other or swapped | calibrate the pair again at its own size, its cameras side by side, left as left |
| `CameraRefused`, `reason="backend"` | on Linux, an OpenCV rig whose `backend` is 700 (DirectShow) or 1400 (Media Foundation), which only OpenCV's Windows builds have; the tree still loads | write 0 (CAP_ANY) or 200 (CAP_V4L2); a RealSense rig never reads the key |
| `CameraBusy` | another owner in this process holds the same device; the message names it | release that owner, or give the rigs distinct device ids |
| `CameraNotOpen` | a grab or a lens read before `open()` | use a `with` block, or call `open()` first |
| `RigNotCalibrated` | `calibration()` on a rig with no `extrinsics` block | calibrate the camera and paste the block the sweep prints |
| `RigCalibrationError` | the declared artifact does not load | the message names the key and the path |
| `ConfigError` when the tree loads | two enabled RGB-D rigs where one has no `serial_number`, or two share one | set one serial per rig; `rs-enumerate-devices` prints them |
| `RuntimeError` from `open()`, naming the cameras the SDK sees | a RealSense request that does not start: no camera, another serial, or a mode the camera does not offer on its USB link | the message names each camera's USB link; connect it over USB 3, or ask for a mode it lists |
| `RuntimeError` from `open()`, naming `depth_units_m` | the device reads back other depth units than the rig configures | remove `depth_units_m`, or write a value the device takes |
| `RuntimeError` from `open()`, naming `realsense.color.<key>` | the colour sensor does not offer the option, the value lies outside its range, or the camera's colour comes off its depth imagers; nothing was written to it | remove the key, or write a value in the range the message names |
| `RuntimeError` from `open()`, naming `record_for_research` | the camera cannot stream both infrared images at this mode, or its link cannot carry them | switch `record_for_research` off, or connect the camera over USB 3 |
| `ConfigError` when the tree loads, naming `realsense.color` | a fixed exposure, gain or white balance without its auto mode written `false` beside it | write `auto_exposure: false` (or `auto_white_balance: false`) beside the value |

A device is known by what identifies it, never by the rig name: a RealSense by its serial, an OpenCV
RGB-D rig or a single-device stereo rig by its `device_index`, a webcam pair by both ids. A RealSense
with no serial collides with every open RealSense, because the SDK then binds whichever camera it
offers first, and a webcam pair with an unset id collides with every open video device.

The catalogue raises `UnknownCameraRigError` for a rig id it does not hold, `FrameProviderStateError`
for a grab on a rig that is not open, `ValueError` for `grab_rectified` on an RGB-D rig, and `KeyError`
for `get_stereo_rig_index` on one. In the catalogue a stereo rig has no single camera matrix, so its
`get_intrinsics()` is `None` and its geometry lives in `StereoCam3D`; its own owner reports the
rectified left lens.

## Status

The legend is the guide's [evidence levels](../../docs/guide/README.md#what-verified-means-on-these-pages).

| Capability | Evidence |
|---|---|
| One owner per device, serialised grabs, stamped frames | run on a physical cell: a wrist D415 on a UR10 (CB3); pinned against device doubles in [`test_camera_noun.py`](../../tests/test_camera_noun.py) |
| The RealSense driver | run on a physical cell: the same D415; in the suite the real librealsense processes its frames, no camera attached ([test](../../tests/test_realsense_sdk_contract.py)), including its filter order and the dropped temporal history after a move ([test](../../tests/test_a_moved_wrist_camera_forgets_the_last_pose.py)) |
| Min-Z, USB link and depth units at open | never touched hardware; the Min-Z figures are Intel's, and the USB and depth-unit messages are pinned against an SDK double ([test](../../tests/test_a_realsense_says_what_it_opened.py)) |
| The colour block and the research recording | never touched hardware; the write order, the read-back, the refusals, the frames and the views file are pinned against an SDK double, the names and the extrinsics against the real librealsense ([colour](../../tests/test_the_colour_sensor_holds_what_its_rig_asks.py), [research](../../tests/test_a_rig_recording_for_research_keeps_its_infrared_images.py)) |
| Device identity by serial and by index | never touched hardware; how a D435 enumerates beside an OpenCV video device is not observed |
| A stereo pair as an RGB-D camera | measured offline on the D415's own infrared pair against its chip's depth ([script](../../scripts/stereo/measure_on_the_d415_ir_pair.py), [regression](../../tests/test_stereo_depth_on_the_d415_ir_pair.py)); swept against the robot on a rendered pair with a known lens and baseline ([test](../../tests/test_a_stereo_rig_is_calibrated_against_the_robot.py)); never run on a webcam pair or a stereo device at a cell |

## Files

| File | Holds |
|---|---|
| [`orchestration/camera.py`](orchestration/camera.py) | `Camera`, its refusals, the process registry, `select_rig`, and `create_streamer` and `create_device_streamer`, the one place a device streamer is built |
| [`stereo_calibration.py`](stereo_calibration.py) | `StereoCalibration` and its report: a pair recorded and solved against its ChArUco board |
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
