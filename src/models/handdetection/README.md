# Where a hand is, and thumbs up or down (`src/models/handdetection`)

MediaPipe hand landmarks behind typed wrappers: where the palm is, in pixels and then in millimetres in
the robot's base frame over a stereo or RGB-D rig, and whether the hand shows thumbs up or thumbs down.
It is optional and standalone: nothing on the grasp path imports it, and no cell builds it for you.

```python
from willy import load_tree
from src.models.handdetection import build_gesture_recognizer

models = load_tree().app_config.models             # the cell WILLY_PROFILE names
with build_gesture_recognizer(models.gesturedetect) as recognizer:
    for hand in recognizer.observe(frame_bgr):     # one OpenCV BGR image
        print(hand.palm.palm_center_xy, hand.gesture.gesture, hand.gesture.confidence)
```

```bash
python scripts/model_weights/fetch.py --mediapipe                    # the two .task bundles
python -m src.models.handdetection --check                           # 0 only when both are present
python -m src.models.handdetection --frame hand.jpg --gestures       # one image: no camera, no robot
python -m src.models.handdetection --rig overhead --transform cam_to_base.npy --intrinsics intrinsics.json --gestures
```

Exit codes: `0` success, `1` nothing detected, `2` a setup problem, so a missing hand and a missing model
file never look alike to a script. `--json` prints the same answer as data. `--frame` isolates the
detector: with a camera, a calibration and a robot in the loop, "it does not work" cannot say which is at
fault. Without `--transform`, `--rig` reports the hand in the camera frame only, and says so.

## Setup

`mediapipe==1.0.1` is pinned in `requirements.txt` and `requirements-cpu.txt`. A `0.10.x` release cannot
be installed beside the numpy and opencv pins the rest of the stack uses. The `.task` bundles are
binaries the operator downloads into `assets/models/hf/mediapipe/`, which git ignores. Their paths are
`models.handdetect.model_path` and `models.gesturedetect.model_path` in
[`config/models/hand.yaml`](../../../config/models/hand.yaml), so bundles kept elsewhere are a config
change.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `ThumbGestureRecognizer` | `build_gesture_recognizer(models.gesturedetect)` | `observe(frame_bgr)` | a `HandObservation` per hand: palm and gesture |
| `PalmDetector` | `build_palm_detector(models.handdetect)` | `observe(frame_bgr)` | a `HandObservation` per hand, landmarks only |
| `HandFinder` | `build_hand_finder(models.handdetect, provider=..., transforms=...)` | `find_hand()` | a `LocatedHand` in the base frame and an annotated image |
| the same, over one open camera | `build_hand_finder_on_camera(models.handdetect, camera)` | `find_hand()` | the same, reading that camera's own calibration and lens |

`build_hand_finder` takes the open frame provider, a 4x4 CAMERA to BASE matrix per rig, a 3x3 camera
matrix per RGB-D rig, a `StereoCam3D` for stereo rigs, and optionally the recognizer as `observer=` so
the located hand carries its gesture. `HandPosition3D.position_base` and `position_cam` are millimetres.
The transforms are not read from config on purpose: CAMERA to BASE is a calibration artefact with its
own loader, and a second spelling of it would be a second source of truth for the most safety-relevant
number here.

`build_hand_finder_on_camera` is the same search over one `Camera` a program already holds open, and
it is what a pick loop uses: `FrameProvider.open()` claims every configured streamer, so asking where
a hand is through a second catalogue takes the cell's other cameras away from it, and one device
opened twice is what the camera owner exists to prevent. It composes no transform either. A fixed
rig's CAMERA to BASE is `RigCalibration.camera_to_base()`, one method with one answer; a WRIST rig is
refused by name, because turning its CAMERA to TOOL into a base position needs the pose the arm stood
at when the shutter opened, and that composition belongs to `Locator`.
[`examples/real_robot/13`](../../../examples/real_robot/13_speak_pick_and_hand_handover.py) brings a
picked part to the hand it finds.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ImportError` when a detector is built | `mediapipe` is not installed (importing the package still works) | install from `requirements.txt` |
| `FileNotFoundError` when a detector is built | a `.task` bundle is missing; it names the key, the absolute path and the download | `python scripts/model_weights/fetch.py --mediapipe` |
| `ValueError` from `find_hand` | an RGB-D rig with no camera matrix, or a stereo rig with no `StereoCam3D` | supply the rig's intrinsics; nothing is invented |
| a skipped rig | a rig with no transform, or more than one hand in view | calibrate the rig; one hand at a time |
| `RigCalibrationError` from `build_hand_finder_on_camera` | the camera is a wrist rig, so it has no fixed CAMERA to BASE | search from a fixed camera, or compose the transform yourself and use `build_hand_finder` |
| `ValueError` from `build_hand_finder_on_camera` | an RGB-D camera whose device reports no intrinsics | give the rig a camera matrix; nothing is invented |
| exit `2` from `--rig` | an id not in `camera.cameras.rigs`, or a disabled rig; it lists the rigs there are | name a configured, enabled rig |

Two hands in a workspace is a reason to stop, not to choose one, because the position decides where a
robot may move. A rig missing a transform never borrows another rig's.

## Decisions worth knowing before you change something

**The palm centre averages six landmarks:** the wrist, the four finger MCPs and `THUMB_CMC`. The common
published definition leaves out the thumb base, which pulls the centroid slightly toward the thumb.
Changing the set moves every 3-D position this package reports, so change it only with a measurement on
a real hand.

**VIDEO mode, not IMAGE mode.** IMAGE mode treats every frame alone and would make `tracking_threshold`
inert. VIDEO mode needs strictly increasing millisecond timestamps, so they come from a monotonic clock
and are pushed forward on a collision.

**The gesture model returns landmarks too**, so `ThumbGestureRecognizer` reports the palm centre from the
same pass. A `PalmDetector` beside it would load a second model and pay for the same inference twice.

**`OTHER` is not `NONE`.** Two of the classifier's shapes are mapped; the rest become `OTHER` with the raw
label kept. "It saw your hand and it was not a thumbs up" and "it saw nothing" call for different actions.
The mapping runs on a case-folded label with separators stripped, so a revision that respells
`Thumb_Up` still maps.

## Status

| Capability | Evidence |
| --- | --- |
| Detection on a cell's own cameras, in the base frame | never touched hardware |

The palm centre, the gesture mapping and the 3-D projection are unit-tested without a model file. On five
of MediaPipe's own test photographs the `float16/latest` bundles found thumbs up (0.73) and
thumbs down (0.77), reported victory and pointing up as `OTHER` with their raw labels, and found both
hands in a two-hand frame; the palm centre landed inside the palm in every one.
`tests/test_hand_detection_bundle.py` reads the label vocabulary out of the bundle, and skips when the
bundles are absent.

## Files

| File | Holds |
| --- | --- |
| [`landmarks.py`](landmarks.py) | the 21-point layout, `PALM_LANDMARKS` and the palm centre; pure geometry |
| [`types.py`](types.py) | `PalmDetection`, `GestureReading`, `HandObservation`, `HandPosition3D`, `LocatedHand`; frozen |
| [`model_files.py`](model_files.py) | the `mediapipe` guard and the fail-closed `.task` resolver |
| [`palm_detector.py`](palm_detector.py), [`gestures.py`](gestures.py) | the landmark detector, and the gesture classifier with its palm centre |
| [`hand_finder.py`](hand_finder.py) | pixels plus depth plus calibration to millimetres in the base frame; `RigFrames` and `OneCamera`, where the frames come from |
| [`factory.py`](factory.py), [`constants.py`](constants.py) | the config readers, and the download URLs quoted in errors |
| [`__main__.py`](__main__.py) | the `--check`, `--frame` and `--rig` command |

## Details

- [`../README.md`](../README.md), the model layer; [`../../camera/`](../../camera/README.md), the frame provider
- Tests: `tests/test_hand_detection.py`, `tests/test_hand_detection_wiring.py`, `tests/test_hand_detection_cli.py`
