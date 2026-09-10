# Hand detection: palm centre and thumbs up or down

MediaPipe hand landmarks behind typed wrappers. Two things it does:

1. **Where the palm is.** The mean of six MediaPipe landmarks, in pixels, and from there in
   millimetres in the robot base frame, over either a stereo rig or an RGB-D rig.
2. **What the hand is doing.** Thumbs up and thumbs down, from MediaPipe's canned classifier.

**Optional and standalone.** Nothing in the grasp pipeline imports this package. Importing any
module here works without `mediapipe` installed; constructing a detector is what refuses, with a
message naming the config key, the resolved absolute path and the download URL.

## What this package guarantees

- **Fail closed on setup.** A missing extra and a missing `.task` bundle are both checked before a
  detector starts, because a detector that silently does not run reads exactly like an empty scene.
- **Per-rig facts stay per rig.** `transforms` and `camera_matrices` are keyed by rig id. A rig
  missing a transform is skipped with a warning rather than borrowing another rig's, which would
  produce confident and wrong base coordinates.
- **No invented intrinsics.** An RGB-D rig with no camera matrix refuses. A fallback such as
  `fx = fy = width / 2` is not a camera; it is a number that completes the arithmetic and turns a
  hand at an unknown distance into a plausible-looking position no measurement supports.
- **Exactly one hand for 3-D.** `HandFinder` skips a rig showing more than one. This decides where
  a robot may move, and two hands in a workspace is a reason to stop, not to choose one.

## Setup

`mediapipe` is pinned in `requirements.txt` (and in `requirements-cpu.txt`). The `.task` bundles
are not in this repository: they are binaries and the operator picks the revision.

```bash
# The script reads the two URLs from constants.py, which is where the code reads them too.
python scripts/model_weights/fetch.py --mediapipe

python -m src.models.handdetection --check      # exit 0 only when both are actually there
python -m src.models.handdetection --frame hand.jpg --gestures   # prove it on a picture
```

`assets/models/hf/mediapipe/` is gitignored. The paths come from `models.handdetect.model_path` and
`models.gesturedetect.model_path` in [`config/models/hand.yaml`](../../../config/models/hand.yaml),
so an operator who keeps the bundles elsewhere changes the config, not the code.

The pin is `mediapipe==1.0.1` and not a `0.10.x` release: `0.10.21` cannot be installed alongside
the numpy and opencv pins the rest of the stack uses, because pip resolves it by downgrading both,
and torch, scipy and Isaac work against the newer ones.

## CLI

```bash
python -m src.models.handdetection --check                 # install state
python -m src.models.handdetection --frame hand.jpg        # one image, no camera
python -m src.models.handdetection --frame hand.jpg --gestures
python -m src.models.handdetection --rig overhead \
    --transform cam_to_base.npy --intrinsics intrinsics.json --gestures
```

Exit codes: `0` found, `1` nothing detected, `2` a setup problem. A missing hand and a missing model
file must not look alike to a script. `--json` prints the same answer machine-readably.

`--rig` names one rig and opens one camera. An id that is not in `camera.cameras.rigs`, or one whose
rig is `enabled: false`, is exit `2` and lists the rigs there are. Measured 2026-09-10, before that
check existed: a mistyped rig id opened every configured camera, matched none of them, and exited
`1`, the code that tells a caller the workspace is clear.

The `--frame` mode exists because "it does not work" with a camera, a calibration and a robot in
the loop gives an operator no way to tell which of the four is wrong. Without `--transform`, `--rig`
reports the hand in the camera frame only and says so.

## Using it from code

```python
from src.config import load_config
from src.models.handdetection import build_gesture_recognizer, build_hand_finder

config = load_config()

# 2-D plus gesture, one model, one pass
with build_gesture_recognizer(config.models.gesturedetect) as recognizer:
    for hand in recognizer.observe(frame_bgr):
        print(hand.palm.palm_center_xy, hand.gesture.gesture, hand.gesture.confidence)

# 3-D in the base frame
finder = build_hand_finder(
    config.models.handdetect,
    provider=frame_provider,                       # already open
    transforms={"overhead": T_cam_to_base},        # 4x4, per rig
    camera_matrices={"overhead": K},               # 3x3, required for RGB-D rigs
    stereo=stereo_cam3d,                           # required for stereo rigs
    observer=recognizer,                           # optional: adds the gesture to the result
)
located, annotated = finder.find_hand()
```

`HandPosition3D.position_base` and `position_cam` are in millimetres, and the base-frame value needs
the same CAMERA to BASE calibration the grasp path uses.

**Transforms are not read from config, deliberately.** A CAMERA to BASE transform is a calibration
artefact with an existing loader, so a second way to spell it under `models.handdetect` would be a
second source of truth for the most safety-relevant number here.

## Layout

| File | Role |
| --- | --- |
| [`landmarks.py`](landmarks.py) | The 21-point layout as an `IntEnum`, `PALM_LANDMARKS`, and the palm centre. Pure geometry: no MediaPipe, no camera, no config. |
| [`types.py`](types.py) | `PalmDetection`, `GestureReading`, `HandObservation`, `HandPosition3D`, `LocatedHand`, `HandGesture`, `Handedness`. All frozen and validating. |
| [`model_files.py`](model_files.py) | The optional-extra guard and the fail-closed `.task` resolver. |
| [`palm_detector.py`](palm_detector.py) | MediaPipe hand landmarker to palm centres in pixels. |
| [`gestures.py`](gestures.py) | Canned classifier to thumbs up or down, with the palm centre from the same pass. |
| [`hand_finder.py`](hand_finder.py) | Pixels plus depth plus calibration to millimetres in the base frame. |
| [`factory.py`](factory.py) | The config readers for `models.handdetect` and `models.gesturedetect`. |
| [`constants.py`](constants.py) | Log files, and the download URLs quoted in errors. |

## Decisions worth knowing before you change something

**The palm centre averages six landmarks:** the wrist, the four finger MCPs, and `THUMB_CMC`. The
commoner published definition omits the thumb base, and including it pulls the centroid slightly
toward the thumb. Changing the set moves every 3-D hand position this package reports, so do not
change it without a measurement on a real hand, and say which one.

**VIDEO mode, not IMAGE mode.** IMAGE mode treats every frame independently, which would make the
`tracking_threshold` config key inert, and a config key that cannot do anything is worse than no
key. The cost is that MediaPipe demands strictly increasing millisecond timestamps, so they are
read from a monotonic clock and forced forward on a collision.

**The gesture model returns landmarks too**, so `ThumbGestureRecognizer` reports the palm centre
from the same pass. Running a separate `PalmDetector` alongside it loads a second model and pays for
the same inference twice.

**`OTHER` is not `NONE`.** Of the seven shapes the canned classifier knows, two are mapped and the
rest become `OTHER`, with the raw label kept on the reading. "The cell saw your hand and it was not
a thumbs-up" and "the cell saw nothing" call for different operator actions.

**Label spelling is normalised, not assumed.** The mapping runs on a case-folded form with
separators stripped, so `Thumb_Up`, `thumb up` and `ThumbUp` all land on the same value. The
bundles are an operator download, so the spelling any given revision uses is unverified here, and
the normalisation is what makes the mapping hold.

**MediaPipe runs on the CPU.** No GPU delegate is configured.

## See also

- [`src/models/`](../README.md), the model layer this package sits in
- [`src/camera/`](../../camera/README.md), the `FrameProvider` the 3-D path streams from
- [`config/models/hand.yaml`](../../../config/models/hand.yaml), every key this package reads
