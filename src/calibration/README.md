# `src/calibration/`: typed extrinsics, the stereo runtime and persistence

The single owner of calibration domain logic: typed extrinsics, the stereo runtime, quality bands
and schema-versioned persistence. The two hand-eye workflows are one level down, in
[`eye_hand/`](eye_hand/README.md).

`NumPy and cv2 only, no torch` `every transform frame-tagged` `millimetres` `deterministic JSON`

Everything here answers one question: where is the camera relative to the robot. It does not own
camera device IO, robot vendor drivers, generic geometry primitives, or application pipelines.
Because it is torch-free, the whole package runs under the mock test suite on any machine.

```python
from src.calibration import (
    Extrinsics, EyeToHandCalibrator, EyeInHandCalibrator, EyeHandCalibrationSettings,
    HandEyeAXXB, QualityBandsMm, classify_rmse, load_extrinsics, save_extrinsics, unit_scaling,
)
from src.calibration.stereo import StereoCam3D, StereoRigConfig
```

## What is in it

| Path | Role |
|---|---|
| [`extrinsics.py`](extrinsics.py) | `Extrinsics`, a typed CAMERA to BASE extrinsic with error metrics, under schema `willy.calibration.extrinsics/1` |
| [`eye_hand/`](eye_hand/README.md) | The datasets, the acceptance settings, `MountingMode`, and the two workflows `EyeToHandCalibrator` and `EyeInHandCalibrator`. It has its own README |
| [`solver/`](solver/) | Pure-NumPy primitives: `HandEyeAXXB` for AX=XB, `UmeyamaRigid` for point-set registration |
| [`stereo/`](stereo/) | Stereo calibration, rectification, disparity, reconstruction and ArUco marker pose: `StereoCam3D`, `StereoRigConfig`, `CalibrationResult`. OpenCV use is isolated here |
| [`quality.py`](quality.py) | `classify_rmse` plus `QualityBandsMm` and `QualityBandsPx` |
| [`serialization.py`](serialization.py) | Deterministic JSON persistence for extrinsics and for camera-to-tool transforms |
| [`rgbd_marker_source.py`](rgbd_marker_source.py) | An RGB-D ArUco marker source for the hand-eye routine's injected `marker_source` seam |
| [`helpers.py`](helpers.py) | `unit_scaling`, `proj_to_K`, stereo image-pair shape validation |
| [`exceptions.py`](exceptions.py) | `CalibrationError` and its subclasses |
| [`constants.py`](constants.py) | `CALIBRATION_LOG_DIR` and the per-module log-file names |

## Frames, which are the whole point of this package

Every public transform carries explicit `Frame` values from [`src/geometry`](../geometry/README.md).
`Extrinsics` accepts only `Transform(from_frame=Frame.CAMERA, to_frame=Frame.BASE)`; the wrong
direction raises `ExtrinsicsError` rather than producing a plausible wrong answer.

| Name | Direction | Meaning |
|---|---|---|
| `T_cam_to_marker` | CAMERA to MARKER | Marker pose out of the rectified left image, as OpenCV reports it |
| `T_base_to_tool` | BASE to TOOL | One robot forward-kinematics sample |
| `T_cam_to_base` | CAMERA to BASE | The fixed-camera, eye-to-hand output |
| `T_cam_to_tool` | CAMERA to TOOL | The wrist-camera, eye-in-hand output |

The conventions behind those names: the camera frame is the rectified left optical centre with Z
forward, X right and Y down; the marker frame sits at the marker centre with its corners in the
`z=0` plane; the tool frame is whichever flange or TCP frame the application uses; the base frame is
the robot base.

## The stereo runtime

```
StereoCam3D
  StereoRigFactory -> StereoCalibration, StereoRigRepository
  StereoRigRunTime[] -> StereoRectifier, DisparityComputer, PointCloudReconstructor, ExtrinsicsTransformer
  ArucoPoseEstimator
```

One `StereoRigRunTime` per configured rig; `StereoCam3D` is the facade that addresses them by index.
Camera capture devices stay outside this package.

| Method | Returns |
|---|---|
| `rectify(left, right, rig=0)` | the rectified `(left, right)` pair |
| `compute_disparity(rect_left, rect_right, rig=0)` | `(disparity_px, valid_mask)`, with invalid disparity as `NaN` |
| `reproject_to_3d(disparity, rig=0)` | an `(H, W, 3)` point cloud in millimetres |
| `compute_depth_map(rect_left, rect_right, unit="mm", rig=0)` | a depth map scaled to `mm`, `cm` or `m` |
| `compute_3D_point(rect_left, rect_right, mask, reducer, unit="mm", rig=0)` | the mean or median 3D point inside a mask |
| `compute_3D_point_xy(rect_left, rect_right, x, y, unit="mm", rig=0)` | the 3D point at one pixel |
| `estimate_marker_pose_left(rect_left, target_id=None, rig=0)` | one 4x4 matrix, `None`, or a `dict[int, ndarray]` of every detected id |
| `set_extrinsics(transform, rig=0)` | stores `T_cam_to_base` for that rig; takes an `Extrinsics`, a `Transform` or a 4x4 |
| `transform_cam_to_base(point, rig=0)` | the `(3,)` point in the robot base frame |

Building it from config:

```python
from src.config import load_config
from src.calibration.stereo import StereoCam3D, StereoRigConfig

cfg = load_config()
rig_cfgs = [
    StereoRigConfig(
        stereomap_file=rig.calibration_paths.stereo_map_file,
        left_glob=rig.calibration_paths.left_images_glob,
        right_glob=rig.calibration_paths.right_images_glob,
    )
    for rig in cfg.camera.cameras.rigs
]
stereo = StereoCam3D(rigs=rig_cfgs,
                     calibration=cfg.camera.cameras.stereo_calibration,
                     stereo_matcher=cfg.camera.stereomatcher)

rect_left, rect_right = stereo.rectify(left_bgr, right_bgr, rig=0)
depth_mm = stereo.compute_depth_map(rect_left, rect_right, unit="mm", rig=0)
```

All rigs share the frame size, board geometry and matcher settings of the one `CalibrationConfig`.
`marker_length_mm` overrides it for marker pose estimation only, and `aruco_dict_name` overrides the
dictionary for pose estimation and for ChArUco calibration alike.

## The traps

**The solvers never write into a `StereoCam3D`.** Solve first, then hand the returned `Extrinsics`,
`Transform` or 4x4 to `stereo.set_extrinsics(...)` yourself, at the application boundary.

**`estimate_marker_pose_left` requires an already rectified frame.** It uses the rig's rectified
intrinsics with zero distortion; handing it a raw image gives a plausible pose that is wrong.

## Units, quality and persistence

**Units.** Stereo geometry and hand-eye translations are millimetres internally, `float64` across
the OpenCV boundary unless a caller asks for `cm` or `m`, and the conversion is pure scaling with no
rounding. The canonical table is `src/utility/unit_scaling.py`; the wrapper here re-raises an
unsupported unit as `CalibrationDataError`.

**Quality bands.** `classify_rmse(value, bands)` maps an RMSE onto a label. The millimetre defaults
are `excellent <= 1.0`, `good <= 2.5`, `marginal <= 5.0`, `poor` above that; the pixel defaults for
stereo reprojection are `0.5`, `1.0` and `2.0`. `None`, a non-finite value and a negative value all
map to `unknown`. Comparisons are inclusive, so a value sitting exactly on a boundary takes the
better label.

**Persistence is deterministic and schema-versioned.** Writes are atomic with sorted keys, so two
identical solves produce identical bytes. Extrinsics carry `willy.calibration.extrinsics/1` with the
nested geometry transform carrying its own schema, and a mismatch at either level raises
`ExtrinsicsError`. An eye-in-hand artifact carries `willy.calibration.cam_to_tool/1`, and its frame
pair is checked on load, so a CAMERA to BASE artifact cannot enter as a wrist calibration. Stereo
calibration under `willy.calibration.stereo/1` stores only the per-camera parameters, that is
intrinsics, distortion, rectification rotations, projections and `Q`; the per-pixel remap tables run
to megabytes and are recomputed on load.

## Logging

Every module that does real work, meaning it solves or it reads or writes an artifact, logs to its
own rotating file under `logs/calibration/`, with the names fixed in [`constants.py`](constants.py).
There is no aggregate file: a session runs exactly one workflow, hand-eye or stereo, so the question
afterwards is always what that one solve saw.

| File | Written by | The lines that matter |
|---|---|---|
| `eye_hand_calibrator.log` | `eye_hand/common.py` | why a sample was skipped or rejected, and the final RMSE and maximum error in mm |
| `eye_hand_dataset.log` | `eye_hand/dataset.py` | a dataset written or loaded: path, sample count, size |
| `serialization.log` | `serialization.py` | which artifact file now carries which solve quality |
| `hand_eye_axxb.log` | `solver/hand_eye_axxb.py` | pairs in, residual RMSE and solved translation norm out |
| `umeyama_rigid.log` | `solver/umeyama_rigid.py` | point pairs in, registration RMSE out |
| `stereo_factory.log` | `stereo/factory.py` | a stereomap loaded or written, and the calibration duration |
| `stereo_calibrator.log` | `stereo/sub_modules/calibrator.py` | the unusable-pair count as a warning, the reprojection RMS, the baseline |

The pure code logs nothing on purpose: the value objects `Extrinsics`, `EyeHandSample` and
`CalibrationResult`, the quality bands, `helpers.py` and `exceptions.py`.

## Where to look next

- [`scripts/examples/api/02_calibration/calibrate_fixed_camera.py`](../../scripts/examples/api/02_calibration/calibrate_fixed_camera.py), the runnable
  walkthrough of both modes against the library API
- `python -m src.robot.execution.real_cell.calibrate --rig <id> --check`, the bring-up command that
  drives a real arm through a pose sweep and writes one artifact. It validates and touches nothing
  under `--check`, and builds the arm and opens the camera but commands no motion under `--dry-run`.
  The routine and the solve have been exercised in simulation only, and nothing in this path has run
  against a physical controller
- `src/willy_sim/run_eth_calibrate.py` and `run_eih_calibrate.py`, the same routine in simulation
- [`../geometry/README.md`](../geometry/README.md), the `Frame` and `Transform` primitives every
  output here is tagged with
- [`../camera/README.md`](../camera/README.md), the capture layer that feeds these runtimes
