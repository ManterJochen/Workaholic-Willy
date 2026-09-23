# Camera calibration (`src/calibration`)

Answers one question: where a camera sits relative to the robot. It holds the hand-eye solvers, the
frame-tagged transforms they produce, the stereo runtime, the quality bands and the files a calibration
is saved as. You calibrate a cell's camera through `HandEyeCalibration`, which moves the arm through a
sweep and calls this package; call the package directly to solve from samples you collected yourself or
to read a saved calibration.

```python
from willy import HandEyeCalibration, SweepOptions, load_tree

calibration = HandEyeCalibration.from_tree(
    load_tree(), rig_id="overhead", mode="eye_to_hand",   # your rig id; eye_in_hand for a wrist camera
    options=SweepOptions(marker_length_mm=40.0),         # the printed marker's edge, measured on the print
)
print(calibration.check())               # the config alone: opens nothing, moves nothing
print(calibration.run(dry_run=True))     # builds the arm and opens the camera, moves nothing
```

> [!WARNING]
> `calibration.run()` without `dry_run` moves the arm through the whole sweep with the camera world
> declined, because the sweep produces the transform that world is built from. Keep the cell clear.

`run()` writes `eth_<rig>.json` or `eih_<rig>.json` and ends its report with the block to paste under the
rig in `camera.cameras.rigs`. The whole programs are
[`07_calibrate_a_fixed_camera.py`](../../examples/real_robot/07_calibrate_a_fixed_camera.py) and
[`09_calibrate_a_wrist_camera.py`](../../examples/real_robot/09_calibrate_a_wrist_camera.py); the
procedure at the cell is [`docs/calibration-setup.md`](../../docs/calibration-setup.md). From a shell:

```bash
python -m src.robot.execution.real_cell.calibrate --rig <rig id> --mode eye_to_hand --check
python -m src.robot.execution.real_cell.calibrate --rig <rig id> --mode eye_to_hand --dry-run
```

It exits 0 when done, 1 when the configuration or the build refused, another process holds the cell or
the connect refused, 2 when it ran and wrote no artifact, and 3 when the sweep raised. The sweep lines
are in [`docs/cli.md`](../../docs/cli.md).

## The nouns

| Noun | Built by | Verb | Returns |
|---|---|---|---|
| `HandEyeCalibration` | `from_tree(tree, rig_id=, mode=, options=)` | `check()`, `run(dry_run=)` | a report that ends with the rig block to paste |
| `EyeToHandCalibrator`, `EyeInHandCalibrator` | `EyeToHandCalibrator(settings=)` | `add_sample(...)`, then `calibrate()` | `EyeHandCalibrationResult`; see [`eye_hand/`](eye_hand/README.md) |
| `RigCalibration` | `camera.calibration()`, `RigCalibration.from_config(rig_id, extrinsics)` | `camera_to_base()`, `camera_to_tool()` | a frame-tagged `Transform` in millimetres |
| `Extrinsics` | `result.to_extrinsics(rig_id=)`, `load_extrinsics(path)` | `save_extrinsics(path, extrinsics)` | CAMERA to BASE with its residuals and quality label |
| `StereoCam3D` | `StereoCapturePipeline(...).run()` in [`src/camera`](../camera/README.md) | `rectify`, `compute_depth_map`, `estimate_marker_pose_left` | rectified images, depth in millimetres, marker poses |

`HandEyeCalibration` and `SweepOptions` come from `willy` ([`hand_eye.py`](../robot/execution/hand_eye.py));
everything else here imports from `src.calibration`.

## Frames

Every transform carries its `Frame` values from [`src/geometry`](../geometry/README.md). `Extrinsics`
accepts only `Transform(from_frame=Frame.CAMERA, to_frame=Frame.BASE)`; the wrong direction raises
`ExtrinsicsError` rather than giving a plausible wrong answer.

| Name | Direction | Meaning |
|---|---|---|
| `T_cam_to_marker` | CAMERA to MARKER | the marker pose in the rectified left image, as OpenCV reports it |
| `T_base_to_tool` | BASE to TOOL | one forward-kinematics sample of the robot |
| `T_cam_to_base` | CAMERA to BASE | what a fixed camera (eye to hand) solves |
| `T_cam_to_tool` | CAMERA to TOOL | what a wrist camera (eye in hand) solves |

The camera frame is the rectified left optical centre (Z forward, X right, Y down); the marker frame sits
at the marker centre, corners in `z=0`; the tool frame is the cell's flange or TCP frame.

## The stereo runtime

`StereoCam3D` holds one runtime per stereo rig and addresses them by index (`rig=0`). Each rectifies,
computes disparity, reprojects to a point cloud in millimetres and poses ArUco markers; OpenCV stays
inside [`stereo/`](stereo/).

```python
rect_left, rect_right = stereo.rectify(left_bgr, right_bgr, rig=0)
depth_mm = stereo.compute_depth_map(rect_left, rect_right, unit="mm", rig=0)
marker = stereo.estimate_marker_pose_left(rect_left, target_id=0, rig=0)   # a 4x4, or None
```

`estimate_marker_pose_left` needs a rectified image: it uses the rectified lens with zero distortion, so
a raw image gives a plausible pose that is wrong. The solvers never write into a `StereoCam3D`; hand a
solved `Extrinsics` to `stereo.set_extrinsics(...)` yourself. All rigs share one
`camera.cameras.stereo_calibration` (frame size and board) and one `camera.stereomatcher`.

## Quality bands

`classify_rmse(value, bands)` labels a residual `excellent`, `good`, `marginal`, `poor` or `unknown`.
Bounds are inclusive, and `None`, a non-finite and a negative value are `unknown`.

| Bands | Grades | Defaults |
|---|---|---|
| `QualityBandsMm` | the hand-eye residual, a self-consistency score and not a distance | 1.0, 2.5, 5.0 |
| `QualityBandsPx` | a stereo reprojection RMSE, in pixels | 0.5, 1.0, 2.0 |

A hand-eye `rmse_mm` of 2.5 does not mean 2.5 mm of error: it mixes rotation and translation and grows
with the camera's distance ([guide 03, section 3](../../docs/guide/03-calibration.md)). No band gates a
save; what proves a calibration is a pick that lands where it was aimed.

## Saved calibrations

| Schema | Holds | Written and read with |
|---|---|---|
| `willy.calibration.extrinsics/1` | CAMERA to BASE, with its residuals and quality | `save_extrinsics`, `load_extrinsics` |
| `willy.calibration.cam_to_tool/1` | CAMERA to TOOL | `save_cam_to_tool`, `load_cam_to_tool` |
| `willy.calibration.cam_to_tool/2` | CAMERA to TOOL, and the flange to TCP the solve was made against | the same two |
| `willy.calibration.eye_hand.dataset/1` | the samples of one sweep | `EyeHandDataset.save`, `EyeHandDataset.load` |
| `willy.calibration.stereo/1` | per-camera stereo parameters; the remap tables are recomputed on load | the stereo factory |

Writes are atomic with sorted keys, so two identical solves give identical bytes. A mismatched schema,
or a transform whose frames are not the ones the file claims, raises `ExtrinsicsError`, so a CAMERA to
BASE file cannot load as a wrist calibration. A cell reads a calibration only where the rig declares it,
`camera.cameras.rigs[<id>].extrinsics`, through [`rig_calibration.py`](rig_calibration.py).

## Status

The legend is the root README's [Status and honest scope](../../README.md#status-and-honest-scope).

| Capability | Evidence |
|---|---|
| Eye-to-hand and eye-in-hand calibration through the sweep | measured in simulation ([guide 03, section 6](../../docs/guide/03-calibration.md)) |
| `HandEyeCalibration` and the `calibrate` command at a real cell | never touched hardware; `--check` and `--dry-run` command no motion |
| The solvers, the frame checks and the file round trips | measured in simulation within the sweep; also pinned by [`test_calibration_core.py`](../../tests/test_calibration_core.py) |
| `RGBDArucoMarkerSource`, the marker source of an RGB-D cell | never touched hardware; posed only a rendered marker ([`test_rgbd_marker_source.py`](../../tests/test_rgbd_marker_source.py)) |
| `CharucoPoseEstimator`, a ChArUco board as the sweep's target | never touched hardware; posed a board rendered with a known camera and pose to within 1 mm at 520 mm, 0.30 px ([`test_calibration_targets.py`](../../tests/test_calibration_targets.py)) |

## Files

| File | Holds |
|---|---|
| [`eye_hand/`](eye_hand/README.md) | the two hand-eye workflows, their settings, results and dataset |
| [`solver/`](solver/) | `HandEyeAXXB` for AX=XB and `UmeyamaRigid` for point-set registration, NumPy only |
| [`stereo/`](stereo/) | `StereoCam3D`, `StereoRigConfig`, `CalibrationResult` and the ArUco pose estimator |
| [`extrinsics.py`](extrinsics.py) | `Extrinsics` |
| [`rig_calibration.py`](rig_calibration.py) | `RigCalibration`, `RigNotCalibrated`, `RigCalibrationError`: the one loader of a rig's calibration |
| [`rgbd_marker_source.py`](rgbd_marker_source.py) | `RGBDArucoMarkerSource`, the target poses of an RGB-D camera for the sweep, and what each judged frame showed |
| [`targets.py`](targets.py) | `Observation`, `CharucoPoseEstimator`, `estimator_for`, `parse_board_spec`: what a sweep looks at, one marker or a ChArUco board |
| [`serialization.py`](serialization.py) | the save and load functions above |
| [`quality.py`](quality.py) | `classify_rmse`, `QualityBandsMm`, `QualityBandsPx` |
| [`helpers.py`](helpers.py), [`exceptions.py`](exceptions.py) | `unit_scaling`, `proj_to_K`; `CalibrationError` and its subclasses |

Every module that solves or reads or writes a file logs to its own file under `logs/calibration/`, named
in [`constants.py`](constants.py); `eye_hand_calibrator.log` says why a sample was skipped or rejected.
The package imports no torch, so it runs on any machine.

## Details

- The derivations, the residual and what has run: [guide 03](../../docs/guide/03-calibration.md).
- Calibrating a real cell step by step: [`docs/calibration-setup.md`](../../docs/calibration-setup.md).
- The same sweep in simulation: `run_eth_calibrate.py` and `run_eih_calibrate.py` in [`src/willy_sim/`](../willy_sim/README.md).
- The capture layer that feeds these runtimes: [`src/camera/`](../camera/README.md).
- Tests: [`test_calibration_core.py`](../../tests/test_calibration_core.py),
  [`test_hand_eye_calibration.py`](../../tests/test_hand_eye_calibration.py),
  [`test_rig_extrinsics.py`](../../tests/test_rig_extrinsics.py).
