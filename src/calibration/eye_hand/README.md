# Hand-eye solving (`src/calibration/eye_hand`)

The two hand-eye workflows and the samples behind them: eye to hand for a camera fixed in the cell, eye
in hand for a camera on the flange. `HandEyeCalibration` collects the samples by moving the arm and
calls these ([`src/calibration/`](../README.md)); call them directly to solve from samples you
collected yourself.

```python
from src.calibration import EyeHandCalibrationSettings, EyeToHandCalibrator, save_extrinsics

calibrator = EyeToHandCalibrator(settings=EyeHandCalibrationSettings(min_samples=6, min_distance_mm=40.0))
for T_base_to_tool, T_cam_to_marker in samples:        # 4x4 arrays in millimetres, one pair per arm pose
    if not calibrator.add_sample(T_base_to_tool, T_cam_to_marker):
        print("skipped: no marker, or too close to a stored pose")
result = calibrator.calibrate()                         # CAMERA to BASE, with rmse_mm and quality
print(result.quality, result.rmse_mm, result.num_samples)
save_extrinsics("calibration/real/eth_overhead.json", result.to_extrinsics(rig_id="overhead"))
```

Then declare the file on the rig, as in the YAML below; until the rig declares it, the cell has no
calibration for that camera. Swap in `EyeInHandCalibrator` for a wrist camera: `result.transform` is
then CAMERA to TOOL, and it is saved with `save_cam_to_tool(path, result.transform, rig_id=...)`.

## Eye to hand and eye in hand

Both solve `AX = XB` with the same `HandEyeAXXB` solver and the same `add_sample(T_base_to_tool,
T_cam_to_marker)` call. What differs is which side is unknown, and so what the answer means when a
pick runs.

| | Eye to hand | Eye in hand |
|---|---|---|
| The camera | bolted to the cell | on the flange |
| It watches | a marker on or near the tool | a marker fixed in the workspace |
| It solves | `T_cam_to_base`, CAMERA to BASE | `T_cam_to_tool`, CAMERA to TOOL |
| When a pick runs | one static transform for every arm pose | composed with the live TCP pose, per frame |
| Calibrator | `EyeToHandCalibrator` | `EyeInHandCalibrator` |
| Resolver in the pick | `StaticCameraToBaseResolver` | `EyeInHandFrameResolver` |
| Saved with | `save_extrinsics` | `save_cam_to_tool` |

## The nouns

| Noun | Built by | Verb | Returns |
|---|---|---|---|
| `EyeToHandCalibrator`, `EyeInHandCalibrator` | `EyeToHandCalibrator(settings=)` | `add_sample(...)`, then `calibrate()` | `EyeHandCalibrationResult` |
| `EyeHandCalibrationSettings` | `EyeHandCalibrationSettings(min_samples=, min_distance_mm=, min_angle_deg=)` | | the acceptance rules; defaults 4, 10.0 mm, 5.0 deg |
| `EyeHandCalibrationResult` | `calibrate()` | `to_extrinsics(rig_id=)`, eye to hand only | `transform`, `rmse_mm`, `max_error_mm`, `num_samples`, `quality`, `captured_at` |
| `EyeHandDataset` | `calibrator.dataset`, `EyeHandDataset.load(path)` | `save(path)` | the samples, under `willy.calibration.eye_hand.dataset/1` |

A pose is stored only when it differs from every stored pose by more than the distance or the angle
threshold. Both calibrators take the robot poses you hand them and never talk to a robot driver.

## What it refuses

| Refusal | When | What to do |
|---|---|---|
| `add_sample` returns `False` | no marker was detected, or the pose is too close to a stored one | read `eye_hand_calibrator.log` for which, and move on to the next pose |
| `CalibrationDataError` from `calibrate()` | fewer than `min_samples`, or rotations about fewer than two independent axes | add poses that tilt about two axes |
| `CalibrationSolveError` | the AX=XB solve failed | check the samples; they stay in `calibrator.dataset` |
| `CalibrationDataError` from `to_extrinsics` | an eye-in-hand result; `Extrinsics` holds CAMERA to BASE only | save it with `save_cam_to_tool` |
| `CalibrationDataError` from the settings | `min_samples` below 4, the least AX=XB can be solved from | raise it |

`add_sample` returns rather than raises because a sweep of twenty poses should not stop at pose three
over one bad viewpoint.

## Several cameras: one calibration each

Each camera is calibrated on its own and declares its file on its own rig,
`camera.cameras.rigs[<id>].extrinsics`. `robot.grasping.fusion.cameras` names which cameras are fused,
by the same rig id, and holds nothing else:

```yaml
camera:
  cameras:
    primary_rig_id: cam_left
    rigs:
      - rig_id: cam_left
        extrinsics:
          mounting_mode: eye_to_hand
          artifact_path: "calibration/real/eth_cam_left.json"
      - rig_id: cam_right
        extrinsics:
          mounting_mode: eye_to_hand
          artifact_path: "calibration/real/eth_cam_right.json"
robot:
  grasping:
    fusion:
      enabled: true
      cameras:
        cam_left: {enabled: true}
        cam_right: {enabled: true}
```

An `eye_in_hand` rig also declares `shutter_motion_tolerance_mm` and `shutter_motion_tolerance_deg`, how
far the arm may move while a frame is taken; an `eye_to_hand` rig declares neither. Every reader loads
the file through `RigCalibration.from_config` in [`rig_calibration.py`](../rig_calibration.py). When the
cell is built, an enabled fused camera whose rig declares no calibration, or whose file is missing or
invalid, is refused, because a smaller camera set than the one asked for must not run silently. A
camera set `enabled: false` under `fusion.cameras` is skipped.

## Traps

**A low residual is not a good calibration.** It says the poses agree with each other, not that the
frame is right. A wrong marker size fits well and is wrong everywhere: the marker length is the printed
black square's edge in millimetres, measured on the print. What proves the frame is a pick that lands
where it was aimed.

**Swapping the two sides of AX=XB gives the exact inverse, not an error.** With the marker on the `A`
side and the gripper on the `B` side, and both index orders reversed, the solve returns the inverse of
the transform you wanted, and a tool to camera transform ships under a camera to tool label. The solver
does the linear algebra; which `X` comes out is decided by the `A` and `B` the caller builds.
[`HandEyeAXXB`](../solver/hand_eye_axxb.py) names the two pairings.

**A result that looks rotated by a right angle or by 180 degrees** is more often a wrong reference than
a wrong solve. Check what it is compared against before you touch the solver.

## Status

The legend is the root README's [Status and honest scope](../../../README.md#status-and-honest-scope).

| Capability | Evidence |
|---|---|
| Both workflows, the solve and the saved file | measured in simulation, through the calibration sweep ([guide 03](../../../docs/guide/03-calibration.md)) |
| The RGB-D marker source a real RGB-D cell sweeps with | never touched hardware; posed only a rendered marker, so its distortion residual is unconfirmed |

## Files

| File | Holds |
|---|---|
| [`eye_to_hand/calibrator.py`](eye_to_hand/calibrator.py) | `EyeToHandCalibrator` |
| [`eye_in_hand/calibrator.py`](eye_in_hand/calibrator.py) | `EyeInHandCalibrator` |
| [`common.py`](common.py) | the sample collection, the diversity filter and the solve both share |
| [`types.py`](types.py) | `MountingMode`, `EyeHandCalibrationSettings`, `EyeHandCalibrationResult` |
| [`dataset.py`](dataset.py) | `EyeHandDataset`, `EyeHandSample` |

## Details

- Why it is AX = XB, and what the residual measures: [guide 03](../../../docs/guide/03-calibration.md).
- The sweep at a real cell: [`docs/calibration-setup.md`](../../../docs/calibration-setup.md), and the
  examples [`07`](../../../examples/real_robot/07_calibrate_a_fixed_camera.py) and
  [`08`](../../../examples/real_robot/08_calibrate_a_wrist_camera.py).
- Tests: [`test_eye_hand_workflows.py`](../../../tests/test_eye_hand_workflows.py),
  [`test_multi_camera_calibration.py`](../../../tests/test_multi_camera_calibration.py),
  [`test_calibration_lives_on_the_rig.py`](../../../tests/test_calibration_lives_on_the_rig.py).
