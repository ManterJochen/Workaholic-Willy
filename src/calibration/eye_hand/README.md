# `src/calibration/eye_hand/`: where the camera is, relative to the robot

The two hand-eye workflows and the dataset behind them. Both solve `AX = XB` with the same
pure-NumPy `HandEyeAXXB` solver and the same `add_sample` call; what differs is which side is
unknown, and therefore what the answer means at run time.

The typed extrinsics these workflows produce, the stereo runtime that feeds them their marker
poses, the quality bands and the persistence all sit one level up, in
[`src/calibration/`](../README.md).

## Eye-to-hand and eye-in-hand

Both workflows use the same pure-NumPy `HandEyeAXXB` solver and the same
`add_sample(T_base_to_tool, T_cam_to_marker)` call. What differs is which side of `AX = XB` is
unknown, and therefore what the answer means at run time.

| | Eye-to-hand | Eye-in-hand |
|---|---|---|
| Where the camera is | bolted to the cell | on the flange |
| What it watches | a marker on or near the tool | a fixed marker in the workspace |
| What it solves | `T_cam_to_base`, CAMERA to BASE | `T_cam_to_tool`, CAMERA to TOOL |
| At run time | one static transform, valid for every arm pose | composed with the live TCP pose, recomputed per frame |
| Class | `EyeToHandCalibrator` | `EyeInHandCalibrator` |
| Resolver | `StaticCameraToBaseResolver` | `EyeInHandFrameResolver` |
| Artifact written by | `save_extrinsics` | `save_cam_to_tool` |

```python
from src.calibration.eye_hand import EyeHandCalibrationSettings, EyeToHandCalibrator

calibrator = EyeToHandCalibrator(
    settings=EyeHandCalibrationSettings(min_samples=6, min_distance_mm=40.0, min_angle_deg=10.0)
)
for T_base_to_tool, T_cam_to_marker in samples:
    calibrator.add_sample(T_base_to_tool, T_cam_to_marker)

result = calibrator.calibrate()
extrinsics = result.to_extrinsics(rig_id="rig-0")
stereo.set_extrinsics(extrinsics, rig=0)          # wired in explicitly, at the application boundary
```

Swap in `EyeInHandCalibrator` and `result.transform` is `T_cam_to_tool` instead. Only an eye-to-hand
result converts through `to_extrinsics`, because an eye-in-hand transform ends at the tool and
`Extrinsics` holds CAMERA to BASE.

Both workflows store their samples in `EyeHandDataset` under schema
`willy.calibration.eye_hand.dataset/1`; validate sample count, finiteness, homogeneity, rotation
matrices and motion diversity; return an `EyeHandCalibrationResult` carrying `rmse_mm`,
`max_error_mm`, `num_samples`, `quality` and `captured_at`; and depend only on the robot pose
matrices the caller supplies, never on a robot vendor implementation.

The acceptance settings default to `min_samples=4`, `min_distance_mm=10.0`, `min_angle_deg=5.0`. Four
is the floor and is enforced: it is the smallest set AX=XB can be solved from. A pose is stored only
if it differs from every stored pose by more than the distance or the angle threshold.

## Multi-camera: one calibration per camera

A multi-view cell calibrates each camera individually and declares them in one map under
`robot.grasping.fusion`, keyed by camera id:

```yaml
robot:
  grasping:
    fusion:
      enabled: true
      extrinsics_artifact_path: "calibration/real/eth_cam_left.json"   # the primary camera
      cameras:                                                        # every camera except the primary
        cam_right:
          enabled: true
          mounting_mode: "eye_to_hand"
          extrinsics_artifact_path: "calibration/real/eth_cam_right.json"
```

| `mounting_mode` | The artifact holds | Written and read with |
|---|---|---|
| `eye_to_hand` | a CAMERA to BASE `Extrinsics` | `save_extrinsics`, `load_extrinsics` |
| `eye_in_hand` | a CAMERA to TOOL `Transform` | `save_cam_to_tool`, `load_cam_to_tool` |

`build_config_frame_resolvers` in `src/robot/execution/autonomous_grasp/builders.py` turns that map
into `{camera_id -> FrameResolver}`. A disabled camera is skipped; an enabled camera whose artifact
is missing, stale or invalid raises at construction, because an incomplete resolver map must not run
silently as a smaller rig than was asked for.

## The traps

**A low RMSE is not a good calibration.** It says the poses are self consistent, not that the frame
is right. A systematically wrong marker size fits beautifully and is wrong everywhere. The marker
length is the printed black square's edge in millimetres, not the white border and not what the PDF
was named; measure the print. What proves the frame is right is a pick that lands where it was
aimed.

**A wrong reference is a harder bug than a wrong solver.** When a hand-eye result looks rotated by a
right angle or by 180 degrees, check what it is being compared against before you touch the solve.

**Swapping the two sides of AX=XB gives the exact inverse, not an error.** Marker on the `A` side and
gripper on the `B` side, with both index orders reversed, is solved by the inverse of the transform
you wanted, so nothing raises and a `T_tool_to_cam` ships under a `T_cam_to_tool` label. The solver
is convention-agnostic on purpose: it runs the linear algebra, and which `X` comes out is decided
entirely by the `A` and `B` the caller builds. `HandEyeAXXB` names the two pairings.

**`add_sample` returns `False`, it does not raise**, when no marker was detected or the pose is not
diverse enough. A routine that raised on a bad viewpoint would abort a twenty-pose sweep at pose
three. Check the return value, and read the log line for the reason.

**The primary camera's artifact does not live in the `cameras` map.** It is the separate
`fusion.extrinsics_artifact_path` key, and it is the only one that satisfies the CAMERA to BASE
refusal when a real cell is built. A cell can be fully calibrated, load every artifact in the map,
and still be refused at build because that key is null.

**The RGB-D marker source has never run against a physical camera.** `RGBDArucoMarkerSource` fills
the hand-eye routine's `marker_source` seam for a cell whose camera is RGB-D rather than a stereo
rig, and it has been exercised offline against a rendered marker only. Its distortion path runs, but
an aligned stream reports coefficients near zero, so the residual is unconfirmed on a real bench.
