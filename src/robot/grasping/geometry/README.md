# Grasp geometry (`src.robot.grasping.geometry`)

The numeric toolbox under grasp generation: turn a masked depth map into a point cloud, thin it,
estimate surface normals, and check a CAMERA to BASE matrix before it is used. Pure NumPy and
deterministic.

You reach it through the grasp calculator; there is no command line. Call it directly to look at the
cloud the calculator would see:

```python
from src.robot.grasping.geometry import CameraIntrinsics, estimate_surface_normals, masked_point_cloud

cloud = masked_point_cloud(mask, depth_map, CameraIntrinsics(fx, fy, cx, cy), unit="m", voxel_size_mm=4.0)
normals = estimate_surface_normals(cloud.points_mm, radius_mm=15.0)
print(cloud.points_mm.shape, normals.normals.shape)
```

`mask` is an HxW boolean or `uint8` array, `depth_map` the matching depth in the `unit` you name (`mm`,
the default, `cm` or `m`). Every point array is `(N, 3)` millimetres in the camera frame unless you
transformed it.

## The calls

The stages run in this order: point cloud, sampling, normals.

| Call | Takes | Returns |
| --- | --- | --- |
| `masked_point_cloud(mask, depth_map, intrinsics, ...)` | a depth band and an optional `voxel_size_mm` | `MaskedPointCloud`: points, pixels, depths |
| `voxel_downsample_indices(points, voxel_size_mm)` | points and a voxel size | the indices to keep, the first point of each voxel |
| `estimate_surface_normals(points, config or overrides)` | points, a `NormalEstimationConfig` | `SurfaceNormals`: normals, confidence, curvature |
| `validate_transform(T)` | a 4x4 matrix | a validated float64 copy |
| `pose_from_grasp_axes(position, approach=..., closing_axis=..., frame=...)` | a grasp's axes | the tool `Pose`: +Z approach, +X closing |

`voxel_downsample_indices` returns keep-indices rather than a new cloud, so a caller slices points,
normals and colours in step. `masked_point_cloud` drops non-finite, zero and negative depths in the same
pass as the depth band; `min_depth_mm` defaults to 1.0, so a zero-depth pixel is never a point. The
back-projection formula is in the `pointcloud` module docstring and the normal estimate in `normals`.
The outlier filters (`CloudOutlierConfig` and its index helpers), the depth-range filter and the uniform
and farthest-point samplers left on 2026-10-18 with the code nothing called.

This package holds no frame algebra. Composing, inverting and applying poses belongs to
[`src.geometry`](../../../geometry/README.md); the one `Pose` built here, in `grasp_frame.py`, is built
through it.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ValueError` | a wrong shape, a non-finite value, a range out of bounds, in any helper | fix the input; the message names it |
| `ValueError` from `validate_transform` | a rotation block not orthonormal within `1e-4`, a determinant off 1 by more than `1e-3`, a wrong last row | recheck the calibration |
| `ValueError` from `estimate_surface_normals` | both a config and keyword overrides | pass one of them |
| `ValueError` from `depth_unit_to_mm` | a unit other than `mm`, `cm` or `m` | name the unit your camera writes |

Every helper raises `ValueError`, never the frame algebra's `InvalidPoseError`, so the grasping stack
has one validation error to catch.

## Traps

- `validate_transform` is looser than `src.geometry` on purpose: `1e-4` and `1e-3` against about `1e-6`
  there. A real calibration matrix carries more slack than a freshly composed `Pose`, and good extrinsics
  would fail the tighter check. Tighten these only alongside a re-check of what the calibration writes.
- Local-PCA normals are noisy near a mask or object edge. `curvature` is an eigenvalue ratio, fit for
  ranking and not for measurement, and the camera-facing orientation is a heuristic.
- SciPy is optional. `RadiusIndex` uses `scipy.spatial.cKDTree` when it imports and a NumPy scan
  otherwise; both return the same neighbours, the scan is slower on a large cloud.

## Status

| Capability | Evidence |
| --- | --- |
| Point cloud, sampling, normals, `validate_transform` | measured in simulation: the grasp calculator runs them on every Isaac pick |

## Files

| File | Holds |
| --- | --- |
| `pointcloud.py` | `CameraIntrinsics`, `MaskedPointCloud`, `masked_point_cloud`, `masked_points`, `depth_unit_to_mm` |
| `sampling.py` | `voxel_downsample_indices` |
| `normals.py` | `NormalEstimationConfig`, `SurfaceNormals`, `estimate_surface_normals` |
| `transforms.py` | `validate_transform` |
| `grasp_frame.py` | `pose_from_grasp_axes`, which every candidate's `pose()` calls |
| `closing_axis.py` | which way a grasp closes about base Z: `closing_along`, `grasps_closing_along` and `result_closing_along` keep the grasps heading within `CLOSING_AXIS_TOLERANCE_DEG` (30, inclusive) of a named axis, each turned the named way round, and `closing_axis_said`, `closing_axis_refusal` and `same_closing_axis` say it; `turned_nearer`, `grasps_closing_toward` and `result_closing_toward` turn every grasp the way round nearer a natural direction, dropping none; `CANDIDATES_THE_AXIS_CHOOSES_AMONG` (36). The reader (`ClosingAxis`, `closing_axis_of`, `natural_closing_axis_of`, `NO_HEADING_WITHIN_DEG_OF_VERTICAL`, 10) lives in [`src/geometry/closing_axis.py`](../../../geometry/closing_axis.py) and is handed on here |
| `_spatial.py` | `RadiusIndex`, the shared neighbour index |
| `_validation.py` | `as_points_nx3`, `as_vec3`, `as_mask_and_depth`, the shared input checks |

## Details

- [`contacts/`](../contacts/README.md) builds contact pairs on these clouds and normals, and
  [`scoring/`](../scoring/README.md) holds the score record the candidates carry downstream.
- [`src.geometry`](../../../geometry/README.md) for the frame-safe algebra this package does not repeat.
- [The grasping maths](../../../../docs/grasping-math.md) for back-projection and normals.
- Tests: `tests/robot/grasping/geometry/test_geometry_hardening.py`, `tests/robot/grasping/geometry/test_surface_normals.py`, `tests/robot/grasping/contacts/test_radius_index.py`,
  `tests/robot/grasping/types/test_a_pick_closes_along_the_axis_its_program_names.py`,
  `tests/robot/execution/test_every_grasp_closes_the_way_round_the_hand_naturally_stands.py`.
