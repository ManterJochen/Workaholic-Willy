# Grasping geometry (`src.robot.grasping.geometry`)

The grasp-specific numeric toolbox: back-project a masked depth map into a point cloud, filter and
sub-sample it, estimate local surface normals, and validate a CAMERA to BASE rigid transform.

## What it guarantees

Pure NumPy, deterministic, and no required dependency beyond NumPy. Every point array is `(N, 3)` in
millimetres, in the camera frame unless the caller has transformed it. The filtering and sampling
helpers return keep-indices rather than new clouds, so a caller can slice its parallel attribute
arrays, points, normals and colours, in step.

Every helper validates shape, finiteness and range, and raises `ValueError` on a bad input. It never
raises the frame algebra's `InvalidPoseError`, so the whole grasping stack has one validation
contract to catch.

The frame-safe `Pose` and `Frame` algebra, compose, invert and apply, belongs to
[`src.geometry`](../../../geometry/README.md). This package deliberately does not duplicate it and
holds no pose, no quaternion and no frame composition. Nor does it hold Open3D, torch, a perception
model or a vendor SDK.

## The public surface

The stages run in this order: `pointcloud -> filters -> sampling -> normals`.

| Module | Public surface |
| --- | --- |
| `pointcloud.py` | `CameraIntrinsics`, `MaskedPointCloud`, `masked_point_cloud`, `masked_points`, `depth_unit_to_mm` |
| `filters.py` | `CloudOutlierConfig`, `apply_cloud_outlier_filter`, `filter_by_depth_range`, `radius_outlier_indices`, `statistical_outlier_indices` |
| `sampling.py` | `uniform_sample_indices`, `voxel_downsample_indices`, `farthest_point_sample_indices` |
| `normals.py` | `NormalEstimationConfig`, `SurfaceNormals`, `estimate_surface_normals` |
| `transforms.py` | `validate_transform`, the one gate a rigid 4x4 passes before any back-projection |
| `_spatial.py` | `RadiusIndex`, the shared neighbour index |
| `_validation.py` | `as_points_nx3`, `as_vec3`, `as_mask_and_depth`, the shared input validators |

The back-projection formula is in the `pointcloud` module docstring, and the local-PCA normal
estimate in `normals`.

## Usage

There is no command line here. It is a library.

```python
from src.robot.grasping.geometry import (
    CameraIntrinsics, masked_point_cloud, statistical_outlier_indices, estimate_surface_normals,
)

cloud = masked_point_cloud(mask, depth_map, CameraIntrinsics(fx, fy, cx, cy), unit="m")
keep = statistical_outlier_indices(cloud.points_mm)
normals = estimate_surface_normals(cloud.points_mm[keep], radius_mm=15.0)
```

`masked_point_cloud` drops non-finite and non-positive depths in the same pass as the optional
`min_depth_mm` and `max_depth_mm` band, and downsamples afterwards when `voxel_size_mm` is given.
`min_depth_mm` defaults to 1.0 mm, so a zero-depth pixel is never a point.

## Traps

`validate_transform` is looser than `src.geometry` on purpose: orthonormality within `1e-4` and a
determinant within `1e-3` of 1, against about `1e-6` there. A real CAMERA to BASE calibration matrix
carries more numerical slack than a freshly composed `Pose`, and otherwise good extrinsics would be
rejected at the tighter tolerance. Tighten these two constants only alongside a re-validation of what
the calibration routines write.

`estimate_surface_normals` takes either a frozen `NormalEstimationConfig` or one-off keyword
overrides. Passing both raises, rather than silently preferring one.

Local-PCA normals are noise-sensitive near a mask or object boundary. `curvature` is an
eigenvalue-ratio proxy, good for ranking and not for metric reconstruction, and the camera-facing
orientation is a heuristic rather than a topology-aware one.

SciPy is an optional accelerator. `RadiusIndex` uses `scipy.spatial.cKDTree` when it imports and
falls back to a NumPy brute-force scan otherwise. Both back ends return the same neighbour set; the
fallback is slower on a large cloud.

`CloudOutlierConfig` is off unless a caller builds one. No configuration key selects it and no
shipped caller supplies one: it reaches the pipeline only through
`GraspCalculator(cloud_outlier_filter=...)` or the same argument on `dense_surface_samples`. It
targets stereo and RGB-D speckle and edge bleed, so it is a no-op on a noise-free renderer, and it
has never been run against a physical depth camera.

## See also

- [`../README.md`](../README.md) for the pipeline this feeds
- [`../../../geometry/README.md`](../../../geometry/README.md) for the frame-safe algebra this module
  intentionally does not duplicate
- [`../contacts/README.md`](../contacts/README.md) for the contact-pair search built on these clouds
  and normals
- [`../scoring/README.md`](../scoring/README.md) for the candidate scoring downstream
