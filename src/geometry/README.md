# `src/geometry/`: frame-safe SE(3) primitives

Poses, transforms and quaternions, each tagged with the coordinate frame it lives in. The single
source of truth for robotics geometry in this repository.

`NumPy only, float64` `millimetres` `unit XYZW quaternion, canonical w >= 0` `no SciPy`
`schema-versioned wire format`

```python
from src.geometry import Frame, Pose, Transform
```

## What it guarantees

Every public value carries its frame, its unit and its dtype, and none of the three is ever guessed.
Translations are millimetres, orientation is a unit XYZW quaternion with canonical sign (`w >= 0`),
angles are radians, and everything is `float64`. Both ndarray fields of a `Pose` and a `Transform`
are made read-only at construction, so a consumer cannot mutate a value it was handed. Nothing here
rounds, and serialization stores full-precision floats.

The package sits near the bottom of the dependency stack. At run time it imports NumPy and its own
modules and nothing else in this repository, and perception, calibration, the robot drivers and the
grasping stack all import it. The single outward reference is a typing-only annotation in
[`adapters/matrix.py`](adapters/matrix.py) naming `Extrinsics`, which adds no run-time edge.

It deliberately does not own camera IO, calibration workflows, robot drivers, model inference or
application orchestration.

## What is in it

| File | Owns |
|---|---|
| [`frame.py`](frame.py) | `Frame`, the canonical coordinate-frame `StrEnum`: `WORLD`, `BASE`, `CAMERA`, `MARKER`, `TCP`, `TOOL`, `OBJECT`, `GRASP`. No raw-string frames anywhere in the public API |
| [`pose.py`](pose.py) | `Pose`, an immutable frame-tagged 6-DoF pose: position, XYZW orientation, frame, optional label |
| [`transform.py`](transform.py) | `Transform`, an immutable typed rigid transform between two frames |
| [`quaternion.py`](quaternion.py) | XYZW quaternion algebra plus axis-angle, Euler, rotation-vector and rotation-matrix conversions |
| [`matrix.py`](matrix.py) | Homogeneous 4x4 helpers for boundary code that speaks plain ndarrays, and for inner loops where one matrix multiply beats allocating a `Transform`. Nothing here carries a frame |
| [`validation.py`](validation.py) | Shape, finiteness, quaternion, rotation, frame-chain and homogeneous-matrix checks, and the three tolerance constants |
| [`serialization.py`](serialization.py) | Schema-versioned dict conversion for `Pose` and `Transform` |
| [`conversions.py`](conversions.py), [`adapters/`](adapters/) | The only modules allowed to know about raw 4x4 ndarrays, axis-angle vectors and the `Extrinsics` boundary |
| [`exceptions.py`](exceptions.py) | `GeometryError` and its subclasses: `InvalidPoseError`, `InvalidQuaternionError`, `InvalidMatrixError`, `InvalidTransformError`, `FrameMismatchError` |

## The one convention to internalise

`Transform(from_frame=A, to_frame=B)` maps a point expressed in frame `A` into frame `B`. That is
the matrix usually written `T_B_A`.

`a.compose(b)` means apply `a` first, then `b`. If `a` runs `A` to `B` and `b` runs `B` to `C`, the
result runs `A` to `C`, and it equals the matrix product `b.to_matrix() @ a.to_matrix()`. A frame
pair that does not join raises `InvalidTransformError` rather than producing a plausible wrong
answer, and `apply_pose` refuses the same way.

```python
import numpy as np
from src.geometry import Frame, Transform

T_cam_to_base = Transform.from_matrix(matrix_4x4, from_frame=Frame.CAMERA, to_frame=Frame.BASE)
point_base_mm = T_cam_to_base.apply_point(point_camera_mm)
```

```python
from src.geometry import transform_to_dict, transform_from_dict

blob = transform_to_dict(T_cam_to_base)   # {"schema": "willy.geometry.transform/1", ...}
same = transform_from_dict(blob)          # a mismatched schema is refused on load
```

## The traps

**Strict validation, and no silent repair.** Positions must be finite and `(3,)`; quaternions finite,
`(4,)` and unit within `1e-6`; rotation matrices orthonormal with `det` near `+1`; homogeneous
matrices rigid 4x4 with a `[0, 0, 0, 1]` bottom row. Constructors do normalise and canonicalise a
quaternion deliberately, because equality and hashing compare the arrays byte-wise and that is only
well defined under a fixed sign convention. No helper repairs a bad rotation matrix.

**This package's tolerance is deliberately tighter than the grasp calculator's.** `validation.py`
uses `1e-6` for quaternion unit length, orthonormality and determinant.
`src/robot/grasping/geometry/transforms.py` accepts orthonormality within `1e-4` and a determinant
within `1e-3`, because a real CAMERA to BASE matrix produced by a hand-eye solve carries more
numerical slack than a freshly composed `Pose`. Composing the two without noticing the difference
turns a good calibration into a refused motion. The fix is to re-orthonormalise at the boundary, not
to loosen the check here.

**Euler angles are a boundary helper, never a storage format.** So is a bare 4x4. `matrix.py` and
the adapters exist for OpenCV, for eye-to-hand calibration and for tight loops; everywhere else, use
`Transform`, which carries explicit frames and is checked at every step. A matrix cannot carry a
frame, so frame agreement becomes the caller's obligation the moment you drop to one.

**The wire format is versioned, and the version is checked.** Dicts carry
`willy.geometry.pose/1` and `willy.geometry.transform/1`, and a mismatch is rejected on load, so a
future change to ordering or to a covariance field cannot be misread as the current one.

**SciPy is not used here.** Every operation is a closed-form rigid-body conversion in NumPy
`float64`. SciPy is used elsewhere in the tree, mostly behind deferred imports; if it is ever needed
for geometry, isolate it behind an adapter rather than importing it into these modules.

**Vendor pose types stay behind their drivers.** Cross-cutting numeric bridges belong in
`conversions.py` and `adapters/`; a vendor-specific adapter, such as the UR axis-angle pose, belongs
in that driver's package.

## Where to look next

- [`../calibration/README.md`](../calibration/README.md), which stores every hand-eye output as a
  `Transform` or an `Extrinsics`
- [`../camera/README.md`](../camera/README.md), which returns image frames only and no robotics
  transforms
- [`../utility/README.md`](../utility/README.md), for the millimetre scaling table these units are
  defined against
