# Poses, frames and transforms (`src/geometry`)

Every position, orientation and rigid transform in this repository, each tagged with the coordinate frame
it lives in. Millimetres, unit XYZW quaternions with `w >= 0`, radians, `float64`, and nothing guessed.
It owns no camera IO, calibration workflow, driver or model inference.

```python
from willy import Frame, Pose
from src.geometry import Transform, transform_from_dict, transform_to_dict

target = Pose.tool_down(450.0, 100.0, 300.0)       # mm in Frame.BASE, the tool's +Z pointing down
camera_to_base = Transform.from_matrix(matrix_4x4, from_frame=Frame.CAMERA, to_frame=Frame.BASE)
point_base_mm = camera_to_base.apply_point(point_camera_mm)
target_in_camera = camera_to_base.inverse().apply_pose(target)

blob = transform_to_dict(camera_to_base)           # {"schema": "willy.geometry.transform/1", ...}
same = transform_from_dict(blob)                   # a different schema is refused on load
```

`Frame` and `Pose` are public through `willy`: every motion verb takes a `Pose`, as in
[03_connect_and_move.py](../../examples/real_robot/03_connect_and_move.py). `Transform`, the quaternion
helpers and the serializers import from `src.geometry`.

## The one convention to learn

`Transform(from_frame=A, to_frame=B)` maps a point expressed in `A` into `B`: the matrix usually
written `T_B_A`. `a.compose(b)` applies `a` first, then `b`. If `a` runs `A` to `B` and `b` runs `B` to
`C`, the result runs `A` to `C` and equals `b.to_matrix() @ a.to_matrix()`. Frames that do not join
raise `InvalidTransformError` rather than returning a plausible wrong answer, and `apply_pose` refuses
a pose in the wrong frame the same way.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `Frame` | a member: `WORLD`, `BASE`, `CAMERA`, `MARKER`, `TCP`, `TOOL`, `OBJECT`, `GRASP` | used as a tag | a `StrEnum` |
| `Pose` | `Pose(position_mm, quaternion_xyzw, frame)`, `tool_down(x, y, z)`, `identity`, `from_matrix` | `distance_to`, `angle_to`, `with_frame` | an immutable pose |
| `Transform` | `Transform(...)`, `identity`, `from_matrix(T, from_frame=, to_frame=)` | `compose`, `inverse`, `apply_point`, `apply_pose` | an immutable transform |

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `InvalidPoseError` | a position that is not finite or not `(3,)` | pass a finite `(3,)` array in millimetres |
| `InvalidQuaternionError` | a quaternion not finite, not `(4,)`, or not unit within `1e-6` | normalise it at the boundary |
| `InvalidMatrixError` | a rotation not orthonormal or with `det` away from `+1`, or a non-rigid 4x4 | re-orthonormalise the matrix where it enters |
| `InvalidTransformError` | a composition or an `apply_pose` whose frames do not join | compose in the order the frames run |

Constructors normalise and canonicalise a quaternion on purpose, because equality and hashing compare
the arrays byte-wise. Nothing repairs a bad rotation matrix. Both arrays of a `Pose` and a `Transform`
are read-only, so a value you were handed cannot change under you.

## Traps

**This tolerance is tighter than the grasp calculator's.** `validation.py` holds `1e-6` for unit length,
orthonormality and determinant. `src/robot/grasping/geometry/transforms.py` accepts orthonormality
within `1e-4` and a determinant within `1e-3`, because a CAMERA to BASE matrix from a hand-eye solve
carries more numerical slack than a freshly composed `Pose`. Passing such a matrix here unchanged turns
a good calibration into a refused motion: re-orthonormalise it at the boundary rather than loosening
the check.

**Euler angles and bare 4x4 matrices are boundary helpers, never storage.** `matrix.py` and `adapters/`
exist for OpenCV, for eye-to-hand calibration and for tight loops. A matrix carries no frame, so frame
agreement becomes the caller's job the moment you drop to one. Everywhere else, use `Transform`.

**The wire format is versioned.** Dicts carry `willy.geometry.pose/1` or `willy.geometry.transform/1`,
and a mismatch is refused on load, so a later layout cannot be misread as this one. Serialization keeps
full-precision floats.

**No SciPy, and no vendor types.** Every operation is closed-form NumPy. A vendor pose type, such as the
UR axis-angle pose, stays in its driver; generic numeric bridges belong in `conversions.py` and `adapters/`.

At run time the package imports NumPy and its own modules only. The one outward reference is a
typing-only annotation of `Extrinsics` in `adapters/matrix.py`.

## Files

| File | Holds |
| --- | --- |
| [`frame.py`](frame.py) | `Frame`, the coordinate frames as a `StrEnum`; no raw-string frames in the public API |
| [`pose.py`](pose.py) | `Pose`, an immutable frame-tagged pose with an optional label |
| [`transform.py`](transform.py) | `Transform`, an immutable rigid transform between two frames |
| [`quaternion.py`](quaternion.py) | XYZW algebra, and axis-angle, Euler, rotation vector and matrix conversions |
| [`matrix.py`](matrix.py) | homogeneous 4x4 helpers for boundary code and inner loops; no frames |
| [`validation.py`](validation.py) | shape, finiteness, rotation and frame checks, and the three tolerances |
| [`serialization.py`](serialization.py) | the versioned dict form of `Pose` and `Transform` |
| [`conversions.py`](conversions.py), [`adapters/`](adapters/) | the only modules that know raw 4x4 arrays, axis-angle vectors and `Extrinsics` |
| [`exceptions.py`](exceptions.py) | `GeometryError` and its subclasses, `FrameMismatchError` among them |

## Details

- [`../calibration/README.md`](../calibration/README.md) stores every hand-eye result as a `Transform` or an `Extrinsics`
- [`../utility/README.md`](../utility/README.md) holds the millimetre scale factors these units are defined against
- Maths: [safety-math.md](../../docs/safety-math.md) and [grasping-math.md](../../docs/grasping-math.md)
- Tests: `tests/test_geometry_hardening.py`, `tests/test_geometry_precision.py`, `tests/test_geometry_error_types.py`
