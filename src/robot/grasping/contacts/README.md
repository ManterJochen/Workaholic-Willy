# Grasp contacts (`src.robot.grasping.contacts`)

Geometry-only primitives that turn a point cloud plus surface normals into the raw material of a
grasp: contact points, antipodal contact pairs, a graspability-weighted dense surface sampling, and
the collision cloud of everything that is not the target.

## What it guarantees

It answers where a gripper could touch this object. It does not answer whether the grasp is
executable, which is decided downstream by the calculator, the collision filter and the safety
layer. There is no reachability here, no collision resolution, no motion and no segmentation
runtime.

Pure NumPy and deterministic. Millimetres throughout. The search runs in whatever frame the caller
supplies, camera before the CAMERA to BASE transform and base after, so the package is indifferent to
an eye-to-hand or an eye-in-hand rig; the caller keeps points and normals in one frame.

Its only dependencies are NumPy and [`geometry/`](../geometry/README.md). SciPy's `cKDTree` is an
optional accelerator behind `geometry.RadiusIndex`, and the mask morphology is hand-rolled NumPy.

## The public surface

| File | Role |
| --- | --- |
| `contact_point.py` | `ContactPoint`, one finite position with an outward normal, plus the shared `as_unit_normal` validator. A normal is accepted within 1 % of unit length and stored exactly unit. |
| `contact_pair.py` | `ContactPair`, two opposing contacts for a parallel jaw. It validates both contacts, that `distance_mm` equals the distance between them, and that every score lies in `[0, 1]`. |
| `antipodal.py` | `find_antipodal_pairs`, a deterministic radius search with the admission gates and the score blend. |
| `dense_sampler.py` | `SurfaceSamples`, `dense_surface_samples`, `scene_collision_cloud`, and the erosion and depth-discontinuity helpers. |

`ContactPair`, `ContactPoint` and `find_antipodal_pairs` are re-exported from the `src.robot.grasping`
package root.

`ContactPoint` is the shared element: `ContactPair` composes two of them, and
`planning/multifinger.py`'s `MultiContactGrasp` composes N, so a contact is validated the same way
everywhere.

Suction is deliberately not built on this. A suction grasp is a contact position plus a pressing
direction, which is not an outward surface normal, and it carries its own seal and wrench physics.
It lives in [`suction/`](../suction/README.md) as `SuctionGrasp`. Forcing a surface-normal contact
onto it would be a category error.

## Usage

```python
from src.robot.grasping.contacts import dense_surface_samples, find_antipodal_pairs

# One mask plus depth plus intrinsics gives a graspability-weighted surface cloud.
samples = dense_surface_samples(mask, depth_map, intrinsics)

# graspability in [0, 1] doubles as the valid_mask.
pairs = find_antipodal_pairs(
    samples.points_mm,
    samples.normals,
    valid_mask=samples.graspability,
)
best = pairs[0]  # sorted by antipodal_score, then width, then index; ties broken deterministically
```

`scene_collision_cloud(object_masks, depth_map, intrinsics, target_index=...)` returns the `(N, 3)`
millimetre cloud of every mask except the target, plus an optional extra scene mask, for the
collision filter.

## Traps

Pass a `SurfaceNormals` bundle, not a raw `(N, 3)` array. A raw array carries no confidence and no
curvature, so the stability term of the score collapses to a constant and the pairs quietly get
worse. Nothing raises; the ranking just stops discriminating.

The scoring weights are hand-tuned priors, not learned: the blend is
`0.45 * opposition + 0.35 * axis_alignment + 0.20 * stability`, admitted only above an opposition of
0.8 and an axis alignment of 0.5. That is deliberate. The calibrated success model downstream
re-weights the same signals as independent features, so this layer stays a cheap deterministic
pre-filter. Learning the prior itself needs a corpus of real grasp outcomes.

The search assumes locally well-behaved surfaces. Concave geometry, and transparent or reflective
material where depth is unreliable and therefore normals are too, degrade the result.

Parallel-jaw contacts only. Multi-finger planning and the force-closure certificate build on these
primitives but live in their own modules, and this package never reaches up into them.

## See also

- [`../README.md`](../README.md) for the tier this package sits in
- [`../geometry/README.md`](../geometry/README.md) for the NumPy sub-layer it consumes
- [`../generation/README.md`](../generation/README.md) for the calculator that turns pairs into
  candidates
- [`../scoring/README.md`](../scoring/README.md) for `force_closure`, which certifies a `ContactPair`
- [`../suction/README.md`](../suction/README.md) for the other end-effector modality
