# Grasp contacts (`src.robot.grasping.contacts`)

Where a parallel jaw could touch a part: contact points, pairs of opposing contacts, a dense
graspability-weighted sampling of one object's surface, and the collision cloud of everything that is
not the target. It says where a jaw could touch; whether the grasp can be executed is decided downstream.

You reach it through the grasp calculator. Call it directly to see the pairs one mask yields:

```python
from src.robot.grasping.contacts import dense_surface_samples, find_antipodal_pairs

samples = dense_surface_samples(mask, depth_mm, intrinsics)     # one part; depth in mm by default
pairs = find_antipodal_pairs(samples.points_mm, samples.normals, valid_mask=samples.graspability)
print(samples.size, len(pairs))
if pairs:
    best = pairs[0]              # best score first, then the narrower width, ties broken by index
    print(best.distance_mm, best.antipodal_score)
```

`intrinsics` is a `geometry.CameraIntrinsics` or a 3x3 matrix. A flat face seen from straight above
yields no pair, because the two contacts must face apart; one view sees one side of a part, which is
what [`multiview/`](../multiview/README.md) addresses.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `SurfaceSamples` | `dense_surface_samples(mask, depth_map, intrinsics, ...)` | read `points_mm`, `normals`, `graspability` | one object's surface |
| `ContactPair` | `find_antipodal_pairs(points_mm, normals, ...)` | read `distance_mm`, `antipodal_score` | the pairs, best first |
| `ContactPoint` | `ContactPoint(point_mm=..., normal=...)` | two make a `ContactPair` | one validated contact |
| a collision cloud | `scene_collision_cloud(object_masks, depth_map, intrinsics, target_index=...)` | hand it to the collision filter | `(N, 3)` mm of every mask but the target |

A `ContactPoint` accepts a normal within 1 % of unit length and stores it exactly unit. The same check,
`as_unit_normal`, validates every contact of a multi-finger `MultiContactGrasp`. `ContactPair`
checks both contacts, that `distance_mm` is the distance between them, and that every score lies in
`[0, 1]`. `ContactPair`, `ContactPoint` and `find_antipodal_pairs` are also exported from
`src.robot.grasping`.

The search runs in whatever frame the caller supplies, camera or base, as long as points and normals
share it. It is NumPy and deterministic, in millimetres. Its only dependencies are NumPy and
[`geometry/`](../geometry/README.md); SciPy speeds up the neighbour search when it is installed.

## How a pair is scored

A pair is admitted only at an opposition of at least 0.8 and an axis alignment of at least 0.5, and scored as
`0.45 * opposition + 0.35 * axis_alignment + 0.20 * stability`. The weights are hand-set priors, kept
cheap and deterministic on purpose: the learned success model downstream re-weights the same signals as
separate features, and learning the prior itself would need a corpus of real grasp outcomes.

Suction is not built on this. A suction grasp is a contact plus a pressing direction, not an outward
normal, and it carries its own seal physics, so it lives in [`suction/`](../suction/README.md).

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ValueError` from `ContactPoint` | a non-finite point, or a normal more than 1 % off unit length | normalise the normal |
| `ValueError` from `ContactPair` | `distance_mm` not the distance between the points, or a score outside `[0, 1]` | build pairs with `find_antipodal_pairs` |
| `ValueError` from `find_antipodal_pairs` | a negative minimum width, or a maximum not above the minimum | fix the width range |
| `ValueError` from `dense_surface_samples` | a negative erosion, or `max_points` below 1 | fix the argument |

## Traps

- Pass a `SurfaceNormals` bundle, not a raw `(N, 3)` array. A raw array carries no confidence and no
  curvature, so the stability term becomes a constant and the ranking quietly stops discriminating.
  Nothing raises.
- The search assumes locally well-behaved surfaces. Concave parts, and transparent or reflective ones
  whose depth is unreliable, degrade the result.
- Parallel-jaw contacts only. Multi-finger planning and the force-closure certificate build on these
  primitives in their own modules.

## Status

| Capability | Evidence |
| --- | --- |
| Dense sampling, the antipodal search, the collision cloud | measured in simulation: the grasp calculator runs them on the Isaac picks |
| On a physical depth camera | never touched hardware: no real frame has reached them |

## Files

| File | Holds |
| --- | --- |
| `contact_point.py` | `ContactPoint`, and the shared `as_unit_normal` check |
| `contact_pair.py` | `ContactPair` |
| `antipodal.py` | `find_antipodal_pairs`: the radius search, the admission gates and the score |
| `dense_sampler.py` | `SurfaceSamples`, `dense_surface_samples`, `scene_collision_cloud`, the erosion and depth-edge helpers |

## Details

- [`generation/`](../generation/README.md) for the calculator that turns pairs into candidates, and
  [`planning/`](../planning/README.md) for the poses built from them.
- [`scoring/`](../scoring/README.md) for `force_closure`, which certifies a `ContactPair`.
- [The grasping maths](../../../../docs/grasping-math.md) for the antipodal conditions.
- Tests: `tests/test_contact_point.py`, `tests/test_dense_sampler.py`, `tests/test_force_closure.py`,
  `tests/test_radius_index.py`.
