# Grasp contacts (`src.robot.grasping.contacts`)

Where a parallel jaw could touch a part: contact points, pairs of opposing contacts, and the collision
cloud of everything that is not the target. It says where a jaw could touch; whether the grasp can be
executed is decided downstream.

The grasp calculator builds its cloud of the other parts with `scene_collision_cloud`, and the
multi-finger planners' `ParallelJawContactPlanner` searches its pairs with `find_antipodal_pairs`. The
dense surface sampler that fed the search a graspability-weighted cloud of one part left on 2026-10-18
with the code nothing called. Call the search directly to see the pairs one mask yields:

```python
from src.robot.grasping.contacts import find_antipodal_pairs
from src.robot.grasping.geometry import CameraIntrinsics, estimate_surface_normals, masked_point_cloud

cloud = masked_point_cloud(mask, depth_mm, CameraIntrinsics(fx, fy, cx, cy), voxel_size_mm=4.0)
normals = estimate_surface_normals(cloud.points_mm, radius_mm=15.0)
pairs = find_antipodal_pairs(cloud.points_mm, normals, max_width_mm=85.0)
print(cloud.size, len(pairs))
if pairs:
    best = pairs[0]              # best score first, then the narrower width, ties broken by index
    print(best.distance_mm, best.antipodal_score)
```

A flat face seen from straight above yields no pair, because the two contacts must face apart; one view
sees one side of a part, which is what [`multiview/`](../multiview/README.md) addresses.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
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
cheap and deterministic on purpose: learning the prior itself would need a corpus of real grasp outcomes (the
learned success model that re-weighted the same signals downstream left on 2026-10-16).

Suction is not built on this. A suction grasp is a contact plus a pressing direction, not an outward
normal, and it carries its own seal physics, so it lives in [`suction/`](../suction/README.md).

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ValueError` from `ContactPoint` | a non-finite point, or a normal more than 1 % off unit length | normalise the normal |
| `ValueError` from `ContactPair` | `distance_mm` not the distance between the points, or a score outside `[0, 1]` | build pairs with `find_antipodal_pairs` |
| `ValueError` from `find_antipodal_pairs` | a negative minimum width, or a maximum not above the minimum | fix the width range |

## Traps

- Pass a `SurfaceNormals` bundle, not a raw `(N, 3)` array. A raw array carries no confidence and no
  curvature, so the stability term becomes a constant and the ranking quietly stops discriminating.
  Nothing raises.
- The search assumes locally well-behaved surfaces. Concave parts, and transparent or reflective ones
  whose depth is unreliable, degrade the result.
- Parallel-jaw contacts only. Multi-finger planning builds on these primitives in its own module.

## Status

| Capability | Evidence |
| --- | --- |
| The collision cloud | measured in simulation: the grasp calculator builds it on the Isaac picks |
| The antipodal search | never touched hardware: the multi-finger planners call it, and no pick path plans with them |
| On a physical depth camera | run on a physical cell: the calculator built the collision cloud on the camera picks of a wrist D415 on a UR10 (CB3); nothing measured there is kept here |

## Files

| File | Holds |
| --- | --- |
| `contact_point.py` | `ContactPoint`, and the shared `as_unit_normal` check |
| `contact_pair.py` | `ContactPair` |
| `antipodal.py` | `find_antipodal_pairs`: the radius search, the admission gates and the score |
| `dense_sampler.py` | `scene_collision_cloud` |

## Details

- [`generation/`](../generation/README.md) for the calculator that keeps the hand off the collision
  cloud, and [`planning/`](../planning/README.md) for the multi-finger planners built on the pairs.
- [The grasping maths](../../../../docs/grasping-math.md) for the antipodal conditions.
- Tests: `tests/robot/grasping/contacts/test_contact_point.py`, `tests/robot/grasping/test_dense_sampler.py`, `tests/robot/grasping/contacts/test_radius_index.py`.
