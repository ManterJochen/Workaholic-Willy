# Suction (`src.robot.grasping.suction`)

Turn a perceived object into ranked suction candidates: flat, sealable surface patches approached
along the local surface normal. It answers where the cup can seal and hold. Performing the attach and
the lift is the driver and the runner job.

## What it guarantees

A second end-effector modality rather than a flag on the parallel-jaw path. Where a jaw grasp has an
aperture, an antipodal pair and a closing direction, a suction grasp is one sealable contact and a
press direction, so it has none of those. It covers what a two-finger gripper is weakest at: wide
flat objects and top-face bin picks.

Pure NumPy, deterministic, no robot and no simulator import. Camera frame and millimetres, or BASE
when a `camera_to_base` transform is supplied and only the winners are transformed.

The quality model is analytical physics written from the published equations. There is no learned
scorer here, and that is a licensing decision rather than an oversight; see the trap below.

## The model

Back-project the mask, estimate normals and curvature, take a uniform spatial subsample of the
surface as candidate contacts, and score each one:

```
seal(flatness, perimeter support, normal alignment) x wrench(gravity + payload about the contact)
```

Without a payload, `quality` is the seal score alone. With `payload_mass_g` and a centre-of-mass
proxy, it is the product. The subsample is a stride over the row-major cloud, which is a regular
spatial grid, rather than a sort by curvature: sorting would cluster every evaluated point into one
region and miss the interior of a flat face entirely.

| File | Role |
| --- | --- |
| `synthesis.py` | `synthesize_suction_grasps`, plus `SuctionGrasp` (one ranked candidate) and `SuctionConfig` |
| `scorer.py` | the `SuctionScorer` Protocol, which is the seam, and `AnalyticalSuctionScorer`, its only implementation; emits `SuctionQuality` |
| `seal.py` | `evaluate_seal`, a deformable concentric-ring model: can the cup rim form an airtight seal here; returns `SealResult` |
| `wrench.py` | `evaluate_wrench_resistance`, asking whether vacuum, friction and the elastic moment hold the part; returns `WrenchResult` |

A `SuctionGrasp` carries `position_mm`, a unit `approach`, `seal_score`, `quality` in `[0, 1]`, its
`frame` and metadata.

## Usage

```python
from src.robot.grasping.suction import (
    synthesize_suction_grasps, SuctionConfig, AnalyticalSuctionScorer,
)

grasps = synthesize_suction_grasps(
    segmentation, depth_map, intrinsics,   # intrinsics is the 3x3 camera matrix
    camera_to_base=cam_to_base,            # optional; omit to stay in the camera frame
    payload_mass_g=250.0,                  # enables the wrench term
    config=SuctionConfig(scorer=AnalyticalSuctionScorer(), max_results=5, min_quality=0.2),
)
best = grasps[0]   # ranked by quality, ties broken toward the centre of mass, deterministic
```

`SuctionConfig` defaults: `max_eval_candidates=80`, `max_results=5`, `min_quality=0.2`,
`min_cloud_points=16`. A cloud smaller than `min_cloud_points` returns an empty list.

## Traps

There is no learned scorer, and there will not be one wrapping a published network trained on
non-commercially licensed data. That boundary is enforced in the checks this repository runs, not
just documented. Nothing structural is lost: the seam is the `SuctionScorer` Protocol, `synthesis.py`
depends on nothing else, and a replacement implements `prepare_scene` and `score` and drops in. What
a replacement needs is licence-clean training data, which is what the scene generator under
[`datagen/`](../../../../datagen/README.md) exists to produce.

`prepare_scene` receives depth in metres although `synthesize_suction_grasps` takes it in
millimetres, because metres is the unit every published RGB-D formulation uses. A future scorer that
assumes otherwise scores at a thousand times the intended scale without saying so.

The analytic scorer is a geometric seal argument, not vacuum physics. A flat patch scores 1.0 and a
curved contact or an edge scores near 0, which is a genuine and meaningful spread over rendered depth.
Air leak, material porosity and line pressure are not modelled and stay a real-hardware concern. The
simulated surface gripper is binary and carries no seal model at all.

The cup envelope is not the default collision envelope. `SuctionCupGripperModel` exists in
[`collision/`](../collision/README.md) and `robot.grasping.gripper_geometry.kind: suction` selects
it, but a cell that leaves that block at its default filters suction candidates against jaw fingers.

## See also

- [`../README.md`](../README.md) for the pick pipeline this modality plugs into
- [`../planning/README.md`](../planning/README.md) for `suction_approach`, which builds motion from a `SuctionGrasp`
- [`../geometry/README.md`](../geometry/README.md) for the point cloud and normals synthesis consumes
- [`../collision/README.md`](../collision/README.md) for the cup envelope these candidates should be checked against
- [`../../grippers/README.md`](../../grippers/README.md) for the vacuum drivers, and for the
  `SuctionCupProfile` shapes the simulated cup is built from
