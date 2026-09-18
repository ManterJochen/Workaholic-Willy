# Suction grasps (`src/robot/grasping/suction`)

Turns a perceived object into ranked suction candidates: flat, sealable patches of its surface, each
approached along the local surface normal. It answers where a cup can seal and hold; the attach and
the lift belong to the gripper driver and the pick service.

```python
from willy import synthesize_suction_grasps
from src.robot.grasping.suction import SuctionConfig

grasps = synthesize_suction_grasps(
    mask, depth_mm, K,                   # a boolean mask, depth in millimetres, the 3x3 camera matrix
    camera_to_base=camera_to_base,       # a Transform; leave it out to stay in the camera frame
    payload_mass_g=250.0,                # turns on the wrench term
    config=SuctionConfig(max_results=5, min_quality=0.2),
)
best = grasps[0] if grasps else None     # ranked by quality, ties toward the centre of mass
```

A `SuctionGrasp` carries `position_mm`, a unit `approach`, `seal_score`, `quality` in `[0, 1]`, its
`frame` and metadata. [jaw_or_suction.py](../../../../examples/offline/grasping/jaw_or_suction.py)
compares it with the jaw on three boxes at a desk.

A suction grasp is a second end-effector modality, not a flag on the jaw path: one sealable contact and
a press direction, with no aperture, no antipodal pair and no closing direction. It covers what a
two-finger gripper does worst, wide flat objects and top faces in a bin. The package is pure numpy and
deterministic, and imports no robot and no simulator.

## The model

The synthesis back-projects the mask, estimates normals and curvature, takes a uniform subsample of the
surface as candidate contacts and scores each:

```
quality = seal(flatness, perimeter support, normal alignment) x wrench(gravity and payload about the contact)
```

Without a payload the quality is the seal score alone. The subsample is a stride over the row-major
cloud, a regular grid, rather than a sort by curvature, which would cluster every evaluated point in one
region and miss the interior of a flat face. The cloud is scored in the camera frame, and only the
winners are transformed to the base frame.

`SuctionConfig` defaults: `max_eval_candidates=80`, `max_results=5`, `min_quality=0.2`,
`min_cloud_points=16`. A cloud smaller than `min_cloud_points` returns an empty list.

## Status

| Capability | Evidence |
| --- | --- |
| Suction candidates and the suction pick | measured in simulation |
| A seal on a physical cup | never touched hardware |

The scorer is a geometric seal argument, not vacuum physics. A flat patch scores 1.0 and a curved
contact or an edge scores near 0; air leak, porosity and line pressure are not modelled. The simulated
surface gripper is binary and carries no seal model at all.

## Traps

- The cup is not the default collision envelope. `SuctionCupGripperModel` in
  [collision/](../collision/README.md) is selected by `robot.grasping.gripper_geometry.kind: suction`;
  a cell that leaves that key at `parallel_jaw` filters suction candidates against jaw fingers.
- `prepare_scene` receives depth in metres, although `synthesize_suction_grasps` takes millimetres,
  because metres is the unit the published RGB-D formulations use. A scorer that assumes otherwise
  scores at a thousand times the intended scale without saying so.
- There is no learned scorer, and there will not be one wrapping a network trained on non-commercially
  licensed data; the repository's licence checks enforce that boundary. The seam is the `SuctionScorer`
  Protocol: a replacement implements `prepare_scene` and `score` and drops in. What it needs is
  licence-clean training data, which [datagen/](../../../../datagen/README.md) exists to produce.

## Files

| File | Holds |
| --- | --- |
| `synthesis.py` | `synthesize_suction_grasps`, `SuctionGrasp` (one ranked candidate) and `SuctionConfig` |
| `scorer.py` | the `SuctionScorer` Protocol, and `AnalyticalSuctionScorer`, its one implementation, returning `SuctionQuality` |
| `seal.py` | `evaluate_seal`: a concentric-ring model of whether the cup rim can seal here; returns `SealResult` |
| `wrench.py` | `evaluate_wrench_resistance`: whether vacuum, friction and the elastic moment hold the part; returns `WrenchResult` |

## Details

- [grasping/](../README.md): the pick stack this modality plugs into
- [planning/](../planning/README.md): the `suction_approach` module, which builds the motion from a
  `SuctionGrasp`
- [geometry/](../geometry/README.md): the point cloud and normals the synthesis consumes
- [grippers/](../../grippers/README.md): the vacuum drivers and the `SuctionCupProfile` shapes of the
  simulated cup
- [guide 06](../../../../docs/guide/06-grippers.md): choosing and configuring a gripper
