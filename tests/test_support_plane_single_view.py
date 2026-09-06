"""The single-view support-plane refinement: it runs now, and it stands down when it is blind.

HISTORY, because the assertions only make sense with it. `pick_loop._target_cloud_base_mm` builds
this view's target cloud so `resolve_support_plane` can locate what the object rests on. It reads
`calculator.camera_matrix`, the analytic calculator never exposed that, so the read returned `None`
and the branch never ran on any path -- only the fused-cloud half of the refinement did. Exposed
2026-08-23 on an owner decision to activate and MEASURE rather than delete.

Measuring it immediately produced the failure that shaped the guard: `real_cell --rehearse` went
3/3 -> exit 2, with telemetry reading `target_cloud_z_min_mm: 50.0, target_cloud_z_max_mm: 50.0`.
A flat cloud -- one surface, one height. `resolve_support_plane` takes the cloud's lowest point as
the plane, so the plane landed on top of the object and every candidate below it was rejected as
under-the-table.

That is not a synthetic-input artefact: a top-down camera sees an object's top face and nothing of
its base, which is the geometry the September cell has (two D435s at 70 deg elevation, 20 deg off
nadir). So the branch now requires the cloud to have vertical extent along the support normal
before it may locate the support -- a cloud without it has not observed the support at all.

MEASURED on the 269-scene v1_proof corpus (single view vs a fused 3-view reference), before and
after the guard, as p90 of how far the single-view plane sits ABOVE the fused one:

    family     before    after
    sparse     1.8 mm    0.1 mm
    packed     9.5 mm    0.4 mm
    pile      24.5 mm   21.1 mm
    bin       38.4 mm    8.8 mm
    ALL       17.5 mm    4.9 mm

The branch still fires on 74 % of views (was 99.6 %), so it keeps doing its job -- it just stops
doing it blind. `pile` is the honest remainder: stacked objects hide the floor from a cloud that
does have vertical extent, and no per-view test can fix that. Two cameras can, which is one more
reason the deep-learning arc requires them.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.collision import resolve_support_plane


class _Seg:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask


class _Frame:
    def __init__(self, depth: np.ndarray) -> None:
        self.depth_map = depth


class _Calculator:
    """Only the members `_target_cloud_base_mm` touches."""

    def __init__(self, camera_matrix: "np.ndarray | None") -> None:
        self.camera_matrix = camera_matrix


def _loop_with(support_config, camera_matrix):
    """A `BinPickingOrchestrator` shell carrying just what the cloud builder reads.

    Constructed via `__new__` on purpose: the real constructor wants an arm, a gripper, a
    perception source and a calculator, and none of them participate in the geometry under test.
    """
    from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator

    loop = BinPickingOrchestrator.__new__(BinPickingOrchestrator)
    loop.support_config = support_config
    loop.calculator = _Calculator(camera_matrix)
    return loop


def _depth_and_mask(*, flat: bool, size: int = 60):
    """A masked patch of depth: one constant plane, or a ramp that walks down the object's side."""
    depth = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=bool)
    mask[20:40, 20:40] = True
    if flat:
        depth[mask] = 700.0
    else:
        # 60 mm of depth across the patch -> a cloud that genuinely spans the object's height.
        ramp = np.linspace(700.0, 760.0, 20, dtype=np.float32)
        depth[20:40, 20:40] = ramp[:, None]
    return _Seg(mask), _Frame(depth)


class _Support:
    normal = (0.0, 0.0, 1.0)


_K = np.array([[600.0, 0.0, 30.0], [0.0, 600.0, 30.0], [0.0, 0.0, 1.0]])
#: CAMERA->BASE for a camera looking straight down from 800 mm: base_z = 800 - camera_z.
_CAM_TO_BASE = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 800.0],
    [0.0, 0.0, 0.0, 1.0],
])


class TheBranchRunsOnlyWhenItHasSeenTheSupportTests(unittest.TestCase):
    def test_a_flat_cloud_refuses_to_locate_the_support(self) -> None:
        """THE measured regression: one surface at one height is not an observation of the floor."""
        seg, frame = _depth_and_mask(flat=True)
        loop = _loop_with(_Support(), _K)
        self.assertIsNone(loop._target_cloud_base_mm(seg, frame, _CAM_TO_BASE))

    def test_a_cloud_with_vertical_extent_is_used(self) -> None:
        seg, frame = _depth_and_mask(flat=False)
        loop = _loop_with(_Support(), _K)
        cloud = loop._target_cloud_base_mm(seg, frame, _CAM_TO_BASE)
        self.assertIsNotNone(cloud)
        assert cloud is not None
        span = float(cloud[:, 2].max() - cloud[:, 2].min())
        self.assertGreater(span, 25.0, "the fixture itself must clear the threshold")

    def test_without_intrinsics_it_stands_down_as_it_always_did(self) -> None:
        """An uncalibrated cell keeps the old behaviour -- `camera_matrix` is `None` there."""
        seg, frame = _depth_and_mask(flat=False)
        loop = _loop_with(_Support(), None)
        self.assertIsNone(loop._target_cloud_base_mm(seg, frame, _CAM_TO_BASE))


class TheConsequenceOfAFlatCloudIsThePlaneOnTopOfTheObjectTests(unittest.TestCase):
    """Why the guard exists, stated as arithmetic rather than as a story.

    Feeds `resolve_support_plane` the two clouds directly, so the failure is visible without a
    pick loop: the flat cloud puts the plane at the object's own height, which is what rejected
    3/3 rehearsal picks.
    """

    def test_a_flat_cloud_would_raise_the_plane_to_the_object(self) -> None:
        flat = np.column_stack([
            np.zeros(50), np.zeros(50), np.full(50, 50.0),  # every point at z = 50 mm
        ])
        resolved = resolve_support_plane(
            declared_height_mm=0.0, target_clouds_base_mm=[flat], refine_from_target=True,
        )
        self.assertEqual(resolved.plane.offset_mm, 50.0)
        self.assertEqual(resolved.source, "observed")

    def test_a_cloud_that_reaches_the_floor_leaves_the_plane_where_it_belongs(self) -> None:
        reaching = np.column_stack([
            np.zeros(50), np.zeros(50), np.linspace(0.0, 50.0, 50),
        ])
        resolved = resolve_support_plane(
            declared_height_mm=0.0, target_clouds_base_mm=[reaching], refine_from_target=True,
        )
        self.assertAlmostEqual(resolved.plane.offset_mm, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
