"""The ladder's floors — the scale every top-1 number in this project has been missing.

`sfe_fused` at 55.9 % top-1 has always been read against zero, and zero is the wrong reference: a
top-down proposal on a box seen from above scores well by construction, because the analytic reference
verdict admits a band around vertical. These three exist so a learned generator's number arrives with
a scale attached, and what they pin here is that each one is the thing it claims to be:

  * `random` stays inside the SAME admissible cone the learned encoding uses. A floor allowed to
    propose grasps the generator structurally cannot is a lower, flattering floor.
  * `topdown` is exactly vertical, which is the floor a "learned" claim has to clear.
  * `normal` approaches INTO the surface. Its first version approached straight UP from underneath --
    the eigenvector local PCA returns has no sign, and nothing had oriented it. That is the fourth
    frame/sign defect this repo has paid for, and it is why the orientation gets its own test.

None of them ranks: every candidate scores 0.0. A floor that ordered its own proposals would be a
generator, and top-1 would be grading that order instead of the scale it was added to establish.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.deep.corpus.grasp_encoding import APPROACH_MAX_TILT_DEG
from src.robot.grasping.types.grasp_point import GraspFrame
from datagen.eval.floors import (
    NormalGraspCalculator,
    RandomGraspCalculator,
    TopDownGraspCalculator,
    unproject_target,
)

_K = ((400.0, 0.0, 32.0), (0.0, 400.0, 24.0), (0.0, 0.0, 1.0))
_FLOORS = (RandomGraspCalculator, TopDownGraspCalculator, NormalGraspCalculator)


def _frame(size: int = 48) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A nearer square in the middle, and a camera ABOVE the table looking DOWN at it.

    The camera pose is not decoration. A first version of this fixture put the camera at base z = 0
    under a surface at z = 600, and in that world "into the surface" really is upward -- it reported
    the `normal` floor as broken when the floor was right and the fixture was not.
    """
    depth = np.full((size, size), 700.0)
    depth[18:30, 18:30] = 600.0
    mask = np.zeros((size, size), dtype=bool)
    mask[18:30, 18:30] = True
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = [400.0, 0.0, 1000.0]
    return depth, mask, transform


def _tilt_deg(approach: np.ndarray) -> np.ndarray:
    """Angle between each approach and straight down in BASE."""
    return np.degrees(np.arccos(np.clip(-approach[:, 2], -1.0, 1.0)))


class SharedContractTests(unittest.TestCase):
    def test_every_floor_emits_well_formed_BASE_grasps(self) -> None:
        depth, mask, transform = _frame()
        for floor in _FLOORS:
            with self.subTest(floor.__name__):
                grasps = floor(camera_matrix=_K, seed=1, candidates=8).compute(mask, depth, transform)
                self.assertTrue(grasps)
                approach = np.array([g.approach for g in grasps])
                axis = np.array([g.axis for g in grasps])
                np.testing.assert_allclose(np.linalg.norm(approach, axis=1), 1.0, atol=1e-9)
                np.testing.assert_allclose(np.linalg.norm(axis, axis=1), 1.0, atol=1e-9)
                # A jaw axis not perpendicular to its approach is not a jaw pose.
                self.assertLess(np.abs((axis * approach).sum(axis=1)).max(), 1e-9)
                for grasp in grasps:
                    self.assertEqual(grasp.frame, GraspFrame.BASE)

    def test_no_floor_RANKS_its_own_proposals(self) -> None:
        depth, mask, transform = _frame()
        for floor in _FLOORS:
            with self.subTest(floor.__name__):
                grasps = floor(camera_matrix=_K, seed=1, candidates=8).compute(mask, depth, transform)
                self.assertEqual({g.score for g in grasps}, {0.0})
                self.assertTrue(all(g.metadata["is_floor"] for g in grasps))

    def test_every_floor_is_deterministic_given_its_seed(self) -> None:
        """Two runs that differ are two experiments, and a floor that moves is not a scale."""
        depth, mask, transform = _frame()
        for floor in _FLOORS:
            with self.subTest(floor.__name__):
                first = floor(camera_matrix=_K, seed=4, candidates=8).compute(mask, depth, transform)
                second = floor(camera_matrix=_K, seed=4, candidates=8).compute(mask, depth, transform)
                np.testing.assert_array_equal(np.array([g.position for g in first]),
                                              np.array([g.position for g in second]))
                np.testing.assert_array_equal(np.array([g.approach for g in first]),
                                              np.array([g.approach for g in second]))

    def test_every_seed_lies_on_the_TARGET(self) -> None:
        """The mask is the only thing separating the object from the table it stands on."""
        depth, mask, transform = _frame()
        for floor in _FLOORS:
            with self.subTest(floor.__name__):
                grasps = floor(camera_matrix=_K, seed=1, candidates=8).compute(mask, depth, transform)
                # depth 600 under a camera at z=1000 puts the target at base z=400; the table is 300.
                for grasp in grasps:
                    self.assertAlmostEqual(float(grasp.position[2]), 400.0, places=3)

    def test_an_empty_mask_and_a_missing_camera_both_yield_nothing(self) -> None:
        depth, mask, transform = _frame()
        for floor in _FLOORS:
            with self.subTest(floor.__name__):
                self.assertEqual(
                    floor(camera_matrix=_K).compute(np.zeros_like(mask), depth, transform), [])
                self.assertEqual(floor(camera_matrix=None).compute(mask, depth, transform), [])
                self.assertEqual(floor(camera_matrix=_K).compute(mask, depth, None), [])

    def test_an_unknown_constructor_argument_is_ignored(self) -> None:
        """The ladder builds one kwargs dict per rung; a floor must not track the analytic surface."""
        for floor in _FLOORS:
            floor(camera_matrix=_K, support_footprint_geometry=True, gripper_model=object())


class CallingConventionTests(unittest.TestCase):
    """THE DEFECT THAT UNIT TESTS COULD NOT SEE, because they all passed a bare ndarray.

    The ladder and the pick loop both pass an OBJECT carrying `.mask`. Run through the real ladder,
    the first version of these floors returned ZERO candidates on every object -- 26 `no_candidate`
    rows per floor -- while every test here stayed green: `np.asarray(obj)` is a 0-d object array, its
    shape never matches the depth map, and the shape guard refused in silence.
    """

    def test_a_segmentation_OBJECT_works_exactly_like_the_array(self) -> None:
        depth, mask, transform = _frame()

        class Segmentation:
            def __init__(self, array): self.mask = array

        for floor in _FLOORS:
            with self.subTest(floor.__name__):
                built = floor(camera_matrix=_K, seed=3, candidates=8)
                from_array = built.compute(mask, depth, transform)
                from_object = built.compute(Segmentation(mask), depth, transform)
                self.assertTrue(from_object, "a SegmentationLike must not silently yield nothing")
                np.testing.assert_array_equal(
                    np.array([g.position for g in from_array]),
                    np.array([g.position for g in from_object]))

    def test_the_mask_helper_accepts_both_and_nothing_else_silently(self) -> None:
        from datagen.eval.floors import target_mask

        array = np.array([[True, False], [False, True]])

        class Segmentation:
            mask = array

        np.testing.assert_array_equal(target_mask(array), array)
        np.testing.assert_array_equal(target_mask(Segmentation()), array)


class HeightTests(unittest.TestCase):
    """Each floor is a different height, and confusing them would misread every comparison."""

    def test_random_stays_inside_the_GENERATORS_admissible_cone(self) -> None:
        depth, mask, transform = _frame()
        grasps = RandomGraspCalculator(camera_matrix=_K, seed=2, candidates=64).compute(
            mask, depth, transform)
        tilt = _tilt_deg(np.array([g.approach for g in grasps]))
        self.assertLessEqual(tilt.max(), APPROACH_MAX_TILT_DEG + 1e-6)

    def test_random_is_AREA_uniform_not_angle_uniform(self) -> None:
        """Uniform in the polar ANGLE crowds proposals around vertical and flatters the floor. Area-
        uniform over a 135 deg cap puts the mean near 82 deg, which is what makes it a weak floor --
        stated openly in the module docstring rather than hidden in a favourable number."""
        depth, mask, transform = _frame()
        grasps = RandomGraspCalculator(camera_matrix=_K, seed=5, candidates=400).compute(
            mask, depth, transform)
        mean = float(_tilt_deg(np.array([g.approach for g in grasps])).mean())
        self.assertGreater(mean, 70.0)
        self.assertLess(mean, 95.0)

    def test_topdown_is_EXACTLY_vertical(self) -> None:
        depth, mask, transform = _frame()
        grasps = TopDownGraspCalculator(camera_matrix=_K, seed=1, candidates=8).compute(
            mask, depth, transform)
        np.testing.assert_allclose(_tilt_deg(np.array([g.approach for g in grasps])), 0.0, atol=1e-9)

    def test_topdown_still_VARIES_its_in_plane_rotation(self) -> None:
        """Fixed approach and fixed rotation would make every proposal on a scene identical, and the
        floor would measure one pose rather than a family."""
        depth, mask, transform = _frame()
        grasps = TopDownGraspCalculator(camera_matrix=_K, seed=1, candidates=8).compute(
            mask, depth, transform)
        axes = np.array([g.axis for g in grasps])
        self.assertGreater(float(np.abs(axes - axes[0]).max()), 0.1)

    def test_normal_approaches_INTO_the_surface_and_not_up_from_underneath(self) -> None:
        """THE DEFECT THIS FILE EXISTS FOR. Local PCA returns an eigenvector, and an eigenvector has
        no sign; `estimate_surface_normals` does not orient one. Unoriented, this floor approached
        straight UP through the table -- every proposal starting inside the object."""
        depth, mask, transform = _frame()
        grasps = NormalGraspCalculator(camera_matrix=_K, seed=1, candidates=8).compute(
            mask, depth, transform)
        approach = np.array([g.approach for g in grasps])
        # A horizontal surface viewed from above: approaching into it is straight down, full stop.
        np.testing.assert_allclose(_tilt_deg(approach), 0.0, atol=1e-6)
        # And the general statement the fixture is a special case of: it points AWAY from the camera.
        camera = transform[:3, 3]
        toward_camera = camera[None, :] - np.array([g.position for g in grasps])
        self.assertTrue(bool(((approach * toward_camera).sum(axis=1) < 0.0).all()))


class UnprojectionTests(unittest.TestCase):
    def test_it_recovers_the_geometry_the_depth_map_encodes(self) -> None:
        depth, mask, transform = _frame()
        points = unproject_target(mask, depth, _K, transform)
        self.assertEqual(len(points), int(mask.sum()))
        np.testing.assert_allclose(points[:, 2], 400.0, atol=1e-9)

    def test_a_mismatched_mask_yields_nothing_rather_than_broadcasting(self) -> None:
        depth, _mask, transform = _frame()
        self.assertEqual(len(unproject_target(np.zeros((4, 4), dtype=bool), depth, _K, transform)), 0)

    def test_non_finite_and_zero_depth_are_dropped(self) -> None:
        depth, mask, transform = _frame()
        depth = depth.copy()
        depth[18:22, 18:30] = np.nan
        depth[22:24, 18:30] = 0.0
        points = unproject_target(mask, depth, _K, transform)
        self.assertEqual(len(points), int(mask.sum()) - 6 * 12)
        self.assertTrue(bool(np.isfinite(points).all()))


if __name__ == "__main__":
    unittest.main()
