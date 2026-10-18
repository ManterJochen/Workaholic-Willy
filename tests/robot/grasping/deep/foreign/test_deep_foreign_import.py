"""Turning a public corpus into scene files our own pipeline reads.

⭐ THE INTEGRATION CLAIM THESE TESTS DEFEND. The import writes `.npz` scenes in the format `datagen`
writes, so the corpus index, the folds, `build_sample`, every probe and every metric run on imported
data with no flag and no change. That claim is only worth anything if the written arrays really are
what our loader expects, so the last test here reads one back through `load_scene` and `build_sample`
and puts it through the contract validator.

Everything runs OFFLINE. The network path is exercised by hand against the real archive; what is
pinned here is the arithmetic, because that is where a silent wrongness would live.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.deep.foreign.grasp_anything import (
    DEFAULT_GRIPPER, SOURCE, _APPROACH_COLUMN, _AXIS_COLUMN, _contact_and_width, estimate_normals,
    fit_centre_offset, scene_arrays)


def _slab(rng: np.random.Generator, count: int = 4000) -> np.ndarray:
    """A box seen from a camera at the origin looking down +z, so only the near faces are sampled.

    2.5D ON PURPOSE. Every real single-view corpus is, and the two places this module has to be
    careful about it are the normal orientation and the fact that only one jaw contact is observable.
    """
    half = np.array([0.04, 0.03, 0.05])
    centre = np.array([0.0, 0.0, 0.5])
    points = rng.uniform(-1.0, 1.0, size=(count, 3)) * half
    face = rng.integers(0, 3, size=count)
    points[np.arange(count), face] = rng.choice([-1.0, 1.0], size=count) * half[face]
    points += centre
    # Keep the faces a camera at the origin could see: nearer than the box centre in z, or on a side.
    return points[(points[:, 2] <= centre[2]) | (np.abs(points[:, 0]) > half[0] * 0.9)]


def _pose(approach: np.ndarray, axis: np.ndarray, centre: np.ndarray,
          offset_m: float) -> np.ndarray:
    """A 4x4 in the source's convention: approach in column 2, closing axis in column 0."""
    columns = {_APPROACH_COLUMN: approach, _AXIS_COLUMN: axis}
    columns[({0, 1, 2} - {_APPROACH_COLUMN, _AXIS_COLUMN}).pop()] = np.cross(approach, axis)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([columns[0], columns[1], columns[2]], axis=1)
    pose[:3, 3] = centre - offset_m * approach
    return pose


class NormalTests(unittest.TestCase):
    """The source ships none, and our contract needs unit normals whose SIGN means something."""

    def test_normals_are_unit_and_mostly_valid_on_a_surface(self) -> None:
        cloud = _slab(np.random.default_rng(0))
        normals, valid = estimate_normals(cloud)
        self.assertEqual(normals.shape, (len(cloud), 3))
        self.assertGreater(float(valid.mean()), 0.8)
        lengths = np.linalg.norm(normals[valid], axis=-1)
        self.assertTrue(np.allclose(lengths, 1.0, atol=1e-4))

    def test_normals_point_TOWARD_the_camera(self) -> None:
        """⛔ ORIENTATION IS NOT A DETAIL. A plane's normal is defined up to sign and our contract's
        normal is sign-meaningful: the control target is `approach = -normal`, so a flipped normal
        teaches a head to approach from inside the object."""
        cloud = _slab(np.random.default_rng(1))
        normals, valid = estimate_normals(cloud)
        view = cloud / np.linalg.norm(cloud, axis=-1, keepdims=True)
        along = np.einsum("ij,ij->i", normals[valid], view[valid])
        self.assertLessEqual(float(along.max()), 1e-6,
                             "a normal pointing away from the camera survived orientation")

    def test_a_cloud_too_small_to_fit_a_patch_returns_nothing_valid(self) -> None:
        normals, valid = estimate_normals(np.zeros((5, 3)))
        self.assertEqual(normals.shape, (5, 3))
        self.assertFalse(bool(valid.any()))


class OffsetTests(unittest.TestCase):
    """The source stores the gripper MOUNTING FACE. Getting this wrong misses by about 17 cm."""

    def test_a_hidden_offset_is_recovered(self) -> None:
        rng = np.random.default_rng(2)
        cloud = _slab(rng)
        approach = np.array([0.0, 0.0, 1.0])
        axis = np.array([1.0, 0.0, 0.0])
        hidden = 0.173
        centres = cloud[rng.permutation(len(cloud))[:60]]
        poses = np.stack([_pose(approach, axis, c, hidden) for c in centres])
        found, residual = fit_centre_offset(cloud, poses)
        self.assertAlmostEqual(found, hidden, delta=0.006)
        self.assertLess(residual, 5.0)

    def test_an_empty_pose_table_refuses(self) -> None:
        with self.assertRaises(ValueError):
            fit_centre_offset(_slab(np.random.default_rng(3)), np.zeros((0, 4, 4)))


class WidthTests(unittest.TestCase):
    """⛔⛔ The stored width is a COMMANDED OPENING and our contract means a contact separation."""

    def test_the_width_is_derived_from_the_cloud_not_from_the_opening(self) -> None:
        """MEASURED on the source: the object fills about 0.658 of the stored half opening, so a
        point at `centre +- w/2 * axis` is in free air. Handed a wildly generous opening, the derived
        width must still come out near the object's real extent."""
        rng = np.random.default_rng(4)
        cloud_mm = _slab(rng) * 1000.0
        centre = np.array([[0.0, 0.0, 480.0]])
        axis = np.array([[1.0, 0.0, 0.0]])
        fitted, contact, width, usable = _contact_and_width(cloud_mm, centre, axis,
                                                            np.array([200.0]))
        self.assertTrue(bool(usable[0]))
        # The slab is 80 mm across in x, so the separation is about that and NOT the 200 offered.
        self.assertAlmostEqual(float(width[0]), 80.0, delta=12.0)
        self.assertLess(abs(float(np.abs(contact[0][0]) - 40.0)), 8.0)
        # ⭐ AND THE CENTRE IS CORRECTED to the midpoint of the two extremes, which for a slab
        # centred on x = 0 means x = 0 whatever the annotated point was.
        self.assertLess(abs(float(fitted[0][0])), 8.0)

    def test_an_annotated_point_off_centre_still_yields_the_object_s_width(self) -> None:
        """⛔⛔ THE DEFECT THIS TEST FOUND. The source's annotated point is a CAMERA-FACING SURFACE
        point, not the midpoint between the jaws, so measuring the extent as twice its distance to
        the furthest observed point gave 3.8 to 7.6 mm on an 80 mm slab whenever it landed near a
        face. A 55 mm hand then ACCEPTED grasps it could not close around. A wrong width is worse
        than a refused grasp, because the width is a trained target."""
        cloud_mm = _slab(np.random.default_rng(8)) * 1000.0
        # An annotated point hard against the +x face, the worst case for the old arithmetic.
        annotated = np.array([[39.0, 0.0, 480.0]])
        _, _, width, usable = _contact_and_width(cloud_mm, annotated,
                                                 np.array([[1.0, 0.0, 0.0]]), np.array([200.0]))
        self.assertTrue(bool(usable[0]))
        self.assertGreater(float(width[0]), 60.0,
                           f"an off-centre annotation gave a width of {float(width[0]):.1f} mm")

    def test_a_separation_too_small_to_believe_is_refused(self) -> None:
        """On a 2.5D cloud a tiny extent means one face was seen, not that the object is 4 mm."""
        cloud_mm = np.array([[0.0, 0.0, 500.0], [2.0, 0.0, 500.0], [4.0, 0.0, 500.0]])
        _, _, width, usable = _contact_and_width(cloud_mm, np.array([[2.0, 0.0, 500.0]]),
                                                 np.array([[1.0, 0.0, 0.0]]), np.array([80.0]))
        self.assertFalse(bool(usable[0]))
        self.assertEqual(float(width[0]), 0.0)

    def test_a_grasp_enclosing_nothing_is_refused_rather_than_given_a_width(self) -> None:
        cloud_mm = _slab(np.random.default_rng(5)) * 1000.0
        far = np.array([[300.0, 300.0, 500.0]])
        _, _, width, usable = _contact_and_width(cloud_mm, far, np.array([[1.0, 0.0, 0.0]]),
                                                 np.array([80.0]))
        self.assertFalse(bool(usable[0]))
        self.assertEqual(float(width[0]), 0.0)


class SceneTests(unittest.TestCase):

    def _scene(self, gripper: str = DEFAULT_GRIPPER, opening_mm: float = 120.0):
        rng = np.random.default_rng(6)
        cloud_m = _slab(rng)
        cloud = np.concatenate([cloud_m, rng.uniform(0.0, 1.0, size=(len(cloud_m), 3))], axis=1)
        approach = np.array([0.0, 0.0, 1.0])
        axis = np.array([1.0, 0.0, 0.0])
        centres = cloud_m[rng.permutation(len(cloud_m))[:40]]
        poses = np.stack([_pose(approach, axis, c, 0.173) for c in centres])
        widths = np.full(len(poses), opening_mm / 1000.0)
        return scene_arrays("abc123", cloud, [(poses, widths)],
                            centre_offset_m=0.173, gripper=gripper)

    def test_the_gripper_is_stamped(self) -> None:
        """⛔ WITHOUT IT THE CORPUS INDEX FALLS BACK TO `2f85`, so labels made for a 140 mm hand would
        train a model that believes an 85 mm hand made them. That is defect 4 of the architecture
        plan, and importing would have reintroduced it from the outside."""
        arrays, _ = self._scene()
        assert arrays is not None
        self.assertEqual(str(arrays["gripper"][0]), DEFAULT_GRIPPER)

    def test_a_separation_wider_than_the_aperture_is_refused_and_counted(self) -> None:
        """A derived separation of 190 mm is a statement about the OBJECT. A hand that opens 140
        cannot grasp it, so it is not a label. The count is returned rather than swallowed."""
        arrays, refused = self._scene(gripper="narrow_55")
        # The slab is 80 mm across, so a 55 mm hand must refuse every one of these.
        self.assertGreater(refused, 0)
        self.assertIsNone(arrays)

    def test_every_grasp_carries_one_observable_contact(self) -> None:
        """⚠ ONE, not two. The cloud is 2.5D, so the far contact has no points and is inferred by
        reflecting the near one through the centre. The pairing then works exactly as it does on our
        own corpus, which is why no loop had to change."""
        arrays, _ = self._scene()
        assert arrays is not None
        self.assertEqual(len(arrays["contact_points_mm"]), len(arrays["grasp_position_mm"]))
        self.assertEqual(sorted(arrays["contact_grasp_index"].tolist()),
                         list(range(len(arrays["grasp_position_mm"]))))

    def test_no_physics_verdict_is_claimed(self) -> None:
        """⚠ The scorer's target is the physics referee, so an imported corpus cannot train one until
        a shake has run over it. `-1` is the format's "not measured" and inventing a 1 here would
        make a scorer trained on it look successful for the wrong reason."""
        arrays, _ = self._scene()
        assert arrays is not None
        self.assertTrue(bool((arrays["grasp_held"] == -1).all()))

    def test_one_asset_group_per_scene(self) -> None:
        """⛔ There is no object identity in this source, so asset-disjoint folds are unavailable and
        scene-disjoint is the honest maximum. It is also what the source demands: grasps arrive in
        blocks of twenty jittered variants of one 2D rectangle."""
        arrays, _ = self._scene()
        assert arrays is not None
        self.assertEqual(len(set(arrays["grasp_asset_id"].tolist())), 1)
        self.assertIn("abc123", str(arrays["object_asset_id"][0]))
        self.assertIn(SOURCE.key, str(arrays["object_asset_id"][0]))

    def test_the_source_is_stamped_so_a_mixed_corpus_says_so(self) -> None:
        arrays, _ = self._scene()
        assert arrays is not None
        self.assertEqual(str(arrays["source_dataset"][0]), SOURCE.key)
        self.assertEqual(str(arrays["engine"][0]), "imported")


class RoundTripTests(unittest.TestCase):
    """⭐ THE CLAIM THAT MATTERS: our own loader reads it and the contract validator passes it."""

    def test_an_imported_scene_survives_our_loader_and_the_contract(self) -> None:
        import tempfile
        from pathlib import Path

        from src.robot.grasping.deep.corpus.sample import SampleSpec, build_sample, load_scene
        from src.robot.grasping.deep.foreign.contract import validate_sample

        arrays, _ = SceneTests()._scene()
        assert arrays is not None
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "abc123.npz"
            np.savez_compressed(path, **arrays)
            scene = load_scene(path)
            sample = build_sample(scene, np.random.default_rng(7),
                                  SampleSpec(grasp_set=True, points=1024, rotate_z=False),
                                  target_instance=None)
        fatal = [p for p in validate_sample(sample) if p.fatal]
        self.assertEqual(fatal, [], f"an imported scene breaks our own contract: {fatal}")


if __name__ == "__main__":
    unittest.main()
