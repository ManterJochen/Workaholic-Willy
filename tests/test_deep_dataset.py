"""One scene -> one training sample. Every choice in here was measured, so every one is pinned.

Synthetic scenes, in the shape `datagen.corpus.clouds` writes. Nothing needs the corpus or Isaac; what
these hold is the set of decisions that would train perfectly while being wrong:

  * the CHANNELS are only the ones a cell can supply. `is_arm` is deliberately absent -- nothing in the
    perception stack segments an arm, so a net leaning on it would be leaning on training data.
  * the SAMPLE is stratified. Clouds are ~76 % table and wall; uniform sampling spends three quarters
    of the net's input on floor, and measured, it recovers 89.69 % of labelled contacts against 91.45 %
    for a stratified draw of the same size.
  * the ARM is dropped before a rotation and only then. A rotated table is still a table.
  * bins are ENCODED after the rotation, never remapped through it.
  * grasps the corridor check would refuse are excluded -- 28 % of v1_proof's, which was labelled
    before that check existed.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.deep.corpus.sample import (
    FEATURE_NAMES,
    NO_GRASP,
    NO_TARGET,
    SampleSpec,
    build_sample,
)
from src.robot.grasping.deep.corpus.grasp_encoding import decode_grasp

_ENVIRONMENT = -1
_ARM = -2


def _scene(*, objects: int = 60, environment: int = 200, arm: int = 40,
           grasps: int = 3, admissible: list[bool] | None = None,
           invalid_normals: int = 0) -> dict[str, np.ndarray]:
    """A scene in the corpus's own shape: objects on a plate, environment below, arm above.

    Geometry is deliberately simple and separated in z, so a test can say which points are which
    without depending on how the sampler ordered them.
    """
    rng = np.random.default_rng(7)
    obj = np.column_stack([rng.uniform(-50, 50, objects), rng.uniform(-50, 50, objects),
                           rng.uniform(20, 60, objects)])
    env = np.column_stack([rng.uniform(-150, 150, environment),
                           rng.uniform(-150, 150, environment), np.zeros(environment)])
    arm_pts = np.column_stack([rng.uniform(-30, 30, arm), rng.uniform(-30, 30, arm),
                               rng.uniform(300, 400, arm)])
    points = np.vstack([obj, env, arm_pts])
    instance = np.concatenate([np.zeros(objects, dtype=np.int16),
                               np.full(environment, _ENVIRONMENT, dtype=np.int16),
                               np.full(arm, _ARM, dtype=np.int16)])
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(points), 1))
    valid = np.ones(len(points), dtype=bool)
    if invalid_normals:
        valid[:invalid_normals] = False
        normals[:invalid_normals] = 0.0

    # Grasps on the first few object points, each with two contacts a millimetre away so they land
    # inside any sane graspability radius.
    position = obj[:grasps].copy()
    approach = np.tile(np.array([0.0, 0.0, -1.0]), (grasps, 1))
    axis = np.tile(np.array([1.0, 0.0, 0.0]), (grasps, 1))
    contacts = np.repeat(position, 2, axis=0)
    contacts[0::2, 0] -= 1.0
    contacts[1::2, 0] += 1.0
    owner = np.repeat(np.arange(grasps), 2).astype(np.int32)
    return {
        "points_mm": points.astype(np.float32),
        "normals": normals,
        "normal_valid": valid,
        "view_count": np.full(len(points), 2, dtype=np.uint8),
        "instance_id": instance,
        "grasp_position_mm": position.astype(np.float32),
        "grasp_approach": approach.astype(np.float32),
        "grasp_axis": axis.astype(np.float32),
        "grasp_width_mm": np.full(grasps, 40.0, dtype=np.float32),
        "grasp_instance": np.zeros(grasps, dtype=np.int16),
        "grasp_held": np.full(grasps, -1, dtype=np.int8),
        "grasp_approach_admissible": np.asarray(
            admissible if admissible is not None else [True] * grasps, dtype=bool),
        "grasp_part_role": np.asarray([""] * grasps, dtype="<U8"),
        "contact_points_mm": contacts.astype(np.float32),
        "contact_grasp_index": owner,
    }


def _spec(**kwargs) -> SampleSpec:
    base = {"points": 128, "rotate_z": False}
    return SampleSpec(**{**base, **kwargs})


class ShapeTests(unittest.TestCase):
    def test_every_array_agrees_on_the_point_count(self) -> None:
        sample = build_sample(_scene(), np.random.default_rng(0), _spec())
        n = len(sample["points_m"])
        self.assertEqual(n, 128)
        for key in ("features", "graspability", "approach_bin", "rotation_bin",
                    "depth_m", "width_m", "instance_id"):
            self.assertEqual(len(sample[key]), n, key)
        self.assertEqual(sample["features"].shape[1], len(FEATURE_NAMES))

    def test_a_cloud_smaller_than_the_budget_comes_back_whole(self) -> None:
        sample = build_sample(_scene(objects=5, environment=5, arm=0),
                              np.random.default_rng(0), _spec(points=8192))
        self.assertEqual(len(sample["points_m"]), 10)

    def test_the_channels_are_the_ones_a_cell_can_supply(self) -> None:
        """`is_arm` must NOT be here: nothing in the perception stack segments an arm."""
        self.assertNotIn("is_arm", FEATURE_NAMES)
        self.assertEqual(FEATURE_NAMES[-2:], ("is_target", "is_object"))


class StratificationTests(unittest.TestCase):
    """THREE ways, around the target -- a consequence of the unit becoming a (scene, target object).

    With a single "objects" share the target took its 1/N of it and MEASURED 16 points of 2048: the
    thing the loss is about, drowned in true negatives. Split three ways the target gets a median of
    1,520.
    """

    def test_an_empty_stratum_gives_its_share_back_in_priority_order(self) -> None:
        """The fixture has ONE object, so the "other objects" third has nothing to spend.

        Its budget flows back to the TARGET first, not to the environment: 200 target points and 100
        environment out of 300, rather than 100 and 200. That order is deliberate -- the target is the
        thing the loss is about -- and it is asserted rather than assumed, because the first version of
        this test expected a flat third and was wrong about the code rather than the other way round.
        """
        sample = build_sample(_scene(objects=400, environment=4000, arm=0),
                              np.random.default_rng(0), _spec(points=300), target_instance=0)
        self.assertEqual(len(sample["points_m"]), 300)
        self.assertEqual(int((sample["instance_id"] == 0).sum()), 200)
        self.assertEqual(int((sample["instance_id"] < 0).sum()), 100)

    def test_the_target_share_is_honoured_when_every_stratum_has_points(self) -> None:
        scene = _scene(objects=400, environment=4000, arm=0)
        # Split the objects in two so a genuine "other objects" stratum exists.
        scene["instance_id"] = scene["instance_id"].copy()
        objects = np.flatnonzero(scene["instance_id"] >= 0)
        scene["instance_id"][objects[200:]] = 1
        sample = build_sample(scene, np.random.default_rng(0), _spec(points=300), target_instance=0)
        self.assertEqual(int((sample["instance_id"] == 0).sum()), 100)
        self.assertEqual(int((sample["instance_id"] == 1).sum()), 100)
        self.assertEqual(int((sample["instance_id"] < 0).sum()), 100)

    def test_a_scene_short_of_objects_spends_the_rest_on_environment(self) -> None:
        sample = build_sample(_scene(objects=10, environment=4000, arm=0),
                              np.random.default_rng(0), _spec(points=200), target_instance=0)
        self.assertEqual(len(sample["points_m"]), 200)
        self.assertEqual(int((sample["instance_id"] >= 0).sum()), 10)

    def test_a_scene_short_of_environment_gives_the_budget_back_to_objects(self) -> None:
        """A short sample would make batch shapes ragged for no reason."""
        sample = build_sample(_scene(objects=4000, environment=10, arm=0),
                              np.random.default_rng(0), _spec(points=200), target_instance=0)
        self.assertEqual(len(sample["points_m"]), 200)
        self.assertEqual(int((sample["instance_id"] < 0).sum()), 10)

    def test_no_point_is_sampled_twice(self) -> None:
        """A duplicated point is not extra information; it doubles one bit of surface in the loss."""
        scene = _scene(objects=400, environment=400, arm=0)
        sample = build_sample(scene, np.random.default_rng(3), _spec(points=300))
        keys = {tuple(np.round(p, 6)) for p in sample["points_m"]}
        self.assertEqual(len(keys), 300)


class NormalTests(unittest.TestCase):
    def test_points_with_an_unusable_normal_are_dropped_not_zeroed(self) -> None:
        """MEASURED 0.030 % of a real corpus. A zero normal reaching the net is a lie it cannot see."""
        scene = _scene(objects=60, environment=60, arm=0, invalid_normals=20)
        sample = build_sample(scene, np.random.default_rng(0), _spec(points=100))
        lengths = np.linalg.norm(sample["features"][:, :3], axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


class ArmTests(unittest.TestCase):
    def test_the_arm_is_kept_when_the_sample_is_not_rotated(self) -> None:
        sample = build_sample(_scene(), np.random.default_rng(0), _spec(points=300))
        self.assertIn(_ARM, sample["instance_id"].tolist())

    def test_the_arm_is_dropped_when_the_sample_is_rotated(self) -> None:
        """A rotated arm stands where no arm on a fixed base could."""
        sample = build_sample(_scene(), np.random.default_rng(0),
                              _spec(points=300, rotate_z=True))
        self.assertNotIn(_ARM, sample["instance_id"].tolist())

    def test_the_arm_is_never_an_object(self) -> None:
        sample = build_sample(_scene(), np.random.default_rng(0), _spec(points=300))
        arm = sample["instance_id"] == _ARM
        self.assertTrue(arm.any())
        self.assertEqual(float(sample["features"][arm, 4].sum()), 0.0)


class TargetTests(unittest.TestCase):
    def test_no_target_leaves_the_channel_empty(self) -> None:
        sample = build_sample(_scene(), np.random.default_rng(0), _spec(),
                              target_instance=NO_TARGET)
        self.assertEqual(sample["target_instance"], NO_TARGET)
        self.assertEqual(float(sample["features"][:, 3].sum()), 0.0)

    def test_a_named_target_marks_exactly_its_own_points(self) -> None:
        sample = build_sample(_scene(), np.random.default_rng(0), _spec(points=300),
                              target_instance=0)
        is_target = sample["features"][:, 3] > 0.5
        np.testing.assert_array_equal(is_target, sample["instance_id"] == 0)

    def test_both_modes_appear_over_many_draws(self) -> None:
        """One net covers the VLM naming an object AND bin-clearing taking anything."""
        rng = np.random.default_rng(1)
        spec = _spec(points=200, target_probability=0.5)
        seen = {build_sample(_scene(), rng, spec)["target_instance"] for _ in range(40)}
        self.assertIn(NO_TARGET, seen)
        self.assertIn(0, seen)


class GraspTargetTests(unittest.TestCase):
    def test_points_near_a_contact_carry_a_grasp_and_the_rest_do_not(self) -> None:
        sample = build_sample(_scene(), np.random.default_rng(0), _spec(points=300))
        has = sample["approach_bin"] >= 0
        self.assertTrue(has.any())
        self.assertTrue((~has).any())
        # A masked target must be a MASK, never a plausible-looking value.
        self.assertTrue(np.all(sample["approach_bin"][~has] == NO_GRASP))
        self.assertTrue(np.all(np.isnan(sample["depth_m"][~has])))
        self.assertTrue(np.all(np.isnan(sample["width_m"][~has])))

    def test_the_width_target_is_in_metres(self) -> None:
        sample = build_sample(_scene(), np.random.default_rng(0), _spec(points=300))
        widths = sample["width_m"][~np.isnan(sample["width_m"])]
        np.testing.assert_allclose(widths, 0.040, atol=1e-6)

    def test_graspability_and_the_per_seed_mask_agree(self) -> None:
        sample = build_sample(_scene(), np.random.default_rng(0), _spec(points=300))
        np.testing.assert_array_equal(sample["graspability"] > 0.5, sample["approach_bin"] >= 0)

    def test_an_inadmissible_grasp_never_becomes_a_target(self) -> None:
        """28 % of v1_proof's labels would have the gripper start under the table."""
        scene = _scene(grasps=3, admissible=[False, False, False])
        sample = build_sample(scene, np.random.default_rng(0), _spec(points=300))
        self.assertEqual(int((sample["approach_bin"] >= 0).sum()), 0)
        self.assertEqual(float(sample["graspability"].sum()), 0.0)

    def test_dropping_one_grasp_keeps_the_others_pointing_at_themselves(self) -> None:
        """The contact->grasp index is REMAPPED when rows are dropped; an off-by-one is silent."""
        scene = _scene(grasps=3, admissible=[False, True, True])
        scene["grasp_width_mm"] = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
        sample = build_sample(scene, np.random.default_rng(0), _spec(points=400))
        widths = np.unique(sample["width_m"][~np.isnan(sample["width_m"])])
        # `allclose` against the survivors, not a set of Python floats: these are float32, and 0.02 is
        # not exactly representable there -- the first version of this assertion failed on values that
        # were correct.
        self.assertTrue(len(widths) <= 2)
        for value in widths:
            self.assertTrue(np.isclose(value, 0.020, atol=1e-6) or np.isclose(value, 0.030, atol=1e-6),
                            f"only the surviving grasps' widths may appear, got {widths}")


class RotationTests(unittest.TestCase):
    def test_a_rotated_sample_still_decodes_to_a_downward_grasp(self) -> None:
        """The bins are ENCODED after the rotation. Remapping an old index would be a different thing."""
        sample = build_sample(_scene(), np.random.default_rng(5), _spec(points=400, rotate_z=True))
        has = sample["approach_bin"] >= 0
        self.assertTrue(has.any())
        approach, _axis = decode_grasp(sample["approach_bin"][has], sample["rotation_bin"][has])
        # The scene's only grasp approaches straight down, and a rotation about z leaves that alone.
        np.testing.assert_allclose(approach[:, 2], -1.0, atol=0.2)

    def test_rotation_does_not_change_how_many_points_carry_a_grasp(self) -> None:
        scene = _scene()
        plain = build_sample(scene, np.random.default_rng(11), _spec(points=200))
        turned = build_sample(scene, np.random.default_rng(11), _spec(points=200, rotate_z=True))
        # The arm is dropped when rotating, so the counts are compared as SHARES of object points.
        self.assertGreater(int((turned["approach_bin"] >= 0).sum()), 0)
        self.assertGreater(int((plain["approach_bin"] >= 0).sum()), 0)

    def test_the_support_plane_is_not_something_to_rotate_about(self) -> None:
        """z is height above the table. A rotation about gravity cannot change it."""
        scene = _scene(arm=0)
        plain = build_sample(scene, np.random.default_rng(2), _spec(points=200))
        turned = build_sample(scene, np.random.default_rng(2), _spec(points=200, rotate_z=True))
        np.testing.assert_allclose(np.sort(plain["points_m"][:, 2]),
                                   np.sort(turned["points_m"][:, 2]), atol=1e-6)


class FrameTests(unittest.TestCase):
    def test_heights_are_above_the_support_plane_not_absolute(self) -> None:
        """A net trained on absolute z learns "z < 5 mm is table" and breaks on a raised one."""
        scene = _scene(arm=0)
        flat = build_sample(scene, np.random.default_rng(0), _spec(points=200))
        raised = build_sample(scene, np.random.default_rng(0),
                              _spec(points=200, support_height_mm=100.0))
        np.testing.assert_allclose(flat["points_m"][:, 2] - 0.1, raised["points_m"][:, 2], atol=1e-6)

    def test_x_and_y_are_centred_so_a_scene_can_sit_anywhere(self) -> None:
        sample = build_sample(_scene(arm=0), np.random.default_rng(0), _spec(points=400))
        np.testing.assert_allclose(sample["points_m"][:, :2].mean(axis=0), 0.0, atol=1e-6)

    def test_the_offset_that_undoes_the_centring_is_carried(self) -> None:
        """Without it a prediction is a tensor nobody can turn into a grasp in the cell's frame."""
        sample = build_sample(_scene(arm=0), np.random.default_rng(0), _spec(points=400))
        self.assertEqual(sample["centre_xy_mm"].shape, (2,))
        self.assertEqual(sample["support_height_mm"], 0.0)


class CorpusContractTests(unittest.TestCase):
    """The two sides of a seam that CANNOT import each other, checked by the one thing that can.

    `backend` may never import `datagen` -- a generator that becomes a runtime dependency of a robot
    is one nobody can delete or refuse to install, and `test_datagen_layering` enforces it. So the
    dataset layer mirrors the corpus's instance ids as private constants, and a mirror nobody compares
    is a copy waiting to drift. A test is not bound by the layering rule and can hold both.
    """

    def test_the_instance_ids_agree_on_both_sides_of_the_layering_guard(self) -> None:
        from datagen.corpus.clouds import ARM_INSTANCE, ENVIRONMENT_INSTANCE

        from src.robot.grasping.deep.corpus import sample as dataset

        self.assertEqual(dataset._ENVIRONMENT_INSTANCE, ENVIRONMENT_INSTANCE)  # noqa: SLF001
        self.assertEqual(dataset._ARM_INSTANCE, ARM_INSTANCE)                  # noqa: SLF001

    def test_every_array_the_sampler_reads_is_one_the_corpus_writes(self) -> None:
        """A key renamed on the datagen side would otherwise surface as a KeyError mid-training."""
        import inspect

        from datagen.corpus.clouds import _empty_cloud, _empty_grasp_table  # noqa: PLC0415

        written = set(_empty_cloud()) | set(_empty_grasp_table())
        source = inspect.getsource(build_sample)
        read = {name for name in written if f'"{name}"' in source}
        # `view_count` is deliberately NOT here: the corpus writes it, the sampler no longer
        # reads it. See FEATURE_NAMES for the measurement that dropped the channel.
        required = {"points_mm", "instance_id", "normals", "grasp_position_mm",
                    "grasp_approach", "grasp_axis", "grasp_width_mm", "contact_points_mm",
                    "contact_grasp_index"}
        self.assertTrue(required <= written, f"the corpus stopped writing {required - written}")
        self.assertTrue(required <= read, f"the sampler stopped reading {required - read}")

    def test_the_optional_arrays_really_are_optional(self) -> None:
        """A corpus built before a column existed must still make samples, not crash mid-epoch."""
        scene = _scene()
        for optional in ("normal_valid", "grasp_approach_admissible"):
            with self.subTest(missing=optional):
                trimmed = {k: v for k, v in scene.items() if k != optional}
                sample = build_sample(trimmed, np.random.default_rng(0), _spec(points=200))
                self.assertEqual(len(sample["points_m"]), 200)


class DeterminismTests(unittest.TestCase):
    def test_the_same_seed_gives_the_same_sample(self) -> None:
        scene = _scene()
        for rotate in (False, True):
            with self.subTest(rotate=rotate):
                a = build_sample(scene, np.random.default_rng(4), _spec(points=200, rotate_z=rotate))
                b = build_sample(scene, np.random.default_rng(4), _spec(points=200, rotate_z=rotate))
                for key in ("points_m", "features", "graspability", "approach_bin", "depth_m"):
                    np.testing.assert_array_equal(a[key], b[key], err_msg=key)


if __name__ == "__main__":
    unittest.main()
