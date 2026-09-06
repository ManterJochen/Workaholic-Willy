"""Recovering a pose convention we chose on purpose, so the answer is known before the test runs.

⛔ WHY THIS IS THE LOAD-BEARING TEST OF THE IMPORTER. A public dataset publishes 4x4 matrices and
says nothing about which column the gripper travels along, which column the jaws close along, whether
the translation is the grasp centre or a palm, or whether the numbers are metres. Importing under a
wrong guess produces a corpus that is geometrically self-consistent and completely wrong, and a
network trains on it to a confident wrong answer without complaining. Four separate frame defects
have shipped in this repository, and not one announced itself.

So the prover is only worth having if it can recover a convention that was deliberately hidden from
it. These fixtures hide one, and the test asserts it comes back.

⚠ TWO EARLIER FIXTURES WERE THE THING AT FAULT, and both failures are preserved below as tests of
their own, because each names a real limit of the method:

    a box floating in free space          the approach sign is genuinely unidentifiable
    grasps whose palm is inside a wall    no real labeller emits these, and with them even the
                                          TRUE convention fails its own clearance claim

Both now produce refusals rather than answers, which is the correct behaviour: not knowing is a state
the importer can act on, and a wrong guess is not.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.deep.foreign.convention import (
    Convention, agree, prove_convention)

_FINGER_MM, _PALM_MM = 50.0, 30.0

#: Half extents and table-standing centres. Three shapes, because one symmetric object admits several
#: readings and the fixture would then be measuring its own symmetry.
_OBJECTS = (
    (np.array([30.0, 20.0, 50.0]), np.array([0.0, 0.0, 50.0])),
    (np.array([15.0, 15.0, 90.0]), np.array([140.0, -40.0, 90.0])),
    (np.array([55.0, 25.0, 22.0]), np.array([-120.0, 60.0, 22.0])),
)


def _box(rng: np.random.Generator, count: int, half: np.ndarray, centre: np.ndarray) -> np.ndarray:
    points = rng.uniform(-1.0, 1.0, size=(count, 3)) * half
    face = rng.integers(0, 3, size=count)
    points[np.arange(count), face] = rng.choice([-1.0, 1.0], size=count) * half[face]
    return points + centre


def _scene(rng: np.random.Generator, count: int = 500, *, table: bool = True) -> np.ndarray:
    parts = [_box(rng, count, half, centre) for half, centre in _OBJECTS]
    if table:
        parts.append(np.concatenate([rng.uniform(-260.0, 260.0, size=(count * 2, 2)),
                                     np.zeros((count * 2, 1))], axis=1))
    return np.concatenate(parts, axis=0)


def _grasps(rng: np.random.Generator, cloud: np.ndarray, *, reachable: bool = True,
            tries: int = 150) -> tuple[np.ndarray, ...]:
    """Antipodal grasps over varied orientations.

    `reachable` keeps only those whose palm has somewhere to be, which is what our own physics
    labeller does and what any published dataset must also have done. Turning it off is how the test
    below shows that an unfiltered fixture cannot judge the clearance claim for ANY convention.
    """
    centres, approaches, axes, widths = [], [], [], []
    for half, base in _OBJECTS:
        for _ in range(tries):
            axis_index, approach_index = rng.permutation(3)[:2]
            axis = np.zeros(3)
            axis[axis_index] = 1.0
            approach = np.zeros(3)
            approach[approach_index] = rng.choice([-1.0, 1.0])
            centre = base + (rng.uniform(-0.75, 0.75, size=3) * half) * (1.0 - axis)
            width = 2.0 * half[axis_index]
            if width > 130.0:
                continue
            if reachable:
                palm = centre - (_FINGER_MM + _PALM_MM * 0.5) * approach
                if float(np.linalg.norm(cloud - palm, axis=1).min()) < max(width * 0.5, 20.0):
                    continue
            centres.append(centre)
            approaches.append(approach)
            axes.append(axis)
            widths.append(width)
    return (np.array(centres), np.array(approaches), np.array(axes), np.array(widths))


def _encode(centre_mm, approach, axis, width_mm, *, approach_col, approach_sign, axis_col,
            unit, offset_mm):
    """Write the grasps out the way some publisher might have, and forget how."""
    columns = {approach_col: approach_sign * approach, axis_col: axis}
    columns[({0, 1, 2} - {approach_col, axis_col}).pop()] = np.cross(approach, axis)
    poses = np.zeros((len(centre_mm), 4, 4))
    poses[:, :3, :3] = np.stack([columns[0], columns[1], columns[2]], axis=2)
    poses[:, :3, 3] = (centre_mm - offset_mm * approach) * (1.0 if unit == "mm" else 0.001)
    poses[:, 3, 3] = 1.0
    return poses, width_mm * (1.0 if unit == "mm" else 0.001)


def _prove(hidden: dict, *, scenes: int = 2, table: bool = True, reachable: bool = True):
    proofs = []
    for scene in range(scenes):
        rng = np.random.default_rng(scene)
        cloud = _scene(rng, table=table)
        centres, approaches, axes, widths = _grasps(rng, cloud, reachable=reachable)
        poses, published = _encode(centres, approaches, axes, widths, **hidden)
        proofs.append(prove_convention(cloud / 1000.0, poses, published,
                                       sample_grasps=120, seed=scene))
    return proofs


class RecoveryTests(unittest.TestCase):
    """⭐ The whole point: a convention deliberately hidden must come back."""

    HIDDEN = (
        ("the translation is the centre, in metres, approach in column 2",
         dict(approach_col=2, approach_sign=1, axis_col=0, unit="m", offset_mm=0.0)),
        ("the translation is a palm 40 mm back, in millimetres, approach NEGATED in column 1",
         dict(approach_col=1, approach_sign=-1, axis_col=2, unit="mm", offset_mm=40.0)),
        ("the translation is the centre, in millimetres, approach in column 0",
         dict(approach_col=0, approach_sign=1, axis_col=1, unit="mm", offset_mm=0.0)),
    )

    def test_every_hidden_convention_is_recovered(self) -> None:
        for label, hidden in self.HIDDEN:
            with self.subTest(label):
                settled = agree(_prove(hidden))
                self.assertIsNotNone(settled, f"refused a convention it should have found: {label}")
                assert settled is not None and settled.chosen is not None
                found: Convention = settled.chosen.convention
                self.assertEqual(found.approach_column, hidden["approach_col"])
                self.assertEqual(found.approach_sign, hidden["approach_sign"])
                self.assertEqual(found.axis_column, hidden["axis_col"])
                self.assertEqual(found.scale, 1.0 if hidden["unit"] == "m" else 0.001)

    def test_the_fitted_offset_lands_near_the_hidden_one(self) -> None:
        """The offset is fitted rather than enumerated, so it is the one answer that can be close
        rather than exact. Within one grid step is what the search can resolve."""
        for label, hidden in self.HIDDEN:
            with self.subTest(label):
                settled = agree(_prove(hidden))
                assert settled is not None and settled.chosen is not None
                found_mm = settled.chosen.convention.centre_offset_m * 1000.0
                self.assertLess(abs(found_mm - hidden["offset_mm"]), 10.0,
                                f"fitted {found_mm:+.0f} mm against a hidden {hidden['offset_mm']:+.0f}")


class RefusalTests(unittest.TestCase):
    """⛔ Refusing is a result. Each of these is a real limit of the method, not a bug in it."""

    def test_poses_unrelated_to_the_cloud_are_refused(self) -> None:
        """The check that stops an importer converting a mismatched pair of files in silence."""
        rng = np.random.default_rng(7)
        cloud = _scene(rng)
        poses = np.zeros((150, 4, 4))
        for row in range(150):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            poses[row, :3, :3] = rotation * np.sign(np.linalg.det(rotation))
            poses[row, :3, 3] = rng.uniform(-0.25, 0.25, size=3)
            poses[row, 3, 3] = 1.0
        proof = prove_convention(cloud / 1000.0, poses, np.full(150, 0.06))
        self.assertFalse(proof.settled)
        assert proof.refusal is not None
        self.assertIn("contacts on the surface", proof.refusal)

    def test_a_lone_small_object_cannot_settle_the_approach(self) -> None:
        """⚠ THE REAL LIMIT, and it is not the one I predicted.

        My assumption was that the support surface does the work, because a reversed approach drives
        the palm under the table. MEASURED, that is wrong: with the table removed the four readings
        still separate 1.00 / 0.78 / 0.72 / 0.43, because an object's own body blocks the reverse of
        most grasps on it. What genuinely defeats the palm claim is an object SMALL next to the
        gripper with nothing around it. Here all four readings score 1.00 and the module refuses,
        which is the outcome an importer can act on.
        """
        rng = np.random.default_rng(3)
        half = np.array([20.0, 20.0, 20.0])
        cloud = _box(rng, 2000, half, np.zeros(3))
        centres, approaches, axes, widths = [], [], [], []
        for _ in range(300):
            axis_index, approach_index = rng.permutation(3)[:2]
            axis = np.zeros(3)
            axis[axis_index] = 1.0
            approach = np.zeros(3)
            approach[approach_index] = rng.choice([-1.0, 1.0])
            centres.append((rng.uniform(-0.7, 0.7, size=3) * half) * (1.0 - axis))
            approaches.append(approach)
            axes.append(axis)
            widths.append(2.0 * half[axis_index])
        poses, published = _encode(np.array(centres), np.array(approaches), np.array(axes),
                                   np.array(widths), approach_col=2, approach_sign=1, axis_col=0,
                                   unit="m", offset_mm=0.0)
        proof = prove_convention(cloud / 1000.0, poses, published, sample_grasps=120)
        self.assertFalse(proof.settled)
        assert proof.refusal is not None
        self.assertIn("the approach is not", proof.refusal)

    def test_the_table_is_NOT_what_settles_the_sign(self) -> None:
        """The refutation of my own hypothesis, kept so it cannot quietly come back. A scene with
        enough object content settles the approach whether or not a support surface is present."""
        settled = agree(_prove(dict(approach_col=2, approach_sign=1, axis_col=0, unit="m",
                                    offset_mm=0.0), table=False))
        self.assertIsNotNone(settled, "removing the table should not have cost the answer")
        assert settled is not None and settled.chosen is not None
        self.assertEqual(settled.chosen.convention.approach_column, 2)
        self.assertEqual(settled.chosen.convention.approach_sign, 1)

    def test_an_unfiltered_fixture_defeats_the_clearance_claim(self) -> None:
        """⛔ THE FIXTURE BUG THAT COST TWO ROUNDS, kept as a test. With unreachable grasps in the
        table, the TRUE convention fails its own palm claim, so nothing separates it from its
        reversal. This is why the fixture filters and why a real dataset can be trusted to have."""
        proofs = _prove(dict(approach_col=2, approach_sign=1, axis_col=0, unit="m", offset_mm=0.0),
                        reachable=False)
        self.assertIsNone(agree(proofs),
                          "unreachable grasps should not produce an agreed convention")

    def test_an_empty_or_tiny_input_refuses_rather_than_raising(self) -> None:
        cloud = _scene(np.random.default_rng(1))
        empty = prove_convention(cloud / 1000.0, np.zeros((0, 4, 4)), np.zeros(0))
        self.assertFalse(empty.settled)
        assert empty.refusal is not None
        self.assertIn("no grasp", empty.refusal)
        tiny = prove_convention(np.zeros((4, 3)), np.tile(np.eye(4), (3, 1, 1)), np.full(3, 0.06))
        self.assertFalse(tiny.settled)

    def test_a_width_that_is_no_gripper_refuses(self) -> None:
        """Neither metres nor millimetres of anything we could hold. Refused before any geometry."""
        cloud = _scene(np.random.default_rng(2))
        proof = prove_convention(cloud / 1000.0, np.tile(np.eye(4), (10, 1, 1)), np.full(10, 900.0))
        self.assertFalse(proof.settled)
        assert proof.refusal is not None
        self.assertIn("plausible jaw opening", proof.refusal)


class ShapeTests(unittest.TestCase):

    def test_a_pose_array_of_the_wrong_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            prove_convention(np.zeros((100, 3)), np.zeros((5, 3, 3)), np.zeros(5))

    def test_mismatched_widths_raise(self) -> None:
        with self.assertRaises(ValueError):
            prove_convention(np.zeros((100, 3)), np.tile(np.eye(4), (5, 1, 1)), np.zeros(4))

    def test_agree_returns_nothing_when_scenes_disagree(self) -> None:
        """⭐ ONE SCENE IS NOT EVIDENCE. A convention that wins by an accident of one object's
        symmetry must not survive a second opinion."""
        first = _prove(dict(approach_col=2, approach_sign=1, axis_col=0, unit="m", offset_mm=0.0),
                       scenes=1)
        second = _prove(dict(approach_col=0, approach_sign=1, axis_col=1, unit="mm", offset_mm=0.0),
                        scenes=1)
        self.assertIsNone(agree(first + second))


if __name__ == "__main__":
    unittest.main()
