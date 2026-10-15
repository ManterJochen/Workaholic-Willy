"""The one chain reproduces the UR guard's hand-typed one, bit for bit (P0d D1).

`src/geometry/kinematics.py` is where every arm's kinematics will live: the four vendors of this arc need
one chain, and today there are two ways to get a UR link frame, the DH table in `safety/_ur_kinematics.py` and
nothing else. D1 adds the chain beside it and proves they are the same arithmetic. Nothing reads the new module
yet; D3 moves the readers one at a time, and D4 deletes the table.

Exact equality, not a tolerance, and that is the point. Every committed UR artefact was measured through
the old loop: the collision bundles (`safety/data/*_collision_meshes.npz`), the cuRobo sphere maps, the retract
rows in `retract.yaml` and the combination evidence files. A replacement that agreed to 1e-12 would regenerate
all of them differently, and the difference would surface as a re-blessed golden rather than as a failure here.
Floating point multiplication is not associative, so "the same formula" is not enough: it has to be the same
spelling, and this file is what says so.

`assertTrue(np.array_equal(...))` rather than `assert_allclose`: `array_equal` is the exact comparison, and it
treats `-0.0` and `0.0` as equal, which is the one difference a zero theta offset can legitimately produce.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.config.arms import available_arms, load_arm
from src.geometry.kinematics import DhRow, SerialChain
from src.robot.safety._ur_kinematics import (
    ur_dh_rows,
    ur_link_origins_mm,
    ur_link_transforms_mm,
)

#: The joint names of every UR, from the arm registry. Until 2026-09-22 this was a hand-written tuple without the
#: `_joint` suffix under a comment claiming it was "as drivers/ur/curobo_motion.UR_ARM_JOINT_NAMES writes it",
#: which it was not: that tuple and UR's own files both carry the suffix. Harmless here, since names do not enter
#: forward kinematics, but a comment that states the wrong fact is the kind the next reader copies.
_UR_JOINT_NAMES = tuple(joint.name for joint in load_arm("ur5e").joints)

#: How many random joint vectors each model is compared on. A fixed seed, so a failure is reproducible and a
#: green run is the same green run tomorrow.
_SAMPLES = 1000
_SEED = 20260920


def _rows(model: str) -> tuple[DhRow, ...]:
    """The arm registry's rows, as the chain's own row type.

    Both sides of every comparison below read the same rows since P0d D4, so this file no longer checks the
    numbers (that is `test_inlined_dh_tables.py`, against the bake scripts, exactly). It checks the arithmetic:
    `SerialChain` and the guard's loop in `safety/_ur_kinematics.py` are two separate implementations, and they
    must turn the same rows into the same frames, bit for bit.
    """
    described = ur_dh_rows(model)
    assert described is not None, f"{model} has no joints in the arm registry"
    return tuple(DhRow(a_m=row.a_m, d_m=row.d_m, alpha_rad=row.alpha_rad) for row in described)


def _chain(model: str) -> SerialChain:
    return SerialChain.from_dh(_rows(model), joint_names=_UR_JOINT_NAMES)


class TheChainIsTheTableTests(unittest.TestCase):
    def test_every_model_reproduces_every_link_frame_exactly(self) -> None:
        rng = np.random.default_rng(_SEED)
        for model in available_arms():
            with self.subTest(model=model):
                chain = _chain(model)
                for _ in range(_SAMPLES):
                    joints = rng.uniform(-2.0 * np.pi, 2.0 * np.pi, size=chain.dof)
                    theirs = ur_link_transforms_mm(model, joints)
                    ours = chain.link_transforms_mm(joints)
                    self.assertIsNotNone(theirs)
                    self.assertIsNotNone(ours)
                    assert theirs is not None and ours is not None  # narrowing for mypy
                    self.assertEqual(len(theirs), len(ours))
                    for index, (a, b) in enumerate(zip(theirs, ours, strict=True)):
                        self.assertTrue(
                            np.array_equal(a, b),
                            f"{model} link {index} differs at joints {joints.tolist()}:\n{a}\n{b}",
                        )

    def test_the_origins_agree_too(self) -> None:
        """The other door into the same loop, which the capsule pipeline uses."""
        rng = np.random.default_rng(_SEED + 1)
        for model in available_arms():
            with self.subTest(model=model):
                chain = _chain(model)
                joints = rng.uniform(-np.pi, np.pi, size=chain.dof)
                theirs = ur_link_origins_mm(model, joints)
                ours = chain.link_origins_mm(joints)
                assert theirs is not None and ours is not None
                for a, b in zip(theirs, ours, strict=True):
                    self.assertTrue(np.array_equal(a, b), f"{model}: {a} != {b}")

    def test_the_zero_pose_is_the_zero_pose(self) -> None:
        """The positive control: a comparison of two things that are both wrong passes silently.

        Both loops could be wrong in the same way and every comparison above would still pass. So this pins the
        flange at the zero pose to three numbers taken from UR's published table by hand, not from either loop:

            x = a2 + a3   = -0.425  - 0.3922 = -0.8172 m
            y = -(d4 + d6) = -(0.1333 + 0.0996) = -0.2329 m
            z = d1 - d5   =  0.1625 - 0.0997 =  0.0628 m

        At all-zero joints the chain is straight, so the two upper-arm lengths add along x, the two wrist offsets
        that point one way add along y, and the base height less the fifth offset is the flange height.
        """
        frames = _chain("ur5e").link_transforms_mm(np.zeros(6))
        assert frames is not None
        flange = frames[-1]
        self.assertAlmostEqual(-817.2, float(flange[0, 3]), places=1, msg="the UR5e reach along x at the zero pose")
        self.assertAlmostEqual(-232.9, float(flange[1, 3]), places=1)
        self.assertAlmostEqual(62.8, float(flange[2, 3]), places=1)


class TheChainRefusesWhatItCannotAnswerTests(unittest.TestCase):
    def test_a_wrong_joint_count_is_unavailable_rather_than_wrong(self) -> None:
        """`None` is the guard's word for "ask something coarser", which a raised exception is not."""
        chain = _chain("ur5e")
        self.assertIsNone(chain.link_transforms_mm(np.zeros(5)))
        self.assertIsNone(chain.link_origins_mm(np.zeros(7)))

    def test_a_chain_with_no_joints_is_refused_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            SerialChain.from_dh((), joint_names=())

    def test_a_name_per_joint_in_the_order_a_vector_arrives_in(self) -> None:
        rows = _rows("ur5e")
        with self.assertRaises(ValueError):
            SerialChain.from_dh(rows, joint_names=("a", "b"))
        with self.assertRaises(ValueError):
            SerialChain.from_dh(rows, joint_names=("a", "a", "c", "d", "e", "f"))

    def test_a_theta_offset_moves_the_frame(self) -> None:
        """The negative control for the field UR does not use: a chain that ignored it would pass every test above."""
        rows = _rows("ur5e")
        turned = (DhRow(rows[0].a_m, rows[0].d_m, rows[0].alpha_rad, theta_offset_rad=np.pi / 2), *rows[1:])

        plain = SerialChain.from_dh(rows, joint_names=_UR_JOINT_NAMES).link_transforms_mm(np.zeros(6))
        offset = SerialChain.from_dh(turned, joint_names=_UR_JOINT_NAMES).link_transforms_mm(np.zeros(6))
        assert plain is not None and offset is not None
        self.assertFalse(np.array_equal(plain[-1], offset[-1]), "the theta offset changed nothing")

    def test_render_describes_the_chain_to_a_person(self) -> None:
        text = _chain("ur3e").render()
        self.assertIn("6 joints", text)
        self.assertIn("shoulder_pan", text)
        self.assertFalse(text.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
