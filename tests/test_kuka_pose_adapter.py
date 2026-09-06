"""
Tests for the KUKA vendor boundary adapter
(``backend.src.robot.drivers.kuka.pose_convert``).

These tests mirror ``test_ur_pose_adapter.py`` to keep vendor parity:
both vendor drivers expose a bidirectional ``Pose`` adapter and a
lightweight value-type for the controller wire format, and both must
satisfy the same numerical / contractual guarantees.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import from_euler
from src.robot.drivers.kuka.pose_convert import (
    KukaCartesian,
    kuka_cartesian_to_pose,
    pose_to_kuka_cartesian,
)


class KukaAdapterRoundTripTests(unittest.TestCase):
    """The adapter must round-trip KukaCartesian <-> Pose exactly."""

    def test_round_trip_identity(self) -> None:
        e6 = KukaCartesian(x=300.0, y=-50.0, z=200.0, a=0.0, b=0.0, c=0.0)
        pose = kuka_cartesian_to_pose(e6, label="ident")
        back = pose_to_kuka_cartesian(pose)
        np.testing.assert_allclose(back.as_tuple(), e6.as_tuple(), atol=1e-9)

    def test_round_trip_random_rotations(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(10):
            # Stay well inside the unambiguous ZYX Euler range (avoid the
            # gimbal-lock band at |B| -> 90 deg).
            a_deg = float(rng.uniform(-170.0, 170.0))
            b_deg = float(rng.uniform(-80.0, 80.0))
            c_deg = float(rng.uniform(-170.0, 170.0))
            e6 = KukaCartesian(
                x=float(rng.uniform(-500.0, 500.0)),
                y=float(rng.uniform(-500.0, 500.0)),
                z=float(rng.uniform(50.0, 600.0)),
                a=a_deg,
                b=b_deg,
                c=c_deg,
            )
            back = pose_to_kuka_cartesian(kuka_cartesian_to_pose(e6))
            # Position: exact.
            np.testing.assert_allclose(
                [back.x, back.y, back.z], [e6.x, e6.y, e6.z], atol=1e-9
            )
            # Orientation: compare rotation matrices to be robust to Euler
            # branch wrapping.
            from src.geometry.quaternion import to_rotation_matrix

            p_in = kuka_cartesian_to_pose(e6)
            p_back = kuka_cartesian_to_pose(back)
            np.testing.assert_allclose(
                to_rotation_matrix(p_in.quaternion_xyzw),
                to_rotation_matrix(p_back.quaternion_xyzw),
                atol=1e-9,
            )


class KukaAdapterContractTests(unittest.TestCase):
    """Frame + label + numerics contract guarantees."""

    def test_default_frame_is_base(self) -> None:
        e6 = KukaCartesian(x=0.0, y=0.0, z=0.0, a=0.0, b=0.0, c=0.0)
        pose = kuka_cartesian_to_pose(e6)
        self.assertIs(pose.frame, Frame.BASE)

    def test_label_is_propagated(self) -> None:
        e6 = KukaCartesian(x=0.0, y=0.0, z=0.0, a=0.0, b=0.0, c=0.0)
        pose = kuka_cartesian_to_pose(e6, label="picked")
        self.assertEqual(pose.label, "picked")

    def test_pose_to_kuka_rejects_non_base_frame(self) -> None:
        quat = from_euler(np.array([0.0, 0.0, 0.0]), order="xyz")
        pose = Pose(
            position_mm=np.array([1.0, 2.0, 3.0]),
            quaternion_xyzw=quat,
            frame=Frame.CAMERA,
        )
        with self.assertRaises(ValueError):
            pose_to_kuka_cartesian(pose)

    def test_position_is_float64_mm(self) -> None:
        e6 = KukaCartesian(x=1.0, y=2.0, z=3.0, a=0.0, b=0.0, c=0.0)
        pose = kuka_cartesian_to_pose(e6)
        self.assertEqual(pose.position_mm.dtype, np.float64)
        np.testing.assert_allclose(pose.position_mm, [1.0, 2.0, 3.0])

    def test_serialization_dict_round_trip(self) -> None:
        e6 = KukaCartesian(x=10.0, y=20.0, z=30.0, a=15.0, b=-5.0, c=45.0)
        d = e6.to_dict()
        self.assertEqual(set(d.keys()), {"X", "Y", "Z", "A", "B", "C"})
        e6_back = KukaCartesian.from_dict(d)
        self.assertEqual(e6.as_tuple(), e6_back.as_tuple())


if __name__ == "__main__":
    unittest.main()
