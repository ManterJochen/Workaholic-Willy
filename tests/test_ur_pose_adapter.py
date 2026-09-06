"""
Tests for the UR vendor boundary adapter
(``backend.src.robot.drivers.ur.pose_adapter``).

These tests mirror ``test_kuka_pose_adapter.py`` to keep vendor parity:
both vendor drivers expose a bidirectional ``Pose`` adapter, and both
must satisfy the same numerical / contractual guarantees.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import from_axis_angle
from src.robot.drivers.ur.pose import URPose
from src.robot.drivers.ur.pose_adapter import pose_to_urpose, urpose_to_pose


class URAdapterRoundTripTests(unittest.TestCase):
    """The adapter must round-trip URPose <-> Pose exactly."""

    def test_round_trip_identity(self) -> None:
        urpose = URPose(x=300.0, y=-50.0, z=200.0, rx=0.0, ry=np.pi, rz=0.0, label="ident")
        pose = urpose_to_pose(urpose)
        back = pose_to_urpose(pose)
        np.testing.assert_allclose(
            [back.x, back.y, back.z, back.rx, back.ry, back.rz],
            [urpose.x, urpose.y, urpose.z, urpose.rx, urpose.ry, urpose.rz],
            atol=1e-9,
        )
        self.assertEqual(back.label, urpose.label)

    def test_round_trip_random_rotations(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(10):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = float(rng.uniform(-np.pi, np.pi))
            rvec = axis * angle
            urpose = URPose(
                x=float(rng.uniform(-500.0, 500.0)),
                y=float(rng.uniform(-500.0, 500.0)),
                z=float(rng.uniform(50.0, 600.0)),
                rx=float(rvec[0]),
                ry=float(rvec[1]),
                rz=float(rvec[2]),
            )
            back = pose_to_urpose(urpose_to_pose(urpose))
            # Position: exact.
            np.testing.assert_allclose(
                [back.x, back.y, back.z], [urpose.x, urpose.y, urpose.z], atol=1e-9
            )
            # Orientation: compare rotation matrices (axis-angle ambiguity).
            np.testing.assert_allclose(urpose.to_T()[:3, :3], back.to_T()[:3, :3], atol=1e-9)


class URAdapterContractTests(unittest.TestCase):
    """Frame + label + numerics contract guarantees."""

    def test_default_frame_is_base(self) -> None:
        urpose = URPose(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0)
        pose = urpose_to_pose(urpose)
        self.assertIs(pose.frame, Frame.BASE)

    def test_explicit_frame_is_honoured(self) -> None:
        urpose = URPose(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0)
        pose = urpose_to_pose(urpose, frame=Frame.CAMERA)
        self.assertIs(pose.frame, Frame.CAMERA)

    def test_label_override_wins(self) -> None:
        urpose = URPose(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0, label="original")
        pose = urpose_to_pose(urpose, label="override")
        self.assertEqual(pose.label, "override")

    def test_label_falls_back_to_urpose_label(self) -> None:
        urpose = URPose(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0, label="kept")
        pose = urpose_to_pose(urpose)
        self.assertEqual(pose.label, "kept")

    def test_position_is_float64_mm(self) -> None:
        urpose = URPose(x=1.0, y=2.0, z=3.0, rx=0.0, ry=0.0, rz=0.0)
        pose = urpose_to_pose(urpose)
        self.assertEqual(pose.position_mm.dtype, np.float64)
        np.testing.assert_allclose(pose.position_mm, [1.0, 2.0, 3.0])

    def test_pose_to_urpose_matches_manual_axis_angle(self) -> None:
        rvec = np.array([0.0, np.pi / 2, 0.0])
        quat = from_axis_angle(rvec)
        pose = Pose(
            position_mm=np.array([10.0, 20.0, 30.0]),
            quaternion_xyzw=quat,
            frame=Frame.BASE,
        )
        urpose = pose_to_urpose(pose)
        np.testing.assert_allclose([urpose.x, urpose.y, urpose.z], [10.0, 20.0, 30.0])
        np.testing.assert_allclose([urpose.rx, urpose.ry, urpose.rz], rvec, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
