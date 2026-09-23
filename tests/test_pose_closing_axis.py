"""``closing_axis``: which horizontal axis a pose's closing axis lines up with (owner, 2026-09-23).

The owner's cell wound wrist 3 on every move made with ``Pose.tool_down`` and ``Pose.aimed_at``. Both fixed the
tool's turn about the vertical in the world, and a UR keeps that only by turning wrist 3 as far as its base turns:
measured with this repository's UR kinematics, one degree of wrist 3 per degree of base. ``closing_axis`` lets a
caller line the closing axis up with ``radial`` or ``tangential`` instead, which follow the point round the base,
so the same wrist angle reaches every point. The default ``x`` is the old pose, bit for bit.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.pose import CLOSING_AXES
from src.robot.safety._ur_kinematics import ur_link_transforms_mm


def _heading_deg(pose: Pose) -> float:
    x_axis = pose.to_matrix()[:3, 0]
    return math.degrees(math.atan2(x_axis[1], x_axis[0]))


def _same_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.allclose(a[:3, :3], b[:3, :3], atol=1e-9))


class TheDefaultIsTheOldPoseTests(unittest.TestCase):
    def test_x_is_bit_for_bit_the_pose_without_the_keyword(self) -> None:
        for yaw in (0.0, -0.0, 12.5, 90.0, -137.0, 180.0, 359.9):
            with self.subTest(yaw=yaw):
                old = Pose.tool_down(400.0, 100.0, 300.0, yaw_deg=yaw)
                new = Pose.tool_down(400.0, 100.0, 300.0, yaw_deg=yaw, closing_axis="x")
                self.assertEqual(old, new)
                self.assertEqual(old.quaternion_xyzw.tobytes(), new.quaternion_xyzw.tobytes())

    def test_aimed_at_without_it_is_unchanged(self) -> None:
        aim = {"target_mm": (500.0, 0.0, 50.0)}
        self.assertEqual(Pose.aimed_at(300.0, 200.0, 400.0, **aim),
                         Pose.aimed_at(300.0, 200.0, 400.0, closing_axis=None, **aim))


class TheAxesTests(unittest.TestCase):
    def test_y_is_a_quarter_turn(self) -> None:
        self.assertTrue(_same_rotation(Pose.tool_down(400.0, 100.0, 300.0, closing_axis="y").to_matrix(),
                                       Pose.tool_down(400.0, 100.0, 300.0, yaw_deg=90.0).to_matrix()))

    def test_radial_points_away_from_the_base_and_tangential_a_quarter_turn_clockwise(self) -> None:
        for x, y in ((400.0, 0.0), (0.0, 300.0), (-250.0, -250.0), (120.0, -480.0)):
            with self.subTest(x=x, y=y):
                bearing = math.degrees(math.atan2(y, x))
                radial = Pose.tool_down(x, y, 200.0, closing_axis="radial")
                tangential = Pose.tool_down(x, y, 200.0, closing_axis="tangential")
                self.assertAlmostEqual(math.cos(math.radians(_heading_deg(radial) - bearing)), 1.0, places=12)
                self.assertAlmostEqual(math.cos(math.radians(_heading_deg(tangential) - bearing + 90.0)), 1.0,
                                       places=12)
                np.testing.assert_allclose(radial.to_matrix()[:3, 2], (0.0, 0.0, -1.0), atol=1e-12)

    def test_a_minus_takes_the_opposite_direction_and_a_plus_changes_nothing(self) -> None:
        at = (300.0, 200.0, 250.0)
        for axis, same_as in (("-y", {"yaw_deg": -90.0}), ("-x", {"yaw_deg": 180.0}),
                              ("+y", {"closing_axis": "y"}), ("+x", {})):
            with self.subTest(axis=axis):
                self.assertTrue(_same_rotation(Pose.tool_down(*at, closing_axis=axis).to_matrix(),
                                               Pose.tool_down(*at, **same_as).to_matrix()))
        toward_base = Pose.tool_down(*at, closing_axis="-radial").to_matrix()[:3, 0]
        np.testing.assert_allclose(toward_base[:2], -np.array(at[:2]) / math.hypot(*at[:2]), atol=1e-12)
        self.assertTrue(_same_rotation(Pose.tool_down(*at, closing_axis="-tangential").to_matrix(),
                                       Pose.tool_down(*at, closing_axis="tangential", yaw_deg=180.0).to_matrix()))

    def test_the_yaw_turns_on_top_of_the_axis(self) -> None:
        a = Pose.tool_down(300.0, 300.0, 200.0, closing_axis="radial", yaw_deg=30.0)
        self.assertAlmostEqual(_heading_deg(a), 45.0 + 30.0, places=9)

    def test_an_unknown_axis_is_refused_naming_the_choices(self) -> None:
        with self.assertRaises(ValueError) as caught:
            Pose.tool_down(400.0, 0.0, 300.0, closing_axis="z")
        for axis in CLOSING_AXES:
            self.assertIn(axis, str(caught.exception))

    def test_radial_on_the_vertical_axis_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            Pose.tool_down(0.0, 0.0, 300.0, closing_axis="radial")
        self.assertIn("vertical axis", str(caught.exception))


class AimedAtFollowsTheAxisTests(unittest.TestCase):
    def test_aimed_straight_down_it_is_tool_down_with_the_same_axis(self) -> None:
        for axis in CLOSING_AXES:
            with self.subTest(axis=axis):
                aimed = Pose.aimed_at(400.0, 100.0, 500.0, target_mm=(400.0, 100.0, 0.0), closing_axis=axis)
                down = Pose.tool_down(400.0, 100.0, 500.0, closing_axis=axis)
                np.testing.assert_allclose(aimed.to_matrix()[:3, :3], down.to_matrix()[:3, :3], atol=1e-12)

    def test_no_other_roll_brings_the_closing_axis_nearer_the_axis(self) -> None:
        eye = (450.0, 150.0, 400.0)
        pose = Pose.aimed_at(*eye, target_mm=(600.0, 0.0, 0.0), closing_axis="tangential")
        rotation = pose.to_matrix()[:3, :3]
        forward = rotation[:, 2]
        aim = np.array([150.0, -150.0, -400.0]) / np.linalg.norm([150.0, -150.0, -400.0])
        np.testing.assert_allclose(forward, aim, atol=1e-12)          # it still points at the target
        heading = math.radians(math.degrees(math.atan2(eye[1], eye[0])) - 90.0)
        wanted = np.array([math.cos(heading), math.sin(heading), 0.0])
        best = float(np.dot(rotation[:, 0], wanted))
        for roll in np.radians(np.arange(-180.0, 180.0, 5.0)):
            turned = math.cos(roll) * rotation[:, 0] + math.sin(roll) * rotation[:, 1]
            self.assertLessEqual(float(np.dot(turned, wanted)), best + 1e-12)

    def test_an_aim_along_the_axis_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            Pose.aimed_at(400.0, 0.0, 300.0, target_mm=(800.0, 0.0, 300.0), closing_axis="radial")
        self.assertIn("runs along", str(caught.exception))


class OneWristAngleReachesEveryPointTests(unittest.TestCase):
    """The point of the keyword, on the robot: the UR3e with wrist 3 held, turned round its base.

    q2..q4 sum to -90 degrees and wrist 2 is -90, so the flange points straight down. Every base angle then reaches
    a pose that ``tangential`` plus one fixed yaw describes, so a caller who asks for that pose round the base gets
    the same wrist 3 each time. ``x`` with a fixed yaw does not: each of those points needs wrist 3 turned by as
    much as the base.
    """

    @staticmethod
    def _flange(q1_deg: float, q6_deg: float) -> np.ndarray:
        q = np.radians([q1_deg, -57.2957795, 57.2957795, -90.0, -90.0, q6_deg])
        return ur_link_transforms_mm("ur3e", q)[-1]

    def test_tangential_with_a_fixed_yaw_is_one_wrist_angle(self) -> None:
        first = self._flange(150.0, 0.0)
        np.testing.assert_allclose(first[:3, 2], (0.0, 0.0, -1.0), atol=1e-6)
        x, y = first[0, 3], first[1, 3]
        yaw = _heading_deg(Pose.from_matrix(first, frame=Frame.BASE)) - (math.degrees(math.atan2(y, x)) - 90.0)
        for q1 in (150.0, 170.0, 190.0, 210.0, 240.0):
            with self.subTest(q1=q1):
                T = self._flange(q1, 0.0)
                pose = Pose.tool_down(T[0, 3], T[1, 3], T[2, 3], closing_axis="tangential", yaw_deg=yaw)
                self.assertTrue(_same_rotation(pose.to_matrix(), T),
                                "the same wrist 3 no longer reaches the pose tangential describes")

    def test_a_fixed_world_yaw_needs_wrist_3_turned(self) -> None:
        """⭐ THE CONTROL: the old pose round the base is not one wrist angle."""
        first = self._flange(150.0, 0.0)
        yaw = _heading_deg(Pose.from_matrix(first, frame=Frame.BASE))
        T = self._flange(190.0, 0.0)
        pose = Pose.tool_down(T[0, 3], T[1, 3], T[2, 3], yaw_deg=yaw)
        self.assertFalse(_same_rotation(pose.to_matrix(), T))
        self.assertTrue(_same_rotation(pose.to_matrix(), self._flange(190.0, 40.0)),
                        "40 degrees of base did not cost 40 degrees of wrist 3")


if __name__ == "__main__":
    unittest.main()
