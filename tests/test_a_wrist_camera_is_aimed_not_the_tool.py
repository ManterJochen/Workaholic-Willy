"""An eye-in-hand sweep aims the CAMERA at its marker, from one look at it, not the tool (chain K, step 2).

The owner's cell (2026-09-23): a D415 on a bracket beside the hand, tilted about 45 degrees away from the tool axis, and
one 150 mm marker lying flat at (-130, -700, 50) mm in the base. The generated tool-down sweep and a ring of
``Pose.aimed_at`` poses both aim the TOOL, so the tilted camera photographs the table beside the marker. Here the
camera is aimed: one view of the marker estimates where the camera sits on the tool, a cone of views round the marker is
aimed from that estimate, and every later view refines it.

Everything below is synthetic geometry: a known mount, the marker pose that mount would see, the estimate, and the
camera axis of the TRUE mount at every aimed station.
"""

from __future__ import annotations

import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from src.calibration import MountingMode
from src.config.schema.robot import WorkspaceLimitsConfig
from src.geometry import Frame, Pose, Transform
from src.robot.core import MotionCommand, MotionResult
from src.robot.execution import camera_aim as aim_module
from src.robot.execution.calibration import CalibrationResult, CalibrationRoutine
from src.robot.execution.camera_aim import (
    DEFAULT_VIEWS,
    AimedSweep,
    ArmReach,
    MarkerAim,
    MountEstimate,
    NotAimed,
    View,
    aim_camera,
    aimed_stations,
    estimate_camera_in_tool,
    nominal_camera_in_tool,
    refine_camera_in_tool,
)
from src.robot.execution.hand_eye import CalibrationOutcome, SweepOptions
from src.robot.execution.pose_provider import SkippedStation
from src.robot.safety._ur_ik import ur_flange_ik
from src.robot.safety._ur_kinematics import ur_link_transforms_mm

_ROOT = Path(__file__).resolve().parents[1]
MARKER = np.array([-130.0, -700.0, 50.0])
#: The TCP 175 mm past the flange along the tool's +Z, so the flange stands at (0, 0, -175) in the tool frame.
FLANGE_TO_TCP = np.eye(4)
FLANGE_TO_TCP[2, 3] = 175.0
FLANGE_IN_TOOL = (0.0, 0.0, -175.0)
#: Where the owner stands the arm for the first look: tool down, its +X along base -Y, turned 10 degrees.
LOOK = Pose.tool_down(-100.0, -300.0, 280.0, closing_axis="-y", yaw_deg=10.0)
#: The D415 colour image, degrees across and up.
_FOV_DEG = (69.0, 42.0)


def _rigid(rotation: np.ndarray, position: Any) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = np.asarray(position, dtype=np.float64)
    return out


def _mount(offset_mm: Any = (70.0, 0.0, -150.0), tilt_deg: float = 45.0) -> np.ndarray:
    """A camera tilted ``tilt_deg`` from the tool axis toward the tool's +X, standing at ``offset_mm`` in the tool."""
    return _rigid(aim_module._exp(np.array([0.0, math.radians(tilt_deg), 0.0])), offset_mm)


#: The owner's shape: the bracket runs out the way the camera looks. And a camera 80 mm to the side of that.
ALONG = _mount((70.0, 0.0, -150.0))
SIDE = _mount((0.0, 80.0, -150.0))


def _marker_in_base(yaw_deg: float = 20.0, at: Any = MARKER) -> np.ndarray:
    """The marker flat and face up, turned about its normal by an angle nothing in the estimate is told."""
    return _rigid(aim_module._rz(math.radians(yaw_deg)), at)


def _seen(tool: Any, mount: np.ndarray, **marker: Any) -> np.ndarray:
    """The marker's pose in the camera, as the estimator would report it from ``tool`` with the true ``mount``."""
    tool_matrix = tool.to_matrix() if isinstance(tool, Pose) else np.asarray(tool)
    return np.linalg.inv(tool_matrix @ mount) @ _marker_in_base(**marker)


def _in_view(marker_in_camera: np.ndarray) -> bool:
    x, y, z = (float(v) for v in marker_in_camera[:3, 3])
    return z > 0.0 and abs(math.degrees(math.atan2(x, z))) < _FOV_DEG[0] / 2 and \
        abs(math.degrees(math.atan2(y, z))) < _FOV_DEG[1] / 2


def _miss_deg(tool: Any, mount: np.ndarray, target: Any = MARKER) -> float:
    """How far the TRUE camera's optical axis at ``tool`` passes from the marker, in degrees."""
    camera = (tool.to_matrix() if isinstance(tool, Pose) else tool) @ mount
    return aim_module._angle_deg(camera[:3, 2], np.asarray(target) - camera[:3, 3])


def _rotation_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(float(np.linalg.norm(aim_module._log(a[:3, :3].T @ b[:3, :3]))))


def _aim(**keywords: Any) -> MarkerAim:
    return MarkerAim(marker_mm=tuple(MARKER), closing_axis="-y", **keywords)


def _estimate(mount: np.ndarray = ALONG, look: Pose = LOOK) -> MountEstimate:
    return estimate_camera_in_tool(look, _seen(look, mount), MARKER, camera_prior_in_tool_mm=FLANGE_IN_TOOL)


def _nearest_to_the_look(stations: Any) -> str:
    """The label of the station whose TCP stands nearest the look's: a roll about the line of sight moves the TCP round
    the camera, so which that is is a question of the rolls as well as the places."""
    return min((float(np.linalg.norm(np.asarray(s.pose.position_mm) - LOOK.position_mm)), s.label)
               for s in stations if s.pose is not None)[1]


# ---------------------------------------------------------------------------------------------------------------------
# (a) Where the camera sits on the tool, from one look
# ---------------------------------------------------------------------------------------------------------------------


class TheMountIsEstimatedFromOneLookTests(unittest.TestCase):

    def test_the_owners_first_look_sees_the_marker(self) -> None:
        """The premise of every test below: from the look pose the true camera has the marker in its image."""
        for mount in (ALONG, SIDE):
            self.assertTrue(_in_view(_seen(LOOK, mount)))

    def test_the_estimate_explains_the_view_it_came_from_exactly(self) -> None:
        for mount in (ALONG, SIDE):
            estimate = _estimate(mount)
            placed = LOOK.to_matrix() @ estimate.matrix() @ _seen(LOOK, mount)
            np.testing.assert_allclose(placed[:3, 3], MARKER, atol=1e-6)
            np.testing.assert_allclose(placed[:3, 2], [0.0, 0.0, 1.0], atol=1e-9)
            self.assertEqual((estimate.source, estimate.views), ("one look", 1))

    def test_a_camera_whose_bracket_runs_the_way_it_looks_is_estimated_within_a_degree(self) -> None:
        """Seen from a look not turned about the vertical, the camera stands nearly on the line from the flange to the
        marker, which is where one view puts it."""
        look = Pose.tool_down(-100.0, -300.0, 280.0, closing_axis="-y")
        estimate = _estimate(ALONG, look)
        self.assertLess(_rotation_deg(estimate.matrix(), ALONG), 1.0)
        self.assertLess(float(np.linalg.norm(estimate.position_in_tool_mm - ALONG[:3, 3])), 6.0)
        self.assertAlmostEqual(estimate.tilt_deg, 45.0, delta=1.0)

    def test_a_camera_to_the_side_is_off_by_what_one_view_cannot_see(self) -> None:
        """The module docstring's limit: 80 mm across the line of sight, about 350 mm out, is 11 to 13 degrees."""
        estimate = _estimate(SIDE)
        self.assertGreater(_rotation_deg(estimate.matrix(), SIDE), 8.0)
        self.assertLess(_rotation_deg(estimate.matrix(), SIDE), 14.0)
        self.assertLess(float(np.linalg.norm(estimate.position_in_tool_mm - SIDE[:3, 3])), 90.0)

    def test_refining_from_the_same_view_changes_nothing(self) -> None:
        """Taking the camera to stand where the first pass put it, rather than at the flange, and estimating again
        from the same view returns the same estimate: the view's own position puts the line of sight where the rotation
        already has it. Only another view improves it (module docstring)."""
        for mount in (ALONG, SIDE):
            one = _estimate(mount)
            again = estimate_camera_in_tool(LOOK, _seen(LOOK, mount), MARKER,
                                            camera_prior_in_tool_mm=tuple(one.position_in_tool_mm))
            np.testing.assert_allclose(again.matrix(), one.matrix(), atol=1e-9)

    def test_a_marker_seen_face_on_names_no_turn_and_is_refused(self) -> None:
        straight = _mount((0.0, 0.0, -150.0), tilt_deg=0.0)
        above = Pose.tool_down(-130.0, -700.0, 600.0, closing_axis="-y")
        with self.assertRaisesRegex(ValueError, "face on"):
            estimate_camera_in_tool(above, _seen(above, straight), MARKER, camera_prior_in_tool_mm=FLANGE_IN_TOOL)

    def test_a_first_place_straight_above_the_marker_is_refused(self) -> None:
        """The flange 200 mm along the tool's +X from the TCP is base -Y here, right above the marker."""
        tool = Pose.tool_down(-130.0, -500.0, 400.0, closing_axis="-y")
        with self.assertRaisesRegex(ValueError, "straight above the marker"):
            estimate_camera_in_tool(tool, _seen(tool, ALONG), MARKER, camera_prior_in_tool_mm=(200.0, 0.0, 0.0))


class RefiningTests(unittest.TestCase):

    def test_two_views_fix_what_one_could_not(self) -> None:
        one = _estimate(SIDE)
        second = aimed_stations(_aim(), one, start_tool_mm=LOOK.position_mm)[0].pose
        assert second is not None
        views = [View.of(LOOK, _seen(LOOK, SIDE)), View.of(second, _seen(second, SIDE))]
        two = refine_camera_in_tool(views, MARKER, start=one, camera_prior_in_tool_mm=FLANGE_IN_TOOL)
        self.assertLess(_rotation_deg(two.matrix(), SIDE), 1e-3)
        self.assertLess(float(np.linalg.norm(two.position_in_tool_mm - SIDE[:3, 3])), 1e-2)
        self.assertEqual((two.source, two.views), ("2 views", 2))
        assert two.rms_mm is not None and two.rms_deg is not None
        self.assertLess(two.rms_mm, 1e-3)
        self.assertIn("2 views", two.line())

    def test_views_also_place_a_marker_stated_a_little_off(self) -> None:
        """Stated 28 mm off: the whole cone of views places it within a tenth of a millimetre, and the first few, which
        turn the tool about much the same axis, leave what they cannot see near where it was stated."""
        stated = MARKER + np.array([20.0, -20.0, 0.0])
        one = estimate_camera_in_tool(LOOK, _seen(LOOK, ALONG), stated, camera_prior_in_tool_mm=FLANGE_IN_TOOL)
        stations = [station.pose for station in aimed_stations(_aim(), one, start_tool_mm=LOOK.position_mm)]
        misses = []
        for count in (3, len(stations)):
            views = [View.of(tool, _seen(tool, ALONG)) for tool in [LOOK, *stations[:count]]]
            refined = refine_camera_in_tool(views, stated, start=one, camera_prior_in_tool_mm=FLANGE_IN_TOOL)
            assert refined.marker_mm is not None
            misses.append(float(np.linalg.norm(np.array(refined.marker_mm) - MARKER)))
        self.assertLess(misses[1], 0.1)
        self.assertLess(misses[1], misses[0])
        self.assertLess(misses[0], 2.0)


class AStatedMountTests(unittest.TestCase):

    def test_the_view_turns_toward_the_named_tool_axis(self) -> None:
        tilt = math.radians(30.0)
        expected = {"+x": (math.sin(tilt), 0.0, math.cos(tilt)), "-x": (-math.sin(tilt), 0.0, math.cos(tilt)),
                    "+y": (0.0, math.sin(tilt), math.cos(tilt)), "-y": (0.0, -math.sin(tilt), math.cos(tilt))}
        for toward, axis in expected.items():
            with self.subTest(toward):
                mount = nominal_camera_in_tool(30.0, toward=toward, offset_mm=(0.0, 0.0, -175.0))
                np.testing.assert_allclose(mount.optical_axis_in_tool, axis, atol=1e-12)
                np.testing.assert_allclose(mount.position_in_tool_mm, (0.0, 0.0, -175.0))
                self.assertEqual(mount.views, 0)
                self.assertTrue(mount.source.startswith("a stated mount"))

    def test_a_bad_axis_or_tilt_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "tool axis"):
            nominal_camera_in_tool(45.0, toward="z")
        with self.assertRaisesRegex(ValueError, "under 90"):
            nominal_camera_in_tool(90.0)

    def test_the_owners_mount_stated_by_hand_matches_the_camera(self) -> None:
        """Tilted toward the way the jaws close, +x, which on a closing_axis -y pose is base -Y."""
        stated = nominal_camera_in_tool(45.0, toward="+x", offset_mm=(70.0, 0.0, -150.0))
        self.assertLess(_rotation_deg(stated.matrix(), ALONG), 1e-9)


# ---------------------------------------------------------------------------------------------------------------------
# (b) Aimed stations
# ---------------------------------------------------------------------------------------------------------------------


class AimedStationsPointTheCameraAtTheMarkerTests(unittest.TestCase):

    def test_every_station_puts_the_true_camera_axis_through_the_marker_from_one_look(self) -> None:
        for name, mount, within in (("along", ALONG, 1.0), ("side", SIDE, 2.0)):
            with self.subTest(name):
                stations = aimed_stations(_aim(), _estimate(mount), start_tool_mm=LOOK.position_mm)
                self.assertEqual(len(stations), len(DEFAULT_VIEWS))
                for station in stations:
                    assert station.pose is not None
                    self.assertLess(_miss_deg(station.pose, mount), within, station.label)
                    self.assertTrue(_in_view(_seen(station.pose, mount)), station.label)

    def test_after_a_second_view_every_station_is_aimed_to_a_hundredth_of_a_degree(self) -> None:
        one = _estimate(SIDE)
        first = aimed_stations(_aim(), one, start_tool_mm=LOOK.position_mm)[0].pose
        assert first is not None
        sweep = AimedSweep(_aim(), one, camera_prior_in_tool_mm=FLANGE_IN_TOOL)
        upcoming = sweep.replan([View.of(LOOK, _seen(LOOK, SIDE)), View.of(first, _seen(first, SIDE))],
                                visited=[aim_module._view_label(DEFAULT_VIEWS[2])])
        assert upcoming is not None
        self.assertEqual(len(upcoming), len(DEFAULT_VIEWS) - 1)
        for station in upcoming:
            assert isinstance(station, Pose)
            self.assertLess(_miss_deg(station, SIDE), 0.01, station.label)
        self.assertEqual([estimate.views for estimate in sweep.estimates], [1, 2])

    def test_the_camera_stands_at_the_distance_asked_and_looks_straight_at_the_marker(self) -> None:
        estimate = _estimate(ALONG)
        for station in aimed_stations(_aim(distance_mm=450.0), estimate):
            assert station.pose is not None
            camera = station.pose.to_matrix() @ estimate.matrix()
            self.assertAlmostEqual(float(np.linalg.norm(camera[:3, 3] - MARKER)), 450.0, places=6)
            self.assertLess(aim_module._angle_deg(camera[:3, 2], MARKER - camera[:3, 3]), 1e-6)

    def test_each_station_tilts_the_tool_from_its_heading_and_then_rolls_the_camera(self) -> None:
        """With the view's roll about the line of sight undone, what is left is the smallest turn from tool-down with
        closing_axis -y that aims the camera: its angle is the tilt, no other rotation reaching the aim is smaller, and
        the tool's +X stays within that tilt of base -Y seen from above."""
        estimate = _estimate(ALONG)
        for station in aimed_stations(_aim(), estimate):
            assert station.pose is not None
            tool = station.pose.to_matrix()
            wanted = MARKER - (tool @ estimate.matrix())[:3, 3]
            tool[:3, :3] = aim_module._exp(-aim_module._unit(wanted) * math.radians(station.roll_deg)) @ tool[:3, :3]
            reference = Pose.tool_down(0.0, 0.0, 0.0, closing_axis="-y").to_matrix()
            turned = _rotation_deg(reference, tool)
            self.assertAlmostEqual(turned, station.tilt_deg, places=6)
            self.assertAlmostEqual(turned, aim_module._angle_deg(reference[:3, :3] @ estimate.optical_axis_in_tool,
                                                                 wanted), places=6)
            # The 17 default views reach 41 degrees of tilt at their widest azimuths (+-60): well inside max_tilt_deg,
            # and every one of them ran on the real sidecar with the heading kept (camera_aim.DEFAULT_VIEWS, measured
            # 2026-09-23); the roll comes on top of that tilt and is not counted in it.
            self.assertLessEqual(station.tilt_deg, 45.0)
            heading = math.degrees(math.atan2(float(tool[1, 0]), float(tool[0, 0])))
            self.assertLessEqual(abs(heading - (-90.0)), station.tilt_deg + 1e-6, station.label)

    def test_the_views_turn_the_camera_enough_and_about_more_than_one_axis(self) -> None:
        """AX=XB needs relative rotations about axes that are not parallel, and each pair of views past the solve's
        10 degree diversity threshold."""
        rotations = [station.pose.to_matrix() for station in aimed_stations(_aim(), _estimate(ALONG))
                     if station.pose is not None]
        pairs = [(a, b) for i, a in enumerate(rotations) for b in rotations[i + 1:]]
        # Every pair stays past the solve's 10 degree diversity threshold (the closest pair of the 17 is 10.3 degrees).
        self.assertGreaterEqual(min(_rotation_deg(a, b) for a, b in pairs), 10.0)
        axes = [aim_module._log(a[:3, :3].T @ b[:3, :3]) for a, b in pairs]
        axes = [axis / np.linalg.norm(axis) for axis in axes]
        self.assertLess(min(abs(float(np.dot(axes[0], axis))) for axis in axes), 0.5)

    def test_aim_camera_refuses_a_camera_standing_on_its_marker(self) -> None:
        with self.assertRaisesRegex(ValueError, "stand on the marker"):
            aim_camera(MARKER, MARKER, _estimate(ALONG))


class ScreeningTests(unittest.TestCase):

    def test_a_station_outside_the_box_less_its_margin_is_reported_and_names_the_box(self) -> None:
        box = WorkspaceLimitsConfig(x_min=-1000.0, x_max=1000.0, y_min=-320.0, y_max=1000.0, z_min=0.0, z_max=1000.0)
        stations = aimed_stations(_aim(), _estimate(ALONG), box=box, margin_mm=20.0)
        out = [station for station in stations if station.pose is None]
        kept = [station for station in stations if station.pose is not None]
        self.assertTrue(out and kept)
        for station in kept:
            self.assertGreaterEqual(float(station.pose.position_mm[1]), -300.0)
        for station in out:
            self.assertEqual(station.reason, "outside_workspace")
            self.assertIn("its TCP", station.detail)
            self.assertIn("y -300.0 to 980.0", station.detail)
            self.assertIsInstance(station.as_station(), SkippedStation)
        self.assertEqual(stations[len(kept):], out, "the stations reported come last")

    def test_a_flange_outside_the_box_is_reported_where_the_arm_says_where_its_flange_is(self) -> None:
        stations = aimed_stations(_aim(), _estimate(ALONG))
        top = max(float(station.pose.position_mm[2]) for station in stations if station.pose is not None)
        box = WorkspaceLimitsConfig(x_min=-2000.0, x_max=2000.0, y_min=-2000.0, y_max=2000.0, z_min=0.0,
                                    z_max=top + 60.0)
        reach = ArmReach(model="ur10", flange_to_tcp=FLANGE_TO_TCP)
        screened = aimed_stations(_aim(), _estimate(ALONG), box=box, reach=reach)
        out = [station for station in screened if station.pose is None]
        self.assertTrue(out, "every TCP is inside, and the highest flanges are not")
        self.assertTrue(all("its flange" in station.detail for station in out))
        for station in screened:
            if station.pose is not None:
                flange = station.pose.to_matrix() @ np.linalg.inv(FLANGE_TO_TCP)
                self.assertLessEqual(float(flange[2, 3]), top + 60.0)
        self.assertEqual(len(aimed_stations(_aim(), _estimate(ALONG), box=box)), len(DEFAULT_VIEWS),
                         "without the arm's tool frame only the TCP is boxed")
        self.assertTrue(all(station.pose is not None for station in aimed_stations(_aim(), _estimate(ALONG), box=box)))

    def test_a_station_tilting_further_than_allowed_is_reported(self) -> None:
        stations = aimed_stations(_aim(max_tilt_deg=10.0), _estimate(ALONG))
        kept = [station.label for station in stations if station.pose is not None]
        self.assertEqual(kept, ["aim_e+0_a+0"])
        self.assertTrue(all(station.reason == "too_tilted" for station in stations if station.pose is None))

    def test_the_first_station_is_the_one_nearest_where_the_tool_stands(self) -> None:
        stations = aimed_stations(_aim(), _estimate(ALONG), start_tool_mm=LOOK.position_mm)
        self.assertEqual(stations[0].label, _nearest_to_the_look(stations))
        self.assertEqual(sorted(station.label for station in stations),
                         sorted(aim_module._view_label(view) for view in DEFAULT_VIEWS))


class ReachAndTheJointWindowTests(unittest.TestCase):
    """A UR10 flange 175 mm behind the TCP, and the window a cell keeps half a turn about home."""

    def setUp(self) -> None:
        solutions = ur_flange_ik("ur10", LOOK.to_matrix() @ np.linalg.inv(FLANGE_TO_TCP))
        assert solutions
        self.start = min(solutions, key=lambda joints: abs(joints[0] - math.radians(40.0)))
        margin = math.radians(5.0)
        self.lower = tuple(q - math.pi + margin for q in self.start)
        self.upper = tuple(q + math.pi - margin for q in self.start)

    def _reach(self, lower: Any = None, upper: Any = None) -> ArmReach:
        return ArmReach(model="ur10", flange_to_tcp=FLANGE_TO_TCP, lower_rad=lower or self.lower,
                        upper_rad=upper or self.upper)

    def test_every_station_gets_a_configuration_inside_the_window_that_puts_its_tcp_there(self) -> None:
        stations = aimed_stations(_aim(), _estimate(ALONG), reach=self._reach(), start_joints=self.start)
        self.assertEqual(len(stations), len(DEFAULT_VIEWS))
        for station in stations:
            assert station.pose is not None and station.joints is not None
            for q, low, high in zip(station.joints, self.lower, self.upper):
                self.assertTrue(low <= q <= high)
            tcp = ur_link_transforms_mm("ur10", np.array(station.joints))[-1] @ FLANGE_TO_TCP
            np.testing.assert_allclose(tcp[:3, 3], station.pose.position_mm, atol=1e-6)

    def test_the_order_turns_the_joints_less_than_the_order_the_views_are_written_in(self) -> None:
        from src.robot.safety._ur_ik import nearest_goals

        aim = _aim()
        stations = aimed_stations(aim, _estimate(ALONG), reach=self._reach(), start_joints=self.start)
        greedy = sum(station.largest_rad for station in stations)
        written, here = 0.0, list(self.start)
        for station in sorted(stations, key=lambda s: aim.views.index(s.view)):
            assert station.pose is not None
            goal = nearest_goals(ur_flange_ik("ur10", station.pose.to_matrix() @ np.linalg.inv(FLANGE_TO_TCP)),
                                 current=here, lower=self.lower, upper=self.upper, velocity=1.0)[0]
            written, here = written + goal.largest_rad, list(goal.joints)
        self.assertLess(greedy, written)
        firsts = [nearest_goals(ur_flange_ik("ur10", s.pose.to_matrix() @ np.linalg.inv(FLANGE_TO_TCP)),
                                current=self.start, lower=self.lower, upper=self.upper, velocity=1.0)[0].largest_rad
                  for s in stations if s.pose is not None]
        self.assertAlmostEqual(stations[0].largest_rad, min(firsts))

    def test_a_station_with_no_configuration_inside_the_window_is_reported(self) -> None:
        tight = 0.02
        stations = aimed_stations(_aim(), _estimate(ALONG), start_joints=self.start, reach=self._reach(
            tuple(q - tight for q in self.start), tuple(q + tight for q in self.start)))
        self.assertTrue(all(station.pose is None for station in stations))
        self.assertTrue(all(station.reason == "outside_joint_window" for station in stations))
        self.assertIn("half a turn either side of home", stations[0].detail)

    def test_reach_and_the_window_are_screened_where_the_arm_cannot_say_where_it_stands(self) -> None:
        tight = 0.02
        reach = self._reach(tuple(q - tight for q in self.start), tuple(q + tight for q in self.start))
        stations = aimed_stations(_aim(), _estimate(ALONG), reach=reach, start_tool_mm=LOOK.position_mm)
        self.assertTrue(all(station.reason == "outside_joint_window" for station in stations))
        kept = aimed_stations(_aim(), _estimate(ALONG), reach=self._reach(), start_tool_mm=LOOK.position_mm)
        self.assertEqual(kept[0].label, _nearest_to_the_look(kept), "ordered by the tool's travel instead")
        self.assertTrue(all(station.pose is not None and station.joints is None for station in kept))

    def test_the_reach_of_a_connected_arm_is_read_best_effort(self) -> None:
        from src.robot.execution.camera_aim import flange_in_tool_mm, reach_of

        window = (tuple(self.lower), tuple(self.upper))
        arm = SimpleNamespace(config=SimpleNamespace(ur=SimpleNamespace(model="UR10")),
                              active_tool_frame=FLANGE_TO_TCP, _goal_joint_window=lambda: window)
        reach = reach_of(arm)
        assert reach is not None
        self.assertEqual((reach.model, reach.lower_rad, reach.upper_rad), ("ur10", window[0], window[1]))
        np.testing.assert_allclose(flange_in_tool_mm(reach.flange_to_tcp), FLANGE_IN_TOOL)
        self.assertEqual(flange_in_tool_mm(None), (0.0, 0.0, 0.0))

        def broken() -> Any:
            raise RuntimeError("no window")

        no_window = reach_of(SimpleNamespace(config=arm.config, active_tool_frame=None, _goal_joint_window=broken))
        assert no_window is not None
        self.assertIsNone(no_window.lower_rad)
        np.testing.assert_allclose(no_window.flange_to_tcp, np.eye(4))
        self.assertIsNone(reach_of(SimpleNamespace(config=SimpleNamespace(ur=SimpleNamespace(model="lbr_iiwa")))))
        self.assertIsNone(reach_of(object()))

    def test_a_marker_out_of_reach_is_reported(self) -> None:
        far = MarkerAim(marker_mm=(3000.0, 0.0, 50.0), closing_axis="-y")
        estimate = dataclass_replace(_estimate(ALONG), marker_mm=None)
        stations = aimed_stations(far, estimate, reach=self._reach(), start_joints=self.start)
        self.assertTrue(all(station.reason == "out_of_reach" for station in stations))


def dataclass_replace(estimate: MountEstimate, **changes: Any) -> MountEstimate:
    import dataclasses

    return dataclasses.replace(estimate, **changes)


class TheAimIsCheckedWhenItIsWrittenTests(unittest.TestCase):

    def test_what_an_aim_refuses(self) -> None:
        cases = {
            "distance": dict(distance_mm=100.0),
            "closing axis": dict(closing_axis="z"),
            "view": dict(views=((50.0, 0.0),)),
            "twice": dict(views=((0.0, 0.0), (0.0, 0.0))),
            "no view": dict(views=()),
            "tilt": dict(max_tilt_deg=0.0),
        }
        for name, keywords in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                MarkerAim(marker_mm=(0.0, 0.0, 0.0), **keywords)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            MarkerAim(marker_mm=(0.0, 0.0, 0.0), mount_if_unseen=np.eye(4))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MarkerAim(marker_mm=(0.0, float("nan"), 0.0))

    def test_a_refinement_the_views_contradict_is_not_taken(self) -> None:
        """The marker was turned 40 degrees between two views: no one marker lying still explains both, and the
        stations stay aimed from the estimate before."""
        one = _estimate(ALONG)
        second = aimed_stations(_aim(), one)[0].pose
        assert second is not None
        views = [View.of(LOOK, _seen(LOOK, ALONG)), View.of(second, _seen(second, ALONG, yaw_deg=60.0))]
        sweep = AimedSweep(_aim(), one, camera_prior_in_tool_mm=FLANGE_IN_TOOL)
        with self.assertLogs("src.robot.execution.camera_aim", level="WARNING") as said:
            self.assertIsNone(sweep.replan(views, visited=()))
        self.assertIn("lying flat and still", "\n".join(said.output))
        self.assertIs(sweep.mount, one)
        self.assertEqual(len(sweep.estimates), 1)


# ---------------------------------------------------------------------------------------------------------------------
# The routine: one look, then the aimed views, re-aimed as the marker is seen
# ---------------------------------------------------------------------------------------------------------------------


class _Arm:
    """An arm that goes where it is told and reports that TCP; its joints are unknown, so the order is by the tool."""

    def __init__(self, at: Pose) -> None:
        self.at = at
        self.moves: list[Pose] = []

    def move(self, pose: Pose, **_: Any) -> MotionResult:
        self.moves.append(pose)
        self.at = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

    def get_tcp_pose(self) -> Pose:
        return self.at

    def get_joint_positions(self) -> Any:
        raise RuntimeError("this double has no joints")

    def wait_until_steady(self, _timeout: float) -> bool:
        return True


class _Camera:
    """The marker as the TRUE mount sees it from wherever the arm stands, or nothing outside the image."""

    def __init__(self, arm: _Arm, mount: np.ndarray) -> None:
        self.arm = arm
        self.mount = mount
        self.last_observation: Any = None
        self.frames = 0

    def __call__(self) -> "np.ndarray | None":
        self.frames += 1
        seen = _seen(self.arm.get_tcp_pose(), self.mount)
        if not _in_view(seen):
            self.last_observation = SimpleNamespace(why_not="no DICT_4X4_100 marker in view", T_cam_to_target=None)
            return None
        self.last_observation = SimpleNamespace(why_not="", T_cam_to_target=seen, reprojection_px=0.1, n_points=4,
                                                hint="")
        return seen


def _routine(arm: _Arm, mount: np.ndarray, *, box: Any = None) -> CalibrationRoutine:
    limits = box or WorkspaceLimitsConfig(x_min=-2000.0, x_max=2000.0, y_min=-2000.0, y_max=2000.0, z_min=-2000.0,
                                          z_max=2000.0)
    arm.active_tool_frame = FLANGE_TO_TCP  # type: ignore[attr-defined]
    return CalibrationRoutine(
        arm=arm, marker_source=_Camera(arm, mount), workspace_limits=limits,  # type: ignore[arg-type]
        calibration_mode="eye_in_hand", settle_time_s=0.0,
        eth_settings=SimpleNamespace(mode="eye_in_hand", min_samples=4, min_distance_mm=10.0,  # type: ignore[arg-type]
                                     min_angle=2.0, min_angle_deg=2.0),
    )


class TheSweepLooksOnceAndAimsTests(unittest.TestCase):

    def test_the_sweep_solves_the_mount_it_was_aimed_by(self) -> None:
        for name, mount in (("along", ALONG), ("side", SIDE)):
            with self.subTest(name):
                arm = _Arm(LOOK)
                routine = _routine(arm, mount)
                result = routine.run_aimed(_aim())
                self.assertEqual(len(arm.moves), len(DEFAULT_VIEWS))
                self.assertTrue(all(pose.label.startswith("aim_") for pose in arm.moves))
                for pose in arm.moves:
                    self.assertLess(_miss_deg(pose, mount), 2.0, pose.label)
                self.assertEqual(result.num_samples, len(DEFAULT_VIEWS))
                self.assertEqual(len(result.pose_log), len(DEFAULT_VIEWS))
                assert result.T_cam_to_tool is not None
                self.assertLess(_rotation_deg(result.T_cam_to_tool, mount), 0.05)
                self.assertLess(float(np.linalg.norm(result.T_cam_to_tool[:3, 3] - mount[:3, 3])), 0.5)
                sources = [estimate.source for estimate in result.aim_estimates]
                self.assertEqual(sources[0], "one look")
                self.assertEqual(sources[-1], f"{len(DEFAULT_VIEWS) + 1} views")

    def test_the_first_look_commands_nothing_and_counts_no_sample(self) -> None:
        arm = _Arm(LOOK)
        routine = _routine(arm, ALONG)
        look = routine.look()
        self.assertEqual(arm.moves, [])
        assert look.marker_in_camera is not None
        np.testing.assert_allclose(look.marker_in_camera, _seen(LOOK, ALONG))
        self.assertIsNone(look.joints)
        self.assertEqual(len(routine.calibrator.dataset), 0)

    def test_a_first_look_that_sees_no_marker_moves_nothing_and_says_what_to_do(self) -> None:
        away = Pose.tool_down(400.0, 300.0, 400.0, closing_axis="-y")
        arm = _Arm(away)
        with self.assertRaises(NotAimed) as caught:
            _routine(arm, ALONG).run_aimed(_aim())
        self.assertEqual(arm.moves, [])
        said = str(caught.exception)
        for part in ("saw no marker", "no DICT_4X4_100 marker in view", "nothing moved", "Jog the arm",
                     "mount_if_unseen"):
            self.assertIn(part, said)

    def test_a_stated_mount_is_aimed_from_until_a_station_sees_the_marker(self) -> None:
        away = Pose.tool_down(400.0, 300.0, 400.0, closing_axis="-y")
        arm = _Arm(away)
        stated = nominal_camera_in_tool(40.0, toward="+x", offset_mm=(50.0, 0.0, -175.0))
        result = _routine(arm, ALONG).run_aimed(_aim(mount_if_unseen=stated))
        sources = [estimate.source for estimate in result.aim_estimates]
        self.assertTrue(sources[0].startswith("a stated mount"), sources)
        self.assertEqual(sources[1], "one look")
        assert result.T_cam_to_tool is not None
        self.assertLess(_rotation_deg(result.T_cam_to_tool, ALONG), 0.05)

    def test_a_view_the_box_refuses_is_reported_and_never_moved_to(self) -> None:
        box = WorkspaceLimitsConfig(x_min=-2000.0, x_max=2000.0, y_min=-320.0, y_max=2000.0, z_min=-2000.0,
                                    z_max=2000.0)
        arm = _Arm(LOOK)
        result = _routine(arm, ALONG, box=box).run_aimed(_aim())
        skipped = [verdict for verdict in result.pose_log if verdict.reason == "outside_workspace"]
        self.assertTrue(skipped)
        self.assertEqual(len(result.pose_log), len(DEFAULT_VIEWS))
        moved = {pose.label for pose in arm.moves}
        self.assertFalse(moved & {verdict.label for verdict in skipped})

    def test_a_sweep_of_no_stations_still_ends_in_the_solves_own_refusal(self) -> None:
        """The loop that takes re-planned stations counts them as it goes; with none it must not lose its count."""
        from src.calibration.exceptions import CalibrationDataError

        arm = _Arm(LOOK)
        with self.assertRaisesRegex(CalibrationDataError, "too few samples"):
            _routine(arm, ALONG).run_with_poses([])
        self.assertEqual(arm.moves, [])

    def test_only_an_eye_in_hand_routine_is_aimed(self) -> None:
        arm = _Arm(LOOK)
        routine = _routine(arm, ALONG)
        routine.calibration_mode = MountingMode.EYE_TO_HAND
        with self.assertRaisesRegex(ValueError, "eye_in_hand"):
            routine.run_aimed(_aim())


# ---------------------------------------------------------------------------------------------------------------------
# The noun, the CLI and example 10
# ---------------------------------------------------------------------------------------------------------------------


def _eih_result(**extra: Any) -> CalibrationResult:
    transform = Transform(translation_mm=np.array([70.0, 0.0, -150.0]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                          from_frame=Frame.CAMERA, to_frame=Frame.TOOL)
    return CalibrationResult(T_cam_to_base=None, T_cam_to_tool=transform.to_matrix(), rmse_mm=1.1, max_error_mm=2.0,
                             num_samples=11, dataset_path="dataset.json", transform=transform,
                             mode=MountingMode.EYE_IN_HAND, **extra)


class TheNounAimsAWristSweepTests(unittest.TestCase):

    def _doubles(self) -> Any:
        from tests.test_hand_eye_calibration import _Doubles

        doubles = _Doubles()
        doubles.setUp()
        self.addCleanup(doubles.doCleanups)
        return doubles

    def _noun(self, *, mode: str = "eye_in_hand", rig_id: str = "wrist", **options: Any) -> Any:
        from tests.test_hand_eye_calibration import _no_geometry

        self.doubles = self._doubles()
        out = str(self.enterContext(tempfile.TemporaryDirectory()))
        return self.doubles._noun(mode=mode, rig_id=rig_id, robot_config=_no_geometry(),  # noqa: SLF001
                                  options=SweepOptions(out_dir=out, **options))

    def test_the_check_says_what_the_sweep_will_aim_at(self) -> None:
        check = self._noun(aim=_aim(), target="aruco:50:150:DICT_4X4_100").check()
        self.assertTrue(check.ok, check.refusal)
        rendered = check.render()
        self.assertIn("  marker     150.0 mm, id 50, DICT_4X4_100", rendered)
        self.assertIn(f"  poses      {len(DEFAULT_VIEWS)}, aimed at the marker at (-130.0, -700.0, 50.0) mm from "
                      "500 mm", rendered)
        self.assertEqual(check.to_dict()["aim"], _aim().line())

    def test_what_the_check_refuses_about_an_aim(self) -> None:
        cases = {
            "a fixed camera": (dict(mode="eye_to_hand", rig_id="overhead"), {}, "is eye_to_hand"),
            "fixed poses too": ({}, dict(fixed_poses=[LOOK]), "both name the stations"),
            "a board": ({}, dict(target="charuco:7x5:30:22:DICT_5X5_100"), "whose pose is at its corner"),
        }
        for name, (noun, options, said) in cases.items():
            with self.subTest(name):
                check = self._noun(**noun, aim=_aim(), **options).check()
                self.assertFalse(check.ok)
                self.assertIn(said, check.refusal)

    def test_the_sweep_runs_the_aim_with_the_arms_reach_and_the_gates_margin(self) -> None:
        estimates = (_estimate(ALONG), _estimate(SIDE))
        run_aimed = MagicMock(return_value=_eih_result(aim_estimates=estimates))
        with patch.object(CalibrationRoutine, "run_aimed", run_aimed), \
                patch.object(CalibrationRoutine, "run_auto", MagicMock(side_effect=AssertionError("run_auto ran"))):
            report = self._noun(aim=_aim(), target="aruco:50:150:DICT_4X4_100").run()
        self.assertIs(report.outcome, CalibrationOutcome.WRITTEN, report.render())
        self.assertEqual(run_aimed.call_args.args, (_aim(),))
        keywords = run_aimed.call_args.kwargs
        self.assertIsNone(keywords["reach"], "a dummy arm names no UR model to solve on")
        self.assertEqual(keywords["margin_mm"], 20.0)
        self.assertTrue(keywords["dataset_save_path"].endswith("eye_in_hand_wrist_dataset.json"))
        rendered = report.render()
        self.assertIn("  aimed from        camera at", rendered)
        self.assertIn("  re-aimed from     camera at", rendered)
        self.assertEqual(len(report.to_dict()["aim_estimates"]), 2)

    def test_a_first_look_that_saw_nothing_is_reported_once_the_arm_is_down(self) -> None:
        refused = NotAimed("the first look, from where the arm stands, saw no marker (none in view), so no station "
                           "could be aimed and nothing moved. Jog the arm")
        with patch.object(CalibrationRoutine, "run_aimed", MagicMock(side_effect=refused)):
            report = self._noun(aim=_aim(), target="aruco:50:150:DICT_4X4_100").run()
        self.assertIs(report.outcome, CalibrationOutcome.SWEEP_FAILED)
        self.assertEqual(report.exit_code, 3)
        self.assertTrue(report.failure.startswith("NotAimed: the first look"))
        assert report.teardown is not None
        self.assertTrue(report.teardown.clean)
        self.assertFalse(self.doubles.camera.is_open)


class TheCliTakesAnAimTests(unittest.TestCase):

    def _main(self, *argv: str) -> tuple[int, str]:
        from src.robot.execution.real_cell import calibrate
        from tests.test_camera_boundaries import _rgbd_rig
        from tests.test_hand_eye_calibration import _app, _no_geometry

        tree = _app(_rgbd_rig("overhead"), _rgbd_rig("wrist"), robot=_no_geometry())
        printed = io.StringIO()
        with patch.object(calibrate, "_load", return_value=tree), redirect_stdout(printed):
            code = calibrate.main(list(argv))
        return code, printed.getvalue()

    def test_the_aim_reaches_the_check(self) -> None:
        code, printed = self._main("--rig", "wrist", "--mode", "eye_in_hand", "--check", "--aim-at=-130,-700,50",
                                   "--aim-distance-mm", "450", "--closing-axis=-y",
                                   "--board", "aruco:50:150:DICT_4X4_100")
        self.assertEqual(code, 0, printed)
        self.assertIn("aimed at the marker at (-130.0, -700.0, 50.0) mm from 450 mm", printed)

    def test_what_the_cli_refuses_about_an_aim(self) -> None:
        cases = {
            "two numbers": (("--aim-at=1,2",), "--aim-at is the marker's centre"),
            "a shape without an aim": (("--aim-distance-mm", "400"), "there is none without --aim-at"),
            "a fixed camera": (("--aim-at=-130,-700,50",), "is eye_to_hand"),
            "too near": (("--aim-at=-130,-700,50", "--aim-distance-mm", "50"), "distance_mm"),
        }
        for name, (argv, said) in cases.items():
            with self.subTest(name):
                code, printed = self._main("--rig", "overhead", "--check", *argv)
                self.assertEqual(code, 1, printed)
                self.assertIn(said, printed)


class TheCheckLooksUpThePlannersEvidenceTests(unittest.TestCase):
    """Audit 3, commissioning: ``--check`` said "usable" for a cell whose sweep planner then refused to start, after
    the arm connected, for want of an evidence file. The check now makes the lookup the planner start makes."""

    def _robot(self, layer: str) -> Any:
        import shutil

        from src.config.loader import load_robot_config, reload_config

        root = Path(self.enterContext(tempfile.TemporaryDirectory())) / "data"
        shutil.copytree(_ROOT / "config", root)
        (root / "robot" / "robot.trial.yaml").write_text(layer, encoding="utf-8")
        reload_config()
        self.addCleanup(reload_config)
        self.root = root
        return load_robot_config(root, profile="ur10,hande,trial")

    def _check(self, robot: Any) -> Any:
        from src.robot.execution.hand_eye import HandEyeCalibration
        from tests.test_camera_boundaries import _rgbd_rig
        from tests.test_hand_eye_calibration import _app

        return HandEyeCalibration.from_config(_app(_rgbd_rig("overhead"), robot=robot), rig_id="overhead",
                                              mode="eye_to_hand", data_dir=self.root).check()

    _MARGIN = "\n".join(("  safety:", "    self_collision:", "      planner_margin_mm: 4.0", ""))

    def test_a_combination_nobody_measured_is_refused_at_the_desk(self) -> None:
        """An I/O coupling measured at 16 mm: no committed file measured that plate (red before: the check said ok)."""
        plate = "\n".join(("robot:", "  gripper:", "    coupling_plates:", "      - name: io_coupling",
                           "        thickness_mm: 16.0", ""))
        robot = self._robot(plate + self._MARGIN)
        check = self._check(robot)
        self.assertFalse(check.ok)
        for part in ("the sweep's planner would refuse to start", "after the arm is connected",
                     "ur10_robotiq_hande_c16mm", "matrix_gate.py"):
            self.assertIn(part, check.refusal)

    def test_a_measured_combination_passes(self) -> None:
        """⭐ THE CONTROL: the shipped 20 mm plate at 4 mm, which a committed file measured."""
        check = self._check(self._robot("robot:" + "\n" + self._MARGIN))
        self.assertTrue(check.ok, check.refusal)

    def test_an_undeclared_margin_is_left_to_the_desk_check(self) -> None:
        from src.robot.execution.hand_eye import HandEyeCalibration
        from tests.test_hand_eye_calibration import _app

        self.assertTrue(HandEyeCalibration.from_config(_app(), rig_id="overhead", mode="eye_to_hand").check().ok)


class ExampleTenIsTheOwnersCalibrationTests(unittest.TestCase):

    def _source(self) -> str:
        (path,) = sorted((_ROOT / "examples" / "real_robot").glob("10_*.py"))
        return path.read_text(encoding="utf-8")

    def test_it_aims_at_the_owners_marker_and_names_the_target_in_full(self) -> None:
        import ast

        source = self._source()
        constants = {node.targets[0].id: ast.literal_eval(node.value)
                     for node in ast.parse(source).body
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id.isupper()}
        self.assertEqual(constants["MARKER_MM"], (-130.0, -700.0, 50.0))
        self.assertEqual(constants["TARGET"], "aruco:50:150:DICT_4X4_100")
        self.assertEqual((constants["DISTANCE_MM"], constants["CLOSING_AXIS"]), (500.0, "-y"))
        self.assertIn("aim=aim", source)
        self.assertIn('preview="auto"', source)
        self.assertLessEqual(len(source.splitlines()), 70)
        MarkerAim(marker_mm=constants["MARKER_MM"], distance_mm=constants["DISTANCE_MM"],
                  closing_axis=constants["CLOSING_AXIS"])

    def test_its_target_is_one_the_schema_takes(self) -> None:
        from pydantic import TypeAdapter

        from src.calibration.targets import parse_board_spec
        from src.config.schema.camera.cam_schema import CalibrationTargetConfig

        target = TypeAdapter(CalibrationTargetConfig).validate_python(parse_board_spec("aruco:50:150:DICT_4X4_100"))
        self.assertEqual((target.kind, target.marker_id, target.marker_length_mm, target.aruco_dict_name),
                         ("aruco", 50, 150.0, "DICT_4X4_100"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
