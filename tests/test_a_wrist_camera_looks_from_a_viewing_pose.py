"""A wrist camera locates from a viewing pose, not from wherever the arm happens to stand.

The owner-cell audit of 2026-09-23 found that examples 11 and 13 locate from where the arm stands: after a connect,
where the last program left it, and in a campaign, the retreat 100 mm above the last grasp, which puts the owner's
D415 (tilted 45 degrees on the wrist) inside the 450 mm it measures nothing nearer than, turned by the last grasp's
yaw. A fixed camera is not affected. ``camera_aim.viewing_pose`` is the tool pose that puts the CALIBRATED camera a
set distance from a work point, looking straight at it, the tool's heading kept, screened as a calibration station is
screened (the box less its margin, the arm's reach, and the joint window half a turn about home keeps the cable in);
11 and 13 move there before they locate, and 18 before every scan.

Held here:

* the geometry: the camera the calibration places stands the distance asked from the work and looks straight at it,
  and the tool only tilts from ``Pose.tool_down`` with the closing axis asked;
* a fixed camera needs none, and a camera with no calibration is refused as it is everywhere;
* the depth: a distance nearer than the D415 measures at the rig's depth mode is refused, 450 mm at 1280 x 720 and
  310 mm at 848 x 480;
* the screen: the joints reported lie inside the arm's window and put its TCP on the pose, and a view nothing admits
  is refused with its reason;
* the heading: one that admits no view gives way to the one of -y, y, x, -x that admits the most, and the text says
  so; where none admits one, the refusal names each;
* the examples, run against doubles of the cell: the arm moves to the viewing pose before anything is located or
  picked, and a wrist camera with no viewing pose, or a move the arm refused, locates and picks nothing.
"""

from __future__ import annotations

import io
import math
import os
import runpy
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

import willy
from src.calibration.rig_calibration import RigCalibration, RigNotCalibrated
from src.config.schema.robot import WorkspaceLimitsConfig
from src.geometry import Frame, Pose, Transform
from src.robot.execution import camera_aim as aim_module
from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome
from src.robot.execution.camera_aim import MarkerAim, MountEstimate, ViewingPose, aimed_stations, viewing_pose
from src.robot.safety._ur_ik import nearest_goals, ur_flange_ik
from src.robot.safety._ur_kinematics import ur_link_transforms_mm

_ROOT = Path(__file__).resolve().parents[1]
_REAL = _ROOT / "examples" / "real_robot"
#: Where the owner's parts lie: the table under the calibration marker, BASE mm.
WORK = (-130.0, -700.0, 50.0)
#: The TCP 175 mm past the flange along the tool's +Z.
FLANGE_TO_TCP = np.eye(4)
FLANGE_TO_TCP[2, 3] = 175.0
#: The UR10 profile's box and the gate's default margin.
UR10_BOX = WorkspaceLimitsConfig(x_min=-760.7, x_max=760.7, y_min=-760.7, y_max=760.7, z_min=100.0, z_max=748.4)
LOOK = Pose.tool_down(-100.0, -300.0, 280.0, closing_axis="-y", yaw_deg=10.0)


def _mount(offset_mm: Any = (60.0, 0.0, -130.0), tilt_deg: float = 45.0,
           about: Any = (0.0, 1.0, 0.0)) -> np.ndarray:
    """A camera tilted ``tilt_deg`` about the tool axis ``about``, standing at ``offset_mm`` in the tool: CAMERA->TOOL."""
    out = np.eye(4)
    out[:3, :3] = aim_module._exp(np.asarray(about, dtype=np.float64) * math.radians(tilt_deg))
    out[:3, 3] = offset_mm
    return out


#: The owner's shape: tilted 45 degrees toward the tool's +X, on a bracket out that way, behind the TCP.
OWNER = _mount()


def _calibration(mount: np.ndarray = OWNER, mode: Any = "eye_in_hand") -> RigCalibration:
    to_frame = Frame.TOOL if mode == "eye_in_hand" else Frame.BASE
    return RigCalibration(rig_id="wrist", mounting_mode=mode, artifact_path="calibration/real/eih_wrist.json",
                          transform=Transform.from_matrix(mount, from_frame=Frame.CAMERA, to_frame=to_frame))


class _Camera:
    """A camera owner as ``viewing_pose`` reads one: its calibration and its rig."""

    def __init__(self, calibration: Any = None, *, body: str | None = "realsense_d415",
                 depth_resolution: tuple[int, int] = (1280, 720)) -> None:
        self._calibration = calibration if calibration is not None else _calibration()
        self.rig_id = "wrist"
        self.rig = SimpleNamespace(rig_id="wrist", depth_resolution=depth_resolution,
                                   body=None if body is None else SimpleNamespace(model=body))

    def calibration(self) -> Any:
        if isinstance(self._calibration, Exception):
            raise self._calibration
        return self._calibration

    # A camera owner is also a context manager, as example 11 opens it.
    def __enter__(self) -> "_Camera":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _home() -> tuple[float, ...]:
    solutions = ur_flange_ik("ur10", LOOK.to_matrix() @ np.linalg.inv(FLANGE_TO_TCP))
    assert solutions
    return tuple(min(solutions, key=lambda joints: abs(joints[0] - math.radians(-80.0)) - (joints[2] > 0)))


HOME = _home()


def _window(home: Any = HOME, half: float = math.pi - math.radians(5.0)) -> tuple[tuple[float, ...], ...]:
    return tuple(q - half for q in home), tuple(q + half for q in home)


def _arm(*, window: Any = None, box: Any = UR10_BOX, joints: Any = HOME, margin: float = 20.0) -> SimpleNamespace:
    """A connected UR10 as ``reach_of`` and the screen read one: model, box, margin, tool frame, window, joints."""
    low, high = window if window is not None else _window()

    def read_joints() -> list[float]:
        if isinstance(joints, Exception):
            raise joints
        return list(joints)

    return SimpleNamespace(
        config=SimpleNamespace(ur=SimpleNamespace(model="UR10"), workspace_limits=box,
                               safety=SimpleNamespace(limits=SimpleNamespace(workspace_margin_mm=margin))),
        active_tool_frame=FLANGE_TO_TCP, _goal_joint_window=lambda: (low, high),
        get_joint_positions=read_joints, get_tcp_pose=lambda: LOOK)


def _camera_at(pose: Pose, mount: np.ndarray = OWNER) -> np.ndarray:
    return pose.to_matrix() @ mount


def _miss_deg(pose: Pose, mount: np.ndarray = OWNER) -> float:
    camera = _camera_at(pose, mount)
    return aim_module._angle_deg(camera[:3, 2], np.asarray(WORK) - camera[:3, 3])


# ---------------------------------------------------------------------------------------------------------------------
# The geometry
# ---------------------------------------------------------------------------------------------------------------------


class TheCalibratedCameraLooksAtTheWorkTests(unittest.TestCase):

    def test_the_camera_stands_the_distance_asked_and_looks_straight_at_the_work(self) -> None:
        view = viewing_pose(_calibration(), WORK, distance_mm=500.0, closing_axis="-y")
        assert view.pose is not None
        camera = _camera_at(view.pose)
        self.assertAlmostEqual(float(np.linalg.norm(np.asarray(WORK) - camera[:3, 3])), 500.0, places=6)
        self.assertLess(_miss_deg(view.pose), 1e-6)
        self.assertTrue(view.ok and view.needed)
        self.assertEqual(view.pose.frame, Frame.BASE)
        self.assertEqual(view.pose.label, "viewing_pose")

    def test_the_tool_keeps_its_heading_and_only_tilts(self) -> None:
        # The owner's camera looks 45 deg down with the tool down, so the view it has there is the viewing pose:
        # the tool straight down, its +X (the way the jaws close) along base -Y, as Pose.tool_down(..., "-y") has it.
        view = viewing_pose(_calibration(), WORK, closing_axis="-y")
        assert view.pose is not None
        tool = view.pose.to_matrix()
        np.testing.assert_allclose(tool[:3, 0], [0.0, -1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(tool[:3, 2], [0.0, 0.0, -1.0], atol=1e-9)
        self.assertEqual((view.view, round(view.tilt_deg, 6)), ((0.0, 0.0), 0.0))
        # A camera that looks straight down with the tool down has to tilt the tool to look at the work from its
        # distance; the turn is a tilt only, with no twist about the tool's own axis.
        down = viewing_pose(_calibration(_mount(tilt_deg=0.0)), WORK, closing_axis="-y")
        assert down.pose is not None
        self.assertLess(_miss_deg(down.pose, _mount(tilt_deg=0.0)), 1e-6)
        tool = down.pose.to_matrix()[:3, :3]
        reference = Pose.tool_down(*(float(v) for v in down.pose.position_mm), closing_axis="-y").to_matrix()[:3, :3]
        turn_deg = math.degrees(float(np.linalg.norm(aim_module._log(reference.T @ tool))))
        self.assertGreater(turn_deg, 1.0)
        self.assertAlmostEqual(turn_deg, aim_module._angle_deg(tool[:, 2], reference[:, 2]), places=6)
        self.assertAlmostEqual(turn_deg, down.tilt_deg, places=6)

    def test_the_pose_is_aimed_from_the_calibration_and_not_from_a_guess(self) -> None:
        side = _mount(offset_mm=(0.0, 80.0, -150.0), tilt_deg=30.0, about=(-1.0, 0.0, 0.0))
        for mount in (OWNER, side):
            with self.subTest(position=tuple(mount[:3, 3])):
                view = viewing_pose(_calibration(mount), WORK, closing_axis="-y")
                assert view.pose is not None
                self.assertLess(_miss_deg(view.pose, mount), 1e-6)
                self.assertAlmostEqual(float(np.linalg.norm(np.asarray(WORK) - _camera_at(view.pose, mount)[:3, 3])),
                                       500.0, places=6)

    def test_a_fixed_camera_needs_none(self) -> None:
        view = viewing_pose(_calibration(np.eye(4), mode="eye_to_hand"), WORK, arm=_arm())
        self.assertEqual((view.needed, view.ok, view.pose), (False, True, None))
        self.assertIn("NOT NEEDED", str(view))
        self.assertIn("fixed camera", str(view))

    def test_a_camera_with_no_calibration_is_refused_as_everywhere(self) -> None:
        with self.assertRaises(RigNotCalibrated):
            viewing_pose(_Camera(RigNotCalibrated("rig 'wrist' declares no calibration")), WORK)
        with self.assertRaises(TypeError):
            viewing_pose(SimpleNamespace(mounting_mode=None), WORK)

    def test_what_the_aim_refuses_is_refused_here(self) -> None:
        for keywords in ({"distance_mm": 100.0}, {"closing_axis": "z"}, {"views": ()}, {"nearest_mm": 0.0}):
            with self.subTest(**{k: str(v) for k, v in keywords.items()}), self.assertRaises(ValueError):
                viewing_pose(_calibration(), WORK, **keywords)


# ---------------------------------------------------------------------------------------------------------------------
# The depth
# ---------------------------------------------------------------------------------------------------------------------


class TheDistanceIsOneTheDepthMeasuresTests(unittest.TestCase):

    def test_a_d415_at_1280_by_720_measures_nothing_nearer_than_450_mm(self) -> None:
        near = viewing_pose(_Camera(), WORK, distance_mm=400.0, closing_axis="-y")
        self.assertEqual((near.ok, near.pose, near.reason, near.nearest_mm), (False, None, "too_near", 450.0))
        self.assertIn("450 mm", near.detail)
        self.assertIn("[848, 480]", near.detail)
        self.assertIn("REFUSED (too_near)", str(near))
        far = viewing_pose(_Camera(), WORK, distance_mm=500.0, closing_axis="-y")
        self.assertTrue(far.ok)
        self.assertIn("measured from 450 mm on (Intel's figure for a D415 at a 1280 x 720 depth mode", str(far))

    def test_at_848_by_480_it_measures_from_310_mm(self) -> None:
        camera = _Camera(depth_resolution=(848, 480))
        self.assertTrue(viewing_pose(camera, WORK, distance_mm=400.0, closing_axis="-y").ok)
        refused = viewing_pose(camera, WORK, distance_mm=300.0, closing_axis="-y")
        self.assertEqual((refused.reason, refused.nearest_mm), ("too_near", 310.0))

    def test_a_stated_depth_wins_and_an_unknown_one_is_not_checked(self) -> None:
        stated = viewing_pose(_Camera(), WORK, distance_mm=400.0, nearest_mm=350.0, closing_axis="-y")
        self.assertTrue(stated.ok)
        self.assertEqual(stated.depth_note, "as stated by the caller")
        unknown = viewing_pose(_Camera(body=None), WORK, distance_mm=300.0, closing_axis="-y")
        self.assertTrue(unknown.ok)
        self.assertIsNone(unknown.nearest_mm)
        self.assertIn("not known here", str(unknown))
        other = viewing_pose(_Camera(body="orbbec_femto"), WORK, distance_mm=300.0, closing_axis="-y")
        self.assertIsNone(other.nearest_mm)


# ---------------------------------------------------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------------------------------------------------


class TheViewingPoseIsScreenedAsAStationIsTests(unittest.TestCase):

    def test_its_joints_lie_inside_the_window_and_put_the_tcp_on_the_pose(self) -> None:
        low, high = _window()
        view = viewing_pose(_Camera(), WORK, arm=_arm(), closing_axis="-y")
        assert view.pose is not None and view.joints is not None
        for q, lower, upper in zip(view.joints, low, high):
            self.assertTrue(lower <= q <= upper)
        tcp = ur_link_transforms_mm("ur10", np.array(view.joints))[-1] @ FLANGE_TO_TCP
        np.testing.assert_allclose(tcp[:3, 3], view.pose.position_mm, atol=1e-6)
        # The nearest configuration from where the arm stands, as the arm's own nearest-goal choice takes it.
        goal = nearest_goals(ur_flange_ik("ur10", view.pose.to_matrix() @ np.linalg.inv(FLANGE_TO_TCP)),
                             current=HOME, lower=low, upper=high, velocity=1.0)[0]
        np.testing.assert_allclose(view.joints, goal.joints, atol=1e-9)
        self.assertAlmostEqual(view.largest_rad, goal.largest_rad)
        text = str(view)
        self.assertIn("the workspace box less 20 mm (TCP and flange), the reach of a ur10 and the joint window", text)
        self.assertIn("plans and judges the move there", text)

    def test_nothing_the_window_admits_is_refused_with_its_reason(self) -> None:
        tight = (tuple(q - 0.02 for q in HOME), tuple(q + 0.02 for q in HOME))
        view = viewing_pose(_Camera(), WORK, arm=_arm(window=tight), closing_axis="-y")
        self.assertEqual((view.ok, view.pose, view.reason), (False, None, "outside_joint_window"))
        self.assertIn("half a turn either side of home", view.detail)
        self.assertIn("nor is any of the 10 views turned from it (outside_joint_window)", view.detail)
        self.assertIn("Nothing should be located", str(view))

    def test_a_view_the_box_refuses_gives_way_to_the_least_tilted_one_it_admits(self) -> None:
        views = ((0.0, 0.0), (15.0, 0.0), (0.0, 20.0))
        unscreened = {station.view: station for station in (
            aimed_stations(MarkerAim(marker_mm=WORK, closing_axis="-y", views=(view,)),
                           MountEstimate(camera_in_tool=aim_module._as_tuple(OWNER), source="test"))[0]
            for view in views)}
        natural, higher = unscreened[(0.0, 0.0)].pose, unscreened[(15.0, 0.0)].pose
        assert natural is not None and higher is not None
        self.assertGreater(float(higher.position_mm[2]), float(natural.position_mm[2]))
        floor = (float(natural.position_mm[2]) + float(higher.position_mm[2])) / 2.0
        box = WorkspaceLimitsConfig(x_min=-760.7, x_max=760.7, y_min=-760.7, y_max=760.7, z_min=floor, z_max=900.0)
        view = viewing_pose(_calibration(), WORK, arm=SimpleNamespace(config=SimpleNamespace(workspace_limits=box)),
                            closing_axis="-y", views=views)
        assert view.pose is not None
        self.assertEqual(view.view, (15.0, 0.0), "the (0, 20) view tilts the tool further")
        self.assertGreater(view.tilt_deg, 1.0)
        self.assertLess(_miss_deg(view.pose), 1e-6)
        self.assertIn("the workspace box less 0 mm (TCP)", view.screened)
        refused = viewing_pose(_calibration(), WORK, arm=SimpleNamespace(config=SimpleNamespace(workspace_limits=box)),
                               closing_axis="-y", views=((0.0, 0.0),))
        self.assertEqual(refused.reason, "outside_workspace")

    def test_without_an_arm_nothing_is_screened_and_it_says_so(self) -> None:
        view = viewing_pose(_calibration(), WORK, closing_axis="-y")
        self.assertIsNone(view.joints)
        self.assertIn("screened against nothing", str(view))

    def test_an_arm_that_cannot_say_its_joints_is_still_screened(self) -> None:
        view = viewing_pose(_calibration(), WORK, arm=_arm(joints=RuntimeError("no link")), closing_axis="-y")
        self.assertTrue(view.ok)
        self.assertIsNone(view.joints)
        self.assertIn("joint window", view.screened)

    def test_the_door_exports_it(self) -> None:
        self.assertIs(willy.viewing_pose, viewing_pose)
        self.assertIs(willy.ViewingPose, ViewingPose)


class AHeadingThatAdmitsNoViewGivesWayTests(unittest.TestCase):
    """Review 4 (2026-09-23): 11, 13 and 18 keep closing_axis "-y", written for a camera tilted toward the tool's +X.
    For one tilted toward -X that heading stands the camera beyond the marker, outside the box: before, REFUSED
    (outside_workspace), the TCP at y -1124, and nothing was located. The heading of -y, y, x, -x that admits the most
    views is kept instead, and the text says so."""

    AGAINST = _mount(offset_mm=(-60.0, 0.0, -130.0), tilt_deg=-45.0)

    def test_the_heading_that_admits_the_most_is_kept_and_said(self) -> None:
        wide = _window(HOME, math.radians(350.0))
        with self.assertLogs("src.robot.execution.camera_aim", level="WARNING") as said:
            view = viewing_pose(_Camera(_calibration(self.AGAINST)), WORK, arm=_arm(window=wide), closing_axis="-y")
        assert view.pose is not None and view.heading is not None
        self.assertEqual((view.closing_axis, view.heading.asked, view.heading.changed), ("y", "-y", True))
        self.assertEqual(dict(view.heading.admitted)["-y"], 0)
        self.assertLess(_miss_deg(view.pose, self.AGAINST), 1e-6)
        tool = view.pose.to_matrix()
        np.testing.assert_allclose(tool[:3, 0], [0.0, 1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(tool[:3, 2], [0.0, 0.0, -1.0], atol=1e-9)
        self.assertIn("  heading: 'y', not the '-y' asked: '-y' admits 0 of the 11 views", str(view))
        self.assertIn("not the '-y' asked", "\n".join(said.output))

    def test_the_heading_asked_is_kept_where_it_admits_a_view(self) -> None:
        view = viewing_pose(_Camera(), WORK, arm=_arm(), closing_axis="-y")
        assert view.heading is not None
        self.assertEqual((view.closing_axis, view.heading.changed), ("-y", False))
        self.assertEqual([heading for heading, _ in view.heading.admitted], ["-y"], "nothing else was tried")
        self.assertNotIn("heading:", str(view))

    def test_where_no_heading_admits_a_view_the_refusal_names_each(self) -> None:
        tight = (tuple(q - 0.02 for q in HOME), tuple(q + 0.02 for q in HOME))
        view = viewing_pose(_Camera(), WORK, arm=_arm(window=tight), closing_axis="-y")
        self.assertEqual((view.ok, view.reason, view.closing_axis), (False, "outside_joint_window", "-y"))
        self.assertIn("and no other heading admits one either (y 0, x 0, -x 0 of the 11 views)", view.detail)


# ---------------------------------------------------------------------------------------------------------------------
# The examples, against doubles of the cell
# ---------------------------------------------------------------------------------------------------------------------


class _Report(SimpleNamespace):
    def __str__(self) -> str:
        return f"{self.what} {'ok' if self.ok else 'REFUSED'}"


class _Cell:
    """What 11, 13 and 18 drive, recorded in order: the arm's moves, the hand's releases, locates and picks."""

    def __init__(self, calls: list[Any], *, move_ok: bool = True, outcomes: Any = ()) -> None:
        self.calls = calls
        self.move_ok = move_ok
        self.outcomes = list(outcomes)
        self.arm = _arm()
        self.arm.get_tcp_pose = lambda: LOOK
        self.teardown = None
        self.service = self
        self.robot = self
        self.last_debug_image_png = None

    # The robot's verbs.
    def move(self, pose: Pose, **_: Any) -> _Report:
        self.calls.append(("move", pose))
        return _Report(what="move", ok=self.move_ok)

    def release(self) -> _Report:
        self.calls.append(("release",))
        return _Report(what="release", ok=True)

    def pick(self, *args: Any, **_: Any) -> Any:
        self.calls.append(("pick",))
        outcome = self.outcomes.pop(0) if self.outcomes else AutonomousGraspOutcome.SUCCEEDED
        return SimpleNamespace(outcome=outcome, fault=None, controller_stopped=False, hold_measured=False,
                               failure_summary=lambda: "", layers_that_ran=lambda: ())

    def connected(self, **_: Any) -> "_Cell":
        return self

    def __enter__(self) -> "_Cell":
        self.calls.append(("connect",))
        return self

    def __exit__(self, *exc: Any) -> None:
        self.calls.append(("teardown",))

    # The cell's and the service's.
    def build(self) -> "_Cell":
        return self

    def preflight(self) -> str:
        return "preflight"

    def enable_debug_image_rendering(self, on: bool) -> None:
        return None

    def enable_record_logging(self, *_: Any, **__: Any) -> None:
        return None

    def set_prompt(self, prompt: str) -> str:
        return "the prompt before"

    def set_target_label(self, label: Any) -> None:
        return None


class _Tree(SimpleNamespace):
    def with_values(self, _values: Any) -> "_Tree":
        return self


def _run(example: str, cell: _Cell, camera: _Camera) -> str:
    """Run ``example`` against ``cell`` and ``camera`` in a scratch directory; what it printed."""
    (path,) = sorted(_REAL.glob(f"{example}_*.py"))
    tree = _Tree(robot=SimpleNamespace())
    located = SimpleNamespace(objects=[])

    def locate(prompt: str) -> Any:
        cell.calls.append(("locate",))
        return located

    doubles = {
        "load_tree": lambda *a, **k: tree,
        "Camera": SimpleNamespace(from_tree=lambda *a, **k: camera),
        "Robot": SimpleNamespace(from_tree=lambda *a, **k: cell),
        "Locator": SimpleNamespace(from_tree=lambda *a, **k: SimpleNamespace(locate=locate)),
        "Cell": SimpleNamespace(from_tree=lambda *a, **k: cell),
    }
    out = io.StringIO()
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)
        try:
            with patch.multiple(willy, create=True, **doubles), redirect_stdout(out):
                try:
                    runpy.run_path(str(path), run_name="__main__")
                except SystemExit:
                    pass
        finally:
            os.chdir(here)
    return out.getvalue()


def _names(calls: list[Any]) -> list[str]:
    return [call[0] for call in calls]


class ElevenMovesToLookBeforeItLocatesTests(unittest.TestCase):

    def test_the_arm_moves_to_the_viewing_pose_and_then_locates(self) -> None:
        calls: list[Any] = []
        printed = _run("11", _Cell(calls), _Camera())
        # The hand opens first, where the last run left the arm, so a part it ended holding does not ride along.
        self.assertEqual(_names(calls), ["connect", "release", "move", "locate", "teardown"])
        expected = viewing_pose(_Camera(), WORK, arm=_arm(), closing_axis="-y")
        assert expected.pose is not None
        np.testing.assert_allclose(calls[2][1].to_matrix(), expected.pose.to_matrix(), atol=1e-9)
        self.assertIn("viewing pose  FOUND", printed)

    def test_a_wrist_camera_with_no_viewing_pose_locates_nothing(self) -> None:
        calls: list[Any] = []
        cell = _Cell(calls)
        cell.arm = _arm(window=(tuple(q - 0.02 for q in HOME), tuple(q + 0.02 for q in HOME)))
        printed = _run("11", cell, _Camera())
        self.assertEqual(_names(calls), ["connect", "release", "teardown"])
        self.assertIn("REFUSED (outside_joint_window)", printed)
        self.assertIn("nothing located", printed)

    def test_a_move_the_arm_refused_locates_nothing(self) -> None:
        calls: list[Any] = []
        _run("11", _Cell(calls, move_ok=False), _Camera())
        self.assertEqual(_names(calls), ["connect", "release", "move", "teardown"])

    def test_a_fixed_camera_locates_from_where_it_is(self) -> None:
        calls: list[Any] = []
        printed = _run("11", _Cell(calls), _Camera(_calibration(np.eye(4), mode="eye_to_hand")))
        self.assertEqual(_names(calls), ["connect", "release", "locate", "teardown"])
        self.assertIn("NOT NEEDED", printed)


class ThirteenLooksBeforeEveryPickTests(unittest.TestCase):

    def test_every_pick_starts_from_the_viewing_pose_with_the_hand_emptied(self) -> None:
        calls: list[Any] = []
        _run("13", _Cell(calls), _Camera())
        self.assertEqual(_names(calls), ["connect"] + ["release", "move", "pick"] * 5 + ["teardown"])

    def test_a_wrist_camera_with_no_viewing_pose_picks_nothing(self) -> None:
        calls: list[Any] = []
        cell = _Cell(calls)
        cell.arm = _arm(window=(tuple(q - 0.02 for q in HOME), tuple(q + 0.02 for q in HOME)))
        printed = _run("13", cell, _Camera())
        self.assertEqual(_names(calls), ["connect", "teardown"])
        self.assertIn("5 attempt(s) never ran", printed)

    def test_a_refused_move_stops_the_campaign_before_its_pick(self) -> None:
        calls: list[Any] = []
        _run("13", _Cell(calls, move_ok=False), _Camera())
        self.assertEqual(_names(calls), ["connect", "release", "move", "teardown"])

    def test_a_fixed_camera_picks_from_where_the_arm_is(self) -> None:
        calls: list[Any] = []
        _run("13", _Cell(calls), _Camera(_calibration(np.eye(4), mode="eye_to_hand")))
        self.assertEqual(_names(calls), ["connect"] + ["pick"] * 5 + ["teardown"])


class EighteenScansFromTheViewingPoseTests(unittest.TestCase):

    def test_every_scan_and_the_lid_start_from_the_viewing_pose(self) -> None:
        calls: list[Any] = []
        outcomes = [AutonomousGraspOutcome.SUCCEEDED, AutonomousGraspOutcome.NO_TARGET]
        _run("18", _Cell(calls, outcomes=outcomes), _Camera())
        self.assertEqual(_names(calls), ["connect"] + ["release", "move", "pick"] * 3 + ["teardown"])

    def test_a_wrist_camera_with_no_viewing_pose_scans_nothing(self) -> None:
        calls: list[Any] = []
        cell = _Cell(calls)
        cell.arm = _arm(window=(tuple(q - 0.02 for q in HOME), tuple(q + 0.02 for q in HOME)))
        printed = _run("18", cell, _Camera())
        self.assertEqual(_names(calls), ["connect", "teardown"])
        self.assertIn("nothing more is picked", printed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
