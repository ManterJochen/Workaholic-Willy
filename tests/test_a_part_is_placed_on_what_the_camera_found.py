"""A part is set down on what a camera found: measured, never written, and released only once it is there.

The owner, 2026-09-24 (example 13): pick an object and place it on a target, both found by the camera from two prompts,
with no coordinate in the program. The part is set down so that it lands with 5 mm of air on the target's top, and only
then released. Three things carry that, and each is held here:

* ``Located.set_down(i, grasp=, part_bottom_mm=)`` (``SetDown.onto``) measures where the tool goes. The target's top is
  the 95th percentile of its surface in BASE Z, so a stray pixel above it does not lift the part as the maximum would;
  the part hangs below the grasp as far as the grasp stood above the part's bottom; the tool keeps the grasp's own
  turn, over the target's centre, at top + hang + air. A target with no surface to read, and a grasp that did not
  stand above the part, give no pose and say why: nothing is set down blind.
* The part's bottom is the support the cell DECLARES (``scene.declared_support_height_mm``: the table, or the
  container floor), a lower bound on where the part stood, and not the scene's ``support_height_mm``, which is raised
  to the part's lowest SEEN point. What the camera missed of the part's foot (the review of 2026-09-24: 8 mm of the
  near face unseen pressed the part 3 mm into the target, 12 mm pressed it 7 mm in) now goes to more air, and a part
  taken off a raised block drops the block's height further, the stated cost. A caller that knows where the part
  stood passes that.
* ``Locator.for_cameras`` builds one perception backend for every camera of a cell; example 13 locates with it.
* ``Robot.place(keep_out=)`` holds the located target out of the live camera world for the place's motions and its
  release, and for nothing else, as ``Robot.pick`` holds its part (``tests/test_robot_pick_and_place.py`` holds the
  verb; the real world is held here), because the held part comes within the line clearance of what it lands on.
* Example 13 itself runs here on the owner's cell doubles (``tests/test_a_toggle_hand_pulses_only_where_the_owner_
  expects.py``: the real toggle driver on a Hand-E that flips on every rising edge, a UR that logs every motion), with
  its cameras and locators scripted: the first camera that locates a prompt answers it, a wrist camera looks from each
  look in turn and a fixed one does not move, a target nobody found leaves the part held with no pulse, and the release
  pulse comes after the place's line in arrived. The owner's invariants I1 to I5 and R6 are checked on every run.
"""

from __future__ import annotations

import io
import json
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

import willy
from src.geometry import Frame, Pose
from src.robot.core.gripper import HoldEvidence
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.robot import Robot
from src.robot.grasping.scene import Scene
from src.robot.perception.locator import Located, LocatedObject, Locator, LocatorRefused, SetDown
from tests.test_a_scene_plans_the_trees_hand import _cube, _hande, _replace
from tests.test_a_toggle_hand_pulses_only_where_the_owner_expects import (
    _MOTIONS,
    _at,
    _cell,
    _Cell,
    _check,
    _connected,
    _line_before,
    _only,
    _robot,
    _verb,
)
from tests.test_robot_hand_verbs import _Hand

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = _ROOT / "examples" / "real_robot" / "13_pick_and_place_with_the_camera.py"

#: Where the scripted plate stands, BASE mm: its centre and its flat top.
_PLATE_XY = (300.0, 250.0)
_PLATE_TOP_MM = 20.0


def _plate(*, centre: "tuple[float, float]" = _PLATE_XY, top_mm: float = _PLATE_TOP_MM, radius_mm: float = 80.0,
           step_mm: float = 4.0) -> np.ndarray:
    """A plate's flat top as a camera measures it: a disc of points at one height."""
    across = np.arange(-radius_mm, radius_mm + 1e-9, step_mm)
    return np.array([[centre[0] + x, centre[1] + y, top_mm] for x in across for y in across
                     if x * x + y * y <= radius_mm * radius_mm])


def _object(label: str, points: np.ndarray) -> LocatedObject:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    centre = None if points.shape[0] == 0 else tuple(float(v) for v in np.median(points, axis=0))
    return LocatedObject(label=label, score=0.9, box_px=None, mask=mask, points_base_mm=points,
                         centre_mm=centre)  # type: ignore[arg-type]


def _located(camera: str, *objects: LocatedObject, stamp: float = 100.0) -> Located:
    return Located(camera=camera, captured_at_s=stamp, mounting="eye_to_hand", tool_to_base_mm=None,
                   objects=tuple(objects))


def _grasp(z_mm: float, *, yaw_deg: float = 30.0) -> Pose:
    return Pose.tool_down(500.0, -300.0, z_mm, yaw_deg=yaw_deg)


def _cube_seen_down_to(hidden_mm: float, *, base_mm: float = 0.0) -> np.ndarray:
    """The scene test's 40 mm cube with the lowest ``hidden_mm`` of its seen face missing: an occluder in front of it,
    or a mask that stops short of its base. Its lowest seen point stands ``hidden_mm`` over what it stands on."""
    cloud = _cube(40.0, base_mm=base_mm)
    return cloud[cloud[:, 2] >= base_mm + hidden_mm - 1e-9]


def _real_air(set_down: SetDown, grasp: Pose, *, stood_on_mm: float) -> float:
    """The gap the part really leaves over the target's top when the hand opens: the set-down's Z less the part's REAL
    hang (the grasp's Z less what the part really stood on) less the target's top."""
    assert set_down.pose is not None and set_down.top_mm is not None, set_down.render()
    real_hang = float(grasp.position_mm[2]) - stood_on_mm
    return float(set_down.pose.position_mm[2]) - real_hang - float(set_down.top_mm)


# ---------------------------------------------------------------------------------------------------
# The set-down, measured
# ---------------------------------------------------------------------------------------------------


class TheSetDownIsMeasuredTests(unittest.TestCase):
    def test_the_tool_goes_over_the_targets_centre_at_its_top_plus_the_hang_plus_five_mm(self) -> None:
        plate = _located("wrist", _object("blue plate", _plate()))
        grasp = _grasp(26.0)

        set_down = plate.set_down(0, grasp=grasp, part_bottom_mm=0.0)

        assert set_down.pose is not None, set_down.render()
        np.testing.assert_allclose(set_down.pose.position_mm, (*_PLATE_XY, _PLATE_TOP_MM + 26.0 + 5.0), atol=1e-9)
        self.assertEqual(_PLATE_TOP_MM, set_down.top_mm)
        self.assertEqual(26.0, set_down.hang_mm)
        self.assertEqual(5.0, set_down.air_mm, "the owner's 5 mm is the default")
        self.assertTrue(set_down.ok)

    def test_the_tool_keeps_the_grasps_own_turn_tilted_or_not(self) -> None:
        plate = _located("wrist", _object("blue plate", _plate()))
        tilted = Pose.aimed_at(450.0, -300.0, 60.0, target_mm=(500.0, -300.0, 20.0), roll_deg=15.0)
        for grasp in (_grasp(26.0, yaw_deg=-75.0), tilted):
            with self.subTest(grasp=grasp.quaternion_xyzw.tolist()):
                set_down = plate.set_down(0, grasp=grasp, part_bottom_mm=0.0)
                assert set_down.pose is not None, set_down.render()
                np.testing.assert_allclose(set_down.pose.quaternion_xyzw, grasp.quaternion_xyzw, atol=1e-12)
                self.assertIs(Frame.BASE, set_down.pose.frame)

    def test_a_stray_pixel_above_the_target_does_not_lift_the_part(self) -> None:
        """The top is a high percentile of the surface: ten flying pixels 100 mm up move it nowhere; the max would."""
        cloud = _plate()
        flying = np.array([[_PLATE_XY[0] + 10.0 * k, _PLATE_XY[1], 120.0] for k in range(10)])
        set_down = SetDown.onto(_object("blue plate", np.vstack([cloud, flying])), grasp=_grasp(26.0),
                                part_bottom_mm=0.0)
        self.assertEqual(_PLATE_TOP_MM, set_down.top_mm, set_down.render())
        self.assertLess(10 / (cloud.shape[0] + 10), 0.05, "the control: the strays are under 5 % of the surface")

    def test_a_rim_above_the_middle_is_the_top_so_the_part_never_presses_into_it(self) -> None:
        cloud = _plate()
        radial = np.hypot(cloud[:, 0] - _PLATE_XY[0], cloud[:, 1] - _PLATE_XY[1])
        cloud[radial > 60.0, 2] = _PLATE_TOP_MM + 12.0
        set_down = SetDown.onto(_object("blue plate", cloud), grasp=_grasp(26.0), part_bottom_mm=0.0)
        assert set_down.pose is not None and set_down.top_mm is not None
        self.assertEqual(_PLATE_TOP_MM + 12.0, set_down.top_mm)
        self.assertEqual(_PLATE_TOP_MM + 12.0 + 26.0 + 5.0, float(set_down.pose.position_mm[2]))

    def test_a_box_seen_from_the_side_is_centred_on_its_top_and_not_where_the_side_pulls_the_median(self) -> None:
        """A tilted camera sees a box's top and the face toward it; the part goes over the top face's middle.

        MEASURED here on the scene test's cube as a wrist camera tilted 45 degrees sees it: the per-axis median of the
        surface, ``centre_mm``, stands 20 mm toward the seen face from the cube's middle, half the cube, so a part set
        down there would hang over the top's near edge. The set-down stands 1 mm off it: the strip of that face within
        the top band crowds the near edge, and the extent is read between percentiles, not at the extremes.
        """
        box = _cube(40.0)  # the top face at 40 mm and the face toward a camera at -y, centred on (500, -300)
        target = _object("red box", box)
        assert target.centre_mm is not None
        self.assertAlmostEqual(-20.0, target.centre_mm[1] - (-300.0), delta=0.5, msg="the control: the median is pulled")

        set_down = SetDown.onto(target, grasp=Pose.tool_down(100.0, 100.0, 30.0), part_bottom_mm=0.0)

        assert set_down.pose is not None, set_down.render()
        np.testing.assert_allclose(set_down.pose.position_mm[:2], (500.0, -300.0), atol=1.5)
        self.assertEqual(40.0 + 30.0 + 5.0, float(set_down.pose.position_mm[2]))

    def test_a_noisy_top_reads_high_by_its_noise_never_low(self) -> None:
        rng = np.random.default_rng(3)
        cloud = _plate()
        cloud[:, 2] += rng.normal(0.0, 1.0, cloud.shape[0])
        set_down = SetDown.onto(_object("blue plate", cloud), grasp=_grasp(26.0), part_bottom_mm=0.0)
        assert set_down.top_mm is not None
        self.assertGreater(set_down.top_mm, _PLATE_TOP_MM, "a noisy top read low would press the part into it")
        self.assertLess(set_down.top_mm, _PLATE_TOP_MM + 2.5)

    def test_the_air_is_the_callers(self) -> None:
        plate = _located("wrist", _object("blue plate", _plate()))
        for air in (0.0, 12.0):
            with self.subTest(air_mm=air):
                pose = plate.set_down(0, grasp=_grasp(26.0), part_bottom_mm=0.0, air_mm=air).pose
                assert pose is not None
                self.assertEqual(_PLATE_TOP_MM + 26.0 + air, float(pose.position_mm[2]))

    def test_a_caller_who_knows_what_the_part_stood_on_is_honoured(self) -> None:
        """A part taken off a 30 mm fixture whose height the caller knows: the hang from there, exactly 5 mm of air."""
        seen = _located("overhead", _object("red cube", _cube(40.0, base_mm=30.0)))
        best = seen.scene(0, _hande()).grasps().best
        assert best is not None
        plate = _located("wrist", _object("blue plate", _plate()))

        set_down = plate.set_down(0, grasp=best.pose(), part_bottom_mm=30.0)

        self.assertAlmostEqual(float(best.position_mm[2]) - 30.0, set_down.hang_mm, places=9)
        self.assertAlmostEqual(5.0, _real_air(set_down, best.pose(), stood_on_mm=30.0), places=9)


class APartIsNeverPressedIntoTheTargetTests(unittest.TestCase):
    """The owner's rule (2026-09-24): 5 mm of air, released only then, NEVER pressed.

    The hang is the grasp's Z less the part's bottom, so a bottom read too HIGH shortens the hang and brings the part
    down into the target by as much. The scene's ``support_height_mm`` is the declared table raised to the part's lowest
    SEEN point: an upper bound on its base, since a camera never sees below it but may miss its foot. The declared
    support is a lower bound, since no part stands below the table (or the container floor) it is declared on; from it,
    every error goes toward more air. These run the real ``Scene.from_robot_config`` on the owner's Hand-E profile, as
    the reviewer's reproduction did.
    """

    def test_a_part_whose_lowest_rows_the_camera_missed_still_lands_with_five_mm_of_air(self) -> None:
        """The reviewer's cases: 8 and 12 mm of the near face unseen pressed the part 3 and 7 mm into the target."""
        plate = _located("wrist", _object("blue plate", _plate()))
        for hidden in (0.0, 4.0, 8.0, 12.0):
            with self.subTest(hidden_mm=hidden):
                scene = _located("overhead", _object("red cube", _cube_seen_down_to(hidden))).scene(0, _hande())
                best = scene.grasps().best
                assert best is not None, scene.grasps().render()
                self.assertAlmostEqual(hidden, scene.support_height_mm, places=6,
                                       msg="the control: the scene's support is raised to the lowest seen point")
                pressed = plate.set_down(0, grasp=best.pose(), part_bottom_mm=scene.support_height_mm)
                self.assertAlmostEqual(5.0 - hidden, _real_air(pressed, best.pose(), stood_on_mm=0.0), places=6,
                                       msg="the control: from the seen support, the part goes into the target")

                set_down = plate.set_down(0, grasp=best.pose(), part_bottom_mm=scene.declared_support_height_mm)

                self.assertEqual(0.0, scene.declared_support_height_mm)
                self.assertGreaterEqual(_real_air(set_down, best.pose(), stood_on_mm=0.0), 5.0 - 1e-9,
                                        set_down.render())

    def test_a_fully_seen_part_errs_toward_air_and_one_off_a_block_drops_the_blocks_height(self) -> None:
        """The cost, stated: from the declared table a part taken off a 30 mm block hangs 30 mm lower than measured,
        so it comes down with 35 mm of air and drops them. A part on the table itself lands with the 5 mm."""
        plate = _located("wrist", _object("blue plate", _plate()))
        for base in (0.0, 30.0):
            with self.subTest(part_stood_on_mm=base):
                scene = _located("overhead", _object("red cube", _cube(40.0, base_mm=base))).scene(0, _hande())
                best = scene.grasps().best
                assert best is not None
                self.assertAlmostEqual(base, scene.support_height_mm, places=6, msg="the control: seen down to its base")

                set_down = plate.set_down(0, grasp=best.pose(), part_bottom_mm=scene.declared_support_height_mm)

                self.assertAlmostEqual(float(best.position_mm[2]), set_down.hang_mm, places=9)
                self.assertAlmostEqual(5.0 + base, _real_air(set_down, best.pose(), stood_on_mm=base), places=9)

    def test_on_the_owners_tilted_wrist_view_a_hidden_foot_never_shortens_the_hang(self) -> None:
        """The reviewer's second case: the owner's D415 tilted 45 degrees on the wrist, through the locator, the lowest
        6 mm of the cube's near face hidden. The frame is read 3 mm high there, so the part's base reads at +3: the seen
        support stood 4.7 to 6.2 mm above even that, and the declared table stands below the base the arm moves to."""
        import tests.test_a_tilted_view_grasps_on_the_part as tilted

        for seed in range(3):
            with self.subTest(seed=seed):
                located = _tilted_located(tilted, seed=seed, hide_below_mm=6.0)
                scene = located.scene(0, tilted._hande())  # noqa: SLF001
                best = scene.grasps().best
                assert best is not None, scene.grasps().render()
                self.assertGreater(scene.support_height_mm - 3.0, 4.0, "the control: the seen support is raised")
                plate = _located("wrist", _object("blue plate", _plate()))

                set_down = plate.set_down(0, grasp=best.pose(), part_bottom_mm=scene.declared_support_height_mm)

                # The arm moves in the real BASE, where the cube stands on the table at 0.
                self.assertGreaterEqual(_real_air(set_down, best.pose(), stood_on_mm=0.0), 5.0 - 1e-9,
                                        set_down.render())


def _tilted_located(tilted: Any, *, seed: int, hide_below_mm: float) -> Located:
    """The tilted-view test's 45 degree wrist frame, 2 px misregistered, through the real locator, with the cube's
    pixels whose true height is below ``hide_below_mm`` left out of its mask (the reviewer's harness)."""
    mask, depth, believed = tilted._frame(45.0, shift_px=2, seed=seed)  # noqa: SLF001
    camera = tilted._camera_to_base(45.0)  # noqa: SLF001
    rotation, origin = camera[:3, :3], camera[:3, 3]
    u, v = np.meshgrid(np.arange(tilted._W), np.arange(tilted._H))  # noqa: SLF001
    rays = np.stack([(u - tilted._K[0, 2]) / tilted._F, (v - tilted._K[1, 2]) / tilted._F,  # noqa: SLF001
                     np.ones(u.shape)], axis=-1) @ rotation.T
    _, along_axis = tilted._ray_cast(camera)  # noqa: SLF001
    heights = origin[2] + rays[..., 2] * (along_axis / (rays @ rotation[:, 2]))
    mask = mask & (heights >= hide_below_mm)
    tcp = Pose(position_mm=believed[:3, 3].copy(), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=Frame.BASE)
    seen = SimpleNamespace(detection=SimpleNamespace(score=0.9, box=None), segmentation=tilted._Segmentation(  # noqa: SLF001
        label="red cube", score=0.9, bbox_xyxy=(0.0, 0.0, 1.0, 1.0), mask=mask))
    backend = SimpleNamespace(perceive=lambda _bgr, _prompt: (seen,))
    tool_frame = SimpleNamespace(source="willy", offset_mm=(0.0, 0.0, 0.0), rotation_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                                 verify_tolerance_mm=1.0)
    return Locator.from_parts(camera=tilted._WristCamera(depth, believed), backend=backend,  # noqa: SLF001
                              tool_pose=lambda: tcp, tool_frame=tool_frame).locate("a red cube")


class TheSceneSaysTheSupportTheCellDeclaresTests(unittest.TestCase):
    def test_it_is_the_table_or_the_container_floor_however_high_the_part_raised_the_seen_support(self) -> None:
        cube = _cube(40.0, base_mm=30.0)
        cfg = _hande()
        rows = (
            ("the table", cfg, 0.0),
            ("a table declared at 12 mm", _replace(cfg, "grasping.support.height_mm", 12.0), 12.0),
            ("a container floor at 8 mm", _replace(cfg, "grasping.support.container.floor_height_mm", 8.0), 8.0),
            ("no refinement from the target", _replace(cfg, "grasping.support.refine_from_target", False), 0.0),
        )
        for label, config, declared in rows:
            with self.subTest(label):
                scene = Scene.from_robot_config(config, cube)
                self.assertEqual(declared, scene.declared_support_height_mm)
                self.assertLessEqual(scene.declared_support_height_mm, scene.support_height_mm)
                self.assertIsInstance(scene.declared_support_height_mm, float)

    def test_a_bare_clouds_declared_support_is_the_one_its_caller_stated(self) -> None:
        scene = Scene.from_cloud(_cube(40.0, base_mm=30.0), support_height_mm=7.0)
        self.assertEqual(7.0, scene.declared_support_height_mm)
        self.assertEqual(7.0, scene.support_height_mm)

    def test_it_is_read_only(self) -> None:
        scene = Scene.from_robot_config(_hande(), _cube(40.0, base_mm=30.0))
        # A frozen slotted dataclass refuses a write to a name that is not a field with TypeError on CPython 3.11,
        # where a property of a plain class would raise AttributeError; either way nothing is written.
        with self.assertRaises((AttributeError, TypeError)):
            scene.declared_support_height_mm = 30.0  # type: ignore[misc]
        self.assertIsNone(vars(Scene)["declared_support_height_mm"].fset, "a property with no setter")
        self.assertEqual(0.0, scene.declared_support_height_mm)


class WhatCannotBeMeasuredIsNotSetDownTests(unittest.TestCase):
    def test_a_target_with_no_surface_gives_no_pose_and_says_why(self) -> None:
        set_down = SetDown.onto(_object("blue plate", np.zeros((0, 3))), grasp=_grasp(26.0), part_bottom_mm=0.0)
        self.assertIsNone(set_down.pose)
        self.assertFalse(set_down.ok)
        self.assertIsNone(set_down.top_mm)
        self.assertIn("no surface", set_down.reason)
        self.assertIn("'blue plate'", set_down.render())

    def test_a_sliver_of_points_is_not_a_top(self) -> None:
        few = _plate()[:5]
        set_down = SetDown.onto(_object("blue plate", few), grasp=_grasp(26.0), part_bottom_mm=0.0)
        self.assertIsNone(set_down.pose)
        self.assertIn("5 point(s)", set_down.reason)

    def test_a_grasp_that_did_not_stand_above_the_part_gives_no_pose(self) -> None:
        for bottom in (26.0, 40.0):
            with self.subTest(part_bottom_mm=bottom):
                set_down = SetDown.onto(_object("blue plate", _plate()), grasp=_grasp(26.0), part_bottom_mm=bottom)
                self.assertIsNone(set_down.pose)
                self.assertIn("above", set_down.reason)

    def test_a_programmers_error_is_raised(self) -> None:
        target = _object("blue plate", _plate())
        camera_pose = Pose(position_mm=np.zeros(3), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=Frame.CAMERA)
        rows = (
            ("a grasp in the camera frame", dict(grasp=camera_pose, part_bottom_mm=0.0)),
            ("negative air", dict(grasp=_grasp(26.0), part_bottom_mm=0.0, air_mm=-1.0)),
            ("air that is not a number", dict(grasp=_grasp(26.0), part_bottom_mm=0.0, air_mm=float("nan"))),
            ("a bottom that is not a number", dict(grasp=_grasp(26.0), part_bottom_mm=float("inf"))),
        )
        for label, kwargs in rows:
            with self.subTest(label), self.assertRaises(ValueError):
                SetDown.onto(target, **kwargs)  # type: ignore[arg-type]

    def test_an_object_the_frame_does_not_hold_is_an_index_error(self) -> None:
        plate = _located("wrist", _object("blue plate", _plate()))
        with self.assertRaises(IndexError):
            plate.set_down(1, grasp=_grasp(26.0), part_bottom_mm=0.0)


class TheSetDownSaysWhatItMeasuredTests(unittest.TestCase):
    def test_it_prints_as_it_renders_in_ascii_and_its_dict_is_plain_data(self) -> None:
        measured = SetDown.onto(_object("blue plate", _plate()), grasp=_grasp(26.0), part_bottom_mm=0.0)
        refused = SetDown.onto(_object("blue plate", np.zeros((0, 3))), grasp=_grasp(26.0), part_bottom_mm=0.0)
        for set_down in (measured, refused):
            with self.subTest(ok=set_down.ok):
                text = set_down.render()
                self.assertEqual(text, str(set_down))
                self.assertTrue(text.isascii())
                self.assertFalse(text.endswith("\n"))
                data = json.loads(json.dumps(set_down.to_dict()))
                self.assertEqual(set_down.ok, data["ok"])
        self.assertIn("20.0 mm", measured.render())
        self.assertIn("5.0 mm of air", measured.render())
        self.assertEqual([*_PLATE_XY, _PLATE_TOP_MM + 31.0], measured.to_dict()["pose_mm"])
        self.assertIsNone(refused.to_dict()["pose_mm"])


# ---------------------------------------------------------------------------------------------------
# The place, against a world wired as the cell wires it
# ---------------------------------------------------------------------------------------------------


class _WorldAskingArm(DummyRobotArm):
    """The dummy arm with the cell's live world: at each motion it asks the world what it would plan against."""

    def __init__(self, world: Any, self_envelope: Any) -> None:
        super().__init__()
        self.live_planner_world = world
        self.self_envelope = self_envelope
        self.perceived: list[int] = []

    def move(self, pose: Pose, **kwargs: Any) -> Any:
        self.live_planner_world.drop_cached_frames()
        seen = self.live_planner_world.world_for(self_envelope=self.self_envelope, now=50.1)
        self.perceived.append(int(seen.perceived_count))
        return super().move(pose, **kwargs)


class ThePlaceLeavesTheLocatedTargetOutOfTheWorldTests(unittest.TestCase):
    def test_the_target_leaves_the_world_for_the_place_and_comes_back_after_it(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.execution.camera_world_wiring import CameraWorldPlan, CameraWorldWiring
        from src.robot.safety.planning.perceived import LinkCapsule, SelfEnvelope
        from tests.test_locator import _backend, _block_frame, _Handle, _Owner

        cfg = RobotConfig.model_validate({
            "vendor": "ur", "safety": {"payload": {"enforce": False}, "planning_world": {
                "enabled": True,
                "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0], "thickness_mm": 50.0},
                "perceived": {"enabled": True},
            }},
        })
        rigs = [SimpleNamespace(rig_id="overhead", enabled=True, source="rgbd",
                                extrinsics=SimpleNamespace(mounting_mode="eye_to_hand"))]
        owner = _Owner(rig_id="overhead", handle=_Handle([_block_frame(captured_at_s=50.0)]))
        plan = CameraWorldPlan.from_config(cfg, rigs, primary_rig_id="overhead")
        world = CameraWorldWiring.from_cameras(cfg, plan=plan, cameras={"overhead": owner}).world
        assert world is not None
        envelope = SelfEnvelope(frames_mm=(np.eye(4),), capsules=(
            LinkCapsule(frame=0, start_mm=(-500.0, -500.0, 600.0), end_mm=(-500.0, -500.0, 700.0), radius_mm=10.0),))
        arm = _WorldAskingArm(world, envelope)
        arm.connect()
        robot = Robot.from_parts(arm=arm, gripper=_Hand(hold=HoldEvidence.EMPTY), lock_key=None)
        target = Locator.from_parts(camera=owner, backend=_backend("blue plate")).locate("the blue plate")
        over = Pose.tool_down(0.0, 0.0, 150.0)

        placed = robot.place(over, keep_out=target.keep_out(0))
        after = robot.move(Pose.tool_down(0.0, 0.0, 300.0))

        self.assertTrue(placed.ok and placed.keep_out_held, placed.render())
        self.assertTrue(after.ok, after.render())
        self.assertEqual([0, 0, 0, 1], arm.perceived, "the target was an obstacle during the place, or not after it")


# ---------------------------------------------------------------------------------------------------
# The locator says where its camera rides
# ---------------------------------------------------------------------------------------------------


class ALocatorSaysWhetherItsCameraRidesOnTheWristTests(unittest.TestCase):
    def test_a_wrist_locator_is_on_the_wrist_and_a_fixed_one_is_not(self) -> None:
        from tests.test_locator import _TOOL_FRAME, _backend, _Owner, _wrist

        fixed = Locator.from_parts(camera=_Owner(), backend=_backend())
        wrist = Locator.from_parts(camera=_Owner(calibration=_wrist(), rig_id="wrist"), backend=_backend(),
                                   tool_pose=lambda: _grasp(300.0), tool_frame=_TOOL_FRAME)
        self.assertFalse(fixed.on_the_wrist)
        self.assertTrue(wrist.on_the_wrist)


# ---------------------------------------------------------------------------------------------------
# The locators of a cell share one perception backend
# ---------------------------------------------------------------------------------------------------


class _Watching:
    """A view that records which cameras it was handed, as ``LiveView.watch`` is called."""

    def __init__(self) -> None:
        self.watched: list[str] = []

    def watch(self, camera: Any) -> None:
        self.watched.append(str(camera.rig_id))


def _models_tree() -> Any:
    from tests.test_locator import _cell_tree

    return SimpleNamespace(robot=_cell_tree(), app_config=SimpleNamespace(models=object()))


class TheLocatorsOfACellShareOneBackendTests(unittest.TestCase):
    """Building a backend loads the detector's and the segmenter's weights (``PerceptionSpec.build``), with no cache.

    A locator per camera from ``Locator.from_tree`` loaded them once per camera; ``Locator.for_cameras`` loads them once
    for the cell, as the cell's own pick path does (``autonomous_grasp.cells``: stateless per call, shared by every
    camera).
    """

    def test_n_cameras_build_one_backend_and_every_locator_uses_it(self) -> None:
        from tests.test_locator import _backend, _Owner, _wrist

        built = _backend("red cube")
        spec = mock.MagicMock()
        spec.from_config.return_value.build.return_value = built
        tree = _models_tree()
        cameras = [_Owner(calibration=_wrist(), rig_id="wrist"), _Owner(rig_id="overhead"), _Owner(rig_id="side")]
        tool_pose = mock.MagicMock(return_value=_grasp(300.0))
        view = _Watching()
        with mock.patch("src.models.perception_spec.PerceptionSpec", spec):
            locators = Locator.for_cameras(tree, cameras, tool_pose=tool_pose, view=view)

        spec.from_config.assert_called_once_with(tree.app_config.models)
        spec.from_config.return_value.build.assert_called_once_with()
        self.assertEqual(3, len(locators))
        self.assertTrue(all(locator._backend is built for locator in locators))  # noqa: SLF001
        self.assertEqual(cameras, [locator._camera for locator in locators])  # noqa: SLF001
        self.assertEqual([True, False, False], [locator.on_the_wrist for locator in locators])
        self.assertEqual([4, 4, 4], [locator._attempts for locator in locators])  # noqa: SLF001
        self.assertEqual(["wrist", "overhead", "side"], view.watched)

    def test_one_camera_it_cannot_place_refuses_them_all_before_any_model_loads_or_window_opens(self) -> None:
        from tests.test_locator import _backend, _Owner, _wrist

        spec = mock.MagicMock()
        spec.from_config.return_value.build.return_value = _backend()
        view = _Watching()
        rows: "tuple[tuple[str, list[Any], dict[str, Any]], ...]" = (
            ("a wrist camera with no TCP reader", [_Owner(rig_id="overhead"), _Owner(calibration=_wrist(), rig_id="wrist")],
             {}),
            ("a rig with no depth", [_Owner(rig_id="overhead"), SimpleNamespace(rig_id="stereo", source="stereo")],
             {"tool_pose": lambda: _grasp(300.0)}),
        )
        for label, cameras, keywords in rows:
            with self.subTest(label), mock.patch("src.models.perception_spec.PerceptionSpec", spec):
                with self.assertRaises(LocatorRefused):
                    Locator.for_cameras(_models_tree(), cameras, view=view, **keywords)
        spec.from_config.assert_not_called()
        self.assertEqual([], view.watched)

    def test_no_camera_is_refused_with_nothing_loaded(self) -> None:
        spec = mock.MagicMock()
        with mock.patch("src.models.perception_spec.PerceptionSpec", spec):
            with self.assertRaises(LocatorRefused) as raised:
                Locator.for_cameras(_models_tree(), [])
        spec.from_config.assert_not_called()
        self.assertIn("no camera", str(raised.exception))

    def test_one_camera_builds_what_from_tree_builds(self) -> None:
        from tests.test_locator import _backend, _Owner

        spec = mock.MagicMock()
        spec.from_config.return_value.build.return_value = _backend()
        tree = _models_tree()
        with mock.patch("src.models.perception_spec.PerceptionSpec", spec):
            (shared,) = Locator.for_cameras(tree, [_Owner()])
            alone = Locator.from_tree(tree, camera=_Owner())
        self.assertIs(shared._backend, alone._backend)  # noqa: SLF001
        self.assertEqual((shared._attempts, shared.on_the_wrist), (alone._attempts, alone.on_the_wrist))  # noqa: SLF001


# ---------------------------------------------------------------------------------------------------
# Example 13 on the owner's cell doubles
# ---------------------------------------------------------------------------------------------------

_OBJECT, _TARGET = "a red cube", "the blue plate"


class _Owners:
    """``Camera`` as example 13 calls it: one owner per rig, each opened and released on the log."""

    def __init__(self, cell: _Cell) -> None:
        self.cell = cell
        self.built: list[str] = []

    def from_tree(self, tree: Any, *, rig_id: str) -> "_Owner13":
        self.built.append(rig_id)
        return _Owner13(self.cell, rig_id)


class _Owner13:
    def __init__(self, cell: _Cell, rig_id: str) -> None:
        self.cell = cell
        self.rig_id = rig_id

    def __enter__(self) -> "_Owner13":
        self.cell.log.write("camera", self.rig_id, "open")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.cell.log.write("camera", self.rig_id, "released")


class _Seeing:
    """One camera's locator: on the wrist or not, and what it finds for each prompt, in turn; each locate on the log."""

    def __init__(self, cell: _Cell, rig_id: str, on_the_wrist: bool, sees: "dict[str, list[np.ndarray | None]]",
                 tool_pose: Any, view: Any = None) -> None:
        self.cell = cell
        self.rig_id = rig_id
        self.on_the_wrist = on_the_wrist
        self.sees = {prompt: list(answers) for prompt, answers in sees.items()}
        self.tool_pose = tool_pose
        self.view = view

    def locate(self, prompt: str) -> Located:
        answers = self.sees.get(prompt, [])
        points = answers.pop(0) if answers else None
        self.cell.log.write("locate", self.rig_id, prompt, points is not None)
        label = "red cube" if prompt == _OBJECT else "blue plate"
        return _located(self.rig_id, *(() if points is None else (_object(label, points),)))


class _Locators:
    """``Locator`` as example 13 calls it: the scripted locator of each camera handed in, and how many perception
    backends were built for them: one per ``for_cameras`` call, one per ``from_tree`` call."""

    def __init__(self, cell: _Cell, script: "dict[str, tuple[bool, dict[str, list[np.ndarray | None]]]]") -> None:
        self.cell = cell
        self.script = script
        self.built: list[_Seeing] = []
        self.backends: list[list[str]] = []

    def _seeing(self, camera: _Owner13, tool_pose: Any, view: Any) -> _Seeing:
        on_the_wrist, sees = self.script[camera.rig_id]
        seeing = _Seeing(self.cell, camera.rig_id, on_the_wrist, sees, tool_pose, view)
        self.built.append(seeing)
        return seeing

    def from_tree(self, tree: Any, *, camera: _Owner13, tool_pose: Any, view: Any = None) -> _Seeing:
        self.backends.append([camera.rig_id])
        return self._seeing(camera, tool_pose, view)

    def for_cameras(self, tree: Any, cameras: list[_Owner13], *, tool_pose: Any, view: Any = None) -> list[_Seeing]:
        self.backends.append([camera.rig_id for camera in cameras])
        return [self._seeing(camera, tool_pose, view) for camera in cameras]


class _View13:
    """The camera windows example 13 opens: which rigs, whether shown, and its opening and closing on the log."""

    def __init__(self, cell: _Cell, rig_ids: list[str], show: bool) -> None:
        self.cell = cell
        self.rig_ids = rig_ids
        self.show = show

    def __enter__(self) -> "_View13":
        self.cell.log.write("view", "opened")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.cell.log.write("view", "closed")


class _Views:
    """``LiveView`` as example 13 calls it: one view over every camera it opened."""

    def __init__(self, cell: _Cell) -> None:
        self.cell = cell
        self.built: list[_View13] = []

    def __call__(self, cameras: list[_Owner13], *, show: bool) -> _View13:
        view = _View13(self.cell, [camera.rig_id for camera in cameras], show)
        self.built.append(view)
        return view


class _Watched:
    """The owner's robot as example 13 holds it: the real ``Robot``, each verb marked on the log for ``_check``."""

    def __init__(self, cell: _Cell) -> None:
        self.cell = cell
        self.robot = _robot(cell)
        self.picks: list[dict[str, Any]] = []
        self.places: list[tuple[Pose, dict[str, Any]]] = []

    @property
    def arm(self) -> Any:
        return self.robot.arm

    def connected(self) -> Any:
        return _connected(self.cell, self.robot)

    def move_joints(self, joints: Any, **keywords: Any) -> Any:
        with _verb(self.cell, "move"):
            return self.robot.move_joints(joints, **keywords)

    def pick(self, pose: Pose, width_mm: float, **keywords: Any) -> Any:
        self.picks.append(dict(keywords, pose=pose, width_mm=width_mm))
        with _verb(self.cell, "pick"):
            return self.robot.pick(pose, width_mm, **keywords)

    def place(self, pose: Pose, **keywords: Any) -> Any:
        self.places.append((pose, keywords))
        with _verb(self.cell, "place"):
            return self.robot.place(pose, **keywords)


class _Robots:
    """``Robot`` as example 13 calls it: the watched robot, and the cameras it was handed."""

    def __init__(self, watched: _Watched) -> None:
        self.watched = watched
        self.cameras: list[str] = []

    def from_tree(self, tree: Any, *, cameras: list[_Owner13]) -> _Watched:
        self.cameras = [camera.rig_id for camera in cameras]
        return self.watched


def _plans(rig_ids: "tuple[str, ...]", refusal: "str | None" = None) -> Any:
    plan = SimpleNamespace(rig_ids=rig_ids, reason="" if rig_ids else "no enabled RGB-D rig declares its calibration",
                           refusal=lambda: refusal)
    return SimpleNamespace(from_config=lambda robot_cfg, rigs, *, primary_rig_id: plan)


class _Run:
    """One run of example 13 on a cell whose cameras see what ``script`` says, primary first."""

    def __init__(self, script: "dict[str, tuple[bool, dict[str, list[np.ndarray | None]]]]", *, answers: str = "",
                 rig_ids: "tuple[str, ...] | None" = None, refusal: "str | None" = None, **arm: Any) -> None:
        self.cell = _cell(answers, **arm)
        self.owners = _Owners(self.cell)
        self.locators = _Locators(self.cell, script)
        self.watched = _Watched(self.cell)
        self.robots = _Robots(self.watched)
        self.views = _Views(self.cell)
        self.rig_ids = tuple(script) if rig_ids is None else rig_ids
        self.refusal = refusal
        self.printed = ""
        self.exit: "SystemExit | None" = None
        self.globals: dict[str, Any] = {}

    def __call__(self) -> "_Run":
        rigs = [SimpleNamespace(rig_id=rig_id) for rig_id in self.rig_ids]
        tree = SimpleNamespace(robot=_hande(), app_config=SimpleNamespace(camera=SimpleNamespace(
            cameras=SimpleNamespace(rigs=rigs, primary_rig_id=self.rig_ids[0] if self.rig_ids else "wrist"))))
        doubles = {"load_tree": lambda: tree, "Camera": self.owners, "CameraWorldPlan": _plans(self.rig_ids, self.refusal),
                   "Locator": self.locators, "Robot": self.robots, "LiveView": self.views}
        out = io.StringIO()
        with mock.patch.dict(vars(willy), doubles), redirect_stdout(out):
            try:
                self.globals = runpy.run_path(str(_EXAMPLE), run_name="__main__")
            except SystemExit as exc:
                self.exit = exc
        self.printed = out.getvalue()
        return self

    def events(self, *kinds: str) -> list[tuple[Any, ...]]:
        return [(e.kind, *e.data) for e in self.cell.log.kinds(*kinds)]

    def looks_and_locates(self) -> list[tuple[Any, ...]]:
        return [("look",) if e.kind == "joints" else (e.kind, *e.data[:3])
                for e in self.cell.log.kinds("joints", "locate")]


def _sees(**by_prompt: "list[np.ndarray | None]") -> "dict[str, list[np.ndarray | None]]":
    return {(_OBJECT if key == "part" else _TARGET): answers for key, answers in by_prompt.items()}


class Example13PicksAndPlacesWhatTheCamerasFindTests(unittest.TestCase):
    def test_the_first_camera_that_finds_a_prompt_answers_it_and_the_part_lands_on_the_target(self) -> None:
        run = _Run({
            "wrist": (True, _sees(part=[None, None], target=[_plate()])),
            "overhead": (False, _sees(part=[_cube(40.0)], target=[_plate(top_mm=99.0)])),
        })()

        self.assertIsNone(run.exit, run.printed[-2000:])
        spans = _check(self, run.cell)
        self.assertEqual([
            ("look",), ("locate", "wrist", _OBJECT, False),
            ("look",), ("locate", "wrist", _OBJECT, False),
            ("locate", "overhead", _OBJECT, True),  # a fixed camera locates as the arm stands
            ("look",), ("locate", "wrist", _TARGET, True),  # the first camera that found the target answers it
        ], run.looks_and_locates())
        pick, place = _only(spans, "pick"), _only(spans, "place")
        self.assertEqual(["closed"], [e.data[0] for e in pick.of("edge")])
        self.assertEqual(["open"], [e.data[0] for e in place.of("edge")])
        # The release came where the place's line in arrived, at the measured set-down, and nowhere before it.
        set_down = run.globals["set_down"]
        best, scene = run.globals["best"], run.globals["scene"]
        self.assertEqual(_at(set_down.pose), _line_before(place, place.of("edge")[0]).data[0])
        hang = float(best.position_mm[2]) - float(scene.declared_support_height_mm)
        np.testing.assert_allclose(set_down.pose.position_mm, (*_PLATE_XY, _PLATE_TOP_MM + hang + 5.0), atol=1e-9)
        # One perception backend for both cameras (review of 2026-09-24): the weights load once for the cell.
        self.assertEqual([["wrist", "overhead"]], run.locators.backends)
        self.assertFalse(run.cell.hand_e.closed, "the part was not let go at the target")
        # Each verb got its own camera's keep-out, and every camera is the robot's and every tool pose the arm's.
        self.assertEqual(("overhead", "red cube"), (run.watched.picks[0]["keep_out"].camera,
                                                    run.watched.picks[0]["keep_out"].target_label))
        self.assertEqual(("wrist", "blue plate"), (run.watched.places[0][1]["keep_out"].camera,
                                                   run.watched.places[0][1]["keep_out"].target_label))
        self.assertEqual(["wrist", "overhead"], run.robots.cameras)
        self.assertTrue(all(seeing.tool_pose == run.cell.arm.get_tcp_pose for seeing in run.locators.built))
        self.assertEqual([("camera", "wrist", "open"), ("camera", "overhead", "open"),
                          ("camera", "overhead", "released"), ("camera", "wrist", "released")], run.events("camera"))
        self.assertIn("5.0 mm of air", run.printed)
        # Every camera in the windows, every locator showing on them, and the windows closed before a camera is given
        # back (the owner's switch, 2026-09-24).
        (view,) = run.views.built
        self.assertEqual((["wrist", "overhead"], True), (view.rig_ids, view.show))
        self.assertTrue(all(seeing.view is view for seeing in run.locators.built), "a locator without the windows")
        self.assertEqual([("camera", "wrist", "open"), ("camera", "overhead", "open"), ("view", "opened"),
                          ("view", "closed"), ("camera", "overhead", "released"), ("camera", "wrist", "released")],
                         run.events("camera", "view"))

    def test_the_first_camera_that_finds_the_part_is_the_only_one_asked(self) -> None:
        run = _Run({
            "wrist": (True, _sees(part=[_cube(40.0)], target=[_plate()])),
            "overhead": (False, _sees(part=[_cube(40.0)], target=[_plate()])),
        })()

        self.assertIsNone(run.exit, run.printed[-2000:])
        _check(self, run.cell)
        self.assertEqual([("look",), ("locate", "wrist", _OBJECT, True), ("look",), ("locate", "wrist", _TARGET, True)],
                         run.looks_and_locates())

    def test_a_part_whose_foot_the_camera_missed_lands_with_its_air_and_is_not_pressed(self) -> None:
        """The reviewer's worst case through the example: 12 mm of the near face unseen, the part on the table at 0."""
        run = _Run({"overhead": (False, _sees(part=[_cube_seen_down_to(12.0)], target=[_plate()]))})()

        self.assertIsNone(run.exit, run.printed[-2000:])
        spans = _check(self, run.cell)
        set_down, best, scene = run.globals["set_down"], run.globals["best"], run.globals["scene"]
        self.assertAlmostEqual(12.0, scene.support_height_mm, places=6, msg="the control: the seen support is raised")
        self.assertGreaterEqual(_real_air(set_down, best.pose(), stood_on_mm=0.0), 5.0 - 1e-9, set_down.render())
        place = _only(spans, "place")
        self.assertEqual(_at(set_down.pose), _line_before(place, place.of("edge")[0]).data[0])

    def test_a_part_no_camera_finds_is_never_picked(self) -> None:
        run = _Run({
            "wrist": (True, _sees(part=[None, None])),
            "overhead": (False, _sees(part=[None])),
        })()

        assert run.exit is not None
        self.assertIn(f"no camera located {_OBJECT!r}", str(run.exit.code))
        self.assertIn("nothing was picked", str(run.exit.code))
        _check(self, run.cell)
        self.assertEqual([
            ("look",), ("locate", "wrist", _OBJECT, False),
            ("look",), ("locate", "wrist", _OBJECT, False),
            ("locate", "overhead", _OBJECT, False),
        ], run.looks_and_locates())
        self.assertEqual([], run.watched.picks)
        self.assertEqual([], run.events("edge"))
        self.assertEqual([], run.events("move"), "the arm moved beyond its looks")
        self.assertEqual([("camera", "wrist", "open"), ("camera", "overhead", "open"),
                          ("camera", "overhead", "released"), ("camera", "wrist", "released")], run.events("camera"))

    def test_a_target_no_camera_finds_leaves_the_part_held_with_no_pulse(self) -> None:
        run = _Run({
            "wrist": (True, _sees(part=[_cube(40.0)], target=[None, None])),
            "overhead": (False, _sees(target=[None])),
        })()

        assert run.exit is not None
        self.assertIn(f"no camera located {_TARGET!r}", str(run.exit.code))
        self.assertIn("the part stays held", str(run.exit.code))
        _check(self, run.cell)
        self.assertEqual([("edge", "closed")], run.events("edge"), "a pulse after the part was held")
        self.assertTrue(run.cell.hand_e.closed)
        self.assertEqual([], run.watched.places)

    def test_a_target_with_no_top_to_read_leaves_the_part_held_with_no_pulse(self) -> None:
        run = _Run({"wrist": (True, _sees(part=[_cube(40.0)], target=[np.zeros((0, 3))]))})()

        assert run.exit is not None
        self.assertIn("the part stays held", str(run.exit.code))
        _check(self, run.cell)
        self.assertEqual([("edge", "closed")], run.events("edge"))
        self.assertTrue(run.cell.hand_e.closed)
        self.assertEqual([], run.watched.places)
        self.assertIn("no surface", run.printed)

    def test_a_refused_pick_ends_the_program_before_the_target_is_looked_for(self) -> None:
        run = _Run({"wrist": (True, _sees(part=[_cube(40.0)], target=[_plate()]))}, refuse=1)()

        assert run.exit is not None
        self.assertIn("the pick did not finish", str(run.exit.code))
        _check(self, run.cell)
        self.assertEqual([], run.events("edge"))
        self.assertNotIn(("locate", "wrist", _TARGET, True), run.looks_and_locates())
        self.assertEqual([], run.watched.places)

    def test_a_refused_place_keeps_the_part(self) -> None:
        """The place's line in is refused: the hand never opens, and the program says the part is still held."""
        run = _Run({"wrist": (True, _sees(part=[_cube(40.0)], target=[_plate()]))}, refuse=4)()

        assert run.exit is not None
        self.assertIn("the place did not finish", str(run.exit.code))
        spans = _check(self, run.cell)
        self.assertEqual([], _only(spans, "place").of("edge"))
        self.assertEqual([("edge", "closed")], run.events("edge"))
        self.assertTrue(run.cell.hand_e.closed)
        self.assertEqual("refused", _only(spans, "place").of(*_MOTIONS)[-1].data[2])

    def test_a_cell_whose_cameras_build_no_world_opens_no_camera(self) -> None:
        for refusal in (None, "this cell plans with cuRobo and calibrates 'wrist', and no live camera world comes of it"):
            with self.subTest(refusal=refusal):
                run = _Run({"wrist": (True, {})}, rig_ids=(), refusal=refusal)()
                assert run.exit is not None
                self.assertIn(refusal or "no camera feeds a world", str(run.exit.code))
                self.assertEqual([], run.owners.built)
                self.assertEqual([], run.views.built)
                self.assertEqual([], run.cell.log.events)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
