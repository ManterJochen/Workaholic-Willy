"""A camera on the wrist locates a part from every look it is handed, fused, until the part's grasp is safe.

The owner, 2026-09-28 and 2026-09-29 (plan step 7, addendum 1, 2, 4 and 7): example 13 finds its part through
``Locator.look_around(prompt, looks, robot=robot)``. For a camera on the wrist the arm goes to each look in turn through
the robot's own verbs, the camera locates there, and every look is fused with the looks before it at the wrist's
association tolerances; object 0 of the first look that located anything is the part, kept across the looks. After each
look the grasp is computed again on the fused cloud (``Located.scene(0, robot_config).grasps().best``), and the looking
stops at the first look whose grasp is valid and which the looks agree on (the default: fast); with ``both_faces`` only
once both jaw contact faces of that grasp were seen, and a part no look showed both faces of is refused. A look the
planner refuses before anything was sent is skipped and said; one that may have moved the arm, a stopped controller or
a camera that cannot vouch end the looking there. The frames of the looks are held in the arm's live planner world from
the look around to the end of the pick, and let go by the pick, a new look around or a disconnect. A fixed camera
locates once, as it stands.

Every frame is ray-cast from where the arm's tool stands (``tests/_wrist_views.py``: the owner's tilted D415 at half
resolution, a 40 mm cube on the bench) and placed through the real locator by the tool pose stamped at its shutter.
Seen from one 45 degree look the cube's best grasp closes across the two faces that look does not face; the looks from
its east and its west together show both faces the grasp then closes on.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
import unittest
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.calibration.rig_calibration import RigCalibration
from src.calibration.serialization import FlangeToTcp
from src.camera.setup.image_taking.frames import RGBDFrame
from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionCommand, MotionResult, MotionStatus
from src.robot.core.arm_capabilities import RobotMode, RobotStatus, SafetyMode
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.core.gripper import HoldEvidence
from src.robot.execution.lifecycle import ConnectedCell
from src.robot.execution.robot import Robot
from src.robot.grasping.generation.depth_steps import pixels_behind_depth_steps
from src.robot.grasping.multiview.scene_geometry import grazing_pixels, to_base_mm
from src.robot.perception.locator import Located, LocatedObject, Locator, LocatorRefused, _Placed
from tests._wrist_views import (
    CUBE,
    CUBE_CENTRE,
    HEIGHT,
    WIDTH,
    Box,
    K,
    LookingArm,
    camera_looking_at,
    camera_to_tool,
    joints_key,
    mount,
    render,
    tool_for,
)
from tests.test_a_scene_plans_the_trees_hand import _hande, _replace
from tests.test_locator import _backend, _Owner, _Segmentation
from tests.test_robot_hand_verbs import _Hand

_LOGGER = "src.robot.perception.locator"
_TOOL_FRAME = SimpleNamespace(source="willy", offset_mm=(0.0, 0.0, 0.0), rotation_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                              verify_tolerance_mm=1.0)


@lru_cache(maxsize=1)
def _cell() -> Any:
    """The owner's hand profile, loaded once."""
    return _hande()


@dataclass(frozen=True)
class _Look:
    """A look a program declares: the joints, the tool pose they put the camera at, and how a report names it."""

    joints: JointPositions
    tool: Pose

    @property
    def label(self) -> str:
        return "(" + ", ".join(f"{v:.1f}" for v in self.joints.degrees()) + ") deg"


def _look(bearing_deg: float, *, elevation_deg: float = 45.0, range_mm: float = 450.0,
          target: "tuple[float, float, float]" = CUBE_CENTRE, key: float = 0.0) -> _Look:
    """A look at ``target`` from ``bearing_deg`` round it; its joints are unique per look (``key`` tells apart two
    looks from one bearing)."""
    joints = JointPositions.deg(bearing_deg, -100.0 - key, -110.0, -60.0, 90.0 + elevation_deg, range_mm / 10.0)
    return _Look(joints, tool_for(camera_looking_at(target, bearing_deg=bearing_deg, elevation_deg=elevation_deg,
                                                    range_mm=range_mm)))


#: The cube's east, west, north and south, each from 45 degrees up at 450 mm, as the owner's wrist looks.
EAST, WEST, NORTH, SOUTH = _look(0.0), _look(180.0), _look(90.0), _look(270.0)


class _Wrist:
    """The owner's wrist D415 as an open camera owner, and the detector behind it: every frame is rendered from where
    the arm's tool stands, and the detector segments each box the last frame met, in the order the boxes are given.

    ``labels`` renames a box in one located frame, by number (0 for the first locate). ``perceived`` counts locates.
    """

    rig_id = "wrist"
    source = "rgbd"

    def __init__(self, arm: Any, boxes: "tuple[Box, ...]" = (CUBE,), *,
                 labels: "dict[int, dict[str, str]] | None" = None) -> None:
        self.arm = arm
        self.boxes = boxes
        self.labels = labels or {}
        self.perceived = 0
        self.grabs = 0
        self._hit: "np.ndarray | None" = None
        self._renders: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}

    # the owner
    def handle(self) -> "_Wrist":
        return self

    def calibration(self) -> RigCalibration:
        return RigCalibration(rig_id="wrist", mounting_mode="eye_in_hand", artifact_path="eih.json",
                              transform=camera_to_tool(), shutter_motion_tolerance_mm=1.0,
                              shutter_motion_tolerance_deg=0.5, flange_to_tcp=FlangeToTcp.from_matrix("willy", np.eye(4)))

    # the handle
    def grab(self) -> RGBDFrame:
        tool = self.arm.get_tcp_pose().to_matrix()
        key = tuple(np.round(tool, 6).ravel().tolist())
        if key not in self._renders:
            self._renders[key] = render(tool @ mount(), self.boxes)
        depth, self._hit = self._renders[key]
        self.grabs += 1
        return RGBDFrame(color=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), depth=depth, captured_at_s=float(self.grabs))

    def get_intrinsics(self) -> np.ndarray:
        return K.copy()

    # the detector
    def perceive(self, _bgr: Any, _prompt: str) -> "tuple[Any, ...]":
        assert self._hit is not None
        renamed = self.labels.get(self.perceived, {})
        self.perceived += 1
        return tuple(
            SimpleNamespace(detection=SimpleNamespace(score=0.9, box=None), segmentation=_Segmentation(
                label=renamed.get(box.label, box.label), score=0.9, bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
                mask=(self._hit == index)))
            for index, box in enumerate(self.boxes) if np.any(self._hit == index))


class _Blind(_Wrist):
    """The wrist camera measuring no depth on the part from the tool poses in ``blind_at``: a dark or shiny face at a
    bad angle, which the detector still masks."""

    def __init__(self, arm: Any, boxes: "tuple[Box, ...]" = (CUBE,), *, blind_at: "tuple[Pose, ...]" = ()) -> None:
        super().__init__(arm, boxes)
        self.blind_at = tuple(pose.to_matrix() for pose in blind_at)

    def grab(self) -> RGBDFrame:
        frame = super().grab()
        tool = self.arm.get_tcp_pose().to_matrix()
        if not any(np.allclose(tool, blind, atol=1e-6) for blind in self.blind_at):
            return frame
        assert self._hit is not None
        depth = np.array(frame.depth, copy=True)
        depth[self._hit == 0] = 0
        return RGBDFrame(color=frame.color, depth=depth, captured_at_s=frame.captured_at_s)


class _Scripted(Locator):
    """A wrist locator whose frames are scripted: each locate finds one part, the next cloud of ``clouds``, seen from a
    camera straight over it; no depth image, so the fusion takes the clouds as they are."""

    def __init__(self, arm: Any, clouds: "list[np.ndarray]") -> None:
        super().__init__(camera=SimpleNamespace(rig_id="wrist"), backend=None, calibration=None,
                         tool_pose=arm.get_tcp_pose, attempts=1)
        self.clouds = list(clouds)

    def _locate(self, prompt: str) -> _Placed:
        cloud = self.clouds.pop(0)
        over = np.eye(4)
        over[:3, 3] = (float(np.median(cloud[:, 0])), float(np.median(cloud[:, 1])), 450.0)
        part = LocatedObject(label="part", score=0.9, box_px=None, mask=np.zeros((4, 4), dtype=bool),
                             points_base_mm=cloud, centre_mm=tuple(float(v) for v in np.median(cloud, axis=0)))
        return _Placed(located=Located(camera="wrist", captured_at_s=1.0, mounting="eye_in_hand", tool_to_base_mm=None,
                                       objects=(part,)), camera_to_base=over)


def _patch(side: int, *, step_mm: float = 2.0) -> np.ndarray:
    """A flat square of ``side`` x ``side`` points ``step_mm`` apart, 40 mm over the bench at the cube's place."""
    across = np.arange(side, dtype=np.float64) * step_mm
    return np.array([[CUBE_CENTRE[0] + x, CUBE_CENTRE[1] + y, 40.0] for x in across for y in across])


class _World:
    """The arm's live planner world as far as the looks are concerned: when it was asked to hold them and to let them
    go (``calls``), and the target a pick or a place held out of it (``offers``)."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.offers: list[dict[str, Any]] = []

    def hold_pick_views(self) -> bool:
        self.calls.append("hold")
        return True

    def forget_pick_views(self) -> None:
        self.calls.append("forget")

    def offer_segmentation(self, **offer: Any) -> None:
        self.offers.append(offer)

    def forget_segmentation(self) -> None:
        return None


def _arm(*looks: _Look, start: "Pose | None" = None, arm: "type[LookingArm]" = LookingArm, **keywords: Any) -> Any:
    table = {look.joints: look.tool for look in looks}
    built = arm(table, start=start if start is not None else tool_for(camera_looking_at(
        (600.0, 0.0, 0.0), bearing_deg=0.0)), **keywords)
    built.live_planner_world = _World()
    return built


def _robot(arm: Any, *, config: Any = None, hold: HoldEvidence = HoldEvidence.HELD) -> Robot:
    return Robot.from_parts(arm=arm, gripper=_Hand(hold=hold, max_width_mm=50.0), lock_key=None,
                            robot_config=_cell() if config is None else config)


def _locator(wrist: _Wrist, *, tool_pose: Any = None) -> Locator:
    return Locator.from_parts(camera=wrist, backend=wrist, tool_pose=tool_pose or wrist.arm.get_tcp_pose,
                              tool_frame=_TOOL_FRAME, attempts=1)


def _looked(*looks: _Look, boxes: "tuple[Box, ...]" = (CUBE,), both_faces: bool = False, **arm: Any,
            ) -> "tuple[Located, Any, _Wrist]":
    built = _arm(*looks, **arm)
    wrist = _Wrist(built, boxes)
    located = _locator(wrist).look_around("a part", [look.joints for look in looks], robot=_robot(built),
                                          both_faces=both_faces)
    return located, built, wrist


def _moved_to(arm: Any) -> list[tuple[float, ...]]:
    return [motion[1] for motion in arm.motions if motion[0] == "joints"]


def _key(look: _Look) -> tuple[float, ...]:
    return tuple(round(value, 1) for value in look.joints.degrees())


# ---------------------------------------------------------------------------------------------------
# The looks, in order, fused
# ---------------------------------------------------------------------------------------------------


class TheLooksAreVisitedInOrderAndFusedTests(unittest.TestCase):
    def test_a_valid_grasp_at_the_first_look_ends_the_looking_there(self) -> None:
        """The default (addendum 1): a valid grasp the looks agree on is safe enough; jaw faces are not asked."""
        located, arm, wrist = _looked(EAST, WEST, NORTH)

        self.assertEqual([_key(EAST)], _moved_to(arm), "the arm went on after a look whose grasp was safe")
        self.assertEqual(1, wrist.perceived)
        self.assertEqual((EAST.label,), located.looks)
        self.assertEqual((EAST.label,), located.looks_fused)
        self.assertEqual("", located.refused)
        best = located.scene(0, _cell()).grasps().best
        assert best is not None
        # One look: exactly what a locate from there returns, placed by the tool pose stamped at its shutter.
        alone = _locator(_Wrist(_arm(EAST, start=EAST.tool))).locate("a part")
        np.testing.assert_array_equal(alone.objects[0].points_base_mm, located.objects[0].points_base_mm)
        np.testing.assert_allclose(located.tool_to_base_mm, EAST.tool.to_matrix(), atol=1e-9)

    def test_both_faces_goes_on_until_both_contact_faces_of_the_grasp_were_seen_and_fuses_the_looks(self) -> None:
        located, arm, wrist = _looked(EAST, WEST, NORTH, both_faces=True)

        self.assertEqual([_key(EAST), _key(WEST)], _moved_to(arm), "the looks were not visited in order, or on")
        self.assertEqual(2, wrist.perceived)
        self.assertEqual((EAST.label, WEST.label), located.looks)
        self.assertEqual((WEST.label, EAST.label), located.looks_fused, "the look the frame is of comes first")
        self.assertEqual((True, True), located.jaw_faces_seen)
        self.assertEqual("", located.refused)
        cube = located.objects[0]
        # The fused cloud holds the east face one look saw and the west face the other saw.
        self.assertGreater(int(np.sum(np.abs(cube.points_base_mm[:, 0] - 20.0) < 1.0)), 100)
        self.assertGreater(int(np.sum(np.abs(cube.points_base_mm[:, 0] + 20.0) < 1.0)), 100)
        self.assertTrue(CUBE.holds(cube.points_base_mm, margin_mm=2.0).all(), "a point off the part joined its cloud")
        best = located.scene(0, _cell()).grasps().best
        assert best is not None
        self.assertAlmostEqual(1.0, abs(float(best.closing_axis[0])), places=3, msg="the grasp closes across the "
                               "faces the two looks saw")
        # The frame is the one of the last look, where the arm stands.
        np.testing.assert_allclose(located.tool_to_base_mm, WEST.tool.to_matrix(), atol=1e-9)

    def test_a_part_no_look_showed_both_faces_of_is_refused_with_both_faces(self) -> None:
        """Fail closed (addendum 1): nothing may be planned on it, and the reason names the contact face not seen."""
        located, arm, _ = _looked(EAST, both_faces=True)

        self.assertIn("contact face", located.refused)
        self.assertIn("of the chosen grasp was not seen", located.refused.replace("faces", "face").replace(
            "were", "was"))
        assert located.jaw_faces_seen is not None
        self.assertFalse(all(located.jaw_faces_seen))
        with self.assertRaises(LocatorRefused):
            located.scene(0, _cell())
        with self.assertRaises(LocatorRefused):
            located.set_down(0, grasp=Pose.tool_down(0.0, -700.0, 60.0), part_bottom_mm=0.0)
        self.assertIn("REFUSED", located.render())
        self.assertEqual([_key(EAST)], _moved_to(arm))

    def test_the_same_look_without_both_faces_is_not_refused(self) -> None:
        located, _, _ = _looked(EAST)
        self.assertEqual("", located.refused)
        self.assertIsNotNone(located.scene(0, _cell()).grasps().best)

    def test_with_no_looks_it_looks_where_the_arm_stands(self) -> None:
        built = _arm(EAST, start=EAST.tool)
        wrist = _Wrist(built)

        located = _locator(wrist).look_around("a part", robot=_robot(built))

        self.assertEqual([], _moved_to(built), "the arm moved with no look handed")
        self.assertEqual(("here",), located.looks)
        self.assertEqual(1, len(located.objects))

    def test_the_keep_out_offers_the_fused_target(self) -> None:
        located, _, _ = _looked(EAST, WEST, both_faces=True)
        offer = located.keep_out(0)
        assert offer.target_points_base_mm is not None
        np.testing.assert_array_equal(located.objects[0].points_base_mm, offer.target_points_base_mm)
        self.assertEqual("wrist", offer.camera)


class ALookThatDoesNotSeeThePartTests(unittest.TestCase):
    def test_it_is_left_out_and_said_and_the_part_stays_as_the_other_looks_saw_it(self) -> None:
        far = Box((290.0, -720.0, 0.0), (310.0, -680.0, 30.0), "block")
        elsewhere = _look(180.0, target=(300.0, -700.0, 15.0), key=1.0)
        with self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            located, arm, _ = _looked(EAST, elsewhere, WEST, boxes=(CUBE, far), both_faces=True)

        self.assertEqual([_key(EAST), _key(elsewhere), _key(WEST)], _moved_to(arm))
        self.assertEqual((EAST.label, elsewhere.label, WEST.label), located.looks)
        self.assertEqual((WEST.label, EAST.label), located.looks_fused, "the look that missed the part was fused")
        self.assertTrue(any(elsewhere.label in line and "did not see the part" in line for line in said.output),
                        said.output)
        self.assertEqual("part", located.objects[0].label)
        self.assertTrue(CUBE.holds(located.objects[0].points_base_mm, margin_mm=2.0).all())

    def test_a_look_that_measured_no_surface_of_the_part_does_not_make_it_the_next_look_does(self) -> None:
        """A dark or shiny face at a bad angle: the detector masks the part, and no depth lies under the mask. Nothing a
        later look sees associates with no surface, so that look makes no part; the next look that measures it does,
        and the looking stops there as it would have at the first."""
        built = _arm(EAST, WEST, NORTH)
        wrist = _Blind(built, blind_at=(EAST.tool,))
        with self.assertLogs(_LOGGER, level=logging.INFO) as said:
            located = _locator(wrist).look_around("a part", [EAST.joints, WEST.joints, NORTH.joints],
                                                  robot=_robot(built))

        self.assertEqual([_key(EAST), _key(WEST)], _moved_to(built), "the looking went on past a look that measured it")
        self.assertEqual((EAST.label, WEST.label), located.looks)
        self.assertEqual((WEST.label,), located.looks_fused)
        self.assertIsNotNone(located.scene(0, _cell()).grasps().best)
        self.assertFalse(any("did not see the part" in line for line in said.output), said.output)
        self.assertTrue(any(EAST.label in line and "no surface" in line for line in said.output), said.output)

    def test_a_part_no_look_measured_is_answered_as_located_with_no_surface(self) -> None:
        built = _arm(EAST, WEST)
        located = _locator(_Blind(built, blind_at=(EAST.tool, WEST.tool))).look_around(
            "a part", [EAST.joints, WEST.joints], robot=_robot(built))

        self.assertEqual([_key(EAST), _key(WEST)], _moved_to(built))
        self.assertEqual(("part",), tuple(obj.label for obj in located.objects), "the part the detector found is said")
        self.assertEqual(0, located.objects[0].points_base_mm.shape[0])
        self.assertEqual((), located.looks_fused)
        self.assertEqual("", located.refused)
        self.assertIsNone(located.scene(0, _cell()).grasps().best)


# ---------------------------------------------------------------------------------------------------
# What the looks add (addendum 7)
# ---------------------------------------------------------------------------------------------------


class WhatTheLooksAddTests(unittest.TestCase):
    def test_a_neighbour_only_an_earlier_look_saw_stays_an_obstacle_to_the_grasp(self) -> None:
        """7.2: the east look sees a block 330 mm west of the cube, the west look stands over it and does not."""
        block = Box((-350.0, -720.0, 0.0), (-310.0, -680.0, 20.0), "block")
        located, _, _ = _looked(EAST, WEST, boxes=(CUBE, block), both_faces=True)

        self.assertEqual(["part", "block"], [obj.label for obj in located.objects])
        carried = located.objects[1]
        self.assertTrue(block.holds(carried.points_base_mm, margin_mm=2.0).all())
        self.assertFalse(carried.mask.any(), "the block has no pixels in the frame of the look that did not see it")
        scene = located.scene(0, _cell())
        assert scene.obstacle_points_base_mm is not None
        np.testing.assert_array_equal(carried.points_base_mm, scene.obstacle_points_base_mm)

    def test_the_support_is_read_from_every_look_that_saw_the_part(self) -> None:
        """7.1: a look straight over the cube sees its top alone and cannot say what it stands on; fused with a look that
        sees its side down to the bench, the support is read at the cube's foot. The cell declares its table 10 mm
        low, so a support raised to the foot shows."""
        config = _replace(_cell(), "grasping.support.height_mm", -10.0)
        over = _look(0.0, elevation_deg=88.0, key=2.0)
        alone = _Wrist(_arm(over, start=over.tool))
        single = _locator(alone).locate("a part")
        self.assertEqual(-10.0, single.scene(0, config).support_height_mm, "the control: the top alone reads no support")

        built = _arm(over, WEST, EAST)
        located = _locator(_Wrist(built)).look_around("a part", [over.joints, WEST.joints, EAST.joints],
                                                      robot=_robot(built, config=config), both_faces=True)

        self.assertEqual((EAST.label, over.label, WEST.label), located.looks_fused, "this frame's look, then in order")
        self.assertGreater(located.objects[0].points_base_mm.shape[0], single.objects[0].points_base_mm.shape[0])
        self.assertAlmostEqual(0.0, located.scene(0, config).support_height_mm, delta=1.0)

    def test_the_looks_disagreeing_on_what_the_part_is_go_on_to_the_next_look(self) -> None:
        """7.3: the west look calls the part a bolt. The grasp is uncertain, so the looking goes on."""
        agreeing, arm, _ = _looked(EAST, WEST, NORTH, both_faces=True)
        self.assertEqual(2, len(_moved_to(arm)), "the control: two looks that agree stop at the second")

        built = _arm(EAST, WEST, NORTH)
        wrist = _Wrist(built, labels={1: {"part": "bolt"}})
        with self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            located = _locator(wrist).look_around("a part", [EAST.joints, WEST.joints, NORTH.joints],
                                                  robot=_robot(built), both_faces=True)

        self.assertEqual([_key(EAST), _key(WEST), _key(NORTH)], _moved_to(built))
        self.assertTrue(any("bolt" in line and "uncertain" in line for line in said.output), said.output)
        self.assertEqual((EAST.label, WEST.label, NORTH.label), located.looks)
        self.assertTrue(agreeing.objects)

    def test_a_detector_that_names_nothing_does_not_disagree_with_itself(self) -> None:
        """A backend that gives no labels: the locator names each object by its place in the frame (``object_1``), and
        the part at another place in a later frame is the same part, not two looks disagreeing on what it is (7.3). The
        west look sees a block 330 mm east of the cube before it, which the east look stands over."""
        east_block = Box((310.0, -720.0, 0.0), (350.0, -680.0, 20.0), "block")
        unnamed = {"part": "", "block": ""}
        built = _arm(EAST, WEST, NORTH)
        wrist = _Wrist(built, (east_block, CUBE), labels={0: unnamed, 1: unnamed, 2: unnamed})

        located = _locator(wrist).look_around("a part", [EAST.joints, WEST.joints, NORTH.joints], robot=_robot(built),
                                              both_faces=True)

        self.assertEqual(["object_1", "object_0"], [obj.label for obj in located.objects],
                         "the control: the west look names the part by its second place")
        self.assertEqual([_key(EAST), _key(WEST)], _moved_to(built), "made-up names were read as a disagreement")
        self.assertEqual((True, True), located.jaw_faces_seen)

    def test_grazing_points_are_thinned_before_the_looks_are_fused(self) -> None:
        """7.4: a low look 250 mm from the cube sees its top at a grazing incidence; its fused cloud is each look's
        surface less the pixels behind a depth step and less the grazing ones."""
        low = _look(0.0, elevation_deg=15.0, range_mm=250.0, key=3.0)
        located, _, _ = _looked(low, WEST, both_faces=True)
        self.assertEqual((WEST.label, low.label), located.looks_fused)

        expected = 0
        unthinned = 0
        for look in (low, WEST):
            camera = look.tool.to_matrix() @ mount()
            depth, hit = render(camera, (CUBE,))
            mask = hit == 0
            surface = mask & ~pixels_behind_depth_steps(mask, depth)
            thinned = surface & ~grazing_pixels(surface, depth, K)
            expected += to_base_mm(thinned, depth, K, camera).shape[0]
            unthinned += to_base_mm(surface, depth, K, camera).shape[0]
        self.assertLess(expected, unthinned - 100, "the control: the low look has grazing points to thin")
        self.assertEqual(expected, located.objects[0].points_base_mm.shape[0])

    def test_the_looks_are_associated_at_the_wrist_tolerances(self) -> None:
        """Two views of one hand-eye disagree by more the more the wrist turned: the association's wrist constants."""
        from src.robot.grasping.multiview import association

        with mock.patch.object(association, "fuse_scene_clouds", wraps=association.fuse_scene_clouds) as fuse:
            _looked(EAST, WEST, both_faces=True)

        (call,) = fuse.call_args_list
        self.assertEqual(
            (association.AssociationMetric.OVERLAP, association.WRIST_VIEW_MIN_SCORE,
             association.WRIST_VIEW_NEIGHBOUR_MM, association.WRIST_VIEW_SCORE_VOXEL_MM),
            (call.kwargs["metric"], call.kwargs["min_score"], call.kwargs["neighbour_mm"],
             call.kwargs["score_voxel_mm"]))

    def test_the_looks_check_the_hand_eye_on_the_surface_they_share(self) -> None:
        """7.5: the median distance between the looks' surfaces of the part; above 6 mm the calibration may have
        drifted, which is said and changes nothing."""
        consistent, _, _ = _looked(EAST, WEST, both_faces=True)
        assert consistent.hand_eye_gap_mm is not None
        self.assertLess(consistent.hand_eye_gap_mm, 2.0)

        built = _arm(EAST, WEST)
        wrist = _Wrist(built)
        west = WEST.tool.to_matrix()

        def stamped() -> Pose:
            """The tool pose as a drifted hand-eye places the west look: 9 mm off where the camera stood."""
            tool = built.get_tcp_pose()
            if np.allclose(tool.to_matrix(), west, atol=1e-6):
                shifted = west.copy()
                shifted[:3, 3] += (0.0, 0.0, 9.0)
                return Pose.from_matrix(shifted, frame=Frame.BASE)
            return tool

        with self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            drifted = _locator(wrist, tool_pose=stamped).look_around("a part", [EAST.joints, WEST.joints],
                                                                     robot=_robot(built), both_faces=True)
        assert drifted.hand_eye_gap_mm is not None
        self.assertGreater(drifted.hand_eye_gap_mm, 6.0)
        self.assertTrue(any("hand-eye calibration may have drifted" in line for line in said.output), said.output)

    def test_the_hand_eye_check_says_nothing_on_fewer_than_twenty_shared_distances(self) -> None:
        """A second look that shared 16 points of the first look's surface says nothing about the calibration; 36 do."""
        for side, says in ((4, False), (6, True)):
            with self.subTest(points=side * side):
                built = _arm(EAST, WEST)
                located = _Scripted(built, [_patch(6), _patch(side)]).look_around(
                    "a part", [EAST.joints, WEST.joints], robot=_robot(built), both_faces=True)
                self.assertEqual((WEST.label, EAST.label), located.looks_fused, "the control: the two looks fused")
                self.assertEqual(says, located.hand_eye_gap_mm is not None, located.hand_eye_gap_mm)


# ---------------------------------------------------------------------------------------------------
# Refused looks (addendum 2)
# ---------------------------------------------------------------------------------------------------


class _RejectingArm(LookingArm):
    """An arm whose controller rejects the looks in ``rejected`` once the motion may have been sent."""

    rejected: "set[tuple[float, ...]]" = set()

    def move_to_joints(self, joints: JointPositions, **keywords: Any) -> MotionResult:
        key = tuple(round(value, 1) for value in joints.degrees())
        if key in self.rejected:
            self.motions.append(("joints", key))
            return MotionResult.failed(MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_JOINTS,
                                       target_joints=joints, message="Driver reported moveJ() failure")
        return super().move_to_joints(joints, **keywords)


class _StoppedArm(LookingArm):
    """An arm whose controller has protective-stopped."""

    def get_robot_status(self) -> RobotStatus:
        return RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.PROTECTIVE_STOP,
                           protective_stopped=True, emergency_stopped=False, message="PROTECTIVE_STOP")

    def recover_from_protective_stop(self) -> bool:  # pragma: no cover - never called
        raise AssertionError("nothing on the looks may clear a stop")


class ARefusedLookTests(unittest.TestCase):
    def test_a_look_the_planner_refused_before_anything_was_sent_is_skipped_and_said(self) -> None:
        with self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            located, arm, _ = _looked(NORTH, EAST, WEST, refuse=(NORTH.joints,))

        self.assertEqual([_key(NORTH), _key(EAST)], _moved_to(arm))
        self.assertEqual((EAST.label,), located.looks)
        self.assertEqual("", located.refused)
        self.assertTrue(located.objects)
        self.assertTrue(any(NORTH.label in line and "workspace_rejected" in line and "skipped" in line
                            for line in said.output), said.output)

    def test_after_a_valid_grasp_a_refused_look_is_skipped_and_the_views_fused_so_far_stand(self) -> None:
        located, arm, _ = _looked(EAST, NORTH, WEST, refuse=(NORTH.joints,), both_faces=True)
        self.assertEqual([_key(EAST), _key(NORTH), _key(WEST)], _moved_to(arm))
        self.assertEqual((EAST.label, WEST.label), located.looks)
        self.assertEqual((True, True), located.jaw_faces_seen)

    def test_no_look_reached_ends_the_looking_with_nothing_located_and_says_the_first_refusal(self) -> None:
        located, arm, wrist = _looked(EAST, WEST, refuse=(EAST.joints, WEST.joints))

        self.assertEqual([_key(EAST), _key(WEST)], _moved_to(arm))
        self.assertEqual(0, wrist.perceived, "a look the arm never reached located something")
        self.assertEqual((), located.objects)
        self.assertEqual((), located.looks)
        self.assertIn(EAST.label, located.refused)
        self.assertIn("workspace_rejected", located.refused.lower())
        with self.assertRaises(LocatorRefused):
            located.scene(0, _cell())
        # No frame was taken: its stamp is when the looking ended, plain data stays strict JSON, and the text says so.
        self.assertTrue(math.isfinite(located.captured_at_s))
        json.dumps(located.to_dict(), allow_nan=False)
        self.assertNotIn("nan", located.render())
        self.assertIn("no frame was taken", located.render())

    def test_a_look_motion_that_may_have_moved_ends_the_looking_there_with_nothing_else_commanded(self) -> None:
        _RejectingArm.rejected = {_key(WEST)}
        try:
            located, arm, wrist = _looked(EAST, WEST, NORTH, arm=_RejectingArm, both_faces=True)
        finally:
            _RejectingArm.rejected = set()

        self.assertEqual([_key(EAST), _key(WEST)], _moved_to(arm), "a look was commanded after one that may have moved")
        self.assertEqual(1, wrist.perceived)
        self.assertIn(WEST.label, located.refused)
        self.assertIn("controller_rejected", located.refused.lower())
        with self.assertRaises(LocatorRefused):
            located.scene(0, _cell())

    def test_a_stopped_controller_ends_the_looking_there(self) -> None:
        located, arm, _ = _looked(EAST, WEST, arm=_StoppedArm, refuse=(EAST.joints,))
        self.assertEqual([_key(EAST)], _moved_to(arm))
        self.assertIn("controller cannot move", located.refused)
        self.assertIn("nothing else was commanded after it", located.refused)
        self.assertNotIn("so nothing was commanded", located.refused, "the hand verb's words, after a look motion")

    def test_a_look_refused_before_any_command_ends_the_looking_and_says_nothing_was_sent(self) -> None:
        """A link that is not open refuses every motion before its command: the arm stood still, and no look is
        reachable, so the looking ends there; its words must not say the arm may have moved."""
        built = _arm(EAST, WEST)
        built.disconnect()
        wrist = _Wrist(built)

        located = _locator(wrist).look_around("a part", [EAST.joints, WEST.joints], robot=_robot(built))

        self.assertEqual([], _moved_to(built))
        self.assertEqual(0, wrist.perceived)
        self.assertIn(EAST.label, located.refused)
        self.assertIn("before any command", located.refused)
        self.assertNotIn("may have been commanded", located.refused)
        self.assertNotIn("may have moved", located.refused)

    def test_a_camera_that_cannot_vouch_on_the_way_ends_the_looking_there(self) -> None:
        located, arm, _ = _looked(EAST, WEST, NORTH, unvouched=(WEST.joints,), both_faces=True)
        self.assertEqual([_key(EAST), _key(WEST)], _moved_to(arm))
        self.assertIn("could not vouch", located.refused)

    def test_a_list_that_names_no_look_is_refused_before_anything_moves(self) -> None:
        built = _arm(EAST)
        with self.assertRaises(ValueError):
            _locator(_Wrist(built)).look_around("a part", [], robot=_robot(built))
        self.assertEqual([], built.motions)
        self.assertEqual([], built.live_planner_world.calls)

    def test_a_robot_that_keeps_no_robot_section_is_refused_before_anything_moves(self) -> None:
        built = _arm(EAST)
        robot = Robot.from_parts(arm=built, gripper=None, lock_key=None)
        with self.assertRaises(LocatorRefused) as refused:
            _locator(_Wrist(built)).look_around("a part", [EAST.joints], robot=robot)
        self.assertIn("robot section", str(refused.exception))
        self.assertEqual([], built.motions)
        self.assertEqual([], built.live_planner_world.calls, "the looks were held before the refusal")

    def test_a_hand_that_is_no_parallel_jaw_is_refused_before_anything_moves(self) -> None:
        """``Located.scene`` plans jaw grasps only, so a look around cannot judge a suction cup's looks: refused before
        the first look, with nothing held, and a fixed camera asked for both faces the same."""
        config = _replace(_cell(), "grasping.gripper_geometry.kind", "suction")
        built = _arm(EAST)
        with self.assertRaises(LocatorRefused) as refused:
            _locator(_Wrist(built)).look_around("a part", [EAST.joints], robot=_robot(built, config=config))
        self.assertIn("parallel jaw", str(refused.exception))
        self.assertEqual([], built.motions)
        self.assertEqual([], built.live_planner_world.calls)

        owner = _Owner()
        with self.assertRaises(LocatorRefused):
            Locator.from_parts(camera=owner, backend=_backend("red cube")).look_around(
                "a red cube", robot=_robot(built, config=config), both_faces=True)
        fixed = Locator.from_parts(camera=_Owner(), backend=_backend("red cube")).look_around(
            "a red cube", robot=_robot(built, config=config))
        self.assertTrue(fixed.objects, "the control: a fixed camera judges nothing without both_faces, so it locates")


# ---------------------------------------------------------------------------------------------------
# The frames of the looks are held for the pick (addendum 4)
# ---------------------------------------------------------------------------------------------------


class TheFramesOfTheLooksAreHeldForThePickTests(unittest.TestCase):
    def _picked(self, *, refuse: "int | None" = None) -> "tuple[Any, list[str], Any]":
        built = _arm(EAST)
        robot = _robot(built)
        if refuse is not None:
            original = built.move

            def move(pose: Any, **keywords: Any) -> MotionResult:
                built.motions.append(("move", tuple(round(float(v), 1) for v in pose.position_mm)))
                return MotionResult.failed(MotionStatus.WORKSPACE_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                                           message="outside the box")

            built.move = move  # type: ignore[method-assign]
            del original
        located = _locator(_Wrist(built)).look_around("a part", [EAST.joints], robot=robot)
        calls = built.live_planner_world.calls
        self.assertEqual(["hold"], calls, "the looks were not held from the look around on")
        best = located.scene(0, _cell()).grasps().best
        assert best is not None
        report = robot.pick(best.pose(), best.grip_width_mm, keep_out=located.keep_out(0))
        return report, calls, robot

    def test_the_pick_lets_go_of_them_when_it_ends(self) -> None:
        report, calls, _ = self._picked()
        self.assertTrue(report.ok, report.render())
        self.assertEqual(["hold", "forget"], calls)

    def test_a_pick_that_fails_lets_go_of_them_too(self) -> None:
        report, calls, _ = self._picked(refuse=0)
        self.assertFalse(report.ok)
        self.assertEqual(["hold", "forget"], calls)

    def test_a_pick_that_raises_lets_go_of_them_too(self) -> None:
        built = _arm(EAST)
        robot = _robot(built)
        _locator(_Wrist(built)).look_around("a part", [EAST.joints], robot=robot)
        with mock.patch("src.robot.execution.handling.pick", side_effect=RuntimeError("a driver raised")):
            with self.assertRaises(RuntimeError):
                robot.pick(Pose.tool_down(0.0, -700.0, 20.0), 40.0)
        self.assertEqual(["hold", "forget"], built.live_planner_world.calls)

    def test_a_look_around_that_raises_lets_go_of_them(self) -> None:
        """A look around that raises answers nothing, so nothing it began holding may stay: a program that catches the
        error and moves on would have every later motion judged against frames of a pick that never came. As the pick
        loop's look around lets go of its own."""

        class _Crashing(_Wrist):
            """The wrist camera, whose detector raises at the second look."""

            def perceive(self, bgr: Any, prompt: str) -> "tuple[Any, ...]":
                if self.perceived == 1:
                    raise RuntimeError("the detector raised")
                return super().perceive(bgr, prompt)

        built = _arm(EAST, WEST)
        with self.assertRaises(RuntimeError):
            _locator(_Crashing(built)).look_around("a part", [EAST.joints, WEST.joints], robot=_robot(built),
                                                   both_faces=True)
        self.assertEqual([_key(EAST), _key(WEST)], _moved_to(built), "the control: the second look was reached")
        self.assertEqual(["hold", "forget"], built.live_planner_world.calls)

    def test_a_cell_that_disconnects_lets_go_of_them_too(self) -> None:
        """``Cell.connected()`` is the other way a program connects the arm a ``cell.robot`` drives: its exit lets go of
        the looks' frames as ``Robot.connected()``'s does."""
        built = _arm(EAST)
        service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=SimpleNamespace(arm=built, gripper=None)))
        with ConnectedCell(service):
            _locator(_Wrist(built)).look_around("a part", [EAST.joints], robot=_robot(built))
            self.assertEqual(["hold"], built.live_planner_world.calls)
        self.assertEqual(["hold", "forget"], built.live_planner_world.calls)

    def test_a_place_keeps_them_and_a_new_look_around_starts_again(self) -> None:
        built = _arm(EAST)
        robot = _robot(built)
        locator = _locator(_Wrist(built))
        locator.look_around("a part", [EAST.joints], robot=robot)
        robot.place(Pose.tool_down(0.0, -700.0, 150.0))
        self.assertEqual(["hold"], built.live_planner_world.calls, "a place let go of the looks")
        locator.look_around("a part", [EAST.joints], robot=robot)
        self.assertEqual(["hold", "hold"], built.live_planner_world.calls)

    def test_a_disconnect_lets_go_of_them(self) -> None:
        built = _arm(EAST)
        robot = _robot(built)
        with robot.connected():
            _locator(_Wrist(built)).look_around("a part", [EAST.joints], robot=robot)
            self.assertEqual(["hold"], built.live_planner_world.calls)
        self.assertEqual(["hold", "forget"], built.live_planner_world.calls)

    def test_a_fixed_camera_holds_nothing_and_moves_nothing(self) -> None:
        built = _arm(EAST)
        fixed = Locator.from_parts(camera=_Owner(), backend=_backend("red cube"))

        located = fixed.look_around("a red cube", [EAST.joints], robot=_robot(built))

        self.assertEqual([], built.motions)
        self.assertEqual([], built.live_planner_world.calls)
        alone = Locator.from_parts(camera=_Owner(), backend=_backend("red cube")).locate("a red cube")
        np.testing.assert_array_equal(alone.objects[0].points_base_mm, located.objects[0].points_base_mm)
        self.assertEqual(((), (), None, ""), (located.looks, located.looks_fused, located.jaw_faces_seen,
                                               located.refused))
        self.assertEqual(alone.render(), located.render())

    def test_with_both_faces_a_fixed_camera_judges_its_one_frame_the_same_way(self) -> None:
        """A camera straight over a block sees its top alone: neither jaw contact face of its grasp, so both_faces
        refuses it, and still nothing moves."""
        built = _arm(EAST)
        fixed = Locator.from_parts(camera=_Owner(), backend=_backend("red cube"))

        located = fixed.look_around("a red cube", robot=_robot(built), both_faces=True)

        self.assertEqual([], built.motions)
        self.assertEqual([], built.live_planner_world.calls)
        self.assertEqual((False, False), located.jaw_faces_seen)
        self.assertIn("contact faces at jaw 1 and jaw 2 of the chosen grasp were not seen", located.refused)
        with self.assertRaises(LocatorRefused):
            located.scene(0, _cell())


# ---------------------------------------------------------------------------------------------------
# The one generated view and the move back: the pick loop's two functions, straight joint lines only
# ---------------------------------------------------------------------------------------------------


class _OrbitingArm(LookingArm):
    """The looking arm, which also names the configuration of a turn of a look about the cube (``nearest_configuration``:
    the base joint turned as far round the cube as the tool stands from the east look's, the rest of the east look) and
    drives the straight joint line to a configuration, and nothing else, when asked to (``lines``, from where it stood
    to where it went, in degrees). ``blocked`` holds targets whose line its world finds not clear, ``answers`` a result
    a drive gave once it was asked. The planner fails the test when asked: ``move_to_joints`` plans around a line, and
    it may run only to a declared look, once each, never to a generated view and never back to a look. Its live
    planner world holds the frames of the looks; home is the east look."""

    def __init__(self, test: unittest.TestCase, *looks: _Look, **keywords: Any) -> None:
        super().__init__({look.joints: look.tool for look in looks},
                         start=tool_for(camera_looking_at((600.0, 0.0, 0.0), bearing_deg=0.0)), **keywords)
        self.live_planner_world = _World()
        self.test = test
        self.declared = set(self.table)
        self.asked: "set[tuple[float, ...]]" = set()
        self.home_joint_positions = tuple(EAST.joints.tolist())
        self.lines: "list[tuple[tuple[float, ...], tuple[float, ...]]]" = []
        self.blocked: "set[tuple[float, ...]]" = set()
        self.answers: "dict[tuple[float, ...], tuple[MotionStatus, str]]" = {}
        self.screened: list[float] = []

    def nearest_configuration(self, pose: Pose) -> JointPositions:
        def bearing(of: Pose) -> float:
            x, y, _ = (float(v) for v in of.position_mm)
            return math.degrees(math.atan2(y - CUBE_CENTRE[1], x - CUBE_CENTRE[0]))

        turned = round((bearing(pose) - bearing(EAST.tool) + 180.0) % 360.0 - 180.0, 3)
        self.screened.append(turned)
        joints = JointPositions.deg(turned, *EAST.joints.degrees()[1:])
        self.table[joints_key(joints)] = pose
        return joints

    def move_to_joints_on_the_line(self, joints: JointPositions) -> MotionResult:
        key = joints_key(joints)
        self.lines.append((joints_key(self._joints), key))
        self.motions.append(("line", key))
        if key in self.answers:
            status, message = self.answers[key]
            return MotionResult.failed(status, MotionCommand.MOVE_JOINTS, target_joints=joints, message=message)
        if key in self.blocked:
            return MotionResult.failed(
                MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_JOINTS, target_joints=joints,
                message="the planner refused this straight joint line: it passes 4 mm from a held frame's box; nothing "
                        "is planned around it")
        self._joints = joints
        self._tcp = self.table[key]
        return MotionResult.executed(MotionCommand.MOVE_JOINTS, target_joints=joints,
                                     message="move_to_joints_on_the_line")

    def move_to_joints(self, joints: JointPositions, **keywords: Any) -> MotionResult:
        key = joints_key(joints)
        if key not in self.declared or key in self.asked:
            self.test.fail(f"the planner was asked for {key}: the generated view and the move back never plan")
        self.asked.add(key)
        return super().move_to_joints(joints, **keywords)


#: The turns of the east look about the cube that show a contact face of its grasp, as this arm names them.
_PLUS_50 = (48.9, -100.0, -110.0, -60.0, 135.0, 45.0)
_MINUS_50 = (-49.2, -100.0, -110.0, -60.0, 135.0, 45.0)


class TheGeneratedViewAndTheMoveBackTests(unittest.TestCase):
    """Addendum 3: once every declared look was visited and the grasp is still not safe enough, the one generated view,
    the last resort, through the pick loop's own ``generated_view.go_to_generated_view``: screened by the arm before it
    moves, the smallest turn first, driven on the straight joint line alone and never planned around; and the move back
    to the look the part was last seen from on that line (``move_back_to_look``), or the pick approaches from where the
    arm stands. From the east look alone the cube's grasp closes across the two faces it does not face."""

    def test_the_view_turns_to_the_unseen_contact_face_and_both_faces_still_refuses_what_it_did_not_show(self) -> None:
        arm = _OrbitingArm(self, EAST)

        located = _locator(_Wrist(arm)).look_around("a part", [EAST.joints], robot=_robot(arm), both_faces=True)

        self.assertEqual([("joints", _key(EAST)), ("line", _PLUS_50)], arm.motions,
                         "the view was not driven on the straight joint line from the east look, or the arm went on")
        self.assertEqual([(_key(EAST), _PLUS_50)], arm.lines)
        self.assertEqual(50.0, located.generated_view_deg)
        generated = located.looks[-1]
        self.assertTrue(generated.startswith("orbit +50 deg ("), located.looks)
        self.assertEqual((generated, EAST.label), located.looks_fused, "the generated view was not fused")
        # The view showed jaw 2's face; jaw 1's no view showed, and both_faces fails closed after the view too.
        self.assertEqual((False, True), located.jaw_faces_seen)
        self.assertIn("the contact face at jaw 1 of the chosen grasp was not seen", located.refused)
        with self.assertRaises(LocatorRefused):
            located.scene(0, _cell())
        self.assertIn("generated one view", located.render())
        self.assertEqual(50.0, json.loads(json.dumps(located.to_dict()))["generated_view_deg"])

    def test_a_turn_whose_straight_line_is_not_clear_gives_way_to_the_next(self) -> None:
        arm = _OrbitingArm(self, EAST)
        arm.blocked.add(_PLUS_50)

        located = _locator(_Wrist(arm)).look_around("a part", [EAST.joints], robot=_robot(arm), both_faces=True)

        self.assertEqual([(_key(EAST), _PLUS_50), (_key(EAST), _MINUS_50)], arm.lines)
        self.assertEqual(-50.0, located.generated_view_deg)

    def test_a_generated_motion_that_may_have_moved_ends_the_looking_there(self) -> None:
        arm = _OrbitingArm(self, EAST)
        arm.answers[_PLUS_50] = (MotionStatus.CONTROLLER_REJECTED, "the drive refused once moveJ may have been sent")

        located = _locator(_Wrist(arm)).look_around("a part", [EAST.joints], robot=_robot(arm), both_faces=True)

        self.assertEqual([(_key(EAST), _PLUS_50)], arm.lines, "the arm was sent on after a motion that may have moved")
        self.assertIn("did not reach the generated view orbit +50 deg", located.refused)
        self.assertIsNone(located.generated_view_deg)
        with self.assertRaises(LocatorRefused):
            located.scene(0, _cell())

    def test_the_simulators_arm_names_no_configuration_so_no_view_is_generated_and_it_says_why(self) -> None:
        """Addendum 5: the Isaac arm, as the desk arm here, has no ``nearest_configuration``; nothing moves for a view."""
        with self.assertLogs("src.robot.execution.generated_view", level=logging.INFO) as said:
            located, arm, _ = _looked(EAST, both_faces=True)

        self.assertTrue(any("no view is generated" in line and "ChoosesConfigurations" in line
                            for line in said.output), said.output)
        self.assertEqual([_key(EAST)], _moved_to(arm))
        self.assertIsNone(located.generated_view_deg)
        self.assertIn("of the chosen grasp", located.refused)

    def _moved_back(self, arm: _OrbitingArm) -> "tuple[Located, _Look]":
        """East, then west (whose detector calls the part a bolt: the grasp is uncertain), then a look that misses it:
        the part was last seen from the west look, both its contact faces seen, so no view is generated and the arm
        goes back to the west look."""
        far = Box((290.0, -720.0, 0.0), (310.0, -680.0, 30.0), "block")
        wrist = _Wrist(arm, (CUBE, far), labels={1: {"part": "bolt"}})
        return _locator(wrist).look_around("a part", [EAST.joints, WEST.joints, self.ELSEWHERE.joints],
                                           robot=_robot(arm), both_faces=True), self.ELSEWHERE

    ELSEWHERE = _look(180.0, target=(300.0, -700.0, 15.0), key=1.0)

    def test_the_arm_goes_back_to_the_look_the_part_was_last_seen_from_on_the_straight_line(self) -> None:
        arm = _OrbitingArm(self, EAST, WEST, self.ELSEWHERE)

        located, elsewhere = self._moved_back(arm)

        self.assertEqual([("joints", _key(EAST)), ("joints", _key(WEST)), ("joints", _key(elsewhere)),
                          ("line", _key(WEST))], arm.motions)
        self.assertEqual([], arm.screened, "a view was generated for a grasp both of whose faces were seen")
        self.assertEqual("", located.refused)
        self.assertEqual((WEST.label, EAST.label), located.looks_fused)
        np.testing.assert_allclose(located.tool_to_base_mm, WEST.tool.to_matrix(), atol=1e-9)
        self.assertEqual(WEST.tool.to_matrix().tolist(), arm.get_tcp_pose().to_matrix().tolist())

    def test_a_move_back_whose_line_is_not_clear_leaves_the_arm_where_it_stands_and_says_so(self) -> None:
        arm = _OrbitingArm(self, EAST, WEST, self.ELSEWHERE)
        arm.blocked.add(_key(WEST))

        with self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            located, elsewhere = self._moved_back(arm)

        self.assertEqual([(_key(elsewhere), _key(WEST))], arm.lines, "the move back was tried other than once")
        self.assertEqual("", located.refused, "a refused move back is no reason to refuse the part")
        warning = next(line for line in said.output if "move back" in line)
        self.assertIn("approaches from where the arm stands", warning)
        np.testing.assert_allclose(arm.get_tcp_pose().to_matrix(), elsewhere.tool.to_matrix(), atol=1e-9)

    def test_a_look_around_handed_no_looks_generates_nothing(self) -> None:
        arm = _OrbitingArm(self, EAST)
        arm.move_to_joints(EAST.joints)

        located = _locator(_Wrist(arm)).look_around("a part", robot=_robot(arm), both_faces=True)

        self.assertEqual([], arm.screened)
        self.assertEqual([], arm.lines)
        self.assertEqual(("here",), located.looks)

    def test_a_declared_look_refused_is_used_up_and_the_view_still_comes_last(self) -> None:
        """As a pick's looks (addendum 1: 'once the declared looks are used up'): the west look is refused before
        anything was sent after the east look's grasp; it is skipped and said, and the one view is still generated."""
        arm = _OrbitingArm(self, EAST, WEST, refuse=(WEST.joints,))

        with self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            located = _locator(_Wrist(arm)).look_around("a part", [EAST.joints, WEST.joints], robot=_robot(arm),
                                                        both_faces=True)

        self.assertEqual([("joints", _key(EAST)), ("joints", _key(WEST)), ("line", _PLUS_50)], arm.motions)
        self.assertEqual(50.0, located.generated_view_deg)
        warning = next(line for line in said.output if "was refused" in line)
        self.assertIn("once none is left and the grasp is still not safe enough, to the one generated view", warning)

    def test_a_world_that_does_not_hold_the_frames_moves_nothing_the_look_around_made_up(self) -> None:
        """Neither the generated view nor the move back runs unless the arm's world holds every frame of the looks: no
        view is generated, and the pick approaches from where the arm stands, as the pick loop's looks do."""

        class _HoldsNothing(_World):
            def hold_pick_views(self) -> bool:
                self.calls.append("hold")
                return False

        arm = _OrbitingArm(self, EAST)
        arm.live_planner_world = _HoldsNothing()
        with self.assertLogs("src.robot.execution.generated_view", level=logging.INFO) as said:
            located = _locator(_Wrist(arm)).look_around("a part", [EAST.joints], robot=_robot(arm), both_faces=True)

        self.assertEqual([], arm.screened)
        self.assertEqual([], arm.lines)
        self.assertIsNone(located.generated_view_deg)
        self.assertTrue(any("no view is generated" in line and "does not hold the frames of this pick" in line
                            for line in said.output), said.output)

        arm = _OrbitingArm(self, EAST, WEST, self.ELSEWHERE)
        arm.live_planner_world = _HoldsNothing()
        with self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            located, elsewhere = self._moved_back(arm)

        self.assertEqual([], arm.lines, "the move back ran on a world that does not hold the frames of the looks")
        self.assertEqual("", located.refused)
        self.assertTrue(any("move back" in line and "does not hold the frames of this pick" in line
                            for line in said.output), said.output)
        np.testing.assert_allclose(arm.get_tcp_pose().to_matrix(), elsewhere.tool.to_matrix(), atol=1e-9)

    @staticmethod
    def _unvouched(arm: _OrbitingArm, key: "tuple[float, ...]") -> None:
        """The camera cannot vouch for the cell on the straight joint line to ``key``: the arm raises as it is asked."""
        line = arm.move_to_joints_on_the_line

        def move_to_joints_on_the_line(joints: JointPositions) -> MotionResult:
            if joints_key(joints) != key:
                return line(joints)
            arm.motions.append(("line", key))
            raise CameraWorldUnavailable(camera="wrist", verdict="blind", attempts=4,
                                         reason="the wrist camera could not vouch for the cell on the way")

        arm.move_to_joints_on_the_line = move_to_joints_on_the_line  # type: ignore[method-assign]

    def test_a_camera_that_cannot_vouch_on_the_way_to_the_view_ends_the_looking_there(self) -> None:
        arm = _OrbitingArm(self, EAST)
        self._unvouched(arm, _PLUS_50)

        located = _locator(_Wrist(arm)).look_around("a part", [EAST.joints], robot=_robot(arm), both_faces=True)

        self.assertEqual([("joints", _key(EAST)), ("line", _PLUS_50)], arm.motions, "the arm was sent on")
        self.assertIn("did not reach the view generated for the part", located.refused)
        self.assertIn("could not vouch", located.refused)
        self.assertIsNone(located.generated_view_deg)
        with self.assertRaises(LocatorRefused):
            located.scene(0, _cell())

    def test_a_camera_that_cannot_vouch_on_the_way_back_ends_the_looking_there(self) -> None:
        arm = _OrbitingArm(self, EAST, WEST, self.ELSEWHERE)
        self._unvouched(arm, _key(WEST))

        located, elsewhere = self._moved_back(arm)

        self.assertEqual([("joints", _key(EAST)), ("joints", _key(WEST)), ("joints", _key(elsewhere)),
                          ("line", _key(WEST))], arm.motions)
        self.assertIn(f"did not reach look {WEST.label} again", located.refused)
        self.assertIn("could not vouch", located.refused)
        with self.assertRaises(LocatorRefused):
            located.scene(0, _cell())

    def test_the_move_back_drives_to_the_joints_read_at_the_look(self) -> None:
        """A look named ``"home"`` names no joints: the arm's own home verb takes it there, and the move back returns
        to the configuration read off the arm at that look. Home here is the west look; its detector calls the part a
        bolt, so the grasp is uncertain, both its faces seen, and the look after it misses the part."""

        class _HomeIsWest(_OrbitingArm):
            def move_home(self) -> bool:
                self.motions.append(("home", _key(WEST)))
                self._joints = WEST.joints
                self._tcp = WEST.tool
                return True

        arm = _HomeIsWest(self, EAST, WEST, self.ELSEWHERE)
        far = Box((290.0, -720.0, 0.0), (310.0, -680.0, 30.0), "block")
        wrist = _Wrist(arm, (CUBE, far), labels={1: {"part": "bolt"}})

        located = _locator(wrist).look_around("a part", [EAST.joints, "home", self.ELSEWHERE.joints],
                                              robot=_robot(arm), both_faces=True)

        self.assertEqual([("joints", _key(EAST)), ("home", _key(WEST)), ("joints", _key(self.ELSEWHERE)),
                          ("line", _key(WEST))], arm.motions)
        self.assertEqual("", located.refused)
        self.assertEqual(("home", EAST.label), located.looks_fused)
        self.assertEqual(WEST.tool.to_matrix().tolist(), arm.get_tcp_pose().to_matrix().tolist())


class TheFramesOfTheLooksAreKeptOnlyWhenAskedTests(unittest.TestCase):
    """Addendum 7.7: ``record_views`` keeps the frames of the looks for training, through the one writer ``PickRun`` uses
    (``src/robot/execution/record_views.py``); off by default."""

    def test_a_look_around_asked_to_keeps_its_looks_in_the_campaigns_layout(self) -> None:
        from src.robot.execution.record_views import RECORD_VIEWS_FORMAT

        built = _arm(EAST, WEST)
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch("src.robot.execution.record_views.RECORD_VIEWS_DIR", folder):
            located = _locator(_Wrist(built)).look_around("a part", [EAST.joints, WEST.joints], robot=_robot(built),
                                                          both_faces=True, record_views=True)
            written = Path(located.views_file)
            self.assertEqual([written], sorted(Path(folder).glob("*.npz")))
            self.assertTrue(written.name.startswith("look_around-a_part"), written.name)
            self.assertIn(written.name, located.render())
            with np.load(written, allow_pickle=False) as views:
                self.assertEqual(RECORD_VIEWS_FORMAT, int(views["format"]))
                self.assertEqual([EAST.label, WEST.label], views["labels"].tolist())
                self.assertEqual([f"wrist@{EAST.label}", f"wrist@{WEST.label}"], views["views"].tolist())
                for index, look in enumerate((EAST, WEST)):
                    self.assertEqual((HEIGHT, WIDTH, 3), views[f"rgb_{index}"].shape)
                    self.assertEqual((HEIGHT, WIDTH), views[f"depth_{index}"].shape)
                    np.testing.assert_allclose(K, views[f"intrinsics_{index}"])
                    np.testing.assert_allclose(look.tool.to_matrix(), views[f"tool_to_base_{index}"], atol=1e-6)
                    np.testing.assert_allclose(look.tool.to_matrix() @ mount(), views[f"camera_to_base_{index}"],
                                               atol=1e-6)
                np.testing.assert_allclose(located.objects[0].points_base_mm, views["target_cloud_base_mm"], atol=1e-3)

    def test_off_by_default_nothing_is_kept(self) -> None:
        built = _arm(EAST)
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch("src.robot.execution.record_views.RECORD_VIEWS_DIR", folder):
            located = _locator(_Wrist(built)).look_around("a part", [EAST.joints], robot=_robot(built))
            self.assertEqual([], list(Path(folder).iterdir()))
        self.assertEqual("", located.views_file)

    def test_a_file_that_cannot_be_written_is_said_and_the_look_around_answers_all_the_same(self) -> None:
        built = _arm(EAST)
        with mock.patch("src.robot.execution.record_views.record_views", side_effect=OSError("disk full")), \
                self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            located = _locator(_Wrist(built)).look_around("a part", [EAST.joints], robot=_robot(built),
                                                          record_views=True)
        self.assertEqual("", located.views_file)
        self.assertEqual("", located.refused)
        self.assertTrue(any("disk full" in line for line in said.output), said.output)


# ---------------------------------------------------------------------------------------------------
# What a looked-around Located says
# ---------------------------------------------------------------------------------------------------


class ALookedAroundLocatedSaysItsLooksTests(unittest.TestCase):
    def test_in_text_and_in_plain_data(self) -> None:
        located, _, _ = _looked(EAST, WEST, both_faces=True)

        text = located.render()
        self.assertEqual(text, str(located))
        self.assertIn(f"looked from 2 look(s): {EAST.label}, {WEST.label}", text)
        self.assertIn(f"{WEST.label}, {EAST.label}", text)
        self.assertIn("jaw 1 seen, jaw 2 seen", text)
        self.assertIn("hand-eye", text)
        data = json.loads(json.dumps(located.to_dict()))
        self.assertEqual([EAST.label, WEST.label], data["looks"])
        self.assertEqual([WEST.label, EAST.label], data["looks_fused"])
        self.assertEqual([True, True], data["jaw_faces_seen"])
        self.assertEqual("", data["refused"])
        self.assertIsInstance(data["hand_eye_gap_mm"], float)

    def test_one_locate_says_none_of_it(self) -> None:
        located = Locator.from_parts(camera=_Owner(), backend=_backend("red cube")).locate("a red cube")
        data = located.to_dict()
        self.assertEqual(([], [], None, "", None), (data["looks"], data["looks_fused"], data["jaw_faces_seen"],
                                                    data["refused"], data["hand_eye_gap_mm"]))
        self.assertNotIn("looked from", located.render())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
