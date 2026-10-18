"""Where no turn of the judged look can be driven, the generated view tries other angles of it (the owner, 2026-10-16:
"Mehr Blickwinkel").

On Thursday's cell every view the pick generated to show the side no look faced was refused, 4 times of 4: each of its
15 turns about the part was out of the arm's reach or its workspace box. A turn keeps the camera's height and distance;
another angle lowers or raises the camera about the part (15 and 30 degrees flatter, 15 steeper), brings it nearer or
farther (0.85 and 1.15 of its distance), and turns it as every turn does (``unseen_side.other_angles``). Pinned:

* the geometry: camera and tool stay one rigid body, the part stays at its pixel, the camera stands as far over the
  horizontal as asked and as far from the part, and nothing flatter than 15 or steeper than 80 degrees is offered;
* the order: the angle nearest the judged look first, a change of distance before a change of angle, the smaller first;
* a face that looks down 30 degrees is shown only by a camera lowered about the part, never by a turn, 15 degrees
  lower at the incidence limit and 30 lower more squarely;
* ``go_to_generated_view``: every plain turn first, each screened as before; where the arm refuses every one, the other
  angles, screened and capped as every turn is, and the arm drives the first it admits on its straight joint line; where
  none can be driven it says no view could be.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.geometry import Pose
from src.robot.core import JointPositions
from src.robot.core.errors import RobotKinematicsError
from src.robot.grasping.multiview.unseen_side import (
    DISTANCE_SCALES,
    LEAST_ELEVATION_DEG,
    MOST_ELEVATION_DEG,
    orbit_views,
    other_angles,
)
from tests.robot.execution.test_the_generated_view_moves_on_its_straight_joint_line_only import _LOOK_TOOL, _Arm, _go
from tests.robot.grasping.multiview.test_the_generated_view_orbits_to_the_unseen_face import _IMAGE, _LENS, _TARGET, _look, _pixel


def _elevation(camera: np.ndarray) -> float:
    sight = camera[:3, 3] - _TARGET
    return math.degrees(math.atan2(float(sight[2]), math.hypot(float(sight[0]), float(sight[1]))))


def _distance(camera: np.ndarray) -> float:
    return float(np.linalg.norm(camera[:3, 3] - _TARGET))


def _angles(faces, *, elevation_deg: float = 45.0, **kwargs):
    camera, tool = _look(0.0, elevation_deg)
    return camera, tool, other_angles(target_mm=_TARGET, camera_to_base_mm=camera, tool_to_base_mm=tool, faces=faces,
                                      intrinsics=_LENS, image_size=_IMAGE, **kwargs)


#: A vertical face on the part's far side, as the side no look faced is aimed at: 180 degrees round from the look.
_FAR_SIDE = [(_TARGET, np.array([-1.0, 0.0, 0.0]))]
#: A face on the camera's side looking 30 degrees down: past the incidence limit from 45 degrees up.
_LOOKS_DOWN = [(_TARGET, np.array([math.cos(math.radians(30.0)), 0.0, -math.sin(math.radians(30.0))]))]


class TheOtherAnglesTests(unittest.TestCase):
    def test_camera_and_tool_stay_one_body_and_the_part_its_pixel(self) -> None:
        camera, tool, views = _angles(_FAR_SIDE)
        self.assertTrue(views)
        held = np.linalg.inv(camera) @ tool
        pixel = _pixel(camera, _TARGET)
        for view in views:
            np.testing.assert_allclose(np.linalg.inv(view.camera_to_base_mm) @ view.tool_to_base_mm, held, atol=1e-9)
            np.testing.assert_allclose(_pixel(view.camera_to_base_mm, _TARGET), pixel, atol=1e-6)

    def test_the_camera_stands_as_far_over_the_horizontal_and_from_the_part_as_asked(self) -> None:
        camera, _tool, views = _angles(_FAR_SIDE)
        for view in views:
            self.assertAlmostEqual(_elevation(camera) + view.elevation_deg, _elevation(view.camera_to_base_mm), 6)
            self.assertAlmostEqual(_distance(camera) * view.distance_scale, _distance(view.camera_to_base_mm), 6)
            self.assertTrue(LEAST_ELEVATION_DEG <= _elevation(view.camera_to_base_mm) <= MOST_ELEVATION_DEG)
        self.assertEqual({0.0, -15.0, -30.0, 15.0}, {view.elevation_deg for view in views})
        self.assertEqual(set(DISTANCE_SCALES), {view.distance_scale for view in views})
        self.assertFalse([view for view in views if not view.elevation_deg and view.distance_scale == 1.0],
                         "a plain turn is orbit_views', not another angle")

    def test_the_nearest_angle_comes_first(self) -> None:
        _camera, _tool, views = _angles(_FAR_SIDE)
        keys = [(abs(view.elevation_deg), abs(view.distance_scale - 1.0), abs(view.azimuth_deg)) for view in views]
        self.assertEqual(sorted(keys), keys)
        self.assertEqual(0.0, views[0].elevation_deg, "a change of angle came before a change of distance")
        self.assertIn(views[0].distance_scale, (0.85, 1.15))
        self.assertIn("x as far", views[0].label)

    def test_nothing_flatter_than_15_or_steeper_than_80_degrees_is_offered(self) -> None:
        _camera, _tool, low = _angles(_FAR_SIDE, elevation_deg=25.0)
        self.assertNotIn(-15.0, {view.elevation_deg for view in low}, "a camera 10 degrees over the part was offered")
        self.assertNotIn(-30.0, {view.elevation_deg for view in low})
        _camera, _tool, high = _angles(_FAR_SIDE, elevation_deg=70.0)
        self.assertNotIn(15.0, {view.elevation_deg for view in high}, "a camera 85 degrees over the part was offered")

    def test_a_face_that_looks_down_shows_only_to_a_camera_lowered_about_the_part(self) -> None:
        camera, tool, views = _angles(_LOOKS_DOWN)
        turns = orbit_views(target_mm=_TARGET, camera_to_base_mm=camera, tool_to_base_mm=tool, faces=_LOOKS_DOWN,
                            intrinsics=_LENS, image_size=_IMAGE)
        self.assertEqual((), turns, "a turn at the look's height showed a face that looks down 30 degrees")
        self.assertTrue(views)
        self.assertTrue(all(view.elevation_deg < 0.0 for view in views), [view.label for view in views])
        self.assertIn("15 deg flatter", views[0].label, "15 degrees lower meets the incidence limit already")
        flatter = [view for view in views if view.elevation_deg == -30.0 and view.distance_scale == 1.0]
        self.assertTrue(flatter and flatter[0].incidence_deg < views[0].incidence_deg,
                        "30 degrees lower does not show the face more squarely")


class _OnlyOtherAngles(_Arm):
    """The desk arm of the generated view's tests, whose inverse kinematics refuses every pose at the judged look's
    height, as the owner's arm refused every turn of Thursday's looks: only another angle can be reached."""

    def nearest_configuration(self, pose: Pose) -> JointPositions:
        if abs(float(pose.position_mm[2]) - float(_LOOK_TOOL.position_mm[2])) < 1.0:
            self.screened.append(round(float(pose.position_mm[2]), 1))
            raise RobotKinematicsError("refused by the inverse kinematics: the flange goal is out of the arm's reach")
        return super().nearest_configuration(pose)


class _NothingReached(_Arm):
    def nearest_configuration(self, pose: Pose) -> JointPositions:
        raise RobotKinematicsError("refused by the inverse kinematics: the flange goal is out of the arm's reach")


class TheGeneratedViewTriesTheOtherAnglesTests(unittest.TestCase):
    def test_where_every_turn_is_refused_the_first_other_angle_the_arm_admits_is_driven(self) -> None:
        arm = _OnlyOtherAngles(self)

        view = _go(arm)

        self.assertTrue(view.moved, view.skipped)
        self.assertIn("x as far", view.label)
        self.assertEqual(1, len(arm.lines), "more than the one straight joint line ran")
        refused = [why for _azimuth, why in view.tried]
        self.assertTrue(refused and all("out of the arm's reach" in why for why in refused), refused)

    def test_where_no_angle_at_all_can_be_driven_it_says_no_view_could_be(self) -> None:
        view = _go(_NothingReached(self))

        self.assertFalse(view.moved)
        self.assertIn("no view that shows", view.skipped)
        self.assertIn("could be driven", view.skipped)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
