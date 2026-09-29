"""The one view a wrist pick generates turns its judged look about the part until the unseen face shows.

When the declared looks are used up and a jaw face is still unmeasured, the pick loop asks
``orbit_views`` where the camera would have to stand to see it. The answer is a rigid turn of the whole
wrist about the vertical through the target, so the camera keeps its distance, its tilt and its aim,
and the only thing that changes is which side of the part faces it. These tests hold that geometry to
analytic expectations: a face at incidence iota from a camera at elevation e, turned by delta about
the vertical, is seen at cos(iota) = cos(e) cos(delta), so the least turn that shows it is known in
closed form and the candidate list must start there.

Pure geometry: no arm, no IK and no planner. Which of these candidates the arm can reach is a later
question, asked of ``nearest_configuration`` before anything moves.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.geometry import Frame
from src.robot.grasping.multiview.association import (
    DEFAULT_MIN_SCORE,
    DEFAULT_NEIGHBOUR_MM,
    WRIST_VIEW_MIN_SCORE,
    WRIST_VIEW_NEIGHBOUR_MM,
    WRIST_VIEW_SCORE_VOXEL_MM,
    AssociationMetric,
    ViewCandidates,
    assign_view,
)
from src.robot.grasping.multiview.unseen_side import (
    MAX_INCIDENCE_DEG,
    ORBIT_MAX_DEG,
    ORBIT_STEP_DEG,
    away_from_views,
    orbit_target,
    orbit_views,
)

_TARGET = np.array([500.0, 0.0, 20.0])
_RANGE_MM = 450.0
#: A D415 colour stream at 1280 x 720, about the lens a wrist frame carries.
_LENS = np.array([[920.0, 0.0, 640.0], [0.0, 920.0, 360.0], [0.0, 0.0, 1.0]])
_IMAGE = (1280, 720)


def _rotation_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


#: A camera tilted 45 degrees beside the tool, 70 mm off its axis: the owner's hand-eye in shape.
_CAMERA_TO_TOOL = np.eye(4)
_CAMERA_TO_TOOL[:3, :3] = _rotation_x(-45.0)
_CAMERA_TO_TOOL[:3, 3] = (0.0, 70.0, -40.0)


def _horizontal(bearing_deg: float) -> np.ndarray:
    b = math.radians(bearing_deg)
    return np.array([math.cos(b), math.sin(b), 0.0])


def _look(bearing_deg: float, elevation_deg: float,
          target: np.ndarray = _TARGET) -> tuple[np.ndarray, np.ndarray]:
    """(camera_to_base, tool_to_base) of a wrist camera at that bearing and elevation, aimed at the
    target."""
    e = math.radians(elevation_deg)
    up = np.array([0.0, 0.0, math.sin(e)])
    centre = target + _RANGE_MM * (math.cos(e) * _horizontal(bearing_deg) + up)
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    camera = np.eye(4)
    camera[:3, :3] = np.column_stack([right, down, forward])
    camera[:3, 3] = centre
    return camera, camera @ np.linalg.inv(_CAMERA_TO_TOOL)


def _views(faces, *, bearing_deg: float = 0.0, elevation_deg: float = 30.0, **kwargs):
    camera, tool = _look(bearing_deg, elevation_deg)
    kwargs.setdefault("intrinsics", _LENS)
    kwargs.setdefault("image_size", _IMAGE)
    return orbit_views(target_mm=_TARGET, camera_to_base_mm=camera, tool_to_base_mm=tool,
                       faces=faces, **kwargs)


def _pixel(camera_to_base: np.ndarray, point: np.ndarray) -> np.ndarray:
    local = np.linalg.inv(camera_to_base) @ np.append(point, 1.0)
    return (_LENS @ (local[:3] / local[2]))[:2]


def _bearing(point: np.ndarray, target: np.ndarray) -> float:
    return math.degrees(math.atan2(point[1] - target[1], point[0] - target[0])) % 360.0


def _apart(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _least_turn(delta0_deg: float, elevation_deg: float) -> float:
    """The analytic least turn, in steps, that brings a face delta0 round from a camera at this
    elevation within the incidence limit: cos(limit) = cos(e) cos(delta)."""
    limit, elevation = math.radians(MAX_INCIDENCE_DEG), math.radians(elevation_deg)
    reach = math.degrees(math.acos(math.cos(limit) / math.cos(elevation)))
    return math.ceil((abs(delta0_deg) - reach) / ORBIT_STEP_DEG - 1e-9) * ORBIT_STEP_DEG


def _cube(centre: np.ndarray, size: float = 40.0) -> list[tuple[np.ndarray, np.ndarray]]:
    half = size / 2.0
    axis = np.arange(-half, half + 1e-9, 1.0)
    u, v = (a.ravel() for a in np.meshgrid(axis, axis, indexing="ij"))
    faces = []
    for k in range(3):
        i, j = (n for n in range(3) if n != k)
        for sign in (-1.0, 1.0):
            local = np.zeros((u.size, 3))
            local[:, i], local[:, j], local[:, k] = u, v, sign * half
            faces.append((local + centre, np.eye(3)[k] * sign))
    return faces


def _seen(faces: list[tuple[np.ndarray, np.ndarray]], camera: np.ndarray) -> np.ndarray:
    return np.vstack([p for p, n in faces if float(n @ (camera - p.mean(axis=0))) > 1e-6])


class GeneratedViewOrbitTests(unittest.TestCase):
    def test_the_smallest_azimuth_that_shows_the_missing_face_comes_first(self) -> None:
        """A face turned 100 degrees from a camera 30 degrees up comes within 60 degrees of incidence
        once the turn leaves it 54.7 degrees off, so the first candidate is the first step past 45.3."""
        views = _views([(_TARGET, _horizontal(100.0))], elevation_deg=30.0)
        expected = _least_turn(100.0, 30.0)
        self.assertEqual(expected, 50.0)
        self.assertEqual(views[0].azimuth_deg, expected)
        self.assertNotIn(expected - ORBIT_STEP_DEG, [v.azimuth_deg for v in views])
        cos_iota = math.cos(math.radians(30.0)) * math.cos(math.radians(100.0 - expected))
        self.assertAlmostEqual(views[0].incidence_deg, math.degrees(math.acos(cos_iota)), places=6)
        self.assertTrue(all(v.azimuth_deg > 0.0 for v in views), "the other way round is 205 degrees")
        self.assertEqual([abs(v.azimuth_deg) for v in views], sorted(abs(v.azimuth_deg) for v in views))

    def test_both_directions_are_offered_nearest_first(self) -> None:
        """A camera looking across the closing axis sees neither jaw face; one lies each way round.
        Equal turns are ordered by incidence, the squarer view first."""
        faces = [(_TARGET, _horizontal(90.0)), (_TARGET, _horizontal(-85.0))]
        views = _views(faces, elevation_deg=40.0)
        self.assertEqual([v.azimuth_deg for v in views[:3]], [-40.0, -45.0, 45.0])
        self.assertLess(views[1].incidence_deg, views[2].incidence_deg)
        self.assertEqual([abs(v.azimuth_deg) for v in views], sorted(abs(v.azimuth_deg) for v in views))
        self.assertIn(50.0, [v.azimuth_deg for v in views])
        self.assertIn(-50.0, [v.azimuth_deg for v in views])

    def test_nothing_beyond_half_a_turn_and_the_face_behind_a_look_is_offered(self) -> None:
        """The owner, 2026-09-29: the generated view may turn up to half a turn, as the very last resort. The face
        directly behind a 45 degree look needs 135 degrees, which the old 120 degree bound never offered; the
        safeguards past 120 degrees (the travel cap, the warning, the straight joint line) are the caller's."""
        self.assertEqual(ORBIT_MAX_DEG, 180.0)
        # A face 100 degrees round from a camera 30 degrees up stays within the incidence limit up to 154.7 degrees.
        views = _views([(_TARGET, _horizontal(100.0))], elevation_deg=30.0)
        self.assertEqual(max(abs(v.azimuth_deg) for v in views), 150.0)
        far = [(_TARGET, _horizontal(180.0))]
        self.assertEqual(_least_turn(180.0, 45.0), 135.0)
        self.assertEqual(sorted(abs(v.azimuth_deg) for v in _views(far, elevation_deg=45.0)[:2]), [135.0, 135.0])
        self.assertEqual(_views(far, elevation_deg=45.0, max_deg=120.0), (), "a caller may still ask for less")

    def test_a_look_steeper_than_the_incidence_limit_generates_no_view(self) -> None:
        """Seen from 70 degrees up, a vertical face is never squarer than 70 degrees, whatever the
        turn."""
        for bearing in (0.0, 90.0, 180.0):
            with self.subTest(face=bearing):
                self.assertEqual(_views([(_TARGET, _horizontal(bearing))], elevation_deg=70.0), ())

    def test_the_camera_keeps_distance_tilt_and_aim(self) -> None:
        camera0, tool0 = _look(0.0, 30.0)
        views = _views([(_TARGET, _horizontal(100.0))], elevation_deg=30.0)
        self.assertGreater(len(views), 3)
        pixel0 = _pixel(camera0, _TARGET)
        for view in views:
            with self.subTest(azimuth=view.azimuth_deg):
                camera, tool = view.camera_to_base_mm, view.tool_to_base_mm
                self.assertAlmostEqual(float(np.linalg.norm(camera[:3, 3] - _TARGET)), _RANGE_MM, places=6)
                self.assertAlmostEqual(camera[2, 2], camera0[2, 2], places=12, msg="the tilt")
                self.assertAlmostEqual(camera[2, 3], camera0[2, 3], places=9, msg="the height")
                np.testing.assert_allclose(_pixel(camera, _TARGET), pixel0, atol=1e-6)
                # One rigid body: the camera stays where the hand-eye puts it on the tool.
                np.testing.assert_allclose(np.linalg.inv(tool) @ camera, np.linalg.inv(tool0) @ camera0,
                                           atol=1e-9)
                self.assertAlmostEqual(_apart(_bearing(camera[:3, 3], _TARGET), 0.0),
                                       abs(view.azimuth_deg), places=6)
                pose = view.pose()
                self.assertIs(pose.frame, Frame.BASE)
                np.testing.assert_allclose(pose.to_matrix(), tool, atol=1e-9)

    def test_without_a_grasp_the_view_turns_away_from_the_views_so_far(self) -> None:
        """No grasp names a face, so the unseen side is the one no look has faced: the bisector of the
        widest gap between the looks' bearings. Two looks from 180 and 270 degrees leave 45 degrees."""
        centre = np.array([500.0, 0.0, 20.0])
        cube = _cube(centre)
        cameras = [_look(b, 40.0, centre)[0][:3, 3] for b in (180.0, 270.0)]
        cloud = np.vstack([_seen(cube, c) for c in cameras])
        strays = np.tile([[700.0, 150.0, 20.0]], (5, 1))
        target = orbit_target(np.vstack([cloud, strays]))
        np.testing.assert_allclose(target[:2], centre[:2], atol=2.0)
        self.assertGreater(abs(float(np.median(cloud[:, 0])) - centre[0]), 5.0,
                           "a plain median leans toward the sides the looks saw")
        away = away_from_views(target, cameras)
        np.testing.assert_allclose(away, _horizontal(45.0), atol=1e-6)
        camera, tool = _look(270.0, 40.0, target)
        views = orbit_views(target_mm=target, camera_to_base_mm=camera, tool_to_base_mm=tool,
                            faces=[(target, away)], intrinsics=_LENS, image_size=_IMAGE)
        self.assertEqual(views[0].azimuth_deg, _least_turn(135.0, 40.0))
        new = _bearing(views[0].camera_to_base_mm[:3, 3], target)
        self.assertGreaterEqual(min(_apart(new, 180.0), _apart(new, 270.0)), 90.0 - 1e-6)
        self.assertEqual(orbit_target(cloud, grasp_centre_mm=(1.0, 2.0, 3.0)).tolist(), [1.0, 2.0, 3.0])

    def test_a_single_look_turns_to_its_opposite(self) -> None:
        camera = _look(30.0, 45.0)[0][:3, 3]
        np.testing.assert_allclose(away_from_views(_TARGET, [camera]), _horizontal(210.0), atol=1e-9)
        with self.assertRaises(ValueError):
            away_from_views(_TARGET, [_TARGET + (0.0, 0.0, 400.0)])

    def test_looks_from_one_bearing_are_one_side(self) -> None:
        """Two looks from the same bearing, at another height or tilt or the same pose twice, face one
        side: the gap between them is no gap, not a full turn. The answer is the bisector of the widest
        real gap, as far as it can be from every look."""
        def at(bearing_deg: float, height_mm: float) -> np.ndarray:
            return _TARGET + 400.0 * _horizontal(bearing_deg) + (0.0, 0.0, height_mm)

        cases = (([0.0, 0.0, 90.0], 225.0), ([0.0, 90.0, 90.0], 225.0),
                 ([10.0, 10.0, 10.0, 200.0], 105.0), ([30.0, 30.0], 210.0))
        for bearings, expected in cases:
            with self.subTest(bearings=bearings):
                cameras = [at(b, 250.0 + 60.0 * i) for i, b in enumerate(bearings)]
                away = away_from_views(_TARGET, cameras)
                np.testing.assert_allclose(away, _horizontal(expected), atol=1e-9)
                self.assertAlmostEqual(float(np.linalg.norm(away)), 1.0, places=12)

    def test_a_tie_between_gaps_does_not_depend_on_the_order_of_the_looks(self) -> None:
        """Two opposite looks leave two equal gaps. The one that starts at the lower bearing wins,
        whichever look came first."""
        east, west = _TARGET + (400.0, 0.0, 300.0), _TARGET + (-400.0, 0.0, 300.0)
        for cameras in ([east, west], [west, east]):
            np.testing.assert_allclose(away_from_views(_TARGET, cameras), _horizontal(90.0), atol=1e-9)

    def test_a_bottom_face_is_never_offered(self) -> None:
        """A vertical closing axis closes on the bottom, which the table hides from every camera, and on
        the top, which the orbit shows no better than the look it turned from."""
        bottom = (_TARGET - (0.0, 0.0, 20.0), np.array([0.0, 0.0, -1.0]))
        top = (_TARGET + (0.0, 0.0, 20.0), np.array([0.0, 0.0, 1.0]))
        for elevation in (10.0, 30.0, 45.0):
            with self.subTest(elevation=elevation):
                self.assertEqual(_views([bottom], elevation_deg=elevation), ())
                self.assertEqual(_views([top], elevation_deg=elevation), ())
                self.assertEqual(_views([bottom, top], elevation_deg=elevation), ())

    def test_a_face_tilted_off_vertical_is_offered_and_a_nearly_vertical_one_is_not(self) -> None:
        """A face whose normal lies within 5 degrees of vertical is a top or a bottom to the orbit, and
        turning about the vertical does not show it better. One tilted 10 degrees toward the camera
        is a face the turn can move, and at the look's 30 degrees up the first step shows it at 50
        degrees. One tilted 3 degrees would pass the incidence limit at 57 degrees but is a top."""
        def tilted(deg: float) -> np.ndarray:
            return math.sin(math.radians(deg)) * _horizontal(0.0) + (0.0, 0.0, math.cos(math.radians(deg)))

        views = _views([(_TARGET, tilted(10.0))], elevation_deg=30.0)
        self.assertEqual([v.azimuth_deg for v in views[:2]], [ORBIT_STEP_DEG, -ORBIT_STEP_DEG])
        self.assertLess(views[0].incidence_deg, MAX_INCIDENCE_DEG)
        nearly = tilted(3.0)
        camera = _look(ORBIT_STEP_DEG, 30.0)[0][:3, 3]
        cosine = float(nearly @ (camera - _TARGET)) / float(np.linalg.norm(camera - _TARGET))
        self.assertLess(math.degrees(math.acos(cosine)), MAX_INCIDENCE_DEG, "the limit alone would pass it")
        self.assertEqual(_views([(_TARGET, nearly)], elevation_deg=30.0), ())

    def test_equal_turns_at_equal_incidence_offer_the_positive_turn_first(self) -> None:
        """A face square to the look is shown alike by the turns either way. The order is total all the
        same: the positive turn first, so the candidate the pick moves to does not hang on how the
        list was built."""
        views = _views([(_TARGET, _horizontal(0.0))], elevation_deg=30.0)
        self.assertEqual([v.azimuth_deg for v in views[:2]], [ORBIT_STEP_DEG, -ORBIT_STEP_DEG])
        self.assertEqual(views[0].incidence_deg, views[1].incidence_deg)

    def test_a_face_behind_the_turned_camera_is_not_offered(self) -> None:
        """A face 200 mm behind the camera, turned toward it, is at a good incidence from every small
        turn but not in front of the lens; straight behind, it would even project onto the principal
        point. It is offered first once the turn brings it in front: cos(phi) < (450 / 650 - 0.25) /
        0.75 at 30 degrees up, first at 55 degrees."""
        camera0 = _look(0.0, 30.0)[0]
        back = camera0[:3, 3] - _TARGET
        face = (camera0[:3, 3] + 200.0 * back / np.linalg.norm(back), _horizontal(180.0))
        first = ORBIT_STEP_DEG * math.ceil(math.degrees(math.acos((450.0 / 650.0 - 0.25) / 0.75))
                                           / ORBIT_STEP_DEG)
        self.assertEqual(first, 55.0)
        for lens in (True, False):
            with self.subTest(lens=lens):
                views = (_views([face], elevation_deg=30.0) if lens else
                         _views([face], elevation_deg=30.0, intrinsics=None, image_size=None))
                self.assertNotIn(ORBIT_STEP_DEG, [v.azimuth_deg for v in views])
                for view in views:
                    local = np.linalg.inv(view.camera_to_base_mm) @ np.append(face[0], 1.0)
                    self.assertGreater(local[2], 0.0, view.label)
                if not lens:
                    self.assertEqual(min(abs(v.azimuth_deg) for v in views), first)

    def test_a_turn_past_half_a_turn_is_refused_and_half_a_turn_is_offered_once(self) -> None:
        """Plus and minus 180 degrees are one pose, and 185 is -175: a bound past half a turn names
        turns twice."""
        far = [(_TARGET, _horizontal(180.0))]
        half = _views(far, elevation_deg=45.0, max_deg=180.0)
        azimuths = [v.azimuth_deg for v in half]
        self.assertEqual(azimuths.count(180.0) + azimuths.count(-180.0), 1)
        self.assertEqual(len(azimuths), len(set(azimuths)))
        with self.assertRaises(ValueError):
            _views(far, elevation_deg=45.0, max_deg=185.0)

    def test_a_face_outside_the_image_is_not_offered(self) -> None:
        face = [(_TARGET + (0.0, 0.0, -300.0), _horizontal(100.0))]
        self.assertGreater(len(_views(face, intrinsics=None, image_size=None)), 0)
        self.assertEqual(_views(face), (), "300 mm below the aim is below the bottom of the image")
        with self.assertRaises(ValueError):
            _views(face, image_size=None)

    def test_the_generated_view_associates_with_the_look_it_turned_from(self) -> None:
        """Two wrist views a quarter turn apart share the top of the part. At the wrist tolerances they
        agree it is one part through 8 mm of hand-eye drift between them, drift toward a neighbour 3 mm
        away, and that neighbour still matches only the rim along the side it stands beside, under the
        gate."""
        self.assertEqual(WRIST_VIEW_MIN_SCORE, DEFAULT_MIN_SCORE)
        self.assertGreater(WRIST_VIEW_NEIGHBOUR_MM, DEFAULT_NEIGHBOUR_MM)
        self.assertLess(WRIST_VIEW_SCORE_VOXEL_MM, WRIST_VIEW_NEIGHBOUR_MM)
        centre = np.array([500.0, 0.0, 20.0])
        part, neighbour = _cube(centre), _cube(centre + (43.0, 0.0, 0.0))
        first = _look(180.0, 45.0, centre)[0][:3, 3]
        turned = _look(270.0, 45.0, centre)[0][:3, 3]
        drift = np.array([-5.0, -5.0, 4.0])
        clouds = (_seen(neighbour, turned) + drift, _seen(part, turned) + drift)
        view = ViewCandidates("wrist@orbit +90 deg", clouds)
        match = assign_view([_seen(part, first)], view, metric=AssociationMetric.OVERLAP,
                            min_score=WRIST_VIEW_MIN_SCORE, neighbour_mm=WRIST_VIEW_NEIGHBOUR_MM,
                            score_voxel_mm=WRIST_VIEW_SCORE_VOXEL_MM)
        self.assertEqual(match.assignment, (1,))
        self.assertGreater(match.scores[0][0], 0.0, "the neighbour's rim is within reach of the part")
        self.assertLess(match.scores[0][0], WRIST_VIEW_MIN_SCORE)
        self.assertGreater(match.scores[0][1], 0.5)

    def test_the_wrist_radius_keeps_a_drift_the_fixed_cameras_radius_loses(self) -> None:
        """The same two views 14 mm apart, 13 of it in height: past the 12 mm the fixed cameras'
        radius spans, so there the part scores under the gate and the generated view is left
        unassigned, and inside the wrist's 15 mm, where it scores well over it and the neighbour still
        does not."""
        centre = np.array([500.0, 0.0, 20.0])
        part, neighbour = _cube(centre), _cube(centre + (43.0, 0.0, 0.0))
        first = _seen(part, _look(180.0, 45.0, centre)[0][:3, 3])
        turned = _look(270.0, 45.0, centre)[0][:3, 3]
        drift = np.array([0.0, -5.0, 13.0])
        view = ViewCandidates("wrist@orbit +90 deg", (_seen(neighbour, turned) + drift,
                                                       _seen(part, turned) + drift))
        fixed = assign_view([first], view, metric=AssociationMetric.OVERLAP, min_score=DEFAULT_MIN_SCORE,
                            neighbour_mm=DEFAULT_NEIGHBOUR_MM)
        wrist = assign_view([first], view, metric=AssociationMetric.OVERLAP,
                            min_score=WRIST_VIEW_MIN_SCORE, neighbour_mm=WRIST_VIEW_NEIGHBOUR_MM,
                            score_voxel_mm=WRIST_VIEW_SCORE_VOXEL_MM)
        self.assertEqual(fixed.assignment, (None,))
        self.assertLess(fixed.scores[0][1], DEFAULT_MIN_SCORE - 0.05)
        self.assertEqual(wrist.assignment, (1,))
        self.assertGreater(wrist.scores[0][1], WRIST_VIEW_MIN_SCORE + 0.2)
        self.assertLess(wrist.scores[0][0], WRIST_VIEW_MIN_SCORE)


if __name__ == "__main__":
    unittest.main()
