"""A closing line wider than the hand closes on is refused whole, each of its builds counted as before.

SFE builds a grasp at every height, tilt and side along each closing line through the part, and refuses one whose span
is wider than the hand closes on (``aperture``). A grasp never spans less than the line through its own anchor, so where
that line is already too wide, every build on it is an aperture refusal: on the owner's cell (2026-10-08) 1,404 of the
1,872 builds of one tilted cube were. Such a line is now refused at once (``_wider_than_the_hand_closes``), only where
nothing rolls, and its builds are counted under ``aperture`` as they were counted one by one.

Pinned here against the builds made one by one: no candidate and no count changes, on a box, cylinders, tilted cubes
and a cylinder beside a wall, with the Hand-E and the library's 2F-85; and the shortcut does skip lines on each.
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest import mock

import numpy as np

from src.robot.grasping.generation import support_footprint as sf
from tests.test_sfe_says_why_it_refused import (
    box_cloud,
    candidate_digest,
    cylinder_cloud,
    default_jaw,
    hande_jaw,
    wall_cloud,
)


def tilted_cube(side_mm: float, *, roll_deg: float, pitch_deg: float, yaw_deg: float,
                step_mm: float = 1.5) -> np.ndarray:
    """A cube's six faces turned by ``roll``, ``pitch`` and ``yaw`` and set down on the support at z 0 on its lowest
    corner, what stands over the support: the owner's cube lying tilted at the mat's edge, in miniature."""
    half = side_mm / 2.0
    grid = np.arange(-half, half + 1e-9, step_mm)
    a, b = np.meshgrid(grid, grid)
    faces = []
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for side in (-half, half):
            face = np.zeros((a.size, 3))
            face[:, axis] = side
            face[:, others[0]] = a.ravel()
            face[:, others[1]] = b.ravel()
            faces.append(face)
    r, p, y = (math.radians(v) for v in (roll_deg, pitch_deg, yaw_deg))
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(r), -math.sin(r)], [0.0, math.sin(r), math.cos(r)]])
    about_y = np.array([[math.cos(p), 0.0, math.sin(p)], [0.0, 1.0, 0.0], [-math.sin(p), 0.0, math.cos(p)]])
    about_z = np.array([[math.cos(y), -math.sin(y), 0.0], [math.sin(y), math.cos(y), 0.0], [0.0, 0.0, 1.0]])
    points = np.vstack(faces) @ (about_z @ about_y @ about_x).T
    points[:, 2] -= points[:, 2].min()
    return points[points[:, 2] > 0.5]


def _scenes() -> "dict[str, tuple[np.ndarray, dict[str, Any]]]":
    """Each scene's cloud and what SFE is asked with besides it."""
    hande = {"jaw": hande_jaw()}
    return {
        "a 30 x 70 box, too long one way": (box_cloud((-15.0, -35.0, 0.0), (15.0, 35.0, 40.0)), hande),
        "a 60 mm cube, too wide both ways": (box_cloud((-30.0, -30.0, 0.0), (30.0, 30.0, 60.0)), hande),
        "a 49 mm cylinder, too wide through its middle": (cylinder_cloud(49.0, 60.0), hande),
        "a 60 mm cylinder, too wide every way": (cylinder_cloud(60.0, 60.0), hande),
        "a 45 mm cube lying tilted": (tilted_cube(45.0, roll_deg=25.0, pitch_deg=15.0, yaw_deg=30.0), hande),
        "a 40 mm cube lying tilted": (tilted_cube(40.0, roll_deg=20.0, pitch_deg=0.0, yaw_deg=10.0), hande),
        "a 49 mm cylinder beside a wall": (cylinder_cloud(49.0, 60.0), {**hande, "obstacle_points_base_mm":
                                                                        wall_cloud(36.0)}),
        "a 90 mm cylinder and the 2F-85": (cylinder_cloud(90.0, 60.0), {"jaw": default_jaw(), "side_approaches": True}),
    }


def _run(cloud: np.ndarray, kwargs: "dict[str, Any]", *, whole: bool) -> "tuple[list[Any], dict[str, int], int]":
    """SFE's grasps and counts, the lines refused whole where ``whole``, else every build made one by one; and how many
    lines were refused whole."""
    counts: dict[str, int] = {}
    real = sf._wider_than_the_hand_closes
    said: list[bool] = []

    def watched(*args: Any) -> bool:
        said.append(bool(real(*args)) if whole else False)
        return said[-1]

    with mock.patch.object(sf, "_wider_than_the_hand_closes", side_effect=watched):
        found = sf.generate_support_footprint_grasps(cloud, support_height_mm=0.0, max_candidates=12,
                                                     refusals=counts, **kwargs)
    return found, counts, sum(said)


class NoCandidateAndNoCountChangesTests(unittest.TestCase):
    def test_every_scene_gets_the_grasps_and_the_counts_of_the_builds_made_one_by_one(self) -> None:
        for name, (cloud, kwargs) in _scenes().items():
            with self.subTest(name):
                whole, whole_counts, skipped = _run(cloud, kwargs, whole=True)
                one_by_one, counts, _ = _run(cloud, kwargs, whole=False)
                self.assertEqual(candidate_digest(one_by_one), candidate_digest(whole))
                self.assertEqual(counts, whole_counts)
                self.assertGreater(skipped, 0, "no line was refused whole: the scene does not ask the shortcut")
                self.assertGreater(whole_counts["aperture"], 0)

    def test_a_part_the_hand_closes_across_everywhere_keeps_every_line(self) -> None:
        """A 30 mm cube: no line through it is wider than the Hand-E closes on, so none is refused whole."""
        found, counts, skipped = _run(box_cloud((-15.0, -15.0, 0.0), (15.0, 15.0, 40.0)), {"jaw": hande_jaw()},
                                      whole=True)
        self.assertEqual(0, skipped)
        self.assertTrue(found)
        self.assertEqual(0, counts["aperture"])


class TheLineIsReadInsideThePartsFacesTests(unittest.TestCase):
    def test_a_box_as_wide_as_the_hand_closes_on_is_left_to_its_builds_and_one_a_hundredth_wider_is_not(self) -> None:
        """A line exactly the stroke less the safety across is left to the builds, which decide it as they always did;
        a hundredth of a millimetre wider, it is refused whole. Either way the grasps and the counts are the builds'."""
        jaw = hande_jaw()
        width = jaw.aperture_mm - jaw.width_safety_mm
        for extra, refused_whole in ((0.0, False), (0.01, True)):
            with self.subTest(wider_by_mm=extra):
                half = (width + extra) / 2.0
                cloud = box_cloud((-half, -10.0, 0.0), (half, 10.0, 40.0), step_mm=0.5)
                prism = sf.reconstruct_support_prism(cloud, 0.0)
                assert prism is not None
                across = np.array([1.0, 0.0, 0.0])
                mid_z = (prism.z0 + prism.z1) / 2.0
                self.assertEqual(refused_whole, sf._wider_than_the_hand_closes(prism, prism.centre, mid_z, across, jaw))
                whole, whole_counts, _ = _run(cloud, {"jaw": jaw}, whole=True)
                one_by_one, counts, _ = _run(cloud, {"jaw": jaw}, whole=False)
                self.assertEqual(candidate_digest(one_by_one), candidate_digest(whole))
                self.assertEqual(counts, whole_counts)

    def test_the_long_way_of_a_box_is_refused_whole_and_its_short_way_never(self) -> None:
        jaw = hande_jaw()
        prism = sf.reconstruct_support_prism(box_cloud((-15.0, -35.0, 0.0), (15.0, 35.0, 40.0)), 0.0)
        assert prism is not None
        mid_z = (prism.z0 + prism.z1) / 2.0
        self.assertTrue(sf._wider_than_the_hand_closes(prism, prism.centre, mid_z, np.array([0.0, 1.0, 0.0]), jaw))
        self.assertFalse(sf._wider_than_the_hand_closes(prism, prism.centre, mid_z, np.array([1.0, 0.0, 0.0]), jaw))


if __name__ == "__main__":
    unittest.main()
