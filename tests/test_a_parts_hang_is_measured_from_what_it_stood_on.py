"""A part's hang below the tool is measured from what it stood on, not from the support the cell declares (2026-10-08).

On the owner's cell every drop stood 85 to 92 mm over the rim: the hang was the grasp's height less the declared
support, the bench at 0, while the part stood on a 55 mm mat. With ``robot.place.part_bottom: measured`` the hang is the
grasp's height less the lowest reading of a surface the pick's own looks found under the part's cloud
(``SupportModel.local_plane_under``), or the cloud's lowest point where that is lower: a lower bound on where the part's
bottom stood either way, so every error still goes toward more air. A part on another part hangs from the surface under
both. The declared support stands where the pick kept no cloud, its looks found no surface under the part, or what they
read does not stand below the grasp. Measured, a part set down at a taught pose keeps the set-down air (5 mm) over it.
As shipped, the declared support, as before.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.core import JointPositions
from tests._place_fakes import FlatSupport, cube_cloud, hande_tree, run_placing, what_the_looks_saw
from tests._task_fakes import (
    BIN_CENTRE,
    BIN_RIM_MM,
    GRASP_Z_MM,
    PLACE_TCP,
    Pick,
    ScriptedLocator,
    TaskArm,
    bin_object,
    motions,
)

L1 = JointPositions.deg(-20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
L2 = JointPositions.deg(10.0, -90.0, -110.0, -60.0, 90.0, 0.0)
AT_L1 = Pose.tool_down(400.0, -300.0, 450.0, label="L1")
AT_L2 = Pose.tool_down(150.0, -650.0, 450.0, label="L2")


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


class WhereThePartStoodTests(unittest.TestCase):
    def test_a_part_on_the_mat_stood_on_the_mat(self) -> None:
        from src.robot.execution.place_target import part_bottom

        bottom = part_bottom(cube_cloud(bottom_mm=55.0), FlatSupport(55.0), declared_mm=0.0)

        self.assertTrue(bottom.measured)
        self.assertAlmostEqual(55.0, bottom.mm, delta=1e-6)
        self.assertIn("55.0", bottom.said)

    def test_a_cloud_that_reaches_lower_than_the_surface_under_it_stood_on_its_lowest_point(self) -> None:
        from src.robot.execution.place_target import part_bottom

        bottom = part_bottom(cube_cloud(bottom_mm=52.0), FlatSupport(55.0), declared_mm=0.0)

        self.assertAlmostEqual(52.0 + 2.0, bottom.mm, delta=1e-6, msg="the cloud's lowest point is its sides' first row")

    def test_a_part_on_another_part_hangs_from_the_surface_under_both(self) -> None:
        from src.robot.execution.place_target import part_bottom

        bottom = part_bottom(cube_cloud(bottom_mm=95.0), FlatSupport(55.0), declared_mm=0.0)

        self.assertAlmostEqual(55.0, bottom.mm, delta=1e-6)

    def test_with_no_surface_under_the_part_or_no_cloud_the_declared_support_stands(self) -> None:
        from src.robot.execution.place_target import part_bottom

        for name, cloud, support in (("no surface model", cube_cloud(), None),
                                     ("no surface under it", cube_cloud(), FlatSupport(55.0, x_mm=(500.0, 900.0))),
                                     ("no cloud", None, FlatSupport(55.0)),
                                     ("too little cloud", np.zeros((1, 3)), FlatSupport(55.0))):
            with self.subTest(name):
                bottom = part_bottom(cloud, support, declared_mm=-5.0)
                self.assertFalse(bottom.measured)
                self.assertEqual(-5.0, bottom.mm)


def _measured(level_mm: float = 10.0, **place: Any) -> Any:
    """An arm whose tree (the Hand-E profile's) measures the hang, its picks' looks reading a mat at ``level_mm``."""
    log: list[Any] = []
    arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2})
    arm.config = hande_tree(**place)  # type: ignore[attr-defined]
    return arm, what_the_looks_saw(cloud=cube_cloud(bottom_mm=level_mm, side_mm=30.0), support=FlatSupport(level_mm))


def _line_in(ran: Any) -> Any:
    return [m for m in motions(ran.after("task.place_started")) if m[0] == "move"][1]


class ATaskHangsItsPartFromWhatItStoodOnTests(unittest.TestCase):
    def test_at_a_taught_pose_the_part_hangs_from_the_mat_and_keeps_5_mm_of_air(self) -> None:
        """Gripped 20 mm up, standing on a 10 mm mat: it hangs 10 mm, not 20, and is let go 5 mm over the taught point."""
        arm, looked = _measured(part_bottom="measured")
        ran = run_placing(["part"], arm=arm, looked_around=looked)

        np.testing.assert_allclose(PLACE_TCP.position_mm + (0.0, 0.0, (GRASP_Z_MM - 10.0) + 5.0), _line_in(ran)[1])
        drop = ran.hooks.of("task.drop_planned")[0]
        self.assertAlmostEqual(GRASP_Z_MM - 10.0, drop["hang_mm"])
        self.assertAlmostEqual(5.0, drop["air_mm"])

    def test_into_a_bin_the_part_hangs_from_the_mat(self) -> None:
        from src.robot.execution.task import PlaceAt

        arm, looked = _measured(part_bottom="measured")
        locator = ScriptedLocator({"blue bin": lambda tcp: [bin_object()] if tcp is AT_L1 else []}, arm=arm,
                                  log=arm.log)
        ran = run_placing(["part"], arm=arm, looked_around=looked, place=PlaceAt(camera="blue bin", air_mm=20.0),
                          wrist=True, looks=(L1, L2), locators=[locator])

        line_in = _line_in(ran)
        np.testing.assert_allclose(BIN_CENTRE, line_in[1][:2], atol=10.0)
        self.assertAlmostEqual(BIN_RIM_MM + (GRASP_Z_MM - 10.0) + 20.0, line_in[1][2], delta=0.5)

    def test_as_shipped_the_part_hangs_from_the_declared_support(self) -> None:
        arm, looked = _measured()
        ran = run_placing(["part"], arm=arm, looked_around=looked)

        np.testing.assert_allclose(PLACE_TCP.position_mm + (0.0, 0.0, GRASP_Z_MM), _line_in(ran)[1])
        self.assertIsNone(ran.hooks.of("task.drop_planned")[0]["air_mm"])

    def test_what_the_looks_read_over_the_grasp_leaves_the_declared_support(self) -> None:
        """A surface read at 25 mm under a grasp at 20 mm says nothing about the part: the declared bench stands."""
        arm, looked = _measured(25.0, part_bottom="measured")
        ran = run_placing(["part"], arm=arm, looked_around=looked)

        np.testing.assert_allclose(PLACE_TCP.position_mm + (0.0, 0.0, GRASP_Z_MM), _line_in(ran)[1])

    def test_a_pick_whose_looks_kept_nothing_leaves_the_declared_support(self) -> None:
        arm, _looked = _measured(part_bottom="measured")
        ran = run_placing([Pick("part")], arm=arm)

        np.testing.assert_allclose(PLACE_TCP.position_mm + (0.0, 0.0, GRASP_Z_MM), _line_in(ran)[1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
