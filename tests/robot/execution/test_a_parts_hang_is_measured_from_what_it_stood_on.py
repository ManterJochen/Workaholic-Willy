"""A part's hang below the tool is measured from what it stood on, not from the support the cell declares (2026-10-08).

On the owner's cell every drop stood 85 to 92 mm over the rim: the hang was the grasp's height less the declared
support, the bench at 0, while the part stood on a 55 mm mat. Wherever the cell declares how far its parts hang
(``safety.planning_world.payload.length_mm``), the hang is measured part by part (the owner, 2026-10-14; the switch
``robot.motion.place.part_bottom`` is gone): the grasp's height less the lowest reading of a surface the pick's own looks found
under the part's cloud (``SupportModel.local_plane_under``), or the cloud's lowest point where that is lower: a lower
bound on where the part's bottom stood either way, so every error still goes toward more air. A part on another part
hangs from the surface under both. The declared support stands where the pick kept no cloud, its looks found no surface
under the part, or what they read does not stand below the grasp. At a taught pose the part's bottom goes where the
fingertips were taught, with no air over it; over a bin it keeps the rim air. A cell that declares no length hangs
nothing below the tool, whatever the looks read: the tool stands where the place says.
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
#: How far the Hand-E profile's fingertips reach past its TCP, where the person taught the part's bottom to go (set in
#: setUpModule, from ``task._fingertips_past_the_tcp_mm``: 10.45 mm).
FINGERTIPS_MM = 0.0


def setUpModule() -> None:
    global FINGERTIPS_MM
    from src.robot.execution.task import _fingertips_past_the_tcp_mm

    FINGERTIPS_MM = _fingertips_past_the_tcp_mm(hande_tree())
    assert FINGERTIPS_MM > 0.0, "the Hand-E profile's fingertips reach nothing past its TCP"


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


def _measured(level_mm: float = 10.0, *, length_mm: "float | None" = 60.0, **place: Any) -> Any:
    """An arm whose tree (the Hand-E profile's) declares how far its parts hang past the fingertips (``length_mm``;
    ``None``: not at all), its picks' looks reading a mat at ``level_mm``."""
    log: list[Any] = []
    arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2})
    arm.config = hande_tree(length_mm=length_mm, **place)  # type: ignore[attr-defined]
    return arm, what_the_looks_saw(cloud=cube_cloud(bottom_mm=level_mm, side_mm=30.0), support=FlatSupport(level_mm))


def _line_in(ran: Any) -> Any:
    return [m for m in motions(ran.after("task.place_started")) if m[0] == "move"][1]


def _into_the_blue_bin(arm: Any, looked: Any) -> Any:
    """One part into the blue bin the wrist camera finds from L1."""
    from src.robot.execution.task import PlaceAt

    locator = ScriptedLocator({"blue bin": lambda tcp: [bin_object()] if tcp is AT_L1 else []}, arm=arm, log=arm.log)
    return run_placing(["part"], arm=arm, looked_around=looked, place=PlaceAt(camera="blue bin", air_mm=20.0),
                       wrist=True, looks=(L1, L2), locators=[locator])


def _over_the_taught_pose(test: unittest.TestCase, ran: Any, rise_mm: float) -> None:
    """The drop planned ``rise_mm`` over the taught pose, and the line in ended there (the log keeps a tenth of a
    millimetre)."""
    drop = ran.hooks.of("task.drop_planned")[0]
    np.testing.assert_allclose(PLACE_TCP.position_mm + (0.0, 0.0, rise_mm), drop["pose_mm"], atol=1e-9)
    np.testing.assert_allclose(drop["pose_mm"], _line_in(ran)[1], atol=0.05 + 1e-6,
                               err_msg="the line in did not end at the drop")


class ATaskHangsItsPartFromWhatItStoodOnTests(unittest.TestCase):
    def test_at_a_taught_pose_the_part_hangs_from_the_mat_its_bottom_where_the_fingertips_were_taught(self) -> None:
        """Gripped 20 mm up, standing on a 4 mm mat: it hangs 16 mm below the tool, not 20, and reaches 5.55 mm past
        the Hand-E's fingertips, so the tool stands that far over the taught pose and the part's bottom where the
        fingertips were taught. No air: before the owner's word of 2026-10-14 a measured hang kept 5 mm over the
        taught point."""
        arm, looked = _measured(4.0)
        ran = run_placing(["part"], arm=arm, looked_around=looked)

        _over_the_taught_pose(self, ran, GRASP_Z_MM - 4.0 - FINGERTIPS_MM)
        drop = ran.hooks.of("task.drop_planned")[0]
        self.assertAlmostEqual(GRASP_Z_MM - 4.0, drop["hang_mm"])
        self.assertIsNone(drop["air_mm"])

    def test_into_a_bin_the_part_hangs_from_the_mat(self) -> None:
        """Gripped 20 mm up, standing on a 10 mm mat: it hangs 10 mm below the tool, which stands that and the rim air
        over the rim, the part's bottom the air over it."""
        arm, looked = _measured()
        ran = _into_the_blue_bin(arm, looked)

        line_in = _line_in(ran)
        np.testing.assert_allclose(BIN_CENTRE, line_in[1][:2], atol=10.0)
        self.assertAlmostEqual(BIN_RIM_MM + (GRASP_Z_MM - 10.0) + 20.0, line_in[1][2], delta=0.5)
        self.assertAlmostEqual(GRASP_Z_MM - 10.0, ran.hooks.of("task.drop_planned")[0]["hang_mm"])

    def test_a_cell_that_declares_no_length_hangs_nothing_below_the_tool_whatever_the_looks_read(self) -> None:
        """The owner, 2026-10-14: "Wenn kein Payloadlength angegeben ist, dann soll es am TCP hängen". The looks read
        the 10 mm mat under the part; the tool goes to the taught pose itself, and stands the rim air over a bin's
        rim."""
        arm, looked = _measured(length_mm=None)
        at_the_pose = run_placing(["part"], arm=arm, looked_around=looked)

        np.testing.assert_allclose(PLACE_TCP.position_mm, _line_in(at_the_pose)[1])
        self.assertEqual(0.0, at_the_pose.hooks.of("task.drop_planned")[0]["hang_mm"])

        arm, looked = _measured(length_mm=None)
        into_the_bin = _into_the_blue_bin(arm, looked)

        self.assertAlmostEqual(BIN_RIM_MM + 20.0, _line_in(into_the_bin)[1][2], delta=0.5)
        self.assertEqual(0.0, into_the_bin.hooks.of("task.drop_planned")[0]["hang_mm"])

    def test_what_the_looks_read_over_the_grasp_leaves_the_declared_support(self) -> None:
        """A surface read at 25 mm under a grasp at 20 mm says nothing about the part: the declared bench stands, the
        part hanging 20 mm below the tool, the tool that less the fingertips' reach over the taught pose."""
        arm, looked = _measured(25.0)
        ran = run_placing(["part"], arm=arm, looked_around=looked)

        _over_the_taught_pose(self, ran, GRASP_Z_MM - FINGERTIPS_MM)
        self.assertAlmostEqual(GRASP_Z_MM, ran.hooks.of("task.drop_planned")[0]["hang_mm"])

    def test_a_pick_whose_looks_kept_nothing_leaves_the_declared_support(self) -> None:
        arm, _looked = _measured()
        ran = run_placing([Pick("part")], arm=arm)

        _over_the_taught_pose(self, ran, GRASP_Z_MM - FINGERTIPS_MM)
        self.assertAlmostEqual(GRASP_Z_MM, ran.hooks.of("task.drop_planned")[0]["hang_mm"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
