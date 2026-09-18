"""The desk holds the declared tool frame to the hand's grasp centre (customer chain lane C4f).

A real cell whose ``tool_frame.offset_mm`` does not put the grasp centre where the hand's registry ``grasp_centre_mm``
plus its plates does commands a grasp centre the guard and the planner do not model: a body written from dimensions is
built around that number. Nothing compared the two. The owner decided on 2026-09-17 (decision 5) that the row WARNs
beyond 1 mm for now: the 2F-85's registry says 146.5 mm, an estimate, while every shipped 2F-85 profile runs 132 mm,
and that is measured before the row may BLOCK.
"""

from __future__ import annotations

import unittest

from src.config.schema.robot import RobotConfig

_IDENTITY = (0.0, 0.0, 0.0, 1.0)
_PLUS_Y = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)


def _row(model: str, offset: tuple[float, float, float], rotation=_IDENTITY, plates=None):
    from src.robot.execution.real_cell.preflight import run_config_preflight

    gripper: dict = {"model": model, "tool_frame": {"source": "willy", "offset_mm": offset,
                                                   "rotation_quat_xyzw": rotation}}
    if plates is not None:
        gripper["coupling_plates"] = plates
    cfg = RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": "curobo"}, "gripper": gripper,
        "safety": {"self_collision": {"kinematics_model": "ur5e", "planner_margin_mm": 4.0, "backend": "fcl"}},
    })
    report = run_config_preflight(cfg, curobo_available=True, collision_engine="coal")
    rows = [check for check in report.checks if check.name == "grasp centre"]
    return rows[0] if rows else None


class TheGraspCentreRowTests(unittest.TestCase):
    def test_a_frame_whose_grasp_centre_is_not_the_hands_is_named(self) -> None:
        row = _row("schunk_egu50", (0.0, 0.0, 132.0))
        self.assertIsNotNone(row)
        self.assertEqual(str(row.status), "warn")
        for part in ("132", "159.1", "schunk_egu50", "+Z"):
            self.assertIn(part, row.detail)

    def test_frames_that_put_the_centre_where_the_hand_does_read_ok(self) -> None:
        """⭐ THE CONTROLS: along +Z, along +Y by the declared rotation, and a mounting face hand with its plate."""
        for label, row in (
            ("egu50 +Z", _row("schunk_egu50", (0.0, 0.0, 159.1))),
            ("egu50 +Y", _row("schunk_egu50", (0.0, 159.1, 0.0), rotation=_PLUS_Y)),
            ("hande plate", _row("robotiq_hande", (0.0, 0.0, 156.2),
                                 plates=[{"name": "plate", "thickness_mm": 20.0}])),
        ):
            with self.subTest(frame=label):
                self.assertIsNotNone(row)
                self.assertEqual(str(row.status), "ok", row.detail)

    def test_an_offset_across_the_approach_is_named(self) -> None:
        row = _row("schunk_egu50", (30.0, 0.0, 159.1))
        self.assertEqual(str(row.status), "warn")
        self.assertIn("across", row.detail)

    def test_the_hande_runbook_number_is_inside_the_tolerance(self) -> None:
        """The runbook says plate + 135.75 while the registry says 136.2: 0.45 mm, which the 1 mm tolerance admits."""
        row = _row("robotiq_hande", (0.0, 0.0, 155.75), plates=[{"name": "plate", "thickness_mm": 20.0}])
        self.assertEqual(str(row.status), "ok", row.detail)

    def test_an_undeclared_frame_gets_no_row(self) -> None:
        from src.robot.execution.real_cell.preflight import run_config_preflight

        cfg = RobotConfig.model_validate({
            "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": "curobo"},
            "gripper": {"model": "schunk_egu50", "tool_frame": {"source": "undeclared"}},
            "safety": {"self_collision": {"kinematics_model": "ur5e", "planner_margin_mm": 4.0, "backend": "fcl"}},
        })
        report = run_config_preflight(cfg, curobo_available=True, collision_engine="coal")
        self.assertEqual([check for check in report.checks if check.name == "grasp centre"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
