"""A cell's planner starts from the command line, through the driver's own path, and stops again (lane C4k).

The owner's decision of 2026-09-17: ``real_cell --start-planner`` with a library twin. A customer who built a hand and
its evidence with the scripts had no public way to learn whether their cell's planner starts: the only door was the
first planned move of a connected arm. The start runs the path that move runs, so every refusal it meets (the margin,
the hand, the retract row, the descriptor, the evidence) is met here, and no controller is asked.

No test here starts a sidecar: ``CuroboPlanClient.start`` is patched to report the identity the committed evidence file
measured (``tests/_sidecar_identity.py``), or a changed one.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from typing import Any
from unittest import mock

from src.config import load_config
from src.config.schema.robot import RobotConfig
from src.robot.safety.planning import CuroboPlanClient

_TOOL = {
    "source": "willy", "offset_mm": (0.0, 132.0, 0.0),
    "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
}


def _cell(**over: Any) -> RobotConfig:
    """A real UR cell the matrix measured: ur5e, 2F-85, +Y+X, 4 mm against a 10 mm guard."""
    tree: dict = {
        "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": "curobo"},
        "gripper": {"model": "robotiq_2f85", "tool_frame": _TOOL},
        "safety": {"self_collision": {"kinematics_model": "ur5e", "planner_margin_mm": 4.0, "backend": "fcl"}},
    }
    for section, fields in over.items():
        tree.setdefault(section, {}).update(fields)
    return RobotConfig.model_validate(tree)


class _Sidecar:
    """Patches the client's start and close, and records both."""

    def __init__(self, case: unittest.TestCase, **identity: Any) -> None:
        from tests._sidecar_identity import arm_identity

        self.started: list[object] = []
        self.closed: list[object] = []
        said = arm_identity()
        fields = {**said.__dict__, **identity}

        def start(client: CuroboPlanClient) -> None:
            from src.robot.safety.planning.curobo_client import SidecarIdentity

            self.started.append(client)
            client.identity = SidecarIdentity(**fields)  # type: ignore[attr-defined]

        def close(client: CuroboPlanClient) -> None:
            self.closed.append(client)

        case.enterContext(mock.patch.object(CuroboPlanClient, "start", autospec=True, side_effect=start))
        case.enterContext(mock.patch.object(CuroboPlanClient, "close", autospec=True, side_effect=close))


class TheLibraryTwinTests(unittest.TestCase):
    def test_a_measured_cell_starts_its_planner_and_stops_it(self) -> None:
        from src.robot.execution.planner_start import PlannerStart

        sidecar = _Sidecar(self)
        report = PlannerStart.from_robot_config(_cell()).run()
        self.assertTrue(report.started, report.render())
        self.assertEqual(report.exit_code, 0)
        self.assertEqual((len(sidecar.started), len(sidecar.closed)), (1, 1))
        self.assertIn("ur5e_robotiq_2f85_c0mm_+Y+X_m4mm_a0.json", report.render())

    def test_a_sidecar_that_loaded_another_robot_is_refused_by_the_driver_s_own_check(self) -> None:
        """⭐ THE CONTROL: the start is the driver's path, so the evidence check it runs is the one that refuses."""
        from src.robot.execution.planner_start import PlannerStart

        sidecar = _Sidecar(self, composed_sha256="0" * 64)
        report = PlannerStart.from_robot_config(_cell()).run()
        self.assertFalse(report.started)
        self.assertEqual(report.exit_code, 1)
        self.assertIn("composed_sha256", report.refusal)
        self.assertEqual(len(sidecar.closed), 1)

    def test_a_margin_nobody_declared_is_refused_before_a_sidecar_starts(self) -> None:
        from src.robot.execution.planner_start import PlannerStart

        sidecar = _Sidecar(self)
        cell = _cell(safety={"self_collision": {"kinematics_model": "ur5e", "backend": "fcl"}})
        report = PlannerStart.from_robot_config(cell).run()
        self.assertFalse(report.started)
        self.assertIn("planner_margin_mm", report.refusal)
        self.assertEqual(sidecar.started, [])

    def test_an_ik_cell_and_a_dummy_arm_start_no_planner_and_say_why(self) -> None:
        from src.robot.execution.planner_start import PlannerStart

        _Sidecar(self)
        ik = PlannerStart.from_robot_config(_cell(ur={"model": "ur5e", "motion_planner": "ik"})).run()
        self.assertIn("motion_planner", ik.refusal)
        dummy = PlannerStart.from_robot_config(_cell().model_copy(update={"vendor": "dummy"})).run()
        self.assertIn("robot.vendor", dummy.refusal)
        self.assertEqual((ik.exit_code, dummy.exit_code), (1, 1))

    def test_the_cell_step_is_the_twin(self) -> None:
        from src.robot.execution.cell import Cell

        _Sidecar(self)
        self.assertTrue(Cell.from_robot_config(_cell()).start_planner().started)

    def test_the_report_is_a_view_on_the_wire(self) -> None:
        from src.robot.execution.planner_start import PlannerStart

        _Sidecar(self)
        report = PlannerStart.from_robot_config(_cell()).run()
        wire = report.to_dict()
        self.assertEqual((wire["started"], wire["arm"], wire["hand"]), (True, "ur5e", "robotiq_2f85"))
        self.assertFalse(report.render().endswith("\n"))


class TheCommandLineTests(unittest.TestCase):
    def test_the_flag_starts_the_planner_of_the_loaded_cell_and_exits_on_the_report(self) -> None:
        from src.robot.execution.real_cell import __main__ as runner

        _Sidecar(self)
        with mock.patch.object(runner, "_load_cell_config", return_value=(load_config(), _cell())), \
                redirect_stdout(io.StringIO()) as out:
            code = runner.main(["--start-planner"])
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("PLANNER", out.getvalue())

    def test_a_refused_start_exits_as_a_config_refusal(self) -> None:
        from src.robot.execution.real_cell import __main__ as runner

        _Sidecar(self, composed_sha256="0" * 64)
        with mock.patch.object(runner, "_load_cell_config", return_value=(load_config(), _cell())), \
                redirect_stdout(io.StringIO()) as out:
            code = runner.main(["--start-planner"])
        self.assertEqual(code, 1, out.getvalue())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
