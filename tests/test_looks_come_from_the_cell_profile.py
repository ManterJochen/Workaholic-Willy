"""A cell profile declares where its wrist camera looks from, in degrees; a program's looks override them (plan step 6).

``robot.look_joint_positions_deg``: one list per look, one value per joint, in degrees as the pendant and ``--where``
show them. It stays in degrees in the loaded config, and the service reads it into ``JointPositions.deg`` once, as
``configured_looks``. There is deliberately no radians twin: a look is where the arm goes next, and a unit that cannot
be told is refused as a look everywhere else. What the loader cannot refuse, radians written into the degrees key, the
desk check says.

Who hands a pick its looks: a program's own (``PickRun(look=...)``), else the configured ones, else home on a wrist
camera, else none; the console the configured ones, else home on a wrist camera, else none. A double that answers every
attribute is read as neither a wrist camera nor a list of looks.
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest.mock import MagicMock

from pydantic import ValidationError

from src.config.schema.robot import RobotConfig
from src.robot.core import JointPositions
from src.robot.execution.autonomous_grasp import AutonomousGraspService
from src.robot.execution.looks import HOME
from src.robot.execution.pick_run import PickRun, Recording
from tests.test_a_console_run_stops_and_looks_as_a_campaign_does import _drive, _Report, _Service
from tests.test_grasping_config_wiring import _calc_and_perception

LOOKS_DEG = [[-90.0, -100.0, -110.0, -60.0, 90.0, 0.0], [-70.0, -100.0, -110.0, -60.0, 90.0, 0.0]]
CONFIGURED = tuple(JointPositions.deg(*row) for row in LOOKS_DEG)
PROGRAM = (JointPositions.deg(0.0, -90.0, -110.0, -60.0, 90.0, 0.0),)


def _config(**keys: Any) -> RobotConfig:
    return RobotConfig(vendor="dummy", gripper={"vendor": "none"}, grasping={"default_mode": "easy"}, **keys)


class TheLooksLoadInDegreesTests(unittest.TestCase):
    def test_degrees_load_and_become_joint_positions(self) -> None:
        config = _config(look_joint_positions_deg=LOOKS_DEG)
        calculator, perception = _calc_and_perception()

        service = AutonomousGraspService.from_robot_config(config, calculator=calculator, perception=perception)

        self.assertEqual(tuple(tuple(row) for row in LOOKS_DEG), config.look_joint_positions_deg,
                         "the loaded config keeps the looks in degrees")
        self.assertEqual(CONFIGURED, service.configured_looks)
        self.assertAlmostEqual(-math.pi / 2, service.configured_looks[0].values[0])

    def test_a_cell_that_declares_none_has_none(self) -> None:
        calculator, perception = _calc_and_perception()

        service = AutonomousGraspService.from_robot_config(_config(), calculator=calculator, perception=perception)

        self.assertIsNone(_config().look_joint_positions_deg)
        self.assertEqual((), service.configured_looks)

    def test_mixed_lengths_non_finite_or_past_a_turn_are_refused_at_load(self) -> None:
        refused = {
            "no look at all": [],
            "an empty look": [[]],
            "mixed lengths": [LOOKS_DEG[0], LOOKS_DEG[1][:5]],
            "not a number": [[0.0, float("nan"), 0.0, 0.0, 0.0, 0.0]],
            "infinite": [[0.0, float("inf"), 0.0, 0.0, 0.0, 0.0]],
            "past a turn": [[0.0, -90.0, 361.0, 0.0, 0.0, 0.0]],
        }
        for why, looks in refused.items():
            with self.subTest(why), self.assertRaises(ValidationError) as caught:
                _config(look_joint_positions_deg=looks)
            self.assertIn("look_joint_positions_deg", str(caught.exception))

    def test_a_turn_either_way_is_a_look(self) -> None:
        self.assertIsNotNone(_config(look_joint_positions_deg=[[360.0, -360.0, 0.0, 0.0, 0.0, 0.0]]))

    def test_a_length_unlike_home_is_refused(self) -> None:
        home = [-90.0, -100.0, -110.0, -60.0, 90.0, 0.0]
        with self.assertRaises(ValidationError) as caught:
            _config(home_joint_positions_deg=home, look_joint_positions_deg=[LOOKS_DEG[0][:5]])
        self.assertIn("home", str(caught.exception))
        self.assertIsNotNone(_config(home_joint_positions_deg=home, look_joint_positions_deg=LOOKS_DEG))


class TheDeskSaysALookThatReadsAsRadiansTests(unittest.TestCase):
    @staticmethod
    def _row(config: RobotConfig) -> Any:
        from src.robot.execution.real_cell.preflight import run_config_preflight

        rows = [check for check in run_config_preflight(config, curobo_available=True, collision_engine="coal").checks
                if check.name == "looks"]
        return rows[0] if rows else None

    def test_the_desk_warns_a_look_that_reads_as_radians(self) -> None:
        from src.robot.execution.real_cell.preflight import CheckStatus

        row = self._row(_config(look_joint_positions_deg=[LOOKS_DEG[0], [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]]))

        self.assertIs(CheckStatus.WARN, row.status)
        self.assertIn("look 2", row.detail)
        self.assertIn("radians", row.detail)
        self.assertIn("degrees", row.fix)

    def test_looks_in_degrees_pass_the_desk(self) -> None:
        from src.robot.execution.real_cell.preflight import CheckStatus

        row = self._row(_config(look_joint_positions_deg=LOOKS_DEG))

        self.assertIs(CheckStatus.OK, row.status)
        self.assertIn("2 look(s)", row.detail)

    def test_a_cell_that_declares_no_look_has_no_row(self) -> None:
        self.assertIsNone(self._row(_config()))


class _Configured(_Service):
    """The console's recording service, with looks from its cell profile."""

    def __init__(self, *, wrist: Any, looks: Any) -> None:
        super().__init__(wrist=wrist)
        self.configured_looks = looks


class TheConsoleHandsEveryPickTheConfiguredLooksTests(unittest.TestCase):
    def test_the_console_hands_every_pick_the_configured_looks(self) -> None:
        for wrist in (True, False):
            with self.subTest(wrist=wrist):
                service = _Configured(wrist=wrist, looks=CONFIGURED)
                _run, _hub = _drive(service, picks=2)
                self.assertEqual([{"look": CONFIGURED}] * 2, service.calls)

    def test_a_wrist_camera_with_none_configured_looks_from_home(self) -> None:
        service = _Configured(wrist=True, looks=())
        _run, _hub = _drive(service, picks=1)
        self.assertEqual([{"look": HOME}], service.calls)

    def test_a_double_that_answers_every_attribute_is_not_read_as_looks(self) -> None:
        from api.runs import _look_for

        self.assertIsNone(_look_for(MagicMock()))
        for looks in (MagicMock(), [CONFIGURED[0]], ("left",), (CONFIGURED[0], [0.0] * 6), "home"):
            with self.subTest(looks=looks):
                self.assertIsNone(_look_for(_Configured(wrist=False, looks=looks)))
                self.assertEqual(HOME, _look_for(_Configured(wrist=True, looks=looks)))
        self.assertEqual((HOME, CONFIGURED[1]), _look_for(_Configured(wrist=False, looks=(HOME, CONFIGURED[1]))))

    def test_each_pick_result_says_where_it_looked_and_what_it_fused(self) -> None:
        class _Looked(_Report):
            looks = ("home", "(0.0) deg")
            looks_fused = ("(0.0) deg", "home")

        class _Plain(_Service):
            def pick(self, **kwargs: Any) -> _Report:
                self.calls.append(dict(kwargs))
                return _Looked() if len(self.calls) == 1 else _Report()

        service = _Plain(wrist=True)
        run, hub = _drive(service, picks=2)

        results = [event for event in hub.since(run.id, 0)[0] if event.type == "pick_result"]
        self.assertEqual(2, len(results))
        first, second = (event.data for event in results)
        self.assertEqual(["home", "(0.0) deg"], first["looks"])
        self.assertEqual(["(0.0) deg", "home"], first["looks_fused"])
        self.assertEqual([], second["looks"], "a report that names no look names none")
        self.assertEqual([], second["looks_fused"])


class ACampaignTakesTheProgramsLooksThenTheConfiguredOnesTests(unittest.TestCase):
    def test_a_programs_looks_override_the_configured_ones(self) -> None:
        service = _Configured(wrist=True, looks=CONFIGURED)

        PickRun.from_service(service, runs=1, recording=Recording.off(), look=PROGRAM).execute()

        self.assertEqual([{"look": PROGRAM}], service.calls)

    def test_a_campaign_with_none_takes_config_then_home(self) -> None:
        cases = (
            (_Configured(wrist=True, looks=CONFIGURED), {"look": CONFIGURED}),
            (_Configured(wrist=False, looks=CONFIGURED), {"look": CONFIGURED}),
            (_Configured(wrist=True, looks=()), {"look": HOME}),
            (_Configured(wrist=False, looks=()), {}),
            (_Configured(wrist=True, looks=MagicMock()), {"look": HOME}),
        )
        for service, handed in cases:
            with self.subTest(wrist=service.perceives_from_the_wrist, looks=service.configured_looks):
                PickRun.from_service(service, runs=1, recording=Recording.off()).execute()
                self.assertEqual([handed], service.calls)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
