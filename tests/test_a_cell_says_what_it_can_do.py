"""A cell says, before anything moves, which of its operator functions it can perform, and why not (P0i i2).

``Cell.abilities()`` asks the arm and the hand the tree builds, never connected, through their capability Protocols,
one row per function the owner named (2026-09-26, decisions 1 and 2), and ``Robot.abilities()`` asks the handles a
robot holds. The property everything after i2 rests on: a REFUSED row's reason is, word for word, the sentence the
function itself refuses with today. P0i i3 makes each function ask its row first; if a row said something else, the
refusal an operator reads would move. So each refusal is taken here from the function itself and compared.
"""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from typing import Any
from unittest import mock

import numpy as np

from src.config import ConfigTree
from src.geometry import Frame, Pose
from src.robot.execution.abilities import DECLARATIONS, Ability, AbilityStatus

_POSE = Pose(position_mm=np.array([400.0, -100.0, 250.0]), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
             frame=Frame.BASE)


def _tree(profile: str | None) -> Any:
    loaded = ConfigTree.from_directory(profile=profile).load()
    assert loaded.ok, loaded.error
    return loaded


def _cell_abilities(profile: str | None) -> Any:
    from src.robot.execution.cell import Cell

    return Cell.from_tree(_tree(profile)).abilities()


class TheDeclarationsTests(unittest.TestCase):
    def test_every_function_is_declared_once_with_what_it_needs(self) -> None:
        self.assertEqual([d.ability for d in DECLARATIONS], list(Ability))
        for declaration in DECLARATIONS:
            with self.subTest(ability=declaration.ability.value):
                self.assertTrue(declaration.functions)
                self.assertTrue(declaration.needs)

    def test_a_tree_whose_arm_does_not_build_refuses_every_row_with_that_refusal(self) -> None:
        abilities = _cell_abilities(None)
        self.assertEqual({row.status for row in abilities.rows}, {AbilityStatus.REFUSED})
        for row in abilities.rows:
            self.assertTrue(row.reason.startswith(
                "the arm this tree selects does not build: ConfigError: robot.gripper.model is unset"), row.reason)
        rendered = abilities.render()
        self.assertEqual(rendered.count("robot.gripper.model is unset"), 1, "one refusal, said once")
        self.assertIn("as move", rendered)

    def test_the_report_is_ascii_json_safe_and_counts_its_rows(self) -> None:
        abilities = _cell_abilities("console_dummy")
        rendered = abilities.render()
        self.assertEqual(rendered, rendered.encode("ascii").decode("ascii"))
        self.assertFalse(rendered.endswith("\n"))
        data = json.loads(json.dumps(abilities.to_dict()))
        self.assertEqual(len(data["abilities"]), len(DECLARATIONS))
        self.assertEqual(data["n_available"] + data["n_refused"] + data["n_bench"], len(DECLARATIONS))
        self.assertIs(abilities.row("teach").ability, Ability.TEACH)
        with self.assertRaises(ValueError):
            abilities.row("fly")


class EachRowIsTheFunctionsOwnRefusalTests(unittest.TestCase):
    """Each REFUSED row compared with what its function answers on the same tree."""

    def test_teach(self) -> None:
        from src.robot.execution.hand_guiding import HandGuidingRefused
        from src.robot.execution.robot import Robot
        from src.robot.execution.teach import teach_poses

        row = _cell_abilities("console_dummy").row(Ability.TEACH)
        self.assertIs(row.status, AbilityStatus.REFUSED)
        with self.assertRaises(HandGuidingRefused) as refused:
            teach_poses(Robot.from_tree(_tree("console_dummy")), out=lambda _line: None)
        self.assertEqual(str(refused.exception), row.reason)

    def test_the_io_bench(self) -> None:
        from src.robot.execution.io_bench import __main__ as bench

        loaded = _tree("console_dummy")
        row = _cell_abilities("console_dummy").row(Ability.IO_BENCH)
        said = io.StringIO()
        with mock.patch.object(bench, "_load_robot_config", lambda *_a: loaded.robot), contextlib.redirect_stdout(said):
            code = bench.main(["--read"])
        self.assertEqual(code, 1)
        self.assertIn(f"REFUSED: {row.reason}\n", said.getvalue())

    def test_the_planner_start(self) -> None:
        from src.robot.execution.cell import Cell

        loaded = _tree("console_dummy")
        row = _cell_abilities("console_dummy").row(Ability.PLANNER_START)
        self.assertEqual(Cell.from_tree(loaded).start_planner().refusal, row.reason)

    def test_calibration_by_hand(self) -> None:
        from src.robot.execution.hand_eye import HandEyeCalibration, SweepOptions

        loaded = _tree("console_dummy,eth2")
        row = _cell_abilities("console_dummy,eth2").row(Ability.CALIBRATE_BY_HAND)
        calibration = HandEyeCalibration.from_tree(loaded, rig_id="cam_left", mode="eye_to_hand",
                                                   options=SweepOptions(freedrive=True))
        self.assertEqual(calibration.check().refusal, row.reason)

    def test_the_hand_verbs_and_the_handling_verbs(self) -> None:
        from src.robot.execution.robot import Robot

        abilities = _cell_abilities("ursim")
        robot = Robot.from_tree(_tree("ursim"))
        self.assertEqual(robot.grasp(40.0).error, abilities.row(Ability.HAND).reason)
        self.assertEqual(robot.release().error, abilities.row(Ability.HAND).reason)
        self.assertEqual(robot.pick(_POSE, 40.0).message, abilities.row(Ability.HANDLE).reason)
        self.assertEqual(robot.place(_POSE).message, abilities.row(Ability.HANDLE).reason)

    def test_the_motion_verbs_on_a_connected_arm(self) -> None:
        """A UR on the ik planner: the verbs say "not connected" first today, so the arm is connected to compare."""
        from src.robot.execution.cell_lock import lock_path_for
        from src.robot.execution.robot import Robot
        from tests._fake_transport import chain_of, fake_transport

        row = _cell_abilities("ursim").row(Ability.MOVE)
        self.assertIs(row.status, AbilityStatus.REFUSED)
        self.addCleanup(lambda: lock_path_for("ur@127.0.0.1").unlink(missing_ok=True))
        with mock.patch("src.robot.drivers.ur.arm.URTransport",
                        lambda *_a: fake_transport(connected=False, kinematics=chain_of("ur5e"))):
            robot = Robot.from_tree(_tree("ursim"), gripper=None)
            with robot.connected():
                answers = [robot.move(_POSE), robot.move_joints([0.1, -1.4, 1.6, -1.77, -1.57, 0.2]), robot.home()]
        for report in answers:
            self.assertEqual(report.result.message, row.reason)

    def test_the_pick_service_on_a_substituted_hand(self) -> None:
        from src.robot.execution.cell import Cell
        from src.robot.execution.lifecycle import NoRealGripper

        loaded = _tree(None)
        cell = Cell.rehearsal(loaded.robot, data_dir=loaded.root)
        row = cell.abilities().row(Ability.PICK)
        self.assertIs(row.status, AbilityStatus.REFUSED)
        cell.build()
        with self.assertRaises(NoRealGripper) as refused:
            with cell.connected():
                pass
        self.assertEqual(str(refused.exception), row.reason)


@contextlib.contextmanager
def _row_says(ability: Ability, sentence: str) -> Any:
    """Every row of ``ability`` REFUSED with ``sentence``: a function that asks its row refuses with it (P0i i3)."""
    from dataclasses import replace

    from src.robot.execution import abilities
    from src.robot.execution.abilities import AbilityRow

    swapped = tuple(
        replace(d, ask=lambda _parts, a=d.ability: AbilityRow(a, AbilityStatus.REFUSED, sentence))
        if d.ability is ability else d for d in abilities.DECLARATIONS)
    with mock.patch.object(abilities, "DECLARATIONS", swapped):
        yield


class EachFunctionAsksItsRowTests(unittest.TestCase):
    """P0i i3: each function refuses with what its row says, so the desk and the function cannot drift apart. The
    row is swapped for one that refuses with a marker; a function that kept a sentence of its own does not say it."""

    _SAID = "the row says no (P0i i3 marker)"

    def test_teach(self) -> None:
        """On an arm a person CAN guide, so a teach that kept its own Protocol check would not refuse at all."""
        from src.robot.execution.hand_guiding import HandGuidingRefused
        from src.robot.execution.robot import Robot
        from src.robot.execution.teach import teach_poses

        robot = Robot.from_tree(_tree("ursim"), gripper=None)
        with _row_says(Ability.TEACH, self._SAID), self.assertRaises(HandGuidingRefused) as refused:
            teach_poses(robot, out=lambda _line: None)
        self.assertEqual(str(refused.exception), self._SAID)

    def test_the_io_bench(self) -> None:
        """On an arm that switches pins, so a bench that kept its own Protocol check would read the bank and exit 0."""
        from src.robot.execution.io_bench import __main__ as bench
        from tests.test_io_bench_pins import _CliArm

        loaded = _tree("console_dummy")
        said = io.StringIO()
        with _row_says(Ability.IO_BENCH, self._SAID), contextlib.redirect_stdout(said), \
                mock.patch.object(bench, "_load_robot_config", lambda *_a: loaded.robot), \
                mock.patch("src.robot.drivers.create_arm", lambda *_a, **_k: _CliArm()):
            code = bench.main(["--read"])
        self.assertEqual(code, 1)
        self.assertIn(f"REFUSED: {self._SAID}\n", said.getvalue())

    def test_the_planner_start(self) -> None:
        """On a UR that has a planner to start, so a start that kept its own Protocol check would reach the sidecar."""
        from src.robot.execution.cell import Cell

        cell = Cell.from_tree(_tree("ursim,ursim_curobo"))
        sidecar = mock.patch("src.robot.drivers.planned.arm.CuroboPlanClient",
                             side_effect=AssertionError("the sidecar was started"))
        with _row_says(Ability.PLANNER_START, self._SAID), sidecar:
            report = cell.start_planner()
        self.assertEqual((report.started, report.refusal), (False, self._SAID))

    def test_the_hand_verbs(self) -> None:
        from src.robot.execution.robot import Robot

        robot = Robot.from_tree(_tree("console_dummy"))
        with _row_says(Ability.HAND, self._SAID):
            answers = [robot.grasp(40.0), robot.release()]
        self.assertEqual([hand.error for hand in answers], [self._SAID] * 2)

    def test_the_pick_service_build(self) -> None:
        """Refused by its row before any camera opens: the camera build is made to fail loudly if it is reached."""
        from src.robot.execution.autonomous_grasp import cells

        loaded = _tree("ursim")
        opened = mock.patch.object(cells, "build_real_components", side_effect=AssertionError("a camera was opened"))
        with _row_says(Ability.PICK, self._SAID), opened, self.assertRaises(cells.CellBuildRefused) as refused:
            cells.build_real_cell(loaded.robot, app_config=loaded.app_config, data_dir=loaded.root)
        self.assertEqual(str(refused.exception), self._SAID)

    def test_the_handling_verbs_ask_the_hand_row_and_the_move_row(self) -> None:
        from src.robot.execution.robot import Robot

        robot = Robot.from_tree(_tree("console_dummy"))
        with _row_says(Ability.HAND, self._SAID):
            self.assertEqual(robot.pick(_POSE, 40.0).message, self._SAID)
        with robot.connected(), _row_says(Ability.MOVE, self._SAID):
            answers = [robot.pick(_POSE, 40.0), robot.place(_POSE)]
        self.assertEqual([report.message for report in answers], [self._SAID] * 2)

    def test_the_motion_verbs(self) -> None:
        from src.robot.execution.robot import Robot

        robot = Robot.from_tree(_tree("console_dummy"), gripper=None)
        with robot.connected(), _row_says(Ability.MOVE, self._SAID):
            answers = [robot.move(_POSE), robot.move_joints([0.1, -1.4, 1.6, -1.77, -1.57, 0.2]), robot.home()]
        self.assertEqual([report.result.message for report in answers], [self._SAID] * 3)

    def test_calibration_by_hand_at_the_desk_and_at_the_build(self) -> None:
        """Changes behaviour on purpose: the desk passed a run by hand on an arm the build then refused."""
        from src.robot.execution.hand_eye import HandEyeCalibration, SweepOptions
        from src.robot.execution.hand_guiding import HandGuidingRefused
        from src.robot.execution.robot_parts import construct_arm

        loaded = _tree("console_dummy,eth2")
        for options in (SweepOptions(freedrive=True), SweepOptions(adjust=True, fixed_poses=[_POSE])):
            calibration = HandEyeCalibration.from_tree(loaded, rig_id="cam_left", mode="eye_to_hand", options=options)
            with self.subTest(freedrive=options.freedrive), _row_says(Ability.CALIBRATE_BY_HAND, self._SAID):
                self.assertEqual(calibration.check().refusal, self._SAID)
                # The build's half, on an arm a person CAN guide: a build that kept its own Protocol check would pass.
                with self.assertRaises(HandGuidingRefused) as refused:
                    calibration._refuse_without_hand_guiding(construct_arm(_tree("ursim").robot))
                self.assertEqual(str(refused.exception), self._SAID)

    def test_the_pick_build_keeps_the_chain_of_what_refused_it(self) -> None:
        """The build raised `from` the wrist body's own error before it asked the row (review of i3)."""
        from src.robot.execution.autonomous_grasp import cells
        from src.robot.execution.wrist_bodies import WristBodies, WristBodyRequired

        loaded = _tree("ursim")
        underlying = WristBodyRequired("the wrist camera's calibration does not load")
        with mock.patch.object(WristBodies, "from_config", side_effect=underlying), \
                mock.patch.object(cells, "build_real_components", side_effect=AssertionError("a camera was opened")), \
                self.assertRaises(cells.CellBuildRefused) as refused:
            cells.build_real_cell(loaded.robot, app_config=loaded.app_config, data_dir=loaded.root)
        self.assertEqual(str(refused.exception), str(underlying))
        self.assertIs(refused.exception.__cause__, underlying)

    def test_the_pick_build_reads_the_camera_half_of_its_own_tree(self) -> None:
        """Changes behaviour on purpose (review of i3): handed no camera half, the build read the repository's tree
        while the desk's rows read the tree at data_dir, and both ask the same pick row."""
        from src.config import loader
        from src.robot.execution.autonomous_grasp import cells

        loaded = _tree("ursim")
        reads: list[object] = []
        real = loader.load_config

        def load_config(data_dir: object = None, **kwargs: Any) -> Any:
            reads.append(data_dir)
            return real(data_dir, **kwargs)

        with mock.patch("src.config.load_config", load_config), \
                mock.patch.object(cells, "build_real_components", side_effect=AssertionError("cameras")), \
                self.assertRaises(AssertionError):
            cells.build_real_cell(loaded.robot, data_dir=loaded.root)
        self.assertEqual(reads, [loaded.root])


class WhatTheDeskAsksTests(unittest.TestCase):
    def test_a_ur_cell_offers_what_its_arm_implements(self) -> None:
        abilities = _cell_abilities("ursim,ursim_curobo")
        for ability in (Ability.MOVE, Ability.TEACH, Ability.IO_BENCH, Ability.ACKNOWLEDGE_FAULT):
            self.assertIs(abilities.row(ability).status, AbilityStatus.AVAILABLE, ability)
        self.assertIs(abilities.row(Ability.PLANNER_START).status, AbilityStatus.BENCH)

    def test_the_desk_commands_nothing(self) -> None:
        """The arm is built, never connected: its transport is sent nothing."""
        from tests._fake_transport import chain_of, fake_transport

        built: list[Any] = []

        def transport(*_a: Any) -> Any:
            built.append(fake_transport(connected=False, kinematics=chain_of("ur5e")))
            return built[-1]

        with mock.patch("src.robot.drivers.ur.arm.URTransport", transport):
            _cell_abilities("ursim,ursim_curobo")
        self.assertEqual(len(built), 1)
        self.assertEqual(built[0].tape, [])

    def test_a_robot_asks_its_own_handles_as_the_cell_asks_the_arm_its_tree_builds(self) -> None:
        from src.robot.execution.robot import Robot

        loaded = _tree("ursim")
        self.assertEqual(Robot.from_tree(loaded).abilities().to_dict(), _cell_abilities("ursim").to_dict())
        armless = Robot.from_tree(loaded, gripper=None).abilities()
        self.assertEqual(armless.row(Ability.HAND).reason, "this robot has no gripper")

    def test_a_robot_without_its_tree_leaves_the_rigs_to_the_bench(self) -> None:
        """Changes behaviour on purpose (review of i2): it was told to switch on a rig nobody had read."""
        from src.robot.execution.robot import Robot

        abilities = Robot.from_config(_tree("console_dummy,eth2").robot).abilities()
        self.assertIs(abilities.row(Ability.CALIBRATE).status, AbilityStatus.BENCH)
        self.assertIn("keeps no tree", abilities.row(Ability.CALIBRATE).reason)
        self.assertIs(abilities.row(Ability.CALIBRATE_BY_HAND).status, AbilityStatus.REFUSED, "no hand guiding first")


if __name__ == "__main__":
    unittest.main()
