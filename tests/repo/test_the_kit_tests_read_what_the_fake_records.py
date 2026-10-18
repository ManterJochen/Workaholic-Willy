"""The sabotage control of P0g slice 13: a test moved onto the fake transport fails when the fake stops recording.

A core test that read a MagicMock connection's calls asserted "nothing was sent" with ``assert_not_called``. Moved onto
the fake, the same negative reads the fake's tape, and a tape that records nothing answers "nothing was sent" for
every motion there is: a mis-wired fake, or one read from the wrong object, turns every fence green. So each file that
reads the tape is run here with every fake silenced (``tests._fake_transport.silenced``), and at least as many of its
tests must go red as did when it moved. A file whose count falls has a negative that no longer reads what the arm
sent; a file that reads the tape and is not listed has not proved it does.

The counts are failures and errors as ``unittest`` reports them, a failing subtest counting once, measured on the day
each file moved.
"""

from __future__ import annotations

import importlib
import unittest

from tests._fake_transport import silenced

#: Module -> the tests of it that must fail with the fake silenced. A number may only rise, or fall with a reason.
_READERS: dict[str, int] = {
    "tests.robot.execution.test_calibration_joint_stations": 2,
    "tests.robot.safety.test_a_motion_without_a_camera_world_is_refused": 1,
    # aurora 17; Willy's joint lines and the gate at the send 20; 15 since 2026-10-17, when the gate became every
    # arm's and the test of a tree that asks for none went with its key.
    "tests.robot.test_the_arm_holds_its_own_steady_gate": 15,
    "tests.robot.test_a_stopped_control_script_is_a_fault_not_a_motion": 5,
    "tests.robot.safety.planning.test_every_verb_meets_the_camera_world": 7,
    "tests.robot.safety.test_a_checked_line_is_sampled_and_solved": 13,
    "tests.robot.safety.test_a_joint_move_is_judged_as_a_path": 10,
    "tests.robot.drivers.planned.test_a_planned_path_runs_as_few_moves_as_judged": 3,
    "tests.robot.safety.test_a_refused_home_names_its_gate": 20,
    "tests.robot.safety.test_ungated_motion_surfaces": 7,
    "tests.robot.safety.planning.test_the_descriptor_names_its_hand": 1,
    "tests.robot.drivers.planned.test_a_planned_arm_types_every_verdict": 12,
    "tests.robot.safety.test_a_planned_arm_hands_the_controller_its_frame": 7,
    # 21 when it moved; 20 once the refusal of a transport with no controller TCP came before the socket (the slice
    # 15 review): that test asserts an empty tape, which no silence can turn red, and its class's control does. 22
    # with the refused reconnect's two subtests, which read the rollback off the tape. 20 since 2026-10-17, when the
    # owner's cleanup took ``safety.payload.enforce``: the bare flange's connect, whose tape read one ``connect``, left
    # with it, and the connect of a bare and a weighed flange, two subtests, is the weighed one's alone.
    "tests.robot.drivers.planned.test_a_planned_arm_connects_through_its_transport": 20,
}


def _red(module_name: str, *, silence: bool) -> tuple[int, int]:
    """(tests run, failures and errors) for ``module_name``, with every fake silenced or not."""
    module = importlib.import_module(module_name)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TestResult()
    if silence:
        with silenced():
            suite.run(result)
    else:
        suite.run(result)
    return result.testsRun, len(result.failures) + len(result.errors)


class EveryMovedNegativeReadsTheTapeTests(unittest.TestCase):
    def test_each_file_that_reads_the_tape_goes_red_when_the_fake_stops_recording(self) -> None:
        for module_name, least in _READERS.items():
            with self.subTest(module=module_name):
                run, red = _red(module_name, silence=True)
                self.assertGreater(run, 0, "the module ran no tests")
                self.assertGreaterEqual(red, least, "a negative passes on a fake that recorded nothing")

    def test_the_same_files_are_green_when_the_fake_records(self) -> None:
        """The control: the red above is the silence, not a file that was already failing."""
        for module_name in _READERS:
            with self.subTest(module=module_name):
                self.assertEqual(0, _red(module_name, silence=False)[1])

    def test_the_silence_is_lifted_after_the_block(self) -> None:
        from tests._fake_transport import fake_transport

        with silenced():
            fake_transport().read_joints()
        fake = fake_transport()
        fake.read_joints()
        self.assertEqual(["read_joints"], [call.verb for call in fake.tape])


if __name__ == "__main__":
    unittest.main()
