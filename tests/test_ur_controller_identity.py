"""The controller is asked what it is, and the answer is used honestly.

Two behaviours, both found by driving the operator console against URSim on 2026-08-19.

**The model check.** ``ur.model`` keys the safety DH chain, the exact-mesh collision bundle and the
cuRobo robot config, and nothing ever compared it against the controller. A UR3e container driven with
the shipped default ``ur.model: ur5e`` was refused -- correctly, by ``_verify_tool_frame`` -- with
"the controller is actually running a tool frame 177.4 mm away", which sends the reader to the pendant's
TCP. The real answer was one dashboard call away.

Its failure direction is deliberate and asserted below: **fail-closed on evidence, fail-open on its
absence.** Only a positive, parsed disagreement refuses. A dashboard that will not answer, or answers
something unparseable, is logged and allowed -- because a false refusal at connect is not a safe
failure, it is a cell that will not start, and the tool-frame check downstream still catches the
dangerous case on its own.

**The simulator disclosure.** URSim reports ``simulated: False`` and is right to: that flag is about the
DRIVER, and URSim is genuine UR controller software on the genuine ports. Measured, it produced a
complete, plausible telemetry panel -- which the investor demo page would have rendered as "physical
arm" over a Docker container. ``controller_is_simulator`` is therefore three-valued and NEVER ``False``:
``True`` on proof, ``None`` for "cannot tell". Claiming a real arm is a simulator is the worse error of
the two, so absence of proof stays absence of a claim.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.robot.core import RobotConnectionError
from src.robot.drivers.ur.arm import URRobotArm


class ModelSizeParsingTests(unittest.TestCase):
    """The comparison is on the size class, because the two sides spell it differently."""

    def test_the_controller_spelling_and_the_config_spelling_agree(self) -> None:
        # MEASURED: URSim's e-Series container reports 'UR3' for a UR3e. A string comparison against
        # 'ur3e' would refuse a correctly configured cell -- the exact false refusal this must avoid.
        self.assertEqual(URRobotArm._model_size("UR3"), URRobotArm._model_size("ur3e"))
        self.assertEqual(URRobotArm._model_size("UR5"), URRobotArm._model_size("ur5e"))

    def test_the_sizes_it_knows(self) -> None:
        for name, expected in [
            ("UR3", "3"), ("ur3e", "3"), ("UR5e CB3", "5"),
            ("UR10", "10"), ("ur16e", "16"), ("UR20", "20"),
        ]:
            with self.subTest(name=name):
                self.assertEqual(URRobotArm._model_size(name), expected)

    def test_anything_unrecognised_is_None_not_a_guess(self) -> None:
        for name in ["", None, "UR", "something else", "UR7", "ur1234"]:
            with self.subTest(name=name):
                self.assertIsNone(URRobotArm._model_size(name))


class _Arm:
    """The two attributes ``_verify_controller_model`` touches, and nothing else."""

    def __init__(self, declared: str, reported: str | None) -> None:
        self._capabilities = MagicMock(model=declared)
        self._conn = MagicMock()
        self._conn.controller_model.return_value = reported
        self.logger = MagicMock()

    verify = URRobotArm._verify_controller_model
    _model_size = URRobotArm._model_size
    _MODEL_SIZES = URRobotArm._MODEL_SIZES


class ModelMismatchTests(unittest.TestCase):
    def test_a_matching_pair_connects(self) -> None:
        _Arm("ur3e", "UR3").verify()  # must not raise

    def test_a_real_disagreement_refuses_and_names_both(self) -> None:
        with self.assertRaises(RobotConnectionError) as caught:
            _Arm("ur5e", "UR3").verify()
        message = str(caught.exception)
        self.assertIn("ur5e", message)
        self.assertIn("UR3", message)
        # The consequence, not just the fact: this is what makes the message actionable.
        self.assertIn("link lengths", message)

    def test_a_silent_dashboard_does_NOT_refuse(self) -> None:
        # Fail-open on absence of evidence. A cell that will not start is not a safe failure.
        _Arm("ur5e", None).verify()

    def test_an_unparseable_answer_does_NOT_refuse(self) -> None:
        _Arm("ur5e", "some future model").verify()

    def test_an_unrecognised_config_model_does_NOT_refuse(self) -> None:
        # There is no evidence of a mismatch when one side cannot be read at all.
        _Arm("mystery", "UR3").verify()


class SimulatorDisclosureTests(unittest.TestCase):
    """``controller_is_simulator`` is three-valued and never False."""

    @staticmethod
    def _snapshot(serial: str | None, host: str | None) -> dict:
        from api.telemetry import _add_simulator_evidence

        arm = MagicMock()
        arm._conn = MagicMock(ip=host)
        arm._conn.controller_serial.return_value = serial
        snapshot: dict = {}
        _add_simulator_evidence(arm, snapshot)
        return snapshot

    def test_ursim_is_proven_a_simulator(self) -> None:
        # The measured pair: URSim's placeholder serial AND a loopback address.
        snapshot = self._snapshot("20195399999", "127.0.0.1")
        self.assertIs(snapshot["controller_is_simulator"], True)
        self.assertEqual(snapshot["controller_serial"], "20195399999")

    def test_a_real_serial_on_a_real_address_makes_NO_claim(self) -> None:
        snapshot = self._snapshot("20195312345", "192.168.1.10")
        self.assertIsNone(snapshot.get("controller_is_simulator"))
        # The evidence is still reported, so a person can judge it.
        self.assertEqual(snapshot["controller_serial"], "20195312345")
        self.assertEqual(snapshot["controller_host"], "192.168.1.10")

    def test_either_half_alone_is_not_proof(self) -> None:
        self.assertIsNone(self._snapshot("20195399999", "192.168.1.10").get("controller_is_simulator"))
        self.assertIsNone(self._snapshot("20195312345", "127.0.0.1").get("controller_is_simulator"))

    def test_it_is_never_False(self) -> None:
        # The whole contract: absence of proof is absence of a claim, not a claim of the opposite.
        for serial, host in [(None, None), ("x", None), (None, "127.0.0.1"), ("20195312345", "10.0.0.5")]:
            with self.subTest(serial=serial, host=host):
                self.assertIsNot(self._snapshot(serial, host).get("controller_is_simulator"), False)

    def test_a_driver_with_no_dashboard_is_untouched(self) -> None:
        from api.telemetry import _add_simulator_evidence

        snapshot: dict = {}
        _add_simulator_evidence(MagicMock(spec=["capabilities"]), snapshot)
        self.assertEqual(snapshot, {})

    def test_a_raising_connection_never_breaks_the_status_panel(self) -> None:
        from api.telemetry import _add_simulator_evidence

        arm = MagicMock()
        arm._conn = MagicMock(ip="127.0.0.1")
        arm._conn.controller_serial.side_effect = RuntimeError("socket gone")
        snapshot: dict = {}
        _add_simulator_evidence(arm, snapshot)
        self.assertEqual(snapshot, {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
