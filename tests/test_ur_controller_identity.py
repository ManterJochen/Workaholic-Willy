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

from src.robot.core import ControllerProvenance, ReportsControllerProvenance, RobotConnectionError
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
    """The attributes ``_verify_controller_model`` touches, and nothing else: the model it asks the transport since
    P0g slice 7 (``ReportsControllerModel``), over the connection that answers it."""

    def __init__(self, declared: str, reported: str | None) -> None:
        from tests._ur_transport import over

        self._capabilities = MagicMock(model=declared)
        connection = MagicMock()
        connection.controller_model.return_value = reported
        self._transport = over(connection)
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


class TheUrControllerSaysWhatItIsTests(unittest.TestCase):
    """The proof rule belongs to the controller, so the driver holds it and answers for itself."""

    @staticmethod
    def _provenance(serial: str | None, host: str | None, *, raises: bool = False) -> ControllerProvenance:
        from tests._ur_transport import over

        arm = URRobotArm.__new__(URRobotArm)
        # Nothing about a hand-built arm was built, its transport included: the connection sits behind one (P0e E7).
        arm._transport = over(MagicMock(ip=host))
        arm.logger = MagicMock()
        if raises:
            arm._conn.controller_serial.side_effect = RuntimeError("socket gone")
        else:
            arm._conn.controller_serial.return_value = serial
        return arm.controller_provenance()

    def test_ursim_is_proven_a_simulator(self) -> None:
        # The measured pair: URSim's placeholder serial AND a loopback address.
        provenance = self._provenance("20195399999", "127.0.0.1")
        self.assertIs(provenance.is_simulator, True)
        self.assertEqual((provenance.serial, provenance.host), ("20195399999", "127.0.0.1"))

    def test_ursim_cb3_is_proven_a_simulator(self) -> None:
        # MEASURED 2026-10-02: URSim CB3 (PolyScope 3.15.8) answers "get serial number" with 2018309999.
        self.assertIs(self._provenance("2018309999", "127.0.0.1").is_simulator, True)

    def test_a_real_serial_on_a_real_address_makes_NO_claim(self) -> None:
        provenance = self._provenance("20195312345", "192.168.1.10")
        self.assertIsNone(provenance.is_simulator)
        # The evidence is still reported, so a person can judge it.
        self.assertEqual((provenance.serial, provenance.host), ("20195312345", "192.168.1.10"))

    def test_either_half_alone_is_not_proof(self) -> None:
        self.assertIsNone(self._provenance("20195399999", "192.168.1.10").is_simulator)
        self.assertIsNone(self._provenance("20195312345", "127.0.0.1").is_simulator)
        self.assertIsNone(self._provenance("2018309999", "192.168.1.10").is_simulator)

    def test_the_placeholder_is_all_digits_and_has_its_length(self) -> None:
        # The owner's rule of 2026-10-02: 10 digits ending 9999 (a CB3) or 11 ending 99999 (an e-Series), nothing else.
        for serial in ("9999", "99999", "309999", "201830999999", "2018309999x", "201830 9999", "20183099990"):
            with self.subTest(serial=serial):
                self.assertIsNone(self._provenance(serial, "127.0.0.1").is_simulator)

    def test_it_is_never_False(self) -> None:
        # The whole contract: absence of proof is absence of a claim, not a claim of the opposite.
        for serial, host in [(None, None), ("x", None), (None, "127.0.0.1"), ("20195312345", "10.0.0.5")]:
            with self.subTest(serial=serial, host=host):
                self.assertIsNot(self._provenance(serial, host).is_simulator, False)
        with self.assertRaises(ValueError):
            ControllerProvenance(serial="1", host="h", is_simulator=False)

    def test_a_controller_that_will_not_answer_says_nothing(self) -> None:
        self.assertTrue(self._provenance(None, "127.0.0.1", raises=True).says_nothing())

    def test_the_ur_driver_declares_the_capability(self) -> None:
        self.assertIsInstance(URRobotArm.__new__(URRobotArm), ReportsControllerProvenance)


class SimulatorDisclosureTests(unittest.TestCase):
    """What the console does with the answer: report the evidence, claim only the proof, never raise."""

    @staticmethod
    def _snapshot(provenance: object) -> dict:
        from api.telemetry import _add_simulator_evidence

        class _Arm:
            def controller_provenance(self) -> object:
                if isinstance(provenance, Exception):
                    raise provenance
                return provenance

        snapshot: dict = {}
        _add_simulator_evidence(_Arm(), snapshot)
        return snapshot

    def test_a_proven_simulator_is_disclosed_with_its_evidence(self) -> None:
        snapshot = self._snapshot(ControllerProvenance(serial="20195399999", host="127.0.0.1", is_simulator=True))
        self.assertIs(snapshot["controller_is_simulator"], True)
        self.assertEqual((snapshot["controller_serial"], snapshot["controller_host"]), ("20195399999", "127.0.0.1"))

    def test_an_unproven_controller_reports_evidence_and_claims_nothing(self) -> None:
        snapshot = self._snapshot(ControllerProvenance(serial="20195312345", host="192.168.1.10"))
        self.assertIsNone(snapshot.get("controller_is_simulator"))
        self.assertEqual(snapshot["controller_serial"], "20195312345")

    def test_a_driver_without_the_capability_is_untouched(self) -> None:
        from api.telemetry import _add_simulator_evidence

        snapshot: dict = {}
        _add_simulator_evidence(MagicMock(spec=["capabilities"]), snapshot)
        self.assertEqual(snapshot, {})

    def test_an_arm_that_says_nothing_adds_no_keys(self) -> None:
        """"Cannot tell" and "never asked" render alike: as nothing at all, not as empty evidence."""
        self.assertEqual(self._snapshot(ControllerProvenance()), {})

    def test_an_answer_that_is_not_a_provenance_is_ignored(self) -> None:
        """A MagicMock satisfies a runtime_checkable Protocol, so the panel checks the answer too."""
        self.assertEqual(self._snapshot(MagicMock()), {})

    def test_a_raising_arm_never_breaks_the_status_panel(self) -> None:
        self.assertEqual(self._snapshot(RuntimeError("socket gone")), {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
