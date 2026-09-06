"""An arm says what it enforces, and silence is never read as consent.

⛔⛔ **THE DEFECT THIS GUARDS, AND IT ALREADY HAPPENED.** `scripts/examples/03_pick.py:7-14` records
it: the example's predecessor printed "cleared safety" and its README claimed the run "clears all six
safety guards", while driving a `DummyRobotArm` whose own docstring says it carries no preflight and
simply drives. **No guard ran.** The fix at the time was for that one script to report which layers
ran; this makes the question askable of any arm, including one a customer wrote.

⚠ **WHY THESE TESTS MATTER MORE THAN USUAL RIGHT NOW.** The suite is being rebuilt for the prod
environment, so the durable protection is the ATTESTATION ITSELF, which is read at runtime and
printed in the report. These tests protect the reading; the printing is what protects the cell.
"""

from __future__ import annotations

import json
import unittest

from src.contracts import Rendered, Structured
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.safety import SafetyAttestation, SafetyGated, SafetyPosture, SafetyPreflight


class _Raises:
    @property
    def safety_preflight(self) -> object:
        raise RuntimeError("half-constructed driver")


class _Garbage:
    @property
    def safety_preflight(self) -> object:
        return "not a preflight at all"


class _Silent:
    """A `RobotArm` a customer wrote, which never heard of this Protocol."""


class _StatedNone:
    safety_preflight = None


class TheFourPosturesTests(unittest.TestCase):

    def test_an_arm_that_says_nothing_is_unstated(self) -> None:
        self.assertIs(SafetyAttestation.of(_Silent()).posture, SafetyPosture.UNSTATED)

    def test_stating_none_is_different_from_saying_nothing(self) -> None:
        """⛔ THE DISTINCTION THE PROTOCOL EXISTS FOR. Both are unsafe to drive, and one is a
        decision while the other is a gap. An operator has to be able to tell them apart."""
        self.assertIs(SafetyAttestation.of(_StatedNone()).posture, SafetyPosture.UNGATED)
        self.assertIs(SafetyAttestation.of(_Silent()).posture, SafetyPosture.UNSTATED)

    def test_a_wired_but_empty_pipeline_is_its_own_posture(self) -> None:
        """⛔ THE MOST MISLEADING STATE OF THE FOUR. Everything is wired, `evaluate()` runs, and it
        can refuse nothing. In every log that is indistinguishable from safety."""
        class Empty:
            safety_preflight = SafetyPreflight(guards=[])

        attestation = SafetyAttestation.of(Empty())
        self.assertIs(attestation.posture, SafetyPosture.EMPTY)
        self.assertFalse(attestation.enforced)

    def test_a_real_pipeline_reports_its_guards_in_order(self) -> None:
        from src.config.schema.robot import RobotConfig

        cfg = RobotConfig()

        class Gated:
            safety_preflight = SafetyPreflight.from_safety_config(cfg.safety, cfg.workspace_limits)

        attestation = SafetyAttestation.of(Gated())
        self.assertIs(attestation.posture, SafetyPosture.GATED)
        self.assertTrue(attestation.enforced)
        self.assertIn("workspace", attestation.guards)
        self.assertGreaterEqual(len(attestation.guards), 1)

    def test_a_family_switched_off_is_NAMED_not_merely_absent(self) -> None:
        """⛔⛔ THE ABSENCE IS THE SAFETY FACT, and it was the one thing not reported.

        `guards` says what will run, so an operator reads "6 guard(s)" and stops. An `enforce: false`
        block removes that family's surface ENTIRELY, as `from_safety_config`'s own comment puts it:
        it "does not just silence the log; the safety surface for that family is *gone*". A cell with
        self-collision off and one with it on both printed a working pipeline.
        """
        from src.config.schema.robot import RobotConfig

        cfg = RobotConfig()
        off = cfg.safety.model_copy(update={
            "self_collision": cfg.safety.self_collision.model_copy(update={"enforce": False}),
        })

        class Partial:
            safety_preflight = SafetyPreflight.from_safety_config(off, cfg.workspace_limits)

        attestation = SafetyAttestation.of(Partial())
        self.assertIn("self_collision", attestation.omitted)
        self.assertNotIn("self_collision", attestation.guards)
        self.assertIn("not enforced: self_collision", attestation.render())
        self.assertIn("self_collision", attestation.to_dict()["omitted"])

    def test_a_full_pipeline_omits_nothing_and_says_nothing(self) -> None:
        """The line is absent when there is nothing to report, rather than saying "none". A cell with
        every family enforced should not have to read a sentence about what is missing."""
        from src.config.schema.robot import RobotConfig

        cfg = RobotConfig()

        class Gated:
            safety_preflight = SafetyPreflight.from_safety_config(cfg.safety, cfg.workspace_limits)

        attestation = SafetyAttestation.of(Gated())
        self.assertEqual(attestation.omitted, ())
        self.assertNotIn("not enforced", attestation.render())

    def test_the_workspace_guard_can_never_be_reported_as_omitted(self) -> None:
        """⚠ IT IS WIRED UNCONDITIONALLY, being the one non-negotiable surface. Seeing it in
        `omitted` would mean the property is wrong, not that the cell is."""
        from src.config.schema.robot import RobotConfig

        cfg = RobotConfig()
        nothing_enforced = cfg.safety.model_copy(update={
            field: getattr(cfg.safety, field).model_copy(update={"enforce": False})
            for field in ("joint_limits", "ik_quality", "self_collision", "payload",
                          "motion_continuity")
        })
        preflight = SafetyPreflight.from_safety_config(nothing_enforced, cfg.workspace_limits)
        self.assertNotIn("workspace", preflight.omitted_guards)
        self.assertIn("workspace", preflight.guard_names)


class FailClosedTests(unittest.TestCase):
    """⛔ `enforced` IS THE PROPERTY A CALLER BRANCHES ON, and it is true for exactly one posture."""

    def test_only_gated_counts_as_enforced(self) -> None:
        for obj in (_Silent(), _StatedNone(), _Raises(), _Garbage()):
            with self.subTest(type(obj).__name__):
                self.assertFalse(SafetyAttestation.of(obj).enforced)

    def test_asking_never_raises(self) -> None:
        """⛔⛔ THE TWO CASES THAT BROKE IT, both found by probing rather than by reading.

        `runtime_checkable` protocol matching goes through `hasattr`, which EVALUATES the property,
        so a driver whose `safety_preflight` raises blew up inside `isinstance` itself while the
        handler sat one line below. Then an arm answering with a non-preflight raised
        `AttributeError` on `guard_names`, one line further on. This is called where an operator is
        deciding whether it is safe to start; a traceback there replaces the answer.
        """
        for obj in (_Raises(), _Garbage()):
            with self.subTest(type(obj).__name__):
                self.assertIs(SafetyAttestation.of(obj).posture, SafetyPosture.UNSTATED)

    def test_an_unusable_answer_is_not_read_as_gated(self) -> None:
        """An arm that answers with garbage has said something meaningless, which for a safety
        question is the same as having said nothing."""
        self.assertIs(SafetyAttestation.of(_Garbage()).posture, SafetyPosture.UNSTATED)


class EveryDriverWeShipAnswersTests(unittest.TestCase):
    """⚠ A COMPLETENESS CHECK, NOT THE GUARD. This cannot see a caller-supplied arm, which is
    exactly why the attestation is read at runtime and printed. It only ensures the drivers in this
    repository are not the ones staying silent."""

    #: The vendor packages that ship an arm. Only the MODULE is named; the classes are discovered,
    #: because the first version of this test hand-wrote them and guessed one wrong (the sim class
    #: is `IsaacRobotArm`, not `SimRobotArm`), which is the stale-list failure the convention
    #: rejects everywhere else. Importing these needs no vendor SDK: the house rule keeps those
    #: inside `connect()`.
    DRIVER_MODULES = ("dummy", "ur", "kuka", "sim")

    @classmethod
    def _arm_classes(cls) -> list[tuple[str, type]]:
        import importlib
        import inspect

        from src.robot.core import RobotArm

        # ⛔ THE MRO, NOT `issubclass`. `RobotArm` is a Protocol with non-method members, so
        # `issubclass(cls, RobotArm)` raises "Protocols with non-method members don't support
        # issubclass()". The drivers all inherit from it explicitly (`class DummyRobotArm(RobotArm)`),
        # so reading `__mro__` answers the same question without going through the Protocol
        # machinery at all.
        found: list[tuple[str, type]] = []
        for vendor in cls.DRIVER_MODULES:
            module = importlib.import_module(f"src.robot.drivers.{vendor}.arm")
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (obj.__module__ == module.__name__ and obj is not RobotArm
                        and RobotArm in getattr(obj, "__mro__", ())):
                    found.append((vendor, obj))
        return found

    def test_every_shipped_driver_implements_the_protocol(self) -> None:
        missing = [f"{vendor}.{cls.__name__}" for vendor, cls in self._arm_classes()
                   if not hasattr(cls, "safety_preflight")]
        self.assertEqual(missing, [], "these drivers cannot say what safety they enforce")

    def test_the_discovery_actually_found_the_drivers(self) -> None:
        """A test that asserts over an empty set passes loudest of all. Four vendor packages ship an
        arm, so anything less means the scan stopped seeing them."""
        found = self._arm_classes()
        self.assertGreaterEqual(len(found), len(self.DRIVER_MODULES), found)
        self.assertEqual({vendor for vendor, _ in found}, set(self.DRIVER_MODULES))

    def test_the_dummy_states_that_it_gates_nothing(self) -> None:
        """⛔ THE ARM THAT CAUSED THIS. A run on the dummy is a wiring rehearsal, never a safety
        demonstration, and it now reports that about itself instead of being silently believed."""
        attestation = SafetyAttestation.of(DummyRobotArm())
        self.assertIs(attestation.posture, SafetyPosture.UNGATED)
        self.assertFalse(attestation.enforced)
        self.assertIn("nothing gates its motion", attestation.render())

    def test_the_guard_can_fail(self) -> None:
        """⭐ THE SELF-FAILING CONTROL: the completeness check must be able to reject."""
        self.assertNotIsInstance(_Silent(), SafetyGated)
        self.assertIsInstance(DummyRobotArm(), SafetyGated)


class ItSatisfiesBothHalvesOfTheContractTests(unittest.TestCase):
    """⭐ THE FIRST CLASS IN THIS REPOSITORY TO CARRY BOTH. Measured on 2026-09-04, `render()` and
    `to_dict()` were disjoint everywhere else, which is why they are two Protocols. This one needs
    both: an operator reads the text before driving a cell, and the record log stores the mapping so
    a run's safety posture is recoverable from telemetry months later."""

    def test_it_is_rendered_and_structured(self) -> None:
        attestation = SafetyAttestation.of(DummyRobotArm())
        self.assertIsInstance(attestation, Rendered)
        self.assertIsInstance(attestation, Structured)

    def test_to_dict_is_plain_data(self) -> None:
        """No custom encoder: a wire format that only serialises by accident breaks on the first
        field somebody adds."""
        payload = SafetyAttestation.of(DummyRobotArm()).to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_render_obeys_the_contract(self) -> None:
        for obj in (DummyRobotArm(), _Silent(), _StatedNone()):
            with self.subTest(type(obj).__name__):
                text = SafetyAttestation.of(obj).render()
                self.assertTrue(text.isascii(), "printed to a cp1252 console")
                self.assertFalse(text.endswith("\n"), "the caller owns the line break")
                self.assertTrue(text.strip())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
