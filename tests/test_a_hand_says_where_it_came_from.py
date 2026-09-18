"""A hand written from its declared dimensions is planned and guarded like any other, and every reader says so (B7).

The owner settled that a bundle built from a registry block has EQUAL STANDING with a scan. Nothing here refuses one.
What is held is the other half of that decision: a reader must never have to guess which of the two it is holding,
because the difference is not visible in the geometry and it costs room.

⚠ **AND THE ROOM IS MEASURED.** The two shipped hands that carry both a baked bundle and a full set of dimensions
needed 5.73 mm and 9.50 mm of inflation, at their own declared grasp centres, before an envelope enclosed the body it
stands for. A cell running on an envelope plans with that much less clearance than a cell running on a scan, and the
only place that fact exists is the bundle's own record.

⭐ **ABSENCE IS THE ANSWER FOR A SCAN.** A baked bundle carries no source key and never will: every committed hand was
written before this record existed. So a bundle with no `hand__source` reads as a bake rather than as unknown, and
that is the control this file keeps, so the reader cannot pass by finding nothing anywhere.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.robot.safety.planning.environment import (
    BAKED_SOURCE,
    DIMENSIONS_SOURCE,
    HandProvenance,
    hand_provenance,
)

_ROOT = Path(__file__).resolve().parents[1]
_BUNDLES = _ROOT / "src" / "robot" / "safety" / "data"
_SHIPPED = ("robotiq_2f85", "robotiq_hande", "schunk_egu50")

#: The inflation this file writes its envelopes with. Above both measured hands, so a bundle written here is one a
#: cell could actually run; the number itself is the caller's and this file states it rather than importing one.
_INFLATION_MM = 10.0


def _writer():
    path = _ROOT / "src" / "robot" / "safety" / "planning" / "robot" / "hand_from_dimensions.py"
    spec = importlib.util.spec_from_file_location("_hand_from_dimensions_for_provenance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TheCommittedHandsReadAsScanned(unittest.TestCase):
    """Every hand this repository ships was baked, and the reader says that without any of them being rewritten."""

    def test_each_shipped_hand_reads_as_a_bake(self) -> None:
        """Which arms a hand may carry is not provenance: it is measured, one evidence file per combination (S22)."""
        for hand in _SHIPPED:
            with self.subTest(hand=hand):
                provenance = hand_provenance(hand)
                assert provenance is not None, f"{hand} has no bundle to read"
                self.assertEqual(provenance.source, BAKED_SOURCE)
                self.assertIsNone(provenance.inflation_mm)
                self.assertFalse(provenance.from_dimensions)

    def test_a_name_with_no_bundle_answers_nothing_rather_than_a_guess(self) -> None:
        """`None` is a real answer here: the hand the arm bundles carry has no file of its own either."""
        self.assertIsNone(hand_provenance("a_hand_nobody_has_written"))

    def test_the_sentence_names_the_source(self) -> None:
        provenance = hand_provenance("schunk_egu50")
        assert provenance is not None
        said = provenance.render()
        self.assertIn("baked", said)
        self.assertNotIn("inflated", said, "a scanned hand has no inflation to report")
        self.assertNotIn("proven on", said, "admission is evidence now, and a provenance line cannot claim it")


class AnEnvelopeSaysItIsOne(unittest.TestCase):
    """⭐ The half that matters: a bundle the writer produced reads back as an envelope, with its inflation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.writer = _writer()

    def test_a_written_hand_reads_as_dimensions_and_carries_the_stated_inflation(self) -> None:
        from src.config.grippers import load_gripper

        with TemporaryDirectory() as folder:
            out = Path(folder) / "schunk_egu50_hand_meshes.npz"
            self.writer.write_bundle("schunk_egu50", out, inflation_mm=_INFLATION_MM, origin="flange")
            provenance = hand_provenance("schunk_egu50", folder)
            with np.load(out) as written:
                self.assertNotIn("hand__admitted_arms", written.files, "the writer records no arm since S22")

        assert provenance is not None
        self.assertEqual(provenance.source, DIMENSIONS_SOURCE)
        self.assertTrue(provenance.from_dimensions)
        self.assertAlmostEqual(provenance.inflation_mm or 0.0, _INFLATION_MM, places=9)
        self.assertIsNotNone(load_gripper("schunk_egu50").jaw, "the registry block this was built from is still there")

    def test_the_sentence_names_the_envelope_and_the_millimetres_it_grew_by(self) -> None:
        with TemporaryDirectory() as folder:
            out = Path(folder) / "robotiq_hande_hand_meshes.npz"
            self.writer.write_bundle("robotiq_hande", out, inflation_mm=_INFLATION_MM, origin="mounting_face")
            said = (hand_provenance("robotiq_hande", folder) or HandProvenance("x", BAKED_SOURCE, None)).render()
        self.assertIn("envelope", said)
        self.assertIn("declared dimensions", said)
        self.assertIn("10 mm", said)
        self.assertNotIn("proven on", said, "an envelope is not proven on anything by being written")

    def test_a_bundle_with_no_record_at_all_reads_as_a_bake(self) -> None:
        """The control. Every committed hand is this shape, and reading them as unknown would be a false alarm."""
        with TemporaryDirectory() as folder:
            path = Path(folder) / "bare_hand_meshes.npz"
            np.savez(path, gripper__v=np.zeros((1, 3)), gripper__f=np.zeros((1, 3), dtype=np.int64))
            provenance = hand_provenance("bare", folder)
        assert provenance is not None
        self.assertEqual(provenance.source, BAKED_SOURCE)

    def test_the_wire_half_is_a_view_of_the_same_answer(self) -> None:
        provenance = HandProvenance("acme_2f", DIMENSIONS_SOURCE, 10.0)
        self.assertEqual(provenance.to_dict(), {
            "hand": "acme_2f",
            "source": DIMENSIONS_SOURCE,
            "inflation_mm": 10.0,
        })


class ThePlannerAndTheGuardBothHoldIt(unittest.TestCase):
    """The resolved hand carries the record, so the planner side and the guard side read one answer and not two."""

    def test_a_resolved_hand_carries_its_bundle_s_provenance(self) -> None:
        from src.config.loader import load_robot_config
        from src.contracts import chosen
        from src.robot.safety.planning.hand import planner_hand

        hand = planner_hand(load_robot_config(profile="hande"))
        assert chosen(hand), "the hande profile names a hand"
        assert hand.provenance is not None, "a hand with a bundle of its own should carry what it says"
        self.assertEqual(hand.provenance.hand, hand.model)
        self.assertFalse(hand.models_an_envelope, "the shipped hands are all scans")

    def test_the_envelope_flag_is_the_provenance_and_not_a_second_opinion(self) -> None:
        from src.robot.safety.planning.hand import PlannerHand

        scanned = HandProvenance("acme_2f", BAKED_SOURCE, None)
        envelope = HandProvenance("acme_2f", DIMENSIONS_SOURCE, 10.0)
        for provenance, expected in ((None, False), (scanned, False), (envelope, True)):
            with self.subTest(provenance=provenance):
                hand = PlannerHand(model="acme_2f", sphere_map=Path("x.yml"), origin="flange", guard_variant="acme_2f",
                                   coupling_mm=0.0, jaw=None, provenance=provenance)
                self.assertEqual(hand.models_an_envelope, expected)


class TheRealCellChecklistSaysIt(unittest.TestCase):
    """An operator reading the bring-up list learns it there, where the rest of the cell's geometry is reported."""

    def test_the_row_stays_quiet_for_a_scanned_hand(self) -> None:
        from src.robot.execution.real_cell.preflight import _hand_geometry_source

        self.assertIsNone(_hand_geometry_source("robotiq_hande"))
        self.assertIsNone(_hand_geometry_source("a_hand_nobody_has_written"))

    def test_the_row_names_the_envelope_for_a_hand_written_from_its_dimensions(self) -> None:
        from src.robot.execution.real_cell.preflight import _hand_geometry_source

        with TemporaryDirectory() as folder:
            _writer().write_bundle("schunk_egu50", Path(folder) / "schunk_egu50_hand_meshes.npz",
                                   inflation_mm=_INFLATION_MM, origin="flange")
            said = _hand_geometry_source("schunk_egu50", folder)
        assert said is not None, "the checklist said nothing about a hand nobody scanned"
        self.assertIn("envelope", said)
        self.assertIn("10 mm", said)


if __name__ == "__main__":
    unittest.main()
