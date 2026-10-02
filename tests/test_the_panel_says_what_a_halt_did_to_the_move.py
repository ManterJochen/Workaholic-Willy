"""The console's halt record says what became of the move in flight, in the library's own words.

``HaltState.brake`` (library L9) tells a brake still running (``pending``) from one that held (``braked``), one nobody
saw end (``unconfirmed``: if the arm still moves, the emergency stop is the answer), a move that ended without one
(``ran_out``) and no move at all (``none``). ``CellOut.halted`` is ``HaltStateOut``: it carries the same field with the
same words, so the cockpit can read ``brake == "unconfirmed"`` directly instead of guessing from ``in_motion``, which
stays true after the move ended. A record that does not say leaves it unsaid (``null``) rather than guessing a word
from ``in_motion`` and ``braked``: a move that ran out with the brake off would read as one still braking.
"""

from __future__ import annotations

import typing
import unittest
from pathlib import Path

from pydantic import ValidationError

from api.schemas import HaltStateOut
from src.robot.core.arm_capabilities import HALT_BRAKE_OUTCOMES, HaltState

_REASON = "the operator pressed halt now"
_SCHEMA_TS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "api" / "schema.d.ts"


def _words(annotation: object) -> set[str]:
    """The string words a field's annotation allows, through an optional and a Literal."""
    words: set[str] = set()
    for arg in typing.get_args(annotation) or (annotation,):
        words |= {word for word in typing.get_args(arg) if isinstance(word, str)}
    return words


class TheHaltRecordTests(unittest.TestCase):
    def test_the_brakes_words_are_the_librarys(self) -> None:
        """Red before: the record had no such field, and the library's sixth field never reached the console."""
        field = HaltStateOut.model_fields.get("brake")
        self.assertIsNotNone(field, "HaltStateOut carries no brake")
        assert field is not None
        self.assertEqual(set(HALT_BRAKE_OUTCOMES), _words(field.annotation))

    def test_every_record_the_library_writes_reads_back_word_for_word(self) -> None:
        for brake in sorted(HALT_BRAKE_OUTCOMES):
            braked = brake == "braked"
            state = HaltState(reason=_REASON, requested_at=12.5, in_motion=brake != "none", braked=braked,
                              brake_s=0.29 if braked else None, brake=brake)
            with self.subTest(brake=brake):
                self.assertEqual(state.to_dict(), HaltStateOut(**state.to_dict()).model_dump())

    def test_a_record_that_does_not_say_leaves_it_unsaid(self) -> None:
        """A move that ran out with the brake off is ``in_motion`` and not ``braked``, as a brake still running is: only
        the arm's record can tell them apart, so nothing here guesses."""
        said = HaltStateOut(reason=_REASON, requested_at=1.0, in_motion=True, braked=False)
        self.assertIsNone(said.brake)

    def test_a_word_the_library_never_says_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            HaltStateOut(reason=_REASON, requested_at=1.0, brake="stopped")  # type: ignore[arg-type]

    def test_the_frontend_types_carry_the_field(self) -> None:
        """``schema.d.ts`` is regenerated from the OpenAPI document; a field the browser cannot read is not there."""
        text = _SCHEMA_TS.read_text(encoding="utf-8")
        start = text.index("HaltStateOut: {")
        block = text[start:text.index("\n        };", start)]
        self.assertIn("brake?:", block)
        for word in HALT_BRAKE_OUTCOMES:
            self.assertIn(f'"{word}"', block)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
