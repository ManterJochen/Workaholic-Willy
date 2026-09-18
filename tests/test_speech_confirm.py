"""A person confirms what speech proposed before it becomes a prompt, and nothing else lets it through.

Voice never acts on its own (owner, 2026-09-11 and 2026-09-18). A transcript is a proposal, and its words
reach `PickRun(prompt=)` or the `Locator` only as `Confirmation.confirmed`, which holds text for an
explicit yes or a typed correction and None for every other answer. These tests pin the four answers the
owner named (an explicit yes, a correction that counts as typed, anything else refused, no terminal
meaning no person) and that nothing heard is never put to a person at all.
"""

from __future__ import annotations

import dataclasses
import io
import json
import subprocess
import sys
import unittest

from src.contracts import Rendered, Structured
from src.models.speech.confirm import (
    Confirmation,
    ConfirmationOutcome,
    Confirmer,
    PromptSource,
    TerminalConfirmer,
)
from src.models.speech.endpointing import UtteranceEnd
from src.models.speech.listener import ListenOutcome, Utterance
from src.models.speech.transcript import LanguageSource, Proposal, SpeechCheck, Transcript

_HEARD = "Nimm den roten Würfel"


class _Terminal(io.StringIO):
    """Typed answers on a stream that says it is a terminal."""

    def isatty(self) -> bool:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        return True


class _CountingConfirmer:
    """A `Confirmer` that answers with fixed text and counts how often it was asked."""

    def __init__(self, answer: str | None) -> None:
        self.answer = answer
        self.asked: list[str] = []

    def reply(self, proposed: str) -> str | None:
        self.asked.append(proposed)
        return self.answer


def _transcript(text: str = _HEARD, language: str | None = "de") -> Transcript:
    return Transcript(
        text=text, language=language, language_source=LanguageSource.DETECTED, duration_s=1.2,
        engine="whisper-transformers", model="org/whisper", device="cpu", latency_ms=60.0,
    )


def _proposal(text: str = _HEARD) -> Proposal:
    speech = SpeechCheck(
        heard_speech=bool(text), duration_s=1.2, checked_s=0.6, peak_probability=0.97 if text else 0.01,
        onset=0.5, min_speech_s=0.25, detector="silero-vad", latency_ms=3.0,
    )
    return Proposal(
        text=text,
        reason=None if text else "no speech was heard",
        speech=speech,
        transcript=_transcript(text) if text else None,
    )


def _utterance(outcome: ListenOutcome, text: str = _HEARD) -> Utterance:
    heard = outcome is ListenOutcome.HEARD
    return Utterance(
        outcome=outcome,
        transcript=_transcript(text) if heard else None,
        ended_by=UtteranceEnd.SOURCE_END if heard else None,
        speech_s=1.2 if heard else 0.0,
        waited_s=2.0,
        timeout_s=10.0,
        overflowed=False,
    )


def _ask(typed: str, text: str = _HEARD) -> tuple[Confirmation, io.StringIO]:
    shown = io.StringIO()
    confirmer = TerminalConfirmer.from_parts(stdin=_Terminal(typed), stdout=shown)
    return Confirmation.from_proposal(proposal=_proposal(text), confirmer=confirmer), shown


class TheFourAnswersTests(unittest.TestCase):
    def test_an_explicit_yes_confirms_the_words_as_spoken(self) -> None:
        for typed in ("y\n", "Y\n", "yes\n", "j\n", "ja\n", "  Ja  \n"):
            with self.subTest(typed=typed):
                confirmation, _ = _ask(typed)
                self.assertIs(confirmation.outcome, ConfirmationOutcome.CONFIRMED)
                self.assertEqual(confirmation.confirmed, _HEARD)
                self.assertIs(confirmation.source, PromptSource.SPOKEN)
                self.assertEqual(confirmation.language, "de")

    def test_a_correction_counts_as_typed(self) -> None:
        confirmation, shown = _ask("e\nNimm den gruenen Wuerfel\n")
        self.assertIs(confirmation.outcome, ConfirmationOutcome.CORRECTED)
        self.assertEqual(confirmation.confirmed, "Nimm den gruenen Wuerfel")
        self.assertIs(confirmation.source, PromptSource.TYPED)
        self.assertEqual(confirmation.proposed, _HEARD, "what was heard stays on the record")
        self.assertIn("The prompt: ", shown.getvalue())

    def test_anything_else_refuses(self) -> None:
        """A bare line of text is not a correction: a typo such as ``yy`` would otherwise become the
        grounding phrase of a pick. Only y and its three siblings confirm, and only e opens a correction."""
        for typed in ("\n", "n\n", "nein\n", "no\n", "yy\n", f"{_HEARD}\n", "", "e\n\n", "e\n"):
            with self.subTest(typed=typed):
                confirmation, _ = _ask(typed)
                self.assertIs(confirmation.outcome, ConfirmationOutcome.REFUSED)
                self.assertIsNone(confirmation.confirmed)
                self.assertIs(confirmation.source, PromptSource.SPOKEN)

    def test_no_terminal_means_no_person(self) -> None:
        """A service, a pipe or a test runner has nobody at the keyboard. Nothing is read and nothing is
        written, and nothing becomes a prompt."""
        typed = io.StringIO("y\n")  # answers waiting on a stream that is not a terminal
        shown = io.StringIO()
        confirmer = TerminalConfirmer.from_parts(stdin=typed, stdout=shown)
        confirmation = Confirmation.from_proposal(proposal=_proposal(), confirmer=confirmer)
        self.assertIs(confirmation.outcome, ConfirmationOutcome.NO_PERSON)
        self.assertIsNone(confirmation.confirmed)
        self.assertEqual(typed.tell(), 0, "a stream that is not a terminal was read")
        self.assertEqual(shown.getvalue(), "")

    def test_a_process_without_stdin_has_no_person_either(self) -> None:
        confirmer = TerminalConfirmer.from_parts(stdin=None, stdout=None)
        confirmation = Confirmation.from_proposal(proposal=_proposal(), confirmer=confirmer)
        self.assertIs(confirmation.outcome, ConfirmationOutcome.NO_PERSON)
        self.assertIsNone(confirmation.confirmed)

    def test_a_closed_stdin_has_no_person(self) -> None:
        closed = _Terminal("y\n")
        closed.close()
        confirmer = TerminalConfirmer.from_parts(stdin=closed, stdout=io.StringIO())
        self.assertIsNone(confirmer.reply(_HEARD))

    def test_the_question_names_what_was_heard_and_the_two_answers_that_let_it_through(self) -> None:
        _, shown = _ask("n\n")
        self.assertIn(f"Heard '{_HEARD}'", shown.getvalue())
        self.assertIn("y to run it as heard", shown.getvalue())
        self.assertIn("e to type the prompt yourself", shown.getvalue())


class NothingHeardIsNeverAskedTests(unittest.TestCase):
    def test_a_proposal_without_words_asks_nobody(self) -> None:
        confirmer = _CountingConfirmer("y")
        confirmation = Confirmation.from_proposal(proposal=_proposal(""), confirmer=confirmer)
        self.assertIs(confirmation.outcome, ConfirmationOutcome.NOTHING_HEARD)
        self.assertIsNone(confirmation.confirmed)
        self.assertEqual(confirmer.asked, [])

    def test_an_utterance_that_was_not_heard_asks_nobody(self) -> None:
        for outcome in (ListenOutcome.TIMED_OUT, ListenOutcome.SOURCE_ENDED):
            with self.subTest(outcome=outcome):
                confirmer = _CountingConfirmer("y")
                confirmation = Confirmation.from_utterance(
                    utterance=_utterance(outcome), confirmer=confirmer
                )
                self.assertIs(confirmation.outcome, ConfirmationOutcome.NOTHING_HEARD)
                self.assertIsNone(confirmation.transcript)
                self.assertEqual(confirmer.asked, [])

    def test_a_transcript_without_words_asks_nobody(self) -> None:
        confirmer = _CountingConfirmer("y")
        confirmation = Confirmation.from_utterance(
            utterance=_utterance(ListenOutcome.HEARD, text=""), confirmer=confirmer
        )
        self.assertIs(confirmation.outcome, ConfirmationOutcome.NOTHING_HEARD)
        self.assertEqual(confirmer.asked, [])


class TheUtteranceDoorTests(unittest.TestCase):
    def test_a_heard_utterance_is_asked_about_its_transcript(self) -> None:
        confirmer = _CountingConfirmer(_HEARD)
        confirmation = Confirmation.from_utterance(
            utterance=_utterance(ListenOutcome.HEARD), confirmer=confirmer
        )
        self.assertEqual(confirmer.asked, [_HEARD])
        self.assertIs(confirmation.outcome, ConfirmationOutcome.CONFIRMED)
        self.assertEqual(confirmation.confirmed, _HEARD)
        self.assertEqual(confirmation.language, "de")

    def test_typing_the_heard_words_again_confirms_them_as_spoken(self) -> None:
        confirmation = Confirmation.from_proposal(
            proposal=_proposal(), confirmer=_CountingConfirmer(f"  {_HEARD}  ")
        )
        self.assertIs(confirmation.outcome, ConfirmationOutcome.CONFIRMED)
        self.assertIs(confirmation.source, PromptSource.SPOKEN)

    def test_a_typed_correction_keeps_the_heard_language(self) -> None:
        confirmation = Confirmation.from_proposal(
            proposal=_proposal(), confirmer=_CountingConfirmer("the green cube")
        )
        self.assertIs(confirmation.outcome, ConfirmationOutcome.CORRECTED)
        self.assertEqual(confirmation.language, "de")

    def test_both_factories_are_keyword_only(self) -> None:
        with self.assertRaises(TypeError):
            Confirmation.from_proposal(_proposal(), _CountingConfirmer("y"))  # type: ignore[misc]
        with self.assertRaises(TypeError):
            Confirmation.from_utterance(  # type: ignore[misc]
                _utterance(ListenOutcome.HEARD), _CountingConfirmer("y")
            )

    def test_the_terminal_is_a_confirmer(self) -> None:
        self.assertIsInstance(TerminalConfirmer.from_parts(stdin=None, stdout=None), Confirmer)


class TheConfirmationReportTests(unittest.TestCase):
    def _each_outcome(self) -> list[Confirmation]:
        return [
            _ask("y\n")[0],
            _ask("e\nNimm den grünen Würfel\n")[0],
            _ask("n\n")[0],
            Confirmation.from_proposal(proposal=_proposal(), confirmer=_CountingConfirmer(None)),
            Confirmation.from_proposal(proposal=_proposal(""), confirmer=_CountingConfirmer("y")),
        ]

    def test_every_outcome_is_reached_once_here(self) -> None:
        self.assertEqual(
            [confirmation.outcome for confirmation in self._each_outcome()], list(ConfirmationOutcome)
        )

    def test_render_is_ascii_and_ends_without_a_newline(self) -> None:
        for confirmation in self._each_outcome():
            with self.subTest(outcome=confirmation.outcome):
                text = confirmation.render()
                text.encode("ascii")
                self.assertFalse(text.endswith("\n"))
                self.assertIn(confirmation.outcome.value.split("_")[0], text.lower())

    def test_to_dict_is_plain_data_carrying_the_transcript(self) -> None:
        confirmation = _ask("e\nthe green cube\n")[0]
        wire = json.loads(json.dumps(confirmation.to_dict()))
        self.assertEqual(
            wire,
            {
                "proposed": _HEARD,
                "confirmed": "the green cube",
                "source": "typed",
                "outcome": "corrected",
                "language": "de",
                "transcript": _transcript().to_dict(),
            },
        )

    def test_it_is_frozen_and_both_halves_of_the_report_contract(self) -> None:
        confirmation = _ask("y\n")[0]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            confirmation.confirmed = "something else"  # type: ignore[misc]
        self.assertIsInstance(confirmation, Rendered)
        self.assertIsInstance(confirmation, Structured)


class TheNullDeviceIsNotAPersonTests(unittest.TestCase):
    """On Windows the NUL device reports isatty() true; a child whose stdin is DEVNULL has nobody to ask."""

    def test_a_child_whose_stdin_is_the_null_device_asks_nothing_and_reads_nothing(self) -> None:
        code = ("from src.models.speech.confirm import TerminalConfirmer;"
                "print(repr(TerminalConfirmer.from_parts().reply('pick the red cube')))")
        proc = subprocess.run([sys.executable, "-c", code], stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(0, proc.returncode, proc.stderr[-1500:])
        self.assertNotIn("Heard", proc.stdout, "a question was asked of the null device")
        self.assertEqual("None", proc.stdout.strip().splitlines()[-1], proc.stdout)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
