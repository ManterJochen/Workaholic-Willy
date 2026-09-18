"""A person confirms what speech proposed before it becomes a prompt: `Confirmation` and `Confirmer`.

Voice never acts on its own. A transcript is a proposal, and its words reach `PickRun(prompt=)` or the
`Locator` only after a person said yes to them, or typed the words they meant instead. "Stopp" is no
exception: stopping stays with the emergency stop.

`Confirmation` is that step as a frozen report. `from_proposal` takes what an upload or a push to talk
turn proposed (`Proposal`), and `from_utterance` what a `Listener` heard (`Utterance`). Both ask a
`Confirmer` only when there are words to ask about. `confirmed` holds text only when a person confirmed
or corrected them; every other outcome leaves it None, and None never becomes a prompt.

A `Confirmer` answers with the text a person approves: the proposed text itself on an explicit yes, other
words when the person typed a correction, an empty string for anything else, and None when no person can
be asked. `TerminalConfirmer` asks at the terminal the process runs in. With no terminal to ask (stdin is
not a TTY: a service, a pipe, a test runner) nobody is asked, and the outcome is NO_PERSON.

In the console the confirmation is the start button: the proposal lands in the prompt box, and the
operator presses the same button, with the same acknowledgement, as for a typed prompt.

stdlib only at import, so a caller can name the types without importing the speech stack.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Protocol, TextIO, runtime_checkable

from src.contracts import UNSET, Maybe, resolve
from src.models.speech.transcript import Transcript, _ascii

if TYPE_CHECKING:  # pragma: no cover, typing only
    from src.models.speech.listener import Utterance
    from src.models.speech.transcript import Proposal

__all__ = ["Confirmation", "ConfirmationOutcome", "Confirmer", "PromptSource", "TerminalConfirmer"]

#: The answers that confirm the words as heard, compared stripped and lower-cased.
_YES: Final[frozenset[str]] = frozenset({"y", "yes", "j", "ja"})
#: The answers that ask to type the prompt instead, compared the same way.
_TYPE_IT: Final[frozenset[str]] = frozenset({"e", "edit"})


class PromptSource(StrEnum):
    """Where the words of a confirmation came from."""

    #: What speech proposed, as it was heard.
    SPOKEN = "spoken"
    #: What a person typed in place of what was heard.
    TYPED = "typed"


class ConfirmationOutcome(StrEnum):
    """What the person answered, or why nobody was asked."""

    #: An explicit yes: the words as heard become the prompt.
    CONFIRMED = "confirmed"
    #: The person typed the prompt they meant, and that text becomes the prompt.
    CORRECTED = "corrected"
    #: Any other answer: nothing becomes a prompt.
    REFUSED = "refused"
    #: No person could be asked, so nothing becomes a prompt.
    NO_PERSON = "no_person"
    #: Speech proposed no words, so nobody was asked.
    NOTHING_HEARD = "nothing_heard"


@runtime_checkable
class Confirmer(Protocol):
    """A person asked whether proposed words may become a prompt."""

    def reply(self, proposed: str) -> str | None:
        """The text the person approves: ``proposed`` itself on an explicit yes, the words they typed on a
        correction, ``""`` for any other answer, and None when no person can be asked."""
        ...


@dataclass(frozen=True, slots=True)
class Confirmation:
    """Whether a person let proposed words become a prompt, and which words."""

    #: The words speech proposed; ``""`` when it proposed none.
    proposed: str
    #: The text that may become a prompt; None unless the outcome is CONFIRMED or CORRECTED.
    confirmed: str | None
    #: SPOKEN for the words as heard, TYPED for a correction.
    source: PromptSource
    outcome: ConfirmationOutcome
    #: The language Whisper heard (``de``, ``en``); None when no transcript carries one. A typed
    #: correction keeps it, since nothing detects the language of typing.
    language: str | None
    #: Whisper's report on what was heard; None when Whisper was not asked.
    transcript: Transcript | None

    @classmethod
    def from_proposal(cls, *, proposal: Proposal, confirmer: Confirmer) -> Confirmation:
        """Ask ``confirmer`` about what an upload or a push to talk turn proposed.

        A proposal without words is NOTHING_HEARD, and nobody is asked.
        """
        return cls._asked(proposed=proposal.text, transcript=proposal.transcript, confirmer=confirmer)

    @classmethod
    def from_utterance(cls, *, utterance: Utterance, confirmer: Confirmer) -> Confirmation:
        """Ask ``confirmer`` about what a `Listener` heard.

        An utterance that was not HEARD, or whose transcript holds no words, is NOTHING_HEARD, and nobody
        is asked.
        """
        from src.models.speech.listener import ListenOutcome

        transcript = utterance.transcript if utterance.outcome is ListenOutcome.HEARD else None
        text = "" if transcript is None else transcript.text
        return cls._asked(proposed=text, transcript=transcript, confirmer=confirmer)

    @classmethod
    def _asked(
        cls, *, proposed: str, transcript: Transcript | None, confirmer: Confirmer
    ) -> Confirmation:
        language = None if transcript is None else transcript.language
        confirmed: str | None = None
        source = PromptSource.SPOKEN
        if not proposed.strip():
            outcome = ConfirmationOutcome.NOTHING_HEARD
        else:
            reply = confirmer.reply(proposed)
            answer = "" if reply is None else reply.strip()
            if reply is None:
                outcome = ConfirmationOutcome.NO_PERSON
            elif not answer:
                outcome = ConfirmationOutcome.REFUSED
            elif answer == proposed.strip():
                outcome, confirmed = ConfirmationOutcome.CONFIRMED, proposed
            else:
                outcome, confirmed, source = ConfirmationOutcome.CORRECTED, answer, PromptSource.TYPED
        return cls(
            proposed=proposed,
            confirmed=confirmed,
            source=source,
            outcome=outcome,
            language=language,
            transcript=transcript,
        )

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The answer in a line, the transcript indented under it. ASCII, no trailing newline."""
        heard = f"'{_ascii(self.proposed)}'"
        language = "" if self.language is None else f" ({_ascii(self.language)})"
        if self.outcome is ConfirmationOutcome.CONFIRMED:
            head = f"Confirmed {heard}{language} as spoken"
        elif self.outcome is ConfirmationOutcome.CORRECTED:
            head = (
                f"Corrected: the prompt is '{_ascii(self.confirmed or '')}', typed in place of "
                f"{heard}{language}"
            )
        elif self.outcome is ConfirmationOutcome.REFUSED:
            head = f"Refused {heard}{language}: it does not become a prompt"
        elif self.outcome is ConfirmationOutcome.NO_PERSON:
            head = f"Nobody could be asked about {heard}{language}: it does not become a prompt"
        else:
            head = "Nothing was heard, so nobody was asked"
        lines = [head]
        if self.transcript is not None:
            lines.extend(f"  {line}" for line in self.transcript.render().split("\n"))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The machine half: plain data, the transcript as its own `to_dict()`."""
        return {
            "proposed": self.proposed,
            "confirmed": self.confirmed,
            "source": self.source.value,
            "outcome": self.outcome.value,
            "language": self.language,
            "transcript": None if self.transcript is None else self.transcript.to_dict(),
        }


def _is_terminal(stream: TextIO) -> bool:
    """Whether a person can answer on ``stream``.

    On Windows the NUL device reports ``isatty()`` true, and a child started with its stdin on DEVNULL
    is handed it, as is a stream pytest captures. A question asked there reads the end of input, which
    would count as a refusal where nobody was asked. So on Windows a descriptor is also asked for its
    console mode, which only a console has. A stream with no descriptor answers ``isatty()`` for itself.
    """
    try:
        if not stream.isatty():
            return False
    except (ValueError, OSError):  # a closed or detached stream
        return False
    if sys.platform == "win32":
        try:
            descriptor = stream.fileno()
        except (ValueError, OSError, AttributeError):
            return True
        import ctypes  # noqa: PLC0415
        import msvcrt  # noqa: PLC0415

        mode = ctypes.c_ulong()
        return bool(ctypes.windll.kernel32.GetConsoleMode(msvcrt.get_osfhandle(descriptor), ctypes.byref(mode)))
    return True


class TerminalConfirmer:
    """Asks at the terminal the process runs in. Build it with `from_parts`; the verb is `reply()`."""

    def __init__(self, *, stdin: TextIO | None, stdout: TextIO | None) -> None:
        self._stdin = stdin
        self._stdout = stdout

    @classmethod
    def from_parts(
        cls, *, stdin: Maybe[TextIO | None] = UNSET, stdout: Maybe[TextIO | None] = UNSET
    ) -> TerminalConfirmer:
        """The Python door. ``stdin`` and ``stdout`` UNSET are the process's own, taken when it is built."""
        return cls(stdin=resolve("stdin", stdin, sys.stdin), stdout=resolve("stdout", stdout, sys.stdout))

    def reply(self, proposed: str) -> str | None:
        """Ask once about ``proposed``.

        y, yes, j or ja return ``proposed``. e or edit ask for the prompt and return what is typed next.
        Anything else returns ``""``, the end of input and an empty correction included. When stdin is
        not a terminal the answer is None, and nothing is read or written.
        """
        stdin = self._stdin
        if stdin is None or not _is_terminal(stdin):
            return None
        self._say(
            f"Heard '{proposed}'. Type y to run it as heard, or e to type the prompt yourself; "
            f"anything else refuses: "
        )
        word = stdin.readline().strip().lower()
        if word in _YES:
            return proposed
        if word in _TYPE_IT:
            self._say("The prompt: ")
            return stdin.readline().strip()
        return ""

    def _say(self, text: str) -> None:
        stdout = self._stdout
        if stdout is None:
            return
        try:
            stdout.write(text)
        except UnicodeEncodeError:  # a console that cannot show the heard words gets them escaped
            stdout.write(_ascii(text))
        stdout.flush()
