"""App-runtime tuning: what the console encodes its frames at, how it answers a greeting, how it reads a command, and
what a task takes where it says nothing itself.

Every block is operator-tunable. ``image_encoding`` is safety-irrelevant: editing it cannot move the robot.
``greeting`` decides whether "Hallo Willy" in the chat waves at once, on a click, or not at all: the wave is a motion,
two swings of the wrist, each judged by the exact guard like every other, and refused where it is not clear.
``commands`` decides whether the cell's everyday sentences are read without the language model: a reading moves
nothing either way, and a task still starts only at ``POST /v1/task``, behind every gate of that route. ``task``
holds what a task takes where it says nothing itself: whether Enter starts it, its mode, its place, the tick for
"anything", how its picks look and whether its parts are critical. They are the cell's, not each browser's (the owner,
2026-10-16), and the console's Settings page writes them into the cell's own layer through the task settings' door
(``src.config.edit.set_task_settings``, the owner, 2026-10-17). None of them starts a task.

Loaded from ``app/runtime.yaml``, which is optional. The schema defaults apply when it is absent.
"""

from __future__ import annotations

import re
from typing import Final, Literal, TypeAlias

from pydantic import Field, field_validator

from ._base import StrictModel
from .robot.robot_schema import pose_name_refusal

#: How the console answers a greeting: wave at once, wave after a dialog's confirm, or greet back only.
GreetingWave: TypeAlias = Literal["direct", "confirm", "off"]
#: How the console starts a typed or spoken task (the owner, 2026-10-08): Enter starts a clean reading at once, or Enter
#: only reads and Start on the card starts.
TaskStart: TypeAlias = Literal["enter", "card"]
#: The mode of a task Enter starts: one part, or until nothing matching is left.
TaskScope: TypeAlias = Literal["once", "until_empty"]
#: How each pick of a task looks (the owner, 2026-10-08 night: "Multi-View" and "Alle Posen" one choice): the first look
#: only, the looks until a grasp is safe, or every look.
TaskLooks: TypeAlias = Literal["first", "when_needed", "every"]
#: What ``runtime.task.place`` names: the cell's default place pose, a taught pose, or a target the camera finds.
TaskPlaceKind: TypeAlias = Literal["default", "pose", "camera"]

#: The longest phrase a camera place holds: what a card's field takes (``api.schemas.MAX_FIELD_CHARS``, which a test
#: pins against this), so a task the console fills from the setting is never refused as too long.
TASK_PHRASE_MAX_CHARS: Final[int] = 200
#: A ``#`` after whitespace: the start of a comment to the line editor that rewrites the setting in place.
_COMMENT_MARK = re.compile(r"\s#")
#: The three forms of a task's place, said in every refusal of one.
_PLACE_FORMS = ("'default' (the cell's default place pose, robot.motion.default_place_pose), 'pose:<name>' (a taught pose) or "
                "'camera:<phrase>' (a target the camera finds, in English: camera:yellow bin)")


def task_place(text: str) -> tuple[TaskPlaceKind, str]:
    """``runtime.task.place`` read: where a task's part goes when its sentence names no place.

    Args:
        text (str): The setting as written: ``default``, ``pose:<name>`` or ``camera:<phrase>``.

    Returns:
        tuple[TaskPlaceKind, str]: ``("default", "")``, ``("pose", <name>)`` or ``("camera", <phrase>)``, the name or
            the phrase without the blanks around it.

    Raises:
        ValueError: Text of none of the three forms, a pose under no pose name (``pose_name_refusal``), or a phrase a
            card's field would not take: empty, over :data:`TASK_PHRASE_MAX_CHARS`, holding a newline or a ``" #"``.
    """
    kind, colon, value = str(text).strip().partition(":")
    if kind == "default" and not colon:
        return "default", ""
    if kind == "pose" and colon:
        refused = pose_name_refusal(value.strip())
        if refused:
            raise ValueError(f"a task's place names a pose under no pose name: {refused}")
        return "pose", value.strip()
    if kind == "camera" and colon:
        refused = _phrase_refusal(value.strip())
        if refused:
            raise ValueError(f"a task's place names a camera place a card would not take: {refused}")
        return "camera", value.strip()
    raise ValueError(f"a task's place is {_PLACE_FORMS}, not {text!r}")


def task_place_text(kind: TaskPlaceKind, value: str = "") -> str:
    """``runtime.task.place`` as it is written: the inverse of :func:`task_place`.

    Args:
        kind (TaskPlaceKind): ``default``, ``pose`` or ``camera``.
        value (str): The pose's name, or the camera's phrase; nothing for the default place (default: "").

    Returns:
        str: ``default``, ``pose:<name>`` or ``camera:<phrase>``, unchecked: the schema checks it where it is read.
    """
    return "default" if kind == "default" else f"{kind}:{value.strip()}"


def _phrase_refusal(phrase: str) -> str:
    """Why ``phrase`` cannot be what the camera looks for in the place a task takes, or ``""`` where it can."""
    if not phrase:
        return "a camera place needs a phrase, what the camera looks for, in English (camera:yellow bin)"
    if len(phrase) > TASK_PHRASE_MAX_CHARS:
        return f"the phrase is longer than the {TASK_PHRASE_MAX_CHARS} characters a card's field takes"
    if not phrase.isprintable():
        return f"the phrase {phrase!r} holds a newline or another character a line cannot show"
    if _COMMENT_MARK.search(phrase):
        return (f"the phrase {phrase!r} holds a '#' after a space, which a later rewrite of its line would take for the "
                "start of a comment")
    return ""


class ImageEncodingConfig(StrictModel):
    """JPEG quality for the frames the console streams (1-100)."""

    #: Quality of each viewfinder frame, read by ``GET /v1/camera``. Lower trades legibility for
    #: bandwidth. At 1280x720 the whole encode costs 1.0 ms and 53 KB at the default 60, so this is
    #: a picture-quality knob and not a performance one.
    frame_quality: int = Field(default=60, ge=1, le=100)


class GreetingConfig(StrictModel):
    """What the console does when someone greets Willy in the chat ("Hallo Willy", "Hi Willy", "Tschüss Willy")."""

    #: ``direct``: Willy waves at once, with no click, two swings of the wrist the exact guard judges like every
    #: motion (the owner, 2026-10-06: "Sofort winken, aber man kann es in der App-Config einstellen, dass man es
    #: vorher bestätigen muss"). ``confirm``: the console asks first, in a dialog like Home's, and the wave starts on
    #: its confirm.
    #: ``off``: Willy only greets back, and nothing moves.
    wave: GreetingWave = "direct"

    @field_validator("wave", mode="before")
    @classmethod
    def _a_bare_switch(cls, value: object) -> object:
        """YAML reads a bare ``off`` as false and ``on`` as true: each means what it says here."""
        if value is False:
            return "off"
        if value is True:
            return "direct"
        return value


class CommandsConfig(StrictModel):
    """How the console reads an operator's sentence (``POST /v1/commands/parse``)."""

    #: ``true``: a known sentence ("Alle grauen Würfel in die Gelbe Kiste.", "Hallo Willy") is read without the
    #: language model, as the model was measured to read it (``src/models/vlm/known.py``), in well under a
    #: millisecond instead of about ten seconds on the cell; every other sentence goes to the model. ``false``: every
    #: sentence goes to the model. The owner, 2026-10-08: "speed first", on.
    known_sentences: bool = True


class TaskConfig(StrictModel):
    """What a task takes where it says nothing itself: the cell's, rather than each browser's (the owner, 2026-10-16).

    The console reads them for every task a person gives it, and its Settings page ("Auftrag") writes them into the
    cell's own layer (the owner, 2026-10-17). What to pick and where to put it come from the sentence; these stand where
    it says nothing, and a card may differ for its own task. None of them starts a task: a task starts only at
    ``POST /v1/task``, behind every gate of that route.
    """

    #: How the console starts a typed or spoken task (the owner, 2026-10-08: "Enter startet sofort"). ``enter``: Enter
    #: reads the sentence and starts a clean reading at once where the cell is ready, and the card opens for every
    #: other. ``card``: Enter only reads, and Start on the card starts.
    start: TaskStart = "enter"
    #: The mode of a task Enter starts: ``once``, one part, or ``until_empty``, until nothing matching is left. "Alle" in
    #: the sentence does not change it, and a sentence that singles out one part runs once either way.
    scope: TaskScope = "once"
    #: Where the part goes when the sentence names no place: ``default``, the cell's default place pose
    #: (``robot.motion.default_place_pose``); ``pose:<name>``, a taught pose; ``camera:<phrase>``, a target the camera finds,
    #: in English (``camera:yellow bin``), at most 200 characters on one line. One value, so a layer that sets it
    #: replaces the whole place.
    place: str = "default"
    #: ``true``: a card whose sentence names no part comes ticked for "alles, was die Kamera sieht", bin walls included
    #: (Q7 A+, the tick given beforehand). ``false``: the person ticks it on the card. Such a card never starts on Enter.
    anything: bool = False
    #: How each pick of a task looks: ``first``, the first look only (multi-view off); ``when_needed``, the looks until
    #: a grasp is safe, as always; ``every``, every configured look, the early stop off. A card may change it for its
    #: task.
    looks: TaskLooks = "when_needed"
    #: Whether the parts are critical (the owner, 2026-10-03). ``false``: a boxed-in part is pushed first, and the push
    #: may rearrange the scene (it shoves the neighbours in its way, the fingers brush the ones beside their stroke;
    #: they come down only on free table seen by the camera, the housing keeps the guard's distance, and nothing is
    #: pushed over an edge); where no push plans, a blocker is cleared. ``true``: nothing is pushed, a blocker is
    #: cleared (gripped and set aside) instead. A task may say otherwise for itself, as the console's advanced drawer
    #: does. Read when the cell is built, so a change applies from its next build. It was
    #: ``robot.grasping.recovery.critical_parts`` until 2026-10-16.
    critical_parts: bool = False

    @field_validator("place", mode="before")
    @classmethod
    def _one_of_three_places(cls, value: object) -> object:
        """A place is ``default``, ``pose:<name>`` or ``camera:<phrase>``, kept without the blanks around its name."""
        if not isinstance(value, str):
            return value  # the schema's own type check says what it is not
        kind, name = task_place(value)
        return task_place_text(kind, name)


class RuntimeConfig(StrictModel):
    """Every app-runtime block, read from the ``runtime`` section of ``app/runtime.yaml``."""

    image_encoding: ImageEncodingConfig = Field(default_factory=ImageEncodingConfig)
    greeting: GreetingConfig = Field(default_factory=GreetingConfig)
    commands: CommandsConfig = Field(default_factory=CommandsConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
