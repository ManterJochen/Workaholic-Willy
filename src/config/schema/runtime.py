"""App-runtime tuning: what the console encodes its frames at, and how it answers a greeting.

Both blocks are operator-tunable. ``image_encoding`` is safety-irrelevant: editing it cannot move the robot.
``greeting`` decides whether "Hallo Willy" in the chat waves at once, on a click, or not at all: the wave is a motion,
two swings of the wrist, each judged by the exact guard like every other, and refused where it is not clear.

Loaded from ``app/runtime.yaml``, which is optional. The schema defaults apply when it is absent.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, field_validator

from ._base import StrictModel

#: How the console answers a greeting: wave at once, wave after a dialog's confirm, or greet back only.
GreetingWave: TypeAlias = Literal["direct", "confirm", "off"]


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


class RuntimeConfig(StrictModel):
    """Every app-runtime block, read from the ``runtime`` section of ``app/runtime.yaml``."""

    image_encoding: ImageEncodingConfig = Field(default_factory=ImageEncodingConfig)
    greeting: GreetingConfig = Field(default_factory=GreetingConfig)
