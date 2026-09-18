"""What a pick looks for: the phrase each camera grounds, and the label a target must carry.

A cell is built with one grounding phrase, the caption every camera's detector reads on every frame. A
prompt an operator types or speaks changes three things together:

* the phrase the detector grounds;
* the labels the camera source maps a detector's words onto, which is the phrase itself. A phrase
  grounder labels a box with the words it matched ("red cube" for "the red cube"), and the label filter
  compares exactly, so a filter set to the typed text alone matches nothing a detector returns;
* the label filter, so only an object grounded by this phrase is a pick target.

:meth:`~src.robot.execution.autonomous_grasp.service.AutonomousGraspService.set_prompt` sets all
three and returns the prompt it replaced; passing that back puts all three back. Nothing reopens and
nothing reloads, because the detector is handed the phrase on every frame.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PickPrompt"]


@dataclass(frozen=True, slots=True)
class PickPrompt:
    """The phrase a cell grounds, the labels it maps detections onto, and the label a target must carry."""

    #: What every camera that grounds a phrase grounds. Empty for a cell whose perception grounds no
    #: phrase, such as the rehearsal scene.
    phrase: str
    #: The label a segmentation must carry to be a pick target. ``None`` makes every grounded object one.
    target_label: str | None = None
    #: The labels a camera source maps a detector's words onto. Empty passes the words through.
    object_labels: tuple[str, ...] = ()

    @classmethod
    def from_text(cls, text: str) -> "PickPrompt":
        """The prompt an operator typed or spoke: the phrase, the phrase as the one label, and the filter on it.

        Refuses a text that names nothing, before anything is set: the detector refuses an empty phrase
        on every frame, and a filter on an empty label matches nothing a detector returns.
        """
        phrase = str(text).strip()
        if not phrase:
            raise ValueError(
                "a pick prompt names what to pick; an empty one grounds nothing and the detector refuses it"
            )
        return cls(phrase=phrase, target_label=phrase, object_labels=(phrase,))
