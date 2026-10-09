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

A sort names several kinds at once (the owner, 2026-10-09: "Grüne Teile in die gelbe Kiste, rote in die
blaue"): the phrase is a class list, every rule's kind of part grounded in one call
(``src.models.vlm.qwen.class_list_prompt``), the object labels are the kinds, and the filter takes any of
them (``target_labels``, with ``target_label`` ``None``):

    kinds = ("green part", "red part")
    PickPrompt(phrase=class_list_prompt(["each separate green part", "each separate red part"]),
               object_labels=kinds, target_labels=kinds)

The camera source maps each box's words onto the kind it names, so a box no rule clearly claims (the
detector's ``ambiguous``, a word no rule names) is no target and stays a neighbour.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PickPrompt"]


@dataclass(frozen=True, slots=True)
class PickPrompt:
    """What a cell grounds, the labels it maps detections onto, and the label a target must carry.

    Attributes:
        phrase (str): What every camera that grounds a phrase grounds; empty for a cell that grounds none.
        target_label (str | None): The label a segmentation must carry to be a target; ``None`` makes every grounded
            object one (default: None).
        object_labels (tuple[str, ...]): The labels a camera source maps a detector's words onto; empty passes them
            through (default: ()).
        target_labels (tuple[str, ...]): Any of these labels makes a target (a sort); ``target_label`` is ``None``
            beside them (default: ()).
    """

    #: What every camera that grounds a phrase grounds. Empty for a cell whose perception grounds no
    #: phrase, such as the rehearsal scene.
    phrase: str
    #: The label a segmentation must carry to be a pick target. ``None`` makes every grounded object one.
    target_label: str | None = None
    #: The labels a camera source maps a detector's words onto. Empty passes the words through.
    object_labels: tuple[str, ...] = ()
    #: The labels a segmentation may carry to be a pick target, any of them (a sort, the owner 2026-10-09), each
    #: compared exactly; ``target_label`` is ``None`` beside them. Empty: ``target_label`` alone decides, as before.
    target_labels: tuple[str, ...] = ()

    @classmethod
    def from_text(cls, text: str) -> "PickPrompt":
        """The prompt an operator typed or spoke: the phrase, the phrase as the one label, and the filter on it.

        Args:
            text (str): What was typed or spoken, such as ``"a red cube"``.

        Returns:
            PickPrompt: The prompt, for ``service.set_prompt``.

        Raises:
            ValueError: A text that names nothing.
        """
        phrase = str(text).strip()
        if not phrase:
            raise ValueError(
                "a pick prompt names what to pick; an empty one grounds nothing and the detector refuses it"
            )
        return cls(phrase=phrase, target_label=phrase, object_labels=(phrase,))
