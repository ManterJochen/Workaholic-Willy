"""The perception stack as a value: the four fields that decide it, resolved before a weight loads.

``build_perception`` is the one builder, and :meth:`PerceptionSpec.build` hands itself to it. The spec's four fields
are exactly the four attributes that function reads, which is why it satisfies the same structural contract a
``ModelsConfig`` does and why the factory annotates its input as :class:`PerceptionFields` rather than
``ModelsConfig``.

Four and not the five a ``ModelsConfig`` carries: the fifth is ``stt``, the speech section, required on
``ModelsConfig`` and never read by perception, so a Python caller builds a stack without one.

Importing this module must not pull torch, because ``autonomous_grasp`` imports it. Every heavy import stays inside
``build()``, as ``factory.py`` already does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.app import ModelsConfig, PerceptionModelsConfig
    from src.config.schema.models.models_schema import (
        ObjectDetectorConfig,
        PipelineConfig,
        SegmenterConfig,
        VlmConfig,
    )
    from src.models.perception_backend import TwoStageBackend

__all__ = ["PerceptionFields", "PerceptionResolution", "PerceptionSpec"]


class PerceptionFields(Protocol):
    """The four attributes ``build_perception`` reads. Structural, so nothing has to inherit it.

    ``ModelsConfig`` satisfies it and so does :class:`PerceptionSpec`. The factory annotates its input as this
    Protocol, which is what lets the narrow value reach the one builder.
    """

    # Property members, not plain attributes: a member written as `x: int` is a settable variable, which the
    # read-only attributes of a frozen dataclass do not satisfy. A read-only property member is satisfied by a frozen
    # field and by a mutable one, and this builder only ever reads.
    @property
    def weights_root(self) -> str: ...
    @property
    def segmenter(self) -> "SegmenterConfig | None": ...
    @property
    def rtdetr(self) -> "ObjectDetectorConfig | None": ...
    @property
    def pipeline(self) -> "PipelineConfig": ...


@dataclass(frozen=True, slots=True)
class PerceptionResolution:
    """What :meth:`PerceptionSpec.build` will construct, decided without loading a weight.

    One resolution, rendered by whoever needs it, ``/v1/diagnostics`` included, so the console cannot describe
    another stack than the one the builder builds.
    """

    #: ``zero_shot`` | ``closed_set``
    kind: str
    #: ``vlm`` | ``rtdetr``
    detector: str
    #: ``sam2``, the one segmenter.
    segmenter: str = "sam2"
    #: The VLM's checkpoint where it is the detector; ``None`` on a closed-set stack.
    vlm_model_id: str | None = None
    #: The message :meth:`PerceptionSpec.build` would raise, verbatim, or ``""``. Read from the same constants in
    #: ``factory.py`` that the builder raises, never paraphrased here in other words.
    refusal: str = ""

    @property
    def buildable(self) -> bool:
        """Whether :meth:`PerceptionSpec.build` would build the stack rather than raise."""
        return not self.refusal

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The stack as a few ASCII lines for a person, with no newline at the end."""
        lines = [f"perception stack: {self.kind} / {self.detector} + {self.segmenter}"]
        if self.vlm_model_id:
            lines.append(f"  vlm             : {self.vlm_model_id}")
        if self.refusal:
            lines.append(f"  REFUSED         : {self.refusal}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The resolution as plain data, ``buildable`` included."""
        return {
            "kind": self.kind,
            "detector": self.detector,
            "segmenter": self.segmenter,
            "vlm_model_id": self.vlm_model_id,
            "refusal": self.refusal,
            "buildable": self.buildable,
        }


def _default_pipeline() -> "PipelineConfig":
    from src.config.schema.models.models_schema import PipelineConfig  # noqa: PLC0415 (pydantic, kept off import)

    return PipelineConfig()


def _default_weights_root() -> str:
    from src.utility.paths import weights_root  # noqa: PLC0415

    return weights_root().as_posix()


@dataclass(frozen=True, slots=True)
class PerceptionSpec:
    """A perception stack, as the four fields that decide it. One verb: :meth:`build`."""

    segmenter: "SegmenterConfig | None"
    rtdetr: "ObjectDetectorConfig | None" = None
    pipeline: "PipelineConfig" = field(default_factory=_default_pipeline)
    #: The folder every model's weights are read from (``models.weights_root``).
    weights_root: str = field(default_factory=_default_weights_root)

    # ---------------------------------------------------------------- factories
    @classmethod
    def from_config(cls, models: "ModelsConfig | PerceptionModelsConfig") -> "PerceptionSpec":
        """The YAML door: take a validated ``ModelsConfig`` and keep the four fields it reads.

        ``load_perception_section`` hands over a ``PerceptionModelsConfig`` instead, the same four fields without the
        Whisper block, so a perception stack is built from config without speech.
        """
        return cls(segmenter=models.segmenter, rtdetr=models.rtdetr, pipeline=models.pipeline,
                   weights_root=models.weights_root)

    @classmethod
    def zero_shot(cls, *, segmenter: "SegmenterConfig", vlm: "VlmConfig | None" = None,
                  weights_root: str | None = None) -> "PerceptionSpec":
        """Open vocabulary, from Python: the VLM grounds the prompt and SAM2 cuts each box, no YAML tree.

        ``vlm`` is the VLM's block; ``None`` is the schema's default checkpoint. ``weights_root`` ``None`` is the
        repository's weights folder.
        """
        from src.config.schema.models.models_schema import (  # noqa: PLC0415
            PipelineConfig,
            VlmConfig,
            ZeroShotPipelineConfig,
        )

        pipeline = PipelineConfig(kind="zero_shot", zero_shot=ZeroShotPipelineConfig(vlm=vlm or VlmConfig()))
        return cls(segmenter=segmenter, rtdetr=None, pipeline=pipeline,
                   weights_root=weights_root or _default_weights_root())

    @classmethod
    def closed_set(cls, *, rtdetr: "ObjectDetectorConfig", segmenter: "SegmenterConfig",
                   weights_root: str | None = None) -> "PerceptionSpec":
        """Fixed vocabulary, from Python: RT-DETR's trained classes and SAM2 for the masks."""
        from src.config.schema.models.models_schema import PipelineConfig  # noqa: PLC0415

        return cls(segmenter=segmenter, rtdetr=rtdetr, pipeline=PipelineConfig(kind="closed_set"),
                   weights_root=weights_root or _default_weights_root())

    # ---------------------------------------------------------------- the verb
    def build(self, *, debug_images: bool = False) -> "TwoStageBackend":
        """Construct the stack. ``build_perception`` does the work, with this spec as its input."""
        from src.models.factory import build_perception

        return build_perception(self, debug_images=debug_images)

    # ---------------------------------------------------------------- the view
    def resolve(self) -> PerceptionResolution:
        """What :meth:`build` would construct, and why, without touching a weight.

        Every refusal string comes from ``factory.py``. A refusal paraphrased here would be a second answer to the
        question the builder answers.
        """
        from src.models import factory

        if self.pipeline.kind == "closed_set":
            refusal = ""
            if self.rtdetr is None:
                refusal = factory.REFUSE_CLOSED_SET_WITHOUT_RTDETR
            elif self.segmenter is None:
                refusal = factory.REFUSE_SAM2_WITHOUT_BLOCK
            return PerceptionResolution(kind="closed_set", detector="rtdetr", refusal=refusal)
        return PerceptionResolution(
            kind="zero_shot", detector="vlm", vlm_model_id=self.pipeline.zero_shot.vlm.model_id,
            refusal="" if self.segmenter is not None else factory.REFUSE_SAM2_WITHOUT_BLOCK,
        )
