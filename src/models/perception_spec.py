"""The perception stack as a value: the seven fields that decide it, resolved before a weight loads.

``build_perception`` is the only function that reads ``models.pipeline``. ``build_object_detector``
and ``build_segmenter``, which ``build_real_components`` calls, read only ``models.detector`` and
``models.segmenter_backend`` (``factory.py:27-61``), so a stack assembled from that pair ignores the
pipeline block with no error and no log line, and ``build_perception``'s fail-closed refusals never
fire for it.

:meth:`PerceptionSpec.build` hands itself to ``build_perception``, so there is one builder. The
spec's seven fields are exactly the seven attributes that function reads, which is why it satisfies
the same structural contract a ``ModelsConfig`` does and why the factory annotates its input as
:class:`PerceptionFields` rather than ``ModelsConfig``.

Seven and not the ten a ``ModelsConfig`` carries: three of those ten are required and one of the
three is ``stt``, eleven Whisper fields with ten mandatory that perception never reads. ``stt`` is
not defaulted, and defaulting it would change validation for every YAML tree in the repository, so
the spec carries seven fields and a Python caller never has to supply ``stt`` at all.

Importing this module must not pull torch, because ``autonomous_grasp`` imports it. Every heavy
import stays inside ``build()``, as ``factory.py`` already does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from src.models.constants import MODELS_FACTORY_LOG_FILE, MODELS_LOG_DIR
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.app import ModelsConfig, PerceptionModelsConfig
    from src.config.schema.models.models_schema import (
        ObjectDetectorConfig,
        OneFormerConfig,
        PipelineConfig,
        SegmenterConfig,
    )
    from src.models.perception_backend import PerceptionBackend

__all__ = ["PerceptionFields", "PerceptionResolution", "PerceptionSpec"]

_LOG = create_logger(__name__, log_file=MODELS_FACTORY_LOG_FILE, log_dir=MODELS_LOG_DIR)


class PerceptionFields(Protocol):
    """The seven attributes ``build_perception`` reads. Structural, so nothing has to inherit it.

    ``ModelsConfig`` satisfies it and so does :class:`PerceptionSpec`. The factory annotates its
    input as this Protocol, which is what lets the narrow value reach the one builder.
    """

    # Property members, not plain attributes: a member written as `x: int` is a settable variable,
    # which the read-only attributes of a frozen dataclass do not satisfy. A read-only property
    # member is satisfied by a frozen field and by a mutable one, and this builder only ever reads.
    @property
    def objectdetector(self) -> "ObjectDetectorConfig | None": ...
    @property
    def segmenter(self) -> "SegmenterConfig | None": ...
    @property
    def detector(self) -> str: ...
    @property
    def segmenter_backend(self) -> str: ...
    @property
    def rtdetr(self) -> "ObjectDetectorConfig | None": ...
    @property
    def oneformer(self) -> "OneFormerConfig | None": ...
    @property
    def pipeline(self) -> "PipelineConfig | None": ...


@dataclass(frozen=True, slots=True)
class PerceptionResolution:
    """What :meth:`PerceptionSpec.build` will construct, decided without loading a weight.

    One resolution, rendered by whoever needs it, ``/v1/diagnostics`` included. Re-deriving the
    same answer from ``models.pipeline`` in prose beside the builder is a second reading of one
    config, and the two readings can disagree about whether the pipeline block reaches hardware and
    about which segmenter the VLM route uses.
    """

    #: Which half of the config decided this: ``"models.pipeline"`` or ``"legacy keys"``.
    source: str
    #: ``zero_shot`` | ``closed_set``
    kind: str
    #: ``groundingdino`` | ``rtdetr`` | ``vlm`` | ``routed``
    detector: str
    #: ``sam2`` | ``oneformer``
    segmenter: str
    router_enabled: bool = False
    vlm_model_id: str | None = None
    vlm_on_unavailable: str | None = None
    #: The message :meth:`PerceptionSpec.build` would raise, verbatim, or ``""``. Read from the same
    #: constants in ``factory.py`` that the builder raises, never paraphrased here in other words.
    refusal: str = ""

    @property
    def buildable(self) -> bool:
        return not self.refusal

    def render(self) -> str:
        lines = [
            f"perception stack: {self.kind} / {self.detector} + {self.segmenter}",
            f"  decided by      : {self.source}",
        ]
        if self.vlm_model_id:
            lines.append(f"  vlm             : {self.vlm_model_id} "
                         f"(on_unavailable={self.vlm_on_unavailable})")
        lines.append(f"  prompt router   : {'on' if self.router_enabled else 'off'}")
        if self.refusal:
            lines.append(f"  REFUSED         : {self.refusal}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "detector": self.detector,
            "segmenter": self.segmenter,
            "router_enabled": self.router_enabled,
            "vlm_model_id": self.vlm_model_id,
            "vlm_on_unavailable": self.vlm_on_unavailable,
            "refusal": self.refusal,
            "buildable": self.buildable,
        }


@dataclass(frozen=True, slots=True)
class PerceptionSpec:
    """A perception stack, as the seven fields that decide it. One verb: :meth:`build`."""

    objectdetector: "ObjectDetectorConfig | None"
    segmenter: "SegmenterConfig | None"
    detector: str
    segmenter_backend: str
    rtdetr: "ObjectDetectorConfig | None"
    oneformer: "OneFormerConfig | None"
    pipeline: "PipelineConfig | None"

    # ---------------------------------------------------------------- factories
    @classmethod
    def from_config(cls, models: "ModelsConfig | PerceptionModelsConfig") -> "PerceptionSpec":
        """The YAML door: take a validated ``ModelsConfig`` and keep the seven fields it reads.

        ``load_perception_section`` hands over a ``PerceptionModelsConfig`` instead, the same seven
        fields without the Whisper block, so a perception stack is built from config without speech.

        A projection, not a normalisation. Folding the legacy keys into an equivalent ``pipeline``
        block would change behaviour: a missing ``models.rtdetr`` is refused with
        ``build_object_detector``'s message on the legacy branch and with ``build_perception``'s on
        the pipeline branch, and an operator reads those two sentences. The spec carries what the
        config says; ``build()`` decides what it means.
        """
        return cls(
            objectdetector=models.objectdetector,
            segmenter=models.segmenter,
            detector=models.detector,
            segmenter_backend=models.segmenter_backend,
            rtdetr=models.rtdetr,
            oneformer=models.oneformer,
            pipeline=models.pipeline,
        )

    @classmethod
    def zero_shot(
        cls,
        *,
        objectdetector: "ObjectDetectorConfig",
        segmenter: "SegmenterConfig",
        pipeline: "PipelineConfig | None" = None,
        oneformer: "OneFormerConfig | None" = None,
    ) -> "PerceptionSpec":
        """Open-vocabulary, from Python: a phrase grounder and a mask source, no YAML tree.

        ``pipeline`` is optional and is how the VLM route and the prompt router are reached. Its
        sub-blocks are all defaulted, so ``PipelineConfig(zero_shot=ZeroShotPipelineConfig(
        backend="vlm"))`` is enough to reach the VLM. Omitted, this builds the legacy stack.
        """
        return cls(
            objectdetector=objectdetector,
            segmenter=segmenter,
            detector="groundingdino",
            segmenter_backend="oneformer" if oneformer is not None and pipeline is None else "sam2",
            rtdetr=None,
            oneformer=oneformer,
            pipeline=pipeline,
        )

    @classmethod
    def closed_set(
        cls,
        *,
        rtdetr: "ObjectDetectorConfig",
        segmenter: "SegmenterConfig",
        pipeline: "PipelineConfig | None" = None,
        oneformer: "OneFormerConfig | None" = None,
    ) -> "PerceptionSpec":
        """Fixed-vocabulary, from Python: RT-DETR's trained classes and a mask source.

        ``objectdetector`` stays ``None``. A closed-set stack has no phrase grounder, and the empty
        field is what keeps ``build_perception``'s router refusal reachable instead of grounding
        with a model this stack never asked for.
        """
        return cls(
            objectdetector=None,
            segmenter=segmenter,
            detector="rtdetr",
            segmenter_backend="oneformer" if oneformer is not None and pipeline is None else "sam2",
            rtdetr=rtdetr,
            oneformer=oneformer,
            pipeline=pipeline,
        )

    # ---------------------------------------------------------------- the verb
    def build(self, *, debug_images: bool = False) -> "PerceptionBackend":
        """Construct the stack. ``build_perception`` does the work, with this spec as its input."""
        from src.models.factory import build_perception

        return build_perception(self, debug_images=debug_images)  # type: ignore[no-any-return]

    # ---------------------------------------------------------------- the view
    def resolve(self) -> PerceptionResolution:
        """What :meth:`build` would construct, and why, without touching a weight.

        Every refusal string comes from ``factory.py``. A refusal paraphrased here would be a
        second answer to the question the builder answers.
        """
        from src.models import factory

        pipeline = self.pipeline
        if pipeline is None:
            refusal = ""
            if self.detector == "rtdetr" and self.rtdetr is None:
                refusal = factory.REFUSE_RTDETR_WITHOUT_BLOCK
            elif self.detector != "rtdetr" and self.objectdetector is None:
                refusal = factory.REFUSE_GROUNDINGDINO_WITHOUT_BLOCK
            elif self.segmenter_backend == "oneformer" and self.oneformer is None:
                refusal = factory.REFUSE_ONEFORMER_WITHOUT_BLOCK
            elif self.segmenter_backend != "oneformer" and self.segmenter is None:
                refusal = factory.REFUSE_SAM2_WITHOUT_BLOCK
            return PerceptionResolution(
                source="legacy keys",
                kind="closed_set" if self.detector == "rtdetr" else "zero_shot",
                detector=self.detector,
                segmenter=self.segmenter_backend,
                refusal=refusal,
            )

        if pipeline.kind == "closed_set":
            refusal = ""
            if self.rtdetr is None:
                refusal = factory.REFUSE_CLOSED_SET_WITHOUT_RTDETR
            elif pipeline.closed_set.segmenter == "oneformer" and self.oneformer is None:
                refusal = factory.REFUSE_NAMED_ONEFORMER_WITHOUT_BLOCK
            elif pipeline.closed_set.segmenter != "oneformer" and self.segmenter is None:
                refusal = factory.REFUSE_SAM2_WITHOUT_BLOCK
            return PerceptionResolution(
                source="models.pipeline", kind="closed_set", detector="rtdetr",
                segmenter=pipeline.closed_set.segmenter, refusal=refusal,
            )

        zs = pipeline.zero_shot
        # The segmenter is read on every branch, the VLM one included, and never pinned here:
        # `_build_vlm_backend` reads `pipeline.zero_shot.segmenter` (factory.py:219) and the schema
        # does not pin `sam2` there.
        segmenter = zs.segmenter
        refusal = ""
        if segmenter == "oneformer" and self.oneformer is None:
            refusal = factory.REFUSE_NAMED_ONEFORMER_WITHOUT_BLOCK
        elif segmenter != "oneformer" and self.segmenter is None:
            refusal = factory.REFUSE_SAM2_WITHOUT_BLOCK
        if zs.backend != "vlm":
            if not refusal and self.objectdetector is None:
                refusal = factory.REFUSE_GROUNDINGDINO_WITHOUT_BLOCK
            return PerceptionResolution(
                source="models.pipeline", kind="zero_shot", detector="groundingdino",
                segmenter=segmenter, refusal=refusal,
            )

        routed = pipeline.router.enabled
        if not refusal:
            if routed and self.objectdetector is None:
                refusal = factory.REFUSE_ROUTER_WITHOUT_OBJECTDETECTOR
            elif zs.vlm.on_unavailable == "degrade" and self.objectdetector is None:
                refusal = factory.REFUSE_DEGRADE_WITHOUT_OBJECTDETECTOR
        return PerceptionResolution(
            source="models.pipeline", kind="zero_shot",
            detector="routed" if routed else "vlm",
            segmenter=segmenter, router_enabled=routed,
            vlm_model_id=zs.vlm.model_id, vlm_on_unavailable=zs.vlm.on_unavailable,
            refusal=refusal,
        )
