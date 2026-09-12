"""Config-driven perception-backend factory.

Resolves ``models.detector`` / ``models.segmenter_backend`` to the concrete wrapper, and
``models.pipeline`` to a whole assembled stack. The torch-heavy wrappers are imported lazily
inside each branch, so this module stays importable without torch, on CI or macOS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.models.constants import MODELS_FACTORY_LOG_FILE, MODELS_LOG_DIR
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    # A structural Protocol over the seven fields this module reads. Both `ModelsConfig` and
    # `PerceptionSpec` satisfy it, so a caller carrying no `stt` block (eleven speech fields, ten
    # of them mandatory, none read here) reaches this builder without a second builder.
    from src.models.perception_spec import PerceptionFields

__all__ = ["build_object_detector", "build_perception", "build_segmenter"]

#: One line per assembled stack. The leaf wrappers log their own weights and device, so what this
#: file adds is the composition: three config shapes (legacy keys, a preset, a routed pipeline)
#: resolve to different stacks, and nothing else records which one this cell got. Stdlib-only, so
#: the no-torch-at-import promise in the module docstring holds.
_LOG = create_logger(__name__, log_file=MODELS_FACTORY_LOG_FILE, log_dir=MODELS_LOG_DIR)

# The refusal texts are named once because they have to be predictable: the operator console answers
# "which stack would this cell build" before any weight loads, through `PerceptionSpec.resolve()`,
# which reads these constants. A predicted refusal and a raised one are then the same sentence.
REFUSE_RTDETR_WITHOUT_BLOCK = "models.detector='rtdetr' requires a models.rtdetr config block"
#: Unreachable from YAML, on purpose: `ModelsConfig` makes `objectdetector` and `segmenter`
#: required, so neither of the next two can fire for a config-driven cell. They exist for
#: `PerceptionSpec`, which may legitimately carry `objectdetector=None` because a closed-set stack
#: has no phrase grounder; without them that reaches an operator as an AttributeError inside a
#: model wrapper rather than as a sentence.
REFUSE_GROUNDINGDINO_WITHOUT_BLOCK = (
    "this stack grounds with GroundingDINO, so an objectdetector config block is required"
)
REFUSE_SAM2_WITHOUT_BLOCK = (
    "this stack cuts masks with SAM2, so a segmenter config block is required"
)
REFUSE_ONEFORMER_WITHOUT_BLOCK = (
    "models.segmenter_backend='oneformer' requires a models.oneformer config block"
)
REFUSE_NAMED_ONEFORMER_WITHOUT_BLOCK = (
    "segmenter 'oneformer' requires a models.oneformer config block"
)
REFUSE_CLOSED_SET_WITHOUT_RTDETR = (
    "models.pipeline.kind='closed_set' needs a models.rtdetr config block: the closed-set "
    "stack is RT-DETR, and nothing else here answers with a fixed class list."
)
REFUSE_ROUTER_WITHOUT_OBJECTDETECTOR = (
    "models.pipeline.router.enabled=true routes simple prompts to the phrase grounder, so a "
    "models.objectdetector block is required. Either add one, or set router.enabled: false."
)
REFUSE_VLM_WITHOUT_BLOCK = (
    "models.pipeline.zero_shot.backend='vlm' needs a models.pipeline.zero_shot.vlm block "
    "naming the checkpoint; refusing to guess which model was meant."
)
REFUSE_DEGRADE_WITHOUT_OBJECTDETECTOR = (
    "on_unavailable='degrade' falls back to the phrase detector, so a models.objectdetector "
    "block is required. Either add one, or use on_unavailable='refuse'."
)


def build_object_detector(models: PerceptionFields, *, debug_images: bool = False) -> Any:
    """Build the configured object detector (``groundingdino`` | ``rtdetr``).

    Both backends expose ``detect(image, prompt)`` and ``detect_all(image, prompt)`` returning
    :class:`~src.models.detection.types.Detection`. RT-DETR ignores an absent prompt and treats a
    present one as a class-name filter.
    """
    if models.detector == "rtdetr":
        if models.rtdetr is None:
            raise ValueError(REFUSE_RTDETR_WITHOUT_BLOCK)
        from src.models.detection.closed_set.detector import RtDetrObjectDetector

        return RtDetrObjectDetector(models.rtdetr, debug_images=debug_images)

    if models.objectdetector is None:
        raise ValueError(REFUSE_GROUNDINGDINO_WITHOUT_BLOCK)
    from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector

    return GroundingDinoObjectDetector(models.objectdetector, debug_images=debug_images)


def build_segmenter(models: PerceptionFields, *, save_debug: bool = False) -> Any:
    """Build the configured segmenter (``sam2`` | ``oneformer``).

    Either backend answers ``segment_detection(image_bgr, detection)`` with a
    :class:`~src.models.segmentation.types.SegmentationResult`.
    """
    if models.segmenter_backend == "oneformer":
        if models.oneformer is None:
            raise ValueError(REFUSE_ONEFORMER_WITHOUT_BLOCK)
        from src.models.segmentation.research.segmenter import OneFormerSegmenter

        return OneFormerSegmenter(models.oneformer, save_debug=save_debug)

    if models.segmenter is None:
        raise ValueError(REFUSE_SAM2_WITHOUT_BLOCK)
    from src.models.segmentation.realtime.segmenter import Sam2Segmenter

    return Sam2Segmenter(models.segmenter, save_debug=save_debug)


def build_perception(models: PerceptionFields, *, debug_images: bool = False) -> Any:
    """Build the whole perception stack in one call, from ``models.pipeline`` or the legacy keys.

    The caller names a stack and gets an object with a ``perceive(image_bgr, prompt)`` method,
    rather than building a detector and a segmenter and chaining them itself.

    With ``models.pipeline`` unset, this resolves the same two objects the legacy keys resolve
    through :func:`build_object_detector` and :func:`build_segmenter`, wrapped in
    :class:`TwoStageBackend`.
    """
    from src.models.perception_backend import TwoStageBackend

    pipeline = getattr(models, "pipeline", None)
    if pipeline is None:
        # Logged before the build: the two constructors below load weights, and a stack that dies
        # during that load is the one whose composition most needs to be on record.
        _LOG.info(
            "building perception: two-stage %s + %s (legacy keys, no models.pipeline)",
            models.detector, models.segmenter_backend,
        )
        return TwoStageBackend(
            detector=build_object_detector(models, debug_images=debug_images),
            segmenter=build_segmenter(models, save_debug=debug_images),
        )

    # Every refusal below is decided before a single weight is loaded: building the segmenter first
    # and only then discovering the stack is unbuildable spends seconds and gigabytes of VRAM on a
    # configuration that cannot run.
    if pipeline.kind == "closed_set" and models.rtdetr is None:
        raise ValueError(REFUSE_CLOSED_SET_WITHOUT_RTDETR)
    if pipeline.kind == "zero_shot" and pipeline.zero_shot.backend == "vlm":
        # The schema guarantees router.enabled implies backend='vlm', so this is the only branch
        # that can build a routed stack; without the router the same VLM stack is built unsplit.
        if pipeline.router.enabled:
            return _build_routed_backend(models, debug_images=debug_images)
        return _build_vlm_backend(models, debug_images=debug_images)

    if pipeline.kind == "closed_set":
        from src.models.detection.closed_set.detector import RtDetrObjectDetector

        assert models.rtdetr is not None  # refused above
        _LOG.info("building perception: closed-set rtdetr + %s", pipeline.closed_set.segmenter)
        detector: Any = RtDetrObjectDetector(models.rtdetr, debug_images=debug_images)
        segmenter_backend = pipeline.closed_set.segmenter
    else:
        if models.objectdetector is None:
            raise ValueError(REFUSE_GROUNDINGDINO_WITHOUT_BLOCK)
        from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector

        _LOG.info("building perception: zero-shot groundingdino + %s", pipeline.zero_shot.segmenter)
        detector = GroundingDinoObjectDetector(models.objectdetector, debug_images=debug_images)
        segmenter_backend = pipeline.zero_shot.segmenter

    return TwoStageBackend(
        detector=detector,
        segmenter=_build_named_segmenter(models, segmenter_backend, save_debug=debug_images),
    )


def _build_routed_backend(models: PerceptionFields, *, debug_images: bool) -> Any:
    """The full pipeline: route each prompt, then run the route it chose.

    Both routes are factories, so neither model loads until a prompt selects it. The two routes
    share one segmenter instance: SAM2 is the mask source on both sides, and building it twice
    would hold a second copy of the same weights.
    """
    from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector
    from src.models.perception_backend import TwoStageBackend
    from src.models.routed_backend import RoutedPerceptionBackend

    assert models.pipeline is not None  # only reached from build_perception's pipeline branch
    if models.objectdetector is None:
        raise ValueError(REFUSE_ROUTER_WITHOUT_OBJECTDETECTOR)

    # Both routes are lazy, so this line is the only record of the routed shape: the per-route
    # "building the ... route (first use)" lines arrive later, in the routing log, or never.
    _LOG.info(
        "building perception: routed (simple=groundingdino | vlm=%s), shared segmenter=%s, "
        "both routes lazy",
        models.pipeline.zero_shot.vlm.model_id if models.pipeline.zero_shot.vlm else "unconfigured",
        models.pipeline.zero_shot.segmenter,
    )
    # Built once and shared: whichever route runs first needs it, and both need it. One segmenter
    # choice covers both routes, so masks do not change shape depending on which grounding model
    # answered.
    segmenter = _build_named_segmenter(
        models, models.pipeline.zero_shot.segmenter, save_debug=debug_images,
    )

    def _simple() -> Any:
        assert models.objectdetector is not None  # the router refusal above already required it
        return TwoStageBackend(
            detector=GroundingDinoObjectDetector(models.objectdetector, debug_images=debug_images),
            segmenter=segmenter,
        )

    def _vlm() -> Any:
        return _build_vlm_backend(models, debug_images=debug_images, segmenter=segmenter)

    return RoutedPerceptionBackend(simple_factory=_simple, vlm_factory=_vlm)


def _build_vlm_backend(models: PerceptionFields, *, debug_images: bool, segmenter: Any = None) -> Any:
    """The VLM grounding route: Qwen as the detector, SAM2 for masks, guarded by ``on_unavailable``.

    The VLM is wired in as a detector, so the same :class:`TwoStageBackend` that runs the phrase
    route composes it with the same segmenter, and there is no second pipeline to keep in step.

    Two things are lazy: the grounder loads weights on first use unless ``preload`` says otherwise,
    and the degrade fallback is a factory rather than an instance, so a cell that never degrades
    never loads GroundingDINO. The refusal is not lazy; an unbuildable configuration is rejected
    here, before any weight loads.
    """
    from src.models.perception_backend import TwoStageBackend
    from src.models.vlm import GuardedVlmBackend, Qwen3VLGrounder

    assert models.pipeline is not None  # only reached from build_perception's pipeline branch
    vlm_cfg = models.pipeline.zero_shot.vlm
    if vlm_cfg is None:
        raise ValueError(REFUSE_VLM_WITHOUT_BLOCK)
    degrade = vlm_cfg.on_unavailable == "degrade"
    if degrade and models.objectdetector is None:
        # Caught at build time rather than at the moment of degradation, which would be mid-pick.
        raise ValueError(REFUSE_DEGRADE_WITHOUT_OBJECTDETECTOR)

    # `on_unavailable` decides whether a missing VLM refuses the prompt or silently grounds it with
    # the wrong model, so it belongs on the record next to the checkpoint.
    _LOG.info(
        "building perception: vlm %s (on_unavailable=%s, preload=%s) + %s",
        vlm_cfg.model_id, vlm_cfg.on_unavailable, vlm_cfg.preload,
        models.pipeline.zero_shot.segmenter,
    )
    grounder = Qwen3VLGrounder(
        model_id=vlm_cfg.model_id,
        model_path=vlm_cfg.model_path,
        local=vlm_cfg.local,
        preload=vlm_cfg.preload,
    )
    # The segmenter is configurable here: the VLM supplies boxes, and OneFormer takes a box exactly
    # as SAM2 does, so both mask sources work on this route. A routed stack passes its own instance
    # in, so both routes share one copy of the weights.
    if segmenter is None:
        segmenter = _build_named_segmenter(
            models, models.pipeline.zero_shot.segmenter, save_debug=debug_images,
        )

    def _fallback() -> Any:
        from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector

        assert models.objectdetector is not None  # the degrade refusal above already required it
        return TwoStageBackend(
            detector=GroundingDinoObjectDetector(models.objectdetector, debug_images=debug_images),
            segmenter=segmenter,
        )

    return GuardedVlmBackend(
        vlm=TwoStageBackend(detector=grounder, segmenter=segmenter),
        model_id=vlm_cfg.model_id,
        degrade=degrade,
        fallback_factory=_fallback if degrade else None,
    )


def _build_named_segmenter(models: PerceptionFields, backend: str, *, save_debug: bool) -> Any:
    """Build the segmenter named by ``backend``.

    The preset path shares this and does not read the legacy ``models.segmenter_backend`` key.
    """
    if backend == "oneformer":
        if models.oneformer is None:
            raise ValueError(REFUSE_NAMED_ONEFORMER_WITHOUT_BLOCK)
        from src.models.segmentation.research.segmenter import OneFormerSegmenter

        return OneFormerSegmenter(models.oneformer, save_debug=save_debug)

    if models.segmenter is None:
        raise ValueError(REFUSE_SAM2_WITHOUT_BLOCK)
    from src.models.segmentation.realtime.segmenter import Sam2Segmenter

    return Sam2Segmenter(models.segmenter, save_debug=save_debug)
