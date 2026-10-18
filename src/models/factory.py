"""The perception stack a cell builds, from ``models.pipeline``: the VLM or RT-DETR finds the boxes, SAM2 cuts them.

``build_perception`` is the one builder. ``zero_shot`` grounds any prompt with the VLM, the process's one copy
(:func:`src.models.vlm.holder.shared_vlm`); ``closed_set`` answers with RT-DETR's trained classes. SAM2 cuts the
masks of both, and every model reads its weights from its ``model_id`` under ``models.weights_root``
(``src/models/weights.py``). The torch-heavy wrappers are imported lazily inside each branch, so this module stays
importable without torch, on CI or macOS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.models.constants import MODELS_FACTORY_LOG_FILE, MODELS_LOG_DIR
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    # A structural Protocol over the four fields this module reads. Both `ModelsConfig` and `PerceptionSpec`
    # satisfy it, so a caller carrying no `stt` block reaches this builder without a second builder.
    from src.models.perception_backend import TwoStageBackend
    from src.models.perception_spec import PerceptionFields

__all__ = ["build_perception"]

#: One line per assembled stack. The leaf wrappers log their own weights and device, so what this file adds is the
#: composition, which nothing else records. Stdlib-only, so the no-torch-at-import promise in the module docstring
#: holds.
_LOG = create_logger(__name__, log_file=MODELS_FACTORY_LOG_FILE, log_dir=MODELS_LOG_DIR)

# The refusal texts are named once because they have to be predictable: the operator console answers "which stack
# would this cell build" before any weight loads, through `PerceptionSpec.resolve()`, which reads these constants. A
# predicted refusal and a raised one are then the same sentence.
#: Unreachable from YAML, on purpose: `ModelsConfig` makes `segmenter` required. It exists for `PerceptionSpec`, which
#: may be built in Python without one; without it that reaches an operator as an AttributeError inside a wrapper.
REFUSE_SAM2_WITHOUT_BLOCK = (
    "this stack cuts masks with SAM2, so a segmenter config block is required"
)
REFUSE_CLOSED_SET_WITHOUT_RTDETR = (
    "models.pipeline.kind='closed_set' needs a models.rtdetr config block: the closed-set "
    "stack is RT-DETR, and nothing else here answers with a fixed class list."
)


def build_perception(models: PerceptionFields, *, debug_images: bool = False) -> TwoStageBackend:
    """Build the whole perception stack in one call, from ``models.pipeline``.

    The caller names a stack and gets an object with a ``perceive(image_bgr, prompt)`` method, rather than building a
    detector and a segmenter and chaining them itself.

    Every refusal is decided before a single weight is loaded: building the segmenter first and only then discovering
    the stack is unbuildable spends seconds and gigabytes of VRAM on a configuration that cannot run.
    """
    from src.models.perception_backend import TwoStageBackend

    pipeline = models.pipeline
    if pipeline.kind == "closed_set" and models.rtdetr is None:
        raise ValueError(REFUSE_CLOSED_SET_WITHOUT_RTDETR)
    if models.segmenter is None:
        raise ValueError(REFUSE_SAM2_WITHOUT_BLOCK)
    if pipeline.kind == "closed_set":
        from src.models.detection.closed_set.detector import RtDetrObjectDetector

        assert models.rtdetr is not None  # refused above
        _LOG.info("building perception: closed-set rtdetr %s + sam2 %s", models.rtdetr.model_id,
                  models.segmenter.model_id)
        detector: Any = RtDetrObjectDetector(models.rtdetr, debug_images=debug_images,
                                             weights_root=models.weights_root)
    else:
        detector = _vlm_grounder(models)
    from src.models.segmentation.realtime.segmenter import Sam2Segmenter

    return TwoStageBackend(
        detector=detector,
        segmenter=Sam2Segmenter(models.segmenter, save_debug=debug_images, weights_root=models.weights_root),
    )


def _vlm_grounder(models: PerceptionFields) -> Any:
    """The VLM as the detector: the process's one copy, never loaded here.

    The VLM is wired in as a detector, so the same :class:`~src.models.perception_backend.TwoStageBackend` that runs
    RT-DETR composes it with the same segmenter, and there is no second pipeline to keep in step. It loads at the
    first prompt that reaches it, as the owner's cell did; a cell whose VLM cannot be loaded answers that prompt with
    nothing and a counted failure, which the console's task reads as ``detector_failed``, and refuses the pick.

    The grounder is the process's one copy (:func:`src.models.vlm.holder.shared_vlm`), not this build's own: the
    console's command reader asks the same object, and a rebuild finds it, loaded or not, instead of putting a second
    copy of the weights on the card. A build that names another checkpoint is how a cell switches: the holder unloads
    the copy it replaces.
    """
    from src.models.vlm.holder import shared_vlm

    vlm = models.pipeline.zero_shot.vlm
    _LOG.info("building perception: zero-shot vlm %s (loads at its first prompt) + sam2 %s", vlm.model_id,
              models.segmenter.model_id if models.segmenter is not None else "none")
    return shared_vlm().grounder_for(vlm, weights_root=models.weights_root)
