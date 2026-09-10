"""Closed-set object detection (RT-DETR): a fixed class vocabulary, no text prompt."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from src.config.schema.models.models_schema import ObjectDetectorConfig
from src.models._inference import (
    autocast_ctx,
    build_load_kwargs,
    finalize_model,
    weight_load_errors,
)
from src.models.constants import MODELS_LOG_DIR, RTDETR_LOG_FILE
from src.models.detection.types import Detection
from src.utility import (
    bgr_to_rgb,
    create_logger,
    debug_dir,
    get_device,
    move_inputs_to_device,
)


#: Longest prompt, in words, still plausible as a class name. Two covers every COCO label and
#: most fine-tuned vocabularies ("traffic light", "cell phone"); three is a description.
_MAX_CLASS_NAME_WORDS = 2


class RtDetrObjectDetector:
    """RT-DETR detector over a fixed class set, for example COCO, with no text prompt.

    Runs the model's own label vocabulary over the whole image, so a scene can be
    perceived without a prompt, and emits the same :class:`Detection` type as the
    zero-shot GroundingDINO backend, so the two are interchangeable. The optional
    ``prompt``, passed positionally by the perception seam, is only a case-insensitive
    class-name filter; ``None`` returns every class above threshold.
    """

    def __init__(self, config: ObjectDetectorConfig, debug_images: bool = False) -> None:
        self.device = get_device()
        self.model_id = config.model_id
        self.model_path = config.model_path
        self.threshold = config.threshold
        self.local = config.local
        self.optim = config.optim
        self.debug_images = debug_images

        self.logger = create_logger("RTDETRObjectDetector", log_file=RTDETR_LOG_FILE, log_dir=MODELS_LOG_DIR)

        source = self.model_path if self.local else self.model_id
        if not source:
            raise ValueError("model_path (local=True) or model_id (local=False) must be set.")
        base_kwargs = {"local_files_only": True} if self.local else {}
        proc_kwargs = dict(base_kwargs)
        model_kwargs = build_load_kwargs(self.optim, self.device, base_kwargs)
        self._model_dtype = model_kwargs.get("torch_dtype")

        self.logger.info(
            "Initializing RTDETRObjectDetector: source='%s', threshold=%.2f, device=%s, dtype=%s",
            source, self.threshold, self.device, self._model_dtype,
        )
        try:
            self.processor = AutoImageProcessor.from_pretrained(source, **proc_kwargs)
            self.model = (
                AutoModelForObjectDetection.from_pretrained(source, **model_kwargs).to(self.device)
            )
        # RuntimeError caught none of the three ways weights are actually unusable
        # (ValueError, OSError, safetensors.SafetensorError; measured 2026-09-10). See
        # weight_load_errors().
        except weight_load_errors():
            self.logger.error("Failed to load model '%s'", source)
            raise
        self.model = finalize_model(self.model, self.device, self.optim, vision=True)
        self._id2label = dict(getattr(self.model.config, "id2label", {}) or {})

        if self.device.type == "cpu":
            self.logger.warning("RT-DETR model '%s' loaded on CPU. Inference may be slow.", source)
        else:
            self.logger.info("RT-DETR model '%s' loaded on %s.", source, self.device.type.upper())

    def _class_filter(self, prompt: str | None) -> str | None:
        """``prompt`` as a class-name filter, or a refusal when it is plainly not a class name.

        RT-DETR has no text encoder, so the prompt never reaches the model and can only be matched
        against the trained class list afterwards. Free text such as "the small green cylindrical
        part" matches no class name, and silently filtering on it would return an empty list that a
        caller cannot tell apart from an empty scene. Raising instead reports that this detector
        cannot answer the question, not that there is nothing there.

        The test is deliberately crude: a class name is one or two words, and anything longer is
        prose aimed at a detector that never reads it.
        """
        if prompt is None or not prompt.strip():
            return None
        cleaned = prompt.strip().rstrip(".").lower()
        if len(cleaned.split()) > _MAX_CLASS_NAME_WORDS:
            known = sorted({str(v) for v in self._id2label.values()})
            shown = ", ".join(known[:12]) + (f", ... (+{len(known) - 12} more)" if len(known) > 12 else "")
            raise ValueError(
                f"RT-DETR is a closed-set detector: it never sees the prompt text, so {prompt!r} can "
                f"only be matched against its trained class names, and it matches none of them. Use a "
                f"class name, or switch to a zero-shot backend for free-text prompts. "
                f"Known classes: {shown}"
            )
        return cleaned

    @torch.inference_mode()
    def detect_all(self, image: np.ndarray, prompt: str | None = None) -> list[Detection]:
        """Every detection above threshold, filtered by class name when ``prompt`` is given.

        Raises ``ValueError`` rather than returning nothing when ``prompt`` is free text instead
        of a class name; see :meth:`_class_filter`. Boxes are in pixels of the input image.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")
        h, w = image.shape[:2]
        pil_image = Image.fromarray(bgr_to_rgb(image))
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = move_inputs_to_device(dict(inputs), self.device)
        if self.optim and self.optim.channels_last and self.device.type == "cuda":
            pv = inputs.get("pixel_values")
            if torch.is_tensor(pv) and pv.dim() == 4:
                inputs["pixel_values"] = pv.to(memory_format=torch.channels_last)
        with autocast_ctx(self.device, dtype=self._model_dtype):
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([(h, w)], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )
        if not results:
            return []
        wanted = self._class_filter(prompt)
        dets: list[Detection] = []
        for box, label_id, score in zip(results[0]["boxes"], results[0]["labels"], results[0]["scores"]):
            name = str(self._id2label.get(int(label_id), int(label_id))).strip() or "object"
            if wanted is not None and wanted not in name.lower():
                continue
            x0, y0, x1, y1 = (float(c) for c in box)
            if not (x1 > x0 and y1 > y0):
                continue
            dets.append(Detection(
                box=[x0, y0, x1, y1],
                x_center=(x0 + x1) / 2.0,
                y_center=(y0 + y1) / 2.0,
                label=name,
                score=float(score),
            ))
        return dets

    @torch.inference_mode()
    def detect(self, image: np.ndarray, prompt: str | None = None) -> Detection:
        """Highest-scoring detection (optionally filtered by class name via ``prompt``)."""
        dets = self.detect_all(image, prompt)
        if not dets:
            raise ValueError(f"No object detected (prompt={prompt!r}, threshold={self.threshold}).")
        best = max(dets, key=lambda d: d.score)
        self.logger.info("RT-DETR: %s (%.3f) box=%s", best.label, best.score, [round(c, 1) for c in best.box])
        if self.debug_images:
            self._draw_detection(Image.fromarray(bgr_to_rgb(image)), best)
        return best

    def _draw_detection(self, image: Image.Image, det: Detection) -> None:
        draw = ImageDraw.Draw(image)
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        try:
            font = ImageFont.truetype("arialbd.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        x0, y0, x1, y1 = det.box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=4)
        text = f"{det.label} ({det.score:.1%})"
        draw.text((x0, y0), text, font=font, fill="white")
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        image.save(debug_dir("detection", max_files=200) / f"rtdetr_debug_{now}.png")
