from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.config.schema.models.models_schema import ObjectDetectorConfig
from src.models._inference import (
    autocast_ctx,
    build_load_kwargs,
    finalize_model,
    weight_load_errors,
)
from src.models.constants import DETECTOR_LOG_FILE, MODELS_LOG_DIR
from src.models.detection.types import Detection
from src.utility import (
    bgr_to_rgb,
    create_logger,
    debug_dir,
    get_device,
    move_inputs_to_device,
)


class GroundingDinoObjectDetector:
    """Text-guided object detection with GroundingDINO, an open-vocabulary detector.

    ``config`` supplies the weights source, ``model_id`` (a Hugging Face Hub id such as
    "ShilongLi/groundingdino-v1.3") or ``model_path`` (a local directory), selected by
    ``local``, plus ``threshold``, the confidence a box must clear. ``debug_images`` writes
    an annotated copy of every frame :meth:`detect` handles. Results are :class:`Detection`
    objects carrying the grounded label and a box and center in pixels of the input image.
    """

    def __init__(self, config: ObjectDetectorConfig, debug_images: bool = False):
        self.device = get_device()
        self.model_id = config.model_id
        self.model_path = config.model_path
        self.threshold = config.threshold
        self.local = config.local
        self.optim = config.optim
        self.debug_images = debug_images

        self.logger = create_logger("GroundingDINOObjectDetector", log_file=DETECTOR_LOG_FILE, log_dir=MODELS_LOG_DIR)

        source = self.model_path if self.local else self.model_id
        if not source:
            raise ValueError("model_path (local=True) or model_id (local=False) must be set.")
        # A missing local directory has to be named as one here. Left to `from_pretrained`, the
        # path is taken for a Hub repo id and rejected as one, with an OSError reading "Repo id
        # must be in the form 'repo_name' or 'namespace/repo_name'", which points at a Hub
        # misconfiguration instead of at weights that are not where config says they are. The
        # shipped `model_path` does not exist in a fresh checkout.
        if self.local and not Path(source).is_dir():
            raise FileNotFoundError(
                f"models.objectdetector.local is true and model_path is {source!r}, but there is no "
                f"such directory (resolved: {Path(source).resolve()}). Nothing is downloaded in local "
                f"mode: that is the point of the flag. So either put the model files there, or "
                f"set local: false and models.objectdetector.model_id to a Hub id "
                f"(e.g. 'IDEA-Research/grounding-dino-base'), which uses the cache and downloads once."
            )
        base_kwargs = {"local_files_only": True} if self.local else {}
        proc_kwargs = dict(base_kwargs)
        model_kwargs = build_load_kwargs(self.optim, self.device, base_kwargs)
        self._model_dtype = model_kwargs.get("torch_dtype")

        self.logger.info(
            f"Initializing GroundingDINOObjectDetector: source='{source}', "
            f"threshold={self.threshold}, device={self.device}, "
            f"dtype={self._model_dtype}, attn={model_kwargs.get('attn_implementation')}"
        )

        try:
            self.processor = AutoProcessor.from_pretrained(source, **proc_kwargs)
            self.model = (
                AutoModelForZeroShotObjectDetection
                .from_pretrained(source, **model_kwargs)
                .to(self.device)
            )
        # Measured 2026-09-10: a README-only directory raises ValueError here, a
        # half-downloaded snapshot OSError, a corrupted model.safetensors
        # safetensors.SafetensorError. The clause caught RuntimeError, so this line, the
        # only one naming the model, was dead in all three. See weight_load_errors() for
        # why the set is named and not `Exception`.
        except weight_load_errors():
            self.logger.error(f"Failed to load model '{source}'")
            raise

        self.model = finalize_model(
            self.model, self.device, self.optim, vision=True,
        )

        device_type = self.device.type
        if device_type == "cpu":
            self.logger.warning(f"Dino Model '{source}' loaded on CPU. Inference may be slow.")
        else:
            self.logger.info(f"Dino Model '{source}' loaded on {device_type.upper()}.")

    @torch.inference_mode()
    def detect(self, image: np.ndarray, prompt: str) -> Detection:
        """Highest-scoring match for ``prompt`` in ``image``, a BGR array as OpenCV delivers it.

        ``prompt`` is a natural-language description of the target object. Returns one
        :class:`Detection` with box ``[x0, y0, x1, y1]`` in pixels, center, label and score, and
        raises ``ValueError`` when nothing clears the threshold; :meth:`detect_all` returns an
        empty list in that case instead.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        start_time = time.time()

        pil_image = Image.fromarray(bgr_to_rgb(image))
        w, h = pil_image.size

        inputs = self.processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        )
        inputs = move_inputs_to_device(dict(inputs), self.device)

        if self.optim and self.optim.channels_last and self.device.type == "cuda":
            pv = inputs.get("pixel_values")
            if torch.is_tensor(pv) and pv.dim() == 4:
                inputs["pixel_values"] = pv.to(memory_format=torch.channels_last)

        with autocast_ctx(self.device, dtype=self._model_dtype):
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.threshold
        )

        if not results or len(results[0]["scores"]) == 0:
            self.logger.info(f"No detections above threshold {self.threshold:.2f} for prompt: '{prompt}'")
            raise ValueError(f"No object detected with description: {prompt}")

        result = results[0]
        boxes = result["boxes"]
        labels = result.get("text_labels", result.get("labels"))
        scores = result["scores"]

        best_idx = scores.argmax()
        box = boxes[best_idx]
        label = labels[best_idx]
        score = scores[best_idx]

        # post_process_grounded_object_detection is called above without `target_sizes`, so its
        # boxes are normalised to [0, 1]. Multiplying by the image (width, height) is what makes
        # the `_px` values pixels; drop the *w/*h and they stay normalised.
        x0, y0, x1, y1 = [float(c) for c in box]

        x0_px, y0_px, x1_px, y1_px = x0 * w, y0 * h, x1 * w, y1 * h
        box_px = [x0_px, y0_px, x1_px, y1_px]

        x_center = (x0_px + x1_px) / 2
        y_center = (y0_px + y1_px) / 2
        score_val = float(score.item()) if hasattr(score, "item") else float(score)

        detection = Detection(
            box=box_px,
            x_center=x_center,
            y_center=y_center,
            label=label,
            score=score_val,
        )

        self._log_detection(detection, prompt)

        if self.debug_images:
            self._draw_detected_obj(pil_image, detection)

        inference_time = time.time() - start_time
        self.logger.info(f"Inference completed in {inference_time:.3f}s")

        return detection

    @torch.inference_mode()
    def detect_all(self, image: np.ndarray, prompt: str) -> list[Detection]:
        """Every detection above threshold, where :meth:`detect` returns only the argmax.

        Each :class:`Detection` carries its own box in pixels, center, label and score, which is
        what clutter and multi-object selection need. Nothing above threshold is an empty list
        rather than a raise, so a caller handles "no objects" without exception flow.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        # Timed as `detect` is: every runner reaches the detector through detect_all, so this is
        # the per-call inference number the system reports, and grounding is the most expensive
        # model in a pick.
        started = time.time()
        pil_image = Image.fromarray(bgr_to_rgb(image))
        w, h = pil_image.size
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt")
        inputs = move_inputs_to_device(dict(inputs), self.device)
        if self.optim and self.optim.channels_last and self.device.type == "cuda":
            pv = inputs.get("pixel_values")
            if torch.is_tensor(pv) and pv.dim() == 4:
                inputs["pixel_values"] = pv.to(memory_format=torch.channels_last)
        with autocast_ctx(self.device, dtype=self._model_dtype):
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=self.threshold
        )
        if not results or len(results[0]["scores"]) == 0:
            # Grounding nothing costs a full inference, so the empty case is timed and logged too.
            self.logger.info(f"Inference completed in {time.time() - started:.3f}s, 0 detections")
            return []
        result = results[0]
        boxes = result["boxes"]
        labels = result.get("text_labels", result.get("labels"))
        scores = result["scores"]
        dets: list[Detection] = []
        for i in range(len(scores)):
            x0, y0, x1, y1 = [float(c) for c in boxes[i]]
            box_px = [x0 * w, y0 * h, x1 * w, y1 * h]
            s = scores[i]
            # GroundingDINO can ground a box to an empty or None phrase, and `Detection` rejects an
            # empty label, so the fallback below supplies one.
            lbl = str(labels[i]).strip() if labels[i] is not None else ""
            dets.append(Detection(
                box=box_px, x_center=(box_px[0] + box_px[2]) / 2, y_center=(box_px[1] + box_px[3]) / 2,
                label=lbl or "object", score=float(s.item()) if hasattr(s, "item") else float(s),
            ))
        self.logger.info(f"Inference completed in {time.time() - started:.3f}s, {len(dets)} detections")
        return dets

    def _log_detection(self, det: Detection, prompt: str):
        self.logger.info(
            f"\n{'='*40}\n"
            f"Prompt      : {prompt}\n"
            f"Label       : {det.label}\n"
            f"Score       : {det.score:.3f}\n"
            f"Box Pixel   : x0={det.box[0]:.1f}, y0={det.box[1]:.1f}, x1={det.box[2]:.1f}, y1={det.box[3]:.1f}\n"
            f"Center      : x={det.x_center:.1f}, y={det.y_center:.1f}\n"
            f"{'='*40}"
        )

    def _draw_detected_obj(self, image: Image.Image, det: Detection):
        draw = ImageDraw.Draw(image)

        font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        try:
            font = ImageFont.truetype("arialbd.ttf", 20)
        except OSError:
            font = ImageFont.load_default()

        x0, y0, x1, y1 = det.box

        draw.rectangle([x0, y0, x1, y1], outline="red", width=4)

        text_to_draw = f"{det.label} ({det.score:.1%})"

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                draw.text(
                    (x0 + dx, y0 + dy),
                    text_to_draw,
                    font=font,
                    fill="black"
                )

        draw.text(
            (x0, y0),
            text_to_draw,
            font=font,
            fill="white"
        )

        # `debug_dir` caps and rotates the logs/debug/detection bucket at 200 files, so a long run
        # with debug_images on cannot fill the disk.
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"detection_debug_{now}.png"
        output_path = debug_dir("detection", max_files=200) / file_name

        image.save(output_path)
