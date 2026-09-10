"""Universal segmentation (OneFormer): higher-accuracy instance masks, GPU-heavy.

Selected by ``segmenter_backend: oneformer`` and meant for capable hardware. It
segments the whole image once and returns the instance whose mask overlaps the
requested box most, emitting the same :class:`SegmentationResult` as
:class:`Sam2Segmenter`, which makes the two interchangeable in the perception
seam.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import cv2
import numpy as np
import torch
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

from src.config.schema.models.models_schema import OneFormerConfig
from src.models._inference import (
    autocast_ctx,
    build_load_kwargs,
    finalize_model,
    weight_load_errors,
)
from src.models.constants import MODELS_LOG_DIR, ONEFORMER_LOG_FILE
from src.models.detection.types import Detection
from src.models.segmentation.types import SegmentationResult
from src.utility import (
    bgr_to_rgb,
    create_logger,
    get_device,
    move_inputs_to_device,
)


class OneFormerSegmenter:
    """One universal-segmentation pass over the image, then the instance that matches the box."""

    def __init__(
        self,
        config: OneFormerConfig,
        save_debug: bool = False,
        keep_largest_component: bool = True,
        morph_kernel_size: int = 5,
    ) -> None:
        self.device = get_device()
        self.model_id = config.model_id
        self.model_path = config.model_path
        self.local = config.local
        self.optim = config.optim
        self.task = config.task
        self.save_debug = save_debug
        self.keep_largest_component = keep_largest_component
        self.morph_kernel_size = int(max(1, morph_kernel_size))

        self.logger = create_logger("OneFormer", log_file=ONEFORMER_LOG_FILE, log_dir=MODELS_LOG_DIR)

        source = self.model_path if self.local else self.model_id
        if not source:
            raise ValueError("model_path (local=True) or model_id (local=False) must be set.")
        base_kwargs = {"local_files_only": True} if self.local else {}
        model_kwargs = build_load_kwargs(self.optim, self.device, base_kwargs)
        self._model_dtype = model_kwargs.get("torch_dtype")

        try:
            self.processor = OneFormerProcessor.from_pretrained(source, **dict(base_kwargs))  # type: ignore[arg-type]  # transformers stub mistypes from_pretrained **kwargs
            self.model = (
                OneFormerForUniversalSegmentation.from_pretrained(source, **model_kwargs).to(self.device)
            )
        # RuntimeError caught none of the three ways weights are actually unusable
        # (ValueError, OSError, safetensors.SafetensorError; measured 2026-09-10). See
        # weight_load_errors().
        except weight_load_errors():
            self.logger.error("Failed to load OneFormer model '%s'", source)
            raise
        self.model = finalize_model(self.model, self.device, self.optim, vision=True)

        if self.device.type == "cpu":
            self.logger.warning("OneFormer model '%s' on CPU: very slow; a GPU is strongly recommended.", source)

    @torch.inference_mode()
    def segment_detection(
        self, image_bgr: np.ndarray, detection: Detection, frame_id: str | None = None
    ) -> SegmentationResult:
        """BGR entry point, mirroring :meth:`Sam2Segmenter.segment_detection`."""
        image_rgb = bgr_to_rgb(image_bgr)
        x0, y0, x1, y1 = detection.box
        return self.segment(
            image_rgb, (int(x0), int(y0), int(x1), int(y1)), detection.label, detection.score, frame_id
        )

    @torch.inference_mode()
    def segment(
        self,
        image_rgb: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
        label: str,
        score: float = 1.0,
        frame_id: str | None = None,
    ) -> SegmentationResult:
        if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape (H, W, 3).")
        h, w = image_rgb.shape[:2]
        box = self._clip_bbox(tuple(int(v) for v in bbox_xyxy), (h, w))
        start = time.time()

        inputs = self.processor(images=image_rgb, task_inputs=[self.task], return_tensors="pt")
        inputs = move_inputs_to_device(dict(inputs), self.device)
        with autocast_ctx(self.device, dtype=self._model_dtype):
            outputs = self.model(**inputs)
        seg = self.processor.post_process_instance_segmentation(outputs, target_sizes=[(h, w)])[0]
        seg_map = np.asarray(seg["segmentation"])
        segments = seg.get("segments_info", [])

        clean = self._clean_mask(self._pick_segment_in_box(seg_map, segments, box))
        ys, xs = np.where(clean > 0)
        if len(xs) == 0:
            raise RuntimeError("OneFormer produced an empty mask for the requested box.")

        result = SegmentationResult(
            label=label,
            score=float(score),
            bbox_xyxy=box,
            mask=clean.astype(np.uint8),
            mask_area_px=int(clean.sum()),
            centroid_xy=(float(np.mean(xs)), float(np.mean(ys))),
            derived_bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            inference_time_s=time.time() - start,
            timestamp_utc=datetime.now(UTC).isoformat(),
            frame_id=frame_id,
            metadata={"device": str(self.device), "task": self.task, "backend": "oneformer"},
        )
        self.logger.info(
            "OneFormer segmentation in %.3fs | label=%s | area=%d px",
            result.inference_time_s, label, result.mask_area_px,
        )
        return result

    @staticmethod
    def _pick_segment_in_box(seg_map: np.ndarray, segments: list, box: tuple[int, int, int, int]) -> np.ndarray:
        """The instance segment covering most of ``box``, or the box rectangle if none overlaps it."""
        x1, y1, x2, y2 = box
        box_area = max(1, (x2 - x1) * (y2 - y1))
        best_mask: np.ndarray | None = None
        best_overlap = 0.0
        for seg in segments:
            mask = seg_map == seg["id"]
            overlap = int(mask[y1:y2, x1:x2].sum()) / box_area
            if overlap > best_overlap:
                best_overlap, best_mask = overlap, mask
        if best_mask is None:
            fallback = np.zeros(seg_map.shape, dtype=np.uint8)
            fallback[y1:y2, x1:x2] = 1
            return fallback
        return best_mask.astype(np.uint8)

    def _clean_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        binary: np.ndarray = (raw_mask > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        if self.keep_largest_component:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if num > 1:
                largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
                binary = (labels == largest).astype(np.uint8)
        return binary

    @staticmethod
    def _clip_bbox(bbox: tuple[int, ...], image_hw: tuple[int, int]) -> tuple[int, int, int, int]:
        h, w = image_hw
        x1, y1, x2, y2 = bbox
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        return x1, y1, x2, y2
