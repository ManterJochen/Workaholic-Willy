from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor

from src.config.schema.models.models_schema import SegmenterConfig
from src.models._inference import (
    autocast_ctx,
    build_load_kwargs,
    finalize_model,
    weight_load_errors,
)
from src.models.constants import MODELS_LOG_DIR, SEGMENTER_LOG_FILE
from src.models.detection.types import Detection
from src.models.segmentation.types import SegmentationResult
from src.utility import (
    bgr_to_rgb,
    create_logger,
    debug_dir,
    get_device,
    move_inputs_to_device,
    rgb_to_bgr,
)


class Sam2Segmenter:

    def __init__(
        self,
        config: SegmenterConfig,
        save_debug: bool = False,
        debug_subdir: str = "segmentation",
        keep_largest_component: bool = True,
        morph_kernel_size: int = 5,
    ) -> None:
        self.device = get_device()
        self.model_id = config.model_id
        self.model_path = config.model_path
        self.local = config.local
        self.optim = config.optim
        self.save_debug = save_debug
        self.debug_subdir = debug_subdir
        self.keep_largest_component = keep_largest_component
        self.morph_kernel_size = int(max(1, morph_kernel_size))

        self.logger = create_logger("SAM2", log_file=SEGMENTER_LOG_FILE, log_dir=MODELS_LOG_DIR)
        self.logger.info("Initializing Sam2Segmenter on device: %s", self.device)

        source = self.model_path if self.local else self.model_id
        if not source:
            raise ValueError("model_path (local=True) or model_id (local=False) must be set.")
        base_kwargs = {"local_files_only": True} if self.local else {}
        proc_kwargs = dict(base_kwargs)
        model_kwargs = build_load_kwargs(self.optim, self.device, base_kwargs)
        self._model_dtype = model_kwargs.get("torch_dtype")

        try:
            self.processor = Sam2Processor.from_pretrained(source, **proc_kwargs)  # type: ignore[arg-type]  # transformers stub mistypes from_pretrained **kwargs
            self.model = Sam2Model.from_pretrained(source, **model_kwargs).to(self.device)
        # RuntimeError caught none of the three ways weights are actually unusable.
        # Measured 2026-09-10 against SAM2 specifically: a README-only directory and a
        # half-downloaded snapshot both raise OSError ("Error no file named
        # model.safetensors, or pytorch_model.bin"), a corrupted file
        # safetensors.SafetensorError. See weight_load_errors().
        except weight_load_errors():
            self.logger.error("Failed to load SAM2 model '%s'", source)
            raise

        self.model = finalize_model(
            self.model, self.device, self.optim, vision=True,
        )

        # Debug images go through :func:`debug_dir` in ``_save_debug_overlay``,
        # which creates and rotates the output directory on demand, so there is
        # nothing to pre-create here.

    @torch.inference_mode()
    def segment_detection(
        self,
        image_bgr: np.ndarray,
        detection: Detection,
        frame_id: str | None = None,
    ) -> SegmentationResult:
        """Segment one ``Detection`` from ``GroundingDinoObjectDetector``.

        Takes the OpenCV BGR image, converts it to RGB and calls ``segment``.
        """
        image_rgb = bgr_to_rgb(image_bgr)
        x0, y0, x1, y1 = detection.box
        bbox = (int(x0), int(y0), int(x1), int(y1))
        return self.segment(
            image_rgb=image_rgb,
            bbox_xyxy=bbox,
            label=detection.label,
            score=detection.score,
            frame_id=frame_id,
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
        """Run SAM2 for exactly one bounding box.

        Parameters:
        - image_rgb: HxWx3 RGB numpy array
        - bbox_xyxy: (x1, y1, x2, y2), clipped to the image bounds
        - label: object label
        - score: optional upstream detection score
        """

        if not isinstance(image_rgb, np.ndarray):
            raise TypeError("image_rgb must be a numpy.ndarray.")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape (H, W, 3).")

        x1, y1, x2, y2 = self._clip_bbox_to_image(bbox_xyxy, image_rgb.shape[:2])
        clipped_box = (x1, y1, x2, y2)

        start = time.time()

        # The processor takes the box tensor on CPU.
        box_tensor = torch.tensor(clipped_box, dtype=torch.float32).view(1, 1, 4)

        inputs = self.processor(
            images=image_rgb,
            input_boxes=box_tensor,
            return_tensors="pt"
        )

        inputs = move_inputs_to_device(dict(inputs), self.device)

        if self.optim and self.optim.channels_last and self.device.type == "cuda":
            pv = inputs.get("pixel_values")
            if torch.is_tensor(pv) and pv.dim() == 4:
                inputs["pixel_values"] = pv.to(memory_format=torch.channels_last)

        with autocast_ctx(self.device, dtype=self._model_dtype):
            outputs = self.model(**inputs)

        post_masks = self.processor.image_processor.post_process_masks(  # type: ignore[attr-defined]  # Sam2Processor stub omits image_processor
            outputs.pred_masks.float().cpu(),
            inputs["original_sizes"]
        )[0]

        raw_mask = self._extract_first_mask(post_masks)
        clean_mask = self._postprocess_mask(raw_mask)

        ys, xs = np.where(clean_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise RuntimeError("SAM2 produced an empty mask after post-processing.")

        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))
        derived_bbox = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        )
        area = int(clean_mask.sum())
        inference_time_s = time.time() - start

        result = SegmentationResult(
            label=label,
            score=float(score),
            bbox_xyxy=clipped_box,
            mask=clean_mask.astype(np.uint8),
            mask_area_px=area,
            centroid_xy=(centroid_x, centroid_y),
            derived_bbox_xyxy=derived_bbox,
            inference_time_s=inference_time_s,
            timestamp_utc=datetime.now(UTC).isoformat(),
            frame_id=frame_id,
            metadata={
                "device": str(self.device),
                "keep_largest_component": self.keep_largest_component,
                "morph_kernel_size": self.morph_kernel_size,
            },
        )

        if self.save_debug:
            self._save_debug_overlay(image_rgb, result)

        self.logger.info(
            "SAM2 segmentation completed in %.3fs | label=%s | area=%d px",
            inference_time_s,
            label,
            area,
        )
        return result

    def _extract_first_mask(self, post_masks: Any) -> np.ndarray:
        """The first 2D mask from the SAM2 post-process output.

        Unwraps any list, tuple or extra leading dimension around it.
        """
        if post_masks is None or len(post_masks) == 0:
            raise RuntimeError("SAM2 post_process_masks returned no masks.")

        obj = post_masks[0]

        while isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                raise RuntimeError("Mask container is empty.")
            obj = obj[0]

        if torch.is_tensor(obj):
            arr = obj.detach().float().cpu().numpy()
        else:
            arr = np.asarray(obj, dtype=np.float32)

        while arr.ndim > 2:
            arr = arr[0]

        if arr.ndim != 2:
            raise RuntimeError(f"Expected a 2D mask, got shape {arr.shape}")

        return arr

    def _postprocess_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        """Threshold above zero, then morphological open and close.

        Keeps only the largest connected component when
        ``keep_largest_component`` is set.
        """
        binary: np.ndarray = (raw_mask > 0).astype(np.uint8)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        if self.keep_largest_component:
            binary = self._largest_connected_component(binary)

        return binary

    @staticmethod
    def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask

        # Row 0 of stats is the background component, hence the offset by one.
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(component_areas)) + 1

        out = np.zeros_like(mask, dtype=np.uint8)
        out[labels == largest_idx] = 1
        return out

    @staticmethod
    def _clip_bbox_to_image(
        bbox_xyxy: tuple[int, int, int, int],
        image_hw: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        h, w = image_hw
        x1, y1, x2, y2 = bbox_xyxy

        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))

        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)

        return x1, y1, x2, y2

    def _save_debug_overlay(self, image_rgb: np.ndarray, result: SegmentationResult) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        target_dir = debug_dir(self.debug_subdir, max_files=200)
        output_path = target_dir / f"sam2_debug_{timestamp}.png"

        mask_uint8 = (result.mask * 255).astype(np.uint8)
        mask_rgb = np.zeros((mask_uint8.shape[0], mask_uint8.shape[1], 3), dtype=np.uint8)
        mask_rgb[:, :, 1] = mask_uint8

        overlay = cv2.addWeighted(image_rgb, 1.0, mask_rgb, 0.45, 0.0)

        x1, y1, x2, y2 = result.bbox_xyxy
        dx1, dy1, dx2, dy2 = result.derived_bbox_xyxy

        overlay_bgr = rgb_to_bgr(overlay)
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(overlay_bgr, (dx1, dy1), (dx2, dy2), (0, 255, 255), 2)

        cx, cy = int(result.centroid_xy[0]), int(result.centroid_xy[1])
        cv2.circle(overlay_bgr, (cx, cy), 4, (0, 0, 255), -1)

        cv2.imwrite(str(output_path), overlay_bgr)
        self.logger.info("SAM2 debug image saved: %s", output_path)
