"""Fixtures for the detector-training tests: small COCO and YOLO datasets on disk, and a tiny RT-DETR.

The tiny model is the real ``RTDetrForObjectDetection`` with every width shrunk (about 80k parameters), built from a
config, so the training loop, its loss, the post-processing, ``save_pretrained`` and the detector's loading all run
on a CPU in seconds without downloading weights.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CLASSES = ("cube", "disc")


def draw_image(path: Path, boxes: Sequence[tuple[int, Sequence[float]]], *, size: tuple[int, int] = (96, 64),
               seed: int = 0) -> None:
    """A noisy image with a filled rectangle per box (label 0 red, label 1 green), corners x1 y1 x2 y2 in pixels."""
    from PIL import Image, ImageDraw

    width, height = size
    rng = np.random.default_rng(seed)
    image = Image.fromarray((rng.random((height, width, 3)) * 80 + 60).astype("uint8"))
    draw = ImageDraw.Draw(image)
    for label, (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], fill=(220, 40, 40) if label == 0 else (40, 200, 60))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def scene_boxes(index: int, size: tuple[int, int] = (96, 64)) -> list[tuple[int, list[float]]]:
    """One or two boxes per image, placed from the index alone."""
    width, height = size
    side = 16 + (index % 3) * 4
    x = 4 + (index * 7) % (width - side - 8)
    y = 4 + (index * 5) % (height - side - 8)
    boxes = [(index % 2, [float(x), float(y), float(x + side), float(y + side)])]
    if index % 3 == 0:
        boxes.append(((index + 1) % 2, [float(width - side - 2), float(height - side - 2), float(width - 2),
                                        float(height - 2)]))
    return boxes


def write_coco(folder: Path, count: int, *, start: int = 0, images_dir: str = "images",
               annotations: str = "annotations.json", categories: Sequence[Mapping[str, Any]] | None = None,
               extra: Sequence[Mapping[str, Any]] = ()) -> Path:
    """A COCO split: ``count`` images under ``folder/images_dir`` and their annotation file; returns the file."""
    cats = list(categories) if categories is not None else [{"id": 1, "name": CLASSES[0]}, {"id": 2, "name": CLASSES[1]}]
    ids = [c["id"] for c in cats if c["name"] in CLASSES]
    doc: dict[str, list[Any]] = {"images": [], "annotations": [], "categories": cats}
    for number in range(start, start + count):
        name = f"img_{number:03d}.png"
        boxes = scene_boxes(number)
        draw_image(folder / images_dir / name if images_dir else folder / name, boxes, seed=number)
        doc["images"].append({"id": number + 1, "file_name": name, "width": 96, "height": 64})
        for label, (x1, y1, x2, y2) in boxes:
            doc["annotations"].append({"id": len(doc["annotations"]) + 1, "image_id": number + 1,
                                       "category_id": ids[label], "bbox": [x1, y1, x2 - x1, y2 - y1],
                                       "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    doc["annotations"].extend(dict(item) for item in extra)
    path = folder / annotations
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def write_yolo_split(images: Path, labels: Path, count: int, *, start: int = 0) -> None:
    """``count`` images and their ``class cx cy w h`` label files."""
    for number in range(start, start + count):
        boxes = scene_boxes(number)
        draw_image(images / f"img_{number:03d}.png", boxes, seed=number)
        lines = []
        for label, (x1, y1, x2, y2) in boxes:
            lines.append(f"{label} {(x1 + x2) / 2 / 96:.6f} {(y1 + y2) / 2 / 64:.6f} {(x2 - x1) / 96:.6f} "
                         f"{(y2 - y1) / 64:.6f}")
        labels.mkdir(parents=True, exist_ok=True)
        (labels / f"img_{number:03d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tiny_factory(plan: Any, id2label: Mapping[int, str]) -> tuple[Any, Any]:
    """The real RT-DETR architecture at toy size, and an image processor at the plan's image size."""
    from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrImageProcessor, RTDetrResNetConfig

    import torch

    torch.manual_seed(0)
    backbone = RTDetrResNetConfig(embedding_size=8, hidden_sizes=[8, 16, 24, 32], depths=[1, 1, 1, 1],
                                  layer_type="basic", out_features=["stage2", "stage3", "stage4"],
                                  downsample_in_bottleneck=False)
    config = RTDetrConfig(backbone_config=backbone, encoder_in_channels=[16, 24, 32], decoder_in_channels=[16, 16, 16],
                          encoder_hidden_dim=16, d_model=16, encoder_ffn_dim=32, decoder_ffn_dim=32, encoder_layers=1,
                          decoder_layers=1, num_queries=12, num_denoising=4, decoder_attention_heads=2,
                          encoder_attention_heads=2, num_feature_levels=3, decoder_n_points=2,
                          num_labels=len(id2label), id2label=dict(id2label),
                          label2id={v: k for k, v in id2label.items()}, use_pretrained_backbone=False)
    processor = RTDetrImageProcessor(size={"height": plan.image_size, "width": plan.image_size})
    return RTDetrForObjectDetection(config), processor
