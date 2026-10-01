"""The training augmentations of the RT-DETR recipe, built from torchvision's box-aware transforms (no extra package).

The strong part, switched off for the last epochs (RT-DETR's ``stop_epoch`` policy, YOLO's ``close_mosaic``):

- ``RandomPhotometricDistort`` (p 0.5): brightness, contrast, saturation and hue;
- ``RandomZoomOut`` (p 0.5): the image shrinks onto a larger black canvas, so parts are also seen small;
- ``RandomIoUCrop`` (p 0.8): a crop that keeps the boxes it overlaps enough, so parts are also seen large and cut.

Always: ``RandomHorizontalFlip`` (p 0.5), and ``SanitizeBoundingBoxes`` (1 px), so no box leaves the image or
degenerates. An image without a box skips the crop, which needs one. Multi-scale (480 to 800 px) is applied per batch by
the trainer. The random draws come from torch's generator, which the trainer seeds per epoch and per loader worker.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["augment_sample", "build_transform"]


def build_transform(*, strong: bool, crop: bool) -> Any:
    """The transform for one sample: the strong part only when ``strong``, the crop only when ``crop``."""
    from torchvision.transforms import v2 as T  # noqa: PLC0415 (torch only where training runs)

    ops: list[Any] = []
    if strong:
        ops.append(T.RandomPhotometricDistort(p=0.5))
        ops.append(T.RandomZoomOut(fill=0, p=0.5))
        if crop:
            ops.append(T.RandomApply([T.RandomIoUCrop()], p=0.8))
    ops.append(T.RandomHorizontalFlip(p=0.5))
    ops.append(T.SanitizeBoundingBoxes(min_size=1))
    return T.Compose(ops)


def augment_sample(image: Any, boxes: Sequence[Sequence[float]], labels: Sequence[int], *,
                   strong: bool) -> tuple[Any, list[list[float]], list[int]]:
    """Augment a PIL image and its boxes (x1 y1 x2 y2, pixels) together; returns the image, boxes and labels left."""
    import torch  # noqa: PLC0415
    from torchvision import tv_tensors  # noqa: PLC0415

    height, width = image.height, image.width
    target = {
        "boxes": tv_tensors.BoundingBoxes(torch.tensor([list(b) for b in boxes], dtype=torch.float32).reshape(-1, 4),
                                          format="XYXY", canvas_size=(height, width)),
        "labels": torch.tensor(list(labels), dtype=torch.int64),
    }
    transform = build_transform(strong=strong, crop=bool(len(boxes)))
    out_image, out = transform(image, target)
    return out_image, out["boxes"].tolist(), out["labels"].tolist()
