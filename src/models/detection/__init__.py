"""Object detection: the shared :class:`Detection` type and the backend packages.

The torch-heavy wrappers live in ``zero_shot`` (GroundingDINO, open vocabulary,
prompt-driven) and ``closed_set`` (RT-DETR, fixed classes, no prompt), and are
imported from there directly. ``object_detector`` answers with every class a
closed-set detector knows in one call, and with SAM2's mask for each object where
asked: :class:`ObjectDetector`, the :class:`Detections` it returns and the
:class:`DetectedObject` they hold. Only :class:`Detection` is imported here; those
three are imported on first use, which keeps an import of this package light and
free of torch.
"""

from __future__ import annotations

from typing import Any

from src.models.detection.types import Detection

#: The names ``object_detector`` defines, imported on first use: it reads OpenCV and the config schema.
_OBJECT_DETECTOR = frozenset({"DetectedObject", "Detections", "ObjectDetector"})


def __getattr__(name: str) -> Any:
    if name in _OBJECT_DETECTOR:
        from src.models.detection import object_detector

        value = getattr(object_detector, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DetectedObject", "Detection", "Detections", "ObjectDetector"]
