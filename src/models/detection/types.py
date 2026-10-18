"""Shared detection result type, emitted by every detector backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Detection"]


@dataclass(frozen=True, slots=True)
class Detection:
    """Immutable detection result emitted by the VLM grounder and the closed-set detector.

    :meth:`__post_init__` enforces the contract: ``box`` has length 4 in
    ``(x0, y0, x1, y1)`` order with ``x1 > x0`` and ``y1 > y0``, the box corners and
    both centers are finite, ``score`` lies in ``[0, 1]``, and ``label`` is a non-empty
    string. A backend that cannot satisfy them cannot build a ``Detection``.
    """

    box: list[float]
    x_center: float
    y_center: float
    label: str
    score: float

    def __post_init__(self) -> None:
        if len(self.box) != 4:
            raise ValueError(
                f"Detection.box must have length 4 (x0, y0, x1, y1), got {len(self.box)}"
            )
        x0, y0, x1, y1 = self.box
        for name, v in (("x0", x0), ("y0", y0), ("x1", x1), ("y1", y1),
                        ("x_center", self.x_center), ("y_center", self.y_center),
                        ("score", self.score)):
            if not np.isfinite(v):
                raise ValueError(f"Detection.{name} must be finite, got {v!r}")
        if not (x1 > x0 and y1 > y0):
            raise ValueError(
                f"Detection.box must satisfy x1>x0 and y1>y0, got {self.box!r}"
            )
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Detection.score must be in [0, 1], got {self.score!r}")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError(f"Detection.label must be a non-empty string, got {self.label!r}")
