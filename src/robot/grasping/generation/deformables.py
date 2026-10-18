"""Deformable-object handling seam for the grasping pipeline.

This seam is inert. No production or sim caller passes
``deformable_strategy=`` or ``deformable_class=`` to the calculator, so the
gate is skipped by default. It is a clean extension point awaiting a real
classifier and handler, not an active capability; its two sample strategies
left on 2026-10-18, as nothing built them.

The :class:`src.robot.grasping.generation.calculator.GraspCalculator`
is engineered for rigid parallel-jaw grasps. Cables, cloth, bags and
similar deformable objects need a different planning regime
(specialised end-effectors, contact-rich strategies, human teleop). To
keep the rigid pipeline honest while still offering a clean extension
point, this module defines:

* :class:`DeformableClass`, a small, vendor-neutral taxonomy for the
  target object class.
* :class:`DeformableHandlingDecision`, the typed result returned by a
  strategy. ``proceed=True`` keeps the rigid pipeline running;
  ``proceed=False`` short-circuits with the supplied failure reasons.
* :class:`DeformableHandlingStrategy`, the
  :class:`typing.Protocol` strategies must satisfy; a strategy that refuses a
  non-rigid target answers with
  :data:`~src.robot.grasping.types.feedback.GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED`.

The seam is strictly additive: a calculator with no
``deformable_strategy`` configured ignores the ``deformable_class``
argument completely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.robot.grasping.types.feedback import GraspFailureReason

__all__ = [
    "DeformableClass",
    "DeformableHandlingDecision",
    "DeformableHandlingStrategy",
]


class DeformableClass(StrEnum):
    """Coarse classification of the target object's deformability.

    The taxonomy is intentionally small. ``RIGID`` is the default and
    triggers no special handling. ``UNKNOWN`` means the upstream
    classifier could not decide; strategies decide how to treat it.
    """

    RIGID = "rigid"
    CABLE = "cable"
    CLOTH = "cloth"
    BAG = "bag"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DeformableHandlingDecision:
    """Result returned by a :class:`DeformableHandlingStrategy`.

    ``proceed=True`` keeps the rigid pipeline running. Any
    ``telemetry`` entries are merged into the calculator's
    ``last_telemetry`` regardless of ``proceed`` so operators can
    inspect classifier hints. When ``proceed=False`` the calculator
    short-circuits with ``reasons`` (deduplicated).
    """

    proceed: bool
    reasons: tuple[GraspFailureReason, ...] = ()
    telemetry: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DeformableHandlingStrategy(Protocol):
    """Protocol implemented by deformable-handling strategies.

    Strategies are pure: they observe the segmentation mask and the
    classifier-supplied :class:`DeformableClass` and return a typed
    decision. They must not mutate the mask or perform IO.

    ``depth_map`` is on the signature and no strategy reads it: both
    implementations mark it ``# noqa: ARG002``. It stayed unread for a
    reason that has just gone, since a depth map whose masked region held
    one scalar carried no shape to diagnose. A strategy may read it now,
    and none does yet.
    """

    def handle(
        self,
        *,
        deformable_class: DeformableClass,
        mask: np.ndarray,
        depth_map: np.ndarray | None = None,
    ) -> DeformableHandlingDecision:
        """Return the routing decision for the given target class."""
        ...


