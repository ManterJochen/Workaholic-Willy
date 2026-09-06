"""Deformable-object handling seam for the grasping pipeline.

This seam is inert. No production or sim caller passes
``deformable_strategy=`` or ``deformable_class=`` to the calculator, so the
gate is skipped by default and ``CablePCAStrategy`` always refuses: it emits a
telemetry hint but never yields a deformable grasp. It is a clean extension
point awaiting a real classifier and handler, not an active capability.

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
  :class:`typing.Protocol` strategies must satisfy.
* :class:`RefuseDeformableStrategy`, the safe default: refuses
  ``CABLE`` / ``CLOTH`` / ``BAG`` with
  :data:`~src.robot.grasping.types.feedback.GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED`.
* :class:`CablePCAStrategy`, an experimental, telemetry-only hint
  module: it stamps a PCA-derived principal axis and midpoint for
  ``CABLE`` targets but still refuses, so the rigid pipeline never
  produces a parallel-jaw plan for a deformable object before a real
  cable handler exists.

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
    "CablePCAStrategy",
    "DeformableClass",
    "DeformableHandlingDecision",
    "DeformableHandlingStrategy",
    "RefuseDeformableStrategy",
]


class DeformableClass(StrEnum):
    """Coarse classification of the target object's deformability.

    The taxonomy is intentionally small. ``RIGID`` is the default and
    triggers no special handling. ``UNKNOWN`` means the upstream
    classifier could not decide; strategies decide how to treat it (the
    default :class:`RefuseDeformableStrategy` treats ``UNKNOWN``
    optimistically as rigid).
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
    decision. They must not mutate the mask or perform IO. They may
    inspect the depth map for PCA / shape diagnostics.
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


@dataclass(frozen=True, slots=True)
class RefuseDeformableStrategy:
    """Safe default: refuse all known non-rigid classes.

    Parameters
    ----------
    refuse_unknown
        When ``True``, ``DeformableClass.UNKNOWN`` is also refused.
        Default ``False`` lets ``UNKNOWN`` proceed, so the gate can be
        enabled before a real classifier exists.
    """

    refuse_unknown: bool = False

    def handle(
        self,
        *,
        deformable_class: DeformableClass,
        mask: np.ndarray,  # noqa: ARG002 (protocol signature)
        depth_map: np.ndarray | None = None,  # noqa: ARG002
    ) -> DeformableHandlingDecision:
        if deformable_class is DeformableClass.RIGID:
            return DeformableHandlingDecision(proceed=True)
        if deformable_class is DeformableClass.UNKNOWN and not self.refuse_unknown:
            return DeformableHandlingDecision(
                proceed=True,
                telemetry={"deformable_class": deformable_class.value},
            )
        return DeformableHandlingDecision(
            proceed=False,
            reasons=(GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED,),
            telemetry={
                "deformable_class": deformable_class.value,
                "deformable_strategy": "refuse",
            },
        )


@dataclass(frozen=True, slots=True)
class CablePCAStrategy:
    """Experimental: telemetry-only PCA hint for ``CABLE`` targets.

    For ``CABLE`` masks, runs principal component analysis on the
    mask's pixel coordinates and stamps the principal axis direction,
    midpoint, and an elongation ratio (``sqrt(lambda_max /
    lambda_min)``) on telemetry under ``cable_pca_*``. The strategy
    always refuses with
    :data:`~src.robot.grasping.types.feedback.GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED`;
    the PCA fields exist so a future cable handler can be wired up
    without having to re-derive geometry. Other non-rigid classes are
    refused with the same ``deformable_class`` and
    ``deformable_strategy`` telemetry but without the ``cable_pca_*``
    fields. ``RIGID`` always proceeds; ``UNKNOWN`` proceeds unless
    ``refuse_unknown=True`` (default ``False``).
    """

    refuse_unknown: bool = False

    def handle(
        self,
        *,
        deformable_class: DeformableClass,
        mask: np.ndarray,
        depth_map: np.ndarray | None = None,  # noqa: ARG002 (reserved)
    ) -> DeformableHandlingDecision:
        if deformable_class is DeformableClass.RIGID:
            return DeformableHandlingDecision(proceed=True)
        if deformable_class is DeformableClass.UNKNOWN and not self.refuse_unknown:
            return DeformableHandlingDecision(
                proceed=True,
                telemetry={
                    "deformable_class": deformable_class.value,
                    "deformable_strategy": "cable_pca",
                },
            )

        telemetry: dict[str, Any] = {
            "deformable_class": deformable_class.value,
            "deformable_strategy": "cable_pca",
        }
        if deformable_class is DeformableClass.CABLE:
            pca = _mask_principal_axis(mask)
            if pca is not None:
                axis, midpoint, elongation = pca
                telemetry["cable_pca_axis_px"] = (float(axis[0]), float(axis[1]))
                telemetry["cable_pca_midpoint_px"] = (
                    float(midpoint[0]),
                    float(midpoint[1]),
                )
                telemetry["cable_pca_elongation"] = float(elongation)
        return DeformableHandlingDecision(
            proceed=False,
            reasons=(GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED,),
            telemetry=telemetry,
        )


def _mask_principal_axis(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return ``(axis_xy, midpoint_xy, elongation)`` for ``mask``, where ``axis_xy`` is a unit
    principal-direction 2-vector in pixels, or ``None`` for empty/degenerate point clouds."""

    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        return None
    ys, xs = np.nonzero(arr)
    if xs.size < 2:
        return None
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    cov = (centred.T @ centred) / float(pts.shape[0])
    # Symmetric 2x2 cov, so eigendecomposition is numerically safe.
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvals are ascending; the principal axis is the last column.
    lam_max = float(eigvals[-1])
    lam_min = float(eigvals[0])
    if lam_max <= 0.0:
        return None
    axis = eigvecs[:, -1]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        return None
    axis = axis / axis_norm
    elongation = float(np.sqrt(lam_max / max(lam_min, 1e-12)))
    return axis, centroid, elongation
