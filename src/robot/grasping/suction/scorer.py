"""The suction scorer seam: one interface, and the physics model that implements it.

The scorer is a :class:`SuctionScorer` Protocol, and that Protocol is the seam. :mod:`.synthesis` depends
on nothing else, so a different scorer swaps in without touching the pick path. ``prepare_scene`` is the
per-scene hook an image-based model needs.

:class:`AnalyticalSuctionScorer` is seal formation times wrench resistance, pure geometry and statics with
no dependency of its own, and it behaves cleanly on Isaac depth. It is the sim-validatable default and the
only implementation there is. A learned scorer may not be trained on a dataset released under a
non-commercial licence, which is not a boundary this project can rely on, so what one needs is not a
different vendor but licence-clean training data. That is what the scene generator is being built for,
with our own scenes, our own labels and no third-party dataset licence in the chain. Until then the
analytical model is the honest answer, and it is measured: a flat patch scores 1.0, a curved one or an
edge scores 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from src.robot.grasping.suction.seal import SealConfig, SealResult, evaluate_seal
from src.robot.grasping.suction.wrench import (
    WrenchConfig,
    WrenchResult,
    evaluate_wrench_resistance,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    pass

__all__ = [
    "SuctionQuality",
    "SuctionScorer",
    "AnalyticalSuctionScorer",
]


@dataclass(frozen=True, slots=True)
class SuctionQuality:
    """Combined suction quality at one contact, where ``quality`` in [0, 1] is the ranking key and ``source`` names the backend that produced it."""

    quality: float
    source: str
    seal: SealResult | None = None
    wrench: WrenchResult | None = None


@runtime_checkable
class SuctionScorer(Protocol):
    """Scores a single suction contact. The analytical model implements it, and so would any learned one."""

    def prepare_scene(
        self,
        rgb: np.ndarray | None,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
    ) -> None:
        """Per-scene setup before scoring. The analytical backend does nothing here; an image-based one would run its network once and cache the result."""
        ...

    def score(
        self,
        contact_point_mm: np.ndarray,
        approach: np.ndarray,
        points_mm: np.ndarray,
        *,
        surface_normal: np.ndarray | None = None,
        payload_mass_g: float | None = None,
        com_mm: np.ndarray | None = None,
        gravity_dir: np.ndarray | None = None,
    ) -> SuctionQuality: ...


@dataclass(frozen=True, slots=True)
class AnalyticalSuctionScorer:
    """The sim-validatable default: ``quality`` is ``seal_score`` times the wrench resist score, or ``seal_score`` alone when no payload is supplied."""

    seal_config: SealConfig = field(default_factory=SealConfig)
    wrench_config: WrenchConfig = field(default_factory=WrenchConfig)

    def prepare_scene(
        self, rgb: np.ndarray | None, depth_m: np.ndarray, intrinsics: np.ndarray
    ) -> None:
        """Does nothing: the analytical model scores directly from the point cloud, with no per-scene network pass."""
        return None

    def score(
        self,
        contact_point_mm: np.ndarray,
        approach: np.ndarray,
        points_mm: np.ndarray,
        *,
        surface_normal: np.ndarray | None = None,
        payload_mass_g: float | None = None,
        com_mm: np.ndarray | None = None,
        gravity_dir: np.ndarray | None = None,
    ) -> SuctionQuality:
        seal = evaluate_seal(
            contact_point_mm, approach, points_mm,
            surface_normal=surface_normal, config=self.seal_config,
        )
        wrench: WrenchResult | None = None
        quality = seal.seal_score
        if payload_mass_g is not None and com_mm is not None:
            wrench = evaluate_wrench_resistance(
                contact_point_mm, approach,
                payload_mass_g=payload_mass_g, com_mm=com_mm, config=self.wrench_config,
                gravity_dir=gravity_dir,
            )
            quality = seal.seal_score * wrench.resist_score
        return SuctionQuality(quality=float(quality), source="analytical", seal=seal, wrench=wrench)
