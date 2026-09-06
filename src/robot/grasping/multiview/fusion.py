"""Bounded, deterministic multi-view fusion substrate.

This module is the substrate layer for multi-view dense decisioning.
Deployment is shadow-only: even when enabled, no grasping, ranking,
decision or recovery path consumes the fused state. The substrate
only:

1. ingests per-attempt depth captures into a bounded voxel-occupancy
   grid in BASE-frame coordinates,
2. tracks two per-voxel counters as saturating ``uint16``
   accumulators, ``hits`` for depth samples that landed there and
   ``seen`` for how many views touched it, and
3. enforces a strict frame and intrinsics policy so a misconfigured
   camera cannot silently poison the grid.

Turning the fused state into a runtime decision driver is future work.
Until then any reader is welcome, whether telemetry, replay or a debug
overlay, but the pick loop must remain byte-identical to the disabled
path when ``RobotGraspingFusionConfig.enabled`` is ``False``.

Determinism contract
--------------------

* Voxel indices are computed with ``np.floor``-based integer
  arithmetic; canonical (i, j, k) order is row-major (x, y, z).
* Per-view aggregation reduces to a unique-voxel list with integer
  per-voxel hit counts before being written to the grid. Aggregation
  uses ``numpy.unique(..., return_counts=True)``, which is sort-based
  and order-stable for a given input.
* Eviction is exact: each retained :class:`FusedView` carries its own
  flat voxel index and per-voxel count arrays, so :meth:`SceneFusion`
  can rebuild the grid by summing the surviving views with no loss of
  fidelity and no need to re-run backprojection.
* No ``set`` iteration, no parallel reduction, no float-sum reordering
  in counter updates.

The module is numpy-only, with no open3d, no torch and no scipy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple, cast

import numpy as np

from src.geometry import Frame, Transform
from src.robot.grasping.multiview._fusion_geometry import backproject_to_unique_voxels


__all__ = [
    "FusionConfig",
    "FusedView",
    "FusionTelemetry",
    "IngestResult",
    "SceneFusion",
    "INGEST_ACCEPTED",
    "INGEST_REFUSED",
    "REFUSE_DISABLED",
    "REFUSE_BAD_FRAME",
    "REFUSE_BAD_INTRINSICS",
    "REFUSE_BAD_DEPTH_SHAPE",
    "REFUSE_INTRINSICS_DRIFT",
    "REFUSE_NO_VALID_SAMPLES",
]


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Frozen runtime carrier mirroring ``RobotGraspingFusionConfig``."""

    enabled: bool = False
    max_views: int = 6
    max_view_age_s: float = 20.0
    voxel_size_mm: float = 6.0
    roi_extent_mm: Tuple[float, float, float] = (600.0, 600.0, 360.0)
    max_voxels: int = 1_000_000
    depth_min_mm: float = 80.0
    depth_max_mm: float = 1_400.0
    intrinsics_atol: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("FusionConfig.enabled must be bool")
        if not isinstance(self.max_views, int) or isinstance(self.max_views, bool):
            raise TypeError("FusionConfig.max_views must be int")
        if self.max_views < 1 or self.max_views > 64:
            raise ValueError("FusionConfig.max_views must be in [1, 64]")
        if not _finite_pos_float(self.max_view_age_s) or self.max_view_age_s > 600.0:
            raise ValueError(
                "FusionConfig.max_view_age_s must be finite, > 0, <= 600"
            )
        if not _finite_pos_float(self.voxel_size_mm) or self.voxel_size_mm > 100.0:
            raise ValueError(
                "FusionConfig.voxel_size_mm must be finite, > 0, <= 100"
            )
        if (
            not isinstance(self.roi_extent_mm, tuple)
            or len(self.roi_extent_mm) != 3
        ):
            raise TypeError(
                "FusionConfig.roi_extent_mm must be a 3-tuple of floats"
            )
        for extent in self.roi_extent_mm:
            if not _finite_pos_float(extent):
                raise ValueError(
                    "FusionConfig.roi_extent_mm components must be > 0"
                )
        vs = float(self.voxel_size_mm)
        for axis_label, extent in zip("xyz", self.roi_extent_mm):
            ratio = float(extent) / vs
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(
                    f"FusionConfig.roi_extent_mm[{axis_label}]={extent} "
                    f"must be an integer multiple of voxel_size_mm={vs}"
                )
        if not isinstance(self.max_voxels, int) or self.max_voxels < 1000:
            raise ValueError("FusionConfig.max_voxels must be int >= 1000")
        total = 1
        for extent in self.roi_extent_mm:
            total *= int(round(float(extent) / vs))
        if total > self.max_voxels:
            raise ValueError(
                f"FusionConfig: ROI/voxel combo allocates {total} voxels "
                f"but max_voxels={self.max_voxels}"
            )
        if (
            not _finite_pos_float(self.depth_min_mm)
            or not _finite_pos_float(self.depth_max_mm)
            or self.depth_min_mm >= self.depth_max_mm
        ):
            raise ValueError(
                "FusionConfig: 0 < depth_min_mm < depth_max_mm required"
            )
        if (
            not isinstance(self.intrinsics_atol, float)
            or self.intrinsics_atol < 0.0
            or self.intrinsics_atol > 1.0
        ):
            raise ValueError(
                "FusionConfig.intrinsics_atol must be float in [0, 1]"
            )

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Number of voxels along each axis (x, y, z)."""

        vs = float(self.voxel_size_mm)
        return cast(
            "tuple[int, int, int]",
            tuple(int(round(float(e) / vs)) for e in self.roi_extent_mm),
        )

    @property
    def grid_origin_mm(self) -> Tuple[float, float, float]:
        """BASE-frame coordinate of the corner of voxel index (0, 0, 0)."""

        return cast(
            "tuple[float, float, float]",
            tuple(-0.5 * float(e) for e in self.roi_extent_mm),
        )


def _finite_pos_float(x: object) -> bool:
    if isinstance(x, bool):
        return False
    if not isinstance(x, (int, float)):
        return False
    f = float(x)
    return np.isfinite(f) and f > 0.0


@dataclass(frozen=True, slots=True)
class FusedView:
    """The bookkeeping for one accepted view, frozen and immutable."""

    # flat_voxel_indices / voxel_hit_counts are parallel arrays: entry k means this view
    # deposited voxel_hit_counts[k] samples in flat voxel flat_voxel_indices[k]. Storing
    # aggregated (not raw) samples bounds memory and lets the grid rebuild exactly after an
    # eviction with no raw-depth retention.
    timestamp_ns: int
    intrinsics: np.ndarray  # 3x3, float64, read-only
    t_cam_to_base: np.ndarray  # 4x4, float64, read-only
    depth_shape: Tuple[int, int]
    valid_sample_count: int
    flat_voxel_indices: np.ndarray  # 1-D int64, read-only
    voxel_hit_counts: np.ndarray  # 1-D uint32, read-only

    def __post_init__(self) -> None:
        if (
            not isinstance(self.intrinsics, np.ndarray)
            or self.intrinsics.shape != (3, 3)
            or self.intrinsics.dtype != np.float64
        ):
            raise TypeError("FusedView.intrinsics must be float64 3x3 ndarray")
        if (
            not isinstance(self.t_cam_to_base, np.ndarray)
            or self.t_cam_to_base.shape != (4, 4)
            or self.t_cam_to_base.dtype != np.float64
        ):
            raise TypeError("FusedView.t_cam_to_base must be float64 4x4 ndarray")
        if (
            not isinstance(self.flat_voxel_indices, np.ndarray)
            or self.flat_voxel_indices.dtype != np.int64
            or self.flat_voxel_indices.ndim != 1
        ):
            raise TypeError(
                "FusedView.flat_voxel_indices must be 1-D int64 ndarray"
            )
        if (
            not isinstance(self.voxel_hit_counts, np.ndarray)
            or self.voxel_hit_counts.dtype != np.uint32
            or self.voxel_hit_counts.ndim != 1
            or self.voxel_hit_counts.shape != self.flat_voxel_indices.shape
        ):
            raise TypeError(
                "FusedView.voxel_hit_counts must be 1-D uint32 ndarray of "
                "matching length"
            )
        self.intrinsics.setflags(write=False)
        self.t_cam_to_base.setflags(write=False)
        self.flat_voxel_indices.setflags(write=False)
        self.voxel_hit_counts.setflags(write=False)


# IngestResult.status values, a frozen wire contract.
INGEST_ACCEPTED = "accepted"
INGEST_REFUSED = "refused"

# Refusal reasons, a frozen wire contract.
REFUSE_DISABLED = "disabled"
REFUSE_BAD_FRAME = "bad_frame"
REFUSE_BAD_INTRINSICS = "bad_intrinsics"
REFUSE_BAD_DEPTH_SHAPE = "bad_depth_shape"
REFUSE_INTRINSICS_DRIFT = "intrinsics_drift"
REFUSE_NO_VALID_SAMPLES = "no_valid_samples"


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of one :meth:`SceneFusion.ingest` call."""

    status: str
    reason: Optional[str] = None
    view: Optional[FusedView] = None
    valid_samples: int = 0
    hit_voxels: int = 0


@dataclass(frozen=True, slots=True)
class FusionTelemetry:
    """Per-pick rolling summary attached to ``PickReport``."""

    enabled: bool
    views_attempted: int
    views_accepted: int
    views_rejected: int
    last_reject_reason: Optional[str]
    voxels_seen: int
    voxels_hit: int


@dataclass(frozen=True, slots=True)
class CorridorEvidence:
    """Read-side query result for one approach corridor: the voxels inside the approach-vector cylinder, clipped to the ROI, with their hit and seen counts. ``hit_fraction`` and ``seen_fraction`` are 0.0 when ``queried_voxels == 0``."""

    queried_voxels: int
    hit_voxels: int
    seen_voxels: int
    hit_fraction: float
    seen_fraction: float
    views_accepted: int


@dataclass(frozen=True, slots=True)
class ViewpointGain:
    """Read-side query result predicting the new information a candidate viewpoint adds. It is an upper bound, because there is no occlusion test against existing hits, which would need a full ray-traversal pass, so it is a deterministic monotone proxy for next-best-viewpoint scoring."""

    unseen_voxels: int
    predicted_visible_unseen: int
    image_shape: Tuple[int, int]


class SceneFusion:
    """Bounded voxel-occupancy fusion grid (BASE frame)."""

    # Not thread-safe: the orchestrator runs one pick at a time on a single thread.

    def __init__(self, config: FusionConfig) -> None:
        if not isinstance(config, FusionConfig):
            raise TypeError("SceneFusion requires a FusionConfig")
        self._cfg = config
        nx, ny, nz = config.grid_shape
        self._shape: Tuple[int, int, int] = (nx, ny, nz)
        # Accumulators: hits is the total depth samples that landed in this voxel,
        # seen is how many distinct accepted views touched it. Free space is not
        # carved out of either.
        self._hits = np.zeros(self._shape, dtype=np.uint16)
        self._seen = np.zeros(self._shape, dtype=np.uint16)
        self._views: Deque[FusedView] = deque()
        self._ref_intrinsics: Optional[np.ndarray] = None
        self._views_attempted = 0
        self._views_accepted = 0
        self._views_rejected = 0
        self._last_reject_reason: Optional[str] = None

    @property
    def config(self) -> FusionConfig:
        return self._cfg

    @property
    def views(self) -> Tuple[FusedView, ...]:
        """Currently retained accepted views in insertion order."""

        return tuple(self._views)

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def hits(self) -> np.ndarray:
        """Read-only view of the hit-count accumulator."""

        v = self._hits.view()
        v.setflags(write=False)
        return v

    @property
    def seen(self) -> np.ndarray:
        """Read-only view of the per-voxel view-count accumulator."""

        v = self._seen.view()
        v.setflags(write=False)
        return v

    def voxels_hit(self) -> int:
        return int(np.count_nonzero(self._hits))

    def voxels_seen(self) -> int:
        return int(np.count_nonzero(self._seen))

    def telemetry(self) -> FusionTelemetry:
        return FusionTelemetry(
            enabled=self._cfg.enabled,
            views_attempted=self._views_attempted,
            views_accepted=self._views_accepted,
            views_rejected=self._views_rejected,
            last_reject_reason=self._last_reject_reason,
            voxels_seen=self.voxels_seen(),
            voxels_hit=self.voxels_hit(),
        )

    def corridor_evidence(
        self,
        *,
        position_mm: np.ndarray,
        approach: np.ndarray,
        length_mm: float,
        radius_mm: float,
    ) -> CorridorEvidence:
        """Read-side query: evidence inside an approach corridor.

        The corridor is the cylinder of radius ``radius_mm`` extruded
        for ``length_mm`` along ``-approach`` (the volume the gripper
        traverses on its way to the grasp), clipped to the configured
        ROI; hits/seen for the enclosed voxel centers come from the
        current :attr:`hits` / :attr:`seen` accumulators. Strictly
        read-only and deterministic (no eviction, no float reordering).

        Parameters
        ----------
        position_mm
            (3,) BASE-frame mm coordinate of the grasp anchor.
        approach
            (3,) approach direction (unit norm not required; the
            method normalises internally and refuses a zero vector).
        length_mm
            Length of the corridor cylinder along ``-approach``
            (positive, finite).
        radius_mm
            Cylinder radius (positive, finite).

        Raises
        ------
        TypeError / ValueError on contract violation.
        """

        from src.robot.grasping.multiview._fusion_queries import (
            corridor_evidence as _corridor_evidence,
        )

        return _corridor_evidence(
            cfg=self._cfg,
            hits=self._hits,
            seen=self._seen,
            shape=self._shape,
            views_accepted=self._views_accepted,
            position_mm=position_mm,
            approach=approach,
            length_mm=length_mm,
            radius_mm=radius_mm,
        )

    def viewpoint_information_gain(
        self,
        *,
        t_cam_to_base: Transform,
        intrinsics: np.ndarray,
        depth_shape: Tuple[int, int],
    ) -> ViewpointGain:
        """Read-side query: predicted new-information count.

        Counts in-ROI voxels that (a) currently have ``seen == 0`` and
        (b) would project inside the candidate camera's image bounds
        and depth range. Strictly read-only and deterministic; callers
        wire this in behind ``fusion.active_perception_use_fusion``
        (default off).

        Parameters
        ----------
        t_cam_to_base
            ``Frame.CAMERA -> Frame.BASE`` transform of the candidate
            viewpoint.
        intrinsics
            (3, 3) float64 intrinsic matrix. Same contract as
            :meth:`ingest`.
        depth_shape
            ``(height, width)`` of the candidate camera image plane.

        Raises
        ------
        TypeError / ValueError on contract violation.
        """

        from src.robot.grasping.multiview._fusion_queries import (
            viewpoint_information_gain as _viewpoint_information_gain,
        )

        return _viewpoint_information_gain(
            cfg=self._cfg,
            seen=self._seen,
            t_cam_to_base=t_cam_to_base,
            intrinsics=intrinsics,
            depth_shape=depth_shape,
        )

    def clear(self) -> None:
        """Reset the grid and view history to an empty state.

        Cumulative attempt/accept/reject counters are preserved so the
        per-session telemetry reflects the full history.
        """

        self._hits.fill(0)
        self._seen.fill(0)
        self._views.clear()
        self._ref_intrinsics = None

    def ingest(
        self,
        *,
        depth_map: np.ndarray,
        intrinsics: np.ndarray,
        t_cam_to_base: Transform,
        timestamp_ns: int,
    ) -> IngestResult:
        """Backproject ``depth_map`` into the grid as one view.

        Strict frame / intrinsic contract:

        * ``t_cam_to_base.from_frame`` must be :attr:`Frame.CAMERA`
          and ``to_frame`` must be :attr:`Frame.BASE`.
        * ``intrinsics`` must be a 3x3 finite matrix with positive
          ``fx`` and ``fy``.
        * ``depth_map`` must be a non-empty 2-D array.
        * If any prior view was accepted, ``intrinsics`` must match the
          reference within :attr:`FusionConfig.intrinsics_atol`.

        Any contract violation refuses the view without mutating the
        grid. Refusals are recorded in cumulative counters.
        """

        self._views_attempted += 1

        if not self._cfg.enabled:
            return self._refuse(REFUSE_DISABLED)

        # Frame contract -------------------------------------------------------
        if not isinstance(t_cam_to_base, Transform):
            return self._refuse(REFUSE_BAD_FRAME)
        if (
            t_cam_to_base.from_frame is not Frame.CAMERA
            or t_cam_to_base.to_frame is not Frame.BASE
        ):
            return self._refuse(REFUSE_BAD_FRAME)

        # Intrinsics contract --------------------------------------------------
        if (
            not isinstance(intrinsics, np.ndarray)
            or intrinsics.ndim != 2
            or intrinsics.shape != (3, 3)
        ):
            return self._refuse(REFUSE_BAD_INTRINSICS)
        try:
            K = np.asarray(intrinsics, dtype=np.float64)
        except (TypeError, ValueError):
            return self._refuse(REFUSE_BAD_INTRINSICS)
        if not np.all(np.isfinite(K)):
            return self._refuse(REFUSE_BAD_INTRINSICS)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        if fx <= 0.0 or fy <= 0.0:
            return self._refuse(REFUSE_BAD_INTRINSICS)
        if self._ref_intrinsics is not None:
            if not np.allclose(
                K, self._ref_intrinsics,
                atol=self._cfg.intrinsics_atol, rtol=0.0,
            ):
                return self._refuse(REFUSE_INTRINSICS_DRIFT)

        # Depth shape contract -------------------------------------------------
        if (
            not isinstance(depth_map, np.ndarray)
            or depth_map.ndim != 2
            or depth_map.size == 0
        ):
            return self._refuse(REFUSE_BAD_DEPTH_SHAPE)
        D = np.asarray(depth_map, dtype=np.float64)

        # Backproject and aggregate -------------------------------------------
        flat_idx, counts, n_valid = self._backproject_to_unique_voxels(
            D, K, t_cam_to_base
        )
        if flat_idx.size == 0:
            return self._refuse(REFUSE_NO_VALID_SAMPLES)

        self._apply_view_counts(flat_idx, counts)

        # Bookkeeping ----------------------------------------------------------
        if self._ref_intrinsics is None:
            self._ref_intrinsics = K.copy()
            self._ref_intrinsics.setflags(write=False)

        view = FusedView(
            timestamp_ns=int(timestamp_ns),
            intrinsics=K.copy(),
            t_cam_to_base=np.asarray(
                t_cam_to_base.to_matrix(), dtype=np.float64
            ).copy(),
            depth_shape=(int(D.shape[0]), int(D.shape[1])),
            valid_sample_count=int(n_valid),
            flat_voxel_indices=flat_idx.copy(),
            voxel_hit_counts=counts.copy(),
        )
        self._views.append(view)
        self._views_accepted += 1

        hit_voxels_this_view = int(flat_idx.size)

        # Apply eviction policies and rebuild from surviving views if
        # any were dropped.
        if self._apply_eviction():
            self._rebuild_grid_from_views()

        return IngestResult(
            status=INGEST_ACCEPTED,
            view=view,
            valid_samples=int(n_valid),
            hit_voxels=hit_voxels_this_view,
        )

    def _refuse(self, reason: str) -> IngestResult:
        self._views_rejected += 1
        self._last_reject_reason = reason
        return IngestResult(status=INGEST_REFUSED, reason=reason)

    def _backproject_to_unique_voxels(
        self,
        depth: np.ndarray,
        K: np.ndarray,
        t_cam_to_base: Transform,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Aggregated ``(flat_idx, counts, n_valid)`` for one view (thin wrapper over the geometry leaf)."""

        return backproject_to_unique_voxels(
            depth,
            K,
            t_cam_to_base,
            grid_origin_mm=self._cfg.grid_origin_mm,
            voxel_size_mm=self._cfg.voxel_size_mm,
            shape=self._shape,
            depth_min_mm=self._cfg.depth_min_mm,
            depth_max_mm=self._cfg.depth_max_mm,
        )

    def _apply_view_counts(
        self, flat_idx: np.ndarray, counts: np.ndarray
    ) -> None:
        """Add per-voxel counts of one view into hits/seen with saturation."""

        if flat_idx.size == 0:
            return
        hits_flat = self._hits.reshape(-1)
        seen_flat = self._seen.reshape(-1)

        # Saturating add for hits (per-voxel sample count is additive).
        current = hits_flat[flat_idx].astype(np.uint32)
        added = current + counts.astype(np.uint32)
        np.minimum(added, np.uint32(0xFFFF), out=added)
        hits_flat[flat_idx] = added.astype(np.uint16)

        # ``seen`` tracks how many distinct views touched a voxel:
        # +1 per view that had any hit there.
        seen_cur = seen_flat[flat_idx].astype(np.uint32)
        seen_added = seen_cur + np.uint32(1)
        np.minimum(seen_added, np.uint32(0xFFFF), out=seen_added)
        seen_flat[flat_idx] = seen_added.astype(np.uint16)

    def _apply_eviction(self) -> bool:
        """Drop views by fifo over-cap and by age. Return ``True`` if any."""

        evicted = False
        while len(self._views) > self._cfg.max_views:
            self._views.popleft()
            evicted = True
        # Age cap is relative to the newest accepted view.
        if not self._views:
            return evicted
        newest_ns = self._views[-1].timestamp_ns
        max_age_ns = int(self._cfg.max_view_age_s * 1e9)
        while self._views and (newest_ns - self._views[0].timestamp_ns) > max_age_ns:
            self._views.popleft()
            evicted = True
        return evicted

    def _rebuild_grid_from_views(self) -> None:
        """Recompute ``hits`` and ``seen`` from the retained views, exactly: each ``FusedView`` carries its own aggregated per-voxel counts."""

        self._hits.fill(0)
        self._seen.fill(0)
        if not self._views:
            self._ref_intrinsics = None
            return
        first = self._views[0]
        self._ref_intrinsics = first.intrinsics.copy()
        self._ref_intrinsics.setflags(write=False)
        for view in self._views:
            self._apply_view_counts(
                view.flat_voxel_indices, view.voxel_hit_counts
            )
