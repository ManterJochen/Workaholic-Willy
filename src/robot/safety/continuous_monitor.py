"""The per-control-step software collision-avoidance monitor.

This is a software collision-avoidance layer and not the safety system. It reduces
collision risk in simulation. The real-hardware safety guarantee is independent,
certified functional safety, meaning hardware E-stop circuits and ISO 10218,
ISO/TS 15066 and ISO 13849, and it runs independently of this software. A green sim
result is not a commercial safety claim.

What it does is run the exact-mesh backend, Coal first, over each control-step
configuration of a planned move: arm against itself and arm against fixtures, at every
interpolation waypoint rather than a sample of them, with a distance margin that stops
at a configurable clearance before contact and a fail-safe of its own. A single check
that overruns its budget, or that cannot run, is a stop or a hold, never a
continuation without a result.

It is opt-in behind :class:`ContinuousGuardProfile`, and the default ``enabled=False``
leaves behaviour byte-identical. The margin has a ceiling: the natural grasp puts its
own ``wrist_1`` and ``wrist_3`` pair at about 19.6 mm, so a margin above that
false-stops a good pick, and the default sits well below it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from ._fcl_self_collision import make_backend
from ._ur_kinematics import ur_link_transforms_mm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._fcl_self_collision import MeshSelfCollisionBackend


class ContinuousGuardAbort(RuntimeError):
    """Raised mid-motion when the continuous guard stops a move, on margin or on fail-safe.

    The motion halts at the last cleared waypoint and never continues past a stop. It
    carries the :class:`MonitorVerdict`, so the caller logs the pair, the distance and
    the reason for the abort.
    """

    def __init__(self, verdict: MonitorVerdict) -> None:
        self.verdict = verdict
        super().__init__(
            f"continuous guard stop: {verdict.status}"
            + (f" {verdict.pair} {verdict.distance_mm:.3f}mm" if verdict.pair else "")
        )


class MonitorStatus(StrEnum):
    """A check is OK only where it ran in budget and every pair stayed clear of the margin."""

    OK = "ok"
    COLLISION = "collision"      # a monitored pair fell below the margin, so stop
    TIMEOUT = "timeout"          # the check overran max_check_ms, so stop: it cannot keep up
    UNAVAILABLE = "unavailable"  # kinematics or backend produced no result, so stop, fail-closed


@dataclass(frozen=True, slots=True)
class ContinuousGuardProfile:
    """The named safety profile, opt-in. The default ``enabled=False`` installs no guard."""

    enabled: bool = False
    margin_mm: float = 8.0      # Stop at this clearance before contact. Keep it well under the
    #                             19.6 mm natural-grasp wrist clearance, or a good pick false-stops.
    max_check_ms: float = 12.0  # Fail-safe budget: a check slower than this stops, because it cannot
    #                             keep up at the control rate. 12 ms is roughly a 60 to 80 Hz monitor.


@dataclass(frozen=True, slots=True)
class MonitorVerdict:
    """The outcome of one per-control-step check. ``stop`` is True for anything but OK."""

    status: MonitorStatus
    elapsed_ms: float
    pair: str | None = None
    distance_mm: float | None = None

    @property
    def stop(self) -> bool:
        return self.status is not MonitorStatus.OK


class ContinuousCollisionMonitor:
    """Per-step mesh collision avoidance with a distance margin and a fail-closed fail-safe."""

    def __init__(
        self,
        backend: MeshSelfCollisionBackend,
        model: str,
        yaw_deg: float,
        fixtures: Sequence[object],
        profile: ContinuousGuardProfile,
    ) -> None:
        self._backend = backend
        self._model = model
        self._yaw = float(yaw_deg)
        self._fixtures = tuple(fixtures)
        self._profile = profile
        self.engine = getattr(backend, "engine", "fcl")

    @classmethod
    def from_model(
        cls,
        model: str,
        yaw_deg: float,
        fixtures: Sequence[object],
        profile: ContinuousGuardProfile,
        mesh_dir: str | None = None,
        variant: str | None = None,
    ) -> ContinuousCollisionMonitor | None:
        """Build the monitor, or ``None`` where the mesh backend has no engine or no bundle.

        ``None`` means the guard cannot be installed, and the caller decides what to do.
        A caller whose profile asked for the guard warns rather than running unguarded
        in silence.

        ``model``, ``yaw_deg`` and ``variant`` are the ones the one-shot guard uses. A
        monitor built on different geometry from the guard beside it is two opinions
        about the same arm.
        """
        # The variant matters as much as the model. Without it, a cell running a
        # mounted gripper has the one-shot guard on that gripper meshes and the
        # in-motion monitor on the baked 2F-85 meshes, so the two disagree about the
        # shape of the thing at the end of the arm.
        backend = make_backend(model, mesh_dir, variant)
        if backend is None:
            return None
        return cls(backend, model, yaw_deg, fixtures, profile)

    @property
    def profile(self) -> ContinuousGuardProfile:
        return self._profile

    def check(self, joints_rad: Sequence[float] | np.ndarray) -> MonitorVerdict:
        """Check one configuration. It is fail-closed: no clear result means stop."""
        t0 = time.perf_counter()
        try:
            transforms = ur_link_transforms_mm(self._model, np.asarray(joints_rad, dtype=np.float64))
            if transforms is None:
                return MonitorVerdict(MonitorStatus.UNAVAILABLE, (time.perf_counter() - t0) * 1000.0)
            hit = self._backend.evaluate(
                transforms, self._yaw, self._fixtures, self._profile.margin_mm, broadphase=True
            )
        except Exception:  # noqa: BLE001 (fail-closed: any backend error is a STOP, never a pass-through)
            return MonitorVerdict(MonitorStatus.UNAVAILABLE, (time.perf_counter() - t0) * 1000.0)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if hit is not None:
            pair, dmm = hit
            return MonitorVerdict(MonitorStatus.COLLISION, elapsed, pair, dmm)
        if elapsed > self._profile.max_check_ms:
            # The check completed, too slowly to trust at the control rate, so the
            # fail-safe stops or holds.
            return MonitorVerdict(MonitorStatus.TIMEOUT, elapsed)
        return MonitorVerdict(MonitorStatus.OK, elapsed)
