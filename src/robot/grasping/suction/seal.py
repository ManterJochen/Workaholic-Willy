"""Analytical suction-seal model.

The built-in Isaac Surface-Gripper is a binary proximity joint and carries no seal-quality signal, so the
seal score is computed here from the rendered-depth point cloud: the deformable-cup seal formulation of
the published literature, re-implemented as a pure-numpy module straight from the equations. No code,
dataset or checkpoint from those projects is used, and the papers are cited in notice under "Prior art",
which is the one place a third-party name belongs.

The model is of a compliant suction cup. Given a candidate contact point ``p`` in BASE millimetres and an
approach direction ``n``, the cup presses along ``n`` into the surface and its rim lies in the plane
through ``p`` perpendicular to ``n``:

* place ``n_ring`` rim vertices on a circle of radius ``cup_radius_mm`` in that plane;
* for each rim vertex find the nearest surface point, matched by its position in the cup plane, and
  measure its signed distance out of the plane along ``n``. That distance is the deformation the elastic
  rim has to absorb to seal there. A flat patch deforms the rim by almost nothing, a curved patch deforms
  it in proportion to its curvature, and an edge or a hole leaves some rim vertices with no surface
  support at all, which is a leak path;
* the seal score, in [0, 1], is the product of rim flatness, taken as the maximum deformation against
  ``deformation_threshold_mm``, rim perimeter support, the fraction of rim vertices that found surface,
  and an optional alignment term for how squarely the approach faces the local surface normal.

The model is depth geometry alone, so no arm kinematics confound it and it can be validated in
simulation. Final real-vacuum fidelity, meaning air leak, cup material and line pressure, stays a
real-hardware concern.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SealConfig", "SealResult", "evaluate_seal"]


@dataclass(frozen=True, slots=True)
class SealConfig:
    """Suction-cup seal-model parameters in millimetres; the defaults describe a 30 mm cup with a 3 mm compliance budget."""

    cup_radius_mm: float = 15.0
    n_ring: int = 16
    # The published models treat the cup as a deformable disk rather than a bare rim, so n_rings concentric
    # rings are sampled. The outer ring at cup_radius is the leak-critical seal perimeter and the inner
    # rings capture how the cup face conforms. Deformation is the maximum over all ring vertices, while
    # perimeter support is measured on the outer ring alone.
    n_rings: int = 3
    deformation_threshold_mm: float = 3.0
    rim_support_radius_mm: float = 6.0
    # A local surface normal whose absolute cosine with the approach falls below this floor earns no
    # alignment credit, because a steeply tilted approach cannot seal. 0.5 is about 60 degrees. The floor
    # applies only when a surface normal is supplied.
    alignment_floor: float = 0.5

    def __post_init__(self) -> None:
        if self.cup_radius_mm <= 0.0:
            raise ValueError(f"cup_radius_mm must be > 0, got {self.cup_radius_mm}")
        if self.n_ring < 4:
            raise ValueError(f"n_ring must be >= 4, got {self.n_ring}")
        if self.n_rings < 1:
            raise ValueError(f"n_rings must be >= 1, got {self.n_rings}")
        if self.deformation_threshold_mm <= 0.0:
            raise ValueError(f"deformation_threshold_mm must be > 0, got {self.deformation_threshold_mm}")
        if self.rim_support_radius_mm <= 0.0:
            raise ValueError(f"rim_support_radius_mm must be > 0, got {self.rim_support_radius_mm}")
        if not 0.0 <= self.alignment_floor < 1.0:
            raise ValueError(f"alignment_floor must be in [0, 1), got {self.alignment_floor}")


@dataclass(frozen=True, slots=True)
class SealResult:
    """Analytical seal evaluation at one contact, where ``seal_score`` is the product of flatness, perimeter support and alignment."""

    seal_score: float
    max_deformation_mm: float
    rms_deformation_mm: float
    perimeter_support: float
    normal_alignment: float
    n_ring: int


def _orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to ``n``, right-handed and deterministic."""
    n = n / float(np.linalg.norm(n))
    # Pick the world axis least aligned with n as the seed so the cross product is well-conditioned.
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed - n * float(np.dot(seed, n))
    u = u / float(np.linalg.norm(u))
    v = np.cross(n, u)
    return u, v


def evaluate_seal(
    contact_point_mm: np.ndarray,
    approach_normal: np.ndarray,
    points_mm: np.ndarray,
    *,
    surface_normal: np.ndarray | None = None,
    config: SealConfig | None = None,
) -> SealResult:
    """Evaluate the analytical suction seal at ``contact_point_mm``, approaching along ``approach_normal`` over the surface cloud ``points_mm``; an optional ``surface_normal`` gates the score by cup against surface alignment."""
    cfg = config or SealConfig()
    p = np.asarray(contact_point_mm, dtype=np.float64).reshape(3)
    n = np.asarray(approach_normal, dtype=np.float64).reshape(3)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-9:
        raise ValueError("approach_normal cannot be the zero vector")
    n = n / n_norm
    pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]

    u, v = _orthonormal_basis(n)
    # In-plane (u, v) and out-of-plane (n) coordinates of every surface point, relative to the contact.
    rel = pts - p
    pu = rel @ u
    pv = rel @ v
    pn = rel @ n  # signed distance out of the cup plane (the deformation a rim point would have to absorb)

    angles = np.linspace(0.0, 2.0 * np.pi, cfg.n_ring, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    # Concentric ring radii from the outer (seal perimeter) inward; the outer ring (index 0) is the leak path.
    radii = np.linspace(cfg.cup_radius_mm, cfg.cup_radius_mm / cfg.n_rings, cfg.n_rings)

    deformations: list[float] = []
    perimeter_supported = 0
    r2 = cfg.rim_support_radius_mm ** 2
    for ring_idx, radius in enumerate(radii):
        ring_u = radius * cos_a
        ring_v = radius * sin_a
        for ru, rv in zip(ring_u, ring_v):
            d2 = (pu - ru) ** 2 + (pv - rv) ** 2  # in-plane distance^2 from this ring vertex to each surface point
            j = int(np.argmin(d2)) if d2.size else -1
            if j < 0 or d2[j] > r2:
                continue  # no surface within the support radius, so this ring vertex finds no contact
            if ring_idx == 0:
                perimeter_supported += 1  # the outer ring is the seal perimeter (leak-critical)
            deformations.append(abs(float(pn[j])))

    perimeter_support = perimeter_supported / float(cfg.n_ring)
    if not deformations:
        return SealResult(0.0, float("inf"), float("inf"), 0.0, 1.0, cfg.n_ring)
    defo = np.asarray(deformations, dtype=np.float64)
    max_def = float(defo.max())
    rms_def = float(np.sqrt(np.mean(defo ** 2)))

    flatness_term = float(np.clip(1.0 - max_def / cfg.deformation_threshold_mm, 0.0, 1.0))

    if surface_normal is not None:
        sn = np.asarray(surface_normal, dtype=np.float64).reshape(3)
        sn_norm = float(np.linalg.norm(sn))
        alignment = abs(float(np.dot(n, sn / sn_norm))) if sn_norm > 1e-9 else 1.0
    else:
        alignment = 1.0
    align_term = float(
        np.clip((alignment - cfg.alignment_floor) / (1.0 - cfg.alignment_floor), 0.0, 1.0)
    )

    seal_score = flatness_term * perimeter_support * align_term
    return SealResult(
        seal_score=seal_score,
        max_deformation_mm=max_def,
        rms_deformation_mm=rms_def,
        perimeter_support=perimeter_support,
        normal_alignment=alignment,
        n_ring=cfg.n_ring,
    )
