"""Tests for the analytical suction-seal model (H7.2 core). Pure numpy, no Isaac.

These also ARE the H7 Phase-0 discriminance gate: the seal score must SEPARATE a sealable flat patch from
curved / edge / too-small surfaces (the H1.1b "must vary" law) — otherwise the seal score is a sim no-op.
"""

from __future__ import annotations

import numpy as np

from src.robot.grasping.suction.seal import SealConfig, evaluate_seal


def _flat_patch(z: float = 100.0, half_mm: float = 30.0, step: float = 1.5) -> np.ndarray:
    g = np.arange(-half_mm, half_mm + step, step)
    gx, gy = np.meshgrid(g, g)
    zz = np.full(gx.size, z)
    return np.column_stack([gx.ravel(), gy.ravel(), zz])


def _cylinder_top(radius: float, half_mm: float = 30.0, step: float = 1.0) -> np.ndarray:
    """Top surface of a cylinder, axis along Y, apex at z=radius. Curved across X, flat along Y."""
    xs = np.arange(-min(half_mm, radius - 0.5), min(half_mm, radius - 0.5) + step, step)
    ys = np.arange(-half_mm, half_mm + step, step)
    gx, gy = np.meshgrid(xs, ys)
    gz = np.sqrt(np.maximum(radius**2 - gx**2, 0.0))
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _sphere_top(radius: float, half_mm: float = 25.0, step: float = 1.0) -> np.ndarray:
    lim = min(half_mm, radius - 0.5)
    g = np.arange(-lim, lim + step, step)
    gx, gy = np.meshgrid(g, g)
    gz = np.sqrt(np.maximum(radius**2 - gx**2 - gy**2, 0.0))
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _step_edge(z_hi: float = 100.0, z_lo: float = 85.0, half_mm: float = 30.0, step: float = 1.0) -> np.ndarray:
    g = np.arange(-half_mm, half_mm + step, step)
    gx, gy = np.meshgrid(g, g)
    gz = np.where(gx < 0.0, z_hi, z_lo)
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def test_flat_patch_seals() -> None:
    pts = _flat_patch(z=100.0)
    r = evaluate_seal([0.0, 0.0, 100.0], [0.0, 0.0, -1.0], pts)
    assert r.seal_score > 0.95
    assert r.max_deformation_mm < 0.5
    assert r.perimeter_support == 1.0
    assert r.normal_alignment == 1.0


def test_cylinder_breaks_seal() -> None:
    # cup radius 15 on a 20 mm-radius cylinder: across-axis rim deviates ~ r²/2R ≈ 6 mm > 3 mm threshold.
    pts = _cylinder_top(radius=20.0)
    apex = [0.0, 0.0, 20.0]
    flat = evaluate_seal([0.0, 0.0, 100.0], [0.0, 0.0, -1.0], _flat_patch(z=100.0))
    cyl = evaluate_seal(apex, [0.0, 0.0, -1.0], pts)
    assert cyl.max_deformation_mm > 3.0
    assert cyl.seal_score < 0.3
    assert cyl.seal_score < flat.seal_score - 0.5  # decisively separated


def test_sphere_breaks_seal_more_than_cylinder() -> None:
    # a sphere curves in BOTH directions -> every rim vertex deviates -> worse than the cylinder.
    cyl = evaluate_seal([0.0, 0.0, 20.0], [0.0, 0.0, -1.0], _cylinder_top(radius=20.0))
    sph = evaluate_seal([0.0, 0.0, 20.0], [0.0, 0.0, -1.0], _sphere_top(radius=20.0))
    assert sph.seal_score <= cyl.seal_score
    assert sph.seal_score < 0.3


def test_edge_breaks_seal() -> None:
    # a step under the cup: half the rim drops 15 mm -> huge deformation -> no seal.
    r = evaluate_seal([0.0, 0.0, 100.0], [0.0, 0.0, -1.0], _step_edge())
    assert r.max_deformation_mm > 10.0
    assert r.seal_score < 0.2


def test_too_small_patch_loses_perimeter_support() -> None:
    # an object smaller than the cup: the rim finds no surface -> partial/zero support -> no seal.
    small = _flat_patch(z=100.0, half_mm=7.0, step=1.0)  # radius ~7 mm < cup 15 mm
    r = evaluate_seal([0.0, 0.0, 100.0], [0.0, 0.0, -1.0], small)
    assert r.perimeter_support < 1.0
    assert r.seal_score < 0.5


def test_tilted_approach_reduces_alignment() -> None:
    pts = _flat_patch(z=100.0)
    n_surface = [0.0, 0.0, 1.0]
    square = evaluate_seal([0.0, 0.0, 100.0], [0.0, 0.0, -1.0], pts, surface_normal=n_surface)
    # a 45° tilted approach: deformation grows AND alignment drops -> lower seal.
    tilt = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    tilted = evaluate_seal([0.0, 0.0, 100.0], tilt, pts, surface_normal=n_surface)
    assert square.seal_score > tilted.seal_score
    assert tilted.normal_alignment < 0.8


def test_flat_seal_score_monotonic_in_curvature() -> None:
    # bigger radius (flatter) -> higher seal, across a sweep -> the score is a graded signal, not binary.
    scores = [
        evaluate_seal([0.0, 0.0, R], [0.0, 0.0, -1.0], _cylinder_top(radius=R)).seal_score
        for R in (20.0, 40.0, 80.0, 200.0)
    ]
    assert scores == sorted(scores)  # monotonically non-decreasing with flatness
    assert scores[-1] > scores[0]


def test_determinism() -> None:
    pts = _cylinder_top(radius=30.0)
    a = evaluate_seal([0.0, 0.0, 30.0], [0.0, 0.0, -1.0], pts)
    b = evaluate_seal([0.0, 0.0, 30.0], [0.0, 0.0, -1.0], pts)
    assert a.seal_score == b.seal_score
    assert a.max_deformation_mm == b.max_deformation_mm


def test_config_validation() -> None:
    for bad in (
        {"cup_radius_mm": 0.0},
        {"n_ring": 3},
        {"deformation_threshold_mm": 0.0},
        {"rim_support_radius_mm": -1.0},
        {"alignment_floor": 1.0},
    ):
        try:
            SealConfig(**bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
