"""Tests for suction grasp synthesis (H7.2) on rendered-depth scenes. Pure numpy, no Isaac."""

from __future__ import annotations

import numpy as np

from src.robot.grasping.suction.synthesis import (
    SuctionConfig,
    synthesize_suction_grasps,
)

_W = _H = 160
_FX = _FY = 500.0
_CX = _CY = 80.0
_K = np.array([[_FX, 0.0, _CX], [0.0, _FY, _CY], [0.0, 0.0, 1.0]])


def _render(world_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole-project world surface points (camera at origin, looking down +Z) to a depth map + mask."""
    depth = np.zeros((_H, _W), dtype=np.float64)
    X, Y, Z = world_pts[:, 0], world_pts[:, 1], world_pts[:, 2]
    col = np.round(_CX + _FX * X / Z).astype(int)
    row = np.round(_CY + _FY * Y / Z).astype(int)
    ok = (col >= 0) & (col < _W) & (row >= 0) & (row < _H)
    for r, c, z in zip(row[ok], col[ok], Z[ok]):
        if depth[r, c] == 0.0 or z < depth[r, c]:  # z-buffer: keep the nearest surface
            depth[r, c] = z
    return depth, depth > 0.0


def _flat(z: float = 300.0, half: float = 40.0, x0: float = 0.0, step: float = 0.5) -> np.ndarray:
    g = np.arange(-half, half + step, step)
    gx, gy = np.meshgrid(g + x0, g)
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])


def _cylinder(radius: float = 18.0, z_apex: float = 300.0, half: float = 40.0, x0: float = 0.0,
              step: float = 0.5) -> np.ndarray:
    xs = np.arange(-min(half, radius - 0.5), min(half, radius - 0.5) + step, step)
    ys = np.arange(-half, half + step, step)
    gx, gy = np.meshgrid(xs, ys)
    gz = z_apex + (radius - np.sqrt(np.maximum(radius**2 - gx**2, 0.0)))  # apex nearest, edges farther
    return np.column_stack([(gx + x0).ravel(), gy.ravel(), gz.ravel()])


def test_flat_box_top_yields_a_high_seal_candidate() -> None:
    depth, mask = _render(_flat())
    grasps = synthesize_suction_grasps(mask, depth, _K)
    assert grasps
    best = grasps[0]
    assert best.seal_score > 0.8
    # approach presses INTO the surface (away from the camera at origin -> +Z-ish)
    assert best.approach[2] > 0.5
    # contact near the patch centre (camera-frame XY ~ 0)
    assert abs(best.position_mm[0]) < 12.0 and abs(best.position_mm[1]) < 12.0


def test_picks_flat_object_over_curved_object() -> None:
    flat_depth, flat_mask = _render(_flat())
    cyl_depth, cyl_mask = _render(_cylinder(radius=18.0))
    flat = synthesize_suction_grasps(flat_mask, flat_depth, _K)
    cyl = synthesize_suction_grasps(cyl_mask, cyl_depth, _K)
    assert flat and flat[0].seal_score > 0.8
    # a tight cylinder either yields no sealable candidate or a much worse one
    best_cyl = cyl[0].seal_score if cyl else 0.0
    assert flat[0].quality > best_cyl + 0.4


def test_best_candidate_lands_on_the_flat_region_of_a_mixed_scene() -> None:
    # one mask: a flat slab on the -X side, a cylinder on the +X side -> the best contact must be on -X.
    scene = np.vstack([_flat(x0=-25.0, half=20.0), _cylinder(radius=14.0, x0=25.0, half=20.0)])
    depth, mask = _render(scene)
    grasps = synthesize_suction_grasps(mask, depth, _K)
    assert grasps
    assert grasps[0].position_mm[0] < 0.0  # on the flat (-X) side


def test_payload_keeps_ranking_and_runs() -> None:
    depth, mask = _render(_flat())
    grasps = synthesize_suction_grasps(mask, depth, _K, payload_mass_g=250.0)
    assert grasps
    assert grasps[0].metadata.get("source") == "analytical"
    assert "wrench_resist" in grasps[0].metadata


def test_base_frame_transform() -> None:
    from src.geometry import Frame, Transform

    depth, mask = _render(_flat())
    # a pure +1000 mm Z shift camera->base
    m = np.eye(4)
    m[2, 3] = 1000.0
    tf = Transform.from_matrix(m, from_frame=Frame.CAMERA, to_frame=Frame.BASE)
    grasps = synthesize_suction_grasps(mask, depth, _K, camera_to_base=tf)
    assert grasps
    from src.robot.grasping.types.grasp_point import GraspFrame

    assert grasps[0].frame is GraspFrame.BASE
    assert grasps[0].position_mm[2] > 1000.0  # shifted into base


def test_empty_mask_returns_nothing() -> None:
    assert synthesize_suction_grasps(np.zeros((_H, _W), dtype=bool), np.zeros((_H, _W)), _K) == []


def test_max_results_capped() -> None:
    depth, mask = _render(_flat())
    grasps = synthesize_suction_grasps(mask, depth, _K, config=SuctionConfig(max_results=3))
    assert len(grasps) <= 3
