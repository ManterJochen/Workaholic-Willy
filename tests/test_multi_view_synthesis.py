"""Tests for the H5.1 Lever C 3D point-cloud grasp synthesis (pure numpy, no Isaac)."""

from __future__ import annotations

import numpy as np

from src.robot.grasping.types.grasp_point import GraspFrame
from src.robot.grasping.multiview.synthesis import (
    fused_grasp_to_point,
    synthesize_grasp_from_cloud,
)


def _box_cloud(cx: float, cy: float, *, lx: float, ly: float, top_z: float, n: int = 12) -> np.ndarray:
    """A filled-box surface cloud: an n x n grid on the top face at z=top_z, centred at (cx, cy)."""
    xs = np.linspace(cx - lx / 2, cx + lx / 2, n)
    ys = np.linspace(cy - ly / 2, cy + ly / 2, n)
    gx, gy = np.meshgrid(xs, ys)
    z = np.full(gx.size, top_z)
    return np.column_stack([gx.ravel(), gy.ravel(), z])


def test_cube_grasp_centroid_and_topdown() -> None:
    cloud = _box_cloud(450.0, 0.0, lx=30.0, ly=30.0, top_z=40.0)
    g = synthesize_grasp_from_cloud(cloud, grasp_depth_mm=8.0, width_margin_mm=6.0)
    assert g is not None
    # centroid in XY, grasp depth below the top surface in Z
    np.testing.assert_allclose(g.position_mm[0], 450.0, atol=1e-9)
    np.testing.assert_allclose(g.position_mm[1], 0.0, atol=1e-9)
    np.testing.assert_allclose(g.position_mm[2], 40.0 - 8.0, atol=1e-9)
    # top-down approach
    np.testing.assert_allclose(g.approach, [0.0, 0.0, -1.0])
    # closing axis horizontal (z == 0) and unit
    assert abs(g.closing_axis[2]) < 1e-9
    np.testing.assert_allclose(np.linalg.norm(g.closing_axis), 1.0, atol=1e-9)
    # width ~ 30 + margin
    assert 35.0 <= g.width_mm <= 37.0


def test_elongated_box_closes_across_minor_axis() -> None:
    # 60 mm (x) x 30 mm (y): the gripper must close across the NARROW (y) axis -> closing axis ~ +/- Y.
    cloud = _box_cloud(450.0, 0.0, lx=60.0, ly=30.0, top_z=40.0)
    g = synthesize_grasp_from_cloud(cloud, width_margin_mm=0.0)
    assert g is not None
    # closing axis is (anti)parallel to base Y
    assert abs(abs(g.closing_axis[1]) - 1.0) < 1e-6
    assert abs(g.closing_axis[0]) < 1e-6
    # width = the minor (30 mm) extent, NOT the 60 mm major
    assert 29.0 <= g.width_mm <= 31.0
    minor_extent, major_extent = g.footprint_mm
    assert 29.0 <= minor_extent <= 31.0
    assert 59.0 <= major_extent <= 61.0


def test_rotated_elongated_box_tracks_orientation() -> None:
    # Build a 60x30 box then rotate it 30 deg about Z; the closing axis must rotate with it.
    base = _box_cloud(0.0, 0.0, lx=60.0, ly=30.0, top_z=40.0)
    theta = np.deg2rad(30.0)
    c, s = np.cos(theta), np.sin(theta)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rot = (rz @ base.T).T + np.array([450.0, 0.0, 0.0])
    g = synthesize_grasp_from_cloud(rot, width_margin_mm=0.0)
    assert g is not None
    # the minor axis of a 60(x)x30(y) box is +/-Y; rotated 30 deg -> rz @ [0,1,0] = [-sin, cos, 0]
    expected = np.array([-s, c, 0.0])
    dot = abs(float(np.dot(g.closing_axis, expected)))
    assert dot > 0.99  # parallel up to sign
    assert 29.0 <= g.width_mm <= 31.0


def test_width_clipped_to_gripper_range() -> None:
    cloud = _box_cloud(450.0, 0.0, lx=200.0, ly=200.0, top_z=40.0)
    g = synthesize_grasp_from_cloud(cloud, min_width_mm=0.0, max_width_mm=85.0, width_margin_mm=6.0)
    assert g is not None
    assert g.width_mm == 85.0  # clipped to the gripper max


def test_too_few_points_returns_none() -> None:
    assert synthesize_grasp_from_cloud(np.zeros((4, 3))) is None
    assert synthesize_grasp_from_cloud(np.empty((0, 3))) is None


def test_non_finite_points_filtered() -> None:
    cloud = _box_cloud(450.0, 0.0, lx=30.0, ly=30.0, top_z=40.0)
    poisoned = np.vstack([cloud, np.full((3, 3), np.nan), np.full((2, 3), np.inf)])
    g = synthesize_grasp_from_cloud(poisoned)
    assert g is not None
    assert g.n_points == cloud.shape[0]  # the non-finite rows were dropped


def test_fused_centroid_beats_single_biased_view() -> None:
    # Each oblique sees only ONE side face -> its cloud is biased toward that side; the fused (both) cloud
    # cancels the bias. Model: a left-biased partial + a right-biased partial of a cube centred at y=0.
    left = _box_cloud(450.0, -8.0, lx=14.0, ly=14.0, top_z=40.0)   # only the -Y side seen
    right = _box_cloud(450.0, +8.0, lx=14.0, ly=14.0, top_z=40.0)  # only the +Y side seen
    g_left = synthesize_grasp_from_cloud(left)
    g_right = synthesize_grasp_from_cloud(right)
    g_fused = synthesize_grasp_from_cloud(np.vstack([left, right]))
    assert g_left is not None and g_right is not None and g_fused is not None
    true_y = 0.0
    err_single = min(abs(g_left.position_mm[1] - true_y), abs(g_right.position_mm[1] - true_y))
    err_fused = abs(g_fused.position_mm[1] - true_y)
    assert err_fused < err_single  # fusing the two biased views recovers the true centroid


def test_determinism() -> None:
    cloud = _box_cloud(450.0, 0.0, lx=30.0, ly=30.0, top_z=40.0)
    a = synthesize_grasp_from_cloud(cloud)
    b = synthesize_grasp_from_cloud(cloud)
    assert a is not None and b is not None
    np.testing.assert_array_equal(a.position_mm, b.position_mm)
    np.testing.assert_array_equal(a.closing_axis, b.closing_axis)
    assert a.width_mm == b.width_mm


def test_fused_grasp_to_point_is_base_frame() -> None:
    cloud = _box_cloud(450.0, 0.0, lx=30.0, ly=30.0, top_z=40.0)
    g = synthesize_grasp_from_cloud(cloud)
    assert g is not None
    gp = fused_grasp_to_point(g, score=0.9, label="t")
    assert gp.frame is GraspFrame.BASE
    np.testing.assert_allclose(gp.position, g.position_mm)
    np.testing.assert_allclose(gp.approach, [0.0, 0.0, -1.0])
    assert gp.grip_width_mm == g.width_mm
    assert gp.score == 0.9
