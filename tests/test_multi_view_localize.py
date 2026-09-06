"""H5.1 multi-view target localization fusion — unit tests (pure, no Isaac)."""

from __future__ import annotations

import numpy as np
import pytest

from src.robot.grasping.multiview.localize import (
    ViewLocalization,
    fuse_scene_points_base,
    fuse_view_localizations,
)


def _v(name: str, xyz: tuple[float, float, float] | None, px: int) -> ViewLocalization:
    return ViewLocalization(name=name, centroid_base_mm=None if xyz is None else np.asarray(xyz, dtype=np.float64), visible_px=px)


def test_no_view_saw_target_returns_none() -> None:
    """All views occluded (None / 0 px) -> cannot localize -> None."""
    assert fuse_view_localizations([_v("overhead", None, 0), _v("oblique_L", (1, 2, 3), 0)]) is None
    assert fuse_view_localizations([]) is None


def test_single_view_passthrough() -> None:
    """One contributing view -> the fused point is exactly that view's centroid."""
    out = fuse_view_localizations([_v("oblique_L", (450.0, 5.0, 25.0), 2801)])
    assert out is not None
    np.testing.assert_allclose(out, [450.0, 5.0, 25.0])


def test_occluded_view_is_ignored_others_carry() -> None:
    """The redundancy property: an occluded view (overhead, arm-blocked) contributes nothing; the
    obliques carry the localization — the measured run_multiview_pick eth3 behaviour."""
    out = fuse_view_localizations([
        _v("overhead", None, 0),                  # arm-blocked -> ignored
        _v("oblique_L", (450.0, 5.0, 25.0), 2801),
        _v("oblique_R", (450.0, 5.0, 25.0), 2801),
    ])
    assert out is not None
    np.testing.assert_allclose(out, [450.0, 5.0, 25.0])


def test_visibility_weighting_biases_toward_clearer_view() -> None:
    """The fused point is a visible-px-weighted mean: the view that sees more of the target pulls harder."""
    out = fuse_view_localizations([
        _v("a", (0.0, 0.0, 0.0), 1000),
        _v("b", (100.0, 0.0, 0.0), 3000),
    ])
    assert out is not None
    # weighted mean: (1000*0 + 3000*100) / 4000 = 75
    np.testing.assert_allclose(out, [75.0, 0.0, 0.0])


def test_zero_px_with_centroid_is_not_counted() -> None:
    """visible_px == 0 means not-seen even if a stale centroid is present -> ignored."""
    assert not _v("x", (1, 2, 3), 0).saw_target
    out = fuse_view_localizations([_v("x", (1, 2, 3), 0), _v("y", (10, 20, 30), 5)])
    assert out is not None
    np.testing.assert_allclose(out, [10.0, 20.0, 30.0])


def test_deterministic_for_fixed_input() -> None:
    views = [_v("a", (1.0, 2.0, 3.0), 10), _v("b", (4.0, 5.0, 6.0), 20)]
    first = fuse_view_localizations(views)
    second = fuse_view_localizations(views)
    assert first is not None and second is not None
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("px", [0, -1])
def test_non_positive_px_never_contributes(px: int) -> None:
    assert fuse_view_localizations([_v("a", (1, 2, 3), px)]) is None


# --- Lever B: fused scene point cloud --------------------------------------------------------------

def test_fuse_scene_points_concatenates_in_order() -> None:
    a = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    b = np.array([[2.0, 2.0, 2.0]])
    out = fuse_scene_points_base([a, None, b])
    assert out is not None
    np.testing.assert_array_equal(out, [[0, 0, 0], [1, 1, 1], [2, 2, 2]])


def test_fuse_scene_points_all_empty_returns_none() -> None:
    assert fuse_scene_points_base([]) is None
    assert fuse_scene_points_base([None, np.empty((0, 3))]) is None


def test_fuse_scene_points_reshapes_flat_input() -> None:
    out = fuse_scene_points_base([np.array([1.0, 2.0, 3.0])])  # a single flat xyz
    assert out is not None
    assert out.shape == (1, 3)
    np.testing.assert_array_equal(out, [[1, 2, 3]])
