"""Unit tests for the WorkspaceGuard (box + diversity + pose coercion).

WorkspaceGuard is the Cartesian-box + orientation-diversity gate every calibration/pipeline pose must
pass. It had no dedicated test file; this pins box inclusivity, the diversity "both-axes-close" rule,
the Frame.BASE-only coercion, and the bookkeeping/ctor guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config.schema.robot import WorkspaceLimitsConfig
from src.geometry import Frame, FrameMismatchError, Pose
from src.geometry.quaternion import from_euler
from src.robot.safety.workspace import WorkspaceGuard

_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])  # XYZW identity
_LIMITS = WorkspaceLimitsConfig(x_min=0.0, x_max=100.0, y_min=0.0, y_max=100.0, z_min=0.0, z_max=100.0)


def _pose(x: float, y: float, z: float, quat: np.ndarray | None = None, label: str = "p") -> Pose:
    return Pose(
        position_mm=np.array([x, y, z], dtype=np.float64),
        quaternion_xyzw=(_IDENTITY.copy() if quat is None else quat),
        frame=Frame.BASE,
        label=label,
    )


class TestWorkspaceBox:
    def test_inside(self) -> None:
        assert WorkspaceGuard(_LIMITS).is_inside_workspace(_pose(50, 50, 50))

    def test_on_boundary_is_inclusive(self) -> None:
        g = WorkspaceGuard(_LIMITS)
        assert g.is_inside_workspace(_pose(0, 0, 0))
        assert g.is_inside_workspace(_pose(100, 100, 100))

    def test_outside_each_axis(self) -> None:
        g = WorkspaceGuard(_LIMITS)
        for p in (
            _pose(-1, 50, 50), _pose(101, 50, 50),
            _pose(50, -1, 50), _pose(50, 101, 50),
            _pose(50, 50, -1), _pose(50, 50, 101),
        ):
            assert not g.is_inside_workspace(p)


class TestCoercePose:
    def test_non_base_frame_raises(self) -> None:
        g = WorkspaceGuard(_LIMITS)
        bad = Pose(
            position_mm=np.array([50.0, 50.0, 50.0]),
            quaternion_xyzw=_IDENTITY.copy(),
            frame=Frame.TCP,
            label="tcp",
        )
        with pytest.raises(FrameMismatchError):
            g.is_inside_workspace(bad)

    def test_wrong_type_raises(self) -> None:
        g = WorkspaceGuard(_LIMITS)
        with pytest.raises(TypeError):
            g.is_inside_workspace(object())


class TestDiversity:
    def test_both_axes_close_rejects(self) -> None:
        g = WorkspaceGuard(_LIMITS, min_distance_mm=30.0, min_angle_deg=5.0)
        g.accept(_pose(50, 50, 50, label="a"))
        # 5 mm away (< 30) AND same orientation (0deg < 5) -> both close -> not diverse.
        assert not g.is_diverse_enough(_pose(55, 50, 50, label="b"))

    def test_far_distance_accepts(self) -> None:
        g = WorkspaceGuard(_LIMITS, min_distance_mm=30.0, min_angle_deg=5.0)
        g.accept(_pose(50, 50, 50, label="a"))
        assert g.is_diverse_enough(_pose(90, 50, 50, label="b"))  # 40 mm > 30

    def test_far_angle_accepts_even_when_close(self) -> None:
        g = WorkspaceGuard(_LIMITS, min_distance_mm=30.0, min_angle_deg=5.0)
        g.accept(_pose(50, 50, 50, label="a"))
        rot = from_euler(np.array([0.0, 0.0, np.radians(30.0)]))  # 30deg > 5 -> diverse
        assert g.is_diverse_enough(_pose(52, 50, 50, quat=rot, label="b"))

    def test_self_comparison_skipped(self) -> None:
        g = WorkspaceGuard(_LIMITS)
        p = _pose(50, 50, 50, label="a")
        g.accept(p)
        assert g.is_diverse_enough(p)  # identical pose -> skipped, not "too similar to itself"


class TestValidateAndBookkeeping:
    def test_validate_requires_box_and_diversity(self) -> None:
        g = WorkspaceGuard(_LIMITS, min_distance_mm=30.0, min_angle_deg=5.0)
        assert g.validate(_pose(50, 50, 50, label="a"))
        g.accept(_pose(50, 50, 50, label="a"))
        assert not g.validate(_pose(55, 50, 50, label="b"))   # in box but too similar
        assert not g.validate(_pose(200, 50, 50, label="c"))  # diverse but outside box

    def test_accept_reset_and_counts(self) -> None:
        g = WorkspaceGuard(_LIMITS)
        assert g.num_accepted == 0
        g.accept(_pose(10, 10, 10))
        g.accept(_pose(90, 90, 90))
        assert g.num_accepted == 2
        assert len(g.accepted_poses) == 2
        g.reset()
        assert g.num_accepted == 0


class TestCtorValidation:
    def test_negative_distance_raises(self) -> None:
        with pytest.raises(ValueError):
            WorkspaceGuard(_LIMITS, min_distance_mm=-1.0)

    def test_negative_angle_raises(self) -> None:
        with pytest.raises(ValueError):
            WorkspaceGuard(_LIMITS, min_angle_deg=-1.0)
