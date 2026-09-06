"""Gap G5 Stufe B (2/2) — build_config_viewpoint_planner: three-flag-gated config auto-build.

Edit #1 (committed) made the U6 commit-gate reobserve RELOCATE the camera WHEN a viewpoint_planner is
wired; this pins the config auto-build that actually wires one in production. The planner is built ONLY
when all three of fusion.enabled + commit_policy.enabled + active_perception_use_fusion are True (the
three-flag gate finally makes the dead active_perception_use_fusion switch live); default-off is
byte-identical (None). Budget is tied to max_reobserve_attempts; the safety_check is a
WorkspaceBoxSafetyCheck from workspace_limits (fail-closed backstop), with a default fallback.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.robot.execution.autonomous_grasp.builders import build_config_viewpoint_planner
from src.robot.grasping.closed_loop.active_perception import (
    ScoringViewpointPlanner,
    WorkspaceBoxSafetyCheck,
)


def _wl() -> SimpleNamespace:
    return SimpleNamespace(x_min=-500.0, x_max=500.0, y_min=-400.0, y_max=400.0, z_min=100.0, z_max=600.0)


def _robot_cfg(wl: SimpleNamespace | None = None) -> SimpleNamespace:
    return SimpleNamespace(workspace_limits=wl)


def _grasping_cfg(*, fusion: bool, active: bool, commit: bool, max_reobserve: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        fusion=SimpleNamespace(
            enabled=fusion,
            active_perception_use_fusion=active,
            commit_policy=SimpleNamespace(enabled=commit, max_reobserve_attempts=max_reobserve),
        )
    )


class TestBuildConfigViewpointPlanner:
    def test_none_grasping_cfg(self) -> None:
        assert build_config_viewpoint_planner(None, _robot_cfg(_wl())) is None

    def test_default_off(self) -> None:
        cfg = _grasping_cfg(fusion=False, active=False, commit=False)
        assert build_config_viewpoint_planner(cfg, _robot_cfg(_wl())) is None

    def test_partial_off_fusion_disabled(self) -> None:
        cfg = _grasping_cfg(fusion=False, active=True, commit=True)
        assert build_config_viewpoint_planner(cfg, _robot_cfg(_wl())) is None

    def test_partial_off_active_perception_off(self) -> None:
        cfg = _grasping_cfg(fusion=True, active=False, commit=True)
        assert build_config_viewpoint_planner(cfg, _robot_cfg(_wl())) is None

    def test_partial_off_commit_disabled(self) -> None:
        cfg = _grasping_cfg(fusion=True, active=True, commit=False)
        assert build_config_viewpoint_planner(cfg, _robot_cfg(_wl())) is None

    def test_all_three_on_builds_planner(self) -> None:
        cfg = _grasping_cfg(fusion=True, active=True, commit=True, max_reobserve=3)
        planner = build_config_viewpoint_planner(cfg, _robot_cfg(_wl()))
        assert isinstance(planner, ScoringViewpointPlanner)
        assert planner.policy.max_viewpoints == 3  # tied to max_reobserve_attempts
        assert isinstance(planner.safety_check, WorkspaceBoxSafetyCheck)
        # center = (min+max)/2, half = (max-min)/2 per axis.
        assert tuple(float(x) for x in planner.safety_check.center_mm) == (0.0, 0.0, 350.0)
        assert tuple(float(x) for x in planner.safety_check.half_extents_mm) == (500.0, 400.0, 250.0)

    def test_all_on_no_workspace_falls_back_to_default_safety(self) -> None:
        cfg = _grasping_cfg(fusion=True, active=True, commit=True)
        planner = build_config_viewpoint_planner(cfg, _robot_cfg(None))
        assert isinstance(planner, ScoringViewpointPlanner)
        assert not isinstance(planner.safety_check, WorkspaceBoxSafetyCheck)
