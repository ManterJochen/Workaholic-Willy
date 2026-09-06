"""R8.4 pin: RunnerEnv.from_env parses each WILLY_* knob with its exact former default + parse.

The validated gates run with every knob UNSET, so the all-unset case must reproduce the documented
defaults (byte-identical), and the two context-dependent defaults (ycb_cam_z by vision; c1_standoff
by view_height_mm) must resolve correctly.
"""

from __future__ import annotations

import pytest

from src.willy_sim.harness.env import RunnerEnv

_KNOBS = [
    "WILLY_YCB_SCENE", "WILLY_YCB_SOLO", "WILLY_YCB_TARGET_Y", "WILLY_RL3_SUBSTRATE",
    "WILLY_RL3_BIG_MM", "WILLY_YCB_CAM_Z", "WILLY_YCB_PENETRATION", "WILLY_YCB_DETECTOR",
    "WILLY_YCB_CLOSE", "WILLY_YCB_SQUEEZE", "WILLY_IK_DEBUG", "WILLY_C1_WORLD_SPACE_TRACKER",
    "WILLY_C1_STANDOFF_MM", "WILLY_TRACE_MOVES", "WILLY_TRACE_CANDS", "WILLY_TRACE_CALC",
    "WILLY_TRACE_G12",
]


@pytest.fixture
def _clear_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _KNOBS:
        monkeypatch.delenv(k, raising=False)


def test_all_unset_defaults_match_the_validated_gate(_clear_knobs: None) -> None:
    env = RunnerEnv.from_env(vision=False, view_height_mm=275.0)
    assert env.ycb_scene == ""
    assert env.ycb_solo is False
    assert env.ycb_target_y is None
    assert env.rl3_substrate == ""
    assert env.rl3_big_mm == 120.0
    assert env.ycb_cam_z == 1800.0  # vision=False -> GT-mask default
    assert env.ycb_penetration == 12.0
    assert env.ycb_detector == "IDEA-Research/grounding-dino-base"
    assert env.ycb_close is None
    assert env.ycb_squeeze == 11.0
    assert env.ik_debug is False
    assert env.c1_world_space_tracker is False
    assert env.c1_standoff_mm == 275.0  # defaults to view_height_mm
    assert not (env.trace_moves or env.trace_cands or env.trace_calc or env.trace_g12)


def test_context_dependent_defaults(_clear_knobs: None) -> None:
    assert RunnerEnv.from_env(vision=True, view_height_mm=275.0).ycb_cam_z == 1200.0
    assert RunnerEnv.from_env(vision=False, view_height_mm=275.0).ycb_cam_z == 1800.0
    assert RunnerEnv.from_env(vision=False, view_height_mm=300.0).c1_standoff_mm == 300.0


def test_parsing_truthiness_float_membership(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WILLY_YCB_SOLO", "1")          # truthiness: any non-empty
    monkeypatch.setenv("WILLY_YCB_TARGET_Y", "-60.0")  # float | None
    monkeypatch.setenv("WILLY_YCB_CLOSE", "33.0")      # float | None
    monkeypatch.setenv("WILLY_YCB_CAM_Z", "1400.0")    # explicit overrides the context default
    monkeypatch.setenv("WILLY_C1_WORLD_SPACE_TRACKER", "Yes")  # membership, case-insensitive
    monkeypatch.setenv("WILLY_RL3_SUBSTRATE", "  Both ")        # stripped + lowered
    env = RunnerEnv.from_env(vision=True, view_height_mm=275.0)
    assert env.ycb_solo is True
    assert env.ycb_target_y == -60.0
    assert env.ycb_close == 33.0
    assert env.ycb_cam_z == 1400.0
    assert env.c1_world_space_tracker is True
    assert env.rl3_substrate == "both"
