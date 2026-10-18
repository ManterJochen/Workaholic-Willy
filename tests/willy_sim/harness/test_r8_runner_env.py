"""R8.4 pin: RunnerEnv.from_env parses each WILLY_* knob with its exact former default + parse.

The validated gates run with every knob UNSET, so the all-unset case must reproduce the documented
defaults (byte-identical), and the context-dependent default (ycb_cam_z by vision) must resolve
correctly. The two WILLY_C1_* knobs (the eye-in-hand two-scan refine's tracker and standoff) left on
2026-09-29 with the refine they tuned, and ``view_height_mm``, which only defaulted one of them; the two
WILLY_RL3_* knobs left on 2026-10-16 with the RL dataset runner they shaped (the owner's cleanup), and
WILLY_YCB_DETECTOR on 2026-10-17 with GroundingDINO, the checkpoint it named (the dense path grounds with the
cell's VLM since).
"""

from __future__ import annotations

import pytest

from src.willy_sim.harness.env import RunnerEnv

_KNOBS = [
    "WILLY_YCB_SCENE", "WILLY_YCB_SOLO", "WILLY_YCB_TARGET_Y", "WILLY_YCB_CAM_Z", "WILLY_YCB_PENETRATION",
    "WILLY_YCB_CLOSE", "WILLY_YCB_SQUEEZE", "WILLY_IK_DEBUG", "WILLY_TRACE_MOVES", "WILLY_TRACE_CANDS",
    "WILLY_TRACE_CALC",
]


@pytest.fixture
def _clear_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _KNOBS:
        monkeypatch.delenv(k, raising=False)


def test_all_unset_defaults_match_the_validated_gate(_clear_knobs: None) -> None:
    env = RunnerEnv.from_env(vision=False)
    assert env.ycb_scene == ""
    assert env.ycb_solo is False
    assert env.ycb_target_y is None
    assert env.ycb_cam_z == 1800.0  # vision=False -> GT-mask default
    assert env.ycb_penetration == 12.0
    assert not hasattr(env, "ycb_detector")
    assert env.ycb_close is None
    assert env.ycb_squeeze == 11.0
    assert env.ik_debug is False
    assert not (env.trace_moves or env.trace_cands or env.trace_calc)


def test_context_dependent_defaults(_clear_knobs: None) -> None:
    assert RunnerEnv.from_env(vision=True).ycb_cam_z == 1200.0
    assert RunnerEnv.from_env(vision=False).ycb_cam_z == 1800.0


def test_parsing_truthiness_float_membership(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WILLY_YCB_SOLO", "1")          # truthiness: any non-empty
    monkeypatch.setenv("WILLY_YCB_TARGET_Y", "-60.0")  # float | None
    monkeypatch.setenv("WILLY_YCB_CLOSE", "33.0")      # float | None
    monkeypatch.setenv("WILLY_YCB_CAM_Z", "1400.0")    # explicit overrides the context default
    env = RunnerEnv.from_env(vision=True)
    assert env.ycb_solo is True
    assert env.ycb_target_y == -60.0
    assert env.ycb_close == 33.0
    assert env.ycb_cam_z == 1400.0
