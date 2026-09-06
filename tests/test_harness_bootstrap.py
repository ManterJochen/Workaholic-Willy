"""Off-box contract pins for the willy_sim sim-cell bootstrap + the robot-narrowing convention.

These run OFF-BOX (no Isaac): they lock the parts the on-box pick gates cannot see —

1. the lazy-import contract of the bootstrap (it must import on macOS / CI with no ``isaacsim``);
2. the exact behaviour of ``require_robot``, whose only observable effect is the ``ValueError`` it
   raises when ``cfg.robot`` is absent (the single behaviour-touching point of the sim-config narrowing);
3. the planning-environment announcement — the cuRobo + Coal anchoring banner printed at boot — which is
   pure-python and testable without Isaac (a ``"curobo"`` run with a missing cuRobo env must be flagged as
   silently degrading to blind IK).

The bootstrap's Isaac path itself is proven by the on-box known-pose / eye-in-hand / dense pick gates.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.willy_sim import config as sim_config
from src.willy_sim.harness.bootstrap import (
    SimCell,
    announce_planning_environment,
    bootstrap_sim_cell,
)


def test_bootstrap_module_imports_without_isaac() -> None:
    """The shared bootstrap must be import-safe off-box (only lazy isaacsim inside the function)."""
    assert SimCell.__name__ == "SimCell"
    assert callable(bootstrap_sim_cell)
    # frozen value object with the documented field set
    assert SimCell.__dataclass_params__.frozen is True
    assert set(SimCell.__dataclass_fields__) == {
        "arm", "gripper", "handles", "cfg", "robot", "sim", "dwell",
        # Which engines the cell routes through but does NOT have installed. Empty on a normal run;
        # non-empty only when the operator explicitly opted into a degraded one, so a runner can stamp
        # it onto its results rather than letting those numbers pass as the configured system's.
        "degraded_engines",
    }


def test_require_robot_returns_present_block() -> None:
    """On the real sim tree the robot block is present, so require_robot returns it unchanged."""
    cfg = sim_config.load_sim_config()
    assert cfg.robot is not None
    assert sim_config.require_robot(cfg) is cfg.robot


def test_require_robot_raises_documented_valueerror_when_absent() -> None:
    """The narrowing raises the SAME ValueError the sim driver/safety builders already raise."""
    cfg_without_robot = SimpleNamespace(robot=None)
    with pytest.raises(ValueError, match="sim config has no robot block"):
        sim_config.require_robot(cfg_without_robot)  # type: ignore[arg-type]


def test_sim_driver_config_and_safety_delegate_to_require_robot() -> None:
    """sim_driver_config / sim_safety_preflight now route through require_robot — same raise."""
    cfg_without_robot = SimpleNamespace(robot=None)
    with pytest.raises(ValueError, match="sim config has no robot block"):
        sim_config.sim_driver_config(cfg_without_robot)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sim config has no robot block"):
        sim_config.sim_safety_preflight(cfg_without_robot)  # type: ignore[arg-type]


def test_announce_curobo_run_flags_missing_env_as_degraded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ``"curobo"`` run with an absent cuRobo env is announced as SILENTLY degrading to blind IK.

    This is the exact trap that made a two-side-ETH pick look like a grasp-quality failure: the arm fell
    back to blind IK, self-collided on the reach, and was (correctly) safety-rejected. The banner turns that
    into an obvious, up-front warning instead of a line buried in the Isaac boot log.
    """
    from src.robot.safety.planning.environment import ENV_CUROBO_PYTHON

    missing = tmp_path / "no_curobo_env" / "python.exe"  # parent does not exist -> env is absent
    monkeypatch.setenv(ENV_CUROBO_PYTHON, str(missing))

    text = announce_planning_environment("curobo")

    assert "cuRobo planner: MISSING" in text
    assert "BLIND IK" in text  # the loud degradation warning fired
    assert ENV_CUROBO_PYTHON in text  # tells the operator exactly which knob to set
    assert text == capsys.readouterr().out.rstrip("\n")  # the returned text is what was printed


def test_announce_non_curobo_planner_never_warns_about_blind_ik(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-curobo planner (e.g. plain IK) is the intended path, so it never triggers the degrade warning."""
    from src.robot.safety.planning.environment import ENV_CUROBO_PYTHON

    monkeypatch.setenv(ENV_CUROBO_PYTHON, str(tmp_path / "still_missing"))

    text = announce_planning_environment("ik")

    assert "BLIND IK" not in text  # not a degradation — "ik" is exactly what was asked for
    assert "cuRobo planner:" in text  # the anchoring status is still reported for the record


def test_announce_curobo_present_reports_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the cuRobo env python EXISTS, a curobo run is announced as available (no blind-IK warning)."""
    from src.robot.safety.planning.environment import ENV_CUROBO_PYTHON

    present = tmp_path / "python.exe"
    present.write_text("", encoding="utf-8")  # existence is the whole build-time probe
    monkeypatch.setenv(ENV_CUROBO_PYTHON, str(present))

    text = announce_planning_environment("curobo")

    assert "cuRobo planner: AVAILABLE" in text
    assert "BLIND IK" not in text
