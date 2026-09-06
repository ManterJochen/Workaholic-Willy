"""Gap G12-slice — RobotGraspingApproachValidationConfig schema: defaults + the EASY lock-out.

The orchestrator behavior (the swept-volume validator + the candidate fall-back) is pinned by
tests/test_pick_loop.py::ApproachPathValidationTests; this pins the config block that enables it from
production config (dense-only, default-off, EASY exempt so M1/EIH/single-object stay byte-identical).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.schema.robot.robot_schema import RobotGraspingApproachValidationConfig


def test_defaults_disabled_and_dense_only() -> None:
    cfg = RobotGraspingApproachValidationConfig()
    assert cfg.enabled is False
    assert tuple(cfg.apply_modes) == ("dense_clutter", "dense_autonomous")
    assert cfg.num_approach_samples == 6
    assert cfg.collision_margin_mm == 0.0


def test_apply_modes_forbids_easy() -> None:
    with pytest.raises(ValidationError):
        RobotGraspingApproachValidationConfig(apply_modes=("easy", "dense_clutter"))


def test_apply_modes_must_be_non_empty_and_unique() -> None:
    with pytest.raises(ValidationError):
        RobotGraspingApproachValidationConfig(apply_modes=())
    with pytest.raises(ValidationError):
        RobotGraspingApproachValidationConfig(apply_modes=("dense_clutter", "dense_clutter"))
