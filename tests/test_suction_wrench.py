"""Tests for the suction wrench-resistance model (H7.2). Pure statics, no Isaac."""

from __future__ import annotations

import numpy as np

from src.robot.grasping.suction.wrench import (
    WrenchConfig,
    evaluate_wrench_resistance,
)

_TOP = [0.0, 0.0, -1.0]  # a top-face suction presses straight down (approach), gravity is also down


def test_light_part_at_contact_is_comfortably_held() -> None:
    r = evaluate_wrench_resistance([0, 0, 100], _TOP, payload_mass_g=100.0, com_mm=[0, 0, 100])
    assert r.feasible
    assert r.resist_score == 1.0
    assert r.required_pull_n > 0.0  # gravity pulls the part off a top suction
    assert r.required_moment_nmm == 0.0  # CoM under the contact -> no tilt


def test_too_heavy_part_is_infeasible() -> None:
    # default cup ~35 N pull-off budget; 5 kg -> ~49 N pull -> cannot hold.
    r = evaluate_wrench_resistance([0, 0, 100], _TOP, payload_mass_g=5000.0, com_mm=[0, 0, 100])
    assert not r.feasible
    assert r.resist_score < 1.0
    assert r.required_pull_n > r.vacuum_force_n


def test_offset_com_reduces_moment_margin() -> None:
    # the further the CoM from the contact (perpendicular to gravity), the larger the tilt moment.
    s0 = evaluate_wrench_resistance([0, 0, 100], _TOP, payload_mass_g=500.0, com_mm=[0, 0, 100])
    s150 = evaluate_wrench_resistance([0, 0, 100], _TOP, payload_mass_g=500.0, com_mm=[150, 0, 100])
    s200 = evaluate_wrench_resistance([0, 0, 100], _TOP, payload_mass_g=500.0, com_mm=[200, 0, 100])
    assert s0.resist_score == 1.0
    assert s200.resist_score < s150.resist_score < s0.resist_score
    assert not s200.feasible


def test_side_suction_holds_light_part_by_friction() -> None:
    # approach horizontal: gravity is pure shear, resisted by friction under the vacuum preload.
    r = evaluate_wrench_resistance([0, 0, 100], [1.0, 0.0, 0.0], payload_mass_g=500.0, com_mm=[0, 0, 100])
    assert r.required_pull_n == 0.0
    assert r.shear_n > 0.0
    assert r.feasible


def test_vacuum_force_scales_with_pressure_and_area() -> None:
    weak = WrenchConfig(vacuum_kpa=20.0, cup_radius_mm=10.0)
    strong = WrenchConfig(vacuum_kpa=70.0, cup_radius_mm=20.0)
    assert strong.vacuum_force_n > weak.vacuum_force_n
    # F = P * pi * r^2: 70 kPa, 20 mm -> 70000 * pi * 0.02^2 ≈ 87.96 N
    np.testing.assert_allclose(strong.vacuum_force_n, 70000.0 * np.pi * 0.02**2, rtol=1e-6)


def test_determinism() -> None:
    a = evaluate_wrench_resistance([0, 0, 100], _TOP, payload_mass_g=300.0, com_mm=[40, 0, 100])
    b = evaluate_wrench_resistance([0, 0, 100], _TOP, payload_mass_g=300.0, com_mm=[40, 0, 100])
    assert a.resist_score == b.resist_score


def test_config_validation() -> None:
    for bad in ({"vacuum_kpa": 0.0}, {"cup_radius_mm": 0.0}, {"friction_coef": -1.0}, {"moment_arm_factor": 0.0}):
        try:
            WrenchConfig(**bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
