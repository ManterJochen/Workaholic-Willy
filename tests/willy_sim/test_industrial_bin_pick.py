"""Off-box contract for the industrial bin-pick flagship (import-safety + the pure scene helper).

The on-box gate proves the two-side localize + full-pipeline pick (5/5 in the shallow KLT tray); here we
lock the parts that gate cannot see off-box: the runner imports without Isaac, and the scene spec places the
prompted target at the tray centre with the requested clutter around it, all jaw-graspable and inside the
tray footprint.
"""

from __future__ import annotations

from src.willy_sim.run_industrial_bin_pick import (
    CLUTTER_PARTS,
    TARGET_PART,
    _scene_specs,
    run_industrial_gate,
)


def test_runner_imports_without_isaac() -> None:
    """The flagship must import on macOS / CI (Isaac stays lazy inside run_industrial_gate)."""
    assert callable(run_industrial_gate)


def test_scene_specs_puts_the_target_first_at_the_tray_centre() -> None:
    specs = _scene_specs(4)
    assert len(specs) == 5  # target + 4 clutter
    assert specs[0].name == TARGET_PART[0]
    assert specs[0].position_mm[0] == 450.0 and specs[0].position_mm[1] == 0.0  # target at the tray centre


def test_scene_specs_are_all_jaw_graspable_and_inside_the_tray() -> None:
    for s in _scene_specs(len(CLUTTER_PARTS)):
        assert s.shape == "cube"
        assert max(s.size_mm[:2]) <= 85.0             # every footprint fits the 2F-85 span
        assert abs(s.position_mm[0] - 450.0) <= 146.0  # inside the tray x half-extent
        assert abs(s.position_mm[1]) <= 94.0           # inside the tray y half-extent


def test_scene_specs_clamps_clutter_to_the_catalog() -> None:
    assert len(_scene_specs(99)) == 1 + len(CLUTTER_PARTS)


def test_scene_specs_zero_clutter_is_target_only() -> None:
    specs = _scene_specs(0)
    assert len(specs) == 1 and specs[0].name == TARGET_PART[0]
