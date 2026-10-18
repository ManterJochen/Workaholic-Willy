"""Off-box contract for the Aletheia GSO loader — curation + path resolution (conversion is on-box).

The ``--convert`` conversion and the sim load are on-box; here we lock the parts that run without Isaac:
the measured graspable/clutter split, the env-overridable source / output paths, and that an object spec
refuses to build until its USD has actually been converted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.willy_sim.gso_assets import (
    GSO_PARTS,
    gso_object_spec,
    gso_source_dir,
    gso_usd_dir,
    gso_usd_path,
)


def test_graspable_split_follows_measured_min_dimension() -> None:
    for p in GSO_PARTS:
        assert p.graspable == (p.min_dim_mm <= 60.0)
    # the curated set has BOTH graspable objects (small toys / thin) and clutter (the wide mugs)
    assert any(p.graspable for p in GSO_PARTS)
    assert any(not p.graspable for p in GSO_PARTS)


def test_the_wide_mugs_are_clutter_the_small_toys_are_graspable() -> None:
    by_label = {p.label: p for p in GSO_PARTS}
    assert by_label["coffee mug"].graspable is False   # 90 mm min dim > the jaw span
    assert by_label["toy dog"].graspable is True        # 35 mm min dim


def test_source_and_output_dirs_are_env_overridable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WILLY_GSO_DIR", str(tmp_path / "src"))
    monkeypatch.delenv("WILLY_GSO_USD_DIR", raising=False)
    assert gso_source_dir() == tmp_path / "src"
    assert gso_usd_dir() == tmp_path / "gso_usd"          # sibling of the source by default
    monkeypatch.setenv("WILLY_GSO_USD_DIR", str(tmp_path / "out"))
    assert gso_usd_dir() == tmp_path / "out"              # explicit override wins


def test_object_spec_requires_a_converted_usd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WILLY_GSO_USD_DIR", str(tmp_path))  # empty dir -> nothing converted
    with pytest.raises(FileNotFoundError, match="Run"):
        gso_object_spec(GSO_PARTS[0], position_mm=(450.0, 0.0, 40.0))


def test_object_spec_references_the_converted_usd_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WILLY_GSO_USD_DIR", str(tmp_path))
    part = GSO_PARTS[0]
    gso_usd_path(part).write_text("#usda 1.0\n", encoding="utf-8")  # pretend it was converted
    spec = gso_object_spec(part, position_mm=(450.0, 0.0, 40.0))
    assert spec.name == part.label
    assert spec.usd_asset_path is not None and spec.usd_asset_path.endswith(".usd")
    assert spec.usd_collision_approximation == part.collision
    assert spec.position_mm == (450.0, 0.0, 40.0)
