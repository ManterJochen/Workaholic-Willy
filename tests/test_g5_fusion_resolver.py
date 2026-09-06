"""Gap G5 (Stufe A) — build_config_frame_resolver: config-built CAMERA->BASE resolver, fail-closed.

This is what makes the U5 fusion substrate + the U6 commit gate REACHABLE from production config: when
grasping.fusion.enabled is True and extrinsics_artifact_path points at a persisted eye-to-hand Extrinsics
JSON, from_robot_config auto-builds a StaticCameraToBaseResolver. Default-off / no-path is byte-identical
(returns None); a set-but-unloadable path RAISES (fail-closed, Tim's decision).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.calibration.extrinsics import Extrinsics
from src.calibration.serialization import save_extrinsics
from src.geometry import Frame, Transform
from src.robot.execution.autonomous_grasp.builders import build_config_frame_resolver
from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver


def _cfg(*, enabled: bool, path: str | None) -> SimpleNamespace:
    return SimpleNamespace(fusion=SimpleNamespace(enabled=enabled, extrinsics_artifact_path=path))


class TestBuildConfigFrameResolver:
    def test_none_grasping_cfg_returns_none(self) -> None:
        assert build_config_frame_resolver(None) is None

    def test_fusion_disabled_returns_none(self) -> None:
        assert build_config_frame_resolver(_cfg(enabled=False, path="/whatever.json")) is None

    def test_enabled_no_path_returns_none(self) -> None:
        # Eye-in-hand cells leave the path null + supply a resolver in code -> no auto-build, not an error.
        assert build_config_frame_resolver(_cfg(enabled=True, path=None)) is None

    def test_enabled_bad_path_raises_fail_closed(self) -> None:
        # FAIL-CLOSED: fusion enabled + a set-but-unloadable path must RAISE, not silently degrade to an
        # unreachable gate while the operator believes fusion is on.
        with pytest.raises(RuntimeError, match="failed to load"):
            build_config_frame_resolver(
                _cfg(enabled=True, path="/definitely/not/a/real/extrinsics_artifact.json")
            )

    def test_enabled_valid_artifact_builds_resolver(self, tmp_path) -> None:
        transform = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        ext = Extrinsics(
            transform=transform, rmse_mm=1.0, max_error_mm=2.0, num_samples=10,
            captured_at=datetime.now(timezone.utc), rig_id="g5_test_rig",
        )
        artifact = tmp_path / "eth_extrinsics.json"
        save_extrinsics(artifact, ext)
        resolver = build_config_frame_resolver(_cfg(enabled=True, path=str(artifact)))
        assert isinstance(resolver, StaticCameraToBaseResolver)
        assert resolver.transform.from_frame is Frame.CAMERA
        assert resolver.transform.to_frame is Frame.BASE
