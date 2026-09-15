"""Gap G5 (Stufe A): build_config_frame_resolver builds the primary camera's resolver from its rig, fail-closed.

This makes the U5 fusion substrate and the U6 commit gate reachable from production config: when the primary
rig declares its calibration, ``camera.cameras.rigs[<primary>].extrinsics``, as eye_to_hand with an artifact
holding a persisted Extrinsics JSON, from_robot_config builds a StaticCameraToBaseResolver. No camera section,
or a primary that declares no calibration, returns None and leaves the path byte-identical; a declared but
unloadable artifact raises (fail-closed, an owner decision) and names the rig key.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from src.config import load_config
from src.config.schema.robot.grasping_schema import RobotGraspingConfig
from src.calibration.extrinsics import Extrinsics
from src.calibration.serialization import save_extrinsics
from src.geometry import Frame, Transform
from src.robot.execution.autonomous_grasp.builders import build_config_frame_resolver
from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver

_RIG = "g5_test_rig"
_KEY = f"camera.cameras.rigs[{_RIG!r}].extrinsics"


def _grasping(*, fusion_enabled: bool) -> RobotGraspingConfig:
    return RobotGraspingConfig.model_validate({"fusion": {"enabled": fusion_enabled}})


def _camera(path: str | None) -> Any:
    """The shipped camera section whose RGB-D rig is the primary, declaring an eye_to_hand artifact at ``path``."""
    shipped = load_config().camera
    data = shipped.model_dump(mode="json")
    rig = next(r for r in data["cameras"]["rigs"] if r["source"] == "rgbd")
    extrinsics = None if path is None else {"mounting_mode": "eye_to_hand", "artifact_path": path}
    rig.update({"rig_id": _RIG, "enabled": True, "extrinsics": extrinsics})
    data["cameras"]["primary_rig_id"] = _RIG
    return type(shipped).model_validate(data)


def _artifact(tmp_path) -> str:
    transform = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)
    ext = Extrinsics(
        transform=transform, rmse_mm=1.0, max_error_mm=2.0, num_samples=10,
        captured_at=datetime.now(timezone.utc), rig_id=_RIG,
    )
    artifact = tmp_path / "eth_extrinsics.json"
    save_extrinsics(artifact, ext)
    return str(artifact)


class TestBuildConfigFrameResolver:
    def test_none_grasping_cfg_returns_none(self) -> None:
        assert build_config_frame_resolver(None) is None

    def test_no_camera_section_returns_none(self) -> None:
        # The calibration is declared on the rig, so a grasping block alone names no artifact to build from.
        assert build_config_frame_resolver(_grasping(fusion_enabled=True)) is None

    def test_fusion_disabled_still_builds_from_the_rig(self, tmp_path) -> None:
        # The resolver turns every grasp into the base frame, and fusion gates only the fusion substrate, so a
        # calibrated primary gets its resolver with fusion switched off.
        resolver = build_config_frame_resolver(_grasping(fusion_enabled=False), camera=_camera(_artifact(tmp_path)))
        assert isinstance(resolver, StaticCameraToBaseResolver)

    def test_uncalibrated_primary_returns_none(self) -> None:
        # A primary that declares no calibration gets no auto-build, not an error: a desk cell passes its
        # resolver in code or needs none.
        assert build_config_frame_resolver(_grasping(fusion_enabled=True), camera=_camera(None)) is None

    def test_declared_bad_path_raises_fail_closed(self) -> None:
        # Fail-closed: a declared but unloadable artifact must raise, not silently degrade to an unreachable
        # gate while the operator believes the camera is calibrated.
        with pytest.raises(RuntimeError, match="does not load") as caught:
            build_config_frame_resolver(
                _grasping(fusion_enabled=True), camera=_camera("/definitely/not/a/real/extrinsics_artifact.json")
            )
        assert _KEY in str(caught.value)

    def test_declared_valid_artifact_builds_resolver(self, tmp_path) -> None:
        resolver = build_config_frame_resolver(_grasping(fusion_enabled=True), camera=_camera(_artifact(tmp_path)))
        assert isinstance(resolver, StaticCameraToBaseResolver)
        assert resolver.transform.from_frame is Frame.CAMERA
        assert resolver.transform.to_frame is Frame.BASE
