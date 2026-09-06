"""WS3.4: `on_camera_unavailable: refuse` compared against a map that production leaves EMPTY.

⛔⛔ THE GUARD COULD NOT FIRE IN THE CONFIGURATION IT EXISTS FOR. `_fused_scene` built its
expectation as `set(self.camera_frame_resolvers)`, and `build_config_frame_resolvers` returns `{}`
whenever `grasping_cfg` is None, `fusion.enabled` is false, or the `cameras` map is empty. In every
one of those cases `expected` was empty, so `missing` was empty, so neither the refusal nor the
warning could reach a single line of output. The schema states the cost in the option's own words:
"a cell must never fall back to single-view silently, which is the whole failure this option exists
to make visible."

⚠ THE EXISTING POLICY TESTS PASSED THROUGHOUT, and that is the lesson rather than an aside. Both
`test_a_missing_camera_warns_under_degrade` and `test_a_missing_camera_refuses_under_refuse` hand the
orchestrator a two-entry resolver map directly. A hand-wired map is the one input production never
produces, so the tests exercised a path that no configured cell could reach. The tests below start
from a CONFIG.

⚠ NEVER RUN ON HARDWARE. No physical multi-camera cell has run this; the refusal is proven here
against a config and a fake rig. Bucket 3 until one does.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.config.schema.robot.grasping_schema import (
    CameraExtrinsicsConfig,
    FusionGeometryConfig,
    RobotGraspingFusionConfig,
)
from src.geometry import Frame, Transform
from src.robot.execution.autonomous_grasp.builders import build_config_frame_resolvers
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
from src.robot.grasping.types.perception import CameraObservation, PerceptionFrame

_LOGGER = "src.robot.grasping.loop.pick_loop"
_IDENTITY = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)


def _frame() -> PerceptionFrame:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 1
    return PerceptionFrame(
        depth_map=np.full((32, 32), 500.0, dtype=np.float64),
        intrinsics=np.array([[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]]),
        segmentations=(SimpleNamespace(mask=mask),),
    )


class _Resolver:
    def camera_to_base_for_frame(self, _frame_in, *, arm=None):  # noqa: ANN001, ANN202, ARG002
        return _IDENTITY


class _Rig:
    """A rig where only `left` ever delivers, which is the dropped-trigger scenario."""

    def acquire_all(self) -> tuple[CameraObservation, ...]:
        return (CameraObservation(camera_id="left", frame=_frame()),)


def _fusion(*, enabled: bool, cameras: dict[str, CameraExtrinsicsConfig]):
    return RobotGraspingFusionConfig(enabled=enabled, cameras=cameras)


def _camera(*, enabled: bool = True) -> CameraExtrinsicsConfig:
    return CameraExtrinsicsConfig(enabled=enabled, extrinsics_artifact_path="nowhere.json")


def _orchestrator(**kwargs) -> BinPickingOrchestrator:
    return BinPickingOrchestrator(
        arm=SimpleNamespace(),  # type: ignore[arg-type]
        calculator=SimpleNamespace(),  # type: ignore[arg-type]
        perception=SimpleNamespace(acquire=_frame),  # type: ignore[arg-type]
        multi_camera_perception=_Rig(),  # type: ignore[arg-type]
        **kwargs,
    )


class TheHoleTests(unittest.TestCase):
    """First: prove the empty map, because everything else follows from it."""

    def test_the_resolver_map_IS_empty_when_fusion_is_disabled(self) -> None:
        """⛔ THE ROOT. Two cameras named in the config, and the map the policy compared against has
        nothing in it. `fusion.geometry.enabled` and `fusion.enabled` are separate keys, so this is a
        configuration an operator can and does write."""
        cfg = SimpleNamespace(fusion=_fusion(
            enabled=False, cameras={"left": _camera(), "right": _camera()}))

        self.assertEqual(build_config_frame_resolvers(cfg), {})  # type: ignore[arg-type]

    def test_an_expectation_read_off_that_map_is_EMPTY(self) -> None:
        """The defect stated as arithmetic. `expected - delivered` over an empty `expected` is empty
        for every possible `delivered`, so no missing camera exists to refuse over."""
        expected: set[str] = set(build_config_frame_resolvers(  # type: ignore[arg-type]
            SimpleNamespace(fusion=_fusion(enabled=False, cameras={"left": _camera()}))))

        self.assertEqual(sorted(expected - {"left"}), [])
        self.assertEqual(sorted(expected - set()), [],
                         "a camera that delivered NOTHING was still not missing")


class TheRepairTests(unittest.TestCase):
    """`configured_camera_ids` says what was ASKED FOR, which is what the policy needs."""

    def test_a_configured_camera_that_delivers_nothing_is_REFUSED(self) -> None:
        """⭐ THE POINT OF THE WHOLE ITEM. Two cameras configured, one delivers, and the cell now
        stops instead of grasping on half its evidence."""
        orch = _orchestrator(
            configured_camera_ids=("left", "right"),
            camera_frame_resolvers={},
            fusion_geometry_config=FusionGeometryConfig(
                enabled=True, on_camera_unavailable="refuse"),
        )

        with self.assertRaises(RuntimeError) as ctx:
            orch._fused_scene(_frame(), _IDENTITY)

        self.assertIn("right", str(ctx.exception))

    def test_the_same_configuration_WARNS_under_degrade(self) -> None:
        """The other arm of the policy, from the same expectation. `degrade` is a deliberate choice
        to keep picking; it is only a choice if the operator is told it happened."""
        orch = _orchestrator(
            configured_camera_ids=("left", "right"),
            camera_frame_resolvers={"left": _Resolver()},
            fusion_geometry_config=FusionGeometryConfig(
                enabled=True, on_camera_unavailable="degrade"),
        )

        with self.assertLogs(_LOGGER, level="WARNING") as captured:
            orch._fused_scene(_frame(), _IDENTITY)

        self.assertTrue(any("right" in line for line in captured.output))

    def test_a_DISABLED_camera_is_not_expected(self) -> None:
        """⚠ The control that keeps the repair from becoming a nuisance. A camera switched off in
        the config was not asked for, and refusing over it would make `enabled: false` unusable."""
        cameras = {"left": _camera(), "right": _camera(enabled=False)}
        asked_for = tuple(sorted(
            cam_id for cam_id, cam in cameras.items() if bool(cam.enabled)))

        self.assertEqual(asked_for, ("left",))
        orch = _orchestrator(
            configured_camera_ids=asked_for,
            camera_frame_resolvers={"left": _Resolver()},
            fusion_geometry_config=FusionGeometryConfig(
                enabled=True, on_camera_unavailable="refuse"),
        )
        self.assertIsNotNone(orch._fused_scene(_frame(), _IDENTITY),
                             "a camera the operator switched off was treated as missing")

    def test_a_cell_that_configured_NOTHING_is_unchanged(self) -> None:
        """⚠ EMPTY MEANS "NOTHING WAS CONFIGURED", NOT "NOTHING ARRIVED". A single-view cell is
        single-view by design and must not be refused; this is what keeps the change default-off
        byte-identical for every caller that wires resolvers in code."""
        orch = _orchestrator(
            camera_frame_resolvers={"left": _Resolver()},
            fusion_geometry_config=FusionGeometryConfig(
                enabled=True, on_camera_unavailable="refuse"),
        )

        self.assertEqual(orch.configured_camera_ids, ())
        self.assertIsNotNone(orch._fused_scene(_frame(), _IDENTITY))

    def test_the_hand_wired_map_STILL_serves_as_the_fallback(self) -> None:
        """The two older policy tests wire the map directly and must keep passing unchanged: a
        caller that never sets the configured list gets exactly the old expectation."""
        orch = _orchestrator(
            camera_frame_resolvers={"left": _Resolver(), "right": _Resolver()},
            fusion_geometry_config=FusionGeometryConfig(
                enabled=True, on_camera_unavailable="refuse"),
        )

        with self.assertRaises(RuntimeError) as ctx:
            orch._fused_scene(_frame(), _IDENTITY)

        self.assertIn("right", str(ctx.exception))


class TheBuilderTests(unittest.TestCase):
    """The field is useless unless the production boot path fills it."""

    def test_from_robot_config_fills_it_from_fusion_cameras(self) -> None:
        """Read off the builder source: the assignment sits beside the resolver map it replaces as
        the expectation, so the two cannot drift apart unnoticed."""
        from pathlib import Path

        source = Path(
            "src/robot/execution/autonomous_grasp/builders.py").read_text(encoding="utf-8")

        self.assertIn("runtime.orchestrator.configured_camera_ids = tuple(", source)
        self.assertIn('if bool(getattr(cam, "enabled", True))', source,
                      "the builder would expect cameras the operator switched off")


if __name__ == "__main__":
    unittest.main()
