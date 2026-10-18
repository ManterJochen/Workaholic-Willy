"""A fused pick waits for every camera its cell fuses but the primary, and says which one did not deliver.

There is no fusion switch since 2026-10-16 (the owner's cleanup): a cell fuses every enabled RGB-D rig its camera
section declares with a calibration, where two or more do (``camera_fusion.fused_cameras``), and the overlay builds a
CAMERA to BASE resolver for each, the primary among them. What the pick then expects is read off that map, built from
the config, never off a map wired by hand: the lesson of the guard that once could not fire because the map a configured
cell got was empty.

The primary is never waited for. It delivers through the cell's main perception source and never through the
extra-camera rig, so a pick that expected it there would find it missing on every pick. A camera that delivered no
frame is warned of, and the pick goes on with the ones that did: a cell never falls back to single view in silence.

NEVER RUN ON HARDWARE: no physical multi-camera cell exists; this is proved against the shipped two-camera tree and a
fake rig.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.config.schema.robot.grasping_schema import RobotGraspingConfig
from src.config.tree import load_tree
from src.geometry import Frame, Transform
from src.robot.execution.autonomous_grasp.builders import apply_orchestrator_overlays
from src.robot.execution.autonomous_grasp.config import GraspMode
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
from src.robot.grasping.types.perception import CameraObservation, PerceptionFrame

_IDENTITY = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)


class _Resolver:
    """A fixed camera's CAMERA to BASE: what a loaded calibration gives, without an artifact on disk."""

    def camera_to_base_for_frame(self, _frame: Any, *, arm: Any = None) -> Transform:
        return _IDENTITY


class _Rig:
    """The other cameras of a fused cell, delivering the ids given and nothing else."""

    def __init__(self, *delivered: str) -> None:
        self._delivered = delivered
        self.last_failures: dict[str, str] = {}

    def acquire_all(self) -> tuple[CameraObservation, ...]:
        frame = PerceptionFrame(depth_map=np.full((8, 8), 500.0), intrinsics=np.eye(3), segmentations=())
        return tuple(CameraObservation(camera_id=camera, frame=frame) for camera in self._delivered)


def _from_config(*, values: dict[str, Any] | None = None, primary: str = "cam_left", delivered: tuple[str, ...] = ()
                 ) -> BinPickingOrchestrator:
    """The pick loop the overlay wires from the shipped two-camera tree, its calibrations stood in for."""
    tree = load_tree("ur5e,eth2")
    if values:
        tree = tree.with_values(values)
    assert tree.ok, tree.error
    orchestrator = BinPickingOrchestrator(
        arm=SimpleNamespace(),  # type: ignore[arg-type]
        calculator=SimpleNamespace(),  # type: ignore[arg-type]
        perception=SimpleNamespace(),  # type: ignore[arg-type]
        multi_camera_perception=_Rig(*delivered),
        primary_camera_id=primary,
    )
    with mock.patch("src.robot.execution.autonomous_grasp.builders._resolver_for_rig",
                    side_effect=lambda _rig_id, _extrinsics: _Resolver()):
        apply_orchestrator_overlays(SimpleNamespace(orchestrator=orchestrator),  # type: ignore[arg-type]
                                    RobotGraspingConfig(), resolved_mode=GraspMode.AUTO,
                                    camera=tree.app_config.camera)
    return orchestrator


def _missing(orchestrator: BinPickingOrchestrator) -> list[str]:
    _views, missing, _grounded_nothing, _clock = orchestrator._fixed_camera_views()
    return missing


class TheConfigBuildsTheMapTests(unittest.TestCase):
    def test_both_cameras_of_the_reference_cell_get_a_resolver(self) -> None:
        orchestrator = _from_config(delivered=("cam_right",))

        self.assertEqual({"cam_left", "cam_right"}, set(orchestrator.camera_frame_resolvers or {}))

    def test_a_rig_switched_off_leaves_one_camera_and_no_map(self) -> None:
        orchestrator = _from_config(values={"camera.cameras.rigs[1].enabled": False})

        self.assertIsNone(orchestrator.camera_frame_resolvers, "one camera has nothing to fuse with")


class ThePrimaryIsNotWaitedForTests(unittest.TestCase):
    def test_the_primary_is_excluded_from_what_a_pick_waits_for(self) -> None:
        self.assertEqual([], _missing(_from_config(delivered=("cam_right",))))

    def test_a_camera_that_delivered_nothing_is_warned_of_and_the_pick_goes_on(self) -> None:
        orchestrator = _from_config(delivered=())

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="WARNING") as logged:
            self.assertEqual(["cam_right"], _missing(orchestrator))
        self.assertTrue(any("delivered no frame" in line and "cam_right" in line for line in logged.output))

    def test_without_a_named_primary_every_camera_is_expected(self) -> None:
        """A simulator runner builds its own rig and hands it in, so nothing there is delivered by a main source."""
        self.assertEqual(["cam_left"], _missing(_from_config(primary="", delivered=("cam_right",))))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
