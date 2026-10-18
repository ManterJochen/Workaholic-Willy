"""P2 dense-clutter substrate: multi-object GT perception + the service label-target seam.

Pure / mock-safe (no isaacsim): a fake camera drives the instance-id segmentation path so we pin that
MultiObjectGroundTruthPerceptionSource emits ONE labelled segmentation per object, and that
AutonomousGraspService.set_target_label threads the prompt-target onto the orchestrator.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.willy_sim.harness.ik_service import ArmBackedIKService
from src.willy_sim.perception import MultiObjectGroundTruthPerceptionSource
from src.robot.core import JointPositions
from src.robot.execution.autonomous_grasp.service import AutonomousGraspService


class _FakeCamera:
    """Minimal stand-in for an Isaac Camera with a 2-object instance-id frame (no isaacsim)."""

    def __init__(self) -> None:
        self._depth_m = np.full((4, 4), 0.5, dtype=np.float64)  # metres
        self._k = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        data = np.zeros((4, 4), dtype=np.int64)
        data[0, :] = 1  # row 0 -> prim Object_0
        data[1, :] = 2  # row 1 -> prim Object_1
        self._frame = {
            "instance_id_segmentation": {
                "data": data,
                "info": {"idToLabels": {"1": "/World/Object_0", "2": "/World/Object_1"}},
            }
        }

    def add_distance_to_image_plane_to_frame(self) -> None: ...
    def add_instance_id_segmentation_to_frame(self) -> None: ...
    def get_depth(self) -> np.ndarray:
        return self._depth_m

    def get_intrinsics_matrix(self) -> np.ndarray:
        return self._k

    def get_current_frame(self) -> dict:
        return self._frame

    def get_rgb(self):  # noqa: ANN201
        return None


class MultiObjectGroundTruthPerceptionTests(unittest.TestCase):
    def test_emits_one_labelled_segmentation_per_object(self) -> None:
        src = MultiObjectGroundTruthPerceptionSource(
            camera=_FakeCamera(),
            targets=[("/World/Object_0", "red"), ("/World/Object_1", "blue")],
        )
        frame = src.acquire()
        self.assertEqual(len(frame.segmentations), 2)
        self.assertEqual([s.label for s in frame.segmentations], ["red", "blue"])
        self.assertEqual([s.prim_path for s in frame.segmentations],
                         ["/World/Object_0", "/World/Object_1"])
        self.assertEqual(int(np.asarray(frame.segmentations[0].mask).sum()), 4)  # row 0
        self.assertEqual(int(np.asarray(frame.segmentations[1].mask).sum()), 4)  # row 1
        # disjoint masks (different objects)
        self.assertEqual(int(np.logical_and(frame.segmentations[0].mask,
                                            frame.segmentations[1].mask).sum()), 0)

    def test_missing_prim_yields_empty_mask_for_that_object(self) -> None:
        src = MultiObjectGroundTruthPerceptionSource(
            camera=_FakeCamera(),
            targets=[("/World/Object_0", "red"), ("/World/NotThere", "ghost")],
        )
        frame = src.acquire()
        self.assertEqual(len(frame.segmentations), 2)
        self.assertEqual(int(np.asarray(frame.segmentations[1].mask).sum()), 0)


class _FakeGraspPose:
    position_mm = np.array([450.0, -95.0, 37.0])
    rotation_matrix = np.eye(3)


class _IKArm:
    def __init__(self, *, raises: bool = False) -> None:
        self._raises = raises

    def ik(self, pose):  # noqa: ANN001
        if self._raises:
            raise RuntimeError("unreachable")
        return JointPositions(np.zeros(6))


def _report(*, near: bool, cond: float, min_sv: float):
    return SimpleNamespace(is_near_singularity=near, condition_number=cond, min_singular_value=min_sv)


class ArmBackedIKServiceTests(unittest.TestCase):
    """P2 B-fix: the IK oracle rejects unreachable AND near-singular grasps; the query-logic branches."""

    def test_ik_failure_is_unreachable(self) -> None:
        res = ArmBackedIKService(_IKArm(raises=True)).query(_FakeGraspPose())
        self.assertFalse(res.reachable)
        self.assertIn("ik_failed", res.reason or "")

    def test_near_singular_is_rejected(self) -> None:
        with mock.patch(
            "src.robot.safety.singularity.analyze_joint_singularity",
            return_value=_report(near=True, cond=1.0e9, min_sv=1.0e-4),
        ):
            res = ArmBackedIKService(_IKArm()).query(_FakeGraspPose())
        self.assertFalse(res.reachable)
        self.assertEqual(res.reason, "near_singular")

    def test_well_conditioned_is_reachable(self) -> None:
        with mock.patch(
            "src.robot.safety.singularity.analyze_joint_singularity",
            return_value=_report(near=False, cond=5.0, min_sv=0.5),
        ):
            res = ArmBackedIKService(_IKArm()).query(_FakeGraspPose())
        self.assertTrue(res.reachable)


class CalculatorIKServiceCtorTests(unittest.TestCase):
    def test_default_off_and_set(self) -> None:
        from src.robot.grasping.generation.calculator import GraspCalculator

        k = np.array([[200.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]])
        self.assertIsNone(GraspCalculator(camera_matrix=k)._default_ik_service)
        svc = ArmBackedIKService(_IKArm())
        self.assertIs(GraspCalculator(camera_matrix=k, ik_service=svc)._default_ik_service, svc)


class BlockingSpecsTests(unittest.TestCase):
    """P3.0a: the blocking-scene variant adds a blocker in the red target's +X approach column."""

    def test_blocking_specs_appends_blocker_in_red_plus_x(self) -> None:
        from src.willy_sim.run_dense_pick import _blocking_specs
        from src.willy_sim.run_fused_pick import _clutter_specs, _select_by_prompt

        specs = _blocking_specs(35.0)
        self.assertEqual(len(specs), len(_clutter_specs()) + 1)
        blocker = specs[-1]
        self.assertEqual(blocker.name, "orange blocker")
        self.assertEqual(tuple(blocker.position_mm), (485.0, -95.0, 25.0))  # +35 in +X from red(-95)
        # the prompt still selects RED (the blocker shares neither "red" nor "cube")
        self.assertEqual(specs[_select_by_prompt(specs, "the red cube")].name, "red cube")

    def test_blocker_offset_is_parametric(self) -> None:
        from src.willy_sim.run_dense_pick import _blocking_specs

        self.assertEqual(tuple(_blocking_specs(28.0)[-1].position_mm), (478.0, -95.0, 25.0))


class ServiceTargetLabelTests(unittest.TestCase):
    def test_set_target_label_threads_to_orchestrator(self) -> None:
        svc = object.__new__(AutonomousGraspService)
        svc.runtime = SimpleNamespace(orchestrator=SimpleNamespace(target_label=None))  # type: ignore[attr-defined]
        svc.set_target_label("red_cube")
        self.assertEqual(svc.runtime.orchestrator.target_label, "red_cube")
        svc.set_target_label(None)
        self.assertIsNone(svc.runtime.orchestrator.target_label)

    def test_set_target_label_noop_without_orchestrator(self) -> None:
        svc = object.__new__(AutonomousGraspService)
        svc.runtime = SimpleNamespace(orchestrator=None)  # type: ignore[attr-defined]
        svc.set_target_label("x")  # must not raise


class S1ColliderAuthoringSpecTests(unittest.TestCase):
    """S1: the schema seam + the collider-authored compact-box scene selector (Isaac-free).

    The runtime UsdPhysics authoring itself is on-box only; here we lock that the opt-in field defaults
    OFF (byte-identical), the extra-box specs opt in correctly, and the default scene is untouched.
    """

    def test_field_defaults_off(self) -> None:
        import os

        from src.config.schema.robot import SimObjectConfig
        from src.willy_sim.run_dense_pick import _ycb_specs
        from src.willy_sim.harness.env import RunnerEnv

        # default: no collider-authoring -> byte-identical for cubes + the pre-rigged Axis_Aligned_Physics YCB
        self.assertIsNone(SimObjectConfig(name="cube").usd_collision_approximation)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WILLY_YCB_SCENE", None)
            for spec in _ycb_specs(False, RunnerEnv.from_env(vision=False)):
                self.assertIsNone(spec.usd_collision_approximation)  # demo scene authors nothing
                self.assertIn("Axis_Aligned_Physics", spec.usd_asset_path or "")

    def test_field_round_trips(self) -> None:
        from src.config.schema.robot import SimObjectConfig

        spec = SimObjectConfig(name="pudding box", usd_collision_approximation="convexHull")
        self.assertEqual(spec.usd_collision_approximation, "convexHull")

    def test_extra_specs_opt_into_authoring(self) -> None:
        from src.willy_sim.run_dense_pick import _EDGE_Y90, _ycb_extra_specs

        for sel, name, usd in (("pudding", "pudding box", "008_pudding_box.usd"),
                               ("gelatin", "gelatin box", "009_gelatin_box.usd")):
            specs = _ycb_extra_specs(sel)
            self.assertEqual(len(specs), 1)
            s = specs[0]
            self.assertEqual(s.name, name)
            self.assertEqual(s.usd_collision_approximation, "convexHull")  # opts into the collider helper
            self.assertIn("Axis_Aligned/", s.usd_asset_path or "")        # the NON-physics root
            self.assertNotIn("Axis_Aligned_Physics", s.usd_asset_path or "")
            self.assertTrue((s.usd_asset_path or "").endswith(usd))
            self.assertEqual(tuple(s.orientation_wxyz or ()), _EDGE_Y90)   # on-edge graspable axis

    def test_extra_scene_has_both_boxes(self) -> None:
        from src.willy_sim.run_dense_pick import _ycb_extra_specs

        specs = _ycb_extra_specs("extra")
        self.assertEqual([s.name for s in specs], ["pudding box", "gelatin box"])
        self.assertTrue(all(s.usd_collision_approximation == "convexHull" for s in specs))
        # distinct positions so the two boxes don't interpenetrate
        self.assertNotEqual(tuple(specs[0].position_mm), tuple(specs[1].position_mm))

    def test_scene_env_selects_extra_specs(self) -> None:
        import os

        from src.willy_sim.run_dense_pick import _ycb_specs
        from src.willy_sim.harness.env import RunnerEnv

        with mock.patch.dict(os.environ, {"WILLY_YCB_SCENE": "pudding"}):
            specs = _ycb_specs(False, RunnerEnv.from_env(vision=False))
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].name, "pudding box")
        self.assertEqual(specs[0].usd_collision_approximation, "convexHull")


if __name__ == "__main__":
    unittest.main()
