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

    def test_near_singular_is_rejected_with_quality(self) -> None:
        with mock.patch(
            "src.robot.safety.singularity.analyze_joint_singularity",
            return_value=_report(near=True, cond=1.0e9, min_sv=1.0e-4),
        ):
            res = ArmBackedIKService(_IKArm()).query(_FakeGraspPose())
        self.assertFalse(res.reachable)
        self.assertEqual(res.reason, "near_singular")
        self.assertEqual(res.quality.condition_number, 1.0e9)

    def test_well_conditioned_is_reachable(self) -> None:
        with mock.patch(
            "src.robot.safety.singularity.analyze_joint_singularity",
            return_value=_report(near=False, cond=5.0, min_sv=0.5),
        ):
            res = ArmBackedIKService(_IKArm()).query(_FakeGraspPose())
        self.assertTrue(res.reachable)
        self.assertEqual(res.quality.min_singular_value, 0.5)

    def test_infinite_condition_number_becomes_none(self) -> None:
        with mock.patch(
            "src.robot.safety.singularity.analyze_joint_singularity",
            return_value=_report(near=True, cond=float("inf"), min_sv=0.0),
        ):
            res = ArmBackedIKService(_IKArm()).query(_FakeGraspPose())
        self.assertFalse(res.reachable)
        self.assertIsNone(res.quality.condition_number)  # inf -> None (JSON/validator-safe)


class _LimitArm:
    """Fake arm returning a fixed joint solution (rad) + UR-like capabilities (S3 joint-margin tests)."""

    def __init__(self, joints_rad) -> None:  # noqa: ANN001
        self._joints = np.asarray(joints_rad, dtype=np.float64)

    @property
    def capabilities(self):  # noqa: ANN201
        return SimpleNamespace(vendor="sim", model="isaac-sim", dof=len(self._joints))

    def ik(self, pose):  # noqa: ANN001
        return JointPositions(self._joints)


class RL3SubstrateTests(unittest.TestCase):
    """RL3: the failure-injection substrate that makes the service recovery loop fire (-> V6 data).
    Default-off byte-identical; engineers NO_CANDIDATES (oversized target) + occlusion failures."""

    @staticmethod
    def _specs() -> list:
        from src.config.schema.robot import SimObjectConfig
        return [
            SimObjectConfig(name="red cube", size_mm=(30.0, 30.0, 50.0), position_mm=(450.0, -95.0, 25.0)),
            SimObjectConfig(name="green cube", size_mm=(30.0, 30.0, 50.0), position_mm=(450.0, 5.0, 25.0)),
        ]

    def test_disabled_byte_identical(self) -> None:
        import os

        from src.willy_sim.run_dense_pick import _apply_rl3_substrate
        from src.willy_sim.harness.env import RunnerEnv

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WILLY_RL3_SUBSTRATE", None)
            out = _apply_rl3_substrate(self._specs(), 0, RunnerEnv.from_env(vision=False, view_height_mm=0.0))
        self.assertEqual(len(out), 2)
        self.assertEqual(tuple(out[0].size_mm), (30.0, 30.0, 50.0))  # untouched

    def test_unreachable_oversizes_target(self) -> None:
        import os

        from src.willy_sim.run_dense_pick import _apply_rl3_substrate
        from src.willy_sim.harness.env import RunnerEnv

        with mock.patch.dict(os.environ, {"WILLY_RL3_SUBSTRATE": "unreachable"}):
            out = _apply_rl3_substrate(self._specs(), 0, RunnerEnv.from_env(vision=False, view_height_mm=0.0))
        self.assertGreater(out[0].size_mm[0], 85.0)  # too wide for the 85mm gripper -> NO_CANDIDATES
        self.assertEqual(len(out), 2)                # no occluder for unreachable-only

    def test_occlusion_appends_occluder_above_target(self) -> None:
        import os

        from src.willy_sim.run_dense_pick import _apply_rl3_substrate
        from src.willy_sim.harness.env import RunnerEnv

        with mock.patch.dict(os.environ, {"WILLY_RL3_SUBSTRATE": "occlusion"}):
            out = _apply_rl3_substrate(self._specs(), 0, RunnerEnv.from_env(vision=False, view_height_mm=0.0))
        self.assertEqual(len(out), 3)
        self.assertEqual(out[-1].name, "occluder")
        self.assertGreater(out[-1].position_mm[2], out[0].position_mm[2])  # between camera + target

    def test_both_applies_both(self) -> None:
        import os

        from src.willy_sim.run_dense_pick import _apply_rl3_substrate
        from src.willy_sim.harness.env import RunnerEnv

        with mock.patch.dict(os.environ, {"WILLY_RL3_SUBSTRATE": "both"}):
            out = _apply_rl3_substrate(self._specs(), 0, RunnerEnv.from_env(vision=False, view_height_mm=0.0))
        self.assertGreater(out[0].size_mm[0], 85.0)
        self.assertEqual(out[-1].name, "occluder")


class RL1SceneFamilyIdTests(unittest.TestCase):
    """RL1: the coarse scene-FAMILY id (scene-kind + target identity) used for V3 pair-grouping + the
    leakage-safe group-aware split (distinct from the per-episode scene_id)."""

    def test_cube_vs_ycb_vs_gso_kind(self) -> None:
        from src.willy_sim.run_dense_pick import _scene_family_id

        self.assertEqual(_scene_family_id(kind="cube", target_label="red cube"), "cube:red_cube")
        self.assertEqual(_scene_family_id(kind="ycb", target_label="sugar box"), "ycb:sugar_box")
        self.assertEqual(_scene_family_id(kind="ycb", target_label="pudding box"), "ycb:pudding_box")
        # RL-diversity: a GSO real-object family carries its own kind so it never collides with a cube/ycb one
        self.assertEqual(_scene_family_id(kind="gso", target_label="toy rhino"), "gso:toy_rhino")

    def test_same_object_same_family_regardless_of_appearance(self) -> None:
        from src.willy_sim.run_dense_pick import _scene_family_id

        # same target object -> same family (scene2 / randomized appearance is WITHIN-family, not a leak)
        self.assertEqual(
            _scene_family_id(kind="ycb", target_label="sugar box"),
            _scene_family_id(kind="ycb", target_label="Sugar Box"),  # case/space-insensitive slug
        )


class S3JointMarginTests(unittest.TestCase):
    """S3: ArmBackedIKService fills IKQualityMetrics.joint_margin_deg = min over axes of
    min(q-lower, upper-q) — the SAME proximity the IKQualityGuard computes."""

    @staticmethod
    def _query(joints_rad, jl_config):  # noqa: ANN001, ANN205
        with mock.patch(
            "src.robot.safety.singularity.analyze_joint_singularity",
            return_value=_report(near=False, cond=5.0, min_sv=0.5),
        ):
            return ArmBackedIKService(
                _LimitArm(joints_rad), joint_limits_config=jl_config
            ).query(_FakeGraspPose())

    def test_none_without_config(self) -> None:
        res = self._query(np.zeros(6), None)  # no joint_limits_config -> legacy None
        self.assertTrue(res.reachable)
        self.assertIsNone(res.quality.joint_margin_deg)

    def test_margin_is_per_axis_minimum(self) -> None:
        from src.config.schema.robot import JointLimitSafetyConfig

        jl = JointLimitSafetyConfig(min_deg=[-90.0] * 6, max_deg=[90.0] * 6)
        res = self._query(np.zeros(6), jl)  # all joints at 0 deg -> margin = min(0-(-90), 90-0) = 90
        self.assertAlmostEqual(res.quality.joint_margin_deg, 90.0, places=4)

    def test_near_limit_gives_small_margin(self) -> None:
        from src.config.schema.robot import JointLimitSafetyConfig

        jl = JointLimitSafetyConfig(min_deg=[-90.0] * 6, max_deg=[90.0] * 6)
        joints = np.zeros(6)
        joints[3] = np.deg2rad(85.0)  # axis 3 at 85 deg -> 5 deg from the +90 limit -> the binding minimum
        res = self._query(joints, jl)
        self.assertAlmostEqual(res.quality.joint_margin_deg, 5.0, places=4)

    def test_past_limit_clamps_to_zero(self) -> None:
        from src.config.schema.robot import JointLimitSafetyConfig

        jl = JointLimitSafetyConfig(min_deg=[-90.0] * 6, max_deg=[90.0] * 6)
        joints = np.zeros(6)
        joints[0] = np.deg2rad(100.0)  # past the +90 limit -> negative margin -> clamped to 0 (carrier needs >=0)
        res = self._query(joints, jl)
        self.assertEqual(res.quality.joint_margin_deg, 0.0)

    def test_unresolvable_limits_yield_none(self) -> None:
        from src.config.schema.robot import JointLimitSafetyConfig

        jl = JointLimitSafetyConfig()  # no static table + vendor 'sim' is not in the UR table -> None
        res = self._query(np.zeros(6), jl)
        self.assertIsNone(res.quality.joint_margin_deg)


class S3FeasibilityDemotionTests(unittest.TestCase):
    """S3: when the feasibility joint_margin sub-score is ENABLED, a near-limit candidate (small margin)
    is demoted vs a well-clear control — proving the producer->carrier->scorer chain works end-to-end."""

    def test_near_limit_demoted_when_enabled(self) -> None:
        from src.robot.grasping.planning.reachability import IKQualityMetrics
        from src.robot.grasping.scoring.feasibility_score import (
            FeasibilityScoreConfig,
            feasibility_grasp_score,
        )

        cfg = FeasibilityScoreConfig(joint_margin_enabled=True)  # only joint_margin enabled
        near = feasibility_grasp_score(
            ik_quality=IKQualityMetrics(joint_margin_deg=5.0), approach_path=None, config=cfg
        )
        far = feasibility_grasp_score(
            ik_quality=IKQualityMetrics(joint_margin_deg=30.0), approach_path=None, config=cfg
        )
        self.assertLess(near, far)            # the near-limit grasp scores worse
        self.assertAlmostEqual(far, 1.0, places=4)   # >= 30 deg saturates the sub-score

    def test_disabled_is_neutral(self) -> None:
        from src.robot.grasping.planning.reachability import IKQualityMetrics
        from src.robot.grasping.scoring.feasibility_score import (
            FeasibilityScoreConfig,
            feasibility_score_components,
        )

        cfg = FeasibilityScoreConfig(joint_margin_enabled=False)  # default -> the metric does NOT bite
        comps = feasibility_score_components(
            ik_quality=IKQualityMetrics(joint_margin_deg=1.0), approach_path=None, config=cfg
        )
        self.assertEqual(comps["joint_margin"], 0.5)  # neutral floor -> gated leverage (honest)


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


class G6RecoveryTests(unittest.TestCase):
    """P3 G6: the recovery setup permits + SELECTS container-agitate on an ALL_COLLIDED clutter jam."""

    def test_build_g6_recovery_config(self) -> None:
        from src.willy_sim.run_dense_pick import _build_g6_recovery
        from src.robot.grasping.recovery.policy import SceneRecoveryAction

        prof, pol, orch = _build_g6_recovery(30.0)
        self.assertIn("container_agitate", prof.recovery_allowed_actions)  # custom profile opts in
        self.assertEqual(pol.allowed_actions, (SceneRecoveryAction.CONTAINER_AGITATE,))
        self.assertTrue(pol.enabled)
        self.assertEqual(pol.fixture.max_agitate_amplitude_mm, 30.0)  # fixture arms the motion
        self.assertFalse(orch.bypass_strategies)  # the real strategy carries the amplitude into the plan
        self.assertIn("dense_clutter", pol.apply_modes)

    def test_selects_agitate_for_all_collided(self) -> None:
        from src.willy_sim.run_dense_pick import _build_g6_recovery
        from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome
        from src.robot.grasping.types.feedback import GraspFailureReason
        from src.robot.grasping.recovery.policy import SceneRecoveryAction, SceneRecoveryContext

        prof, pol, orch = _build_g6_recovery(30.0)
        ctx = SceneRecoveryContext(
            profile=prof, policy=pol, last_outcome=list(AutonomousGraspOutcome)[0],
            failure_reasons=(GraspFailureReason.ALL_COLLIDED,),
        )
        plan = orch.next_step(ctx, typed_history=())
        self.assertIsNotNone(plan)
        self.assertEqual(plan.action, SceneRecoveryAction.CONTAINER_AGITATE)  # 4 gates pass -> selected
        self.assertEqual(plan.agitate_amplitude_mm, 30.0)  # the strategy carries the amplitude into the plan

    def test_no_recovery_without_a_failure(self) -> None:
        from src.willy_sim.run_dense_pick import _build_g6_recovery
        from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome
        from src.robot.grasping.recovery.policy import SceneRecoveryContext

        prof, pol, orch = _build_g6_recovery(30.0)
        ctx = SceneRecoveryContext(
            profile=prof, policy=pol, last_outcome=list(AutonomousGraspOutcome)[0], failure_reasons=(),
        )
        self.assertIsNone(orch.next_step(ctx, typed_history=()))

    def test_adapter_maps_approach_blocked_to_all_collided(self) -> None:
        from src.willy_sim.run_dense_pick import _G6RecoveryAdapter
        from src.robot.grasping.types.feedback import GraspFailureReason

        pr = SimpleNamespace(outcome=SimpleNamespace(value="approach_path_blocked"),
                             attempts=(SimpleNamespace(reasons=()),))
        report = SimpleNamespace(outcome=SimpleNamespace(value="execution_failed"), pick_report=pr)
        self.assertIn(GraspFailureReason.ALL_COLLIDED, _G6RecoveryAdapter(report).failure_reasons)

    def test_adapter_leaves_non_blocked_outcomes_untouched(self) -> None:
        from src.willy_sim.run_dense_pick import _G6RecoveryAdapter

        pr = SimpleNamespace(outcome=SimpleNamespace(value="succeeded"),
                             attempts=(SimpleNamespace(reasons=()),))
        report = SimpleNamespace(outcome=SimpleNamespace(value="succeeded"), pick_report=pr)
        self.assertEqual(_G6RecoveryAdapter(report).failure_reasons, ())


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


class FailureTaxonomyFromSimTests(unittest.TestCase):
    """P4.4: the SIM-ground-truth -> U7 failure-taxonomy mapping that makes the collected dataset's outcome
    class HONEST (derive_outcome_class reads extra.failure_taxonomy_class first), not the pipeline's
    'succeeded'-even-on-slip self-report."""

    def test_real_lift_is_success_none(self) -> None:
        from src.willy_sim.run_dense_pick import _failure_taxonomy_from_sim

        self.assertIsNone(_failure_taxonomy_from_sim("succeeded", "executed", True))

    def test_slip_executed_but_no_lift(self) -> None:
        from src.willy_sim.run_dense_pick import _failure_taxonomy_from_sim

        # non-detecting gripper reports 'succeeded' but sim_lifted=False -> a SLIP (a negative label)
        self.assertEqual(
            _failure_taxonomy_from_sim("succeeded", "executed", False), "slip_after_grasp"
        )

    def test_no_valid_grasp_is_empty_air(self) -> None:
        from src.willy_sim.run_dense_pick import _failure_taxonomy_from_sim

        self.assertEqual(
            _failure_taxonomy_from_sim("no_valid_grasp", "rescanned_exhausted", False), "empty_air_grasp"
        )

    def test_collision_outcome_maps_to_collision_rejection(self) -> None:
        from src.willy_sim.run_dense_pick import _failure_taxonomy_from_sim

        self.assertEqual(
            _failure_taxonomy_from_sim("collision_detected", "executed", False), "collision_rejection"
        )


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
            for spec in _ycb_specs(False, RunnerEnv.from_env(vision=False, view_height_mm=0.0)):
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
            specs = _ycb_specs(False, RunnerEnv.from_env(vision=False, view_height_mm=0.0))
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].name, "pudding box")
        self.assertEqual(specs[0].usd_collision_approximation, "convexHull")


if __name__ == "__main__":
    unittest.main()
