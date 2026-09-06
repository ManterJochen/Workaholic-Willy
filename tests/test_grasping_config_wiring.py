"""Phase T0 — runtime wiring tests for ``robot.grasping`` config.

These tests pin the locked T0 contract (operator-confirmed):

* ``AutonomousGraspService.from_robot_config`` resolves the
  :class:`GraspMode` and ``max_attempts`` from
  ``robot_cfg.grasping`` when the caller does not pass explicit
  overrides.
* The factory **fail-closes** when ``mode`` is not supplied *and* the
  ``grasping`` block was not explicitly set on the ``RobotConfig`` (i.e.
  the operator relied on schema defaults silently). Production
  deployments must declare the block explicitly.
* Unknown mode strings and unknown recovery action names raise.
* Sub-policies (``RefinementPolicy``, ``GraspVerificationPolicy``,
  ``SceneRecoveryPolicy``) are auto-built from their schema sub-blocks
  when the operator did not pass explicit instances *and* the sub-block
  is enabled. Disabled sub-blocks leave the slot as :data:`None` so the
  honest ``MODE_NOT_AVAILABLE`` gates continue to fire.
* Physical recovery actions requested by config without a caller-
  supplied :class:`FixtureEnvelope` are refused (strict fail-closed).
* Every :class:`AutonomousGraspReport` carries an
  :class:`EffectiveGraspingConfig` snapshot when the service was wired
  from config; the legacy ``from_components`` path leaves it ``None``
  (feature-flag compatibility per T0 deliverable #4).

The test rig mirrors :mod:`tests.test_autonomous_grasp_service` to
avoid spawning a parallel doubles tree.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.robot.core import (
    MotionCommand,
    MotionResult,
    RobotCapabilities,
)
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspService,
    EffectiveGraspingConfig,
    GraspMode,
)
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.modes import GraspSamplingMode
from src.robot.grasping.telemetry.outcome_logging import iter_jsonl
from src.robot.grasping.types.perception import PerceptionFrame
from src.robot.grasping.recovery.policy import (
    FixtureEnvelope,
    SceneRecoveryAction,
)


# ---------------------------------------------------------------------------
# Test doubles (mirrors tests/test_autonomous_grasp_service.py)
# ---------------------------------------------------------------------------


_REAL_UR_CAPS = RobotCapabilities(
    vendor="ur",
    model="ur5e",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=False,
    has_native_fk=True,
    has_native_ik=True,
    has_force_control=False,
    is_simulated=False,
)


class _TypedFakeArm:
    def __init__(self) -> None:
        self._tcp = Pose(
            position_mm=np.array([0.0, 0.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
            label="home",
        )

    @property
    def capabilities(self) -> RobotCapabilities:
        return _REAL_UR_CAPS

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        if pose.frame == Frame.BASE:
            self._tcp = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


class _FakePerception:
    def __init__(self, frames: list[PerceptionFrame]) -> None:
        self._frames = frames
        self.calls = 0

    def acquire(self) -> PerceptionFrame:
        idx = min(self.calls, len(self._frames) - 1)
        self.calls += 1
        return self._frames[idx]


class _ScriptedCalculator:
    def __init__(self, results: list[GraspResult]) -> None:
        self._results = results
        self.calls = 0

    def compute_result(self, *_args: object, **_kwargs: object) -> GraspResult:
        idx = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[idx]


def _segmentation() -> SimpleNamespace:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 1
    return SimpleNamespace(mask=mask)


def _perception_frame() -> PerceptionFrame:
    depth = np.full((32, 32), 500.0, dtype=np.float64)
    intrinsics = np.array(
        [[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return PerceptionFrame(
        depth_map=depth,
        intrinsics=intrinsics,
        segmentations=(_segmentation(),),
    )


def _success_result() -> GraspResult:
    return GraspResult(
        candidates=(
            GraspPoint(
                position=np.array([100.0, 50.0, 400.0]),
                approach=np.array([0.0, 0.0, 1.0]),
                axis=np.array([1.0, 0.0, 0.0]),
                grip_width_mm=40.0,
                score=0.9,
                frame=GraspFrame.BASE,
                label="test",
            ),
        ),
        reasons=(),
        top_score=0.9,
    )


def _calc_and_perception():
    return (
        _ScriptedCalculator([_success_result()]),
        _FakePerception([_perception_frame()]),
    )


def _robot_cfg_with_grasping(**grasping_overrides):
    """Build a ``RobotConfig`` whose ``grasping`` block is *explicitly* set.

    The operator-confirmed T0 contract treats absence of an explicit
    ``grasping`` block as a misconfiguration on production hosts. The
    helper takes the schema default and re-supplies it so
    ``'grasping' in cfg.model_fields_set`` evaluates ``True``.
    """

    return RobotConfig(
        vendor="dummy",
        gripper={"vendor": "none"},
        grasping={
            "default_mode": grasping_overrides.pop("default_mode", "auto"),
            "max_attempts": grasping_overrides.pop("max_attempts", 5),
            "record_log_path": grasping_overrides.pop("record_log_path", None),
            "closed_loop": grasping_overrides.pop("closed_loop", {}),
            "verification": grasping_overrides.pop("verification", {}),
            "dense_recovery": grasping_overrides.pop("dense_recovery", {}),
        },
    )


# ---------------------------------------------------------------------------
# EffectiveGraspingConfig dataclass tests
# ---------------------------------------------------------------------------


class EffectiveGraspingConfigTests(unittest.TestCase):
    """The T0 snapshot dataclass must be frozen + JSON-round-trip friendly."""

    def test_is_frozen(self) -> None:
        snap = EffectiveGraspingConfig(
            default_mode=GraspMode.AUTO,
            max_attempts=5,
            closed_loop_enabled=False,
            verification_enabled=False,
            dense_recovery_enabled=False,
            dense_recovery_allowed_actions=(),
        )
        with self.assertRaises((AttributeError, TypeError)):
            snap.default_mode = GraspMode.EASY  # type: ignore[misc]

    def test_to_dict_is_json_safe(self) -> None:
        snap = EffectiveGraspingConfig(
            default_mode=GraspMode.DENSE_AUTONOMOUS,
            max_attempts=7,
            closed_loop_enabled=True,
            verification_enabled=True,
            dense_recovery_enabled=True,
            dense_recovery_allowed_actions=("next_viewpoint", "nudge_target"),
        )
        d = snap.to_dict()
        self.assertEqual(d["default_mode"], "dense_autonomous")
        self.assertEqual(d["max_attempts"], 7)
        self.assertTrue(d["closed_loop_enabled"])
        self.assertTrue(d["verification_enabled"])
        self.assertTrue(d["dense_recovery_enabled"])
        self.assertEqual(
            d["dense_recovery_allowed_actions"],
            ["next_viewpoint", "nudge_target"],
        )


# ---------------------------------------------------------------------------
# from_robot_config mode resolution
# ---------------------------------------------------------------------------


class FromRobotConfigModeResolutionTests(unittest.TestCase):
    """``default_mode`` and ``max_attempts`` flow from config into the service."""

    def test_default_mode_easy_resolves_easy_profile(self) -> None:
        cfg = _robot_cfg_with_grasping(default_mode="easy")
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        self.assertIs(service.default_mode, GraspMode.EASY)
        self.assertIs(
            service.runtime.orchestrator.grasp_sampling_mode,
            GraspSamplingMode.SINGLE_OBJECT,
        )

    def test_default_mode_dense_clutter(self) -> None:
        cfg = _robot_cfg_with_grasping(default_mode="dense_clutter")
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        self.assertIs(service.default_mode, GraspMode.DENSE_CLUTTER)
        self.assertIs(
            service.runtime.orchestrator.grasp_sampling_mode,
            GraspSamplingMode.DENSE_CLUTTER,
        )

    def test_max_attempts_pulled_from_config(self) -> None:
        cfg = _robot_cfg_with_grasping(default_mode="auto", max_attempts=9)
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        self.assertEqual(service.runtime.orchestrator.max_attempts, 9)

    def test_explicit_mode_overrides_config(self) -> None:
        cfg = _robot_cfg_with_grasping(default_mode="easy")
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            mode=GraspMode.AUTO,
        )
        self.assertIs(service.default_mode, GraspMode.AUTO)

    def test_explicit_max_attempts_overrides_config(self) -> None:
        cfg = _robot_cfg_with_grasping(default_mode="auto", max_attempts=9)
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            max_attempts=3,
        )
        self.assertEqual(service.runtime.orchestrator.max_attempts, 3)


# ---------------------------------------------------------------------------
# Strict fail-closed: missing / malformed grasping block
# ---------------------------------------------------------------------------


class FromRobotConfigFailClosedTests(unittest.TestCase):
    """Production YAML must declare the grasping block explicitly."""

    def test_missing_grasping_block_raises_when_mode_not_passed(self) -> None:
        # ``RobotConfig`` provides a schema default for ``grasping``, but
        # the operator's locked T0 stance treats reliance on that default
        # as a misconfiguration when ``mode`` is also unspecified.
        cfg = RobotConfig(vendor="dummy", gripper={"vendor": "none"})
        calc, perception = _calc_and_perception()
        with self.assertRaises(ValueError) as ctx:
            AutonomousGraspService.from_robot_config(
                cfg,
                calculator=calc,  # type: ignore[arg-type]
                perception=perception,
            )
        msg = str(ctx.exception)
        self.assertIn("grasping", msg)

    def test_missing_grasping_block_ok_when_mode_passed_explicitly(self) -> None:
        # Test rigs that pin a mode directly should not be forced to
        # populate the full grasping block.
        cfg = RobotConfig(vendor="dummy", gripper={"vendor": "none"})
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            mode=GraspMode.AUTO,
        )
        self.assertIs(service.default_mode, GraspMode.AUTO)
        # Without a config-driven build there is no effective snapshot.
        self.assertIsNone(service.effective_config)

    def test_unknown_default_mode_string_raises(self) -> None:
        cfg = _robot_cfg_with_grasping(default_mode="turbo")
        calc, perception = _calc_and_perception()
        with self.assertRaises(ValueError):
            AutonomousGraspService.from_robot_config(
                cfg,
                calculator=calc,  # type: ignore[arg-type]
                perception=perception,
            )


# ---------------------------------------------------------------------------
# Sub-policy auto-build
# ---------------------------------------------------------------------------


class FromRobotConfigSubPolicyAutoBuildTests(unittest.TestCase):
    """Enabled sub-blocks are materialised into policy objects."""

    def test_disabled_closed_loop_leaves_refinement_policy_none(self) -> None:
        cfg = _robot_cfg_with_grasping(default_mode="auto")
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        self.assertIsNone(service.refinement_policy)

    def test_enabled_closed_loop_builds_refinement_policy(self) -> None:
        cfg = _robot_cfg_with_grasping(
            default_mode="closed_loop",
            closed_loop={
                "enabled": True,
                "pregrasp_rescan": True,
                "max_position_correction_mm": 12.0,
                "max_orientation_correction_deg": 8.0,
                "max_grip_width_correction_mm": 7.0,
                "target_match_iou_threshold": 0.4,
            },
        )
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        self.assertIsNotNone(service.refinement_policy)
        rp = service.refinement_policy
        assert rp is not None  # for type checker
        self.assertTrue(rp.enabled)
        self.assertEqual(rp.max_position_correction_mm, 12.0)
        self.assertEqual(rp.max_orientation_correction_deg, 8.0)
        self.assertEqual(rp.max_grip_width_correction_mm, 7.0)
        self.assertEqual(rp.target_match_iou_threshold, 0.4)

    def test_enabled_verification_builds_policy(self) -> None:
        cfg = _robot_cfg_with_grasping(
            default_mode="auto",
            verification={
                "enabled": True,
                "require_object_detected": True,
                "width_delta_min_mm": 3.0,
                "post_lift_vision_check": False,
                "vision_displacement_iou_max": 0.15,
                "fail_closed": False,
            },
        )
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        vp = service.verification_policy
        self.assertIsNotNone(vp)
        assert vp is not None
        self.assertTrue(vp.enabled)
        self.assertTrue(vp.require_object_detected)
        self.assertEqual(vp.width_delta_min_mm, 3.0)
        self.assertFalse(vp.fail_closed)

    def test_enabled_non_physical_recovery_builds_policy(self) -> None:
        cfg = _robot_cfg_with_grasping(
            default_mode="dense_clutter",
            dense_recovery={
                "enabled": True,
                "allowed_actions": ["rescan", "next_viewpoint"],
                "max_recovery_actions": 3,
            },
        )
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        rp = service.recovery_policy
        self.assertIsNotNone(rp)
        assert rp is not None
        self.assertTrue(rp.enabled)
        self.assertEqual(
            rp.allowed_actions,
            (SceneRecoveryAction.RESCAN, SceneRecoveryAction.NEXT_VIEWPOINT),
        )
        self.assertEqual(rp.max_recovery_actions, 3)

    def test_physical_recovery_without_fixture_raises(self) -> None:
        cfg = _robot_cfg_with_grasping(
            default_mode="dense_autonomous",
            dense_recovery={
                "enabled": True,
                "allowed_actions": ["next_viewpoint", "nudge_target"],
            },
        )
        calc, perception = _calc_and_perception()
        with self.assertRaises(ValueError) as ctx:
            AutonomousGraspService.from_robot_config(
                cfg,
                calculator=calc,  # type: ignore[arg-type]
                perception=perception,
            )
        self.assertIn("fixture", str(ctx.exception).lower())

    def test_physical_recovery_with_fixture_builds_policy(self) -> None:
        cfg = _robot_cfg_with_grasping(
            default_mode="dense_autonomous",
            dense_recovery={
                "enabled": True,
                "allowed_actions": ["next_viewpoint", "nudge_target"],
            },
        )
        fixture = FixtureEnvelope(
            center_mm=(0.0, 0.0, 200.0),
            half_extents_mm=(300.0, 300.0, 100.0),
            max_nudge_mm=8.0,
        )
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            recovery_fixture=fixture,
        )
        rp = service.recovery_policy
        self.assertIsNotNone(rp)
        assert rp is not None
        self.assertIn(SceneRecoveryAction.NUDGE_TARGET, rp.allowed_actions)
        self.assertIs(rp.fixture, fixture)

    def test_unknown_recovery_action_raises(self) -> None:
        cfg = _robot_cfg_with_grasping(
            default_mode="dense_clutter",
            dense_recovery={
                "enabled": True,
                "allowed_actions": ["bogus_action"],
            },
        )
        calc, perception = _calc_and_perception()
        with self.assertRaises(ValueError) as ctx:
            AutonomousGraspService.from_robot_config(
                cfg,
                calculator=calc,  # type: ignore[arg-type]
                perception=perception,
            )
        self.assertIn("bogus_action", str(ctx.exception))

    def test_explicit_refinement_policy_wins_over_config(self) -> None:
        # The caller's explicit object must not be silently rebuilt
        # from config — that's the feature-flag compatibility path
        # locked at T0.
        from src.robot.grasping.closed_loop.refinement import RefinementPolicy

        cfg = _robot_cfg_with_grasping(
            default_mode="closed_loop",
            closed_loop={"enabled": True, "max_position_correction_mm": 12.0},
        )
        explicit = RefinementPolicy(
            enabled=True, max_position_correction_mm=42.0
        )
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            refinement_policy=explicit,
        )
        self.assertIs(service.refinement_policy, explicit)


# ---------------------------------------------------------------------------
# Effective config snapshot
# ---------------------------------------------------------------------------


class EffectiveConfigSnapshotTests(unittest.TestCase):
    """The effective config must round-trip through the report."""

    def test_service_carries_effective_config_when_built_from_config(self) -> None:
        cfg = _robot_cfg_with_grasping(default_mode="auto", max_attempts=4)
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        snap = service.effective_config
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertIs(snap.default_mode, GraspMode.AUTO)
        self.assertEqual(snap.max_attempts, 4)
        self.assertFalse(snap.closed_loop_enabled)
        self.assertFalse(snap.verification_enabled)
        self.assertFalse(snap.dense_recovery_enabled)
        # The snapshot faithfully mirrors the schema, including the
        # ``allowed_actions`` default. We assert ``enabled is False``
        # is what gates behaviour --- the allow-list value is
        # informational when the block is disabled.
        self.assertEqual(snap.dense_recovery_allowed_actions, ("next_viewpoint",))

    def test_report_contains_effective_config(self) -> None:
        cfg = _robot_cfg_with_grasping(
            default_mode="auto",
            max_attempts=2,
            dense_recovery={
                "enabled": True,
                "allowed_actions": ["rescan", "next_viewpoint"],
            },
        )
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        report = service.pick()
        # We don't pin the outcome here — the dummy arm built by
        # from_robot_config may or may not execute the grasp depending
        # on driver behaviour. The contract is: the snapshot must
        # survive into the report regardless of outcome.
        assert report.effective_config is not None
        self.assertIs(report.effective_config.default_mode, GraspMode.AUTO)
        self.assertEqual(report.effective_config.max_attempts, 2)
        self.assertTrue(report.effective_config.dense_recovery_enabled)
        self.assertEqual(
            report.effective_config.dense_recovery_allowed_actions,
            ("rescan", "next_viewpoint"),
        )

    def test_from_components_leaves_effective_config_none(self) -> None:
        # Feature-flag compatibility path: legacy callers do not
        # opt into config-driven wiring and must not pay for it.
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.AUTO,
        )
        self.assertIsNone(service.effective_config)
        report = service.pick()
        self.assertIsNone(report.effective_config)


class FromRobotConfigRecordLoggingTests(unittest.TestCase):
    """H0.1 — ``grasping.record_log_path`` opts the prod (from_robot_config) path into JSONL logging."""

    def test_config_path_logs_one_record_per_pick_with_unique_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "sub" / "pick.jsonl"  # parent dir is created on first append
            cfg = _robot_cfg_with_grasping(default_mode="easy", record_log_path=str(log))
            calc, perception = _calc_and_perception()
            service = AutonomousGraspService.from_robot_config(
                cfg,
                calculator=calc,  # type: ignore[arg-type]
                perception=perception,
            )
            service.pick()
            service.pick()
            recs = tuple(iter_jsonl(log))
        self.assertEqual(len(recs), 2)
        ids = {r.attempt_id for r in recs}
        # The H0.1a fix: a unique id per pick (the default path no longer collides on "pick").
        self.assertEqual(len(ids), 2)
        for r in recs:
            self.assertTrue(r.attempt_id and r.attempt_id != "pick")
            self.assertTrue(r.attempt_id.startswith(("pick-", "auto-")))

    def test_no_config_path_logs_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "pick.jsonl"
            cfg = _robot_cfg_with_grasping(default_mode="easy")  # record_log_path defaults None
            calc, perception = _calc_and_perception()
            service = AutonomousGraspService.from_robot_config(
                cfg,
                calculator=calc,  # type: ignore[arg-type]
                perception=perception,
            )
            service.pick()
            self.assertFalse(log.exists())

    def test_empty_string_path_is_off(self) -> None:
        # ``record_log_path: "${WILLY_RECORD_LOG:-}"`` with an unset env collapses to "" -> off.
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "pick.jsonl"
            cfg = _robot_cfg_with_grasping(default_mode="easy", record_log_path="")
            calc, perception = _calc_and_perception()
            service = AutonomousGraspService.from_robot_config(
                cfg,
                calculator=calc,  # type: ignore[arg-type]
                perception=perception,
            )
            service.pick()
            self.assertFalse(log.exists())


class FromRobotConfigGripperFallbackWarnsTests(unittest.TestCase):
    """H0.3a — from_robot_config WARNS (not silently) when it falls back to a NullGripper."""

    _LOGGER = "src.robot.execution.runtime_pick"

    @staticmethod
    def _cfg(gripper_vendor: str) -> RobotConfig:
        return RobotConfig(
            vendor="dummy",
            gripper={"vendor": gripper_vendor},
            grasping={"default_mode": "easy"},
        )

    def test_robotiq_on_non_ur_arm_warns(self) -> None:
        # The Isaac-sim shape (gripper.vendor=robotiq on a non-UR arm) cannot build a real gripper
        # through from_robot_config -> NullGripper, now with an explicit WARNING.
        calc, perception = _calc_and_perception()
        with self.assertLogs(self._LOGGER, level="WARNING") as cm:
            AutonomousGraspService.from_robot_config(
                self._cfg("robotiq"),
                calculator=calc,  # type: ignore[arg-type]
                perception=perception,
            )
        self.assertTrue(any("NullGripper" in m and "robotiq" in m for m in cm.output))

    def test_explicit_none_gripper_is_silent(self) -> None:
        # An explicit ``gripper.vendor=none`` is intentional -> no warning.
        calc, perception = _calc_and_perception()
        with self.assertNoLogs(self._LOGGER, level="WARNING"):
            AutonomousGraspService.from_robot_config(
                self._cfg("none"),
                calculator=calc,  # type: ignore[arg-type]
                perception=perception,
            )

    def _gripper_for(self, vendor: str) -> object:
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            self._cfg(vendor),
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        return getattr(service.runtime.orchestrator, "gripper", None)

    def test_the_substitution_is_readable_from_the_built_service_not_only_from_the_log(self) -> None:
        """A warning is for a human reading a terminal. A server has neither a terminal nor a human.

        This is the most dangerous silent state in the build path: every one of these configs produces
        a WORKING ``NullGripper``, so the cell connects, each pick reports success, and the jaws close
        on nothing. The console has to be able to refuse -- which means the reason has to be an object,
        not a log line it would otherwise have to parse.
        """
        from src.robot.grippers import SubstitutionReason

        expected = {
            "robotiq": SubstitutionReason.ROBOTIQ_NEEDS_UR,
            "vacuum": SubstitutionReason.VACUUM_NEEDS_DIGITAL_IO,
            "franka_hand": SubstitutionReason.NO_DRIVER,
        }
        for vendor, reason in expected.items():
            with self.subTest(vendor=vendor):
                substitution = getattr(self._gripper_for(vendor), "substitution", None)
                self.assertIsNotNone(substitution, f"{vendor} fell back with no recorded reason")
                assert substitution is not None
                self.assertIs(substitution.reason, reason)
                self.assertEqual(substitution.requested, vendor)
                # Both halves are for an operator: what is wrong, and what to do about it.
                self.assertIn(vendor, substitution.detail)
                self.assertTrue(substitution.fix.strip(), f"{vendor} states no fix")

    def test_a_cell_with_no_end_effector_on_purpose_carries_no_substitution(self) -> None:
        """The distinction the console renders on. Without it the banner cries wolf on every sim cell,
        and a banner that cries wolf is worse than no banner."""
        self.assertIsNone(getattr(self._gripper_for("none"), "substitution", None))

    def test_an_unknown_vendor_never_reaches_the_fallback_because_the_schema_refuses_it_first(
        self,
    ) -> None:
        """Measured, and better than the fallback it makes unreachable.

        ``from_robot_config`` carries an ``UNKNOWN_VENDOR`` substitution branch for a
        ``gripper.vendor`` string it does not recognise. No validated config can reach it: the schema
        rejects the value at load, naming every accepted alternative. So the branch is defence in depth
        against a hand-built ``model_construct`` config, not a state an operator can produce -- and a
        typo is caught where it is cheapest, before anything is built at all.
        """
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._cfg("nonesuch")
        message = str(ctx.exception)
        self.assertIn("unknown gripper vendor", message)
        self.assertIn("robotiq", message)  # the refusal lists what IS accepted


class FromRobotConfigOverlaysFireTests(unittest.TestCase):
    """H0.3b — the effective_config-gated overlays (T1 decision / U8 uncertainty / U9 watchdog / U10 SLO /
    T5 recovery) are ACTIVATED through from_robot_config — the path that from_components SILENCES by leaving
    ``effective_config=None`` (see ``test_from_components_leaves_effective_config_none`` for the contrast).
    This closes the deep-map BUILT_DEAD concern: the prod composition root's overlays are proven to wire +
    fire end-to-end through the config-driven build, not only via from_components + manual injection."""

    @staticmethod
    def _full_stack_cfg() -> RobotConfig:
        return RobotConfig(
            vendor="dummy",
            gripper={"vendor": "none"},
            grasping={
                "default_mode": "auto",
                "decision": {"enabled": True},
                "recovery": {"enabled": True},
                "uncertainty": {"enabled": True},
                "watchdog": {"mode": "active"},
                "performance": {"enabled": True},
            },
        )

    def test_effective_config_overlays_activated(self) -> None:
        calc, perception = _calc_and_perception()
        svc = AutonomousGraspService.from_robot_config(
            self._full_stack_cfg(),
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        ec = svc.effective_config
        self.assertIsNotNone(ec)
        assert ec is not None  # type-narrow for the attribute reads below
        self.assertTrue(ec.decision.enabled, "T1 decision overlay not activated via from_robot_config")
        self.assertTrue(ec.recovery_orchestrator.enabled, "T5 recovery overlay not activated")
        self.assertTrue(ec.uncertainty.enabled, "U8 uncertainty overlay not activated")
        self.assertEqual(ec.watchdog.mode, "active", "U9 watchdog overlay not activated via from_robot_config")
        self.assertTrue(ec.performance.enabled, "U10 SLO overlay not activated")

    def test_decision_overlay_fires_end_to_end(self) -> None:
        # Through the CONFIG-DRIVEN build (not from_components + manual injection like the T1 unit tests):
        # an AUTO pick with a confident grasp runs the decision gate and populates report.decision.
        cfg = RobotConfig(
            vendor="dummy",
            gripper={"vendor": "none"},
            grasping={
                "default_mode": "auto",
                "decision": {"enabled": True},
                "uncertainty": {"enabled": True},
            },
        )
        calc, perception = _calc_and_perception()
        report = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        ).pick()
        # The AUTO decision gate RAN through the config-driven build: report.decision is populated and the
        # attempt id is the decision path's ``auto-<uuid>``. (The terminal outcome depends on the in-memory
        # dummy arm's execution and is deliberately not what this test pins.)
        self.assertIsNotNone(report.decision, "the AUTO decision gate did not run through from_robot_config")
        self.assertTrue(str(report.telemetry.get("attempt_id", "")).startswith("auto-"))


class FromRobotConfigRerankOverlayTests(unittest.TestCase):
    """H2.1a — the G4 per-candidate uncertainty-rerank CONSUMER (``uncertainty.rerank_enabled``) is WIRED onto
    the orchestrator through from_robot_config. This is the prod surface of the corridor-risk rerank: the
    consumer config-plumb has existed since G4/gap-closure but is reachable ONLY via from_robot_config
    (apply_orchestrator_overlays) — the live sim-runner dense pick uses from_components, which leaves the slot
    ``None``. The H2.1 deep-read found the rerank had NEVER fired end-to-end on any live path; this test pins
    that the PROD path at least WIRES the consumer (the off-box reorder math itself is proven by
    ``tests/test_g4_uncertainty_rerank.py::test_uncertainty_flips_order_so_runner_up_wins``). Mirrors the H0.3b
    overlay-fires tests above."""

    @staticmethod
    def _cfg(*, rerank_enabled: bool, rerank_weight: float = 0.2) -> RobotConfig:
        return RobotConfig(
            vendor="dummy",
            gripper={"vendor": "none"},
            grasping={
                "default_mode": "dense_clutter",
                "uncertainty": {
                    "rerank_enabled": rerank_enabled,
                    "rerank_weight": rerank_weight,
                },
            },
        )

    def test_rerank_overlay_wired_when_enabled(self) -> None:
        calc, perception = _calc_and_perception()
        svc = AutonomousGraspService.from_robot_config(
            self._cfg(rerank_enabled=True, rerank_weight=0.2),
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        cfg = svc.runtime.orchestrator.uncertainty_rerank_config
        self.assertIsNotNone(cfg, "G4 rerank consumer not wired via from_robot_config")
        assert cfg is not None  # type-narrow for the attribute reads below
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.weight, 0.2)
        self.assertEqual(cfg.modes, ("dense_clutter", "dense_autonomous"))

    def test_rerank_overlay_none_when_disabled(self) -> None:
        # Default-off (rerank_enabled absent) -> the slot stays ``None`` -> byte-identical no-op.
        calc, perception = _calc_and_perception()
        svc = AutonomousGraspService.from_robot_config(
            self._cfg(rerank_enabled=False),
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        self.assertIsNone(svc.runtime.orchestrator.uncertainty_rerank_config)


class FromRobotConfigApproachValidationOverlayTests(unittest.TestCase):
    """H2.2a — the G12 swept-volume approach-validation overlay (``approach_validation.enabled``) is WIRED onto
    the orchestrator through from_robot_config: ``approach_path_policy`` is set + ``_approach_path_modes`` is
    populated, so the dense-mode per-pick gate (``mode_label in _approach_path_modes``) can fire. Like the H2.1
    rerank consumer, this config-plumb is reachable ONLY via from_robot_config (apply_orchestrator_overlays) —
    the live sim-runner dense pick uses from_components + a manual ``--g12`` owner-set, which is why the
    on-box refuse proof runs through run_dense_pick, not this prod path. The validator's refuse / fall-back /
    no-false-refusal LOGIC is pinned off-box by ``tests/test_pick_loop.py``; this test pins only that the PROD
    path WIRES it (the apply_orchestrator_overlays seam neither test_pick_loop nor the seam0 schema golden
    cover). Mirrors the H0.3b / H2.1a overlay-fires tests above."""

    @staticmethod
    def _cfg(*, enabled: bool) -> RobotConfig:
        return RobotConfig(
            vendor="dummy",
            gripper={"vendor": "none"},
            grasping={
                "default_mode": "dense_clutter",
                "approach_validation": {"enabled": enabled},
            },
        )

    def test_overlay_wired_when_enabled(self) -> None:
        calc, perception = _calc_and_perception()
        svc = AutonomousGraspService.from_robot_config(
            self._cfg(enabled=True),
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        orch = svc.runtime.orchestrator
        self.assertIsNotNone(
            orch.approach_path_policy, "G12 approach validator not wired via from_robot_config"
        )
        self.assertEqual(
            orch._approach_path_modes, frozenset({"dense_clutter", "dense_autonomous"})
        )

    def test_overlay_off_when_disabled(self) -> None:
        # Default-off -> no policy + empty modes -> the gate is False -> byte-identical legacy execute path.
        calc, perception = _calc_and_perception()
        svc = AutonomousGraspService.from_robot_config(
            self._cfg(enabled=False),
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        orch = svc.runtime.orchestrator
        self.assertIsNone(orch.approach_path_policy)
        self.assertEqual(orch._approach_path_modes, frozenset())


class FromRobotConfigFusionCommitOverlayTests(unittest.TestCase):
    """H2.3a — the U5 multi-view fusion substrate (``orch.scene_fusion``) + the U6 commit-gate policy
    (``orch.commit_policy``) are WIRED onto the orchestrator through from_robot_config when
    ``grasping.fusion.enabled`` (+ a loadable ``extrinsics_artifact_path`` so the CAMERA→BASE frame resolver
    auto-builds) and ``grasping.fusion.commit_policy.enabled``. Like the H2.1/H2.2 overlays this config-plumb
    is reachable ONLY via from_robot_config (no live caller), AND it is **structurally inert on a fixed camera**
    — the commit gate needs a MOVING (eye-in-hand) camera to accumulate diverse-viewpoint evidence, so the
    on-box diverse-vs-duplicate proof is ``run_commit_gate`` (EIH), NOT the fixed-overhead H2 gate. The helper
    ``build_config_frame_resolver`` + its fail-closed contract are pinned by ``tests/test_g5_fusion_resolver.py``;
    the substrate + gate logic by ``test_u5_*`` / ``test_u6_*``. This pins only the end-to-end overlay wiring
    (the orch carriers) — the one seam none of those cover. Mirrors the H2.1a / H2.2a overlay-fires tests."""

    @staticmethod
    def _cfg(*, fusion_enabled: bool, extrinsics_path: str | None, commit_enabled: bool) -> RobotConfig:
        fusion: dict = {"enabled": fusion_enabled, "commit_policy": {"enabled": commit_enabled}}
        if extrinsics_path is not None:
            fusion["extrinsics_artifact_path"] = extrinsics_path
        return RobotConfig(
            vendor="dummy",
            gripper={"vendor": "none"},
            grasping={"default_mode": "dense_clutter", "fusion": fusion},
        )

    @staticmethod
    def _write_extrinsics(directory: str) -> str:
        # A valid persisted eye-to-hand Extrinsics artifact (identity CAMERA->BASE) so the config frame
        # resolver auto-builds -- the same save_extrinsics pattern test_g5_fusion_resolver uses.
        from datetime import datetime, timezone
        from pathlib import Path

        from src.calibration.extrinsics import Extrinsics
        from src.calibration.serialization import save_extrinsics
        from src.geometry import Frame, Transform

        artifact = Path(directory) / "eth_extrinsics.json"
        save_extrinsics(
            artifact,
            Extrinsics(
                transform=Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE),
                rmse_mm=1.0,
                max_error_mm=2.0,
                num_samples=10,
                captured_at=datetime.now(timezone.utc),
                rig_id="h2_3_test_rig",
            ),
        )
        return str(artifact)

    def test_fusion_and_commit_overlays_wired(self) -> None:
        import tempfile

        calc, perception = _calc_and_perception()
        with tempfile.TemporaryDirectory() as tmp:
            svc = AutonomousGraspService.from_robot_config(
                self._cfg(
                    fusion_enabled=True,
                    extrinsics_path=self._write_extrinsics(tmp),
                    commit_enabled=True,
                ),
                calculator=calc,  # type: ignore[arg-type]
                perception=perception,
            )
        orch = svc.runtime.orchestrator
        self.assertIsNotNone(orch.scene_fusion, "U5 fusion substrate not wired via from_robot_config")
        self.assertIsNotNone(orch.commit_policy, "U6 commit policy not wired via from_robot_config")

    def test_overlays_off_when_disabled(self) -> None:
        # Default-off -> both carriers None -> byte-identical pre-U5/U6 path.
        calc, perception = _calc_and_perception()
        svc = AutonomousGraspService.from_robot_config(
            self._cfg(fusion_enabled=False, extrinsics_path=None, commit_enabled=False),
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        orch = svc.runtime.orchestrator
        self.assertIsNone(orch.scene_fusion)
        self.assertIsNone(orch.commit_policy)

    def test_fail_closed_bad_extrinsics_raises_end_to_end(self) -> None:
        # FAIL-CLOSED end-to-end: fusion enabled + a set-but-unloadable extrinsics path must RAISE through the
        # whole from_robot_config composition (not just the build_config_frame_resolver helper) -- the operator
        # must not get a silently-unreachable commit gate while believing fusion is on.
        calc, perception = _calc_and_perception()
        with self.assertRaises(RuntimeError):
            AutonomousGraspService.from_robot_config(
                self._cfg(
                    fusion_enabled=True,
                    extrinsics_path="/definitely/not/a/real/extrinsics_artifact.json",
                    commit_enabled=True,
                ),
                calculator=calc,  # type: ignore[arg-type]
                perception=perception,
            )


if __name__ == "__main__":
    unittest.main()
