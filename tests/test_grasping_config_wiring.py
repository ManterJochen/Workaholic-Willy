"""Phase T0 — runtime wiring tests for ``robot.grasping`` config.

These tests pin the locked T0 contract (operator-confirmed):

* ``AutonomousGraspService.from_robot_config`` resolves the
  :class:`GraspMode` from ``robot_cfg.grasping`` when the caller does
  not pass one; a pick makes the cell's ``MAX_ATTEMPTS`` (5) unless the
  call asks fewer (``robot.grasping.max_attempts`` until 2026-10-17).
* The factory **fail-closes** when ``mode`` is not supplied *and* the
  ``grasping`` block was not explicitly set on the ``RobotConfig`` (i.e.
  the operator relied on schema defaults silently). Production
  deployments must declare the block explicitly.
* Unknown mode strings raise.
* The sub-policies config built on the service (``RefinementPolicy``,
  ``GraspVerificationPolicy`` and the ``dense_recovery`` ``SceneRecoveryPolicy``)
  left on 2026-09-29 with their blocks; the recovery a pick runs is the cell's
  own since 2026-10-16 (``CELL_RECOVERY``), in the snapshot below.
* Every :class:`AutonomousGraspReport` carries an
  :class:`EffectiveGraspingConfig` snapshot when the service was wired
  from config; the legacy ``from_components`` path leaves it ``None``
  (feature-flag compatibility per T0 deliverable #4).

The test rig mirrors :mod:`tests.test_autonomous_grasp_service` to
avoid spawning a parallel doubles tree.
"""

from __future__ import annotations

import json
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
    EffectiveRecoveryOrchestratorConfig,
    GraspMode,
)
from src.robot.execution.autonomous_grasp.config import CELL_RECOVERY, MAX_ATTEMPTS
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.telemetry.outcome_logging import iter_jsonl
from src.robot.grasping.types.perception import PerceptionFrame


# ---------------------------------------------------------------------------
# Test doubles (mirrors tests/test_autonomous_grasp_service.py)
# ---------------------------------------------------------------------------


_REAL_UR_CAPS = RobotCapabilities(
    vendor="ur",
    model="ur5e",
    dof=6,
    has_native_fk=True,
    is_simulated=False,
    controller=None, owns_a_controller=True,
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
            "record_log_path": grasping_overrides.pop("record_log_path", None),
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
        )
        with self.assertRaises((AttributeError, TypeError)):
            snap.default_mode = GraspMode.EASY  # type: ignore[misc]

    def test_to_dict_is_json_safe(self) -> None:
        snap = EffectiveGraspingConfig(
            default_mode=GraspMode.DENSE_CLUTTER,
            max_attempts=7,
            recovery_orchestrator=EffectiveRecoveryOrchestratorConfig(
                enabled=True, allowed_actions=("rescan", "nudge_target"),
            ),
        )
        d = snap.to_dict()
        json.dumps(d)
        self.assertEqual(d["default_mode"], "dense_clutter")
        self.assertEqual(d["max_attempts"], 7)
        self.assertTrue(d["recovery_orchestrator_enabled"])
        self.assertEqual(d["recovery_orchestrator_allowed_actions"], ["rescan", "nudge_target"])
        # Each left on 2026-09-29 with the block it reported.
        for gone in ("closed_loop_enabled", "verification_enabled", "dense_recovery_enabled",
                     "dense_recovery_allowed_actions"):
            with self.subTest(key=gone):
                self.assertNotIn(gone, d)


# ---------------------------------------------------------------------------
# from_robot_config mode resolution
# ---------------------------------------------------------------------------


class FromRobotConfigModeResolutionTests(unittest.TestCase):
    """``default_mode`` flows from config into the service; the attempts are the cell's or the call's."""

    def test_default_mode_easy_resolves_easy_profile(self) -> None:
        cfg = _robot_cfg_with_grasping(default_mode="easy")
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        self.assertIs(service.default_mode, GraspMode.EASY)

    def test_default_mode_dense_clutter(self) -> None:
        cfg = _robot_cfg_with_grasping(default_mode="dense_clutter")
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        self.assertIs(service.default_mode, GraspMode.DENSE_CLUTTER)

    def test_a_tree_makes_the_cells_attempts(self) -> None:
        cfg = _robot_cfg_with_grasping(default_mode="auto")
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        self.assertEqual(service.runtime.orchestrator.max_attempts, MAX_ATTEMPTS)
        self.assertEqual(5, MAX_ATTEMPTS)

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

    def test_explicit_max_attempts_overrides_the_cells(self) -> None:
        cfg = _robot_cfg_with_grasping(default_mode="auto")
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


# `FromRobotConfigSubPolicyAutoBuildTests` left on 2026-09-29 with what it pinned: the `verification`
# and `dense_recovery` blocks built into policies on the service. The recovery a pick runs is fixed
# since 2026-10-16 at the values the owner's cell ran (`CELL_RECOVERY`), pinned in
# `tests/test_a_config_built_recovery_plans_only_what_its_profile_allows.py`.


# ---------------------------------------------------------------------------
# Effective config snapshot
# ---------------------------------------------------------------------------


class EffectiveConfigSnapshotTests(unittest.TestCase):
    """The effective config must round-trip through the report."""

    def test_service_carries_effective_config_when_built_from_config(self) -> None:
        cfg = _robot_cfg_with_grasping(default_mode="auto")
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
        self.assertEqual(snap.max_attempts, MAX_ATTEMPTS)
        # No key sets the recovery since 2026-10-16: an auto pick runs the cell's.
        self.assertEqual(CELL_RECOVERY, snap.recovery_orchestrator)

    def test_report_contains_effective_config(self) -> None:
        cfg = _robot_cfg_with_grasping(default_mode="auto")
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            max_attempts=2,
        )
        report = service.pick()
        # We don't pin the outcome here — the dummy arm built by
        # from_robot_config may or may not execute the grasp depending
        # on driver behaviour. The contract is: the snapshot must
        # survive into the report regardless of outcome.
        assert report.effective_config is not None
        self.assertIs(report.effective_config.default_mode, GraspMode.AUTO)
        self.assertEqual(report.effective_config.max_attempts, 2)
        self.assertEqual(CELL_RECOVERY, report.effective_config.recovery_orchestrator)

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

    _LOGGER = "src.robot.execution.robot_parts"

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
    """H0.3b — the effective_config-gated overlay (T5 recovery) is ACTIVATED through from_robot_config — the path
    that from_components SILENCES by leaving ``effective_config=None`` (see
    ``test_from_components_leaves_effective_config_none`` for the contrast). The decision, uncertainty, watchdog and
    SLO overlays it proved beside it left on 2026-10-16 with their blocks (the owner's cleanup)."""

    @staticmethod
    def _full_stack_cfg() -> RobotConfig:
        return RobotConfig(
            vendor="dummy",
            gripper={"vendor": "none"},
            grasping={"default_mode": "auto"},
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
        self.assertTrue(ec.recovery_orchestrator.enabled, "T5 recovery overlay not activated")


class FromRobotConfigPrimaryResolverTests(unittest.TestCase):
    """A primary rig whose declared calibration does not load refuses the whole ``from_robot_config``
    composition, not only the ``build_config_frame_resolver`` helper whose own fail-closed contract
    ``tests/test_g5_fusion_resolver.py`` pins. This pins the end-to-end refusal, the one seam that file
    does not cover."""

    @staticmethod
    def _cfg() -> RobotConfig:
        return RobotConfig(vendor="dummy", gripper={"vendor": "none"}, grasping={"default_mode": "dense_clutter"})

    @staticmethod
    def _camera(extrinsics_path: str):
        """The shipped camera section whose RGB-D rig is the primary and declares its calibration on the rig,
        ``camera.cameras.rigs[<id>].extrinsics``, eye_to_hand from ``extrinsics_path``."""
        from src.config import load_config

        shipped = load_config().camera
        data = shipped.model_dump(mode="json")
        rig = next(r for r in data["cameras"]["rigs"] if r["source"] == "rgbd")
        rig.update({"enabled": True, "extrinsics": {"mounting_mode": "eye_to_hand", "artifact_path": extrinsics_path}})
        data["cameras"]["primary_rig_id"] = rig["rig_id"]
        return type(shipped).model_validate(data)

    def test_fail_closed_bad_extrinsics_raises_end_to_end(self) -> None:
        # Fail-closed end to end: a primary rig that declares an unloadable artifact must raise through the
        # whole from_robot_config composition (not only the build_config_frame_resolver helper), so the
        # operator does not get a cell that cannot place its camera while believing it calibrated.
        calc, perception = _calc_and_perception()
        with self.assertRaises(RuntimeError) as caught:
            AutonomousGraspService.from_robot_config(
                self._cfg(),
                calculator=calc,  # type: ignore[arg-type]
                perception=perception,
                camera=self._camera("/definitely/not/a/real/extrinsics_artifact.json"),
            )
        self.assertIn("].extrinsics", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
