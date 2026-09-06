"""Tests for the Phase S1 :class:`AutonomousGraspService` fa\u00e7ade.

These tests pin the S0 locked contract:

* :class:`GraspMode` is a strictly additive wrap around
  :class:`GraspSamplingMode`.
* :class:`AutonomousGraspService` delegates to the existing
  :class:`RuntimePickService` without changing its surface, so EASY /
  AUTO / DENSE_CLUTTER stay byte-equivalent to today's behavior.
* Modes that need refinement or verification (``CLOSED_LOOP``,
  ``DENSE_AUTONOMOUS``) refuse to dispatch in S1 and return a typed
  :attr:`AutonomousGraspOutcome.MODE_NOT_AVAILABLE` outcome instead of
  silently downgrading to AUTO.
* The locked recovery allow-lists per mode are exposed unchanged on
  :class:`GraspBehaviorProfile`.

The fakes mirror ``tests/test_runtime_pick.py`` so we do not introduce
a parallel test rig.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.robot.grasping.telemetry.outcome_logging import iter_jsonl

from src.geometry import Frame, Pose
from src.robot.core import (
    MotionCommand,
    MotionResult,
    RobotCapabilities,
)
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspReport,
    AutonomousGraspService,
    GraspBehaviorProfile,
    GraspMode,
    resolve_grasp_mode,
)
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.modes import GraspSamplingMode
from src.robot.grasping.loop.pick_loop import (
    PerceptionFrame,
    PickOutcome,
)


# ---------------------------------------------------------------------------
# Test doubles (mirrors tests/test_runtime_pick.py)
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
        self.move_calls: list[Pose] = []

    @property
    def capabilities(self) -> RobotCapabilities:
        return _REAL_UR_CAPS

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        self.move_calls.append(pose)
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
        self.kwargs_seen: list[dict] = []

    def compute_result(self, *_args: object, **kwargs: object) -> GraspResult:
        self.kwargs_seen.append(dict(kwargs))
        idx = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[idx]


def _segmentation(shape: tuple[int, int] = (32, 32)) -> SimpleNamespace:
    mask = np.zeros(shape, dtype=np.uint8)
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


def _empty_perception_frame() -> PerceptionFrame:
    return PerceptionFrame(
        depth_map=np.zeros((4, 4), dtype=np.float64),
        intrinsics=np.eye(3, dtype=np.float64),
        segmentations=(),
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


def _rescan_only_result() -> GraspResult:
    return GraspResult(
        candidates=(),
        reasons=(GraspFailureReason.RESCAN_RECOMMENDED,),
        top_score=0.0,
    )


# ---------------------------------------------------------------------------
# Enum / resolver tests
# ---------------------------------------------------------------------------


class GraspModeResolverTests(unittest.TestCase):
    """Pin the locked S0 alias table for :func:`resolve_grasp_mode`."""

    def test_none_resolves_to_auto(self) -> None:
        self.assertIs(resolve_grasp_mode(None), GraspMode.AUTO)

    def test_passthrough_for_enum(self) -> None:
        for mode in GraspMode:
            with self.subTest(mode=mode):
                self.assertIs(resolve_grasp_mode(mode), mode)

    def test_known_string_aliases(self) -> None:
        for raw, expected in (
            ("easy", GraspMode.EASY),
            ("single", GraspMode.EASY),
            ("single_object", GraspMode.EASY),
            ("auto", GraspMode.AUTO),
            ("dense", GraspMode.DENSE_CLUTTER),
            ("dense_clutter", GraspMode.DENSE_CLUTTER),
            ("closed_loop", GraspMode.CLOSED_LOOP),
            ("dense_autonomous", GraspMode.DENSE_AUTONOMOUS),
            ("autonomous", GraspMode.DENSE_AUTONOMOUS),
            # Case- and whitespace-insensitive.
            ("  EASY  ", GraspMode.EASY),
            ("Dense_Clutter", GraspMode.DENSE_CLUTTER),
        ):
            with self.subTest(raw=raw):
                self.assertIs(resolve_grasp_mode(raw), expected)

    def test_unknown_value_raises_value_error_listing_options(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_grasp_mode("turbo")
        msg = str(ctx.exception)
        self.assertIn("turbo", msg)
        self.assertIn("easy", msg)
        self.assertIn("dense_autonomous", msg)

    def test_bool_is_not_a_valid_input(self) -> None:
        # GraspMode is a *profile* selector, not the low-level sampling
        # mode. Accepting bool here would re-introduce the legacy
        # dense_sampling footgun on the high-level surface.
        with self.assertRaises(ValueError):
            resolve_grasp_mode(True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Profile tests
# ---------------------------------------------------------------------------


class GraspBehaviorProfileTests(unittest.TestCase):
    """The locked per-mode S0 profile defaults must not drift silently."""

    def _build_service(self) -> AutonomousGraspService:
        return AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.EASY,
        )

    def test_easy_profile_is_locked(self) -> None:
        service = self._build_service()
        report = service.pick(mode=GraspMode.EASY)
        profile = report.profile
        self.assertIs(profile.mode, GraspMode.EASY)
        self.assertIs(profile.sampling_mode, GraspSamplingMode.SINGLE_OBJECT)
        self.assertFalse(profile.refinement_enabled)
        self.assertFalse(profile.verification_enabled)
        # EASY must never produce recovery motion --- safety guarantee.
        self.assertEqual(profile.recovery_allowed_actions, ())

    def test_auto_profile_locked_recovery_allowlist(self) -> None:
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.AUTO,
        )
        report = service.pick()
        self.assertIs(report.profile.sampling_mode, GraspSamplingMode.AUTO)
        self.assertEqual(
            report.profile.recovery_allowed_actions,
            ("rescan", "next_viewpoint"),
        )

    def test_dense_clutter_profile_locked(self) -> None:
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.DENSE_CLUTTER,
        )
        report = service.pick()
        self.assertIs(
            report.profile.sampling_mode, GraspSamplingMode.DENSE_CLUTTER
        )
        self.assertEqual(
            report.profile.recovery_allowed_actions,
            ("rescan", "next_viewpoint"),
        )

    def test_dense_autonomous_profile_locked(self) -> None:
        # Build a service for the dense profile so the per-call mode
        # is compatible with the configured sampling mode. The pick
        # call will still refuse (refinement+verification not yet
        # implemented), but the snapshot profile must match the lock.
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.DENSE_AUTONOMOUS,
        )
        report = service.pick()
        self.assertIs(report.outcome, AutonomousGraspOutcome.MODE_NOT_AVAILABLE)
        self.assertTrue(report.profile.refinement_enabled)
        self.assertTrue(report.profile.verification_enabled)
        self.assertEqual(
            report.profile.recovery_allowed_actions,
            ("rescan", "next_viewpoint", "nudge_target"),
        )

    def test_profile_is_frozen(self) -> None:
        profile = GraspBehaviorProfile(
            mode=GraspMode.EASY,
            sampling_mode=GraspSamplingMode.SINGLE_OBJECT,
        )
        with self.assertRaises(Exception):
            profile.refinement_enabled = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Service behavior tests
# ---------------------------------------------------------------------------


class AutonomousGraspServiceTests(unittest.TestCase):
    def test_lazy_export_via_execution_package(self) -> None:
        import src.robot.execution as execution_pkg

        self.assertIs(execution_pkg.AutonomousGraspService, AutonomousGraspService)
        self.assertIs(execution_pkg.AutonomousGraspReport, AutonomousGraspReport)
        self.assertIs(execution_pkg.AutonomousGraspOutcome, AutonomousGraspOutcome)
        self.assertIs(execution_pkg.GraspMode, GraspMode)
        self.assertIs(execution_pkg.GraspBehaviorProfile, GraspBehaviorProfile)
        self.assertIs(execution_pkg.resolve_grasp_mode, resolve_grasp_mode)

    def test_easy_mode_executes_with_single_object_sampling(self) -> None:
        arm = _TypedFakeArm()
        calculator = _ScriptedCalculator([_success_result()])
        service = AutonomousGraspService.from_components(
            arm=arm,  # type: ignore[arg-type]
            calculator=calculator,  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.EASY,
        )

        report = service.pick()

        self.assertIsInstance(report, AutonomousGraspReport)
        self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertTrue(report.succeeded)
        self.assertIs(report.mode, GraspMode.EASY)
        self.assertIsNotNone(report.pick_report)
        assert report.pick_report is not None  # for type-checkers
        self.assertIs(report.pick_report.outcome, PickOutcome.EXECUTED)
        # The wrapped orchestrator was configured with SINGLE_OBJECT.
        self.assertIs(
            service.runtime.orchestrator.grasp_sampling_mode,
            GraspSamplingMode.SINGLE_OBJECT,
        )

    def test_auto_is_default_when_mode_is_none(self) -> None:
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
        )
        self.assertIs(service.default_mode, GraspMode.AUTO)
        report = service.pick()
        self.assertIs(report.mode, GraspMode.AUTO)
        self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)

    def test_no_perception_maps_to_no_target(self) -> None:
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_rescan_only_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_empty_perception_frame()]),
            mode=GraspMode.EASY,
            max_attempts=1,
        )

        report = service.pick()

        self.assertIs(report.outcome, AutonomousGraspOutcome.NO_TARGET)
        self.assertFalse(report.succeeded)
        self.assertIsNotNone(report.pick_report)

    def test_rescan_exhausted_maps_to_no_valid_grasp(self) -> None:
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_rescan_only_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.AUTO,
            max_attempts=2,
        )

        report = service.pick()

        self.assertIs(report.outcome, AutonomousGraspOutcome.NO_VALID_GRASP)

    def test_closed_loop_refuses_until_s3_lands(self) -> None:
        # CLOSED_LOOP requires refinement (S3) + verification (S4). The
        # service must NOT silently downgrade to AUTO when those slices
        # have not landed yet --- doing so would lie to the operator
        # about which guarantees are in effect. After S3 lands, the
        # refusal is gated on whether a refiner is wired; an
        # operator who selects CLOSED_LOOP without wiring a
        # RefinementPolicy + refiner still gets MODE_NOT_AVAILABLE.
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.CLOSED_LOOP,
        )

        report = service.pick()

        self.assertIs(report.outcome, AutonomousGraspOutcome.MODE_NOT_AVAILABLE)
        self.assertIsNone(report.pick_report)
        self.assertEqual(
            report.telemetry.get("reason"),
            "mode_requires_refinement_but_no_refiner_wired",
        )
        self.assertTrue(report.telemetry.get("refinement_required"))
        self.assertTrue(report.telemetry.get("verification_required"))

    def test_dense_autonomous_refuses_until_recovery_lands(self) -> None:
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.DENSE_AUTONOMOUS,
        )

        report = service.pick()

        self.assertIs(report.outcome, AutonomousGraspOutcome.MODE_NOT_AVAILABLE)
        self.assertIsNone(report.pick_report)

    def test_per_call_mode_with_incompatible_sampling_is_refused(self) -> None:
        # Service was built for EASY (SINGLE_OBJECT). Asking it to run a
        # DENSE_CLUTTER pick at call time cannot magically change the
        # orchestrator's bound sampling mode --- refuse honestly.
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.EASY,
        )

        report = service.pick(mode=GraspMode.DENSE_CLUTTER)

        self.assertIs(report.outcome, AutonomousGraspOutcome.MODE_NOT_AVAILABLE)
        self.assertIs(report.mode, GraspMode.DENSE_CLUTTER)
        self.assertIsNone(report.pick_report)
        self.assertEqual(
            report.telemetry.get("reason"),
            "per_call_mode_requires_different_sampling_mode",
        )

    def test_report_is_frozen(self) -> None:
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.EASY,
        )
        report = service.pick()
        with self.assertRaises(Exception):
            report.outcome = AutonomousGraspOutcome.NO_TARGET  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Legacy non-regression
# ---------------------------------------------------------------------------


class LegacySurfaceNonRegressionTests(unittest.TestCase):
    """:class:`AutonomousGraspService` must not change the underlying
    :class:`RuntimePickService` behavior for the legacy sampling modes.
    """

    def test_easy_matches_runtime_pick_service_single_object_path(self) -> None:
        # Build both surfaces with equivalent inputs and assert the
        # underlying PickSessionReport.outcome is identical for the
        # happy path. We re-build the components for each side so the
        # state is not shared.
        from src.robot.execution.runtime_pick import RuntimePickService

        runtime = RuntimePickService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            grasp_sampling_mode=GraspSamplingMode.SINGLE_OBJECT,
        )
        runtime_report = runtime.run_attempt()

        wrapper = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.EASY,
        )
        wrapper_report = wrapper.pick()

        self.assertIs(runtime_report.outcome, PickOutcome.EXECUTED)
        assert wrapper_report.pick_report is not None
        self.assertIs(wrapper_report.pick_report.outcome, PickOutcome.EXECUTED)
        self.assertEqual(
            runtime_report.selected_score,
            wrapper_report.pick_report.selected_score,
        )


class RecordLoggingTests(unittest.TestCase):
    """L6 K1 (2/2): the opt-in production record-logging seam on AutonomousGraspService.pick."""

    def _service(self) -> AutonomousGraspService:
        return AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
        )

    def test_default_off_logs_nothing(self) -> None:
        service = self._service()
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "pick.jsonl"
            service.pick()
            self.assertFalse(log.exists())

    def test_enabled_logs_one_record_per_pick(self) -> None:
        service = self._service()
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "sub" / "pick.jsonl"
            service.enable_record_logging(log)
            service.pick()
            service.pick()
            recs = tuple(iter_jsonl(log))
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0].final_outcome, "succeeded")
        # the K1 KPI source key is present (False on a clean success)
        self.assertIn("safety_rejected", recs[0].extra)
        self.assertFalse(recs[0].extra["safety_rejected"])

    def test_disable_stops_logging(self) -> None:
        service = self._service()
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "pick.jsonl"
            service.enable_record_logging(log)
            service.pick()
            service.enable_record_logging(None)
            service.pick()
            recs = tuple(iter_jsonl(log))
        self.assertEqual(len(recs), 1)

    def test_from_components_record_log_path_param_logs(self) -> None:
        # H0.1 — the record_log_path constructor param opts in directly (no manual enable call) and
        # every record carries a unique attempt_id (H0.1a; the default path no longer collides on "pick").
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "sub" / "pick.jsonl"
            service = AutonomousGraspService.from_components(
                arm=_TypedFakeArm(),  # type: ignore[arg-type]
                calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
                perception=_FakePerception([_perception_frame()]),
                record_log_path=log,
            )
            service.pick()
            service.pick()
            recs = tuple(iter_jsonl(log))
        self.assertEqual(len(recs), 2)
        self.assertEqual(len({r.attempt_id for r in recs}), 2)
        for r in recs:
            self.assertTrue(r.attempt_id.startswith("pick-"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
