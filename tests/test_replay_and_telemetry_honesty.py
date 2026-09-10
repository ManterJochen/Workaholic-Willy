"""The replay command lines and the record serializer, judged on what they SAY.

Every test here corresponds to a defect reproduced against the shipped code before the repair, and
each one names the failure it caught:

* three CLI legs answered a missing input file with a raw ``FileNotFoundError`` traceback while
  every neighbouring leg answered with a sentence and an exit code;
* ``--baseline-report`` silently ignored ``--out`` and rewrote the git-tracked baseline instead;
* the library KPI leg printed ``false_positive_grasp_rate: 0.0`` as a measurement, a number that
  comes from a field no writer on this stack ever sets;
* the record serializer read ``report.verification``, an attribute ``AutonomousGraspReport`` has
  never had, so the verification block was ``None`` on every real record;
* nothing wrote the ``refinement`` block, so the three refine-stage outcomes that the telemetry
  catalog requires it for could never produce a complete record.
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from src.robot.grasping.replay import __main__ as replay_main
from src.robot.grasping.replay.runs import SoakGate


def _run_cli(argv: list[str]) -> tuple[int, str]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = replay_main.main(argv)
    return code, buffer.getvalue()


class MissingInputIsAnAnswerTests(unittest.TestCase):
    """A missing input file is an operator typo, not a crash site."""

    def test_records_gate_answers_a_missing_log(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.jsonl"
            code, out = _run_cli(["--records-gate", str(missing)])
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertIn("error", payload)
        self.assertIn(str(missing), payload["error"])

    def test_sim_soak_report_answers_a_missing_log_and_writes_no_report(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.jsonl"
            report = Path(tmp) / "report.json"
            code, out = _run_cli(
                ["--sim-soak-report", str(missing), "--baseline-out", str(report)]
            )
            self.assertFalse(
                report.exists(), "a report was written for an input nobody could read"
            )
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertIn("error", payload)
        self.assertIn(str(missing), payload["error"])

    def test_failure_taxonomy_answers_a_missing_pack(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "pack.jsonl"
            code, out = _run_cli(
                ["--failure-taxonomy", str(missing), "--out", str(Path(tmp) / "r.json")]
            )
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertIn("error", payload)
        self.assertIn(str(missing), payload["error"])

    def test_the_gate_itself_returns_a_verdict_rather_than_raising(self) -> None:
        """The library twin of the CLI leg above. ``RecordLog`` has answered this way since it was
        written; the gate raised.
        """
        verdict = SoakGate.over_records("does/not/exist.jsonl").evaluate()
        self.assertFalse(verdict.passes)
        self.assertEqual(verdict.exit_code, 2)
        self.assertTrue(verdict.violations)


class BaselineOutIsHonouredTests(unittest.TestCase):
    """``--out`` names a file. A mode that ignores it overwrites a committed one instead."""

    def test_baseline_report_writes_where_the_operator_asked(self) -> None:
        written: list[Path] = []

        class _Measured:
            report = {"stub": True}
            audit_offenders = 0
            exit_code = 0

            def write(self, path: Path) -> Path:
                written.append(Path(path))
                return Path(path)

        class _Baseline:
            @classmethod
            def canonical(cls) -> "_Baseline":
                return cls()

            def measure(self) -> _Measured:
                return _Measured()

        with TemporaryDirectory() as tmp:
            asked = Path(tmp) / "mine" / "baseline.json"
            with mock.patch.object(replay_main, "Baseline", _Baseline):
                code, out = _run_cli(["--baseline-report", "--out", str(asked)])
            self.assertEqual(code, 0)
            self.assertEqual(written, [asked.resolve()])
            self.assertEqual(json.loads(out)["report_path"], str(asked.resolve()))


class UnmeasurableRatesAreNamedTests(unittest.TestCase):
    """A rate with no source must not print as a number, in the library as well as the console."""

    def _log(self, tmp: str) -> Path:
        from src.robot.grasping.replay.soak import (
            SoakScenarioSpec,
            generate_soak_records,
        )

        records = generate_soak_records(
            SoakScenarioSpec(
                name="unit",
                mode="easy",
                attempts=12,
                failure_class_weights={"succeeded": 9.0, "no_valid_grasp": 1.0},
                recovery_success_rate=0.0,
                cycle_time_mean_s=1.5,
                cycle_time_jitter_s=0.2,
                seed=7,
            )
        )
        path = Path(tmp) / "records.jsonl"
        path.write_text("\n".join(r.to_json() for r in records) + "\n", encoding="utf-8")
        return path

    def test_the_records_leg_names_the_rate_it_cannot_measure(self) -> None:
        with TemporaryDirectory() as tmp:
            code, out = _run_cli(["--records", str(self._log(tmp))])
        payload = json.loads(out)
        self.assertIn("unmeasurable", payload)
        self.assertIn("false_positive_grasp_rate", payload["unmeasurable"])
        self.assertNotIn("false_positive_grasp_rate", payload["kpi"])
        self.assertIn(
            "post-grasp re-check", payload["unmeasurable"]["false_positive_grasp_rate"]
        )
        self.assertEqual(code, 0)

    def test_the_console_reads_the_library_rather_than_its_own_copy(self) -> None:
        from api.history import UNMEASURABLE_KPIS
        from src.robot.grasping.replay.kpi import UNMEASURABLE_KPIS as LIBRARY

        self.assertEqual(dict(UNMEASURABLE_KPIS), dict(LIBRARY))


class RecordSerializerReadsWhatExistsTests(unittest.TestCase):
    """``getattr(report, "verification", None)`` on a report with no such attribute is always
    ``None``: a lookup that can neither fail nor succeed.
    """

    def test_the_report_has_never_had_a_verification_attribute(self) -> None:
        from src.robot.execution.autonomous_grasp.report import (
            AutonomousGraspReport,
        )

        self.assertNotIn("verification", {f.name for f in fields(AutonomousGraspReport)})

    def test_the_verification_block_comes_from_the_telemetry_the_service_stamps(self) -> None:
        from src.robot.execution.autonomous_grasp.record_logging import (
            to_attempt_record,
        )

        class _Report:
            outcome = "succeeded"
            mode = "dense_clutter"
            profile = None
            pick_report = None
            telemetry = {
                "verification_outcome": "passed",
                "verification_reason": "width_delta_within_tolerance",
                "verification_telemetry": {"width_delta_mm": 3.2},
            }

        record = to_attempt_record(_Report(), attempt_id="a1")
        self.assertIsNotNone(record.verification)
        assert record.verification is not None
        self.assertEqual(record.verification["reason"], "width_delta_within_tolerance")
        self.assertEqual(record.verification["outcome"], "passed")

    def test_a_refined_attempt_carries_the_refinement_block_the_catalog_requires(self) -> None:
        from src.robot.execution.autonomous_grasp.record_logging import (
            to_attempt_record,
        )
        from src.robot.grasping.replay.telemetry_catalog import audit_record

        class _Refinement:
            outcome = "diverged"
            matched_segmentation_index = 2
            match_iou = 0.71
            position_delta_mm = 41.0
            orientation_delta_deg = 3.0
            grip_width_delta_mm = 1.0
            failure_reason = "refinement_diverged"
            telemetry = {"stage": "refiner"}

        class _Report:
            outcome = "refinement_diverged"
            mode = "dense_clutter"
            profile = None
            pick_report = None
            telemetry: dict = {}
            refinement = _Refinement()

        record = to_attempt_record(_Report(), attempt_id="a2")
        self.assertIsNotNone(record.refinement)
        self.assertEqual(audit_record(record), ())

    def test_the_service_carries_its_refinement_report_on_the_report(self) -> None:
        from src.robot.execution.autonomous_grasp.report import (
            AutonomousGraspReport,
        )

        self.assertIn("refinement", {f.name for f in fields(AutonomousGraspReport)})


class ARefineStageFailureLogsACompleteRecordTests(unittest.TestCase):
    """End to end, because the serializer half proves nothing on its own.

    A real ``target_lost_during_refine`` is produced by driving the two-scan path with a second frame
    the tracker cannot match. The telemetry catalog requires a ``refinement`` block for that outcome
    and no writer set one, so this record could not be complete however the cell was configured.
    """

    def _service(self, refined_mask: "tuple[slice, slice]"):
        import numpy as np

        from src.geometry import Frame, Pose
        from src.robot.core import (
            MotionCommand,
            MotionResult,
            RobotCapabilities,
        )
        from src.robot.execution.autonomous_grasp import (
            AutonomousGraspService,
            GraspMode,
        )
        from src.robot.grasping import (
            DefaultPreGraspRefiner,
            GraspVerificationPolicy,
            IdentityFrameResolver,
            NoOpVerifier,
            RefinementPolicy,
        )
        from src.robot.grasping.types.feedback import GraspResult
        from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
        from src.robot.grasping.types.perception import PerceptionFrame

        def _frame(rows: slice, cols: slice) -> PerceptionFrame:
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[rows, cols] = 1
            return PerceptionFrame(
                depth_map=np.full((32, 32), 500.0, dtype=np.float64),
                intrinsics=np.array(
                    [[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                ),
                segmentations=(SimpleNamespace(mask=mask),),
            )

        grasp = GraspPoint(
            position=np.array([100.0, 50.0, 400.0]),
            approach=np.array([0.0, 0.0, 1.0]),
            axis=np.array([1.0, 0.0, 0.0]),
            grip_width_mm=40.0,
            score=0.9,
            frame=GraspFrame.BASE,
            label="grasp",
        )

        class _Arm:
            def __init__(self) -> None:
                self._tcp = Pose(
                    position_mm=np.array([0.0, 0.0, 500.0]),
                    quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                    frame=Frame.BASE,
                    label="home",
                )

            @property
            def capabilities(self) -> RobotCapabilities:
                return RobotCapabilities(
                    vendor="ur", model="ur5e", dof=6, supports_joint_move=True,
                    supports_linear_move=True, supports_async_move=False,
                    has_native_fk=True, has_native_ik=True, has_force_control=False,
                    is_simulated=False,
                )

            def get_tcp_pose(self) -> Pose:
                return self._tcp

            def move(self, pose: Pose, **_: object) -> MotionResult:
                if pose.frame == Frame.BASE:
                    self._tcp = pose
                return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

        class _Perception:
            def __init__(self, frames: list) -> None:
                self._frames, self.calls = frames, 0

            def acquire(self) -> PerceptionFrame:
                frame = self._frames[min(self.calls, len(self._frames) - 1)]
                self.calls += 1
                return frame

        class _Calculator:
            def compute_result(self, *_a: object, **_k: object) -> GraspResult:
                return GraspResult(candidates=(grasp,), reasons=(), top_score=grasp.score)

        policy = RefinementPolicy(
            enabled=True, standoff_mm=80.0, max_position_correction_mm=50.0,
            max_grip_width_correction_mm=50.0, max_orientation_correction_deg=30.0,
            target_match_iou_threshold=0.3,
        )
        return AutonomousGraspService.from_components(
            arm=_Arm(),  # type: ignore[arg-type]
            calculator=_Calculator(),  # type: ignore[arg-type]
            perception=_Perception(
                [_frame(slice(10, 20), slice(10, 20)), _frame(*refined_mask)]
            ),
            mode=GraspMode.CLOSED_LOOP,
            frame_resolver=IdentityFrameResolver(),
            refinement_policy=policy,
            refiner=DefaultPreGraspRefiner(policy=policy),
            verification_policy=GraspVerificationPolicy(enabled=True),
            verifier=NoOpVerifier(),
        )

    def test_target_lost_during_refine_now_passes_the_telemetry_audit(self) -> None:
        from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
        from src.robot.execution.autonomous_grasp.record_logging import (
            to_attempt_record,
        )
        from src.robot.grasping.replay.telemetry_catalog import audit_record

        # A second frame whose only mask is disjoint from the first: the tracker refuses the match.
        report = self._service((slice(0, 2), slice(0, 2))).pick()
        self.assertIs(
            report.outcome, AutonomousGraspOutcome.TARGET_LOST_DURING_REFINE
        )
        self.assertIsNotNone(report.refinement)

        record = to_attempt_record(report, attempt_id="lost-1")
        self.assertIsNotNone(record.refinement)
        self.assertEqual(audit_record(record), ())

    def test_a_verified_attempt_logs_the_verifier_verdict(self) -> None:
        """The end-to-end half of the ``report.verification`` repair: the verifier runs, and its
        verdict has to reach the record from wherever the service actually puts it.
        """
        from src.robot.execution.autonomous_grasp.record_logging import (
            to_attempt_record,
        )

        # A second frame the tracker DOES match (1-px shift), so the attempt reaches verification.
        report = self._service((slice(10, 20), slice(11, 21))).pick()
        self.assertIn("verification_outcome", report.telemetry)

        record = to_attempt_record(report, attempt_id="verified-1")
        self.assertIsNotNone(record.verification)
        assert record.verification is not None
        self.assertEqual(
            record.verification["outcome"], str(report.telemetry["verification_outcome"])
        )
        self.assertEqual(
            record.verification["reason"], report.telemetry["verification_reason"]
        )

    def test_an_open_loop_attempt_still_carries_no_refinement_block(self) -> None:
        """⚠ THE BYTE-IDENTICAL HALF. The default pick runs no refiner, and a fabricated block there
        would be exactly the invention this writer exists to avoid.
        """
        from src.robot.execution.autonomous_grasp.record_logging import (
            to_attempt_record,
        )

        class _Report:
            outcome = "succeeded"
            mode = "auto"
            profile = None
            pick_report = None
            telemetry: dict = {}

        self.assertIsNone(to_attempt_record(_Report(), attempt_id="open-1").refinement)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
