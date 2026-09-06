"""P0 — opt-in pick instrumentation for the Isaac sim runners.

Pins the two seams the runners now wire: ``write_pick_artifacts`` (append a GraspAttemptRecord JSONL
line + dump the grasp-point PNG, both opt-in -> default-off no-op) and the
``AutonomousGraspService.last_debug_image_png`` accessor (the grasp-point viewer the runners dump).
Pure-Python / mock-safe (no isaacsim, no cv2).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.willy_sim.harness.instrumentation import write_pick_artifacts
from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome
from src.robot.execution.autonomous_grasp.service import AutonomousGraspService
from src.robot.grasping.telemetry.outcome_logging import iter_jsonl

_PNG = b"\x89PNG\r\n\x1a\n-fake-bytes"


def _report(outcome: AutonomousGraspOutcome = AutonomousGraspOutcome.SUCCEEDED, *, telemetry=None):
    return SimpleNamespace(
        outcome=outcome,
        mode=SimpleNamespace(value="easy"),
        profile=SimpleNamespace(),
        telemetry=telemetry or {},
        pick_report=None,
    )


class WritePickArtifactsTests(unittest.TestCase):
    def test_default_off_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            write_pick_artifacts(_report(), attempt_id="run-0000")
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_record_log_appends_and_roundtrips_with_gt_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "sub" / "pick_log.jsonl"  # parent dir auto-created
            write_pick_artifacts(
                _report(telemetry={"best_score": 0.85}),
                attempt_id="m1-1700000000-0000",
                record_log=log,
                extra={"sim_runner": "m1", "sim_lift_mm": 99.6, "sim_lifted": True},
            )
            recs = tuple(iter_jsonl(log))
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].attempt_id, "m1-1700000000-0000")
        self.assertEqual(recs[0].extra["sim_lift_mm"], 99.6)
        self.assertIs(recs[0].extra["sim_lifted"], True)
        self.assertEqual(recs[0].extra["best_score"], 0.85)  # telemetry preserved
        self.assertIn("safety_rejected", recs[0].extra)

    def test_debug_png_dumped_under_attempt_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            frames = Path(td) / "grasp_debug"
            write_pick_artifacts(
                _report(), attempt_id="eih-42-0003", debug_dir=frames, debug_png=_PNG,
            )
            out = frames / "eih-42-0003.png"
            self.assertTrue(out.exists())
            self.assertEqual(out.read_bytes(), _PNG)

    def test_debug_dir_set_but_no_png_skips(self) -> None:
        # A pick whose perception frame carried no rgb renders nothing -> no file, no crash.
        with tempfile.TemporaryDirectory() as td:
            frames = Path(td) / "grasp_debug"
            write_pick_artifacts(_report(), attempt_id="a", debug_dir=frames, debug_png=None)
            self.assertFalse(frames.exists() and any(frames.iterdir()))


class ServiceDebugImageAccessorTests(unittest.TestCase):
    def test_reads_calculator_last_debug_image_png(self) -> None:
        svc = object.__new__(AutonomousGraspService)
        svc.runtime = SimpleNamespace(  # type: ignore[attr-defined]
            orchestrator=SimpleNamespace(calculator=SimpleNamespace(last_debug_image_png=_PNG))
        )
        self.assertEqual(svc.last_debug_image_png, _PNG)

    def test_none_when_no_calculator_or_image(self) -> None:
        svc = object.__new__(AutonomousGraspService)
        svc.runtime = SimpleNamespace(orchestrator=None)  # type: ignore[attr-defined]
        self.assertIsNone(svc.last_debug_image_png)
        svc.runtime = SimpleNamespace(  # type: ignore[attr-defined]
            orchestrator=SimpleNamespace(calculator=SimpleNamespace(last_debug_image_png=None))
        )
        self.assertIsNone(svc.last_debug_image_png)

    def test_enable_debug_image_rendering_flips_calculator_flag(self) -> None:
        calc = SimpleNamespace(render_debug_images=False, last_debug_image_png=None)
        svc = object.__new__(AutonomousGraspService)
        svc.runtime = SimpleNamespace(orchestrator=SimpleNamespace(calculator=calc))  # type: ignore[attr-defined]
        svc.enable_debug_image_rendering()
        self.assertTrue(calc.render_debug_images)
        svc.enable_debug_image_rendering(False)
        self.assertFalse(calc.render_debug_images)

    def test_enable_debug_image_rendering_noop_without_calculator(self) -> None:
        svc = object.__new__(AutonomousGraspService)
        svc.runtime = SimpleNamespace(orchestrator=None)  # type: ignore[attr-defined]
        svc.enable_debug_image_rendering()  # must not raise


if __name__ == "__main__":
    unittest.main()
