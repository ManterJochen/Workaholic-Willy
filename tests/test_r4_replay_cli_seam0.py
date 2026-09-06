"""R4 K1 Seam-0 — pin the replay CLI decomposition (handlers moved to soak_cli / adaptation_cli, dispatch intact).

main()'s argparse + if/elif dispatch + exit codes stay verbatim in __main__; the mode handlers moved to two leaves.
This pins that the handlers are importable from the leaves AND that __main__ re-imports them (so main()'s dispatch +
help-text still resolve). The per-phase u7-u12 suites exercise the real CLI via subprocess; the U12 --soak-report
byte-diff is the CI gate (verified byte-identical at build time).
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


class ReplayCliSeam0Tests(unittest.TestCase):
    def test_handlers_importable_from_leaves(self) -> None:
        from src.robot.grasping.replay.adaptation_cli import (
            _adaptation_apply_mode,
            _adaptation_plan_mode,
            _adaptation_rollback_mode,
            _adaptation_verify_mode,
        )
        from src.robot.grasping.replay.soak_cli import (
            _records_gate_mode,
            _records_mode,
            _sim_soak_report_mode,
            _soak_mode,
            _soak_report_mode,
        )

        for fn in (
            _records_mode, _records_gate_mode, _sim_soak_report_mode, _soak_mode, _soak_report_mode,
            _adaptation_plan_mode, _adaptation_verify_mode, _adaptation_apply_mode, _adaptation_rollback_mode,
        ):
            self.assertTrue(callable(fn))

    def test_main_dispatch_reimports_handlers(self) -> None:
        # main()'s dispatch calls the moved handlers — they must resolve in __main__'s namespace (same object).
        from src.robot.grasping.replay import __main__ as m
        from src.robot.grasping.replay import adaptation_cli, soak_cli

        self.assertIs(m._soak_report_mode, soak_cli._soak_report_mode)
        self.assertIs(m._adaptation_plan_mode, adaptation_cli._adaptation_plan_mode)
        self.assertIs(m._SIM_SOAK_REPORT_RELATIVE_PATH, soak_cli._SIM_SOAK_REPORT_RELATIVE_PATH)

    def test_help_exits_zero(self) -> None:
        out = subprocess.run(
            [sys.executable, "-m", "src.robot.grasping.replay", "--help"],
            cwd=str(_REPO),
            capture_output=True,
            text=True,
        )
        self.assertEqual(out.returncode, 0)
        self.assertIn("adaptation-plan", out.stdout)
        self.assertIn("soak-report", out.stdout)


if __name__ == "__main__":
    unittest.main()
