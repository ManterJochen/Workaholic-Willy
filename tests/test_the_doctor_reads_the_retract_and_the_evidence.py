"""The doctor reads the retract row and the evidence file a planner start reads (customer chain lane C3i).

``--doctor`` could exit 0 for a cell whose first planned move refuses, for a missing retract row at its placement or a
missing evidence file (finding 19 of the audit of 2026-09-17): it probed the engines, the bundles and the map, and
nothing a planner start actually looks up.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock


def _doctor(cell: Any, gripper: "str | None" = "robotiq_2f85") -> Any:
    from src.contracts import UNSET
    from src.robot.safety.planning import doctor

    ok = doctor.Probe("engine", doctor.ProbeStatus.OK, "patched")
    with mock.patch.object(doctor, "_probe_coal", lambda blocks: ok), \
            mock.patch.object(doctor, "_probe_curobo", lambda *a, **k: (ok,)), \
            mock.patch.object(doctor, "code_integrity_blocks", lambda: ()):
        return doctor.run_doctor(model="ur5e", robot_config="willy_ur5e.yml", gripper=gripper,
                                 cell=UNSET if cell is None else cell)


def _probe(report: Any, name: str) -> Any:
    (probe,) = [p for p in report.probes if p.name.startswith(name)]
    return probe


def _ursim(**self_collision: Any) -> Any:
    from src.config.loader import load_robot_config

    cell = load_robot_config(profile="ursim,ursim_curobo")
    if self_collision:
        cell = cell.model_copy(update={"safety": cell.safety.model_copy(update={
            "self_collision": cell.safety.self_collision.model_copy(update=self_collision)})})
    return cell


class TheDoctorLooksUpWhatAStartLooksUpTests(unittest.TestCase):
    def test_the_ursim_curobo_cell_reports_its_row_and_its_file(self) -> None:
        report = _doctor(_ursim())
        row, evidence = _probe(report, "retract row"), _probe(report, "combination evidence")
        self.assertEqual(str(row.status), "ok", row.detail)
        self.assertIn("+Z+X", row.detail)
        self.assertEqual(str(evidence.status), "ok", evidence.detail)
        self.assertIn("ur5e_robotiq_2f85_c0mm_+Z+X_m4mm_a0.json", evidence.detail)

    def test_a_margin_nobody_judged_is_missing_with_the_command(self) -> None:
        report = _doctor(_ursim(planner_margin_mm=6.0))
        row, evidence = _probe(report, "retract row"), _probe(report, "combination evidence")
        self.assertEqual(str(row.status), "missing")
        self.assertIn("choose_ur_retract.py", row.detail + row.remedy)
        self.assertEqual(str(evidence.status), "missing")
        self.assertIn("m6mm", evidence.detail)
        self.assertIn("matrix_gate.py", evidence.detail + evidence.remedy)
        self.assertEqual(report.exit_code, 1)

    def test_an_ik_cell_needs_neither(self) -> None:
        cell = _ursim()
        cell = cell.model_copy(update={"ur": cell.ur.model_copy(update={"motion_planner": "ik"})})
        report = _doctor(cell)
        for name in ("retract row", "combination evidence"):
            with self.subTest(probe=name):
                probe = _probe(report, name)
                self.assertEqual(str(probe.status), "ok")
                self.assertIn("no planner", probe.detail)

    def test_a_doctor_given_no_cell_says_it_could_not_look(self) -> None:
        report = _doctor(None)
        probe = _probe(report, "planner combination")
        self.assertEqual(str(probe.status), "missing")
        self.assertIn("--profile", probe.remedy)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
