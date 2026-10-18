"""Phase U12 — runbook manifest, the soak gate over a log, docs links.

This module locks the Phase U12 deliverables:

1. ``docs/runbooks/runbooks_index.json`` exists, validates against the
   locked schema, and points at every shipped runbook.
2. Every listed runbook file exists and contains all five required H2
   sections (``Trigger`` / ``Diagnose`` / ``Mitigate`` / ``Verify`` /
   ``Rollback``).
3. The soak gate over a real log (``--records-gate``) and over a sim log (``--sim-soak-report``) can
   fail, and passes a clean log. The consolidated synthetic soak report left on 2026-10-16.
4. ``docs/Quickstart.md`` and the three operator READMEs reference
   ``docs/runbooks/`` so on-call operators can find the runbooks from
   any entry-point document.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNBOOKS_DIR = _REPO_ROOT / "docs" / "runbooks"
_RUNBOOKS_INDEX = _RUNBOOKS_DIR / "runbooks_index.json"
_REQUIRED_RUNBOOK_SECTIONS = (
    "Trigger",
    "Diagnose",
    "Mitigate",
    "Verify",
    "Rollback",
)
_DOC_LINK_TARGETS = (
    "docs/Quickstart.md",
    "src/robot/grasping/README.md",
    "src/robot/README.md",
    "src/config/README.md",
)


def _records(attempts: int, *, failed_every: int = 100, mode: str = "easy") -> list[GraspAttemptRecord]:
    """A clean log: every ``failed_every``-th attempt found no grasp, every other one succeeded with its execution
    block. No dead loop, every outcome typed."""
    return [
        GraspAttemptRecord.new(
            attempt_id=f"log-{index:05d}", mode=mode, timestamp=float(index),
            final_outcome="no_valid_grasp" if index % failed_every == failed_every - 1 else "succeeded",
            execution=None if index % failed_every == failed_every - 1 else {"outcome": "executed"},
        )
        for index in range(attempts)
    ]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _runbooks_on_disk() -> tuple[Path, ...]:
    """Every runbook the directory actually holds, in a stable order.

    ⛔ MEASURED 2026-09-10: the manifest listed three files while `docs/runbooks/` held six.
    `corpus_v5_build.md`, `train_your_own_generator.md` and `ur_family_bringup.md` were never
    on the list, so the five-section check below never looked at them, and two of the three
    carried none of the five sections while the gate stayed green. A gate that iterates a list
    can only ever confirm the list; what escapes a list is exactly what is not on it, so the
    loop starts at the directory now and the manifest is checked against it.
    """
    return tuple(sorted(_RUNBOOKS_DIR.glob("*.md")))


def _manifest_relative_path(path: Path) -> str:
    return "docs/runbooks/" + path.name


class RunbookManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(
            _RUNBOOKS_INDEX.is_file(),
            f"missing {_RUNBOOKS_INDEX}",
        )
        self.manifest = json.loads(_read(_RUNBOOKS_INDEX))

    def test_manifest_top_level_shape(self) -> None:
        self.assertEqual(self.manifest.get("capability_group"), "soak_gate")
        self.assertEqual(self.manifest.get("manifest_version"), 1)
        self.assertEqual(
            tuple(self.manifest["required_sections"]),
            _REQUIRED_RUNBOOK_SECTIONS,
        )
        runbooks = self.manifest.get("runbooks")
        self.assertIsInstance(runbooks, list)
        # NOT A COUNT ANY MORE, because a count never measured anything. This asserted at least five
        # runbooks back when there were eleven, nine of which described a model-promotion and RL
        # on-call lifecycle the product never built; four of those instructed operators to set config
        # keys that do not exist, which under `extra='forbid'` takes a cell from degraded to will not
        # boot. They were deleted on 2026-08-28. What the manifest is actually for is the FORMAT
        # contract below: every listed file exists and carries all five sections with real content.
        self.assertTrue(runbooks, "the manifest lists no runbooks at all")

    def test_each_runbook_entry_well_formed(self) -> None:
        seen_ids: set[str] = set()
        for entry in self.manifest["runbooks"]:
            for field in ("id", "path", "summary", "owner_role"):
                self.assertIn(field, entry)
                self.assertIsInstance(entry[field], str)
                self.assertTrue(entry[field].strip())
            self.assertNotIn(entry["id"], seen_ids, "duplicate runbook id")
            seen_ids.add(entry["id"])
            self.assertTrue(entry["path"].startswith("docs/runbooks/"))
            self.assertTrue(entry["path"].endswith(".md"))

    def test_the_manifest_and_the_directory_cover_each_other(self) -> None:
        """Both directions, because each one catches a different rot.

        ⛔ THE DIRECTION THAT WAS MISSING. Every check in this class iterated
        ``manifest["runbooks"]``, so a runbook added to ``docs/runbooks/`` and not to the
        manifest was invisible to all of them: it could carry none of the five sections and the
        gate stayed green. Measured on 2026-09-10, three files were in exactly that state.

        The other direction (a manifest entry whose file is gone) was already covered by the
        section check, which opened each listed path; it is asserted here explicitly so the two
        failures read as two different sentences instead of one ``missing <path>``.
        """
        listed = {entry["path"] for entry in self.manifest["runbooks"]}
        on_disk = {_manifest_relative_path(path) for path in _runbooks_on_disk()}
        self.assertEqual(
            sorted(on_disk - listed),
            [],
            "runbooks that exist but are not in runbooks_index.json, so no gate looks at them",
        )
        self.assertEqual(
            sorted(listed - on_disk),
            [],
            "runbooks_index.json lists files that docs/runbooks/ does not have",
        )

    def test_each_runbook_file_exists_and_has_required_sections(
        self,
    ) -> None:
        # ⭐ DRIVEN BY THE DIRECTORY, NOT BY THE MANIFEST. The manifest is the thing under test
        # one method up; using it as the loop source here is what let three files out.
        for path in _runbooks_on_disk():
            entry = {"id": path.stem, "path": _manifest_relative_path(path)}
            self.assertTrue(path.is_file(), f"missing {path}")
            body = _read(path)
            headings = set(re.findall(r"^##\s+(.+?)\s*$", body, re.MULTILINE))
            for required in _REQUIRED_RUNBOOK_SECTIONS:
                self.assertIn(
                    required,
                    headings,
                    f"runbook {entry['id']} missing H2 '{required}'",
                )
            # Each required section must have a non-empty body — i.e.
            # at least one non-blank line of content before the next
            # H2 / EOF.
            for required in _REQUIRED_RUNBOOK_SECTIONS:
                pattern = (
                    rf"^##\s+{re.escape(required)}\s*$\n+"
                    rf"(.*?)(?=^##\s+|\Z)"
                )
                match = re.search(
                    pattern, body, re.MULTILINE | re.DOTALL
                )
                self.assertIsNotNone(
                    match,
                    f"{entry['id']} '{required}' regex did not match",
                )
                section_body = (match.group(1) if match else "").strip()
                self.assertTrue(
                    section_body,
                    f"{entry['id']} '{required}' has empty body",
                )


class DocLinksToRunbooksTests(unittest.TestCase):
    """Every operator entry-point doc must mention ``docs/runbooks``."""

    def test_entry_docs_link_to_runbooks(self) -> None:
        for relative in _DOC_LINK_TARGETS:
            doc_path = _REPO_ROOT / relative
            self.assertTrue(doc_path.is_file(), f"missing {doc_path}")
            body = _read(doc_path)
            self.assertIn(
                "docs/runbooks",
                body,
                f"{relative} does not reference docs/runbooks/",
            )


class RecordsGateTests(unittest.TestCase):
    """K2: --records-gate applies the soak thresholds over a REAL log, and this CAN fail."""

    def _run_records_gate(self, log_path: Path) -> tuple[int, dict]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.robot.grasping.replay",
                "--records-gate",
                str(log_path),
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.returncode, json.loads(result.stdout)

    def test_records_gate_can_fail_on_real_log(self) -> None:
        # A log far short of the 2000-attempt floor honestly FAILS (exit 1) -- the whole point of a real-log gate.
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "short.jsonl"
            pack.write_text("\n".join(r.to_json() for r in _records(12)) + "\n", encoding="utf-8")
            rc, payload = self._run_records_gate(pack)
        self.assertEqual(rc, 1)
        self.assertEqual(payload["mode"], "records-gate")
        self.assertEqual(
            payload["input_provenance"], "real_grasp_attempt_record_log"
        )
        self.assertFalse(payload["gate"]["passes"])
        self.assertTrue(
            any("min_attempts" in v for v in payload["violations"])
        )

    def test_records_gate_passes_on_good_log(self) -> None:
        # A >= 2000-record log of mostly-successful attempts (no dead loops, valid outcomes) passes the gate (exit 0).
        records = _records(2100)
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "good.jsonl"
            log.write_text(
                "\n".join(json.dumps(r.to_dict()) for r in records) + "\n",
                encoding="utf-8",
            )
            rc, payload = self._run_records_gate(log)
        self.assertEqual(rc, 0, msg=str(payload["violations"]))
        self.assertTrue(payload["gate"]["passes"])
        self.assertTrue(payload["gate"]["min_attempts_met"])


class SimSoakReportTests(unittest.TestCase):
    """S4: --sim-soak-report runs the record-intrinsic gate over a REAL sim log + writes a persistent
    report with an honest sim provenance banner. pick_success_rate is REPORTED but not gated; a smaller honest sim
    min_attempts floor."""

    def _run(self, log_path: Path, *, min_attempts: int, out_path: Path) -> tuple[int, dict]:
        result = subprocess.run(
            [
                sys.executable, "-m", "src.robot.grasping.replay",
                "--sim-soak-report", str(log_path),
                "--sim-min-attempts", str(min_attempts),
                "--out", str(out_path),
            ],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=300,
        )
        return result.returncode, json.loads(result.stdout)

    @staticmethod
    def _good_records(attempts: int, *, failed_every: int = 100) -> list[GraspAttemptRecord]:
        return _records(attempts, failed_every=failed_every, mode="dense_clutter")

    def _write(self, td: str, records) -> Path:  # noqa: ANN001
        log = Path(td) / "sim.jsonl"
        log.write_text("\n".join(json.dumps(r.to_dict()) for r in records) + "\n", encoding="utf-8")
        return log

    def test_passes_on_good_sim_log_and_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = self._write(td, self._good_records(320))
            out = Path(td) / "sim_soak_report.json"
            rc, payload = self._run(log, min_attempts=300, out_path=out)
            self.assertEqual(rc, 0, msg=str(payload.get("violations")))
            self.assertEqual(payload["mode"], "sim-soak-report")
            self.assertTrue(payload["gate_passes"])
            self.assertTrue(out.exists())
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["provenance"]["input"], "real_sim_grasp_record_log")
            self.assertEqual(report["min_attempts_floor"], 300)
            self.assertIn("min_attempts floor=300", report["provenance"]["note"])
            # De-shouted by the docs/prose rewrite, same sentence: `soak_cli.py` emitted
            # "REAL Isaac-physics sim picks (NOT hardware-representative)." before the migration
            # and emits "Real Isaac-physics sim picks (not hardware-representative)." now. The
            # claim the banner has to carry is unchanged, so pin the words rather than the shout.
            self.assertIn("not hardware-representative", report["provenance"]["note"])
            # KPI is present + honest (false_positive structurally 0 in sim)
            self.assertEqual(report["kpi"]["false_positive_grasp_rate"], 0.0)

    def test_can_fail_below_floor(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = self._write(td, self._good_records(50))  # < the 300 floor -> honest failure
            out = Path(td) / "r.json"
            rc, payload = self._run(log, min_attempts=300, out_path=out)
            self.assertEqual(rc, 1)
            self.assertFalse(payload["gate_passes"])
            self.assertTrue(any("min_attempts" in v for v in payload["violations"]))

    def test_pick_success_reported_not_gated(self) -> None:
        # A LOW-success but integrity-clean log (enough attempts, valid outcomes, no dead loops) still
        # PASSES -- pick_success is reported, not gated.
        with tempfile.TemporaryDirectory() as td:
            log = self._write(td, self._good_records(320, failed_every=2))  # 50% success
            out = Path(td) / "r.json"
            rc, payload = self._run(log, min_attempts=300, out_path=out)
            self.assertEqual(rc, 0, msg=str(payload.get("violations")))
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertLess(report["kpi"]["pick_success_rate"], 0.9)  # genuinely low, yet the gate passed


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
