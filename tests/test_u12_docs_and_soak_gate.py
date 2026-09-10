"""Phase U12 — runbook manifest, consolidated soak gate, docs links.

This module locks the Phase U12 deliverables:

1. ``docs/runbooks/runbooks_index.json`` exists, validates against the
   locked schema, and points at every shipped runbook.
2. Every listed runbook file exists and contains all five required H2
   sections (``Trigger`` / ``Diagnose`` / ``Mitigate`` / ``Verify`` /
   ``Rollback``).
3. :func:`build_soak_report` produces ``\u2265 2000`` attempts, a
   13-key gate dict, ``passes=True``, and a populated
   ``easy_attempt_wall_time`` block on the current baseline.
4. Two back-to-back builds yield byte-identical KPI + scenarios + gate
   sections (determinism).
5. The ``--soak-report`` CLI exits ``0``, writes the JSON artifact, and
   the artifact round-trips through :mod:`json`.
6. ``docs/Quickstart.md`` and the three operator READMEs reference
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

from src.robot.grasping.replay.soak import (
    SoakScenarioSpec,
    SOAK_MIN_ATTEMPTS,
    build_soak_report,
    generate_soak_records,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
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
_EXPECTED_GATE_KEYS = (
    "min_attempts_met",
    "untyped_outcomes_zero",
    "unbounded_retry_loops_zero",
    "telemetry_offenders_zero",
    "extra_type_offenders_zero",
    "dead_loop_rate_within_gate",
    "pick_success_rate_non_regression",
    "slo_packs_pass",
    "drift_gate_pass",
    "ood_gate_pass",
    "failure_taxonomy_classifier_pass",
    "easy_attempt_wall_time_within_budget",
    "passes",
)


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


class BuildU12SoakReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload, cls.violations = build_soak_report(_REPO_ROOT)

    def test_report_shape_and_gate_keys(self) -> None:
        self.assertEqual(self.payload["capability_group"], "soak_gate")
        self.assertEqual(self.payload["report_version"], 1)
        self.assertGreaterEqual(
            int(self.payload["total_attempts"]), SOAK_MIN_ATTEMPTS
        )
        gate = self.payload["gate"]
        self.assertIsInstance(gate, dict)
        for key in _EXPECTED_GATE_KEYS:
            self.assertIn(key, gate, f"gate missing key {key!r}")
        # No extra keys beyond the locked set — keeps the contract honest.
        self.assertEqual(set(gate.keys()), set(_EXPECTED_GATE_KEYS))

    def test_gate_passes_on_current_baseline(self) -> None:
        self.assertEqual(
            tuple(self.violations),
            (),
            f"unexpected violations: {self.violations}",
        )
        self.assertTrue(bool(self.payload["gate"]["passes"]))

    def test_easy_wall_time_block_finite_and_within_budget(self) -> None:
        block = self.payload["easy_attempt_wall_time"]
        self.assertEqual(block["capability_group"], "soak_gate")
        for key in ("current_p95_s", "baseline_p95_s", "budget_p95_s"):
            value = block[key]
            self.assertIsNotNone(value, f"{key} unexpectedly None")
            self.assertIsInstance(value, float)
            self.assertGreater(float(value), 0.0)
        self.assertTrue(bool(block["within_budget"]))
        self.assertAlmostEqual(
            float(block["budget_multiplier"]), 1.05, places=6
        )

    def test_scenarios_meet_minimum_attempts(self) -> None:
        scenarios = self.payload["scenarios"]
        self.assertGreaterEqual(len(scenarios), 3)
        total = sum(int(s["attempts"]) for s in scenarios)
        self.assertGreaterEqual(total, SOAK_MIN_ATTEMPTS)
        # Seeds must be unique so the soak streams cannot accidentally
        # collide on identical attempt_ids.
        seeds = [int(s["seed"]) for s in scenarios]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_the_taxonomy_leg_FAILS_when_the_classifier_stops_classifying(self) -> None:
        """⛔ THE LEG WAS INERT, AND A DEAD CLASSIFIER PASSED IT. MEASURED 2026-09-10: the synthetic
        soak stream stamps no ``extra.*_evidence`` flag anywhere, so all 168 of its failures already
        classify as ``unclassified`` (coverage 0.0). A classifier that names nothing produces the
        same 168 and the same 0.0, the ``failure_taxonomy`` block comes out byte-identical, and the
        gate still says ``passes``. The soak carried a taxonomy section that could not tell a working
        classifier from a deleted one.

        So the leg is judged against the LABELED pack instead, whose rows carry
        ``extra.expected_root_cause``. This test kills the classifier and requires the gate to
        notice."""
        from unittest import mock

        from src.robot.grasping.replay import failure_taxonomy

        def _names_nothing(final_outcome, extra, *, attempt_id="<unnamed>"):  # noqa: ANN001
            return failure_taxonomy.TaxonomyVerdict(
                primary=failure_taxonomy.FailureRootCause.UNCLASSIFIED,
                also_matched=(),
                evidence={"final_outcome": final_outcome, "rules_fired": {}},
            )

        with mock.patch.object(failure_taxonomy, "classify_symptoms", _names_nothing):
            dead_payload, dead_violations = build_soak_report(_REPO_ROOT)

        self.assertEqual(
            dead_payload["failure_taxonomy"],
            self.payload["failure_taxonomy"],
            "the synthetic taxonomy block CHANGED under a dead classifier. If this ever fails, "
            "the stream grew evidence flags and this test's premise needs re-measuring",
        )
        self.assertFalse(
            bool(dead_payload["gate"]["passes"]),
            "a classifier that names nothing still passed the soak gate",
        )
        self.assertIn("failure_taxonomy_classifier_failed", dead_violations)

    def test_the_classifier_leg_reports_what_it_judged(self) -> None:
        """A gate key with no numbers behind it cannot be argued with."""
        block = self.payload["failure_taxonomy_classifier"]
        self.assertEqual(block["mismatches"], [])
        self.assertGreaterEqual(int(block["judged_failures"]), 60)
        self.assertAlmostEqual(float(block["label_agreement"]), 1.0, places=6)
        self.assertGreaterEqual(float(block["coverage_fraction"]), 0.95)
        self.assertTrue(bool(block["passes_gate"]))

    def test_determinism_two_builds_match(self) -> None:
        payload_a, _ = build_soak_report(_REPO_ROOT)
        payload_b, _ = build_soak_report(_REPO_ROOT)
        for key in ("kpi", "scenarios", "gate", "total_attempts"):
            self.assertEqual(
                payload_a[key],
                payload_b[key],
                f"non-deterministic key {key!r}",
            )


class SoakReportCLITests(unittest.TestCase):
    """Smoke the ``--soak-report`` CLI: exit 0 + writes a valid JSON."""

    def test_cli_writes_artifact_and_exits_zero(self) -> None:
        # Write to the same canonical location the CLI uses by default
        # (``logs/u12/soak_report.json``) so we also lock the default
        # path contract. K4: the file IS git-tracked and byte-stable (the
        # baseline_path is now POSIX-normalized); its regen-stability is
        # guarded by ``test_committed_report_is_regen_stable`` below.
        out_path = _REPO_ROOT / "logs" / "u12" / "soak_report.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.robot.grasping.replay",
                "--soak-report",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"CLI exited {result.returncode}\nstdout={result.stdout}"
            f"\nstderr={result.stderr}",
        )
        self.assertTrue(
            out_path.is_file(), f"CLI did not write {out_path}"
        )
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("capability_group"), "soak_gate")
        self.assertEqual(payload.get("report_version"), 1)
        self.assertTrue(bool(payload["gate"]["passes"]))

    def test_committed_report_is_regen_stable(self) -> None:
        # K4: the committed soak report must stay byte-in-sync with a fresh regen, or the golden silently
        # rots (it had drifted on Windows via a backslash baseline_path). Object-equality (like the U+
        # baseline guard) catches value + path-separator drift. The three lines that used to sit here
        # called this test platform-locked and named a conftest allow-list. MEASURED 2026-09-10: that
        # list gated on an environment variable nothing in the tree ever set, so this ran on no box at
        # all; opened by hand it passes here first try, which is the evidence the
        # "float/BLAS-sensitive" reason never had.
        committed = json.loads(
            (_REPO_ROOT / "logs" / "u12" / "soak_report.json").read_text(encoding="utf-8")
        )
        payload, _violations = build_soak_report(_REPO_ROOT)
        self.assertEqual(
            payload,
            committed,
            msg=(
                "committed logs/u12/soak_report.json drifted from a fresh regen — re-run "
                "`python -m src.robot.grasping.replay --soak-report` on the canonical "
                "platform and recommit."
            ),
        )


class RecordsGateTests(unittest.TestCase):
    """K2: --records-gate applies the record-intrinsic soak thresholds over a REAL log -- unlike the
    synthetic --soak-report (which passes by construction), this CAN fail, and marks the pack-dependent
    keys not_applicable."""

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
        # The dense canonical pack is small (< 2000) and has a high dead-loop rate, so the record-intrinsic
        # gate honestly FAILS (exit 1) -- the whole point of a real-log gate.
        pack = (
            _REPO_ROOT / "tests" / "data" / "replay"
            / "replay_dense_canonical_v1.jsonl"
        )
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
        # pack-dependent keys are not judgeable from a record log alone
        self.assertEqual(payload["gate"]["slo_packs_pass"], "not_applicable")
        self.assertEqual(payload["gate"]["drift_gate_pass"], "not_applicable")

    def test_records_gate_passes_on_good_log(self) -> None:
        # A >= 2000-record log of mostly-successful attempts (no dead loops, valid outcomes) passes the
        # record-intrinsic gate (exit 0); the pack-dependent keys stay not_applicable.
        records = generate_soak_records(
            SoakScenarioSpec(
                name="good",
                mode="easy",
                attempts=2100,
                failure_class_weights={"succeeded": 99.0, "no_valid_grasp": 1.0},
                recovery_success_rate=0.0,
                cycle_time_mean_s=1.5,
                cycle_time_jitter_s=0.2,
                seed=7,
            )
        )
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
        self.assertEqual(payload["gate"]["slo_packs_pass"], "not_applicable")


class SimSoakReportTests(unittest.TestCase):
    """S4: --sim-soak-report runs the record-intrinsic gate over a REAL sim log + writes a persistent
    report with an honest sim provenance banner. AUGMENTS the synthetic --soak-report (which stays
    byte-identical). pick_success_rate is REPORTED but not gated; a smaller honest sim min_attempts floor."""

    def _run(self, log_path: Path, *, min_attempts: int, out_path: Path) -> tuple[int, dict]:
        result = subprocess.run(
            [
                sys.executable, "-m", "src.robot.grasping.replay",
                "--sim-soak-report", str(log_path),
                "--sim-min-attempts", str(min_attempts),
                "--baseline-out", str(out_path),
            ],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=300,
        )
        return result.returncode, json.loads(result.stdout)

    @staticmethod
    def _good_records(attempts: int, *, success: float = 99.0, seed: int = 5):
        return generate_soak_records(
            SoakScenarioSpec(
                name="sim", mode="dense_clutter", attempts=attempts,
                failure_class_weights={"succeeded": success, "no_valid_grasp": 1.0},
                recovery_success_rate=0.0, cycle_time_mean_s=3.5, cycle_time_jitter_s=0.5, seed=seed,
            )
        )

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
            self.assertIsNone(report["baseline_pick_rate"])  # pick_success NOT gated cross-population
            self.assertEqual(report["gate"]["slo_packs_pass"], "not_applicable")
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
        # PASSES -- pick_success is reported, not gated against the (incomparable) synthetic baseline.
        with tempfile.TemporaryDirectory() as td:
            log = self._write(td, self._good_records(320, success=1.0))  # ~50% success
            out = Path(td) / "r.json"
            rc, payload = self._run(log, min_attempts=300, out_path=out)
            self.assertEqual(rc, 0, msg=str(payload.get("violations")))
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertLess(report["kpi"]["pick_success_rate"], 0.9)  # genuinely low, yet the gate passed


class SyntheticSoakUnchangedByS4Tests(unittest.TestCase):
    """S4 must AUGMENT, not alter: the synthetic soak report stays byte-identical (S4 only added a
    defaulted min_attempts param to evaluate_soak_gate_over_records, which build_soak_report doesn't call)."""

    def test_synthetic_soak_determinism_holds(self) -> None:
        p1, v1 = build_soak_report(_REPO_ROOT)
        p2, v2 = build_soak_report(_REPO_ROOT)
        self.assertEqual(p1["kpi"], p2["kpi"])
        self.assertEqual(p1["gate"], p2["gate"])
        self.assertEqual(list(v1), list(v2))
        self.assertEqual(p1["provenance"]["input"], "synthetic_generator")  # unchanged banner


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
