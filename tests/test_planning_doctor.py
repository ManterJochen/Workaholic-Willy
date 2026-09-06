"""The motion-stack doctor: classification, exit-code contract, and the report it prints.

These tests deliberately never touch a real engine. The doctor's VALUE is its judgement -- "an OS policy
refused this" vs "this is not installed" vs "one backend is carrying the planner" -- and that judgement
has to hold on CI, on macOS, and on a box with no GPU. The engine-loading paths are exercised on-box by
`--doctor` itself.
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.robot.safety.planning import doctor as doc


class LooksPolicyBlockedTests(unittest.TestCase):
    """The block detector, whose whole job is to not miss a refusal in a language it was not written in."""

    def test_english_message_is_detected(self) -> None:
        self.assertTrue(doc.looks_policy_blocked(
            "ImportError: DLL load failed while importing _event: "
            "This file is blocked by an application control policy."
        ))

    def test_german_message_is_detected(self) -> None:
        # The message is localized by Windows, so the machine that first hit this reported it in German.
        self.assertTrue(doc.looks_policy_blocked(
            "ImportError: DLL load failed while importing _bz2: "
            "Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert."
        ))

    def test_unknown_locale_is_caught_via_the_event_log(self) -> None:
        """A locale we have no fragment for still resolves, because the event log names the file."""
        blocks = (r"\Device\HarddiskVolume5\ext_deps\curobo_env\Lib\site-packages\cuda\_event.pyd",)
        self.assertTrue(doc.looks_policy_blocked(
            "ImportError: DLL load failed while importing _event.pyd: <some other language>",
            blocks=blocks,
        ))

    def test_an_ordinary_import_error_is_not_a_policy_block(self) -> None:
        self.assertFalse(doc.looks_policy_blocked("ModuleNotFoundError: No module named 'coal'"))
        self.assertFalse(doc.looks_policy_blocked("ImportError: DLL load failed: not a valid Win32 app"))

    def test_an_unrelated_blocked_file_does_not_match(self) -> None:
        """The event log is machine-wide; a block in someone else's software must not be claimed."""
        self.assertFalse(doc.looks_policy_blocked(
            "ModuleNotFoundError: No module named 'coal'",
            blocks=(r"\Device\HarddiskVolume3\Program Files\Unrelated\thing.dll",),
        ))


class KernelBackendProbeTests(unittest.TestCase):
    """cuRobo's two kernel backends -- the redundancy whose absence caused the 2026-08-10 outage."""

    def test_both_backends_is_ok(self) -> None:
        probe = doc._kernel_backend_probe({"backends": ["cuda_core", "pybind"]}, ())
        self.assertIs(probe.status, doc.ProbeStatus.OK)

    def test_one_backend_warns_but_does_not_fail(self) -> None:
        """One backend plans exactly as well as two. What it lacks is a spare -- that is a warning."""
        probe = doc._kernel_backend_probe(
            {"backends": ["pybind"], "backend_failures": {"cuda_core": "ModuleNotFoundError: cuda.core"}}, ()
        )
        self.assertIs(probe.status, doc.ProbeStatus.WARN)
        self.assertTrue(probe.ok)

    def test_the_remedy_names_the_backend_that_is_actually_missing(self) -> None:
        """The first version of this probe told operators to build a backend that was already built."""
        probe = doc._kernel_backend_probe(
            {"backends": ["pybind"], "backend_failures": {"cuda_core": "ModuleNotFoundError: cuda.core"}}, ()
        )
        self.assertIn("cuda-core", probe.remedy)
        self.assertNotIn("build_compiled_backend", probe.remedy)

        other = doc._kernel_backend_probe(
            {"backends": ["cuda_core"], "backend_failures": {"pybind": "ImportError: geom_cu"}}, ()
        )
        self.assertIn("build_compiled_backend", other.remedy)

    def test_a_policy_blocked_backend_says_so_and_names_the_survivor(self) -> None:
        probe = doc._kernel_backend_probe(
            {
                "backends": ["pybind"],
                "backend_failures": {
                    "cuda_core": "ImportError: DLL load failed while importing _event: "
                                 "Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert."
                },
            },
            (),
        )
        self.assertIs(probe.status, doc.ProbeStatus.WARN)
        self.assertIn("cuda_core", probe.detail)
        self.assertIn("pybind", probe.detail)
        self.assertIn("code-integrity", probe.remedy)

    def test_no_backend_at_all_is_broken(self) -> None:
        probe = doc._kernel_backend_probe({"backends": [], "backend_failures": {}}, ())
        self.assertIs(probe.status, doc.ProbeStatus.BROKEN)
        self.assertFalse(probe.ok)


class ExitCodeContractTests(unittest.TestCase):
    """0 healthy | 1 degraded | 2 blocked by policy -- the contract the install script switches on."""

    @staticmethod
    def _report(*statuses: doc.ProbeStatus, blocked_files: tuple[str, ...] = ()) -> doc.DoctorReport:
        probes = tuple(doc.Probe(f"probe{i}", s, "detail") for i, s in enumerate(statuses))
        return doc.DoctorReport(probes=probes, blocked_files=blocked_files)

    def test_all_ok_exits_zero(self) -> None:
        self.assertEqual(self._report(doc.ProbeStatus.OK, doc.ProbeStatus.OK).exit_code, 0)

    def test_a_warning_still_exits_zero(self) -> None:
        """A box with one kernel backend can plan. Failing it would make the install script lie."""
        report = self._report(doc.ProbeStatus.OK, doc.ProbeStatus.WARN)
        self.assertEqual(report.exit_code, 0)
        self.assertTrue(report.healthy)

    def test_missing_exits_one(self) -> None:
        self.assertEqual(self._report(doc.ProbeStatus.OK, doc.ProbeStatus.MISSING).exit_code, 1)

    def test_broken_exits_one(self) -> None:
        self.assertEqual(self._report(doc.ProbeStatus.BROKEN).exit_code, 1)

    def test_blocked_exits_two_and_outranks_missing(self) -> None:
        """A policy block is the diagnosis; anything else it causes downstream is a symptom."""
        report = self._report(doc.ProbeStatus.MISSING, doc.ProbeStatus.BLOCKED)
        self.assertEqual(report.exit_code, 2)
        self.assertTrue(report.blocked)

    def test_the_report_names_blocked_files_and_the_remedy(self) -> None:
        report = self._report(doc.ProbeStatus.BLOCKED, blocked_files=(r"\Device\X\ext_deps\thing.pyd",))
        text = report.render()
        self.assertIn("thing.pyd", text)
        self.assertIn("code-integrity.md", text)
        self.assertIn("PIN", text)


class CodeIntegrityBlocksTests(unittest.TestCase):
    """The event-log reader is a diagnostic aid -- it must never become a reason the doctor fails."""

    def test_non_windows_returns_empty_without_spawning_anything(self) -> None:
        with mock.patch.object(doc.sys, "platform", "linux"), \
             mock.patch.object(doc.subprocess, "run") as run:
            self.assertEqual(doc.code_integrity_blocks(), ())
        run.assert_not_called()

    def test_a_powershell_failure_is_swallowed(self) -> None:
        with mock.patch.object(doc.sys, "platform", "win32"), \
             mock.patch.object(doc.subprocess, "run", side_effect=OSError("no powershell")):
            self.assertEqual(doc.code_integrity_blocks(), ())

    def test_a_timeout_is_swallowed(self) -> None:
        timeout = doc.subprocess.TimeoutExpired(cmd="powershell", timeout=30)
        with mock.patch.object(doc.sys, "platform", "win32"), \
             mock.patch.object(doc.subprocess, "run", side_effect=timeout):
            self.assertEqual(doc.code_integrity_blocks(), ())

    def test_output_is_parsed_and_deduplicated_in_order(self) -> None:
        completed = mock.Mock(stdout="  a.dll \n b.pyd\na.dll\n\n", stderr="")
        with mock.patch.object(doc.sys, "platform", "win32"), \
             mock.patch.object(doc.subprocess, "run", return_value=completed):
            self.assertEqual(doc.code_integrity_blocks(), ("a.dll", "b.pyd"))


if __name__ == "__main__":
    unittest.main()
