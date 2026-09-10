"""The motion-stack doctor: classification, exit-code contract, and the report it prints.

These tests deliberately never touch a real engine. The doctor's VALUE is its judgement -- "an OS policy
refused this" vs "this is not installed" vs "one backend is carrying the planner" -- and that judgement
has to hold on CI, on macOS, and on a box with no GPU. The engine-loading paths are exercised on-box by
`--doctor` itself.
"""

from __future__ import annotations

import contextlib
import importlib
import pathlib
import sys
import tempfile
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
        self.assertIn("pin the dependency", text)


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


class CoalPolicyRefusalTests(unittest.TestCase):
    r"""⛔ THE MEASURED FALSE GREEN. On 2026-09-10 the CodeIntegrity log of this workstation
    recorded event 3033 for this repository's own
    ``ext_deps\coal_env\Library\bin\coal.dll``: the OS refused a build we ship.
    ``--doctor`` exited 0 with ``policy_blocked=false``, because `import_collision_engine()`
    swallows Coal's exception and returns python-fcl, so the BLOCKED branch of `_probe_coal` can
    never see a refusal and is unreachable from the command line.

    The refusal is reproduced here by a `coal` module that raises the exact OSError Windows raises,
    placed ahead of the real one on `sys.path`: no mock of the unit under test, the same exception on
    the same import.
    """

    #: The message Windows produced on this box, verbatim (German locale, WinError 4551).
    BLOCK = ("[WinError 4551] Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert: "
             r"'ext_deps\coal_env\Library\bin\coal.dll'")

    @contextlib.contextmanager
    def _coal_refused_by_the_os(self):
        """Make ``import coal`` raise what the OS raised, for the duration of the block."""
        with tempfile.TemporaryDirectory() as tmp:
            (pathlib.Path(tmp) / "coal.py").write_text(
                f"raise OSError({self.BLOCK!r})\n", encoding="utf-8"
            )
            saved_path, saved_module = list(sys.path), sys.modules.pop("coal", None)
            sys.path.insert(0, tmp)
            importlib.invalidate_caches()
            try:
                yield
            finally:
                sys.path[:] = saved_path
                sys.modules.pop("coal", None)
                if saved_module is not None:
                    sys.modules["coal"] = saved_module
                importlib.invalidate_caches()

    def test_the_refusal_reaches_the_probe_instead_of_being_swallowed(self) -> None:
        with self._coal_refused_by_the_os():
            probe = doc._probe_coal(())
        self.assertIs(probe.status, doc.ProbeStatus.BLOCKED,
                      f"an OS refusal of coal.dll reported as {probe.status}: {probe.detail}")
        self.assertIn("code-integrity", probe.remedy)

    def test_the_exit_code_is_two_so_the_install_script_can_branch(self) -> None:
        """0 all green, 1 degraded, 2 an OS policy is blocking a binary. It answered 0."""
        curobo_ok = (doc.Probe("cuRobo planner sidecar", doc.ProbeStatus.OK, "faked"),)
        with self._coal_refused_by_the_os(), \
             mock.patch.object(doc, "_probe_curobo", return_value=curobo_ok), \
             mock.patch.object(doc, "code_integrity_blocks", return_value=()):
            report = doc.run_doctor(model="ur5e")
        self.assertTrue(report.blocked, report.render())
        self.assertEqual(report.exit_code, 2)

    def test_the_detail_names_what_carried_the_guard(self) -> None:
        """BLOCKED must not read as "no exact-mesh checking": python-fcl is still holding the guard,
        and an operator who cannot see that will go looking for a second failure."""
        with self._coal_refused_by_the_os():
            probe = doc._probe_coal(())
        self.assertIn("fcl", probe.detail)
        self.assertIn("4551", probe.detail, "the detail must quote what the OS actually said")


class EveryBlockedSiteIsReachableTests(unittest.TestCase):
    """⛔ THE COAL SITE WAS DEAD CODE, SO THE OTHER THREE ARE WORTH PROVING RATHER THAN ASSUMING.

    ``looks_policy_blocked`` is consulted at four places, and one of them could not fire at all (the
    class above). That is the reason this class exists: a branch nobody can reach and a branch nobody
    has tested look identical from the outside. The kernel-backend site is covered by
    `KernelBackendProbeTests`; these are the two in `_probe_curobo`, driven through a faked sidecar,
    since what is under test is the classification and not the subprocess.
    """

    #: What the sidecar reports when the OS refuses one of cuRobo's DLLs. Same family as the live
    #: case on this box: CodeIntegrity event 3033 for
    #: ``ext_deps/curobo_env/Lib/site-packages/warp/bin/warp-clang.dll``, recorded 2026-09-10.
    REFUSAL = ("ImportError: DLL load failed while importing warp-clang: "
               "Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert.")

    def _sidecar(self, *, stdout: str = "", stderr: str = "") -> doc.DoctorReport:
        """The probe against an interpreter that exists and answers with exactly this."""
        completed = mock.Mock(stdout=stdout, stderr=stderr)
        with mock.patch.object(doc, "curobo_python_path", return_value=sys.executable), \
             mock.patch.object(doc.subprocess, "run", return_value=completed):
            probes = doc._probe_curobo((), "ur5e.yml")
        return doc.DoctorReport(probes=probes, blocked_files=())

    def test_a_sidecar_that_prints_nothing_but_a_refusal_is_blocked(self) -> None:
        report = self._sidecar(stderr=self.REFUSAL)
        self.assertIs(report.probes[0].status, doc.ProbeStatus.BLOCKED)
        self.assertEqual(report.exit_code, 2)

    def test_a_refusal_reported_inside_the_payload_is_blocked(self) -> None:
        """The sidecar catches its own import errors and prints them as JSON, so this is the shape
        the live path actually produces."""
        report = self._sidecar(stdout='{"python": "x", "error": "curobo: %s"}' % self.REFUSAL)
        self.assertIs(report.probes[0].status, doc.ProbeStatus.BLOCKED)
        self.assertIn("code-integrity", report.probes[0].remedy)

    def test_an_ordinary_import_failure_is_broken_not_blocked(self) -> None:
        """The control: without the refusal the same shape must NOT claim a policy block, or exit 2
        stops meaning anything."""
        report = self._sidecar(stdout='{"python": "x", "error": "torch: ModuleNotFoundError: torch"}')
        self.assertIs(report.probes[0].status, doc.ProbeStatus.BROKEN)
        self.assertEqual(report.exit_code, 1)


if __name__ == "__main__":
    unittest.main()
