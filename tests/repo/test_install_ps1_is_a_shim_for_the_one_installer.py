"""``install.ps1`` is a shim for ``install.py``, and every text that names the installer names this platform's (P0h h3).

The PowerShell script held the whole install until P0h h3; ``install.py`` now does, on Windows exactly what the script
did (``test_the_one_installer_runs_what_install_ps1_ran``). What is left of the script finds a Python 3.10 or newer,
hands the installer its four switches under their Python names and returns its exit code, so a Windows user types the
same command as before. The texts that told a person to run ``install.ps1`` say ``python3 scripts/ext_deps/install.py``
off Windows, and the doctor's remedy for a missing compiled backend names the installer there, since the .bat is
Windows' only.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parents[2]
_SHIM = _REPO / "scripts" / "ext_deps" / "install.ps1"


class TheScriptOnlyRunsTheInstallerTests(unittest.TestCase):

    def setUp(self) -> None:
        self.shim = _SHIM.read_text(encoding="utf-8")

    def test_it_runs_install_py_and_returns_its_exit_code(self) -> None:
        self.assertIn("Join-Path $PSScriptRoot 'install.py'", self.shim)
        self.assertRegex(self.shim, r"& \$exe @rest \$installer @arguments\s+exit \$LASTEXITCODE")

    def test_it_installs_nothing_itself(self) -> None:
        for logic in ("micromamba", "pip install", "git clone", "descriptor.py", "build_compiled_backend",
                      "--doctor"):
            with self.subTest(logic):
                self.assertNotIn(logic, self.shim.split("#>", 1)[1])

    def test_each_switch_reaches_the_installer_under_its_python_name(self) -> None:
        for switch, flag in (("$Clean", "--clean"), ("$SkipCompiledBackend", "--skip-compiled-backend"),
                             ("$SslNoRevoke", "--ssl-no-revoke")):
            with self.subTest(switch):
                self.assertIn(f"if ({switch}) {{ $arguments += '{flag}' }}", self.shim)
        self.assertIn("$arguments = @('--component', $Component)", self.shim)
        self.assertIn("[ValidateSet('all', 'coal', 'curobo')] [string] $Component = 'all'", self.shim)

    def test_no_python_is_named_as_the_one_prerequisite_with_exit_2(self) -> None:
        self.assertRegex(self.shim, r"missing prerequisite: Python 3\.10 or newer[^\n]*\n(?:[^\n]*\n){0,3}\s*exit 2")
        self.assertIn("sys.version_info >= (3, 10)", self.shim)


class EveryTextNamesThisPlatformsInstallerTests(unittest.TestCase):

    def test_the_installer_command_per_platform(self) -> None:
        from src.utility.interpreters import installer_command

        self.assertEqual(installer_command(platform="win32"), "scripts/ext_deps/install.ps1")
        self.assertEqual(installer_command(platform="linux"), "python3 scripts/ext_deps/install.py")

    def test_the_doctors_compiled_backend_remedy_names_this_platforms_build_script(self) -> None:
        """Since P0h h5 Linux has the .bat's twin, which builds the backend alone; the installer would reinstall the
        whole cuRobo environment for it. The first Linux run of the suite (P0h h6) failed test_planning_doctor on
        the installer's text, which names no build script."""
        from src.robot.safety.planning import doctor

        payload: dict[str, object] = {"backends": ["cuda_core"],
                                      "backend_failures": {"pybind": "ImportError: no pybind module"}}
        with mock.patch.object(sys, "platform", "linux"):
            linux = doctor._kernel_backend_probe(payload, ()).remedy  # noqa: SLF001
        with mock.patch.object(sys, "platform", "win32"):
            windows = doctor._kernel_backend_probe(payload, ()).remedy  # noqa: SLF001
        self.assertEqual(linux, "build the compiled backend: bash scripts/curobo/build_compiled_backend.sh "
                                "(ext_deps/README.md 2b)")
        self.assertEqual(windows, "build the compiled backend: scripts/curobo/build_compiled_backend.bat "
                                  "(ext_deps/README.md 2b)")

    def test_no_library_text_names_install_ps1_off_windows(self) -> None:
        """The two runtime texts that named it now ask ``installer_command``; the docstrings name it as Windows'."""
        for path in (_REPO / "scripts" / "arms" / "build_arm.py", _REPO / "src" / "willy_sim" /
                     "run_sorting_demo.py"):
            with self.subTest(path.name):
                code = path.read_text(encoding="utf-8")
                self.assertIn("{installer_command()}", code)
                self.assertIsNone(re.search(r'"[^"\n]*scripts/ext_deps/install\.ps1[^"\n]*"', code),
                                  "a string literal still names install.ps1")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
