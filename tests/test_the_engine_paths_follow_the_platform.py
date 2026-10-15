"""The cuRobo interpreter and the Coal prefix are laid out per platform, byte-identical on Windows (P0h h1).

A conda or micromamba environment keeps its interpreter at ``python.exe`` in its root on Windows and at ``bin/python``
everywhere else, and a package at ``Lib/site-packages`` on Windows but ``lib/pythonX.Y/site-packages`` elsewhere, with
no DLL directory to add. The Windows layout was the only one written down: on Linux the default cuRobo interpreter was
a ``python.exe`` that never exists, and a Coal prefix was never put on ``sys.path`` because ``Lib/site-packages`` is not
where the package lies, silently, while the docstring said it did nothing off Windows. The platform is read when a
path is asked for, so each layout is tested here on either OS.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.robot.safety.planning import environment as env
from src.utility.paths import project_root


class TheCuroboInterpreterFollowsThePlatformTests(unittest.TestCase):

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(env.ENV_CUROBO_PYTHON, None)

    def test_on_windows_it_is_the_same_path_as_before(self) -> None:
        with mock.patch.object(sys, "platform", "win32"):
            self.assertEqual(env.curobo_python_path(),
                             str(project_root() / "ext_deps" / "curobo_env" / "python.exe"))

    def test_on_linux_it_is_the_environments_bin_python(self) -> None:
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(Path(env.curobo_python_path()).parts[-3:], ("curobo_env", "bin", "python"))

    def test_the_variable_wins_on_either_platform(self) -> None:
        os.environ[env.ENV_CUROBO_PYTHON] = "/opt/curobo/bin/python3"
        for platform in ("win32", "linux", "darwin"):
            with self.subTest(platform), mock.patch.object(sys, "platform", platform):
                self.assertEqual(env.curobo_python_path(), "/opt/curobo/bin/python3")

    def test_any_environment_lays_its_interpreter_out_this_way(self) -> None:
        from src.robot.safety.planning.environment import env_python

        self.assertEqual(env_python("D:/envs/x", platform="win32"), str(Path("D:/envs/x") / "python.exe"))
        for platform in ("linux", "darwin"):
            with self.subTest(platform):
                self.assertEqual(Path(env_python("/envs/x", platform=platform)).parts[-3:], ("x", "bin", "python"))


class TheCoalPrefixFollowsThePlatformTests(unittest.TestCase):

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.prefix = Path(self.enterContext(tempfile.TemporaryDirectory()))
        os.environ[env.ENV_COAL_PREFIX] = str(self.prefix)
        kept = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), kept))

    def test_the_layout_on_windows_is_the_one_it_always_was(self) -> None:
        from src.robot.safety.planning.environment import coal_prefix_layout

        layout = coal_prefix_layout("C:/coal_env", platform="win32")
        self.assertEqual(layout.dll_dir, os.path.join("C:/coal_env", "Library", "bin"))
        self.assertEqual(layout.site_packages, os.path.join("C:/coal_env", "Lib", "site-packages"))

    def test_the_layout_elsewhere_has_no_dll_directory_and_a_versioned_site_packages(self) -> None:
        from src.robot.safety.planning.environment import coal_prefix_layout

        layout = coal_prefix_layout("/opt/coal_env", platform="linux", python_version=(3, 11))
        self.assertIsNone(layout.dll_dir)
        self.assertEqual(layout.site_packages, os.path.join("/opt/coal_env", "lib", "python3.11", "site-packages"))

    def test_on_linux_the_prefix_reaches_sys_path(self) -> None:
        site = self.prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        site.mkdir(parents=True)
        added: list[str] = []
        with mock.patch.object(sys, "platform", "linux"), \
                mock.patch.object(os, "add_dll_directory", added.append, create=True):
            env.inject_coal_prefix()
        self.assertIn(str(site), sys.path)
        self.assertEqual(added, [], "no DLL directory is added off Windows")

    def test_on_windows_the_prefix_reaches_sys_path_as_before(self) -> None:
        """THE CONTROL: the layout that always worked still does."""
        (self.prefix / "Library" / "bin").mkdir(parents=True)
        site = self.prefix / "Lib" / "site-packages"
        site.mkdir(parents=True)
        added: list[str] = []
        with mock.patch.object(sys, "platform", "win32"), \
                mock.patch.object(os, "add_dll_directory", added.append, create=True):
            env.inject_coal_prefix()
        self.assertIn(os.path.join(str(self.prefix), "Lib", "site-packages"), sys.path)
        self.assertEqual(added, [os.path.join(str(self.prefix), "Library", "bin")])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
