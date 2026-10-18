"""The one installer, ``scripts/ext_deps/install.py``, runs on Windows exactly what ``install.ps1`` ran (P0h h3).

``tests/data/install_plan/win-64.txt`` is every external command ``install.ps1`` ran on a Windows box with nothing under
``ext_deps/``, in order, derived by reading that script before the switch. The installer's ``--plan --from-nothing``
must print it line for line, which is the proof that the Windows install did not change without reinstalling the
working ``ext_deps/`` (a fresh build there can be refused by code integrity). Around the commands it fails, refuses and
tolerates where ``install.ps1`` did; the reviewer of this step listed where a first version did not, and each of those
is a case below. Linux takes the same steps in its own layout, and the arms built are the ones this repository can
build correctly, by the rule the hand-kept list of seven stood for.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parents[2]
_INSTALL = _REPO / "scripts" / "ext_deps" / "install.py"
_TAPE = _REPO / "tests" / "data" / "install_plan" / "win-64.txt"


def _installer():  # noqa: ANN202 - the script, loaded by path as a user runs it
    name = "willy_install_under_test"
    spec = importlib.util.spec_from_file_location(name, _INSTALL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _plan(*args: str) -> list[str]:
    out = subprocess.run([sys.executable, str(_INSTALL), "--plan", "--from-nothing", *args],
                         capture_output=True, text=True, check=True).stdout
    return [line.rstrip("\r") for line in out.splitlines()]


def _python_exiting(code: int) -> tuple[str, ...]:
    return (sys.executable, "-c", f"import sys; sys.exit({code})")


class OnWindowsItRunsWhatInstallPs1RanTests(unittest.TestCase):

    def test_the_plan_is_the_tape_line_for_line_with_the_pip_lock_after_torch(self) -> None:
        """The tape is what install.ps1 ran; P0h h4 adds exactly one line, the pip lock, right after torch."""
        tape = [line for line in _TAPE.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
        torch = next(index for index, line in enumerate(tape) if "torch==2.7.0" in line)
        locked = ("run <ext_deps>/curobo_env/python.exe -m pip install --quiet -r "
                  "<ext_deps>/locks/curobo_env.win-64.pip.lock --extra-index-url https://download.pytorch.org/whl/cu128")
        self.assertEqual(_plan("--platform", "win-64"), [*tape[:torch + 1], locked, *tape[torch + 1:]])

    def test_the_arms_built_are_the_ones_with_a_bundle_and_a_fit_in_natural_order(self) -> None:
        maps = _REPO / "src" / "robot" / "safety" / "planning" / "robot"
        bundles = _REPO / "src" / "robot" / "safety" / "data"
        described = {path.stem for path in (_REPO / "config" / "arms").glob("*.yaml")}
        buildable = {arm for arm in described
                     if (bundles / f"{arm}_collision_meshes.npz").is_file() and (maps / f"{arm}_arm_spheres.yml").is_file()}
        arms = list(_installer().buildable_arms())
        self.assertEqual(set(arms), buildable)
        self.assertEqual(arms, ["ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e"])
        self.assertLess(len(buildable), len(described), "an arm with no bundle (ur20 today) is described and not built")


class AroundTheCommandsItDoesWhatInstallPs1DidTests(unittest.TestCase):
    """The reviewer's list: each place a first version of install.py failed, refused or tolerated differently."""

    def setUp(self) -> None:
        self.installer = _installer()
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.layout = self.installer.Layout("win-64", self.folder)

    def test_every_missing_prerequisite_is_named_before_anything_and_exits_2(self) -> None:
        steps = [self.installer.Step("require", argv=("willy-no-such-tool-a",)),
                 self.installer.Step("require", argv=("willy-no-such-tool-b",)),
                 self.installer.Step("download", source="https://example.invalid/x", target=self.folder / "x")]
        said = io.StringIO()
        with contextlib.redirect_stderr(said):
            outcome = self.installer.execute(steps, self.layout)
        self.assertEqual(outcome.code, 2)
        self.assertIn("missing prerequisite(s): willy-no-such-tool-a, willy-no-such-tool-b", said.getvalue())
        self.assertFalse((self.folder / "x").exists(), "nothing is downloaded before the prerequisites are there")

    def test_clean_with_a_missing_lock_refuses_before_it_removes_the_environment(self) -> None:
        (self.folder / "coal_env" / "kept").mkdir(parents=True)
        self.layout.mamba.parent.mkdir(parents=True)
        self.layout.mamba.write_bytes(b"")  # micromamba is there, so nothing is downloaded
        steps = self.installer.plan(self.layout, component="coal", clean=True)
        with self.assertRaises(self.installer.InstallRefused) as refused, contextlib.redirect_stdout(io.StringIO()):
            self.installer.execute([step for step in steps if step.kind != "require"], self.layout)
        self.assertIn("no lockfile at", str(refused.exception))
        self.assertTrue((self.folder / "coal_env" / "kept").is_dir(), "the environment was removed before the refusal")

    def test_a_removal_takes_read_only_files_as_remove_item_force_did(self) -> None:
        pack = self.folder / "curobo" / ".git" / "objects" / "pack" / "pack-1.pack"
        pack.parent.mkdir(parents=True)
        pack.write_bytes(b"git writes its packs read-only")
        os.chmod(pack, stat.S_IREAD)
        self.installer._remove(self.folder / "curobo")  # noqa: SLF001
        self.assertFalse((self.folder / "curobo").exists())

    def test_the_fetch_and_the_unpacking_are_tolerated_and_what_they_should_leave_is_checked(self) -> None:
        steps = self.installer.plan(self.installer.Layout("win-64", self.folder, fresh=True), component="all")
        tar = next(i for i, step in enumerate(steps) if step.kind == "run" and step.argv[:1] == ("tar",))
        fetch = next(step for step in steps if step.argv[:2] == ("git", "fetch"))
        self.assertTrue(steps[tar].tolerate and fetch.tolerate)
        self.assertEqual((steps[tar + 1].kind, steps[tar + 1].target), ("need", self.layout.mamba))
        with contextlib.redirect_stdout(io.StringIO()):
            code = self.installer._run(self.installer.Step("run", _python_exiting(1), tolerate=True), {},  # noqa: SLF001
                                       self.layout)
        self.assertEqual(code, 1, "a tolerated failure is returned, not raised")

    def test_the_curobo_interpreter_is_checked_after_its_environment(self) -> None:
        steps = self.installer.plan(self.installer.Layout("win-64", self.folder, fresh=True), component="curobo")
        needs = [step.target for step in steps if step.kind == "need"]
        self.assertIn(self.layout.env_python("curobo_env"), needs)

    def test_a_failed_download_is_a_refusal_with_its_reason(self) -> None:
        step = self.installer.Step("download", source="https://example.invalid/x", target=self.folder / "x")
        with mock.patch("urllib.request.urlretrieve", side_effect=OSError("no route")), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(self.installer.InstallRefused) as refused:
            self.installer._download(step, self.layout)  # noqa: SLF001
        self.assertIn("downloading https://example.invalid/x failed: no route", str(refused.exception))

    def test_no_main_environment_is_refused_and_not_green(self) -> None:
        """P0h h3d, on purpose: install.ps1 warned and exited 0 here, green on the box that has nothing yet."""
        step = self.installer.Step("run", (str(self.folder / "no_python.exe"), "-m", "x", "--doctor"), doctor=True)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(self.installer.InstallRefused) as refused:
            self.installer.execute([step], self.layout)
        said = str(refused.exception)
        self.assertIn("nothing verified that this box can plan", said)
        self.assertIn(str(self.folder / "no_python.exe"), said)
        self.assertIn("python -m src.robot.safety.planning --doctor", said)

    def test_another_ext_deps_reaches_the_build_script_and_the_default_adds_nothing(self) -> None:
        other = next(step for step in self.installer.plan(self.installer.Layout("win-64", self.folder, fresh=True),
                                                          component="curobo") if step.argv[:2] == ("cmd", "/c"))
        self.assertEqual(other.env, {"CUROBO_ENV": str(self.folder / "curobo_env"),
                                     "CUROBO_SRC": str(self.folder / "curobo")})
        default = next(step for step in self.installer.plan(
            self.installer.Layout("win-64", self.installer.DEFAULT_EXT_DEPS, fresh=True), component="curobo")
            if step.argv[:2] == ("cmd", "/c"))
        self.assertEqual(default.env, {})


class OnLinuxItTakesTheSameStepsInItsOwnLayoutTests(unittest.TestCase):

    def setUp(self) -> None:
        self.plan = _plan("--platform", "linux-64")
        self.text = "\n".join(self.plan)

    def test_nothing_of_windows_is_in_it(self) -> None:
        for windows in (".exe", "cmd /c", ".bat", "win-64", "require tar", "Scripts/"):
            with self.subTest(windows):
                self.assertNotIn(windows, self.text)

    def test_the_same_pins_in_the_linux_layout(self) -> None:
        for line in ("download https://github.com/mamba-org/micromamba-releases/releases/latest/download/"
                     "micromamba-linux-64 -> <ext_deps>/micromamba/bin/micromamba",
                     "run <ext_deps>/micromamba/bin/micromamba create -y -p <ext_deps>/coal_env --file "
                     "<ext_deps>/locks/coal_env.linux-64.lock -c conda-forge",
                     "run <ext_deps>/curobo_env/bin/python -m pip install --quiet torch==2.7.0 --index-url "
                     "https://download.pytorch.org/whl/cu128",
                     "in <ext_deps>/curobo: run git checkout --quiet 8e734f3",
                     "run <ext_deps>/curobo_env/bin/python -m pip install --quiet -r "
                     "<ext_deps>/locks/curobo_env.linux-64.pip.lock --extra-index-url https://download.pytorch.org/whl/cu128",
                     "run <ext_deps>/curobo_env/bin/python -m pip install --quiet cuda-core==1.1.0",
                     "run <ext_deps>/curobo_env/bin/python -m pip install --quiet warp-lang==1.16.0"):
            with self.subTest(line):
                self.assertIn(line, self.plan)

    def test_the_compiled_backend_is_built_by_the_bat_s_linux_twin(self) -> None:
        """P0h h5, measured in Ubuntu-24.04: cuRobo's bare ``CUROBO_USE_PYBIND=1 setup.py build_ext`` found no ``gcc``
        there, and with the environment's compilers on PATH no ``cuda_runtime_api.h`` (conda-forge keeps the CUDA
        headers under ``targets/x86_64-linux/include``). The twin supplies both; without CUROBO_USE_PYBIND=1
        setup.py compiles nothing."""
        self.assertIn("run bash <repo>/scripts/curobo/build_compiled_backend.sh", self.plan)
        self.assertNotIn("setup.py", self.text)
        script = (_REPO / "scripts" / "curobo" / "build_compiled_backend.sh").read_text(encoding="utf-8")
        for needed in ("export CUROBO_USE_PYBIND=1", "export CUDA_HOME=$CUROBO_ENV", "export PATH=$CUROBO_ENV/bin:$PATH",
                       "export CPATH=$CUROBO_ENV/targets/x86_64-linux/include"):
            with self.subTest(needed):
                self.assertIn(needed, script)

    def test_the_twin_fails_the_install_where_the_bat_s_exit_2_was_an_emergency(self) -> None:
        """On Windows exit 2 is "no toolchain", which needs administrator rights, so the install goes on with one
        backend. On Linux the lock carries the toolchain, so an exit 2 is an environment not made from the lock."""
        installer = _installer()
        step = installer._compiled_backend(installer.Layout("linux-64", installer.DEFAULT_EXT_DEPS, fresh=True))  # noqa: SLF001
        self.assertEqual((step.emergency, step.tolerate, step.env), ((), False, {}))

    def test_another_ext_deps_reaches_the_twin_as_it_reaches_the_bat(self) -> None:
        installer = _installer()
        folder = Path("/elsewhere/ext_deps")
        step = installer._compiled_backend(installer.Layout("linux-64", folder, fresh=True))  # noqa: SLF001
        self.assertEqual(step.env, {"CUROBO_ENV": str(folder / "curobo_env"), "CUROBO_SRC": str(folder / "curobo")})

    def test_the_main_environment_is_made_and_the_doctor_runs_under_it_last(self) -> None:
        main = ["run <ext_deps>/micromamba/bin/micromamba create -y -p <repo>/.venv python=3.11 -c conda-forge",
                "in <repo>: run <repo>/.venv/bin/python -m pip install --quiet -r <repo>/requirements.txt",
                "in <repo>: run <repo>/.venv/bin/python -m pip install --quiet -e <repo> --no-deps"]
        start = self.plan.index(main[0])
        self.assertEqual(self.plan[start:start + 3], main)
        self.assertTrue(self.plan[start + 3].startswith("system portaudio: "), "the microphone's system library is named")
        self.assertEqual(self.plan[-1], "in <repo>: run <repo>/.venv/bin/python -m src.robot.safety.planning "
                                        "--doctor")

    def test_the_realsense_system_library_is_named_beside_the_microphones(self) -> None:
        """P0h h6, measured on the first Linux install: requirements.txt installed pyrealsense2 2.58.3, and it failed to
        import on libusb-1.0.so.0, which only root can install."""
        start = self.plan.index("in <repo>: run <repo>/.venv/bin/python -m pip install --quiet -e <repo> --no-deps")
        self.assertEqual(self.plan[start + 2], "system usb-1.0: an Intel RealSense camera needs libusb-1.0, which only "
                                               "root can install: sudo apt install libusb-1.0-0")

    def test_a_missing_system_library_is_named_by_its_own_name(self) -> None:
        installer = _installer()
        layout = installer.Layout("linux-64", Path(tempfile.gettempdir()))
        steps = [installer.Step("system", argv=("willy-no-such-library",), source="needs it: apt install it")]
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            installer.execute(steps, layout)
        self.assertIn("WARNING: needs it: apt install it", said.getvalue())


class WhatItRefusesTests(unittest.TestCase):

    def _refused(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(_INSTALL), *args], capture_output=True, text=True)

    def test_another_platform_only_plans(self) -> None:
        other = "linux-64" if sys.platform == "win32" else "win-64"
        done = self._refused("--platform", other)
        self.assertEqual(done.returncode, 2)
        self.assertIn("it runs only with --plan", done.stderr)

    def test_from_nothing_only_plans(self) -> None:
        done = self._refused("--from-nothing")
        self.assertEqual(done.returncode, 2)
        self.assertIn("an install always starts from what is there", done.stderr)

    def test_the_windows_main_environment_is_not_the_installers(self) -> None:
        done = self._refused("--plan", "--platform", "win-64", "--component", "main")
        self.assertEqual(done.returncode, 2)
        self.assertIn("on Windows .venv is not the installer's", done.stderr)


class ThePipHalfIsLockedTests(unittest.TestCase):
    """P0h h4: what cuRobo's [cu12] pulls with pip was pinned nowhere, so a reinstall got whatever was newest."""

    def setUp(self) -> None:
        self.installer = _installer()
        self.lock = _REPO / "ext_deps" / "locks" / "curobo_env.win-64.pip.lock"
        self.pins = [line for line in self.lock.read_text(encoding="utf-8").splitlines()
                     if line and not line.startswith("#")]

    def test_every_line_pins_one_version(self) -> None:
        self.assertGreater(len(self.pins), 50)
        for line in self.pins:
            with self.subTest(line):
                self.assertRegex(line, r"^[A-Za-z0-9._-]+==[A-Za-z0-9.+!-]+$")

    def test_it_holds_the_installers_own_pins(self) -> None:
        installer = self.installer
        for pin in (f"{installer.TORCH_SPEC}+cu128", installer.CUDA_CORE_PIN, installer.WARP_PIN):
            with self.subTest(pin):
                self.assertIn(pin, self.pins)

    def test_neither_cuRobo_itself_nor_a_conda_package_is_in_it(self) -> None:
        names = {line.split("==")[0].lower() for line in self.pins}
        for absent in ("nvidia-curobo", "setuptools", "wheel", "pip", "packaging"):
            with self.subTest(absent):
                self.assertNotIn(absent, names)

    def test_the_lock_is_written_from_what_pip_installed(self) -> None:
        inspected = {"installed": [
            {"installer": "pip", "metadata": {"name": "Torch", "version": "2.7.0+cu128"}},
            {"installer": "pip", "metadata": {"name": "anyio", "version": "4.14.2"}},
            {"installer": "pip", "metadata": {"name": "nvidia-curobo", "version": "0.0.0"},
             "direct_url": {"dir_info": {"editable": True}}},
            {"installer": "conda", "metadata": {"name": "pip", "version": "26.2.1"}},
            {"metadata": {"name": "setuptools", "version": "84.0.0"}},
        ]}
        self.assertEqual(self.installer.pip_lock_lines(inspected), ["anyio==4.14.2", "Torch==2.7.0+cu128"])


class ItNeedsNothingButPythonTests(unittest.TestCase):

    def test_it_imports_only_the_standard_library(self) -> None:
        tree = ast.parse(_INSTALL.read_text(encoding="utf-8"))
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom) and node.module and node.level == 0}
        self.assertEqual(sorted(imported - set(sys.stdlib_module_names) - {"__future__"}), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
