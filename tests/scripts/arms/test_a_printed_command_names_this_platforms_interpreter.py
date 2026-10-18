"""A refusal that ends in a command names the interpreter this platform runs it with (P0h h2).

Every text below told a person which command fixes what it refused, and every one named the Windows interpreter:
``.venv/Scripts/python.exe``, ``ext_deps/curobo_env/python.exe`` or Isaac's ``python.bat``. On Linux none of them
exists (``.venv/bin/python``, ``ext_deps/curobo_env/bin/python``, ``python.sh``), so the command a refusal ended in
could not run. The texts now ask ``src.utility.interpreters``, which reads the platform when a text is written:
on Windows each is the text it was, byte for byte, which the tests that pin them keep holding.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_REPO = Path(__file__).resolve().parents[3]


def _fit_script():  # noqa: ANN202 - a module loaded from its path, as the other fit tests load it
    name = "_fit_under_test_interpreters"
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "arms" / "fit.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _each_refusal() -> dict[str, str]:
    """Every text that names an interpreter, written now, under whatever ``sys.platform`` says."""
    from src.robot.safety.planning import hand
    from src.robot.safety.planning.robot import retract_table
    from src.robot.safety.planning.evidence import gate_command
    from src.willy_sim.harness.bootstrap import motion_stack_banner
    from datagen.render.engine import engine_is_available

    said: dict[str, str] = {}
    said["gate command"] = gate_command(arm="ur5e", hand="robotiq_2f85", coupling_mm=0.0, approach="+Z",
                                        closing="+X", planner_margin_mm=4.0, guard_margin_mm=10.0)
    with tempfile.TemporaryDirectory() as folder:
        try:
            retract_table.read_arm_retract(Path(folder) / "retract.yaml", "ur5e")
        except retract_table.RetractMissing as exc:
            said["no retract table"] = str(exc)
        table = Path(folder) / "table.yaml"
        table.write_text("retracts:\n  - {arm: ur3e, hand: robotiq_2f85, plate_mm: 0.0}\n", encoding="utf-8")
        try:
            retract_table.read_arm_retract(table, "ur5e")
        except retract_table.RetractMissing as exc:
            said["no row for the arm"] = str(exc)
        table.write_text("retracts:\n  - {arm: ur5e, hand: robotiq_hande, plate_mm: 0.0}\n", encoding="utf-8")
        try:
            retract_table.read_arm_retract(table, "ur5e")
        except retract_table.RetractMissing as exc:
            said["no fallback row"] = str(exc)
    said["retract command"] = retract_table.retract_command("ur5e", "robotiq_2f85", 0.0, None, "+Z+X")
    said["descriptor"] = hand.descriptor_refusal(SimpleNamespace(provenance=None), SimpleNamespace(model="robotiq_2f85"),
                                                arm="ur5e") or ""
    said["banner"] = motion_stack_banner(object(), planner="curobo", missing=["cuRobo planner"])
    with mock.patch("importlib.util.find_spec", return_value=None):
        said["isaac engine"] = engine_is_available("isaac")[1]
    fit = _fit_script()
    try:
        fit.planned({"lfinger": 1}, {}, target="--hand schunk_egu50")
    except SystemExit as exc:
        said["fit plan"] = str(exc)
    return said


class TheHelperNamesEachInterpreterPerPlatformTests(unittest.TestCase):

    def test_on_windows_the_names_every_text_used(self) -> None:
        from src.utility.interpreters import Interpreter, interpreter_command

        self.assertEqual(interpreter_command(Interpreter.MAIN, platform="win32"), ".venv/Scripts/python.exe")
        self.assertEqual(interpreter_command(Interpreter.CUROBO, platform="win32"), "ext_deps/curobo_env/python.exe")
        self.assertEqual(interpreter_command(Interpreter.ISAAC, platform="win32"), "python.bat")

    def test_elsewhere_bin_python_and_python_sh(self) -> None:
        from src.utility.interpreters import Interpreter, interpreter_command

        for platform in ("linux", "darwin"):
            with self.subTest(platform):
                self.assertEqual(interpreter_command(Interpreter.MAIN, platform=platform), ".venv/bin/python")
                self.assertEqual(interpreter_command(Interpreter.CUROBO, platform=platform),
                                 "ext_deps/curobo_env/bin/python")
                self.assertEqual(interpreter_command(Interpreter.ISAAC, platform=platform), "python.sh")

    def test_the_curobo_command_and_the_interpreter_the_planner_starts_share_one_layout(self) -> None:
        from src.robot.safety.planning.environment import env_python
        from src.utility.interpreters import Interpreter, interpreter_command

        for platform in ("win32", "linux"):
            with self.subTest(platform):
                started = Path(env_python("ext_deps/curobo_env", platform=platform))
                self.assertEqual(started.as_posix(), interpreter_command(Interpreter.CUROBO, platform=platform))


class EveryRefusalNamesThisPlatformsInterpreterTests(unittest.TestCase):

    def test_on_linux_no_refusal_names_a_windows_interpreter(self) -> None:
        with mock.patch.object(sys, "platform", "linux"):
            said = _each_refusal()
        self.assertEqual(len(said), 9, sorted(said))
        for name, text in said.items():
            with self.subTest(name):
                for windows in ("Scripts/python.exe", "curobo_env/python.exe", "python.bat", "python.exe"):
                    self.assertNotIn(windows, text)
        self.assertIn(".venv/bin/python scripts/arms/gate.py", said["gate command"])
        for name in ("no retract table", "no row for the arm", "no fallback row", "retract command"):
            with self.subTest(name):
                self.assertIn(".venv/bin/python scripts/arms/retract.py", said[name])
        self.assertIn("ext_deps/curobo_env/bin/python scripts/arms/descriptor.py ur5e", said["descriptor"])
        self.assertIn("the cuRobo env's bin/python", said["banner"].replace("#", "").replace("\n", " ").replace("  ", " "))
        self.assertIn("`python.sh`", said["isaac engine"])
        self.assertIn(".venv/bin/python scripts/arms/fit.py", said["fit plan"])

    def test_on_windows_every_refusal_says_what_it_said(self) -> None:
        """THE CONTROL: byte for byte on Windows, the interpreter every text named before."""
        with mock.patch.object(sys, "platform", "win32"):
            said = _each_refusal()
        self.assertIn(".venv/Scripts/python.exe scripts/arms/gate.py", said["gate command"])
        for name in ("no retract table", "no row for the arm", "no fallback row", "retract command"):
            with self.subTest(name):
                self.assertIn(".venv/Scripts/python.exe scripts/arms/retract.py", said[name])
        self.assertIn("ext_deps/curobo_env/python.exe scripts/arms/descriptor.py ur5e", said["descriptor"])
        self.assertIn("the cuRobo env's python.exe", said["banner"].replace("#", "").replace("\n", " ").replace("  ", " "))
        self.assertIn("`python.bat`", said["isaac engine"])
        self.assertIn(".venv/Scripts/python.exe scripts/arms/fit.py", said["fit plan"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
