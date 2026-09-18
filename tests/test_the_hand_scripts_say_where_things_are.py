"""The scripts a customer runs to add a hand say where the registry is, and their usage lines are commands that run.

Customer chain lane C1h (finding 31 of the audit of 2026-09-17). A customer follows a docstring before any runbook,
and three of them sent the customer to a ``grippers`` directory the loader does not read, while the jaw
measurement's first usage line passed an ``--arm`` its parser has not had since UM lane S05.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

#: Every module of the customer chain whose prose names the registry file.
_NAMING_THE_REGISTRY = (
    "scripts/grippers/write_hand_from_dimensions.py",
    "src/robot/safety/planning/robot/hand_from_dimensions.py",
    "src/robot/safety/planning/hand.py",
    "src/robot/safety/planning/environment.py",
    "src/robot/grasping/deep/hands.py",
    "tests/test_a_hand_from_its_dimensions.py",
)
#: A path ending in ``grippers/<something>.yaml``, the way a docstring names one hand's registry file.
_REGISTRY_PATH = re.compile(r"([A-Za-z_][\w./]*)/grippers/<\w+>\.yaml")


def _registry_dir() -> Path:
    from src.config.loader import _DEFAULT_DATA_DIR

    return (_DEFAULT_DATA_DIR / "grippers").resolve()


def _wrong_paths(text: str) -> list[str]:
    return [m.group(0) for m in _REGISTRY_PATH.finditer(text)
            if (_REPO / m.group(1) / "grippers").resolve() != _registry_dir()]


def _jaw_measurement():
    path = _REPO / "scripts" / "grippers" / "measure_jaw_from_bundle.py"
    spec = importlib.util.spec_from_file_location("_measure_jaw_for_usage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


class TheRegistryIsNamedWhereItIsTests(unittest.TestCase):
    def test_every_registry_path_the_hand_modules_name_is_the_registry(self) -> None:
        for rel in _NAMING_THE_REGISTRY:
            with self.subTest(module=rel):
                text = (_REPO / rel).read_text(encoding="utf-8")
                self.assertEqual(_wrong_paths(text), [], f"{rel} sends a reader to a directory the registry is not in")

    def test_the_scan_finds_a_wrong_path_and_accepts_the_right_one(self) -> None:
        """⭐ THE CONTROL. A scan that matched nothing would pass every module."""
        self.assertEqual(_wrong_paths("describe it in ``src/config/grippers/<name>.yaml``"),
                         ["src/config/grippers/<name>.yaml"])
        self.assertEqual(_wrong_paths("describe it in ``config/grippers/<name>.yaml``"), [])


class TheUsageLinesRunTests(unittest.TestCase):
    def test_every_usage_line_of_the_jaw_measurement_parses_and_runs(self) -> None:
        module, path = _jaw_measurement()
        prefix = "python scripts/grippers/measure_jaw_from_bundle.py"
        lines = [line.strip() for line in (module.__doc__ or "").splitlines() if line.strip().startswith(prefix)]
        self.assertTrue(lines, f"{path.name} shows no usage line")
        for line in lines:
            with self.subTest(usage=line):
                argv = line[len(prefix):].split()
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        code = module.main(argv)
                except SystemExit as exc:
                    self.fail(f"{line!r} exits {exc.code}: its parser refuses it")
                self.assertEqual(code, 0, line)

    def test_a_flag_the_parser_lacks_is_seen(self) -> None:
        """⭐ THE CONTROL: the parser really refuses an unknown flag, so a usage line carrying one cannot pass."""
        module, _ = _jaw_measurement()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            module.main(["robotiq_hande", "--no-such-flag"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
