"""The guide and the config README describe the gripper registry as it is (customer chain lane C1j).

Each retracted sentence is paired, in the same test, with the behaviour that refutes it, so the absence of a sentence
cannot pass on a validator or a driver that really had regressed (negative controls rot by growth).
"""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_GUIDE = _REPO / "docs" / "guide"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TheGuideDescribesTheRegistryTests(unittest.TestCase):
    def test_the_guide_and_the_validator_agree_about_a_broken_hand_file(self) -> None:
        from src.config.__main__ import main

        root = Path(self.enterContext(tempfile.TemporaryDirectory())) / "data"
        shutil.copytree(_REPO / "config", root)
        (root / "grippers" / "robotiq_2f85.yaml").write_text("gripper: [never closed\n", encoding="utf-8")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--data", str(root)]), 1)
        for page in ("06-grippers.md", "01-configuration.md"):
            with self.subTest(page=page):
                self.assertNotIn("still passes `python -m src.config`", _text(_GUIDE / page))
                self.assertNotIn("a malformed hand file\npasses it", _text(_GUIDE / page).replace("\r\n", "\n"))

    def test_the_guide_and_the_driver_agree_that_the_pick_path_reads_the_hand(self) -> None:
        from src.config.loader import ConfigError
        from src.config.schema.robot import RobotConfig
        from src.robot.drivers.ur.arm import URRobotArm

        with self.assertRaises(ConfigError):
            URRobotArm(RobotConfig.model_validate({"vendor": "ur", "gripper": {"model": "no_such_hand"}}))
        for page in (_GUIDE / "01-configuration.md", _GUIDE / "04-robot-and-safety.md",
                     _REPO / "src" / "config" / "README.md"):
            with self.subTest(page=page.name):
                text = _text(page)
                self.assertNotIn("by nothing on the pick path yet", text)
                self.assertNotIn("Nothing on the pick path reads the key yet", text)

    def test_the_gripper_pages_link_the_customer_runbook(self) -> None:
        for page in ("06-grippers.md", "04-robot-and-safety.md"):
            with self.subTest(page=page):
                self.assertIn("../runbooks/your_own_gripper.md", _text(_GUIDE / page))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
