"""The console's abilities are the rows ``real_cell --check`` prints after its checklist (P0i i4, owner decision 7).

``GET /v1/abilities`` beside ``GET /v1/preflight``: what each operator function can do on the console's cell, asked of
the arm its tree builds, never connected. One implementation serves both, so the browser and the terminal cannot tell an
operator two things; this asserts they agree row for row, on a UR tree and on a desk tree.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - the optional extra
    TestClient = None  # type: ignore[assignment,misc]

from src.config.loader import active_profile, reload_config, set_active_profile

_SHIPPED = Path(__file__).resolve().parents[1] / "config"


@unittest.skipIf(TestClient is None, "requirements/api.txt is not installed (optional extra)")
class TheBrowsersAbilitiesAreTheChecksTests(unittest.TestCase):
    def setUp(self) -> None:
        from api.app import create_app
        from api.cell import Console, set_console

        self.tmp = Path(tempfile.mkdtemp()) / "data"
        shutil.copytree(_SHIPPED, self.tmp)
        self._previous_profile = active_profile()
        self.cell = Console(root=self.tmp, profile=None)
        self._previous_console = set_console(self.cell)
        self.client = TestClient(create_app())
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        from api.cell import set_console

        set_console(self._previous_console)
        set_active_profile(self._previous_profile)
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    def test_the_browser_rows_are_the_rows_the_check_prints(self) -> None:
        from src.robot.execution.abilities import desk_abilities

        for profile in ("ursim,ursim_curobo", "console_dummy", None):
            with self.subTest(profile=profile):
                self.cell.profile = profile
                config = self.cell.config()
                assert config.robot is not None
                expected = desk_abilities(config.robot, camera=config.camera, data_dir=self.tmp).to_dict()
                body = self.client.get("/v1/abilities").json()
                self.assertEqual(body["abilities"], expected["abilities"])
                self.assertEqual((body["n_available"], body["n_refused"], body["n_bench"]),
                                 (expected["n_available"], expected["n_refused"], expected["n_bench"]))
                self.assertEqual(body["profile"], profile)

    def test_a_refused_function_is_not_a_blocking_checklist_row(self) -> None:
        """The abilities stand beside the checklist and never enter it: a desk tree refuses teaching and checks green."""
        self.cell.profile = "console_dummy"
        abilities = self.client.get("/v1/abilities").json()
        preflight = self.client.get("/v1/preflight").json()
        self.assertGreater(abilities["n_refused"], 0)
        self.assertTrue(preflight["ok"])
        self.assertNotIn("teach", [check["name"] for check in preflight["checks"]])


if __name__ == "__main__":
    unittest.main()
