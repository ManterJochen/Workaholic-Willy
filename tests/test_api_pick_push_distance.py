"""The console takes a push distance with a pick: ``POST /v1/pick`` with ``push_mm`` (owner, 2026-09-29).

On a connected dummy cell, 51 mm is refused because the hard cap is 50 mm and 9 mm because it cannot open the 10 mm a
finger needs: each a 422 ``push_distance_refused`` with the planner's sentence, and no run started. 40 mm is taken,
since the cell allows it (the shipped config declares no fixture, so the hard cap of 50 mm is its ceiling), and the
run's campaign carries it. The push itself, inside the pick and through the console's run body, is pinned end to end in
``tests/test_a_boxed_in_part_is_pushed_inside_the_pick.py``; this half needs the web framework's test client, which
only a ``test_api_*`` file may import (``tests/test_no_web_framework_import.py``).
"""

from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - the console is an optional extra
    TestClient = None  # type: ignore[assignment,misc]


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class TheConsoleTakesAPushDistanceTests(unittest.TestCase):
    """``POST /v1/pick`` with ``push_mm`` on a connected dummy cell: 51 and 9 refused with the sentence, 40 taken."""

    def setUp(self) -> None:
        from api.app import create_app
        from api.cell import Console, set_console
        from src.config.loader import active_profile
        from tests.test_api_pick import _dummy_tree

        self.tmp = Path(tempfile.mkdtemp()) / "data"
        _dummy_tree(self.tmp)
        self._previous_profile = active_profile()
        self.cell = Console(root=self.tmp, profile=None)
        self._previous_console = set_console(self.cell)
        self.client = TestClient(create_app())
        self.addCleanup(self._restore)
        self.assertEqual(200, self.client.post("/v1/cell/build", params={"rehearse": True}).status_code)
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        self.assertEqual(200, self.client.post("/v1/cell/connect", json={"token": token}).status_code)

    def _restore(self) -> None:
        from api.cell import set_console
        from src.config.loader import reload_config, set_active_profile

        try:
            self.cell.session.disconnect()
        except Exception:  # pragma: no cover
            pass
        set_console(self._previous_console)
        set_active_profile(self._previous_profile)
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    def test_51_and_9_are_refused_with_the_sentence_and_40_is_taken(self) -> None:
        for asked, said in ((51.0, "the hard cap is 50 mm"), (9.0, "cannot open the 10 mm")):
            with self.subTest(push_mm=asked):
                response = self.client.post("/v1/pick", json={"picks": 1, "push_mm": asked})
                self.assertEqual(422, response.status_code, response.text)
                self.assertEqual("push_distance_refused", response.json()["code"])
                self.assertIn(said, response.json()["message"])
                self.assertIsNone(self.cell.registry.active())

        response = self.client.post("/v1/pick", json={"picks": 1, "push_mm": 40.0})

        self.assertEqual(202, response.status_code, response.text)
        run_id = response.json()["id"]
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and self.client.get(f"/v1/runs/{run_id}").json()["state"] == "running":
            time.sleep(0.02)
        self.assertNotEqual("running", self.client.get(f"/v1/runs/{run_id}").json()["state"])
        self.assertEqual(40.0, self.cell.session.service.campaign.distance_mm)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
