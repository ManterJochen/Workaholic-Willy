"""The console takes a push distance with a pick: ``POST /v1/pick`` with ``push_mm`` (owner, 2026-09-29).

On a connected dummy cell, 51 mm is refused because the hard cap is 50 mm and 9 mm because it cannot open the 10 mm a
finger needs: each a 422 ``push_distance_refused`` with the planner's sentence, and no run started. 40 mm is taken,
since the cell allows it (the shipped config declares no fixture, so the hard cap of 50 mm is its ceiling), and the
run's campaign carries it. The push itself, inside the pick and through the console's run body, is pinned end to end in
``tests/test_a_boxed_in_part_is_pushed_inside_the_pick.py``; this half needs the web framework's test client, which
only a ``test_api_*`` file may import (``tests/test_no_web_framework_import.py``).

A cell that arms the push and declares no fixture (owner, 2026-10-01: the push needs none) builds, connects and takes
the same answers: 30 mm when nobody asks, up to 50 mm as asked, above 50 or under 10 mm refused.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - the console is an optional extra
    TestClient = None  # type: ignore[assignment,misc]


class _OnAConnectedDummyCell:
    """A dummy cell built from the tree :meth:`_tree` writes, connected, and the console's test client on it."""

    def _tree(self, target: Path) -> None:
        from tests.test_api_pick import _dummy_tree

        _dummy_tree(target)

    def setUp(self) -> None:
        from api.app import create_app
        from api.cell import Console, set_console
        from src.config.loader import active_profile

        test: Any = self
        self.tmp = Path(tempfile.mkdtemp()) / "data"
        self._tree(self.tmp)
        self._previous_profile = active_profile()
        self.cell = Console(root=self.tmp, profile=None)
        self._previous_console = set_console(self.cell)
        self.client = TestClient(create_app())
        test.addCleanup(self._restore)
        built = self.client.post("/v1/cell/build", params={"rehearse": True})
        test.assertEqual(200, built.status_code, built.text)
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        test.assertEqual(200, self.client.post("/v1/cell/connect", json={"token": token}).status_code)

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

    def _refused(self, asked: float, said: str) -> None:
        test: Any = self
        response = self.client.post("/v1/pick", json={"picks": 1, "push_mm": asked})
        test.assertEqual(422, response.status_code, response.text)
        test.assertEqual("push_distance_refused", response.json()["code"])
        test.assertIn(said, response.json()["message"])
        test.assertIsNone(self.cell.registry.active())

    def _run(self, body: dict[str, Any]) -> None:
        """Start a run with ``body`` and wait until it is no longer running."""
        test: Any = self
        response = self.client.post("/v1/pick", json=body)
        test.assertEqual(202, response.status_code, response.text)
        run_id = response.json()["id"]
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and self.client.get(f"/v1/runs/{run_id}").json()["state"] == "running":
            time.sleep(0.02)
        test.assertNotEqual("running", self.client.get(f"/v1/runs/{run_id}").json()["state"])


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class TheConsoleTakesAPushDistanceTests(_OnAConnectedDummyCell, unittest.TestCase):
    """``POST /v1/pick`` with ``push_mm`` on a connected dummy cell: 51 and 9 refused with the sentence, 40 taken."""

    def test_51_and_9_are_refused_with_the_sentence_and_40_is_taken(self) -> None:
        for asked, said in ((51.0, "the hard cap is 50 mm"), (9.0, "cannot open the 10 mm")):
            with self.subTest(push_mm=asked):
                self._refused(asked, said)

        self._run({"picks": 1, "push_mm": 40.0})

        self.assertEqual(40.0, self.cell.session.service.campaign.distance_mm)


def _pushing_tree(target: Path) -> None:
    """The dummy tree with the owner's push armed (``dense_clutter``, ``nudge_target``) and no fixture declared."""
    from tests.test_api_pick import _dummy_tree

    _dummy_tree(target)
    robot = target / "robot" / "robot.yaml"
    text = robot.read_text(encoding="utf-8")
    text, hits = re.subn(
        r'^(\s*)default_mode:\s*"auto"$',
        r'\g<1>default_mode: "dense_clutter"\n\g<1>recovery:\n\g<1>  enabled: true\n'
        r'\g<1>  allowed_actions: [rescan, next_target, nudge_target]',
        text, count=1, flags=re.MULTILINE,
    )
    assert hits == 1, "the push substitution found nothing"
    robot.write_text(text, encoding="utf-8")


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class ACellThatPushesWithoutAFixtureTests(_OnAConnectedDummyCell, unittest.TestCase):
    """The owner's push config with no ``recovery.fixture``: it builds, a push is 30 mm unless asked, 50 mm at most."""

    def _tree(self, target: Path) -> None:
        _pushing_tree(target)

    def test_it_builds_with_the_push_armed_and_no_fixture(self) -> None:
        recovery = self.cell.session.service.effective_config.recovery_orchestrator
        self.assertIsNone(recovery.fixture)
        self.assertIn("nudge_target", recovery.allowed_actions)
        self.assertEqual(30.0, recovery.push_distance_mm)

    def test_51_and_9_are_refused_50_is_taken_and_30_when_nobody_asks(self) -> None:
        for asked, said in ((51.0, "the hard cap is 50 mm"), (9.0, "cannot open the 10 mm")):
            with self.subTest(push_mm=asked):
                self._refused(asked, said)

        self._run({"picks": 1, "push_mm": 50.0})
        self.assertEqual(50.0, self.cell.session.service.campaign.distance_mm)

        self._run({"picks": 1})
        self.assertEqual(30.0, self.cell.session.service.campaign.distance_mm)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
