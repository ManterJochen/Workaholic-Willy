"""The console's perception panel: which stack is configured, and whether it can run on this box.

It loads no model. That is the point -- an operator asks BEFORE spending VRAM, and on a bench with no weights at all
the answer must still be correct. The route preview (``GET /v1/diagnostics/route``) left with the prompt router and
GroundingDINO on 2026-10-17: the VLM grounds every prompt, so there is no route to preview.
"""

from __future__ import annotations

import unittest
from unittest import mock

from fastapi.testclient import TestClient

from api.app import create_app


class PerceptionPanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def _panel(self, *, present: bool) -> dict:
        with mock.patch("api.routers.diagnostics._vlm_weights_present", return_value=present):
            return self.client.get("/v1/diagnostics").json()["perception"]

    def test_the_default_cell_reports_the_vlm_and_sam2(self) -> None:
        """The shipped tree's stack, the owner's: the VLM grounds, SAM2 cuts.

        ⛔ The panel once read ``cell.config`` -- the BOUND METHOD -- instead of calling it. A bound method has no
        ``models`` and is truthy, so the guard never fired and the answer was always one stack, whatever the tree
        said. Found while wiring the frontend, 2026-08-19.
        """
        body = self._panel(present=True)
        self.assertEqual(("zero_shot", "vlm", "sam2"), (body["kind"], body["backend"], body["segmenter"]))
        self.assertEqual("Qwen/Qwen3-VL-4B-Instruct", body["vlm_model_id"])
        self.assertTrue(body["vlm_weights_present"])
        self.assertIn("VLM stack ready", body["detail"])
        for gone in ("pipeline_configured", "router_enabled", "vlm_on_unavailable"):
            self.assertNotIn(gone, body)

    def test_a_vlm_without_its_weights_says_every_pick_is_refused_and_the_fetch(self) -> None:
        body = self._panel(present=False)
        self.assertFalse(body["vlm_weights_present"])
        self.assertIn("every pick is refused", body["detail"])
        self.assertIn("python scripts/model_weights/fetch.py vlm-4b", body["detail"])

    def test_the_panel_reads_the_tree_and_not_a_bound_method(self) -> None:
        """The regression guard for the shape of the bug, independent of what the tree happens to say."""
        from api.cell import Console

        self.assertTrue(callable(Console.config), "config is a method; reading it unwrapped was the bug")
        self.assertEqual("zero_shot", self._panel(present=True)["kind"])

    def test_diagnostics_loads_no_model(self) -> None:
        """A diagnostics call that costs nine gigabytes of VRAM is not diagnostics."""
        with mock.patch("src.models.factory.build_perception") as build:
            self.client.get("/v1/diagnostics")
        build.assert_not_called()

    def test_there_is_no_route_to_preview(self) -> None:
        self.assertEqual(404, self.client.get("/v1/diagnostics/route", params={"prompt": "a red cube"}).status_code)


if __name__ == "__main__":
    unittest.main()
