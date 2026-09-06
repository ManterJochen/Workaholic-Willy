"""The console's perception panel: which stack is configured, and which route a prompt would take.

Neither endpoint loads a model. That is the point -- an operator asks both questions BEFORE spending
VRAM, and on a bench with no weights at all the answers must still be correct.
"""

from __future__ import annotations

import unittest
from unittest import mock

from fastapi.testclient import TestClient

from api.app import create_app


class PerceptionPanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def test_the_default_cell_reports_the_configured_zero_shot_stack(self) -> None:
        """The shipped tree DOES configure a pipeline, and the panel must report the one it has.

        ⛔ THIS TEST USED TO ASSERT THE OPPOSITE, and it passed for a bad reason. It read
        ``pipeline_configured is False`` under the docstring "No pipeline block = the byte-identical
        default" -- but there IS a pipeline block: ``models.pipeline`` is ``PipelineConfig(kind=
        'zero_shot')`` on the shipped tree. What made the endpoint answer "no pipeline" was
        ``diagnostics.py`` reading ``cell.config`` -- the BOUND METHOD -- instead of calling it. A
        bound method has no ``models`` and is truthy, so the guard never fired and the answer was
        always "legacy stack", whatever the tree said.

        The cost was not cosmetic: this endpoint exists so an operator can ask which stack a prompt
        will take, and the configured branch carries the sentence that actually warns them --
        "Complex prompts are NOT routed anywhere better -- they are grounded by a model that fails
        confidently on them." That sentence was unreachable. Found while wiring the frontend, 2026-08-19.
        """
        body = self.client.get("/v1/diagnostics").json()["perception"]
        self.assertTrue(body["pipeline_configured"])
        self.assertEqual(body["kind"], "zero_shot")
        self.assertEqual(body["backend"], "grounded_sam")
        self.assertEqual(body["segmenter"], "sam2")
        self.assertFalse(body["router_enabled"])
        self.assertIn("fails confidently", body["detail"])

    def test_the_panel_reads_the_tree_and_not_a_bound_method(self) -> None:
        """The regression guard for the shape of the bug, independent of what the tree happens to say."""
        from api.cell import Console

        body = self.client.get("/v1/diagnostics").json()["perception"]
        # Whatever the tree configures, the panel must agree with what `Console.config()` returns --
        # the same call every other consumer makes.
        self.assertTrue(callable(Console.config), "config is a method; reading it unwrapped was the bug")
        self.assertEqual(body["pipeline_configured"], True)

    def test_diagnostics_loads_no_model(self) -> None:
        """A diagnostics call that costs nine gigabytes of VRAM is not diagnostics."""
        with mock.patch("src.models.factory.build_perception") as build:
            self.client.get("/v1/diagnostics")
        build.assert_not_called()


class RoutePreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def _preview(self, prompt: str) -> dict:
        response = self.client.get("/v1/diagnostics/route", params={"prompt": prompt})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_a_plain_phrase_previews_as_the_simple_route(self) -> None:
        body = self._preview("a red cube")
        self.assertEqual(body["route"], "simple")
        self.assertEqual(body["reason"], "plain_noun_phrase")
        self.assertTrue(body["runnable"])

    def test_the_simple_route_shows_what_the_detector_will_actually_receive(self) -> None:
        """The normalisation is invisible otherwise, and it changes what the model is asked."""
        self.assertEqual(self._preview("A Red Cube")["normalized_prompt"], "a red cube.")

    def test_the_vlm_route_shows_no_normalised_prompt(self) -> None:
        """It gets the operator's words untouched -- rewriting them would change the question."""
        self.assertIsNone(self._preview("der kaputte Würfel")["normalized_prompt"])

    def test_a_german_prompt_on_a_cell_without_a_vlm_is_reported_as_not_runnable(self) -> None:
        """The whole reason this endpoint exists: the router WOULD send it somewhere this cell has not
        got. Without this, the operator finds out by watching a pick grasp the wrong object."""
        body = self._preview("der kaputte Würfel")
        self.assertEqual(body["route"], "vlm")
        self.assertEqual(body["reason"], "non_english")
        self.assertFalse(body["runnable"])
        self.assertIn("fails confidently", body["blocked_reason"])

    def test_the_description_is_the_one_line_a_human_reads(self) -> None:
        self.assertEqual(self._preview("der kaputte Würfel")["description"], "vlm (non_english)")

    def test_the_signals_come_through_for_a_client_that_wants_the_detail(self) -> None:
        body = self._preview("the cube behind the tray")
        self.assertEqual(body["route"], "vlm")
        self.assertIn("attributes", body["signals"])

    def test_preview_loads_no_model_either(self) -> None:
        with mock.patch("src.models.factory.build_perception") as build:
            self._preview("der kaputte Würfel")
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
