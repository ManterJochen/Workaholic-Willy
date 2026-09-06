"""POST /v1/pick refuses a prompt this cell cannot ground correctly -- before anything moves.

The failure being guarded is silent: a complex prompt handed to the phrase grounder does not come back
empty, it comes back with a confident box on the WRONG object, and the log then reads like a successful
pick. Refusing at the API boundary is free; discovering it after the arm has moved is not.
"""

from __future__ import annotations

import unittest
from unittest import mock

from fastapi.testclient import TestClient

from api.app import create_app
from api.routers import pick as pick_router


class _Stack:
    """Stand-in for the diagnostics perception reading."""

    def __init__(self, backend="grounded_sam", present=None, on_unavailable=None, model_id="qwen"):
        self.backend, self.vlm_weights_present = backend, present
        self.vlm_on_unavailable, self.vlm_model_id = on_unavailable, model_id


class RouteGuardTests(unittest.TestCase):
    """The guard in isolation -- it runs before any cell state, so it is tested directly."""

    def _guard(self, prompt: str, stack: _Stack) -> None:
        with mock.patch.object(pick_router, "_perception", return_value=stack, create=True):
            with mock.patch("api.routers.diagnostics._perception", return_value=stack):
                pick_router._refuse_unroutable_prompt(mock.Mock(), prompt)

    def test_a_simple_prompt_passes_on_a_phrase_only_cell(self) -> None:
        self._guard("a red cube", _Stack())

    def test_an_empty_prompt_passes(self) -> None:
        """'whatever the source already targets' -- there is nothing to route."""
        self._guard("   ", _Stack())

    def test_a_complex_prompt_is_refused_on_a_phrase_only_cell(self) -> None:
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as caught:
            self._guard("der kaputte Würfel", _Stack())
        self.assertEqual(caught.exception.status_code, 422)
        detail = caught.exception.detail
        self.assertEqual(detail["code"], "prompt_not_routable")
        self.assertIn("WRONG object", detail["message"])
        self.assertEqual(detail["detail"]["reason"], "non_english")

    def test_a_complex_prompt_passes_when_the_vlm_is_present(self) -> None:
        self._guard("der kaputte Würfel", _Stack(backend="vlm", present=True))

    def test_a_complex_prompt_is_refused_when_the_vlm_weights_are_missing(self) -> None:
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as caught:
            self._guard("der kaputte Würfel", _Stack(backend="vlm", present=False, on_unavailable="refuse"))
        self.assertIn("not on this box", caught.exception.detail["message"])

    def test_degrade_is_a_deliberate_choice_and_is_not_second_guessed(self) -> None:
        """The operator configured a loudly-logged fallback. Refusing anyway would override them."""
        self._guard("der kaputte Würfel", _Stack(backend="vlm", present=False, on_unavailable="degrade"))


class PickEndpointTests(unittest.TestCase):
    def test_the_guard_runs_before_the_run_is_started(self) -> None:
        """Order matters: a refused prompt must not leave a run in the registry."""
        client = TestClient(create_app())
        response = client.post("/v1/pick", json={"prompt": "der kaputte Würfel", "picks": 1})
        # Not connected wins (409) or the route guard fires (422) -- either way nothing was started.
        self.assertIn(response.status_code, (409, 422))
        self.assertEqual(client.get("/v1/runs").json(), [])


if __name__ == "__main__":
    unittest.main()
