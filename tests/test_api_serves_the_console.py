"""The backend serves the built console -- and behaves identically when it has not been built.

The console is an OPTIONAL face on a library. A headless deployment, a CI run and every existing test
have no ``api/static/`` directory, and mounting one must change nothing for them. That is the property
most of this file asserts, because it is the one that breaks silently: the mount is conditional on a
directory that is in ``.gitignore``, so the shipped tree and a developer's tree differ by exactly this,
and a test that only ever ran in one of the two states would prove nothing about the other.

The second property is the API/SPA boundary. A catch-all that returns ``index.html`` for an unknown
``/v1`` path is worse than a 404: the client asked for JSON, receives HTML, and reports a parse error
that sends whoever is on shift looking for a serialisation bug that does not exist.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import api.app as app_module
from api.app import create_app
from api.cell import Console, set_console


def _client(static: Path | None) -> TestClient:
    """A client whose app was built for ``static`` -- and keeps being built for it.

    The patch used to be a `with` around `create_app()` only, which is wrong in a way that HID a
    defect rather than finding one: `_mount_console` read the module global at REQUEST time, so by the
    time a test made a request the real `api/static/` was back. On a developer's box that directory
    exists, so the "with a built console" tests passed while serving the real build instead of the
    fixture -- and they failed the moment it was removed, which is exactly the CI state. The app now
    binds its root at mount time, and this patch stays up for the client's whole life.
    """
    set_console(Console(profile="console_dummy"))
    patcher = patch.object(app_module, "_STATIC", static or Path("does-not-exist"))
    patcher.start()
    try:
        return TestClient(create_app())
    finally:
        patcher.stop()


class WithoutABuiltConsoleTests(unittest.TestCase):
    """The shipped state. Everything must be exactly as it was before the mount existed."""

    def setUp(self) -> None:
        self.client = _client(None)

    def test_the_api_answers(self) -> None:
        self.assertEqual(self.client.get("/v1/health").status_code, 200)

    def test_an_unknown_path_is_a_json_404_not_an_html_page(self) -> None:
        response = self.client.get("/anything")
        self.assertEqual(response.status_code, 404)
        self.assertIn("application/json", response.headers["content-type"])

    def test_no_catch_all_route_was_registered(self) -> None:
        paths = {getattr(route, "path", "") for route in self.client.app.routes}  # type: ignore[attr-defined]
        self.assertNotIn("/{full_path:path}", paths)


class WithABuiltConsoleTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "assets").mkdir()
        (root / "index.html").write_text("<!doctype html><title>console</title>", encoding="utf-8")
        (root / "demo.html").write_text("<!doctype html><title>demo</title>", encoding="utf-8")
        (root / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
        self.root = root
        self.client = _client(root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_the_shell_is_served_at_the_root(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("console", response.text)

    def test_a_deep_link_survives_a_reload(self) -> None:
        # The whole reason for a catch-all: an operator bookmarked /pick and pressed F5.
        response = self.client.get("/pick")
        self.assertEqual(response.status_code, 200)
        self.assertIn("console", response.text)

    def test_the_demo_page_is_its_own_page(self) -> None:
        response = self.client.get("/demo.html")
        self.assertEqual(response.status_code, 200)
        self.assertIn("demo", response.text)

    def test_assets_are_served(self) -> None:
        self.assertEqual(self.client.get("/assets/app.js").status_code, 200)

    def test_the_api_still_wins(self) -> None:
        response = self.client.get("/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_an_unknown_api_path_stays_json(self) -> None:
        # The boundary. HTML here would be reported as a JSON parse error by every client.
        response = self.client.get("/v1/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertIn("application/json", response.headers["content-type"])
        self.assertEqual(response.json()["code"], "http_404")

    def test_a_path_that_escapes_the_bundle_gets_the_shell_not_the_file(self) -> None:
        # A console that serves files from outside its own directory is a file server for the box.
        response = self.client.get("/../../pyproject.toml")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("[project]", response.text)


class TheErrorEnvelopeIsUniversalTests(unittest.TestCase):
    """One shape for every failure -- which was NOT true for the router's own 404 until 2026-08-19."""

    def setUp(self) -> None:
        self.client = _client(None)

    def test_a_router_404_uses_the_envelope(self) -> None:
        body = self.client.get("/v1/nope").json()
        self.assertEqual(set(body), {"code", "message", "detail"})

    def test_a_validation_failure_uses_the_envelope(self) -> None:
        body = self.client.post("/v1/pick", json={"picks": "not a number"}).json()
        self.assertEqual(set(body), {"code", "message", "detail"})
        self.assertEqual(body["code"], "bad_request")

    def test_the_envelope_is_in_the_openapi_document(self) -> None:
        # It was defined and referenced by nothing, so a generated client could type every success and
        # not one failure -- the one shape it must handle on every single call.
        spec = self.client.app.openapi()  # type: ignore[attr-defined]
        self.assertIn("ErrorOut", spec["components"]["schemas"])
        self.assertIn("default", spec["paths"]["/v1/pick"]["post"]["responses"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
