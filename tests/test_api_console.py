"""The operator console's HTTP surface (stage 1: preflight and preparing a cell).

Every test that can write runs against a COPY of the shipped config tree. The console's whole purpose is
to change files an operator depends on, so a test suite that pointed it at the real tree would be one
bug away from editing the repository.

The load-bearing test here is
``test_the_browser_preflight_is_the_same_verdict_the_cli_prints``. An operator uses whichever of the two
is in front of them and must not have to wonder which one is right; the guarantee is not "they were
written to agree" but "there is one implementation", and this asserts it row for row.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - the console is an optional extra
    TestClient = None  # type: ignore[assignment,misc]

from src.config.loader import active_profile, reload_config, set_active_profile

_SHIPPED = Path(__file__).resolve().parents[1] / "config"


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class ConsoleApiTests(unittest.TestCase):
    """A fresh app and a scratch config tree per test."""

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

    # -- health ------------------------------------------------------------------------------------

    def test_health_says_nothing_about_the_cell(self) -> None:
        """Merging "the server runs" with "the robot is ready" makes a green light mean neither."""
        body = self.client.get("/v1/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertNotIn("robot", body)
        self.assertNotIn("preflight", body)

    # -- preflight ---------------------------------------------------------------------------------

    def test_the_browser_preflight_is_the_same_verdict_the_cli_prints(self) -> None:
        from src.robot.execution.real_cell.preflight import run_config_preflight

        for profile in (None, "ur3e"):
            with self.subTest(profile=profile):
                self.cell.profile = profile
                expected = run_config_preflight(self.cell.config().robot)
                body = self.client.get("/v1/preflight").json()

                self.assertEqual(
                    [(c.name, str(c.status), c.detail, c.fix) for c in expected.checks],
                    [(c["name"], c["status"], c["detail"], c["fix"]) for c in body["checks"]],
                )
                self.assertEqual(body["ok"], expected.ok)
                self.assertEqual(body["n_blocking"], len(expected.blocking))
                self.assertEqual(body["profile"], profile)

    def test_bench_rows_are_counted_apart_from_warnings(self) -> None:
        """A `bench` row is not a milder warning -- no software can answer it, so it never clears."""
        body = self.client.get("/v1/preflight").json()
        self.assertGreater(body["n_bench"], 0)
        self.assertTrue(all(c["fix"] for c in body["checks"] if c["status"] != "ok"))

    # -- reading the config ------------------------------------------------------------------------

    def test_a_value_arrives_with_the_layer_that_set_it(self) -> None:
        self.cell.profile = "ur3e"
        body = self.client.get(
            "/v1/config/explain", params={"key": "robot.safety.payload.max_mass_kg"}
        ).json()

        self.assertEqual(body["key"], "robot.safety.payload.max_mass_kg")
        self.assertIn("robot.ur3e.yaml", body["source"])
        # Two layers write this key; the UI must be able to show the one that lost, too.
        self.assertEqual(len(body["layers"]), 2)
        self.assertTrue(body["layers"][-1]["winner"])
        self.assertFalse(body["layers"][0]["winner"])
        self.assertTrue(body["writable"] is False)

    def test_the_rendered_text_is_the_cli_rendering_verbatim(self) -> None:
        """Carried so a UI can show what the terminal shows instead of typesetting a second version.

        ⛔⛔ **THIS GUARD WAS GREEN WHILE THE TWO DISAGREED, AND THE KEY IS WHY.**
        ``robot.safety.payload.mass_kg`` is a float, so it exercises the one case where the two
        paths could not differ. MEASURED on the default tree: 45 of 468 keys hold ``None``, and for
        every one of them the console rendered ``key = None`` while the terminal rendered ``key``,
        because the CLI's own walker could not tell "absent" from "the value is None" and let
        ``has_value`` default to ``value is not None``.

        The second key below is one of those 45. It is the case this guard existed for and could
        not see.
        """
        from src.config.explain import explain_in

        for key in ("robot.safety.payload.mass_kg", "camera.cameras.active_rig_id"):
            with self.subTest(key):
                body = self.client.get("/v1/config/explain", params={"key": key}).json()
                expected = explain_in(
                    self.cell.config(), key, self.cell.root, self.cell.layers
                ).render()
                self.assertEqual(body["text"], expected)

    def test_a_key_whose_value_is_none_reads_as_set_not_as_absent(self) -> None:
        """⛔ THE DISTINCTION THE DRIFT DESTROYED. An unconfigured ``serial_number`` IS ``null``;
        reporting that as "nobody set this" tells an operator the opposite of what the tree says."""
        body = self.client.get(
            "/v1/config/explain", params={"key": "camera.cameras.active_rig_id"}
        ).json()
        self.assertIn("= None", body["text"])

    def test_a_typo_is_answered_with_the_schema_s_near_misses(self) -> None:
        response = self.client.get(
            "/v1/config/explain", params={"key": "robot.safety.payload.mass_kilograms"}
        )
        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertEqual(body["code"], "unknown_key")
        self.assertIn("robot.safety.payload.mass_kg", body["detail"]["suggestions"])

    def test_a_list_element_still_reads_back_although_the_schema_index_cannot_see_it(self) -> None:
        """``explain`` treats a list as a leaf; the loaded tree settles what the value actually is."""
        self.cell.profile = "tiltcam"
        response = self.client.get(
            "/v1/config/explain", params={"key": "camera.cameras.rigs[0].serial_number"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()["value"])
        self.assertTrue(response.json()["writable"])

    # -- writing measurements ----------------------------------------------------------------------

    def test_the_form_describes_itself_from_the_library(self) -> None:
        body = self.client.get("/v1/config/writable").json()
        keys = {entry["key"] for entry in body}
        self.assertIn("robot.safety.payload.mass_kg", keys)
        self.assertNotIn("robot.workspace_limits.z_min", keys)
        for entry in body:
            self.assertGreater(len(entry["measure"]), 40, f"{entry['key']} has no real instruction")

    def test_a_measured_tool_frame_clears_its_blocking_row(self) -> None:
        """The end-to-end claim of stage 1: bench work done in a browser moves the checklist."""
        before = self.client.get("/v1/preflight").json()
        self.assertIn("tool frame", [c["name"] for c in before["checks"] if c["status"] == "block"])

        response = self.client.patch(
            "/v1/config",
            json={
                "robot.gripper.tool_frame.source": "willy",
                "robot.gripper.tool_frame.offset_mm": [0.0, 132.0, 0.0],
                "robot.gripper.tool_frame.rotation_quat_xyzw": [-0.70710678, 0.0, 0.0, 0.70710678],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["applied"])
        self.assertEqual(body["values"]["robot.gripper.tool_frame.source"], "willy")
        # The response carries the re-rendered checklist, so the UI does not have to refetch to see
        # the row it just cleared.
        blocking = [c["name"] for c in body["preflight"]["checks"] if c["status"] == "block"]
        self.assertNotIn("tool frame", blocking)

    def test_a_safety_limit_is_refused_with_403_and_a_reason(self) -> None:
        response = self.client.patch("/v1/config", json={"robot.workspace_limits.z_min": -500.0})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["code"], "not_writable")
        self.assertIn("comment", response.json()["message"])
        # And nothing moved: the limit still reads as whatever the YAML says, not as the attempt.
        still = self.client.get(
            "/v1/config/explain", params={"key": "robot.workspace_limits.z_min"}
        ).json()["value"]
        self.assertEqual(still, self.cell.config().robot.workspace_limits.z_min)
        self.assertNotEqual(still, -500.0)

    def test_a_value_the_validators_reject_is_422_and_leaves_the_file_alone(self) -> None:
        target = self.tmp / "robot" / "robot.yaml"
        before = target.read_bytes()

        response = self.client.patch("/v1/config", json={"robot.safety.payload.mass_kg": 999.0})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "invalid_value")
        self.assertEqual(target.read_bytes(), before)

    def test_no_write_happens_while_a_run_owns_the_cell(self) -> None:
        """The rule the write endpoint is allowed to exist because of, asserted before runs exist."""
        self.cell.active_run_id = "run-2026-08-09-0001"
        target = self.tmp / "robot" / "robot.yaml"
        before = target.read_bytes()

        response = self.client.patch("/v1/config", json={"robot.safety.payload.mass_kg": 1.15})
        self.assertEqual(response.status_code, 409)
        body = response.json()
        self.assertEqual(body["code"], "run_active")
        self.assertEqual(body["detail"]["run_id"], "run-2026-08-09-0001")
        # Refused, not queued: nothing may land after the run ends either.
        self.assertEqual(target.read_bytes(), before)
        self.cell.active_run_id = None
        self.assertEqual(target.read_bytes(), before)

    # -- what the console must NOT offer -------------------------------------------------------------

    def test_there_is_no_emergency_stop_endpoint(self) -> None:
        """A stop that travels over a socket is an illusion, and offering one invites relying on it.

        Asserted rather than merely documented, because "add an emergency stop" is the single most
        obvious thing a future contributor would think this console is missing.

        The one endpoint that IS called ``stop`` is ``/v1/pick/stop``, and it is a different thing: it
        sets a flag the pick loop reads *between attempts*, so a motion already in flight completes.
        The test below pins that distinction in the endpoint's own documentation, because a route named
        "stop" that quietly grew emergency semantics would pass a name check while being the exact
        thing this console must not have.
        """
        paths = {route.path for route in self.client.app.routes}  # type: ignore[attr-defined]
        offenders = [
            p for p in paths
            if any(word in p.lower() for word in ("estop", "e_stop", "emergency", "halt", "abort"))
        ]
        self.assertEqual(offenders, [], f"the console must not expose an emergency stop: {offenders}")

    def test_the_run_stop_endpoint_says_it_is_not_an_emergency_stop(self) -> None:
        """The distinction has to survive in the API's own description, not only in a plan document.

        A browser stop cannot be an emergency stop -- latency, a closed tab, a sleeping laptop -- so the
        endpoint that exists must say what it actually does, where the next person reads it.
        """
        schema = self.client.get("/openapi.json").json()
        stop = schema["paths"]["/v1/pick/stop"]["post"]
        description = (stop.get("description", "") + stop.get("summary", "")).lower()

        self.assertIn("before it begins the next attempt", description)
        self.assertIn("in flight", description)
        # And the top-level API description still names the physical button as the real one.
        self.assertIn("mushroom", schema["info"]["description"].lower())

    def test_no_endpoint_writes_a_motion_parameter(self) -> None:
        """The console issues tasks, not parameters. The writable set is the whole permitted surface."""
        from src.config.edit import WRITABLE

        forbidden = ("motion", "speed", "accel", "workspace_limits", "self_collision")
        for entry in WRITABLE:
            self.assertFalse(
                any(word in entry.path for word in forbidden),
                f"{entry.path} is a motion parameter and must not be writable from a browser",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
