"""The console's contract (build plan section 1), pinned before anything is built on it.

The frontend and every API track of commit 2 build against one contract: the routes of the endpoint table with their
methods and wire types, the typed codes of ``GET /v1/codes``, the event log the reducer replays, and the shared seams
both API tracks call (``RunRegistry.start_kind``, the console's recovery record, its stamps, the cell's own stream).
This file is what keeps that contract one thing:

* **every contract route exists** with its method, its request model and its response model, as the OpenAPI document
  the frontend's types are generated from says it;
* **no route path contains halt, abort, estop, e_stop or emergency**: "halt now" is ``POST /v1/cell/brake``, and its
  own description says it is not an emergency stop, the red button is;
* **``/v1/codes`` equals the StrEnums**, each Literal equals its StrEnum, the frontend's ``codes.ts`` lists the same
  values, and every code the API package can answer is in the catalog;
* **one code, one meaning**: ``not_acknowledged`` is Connect's 428 and nothing else; the stop card's gate is
  ``cell_not_cleared``;
* **a run started with ``start_kind`` that ends on a problem code writes the recovery record**, before it lets go of
  the run lock, and the record survives a rebuild (``adopt``);
* **what the gates read fails closed**: a stamp is set only through its method, a hand or an arm whose read raises is
  said and never raised, a carried part nobody can read keeps "no part held" shut, and a build whose jaws question
  cannot be installed leaves the cell unbuilt;
* **a route not built yet moves nothing**, and the library symbols the API reads keep the shape it reads, once their
  tracks have landed;
* the hand-written event logs the frontend replays speak only this contract;
* **the endpoint table of ``api/README.md`` is the app's routes**: every route the table names exists, every ``/v1``
  route the app serves is in the table, and the command reader's vocabularies are the catalog's.

Honesty bucket (2): real objects, a real rehearsal build on a dummy tree, real threads; no controller anywhere.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import re
import shutil
import tempfile
import threading
import time
import types
import typing
import unittest
from pathlib import Path
from typing import Any

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - the console is an optional extra
    TestClient = None  # type: ignore[assignment,misc]

from src.config.loader import active_profile, reload_config, set_active_profile

_ROOT = Path(__file__).resolve().parents[1]
_CODES_TS = _ROOT / "frontend" / "src" / "api" / "codes.ts"
_FIXTURES = _ROOT / "frontend" / "src" / "test" / "fixtures"
_API_README = _ROOT / "api" / "README.md"

#: The endpoint table of build plan 1.11, plus the two pick routes it keeps: (method, path, request model or None,
#: response model or media type, success status, query parameters the route must take).
_CONTRACT: tuple[tuple[str, str, str | None, str, int, tuple[str, ...]], ...] = (
    ("GET", "/v1/codes", None, "CodesOut", 200, ()),
    ("POST", "/v1/task", "TaskIn", "RunOut", 202, ()),
    ("POST", "/v1/task/stop", None, "RunOut", 200, ("run_id",)),
    ("POST", "/v1/task/restart", "RestartIn", "RunOut", 202, ()),
    ("GET", "/v1/cell", None, "CellOut", 200, ()),
    ("GET", "/v1/cell/facts", None, "CellFactsOut", 200, ()),
    ("GET", "/v1/cell/readiness", None, "ReadinessOut", 200, ()),
    ("POST", "/v1/cell/acknowledge", "AcknowledgeIn", "CellOut", 200, ()),
    ("POST", "/v1/cell/brake", None, "BrakeOut", 200, ()),
    ("POST", "/v1/cell/home", "HomeIn", "RunOut", 202, ()),
    ("POST", "/v1/cell/planner", None, "RunOut", 202, ()),
    ("POST", "/v1/cell/connect", "ConnectIn", "CellOut", 200, ()),
    ("POST", "/v1/cell/disconnect", None, "CellOut", 200, ()),
    ("GET", "/v1/cell/jaws", None, "JawsOut", 200, ()),
    ("POST", "/v1/cell/jaws/answer", "JawAnswerIn", "JawsOut", 200, ()),
    ("POST", "/v1/cell/jaws/check", None, "JawsOut", 200, ()),
    ("GET", "/v1/camera/live", None, "LiveFrameOut", 200, ("rig", "max_width")),
    ("GET", "/v1/runs/{run_id}/overlays/{n}", None, "image/png", 200, ()),
    ("GET", "/v1/runs/{run_id}/target/overlay", None, "image/png", 200, ()),
    ("GET", "/v1/poses", None, "PosesOut", 200, ()),
    ("PUT", "/v1/poses/default-place", "DefaultPlaceIn", "PosesOut", 200, ()),
    ("GET", "/v1/teach/payload", None, "PayloadOut", 200, ()),
    ("POST", "/v1/teach", "TeachIn", "TeachStartOut", 202, ()),
    ("GET", "/v1/teach/{run_id}", None, "TeachStateOut", 200, ("token",)),
    ("POST", "/v1/teach/{run_id}/capture", None, "TeachStateOut", 200, ("token",)),
    ("POST", "/v1/teach/{run_id}/cancel", None, "TeachStateOut", 200, ("token",)),
    ("POST", "/v1/commands/parse", "CommandIn", "CommandOut", 200, ()),
    ("GET", "/v1/commands/status", None, "CommandStatusOut", 200, ()),
    ("POST", "/v1/commands/warmup", None, "CommandStatusOut", 200, ()),
    ("POST", "/v1/pick", "PickIn", "RunOut", 202, ()),
    ("POST", "/v1/pick/stop", None, "RunOut", 200, ("run_id",)),
)

#: The catalogs of ``CodesOut``: the field, its StrEnum's name in ``api.codes``, and the Literal beside it.
_CATALOGS: tuple[tuple[str, str, str], ...] = (
    ("run_kinds", "RunKind", "RunKindName"),
    ("stop_codes", "StopCode", "StopCodeName"),
    ("stop_classes", "StopClass", "StopClassName"),
    ("event_types", "EventType", "EventTypeName"),
    ("refusal_codes", "RefusalCode", "RefusalCodeName"),
    ("light_ids", "LightId", "LightIdName"),
    ("light_states", "LightState", "LightStateName"),
    ("light_codes", "LightCode", "LightCodeName"),
    ("blocker_codes", "BlockerCode", "BlockerCodeName"),
    ("jaws_stages", "JawsStage", "JawsStageName"),
    ("jaws_choices", "JawsChoice", "JawsChoiceName"),
    ("command_notes", "CommandNote", "CommandNoteName"),
)

#: The words no route path may contain: a stop that travels over a socket is not an emergency stop.
_FORBIDDEN_IN_PATHS = ("estop", "e_stop", "emergency", "halt", "abort")


def _dummy_tree(target: Path) -> None:
    """The shipped tree re-pointed at a dummy arm and a dummy hand: the one cell that builds with no hardware."""
    shutil.copytree(_ROOT / "config", target)
    robot = target / "robot" / "robot.yaml"
    text = robot.read_text(encoding="utf-8")
    text, arm_hits = re.subn(r'^(\s*)vendor:\s*"ur"$', r'\g<1>vendor: "dummy"', text, count=1, flags=re.MULTILINE)
    text, hand_hits = re.subn(
        r'^(\s*)vendor:\s*"robotiq"$', r'\g<1>vendor: "dummy"', text, count=1, flags=re.MULTILINE
    )
    assert arm_hits == 1 and hand_hits == 1, "the dummy substitution found nothing"
    robot.write_text(text, encoding="utf-8")


def _wait_for_event(hub: Any, stream: str, event_type: str, *, timeout: float = 10.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event in hub.since(stream, 0)[0]:
            if event.type == event_type:
                return event
        time.sleep(0.005)
    raise AssertionError(f"no {event_type} on {stream} within {timeout} s")


@dataclasses.dataclass(frozen=True)
class _JawAsking:
    """A jaws question shaped as the hand asks it (library L10's ``JawAsking``), for the browser seam."""

    stage: str = "where"
    at_connect: bool = True
    reason: str = "every change of its output moves them and nothing reads them back"
    where: str = "tool output 0"
    choices: tuple[str, ...] = ("open", "closed")
    text: str = "Jaws on tool output 0: Do they stand OPEN?"
    attempt: int = 1
    why_again: str = ""


def _record_calls(target: Any, label: str, calls: list[str]) -> None:
    """Replace every method of ``target`` with one that records its name before it runs: what a route commanded."""
    for name in dir(type(target)):
        if name.startswith("__") or isinstance(inspect.getattr_static(type(target), name, None), property):
            continue
        bound = getattr(target, name, None)
        if not callable(bound):
            continue

        def recorded(*args: Any, _name: str = name, _bound: Any = bound, **kwargs: Any) -> Any:
            calls.append(f"{label}.{_name}")
            return _bound(*args, **kwargs)

        try:
            setattr(target, name, recorded)
        except (AttributeError, TypeError):  # pragma: no cover - a slot or a read-only attribute keeps its method
            pass


def _stub_routes(app: Any) -> set[tuple[str, str]]:
    """Every (method, path) whose handler is still the contract stage's stub: it raises ``not_built_yet``."""
    found: set[tuple[str, str]] = set()
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        try:
            source = inspect.getsource(endpoint) if endpoint is not None else ""
        except (OSError, TypeError):
            continue
        if "not_built_yet(" in source:
            found |= {(method, route.path) for method in route.methods}
    return found


class _ScratchConsole(unittest.TestCase):
    """A fresh console on a scratch copy of the shipped tree, installed as the process console for the app."""

    def setUp(self) -> None:
        from api.cell import Console, set_console

        self.tmp = Path(tempfile.mkdtemp()) / "data"
        _dummy_tree(self.tmp)
        self._previous_profile = active_profile()
        self.cell = Console(root=self.tmp, profile=None)
        self.cell.record_log_path = self.tmp.parent / "grasp_records.jsonl"
        self._previous_console = set_console(self.cell)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        from api.cell import set_console

        try:
            self.cell.session.release()
        except Exception:  # pragma: no cover - cleanup must not mask a failure
            pass
        set_console(self._previous_console)
        set_active_profile(self._previous_profile)
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    def _client(self) -> Any:
        from api.app import create_app

        return TestClient(create_app())


# ---------------------------------------------------------------------------------------------------------------------
# The routes
# ---------------------------------------------------------------------------------------------------------------------


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class EveryContractRouteExistsTests(unittest.TestCase):
    """Each route of the endpoint table, with its method and its wire types, in the OpenAPI document."""

    @classmethod
    def setUpClass(cls) -> None:
        from api.app import create_app

        cls.spec = create_app().openapi()

    @staticmethod
    def _ref_name(schema: dict[str, Any]) -> str:
        ref = schema.get("$ref", "")
        return ref.rsplit("/", 1)[-1]

    def test_every_route_has_its_method_request_and_response(self) -> None:
        paths = self.spec["paths"]
        for method, path, request, response, status, query in _CONTRACT:
            with self.subTest(route=f"{method} {path}"):
                self.assertIn(path, paths, f"{path} is not a route")
                operation = paths[path].get(method.lower())
                self.assertIsNotNone(operation, f"{path} does not answer {method}")
                body = operation.get("requestBody")
                if request is None:
                    self.assertIsNone(body, f"{method} {path} takes no body in the contract")
                else:
                    self.assertIsNotNone(body, f"{method} {path} takes {request}")
                    schema = body["content"]["application/json"]["schema"]
                    self.assertEqual(request, self._ref_name(schema))
                answers = operation["responses"].get(str(status))
                self.assertIsNotNone(answers, f"{method} {path} does not answer {status}")
                content = answers.get("content", {})
                if response.startswith("image/"):
                    self.assertIn(response, content, f"{method} {path} answers no {response}")
                else:
                    self.assertEqual(response, self._ref_name(content["application/json"]["schema"]))
                said = {p["name"] for p in operation.get("parameters", ()) if p.get("in") == "query"}
                for name in query:
                    self.assertIn(name, said, f"{method} {path} takes no ?{name}=")

    def test_every_route_documents_the_one_failure_envelope(self) -> None:
        for method, path, *_ in _CONTRACT:
            with self.subTest(route=f"{method} {path}"):
                answers = self.spec["paths"][path][method.lower()]["responses"]
                self.assertIn("default", answers)
                self.assertEqual("ErrorOut", self._ref_name(answers["default"]["content"]["application/json"]["schema"]))

    def test_no_route_path_names_an_emergency_stop(self) -> None:
        from api.app import create_app

        paths = {getattr(route, "path", "") for route in create_app().routes} | set(self.spec["paths"])
        offenders = sorted(p for p in paths if any(word in p.lower() for word in _FORBIDDEN_IN_PATHS))
        self.assertEqual([], offenders, "no route may pose as an emergency stop")

    def test_the_brake_says_it_is_not_an_emergency_stop_and_the_red_button_is(self) -> None:
        brake = self.spec["paths"]["/v1/cell/brake"]["post"]
        said = (brake.get("summary", "") + " " + brake.get("description", "")).lower()
        # What it does as shipped (robot.ur.brake_on_halt off): it latches; a brake is claimed only where enabled.
        self.assertIn("stops the run and latches the arm", said)
        self.assertIn("brakes a move in flight where enabled", said)
        self.assertNotIn("brakes the arm under control", said)
        self.assertIn("not an emergency stop", said)
        self.assertIn("the red button is", said)
        # And the API's own description still sends the reader to the mushroom, and says what the brake is not.
        description = self.spec["info"]["description"].lower()
        self.assertIn("mushroom", description)
        self.assertIn("brake", description)
        self.assertIn("not an emergency stop", description)

    def test_the_pick_stop_keeps_its_pinned_words(self) -> None:
        stop = self.spec["paths"]["/v1/pick/stop"]["post"]
        said = (stop.get("description", "") + stop.get("summary", "")).lower()
        self.assertIn("before it begins the next attempt", said)
        self.assertIn("in flight", said)


# ---------------------------------------------------------------------------------------------------------------------
# The codes
# ---------------------------------------------------------------------------------------------------------------------


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class TheCodesAreOneVocabularyTests(_ScratchConsole):
    def test_v1_codes_equals_the_str_enums(self) -> None:
        import api.codes as codes

        body = self._client().get("/v1/codes")
        self.assertEqual(200, body.status_code, body.text)
        served = body.json()
        for field, enum_name, _literal in _CATALOGS:
            with self.subTest(catalog=field):
                self.assertEqual([m.value for m in getattr(codes, enum_name)], served[field])
        self.assertEqual(
            [k.value for k in codes.RunKind if k in codes.MOVING_KINDS], served["moving_kinds"]
        )
        self.assertEqual({c.value: codes.STOP_CLASS[c].value for c in codes.StopCode}, served["stop_class_of"])
        self.assertEqual(
            {light.value: [c.value for c in found] for light, found in codes.LIGHT_CODES.items()},
            served["light_codes_of"],
        )

    def test_each_literal_equals_its_str_enum_in_order(self) -> None:
        import api.codes as codes

        for _field, enum_name, literal_name in _CATALOGS:
            with self.subTest(catalog=enum_name):
                literal = getattr(codes, literal_name)
                self.assertEqual([m.value for m in getattr(codes, enum_name)], list(typing.get_args(literal)))

    def test_the_openapi_document_carries_every_union(self) -> None:
        """The frontend derives ``type StopCode = Schemas['CodesOut']['stop_codes'][number]`` from this."""
        import api.codes as codes
        from api.app import create_app

        schema = create_app().openapi()["components"]["schemas"]["CodesOut"]["properties"]
        for field, enum_name, _literal in _CATALOGS:
            with self.subTest(catalog=field):
                self.assertEqual([m.value for m in getattr(codes, enum_name)], schema[field]["items"]["enum"])

    def test_the_classes_the_states_and_the_lights_are_total(self) -> None:
        from api.codes import LIGHT_CODES, REFUSAL_STATUS, STOP_CLASS, LightCode, LightId, RefusalCode, StopCode

        self.assertEqual(set(StopCode), set(STOP_CLASS), "a stop code without a class")
        self.assertEqual(set(RefusalCode), set(REFUSAL_STATUS), "a refusal without its one status")
        self.assertEqual(set(LightId), set(LIGHT_CODES), "a light without its codes")
        self.assertEqual(set(LightCode), {c for found in LIGHT_CODES.values() for c in found}, "a code no light has")

    def test_the_library_s_codes_are_in_the_catalog(self) -> None:
        from api.codes import BlockerCode, EventType, RefusalCode
        from api.lifecycle import ConnectRefused
        from src.config.edit import WriteRefused
        from src.robot.grasping.loop.progress import PickStage

        refusals = {c.value for c in RefusalCode}
        self.assertLessEqual({c.value for c in ConnectRefused}, refusals)
        self.assertLessEqual({c.value for c in WriteRefused}, refusals)
        # A blocker is the refusal of the same name, said before a button is pressed.
        self.assertLessEqual({c.value for c in BlockerCode}, refusals)
        self.assertLessEqual({f"pick.{stage.value}" for stage in PickStage}, {e.value for e in EventType})

    def test_every_code_the_api_answers_with_is_in_the_catalog(self) -> None:
        """A code raised anywhere under ``api/`` that the catalog does not name would reach the browser untranslated."""
        from api.codes import RefusalCode

        known = {c.value for c in RefusalCode}
        found: dict[str, str] = {}
        pattern = re.compile(r'"code":\s*"([a-z_]+)"|\((?:4|5)\d\d,\s*"([a-z_]+)"')
        for source in sorted((_ROOT / "api").rglob("*.py")):
            for match in pattern.finditer(source.read_text(encoding="utf-8")):
                found[match.group(1) or match.group(2)] = source.name
        self.assertGreater(len(found), 10, "the scan found nothing: the pattern no longer matches the routers")
        self.assertEqual({}, {code: where for code, where in found.items() if code not in known})

    def test_not_acknowledged_means_only_connect_s_428(self) -> None:
        from api.codes import REFUSAL_STATUS, BlockerCode, LightCode, RefusalCode, StopCode

        self.assertEqual(428, REFUSAL_STATUS[RefusalCode.NOT_ACKNOWLEDGED])
        for other in (StopCode, LightCode, BlockerCode):
            self.assertNotIn("not_acknowledged", {c.value for c in other}, other.__name__)
        # The stop card's gate is its own code, never Connect's.
        self.assertIn("cell_not_cleared", {c.value for c in BlockerCode})
        # Written in exactly one place that answers with it: Connect's own refusals.
        writers = sorted(
            source.relative_to(_ROOT).as_posix() for source in (_ROOT / "api").rglob("*.py")
            if '"not_acknowledged"' in source.read_text(encoding="utf-8")
        )
        self.assertEqual(["api/codes.py", "api/lifecycle.py"], writers)
        # And a bare connect is answered with it, as a 428.
        client = self._client()
        self.assertEqual(200, client.post("/v1/cell/build", params={"rehearse": True}).status_code)
        refused = client.post("/v1/cell/connect", json={"token": "not-a-real-token"})
        self.assertEqual(428, refused.status_code, refused.text)
        self.assertEqual("not_acknowledged", refused.json()["code"])

    def test_the_frontend_lists_the_same_codes(self) -> None:
        """``frontend/src/api/codes.ts`` mirrors ``GET /v1/codes``; tsc checks it against ``schema.d.ts`` too."""
        import api.codes as codes

        text = _CODES_TS.read_text(encoding="utf-8")
        lists = {
            name: re.findall(r"'([a-z_.]+)'", body)
            for name, body in re.findall(r"export const ([A-Z_]+) = every<\w+>\(\)\(\[(.*?)\]\)", text, flags=re.S)
        }
        for field, enum_name, _literal in _CATALOGS:
            with self.subTest(catalog=field):
                self.assertEqual([m.value for m in getattr(codes, enum_name)], lists.get(field.upper()))
        moving = re.search(r"export const MOVING_KINDS[^=]*=\s*\[(.*?)\]", text, flags=re.S)
        self.assertIsNotNone(moving)
        assert moving is not None
        self.assertEqual(
            [k.value for k in codes.RunKind if k in codes.MOVING_KINDS], re.findall(r"'([a-z_]+)'", moving.group(1))
        )
        table = re.search(r"export const STOP_CLASS_OF[^{]*\{(.*?)\}", text, flags=re.S)
        self.assertIsNotNone(table)
        assert table is not None
        classes = dict(re.findall(r"^\s+([a-z_]+): '([a-z_]+)',?$", table.group(1), flags=re.M))
        self.assertEqual({c.value: codes.STOP_CLASS[c].value for c in codes.StopCode}, classes)

    def test_no_part_modelled_is_one_rule_for_the_server_and_the_browser(self) -> None:
        """The stop card's "no part held, now" gate (build plan 1.3.5) reads ``CellOut.payload_model`` through one set.

        ``none`` and ``not_applicable`` (an arm that models no part at all: the dummy, sim) both model none, so the
        gate opens on the rehearsal cell as on a UR; ``unknown`` (a read that failed) is not in it, so a part nobody
        can rule out keeps the gate shut. ``codes.ts`` lists the same set for the cockpit.
        """
        from api.schemas import NO_PART_MODELLED, PayloadModelName

        self.assertEqual(frozenset({"none", "not_applicable"}), NO_PART_MODELLED)
        self.assertLessEqual(NO_PART_MODELLED, set(typing.get_args(PayloadModelName)))
        self.assertIn("unknown", typing.get_args(PayloadModelName))
        listed = re.search(r"export const NO_PART_MODELLED[^=]*=\s*\[(.*?)\]", _CODES_TS.read_text(encoding="utf-8"),
                           flags=re.S)
        self.assertIsNotNone(listed, "codes.ts does not list NO_PART_MODELLED")
        assert listed is not None
        self.assertEqual(sorted(NO_PART_MODELLED), sorted(re.findall(r"'([a-z_]+)'", listed.group(1))))


# ---------------------------------------------------------------------------------------------------------------------
# The shared seams: start_kind, the recovery record, the stamps, the cell's stream
# ---------------------------------------------------------------------------------------------------------------------


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class AProblemStopWritesTheRecoveryRecordTests(_ScratchConsole):
    """A moving run that ends on a problem code leaves a record on the console; the record gates every new motion."""

    def _run(self, kind: str, code: str, *, error: str = "", raises: BaseException | None = None) -> Any:
        from api.runs import RunKind

        def driver(_console: Any, run: Any) -> str:
            if raises is not None:
                raise raises
            run.error = error
            return code

        run = self.cell.registry.start_kind(self.cell, RunKind(kind), driver)
        _wait_for_event(self.cell.hub, run.id, "run_finished")
        return run

    def test_a_halted_task_writes_the_record_and_publishes_it_on_the_cell_stream(self) -> None:
        from api.events import CELL_STREAM

        seen_lock: list[Any] = []
        record_problem_stop = self.cell.record_problem_stop

        def recorder(run: Any) -> Any:
            seen_lock.append(self.cell.active_run_id)
            return record_problem_stop(run)

        self.cell.record_problem_stop = recorder  # type: ignore[method-assign]
        run = self._run("task", "halted", error="the console halted the arm (halt now); nothing was commanded after it")

        record = self.cell.recovery
        self.assertIsNotNone(record, "a problem stop of a moving run left no recovery record")
        assert record is not None
        self.assertEqual((run.id, "task", "halted"), (record.run_id, record.kind, record.stop_code))
        self.assertIsNone(record.cleared_at)
        self.assertEqual([run.id], seen_lock, "the run let go of its lock before the record stood")
        self.assertIsNone(self.cell.active_run_id)

        body = self._client().get(f"/v1/runs/{run.id}").json()
        self.assertEqual(("failed", "task", "halted", "problem"),
                         (body["state"], body["kind"], body["stop_code"], body["stop_class"]))
        kinds = [e.type for e in self.cell.hub.since(run.id, 0)[0]]
        self.assertEqual(["run_started", "run_error", "run_finished"], kinds)
        error = next(e for e in self.cell.hub.since(run.id, 0)[0] if e.type == "run_error")
        self.assertEqual("halted", error.data["stop_code"])
        announced = _wait_for_event(self.cell.hub, CELL_STREAM, "cell.recovery")
        self.assertEqual({"run_id": run.id, "kind": "task", "stop_code": "halted", "at": record.at},
                         {k: announced.data[k] for k in ("run_id", "kind", "stop_code", "at")})
        # And /v1/cell says so, for the stop card of a page opened afterwards.
        cell = self._client().get("/v1/cell").json()
        self.assertEqual(run.id, cell["recovery"]["run_id"])
        self.assertEqual("cell_not_cleared", self.cell.recovery_gate())

    def test_run_finished_carries_the_whole_run_on_both_paths(self) -> None:
        """``run_finished`` is the full ``RunOut``: a run's own ``step`` is data, never the envelope's ``step``."""
        import types

        from api.events import EventHub
        from api.runs import RunRegistry
        from api.schemas import RunOut

        def home(_console: Any, run: Any) -> str:
            run.step = "return"      # the last timeline step, as a driver leaves it
            return "finished"

        run = self.cell.registry.start_kind(self.cell, "home", home, inputs={"to": "home"})
        finished = _wait_for_event(self.cell.hub, run.id, "run_finished")
        self.assertEqual(set(RunOut.model_fields), set(finished.data))
        self.assertEqual(("return", ""), (finished.data["step"], finished.step), "the run's step went into the envelope")
        started = _wait_for_event(self.cell.hub, run.id, "run_started")
        self.assertEqual({"kind": "home", "to": "home"}, started.data)

        # And a pick run, whose body is the pinned one: its record grows by the same fields.
        hub = EventHub()
        service = types.SimpleNamespace(attach_progress_listener=lambda _l: None, set_cancel_check=lambda _c: None)
        bare = types.SimpleNamespace(active_run_id=None, session=types.SimpleNamespace(service=service))
        pick = RunRegistry(hub).start(bare, prompt="", picks=0)
        said = _wait_for_event(hub, pick.id, "run_finished")
        self.assertEqual(set(RunOut.model_fields), set(said.data))
        self.assertEqual(("pick", ""), (said.data["kind"], said.data["step"]))

    def test_the_record_survives_a_rebuild(self) -> None:
        run = self._run("home", "return_failed", error="the planned move home was refused")
        record = self.cell.recovery
        self.assertIsNotNone(record)
        self.cell.build(rehearse=True)          # adopt: a new service, the record stays
        self.assertIs(record, self.cell.recovery, "the rebuild dropped the recovery record")
        self.assertEqual(run.id, self._client().get("/v1/cell").json()["recovery"]["run_id"])

    def test_a_driver_that_raises_is_a_software_error_and_still_writes_the_record(self) -> None:
        run = self._run("task", "finished", raises=RuntimeError("the library raised"))
        self.assertEqual(("software_error", "problem"), (run.stop_code, run.stop_class))
        self.assertIn("RuntimeError: the library raised", run.error)
        self.assertIsNotNone(self.cell.recovery)
        self.assertIsNone(self.cell.active_run_id)

    def test_no_record_for_a_benign_end_nor_for_a_run_that_moves_nothing(self) -> None:
        for kind, code in (("task", "finished"), ("task", "stopped_after_part"), ("task", "target_lost"),
                           ("teach", "disconnected"), ("planner", "planner_failed"), ("teach", "heartbeat_lost")):
            with self.subTest(kind=kind, code=code):
                run = self._run(kind, code)
                self.assertEqual(code, run.stop_code)
                self.assertIsNone(self.cell.recovery, f"{kind} ending {code} wrote a recovery record")

    def test_each_class_ends_in_its_run_state(self) -> None:
        from api.codes import STOP_CLASS, StopClass

        expected = {StopClass.DONE: "finished", StopClass.OPERATOR: "cancelled", StopClass.ASK: "finished",
                    StopClass.TEACH: "failed", StopClass.PLANNER: "failed", StopClass.PROBLEM: "failed"}
        for code, cls in STOP_CLASS.items():
            with self.subTest(code=code):
                run = self._run("teach", code.value)  # teach: nothing moves, so no record comes in the way
                self.assertEqual((expected[cls], cls.value), (str(run.state), run.stop_class))

    def test_one_run_at_a_time_across_every_kind(self) -> None:
        from api.runs import RunConflict, RunKind

        release = threading.Event()
        self.addCleanup(release.set)

        def holds(_console: Any, _run: Any) -> str:
            release.wait(timeout=10)
            return "finished"

        first = self.cell.registry.start_kind(self.cell, RunKind.PLANNER, holds)
        with self.assertRaises(RunConflict):
            self.cell.registry.start_kind(self.cell, RunKind.HOME, holds)
        self.assertEqual(first.id, self.cell.active_run_id)
        release.set()
        _wait_for_event(self.cell.hub, first.id, "run_finished")
        self.assertIsNone(self.cell.active_run_id)

    def test_the_record_gates_until_cleared_and_ends_on_arrival(self) -> None:
        from api.events import CELL_STREAM

        stopped = self._run("task", "part_still_held", error="the place was refused before the release")
        self.assertEqual("cell_not_cleared", self.cell.recovery_gate())
        cleared = self.cell.mark_cell_clear()
        assert cleared is not None
        self.assertGreaterEqual(cleared.cleared_at or 0.0, cleared.at)
        self.assertEqual("restart_required", self.cell.recovery_gate())
        ended = self.cell.end_recovery("run-restart", "restart")
        self.assertIsNotNone(ended)
        self.assertIsNone(self.cell.recovery)
        self.assertEqual("", self.cell.recovery_gate())
        said = _wait_for_event(self.cell.hub, CELL_STREAM, "cell.recovery_ended")
        self.assertEqual({"run_id": stopped.id, "by": "restart", "ended_by": "run-restart"}, said.data)
        self.assertIsNone(self.cell.end_recovery("run-other", "home"), "nothing stands, so nothing ends")

    def test_a_due_countdown_moves_nothing_and_a_stop_during_it_cancels(self) -> None:
        """Build plan item 11: after a teach, the next moving run counts down before its first motion, and a stop during
        the countdown ends it ``cancelled`` with nothing moved. True of the contract stage's hook, which cancels a due
        run at once, and of the built countdown, which reads the run's stop every 100 ms."""
        self.cell.stamp_teach_ended()
        driven: list[str] = []

        def home(_console: Any, run: Any) -> str:
            driven.append(run.id)
            return "finished"

        run = self.cell.registry.start_kind(self.cell, "home", home, inputs={"to": "home"})
        self.cell.registry.stop(run.id)
        _wait_for_event(self.cell.hub, run.id, "run_finished")
        self.assertEqual([], driven, "the run drove the arm although its countdown was due and stopped")
        self.assertEqual(("cancelled", "operator", "cancelled"), (run.stop_code, run.stop_class, str(run.state)))
        self.assertIsNone(self.cell.recovery, "a run that never moved left a recovery record")
        self.assertIsNone(self.cell.last_motion_started_at, "a run that never moved stamped a motion")
        self.assertTrue(self.cell.countdown_due(), "a run that never moved took the teach's countdown with it")

    def test_a_moving_run_the_cell_went_down_under_ends_disconnected_and_leaves_the_record(self) -> None:
        """A Disconnect abandons the run (``RunRegistry.abandon``): whatever its driver answers, the run ends
        ``disconnected``, a problem, and the record names it; its stop request says it came from the disconnect."""
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def task(_console: Any, _run: Any) -> str:
            entered.set()
            release.wait(timeout=10)
            return "finished"

        run = self.cell.registry.start_kind(self.cell, "task", task)
        self.assertTrue(entered.wait(timeout=10), "the driver never ran")
        because = "the cell was disconnected from the console while the run was active"
        self.assertIs(run, self.cell.registry.abandon(because))
        release.set()
        _wait_for_event(self.cell.hub, run.id, "run_finished")
        self.assertEqual(("disconnected", "problem", "failed"), (run.stop_code, run.stop_class, str(run.state)))
        self.assertIn(because, run.error)
        record = self.cell.recovery
        self.assertIsNotNone(record, "a moving run the cell went down under left no recovery record")
        assert record is not None
        self.assertEqual((run.id, "task", "disconnected"), (record.run_id, record.kind, record.stop_code))
        asked = _wait_for_event(self.cell.hub, run.id, "run_stop_requested")
        self.assertEqual("disconnect", asked.data.get("scope"))


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class TheConsoleStampsAndTheCellTests(_ScratchConsole):
    def test_a_teach_or_open_now_makes_the_next_motion_count_down(self) -> None:
        self.assertFalse(self.cell.countdown_due())
        self.cell.stamp_teach_ended()
        self.assertTrue(self.cell.countdown_due())
        self.assertEqual("teach", self.cell.countdown_because())
        self.cell.stamp_motion_started()
        self.assertFalse(self.cell.countdown_due(), "a motion since the teach still counts down")
        self.cell.stamp_jaws_open(opened=True)
        self.assertEqual("jaws_opened", self.cell.countdown_because())
        self.assertIsNotNone(self.cell.jaws_confirmed_at)
        self.cell.stamp_motion_started()
        # An answer that found the jaws open sends nothing, so nothing to keep hands away from.
        self.cell.stamp_jaws_open(opened=False)
        self.assertFalse(self.cell.countdown_due())
        self.assertTrue(self._client().get("/v1/cell").json()["countdown_due"] is False)

    def test_a_fresh_cell_renders_every_new_field(self) -> None:
        body = self._client().get("/v1/cell").json()
        self.assertEqual("none", body["hand"]["kind"])
        self.assertEqual("", body["needs_person"])
        self.assertIsNone(body["halted"])
        self.assertEqual("not_used", body["planner"]["state"])
        self.assertEqual("not_applicable", body["payload_model"])
        self.assertIsNone(body["recovery"])
        self.assertIsNone(body["jaws_confirmed_at"])
        self.assertFalse(body["countdown_due"])
        self.assertFalse(body["jaws_question"])

    def test_a_built_cell_names_its_hand(self) -> None:
        client = self._client()
        self.assertEqual(200, client.post("/v1/cell/build", params={"rehearse": True}).status_code)
        hand = client.get("/v1/cell").json()["hand"]
        self.assertEqual(("width", "DummyGripper", "not_counted"), (hand["kind"], hand["driver"], hand["jaws"]))
        self.assertFalse(hand["connected"])

    def test_the_build_installs_the_browser_question_after_it_adopted_the_cell(self) -> None:
        from unittest.mock import patch

        seen: list[tuple[str, bool]] = []

        def install(console: Any) -> None:
            seen.append((str(console.session.state), console.session.service is not None))

        with patch("api.jaws.install", side_effect=install):
            self.cell.build(rehearse=True)
        self.assertEqual([("built", True)], seen)

    def test_a_build_whose_jaws_question_cannot_be_installed_leaves_the_cell_unbuilt(self) -> None:
        """Build plan 1.6: a failed install fails the build, so Connect's ``jaws_seam_missing`` is only a backstop. The
        cell is left as any failed build leaves it, unbuilt, never built with a hand that would ask at the terminal."""
        from unittest.mock import patch

        def refuses(_console: Any) -> None:
            raise RuntimeError("the browser seam could not be installed on this hand")

        client = self._client()
        with patch("api.jaws.install", side_effect=refuses):
            refused = client.post("/v1/cell/build", params={"rehearse": True})
        self.assertEqual((422, "build_refused"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertIn("could not be installed", refused.json()["message"])
        body = client.get("/v1/cell").json()
        self.assertEqual(("disconnected", None, None), (body["state"], body["arm"], body["gripper"]))
        self.assertIsNone(self.cell.session.service)
        preview = client.get("/v1/cell/connect-preview")
        self.assertEqual((409, "not_built"), (preview.status_code, preview.json()["code"]), preview.text)
        # And the next build, whose install succeeds, builds as ever.
        self.assertEqual(200, client.post("/v1/cell/build", params={"rehearse": True}).status_code)
        self.assertEqual("built", client.get("/v1/cell").json()["state"])

    def test_the_stamps_are_set_only_through_their_methods(self) -> None:
        """The countdown orders the stamps by the methods' own counter, never by the wall clock, so a stamp written
        directly would turn the hands-off countdown off unseen. Each stamp is read-only instead: a track that writes one
        fails at once."""
        for name in ("jaws_confirmed_at", "jaws_opened_at", "teach_ended_at", "last_motion_started_at"):
            with self.subTest(stamp=name):
                with self.assertRaises(AttributeError):
                    setattr(self.cell, name, time.time())
                self.assertIsNone(getattr(self.cell, name))
        self.assertFalse(self.cell.countdown_due())
        self.cell.stamp_jaws_open(opened=True)
        self.assertIsNotNone(self.cell.jaws_opened_at)
        self.assertEqual(self.cell.jaws_confirmed_at, self.cell.jaws_opened_at)
        self.assertTrue(self.cell.countdown_due())

    def test_the_cell_answer_reads_the_recovery_record_once(self) -> None:
        """A Restart's arrival ends the record on its run's thread while a browser polls ``GET /v1/cell``: the answer
        reads the record once, so a record ended between two reads can never turn the poll into a 500."""
        from api.cell import RecoveryRecord
        from api.codes import RunKind, StopCode
        from api.routers.cell import _render

        record = RecoveryRecord(run_id="run-stopped", kind=RunKind.TASK, stop_code=StopCode.HALTED, at=1.0)
        cell, reads = self.cell, []

        class EndedBetweenTwoReads:
            """The console, whose record a run thread ends right after the first read."""

            def __getattr__(self, name: str) -> Any:
                if name == "recovery":
                    reads.append(name)
                    return record if len(reads) == 1 else None
                return getattr(cell, name)

        rendered = _render(EndedBetweenTwoReads())  # type: ignore[arg-type]
        self.assertIsNotNone(rendered.recovery)
        assert rendered.recovery is not None
        self.assertEqual("run-stopped", rendered.recovery.run_id)

    def test_on_the_rehearsal_cell_no_part_is_modelled(self) -> None:
        """The stop card's "no part held, now" gate on the dummy cell, where the smoke test restarts after a stop: the
        dummy arm models no part at all (``not_applicable``), which is no part modelled, and its hand holds nothing."""
        from api.schemas import NO_PART_MODELLED

        client = self._client()
        self.assertEqual(200, client.post("/v1/cell/build", params={"rehearse": True}).status_code)
        body = client.get("/v1/cell").json()
        self.assertEqual("not_applicable", body["payload_model"])
        self.assertIn(body["payload_model"], NO_PART_MODELLED)
        self.assertNotEqual("closed", body["hand"]["jaws"])

    def test_a_panel_read_that_raises_never_fails_the_cell_answer(self) -> None:
        """``GET /v1/cell`` reads the arm and the hand on every poll, and build, connect and disconnect answer with the
        same read: an arm or a hand whose read raises is said, never a 500 after a transition that happened."""
        from unittest.mock import PropertyMock, patch

        client = self._client()
        self.assertEqual(200, client.post("/v1/cell/build", params={"rehearse": True}).status_code)
        arm, hand = type(self.cell.session.arm), type(self.cell.session.gripper)
        with patch.object(arm, "halt_state", PropertyMock(side_effect=RuntimeError("the latch lock is held")),
                          create=True), \
             patch.object(arm, "planner_state", PropertyMock(side_effect=RuntimeError("the planner lock is held")),
                          create=True), \
             patch.object(hand, "where_pin", PropertyMock(side_effect=ConnectionError("I/O link down")), create=True):
            answered = client.get("/v1/cell")
        self.assertEqual(200, answered.status_code, answered.text)
        body = answered.json()
        self.assertIsNone(body["halted"])
        self.assertEqual("off", body["planner"]["state"])
        self.assertEqual("", body["hand"]["where"])

    def test_the_cell_stream_is_a_stream_no_run_can_be_called(self) -> None:
        from api.events import CELL_STREAM

        self.assertEqual("cell", CELL_STREAM)
        run = self.cell.registry.start_kind(self.cell, "planner", lambda _c, _r: "planner_ready")
        self.assertTrue(run.id.startswith("run-"))
        self.assertNotEqual(CELL_STREAM, run.id)
        _wait_for_event(self.cell.hub, run.id, "run_finished")


class TheHandIsNamedTests(unittest.TestCase):
    """``api.jaws.hand_of``: the hand chip and the stop card's gates read the hand through this one function."""

    def test_the_owner_s_toggle_on_tool_do0(self) -> None:
        from api.jaws import hand_of
        from tests.test_a_toggle_asks_where_its_jaws_stand import Person, _toggle

        jaws = _toggle([], Person(""))
        before = hand_of(jaws)
        self.assertEqual(("toggle", "not_counted", False), (before.kind, before.jaws, before.connected))
        self.assertIn("not connected", before.why_unknown)
        jaws.connect()
        hand = hand_of(jaws)
        self.assertEqual(("toggle", "JawIOGripper", "tool output 0", "open", "", True, True),
                         (hand.kind, hand.driver, hand.where, hand.jaws, hand.why_unknown, hand.no_sensor,
                          hand.connected))
        self.assertEqual(jaws.commands_sent, hand.commands_sent)

    def test_an_output_switched_at_the_pendant_is_unknown_never_open(self) -> None:
        from api.jaws import hand_of
        from tests.test_a_toggle_asks_where_its_jaws_stand import Person, _toggle

        jaws = _toggle([], Person(""))
        jaws.connect()
        jaws._io.do[0] = not jaws._io.do.get(0, False)   # type: ignore[attr-defined] # somebody at the pendant
        hand = hand_of(jaws)
        self.assertEqual("unknown", hand.jaws)
        self.assertIn("somebody switched it", hand.why_unknown)

    def test_a_read_that_raises_is_said_not_raised(self) -> None:
        from api.jaws import hand_of
        from tests.test_a_toggle_asks_where_its_jaws_stand import Person, _toggle

        jaws = _toggle([], Person(""))
        jaws.connect()

        def broken(*_a: Any, **_k: Any) -> bool:
            raise RuntimeError("RTDE receive dropped")

        jaws._io.get_digital_output = broken  # type: ignore[attr-defined,method-assign]
        hand = hand_of(jaws)
        self.assertEqual("unknown", hand.jaws)
        self.assertIn("RTDE receive dropped", hand.why_unknown)

    def test_a_hand_whose_reads_raise_is_said_never_raised(self) -> None:
        """Every read of the hand, the pin it is driven on and whether it is a toggle or measures its width included, is
        said when it raises, never raised to the page; a toggle nobody can read stands ``unknown``, never ``open``."""
        from api.jaws import hand_of

        class WhereRaises:
            is_connected = True

            @property
            def where_pin(self) -> str:
                raise ConnectionError("I/O link down")

        class WidthRaises:
            is_connected = True

            def width_is_measured(self) -> bool:
                raise TimeoutError("position register did not answer")

        class ToggleNobodyCanRead:
            toggles_without_sensor = True
            is_connected = True
            edge_unknown = False

            @property
            def jaws_closed(self) -> bool:
                raise ConnectionError("RTDE receive dropped")

            def jaws_open_for_a_pick(self) -> str:
                return ""

            def why_jaws_unknown(self) -> str:
                return ""

        hand = hand_of(WhereRaises())
        self.assertEqual(("width", ""), (hand.kind, hand.where))
        hand = hand_of(WidthRaises())
        self.assertEqual(("width", True), (hand.kind, hand.no_sensor), "an unreadable sensor was claimed")
        hand = hand_of(ToggleNobodyCanRead())
        self.assertEqual(("toggle", "unknown"), (hand.kind, hand.jaws))
        self.assertTrue(hand.why_unknown, "an unknown count without a reason")


class ThePanelReadsNeverRaiseTests(unittest.TestCase):
    """What ``GET /v1/cell`` reads of the arm and the service besides the hand (``api/routers/cell.py``), on every poll
    and after every build, connect and disconnect: a read that raises is never a 500, and each fails closed."""

    def test_the_halt_latch(self) -> None:
        from unittest.mock import MagicMock

        from api.routers.cell import _halted

        class Latched:
            def halt_state(self) -> Any:
                return types.SimpleNamespace(reason="the operator pressed halt now", requested_at=12.5, in_motion=True,
                                             braked=False, brake_s=None)

        class ReaderRaises:
            @property
            def halt_state(self) -> Any:
                raise RuntimeError("the latch lock is held")

        class TornState:
            @property
            def reason(self) -> str:
                raise RuntimeError("a torn read")

            requested_at = 1.0

        class StateRaises:
            def halt_state(self) -> Any:
                return TornState()

        halted = _halted(Latched())
        self.assertIsNotNone(halted)
        assert halted is not None
        self.assertEqual(("the operator pressed halt now", 12.5, True, False, None),
                         (halted.reason, halted.requested_at, halted.in_motion, halted.braked, halted.brake_s))
        for arm in (None, object(), MagicMock(), ReaderRaises(), StateRaises()):
            with self.subTest(arm=type(arm).__name__):
                self.assertIsNone(_halted(arm))

    def test_the_planner(self) -> None:
        from unittest.mock import MagicMock

        from api.routers.cell import _planner

        class Ready:
            planner_state = "ready"

        class Starting:
            def planner_state(self) -> str:
                return "starting"

        class Raises:
            @property
            def planner_state(self) -> str:
                raise RuntimeError("the planner lock is held")

        self.assertEqual("ready", _planner(Ready()).state)
        self.assertEqual("starting", _planner(Starting()).state)
        for plans_nothing in (None, object(), MagicMock()):
            with self.subTest(arm=type(plans_nothing).__name__):
                self.assertEqual("not_used", _planner(plans_nothing).state)
        # An arm that has a planner and cannot say its state is not ready: never "not_used", which reads as nothing
        # to wait for.
        self.assertEqual("off", _planner(Raises()).state)

    def test_the_carried_part(self) -> None:
        from api.routers.cell import _payload_model
        from api.schemas import NO_PART_MODELLED
        from src.robot.core.arm_capabilities import PayloadModel

        class Carries:
            def __init__(self, answer: Any) -> None:
                self.answer = answer

            def attach_payload(self, grip_width_mm: float) -> bool:
                return False

            def detach_payload(self) -> bool:
                return True

            def payload_declined_reason(self) -> str | None:
                return None

            def payload_model(self) -> Any:
                if isinstance(self.answer, BaseException):
                    raise self.answer
                return self.answer

        for answer, expected in ((PayloadModel.NONE, "none"), (PayloadModel.FILTER_ONLY, "filter_only"),
                                 (PayloadModel.PLANNER_AND_FILTER, "planner_and_filter"),
                                 (RuntimeError("the planner client dropped"), "unknown"),
                                 ("carried_by_a_new_model", "unknown")):
            with self.subTest(answer=str(answer)):
                self.assertEqual(expected, _payload_model(Carries(answer)))
        for models_none in (None, object()):
            with self.subTest(arm=type(models_none).__name__):
                self.assertEqual("not_applicable", _payload_model(models_none))
        self.assertNotIn("unknown", NO_PART_MODELLED, "a part nobody could rule out would open 'no part held'")

    def test_the_service_s_latch(self) -> None:
        from api.routers.cell import _needs_person

        class Raises:
            @property
            def stopped_where_the_arm_stands(self) -> str:
                raise RuntimeError("a torn read")

        self.assertEqual("", _needs_person(None))
        self.assertEqual("", _needs_person(types.SimpleNamespace(stopped_where_the_arm_stands="")))
        stopped = types.SimpleNamespace(stopped_where_the_arm_stands="the push stopped")
        self.assertEqual("the push stopped", _needs_person(stopped))
        # A latch nobody could read is never "nobody is needed".
        self.assertIn("could not be read", _needs_person(Raises()))


class APickRunSaysItsKindAndHowItStopsTests(unittest.TestCase):
    """What every run's events gain (build plan 1.4): ``run_started`` names the kind, a pick run's included, and
    ``run_stop_requested`` the scope, so the timeline tells a pick's stop between attempts from a task's after the part
    and from a Disconnect's."""

    def test_a_pick_run_starts_as_a_pick_and_its_stop_acts_between_attempts(self) -> None:
        from api.events import EventHub
        from api.runs import RunRegistry

        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def pick(**_kwargs: Any) -> Any:
            entered.set()
            release.wait(timeout=10)
            return types.SimpleNamespace(outcome="no_target", succeeded=False, failure_summary=lambda: "nothing seen")

        service = types.SimpleNamespace(attach_progress_listener=lambda _listener: None,
                                        set_cancel_check=lambda _check: None, pick=pick)
        console = types.SimpleNamespace(active_run_id=None, session=types.SimpleNamespace(service=service))
        hub = EventHub()
        registry = RunRegistry(hub)
        run = registry.start(console, prompt="", picks=2)
        self.assertTrue(entered.wait(timeout=10), "the pick never started")
        self.assertTrue(registry.stop(run.id))
        release.set()
        _wait_for_event(hub, run.id, "run_finished")
        self.assertEqual(("cancelled", 1), (str(run.state), run.attempted))
        self.assertEqual("pick", _wait_for_event(hub, run.id, "run_started").data.get("kind"))
        self.assertEqual("between_attempts", _wait_for_event(hub, run.id, "run_stop_requested").data.get("scope"))

    def test_no_hand_is_none(self) -> None:
        from api.jaws import hand_of
        from src.robot.grippers.null import NullGripper

        for nothing in (None, NullGripper()):
            with self.subTest(hand=type(nothing).__name__):
                self.assertEqual("none", hand_of(nothing).kind)


# ---------------------------------------------------------------------------------------------------------------------
# The interfaces the API tracks code against
# ---------------------------------------------------------------------------------------------------------------------


class TheSeamsTheTracksCallExistTests(unittest.TestCase):
    """The library's names and parameters the API codes against, fixed before either side was built."""

    _SIGNATURES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("api.runs", "RunRegistry.start_kind", ("self", "console", "kind", "driver", "inputs", "fields")),
        ("api.cell", "Console.countdown_due", ("self",)),
        ("api.cell", "Console.countdown_because", ("self",)),
        ("api.cell", "Console.record_problem_stop", ("self", "run")),
        ("api.cell", "Console.end_recovery", ("self", "run_id", "by")),
        ("api.cell", "Console.mark_cell_clear", ("self", "at")),
        ("api.cell", "Console.recovery_gate", ("self",)),
        ("api.cell", "Console.stamp_jaws_open", ("self", "opened", "at")),
        ("api.cell", "Console.stamp_teach_ended", ("self", "at")),
        ("api.cell", "Console.stamp_motion_started", ("self", "at")),
        ("api.cell", "take_down", ("console", "why")),
        ("api.jaws", "hand_of", ("gripper",)),
        ("api.jaws", "install", ("console",)),
        ("api.jaws", "cancel_pending", ("console", "why")),
        ("api.jaws", "pending", ("console",)),
        ("api.jaws", "answer", ("console", "question_id", "choice")),
        ("api.jaws", "ended_open", ("console", "opened")),
        ("api.jaws", "before_change", ("console",)),
        ("api.teach", "end_and_join", ("console", "timeout_s")),
        ("api.teach", "drive_teach", ("console", "run")),
        ("api.teach", "snapshot", ("console", "run_id", "token")),
        ("api.live", "LiveFrames.frame", ("self", "service", "rig", "max_width")),
        ("api.live", "LiveFrames.ages", ("self",)),
        ("api.live", "LiveFrames.refresh_stale", ("self", "service", "max_age_s")),
        ("api.overlays", "OverlayStore.put", ("self", "run_id", "key", "png")),
        ("api.overlays", "OverlayStore.get", ("self", "run_id", "key")),
        ("api.overlays", "OverlayStore.forget", ("self", "run_id")),
        ("api.readiness", "readiness", ("console",)),
        ("api.readiness", "controller_gate", ("arm",)),
        ("api.readiness", "facts", ("console",)),
        ("api.task_run", "drive_task", ("console", "run")),
        ("api.task_run", "drive_home", ("console", "run")),
        ("api.task_run", "drive_planner", ("console", "run")),
    )

    def test_every_seam_has_its_name_and_parameters(self) -> None:
        import importlib

        for module_name, dotted, parameters in self._SIGNATURES:
            with self.subTest(seam=f"{module_name}.{dotted}"):
                target: Any = importlib.import_module(module_name)
                for part in dotted.split("."):
                    target = getattr(target, part)
                self.assertEqual(parameters, tuple(inspect.signature(target).parameters))

    def test_the_console_carries_its_new_state(self) -> None:
        from api.cell import Console
        from api.jaws import BrowserJawQuestions
        from api.live import LiveFrames
        from api.overlays import OverlayStore

        console = Console(root=_ROOT / "config", profile=None)
        for name in ("recovery", "jaws_confirmed_at", "jaws_opened_at", "teach_ended_at", "last_motion_started_at"):
            self.assertIsNone(getattr(console, name), name)
        self.assertIsInstance(console.jaws, BrowserJawQuestions)
        self.assertIsInstance(console.live, LiveFrames)
        self.assertIsInstance(console.overlays, OverlayStore)

    def test_the_seams_answer_in_their_types_while_nothing_is_going_on(self) -> None:
        """What each seam answers on a fresh console, true of the stage-0 stubs and of the built seams alike."""
        from api import jaws, readiness, teach
        from api.cell import Console
        from api.live import LiveFrames
        from api.overlays import OverlayStore, overlay_url
        from api.runs import Run
        from api.schemas import LiveFrameOut, ReadinessOut
        from api.task_run import ConsoleTaskHooks

        console = Console(root=_ROOT / "config", profile=None)
        self.assertIsInstance(readiness.readiness(console), ReadinessOut)
        self.assertIsInstance(readiness.controller_gate(None), str)
        live = LiveFrames()
        self.assertIsInstance(live.frame(None), LiveFrameOut)
        self.assertEqual({}, live.ages())
        self.assertEqual({}, live.refresh_stale(None))
        store = OverlayStore()
        self.assertIsNone(store.get("run-nobody", 1))
        self.assertIsNone(store.forget("run-nobody"))
        self.assertEqual("/v1/runs/run-a/overlays/2", overlay_url("run-a", 2))
        self.assertEqual("/v1/runs/run-a/target/overlay", overlay_url("run-a", "target"))
        self.assertTrue(teach.end_and_join(console, timeout_s=0.1), "a teach nobody started kept the lock")
        self.assertIsNone(teach.snapshot(console, "run-nobody", "token"))
        self.assertIsNone(jaws.pending(console))
        self.assertFalse(jaws.cancel_pending(console, "nothing waits"))
        self.assertTrue(callable(jaws.before_change(console)))
        run = Run(id="run-hooks", prompt="", requested_picks=0)
        hooks = ConsoleTaskHooks(console, run)
        self.assertEqual((False, False, ""), (hooks.stop_after_part(), hooks.halted(), hooks.abandoned()))
        run.stop_after_part, run.halt_requested, run.abandoned = True, True, "the cell was disconnected"
        self.assertEqual((True, True, "the cell was disconnected"),
                         (hooks.stop_after_part(), hooks.halted(), hooks.abandoned()))

    def test_take_down_keeps_the_order_and_always_puts_the_cell_down(self) -> None:
        """Build plan item 18: the question, the teach, the run, then the cell; a step that raises stops none after."""
        from unittest.mock import patch

        from api.cell import Console, take_down

        console = Console(root=_ROOT / "config", profile=None)
        order: list[str] = []
        with patch("api.jaws.cancel_pending", side_effect=lambda *_a, **_k: order.append("question")), \
             patch("api.teach.end_and_join", side_effect=lambda *_a, **_k: order.append("teach")), \
             patch.object(console.registry, "abandon", side_effect=lambda *_a: order.append("run")), \
             patch.object(console.session, "disconnect", side_effect=lambda: order.append("disconnect")):
            take_down(console, "the operator pressed Disconnect")
        self.assertEqual(["question", "teach", "run", "disconnect"], order)

        order.clear()

        def refuses(*_a: Any, **_k: Any) -> None:
            order.append("question")
            raise RuntimeError("the question would not end")

        with patch("api.jaws.cancel_pending", side_effect=refuses), \
             patch("api.teach.end_and_join", side_effect=lambda *_a, **_k: order.append("teach")), \
             patch.object(console.registry, "abandon", side_effect=lambda *_a: order.append("run")), \
             patch.object(console.session, "disconnect", side_effect=lambda: order.append("disconnect")):
            with self.assertRaises(RuntimeError):
                take_down(console, "the operator pressed Disconnect")
        self.assertEqual(["question", "teach", "run", "disconnect"], order, "a raising step kept the cell up")

    def test_a_question_nobody_answered_is_never_open(self) -> None:
        """The browser seam's one promise, stub or built: a question nobody answers ends in ``EOFError``, which the hand
        takes as no answer, never "open".

        Asked as the hand asks it (library L10's ``JawAsking``), on its own thread, and ended the way a Disconnect ends
        it: ``cancel`` once the question waits (``pending``). So the built seam, which waits up to 120 s for the
        browser, is never waited for here, and the stub, which answers ``EOFError`` at once, passes the same test.
        """
        from api.events import EventHub
        from api.jaws import BrowserJawQuestions

        seam = BrowserJawQuestions(EventHub())
        outcome: dict[str, BaseException | str] = {}

        def ask() -> None:
            try:
                outcome["answered"] = seam(_JawAsking())
            except EOFError as nobody:
                outcome["no_answer"] = nobody
            except BaseException as other:  # noqa: BLE001 (any other end is the failure this test reports)
                outcome["raised"] = other

        asking = threading.Thread(target=ask, daemon=True)
        asking.start()
        deadline = time.monotonic() + 10
        while asking.is_alive() and seam.pending() is None and time.monotonic() < deadline:
            time.sleep(0.005)
        seam.cancel("the contract test ends the question unanswered")
        asking.join(timeout=10)
        self.assertFalse(asking.is_alive(), "a cancelled question still waits for an answer")
        self.assertEqual({"no_answer"}, set(outcome), outcome)


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class ARouteNotBuiltYetMovesNothingTests(_ScratchConsole):
    """A contract route whose body a later track builds answers ``501 not_built_yet`` in the meantime, on a connected
    cell too, and touches nothing: no command to the arm, the hand or the service, and no run.

    Only the routes still stubbed are called (their handler raises ``not_built_yet``), so the test holds at every stage
    of the build and never drives a route a track has built.
    """

    #: One valid request per stubbed route: (method, route path, URL, JSON body).
    _REQUESTS: tuple[tuple[str, str, str, Any], ...] = (
        ("POST", "/v1/task", "/v1/task", {"object": "", "place": {"kind": "pose", "pose": None}}),
        ("POST", "/v1/task/stop", "/v1/task/stop?run_id=run-nobody", None),
        ("POST", "/v1/task/restart", "/v1/task/restart", {"run_id": "run-nobody"}),
        ("GET", "/v1/runs/{run_id}/overlays/{n}", "/v1/runs/run-nobody/overlays/1", None),
        ("GET", "/v1/runs/{run_id}/target/overlay", "/v1/runs/run-nobody/target/overlay", None),
        ("GET", "/v1/cell/facts", "/v1/cell/facts", None),
        ("GET", "/v1/cell/readiness", "/v1/cell/readiness", None),
        ("POST", "/v1/cell/acknowledge", "/v1/cell/acknowledge", {"cell_clear": True, "jaws_empty": True}),
        ("POST", "/v1/cell/brake", "/v1/cell/brake", None),
        ("POST", "/v1/cell/home", "/v1/cell/home", {"to": "home"}),
        ("POST", "/v1/cell/planner", "/v1/cell/planner", None),
        ("GET", "/v1/cell/jaws", "/v1/cell/jaws", None),
        ("POST", "/v1/cell/jaws/answer", "/v1/cell/jaws/answer", {"question_id": "q-nobody", "choice": "open_now"}),
        ("POST", "/v1/cell/jaws/check", "/v1/cell/jaws/check", None),
        ("GET", "/v1/camera/live", "/v1/camera/live", None),
        ("GET", "/v1/poses", "/v1/poses", None),
        ("PUT", "/v1/poses/default-place", "/v1/poses/default-place", {"name": None}),
        ("GET", "/v1/teach/payload", "/v1/teach/payload", None),
        ("POST", "/v1/teach", "/v1/teach", {"name": "drop_left", "label": "Ablage links", "payload_seen": None}),
        ("GET", "/v1/teach/{run_id}", "/v1/teach/run-nobody?token=t", None),
        ("POST", "/v1/teach/{run_id}/capture", "/v1/teach/run-nobody/capture?token=t", None),
        ("POST", "/v1/teach/{run_id}/cancel", "/v1/teach/run-nobody/cancel?token=t", None),
        ("POST", "/v1/commands/parse", "/v1/commands/parse", {"text": "nimm den grünen Würfel"}),
        ("GET", "/v1/commands/status", "/v1/commands/status", None),
        ("POST", "/v1/commands/warmup", "/v1/commands/warmup", None),
    )

    def test_every_route_not_built_yet_says_so_and_moves_nothing(self) -> None:
        from api.app import create_app

        app = create_app()
        stubs = _stub_routes(app)
        asked = {(method, path) for method, path, _url, _body in self._REQUESTS}
        self.assertLessEqual(stubs, asked, "a stubbed route has no request here")
        client = TestClient(app)
        self.assertEqual(200, client.post("/v1/cell/build", params={"rehearse": True}).status_code)
        token = client.get("/v1/cell/connect-preview").json()["token"]
        self.assertEqual(200, client.post("/v1/cell/connect", json={"token": token}).status_code)
        calls: list[str] = []
        for label, target in (("arm", self.cell.session.arm), ("hand", self.cell.session.gripper),
                              ("service", self.cell.session.service)):
            _record_calls(target, label, calls)

        for method, path, url, body in self._REQUESTS:
            if (method, path) not in stubs:
                continue  # built by its track, whose own tests say what it does
            with self.subTest(route=f"{method} {path}"):
                before = len(calls)
                answered = client.request(method, url, json=body)
                self.assertEqual((501, "not_built_yet"), (answered.status_code, answered.json().get("code")),
                                 answered.text)
                self.assertEqual(f"{method} {path}", answered.json()["detail"]["route"])
                self.assertEqual([], calls[before:], f"{method} {path} commanded the cell although it is not built")
        self.assertEqual([], client.get("/v1/runs").json(), "a route not built yet started a run")
        self.assertIsNone(self.cell.active_run_id)
        self.assertEqual("connected", client.get("/v1/cell").json()["state"])


# ---------------------------------------------------------------------------------------------------------------------
# The event logs the frontend replays
# ---------------------------------------------------------------------------------------------------------------------


class TheFixturesSpeakTheContractTests(unittest.TestCase):
    """``frontend/src/test/fixtures/*.json``: captured by ``scripts/console/capture_event_log.py``, but for
    ``teach_one_pose``, which is written by hand (no capture scenario teaches)."""

    _EXPECTED = {
        "task_two_parts_nothing_left", "task_camera_target_lost", "task_halted_then_restart",
        "jaws_question_round_trip", "teach_one_pose",
    }
    _ENVELOPE = {"type", "run_id", "seq", "ts", "severity", "human", "step", "step_index", "step_total", "data"}

    def _fixtures(self) -> dict[str, dict[str, Any]]:
        return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(_FIXTURES.glob("*.json"))}

    def test_the_five_scenarios_are_there(self) -> None:
        self.assertLessEqual(self._EXPECTED, set(self._fixtures()))

    def test_every_event_is_a_contract_event(self) -> None:
        from api.codes import STOP_CLASS, EventType, RunKind, StopCode
        from api.events import CELL_STREAM, Severity
        from api.schemas import JawQuestionOut, RunOut, TargetOut, TaskPlanOut

        for name, fixture in self._fixtures().items():
            with self.subTest(fixture=name):
                self.assertEqual({"about", "scenario", "events", "runs"}, set(fixture))
                self.assertEqual(name, fixture["scenario"])
                runs = {run_id: RunOut.model_validate(run) for run_id, run in fixture["runs"].items()}
                seqs: dict[str, int] = {}
                last_ts = 0.0
                for event in fixture["events"]:
                    self.assertEqual(self._ENVELOPE, set(event), event)
                    self.assertIn(event["type"], {e.value for e in EventType}, event["type"])
                    self.assertIn(event["severity"], {s.value for s in Severity})
                    stream = event["run_id"]
                    self.assertTrue(stream == CELL_STREAM or stream in runs, stream)
                    self.assertEqual(stream == CELL_STREAM, event["type"].startswith("cell."), event["type"])
                    seqs[stream] = seqs.get(stream, 0) + 1
                    self.assertEqual(seqs[stream], event["seq"], f"{stream} skips a seq")
                    self.assertGreaterEqual(event["ts"], last_ts, "events out of order")
                    last_ts = event["ts"]
                    data = event["data"]
                    if "stop_code" in data and data["stop_code"]:
                        self.assertIn(data["stop_code"], {c.value for c in StopCode})
                    if event["type"] == "run_started":
                        self.assertIn(data["kind"], {k.value for k in RunKind})
                        if data.get("plan") is not None:
                            TaskPlanOut.model_validate(data["plan"])
                    if event["type"] == "run_finished":
                        self.assertEqual(runs[stream], RunOut.model_validate(data), "run_finished is not the run")
                    if event["type"] == "task.target_found":
                        TargetOut.model_validate(data["target"])
                    if event["type"] == "cell.jaws_question":
                        JawQuestionOut.model_validate(data)
                for run_id, run in runs.items():
                    self.assertTrue(any(e["type"] == "run_finished" and e["run_id"] == run_id
                                        for e in fixture["events"]), f"{run_id} never finished")
                    if run.stop_code:
                        self.assertEqual(STOP_CLASS[StopCode(run.stop_code)].value, run.stop_class)

    def test_a_task_screens_its_taught_poses_before_its_first_motion(self) -> None:
        """Build plan 1.3.3: run_started, [gate], [screen], [setup], [restart], [survey], then the parts. A task whose
        plan uses a taught pose (a pose place, a taught return) screens it before it moves, a Restart before the
        planned move home included."""
        moves = {"task.return_started", "task.survey_started", "task.part_started", "task.carry_started",
                 "pick.pick_started"}
        for name, fixture in self._fixtures().items():
            for run_id, run in fixture["runs"].items():
                plan = run.get("plan")
                if run["kind"] != "task" or plan is None:
                    continue
                if plan["place"]["kind"] != "pose" and plan["return_to"] == "home":
                    continue
                own = [e["type"] for e in fixture["events"] if e["run_id"] == run_id]
                first_motion = next((i for i, kind in enumerate(own) if kind in moves), len(own))
                with self.subTest(fixture=name, run=run_id):
                    self.assertIn("task.pose_screened", own[:first_motion], f"{run_id} moved before it screened")


# ---------------------------------------------------------------------------------------------------------------------
# The library symbols the API reads, once their tracks have landed
# ---------------------------------------------------------------------------------------------------------------------


class TheLibraryShapesTheApiReadsTests(unittest.TestCase):
    """The shapes the API reads of what the library tracks build (fixed before either side was built).

    The arm and the hand are ``Any`` on the API side (``CellSession``), so no type checker sees a name drift: a halt
    state whose ``requested_at`` became ``requested_at_s`` would read as "not halted" on every poll. The Protocols in
    ``api`` name what is read; each test here waits for its library track and then pins that shape against the
    library's own objects. Until the track has landed it skips and says which; once any of its symbols has landed, it
    runs.
    """

    @staticmethod
    def _read_by_the_api(protocol: type) -> set[str]:
        return {name for name, member in vars(protocol).items() if isinstance(member, property)}

    @staticmethod
    def _carried_by(cls: Any) -> set[str]:
        """The fields a library record carries: a dataclass's, a named tuple's, or its annotations."""
        if dataclasses.is_dataclass(cls):
            return {f.name for f in dataclasses.fields(cls)}
        return set(getattr(cls, "_fields", ())) | set(getattr(cls, "__annotations__", {}))

    def test_a_latched_arm_reads_as_halted(self) -> None:
        from api.readiness import HaltStateLike
        from api.routers.cell import _halted
        from src.robot.core import arm_capabilities
        from src.robot.drivers.dummy.arm import DummyRobotArm

        if not hasattr(arm_capabilities, "SupportsHalt") and not callable(getattr(DummyRobotArm, "halt", None)):
            self.skipTest("the halt latch (library L9: SupportsHalt, HaltState) has not landed")
        halt_state = getattr(arm_capabilities, "HaltState", None)
        self.assertIsNotNone(halt_state, "the halt latch landed without HaltState")
        self.assertLessEqual(self._read_by_the_api(HaltStateLike), self._carried_by(halt_state))
        arm = DummyRobotArm()
        arm.connect()
        self.addCleanup(arm.disconnect)
        self.assertIsNone(_halted(arm))
        arm.halt("the contract test pressed halt now")
        halted = _halted(arm)
        self.assertIsNotNone(halted, "a latched arm reads as not halted: the latch drifted from what the console reads")
        assert halted is not None
        self.assertEqual("the contract test pressed halt now", halted.reason)
        arm.clear_halt()
        self.assertIsNone(_halted(arm))

    def test_the_hand_s_question_carries_what_the_browser_seam_reads(self) -> None:
        from api.jaws import JawAskingLike
        from src.robot.grippers import jaw_io

        if not hasattr(jaw_io, "JawAsking") and not hasattr(jaw_io.JawIOGripper, "answer_questions_with"):
            self.skipTest("the structured jaws question (library L10: JawAsking) has not landed")
        asking = getattr(jaw_io, "JawAsking", None)
        self.assertIsNotNone(asking, "the structured jaws question landed without JawAsking")
        self.assertLessEqual(self._read_by_the_api(JawAskingLike), self._carried_by(asking))

    def test_the_console_answers_every_hook_the_task_calls(self) -> None:
        from api.task_run import ConsoleTaskHooks

        if importlib.util.find_spec("src.robot.execution.task") is None:
            self.skipTest("the task run (library L1: run_task, TaskHooks) has not landed")
        from src.robot.execution import task

        hooks = getattr(task, "TaskHooks", None)
        self.assertIsNotNone(hooks, "the task run landed without TaskHooks")
        called = {name for name, member in vars(hooks).items() if not name.startswith("_") and callable(member)}
        answered = {name for name in dir(ConsoleTaskHooks) if callable(getattr(ConsoleTaskHooks, name))}
        self.assertLessEqual(called, answered)



# ---------------------------------------------------------------------------------------------------------------------
# The endpoint table of api/README.md, and the reader's vocabularies
# ---------------------------------------------------------------------------------------------------------------------

#: A route as the README's table writes it: a backticked method and a ``/v1`` path, its query string left out.
_README_ROUTE = re.compile(r"`(GET|POST|PUT|PATCH|DELETE|WS) (/v1/?[^`?\s]*)")


def _route_key(method: str, path: str) -> tuple[str, str]:
    """``(method, path)`` with every path parameter one placeholder: the table may say ``{id}`` for ``{run_id}``."""
    return method, re.sub(r"\{[^}]+\}", "{}", path.rstrip("/") or "/")


def _readme_routes(text: str) -> set[tuple[str, str]]:
    """Every route the ``## The endpoints`` section of ``api/README.md`` names, in its tables or its prose."""
    start = text.index("## The endpoints")
    end = text.find("\n## ", start + 1)
    section = text[start:] if end < 0 else text[start:end]
    return {_route_key(method, path) for method, path in _README_ROUTE.findall(section)}


def _app_routes() -> set[tuple[str, str]]:
    """Every ``/v1`` route the app serves: each HTTP method of each route, and each WebSocket as ``WS``."""
    from fastapi.routing import APIRoute, APIWebSocketRoute

    from api.app import create_app

    found: set[tuple[str, str]] = set()
    for route in create_app().routes:
        path = getattr(route, "path", "")
        if not path.startswith("/v1"):
            continue
        if isinstance(route, APIWebSocketRoute):
            found.add(_route_key("WS", path))
        elif isinstance(route, APIRoute):
            found |= {_route_key(method, path) for method in route.methods - {"HEAD", "OPTIONS"}}
    return found


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class TheReadmeTableIsTheRoutesTests(unittest.TestCase):
    """The endpoint table a reader of ``api/README.md`` reaches for is the app, both ways.

    A route the table names that the app does not serve sends a client to a 404; a route the app serves that the table
    leaves out is a door nobody documented, a moving one included. Both drift silently, since a README is read by
    people, not by the server.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = _API_README.read_text(encoding="utf-8")
        cls.table = _readme_routes(cls.text)
        cls.app = _app_routes()

    def test_the_table_is_read_at_all(self) -> None:
        """A pattern that stopped matching would pass both directions below over an empty set."""
        self.assertGreaterEqual(len(self.table), 50, sorted(self.table))
        self.assertIn(("POST", "/v1/task"), self.table)
        self.assertIn(("WS", "/v1/events"), self.table)

    def test_every_route_in_the_table_exists(self) -> None:
        self.assertEqual([], sorted(self.table - self.app), "api/README.md names routes the app does not serve")

    def test_every_route_of_the_app_is_in_the_table(self) -> None:
        self.assertEqual([], sorted(self.app - self.table), "the app serves /v1 routes api/README.md does not name")

    def test_every_route_of_the_contract_is_in_the_table(self) -> None:
        contract = {_route_key(method, path) for method, path, *_ in _CONTRACT}
        self.assertEqual([], sorted(contract - self.table))

    def test_a_route_left_out_of_the_table_is_seen(self) -> None:
        """The control: the same comparison over the README with the brake's row deleted fails."""
        cut = "\n".join(line for line in self.text.splitlines() if not line.startswith("| `POST /v1/cell/brake`"))
        self.assertIn(("POST", "/v1/cell/brake"), self.app - _readme_routes(cut))

    def test_every_refusal_code_is_in_the_refusal_table(self) -> None:
        """Every code of the catalog has its row under ``## What it refuses``, so no refusal reaches a client unread."""
        from api.codes import RefusalCode

        start = self.text.index("## What it refuses")
        section = self.text[start:self.text.index("\n## ", start + 1)]
        listed = set(re.findall(r"^\| ((?:`[a-z_]+`(?:, )?)+) \|", section, flags=re.M))
        named = {code for cell in listed for code in re.findall(r"`([a-z_]+)`", cell)}
        self.assertEqual([], sorted({code.value for code in RefusalCode} - named))


class TheReaderSpeaksTheCatalogTests(unittest.TestCase):
    """The command reader's own words are the console's catalog: its refusals are exactly the three
    ``vlm_*`` codes, its notes ``CommandNote`` in order, and its states those of ``CommandStatusOut`` and of the ready
    bar's commands light."""

    def test_the_reader_s_vocabularies_are_the_catalog_s(self) -> None:
        from api.codes import LIGHT_CODES, CommandNote, LightId, RefusalCode
        from api.schemas import CommandStatusOut
        from src.models.vlm.availability import ReaderRefusal, ReaderState
        from src.models.vlm.command import NOTE_ORDER

        self.assertEqual({code.value for code in RefusalCode if code.value.startswith("vlm_")},
                         set(typing.get_args(ReaderRefusal)))
        self.assertEqual(tuple(note.value for note in CommandNote), NOTE_ORDER)
        self.assertEqual(typing.get_args(ReaderState),
                         typing.get_args(CommandStatusOut.model_fields["state"].annotation))
        self.assertEqual(set(typing.get_args(ReaderState)), {code.value for code in LIGHT_CODES[LightId.COMMANDS]})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
