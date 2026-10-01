"""The planner judges refused samples again with the boxes the camera saw set aside, and puts its world back exactly.

The owner's Option 1 (after the guard fixes of 2026-09-30): cuRobo's world term reaches 25 to 29 mm past the UR10's
shoulder housing, so a bin the camera saw needed about 50 mm for a plan and 60 for a line where the exact guard alone
takes about 26. Where only the camera's boxes refuse a sample on the planner's world, the exact guard decides them, as it
decides the arm's own pairs (F1). To know that, the driver asks the planner once more, with every box the camera saw set
aside: ``check_js`` takes ``ignore_perceived``. This file holds that question on the CPU:

* the sidecar's helper (``_curobo_perceived``): it sets aside exactly the camera's boxes it is named, every one the
  planner holds, only those, and puts every flag back as it was, or says the world can no longer be vouched for;
* the sidecar's ``check_js`` branch, read as source (it runs only in the cuRobo environment): it sets aside only under
  the new key, says which in its reply, refuses while it may hold a carried part, and exits where its world cannot be
  put back; ``attach`` marks a carried part before it hangs one and ``detach`` clears it only once detached;
* the client: the key goes out only where asked, a request without it is the one it always was, and a reply that does
  not say it set aside exactly those boxes is no report;
* the glue: it asks only with the boxes its last refresh handed the planner, never a declared one's name, never while
  a part may be carried.

The stubs are real subprocesses, as in ``test_curobo_state_refusal.py``. The GPU half is
``scripts/curobo/probe_turned_boxes.py``.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

import numpy as np

import src.robot.safety.planning.curobo_client as client_module
from src.robot.safety.planning.curobo_client import CuroboPlanClient, CuroboUnavailableError

_ROOT = Path(__file__).resolve().parents[1]
_TIMEOUT_S = 0.8


# ---------------------------------------------------------------------------------------------------------------------
# The helper the sidecar sets the camera's boxes aside with
# ---------------------------------------------------------------------------------------------------------------------


class _Store:
    """A cuboid storage as the sidecar's adapter answers it: names in slot order, one enable flag each.

    ``stuck`` names flags it will not drop, ``deaf`` flags it silently will not put back (no error, the box stays out),
    and ``fail_restore`` raises on every put-back.
    """

    def __init__(self, flags: "dict[str, bool]", *, fail_restore: bool = False, stuck: "tuple[str, ...]" = (),
                 deaf: "tuple[str, ...]" = ()) -> None:
        self.flags = dict(flags)
        self.fail_restore = fail_restore
        self.stuck = stuck
        self.deaf = deaf
        self.set_calls: list[tuple[str, bool]] = []
        self._restoring = False

    def names(self) -> "list[str]":
        return list(self.flags)

    def enabled(self, name: str) -> bool:
        return self.flags[name]

    def set_enabled(self, name: str, enabled: bool) -> None:
        self.set_calls.append((name, enabled))
        if name in self.stuck and not enabled:
            return  # a flag the storage will not drop
        if name in self.deaf and enabled:
            return  # a flag the storage will not put back, and says nothing
        if self.fail_restore and enabled:
            raise RuntimeError("the storage refused the flag")
        self.flags[name] = bool(enabled)


_WORLD = {"support_plane": True, "bin_wall": True, "seen_00": True, "seen_01_bin": True, "probe_far": True}


class TheCamerasBoxesAreSetAsideAndPutBackExactlyTests(unittest.TestCase):
    def setUp(self) -> None:
        from src.robot.safety.planning import _curobo_perceived as perceived

        self.p = perceived

    def test_only_the_named_boxes_are_set_aside_and_every_flag_comes_back(self) -> None:
        """⭐ THE INVARIANT: inside, the camera's boxes are off and nothing else moved; after, every flag is as it was."""
        store = _Store(_WORLD)
        names = self.p.requested_names(["seen_01_bin", "seen_00"], "seen_")
        self.assertEqual(names, ("seen_00", "seen_01_bin"))
        with self.p.SetAside(store, names, prefix="seen_"):
            self.assertEqual({n for n, on in store.flags.items() if not on}, {"seen_00", "seen_01_bin"})
            self.assertTrue(store.flags["support_plane"] and store.flags["bin_wall"] and store.flags["probe_far"])
        self.assertEqual(store.flags, _WORLD)

    def test_a_flag_that_was_off_before_is_off_after(self) -> None:
        store = _Store(dict(_WORLD, seen_00=False))
        with self.p.SetAside(store, ("seen_00", "seen_01_bin"), prefix="seen_"):
            self.assertFalse(store.flags["seen_00"])
        self.assertFalse(store.flags["seen_00"], "put back as it was, not switched on")
        self.assertTrue(store.flags["seen_01_bin"])

    def test_the_world_comes_back_when_the_judgement_raises(self) -> None:
        store = _Store(_WORLD)
        with self.assertRaises(ZeroDivisionError), self.p.SetAside(store, ("seen_00", "seen_01_bin"), prefix="seen_"):
            raise ZeroDivisionError("the judgement failed")
        self.assertEqual(store.flags, _WORLD)

    def test_it_refuses_to_set_aside_what_it_was_not_asked_or_what_is_not_there(self) -> None:
        """⭐ FAIL CLOSED: the request names every camera box the planner holds and only those, or nothing is touched. A
        box left on is one the planner judges anyway; a name it does not hold is a request about another world."""
        for what, names in (("one camera box not named", ("seen_00",)),
                            ("a name the planner does not hold", ("seen_00", "seen_01_bin", "seen_02")),
                            ("nothing held is named", ("seen_07",))):
            with self.subTest(what=what):
                store = _Store(_WORLD)
                with self.assertRaises(self.p.PerceivedAsideError), self.p.SetAside(store, names, prefix="seen_"):
                    self.fail("judged a world nobody asked for")
                self.assertEqual(store.flags, _WORLD)
                self.assertEqual(store.set_calls, [], "nothing was touched")

    def test_a_request_that_names_no_camera_box_is_refused_before_anything(self) -> None:
        """Only boxes the camera world named are ever set aside: never the bench, a declared fixture or a stray name."""
        for value in (None, [], "seen_00", ["seen_00", "seen_00"], ["support_plane"], ["seen_"], ["seen_00", 3],
                      [""], {"seen_00": True}):
            with self.subTest(value=value), self.assertRaises(self.p.PerceivedAsideError):
                self.p.requested_names(value, "seen_")

    def test_a_flag_the_storage_will_not_drop_refuses_and_restores(self) -> None:
        store = _Store(_WORLD, stuck=("seen_01_bin",))
        with self.assertRaises(self.p.PerceivedAsideError), self.p.SetAside(store, ("seen_00", "seen_01_bin"),
                                                                               prefix="seen_"):
            self.fail("judged with a camera box still in the world")
        self.assertEqual(store.flags, _WORLD)

    def test_a_world_that_cannot_be_put_back_is_said_as_such(self) -> None:
        """⭐ A flag that does not come back is a world nobody can vouch for: the sidecar exits on this error."""
        store = _Store(_WORLD, fail_restore=True)
        with self.assertRaises(self.p.WorldRestoreError), self.p.SetAside(store, ("seen_00", "seen_01_bin"),
                                                                             prefix="seen_"):
            pass
        self.assertTrue(issubclass(self.p.WorldRestoreError, RuntimeError))
        self.assertFalse(issubclass(self.p.WorldRestoreError, self.p.PerceivedAsideError))

    def test_a_flag_the_storage_does_not_put_back_is_found_by_reading_it_back(self) -> None:
        """⭐ A storage that takes the put-back without a word and keeps the box out: only reading every flag back after
        the put-back finds it, and the sidecar exits on what it raises."""
        store = _Store(_WORLD, deaf=("seen_00",))
        with self.assertRaises(self.p.WorldRestoreError) as caught, self.p.SetAside(store, ("seen_00", "seen_01_bin"),
                                                                                    prefix="seen_"):
            pass
        self.assertIn("seen_00", str(caught.exception))
        self.assertNotIn("seen_01_bin", str(caught.exception), "the box that came back is not named")
        self.assertFalse(store.flags["seen_00"], "the storage really kept it out, and said nothing")
        self.assertIn(("seen_00", True), store.set_calls, "it was asked to put it back")

    def test_a_plain_check_asks_nothing_aside_and_touches_no_storage(self) -> None:
        """⭐ THE CONTROL the sidecar runs: a check_js request without the key asks nothing to be set aside, carried part or
        not, and the world it is judged in is the planner's whole world: the storage is not even built."""
        for carrying in (False, True):
            with self.subTest(carrying=carrying):
                self.assertIsNone(self.p.requested_aside({"cmd": "check_js", "joints": [[0.0] * 6]},
                                                         key="ignore_perceived", prefix="seen_", carrying=carrying))
        built: list[_Store] = []

        def store() -> _Store:
            built.append(_Store(_WORLD))
            return built[-1]

        with self.p.judged_world(None, store, prefix="seen_"):
            pass
        self.assertEqual(built, [], "a plain judgement builds no storage and switches nothing")

    def test_a_request_with_the_key_is_refused_while_a_part_may_be_carried(self) -> None:
        """⭐ The sidecar's own word on a carried part: with the key, the names come back sorted where no part may be held,
        and nothing is set aside while one may be; a malformed list is refused either way."""
        request = {"cmd": "check_js", "ignore_perceived": ["seen_01_bin", "seen_00"]}
        self.assertEqual(self.p.requested_aside(request, key="ignore_perceived", prefix="seen_", carrying=False),
                         ("seen_00", "seen_01_bin"))
        with self.assertRaises(self.p.PerceivedAsideError) as caught:
            self.p.requested_aside(request, key="ignore_perceived", prefix="seen_", carrying=True)
        self.assertIn("carried part", str(caught.exception))
        for value in (None, [], ["support_plane"], "seen_00"):
            with self.subTest(value=value), self.assertRaises(self.p.PerceivedAsideError):
                self.p.requested_aside({"ignore_perceived": value}, key="ignore_perceived", prefix="seen_",
                                       carrying=False)
        store = _Store(_WORLD)
        with self.p.judged_world(("seen_00", "seen_01_bin"), lambda: store, prefix="seen_"):
            self.assertEqual({n for n, on in store.flags.items() if not on}, {"seen_00", "seen_01_bin"})
        self.assertEqual(store.flags, _WORLD)

    def test_the_module_stands_alone_as_the_sidecar_imports_it(self) -> None:
        """The sidecar runs on python 3.10 in the cuRobo environment and cannot import Willy: it loads this module as a
        plain sibling of the server script. So the module reaches for the standard library alone, and loads by path."""
        import importlib.util

        path = _ROOT / "src" / "robot" / "safety" / "planning" / "_curobo_perceived.py"
        source = path.read_text(encoding="utf-8")
        for reach in ("from src", "import src", "from ."):
            self.assertNotIn(reach, source)
        spec = importlib.util.spec_from_file_location("_curobo_perceived_as_a_sibling", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in ("SetAside", "requested_names", "requested_aside", "judged_world", "PerceivedAsideError",
                     "WorldRestoreError"):
            self.assertTrue(hasattr(module, name), name)

    def test_the_prefix_is_the_one_the_camera_world_names_its_boxes_with(self) -> None:
        """The sidecar tells the camera's boxes by this prefix; the camera world names every box it builds with it."""
        from src.robot.safety.planning._curobo_protocol import PERCEIVED_PREFIX
        from src.robot.safety.planning.perceived import WorldBuildTuning

        self.assertEqual(WorldBuildTuning().name_prefix, PERCEIVED_PREFIX)
        self.assertEqual(PERCEIVED_PREFIX, "seen_")


# ---------------------------------------------------------------------------------------------------------------------
# The sidecar, read as source
# ---------------------------------------------------------------------------------------------------------------------


def _server_source() -> str:
    return (_ROOT / "src" / "robot" / "safety" / "planning" / "curobo_planner_server.py").read_text(encoding="utf-8")


def _block(source: str, start: str, end: str) -> str:
    begin = source.find(start)
    stop = source.find(end, begin + 1)
    return source[begin:stop] if 0 <= begin < stop else ""


def _inside_block(source: str, header: str, at: int) -> bool:
    """Whether position ``at`` lies inside the block the last ``header`` line before it opens (indented deeper)."""
    start = source.rfind(header, 0, at)
    if start < 0:
        return False
    line_start = source.rfind("\n", 0, start) + 1
    indent = start - line_start
    body = source[source.find("\n", start) + 1:source.rfind("\n", 0, at) + 1]
    return all(len(line) - len(line.lstrip(" ")) > indent for line in body.splitlines() if line.strip()) and (
        at - (source.rfind("\n", 0, at) + 1) > indent)


#: How the check_js branch asks what to set aside: the CPU-tested helper, with the key, the prefix and this sidecar's own
#: carried-part mark (``_curobo_perceived.requested_aside``).
_ASKED = "requested_aside(req, key=IGNORE_PERCEIVED_KEY, prefix=PERCEIVED_PREFIX, carrying=_CARRYING)"


def _sets_aside_only_when_asked(check: str) -> bool:
    """True when the check_js branch asks the helper what to set aside with its own carried-part mark, judges inside the
    world the helper opens (``judged_world``, which opens nothing for a plain check), never opens a set-aside on its own
    terms, echoes the names once, after that world closed and only inside ``if _aside is not None:``, and a world it
    cannot put back ends the sidecar."""
    asked = check.find(_ASKED)
    opened = check.find("with judged_world(_aside, ")
    judged = check.find("_terms(req[\"joints\"]")
    echo = check.find("_reply[PERCEIVED_IGNORED_KEY]")
    return (0 <= asked < opened < judged and _inside_block(check, "with judged_world(", judged)
            and check.count("_reply[PERCEIVED_IGNORED_KEY]") == 1 and echo > judged
            and not _inside_block(check, "with judged_world(", echo)
            and _inside_block(check, "if _aside is not None:", echo) and "SetAside(" not in check
            and "WorldRestoreError" in check and "sys.exit(" in check[check.find("WorldRestoreError"):])


def _marks_the_carried_part(source: str) -> bool:
    """True when attach marks a carried part before it hangs one, and detach clears the mark only after it detached."""
    attach = _block(source, 'if cmd == "attach":', 'if cmd == "detach":')
    detach = _block(source, 'if cmd == "detach":', 'if cmd == "set_voxels":')
    mark, hang = attach.find("_CARRYING = True"), attach.find("attachment_manager.attach(")
    done, clear = detach.find("attachment_manager.detach("), detach.find("_CARRYING = False")
    return 0 <= mark < hang and 0 <= done < clear and detach.find("except", done) > clear


class TheSidecarSetsAsideOnlyWhenAskedTests(unittest.TestCase):
    """The sidecar runs only in the cuRobo environment, so its half is read as source; the GPU run is
    ``scripts/curobo/probe_turned_boxes.py``."""

    def test_the_check_sets_the_cameras_boxes_aside_only_under_its_key(self) -> None:
        check = _block(_server_source(), 'if cmd == "check_js":', 'if cmd == "explain_js":')
        self.assertTrue(check, "the sidecar has no check_js branch")
        self.assertTrue(_sets_aside_only_when_asked(check))
        self.assertIn("PERCEIVED_PREFIX", check)
        self.assertIn("_PlannerCuboids(_planner.scene_collision_checker)", check)

    def test_the_carried_part_is_marked_before_it_hangs_and_cleared_once_it_is_off(self) -> None:
        source = _server_source()
        self.assertTrue(_marks_the_carried_part(source))
        self.assertIn("_CARRYING = False", source[:source.find("for _line in sys.stdin:")],
                      "a sidecar starts holding no part")

    def test_the_planners_own_switch_sets_the_box_aside(self) -> None:
        """cuRobo's own ``enable_obstacle`` switches a box, and its flag is read back off cuRobo's cuboid storage."""
        source = _server_source()
        self.assertIn(".enable_obstacle(", source)
        self.assertIn("scene_collision_checker", source[source.find("class _PlannerCuboids"):])

    def test_the_scans_can_fail(self) -> None:
        """⭐ THE CONTROL: a branch that echoes whatever was asked, echoes before the world is back, judges outside the
        world the helper opens, opens a set-aside on its own terms, tells the helper no part is carried, or never exits on
        a world it cannot put back reads False; and an attach that hangs the part before marking it, too."""
        good = (f'_aside = {_ASKED}\n'
                'with judged_world(_aside, lambda: store, prefix=PERCEIVED_PREFIX):\n'
                '    _bound = _terms(req["joints"], c)\n'
                'if _aside is not None:\n    _reply[PERCEIVED_IGNORED_KEY] = list(_aside)\n'
                'except WorldRestoreError:\n    sys.exit(3)\n')
        self.assertTrue(_sets_aside_only_when_asked(good))
        bad = {
            "echoes what nobody asked": good.replace("if _aside is not None:\n", "if True:\n"),
            "judges on in a world it cannot vouch for": good.replace("sys.exit(3)", "pass"),
            "sets aside while a part may be carried": good.replace("carrying=_CARRYING", "carrying=False"),
            "judges outside the world it opened": good.replace('    _bound = _terms(req["joints"], c)\n', "    pass\n")
            + '_bound = _terms(req["joints"], c)\n',
            "echoes before the world is back": good.replace(
                'if _aside is not None:\n    _reply[PERCEIVED_IGNORED_KEY] = list(_aside)\n',
                '    if _aside is not None:\n        _reply[PERCEIVED_IGNORED_KEY] = list(_aside)\n'),
            "opens a set-aside on its own terms": good.replace(
                "with judged_world(_aside, lambda: store, prefix=PERCEIVED_PREFIX):",
                "with judged_world(_aside, lambda: store, prefix=PERCEIVED_PREFIX), SetAside(store, _aside):"),
        }
        for what, branch in bad.items():
            with self.subTest(what=what):
                self.assertFalse(_sets_aside_only_when_asked(branch))
        late = ('if cmd == "attach":\n    _planner.attachment_manager.attach(x)\n    _CARRYING = True\n'
                'if cmd == "detach":\n    _planner.attachment_manager.detach(y)\n    _CARRYING = False\n'
                '    except E:\nif cmd == "set_voxels":')
        self.assertFalse(_marks_the_carried_part(late))
        self.assertTrue(_marks_the_carried_part(late.replace(
            "_planner.attachment_manager.attach(x)\n    _CARRYING = True",
            "_CARRYING = True\n    _planner.attachment_manager.attach(x)")))


# ---------------------------------------------------------------------------------------------------------------------
# The client, against stub sidecars
# ---------------------------------------------------------------------------------------------------------------------

#: Answers check_js from the rows it receives, as the sidecar does: a row whose first joint reaches 1.0 rad is a self
#: collision of forearm_link and wrist_2_link, one whose third joint reaches 1.0 rad meets the world, but only while the
#: camera's boxes are in it, and one whose fourth reaches 1.0 rad meets the bench, which is never set aside. It echoes
#: the keys it was sent, and the names it set aside under ``perceived_ignored``.
_SEES = textwrap.dedent("""
    import json, sys
    sys.stdout.write(json.dumps({"status": "ready", "joint_names": ["j%d" % i for i in range(6)], "dt": 0.02}) + chr(10))
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "shutdown":
            break
        aside = req.get("ignore_perceived")
        rows = req["joints"]
        terms = [(True, row[0] < 1.0, (row[2] < 1.0 or aside is not None) and row[3] < 1.0) for row in rows]
        bad = [i for i, (bound, self_ok, world) in enumerate(terms) if not (bound and self_ok and world)]
        reply = {"success": True, "valid": not bad, "first_invalid": bad[0] if bad else None, "checked": len(rows),
                 "clearance_m": float(req.get("clearance_m", 0.0)), "asked": sorted(req)}
        if req.get("report_refused"):
            named = bool(req.get("name_pairs", True))
            reply["refused"] = [{
                "index": i, "bound_ok": terms[i][0], "self_ok": terms[i][1], "world_ok": terms[i][2],
                "pairs": ([["forearm_link", "wrist_2_link", 1.21, -2.79]] if not terms[i][1] else []) if named else None,
            } for i in bad]
            reply["pairs_named"] = named
        if aside is not None:
            reply["perceived_ignored"] = list(aside)
        if req.get("id") is not None:
            reply["id"] = req["id"]
        sys.stdout.write(json.dumps(reply) + chr(10)); sys.stdout.flush()
""")

_CLEAR = [0.0] * 6
_IN_BAND = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_BY_THE_BIN = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
_ON_THE_BENCH = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
_READY: dict[str, Any] = {"status": "ready", "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"], "dt": 0.02}


def _stub(source: str) -> str:
    path = Path(tempfile.mkdtemp()) / "sidecar.py"
    path.write_text(source, encoding="utf-8")
    return str(path)


def _replying(replies: "list[dict[str, Any]]") -> str:
    """A stub that answers every request from ``replies`` in turn, repeating the last, stamped with the request's id."""
    return _stub(textwrap.dedent(f"""
        import json, sys
        READY = json.loads({json.dumps(_READY)!r})
        REPLIES = json.loads({json.dumps(replies)!r})
        sys.stdout.write(json.dumps(READY) + chr(10)); sys.stdout.flush()
        n = 0
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            if req.get("cmd") == "shutdown":
                break
            reply = dict(REPLIES[min(n, len(REPLIES) - 1)])
            n += 1
            if req.get("id") is not None:
                reply["id"] = req["id"]
            sys.stdout.write(json.dumps(reply) + chr(10)); sys.stdout.flush()
    """))


class TheClientAsksOnlyWhereItIsToldTests(unittest.TestCase):
    def _client(self, script: "str | None" = None) -> CuroboPlanClient:
        saved = client_module._PLAN_TIMEOUT_S  # noqa: SLF001
        client_module._PLAN_TIMEOUT_S = _TIMEOUT_S  # noqa: SLF001
        self.addCleanup(setattr, client_module, "_PLAN_TIMEOUT_S", saved)
        client = CuroboPlanClient(python_path=sys.executable, server_script=script or _stub(_SEES),
                                  robot_config="unused-by-the-stub", scene_config=None)
        self.addCleanup(client.close)
        return client

    def _asked(self, client: CuroboPlanClient, request: "dict[str, Any]") -> "list[str]":
        want = client._send(request)  # noqa: SLF001 (reads the stub's echo of the keys)
        reply = client._recv(_TIMEOUT_S * 5, want=want)  # noqa: SLF001
        assert reply is not None
        return reply["asked"]

    def test_a_plain_check_and_a_plain_report_are_the_requests_they_always_were(self) -> None:
        """⭐ THE CONTROL: no caller that did not ask sends the key, so every other request and reply is unchanged."""
        client = self._client()
        verdict = client.check_joints([_CLEAR, _BY_THE_BIN])
        self.assertEqual(verdict.first_invalid, 1)
        judged = client.judge_joints([_BY_THE_BIN])
        self.assertEqual([(r.index, r.world_ok) for r in judged.refused], [(0, False)])
        self.assertEqual(judged.perceived_ignored, ())
        sent: list[dict[str, Any]] = []
        real_send = client._send  # noqa: SLF001

        def recording(obj: "dict[str, Any]") -> int:
            sent.append(dict(obj))
            return real_send(obj)

        client._send = recording  # type: ignore[method-assign]  # noqa: SLF001
        client.check_joints([_CLEAR])
        client.judge_joints([_CLEAR], clearance_mm=10.0)
        self.assertEqual([sorted(request) for request in sent],
                         [["cmd", "joints"], ["clearance_m", "cmd", "joints", "name_pairs", "report_refused"]])

    def test_the_cameras_boxes_go_out_by_name_and_the_reply_says_it_set_them_aside(self) -> None:
        client = self._client()
        judged = client.judge_joints([_BY_THE_BIN, _IN_BAND, _ON_THE_BENCH], clearance_mm=10.0, name_pairs=False,
                                     ignore_perceived=("seen_01_bin", "seen_00"))
        self.assertEqual(judged.perceived_ignored, ("seen_00", "seen_01_bin"))
        self.assertEqual([(r.index, r.self_ok, r.world_ok) for r in judged.refused],
                         [(1, False, True), (2, True, False)], "the bin is gone; the band and the bench are not")
        keys = self._asked(client, {"cmd": "check_js", "joints": [_CLEAR], "report_refused": True,
                                    "ignore_perceived": ["seen_00"]})
        self.assertIn("ignore_perceived", keys)

    def test_every_batch_of_a_long_path_sets_them_aside(self) -> None:
        rows = [list(_CLEAR) for _ in range(1500)]
        rows[1200] = list(_BY_THE_BIN)
        rows[1300] = list(_ON_THE_BENCH)
        judged = self._client().judge_joints(rows, ignore_perceived=("seen_00",))
        self.assertEqual([row.index for row in judged.refused], [1300])
        self.assertEqual(judged.checked, 1500)

    def test_names_that_are_no_camera_boxes_are_refused_before_anything_is_sent(self) -> None:
        client = self._client(_replying([{"success": False, "planner_error": True, "reason": "never asked"}]))
        for names in ((), ("support_plane",), ("seen_00", "seen_00"), ("seen_",), ("",)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                client.judge_joints([_CLEAR], ignore_perceived=names)
        self.assertIsNone(client._proc, "nothing was started for a question that is not one")  # noqa: SLF001

    def test_a_reply_that_does_not_say_exactly_what_it_set_aside_is_no_report(self) -> None:
        """⭐ FAIL CLOSED: a sidecar older than the key judges with the camera's boxes in and says nothing; one that set
        aside others, or set aside boxes nobody asked it to, answered another question. None of them is read."""
        base = {"success": True, "valid": False, "first_invalid": 0, "checked": 1, "clearance_m": 0.0,
                "pairs_named": True, "refused": [{"index": 0, "bound_ok": True, "self_ok": True, "world_ok": False,
                                                  "pairs": []}]}
        asked_cases = {
            "an old sidecar": base,
            "other boxes": dict(base, perceived_ignored=["seen_00"]),
            "not a list": dict(base, perceived_ignored="seen_00 seen_01"),
            "a failed call": {"success": False, "planner_error": True,
                              "reason": "PerceivedAsideError: the planner holds the boxes the camera saw ..."},
        }
        for what, reply in asked_cases.items():
            with self.subTest(what=what):
                client = self._client(_replying([reply]))
                with self.assertRaises(CuroboUnavailableError) as caught:
                    client.judge_joints([_BY_THE_BIN], ignore_perceived=("seen_00", "seen_01"))
                if what == "an old sidecar":
                    self.assertIn("restart", str(caught.exception))
        client = self._client(_replying([dict(base, perceived_ignored=["seen_00"])]))
        with self.assertRaises(CuroboUnavailableError):
            client.judge_joints([_BY_THE_BIN])


# ---------------------------------------------------------------------------------------------------------------------
# The glue: it asks only with what its last refresh handed the planner
# ---------------------------------------------------------------------------------------------------------------------


class _Judging:
    """A planner client that keeps every world it was handed and every report it was asked for."""

    def __init__(self) -> None:
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                            "wrist_2_joint", "wrist_3_joint"]
        self.worlds: list[list[dict[str, Any]]] = []
        self.judged: list[dict[str, Any]] = []
        self.last_refusal = None
        #: How many boxes of a world it is sent the planner leaves unregistered: a refused refresh where not 0.
        self.short = 0

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def set_world(self, boxes: list, meshes: "list | None" = None) -> int:
        self.worlds.append([dict(box) for box in boxes])
        return len(boxes) + len(meshes or ()) - self.short

    def attach_payload(self, joints: Any, dims_m: Any, pose: Any) -> bool:
        return True

    def detach_payload(self) -> bool:
        return True

    def judge_joints(self, configs: Any, **asked: Any) -> Any:
        from src.robot.safety.planning.curobo_client import PathJudgement

        self.judged.append(dict(asked))
        return PathJudgement(checked=len(configs), clearance_mm=float(asked.get("clearance_mm", 0.0)), refused=(),
                             pairs_named=bool(asked.get("name_pairs", True)),
                             perceived_ignored=tuple(asked.get("ignore_perceived") or ()))


class _Conn:
    is_connected = True

    def get_joint_positions(self) -> "list[float]":
        return [0.0, -math.pi / 3.0, 4.0 * math.pi / 9.0, -11.0 * math.pi / 18.0, -math.pi / 2.0, 0.0]


def _glue(*, fixtures: "tuple[dict[str, Any], ...]" = ()) -> "tuple[Any, _Judging, Any]":
    """The UR glue with a live world of a bin beside the base, and the client it asks."""
    from src.robot.drivers.ur.curobo_motion import CuroboUrPlanner
    from src.robot.safety.planning.live_world import CameraView, DepthSnapshot, LivePlannerWorld
    from src.robot.safety.planning.perceived import SelfEnvelope, WorldBuildTuning
    from src.robot.safety.planning.self_envelope import arm_capsules, yawed_link_transforms_mm
    from src.robot.safety.planning.world import planner_cuboid
    from tests._seen_scenes import K, LIMITS, looking_down, open_bin, render

    camera = looking_down(0.0, -400.0)
    depth = render(camera, open_bin((0.0, -400.0), (300.0, 200.0), 40.0, yaw_deg=30.0))

    class _Camera:
        def grab_surface_depth(self) -> DepthSnapshot:
            import time

            return DepthSnapshot(depth_mm=depth, intrinsics=K, timestamp=time.time())

    declared = (planner_cuboid("support_plane", (0.0, 0.0, -25.0), (2000.0, 2000.0, 50.0)), *fixtures)
    world = LivePlannerWorld(cameras=(CameraView(name="look", depth_source=_Camera(), camera_to_base=camera),),
                             declared=declared, limits=LIMITS, tuning=WorldBuildTuning(pixel_stride=2),
                             max_age_ms=60000.0)
    joints = np.asarray(_Conn().get_joint_positions())
    frames = yawed_link_transforms_mm("ur10", joints, 0.0)
    capsules = arm_capsules("ur10")
    assert frames is not None and capsules is not None
    envelope = SelfEnvelope(frames_mm=tuple(frames), capsules=capsules)
    client = _Judging()
    told: list[Any] = []
    glue = CuroboUrPlanner(_Conn(), client_factory=lambda: client, world_cuboids=list(declared), live_world=world,
                           self_envelope=lambda: envelope, on_perceived_obstacles=told.append)
    return glue, client, told


class TheGlueAsksOnlyWithWhatItsRefreshHandedThePlannerTests(unittest.TestCase):
    def test_the_glue_says_which_camera_boxes_the_planner_holds(self) -> None:
        glue, client, told = _glue()
        self.assertEqual(glue.perceived_in_world(), (), "a planner nobody refreshed holds no camera box")
        glue.refresh_world(near_point_mm=(600.0, 0.0, 300.0))
        held = glue.perceived_in_world()
        assert held is not None
        self.assertTrue(held and all(name.startswith("seen_") for name in held))
        self.assertEqual(sorted(held), sorted(box.name for box in told[-1]), "the very boxes the guard was handed")
        self.assertEqual(sorted(held), sorted(box["name"] for box in client.worlds[-1] if box["name"].startswith("seen_")))
        glue.set_world([{"name": "seen_00", "dims_m": [0.1] * 3, "pose": [0.0, -0.4, 0.05, 1.0, 0.0, 0.0, 0.0]}])
        self.assertIsNone(glue.perceived_in_world(), "a world set by hand is no refresh's: nobody can say")

    def test_it_asks_with_the_cameras_boxes_its_refresh_handed_the_planner(self) -> None:
        glue, client, _ = _glue()
        glue.refresh_world(near_point_mm=(600.0, 0.0, 300.0))
        held = glue.perceived_in_world()
        assert held
        judged = glue.judge_joint_path([[0.0] * 6], clearance_mm=10.0, name_pairs=False, ignore_perceived=held)
        self.assertEqual(sorted(judged.perceived_ignored), sorted(held))
        self.assertEqual(sorted(client.judged[-1]["ignore_perceived"]), sorted(held))
        glue.judge_joint_path([[0.0] * 6], clearance_mm=0.0)
        self.assertNotIn("ignore_perceived", client.judged[-1], "a report nobody asked to set aside is unchanged")

    def test_it_never_asks_with_other_names_a_declared_name_or_while_a_part_may_be_carried(self) -> None:
        """⭐ FAIL CLOSED before the sidecar is asked: the names the guard holds and the boxes the planner was handed by
        the last refresh are one list; a declared fixture is never set aside, whatever it is called; and nothing is set
        aside while the planner may hold a carried part, which only it holds."""
        glue, client, _ = _glue()
        glue.refresh_world(near_point_mm=(600.0, 0.0, 300.0))
        held = tuple(glue.perceived_in_world() or ())
        asked = len(client.judged)
        for names in (held[:-1], (*held, "seen_99")):
            with self.subTest(names=names), self.assertRaises(CuroboUnavailableError):
                glue.judge_joint_path([[0.0] * 6], ignore_perceived=names)
        self.assertTrue(glue.attach_payload([0.0] * 6, (50.0, 50.0, 80.0), (0.0, 0.0, 200.0)))
        self.assertTrue(glue.carries_part)
        with self.assertRaises(CuroboUnavailableError) as caught:
            glue.judge_joint_path([[0.0] * 6], ignore_perceived=held)
        self.assertIn("carried", str(caught.exception))
        self.assertTrue(glue.detach_payload())
        self.assertFalse(glue.carries_part)
        self.assertEqual(len(client.judged), asked, "the sidecar was asked none of them")

        named_like_it = {"name": held[0], "dims_m": [0.1] * 3, "pose": [0.0, -0.4, 0.05, 1.0, 0.0, 0.0, 0.0]}
        glue, client, _ = _glue(fixtures=(named_like_it,))
        glue.refresh_world(near_point_mm=(600.0, 0.0, 300.0))
        with self.assertRaises(CuroboUnavailableError) as caught:
            glue.judge_joint_path([[0.0] * 6], ignore_perceived=tuple(glue.perceived_in_world() or ()) or held)
        self.assertIn("declared", str(caught.exception))

    def test_nobody_can_say_what_it_holds_once_its_world_is_no_confirmed_refresh(self) -> None:
        """⭐ FAIL CLOSED, the glue's own word: after a world set by hand, a refresh the planner refused, a refresh whose
        camera could not vouch for the cell, or a closed sidecar, the glue cannot say which of the camera's boxes the
        planner holds. It says so (``None``), and a set-aside asked with the very names its last good refresh handed over
        is refused before the sidecar hears of it."""
        from unittest import mock

        import src.robot.drivers.ur.curobo_motion as glue_module
        from src.robot.core.errors import CameraWorldUnavailable

        def by_hand(glue: Any, client: _Judging, held: "tuple[str, ...]") -> None:
            glue.set_world([{"name": name, "dims_m": [0.1] * 3, "pose": [0.0, -0.4, 0.05, 1.0, 0.0, 0.0, 0.0]}
                            for name in held])

        def refused(glue: Any, client: _Judging, held: "tuple[str, ...]") -> None:
            client.short = 1
            with self.assertRaises(CuroboUnavailableError):
                glue.refresh_world(near_point_mm=(600.0, 0.0, 300.0))
            client.short = 0

        def blind(glue: Any, client: _Judging, held: "tuple[str, ...]") -> None:
            fault = CameraWorldUnavailable(camera="look", verdict="no_frame", attempts=2, reason="nothing answered")
            with mock.patch.object(glue_module, "refresh_planner_world", side_effect=fault), \
                    self.assertRaises(CameraWorldUnavailable):
                glue.refresh_world(near_point_mm=(600.0, 0.0, 300.0))

        def closed(glue: Any, client: _Judging, held: "tuple[str, ...]") -> None:
            glue.close()

        for what, after in (("a world set by hand", by_hand), ("a refused refresh", refused),
                            ("a camera that could not vouch", blind), ("a closed sidecar", closed)):
            with self.subTest(what=what):
                glue, client, _ = _glue()
                glue.refresh_world(near_point_mm=(600.0, 0.0, 300.0))
                held = tuple(glue.perceived_in_world() or ())
                self.assertTrue(held, "the refresh before handed the planner the camera's boxes")
                after(glue, client, held)
                self.assertIsNone(glue.perceived_in_world())
                asked = len(client.judged)
                with self.assertRaises(CuroboUnavailableError) as caught:
                    glue.judge_joint_path([[0.0] * 6], ignore_perceived=held)
                self.assertIn("no refresh it confirmed", str(caught.exception))
                self.assertEqual(len(client.judged), asked, "the sidecar was not asked")
                glue.judge_joint_path([[0.0] * 6])
                self.assertEqual(len(client.judged), asked + 1, "a plain report is asked as it always was")


if __name__ == "__main__":
    unittest.main()
