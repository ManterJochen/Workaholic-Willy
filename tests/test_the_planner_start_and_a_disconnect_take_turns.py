"""A disconnect during a planner start waits for it, and closes the sidecar it started: none is orphaned.

The console starts cuRobo after every Connect (build plan 0.19), a run of its own that takes about a minute and moves
nothing, and a person can press Disconnect meanwhile. The sidecar a start spawns is kept by the planner glue only once
it has started and passed its descriptor check (``CuroboUrPlanner._client_or_start``), so a disconnect that ran in the
middle found nothing to close, dropped the glue, and the start then finished into a glue nobody held: a sidecar process
and its GPU memory, orphaned until the server exits. A lock now guards building, starting and closing the planner, on
the arm (``start_planner``, ``stop_planner``, ``disconnect``) and inside the glue (a start a move made lazily), so the
closing side waits for the start and then closes what it started.

``planner_state`` says where the planner is, for the ready bar: ``not_used`` on an arm that plans nothing, ``off``,
``starting``, ``ready``. It is read on every poll of ``GET /v1/cell``, so it never takes that lock and never raises.

A disconnect and a ``stop_planner`` retire the planner glue as well as closing it: a move still judging its route holds
the glue it took at its start, and every check, plan and refresh of that route starts the sidecar again where none is
running. The retired glue refuses that (``CuroboUnavailableError``) rather than spawning a sidecar the arm no longer
holds, and a halted arm keeps no fresh glue it builds for a move either, so a move in flight across the console's
Disconnect (which halts first) starts nothing at all. One level down the same holds for the client: ``CuroboPlanClient``
starts its sidecar again on any call once closed, so the glue hands out its client kept behind its start lock, a retire
waits for the call in flight, and every call after it is refused. A plain ``close()`` stays what it was: the next start
on that glue starts a fresh, empty sidecar. ``start_planner`` builds and starts one whenever a person asks, at a desk and
halted included.
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any

from src.robot.drivers.ur.curobo_motion import UR_ARM_JOINT_NAMES, CuroboUrPlanner
from src.robot.safety.planning import CuroboUnavailableError, JointCheckVerdict
from tests._halt_fakes import RecordingRtde, ur_arm_on

_HERE = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]


class _SlowClient:
    """A cuRobo client whose start takes as long as the test says, counting its starts and its closes."""

    def __init__(self, *, fail: bool = False) -> None:
        self.release = threading.Event()
        self.starting = threading.Event()
        self.started = 0
        self.closed = 0
        self.fail = fail
        self.joint_names = list(UR_ARM_JOINT_NAMES)

    def start(self) -> None:
        self.starting.set()
        assert self.release.wait(5.0), "the test never let the start finish"
        if self.fail:
            raise CuroboUnavailableError("the sidecar did not come up")
        self.started += 1

    def close(self) -> None:
        self.closed += 1

    def check_joints(self, configs: list[list[float]], **_kwargs: Any) -> JointCheckVerdict:
        return JointCheckVerdict(valid=True, first_invalid=None, checked=len(configs), reason="clear")


class _Sidecars:
    """A client factory that writes down every client it made; each starts at once and counts its starts and closes."""

    def __init__(self) -> None:
        self.made: list[_SlowClient] = []

    def __call__(self) -> _SlowClient:
        client = _SlowClient()
        client.release.set()
        self.made.append(client)
        return client

    def orphaned(self) -> int:
        """Sidecars started and never closed."""
        return sum(1 for client in self.made if client.started and not client.closed)


class _RestartingClient:
    """A planner client that starts its sidecar again on any call once it was closed, as ``CuroboPlanClient`` does
    (``if self._proc is None: self.start()``), counting its starts and closes."""

    def __init__(self) -> None:
        self.joint_names = list(UR_ARM_JOINT_NAMES)
        self.running = False
        self.starts = 0
        self.closes = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def start(self) -> None:
        if not self.running:
            self.running = True
            self.starts += 1

    def close(self) -> None:
        if self.running:
            self.running = False
            self.closes += 1

    def check_joints(self, configs: list[list[float]], **_kwargs: Any) -> JointCheckVerdict:
        if not self.running:
            self.start()
        self.entered.set()
        assert self.release.wait(5.0), "the test never let the check finish"
        return JointCheckVerdict(valid=True, first_invalid=None, checked=len(configs), reason="clear")


def _arm(client: _SlowClient) -> Any:
    arm = ur_arm_on(RecordingRtde(), "curobo")
    arm._curobo_ur = CuroboUrPlanner(arm._conn, client_factory=lambda: client)
    return arm


def _arm_building_on(sidecars: _Sidecars) -> Any:
    """A cuRobo arm that builds its own planner glue on ``sidecars``; they report no descriptor, so none is checked."""
    arm = ur_arm_on(RecordingRtde(), "curobo")
    arm._curobo_ur = None
    arm._curobo_client_factory = sidecars
    arm._descriptor_check = lambda: None
    return arm


def _thread(target: Any) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


class TheyTakeTurnsTests(unittest.TestCase):
    def test_a_disconnect_during_a_planner_start_waits_for_it_and_closes_what_it_started(self) -> None:
        """Red before: the disconnect returned at once with nothing to close, and the started sidecar was orphaned."""
        client = _SlowClient()
        arm = _arm(client)
        starting = _thread(arm.start_planner)
        self.assertTrue(client.starting.wait(2.0))
        self.assertEqual("starting", arm.planner_state)
        closing = _thread(arm.disconnect)
        time.sleep(0.2)
        self.assertTrue(closing.is_alive(), "the disconnect did not wait for the planner start")
        client.release.set()
        starting.join(5.0)
        closing.join(5.0)
        self.assertFalse(starting.is_alive() or closing.is_alive())
        self.assertEqual((1, 1), (client.started, client.closed), "the sidecar the start spawned was left running")
        self.assertEqual("off", arm.planner_state)
        self.assertFalse(arm.is_connected)

    def test_a_disconnect_during_a_start_a_move_made_waits_for_it_too(self) -> None:
        """A move starts the planner lazily, below the arm's lock: the glue's own lock makes the close wait."""
        client = _SlowClient()
        arm = _arm(client)
        planner = arm._curobo_ur
        starting = _thread(planner.start)
        self.assertTrue(client.starting.wait(2.0))
        self.assertEqual("starting", planner.state)
        closing = _thread(arm.disconnect)
        time.sleep(0.2)
        self.assertTrue(closing.is_alive(), "the close did not wait for the start a move made")
        client.release.set()
        starting.join(5.0)
        closing.join(5.0)
        self.assertEqual((1, 1), (client.started, client.closed))
        self.assertEqual("off", planner.state)

    def test_stop_planner_during_a_start_waits_and_closes(self) -> None:
        client = _SlowClient()
        arm = _arm(client)
        starting = _thread(arm.start_planner)
        self.assertTrue(client.starting.wait(2.0))
        stopping = _thread(arm.stop_planner)
        time.sleep(0.2)
        self.assertTrue(stopping.is_alive())
        client.release.set()
        starting.join(5.0)
        stopping.join(5.0)
        self.assertEqual((1, 1), (client.started, client.closed))
        self.assertEqual("off", arm.planner_state)
        self.assertTrue(arm.is_connected, "stopping the planner leaves the connection as it was")


class PlannerStateTests(unittest.TestCase):
    def test_an_arm_that_plans_nothing_says_not_used(self) -> None:
        self.assertEqual("not_used", ur_arm_on(RecordingRtde(), "ik").planner_state)

    def test_off_starting_ready_and_off_again(self) -> None:
        """Red before: the arm had no planner_state, so the ready bar could not tell a planner starting from none."""
        client = _SlowClient()
        arm = _arm(client)
        self.assertEqual("off", arm.planner_state)
        starting = _thread(arm.start_planner)
        self.assertTrue(client.starting.wait(2.0))
        read_at = time.monotonic()
        self.assertEqual("starting", arm.planner_state)
        self.assertLess(time.monotonic() - read_at, 0.1, "reading the state waited for the start")
        client.release.set()
        starting.join(5.0)
        self.assertEqual("ready", arm.planner_state)
        arm.stop_planner()
        self.assertEqual("off", arm.planner_state)

    def test_a_start_that_fails_leaves_the_planner_off(self) -> None:
        client = _SlowClient(fail=True)
        arm = _arm(client)
        client.release.set()
        with self.assertRaises(CuroboUnavailableError):
            arm.start_planner()
        self.assertEqual("off", arm.planner_state)

    def test_the_glue_alone_says_off_and_ready(self) -> None:
        client = _SlowClient()
        client.release.set()
        planner = CuroboUrPlanner(object(), client_factory=lambda: client)  # type: ignore[arg-type]
        self.assertEqual("off", planner.state)
        planner.start()
        self.assertEqual("ready", planner.state)
        planner.close()
        self.assertEqual("off", planner.state)


class ADisconnectRetiresThePlannerTests(unittest.TestCase):
    """A move that still holds the planner after a disconnect or a stop_planner starts no sidecar on it, or on a new one."""

    def test_the_planner_a_move_holds_starts_no_sidecar_after_a_disconnect(self) -> None:
        """Red before: the move's next check started a sidecar on the glue the disconnect had closed, which nobody held,
        and planner_state said off."""
        sidecars = _Sidecars()
        arm = _arm_building_on(sidecars)
        arm.start_planner()
        held = arm._curobo_ur_planner()  # what a move takes, once, at the start of its route
        arm.disconnect()
        with self.assertRaises(CuroboUnavailableError) as refused:
            held.check_joint_path([_HERE], refresh=False)
        self.assertIn("shut down", str(refused.exception))
        self.assertEqual(1, len(sidecars.made), "the planner the move held started another sidecar")
        self.assertEqual(0, sidecars.orphaned())
        self.assertEqual("off", arm.planner_state)
        self.assertEqual("off", held.state)
        arm.start_planner()  # a person's start, at a desk: one fresh sidecar
        self.assertEqual(2, len(sidecars.made))
        self.assertEqual("ready", arm.planner_state)
        arm.stop_planner()
        self.assertEqual(0, sidecars.orphaned())

    def test_a_stop_planner_retires_it_too(self) -> None:
        sidecars = _Sidecars()
        arm = _arm_building_on(sidecars)
        arm.start_planner()
        held = arm._curobo_ur_planner()
        arm.stop_planner()
        for ask in (lambda: held.check_joint_path([_HERE], refresh=False), held.start,
                    lambda: held.plan_joint(_HERE, start_ur=_HERE), lambda: held.judge_joint_path([_HERE])):
            with self.subTest(ask=ask):
                with self.assertRaises(CuroboUnavailableError):
                    ask()
        self.assertEqual(1, len(sidecars.made))
        self.assertEqual(0, sidecars.orphaned())
        self.assertTrue(arm.is_connected, "stopping the planner leaves the connection as it was")

    def test_a_move_judging_its_route_across_a_disconnect_orphans_nothing(self) -> None:
        """The reviewer's case, on two threads: a move judges its legs on the glue it took, and Disconnect comes between
        two legs. Red before: the second leg started a sidecar nobody held."""
        sidecars = _Sidecars()
        arm = _arm_building_on(sidecars)
        arm.start_planner()
        judging, disconnected = threading.Event(), threading.Event()
        legs: list[str] = []

        def move() -> None:
            planner = arm._curobo_ur_planner()
            for leg in range(3):
                try:
                    planner.check_joint_path([_HERE], refresh=False)
                    legs.append("judged")
                except CuroboUnavailableError:
                    legs.append("refused")
                if leg == 0:
                    judging.set()
                    assert disconnected.wait(5.0)

        mover = _thread(move)
        self.assertTrue(judging.wait(5.0))
        arm.halt("disconnect")  # the console's Disconnect halts first, then disconnects
        arm.disconnect()
        disconnected.set()
        mover.join(5.0)
        self.assertFalse(mover.is_alive())
        self.assertEqual(["judged", "refused", "refused"], legs)
        self.assertEqual([(1, 1)], [(client.started, client.closed) for client in sidecars.made])

    def test_a_halted_arm_keeps_no_fresh_planner_for_a_move_and_starts_none(self) -> None:
        """A move in flight that asks the arm for its planner again after the Disconnect dropped it gets one that starts
        nothing, and the arm keeps none. Red before: it got a fresh glue the arm kept, and its next refresh or check
        started a sidecar for a move that could send nothing, on an arm the console may build anew."""
        sidecars = _Sidecars()
        arm = _arm_building_on(sidecars)
        arm.start_planner()
        arm.halt("disconnect")
        arm.disconnect()
        fresh = arm._curobo_ur_planner()  # the move's next call for the planner
        with self.assertRaises(CuroboUnavailableError) as refused:
            fresh.check_joint_path([_HERE], refresh=False)
        self.assertIn("halted", str(refused.exception))
        self.assertIsNone(arm._curobo_ur, "the arm kept a planner built for a move on a halted arm")
        self.assertEqual("off", arm.planner_state)
        self.assertEqual(1, len(sidecars.made))
        arm.start_planner()  # a person's own start is no move: it starts, halted and at a desk included
        self.assertEqual("ready", arm.planner_state)
        self.assertEqual(2, len(sidecars.made))
        arm.stop_planner()
        arm.clear_halt()
        kept = arm._curobo_ur_planner()
        self.assertIs(kept, arm._curobo_ur, "a cleared arm keeps the planner it builds for a move again")
        self.assertEqual(2, len(sidecars.made), "building the glue starts nothing: the first plan does")
        kept.check_joint_path([_HERE], refresh=False)
        self.assertEqual(3, len(sidecars.made))
        arm.stop_planner()
        self.assertEqual(0, sidecars.orphaned())

    def test_a_plain_close_still_lets_the_next_start_begin_empty(self) -> None:
        """``close`` is not ``retire``: the glue it closed starts a fresh sidecar on the next start, as it always did."""
        sidecars = _Sidecars()
        planner = CuroboUrPlanner(object(), client_factory=sidecars)  # type: ignore[arg-type]
        planner.start()
        planner.close()
        planner.start()
        self.assertEqual([(1, 1), (1, 0)], [(client.started, client.closed) for client in sidecars.made])
        planner.retire()
        planner.retire()  # idempotent
        self.assertEqual([(1, 1), (1, 1)], [(client.started, client.closed) for client in sidecars.made])
        self.assertEqual("off", planner.state)

    def test_a_client_a_move_took_before_the_disconnect_starts_no_sidecar_either(self) -> None:
        """The client itself restarts its sidecar on any call once closed. A move takes it before a camera refresh and
        calls it after: red before, a Disconnect during the refresh closed it, and the move's call started a sidecar on
        a client the glue no longer held."""
        client = _RestartingClient()
        arm = ur_arm_on(RecordingRtde(), "curobo")
        arm._curobo_ur = CuroboUrPlanner(arm._conn, client_factory=lambda: client)
        arm.start_planner()
        taken = arm._curobo_ur._client_or_start()  # what refresh_planner_world holds while it reads the camera
        arm.halt("disconnect")
        arm.disconnect()
        with self.assertRaises(CuroboUnavailableError):
            taken.check_joints([_HERE])
        self.assertEqual((1, 1, False), (client.starts, client.closes, client.running),
                         "the closed client started its sidecar again")

    def test_a_disconnect_waits_for_the_planner_call_in_flight_and_then_closes(self) -> None:
        """A call already inside the client finishes on the sidecar it began on; the retire waits for it (the client's
        own timeouts bound that), then closes, and nothing starts again."""
        client = _RestartingClient()
        arm = ur_arm_on(RecordingRtde(), "curobo")
        arm._curobo_ur = CuroboUrPlanner(arm._conn, client_factory=lambda: client)
        arm.start_planner()
        held = arm._curobo_ur
        client.release.clear()
        verdicts: list[Any] = []
        checking = _thread(lambda: verdicts.append(held.check_joint_path([_HERE], refresh=False)))
        self.assertTrue(client.entered.wait(5.0))
        closing = _thread(arm.disconnect)
        time.sleep(0.2)
        self.assertTrue(closing.is_alive(), "the disconnect closed the sidecar under a call in flight")
        client.release.set()
        checking.join(5.0)
        closing.join(5.0)
        self.assertFalse(checking.is_alive() or closing.is_alive())
        self.assertTrue(verdicts and verdicts[0].valid)
        self.assertEqual((1, 1, False), (client.starts, client.closes, client.running))

    def test_a_copy_of_the_kept_client_neither_recurses_nor_outlives_the_retire(self) -> None:
        """A copy is built without ``__init__``, so its first attribute read finds no client yet: that read must fail
        plainly rather than recurse, and the copy answers to the glue as the original does."""
        import copy

        client = _RestartingClient()
        planner = CuroboUrPlanner(object(), client_factory=lambda: client)  # type: ignore[arg-type]
        planner.start()
        copied = copy.copy(planner._client_or_start())
        self.assertTrue(copied.check_joints([_HERE]).valid)
        planner.retire()
        with self.assertRaises(CuroboUnavailableError):
            copied.check_joints([_HERE])
        self.assertEqual((1, 1, False), (client.starts, client.closes, client.running))

    def test_a_double_with_only_close_is_closed(self) -> None:
        """An arm whose planner is a double without ``retire`` closes it, as before."""
        closed: list[str] = []

        class _Closing:
            state = "ready"

            def close(self) -> None:
                closed.append("close")

        arm = ur_arm_on(RecordingRtde(), "curobo")
        arm._curobo_ur = _Closing()
        arm.disconnect()
        self.assertEqual(["close"], closed)
        self.assertIsNone(arm._curobo_ur)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
