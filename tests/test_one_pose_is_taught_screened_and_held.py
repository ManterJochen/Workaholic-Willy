"""One pose taught by hand from the console: freed, captured once still, held, screened at once, written only if clear.

The owner (decision 15, Q10, Q14, 2026-09-30): a pose a task places at or returns to is taught in the console by
freedrive, screened by the exact guard and the planner the moment it is held, and written into the cell's own layer
only where both clear it or it lies in the planner's band. ``teach.teach_one`` is the library half: ONE pose per
session, over the same ``HandGuide.wait`` (50 Hz, the stillness gate, no capture outside the cable window or the box)
and the same ``TaughtPose`` that ``teach_poses`` uses. What is held here, over the scripted hand-guided arm, console and
clock of ``tests/test_a_pose_is_taught_by_hand.py``:

* Save captures only once the arm has stood still for half a second; the arm is held before it is read, it is screened
  once while it holds, and a CLEAR or BAND pose is handed to the store; nothing moves by itself at any point;
* an ERROR pose (the exact guard or the planner refuses it) is never written, and neither is an UNSCREENED one or one
  whose screen raised: the arm stays held and the result says why;
* everything that would make a pose unwritable is refused BEFORE the arm is freed: a name that is no pose name, an arm
  with no hand guiding, an arm that screens nothing or was not handed its wrist camera's housing, a planner that is not
  ready, a store that names no file;
* Cancel holds at once, wherever the arm is, also while a Save still waits for the arm to stand still, and then nothing
  is written; a hold the caller asks for (the browser's heartbeat lapsed, the time limit) holds only once the arm
  stands still, never while it moves in a person's hands (Q14), and one asked for before the arm is freed frees
  nothing;
* every sample reaches the caller's watcher, and a watcher that raises never ends the session;
* the stillness gate the session uses is hand_guiding's own, offered to the console as ``StillnessGate``;
* the checks made before the arm is freed are one rule, offered to the console as ``teach_refusal`` (a pose that could
  not be taught is said before anyone asks to teach it).
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.config.loader import load_config, reload_config
from src.config.tree import ConfigTree
from src.robot.core.freedrive import FreedriveSample
from src.robot.execution.hand_guiding import (
    FINISH,
    SKIP,
    STILL_FOR_S,
    STILL_RAD_S,
    HandGuide,
    HandGuidingLimits,
    HandGuidingRefused,
)
from src.robot.execution.teach import (
    HELD_WHEN_STILL,
    ProfilePoseStore,
    StillnessGate,
    TaughtOne,
    TaughtPose,
    TeachRefused,
    teach_one,
)
from src.robot.safety.planning.band import PoseScreen, PoseVerdict
from tests.test_a_pose_is_taught_by_hand import (
    BOX,
    LOOK_1_DEG,
    TREE,
    _answers,
    _Clock,
    _Console,
    _GuidedArm,
    _PlainArm,
    _sample,
    _stand,
    _View,
)

CLEAR = PoseScreen(PoseVerdict.CLEAR, "the exact guard and the planner both clear it.")
BAND = PoseScreen(PoseVerdict.BAND, "forearm|wrist_2 overlaps 1.2 mm, and the exact meshes keep 19.0 mm.",
                  nearby=tuple(math.radians(v) for v in (-45.0, -100.2, -110.0, -64.0, 90.0, 0.0)))
GUARD_REFUSED = PoseScreen(PoseVerdict.GUARD_REFUSED, "forearm|wrist_2: mesh distance 4.1 mm < 10.000 mm. No move goes "
                                                      "there.", nearby=tuple(math.radians(v) for v in LOOK_1_DEG))
PLANNER_REFUSED = PoseScreen(PoseVerdict.PLANNER_REFUSED, "the planner's world holds a box there.")
UNSCREENED = PoseScreen(PoseVerdict.UNSCREENED, "the exact guard accepts it; the planner could not be asked: no sidecar",
                        planner_unavailable=True)
TARGET = "config/robot/robot.cell.yaml"


class _Store:
    """A pose store that keeps what it was handed: ``refuse`` is what its write answers ("" once written)."""

    def __init__(self, refuse: str = "", target: str = TARGET) -> None:
        self.refuse = refuse
        self._target = target
        self.writes: list[tuple[TaughtPose, PoseScreen]] = []
        self.target_reads = 0

    @property
    def target(self) -> str:
        self.target_reads += 1
        return self._target

    def write(self, pose: TaughtPose, screen: PoseScreen) -> str:
        self.writes.append((pose, screen))
        return self.refuse


class _NoFileStore(_Store):
    @property
    def target(self) -> str:
        raise TeachRefused("no_layer", "the chain has no layer of its own, so a pose would land in robot.yaml")


def _screening(*verdicts: Any) -> Any:
    """A ``screen_configuration`` answering ``verdicts`` in turn (an exception is raised), keeping what it was asked and
    when, on the arm's own log."""

    def screen(self: Any, joints: Any, *, ask_planner: bool = True) -> Any:
        screen.asked.append(([round(math.degrees(v), 4) for v in joints.tolist()], ask_planner))  # type: ignore[attr-defined]
        self.log.append("screen")
        verdict = verdicts[min(len(screen.asked) - 1, len(verdicts) - 1)]  # type: ignore[attr-defined]
        if isinstance(verdict, BaseException):
            raise verdict
        return verdict

    screen.asked = []  # type: ignore[attr-defined]
    return screen


class _Case(unittest.TestCase):
    """A scripted arm that carries its wrist camera's housing and screens, a scripted console, the test's own clock."""

    def arm(self, *stands: list[FreedriveSample], screen: Any = CLEAR, housing: bool = True, plain: bool = False,
            planner: Any = None) -> _GuidedArm:
        self.clock = _Clock()
        kind = _PlainArm if plain else _GuidedArm
        attrs: dict[str, Any] = {}
        if screen is not None:
            attrs["screen_configuration"] = screen if callable(screen) else _screening(screen)
        if planner is not None:
            attrs["planner_state"] = planner
        arm = type("_TeachArm", (kind,), attrs)(list(stands) or [_stand(LOOK_1_DEG)], self.clock)
        # The housing of the one wrist camera the tests' tree hangs on the arm, as Robot.from_tree hands it.
        arm.safety_preflight.wrist_bodies = (lambda _arm=None: (SimpleNamespace(rig_id="wrist"),)) if housing else (
            lambda _arm=None: ())
        self.guided = arm
        return arm

    def teach(self, lines: "list[str | None] | None" = None, *, name: str = "drop_left", store: Any = None,
              confirmed: bool = True, view: Any = None, **keywords: Any) -> TaughtOne:
        self.console = _Console(lines if lines is not None else _answers(""), self.guided.log)
        self.guide = HandGuide(self.console, view, clock=self.clock, sleep=self.clock.sleep)
        self.guide.payload_confirmed = confirmed
        self.store = store if store is not None else _Store()
        keywords.setdefault("tree", TREE)
        return teach_one(self.guided, self.guide, HandGuidingLimits.of(self.guided, BOX), name, store=self.store,
                         **keywords)

    def steps(self) -> list[str]:
        """What the arm was asked and did, without the console's lines and the hold times."""
        return [line.split(" at ")[0] for line in self.guided.log if not line.startswith("say")]

    def held_at(self) -> list[float]:
        return [float(line.split(" at ")[1]) for line in self.guided.log if line.startswith("hold at")]


class SaveCapturesHoldsScreensAndWritesTests(_Case):
    def test_a_clear_pose_is_captured_once_still_held_screened_and_written(self) -> None:
        arm = self.arm(_stand(LOOK_1_DEG))

        taught = self.teach()

        self.assertEqual("saved", taught.outcome, taught.render())
        self.assertEqual(TARGET, taught.written)
        self.assertEqual(["session", "free", "hold", "screen", "left, held"], self.steps())
        (held,) = self.held_at()
        self.assertGreaterEqual(round(held - arm.still_since[0], 6), STILL_FOR_S, "captured before it stood still")
        ((pose, screen),) = self.store.writes
        self.assertEqual("drop_left", pose.name)
        self.assertEqual([round(v, 4) for v in LOOK_1_DEG], [round(v, 4) for v in pose.joints_deg])
        self.assertIs(CLEAR, screen)
        (asked,) = type(arm).screen_configuration.asked  # type: ignore[attr-defined]
        self.assertEqual([round(v, 4) for v in LOOK_1_DEG], asked[0])
        self.assertTrue(asked[1], "the planner was not asked: an unscreened pose is never written")
        self.assertEqual(1, arm.sessions, "one pose per session")
        self.assertEqual(1, self.guided.log.count("free"))

    def test_a_band_pose_is_written_with_its_screen(self) -> None:
        self.arm(screen=BAND)
        taught = self.teach()
        self.assertEqual("saved", taught.outcome)
        self.assertIs(BAND, self.store.writes[0][1])
        self.assertIsNotNone(taught.screen)

    def test_the_target_is_said_before_the_arm_is_freed(self) -> None:
        self.arm()
        self.teach()
        said = [line for line in self.guided.log if line.startswith("say") or line == "free"]
        target = next(i for i, line in enumerate(said) if TARGET in line)
        self.assertLess(target, said.index("free"))
        self.assertGreaterEqual(self.store.target_reads, 1)

    def test_nothing_moves_by_itself(self) -> None:
        self.arm()
        self.teach()
        self.assertFalse([line for line in self.steps() if "move" in line])

    def test_a_store_that_refuses_the_write_leaves_the_pose_unsaved(self) -> None:
        self.arm()
        taught = self.teach(store=_Store(refuse="the loader refused it: two poses share the label 'Ablage links'"))
        self.assertEqual("not_saved", taught.outcome)
        self.assertEqual("", taught.written)
        self.assertIn("two poses share the label", taught.not_written)

    def test_a_store_that_raises_leaves_the_pose_unsaved_with_the_arm_held(self) -> None:
        class _Raising(_Store):
            def write(self, pose: TaughtPose, screen: PoseScreen) -> str:
                raise OSError("the disk is full")

        self.arm()
        taught = self.teach(store=_Raising())
        self.assertEqual("not_saved", taught.outcome)
        self.assertIn("the disk is full", taught.not_written)
        self.assertEqual("left, held", self.steps()[-1])


class OnlyAClearedPoseIsWrittenTests(_Case):
    def test_an_error_pose_is_never_written_and_carries_the_pose_to_teach_instead(self) -> None:
        for verdict in (GUARD_REFUSED, PLANNER_REFUSED):
            with self.subTest(verdict=verdict.verdict):
                self.arm(screen=verdict)
                taught = self.teach()
                self.assertEqual("refused", taught.outcome)
                self.assertEqual([], self.store.writes)
                self.assertIs(verdict, taught.screen)
                self.assertIn(verdict.detail.split(":")[0], taught.not_written)
                self.assertIsNotNone(taught.pose)

    def test_an_unscreened_pose_is_never_written(self) -> None:
        self.arm(screen=UNSCREENED)
        taught = self.teach()
        self.assertEqual("not_saved", taught.outcome)
        self.assertEqual([], self.store.writes)
        self.assertIn("not screened", taught.not_written)

    def test_a_screen_that_raises_writes_nothing_and_the_arm_stays_held(self) -> None:
        self.arm(screen=_screening(RuntimeError("the sidecar stopped answering")))
        taught = self.teach()
        self.assertEqual("not_saved", taught.outcome)
        self.assertIsNone(taught.screen)
        self.assertIn("the sidecar stopped answering", taught.screen_error)
        self.assertEqual([], self.store.writes)
        self.assertEqual(["session", "free", "hold", "screen", "left, held"], self.steps())


class RefusedBeforeTheArmIsFreedTests(_Case):
    def _refused(self, code: str, **keywords: Any) -> TeachRefused:
        with self.assertRaises(TeachRefused) as caught:
            self.teach(**keywords)
        self.assertEqual(code, caught.exception.code)
        self.assertIsInstance(caught.exception, HandGuidingRefused)
        self.assertNotIn("free", self.guided.log, "the arm was freed before the refusal")
        self.assertNotIn("session", self.guided.log)
        self.assertEqual([], getattr(self, "store", _Store()).writes)
        return caught.exception

    def test_an_arm_without_its_wrist_cameras_housing_is_refused(self) -> None:
        """⛔ HONEST SCREENS (review of F1, 2026-09-30): both models would judge the pose without the housing that
        hangs there, so a pose the housing meets would read clear, and be written."""
        arm = self.arm(housing=False)
        refused = self._refused("screen_unavailable")
        for part in ("'wrist'", "housing"):
            self.assertIn(part, str(refused))
        self.assertEqual([], type(arm).screen_configuration.asked)  # type: ignore[attr-defined]

    def test_an_arm_with_no_tree_to_say_which_cameras_hang_on_it_is_refused(self) -> None:
        self.arm()
        self._refused("screen_unavailable", tree=None)

    def test_an_arm_that_screens_nothing_is_refused(self) -> None:
        self.arm(screen=None)
        self.assertIn("screen", str(self._refused("screen_unavailable")))

    def test_an_arm_that_plans_with_no_planner_is_refused(self) -> None:
        self.arm(planner="not_used")
        self._refused("screen_unavailable")

    def test_a_planner_that_is_not_ready_is_refused(self) -> None:
        for state in ("off", "starting"):
            with self.subTest(state=state):
                self.arm(planner=state)
                refused = self._refused("planner_not_ready")
                self.assertIn(state, str(refused))

    def test_a_planner_state_read_through_a_method_counts_as_well(self) -> None:
        self.arm(planner=lambda _self: "off")
        self._refused("planner_not_ready")
        self.arm(planner=lambda _self: "ready")
        self.assertEqual("saved", self.teach().outcome)

    def test_a_planner_state_nobody_can_read_is_no_ready_planner(self) -> None:
        def unreadable(_self: Any) -> str:
            raise ConnectionError("the planner lock is held")

        self.arm(planner=unreadable)
        self.assertIn("the planner lock is held", str(self._refused("planner_not_ready")))

    def test_a_ready_planner_lets_the_teach_go_on(self) -> None:
        self.arm(planner="ready")
        self.assertEqual("saved", self.teach().outcome)

    def test_a_name_that_is_no_pose_name_is_refused_before_anything_is_read(self) -> None:
        for name in ("home", "drop left", "class", "x" * 33, "yes", "Off", "__null__"):
            with self.subTest(name=name):
                self.arm()
                self._refused("invalid_name", name=name)
                self.assertNotIn("payload read", self.guided.log)

    def test_an_arm_that_offers_no_hand_guiding_is_refused_naming_its_vendor(self) -> None:
        self.arm(plain=True, screen=None)
        self.assertIn("vendor 'ur'", str(self._refused("no_hand_guiding")))

    def test_an_arm_that_is_not_connected_is_refused(self) -> None:
        arm = self.arm()
        arm.is_connected = False
        self._refused("not_connected")

    def test_a_store_that_names_no_file_is_refused(self) -> None:
        self.arm()
        self.assertIn("robot.yaml", str(self._refused("no_layer", store=_NoFileStore())))

    def test_the_payload_is_asked_where_the_caller_did_not_confirm_it(self) -> None:
        self.arm()
        taught = self.teach(_answers("", ""), confirmed=False)
        self.assertEqual("saved", taught.outcome)
        self.assertLess(self.guided.log.index("payload read"), self.guided.log.index("session"))
        self.arm()
        with self.assertRaises(HandGuidingRefused):
            self.teach(_answers("n"), confirmed=False)
        self.assertNotIn("free", self.guided.log)


class TheArmIsHeldTests(_Case):
    def test_cancel_holds_at_once_while_the_arm_still_moves(self) -> None:
        for key in ("q", "s"):
            with self.subTest(key=key):
                arm = self.arm(_stand(LOOK_1_DEG, moving=200))
                taught = self.teach(_answers(key))
                self.assertEqual("cancelled", taught.outcome)
                self.assertEqual(["session", "free", "hold", "left, held"], self.steps())
                self.assertEqual({}, arm.still_since, "the hold waited for the arm to stand still")
                self.assertEqual([], self.store.writes)
                self.assertIsNone(taught.pose)

    def test_a_cancel_sent_through_the_view_before_the_wait_is_kept_and_holds_at_once(self) -> None:
        """``HandGuide.wait`` forgets what the console sent before it waits, as the terminal does, and keeps a finish
        from the view: the console sends Cancel, a halt and a disconnect that way, so none of them is lost."""
        arm = self.arm(_stand(LOOK_1_DEG, moving=200))
        taught = self.teach([], view=_View(["finish"]))
        self.assertEqual("cancelled", taught.outcome)
        self.assertEqual(["session", "free", "hold", "left, held"], self.steps())
        self.assertEqual({}, arm.still_since)

    def test_a_lapsed_heartbeat_holds_only_once_the_arm_stands_still(self) -> None:
        """Q14 = A: an arm that locks in a person's hands is the injury the freedrive design avoids."""
        arm = self.arm(_stand(LOOK_1_DEG, moving=150))   # three seconds in a person's hands, then still
        lapsed_at = 1.0

        taught = self.teach([], hold_when_still=lambda: "heartbeat" if self.clock() >= lapsed_at else "")

        self.assertEqual(HELD_WHEN_STILL, taught.choice)
        self.assertEqual("held_when_still", taught.outcome)
        self.assertEqual("heartbeat", taught.because)
        self.assertEqual(["session", "free", "hold", "left, held"], self.steps())
        (held,) = self.held_at()
        self.assertGreater(held, 3.0 - 1e-9, "held while the arm still moved")
        self.assertGreaterEqual(round(held - arm.still_since[0], 6), STILL_FOR_S)
        self.assertLess(held - arm.still_since[0], STILL_FOR_S + 0.1, "held long after it stood still")
        self.assertEqual([], self.store.writes)
        self.assertIsNone(taught.pose)

    def test_the_time_limit_holds_the_same_way(self) -> None:
        arm = self.arm(_stand(LOOK_1_DEG, moving=0))
        taught = self.teach([], hold_when_still=lambda: "time_limit" if self.clock() >= 2.0 else "")
        self.assertEqual(("held_when_still", "time_limit"), (taught.outcome, taught.because))
        (held,) = self.held_at()
        self.assertGreaterEqual(held, 2.0)
        self.assertGreaterEqual(round(held - arm.still_since[0], 6), STILL_FOR_S)

    def test_a_hold_asked_for_before_the_arm_is_freed_frees_nothing(self) -> None:
        self.arm()
        taught = self.teach([], hold_when_still=lambda: "heartbeat")
        self.assertEqual(("held_when_still", "heartbeat"), (taught.outcome, taught.because))
        self.assertNotIn("free", self.guided.log)

    def test_a_person_s_save_wins_over_a_hold_asked_for_at_the_same_time(self) -> None:
        self.arm(_stand(LOOK_1_DEG, moving=0))
        taught = self.teach(_answers(""), hold_when_still=lambda: "heartbeat" if self.clock() > 0.0 else "")
        self.assertEqual("saved", taught.outcome)

    def test_a_hold_request_that_cannot_be_read_holds_once_still(self) -> None:
        self.arm(_stand(LOOK_1_DEG, moving=0))

        def broken() -> str:
            raise RuntimeError("the heartbeat clock is gone")

        taught = self.teach([], hold_when_still=broken)
        self.assertEqual("held_when_still", taught.outcome)
        self.assertIn("the heartbeat clock is gone", taught.because)

    def test_a_sample_that_raises_ends_the_session_with_the_arm_held(self) -> None:
        arm = self.arm()

        def stopped() -> FreedriveSample:
            raise RuntimeError("protective stop")

        arm.next_sample = stopped  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            self.teach()
        self.assertEqual("left, held", self.steps()[-1])
        self.assertEqual([], self.store.writes)


class ASaveStillWaitingIsCancelledAtOnceTests(_Case):
    """⛔ Q14 = A: Save, Hold, Cancel, closing the tab, Halt and Disconnect hold the arm at once. A Save waits for the arm
    to stand still, up to 5 s (``HandGuide._still``), and that wait reads no key: a Cancel sent meanwhile (the view's
    ``finish``, as the console sends Cancel, Hold, Halt and Disconnect) used to be read only after it, so the arm stayed
    free, and a pose the person had cancelled was written once the arm stood still (review of 2026-10-01)."""

    def _send_at(self, at: float, send: Any) -> list[float]:
        """``send()`` once, as soon as the guide's clock reaches ``at``; when it was sent."""
        sent: list[float] = []

        def on_sleep(_seconds: float) -> None:
            if not sent and self.clock() >= at - 1e-9:
                send()
                sent.append(self.clock())

        self.clock.on_sleep = on_sleep
        return sent

    def _held_within_one_sample_of(self, sent: list[float]) -> None:
        (held,) = self.held_at()
        self.assertEqual(1, len(sent), "the cancel was never sent: the session ended before it")
        self.assertLessEqual(held - sent[0], self.guide.period_s + 1e-6,
                             f"held at {held:.2f}, {held - sent[0]:.2f} s after the cancel: the arm stayed free")

    def test_a_cancel_while_save_waits_writes_nothing_though_the_arm_then_stands_still(self) -> None:
        arm = self.arm(_stand(LOOK_1_DEG, moving=50))       # a second in a person's hands, then still
        view = _View()
        sent = self._send_at(0.1, lambda: view.keys.append("finish"))

        taught = self.teach(_answers(""), view=view)        # Save at once, Cancel 0.1 s later

        self.assertEqual("cancelled", taught.outcome, taught.render())
        self.assertEqual(FINISH, taught.choice)
        self.assertEqual([], self.store.writes, "a pose the person cancelled was written")
        self.assertIsNone(taught.pose)
        self._held_within_one_sample_of(sent)
        self.assertEqual({}, arm.still_since, "the hold waited for the arm to stand still")
        self.assertEqual(["session", "free", "hold", "left, held"], self.steps())

    def test_a_cancel_while_save_waits_holds_an_arm_that_keeps_moving_at_once(self) -> None:
        self.arm(_stand(LOOK_1_DEG, moving=400))            # eight seconds in a person's hands
        view = _View()
        sent = self._send_at(0.1, lambda: view.keys.append("finish"))

        taught = self.teach(_answers(""), view=view)

        self.assertEqual("cancelled", taught.outcome)
        self._held_within_one_sample_of(sent)
        self.assertEqual([], self.store.writes)

    def test_every_key_that_holds_holds_at_once_while_save_waits(self) -> None:
        for where, key, choice in (("view", "finish", FINISH), ("view", "closed", FINISH), ("view", "skip", SKIP),
                                   ("console", "q", FINISH), ("console", "s", SKIP)):
            with self.subTest(where=where, key=key):
                self.arm(_stand(LOOK_1_DEG, moving=50))
                view = _View()

                def send(where: str = where, key: str = key, view: _View = view) -> None:
                    (view.keys if where == "view" else self.console.lines).append(key)

                sent = self._send_at(0.1, send)
                taught = self.teach(_answers(""), view=view)
                self.assertEqual((choice, "cancelled"), (taught.choice, taught.outcome))
                self.assertEqual([], self.store.writes)
                self._held_within_one_sample_of(sent)

    def test_a_key_that_holds_nothing_read_while_save_waits_is_kept_for_the_wait(self) -> None:
        """⭐ THE CONTROL: the keys a Save's wait reads early are handed on in order, never lost. A second Save sent
        while the first waits on an arm that keeps moving captures once the arm stands still."""
        arm = self.arm(_stand(LOOK_1_DEG, moving=300))      # six seconds in a person's hands, then still
        view = _View()
        self._send_at(0.1, lambda: view.keys.append("enter"))

        taught = self.teach(_answers(""), view=view)

        self.assertEqual("saved", taught.outcome, taught.render())
        (held,) = self.held_at()
        self.assertGreaterEqual(round(held - arm.still_since[0], 6), STILL_FOR_S)
        self.assertTrue([line for line in self.console.said if "still moves" in line], "the first Save never gave up")

    def test_a_hold_asked_for_while_save_waits_leaves_the_persons_save_to_finish(self) -> None:
        """A person's line comes first: the console's own hold (a lapsed heartbeat) does not cut a Save the person sent
        before it, and the arm is held once it stands still either way."""
        self.arm(_stand(LOOK_1_DEG, moving=20))
        taught = self.teach(_answers(""), hold_when_still=lambda: "heartbeat" if self.clock() >= 0.1 else "")
        self.assertEqual("saved", taught.outcome, taught.render())


class TheChecksBeforeTheFreeAreOneRuleTests(_Case):
    """``teach_refusal`` makes the checks ``teach_one`` makes before the arm is freed, without freeing, asking or
    writing anything: the console says why a pose cannot be taught before anyone asks to teach one (``GET /v1/poses``,
    ``teachable`` and ``why_not``), by the rule the teach itself refuses by."""

    def test_it_answers_what_teach_one_refuses_and_frees_nothing(self) -> None:
        from src.robot.execution.teach import teach_refusal

        for setup, keywords, code in (({"housing": False}, {}, "screen_unavailable"),
                                      ({"screen": None}, {}, "screen_unavailable"),
                                      ({"planner": "not_used"}, {}, "screen_unavailable"),
                                      ({"planner": "starting"}, {}, "planner_not_ready"),
                                      ({}, {"name": "home"}, "invalid_name"),
                                      ({}, {"store": _NoFileStore()}, "no_layer"),
                                      ({}, {"tree": None}, "screen_unavailable")):
            with self.subTest(setup=setup, keywords=keywords):
                arm = self.arm(**setup)
                asked: dict[str, Any] = {"tree": TREE, **keywords}

                refused = teach_refusal(arm, **asked)

                self.assertIsInstance(refused, TeachRefused)
                assert refused is not None
                self.assertEqual(code, refused.code)
                self.assertEqual([], arm.log, "a check before the teach asked or freed something")
                with self.assertRaises(TeachRefused) as caught:
                    self.teach(**asked)
                self.assertEqual(code, caught.exception.code, "the teach refuses by another rule")

    def test_an_arm_that_is_not_connected_or_cannot_be_guided_is_said(self) -> None:
        from src.robot.execution.teach import teach_refusal

        arm = self.arm()
        arm.is_connected = False
        self.assertEqual("not_connected", getattr(teach_refusal(arm, tree=TREE), "code", None))
        self.assertEqual("no_hand_guiding", getattr(teach_refusal(self.arm(plain=True, screen=None), tree=TREE),
                                                    "code", None))

    def test_a_teach_that_could_go_on_answers_none_and_writes_nothing(self) -> None:
        from src.robot.execution.teach import teach_refusal

        arm = self.arm(planner="ready")
        store = _Store()
        self.assertIsNone(teach_refusal(arm, tree=TREE, name="drop_left", store=store))
        self.assertEqual([], arm.log)
        self.assertGreaterEqual(store.target_reads, 1, "the store's file was never asked for")
        self.assertEqual([], store.writes)

    def test_the_planner_is_always_asked_and_a_caller_that_says_otherwise_is_refused(self) -> None:
        """Only a pose both authorities clear, or one in the planner's band, is written: a screen without the planner
        could write nothing, so ``ask_planner=False`` is a caller's mistake, refused before anything is asked."""
        self.arm()
        with self.assertRaises(ValueError) as caught:
            self.teach(ask_planner=False)
        self.assertIn("planner", str(caught.exception))
        self.assertEqual([], [line for line in self.guided.log if not line.startswith("say")])
        self.arm()
        self.assertEqual("saved", self.teach(ask_planner=True).outcome)


class TheCallerSeesEverySampleTests(_Case):
    def test_every_sample_reaches_the_watcher(self) -> None:
        self.arm(_stand(LOOK_1_DEG, moving=3))
        seen: list[FreedriveSample] = []
        self.teach(watch=seen.append)
        self.assertGreater(len(seen), 25, "the stillness gate's samples did not reach the watcher")
        self.assertEqual(0.3, seen[0].peak_joint_speed_rad_s)
        self.assertEqual(0.0, seen[-1].peak_joint_speed_rad_s)

    def test_a_watcher_that_raises_never_ends_the_session(self) -> None:
        self.arm()

        def broken(_sample: FreedriveSample) -> None:
            raise ValueError("the snapshot lock is gone")

        self.assertEqual("saved", self.teach(watch=broken).outcome)


class TheCallerHearsEveryStepTests(_Case):
    """The console's events (``teach.free``, ``teach.holding``, ``teach.screening``) come from the session's own steps,
    never from its sentences."""

    def test_a_capture_is_freed_held_and_screened(self) -> None:
        self.arm()
        steps: list[str] = []
        self.teach(on_step=steps.append)
        self.assertEqual(["free", "holding", "screening"], steps)

    def test_a_cancel_and_a_hold_once_still_are_freed_and_held(self) -> None:
        for lines, hold in ((_answers("q"), None), ([], lambda: "heartbeat" if self.clock() >= 0.5 else "")):
            with self.subTest(hold=hold is not None):
                self.arm(_stand(LOOK_1_DEG, moving=0))
                steps: list[str] = []
                self.teach(lines, hold_when_still=hold, on_step=steps.append)
                self.assertEqual(["free", "holding"], steps)

    def test_a_hold_asked_for_before_the_arm_is_freed_takes_no_step(self) -> None:
        self.arm()
        steps: list[str] = []
        self.teach([], hold_when_still=lambda: "heartbeat", on_step=steps.append)
        self.assertEqual([], steps)

    def test_a_listener_that_raises_never_ends_the_session(self) -> None:
        self.arm()

        def broken(_step: str) -> None:
            raise RuntimeError("the event hub is gone")

        self.assertEqual("saved", self.teach(on_step=broken).outcome)


_SHIPPED = Path(__file__).resolve().parents[1] / "config"
CHAIN = "ur10,hande,cell"
LAYERS = ("ur10", "hande", "cell")


class TheCellsOwnLayerKeepsThePoseTests(_Case):
    """``ProfilePoseStore``: the pose door into the cell's own layer, with every refusal said before the arm is freed."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp()) / "data"
        shutil.copytree(_SHIPPED, self.tmp)
        self.cell = self.tmp / "robot" / "robot.cell.yaml"
        self.cell.write_text("# The cell's own layer: what this cell measured and taught. Not in git.\n",
                             encoding="utf-8")
        self.tree = ConfigTree(root=self.tmp, profile=CHAIN, layers=LAYERS)
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    def robot(self) -> Any:
        reload_config()
        return load_config(self.tmp, profile=CHAIN).robot

    def test_a_clear_pose_lands_in_the_cells_own_layer_with_its_label(self) -> None:
        self.arm()
        taught = self.teach(store=ProfilePoseStore(self.tree, "drop_left", "Ablage links", make_default_place=True))
        self.assertEqual("saved", taught.outcome, taught.render())
        self.assertTrue(Path(taught.written).samefile(self.cell))
        robot = self.robot()
        pose = robot.named_poses["drop_left"]
        self.assertEqual([round(v, 4) + 0.0 for v in LOOK_1_DEG], list(pose.joints_deg))
        self.assertEqual(("Ablage links", "clear", ""), (pose.label, pose.screen, pose.note))
        assert taught.pose is not None
        self.assertEqual(taught.pose.taught_at, pose.taught_at)
        self.assertEqual("drop_left", robot.default_place_pose)

    def test_a_band_pose_keeps_the_screens_own_words_on_one_line(self) -> None:
        self.arm(screen=BAND)
        taught = self.teach(store=ProfilePoseStore(self.tree, "drop_left", "Ablage links"))
        self.assertEqual("saved", taught.outcome, taught.render())
        pose = self.robot().named_poses["drop_left"]
        self.assertEqual("band", pose.screen)
        for part in ("cushion band", "forearm|wrist_2", "Nearby, both clear"):
            self.assertIn(part, pose.note)
        self.assertNotIn("\n", pose.note)

    def test_what_could_never_be_written_is_refused_as_the_store_is_made(self) -> None:
        for name, label, code in (("home", "Home", "invalid_name"), ("drop", "Kiste #2", "invalid_label"),
                                  ("drop", "", "invalid_label")):
            with self.subTest(name=name, label=label), self.assertRaises(TeachRefused) as caught:
                ProfilePoseStore(self.tree, name, label)
            self.assertEqual(code, caught.exception.code)
        # Over the shipped tree (the scratch tree stands for it), a chain with no layer would write the robot.yaml
        # every cell reads; a customer's own tree outside the repository takes the pose there (2026-10-05).
        from unittest.mock import patch

        with patch("src.config.edit._is_shipped_tree", return_value=True), self.assertRaises(TeachRefused) as caught:
            ProfilePoseStore(ConfigTree(root=self.tmp, profile=None, layers=()), "drop", "Ablage")
        self.assertEqual("no_layer", caught.exception.code)
        self.assertIn("robot.yaml", str(caught.exception))
        own = ProfilePoseStore(ConfigTree(root=self.tmp, profile=None, layers=()), "drop", "Ablage")
        self.assertEqual(str(self.tmp / "robot" / "robot.yaml"), own.target)

    def test_a_name_the_tree_holds_is_taken_unless_it_is_replaced(self) -> None:
        self.assertTrue(self.tree.write_named_pose("drop_left", joints_deg=list(LOOK_1_DEG), label="Ablage links",
                                                   screen="clear", taught_at="2026-10-01T09:12:00+02:00").applied)
        with self.assertRaises(TeachRefused) as caught:
            ProfilePoseStore(self.tree, "drop_left", "Ablage links")
        self.assertEqual("name_taken", caught.exception.code)
        ProfilePoseStore(self.tree, "drop_left", "Ablage links", replace=True)   # its own label is no other pose's

    def test_a_label_another_pose_carries_is_refused(self) -> None:
        self.assertTrue(self.tree.write_named_pose("drop_left", joints_deg=list(LOOK_1_DEG), label="Ablage links",
                                                   screen="clear", taught_at="2026-10-01T09:12:00+02:00").applied)
        with self.assertRaises(TeachRefused) as caught:
            ProfilePoseStore(self.tree, "park", "  ablage   LINKS ")
        self.assertEqual("invalid_label", caught.exception.code)
        self.assertIn("'drop_left'", str(caught.exception))

    def test_a_word_another_pose_answers_to_is_refused_before_anything_is_freed(self) -> None:
        """A pose answers to its name and to its label (a pose written by hand with no label is said by its name), and
        the command reader hands a pose on by the word a person said: two poses answering to one word could send a part
        to either (review of 2026-10-01)."""
        self.cell.write_text("robot:\n  named_poses:\n    park:\n"
                             "      joints_deg: [-90.0, -80.0, -110.0, -80.0, 90.0, 0.0]\n", encoding="utf-8")
        for name, label, code in (("drop", "Park", "invalid_label"),     # the label is the unlabelled pose's word
                                  ("Park", "Parken", "invalid_name"),    # the name differs from 'park' only in case
                                  ("drop", "park", "invalid_label")):
            with self.subTest(name=name, label=label), self.assertRaises(TeachRefused) as caught:
                ProfilePoseStore(self.tree, name, label)
            self.assertEqual(code, caught.exception.code)
            self.assertIn("'park'", str(caught.exception))
        ProfilePoseStore(self.tree, "drop", "Ablage")                     # a word no other pose answers to
        ProfilePoseStore(self.tree, "park", "park", replace=True)         # its own name is its own word

    def test_a_name_yaml_reads_as_another_value_is_refused_before_anything_is_freed(self) -> None:
        for name in ("yes", "On", "null", "__null__"):
            with self.subTest(name=name), self.assertRaises(TeachRefused) as caught:
                ProfilePoseStore(self.tree, name, "Ablage")
            self.assertEqual("invalid_name", caught.exception.code)

    @unittest.skipUnless(shutil.which("git"), "git is not on PATH")
    def test_a_cell_layer_git_does_not_keep_out_is_refused_before_anything_is_freed(self) -> None:
        """⛔ Q10 = A: taught poses go into the cell's own layer, which git keeps out. In a work tree that does not
        ignore the cell's file, a pose would go into the repository with the next ``git add``."""
        repo = self.tmp.parent
        subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
        before = self.cell.read_bytes()
        with self.assertRaises(TeachRefused) as caught:
            ProfilePoseStore(self.tree, "drop_left", "Ablage links")
        self.assertEqual("no_layer", caught.exception.code)
        for part in ("data/robot/robot.cell.yaml", ".gitignore", "Nothing was freed"):
            self.assertIn(part, str(caught.exception))
        self.assertEqual(before, self.cell.read_bytes())

        (repo / ".gitignore").write_text("data/**/*.cell.yaml\n", encoding="utf-8")
        self.arm()
        taught = self.teach(store=ProfilePoseStore(self.tree, "drop_left", "Ablage links"))
        self.assertEqual("saved", taught.outcome, taught.render())

    def test_a_write_the_loader_refuses_comes_back_as_the_reason_with_the_file_untouched(self) -> None:
        """A pose written by hand in block style cannot be rewritten line by line: the teach ends not saved, saying the
        loader's own words, and the cell's file keeps every byte."""
        self.cell.write_text("robot:\n  named_poses:\n    drop_left:\n      joints_deg:\n"
                             + "".join(f"        - {value}\n" for value in (-60.0, -95.0, -120.0, -55.0, 90.0, 0.0)),
                             encoding="utf-8")
        before = self.cell.read_bytes()
        self.arm()
        taught = self.teach(store=ProfilePoseStore(self.tree, "drop_left", "Ablage links", replace=True))
        self.assertEqual("not_saved", taught.outcome)
        self.assertTrue(taught.not_written)
        self.assertEqual(before, self.cell.read_bytes())

    def test_a_tree_that_does_not_load_is_left_to_the_pose_doors_own_refusal(self) -> None:
        """Nothing can be checked against a tree that does not load: the store is made, and the write that follows is
        refused with the loader's own words, the file untouched."""
        self.cell.write_text("robot:\n  named_poses: [1, 2]\n", encoding="utf-8")
        before = self.cell.read_bytes()
        store = ProfilePoseStore(self.tree, "drop_left", "Ablage links")
        self.assertTrue(store.write(TaughtPose.from_sample("drop_left", _sample(LOOK_1_DEG)), CLEAR))
        self.assertEqual(before, self.cell.read_bytes())

    def test_the_store_writes_no_pose_but_its_own(self) -> None:
        store = ProfilePoseStore(self.tree, "drop_left", "Ablage links")
        before = self.cell.read_bytes()
        refused = store.write(TaughtPose.from_sample("park", _sample(LOOK_1_DEG)), CLEAR)
        self.assertIn("'park'", refused)
        self.assertEqual(before, self.cell.read_bytes())


class TheWatchedSessionIsTheSessionTests(_Case):
    """The session ``HandGuide.wait`` samples is the arm's own, wrapped: every call but the sample's bookkeeping reaches
    the arm's session unchanged, so a guide that holds, asks whether it is free or leaves it does exactly that."""

    def test_every_call_reaches_the_arms_session(self) -> None:
        from src.robot.execution.teach import _Watched

        arm = self.arm()
        seen: list[FreedriveSample] = []
        gate = StillnessGate()
        with _Watched(arm.freedrive(), gate, self.clock, seen.append) as watched:
            watched.free()
            self.assertTrue(watched.is_free)
            sample = watched.sample()
            watched.hold()
            self.assertFalse(watched.is_free)
        self.assertEqual([sample], seen)
        self.assertEqual(["session", "free", "hold", "left, held"], self.steps())

    def test_the_view_the_wait_reads_is_the_guides_own_with_its_early_keys_in_order(self) -> None:
        from src.robot.execution.teach import _ViewKeys

        view = _View(["enter", "x", "finish", "skip"])
        keys = _ViewKeys(view)
        self.assertEqual(FINISH, keys.look(), "a finish read early holds")
        self.assertEqual(["enter", "x", "skip"], [keys.poll_key(), keys.poll_key(), keys.poll_key()],
                         "the keys read before it, then the view's own, in order")
        keys.guide(["HOLD THE ARM STILL"], "guide")
        self.assertIs(view.shown, keys.shown, "anything else is the view's own")
        with self.assertRaises(AttributeError):                # its own slots are never looked up on the view
            _ = keys._missing


class TheStillnessGateTests(unittest.TestCase):
    def test_it_is_hand_guidings_own_gate(self) -> None:
        gate = StillnessGate()
        self.assertEqual((STILL_RAD_S, STILL_FOR_S), (gate.still_rad_s, gate.still_for_s))

    def test_still_only_after_half_a_second_below_the_threshold(self) -> None:
        gate = StillnessGate()
        still = _sample(LOOK_1_DEG)
        self.assertFalse(gate.feed(still, 10.0))
        self.assertFalse(gate.feed(still, 10.49))
        self.assertTrue(gate.feed(still, 10.5))
        self.assertTrue(gate.still)

    def test_a_moving_sample_starts_it_again(self) -> None:
        gate = StillnessGate()
        still, moving = _sample(LOOK_1_DEG), _sample(LOOK_1_DEG, speed=0.3)
        gate.feed(still, 0.0)
        gate.feed(moving, 0.4)
        self.assertFalse(gate.feed(still, 0.6))
        self.assertFalse(gate.still)
        self.assertTrue(gate.feed(still, 1.1))
        gate.reset()
        self.assertFalse(gate.still)
        self.assertFalse(gate.feed(still, 1.2))

    def test_a_speed_at_the_threshold_is_not_still(self) -> None:
        gate = StillnessGate()
        at = _sample(LOOK_1_DEG, speed=STILL_RAD_S)
        gate.feed(at, 0.0)
        self.assertFalse(gate.feed(at, 5.0))


class TheResultSaysWhatHappenedTests(_Case):
    def test_it_renders_and_dumps_what_the_console_shows(self) -> None:
        self.arm(screen=GUARD_REFUSED)
        taught = self.teach()
        record = taught.to_dict()
        self.assertEqual("refused", record["outcome"])
        self.assertEqual("guard_refused", record["verdict"])
        self.assertEqual("drop_left", record["name"])
        self.assertEqual(6, len(record["joints_deg"]))
        self.assertEqual(6, len(record["nearby_deg"]))
        self.assertIn("drop_left", taught.render())
        self.assertEqual(taught.render(), str(taught))

    def test_every_outcome_renders_what_happened(self) -> None:
        self.arm()
        self.assertIn("written to config/robot/robot.cell.yaml", self.teach().render())
        self.arm(screen=UNSCREENED)
        self.assertIn("not written", self.teach().render())
        self.arm(_stand(LOOK_1_DEG, moving=0))
        self.assertIn("held at once", self.teach(_answers("q")).render())
        self.arm(_stand(LOOK_1_DEG, moving=0))
        held = self.teach([], hold_when_still=lambda: "time_limit")
        self.assertIn("held once the arm stood still (time_limit)", held.render())
        record = held.to_dict()
        self.assertEqual((None, None, None), (record["joints_deg"], record["verdict"], record["nearby_deg"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
