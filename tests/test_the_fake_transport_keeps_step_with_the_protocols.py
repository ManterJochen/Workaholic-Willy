"""The fake transport and the fake planner keep step with what they stand in for (P0g slice 5).

mypy does not check ``tests/`` (``ci.yml`` names ``src api datagen scripts examples willy``), so nothing but this file
holds a fake to the surface it doubles. A Protocol that gains a verb, a mixin that lacks one, a verb that only exists
as the Protocol's stub, a signature that drifts from the client's: each is a test that passes against the fake and
fails against the product, and each fails here first.
"""

from __future__ import annotations

import ast
import inspect
import typing
import unittest
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionStatus
from src.robot.core import transport as core_transport
from src.robot.core.transport import (
    ArmTransport,
    ControllerKinematics,
    PlannedArmTransport,
)
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from src.robot.safety.planning.curobo_client import CuroboPlanClient
from tests._fake_planner import FakePlanner
from tests._fake_transport import (
    MIXINS,
    NotScripted,
    Raises,
    Rejects,
    ScriptExhausted,
    chain_of,
    fake_transport,
)

_PLANNED = Path(__file__).resolve().parents[1] / "src" / "robot" / "drivers" / "planned"
_LINE = [[0.0] * 6, [0.1] * 6, [0.2] * 6]


def _members(protocol: type) -> set[str]:
    return set(typing._get_protocol_attrs(protocol))  # type: ignore[attr-defined]


def _drift(protocol: type, mixin: type) -> list[str]:
    """What ``mixin`` lacks or adds against ``protocol``, and every member it has only as a stub or with other
    parameters. Empty when the two keep step."""
    wanted = _members(protocol)
    found: list[str] = []
    for name in sorted(wanted):
        own = mixin.__dict__.get(name)
        if own is None:
            found.append(f"{name}: missing")
            continue
        if isinstance(own, property):
            continue
        want = [p for p in inspect.signature(getattr(protocol, name)).parameters if p != "self"]
        have = [p for p in inspect.signature(own).parameters if p != "self"]
        if want != have:
            found.append(f"{name}: parameters {have}, the Protocol's {want}")
    extra = {name for name in mixin.__dict__ if not name.startswith("_")} - wanted
    found += [f"{name}: not on the Protocol" for name in sorted(extra)]
    return found


class TheMixinsKeepStepWithTheProtocolsTests(unittest.TestCase):
    def test_every_transport_protocol_has_a_mixin(self) -> None:
        protocols = {value for value in vars(core_transport).values()
                     if inspect.isclass(value) and getattr(value, "_is_runtime_protocol", False)}
        self.assertEqual(protocols - {PlannedArmTransport}, set(MIXINS), "a Protocol the fake cannot offer")

    def test_each_mixin_has_exactly_its_protocols_verbs_as_real_members(self) -> None:
        for protocol, mixin in MIXINS.items():
            with self.subTest(protocol=protocol.__name__):
                self.assertEqual([], _drift(protocol, mixin))

    def test_a_protocol_that_gains_a_verb_is_found(self) -> None:
        """The positive control: a Protocol one verb wider than its mixin is drift, not a pass."""

        @runtime_checkable
        class _Wider(ArmTransport, Protocol):
            def home(self) -> None: ...

        self.assertEqual(["home: missing"], _drift(_Wider, MIXINS[ArmTransport]))

    def test_the_whole_fake_is_every_protocol_and_an_arm_only_one_is_one(self) -> None:
        whole = fake_transport()
        for protocol in (*MIXINS, PlannedArmTransport):
            with self.subTest(protocol=protocol.__name__):
                self.assertIsInstance(whole, protocol)
        bare = fake_transport(offers=(ArmTransport,))
        self.assertIsInstance(bare, ArmTransport)
        for protocol in (p for p in MIXINS if p is not ArmTransport):
            with self.subTest(bare=protocol.__name__):
                self.assertNotIsInstance(bare, protocol)

    def test_a_fake_without_arm_transport_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "every transport offers ArmTransport"):
            fake_transport(offers=(ControllerKinematics,))


def _client_members_the_planner_reaches() -> set[str]:
    """Every ``client.<name>`` and ``getattr(client, "<name>")`` the planned package reads, by AST."""
    names: set[str] = set()
    for path in sorted(_PLANNED.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "client":
                names.add(node.attr)
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "_client"):
                names.add(node.attr)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
                    and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id == "client"
                    and isinstance(node.args[1], ast.Constant)):
                names.add(str(node.args[1].value))
    return names


class TheFakePlannerKeepsStepWithTheClientTests(unittest.TestCase):
    def test_it_offers_every_member_the_planner_reaches(self) -> None:
        reached = _client_members_the_planner_reaches()
        self.assertGreaterEqual(len(reached), 8, "a scan that finds the planner reaching no client is blind")
        planner = FakePlanner()
        self.assertEqual(set(), {name for name in reached if not hasattr(planner, name)})

    def test_its_methods_take_what_the_client_takes(self) -> None:
        for name, own in vars(FakePlanner).items():
            if name.startswith("_") or not callable(own) or name in {"script", "calls_of"}:
                continue
            with self.subTest(method=name):
                real = getattr(CuroboPlanClient, name)
                self.assertEqual([p for p in inspect.signature(real).parameters],
                                 [p for p in inspect.signature(own).parameters])

    def test_the_glue_doubles_take_what_the_glue_takes(self) -> None:
        """The planner doubles the route tests stand in for ``CuroboPlanner`` (the arm's ``_curobo``) take exactly the
        arguments the glue takes, so a route that asked for something the glue cannot give fails in the double too (the
        review of the straight-line-first port, 2026-09-25: a double that took more hid a wrong call)."""
        from src.robot.drivers.planned.curobo_motion import CuroboPlanner
        from tests._arm_kit import PlannerDouble, RecordingPlanner
        from tests._route_planner import RoutePlanner

        def shape(function: Any) -> list[tuple[Any, Any]]:
            # The positional names are the double's own; the keyword-only names are the contract.
            return [(p.kind, p.name if p.kind is inspect.Parameter.KEYWORD_ONLY else None)
                    for p in inspect.signature(function).parameters.values()]

        checked = 0
        for double in (RoutePlanner, PlannerDouble, RecordingPlanner):
            for name in ("plan_joint", "check_joint_path", "execute", "refresh_world"):
                own = vars(double).get(name)
                if own is None:
                    continue
                checked += 1
                with self.subTest(double=double.__name__, verb=name):
                    self.assertEqual(shape(getattr(CuroboPlanner, name)), shape(own))
        self.assertGreaterEqual(checked, 7, "a check over no double passes loudest")

    def test_it_says_who_it_is_as_the_committed_descriptor_does(self) -> None:
        from src.robot.safety.planning.curobo_client import SidecarIdentity

        identity = SidecarIdentity.from_client(FakePlanner())
        self.assertEqual("ur5e", identity.provenance["arm"])
        self.assertEqual(["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                          "wrist_2_joint", "wrist_3_joint"], FakePlanner().joint_names)

    def test_a_pose_plan_is_scripted_or_none_and_a_joint_plan_is_a_line(self) -> None:
        planner = FakePlanner()
        self.assertIsNone(planner.plan([0.0] * 6, [0.3, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0]))
        line = planner.plan_joint([0.0] * 6, [0.4] * 6)
        assert line is not None
        self.assertEqual(5, len(line))
        self.assertEqual([0.0] * 6, line[0])
        self.assertEqual([0.4] * 6, line[-1])
        planner.script("plan", [[0.0] * 6, [0.1] * 6])
        self.assertEqual([[0.0] * 6, [0.1] * 6], planner.plan([0.0] * 6, [0.3, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0]))
        with self.assertRaises(ScriptExhausted):
            planner.plan([0.0] * 6, [0.3, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])


class TheFakeTransportAnswersAsItSaysTests(unittest.TestCase):
    def test_the_tape_keeps_every_argument_unrounded(self) -> None:
        fake = fake_transport()
        fake.run_trajectory([[0.1 + 0.2] * 6], vel=0.25, acc=None)
        call = fake.calls("run_trajectory")[0]
        self.assertEqual(0.30000000000000004, call.args[0][0][0])
        self.assertEqual({"vel": 0.25, "acc": None}, call.kwargs)

    def test_a_rejected_waypoint_is_sent_and_nothing_after_it(self) -> None:
        fake = fake_transport()
        fake.script("run_trajectory", Rejects(at=1))
        self.assertFalse(fake.run_trajectory(_LINE, vel=None, acc=None))
        self.assertEqual(_LINE[:2], fake.sent)
        self.assertEqual(_LINE[0], fake.read_joints(), "the arm stands at the last waypoint it took")

    def test_a_value_error_means_the_waypoint_was_not_sent(self) -> None:
        fake = fake_transport()
        fake.script("run_trajectory", Raises(ValueError("speed out of range"), at=1))
        with self.assertRaises(ValueError):
            fake.run_trajectory(_LINE, vel=-1.0, acc=None)
        self.assertEqual(_LINE[:1], fake.sent)

    def test_a_script_past_the_end_of_the_path_is_a_script_error(self) -> None:
        fake = fake_transport()
        fake.script("run_trajectory", Rejects(at=3))
        with self.assertRaisesRegex(AssertionError, "waypoint 3 of a 3-waypoint"):
            fake.run_trajectory(_LINE, vel=None, acc=None)

    def test_an_exhausted_script_raises_rather_than_falling_back(self) -> None:
        fake = fake_transport()
        fake.script("wait_until_steady", False)
        self.assertFalse(fake.wait_until_steady(1.0, 0.01))
        with self.assertRaises(ScriptExhausted):
            fake.wait_until_steady(1.0, 0.01)

    def test_an_answer_given_always_answers_every_call_after_the_queue(self) -> None:
        fake = fake_transport()
        fake.script("wait_until_steady", False)
        fake.always("wait_until_steady", True)
        self.assertEqual([False, True, True], [fake.wait_until_steady(1.0, 0.01) for _ in range(3)])
        fake.always("read_joints", RuntimeError("receive is down"))
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "receive is down"):
                fake.read_joints()
        with self.assertRaises(AttributeError):
            fake.always("is_connected", True)

    def test_a_responder_answers_from_what_the_call_asked(self) -> None:
        fake = fake_transport()
        fake.respond("controller_fk", lambda joints, label="": Pose(
            position_mm=np.array([joints[0] * 100.0, 0.0, 0.0]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE))
        self.assertEqual(200.0, fake.controller_fk([2.0] * 6, label="at").position_mm[0])
        self.assertEqual("at", fake.controller_fk([1.0] * 6, label="at").label)
        fake.respond("run_trajectory", lambda path, vel, acc: Rejects(at=0) if path[0][0] > 0.5 else True)
        self.assertTrue(fake.run_trajectory([[0.1] * 6], vel=None, acc=None))
        self.assertFalse(fake.run_trajectory([[0.9] * 6], vel=None, acc=None))
        fake.script("run_trajectory", False)
        self.assertFalse(fake.run_trajectory([[0.1] * 6], vel=None, acc=None), "a queued script answers first")

    def test_ik_finds_no_solution_unless_scripted(self) -> None:
        fake = fake_transport()
        pose = Pose(position_mm=np.array([300.0, 0.0, 200.0]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                    frame=Frame.BASE)
        self.assertIsNone(fake.controller_ik(pose))
        fake.script("controller_ik", [0.1] * 6)
        self.assertEqual([0.1] * 6, fake.controller_ik(pose, seed=[0.0] * 6))
        self.assertEqual({"seed": [0.0] * 6}, fake.calls("controller_ik")[1].kwargs)

    def test_kinematics_are_scripted_unless_a_chain_is_handed_in(self) -> None:
        with self.assertRaises(NotScripted):
            fake_transport().controller_fk([0.0] * 6)
        q = [0.3, -1.2, 1.1, -1.4, -1.5, 0.2]
        pose = fake_transport(kinematics=chain_of("ur5e")).controller_fk(q, label="here")
        flange = ur_link_transforms_mm("ur5e", np.asarray(q))
        assert flange is not None
        self.assertTrue(np.array_equal(flange[-1][:3, 3], pose.position_mm), "the chain's flange, bit for bit")
        self.assertEqual("here", pose.label)
        self.assertIs(Frame.BASE, pose.frame)

    def test_a_rejected_cartesian_move_says_why(self) -> None:
        fake = fake_transport()
        pose = Pose(position_mm=np.zeros(3), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=Frame.BASE)
        fake.script("move_pose", Rejects(status=MotionStatus.IK_FAILED), True)
        self.assertFalse(fake.move_pose(pose, linear=False, vel=None, acc=None, register=True, workspace_pose=pose))
        self.assertIs(MotionStatus.IK_FAILED, fake.last_reject_status)
        self.assertTrue(fake.move_pose(pose, linear=False, vel=None, acc=None, register=True, workspace_pose=pose))
        self.assertIsNone(fake.last_reject_status)

    def test_a_script_nothing_reads_is_refused(self) -> None:
        fake = fake_transport(offers=(ArmTransport,))
        for verb in ("is_connected", "last_reject_status", "controller_fk", "home"):
            with self.subTest(verb=verb), self.assertRaises(AttributeError):
                fake.script(verb, True)

    def test_a_write_that_would_create_a_name_on_it_is_refused(self) -> None:
        fake = fake_transport()
        fake.joints = [0.0] * 6
        with self.assertRaisesRegex(AttributeError, "control nothing"):
            fake.joint = [0.0] * 6


class APlannedArmRunsOverTheFakeTests(unittest.TestCase):
    def test_a_joint_move_on_the_transports_route_asks_the_steady_gate_then_sends_one_trajectory(self) -> None:
        from tests.test_a_planned_arm_is_built_by_keyword import _pieces

        fake = fake_transport(kinematics=chain_of("ur5e"))
        pieces = _pieces(fake)
        pieces["config"] = pieces["config"].model_copy(update={"motion": pieces["config"].motion.model_copy(
            update={"planner": "ik"})})
        from src.robot.drivers.planned.arm import PlannedArm

        arm = PlannedArm(**pieces)
        goal = [0.1, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        result = arm.move_to_joints(JointPositions(goal))
        self.assertTrue(result.ok, result.render())
        self.assertEqual(["wait_until_steady", "run_trajectory"],
                         [c.verb for c in fake.tape if c.verb in {"wait_until_steady", "run_trajectory"}])
        self.assertEqual([goal], fake.sent)

    def test_a_bare_planned_arm_refuses_a_write_that_would_create_a_name(self) -> None:
        from tests.test_a_planned_arm_is_built_by_keyword import _arm

        arm: Any = _arm()
        arm._home_joints = [0.0] * 6
        with self.assertRaisesRegex(AttributeError, "control nothing"):
            arm._conn = object()


if __name__ == "__main__":
    unittest.main()
