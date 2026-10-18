"""A planned arm over a transport that offers only what every transport must, plus a payload register (P0g slice 14).

The core tests run on a fake that offers all eleven transport Protocols, which is what ``URTransport`` offers. A Franka
or KUKA transport will offer less: no controller kinematics (``ControllerKinematics``), no Cartesian route
(``ControllerMotion``). This column builds the ur5e tree over a fake that offers ``ArmTransport`` and ``TakesPayload``,
drives every verb on both routes, and sorts what each does: it answers, it refuses typed, or it reaches an optional
transport verb with no guard. The last kind is either raised (an ``AttributeError`` out of a Protocol verb) or
swallowed into a refusal whose sentence is that ``AttributeError`` (the home box says the controller's forward
kinematics "could not be read" of a transport that has none). ``move_home`` is ``move_to_home``'s ``ok``
(``tests/robot/safety/test_a_refused_home_names_its_gate.py``), so it is read through its twin.

Those verbs are the counted list ``_UNGUARDED``, which may only shrink: P1 guards each, with a typed refusal naming the
Protocol the transport lacks or a route through the description's chain, and lowers the list in the same commit. A verb
that starts to reach an unguarded call fails here. The control column is the same arm over the full fake, whose
controller answers its kinematics: nothing there is unguarded, so the column's findings are the profile's. FR3 joins
in P1, with a description and a transport of its own.
"""

from __future__ import annotations

import contextlib
import functools
import unittest
from collections.abc import Callable
from typing import Any

from src.robot.core import JointPositions
from src.robot.core.camera_world import CameraWorldDecline
from src.robot.core.transport import ArmTransport, TakesPayload
from tests._arm_kit import HERE, kit_pose, kit_tree, planned_arm
from tests._fake_transport import chain_of, fake_transport

#: What every transport must offer, and the payload register a controller that takes a payload offers beside it.
_PROFILE = (ArmTransport, TakesPayload)
_DECLINE = CameraWorldDecline("transport-profile column: no camera world is wired to a fake")
_THERE = JointPositions([0.2, -1.5, 1.5, 0.0, 1.5, 0.0])

#: Every verb the column drives, by name; ``camera_world`` declined where the verb takes it, a block where it does not.
_VERBS: dict[str, Callable[[Any], Any]] = {
    "get_joint_positions": lambda arm: arm.get_joint_positions(),
    "get_tcp_pose": lambda arm: arm.get_tcp_pose(),
    "fk": lambda arm: arm.fk(JointPositions(HERE)),
    "ik": lambda arm: arm.ik(kit_pose()),
    "is_inside_workspace": lambda arm: arm.is_inside_workspace(kit_pose()),
    "move": lambda arm: arm.move(kit_pose(), camera_world=_DECLINE),
    "move linear": lambda arm: arm.move(kit_pose(), linear=True, camera_world=_DECLINE),
    "move_linear": lambda arm: arm.move_linear(kit_pose()),
    "move_to_joints": lambda arm: arm.move_to_joints(_THERE, camera_world=_DECLINE),
    "move_to_home": lambda arm: arm.move_to_home(),
    "stop": lambda arm: arm.stop(),
    "wait_until_steady": lambda arm: arm.wait_until_steady(),
    "attach_payload": lambda arm: arm.attach_payload(40.0),
    "detach_payload": lambda arm: arm.detach_payload(),
    "disconnect": lambda arm: arm.disconnect(),
}

#: (route, verb) -> the optional transport verb it reaches with no guard over ``_PROFILE``, measured 2026-09-24.
#: The list may only shrink: P1 guards a verb and removes its row in the same commit.
_UNGUARDED: dict[tuple[str, str], str] = {
    ("ik", "get_tcp_pose"): "read_tcp",
    ("ik", "fk"): "controller_fk",
    ("ik", "ik"): "controller_ik",
    ("ik", "move"): "controller_ik",
    ("ik", "move linear"): "controller_ik",
    ("ik", "move_linear"): "controller_ik",
    ("ik", "move_to_home"): "controller_fk",
    ("curobo", "get_tcp_pose"): "read_tcp",
    ("curobo", "fk"): "controller_fk",
    ("curobo", "ik"): "controller_ik",
    ("curobo", "move linear"): "read_tcp",
    ("curobo", "move_linear"): "read_tcp",
    ("curobo", "move_to_home"): "controller_fk",
    # Willy's own: stop() latches the halt on the link (HoldsAHaltLatch.request_halt), which a transport that holds no
    # latch lacks.
    ("ik", "stop"): "request_halt",
    ("curobo", "stop"): "request_halt",
}

_ROUTES = ("ik", "curobo")
#: The verbs that take no camera-world keyword, run inside a block instead.
_IN_A_BLOCK = frozenset({"move_linear", "move_to_home"})


def _arm(route: str, *, profile: tuple[type, ...] | None) -> Any:
    """The ur5e tree on ``route``, over the fake offering ``profile``, or over the full fake with the chain's
    kinematics and a controller IK that answers, which is the control."""
    if profile is None:
        fake = fake_transport(joints=HERE, kinematics=chain_of("ur5e"))
        fake.always("controller_ik", list(HERE))
    else:
        fake = fake_transport(offers=profile, joints=HERE)
    tree = kit_tree(route, safety={"self_collision": {"planner_margin_mm": 4.0}})
    return planned_arm(tree, transport=fake)


def unguarded_call(verb: Callable[[Any], Any], arm: Any, *, block: bool) -> str | None:
    """The transport verb ``verb`` reached with no guard, or ``None``: an ``AttributeError`` it raised naming a
    attribute the transport lacks, or a refusal whose sentence carries one."""
    inside = arm.without_camera_world("transport-profile column") if block else contextlib.nullcontext()
    try:
        with inside:
            out = verb(arm)
    except AttributeError as missing:
        # Only a name the transport lacks: an AttributeError from anywhere else is another fault.
        return missing.name if missing.obj is arm._transport else None
    except Exception:  # noqa: BLE001 (a typed refusal or a fault the verb raised: not this column's question)
        return None
    message = getattr(out, "message", None)
    if isinstance(message, str) and "AttributeError" in message:
        for name in ("read_tcp", "controller_fk", "controller_ik", "move_pose", "last_reject_status"):
            if f"no attribute '{name}'" in message:
                return name
    return None


@functools.lru_cache(maxsize=None)
def _measured(profile: tuple[type, ...] | None) -> dict[tuple[str, str], str]:
    """(route, verb) -> the unguarded transport verb it reached; measured once per profile, since every test reads it."""
    found: dict[tuple[str, str], str] = {}
    for route in _ROUTES:
        for name, verb in _VERBS.items():
            reached = unguarded_call(verb, _arm(route, profile=profile), block=name in _IN_A_BLOCK)
            if reached is not None:
                found[(route, name)] = reached
    return found


class TheProfileColumnTests(unittest.TestCase):
    def test_every_verb_that_reaches_an_unguarded_call_is_listed(self) -> None:
        measured = _measured(_PROFILE)
        grown = {key: name for key, name in measured.items() if _UNGUARDED.get(key) != name}
        self.assertEqual({}, grown, "(route, verb): a verb reaches a transport verb the profile does not offer, with "
                                    "no guard; guard it (a typed refusal naming the Protocol), do not list it")

    def test_the_list_is_the_measure(self) -> None:
        """A row a verb no longer earns would admit it again later without anyone deciding it."""
        measured = _measured(_PROFILE)
        stale = sorted(key for key in _UNGUARDED if key not in measured)
        self.assertEqual([], stale, "guarded now: remove the row")

    def test_the_full_transport_reaches_no_unguarded_call(self) -> None:
        """THE CONTROL: the same verbs on the same arm over the full fake, whose controller answers its kinematics,
        reach nothing unguarded, so what the column lists is the profile's and not the verbs' own faults."""
        self.assertEqual({}, _measured(None))

    def test_the_classifier_finds_a_raised_and_a_swallowed_call_and_nothing_else(self) -> None:
        """The positive control: a synthetic verb of each kind is found, and an AttributeError elsewhere is not."""
        from types import SimpleNamespace

        arm = _arm("ik", profile=_PROFILE)
        self.assertEqual("read_tcp", unguarded_call(lambda a: a._transport.read_tcp(), arm, block=False))
        swallowed = SimpleNamespace(message="refused: AttributeError: 'X' object has no attribute 'controller_fk'")
        self.assertEqual("controller_fk", unguarded_call(lambda a: swallowed, arm, block=False))
        self.assertIsNone(unguarded_call(lambda a: object().no_such_name, arm, block=False))  # type: ignore[attr-defined]
        self.assertIsNone(unguarded_call(lambda a: a.get_joint_positions(), arm, block=False))


if __name__ == "__main__":
    unittest.main()
