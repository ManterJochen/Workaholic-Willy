"""A planned arm connects itself: refused before the transport opens, the payload pushed, the tool frame derived (P0g
slice 15).

Until slice 15 connect was the UR driver's alone and ``PlannedArm.connect`` refused, so a vendor after the UR would
have written its own copy of every gate below. The neutral half is ``PlannedArm``'s now, and a vendor answers three
hooks of it (its address, its model, another model that fits better). These run it on the kit's planned arm over the
fake transport, whose controller answers its TCP from the description's chain with a tool register the test chooses
(``chain_of``, ``tcp_register``), and read what the controller was asked off the fake's tape.

What each part holds, in connect's order: a contradictory payload, a payload guard with no register to push to and an
undeclared tool frame are refused before ``connect`` reaches the transport; a payload push that fails, in any way, rolls
the connection back (the payload tests moved here from ``tests/test_ur_arm.py`` and ``tests/test_safety_payload.py``,
whose UR fixtures they were written on); the tool frame the controller applies is derived as
``inv(chain flange) @ controller TCP`` and refused, with the connection rolled back, when it is not the one the cell
declares; a transport that reports no controller TCP, and an arm that is moving, are refused the same way. The UR's own answers (its address, its size class,
its series twin) stay in ``tests/test_ur_tool_frame.py``, ``tests/test_ur_controller_identity.py`` and
``tests/test_ur_series_twin.py``.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.geometry.matrix import compare_tool_frames, position_quaternion_to_matrix
from src.robot.core import RobotConnectionError
from src.robot.core.transport import ArmTransport, ControllerKinematics, TakesPayload
from src.robot.drivers.planned.arm import PlannedArm
from tests._arm_kit import kit_tree, planned_arm
from tests._fake_transport import chain_of, fake_transport

_OFFSET = (0.0, 132.0, 0.0)
_QUAT = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
_TOOL = position_quaternion_to_matrix(np.asarray(_OFFSET), np.asarray(_QUAT))
_Q = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]
_BARE: dict[str, Any] = {"enforce": False}
_WEIGHED: dict[str, Any] = {"enforce": True, "mass_kg": 2.5, "cog_mm": (0.0, 0.0, 60.0)}


def _arm(source: str = "willy", *, payload: dict[str, Any] | None = None, holds_the_tool: bool | None = None,
         offers: tuple[type, ...] | None = None) -> tuple[PlannedArm, Any]:
    """A ur5e cell declaring the +Y tool frame under ``source``, over a fake controller at ``_Q`` whose register holds
    the tool (``holds_the_tool``, by default what ``source`` expects); the arm and its fake."""
    holds = (source == "controller") if holds_the_tool is None else holds_the_tool
    state: dict[str, Any] = {"joints": _Q, "kinematics": chain_of("ur5e"), "tcp_register": _TOOL if holds else None}
    fake = fake_transport(**state) if offers is None else fake_transport(offers=offers, **state)
    fake.connected = False
    tree = kit_tree("ik", safety={"payload": payload if payload is not None else _BARE},
                    gripper={"tool_frame": {"source": source, "offset_mm": _OFFSET, "rotation_quat_xyzw": _QUAT}})
    return planned_arm(tree, transport=fake), fake


def _verbs(fake: Any) -> list[str]:
    return [call.verb for call in fake.tape]


class RefusedBeforeTheTransportOpensTests(unittest.TestCase):
    """Nothing is asked of the controller: each refusal comes before ``connect`` reaches the transport."""

    def _refused(self, arm: PlannedArm, fake: Any, said: str) -> None:
        with self.assertRaises(RobotConnectionError) as caught:
            arm.connect()
        self.assertIn(said, str(caught.exception))
        self.assertEqual([], fake.tape, "the controller was asked something before the cell was refused")
        self.assertFalse(fake.is_connected)

    def test_an_undeclared_tool_frame_is_refused(self) -> None:
        fake = fake_transport(joints=_Q, kinematics=chain_of("ur5e"))
        fake.connected = False
        arm = planned_arm(kit_tree("ik", safety={"payload": _BARE}), transport=fake)
        self._refused(arm, fake, "gripper.tool_frame.source is 'undeclared'")

    def test_a_payload_guard_over_a_transport_with_no_payload_register_is_refused(self) -> None:
        arm, fake = _arm(payload=_WEIGHED, offers=(ArmTransport, ControllerKinematics))
        self._refused(arm, fake, "takes no payload register (TakesPayload)")

    def test_a_transport_that_reports_no_controller_tcp_is_refused_before_a_payload_is_written(self) -> None:
        """The tool frame of such a controller can never be checked, and connect would otherwise open the transport
        and write the payload into the controller before it found out, leaving the register behind it."""
        arm, fake = _arm(payload=_WEIGHED, offers=(ArmTransport, TakesPayload))
        self._refused(arm, fake, "reports no controller TCP (ControllerKinematics)")

    def test_the_control_the_same_cells_connect_once_the_contradiction_is_gone(self) -> None:
        """THE CONTROL: the tape is empty above because of the refusal, not because the fake records nothing."""
        for payload in (_BARE, _WEIGHED):
            with self.subTest(payload=payload):
                arm, fake = _arm(payload=payload)
                arm.connect()
                self.assertEqual("connect", _verbs(fake)[0])
                self.assertTrue(fake.is_connected)


def _cell(**payload: Any) -> tuple[PlannedArm, Any]:
    """A willy cell over a bare controller whose payload block is ``payload``."""
    return _arm(payload=payload)


class ConnectPayloadRefuseTests(unittest.TestCase):
    """S0 / decision D4: ``connect()`` must fail closed on ``enforce: true`` + ``mass_kg: 0.0``.

    That is the SHIPPED default, and pushing ``setPayload(0.0)`` to a real controller silently
    overwrites the payload a mounted tool was configured for -- fail-OPEN in the dangerous direction.
    These assert on BEHAVIOUR (did the socket open? was set_payload pushed?), not just that connect
    returned, because a guard that is accepted and does nothing is the exact failure this closes.
    Moved from tests/test_safety_payload.py in P0g slice 15, when connect became the planned arm's.
    """

    def test_enforce_true_zero_mass_refuses_before_opening_the_socket(self) -> None:
        arm, fake = _cell(enforce=True, mass_kg=0.0)
        with self.assertRaises(RobotConnectionError) as cm:
            arm.connect()
        self.assertEqual([], fake.calls("connect"))       # refused BEFORE opening anything
        self.assertEqual([], fake.calls("set_payload"))
        self.assertIn("mass_kg", str(cm.exception))

    def test_enforce_true_real_mass_connects_and_pushes(self) -> None:
        arm, fake = _cell(enforce=True, mass_kg=1.5, cog_mm=(0.0, 0.0, 40.0))
        arm.connect()
        self.assertEqual(1, len(fake.calls("connect")))
        self.assertEqual([(1.5, [0.0, 0.0, 40.0])], [call.args for call in fake.calls("set_payload")])

    def test_enforce_false_bare_flange_connects_without_pushing(self) -> None:
        arm, fake = _cell(enforce=False, mass_kg=0.0)
        arm.connect()
        self.assertEqual(1, len(fake.calls("connect")))
        self.assertEqual([], fake.calls("set_payload"))    # enforce:false never touches controller payload


class ConnectPayloadRollbackTests(unittest.TestCase):
    """`connect()` pushes the configured payload, and MUST roll the connection back if that push fails.

    The driver's own comment states the invariant: "an unverified payload on a connected controller is
    more dangerous than no connection at all" -- the controller's protective-stop model would then
    under-read the real mass. Measured 2026-08-09: the handler had NO test at all, and enumerated
    `(RuntimeError, OSError, ValueError)`, so several realistic failures skipped the rollback entirely
    and left the arm connected. A handler whose whole purpose is to fail closed must not enumerate
    failure types. Moved from tests/test_ur_arm.py in P0g slice 15.
    """

    @staticmethod
    def _arm_with_payload(mass_kg: float = 2.5) -> tuple[PlannedArm, Any]:
        return _cell(enforce=True, mass_kg=mass_kg, cog_mm=(0.0, 0.0, 60.0))

    def test_successful_push_keeps_the_connection(self) -> None:
        arm, fake = self._arm_with_payload()
        arm.connect()
        self.assertEqual([(2.5, [0.0, 0.0, 60.0])], [call.args for call in fake.calls("set_payload")])
        self.assertEqual([], fake.calls("disconnect"))

    def test_every_failure_type_rolls_the_connection_back(self) -> None:
        """AttributeError and TypeError are the realistic ones and are exactly what used to escape.

        AttributeError is what a missing or renamed `setPayload` raises, and TypeError is a signature change. If
        either skips the rollback, the arm stays connected with an unverified payload.
        """
        for exc_type in (AttributeError, TypeError, RuntimeError, OSError, ValueError):
            with self.subTest(exception=exc_type.__name__):
                arm, fake = self._arm_with_payload()
                fake.script("set_payload", exc_type("controller said no"))
                with self.assertRaises(RobotConnectionError) as ctx:
                    arm.connect()
                self.assertEqual(1, len(fake.calls("disconnect")))
                self.assertFalse(fake.is_connected)
                self.assertIn("setPayload", str(ctx.exception))
                self.assertIsInstance(ctx.exception.__cause__, exc_type)

    def test_a_keyboard_interrupt_mid_push_also_disconnects_and_propagates_unchanged(self) -> None:
        """Ctrl-C during the push leaves the same dangerous state. Disconnect, then let it through --
        it must NOT be reclassified as a RobotConnectionError, or an operator's abort looks like a
        hardware fault."""
        arm, fake = self._arm_with_payload()
        fake.script("set_payload", KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            arm.connect()
        self.assertEqual(1, len(fake.calls("disconnect")))

    def test_enforce_false_never_touches_the_controller_payload(self) -> None:
        """A genuinely bare flange is expressed by enforce: false, and then the arm must leave
        whatever the pendant has alone."""
        arm, fake = _cell(enforce=False, mass_kg=0.0)
        arm.connect()
        self.assertEqual([], fake.calls("set_payload"))
        self.assertEqual([], fake.calls("disconnect"))


class ConnectPayloadCogGateTests(unittest.TestCase):
    """`connect()` must refuse a declared mass with an unmeasured CoG.

    The mass gate's own message already instructs the operator to "set safety.payload.mass_kg AND
    cog_mm" -- and nothing checked the second half. Measured 2026-08-09: mass_kg 2.2 with the default
    cog_mm [0,0,0] connected happily and pushed `setPayload(2.2, (0.0, 0.0, 0.0))`, telling the
    controller a 2.2 kg tool is a point mass at the flange face.

    That is the shape this repo already removed once as worse-than-nothing (the `quality_threshold_mm`
    key that documented a gate which did not exist), sitting inside the very handler held up as the
    model of fail-closed config. The physical stake: at the UR3e's shipped max_mass_kg 3.0 and the
    132 mm tool this project ships, it is 3.88 N*m of wrist torque the controller does not model --
    in the protective-stop calculation AND in the gravity compensation behind get_tcp_wrench(), which
    the UR driver documents as the hand-over signal. Moved from tests/test_ur_arm.py in P0g slice 15.
    """

    def test_declared_mass_with_default_cog_is_refused_before_the_socket_opens(self) -> None:
        arm, fake = _cell(enforce=True, mass_kg=2.2)
        with self.assertRaises(RobotConnectionError) as ctx:
            arm.connect()
        self.assertEqual([], fake.calls("connect"))       # fail closed BEFORE touching the network
        self.assertEqual([], fake.calls("set_payload"))
        msg = str(ctx.exception)
        self.assertIn("cog_mm", msg)
        self.assertIn("enforce: false", msg)       # the message must name its own escape hatch

    def test_a_measured_cog_connects(self) -> None:
        arm, fake = _cell(enforce=True, mass_kg=2.2, cog_mm=(0.0, 0.0, 60.0))
        arm.connect()
        self.assertEqual([(2.2, [0.0, 0.0, 60.0])], [call.args for call in fake.calls("set_payload")])

    def test_any_nonzero_axis_counts_as_measured(self) -> None:
        """The gate detects the literal default, not 'small' -- it must not second-guess a real
        measurement that happens to be axis-centred on two of three axes."""
        for cog in ((3.0, 0.0, 0.0), (0.0, -4.5, 0.0), (0.0, 0.0, 1.0)):
            with self.subTest(cog=cog):
                arm, fake = _cell(enforce=True, mass_kg=1.0, cog_mm=cog)
                arm.connect()
                self.assertEqual([(1.0, list(cog))], [call.args for call in fake.calls("set_payload")])

    def test_the_zero_mass_gate_still_fires_first_and_unchanged(self) -> None:
        """Both gates refuse a fully-default payload block; the mass one owns that message."""
        arm, _ = _cell(enforce=True, mass_kg=0.0)
        with self.assertRaises(RobotConnectionError) as ctx:
            arm.connect()
        self.assertIn("mass_kg: 0.0", str(ctx.exception))

    def test_enforce_false_is_unaffected(self) -> None:
        arm, fake = _cell(enforce=False, mass_kg=2.2)
        arm.connect()
        self.assertEqual([], fake.calls("set_payload"))


class TheToolFrameIsDerivedFromTheChainTests(unittest.TestCase):
    def test_a_willy_cell_over_a_bare_controller_connects_and_keeps_the_declared_frame(self) -> None:
        arm, fake = _arm("willy")
        arm.connect()
        self.assertTrue(fake.is_connected)
        self.assertIn("controller_tcp_now", _verbs(fake))
        np.testing.assert_allclose(arm.active_tool_frame, _TOOL, atol=0.0)

    def test_a_controller_cell_keeps_the_frame_it_derived_and_forgets_it_at_disconnect(self) -> None:
        arm, fake = _arm("controller")
        self.assertIsNone(arm.active_tool_frame, "a frame was known before anything was derived")
        arm.connect()
        observed = arm.active_tool_frame
        assert observed is not None
        d_t, d_r = compare_tool_frames(observed, _TOOL)
        self.assertLess(d_t, 1e-6)
        self.assertLess(d_r, 1e-6)
        arm.disconnect()
        self.assertIsNone(arm.active_tool_frame)

    def test_a_cell_in_the_other_mode_than_its_controller_is_refused_and_rolled_back(self) -> None:
        for source, holds in (("willy", True), ("controller", False)):
            with self.subTest(source=source):
                arm, fake = _arm(source, holds_the_tool=holds)
                with self.assertRaises(RobotConnectionError) as caught:
                    arm.connect()
                self.assertIn(f"gripper.tool_frame.source is {source!r}", str(caught.exception))
                self.assertIn("132.0 mm", str(caught.exception))
                self.assertEqual("disconnect", _verbs(fake)[-1])
                self.assertIsNone(arm._observed_tool_frame)

    def test_a_refused_reconnect_forgets_the_frame_the_last_connect_derived(self) -> None:
        """Changes behaviour on purpose (after the P0g slice 15 review): a rollback disconnects the transport, and a
        frame derived from a controller that is no longer connected is not the frame it applies. Until then the two
        rollbacks called the transport's disconnect alone, so a controller cell refused on its second connect kept
        reporting the first connect's frame, against what ``active_tool_frame`` says of itself."""
        for refusal in ("the arm is moving", "the payload push fails"):
            with self.subTest(refusal=refusal):
                arm, fake = _arm("controller", payload=_WEIGHED)
                arm.connect()
                self.assertIsNotNone(arm.active_tool_frame, "the first connect derived no frame")
                if refusal == "the arm is moving":
                    fake.script("is_steady", False)
                else:
                    fake.script("set_payload", RuntimeError("controller said no"))
                with self.assertRaises(RobotConnectionError):
                    arm.connect()
                self.assertEqual("disconnect", _verbs(fake)[-1])
                self.assertIsNone(arm.active_tool_frame, "a disconnected controller cell still reports a frame")

    def test_a_moving_arm_is_refused_and_rolled_back(self) -> None:
        arm, fake = _arm()
        fake.script("is_steady", False)
        with self.assertRaises(RobotConnectionError) as caught:
            arm.connect()
        self.assertIn("while the arm is moving", str(caught.exception))
        self.assertEqual("disconnect", _verbs(fake)[-1])
        self.assertEqual([], fake.calls("controller_tcp_now"), "a moving arm's pose was paired with its joints")

    def test_the_context_manager_connects_and_disconnects(self) -> None:
        arm, fake = _arm()
        with arm as held:
            self.assertIs(arm, held)
            self.assertTrue(fake.is_connected)
        self.assertEqual("disconnect", _verbs(fake)[-1])


class TheChainIsTheDescriptionsTests(unittest.TestCase):
    """The flange the tool frame is derived from is the description's chain, read as standard DH, UR descriptions only."""

    def test_a_ur_description_gets_the_chain_the_fake_controller_answers_from(self) -> None:
        from src.robot.drivers.planned.tool_frame import described_chain

        ours, theirs = described_chain("UR5e"), chain_of("ur5e")
        assert ours is not None
        joints = np.asarray(_Q, dtype=np.float64)
        self.assertTrue(np.array_equal(ours.link_transforms_mm(joints)[-1], theirs.link_transforms_mm(joints)[-1]))

    def test_another_vendors_description_gets_no_chain(self) -> None:
        """As the UR loop it replaced: a description whose convention nobody stated is not read as a UR's."""
        from unittest import mock

        from src.config import arms
        from src.robot.drivers.planned import tool_frame

        foreign = arms.load_arm("ur5e").model_copy(update={"vendor": "franka"})
        tool_frame._chain_for.cache_clear()
        self.addCleanup(tool_frame._chain_for.cache_clear)
        with mock.patch.object(arms, "load_arm", return_value=foreign):
            self.assertIsNone(tool_frame.described_chain("a_franka_described_like_a_ur"))
            self.assertIsNone(tool_frame.derive_active_tool_frame("a_franka_described_like_a_ur", _Q, np.eye(4)))


class TheHooksAnswerNothingOnTheirOwnTests(unittest.TestCase):
    def test_a_planned_arm_with_no_vendor_asks_no_address_no_model_and_no_other_model(self) -> None:
        arm, _ = _arm()
        self.assertIsNone(arm._address_refusal())
        self.assertIsNone(arm._verify_controller_model())
        self.assertIsNone(arm._refuse_if_another_model_fits_better(list(_Q), np.eye(4), np.eye(4), 0.0))


if __name__ == "__main__":
    unittest.main()
