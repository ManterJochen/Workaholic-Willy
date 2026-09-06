"""Every Gripper implementation conforms to the Gripper Protocol -- returns AND parameters.

Return types came first (iteration 2). The PARAMETER half was added 2026-08-09 after the same class
of drift bit again, on the half these tests did not cover.

``@runtime_checkable`` validates member *presence* only, not signatures, which
is how ``GripperController`` drifted to returning ``int`` from ``set_width_mm``
(Protocol: ``None``) and ``get_width_mm`` (Protocol: ``float``). These tests lock
the return types of all three implementations so the drift cannot recur.
"""

from __future__ import annotations

import typing
import unittest

from src.config.schema.robot.robot_schema import RobotConfig
from src.robot.core.gripper import Gripper
from src.robot.grippers.dummy import DummyGripper
from src.robot.grippers.null import NullGripper
from src.robot.grippers.robotiq import GripperController
from src.robot.grippers.vacuum import VacuumGripper

_NONE = type(None)


def _return_hint(cls: type, method: str) -> object:
    return typing.get_type_hints(getattr(cls, method)).get("return")


def _accepts_none(cls: type, method: str, param: str) -> bool:
    """Does ``cls.method`` declare ``param`` as optional (``X | None``)?"""
    hint = typing.get_type_hints(getattr(cls, method)).get(param)
    return _NONE in typing.get_args(hint) if hint is not None else False


class GripperReturnTypeConformanceTests(unittest.TestCase):
    IMPLS = (GripperController, DummyGripper, NullGripper)

    def test_protocol_baseline(self) -> None:
        self.assertIs(_return_hint(Gripper, "set_width_mm"), _NONE)
        self.assertIs(_return_hint(Gripper, "get_width_mm"), float)

    def test_set_width_mm_returns_none(self) -> None:
        for impl in self.IMPLS:
            with self.subTest(impl=impl.__name__):
                self.assertIs(_return_hint(impl, "set_width_mm"), _NONE)

    def test_get_width_mm_returns_float(self) -> None:
        for impl in self.IMPLS:
            with self.subTest(impl=impl.__name__):
                self.assertIs(_return_hint(impl, "get_width_mm"), float)

    def test_optional_speed_and_force_are_honoured_by_every_implementation(self) -> None:
        """The Protocol declares ``speed``/``force`` as ``float | None``, and an implementation that
        narrows them to a plain ``float`` is NOT conformant -- even though it still type-checks at every
        call site that passes a number.

        MEASURED 2026-08-09, on real-hardware code: GraspExecutionPolicy.close_speed / close_force_n
        default to None (execution_policy.py:170-171) and are passed EXPLICITLY (:283-287), and nothing
        in the repo sets them. GripperController declared ``speed: float = 1.0``, so the caller's
        explicit None overrode the default and the first commanded close on a real UR raised
        ``TypeError: float() argument must be ... not 'NoneType'``. The sim and vacuum grippers honoured
        the contract; Robotiq -- the one on the September cell -- did not.
        """
        for impl in (*self.IMPLS, VacuumGripper):
            for param in ("speed", "force"):
                with self.subTest(impl=impl.__name__, param=param):
                    self.assertTrue(
                        _accepts_none(impl, "set_width_mm", param),
                        f"{impl.__name__}.set_width_mm narrows {param!r} to a non-optional type; the "
                        f"Gripper Protocol declares it 'float | None' and callers do pass None.",
                    )

    def test_robotiq_open_close_also_accept_none(self) -> None:
        """open()/close() are robotiq conveniences that forward straight into set_width_mm, so a
        narrowed signature here reaches the same vendor arithmetic by a different door."""
        for method in ("open", "close"):
            for param in ("speed", "force"):
                with self.subTest(method=method, param=param):
                    self.assertTrue(_accepts_none(GripperController, method, param))

    def test_robotiq_open_close_return_none(self) -> None:
        # open/close are robotiq conveniences (not in the Protocol) that used to
        # leak set_width_mm's int return; lock them to None as well.
        self.assertIs(_return_hint(GripperController, "open"), _NONE)
        self.assertIs(_return_hint(GripperController, "close"), _NONE)


class _FakeDriver:
    """Minimal robotiq-like driver seam (no SDK, no hardware)."""

    def __init__(self) -> None:
        self._pos = 0

    def connect(self, ip: str, port: int) -> None:
        return None

    def activate_if_needed(self) -> None:
        return None

    def move(self, count: int, speed: int, force: int) -> None:
        self._pos = count

    def get_current_position(self) -> int:
        return self._pos

    def disconnect(self) -> None:
        return None


class GripperControllerRuntimeReturnTests(unittest.TestCase):
    def _gripper(self) -> GripperController:
        from src.config.schema.robot import GripperConfig

        cfg = GripperConfig(min_width_mm=0.0, max_width_mm=150.0)
        g = GripperController(cfg, ip="127.0.0.1", driver_factory=_FakeDriver)
        g.connect()
        g.activate()
        return g

    def test_runtime_return_values_match_contract(self) -> None:
        g = self._gripper()
        self.assertIsNone(g.set_width_mm(50.0))
        width = g.get_width_mm()
        self.assertIsInstance(width, float)
        self.assertIsNone(g.open())
        self.assertIsNone(g.close())


class RobotiqCountMapTests(unittest.TestCase):
    """S0: the count map is anchored on the PHYSICAL travel (0 mm closed .. max_width_mm open),
    NOT on ``min_width_mm``. Before the fix, ``min_width_mm=5`` / ``max_width_mm=150`` sent a
    commanded 40 mm to count 193 -> ~20 mm on a real 2F-85. These pin the physical anchoring so it
    cannot silently regress.
    """

    @staticmethod
    def _g(min_mm: float, max_mm: float) -> GripperController:
        from src.config.schema.robot import GripperConfig

        return GripperController(
            GripperConfig(vendor="robotiq", min_width_mm=min_mm, max_width_mm=max_mm), ip="127.0.0.1"
        )

    def test_map_endpoints_are_physical(self) -> None:
        g = self._g(5.0, 85.0)
        self.assertEqual(g._mm_to_count(0.0), 255)     # fingers touching
        self.assertEqual(g._mm_to_count(85.0), 0)      # fully open
        self.assertEqual(g._count_to_mm(255), 0.0)
        self.assertEqual(g._count_to_mm(0), 85.0)

    def test_commanded_width_round_trips_exactly(self) -> None:
        g = self._g(5.0, 85.0)
        for w in (5.0, 20.0, 40.0, 60.0):
            self.assertAlmostEqual(g._count_to_mm(g._mm_to_count(w)), w, delta=0.5)

    def test_min_width_mm_does_not_shift_the_map(self) -> None:
        # Two configs differing only in the policy floor must produce the SAME count for a given mm.
        a, b = self._g(0.0, 85.0), self._g(5.0, 85.0)
        for w in (10.0, 40.0, 70.0):
            self.assertEqual(a._mm_to_count(w), b._mm_to_count(w))

    def test_clamp_still_applies_the_policy_floor(self) -> None:
        g = self._g(5.0, 85.0)
        self.assertEqual(g._clamp_width_mm(3.0), 5.0)   # below floor -> floor
        self.assertEqual(g._clamp_width_mm(40.0), 40.0)  # in range -> unchanged
        self.assertEqual(g._clamp_width_mm(99.0), 85.0)  # above max -> max


if __name__ == "__main__":
    unittest.main()


class RobotiqOptionalSpeedForceRuntimeTests(unittest.TestCase):
    """The end-to-end repro, not just the signature: what the policy sends must reach the driver."""

    def _gripper(self) -> tuple[GripperController, object]:
        from unittest.mock import MagicMock

        drv = MagicMock()
        drv.is_active.return_value = True
        from src.config.schema.robot import RobotConfig
        g = GripperController(RobotConfig().gripper, ip="127.0.0.1", driver_factory=lambda: drv)
        g.connect()
        g._activated = True
        return g, drv

    def test_none_speed_and_force_do_not_raise(self) -> None:
        """Exactly what GraspExecutionPolicy sends when nobody configures close_speed/close_force_n."""
        g, drv = self._gripper()
        g.set_width_mm(40.0, speed=None, force=None)   # used to raise TypeError
        self.assertTrue(drv.move.called)

    def test_none_resolves_to_the_documented_driver_defaults(self) -> None:
        """None must mean 'the driver default', not 0 -- a zero force would command no grip at all."""
        g, drv = self._gripper()
        g.set_width_mm(40.0, speed=1.0, force=0.5)
        explicit = drv.move.call_args
        g.set_width_mm(40.0, speed=None, force=None)
        self.assertEqual(drv.move.call_args, explicit)

    def test_open_and_close_accept_none_at_runtime(self) -> None:
        g, drv = self._gripper()
        g.close(speed=None, force=None)
        g.open(speed=None, force=None)
        self.assertTrue(drv.move.called)


class ActivationHappensAtConnectTests(unittest.TestCase):
    """Robotiq activation is a CALIBRATION -- the fingers travel their full range -- so WHEN it happens
    is a physical question, not a lifecycle detail.

    Until 2026-08-09 `connect()` only opened the socket and activation fired lazily on the first
    commanded width (`set_width_mm` logged "called before activate(); activating now"). On a real cell
    the first commanded width IS the grasp: the arm has descended, the fingers are around the object,
    and the gripper would pick that moment to sweep its travel. Now it happens at connect, which the
    bring-up order (real_cell connects BEFORE any motion) puts in the arm's starting pose.
    """

    @staticmethod
    def _driver():
        from unittest.mock import MagicMock as _MM

        drv = _MM()
        drv.is_active.return_value = True
        return drv

    def _gripper(self, drv):
        from src.robot.grippers.robotiq import GripperController

        return GripperController(RobotConfig().gripper, ip="127.0.0.1", driver_factory=lambda: drv)

    def test_connect_activates(self) -> None:
        drv = self._driver()
        g = self._gripper(drv)
        g.connect()
        self.assertTrue(drv.activate_if_needed.called or drv.activate.called)

    def test_the_first_commanded_close_no_longer_activates(self) -> None:
        """The whole point: by the time a width is commanded, the calibration sweep is already done."""
        drv = self._driver()
        g = self._gripper(drv)
        g.connect()
        drv.reset_mock()
        g.set_width_mm(40.0)
        self.assertFalse(drv.activate_if_needed.called or drv.activate.called)
        self.assertTrue(drv.move.called)

    def test_a_failed_activation_rolls_the_connection_back(self) -> None:
        """A connected-but-unactivated gripper accepts commands and does not grip -- which reads as a
        bad grasp rather than an unconfigured tool. Fail closed instead."""
        drv = self._driver()
        drv.activate_if_needed.side_effect = RuntimeError("no 24 V on the tool connector")
        drv.activate.side_effect = RuntimeError("no 24 V on the tool connector")
        g = self._gripper(drv)
        with self.assertRaises(RuntimeError):
            g.connect()
        self.assertFalse(g.is_connected)
        self.assertTrue(drv.disconnect.called)

    def test_the_lazy_path_survives_as_a_net(self) -> None:
        """A gripper constructed outside the bring-up order must still work -- just not silently."""
        drv = self._driver()
        g = self._gripper(drv)
        g._driver = drv          # connected without going through connect()
        g.set_width_mm(40.0)
        self.assertTrue(drv.activate_if_needed.called or drv.activate.called)
