"""Tests for the H7.1 Isaac suction gripper driver + cup-mount spec (CI-safe, mock — no Isaac).

The on-box attach (cup mounts + bonds + holds through a lift) is validated by the ``run_suction_pick
--probe`` on-box runner; here we lock the vendor-neutral contract: the width<->vacuum reinterpretation,
the Protocol conformance (Gripper + ObjectDetectingGripper), and the spec defaults.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.willy_sim.suction_mount import SUCTION_CUP, SuctionCupSpec
from src.robot.core import Gripper, ObjectDetectingGripper, RobotConnectionError
from src.robot.grippers.sim import IsaacSuctionGripper


def _mock_gripper() -> IsaacSuctionGripper:
    g = IsaacSuctionGripper(session=object(), gripper_prim_path=None, mock_mode=True)
    g.connect()
    return g


def test_is_protocol_conformant() -> None:
    g = _mock_gripper()
    assert isinstance(g, Gripper)
    assert isinstance(g, ObjectDetectingGripper)  # opts into post-close verification


def test_width_band_defaults() -> None:
    g = _mock_gripper()
    assert g.min_width_mm == 0.0
    assert g.max_width_mm == pytest.approx(30.0)


def test_close_engages_vacuum_open_releases() -> None:
    g = _mock_gripper()
    g.close()
    assert g.is_object_detected() is True
    assert g.get_width_mm() == 0.0
    g.open()
    assert g.is_object_detected() is False
    assert g.get_width_mm() == pytest.approx(30.0)


def test_width_threshold_reinterpretation() -> None:
    g = IsaacSuctionGripper(
        session=object(), gripper_prim_path=None, mock_mode=True, vacuum_on_below_mm=5.0,
    )
    g.connect()
    g.set_width_mm(2.0)   # below threshold -> vacuum ON
    assert g.is_object_detected() is True
    g.set_width_mm(20.0)  # above threshold -> vacuum OFF
    assert g.is_object_detected() is False


def test_commands_require_connection() -> None:
    g = IsaacSuctionGripper(session=object(), gripper_prim_path=None, mock_mode=True)
    with pytest.raises(RobotConnectionError):
        g.set_width_mm(0.0)
    with pytest.raises(RobotConnectionError):
        g.is_object_detected()
    with pytest.raises(RobotConnectionError):
        g.get_width_mm()


def test_connect_is_idempotent() -> None:
    g = _mock_gripper()
    g.connect()  # second connect is a no-op
    assert g.is_connected is True
    g.disconnect()
    assert g.is_connected is False


def test_non_mock_requires_prim_path() -> None:
    g = IsaacSuctionGripper(session=object(), gripper_prim_path=None, mock_mode=False)
    with pytest.raises(RobotConnectionError):
        g.connect()


def test_cup_spec_defaults() -> None:
    assert SUCTION_CUP.name == "suction_cup"
    # The anchor sits ~the 2F-85 grasp centre along the wrist +Y approach (no separate cup body -> no spin).
    assert SUCTION_CUP.anchor_offset_mm == pytest.approx(132.0)
    assert SUCTION_CUP.max_grip_distance_mm > 0.0
    # forwardAxis (joint local Z) maps onto wrist +Y: a -90 deg rotation about wrist X (unit quat).
    assert abs(float(np.linalg.norm(SUCTION_CUP.forward_axis_rot_wxyz)) - 1.0) < 1e-6


def test_cup_spec_is_frozen() -> None:
    spec = SuctionCupSpec()
    with pytest.raises(Exception):  # noqa: B017 - dataclass(frozen=True) -> FrozenInstanceError
        spec.anchor_offset_mm = 99.0  # type: ignore[misc]


def test_cup_profiles_width_is_contact_diameter() -> None:
    from src.robot.grippers.sim import SLIM_SUCTION_CUP, STANDARD_SUCTION_CUP

    # max_width_mm is the cup contact diameter = 2 * cup_radius_mm.
    assert STANDARD_SUCTION_CUP.max_width_mm == pytest.approx(30.0)  # 15 mm radius
    assert SLIM_SUCTION_CUP.max_width_mm == pytest.approx(20.0)      # 10 mm radius (the finer cup)
    assert SLIM_SUCTION_CUP.cup_radius_mm < STANDARD_SUCTION_CUP.cup_radius_mm


def test_driver_takes_a_cup_profile() -> None:
    from src.robot.grippers.sim import SLIM_SUCTION_CUP

    g = IsaacSuctionGripper(session=object(), gripper_prim_path=None, profile=SLIM_SUCTION_CUP, mock_mode=True)
    g.connect()
    assert g.profile is SLIM_SUCTION_CUP
    assert g.max_width_mm == pytest.approx(20.0)  # derived from the slim profile


def test_suction_cups_registry() -> None:
    from src.willy_sim.grippers import SUCTION_CUPS

    assert set(SUCTION_CUPS) == {"standard", "slim"}
    assert SUCTION_CUPS["slim"].name == "slim"


def test_resolve_suction_cup_override_config_and_default() -> None:
    from src.robot.grippers.sim import SLIM_SUCTION_CUP, STANDARD_SUCTION_CUP
    from src.willy_sim.grippers import resolve_suction_cup

    # nothing set -> the standard cup
    assert resolve_suction_cup(None) is STANDARD_SUCTION_CUP
    # a CLI override wins
    assert resolve_suction_cup(None, override="slim") is SLIM_SUCTION_CUP

    # config field is honoured when no override
    class _Sim:
        suction_cup = "slim"

    assert resolve_suction_cup(_Sim()) is SLIM_SUCTION_CUP
    # override beats config
    assert resolve_suction_cup(_Sim(), override="standard") is STANDARD_SUCTION_CUP
    # an unknown key fails loudly (a typo never silently mounts the wrong cup)
    with pytest.raises(KeyError):
        resolve_suction_cup(None, override="nope")
