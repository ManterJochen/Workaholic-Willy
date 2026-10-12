"""The hand is judged where the arm's controller puts the flange, and the links on the nominal table (the owner,
2026-10-12).

The owner's UR10 CB3 is calibrated at the factory: its controller puts the flange 6.0 to 6.3 mm and 0.6 degrees from
where the nominal table puts it at the same joints (2026-10-08). The camera's boxes stand in the controller's world, an
eye in hand frame placed by the TCP the controller reports, and the exact guard placed the hand, its fingers and the
wrist camera by the nominal table: up to 6 mm off the world it judged them in, keeping 3 mm. The owner's answer, "Alles
am Flansch": everything that hangs from the flange where the controller puts it, the links by the nominal table, and no
feature of one vendor: an arm says where its controller puts its flange (``KnowsItsFlange``), a UR from its controller's
rows.

* a chain's flanges asked at once are each row's flange; the nominal chain's are ``ur_link_transforms_mm_many``'s to the
  bit, and a row asked alone is the row asked among others;
* the guard's chain is the nominal table's but for the flange, which is the arm's controller's where it says: an arm that
  does not say, a double that answers everything, an answer that raises, has the wrong shape or is not finite keep the
  nominal flange; a configuration placed alone is the one placed in a path, to the bit;
* the exact guard judges, the whole path judge, the planner's pair distances and the self envelope place the hand on the
  controller's flange; a box 7.5 mm beside the nominal fingers, 1.3 mm beside the controller's, is admitted on the nominal
  flange and refused on the controller's;
* the UR driver vouches for its controller's rows at connect, where they put the flange where the controller says it
  stands, and keeps the nominal table where they or its TCP cannot be read or the rows put it elsewhere; a disconnect
  forgets them.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.core import JointPositions, MotionCommand
from src.robot.core.arm_capabilities import KnowsItsFlange, controller_flanges_of
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.drivers.ur.connection import ControllerKinematics
from src.robot.drivers.ur.pose import URPose
from src.robot.safety import SafetyContext
from src.robot.safety._capsule import AxisAlignedBox
from src.robot.safety._ur_kinematics import (
    UR_DH_TABLES_M,
    URDhChain,
    arm_link_transforms_mm,
    arm_link_transforms_mm_many,
    ur_link_transforms_mm,
    ur_link_transforms_mm_many,
)
from src.robot.safety.planning.self_envelope import self_envelope
from tests.test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards import AWAY, _arm, _backend
from tests.test_a_line_is_solved_on_the_controller_s_own_kinematics import _DELTAS

#: Twice the calibration URSim was given for the line IK test: it puts the UR10's flange 7.0 mm and its TCP 6.7 mm from
#: the nominal table's at ``AWAY``, the size of the owner's.
_TWICE = {key: tuple(2.0 * v for v in values) for key, values in _DELTAS.items()}
_TCP_MM = 155.75


def _controller(model: str = "ur10", *, a2_off_m: float = 0.0) -> ControllerKinematics:
    """The kinematics info of a controller of ``model`` calibrated by ``_TWICE``; ``a2_off_m`` added to its a2."""
    table = UR_DH_TABLES_M[model]
    a = [row.a_m + da for row, da in zip(table, _TWICE["a_m"])]
    a[1] += a2_off_m
    return ControllerKinematics(
        theta_rad=tuple(_TWICE["theta_rad"]), a_m=tuple(a),
        d_m=tuple(row.d_m + dd for row, dd in zip(table, _TWICE["d_m"])),
        alpha_rad=tuple(row.alpha_rad + dal for row, dal in zip(table, _TWICE["alpha_rad"])),
        checksums=(0xFFFFFFFF,) * 6, calibration_status=0, port=30011,
    )


def _chain(rows: "ControllerKinematics | None" = None) -> URDhChain:
    rows = _controller() if rows is None else rows
    return URDhChain(theta_rad=rows.theta_rad, a_m=rows.a_m, d_m=rows.d_m, alpha_rad=rows.alpha_rad)


class _Says:
    """An arm whose controller puts its flange on ``chain`` (``KnowsItsFlange``)."""

    def __init__(self, chain: URDhChain) -> None:
        self.chain = chain

    def controller_flanges_mm(self, joints_rad: np.ndarray) -> np.ndarray:
        return self.chain.flanges_mm(joints_rad)


class _Answers:
    """An arm that answers ``answer`` whatever it is asked, or raises it."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer

    def controller_flanges_mm(self, joints_rad: np.ndarray) -> Any:
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _configs(count: int, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-np.pi, np.pi, (count, 6))


class AChainSaysWhereItPutsTheFlangeTests(unittest.TestCase):
    def test_every_row_is_its_flange_and_the_nominal_chain_is_the_tables_to_the_bit(self) -> None:
        configs = _configs(3000)
        chain = _chain()
        flanges = chain.flanges_mm(configs)
        for q, flange in zip(configs[:200], flanges[:200]):
            expected = chain.flange_m(q).copy()
            expected[:3, 3] *= 1000.0
            np.testing.assert_allclose(flange, expected, rtol=0.0, atol=1e-9)
        nominal = URDhChain.nominal("ur10")
        assert nominal is not None
        many = ur_link_transforms_mm_many("ur10", configs)
        assert many is not None
        self.assertTrue(np.array_equal(many[:, -1], nominal.flanges_mm(configs)))
        self.assertTrue(all(np.array_equal(chain.flanges_mm(configs[i:i + 1])[0], flanges[i]) for i in range(3000)))

    def test_rows_that_are_not_six_joints_each_are_refused(self) -> None:
        for joints in (np.zeros(6), np.zeros((2, 5))):
            with self.subTest(shape=joints.shape), self.assertRaises(ValueError):
                _chain().flanges_mm(joints)


class TheGuardsChainTests(unittest.TestCase):
    def test_the_links_are_the_tables_and_the_flange_the_controllers(self) -> None:
        q = np.asarray(AWAY, dtype=np.float64)
        arm = _Says(_chain())
        self.assertIsInstance(arm, KnowsItsFlange)
        placed, table = arm_link_transforms_mm("ur10", q, arm), ur_link_transforms_mm("ur10", q)
        assert placed is not None and table is not None
        for frame in range(6):
            self.assertTrue(np.array_equal(placed[frame], table[frame]), frame)
        self.assertTrue(np.array_equal(placed[-1], _chain().flanges_mm(q[None])[0]))
        self.assertAlmostEqual(7.0, float(np.linalg.norm(placed[-1][:3, 3] - table[-1][:3, 3])), delta=0.1)

    def test_an_arm_that_does_not_say_keeps_the_nominal_flange(self) -> None:
        q = np.asarray(AWAY, dtype=np.float64)
        table = ur_link_transforms_mm("ur10", q)
        assert table is not None
        for arm in (None, object(), MagicMock(), _Answers(RuntimeError("no rows")), _Answers(None),
                    _Answers(np.zeros((1, 3, 3))), _Answers(np.full((1, 4, 4), np.nan)), _Answers(np.zeros((2, 4, 4)))):
            with self.subTest(arm=type(arm).__name__):
                self.assertIsNone(controller_flanges_of(arm, q[None]))
                placed = arm_link_transforms_mm("ur10", q, arm)
                assert placed is not None
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(placed, table)))

    def test_a_configuration_placed_alone_is_the_one_placed_in_a_path(self) -> None:
        configs = _configs(1500, seed=11)
        arm = _Says(_chain())
        many = arm_link_transforms_mm_many("ur10", configs, arm)
        assert many is not None
        for index in range(1500):
            alone = arm_link_transforms_mm("ur10", configs[index], arm)
            assert alone is not None
            self.assertTrue(all(np.array_equal(many[index, frame], alone[frame]) for frame in range(7)), index)


class TheExactGuardJudgesTheHandOnTheControllersFlangeTests(unittest.TestCase):
    """The owner-like UR10 with the Hand-E at ``AWAY``, a seen box kept at 5 mm."""

    def setUp(self) -> None:
        self.arm = _arm()
        self.guard = self.arm._preflight._path_authority(self.arm)
        self.backend = _backend(self.arm)
        self.q = np.asarray(AWAY, dtype=np.float64)
        table = ur_link_transforms_mm("ur10", self.q)
        assert table is not None
        tip = np.array([0.0, 0.0, _TCP_MM, 1.0])
        nominal_tcp = (table[-1] @ tip)[:3]
        toward = (_chain().flanges_mm(self.q[None])[0] @ tip)[:3] - nominal_tcp
        # A 30 mm part 60 mm from the nominal TCP the way the controller's TCP stands from it: 7.5 mm from the nominal
        # fingers, 1.3 mm from the controller's (lever2_scene, 2026-10-12).
        self.box = AxisAlignedBox(center_mm=nominal_tcp + toward / np.linalg.norm(toward) * 60.0,
                                  half_extents_mm=np.array([15.0, 15.0, 15.0]), name="seen_00_part")
        self.guard.set_perceived_fixtures((self.box,))

    def _judged(self) -> Any:
        return self.guard.evaluate(SafetyContext(command=MotionCommand.MOVE_JOINTS, target_joints=JointPositions(self.q),
                                                 arm=self.arm))

    def test_a_box_the_nominal_fingers_clear_is_refused_where_the_controller_puts_them(self) -> None:
        self.assertTrue(self._judged().accepted)
        self.arm._flange_chain = _chain()
        decision = self._judged()
        self.assertFalse(decision.accepted)
        self.assertIn("seen_00_part", decision.message)

    def test_every_reader_of_the_guard_places_the_hand_there(self) -> None:
        self.arm._flange_chain = _chain()
        flange = _chain().flanges_mm(self.q[None])[0]
        with patch.object(self.backend, "evaluate", wraps=self.backend.evaluate) as judged:
            self._judged()
        self.assertTrue(np.array_equal(judged.call_args_list[0].args[0][-1], flange))
        path = np.stack([self.q, self.q + 0.01])
        with patch.object(self.backend, "first_suspect", wraps=self.backend.first_suspect) as whole:
            self.guard.first_suspect(self.arm, path, 2)
        self.assertTrue(np.array_equal(whole.call_args.args[0][:, -1], _chain().flanges_mm(path)))
        pairs = self.guard.exact_pairs(self.arm)
        assert pairs is not None
        with patch.object(self.backend, "distance_mm", wraps=self.backend.distance_mm) as measured:
            pairs.distance(list(self.q), "upper_arm", "lfinger")
        self.assertTrue(np.array_equal(measured.call_args.args[0][-1], flange))
        envelope = self_envelope(self.arm._preflight, self.arm, list(self.q))
        assert envelope is not None
        self.assertTrue(np.array_equal(np.asarray(envelope.frames_mm[-1]), flange))


_Q = list(AWAY)


def _connected(rows: "ControllerKinematics | None", *, fk_rows: "ControllerKinematics | None" = None,
               tcp: "list[float] | None" = None) -> URRobotArm:
    """The owner-like UR10 connected to a controller whose FK runs on ``fk_rows`` (``rows`` where not given) with a bare
    tool register, which reports ``rows`` as its kinematics info and ``tcp`` as its active TCP."""
    arm = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": "ur10"},
        "safety": {"payload": {"enforce": False}},
        "gripper": {"model": "robotiq_hande", "coupling_plates": [{"name": "adapter", "thickness_mm": 20.0}],
                    "tool_frame": {"source": "willy", "offset_mm": [0.0, 0.0, _TCP_MM],
                                   "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
    }))
    computes_on = fk_rows if fk_rows is not None else rows if rows is not None else _controller()
    flange = _chain(computes_on).flanges_mm(np.asarray([_Q]))[0]
    conn = MagicMock()
    conn.is_connected = True
    conn.get_joint_positions.return_value = list(_Q)
    conn.is_steady.return_value = True
    conn.fk_current.return_value = URPose.from_T(flange).to_ur_list()
    conn.controller_kinematics.return_value = rows
    conn.active_tcp_offset.return_value = [0.0] * 6 if tcp is None else tcp
    arm._conn = conn
    arm.connect()
    return arm


class TheURDriverVouchesForItsControllersRowsTests(unittest.TestCase):
    def test_rows_that_put_the_flange_where_the_controller_says_place_it_until_a_disconnect(self) -> None:
        arm = _connected(_controller())
        configs = _configs(50, seed=3)
        self.assertTrue(np.array_equal(arm.controller_flanges_mm(configs), _chain().flanges_mm(configs)))
        self.assertIsInstance(arm, KnowsItsFlange)
        arm.disconnect()
        self.assertIsNone(arm.controller_flanges_mm(configs))

    def test_rows_that_cannot_be_read_or_put_the_flange_elsewhere_keep_the_nominal_table(self) -> None:
        for why, arm in (("no rows", _connected(None)),
                         ("no TCP", _connected(_controller(), tcp=MagicMock())),
                         ("rows 1 mm off what the controller computes on",
                          _connected(_controller(a2_off_m=0.001), fk_rows=_controller()))):
            with self.subTest(why), self.assertLogs(arm.logger, level="WARNING") as said:
                arm._flange_chain = arm._vouched_flange_chain(_Q, URPose.from_ur_list(
                    arm._conn.fk_current()).to_T())
            self.assertIsNone(arm.controller_flanges_mm(np.asarray([_Q])), why)
            self.assertIn("by the nominal ur10 table", "\n".join(said.output), why)


if __name__ == "__main__":
    unittest.main()
