"""The singularity check before a move runs on the arm's own DH chain where the cell asks, with the controller's verdict
(``robot.safety.ik_quality.singularity_fk: dh``, the owner, 2026-10-09).

Before every ``moveL`` (and every ``moveJ`` ``MotionController.move_to`` sends) the check estimates the TCP's Jacobian
by central differences of a forward kinematics and refuses below a least singular value of 0.005 or above a condition
number of 250. The controller's FK costs 12 round trips, 0.4 s before every line on the cell. With ``dh`` the same
differences run on the arm's own chain here: the controller's calibrated rows as its primary interface reports them, or
the model's nominal table where the controller's own FK confirms it is its chain, times the controller's active TCP.

What this file pins:

* on URSim CB3 (UR10) with the owner's TCP set on the controller, nominal and calibrated, the chain's verdict is the
  controller's own on the 21 line ends the cell checked on 2026-10-07/08 and on 156 configurations of lines toward the
  wrist, elbow and shoulder singularities, taken around where the verdict flips and down to 1e-7 rad of it
  (``tests/data/singularity_fk/ursim_cb3_line_ends.json``, recorded through the controller's own FK); not one round trip
  is made;
* the cell's line ends keep the thresholds 47 to 70 times over on the least singular value;
* the nominal table stands in only where the controller's FK is the table's, once per connection, and a calibrated
  controller whose rows cannot be read is asked as always;
* an unreadable TCP, no chain at all, or a connection that offers neither keep the controller's FK;
* the controller's FK, 12 round trips, stays the default, and the key loads.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from src.robot.core import JointPositions
from src.robot.drivers.ur.connection import ControllerKinematics
from src.robot.drivers.ur.motion import MotionController
from src.robot.safety._ur_ik import ur_pose_matrix_m
from src.robot.safety._ur_kinematics import URDhChain
from src.robot.safety.singularity import analyze_joint_singularity

_FIXTURE = Path(__file__).parent / "data" / "singularity_fk" / "ursim_cb3_line_ends.json"


def _recorded() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _rows(variant: dict[str, Any]) -> ControllerKinematics:
    k = variant["kinematics"]
    return ControllerKinematics(theta_rad=tuple(k["theta_rad"]), a_m=tuple(k["a_m"]), d_m=tuple(k["d_m"]),
                                alpha_rad=tuple(k["alpha_rad"]), checksums=tuple(k["checksums"]),
                                calibration_status=k["calibration_status"], port=30011)


def _chain_of(rows: ControllerKinematics) -> URDhChain:
    return URDhChain(theta_rad=rows.theta_rad, a_m=rows.a_m, d_m=rows.d_m, alpha_rad=rows.alpha_rad)


def _pose_vector(T: np.ndarray) -> list[float]:
    """A 4x4 in metres as the ``[x, y, z, rx, ry, rz]`` a UR answers, its rotation vector by the matrix's own axis."""
    R = T[:3, :3]
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0)))
    if angle < 1e-12:
        axis = np.zeros(3)
    elif angle > math.pi - 1e-6:
        B = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(B)))
        axis = B[:, i] / math.sqrt(B[i, i]) * angle
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * angle / (2.0 * math.sin(angle))
    return [float(v) for v in T[:3, 3]] + [float(v) for v in axis]


class _Controller:
    """The connection as the check reads it: rows and a TCP it reports, and an FK it counts, on ``fk_chain``."""

    def __init__(self, rows: "ControllerKinematics | None", tcp: "list[float] | None", fk_chain: "URDhChain | None",
                 fk_tcp: "list[float] | None" = None) -> None:
        self.rows, self.tcp, self.fk_chain = rows, tcp, fk_chain
        self.fk_tcp = fk_tcp if fk_tcp is not None else tcp
        self.fk_calls = 0
        self.rows_asked = 0
        self.kinematics_epoch = 0

    def controller_kinematics(self) -> "ControllerKinematics | None":
        self.rows_asked += 1
        return self.rows

    def active_tcp_offset(self) -> "list[float] | None":
        return None if self.tcp is None else list(self.tcp)

    def fk(self, joints: list[float]) -> list[float]:
        self.fk_calls += 1
        if self.fk_chain is None:
            raise RuntimeError("getForwardKinematics() function did not succeed!")
        assert self.fk_tcp is not None
        return _pose_vector(self.fk_chain.flange_m(joints) @ ur_pose_matrix_m(self.fk_tcp))


def _check(controller: _Controller, *, singularity_fk: str = "dh", arm_model: "str | None" = "ur10") -> MotionController:
    return MotionController(controller, object(), singularity_fk=singularity_fk, arm_model=arm_model)  # type: ignore[arg-type]


class TheChainsVerdictIsTheControllersTests(unittest.TestCase):
    def test_on_the_cells_line_ends_and_lines_toward_every_singularity_with_no_round_trip(self) -> None:
        """⭐ 177 configurations on each of the two controllers, every verdict the controller's own."""
        for name, variant in _recorded()["variants"].items():
            rows = _rows(variant)
            controller = _Controller(rows, variant["tcp"], fk_chain=None)
            check = _check(controller)
            near = 0
            for row in variant["rows"]:
                with self.subTest(controller=name, line_end=row["label"]):
                    report = analyze_joint_singularity(check._singularity_fk(), JointPositions(tuple(row["joints"])),
                                                       thresholds=check.singularity_thresholds)
                    self.assertEqual(row["near"], report.is_near_singularity)
                    self.assertEqual(row["near"], check._is_singularity_risky(list(row["joints"])))
                    # The controller's own FK rounds its rotation vector near a half turn (tool straight down): up to
                    # 7e-7 rad measured, 9e-5 of the least singular value. The chain here is the more exact of the two.
                    self.assertLess(abs(report.min_singular_value - row["min_sigma"]) / row["min_sigma"], 2e-4)
                    near += int(report.is_near_singularity)
            self.assertEqual(0, controller.fk_calls, "the chain needs no round trip")
            self.assertGreater(near, 50, "the lines toward the singularities cross the thresholds")

    def test_the_cells_line_ends_keep_the_thresholds_many_times_over(self) -> None:
        for name, variant in _recorded()["variants"].items():
            cell = [row for row in variant["rows"] if row["label"].split(":")[0] not in ("wrist", "elbow", "shoulder")]
            with self.subTest(controller=name):
                self.assertEqual(21, len(cell))
                self.assertFalse(any(row["near"] for row in cell))
                self.assertGreater(min(row["min_sigma"] for row in cell) / 0.005, 45.0)
                self.assertGreater(250.0 / max(row["condition"] for row in cell), 25.0)


class TheNominalTableStandsInOnlyForTheControllersOwnChainTests(unittest.TestCase):
    def setUp(self) -> None:
        recorded = _recorded()["variants"]
        self.tcp = recorded["nominal"]["tcp"]
        self.calibrated = _chain_of(_rows(recorded["calibrated"]))
        self.row = recorded["calibrated"]["rows"][0]

    def test_rows_that_cannot_be_read_on_an_uncalibrated_controller_take_the_nominal_table(self) -> None:
        controller = _Controller(None, self.tcp, fk_chain=URDhChain.nominal("ur10"))
        check = _check(controller)
        for _ in range(3):
            check._is_singularity_risky(list(self.row["joints"]))
        self.assertEqual(2, controller.fk_calls, "two probes once, then the table alone")

    def test_rows_that_cannot_be_read_on_a_calibrated_controller_keep_its_fk(self) -> None:
        controller = _Controller(None, self.tcp, fk_chain=self.calibrated)
        check = _check(controller)
        with self.assertLogs("MotionController", level="WARNING") as said:
            self.assertFalse(check._is_singularity_risky(list(self.row["joints"])))
        self.assertEqual(1 + 12, controller.fk_calls, "the first probe says the table is not its chain; 12 as always")
        self.assertIn("a calibration the nominal table does not hold", "\n".join(said.output))
        check._is_singularity_risky(list(self.row["joints"]))
        self.assertEqual(1 + 12 + 12, controller.fk_calls, "the probes are asked once per connection")

    def test_a_new_connection_has_the_table_checked_again(self) -> None:
        controller = _Controller(None, self.tcp, fk_chain=URDhChain.nominal("ur10"))
        check = _check(controller)
        check._is_singularity_risky(list(self.row["joints"]))
        controller.kinematics_epoch += 1
        check._is_singularity_risky(list(self.row["joints"]))
        self.assertEqual(4, controller.fk_calls)


class TheControllersFkWhereNoChainCanBeHadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = _recorded()["variants"]["nominal"]["rows"][0]
        self.tcp = _recorded()["variants"]["nominal"]["tcp"]

    def test_an_unreadable_tcp_keeps_the_controllers_fk(self) -> None:
        controller = _Controller(_rows(_recorded()["variants"]["nominal"]), None, fk_chain=URDhChain.nominal("ur10"),
                                 fk_tcp=self.tcp)
        _check(controller)._is_singularity_risky(list(self.row["joints"]))
        self.assertEqual(12, controller.fk_calls)

    def test_no_rows_and_no_model_keep_the_controllers_fk(self) -> None:
        controller = _Controller(None, self.tcp, fk_chain=URDhChain.nominal("ur10"))
        _check(controller, arm_model=None)._is_singularity_risky(list(self.row["joints"]))
        self.assertEqual(12, controller.fk_calls)

    def test_a_connection_that_offers_neither_read_keeps_the_controllers_fk(self) -> None:
        class Bare:
            def __init__(self) -> None:
                self.calls = 0

            def fk(self, joints: list[float]) -> list[float]:
                self.calls += 1
                return _pose_vector(URDhChain.nominal("ur10").flange_m(joints) @ ur_pose_matrix_m(self_tcp))

        self_tcp = self.tcp
        bare = Bare()
        MotionController(bare, object(), singularity_fk="dh", arm_model="ur10")._is_singularity_risky(  # type: ignore[arg-type]
            list(self.row["joints"]))
        self.assertEqual(12, bare.calls)


class TheControllersFkStaysTheDefaultTests(unittest.TestCase):
    def test_the_check_asks_the_controller_twelve_times_and_reads_no_rows(self) -> None:
        variant = _recorded()["variants"]["nominal"]
        controller = _Controller(_rows(variant), variant["tcp"], fk_chain=URDhChain.nominal("ur10"))
        check = MotionController(controller, object())  # type: ignore[arg-type]
        self.assertFalse(check._is_singularity_risky(list(variant["rows"][0]["joints"])))
        self.assertEqual(12, controller.fk_calls)
        self.assertEqual(0, controller.rows_asked)

    def test_anything_but_controller_or_dh_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            MotionController(object(), object(), singularity_fk="local")  # type: ignore[arg-type]

    def test_the_key_loads_and_defaults_to_the_controller(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import RobotConfig

        self.assertEqual("controller", RobotConfig.model_validate({"vendor": "ur"}).safety.ik_quality.singularity_fk)
        asked = RobotConfig.model_validate({"vendor": "ur", "safety": {"ik_quality": {"singularity_fk": "dh"}}})
        self.assertEqual("dh", asked.safety.ik_quality.singularity_fk)
        with self.assertRaises(ValidationError):
            RobotConfig.model_validate({"vendor": "ur", "safety": {"ik_quality": {"singularity_fk": "nominal"}}})


if __name__ == "__main__":
    unittest.main()
