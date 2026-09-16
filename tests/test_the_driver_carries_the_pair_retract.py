"""A cell's planner starts from the pose its own arm AND hand were judged at (B4 revisited, owner 2026-09-16).

The committed table holds one retract per arm, hand, plate and planner margin, because a pose that works with one hand
on the flange can be a self collision with another: measured, ur3 keeps its anchor with the 2F-85 and needs a pose
three steps away with the Hand-E, and the EGU-50 on that arm has no working pose at all.

The descriptor cannot carry that, because a descriptor is per ARM since S11. So the DRIVER reads the row for the pair
it is about to plan with and hands it to the client, which puts it in the sidecar's environment, and the composition
that adds the hand writes it into the config the planner loads.

What this file holds:

* the driver passes the pose the table judged for ITS pair, not the arm's first row;
* a pair the table found no pose for refuses the planner by name, rather than starting one that cannot become ready;
* a cell keeping a margin the pair was never judged at refuses too, because a pose the planner admits at one
  clearance it can refuse at another.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import yaml

from src.config.schema.robot import RobotConfig

#: The committed table, read for the pairs it refuses rather than carrying their names here.
_TABLE = (Path(__file__).resolve().parents[1] / "src" / "robot" / "safety" / "planning"
          / "robot" / "ur_retract.yaml")

_GOOD_FRAME = {"source": "willy", "offset_mm": [0.0, 132.0, 0.0],
               "rotation_quat_xyzw": [-0.70710678, 0.0, 0.0, 0.70710678]}


# The margin every layer declares since B6. The retract table keys its rows by it, so a cell asking for
# another one has no row and the driver refuses before a sidecar starts.
def _cell(hand: str, *, model: str = "ur5e", margin: float = 4.0, plates=None) -> RobotConfig:
    gripper: dict = {"model": hand, "tool_frame": _GOOD_FRAME}
    if plates is not None:
        gripper["coupling_plates_mm"] = list(plates)
    return RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": model, "motion_planner": "curobo"},
        "gripper": gripper,
        "safety": {"self_collision": {"kinematics_model": model, "planner_margin_mm": margin}},
    })


class TheDriverReadsThePairsRowTests(unittest.TestCase):
    def _built(self, config: RobotConfig):
        from src.robot.drivers.ur.arm import URRobotArm

        with mock.patch("src.robot.drivers.ur.arm.CuroboPlanClient") as client_class:
            URRobotArm(config)._default_curobo_client_factory()()  # noqa: SLF001
        return client_class.call_args.kwargs

    def test_the_client_is_handed_the_pose_this_pair_was_judged_at(self) -> None:
        import yaml
        from pathlib import Path

        table = yaml.safe_load(
            (Path(__file__).resolve().parents[1]
             / "src/robot/safety/planning/robot/ur_retract.yaml").read_text(encoding="utf-8"))
        row = next(r for r in table["retracts"]
                   if r["arm"] == "ur5e" and r["hand"] == "robotiq_2f85" and r["plate_mm"] == 0.0)

        kwargs = self._built(_cell("robotiq_2f85"))
        self.assertEqual([round(v, 6) for v in kwargs["default_q"]],
                         [round(float(v), 6) for v in row["retract"]])

    def test_two_hands_on_one_arm_can_get_two_poses(self) -> None:
        """⭐ THE CONTROL. A driver that read the arm's first row would hand both hands the same pose, and this is
        the whole reason the table stopped being per arm."""
        import yaml
        from pathlib import Path

        table = yaml.safe_load(
            (Path(__file__).resolve().parents[1]
             / "src/robot/safety/planning/robot/ur_retract.yaml").read_text(encoding="utf-8"))
        rows = {(r["arm"], r["hand"], r["plate_mm"]): r["retract"] for r in table["retracts"]}
        pairs = [(arm, hand, plate) for (arm, hand, plate) in rows if arm == "ur3"]
        if len({tuple(round(float(v), 6) for v in rows[key]) for key in pairs}) < 2:
            self.skipTest("this table gives every ur3 pair the same pose, so nothing here could tell them apart")

        first, second = pairs[0], next(k for k in pairs if rows[k] != rows[pairs[0]])
        for arm, hand, plate in (first, second):
            plates = [plate] if plate else []
            kwargs = self._built(_cell(hand, model=arm, plates=plates))
            with self.subTest(hand=hand, plate=plate):
                self.assertEqual([round(v, 6) for v in kwargs["default_q"]],
                                 [round(float(v), 6) for v in rows[(arm, hand, plate)]])

    def test_a_pair_with_no_pose_refuses_the_planner_by_name(self) -> None:
        """⭐ The one pair the family still owes, and it refuses at the desk rather than at a sidecar.

        A UR5 carrying the EGU-50 has no pose its planner accepts at the margin every layer declares: that
        sidecar starts at 0 and 2 mm and cannot start at 4, because cuRobo's graph warmup finds no collision
        free sample to build from. The hand's own bundle records ur5e as the only arm it was proven on.

        It was ur3 and ur5 before the refit; ur3 has a pose now, which is why the pair named here is read
        from the table rather than kept by hand.
        """
        from src.robot.safety.planning import CuroboUnavailableError
        from src.robot.drivers.ur.arm import URRobotArm

        refused = (yaml.safe_load(_TABLE.read_text(encoding="utf-8"))
                   ["rule"].get("pairs_with_no_retract") or [])
        self.assertTrue(refused, "no pair is refused, so this assertion has nothing to check")
        row = refused[0]

        arm = URRobotArm(_cell(row["hand"], model=row["arm"]))
        with self.assertRaises(CuroboUnavailableError) as caught:
            arm._default_curobo_client_factory()()  # noqa: SLF001
        said = str(caught.exception)
        self.assertIn(row["hand"], said)
        self.assertIn(row["arm"], said)

    def test_a_margin_the_pair_was_never_judged_at_refuses(self) -> None:
        from src.robot.safety.planning import CuroboUnavailableError
        from src.robot.drivers.ur.arm import URRobotArm

        arm = URRobotArm(_cell("robotiq_2f85", margin=7.5))
        with self.assertRaises(CuroboUnavailableError) as caught:
            arm._default_curobo_client_factory()()  # noqa: SLF001
        self.assertIn("7.5", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
