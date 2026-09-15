"""A cuRobo descriptor is named by its arm and its hand, and one built for another hand refuses (Step 4i).

Owner Q5 (.commits/robot/51-the-world-and-the-hand.md): the hand is robot.gripper.model and nothing else,
descriptors are ``{arm}_{hand}.yml``, there is no fallback to ``{arm}.yml``, and a descriptor without ``_provenance``
refuses. Q10: the plate a descriptor was built with is compared with ``robot.gripper.coupling_plates_mm``.

Read on the box before this step (2026-09-15): ur5e.yml and ur3e.yml carry no ``_provenance``, and ur3, ur5, ur10 and
ur10e record ``gripper_key: ur5e``, the sphere map stem before Step 4f. Nothing read that provenance, so a cell that
changed hands kept planning against the old one and every file looked correct. Each of them refuses under this step until
it is rebuilt.
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.robot.core import MotionStatus


def _hand(model: str = "robotiq_2f85", plates: "list[float] | None" = None) -> Any:
    from src.robot.safety.planning.hand import planner_hand

    gripper: dict[str, Any] = {"model": model}
    if plates is not None:
        gripper["coupling_plates_mm"] = plates
    return planner_hand(RobotConfig.model_validate({"vendor": "ur", "gripper": gripper}))


def _provenance(arm: str = "ur5e", gripper_key: str = "robotiq_2f85", coupling_mm: "float | None" = None) -> dict:
    """What build_ur_config.py writes (scripts/curobo/build_ur_config.py, the _provenance block)."""
    return {
        "arm": arm, "gripper": "as the map names it", "gripper_key": gripper_key, "coupling_mm": coupling_mm,
        "generated_by": "scripts/curobo/build_ur_config.py",
    }


class TheDescriptorIsNamedByArmAndHandTests(unittest.TestCase):
    def test_the_name_carries_the_arm_and_the_hand(self) -> None:
        from src.robot.drivers.sim.robot_models import curobo_robot_yml

        self.assertEqual(curobo_robot_yml("ur5e", "robotiq_hande"), "ur5e_robotiq_hande.yml")
        self.assertEqual(curobo_robot_yml("UR3e", "robotiq_2f85"), "ur3e_robotiq_2f85.yml")

    def test_a_hand_the_registry_does_not_hold_names_no_file(self) -> None:
        from src.robot.drivers.sim.robot_models import curobo_robot_yml

        with self.assertRaises(ValueError) as ctx:
            curobo_robot_yml("ur5e", "robotiq_2f58")
        self.assertIn("robotiq_2f58", str(ctx.exception))


class ADescriptorForAnotherHandRefusesTests(unittest.TestCase):
    def test_the_descriptor_built_for_this_arm_and_hand_is_accepted(self) -> None:
        """The control: a matching descriptor, with and without a plate."""
        from src.robot.safety.planning.hand import descriptor_refusal

        self.assertIsNone(descriptor_refusal(_provenance(), _hand(), arm="ur5e"))
        self.assertIsNone(descriptor_refusal(
            _provenance(gripper_key="robotiq_hande", coupling_mm=20.0), _hand("robotiq_hande", [20.0]), arm="ur5e",
        ))

    def test_a_descriptor_for_another_hand_refuses_naming_both(self) -> None:
        from src.robot.safety.planning.hand import descriptor_refusal

        refusal = descriptor_refusal(_provenance(gripper_key="ur5e"), _hand("robotiq_hande", [20.0]), arm="ur5e")
        assert refusal is not None
        self.assertIn("robotiq_hande", refusal)
        self.assertIn("'ur5e'", refusal)

    def test_a_descriptor_for_another_arm_refuses(self) -> None:
        from src.robot.safety.planning.hand import descriptor_refusal

        refusal = descriptor_refusal(_provenance(arm="ur3e"), _hand(), arm="ur5e")
        assert refusal is not None
        self.assertIn("ur3e", refusal)

    def test_a_descriptor_that_says_nothing_about_its_hand_refuses(self) -> None:
        """ur5e.yml and ur3e.yml on the box today: built before provenance existed."""
        from src.robot.safety.planning.hand import descriptor_refusal

        refusal = descriptor_refusal(None, _hand(), arm="ur5e")
        assert refusal is not None
        self.assertIn("_provenance", refusal)

    def test_a_descriptor_built_without_the_plate_refuses_naming_both_plates(self) -> None:
        """Q10: the Hand-E cell writes [20.0]; a descriptor built without --coupling-mm modelled the hand 20 mm short."""
        from src.robot.safety.planning.hand import descriptor_refusal

        refusal = descriptor_refusal(
            _provenance(gripper_key="robotiq_hande", coupling_mm=None), _hand("robotiq_hande", [20.0]), arm="ur5e",
        )
        assert refusal is not None
        self.assertIn("20", refusal)
        self.assertIn("0 mm", refusal)


class _Client:
    def __init__(self, provenance: "dict | None") -> None:
        from src.robot.drivers.ur.curobo_motion import UR_ARM_JOINT_NAMES

        self.joint_names = list(UR_ARM_JOINT_NAMES)
        self.dt = 0.0
        self.descriptor_provenance = provenance
        self.calls: list[str] = []
        self.closed = False

    def start(self) -> None:
        self.calls.append("start")

    def reserve_world(self, reservation: object) -> None:
        self.calls.append("reserve_world")

    def set_world(self, cuboids: list) -> int:
        self.calls.append("set_world")
        return len(cuboids)

    def plan(self, start: list, pos_m: list, quat_wxyz: list) -> list:
        self.calls.append("plan")
        return [[0.0] * 6]

    def close(self) -> None:
        self.closed = True


class _Conn:
    is_connected = True

    def __init__(self) -> None:
        self.moves: list[list[float]] = []

    def get_joint_positions(self) -> list[float]:
        return [0.0] * 6

    def moveJ(self, joints: list, vel: object = None, acc: object = None) -> bool:  # noqa: N802 - the RTDE name
        self.moves.append([float(v) for v in joints])
        return True


def _pose() -> Pose:
    return Pose(
        position_mm=np.array([400.0, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        frame=Frame.BASE, label="goal",
    )


class TheURPlannerRefusesAnotherHandsDescriptorTests(unittest.TestCase):
    def _planner(self, client: _Client, hand: Any) -> Any:
        from src.robot.drivers.ur.curobo_motion import CuroboUrPlanner
        from src.robot.safety.planning.hand import descriptor_refusal

        return CuroboUrPlanner(
            _Conn(), client_factory=lambda: client, require_registration=False,
            descriptor_check=lambda provenance: descriptor_refusal(provenance, hand, arm="ur5e"),
        )

    def test_a_move_on_another_hands_descriptor_is_rejected_and_nothing_is_sent(self) -> None:
        client = _Client(_provenance(gripper_key="ur5e"))
        planner = self._planner(client, _hand("robotiq_hande", [20.0]))
        with self.assertLogs("CuroboUrPlanner", level="ERROR") as logs:
            result = planner.move(_pose())
        self.assertIs(result.status, MotionStatus.CONTROLLER_REJECTED)
        self.assertIn("robotiq_hande", result.message)
        self.assertNotIn("plan", client.calls)
        self.assertEqual(planner._conn.moves, [])  # noqa: SLF001
        self.assertTrue(client.closed, "a client for the wrong hand is closed, not kept for the next move")
        self.assertEqual(len([r for r in logs.records if r.levelname == "ERROR"]), 1, logs.output)

    def test_the_matching_descriptor_plans(self) -> None:
        """The control."""
        client = _Client(_provenance())
        result = self._planner(client, _hand()).move(_pose())
        self.assertIs(result.status, MotionStatus.EXECUTED, result.message)
        self.assertIn("plan", client.calls)

    def test_the_ur_arm_checks_against_the_hand_it_names(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        arm = URRobotArm(RobotConfig.model_validate({
            "vendor": "ur", "ur": {"motion_planner": "curobo"},
            "gripper": {"model": "robotiq_hande", "coupling_plates_mm": [20.0]},
        }))
        arm._curobo_client_factory = lambda: _Client(_provenance(gripper_key="ur5e"))  # noqa: SLF001
        check = arm._curobo_ur_planner()._descriptor_check  # noqa: SLF001
        self.assertIsNotNone(check)
        self.assertIsNotNone(check(_provenance(gripper_key="ur5e")))
        self.assertIsNone(check(_provenance(gripper_key="robotiq_hande", coupling_mm=20.0)))


class TheSimArmRefusesAnotherHandsDescriptorTests(unittest.TestCase):
    """A non UR double: the Isaac arm builds its own client and names its descriptor itself."""

    def _arm(self) -> Any:
        from src.robot.drivers.sim.arm import IsaacRobotArm
        from src.robot.drivers.sim.config import SimRobotConfig
        from src.willy_sim.config import load_sim_config, sim_safety_preflight

        return IsaacRobotArm(
            SimRobotConfig(enabled=True, mock_mode=True), safety_preflight=sim_safety_preflight(load_sim_config()),
        )

    def _patched(self, provenance: "dict | None") -> tuple[Any, dict]:
        built: dict = {}

        def factory(**kwargs: Any) -> _Client:
            built.update(kwargs)
            built["client"] = _Client(provenance)
            return built["client"]

        return mock.patch("src.robot.drivers.sim.arm.CuroboPlanClient", side_effect=factory), built

    def test_another_hands_descriptor_refuses_and_the_client_is_closed(self) -> None:
        from src.robot.safety.planning import CuroboUnavailableError

        patch, built = self._patched(_provenance(gripper_key="ur5e"))
        arm = self._arm()
        with patch, self.assertRaises(CuroboUnavailableError) as ctx:
            arm._get_curobo_client()  # noqa: SLF001
        self.assertIn("robotiq_2f85", str(ctx.exception))
        self.assertTrue(built["client"].closed)
        self.assertIsNone(arm._curobo_client)  # noqa: SLF001

    def test_the_sim_names_the_descriptor_by_its_arm_and_hand_and_the_matching_one_is_kept(self) -> None:
        patch, built = self._patched(_provenance())
        arm = self._arm()
        with patch:
            client = arm._get_curobo_client()  # noqa: SLF001
        self.assertEqual(built["robot_config"], "ur5e_robotiq_2f85.yml")
        self.assertIs(client, built["client"])


class TheSidecarSaysWhichDescriptorItLoadedTests(unittest.TestCase):
    def test_the_ready_line_carries_the_descriptor(self) -> None:
        """A source check, as test_curobo_client_error_vs_verdict.py does: running the sidecar needs a GPU."""
        source = (
            Path(__file__).resolve().parents[1] / "src/robot/safety/planning/curobo_planner_server.py"
        ).read_text(encoding="utf-8")
        start = source.index('_emit({"status": "ready"')
        self.assertIn('"descriptor"', source[start:start + 400])

    def test_the_client_keeps_what_the_ready_line_said_and_none_when_it_said_nothing(self) -> None:
        from src.robot.safety.planning.curobo_client import CuroboPlanClient

        for said, expected in (({"descriptor": _provenance()}, _provenance()), ({}, None)):
            ready = {"status": "ready", "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"], "dt": 0.02, **said}
            script = Path(tempfile.mkdtemp()) / "sidecar.py"
            script.write_text(textwrap.dedent(f"""
                import json, sys
                sys.stdout.write(json.dumps({ready!r}) + chr(10)); sys.stdout.flush()
                for line in sys.stdin:
                    if json.loads(line).get("cmd") == "shutdown":
                        break
            """), encoding="utf-8")
            client = CuroboPlanClient(
                python_path=sys.executable, server_script=str(script), robot_config="unused", scene_config=None,
            )
            with self.subTest(said=sorted(said)):
                try:
                    client.start()
                    self.assertEqual(client.descriptor_provenance, expected)
                finally:
                    client.close()


class EveryReadingNamesTheDescriptorTests(unittest.TestCase):
    def test_the_doctor_says_a_cell_with_no_hand_has_no_descriptor(self) -> None:
        from unittest import mock

        from src.robot.drivers.sim.robot_models import NO_DESCRIPTOR
        from src.robot.safety.planning import doctor as doc

        completed = mock.Mock(stdout='{"curobo": "x", "torch": "y", "cuda": true, "backend": "cuda_core"}', stderr="")
        with mock.patch.object(doc, "curobo_python_path", return_value=sys.executable), \
             mock.patch.object(doc.subprocess, "run", return_value=completed):
            probes = doc._probe_curobo((), NO_DESCRIPTOR)  # noqa: SLF001
        (descriptor,) = [p for p in probes if p.name.startswith("cuRobo robot descriptor")]
        self.assertIs(descriptor.status, doc.ProbeStatus.MISSING)
        self.assertIn("robot.gripper.model", descriptor.detail + descriptor.remedy)

    def test_the_real_cell_hand_row_names_the_descriptor(self) -> None:
        from src.robot.execution.real_cell.preflight import run_config_preflight

        report = run_config_preflight(
            RobotConfig.model_validate({"vendor": "ur", "gripper": {"model": "robotiq_2f85"}}),
            curobo_available=True,
        )
        (row,) = [check for check in report.checks if check.name == "hand"]
        self.assertIn("ur5e_robotiq_2f85.yml", row.detail)

    def test_the_build_script_writes_and_the_checker_finds_arm_and_hand(self) -> None:
        """Source checks: both scripts run in the cuRobo environment, where this repository cannot be imported."""
        root = Path(__file__).resolve().parents[1]
        build = (root / "scripts/curobo/build_ur_config.py").read_text(encoding="utf-8")
        check = (root / "scripts/curobo/check_ur_descriptors.py").read_text(encoding="utf-8")
        self.assertIn('f"configs/robot/{MODEL}_{GRIPPER}.yml"', build)
        self.assertIn('glob("ur*_*.yml")', check)
