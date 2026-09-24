"""A planner starts from its arm's descriptor, adds the hand the cell names, and refuses anything else it loaded.

Owner Q5 (.commits/robot/51-the-world-and-the-hand.md): the hand is robot.gripper.model and nothing else,
and a descriptor without ``_provenance`` refuses. Until UM lane S11 a descriptor carried the hand, ``{arm}_{hand}.yml``,
and the refusal compared the hand and plate it was built with. A descriptor is now built per arm, ``willy_{arm}.yml``,
and says ``carries_hand: false``; the hand is a body link the sidecar adds when it starts (S10), derived from the hand
the cell names and where its declared tool frame puts it (S12). What is refused is what the sidecar reports it loaded:
no provenance, another arm, a descriptor that carries a hand (the per hand files an earlier build wrote stay in the
content directory), and a sidecar that added no hand.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.contracts import UNSET
from src.geometry import Frame, Pose
from src.robot.core import MotionStatus
from tests._sidecar_identity import arm_identity

_ROOT = Path(__file__).resolve().parents[1]


def _hand(model: str = "robotiq_2f85", plates: "list[float] | None" = None) -> Any:
    from src.robot.safety.planning.hand import planner_hand

    gripper: dict[str, Any] = {"model": model}
    if plates is not None:
        gripper["coupling_plates"] = [{"name": f"plate_{i}", "thickness_mm": float(mm)} for i, mm in enumerate(plates)]
    return planner_hand(RobotConfig.model_validate({"vendor": "ur", "gripper": gripper}))


def _link(model: str = "robotiq_2f85", plates: "list[float] | None" = None) -> Any:
    from src.robot.safety.planning.body_link import HandLink

    return HandLink.from_hand(_hand(model, plates))


def _per_hand_file(arm: str = "ur5e", hand: str = "robotiq_2f85") -> Any:
    """What a sidecar started on a per hand file from an earlier build reports: the hand it carries, no body link."""
    from src.robot.safety.planning.curobo_client import SidecarIdentity

    return SidecarIdentity(
        provenance={"arm": arm, "gripper_key": hand, "coupling_mm": None,
                    "generated_by": "scripts/curobo/build_ur_config.py"},
        bodies=(),
    )


class TheDescriptorIsNamedByTheArmTests(unittest.TestCase):
    def test_the_name_is_the_arm_alone(self) -> None:
        from src.robot.drivers.sim.robot_models import curobo_arm_descriptor

        self.assertEqual(curobo_arm_descriptor("ur5e"), "willy_ur5e.yml")
        self.assertEqual(curobo_arm_descriptor("UR3e"), "willy_ur3e.yml")

    def test_an_arm_the_registry_does_not_hold_names_no_file(self) -> None:
        from src.robot.drivers.sim.robot_models import curobo_arm_descriptor

        with self.assertRaises(ValueError) as ctx:
            curobo_arm_descriptor("ur99")
        self.assertIn("ur99", str(ctx.exception))


class WhatThePlannerLoadedIsCheckedTests(unittest.TestCase):
    """Each refusal changes exactly one field of an identity that is accepted."""

    def _refusal(self, identity: Any, link: Any = None, arm: str = "ur5e") -> "str | None":
        from src.robot.safety.planning.hand import descriptor_refusal

        return descriptor_refusal(identity, link if link is not None else _link(), arm=arm)

    def test_the_arm_descriptor_with_the_hand_added_is_accepted(self) -> None:
        """The control, for a flange hand and for a hand behind a plate."""
        self.assertIsNone(self._refusal(arm_identity()))
        self.assertIsNone(self._refusal(arm_identity(), _link("robotiq_hande", [20.0])))

    def test_no_provenance(self) -> None:
        refusal = self._refusal(dataclasses.replace(arm_identity(), provenance=None))
        assert refusal is not None
        self.assertIn("_provenance", refusal)

    def test_another_arm(self) -> None:
        identity = arm_identity()
        refusal = self._refusal(dataclasses.replace(identity, provenance={**identity.provenance, "arm": "ur3e"}))
        assert refusal is not None
        self.assertIn("'ur3e'", refusal)
        self.assertIn("'ur5e'", refusal)

    def test_a_per_hand_file_says_how_to_build_the_arm(self) -> None:
        identity = arm_identity()
        refusal = self._refusal(dataclasses.replace(
            identity, provenance={**identity.provenance, "gripper_key": "robotiq_hande"}))
        assert refusal is not None
        self.assertIn("robotiq_hande", refusal)
        self.assertIn("build_ur_config.py ur5e", refusal)
        self.assertNotIn("--gripper", refusal)

    def test_a_descriptor_that_does_not_say_it_carries_no_hand(self) -> None:
        identity = arm_identity()
        provenance = {key: value for key, value in identity.provenance.items() if key != "carries_hand"}
        self.assertIsNotNone(self._refusal(dataclasses.replace(identity, provenance=provenance)))

    def test_a_sidecar_that_added_no_hand(self) -> None:
        for label, bodies in (("none", ()), ("another body", ({"link": "plate", "parent": "tool0"},)),
                              ("not reported", UNSET)):
            with self.subTest(label):
                refusal = self._refusal(dataclasses.replace(arm_identity(), bodies=bodies))
                assert refusal is not None
                self.assertIn("hand", refusal)


class _Client:
    def __init__(self, identity: Any) -> None:
        from src.robot.drivers.ur.curobo_motion import UR_ARM_JOINT_NAMES

        self.joint_names = list(UR_ARM_JOINT_NAMES)
        self.dt = 0.0
        self.identity = identity
        self.calls: list[str] = []
        self.closed = False

    def start(self) -> None:
        self.calls.append("start")

    def reserve_world(self, reservation: object) -> None:
        self.calls.append("reserve_world")

    def set_world(self, cuboids: list) -> int:
        self.calls.append("set_world")
        return len(cuboids)

    def plan_joint(self, start: list, goal: list) -> list:
        self.calls.append("plan_joint")
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


def _ur_arm(gripper: "dict | None") -> Any:
    from src.robot.drivers.ur.arm import URRobotArm

    # Declares its planner margin, as every UR cuRobo cell must since B1 S17: without one the factory refuses
    # before it reaches the descriptor, which is this test's subject.
    config: dict[str, Any] = {"vendor": "ur", "ur": {"motion_planner": "curobo"},
                              "safety": {"self_collision": {"planner_margin_mm": 4.0}}}
    if gripper is not None:
        config["gripper"] = gripper
    return URRobotArm(RobotConfig.model_validate(config))


class TheURPlannerChecksWhatItLoadedTests(unittest.TestCase):
    def _planner(self, client: _Client, link: Any) -> Any:
        from src.robot.drivers.ur.curobo_motion import CuroboUrPlanner
        from src.robot.safety.planning.hand import descriptor_refusal

        return CuroboUrPlanner(
            _Conn(), client_factory=lambda: client, require_registration=False,
            descriptor_check=lambda identity: descriptor_refusal(identity, link, arm="ur5e"),
        )

    def test_a_move_on_a_per_hand_file_is_rejected_and_nothing_is_sent(self) -> None:
        client = _Client(_per_hand_file(hand="robotiq_hande"))
        planner = self._planner(client, _link("robotiq_hande", [20.0]))
        from src.robot.safety.planning import CuroboUnavailableError

        with self.assertLogs("CuroboUrPlanner", level="ERROR") as logs, \
                self.assertRaises(CuroboUnavailableError) as refused:
            planner.plan_joint([0.1] * 6)
        self.assertIn("build_ur_config.py", str(refused.exception))
        self.assertNotIn("plan_joint", client.calls)
        self.assertEqual(planner._conn.moves, [])  # noqa: SLF001
        self.assertTrue(client.closed, "a client on the wrong descriptor is closed, not kept for the next move")
        self.assertEqual(len([r for r in logs.records if r.levelname == "ERROR"]), 1, logs.output)

    def test_the_arm_descriptor_with_the_hand_plans(self) -> None:
        """The control."""
        client = _Client(arm_identity())
        planner = self._planner(client, _link())
        result = planner.execute(planner.plan_joint([0.1] * 6), _pose())
        self.assertIs(result.status, MotionStatus.EXECUTED, result.message)
        self.assertIn("plan_joint", client.calls)

    def test_the_ur_arm_starts_its_planner_on_the_arm_descriptor_with_the_hand_link(self) -> None:
        arm = _ur_arm({"model": "robotiq_hande", "coupling_plates": [{"name": "plate", "thickness_mm": 20.0}]})
        with mock.patch("src.robot.drivers.ur.arm.CuroboPlanClient") as client_class:
            arm._default_curobo_client_factory()()  # noqa: SLF001
        kwargs = client_class.call_args.kwargs
        self.assertEqual(kwargs["robot_config"], "willy_ur5e.yml")
        self.assertEqual([body["link"] for body in kwargs["body_links"]], ["hand"])
        self.assertEqual(kwargs["body_links"][0]["fixed_transform"], [0.0, 0.02, 0.0, 1.0, 0.0, 0.0, 0.0])

    def test_the_ur_arm_checks_what_the_sidecar_loaded(self) -> None:
        arm = _ur_arm({"model": "robotiq_hande", "coupling_plates": [{"name": "plate", "thickness_mm": 20.0}]})
        arm._curobo_client_factory = lambda: _Client(arm_identity())  # noqa: SLF001
        check = arm._curobo_ur_planner()._descriptor_check  # noqa: SLF001
        self.assertIsNotNone(check)
        self.assertIsNotNone(check(_per_hand_file(hand="robotiq_hande")))
        self.assertIsNone(check(arm_identity(hand="robotiq_hande", coupling_mm=20.0)))

    def test_a_cell_with_no_hand_builds_no_client(self) -> None:
        """The control: nothing is started for a hand nobody named.

        On the capsule guard, because a cell whose exact mesh guard reads hand geometry is already refused when it is
        built with no hand, one layer before any planner, which is the stronger refusal and not the one tested here.
        """
        from src.robot.drivers.ur.arm import URRobotArm
        from src.robot.safety.planning import CuroboUnavailableError

        arm = URRobotArm(RobotConfig.model_validate({
            "vendor": "ur", "ur": {"motion_planner": "curobo"}, "safety": {"self_collision": {"backend": "capsule"}},
        }))
        with mock.patch("src.robot.drivers.ur.arm.CuroboPlanClient") as client_class:
            with self.assertRaises(CuroboUnavailableError):
                arm._default_curobo_client_factory()()  # noqa: SLF001
        client_class.assert_not_called()


class TheSimArmChecksWhatItLoadedTests(unittest.TestCase):
    """A non UR double: the Isaac arm builds its own client and names its descriptor itself."""

    def _arm(self) -> Any:
        from src.robot.drivers.sim.arm import IsaacRobotArm
        from src.robot.drivers.sim.config import SimRobotConfig
        from src.willy_sim.config import load_sim_config, sim_safety_preflight

        return IsaacRobotArm(
            SimRobotConfig(enabled=True, mock_mode=True), safety_preflight=sim_safety_preflight(load_sim_config()),
        )

    def _patched(self, identity: Any) -> tuple[Any, dict]:
        built: dict = {}

        def factory(**kwargs: Any) -> _Client:
            built.update(kwargs)
            built["client"] = _Client(identity)
            return built["client"]

        return mock.patch("src.robot.drivers.sim.arm.CuroboPlanClient", side_effect=factory), built

    def test_a_per_hand_file_refuses_and_the_client_is_closed(self) -> None:
        from src.robot.safety.planning import CuroboUnavailableError

        patch, built = self._patched(_per_hand_file())
        arm = self._arm()
        with patch, self.assertRaises(CuroboUnavailableError) as ctx:
            arm._get_curobo_client()  # noqa: SLF001
        self.assertIn("build_ur_config.py", str(ctx.exception))
        self.assertTrue(built["client"].closed)
        self.assertIsNone(arm._curobo_client)  # noqa: SLF001

    def test_the_sim_starts_on_the_arm_descriptor_with_the_hand_link_and_keeps_a_matching_client(self) -> None:
        patch, built = self._patched(arm_identity())
        arm = self._arm()
        with patch:
            client = arm._get_curobo_client()  # noqa: SLF001
        self.assertEqual(built["robot_config"], "willy_ur5e.yml")
        self.assertEqual([body["link"] for body in built["body_links"]], ["hand"])
        self.assertIs(client, built["client"])


def _started_after(ready: dict) -> Any:
    from src.robot.safety.planning.curobo_client import CuroboPlanClient

    script = Path(tempfile.mkdtemp()) / "sidecar.py"
    script.write_text(textwrap.dedent(f"""
        import json, sys
        sys.stdout.write(json.dumps({ready!r}) + chr(10)); sys.stdout.flush()
        for line in sys.stdin:
            if json.loads(line).get("cmd") == "shutdown":
                break
    """), encoding="utf-8")
    return CuroboPlanClient(python_path=sys.executable, server_script=str(script), robot_config="unused",
                            scene_config=None)


class TheSidecarAddsTheBodiesItIsGivenTests(unittest.TestCase):
    def test_the_client_hands_its_bodies_to_the_sidecar_and_never_a_stale_one(self) -> None:
        from src.robot.safety.planning._curobo_body_links import ENV_BODY_LINKS
        from src.robot.safety.planning.curobo_client import CuroboPlanClient

        body = _link().to_dict()
        with mock.patch.dict(os.environ, {ENV_BODY_LINKS: json.dumps([{"link": "stale"}])}):
            given = CuroboPlanClient(python_path=sys.executable, robot_config="willy_ur5e.yml", body_links=[body])
            bare = CuroboPlanClient(python_path=sys.executable, robot_config="willy_ur5e.yml")
            self.assertEqual(json.loads(given._sidecar_env()[ENV_BODY_LINKS]), [body])  # noqa: SLF001
            self.assertNotIn(ENV_BODY_LINKS, bare._sidecar_env())  # noqa: SLF001

    def test_the_sidecar_composes_them_and_reports_what_it_loaded(self) -> None:
        """A source check, as test_curobo_client_error_vs_verdict.py does: running the sidecar needs a GPU."""
        source = (_ROOT / "src/robot/safety/planning/curobo_planner_server.py").read_text(encoding="utf-8")
        for name in ("os.environ.get(ENV_BODY_LINKS)", "bodies=_bodies", "body_report(_COMPOSED"):
            with self.subTest(present=name):
                self.assertIn(name, source)
        start = source.index('_emit({"status": "ready"')
        self.assertIn('"bodies": _body_rows', source[start:start + 600])

    def test_the_client_keeps_the_rows_the_ready_line_reported(self) -> None:
        row = {"link": "hand", "parent": "tool0", "spheres": 36, "slots": 0, "spheres_sha256": "e" * 64}
        client = _started_after({"status": "ready", "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"], "dt": 0.02,
                                 "descriptor": arm_identity().provenance, "bodies": [row]})
        try:
            client.start()
            self.assertEqual(client.identity.body_names, ("hand",))
            self.assertEqual(client.identity.bodies, (row,))
            self.assertEqual(client.descriptor_provenance, arm_identity().provenance)
        finally:
            client.close()


class EveryReadingNamesTheDescriptorTests(unittest.TestCase):
    def test_the_doctor_says_a_cell_with_no_hand_has_no_descriptor(self) -> None:
        from src.robot.drivers.sim.robot_models import NO_DESCRIPTOR
        from src.robot.safety.planning import doctor as doc

        completed = mock.Mock(stdout='{"curobo": "x", "torch": "y", "cuda": true, "backend": "cuda_core"}', stderr="")
        with mock.patch.object(doc, "curobo_python_path", return_value=sys.executable), \
             mock.patch.object(doc.subprocess, "run", return_value=completed):
            probes = doc._probe_curobo((), NO_DESCRIPTOR)  # noqa: SLF001
        (descriptor,) = [p for p in probes if p.name.startswith("cuRobo robot descriptor")]
        self.assertIs(descriptor.status, doc.ProbeStatus.MISSING)
        self.assertIn("robot.gripper.model", descriptor.detail + descriptor.remedy)

    def test_the_doctor_reads_the_arm_descriptor_and_names_the_hand_it_adds(self) -> None:
        """A healthy arm descriptor is OK, with the hand in the detail rather than joined into a file name."""
        from src.robot.safety.planning import doctor as doc

        payload = {"curobo": "x", "torch": "y", "cuda": True, "backend": "cuda_core",
                   "descriptor": "content/configs/robot/willy_ur5e.yml", "descriptor_present": True}
        completed = mock.Mock(stdout=json.dumps(payload), stderr="")
        with mock.patch.object(doc, "curobo_python_path", return_value=sys.executable), \
             mock.patch.object(doc.subprocess, "run", return_value=completed) as run:
            probes = doc._probe_curobo((), "willy_ur5e.yml", gripper="robotiq_2f85")  # noqa: SLF001
        (descriptor,) = [p for p in probes if p.name.startswith("cuRobo robot descriptor")]
        self.assertIs(descriptor.status, doc.ProbeStatus.OK)
        self.assertIn("willy_ur5e.yml", descriptor.name)
        self.assertIn("robotiq_2f85", descriptor.detail)
        self.assertEqual(run.call_args.kwargs["env"]["WILLY_DOCTOR_ROBOT"], "willy_ur5e.yml")

    def test_the_real_cell_hand_row_names_the_arm_descriptor_and_the_hand_link(self) -> None:
        from src.robot.execution.real_cell.preflight import run_config_preflight

        report = run_config_preflight(
            RobotConfig.model_validate({"vendor": "ur", "gripper": {"model": "robotiq_2f85"}}),
            curobo_available=True,
        )
        (row,) = [check for check in report.checks if check.name == "hand"]
        self.assertIn("willy_ur5e.yml", row.detail)
        self.assertIn("body link", row.detail)

    def test_the_checker_plans_an_arm_descriptor_with_a_hand_added(self) -> None:
        """A source check: the checker runs in the cuRobo environment, where this repository cannot be imported."""
        check = (_ROOT / "scripts/curobo/check_ur_descriptors.py").read_text(encoding="utf-8")
        self.assertIn('glob("willy_ur*.yml")', check)
        self.assertIn('"--hand"', check)
        self.assertIn("compose_sidecar_config(", check)


if __name__ == "__main__":
    unittest.main()
