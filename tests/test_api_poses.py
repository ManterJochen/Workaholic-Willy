"""Poses in the console (build plan 1.8, OD 6, OD 15, Q10): Home, the taught poses, and the default place.

Home comes from the config and is read-only here. A taught pose is written into the cell's own layer, the last of the
profile chain, by the pose door alone (``src.config.edit.set_named_pose``), and the console says which file that is
before anybody frees the arm. What is held here:

* ``GET /v1/poses`` answers Home in degrees, every named pose with its label and screen (``taught`` where the console
  wrote it, ``config`` where a person did), the default place, the file a taught pose is written to, and whether a pose
  can be taught now, with the reason it cannot: the teach's own refusal, asked without freeing anything;
* ``PUT /v1/poses/default-place`` writes the choice into the cell's own layer, all or nothing; ``null`` clears it;
  a name that is no pose of the tree is ``unknown_pose``, a run in progress is ``run_active``, and a chain with no
  layer of the cell's own is ``no_layer``: the shared ``robot.yaml`` takes no cell's poses.

Honesty bucket (2): a real config tree in a scratch directory, real writes through the pose door, real HTTP.
"""

from __future__ import annotations

import math
import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - the console is an optional extra
    TestClient = None  # type: ignore[assignment,misc]

from src.config.loader import active_profile, reload_config, set_active_profile
from src.robot.core.arm_capabilities import HaltState, RobotStatus
from src.robot.core.freedrive import ControllerPayload
from src.robot.safety.planning.band import PoseScreen, PoseVerdict
from tests.test_a_pose_is_taught_by_hand import LOOK_1_DEG, _GuidedArm, _stand
from tests.test_api_jaws import RUNNING

_ROOT = Path(__file__).resolve().parents[1]

#: The cell's own layer: two poses, one taught in the console and one a person wrote, and the default place.
CELL_LAYER = """\
# The cell's own layer, git-ignored on a real cell (Q10).
robot:
  named_poses:
    drop_left:
      joints_deg: [-45.0, -100.0, -110.0, -60.0, 90.0, 0.0]
      label: "Ablage links"
      taught_at: "2026-10-01T10:00:00+02:00"
      screen: "clear"
      note: ""
    park:
      joints_deg: [0.0, -90.0, -90.0, -90.0, 90.0, 0.0]
  default_place_pose: drop_left
"""

CLEAR = PoseScreen(PoseVerdict.CLEAR, "the exact guard and the planner both clear it.")
GUARD_REFUSED = PoseScreen(PoseVerdict.GUARD_REFUSED, "forearm|wrist_2: mesh distance 4.1 mm < 10.000 mm.",
                           nearby=tuple(math.radians(v) for v in (-45.0, -100.2, -110.0, -64.0, 90.0, 0.0)))
UNSCREENED = PoseScreen(PoseVerdict.UNSCREENED, "the planner could not be asked: no sidecar", planner_unavailable=True)


def cell_tree(target: Path, *, layer: str | None = CELL_LAYER) -> None:
    """The shipped tree, the arm and the hand dummies; ``layer`` is the cell's own overlay (``robot.cell.yaml``)."""
    shutil.copytree(_ROOT / "config", target)
    robot = target / "robot" / "robot.yaml"
    text = robot.read_text(encoding="utf-8")
    text = re.sub(r'^(\s*)vendor:\s*"ur"$', r'\g<1>vendor: "dummy"', text, count=1, flags=re.MULTILINE)
    text = re.sub(r'^(\s*)vendor:\s*"robotiq"$', r'\g<1>vendor: "dummy"', text, count=1, flags=re.MULTILINE)
    robot.write_text(text, encoding="utf-8")
    if layer is not None:
        (target / "robot" / "robot.cell.yaml").write_text(layer, encoding="utf-8")


class TeachArm(_GuidedArm):
    """A connected arm a person can guide, that screens a pose as the UR does, whose planner is ready, with the halt
    latch and the receive-stream status the console reads. Scripted stand by stand; it never moves by itself."""

    def __init__(self, *stands: Any, screen: Any = CLEAR, planner: str = "ready",
                 payload: ControllerPayload | None = ControllerPayload(1.9, (0.0, 12.0, 61.0))) -> None:
        super().__init__(list(stands) or [_stand(LOOK_1_DEG)], time.monotonic, payload=payload)
        self.screen = screen
        self.planner_state = planner
        self.screened: list[list[float]] = []
        self.latch: HaltState | None = None
        #: What the receive stream says of the controller; ``None`` reads as a link that cannot be read.
        self.status: RobotStatus | None = RUNNING

    def screen_configuration(self, joints: Any, *, ask_planner: bool = True) -> PoseScreen:
        self.screened.append([round(math.degrees(v), 4) for v in joints.tolist()])
        self.log.append("screen")
        if isinstance(self.screen, BaseException):
            raise self.screen
        return self.screen

    def halt(self, reason: str) -> HaltState:
        self.latch = self.latch or HaltState(reason=reason, requested_at=time.time())
        return self.latch

    def clear_halt(self) -> None:
        self.latch = None

    def halt_state(self) -> HaltState | None:
        return self.latch

    def quick_robot_status(self) -> RobotStatus:
        if self.status is None:
            raise ConnectionError("RTDE receive dropped")
        return self.status

    def disconnect(self) -> None:
        self.is_connected = False


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class LayerCell(unittest.TestCase):
    """A console on a scratch tree whose chain ends in the cell's own layer (``--profile cell``)."""

    PROFILE: str | None = "cell"
    LAYER: str | None = CELL_LAYER

    def setUp(self) -> None:
        from api.app import create_app
        from api.cell import Console, set_console

        self.tmp = Path(tempfile.mkdtemp()) / "data"
        cell_tree(self.tmp, layer=self.LAYER)
        self._previous_profile = active_profile()
        self.cell = Console(root=self.tmp, profile=self.PROFILE)
        self.cell.record_log_path = self.tmp.parent / "grasp_records.jsonl"
        self._previous_console = set_console(self.cell)
        self.addCleanup(self._restore)
        self.app = create_app()
        self.client = TestClient(self.app)

    def _restore(self) -> None:
        from api.cell import set_console

        try:
            self.cell.session.release()
        except Exception:  # pragma: no cover - cleanup must not mask a failure
            pass
        set_console(self._previous_console)
        set_active_profile(self._previous_profile)
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    def connect_arm(self, arm: Any, gripper: Any = None) -> Any:
        """Stand ``arm`` (and ``gripper``) in the console's session as a connected cell."""
        from api.lifecycle import CellState

        self.cell.session.service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=SimpleNamespace(
            arm=arm, gripper=gripper)))
        self.cell.session.state = CellState.CONNECTED
        return arm

    def layer_text(self) -> str:
        return (self.tmp / "robot" / "robot.cell.yaml").read_text(encoding="utf-8")


class ThePosesAreReadTests(LayerCell):
    def test_home_and_every_named_pose_with_its_label_screen_and_source(self) -> None:
        answered = self.client.get("/v1/poses")
        self.assertEqual(200, answered.status_code, answered.text)
        body = answered.json()
        home = body["home"]
        self.assertEqual(("home", "config"), (home["name"], home["source"]))
        self.assertEqual([0.0, -90.0, 0.0, -90.0, 0.0, 0.0], [round(v, 6) for v in home["joints_deg"]])
        drop, park = body["poses"]
        self.assertEqual(("drop_left", "Ablage links", "taught", "clear", "2026-10-01T10:00:00+02:00"),
                         (drop["name"], drop["label"], drop["source"], drop["screen"], drop["taught_at"]))
        self.assertEqual([-45.0, -100.0, -110.0, -60.0, 90.0, 0.0], drop["joints_deg"])
        self.assertEqual(("park", "", "config", None, None),
                         (park["name"], park["label"], park["source"], park["screen"], park["taught_at"]))
        self.assertEqual("drop_left", body["default_place"])
        self.assertTrue(body["target_file"].replace("\\", "/").endswith("robot/robot.cell.yaml"), body["target_file"])

    def test_a_pose_cannot_be_taught_on_a_cell_that_is_not_connected_and_it_says_why(self) -> None:
        body = self.client.get("/v1/poses").json()
        self.assertFalse(body["teachable"])
        self.assertIn("not connected", body["why_not"])
        self.assertEqual("not_connected", body["why_not_code"])

    def test_a_connected_arm_that_can_be_guided_and_screens_can_teach(self) -> None:
        arm = self.connect_arm(TeachArm())
        body = self.client.get("/v1/poses").json()
        self.assertTrue(body["teachable"], body["why_not"])
        self.assertEqual(("", ""), (body["why_not"], body["why_not_code"]))
        arm.planner_state = "starting"
        body = self.client.get("/v1/poses").json()
        self.assertFalse(body["teachable"])
        self.assertIn("not ready", body["why_not"])
        self.assertEqual("planner_not_ready", body["why_not_code"], "the browser's 'Planer startet noch' reads the code")
        self.assertEqual([], [line for line in arm.log if line in ("free", "screen")], "asking freed or screened")

    def test_an_arm_that_offers_no_hand_guiding_says_so(self) -> None:
        from src.robot.drivers.dummy.arm import DummyRobotArm

        arm = DummyRobotArm()
        arm.connect()
        self.connect_arm(arm)
        body = self.client.get("/v1/poses").json()
        self.assertFalse(body["teachable"])
        self.assertTrue(body["why_not"])


class TheDefaultPlaceIsChosenTests(LayerCell):
    def test_the_choice_is_written_into_the_cell_s_own_layer_and_read_back(self) -> None:
        chosen = self.client.put("/v1/poses/default-place", json={"name": "park"})
        self.assertEqual(200, chosen.status_code, chosen.text)
        self.assertEqual("park", chosen.json()["default_place"])
        self.assertIn("default_place_pose: \"park\"", self.layer_text())
        self.assertEqual("park", self.client.get("/v1/poses").json()["default_place"])
        base = (self.tmp / "robot" / "robot.yaml").read_text(encoding="utf-8")
        self.assertNotIn("park", base, "the shared robot.yaml took the cell's choice")

    def test_null_clears_it(self) -> None:
        cleared = self.client.put("/v1/poses/default-place", json={"name": None})
        self.assertEqual(200, cleared.status_code, cleared.text)
        self.assertIsNone(cleared.json()["default_place"])
        self.assertIsNone(self.client.get("/v1/poses").json()["default_place"])

    def test_a_name_that_is_no_pose_of_the_tree_is_refused_with_nothing_written(self) -> None:
        before = self.layer_text()
        refused = self.client.put("/v1/poses/default-place", json={"name": "nowhere"})
        self.assertEqual((422, "unknown_pose"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual(before, self.layer_text())

    def test_a_run_in_progress_refuses_the_write(self) -> None:
        self.cell.active_run_id = "run-busy"
        refused = self.client.put("/v1/poses/default-place", json={"name": "park"})
        self.assertEqual((409, "run_active"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual("run-busy", refused.json()["detail"]["run_id"])

    def test_the_name_must_be_said_even_when_it_is_none(self) -> None:
        self.assertEqual(422, self.client.put("/v1/poses/default-place", json={}).status_code)


class AChainWithNoLayerOfItsOwnTests(LayerCell):
    PROFILE = None
    LAYER = None

    def test_the_default_place_is_refused_and_the_shared_file_kept(self) -> None:
        base = (self.tmp / "robot" / "robot.yaml").read_text(encoding="utf-8")
        refused = self.client.put("/v1/poses/default-place", json={"name": None})
        self.assertEqual((422, "no_layer"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual(base, (self.tmp / "robot" / "robot.yaml").read_text(encoding="utf-8"))

    def test_the_poses_say_no_file_takes_a_taught_pose(self) -> None:
        body = self.client.get("/v1/poses").json()
        self.assertIsNone(body["target_file"])
        self.assertEqual([], body["poses"])
        self.assertIsNone(body["default_place"])

    def test_a_connected_arm_that_could_teach_is_told_no_file_of_the_cell_s_own_takes_the_pose(self) -> None:
        """Everything about the arm would let a pose be taught; the chain has no layer of the cell's own, so the shared
        ``robot.yaml`` would take it (Q10): the setup page says so before anybody frees the arm."""
        self.connect_arm(TeachArm())
        body = self.client.get("/v1/poses").json()
        self.assertFalse(body["teachable"])
        self.assertIn("robot.yaml", body["why_not"])
        self.assertEqual("no_layer", body["why_not_code"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
