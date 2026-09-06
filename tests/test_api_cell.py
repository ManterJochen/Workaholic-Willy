"""Bringing a cell up and down from the console: look, acknowledge, act -- and roll back.

Every test runs against a COPY of the shipped config tree, patched to a **dummy** vendor. That is not a
weakening: the dummy arm exercises the same lifecycle code the UR does (build -> preview -> token ->
arm.connect -> gripper.connect -> rollback), and it is the only vendor whose driver can be constructed on
a machine with no ``ur_rtde``. The UR-specific half -- payload push, tool-frame derivation -- has its own
suite in ``test_ur_tool_frame.py``.

The three tests that carry the design:

* ``test_connect_without_a_token_is_refused`` -- connect is motion on this cell, and a bare POST must not
  be able to cause it.
* ``test_a_gripper_that_refuses_rolls_the_arm_back_and_frees_the_cell`` -- the transaction. A cell that
  came up halfway would hold a UR controller's single control script while reporting failure.
* ``test_a_substituted_gripper_blocks_the_connect`` -- the silent-success trap: a misconfigured
  end-effector yields a working ``NullGripper``, so the cell connects, every pick reports success, and
  nothing is ever gripped.

Honesty bucket (2): real objects, real transitions, no physical controller anywhere.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - the console is an optional extra
    TestClient = None  # type: ignore[assignment,misc]

from src.config.loader import active_profile, reload_config, set_active_profile

_SHIPPED = Path(__file__).resolve().parents[1] / "src" / "config" / "data"


def _dummy_tree(target: Path, *, gripper: str = "none") -> None:
    """A copy of the shipped tree, re-pointed at a dummy arm and the given end-effector.

    Both keys are indented (they live under ``robot:`` and ``robot.gripper:``), and both are asserted to
    have matched -- a substitution that silently found nothing would leave every test in this file
    exercising the shipped UR config and failing for a reason that has nothing to do with what it tests.
    """
    shutil.copytree(_SHIPPED, target)
    robot = target / "robot" / "robot.yaml"
    text = robot.read_text(encoding="utf-8")
    text, arm_hits = re.subn(
        r'^(\s*)vendor:\s*"ur"$', r'\g<1>vendor: "dummy"', text, count=1, flags=re.MULTILINE
    )
    text, gripper_hits = re.subn(
        r'^(\s*)vendor:\s*"robotiq"$', rf'\g<1>vendor: "{gripper}"', text, count=1, flags=re.MULTILINE
    )
    assert arm_hits == 1, "robot.vendor was not re-pointed at the dummy arm"
    assert gripper_hits == 1, "robot.gripper.vendor was not re-pointed"
    robot.write_text(text, encoding="utf-8")


@unittest.skipIf(TestClient is None, "requirements/api.txt is not installed (optional extra)")
class CellLifecycleTests(unittest.TestCase):
    GRIPPER = "none"

    def setUp(self) -> None:
        from api.app import create_app
        from api.cell import Console, set_console

        self.tmp = Path(tempfile.mkdtemp()) / "data"
        _dummy_tree(self.tmp, gripper=self.GRIPPER)
        self._previous_profile = active_profile()
        self.cell = Console(root=self.tmp, profile=None)
        self._previous_console = set_console(self.cell)
        self.client = TestClient(create_app())
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        from api.cell import set_console

        try:
            self.cell.session.disconnect()
        except Exception:  # pragma: no cover - cleanup must not mask a failure
            pass
        set_console(self._previous_console)
        set_active_profile(self._previous_profile)
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    def _build(self) -> dict:
        response = self.client.post("/v1/cell/build", params={"rehearse": True})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _token(self) -> str:
        preview = self.client.get("/v1/cell/connect-preview")
        self.assertEqual(preview.status_code, 200, preview.text)
        return preview.json()["token"]

    # -- the sequence ------------------------------------------------------------------------------

    def test_a_fresh_console_has_built_nothing(self) -> None:
        body = self.client.get("/v1/cell").json()
        self.assertEqual(body["state"], "disconnected")
        self.assertIsNone(body["arm"])
        self.assertEqual(body["vendor"], "dummy")
        # A dummy cell claims no controller, so two consoles (or a console and the CLI) do not contend.
        self.assertIsNone(body["lock_key"])

    def test_a_preview_before_a_build_says_so_rather_than_inventing_one(self) -> None:
        response = self.client.get("/v1/cell/connect-preview")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "not_built")

    def test_building_touches_no_robot(self) -> None:
        """``from_robot_config`` deliberately never connects the arm -- the caller owns the lifecycle."""
        from src.robot.drivers.dummy.arm import DummyRobotArm

        with patch.object(DummyRobotArm, "connect", autospec=True) as connect:
            body = self._build()
        connect.assert_not_called()
        self.assertEqual(body["state"], "built")
        self.assertEqual(body["arm"], "DummyRobotArm")
        self.assertIsNone(body["gripper_substitution"])

    def test_the_whole_sequence_brings_the_cell_up_and_down(self) -> None:
        self._build()
        connected = self.client.post("/v1/cell/connect", json={"token": self._token()})
        self.assertEqual(connected.status_code, 200, connected.text)
        self.assertEqual(connected.json()["state"], "connected")

        down = self.client.post("/v1/cell/disconnect")
        self.assertEqual(down.status_code, 200)
        self.assertEqual(down.json()["state"], "built", "the built cell survives a disconnect")

    def test_disconnect_is_idempotent(self) -> None:
        """The one thing an operator must always be able to do is put the cell down."""
        self._build()
        self.client.post("/v1/cell/connect", json={"token": self._token()})
        for _ in range(3):
            self.assertEqual(self.client.post("/v1/cell/disconnect").status_code, 200)

    # -- the acknowledgement -----------------------------------------------------------------------

    def test_connect_without_a_token_is_refused(self) -> None:
        """Connect is MOTION on a real cell; a bare POST must not be able to cause it.

        428 rather than 400: the request is well-formed, it is missing a precondition -- somebody has to
        have read what this specific cell will do.
        """
        self._build()
        response = self.client.post("/v1/cell/connect", json={"token": "not-a-real-token"})
        self.assertEqual(response.status_code, 428)
        self.assertEqual(response.json()["code"], "not_acknowledged")
        self.assertEqual(self.client.get("/v1/cell").json()["state"], "built")

    def test_a_token_does_not_survive_a_config_change(self) -> None:
        """What was acknowledged is no longer what will happen, so the acknowledgement is void.

        This is the case a timeout alone would miss: an operator reads the preview, someone writes a
        payload mass in the other tab, and the connect that follows would push a different number to the
        controller than the one that was reviewed.
        """
        self._build()
        token = self._token()
        patch_response = self.client.patch(
            "/v1/config", json={"robot.safety.payload.mass_kg": 1.15}
        )
        self.assertEqual(patch_response.status_code, 200, patch_response.text)

        response = self.client.post("/v1/cell/connect", json={"token": token})
        self.assertEqual(response.status_code, 428)
        self.assertEqual(response.json()["code"], "stale_token")

    def test_a_token_is_single_use(self) -> None:
        """A replayed request must not be able to connect a cell a second time."""
        self._build()
        token = self._token()
        self.assertEqual(
            self.client.post("/v1/cell/connect", json={"token": token}).status_code, 200
        )
        self.client.post("/v1/cell/disconnect")
        replay = self.client.post("/v1/cell/connect", json={"token": token})
        self.assertEqual(replay.status_code, 428)
        self.assertEqual(replay.json()["code"], "not_acknowledged")

    def test_a_preview_names_the_arm_and_gripper_it_was_issued_for(self) -> None:
        self._build()
        preview = self.client.get("/v1/cell/connect-preview").json()
        self.assertEqual(preview["arm"], "DummyRobotArm")
        self.assertTrue(preview["expires_at"])
        # A NullGripper moves nothing on connect, so there is nothing to warn about. Warning anyway
        # would train an operator to skip the warning that matters.
        self.assertEqual(preview["warnings"], [])

    # -- the transaction ---------------------------------------------------------------------------

    def test_a_gripper_that_refuses_rolls_the_arm_back_and_frees_the_cell(self) -> None:
        from src.robot.drivers.dummy.arm import DummyRobotArm
        from src.robot.grippers.null import NullGripper

        self._build()
        order: list[str] = []
        with patch.object(DummyRobotArm, "connect", autospec=True,
                          side_effect=lambda self: order.append("arm.connect")), \
             patch.object(DummyRobotArm, "disconnect", autospec=True,
                          side_effect=lambda self: order.append("arm.disconnect")), \
             patch.object(NullGripper, "connect", autospec=True,
                          side_effect=RuntimeError("URCap socket 63352 refused")):
            response = self.client.post("/v1/cell/connect", json={"token": self._token()})

        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(response.json()["code"], "driver_refused")
        self.assertIn("63352", response.json()["message"])
        self.assertEqual(order, ["arm.connect", "arm.disconnect"])
        # Back to exactly where we were -- not "connecting", not a third state to reason about.
        self.assertEqual(self.client.get("/v1/cell").json()["state"], "built")

    def test_rebuilding_under_a_live_connection_is_refused(self) -> None:
        """A rebuild would orphan the connected arm and gripper with nothing holding a reference."""
        self._build()
        self.client.post("/v1/cell/connect", json={"token": self._token()})
        response = self.client.post("/v1/cell/build", params={"rehearse": True})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "wrong_state")


@unittest.skipIf(TestClient is None, "requirements/api.txt is not installed (optional extra)")
class SubstitutedGripperTests(CellLifecycleTests):
    """A config asking for a Robotiq on a non-UR arm -- the silent-success trap, refused."""

    GRIPPER = "robotiq"

    def test_a_substituted_gripper_blocks_the_connect(self) -> None:
        body = self._build()
        substitution = body["gripper_substitution"]
        self.assertIsNotNone(substitution, "the build must report why it substituted")
        self.assertEqual(substitution["reason"], "robotiq_needs_ur")

        preview = self.client.get("/v1/cell/connect-preview").json()
        self.assertTrue(preview["blocking"], "the preview must say the connect will be refused")

        response = self.client.post("/v1/cell/connect", json={"token": preview["token"]})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["code"], "no_real_gripper")
        # The message has to say WHY it would have looked like it worked.
        self.assertIn("close on nothing", response.json()["message"])

    # These inherit a config that can never connect, so the sequence tests below do not apply.
    test_the_whole_sequence_brings_the_cell_up_and_down = None  # type: ignore[assignment]
    test_disconnect_is_idempotent = None  # type: ignore[assignment]
    test_a_token_is_single_use = None  # type: ignore[assignment]
    test_a_gripper_that_refuses_rolls_the_arm_back_and_frees_the_cell = None  # type: ignore[assignment]
    test_rebuilding_under_a_live_connection_is_refused = None  # type: ignore[assignment]
    test_building_touches_no_robot = None  # type: ignore[assignment]
    test_a_preview_names_the_arm_and_gripper_it_was_issued_for = None  # type: ignore[assignment]


@unittest.skipIf(TestClient is None, "requirements/api.txt is not installed (optional extra)")
class MotionWarningTests(unittest.TestCase):
    """What the preview warns about, derived from the BUILT gripper rather than from config text."""

    def test_a_robotiq_warns_about_the_activation_sweep(self) -> None:
        from api.lifecycle import motion_warnings
        from src.config.schema.robot import RobotConfig

        config = RobotConfig.model_validate({"vendor": "ur", "gripper": {"vendor": "robotiq"}})
        warnings = motion_warnings(config, object())  # a stand-in for a real Robotiq driver
        self.assertEqual(len(warnings), 1)
        self.assertIn("CALIBRATION", warnings[0].what)
        self.assertIn("hands clear", warnings[0].precaution.lower())

    def test_a_vacuum_warns_that_it_drops_what_it_holds(self) -> None:
        """Reconnecting a session -- a browser refresh, a second operator -- releases a held part."""
        from api.lifecycle import motion_warnings
        from src.config.schema.robot import RobotConfig

        config = RobotConfig.model_validate({"vendor": "ur", "gripper": {"vendor": "vacuum"}})
        warnings = motion_warnings(config, object())
        self.assertEqual(len(warnings), 1)
        self.assertIn("RELEASED", warnings[0].precaution)

    def test_a_substituted_gripper_warns_about_nothing(self) -> None:
        """It cannot move. Warning about a sweep that cannot happen teaches operators to skip warnings."""
        from api.lifecycle import motion_warnings
        from src.config.schema.robot import RobotConfig
        from src.robot.grippers.null import NullGripper

        config = RobotConfig.model_validate({"vendor": "ur", "gripper": {"vendor": "robotiq"}})
        self.assertEqual(motion_warnings(config, NullGripper()), ())

    def test_the_jaw_warning_depends_on_the_wiring_not_on_the_vendor(self) -> None:
        """⛔⛔ THERE WAS NO jaw_io BRANCH AT ALL UNTIL 2026-09-04, and this dispatch is hand-kept.
        `JawIOGripper.connect()` calls `_actuate(close=False)` -- it OPENS THE JAWS -- whenever the
        end-stops read empty, and unconditionally when there is no feedback and the operator opted
        in. So the one vendor whose connect behaviour DEPENDS on wiring was the one an operator got
        no sentence about.

        ⭐ And the sentence differs per cell, which is why it is read off the BUILT gripper. With
        feedback the driver refuses to drop a held part; without it, the behaviour is whatever was
        opted into; without either, it does not actuate and warns about nothing. A flat "the jaws
        will open" would be false in two of those three, and a warning that is false is one an
        operator learns to skip.
        """
        from api.lifecycle import motion_warnings
        from src.config.schema.robot import RobotConfig

        config = RobotConfig.model_validate({"vendor": "ur", "gripper": {"vendor": "jaw_io"}})

        class _WithFeedback:
            has_feedback = True

        class _NoFeedbackOptedIn:
            has_feedback = False
            _open_on_connect_without_feedback = True

        class _NoFeedback:
            has_feedback = False
            _open_on_connect_without_feedback = False

        wired = motion_warnings(config, _WithFeedback())
        self.assertEqual(len(wired), 1)
        self.assertIn("EMPTY", wired[0].what)
        self.assertIn("miswired", wired[0].precaution)

        blind = motion_warnings(config, _NoFeedbackOptedIn())
        self.assertEqual(len(blind), 1)
        self.assertIn("UNCONDITIONALLY", blind[0].what)
        self.assertIn("DROPPED", blind[0].precaution)

        self.assertEqual(
            motion_warnings(config, _NoFeedback()), (),
            "this cell's driver does not actuate on connect, so there is nothing to warn about",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


@unittest.skipIf(TestClient is None, "requirements/api.txt is not installed (optional extra)")
class StatusPanelTests(CellLifecycleTests):
    """What the panel reads, and -- more importantly -- what it refuses to read."""

    def test_a_disconnected_cell_reports_state_rather_than_raising(self) -> None:
        """A panel that throws when the arm goes away hides exactly the fact it exists to show."""
        body = self.client.get("/v1/cell/status").json()
        self.assertFalse(body["connected"])
        self.assertIsNone(body["tcp_position_mm"])
        self.assertEqual(body["unavailable"], {}, "unsupported is not the same as failed")

    def test_a_connected_cell_reports_pose_and_joints(self) -> None:
        self._build()
        self.client.post("/v1/cell/connect", json={"token": self._token()})
        body = self.client.get("/v1/cell/status").json()

        self.assertTrue(body["connected"])
        self.assertEqual(len(body["tcp_position_mm"]), 3)
        self.assertEqual(len(body["tcp_quaternion_xyzw"]), 4)
        self.assertEqual(len(body["joint_positions"]), 6)
        self.assertEqual(body["unavailable"], {})

    def test_the_panel_says_when_the_arm_is_simulated(self) -> None:
        """The numbers from a sim and a real cell look identical; only this field distinguishes them."""
        self._build()
        self.client.post("/v1/cell/connect", json={"token": self._token()})
        self.assertIn("simulated", self.client.get("/v1/cell/status").json())

    def test_a_read_that_fails_is_reported_not_raised(self) -> None:
        from src.robot.drivers.dummy.arm import DummyRobotArm

        self._build()
        self.client.post("/v1/cell/connect", json={"token": self._token()})
        with patch.object(DummyRobotArm, "get_tcp_pose", autospec=True,
                          side_effect=RuntimeError("RTDE receive dropped")):
            body = self.client.get("/v1/cell/status").json()

        self.assertEqual(self.client.get("/v1/cell/status").status_code, 200)
        self.assertIn("tcp_pose", body["unavailable"])
        self.assertIn("RTDE receive dropped", body["unavailable"]["tcp_pose"])
        # The rest of the snapshot still arrives -- one dead read must not blank the panel.
        self.assertIsNotNone(body["joint_positions"])


@unittest.skipIf(TestClient is None, "requirements/api.txt is not installed (optional extra)")
class DiagnosticsTests(CellLifecycleTests):
    """The buttons that move nothing."""

    def test_the_reading_is_for_the_model_this_cell_configures(self) -> None:
        """A UR3e cell told about UR5e descriptors is the defect this endpoint exists to prevent.

        The mesh bundle ships as ``{model}_collision_meshes.npz`` and the cuRobo descriptor as
        ``{model}.yml``, so a present ur5e bundle says nothing about a ur3e cell.
        """
        body = self.client.get("/v1/diagnostics").json()
        stack = body["motion_stack"]
        self.assertTrue(stack["model"])
        self.assertNotEqual(stack["model_source"], "", "a reading must say where its model came from")
        self.assertIn(str(stack["model"]), str(stack["curobo_robot_config"]))

    def test_the_curobo_reading_states_its_own_limit(self) -> None:
        """'available' means a Python interpreter exists on disk. It is not a planning guarantee."""
        stack = self.client.get("/v1/diagnostics").json()["motion_stack"]
        self.assertIn("not verified", stack["curobo_caveat"].lower())

    def test_the_sdk_table_distinguishes_registered_from_ready(self) -> None:
        """A registered driver whose SDK is absent is the single most common bring-up surprise."""
        vendors = {v["vendor"]: v for v in self.client.get("/v1/diagnostics").json()["vendors"]}
        self.assertIn("ur", vendors)
        self.assertTrue(vendors["dummy"]["ready"], "the dummy arm needs no SDK and is always ready")
        self.assertIn("kind", vendors["ur"])

    def test_reachability_states_the_limit_of_what_it_proves(self) -> None:
        """It opens a TCP socket and closes it -- nothing more, and the payload must not imply more."""
        body = self.client.get("/v1/diagnostics").json()["reachability"]
        # A dummy cell configures no controller address, so there is nothing to reach.
        self.assertFalse(body["checked"])
        self.assertIn("nothing to reach", body["detail"])

    # The lifecycle cases are inherited machinery, not part of what this class is about.
    test_a_fresh_console_has_built_nothing = None  # type: ignore[assignment]
    test_a_preview_before_a_build_says_so_rather_than_inventing_one = None  # type: ignore[assignment]
    test_building_touches_no_robot = None  # type: ignore[assignment]
    test_the_whole_sequence_brings_the_cell_up_and_down = None  # type: ignore[assignment]
    test_disconnect_is_idempotent = None  # type: ignore[assignment]
    test_connect_without_a_token_is_refused = None  # type: ignore[assignment]
    test_a_token_does_not_survive_a_config_change = None  # type: ignore[assignment]
    test_a_token_is_single_use = None  # type: ignore[assignment]
    test_a_preview_names_the_arm_and_gripper_it_was_issued_for = None  # type: ignore[assignment]
    test_a_gripper_that_refuses_rolls_the_arm_back_and_frees_the_cell = None  # type: ignore[assignment]
    test_rebuilding_under_a_live_connection_is_refused = None  # type: ignore[assignment]


class _CapableArm:
    """An arm that implements BOTH optional capability Protocols, so the branches are real.

    Needed because the dummy driver implements neither -- ``isinstance(dummy, SupportsForceTorque)`` is
    False -- which means an HTTP-level test of "the panel does not call ``get_joint_torques``" would
    pass without the branch ever being entered. It would assert nothing and look like it asserted
    something, which is worse than no test.
    """

    def __init__(self) -> None:
        import numpy as np

        from src.geometry import Frame, Pose

        self.calls: list[str] = []
        self._pose = Pose(
            position_mm=np.array([100.0, 0.0, 400.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
        )

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def capabilities(self):  # noqa: ANN201 - a stand-in, not a Protocol implementation under test
        from src.robot.core import RobotCapabilities

        return RobotCapabilities(
            vendor="ur", model="ur3e", dof=6,
            supports_joint_move=True, supports_linear_move=True, supports_async_move=True,
            has_native_fk=True, has_native_ik=True, has_force_control=True, is_simulated=False,
        )

    def get_tcp_pose(self):  # noqa: ANN201
        self.calls.append("get_tcp_pose")
        return self._pose

    def get_joint_positions(self):  # noqa: ANN201
        self.calls.append("get_joint_positions")
        return (0.0, -1.2, 1.3, -0.4, 1.5, 0.2)

    def get_tcp_wrench(self):  # noqa: ANN201
        from src.robot.core import Wrench

        self.calls.append("get_tcp_wrench")
        return Wrench(fx=1.0, fy=2.0, fz=3.0, tx=0.1, ty=0.2, tz=0.3)

    def get_joint_torques(self) -> tuple[float, ...]:
        self.calls.append("get_joint_torques")
        return (0.0,) * 6

    def get_robot_status(self):  # noqa: ANN201
        from src.robot.core import RobotMode, RobotStatus, SafetyMode

        self.calls.append("get_robot_status")
        return RobotStatus(
            robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.NORMAL,
            protective_stopped=False, emergency_stopped=False, message="Normal",
        )

    def recover_from_protective_stop(self) -> bool:  # pragma: no cover - never called by the panel
        self.calls.append("recover_from_protective_stop")
        return True


class TelemetryCostTests(unittest.TestCase):
    """What one panel tick actually costs, asserted on the calls it makes.

    Not routed through HTTP: the point is which driver methods are invoked, and the dummy arm that the
    console can build implements neither capability Protocol, so an HTTP test could not reach them.
    """

    def setUp(self) -> None:
        from src.robot.core import SupportsForceTorque, SupportsRobotStatus

        self.arm = _CapableArm()
        # Guard the guard: if this stand-in ever stops satisfying the Protocols, every assertion below
        # would silently start passing for the wrong reason.
        self.assertIsInstance(self.arm, SupportsForceTorque)
        self.assertIsInstance(self.arm, SupportsRobotStatus)

    def test_the_default_tick_reads_only_the_receive_stream(self) -> None:
        """``get_joint_torques`` goes through the CONTROL interface a running pick is using.

        It sits in the same Protocol as ``get_tcp_wrench``, in the same units, and reads identically
        from a signature. That is exactly why this asserts on the call list rather than on intent.
        """
        from api.telemetry import read_telemetry

        snapshot = read_telemetry(self.arm)

        self.assertEqual(
            self.arm.calls, ["get_tcp_pose", "get_joint_positions", "get_tcp_wrench"]
        )
        self.assertNotIn("get_joint_torques", self.arm.calls)
        self.assertNotIn("get_robot_status", self.arm.calls)
        self.assertEqual(snapshot.tcp_force_n, (1.0, 2.0, 3.0))
        self.assertEqual(snapshot.tcp_torque_nm, (0.1, 0.2, 0.3))
        self.assertIsNone(snapshot.robot_mode)

    def test_the_dashboard_round_trip_happens_only_when_asked_for(self) -> None:
        """``get_robot_status()`` looks like four cheap enum reads plus one dashboard TCP round trip."""
        from api.telemetry import read_telemetry

        snapshot = read_telemetry(self.arm, include_controller_state=True)

        self.assertIn("get_robot_status", self.arm.calls)
        self.assertNotIn("get_joint_torques", self.arm.calls, "still never, at any price")
        self.assertEqual(snapshot.safety_mode, "normal")
        self.assertEqual(snapshot.controller_message, "Normal")

    def test_the_panel_never_offers_to_clear_a_protective_stop(self) -> None:
        """Read-only in the strong sense -- the same reasoning as the missing E-stop button.

        ``recover_from_protective_stop`` is on the very Protocol the panel reads from, one method away
        from what it does call, and it drives the dashboard's closeSafetyPopup + unlockProtectiveStop.
        Clearing a stop belongs where the arm is visible.
        """
        from api.telemetry import read_telemetry

        read_telemetry(self.arm, include_controller_state=True)
        self.assertNotIn("recover_from_protective_stop", self.arm.calls)
