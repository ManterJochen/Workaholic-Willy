"""A live device handle must not cost the whole config path.

**The gap this closes.** An ``IsaacGripper`` needs the simulator session its arm already lives on --
it is a handle, not data, exactly like the multi-camera rig. Because config could not describe it,
the entire Isaac path ran on ``from_components``, which leaves ``effective_config=None`` and with it
silences the U8/U9/U10 + T2-U10 overlays unless a runner re-enables them by hand. So the simulator
measured a hand-wired stack while a real cell would run the config-driven one, and nothing ever
compared the two. That is the ``sim + real HW are meant to converge on from_robot_config later``
note in CLAUDE.md, and this is the seam that lets them.

The dangerous half is the ``sim`` profile's own configuration: ``gripper.vendor: robotiq`` on a
``sim`` arm. Down the normal path that substitutes a NullGripper -- a cell that connects, reports
every pick a success, and closes on nothing. These tests pin that a supplied handle wins over that
substitution, and that supplying one changes nothing else.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.config.schema.robot.robot_schema import RobotConfig
from src.robot.execution.autonomous_grasp import AutonomousGraspService
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.perception import PerceptionFrame


def _frame() -> PerceptionFrame:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 1
    return PerceptionFrame(
        depth_map=np.full((32, 32), 500.0, dtype=np.float64),
        intrinsics=np.eye(3, dtype=np.float64),
        segmentations=(SimpleNamespace(mask=mask),),
    )


def _calculator():
    return SimpleNamespace(
        compute_result=lambda *a, **k: GraspResult(candidates=(), reasons=(), telemetry={}),
        render_debug_images=False,
    )


def _live_gripper():
    """Stands in for the IsaacGripper: a real object that no config field could name."""

    return SimpleNamespace(
        open=lambda *a, **k: None,
        close=lambda *a, **k: None,
        connect=lambda: None,
        disconnect=lambda: None,
        is_connected=lambda: True,
        min_width_mm=0.0,
        max_width_mm=85.0,
    )


def _service(config: dict, **handles) -> AutonomousGraspService:
    return AutonomousGraspService.from_robot_config(
        RobotConfig(**config),
        calculator=_calculator(),  # type: ignore[arg-type]
        perception=SimpleNamespace(acquire=_frame),  # type: ignore[arg-type]
        **handles,
    )


_SIM_LIKE = {
    "vendor": "dummy",
    "gripper": {"vendor": "robotiq"},
    "grasping": {"default_mode": "auto"},
}


class LiveHandleTests(unittest.TestCase):
    def test_without_a_handle_the_sim_shaped_config_substitutes_a_null_gripper(self) -> None:
        """The state this seam exists to avoid: jaws that close on nothing and report success."""

        with self.assertLogs(level="WARNING") as captured:
            service = _service(_SIM_LIKE)

        self.assertTrue(any("NullGripper" in line for line in captured.output))
        substitution = getattr(service.runtime.orchestrator.gripper, "substitution", None)
        self.assertIsNotNone(substitution, "the substituted gripper did not record its reason")

    def test_a_supplied_gripper_wins_over_the_substitution(self) -> None:
        live = _live_gripper()

        service = _service(_SIM_LIKE, gripper=live)

        self.assertIs(service.runtime.orchestrator.gripper, live)

    def test_a_supplied_arm_wins(self) -> None:
        from src.robot.drivers import create_arm
        from src.robot.core import RobotVendor

        live_arm = create_arm(RobotVendor.DUMMY)

        service = _service(_SIM_LIKE, arm=live_arm, gripper=_live_gripper())

        self.assertIs(service.runtime.orchestrator.arm, live_arm)

    def test_the_rest_still_comes_from_config(self) -> None:
        """The whole point: a handle replaces its own construction step and nothing else.

        ``effective_config`` is the tell. ``from_components`` leaves it ``None`` -- which is exactly
        why the sim silently lost the U-stack overlays -- so its presence proves the config-driven
        path really ran.
        """

        service = _service(
            {**_SIM_LIKE, "grasping": {"default_mode": "auto", "closed_loop": {"enabled": True}}},
            gripper=_live_gripper(),
        )

        self.assertIsNotNone(
            service.effective_config,
            "a supplied handle dropped the build back onto the from_components path",
        )
        self.assertIsNotNone(
            service.refiner, "config-built closed-loop actors did not survive the handle path"
        )


if __name__ == "__main__":
    unittest.main()
