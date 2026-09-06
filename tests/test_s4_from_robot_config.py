"""S4 (buildable-now half): the real-cell `from_robot_config` gaps closed for September.

* the suction/vacuum gripper branch (a driver that ships but `from_robot_config` never built),
* the driver doctor no longer lying that vacuum has no driver,
* record logging stamping robot identity so a UR3e cell's data is not silently pooled with the UR5e's.

All hardware-free: a UR arm is CONSTRUCTED (ur_rtde is deferred to connect, never reached here) and a
dummy arm drives the end-to-end record. Reuses the proven calc/perception fakes.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.config.schema.robot import RobotConfig
from src.robot.drivers.doctor import gripper_vendor_readiness
from src.robot.execution.autonomous_grasp import AutonomousGraspService
from src.robot.execution.runtime_pick import RuntimePickService
from src.robot.grippers.null import NullGripper
from src.robot.grippers.vacuum import VacuumGripper
from tests.test_grasping_config_wiring import _calc_and_perception

_GRASPING = {"default_mode": "auto", "max_attempts": 5}

#: A UR arm satisfies SupportsDigitalIO (the vacuum branch needs it) but from_robot_config fails early
#: without the ur_rtde SDK. Patch that readiness gate off: URRobotArm still CONSTRUCTS (ur_rtde is
#: deferred to connect(), never reached), so the gripper branch runs against a real UR arm.
_READY = "src.robot.drivers.doctor.require_arm_vendor_ready"


def _cfg(vendor: str, gripper: dict, **extra: object) -> RobotConfig:
    grasping = extra.pop("grasping", dict(_GRASPING))
    return RobotConfig(vendor=vendor, gripper=gripper, grasping=grasping, **extra)  # type: ignore[arg-type]


class VacuumBranchTests(unittest.TestCase):
    def test_ur_arm_builds_a_real_vacuum_gripper(self) -> None:
        calc, perc = _calc_and_perception()
        with patch(_READY):
            svc = RuntimePickService.from_robot_config(
                _cfg("ur", {"vendor": "vacuum", "vacuum": {"vacuum_output_pin": 2, "io_port": "tool"}}),
                calculator=calc, perception=perc,  # type: ignore[arg-type]
            )
        self.assertIsInstance(svc.orchestrator.gripper, VacuumGripper)

    def test_non_dio_arm_falls_back_to_null(self) -> None:
        # A dummy arm does not satisfy SupportsDigitalIO -> fail closed to NullGripper, not a crash.
        calc, perc = _calc_and_perception()
        svc = RuntimePickService.from_robot_config(
            _cfg("dummy", {"vendor": "vacuum"}),
            calculator=calc, perception=perc,  # type: ignore[arg-type]
        )
        self.assertIsInstance(svc.orchestrator.gripper, NullGripper)


class DoctorTests(unittest.TestCase):
    def test_vacuum_reports_registered_not_missing(self) -> None:
        rows = {r.vendor: r for r in gripper_vendor_readiness()}
        self.assertIn("vacuum", rows)
        self.assertTrue(rows["vacuum"].registered)          # was False before the fix
        self.assertNotIn("no driver", rows["vacuum"].note)  # the honest-log bug


def _resolver():
    """A minimal static CAMERA->BASE resolver -- what an eye-in-hand cell passes in code."""
    import numpy as np

    from src.geometry import Frame, Transform
    from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver

    return StaticCameraToBaseResolver(transform=Transform(
        translation_mm=np.array([0.0, 0.0, 800.0]),
        quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
        from_frame=Frame.CAMERA, to_frame=Frame.BASE,
    ))


class RecordProvenanceTests(unittest.TestCase):
    def test_from_robot_config_stamps_vendor_and_model(self) -> None:
        calc, perc = _calc_and_perception()
        cfg = _cfg("ur", {"vendor": "none"}, ur={"model": "ur3e"},
                   grasping={**_GRASPING, "record_log_path": "logs/unused.jsonl"})
        # A real (non-sim) cell must carry a CAMERA->BASE resolver or from_robot_config refuses -- see
        # RealCellFrameResolverGateTests below. This test is about record provenance, so hand it the
        # in-code resolver an eye-in-hand cell would supply.
        with patch(_READY):
            svc = AutonomousGraspService.from_robot_config(
                cfg, calculator=calc, perception=perc, frame_resolver=_resolver(),  # type: ignore[arg-type]
            )
        self.assertEqual(svc._record_provenance, {"robot_vendor": "ur", "robot_model": "ur3e"})

    def test_no_record_path_leaves_provenance_unset(self) -> None:
        calc, perc = _calc_and_perception()
        svc = AutonomousGraspService.from_robot_config(_cfg("dummy", {"vendor": "none"}),
                                                       calculator=calc, perception=perc)  # type: ignore[arg-type]
        self.assertIsNone(svc._record_provenance)

    def test_provenance_reaches_the_logged_record(self) -> None:
        # End to end on a dummy arm (a real pick completes): the stamped identity must land in the
        # record's `extra`, and the frozen contract's required keys are untouched.
        calc, perc = _calc_and_perception()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            cfg = _cfg("dummy", {"vendor": "none"},
                       grasping={**_GRASPING, "record_log_path": str(path)})
            svc = AutonomousGraspService.from_robot_config(cfg, calculator=calc, perception=perc)  # type: ignore[arg-type]
            self.assertEqual(svc._record_provenance, {"robot_vendor": "dummy", "robot_model": "unknown"})
            svc.pick()
            self.assertTrue(path.exists(), "a record should have been written")
            rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            # provenance lands in the free-form `extra` bag (leaving the frozen required keys untouched).
            self.assertEqual(rec["extra"].get("robot_vendor"), "dummy")
            self.assertEqual(rec["extra"].get("robot_model"), "unknown")


if __name__ == "__main__":
    unittest.main()


class RealCellFrameResolverGateTests(unittest.TestCase):
    """A real cell must have a CAMERA->BASE resolver, or building the service refuses.

    Without one, `require_base_frame_grasp` is never switched on, the grasp stays `frame=camera`, and
    `URRobotArm.move` rejects EVERY motion as INVALID_TARGET. Safe -- and indistinguishable at the bench
    from a broken cell: 100% rejection with no hint that the cause is a missing calibration artifact.
    The shipped robot.yaml has `fusion.enabled: false` and no artifact path, so this is the DEFAULT
    state of a freshly configured real cell, which is exactly why it has to be loud.
    """

    def test_a_real_cell_without_any_resolver_refuses_to_build(self) -> None:
        calc, perc = _calc_and_perception()
        with patch(_READY), self.assertRaises(ValueError) as ctx:
            AutonomousGraspService.from_robot_config(
                _cfg("ur", {"vendor": "none"}), calculator=calc, perception=perc,  # type: ignore[arg-type]
            )
        msg = str(ctx.exception)
        self.assertIn("INVALID_TARGET", msg)                    # names the symptom the operator sees
        self.assertIn("extrinsics_artifact_path", msg)          # and both concrete fixes
        self.assertIn("frame_resolver", msg)

    def test_an_in_code_resolver_satisfies_it(self) -> None:
        """The eye-in-hand route: a live TCP-composed resolver cannot be serialized to an artifact."""
        calc, perc = _calc_and_perception()
        with patch(_READY):
            svc = AutonomousGraspService.from_robot_config(
                _cfg("ur", {"vendor": "none"}), calculator=calc, perception=perc,  # type: ignore[arg-type]
                frame_resolver=_resolver(),
            )
        self.assertIsNotNone(svc)

    def test_sim_and_dummy_are_exempt(self) -> None:
        """They legitimately wire a resolver in code after construction; refusing them would break the
        validated sim path for no safety gain."""
        for vendor in ("sim", "dummy"):
            with self.subTest(vendor=vendor):
                calc, perc = _calc_and_perception()
                # patch(_READY) only stubs the vendor-SDK doctor gate (sim needs Isaac); it has nothing
                # to do with the frame resolver this test is about.
                with patch(_READY):
                    svc = AutonomousGraspService.from_robot_config(
                        _cfg(vendor, {"vendor": "none"}), calculator=calc, perception=perc,  # type: ignore[arg-type]
                    )
                self.assertIsNotNone(svc)
