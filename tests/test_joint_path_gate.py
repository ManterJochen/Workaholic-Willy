"""A planned path is judged sample by sample, and a gate that cannot see the tool refuses to pretend.

The gate this replaced had an off switch and a stride, and every profile this repository shipped
turned it off. `gate_joint_path` has neither: it is reached only by a caller that already decided a
path exists, and a path that is not checked is exactly the hole the whole step is about. So its
questions are only about what it refuses and why.

Two of its refusals are not about the path at all. Without a self collision guard there is nothing to
ask, and with the capsule proxy the answer would be a fiction: the proxy models the arm as capsules and
cannot see the gripper on a joint only context, so a finger folded into the forearm passes it. The owner
chose exact meshes on every sample (Q4, 2026-09-11), and a gate that silently degraded to capsules would
report a check that never happened.

The exact mesh backend needs python-fcl or Coal and the baked mesh bundle. Where it is missing these
tests skip rather than assert something weaker, and say so.
"""

from __future__ import annotations

import unittest

import pytest

from src.config.schema.robot import RobotConfig, RobotSafetyConfig, WorkspaceLimitsConfig
from src.robot.core.capabilities import RobotCapabilities
from src.robot.core.motion_result import MotionCommand, MotionStatus
from src.robot.safety import SafetyPreflight
from src.robot.safety._fcl_self_collision import mesh_backend_status
from src.robot.safety.path_samples import PathSamples
from src.robot.safety.planning.hand import planner_hand

#: A configuration that folds a finger into the forearm. The exact mesh backend refuses it with
#: "forearm|lfinger: mesh distance 0.367 mm"; the capsule proxy cannot see the gripper at all here.
#: Both are the ones test_planning_world.py uses, so the two gates are compared on the same path.
_FOLDED = (1.95, 0.38, -1.33, -0.55, 2.00, 0.79)
_CLEAR = (0.0, -1.5, 1.5, 0.0, 0.0, 0.0)

_REACH_MM = 850.0


class _Arm:
    capabilities = RobotCapabilities(vendor="ur", model="ur5e", has_native_fk=True)


class _SimArm:
    """What the Isaac sim driver actually reports: a vendor and a model nobody has a DH table for."""

    capabilities = RobotCapabilities(vendor="sim", model="isaac-sim", is_simulated=True)


def _preflight(**overrides: object) -> SafetyPreflight:
    """A preflight whose only live guard is self collision, with the trajectory check OFF."""
    self_collision: dict[str, object] = {"fixtures": []}
    self_collision.update(overrides)
    safety = RobotSafetyConfig.model_validate({
        "payload": {"enforce": False},
        "self_collision": self_collision,
    })
    # The hand every committed arm bundle carries, named, so a declared kinematics model builds (Step 4f).
    hand = planner_hand(RobotConfig.model_validate({"gripper": {"model": "robotiq_2f85"}}))
    return SafetyPreflight.from_safety_config(safety, WorkspaceLimitsConfig(), hand=hand)


def _samples(*configs: tuple[float, ...]) -> PathSamples:
    return PathSamples(configs=tuple(configs), step_bound_mm=9.5)


def _mesh_backend_available() -> bool:
    return mesh_backend_status("ur5e") == "ok"


class ThePathIsJudgedTests(unittest.TestCase):
    """The reason this gate exists: the middle of a path, with the old switch off."""

    def test_a_collision_in_the_middle_is_caught(self) -> None:
        if not _mesh_backend_available():
            pytest.skip("no exact mesh backend on this box")
        preflight = _preflight()
        refused = preflight.gate_joint_path(
            _samples(_CLEAR, _FOLDED, _CLEAR), arm=_Arm(), command=MotionCommand.MOVE_JOINTS
        )
        self.assertIsNotNone(refused, "a path through a self collision was accepted")
        assert refused is not None
        self.assertEqual(MotionStatus.SELF_COLLISION_REJECTED, refused.status)

    def test_a_clear_path_is_accepted(self) -> None:
        if not _mesh_backend_available():
            pytest.skip("no exact mesh backend on this box")
        preflight = _preflight()
        self.assertIsNone(
            preflight.gate_joint_path(
                _samples(_CLEAR, _CLEAR, _CLEAR), arm=_Arm(), command=MotionCommand.MOVE_JOINTS
            )
        )

    def test_the_refusal_names_the_sample(self) -> None:
        """A refusal that says only that the path is bad leaves the reader to find where."""
        if not _mesh_backend_available():
            pytest.skip("no exact mesh backend on this box")
        refused = _preflight().gate_joint_path(
            _samples(_CLEAR, _CLEAR, _FOLDED, _CLEAR), arm=_Arm(), command=MotionCommand.MOVE_JOINTS
        )
        assert refused is not None
        self.assertIn("sample 3 of 4", refused.message or "")

    def test_the_refusal_carries_the_command_it_was_given(self) -> None:
        if not _mesh_backend_available():
            pytest.skip("no exact mesh backend on this box")
        refused = _preflight().gate_joint_path(
            _samples(_FOLDED,), arm=_Arm(), command=MotionCommand.MOVE_TO
        )
        assert refused is not None
        self.assertEqual(MotionCommand.MOVE_TO, refused.command)

    def test_an_empty_path_is_nothing_to_judge(self) -> None:
        self.assertIsNone(
            _preflight().gate_joint_path(
                PathSamples(configs=(), step_bound_mm=0.0),
                arm=_Arm(),
                command=MotionCommand.MOVE_JOINTS,
            )
        )


class AGateThatCannotSeeRefusesTests(unittest.TestCase):
    """Fail closed: the two ways the check would be a fiction are refusals, not warnings."""

    def test_without_a_self_collision_guard_the_path_is_refused(self) -> None:
        preflight = _preflight(enforce=False)
        refused = preflight.gate_joint_path(
            _samples(_CLEAR, _CLEAR), arm=_Arm(), command=MotionCommand.MOVE_JOINTS
        )
        self.assertIsNotNone(refused, "a path was judged by a preflight with nothing to judge it")
        assert refused is not None
        self.assertIn("self_collision", refused.message or "")
        self.assertIn("enforce", refused.message or "")

    def test_the_capsule_proxy_is_refused_by_name(self) -> None:
        preflight = _preflight(backend="capsule")
        refused = preflight.gate_joint_path(
            _samples(_CLEAR, _CLEAR), arm=_Arm(), command=MotionCommand.MOVE_JOINTS
        )
        self.assertIsNotNone(refused, "the capsule proxy was allowed to judge a path")
        assert refused is not None
        for word in ("capsule", "exact"):
            with self.subTest(word=word):
                self.assertIn(word, (refused.message or "").lower())

    def test_the_control_the_same_path_passes_on_the_mesh_backend(self) -> None:
        """Without this the refusal above could be the path rather than the backend."""
        if not _mesh_backend_available():
            pytest.skip("no exact mesh backend on this box")
        self.assertIsNone(
            _preflight().gate_joint_path(
                _samples(_CLEAR, _CLEAR), arm=_Arm(), command=MotionCommand.MOVE_JOINTS
            )
        )

    def test_the_engine_a_guard_would_run_is_public(self) -> None:
        from src.robot.safety import SelfCollisionGuard

        capsule = SelfCollisionGuard(
            RobotSafetyConfig.model_validate({"self_collision": {"backend": "capsule"}}).self_collision
        )
        self.assertIsNone(capsule.exact_mesh_engine(_Arm()))
        exact = SelfCollisionGuard(
            RobotSafetyConfig.model_validate({"self_collision": {}}).self_collision
        )
        engine = exact.exact_mesh_engine(_Arm())
        if _mesh_backend_available():
            self.assertIn(engine, ("coal", "fcl"))
        else:
            self.assertIsNone(engine)


class TheRobotTheSafetyLayerThinksItHoldsTests(unittest.TestCase):
    """Which robot a path is sampled against comes from the config first, then the arm.

    ⛔ MEASURED THE HARD WAY, 2026-09-12. The sampling asked the ARM for its model, and the Isaac sim
    arm answers `isaac-sim`, which is not a robot anybody has a DH table for. So no reach derived, the
    path gate refused every motion as UNSUPPORTED, and the M1 gate went from 10 of 10 to 0 of 10 in
    0.7 s a run, with no motion attempted and no refusal in the log that named the real cause.

    The self collision guard had it right all along: `safety.self_collision.kinematics_model` is what
    a cell says its arm IS for safety purposes, and the arm's own capabilities are the fallback. One
    answer, from one place, or the guards and the sampler are judging two different robots.
    """

    def test_a_sim_arm_with_a_declared_kinematics_model_gets_a_reach(self) -> None:
        preflight = _preflight(kinematics_model="ur5e")
        self.assertIsNotNone(
            preflight.joint_radii_mm(_SimArm()),
            "an arm whose cell declares its kinematics has no reach",
        )

    def test_the_declared_model_wins_over_the_arm(self) -> None:
        """A ur3e cell driving a ur5e-reporting arm is sampled as the ur3e it declared."""
        as_ur3e = _preflight(kinematics_model="ur3e").joint_radii_mm(_Arm())
        as_ur5e = _preflight(kinematics_model="ur5e").joint_radii_mm(_Arm())
        self.assertIsNotNone(as_ur3e)
        self.assertIsNotNone(as_ur5e)
        self.assertNotEqual(as_ur3e, as_ur5e, "the two arms came out the same size")

    def test_the_arm_answers_when_the_cell_declares_nothing(self) -> None:
        self.assertIsNotNone(_preflight().joint_radii_mm(_Arm()))

    def test_neither_says_and_the_path_is_refused(self) -> None:
        """The fail closed half: an arm nobody has identified is not judged, so it is not moved.

        The first thing missing is the exact mesh engine, which cannot load a bundle for a robot it
        cannot name, so that is the sentence the operator gets. The reach is missing too, and would
        say so if the meshes somehow resolved.
        """
        refused = _preflight().gate_planned_path(
            [list(_CLEAR), list(_CLEAR)], arm=_SimArm(), command=MotionCommand.MOVE_JOINTS
        )
        self.assertIsNotNone(refused)
        assert refused is not None
        self.assertIs(MotionStatus.UNSUPPORTED, refused.status)
        self.assertIn("exact mesh", (refused.message or "").lower())
        self.assertIsNone(_preflight().joint_radii_mm(_SimArm()))

    def test_a_sim_arm_with_a_declared_model_has_its_path_sampled(self) -> None:
        """The positive half. What it pins is that the path was SAMPLED, not that it passed.

        This preflight carries no joint limit table, so the guards refuse the sample they are handed.
        That refusal is the point: it names a sample, which means the line was turned into
        configurations rather than refused as unsampleable. Before the fix, a sim arm never got that
        far, and the whole M1 gate read 0 of 10 with no motion attempted.
        """
        if not _mesh_backend_available():
            pytest.skip("no exact mesh backend on this box")
        result = _preflight(kinematics_model="ur5e").gate_planned_path(
            [list(_CLEAR), list(_CLEAR)], arm=_SimArm(), command=MotionCommand.MOVE_JOINTS
        )
        if result is not None:
            self.assertIn("sample", result.message or "")
            self.assertNotIn("reach", result.message or "")


class ARefusalSaysSoWhereSomebodyWillReadItTests(unittest.TestCase):
    """A gate that refuses without a log line is a cell that stopped for no stated reason.

    ⛔ MEASURED 2026-09-12. The reach resolution defect above refused every motion in the Isaac sim,
    and the whole run produced not one line naming the cause: the guards log when they refuse a
    sample, but the two refusals that happen BEFORE any sample is judged said nothing at all. The
    evidence an operator had was a pass rate of zero and a log full of successful perception.
    """

    def _refuse_and_capture(self, **overrides: object) -> list[str]:
        import logging

        lines: list[str] = []

        class _Recorder(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                lines.append(record.getMessage())

        preflight = _preflight(**overrides)
        handler = _Recorder()
        preflight._logger.addHandler(handler)
        try:
            refused = preflight.gate_planned_path(
                [list(_CLEAR), list(_CLEAR)], arm=_Arm(), command=MotionCommand.MOVE_JOINTS
            )
        finally:
            preflight._logger.removeHandler(handler)
        self.assertIsNotNone(refused, "this case was supposed to refuse")
        return lines

    def test_a_missing_guard_is_logged(self) -> None:
        said = self._refuse_and_capture(enforce=False)
        self.assertTrue(any("self_collision" in line for line in said), said)

    def test_the_capsule_proxy_is_logged(self) -> None:
        said = self._refuse_and_capture(backend="capsule")
        self.assertTrue(any("capsule" in line.lower() for line in said), said)

    def test_the_control_an_accepted_path_says_nothing(self) -> None:
        """Otherwise a cell that works would print a line per motion, which teaches nobody anything."""
        import logging

        if not _mesh_backend_available():
            pytest.skip("no exact mesh backend on this box")
        lines: list[str] = []

        class _Recorder(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                lines.append(record.getMessage())

        preflight = _preflight()
        handler = _Recorder()
        preflight._logger.addHandler(handler)
        try:
            preflight.gate_planned_path(
                [list(_CLEAR), list(_CLEAR)], arm=_Arm(), command=MotionCommand.MOVE_JOINTS
            )
        finally:
            preflight._logger.removeHandler(handler)
        self.assertEqual([], lines)


class TheMemoIsCleanOnBothSidesTests(unittest.TestCase):
    """A judged path is a restart, and it leaves nothing behind that would refuse the next move."""

    def test_the_memo_is_none_after_an_accepted_path(self) -> None:
        if not _mesh_backend_available():
            pytest.skip("no exact mesh backend on this box")
        preflight = _preflight()
        preflight.gate_joint_path(
            _samples(_CLEAR, _CLEAR), arm=_Arm(), command=MotionCommand.MOVE_JOINTS
        )
        self.assertIsNone(preflight._last_target_joints)
        self.assertIsNone(preflight._last_target_pose)

    def test_the_memo_is_none_after_a_refused_path(self) -> None:
        if not _mesh_backend_available():
            pytest.skip("no exact mesh backend on this box")
        preflight = _preflight()
        refused = preflight.gate_joint_path(
            _samples(_CLEAR, _FOLDED), arm=_Arm(), command=MotionCommand.MOVE_JOINTS
        )
        self.assertIsNotNone(refused)
        self.assertIsNone(preflight._last_target_joints)
        self.assertIsNone(preflight._last_target_pose)


if __name__ == "__main__":
    unittest.main()
