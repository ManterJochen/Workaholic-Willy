"""A release after a grasp that modelled nothing reports DETACHED, and a sidecar that fails to forget a part still fails.

A cell whose ``safety.planning_world.payload`` is off grasps (payload NOT_MODELLED: nothing is handed to the
planner), then places or releases, and the hand verb reported DETACH_FAILED. The arm asked the sidecar to detach a
part it had never been given; a sidecar started with no sphere budget has no attachment link at all, and cuRobo's
``detach`` raises on it. Measured 2026-09-23 on the UR10 descriptor and this project's GPU: ``ValueError:
attached_object not found in spheres``, answered as ``detached: false``. The sidecar double below answers the same
way. The planner glue now answers True without asking where nothing was handed to its sidecar since it last
confirmed a detach, and still asks, and still fails, wherever something may have been.

An attach the sidecar did not confirm is still asked to detach: the client's answer cannot tell a refusal from a
reply that never came, and a timed-out attach may have landed.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

from src.robot.core.gripper import HoldEvidence
from src.robot.drivers.ur.curobo_motion import UR_ARM_JOINT_NAMES, CuroboUrPlanner
from src.robot.execution.robot import Robot
from src.robot.safety.planning import CuroboUnavailableError


class _Sidecar:
    """A cuRobo client that answers attach and detach the way the sidecar does, with or without a sphere budget."""

    def __init__(self, *, spheres: int = 0, attach_answer: "bool | None" = None) -> None:
        self.joint_names = list(UR_ARM_JOINT_NAMES)
        self.dt = 0.025
        self.spheres = spheres
        #: What attach answers; None answers the way the sidecar does for this budget.
        self.attach_answer = attach_answer
        self.detach_answer: "bool | None" = None
        self.detach_raises: "BaseException | None" = None
        self.holding = False
        self.calls: list[str] = []

    def start(self) -> None:
        self.calls.append("start")

    def close(self) -> None:
        self.calls.append("close")

    def attach_payload(self, joints: Any, dims_m: Any, pose: Any, **_: Any) -> bool:
        self.calls.append("attach")
        answer = self.spheres > 0 if self.attach_answer is None else self.attach_answer
        self.holding = self.holding or answer
        return answer

    def detach_payload(self) -> bool:
        self.calls.append("detach")
        if self.detach_raises is not None:
            raise self.detach_raises
        if self.detach_answer is not None:
            return self.detach_answer
        # cuRobo resets the attachment link's spheres; a robot built with no budget has no such link and raises,
        # which the sidecar answers as {"detached": false}.
        if self.spheres <= 0:
            return False
        self.holding = False
        return True


def _started(sidecar: _Sidecar) -> CuroboUrPlanner:
    """The glue as it stands after the first planned move: its sidecar started."""
    conn = MagicMock()
    conn.get_joint_positions.return_value = [0.0] * 6
    glue = CuroboUrPlanner(conn, client_factory=lambda: sidecar)
    glue._client_or_start()
    return glue


class TheGlueForgetsOnlyWhatItHandedOver(unittest.TestCase):
    def test_nothing_handed_over_is_detached_without_asking_the_sidecar(self) -> None:
        sidecar = _Sidecar(spheres=0)
        glue = _started(sidecar)
        self.assertTrue(glue.detach_payload())
        self.assertNotIn("detach", sidecar.calls)

    def test_a_confirmed_part_is_forgotten_by_the_sidecar_and_then_not_asked_again(self) -> None:
        sidecar = _Sidecar(spheres=16)
        glue = _started(sidecar)
        self.assertTrue(glue.attach_payload([0.0] * 6, (50.0, 50.0, 100.0), (0.0, 0.0, 150.0)))
        self.assertTrue(glue.detach_payload())
        self.assertEqual(1, sidecar.calls.count("detach"))
        self.assertTrue(glue.detach_payload())
        self.assertEqual(1, sidecar.calls.count("detach"), "a part already forgotten was asked about again")

    def test_a_sidecar_that_does_not_confirm_the_detach_is_still_a_failure_and_is_asked_again(self) -> None:
        sidecar = _Sidecar(spheres=16)
        glue = _started(sidecar)
        glue.attach_payload([0.0] * 6, (50.0, 50.0, 100.0), (0.0, 0.0, 150.0))
        sidecar.detach_answer = False
        self.assertFalse(glue.detach_payload())
        self.assertFalse(glue.detach_payload())
        self.assertEqual(2, sidecar.calls.count("detach"))

    def test_a_sidecar_gone_at_the_detach_is_a_failure(self) -> None:
        sidecar = _Sidecar(spheres=16)
        glue = _started(sidecar)
        glue.attach_payload([0.0] * 6, (50.0, 50.0, 100.0), (0.0, 0.0, 150.0))
        sidecar.detach_raises = CuroboUnavailableError("the cuRobo sidecar exited with code 1")
        with self.assertLogs("CuroboUrPlanner", level="ERROR"):
            self.assertFalse(glue.detach_payload())

    def test_an_attach_the_sidecar_did_not_confirm_is_still_asked_to_detach(self) -> None:
        """False is a refusal or a reply that never came, and a late attach may have landed."""
        sidecar = _Sidecar(spheres=16, attach_answer=False)
        glue = _started(sidecar)
        self.assertFalse(glue.attach_payload([0.0] * 6, (50.0, 50.0, 100.0), (0.0, 0.0, 150.0)))
        self.assertTrue(glue.detach_payload())
        self.assertEqual(1, sidecar.calls.count("detach"))

    def test_a_closed_sidecar_holds_nothing(self) -> None:
        sidecar = _Sidecar(spheres=16)
        glue = _started(sidecar)
        glue.attach_payload([0.0] * 6, (50.0, 50.0, 100.0), (0.0, 0.0, 150.0))
        glue.close()
        self.assertTrue(glue.detach_payload())
        glue._client_or_start()
        self.assertTrue(glue.detach_payload())
        self.assertNotIn("detach", sidecar.calls, "a restarted sidecar was asked to forget its predecessor's part")


class _Hand:
    """Jaws that close on a part nothing measures, and open."""

    max_width_mm = 50.0
    min_width_mm = 0.0
    is_connected = True

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def open(self, **_: Any) -> None:
        return None

    def close(self, **_: Any) -> None:
        return None

    def set_width_mm(self, width_mm: float, **_: Any) -> None:
        return None

    def get_width_mm(self) -> float:
        return 0.0

    def hold_evidence(self) -> HoldEvidence:
        return HoldEvidence.UNMEASURED

    def width_is_measured(self) -> bool:
        return False


class TheOwnersReleaseTests(unittest.TestCase):
    def test_a_release_after_a_grasp_that_modelled_nothing_reports_detached(self) -> None:
        from src.robot.execution import handling
        from tests.test_what_the_hand_verbs_read import _payload_ur

        arm = _payload_ur(enabled=False)
        sidecar = _Sidecar(spheres=0)
        arm._curobo_ur = _started(sidecar)
        robot = Robot.from_parts(arm=arm, gripper=_Hand(), lock_key=None)

        grasped = robot.grasp(0.0)
        self.assertIs(handling.PayloadState.NOT_MODELLED, grasped.payload, grasped.render())
        released = robot.release()

        self.assertIs(handling.PayloadState.DETACHED, released.payload, released.render())
        self.assertNotIn("detach", sidecar.calls)


if __name__ == "__main__":
    unittest.main()
