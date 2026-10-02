"""Every refusal by the planner's world names the boxes it met, and its depth says what it is: a sum.

The owner's cell, 2026-10-01 (fix plan RC6): R13's standoff was refused with "it reaches 37.0 mm into the world the
planner holds", and R2's with 393 mm on all eight goals. Neither is a depth. cuRobo sums its collision term over every
sphere that touches the world, so 37 mm was three spheres a few millimetres in each. And no line said which boxes the
pose met, so nobody could tell the mat strips from the pile from a box at the robot's base.

Now the sentence says "summed over every sphere that touches it", and the client keeps the world the sidecar last
confirmed: on a refusal by the world it says, at WARNING, which of those boxes lie within reach of the refused pose,
nearest first, each with its name, kind (a support's solid, the bench's, a camera box or a declared one), centre, size
and tilt with its quaternion, and how many. The goal of a Cartesian plan places the pose; a refused configuration is
placed by the bundled UR chain where the descriptor names a UR model.
"""

from __future__ import annotations

import logging
import math
import unittest
from typing import Any

import numpy as np

from src.robot.safety.planning.curobo_client import CuroboPlanClient, StateRefusal

_LOGGER = "CuroboPlanClient"


def _cuboid(name: str, centre_mm: "tuple[float, float, float]", dims_mm: "tuple[float, float, float]",
            tilt_deg: float = 0.0) -> "dict[str, Any]":
    """A box of the planner's world in its wire format: metres and a WXYZ quaternion, tilted about BASE x."""
    half = math.radians(tilt_deg) / 2.0
    return {"name": name, "dims_m": [v / 1000.0 for v in dims_mm],
            "pose": [*(v / 1000.0 for v in centre_mm), math.cos(half), math.sin(half), 0.0, 0.0]}


#: The owner's look at the pile, as the camera world registers it: the mat's tilted solid, a seen box of the pile, the
#: bench, and a box at the far end of the cell.
WORLD = [
    _cuboid("seen_s00_support0", (-150.0, -690.0, 28.0), (550.0, 360.0, 66.0), tilt_deg=1.0),
    _cuboid("seen_07", (-170.0, -620.0, 90.0), (60.0, 50.0, 60.0)),
    _cuboid("seen_s07_bench", (-10.0, -585.0, -21.5), (850.0, 660.0, 57.0)),
    _cuboid("shelf", (900.0, 900.0, 400.0), (200.0, 200.0, 20.0)),
]
#: The configuration of R13's refused goal, radians.
JOINTS: list[float] = [-1.5272, -1.1259, -1.4973, -2.2658, 1.5755, -1.8996]
#: A WORLD refusal as the sidecar writes it: R13's 37.0 mm, summed, at the goal.
_R13: "dict[str, Any]" = {"where": "goal", "kind": "world", "joints": JOINTS, "pair": None, "depth_mm": 37.0}


class _Wire:
    """Stands in for the pipe: records what the client sends and answers each request in turn."""

    def __init__(self, *replies: "dict[str, Any] | None") -> None:
        self.replies = list(replies)
        self.sent: list[dict[str, Any]] = []

    def send(self, request: "dict[str, Any]") -> int:
        self.sent.append(request)
        return len(self.sent)

    def recv(self, timeout_s: float, *, want: "int | None" = None) -> "dict[str, Any] | None":
        reply = self.replies.pop(0)
        return None if reply is None else {**reply, "id": want}


def _client(wire: _Wire, robot: str = "ur10.yml") -> CuroboPlanClient:
    client = CuroboPlanClient(python_path="no-python-is-spawned", robot_config=robot)
    client._proc = object()  # type: ignore[assignment]  # noqa: SLF001 (nothing is spawned)
    client._send = wire.send  # type: ignore[method-assign, assignment]  # noqa: SLF001
    client._recv = wire.recv  # type: ignore[method-assign]  # noqa: SLF001
    client.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                          "wrist_2_joint", "wrist_3_joint"]
    return client


def _refused(**refusal: Any) -> "dict[str, Any]":
    return {"success": False, "planner_error": False, "reason": "refused before planning",
            "refusal": {**_R13, **refusal}}


class TheDepthSaysItIsASumTests(unittest.TestCase):
    def test_a_world_refusal_says_summed_over_every_sphere(self) -> None:
        refusal = StateRefusal.from_reply(_R13)
        said = refusal.render()
        self.assertIn("37.0 mm into the world the planner holds", said)
        self.assertIn("summed over every sphere that touches it", said)

    def test_a_clearance_refusal_and_a_self_collision_keep_their_words(self) -> None:
        near = StateRefusal.from_reply({**_R13, "depth_mm": None, "clearance_mm": 4.0}).render()
        self.assertIn("comes closer than 4.0 mm", near)
        self.assertNotIn("summed", near)
        pair = StateRefusal.from_reply({**_R13, "kind": "self_collision", "pair": ["forearm_link", "wrist_2_link"],
                                        "depth_mm": 4.2}).render()
        self.assertIn("forearm_link and wrist_2_link overlap by 4.2 mm", pair)


class AWorldRefusalNamesItsBoxesTests(unittest.TestCase):
    def test_a_refused_goal_names_the_boxes_near_it_nearest_first(self) -> None:
        wire = _Wire({"world_set": len(WORLD)}, _refused())
        client = _client(wire)
        self.assertEqual(len(WORLD), client.set_world(WORLD))
        with self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            self.assertIsNone(client.plan([0.0] * 6, [-0.17, -0.62, 0.135], [0.0, 1.0, 0.0, 0.0]))
        line = next(entry for entry in said.output if "the world refused near" in entry)
        self.assertIn("the goal", line)
        self.assertIn("3 of the 4 box(es) it holds lie within 300 mm of it", line)
        self.assertLess(line.index("seen_07 [seen]"), line.index("seen_s00_support0 [support]"))
        self.assertIn("seen_s00_support0 [support]", line)
        self.assertIn("tilted 1.00 deg (wxyz 1.0000, 0.0087, 0.0000, 0.0000)", line)
        self.assertIn("seen_s07_bench [bench]", line)
        self.assertNotIn("shelf", line)
        self.assertIn("summed over every sphere", "\n".join(said.output))

    def test_a_refused_configuration_is_placed_by_the_ur_chain(self) -> None:
        wire = _Wire({"world_set": len(WORLD)}, _refused(where="goal"))
        client = _client(wire)
        client.set_world(WORLD)
        with self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            self.assertIsNone(client.plan_joint([0.0] * 6, list(JOINTS)))
        line = next(entry for entry in said.output if "the world refused near" in entry)
        self.assertIn("the arm's frames at the refused joints (ur10)", line)
        self.assertIn("box(es) it holds lie within 300 mm", line)

    def test_a_checked_path_refused_by_the_world_says_it_too(self) -> None:
        refused = {"success": True, "valid": False, "first_invalid": 3, "checked": 8,
                   "refusal": {**_R13, "where": "path"}}
        wire = _Wire({"world_set": len(WORLD)}, refused)
        client = _client(wire)
        client.set_world(WORLD)
        with self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            verdict = client.check_joints([list(JOINTS)] * 8)
        self.assertFalse(verdict.valid)
        self.assertTrue(any("the world refused near the arm's frames" in entry for entry in said.output))

    def test_a_self_collision_names_no_box(self) -> None:
        wire = _Wire({"world_set": len(WORLD)}, _refused(kind="self_collision", pair=["forearm_link", "hand"]))
        client = _client(wire)
        client.set_world(WORLD)
        with self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            client.plan([0.0] * 6, [-0.17, -0.62, 0.135], [0.0, 1.0, 0.0, 0.0])
        self.assertFalse(any("the world refused near" in entry for entry in said.output))

    def test_a_world_the_sidecar_did_not_confirm_is_not_the_one_said(self) -> None:
        """The planner keeps its previous world when a new one is not confirmed; so does what a refusal is said
        against."""
        wire = _Wire({"world_set": len(WORLD)}, None, _refused())
        client = _client(wire)
        client.set_world(WORLD)
        self.assertEqual(0, client.set_world([_cuboid("other", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))]))
        self.assertEqual([box["name"] for box in WORLD], [box["name"] for box in client.world_held])

    def test_a_refusal_with_no_world_registered_says_so(self) -> None:
        client = _client(_Wire(_refused()))
        with self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            client.plan([0.0] * 6, [-0.17, -0.62, 0.135], [0.0, 1.0, 0.0, 0.0])
        self.assertTrue(any("holds no world it registered" in entry for entry in said.output))

    def test_a_descriptor_that_names_no_ur_places_no_configuration(self) -> None:
        wire = _Wire({"world_set": len(WORLD)}, _refused())
        client = _client(wire, robot="franka.yml")
        client.set_world(WORLD)
        with self.assertLogs(_LOGGER, level=logging.WARNING) as said:
            client.plan_joint([0.0] * 6, list(JOINTS))
        self.assertTrue(any("names no arm this client can place: 4 box(es) held" in entry for entry in said.output))


class WorldNearTests(unittest.TestCase):
    """The distance a box is said at is its own: a point inside it is 0, beside its tilted top it is the analytic."""

    def test_distances(self) -> None:
        from src.robot.safety.planning.curobo_client import world_near

        near = dict((box["name"], d) for d, box in world_near(WORLD, [[-150.0, -690.0, 100.0]], reach_mm=1000.0))
        top = 28.0 + 33.0 / math.cos(math.radians(1.0))       # the tilted solid's top over its centre, nearly
        self.assertAlmostEqual(100.0 - top, near["seen_s00_support0"], delta=0.6)
        self.assertEqual(0.0, dict((box["name"], d) for d, box in world_near(WORLD, [[-170.0, -620.0, 90.0]]))[
            "seen_07"])
        self.assertNotIn("shelf", near)
        self.assertTrue(np.all(np.diff([d for d, _ in world_near(WORLD, [[0.0, -600.0, 100.0]])]) >= 0.0))


if __name__ == "__main__":
    unittest.main()
