"""The exact guard decides the planner's self pairs it judges, where it accepted the configuration (F1, 2026-09-30).

At the owner's LOOK[0], (-42.5, -67.9, -78.6, -136.2, 94.6, -34.1) deg on the UR10 with the Hand-E, the exact mesh
guard keeps 19.04 mm on forearm|wrist_2 against its 10 mm and the planner's sphere model, padded by the 4 mm margin,
overlaps the same pair by 1.2 mm. Every motion out of that pose was refused before anything was sent. The owner's
decision: for arm against arm, and hand or wrist camera against arm, the exact guard decides; the carried part, a pair
the guard does not check, the planner's world and its joint bounds stay the planner's.

What this file holds, on the CPU:

* which planner link is which of the guard's parts, and that the pair rule is the guard backend's own (not a copy): a
  pair on neighbouring DH frames is not the guard's unless a wrist camera is in it, one on the same frame never is;
* why a refusal stands (:func:`admission_refusal`): bounds, world, the carried part, an unnamed pair, a pair the guard
  does not check, and only the rest is left to the guard;
* which joints turn one link of a pair against the other, and the neighbours of a band pose: at most 20 degrees per
  joint (the owner, 2026-09-30, after LOOK[1]), on those joints alone, one check request at most, nearest by the path
  gate's own measure;
* the screen's line: ERROR where no move goes, the nearby pose in degrees.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from src.contracts import UNSET
from src.robot.safety._fcl_self_collision import MeshSelfCollisionBackend
from src.robot.safety.planning._curobo_pairs import PairOverlap
from src.robot.safety.planning.band import (
    BAND_LEG_MAX_DEG,
    NEIGHBOUR_BUDGET,
    ExactPairs,
    PoseScreen,
    PoseVerdict,
    admission_refusal,
    band_neighbours,
    band_sentence,
    degrees_line,
    guard_parts,
    joints_between,
)
from src.robot.safety.planning.curobo_client import RefusedSample

#: The parts an owner-like exact guard holds and the DH frame each hangs from: the UR10 bundle, the Hand-E's three
#: parts on the flange, and a wrist camera's housing.
_FRAMES = {"shoulder": 1, "upper_arm": 2, "forearm": 3, "wrist_1": 4, "wrist_2": 5, "wrist_3": 6,
           "gripper": 6, "lfinger": 6, "rfinger": 6, "wrist_camera_wrist": 6}
_CAMERA = frozenset({"wrist_camera_wrist"})


class _RecordingAdapter:
    """An engine that places nothing and answers every distance with a kilometre, recording which pairs it was asked."""

    kind = "fake"

    def __init__(self) -> None:
        self.asked: set[frozenset[str]] = set()
        self._names: dict[int, str] = {}

    def build_object(self, verts: np.ndarray, faces: np.ndarray) -> object:
        return object()

    def set_transform(self, obj: object, R: np.ndarray, t: np.ndarray) -> None:
        return None

    def box_object(self, half_extents: np.ndarray, center: np.ndarray) -> object:
        return object()

    def distance(self, a: object, b: object) -> float:
        self.asked.add(frozenset((self._names[id(a)], self._names[id(b)])))
        return 1.0e6


def _backend() -> tuple[MeshSelfCollisionBackend, _RecordingAdapter]:
    adapter = _RecordingAdapter()
    meshes = {name: (np.zeros((3, 3)), np.zeros((1, 3), dtype=np.int64), frame) for name, frame in _FRAMES.items()}
    backend = MeshSelfCollisionBackend(adapter, meshes, wrist_parts=_CAMERA)  # type: ignore[arg-type]
    adapter._names = {id(backend._models[name]): name for name in _FRAMES}  # noqa: SLF001
    return backend, adapter


def _exact(backend: MeshSelfCollisionBackend) -> ExactPairs:
    return ExactPairs(checks=backend.checks, frames=backend.part_frames,
                      distance=lambda joints, a, b: 19.04 if {a, b} == {"forearm", "wrist_2"} else 50.0,
                      min_distance_mm=10.0)


#: The owner-like cell's planner_margin_mm: every link the planner pads carries half of it, so a pair of two carries it
#: whole (``_curobo_margin``).
_MARGIN_MM = 4.0


def _row(*pairs: "tuple[str, str]", bound_ok: bool = True, world_ok: bool = True, self_ok: bool = False,
         named: bool = True, padding_mm: float = _MARGIN_MM) -> RefusedSample:
    overlaps = tuple(PairOverlap(link_a=a, link_b=b, depth_mm=-2.79 + padding_mm, unpadded_depth_mm=-2.79)
                     for a, b in pairs)
    return RefusedSample(index=0, bound_ok=bound_ok, self_ok=self_ok, world_ok=world_ok,
                         pairs=overlaps if named else UNSET)


class ThePairRuleIsTheGuardsOwnTests(unittest.TestCase):
    def test_the_rule_asked_alone_is_the_rule_evaluate_judges_by(self) -> None:
        """⭐ THE INVARIANT the admission rests on: ``checks`` is not a second copy of the pair rule, it IS the rule the
        backend evaluates by, so a pair the admission leaves to the guard is a pair the guard judged."""
        backend, adapter = _backend()
        self.assertIsNone(backend.evaluate([np.eye(4)] * 7, 0.0, (), 10.0))
        rule = {frozenset((a, b)) for a in _FRAMES for b in _FRAMES if a < b and backend.checks(a, b)}
        self.assertEqual(adapter.asked, rule)
        self.assertIn(frozenset(("forearm", "wrist_2")), rule)

    def test_neighbouring_frames_are_skipped_unless_a_wrist_camera_is_in_the_pair(self) -> None:
        backend, _ = _backend()
        self.assertTrue(backend.checks("forearm", "wrist_2"))
        self.assertFalse(backend.checks("wrist_2", "gripper"), "a hand part one frame from wrist_2 is skipped")
        self.assertTrue(backend.checks("wrist_2", "wrist_camera_wrist"), "a wrist camera stays checked against it")
        self.assertFalse(backend.checks("wrist_3", "gripper"), "one frame is never a pair")
        self.assertFalse(backend.checks("forearm", "forearm"))
        self.assertFalse(backend.checks("forearm", "plate_nobody_holds"))
        self.assertEqual(backend.part_frames["wrist_2"], 5)


class WhichPlannerLinkIsWhichPartTests(unittest.TestCase):
    def test_the_links_the_guard_holds(self) -> None:
        self.assertEqual(guard_parts("forearm_link"), ("forearm",))
        self.assertEqual(guard_parts("wrist_2_link"), ("wrist_2",))
        self.assertEqual(guard_parts("hand"), ("gripper", "lfinger", "rfinger"))
        self.assertEqual(guard_parts("wrist_camera_wrist"), ("wrist_camera_wrist",))

    def test_the_links_it_does_not(self) -> None:
        """⭐ The carried part is the planner's alone, and so is anything this does not know."""
        for link in ("attached_object", "coupling", "tool0", "base_link", "wrist_camera_", "elbow"):
            with self.subTest(link=link):
                self.assertEqual(guard_parts(link), ())

    def test_the_shoulder_link_is_never_the_guards_because_the_guard_holds_no_base(self) -> None:
        """⭐ THE BASE (review of F1, 2026-09-30). The guard's UR bundles start at the shoulder, whose mesh begins
        38.3 mm above the base plate on a UR10, and hold no part for the base below it. The planner's shoulder_link is
        more than the shoulder: cuRobo's stock 70 mm cushion on it (``self_collision_buffer``) and spheres reaching down
        to 18 mm are the only model of the base anywhere. Left to the guard, hand|shoulder_link admitted the Hand-E
        touching the UR10's base. So no pair with the shoulder_link is ever the guard's to decide."""
        self.assertEqual(guard_parts("shoulder_link"), ())
        exact = _exact(_backend()[0])
        for other in ("hand", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link",
                      "wrist_camera_wrist"):
            with self.subTest(other=other):
                self.assertFalse(exact.decides(other, "shoulder_link"))
                self.assertFalse(exact.decides("shoulder_link", other))
        self.assertIsNone(exact.frame("shoulder_link"))

    def test_every_committed_arm_bundle_holds_the_parts_the_planner_links_map_onto(self) -> None:
        """⭐ The map from the planner's link names to the guard's parts is a table here; a bundle that named its parts
        otherwise would leave every pair to the planner in silence. Every committed UR bundle holds them, one frame
        apart from base to flange, and the hand's three on the flange."""
        from pathlib import Path

        from src.robot.safety.planning.band import _ARM_PARTS

        bundles = sorted((Path(__file__).resolve().parents[1] / "src" / "robot" / "safety" / "data")
                         .glob("ur*_collision_meshes.npz"))
        self.assertTrue(bundles)
        want = {"shoulder": 1, "upper_arm": 2, "forearm": 3, "wrist_1": 4, "wrist_2": 5, "wrist_3": 6,
                "gripper": 6, "lfinger": 6, "rfinger": 6}
        # The shoulder is in every bundle and is never mapped: the planner's shoulder_link covers the base too.
        self.assertEqual({part for parts in _ARM_PARTS.values() for part in parts} | {"gripper", "lfinger", "rfinger"},
                         set(want) - {"shoulder"})
        for path in bundles:
            with self.subTest(bundle=path.name), np.load(path, allow_pickle=True) as data:
                frames = {name: int(np.asarray(data[f"{name}__frame"]).reshape(-1)[0]) for name in want}
                self.assertEqual(frames, want)
                # The reason the shoulder_link stays the planner's: no bundle holds the base. One that did would be
                # the moment to map the shoulder_link onto the shoulder and the base together, and not before.
                self.assertEqual(sorted({key.split("__")[0] for key in data.files} - set(want)), [])

    def test_a_pair_is_the_guards_where_every_part_of_it_is_checked(self) -> None:
        exact = _exact(_backend()[0])
        self.assertTrue(exact.decides("forearm_link", "wrist_2_link"))
        self.assertTrue(exact.decides("wrist_1_link", "hand"))
        self.assertTrue(exact.decides("wrist_2_link", "wrist_camera_wrist"))
        self.assertTrue(exact.decides("upper_arm_link", "wrist_camera_wrist"))
        self.assertFalse(exact.decides("shoulder_link", "wrist_camera_wrist"), "the shoulder_link covers the base")
        self.assertFalse(exact.decides("wrist_2_link", "hand"), "the guard skips the hand against wrist_2")
        self.assertFalse(exact.decides("wrist_3_link", "hand"))
        self.assertFalse(exact.decides("hand", "attached_object"))
        self.assertFalse(exact.decides("wrist_1_link", "coupling"))
        self.assertFalse(exact.decides("wrist_1_link", "wrist_camera_other"), "a camera the guard does not hold")
        self.assertEqual(exact.frame("hand"), 6)
        self.assertIsNone(exact.frame("attached_object"))
        self.assertAlmostEqual(exact.distance_mm([0.0] * 6, "forearm_link", "wrist_2_link") or 0.0, 19.04)
        self.assertIsNone(exact.distance_mm([0.0] * 6, "forearm_link", "attached_object"))


class WhenTheRefusalStandsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.exact = _exact(_backend()[0])

    def test_look0_is_the_guards_to_decide(self) -> None:
        self.assertIsNone(admission_refusal(_row(("forearm_link", "wrist_2_link")), self.exact, margin_mm=_MARGIN_MM))
        self.assertIsNone(admission_refusal(_row(("forearm_link", "wrist_2_link"), ("wrist_1_link", "hand")),
                                            self.exact, margin_mm=_MARGIN_MM))
        # Even where the spheres alone overlap (LOOK[1] at some wrist_2): the owner's decision leaves the pair whole.
        deep = RefusedSample(index=0, bound_ok=True, self_ok=False, world_ok=True,
                             pairs=(PairOverlap("forearm_link", "wrist_2_link", 5.2, 1.2),))
        self.assertIsNone(admission_refusal(deep, self.exact, margin_mm=_MARGIN_MM))

    def test_everything_else_stays_the_planners(self) -> None:
        cases = {
            "bounds": _row(("forearm_link", "wrist_2_link"), bound_ok=False),
            "world": _row(("forearm_link", "wrist_2_link"), world_ok=False),
            "the carried part": _row(("forearm_link", "wrist_2_link"), ("attached_object", "forearm_link")),
            "a pair the guard skips": _row(("hand", "wrist_2_link")),
            "the robot's base": _row(("hand", "shoulder_link"), padding_mm=_MARGIN_MM + 70.0),
            "an unnamed pair": _row(("forearm_link", "wrist_2_link"), named=False),
            "no pair named": _row(),
            "no self collision": _row(self_ok=True, bound_ok=False),
        }
        for what, row in cases.items():
            with self.subTest(what=what):
                self.assertIsNotNone(admission_refusal(row, self.exact, margin_mm=_MARGIN_MM))
        self.assertIn("carried part", admission_refusal(cases["the carried part"], self.exact,
                                                        margin_mm=_MARGIN_MM) or "")

    def test_only_the_margins_padding_is_ever_lifted(self) -> None:
        """⭐ WHICH BUFFERS THE ADMISSION MAY LIFT (review of F1, 2026-09-30): the planner_margin_mm padding and the
        spheres' own reach past the meshes, on pairs the guard judges, and nothing else. A pair padded by more than the
        margin carries a cushion the descriptor put there for something else (cuRobo's 70 mm on the shoulder_link, which
        covers the base), and stays the planner's whatever the two links are. Read off the report itself: the padding
        of a pair is its depth with the padding less its depth without."""
        at_the_margin = _row(("forearm_link", "wrist_2_link"), padding_mm=_MARGIN_MM)
        self.assertIsNone(admission_refusal(at_the_margin, self.exact, margin_mm=_MARGIN_MM))
        for padding in (_MARGIN_MM + 0.01, 8.0, 74.0):
            with self.subTest(padding_mm=padding):
                beyond = _row(("forearm_link", "wrist_2_link"), padding_mm=padding)
                said = admission_refusal(beyond, self.exact, margin_mm=_MARGIN_MM)
                self.assertIsNotNone(said)
                self.assertIn(f"{padding:g} mm", said or "")
                self.assertIn("margin", said or "")
        # A smaller margin lifts less: the same pair padded by 4 mm stands where the cell declares 2.
        self.assertIsNotNone(admission_refusal(at_the_margin, self.exact, margin_mm=2.0))

    def test_the_sentence_names_the_depths_and_the_exact_distance(self) -> None:
        said = band_sentence(_row(("forearm_link", "wrist_2_link")), self.exact, [0.0] * 6)
        for expected in ("forearm_link and wrist_2_link", "1.2 mm", "2.8 mm apart", "19.0 mm", "10 mm"):
            self.assertIn(expected, said)


class TheNeighboursOfABandPoseTests(unittest.TestCase):
    LOOK0 = tuple(math.radians(v) for v in (-42.5, -67.9, -78.6, -136.2, 94.6, -34.1))
    RADII = (1856.7, 1729.4, 1117.4, 545.1, 381.1, 92.2)

    def test_forearm_and_wrist_2_turn_on_wrist_1_and_wrist_2(self) -> None:
        exact = _exact(_backend()[0])
        pair = PairOverlap("forearm_link", "wrist_2_link", 1.21, -2.79)
        self.assertEqual(joints_between([pair], exact), (3, 4))
        self.assertEqual(joints_between([PairOverlap("forearm_link", "hand", 1.0, -1.0)], exact), (3, 4, 5))
        self.assertEqual(joints_between([PairOverlap("forearm_link", "attached_object", 1.0, -1.0)], exact),
                         tuple(range(6)))

    def test_the_cap_is_twenty_degrees(self) -> None:
        """⭐ THE OWNER'S NUMBER (2026-09-30, after the fix round): LOOK[1]'s nearest pose both authorities clear is
        wrist_1 +19.5 degrees away, so the cap went from 10 to 20 degrees per joint."""
        self.assertEqual(BAND_LEG_MAX_DEG, 20.0)

    def test_no_neighbour_turns_a_joint_more_than_twenty_degrees_or_turns_another_joint(self) -> None:
        """⭐ THE CAP (the owner, 2026-09-30): an escape or approach leg turns no joint more than 20 degrees, and the
        grid reaches the cap on every joint it turns."""
        for joints in ((3,), (3, 4), (3, 4, 5), (1, 2, 3, 4), tuple(range(6))):
            neighbours = band_neighbours(self.LOOK0, joints, self.RADII)
            with self.subTest(joints=joints):
                self.assertTrue(0 < len(neighbours) <= NEIGHBOUR_BUDGET)
                self.assertNotIn(self.LOOK0, neighbours)
                for values in neighbours:
                    for index, (a, b) in enumerate(zip(values, self.LOOK0)):
                        if index in joints:
                            self.assertLessEqual(abs(math.degrees(a - b)), 20.0 + 1e-9)
                        else:
                            self.assertEqual(a, b)
                for joint in joints:
                    reach = max(abs(math.degrees(values[joint] - self.LOOK0[joint])) for values in neighbours)
                    self.assertAlmostEqual(reach, 20.0, places=9)

    def test_the_grid_is_one_check_request_whatever_the_joints_and_widens_its_step_to_stay_one(self) -> None:
        """At most ``NEIGHBOUR_BUDGET`` (1000) configurations, one planner check request, however many joints turn: the
        step is as fine as that allows and never finer than half a degree. Twice the reach costs a coarser step on two
        joints and more, never a second request."""
        steps = {1: 0.5, 2: 4.0 / 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 20.0}
        for count, step in steps.items():
            joints = tuple(range(count))
            neighbours = band_neighbours(self.LOOK0, joints, self.RADII)
            with self.subTest(joints=count):
                per_axis = 2 * round(20.0 / step) + 1
                self.assertEqual(len(neighbours), per_axis ** count - 1)
                self.assertLessEqual(len(neighbours), NEIGHBOUR_BUDGET)
                finest = min(abs(math.degrees(values[0] - self.LOOK0[0])) for values in neighbours
                             if abs(values[0] - self.LOOK0[0]) > 1e-12)
                self.assertAlmostEqual(finest, step, places=9)

    def test_they_come_nearest_first_by_the_path_gates_measure(self) -> None:
        neighbours = band_neighbours(self.LOOK0, (3, 4), self.RADII)

        def travel(values: "tuple[float, ...]") -> float:
            return sum(abs(a - b) * r for a, b, r in zip(values, self.LOOK0, self.RADII))

        self.assertEqual([travel(v) for v in neighbours], sorted(travel(v) for v in neighbours))
        # The cheapest single step turns wrist_2 (381 mm) before wrist_1 (545 mm).
        self.assertNotEqual(neighbours[0][4], self.LOOK0[4])
        # Two joints get a grid of 31 x 31 less the pose itself, at four thirds of a degree up to 20.
        self.assertEqual(len(neighbours), 31 * 31 - 1)
        self.assertEqual(len(band_neighbours(self.LOOK0, (3,), self.RADII)), 80, "half a degree on one joint")


def owner_like_arm() -> "object":
    """The owner's cell as far as the guard reads it: a UR10 on cuRobo, the Hand-E on a 20 mm plate, approach +Z and
    closing +X on the flange (``ur10_robotiq_hande_c20mm_+Z+X_m4mm``), the 4 mm margin."""
    from src.config.schema.robot import RobotConfig
    from src.robot.drivers.ur.arm import URRobotArm

    from tests._plan_end import OPEN_WORKSPACE

    return URRobotArm(RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": "ur10", "motion_planner": "curobo"},
        # Where the owner's looks put the tool is the cell's box to say, and this file is about the pairs.
        "workspace_limits": OPEN_WORKSPACE,
        "gripper": {"model": "robotiq_hande", "coupling_plates": [{"name": "adapter", "thickness_mm": 20.0}],
                    "tool_frame": {"source": "willy", "offset_mm": [0.0, 0.0, 155.75],
                                   "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "safety": {"payload": {"enforce": False},
                   "self_collision": {"backend": "fcl", "min_distance_mm": 10.0, "kinematics_model": "ur10",
                                      "planner_margin_mm": 4.0}},
    }))


LOOK0_DEG = (-42.5, -67.9, -78.6, -136.2, 94.6, -34.1)
#: LOOK[0] with the wrist folded where forearm|wrist_2 really touch (research: 0 mm at wrist_1 80, wrist_2 -140).
FOLDED_DEG = (-42.5, -67.9, -78.6, 80.0, -140.0, -34.1)


class TheRealGuardAtLook0Tests(unittest.TestCase):
    """The same questions of the exact guard an owner-like cell builds, on the committed UR10 and Hand-E bundles."""

    def setUp(self) -> None:
        from src.robot.safety._fcl_self_collision import mesh_backend_status

        if mesh_backend_status("ur10", mesh_name="robotiq_hande") != "ok":
            self.skipTest("no exact mesh backend on this box")
        self.arm: Any = owner_like_arm()
        exact = self.arm.safety_preflight.exact_pairs(self.arm)
        assert exact is not None
        self.exact = exact

    def test_the_guard_keeps_19_mm_on_forearm_and_wrist_2_at_look0(self) -> None:
        look0 = [math.radians(v) for v in LOOK0_DEG]
        self.assertAlmostEqual(self.exact.distance_mm(look0, "forearm_link", "wrist_2_link") or 0.0, 19.04, delta=0.05)
        self.assertTrue(self.exact.decides("forearm_link", "wrist_2_link"))
        self.assertFalse(self.exact.decides("wrist_2_link", "hand"))
        self.assertTrue(self.exact.decides("wrist_1_link", "hand"))
        self.assertEqual(self.exact.min_distance_mm, 10.0)

    def test_the_folded_wrist_is_really_in_contact(self) -> None:
        """⭐ THE CONTROL: the band is not everything. Where the meshes touch, the guard refuses whatever the planner
        says, and the admission never sees such a pose accepted."""
        from src.robot.core import JointPositions

        folded = JointPositions(tuple(math.radians(v) for v in FOLDED_DEG))
        touching = self.exact.distance_mm(list(folded.values), "forearm_link", "wrist_2_link")
        assert touching is not None
        self.assertLess(touching, 10.0)
        self.assertIsNotNone(self.arm.safety_preflight.gate_joint_target(folded, arm=self.arm))
        look0 = JointPositions(tuple(math.radians(v) for v in LOOK0_DEG))
        self.assertIsNone(self.arm.safety_preflight.gate_joint_target(look0, arm=self.arm))


class TheComposedConfigIsTheOneTheEvidenceMeasuredTests(unittest.TestCase):
    """⭐ The owner's decision keeps the composed config, its hash and the 4 mm margin: the evidence stays valid. The
    planner start refuses a sidecar whose ``composed_sha256`` is not the file's; this composes the owner-like cell as
    the sidecar does, where the descriptor is installed, and reads the hash the committed file names."""

    def test_the_owner_like_cell_composes_to_the_hash_its_evidence_names(self) -> None:
        import json

        import yaml

        from src.robot.safety.planning._curobo_body_links import canonical_sha256, compose_for_cell
        from src.robot.safety.planning.body_link import HandLink, coupling_bodies
        from src.robot.safety.planning.environment import project_root
        from src.robot.safety.planning.evidence import EVIDENCE_DIR
        from src.robot.safety.planning.robot.retract_table import read_retract

        descriptor = project_root() / "ext_deps" / "curobo" / "curobo" / "content" / "configs" / "robot" / "willy_ur10.yml"
        if not descriptor.exists():
            self.skipTest(f"no installed cuRobo descriptor at {descriptor}")
        arm: Any = owner_like_arm()
        hand = arm.safety_preflight.planner_hand(arm)
        link = HandLink.from_hand(hand)
        default_q = read_retract("ur10", hand.model, float(hand.coupling_mm), 4.0,
                                 placement=f"{link.placement.approach}{link.placement.closing}")
        _, evidence, _, _, _ = compose_for_cell(
            yaml.safe_load(descriptor.read_text(encoding="utf-8")), bodies=[link.to_dict(), *coupling_bodies(hand)],
            margin_mm=4.0, attach_spheres=0, default_q=default_q)
        committed = json.loads((EVIDENCE_DIR / "ur10_robotiq_hande_c20mm_+Z+X_m4mm_a0.json").read_text(encoding="utf-8"))
        self.assertEqual(canonical_sha256(evidence), committed["composed_sha256"])


class TheScreenSaysItInOneLineTests(unittest.TestCase):
    def test_an_error_says_so_first_and_a_nearby_pose_in_degrees(self) -> None:
        refused = PoseScreen(PoseVerdict.GUARD_REFUSED, "forearm|wrist_2: mesh distance 4.1 mm < 10.000 mm.",
                             nearby=tuple(math.radians(v) for v in (0.0, -90.0, 0.0, -1e-6, 90.0, 0.0)))
        line = refused.line("LOOK_1")
        self.assertTrue(line.startswith("ERROR LOOK_1: refused by the exact guard"))
        self.assertIn("(0.0, -90.0, 0.0, 0.0, 90.0, 0.0) deg", line)
        self.assertNotIn("-0.0", line)
        self.assertTrue(refused.is_error)

    def test_a_band_pose_is_no_error(self) -> None:
        band = PoseScreen(PoseVerdict.BAND, "straight lines run into and out of it.")
        self.assertFalse(band.is_error)
        self.assertEqual(band.line("LOOK_0"), "LOOK_0: in the planner's cushion band: straight lines run into and "
                                                "out of it.")
        self.assertFalse(PoseScreen(PoseVerdict.CLEAR, "both clear it.").is_error)
        self.assertTrue(PoseScreen(PoseVerdict.PLANNER_REFUSED, "its world.").is_error)
        self.assertEqual(degrees_line([0.0, math.pi]), "(0.0, 180.0) deg")


if __name__ == "__main__":
    unittest.main()
