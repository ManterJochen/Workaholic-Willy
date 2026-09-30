"""A pose only the planner's padded spheres refuse is the exact guard's to decide, and a plan leaves it by a short leg.

The owner's LOOK[0] on the UR10 with the Hand-E, (-42.5, -67.9, -78.6, -136.2, 94.6, -34.1) deg: the exact meshes keep
19.0 mm on forearm|wrist_2 and the planner's spheres, padded by the 4 mm margin, overlap by 1.2 mm. Every motion out of
it was refused before anything was sent, the move back to it too. The owner's decision, 2026-09-30:

1. The exact guard decides the planner's self pairs it judges, where it accepted the very configuration: at the goal
   screen, on every straight line and plan leg (``_judge_legs``) and on a tool line (moveL). The planner's world, its
   bounds and the carried part stay its own, and anything the driver cannot read whole leaves the refusal standing.
2. A PLANNED move out of such a pose, or into one, takes a straight leg of at most 20 degrees per joint (the owner,
   2026-09-30: 10 at first, 20 once LOOK[1] needed wrist_1 +19.5) to the nearest configuration both authorities clear,
   judged by both, and cuRobo plans from there (``plan_joint`` with an explicit start). A line-only move never does.
3. A start that no leg leaves says what the two authorities found there: the pair, the planner's depth, the distance
   the exact meshes keep.

And from the review of F1 (2026-09-30): the shoulder_link, the planner's only model of the robot's own base, is never
the guard's, nor is padding past the cell's margin; a plan that fails after a judged escape is that goal's failure,
never the start's; the escape and approach legs run as the moveJ they were judged as; a refusal names the sample and
the term it stands on.

The exact guard here is the real one, on the committed UR10 and Hand-E bundles; the planner is a double that answers
as the sidecar does in the band: a wrist_1 between -140.5 and -101 degrees is a forearm_link|wrist_2_link self
collision to its padded spheres.
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.contracts import UNSET
from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionCommand, MotionResult, MotionStatus
from src.robot.drivers.ur.pose_adapter import pose_to_urpose
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from src.robot.safety._fcl_self_collision import mesh_backend_status
from src.robot.safety.planning import JointCheckVerdict, StateRefusal, StateRefusalKind, StateWhere
from src.robot.safety.planning._curobo_pairs import PairOverlap
from src.robot.safety.planning.curobo_client import PathJudgement, RefusedSample
from tests._plan_end import pose_where_it_ends
from tests._route_planner import line
from tests.test_the_exact_guard_decides_the_planners_self_pairs import FOLDED_DEG, LOOK0_DEG, owner_like_arm

_DECLINED = "unit double: this file reads what the exact guard decides on a cuRobo arm, and no camera world is wired"

#: The owner's leg cap, degrees per joint (``band.BAND_LEG_MAX_DEG``): pinned here as a number, so a change to it is a
#: change to this file too.
_CAP_DEG = 20.0


def _deg(*values: float) -> list[float]:
    return [math.radians(v) for v in values]


LOOK0 = _deg(*LOOK0_DEG)
#: LOOK[0] with wrist_1 at -125 deg: still in the band, and a line from LOOK[0] to it is in the band throughout.
LOOK0_DEEPER = _deg(-42.5, -67.9, -78.6, -125.0, 94.6, -34.1)
#: LOOK[0] with wrist_1 at -150 deg: out of the band.
CLEAR = _deg(-42.5, -67.9, -78.6, -150.0, 94.6, -34.1)
#: A clear pose across the shoulder pan: the straight line to it from LOOK[0] crosses the double's world.
ACROSS = _deg(-22.5, -67.9, -78.6, -150.0, 94.6, -34.1)
#: The owner's LOOK[1]: the exact meshes keep 18.4 mm on forearm|wrist_2, the planner's padded spheres overlap by 2.75 mm,
#: and the nearest pose both clear turns wrist_1 alone by +19.5 degrees (-102.27) or -23.5 (-145.27), which is the band
#: the double holds for it (CPU replica and the real guard, 2026-09-30).
LOOK1 = _deg(-89.08, -42.64, -105.24, -121.77, 88.65, -102.29)
LOOK1_BAND = (-145.27, -102.27)
#: A clear pose across the shoulder pan from LOOK[1], wrist_1 out of its band: the line to it crosses a world at -80..-75.
ACROSS_LOOK1 = _deg(-69.08, -42.64, -105.24, -95.0, 88.65, -102.29)
_PAIR = ("forearm_link", "wrist_2_link")
#: The Hand-E's finger 3.3 mm from the UR10's own base (review of F1, 2026-09-30, measured against the base mesh): the
#: exact guard holds no part for the base, so it accepts this; and a start six degrees of shoulder pan away.
BASE_NEAR = _deg(71.47, -108.72, 163.32, -98.41, -139.73, -162.17)
BASE_START = _deg(65.47, -108.72, 163.32, -98.41, -139.73, -162.17)


class BandPlanner:
    """The planner as the sidecar answers it near the owner's looks: a band on wrist_1, a world across the pan.

    ``band`` is the wrist_1 range, degrees, its padded spheres refuse as ``pairs``; ``world_pan`` the shoulder pan range
    its world refuses, on a straight line judged at a clearance only (a plan's legs clear it, as a plan around it does);
    ``row_changes`` rewrites the band's report rows, to make them say what only the planner judges. Every question is
    recorded in ``calls``, in order.
    """

    def __init__(self, *, band: "tuple[float, float]" = (-140.5, -101.0), world_pan: "tuple[float, float] | None" = None,
                 pairs: "tuple[tuple[str, str], ...]" = (_PAIR,), row_changes: "dict[str, Any] | None" = None,
                 reports: bool = True, band_on: int = 3, depths: "tuple[float, float]" = (1.21, -2.79)) -> None:
        self.band = band
        self.band_on = band_on
        self.depths = depths
        self.world_pan = world_pan
        self.pairs = pairs
        self.row_changes = dict(row_changes or {})
        self.calls: list[tuple[Any, ...]] = []
        self.last_refusal: "StateRefusal | None" = None
        if not reports:
            self.judge_joint_path = None  # type: ignore[assignment]

    # what it finds in one configuration

    def in_band(self, config: "list[float] | tuple[float, ...]") -> bool:
        return self.band[0] < math.degrees(config[self.band_on]) < self.band[1]

    def in_world(self, config: "list[float] | tuple[float, ...]", clearance_mm: float) -> bool:
        return (clearance_mm > 0.0 and self.world_pan is not None
                and self.world_pan[0] < math.degrees(config[0]) < self.world_pan[1])

    def _row(self, index: int, config: "list[float]", clearance_mm: float, named: bool) -> "RefusedSample | None":
        band, world = self.in_band(config), self.in_world(config, clearance_mm)
        if not band and not world:
            return None
        pairs = tuple(PairOverlap(a, b, *self.depths) for a, b in self.pairs) if band else ()
        row = RefusedSample(index=index, bound_ok=True, self_ok=not band, world_ok=not world,
                            pairs=pairs if named else UNSET)
        if band and self.row_changes:
            row = RefusedSample(**{**{f: getattr(row, f) for f in ("index", "bound_ok", "self_ok", "world_ok",
                                                                   "pairs")}, **self.row_changes})
        return row

    # the questions the arm asks

    def check_joint_path(self, samples: Any, *, refresh: bool = True, clearance_mm: float = 0.0) -> JointCheckVerdict:
        configs = [list(map(float, s)) for s in samples]
        self.calls.append(("check", configs, float(clearance_mm)))
        for index, config in enumerate(configs):
            row = self._row(index, config, clearance_mm, True)
            if row is None:
                continue
            kind = StateRefusalKind.SELF_COLLISION if not row.self_ok else StateRefusalKind.WORLD
            refusal = StateRefusal(where=StateWhere.PATH, kind=kind, joints=tuple(config),
                                   **({"link_a": self.pairs[0][0], "link_b": self.pairs[0][1],
                                       "depth_mm": self.depths[0]}
                                      if kind is StateRefusalKind.SELF_COLLISION else {"clearance_mm": clearance_mm}))
            return JointCheckVerdict(valid=False, first_invalid=index, checked=len(configs),
                                     reason=f"the cuRobo check refuses sample {index}", refusal=refusal)
        return JointCheckVerdict(valid=True, first_invalid=None, checked=len(configs), reason="clear")

    def judge_joint_path(self, samples: Any, *, clearance_mm: float = 0.0, name_pairs: bool = True) -> PathJudgement:
        configs = [list(map(float, s)) for s in samples]
        self.calls.append(("judge", configs, float(clearance_mm), bool(name_pairs)))
        rows = tuple(row for index, config in enumerate(configs)
                     if (row := self._row(index, config, clearance_mm, name_pairs)) is not None)
        return PathJudgement(checked=len(configs), clearance_mm=clearance_mm, refused=rows, pairs_named=name_pairs)

    def plan_joint(self, goal: Any, *, start_ur: Any = None, refresh: bool = True, **_: Any) -> "list[list[float]] | None":
        start = list(map(float, start_ur)) if start_ur is not None else list(self.here)
        asked = list(map(float, goal))
        self.calls.append(("plan", start, asked, start_ur is not None))
        for where, config in ((StateWhere.START, start), (StateWhere.GOAL, asked)):
            if self.in_band(config):
                self.last_refusal = StateRefusal(where=where, kind=StateRefusalKind.SELF_COLLISION,
                                                 joints=tuple(config), link_a=self.pairs[0][0],
                                                 link_b=self.pairs[0][1], depth_mm=self.depths[0])
                return None
        self.last_refusal = None
        return line(start, asked, 11)

    def execute(self, traj: Any, pose: Any = None, *, command: MotionCommand = MotionCommand.MOVE_TO,
                target_joints: Any = None, vel: Any = None, acc: Any = None) -> MotionResult:
        self.calls.append(("execute", [list(map(float, w)) for w in traj]))
        return MotionResult.executed(command, target_pose=pose, target_joints=target_joints, message="curobo")

    def named(self, name: str) -> list[tuple[Any, ...]]:
        return [call for call in self.calls if call[0] == name]

    here: "list[float]" = LOOK0


class NoPlanFromTheEscape(BandPlanner):
    """The scene blocks every way from a clear configuration: cuRobo, asked from an explicit start, finds no plan, as
    ``plan_js`` answers a goal nothing reaches (no typed refusal). ``fail`` is how many of those asks fail, all of them
    where ``None``."""

    def __init__(self, *, fail: "int | None" = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fail = fail
        self.explicit: list[list[float]] = []

    def plan_joint(self, goal: Any, *, start_ur: Any = None, refresh: bool = True, **_: Any) -> "list[list[float]] | None":
        if start_ur is not None:
            self.explicit.append(list(map(float, goal)))
            if self.fail is None or len(self.explicit) <= self.fail:
                self.calls.append(("plan", list(map(float, start_ur)), list(map(float, goal)), True))
                self.last_refusal = None
                return None
        return super().plan_joint(goal, start_ur=start_ur, refresh=refresh)


class CornerPlanner(BandPlanner):
    """A world across the pan that meets only a line with wrist_1 above -165 degrees there, and a plan that turns
    wrist_1 first and the pan after: the escape leg and a long wrist_1 chord in one straight line would pass too."""

    def in_world(self, config: "list[float] | tuple[float, ...]", clearance_mm: float) -> bool:
        return (clearance_mm > 0.0 and -35.0 < math.degrees(config[0]) < -30.0
                and math.degrees(config[3]) > -165.0)

    def plan_joint(self, goal: Any, *, start_ur: Any = None, refresh: bool = True, **_: Any) -> "list[list[float]] | None":
        start = list(map(float, start_ur)) if start_ur is not None else list(self.here)
        asked = list(map(float, goal))
        if self.in_band(start) or self.in_band(asked):
            return super().plan_joint(goal, start_ur=start_ur, refresh=refresh)
        self.calls.append(("plan", start, asked, start_ur is not None))
        self.last_refusal = None
        corner = list(start)
        corner[3] = asked[3]
        return line(start, corner, 11)[:-1] + line(corner, asked, 11)


class ShiftedReport(BandPlanner):
    """A report that leaves out the first refused sample: it opens on another sample than the verdict named."""

    def judge_joint_path(self, samples: Any, *, clearance_mm: float = 0.0, name_pairs: bool = True) -> PathJudgement:
        judged = super().judge_joint_path(samples, clearance_mm=clearance_mm, name_pairs=name_pairs)
        return PathJudgement(checked=judged.checked, clearance_mm=judged.clearance_mm, refused=judged.refused[1:],
                             pairs_named=judged.pairs_named)


class MiscountedReport(BandPlanner):
    """A report that says it judged one sample more than were sent."""

    def judge_joint_path(self, samples: Any, *, clearance_mm: float = 0.0, name_pairs: bool = True) -> PathJudgement:
        judged = super().judge_joint_path(samples, clearance_mm=clearance_mm, name_pairs=name_pairs)
        return PathJudgement(checked=judged.checked + 1, clearance_mm=judged.clearance_mm, refused=judged.refused,
                             pairs_named=judged.pairs_named)


class WorldTypedVerdict(BandPlanner):
    """A verdict the planner typed as its world, while its report finds only the band: never the guard's to decide."""

    def check_joint_path(self, samples: Any, *, refresh: bool = True, clearance_mm: float = 0.0) -> JointCheckVerdict:
        verdict = super().check_joint_path(samples, refresh=refresh, clearance_mm=clearance_mm)
        if verdict.valid or verdict.refusal is None:
            return verdict
        refusal = StateRefusal(where=StateWhere.PATH, kind=StateRefusalKind.WORLD, joints=verdict.refusal.joints,
                               depth_mm=1.0)
        return JointCheckVerdict(valid=False, first_invalid=verdict.first_invalid, checked=verdict.checked,
                                 reason=verdict.reason, refusal=refusal)


def _arm(planner: BandPlanner, *, here: "list[float]") -> Any:
    if mesh_backend_status("ur10", mesh_name="robotiq_hande") != "ok":
        raise unittest.SkipTest("no exact mesh backend on this box")
    arm: Any = owner_like_arm()
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = list(here)
    arm._conn.moveJ.return_value = True
    planner.here = list(here)
    arm._curobo_ur = planner
    return arm


def _joint_move(arm: Any, target: "list[float]", *, on_the_line: bool = False) -> MotionResult:
    with arm.without_camera_world(_DECLINED):
        verb = arm.move_to_joints_on_the_line if on_the_line else arm.move_to_joints
        return verb(JointPositions(tuple(target)))


class TheExactGuardDecidesTheBandTests(unittest.TestCase):
    def test_a_move_to_the_look_the_arm_stands_at_runs(self) -> None:
        """The log that started this: standing at LOOK[0], the move to LOOK[0] was refused at sample 0 of 2."""
        planner = BandPlanner()
        arm = _arm(planner, here=LOOK0)
        result = _joint_move(arm, LOOK0)
        self.assertTrue(result.ok, result.message)
        arm._conn.moveJ.assert_called_once()
        self.assertFalse(planner.named("plan"), "a line the exact guard decides is not planned around")

    def test_a_line_between_two_band_poses_runs_and_the_report_is_on_the_samples_checked(self) -> None:
        planner = BandPlanner()
        arm = _arm(planner, here=LOOK0)
        result = _joint_move(arm, LOOK0_DEEPER)
        self.assertTrue(result.ok, result.message)
        (check,) = [c for c in planner.named("check") if len(c[1]) > 1]
        (judge,) = [c for c in planner.named("judge") if c[1] == check[1]]
        self.assertEqual(judge[2], check[2], "the report is asked at the clearance the line was judged at")
        self.assertTrue(judge[3], "and with the pairs named")
        self.assertTrue(all(planner.in_band(config) for config in check[1]))

    def test_the_move_back_to_a_band_look_on_the_line_runs(self) -> None:
        planner = BandPlanner()
        arm = _arm(planner, here=LOOK0_DEEPER)
        result = _joint_move(arm, LOOK0, on_the_line=True)
        self.assertTrue(result.ok, result.message)
        self.assertFalse(planner.named("plan"))

    def test_a_planner_that_reports_nothing_leaves_the_refusal_standing(self) -> None:
        """⭐ THE CONTROL: without the report the driver cannot know every pair, so nothing is left to the guard."""
        planner = BandPlanner(reports=False)
        arm = _arm(planner, here=LOOK0)
        result = _joint_move(arm, LOOK0_DEEPER, on_the_line=True)
        self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)
        arm._conn.moveJ.assert_not_called()

    def test_the_world_the_bounds_the_carried_part_and_a_pair_the_guard_skips_stay_the_planners(self) -> None:
        cases = {
            "bounds": {"row_changes": {"bound_ok": False}},
            "world": {"row_changes": {"world_ok": False}},
            "the carried part": {"pairs": (_PAIR, ("attached_object", "forearm_link"))},
            "a pair the guard skips": {"pairs": (("hand", "wrist_2_link"),)},
            "an unnamed self collision": {"row_changes": {"pairs": ()}},
        }
        for what, kwargs in cases.items():
            with self.subTest(what=what):
                planner = BandPlanner(**kwargs)
                arm = _arm(planner, here=LOOK0)
                result = _joint_move(arm, LOOK0_DEEPER, on_the_line=True)
                self.assertFalse(result.ok, what)
                arm._conn.moveJ.assert_not_called()

    def test_the_exact_guard_is_asked_again_whatever_the_caller_judged(self) -> None:
        """⭐ THE INVARIANT: a refused sample is left to the exact guard only where that guard accepts THAT sample, judged
        at the admission itself. A folded wrist the planner's report names forearm|wrist_2 on stays refused."""
        planner = BandPlanner(band=(-179.0, 179.0))
        arm = _arm(planner, here=LOOK0)
        folded = _deg(*FOLDED_DEG)
        for configs in ([folded], [LOOK0, folded]):
            verdict = planner.check_joint_path(configs, refresh=False)
            why = arm._exact_guard_decides(planner, configs, verdict, clearance_mm=0.0,
                                           command=MotionCommand.MOVE_JOINTS)
            assert why is not None
            self.assertIn("exact guard refuses", why.reason)
            assert why.row is not None
            self.assertEqual(why.row.index, len(configs) - 1, "the refusal stands on the folded sample, named")
            self.assertIs(why.status, MotionStatus.SELF_COLLISION_REJECTED)
        verdict = planner.check_joint_path([LOOK0], refresh=False)
        self.assertIsNone(arm._exact_guard_decides(planner, [LOOK0], verdict, clearance_mm=0.0,
                                                   command=MotionCommand.MOVE_JOINTS))

    def test_the_exact_gate_runs_first_and_its_refusal_asks_no_planner(self) -> None:
        planner = BandPlanner()
        arm = _arm(planner, here=LOOK0)
        gate = arm._preflight.gate_planned_path
        order: list[str] = []

        def judged(waypoints: Any, **kwargs: Any) -> Any:
            order.append("gate")
            return gate(waypoints, **kwargs)

        arm._preflight.gate_planned_path = judged
        self.assertTrue(_joint_move(arm, LOOK0_DEEPER).ok)
        self.assertEqual(order, ["gate"])
        self.assertTrue(planner.calls and planner.calls[0][0] == "check")

        refused = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_JOINTS,
                                      message="[safety:self_collision/self_collision] the gate")
        arm._preflight.gate_planned_path = lambda waypoints, **kwargs: refused
        planner.calls.clear()
        self.assertFalse(_joint_move(arm, LOOK0_DEEPER, on_the_line=True).ok)
        self.assertEqual(planner.calls, [], "a line the exact gate refused is never asked of the planner")

    def test_a_cartesian_goal_in_the_band_is_the_exact_guards_at_the_screen(self) -> None:
        planner = BandPlanner()
        arm = _arm(planner, here=LOOK0)
        with arm.without_camera_world(_DECLINED), self.assertLogs("URRobotArm", level="INFO") as logs:
            result = arm.move(pose_where_it_ends(arm, LOOK0_DEEPER))
        self.assertTrue(result.ok, result.message)
        screened = [c for c in planner.named("check") if len(c[1]) == 1]
        self.assertTrue(screened and planner.in_band(screened[0][1][0]), "the goal was screened in the band")
        self.assertFalse(planner.named("plan"))
        # The goal on the arm's own branch is the one reached, on the line: a screen that refused it would send the arm
        # to another branch's goal, half a turn of the shoulder pan away.
        (executed,) = planner.named("execute")
        self.assertEqual(len(executed[1]), 2)
        self.assertLess(max(abs(a - b) for a, b in zip(executed[1][-1], LOOK0_DEEPER)), 1e-6)
        self.assertFalse([line for line in logs.output if "changes the arm's branch" in line])

    def test_a_tool_line_out_of_the_band_runs(self) -> None:
        """moveL: every sample is solved and judged by the exact guard, then the planner; the band is the guard's."""
        planner = BandPlanner()
        arm = _arm(planner, here=LOOK0)
        arm.ik = lambda pose, *, seed=None: JointPositions(tuple(LOOK0))
        arm._motion = MagicMock()
        arm._motion.move_to.return_value = True
        flange = np.asarray(ur_link_transforms_mm("ur10", np.asarray(LOOK0))[-1])
        arm._motion.get_current_pose.return_value = pose_to_urpose(Pose.from_matrix(flange, frame=Frame.BASE))
        start = arm.get_tcp_pose()
        with arm.without_camera_world(_DECLINED):
            result = arm.move(start, linear=True)
        self.assertTrue(result.ok, result.message)
        self.assertTrue(planner.named("judge"), "the tool line's band samples were reported, then decided")
        arm._motion.move_to.assert_called_once()
        (check,) = planner.named("check")
        (judge,) = planner.named("judge")
        self.assertEqual(judge[1], check[1], "the report is on the very samples the line was judged on")
        self.assertEqual(judge[2], check[2], "and at the clearance they were judged at")

    def test_a_pair_padded_past_the_cells_margin_stays_the_planners(self) -> None:
        """⭐ Only the cell's planner_margin_mm is the guard's to decide past: the same band pair carrying 12 mm of
        padding (a stock cushion on top of the margin's 4) is refused as it always was, and nothing is sent."""
        planner = BandPlanner(depths=(9.21, -2.79))
        arm = _arm(planner, here=LOOK0)
        result = _joint_move(arm, LOOK0_DEEPER, on_the_line=True)
        self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)
        self.assertIn("more than the 4 mm margin", result.message or "")
        arm._conn.moveJ.assert_not_called()

    def test_a_report_that_is_not_the_verdicts_leaves_the_refusal_standing(self) -> None:
        """⭐ FAIL CLOSED: a report that opens on another sample than the verdict, that counts other samples than were
        sent, or a verdict the planner typed as something else than a self collision, is no report on these samples."""
        for double in (ShiftedReport, MiscountedReport, WorldTypedVerdict):
            with self.subTest(double=double.__name__):
                planner = double()
                arm = _arm(planner, here=LOOK0)
                result = _joint_move(arm, LOOK0_DEEPER, on_the_line=True)
                self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)
                arm._conn.moveJ.assert_not_called()

    def test_the_refusal_says_the_sample_and_the_term_it_stands_on(self) -> None:
        """Where the band is the guard's and a later sample reaches into the planner's world, the refusal names that
        sample and the world, not the band pair the guard already decided; a bound it stands on reads as a bound."""
        planner = BandPlanner(world_pan=(-35.0, -30.0))
        arm = _arm(planner, here=LOOK0)
        result = _joint_move(arm, ACROSS, on_the_line=True)
        self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)
        (check,) = [c for c in planner.named("check") if len(c[1]) > 2]
        stood = next(i for i, config in enumerate(check[1]) if planner.in_world(config, check[2]))
        self.assertIn(f"at sample {stood} of {len(check[1])}", result.message or "")
        self.assertIn("world", result.message or "")
        self.assertNotIn("forearm_link and wrist_2_link", result.message or "")
        self.assertIn("nothing is planned around it", result.message or "")

        bounded = _arm(BandPlanner(row_changes={"bound_ok": False}), here=LOOK0)
        result = _joint_move(bounded, LOOK0_DEEPER, on_the_line=True)
        self.assertIs(result.status, MotionStatus.JOINT_LIMIT_REJECTED)
        self.assertIn("at sample 0 of", result.message or "")
        self.assertIn("bounds", result.message or "")
        bounded._conn.moveJ.assert_not_called()


def _base_planner() -> BandPlanner:
    """The planner at the UR10's base: the shoulder_link's cushion, cuRobo's own 70 mm and the margin's 2, meets the
    hand past a shoulder pan of 68 degrees, as its spheres did at BASE_NEAR (87.3 mm padded, 13.3 mm without)."""
    return BandPlanner(band=(68.0, 90.0), band_on=0, pairs=(("hand", "shoulder_link"),), depths=(87.3, 13.3))


class TheRobotsOwnBaseStaysThePlannersTests(unittest.TestCase):
    """⭐ THE BASE (review of F1, 2026-09-30): the exact guard holds no part for the UR's base, and the planner's
    shoulder_link is the only model of it. A pair with the shoulder_link is never the guard's to decide, so a move that
    ends with the hand at the base is refused before anything is sent, as it was before the owner's decision."""

    def test_the_exact_guard_accepts_the_hand_at_the_base_because_it_holds_no_base(self) -> None:
        """The premise, on the real guard: its verdict here says nothing about the base."""
        arm = _arm(_base_planner(), here=BASE_START)
        self.assertIsNone(arm._preflight.gate_joint_target(JointPositions(tuple(BASE_NEAR)), arm=arm))

    def test_a_joint_move_to_the_base_is_refused_and_nothing_is_sent(self) -> None:
        planner = _base_planner()
        arm = _arm(planner, here=BASE_START)
        result = _joint_move(arm, BASE_NEAR)
        self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)
        arm._conn.moveJ.assert_not_called()
        self.assertFalse(planner.named("execute"))
        self.assertIn("hand and shoulder_link", result.message or "")

    def test_a_tool_line_at_the_base_is_refused_and_no_movel_is_sent(self) -> None:
        planner = _base_planner()
        arm = _arm(planner, here=BASE_NEAR)
        arm.ik = lambda pose, *, seed=None: JointPositions(tuple(BASE_NEAR))
        arm._motion = MagicMock()
        arm._motion.move_to.return_value = True
        flange = np.asarray(ur_link_transforms_mm("ur10", np.asarray(BASE_NEAR))[-1])
        arm._motion.get_current_pose.return_value = pose_to_urpose(Pose.from_matrix(flange, frame=Frame.BASE))
        start = arm.get_tcp_pose()
        with arm.without_camera_world(_DECLINED):
            result = arm.move(start, linear=True)
        self.assertFalse(result.ok)
        self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)
        arm._motion.move_to.assert_not_called()


class APlannedMoveLeavesTheBandByAShortLegTests(unittest.TestCase):
    def test_a_planned_move_out_of_the_band_takes_a_short_straight_leg_first(self) -> None:
        planner = BandPlanner(world_pan=(-35.0, -30.0))
        arm = _arm(planner, here=LOOK0)
        result = _joint_move(arm, ACROSS)
        self.assertTrue(result.ok, result.message)
        first, second = planner.named("plan")
        self.assertFalse(first[3], "cuRobo was asked from where the arm stands, and refused the start")
        self.assertTrue(second[3], "then from an explicit start: the escape")
        escape = second[1]
        self.assertFalse(planner.in_band(escape))
        turned = [abs(math.degrees(a - b)) for a, b in zip(escape, LOOK0)]
        self.assertLessEqual(max(turned), _CAP_DEG + 1e-9, "the escape leg turns no joint more than 20 degrees")
        self.assertEqual([i for i, t in enumerate(turned) if t > 1e-9], [3], "wrist_1 alone gets out of the band here")
        (executed,) = planner.named("execute")
        self.assertEqual(executed[1][0], LOOK0, "the route opens where the arm stands")
        self.assertEqual(executed[1][1], escape, "then the escape")
        self.assertEqual(executed[1][-1], ACROSS)
        legs = [c for c in planner.named("check") if c[1][0] == LOOK0 and c[1][-1] == escape and c[2] > 0.0]
        self.assertTrue(legs, "the escape leg was judged as a straight line, at the line clearance")

    def test_a_planned_move_into_the_band_arrives_by_a_short_straight_leg(self) -> None:
        planner = BandPlanner(world_pan=(-35.0, -30.0))
        start = _deg(-22.5, -67.9, -78.6, -150.0, 94.6, -34.1)
        arm = _arm(planner, here=start)
        result = _joint_move(arm, LOOK0)
        self.assertTrue(result.ok, result.message)
        (executed,) = planner.named("execute")
        approach = executed[1][-2]
        self.assertEqual(executed[1][-1], LOOK0, "the route ends on the look")
        self.assertFalse(planner.in_band(approach))
        self.assertLessEqual(max(abs(math.degrees(a - b)) for a, b in zip(approach, LOOK0)), _CAP_DEG + 1e-9)
        (_, plan_start, plan_goal, explicit) = planner.named("plan")[-1]
        self.assertEqual(plan_goal, approach, "cuRobo plans to the approach, never into the band")

    def test_an_escape_leg_the_exact_gate_refuses_sends_nothing_and_says_what_was_found(self) -> None:
        planner = BandPlanner(world_pan=(-35.0, -30.0))
        arm = _arm(planner, here=LOOK0)
        gate = arm._preflight.gate_planned_path
        refused = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_JOINTS,
                                      message="[safety:self_collision/self_collision] refused by the double")

        def no_leg_out(waypoints: Any, **kwargs: Any) -> Any:
            if len(waypoints) == 2 and list(waypoints[0]) == LOOK0 and list(waypoints[1]) != ACROSS:
                return refused
            return gate(waypoints, **kwargs)

        arm._preflight.gate_planned_path = no_leg_out
        legs: list[Any] = []
        counted = no_leg_out

        def counting(waypoints: Any, **kwargs: Any) -> Any:
            if len(waypoints) == 2 and list(waypoints[0]) == LOOK0 and list(waypoints[1]) != ACROSS:
                legs.append(list(waypoints[1]))
            return counted(waypoints, **kwargs)

        arm._preflight.gate_planned_path = counting
        result = _joint_move(arm, ACROSS)
        self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)
        self.assertFalse(planner.named("execute"))
        arm._conn.moveJ.assert_not_called()
        self.assertEqual(len(planner.named("plan")), 1, "no plan without a judged escape leg")
        self.assertEqual(len(legs), 3, "three escape legs at most are judged, the nearest first")
        for said in ("will not start from", "cushion band", "forearm_link and wrist_2_link", "19.0 mm", "20 deg"):
            self.assertIn(said, result.message or "")

    def test_no_plan_from_the_escape_is_no_plan_and_not_a_start_the_arm_cannot_leave(self) -> None:
        """The escape leg exists and both judged it; cuRobo finds no plan from it to the target. The arm can leave where
        it stands, so the move is refused as a line no plan goes around, never as a start the planner will not leave."""
        planner = NoPlanFromTheEscape(world_pan=(-35.0, -30.0))
        arm = _arm(planner, here=LOOK0)
        result = _joint_move(arm, ACROSS)
        self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)
        arm._conn.moveJ.assert_not_called()
        self.assertEqual([call[3] for call in planner.named("plan")], [False, True])
        message = result.message or ""
        self.assertNotIn("will not start from", message)
        for said in ("no plan goes around it", "escape leg", "cuRobo found no plan"):
            self.assertIn(said, message)

    def test_a_goal_the_planners_world_holds_is_the_goals_failure_not_the_starts(self) -> None:
        """The start is left by a judged escape; the goal itself reaches into the planner's world, which only the
        planner judges. That is this goal's failure: no plan goes around the line to it, and nothing says the arm
        cannot leave where it stands."""

        class GoalInItsWorld(BandPlanner):
            def in_world(self, config: "list[float] | tuple[float, ...]", clearance_mm: float) -> bool:
                return max(abs(a - b) for a, b in zip(config, ACROSS)) < 1e-9

        planner = GoalInItsWorld()
        arm = _arm(planner, here=LOOK0)
        result = _joint_move(arm, ACROSS)
        self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)
        arm._conn.moveJ.assert_not_called()
        self.assertFalse(planner.named("execute"))
        message = result.message or ""
        self.assertNotIn("will not start from", message)
        for said in ("no plan goes around it", "escape leg", "the planner refuses the goal", "world"):
            self.assertIn(said, message)

    def test_a_cartesian_move_out_of_the_band_tries_the_next_goal_from_the_escape(self) -> None:
        """A goal the escape plans to and cuRobo finds no way to is that goal's failure: the next goal is planned from the
        escape, as any goal after one that did not plan."""
        planner = NoPlanFromTheEscape(fail=1, world_pan=(-35.0, -30.0))
        arm = _arm(planner, here=LOOK0)
        with arm.without_camera_world(_DECLINED):
            result = arm.move(pose_where_it_ends(arm, ACROSS))
        self.assertTrue(result.ok, result.message)
        self.assertEqual(len(planner.explicit), 2, "the second goal was planned from the escape")
        self.assertNotEqual(planner.explicit[0], planner.explicit[1])
        (executed,) = planner.named("execute")
        self.assertEqual(executed[1][0], LOOK0)
        self.assertFalse(planner.in_band(executed[1][1]), "the first leg is the escape")
        self.assertLessEqual(max(abs(math.degrees(a - b)) for a, b in zip(executed[1][1], LOOK0)), _CAP_DEG + 1e-9)
        self.assertEqual(executed[1][-1], planner.explicit[1])

    def test_a_cartesian_move_no_goal_of_which_plans_from_the_escape_is_no_plan(self) -> None:
        """Every goal planned from the escape and none reached: the no-plan refusal a pick looks again at, not a start
        the arm cannot leave."""
        from src.robot.core.motion_result import NO_PLAN_FAIL_SAFE_MESSAGE

        planner = NoPlanFromTheEscape(world_pan=(-35.0, -30.0))
        arm = _arm(planner, here=LOOK0)
        with arm.without_camera_world(_DECLINED):
            result = arm.move(pose_where_it_ends(arm, ACROSS))
        self.assertIs(result.status, MotionStatus.TIMEOUT)
        self.assertEqual(result.message, NO_PLAN_FAIL_SAFE_MESSAGE)
        self.assertEqual(len(planner.explicit), 3, "as many goals as a move plans to, each from the escape")
        self.assertFalse(planner.named("execute"))

    def test_the_escape_leg_is_run_as_the_leg_it_was_judged(self) -> None:
        """⭐ THE CAP HOLDS ON WHAT RUNS: the shortening of the plan keeps the escape leg as its own moveJ, rather than
        merging it into one long straight line out of the band that no escape search chose."""
        planner = CornerPlanner()
        arm = _arm(planner, here=LOOK0)
        goal = _deg(-22.5, -67.9, -78.6, -175.0, 94.6, -34.1)
        with self.assertLogs("URRobotArm", level="INFO") as logs:
            result = _joint_move(arm, goal)
        self.assertTrue(result.ok, result.message)
        (executed,) = planner.named("execute")
        route = executed[1]
        self.assertEqual(route[0], LOOK0)
        self.assertLessEqual(max(abs(math.degrees(a - b)) for a, b in zip(route[1], LOOK0)), _CAP_DEG + 1e-9,
                             "the first moveJ out of the band is the escape leg")
        self.assertFalse(planner.in_band(route[1]))
        self.assertLess(max(abs(a - b) for a, b in zip(route[-1], goal)), 1e-9)
        self.assertLess(len(route), 12, "the plan behind the escape was still shortened")
        routed = [line for line in logs.output if "escape leg" in line and "executed" in line]
        self.assertTrue(routed, "the route's log line says it took the escape leg")

    def test_a_goal_no_approach_reaches_says_so_as_a_goal(self) -> None:
        """A band goal no 20-degree approach reaches: no plan REACHES it, which a straight line may, and the arm is not
        standing there, so nothing is said about moving it by hand."""
        planner = BandPlanner(band=(-170.0, -95.0), world_pan=(-35.0, -30.0))
        arm = _arm(planner, here=_deg(-22.5, -67.9, -78.6, -175.0, 94.6, -34.1))
        result = _joint_move(arm, LOOK0)
        self.assertFalse(result.ok)
        arm._conn.moveJ.assert_not_called()
        message = result.message or ""
        self.assertIn("the goal is in the planner's cushion band", message)
        self.assertIn("so no plan reaches it", message)
        self.assertNotIn("move the arm by hand", message)
        self.assertNotIn("will not start from", message)

    def test_no_escape_leg_is_taken_further_than_twenty_degrees(self) -> None:
        """⭐ THE CAP: a band too wide to leave within 20 degrees per joint (wrist_1 34 and 41 degrees from its edges) is
        left by hand, and says so."""
        planner = BandPlanner(band=(-170.0, -95.0), world_pan=(-35.0, -30.0))
        arm = _arm(planner, here=LOOK0)
        result = _joint_move(arm, ACROSS)
        self.assertFalse(result.ok)
        self.assertEqual(len(planner.named("plan")), 1)
        self.assertFalse(planner.named("execute"))
        for said in ("cushion band", "19.0 mm", "within 20 deg", "move the arm by hand"):
            self.assertIn(said, result.message or "")

    def test_a_band_as_deep_as_look1_is_left_by_a_leg_of_at_most_twenty_degrees(self) -> None:
        """⭐ THE OWNER'S DECISION (2026-09-30, after the fix round): LOOK[1]'s nearest pose both clear is wrist_1 +19.5
        degrees away, which the 10-degree cap refused to plan out of. At 20 a planned move out of it takes that leg,
        judged by both authorities as a straight line, and cuRobo plans from its end."""
        planner = BandPlanner(band=LOOK1_BAND, world_pan=(-80.0, -75.0), depths=(2.75, -1.25))
        arm = _arm(planner, here=LOOK1)
        result = _joint_move(arm, ACROSS_LOOK1)
        self.assertTrue(result.ok, result.message)
        first, second = planner.named("plan")
        self.assertFalse(first[3], "cuRobo was asked from where the arm stands, and refused the start")
        self.assertTrue(second[3], "then from an explicit start: the escape")
        escape = second[1]
        self.assertFalse(planner.in_band(escape))
        turned = [math.degrees(a - b) for a, b in zip(escape, LOOK1)]
        self.assertEqual([i for i, t in enumerate(turned) if abs(t) > 1e-9], [3], "wrist_1 alone leaves this band")
        self.assertGreater(turned[3], 19.5 - 1e-9, "no nearer pose on wrist_1 is out of the band")
        self.assertLessEqual(turned[3], _CAP_DEG + 1e-9)
        (executed,) = planner.named("execute")
        self.assertEqual(executed[1][0], LOOK1, "the route opens where the arm stands")
        self.assertEqual(executed[1][1], escape, "then the escape, run as the leg it was judged as")
        self.assertEqual(executed[1][-1], ACROSS_LOOK1)
        legs = [c for c in planner.named("check") if c[1][0] == LOOK1 and c[1][-1] == escape and c[2] > 0.0]
        self.assertTrue(legs, "the escape leg was judged as a straight line, at the line clearance")

    def test_a_line_only_move_never_escapes(self) -> None:
        """⭐ The owner's rule (2026-09-29): the generated view and its move back run on the line or not at all."""
        planner = BandPlanner(world_pan=(-35.0, -30.0))
        arm = _arm(planner, here=LOOK0)
        result = _joint_move(arm, ACROSS, on_the_line=True)
        self.assertFalse(result.ok)
        self.assertIn("nothing is planned around it", result.message or "")
        self.assertEqual(planner.named("plan"), [])
        self.assertFalse([c for c in planner.named("judge") if not c[3]], "no escape was ever looked for")


class APoseIsScreenedBeforeAPickMeetsItTests(unittest.TestCase):
    """The owner, 2026-09-30: where a pose is taught, where a campaign starts and at the desk, both authorities are
    asked about it and the verdict is said: clear, in the band (it runs; a planned move takes a short escape), or
    refused (an ERROR), with a pose nearby both clear. Nothing moves for a screen."""

    def test_look0_is_in_the_band_and_a_pose_nearby_clears_both(self) -> None:
        from src.robot.safety.planning.band import PoseVerdict

        planner = BandPlanner()
        arm = _arm(planner, here=CLEAR)
        screen = arm.screen_configuration(JointPositions(tuple(LOOK0)))
        self.assertIs(screen.verdict, PoseVerdict.BAND)
        self.assertFalse(screen.is_error)
        for said in ("forearm_link and wrist_2_link", "1.2 mm", "2.8 mm apart", "19.0 mm", "10 mm",
                     "Straight lines run", "at most 20 deg per joint"):
            self.assertIn(said, screen.detail)
        assert screen.nearby is not None
        self.assertFalse(planner.in_band(list(screen.nearby)))
        self.assertLessEqual(max(abs(math.degrees(a - b)) for a, b in zip(screen.nearby, LOOK0)), _CAP_DEG + 1e-9)
        # The grid over two joints steps 4/3 degree to reach 20 in one request: wrist_1 -140.2 is still in the band,
        # -141.5 is the nearest pose out of it.
        self.assertIn("Nearby, both clear: (-42.5, -67.9, -78.6, -141.5, 94.6, -34.1) deg", screen.line("LOOK_0"))
        arm._conn.moveJ.assert_not_called()
        self.assertFalse(planner.named("plan"))

    def test_a_band_pose_no_short_leg_leaves_says_a_planned_move_is_refused(self) -> None:
        """A pose deeper in the band than 20 degrees per joint reach. Straight lines run; a planned move out of it is
        refused, and the screen says so rather than promising an escape leg."""
        from src.robot.safety.planning.band import PoseVerdict

        screen = _arm(BandPlanner(band=(-170.0, -95.0)), here=CLEAR).screen_configuration(JointPositions(tuple(LOOK0)))
        self.assertIs(screen.verdict, PoseVerdict.BAND)
        self.assertIsNone(screen.nearby)
        self.assertIn("no pose within 20 deg per joint of it clears both", screen.detail)
        self.assertNotIn("takes a straight leg", screen.detail)

    def test_look1_is_in_the_band_and_a_pose_within_twenty_degrees_clears_both(self) -> None:
        """LOOK[1] on the cell (the owner, 2026-09-30): it runs, and now a planned move out of it or into it takes a leg of
        wrist_1 +20 degrees, the nearest pose the widened grid holds out of its band (+19.5 is the edge)."""
        from src.robot.safety.planning.band import PoseVerdict

        planner = BandPlanner(band=LOOK1_BAND, depths=(2.75, -1.25))
        screen = _arm(planner, here=CLEAR).screen_configuration(JointPositions(tuple(LOOK1)))
        self.assertIs(screen.verdict, PoseVerdict.BAND)
        self.assertFalse(screen.is_error)
        for said in ("2.8 mm", "1.2 mm apart", "18.4 mm", "takes a straight leg of at most 20 deg per joint"):
            self.assertIn(said, screen.detail)
        assert screen.nearby is not None
        self.assertFalse(planner.in_band(list(screen.nearby)))
        self.assertIn("Nearby, both clear: (-89.1, -42.6, -105.2, -101.8, 88.7, -102.3) deg", screen.line("LOOK_1"))
        self.assertFalse([c for c in planner.calls if c[0] in ("plan", "execute")], "a screen moves nothing")

    def test_a_clear_pose_is_clear(self) -> None:
        from src.robot.safety.planning.band import PoseVerdict

        screen = _arm(BandPlanner(), here=LOOK0).screen_configuration(JointPositions(tuple(CLEAR)))
        self.assertIs(screen.verdict, PoseVerdict.CLEAR)
        self.assertIsNone(screen.nearby)

    def test_a_pose_the_exact_guard_refuses_is_an_error(self) -> None:
        """The folded wrist: the meshes touch, so no move goes there, and the pose to teach instead is one the exact
        guard really accepts within 20 degrees per joint of it (10 degrees held none; 20 holds wrist_1 and wrist_2
        each turned 20 degrees on the double, which names no pair there, so every joint was free to turn)."""
        from src.robot.core import MotionCommand as _Command
        from src.robot.safety.planning.band import PoseVerdict

        planner = BandPlanner()
        arm = _arm(planner, here=LOOK0)
        folded = _deg(*FOLDED_DEG)
        screen = arm.screen_configuration(JointPositions(tuple(folded)))
        self.assertIs(screen.verdict, PoseVerdict.GUARD_REFUSED)
        self.assertTrue(screen.line("LOOK_9").startswith("ERROR LOOK_9: refused by the exact guard"))
        self.assertIn("mesh distance", screen.detail)
        self.assertTrue([c for c in planner.named("judge") if not c[3]], "a pose nearby both clear was looked for")
        assert screen.nearby is not None
        self.assertLessEqual(max(abs(math.degrees(a - b)) for a, b in zip(screen.nearby, folded)), _CAP_DEG + 1e-9)
        self.assertTrue(arm._exact_guard_accepts(list(screen.nearby), _Command.MOVE_JOINTS),
                        "the pose offered instead is one the exact guard accepts")
        self.assertFalse(arm._exact_guard_accepts(folded, _Command.MOVE_JOINTS))
        self.assertIn("Nearby, both clear: ", screen.line("LOOK_9"))
        arm._conn.moveJ.assert_not_called()

    def test_a_pose_only_the_planner_refuses_for_its_world_is_an_error(self) -> None:
        from src.robot.safety.planning.band import PoseVerdict

        screen = _arm(BandPlanner(row_changes={"world_ok": False}), here=CLEAR).screen_configuration(
            JointPositions(tuple(LOOK0)))
        self.assertIs(screen.verdict, PoseVerdict.PLANNER_REFUSED)
        self.assertTrue(screen.is_error)
        self.assertIn("world", screen.detail)

    def test_a_planner_that_cannot_be_asked_leaves_the_exact_guards_verdict(self) -> None:
        from src.robot.safety.planning import CuroboUnavailableError
        from src.robot.safety.planning.band import PoseVerdict

        planner = BandPlanner()

        def unavailable(*_: Any, **__: Any) -> Any:
            raise CuroboUnavailableError("no cuRobo env on this desk")

        planner.judge_joint_path = unavailable  # type: ignore[method-assign]
        screen = _arm(planner, here=LOOK0).screen_configuration(JointPositions(tuple(LOOK0)))
        self.assertIs(screen.verdict, PoseVerdict.UNSCREENED)
        self.assertTrue(screen.planner_unavailable)
        self.assertIn("the exact guard accepts it", screen.detail)
        self.assertIn("no cuRobo env on this desk", screen.detail)
        asked_not = _arm(BandPlanner(), here=LOOK0).screen_configuration(JointPositions(tuple(LOOK0)),
                                                                        ask_planner=False)
        self.assertIs(asked_not.verdict, PoseVerdict.UNSCREENED)
        self.assertIn("not asked", asked_not.detail)


def _canned_screen(verdicts: "list[Any]") -> Any:
    """A ``screen_configuration`` answering from ``verdicts`` in turn and recording what it was asked."""
    asked: list[tuple[list[float], bool]] = []

    def screen(self: Any, joints: Any, *, ask_planner: bool = True) -> Any:
        asked.append(([float(v) for v in joints.tolist()], ask_planner))
        return verdicts[min(len(asked) - 1, len(verdicts) - 1)]

    screen.asked = asked  # type: ignore[attr-defined]
    return screen


class TheScreenIsSaidWherePosesAreMetTests(unittest.TestCase):
    def test_teaching_says_each_verdict_once_the_pose_is_held(self) -> None:
        from src.robot.execution.hand_guiding import HandGuide
        from src.robot.execution.teach import teach_poses
        from src.robot.safety.planning.band import PoseScreen, PoseVerdict
        from tests.test_a_pose_is_taught_by_hand import (
            LOOK_1_DEG,
            LOOK_2_DEG,
            TREE,
            _answers,
            _Clock,
            _Console,
            _GuidedArm,
            _stand,
        )

        verdicts = [PoseScreen(PoseVerdict.BAND, "the planner's spheres overlap.", nearby=tuple(CLEAR),
                               planner_unavailable=False),
                    PoseScreen(PoseVerdict.GUARD_REFUSED, "forearm|wrist_2: mesh distance 4.1 mm < 10.000 mm.")]
        screen = _canned_screen(verdicts)

        class _ScreeningArm(_GuidedArm):
            screen_configuration = screen

        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        clock = _Clock()
        arm = _ScreeningArm([_stand(LOOK_1_DEG), _stand(LOOK_2_DEG)], clock)
        # The arm carries the housing of the one wrist camera the tree hangs on it, as Robot.from_tree hands it.
        arm.safety_preflight.wrist_bodies = lambda _arm=None: (SimpleNamespace(rig_id="wrist"),)
        console = _Console(_answers("", "", "", "", "", "q"), arm.log)
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        teach_poses(arm, tree=TREE, store=folder / "poses.json", out=lambda line: None,
                    guide=HandGuide(console, None, clock=clock, sleep=clock.sleep))
        said = console.said
        band = next(i for i, line in enumerate(said) if line.startswith("LOOK_1: in the planner's cushion band"))
        error = next(i for i, line in enumerate(said) if line.startswith("ERROR LOOK_2: refused by the exact guard"))
        self.assertLess(band, error)
        self.assertLess(next(i for i, line in enumerate(said) if line.startswith("LOOK_1 taught")), band)
        self.assertIn("Nearby, both clear", said[band])
        self.assertEqual(len(screen.asked), 2)
        self.assertEqual([round(math.degrees(v), 4) for v in screen.asked[0][0]], [round(v, 4) for v in LOOK_1_DEG])
        self.assertTrue(any("screened once it is taught" in line for line in said))
        self.assertNotIn("move", " ".join(line for line in arm.log if not line.startswith("say")))

    def test_teaching_screens_nothing_on_an_arm_without_the_trees_wrist_camera(self) -> None:
        """⭐ HONEST SCREENS (review of F1, 2026-09-30): the tree hangs wrist camera 'wrist' on the arm and this arm was
        not handed its housing (``Robot.from_config`` hands none), so both models would judge a pose without it and a
        pose its housing collides in would read clear. Nothing is screened and no planner or guard is built for it
        (a housing handed in after them would be refused); one line says why, and where the looks are screened."""
        import tempfile
        from pathlib import Path

        from src.robot.execution.hand_guiding import HandGuide
        from src.robot.execution.teach import teach_poses
        from src.robot.safety.planning.band import PoseScreen, PoseVerdict
        from tests.test_a_pose_is_taught_by_hand import LOOK_1_DEG, TREE, _answers, _Clock, _Console, _GuidedArm, _stand

        screen = _canned_screen([PoseScreen(PoseVerdict.CLEAR, "the exact guard and the planner both clear it.")])

        class _BareArm(_GuidedArm):
            screen_configuration = screen

        clock = _Clock()
        arm = _BareArm([_stand(LOOK_1_DEG)], clock)
        arm.safety_preflight.wrist_bodies = lambda _arm=None: ()
        console = _Console(_answers("", "", "", "q"), arm.log)
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        taught = teach_poses(arm, tree=TREE, store=folder / "poses.json", out=lambda line: None,
                             guide=HandGuide(console, None, clock=clock, sleep=clock.sleep))
        self.assertEqual(len(taught), 1, "the pose is taught all the same")
        self.assertEqual(screen.asked, [], "no screen, so no planner and no guard is built")
        said = console.said
        self.assertFalse(any("screened once it is taught" in line for line in said))
        (why,) = [line for line in said if "not screened" in line]
        for part in ("'wrist'", "housing", "Robot.from_tree", "--start-planner"):
            self.assertIn(part, why)

    def test_a_campaign_says_each_look_before_its_first_pick(self) -> None:
        from types import SimpleNamespace

        from src.robot.execution.pick_run import PickRun, Recording
        from src.robot.safety.planning.band import PoseScreen, PoseVerdict

        screen = _canned_screen([PoseScreen(PoseVerdict.BAND, "straight lines run."),
                                 PoseScreen(PoseVerdict.GUARD_REFUSED, "no move goes there.")])
        arm = type("_Arm", (), {"screen_configuration": screen, "home_joint_positions": (0.0,) * 6})()
        service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=SimpleNamespace(arm=arm)))
        run = PickRun.from_service(service, runs=0, recording=Recording.off(),
                                   look=[JointPositions(tuple(LOOK0)), "home"])
        with self.assertLogs("src.robot.execution.pick_run", "INFO") as logs:
            run.execute()
        text = "\n".join(logs.output)
        self.assertIn("INFO:src.robot.execution.pick_run:pick run: look (-42.5, -67.9, -78.6, -136.2, 94.6, -34.1) "
                      "deg: in the planner's cushion band", text)
        self.assertIn("ERROR:src.robot.execution.pick_run:pick run: ERROR look home: refused by the exact guard", text)
        self.assertEqual(screen.asked[1][0], [0.0] * 6, "home is screened at the arm's own home joints")

    def test_the_desk_says_each_configured_look_before_the_planner_stops(self) -> None:
        from unittest import mock

        from src.config.schema.robot import RobotConfig
        from src.robot.drivers.ur.arm import URRobotArm
        from src.robot.execution.planner_start import PlannerStart
        from src.robot.safety.planning.band import PoseScreen, PoseVerdict
        from tests.test_a_planner_starts_from_the_command_line import _TOOL, _Sidecar

        from types import SimpleNamespace

        _Sidecar(self)
        screen = _canned_screen([PoseScreen(PoseVerdict.BAND, "straight lines run.")])
        self.enterContext(mock.patch.object(URRobotArm, "screen_configuration", screen))
        cell = RobotConfig.model_validate({
            "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": "curobo"},
            "gripper": {"model": "robotiq_2f85", "tool_frame": _TOOL},
            "safety": {"self_collision": {"kinematics_model": "ur5e", "planner_margin_mm": 4.0, "backend": "fcl"}},
            "look_joint_positions_deg": [list(LOOK0_DEG)],
        })
        # The camera section of the same tree, as Cell.start_planner hands it: here no wrist camera hangs on the arm.
        report = PlannerStart.from_robot_config(cell, camera=SimpleNamespace(cameras=SimpleNamespace(rigs=[]))).run()
        self.assertTrue(report.started, report.render())
        self.assertEqual(report.looks, ("look 1 (-42.5, -67.9, -78.6, -136.2, 94.6, -34.1) deg: in the planner's "
                                        "cushion band: straight lines run.",))
        self.assertIn("  look 1 (-42.5", report.render())
        self.assertEqual(report.to_dict()["looks"], list(report.looks))

        # Without the camera section nobody knows which wrist cameras hang on the arm: no look is screened, and the
        # report says so rather than screening them without a housing that may be there.
        screen.asked.clear()  # type: ignore[attr-defined]
        blind = PlannerStart.from_robot_config(cell).run()
        self.assertTrue(blind.started, blind.render())
        self.assertEqual(screen.asked, [])  # type: ignore[attr-defined]
        (said,) = blind.looks
        self.assertIn("not screened", said)
        self.assertIn("no camera section", said)


if __name__ == "__main__":
    unittest.main()
