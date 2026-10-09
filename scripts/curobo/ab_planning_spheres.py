"""The desk A/B before a cell plans on another robot or with fewer passes: plan success, the plans the judgement then
refuses, and the plan's time, between home and the looks and in the bench's hard scenes.

    .venv/Scripts/python.exe scripts/curobo/ab_planning_spheres.py --dry-run          # CPU: the problems, both robots
    WILLY_CUROBO_PYTHON=D:/dev/willy_handover/2026-10-02/final/gpu_aurora/wsl_sidecar.cmd \
    WILLY_PROBE_CUROBO_CONTENT=D:/dev/Workaholic-Willy/ext_deps/curobo/curobo/content \
    WILLY_COAL_PREFIX=D:/dev/Workaholic-Willy/ext_deps/coal_env \
        .venv/Scripts/python.exe scripts/curobo/ab_planning_spheres.py --work logs/curobo/ab_planning
    ... --models ,lean --finetune 3,0 --in-place 0     # the variants are every combination of the three lists

What runs is the cell's own planner, as ``real_cell --start-planner`` starts it: the arm built alone from the tree
(``--tree``, ``--profile``), its wrist cameras handed in, ``start_planner`` (the descriptor and evidence checks, the
declared world registered), with the tree changed in three keys only, per variant:
``safety.planned_motion.planning_spheres`` (``--models``, ``""`` today's robot, ``lean`` the planning model),
``safety.planned_motion.finetune_passes`` (``--finetune``) and ``safety.planning_world.register_in_place``
(``--in-place``). The first variant is the one the others are compared with.

The problems, the same for every variant:

* home to each look, each look to home, and each look to the next (``robot.look_joint_positions_deg``), in the
  declared world;
* each of the bench's hard scenes (``scripts/bench/scenes.py``, ``--scenes``), its parts and bins standing on the
  owner's mat as ``scripts/bench/run_bench.py`` places them: every part becomes the box the camera world would hand
  the planner and the guard (``seen_<part>``, turned as it stands), and for its target (every movable part of a
  clearing scene, at most ``--parts``) the arm's moves: the first look to above the part (``STANDOFF_MM``), above to
  the grasp, the grasp back up, above back to the look, and home to above. Each goal is the configuration nearest
  the start the arm's own IK finds (``ur_flange_ik``, ``nearest_goals``).

Every problem is planned, whatever the straight line would do: the arm plans only where the exact guard refuses the
line, about one move in a hundred on the bench, too few to compare. The world is set first (``CuroboUrPlanner.
set_world``, its round trip timed; the guard is handed the same boxes), then ``plan_joint`` from the problem's start
with the refresh off, timed on the wall clock, then a plan is judged as the arm judges one
(``URRobotArm._judged_waypoints``: shortened to the waypoints both authorities pass, or the plan as cuRobo returned
it). Nothing moves: the arm is never connected.

It writes ``<work>/ab_planning.json`` (every problem, every variant) and ``<work>/ab_planning.txt``: per variant the
problems planned, the plans found, the plans the judgement refused, the starts and goals the check refused before any
plan, the plan time p50 and p90 over every plan asked and over the plans found, and the set_world p50; then, against
the first variant, the problems both plan, only one plans, and the plans only one has judged clear. Exit 0 when every
variant ran, 2 when a planner could not start.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _path in (str(REPO), str(REPO / "scripts" / "bench"), str(REPO / "scripts" / "ursim")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: The owner's cell as it was handed over on 2026-10-08 (the 131 mm enclosure, the looks 0, 3, 2, 1).
DEFAULT_TREE = Path("D:/dev/willy_handover/2026-10-08/robominds_config/config")
#: The wrist camera's calibration the bench places the camera with (``run_bench.DEFAULT_CALIBRATION``).
DEFAULT_CALIBRATION = Path("D:/helper_cell/calibration/real/eih_EIH_Cam.json")
#: The recorded look the bench stands its scenes on (``run_bench.DEFAULT_CROP``).
DEFAULT_CROP = REPO / "tests" / "data" / "cell_2026_10_01" / "p1_pick-63e1b3dac29f.npz"
#: The policy's standoff above a grasp, millimetres, as the console's pick moves.
STANDOFF_MM = 80.0
#: How far under a part's top the TCP stands at the grasp, millimetres: about half a Hand-E pad.
GRASP_DEPTH_MM = 15.0
#: The joint speed the arm ranks its goals by, rad/s (``motion_limits.max_velocity`` on the owner's cell).
RANK_SPEED = 0.5


@dataclass
class Problem:
    """One move to plan: where from, where to, in which world. Joints in UR order, radians; boxes in the controller's
    base, the planner's wire format (metres, WXYZ) and the guard's boxes."""

    key: str
    family: str
    start: list[float]
    goal: list[float]
    world: list[dict[str, Any]] = field(default_factory=list)
    guard: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class Variant:
    model: str
    finetune: int
    in_place: bool

    @property
    def label(self) -> str:
        return f"{self.model or 'today'}/finetune {self.finetune}/{'in place' if self.in_place else 'update_world'}"

    def values(self) -> "dict[str, Any]":
        return {"robot.safety.planned_motion.planning_spheres": self.model,
                "robot.safety.planned_motion.finetune_passes": self.finetune,
                "robot.safety.planning_world.register_in_place": self.in_place}


def load_cell(tree: Path, profile: str, calibration: "Path | None") -> Any:
    """The cell's tree, the calibration the bench uses written over its primary rig where one is named."""
    from src.config.tree import load_tree  # noqa: PLC0415

    loaded = load_tree(profile or None, root=tree)
    if not loaded.ok:
        raise SystemExit(f"the tree at {tree} does not load: {loaded.error}")
    if calibration is not None and calibration.is_file():
        section = loaded.app_config.camera.cameras
        for index, rig in enumerate(section.rigs):
            if rig.rig_id == section.primary_rig_id and getattr(rig, "extrinsics", None) is not None:
                loaded = loaded.with_values({f"camera.cameras.rigs[{index}].extrinsics.artifact_path":
                                             calibration.resolve().as_posix()})
    return loaded


def _quaternion_wxyz(rotation: np.ndarray) -> "list[float]":
    from src.robot.safety.planning._curobo_body_links import rotation_to_wxyz  # noqa: PLC0415

    return list(rotation_to_wxyz(rotation.tolist()))


def part_box(part: Any) -> "tuple[dict[str, Any], Any]":
    """A bench part as the camera world would give it: the planner's box (wire format) and the guard's, turned alike."""
    from src.robot.safety._capsule import AxisAlignedBox, TurnedBox  # noqa: PLC0415

    hx = hy = float(part.dims[0]) if part.shape == "cylinder" else None
    if hx is None:
        hx, hy = float(part.dims[0]) / 2.0, float(part.dims[1]) / 2.0
    half = np.array([hx, hy, float(part.height_mm) / 2.0])
    frame = np.asarray(part.frame, dtype=np.float64)
    centre = frame[:3, :3] @ np.array([0.0, 0.0, half[2]]) + frame[:3, 3]
    rotation = frame[:3, :3]
    name = f"seen_{part.name}"
    wire = {"name": name, "dims_m": [float(v) * 2.0 / 1000.0 for v in half],
            "pose": [*(float(v) / 1000.0 for v in centre), *_quaternion_wxyz(rotation)]}
    enclosing = np.abs(rotation) @ half
    guard = AxisAlignedBox(center_mm=centre, half_extents_mm=enclosing, name=name,
                           turned=TurnedBox(half_extents_mm=half, yaw_rad=math.atan2(rotation[1, 0], rotation[0, 0]),
                                            rotation=rotation))
    return wire, guard


def goal_for(robot: Any, tcp_mm: np.ndarray, near: "list[float]", window: "tuple[list[float], list[float]]") \
        -> "list[float] | None":
    """The configuration nearest ``near`` that puts the TCP at ``tcp_mm``, as the arm chooses a goal, or ``None``."""
    from src.robot.drivers.ur.tool_frame import tool_frame_matrix  # noqa: PLC0415
    from src.robot.safety._ur_ik import nearest_goals, ur_flange_ik  # noqa: PLC0415

    frame = robot.gripper.tool_frame
    tool = tool_frame_matrix(tuple(frame.offset_mm), tuple(frame.rotation_quat_xyzw))
    flange = np.asarray(tcp_mm, dtype=np.float64) @ np.linalg.inv(tool)
    solutions = ur_flange_ik(str(robot.ur.model).lower(), flange) or ()
    goals = nearest_goals(solutions, current=near, lower=window[0], upper=window[1], velocity=RANK_SPEED)
    return [float(v) for v in goals[0].joints] if goals else None


def _top_down(part: Any, depth_mm: float) -> np.ndarray:
    """A TCP over ``part``: approaching against its up axis, closing along its own x, ``depth_mm`` under its top."""
    frame = np.asarray(part.frame, dtype=np.float64)
    up = frame[:3, 2] / np.linalg.norm(frame[:3, 2])
    across = frame[:3, 0] - up * float(frame[:3, 0] @ up)
    across /= np.linalg.norm(across)
    tcp = np.eye(4)
    tcp[:3, 2] = -up
    tcp[:3, 0] = across
    tcp[:3, 1] = np.cross(tcp[:3, 2], tcp[:3, 0])
    tcp[:3, 3] = frame[:3, 3] + up * (float(part.height_mm) - depth_mm)
    return tcp


def problems_for(loaded: Any, *, scenes: str, parts: int, crop: Path) -> "tuple[list[Problem], list[str]]":
    """Every problem of this A/B, and a line for every move that has no goal (out of reach, no solution in the window)."""
    from src.robot.drivers.ur.curobo_motion import PLANNER_JOINT_ENVELOPE_RAD  # noqa: PLC0415

    robot = loaded.robot
    # Home is read in radians: the schema turns home_joint_positions_deg into home_joint_positions and clears the one.
    home = [float(v) for v in (robot.home_joint_positions or ())]
    looks = [[math.radians(float(v)) for v in row] for row in (robot.look_joint_positions_deg or ())]
    window = (list(PLANNER_JOINT_ENVELOPE_RAD[0]), list(PLANNER_JOINT_ENVELOPE_RAD[1]))
    out: list[Problem] = []
    skipped: list[str] = []
    for index, look in enumerate(looks):
        out.append(Problem(f"home->look{index}", "looks", list(home), list(look)))
        out.append(Problem(f"look{index}->home", "looks", list(look), list(home)))
        if index + 1 < len(looks):
            out.append(Problem(f"look{index}->look{index + 1}", "looks", list(look), list(looks[index + 1])))
    if not scenes:
        return out, skipped
    import _mat_scene as ms  # noqa: PLC0415
    from scenes import SCENES, scene_names  # noqa: PLC0415

    static = ms.StaticScene.from_crop(crop)
    static.whole_bench = True
    first = looks[0] if looks else home
    for name in scene_names(scenes):
        scene = SCENES[name]
        built = scene.build(static)
        boxes = [part_box(part) for part in built]
        world, guard = [wire for wire, _ in boxes], [held for _, held in boxes]
        targets = [part for part in built if part.name == scene.target] or [part for part in built if not part.fixed]
        for part in targets[:max(1, parts)]:
            grasp_tcp, above_tcp = _top_down(part, GRASP_DEPTH_MM), _top_down(part, GRASP_DEPTH_MM - STANDOFF_MM)
            above = goal_for(robot, above_tcp, first, window)
            grasp = goal_for(robot, grasp_tcp, above, window) if above is not None else None
            if above is None or grasp is None:
                skipped.append(f"{name}/{part.name}: no goal {'above' if above is None else 'at'} the part in the window")
                continue
            from_home = goal_for(robot, above_tcp, home, window)
            moves = [("look->above", first, above), ("above->grasp", above, grasp), ("grasp->above", grasp, above),
                     ("above->look", above, first)]
            if from_home is not None:
                moves.append(("home->above", home, from_home))
            for move, start, goal in moves:
                out.append(Problem(f"{name}/{part.name}/{move}", scene.family, list(start), list(goal),
                                   world=list(world), guard=list(guard)))
    return out, skipped


def _travel_deg(path: "list[list[float]]") -> float:
    steps = np.abs(np.diff(np.asarray(path, dtype=np.float64), axis=0))
    return float(np.degrees(steps.sum())) if len(path) > 1 else 0.0


def run_variant(loaded: Any, variant: Variant, problems: "list[Problem]") -> "tuple[list[dict[str, Any]], str]":
    """Start the cell's planner on ``variant``, plan and judge every problem, stop it. Rows, and what it loaded."""
    from src.robot.core import MotionCommand  # noqa: PLC0415
    from src.robot.core.joint_positions import JointPositions  # noqa: PLC0415
    from src.robot.drivers import create_arm  # noqa: PLC0415
    from src.robot.execution.wrist_bodies import WristBodies  # noqa: PLC0415

    tree = loaded.with_values(variant.values())
    robot = tree.robot
    arm: Any = create_arm("ur", config=robot)
    WristBodies.from_config(robot, tree.app_config.camera, data_dir=tree.root).hand_to(arm)
    rows: list[dict[str, Any]] = []
    try:
        began = time.perf_counter()
        identity = arm.start_planner()
        print(f"  started in {time.perf_counter() - began:.1f} s: plans on "
              f"{identity.plans_on or 'the evidence model'}", flush=True)
        planner = arm._curobo_ur  # noqa: SLF001 (the glue the start kept: what a move asks)
        for problem in problems:
            row: dict[str, Any] = {"key": problem.key, "family": problem.family}
            try:
                arm._preflight.set_perceived_obstacles(problem.guard)  # noqa: SLF001
                began = time.perf_counter()
                held = planner.set_world([dict(box) for box in problem.world])
                row["set_world_ms"] = (time.perf_counter() - began) * 1000.0
                # The camera's boxes the planner and the guard now both hold, as a confirmed refresh leaves them.
                planner._seen_held = tuple(box["name"] for box in problem.world)  # noqa: SLF001
                row["world"] = held
                began = time.perf_counter()
                plan = planner.plan_joint(problem.goal, start_ur=problem.start, refresh=False)
                row["plan_ms"] = (time.perf_counter() - began) * 1000.0
                refusal = planner.last_refusal
                row["found"] = bool(plan)
                row["refused_before"] = (str(getattr(refusal, "where", "")) if not plan and refusal is not None
                                         and str(getattr(refusal, "where", "")) in ("start", "goal") else "")
                if plan:
                    row["waypoints"] = len(plan)
                    row["travel_deg"] = _travel_deg(plan)
                    runs, refused = arm._judged_waypoints(  # noqa: SLF001 (the arm's own judgement of a plan)
                        planner, [list(w) for w in plan], command=MotionCommand.MOVE_JOINTS,
                        target_joints=JointPositions.from_list(problem.goal))
                    row["judged"] = "refused" if refused is not None else "clear"
                    row["judged_why"] = str(refused.message)[:300] if refused is not None else ""
                    row["moves"] = len(runs)
            except Exception as exc:  # noqa: BLE001 (one problem that raised is a row that says so, not a lost run)
                row["raised"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()[-1500:]
            rows.append(row)
            print(f"    {problem.key:48s} " + _said(row), flush=True)
        loaded_text = identity.render()
    finally:
        arm.stop_planner()
    return rows, loaded_text


def _said(row: "dict[str, Any]") -> str:
    if "raised" in row:
        return f"RAISED {row['raised'][:120]}"
    if row.get("refused_before"):
        return f"refused at the {row['refused_before']} before any plan ({row['plan_ms']:.0f} ms)"
    if not row.get("found"):
        return f"no plan ({row['plan_ms']:.0f} ms)"
    return f"plan of {row['waypoints']} in {row['plan_ms']:.0f} ms, judged {row['judged']}"


def _percentiles(values: "list[float]") -> "dict[str, float | None]":
    if not values:
        return {"p50": None, "p90": None, "n": 0}
    ordered = sorted(values)
    return {"p50": float(statistics.median(ordered)),
            "p90": float(ordered[min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered))) - 1)]), "n": len(ordered)}


def summary(rows: "list[dict[str, Any]]") -> "dict[str, Any]":
    asked = [row for row in rows if "raised" not in row and not row.get("refused_before")]
    found = [row for row in asked if row.get("found")]
    return {
        "problems": len(rows),
        "raised": sum(1 for row in rows if "raised" in row),
        "refused_before_any_plan": sum(1 for row in rows if row.get("refused_before")),
        "planned": len(asked),
        "found": len(found),
        "judged_refused": sum(1 for row in found if row.get("judged") == "refused"),
        "judged_clear": sum(1 for row in found if row.get("judged") == "clear"),
        "plan_ms": _percentiles([row["plan_ms"] for row in asked]),
        "plan_ms_found": _percentiles([row["plan_ms"] for row in found]),
        "plan_ms_not_found": _percentiles([row["plan_ms"] for row in asked if not row.get("found")]),
        "set_world_ms": _percentiles([row["set_world_ms"] for row in rows if "set_world_ms" in row and row["world"]]),
        "travel_deg_found": _percentiles([row["travel_deg"] for row in found]),
    }


def against(base: "list[dict[str, Any]]", other: "list[dict[str, Any]]") -> "dict[str, Any]":
    """The problems both variants plan, and those only one plans or only one has judged clear."""
    by_key = {row["key"]: row for row in other}
    both = only_base = only_other = clear_only_base = clear_only_other = 0
    for row in base:
        theirs = by_key.get(row["key"])
        if theirs is None:
            continue
        mine, yours = bool(row.get("found")), bool(theirs.get("found"))
        both += mine and yours
        only_base += mine and not yours
        only_other += yours and not mine
        clear_mine, clear_yours = row.get("judged") == "clear", theirs.get("judged") == "clear"
        clear_only_base += clear_mine and not clear_yours
        clear_only_other += clear_yours and not clear_mine
    return {"both_found": both, "only_base_found": only_base, "only_other_found": only_other,
            "only_base_judged_clear": clear_only_base, "only_other_judged_clear": clear_only_other}


def dry_run(loaded: Any, models: "list[str]", problems: "list[Problem]", skipped: "list[str]", *,
            dump: "Path | None" = None) -> int:
    """The problems, and each model's spheres and self pairs composed on the CPU as the sidecar would compose them;
    with ``dump``, the robots written there for ``probe_planning_model.py`` to load into cuRobo."""
    import yaml  # noqa: PLC0415

    from src.robot.drivers.sim.robot_models import curobo_arm_descriptor  # noqa: PLC0415
    from src.robot.execution.wrist_bodies import WristBodies  # noqa: PLC0415
    from src.robot.safety.planning import _curobo_planning as planning  # noqa: PLC0415
    from src.robot.safety.planning._curobo_body_links import compose_for_cell  # noqa: PLC0415
    from src.robot.safety.planning.body_link import HandLink, coupling_bodies  # noqa: PLC0415
    from src.robot.safety.planning.hand import PlannerHand, planner_hand  # noqa: PLC0415
    from src.robot.safety.planning.planning_model import planning_payload  # noqa: PLC0415

    families: dict[str, int] = {}
    for problem in problems:
        families[problem.family] = families.get(problem.family, 0) + 1
    print(f"[problems] {len(problems)}: " + ", ".join(f"{family} {count}" for family, count in families.items()))
    for line in skipped:
        print(f"  skipped {line}")
    robot = loaded.robot
    content = Path(__import__("os").environ.get("WILLY_PROBE_CUROBO_CONTENT") or
                   REPO / "ext_deps" / "curobo" / "curobo" / "content")
    descriptor = content / "configs" / "robot" / curobo_arm_descriptor(str(robot.ur.model))
    if not descriptor.is_file():
        print(f"[robots] no {descriptor.name} under {content}: nothing to compose here")
        return 0
    raw = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
    hand = planner_hand(robot, data_dir=loaded.root)
    if not isinstance(hand, PlannerHand):
        print("[robots] the tree names no hand: nothing to compose here")
        return 0
    bodies = [HandLink.from_hand(hand).to_dict(), *coupling_bodies(hand)]
    wrist = [body.link() for body in WristBodies.from_config(robot, loaded.app_config.camera,
                                                            data_dir=loaded.root).bodies]
    margin = float(robot.safety.self_collision.planner_margin_mm or 0.0)
    today, _, _, _, _ = compose_for_cell(raw, bodies=bodies, wrist_bodies=wrist, margin_mm=margin)
    print(f"[robots] today: {sum(planning.sphere_count(today).values())} spheres, "
          f"{planning.self_pair_count(today)} self pairs ({planning.sphere_count(today)})")
    urdf = content / "assets" / str(raw["robot_cfg"]["kinematics"].get("urdf_path", ""))
    written = {"today": today}
    for model in [model for model in models if model]:
        sent = planning_payload(model, arm=str(robot.ur.model).lower(), body_links=bodies, wrist_body_links=wrist)
        if isinstance(sent, str):
            print(f"[robots] {model}: not composed: {sent}")
            continue
        if not urdf.is_file():
            print(f"[robots] {model}: {sent.render()}; no URDF at {urdf} to place it in")
            continue
        planned = planning.planning_config(raw, planning.read_payload(sent.to_json()),
                                           urdf_text=urdf.read_text(encoding="utf-8"), evidence=today, margin_mm=margin)
        print(f"[robots] {model}: {sum(planning.sphere_count(planned).values())} spheres, "
              f"{planning.self_pair_count(planned)} self pairs ({planning.sphere_count(planned)})")
        written[model] = planned
    if dump is not None:
        dump.mkdir(parents=True, exist_ok=True)
        for name, config in written.items():
            (dump / f"{name}.json").write_text(json.dumps(config), encoding="utf-8")
        (dump / "manifest.json").write_text(json.dumps({"robots": list(written), "urdf": urdf.resolve().as_posix()}),
                                           encoding="utf-8")
        print(f"[robots] written to {dump}")
    return 0


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tree", default=str(DEFAULT_TREE), help="the cell's config folder; read, never written")
    parser.add_argument("--profile", default="", help="the tree's chain; empty for its base")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION),
                        help="the wrist camera's calibration, written over the primary rig's in memory where it exists")
    parser.add_argument("--crop", default=str(DEFAULT_CROP), help="the recorded look the scenes stand on")
    parser.add_argument("--scenes", default="bin,clutter,tray,shapes",
                        help="all, families or names of scripts/bench/scenes.py; empty for the looks alone")
    parser.add_argument("--parts", type=int, default=2, help="targets per clearing scene")
    parser.add_argument("--models", default=",lean", help="planning_spheres per variant, comma separated; '' is today's")
    parser.add_argument("--finetune", default="3", help="finetune_passes per variant, comma separated")
    parser.add_argument("--in-place", default="0", help="register_in_place per variant, 0 or 1, comma separated")
    parser.add_argument("--limit", type=int, default=0, help="at most this many problems (0: all)")
    parser.add_argument("--work", default=str(REPO / "logs" / "curobo" / time.strftime("ab_planning_%Y%m%d_%H%M%S")))
    parser.add_argument("--dry-run", action="store_true", help="build the problems and compose both robots, plan nothing")
    parser.add_argument("--dump", default="", help="with --dry-run: write the robots here for probe_planning_model.py")
    args = parser.parse_args(argv)

    models = [value.strip() for value in args.models.split(",")]
    variants = [Variant(model=model, finetune=int(finetune), in_place=bool(int(in_place)))
                for model in models for finetune in args.finetune.split(",") for in_place in args.in_place.split(",")]
    loaded = load_cell(Path(args.tree), args.profile, Path(args.calibration) if args.calibration else None)
    problems, skipped = problems_for(loaded, scenes=args.scenes, parts=args.parts, crop=Path(args.crop))
    if args.limit > 0:
        problems = problems[:args.limit]
    if args.dry_run:
        return dry_run(loaded, models, problems, skipped, dump=Path(args.dump) if args.dump else None)

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"tree": args.tree, "profile": args.profile, "scenes": args.scenes,
                              "problems": [problem.key for problem in problems], "skipped": skipped, "variants": {}}
    for variant in variants:
        print(f"[variant] {variant.label}", flush=True)
        try:
            rows, said = run_variant(loaded, variant, problems)
        except Exception as exc:  # noqa: BLE001 (a planner that does not start is this run's answer)
            print(f"  the planner did not start: {type(exc).__name__}: {exc}", flush=True)
            report["variants"][variant.label] = {"started": False, "reason": f"{type(exc).__name__}: {exc}"}
            (work / "ab_planning.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
            return 2
        report["variants"][variant.label] = {"started": True, "loaded": said, "summary": summary(rows), "rows": rows}
        (work / "ab_planning.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    labels = [variant.label for variant in variants]
    lines = []
    for label in labels:
        lines.append(f"{label}: " + json.dumps(report["variants"][label]["summary"]))
    for label in labels[1:]:
        compared = against(report["variants"][labels[0]]["rows"], report["variants"][label]["rows"])
        report.setdefault("against_first", {})[label] = compared
        lines.append(f"{label} against {labels[0]}: " + json.dumps(compared))
    (work / "ab_planning.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (work / "ab_planning.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
