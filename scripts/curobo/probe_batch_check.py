"""Probe the batch collision query of the installed cuRobo, on the planner the sidecar builds.

Run from the repository root with the cuRobo environment's interpreter, once per model::

    PYTHONIOENCODING=utf-8 ext_deps/curobo_env/python.exe scripts/curobo/probe_batch_check.py ur5e
    PYTHONIOENCODING=utf-8 ext_deps/curobo_env/python.exe scripts/curobo/probe_batch_check.py ur3e

The self-collision margin defaults to the figure the cell config carries for the model (4.0 mm for both,
robot.sim.yaml and robot.ur3e.yaml) and can be overridden with ``--margin-mm``.

Why it exists. The sidecar cannot be imported in CI, so its branches are pinned by source scans, and a
scan can only pin a method name somebody has actually called. A name that did not exist (``plan_js`` on
the planner, where this build has ``plan_cspace``) once cost three commits. Every name this probe lists
as proven is one it called and got an answer from; a call that raised is listed with its message.

What it does, in order:

  1. builds the planner as ``curobo_planner_server.py`` does (same descriptor, same derived margin and
     payload link, same boot world, same cuboid cache, same warm-up), and a ``RobotCollisionChecker`` on
     the planner's own ``scene_collision_checker``;
  2. checks a straight joint path through a wall and a clear one, through ``validate``, through
     ``get_scene_self_collision_distance_from_joints``, and through a planner-sphere variant;
  3. moves the wall with the planner's ``update_world`` and checks again;
  4. attaches a payload to the planner and checks whether each variant sees it;
  5. sweeps a wall a few millimetres either side of the arm to settle what "valid" means;
  6. settles which self-collision config the checker uses (derived file against raw descriptor);
  7. times batches of 1, 50, 200 and 800 configurations after warm-up;
  8. restores the boot world and plans once more, to show the probe left the planner working.

It writes ``logs/curobo/batch_check_probe_<model>.json`` and prints a short summary. Exit code 0 when
every section ran without a failed call, 1 otherwise; the JSON names each failure with its message.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PLANNING_DIR = REPO / "src" / "robot" / "safety" / "planning"
# The sidecar imports the module that composes its config as a plain sibling, because the client spawns
# it by path. Putting the same directory first makes the same composition run here.
sys.path.insert(0, str(PLANNING_DIR))

#: The planner margin each model's cell config carries (robot.sim.yaml and robot.ur3e.yaml, planner_margin_mm),
#: the same 4.0 for the whole family on the refitted sphere map.
CONFIG_MARGIN_MM = {"ur5e": 4.0, "ur3e": 4.0}
#: robot.yaml payload.sphere_slots, the value a cell that declares a carried part length hands the sidecar.
PAYLOAD_SPHERE_SLOTS = 16
#: Mirrors of the sidecar's module constants in curobo_planner_server.py.
PLAN_MAX_ATTEMPTS = int(os.environ.get("WILLY_CUROBO_MAX_ATTEMPTS", "16"))
PLAN_GRAPH_FROM = int(os.environ.get("WILLY_CUROBO_GRAPH_FROM_ATTEMPT", "1"))
CUBOID_CACHE = 16
BOOT_WORLD = {"table": {"dims": [1.6, 1.6, 0.05], "pose": [0.0, 0.0, -0.026, 1, 0, 0, 0]}}
#: A world with nothing near the arm, used as the control: the same wall, five metres below the base.
FAR_WORLD = {"wall": {"dims": [0.20, 0.02, 0.40], "pose": [0.0, 0.0, -5.0, 1, 0, 0, 0]}}

BATCH_SIZES = (1, 50, 200, 800)
PATH_SAMPLES = 41
PAN_AMPLITUDE_RAD = 0.9
WRIST_SWING_RAD = 0.3
GAPS_MM = (20.0, 5.0, 3.0, 1.0, 0.5, -0.5, -1.0, -3.0, -5.0)
SELF_SAMPLES = 2000


#
# Pure helpers
#


def _verdict(flags: "list[bool]") -> dict:
    """One verdict over per-sample flags: valid, the first invalid index, and how many were refused."""
    first = next((i for i, ok in enumerate(flags) if not ok), None)
    return {
        "valid": first is None,
        "first_invalid": first,
        "invalid_count": sum(1 for ok in flags if not ok),
        "checked": len(flags),
    }


def _interpolate(start: "list[float]", end: "list[float]", n: int) -> "list[list[float]]":
    """``n`` samples of the straight joint path, both ends included; one sample is the midpoint."""
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(end, dtype=np.float64)
    if n == 1:
        return [((a + b) / 2.0).tolist()]
    return [(a + (b - a) * (i / (n - 1))).tolist() for i in range(n)]


def _quat_about_z(theta: float) -> "list[float]":
    return [math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)]


def _quat_wxyz_to_matrix(q: "np.ndarray") -> "np.ndarray":
    w, x, y, z = (float(v) for v in q / np.linalg.norm(q))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_x_onto(axis: "np.ndarray") -> "list[float]":
    """The WXYZ rotation that turns the box's local x axis onto ``axis``."""
    a = axis / np.linalg.norm(axis)
    x = np.array([1.0, 0.0, 0.0])
    d = float(np.dot(x, a))
    if d < -0.999999:
        return [0.0, 0.0, 0.0, 1.0]
    c = np.cross(x, a)
    q = np.array([1.0 + d, c[0], c[1], c[2]])
    return (q / np.linalg.norm(q)).tolist()


def _min_clearance_to_aabb_mm(spheres: "np.ndarray", centre: "list[float]", dims: "list[float]") -> float:
    """Smallest sphere surface to box distance, negative when a sphere penetrates. Axis-aligned box."""
    c = np.asarray(centre, dtype=np.float64)
    half = np.asarray(dims, dtype=np.float64) / 2.0
    d = np.abs(spheres[:, :3] - c) - half
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    return float(np.min(outside + inside - spheres[:, 3]) * 1000.0)


def _stats_ms(samples: "list[float]") -> dict:
    ordered = sorted(samples)
    p90 = ordered[min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered))) - 1)]
    return {
        "median_ms": round(statistics.median(ordered), 3),
        "p90_ms": round(p90, 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "repeats": len(ordered),
    }


#
# The probe
#


class Probe:
    """Holds the planner, the checkers and every measurement, so a failed section cannot hide."""

    def __init__(self, model: str, margin_mm: float, attach_spheres: int, repeats: int) -> None:
        self.model = model
        self.margin_mm = float(margin_mm)
        self.attach_spheres = int(attach_spheres)
        self.repeats = int(repeats)
        self.proven: "list[str]" = []
        self.failures: "list[dict]" = []
        self.result: dict = {"model": model, "margin_mm": self.margin_mm, "sections": {},
                             "section_time_s": {}}
        # Filled by build(). Typed Any rather than left to inference: they hold cuRobo and torch
        # objects that only exist inside the sidecar's own interpreter, so there is no type here to
        # name and a checker would otherwise read every use as an attribute of None.
        self.torch: Any = None
        self.planner: Any = None
        self.checker: Any = None
        self.default_q: "list[float]" = []
        self.through: "list[list[float]]" = []
        self.clear: "list[list[float]]" = []
        self.wall_a: dict = {}

    # bookkeeping

    def call(self, label: str, fn, *args, **kwargs):
        """Call ``fn``; a success records ``label`` as proven, a failure records the exact message."""
        try:
            out = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001, recorded with its message and re-raised
            self.failures.append({"call": label, "error": f"{type(exc).__name__}: {exc}"})
            raise
        if label not in self.proven:
            self.proven.append(label)
        return out

    def section(self, name: str, fn) -> bool:
        started = time.perf_counter()
        ok = True
        try:
            self.result["sections"][name] = fn()
        except Exception as exc:  # noqa: BLE001, every section is reported rather than stopping the run
            ok = False
            message = f"{type(exc).__name__}: {exc}"
            self.result["sections"][name] = {"error": message,
                                             "traceback": traceback.format_exc(limit=8)}
            self.failures.append({"section": name, "error": message})
            print(f"[{name}] FAILED: {message}", flush=True)
        self.result["section_time_s"][name] = round(time.perf_counter() - started, 3)
        return ok

    # building, exactly as the sidecar does

    def build(self) -> dict:
        import torch  # type: ignore[import-not-found]

        import curobo  # type: ignore[import-not-found]
        from curobo._src.geom.types import SceneCfg  # type: ignore[import-not-found]
        from curobo.collision_checking import (  # type: ignore[import-not-found]
            RobotCollisionChecker,
            RobotCollisionCheckerCfg,
        )
        from curobo.content import get_content_root  # type: ignore[import-not-found]
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # type: ignore[import-not-found]
        from curobo.types import JointState  # type: ignore[import-not-found]

        from curobo._src.util.config_io import resolve_config  # type: ignore[import-not-found]

        from _curobo_body_links import canonical_sha256, compose_with_counts  # type: ignore[import-not-found]

        self.torch = torch
        self.SceneCfg = SceneCfg
        self.JointState = JointState
        self.RobotCollisionChecker = RobotCollisionChecker
        self.RobotCollisionCheckerCfg = RobotCollisionCheckerCfg

        info: dict = {
            "curobo_version": str(getattr(curobo, "__version__", "unknown")),
            "torch_version": str(torch.__version__),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": sys.version.split()[0],
        }
        robot = f"{self.model}.yml"
        raw = os.path.join(str(get_content_root()), "configs", "robot", robot)
        self.raw_descriptor = raw
        info["descriptor"] = robot
        info["raw_descriptor_path"] = raw
        # The one dict the sidecar composes (curobo_planner_server.py, _curobo_body_links): the margin, then
        # the payload link, with every consumer built from its own deep copy.
        composed, adjusted, added = compose_with_counts(
            resolve_config(raw), margin_mm=self.margin_mm, attach_spheres=self.attach_spheres,
        )
        if self.margin_mm > 0.0:
            info["margin_links_adjusted"] = adjusted
        if self.attach_spheres > 0:
            info["attach_link_added"] = bool(added)
        info["robot_in_use"] = f"{robot} composed in memory"
        info["composed_sha256"] = canonical_sha256(composed)
        self.robot_in_use = composed
        self.robot_in_use_label = info["robot_in_use"]

        started = time.perf_counter()
        cfg = self.call("curobo.motion_planner.MotionPlannerCfg.create", MotionPlannerCfg.create,
                        robot=copy.deepcopy(composed), scene_model={"cuboid": dict(BOOT_WORLD)},
                        collision_cache={"cuboid": CUBOID_CACHE})
        self.planner = self.call("curobo.motion_planner.MotionPlanner", MotionPlanner, cfg)
        self.call("MotionPlanner.warmup", self.planner.warmup, enable_graph=True, num_warmup_iterations=5)
        info["planner_ready_s"] = round(time.perf_counter() - started, 2)
        self.default_q = self.planner.default_joint_state.position.squeeze().cpu().tolist()
        info["joint_names"] = list(self.planner.joint_names)
        info["default_q"] = [round(v, 4) for v in self.default_q]
        limits = self.call("Kinematics.get_joint_limits", self.planner.kinematics.get_joint_limits)
        self.joint_low = limits.position[0].cpu().tolist()
        self.joint_high = limits.position[1].cpu().tolist()
        info["joint_limits"] = [[round(a, 4), round(b, 4)] for a, b in zip(self.joint_low, self.joint_high)]

        info["plan_before"] = self.plan_once()

        started = time.perf_counter()
        self.checker = self.make_checker(self.robot_in_use, collision_activation_distance=0.0)
        info["checker_build_s"] = round(time.perf_counter() - started, 2)
        info["checker_joint_names_match_planner"] = (
            list(self.checker.kinematics.joint_names) == list(self.planner.joint_names))
        info["checker_scene_model_is_planner_scene_collision_checker"] = (
            self.checker.scene_model is self.planner.scene_collision_checker)
        info["checker_constraint_uses_planner_scene_collision_checker"] = (
            self.checker.collision_constraint.config.scene_collision_checker
            is self.planner.scene_collision_checker)
        info["checker_total_spheres"] = int(self.checker.kinematics.total_spheres)
        info["planner_total_spheres"] = int(self.planner.kinematics.total_spheres)
        return info

    def make_checker(self, robot_config: "str | dict", **kwargs):
        cfg = self.call("curobo.collision_checking.RobotCollisionCheckerCfg.load_from_config",
                        self.RobotCollisionCheckerCfg.load_from_config, robot_config=copy.deepcopy(robot_config),
                        scene_collision_checker=self.planner.scene_collision_checker, **kwargs)
        return self.call("curobo.collision_checking.RobotCollisionChecker", self.RobotCollisionChecker, cfg)

    # the three ways to ask

    def q3(self, configs: "list[list[float]]"):
        """``[batch=1, horizon=len(configs), dof]`` on the GPU, in the planner's joint order."""
        return self.torch.tensor(configs, device="cuda", dtype=self.torch.float32).unsqueeze(0)

    def set_world(self, cuboids: dict) -> None:
        """Replace the planner's world the way the sidecar's set_world branch does."""
        self.call("MotionPlanner.update_world", self.planner.update_world,
                  self.SceneCfg.create({"cuboid": cuboids}))

    def by_validate(self, checker, configs: "list[list[float]]", flags_out: "list[bool] | None" = None) -> dict:
        mask = self.call("RobotCollisionChecker.validate", checker.validate, self.q3(configs))
        flags = [bool(v) for v in mask.reshape(-1).cpu().tolist()]
        if flags_out is not None:
            flags_out.extend(flags)
        return _verdict(flags)

    def by_validate_batch_layout(self, checker, configs: "list[list[float]]") -> dict:
        """The same configurations as ``[batch=N, horizon=1, dof]``."""
        q = self.torch.tensor(configs, device="cuda", dtype=self.torch.float32).unsqueeze(1)
        mask = self.call("RobotCollisionChecker.validate", checker.validate, q)
        return _verdict([bool(v) for v in mask.reshape(-1).cpu().tolist()])

    def by_distance(self, checker, configs: "list[list[float]]") -> dict:
        h = len(configs)
        d_world, d_self = self.call("RobotCollisionChecker.get_scene_self_collision_distance_from_joints",
                                    checker.get_scene_self_collision_distance_from_joints, self.q3(configs))
        world = d_world.reshape(1, h, -1).amax(dim=-1).reshape(-1).cpu().tolist()
        own = d_self.reshape(1, h, -1).amax(dim=-1).reshape(-1).cpu().tolist()
        out = _verdict([w <= 0.0 and s <= 0.0 for w, s in zip(world, own)])
        out["max_scene_term"] = float(max(world))
        out["max_self_term"] = float(max(own))
        out["first_scene_positive"] = next((i for i, w in enumerate(world) if w > 0.0), None)
        return out

    def planner_spheres(self, configs: "list[list[float]]"):
        h = len(configs)
        state = self.call("MotionPlanner.compute_kinematics", self.planner.compute_kinematics,
                          self.JointState.from_position(self.q3(configs), joint_names=self.planner.joint_names))
        return state.robot_spheres.reshape(1, h, -1, 4)

    def by_planner_spheres(self, checker, configs: "list[list[float]]",
                           flags_out: "list[bool] | None" = None) -> dict:
        """validate's own sum, from collision_robot_scene.py, over the planner's spheres."""
        torch = self.torch
        h = len(configs)
        q = self.q3(configs)
        spheres = self.planner_spheres(configs)
        d_bound = self.call("RobotCollisionChecker.get_bound", checker.get_bound, q)
        d_self = self.call("RobotCollisionChecker.get_self_collision", checker.get_self_collision, spheres)
        d_world = self.call("RobotCollisionChecker.get_collision_constraint",
                            checker.get_collision_constraint, spheres)
        terms = torch.cat(
            [
                d_bound.reshape(1, h, -1).sum(dim=-1, keepdim=True),
                d_self.reshape(1, h, -1).sum(dim=-1, keepdim=True),
                d_world.reshape(1, h, -1).sum(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        mask = torch.sum(terms, dim=-1) == 0.0
        flags = [bool(v) for v in mask.reshape(-1).cpu().tolist()]
        if flags_out is not None:
            flags_out.extend(flags)
        out = _verdict(flags)
        out["max_scene_term"] = float(d_world.max())
        out["max_self_term"] = float(d_self.max())
        out["max_bound_term"] = float(d_bound.max())
        return out

    def all_three(self, configs: "list[list[float]]") -> dict:
        return {
            "validate": self.by_validate(self.checker, configs),
            "distance": self.by_distance(self.checker, configs),
            "planner_spheres": self.by_planner_spheres(self.checker, configs),
        }

    def tool_and_spheres(self, q: "list[float]"):
        """Tool position, WXYZ quaternion and the live spheres (radius above zero) at one configuration."""
        state = self.call("MotionPlanner.compute_kinematics", self.planner.compute_kinematics,
                          self.JointState.from_position(self.q3([q]), joint_names=self.planner.joint_names))
        pose = state.tool_poses.get_link_pose(self.planner.tool_frames[0])
        pos = np.asarray(pose.position.reshape(-1)[:3].cpu().tolist(), dtype=np.float64)
        quat = np.asarray(pose.quaternion.reshape(-1)[:4].cpu().tolist(), dtype=np.float64)
        spheres = np.asarray(state.robot_spheres.reshape(-1, 4).cpu().tolist(), dtype=np.float64)
        return pos, quat, spheres[spheres[:, 3] > 0.0]

    def within_limits(self, configs: "list[list[float]]", slack: float = 0.05) -> bool:
        return all(lo + slack <= v <= hi - slack
                   for c in configs for v, lo, hi in zip(c, self.joint_low, self.joint_high))

    def default_activation_checker(self):
        """A checker left at the library default activation distance (0.2 m), for comparison only."""
        if getattr(self, "_default_checker", None) is None:
            self._default_checker = self.make_checker(self.robot_in_use)
        return self._default_checker

    # a plan, before and after, to show the planner still works

    def plan_once(self) -> dict:
        from curobo.types import GoalToolPose  # type: ignore[import-not-found]

        torch = self.torch
        q0 = self.default_q
        pos, quat, _spheres = self.tool_and_spheres(q0)
        goal_pos = [float(pos[0]) + 0.04, float(pos[1]), float(pos[2])]
        start = self.JointState.from_position(
            torch.tensor(q0, device="cuda", dtype=torch.float32).unsqueeze(0),
            joint_names=self.planner.joint_names)
        # Shaped as the sidecar's plan branch in curobo_planner_server.py shapes them.
        goal = GoalToolPose(
            tool_frames=self.planner.tool_frames,
            position=torch.tensor([[[[goal_pos]]]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([[[[quat.tolist()]]]], device="cuda", dtype=torch.float32),
        )
        started = time.perf_counter()
        result = self.call("MotionPlanner.plan_pose", self.planner.plan_pose, goal, start,
                           max_attempts=PLAN_MAX_ATTEMPTS, enable_graph_attempt=PLAN_GRAPH_FROM)
        took_ms = (time.perf_counter() - started) * 1000.0
        ok = result is not None and bool(result.success.any())
        out = {"success": ok, "plan_ms": round(took_ms, 1),
               "tool_position_at_default_q_m": [round(float(v), 6) for v in pos]}
        if ok:
            plan = self.call("TrajOptSolverResult.get_interpolated_plan", result.get_interpolated_plan)
            traj = plan.position.reshape(-1, len(self.planner.joint_names)).cpu().tolist()
            end_pos, _q, _s = self.tool_and_spheres(traj[-1])
            out["waypoints"] = len(traj)
            out["final_tool_error_mm"] = round(float(np.linalg.norm(end_pos - np.asarray(goal_pos))) * 1000.0, 3)
        return out

    # 1 and 2: a path through a wall, a clear path, and a world that moves

    def paths_and_world(self) -> dict:
        q0 = self.default_q
        pos, _quat, _spheres = self.tool_and_spheres(q0)
        theta = math.atan2(pos[1], pos[0])
        radius = math.hypot(pos[0], pos[1])

        def swing(offset: float) -> "list[list[float]]":
            start, end = list(q0), list(q0)
            start[0] += offset - PAN_AMPLITUDE_RAD
            end[0] += offset + PAN_AMPLITUDE_RAD
            start[5] -= WRIST_SWING_RAD
            end[5] += WRIST_SWING_RAD
            return _interpolate(start, end, PATH_SAMPLES)

        through = swing(0.0)
        # The clear path swings on the far side of the base, as far round as the joint limits allow and
        # never closer to the wall than one pan amplitude.
        clear_offset = next((c for c in (math.pi, -math.pi, 2.4, -2.4, 2.0, -2.0, 1.8, -1.8)
                             if self.within_limits(swing(c))), None)
        if clear_offset is None:
            raise RuntimeError("no pan offset keeps the clear path inside the joint limits")
        clear = swing(clear_offset)

        def wall_at(azimuth: float) -> dict:
            # A radial slab across the pan sweep: 200 mm radial, 20 mm thick, 400 mm tall, centred where
            # the tool sits at the middle sample of the path.
            return {"wall": {"dims": [0.20, 0.02, 0.40],
                             "pose": [radius * math.cos(azimuth), radius * math.sin(azimuth), float(pos[2]),
                                      *_quat_about_z(azimuth)]}}

        self.through, self.clear = through, clear
        self.wall_a = wall_at(theta)
        wall_b = wall_at(theta + clear_offset)
        out: dict = {
            "tool_position_m": [round(float(v), 4) for v in pos],
            "wall_azimuth_rad": round(theta, 4),
            "wall_radius_m": round(radius, 4),
            "clear_offset_rad": round(clear_offset, 4),
            "path_samples": PATH_SAMPLES,
            "path_layout": "[batch=1, horizon=41, dof=6]",
            "through_path_within_limits": self.within_limits(through),
            "worlds": {},
        }
        sequence = (
            ("control_wall_far", FAR_WORLD),
            ("A_wall_on_through_path", self.wall_a),
            ("B_wall_moved_onto_clear_path", wall_b),
            ("A_again", self.wall_a),
        )
        for label, world in sequence:
            self.set_world(world)
            out["worlds"][label] = {"world": world, "through": self.all_three(through),
                                    "clear": self.all_three(clear)}
            if label == "A_wall_on_through_path":
                default = self.default_activation_checker()
                out["default_activation_checker_on_A"] = {
                    "through": {"validate": self.by_validate(default, through),
                                "distance": self.by_distance(default, through)},
                    "clear": {"validate": self.by_validate(default, clear),
                              "distance": self.by_distance(default, clear)},
                }
                out["batch_layout_on_A"] = {"through": self.by_validate_batch_layout(self.checker, through),
                                            "clear": self.by_validate_batch_layout(self.checker, clear)}

        def fits(v: dict, valid: bool) -> bool:
            if valid:
                return bool(v["valid"])
            return (not v["valid"]) and 0 < int(v["first_invalid"]) < PATH_SAMPLES - 1

        w = out["worlds"]
        holds = {}
        for m in ("validate", "distance", "planner_spheres"):
            check1 = (fits(w["control_wall_far"]["through"][m], True)
                      and fits(w["control_wall_far"]["clear"][m], True)
                      and fits(w["A_wall_on_through_path"]["through"][m], False)
                      and fits(w["A_wall_on_through_path"]["clear"][m], True))
            check2 = (fits(w["B_wall_moved_onto_clear_path"]["through"][m], True)
                      and fits(w["B_wall_moved_onto_clear_path"]["clear"][m], False)
                      and fits(w["A_again"]["through"][m], False)
                      and fits(w["A_again"]["clear"][m], True)
                      and w["A_again"]["through"][m]["first_invalid"]
                      == w["A_wall_on_through_path"]["through"][m]["first_invalid"])
            holds[m] = {"check1_paths": check1, "check2_world_follows": check2}
        out["holds"] = holds
        return out

    # caveat 2: what "valid" means

    def activation(self) -> dict:
        q0 = self.default_q
        _pos, _quat, spheres = self.tool_and_spheres(q0)
        reach = spheres[:, 0] + spheres[:, 3]
        i = int(np.argmax(reach))
        top = float(reach[i])
        cy, cz = float(spheres[i, 1]), float(spheres[i, 2])
        default = self.default_activation_checker()

        def scalar(x) -> float:
            return float(x.reshape(-1)[0]) if hasattr(x, "reshape") else float(x)

        out: dict = {
            "configuration": "default_q",
            "wall": "axis-aligned slab 50 x 1200 x 1200 mm, its -x face placed gap_mm beyond the arm's +x extent",
            "configured_activation_distance_m": {
                "checker_collision_constraint": scalar(self.checker.collision_constraint.config.activation_distance),
                "checker_collision_cost": scalar(self.checker.collision_cost.config.activation_distance),
                "default_checker_collision_constraint": scalar(default.collision_constraint.config.activation_distance),
                "default_checker_collision_cost": scalar(default.collision_cost.config.activation_distance),
            },
            "sweep": [],
        }
        for gap in GAPS_MM:
            dims = [0.05, 1.2, 1.2]
            centre = [top + gap / 1000.0 + 0.025, cy, cz]
            self.set_world({"wall": {"dims": dims, "pose": [*centre, 1, 0, 0, 0]}})
            views = self.all_three([q0])
            row: dict[str, Any] = {
                "gap_mm": gap,
                "sphere_clearance_mm": round(_min_clearance_to_aabb_mm(spheres, centre, dims), 3),
            }
            for name, v in views.items():
                # validate answers a mask only; the two other ways also say how large the scene term was.
                row[name] = {"valid": v["valid"], "scene_term": v.get("max_scene_term")}
            row["default_checker_validate_valid"] = self.by_validate(default, [q0])["valid"]
            row["default_checker_distance_scene_term"] = self.by_distance(default, [q0])["max_scene_term"]
            out["sweep"].append(row)
        sweep = out["sweep"]
        out["holds"] = {
            "validate_valid_exactly_when_not_penetrating": all(r["validate"]["valid"] == (r["gap_mm"] > 0) for r in sweep),
            "validate_valid_at_3mm": next(r["validate"]["valid"] for r in sweep if r["gap_mm"] == 3.0),
            "validate_valid_at_0.5mm": next(r["validate"]["valid"] for r in sweep if r["gap_mm"] == 0.5),
            "default_checker_validate_same_as_checker": all(
                r["default_checker_validate_valid"] == r["validate"]["valid"] for r in sweep),
        }
        return out

    # 3 and caveat 1: a carried part

    def payload_views(self, q: "list[float]") -> dict:
        views = self.all_three([q])
        _p, _o, live = self.tool_and_spheres(q)
        views["planner_live_spheres"] = int(live.shape[0])
        state = self.call("RobotCollisionChecker.get_kinematics", self.checker.get_kinematics, self.q3([q]))
        radii = state.robot_spheres.reshape(-1, 4)[:, 3]
        views["checker_live_spheres"] = int((radii > 0.0).sum().item())
        return views

    def payload(self) -> dict:
        from curobo._src.geom.types import Cuboid  # type: ignore[import-not-found]

        from _curobo_attach import ATTACHED_LINK_NAME  # type: ignore[import-not-found]

        torch = self.torch
        q0 = self.default_q
        pos, quat, spheres = self.tool_and_spheres(q0)
        axis = _quat_wxyz_to_matrix(quat)[:, 2]
        # Every sphere of the arm and hand ends this far along the tool axis; the wall starts 30 mm
        # beyond that, so the empty hand is clear and only a part hanging from the flange reaches it.
        reach = float(np.max((spheres[:, :3] - pos) @ axis + spheres[:, 3]))
        face = reach + 0.03
        centre = pos + axis * (face + 0.05)
        wall = {"wall": {"dims": [0.10, 0.5, 0.5], "pose": [*centre.tolist(), *_quat_x_onto(axis)]}}
        length = face + 0.05
        out: dict = {
            "tool_axis": [round(float(v), 4) for v in axis],
            "arm_reach_along_tool_axis_mm": round(reach * 1000.0, 2),
            "wall_face_beyond_flange_mm": round(face * 1000.0, 2),
            "payload_dims_mm": [40.0, 40.0, round(length * 1000.0, 2)],
            "payload_pose_tool_frame": [0.0, 0.0, round(length / 2.0, 4), 1, 0, 0, 0],
            "attach_sphere_slots": self.attach_spheres,
        }
        self.set_world(wall)
        out["before_attach"] = self.payload_views(q0)
        # As the sidecar's attach branch in curobo_planner_server.py does it.
        state = self.JointState.from_position(
            torch.tensor(q0, device="cuda", dtype=torch.float32).unsqueeze(0),
            joint_names=self.planner.joint_names)
        box = Cuboid(name="payload", pose=[0.0, 0.0, length / 2.0, 1, 0, 0, 0], dims=[0.04, 0.04, length])
        self.call("MotionPlanner.attachment_manager.attach", self.planner.attachment_manager.attach,
                  state, [box], link_name=ATTACHED_LINK_NAME, num_spheres=self.attach_spheres)
        try:
            out["after_attach"] = self.payload_views(q0)
            self.set_world(FAR_WORLD)
            out["after_attach_wall_far"] = self.payload_views(q0)
            self.set_world(wall)
        finally:
            self.call("MotionPlanner.attachment_manager.detach", self.planner.attachment_manager.detach,
                      link_name=ATTACHED_LINK_NAME)
        out["after_detach"] = self.payload_views(q0)
        b, a, f, d = out["before_attach"], out["after_attach"], out["after_attach_wall_far"], out["after_detach"]
        out["holds"] = {
            "separate_checker_validate_sees_payload": b["validate"]["valid"] and not a["validate"]["valid"],
            "separate_checker_distance_sees_payload": b["distance"]["valid"] and not a["distance"]["valid"],
            "planner_spheres_sees_payload": (b["planner_spheres"]["valid"] and not a["planner_spheres"]["valid"]
                                             and f["planner_spheres"]["valid"] and d["planner_spheres"]["valid"]),
        }
        return out

    # caveat 3: which self-collision config the checker judges with

    def self_config(self) -> dict:
        raw_checker = self.make_checker(self.raw_descriptor, collision_activation_distance=0.0)
        cfgs = {
            "checker": self.call("Kinematics.get_self_collision_config",
                                 self.checker.kinematics.get_self_collision_config),
            "planner": self.planner.kinematics.get_self_collision_config(),
            "raw_checker": raw_checker.kinematics.get_self_collision_config(),
        }

        def padding(cfg) -> "list[float]":
            if cfg.sphere_padding is None:
                return []
            return [round(float(v), 6) for v in cfg.sphere_padding.reshape(-1).cpu().tolist()]

        pads = {k: padding(v) for k, v in cfgs.items()}
        n = min(len(pads["checker"]), len(pads["raw_checker"]))
        out: dict = {
            "robot_in_use": self.robot_in_use_label,
            "raw_descriptor": self.raw_descriptor,
            "sphere_count": {k: int(v.num_spheres) for k, v in cfgs.items()},
            "collision_pairs": {k: int(v.collision_pairs.shape[0]) for k, v in cfgs.items()},
            "checker_padding_equals_planner_padding": pads["checker"] == pads["planner"],
            "distinct_padding_m": {k: sorted(set(v)) for k, v in pads.items()},
            "padding_difference_checker_minus_raw_m": sorted(
                {round(a - b, 6) for a, b in zip(pads["checker"][:n], pads["raw_checker"][:n])}),
        }
        # Behaviour, not just tensors: the same sampled configurations judged by all three, with nothing
        # near the arm, so any disagreement is self-collision alone.
        self.set_world(FAR_WORLD)
        samples = self.call("RobotCollisionChecker.sample", self.checker.sample, SELF_SAMPLES, mask_valid=False)
        configs = samples.reshape(-1, len(self.planner.joint_names)).cpu().tolist()
        fc: "list[bool]" = []
        fr: "list[bool]" = []
        fs: "list[bool]" = []
        self.by_validate(self.checker, configs, flags_out=fc)
        self.by_validate(raw_checker, configs, flags_out=fr)
        self.by_planner_spheres(self.checker, configs, flags_out=fs)
        out["sampled_configurations"] = len(configs)
        out["valid_counts"] = {
            "both": sum(1 for a, b in zip(fc, fr) if a and b),
            "raw_only_refused_by_margin": sum(1 for a, b in zip(fc, fr) if b and not a),
            "checker_only": sum(1 for a, b in zip(fc, fr) if a and not b),
            "neither": sum(1 for a, b in zip(fc, fr) if not a and not b),
        }
        out["planner_spheres_disagree_with_checker"] = sum(1 for a, b in zip(fc, fs) if a != b)
        example = next((i for i, (a, b) in enumerate(zip(fc, fr)) if b and not a), None)
        if example is not None:
            q = configs[example]
            out["example_refused_only_with_margin"] = {
                "q": [round(v, 4) for v in q],
                "checker_distance": self.by_distance(self.checker, [q]),
                "raw_checker_distance": self.by_distance(raw_checker, [q]),
            }
        out["holds"] = {
            "checker_self_config_matches_planner": pads["checker"] == pads["planner"],
            "checker_self_config_is_not_raw": out["valid_counts"]["raw_only_refused_by_margin"] > 0,
            "nothing_valid_only_with_margin": out["valid_counts"]["checker_only"] == 0,
            "planner_spheres_agree_on_samples": out["planner_spheres_disagree_with_checker"] == 0,
        }
        return out

    # 4: timings

    def timings(self) -> dict:
        torch = self.torch
        if not self.through:
            raise RuntimeError("the path section did not run, so there is no path to time")
        self.set_world(self.wall_a)
        start, end = self.through[0], self.through[-1]
        methods = (
            ("validate [1,N,dof]", lambda c: self.by_validate(self.checker, c)),
            ("validate [N,1,dof]", lambda c: self.by_validate_batch_layout(self.checker, c)),
            ("distance [1,N,dof]", lambda c: self.by_distance(self.checker, c)),
            ("planner_spheres [1,N,dof]", lambda c: self.by_planner_spheres(self.checker, c)),
        )

        def timed(fn, configs) -> "tuple[float, dict]":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            verdict = fn(configs)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0, verdict

        rows = []
        for n in BATCH_SIZES:
            configs = _interpolate(start, end, n)
            for name, fn in methods:
                first_ms, _v = timed(fn, configs)
                for _ in range(5):
                    fn(configs)
                samples = []
                verdict: dict = {}
                for _ in range(self.repeats):
                    ms, verdict = timed(fn, configs)
                    samples.append(ms)
                rows.append({"n": n, "method": name, "first_call_ms": round(first_ms, 3), **_stats_ms(samples),
                             "valid": verdict["valid"], "first_invalid": verdict["first_invalid"],
                             "invalid_count": verdict["invalid_count"]})
        # Requests of changing size: every call here sees a different N from the call before it, which
        # is what a sidecar serving paths of different lengths will see.
        cached = {n: _interpolate(start, end, n) for n in BATCH_SIZES}
        alternating: "dict[int, list[float]]" = {n: [] for n in BATCH_SIZES}
        for _ in range(10):
            for n in BATCH_SIZES:
                ms, _v = timed(lambda c: self.by_validate(self.checker, c), cached[n])
                alternating[n].append(ms)
        return {
            "world": "A_wall_on_through_path",
            "configurations": "the through path resampled to N (N = 1 is its midpoint)",
            "timer": "perf_counter from python lists in to verdict out, cuda synchronized on both sides",
            "warmup_calls_per_row": 5,
            "rows": rows,
            "alternating_sizes_validate_1_N_dof": {str(n): _stats_ms(v) for n, v in alternating.items()},
            "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        }

    # 8: the planner still plans

    def plan_after(self) -> dict:
        self.set_world(dict(BOOT_WORLD))
        after = self.plan_once()
        before = self.result["sections"].get("build", {}).get("plan_before", {})
        shift = None
        if before.get("tool_position_at_default_q_m") and after.get("tool_position_at_default_q_m"):
            shift = round(max(abs(a - b) for a, b in zip(before["tool_position_at_default_q_m"],
                                                       after["tool_position_at_default_q_m"])) * 1000.0, 6)
        return {
            "world": "boot table restored",
            "plan_before": before,
            "plan_after": after,
            "tool_position_shift_mm": shift,
            "holds": {"planner_still_plans": bool(before.get("success")) and bool(after.get("success")),
                      "fk_unchanged": shift is not None and shift < 1e-3},
        }

    # the gate

    def decide(self) -> dict:
        s = self.result["sections"]
        paths = s.get("paths_and_world", {}).get("holds", {})
        payload = s.get("payload", {}).get("holds", {})
        after = s.get("plan_after", {}).get("holds", {})

        def holds(method: str) -> bool:
            row = paths.get(method, {})
            return bool(row.get("check1_paths")) and bool(row.get("check2_world_follows"))

        inputs = {
            "validate_checks_1_and_2": holds("validate"),
            "distance_checks_1_and_2": holds("distance"),
            "planner_spheres_checks_1_and_2": holds("planner_spheres"),
            "separate_checker_sees_payload": bool(payload.get("separate_checker_validate_sees_payload")),
            "planner_spheres_sees_payload": bool(payload.get("planner_spheres_sees_payload")),
            "planner_still_plans_after_probe": bool(after.get("planner_still_plans")),
        }
        if inputs["validate_checks_1_and_2"] and inputs["separate_checker_sees_payload"]:
            choice = "proven_names"
            text = ("3b uses the proven names: RobotCollisionChecker.validate on a checker built by "
                    "RobotCollisionCheckerCfg.load_from_config with the robot in use and the planner's "
                    "scene_collision_checker.")
        elif (inputs["planner_spheres_checks_1_and_2"] and inputs["planner_spheres_sees_payload"]
              and inputs["planner_still_plans_after_probe"]):
            choice = "planner_sphere_variant"
            text = ("3b uses the planner-sphere variant: the separately built checker is blind to an "
                    "attached payload, so the spheres come from MotionPlanner.compute_kinematics and are "
                    "judged by the checker's get_bound, get_self_collision and get_collision_constraint.")
        else:
            choice = "q2_refuse"
            text = "No batch query works: the owner's Q2 applies and the path verbs refuse on curobo cells."
        return {"choice": choice, "decision": text, "inputs": inputs}

    def summary(self) -> str:
        s = self.result["sections"]
        b = s.get("build", {})
        lines = [f"model {self.model}, margin {self.margin_mm:g} mm, cuRobo {b.get('curobo_version')}, "
                 f"robot in use {b.get('robot_in_use')}"]

        def word(v: dict) -> str:
            return "valid" if v.get("valid") else f"INVALID from {v.get('first_invalid')}"

        for world, row in s.get("paths_and_world", {}).get("worlds", {}).items():
            for path in ("through", "clear"):
                cells = ", ".join(f"{m} {word(v)}" for m, v in row[path].items())
                lines.append(f"  {world:30s} {path:8s} {cells}")
        pay = s.get("payload", {})
        if "after_attach" in pay:
            lines.append("  payload attached: " + ", ".join(
                f"{m} {word(pay['after_attach'][m])}" for m in ("validate", "distance", "planner_spheres"))
                + f"; live spheres planner {pay['after_attach']['planner_live_spheres']}, "
                  f"checker {pay['after_attach']['checker_live_spheres']}")
        for row in s.get("activation", {}).get("sweep", []):
            lines.append(f"  gap {row['gap_mm']:+5.1f} mm (spheres {row['sphere_clearance_mm']:+.3f} mm): "
                         f"validate {row['validate']['valid']}, planner_spheres {row['planner_spheres']['valid']}, "
                         f"distance scene term {row['distance']['scene_term']:.6f}")
        sc = s.get("self_config", {})
        if "valid_counts" in sc:
            lines.append(f"  self config: padding equals planner {sc['checker_padding_equals_planner_padding']}, "
                         f"checker minus raw {sc['padding_difference_checker_minus_raw_m']}, "
                         f"valid counts {sc['valid_counts']}")
        for row in s.get("timings", {}).get("rows", []):
            lines.append(f"  N={row['n']:4d} {row['method']:26s} median {row['median_ms']:8.3f} ms  "
                         f"p90 {row['p90_ms']:8.3f}  first {row['first_call_ms']:9.3f}")
        pa = s.get("plan_after", {})
        if "holds" in pa:
            lines.append(f"  plan after the probe: {pa['holds']}")
        lines.append(f"  decision: {self.result.get('decision', {}).get('choice')}")
        for f in self.failures:
            lines.append(f"  FAILURE {f}")
        return "\n".join(lines)


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        prog="probe_batch_check",
        description="Prove the batch collision query names and verdicts on the planner the sidecar builds.",
    )
    parser.add_argument("model", choices=sorted(CONFIG_MARGIN_MM))
    parser.add_argument("--margin-mm", type=float, default=None,
                        help="guard margin handed to the planner; defaults to the cell config figure")
    parser.add_argument("--attach-spheres", type=int, default=PAYLOAD_SPHERE_SLOTS,
                        help="payload sphere slots the planner is derived with (robot.yaml payload.sphere_slots)")
    parser.add_argument("--repeats", type=int, default=30, help="timed calls per batch size and method")
    parser.add_argument("--out", default=None, help="result file (default logs/curobo/batch_check_probe_<model>.json)")
    args = parser.parse_args(argv)

    margin = CONFIG_MARGIN_MM[args.model] if args.margin_mm is None else float(args.margin_mm)
    probe = Probe(args.model, margin, args.attach_spheres, args.repeats)
    started = time.perf_counter()
    if probe.section("build", probe.build):
        for name, fn in (
            ("paths_and_world", probe.paths_and_world),
            ("activation", probe.activation),
            ("payload", probe.payload),
            ("self_config", probe.self_config),
            ("timings", probe.timings),
            ("plan_after", probe.plan_after),
        ):
            probe.section(name, fn)
    probe.result["decision"] = probe.decide()
    probe.result["proven_calls"] = list(probe.proven)
    probe.result["failures"] = list(probe.failures)
    probe.result["wall_time_s"] = round(time.perf_counter() - started, 2)

    out = Path(args.out) if args.out else REPO / "logs" / "curobo" / f"batch_check_probe_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(probe.result, indent=2, default=str), encoding="utf-8")
    print(probe.summary(), flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if not probe.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
