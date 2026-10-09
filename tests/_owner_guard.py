"""The owner's exact guard as the cell ran it on 2026-10-08, built offline, for the tests of the whole-path judge.

The UR10, the Hand-E on its 23 mm plate with the PolyScope tool frame, the D415 in its printed enclosure on the wrist
(``config/cameras/d415_enclosure.yaml``, 3 mm of margin, placed by the recorded frames' own camera and tool poses), 3 mm
from itself, from a declared fixture and from a box a camera saw, every joint within half a turn of home and the
cell's payload: every guard a joint move asks (``SafetyPreflight._JOINT_MOVE_SKIP_GUARDS`` skips the others).
``whole_path_judge`` is the switch the tests turn.

The worlds are the three cell frames of 2026-10-07 (``tests/test_a_faster_camera_world_builds_the_same_world.py``),
handed to the guard as the arm's refresh hands them (``live_world._guard_boxes``). Solutions come from the repository's
closed form (``_ur_ik.ur_flange_ik``) nearest the previous one, through the declared tool frame: the cell solves them
on the controller, and only the guard is judged here.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

#: The owner's home and LOOK[0] of 2026-10-08, degrees (``robot.home_joint_positions_deg``).
HOME_DEG = (-75.10, -83.00, -72.40, -137.31, 96.02, -76.28)

#: The cell's guard distances, millimetres.
GUARD_MM = 3.0


def needs_the_engine() -> None:
    """Skip where the exact mesh backend cannot run: no Coal and no python-fcl, or no UR10 or Hand-E bundle."""
    import unittest

    from src.robot.safety._fcl_self_collision import mesh_backend_status

    if mesh_backend_status("ur10", mesh_name="robotiq_hande") != "ok":
        raise unittest.SkipTest("no exact mesh backend on this box")


def owner_robot(*, whole: bool, fixtures: "list[dict[str, Any]] | None" = None) -> dict[str, Any]:
    """The owner's ``robot`` block as far as the guard reads it, ``whole_path_judge`` at ``whole``."""
    from tests import _cell_2026_10_01 as cell

    robot = cell.owner_robot(support_plane={"height_mm": 0.0, "extent_mm": [900.0, 700.0], "thickness_mm": 50.0,
                                            "center_mm": [-10.0, -585.0]})
    robot["workspace_limits"] = {"x_min": -470.0, "x_max": 400.0, "y_min": -900.0, "y_max": -200.0, "z_min": 16.0,
                                 "z_max": 800.0}
    robot["home_joint_positions_deg"] = list(HOME_DEG)
    safety = robot["safety"]
    safety["joint_limits"] = {"enforce": True, "margin_deg": 5.0, "within_half_turn_of_home": True}
    safety["payload"] = {"enforce": True, "mass_kg": 1.3, "max_mass_kg": 10.0, "cog_mm": [0.0, 0.0, 65.0]}
    safety["self_collision"].update(min_distance_mm=GUARD_MM, perceived_min_distance_mm=GUARD_MM, planner_margin_mm=0.0,
                                    whole_path_judge=bool(whole), fixtures=list(fixtures or []))
    return robot


def tool() -> np.ndarray:
    """The owner's tool frame, FLANGE to TCP, millimetres: 157 mm out and a quarter turn about the flange axis."""
    from src.robot.drivers.ur.tool_frame import tool_frame_matrix

    return np.asarray(tool_frame_matrix((0.0, 0.0, 157.0), (0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))),
                      dtype=np.float64)


@lru_cache(maxsize=1)
def camera_body() -> Any:
    """The D415 in its enclosure where the cell's frames place it (frame F1's camera and tool poses)."""
    from src.calibration.serialization import FlangeToTcp
    from src.config.cameras import load_camera
    from src.geometry import Frame, Transform
    from src.robot.safety.planning.body_link import WristBody
    from tests.test_a_faster_camera_world_builds_the_same_world import frame

    recorded = frame("F1")
    camera_to_tool = Transform.from_matrix(np.linalg.inv(recorded["tool_to_base"]) @ recorded["camera_to_base"],
                                           from_frame=Frame.CAMERA, to_frame=Frame.TOOL)
    return WristBody.from_parts(rig_id="EIH_Cam", spec=load_camera("d415_enclosure"), bracket=None, margin_mm=3.0,
                                camera_to_tool=camera_to_tool, flange_to_tcp=FlangeToTcp.from_matrix("polyscope", tool()))


class OwnerArm:
    """What the guards read of the owner's arm: a UR10, its config and the home it drives to."""

    def __init__(self, config: Any) -> None:
        from src.robot.core.capabilities import RobotCapabilities

        self.capabilities = RobotCapabilities(vendor="ur", model="ur10", has_native_fk=True)
        self.config = config
        self.home_joint_positions = tuple(config.home_joint_positions)


def owner_cell(*, whole: bool, fixtures: "list[dict[str, Any]] | None" = None) -> "tuple[Any, Any]":
    """The owner's preflight, the camera handed in, and the arm it judges."""
    from src.config.schema.robot import RobotConfig
    from src.robot.safety import SafetyPreflight
    from src.robot.safety.planning.hand import planner_hand

    config = RobotConfig.model_validate(owner_robot(whole=whole, fixtures=fixtures))
    preflight = SafetyPreflight.from_safety_config(config.safety, config.workspace_limits, hand=planner_hand(config),
                                                   arm_model="ur10")
    preflight.set_wrist_bodies((camera_body(),))
    return preflight, OwnerArm(config)


def nearest_solution(tcp: np.ndarray, near: "np.ndarray | Any") -> "np.ndarray | None":
    """The closed-form joints that put the TCP at ``tcp``, nearest ``near`` joint by joint, unwrapped onto it."""
    from src.robot.safety._ur_ik import ur_flange_ik

    near = np.asarray(near, dtype=np.float64)
    found = ur_flange_ik("ur10", np.asarray(tcp, dtype=np.float64) @ np.linalg.inv(tool()), q6_if_singular=float(near[-1]))
    if not found:
        return None

    def apart(solution: Any) -> np.ndarray:
        return ((np.asarray(solution, dtype=np.float64) - near) + math.pi) % (2.0 * math.pi) - math.pi

    best = min(found, key=lambda solution: float(np.abs(apart(solution)).sum()))
    return near + apart(best)


def line_samples(top: np.ndarray, bottom: np.ndarray, near: Any, reach_mm: "tuple[float, ...]",
                 step_mm: float = GUARD_MM) -> Any:
    """The straight TCP line from ``top`` to ``bottom`` as the gate judges a commanded line: a solution every
    ``step_mm`` along it, the joint line between neighbours filled at the guard's step; ``None`` where one is out of
    reach."""
    from src.robot.safety.path_samples import PathSamples, joint_path_samples

    length = float(np.linalg.norm(np.asarray(top)[:3, 3] - np.asarray(bottom)[:3, 3]))
    steps = max(1, math.ceil(length / step_mm))
    previous = nearest_solution(top, near)
    if previous is None:
        return None
    configs: list[tuple[float, ...]] = [tuple(float(v) for v in previous)]
    for k in range(1, steps + 1):
        goal = np.array(top, dtype=np.float64, copy=True)
        goal[:3, 3] = np.asarray(top)[:3, 3] + (np.asarray(bottom)[:3, 3] - np.asarray(top)[:3, 3]) * k / steps
        following = nearest_solution(goal, previous)
        if following is None:
            return None
        fill = joint_path_samples(previous.tolist(), following.tolist(), reach_mm=reach_mm, max_step_mm=GUARD_MM)
        configs.extend(fill.configs[1:-1])
        configs.append(tuple(float(v) for v in following))
        previous = following
    return PathSamples(configs=tuple(configs), step_bound_mm=GUARD_MM)


def seen_world(name: str, case: str) -> "tuple[Any, ...]":
    """The boxes the cell's camera world of frame ``name`` builds in ``case``, as the guard holds them."""
    from src.robot.safety.planning.live_world import _guard_boxes
    from tests.test_a_faster_camera_world_builds_the_same_world import build

    return tuple(_guard_boxes(build(name, case).perceived))


def verdict(result: Any) -> Any:
    """What a gate's answer says: ``None``, or the status, the whole message and the joints it names."""
    if result is None:
        return None
    joints = None if result.target_joints is None else tuple(float(v) for v in result.target_joints.values)
    return result.status, result.message, joints
