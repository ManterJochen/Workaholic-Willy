"""The arm descriptor with the hand as a body link, against the per hand file it replaces.

    ext_deps/curobo_env/python.exe scripts/curobo/probe_hand_link_equivalence.py ur5e robotiq_hande \
        --coupling-mm 20 --planner-margin-mm 10 --out logs/b3/equivalence_ur5e_robotiq_hande.json

A cell used to load ``{arm}_{hand}.yml``, the hand's spheres written onto ``tool0`` by the builder. It loads
``willy_{arm}.yml`` instead, and the sidecar adds the hand as a fixed link under tool0 when it starts. Both have to be
the same robot, so this builds both and compares, on the pose set every gate judges (``_matrix_gate.pose_set``):

* the arm's own spheres, link by link, as numbers in the file;
* ``default_q``, the retract both chains start from;
* the self collision pair table, as index pairs, and the sphere padding;
* every sphere's world position over the poses, which is where a wrong placement shows (1e-6 m);
* the check_js verdict per pose: the bound, self collision and world terms the sidecar sums, over the planner's own
  boot table world.

The control is ``--control-plate-sign``, which puts the Hand-E's plate on the wrong side of the flange. A probe that
reports agreement then is not measuring the placement and the run is worthless. Verdict equality is the gate rather
than bitwise equality, because float32 forward kinematics composes the fixed transform on the GPU.

Exit code is 0 when everything agrees, 1 otherwise. cuRobo environment and GPU only.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch  # type: ignore[import-not-found]
from curobo._src.util.config_io import resolve_config  # type: ignore[import-not-found]
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg  # type: ignore[import-not-found]
from curobo.content import get_content_root  # type: ignore[import-not-found]
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # type: ignore[import-not-found]
from curobo.types import JointState  # type: ignore[import-not-found]

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hand_body import SIM_TOOL_ROTATION_XYZW, hand_body  # noqa: E402
from _matrix_gate import pose_set  # noqa: E402

CONTENT = Path(str(get_content_root()))
#: The world the sidecar boots with, so both chains are judged against the same scene.
TABLE = {"table": {"dims": [1.6, 1.6, 0.05], "pose": [0.0, 0.0, -0.026, 1, 0, 0, 0]}}
ARM_LINKS = ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")


def _compose(config: dict, bodies: list, margin_mm: float) -> dict:
    from _curobo_body_links import compose_sidecar_config  # type: ignore[import-not-found]

    return compose_sidecar_config(config, bodies=bodies, margin_mm=margin_mm)


def _build(config: dict) -> "tuple[Any, Any, float]":
    started = time.perf_counter()
    planner = MotionPlanner(MotionPlannerCfg.create(
        robot=copy.deepcopy(config), scene_model={"cuboid": TABLE}, collision_cache={"cuboid": 16}))
    checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=copy.deepcopy(config), scene_collision_checker=planner.scene_collision_checker,
        collision_activation_distance=0.0))
    return planner, checker, round(time.perf_counter() - started, 2)


def _judge(planner: Any, checker: Any, poses: "list[list[float]]") -> dict:
    """The sidecar's check_js, term by term, plus every sphere's world position."""
    q = torch.tensor(np.asarray(poses, dtype=np.float32), device="cuda").unsqueeze(0)
    horizon = int(q.shape[1])
    spheres = planner.compute_kinematics(
        JointState.from_position(q, joint_names=planner.joint_names)).robot_spheres.reshape(1, horizon, -1, 4)
    bound = checker.get_bound(q).reshape(1, horizon, -1).sum(dim=-1)
    self_term = checker.get_self_collision(spheres).reshape(1, horizon, -1).sum(dim=-1)
    world = checker.get_collision_constraint(spheres).reshape(1, horizon, -1).sum(dim=-1)
    config = checker.kinematics.get_self_collision_config()
    return {
        "valid": ((bound + self_term + world) == 0.0).reshape(-1).cpu().numpy(),
        "self": self_term.reshape(-1).cpu().numpy(),
        "spheres": spheres.reshape(horizon, -1, 4).cpu().numpy(),
        "joint_names": list(planner.joint_names),
        "default_q": [round(float(v), 6) for v in planner.default_joint_state.position.reshape(-1).cpu().tolist()],
        "pairs": sorted(tuple(sorted(pair)) for pair in config.collision_pairs.cpu().numpy().tolist()),
        "padding": None if config.sphere_padding is None else config.sphere_padding.reshape(-1).cpu().numpy().tolist(),
    }


def _arm_spheres(kinematics: dict) -> dict:
    return {link: kinematics["collision_spheres"].get(link) for link in ARM_LINKS}


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="The hand as a body link against the per hand descriptor it replaces.")
    parser.add_argument("arm")
    parser.add_argument("hand")
    parser.add_argument("--legacy", type=Path, default=None,
                        help="the per hand descriptor to compare against; s13_legacy_{arm}_{hand}.yml by default")
    parser.add_argument("--coupling-mm", type=float, default=None)
    parser.add_argument("--planner-margin-mm", type=float, default=10.0)
    parser.add_argument("--tool-rotation-xyzw", type=float, nargs=4, default=list(SIM_TOOL_ROTATION_XYZW))
    parser.add_argument("--control-plate-sign", action="store_true",
                        help="the control: put the plate on the wrong side of the flange; the probe must then disagree")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    legacy_path = args.legacy or CONTENT / "configs" / "robot" / f"s13_legacy_{args.arm}_{args.hand}.yml"
    arm_path = CONTENT / "configs" / "robot" / f"willy_{args.arm}.yml"
    for path in (legacy_path, arm_path):
        if not path.is_file():
            print(f"no descriptor at {path}; nothing compared")
            return 2

    body = hand_body(args.hand, coupling_mm=args.coupling_mm, rotation_xyzw=tuple(args.tool_rotation_xyzw))
    if args.control_plate_sign:
        body = {**body, "fixed_transform": [-v for v in body["fixed_transform"][:3]] + list(body["fixed_transform"][3:])}

    legacy_config = _compose(resolve_config(str(legacy_path)), [], args.planner_margin_mm)
    composed = _compose(resolve_config(str(arm_path)), [body], args.planner_margin_mm)

    legacy_planner, legacy_checker, legacy_seconds = _build(legacy_config)
    composed_planner, composed_checker, composed_seconds = _build(composed)

    retract = [float(v) for v in legacy_planner.default_joint_state.position.reshape(-1).cpu().tolist()]
    poses, kinds = pose_set(retract)
    old = _judge(legacy_planner, legacy_checker, poses)
    new = _judge(composed_planner, composed_checker, poses)

    arm_equal = _arm_spheres(legacy_config["robot_cfg"]["kinematics"]) == _arm_spheres(composed["robot_cfg"]["kinematics"])
    same_shape = old["spheres"].shape == new["spheres"].shape
    deviation = float(np.max(np.abs(old["spheres"][..., :3] - new["spheres"][..., :3]))) if same_shape else None
    radius_deviation = float(np.max(np.abs(old["spheres"][..., 3] - new["spheres"][..., 3]))) if same_shape else None
    disagree = [int(i) for i in np.nonzero(old["valid"] != new["valid"])[0]]

    report = {
        "arm": args.arm, "hand": args.hand, "poses": len(poses),
        "legacy_file": legacy_path.name, "arm_file": arm_path.name,
        "planner_margin_mm": args.planner_margin_mm, "coupling_mm": args.coupling_mm,
        "placement": body["fixed_transform"], "control_plate_sign": bool(args.control_plate_sign),
        "arm_spheres_equal": arm_equal,
        "default_q_equal": old["default_q"] == new["default_q"], "default_q": old["default_q"],
        "joint_names_equal": old["joint_names"] == new["joint_names"],
        "sphere_count": [None if not same_shape else int(old["spheres"].shape[1]),
                         None if not same_shape else int(new["spheres"].shape[1])],
        "sphere_position_max_diff_m": deviation, "sphere_radius_max_diff_m": radius_deviation,
        "pairs_equal": old["pairs"] == new["pairs"], "pair_count": [len(old["pairs"]), len(new["pairs"])],
        "padding_equal": old["padding"] == new["padding"],
        "valid_poses": [int(old["valid"].sum()), int(new["valid"].sum())],
        "verdict_disagreements": len(disagree), "first_disagreements": disagree[:10],
        "disagreement_kinds": sorted({kinds[i] for i in disagree}),
        "self_term_max_diff": float(np.max(np.abs(old["self"] - new["self"]))),
        "startup_seconds": {"legacy": legacy_seconds, "arm_plus_hand": composed_seconds},
    }
    print(json.dumps(report, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    agrees = bool(
        arm_equal and report["default_q_equal"] and report["joint_names_equal"] and report["pairs_equal"]
        and same_shape and (deviation or 0.0) <= 1e-6 and not disagree
    )
    return 0 if agrees else 1


if __name__ == "__main__":
    raise SystemExit(main())
