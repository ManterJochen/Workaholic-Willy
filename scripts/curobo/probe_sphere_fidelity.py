"""Compare a UR descriptor's spheres with the exact meshes, on the poses exact_self_collision_poses.py judged.

    ext_deps/curobo_env/python.exe scripts/curobo/probe_sphere_fidelity.py ur5_robotiq_hande \
        --exact logs/u2/exact_ur5_robotiq_hande.json --planner-margin-mm 10 --guard-margin-mm 10 \
        --out logs/u2/fidelity_ur5_robotiq_hande.json

The descriptor argument names the pair the pose set was judged for. The file that is loaded is the arm's own,
``willy_{arm}.yml``, with the hand added as a body link where the declared tool frame puts it, which is what a cell's
planner loads. A pose set judged at another placement is refused rather than compared.

Two measurements, both against the committed geometry the guard checks:

* Per link: every arm sphere's reach past the convex hull of its own committed link mesh (``_arm_spheres.LinkHull``),
  carried into the bundle frame with the chain ``build_ur_config.py`` places fitted spheres with
  (``inv(urdf link frame) @ Rz(180) @ T_dh[frame]``). Worst reach, the count past 5 and past 10 mm, and whether the
  worst sphere is Isaac's (vendor) or fitted. An arm with no committed bundle has no links to judge.
* Per pose: the composed config's spheres, arm plus hand link plus planner margin, composed by the module the sidecar
  composes with (``_curobo_body_links``), judged with the self collision term of the sidecar's check_js
  (``curobo_planner_server.py``), against the exact distance at the guard margin, counted by
  ``_sphere_fidelity.Agreement`` over all poses and per kind (retract, random, sweep).

Joint limits are not judged, because a random pose outside them is a pose neither engine would be asked about and the
exact half does not know them. cuRobo environment only.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch  # type: ignore[import-not-found]
import yaml
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg  # type: ignore[import-not-found]
from curobo.content import get_content_root  # type: ignore[import-not-found]
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # type: ignore[import-not-found]
from curobo.types import JointState  # type: ignore[import-not-found]

REPO = Path(__file__).resolve().parents[2]
CONTENT = Path(str(get_content_root()))
ISAAC_MP = Path("D:/isaacsim/isaac-sim-standalone-5.1.0-windows-x86_64/exts/isaacsim.robot_motion.motion_generation/"
                "motion_policy_configs/universal_robots")
#: bundle key -> (URDF link, DH frame), the map build_ur_config.py and the bake use.
LINKS = {"shoulder": ("shoulder_link", 1), "upper_arm": ("upper_arm_link", 2), "forearm": ("forearm_link", 3),
         "wrist_1": ("wrist_1_link", 4), "wrist_2": ("wrist_2_link", 5), "wrist_3": ("wrist_3_link", 6)}


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


ARM = _load(REPO / "scripts" / "curobo" / "_arm_spheres.py")
FIDELITY = _load(REPO / "scripts" / "curobo" / "_sphere_fidelity.py")
#: The hand as a cell's planner sends it, and the composition its sidecar runs. Loading the same two modules is what
#: makes this probe measure the robot the planner builds, rather than a file that resembles it.
HAND = _load(REPO / "scripts" / "curobo" / "_hand_body.py")
COMPOSE = _load(REPO / "src" / "robot" / "safety" / "planning" / "_curobo_body_links.py")
#: The standard UR DH table as the URDF baker inlines it. Every inlined copy of it is held against the others.
UR_DH = _load(REPO / "scripts" / "isaac" / "bake_ur_meshes_from_urdf.py").UR_DH


def _axis_angle(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) or 1.0)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = a
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def _urdf_frames(text: str):
    import xml.etree.ElementTree as ET

    kids = {}
    for joint in ET.fromstring(text).findall("joint"):
        origin = joint.find("origin")
        rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()] if origin is not None else [0.0] * 3
        xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()] if origin is not None else [0.0] * 3
        m = np.eye(4)
        m[:3, :3] = _axis_angle((0, 0, 1), rpy[2]) @ _axis_angle((0, 1, 0), rpy[1]) @ _axis_angle((1, 0, 0), rpy[0])
        m[:3, 3] = xyz
        child_tag, parent_tag = joint.find("child"), joint.find("parent")
        if child_tag is None or parent_tag is None:
            raise ValueError(f"joint {joint.attrib.get('name', '?')!r} names no child or no parent link")
        kids[child_tag.attrib["link"]] = (parent_tag.attrib["link"], m)
    base = next(iter({parent for parent, _ in kids.values()} - set(kids)))

    def frame(link: str) -> np.ndarray:
        if link == base:
            return np.eye(4)
        parent, m = kids[link]
        return frame(parent) @ m
    return frame


def _dh_frames(model: str) -> list[np.ndarray]:
    frames, t = [np.eye(4)], np.eye(4)
    for a, d, alpha in UR_DH[model]:
        ca, sa = np.cos(alpha), np.sin(alpha)
        t = t @ np.array([[1.0, 0.0, 0.0, a], [0.0, ca, -sa, 0.0], [0.0, sa, ca, d], [0.0, 0.0, 0.0, 1.0]])
        frames.append(t.copy())
    return frames


def link_reach(name: str, kinematics: dict) -> dict[str, dict]:
    """Per arm link: how far its spheres reach past the hull of its committed mesh. Empty with no bundle."""
    model = name.split("_", 1)[0]
    bundle_path = REPO / "src" / "robot" / "safety" / "data" / f"{model}_collision_meshes.npz"
    if not bundle_path.is_file() or model not in UR_DH:
        return {}
    urdf = (CONTENT / "assets" / "robot" / "ur_description" / f"{model}.urdf").read_text(encoding="utf-8")
    frame = _urdf_frames(urdf)
    dh = _dh_frames(model)
    rz = np.diag([-1.0, -1.0, 1.0, 1.0])
    bundle = np.load(bundle_path)
    lula = yaml.safe_load((ISAAC_MP / model / "rmpflow" / f"{model}_robot_description.yaml").read_text(encoding="utf-8"))
    vendor = {link: {tuple(np.round(np.asarray(s["center"], float), 3)) for s in spheres}
              for entry in lula["collision_spheres"] for link, spheres in entry.items()}
    out: dict[str, dict] = {}
    for key, (link, index) in LINKS.items():
        hull = ARM.LinkHull.from_vertices(np.asarray(bundle[f"{key}__v"], dtype=np.float64) / 1000.0)
        back = np.linalg.inv(np.linalg.inv(frame(link)) @ rz @ dh[index])
        rows = []
        for sphere in kinematics["collision_spheres"].get(link, []):
            centre = np.asarray(sphere["center"], dtype=np.float64)
            local = back[:3, :3] @ centre + back[:3, 3]
            reach_mm = hull.reach_m(local, float(sphere["radius"])) * 1000.0
            tag = "vendor" if tuple(np.round(centre, 3)) in vendor.get(link, set()) else "fit"
            rows.append((reach_mm, tag))
        if rows:
            worst = max(rows)
            out[link] = {"spheres": len(rows), "worst_reach_mm": round(worst[0], 1), "worst_is": worst[1],
                         "past_5_mm": sum(1 for r, _ in rows if r > 5.0), "past_10_mm": sum(1 for r, _ in rows if r > 10.0)}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a UR descriptor's spheres with the exact meshes.")
    parser.add_argument("descriptor", help="{arm}_{hand}, the pair the pose set was judged for")
    parser.add_argument("--yml", type=Path, default=None,
                        help="an arm descriptor to measure instead of the willy_{arm}.yml the cuRobo content holds")
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--planner-margin-mm", type=float, required=True)
    parser.add_argument("--guard-margin-mm", type=float, required=True)
    parser.add_argument("--tool-rotation-xyzw", type=float, nargs=4, default=list(HAND.SIM_TOOL_ROTATION_XYZW),
                        help="where the declared tool frame puts the hand; it must match the pose set's placement")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    exact = json.loads(args.exact.read_text(encoding="utf-8"))
    arm, hand = args.descriptor.split("_", 1)
    if (exact["arm"], exact["hand"]) != (arm, hand):
        print(f"{args.exact} judged {exact['arm']} with {exact['hand']}, not {args.descriptor}; nothing compared")
        return 1
    placed = HAND.placement(tuple(args.tool_rotation_xyzw))
    judged = exact.get("placement")
    if judged is not None and [list(row) for row in placed.rotation] != [list(row) for row in judged["rotation"]]:
        print(f"{args.exact.name} judged the hand at {judged['approach']}{judged['closing']} and this run places it at "
              f"{placed.approach}{placed.closing}: the two halves would model different hands, nothing compared")
        return 1
    source = args.yml if args.yml is not None else CONTENT / "configs" / "robot" / f"willy_{arm}.yml"
    if not source.is_file():
        print(f"no arm descriptor at {source}; build it with build_ur_config.py {arm}")
        return 1

    # What the sidecar loads for this cell, by the same modules: the arm descriptor, the hand as a body link where the
    # declared frame puts it, then the planner margin.
    body = HAND.hand_body(hand, coupling_mm=exact.get("coupling_mm"), rotation_xyzw=tuple(args.tool_rotation_xyzw))
    composed, adjusted, _ = COMPOSE.compose_with_counts(
        yaml.safe_load(source.read_text(encoding="utf-8")), bodies=[body], margin_mm=args.planner_margin_mm,
    )
    planner = MotionPlanner(MotionPlannerCfg.create(robot=copy.deepcopy(composed), scene_model={"cuboid": {}}))
    checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=copy.deepcopy(composed), scene_collision_checker=planner.scene_collision_checker,
        collision_activation_distance=0.0))
    joints = torch.tensor(np.asarray(exact["poses"], dtype=np.float32), device="cuda").unsqueeze(0)
    horizon = int(joints.shape[1])
    spheres = planner.compute_kinematics(
        JointState.from_position(joints, joint_names=planner.joint_names)).robot_spheres.reshape(1, horizon, -1, 4)
    cost = checker.get_self_collision(spheres).reshape(1, horizon, -1).sum(dim=-1).reshape(-1).cpu().numpy()

    collides = [bool(c > 0.0) for c in cost]
    distances = [float(d) for d in exact["exact_distance_mm"]]
    kinds = exact["kinds"]
    overall = FIDELITY.Agreement.from_verdicts(
        sphere_collides=collides, exact_distance_mm=distances, margin_mm=args.guard_margin_mm)
    per_kind = {}
    for kind in dict.fromkeys(kinds):
        chosen = [i for i, k in enumerate(kinds) if k == kind]
        per_kind[kind] = FIDELITY.Agreement.from_verdicts(
            sphere_collides=[collides[i] for i in chosen], exact_distance_mm=[distances[i] for i in chosen],
            margin_mm=args.guard_margin_mm)
    reach = link_reach(args.descriptor, composed["robot_cfg"]["kinematics"])

    print(f"== {args.descriptor}: {source.name} with the hand as a body link at {placed.approach}{placed.closing}, "
          f"planner margin {args.planner_margin_mm:g} mm on {adjusted} link buffers, "
          f"guard margin {args.guard_margin_mm:g} mm")
    print(f"  retract: spheres {'collide' if collides[0] else 'clear'}, exact {distances[0]:.1f} mm "
          f"({exact['nearest_pair'][0]})")
    print(f"  all: {overall.render()}")
    for kind, agreement in per_kind.items():
        print(f"  {kind}: {agreement.render()}")
    for link, row in reach.items():
        print(f"  {link:14s} {row['spheres']:2d} spheres, worst {row['worst_reach_mm']:+.1f} mm ({row['worst_is']}), "
              f"{row['past_5_mm']} past 5 mm, {row['past_10_mm']} past 10 mm")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "descriptor": args.descriptor, "descriptor_file": source.name, "placement": placed.to_dict(),
        "planner_margin_mm": args.planner_margin_mm,
        "guard_margin_mm": args.guard_margin_mm, "retract": {"spheres_collide": collides[0], "exact_mm": distances[0]},
        "all": overall.to_dict(), "per_kind": {k: v.to_dict() for k, v in per_kind.items()}, "links": reach,
    }, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
