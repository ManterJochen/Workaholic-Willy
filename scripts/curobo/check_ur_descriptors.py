"""Load every built UR arm descriptor in cuRobo with a hand added, and plan one motion with it. Run it after a build.

    ext_deps/curobo_env/python.exe scripts/curobo/check_ur_descriptors.py --hand robotiq_2f85
    ext_deps/curobo_env/python.exe scripts/curobo/check_ur_descriptors.py willy_ur3e willy_ur5e --hand robotiq_hande \
        --coupling-mm 20

A descriptor is built per arm and carries no hand; a planner adds the hand a cell names as a body link when its sidecar
starts. So this composes what the sidecar composes (``_curobo_body_links``): the arm descriptor, the hand from its
committed sphere map placed by ``--tool-rotation-xyzw`` (default: the Isaac cell's declared frame, which places the hand
on the identity) and ``--coupling-mm``, then ``--planner-margin-mm``. With no descriptor named, every ``willy_ur*.yml``
in the content is checked.

A loader rather than a reader: ``build_ur_config.py`` writes a yml and says so, and a yml that parses is not yet a robot
cuRobo can plan for. The joint names have to match the urdf, the sphere map has to cover the links the kinematics
declares, and the mesh references have to resolve far enough for the loader to get through. Each of those fails at plan
time, on the box, in front of an operator.

What that settles: ``ur3``, ``ur3e`` and ``ur5`` reference mesh files the cuRobo asset root does not carry, which ships
``ur5e`` and ``ur10e`` meshes only, and the build prints a warning about it. cuRobo plans against the sphere map rather
than the urdf meshes, and only the loader knows how far it reads, so this asks it on the box with the real library. A
model that plans is fine with the warning, and a model that does not is not.

Exit code is 0 when every requested model plans, 1 otherwise, so it can gate a build.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Any

# Beside this script, because it runs in the cuRobo environment where the repository cannot be imported. It reads the
# committed sphere map and places it the way a cell's planner does, and it puts the planner's own stdlib modules on the
# path for the composition below.
from _hand_body import SIM_TOOL_ROTATION_XYZW, HandBodyError, hand_body  # type: ignore[import-not-found]


def _content_dir() -> Path:
    """The cuRobo ``content`` dir. Same order as build_ur_config.py: env, then cuRobo's own answer."""
    env = os.environ.get("WILLY_CUROBO_CONTENT")
    if env:
        return Path(env)
    from curobo.content import get_content_root  # type: ignore[import-not-found]

    return Path(str(get_content_root()))


def _plan_once(stem: str, body: dict[str, Any], margin_mm: float) -> tuple[bool, str]:
    """Build a planner for the arm descriptor ``stem`` with ``body`` added and plan one short motion."""
    import torch  # type: ignore[import-not-found]
    from curobo._src.util.config_io import resolve_config  # type: ignore[import-not-found]
    from curobo.kinematics import Kinematics, KinematicsCfg  # type: ignore[import-not-found]
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # type: ignore[import-not-found]
    from curobo.types import GoalToolPose, JointState  # type: ignore[import-not-found]

    from _curobo_body_links import compose_sidecar_config  # type: ignore[import-not-found]

    cfg = _content_dir() / "configs" / "robot" / f"{stem}.yml"
    if not cfg.is_file():
        return False, f"no descriptor at {cfg}"

    started = time.time()
    composed = compose_sidecar_config(resolve_config(str(cfg)), bodies=[body], margin_mm=margin_mm)
    planner = MotionPlanner(MotionPlannerCfg.create(robot=copy.deepcopy(composed), scene_model={"cuboid": {}}))
    planner.warmup(enable_graph=False, num_warmup_iterations=2)
    kin = Kinematics(KinematicsCfg.from_data_dict(copy.deepcopy(composed)))

    # Start from the descriptor's own retract pose, so this asks about the robot rather than about a
    # pose somebody picked. The goal is that pose nudged 40 mm along +x: short, inside a UR3 envelope,
    # and far enough to need a real trajectory rather than a no-op.
    # What the planner may propose, read back from the loaded robot rather than from the file: cuRobo applies
    # `position_limit_clip` itself, and the elbow limit comes out of the description. A descriptor whose loaded
    # envelope is wider than UR's is one the guard can refuse mid plan, so it fails here instead.
    from _planner_limits import planner_envelope_rad  # type: ignore[import-not-found]

    loaded = planner.kinematics.get_joint_limits()
    lower = [float(v) for v in loaded.position[0].cpu().tolist()]
    upper = [float(v) for v in loaded.position[1].cpu().tolist()]
    envelope_lower, envelope_upper = planner_envelope_rad()
    wide = [
        f"joint {axis}: loaded [{lower[axis]:.4f}, {upper[axis]:.4f}] against "
        f"[{envelope_lower[axis]:.4f}, {envelope_upper[axis]:.4f}]"
        for axis in range(min(len(lower), len(envelope_lower)))
        if lower[axis] < envelope_lower[axis] - 1e-6 or upper[axis] > envelope_upper[axis] + 1e-6
    ]
    if wide:
        return False, "loaded, and its joint envelope is wider than UR's planning limits: " + "; ".join(wide)

    q0 = planner.default_joint_state.position.squeeze().cpu().tolist()
    q = torch.tensor(q0, device="cuda", dtype=torch.float32).unsqueeze(0)
    start = JointState.from_position(q, joint_names=planner.joint_names)
    here = kin.compute_kinematics(
        JointState.from_position(q, joint_names=kin.joint_names)
    ).tool_poses.get_link_pose(kin.tool_frames[0])
    pos = here.position.squeeze().cpu().tolist()
    quat = here.quaternion.squeeze().cpu().tolist()
    goal = GoalToolPose(
        tool_frames=planner.tool_frames,
        # 5D [B, H, L, G, 3]: the quaternion below nests one deeper by accident of holding a
        # list, and the position has to be spelled out to match it.
        position=torch.tensor([[[[[pos[0] + 0.04, pos[1], pos[2]]]]]], device="cuda",
                              dtype=torch.float32),
        quaternion=torch.tensor([[[[quat]]]], device="cuda", dtype=torch.float32),
    )
    result = planner.plan_pose(goal, start, max_attempts=4)
    took = time.time() - started
    if result is None or not bool(result.success.any()):
        return False, f"loaded, and the plan failed ({took:.1f}s)"
    steps = int(result.get_interpolated_plan().position.reshape(-1, len(planner.joint_names)).shape[0])
    return True, (f"planned {steps} waypoints over {len(planner.joint_names)} joints from "
                  f"default_q={[round(v, 2) for v in q0]} ({took:.1f}s)")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Plan one motion with every arm descriptor and a hand added.")
    parser.add_argument("descriptors", nargs="*", help="willy_{arm} stems; every willy_ur*.yml when none is named")
    parser.add_argument("--hand", required=True, help="the registry name of the hand to add, robot.gripper.model")
    parser.add_argument("--coupling-mm", type=float, default=None, help="the plates between flange and hand, summed")
    parser.add_argument("--tool-rotation-xyzw", type=float, nargs=4, default=SIM_TOOL_ROTATION_XYZW,
                        help="robot.gripper.tool_frame.rotation_quat_xyzw; the Isaac cell's frame by default")
    parser.add_argument("--planner-margin-mm", type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    stems = list(args.descriptors)
    if not stems:
        content = _content_dir() / "configs" / "robot"
        stems = sorted(p.stem for p in content.glob("willy_ur*.yml"))
        if not stems:
            # A check with nothing to check is not a pass. Reporting that all 0 descriptors load and plan would
            # exit 0 on a box where no cell can plan at all.
            print(f"no descriptor under {content} matches willy_ur*.yml; build one with build_ur_config.py first",
                  flush=True)
            return 1
    try:
        body = hand_body(args.hand, coupling_mm=args.coupling_mm, rotation_xyzw=tuple(args.tool_rotation_xyzw))
    except HandBodyError as exc:
        raise SystemExit(str(exc)) from None
    print(f"checking {len(stems)} descriptor(s) with {args.hand} added at {body['fixed_transform']}: "
          f"{', '.join(stems)}\n", flush=True)

    failed: list[str] = []
    for stem in stems:
        try:
            ok, detail = _plan_once(stem, body, args.planner_margin_mm)
        except Exception as exc:  # noqa: BLE001 (every model is reported rather than raised on)
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        print(f"[{'ok  ' if ok else 'FAIL'}] {stem:12s} {detail}"[:400], flush=True)
        if not ok:
            failed.append(stem)

    print(flush=True)
    if failed:
        print(f"{len(failed)} of {len(stems)} cannot be planned with {args.hand}: {', '.join(failed)}")
        return 1
    print(f"all {len(stems)} descriptors load and plan with {args.hand}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
