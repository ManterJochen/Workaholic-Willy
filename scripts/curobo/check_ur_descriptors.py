"""Load every built UR descriptor in cuRobo and plan one motion with it. Run it after a build.

    ext_deps/curobo_env/python.exe scripts/curobo/check_ur_descriptors.py
    ext_deps/curobo_env/python.exe scripts/curobo/check_ur_descriptors.py ur3 ur5

⭐ **WHY A LOADER AND NOT A READER.** ``build_ur_config.py`` writes a yml and says so, and until now
that was the only evidence a model could be planned at all. A yml that parses is not a robot cuRobo
can plan for: the joint names have to match the urdf, the sphere map has to cover the links the
kinematics declares, and the mesh references have to resolve far enough for the loader to get
through. Each of those fails at plan time, on the box, in front of an operator.

⚠ **THE THING THIS WAS WRITTEN TO SETTLE.** ``ur3``, ``ur3e`` and ``ur5`` reference mesh files the
cuRobo asset root does not carry (measured 2026-09-09: it ships ``ur5e`` and ``ur10e`` meshes only),
and the build prints a warning about it. Whether that matters is not a thing to reason about, because
cuRobo plans against the SPHERE map rather than the urdf meshes and only the loader knows how far it
reads. So this asks it, on the box, with the real library. A model that plans is fine with the
warning; a model that does not is not, and either way nobody has to guess again.

Exit code is 0 when every requested model plans, 1 otherwise, so it can gate a build.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _content_dir() -> Path:
    """The cuRobo ``content`` dir. Same order as build_ur_config.py: env, then cuRobo's own answer."""
    env = os.environ.get("WILLY_CUROBO_CONTENT")
    if env:
        return Path(env)
    from curobo.content import get_content_root  # type: ignore[import-not-found]

    return Path(str(get_content_root()))


def _plan_once(model: str) -> tuple[bool, str]:
    """Build a planner for ``model`` and plan one short motion. Returns (ok, what happened)."""
    import torch  # type: ignore[import-not-found]
    from curobo.kinematics import Kinematics, KinematicsCfg  # type: ignore[import-not-found]
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # type: ignore[import-not-found]
    from curobo.types import GoalToolPose, JointState  # type: ignore[import-not-found]

    cfg = _content_dir() / "configs" / "robot" / f"{model}.yml"
    if not cfg.is_file():
        return False, f"no descriptor at {cfg}"

    started = time.time()
    planner = MotionPlanner(MotionPlannerCfg.create(robot=str(cfg), scene_model={"cuboid": {}}))
    planner.warmup(enable_graph=False, num_warmup_iterations=2)
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(str(cfg)))

    # Start from the descriptor's OWN retract pose, so this asks about the robot rather than about a
    # pose somebody picked. The goal is that pose nudged 40 mm along +x: short, inside a UR3 envelope,
    # and far enough to need a real trajectory rather than a no-op.
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
    args = list(sys.argv[1:] if argv is None else argv)
    models = [a for a in args if not a.startswith("-")]
    if not models:
        content = _content_dir() / "configs" / "robot"
        models = sorted(p.stem for p in content.glob("ur*.yml")
                        if not p.stem.startswith("_") and "dual" not in p.stem)
    print(f"checking {len(models)} descriptor(s): {', '.join(models)}\n", flush=True)

    failed: list[str] = []
    for model in models:
        try:
            ok, detail = _plan_once(model)
        except Exception as exc:  # noqa: BLE001 - the point is to report every model, not to raise
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        print(f"[{'ok  ' if ok else 'FAIL'}] {model:7s} {detail}"[:400], flush=True)
        if not ok:
            failed.append(model)

    print(flush=True)
    if failed:
        print(f"{len(failed)} of {len(models)} cannot be planned with: {', '.join(failed)}")
        return 1
    print(f"all {len(models)} descriptors load and plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
