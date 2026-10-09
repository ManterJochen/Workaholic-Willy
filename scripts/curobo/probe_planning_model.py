"""cuRobo loads the planning model the client composes, counts its spheres and pairs as this repository does, and holds
it where the evidence model is. No sidecar and no plan: cuRobo's own kinematics, on the GPU.

    .venv/Scripts/python.exe scripts/curobo/ab_planning_spheres.py --dry-run --dump logs/curobo/planning_model
    <cuRobo-env-python> scripts/curobo/probe_planning_model.py logs/curobo/planning_model

cuRobo's URDF loader builds its tensors on ``cuda`` whatever it is asked (``parser_urdf.py``), so this needs the GPU the
planner runs on; the CPU suite holds the same composition and counts (``tests/test_the_planner_plans_on_a_lean_model_
and_judges_on_the_evidence.py``).

The dry run of the A/B writes the two robots the sidecar would load for the cell (``today.json``, the evidence model,
and ``<model>.json``, the planning model) as it composes them, with the URDF they resolve. This loads each into cuRobo's
own ``Kinematics``, as the sidecar does, and asks:

* does cuRobo hold as many spheres per link, and list as many self collision pairs, as ``_curobo_planning`` counted;
* at the tool frame, do the two robots agree at every configuration (one URDF, one chain);
* is every sphere of the evidence model, at a few hundred configurations, within the planning model's recorded hole of
  one of its spheres: a planning model placed in the wrong frame, a link length away or a quarter turn off, loads and
  plans without a word, and this is where it would show (the descriptor builder's own warning,
  ``_arm_spheres.link_frames``).

Exit 0 when every question holds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

#: How far an evidence sphere's centre may lie outside every planning sphere, metres: the largest hole a lean map
#: records (20 mm on a UR10's shoulder and upper arm), the centre being inside the body by its own construction.
CENTRE_HOLE_M = 0.021
CONFIGURATIONS = 300


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path, help="where ab_planning_spheres.py --dry-run --dump wrote the robots")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--configurations", type=int, default=CONFIGURATIONS)
    args = parser.parse_args(argv)

    import torch  # noqa: PLC0415
    from curobo._src.types.device_cfg import DeviceCfg  # noqa: PLC0415

    from curobo.kinematics import Kinematics, KinematicsCfg  # noqa: PLC0415
    from curobo.types import JointState  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "robot" / "safety" / "planning"))
    from _curobo_planning import self_pair_count, sphere_count  # noqa: PLC0415

    device = DeviceCfg(device=torch.device(args.device))
    manifest = json.loads((args.folder / "manifest.json").read_text(encoding="utf-8"))
    urdf = str(manifest["urdf"])
    if not Path(urdf).is_file() and len(urdf) > 2 and urdf[1] == ":":
        # Written on Windows, read in WSL: the same file under /mnt/<drive>.
        urdf = f"/mnt/{urdf[0].lower()}/{urdf[3:]}"
    manifest["urdf"] = urdf
    robots = {}
    held = True
    for name in manifest["robots"]:
        config = json.loads((args.folder / f"{name}.json").read_text(encoding="utf-8"))
        kinematics = config["robot_cfg"]["kinematics"]
        kinematics["urdf_path"] = manifest["urdf"]
        kinematics.pop("asset_root_path", None)
        model = Kinematics(KinematicsCfg.from_data_dict(json.loads(json.dumps(config)), device_cfg=device))
        loaded = model.config.kinematics_config
        pairs = model.config.self_collision_config.collision_pairs
        counted = sum(sphere_count(config).values()), self_pair_count(config)
        theirs = int(loaded.total_spheres), int(pairs.shape[0]) if pairs is not None else 0
        same = counted == theirs
        held = held and same
        print(f"  {name}: cuRobo holds {theirs[0]} spheres and lists {theirs[1]} self pairs; counted here "
              f"{counted[0]} and {counted[1]}: {'the same' if same else 'DIFFERENT'}", flush=True)
        robots[name] = model

    names = list(robots)
    evidence, planned = robots[names[0]], robots[names[-1]]
    generator = torch.Generator().manual_seed(20261009)
    lower = torch.tensor([-math.pi, -math.pi, -2.8, -math.pi, -math.pi, -math.pi])
    upper = torch.tensor([math.pi, 0.0, 2.8, math.pi, math.pi, math.pi])
    q = lower + (upper - lower) * torch.rand((args.configurations, 6), generator=generator)
    q = q.to(device=device.device, dtype=torch.float32)

    def state(model: Kinematics) -> Any:
        return model.compute_kinematics(JointState.from_position(q.unsqueeze(1), joint_names=model.joint_names))

    ours, theirs = state(evidence), state(planned)
    frame = evidence.tool_frames[0]
    gap = float((ours.tool_poses.get_link_pose(frame).position
                 - theirs.tool_poses.get_link_pose(frame).position).abs().max())
    print(f"  the tool frame of the two robots agrees within {gap * 1000.0:.6f} mm over {len(q)} configurations",
          flush=True)
    held = held and gap < 1e-6
    a = ours.robot_spheres.reshape(len(q), -1, 4)
    b = theirs.robot_spheres.reshape(len(q), -1, 4)
    b = b[:, b[0, :, 3] > 0.0]
    distance = torch.cdist(a[..., :3], b[..., :3]) - b[:, None, :, 3]
    worst = float(distance.min(dim=2).values.max())
    print(f"  every evidence sphere centre lies within {worst * 1000.0:.1f} mm outside the planning model's spheres "
          f"(at most {CENTRE_HOLE_M * 1000.0:.0f} expected)", flush=True)
    held = held and worst <= CENTRE_HOLE_M
    print("[result] " + ("every question held" if held else "a question did NOT hold"), flush=True)
    return 0 if held else 1


if __name__ == "__main__":
    raise SystemExit(main())
