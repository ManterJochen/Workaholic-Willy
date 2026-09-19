"""Judge a seeded set of UR joint poses by the exact meshes, for the sphere fidelity probe to compare against.

    .venv/Scripts/python.exe scripts/curobo/exact_self_collision_poses.py ur5 --hand robotiq_hande --coupling-mm 20 \
        --out logs/u2/exact_ur5_robotiq_hande.json

For one arm and hand it writes the poses and, for each, the smallest distance the exact meshes keep over every pair the
self collision guard checks (links more than one DH frame apart, as ``_fcl_self_collision.py`` ``evaluate`` skips):

* the retract pose the descriptor starts from, read from the committed table ``ur_retract.yaml`` the builder reads;
* ``--random`` joint vectors, uniform in [-pi, pi] from ``--seed``;
* the grid over joints 5 and 6, the only joints between wrist_1 and the hand;
* Isaac's Lula anchor last, so the file says what the pose the arm shipped with was worth.

Coal answers 0.0 for contact and for penetration alike, so 0.0 means touching or inside and a threshold below 0.0 reads
as no collision on any arm. The folded elbow, which collides on every arm, is therefore judged first, and a file whose
control reads clear is not written.

The hand's meshes come from the hand's own bundle and are placed exactly as the guard places them: one plate along the
model's approach for a mounting face bundle, then the rotation the declared tool frame gives (``--tool-rotation-xyzw``,
the Isaac cell's frame by default, which places the hand model on the identity). The file records that placement, so a
fidelity run that places the hand elsewhere refuses to compare against it. An arm with no committed bundle
cannot be judged here and is refused. Project venv only, because it needs Coal or python-fcl and the committed bundles.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Beside this script: the tool frame the Isaac cell declares, so this file and the descriptor check name one default,
# and the pose set every gate judges, so no two gates measure different configurations.
from _hand_body import SIM_TOOL_ROTATION_XYZW  # noqa: E402
from _matrix_gate import pose_set  # noqa: E402
from _retract_rule import ANCHORS, read_arm_retract  # noqa: E402

from src.robot.safety._fcl_self_collision import (  # noqa: E402
    _HAND_PARTS,
    MeshSelfCollisionBackend,
    _EngineAdapter,
    import_collision_engine,
    place_hand_vertices,
)
from src.robot.safety.planning._hand_placement import HandPlacement  # noqa: E402
from src.robot.safety._ur_kinematics import ur_link_transforms_mm  # noqa: E402
from src.robot.safety.planning.environment import collision_mesh_bundle  # noqa: E402

ISAAC_MP = Path("D:/isaacsim/isaac-sim-standalone-5.1.0-windows-x86_64/exts/isaacsim.robot_motion.motion_generation/"
                "motion_policy_configs/universal_robots")
ARM_LINKS = ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")
#: The elbow folded back onto the upper arm: a collision on every UR, and this file's control.
FOLDED = (0.0, -1.57, 3.1, 0.0, 0.0, 0.0)


def _parts(
    path: Path, names: tuple[str, ...], plate_mm: float, placement: "HandPlacement | None" = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """The meshes of ``names``, with hand parts placed exactly as the guard places them (``place_hand_vertices``)."""
    data = np.load(path)
    origin = str(np.asarray(data["gripper__origin"]).reshape(-1)[0]) if "gripper__origin" in data.files else ""
    out = {}
    for name in names:
        vertices = np.asarray(data[f"{name}__v"], dtype=np.float64).copy()
        vertices = place_hand_vertices(name, vertices, origin=origin, coupling_mm=plate_mm, placement=placement)
        out[name] = (vertices, np.asarray(data[f"{name}__f"]), int(np.asarray(data[f"{name}__frame"]).reshape(-1)[0]))
    return out


def _hand_bundle(model: str, hand: str) -> Path:
    """The hand's own bundle, which is the same on every arm; the guard composes it onto ``model`` at load."""
    from src.robot.safety.planning.environment import hand_mesh_bundle

    return hand_mesh_bundle(hand)


def _retract(model: str) -> list[float]:
    """The pose the descriptor starts from, which is the committed table's and not Isaac's.

    A fidelity measurement whose first pose is not the pose the planner starts from is measuring a robot nobody runs.
    An arm the table never judged raises, rather than falling back to the pose the rule exists to replace.
    """
    return read_arm_retract(REPO / "src/robot/safety/planning/robot/ur_retract.yaml", model).retract


def judged_pose_set(model: str, *, random_n: int = 1000, seed: int = 20260915) -> "tuple[list[list[float]], list[str]]":
    """The pose set, with Isaac's anchor appended as the last pose.

    Appended rather than substituted, so every earlier index stays the pose an earlier run measured and the two sets
    compare, while the file still answers what the pose the arm shipped with is worth with these hands.
    """
    poses, kinds = pose_set(_retract(model), random_n=random_n, seed=seed)
    poses.append([float(v) for v in ANCHORS[model]])
    kinds.append("anchor")
    return poses, kinds


class ExactJudge:
    """The smallest exact distance over the pairs the guard checks, at one pose."""

    def __init__(self, model: str, hand: str, plate_mm: float, placement: "HandPlacement | None" = None) -> None:
        mod, kind = import_collision_engine()
        if mod is None or kind is None:
            raise SystemExit("no collision engine: install Coal or python-fcl in this interpreter")
        self.engine = kind
        self.model = model
        self._adapter = _EngineAdapter(mod, kind)
        meshes = {**_parts(collision_mesh_bundle(model), ARM_LINKS, 0.0),
                  **_parts(_hand_bundle(model, hand), tuple(_HAND_PARTS), plate_mm, placement)}
        #: Kept as numbers and not only as engine models, because the table check needs vertices and a second read of
        #: the bundle could place the hand differently from the one the distances were measured on.
        self._meshes = meshes
        self._backend = MeshSelfCollisionBackend(self._adapter, meshes)
        names = self._backend._names
        frame = self._backend._frame
        self._pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if abs(frame[a] - frame[b]) > 1]

    def lowest_z_mm(self, joints: list[float], min_frame: int = 3) -> float:
        """How far the lowest vertex of the forearm, the wrists and the hand stands above the base plane, in mm.

        A retract that clears itself can still hang in the bench, and the bench is not in the guard's self collision
        model at all. Shoulder and upper arm are excluded, because their low points sit at joints the retract search
        leaves alone.
        """
        frames = ur_link_transforms_mm(self.model, np.asarray(joints, dtype=np.float64))
        if frames is None:
            raise SystemExit(f"no DH chain for {self.model}")
        lowest = float("inf")
        for vertices, _faces, frame in self._meshes.values():
            if frame < min_frame:
                continue
            transform = frames[frame]
            placed = vertices @ transform[:3, :3].T + transform[:3, 3]
            lowest = min(lowest, float(placed[:, 2].min()))
        return lowest

    def nearest(self, joints: list[float]) -> tuple[float, str]:
        frames = ur_link_transforms_mm(self.model, np.asarray(joints, dtype=np.float64))
        if frames is None:
            raise SystemExit(f"no DH chain for {self.model}")
        for name in self._backend._names:
            t = frames[self._backend._frame[name]]
            self._adapter.set_transform(self._backend._models[name], t[:3, :3], t[:3, 3])
        best, pair = float("inf"), ""
        for a, b in self._pairs:
            d = float(self._adapter.distance(self._backend._models[a], self._backend._models[b]))
            if d < best:
                best, pair = d, f"{a}|{b}"
        return best, pair


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge seeded UR poses by the exact meshes.")
    parser.add_argument("arm")
    parser.add_argument("--hand", required=True)
    parser.add_argument("--coupling-mm", type=float, default=0.0)
    parser.add_argument("--tool-rotation-xyzw", type=float, nargs=4, default=list(SIM_TOOL_ROTATION_XYZW),
                        help="robot.gripper.tool_frame.rotation_quat_xyzw; the Isaac cell's frame by default, which "
                             "places the hand model on the identity")
    parser.add_argument("--random", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    # Where the hand sits on the flange, by the same derivation the guard and the planner use. A pose set is judged for
    # one placement, so the file says which, and a fidelity run against another placement can refuse.
    placement = HandPlacement.from_quaternion_xyzw(tuple(args.tool_rotation_xyzw))

    if not collision_mesh_bundle(args.arm).is_file():
        print(f"{args.arm} has no committed bundle ({collision_mesh_bundle(args.arm).name}), so its exact meshes cannot "
              "be judged; nothing written")
        return 2
    if not _hand_bundle(args.arm, args.hand).is_file():
        print(f"no bundle carries the {args.hand} meshes; nothing written")
        return 2

    judge = ExactJudge(args.arm, args.hand, args.coupling_mm, placement)
    control, control_pair = judge.nearest(list(FOLDED))
    if control > 0.001:
        print(f"CONTROL FAILED: the folded elbow keeps {control:.1f} mm ({control_pair}) on {args.arm}; nothing written")
        return 1

    poses, kinds = judged_pose_set(args.arm, random_n=args.random, seed=args.seed)

    distances, pairs = [], []
    for pose in poses:
        d, pair = judge.nearest(pose)
        distances.append(round(d, 3))
        pairs.append(pair)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "arm": args.arm, "hand": args.hand, "coupling_mm": args.coupling_mm, "seed": args.seed, "engine": judge.engine,
        "placement": placement.to_dict(),
        "control_folded_elbow_mm": round(control, 3), "kinds": kinds, "poses": poses,
        "exact_distance_mm": distances, "nearest_pair": pairs,
    }), encoding="utf-8")
    touching = sum(1 for d in distances if d <= 0.001)
    print(f"{args.arm} {args.hand}: {len(poses)} poses judged by {judge.engine}, {touching} touching or inside, "
          f"retract {distances[0]:.1f} mm ({pairs[0]}); control {control:.1f} mm -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
