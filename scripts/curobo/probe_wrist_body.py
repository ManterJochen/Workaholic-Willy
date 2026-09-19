"""What a wrist camera's body costs, on the exact meshes and in the planner.

    .venv/Scripts/python.exe scripts/curobo/probe_wrist_body.py --clearance
    .venv/Scripts/python.exe scripts/curobo/probe_wrist_body.py --planner --arms ur5e --hands robotiq_2f85

Run it with the project venv. ``--clearance`` needs Coal or python-fcl and no GPU; ``--planner`` spawns the cuRobo
sidecar per combination and runs on the box.

The camera is a RealSense housing from the registry on a bracket nobody measured, so its placement is a parameter and
the numbers printed describe that placement: the optical axis along the hand's approach, ``--lateral-mm`` off the
flange axis along tool0 +X and ``--axial-mm`` out along the approach. The default, 130 mm and 20 mm, is the Isaac
cell's wrist camera mount (``robot.sim.cameras.wrist.mount_offset_mm``), the one camera on a wrist this repository has
run.

``--clearance`` measures what the guard's pair rule implies. The camera sits on frame 6 and, unlike the hand, is
checked against wrist_2 on frame 5, whose pose relative to the flange depends on joint 6 alone, and against wrist_1,
which depends on joints 5 and 6. So joint 6 is swept alone for wrist_2 and joints 5 and 6 together for wrist_1, and the
smallest distance and where it occurs are printed for the housing grown by each margin. A housing that comes within the
guard's margin at some angle makes a cell that refuses those wrist angles, and that goes to the owner before any cell
declares the mount.

``--planner`` starts the sidecar with the camera on top of each committed combination, measure only, and asks three
things: whether the camera turns the combination's own retract into a self collision (a camera the retract table cannot
hold), whether ``composed_sha256`` is still the committed evidence file's, since the evidence hashes are formed without
the camera, and, with ``--pose-set``, how many of the gate's poses the planner calls clear while the exact meshes with
the camera collide. That count must be zero, because the sphere fill is proven to hold the box.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exact_self_collision_poses import ARM_LINKS, _hand_bundle, _parts  # noqa: E402

from src.robot.safety._fcl_self_collision import (  # noqa: E402
    _HAND_PARTS,
    MeshSelfCollisionBackend,
    _EngineAdapter,
    import_collision_engine,
)
from src.robot.safety._ur_kinematics import ur_link_transforms_mm  # noqa: E402
from src.robot.safety.planning.environment import collision_mesh_bundle  # noqa: E402

#: The two placements the committed evidence holds: the Isaac cell's hand along tool0 +Y, a real flange's along +Z.
PLACEMENTS = {"+Y": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476), "+Z": (0.0, 0.0, 0.0, 1.0)}
#: The arms the evidence measured.
ARMS = ("ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e")
HANDS = ("robotiq_2f85", "robotiq_hande", "schunk_egu50")
#: The link the camera becomes, for a rig named ``probe``.
CAMERA = "wrist_camera_probe"


def camera_body(approach: str, *, model: str, lateral_mm: float, axial_mm: float, margin_mm: float):
    """The camera as ``WristBody``: its optical axis along the approach, off the flange axis along tool0 +X."""
    from src.config.cameras import load_camera
    from src.calibration.serialization import FlangeToTcp
    from src.geometry import Frame, Transform
    from src.robot.safety.planning.body_link import WristBody

    z = np.array([0.0, 1.0, 0.0]) if approach == "+Y" else np.array([0.0, 0.0, 1.0])
    x = np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    placement = np.eye(4)
    placement[:3, :3] = np.stack([x, y, z], axis=1)
    placement[:3, 3] = lateral_mm * x + axial_mm * z
    return WristBody.from_parts(
        rig_id="probe", spec=load_camera(model), bracket=None, margin_mm=margin_mm,
        camera_to_tool=Transform.from_matrix(placement, from_frame=Frame.CAMERA, to_frame=Frame.TOOL),
        flange_to_tcp=FlangeToTcp.from_matrix("willy", np.eye(4)), artifact_path="probe")


class CameraJudge:
    """An arm, optionally its hand, and the camera part, judged with the guard's own pair rule."""

    def __init__(self, arm: str, body, *, hand: "str | None" = None, placement=None) -> None:
        mod, kind = import_collision_engine()
        if mod is None or kind is None:
            raise SystemExit("no collision engine: install Coal or python-fcl in this interpreter")
        self.arm = arm
        self._adapter = _EngineAdapter(mod, kind)
        meshes = dict(_parts(collision_mesh_bundle(arm), ARM_LINKS, 0.0))
        if hand is not None:
            meshes.update(_parts(_hand_bundle(arm, hand), tuple(_HAND_PARTS), 0.0, placement))
        parts = body.guard_parts()
        meshes[CAMERA] = (parts[f"{CAMERA}__v"], parts[f"{CAMERA}__f"], int(parts[f"{CAMERA}__frame"][0]))
        self._backend = MeshSelfCollisionBackend(self._adapter, meshes, wrist_parts=frozenset({CAMERA}))
        frame = self._backend._frame
        names = self._backend._names
        self._pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                       if abs(frame[a] - frame[b]) > 1
                       or (abs(frame[a] - frame[b]) == 1 and CAMERA in (a, b))]

    def distance(self, joints, *, partner: "str | None" = None) -> "tuple[float, str]":
        frames = ur_link_transforms_mm(self.arm, np.asarray(joints, dtype=np.float64))
        if frames is None:
            raise SystemExit(f"no DH chain for {self.arm}")
        for name in self._backend._names:
            t = frames[self._backend._frame[name]]
            self._adapter.set_transform(self._backend._models[name], t[:3, :3], t[:3, 3])
        best, pair = float("inf"), ""
        for a, b in self._pairs:
            if partner is not None and {a, b} != {CAMERA, partner}:
                continue
            d = float(self._adapter.distance(self._backend._models[a], self._backend._models[b]))
            if d < best:
                best, pair = d, f"{a}|{b}"
        return best, pair


def clearance(args: argparse.Namespace) -> int:
    q6 = np.radians(np.arange(-180.0, 180.0 + 1e-9, args.step_deg))
    q5 = np.radians(np.arange(-180.0, 180.0 + 1e-9, args.grid_deg))
    q6_grid = np.radians(np.arange(-180.0, 180.0 + 1e-9, args.grid_deg))
    print(f"[clearance] {args.model} at {args.lateral_mm:g} mm off the flange axis and {args.axial_mm:g} mm out, "
          f"guard margin {args.guard_margin_mm:g} mm")
    worst_overall = math.inf
    for arm in args.arms:
        for approach in args.placements:
            for grown in args.margins_mm:
                body = camera_body(approach, model=args.model, lateral_mm=args.lateral_mm, axial_mm=args.axial_mm,
                                   margin_mm=grown)
                judge = CameraJudge(arm, body)
                w2 = min((judge.distance([0.0, -1.57, 1.57, 0.0, 0.0, float(a)], partner="wrist_2")[0], float(a))
                         for a in q6)
                w1 = min((judge.distance([0.0, -1.57, 1.57, 0.0, float(b), float(a)], partner="wrist_1")[0],
                          float(b), float(a)) for b in q5 for a in q6_grid)
                flag = "  <- within the guard margin: to the owner" if min(w2[0], w1[0]) < args.guard_margin_mm else ""
                worst_overall = min(worst_overall, w2[0], w1[0])
                print(f"  {arm:6s} {approach} grown {grown:4.1f} mm: wrist_2 {w2[0]:8.3f} mm at q6 "
                      f"{math.degrees(w2[1]):7.1f} deg; wrist_1 {w1[0]:8.3f} mm at q5 {math.degrees(w1[1]):6.1f} q6 "
                      f"{math.degrees(w1[2]):6.1f} deg{flag}")
    print(f"[clearance] smallest over everything: {worst_overall:.3f} mm")
    return 0


def planner(args: argparse.Namespace) -> int:
    from matrix_gate import _robot_config
    from _matrix_gate import false_clears, pose_set

    from src.contracts import chosen
    from src.robot.drivers.sim.robot_models import curobo_arm_descriptor
    from src.robot.safety.planning.body_link import HandLink, coupling_bodies
    from src.robot.safety.planning.curobo_client import CuroboPlanClient
    from src.robot.safety.planning.hand import planner_hand
    from src.robot.safety.planning.robot.retract_table import RetractMissing, read_retract

    evidence_dir = REPO / "src/robot/safety/planning/robot/evidence"
    print(f"[planner] {args.model} at {args.lateral_mm:g} mm off axis, {args.axial_mm:g} mm out, grown "
          f"{args.margins_mm[-1]:g} mm, planner margin {args.planner_margin_mm:g} mm")
    failures = 0
    for arm in args.arms:
        for hand in args.hands:
            for approach in args.placements:
                cfg = _robot_config(arm, hand, 0.0, args.planner_margin_mm, PLACEMENTS[approach])
                resolved = planner_hand(cfg)
                if not chosen(resolved) or not chosen(resolved.placement):
                    print(f"  {arm:6s} {hand:14s} {approach}: no placed hand")
                    continue
                placed = resolved.placement
                where = f"{placed.approach}{placed.closing}"
                try:
                    retract = read_retract(arm, hand, 0.0, args.planner_margin_mm, placement=where)
                except RetractMissing:
                    print(f"  {arm:6s} {hand:14s} {where}: no judged retract")
                    continue
                name = f"{arm}_{hand}_c0mm_{where}_m{args.planner_margin_mm:g}mm_a0.json"
                committed = evidence_dir / name
                expected = (json.loads(committed.read_text(encoding="utf-8")).get("composed_sha256")
                            if committed.is_file() else None)
                body = camera_body(placed.approach, model=args.model, lateral_mm=args.lateral_mm,
                                   axial_mm=args.axial_mm, margin_mm=args.margins_mm[-1])
                link = HandLink.from_hand(resolved)
                poses = [retract]
                if args.pose_set:
                    poses, _ = pose_set(retract, random_n=args.random)
                with CuroboPlanClient(robot_config=curobo_arm_descriptor(arm),
                                      self_collision_margin_mm=args.planner_margin_mm,
                                      body_links=[link.to_dict(), *coupling_bodies(resolved)],
                                      wrist_body_links=[body.link()], default_q=retract, measure_only=True) as client:
                    identity = client.identity
                    explained = client.explain_joints(poses, name_pairs=True)
                if not chosen(explained.pairs):
                    raise SystemExit(f"{arm} with {hand}: the sidecar was asked to name the pairs and named none")
                retract_hit = bool(explained.self_collides[0])
                pair = explained.pairs[0] if retract_hit else ""
                same = expected is not None and identity.composed_sha256 == expected
                line = (f"  {arm:6s} {hand:14s} {where}: retract {'refused at ' + str(pair) if retract_hit else 'clear'}, "
                        f"composed_sha256 {'is the evidence file' if same else 'differs from ' + name}")
                if args.pose_set:
                    judge = CameraJudge(arm, body, hand=hand, placement=placed)
                    exact_clear = [judge.distance(pose)[0] > 0.0 for pose in poses]
                    missed = false_clears(list(explained.self_collides), exact_clear)
                    line += f", {sum(explained.self_collides)} of {len(poses)} refused, {missed} false clears"
                    failures += int(missed > 0)
                failures += int(not same)
                print(line, flush=True)
    return 1 if failures else 0


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Measure what a wrist camera's body costs.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--clearance", action="store_true", help="exact mesh distances to wrist_1 and wrist_2")
    mode.add_argument("--planner", action="store_true", help="the camera in the sidecar on each combination")
    parser.add_argument("--model", default="realsense_d435", help="the camera registry name")
    parser.add_argument("--lateral-mm", type=float, default=130.0, help="off the flange axis, along tool0 +X")
    parser.add_argument("--axial-mm", type=float, default=20.0, help="out along the hand's approach")
    parser.add_argument("--margins-mm", type=lambda s: [float(v) for v in s.split(",")], default=[0.001, 10.0],
                        help="how far the housing is grown, comma separated, each above 0; --planner reads the last")
    parser.add_argument("--arms", type=lambda s: [v for v in s.split(",") if v], default=list(ARMS))
    parser.add_argument("--hands", type=lambda s: [v for v in s.split(",") if v], default=list(HANDS))
    parser.add_argument("--placements", type=lambda s: [v for v in s.split(",") if v], default=["+Y", "+Z"])
    parser.add_argument("--step-deg", type=float, default=0.5, help="the joint 6 sweep for wrist_2")
    parser.add_argument("--grid-deg", type=float, default=5.0, help="the joint 5 by joint 6 grid for wrist_1")
    parser.add_argument("--guard-margin-mm", type=float, default=10.0)
    parser.add_argument("--planner-margin-mm", type=float, default=4.0)
    parser.add_argument("--pose-set", action="store_true", help="judge the gate's whole pose set, not the retract")
    parser.add_argument("--random", type=int, default=1000)
    args = parser.parse_args(argv)
    return clearance(args) if args.clearance else planner(args)


if __name__ == "__main__":
    raise SystemExit(main())
