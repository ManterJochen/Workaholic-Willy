"""Measure and commit one retract pose per arm, chosen by the rule on the exact meshes with every hand.

    .venv/Scripts/python.exe scripts/curobo/choose_ur_retract.py                  # every arm with a bundle
    .venv/Scripts/python.exe scripts/curobo/choose_ur_retract.py ur5 --reserve-mm 5
    .venv/Scripts/python.exe scripts/curobo/choose_ur_retract.py --check          # exits 1 if the table would change

The anchor a retract starts from is Isaac's Lula ``default_q``, and cuRobo does more with such a pose than label it: it
biases the graph search, regularises IK and warms up from it. At that anchor ur5 keeps 9.5 mm to the Hand-E and 6.9 mm
to the EGU-50, inside the guard's own 10 mm margin.

So the rule (``_retract_rule``) walks outwards from that pose and takes the first candidate that clears every registry
hand by the guard margin plus a reserve, stands clear of the table, and lies inside the planner's envelope
(``_planner_limits``). This script is the half that can judge, in the project venv where Coal and the committed bundles
live. It writes ``src/robot/safety/planning/robot/ur_retract.yaml``, which the builder reads in the cuRobo environment.

Every judge proves itself first. The folded elbow collides on every UR, so each judge has to read 0.0 there before
anything is written. A judge that answers a distance for a pose that is inside itself is not measuring the meshes.

The EGU-50 is judged on every arm, although the guard admits it only where evidence says so, because a retract has to
clear every hand a cell could name and the arm the cell runs is not known when the descriptor is built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _planner_limits import planner_envelope_rad  # noqa: E402
from _retract_rule import ANCHORS, RetractRefused, choose  # noqa: E402
from exact_self_collision_poses import FOLDED, ExactJudge, _hand_bundle  # noqa: E402

from src.config.grippers import available_grippers  # noqa: E402
from src.robot.safety.planning.environment import collision_mesh_bundle  # noqa: E402
from src.robot.safety.planning._hand_placement import HandPlacement  # noqa: E402

TABLE = REPO / "src" / "robot" / "safety" / "planning" / "robot" / "ur_retract.yaml"
MAPS = REPO / "src" / "robot" / "safety" / "planning" / "robot"
#: The Isaac cell's declared tool frame, which places the hand model on the identity. A real cell's +Z comes with its
#: own evidence, and the table records which placements it judged.
SIM_TOOL_ROTATION_XYZW = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
#: The plates a Hand-E cell declares (robot.hande.yaml) and the bare flange, both judged.
MOUNTING_FACE_PLATES_MM = (0.0, 20.0)
ANCHOR_SOURCE = "Isaac Lula default_q, motion_policy_configs/universal_robots/{arm}/rmpflow/{arm}_robot_description.yaml"


class Hand:
    """One hand at one plate under one placement: what the rule asks, and what the table records."""

    def __init__(self, arm: str, hand: str, plate_mm: float, placement: HandPlacement) -> None:
        self.arm, self.hand, self.plate_mm = arm, hand, plate_mm
        self.label = f"{hand}@{plate_mm:g}mm"
        self.judge = ExactJudge(arm, hand, plate_mm, placement)

    def nearest(self, joints: "list[float]") -> "tuple[float, str]":
        return self.judge.nearest(joints)

    def lowest_z_mm(self, joints: "list[float]") -> float:
        return self.judge.lowest_z_mm(joints)


#: The planner margin each arm's cells ship, where it is not the default. A retract has to be judged at the margin
#: the cell will actually run, not at a number nobody uses, and the table keys its rows by that margin.
#:
#: Empty, because every layer declares the same 4 mm; an arm whose cells ship another margin belongs in here.
MARGINS: dict[str, float] = {}


def _origin(hand: str) -> str:
    document = yaml.safe_load((MAPS / f"{hand}_gripper_spheres.yml").read_text(encoding="utf-8")) or {}
    return str((document.get("_provenance") or {}).get("origin", "flange"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hands(arm: str, placement: HandPlacement) -> "list[Hand]":
    out: list[Hand] = []
    for hand in available_grippers():
        if not _hand_bundle(arm, hand).is_file():
            print(f"  {hand}: no bundle carries its meshes, so it cannot be judged; skipped")
            continue
        plates = MOUNTING_FACE_PLATES_MM if _origin(hand) == "mounting_face" else (0.0,)
        out.extend(Hand(arm, hand, plate, placement) for plate in plates)
    return out


class PlannerJudge:
    """The planner's own sphere model, asked once over a whole candidate family.

    A pose can clear the exact meshes by 13.4 mm while this model calls it a self collision, and the sidecar
    then refuses to become ready and the cell does not come up, so both models judge a retract.
    Measure only, because a sidecar that refuses its own retract is exactly what has to stay up to be asked.
    """

    def __init__(self, arm: str, hand: str, plate_mm: float, rotation_xyzw: Any, margin_mm: float) -> None:
        from src.robot.safety.planning.body_link import HandLink
        from src.robot.safety.planning.curobo_client import CuroboPlanClient
        from src.robot.safety.planning.hand import planner_hand
        from src.config.schema.robot import RobotConfig

        gripper: dict = {"model": hand, "tool_frame": {
            "source": "willy", "offset_mm": [0.0, 132.0, 0.0],
            "rotation_quat_xyzw": [float(v) for v in rotation_xyzw],
        }}
        # Always stated and never left out. A hand whose map starts at its own mounting face refuses a cell that
        # does not say what sits between that face and the flange, and an empty list is the way to say "nothing does".
        gripper["coupling_plates_mm"] = [float(plate_mm)] if plate_mm else []
        link = HandLink.from_hand(planner_hand(RobotConfig.model_validate(
            {"vendor": "ur", "gripper": gripper})))
        self.client = CuroboPlanClient(
            robot_config=f"willy_{arm}.yml", self_collision_margin_mm=float(margin_mm),
            body_links=[link.to_dict()], measure_only=True,
        )
        self.client.start()
        self.margin_mm = float(margin_mm)

    def admits(self, poses: list) -> list:
        """Which of these poses the planner's own model accepts. The verdicts only, on purpose.

        Asking for the deepest link pair as well costs 8.29 ms a pose on the CPU with the committed sphere
        maps (ur5 with the EGU-50, 590 spheres), which is 51 minutes over this rule's own candidate family,
        and this method reads neither the pair nor the depth. `depth_at` asks for them, for the one pose the
        rule ends up choosing.
        """
        explained = self.client.explain_joints([list(pose) for pose in poses], name_pairs=False)
        return [not hit and ok for hit, ok in zip(explained.self_collides, explained.bound_ok)]

    def depth_at(self, pose: list) -> tuple:
        """What the sphere model says about one pose: depth in mm and the pair, or (None, None)."""
        explained = self.client.explain_joints([list(pose)])
        return explained.depths_mm[0], explained.pairs[0]

    def close(self) -> None:
        self.client.close()


def _control_holds(hands: "list[Hand]") -> bool:
    """The folded elbow collides on every UR, so a judge that reads a distance there is not reading the meshes."""
    ok = True
    for hand in hands:
        distance, pair = hand.nearest(list(FOLDED))
        if distance > 0.001:
            print(f"  CONTROL FAILED: the folded elbow keeps {distance:.1f} mm ({pair}) with {hand.label}")
            ok = False
    return ok


def _row(arm: str, hands: "list[Hand]", chosen: Any) -> dict:
    per_hand: dict[str, Any] = {}
    for hand in hands:
        anchor_clearance, anchor_pair = hand.nearest(list(chosen.anchor))
        entry = per_hand.setdefault(hand.hand, {"plates_mm": [], "judged": []})
        entry["plates_mm"].append(float(hand.plate_mm))
        clearance, pair = hand.nearest(chosen.retract)
        entry["judged"].append({
            "plate_mm": float(hand.plate_mm),
            "clearance_mm": round(float(clearance), 3),
            "nearest_pair": pair,
            "lowest_z_mm": round(float(hand.lowest_z_mm(chosen.retract)), 3),
            "anchor_clearance_mm": round(float(anchor_clearance), 3),
            "anchor_nearest_pair": anchor_pair,
        })
    bundles = {"arm": _sha256(collision_mesh_bundle(arm))}
    # Sorted, because a set iterates in a different order per run and the committed file has to be reproducible.
    for hand in sorted({h.hand for h in hands}):
        bundles[hand] = _sha256(_hand_bundle(arm, hand))
    return {
        "anchor": [round(float(v), 6) for v in chosen.anchor],
        "anchor_source": ANCHOR_SOURCE.format(arm=arm),
        "steps": list(chosen.steps),
        "retract": [round(float(v), 6) for v in chosen.retract],
        "candidates_tried": chosen.tried,
        "hands": per_hand,
        "bundles": bundles,
    }


def _report_cell_poses(hands: "list[Hand]") -> None:
    """Report only and never a gate: what the poses cells already carry would measure with these hands."""
    poses: dict[str, list[float]] = {}
    try:
        sim = yaml.safe_load((REPO / "config/robot/robot.sim.yaml").read_text(encoding="utf-8")) or {}
        block = (sim.get("robot") or {}).get("sim") or {}
        for name in ("home_joint_positions", "park_joint_positions"):
            if isinstance(block.get(name), list):
                poses[f"robot.sim.yaml {name}"] = [float(v) for v in block[name]]
    except OSError:
        pass
    for name, value in poses.items():
        worst = min((hand.nearest(value)[0], hand.label, hand.nearest(value)[1]) for hand in hands)
        print(f"  [report] {name}: {worst[0]:.1f} mm with {worst[1]} at {worst[2]}")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Choose and commit one retract per arm, judged on the exact meshes.")
    parser.add_argument("arms", nargs="*", help="arms to judge; every arm with a committed bundle when empty")
    parser.add_argument("--planner-margin-mm", type=float, default=4.0,
                        help="the planner margin a pose is judged in the sphere model at, unless MARGINS says else")
    parser.add_argument("--guard-margin-mm", type=float, default=10.0)
    # 3 mm, and the measurement is why. At 5 mm ur3 would have to move for `forearm|wrist_2` at 13.4 mm, a pair no
    # hand takes part in, so the reserve would be paying for arm geometry the sphere fit covers. At 3 mm exactly the
    # two arms whose hands sit inside or beside the guard band move: ur5 (6.9 mm with the EGU-50) and ur10 (11.1 mm).
    parser.add_argument("--reserve-mm", type=float, default=3.0)
    parser.add_argument("--table-clearance-mm", type=float, default=10.0)
    # 0.25 rad per step, so six steps reach 1.5 rad on a joint. Sweeping ur5 joint by joint, the hand against
    # wrist_1 opens only for wrist_2 and wrist_3, and only at about 1.5 rad. A finer step cannot reach that pose at
    # 15 to 90 ms per candidate, and the four joints before the wrist do not change that pair at all, because they
    # move the whole wrist assembly rigidly.
    parser.add_argument("--step-rad", type=float, default=0.25, help="how far one step moves a joint")
    parser.add_argument("--max-steps", type=int, default=6, help="how many steps the search may walk on each joint")
    parser.add_argument("--tool-rotation-xyzw", type=float, nargs=4, default=list(SIM_TOOL_ROTATION_XYZW))
    parser.add_argument("--check", action="store_true", help="recompute and exit 1 when the committed table differs")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    placement = HandPlacement.from_quaternion_xyzw(tuple(args.tool_rotation_xyzw))
    arms = args.arms or [arm for arm in ANCHORS if collision_mesh_bundle(arm).is_file()]
    skipped = [arm for arm in ANCHORS if arm not in arms]
    envelope = planner_envelope_rad()

    table: dict[str, Any] = {
        "rule": {
            "generated_by": "scripts/curobo/choose_ur_retract.py",
            "guard_margin_mm": args.guard_margin_mm,
            "reserve_mm": args.reserve_mm,
            "table_clearance_mm": args.table_clearance_mm,
            "step_rad": args.step_rad,
            "max_steps": args.max_steps,
            "placement": f"{placement.approach}{placement.closing}",
            "tool_rotation_xyzw": [float(v) for v in args.tool_rotation_xyzw],
            "plates_mm_judged_for_mounting_face_hands": list(MOUNTING_FACE_PLATES_MM),
            "engine": "",
            "arms_without_a_bundle": skipped,
            "planner_margin_mm": float(args.planner_margin_mm),
        },
        "retracts": [],
    }

    refused: list = []
    for arm in arms:
        hands = _hands(arm, placement)
        if not hands:
            print(f"  {arm} has no hand to judge; nothing written")
            return 2
        table["rule"]["engine"] = hands[0].judge.engine
        if not _control_holds(hands):
            print("  nothing written")
            return 1

        for hand in hands:
            margin = MARGINS.get(arm, args.planner_margin_mm)
            print(f"{arm} with {hand.label} at a {margin:g} mm planner margin:", flush=True)
            try:
                planner = PlannerJudge(arm, hand.hand, hand.plate_mm, args.tool_rotation_xyzw, margin)
            except Exception as exc:  # noqa: BLE001 (an arm with no descriptor cannot be asked, and says so)
                # ur16e lands here: it has a bundle and no descriptor, because its vendor sphere map does not
                # describe that arm. A pair the planner cannot be asked about has no retract, and saying which
                # of the two halves is missing is the whole point of writing it down.
                print(f"  NOT ASKED: {type(exc).__name__}: {exc}", flush=True)
                refused.append({"arm": arm, "hand": hand.hand, "plate_mm": float(hand.plate_mm),
                                "planner_margin_mm": float(margin),
                                "reason": f"the planner could not be asked: {exc}"})
                continue
            try:
                chosen = choose(
                    arm, ANCHORS[arm], [hand], planner_envelope=envelope,
                    guard_margin_mm=args.guard_margin_mm, reserve_mm=args.reserve_mm,
                    table_clearance_mm=args.table_clearance_mm, step_rad=args.step_rad,
                    max_steps=args.max_steps, planner_admits=planner.admits,
                )
            except RetractRefused as exc:
                print(f"  REFUSED: {exc}", flush=True)
                refused.append({"arm": arm, "hand": hand.hand,
                                "plate_mm": float(hand.plate_mm),
                                "planner_margin_mm": float(margin),
                                "reason": str(exc)})
                planner.close()
                continue
            depth, pair = planner.depth_at(chosen.retract)
            anchor_spheres = planner.depth_at(list(chosen.anchor))
            planner.close()

            clearance, mesh_pair = hand.nearest(chosen.retract)
            # What the anchor itself measures, in both models. It is the evidence that a move was necessary: a row
            # whose steps are non zero and whose anchor clears both judges would mean the rule walked for nothing.
            anchor_clearance, anchor_mesh_pair = hand.nearest(list(chosen.anchor))
            anchor_depth, anchor_sphere_pair = anchor_spheres
            table["retracts"].append({
                "arm": arm,
                "hand": hand.hand,
                "plate_mm": float(hand.plate_mm),
                "planner_margin_mm": float(margin),
                "anchor": [float(v) for v in chosen.anchor],
                "steps": list(chosen.steps),
                "retract": [round(float(v), 6) for v in chosen.retract],
                "candidates_tried": int(chosen.tried),
                "exact": {
                    "clearance_mm": round(float(clearance), 3),
                    "nearest_pair": str(mesh_pair),
                    "lowest_z_mm": round(float(hand.lowest_z_mm(chosen.retract)), 3),
                    "anchor_clearance_mm": round(float(anchor_clearance), 3),
                    "anchor_nearest_pair": str(anchor_mesh_pair),
                },
                "spheres": {
                    "depth_mm": None if depth is None else round(float(depth), 3),
                    "pair": None if pair is None else list(pair),
                    "anchor_depth_mm": None if anchor_depth is None else round(float(anchor_depth), 3),
                    "anchor_pair": None if anchor_sphere_pair is None else list(anchor_sphere_pair),
                },
                # A re-bake of either bundle invalidates this row, and this is what says so without an engine:
                # the pose is judged on these exact bytes, in both models.
                "bundles": {
                    "arm": _sha256(collision_mesh_bundle(arm)),
                    hand.hand: _sha256(_hand_bundle(arm, hand.hand)),
                },
                "anchor_source": ANCHOR_SOURCE.format(arm=arm),
            })
            moved = ("kept the anchor" if chosen.steps == (0, 0, 0, 0, 0)
                     else f"moved by steps {list(chosen.steps)}")
            print(f"  {moved} after {chosen.tried} admitted candidate(s); exact {clearance:.1f} mm at "
                  f"{mesh_pair}, spheres {depth} mm", flush=True)

    table["rule"]["pairs_with_no_retract"] = refused

    text = yaml.safe_dump(table, sort_keys=False, default_flow_style=False)
    if args.check:
        if not TABLE.is_file():
            print(f"no committed table at {TABLE}")
            return 1
        if TABLE.read_text(encoding="utf-8") != text:
            print(f"{TABLE.name} differs from what this run measured; regenerate it")
            return 1
        print(f"{TABLE.name} is what this box measures")
        return 0
    TABLE.write_text(text, encoding="utf-8")
    print(f"\nwrote {TABLE}: {len(table['retracts'])} arm and hand pair(s), {len(refused)} with no retract")
    print(json.dumps({f"{row['arm']}/{row['hand']}@{row['plate_mm']:g}": row['steps']
                      for row in table['retracts']}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
