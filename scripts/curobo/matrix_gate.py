"""Measure one arm with one hand through the sidecar, and write the file that admits the pair.

    .venv/Scripts/python.exe scripts/curobo/matrix_gate.py --arm ur5e --hand robotiq_2f85 --write
    .venv/Scripts/python.exe scripts/curobo/matrix_gate.py --arm ur3e --hand robotiq_hande --coupling-mm 20 --write

Run it with the project venv: it resolves the hand through the registry, composes the guard, and spawns the cuRobo
sidecar in its own environment. It needs a GPU and a built descriptor, so it runs on the box.

Which arm may carry which hand is a measurement, written to a file named after the combination it measured
(``planning/evidence.py``). A list of arms baked into each hand bundle would admit pairings nobody measured, and four
such pairings are pairs cuRobo refuses to plan from at all.

It measures and does not plan. The sidecar starts with ``measure_only``, so a combination whose own retract its
spheres call a self collision stays up and can be counted. A gate that needed a ready planner could only ever record
the pairs that already work, which is the half nobody needs a file for.

A descriptor that can name no pair is refused rather than measured. The attribution count asks whether the planner's
verdict and its own account of that verdict agree. A descriptor whose spheres came from a file carries no ownership
layout, so it names nothing, and every colliding pose would read as a disagreement: a number about the descriptor's
shape rather than about the robot.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from _matrix_gate import attribution_disagreements, false_clears, pose_set, poses_sha256  # noqa: E402

from src.contracts import chosen  # noqa: E402

#: The guard margin a cell keeps clear while this is measured. Recorded in the file, because a combination measured
#: against another guard is a different measurement.
_DEFAULT_GUARD_MARGIN_MM = 10.0


def _exact_judge() -> ModuleType:
    """``exact_self_collision_poses`` by path: it sits beside this file and imports the repository itself."""
    path = Path(__file__).resolve().parent / "exact_self_collision_poses.py"
    spec = importlib.util.spec_from_file_location("_exact_for_matrix_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plate_spec(text: str) -> dict:
    """``NAME:THICKNESS_MM`` or ``NAME:THICKNESS_MM:HX,HY``: one ``robot.gripper.coupling_plates`` entry."""
    form = "NAME:THICKNESS_MM or NAME:THICKNESS_MM:X_MM,Y_MM"
    fields = text.split(":")
    if len(fields) not in (2, 3) or not fields[0].strip():
        raise argparse.ArgumentTypeError(f"{text!r} is not a plate; a plate is {form}")
    try:
        thickness = float(fields[1])
        section = [float(v) for v in fields[2].split(",")] if len(fields) == 3 else None
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a plate; a plate is {form}") from None
    if thickness <= 0.0 or (section is not None and (len(section) != 2 or min(section) <= 0.0)):
        raise argparse.ArgumentTypeError(f"{text!r} is not a plate: a thickness above 0, and a cross section of two sides above 0; {form}")
    plate: dict = {"name": fields[0].strip(), "thickness_mm": thickness}
    if section is not None:
        plate["cross_section_mm"] = section
    return plate


def _robot_config(arm: str, hand: str, coupling_mm: float, planner_margin_mm: float, rotation_xyzw, *, plates=None):
    """A cell exactly as a profile would declare it, so what is measured is what a cell would run.

    ``plates`` are ``robot.gripper.coupling_plates`` entries as a cell writes them, a cross section included, and take
    the place of the one plain plate ``coupling_mm`` would declare.
    """
    from src.config.schema.robot import RobotConfig

    gripper: dict = {"model": hand}
    if plates:
        gripper["coupling_plates"] = [dict(plate) for plate in plates]
    elif coupling_mm:
        gripper["coupling_plates"] = [{"name": "plate", "thickness_mm": float(coupling_mm)}]
    else:
        gripper["coupling_plates"] = []
    if rotation_xyzw is not None:
        # The offset places nothing measured here (where the hand model sits is the rotation alone,
        # ``_hand_placement``), so it is one millimetre along the declared approach. An offset of zero beside an
        # identity rotation is the marker the schema refuses as an unmeasured cell.
        from src.geometry.quaternion import to_rotation_matrix

        approach = to_rotation_matrix(np.asarray(rotation_xyzw, dtype=np.float64))[:, 2]
        gripper["tool_frame"] = {"source": "willy", "offset_mm": [float(v) for v in approach],
                                 "rotation_quat_xyzw": [float(v) for v in rotation_xyzw]}
    return RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": arm},
        "safety": {"payload": {"enforce": False},
                   "self_collision": {"backend": "fcl", "kinematics_model": arm,
                                      "planner_margin_mm": float(planner_margin_mm)}},
        "gripper": gripper,
    })


def measure(
    *,
    arm: str,
    hand: str,
    coupling_mm: float,
    planner_margin_mm: float,
    attach_spheres: int,
    guard_margin_mm: float,
    rotation_xyzw,
    random_n: int,
    seed: int,
    plates=None,
):
    """Judge one combination in both models and return its :class:`CombinationEvidence`.

    The two judgements are over the same pose set, from the same rule every other gate reads, so the difference
    between them is a property of the models rather than of two ways of choosing configurations.
    """
    from src.robot.safety._fcl_self_collision import composed_parts
    from src.robot.safety.planning.body_link import HandLink, coupling_bodies
    from src.robot.safety.planning.curobo_client import CuroboPlanClient
    from src.robot.safety.planning.evidence import CombinationEvidence, Measured, guard_sha256
    from src.robot.safety.planning.hand import planner_hand
    from src.robot.safety.planning.robot.retract_table import read_retract
    from src.robot.drivers.sim.robot_models import curobo_arm_descriptor

    cfg = _robot_config(arm, hand, coupling_mm, planner_margin_mm, rotation_xyzw, plates=plates)
    resolved = planner_hand(cfg)
    if not chosen(resolved):
        raise SystemExit(f"{arm} with {hand}: the cell names no hand, so there is no combination to measure")
    # The plates a cell declares, summed as its own planner_hand sums them: the one number every half below reads.
    coupling_mm = float(resolved.coupling_mm)
    link = HandLink.from_hand(resolved)
    # The link's own placement, which is the resolved hand's: from_hand refuses a hand its declared tool frame
    # places nowhere, so the one it built on is always chosen.
    placement = link.placement

    where = f"{placement.approach}{placement.closing}"
    retract = read_retract(arm, hand, float(coupling_mm), float(planner_margin_mm), placement=where)
    poses, kinds = pose_set(retract, random_n=random_n, seed=seed)

    # The guard's side, and the control. The clearance at the retract is measured again here, against the number the
    # committed table recorded for this same combination. Both came from the same ruler on the same geometry, so a
    # difference means something moved between the table and now, which makes it the cheapest check there is on
    # whether this gate is talking about the robot it thinks it is.
    judge = _exact_judge().ExactJudge(arm, hand, float(coupling_mm),
                                      placement if chosen(placement) else None)
    exact = [judge.nearest(pose)[0] > 0.0 for pose in poses]
    recorded = _recorded_clearance(arm, hand, float(coupling_mm), float(planner_margin_mm), placement=where)
    control_mm = abs(judge.nearest(retract)[0] - recorded) if recorded is not None else 0.0

    parts = composed_parts(arm, None, resolved.guard_variant, float(coupling_mm),
                           placement=placement if chosen(placement) else None,
                           coupling_boxes=resolved.coupling_boxes)

    # The context manager starts and closes it. A sidecar left running holds a GPU, and this gate is meant to be
    # called once per combination over a whole matrix.
    with CuroboPlanClient(
        robot_config=curobo_arm_descriptor(arm),
        self_collision_margin_mm=float(planner_margin_mm),
        body_links=[link.to_dict(), *coupling_bodies(resolved)],
        default_q=retract,
        measure_only=True,
        # The slots it records are the slots it starts with. A cell with payload spheres composes them into the
        # config its sidecar hashes, so a file measured without them could never admit it.
        attach_spheres=int(attach_spheres),
    ) as client:
        identity = client.identity
        explained = client.explain_joints(poses, name_pairs=True)

    # What ``pairs_named`` asks, asked of both rows, so that what follows reads them as chosen.
    if not chosen(explained.pairs) or not chosen(explained.depths_mm):
        raise SystemExit(
            f"{arm} with {hand}: the sidecar names no pair for any pose, so its descriptor carries no sphere "
            f"ownership. Every collision would read as an attribution disagreement, which is a number about the "
            f"descriptor's shape and not about this robot. Build the descriptor with build_ur_config.py and retry."
        )
    pairs = tuple(explained.pairs)
    depths = tuple(explained.depths_mm)
    collides = list(explained.self_collides)

    measured = Measured(
        poses=len(poses),
        poses_sha256=poses_sha256(poses),
        retract_clear=not collides[kinds.index("retract")],
        refused=sum(1 for hit in collides if hit),
        false_clears=false_clears(collides, exact),
        attribution_disagreements=attribution_disagreements(collides, pairs, depths),
        control_mm=float(control_mm),
    )
    return CombinationEvidence(
        arm=arm, hand=hand, coupling_mm=float(coupling_mm),
        approach=placement.approach, closing=placement.closing,
        planner_margin_mm=float(planner_margin_mm), attach_spheres=int(attach_spheres),
        guard_margin_mm=float(guard_margin_mm),
        composed_sha256=str(identity.composed_sha256) if chosen(identity.composed_sha256) else "",
        urdf_sha256=str(identity.urdf_sha256) if chosen(identity.urdf_sha256) else "",
        arm_descriptor_sha256=(str(identity.arm_descriptor_sha256)
                               if chosen(identity.arm_descriptor_sha256) else ""),
        hand_map_sha256=link.spheres_sha256,
        guard_sha256=guard_sha256(parts),
        measured=measured,
    )


def _recorded_clearance(
    arm: str, hand: str, coupling_mm: float, planner_margin_mm: float, *, placement: str,
) -> "float | None":
    """What the committed retract table measured at this combination's own pose, or ``None`` where it recorded none."""
    import yaml

    from src.robot.safety.planning.robot.retract_table import TABLE_PATH, placement_of

    table = yaml.safe_load(Path(TABLE_PATH).read_text(encoding="utf-8")) or {}
    for row in table.get("retracts") or []:
        if (row.get("arm") == arm and row.get("hand") == hand and placement_of(row, table) == placement
                and abs(float(row.get("plate_mm", 0.0)) - coupling_mm) <= 1e-6
                and abs(float(row.get("planner_margin_mm", 0.0)) - planner_margin_mm) <= 1e-6):
            exact = row.get("exact") or {}
            return None if exact.get("clearance_mm") is None else float(exact["clearance_mm"])
    return None


def parser() -> argparse.ArgumentParser:
    """The gate's command line, on its own so a printed command can be parsed back."""
    parser = argparse.ArgumentParser(description="Measure one arm and hand combination and write its evidence file.")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--hand", required=True)
    parser.add_argument("--coupling-mm", type=float, default=0.0)
    parser.add_argument("--planner-margin-mm", type=float, required=True,
                        help="the margin this cell plans at; a combination at another margin is another file")
    parser.add_argument("--attach", type=int, default=0, help="collision sphere slots reserved for a payload")
    parser.add_argument("--plate", type=_plate_spec, action="append", default=None,
                        help="one coupling plate as the cell declares it, NAME:THICKNESS_MM[:X_MM,Y_MM] (repeatable, "
                             "flange first); takes the place of --coupling-mm")
    parser.add_argument("--guard-margin-mm", type=float, default=_DEFAULT_GUARD_MARGIN_MM)
    parser.add_argument("--tool-rotation-xyzw", type=float, nargs=4, default=None, metavar=("X", "Y", "Z", "W"),
                        help="the cell's declared tool frame, four numbers x y z w as choose_ur_retract.py takes them. "
                             "Omitted, the frame is undeclared and the hand sits on its model's own axes, +Y+X")
    parser.add_argument("--random", type=int, default=None, help="how many seeded draws the pose set holds")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--write", action="store_true", help="write the file; without it nothing is written")
    parser.add_argument("--evidence-dir", default=None)
    return parser


def main(argv: "list[str] | None" = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.plate and args.coupling_mm:
        cli.error("--plate declares the plates, so --coupling-mm has nothing left to say: give one or the other")

    from _matrix_gate import RANDOM_POSES, SEED

    rotation = [float(v) for v in args.tool_rotation_xyzw] if args.tool_rotation_xyzw is not None else None

    evidence = measure(
        arm=args.arm, hand=args.hand, coupling_mm=args.coupling_mm,
        planner_margin_mm=args.planner_margin_mm, attach_spheres=args.attach,
        guard_margin_mm=args.guard_margin_mm, rotation_xyzw=rotation, plates=args.plate,
        random_n=RANDOM_POSES if args.random is None else args.random,
        seed=SEED if args.seed is None else args.seed,
    )
    print(f"[gate] {evidence.render()}")

    folder = Path(args.evidence_dir) if args.evidence_dir else None
    path = evidence.path if folder is None else folder / evidence.path.name
    if not args.write:
        print(f"[gate] nothing written. Pass --write to put this in {path}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(evidence.evidence_bytes())
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
