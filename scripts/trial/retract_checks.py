"""Adding a hand adds its retract rows and moves nothing, and two hands' exact guards agree where they must.

    .venv/Scripts/python.exe scripts/trial/retract_checks.py table-diff --before A.yaml --after B.yaml --added ur10e/acme_dims@0
    .venv/Scripts/python.exe scripts/trial/retract_checks.py guard-agree --arm ur10e --hand acme_mesh --reference robotiq_hande \\
        --coupling-mm 0 --tool-rotation-xyzw 0 0 0 1 --tolerance-mm 0.001

A trial instrument, not product code.

``table-diff`` keys rows the way a merge keys them (``retract_table.pair_key``), so a run that added a customer hand
may add exactly the declared pairs: every other row and refusal is byte for byte what it was, the rule is unchanged
but for refusals of declared pairs, and the arm's fallback retract did not move.

``guard-agree`` asks the exact guard of two hands the same question over one pose set: the reference hand's committed
retract at that placement, with the seeded random and sweep poses every gate judges. A hand written from a vendor STL of
the Hand-E must give the Hand-E's distances; a different hand must not, which is the control.

Exit codes: 0 yes, 1 no with every difference named, 2 the question cannot be asked.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO / "scripts" / "curobo"), str(REPO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _declared(added: "list[str]") -> "set[tuple[str, str, float]]":
    """``ARM/HAND@PLATE`` words as (arm, hand, plate)."""
    pairs = set()
    for word in added:
        arm_hand, _, plate = word.partition("@")
        arm, _, hand = arm_hand.partition("/")
        if not arm or not hand:
            raise ValueError(f"{word!r} is not ARM/HAND@PLATE")
        pairs.add((arm, hand, round(float(plate or 0.0), 6)))
    return pairs


@dataclass(frozen=True)
class TableDiff:
    added: tuple[str, ...]
    moved: tuple[str, ...]
    removed: tuple[str, ...]
    undeclared: tuple[str, ...]
    rule_changed: tuple[str, ...]
    fallback_moved: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return 0 if not (self.moved or self.removed or self.undeclared or self.rule_changed or self.fallback_moved) else 1

    def render(self) -> str:
        if not self.exit_code:
            return f"the table added {len(self.added)} declared entr{'y' if len(self.added) == 1 else 'ies'} and moved nothing"
        lines = ["the table moved more than it declared"]
        for label, names in (("moved", self.moved), ("removed", self.removed), ("undeclared", self.undeclared),
                             ("rule", self.rule_changed), ("fallback", self.fallback_moved)):
            lines += [f"  {label:<10} {name}" for name in names]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"added": list(self.added), "moved": list(self.moved), "removed": list(self.removed),
                "undeclared": list(self.undeclared), "rule_changed": list(self.rule_changed),
                "fallback_moved": list(self.fallback_moved), "exit_code": self.exit_code}


def table_diff(before: dict, after: dict, *, added: "set[tuple[str, str, float]]") -> TableDiff:
    from src.robot.safety.planning.robot.retract_table import pair_key, placement_of, read_arm_retract

    def entries(table: dict) -> dict:
        rule = table.get("rule") or {}
        out = {}
        for kind, rows in (("row", table.get("retracts") or []), ("refusal", rule.get("pairs_with_no_retract") or [])):
            for row in rows:
                out[(kind, *pair_key(row, placement_of(row, table)))] = row
        return out

    def declared(key: tuple) -> bool:
        return (key[1], key[2], key[3]) in added

    def said(key: tuple) -> str:
        return f"{key[0]} {key[1]} {key[2]} plate {key[3]:g} mm margin {key[4]:g} mm at {key[5]}"

    old, new = entries(before), entries(after)
    moved = [said(key) for key in old if key in new and old[key] != new[key] and not declared(key)]
    removed = [said(key) for key in old if key not in new]
    fresh = [key for key in new if key not in old]
    rule_before = {k: v for k, v in (before.get("rule") or {}).items() if k != "pairs_with_no_retract"}
    rule_after = {k: v for k, v in (after.get("rule") or {}).items() if k != "pairs_with_no_retract"}
    rule_changed = sorted(k for k in set(rule_before) | set(rule_after) if rule_before.get(k) != rule_after.get(k))

    fallback_moved = []
    arms = sorted({str(row.get("arm")) for row in (before.get("retracts") or [])})
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        paths = {}
        for name, table in (("before", before), ("after", after)):
            paths[name] = Path(scratch) / f"{name}.yaml"
            paths[name].write_text(yaml.safe_dump(table, sort_keys=False), encoding="utf-8")
        for arm in arms:
            try:
                a = read_arm_retract(paths["before"], arm).retract
                b = read_arm_retract(paths["after"], arm).retract
            except Exception as exc:  # noqa: BLE001 (an arm that lost its fallback is exactly what this names)
                fallback_moved.append(f"{arm}: {type(exc).__name__}: {exc}")
                continue
            if list(a) != list(b):
                fallback_moved.append(f"{arm}: {list(a)} became {list(b)}")
    return TableDiff(
        added=tuple(said(key) for key in fresh if declared(key)), moved=tuple(moved), removed=tuple(removed),
        undeclared=tuple(said(key) for key in fresh if not declared(key)), rule_changed=tuple(rule_changed),
        fallback_moved=tuple(fallback_moved),
    )


@dataclass(frozen=True)
class GuardAgreement:
    arm: str
    hand: str
    reference: str
    poses: int
    poses_sha256: str
    max_abs_diff_mm: float
    worst_pose: int
    pairs: tuple[str, str]
    tolerance_mm: float

    @property
    def exit_code(self) -> int:
        return 0 if self.max_abs_diff_mm <= self.tolerance_mm else 1

    def render(self) -> str:
        verdict = "agree" if not self.exit_code else "DISAGREE"
        return (f"{self.hand} and {self.reference} on {self.arm} {verdict}: largest difference "
                f"{self.max_abs_diff_mm:.6g} mm over {self.poses} poses (tolerance {self.tolerance_mm:g} mm), worst at "
                f"pose {self.worst_pose} ({self.pairs[0]} against {self.pairs[1]}); poses_sha256 {self.poses_sha256}")

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "hand": self.hand, "reference": self.reference, "poses": self.poses,
                "poses_sha256": self.poses_sha256, "max_abs_diff_mm": self.max_abs_diff_mm,
                "worst_pose": self.worst_pose, "pairs": list(self.pairs), "tolerance_mm": self.tolerance_mm,
                "exit_code": self.exit_code}


def guard_agree(*, arm: str, hand: str, reference: str, coupling_mm: float, rotation_xyzw: "tuple[float, ...]",
                random_n: "int | None" = None, tolerance_mm: float = 1e-3,
                planner_margin_mm: float = 4.0) -> GuardAgreement:
    from _matrix_gate import RANDOM_POSES, pose_set, poses_sha256
    from exact_self_collision_poses import ExactJudge

    from src.robot.safety.planning._hand_placement import HandPlacement
    from src.robot.safety.planning.robot.retract_table import read_retract

    placement = HandPlacement.from_quaternion_xyzw(rotation_xyzw)
    key = f"{placement.approach}{placement.closing}"
    retract = read_retract(arm, reference, float(coupling_mm), float(planner_margin_mm), placement=key)
    poses, _ = pose_set(list(retract),
                        random_n=RANDOM_POSES if random_n is None else int(random_n))
    ours = ExactJudge(arm, hand, float(coupling_mm), placement)
    theirs = ExactJudge(arm, reference, float(coupling_mm), placement)
    worst, index, pairs = -1.0, -1, ("", "")
    for i, pose in enumerate(poses):
        a, pair_a = ours.nearest(pose)
        b, pair_b = theirs.nearest(pose)
        if abs(a - b) > worst:
            worst, index, pairs = abs(a - b), i, (pair_a, pair_b)
    return GuardAgreement(arm=arm, hand=hand, reference=reference, poses=len(poses), poses_sha256=poses_sha256(poses),
                          max_abs_diff_mm=worst, worst_pose=index, pairs=pairs, tolerance_mm=float(tolerance_mm))


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Retract table and exact guard checks for the customer chain trial.")
    verbs = parser.add_subparsers(dest="verb", required=True)
    diff = verbs.add_parser("table-diff")
    diff.add_argument("--before", required=True)
    diff.add_argument("--after", required=True)
    diff.add_argument("--added", default="", help="ARM/HAND@PLATE[,ARM/HAND@PLATE...], the pairs the run was for")
    agree = verbs.add_parser("guard-agree")
    agree.add_argument("--arm", required=True)
    agree.add_argument("--hand", required=True)
    agree.add_argument("--reference", required=True)
    agree.add_argument("--coupling-mm", type=float, default=0.0)
    agree.add_argument("--tool-rotation-xyzw", type=float, nargs=4, required=True)
    agree.add_argument("--random", type=int, default=None, help="random poses; the gates' count when left out")
    agree.add_argument("--tolerance-mm", type=float, default=1e-3)
    args = parser.parse_args(argv)

    if args.verb == "table-diff":
        for name in (args.before, args.after):
            if not Path(name).is_file():
                print(f"cannot ask: no table at {name}", file=sys.stderr)
                return 2
        report: Any = table_diff(
            yaml.safe_load(Path(args.before).read_text(encoding="utf-8")),
            yaml.safe_load(Path(args.after).read_text(encoding="utf-8")),
            added=_declared([word for word in args.added.split(",") if word]),
        )
    else:
        from src.robot.safety.planning.environment import import_collision_engine

        if import_collision_engine()[1] is None:
            print("cannot ask: no collision engine (Coal or python-fcl) in this interpreter", file=sys.stderr)
            return 2
        report = guard_agree(arm=args.arm, hand=args.hand, reference=args.reference, coupling_mm=args.coupling_mm,
                             rotation_xyzw=tuple(args.tool_rotation_xyzw), random_n=args.random,
                             tolerance_mm=args.tolerance_mm)
    print(report.render())
    return int(report.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
