"""Reading the committed retract table. Stdlib plus PyYAML, beside the table it reads.

The table (``ur_retract.yaml``) holds one retract per arm, hand, plate and planner margin, chosen by
the rule in ``scripts/curobo/choose_ur_retract.py``. Three interpreters need to read it and none of
them can import the other two: the project venv (the drivers, which hand the pose to the sidecar),
the cuRobo environment (the descriptor builder, which writes the arm's fallback), and the measuring
scripts. So the reader lives here, next to the file, with nothing but the standard library and
PyYAML under it, and is loaded by path where it cannot be imported.

The key is a pair and not an arm. A descriptor is per arm and the hand arrives as a body link when
the sidecar starts, so a retract written into the descriptor can only ever be right about a bare
arm. At ur3's own retract the exact meshes clear the Hand-E by 13.4 mm and the planner's sphere
model calls the same pose a self collision, so that sidecar never becomes ready and the cell does
not come up at all. One wrist turn away it is fine. The pose that works belongs to the pair.

No fallback, anywhere. A pose judged with another hand, at another plate, or in a planner keeping a
different clearance, is not this cell's pose, and the rule exists to remove exactly that kind of
substitution.

The placement is part of the key. The rule's own placement holds the hand along flange +Y, the Isaac
cell's frame, and a real UR flange puts it along +Z: the same joints then hold the hand somewhere
else, and a pose that clears it along one axis can put it through the wrist along the other. A row
judged at another placement says so in its own ``placement``; a row that names none was judged at the
rule's. :func:`merge_judged` puts one run of the rule into the table without touching the rows it did
not judge.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

__all__ = ["FALLBACK_HAND", "RetractChange", "RetractMissing", "RetractRow", "SHARED_RULE_KEYS", "TABLE_PATH",
           "merge_judged", "pair_key", "placement_of", "read_arm_retract", "read_retract", "retract_changes"]

#: The committed table, beside this module.
TABLE_PATH = Path(__file__).resolve().with_name("ur_retract.yaml")

_GENERATE = ("Generate it in the project venv with Coal and a GPU: "
             ".venv/Scripts/python.exe scripts/curobo/choose_ur_retract.py {arm}")

#: The hand whose row at a bare flange, at the table's own placement, is an arm descriptor's fallback
#: retract: the hand every committed arm bundle carries (``hand.ARM_BUNDLE_HAND``). Chosen by rule,
#: because the first row written for an arm moves the moment a customer hand sorts before it.
FALLBACK_HAND = "robotiq_2f85"

#: The rule parameters every row of one table was judged under, whatever placement it was judged at.
#: The planner margin and the plate are not among them: every row carries both and a read filters by
#: both, so a customer cell at another margin or on another plate goes in beside the rows it does not
#: contradict.
SHARED_RULE_KEYS = ("guard_margin_mm", "reserve_mm", "table_clearance_mm", "step_rad", "max_steps", "engine")


class RetractMissing(LookupError):
    """The table judged no retract for what was asked, and there is nothing to fall back to."""


@dataclass(frozen=True)
class RetractRow:
    """One arm's fallback retract, and the pairs the table judged for it.

    The fallback is what a descriptor carries and what a sidecar with no hand would start from. No
    real cell is in that state: a cell names a hand, and the driver hands the sidecar that pair's
    own pose.
    """

    arm: str
    retract: list[float]
    steps: list[int]
    anchor: list[float]
    anchor_source: str
    #: hand -> the plates the table judged it at, in millimetres.
    hands: dict[str, list[float]]
    table_sha256: str

    def render(self) -> str:
        pairs = ", ".join(f"{hand} at {plates} mm" for hand, plates in sorted(self.hands.items()))
        return f"{self.arm} falls back to {[round(v, 4) for v in self.retract]}; judged pairs: {pairs or 'none'}"

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "retract": list(self.retract), "steps": list(self.steps),
                "anchor": list(self.anchor), "anchor_source": self.anchor_source,
                "hands": {hand: list(plates) for hand, plates in self.hands.items()},
                "table_sha256": self.table_sha256}


def _load(table_path: "Path | str", arm: str) -> "tuple[dict, bytes]":
    path = Path(table_path)
    if not path.is_file():
        raise RetractMissing(f"no retract table at {path}. {_GENERATE.format(arm=arm)}")
    raw = path.read_bytes()
    return yaml.safe_load(raw.decode("utf-8")) or {}, raw


def _rows(table: dict) -> list[dict]:
    rows = table.get("retracts")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _pairs(table: dict) -> str:
    return ", ".join(sorted({f"{row['arm']}/{row['hand']}" for row in _rows(table)})) or "nothing"


def retract_command(arm: str, hand: str, plate_mm: float, planner_margin_mm: "float | None", placement: str) -> str:
    """The whole ``choose_ur_retract.py`` command that judges exactly this pair at this placement."""
    import importlib.util
    import sys

    name = "willy_hand_placement_for_retract"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "_hand_placement.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    rotation = sys.modules[name].placement_quaternion_xyzw(placement[:2], placement[2:])
    parts = [".venv/Scripts/python.exe", "scripts/curobo/choose_ur_retract.py", arm, "--hand", hand]
    if float(plate_mm):
        parts += ["--plate-mm", f"{float(plate_mm):g}"]
    if planner_margin_mm is not None:
        parts += ["--planner-margin-mm", f"{float(planner_margin_mm):g}"]
    parts += ["--tool-rotation-xyzw", *(repr(v) for v in rotation)]
    return " ".join(parts)


def placement_of(row: dict, table: dict) -> str:
    """Where the hand sat when ``row`` was judged: its own ``placement``, or the rule's for a row that names none."""
    return str(row.get("placement") or (table.get("rule") or {}).get("placement") or "")


def read_arm_retract(table_path: "Path | str" = TABLE_PATH, arm: str = "ur5e") -> RetractRow:
    """The arm's fallback retract, with every pair the table judged.

    The fallback is the :data:`FALLBACK_HAND` row at a 0 mm plate, at the table's own placement and,
    where the rule names one, at its planner margin. It is a rule, so a hand added to the table never
    moves an arm descriptor's ``default_q``.
    """
    table, raw = _load(table_path, arm)
    rows = [row for row in _rows(table) if row.get("arm") == arm]
    if not rows:
        raise RetractMissing(
            f"{Path(table_path).name} judged no retract for {arm!r}: it holds {_pairs(table)}. "
            f"{_GENERATE.format(arm=arm)}"
        )
    hands: dict[str, list[float]] = {}
    for row in rows:
        hands.setdefault(str(row["hand"]), []).append(float(row.get("plate_mm", 0.0)))
    rule = table.get("rule") or {}
    candidates = [row for row in rows if row.get("hand") == FALLBACK_HAND and float(row.get("plate_mm", 0.0)) == 0.0
                  and placement_of(row, table) == str(rule.get("placement") or placement_of(row, table))]
    if rule.get("planner_margin_mm") is not None:
        at_margin = [row for row in candidates
                     if abs(float(row.get("planner_margin_mm", 0.0)) - float(rule["planner_margin_mm"])) <= 1e-6]
        candidates = at_margin or candidates
    if not candidates:
        raise RetractMissing(
            f"{Path(table_path).name} holds no {FALLBACK_HAND} row for {arm} at a bare flange at its own "
            f"placement, and that row is an arm descriptor's fallback retract. Generate it: "
            f".venv/Scripts/python.exe scripts/curobo/choose_ur_retract.py {arm} --hand {FALLBACK_HAND}, or build "
            f"a seed descriptor with scripts/curobo/build_ur_config.py {arm} --retract-from-anchor"
        )
    fallback = candidates[0]
    return RetractRow(
        arm=arm,
        retract=[float(v) for v in fallback["retract"]],
        steps=[int(v) for v in fallback.get("steps") or ()],
        anchor=[float(v) for v in fallback.get("anchor") or ()],
        anchor_source=str(fallback.get("anchor_source", "")),
        hands=hands,
        table_sha256=hashlib.sha256(raw).hexdigest(),
    )


def read_retract(
    arm: str,
    hand: str,
    plate_mm: float,
    planner_margin_mm: "float | None" = None,
    *,
    placement: str,
    table_path: "Path | str" = TABLE_PATH,
) -> list[float]:
    """The pose this pair was judged at, or :class:`RetractMissing` saying which half is missing.

    ``placement`` is where the cell puts the hand, spelled as approach and closing, for example ``+Y+X``.
    """
    table, _ = _load(table_path, arm)
    name = Path(table_path).name
    pair = [row for row in _rows(table)
            if row.get("arm") == arm and row.get("hand") == hand
            and abs(float(row.get("plate_mm", 0.0)) - float(plate_mm)) <= 1e-6]
    wanted = [row for row in pair if placement_of(row, table) == placement]
    if not wanted:
        refused = [row for row in (table.get("rule") or {}).get("pairs_with_no_retract") or []
                   if row.get("arm") == arm and row.get("hand") == hand and placement_of(row, table) == placement]
        if refused:
            raise RetractMissing(
                f"{name} found no retract for {arm} with {hand} placed {placement}: {refused[0].get('reason', '')} "
                f"This pair cannot start a planner until that is fixed."
            )
        if pair:
            placed = sorted({placement_of(row, table) or 'no placement' for row in pair})
            raise RetractMissing(
                f"{name} judged {arm} with {hand} with the hand placed {', '.join(placed)}, and this cell places it "
                f"{placement}: the same joints hold the hand somewhere else, and a pose that clears it along one "
                f"flange axis can put it through the wrist along another. Judge it in the project venv with Coal and a "
                f"GPU: {retract_command(arm, hand, plate_mm, planner_margin_mm, placement)}"
            )
        raise RetractMissing(
            f"{name} judged no retract for {arm} with {hand} at a {float(plate_mm):g} mm plate. It holds "
            f"{_pairs(table)}. Judge it in the project venv with Coal and a GPU: "
            f"{retract_command(arm, hand, plate_mm, planner_margin_mm, placement)}"
        )
    if planner_margin_mm is not None:
        at_margin = [row for row in wanted
                     if abs(float(row.get("planner_margin_mm", 0.0)) - float(planner_margin_mm)) <= 1e-6]
        if not at_margin:
            judged = sorted({float(row.get("planner_margin_mm", 0.0)) for row in wanted})
            raise RetractMissing(
                f"{name} judged {arm} with {hand} in a planner keeping {judged} mm, and this cell keeps "
                f"{float(planner_margin_mm):g} mm: a pose the planner admits at one clearance it can refuse at "
                f"another. Judge it in the project venv with Coal and a GPU: "
                f"{retract_command(arm, hand, plate_mm, planner_margin_mm, placement)}"
            )
        wanted = at_margin
    return [float(v) for v in wanted[0]["retract"]]


def _stamped(row: dict, placement: str, rotation: Any) -> dict:
    """``row`` with its placement written beside its margin, for a row judged at another placement than the rule's."""
    out: dict = {}
    for key, value in row.items():
        if key in ("placement", "tool_rotation_xyzw"):
            continue
        out[key] = value
        if key == "planner_margin_mm":
            out["placement"] = placement
            out["tool_rotation_xyzw"] = [float(v) for v in rotation]
    if "placement" not in out:
        out["placement"] = placement
        out["tool_rotation_xyzw"] = [float(v) for v in rotation]
    return out


def pair_key(row: dict, placement: str) -> tuple:
    """The pair a retract row or refusal is for, as a merge keys it: arm, hand, plate, planner margin and placement.

    Public so a check that asks whether a run moved a row keys rows the way the merge does, not a second way.
    """
    return (str(row.get("arm")), str(row.get("hand")), round(float(row.get("plate_mm", 0.0)), 6),
            round(float(row.get("planner_margin_mm", 0.0)), 6), placement)


def merge_judged(
    existing: dict, fresh: dict, *, not_asked: Sequence[tuple[str, str, float, float]] = (),
) -> dict:
    """``existing`` with one run of the rule put in, replacing exactly the pairs the run judged.

    ``fresh`` is what one run of ``choose_ur_retract.py`` judged: rows and refusals naming no placement,
    and a ``rule`` that names the run's. A pair is (arm, hand, plate, planner margin) at the run's
    placement. The run replaces an existing row or refusal only for a pair it judged: it found a
    retract, or the rule refused every candidate. A pair the planner could not be asked about
    (``not_asked``, each also a refusal in ``fresh``) is no judgement: it never replaces what the table
    holds, and is added only where the table holds nothing for that pair. Everything else is kept:
    another hand, another plate, another margin, another placement.

    A run at the table's own placement rewrites the rule, but keeps the old planner margin while rows at
    it remain and lists every plate either judged; a run at another placement keeps the rule and writes
    the placement into each of its entries. Rows keep their order, and a new one lands at the end of its
    placement's block, the rule's placement first.

    Refused (``ValueError``) where entries judged under one rule would be kept beside entries judged
    under another, naming both values; where ``fresh`` carries an entry of another placement; and where
    ``not_asked`` names a pair the run recorded no refusal for.
    """
    fresh_rule = dict(fresh.get("rule") or {})
    here = str(fresh_rule.get("placement") or "")
    if not here:
        raise ValueError("a run of the retract rule has to say which placement it judged")
    fresh_refusals = [row for row in fresh_rule.get("pairs_with_no_retract") or [] if isinstance(row, dict)]
    foreign = sorted({str(row.get("placement")) for row in [*_rows(fresh), *fresh_refusals]
                      if row.get("placement") and row.get("placement") != here})
    if foreign:
        raise ValueError(f"a run at {here} carries rows judged at {', '.join(foreign)}: one run of the rule judges one placement")
    old_rule = dict(existing.get("rule") or {})
    base = str(old_rule.get("placement") or here)
    placed = {"rule": {"placement": base}}

    def key(row: dict, placement: str | None = None) -> tuple:
        return pair_key(row, placement or placement_of(row, placed))

    unasked = {(str(a), str(h), round(float(p), 6), round(float(m), 6), here) for a, h, p, m in not_asked}
    recorded = {key(row, here) for row in fresh_refusals}
    stray = sorted(unasked - recorded)
    if stray:
        raise ValueError(f"not_asked names {stray}, and the run recorded no refusal for them")
    judged = {key(row, here) for row in _rows(fresh)} | (recorded - unasked)

    old_refusals = [row for row in old_rule.get("pairs_with_no_retract") or [] if isinstance(row, dict)]
    kept_rows = [row for row in _rows(existing) if key(row) not in judged]
    kept_refusals = [row for row in old_refusals if key(row) not in judged]
    if kept_rows or kept_refusals:
        differs = [f"{name} (the table {old_rule[name]!r}, this run {fresh_rule[name]!r})" for name in SHARED_RULE_KEYS
                   if name in old_rule and name in fresh_rule and old_rule[name] != fresh_rule[name]]
        if differs:
            raise ValueError(
                f"the table keeps entries this run did not judge, under a rule that differs in {', '.join(differs)}: "
                f"judge those entries again under this rule, or run it with the table's own parameters"
            )
    rotation = fresh_rule.get("tool_rotation_xyzw") or []

    def stamp(row: dict) -> dict:
        return dict(row) if here == base else _stamped(row, here, rotation)

    fresh_rows = {key(row, here): stamp(row) for row in _rows(fresh)}
    rows: list[dict] = []
    used: set = set()
    for row in _rows(existing):
        if key(row) not in judged:
            rows.append(row)
        elif key(row) in fresh_rows and key(row) not in used:
            rows.append(fresh_rows[key(row)])
            used.add(key(row))
    rows.extend(row for pair, row in fresh_rows.items() if pair not in used)

    held = {key(row) for row in [*rows, *kept_refusals]}
    refusals = list(kept_refusals)
    for row in fresh_refusals:
        pair = key(row, here)
        if pair in unasked and pair in held:
            continue
        refusals.append(stamp(row))

    order: list[str] = [base]
    for row in [*rows, *refusals]:
        if placement_of(row, placed) not in order:
            order.append(placement_of(row, placed))

    def rank(row: dict) -> int:
        return order.index(placement_of(row, placed))

    if here == base:
        rule = dict(fresh_rule)
        old_margin = old_rule.get("planner_margin_mm")
        if old_margin is not None and any(round(float(row.get("planner_margin_mm", 0.0)), 6) == round(float(old_margin), 6)
                                          for row in rows):
            rule["planner_margin_mm"] = old_margin
    else:
        rule = old_rule
    plates = sorted({float(v) for v in [*(old_rule.get("plates_mm_judged_for_mounting_face_hands") or []),
                                        *(fresh_rule.get("plates_mm_judged_for_mounting_face_hands") or [])]})
    if "plates_mm_judged_for_mounting_face_hands" in rule or plates:
        rule["plates_mm_judged_for_mounting_face_hands"] = plates
    rule["pairs_with_no_retract"] = sorted(refusals, key=rank)
    return {"rule": rule, "retracts": sorted(rows, key=rank)}


@dataclass(frozen=True)
class RetractChange:
    """A committed pair whose retract a run moved, or refused, and the evidence files that named that pose."""

    arm: str
    hand: str
    plate_mm: float
    planner_margin_mm: float
    placement: str
    before: "list[float] | None"
    #: The new pose, or ``None`` where the run refused the pair.
    after: "list[float] | None"
    evidence: "tuple[str, ...]"

    def render(self) -> str:
        what = "is refused now" if self.after is None else f"moved from {self.before} to {self.after}"
        files = (f"; the committed evidence {', '.join(self.evidence)} measured the old pose and refuses at the next planner "
                 f"start until it is measured again" if self.evidence else "; no committed evidence file names it")
        return (f"{self.arm} with {self.hand} at a {self.plate_mm:g} mm plate, {self.placement}, a {self.planner_margin_mm:g} mm "
                f"planner margin: the retract {what}{files}")

    def to_dict(self) -> "dict[str, Any]":
        return {"arm": self.arm, "hand": self.hand, "plate_mm": self.plate_mm, "planner_margin_mm": self.planner_margin_mm,
                "placement": self.placement, "before": self.before, "after": self.after, "evidence": list(self.evidence)}


def retract_changes(existing: dict, merged: dict, *, evidence_dir: "Path | str | None" = None) -> "list[RetractChange]":
    """Every pair of ``existing`` whose retract ``merged`` moved or dropped, with the evidence files named after it.

    A retract is ``default_q`` inside the composed config a sidecar hashes, so an evidence file measured at
    the old pose refuses on ``composed_sha256`` at the next start. Said when the table is written.
    """
    folder = Path(evidence_dir) if evidence_dir is not None else TABLE_PATH.parent / "evidence"

    def keyed(table: dict) -> "dict[tuple, dict]":
        placed = {"rule": {"placement": (table.get("rule") or {}).get("placement")}}
        return {(str(r.get("arm")), str(r.get("hand")), round(float(r.get("plate_mm", 0.0)), 6),
                 round(float(r.get("planner_margin_mm", 0.0)), 6), placement_of(r, placed)): r for r in _rows(table)}

    before, after = keyed(existing), keyed(merged)
    out: list[RetractChange] = []
    for pair, row in before.items():
        new = after.get(pair)
        old_pose = [float(v) for v in row.get("retract") or []]
        new_pose = [float(v) for v in new.get("retract") or []] if new is not None else None
        if new_pose == old_pose:
            continue
        arm, hand, plate, margin, placement = pair
        pattern = f"{arm}_{hand}_c{plate:g}mm_{placement}_m{margin:g}mm_a*.json"
        files = tuple(sorted(path.name for path in folder.glob(pattern))) if folder.is_dir() else ()
        out.append(RetractChange(arm=arm, hand=hand, plate_mm=plate, planner_margin_mm=margin, placement=placement,
                                 before=old_pose, after=new_pose, evidence=files))
    return out
