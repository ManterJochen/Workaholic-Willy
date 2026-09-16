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
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["RetractMissing", "RetractRow", "TABLE_PATH", "read_arm_retract", "read_retract"]

#: The committed table, beside this module.
TABLE_PATH = Path(__file__).resolve().with_name("ur_retract.yaml")

_GENERATE = ("Generate it in the project venv with Coal and a GPU: "
             ".venv/Scripts/python.exe scripts/curobo/choose_ur_retract.py {arm}")


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


def read_arm_retract(table_path: "Path | str" = TABLE_PATH, arm: str = "ur5e") -> RetractRow:
    """The arm's fallback retract, from the first row written for it, with every pair the table judged."""
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
    return RetractRow(
        arm=arm,
        retract=[float(v) for v in rows[0]["retract"]],
        steps=[int(v) for v in rows[0].get("steps") or ()],
        anchor=[float(v) for v in rows[0].get("anchor") or ()],
        anchor_source=str(rows[0].get("anchor_source", "")),
        hands=hands,
        table_sha256=hashlib.sha256(raw).hexdigest(),
    )


def read_retract(
    arm: str,
    hand: str,
    plate_mm: float,
    planner_margin_mm: "float | None" = None,
    *,
    table_path: "Path | str" = TABLE_PATH,
) -> list[float]:
    """The pose this pair was judged at, or :class:`RetractMissing` saying which half is missing."""
    table, _ = _load(table_path, arm)
    name = Path(table_path).name
    wanted = [row for row in _rows(table)
              if row.get("arm") == arm and row.get("hand") == hand
              and abs(float(row.get("plate_mm", 0.0)) - float(plate_mm)) <= 1e-6]
    if not wanted:
        refused = [row for row in (table.get("rule") or {}).get("pairs_with_no_retract") or []
                   if row.get("arm") == arm and row.get("hand") == hand]
        if refused:
            raise RetractMissing(
                f"{name} found no retract for {arm} with {hand}: {refused[0].get('reason', '')} This pair cannot "
                f"start a planner until that is fixed."
            )
        raise RetractMissing(
            f"{name} judged no retract for {arm} with {hand} at a {float(plate_mm):g} mm plate. It holds "
            f"{_pairs(table)}. {_GENERATE.format(arm=arm)}"
        )
    if planner_margin_mm is not None:
        at_margin = [row for row in wanted
                     if abs(float(row.get("planner_margin_mm", 0.0)) - float(planner_margin_mm)) <= 1e-6]
        if not at_margin:
            judged = sorted({float(row.get("planner_margin_mm", 0.0)) for row in wanted})
            raise RetractMissing(
                f"{name} judged {arm} with {hand} in a planner keeping {judged} mm, and this cell keeps "
                f"{float(planner_margin_mm):g} mm: a pose the planner admits at one clearance it can refuse at "
                f"another. {_GENERATE.format(arm=arm)}"
            )
        wanted = at_margin
    return [float(v) for v in wanted[0]["retract"]]
