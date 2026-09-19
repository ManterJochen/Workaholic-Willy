"""Which pose an arm retracts to with one hand, chosen by a rule on the exact meshes and the spheres. Stdlib and PyYAML.

A retract is not a neutral label: cuRobo biases its graph search and its IK regularisation towards it and warms up from
it. Isaac's Lula ``default_q`` is not safe to take as one unchecked: on ur5 that pose keeps 9.5 mm to the Hand-E and
6.9 mm to the EGU-50, inside the guard's 10 mm margin, so the arm would start where its own guard refuses to be.

The rule: start at the Lula pose (the anchor), walk outwards in fixed steps on the five joints that move the hand, and
take the first candidate where the hand being judged clears the guard margin plus a reserve on the exact meshes, the
planner's sphere model admits it, the arm stands clear of the table, and every joint lies inside the planner's envelope.
There is one row per arm, hand, plate, planner margin and placement, because a pose that clears one hand can be a self
collision with another. Deterministic: one order, no random draw, first pass wins, so another box with the same engine
reproduces the committed table.

The pan is fixed because turning about the base moves no self distance and no height. The judging itself lives in the
interpreter that holds Coal and the bundles (``choose_ur_retract.py``); this module is the order, the rule and the
reader, so both interpreters share one implementation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
import sys
from pathlib import Path
from typing import Any


__all__ = [
    "ANCHORS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_STEP_RAD",
    "ENVELOPE_RESERVE_RAD",
    "RetractChoice",
    "RetractRow",
    "RetractMissing",
    "RetractRefused",
    "candidates",
    "choose",
    "read_arm_retract",
    "read_retract",
]

#: Isaac's Lula ``default_q`` per arm, with the file it is read from
#: (motion_policy_configs/universal_robots/{arm}/rmpflow/{arm}_robot_description.yaml).
ANCHORS: dict[str, tuple[float, ...]] = {
    "ur3": (0.0, -1.0, 0.9, 0.0, 0.0, 0.0),
    "ur3e": (0.0, -1.0, 0.9, 0.0, 0.0, 0.0),
    "ur5": (0.0, -1.0, 0.9, 0.0, 0.0, 0.0),
    "ur5e": (0.0, -1.0, 0.9, 0.0, 0.0, 0.0),
    "ur10": (-1.57, -1.57, -1.57, -1.57, 1.57, 0.0),
    "ur10e": (0.0, -1.2, 1.1, 0.0, 0.0, 0.0),
    "ur16e": (0.0, -1.2, 1.1, 0.0, 0.0, 0.0),
}

#: The step and how far the search may walk: 0.05 rad per step, six steps, on joints 2 to 6.
DEFAULT_STEP_RAD = 0.05
DEFAULT_MAX_STEPS = 6

#: How many candidate poses one question to the planner carries. The family is 371,293 poses on the full
#: ladder and the rule takes the first that passes everything, so asking in chunks costs a pair whose anchor
#: already works one round trip instead of ninety.
_PLANNER_CHUNK = 4096
#: How far inside the planner's envelope a retract must stay, so a cell that jogs a little does not leave it.
ENVELOPE_RESERVE_RAD = 0.05
#: The joints the search moves. Index 0, the pan, turns about the base axis and changes no distance and no height.
SEARCH_JOINTS = (1, 2, 3, 4, 5)


class RetractRefused(RuntimeError):
    """No candidate cleared every hand: this arm has no retract under the rule as parameterised."""


@dataclass(frozen=True)
class RetractChoice:
    """The pose the rule chose, how far it walked, and what every hand measured there."""

    arm: str
    retract: list[float]
    steps: tuple[int, ...]
    anchor: list[float]
    rows: list[dict[str, Any]] = field(default_factory=list)
    tried: int = 0


def candidates(
    anchor: "list[float] | tuple[float, ...]",
    *,
    step_rad: float = DEFAULT_STEP_RAD,
    max_steps: int = DEFAULT_MAX_STEPS,
    joints: "tuple[int, ...]" = SEARCH_JOINTS,
):
    """``(pose, steps)`` for every step vector, the anchor first, then outwards in one fixed order.

    The order is ``(how many joints moved, sum of squared steps, largest single step, the vector itself)``: every
    single joint turn first, then every pair, and so on. On ur5 the pair that refuses its anchor, the hand against
    wrist_1, is opened by one joint turning about 1.5 rad, and joints 1 to 4 do not change it at all because they move
    the whole wrist assembly rigidly. An order that takes the smallest combined perturbations first walks thousands of
    candidates that cannot help, at 15 to 90 ms each. Fewest joints first also keeps a chosen retract recognisable:
    one joint away from the pose the arm shipped with.
    """
    base = [float(v) for v in anchor]
    vectors = sorted(
        itertools.product(range(-max_steps, max_steps + 1), repeat=len(joints)),
        key=lambda k: (sum(1 for v in k if v), sum(v * v for v in k), max(abs(v) for v in k), k),
    )
    for steps in vectors:
        pose = list(base)
        for joint, step in zip(joints, steps):
            pose[joint] = base[joint] + step * step_rad
        yield pose, steps


def _inside(pose: "list[float]", envelope: "tuple[tuple[float, ...], tuple[float, ...]]") -> bool:
    lower, upper = envelope
    return all(
        low + ENVELOPE_RESERVE_RAD <= value <= high - ENVELOPE_RESERVE_RAD
        for value, low, high in zip(pose, lower, upper)
    )


def choose(
    arm: str,
    anchor: "list[float] | tuple[float, ...]",
    judges: "list[Any]",
    *,
    planner_envelope: "tuple[tuple[float, ...], tuple[float, ...]]",
    guard_margin_mm: float,
    reserve_mm: float,
    table_clearance_mm: float,
    step_rad: float = DEFAULT_STEP_RAD,
    max_steps: int = DEFAULT_MAX_STEPS,
    planner_admits: Any = None,
    progress: Any = None,
    progress_every: int = 2000,
) -> RetractChoice:
    """The first candidate the planner admits that also clears every judge, the table and the envelope.

    ``judges`` answer ``nearest(pose) -> (mm, pair)`` and ``lowest_z_mm(pose)``, one per hand, plate and placement. A
    candidate has to satisfy them all: the cell that carries the closest hand is the one it has to clear.

    ``planner_admits(poses) -> sequence of bool`` is the second judge, and the rule is wrong without it. A pose
    can clear the exact meshes by 13.4 mm while the planner's own sphere model calls it a self collision, and
    then the sidecar never becomes ready and the cell does not come up at all. It is asked once, over the whole
    candidate family, because each answer is a round trip to a process with a GPU in it and because that is what
    the planner's explain command is for.
    """
    if not judges:
        raise RetractRefused(f"{arm}: no hand to judge a retract against, so nothing was chosen")
    needed = float(guard_margin_mm) + float(reserve_mm)
    # The closest hand at the anchor is the one that will refuse most candidates, so it is asked first and the rest are
    # never asked for a pose it already rejects. It changes which judge a refusal names, never which pose is chosen.
    judges = sorted(judges, key=lambda judge: float(judge.nearest(list(anchor))[0]))
    #: The best any candidate managed, measured at the judge that stopped it: clearance, hand, pair and which rule bit.
    #: The largest clearance over all judges would say nothing, because the search fails at the hand that is closest.
    best: "tuple[float, str, str, str] | None" = None
    tried = 0

    family = [(pose, steps) for pose, steps in candidates(anchor, step_rad=step_rad, max_steps=max_steps)
              if _inside(pose, planner_envelope)]

    def admitted_in_order() -> "Any":
        """The family, in its own order, filtered by the planner where there is one.

        Asked in chunks rather than all at once: the family is 371,293 poses on the full ladder, each answer is a
        round trip to a process with a GPU in it, and the rule takes the first pose that passes everything. A pair
        whose anchor already works then costs one chunk instead of the whole ladder.
        """
        if planner_admits is None:
            yield from family
            return
        for start in range(0, len(family), _PLANNER_CHUNK):
            block = family[start:start + _PLANNER_CHUNK]
            verdicts = list(planner_admits([pose for pose, _ in block]))
            if len(verdicts) != len(block):
                raise RetractRefused(
                    f"{arm}: the planner answered about {len(verdicts)} of {len(block)} candidate poses, "
                    f"so nothing here knows which ones it admits"
                )
            for row, good in zip(block, verdicts):
                if good:
                    yield row

    seen_admitted = 0
    for pose, steps in admitted_in_order():
        seen_admitted += 1
        tried += 1
        rows: list[dict[str, Any]] = []
        limiting: "tuple[float, str, str, str] | None" = None
        for judge in judges:
            clearance, pair = judge.nearest(pose)
            if float(clearance) < needed:
                limiting = (float(clearance), str(judge.label), str(pair), "clearance")
                break
            # Only now: the height transforms every vertex of every mesh, and a candidate that already fails the
            # clearance never needs it. That is most of them, and the search is what the box spends its time on.
            height = judge.lowest_z_mm(pose)
            rows.append({
                "hand": judge.label, "clearance_mm": round(float(clearance), 3), "nearest_pair": pair,
                "lowest_z_mm": round(float(height), 3),
            })
            if float(height) < float(table_clearance_mm):
                limiting = (float(height), str(judge.label), str(pair), "table")
                break
        if limiting is None:
            return RetractChoice(arm=arm, retract=pose, steps=steps, anchor=[float(v) for v in anchor],
                                 rows=rows, tried=tried)
        if best is None or limiting[0] > best[0]:
            best = limiting
        if progress is not None and tried % progress_every == 0:
            progress(tried, steps, best)
    if planner_admits is not None and seen_admitted == 0:
        raise RetractRefused(
            f"{arm}: the planner's own sphere model refuses every one of the {len(family)} candidate poses, so this "
            f"arm and hand have no retract the sidecar would start from. The exact meshes are not the question "
            f"here: refit the spheres, or carry this pair as one that cannot plan."
        )
    worst = best or (float("nan"), "none", "none", "none")
    reason = ("kept {:.1f} mm to".format(worst[0]) if worst[3] == "clearance"
              else "stood {:.1f} mm above the table with".format(worst[0]))
    raise RetractRefused(
        f"{arm}: no pose within {max_steps} steps of {step_rad:g} rad clears every hand by {needed:g} mm and stands "
        f"{table_clearance_mm:g} mm above the table. The best candidate {reason} {worst[1]} at {worst[2]}. Widen the "
        "search on purpose, or refit the geometry."
    )


# ---- reading the table -------------------------------------------------------------------------------------
# One implementation, and it lives beside the table (src/robot/safety/planning/robot/retract_table.py) because three
# interpreters read that file and none of them can import the other two. Re-exported here so the scripts that ask this
# module keep asking it, and so there is no second copy to drift.
import importlib.util as _importlib_util  # noqa: E402

_TABLE_MODULE = Path(__file__).resolve().parents[2] / "src/robot/safety/planning/robot/retract_table.py"
_spec = _importlib_util.spec_from_file_location("willy_retract_table", _TABLE_MODULE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"no loader for the retract table reader at {_TABLE_MODULE}")
_table = _importlib_util.module_from_spec(_spec)
sys.modules["willy_retract_table"] = _table
_spec.loader.exec_module(_table)

RetractMissing = _table.RetractMissing
RetractRow = _table.RetractRow
read_arm_retract = _table.read_arm_retract
read_retract = _table.read_retract
