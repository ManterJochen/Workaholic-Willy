"""An arm description states where its retract search starts, and the committed table agrees (P0f F6g, owner D4).

The anchor is Isaac's Lula ``default_q`` for the arm, the pose every descriptor carried before the retract rule (UM5):
``scripts/arms/retract.py`` starts its search there and writes the anchor and its source into all 54 rows of
``retract.yaml``, whose rows are hashed into every evidence file through ``arm_rows_sha256``. So the values are
compared by ``float.hex``, where a sign and a last bit count: Isaac's own files spell ur10e's pan joint and ur16e's
wrist 3 as ``-0.0``, and a ``-0.0`` copied in would survive into the table and move eight evidence files that an
``==`` cannot tell apart. Each description writes ``0.0``, and ur10's ``-1.57`` stays as Isaac wrote it.

Since P0f F7c the rule reads the anchor and its source from the description; what stays here holds the committed
table and the raw file to it.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import yaml

from src.config.arms import available_arms, load_arm

_ROOT = Path(__file__).resolve().parents[3]
_TABLE = _ROOT / "src" / "robot" / "safety" / "planning" / "robot" / "retract.yaml"
_ARMS = _ROOT / "config" / "arms"
#: The hands whose rows this repository ships. A customer's cell judges its own hand into the same table (the
#: customer-hand trial adds three rows), so the shipped rows may not be missing and more may be there, and every row
#: there is held to its arm's anchor.
_SHIPPED_HANDS = frozenset({"robotiq_2f85", "robotiq_hande", "schunk_egu50"})


def _hex(values: "tuple[float, ...] | list[float]") -> list[str]:
    return [float(value).hex() for value in values]


def _anchor(model: str) -> "tuple[float, ...] | None":
    joints = load_arm(model).joints
    if all(joint.retract_anchor_rad is None for joint in joints):
        return None
    return tuple(float(joint.retract_anchor_rad) for joint in joints)  # type: ignore[arg-type]


class TheTableCarriesTheDescriptionsAnchorTests(unittest.TestCase):
    """Permanent: every committed retract row names its arm's anchor and source, bit for bit."""

    def test_every_row_carries_its_arms_anchor_and_source(self) -> None:
        rows = yaml.safe_load(_TABLE.read_text(encoding="utf-8"))["retracts"]
        shipped = [row for row in rows if row["hand"] in _SHIPPED_HANDS]
        self.assertEqual(54, len(shipped))
        self.assertEqual(7, len({row["arm"] for row in shipped}))
        wrong = [f"{row['arm']} {row['hand']} plate {row['plate_mm']}" for row in rows
                 if _hex(row["anchor"]) != _hex(_anchor(row["arm"]) or ())
                 or row["anchor_source"] != load_arm(row["arm"]).retract_anchor_source]
        self.assertEqual([], wrong)

    def test_every_anchor_in_the_raw_file_is_a_float(self) -> None:
        """``descriptor.py`` reads the file by path in the cuRobo environment, with no schema to turn an int into a float."""
        for model in available_arms():
            joints = yaml.safe_load((_ARMS / f"{model}.yaml").read_text(encoding="utf-8"))["arm"]["joints"]
            with self.subTest(model=model):
                kinds = {type(joint["retract_anchor_rad"]) for joint in joints if "retract_anchor_rad" in joint}
                self.assertLessEqual(kinds, {float})

    def test_a_signed_zero_and_a_last_bit_are_seen(self) -> None:
        """THE CONTROLS: ``==`` calls both of these equal, and the comparison above does not."""
        self.assertEqual(-0.0, 0.0)
        self.assertNotEqual(_hex([-0.0]), _hex([0.0]))
        anchor = _anchor("ur10e")
        assert anchor is not None
        moved = (math.nextafter(anchor[0], 1.0),) + anchor[1:]
        self.assertNotEqual(_hex(anchor), _hex(moved))


if __name__ == "__main__":
    unittest.main()
