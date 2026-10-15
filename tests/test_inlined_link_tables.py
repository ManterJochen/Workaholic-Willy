"""Every copy of an arm's collision links equals what its description says (P0f F6c, owner decision D5).

The links a joint moves, the bundle part each one's arrays are baked under, and the DH frame each is baked in were
written out by hand in eight tables across the build scripts and the runtime, in five shapes, all model-independent,
and no test tied any two of them together. Each description now carries ``link`` and ``part`` on every joint row; the
frame is the row's number and the flange frame is ``dof``, so neither can be mistyped. Since P0f F7a the bake, the
comparator, the sphere-map ruler, the exact pose tool and the fidelity probe read the description, since F8 the fit
plan takes its frames from it too, and since F9 the runtime's links, flange frames and body link names; this test
holds the copy left, the UR adapter's own chain, to it, and the data half holds the committed bundles and sphere maps to
it for good.

Nothing reads the keys yet, so this commit moves no byte of any artefact.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.config.arms import available_arms, load_arm
from src.config.schema.arms import arm_schema
from src.robot.safety.planning import _hand_bundle

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "src" / "robot" / "safety" / "data"
_MAPS = _ROOT / "src" / "robot" / "safety" / "planning" / "robot"


def _assigned(relative: str, name: str) -> ast.expr:
    """The expression ``name`` is assigned in the module at ``relative``, read without importing it."""
    for node in ast.parse((_ROOT / relative).read_text(encoding="utf-8")).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(getattr(target, "id", "") == name for target in targets) and node.value is not None:
            return node.value
    raise AssertionError(f"{relative} assigns no {name}")


def _literal(relative: str, name: str) -> object:
    return ast.literal_eval(_assigned(relative, name))


def _the_copies() -> dict[str, object]:
    """Each copy, normalised to (part, link, frame) rows or to the fields it holds."""
    chain = _literal("scripts/arms/_ur_description.py", "_CHAIN")
    return {
        "scripts/arms/_ur_description.py _CHAIN": [(joint, part, child) for joint, part, _parent, child
                                                   in chain],  # type: ignore[union-attr]
        # Willy's own: the planner links the exact guard decides, a choice measured on URs (band.py).
        "src/robot/safety/planning/band.py _ARM_PARTS": _literal("src/robot/safety/planning/band.py", "_ARM_PARTS"),
    }


def _what_the_description_says(model: str) -> dict[str, object]:
    arm = load_arm(model)
    rows = list(arm.collision_links)
    return {
        "scripts/arms/_ur_description.py _CHAIN": [(joint.name, joint.part, joint.link) for joint in arm.joints],
        # The shoulder_link is the planner's model of the base too, and no guard part holds the base (band.py).
        "src/robot/safety/planning/band.py _ARM_PARTS": {link: (part,) for part, link, _frame in rows[1:]},
    }


class EveryCopyIsTheDescriptionTests(unittest.TestCase):
    def test_every_inlined_table_equals_every_description(self) -> None:
        copies = _the_copies()
        for model in available_arms():
            said = _what_the_description_says(model)
            for name, copy in copies.items():
                with self.subTest(model=model, copy=name):
                    self.assertEqual(said[name], copy)

    def test_the_frame_is_the_row_and_the_flange_is_dof(self) -> None:
        for model in available_arms():
            arm = load_arm(model)
            with self.subTest(model=model):
                self.assertEqual(list(range(1, arm.dof + 1)), [frame for _part, _link, frame in arm.collision_links])

    def test_a_ur_link_is_its_part_with_link_after_it(self) -> None:
        """The rule ``descriptor.py`` and ``_arm_spheres.py`` apply instead of a table (``f"{body}_link"``)."""
        for model in available_arms():
            for joint in load_arm(model).joints:
                with self.subTest(model=model, joint=joint.name):
                    self.assertEqual(f"{joint.part}_link", joint.link)

    def test_a_part_may_not_be_a_hand_part(self) -> None:
        """The schema cannot import the safety layer, so its list of hand parts is tied to the one the bundle uses."""
        self.assertEqual(_hand_bundle.HAND_PARTS, arm_schema._HAND_PARTS)

    def test_a_swapped_or_renamed_link_is_seen(self) -> None:
        """THE CONTROL: the comparison above sees one swap and one link renamed."""
        name = "scripts/arms/_ur_description.py _CHAIN"
        rows = list(_what_the_description_says("ur5e")[name])  # type: ignore[call-overload]
        self.assertEqual(_the_copies()[name], rows)
        swapped = list(rows)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        self.assertNotEqual(_the_copies()[name], swapped)
        renamed = [(joint, part, f"{link}_x" if part == "wrist_1" else link) for joint, part, link in rows]
        self.assertNotEqual(_the_copies()[name], renamed)


_PARTS = frozenset({"shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"})
_LINKS = frozenset(f"{part}_link" for part in _PARTS)
_LITERALS = (ast.Tuple, ast.List, ast.Set, ast.Dict)

#: Each copy above, by the file and the name it is assigned to.
_COPIES = {"scripts/arms/_ur_description.py _CHAIN", "src/robot/safety/planning/band.py _ARM_PARTS"}
#: Literals that hold the six links or parts and are not copies of a description, each with the reason.
_NOT_COPIES = {
    "scripts/arms/fit.py ARM_PLAN": "the fit's budget per part, whose frames are the description's since P0f F8",
    "scripts/curobo/fit_planning_spheres.py ARM_PLAN": "the planning fit's budget per part, framed as the fit's",
    "scripts/arms/_ur_description.py JOINTS": "the UR package's own kinematics block names, which the adapter reads",
    "scripts/arms/_ur_description.py LINKS": "the UR package's mesh keys, base included",
    "scripts/arms/_ur_description.py def dh_rows_from": "the checks of UR's kinematics file, block by block",
    "scripts/arms/_urdf_source.py _REQUIRED_LINKS": "what a URDF from the UR adapter must carry",
    "src/willy_sim/run_suction_pick.py def _klt_probe_booted_cell": "an Isaac runner that only ever stands a UR5e",
}


def _where(statement: ast.stmt) -> str:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return f"def {statement.name}"
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        return getattr(statement.targets[0], "id", f"line {statement.lineno}")
    if isinstance(statement, ast.AnnAssign):
        return getattr(statement.target, "id", f"line {statement.lineno}")
    return f"line {statement.lineno}"


def _holds_the_links(node: ast.AST) -> bool:
    if not isinstance(node, _LITERALS):
        return False
    strings = {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return len(strings & _PARTS) >= 4 or len(strings & _LINKS) >= 4


def _literals_holding_the_links() -> set[str]:
    found: set[str] = set()
    for root in ("scripts", "src"):
        for path in sorted((_ROOT / root).rglob("*.py")):
            if "__pycache__" in path.parts or "ur_ros2_description" in path.parts:
                continue
            relative = path.relative_to(_ROOT).as_posix()
            for statement in ast.parse(path.read_text(encoding="utf-8")).body:
                if any(_holds_the_links(node) for node in ast.walk(statement)):
                    found.add(f"{relative} {_where(statement)}")
    return found


class NoCopyGoesUnlistedTests(unittest.TestCase):
    def test_every_literal_holding_the_links_is_a_listed_copy_or_named_as_none(self) -> None:
        """A new table of the six links anywhere in scripts/ or src/ fails here until it is listed."""
        self.assertEqual(sorted(_COPIES | set(_NOT_COPIES)), sorted(_literals_holding_the_links()))

    def test_the_scan_sees_a_new_table(self) -> None:
        """THE CONTROL: the shape the scan looks for, planted in a module it has not seen."""
        planted = ast.parse('TABLE = ("shoulder", "upper_arm", "forearm", "wrist_1")')
        self.assertTrue(any(_holds_the_links(node) for node in ast.walk(planted)))


class TheCommittedDataIsTheDescriptionTests(unittest.TestCase):
    """The data half, which stays after F7 and F9: the bundles and sphere maps were baked in these frames."""

    def test_every_bundle_bakes_each_part_in_its_rows_frame(self) -> None:
        for model in available_arms():
            bundle = _DATA / f"{model}_collision_meshes.npz"
            if model == "ur20":
                self.assertFalse(bundle.exists(), "ur20 has no bundle; if it gains one, this test reads it")
                continue
            with np.load(bundle, allow_pickle=False) as arrays:
                for part, _link, frame in load_arm(model).collision_links:
                    with self.subTest(model=model, part=part):
                        values = arrays[f"{part}__frame"]
                        self.assertEqual(np.dtype(np.int64), values.dtype)
                        self.assertEqual([frame], values.tolist())

    def test_every_sphere_map_fits_each_part_in_its_rows_frame_in_order(self) -> None:
        for model in available_arms():
            path = _MAPS / f"{model}_arm_spheres.yml"
            if model == "ur20":
                self.assertFalse(path.exists(), "ur20 has no sphere map; if it gains one, this test reads it")
                continue
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            rows = [(part, frame) for part, _link, frame in load_arm(model).collision_links]
            with self.subTest(model=model):
                self.assertEqual(rows, [(body, spec["frame"]) for body, spec in data["collision_spheres"].items()])
                self.assertEqual(rows, [(body["body"], body["frame"]) for body in data["_provenance"]["bodies"]])


if __name__ == "__main__":
    unittest.main()
