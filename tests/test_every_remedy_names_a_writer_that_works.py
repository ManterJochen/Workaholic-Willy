"""Every refusal and remedy for a hand names a writer that works, and the planner reads only a cover fit of that hand.

Customer chain lane C2g, C4h and C2h (findings 16, 17, 18 and 22 of the audit of 2026-09-17). A missing map sent the
operator to the grid fit B6 retired for its holes, written straight to the committed name; a missing hand bundle sent
them to a baker that exits 2 for any hand outside its presets; the desk told them to bake the ARM when only the hand
was missing; and planner_hand admitted any map whose origin it could read.
"""

from __future__ import annotations

import ast
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

_REPO = Path(__file__).resolve().parents[1]
_MAPS = _REPO / "src" / "robot" / "safety" / "planning" / "robot"
_DATA = _REPO / "src" / "robot" / "safety" / "data"


def _config(hand: str = "robotiq_hande", planner: str = "curobo", **gripper: Any) -> Any:
    from src.config.schema.robot import RobotConfig

    block: dict[str, Any] = {"model": hand, **gripper}
    if hand == "robotiq_hande" and "coupling_plates" not in gripper:
        block["coupling_plates"] = [{"name": "plate", "thickness_mm": 20.0}]
    return RobotConfig.model_validate({"vendor": "ur", "ur": {"motion_planner": planner}, "gripper": block})


def _flags(script: Path) -> set[str]:
    flags: set[str] = set()
    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            flags.update(arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str))
            for arg in node.args:
                if isinstance(arg, ast.JoinedStr):
                    flags.update(f"--{part}-mesh" for part in ("gripper", "lfinger", "rfinger"))
                    flags.update(f"--{word}" for word in ("closing", "approach", "binormal"))
    return flags


def _named_commands_are_real(sentence: str) -> list[str]:
    """Every scripts/...py the sentence names exists, and every --flag after it is one that script declares."""
    wrong: list[str] = []
    for match in re.finditer(r"(scripts/[\w/]+\.py)([^.;]*)", sentence):
        script = _REPO / match.group(1)
        if not script.is_file():
            wrong.append(f"{match.group(1)} does not exist")
            continue
        declared = _flags(script)
        wrong.extend(f"{match.group(1)} has no {flag}" for flag in re.findall(r"--[a-z][a-z-]*", match.group(2))
                     if flag not in declared)
    return wrong


class TheSentencesNameRealToolsTests(unittest.TestCase):
    def test_every_named_script_and_flag_exists(self) -> None:
        from src.robot.safety.planning.environment import hand_writers_sentence, sphere_map_writer_sentence

        for sentence in (hand_writers_sentence("acme_2f"), sphere_map_writer_sentence("acme_2f")):
            with self.subTest(sentence=sentence[:60]):
                self.assertEqual(_named_commands_are_real(sentence), [])
        self.assertIn("write_hand_from_mesh.py", hand_writers_sentence("acme_2f"))

    def test_the_scan_sees_a_flag_a_script_lacks(self) -> None:
        """⭐ THE CONTROL."""
        self.assertEqual(_named_commands_are_real("run scripts/curobo/fit_cover_spheres.py --no-such-flag"),
                         ["scripts/curobo/fit_cover_spheres.py has no --no-such-flag"])

    def test_the_guard_hint_names_every_hand_writer(self) -> None:
        from src.robot.safety._fcl_self_collision import _STATUS_HINTS

        for writer in ("write_hand_from_dimensions.py", "write_hand_from_mesh.py", "bake_gripper_variant.py --usd"):
            self.assertIn(writer, _STATUS_HINTS["no_hand_bundle"])

    def test_build_gripper_spheres_writes_no_map(self) -> None:
        import contextlib
        import io

        from src.robot.safety.planning.robot import build_gripper_spheres

        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for argv in (["--variant", "schunk_egu50", "--out", str(folder / "x.yml")],
                     ["--mesh", str(folder / "any.stl"), "--gripper", "g", "--cell-mm", "34", "--rmax-mm", "17",
                      "--origin", "flange", "--out", str(folder / "y.yml")]):
            with self.subTest(argv=argv[0]), contextlib.redirect_stderr(io.StringIO()) as err, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(build_gripper_spheres.main(argv), 2)
            self.assertIn("fit_cover_spheres.py", err.getvalue())
        self.assertEqual(list(folder.iterdir()), [])


class TheMapRefusalSaysWhyAndWhatTests(unittest.TestCase):
    def test_planner_hand_names_the_cover_fit_for_a_missing_map(self) -> None:
        from src.config.loader import ConfigError
        from src.robot.safety.planning import hand as hand_module

        empty = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with mock.patch.object(hand_module, "_MAPS", empty), self.assertRaises(ConfigError) as caught:
            hand_module.planner_hand(_config())
        self.assertIn("scripts/curobo/fit_cover_spheres.py --hand robotiq_hande --write", str(caught.exception))
        self.assertNotIn("build_gripper_spheres", str(caught.exception))

    def test_the_missing_map_refusal_on_an_ik_cell_names_what_reads_it(self) -> None:
        from src.config.loader import ConfigError
        from src.robot.safety.planning import hand as hand_module

        empty = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with mock.patch.object(hand_module, "_MAPS", empty), self.assertRaises(ConfigError) as caught:
            hand_module.planner_hand(_config("schunk_egu50", planner="ik"))
        said = str(caught.exception)
        self.assertIn("no planner starts on this cell", said)
        self.assertIn("coupling_plates", said)
        self.assertIn("fit_cover_spheres.py --hand schunk_egu50 --write", said)

    def test_a_cuRobo_cell_is_told_its_planner_reads_the_map(self) -> None:
        """⭐ THE CONTROL: the reason follows the planner, it is not one sentence for every cell."""
        from src.config.loader import ConfigError
        from src.robot.safety.planning import hand as hand_module

        empty = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with mock.patch.object(hand_module, "_MAPS", empty), self.assertRaises(ConfigError) as caught:
            hand_module.planner_hand(_config("schunk_egu50", planner="curobo"))
        self.assertIn("planner adds the hand", str(caught.exception))


class OnlyACoverFitOfThisHandIsReadTests(unittest.TestCase):
    def _refused_with(self, edit: Any) -> str:
        from src.config.loader import ConfigError
        from src.robot.safety.planning import hand as hand_module

        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for path in _MAPS.glob("*_gripper_spheres.yml"):
            shutil.copy(path, folder / path.name)
        target = folder / "robotiq_hande_gripper_spheres.yml"
        document = yaml.safe_load(target.read_text(encoding="utf-8"))
        edit(document["_provenance"])
        target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        with mock.patch.object(hand_module, "_MAPS", folder), self.assertRaises(ConfigError) as caught:
            hand_module.planner_hand(_config())
        return str(caught.exception)

    def test_a_grid_fit_map_is_refused(self) -> None:
        def grid(provenance: dict) -> None:
            provenance["generated_by"] = "src/robot/safety/planning/robot/build_gripper_spheres.py"
            provenance.pop("bodies", None)

        said = self._refused_with(grid)
        self.assertIn("fit_cover_spheres.py", said)
        self.assertIn("hole", said)

    def test_a_map_fitted_from_another_hand_is_refused(self) -> None:
        said = self._refused_with(lambda provenance: provenance.update(source="schunk_egu50_hand_meshes.npz"))
        self.assertIn("schunk_egu50_hand_meshes.npz", said)
        self.assertIn("robotiq_hande_hand_meshes.npz", said)

    def test_a_map_with_an_uncovered_body_is_refused(self) -> None:
        def uncovered(provenance: dict) -> None:
            for row in provenance["bodies"]:
                if row["body"] == "lfinger":
                    row["fresh_uncovered_max_mm"] = 0.4

        said = self._refused_with(uncovered)
        self.assertIn("lfinger", said)
        self.assertIn("0.4", said)
        said = self._refused_with(lambda provenance: provenance.update(bodies=[r for r in provenance["bodies"]
                                                                               if r["body"] != "rfinger"]))
        self.assertIn("rfinger", said)

    def test_every_committed_map_is_admitted(self) -> None:
        """⭐ THE CONTROL: the check refuses what is wrong, and every shipped hand resolves."""
        from src.robot.safety.planning.hand import planner_hand

        for hand in ("robotiq_2f85", "robotiq_hande", "schunk_egu50"):
            with self.subTest(hand=hand):
                planner_hand(_config(hand))


class TheDoctorAndTheDeskFollowTheTokenTests(unittest.TestCase):
    def test_the_doctor_names_every_writer_for_a_missing_bundle_and_the_cover_fit_for_a_missing_map(self) -> None:
        from src.robot.safety.planning import doctor, environment
        from src.robot.safety.planning import hand as hand_module

        empty = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with mock.patch.object(environment, "COLLISION_MESH_DIR", empty):
            (probe,) = doctor._probe_gripper("ur5e", "robotiq_hande")
        for writer in ("write_hand_from_dimensions.py", "write_hand_from_mesh.py", "bake_gripper_variant.py --usd"):
            self.assertIn(writer, probe.remedy)
        with mock.patch.object(hand_module, "_MAPS", empty):
            probes = doctor._probe_gripper("ur5e", "robotiq_hande")
        (missing,) = [probe for probe in probes if probe.name.startswith("gripper sphere map")]
        self.assertIn("fit_cover_spheres.py --hand robotiq_hande --write", missing.remedy)
        self.assertNotIn("build_gripper_spheres", missing.remedy)

    def test_the_desk_fix_follows_the_token(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.execution.real_cell.preflight import _exact_mesh_engine_row

        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        shutil.copy(_DATA / "ur5e_collision_meshes.npz", folder / "ur5e_collision_meshes.npz")
        config = RobotConfig.model_validate({
            "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": "curobo"},
            "gripper": {"model": "robotiq_hande", "coupling_plates": [{"name": "plate", "thickness_mm": 20.0}]},
            "safety": {"self_collision": {"backend": "fcl", "mesh_dir": str(folder)}},
        })
        row = _exact_mesh_engine_row(config, "coal")
        self.assertIn("no_hand_bundle", row.fix)
        self.assertIn("write_hand_from_dimensions.py", row.fix)
        self.assertNotIn("bake_ur_collision_meshes.py", row.fix)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
