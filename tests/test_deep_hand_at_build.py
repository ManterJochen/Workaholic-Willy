"""The deep generator is built for the hand the cell names, and a hand its artifact never saw refuses at build.

Lane (i) D2. `robot.gripper.model` is the one name for a hand (Step 4 Q5). The learned generator conditions on a hand,
and before D2 that hand came from `robot.grasping.deep_generator.gripper`, a second name nothing tied to the first,
checked only inside the calculator's lazy loader, where a cell's compute path swallows the refusal. Now the factory
compares the cell's hand with the artifact's trained hands at build, after the artifact checks, by canonical name on
both sides (the `2f85` stamp is `robotiq_2f85`), and raises `ValueError`, the type its callers already catch. The
registry is read from the cell's own tree, so every build site hands its tree down.
"""

from __future__ import annotations

import ast
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import torch

from src.config import ConfigError, reload_config
from src.config.schema.robot import RobotConfig
from src.robot.grasping.deep.set_artifact import SET_ARTIFACT_KIND, SET_ARTIFACT_VERSION

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "config"
_DEEP = "src.robot.grasping.deep.calculator.DeepGraspCalculator"


def _artifact(directory: Path, *, stamp: str = "2f85", trained: tuple[str, ...] = ()) -> str:
    """The smallest file the factory's artifact checks accept, carrying the hands it was trained across."""
    path = directory / "generator.pt"
    torch.save({"kind": SET_ARTIFACT_KIND, "artifact_version": SET_ARTIFACT_VERSION, "gripper": stamp,
                "trained_grippers": list(trained or (stamp,))}, path)
    return str(path)


def _deep_cell(artifact: str, hand: str | None = "robotiq_2f85") -> RobotConfig:
    tree: dict[str, Any] = {
        "vendor": "dummy",
        "grasping": {"calculator": "deep", "deep_generator": {"artifact_path": artifact}},
    }
    if hand is not None:
        tree["gripper"] = {"model": hand}
    return RobotConfig.model_validate(tree)


class _WithAnArtifact(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="willy_d2_")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _built_config(self, cell: RobotConfig, **kwargs: Any) -> Any:
        """The `DeepCalculatorConfig` the factory builds for ``cell``. The net itself is not built."""
        from src.robot.grasping.calculator_factory import build_calculator

        with mock.patch(_DEEP) as deep:
            build_calculator(cell, camera_matrix=None, **kwargs)
        return deep.call_args.args[0]


class TheFactoryBuildsForTheCellsHandTests(_WithAnArtifact):
    def test_a_deep_cell_is_built_for_the_hand_it_names(self) -> None:
        artifact = _artifact(self.dir, stamp="slim_pad", trained=("slim_pad", "2f85"))
        self.assertEqual(self._built_config(_deep_cell(artifact)).gripper, "robotiq_2f85")

    def test_the_2f85_stamp_is_the_robotiq_2f85(self) -> None:
        """Every artifact written before the registry is stamped `2f85`, the alias of the cell's model name."""
        self.assertEqual(self._built_config(_deep_cell(_artifact(self.dir))).gripper, "robotiq_2f85")

    def test_the_cells_tree_reaches_the_calculator(self) -> None:
        """The conditioning vector is resolved from the registry the build was checked against."""
        config = self._built_config(_deep_cell(_artifact(self.dir)), data_dir=_DATA)
        self.assertEqual(config.data_dir, str(_DATA))


class AHandTheArtifactNeverSawRefusesAtBuildTests(_WithAnArtifact):
    def test_the_build_refuses_naming_the_hand_key(self) -> None:
        from src.robot.grasping.calculator_factory import build_calculator

        cell = _deep_cell(_artifact(self.dir), hand="robotiq_hande")
        with mock.patch(_DEEP) as deep, self.assertRaises(ValueError) as caught:
            build_calculator(cell, camera_matrix=None)
        deep.assert_not_called()
        message = str(caught.exception)
        self.assertIn("robot.gripper.model", message)
        self.assertIn("robotiq_hande", message)
        self.assertIn("robotiq_2f85", message)

    def test_the_preflight_refuses_the_same(self) -> None:
        from src.robot.grasping.calculator_factory import preflight_calculator

        with self.assertRaises(ValueError) as caught:
            preflight_calculator(_deep_cell(_artifact(self.dir), hand="robotiq_hande"))
        self.assertIn("robot.gripper.model", str(caught.exception))

    def test_the_rehearsal_cell_refuses_the_same(self) -> None:
        """Before D2 it built, and the refusal waited inside the calculator for the first compute."""
        from src.robot.execution.autonomous_grasp.cells import build_rehearsal_components

        with self.assertRaises(ValueError) as caught:
            build_rehearsal_components(_deep_cell(_artifact(self.dir), hand="robotiq_hande"))
        self.assertIn("robot.gripper.model", str(caught.exception))


class AnUnsetHandRefusesADeepCellTests(_WithAnArtifact):
    def test_the_build_refuses_an_unset_hand_naming_the_key(self) -> None:
        from src.robot.grasping.calculator_factory import build_calculator

        with mock.patch(_DEEP), self.assertRaises(ValueError) as caught:
            build_calculator(_deep_cell(_artifact(self.dir), hand=None), camera_matrix=None)
        self.assertIn("robot.gripper.model", str(caught.exception))

    def test_the_preflight_refuses_an_unset_hand(self) -> None:
        from src.robot.grasping.calculator_factory import preflight_calculator

        with self.assertRaises(ValueError) as caught:
            preflight_calculator(_deep_cell(_artifact(self.dir), hand=None))
        self.assertIn("robot.gripper.model", str(caught.exception))

    def test_a_geometric_cell_without_a_hand_still_builds(self) -> None:
        """The control. The analytic generator reads no hand geometry from the registry."""
        from src.robot.grasping.calculator_factory import build_calculator, preflight_calculator

        cell = RobotConfig.model_validate({"grasping": {"calculator": "geometric"}})
        self.assertEqual(type(build_calculator(cell, camera_matrix=None)).__name__, "GraspCalculator")
        self.assertEqual(preflight_calculator(cell), "geometric")

    def test_a_missing_artifact_still_refuses_first_as_a_missing_file(self) -> None:
        """The order: two examples and the ladder catch `FileNotFoundError` for a tree with no weights."""
        from src.robot.grasping.calculator_factory import build_calculator, preflight_calculator

        cell = _deep_cell(str(self.dir / "absent.pt"), hand=None)
        with self.assertRaises(FileNotFoundError):
            build_calculator(cell, camera_matrix=None)
        with self.assertRaises(FileNotFoundError):
            preflight_calculator(cell)


class TheRegistryAnswersFromTheCellsTreeTests(_WithAnArtifact):
    def test_the_short_name_refuses_as_a_value_error_naming_the_model(self) -> None:
        from src.robot.grasping.calculator_factory import build_calculator

        with mock.patch(_DEEP), self.assertRaises(ValueError) as caught:
            build_calculator(_deep_cell(_artifact(self.dir), hand="2f85"), camera_matrix=None)
        self.assertNotIsInstance(caught.exception, ConfigError)
        self.assertIn("robotiq_2f85", str(caught.exception))

    def test_a_tree_without_the_hand_refuses_naming_that_directory(self) -> None:
        from src.robot.grasping.calculator_factory import build_calculator

        root = self.dir / "data"
        shutil.copytree(_DATA / "grippers", root / "grippers")
        (root / "grippers" / "robotiq_2f85.yaml").unlink()
        with mock.patch(_DEEP), self.assertRaises(ValueError) as caught:
            build_calculator(_deep_cell(_artifact(self.dir)), camera_matrix=None, data_dir=root)
        self.assertIn("robotiq_2f85", str(caught.exception))
        self.assertIn("willy_d2_", str(caught.exception))


class TheCalculatorComparesOneNameTests(unittest.TestCase):
    """The calculator's own check, behind the factory's, compares canonical names too."""

    def test_the_cells_model_name_loads_an_artifact_stamped_with_its_alias(self) -> None:
        from tests.test_deep_set_calculator import _artifact as _real_artifact
        from tests.test_deep_set_calculator import _calculator

        with tempfile.TemporaryDirectory() as name:
            calculator = _calculator(_real_artifact(Path(name)), gripper="robotiq_2f85")
            calculator.preload()
            self.assertEqual(calculator._set_gripper, "robotiq_2f85")  # noqa: SLF001

    def test_a_hand_the_artifact_never_saw_refuses_naming_the_hand_key(self) -> None:
        from tests.test_deep_set_calculator import _artifact as _real_artifact
        from tests.test_deep_set_calculator import _calculator

        with tempfile.TemporaryDirectory() as name:
            calculator = _calculator(_real_artifact(Path(name)), gripper="robotiq_hande")
            with self.assertRaises(ValueError) as caught:
                calculator.preload()
        self.assertIn("robot.gripper.model", str(caught.exception))


class TheOldKeyIsRefusedTests(unittest.TestCase):
    def test_the_schema_refuses_the_deep_generators_own_hand(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            RobotConfig.model_validate({"grasping": {"deep_generator": {"gripper": "wide_140"}}})

    def test_a_tree_that_still_writes_it_is_refused_naming_the_hand_key(self) -> None:
        from src.config import load_config
        from src.config.loader import set_active_profile

        with tempfile.TemporaryDirectory(prefix="willy_d2_tree_") as tmp:
            root = Path(tmp) / "data"
            shutil.copytree(_DATA, root)
            (root / "robot" / "robot.deepcheck.yaml").write_text(
                "robot:\n  grasping:\n    deep_generator:\n      gripper: wide_140\n", encoding="utf-8",
            )
            try:
                set_active_profile("deepcheck")
                reload_config()
                with self.assertRaises(ConfigError) as caught:
                    load_config(root)
            finally:
                set_active_profile(None)
                reload_config()
        message = str(caught.exception)
        self.assertIn("robot.grasping.deep_generator.gripper", message)
        self.assertIn("robot.gripper.model", message)


def _calls_without_a_tree(path: Path, names: set[str]) -> list[str]:
    """Every call in ``path`` to one of ``names`` that passes no ``data_dir``, as ``name:line``."""
    missing = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name in names and not any(keyword.arg == "data_dir" for keyword in node.keywords):
            missing.append(f"{name}:{node.lineno}")
    return missing


class EveryBuildSiteHandsDownItsTreeTests(unittest.TestCase):
    """A registry read from the repository's tree under a cell run from its own is the wrong hand, quietly."""

    _SITES: dict[str, set[str]] = {
        "src/robot/execution/autonomous_grasp/cells.py": {
            "build_calculator", "build_real_components", "build_rehearsal_components",
            "_build_on_open_cameras", "_build_camera_calculators",
        },
        "src/robot/execution/cell.py": {"build_real_cell", "build_rehearsal_cell"},
        "src/robot/execution/real_cell/__main__.py": {"rehearsal", "from_robot_config"},
        "api/cell.py": {"build_real_cell", "build_rehearsal_cell"},
        "src/willy_sim/run_m1_pick.py": {"build_calculator"},
        "src/willy_sim/run_m2_pick.py": {"build_calculator"},
        "src/willy_sim/run_eih_pick.py": {"build_calculator"},
        "src/willy_sim/run_dense_pick.py": {"build_calculator"},
        "src/willy_sim/run_attribute_pick.py": {"build_calculator", "_service_for"},
        "src/willy_sim/run_multiview_pick.py": {"build_calculator", "compare_grasp_views"},
    }

    def test_every_build_site_passes_its_tree(self) -> None:
        for relative, names in self._SITES.items():
            with self.subTest(file=relative):
                self.assertEqual(_calls_without_a_tree(_ROOT / relative, names), [])

    def test_the_scan_sees_a_call_without_a_tree(self) -> None:
        """The control: a scan that found nothing on any text would pass the test above."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "site.py"
            path.write_text("build_calculator(cfg, camera_matrix=None)\nbuild_calculator(cfg, data_dir=d)\n",
                            encoding="utf-8")
            self.assertEqual(_calls_without_a_tree(path, {"build_calculator"}), ["build_calculator:1"])

    def test_a_cell_hands_its_tree_to_the_builder(self) -> None:
        from src.robot.execution.cell import Cell

        cell_cfg = RobotConfig.model_validate({"grasping": {}})
        with mock.patch("src.robot.execution.autonomous_grasp.build_rehearsal_cell") as rehearsal, \
             mock.patch("src.robot.execution.autonomous_grasp.build_real_cell") as real:
            Cell.rehearsal(cell_cfg, data_dir="/a/tree").build()
            Cell.from_robot_config(cell_cfg, data_dir="/a/tree").build()
        self.assertEqual(rehearsal.call_args.kwargs["data_dir"], "/a/tree")
        self.assertEqual(real.call_args.kwargs["data_dir"], "/a/tree")


if __name__ == "__main__":
    unittest.main()
