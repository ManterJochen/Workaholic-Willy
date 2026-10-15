"""The build scripts' reader answers what the registry answers, bit for bit (P0f F7).

``scripts/arms/_arm_file.py`` reads an arm description by path with PyYAML and no schema, because the cuRobo
environment runs Python 3.10 and cannot import the config layer. Its answers feed files that are hashed into the
evidence, so each one is compared with ``load_arm``'s by ``float.hex`` where it is a float: a reader that returned an
int for a float, or dropped a sign, would move a byte that an ``==`` cannot see.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path

from src.config.arms import available_arms, load_arm

_ROOT = Path(__file__).resolve().parents[1]
_READER = _ROOT / "scripts" / "arms" / "_arm_file.py"


def _reader():
    spec = importlib.util.spec_from_file_location("_arm_file_under_test", _READER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bits(value: object) -> object:
    """A value with every float spelled by ``float.hex`` and every sequence as a tuple, so type and sign count."""
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, (list, tuple)):
        return tuple(_bits(item) for item in value)
    return (type(value).__name__, value)


class TheReaderAnswersAsTheRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reader = _reader()

    def _registry(self, model: str) -> dict[str, object]:
        arm = load_arm(model)
        anchors = [joint.retract_anchor_rad for joint in arm.joints]
        return {
            "joint_names": [joint.name for joint in arm.joints],
            "limits_deg": [tuple(joint.limit_deg) for joint in arm.joints],
            "dh_rows": [(joint.a_m, joint.d_m, joint.alpha_rad) for joint in arm.joints],
            "collision_links": list(arm.collision_links),
            "chain_yaw_deg": arm.base.chain_yaw_deg,
            "flange_link": arm.flange.link,
            "planner_limits_deg": [tuple(joint.planner_limit_deg) for joint in arm.joints],
            "position_limit_clip_rad": arm.planner.position_limit_clip_rad,
            "retract_anchor": None if all(a is None for a in anchors) else tuple(anchors),
            "retract_anchor_source": arm.retract_anchor_source,
        }

    def test_every_answer_is_the_registrys_bit_for_bit(self) -> None:
        self.assertEqual(available_arms(), self.reader.described())
        for model in available_arms():
            for name, said in self._registry(model).items():
                with self.subTest(model=model, answer=name):
                    self.assertEqual(_bits(said), _bits(getattr(self.reader, name)(model)))

    def test_an_arm_nobody_describes_is_refused_naming_the_ones_there_are(self) -> None:
        with self.assertRaises(self.reader.ArmFileError) as caught:
            self.reader.read("ur99")
        self.assertIn("ur5e", str(caught.exception))

    def test_a_sign_or_a_type_is_seen(self) -> None:
        """THE CONTROL: the comparison above tells -0.0 from 0.0 and 180 from 180.0, which ``==`` does not."""
        self.assertNotEqual(_bits(-0.0), _bits(0.0))
        self.assertNotEqual(_bits(180), _bits(180.0))


class TheReaderRunsWhereTheBuildRunsTests(unittest.TestCase):
    def test_it_parses_as_python_3_10(self) -> None:
        """The cuRobo environment is Python 3.10: syntax a later Python added would stop the descriptor build."""
        ast.parse(_READER.read_text(encoding="utf-8"), feature_version=(3, 10))

    def test_it_imports_the_standard_library_and_pyyaml_only(self) -> None:
        tree = ast.parse(_READER.read_text(encoding="utf-8"))
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {str(node.module).split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertLessEqual(imported, {"__future__", "functools", "pathlib", "typing", "yaml"})


if __name__ == "__main__":
    unittest.main()
