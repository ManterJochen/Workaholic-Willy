"""Universal Robots' description config is pinned and vendored beside the build scripts, byte for byte as its commit holds it.

The meshes live in cuRobo's asset root and are held by ``scripts/arms/ur_meshes.sha256``. The kinematics, joint
limits, physical and visual parameters, and the xacro macro they feed, are vendored under
``scripts/arms/ur_ros2_description`` and held by ``ur_ros2_description.sha256``. That is what lets the URDF writer
and its forward kinematics proof run off the box, in every interpreter, without Isaac. The folder is named after the
upstream repository on purpose: the mesh destination in the asset root is called ``ur_description``, and two trees of
one name are a ``--dest`` typo away from a green check on the wrong one.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from src.config.arms import available_arms

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_FETCH = _ROOT / "scripts" / "arms" / "fetch.py"
_KINDS = ("default_kinematics", "joint_limits", "physical_parameters", "visual_parameters")


def _fetch():
    spec = importlib.util.spec_from_file_location("_fetch_for_description_tests", _FETCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TheDescriptionIsPinnedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fetch = _fetch()
        cls.manifest = cls.fetch.DESCRIPTION_MANIFEST
        cls.vendored = cls.fetch.DESCRIPTION
        cls.pinned = cls.fetch.read_manifest(cls.manifest)

    def test_the_header_names_the_commit_and_the_licence(self) -> None:
        header = "\n".join(line for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.startswith("#"))
        self.assertIn(self.fetch.COMMIT, header)
        self.assertIn("BSD-3-Clause", header)
        self.assertIn(self.vendored.relative_to(_ROOT).as_posix(), header)

    def test_the_description_and_both_pins_sit_beside_the_script_that_fetches_them(self) -> None:
        """P0f F6 moved them out of ``scripts/curobo/`` (owner, 2026-09-23: flat in ``scripts/arms/``)."""
        for path in (self.vendored, self.manifest, self.fetch.MANIFEST):
            with self.subTest(path=path.name):
                self.assertEqual(_FETCH.parent, path.parent)
                self.assertTrue(path.exists(), path)

    def test_every_model_carries_its_four_files(self) -> None:
        models = sorted({path.split("/")[1] for path in self.pinned if path.startswith("config/")})
        self.assertGreaterEqual(len(models), 14, f"the description pins {models}")
        self.assertLessEqual(set(available_arms()), set(models), "an arm the registry describes is not pinned")
        for model in models:
            for kind in _KINDS:
                with self.subTest(model=model, kind=kind):
                    self.assertIn(f"config/{model}/{kind}.yaml", self.pinned)

    def test_the_macro_the_writer_transcribes_is_pinned(self) -> None:
        for relative in ("urdf/ur_macro.xacro", "urdf/inc/ur_common.xacro", "urdf/ur.urdf.xacro", "LICENSE"):
            with self.subTest(relative):
                self.assertIn(relative, self.pinned)

    def test_no_mesh_is_pinned_here(self) -> None:
        """Meshes are the other manifest's, and five of their sets carry other terms than this file's header states."""
        self.assertEqual([path for path in self.pinned if path.startswith("meshes/")], [])

    def test_the_vendored_bytes_are_the_pinned_bytes(self) -> None:
        self.assertEqual(self.fetch.check(self.vendored, manifest=self.manifest), [])

    def test_a_changed_byte_is_named(self) -> None:
        """The control: a check that cannot see one flipped bit in a kinematics file proves nothing about the others."""
        with tempfile.TemporaryDirectory() as tmp:
            copy = pathlib.Path(tmp) / "description"
            shutil.copytree(self.vendored, copy)
            target = copy / "config" / "ur5e" / "default_kinematics.yaml"
            data = bytearray(target.read_bytes())
            data[len(data) // 2] ^= 0x01
            target.write_bytes(bytes(data))
            problems = self.fetch.check(copy, manifest=self.manifest)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("config/ur5e/default_kinematics.yaml", problems[0])

    def test_git_keeps_the_bytes(self) -> None:
        """This box sets core.autocrlf=true; a checkout without -text would rewrite every YAML's line endings."""
        attributes = (_ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn(f"{self.vendored.relative_to(_ROOT).as_posix()}/** -text", attributes)

    def test_git_reads_the_rule_for_the_files_where_they_sit(self) -> None:
        """The rule as git applies it, not as the file spells it: a rule left at the old folder matches nothing here,
        and a fresh checkout with core.autocrlf=true then writes CRLF into every YAML and fails the pin."""
        if not (_ROOT / ".git").exists():
            # A copy of the files git sees, as the customer-hand trial makes, has no repository: no git applies a rule
            # there, and the bytes were copied as git holds them. The rule as written is test_git_keeps_the_bytes.
            self.skipTest(f"{_ROOT} is not a git checkout, so git applies no attribute rule to it")
        for relative in ("LICENSE", "config/ur5e/default_kinematics.yaml", "urdf/ur_macro.xacro"):
            path = (self.vendored / relative).relative_to(_ROOT).as_posix()
            with self.subTest(path=path):
                answer = subprocess.run(["git", "check-attr", "text", "--", path], cwd=_ROOT, capture_output=True,
                                        text=True, check=True).stdout.strip()
                self.assertEqual(f"{path}: text: unset", answer)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
