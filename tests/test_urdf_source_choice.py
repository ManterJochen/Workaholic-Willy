"""Where a descriptor's URDF comes from is a choice the builder makes out loud (B5, S9).

Until now it was not a choice at all: the builder scanned Isaac's motion policy folder, took the first file that
described a body, repaired a stray parenthesis in it, and grafted a tool frame onto it from a sibling when it had
none. That works, and it means a customer box cannot build a descriptor without a simulator installed, and that ur10
was built from an asset whose frames are a different family from the vendor's own (measured: its links sit up to
65 mm from where UR's description puts them).

``_urdf_source.choose`` names the two sources and what each promises.

* ``isaac`` is exactly what the builder did: a file on Isaac's disk, whatever it says.
* ``ur`` renders Universal Robots' own description, pinned to one upstream commit. It has a tool frame, complete
  inertia and an elbow already inside UR's planning limit, so nothing is repaired and nothing is grafted. What it
  cannot supply is REFUSED rather than transcribed from a neighbour: a tool frame written from somewhere else is how
  a hand ends up pointing where it is not.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts" / "curobo"


def _load(name: str):
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(_SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SCRIPTS))
    return module


class TheSourceIsNamedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _load("_urdf_source")
        self.description = _load("_ur_description")

    def test_ur_renders_the_vendors_own_description_and_says_so(self) -> None:
        chosen = self.source.choose("ur5e", source="ur")

        self.assertEqual(chosen.text, self.description.render_urdf("ur5e"))
        self.assertEqual(chosen.provenance["urdf_from"], "ur")
        self.assertEqual(chosen.provenance["description_commit"], self.description._commit())  # noqa: SLF001
        self.assertEqual(len(chosen.provenance["urdf_sha256"]), 64)
        self.assertIsNone(chosen.path, "a rendered description was read from a file")

    def test_ur_needs_no_isaac_at_all(self) -> None:
        """⭐ THE POINT OF THE STEP. A customer box has no simulator, and this path must not look for one."""
        chosen = self.source.choose("ur10", source="ur", isaac_dir=Path("D:/there-is-no-isaac-here"))
        self.assertIn("tool0", chosen.text)
        self.assertEqual(chosen.provenance["urdf_from"], "ur")

    def test_every_model_renders_a_body_the_builder_can_use(self) -> None:
        for model in ("ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e"):
            with self.subTest(model=model):
                chosen = self.source.choose(model, source="ur")
                self.assertIn('<link name="tool0"', chosen.text)
                self.assertIn('<link name="base_link_inertia"', chosen.text)

    def test_a_render_without_a_tool_frame_is_refused_and_never_grafted(self) -> None:
        """⭐ THE CONTROL. Under ``isaac`` a missing tool0 is transcribed from a sibling description, because that
        generation of ur_description has none. Under ``ur`` there is no sibling to ask and no reason to: a rendered
        description that cannot name its own tool frame is broken, and the answer is to say so."""
        original = self.source.render_urdf
        try:
            self.source.render_urdf = lambda model: original(model).replace('<link name="tool0"/>', "")
            with self.assertRaises(SystemExit) as caught:
                self.source.choose("ur5e", source="ur")
        finally:
            self.source.render_urdf = original
        self.assertIn("tool0", str(caught.exception))

    def test_an_unknown_source_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self.source.choose("ur5e", source="whatever-the-shell-said")

    def test_the_builder_defaults_to_the_vendors_description_and_names_every_exception(self) -> None:
        """B5 S10. A default is what a customer box gets, so it is the vendor's own description.

        ur10 is the one exception and it is passed explicitly: its arm sphere map is Isaac's and lives in Isaac's
        link frames, so rendering that arm from UR's description puts the map 75.9 mm off its own upper arm and the
        builder refuses. Measured 2026-09-16. B6 fits those spheres to the committed bundle and the exception goes.
        """
        builder = (_SCRIPTS / "build_ur_config.py").read_text(encoding="utf-8")
        self.assertIn('URDF_FROM = "ur"', builder, "the builder still defaults to a simulator's copy")

        install = (_ROOT / "scripts" / "ext_deps" / "install.ps1").read_text(encoding="utf-8")
        self.assertIn("--urdf-from", install, "the install never states a source, so the default is all there is")
        self.assertIn("ur10", install)

    def test_isaac_reads_the_file_it_names(self) -> None:
        """The other half, unchanged: whatever Isaac's disk holds, with its path recorded."""
        isaac = self.source.isaac_root("ur5e")
        if isaac is None or not isaac.is_dir():
            self.skipTest("no Isaac motion policy configs on this box")
        assets = _ROOT / "ext_deps" / "curobo" / "curobo" / "content" / "assets" / "robot" / "ur_description"
        if not assets.is_dir():
            self.skipTest("no cuRobo asset root on this box, so Isaac's mesh references cannot resolve")
        chosen = self.source.choose("ur5e", source="isaac", isaac_dir=isaac, asset_root=assets)
        self.assertEqual(chosen.provenance["urdf_from"], "isaac")
        self.assertTrue(str(chosen.path).endswith(".urdf"))
        self.assertEqual(chosen.text, Path(chosen.path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
