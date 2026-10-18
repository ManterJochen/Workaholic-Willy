"""Where a descriptor's URDF comes from is said out loud, and there is one answer (B5 S9, P0f F5).

Until B5 it was not a choice at all: the builder scanned Isaac's motion policy folder, took the first file that
described a body, repaired a stray parenthesis in it, and grafted a tool frame onto it from a sibling when it had
none. That meant a customer box could not build a descriptor without a simulator installed, and that ur10 was built
from an asset whose frames are a different family from the vendor's own (measured: its links sit up to 65 mm from
where UR's description puts them).

``_urdf_source.choose`` renders Universal Robots' own description, pinned to one upstream commit. It has a tool frame,
complete inertia and an elbow already inside UR's planning limit, so nothing is repaired and nothing is grafted. What it
cannot supply is REFUSED rather than transcribed from a neighbour. The ``isaac`` source, a file on a simulator's disk,
was retired in P0f F5, every arm having been built from ``ur`` since B6; it is refused by name.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _ROOT / "scripts" / "arms"


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

    def test_ur_needs_no_isaac_at_all(self) -> None:
        """⭐ THE POINT OF THE STEP. A customer box has no simulator, and nothing here looks for one."""
        chosen = self.source.choose("ur10", source="ur")
        self.assertIn("tool0", chosen.text)
        self.assertEqual(("ur",), self.source.SOURCES)
        self.assertFalse(hasattr(self.source, "isaac_root"), "the chooser still knows where Isaac lives")

    def test_every_model_renders_a_body_the_builder_can_use(self) -> None:
        for model in ("ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e"):
            with self.subTest(model=model):
                chosen = self.source.choose(model, source="ur")
                self.assertIn('<link name="tool0"', chosen.text)
                self.assertIn('<link name="base_link_inertia"', chosen.text)

    def test_a_render_without_a_tool_frame_is_refused_and_never_grafted(self) -> None:
        """⭐ THE CONTROL. The retired isaac path transcribed a missing tool0 from a sibling description. Here there is
        no sibling to ask and no reason to: a rendered description that cannot name its own tool frame is broken, and
        the answer is to say so."""
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

    def test_the_retired_isaac_source_is_refused_by_name(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.source.choose("ur5e", source="isaac")
        self.assertIn("retired in P0f F5", str(caught.exception))

    def test_the_builder_defaults_to_the_vendors_description_and_the_install_asks_for_it(self) -> None:
        """B5 S10. A default is what a customer box gets, so it is the vendor's own description.

        ur10 was the one exception until B6: its arm sphere map was Isaac's and lived in Isaac's link frames, so
        rendering that arm from UR's description put the map 75.9 mm off its own upper arm (measured 2026-09-16).
        B6 fitted every arm's spheres to its committed bundle, and the install passes ``ur`` for all seven since.
        """
        builder = (_SCRIPTS / "descriptor.py").read_text(encoding="utf-8")
        self.assertIn('URDF_FROM = "ur"', builder, "the builder still defaults to a simulator's copy")

        import subprocess

        plan = subprocess.run([sys.executable, str(_ROOT / "scripts" / "ext_deps" / "install.py"), "--plan",
                               "--from-nothing", "--platform", "win-64"], capture_output=True, text=True,
                              check=True).stdout
        self.assertIn("--urdf-from", plan, "the install never states a source, so the default is all there is")
        self.assertIn("--urdf-from ur", plan)
        self.assertNotIn("--urdf-from isaac", plan, "an arm still takes a simulator's description")


if __name__ == "__main__":
    unittest.main()
