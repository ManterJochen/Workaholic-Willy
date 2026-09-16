"""Every committed arm bundle rebuilds, array for array, from pinned files alone (B5, S6 to S8).

A collision bundle is the exact mesh guard's whole authority, and until now it was a binary somebody produced once on
a box with a simulator installed. Nobody could rebuild it, so nobody could check it: a wrong frame in a bundle does
not fail loudly, it guards empty space and passes every pose.

Now the inputs are all pinned. Universal Robots' collision STL files are held by ``ur_meshes.sha256`` at one upstream
commit, their config is vendored under ``scripts/curobo/ur_description``, and the URDF that places them is rendered
from that config rather than shipped. So the bundle is a FUNCTION of pinned bytes, and this test evaluates it.

``np.array_equal``, not a tolerance: the same reader over the same bytes with the same arithmetic produces the same
numbers, and anything looser would hide exactly the drift this is for. It skips, rather than passes, where the pinned
meshes are not on the box: a measurement nobody could take is not a pass.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
_DATA = _ROOT / "src" / "robot" / "safety" / "data"

#: The source string that says a bundle came from the pinned STLs, and so can be rebuilt here.
FROM_PINNED_STL = "ur_description_stl"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(_SCRIPTS / "curobo"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SCRIPTS / "curobo"))
        sys.path.remove(str(path.parent))
    return module


def _why_not() -> "str | None":
    try:
        import trimesh  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"trimesh is not importable: {exc}"
    fetch = _load("fetch_for_reproduce", _SCRIPTS / "curobo" / "fetch_ur_meshes.py")
    root = fetch._default_dest()  # noqa: SLF001
    if not root.is_dir():
        return f"the pinned mesh tree is not on this box ({root})"
    missing = fetch.check(root)
    if missing:
        return f"{len(missing)} pinned mesh file(s) are missing or changed under {root}"
    return None


class TheBundlesAreAFunctionOfPinnedBytesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reason = _why_not()
        if reason is not None:
            raise unittest.SkipTest(reason)
        cls.bake = _load("bake_for_reproduce", _SCRIPTS / "isaac" / "bake_ur_meshes_from_urdf.py")
        cls.index = json.loads((_DATA / "bundles.json").read_text(encoding="utf-8"))["bundles"]

    def _from_pinned(self) -> list[str]:
        return sorted(
            entry["model"] for entry in self.index.values()
            if entry.get("source") == FROM_PINNED_STL and "model" in entry
        )

    def _committed(self, model: str) -> dict:
        with np.load(_DATA / f"{model}_collision_meshes.npz", allow_pickle=True) as bundle:
            return {name: np.array(bundle[name]) for name in bundle.files}

    def test_every_arm_bundle_rebuilds_to_the_same_arrays(self) -> None:
        models = self._from_pinned()
        self.assertGreaterEqual(len(models), 5, "too few bundles claim the pinned source for this to prove anything")
        for model in models:
            rebaked = self.bake.bake(model)
            committed = self._committed(model)
            for key in sorted(rebaked):
                with self.subTest(model=model, array=key):
                    self.assertIn(key, committed)
                    self.assertTrue(
                        np.array_equal(committed[key], rebaked[key]),
                        f"{model} {key} does not rebuild from the pinned files",
                    )

    def _rebake_with(self, model: str, urdf: str) -> dict:
        original = self.bake.sources
        try:
            self.bake.sources = lambda name: dict(original(name), urdf_text=urdf)
            return self.bake.bake(model)
        finally:
            self.bake.sources = original

    def test_a_moved_collision_origin_does_not_rebuild_and_a_moved_visual_one_changes_nothing(self) -> None:
        """⭐ THE CONTROL PAIR. Equality that survives a moved link is not equality.

        And the second half says WHICH geometry the bundle is made of. A UR link declares two mesh origins, one for
        what you see and one for what collides, and they are not always the same number. Moving the visual one must
        change nothing: the guard is not allowed to be watching the pretty mesh.
        """
        model = "ur5e"
        description = _load("_ur_description_for_reproduce", _SCRIPTS / "curobo" / "_ur_description.py")
        urdf = description.render_urdf(model)
        committed = self._committed(model)

        visual_only = urdf.replace('xyz="0.0 0.0 -0.0997"', 'xyz="0.0 0.0 -0.0987"', 1)
        self.assertNotEqual(visual_only, urdf, "the visual origin this control moves is not in the rendered text")
        self.assertTrue(np.array_equal(committed["wrist_2__v"], self._rebake_with(model, visual_only)["wrist_2__v"]),
                        "moving the VISUAL origin changed the bundle, so the guard is reading the wrong geometry")

        both = urdf.replace('xyz="0.0 0.0 -0.0997"', 'xyz="0.0 0.0 -0.0987"')
        self.assertNotEqual(both, visual_only, "wrist_2 has no separate collision origin in the rendered text")
        self.assertFalse(np.array_equal(committed["wrist_2__v"], self._rebake_with(model, both)["wrist_2__v"]))

    def test_the_hulled_and_the_simulator_sourced_bundles_are_named_rather_than_silently_skipped(self) -> None:
        """Whatever cannot be rebuilt here says so in its own entry, so the gap is visible instead of implied."""
        for name, entry in sorted(self.index.items()):
            if entry.get("source") == FROM_PINNED_STL:
                continue
            with self.subTest(bundle=name):
                self.assertTrue(entry["note"].strip())
                self.assertIn(entry["source"], {"isaac_usd", "isaac_importer_visual_hull"})


if __name__ == "__main__":
    unittest.main()
