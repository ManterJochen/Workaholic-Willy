"""The decomposition, its cache, and the refusal it replaced.

⛔ WHAT THIS EXISTS TO PREVENT. `MujocoRenderer` used to refuse scanned and composite assets by name,
because MuJoCo resolves mesh contact against the CONVEX HULL and a mug's hull is a solid cylinder. The
refusal was correct and it made the engine useless for the corpus we actually want, so it was replaced
by a decomposition. The failure mode of the replacement is the opposite of the old one and much
quieter: instead of refusing, it settles something plausible against the wrong shape, and nothing
downstream can tell.

So the tests below check the three things that would make it quiet: that the cache returns what was
computed rather than something merely similar, that a changed SETTING is a changed key, and that a
machine without the library still refuses instead of falling back to the hull.
"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from unittest import mock

import numpy as np

from datagen.render.convex_decomposition import (
    DEFAULT_DECOMPOSITION,
    DecompositionSettings,
    DecompositionUnavailable,
    _cache_key,
    convex_parts,
    decomposition_available,
)


def _l_shape() -> tuple[np.ndarray, np.ndarray]:
    """A genuinely concave solid: two boxes meeting at a corner, whose hull fills the notch."""
    import trimesh

    first = trimesh.creation.box(extents=(100.0, 30.0, 30.0))
    second = trimesh.creation.box(extents=(30.0, 100.0, 30.0))
    second.apply_translation((35.0, 35.0, 0.0))
    merged = trimesh.util.concatenate([first, second])
    return np.asarray(merged.vertices, dtype=np.float64), np.asarray(merged.faces)


class ConvexDecompositionCacheTest(unittest.TestCase):
    """The cache is an equality claim, so it is tested as one."""

    @unittest.skipUnless(decomposition_available(), "coacd is not installed on this machine")
    def test_cache_returns_the_same_parts_it_computed(self) -> None:
        vertices, faces = _l_shape()
        with tempfile.TemporaryDirectory() as cache:
            cold = convex_parts(vertices, faces, cache_dir=cache)
            warm = convex_parts(vertices, faces, cache_dir=cache)
        self.assertEqual(len(cold), len(warm))
        for (cold_v, cold_f), (warm_v, warm_f) in zip(cold, warm):
            np.testing.assert_array_equal(cold_v, warm_v)
            np.testing.assert_array_equal(cold_f, warm_f)

    @unittest.skipUnless(decomposition_available(), "coacd is not installed on this machine")
    def test_two_cold_runs_agree(self) -> None:
        """⭑ THE SEED IS LOAD-BEARING. CoACD's search is stochastic; unseeded, two renders of one
        scene would settle differently and the corpus would not be reproducible."""
        vertices, faces = _l_shape()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            left = convex_parts(vertices, faces, cache_dir=first)
            right = convex_parts(vertices, faces, cache_dir=second)
        self.assertEqual(len(left), len(right))
        for (left_v, _), (right_v, _) in zip(left, right):
            np.testing.assert_array_equal(left_v, right_v)

    def test_every_setting_is_in_the_cache_key(self) -> None:
        """A decomposition computed under other settings is a different decomposition.

        Reusing one under the same key is how a cache turns into a lie, and the way that happens is a
        field added to the dataclass and forgotten in the hash.
        """
        vertices, faces = _l_shape()
        baseline = _cache_key(vertices, faces, DEFAULT_DECOMPOSITION)
        for field in dataclasses.fields(DecompositionSettings):
            current = getattr(DEFAULT_DECOMPOSITION, field.name)
            altered = (not current) if isinstance(current, bool) else type(current)(current) + 1
            changed = dataclasses.replace(DEFAULT_DECOMPOSITION, **{field.name: altered})
            self.assertNotEqual(
                baseline, _cache_key(vertices, faces, changed),
                f"{field.name} does not reach the cache key, so two different decompositions would "
                f"share it")

    def test_changed_geometry_is_a_changed_key(self) -> None:
        vertices, faces = _l_shape()
        moved = vertices.copy()
        moved[0, 0] += 0.5
        self.assertNotEqual(_cache_key(vertices, faces, DEFAULT_DECOMPOSITION),
                            _cache_key(moved, faces, DEFAULT_DECOMPOSITION))

    def test_without_the_library_it_refuses_rather_than_returning_the_hull(self) -> None:
        """⛔ THE ONE BEHAVIOUR THAT MUST NEVER BECOME A FALLBACK.

        Handing back the convex hull would answer a question nobody asked, for the life of the corpus,
        and it would look exactly like success.
        """
        vertices, faces = _l_shape()
        real_import = __import__

        def refuse_coacd(name, *args, **kwargs):
            if name == "coacd":
                raise ImportError("no coacd here")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as cache, \
                mock.patch("builtins.__import__", side_effect=refuse_coacd):
            with self.assertRaises(DecompositionUnavailable) as caught:
                convex_parts(vertices, faces, cache_dir=cache)
        self.assertIn("coacd", str(caught.exception))

    @unittest.skipUnless(decomposition_available(), "coacd is not installed on this machine")
    def test_the_parts_cover_the_shape(self) -> None:
        """Each part is convex and their union is roughly the input, checked on bounds and volume.

        Not an exact claim: a decomposition approximates. But a decomposition that lost a limb, or that
        returned the hull, fails both of these by a wide margin.
        """
        import trimesh

        vertices, faces = _l_shape()
        with tempfile.TemporaryDirectory() as cache:
            parts = convex_parts(vertices, faces, cache_dir=cache)
        self.assertGreater(len(parts), 1, "an L-shape that decomposes to one part IS the hull")

        source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        union_low = np.min([part_v.min(axis=0) for part_v, _ in parts], axis=0)
        union_high = np.max([part_v.max(axis=0) for part_v, _ in parts], axis=0)
        np.testing.assert_allclose(union_low, source.vertices.min(axis=0), atol=3.0)
        np.testing.assert_allclose(union_high, source.vertices.max(axis=0), atol=3.0)

        total = sum(abs(float(trimesh.Trimesh(vertices=v, faces=f, process=False).volume))
                    for v, f in parts)
        hull_volume = abs(float(source.convex_hull.volume))
        self.assertLess(total, hull_volume,
                        "the parts fill more than the hull, so the notch was not preserved")
        self.assertGreater(total, 0.5 * abs(float(source.volume)))


class MujocoConcaveAssetsTest(unittest.TestCase):
    """The engine's side of it: what used to be refused by name is now built from parts."""

    def test_scanned_sources_still_need_decomposing(self) -> None:
        from datagen.render.mujoco_engine import _needs_decomposition

        for source in ("gso", "ycb", "objaverse", "custom"):
            self.assertTrue(_needs_decomposition(
                type("R", (), {"source": source, "kind": "mesh"})()))
        self.assertFalse(_needs_decomposition(
            type("R", (), {"source": "procedural", "kind": "box"})()))

    def test_a_composite_uses_its_authored_parts_and_never_calls_coacd(self) -> None:
        """⭑ THEY ARE ALREADY CONVEX. A `CompositePart` is a box, a cylinder or a sphere by
        construction, so decomposing their union would spend seconds to APPROXIMATE shapes we authored
        precisely -- and would lose the pocket a handle arch encloses."""
        from datagen.render.mujoco_engine import _collision_parts

        part = type("P", (), {"primitive": "box", "extent_mm": (40.0, 20.0, 20.0),
                              "offset_mm": (0.0, 0.0, 0.0), "yaw_deg": 0.0})()
        other = type("P", (), {"primitive": "cylinder", "extent_mm": (20.0, 20.0, 60.0),
                               "offset_mm": (30.0, 0.0, 0.0), "yaw_deg": 0.0})()
        asset = type("R", (), {"source": "procedural", "kind": "composite",
                               "parts": (part, other), "extent_mm": (70.0, 20.0, 60.0),
                               "mesh_path": ""})()
        with mock.patch("datagen.render.mujoco_engine.convex_parts") as never:
            parts = _collision_parts(asset, np.zeros((0, 3)), np.zeros((0, 3), dtype=int), 1.0)
        never.assert_not_called()
        self.assertEqual(len(parts), 2)

    def test_the_render_mesh_of_a_composite_is_its_parts_not_a_box(self) -> None:
        """⛔ THE DIVERGENCE THIS FIXED. `render/isaac.py` authors one collider per part; `_mesh_for`
        read `.parts` nowhere and fell through to `primitive_for_kind("composite")`, which is `box`.
        The two engines rendered a different object, and only `composite_weight`'s 0.0 default kept it
        out of a corpus."""
        from datagen.render.noengine import _mesh_for

        near = type("P", (), {"primitive": "box", "extent_mm": (40.0, 20.0, 20.0),
                              "offset_mm": (0.0, 0.0, 0.0), "yaw_deg": 0.0})()
        far = type("P", (), {"primitive": "box", "extent_mm": (20.0, 20.0, 20.0),
                             "offset_mm": (200.0, 0.0, 0.0), "yaw_deg": 0.0})()
        asset = type("R", (), {"source": "procedural", "kind": "composite",
                               "parts": (near, far), "extent_mm": (40.0, 20.0, 20.0),
                               "mesh_path": ""})()
        vertices, _ = _mesh_for(asset, 1.0)
        span = float(vertices[:, 0].max() - vertices[:, 0].min())
        self.assertGreater(span, 150.0,
                           "the far part is missing, so this rendered as the bounding primitive")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
