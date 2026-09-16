"""A descriptor's arm spheres come from this repository's own cover fit, and a map with a hole writes nothing (B6).

Until now the arm sphere map was Isaac's Lula family, repaired in the builder: vendor spheres dropped when they reached
too far past their link's hull, surface spheres added where the vendor's were thin. That repair is measured and it
works, but it starts from a compromise somebody else struck, and the family it ships reaches 0 to 61 mm past its own
links, leaves gaps up to 45 mm once clamped, and on ur10 belongs to a different link-frame family altogether.

The cover fit starts from the same collision bundle the exact mesh guard judges against, and makes the safe half a
property of the construction: every surface sample of every link ends up inside a sphere, or there is no map. So the
builder reads the committed fit where there is one, and Isaac only where there is not.

⛔ What is tested hardest here is the refusal. A HOLE is the unsafe error, the fitter cannot produce one, and a file
claiming one has been edited or truncated: then the right answer is no descriptor at all, not a descriptor whose arm is
invisible in one place.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_MAPS = _ROOT / "src" / "robot" / "safety" / "planning" / "robot"
_BUNDLES = _ROOT / "src" / "robot" / "safety" / "data"


def _arm_spheres():
    path = _ROOT / "scripts" / "curobo" / "_arm_spheres.py"
    spec = importlib.util.spec_from_file_location("_arm_spheres_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _a_map(uncovered_mm: float = -1.5) -> dict:
    """A cover-fit map shaped as the fitter writes one, with the two fields the reader judges."""
    return {
        "_provenance": {
            "what": "urX arm links",
            "bodies": [{"body": "wrist_1", "fresh_uncovered_max_mm": uncovered_mm, "fresh_reach_max_mm": 8.0}],
        },
        "collision_spheres": {
            "wrist_1": {"frame": 4, "spheres": [{"center": [0.0, 0.0, 0.01], "radius": 0.03}]},
        },
    }


class TheReaderTranslatesAndRefuses(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _arm_spheres()

    def test_a_body_name_becomes_the_link_name_a_descriptor_uses(self) -> None:
        """The bundle says `wrist_1` because that is what it measured; cuRobo says `wrist_1_link`."""
        read = self.module.link_spheres(_a_map(), source="test.yml")

        self.assertEqual(sorted(read), ["wrist_1_link"])
        self.assertEqual(read["wrist_1_link"], [{"center": [0.0, 0.0, 0.01], "radius": 0.03}])

    def test_a_map_recording_a_hole_is_refused_by_name(self) -> None:
        """⭐ THE ONE THAT MATTERS. A hole is a false clear, and a false clear is the arm somewhere the model is empty."""
        with self.assertRaises(self.module.CoverMapError) as raised:
            self.module.link_spheres(_a_map(uncovered_mm=3.2), source="ur5e_arm_spheres.yml")

        said = str(raised.exception)
        self.assertIn("hole", said)
        self.assertIn("wrist_1", said)
        self.assertIn("ur5e_arm_spheres.yml", said)

    def test_a_link_with_an_empty_list_is_refused_rather_than_counted_as_present(self) -> None:
        """⛔ COUNT THE SPHERES, NOT THE KEYS. A present key with no spheres is a link nothing checks, and it passes
        every emptiness test written against the keys. This repository has written that descriptor once already."""
        document = _a_map()
        document["collision_spheres"]["wrist_1"]["spheres"] = []

        with self.assertRaises(self.module.CoverMapError) as raised:
            self.module.link_spheres(document, source="test.yml")

        self.assertIn("wrist_1", str(raised.exception))

    def test_a_document_with_no_spheres_at_all_is_refused(self) -> None:
        with self.assertRaises(self.module.CoverMapError):
            self.module.link_spheres({"_provenance": {}}, source="test.yml")

    def test_the_label_says_what_the_map_is_and_what_it_cost(self) -> None:
        label = self.module.cover_spheres_label(source="ur5e_arm_spheres.yml", reach_mm=30.0, spheres=339)

        self.assertIn("cover fit", label)
        self.assertIn("ur5e_arm_spheres.yml", label)
        self.assertIn("339", label)
        self.assertIn("30.0 mm", label)
        self.assertNotEqual(label, self.module.arm_spheres_label(bundle=True, recipe="surface", bound_mm=60.0),
                            "a fitted map and a repaired vendor map must not describe themselves the same way")


class EveryArmWithGeometryHasOne(unittest.TestCase):
    """The property, not a file list: whatever arm this repository has baked a bundle for, the planner has its map."""

    def test_every_baked_arm_has_a_committed_cover_fit(self) -> None:
        bundles = sorted(_BUNDLES.glob("*_collision_meshes.npz"))
        self.assertGreaterEqual(len(bundles), 5, "no arm bundles were read, so this property checks nothing")
        for bundle in bundles:
            arm = bundle.name.replace("_collision_meshes.npz", "")
            with self.subTest(arm=arm):
                self.assertTrue(
                    (_MAPS / f"{arm}_arm_spheres.yml").is_file(),
                    f"{arm} has exact geometry and no committed sphere map, so its descriptor would fall back to "
                    f"Isaac's. Fit one: scripts/curobo/fit_cover_spheres.py --arm {arm} --write")

    def test_every_committed_map_covers_every_link_the_template_guards(self) -> None:
        """cuRobo raises KeyError on a collision link with no spheres, at the first load and not before."""
        guarded = {"shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"}
        maps = sorted(_MAPS.glob("*_arm_spheres.yml"))
        self.assertGreaterEqual(len(maps), 5, "no committed arm maps were read")
        module = _arm_spheres()
        for path in maps:
            with self.subTest(arm=path.stem):
                read = module.link_spheres(yaml.safe_load(path.read_text(encoding="utf-8")), source=path.name)
                self.assertEqual(guarded, set(read), f"{path.name} does not describe the links the template guards")

    def test_every_committed_map_records_no_hole_and_the_reach_it_asked_for(self) -> None:
        """What the fit measured on points it never saw, read back from the file it wrote."""
        for path in sorted(_MAPS.glob("*_arm_spheres.yml")):
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            for row in document["_provenance"]["bodies"]:
                with self.subTest(arm=path.stem, body=row["body"]):
                    self.assertLess(row["fresh_uncovered_max_mm"], 0.0,
                                    "a surface point of this link lies outside every sphere")
                    self.assertLessEqual(row["fresh_reach_max_mm"], row["reach_mm"] + 0.1,
                                         "a sphere reaches further past the link than the map was fitted at")

    def test_two_arms_do_not_share_one_map(self) -> None:
        """The negative half. Without it every assertion above would pass on one arm copied six times.

        ⚠ It compares FOREARMS on purpose. ur10e and ur16e carry the same wrist castings, and their wrist_1,
        wrist_2 and wrist_3 meshes agree to 2.8e-14 mm, so their wrist spheres are identical and that is the
        right answer rather than a copied file. The forearms differ by design: 281 spheres against 183.
        """
        seen: dict[str, str] = {}
        for path in sorted(_MAPS.glob("*_arm_spheres.yml")):
            body = yaml.safe_load(path.read_text(encoding="utf-8"))["collision_spheres"]["forearm"]
            key = repr(body["spheres"][:3])
            self.assertNotIn(key, seen, f"{path.name} has the same forearm spheres as {seen.get(key)}")
            seen[key] = path.name


class TheSpheresLandWhereTheArmIs(unittest.TestCase):
    """⛔⛔ THE DEFECT THIS EXISTS FOR, AND IT SHIPPED FOR AN HOUR ON 2026-09-16.

    A committed bundle holds each link's mesh in that link's own DH frame. A cuRobo descriptor holds each
    link's spheres in the URDF LINK frame. The two differ per link: measured on the ur5e, the upper arm's
    frames sit 425 mm apart ALONG the link, the shoulder's differ by a quarter turn about x. A map handed
    over untransformed puts the upper arm's spheres 425 mm from the upper arm.

    Nothing downstream measures orientation. The descriptor loaded, the planner planned, the ready gate said
    ready for all 28 arm and hand pairs, and the spheres were somewhere the arm is not. What caught it was the
    planner's own boot world: a mirrored arm dips under the table the sidecar stands on, and the refusal said
    it reached 847.5 mm into a world whose only object is a slab 1 mm below the base plate.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _arm_spheres()

    def _placement(self, model: str):
        import sys as _sys

        path = _ROOT / "scripts" / "curobo"
        _sys.path.insert(0, str(path))
        try:
            import _ur_description  # type: ignore[import-not-found]

            urdf = _ur_description.render_urdf(model)
            rows = _ur_description.dh_rows(model)
        finally:
            _sys.path.remove(str(path))
        return self.module.link_frames(urdf, rows), urdf

    def test_the_two_frames_are_not_the_same_frame(self) -> None:
        """And they differ in a different way per link, which is why there is no shortcut here.

        Measured on the ur5e: the upper arm's DH frame sits 425 mm along the link from its URDF frame, which
        is the link's own length; the shoulder's two frames differ by a quarter turn about x, which is the
        alpha of its DH row. A build that skipped this put the upper arm's spheres 425 mm away from it.
        """
        place, _ = self._placement("ur5e")

        upper = place("upper_arm_link", 2)
        shoulder = place("shoulder_link", 1)

        self.assertAlmostEqual(float(upper[0, 3]), -0.425, places=3,
                               msg=f"the upper arm frames do not sit a link length apart: {upper[:3, 3]}")
        self.assertFalse(np.allclose(shoulder[:3, :3], np.eye(3)), 
                         f"the shoulder frames do not differ by a rotation: {shoulder[:3, :3]}")

    def test_the_committed_fit_lands_where_the_vendor_puts_its_own_spheres(self) -> None:
        """⭐ THE CROSS-CHECK, against geometry nobody here authored. Isaac's map for the same link is in the
        URDF link frame by construction; the placed fit has to occupy the same box, not its mirror image.

        Measured 2026-09-16 on the ur5e: the vendor's upper_arm_link centres run x from -426 to +1 mm, and the
        fit's own DH-frame centres run -49 to +423. Untransformed they overlap the vendor's range by nothing.
        """
        place, _ = self._placement("ur5e")
        document = yaml.safe_load((_MAPS / "ur5e_arm_spheres.yml").read_text(encoding="utf-8"))
        block = document["collision_spheres"]["upper_arm"]

        placed = self.module.place_spheres(block["spheres"], place("upper_arm_link", block["frame"]))

        x = np.asarray([sphere["center"][0] for sphere in placed])
        self.assertLess(float(x.max()), 0.05,
                        f"the placed upper arm still reaches along +x, so it is mirrored: max {x.max():.3f} m")
        self.assertGreater(float(x.min()), -0.50,
                          f"the placed upper arm reaches further back than the link is long: {x.min():.3f} m")

    def test_placing_moves_the_centres_and_leaves_the_radii_alone(self) -> None:
        """A rigid motion. A radius that changed would mean the transform carried a scale."""
        M = np.eye(4)
        M[:3, 3] = [1.0, 2.0, 3.0]
        spheres = [{"center": [0.1, 0.2, 0.3], "radius": 0.04}]

        placed = self.module.place_spheres(spheres, M)

        self.assertEqual(placed[0]["radius"], 0.04)
        self.assertAlmostEqual(placed[0]["center"][0], 1.1)
        self.assertAlmostEqual(placed[0]["center"][2], 3.3)
        self.assertEqual(self.module.place_spheres([], M), [])


class TheBuilderPrefersTheFit(unittest.TestCase):
    """The builder runs top to bottom in the cuRobo environment and cannot be imported here, so its source is read.

    A weaker check than calling it, and the reason the translation and the refusal moved into `_arm_spheres.py` where
    the tests above call them for real. What is left to check here is only which branch the builder takes first.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (_ROOT / "scripts" / "curobo" / "build_ur_config.py").read_text(encoding="utf-8")

    def test_the_committed_fit_is_read_before_isaac_is_asked_for_anything(self) -> None:
        fit = self.source.index("_FITTED_MAP = ")
        lula = self.source.index('_lula_src = _isaac_model_dir()')

        self.assertLess(fit, lula, "Isaac is consulted before the committed map, so the fit would never be used")

    def test_the_surface_augmentation_does_not_run_over_a_fitted_map(self) -> None:
        """It exists to repair the vendor's map. Over a cover fit it would add spheres, reach and plan time."""
        self.assertIn("if _fitted_doc is not None:\n    print(\"arm-link surface augmentation skipped", self.source)
        self.assertIn("elif _ARM_NPZ.is_file():", self.source)


if __name__ == "__main__":
    unittest.main()
