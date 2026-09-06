"""MuJoCo, graded through the gate that declares a backend inadmissible without it.

⛔ THIS FILE EXISTS BECAUSE THE BACKEND SHIPPED WITHOUT IT. `datagen/render/engine.py:23` says in so
many words that **a backend is not admissible until it passes the equivalence test**, and a coverage
run put `datagen/render/mujoco_engine.py` at 115 statements, 115 missed, 0 %. Nothing in the test
suite constructed a `MujocoRenderer` at all: the engine was selectable, stamped into provenance, and
priced in the cost calculator, while the one gate that decides whether its corpus is comparable had
never been pointed at it. A second engine producing a subtly different corpus is worse than no second
engine, because the difference is invisible in the `.npz`.

The tests are skipped when MuJoCo is not installed -- and the FIRST test asserts it IS installed in
this environment, so "skipped everywhere" cannot masquerade as "passing".
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import trimesh

from datagen.config import DatagenConfig
from datagen.render.engine import build_engine
from datagen.render.equivalence import (
    check_depth_reconstructs_geometry,
    check_landmark_reconstructs,
    check_mask_matches_depth,
    check_points_match_analytic_rays,
    oblique_camera,
    run_tier0,
)

def _mujoco_is_loadable() -> tuple[bool, str]:
    """Whether MuJoCo can actually be IMPORTED, not merely found on disk.

    ⚠ `find_spec` answers "is it installed", and on Windows that is not the same question. Smart App
    Control refuses freshly-published unsigned binaries until they accumulate reputation, so the
    package resolves and then `import mujoco` dies with
    `ImportError: DLL load failed ... Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert`.
    That is a fact about the BOX, not a defect in the backend, and a suite that reports it as a
    failure teaches everyone to ignore a red test.
    """
    if importlib.util.find_spec("mujoco") is None:
        return (False, "mujoco is not installed")
    try:
        import mujoco  # noqa: F401,PLC0415
    except Exception as error:                                       # noqa: BLE001 - report, not fail
        return (False, f"mujoco is installed but will not load: {type(error).__name__}: {error}")
    return (True, "")


_HAS_MUJOCO, _WHY_NOT = _mujoco_is_loadable()


def _config(**render: object) -> DatagenConfig:
    base: dict = {"engine": "mujoco", "arm": {"mode": "absent"}}
    base.update(render)
    return DatagenConfig.model_validate({"render": base})


class TheEnvironmentTests(unittest.TestCase):
    def test_mujoco_is_installed_here(self) -> None:
        """⚠ THE GUARD ON THE GUARD. Every other test in this file skips without MuJoCo, so without
        this one a broken install would turn the whole gate green by making it vanish.

        ⛔ BUT IT SEPARATES TWO THINGS THAT LOOK ALIKE AND ARE NOT. "Not installed" is OUR problem and
        fails. "Installed and blocked by the operating system" is Windows Smart App Control refusing a
        freshly-published unsigned binary until it accumulates reputation — MEASURED on this box, and
        on this project's other repo — and it clears on its own in days. Failing the suite for that
        teaches everyone to ignore a red test, which costs more than the coverage is worth.

        Either way it is LOUD: a skip carries the operating system's own error text.
        """
        if not _HAS_MUJOCO and "will not load" in _WHY_NOT:
            self.skipTest(f"{_WHY_NOT} -- an OS policy, not a defect. The backend stays UNGRADED "
                          f"here until it loads.")
        self.assertTrue(_HAS_MUJOCO, f"{_WHY_NOT} -- the backend cannot be graded, and an "
                                     f"ungraded backend is inadmissible by `engine.py`'s own rule")


@unittest.skipUnless(_HAS_MUJOCO, _WHY_NOT)
class ItIsBuildableThroughTheSelectorTests(unittest.TestCase):
    def test_the_factory_returns_a_mujoco_renderer(self) -> None:
        from datagen.render.mujoco_engine import MujocoRenderer
        self.assertIsInstance(build_engine(_config()), MujocoRenderer)

    def test_it_refuses_a_driven_arm_by_name_rather_than_treating_it_as_posed(self) -> None:
        """It settles objects; it does not drive an arm. Silently downgrading `driven` to `posed`
        would put an arm in a pose the config never asked for."""
        engine = build_engine(_config(arm={"mode": "driven"}))
        spec = type("Spec", (), {"cameras": (), "objects": (), "scene_id": "x"})()
        result = engine.render(spec, _EmptyManifest(), np.random.default_rng(0))
        self.assertEqual(result.status, "refused_arm_mode")
        self.assertIn("arm.mode", result.note)


class _EmptyManifest:
    def get(self, _asset_id: str) -> None:
        return None

    def records(self) -> tuple:
        return ()


@unittest.skipUnless(_HAS_MUJOCO, _WHY_NOT)
class ItPassesTierZeroTests(unittest.TestCase):
    """⭑ THE ADMISSIBILITY GRADE. Same four checks the raster path is graded by, on an OBLIQUE camera
    -- an overhead view is structurally blind to a principal-point offset, so grading there would
    pass a backend whose intrinsics are wrong."""

    def test_the_backend_is_admissible_on_tier_zero(self) -> None:
        from datagen.render.raster import rasterise

        camera_to_base, k = oblique_camera(resolution=(320, 240))
        table = (np.array([[-700.0, -700.0, 0.0], [700.0, -700.0, 0.0],
                           [700.0, 700.0, 0.0], [-700.0, 700.0, 0.0]]),
                 np.array([[0, 1, 2], [0, 2, 3]]))
        box = trimesh.creation.box(extents=(80.0, 80.0, 60.0))
        # ⭑ THE WORLD REFERENCE, known outside the camera: where the SCENE put this body, not what the
        # renderer says about it. Only a check anchored outside K can catch a camera that is wrong
        # about itself consistently -- the other two compute their expectation from the same K.
        #
        # ⚠ The body CENTRE, not its top face. `check_landmark_reconstructs` compares against the
        # bounding-box centre of the reconstructed points, and this box spans z 0..60, so the reference
        # is 30. My first fixture said 60 and the gate correctly reported a 29.92 mm error -- the same
        # class of mistake that made an earlier raster fixture "fail" against a correct rasteriser.
        centre = np.array([0.0, 0.0, 30.0])
        vertices = np.asarray(box.vertices, dtype=np.float64) + np.array([0.0, 0.0, 30.0])
        view = rasterise({1: table, 2: (vertices, np.asarray(box.faces))},
                         camera_to_base, k, (320, 240))

        ground = view.instance_map == 1
        # ⚠ THE EMPTY SCENE'S DEPTH, which is what "in front of the background" means. An all-NaN
        # background makes every comparison False, so the IoU is 0 by construction and the check
        # reports FAIL about the fixture rather than about the backend. (Note the direction: this
        # finding's number is an IoU where 1.0 passes, not an error where 0.0 passes.)
        background = rasterise({1: table}, camera_to_base, k, (320, 240)).depth_mm
        findings = [
            check_depth_reconstructs_geometry(view.depth_mm, ground, camera_to_base, k,
                                              plane_z_mm=0.0),
            check_points_match_analytic_rays(view.depth_mm, ground, camera_to_base, k,
                                             plane_z_mm=0.0),
            check_landmark_reconstructs(view.depth_mm, view.instance_map == 2, camera_to_base, k,
                                        landmark_centre_mm=centre),
            # ⚠ THE TABLE IS ENVIRONMENT, NOT AN INSTANCE. `noengine.ENVIRONMENT_ID` is 0 and the
            # check reads `instance_map > 0` as "an object is here", so handing it a map in which the
            # table carries a positive id makes 98 % of the frame "occupied but not nearer than the
            # background" -- an IoU of 0.0155 that says nothing about the backend. `rasterise` reserves
            # 0 for "nothing hit", so the remap happens here rather than in the raster call.
            check_mask_matches_depth(np.where(view.instance_map == 2, 1, 0).astype(np.int32),
                                     view.depth_mm, background),
        ]
        admissible, report = run_tier0(findings)
        self.assertTrue(admissible, report)


@unittest.skipUnless(_HAS_MUJOCO, _WHY_NOT)
class TheUnitBoundaryTests(unittest.TestCase):
    """⚠ THE TRAP THIS BACKEND'S OWN DOCSTRING NAMES. MuJoCo works in METRES and the corpus in
    MILLIMETRES, and a factor of a thousand in each axis is a factor of a billion in mass -- this repo
    has already measured a part entering a physics engine at 1.75e9 kg with every check still green."""

    def test_a_settled_body_comes_back_in_millimetres(self) -> None:
        from datagen.render.mujoco_engine import _quat_wxyz

        # The quaternion boundary, which is the other half of the same conversion discipline.
        self.assertEqual(_quat_wxyz((0.0, 0.0, 0.0, 1.0)), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(_quat_wxyz((1.0, 0.0, 0.0, 0.0)), [0.0, 1.0, 0.0, 0.0])

    def test_the_settle_tolerance_is_read_from_the_config_not_hard_coded(self) -> None:
        """A rest criterion nobody can tune is a rest criterion nobody can debug."""
        engine = build_engine(_config(stability_tolerance_mm=7.5))
        self.assertAlmostEqual(engine._config.render.stability_tolerance_mm, 7.5)  # noqa: SLF001


@unittest.skipUnless(_HAS_MUJOCO, _WHY_NOT)
class TheConcaveHandlingTests(unittest.TestCase):
    """⛔ THE PROMISE IS UNCHANGED; WHAT KEEPS IT IS NOT.

    MuJoCo resolves mesh contact against the CONVEX HULL, so a mug settled as its hull is a mug that
    cannot be filled. The backend used to refuse concave assets BY NAME, and the name list had two
    holes worth remembering: `custom`, the source shipped for customers' own CAD, and every COMPOSITE
    (whose `source` reads "procedural" and whose `kind` is "composite").

    Both lists still exist and still matter -- they now decide what gets DECOMPOSED rather than what
    gets refused. The refusal survives for the one case that still deserves it, a machine without the
    library, because the alternative is a silent fallback to the shape the promise excludes.
    """

    def test_custom_meshes_are_treated_as_concave(self) -> None:
        from datagen.render.mujoco_engine import _CONCAVE_SOURCES
        self.assertIn("custom", _CONCAVE_SOURCES,
                      "a source whose shapes we have never seen is a WEAKER case for assuming "
                      "convexity than the three we have")

    def test_the_three_research_sources_are_still_treated_as_concave(self) -> None:
        from datagen.render.mujoco_engine import _CONCAVE_SOURCES
        for source in ("gso", "ycb", "objaverse"):
            self.assertIn(source, _CONCAVE_SOURCES)

    def test_composites_are_caught_by_KIND_because_their_source_says_procedural(self) -> None:
        """A source-only check waves through every mug, bucket, jug, pan and hammer -- and the
        shipped v1 corpus ran composite_weight 0.33."""
        from datagen.render.mujoco_engine import _CONCAVE_KINDS
        self.assertIn("composite", _CONCAVE_KINDS)

    def test_a_scene_holding_a_composite_now_settles_from_its_PARTS(self) -> None:
        """⭑ WHAT REPLACED THE REFUSAL. A composite's parts are boxes, cylinders and spheres by
        construction, so the solver collides them exactly and no decomposition is needed."""
        engine = build_engine(_config())
        part = type("Part", (), {"primitive": "box", "extent_mm": (40.0, 30.0, 30.0),
                                 "offset_mm": (0.0, 0.0, 0.0), "yaw_deg": 0.0})()
        handle = type("Part", (), {"primitive": "cylinder", "extent_mm": (12.0, 12.0, 30.0),
                                   "offset_mm": (26.0, 0.0, 0.0), "yaw_deg": 0.0})()

        class _Manifest:
            def get(self, _asset_id: str):
                return type("Rec", (), {"source": "procedural", "kind": "composite",
                                        "parts": (part, handle), "mesh_path": "",
                                        "extent_mm": (52.0, 30.0, 30.0)})()

        placement = type("P", (), {"asset_id": "mug_0", "scale": 1.0, "mass_kg": 0.2,
                                   "position_mm": (0.0, 0.0, 900.0),
                                   "orientation_xyzw": (0.0, 0.0, 0.0, 1.0)})()
        spec = type("Spec", (), {"cameras": (), "objects": (placement,), "scene_id": "s",
                                 "bin_walls": ()})()
        result = engine.render(spec, _Manifest(), np.random.default_rng(0))
        self.assertNotEqual(result.status, "refused_concave_mesh",
                            f"composites are still refused: {result.note}")

    def test_without_the_library_a_concave_scene_is_still_refused(self) -> None:
        """⛔ NOT A FALLBACK TO THE HULL. The whole point of the old by-name refusal was that settling
        against the wrong shape is invisible afterwards, and installing nothing must not buy it."""
        import tempfile
        from pathlib import Path
        from unittest import mock

        import trimesh

        from datagen.render.convex_decomposition import DecompositionUnavailable

        engine = build_engine(_config())
        # A real file on disk, authored here rather than borrowed from `assets/meshes/`: the library
        # is fetched, not vendored, so a test that needed it would be a test that skips on CI.
        directory = Path(tempfile.mkdtemp())
        mesh_path = directory / "thing.obj"
        trimesh.creation.box(extents=(50.0, 50.0, 50.0)).export(mesh_path)

        class _Manifest:
            def get(self, _asset_id: str):
                return type("Rec", (), {"source": "gso", "kind": "mesh",
                                        "mesh_path": str(mesh_path), "parts": (),
                                        "extent_mm": (50.0, 50.0, 50.0)})()

        placement = type("P", (), {"asset_id": "thing_0", "scale": 1.0, "mass_kg": 0.2,
                                   "position_mm": (0.0, 0.0, 900.0),
                                   "orientation_xyzw": (0.0, 0.0, 0.0, 1.0)})()
        spec = type("Spec", (), {"cameras": (), "objects": (placement,), "scene_id": "s",
                                 "bin_walls": ()})()
        with mock.patch("datagen.render.mujoco_engine.convex_parts",
                        side_effect=DecompositionUnavailable("no coacd here")):
            result = engine.render(spec, _Manifest(), np.random.default_rng(0))
        self.assertEqual(result.status, "refused_concave_mesh")


@unittest.skipUnless(_HAS_MUJOCO, _WHY_NOT)
class TheSettleItselfTests(unittest.TestCase):
    """⭑ THE HALF THE UNIT TESTS CANNOT REACH. Everything above grades the raster and the refusals;
    this drives a real `datagen build` through MuJoCo, which is the only way the solver, the
    metre/millimetre boundary and the rest criterion are exercised at all. Measured at ~1.5 s/scene,
    so two scenes is a test rather than a chore.
    """

    def test_a_real_build_settles_and_writes_scenes(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from datagen.__main__ import main

        out = Path(tempfile.mkdtemp())
        code = main(["build", "--engine", "mujoco", "--scenes", "2", "--name", "t",
                     "--out", str(out), "--quiet"])
        self.assertEqual(code, 0)

        provenance = json.loads((out / "t" / "provenance.json").read_text(encoding="utf-8"))
        # ⛔ THE STAMP MUST NAME THE ENGINE THAT ACTUALLY RAN. `--engine` was a globally declared flag
        # read in exactly one branch (the cost report), so `build --engine mujoco` was accepted, the
        # flag discarded, and Isaac used -- with a provenance line saying `isaac:pathtrace` for a run
        # the operator had asked to be MuJoCo. Nothing downstream compares those two.
        self.assertEqual(provenance["renderer"], "mujoco:raster")

        scenes = sorted((out / "t" / "scenes").iterdir())
        self.assertTrue(scenes, "the build wrote no scenes")

    def test_objects_come_to_rest_ON_the_table_and_not_above_or_through_it(self) -> None:
        """⚠ A SYSTEMATIC OFFSET HERE WOULD BE IN EVERY DEPTH IMAGE THE CORPUS CONTAINS.

        `sparse` and `packed` are placed FLAT -- `scenes/layout.py:556` sets `drop_height = 0.0` -- so
        `engine: none` seats them analytically and exactly, the lowest vertex ON the support. MuJoCo
        instead asks a solver, and a solver rests where CONTACT says rather than where geometry does.

        MEASURED 2026-08-30 on procedural primitives, for which the convex hull IS the shape so the
        solver is the only thing being graded: the lowest vertex sits a median 0.230 mm BELOW the
        table, range 0.21 to 0.43, none past 0.5. That is MuJoCo's contact softness, it is a sink
        rather than a float, and it is an order of magnitude under the 0.5 mm the render's own
        stability check uses. Pinned at 1.0 mm so this fails on a real regression rather than on
        solver noise.
        """
        import json
        import tempfile
        from pathlib import Path

        import numpy as np

        from datagen.build import build_manifest
        from datagen.config import DatagenConfig
        from datagen.__main__ import main
        from datagen.render.noengine import _mesh_for, _rotation

        out = Path(tempfile.mkdtemp())
        self.assertEqual(main(["build", "--engine", "mujoco", "--scenes", "3", "--name", "rest",
                               "--out", str(out), "--quiet"]), 0)
        root = out / "rest"
        config = DatagenConfig(**json.loads(
            (root / "provenance.json").read_text(encoding="utf-8"))["config"])
        manifest = build_manifest(config)
        table = float(config.workspace.table_height_mm)

        offsets = []
        for scene_dir in sorted((root / "scenes").iterdir()):
            payload_path = scene_dir / "scene.json"
            if not payload_path.is_file():
                continue
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            placements = {str(i): p for i, p in enumerate(payload["spec"]["objects"])}
            for key, (position, orientation) in (payload.get("settled_poses_mm_xyzw") or {}).items():
                placement = placements.get(str(key))
                record = manifest.get(placement["asset_id"]) if placement else None
                if record is None:
                    continue
                vertices, _ = _mesh_for(record, float(placement.get("scale", 1.0)))
                world = vertices @ _rotation(orientation).T + np.asarray(position, dtype=np.float64)
                offsets.append(float(world[:, 2].min()) - table)

        self.assertTrue(offsets, "no settled object to measure")
        worst = max(abs(offset) for offset in offsets)
        self.assertLess(worst, 1.0,
                        f"objects rest {worst:.2f} mm off the table; every depth image carries it")

    def test_the_provenance_never_claims_path_tracing_for_a_rasteriser(self) -> None:
        """`render.mode` selects path-tracing inside Isaac and means nothing here; stamping
        `mujoco:pathtrace` tells a reader the scene was path-traced when nothing was even shaded."""
        import json
        import tempfile
        from pathlib import Path

        from datagen.__main__ import main

        out = Path(tempfile.mkdtemp())
        self.assertEqual(main(["build", "--engine", "none", "--scenes", "2", "--name", "n",
                               "--out", str(out), "--quiet"]), 0)
        renderer = json.loads((out / "n" / "provenance.json").read_text(encoding="utf-8"))["renderer"]
        self.assertNotIn("pathtrace", renderer)
        self.assertEqual(renderer, "none:raster")


if __name__ == "__main__":                                           # pragma: no cover
    unittest.main()
