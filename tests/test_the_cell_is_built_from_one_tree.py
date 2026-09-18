"""A cell's robot half and its camera half come from the same config tree.

⛔ **MEASURED 2026-09-10, AND THREE AGENTS FOUND IT INDEPENDENTLY.** `real_cell --profile ur3e`
resolved the robot through `load_robot_config(data_dir, profile=...)`, and then
`build_real_components` read the camera section with a bare `load_config()` at cells.py, and
`build_real_cell` did it again for `primary_rig_id`. A no-argument load takes `profile=UNSET`, which
falls back to `WILLY_PROFILE`, and `data_dir=None`, which falls back to the checkout's own tree. So
one command built a robot from the flag and cameras from the environment.

Measured consequences, all three reproduced: the same chain through `--profile` and through
`WILLY_PROFILE` produced two DIFFERENT refusals; a scratch tree that could build was refused quoting
the repository tree's rigs; and there is no environment escape hatch for `--data-dir` at all, so an
operator pointing the console at a deployment tree got that tree's arm and this checkout's cameras.

⭐ **THE COMMENT AT THE SECOND SITE WAS TRUE AND NOT ABOUT THE RIGHT THING.** It read "the same loader
`build_real_components` used, and it caches, so this is the same object rather than a second read of
the tree", which is correct about the cache and silent about WHICH tree was cached. Two calls agreeing
with each other is not the same as either one being right.

What this file pins is the property: when a caller supplies the application config, nothing downstream
re-reads it. A test that only compared rig ids would pass on a box whose default tree happens to match.
"""

from __future__ import annotations

import ast
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.config import load_config, load_robot_config
from src.robot.execution.autonomous_grasp import cells


class BothHalvesComeFromTheSuppliedTreeTests(unittest.TestCase):
    def test_the_builder_does_not_re_read_the_tree_it_was_given(self) -> None:
        """The property, asserted structurally so a matching default cannot hide it.

        `build_real_components` opens a camera, so this stops it before the device: what is measured
        is whether the loader was consulted, not whether a cell came up.
        """
        app_cfg = load_config()
        robot = load_robot_config()
        loads: list = []

        def watched(*args, **kwargs):
            loads.append((args, kwargs))
            return app_cfg

        # Patched at the module that owns it, so a builder importing it by any route is covered.
        with mock.patch("src.config.load_config", side_effect=watched):
            with mock.patch(
                "src.camera.orchestration.camera.Camera"
            ) as camera:
                camera.from_config.side_effect = RuntimeError("stop before the device")
                with self.assertRaises(Exception):
                    cells.build_real_components(robot, "a box", app_config=app_cfg)

        self.assertEqual(
            loads, [],
            "the builder was handed an application config and read the tree anyway",
        )

    def test_the_primary_rig_comes_from_the_supplied_config(self) -> None:
        """The visible half of the same property.

        A config naming a rig the default tree does not carry proves the supplied object was the one
        consulted, whatever the checkout holds.
        """
        app_cfg = load_config()
        rigs = app_cfg.camera.cameras.rigs
        if not rigs:
            self.skipTest("this tree names no camera rigs")
        invented = "rig_that_no_shipped_tree_carries"
        self.assertNotIn(invented, [getattr(r, "rig_id", None) for r in rigs])

        # The schema is frozen, so a new object rather than an assignment: `model_copy(update=...)`
        # at each level down to the field, which is also how the runtime builds an overlay.
        cameras = app_cfg.camera.cameras.model_copy(update={"primary_rig_id": invented})
        camera = app_cfg.camera.model_copy(update={"cameras": cameras})
        patched = app_cfg.model_copy(update={"camera": camera})
        robot = load_robot_config()

        with self.assertRaises(Exception) as caught:
            cells.build_real_components(robot, "a box", app_config=patched)
        self.assertIn(
            invented, str(caught.exception),
            "the refusal quoted a rig from some other tree than the one supplied",
        )


class NoCallerHadToChangeTests(unittest.TestCase):
    """Supplying nothing keeps the old behaviour exactly, which is what every existing caller does.

    The repair adds a door rather than moving one: `UNSET` means the caller did not choose a tree,
    and the default tree is then the honest answer. The defect was never that the default existed, it
    was that a caller who HAD chosen could not say so.
    """

    def test_omitting_the_config_still_reads_the_default_tree(self) -> None:
        robot = load_robot_config()
        seen: list = []

        real = cells.load_config if hasattr(cells, "load_config") else load_config

        def watched(*args, **kwargs):
            seen.append(True)
            return real(*args, **kwargs)

        with mock.patch("src.config.load_config", side_effect=watched):
            with mock.patch(
                "src.camera.orchestration.camera.Camera"
            ) as camera:
                camera.from_config.side_effect = RuntimeError("stop before the device")
                with self.assertRaises(Exception):
                    cells.build_real_components(robot, "a box")
        self.assertTrue(seen, "the default path must still read the tree")


# Customer chain lane C1g: the tree's hand is the repository's, at the desk and at the build.

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "config"
_DIFFERS = "describes the hand robotiq_hande differently"


def _scratch(case: unittest.TestCase, *, moved: bool) -> Path:
    from src.config import reload_config

    root = Path(case.enterContext(tempfile.TemporaryDirectory())) / "data"
    shutil.copytree(_DATA, root)
    if moved:
        path = root / "grippers" / "robotiq_hande.yaml"
        path.write_text(path.read_text(encoding="utf-8").replace("aperture_mm: 49.99", "aperture_mm: 50.99"),
                        encoding="utf-8")
    reload_config()
    return root


def _hand_row(report: object) -> object:
    (row,) = [check for check in report.checks if check.name == "hand"]  # type: ignore[attr-defined]
    return row


class TheTreeHandIsTheRepositoryHandTests(unittest.TestCase):
    def test_the_desk_names_a_tree_that_describes_its_hand_differently(self) -> None:
        from src.robot.execution.cell import Cell

        root = _scratch(self, moved=True)
        cell = Cell.from_robot_config(load_robot_config(root, profile="hande"),
                                      app_config=load_config(root, profile="hande"), data_dir=root)
        row = _hand_row(cell.preflight())
        self.assertEqual(str(row.status), "block")  # type: ignore[attr-defined]
        for part in (_DIFFERS, "config/grippers/robotiq_hande.yaml", "aperture_mm"):
            self.assertIn(part, row.detail)  # type: ignore[attr-defined]

    def test_an_identical_copy_of_the_repository_tree_reads_no_disagreement(self) -> None:
        """⭐ THE CONTROL: the same flow over an unmodified copy says nothing about the tree's hand."""
        from src.robot.execution.cell import Cell

        root = _scratch(self, moved=False)
        cell = Cell.from_robot_config(load_robot_config(root, profile="hande"),
                                      app_config=load_config(root, profile="hande"), data_dir=root)
        self.assertNotIn("differently", _hand_row(cell.preflight()).detail)  # type: ignore[attr-defined]

    def test_a_cell_built_from_that_tree_is_refused_before_anything_opens(self) -> None:
        root = _scratch(self, moved=True)
        robot = load_robot_config(root, profile="hande").model_copy(update={"vendor": "dummy"})
        with self.assertRaises(cells.CellBuildRefused) as rehearsal:
            cells.build_rehearsal_cell(robot, data_dir=root)
        self.assertIn(_DIFFERS, str(rehearsal.exception))
        with mock.patch("src.camera.orchestration.camera.Camera") as camera, \
                self.assertRaises(cells.CellBuildRefused) as real:
            camera.from_config.side_effect = AssertionError("a camera opened before the refusal")
            cells.build_real_cell(load_robot_config(root, profile="hande"),
                                  app_config=load_config(root, profile="hande"), data_dir=root)
        self.assertIn(_DIFFERS, str(real.exception))
        camera.from_config.assert_not_called()

    def test_the_build_refusal_is_about_the_tree_and_not_the_hand(self) -> None:
        """⭐ THE CONTROL: the identical copy passes the tree check, whatever the rest of the build then says."""
        root = _scratch(self, moved=False)
        robot = load_robot_config(root, profile="hande").model_copy(update={"vendor": "dummy"})
        try:
            cells.build_rehearsal_cell(robot, data_dir=root)
        except Exception as exc:  # noqa: BLE001 (only the tree's sentence is under test)
            self.assertNotIn("differently", str(exc))


def _calls_without_a_tree(path: Path, names: set[str]) -> list[str]:
    missing = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name in names and not any(keyword.arg == "data_dir" for keyword in node.keywords):
            missing.append(f"{name}:{node.lineno}")
    return missing


class EveryDeskSiteHandsDownItsTreeTests(unittest.TestCase):
    """A desk that resolves the hand without the cell's tree reads OK over a tree the build then refuses.

    Left out on purpose: drivers/ur/arm.py resolves the repository registry, which every tree-built cell compared
    first; the sim runners load the repository tree; scripts/curobo and scripts/grippers write package artefacts
    from the repository registry. The doctor is in (owner decision of 2026-09-17): it reads a loaded cell.
    """

    _NAMES = {"run_config_preflight", "planner_hand", "desk_evidence_refusal"}
    _SITES = (
        "src/robot/execution/cell.py",
        "src/robot/execution/real_cell/preflight.py",
        "src/robot/safety/planning/evidence.py",
        "src/robot/safety/planning/doctor.py",
        "api/cell.py",
        "api/routers/preflight.py",
    )

    def test_every_desk_site_hands_down_its_tree(self) -> None:
        for relative in self._SITES:
            with self.subTest(file=relative):
                self.assertEqual(_calls_without_a_tree(_REPO / relative, self._NAMES), [])

    def test_the_scan_sees_a_call_without_a_tree(self) -> None:
        """⭐ THE CONTROL: one call with and one without, reported exactly once."""
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "site.py"
        path.write_text("planner_hand(cfg)\nplanner_hand(cfg, data_dir=root)\n", encoding="utf-8")
        self.assertEqual(_calls_without_a_tree(path, self._NAMES), ["planner_hand:1"])

    def test_the_doctor_hands_its_tree_to_the_combination_probe(self) -> None:
        from src.contracts import UNSET
        from src.robot.safety.planning import doctor

        ok = doctor.Probe("engine", doctor.ProbeStatus.OK, "patched")
        with mock.patch.object(doctor, "_probe_planner_combination", return_value=()) as probe, \
                mock.patch.object(doctor, "_probe_coal", lambda blocks: ok), \
                mock.patch.object(doctor, "_probe_curobo", lambda *a, **k: (ok,)), \
                mock.patch.object(doctor, "code_integrity_blocks", lambda: ()):
            doctor.run_doctor(model="ur5e", robot_config="willy_ur5e.yml", gripper=None, cell=UNSET,
                              data_dir="/a/tree")
        self.assertEqual(probe.call_args.kwargs["data_dir"], "/a/tree")


if __name__ == "__main__":
    unittest.main()
