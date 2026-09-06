"""Generating a dataset from code, without building a command line and parsing it back.

⭐ WHAT THIS IS FOR. The only code-level entry point into datagen was `__main__.main(argv)`, a shim
taking a list of STRINGS, so "call datagen from Python" meant assembling a command line. A customer
who wants their own script had to write one for a parser to take apart again.

⚠ THESE TESTS DO NOT RENDER. Rendering is the expensive step and none of what is checked here is
about pixels: the merge, the defaults, the stage ordering and the reports are all decidable without
one. The one end-to-end run that proves the chain lives in the commit message, measured rather than
asserted: 4 scenes, 30 objects, 717 labels, engine `none`, no CLI argument anywhere.
"""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.contracts.options import UNSET, _Unset
from datagen.api import DatasetBuild, DatasetReport, Stage, StageResult, merged_settings
from datagen.config import CameraSpec, DatagenConfig


class TheMergeHasOneImplementationTests(unittest.TestCase):
    """⛔⛔ IT LIVED ONLY IN THE CLI, as `__main__._config(args)` reading an argparse.Namespace.

    So the one place that knows `--engine` folds into the NESTED `render.engine` was unreachable
    from Python, and a code caller who wrote `{"engine": "mujoco"}` at the top level got a config
    that silently kept Isaac.

    ⚠ THAT FLAG HAS ALREADY COST A RUN. It was declared globally and read in exactly one branch, so
    `datagen build --engine mujoco` was accepted by argparse, discarded, and rendered on Isaac. The
    only signal was a provenance.json saying `isaac:pathtrace` for a run somebody asked to be MuJoCo,
    and nothing downstream compares those two.
    """

    def test_engine_lands_nested_under_render(self) -> None:
        base, applied = merged_settings(engine="mujoco")
        self.assertEqual({"render": {"engine": "mujoco"}}, base)
        self.assertEqual("mujoco", applied["render.engine"])

    def test_the_cli_and_the_api_produce_the_same_config(self) -> None:
        """The point of sharing the implementation, checked rather than asserted in a comment."""
        import sys

        from datagen.__main__ import _config, build_parser                    # noqa: PLC0415

        argv = sys.argv
        try:
            sys.argv = ["datagen"]
            args = build_parser().parse_args(
                ["build", "--name", "x", "--engine", "mujoco", "--scenes", "7", "--seed", "3"])
            from_cli = _config(args)                                          # noqa: SLF001
        finally:
            sys.argv = argv
        from_api = DatasetBuild.from_file(None, name="x", engine="mujoco", scenes=7, seed=3).config
        self.assertEqual(from_cli.model_dump(), from_api.model_dump())

    def test_overrides_merge_one_level_and_are_recorded(self) -> None:
        """A caller handing in a camera_rig block must not have to restate the rest of it."""
        base, applied = merged_settings(
            None, scenes=5, overrides={"camera_rig": {"resolution": (320, 240)}})
        self.assertEqual((320, 240), base["camera_rig"]["resolution"])
        self.assertEqual(5, applied["scenes"])
        self.assertEqual("overridden", applied["camera_rig"])

    def test_a_file_is_read_and_overrides_win(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "cell.json"
            path.write_text(json.dumps({"scenes": 100, "seed": 1}), encoding="utf-8")
            base, _ = merged_settings(path, scenes=250)
        self.assertEqual((250, 1), (base["scenes"], base["seed"]))


class TheWrapperDeclaresNoCompetingDefaultsTests(unittest.TestCase):
    """⛔⛔ A DEFAULT IS A CLAIM ABOUT WHAT SOMEONE ELSE DOES.

    These verbs wrap `build_dataset`, `label_dataset` and `build_cloud_corpus`. Nine defaults were
    duplicated here on the first draft and all nine AGREED, which is exactly the state in which a
    drift is invisible: agreement today proves nothing about tomorrow, and both sides are valid
    values of the right type so nothing raises when they part.

    A peer session hit the live version an hour earlier: a wrapper declaring `seed: int = 0` where
    every trainer declares `seed: int = 1`. Same nominal call, DIFFERENT artifact sha256, no test
    failed, and the wrong number looked exactly as plausible as the right one.

    ⚠ THE HARDEST OF THE NINE TO SEE BY EYE was `density`: a string key here against a resolved
    `LabelDensity` object there. They agree only because `DENSITIES["default"] is DEFAULT_DENSITY`
    today, and the same object scores 84 labels at one setting and 1,960 at another.
    """

    def test_no_verb_redeclares_a_default_the_callee_owns(self) -> None:
        from datagen.build import build_dataset                               # noqa: PLC0415
        from datagen.corpus.clouds import build_cloud_corpus                  # noqa: PLC0415
        from datagen.grasps.labels import label_dataset                       # noqa: PLC0415

        competing: list[str] = []
        for verb, wrapped in (("render", build_dataset), ("label", label_dataset),
                              ("clouds", build_cloud_corpus)):
            theirs = inspect.signature(wrapped).parameters
            for name, mine in inspect.signature(getattr(DatasetBuild, verb)).parameters.items():
                if name == "self" or mine.default is inspect.Parameter.empty:
                    continue
                other = theirs.get(name)
                if other is None or other.default is inspect.Parameter.empty:
                    continue
                if not isinstance(mine.default, _Unset) and mine.default is not None:
                    competing.append(f"{verb}.{name}={mine.default!r} vs "
                                     f"{wrapped.__name__}.{name}={other.default!r}")
        self.assertEqual([], competing,
                         "a wrapper default competes with the wrapped function's; the wrapped one "
                         "is the single declaration and this layer must forward, not restate")

    def test_an_unchosen_argument_never_reaches_the_callee(self) -> None:
        """The mechanism. Forwarding `None` would be a choice; forwarding nothing is not."""
        seen: dict[str, object] = {}

        def capture(config, **kwargs):                # type: ignore[no-untyped-def]
            seen.update(kwargs)
            return {"by_status": {"ok": 1}}

        build = DatasetBuild.from_config(DatagenConfig(), name="x")
        with mock.patch("datagen.build.build_dataset", side_effect=capture):
            build.render()
        self.assertNotIn("headless", seen)
        self.assertNotIn("preview", seen)

    def test_a_chosen_argument_does_reach_it(self) -> None:
        """The control. Without it the test above passes for a verb that forwards nothing at all."""
        seen: dict[str, object] = {}

        def capture(config, **kwargs):                # type: ignore[no-untyped-def]
            seen.update(kwargs)
            return {"by_status": {"ok": 1}}

        build = DatasetBuild.from_config(DatagenConfig(), name="x")
        with mock.patch("datagen.build.build_dataset", side_effect=capture):
            build.render(headless=False)
        self.assertIs(False, seen["headless"])

    def test_UNSET_is_falsy_and_is_not_False(self) -> None:
        self.assertFalse(bool(UNSET))
        self.assertIsNot(UNSET, False)
        self.assertNotIsInstance(UNSET, bool)


class TheChainStopsAtTheFirstFailureTests(unittest.TestCase):
    """⛔ IT STOPS RATHER THAN CONTINUING. Labelling a directory nothing rendered into produces a
    report saying zero labels, which reads like a labelling problem and is not one."""

    @staticmethod
    def _build() -> DatasetBuild:
        return DatasetBuild.from_config(DatagenConfig(), name="x", out_root=tempfile.mkdtemp())

    def test_a_failed_render_stops_the_run(self) -> None:
        build = self._build()
        with mock.patch("datagen.build.build_dataset",
                        return_value={"by_status": {"render_error": 3}}), \
                mock.patch("datagen.grasps.labels.label_dataset") as labeller:
            report = build.run()
        labeller.assert_not_called()
        self.assertFalse(report.succeeded)
        self.assertEqual([Stage.RENDER], [s.stage for s in report.stages])
        self.assertIn("no scene rendered", report.failure_summary())

    def test_a_good_render_reaches_the_labeller(self) -> None:
        build = self._build()
        with mock.patch("datagen.build.build_dataset", return_value={"by_status": {"ok": 4}}), \
                mock.patch("datagen.grasps.labels.label_dataset",
                           return_value={"jaw": 252, "suction": 465}):
            report = build.run()
        self.assertEqual([Stage.RENDER, Stage.LABEL], [s.stage for s in report.stages])
        self.assertTrue(report.succeeded)

    def test_the_corpus_step_runs_only_when_asked_for(self) -> None:
        build = self._build()
        with mock.patch("datagen.build.build_dataset", return_value={"by_status": {"ok": 1}}), \
                mock.patch("datagen.grasps.labels.label_dataset", return_value={"jaw": 1}), \
                mock.patch("datagen.corpus.clouds.build_cloud_corpus") as clouds:
            build.run()
        clouds.assert_not_called()

    def test_the_corpus_step_reports_SUCCESS_when_it_wrote_something(self) -> None:
        """⛔⛔ THE MISSING CASE, AND ITS ABSENCE HID A LIVE DEFECT FOR A DAY. The verb read
        `summary["written"]`; `build_cloud_corpus` returns `scenes_written`. A missing key returns
        the fallback, so a perfect corpus reported ok=False with the reason "no cloud was written",
        and nothing raised, because 0 is a legal value of the right type.

        The sibling test above asserts the step is NOT called when unasked. That is the control. This
        is the case, and a control without its case passes for a verb that never works at all.
        """
        build = self._build()
        with mock.patch("datagen.build.build_dataset", return_value={"by_status": {"ok": 1}}),                 mock.patch("datagen.grasps.labels.label_dataset", return_value={"jaw": 1}),                 mock.patch("datagen.corpus.clouds.build_cloud_corpus",
                           return_value={"scenes_written": 4, "grasps": 900}):
            report = build.run(corpus_out=tempfile.mkdtemp())
        clouds = report.result(Stage.CLOUDS)
        self.assertIsNotNone(clouds)
        assert clouds is not None
        self.assertTrue(clouds.ok, clouds.reason)
        self.assertTrue(report.succeeded)

    def test_every_key_a_verb_READS_is_a_key_its_callee_WRITES(self) -> None:
        """⭐ THE WHOLE CLASS, NOT THE ONE INSTANCE. Three verbs read a summary dict by string key,
        and each string is an unchecked guess about another function's return value. `label()`
        happened to be right and `clouds()` happened to be wrong, and neither fact was visible: a
        dict of `str` to `Any` carries no type a checker can compare, and a missing key returns the
        fallback rather than raising.

        ⚠ THE FIRST VERSION OF THIS TEST DID NOT WORK. It compared the callee's keys against a set
        written out by hand here, so it asserted that `scenes_written` exists, which was never in
        doubt. Restoring the bug left it GREEN. It now parses what `api.py` actually reads, which is
        the only side of the comparison that can be wrong.
        """
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]

        def _function(module: str, name: str) -> ast.FunctionDef:
            tree = ast.parse((root / module).read_text(encoding="utf-8"))
            return next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and n.name == name)

        def keys_written(module: str, name: str) -> set[str]:
            return {k.value for node in ast.walk(_function(module, name))
                    if isinstance(node, ast.Dict)
                    for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}

        def keys_read(verb: str) -> set[str]:
            """Every `summary.get("...")` and `summary["..."]` inside one verb of DatasetBuild."""
            found: set[str] = set()
            for node in ast.walk(_function("datagen/api.py", verb)):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "get"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "summary" and node.args
                        and isinstance(node.args[0], ast.Constant)):
                    found.add(node.args[0].value)
                if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                        and node.value.id == "summary" and isinstance(node.slice, ast.Constant)):
                    found.add(node.slice.value)
            return found

        callees = {"clouds": ("datagen/corpus/clouds.py", "build_cloud_corpus"),
                   "label": ("datagen/grasps/labels.py", "label_dataset")}
        for verb, (module, function) in callees.items():
            read = keys_read(verb)
            self.assertTrue(read, f"found no summary key read in DatasetBuild.{verb}; "
                                  f"if the verb stopped reading one, delete this entry")
            missing = read - keys_written(module, function)
            self.assertEqual(set(), missing,
                             f"DatasetBuild.{verb} reads {sorted(missing)}, "
                             f"which {function} never writes")

    def test_no_verb_HIDES_a_parameter_its_callee_accepts(self) -> None:
        """⛔⛔ THE QUIETER HALF OF THE WRAPPER PROBLEM, AND IT COST FIVE ARGUMENTS.

        A wrapper that declares a competing default is loud once it drifts. A wrapper that simply
        omits a parameter is never loud at all: nothing raises, the caller gets the callee's default,
        and the artifact is not the one they meant. `clouds()` forwarded four of `build_cloud_corpus`'s
        nine, so the API could build exactly ONE shape of corpus -- ground-truth masks, the default
        voxel size, environment points on. `--masks pred`, the corpus a real camera would produce,
        had a CLI route and no code route.

        ⚠ THE COMPARISON IS AGAINST THE CALLEE'S SIGNATURE, not against a list written here. A list
        would have to be updated by whoever adds the next parameter, which is the same failure one
        level up.
        """
        import inspect

        from datagen.build import build_dataset                              # noqa: PLC0415
        from datagen.corpus.clouds import build_cloud_corpus                 # noqa: PLC0415
        from datagen.grasps.labels import label_dataset                      # noqa: PLC0415

        #: The first positional of each callee is the dataset root, which the noun already owns.
        owned = {"root", "config", "out_dir", "name", "out_root", "self", "density", "report"}
        for verb, callee in (("render", build_dataset), ("label", label_dataset),
                             ("clouds", build_cloud_corpus)):
            theirs = {n for n, p in inspect.signature(callee).parameters.items()
                      if p.kind is not inspect.Parameter.VAR_KEYWORD} - owned
            mine = set(inspect.signature(getattr(DatasetBuild, verb)).parameters) - owned
            self.assertEqual(set(), theirs - mine,
                             f"DatasetBuild.{verb} cannot reach {sorted(theirs - mine)}, which "
                             f"{callee.__name__} accepts; a caller gets the default and no warning")

    def test_a_listener_sees_every_finished_stage(self) -> None:
        build = self._build()
        seen: list[str] = []
        build.attach_stage_listener(lambda r: seen.append(str(r.stage)))
        with mock.patch("datagen.build.build_dataset", return_value={"by_status": {"ok": 1}}), \
                mock.patch("datagen.grasps.labels.label_dataset", return_value={"jaw": 1}):
            build.run()
        self.assertEqual(["render", "label"], seen)


class TheReportIsReadableAndTypedTests(unittest.TestCase):

    @staticmethod
    def _report(*stages: StageResult) -> DatasetReport:
        return DatasetReport(root=Path("some/dataset"), stages=stages)

    def test_a_stage_result_can_be_looked_up(self) -> None:
        report = self._report(StageResult(stage=Stage.RENDER, ok=True, summary={"ok": 4}))
        self.assertIsNotNone(report.result(Stage.RENDER))
        self.assertIsNone(report.result(Stage.LABEL), "a stage that never ran is None, not False")

    def test_the_rendered_text_is_ascii(self) -> None:
        """⚠ A Windows console under cp1252 cannot print anything else, and a non-ASCII character
        in a printed report line has broken this repository's CLI before."""
        self._report(StageResult(stage=Stage.RENDER, ok=False, reason="nothing")).render()\
            .encode("ascii")

    def test_describe_and_render_do_not_repeat_each_other(self) -> None:
        """The command prints both; a configuration block appearing twice makes a reader check
        whether the two differ, which is work the report should have done."""
        build = DatasetBuild.from_config(DatagenConfig(), name="x")
        described = set(build.describe().split(chr(10)))
        rendered = set(self._report(
            StageResult(stage=Stage.LABEL, ok=True, summary={"jaw": 1})).render().split(chr(10)))
        self.assertEqual(set(), described & rendered)

    def test_describe_names_the_declared_cameras(self) -> None:
        """`describe()` runs BEFORE anything costs, so a cell that is not the one the caller meant
        is visible in the first second rather than the sixth hour."""
        build = DatasetBuild.from_file(
            None, name="x", overrides={"camera_rig": {"cameras": (
                CameraSpec(name="left", position_mm=(0.0, -400.0, 700.0)),
                CameraSpec(name="eih", mount="wrist"))}})
        text = build.describe()
        self.assertIn("left(fixed)", text)
        self.assertIn("eih(wrist)", text)


class TheDatasetRootIsComputedOnceTests(unittest.TestCase):
    """The CLI computed it in three handlers, each its own copy."""

    def test_out_root_wins_over_the_config(self) -> None:
        build = DatasetBuild.from_config(DatagenConfig(), name="run_01", out_root="D:/elsewhere")
        self.assertEqual(Path("D:/elsewhere/run_01"), build.root)

    def test_without_one_the_config_decides(self) -> None:
        config = DatagenConfig()
        build = DatasetBuild.from_config(config, name="run_01")
        self.assertEqual(Path(config.output.root) / "run_01", build.root)


if __name__ == "__main__":
    unittest.main()
