"""Stage 3.6: the scorer fitted on the generator's OWN proposals, with physics as the target.

⭐ THE MEASUREMENTS THIS FILE DEFENDS were taken before the code was written. Training on the physics
referee rather than on our own analytic verdict is worth +11.97 AUROC pooled on identical rows and
folds, and +12.2 on `finger_collision` alone, which is the stratum where the cheap argument said a
learner should not need physics at all.

Everything below guards a way the fit could look trained and mean nothing: a join that silently drops
rows, a refusal counted as a failed grasp, a degenerate target, a split that puts one point cloud on
both sides, and an AUC published without the floor it has to beat.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.robot.grasping.deep.eval.proposal_scorer_training import (
    ScorerData, fit_scorer, load_pairs, summarise, unjudged)


def _proposal(scene: str, rank: int, **over) -> dict:
    row = {"scene_id": scene, "instance_id": 1, "position_mm": [10.0, 20.0, 60.0],
           "approach": [0.0, 0.0, -1.0], "closing_axis": [1.0, 0.0, 0.0],
           "width_mm": 60.0, "confidence": 0.5, "seed_index": 0, "slot": 0, "rank": rank}
    row.update(over)
    return row


def _verdict(scene: str, row_index: int, held: bool, note: str = "drift 0.01 mm") -> dict:
    return {"scene_id": scene, "instance_id": 1, "source": "generator",
            "position_mm": [10.0, 20.0, 60.0], "approach": [0.0, 0.0, -1.0],
            "closing_axis": [1.0, 0.0, 0.0], "width_mm": 60.0, "row_index": row_index,
            "held": held, "rise_mm": 0.0, "note": note}


def _write(directory: Path, proposals, verdicts) -> tuple[Path, Path]:
    p = directory / "proposals.jsonl"
    v = directory / "physics.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in proposals), encoding="utf-8")
    v.write_text("".join(json.dumps(r) + "\n" for r in verdicts), encoding="utf-8")
    return p, v


class JoinTests(unittest.TestCase):

    def test_the_join_is_by_scene_and_rank_and_NOT_by_pose(self) -> None:
        """⛔ THE TRAP THIS REPOSITORY ALREADY MEASURED. A pose join is ambiguous for 53.6 % of this
        corpus, structurally, because a label and the row describing it carry identical poses. Here
        two proposals in one scene share a pose exactly and hold differently, so a pose join would
        have to guess and a (scene, rank) join cannot."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            p, v = _write(path,
                          [_proposal("s1", 0, width_mm=40.0), _proposal("s1", 1, width_mm=90.0)],
                          [_verdict("s1", 0, True), _verdict("s1", 1, False)])
            data = load_pairs(p, v)
            self.assertEqual(len(data.held), 2)
            order = np.argsort(data.features[:, data.names.index("width_mm")])
            self.assertEqual([bool(x) for x in data.held[order]], [True, False])

    def test_a_REFUSED_trial_is_dropped_and_never_scored_as_a_failure(self) -> None:
        """⛔ A refusal is a trial the harness would not stand behind, not a grasp that failed.
        Scoring it as a negative would teach the scorer that the harness's own faults are properties
        of the grasp, and this arc has already watched one burst trial poison 26 of 36 that followed
        it."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            p, v = _write(path, [_proposal("s1", 0), _proposal("s1", 1)],
                          [_verdict("s1", 0, True),
                           _verdict("s1", 1, False, note="refused: gripper shoved 40.0 mm")])
            data = load_pairs(p, v)
            self.assertEqual(len(data.held), 1)
            self.assertTrue(bool(data.held[0]))
            self.assertEqual(unjudged(p, v), 1)

    def test_the_refusal_test_matches_the_WRITER_and_is_not_a_substring_search(self) -> None:
        """⚠ The writer marks a refusal by STARTING the note with the word. A substring test would
        also drop an ordinary note that merely mentions one, and two holes in this repository came
        from exactly that shape of loose match."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            p, v = _write(path, [_proposal("s1", 0)],
                          [_verdict("s1", 0, True, note="held; a neighbour trial was refused")])
            self.assertEqual(len(load_pairs(p, v).held), 1)

    def test_the_controls_row_is_not_a_trial(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            p, v = _write(path, [_proposal("s1", 0)],
                          [{"controls": {"engine": "mujoco", "jaw_is_solid": True}},
                           _verdict("s1", 0, True)])
            self.assertEqual(len(load_pairs(p, v).held), 1)

    def test_an_unjudged_proposal_is_dropped_and_COUNTED(self) -> None:
        """⚠ Never given a default. A run that quietly halved its own dataset would still print a
        clean AUC over whatever survived."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            p, v = _write(path, [_proposal("s1", r) for r in range(4)],
                          [_verdict("s1", 0, True), _verdict("s1", 2, False)])
            self.assertEqual(len(load_pairs(p, v).held), 2)
            self.assertEqual(unjudged(p, v), 2)


def _dataset(scenes: int = 6, per_scene: int = 6, seed: int = 0) -> ScorerData:
    rng = np.random.default_rng(seed)
    features, held, groups = [], [], []
    for scene in range(scenes):
        for _ in range(per_scene):
            width = float(rng.uniform(20.0, 100.0))
            features.append([width, float(rng.uniform(0.0, 1.0)), -1.0, 0.0, 0.0, 60.0])
            held.append(width < 60.0)
            groups.append(f"s{scene}")
    return ScorerData(np.asarray(features), np.asarray(held, dtype=bool),
                      np.asarray(groups, dtype=object),
                      ("width_mm", "confidence", "approach_z", "axis_z", "approach_tilt",
                       "height_mm"))


class FitTests(unittest.TestCase):

    def test_a_degenerate_target_is_refused_BY_NAME(self) -> None:
        """⛔ Everything held, or nothing did: there is no ranking to learn. The fit would return a
        constant, its AUC would be undefined, and the report would be indistinguishable from a
        trained one. That last part is why this raises instead of warning."""
        data = _dataset()
        for value in (True, False):
            with self.subTest(value):
                flat = ScorerData(data.features, np.full(len(data.held), value),
                                  data.groups, data.names)
                with self.assertRaises(ValueError) as caught:
                    fit_scorer(flat)
                self.assertIn("degenerate", str(caught.exception))

    def test_too_few_rows_are_refused(self) -> None:
        data = _dataset(scenes=2, per_scene=2)
        with self.assertRaises(ValueError):
            fit_scorer(data, folds=5)

    def test_the_split_receives_the_SCENE_as_its_group(self) -> None:
        """⛔ SCENE-DISJOINT OR IT MEASURES THE WRONG THING. Two proposals in one scene share a point
        cloud, so a random split reports memorising a scene rather than judging a grasp. This
        repository has paid for the sibling mistake once: a held-out set that was a PREFIX of a
        group-ordered array turned out to be a handful of asset groups.

        Asserted by watching what the splitter is actually handed, because swapping `GroupKFold` for
        a plain `KFold` is a one-word edit that nothing else here would notice."""
        import sklearn.model_selection as ms

        seen: list = []
        original = ms.GroupKFold

        class Spy(original):  # type: ignore[misc, valid-type]
            def split(self, X, y=None, groups=None):  # noqa: N803
                seen.append(groups)
                return super().split(X, y, groups)

        ms.GroupKFold = Spy
        try:
            fit_scorer(_dataset())
        finally:
            ms.GroupKFold = original
        self.assertEqual(len(seen), 1)
        self.assertIsNotNone(seen[0])
        self.assertEqual(sorted(set(seen[0].tolist())), [f"s{i}" for i in range(6)])

    def test_more_folds_than_scenes_does_not_crash(self) -> None:
        """The split cannot have more parts than there are scenes, and the honest response is fewer
        folds rather than a fold that borrows a scene from its neighbour."""
        report = fit_scorer(_dataset(scenes=3, per_scene=8), folds=10)
        self.assertEqual(report["scenes"], 3)

    def test_the_FLOORS_are_reported_beside_the_score(self) -> None:
        """⭐ The head's own confidence is one of the six features, so a scorer that learned nothing
        beyond trusting the head still beats 0.5. An AUC without its floor has been published in this
        repository once, and the fix was to make the floor part of the report rather than part of the
        discipline of whoever reads it."""
        report = fit_scorer(_dataset())
        for key in ("auc", "auc_head_confidence_alone", "auc_width_alone", "hold_rate", "scenes"):
            self.assertIn(key, report)
        text = "\n".join(summarise(report))
        self.assertIn("head confidence alone", text)
        self.assertIn("width alone", text)

    def test_the_report_calls_the_hold_rate_the_REFEREE_success_rate(self) -> None:
        """⚠ Two different hold rates live in this repository. Every one published before stage 3.6
        came from grasps drawn out of the LABELS; this one comes from grasps the MODEL chose. They
        answer different questions and must not share a name in a report a person reads."""
        self.assertIn("referee success rate", "\n".join(summarise(fit_scorer(_dataset()))))

    def test_a_learnable_target_is_actually_learned(self) -> None:
        """The control. Width decides the outcome in `_dataset`, so a fit that cannot beat chance
        here is broken rather than facing a hard problem."""
        self.assertGreater(fit_scorer(_dataset(scenes=8, per_scene=10))["auc"], 0.8)


class CliTests(unittest.TestCase):

    def test_the_subcommand_runs_the_whole_stage(self) -> None:
        from src.robot.grasping.deep.__main__ import main

        rng = np.random.default_rng(3)
        proposals, verdicts = [], []
        for scene in range(8):
            for rank in range(10):
                width = float(rng.uniform(20.0, 100.0))
                proposals.append(_proposal(f"s{scene}", rank, width_mm=width,
                                           confidence=float(rng.uniform(0, 1))))
                verdicts.append(_verdict(f"s{scene}", rank, width < 60.0))
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            p, v = _write(path, proposals, verdicts)
            out = path / "report.json"
            self.assertEqual(main(["score-proposals", "--proposals", str(p),
                                   "--verdicts", str(v), "--out", str(out)]), 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["proposals"], 80)
            self.assertEqual(report["scenes"], 8)

    def test_a_degenerate_file_exits_nonzero_instead_of_reporting(self) -> None:
        from src.robot.grasping.deep.__main__ import main

        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            p, v = _write(path, [_proposal(f"s{i // 4}", i) for i in range(20)],
                          [_verdict(f"s{i // 4}", i, True) for i in range(20)])
            self.assertNotEqual(main(["score-proposals", "--proposals", str(p),
                                      "--verdicts", str(v)]), 0)


if __name__ == "__main__":
    unittest.main()
