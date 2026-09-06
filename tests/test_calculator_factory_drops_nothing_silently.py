"""A kwarg the deep branch cannot honour is a refusal or a stated loss, never silence.

⛔⛔ **THE SHAPE OF DEFECT THIS FILE EXISTS FOR HAS ALREADY COST THIS REPOSITORY A FULL LADDER RUN.**
`datagen/eval/ladder.py` records a 103-scene sweep that came back byte-identical because
`depth_source` was missing from an explicit argument list exactly like the factory's.

`build_calculator`'s two branches are asymmetric by construction: the geometric one is
`GraspCalculator(**kwargs)` verbatim, so it honours everything; the deep one maps an EXPLICIT list, so
it honours what that list names and drops the rest. That asymmetry is fine. Dropping without saying so
is not.

Two of the droppable kwargs change what a pick DOES:

  `ik_service`                  the reachability filter that DISCARDS unreachable candidates, and the
                                source of `joint_margin_deg`. Without it nothing filters and
                                `rejected_ik` stays 0, which reads as "nothing was unreachable".
  `corridor_risk_per_candidate` the G4 producer. Without it `--g4-rerank` reorders nothing and reports
                                `missing_uncertainty`.

The rest are analytic-only: the deep decoder predicts its own approach, closing axis and seeds, so an
oblique sweep or a radial-closing flag genuinely has nothing to act on. Those get a log line, because
`robot.ur3e.yaml` sets `isotropic_radial_closing: true` on a cell measured at 5/10 -> 10/10, and a
banner printing `True` under a deep calculator is describing the other path.
"""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.robot.grasping.calculator_factory import (
    _CARRIED,
    _IGNORED_BY_DEEP,
    build_calculator,
)
from src.robot.grasping.deep.set_artifact import (
    SET_ARTIFACT_KIND as ARTIFACT_KIND,
    SET_ARTIFACT_VERSION as ARTIFACT_VERSION,
)


def _config(choice: str, artifact: str = ""):
    """A stand-in for RobotConfig carrying only what the factory reads."""
    from types import SimpleNamespace

    return SimpleNamespace(grasping=SimpleNamespace(
        calculator=choice,
        deep_generator=SimpleNamespace(artifact_path=artifact, device="cpu", minimum_score=0.5),
        support=SimpleNamespace(height_mm=0.0)))


def _artifact(directory: Path) -> str:
    path = directory / "generator.pt"
    torch.save({"kind": ARTIFACT_KIND, "artifact_version": ARTIFACT_VERSION}, path)
    return str(path)


class RefusalTests(unittest.TestCase):

    def _build(self, **kwargs):
        with tempfile.TemporaryDirectory() as name:
            cfg = _config("deep", _artifact(Path(name)))
            # The construction itself is mocked out: what is under test is the GATE in front of it,
            # and building a real net here would test torch instead.
            with mock.patch("src.robot.grasping.deep.calculator.DeepGraspCalculator"):
                return build_calculator(cfg, camera_matrix=None, **kwargs)

    def test_ik_service_is_REFUSED_BY_NAME(self) -> None:
        """⛔ It discards unreachable candidates. Running without it is a different experiment."""
        with self.assertRaises(ValueError) as caught:
            self._build(ik_service=object())
        self.assertIn("ik_service", str(caught.exception))

    def test_corridor_risk_per_candidate_is_REFUSED_BY_NAME(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._build(corridor_risk_per_candidate=True)
        self.assertIn("corridor_risk_per_candidate", str(caught.exception))

    def test_the_refusal_says_WHY_and_offers_the_way_out(self) -> None:
        """A refusal a reader cannot act on becomes a flag someone deletes."""
        with self.assertRaises(ValueError) as caught:
            self._build(depth_band_mm=12.0)
        message = str(caught.exception)
        self.assertIn("calculator: geometric", message)
        self.assertIn("different experiment", message)

    def test_EVERY_unlisted_kwarg_is_refused_not_just_the_two_known_ones(self) -> None:
        """⭐ The property. A hardcoded pair would pass while the next new knob went dark."""
        for name in ("grasp_depth_reference", "cloud_outlier_filter", "some_future_knob"):
            with self.subTest(name), self.assertRaises(ValueError):
                self._build(**{name: 1})


class OffPositionTests(unittest.TestCase):
    """⛔ THE CHECK IS ON THE VALUE, NOT THE KEY, and the first version got that wrong.

    `run_eih_pick` passes `ik_service=None` and `corridor_risk_per_candidate=False` on EVERY run:
    those are the OFF positions of two optional features. A check that fired on the key would refuse
    a deep run that loses nothing at all, which is a refusal nobody can act on and the fastest way to
    have a guard deleted.
    """

    def _build(self, **kwargs):
        with tempfile.TemporaryDirectory() as name:
            cfg = _config("deep", _artifact(Path(name)))
            with mock.patch("src.robot.grasping.deep.calculator.DeepGraspCalculator"):
                return build_calculator(cfg, camera_matrix=None, **kwargs)

    def test_a_switched_OFF_kwarg_does_not_refuse(self) -> None:
        self._build(ik_service=None, corridor_risk_per_candidate=False)

    def test_a_ZERO_threshold_counts_as_off(self) -> None:
        """Every kwarg on this path is a switch, a live handle or a millimetre threshold, and a zero
        threshold means "do not apply it" in all of them."""
        self._build(grasp_top_penetration_mm=0.0, depth_band_mm=0)

    def test_an_EMPTY_collection_counts_as_off(self) -> None:
        self._build(oblique_azimuths=())

    def test_the_SAME_kwarg_switched_ON_still_refuses(self) -> None:
        """The control that makes the four tests above mean something."""
        with self.assertRaises(ValueError):
            self._build(ik_service=object())
        with self.assertRaises(ValueError):
            self._build(corridor_risk_per_candidate=True)
        with self.assertRaises(ValueError):
            self._build(depth_band_mm=12.0)

    def test_an_analytic_knob_switched_OFF_is_not_even_logged(self) -> None:
        """A warning on every run for a lever nobody switched on is noise, and noise gets filtered."""
        with tempfile.TemporaryDirectory() as name:
            cfg = _config("deep", _artifact(Path(name)))
            with mock.patch("src.robot.grasping.deep.calculator.DeepGraspCalculator"):
                with self.assertNoLogs("CalculatorFactory", level=logging.WARNING):
                    build_calculator(cfg, camera_matrix=None, isotropic_radial_closing=False)


class CarriedTests(unittest.TestCase):

    def test_the_CARRIED_kwargs_do_not_refuse(self) -> None:
        """The control. If everything refused, the tests above would pass on a factory that had
        stopped building anything at all."""
        with tempfile.TemporaryDirectory() as name:
            cfg = _config("deep", _artifact(Path(name)))
            with mock.patch("src.robot.grasping.deep.calculator.DeepGraspCalculator"):
                build_calculator(cfg, camera_matrix=None, max_candidates=12,
                                 min_grip_width_mm=5.0, max_grip_width_mm=85.0)

    def test_the_carried_set_matches_what_the_code_actually_reads(self) -> None:
        """⚠ Two sources of truth drift. The set is asserted against the constructor call itself."""
        source = Path("src/robot/grasping/calculator_factory.py").read_text(encoding="utf-8")
        body = source[source.index("DeepCalculatorConfig("):]
        for name in _CARRIED:
            with self.subTest(name):
                self.assertIn(f'kwargs.get("{name}"', body,
                              f"{name} is in _CARRIED but the deep branch never reads it")


class IgnoredTests(unittest.TestCase):

    def test_an_analytic_only_knob_is_LOGGED_rather_than_refused(self) -> None:
        """⚠ The deep decoder predicts its own closing axis, so this one genuinely has nothing to act
        on. Refusing it would block a legitimate config; dropping it in silence lets a banner lie."""
        with tempfile.TemporaryDirectory() as name:
            cfg = _config("deep", _artifact(Path(name)))
            with mock.patch("src.robot.grasping.deep.calculator.DeepGraspCalculator"), \
                 self.assertLogs("CalculatorFactory", level=logging.WARNING) as caught:
                build_calculator(cfg, camera_matrix=None, isotropic_radial_closing=True)
        self.assertIn("isotropic_radial_closing", "\n".join(caught.output))

    def test_the_two_sets_do_not_overlap(self) -> None:
        """A kwarg that is both carried and ignored would take whichever branch ran first."""
        self.assertEqual(_CARRIED & _IGNORED_BY_DEEP, frozenset())


class GeometricIsUntouchedTests(unittest.TestCase):

    def test_the_geometric_branch_still_honours_EVERYTHING(self) -> None:
        """⛔ DEFAULT-OFF BYTE-IDENTICAL. `grasping.calculator` defaults to geometric and appears in
        no shipped YAML, so every measured number in this repository came through this branch. It
        forwards `**kwargs` verbatim and this test fails if that ever becomes a filtered list."""
        seen: dict = {}

        class Spy:
            def __init__(self, **kwargs):
                seen.update(kwargs)

        with mock.patch("src.robot.grasping.generation.calculator.GraspCalculator", Spy):
            build_calculator(_config("geometric"), camera_matrix=None, ik_service="x",
                             isotropic_radial_closing=True, anything_at_all=7)
        self.assertEqual(set(seen), {"camera_matrix", "ik_service", "isotropic_radial_closing",
                                     "anything_at_all"})


if __name__ == "__main__":
    unittest.main()
