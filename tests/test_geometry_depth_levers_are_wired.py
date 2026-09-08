"""`robot.grasping.geometry`'s depth levers reach the calculator, and change what it does.

Five keys decide where inside an object a grasp is anchored and how much of the depth under a mask is
believed at all: ``grasp_depth_reference``, ``grasp_top_penetration_mm``, ``grasp_top_quantile``,
``depth_band_mm`` and ``depth_band_near_pct``.

⛔⛔ THEY EXISTED ON THE CALCULATOR AND NO CONFIG COULD REACH THEM. The only caller that ever set one
was a simulator runner's argparse. A physical cell had no way to say where its jaw should close, and
that was invisible because the question could not arise: the perception producer overwrote the depth
inside every mask with one scalar, so every percentile of the masked depth was that same scalar and
all five keys were arithmetically incapable of doing anything. The producer publishes what it
measured now, so these are live for the first time.

⚠ TWO OBLIGATIONS, AND THE SECOND IS THE ONE THAT ROTS. The defaults must forward NOTHING, so a cell
that leaves the block alone builds the calculator with the arguments it built before. And a key that
is set must reach the runtime and change it: "flag on, runtime unchanged" is the failure this
repository built a wiring guard for, and a selector is as capable of being inert as a switch.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.config.schema.robot.grasping_schema import GraspingGeometryStageConfig
from src.robot.grasping.calculator_factory import _DEPTH_LEVERS, _depth_kwargs
from src.robot.grasping.generation.calculator import GraspCalculator


def _cfg(**overrides: object) -> SimpleNamespace:
    geometry = GraspingGeometryStageConfig(**overrides)  # type: ignore[arg-type]
    return SimpleNamespace(grasping=SimpleNamespace(geometry=geometry))


class TheDefaultsForwardNothingTests(unittest.TestCase):
    def test_an_untouched_block_adds_no_argument_at_all(self) -> None:
        """Not "adds the default value": adds no key. The calculator is constructed exactly as it
        was, so nothing that was measured against it moved."""
        self.assertEqual({}, _depth_kwargs(_cfg(), {}))

    def test_the_inert_values_ARE_the_schema_defaults(self) -> None:
        """The two lists live in different files and would drift apart silently. If a schema default
        ever stops matching the value the factory treats as inert, a cell would start forwarding a
        key that asks for nothing, and the deep calculator would refuse to build over it."""
        schema_defaults = {
            name: GraspingGeometryStageConfig.model_fields[name].get_default()
            for name in _DEPTH_LEVERS
        }
        self.assertEqual(_DEPTH_LEVERS, schema_defaults)

    def test_a_construction_site_that_passed_one_ITSELF_keeps_it(self) -> None:
        """An explicit argument at a call site is a runner's deliberate choice; config is the default
        for cells that do not make one."""
        supplied = {"grasp_depth_reference": "centre"}

        self.assertEqual({}, _depth_kwargs(_cfg(grasp_depth_reference="top"), supplied))

    def test_a_config_without_the_block_is_not_an_error(self) -> None:
        self.assertEqual({}, _depth_kwargs(SimpleNamespace(grasping=SimpleNamespace(geometry=None)), {}))


class TheKeysForwardWhenSetTests(unittest.TestCase):
    def test_every_one_of_the_five_reaches_the_calculator(self) -> None:
        """Enumerated from the schema rather than listed by hand, so a sixth key added to the block
        and forgotten here fails instead of passing."""
        non_default = {
            "grasp_depth_reference": "top",
            "grasp_top_penetration_mm": 4.0,
            "grasp_top_quantile": 25.0,
            "depth_band_mm": 30.0,
            "depth_band_near_pct": 20.0,
        }
        self.assertEqual(set(_DEPTH_LEVERS), set(non_default), "a lever is missing from this test")

        forwarded = _depth_kwargs(_cfg(**non_default), {})

        self.assertEqual(non_default, forwarded)


class TheyChangeWhatTheCalculatorDoesTests(unittest.TestCase):
    """Forwarding is not wiring. These assert the effect on a real calculator.

    ⚠ EVERY SCENE HERE HAS RELIEF UNDER THE MASK, which is the whole point. Run the same assertions
    against the flat sheet the producer used to publish and they cannot fail, because a constant has
    only one quantile.
    """

    _K = np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 32.0], [0.0, 0.0, 1.0]])

    def _scene(self) -> "tuple[SimpleNamespace, np.ndarray]":
        """A 20 mm-deep face under the mask, and a rim of bench showing through its edge."""
        depth = np.full((64, 64), 900.0)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        depth[20:40, 20:40] = np.linspace(700.0, 720.0, 20)[:, None]
        depth[38:40, 20:40] = 900.0  # mask edge spilling onto the bench, 180 mm behind the object
        return SimpleNamespace(mask=mask), depth

    def _anchor(self, **kwargs: object) -> float:
        """Where the winning candidate sits along the camera's depth axis, in mm.

        The candidate's own position rather than a telemetry key: this is the number that reaches the
        arm, so a lever that moves it has moved the grasp and not a report about the grasp.
        """
        seg, depth = self._scene()
        calc = GraspCalculator(camera_matrix=self._K, **kwargs)  # type: ignore[arg-type]
        result = calc.compute_result(seg, depth, pixel_to_mm=None)
        return float(result.candidates[0].position[2])

    def test_the_reference_moves_the_anchor_towards_the_near_face(self) -> None:
        centre = self._anchor(grasp_depth_reference="centre")
        top = self._anchor(grasp_depth_reference="top", grasp_top_quantile=10.0)

        self.assertLess(top, centre, "'top' must anchor NEARER the camera than 'centre'")

    def test_the_penetration_drives_the_anchor_deeper(self) -> None:
        without = self._anchor(grasp_depth_reference="top")
        with_descend = self._anchor(grasp_depth_reference="top", grasp_top_penetration_mm=6.0)

        self.assertAlmostEqual(with_descend - without, 6.0, places=3)

    def test_the_quantile_decides_how_near_the_near_face_is(self) -> None:
        q10 = self._anchor(grasp_depth_reference="top", grasp_top_quantile=10.0)
        q40 = self._anchor(grasp_depth_reference="top", grasp_top_quantile=40.0)

        self.assertLess(q10, q40)

    def test_the_band_discards_the_bench_showing_through_the_mask_edge(self) -> None:
        """The lever's reason for existing. Those bench pixels are 180 mm behind the object and they
        drag the median that 'centre' anchors on."""
        unbanded = self._anchor(grasp_depth_reference="centre")
        banded = self._anchor(grasp_depth_reference="centre", depth_band_mm=40.0)

        self.assertLess(banded, unbanded)
        self.assertLess(abs(banded - 710.0), 12.0, "the band should land the anchor on the object")

    def test_none_of_this_can_happen_on_the_sheet_the_producer_used_to_publish(self) -> None:
        """⛔ THE CONTROL, and the reason all five keys were dead rather than merely unused. Replace
        the relief with the single scalar the producer wrote and every lever above answers the same
        number, because every percentile of a constant is that constant."""
        seg, _ = self._scene()
        flat = np.full((64, 64), 900.0)
        flat[20:40, 20:40] = 703.0  # min(surface) + 3 mm, which is what the producer wrote

        def anchor(**kwargs: object) -> float:
            calc = GraspCalculator(camera_matrix=self._K, **kwargs)  # type: ignore[arg-type]
            return float(calc.compute_result(seg, flat, pixel_to_mm=None)
                         .candidates[0].position[2])

        self.assertEqual(anchor(grasp_depth_reference="centre"),
                         anchor(grasp_depth_reference="top", grasp_top_quantile=10.0))
        self.assertEqual(anchor(grasp_depth_reference="centre"),
                         anchor(grasp_depth_reference="centre", depth_band_mm=40.0))


if __name__ == "__main__":
    unittest.main()
