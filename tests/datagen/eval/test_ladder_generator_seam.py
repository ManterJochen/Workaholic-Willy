"""The ladder's generator seam — grading something OTHER than `GraspCalculator`.

The eval-grasps ladder existed to grade one class, so the class was written into its loop. Grading a
second generator -- a learned one, or the same one wrapped in a reranker -- against the SAME scenes,
the SAME analytic reference and the SAME denominators is the only way a claim like "the learned
generator beats `sfe_fused`" can be made at all, and it cannot be made through a hardcoded name.

No rendering and no Isaac here: what these pin is that the injection point exists, that leaving it
alone constructs exactly what it always did, and that the seam does not quietly widen.
"""

from __future__ import annotations

import unittest
from typing import Any

from datagen.eval.ladder import CalculatorConfig


class GeneratorSeamTests(unittest.TestCase):
    def test_the_default_is_none_so_every_existing_rung_is_unchanged(self) -> None:
        config = CalculatorConfig(name="sfe_fused", build=lambda _i, _c: {})
        self.assertIsNone(config.make)

    def test_none_resolves_to_the_analytic_calculator(self) -> None:
        """The `or` in the loop is the whole compatibility story; it is worth a test of its own."""
        from src.robot.grasping.generation.calculator import GraspCalculator

        config = CalculatorConfig(name="sfe_fused", build=lambda _i, _c: {})
        self.assertIs(config.make or GraspCalculator, GraspCalculator)

    def test_a_factory_is_carried_and_receives_the_built_kwargs(self) -> None:
        seen: list[dict[str, Any]] = []

        class Stand_in:
            def __init__(self, **kwargs: Any) -> None:
                seen.append(kwargs)

        config = CalculatorConfig(
            name="learned", build=lambda _i, _c: {"gripper_width_mm": 85.0}, make=Stand_in)
        built = (config.make or object)(**config.build(None, None), extra=1)  # type: ignore[arg-type]

        self.assertIsInstance(built, Stand_in)
        self.assertEqual(seen, [{"gripper_width_mm": 85.0, "extra": 1}])

    def test_the_config_is_still_frozen(self) -> None:
        """A rung that could be mutated mid-ladder would make the run unreproducible."""
        config = CalculatorConfig(name="x", build=lambda _i, _c: {})
        with self.assertRaises(Exception):
            config.make = object          # type: ignore[misc]


class TheTwoContractsAreDifferentTests(unittest.TestCase):
    """`compute()` is what the ladder calls; `compute_result()` is what the runtime calls.

    Satisfying `deep.protocol.GraspCandidateGenerator` does NOT make a generator measurable on the
    ladder, and passing the ladder does not make one runtime-ready. Both are required. This pins the
    asymmetry so it is discovered here rather than by a green ladder and a cell that cannot call the
    thing it just validated.
    """

    def test_the_runtime_protocol_names_compute_result_not_compute(self) -> None:
        from src.robot.grasping.deep.protocol import GraspCandidateGenerator

        # `typing._get_protocol_attrs` rather than `__protocol_attrs__`: the latter is empty on this
        # interpreter, which made the first version of this test pass vacuously against `set()`.
        import typing

        members = set(typing._get_protocol_attrs(GraspCandidateGenerator))  # noqa: SLF001
        self.assertTrue(members, "an empty member set would make every assertion below vacuous")
        self.assertIn("compute_result", members)
        self.assertNotIn("compute", members)

    def test_the_ladder_calls_compute(self) -> None:
        import inspect

        import datagen.eval.ladder as evaluate

        source = inspect.getsource(evaluate)
        self.assertIn("calculator.compute(", source)


if __name__ == "__main__":
    unittest.main()
