"""The seam is real: the shipping calculator satisfies it, and the runtime reads nothing else.

Two directions, and both are needed. A protocol that the one existing implementation fails is
wrong about the code; a protocol the runtime has outgrown is wrong about the callers. So:

* the analytic `GraspCalculator` is checked AGAINST the protocol, and
* the protocol is checked against every attribute the live paths actually touch, discovered by
  parsing the source rather than by remembering.

The second test is the one that will earn its keep. Someone adding `self.calculator.foo` to the
pick loop is not thinking about a protocol in another package; this fails and tells them.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

from src.robot.grasping.deep import GraspCandidateGenerator

_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Where a calculator is USED at runtime. Not the calculator's own package -- a class may read its
#: own privates without that being part of any seam.
_RUNTIME_PATHS = (
    "src/robot/grasping/loop",
    "src/robot/grasping/closed_loop",
    "src/robot/execution/autonomous_grasp",
)

#: Attribute reads that are NOT required protocol members, with the reason each is exempt.
_NOT_THE_SEAM = {
    # `calculator.py` inside a string/comment or a module path, never an attribute access.
    "py",
    # Read defensively with `getattr(..., None)` and treated as "stand down" when absent, so it is
    # an OPTIONAL capability rather than a required member: a generator without intrinsics is a
    # legitimate generator, it just does not get the single-view support-plane refinement. The
    # analytic calculator DOES expose it since 2026-08-23 -- see the tests at the bottom of this
    # file and the note in protocol.py.
    "camera_matrix",
}


class TheShippingCalculatorSatisfiesTheProtocolTests(unittest.TestCase):
    def test_the_analytic_calculator_is_a_generator(self) -> None:
        """If this fails, the protocol describes a calculator that does not exist."""
        from src.robot.grasping.generation.calculator import GraspCalculator

        calculator = GraspCalculator(max_grip_width_mm=85.0, min_grip_width_mm=0.0)
        self.assertIsInstance(calculator, GraspCandidateGenerator)

    def test_a_stand_in_with_the_three_members_is_accepted(self) -> None:
        """The point of the seam: something that is not GraspCalculator can be one."""

        class _StandIn:
            render_debug_images = False
            last_debug_image_png = None

            def compute_result(self, *args: object, **kwargs: object) -> object:
                return None

        self.assertIsInstance(_StandIn(), GraspCandidateGenerator)

    def test_something_missing_compute_result_is_not(self) -> None:
        """A runtime_checkable Protocol checks membership, so the negative case has to be shown."""

        class _NotAGenerator:
            render_debug_images = False
            last_debug_image_png = None

        self.assertNotIsInstance(_NotAGenerator(), GraspCandidateGenerator)


class TheProtocolCoversWhatTheRuntimeActuallyReadsTests(unittest.TestCase):
    """Parse the live paths for every `<...>calculator.<attr>` and `getattr(<...>calculator, "x")`.

    Source-parsed rather than hand-listed: the failure mode this guards against is somebody adding
    an access, and a hand-written list is exactly what such a person does not update.
    """

    def _accessed_attributes(self) -> dict[str, str]:
        found: dict[str, str] = {}
        for rel in _RUNTIME_PATHS:
            for path in sorted((_ROOT / rel).rglob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    where = f"{rel}/{path.name}:{getattr(node, 'lineno', 0)}"
                    # `x.calculator.attr`
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
                        if node.value.attr.endswith("calculator"):
                            found.setdefault(node.attr, where)
                    # `getattr(<expr ending in .calculator | name 'calculator'>, "attr", ...)`
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "getattr"
                        and len(node.args) >= 2
                        and isinstance(node.args[1], ast.Constant)
                        and isinstance(node.args[1].value, str)
                    ):
                        target = node.args[0]
                        name = (
                            target.attr if isinstance(target, ast.Attribute)
                            else target.id if isinstance(target, ast.Name)
                            else ""
                        )
                        if name.endswith("calculator"):
                            found.setdefault(node.args[1].value, where)
        return {k: v for k, v in found.items() if k not in _NOT_THE_SEAM}

    def test_every_attribute_the_runtime_reads_is_in_the_protocol(self) -> None:
        import typing

        # `typing._get_protocol_attrs` is the only way to ask a Protocol what it declares; the
        # public API exposes no such accessor. Private, but stable across 3.9-3.13 and checked
        # by the companion test below, which fails loudly if it ever returns nothing.
        # ⭑ THE UNION OF BOTH PROTOCOLS. `preload` is OPTIONAL: the analytic calculator has none
        # and is still a generator, so it lives in `PreloadableGenerator` rather than as a fourth
        # required member. What the guard cares about is that every name the runtime reads is
        # DECLARED somewhere in the seam, not that every generator implements it.
        from src.robot.grasping.deep.protocol import (  # noqa: PLC0415
            PreloadableGenerator,
        )

        declared = (set(typing._get_protocol_attrs(GraspCandidateGenerator))
                    | set(typing._get_protocol_attrs(PreloadableGenerator)))
        accessed = self._accessed_attributes()
        missing = {name: where for name, where in accessed.items() if name not in declared}
        self.assertEqual(
            missing, {},
            "the runtime reads calculator attributes the protocol does not declare -- either add "
            "them to GraspCandidateGenerator, or to PreloadableGenerator if optional, or stop "
            "reading them off the calculator",
        )

    def test_the_measurement_found_the_known_members(self) -> None:
        """A parser that silently found nothing would make the test above vacuously green."""
        accessed = self._accessed_attributes()
        for expected in ("compute_result", "render_debug_images", "last_debug_image_png"):
            with self.subTest(attribute=expected):
                self.assertIn(expected, accessed)


class TheCameraMatrixIsTheCalibratedOneOrNothingTests(unittest.TestCase):
    """`camera_matrix` was exposed on 2026-08-23 to bring a dead branch to life. Safely.

    THE HISTORY, because it explains the shape of the assertions. `loop/pick_loop.py:836` reads
    `calculator.camera_matrix` to back-project this view's target cloud for the support-plane
    refinement under `support.refine_from_target` (which defaults to True). The class kept its K in
    `self._geom._K` and never surfaced it, so that read always returned `None` and the single-view
    half of the refinement never ran on any path -- only the fused-cloud half did.

    THE SAFETY ARGUMENT is that the property reads the calibrated matrix DIRECTLY rather than going
    through `_SharedGeometryUtil.intrinsics()`, which synthesizes `focal = width / 2` when no
    calibration was supplied. That fallback's own comment: "back-projects to METRICALLY WRONG 3D ->
    fine for silhouette ranking, never trust a BASE-frame metric grasp on it". A support plane is a
    BASE-frame metric quantity. So an uncalibrated cell must keep getting `None` and must keep
    standing the branch down -- and that, not the positive case, is the assertion that matters.
    """

    def _calculator(self, *, calibrated: bool):
        import numpy as np

        from src.robot.grasping.generation.calculator import GraspCalculator

        k = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        return GraspCalculator(
            max_grip_width_mm=85.0,
            min_grip_width_mm=0.0,
            camera_matrix=k if calibrated else None,
        )

    def test_a_calibrated_calculator_exposes_its_k(self) -> None:
        import numpy as np

        matrix = self._calculator(calibrated=True).camera_matrix
        self.assertIsNotNone(matrix)
        assert matrix is not None
        np.testing.assert_allclose(
            matrix, [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
        )

    def test_an_uncalibrated_calculator_reports_none_not_a_guess(self) -> None:
        """THE load-bearing one: no calibration must not become a plausible-looking matrix.

        The generator will happily invent intrinsics for silhouette work. If that invention leaked
        out here, an uncalibrated cell would start estimating a metric support plane from a
        metrically wrong back-projection -- confidently, and with nothing in the output to say so.
        """
        self.assertIsNone(self._calculator(calibrated=False).camera_matrix)

    def test_it_is_not_the_synthesized_fallback(self) -> None:
        """Pin the distinction itself, so a later 'simplification' to `intrinsics()` fails here."""
        calculator = self._calculator(calibrated=False)
        synthesized = calculator._geom.intrinsics((480, 640))
        self.assertEqual(synthesized.fx, 320.0, "the fallback stopped being width/2; re-read this")
        self.assertIsNone(
            calculator.camera_matrix,
            "camera_matrix now returns the SYNTHESIZED intrinsics -- that back-projects to "
            "metrically wrong 3-D and the support-plane branch would consume it as a real "
            "measurement",
        )

    def test_it_is_read_only(self) -> None:
        """A settable attribute could disagree with the K the generator back-projects with."""
        calculator = self._calculator(calibrated=True)
        with self.assertRaises(AttributeError):
            calculator.camera_matrix = None  # type: ignore[misc]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
