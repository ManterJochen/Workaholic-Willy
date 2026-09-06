"""WS3.6: the deep calculator picks its device the way every other model wrapper picks one.

⛔ **IT HAD ITS OWN RESOLVER**, one line repeated in both loaders:

    device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")

Three things followed, and every one of them made a cell quieter than it should be.

1. `WILLY_DEVICE` was ignored **on this path only**. A box pinned to one backend ran its detector and
   its segmenter there, and its grasp generator wherever that line decided.
2. MPS was never a candidate, so on a Mac the generator sat on the CPU while every other model used
   the GPU.
3. Landing on the CPU said nothing. `utility.device.get_device` warns there on purpose: a silent
   order-of-magnitude slowdown is what a warning is for, and this calculator sits in the pick loop's
   inner path.

⚠ ONE BEHAVIOUR CHANGE, DELIBERATELY. A config naming a device the machine does not have now raises
instead of quietly running on the CPU, which is what filed a CPU run's latency under a GPU run's
name. It can only fire on a config that names a device that is not there.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from src.robot.grasping.deep.calculator import _resolve_device

_DEVICE = "backend.src.utility.device"


class ItUsesTheSharedHelperTests(unittest.TestCase):

    def test_WILLY_DEVICE_is_honoured(self) -> None:
        """⛔ THE DIVERGENCE ITSELF. The env var reached every model wrapper except this one."""
        with mock.patch.dict("os.environ", {"WILLY_DEVICE": "cpu"}):
            self.assertEqual(str(_resolve_device(None)), "cpu")

    def test_the_config_still_WINS_over_the_environment(self) -> None:
        """An explicit `deep_generator.device` is the operator saying it for this cell."""
        with mock.patch.dict("os.environ", {"WILLY_DEVICE": "cuda"}):
            self.assertEqual(str(_resolve_device("cpu")), "cpu")

    def test_a_named_device_that_is_ABSENT_raises(self) -> None:
        """⚠ THE ONE BEHAVIOUR CHANGE. `device: cuda` on a box with no CUDA used to fall through to
        the CPU and keep picking, which reports a CPU run's latency under a GPU run's name."""
        with mock.patch(f"{_DEVICE}.torch.cuda.is_available", return_value=False), \
             self.assertRaises(RuntimeError) as caught:
            _resolve_device("cuda")

        self.assertIn("CUDA is not available", str(caught.exception))

    def test_MPS_is_reachable_at_all(self) -> None:
        """It was not: the old line offered cuda or cpu and nothing else, so a Mac ran the generator
        on the CPU while its detector and segmenter used the GPU."""
        with mock.patch(f"{_DEVICE}.torch.cuda.is_available", return_value=False), \
             mock.patch(f"{_DEVICE}.torch.backends.mps.is_available", return_value=True):
            self.assertEqual(str(_resolve_device(None)), "mps")

    def test_landing_on_the_CPU_is_WARNED_about(self) -> None:
        """A silent order-of-magnitude slowdown in the pick loop's inner path. The shared helper
        already warns; the point is that the deep path now goes through it.

        ⚠ THE LOGGER BY NAME. `utility_logger` sets `propagate = False`, so a root-level
        `assertLogs` captures nothing here and would pass whether the warning fired or not.
        """
        with mock.patch(f"{_DEVICE}.torch.cuda.is_available", return_value=False), \
             mock.patch(f"{_DEVICE}.torch.backends.mps.is_available", return_value=False), \
             self.assertLogs("UtilityDevice", level="WARNING") as captured:
            self.assertEqual(str(_resolve_device(None)), "cpu")

        self.assertTrue(any("slow" in line for line in captured.output), captured.output)

    def test_an_INDEXED_device_still_passes_through(self) -> None:
        """⚠ `get_device` accepts only auto/cuda/cpu/mps, so routing everything through it would
        make `device: "cuda:1"` raise on a multi-GPU box. An ordinal names one device explicitly and
        torch fails loudly on its own if it is not there; removing that is no part of this item."""
        self.assertEqual(str(_resolve_device("cuda:1")), "cuda:1")

    def test_the_deep_path_and_the_perception_path_AGREE(self) -> None:
        """The control that makes the four above mean something: same input, same answer, because it
        is now literally the same function."""
        from src.utility.device import get_device

        with mock.patch.dict("os.environ", {"WILLY_DEVICE": "cpu"}):
            self.assertEqual(str(_resolve_device(None)), str(get_device(None)))


class TheOldLineIsGoneTests(unittest.TestCase):

    def test_neither_loader_rolls_its_own_resolution(self) -> None:
        """Read off the source, because the failure mode is that ONE of the two loaders keeps the old
        line and a binned artifact then lands somewhere a set artifact would not.

        ⚠ THE STRUCTURE, NOT THE WORDS. `_resolve_device`'s own docstring QUOTES the line it
        replaced, so a test searching for that text passes on the defect and fails on the repair.
        That exact shape cost a test earlier the same day, on `noqa: ARG002`.
        """
        import ast

        tree = ast.parse(Path(
            "src/robot/grasping/deep/calculator.py").read_text(encoding="utf-8"))
        holder = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.ClassDef) and node.name == "DeepGraspCalculator")

        # ⛔ ONE LOADER SINCE 2026-09-04. There used to be two, one per family, and the failure this
        # guarded was that ONE of them kept the old line so a binned artifact landed on a device a
        # set artifact would not. The binned family is gone, so `_ensure_model` is a lazy guard that
        # DELEGATES and holds no device decision of its own. Asserting it still calls the resolver
        # would force a resolution back into a method that must not make one.
        loader = next(node for node in holder.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_load_set_generator")
        calls = [n.func.id for n in ast.walk(loader)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertIn("_resolve_device", calls, "the loader resolves its device some other way")

        # No method in the class may decide a device with a conditional expression, which is the
        # shape the deleted line had. Checked over the WHOLE class rather than a named pair, so a
        # loader added later cannot reintroduce it in a method this test does not know about.
        for method in [n for n in holder.body if isinstance(n, ast.FunctionDef)]:
            with self.subTest(method.name):
                conditionals = [n for n in ast.walk(method)
                                if isinstance(n, ast.IfExp)
                                and "device" in ast.unparse(n).lower()]
                self.assertFalse(
                    conditionals,
                    f"{method.name} decides a device with a conditional expression: "
                    + "; ".join(ast.unparse(n) for n in conditionals))


if __name__ == "__main__":
    unittest.main()
