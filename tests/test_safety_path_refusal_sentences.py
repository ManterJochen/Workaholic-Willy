"""The two sentences a path is refused with have one home, so a checklist row can say exactly what the gate says.

``SafetyPreflight`` refuses every planned or sampled path when no self collision guard is wired, and when the guard
would answer with the capsule proxy instead of the exact mesh engine. The real cell checklist gains an engine row that
has to tell an operator the same thing before the cell moves. Two copies of one sentence drift, so both live in
functions the gate calls and the row reads.
"""

from __future__ import annotations

import inspect
import unittest

from src.robot.safety import preflight as safety_preflight


class ThePathRefusalsHaveOneHomeTests(unittest.TestCase):
    def test_the_sentences_say_what_is_missing(self) -> None:
        self.assertIn("safety.self_collision.enforce is false", safety_preflight.no_path_guard_refusal())
        refusal = safety_preflight.exact_mesh_path_refusal()
        self.assertIn("exact mesh engine", refusal)
        self.assertIn("Coal", refusal)

    def test_the_path_gate_refuses_with_them(self) -> None:
        source = inspect.getsource(safety_preflight.SafetyPreflight)
        self.assertIn("no_path_guard_refusal()", source)
        self.assertIn("exact_mesh_path_refusal()", source)

    def test_no_second_copy_of_either_sentence_is_left_inline(self) -> None:
        """The control: a gate that kept its literal beside the function would pass the test above and drift."""
        source = inspect.getsource(safety_preflight.SafetyPreflight)
        self.assertNotIn("no self_collision guard is wired, so nothing here can judge a path", source)
        self.assertNotIn("would answer this path with the capsule proxy", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
