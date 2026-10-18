"""A key removed on purpose is answered with its sentence wherever it is asked about, not only at load.

``REMOVED_KEYS`` (``src/config/schema/_removed.py``) holds, for every key that left the schema on purpose, the one
sentence that says what to write instead. The tree loader adds it to the refusal of a tree that still writes the key.
One other door sees the same key and must say the same sentence (cleanup phase 5, 2026-09-29):
``python -m src.config explain <dotted key>``, which an operator asks before editing, and which would otherwise answer
"NOT A KNOWN KEY" with the nearest living keys (the nearest to ``robot.grasping.closed_loop`` is
``robot.grasping.record_log_path``, which has nothing to do with it). The grasp presets, the second door beside it,
left on 2026-10-16 (the owner's cleanup).

The phase tests pin their own keys (``test_no_second_look_before_the_close.py``, ``test_no_viewpoint_is_planned.py``,
``test_the_gripper_says_whether_it_holds.py``). This one walks the whole table, so an entry added later is held to
that door without anyone remembering to add it here. A ``*`` segment stands for one map key, a camera id say.
"""

from __future__ import annotations

import contextlib
import io
import unittest


from src.config.__main__ import main as config_cli
from src.config.schema._removed import REMOVED_KEYS

def _concrete(dotted: str) -> str:
    """The key as a file would write it: a ``*`` map segment becomes a camera id."""
    return dotted.replace("*", "cam0")


def _one(text: str) -> str:
    """Whitespace folded, so a sentence the renderer wrapped still reads as the sentence."""
    return " ".join(text.split())


class TheTerminalExplainsARemovedKeyTests(unittest.TestCase):
    """``python -m src.config explain`` on every removed key: its own sentence, exit 0, and no near misses."""

    def _explain(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = config_cli(["explain", *argv])
        return code, out.getvalue()

    def test_every_removed_key_is_answered_with_its_own_sentence(self) -> None:
        for dotted, sentence in REMOVED_KEYS.items():
            key = _concrete(dotted)
            with self.subTest(key=key):
                code, said = self._explain(key)
                self.assertEqual(code, 0)
                self.assertIn("REMOVED ON PURPOSE", said)
                self.assertIn(_one(sentence), _one(said))
                self.assertNotIn("NOT A KNOWN KEY", said)
                self.assertNotIn("did you mean", said)

    def test_a_key_inside_a_removed_block_is_answered_with_the_block_s_sentence(self) -> None:
        code, said = self._explain("robot.grasping.dense_recovery.strategy.max_nudges")
        self.assertEqual(code, 0)
        self.assertIn(_one(REMOVED_KEYS["robot.grasping.dense_recovery"]), _one(said))

    def test_a_typo_is_still_a_typo(self) -> None:
        code, said = self._explain("robot.grasping.deep_ranker.enabeld")
        self.assertEqual(code, 0)
        self.assertIn("NOT A KNOWN KEY", said)
        self.assertNotIn("REMOVED ON PURPOSE", said)


if __name__ == "__main__":
    unittest.main()
