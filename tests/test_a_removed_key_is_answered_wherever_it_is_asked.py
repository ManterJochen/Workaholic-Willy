"""A key removed on purpose is answered with its sentence wherever it is asked about, not only at load.

``REMOVED_KEYS`` (``src/config/schema/_removed.py``) holds, for every key that left the schema on purpose, the one
sentence that says what to write instead. The tree loader adds it to the refusal of a tree that still writes the key.
Two other doors see the same key and must say the same sentence (cleanup phase 5, 2026-09-29):

* ``python -m src.config explain <dotted key>``, which an operator asks before editing, and which would otherwise
  answer "NOT A KNOWN KEY" with the nearest living keys (the nearest to ``robot.grasping.closed_loop`` is
  ``robot.grasping.occlusion``, which has nothing to do with it);
* ``validate_preset`` (``src.robot.grasping.replay.presets``), which validates a preset outside the tree loader and
  would otherwise surface pydantic's bare "Extra inputs are not permitted".

The phase tests pin their own keys (``test_no_second_look_before_the_close.py``, ``test_no_viewpoint_is_planned.py``,
``test_the_gripper_says_whether_it_holds.py``). This one walks the whole table, so an entry added later is held to
both doors without anyone remembering to add it here. A ``*`` segment stands for one map key, a camera id say.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
from pydantic import ValidationError

from src.config.__main__ import main as config_cli
from src.config.schema._removed import REMOVED_KEYS, removed_key_sentence

#: A robot block with no hardware, the smallest base a preset validates against.
_BASE = {"vendor": "dummy", "gripper": {"vendor": "none"}}


def _concrete(dotted: str) -> str:
    """The key as a file would write it: a ``*`` map segment becomes a camera id."""
    return dotted.replace("*", "cam0")


def _one(text: str) -> str:
    """Whitespace folded, so a sentence the renderer wrapped still reads as the sentence."""
    return " ".join(text.split())


def _refused_as(dotted: str) -> str:
    """The sentence a schema refusal names: the outermost removed block the key sits in, or the key itself.

    ``extra='forbid'`` stops at the first segment it does not know, so a key inside a block that left whole is refused
    at the block (``robot.grasping.verification.post_lift_vision_check`` is refused as ``robot.grasping.verification``).
    """
    parts = dotted.split(".")
    for end in range(1, len(parts) + 1):
        sentence = removed_key_sentence(".".join(parts[:end]))
        if sentence is not None:
            return sentence
    raise AssertionError(f"{dotted} has no removed-on-purpose sentence")


def _as_preset(dotted: str) -> dict:
    """A preset overlay that writes ``dotted`` (a robot key) with a value an old file might have held."""
    parts = dotted.split(".")[1:]
    overlay: dict = {}
    here = overlay
    for part in parts[:-1]:
        here = here.setdefault(part, {})
    here[parts[-1]] = 1
    return overlay


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
        code, said = self._explain("robot.grasping.fusion.enabeld")
        self.assertEqual(code, 0)
        self.assertIn("NOT A KNOWN KEY", said)
        self.assertNotIn("REMOVED ON PURPOSE", said)


class APresetThatWritesARemovedKeyTests(unittest.TestCase):
    """``validate_preset`` on a preset that still writes each removed key: refused, naming the sentence once."""

    def _refusal(self, overlay: dict) -> str:
        from src.robot.grasping.replay import presets

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "old.yaml").write_text(yaml.safe_dump(overlay), encoding="utf-8")
            with mock.patch.object(presets, "_PRESETS_DIR", Path(tmp)):
                with self.assertRaises(ValidationError) as caught:
                    presets.validate_preset("old", base=_BASE)
        return _one(str(caught.exception))

    def test_every_removed_key_is_refused_with_its_sentence(self) -> None:
        for dotted in REMOVED_KEYS:
            key = _concrete(dotted)
            with self.subTest(key=key):
                said = self._refusal(_as_preset(key))
                self.assertEqual(said.count("removed on purpose"), 1, said)
                self.assertIn(_one(_refused_as(key)), said)


if __name__ == "__main__":
    unittest.main()
