"""Every `FrameProvider` call site passes the same kind of thing.

⛔⛔ **FOUR OF FIVE WERE RIGHT, AND THAT IS WHY THE FIFTH SURVIVED.**
`backend/src/models/handdetection/__main__.py` passed `config.camera.cameras` -- the
`CameraSystemConfig` MODEL -- where the other four pass `list(cfg.camera.cameras.rigs)`.

The failure is the quiet kind twice over. A Pydantic model is TRUTHY, so `FrameProvider`'s own
`if not rigs` guard let it through; iterating a model yields `('primary_rig_id', 'webcam_main')` pairs, so
`rig.rig_id` raised `AttributeError`; and the command's `except (FileNotFoundError, ImportError,
ValueError)` does not catch that, so `handdetection --rig` exited 1 with a traceback that reads like
"no hand was found".

⚠ **NO TEST IMPORTED THAT MODULE**, which is the whole reason four correct siblings did not make the
fifth correct. Found by a peer session sweeping a different subsystem.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TREES = ("src", "api")


def _call_sites() -> list[tuple[str, int, str]]:
    """Every `FrameProvider(...)` construction, with the source of its first argument."""
    found: list[tuple[str, int, str]] = []
    for tree in _TREES:
        root = _ROOT / tree
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            try:
                module = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:                       # pragma: no cover - not our problem here
                continue
            for node in ast.walk(module):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "FrameProvider"):
                    continue
                argument = (node.args[0] if node.args
                            else next((k.value for k in node.keywords if k.arg == "rigs"), None))
                text = ast.unparse(argument) if argument is not None else "<none>"
                found.append((str(path.relative_to(_ROOT)), node.lineno, text))
    return found


class EveryFrameProviderCallSitePassesRigsTests(unittest.TestCase):

    def test_the_sweep_finds_the_call_sites_at_all(self) -> None:
        """The control. Over an empty list the test below passes while checking nothing."""
        self.assertGreaterEqual(len(_call_sites()), 4, _call_sites())

    def test_none_of_them_passes_the_container_instead_of_its_rigs(self) -> None:
        """⛔ `cameras` IS THE CONTAINER; `cameras.rigs` IS THE LIST. The container is truthy and
        iterable, so it survives both the guard and the loop's first step, and fails on an attribute
        several frames away from the mistake."""
        offenders = [f"{path}:{line} passes {text!r}" for path, line, text in _call_sites()
                     if text.endswith(".cameras") or text.endswith(".cameras)")]
        self.assertEqual([], offenders,
                         "a FrameProvider call site passes the camera CONTAINER, not its rigs")

    def test_every_site_names_rigs_somewhere_in_its_argument(self) -> None:
        """The positive form: whatever expression is used, `rigs` has to appear in it. Loose on
        purpose, because the four correct sites already differ (`list(...)`, a local variable, a
        keyword), and a test that demanded one spelling would be about style."""
        vague = [f"{path}:{line} passes {text!r}" for path, line, text in _call_sites()
                 if "rig" not in text.lower()]
        self.assertEqual([], vague,
                         "a FrameProvider argument mentions no rigs; if that is deliberate, say so "
                         "here rather than leaving the next reader to check five call sites")


if __name__ == "__main__":
    unittest.main()
