"""Every camera opener passes the same kind of thing, and the openers are the modules they should be.

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

The cell, the calibration CLI and the perception exerciser open one rig through
`Camera.from_config(camera_section)`, and `FrameProvider` stays the catalogue that stereo capture and
hand detection use. So there are two kinds of opener, each with its own argument, and the control
names the modules rather than counting them: a count breaks as soon as a site moves from one noun to
the other, and it cannot say which module moved.
"""

from __future__ import annotations

import ast
import unittest
from collections.abc import Callable
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TREES = ("src", "api")

#: The modules that open one rig through the camera noun.
_NOUN_OPENERS = {
    "src/robot/execution/autonomous_grasp/cells.py",
    "src/robot/execution/real_cell/calibrate.py",
    "src/robot/perception/__main__.py",
}
#: The modules that build the multi-rig catalogue.
_CATALOGUE_BUILDERS = {
    "src/models/handdetection/__main__.py",
    "src/camera/pipeline/stereo_capture.py",
}


def _sites(matches: Callable[[ast.expr], bool], keyword: str) -> list[tuple[str, int, str]]:
    """Every call whose callee `matches`, with the source of its first argument."""
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
                if not (isinstance(node, ast.Call) and matches(node.func)):
                    continue
                argument = (node.args[0] if node.args
                            else next((k.value for k in node.keywords if k.arg == keyword), None))
                text = ast.unparse(argument) if argument is not None else "<none>"
                found.append((path.relative_to(_ROOT).as_posix(), node.lineno, text))
    return found


def _catalogue_sites() -> list[tuple[str, int, str]]:
    return _sites(lambda f: isinstance(f, ast.Name) and f.id == "FrameProvider", "rigs")


def _noun_sites() -> list[tuple[str, int, str]]:
    return _sites(lambda f: (isinstance(f, ast.Attribute) and f.attr == "from_config"
                             and isinstance(f.value, ast.Name) and f.value.id == "Camera"), "camera_cfg")


class TheOpenersAreTheModulesTheyShouldBeTests(unittest.TestCase):

    def test_the_sweep_finds_exactly_the_modules_that_open_cameras(self) -> None:
        """The control. Over an empty list every test below passes while checking nothing, and a named
        set says which module moved when this fails."""
        self.assertEqual({path for path, _, _ in _noun_sites()}, _NOUN_OPENERS, _noun_sites())
        self.assertEqual({path for path, _, _ in _catalogue_sites()}, _CATALOGUE_BUILDERS, _catalogue_sites())


class EveryFrameProviderCallSitePassesRigsTests(unittest.TestCase):

    def test_none_of_them_passes_the_container_instead_of_its_rigs(self) -> None:
        """⛔ `cameras` IS THE CONTAINER; `cameras.rigs` IS THE LIST. The container is truthy and
        iterable, so it survives both the guard and the loop's first step, and fails on an attribute
        several frames away from the mistake."""
        offenders = [f"{path}:{line} passes {text!r}" for path, line, text in _catalogue_sites()
                     if text.endswith(".cameras") or text.endswith(".cameras)")]
        self.assertEqual([], offenders,
                         "a FrameProvider call site passes the camera CONTAINER, not its rigs")

    def test_every_site_names_rigs_somewhere_in_its_argument(self) -> None:
        """The positive form: whatever expression is used, `rigs` has to appear in it. Loose on
        purpose, because the correct sites already differ (`list(...)`, a local variable, a keyword),
        and a test that demanded one spelling would be about style."""
        vague = [f"{path}:{line} passes {text!r}" for path, line, text in _catalogue_sites()
                 if "rig" not in text.lower()]
        self.assertEqual([], vague,
                         "a FrameProvider argument mentions no rigs; if that is deliberate, say so "
                         "here rather than leaving the next reader to check every call site")


class EveryCameraNounSitePassesTheCameraSectionTests(unittest.TestCase):

    def test_every_site_passes_the_camera_section(self) -> None:
        """`Camera.from_config` takes the section (`app_cfg.camera`), which holds both the rigs and the
        primary key. Handing it `cameras` or `cameras.rigs` is the fifth site's mistake one level down."""
        wrong = [f"{path}:{line} passes {text!r}" for path, line, text in _noun_sites()
                 if not text.endswith(".camera")]
        self.assertEqual([], wrong, "a Camera.from_config call site does not pass the camera section")


if __name__ == "__main__":
    unittest.main()
