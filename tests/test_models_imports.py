"""Gap L1.12 (+ D3-flavoured) — import-smoke for the optional voice/gesture model surface.

handdetection (MediaPipe) + speech (sounddevice/transformers-Whisper) are NOT part of the grasp
pipeline and are omitted from coverage, so import-path regressions (the kind that left palm_finder's
module paths stale + invisible) never reach CI. This smoke-imports them so such breakage is caught.

It also pins the shape of the optional-dependency guard, which is the property that makes the extra
optional at all: importing any module here must work WITHOUT mediapipe, and only CONSTRUCTING a
detector may refuse. A module that imported mediapipe at top level would break every host that runs
grasping alone -- and would do it at import time, far from anything that mentions hands.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "src.models.handdetection",
        "src.models.handdetection.constants",
        "src.models.handdetection.factory",
        "src.models.handdetection.gestures",
        "src.models.handdetection.hand_finder",
        "src.models.handdetection.landmarks",
        "src.models.handdetection.model_files",
        "src.models.handdetection.palm_detector",
        "src.models.handdetection.types",
        "src.models.speech.speech_to_text",
    ],
)
def test_optional_model_module_imports(module: str) -> None:
    importlib.import_module(module)  # must not raise (mediapipe stays optional)


def test_no_handdetection_module_imports_mediapipe_at_top_level() -> None:
    """The guard has to be at CONSTRUCTION, not at import -- checked in the source, not by luck.

    An `import mediapipe` at module scope would still pass the smoke test above on a machine that
    happens to have the extra installed, which is exactly the machine where this is written. So the
    check reads the files.
    """
    import ast
    import pathlib

    package = pathlib.Path("src/models/handdetection")
    offenders: list[str] = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # top level ONLY: a function-scoped import is the correct pattern
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Try):
                continue  # the guarded probe in model_files.py is the one legitimate place
            if any(name.split(".")[0] == "mediapipe" for name in names):
                offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, (
        f"mediapipe is imported at module scope in {offenders} -- that makes the optional extra "
        f"mandatory for every host that imports this package."
    )


def test_the_mediapipe_guard_names_what_was_being_built() -> None:
    """`require_mediapipe` must say which feature is unavailable, not just that a library is."""
    from src.models.handdetection import model_files

    original = model_files.MEDIAPIPE_AVAILABLE
    model_files.MEDIAPIPE_AVAILABLE = False  # type: ignore[misc]
    try:
        with pytest.raises(ImportError) as caught:
            model_files.require_mediapipe("gesture recognition")
    finally:
        model_files.MEDIAPIPE_AVAILABLE = original  # type: ignore[misc]

    message = str(caught.value)
    assert "gesture recognition" in message
    assert "requirements.txt" in message
