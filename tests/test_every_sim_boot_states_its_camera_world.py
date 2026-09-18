"""Every Isaac boot in the repository states its camera world at the call, as the owner decided.

The bootstrap refuses a cuRobo boot that states nothing, but only when it runs, which is on the box, a minute into an
Isaac start. This reads every runner instead: each ``bootstrap_sim_cell`` and ``build_service`` call passes
``camera_world=``. A runner added without it turns this red off the box.

The count is pinned from below so a scan that stopped finding calls (a moved directory, a renamed function) cannot pass
by finding nothing, and the control proves the scan flags a call that leaves the keyword out.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCANNED = ("src/willy_sim", "scripts/isaac")
_BOOTS = ("bootstrap_sim_cell", "build_service")
#: This tree holds 45 calls across the Isaac runners and scripts, counted with ``_calls``.
_AT_LEAST = 40


def _calls(source: str) -> list[tuple[str, int, bool]]:
    rows = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
        if name in _BOOTS:
            # A ``**kwargs`` spread does not count: every runner spreads its cell kwargs, which never carry it.
            rows.append((name, node.lineno, any(kw.arg == "camera_world" for kw in node.keywords)))
    return rows


class EveryBootStatesItsCameraWorldTests(unittest.TestCase):
    def test_every_call_passes_camera_world(self) -> None:
        found = 0
        silent = []
        for folder in _SCANNED:
            for path in sorted((_ROOT / folder).rglob("*.py")):
                for name, line, stated in _calls(path.read_text(encoding="utf-8")):
                    found += 1
                    if not stated:
                        silent.append(f"{path.relative_to(_ROOT).as_posix()}:{line} {name}")
        self.assertEqual([], silent, "an Isaac boot states no camera world; pass a CameraWorldDecline naming the runner")
        self.assertGreaterEqual(found, _AT_LEAST, "the scan found fewer boots than exist; it stopped looking")

    def test_the_control_a_call_without_the_keyword_is_flagged(self) -> None:
        source = (
            "cell = bootstrap_sim_cell(None, headless=True)\n"
            "svc = build_service(prompt='x', camera_world=decline)\n"
            "boot.bootstrap_sim_cell(None, headless=True, **cell_kwargs)\n"
        )
        self.assertEqual(
            [("bootstrap_sim_cell", 1, False), ("build_service", 2, True), ("bootstrap_sim_cell", 3, False)],
            _calls(source),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
