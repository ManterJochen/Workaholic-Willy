"""A guard that catches only `ImportError` cannot see a library the OS refused to load.

⛔⛔ **MEASURED 2026-09-05 on this box.** Windows Smart App Control began refusing `mujoco.dll` -- an
unsigned native build whose reputation it re-evaluated days after installation -- and the import
raised

    OSError: [WinError 4551] Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert

`tests/test_physics_mujoco.py` had the RIGHT shape (`mujoco = None`, `try: import ... except
ImportError`) and caught the wrong half, so the whole suite died at COLLECTION: exit 2, zero of 6,300
tests. Fixed in `6929bbd` by the parallel session.

⭐ **THIS FILE IS ABOUT THE TWO INSTANCES THAT WOULD HAVE COST MORE.** `ur_rtde` is the UR driver's
SDK, and its four module-level guards build a degrade path with an actionable message -- which a raw
`OSError` at import skips entirely, on the path a real arm is brought up on in September. Open3D's
`_import_open3d` exists to turn a missing library into a sentence, and a blocked one escaped its own
contract.

⚠ **AND WIDENING THE CATCH IS ONLY HALF THE FIX, WHICH IS WHY THESE TESTS ASSERT ON THE MESSAGE.**
"ur_rtde is not installed. Run: pip install ur_rtde" is FALSE when it is installed and blocked.
Sending an operator to reinstall a package they already have, next to a powered arm, is worse than
the raw OSError -- which at least names the policy. Absent and unloadable are two faults with two
remedies, and the guard has to tell them apart.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_CONNECTION = _REPO / "src" / "robot" / "drivers" / "ur" / "connection.py"
_VIEWER = _REPO / "src" / "robot" / "grasping" / "visualization" / "open3d_viewer.py"


class _BlockImport:
    """Make `import <name>` raise what a code-integrity policy raises, and nothing else."""

    def __init__(self, *names: str, error: Exception | None = None) -> None:
        self._names = set(names)
        self._error = error or OSError(
            "[WinError 4551] Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert"
        )
        self._real = builtins.__import__

    def __enter__(self) -> "_BlockImport":
        def _guard(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002, ANN001, ANN202
            if name in self._names:
                raise self._error
            return self._real(name, globals, locals, fromlist, level)

        builtins.__import__ = _guard
        return self

    def __exit__(self, *exc: object) -> None:
        builtins.__import__ = self._real


class TheDriverDegradesInsteadOfDyingTests(unittest.TestCase):
    """⚠ These RELOAD the driver module, so each restores it. A module left half-imported would
    poison every later test in the run, which is the same class of fault as the one under test."""

    def setUp(self) -> None:
        from src.robot.drivers.ur import connection

        self.connection = connection
        self.addCleanup(importlib.reload, connection)

    def _reload_with(self, *blocked: str, error: Exception | None = None):  # noqa: ANN202
        with _BlockImport(*blocked, error=error):
            return importlib.reload(self.connection)

    def test_the_sdk_is_installed_here_so_the_guard_is_dormant(self) -> None:
        """⭐ THE CONTROL, AND IT MATTERS MORE THAN IT LOOKS. All four modules import fine on this
        box, so every assertion below is about a path that is NOT taken here -- which is exactly how
        this defect survived. If this ever fails, the tests underneath are measuring something else.
        """
        for name in ("rtde_control", "rtde_receive", "rtde_io", "dashboard_client"):
            with self.subTest(module=name):
                importlib.import_module(name)
        self.assertEqual(self.connection._CORE_UNAVAILABLE, "")

    def test_a_blocked_core_degrades_to_none_rather_than_killing_the_import(self) -> None:
        """⛔ THE DEFECT. Before the fix this reload raised OSError out of a module-level import, so
        the UR driver could not be imported at all and `connect()`'s message never ran."""
        mod = self._reload_with("rtde_control", "rtde_receive")
        self.assertIsNone(mod.rtde_control)
        self.assertIsNone(mod.rtde_receive)
        self.assertTrue(mod._CORE_UNAVAILABLE)

    def test_the_message_says_INSTALLED_rather_than_telling_them_to_install_it(self) -> None:
        """⛔⛔ THE HALF THAT IS NOT ABOUT THE EXCEPTION TYPE. Widening the catch alone would hand a
        blocked operator "ur_rtde is not installed. Run: pip install ur_rtde" -- a sentence that is
        false, and that sends them to reinstall something they already have."""
        mod = self._reload_with("rtde_control", "rtde_receive")
        why = mod._CORE_UNAVAILABLE
        self.assertIn("IS installed", why)
        self.assertIn("could not be LOADED", why)
        self.assertNotIn("pip install", why)
        self.assertIn("4551", why, "the operating system's own reason is carried through")

    def test_an_absent_sdk_still_says_pip_install(self) -> None:
        """⭐ THE CONTROL ON THE TEST ABOVE. If the two branches were collapsed, this would go green
        while an absent SDK started telling people to read an event log."""
        mod = self._reload_with(
            "rtde_control", "rtde_receive", error=ImportError("No module named 'rtde_control'"))
        self.assertIn("pip install ur_rtde", mod._CORE_UNAVAILABLE)
        self.assertNotIn("IS installed", mod._CORE_UNAVAILABLE)

    def test_connect_raises_the_recorded_reason_rather_than_a_fixed_sentence(self) -> None:
        """The message an operator actually reads is the one `connect()` raises, not the constant."""
        mod = self._reload_with("rtde_control", "rtde_receive")
        conn = mod.URConnection.__new__(mod.URConnection)
        with self.assertRaises(ImportError) as caught:
            mod.URConnection.connect(conn)
        self.assertIn("could not be LOADED", str(caught.exception))

    def test_a_blocked_OPTIONAL_extra_degrades_and_says_why(self) -> None:
        """⚠ `rtde_io` and `dashboard_client` are best-effort by design, so a blocked one must
        degrade exactly as an absent one does -- but not silently. `_require_io`'s own message asks
        'is ur_rtde's rtde_io present?', and the recorded reason answers it."""
        mod = self._reload_with("rtde_io")
        self.assertIsNone(mod.rtde_io)
        self.assertIn("IS installed", mod._IO_UNAVAILABLE)
        self.assertIsNotNone(mod.rtde_control, "the move core must be unaffected")

        conn = mod.URConnection.__new__(mod.URConnection)
        conn._io = None
        with self.assertRaises(RuntimeError) as caught:
            mod.URConnection._require_io(conn)
        self.assertIn("could not be LOADED", str(caught.exception))


class TheViewerTurnsBothFaultsIntoASentenceTests(unittest.TestCase):
    def test_a_blocked_open3d_does_not_escape_as_a_raw_oserror(self) -> None:
        from src.robot.grasping.visualization import open3d_viewer

        with _BlockImport("open3d"):
            with self.assertRaises(ImportError) as caught:
                open3d_viewer._import_open3d()
        message = str(caught.exception)
        self.assertIn("open3d is installed", message)
        self.assertNotIn("pip install open3d", message)
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_an_absent_open3d_still_gets_the_install_hint(self) -> None:
        from src.robot.grasping.visualization import open3d_viewer

        with _BlockImport("open3d", error=ImportError("No module named 'open3d'")):
            with self.assertRaises(ImportError) as caught:
                open3d_viewer._import_open3d()
        self.assertIn("pip install open3d", str(caught.exception))


class TheGuardsInTheseTwoFilesAreCompleteTests(unittest.TestCase):
    """⭐ PARSED, NOT GREPPED, and that distinction is the reason this class exists.

    The parallel session first swept for this defect with a hand-written library-name list and
    reported five sites where there are 28. A list of names has to be maintained by whoever adds the
    next library, which is precisely the person who will not know it is there. Reading the AST asks
    the structural question instead: does every `try` that contains an `import` also catch `OSError`?

    ⚠ SCOPED TO THE TWO FILES THIS COMMIT REPAIRS. Twenty-six other sites in the repository still
    have the narrow guard; most are pure-Python libraries, which CANNOT hit this -- absent is
    `ImportError` and there is nothing else. Widening this test to the whole tree would be a claim
    nobody has measured, and the honest thing is a scope that says so.
    """

    def _narrow_guards(self, path: Path) -> list[int]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            imports = any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(node))
            if not imports or not node.handlers:
                continue
            caught: set[str] = set()
            for handler in node.handlers:
                for name in ast.walk(handler.type) if handler.type else []:
                    if isinstance(name, ast.Name):
                        caught.add(name.id)
            if "ImportError" in caught and not ({"OSError", "Exception", "BaseException"} & caught):
                offenders.append(node.lineno)
        return offenders

    def test_neither_file_still_catches_only_import_error_around_an_import(self) -> None:
        for path in (_CONNECTION, _VIEWER):
            with self.subTest(file=path.name):
                self.assertEqual(self._narrow_guards(path), [])

    def test_the_scan_can_actually_see_a_narrow_guard(self) -> None:
        """⚠ A test over an empty set passes loudest. This proves the detector fires."""
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
            fh.write("try:\n    import mujoco\nexcept ImportError:\n    mujoco = None\n")
            probe = Path(fh.name)
        self.addCleanup(probe.unlink, True)
        self.assertEqual(self._narrow_guards(probe), [1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
