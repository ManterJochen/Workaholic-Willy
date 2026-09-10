"""An engine that cannot be imported is not available, whatever the module finder says.

⛔ **MEASURED 2026-09-10 ON THIS WORKSTATION.** `engine_is_available("mujoco")` answered
``(True, "")`` while `import mujoco` raised
``OSError: [WinError 4551] Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert``. MuJoCo's
own ``__init__`` loads its bundled plugins through ``ctypes.CDLL``, and a Windows application-control
policy refuses those DLLs. The module is findable and unusable at the same time, and
``importlib.util.find_spec`` only ever answered the first half.

⚠ **THE NARROW CHECK WAS DELIBERATE AND IT WAS RIGHT FOR ITS OWN CASE.** The docstring said it: probe
"WITHOUT importing anything heavy on failure", which is the correct instinct for Isaac, a multi-gigabyte
import that starts a renderer. It was then applied to MuJoCo, a 17 MB wheel, where the cost it avoids
is negligible and the question it fails to answer is the only one an operator asked. A rule written
from the case that prompted it, missing the property that matters.

⭐ **MACHINE INDEPENDENT ON PURPOSE.** Asserting that MuJoCo is unavailable would pass on this box for
the wrong reason and fail on a healthy one. So the import is made to fail here, and what is pinned is
that the function believes the failure rather than the finder.
"""

from __future__ import annotations

import builtins
import sys
import types
import unittest
from contextlib import contextmanager

from datagen.render.engine import engine_is_available


@contextmanager
def _import_of(name: str, raises: BaseException):
    """Make one module name fail to import, however it is imported, and restore afterwards."""
    real = builtins.__import__

    def fake(module, *args, **kwargs):
        if module == name or module.startswith(f"{name}."):
            raise raises
        return real(module, *args, **kwargs)

    # ⛔ THE CACHE IS THE OTHER HALF OF "however it is imported". `importlib.import_module` returns
    # a module already in `sys.modules` without consulting the import machinery at all, so patching
    # `__import__` alone makes a failure that a loaded module silently satisfies. MEASURED: with
    # `mujoco` pre-imported, three assertions in this file invert, and the file still passes when it
    # is run alone. That was invisible while this box refused the plugin DLLs, because nothing could
    # fill the cache; it appeared the morning the policy let them through.
    cached = {k: v for k, v in sys.modules.items() if k == name or k.startswith(f"{name}.")}
    for key in cached:
        del sys.modules[key]

    builtins.__import__ = fake
    try:
        yield
    finally:
        builtins.__import__ = real
        sys.modules.update(cached)


class AnUnimportableEngineIsUnavailableTests(unittest.TestCase):
    def test_a_blocked_library_is_reported_unavailable(self) -> None:
        """The measured case: findable, and refused by the operating system."""
        blocked = OSError(
            "[WinError 4551] Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert"
        )
        with _import_of("mujoco", blocked):
            available, why_not = engine_is_available("mujoco")
        self.assertFalse(available)
        self.assertIn("4551", why_not)

    def test_the_reason_names_the_failure_rather_than_guessing(self) -> None:
        """An operator whose DLL is blocked must not be told to `pip install mujoco`.

        That was the only sentence the old refusal could produce, and it is the wrong instruction for
        every cause except an absent package. Sending somebody to reinstall a package that is already
        installed costs an hour and teaches them the message cannot be trusted.
        """
        with _import_of("mujoco", OSError("blocked by policy")):
            _available, why_not = engine_is_available("mujoco")
        self.assertNotIn("pip install", why_not)
        self.assertIn("blocked by policy", why_not)

    def test_a_loaded_module_does_not_satisfy_the_refusal(self) -> None:
        """⛔ THE ORDERING CONTROL, and it is the defect this file shipped with.

        Every assertion above asks the engine a question through a fake import. A module already in
        `sys.modules` answers it first, and then what is measured is the cache rather than the
        function. The three tests around this one passed alone and failed in the suite for exactly
        that reason, on the day this workstation stopped refusing MuJoCo.

        Pinned on the HELPER and on a sentinel module, so it means the same thing on a box that can
        import MuJoCo and on one that cannot, which is what the module docstring promises.
        """
        before = sys.modules.get("mujoco")
        sys.modules["mujoco"] = types.ModuleType("mujoco")
        try:
            with _import_of("mujoco", OSError("blocked by policy")):
                # assertFalse, not assertNotIn: the latter prints the whole of sys.modules,
                # and a quarter of a megabyte of dictionary is not a diagnosis.
                self.assertFalse("mujoco" in sys.modules,
                                 "a loaded module shadows the fake, so the refusal is never asked")
                available, why_not = engine_is_available("mujoco")
            self.assertFalse(available)
            self.assertIn("blocked by policy", why_not)
            self.assertIsInstance(sys.modules.get("mujoco"), types.ModuleType,
                                  "the helper did not put the cache back")
        finally:
            if before is None:
                sys.modules.pop("mujoco", None)
            else:
                sys.modules["mujoco"] = before

    def test_a_missing_package_still_says_pip_install(self) -> None:
        """The byte-identical half: an absent package keeps the instruction that fits it."""
        with _import_of("mujoco", ModuleNotFoundError("No module named 'mujoco'")):
            available, why_not = engine_is_available("mujoco")
        self.assertFalse(available)
        self.assertIn("pip install mujoco", why_not)

    def test_isaac_is_still_probed_without_importing_it(self) -> None:
        """Isaac keeps the cheap probe, and that asymmetry is the point rather than an oversight.

        Importing `isaacsim` takes tens of seconds and starts a renderer, so a probe that did it is a
        probe nobody would call. MuJoCo is a 17 MB wheel. The two engines get different treatment
        because the cost of the honest answer differs by three orders of magnitude, and that reasoning
        belongs in the code rather than in a reviewer's head.
        """
        source_has_import_module = "isaacsim" in _engine_source()
        self.assertTrue(source_has_import_module)
        self.assertIn("find_spec", _engine_source())

    def test_none_needs_no_probe_at_all(self) -> None:
        available, why_not = engine_is_available("none")
        self.assertTrue(available)
        self.assertEqual(why_not, "")


def _engine_source() -> str:
    import pathlib

    return (
        pathlib.Path(__file__).resolve().parents[1] / "datagen" / "render" / "engine.py"
    ).read_text(encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
