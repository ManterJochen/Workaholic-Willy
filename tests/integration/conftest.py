"""Isaac integration harness: skip-gating + a session-scoped ``SimulationApp``.

``@pytest.mark.isaac`` tests auto-skip anywhere ``isaacsim`` is absent (the MacBook /
CI), so the default ``pytest tests`` run stays green everywhere. On the Windows RTX
workstation they run against a single headless Isaac app booted by the ``isaac_world``
fixture. See ``docs/ISAAC_VALIDATION_PLATFORM.md``.

Three workstation-specific robustness measures live here (all validated on the RTX 5080,
2026-06-04 — each was an actual failure mode hit during M0 bring-up):

1. **argv sanitization** — ``SimulationApp`` forwards ``sys.argv`` to Omniverse Kit's own
   parser, which rejects pytest's flags ("Ill formed parameter: -m") and then crashes.
2. **capture suspension** — pytest's stdout/stderr fd-capture destabilizes Kit's boot
   (boots fine only with ``-s``); we suspend global capture around the session so ``-s``
   is not required.
3. **eager result persistence** — Isaac's headless ``SimulationApp.close()`` can segfault
   during session teardown (upstream bug, IsaacLab #3730), losing pytest's junit + summary.
   ``pytest_runtest_logreport`` writes each outcome the moment the test body finishes, so
   the gate survives the shutdown crash (set ``WILLY_ISAAC_RESULTS`` to a file to enable).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest

_HAS_ISAAC = importlib.util.find_spec("isaacsim") is not None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if _HAS_ISAAC:
        return
    skip_isaac = pytest.mark.skip(
        reason="requires a real Isaac Sim install (run on the Windows workstation)"
    )
    for item in items:
        if "isaac" in item.keywords:
            item.add_marker(skip_isaac)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Persist each test's call-phase outcome immediately (see module docstring #3)."""

    if report.when != "call":
        return
    path = os.environ.get("WILLY_ISAAC_RESULTS")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{report.outcome.upper()} {report.nodeid}\n")


@pytest.fixture(scope="session")
def isaac_world(request: pytest.FixtureRequest):
    """Boot ONE headless Isaac ``SimulationApp`` for the whole isaac test session.

    ``SimulationApp`` is a *process singleton*, so this is **session-scoped** — a single
    boot shared by every isaac-marked test. The ``isaacsim.*`` / ``pxr`` / ``omni.*``
    runtime APIs only exist once the app has loaded its extensions, so every such import
    is **deferred into the fixture body** (after the boot) — never at module top, which
    keeps the macOS / CI mock suite importable. A ``DomeLight`` + ``DistantLight`` are
    added because ``add_default_ground_plane()`` carries no light (RGB renders black
    otherwise; depth, being geometric, is unaffected — see §2 of the platform doc).

    Yields a namespace with the booted ``world`` + the deferred handles tests need.
    Validated on the Windows RTX 5080 (2026-06-04): warm boot ~7-9 s, non-empty RGB+depth.
    """

    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"  # must be set before the first isaacsim import
    from isaacsim import SimulationApp

    # See module docstring (1) + (2): strip argv to the program name and suspend pytest's
    # global capture across the boot so Kit boots cleanly under pytest.
    capman = request.config.pluginmanager.get_plugin("capturemanager")
    if capman is not None:
        capman.suspend_global_capture(in_=True)
    saved_argv, sys.argv = sys.argv, sys.argv[:1]

    app = None
    try:
        app = SimulationApp({"headless": True})
        sys.argv = saved_argv

        import numpy as np
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Sdf, UsdLux

        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        stage = omni.usd.get_context().get_stage()
        UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(1000.0)
        UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight")).CreateIntensityAttr(3000.0)
        world.reset()

        yield SimpleNamespace(
            app=app,
            world=world,
            stage=stage,
            np=np,
            Camera=Camera,
            get_assets_root_path=get_assets_root_path,
        )
    finally:
        sys.argv = saved_argv
        if app is not None:
            app.close()  # may segfault on headless shutdown (see module docstring #3)
        if capman is not None:
            capman.resume_global_capture()
