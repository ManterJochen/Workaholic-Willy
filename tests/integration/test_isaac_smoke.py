"""Isaac Sim smoke gates (M0) — run on the Windows RTX workstation only.

These correspond to the M0 gate in ``docs/ISAAC_VALIDATION_PLATFORM.md`` §2/§6. They
carry the ``isaac`` marker and auto-skip anywhere ``isaacsim`` is absent (see
``conftest.py``), so the default ``pytest tests`` run stays green on the MacBook / CI.
On the workstation they boot a single headless ``SimulationApp`` (via the session-scoped
``isaac_world`` fixture) and assert a non-empty RGB + depth render.

Validated on the RTX 5080 (2026-06-04): warm boot ~7-9 s, RGB mean ~228, depth > 0.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.isaac


def test_isaac_sdk_importable() -> None:
    """Precondition for every other isaac gate: the SDK imports on this box."""

    assert importlib.util.find_spec("isaacsim") is not None


def test_isaac_headless_render(isaac_world) -> None:
    """§2 Smoke 0: headless boot + render produces non-empty RGB + depth on the 5080.

    The ``isaac_world`` fixture boots the app once and adds lighting (the default ground
    plane has none → RGB would be black while depth, being geometric, is fine). RTX needs
    a few frames to converge from black, hence the warmup loop.
    """

    iw = isaac_world

    # The asset root must resolve, else robot USD loading in M1 silently fails / hangs.
    assert iw.get_assets_root_path() is not None, "Isaac asset root did not resolve"

    # Standard Camera (NOT TiledCamera — that hangs on Blackwell, see §1/§10).
    cam = iw.Camera(prim_path="/World/smoke_cam", resolution=(1280, 720))
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    for _ in range(30):  # first frames are black; let the RTX render converge
        iw.world.step(render=True)

    rgb = iw.np.asarray(cam.get_rgb())
    depth = iw.np.asarray(cam.get_depth())  # distance_to_image_plane (true Z), metres
    K = iw.np.asarray(cam.get_intrinsics_matrix())

    assert rgb.shape == (720, 1280, 3) and rgb.any(), "RGB frame is empty/black"
    assert depth.shape == (720, 1280) and (depth > 0).any(), "depth has no positive values"
    assert K.shape == (3, 3) and K[0, 0] > 0 and K[1, 1] > 0, "intrinsics look invalid"
