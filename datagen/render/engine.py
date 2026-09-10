"""Which engine renders a scene: `render.engine`, and the factory that makes the key mean something.

Why a second engine exists at all. Scene generation and training are the user's job, and Isaac Sim is
a multi-GB NVIDIA install that in many companies needs an approval process. Installability is not a
convenience here; it is the wall the whole plan hits.

The contract is a file format, not an API. `build_cloud_corpus` reads exactly `scene.json`,
`<view>_depth.png`, `<view>_instances.png`, an optional `<view>_arm.png`, `grasps.jsonl` and
`provenance.json`; `label-grasps` never opens an image and its only engine input is
`scene.json.settled_poses_mm_xyzw`. So a backend owes four things: depth in mm, per-object
silhouettes, settled poses, and the camera pose and K it used. Everything else is colour, and `rgb`
may stay `None` throughout.

A selector is as capable of being inert as a switch. A fail-closed factory that no construction site
calls changes nothing, and a guard that enumerates boolean `enabled` fields cannot see a string
selector at all. Every site that builds an engine for a run goes through `build_engine`.

A backend is not admissible until it passes the checks in `render/equivalence.py`, and they grade on
an oblique camera. A principal-point offset of one pixel moves reconstructed geometry by roughly 0.5
to 2 mm, while an overhead view is structurally blind to it, because a principal-point shift cannot
change recovered Z on a plane normal to the axis. The repo's own `_DEPTH_MATCH_TOLERANCE_MM = 2.0`
compares two renders of the same geometry, so it cannot see a constant offset either. A second engine
that produces a subtly different corpus is worse than no second engine: the difference is invisible
in the `.npz` and confounds engine with scene family in every later number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from datagen.assets.manifest import AssetManifest
    from datagen.config import DatagenConfig
    from datagen.render.result import SceneRenderResult

__all__ = ["ENGINES", "SceneEngine", "build_engine", "engine_is_available"]

#: Every engine this build knows, and what each one costs an operator to install. The value is the
#: honest one-line answer to "can I run this here", because that is the question the key exists for.
ENGINES: dict[str, str] = {
    "isaac": "NVIDIA Isaac Sim 5.1; highest fidelity, needs a multi-GB install and an NVIDIA GPU",
    "mujoco": ("MuJoCo settles the scene; the z-buffer renders it. `pip install mujoco`; 24 wheels "
               "across Windows, macOS, Linux and ARM for python 3.10-3.14, Apache-2.0, ~17 MB. Needed "
               "only for the `pile` family, which is the one that drops objects"),
    "none": ("no physics engine at all; analytic seating, a support-polygon stability test and a "
             "numpy z-buffer. Nothing to install beyond this repo's own requirements, and it REFUSES "
             "the `pile` family by name rather than faking a settle"),
}


@runtime_checkable
class SceneEngine(Protocol):
    """What datagen asks of a rendering backend. Four things, and a lifecycle.

    Deliberately narrow. Everything the engine-free modules already do (the camera model in
    `camera.py`, the viewpoint policy in `views.py`, the depth-sensor model in `depth_noise.py`, the
    label extraction in `labels.py` and the writer in `writer.py`) stays out of here, because a
    second engine that re-implemented any of them would be a second definition of what a label means.
    """

    def render(self, spec: Any, manifest: "AssetManifest",
               rng: np.random.Generator) -> "SceneRenderResult":
        """Settle the scene, render every planned view, and return the result.

        The retry in `build.py` calls this again on failure, so it must be safe to call repeatedly
        with the same spec. The lifecycle control in `grasps/physics.py` (scene A, then scene B, then
        scene A in one session must equal scene A fresh) is part of that contract, not an
        optimisation.
        """
        ...

    def __enter__(self) -> "SceneEngine":
        """Open whatever the backend needs. `build.py` drives every engine as a context manager."""
        ...

    def __exit__(self, *exc: Any) -> None:
        ...


def engine_is_available(name: str) -> tuple[bool, str]:
    """``(importable, why not)`` for one engine, without importing anything heavy on failure.

    Used by the refusal in `build_engine`, so an operator learns which engines this machine can run
    before spending an hour finding out. Returns the reason as prose, because "ModuleNotFoundError:
    isaacsim" is not an instruction.
    """
    import importlib.util  # noqa: PLC0415

    if name not in ENGINES:
        return False, f"unknown engine {name!r}; expected one of {sorted(ENGINES)}"
    if name == "mujoco":
        # Imported, not looked up, and the asymmetry with Isaac below is deliberate. Measured
        # 2026-09-10 on the development workstation: `find_spec("mujoco")` answered yes while
        # `import mujoco` raised `OSError: [WinError 4551]`, a Windows application-control policy
        # refusing the bundled plugin DLLs that MuJoCo's own `__init__` loads through `ctypes.CDLL`.
        # The module is findable and unusable at once, and a caller that trusted the finder met the
        # real answer several minutes later inside a run. The wheel is about 17 MB and imports in
        # well under a second, so the honest probe costs nothing worth saving. Isaac keeps
        # `find_spec` because importing it takes tens of seconds and starts a renderer: a probe that
        # did that is a probe nobody would call.
        try:
            importlib.import_module("mujoco")
        except ModuleNotFoundError:
            return False, ("MuJoCo is not installed. `pip install mujoco`; it ships wheels for "
                           "Windows, macOS, Linux and ARM on python 3.10-3.14 (Apache-2.0, ~17 MB), "
                           "so unlike Isaac it needs no separate installer, GPU or admin rights.")
        except Exception as error:  # noqa: BLE001 - report whatever refused, do not guess
            # Anything else is installed and refusing: a blocked DLL, a broken build, an ABI
            # mismatch against numpy. The cause is quoted rather than classified, because
            # "pip install mujoco" is the wrong instruction for every one of them, and sending an
            # operator to reinstall a package they already have teaches them not to trust the message.
            return False, (f"MuJoCo is installed and will not import: {type(error).__name__}: "
                           f"{error}. On Windows this is usually an application-control policy "
                           "refusing the bundled plugin DLLs under `mujoco/plugin`.")
        return True, ""
    if name == "none":
        # Nothing to probe: it is numpy and trimesh, both already required. That is the point: the
        # engine question exists because Isaac cannot be installed in many companies.
        return True, ""
    if name == "isaac":
        if importlib.util.find_spec("isaacsim") is None:
            return False, ("Isaac Sim is not importable in this interpreter. It is a separate "
                           "multi-GB install with its own bundled Python; run datagen with "
                           "Isaac's `python.bat`, or choose another engine.")
        return True, ""
    return False, f"engine {name!r} is registered but has no availability probe"   # pragma: no cover


def build_engine(config: "DatagenConfig", *, headless: bool = True, **kwargs: Any) -> "SceneEngine":
    """The engine this config asks for.

    Fails closed, and never falls back. An engine that quietly substituted another would file one
    backend's geometry under the other's name and `provenance.json` would say so wrongly, which is
    worse than not starting, because the corpus would look fine.
    """
    name = str(getattr(config.render, "engine", "isaac"))
    if name not in ENGINES:
        raise ValueError(f"unknown render.engine {name!r}; expected one of {sorted(ENGINES)}")

    available, reason = engine_is_available(name)
    if not available:
        raise RuntimeError(f"render.engine is {name!r} but it cannot be used here: {reason}")

    if name == "mujoco":
        from datagen.render.mujoco_engine import MujocoRenderer  # noqa: PLC0415

        return MujocoRenderer(config, headless=headless, **kwargs)
    if name == "none":
        from datagen.render.noengine import NoEngineRenderer  # noqa: PLC0415

        return NoEngineRenderer(config, headless=headless, **kwargs)
    if name == "isaac":
        from datagen.render.isaac import IsaacRenderer  # noqa: PLC0415 (it boots Isaac)

        return IsaacRenderer(config, headless=headless, **kwargs)

    raise ValueError(f"render.engine {name!r} is registered but not constructible")  # pragma: no cover
