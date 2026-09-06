"""Config access for the Isaac sim validation cell.

The sim cell is the ``sim`` profile of the standard ``config`` tree: the ``*.sim.yaml`` overlays
deep-merge onto the production YAMLs (``robot.sim.yaml``, ``camera/hand_eye.sim.yaml``,
``models/{object,segmenting}.sim.yaml``). Scene, prim and calibration constants come from that
tree, not from this package. It reads ``cfg.robot.sim`` (driver and scene authoring),
``cfg.camera.hand_eye`` (calibration tuning), ``cfg.robot.calibration``,
``cfg.robot.workspace_limits`` and ``cfg.models.*``. This module is the bridge: activate the
``sim`` profile, load the tree, and convert ``cfg.robot.sim`` into the driver-side
:class:`SimRobotConfig` (reusing :func:`src.robot.execution.runtime_pick.build_sim_driver_config`).

Imports no ``isaacsim``, so it stays safe on macOS and CI.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import load_config
from src.config.loader import active_profile, join_profiles, set_active_profile
from src.robot.execution.runtime_pick import build_sim_driver_config

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.app import AppConfig
    from src.config.schema.robot import RobotConfig
    from src.robot.drivers.sim.config import SimRobotConfig

__all__ = [
    "DEFAULT_SIM_DATA_DIR",
    "SIM_DEFAULT_ROBOT_MODEL",
    "SIM_PROFILE",
    "load_sim_config",
    "require_robot",
    "sim_driver_config",
    "sim_profile_chain",
    "sim_safety_preflight",
]

#: The config profile the Isaac cell runs under. ``WILLY_PROFILE=sim`` makes the loader deep-merge
#: every ``*.sim.yaml`` overlay onto its base YAML (see ``src/config/loader.py``).
SIM_PROFILE = "sim"

#: The robot the ``sim`` layer alone describes. Selecting this model adds no second layer, so the
#: default cell stays a plain ``WILLY_PROFILE=sim`` load and is byte-identical.
SIM_DEFAULT_ROBOT_MODEL = "ur5e"

# From ``src/willy_sim/config.py``, ``parents[2]`` is the repository root, so this resolves to
# ``config/`` beside ``src/``. The sim cell loads the standard production tree and lets the `sim`
# profile overlay it; it has no config tree of its own.
DEFAULT_SIM_DATA_DIR = Path(__file__).resolve().parents[2] / "config"


def sim_profile_chain(
    robot_model: str | None = None, extra_profiles: "Sequence[str] | None" = None,
) -> str:
    """The profile chain for a sim cell driving ``robot_model`` (``"sim"`` or ``"sim,ur3e,tiltcam"``).

    A sim cell is several independent config dimensions: "this is the Isaac cell" (the ``sim``
    layer: detector dtypes, scene authoring, sim-specific safety thresholds), "this cell drives a
    UR3e" (reach-dependent geometry, the model's kinematics and cuRobo config), and optional rig
    layers such as ``tiltcam`` (the real cell's tilted D435 pair). Chaining states each dimension
    exactly once, so a measured value is never copied per combination and one dimension can be
    changed alone, which is what makes a pick-rate difference interpretable.
    """
    layers = [SIM_PROFILE]
    if robot_model and robot_model != SIM_DEFAULT_ROBOT_MODEL:
        layers.append(robot_model)
    layers.extend(extra_profiles or ())
    return join_profiles(*layers)


def load_sim_config(
    data_dir: str | Path | None = None,
    *,
    robot_model: str | None = None,
    extra_profiles: "Sequence[str] | None" = None,
) -> "AppConfig":
    """Load and validate the sim config: the ``config`` tree under the ``sim`` profile.

    Forces ``WILLY_PROFILE=sim`` for the duration of the load and restores whatever was set
    before, so every sim runner gets the ``*.sim.yaml`` overlays merged in without the caller
    flipping the switch. Returns the frozen :class:`AppConfig`; read ``cfg.robot.sim`` and the
    rest off it. ``load_config`` caches by ``(root, profile)``, so the sim config and any
    production config coexist in the cache (call ``src.config.reload_config()`` after editing
    YAML).

    ``robot_model`` selects a second profile layer on top of ``sim``: ``"ur3e"`` gives the chain
    ``"sim,ur3e"`` and merges every ``*.ur3e.yaml``, so a cell driving a different UR gets that
    robot's reach-dependent geometry without forking the sim overlays. The default model adds no
    layer, so the ordinary sim load is unchanged. ``extra_profiles`` appends further layers on
    top, for example ``("tiltcam",)`` for the real cell's tilted D435 rig.

    A custom ``data_dir`` (the ``--data-dir`` runner flag) must be a tree that carries the sim
    overlays (``*.sim.yaml``); the ``sim`` profile fail-closes with a clear ``ConfigError``
    otherwise. That refusal is deliberate: a tree without the sim overlays runs the fp16
    production detectors through the sim cell, the configuration that drops small-object recall.
    """
    root = str(data_dir) if data_dir is not None else str(DEFAULT_SIM_DATA_DIR)
    previous = active_profile()
    set_active_profile(sim_profile_chain(robot_model, extra_profiles))
    try:
        return load_config(root)
    finally:
        set_active_profile(previous)


def require_robot(cfg: "AppConfig") -> "RobotConfig":
    """Narrow ``cfg.robot`` to a non-``None`` :class:`RobotConfig` for the sim cell.

    The sim config tree always carries a ``robot`` block (``robot.vendor == 'sim'``), so the sim
    runners can read ``robot.sim``, ``.gripper``, ``.calibration`` and ``.workspace_limits``
    without the ``RobotConfig | None`` union poisoning every access. Raises the same
    ``ValueError`` that :func:`sim_driver_config` and :func:`sim_safety_preflight` raise on the
    same precondition. On a sim run the block is present and the raise is unreachable.
    """
    if cfg.robot is None:
        raise ValueError("sim config has no robot block (expected robot.vendor == 'sim').")
    return cfg.robot


def sim_driver_config(cfg: "AppConfig", *, headless: bool | None = None) -> "SimRobotConfig":
    """Build the driver-side :class:`SimRobotConfig` from a loaded sim ``cfg``.

    ``headless`` overrides the YAML value (e.g. a ``--gui`` CLI flag) without mutating the frozen
    config. All other driver fields come straight from ``cfg.robot.sim`` via the shared converter.
    """
    _robot = require_robot(cfg)
    driver = build_sim_driver_config(_robot.sim, _robot.gripper.tool_frame)
    if headless is not None and headless != driver.headless:
        driver = replace(driver, headless=headless)
    return driver


def sim_safety_preflight(cfg: "AppConfig"):
    """Build the vendor-neutral fail-closed ``SafetyPreflight`` for the sim cell.

    Mirrors how the UR driver wires safety: the same six guards in order (workspace, joint-limit,
    IK-quality, self-collision, payload, motion-continuity) from ``cfg.robot.safety`` and
    ``cfg.robot.workspace_limits``. Pass the result to ``IsaacRobotArm(safety_preflight=...)`` so
    the sim pick path runs through the guards instead of bypassing them.
    """
    from src.robot.safety.preflight import SafetyPreflight

    robot = require_robot(cfg)
    return SafetyPreflight.from_safety_config(robot.safety, robot.workspace_limits)
