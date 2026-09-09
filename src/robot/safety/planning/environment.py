"""The single anchor point for the two external engines the motion stack builds on.

The perceive, grasp and move stack depends on two engines that cannot be plain ``pip``
dependencies. Rather than hiding that behind optional-import guards scattered through
the code, this module makes the dependency explicit, named and probeable in one place:

* cuRobo, the GPU collision-aware trajectory planner, run as a process-isolated
  sidecar in :mod:`.curobo_client` and located on this box through the python
  interpreter of the cuRobo environment.
* Coal or python-fcl, the exact mesh-against-mesh distance engine behind the
  fail-closed self-collision guard in :mod:`.._fcl_self_collision`. Coal is preferred
  and python-fcl is the fallback.

Neither can ever be a hard import:

* cuRobo needs python 3.10, a CUDA build and a ``warp`` version that clashes with the
  one Isaac ships, and it has no wheel, so it can only run out of process in its own
  interpreter.
* Coal has no Windows wheel. Install it from conda-forge and point
  :data:`ENV_COAL_PREFIX` at that environment.

Both therefore stay lazily imported and fail closed. What this module adds is a
discoverable surface: every environment-variable name and default path lives here once,
and the ``probe_*`` helpers report what is wired on this box.
``python -m src.robot.safety.planning --check`` prints that report, so an operator
verifies the anchoring on any machine.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.robot.constants import PLANNING_ENVIRONMENT_LOG_FILE, create_robot_logger
from src.utility.paths import project_root

__all__ = [
    "ENV_CUROBO_PYTHON",
    "ENV_CUROBO_ROBOT",
    "ENV_CUROBO_CUBOID_CACHE",
    "ENV_CUROBO_STDERR",
    "ENV_CUROBO_MAX_ATTEMPTS",
    "ENV_CUROBO_GRAPH_FROM_ATTEMPT",
    "ENV_COAL_PREFIX",
    "COLLISION_MESH_DIR",
    "curobo_python_path",
    "curobo_robot_config",
    "curobo_cuboid_cache",
    "curobo_env_available",
    "collision_mesh_bundle",
    "inject_coal_prefix",
    "import_collision_engine",
    "CuroboStatus",
    "CollisionEngineStatus",
    "PlanningEnvironment",
    "probe_curobo",
    "probe_collision_engine",
    "probe_planning_environment",
]

# Which engine answered is not knowable after the fact: both imports are lazy, both
# have fallbacks, and the fallback is markedly weaker. So the resolution is written
# down, because a guard running on the capsule proxy in silence is the failure this
# prevents.
logger = create_robot_logger("PlanningEnvironment", PLANNING_ENVIRONMENT_LOG_FILE)

# --- cuRobo planner sidecar -----------------------------------------------------------------------
# The client reads the first four. The python 3.10 sidecar ``curobo_planner_server.py``
# reads the last two and cannot import this module, so they are named here to keep every
# knob discoverable in one place.
ENV_CUROBO_PYTHON = "WILLY_CUROBO_PYTHON"              #: interpreter of the cuRobo env
ENV_CUROBO_ROBOT = "WILLY_CUROBO_ROBOT"               #: cuRobo robot descriptor (default ``ur5e.yml``)
ENV_CUROBO_CUBOID_CACHE = "WILLY_CUROBO_CUBOID_CACHE"  #: reserved collision-world cuboid slots
ENV_CUROBO_MESH_CACHE = "WILLY_CUROBO_MESH_CACHE"      #: reserved collision-world mesh slots
ENV_CUROBO_VOXEL_GRID = "WILLY_CUROBO_VOXEL_GRID"      #: live-scene grid, ``x,y,z,voxel`` in metres
ENV_CUROBO_STDERR = "WILLY_CUROBO_STDERR"             #: optional server-stderr log file
ENV_CUROBO_MAX_ATTEMPTS = "WILLY_CUROBO_MAX_ATTEMPTS"  #: sidecar: plan attempts (seed batches)
ENV_CUROBO_GRAPH_FROM_ATTEMPT = "WILLY_CUROBO_GRAPH_FROM_ATTEMPT"  #: sidecar: first graph-seeded attempt

_DEFAULT_CUROBO_ROBOT = "ur5e.yml"
_DEFAULT_CUROBO_CUBOID_CACHE = "16"


def curobo_python_path() -> str:
    """Path to the python of the cuRobo environment, from the variable or the ``ext_deps`` install."""
    return os.environ.get(
        ENV_CUROBO_PYTHON, str(project_root() / "ext_deps" / "curobo_env" / "python.exe")
    )


def curobo_robot_config() -> str:
    """The cuRobo robot descriptor file name the sidecar loads."""
    return os.environ.get(ENV_CUROBO_ROBOT, _DEFAULT_CUROBO_ROBOT)


def curobo_cuboid_cache() -> str:
    """How many collision-world cuboid slots the sidecar reserves at boot."""
    return os.environ.get(ENV_CUROBO_CUBOID_CACHE, _DEFAULT_CUROBO_CUBOID_CACHE)


def curobo_mesh_cache() -> str:
    """Mesh slots the planner reserves at boot. ``"0"`` means the mesh channel is not available.

    Reserving is what makes the channel exist at all: cuRobo allocates its collision storage once,
    when the planner is built, so a mesh sent to a planner that reserved none has nowhere to go.
    """
    return os.environ.get(ENV_CUROBO_MESH_CACHE, "0")


def curobo_voxel_grid() -> str:
    """The live-scene grid as ``x,y,z,voxel`` in metres, or empty for no voxel storage.

    Same reservation rule as the meshes, with one more number: the size and resolution of the grid
    are fixed when the planner starts, because the storage is allocated for exactly that many cells.
    """
    return os.environ.get(ENV_CUROBO_VOXEL_GRID, "")


def curobo_env_available() -> bool:
    """True only where the python of the cuRobo environment exists on this box.

    It is a light build-time check that neither spawns nor JIT-warms the server, which
    lets a caller resolve a ``"curobo"`` request down to blind IK at build time, so the
    per-planner config matches the planner that will run. A deeper boot or JIT failure
    on a present environment is caught later by the driver move-time fallback.
    """
    return Path(curobo_python_path()).exists()


# --- Coal / python-fcl exact-mesh collision engine ------------------------------------------------
ENV_COAL_PREFIX = "WILLY_COAL_PREFIX"  #: conda env prefix that provides Coal (Windows has no wheel)

#: Home of the committed, DH-baked, vertex-validated per-link collision meshes that the
#: Coal self-collision guard and the cuRobo sphere fit share.
COLLISION_MESH_DIR = Path(__file__).resolve().parents[1] / "data"


def collision_mesh_bundle(model: str = "ur5e", variant: str | None = None) -> Path:
    """Path to the mesh bundle for ``model``, or for ``variant`` mounted on ``model``.

    A variant bundle is an arm plus a hand, not a hand. It carries the arm meshes of the robot it
    was baked from and swaps only the three gripper arrays, which is why
    ``_variant_is_for_another_model`` exists and why a mismatch drops the whole cell to the capsule
    proxy rather than only the hand: the arm loses exact-mesh checking too.

    So ``variant`` names the hand and the arm is composed in here. ``robotiq_hande`` finds
    ``robotiq_hande_ur3e_collision_meshes.npz`` on a UR3e and the ur5e file on a UR5e. Without
    that the arm has to be written into the config key, and a cell that changes arms then keeps a
    bundle for the old one. Measured: ``schunk_egu50`` on a ur3e does exactly that today.

    The flat ``{variant}_collision_meshes.npz`` is still found when no per-arm file exists, which
    is what ``schunk_egu50`` is: baked before this, and implicitly a UR5e.
    """
    model = model.lower()
    own = COLLISION_MESH_DIR / f"{model}_collision_meshes.npz"
    if not variant:
        return own
    variant = variant.lower()
    per_arm = COLLISION_MESH_DIR / f"{variant}_{model}_collision_meshes.npz"
    if per_arm.is_file():
        return per_arm
    # The flat name is offered only where its arm can be checked. A flat bundle carries the arm
    # meshes of whatever robot it was baked from, and ``_variant_is_for_another_model`` catches
    # that by comparing one arm link against the own bundle of this model. With no own bundle
    # there is nothing to compare, so that check returns False, and a cell would run the geometry
    # of a different arm under a status of ``ok``.
    #
    # Measured before this line existed: ur10e with ``collision_mesh_variant: schunk_egu50``
    # reported ``ok`` and checked a UR10e against UR5e arm meshes, while the same variant on a
    # ur3e correctly reported ``variant_model_mismatch``. The difference was only that ur3e has a
    # bundle to be compared with. The question is unanswerable exactly when an arm is new, which
    # is when the guard should be least willing to guess.
    if not own.is_file():
        return per_arm  # absent, so the caller reports no_bundle rather than a foreign arm
    return COLLISION_MESH_DIR / f"{variant}_collision_meshes.npz"


def inject_coal_prefix() -> None:
    """Where :data:`ENV_COAL_PREFIX` points at a conda environment, make its Coal importable.

    On Windows the dependency DLLs of the Coal extension live in
    ``<prefix>/Library/bin``, which PATH has not reached for extension modules since
    python 3.8, and the package lives in ``<prefix>/Lib/site-packages``. site-packages
    is appended rather than prepended, so the host interpreter own numpy still wins, and
    Coal works against numpy 1.x and 2.x either way. With the variable unset it falls
    back to the in-repo ``ext_deps/coal_env``, and it does nothing off Windows or where
    neither is present.
    """
    prefix = os.environ.get(ENV_COAL_PREFIX)
    if not prefix:
        default = project_root() / "ext_deps" / "coal_env"
        if not default.is_dir():
            return
        prefix = str(default)
    bindir = os.path.join(prefix, "Library", "bin")
    site = os.path.join(prefix, "Lib", "site-packages")
    if hasattr(os, "add_dll_directory") and os.path.isdir(bindir):
        try:
            os.add_dll_directory(bindir)
        except OSError:
            pass
    if os.path.isdir(site) and site not in sys.path:
        sys.path.append(site)
        # A sys.path mutation that decides whether the exact-mesh guard exists at all,
        # which is worth a line: a missing Coal and a present Coal whose prefix was
        # never injected look identical from the guard side and have different fixes.
        logger.info("injected the Coal prefix %s onto sys.path", prefix)


def import_collision_engine() -> tuple[Any, str] | tuple[None, None]:
    """Import the exact-mesh engine, preferring ``(module, "coal")`` over ``(module, "fcl")``.

    It returns ``(None, None)`` where neither imports, and the caller then falls back to
    the capsule proxy. Coal is tried first; on one failure the conda-environment prefix
    is injected and Coal retried, then python-fcl. This is the single place the
    self-collision guard and the continuous monitor both resolve their engine through.
    """
    for _attempt in (0, 1):
        try:
            import coal  # type: ignore[import-not-found]
            logger.debug("exact-mesh collision engine resolved: coal")
            return coal, "coal"
        except Exception:  # noqa: BLE001 (optional; inject the prefix once, then try python-fcl)
            if _attempt == 0:
                inject_coal_prefix()
    try:
        import fcl  # type: ignore[import-not-found]
        logger.debug("exact-mesh collision engine resolved: python-fcl (Coal was not importable)")
        return fcl, "fcl"
    except Exception:  # noqa: BLE001 (neither engine present -> capsule fallback)
        # Debug rather than warning, because ``_fcl_self_collision.make_backend``
        # already warns about the capsule fallback with its status token, and this
        # helper is called several times per build.
        logger.debug("no exact-mesh collision engine importable (neither coal nor fcl)")
        return None, None


# --- Typed status snapshots (what the --check CLI reports) ----------------------------------------


@dataclass(frozen=True, slots=True)
class CuroboStatus:
    """Whether the cuRobo planner sidecar is wired on this box.

    ``available`` means the environment python exists and nothing more. It does not mean
    the sidecar will plan, and in particular it says nothing about ``robot_config``. The
    sidecar resolves that name against ``get_content_root()/configs/robot/`` inside the
    cuRobo installation, in a separate environment this process cannot introspect
    without spawning it, and globbing the environment root for that directory finds
    nothing. This probe is deliberately spawn-free.

    So a cell configured for a robot whose descriptor was never built, a UR3e with no
    ``ur3e.yml``, passes this check and fails later inside the sidecar. That gap is
    named here rather than papered over: verifying it is a bench step in
    ``docs/runbooks/real_cell_first_pick.md``, the same shape as the arm-model
    cross-check, which is also a human checkpoint because no honest automated version
    exists.
    """

    python_path: str
    available: bool
    robot_config: str

    @property
    def summary(self) -> str:
        state = "AVAILABLE" if self.available else "MISSING"
        return (
            f"cuRobo planner: {state}  (python={self.python_path}, robot={self.robot_config} "
            f"not verified: the descriptor lives in the sidecar env; confirm it was built for this "
            f"robot)"
        )


@dataclass(frozen=True, slots=True)
class CollisionEngineStatus:
    """Which exact-mesh engine is importable, and whether this model mesh bundle ships."""

    engine: str | None  # "coal" (preferred) | "fcl" (fallback) | None (capsule-only)
    mesh_bundle_present: bool
    coal_prefix: str | None
    model: str = "ur5e"  #: which robot bundle was checked, since bundles are per-robot

    @property
    def available(self) -> bool:
        return self.engine is not None and self.mesh_bundle_present

    @property
    def summary(self) -> str:
        engine = self.engine or "none (capsule fallback)"
        mesh = "present" if self.mesh_bundle_present else "MISSING"
        prefix = f", coal_prefix={self.coal_prefix}" if self.coal_prefix else ""
        return (
            f"exact-mesh collision engine: {engine}  ({self.model} mesh bundle {mesh}{prefix})"
        )


@dataclass(frozen=True, slots=True)
class PlanningEnvironment:
    """A snapshot of both external engines, the anchor status for the whole motion stack."""

    curobo: CuroboStatus
    collision: CollisionEngineStatus

    @property
    def fully_anchored(self) -> bool:
        """True only where the cuRobo environment, an exact-mesh engine and its bundle are all present."""
        return self.curobo.available and self.collision.available

    def render(self) -> str:
        """Which external motion engines are present, written for a person.

        ``summary`` on the two status objects keeps its name by convention:
        ``render()`` returns the whole object and a ``summary`` describes a fragment.
        This method composes the two fragments into the whole.
        """
        verdict = "fully anchored" if self.fully_anchored else "partially anchored (degraded fallbacks active)"
        return (
            "Workaholic-Willy motion-stack external engines\n"
            f"  {self.curobo.summary}\n"
            f"  {self.collision.summary}\n"
            f"  => {verdict}"
        )


def probe_curobo(robot_config: str | None = None) -> CuroboStatus:
    """A light presence check of the cuRobo environment, with no spawn and no JIT.

    ``robot_config`` is the descriptor the caller knows will be used, and a caller with
    a configured robot model passes it. Without it this falls back to the variable or
    the default, which is blind to the cell model and reports ``ur5e.yml`` for a UR3e
    cell. A banner naming the wrong robot during a bring-up is worse than no banner at
    all.
    """
    py = curobo_python_path()
    return CuroboStatus(
        python_path=py, available=Path(py).exists(),
        robot_config=robot_config or curobo_robot_config(),
    )


def probe_collision_engine(model: str = "ur5e") -> CollisionEngineStatus:
    """Import-probe the exact-mesh engine and check that this model mesh bundle ships.

    ``model`` matters, because bundles are per-robot as
    ``{model}_collision_meshes.npz``, and a present ur5e bundle says nothing about
    whether a ur3e cell has an exact-mesh self-collision guard. It defaults to ``ur5e``
    for a caller with no model.
    """
    _mod, kind = import_collision_engine()
    return CollisionEngineStatus(
        engine=kind,
        mesh_bundle_present=collision_mesh_bundle(model).exists(),
        coal_prefix=os.environ.get(ENV_COAL_PREFIX),
        model=model,
    )


def probe_planning_environment(
    *, robot_config: str | None = None, kinematics_model: str = "ur5e",
) -> PlanningEnvironment:
    """Probe both engines and return the combined anchor status for one cell.

    Both arguments are optional, which keeps the model-agnostic CLI reading available. A
    caller that knows its cell, such as the sim bootstrap, passes them so the banner
    describes the robot that will run.
    """
    return PlanningEnvironment(
        curobo=probe_curobo(robot_config), collision=probe_collision_engine(kinematics_model),
    )
