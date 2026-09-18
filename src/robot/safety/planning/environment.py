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
from typing import Any, Final

from src.robot.constants import PLANNING_ENVIRONMENT_LOG_FILE, create_robot_logger
from src.utility.paths import project_root

from ._hand_bundle import HAND_PARTS, RECORD_PREFIX, HandBundleRefused, hand_bundle_refusal

__all__ = [
    "ENV_CUROBO_PYTHON",
    "ENV_CUROBO_ROBOT",
    "ENV_CUROBO_CUBOID_CACHE",
    "ENV_CUROBO_MESH_CACHE",
    "ENV_CUROBO_VOXEL_GRID",
    "ENV_CUROBO_STDERR",
    "ENV_CUROBO_MAX_ATTEMPTS",
    "ENV_CUROBO_GRAPH_FROM_ATTEMPT",
    "ENV_COAL_PREFIX",
    "COLLISION_MESH_DIR",
    "curobo_python_path",
    "curobo_robot_config",
    "curobo_cuboid_cache",
    "curobo_mesh_cache",
    "curobo_voxel_grid",
    "curobo_env_available",
    "collision_mesh_bundle",
    "hand_mesh_bundle",
    "compose_collision_meshes",
    "BAKED_SOURCE",
    "DIMENSIONS_SOURCE",
    "HandProvenance",
    "hand_provenance",
    "inject_coal_prefix",
    "import_collision_engine",
    "resolve_collision_engine",
    "CollisionEngineResolution",
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
# The client reads the first six, and a cell's own reservation (``reservation.py``) wins
# over the three slot variables when it has one. The python 3.10 sidecar
# ``curobo_planner_server.py`` reads the last two and cannot import this module, so they
# are named here to keep every knob discoverable in one place.
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


def curobo_cuboid_cache() -> str:
    """How many collision-world cuboid slots the sidecar reserves at boot."""
    return os.environ.get(ENV_CUROBO_CUBOID_CACHE, _DEFAULT_CUROBO_CUBOID_CACHE)


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


def collision_mesh_bundle(model: str = "ur5e") -> Path:
    """Path to the arm bundle for ``model``: its six links, with the Robotiq 2F-85 its asset was baked with.

    No other hand is in here. Another hand is its own bundle, :func:`hand_mesh_bundle`, composed onto
    the arm when the guard loads, through :func:`compose_collision_meshes`. A file per arm and hand
    that carries a copy of the arm, and a flat one that carries whatever arm it was baked on, can
    describe the wrong arm: a ur10e cell with the EGU-50 checked UR10e joint angles against UR5e arm
    meshes and reported ``ok``. A hand bundle carries no arm to get wrong, and which arms it may be
    composed onto is measured, one file per combination.
    """
    return COLLISION_MESH_DIR / f"{model.lower()}_collision_meshes.npz"


#: The arrays of an arm bundle that are the hand rather than the arm, and the prefix of the record
#: keys of a hand bundle, which are never meshes. Both belong to ``_hand_bundle``, which also refuses
#: a hand bundle holding anything but these three parts, or fingers off the model's +Y, before it is
#: composed.
_HAND_PARTS = HAND_PARTS
_HAND_RECORD_PREFIX = RECORD_PREFIX


def hand_writers_sentence(hand: str) -> str:
    """The three ways a hand gets a guard body, as one sentence every refusal and remedy for a missing one reads.

    One owner, so no refusal names a baker that exits 2 for any hand outside its presets.
    """
    return (
        f"write its body from its registry numbers with scripts/grippers/write_hand_from_dimensions.py {hand} "
        f"--inflation-mm <mm> --origin <flange|mounting_face>, from vendor mesh files with "
        f"scripts/grippers/write_hand_from_mesh.py {hand} --gripper-mesh <file> --lfinger-mesh <file> --rfinger-mesh <file> "
        f"--scale-to-mm <mm per unit> --closing <axis> --approach <axis> --binormal <axis> --mount-face-mm <mm> "
        f"--origin <flange|mounting_face> --write, or from one standalone USD with scripts/grippers/bake_gripper_variant.py "
        f"--usd <file> --hand {hand} --bodies <housing,left,right> --closing <axis> --approach <axis> --binormal <axis> "
        f"--mount-face-mm <mm> --origin <flange|mounting_face> --write"
    )


def sphere_map_writer_sentence(hand: str) -> str:
    """How a hand with a body gets the planner map every reader of a missing one names."""
    return f"fit its planner map from its body with scripts/curobo/fit_cover_spheres.py --hand {hand} --write"


def hand_mesh_bundle(hand: str) -> Path:
    """The guard bundle of the hand called ``hand``: its arrays alone, composed onto an arm at load.

    A newly baked hand is written by ``scripts/grippers/bake_gripper_variant.py``. Which arms a hand
    may be composed onto is not recorded in the bundle: it is measured, one file per combination,
    under ``planning/robot/evidence``.
    """
    return COLLISION_MESH_DIR / f"{hand.lower()}_hand_meshes.npz"


#: What a hand bundle records about itself. A bake writes no source key, so its absence is the
#: answer for every hand that was scanned. No record names the arms a hand was proven on: admission
#: is a measurement, and it lives in the evidence files beside the sphere maps, where a pairing
#: nobody measured has no file.
_SOURCE_KEY = "hand__source"
_INFLATION_KEY = "hand__inflation_mm"

#: The two sources a hand's geometry can have. They have equal standing: a customer who describes
#: their gripper in ``config/grippers/<name>.yaml`` gets a bundle the planner and the guard read like
#: any other. What separates them is the inflation a bundle written from dimensions records, and that
#: is why both are named rather than assumed.
BAKED_SOURCE: Final = "bake"
DIMENSIONS_SOURCE: Final = "dimensions"


@dataclass(frozen=True, slots=True)
class HandProvenance:
    """Where the geometry of one hand came from, read off the hand's own bundle.

    An envelope built from five declared numbers is not the hand. Measured against the two shipped
    hands that carry both a bundle and a full set of dimensions, at each hand's own declared grasp
    centre, a bare envelope leaves the Hand-E 5.73 mm and the EGU-50 9.50 mm outside it. So a bundle
    written from dimensions carries the inflation its writer was told to use, and every reader that
    models a hand can say which of the two it holds. Nothing refuses an envelope; what would be
    wrong is a cell that cannot tell.
    """

    #: The registry name of the hand, as it was asked for.
    hand: str
    #: :data:`BAKED_SOURCE` or :data:`DIMENSIONS_SOURCE`.
    source: str
    #: How far every declared box grew past the measurements, or ``None`` for a hand that was scanned.
    inflation_mm: float | None

    @property
    def from_dimensions(self) -> bool:
        """Whether this hand is an envelope built from its registry block rather than a scan."""
        return self.source == DIMENSIONS_SOURCE

    def render(self) -> str:
        if not self.from_dimensions:
            return f"{self.hand}: baked from its own asset"
        inflated = "no stated inflation" if self.inflation_mm is None else f"inflated {self.inflation_mm:g} mm"
        return f"{self.hand}: an envelope from its declared dimensions, {inflated}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand": self.hand,
            "source": self.source,
            "inflation_mm": self.inflation_mm,
        }


def hand_provenance(hand: str, mesh_dir: str | Path | None = None) -> HandProvenance | None:
    """What ``hand``'s bundle says about itself, or ``None`` where there is no bundle of its own to ask.

    ``None`` is a real answer and not a failure: a hand nobody has written a bundle for has no file
    to ask. A bundle that cannot be read answers ``None`` as well, because a file nobody can open
    proves nothing about where its numbers came from.
    """
    import numpy as np

    folder = Path(mesh_dir) if mesh_dir else COLLISION_MESH_DIR
    path = folder / f"{hand.lower()}_hand_meshes.npz"
    try:
        with np.load(path, allow_pickle=True) as data:
            held = {key: np.asarray(data[key]).reshape(-1) for key in data.files if key.startswith(_HAND_RECORD_PREFIX)}
    except (OSError, ValueError):
        # A path that is not there, and a file that is not a readable npz. Every record below is
        # written as a one element array by both writers, so it is flattened rather than converted:
        # float() on a shape (1,) array raises, and str() on one spells the brackets into the sentence.
        return None
    source = str(held[_SOURCE_KEY][0]) if _SOURCE_KEY in held and held[_SOURCE_KEY].size else BAKED_SOURCE
    inflation = float(held[_INFLATION_KEY][0]) if _INFLATION_KEY in held and held[_INFLATION_KEY].size else None
    return HandProvenance(hand=hand, source=source, inflation_mm=inflation)


def compose_collision_meshes(model: str, hand: str | None, mesh_dir: str | Path | None = None) -> dict:
    """The arrays the exact-mesh guard reads for ``hand`` on ``model``, composed at load.

    The arm's own bundle without its hand arrays, plus every mesh array of the hand's bundle. A
    ``hand`` of ``None`` gives the arm bundle as it is. The record keys of a hand bundle never reach
    the result. ``mesh_dir`` reads both files from a folder other than the committed one. A hand
    bundle that is not exactly the three parts at frame 6 with its fingers along the model's +Y
    raises :class:`HandBundleRefused` naming what is wrong (``_hand_bundle.hand_bundle_refusal``).
    """
    import numpy as np

    folder = Path(mesh_dir) if mesh_dir else COLLISION_MESH_DIR
    with np.load(folder / f"{model.lower()}_collision_meshes.npz") as data:
        arrays = {key: data[key] for key in data.files}
    if hand is None:
        return arrays
    composed = {key: value for key, value in arrays.items() if key.split("__")[0] not in _HAND_PARTS}
    path = folder / f"{hand.lower()}_hand_meshes.npz"
    with np.load(path, allow_pickle=True) as data:
        held = {key: data[key] for key in data.files}
    refusal = hand_bundle_refusal(held, name=path.name)
    if refusal is not None:
        raise HandBundleRefused(refusal)
    composed.update({key: value for key, value in held.items() if not key.startswith(_HAND_RECORD_PREFIX)})
    return composed


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


@dataclass(frozen=True, slots=True)
class CollisionEngineResolution:
    """Which exact-mesh engine answered, and what the engines ahead of it said when they did not.

    The reasons are kept because throwing them away produced a false green. With an
    application-control policy refusing this repository's own
    ``ext_deps/coal_env/Library/bin/coal.dll``,
    ``python -m src.robot.safety.planning --doctor`` exited 0 with ``policy_blocked=false``
    and reported ``[ok] exact-mesh collision engine: fcl 0.7.0.11``. The resolution below
    caught Coal's exception and returned python-fcl, so the doctor's blocked branch could
    never see a refusal and was unreachable from the command line.

    The guard wants that substitution and does not care why, and
    :func:`import_collision_engine` still hands it exactly what it always did. The doctor
    is asked the opposite question, which is what refused and what the operator does about
    it, and it cannot answer from a value discarded one frame down.
    """

    module: Any | None
    backend: str | None
    #: Why Coal, the preferred engine, is not the one in ``backend``. Empty when it is.
    coal_error: str = ""
    #: Why python-fcl did not carry it either. Empty unless the fallback was reached and failed.
    fcl_error: str = ""


def resolve_collision_engine() -> CollisionEngineResolution:
    """Resolve the exact-mesh engine, keeping what each candidate said when it refused.

    Coal is tried first; on one failure the conda-environment prefix is injected and Coal
    retried, then python-fcl. This is the single place the self-collision guard and the
    continuous monitor both resolve their engine through, and the doctor probes through it
    too, so what an operator is told about the engine and what the guard actually loaded
    cannot part.
    """
    coal_error = ""
    for _attempt in (0, 1):
        try:
            import coal  # type: ignore[import-not-found]
            logger.debug("exact-mesh collision engine resolved: coal")
            return CollisionEngineResolution(coal, "coal")
        except Exception as exc:  # noqa: BLE001 (optional; inject the prefix once, then try python-fcl)
            coal_error = f"{type(exc).__name__}: {exc}"
            if _attempt == 0:
                inject_coal_prefix()
    try:
        import fcl  # type: ignore[import-not-found]
        logger.debug("exact-mesh collision engine resolved: python-fcl (Coal was not importable)")
        return CollisionEngineResolution(fcl, "fcl", coal_error=coal_error)
    except Exception as exc:  # noqa: BLE001 (neither engine present -> capsule fallback)
        # Debug rather than warning, because ``_fcl_self_collision.make_backend``
        # already warns about the capsule fallback with its status token, and this
        # helper is called several times per build.
        logger.debug("no exact-mesh collision engine importable (neither coal nor fcl)")
        return CollisionEngineResolution(
            None, None, coal_error=coal_error, fcl_error=f"{type(exc).__name__}: {exc}"
        )


def import_collision_engine() -> tuple[Any, str] | tuple[None, None]:
    """Import the exact-mesh engine, preferring ``(module, "coal")`` over ``(module, "fcl")``.

    It returns ``(None, None)`` where neither imports, and the caller then falls back to
    the capsule proxy. A view over :func:`resolve_collision_engine` for the callers that
    need only the pair.
    """
    resolution = resolve_collision_engine()
    if resolution.module is None or resolution.backend is None:
        return None, None
    return resolution.module, resolution.backend


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
