"""Install Workaholic-Willy's native dependencies into this repository, from nothing, on Windows or Linux (P0h h3).

    python scripts/ext_deps/install.py                     # everything this platform installs, then the doctor
    python scripts/ext_deps/install.py --component coal    # coal, curobo, or (Linux) main
    python scripts/ext_deps/install.py --plan              # print every external command it would run, run nothing
    python scripts/ext_deps/install.py --plan --from-nothing --platform linux-64   # another platform, from nothing

One command, from nothing to a verified motion stack. It bootstraps its own micromamba, creates the environments from
the committed lock files, installs cuRobo's pip packages at the versions its pip lock holds (P0h h4: the conda lock pinned
only the conda half, and whatever cuRobo pulled with pip floated), clones and builds cuRobo with BOTH kernel backends, fetches the pinned UR meshes, builds a
descriptor for every arm this repository can build correctly, and finishes by running the doctor, so the exit code
means "this box can plan", not "the downloads finished". With no main environment to run the doctor with, nothing
said the box can plan, so the install is refused with exit 1 rather than reported green (P0h h3d; install.ps1 warned
and exited 0 there). Standard library only, runnable by any Python 3.10 or newer
before anything else exists.

On Windows it runs exactly the commands ``install.ps1`` ran before it became a shim for this file, in the same order,
and fails, refuses and tolerates where that script did (``tests/data/install_plan/win-64.txt`` is the list, derived
from ``install.ps1`` before the switch; ``--plan --from-nothing --platform win-64`` must print it). The Windows ``.venv``
is never touched: it is the owner's, shared with another checkout. Windows stays a full install path of its own, which a
cell that drives only a UR needs and nothing else (the owner, 2026-09-26).

On Linux the same steps take the platform's layout (``bin/python``, ``<name>.linux-64.lock``, micromamba's single
static binary, since a fresh distribution may have no bzip2), the compiled kernels are built by
``build_compiled_backend.sh``, the .bat's twin, with the environment's compilers, and ``main`` makes the repository's
main environment, ``.venv``, a micromamba environment with Python 3.11 holding ``requirements.txt`` (owner decision 6).
The system libraries it cannot install without root, ``libportaudio2`` for the microphone and ``libusb-1.0-0`` for a
RealSense camera, are named with their ``apt`` line when missing.

Nothing outside ``ext_deps/`` is touched, and on Linux ``.venv``: no system conda, no PATH or registry changes, no
administrator rights, and the package caches live inside ``ext_deps/``. Deleting those folders undoes the install.
"""

from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_EXT_DEPS = REPO / "ext_deps"

#: The cuRobo revision. 8e734f3 (2026-07-22) carries 82ea96e, the cuda.core to pybind fallback that returns instead of
#: raising, and itself the mesh SDF gradient sign for points outside the surface. The newest tag (v0.8.0) is older.
CUROBO_REV = "8e734f3"
CUROBO_URL = "https://github.com/NVlabs/curobo.git"
TORCH_SPEC = "torch==2.7.0"
#: cu128 is Blackwell (sm_120).
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
#: Pinned DOWN, not up: [cu12] pulls the newest cuda-core, and on 2026-08-30 1.1.1 was refused by Smart App Control
#: (``_resource_handles.cp310-win_amd64.pyd``) while 1.1.0 loads. Reach, not recency, decides that verdict, so the
#: previous build is the one with reputation (docs/code-integrity.md). If 1.1.0 is refused in turn, step back again.
CUDA_CORE_PIN = "cuda-core==1.1.0"
#: The same reason, measured 2026-09-24: warp-lang 1.17.0's warp.dll was refused at load (CodeIntegrity event 3033),
#: which stops the sidecar before it is ready, while 1.16.0, 1.15.0 and 1.14.0 loaded; 1.16.0 sees an sm_120 GPU.
WARP_PIN = "warp-lang==1.16.0"
#: The Python of the Linux main environment: the interpreter the Windows ``.venv`` runs.
MAIN_PYTHON = "3.11"
#: The four pybind extensions cuRobo's compiled backend consists of.
PYBIND_MODULES = ("geometry", "kinematics", "optimization", "trajectory")
#: The exit code of a box that lacks a program the install cannot bootstrap, as install.ps1 exited.
EXIT_PREREQUISITE = 2

_PLATFORMS = ("win-64", "linux-64")


class InstallRefused(RuntimeError):
    """A step failed or cannot start; the message says which and what to do."""


@dataclass(frozen=True)
class Step:
    """One thing the install does, in order.

    ``env``, ``require``, ``download``, ``remove``, ``warn`` and ``system`` are the install's own; ``run`` is an external
    command; ``need`` is a check between two of them (a file that has to be there), which the plan does not print
    because it runs nothing.
    """

    kind: str
    argv: tuple[str, ...] = ()
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    source: str = ""
    target: Path | None = None
    #: What a failure of this step means, as the refusal says it.
    fails: str = ""
    #: The exit codes of a ``run`` that are an outcome and not a failure (the compiled backend's "no toolchain").
    emergency: tuple[int, ...] = ()
    #: A ``run`` whose exit code install.ps1 never looked at.
    tolerate: bool = False
    #: The ``run`` that is the doctor, whose exit code is the install's.
    doctor: bool = False


@dataclass(frozen=True)
class Layout:
    """Where everything lies for one platform. ``fresh`` plans as though nothing were there yet."""

    platform: str
    ext_deps: Path
    fresh: bool = False

    @property
    def windows(self) -> bool:
        return self.platform == "win-64"

    def exists(self, path: Path) -> bool:
        return not self.fresh and path.exists()

    def env_python(self, name: str) -> Path:
        prefix = self.ext_deps / name
        return prefix / "python.exe" if self.windows else prefix / "bin" / "python"

    @property
    def mamba(self) -> Path:
        root = self.ext_deps / "micromamba"
        return root / "Library" / "bin" / "micromamba.exe" if self.windows else root / "bin" / "micromamba"

    @property
    def mamba_root(self) -> Path:
        return self.ext_deps / "micromamba" / "root"

    @property
    def source(self) -> Path:
        return self.ext_deps / "curobo"

    def lock(self, name: str) -> Path:
        return self.ext_deps / "locks" / f"{name}.{self.platform}.lock"

    def pip_lock(self, name: str) -> Path:
        """The pip half of environment ``name``: every package pip installed there, at its version (P0h h4)."""
        return self.ext_deps / "locks" / f"{name}.{self.platform}.pip.lock"

    @property
    def host_python(self) -> Path:
        venv = REPO / ".venv"
        return venv / "Scripts" / "python.exe" if self.windows else venv / "bin" / "python"

    @property
    def pybind_built(self) -> tuple[Path, ...]:
        folder = self.source / "curobo" / "_src" / "curobolib" / "backends" / "pybind"
        suffix = ".cp310-win_amd64.pyd" if self.windows else ".cpython-310-x86_64-linux-gnu.so"
        return tuple(folder / f"{name}{suffix}" for name in PYBIND_MODULES)


def buildable_arms() -> tuple[str, ...]:
    """Every arm this repository can build a descriptor for CORRECTLY, in natural order (ur3 before ur10).

    An arm the description registry holds (``config/arms/<arm>.yaml``) that has a committed collision
    bundle and a sphere map fitted to it: the rule ``install.ps1`` wrote as a hand-kept list of seven, which is these
    seven today (ur20 has no bundle yet). A descriptor carries the retract judged on the exact meshes, so an arm with no
    bundle has no row and its build would be refused.
    """
    arms = REPO / "config" / "arms"
    bundles = REPO / "src" / "robot" / "safety" / "data"
    maps = REPO / "src" / "robot" / "safety" / "planning" / "robot"
    found = [path.stem for path in arms.glob("*.yaml")
             if (bundles / f"{path.stem}_collision_meshes.npz").is_file()
             and (maps / f"{path.stem}_arm_spheres.yml").is_file()]
    return tuple(sorted(found, key=_natural))


def _natural(name: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def plan(layout: Layout, *, component: str, clean: bool = False, skip_compiled: bool = False,
         ssl_no_revoke: bool = False) -> list[Step]:
    """Every step of an install on ``layout``, given what is already there, in the order it runs."""
    steps: list[Step] = [
        Step("env", env={"MAMBA_ROOT_PREFIX": str(layout.mamba_root)}),
        Step("env", env={"PIP_CACHE_DIR": str(layout.ext_deps / "pip-cache")}),
        Step("require", argv=("git",)),
    ]
    if layout.windows:
        steps.append(Step("require", argv=("tar",)))
    if not layout.exists(layout.mamba):
        if layout.windows:
            archive = layout.ext_deps / "micromamba" / "micromamba.tar.bz2"
            steps.append(Step("download", source="https://micro.mamba.pm/api/micromamba/win-64/latest", target=archive))
            # install.ps1 never looked at tar's exit code, and refused when the exe was not there afterwards.
            steps.append(Step("run", ("tar", "-xf", str(archive), "-C", str(layout.ext_deps / "micromamba")),
                              tolerate=True))
            steps.append(Step("need", target=layout.mamba, fails=f"micromamba bootstrap failed -- no exe at {layout.mamba}"))
        else:
            steps.append(Step("download", target=layout.mamba,
                              source="https://github.com/mamba-org/micromamba-releases/releases/latest/download/"
                                     "micromamba-linux-64"))
    mamba_args = ("-c", "conda-forge", *(("--ssl-no-revoke",) if ssl_no_revoke else ()))
    if component in ("all", "coal"):
        steps += _environment(layout, "coal_env", mamba_args, clean)
    if component in ("all", "curobo"):
        steps += _curobo(layout, mamba_args, clean, skip_compiled)
    if component in ("all", "main") and not layout.windows:
        steps += _main_environment(layout, mamba_args, clean)
    steps.append(Step("run", (str(layout.host_python), "-m", "src.robot.safety.planning", "--doctor"),
                      cwd=REPO, doctor=True))
    return steps


def _environment(layout: Layout, name: str, mamba_args: tuple[str, ...], clean: bool) -> list[Step]:
    """One environment from its lock: the lock is checked before anything is removed, as install.ps1 did."""
    prefix = layout.ext_deps / name
    lock = layout.lock(name)
    steps = [Step("need", target=lock, fails=f"no lockfile at {lock}")]
    if clean and layout.exists(prefix):
        steps.append(Step("remove", target=prefix))
    steps.append(Step("run", (str(layout.mamba), "create", "-y", "-p", str(prefix), "--file", str(lock), *mamba_args),
                      fails=f"micromamba create failed for {name}. If the failure looks like a TLS or download error "
                            "on a corporate network, retry with --ssl-no-revoke"))
    steps.append(Step("need", target=prefix, fails=f"micromamba reported success but {prefix} does not exist"))
    return steps


def _curobo(layout: Layout, mamba_args: tuple[str, ...], clean: bool, skip_compiled: bool) -> list[Step]:
    python_path = layout.env_python("curobo_env")
    python = str(python_path)
    source = layout.source
    steps = _environment(layout, "curobo_env", mamba_args, clean)
    steps.append(Step("need", target=python_path, fails=f"no interpreter at {python} after creating the env"))
    steps.append(Step("run", (python, "-m", "pip", "install", "--quiet", TORCH_SPEC, "--index-url", TORCH_INDEX),
                      fails="torch install failed"))
    # Everything cuRobo's [cu12] pulls with pip, at the versions that run on the box the lock was written on, so that
    # the editable install below finds each requirement met and floats nothing; torch's +cu128 build needs its index.
    pip_lock = layout.pip_lock("curobo_env")
    steps.append(Step("need", target=pip_lock, fails=f"no pip lockfile at {pip_lock}"))
    steps.append(Step("run", (python, "-m", "pip", "install", "--quiet", "-r", str(pip_lock), "--extra-index-url",
                              TORCH_INDEX), fails="installing the pinned pip packages failed"))
    if clean and layout.exists(source):
        steps.append(Step("remove", target=source))
    if clean or not layout.exists(source):
        # No `git lfs install`: it writes filter.lfs.* into the user's global .gitconfig, which outlives ext_deps/,
        # and the pinned revision declares no LFS filters anyway.
        steps.append(Step("run", ("git", "clone", CUROBO_URL, str(source)), fails="git clone failed"))
    # install.ps1 never looked at the fetch's exit code: an offline re-run over a clone that holds the pinned revision
    # checks it out and goes on.
    steps.append(Step("run", ("git", "fetch", "--quiet", "origin"), cwd=source, tolerate=True))
    steps.append(Step("run", ("git", "checkout", "--quiet", CUROBO_REV), cwd=source,
                      fails=f"git checkout {CUROBO_REV} failed"))
    steps.append(Step("run", (python, "-m", "pip", "install", "--quiet", "-e", f"{source}[cu12]",
                              "--no-build-isolation"), fails="cuRobo [cu12] install failed"))
    steps.append(Step("run", (python, "-m", "pip", "install", "--quiet", CUDA_CORE_PIN), fails="cuda-core pin failed"))
    steps.append(Step("run", (python, "-m", "pip", "install", "--quiet", WARP_PIN), fails="warp-lang pin failed"))
    built = all(layout.exists(path) for path in layout.pybind_built)
    if skip_compiled:
        steps.append(Step("warn", source="skipping the compiled backend on request: the planner has ONE kernel "
                                         "backend, the exact configuration that failed on 2026-08-10. Build it when a "
                                         "C++ toolchain is available"))
    elif not (built and not clean):
        steps.append(_compiled_backend(layout))
    steps.append(Step("run", (python, str(REPO / "scripts" / "arms" / "fetch.py")),
                      fails="fetching the pinned UR meshes failed"))
    for arm in buildable_arms():
        steps.append(Step("run", (python, str(REPO / "scripts" / "arms" / "descriptor.py"), arm, "--urdf-from", "ur"),
                          fails=f"building the willy_{arm} descriptor failed"))
    return steps


def _compiled_backend(layout: Layout) -> Step:
    """cuRobo's compiled (pybind) kernel backend: the NORMAL case, which a missing toolchain turns into an emergency.

    On Windows ``build_compiled_backend.bat`` finds MSVC and defuses its traps, and its exit 2 means it never started
    (no toolchain, which needs administrator rights to install), so the install continues and the doctor decides. On
    Linux its twin ``build_compiled_backend.sh`` builds with the environment's own nvcc and GCC, which the lock carries,
    so there is no missing toolchain to tolerate and every failure fails (P0h h5 measured what the bare ``setup.py``
    lacks there). Each script finds the environment and the source beside itself, so an install into another
    ``ext_deps`` names them (``CUROBO_ENV``, ``CUROBO_SRC``, which both read).
    """
    env = {} if layout.ext_deps == DEFAULT_EXT_DEPS else {
        "CUROBO_ENV": str(layout.ext_deps / "curobo_env"), "CUROBO_SRC": str(layout.source)}
    if layout.windows:
        bat = REPO / "scripts" / "curobo" / "build_compiled_backend.bat"
        return Step("run", ("cmd", "/c", str(bat)), env=env, fails="compiled backend build failed", emergency=(2,))
    script = REPO / "scripts" / "curobo" / "build_compiled_backend.sh"
    return Step("run", ("bash", str(script)), env=env, fails="compiled backend build failed")


def _main_environment(layout: Layout, mamba_args: tuple[str, ...], clean: bool) -> list[Step]:
    """Linux only: the repository's main environment at ``.venv``, Python 3.11 from conda-forge, ``requirements.txt``."""
    prefix = REPO / ".venv"
    python = str(layout.host_python)
    steps = []
    if clean and layout.exists(prefix):
        steps.append(Step("remove", target=prefix))
    steps.append(Step("run", (str(layout.mamba), "create", "-y", "-p", str(prefix), f"python={MAIN_PYTHON}",
                              *mamba_args), fails="micromamba create failed for the main environment"))
    steps.append(Step("run", (python, "-m", "pip", "install", "--quiet", "-r", str(REPO / "requirements.txt")),
                      cwd=REPO, fails="installing requirements.txt into the main environment failed"))
    steps.append(Step("run", (python, "-m", "pip", "install", "--quiet", "-e", str(REPO), "--no-deps"), cwd=REPO,
                      fails="installing this repository into the main environment failed"))
    steps.append(Step("system", argv=("portaudio",),
                      source="the microphone needs libportaudio2, which only root can install: "
                             "sudo apt install libportaudio2"))
    # Measured on the first Linux install (P0h h6): pyrealsense2's wheel imports only with libusb-1.0 on the system.
    steps.append(Step("system", argv=("usb-1.0",),
                      source="an Intel RealSense camera needs libusb-1.0, which only root can install: "
                             "sudo apt install libusb-1.0-0"))
    return steps


def pip_lock_lines(inspected: dict[str, Any]) -> list[str]:
    """The lines of a pip lock from ``pip inspect --local`` of an environment: every distribution pip installed there,
    ``name==version``, sorted without case. An editable install (cuRobo itself) and what conda installed (the conda
    lock pins those) are left out."""
    lines = []
    for dist in inspected.get("installed", []):
        if str(dist.get("installer") or "").strip() != "pip":
            continue
        if (dist.get("direct_url") or {}).get("dir_info", {}).get("editable"):
            continue
        meta = dist["metadata"]
        lines.append(f"{meta['name']}=={meta['version']}")
    return sorted(lines, key=str.lower)


def write_pip_lock(layout: Layout, name: str = "curobo_env") -> Path:
    """Write environment ``name``'s pip lock from what is installed in it now, and return where."""
    python = layout.env_python(name)
    if not python.is_file():
        raise InstallRefused(f"no interpreter at {python}: install the environment before its pip lock is written")
    done = subprocess.run([str(python), "-m", "pip", "inspect", "--local"], capture_output=True, check=False,
                          env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    if done.returncode != 0:
        raise InstallRefused(f"pip inspect failed in {name} (exit {done.returncode})")
    lines = pip_lock_lines(json.loads(done.stdout.decode("utf-8", errors="replace")))
    target = layout.pip_lock(name)
    header = [f"# The pip half of {name} on {layout.platform}: every package pip installed there, at the version that",
              "# runs, written by `python scripts/ext_deps/install.py --write-pip-lock` from `pip inspect --local`.",
              "# An editable install (cuRobo itself) and what conda installed (its lock pins those) are left out."]
    target.write_text("\n".join([*header, *lines]) + "\n", encoding="utf-8", newline="\n")
    return target


def render(step: Step, layout: Layout) -> str | None:
    """One step as a line of the plan: paths under ``<repo>`` and ``<ext_deps>``, with forward slashes. ``None`` for a
    check between two steps, which runs nothing."""

    def said(text: str) -> str:
        spelled = text.replace("\\", "/")
        for path, name in ((layout.ext_deps, "<ext_deps>"), (REPO, "<repo>")):
            spelled = spelled.replace(str(path).replace("\\", "/"), name)
        return spelled

    if step.kind == "need":
        return None
    if step.kind == "env":
        (key, value), = step.env.items()
        return f"env {key}={said(value)}"
    if step.kind == "require":
        return f"require {step.argv[0]}"
    if step.kind == "download":
        return f"download {step.source} -> {said(str(step.target))}"
    if step.kind == "remove":
        return f"remove {said(str(step.target))}"
    if step.kind == "warn":
        return f"warn {step.source}"
    if step.kind == "system":
        return f"system {step.argv[0]}: {step.source}"
    argv = list(step.argv)
    command = f'cmd /c "{said(argv[2])}"' if argv[:2] == ["cmd", "/c"] else " ".join(said(part) for part in argv)
    prefix = f"in {said(str(step.cwd))}: " if step.cwd is not None else ""
    env = "".join(f"{key}={said(value)} " for key, value in step.env.items())
    return f"{prefix}run {env}{command}"


@dataclass(frozen=True)
class Outcome:
    """How an install ended: its exit code, and whether the doctor said it."""

    code: int
    doctor_ran: bool = False


def execute(steps: list[Step], layout: Layout) -> Outcome:
    """Run ``steps``. Every missing prerequisite is named before anything is downloaded, as install.ps1 did."""
    missing = [step.argv[0] for step in steps if step.kind == "require" and shutil.which(step.argv[0]) is None]
    if missing:
        print(f"missing prerequisite(s): {', '.join(missing)}", file=sys.stderr, flush=True)
        for tool in missing:
            print(f"  {tool}: {_missing_tool(tool, layout)}", file=sys.stderr, flush=True)
        return Outcome(EXIT_PREREQUISITE)
    env = dict(os.environ)
    for step in steps:
        if step.kind == "env":
            env.update(step.env)
        elif step.kind == "need":
            assert step.target is not None
            if not step.target.exists():
                raise InstallRefused(step.fails)
        elif step.kind == "download":
            _download(step, layout)
        elif step.kind == "remove":
            assert step.target is not None
            print(f"    removing {step.target}", flush=True)
            _remove(step.target)
        elif step.kind == "warn":
            print(f"WARNING: {step.source}", flush=True)
        elif step.kind == "system":
            if ctypes.util.find_library(step.argv[0]) is None:
                print(f"WARNING: {step.source}", flush=True)
        elif step.kind == "run":
            if step.doctor and not Path(step.argv[0]).is_file():
                # Nothing verified that this box can plan, so this is no success (P0h h3d): install.ps1 warned and
                # exited 0 here, which read as green on exactly the box that has nothing yet.
                where = "make it and run" if layout.windows else "run this again, or make it and run"
                raise InstallRefused(
                    f"the environments are installed, and nothing verified that this box can plan: the doctor runs "
                    f"under the main environment, and there is none at {step.argv[0]}. To verify, {where} "
                    "python -m src.robot.safety.planning --doctor")
            code = _run(step, env, layout)
            if step.doctor:
                return Outcome(code, doctor_ran=True)
    return Outcome(0)


def _download(step: Step, layout: Layout) -> None:
    assert step.target is not None
    _banner(f"downloading {step.source}")
    step.target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(step.source, step.target)  # noqa: S310 - fixed https URLs above
    except OSError as exc:
        raise InstallRefused(f"downloading {step.source} failed: {exc}") from exc
    if not layout.windows:
        step.target.chmod(0o755)


def _remove(path: Path) -> None:
    """Remove a folder as ``Remove-Item -Recurse -Force`` does: a read-only file, as Git writes its packs, goes too."""

    def writable_then_again(function: Callable[..., Any], name: str, _excinfo: Any) -> None:
        os.chmod(name, stat.S_IWRITE)
        function(name)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=writable_then_again)
    else:
        shutil.rmtree(path, onerror=writable_then_again)


def _run(step: Step, env: dict[str, str], layout: Layout) -> int:
    argv = list(step.argv)
    _banner(render(step, layout) or "")
    if argv[:2] == ["cmd", "/c"]:
        # The command line install.ps1 handed cmd: `cmd /c "<bat>"`.
        code = subprocess.run(f'cmd /c "{argv[2]}"', cwd=step.cwd, env={**env, **step.env}, check=False).returncode
    else:
        code = subprocess.run(argv, cwd=step.cwd, env={**env, **step.env}, check=False).returncode
    if code == 0 or step.tolerate or step.doctor:
        return code
    if code in step.emergency:
        print("WARNING: EMERGENCY FALLBACK: no C++ toolchain, so the compiled kernel backend was not built. The planner "
              "will run on cuda.core alone, with no spare. On 2026-08-30 an application control policy refused "
              "cuda.core on this very machine, and the compiled backend was the only reason the planner kept working. "
              "Build it as soon as a toolchain exists, from a 64-bit developer prompt: "
              "scripts\\curobo\\build_compiled_backend.bat", flush=True)
        return code
    raise InstallRefused(f"{step.fails} (exit {code})")


def _banner(text: str) -> None:
    print(f"\n=== {text} ===", flush=True)


def _missing_tool(tool: str, layout: Layout) -> str:
    if tool == "git":
        if layout.windows:
            return ("needed to clone the pinned cuRobo revision. Git for Windows installs for a single user without "
                    "administrator rights: choose \"Only for me\" in the installer, or use the portable build.")
        return "needed to clone the pinned cuRobo revision. Install it with the system's package manager (sudo apt install git)."
    return "ships with Windows 10 1803 and later; on an older build, install it or unpack ext_deps/micromamba by hand."


def _doctor_said(code: int) -> str:
    if code == 0:
        return "done: the motion stack loads on this box."
    if code == 2:
        return ("an OS application-control policy is blocking a dependency. See docs/code-integrity.md. The fix is to "
                "pin that package to a build with reputation; re-downloading or moving the install will not help.")
    return "the stack installed but something is degraded: read the doctor above."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install Coal, cuRobo and (Linux) the main environment.")
    parser.add_argument("--component", choices=("all", "coal", "curobo", "main"), default="all")
    parser.add_argument("--clean", action="store_true", help="delete the target environment(s) first")
    parser.add_argument("--skip-compiled-backend", action="store_true",
                        help="skip cuRobo's compiled kernel backend (NOT recommended: one backend, no spare)")
    parser.add_argument("--ssl-no-revoke", action="store_true",
                        help="skip TLS certificate-revocation checks in micromamba (corporate TLS interception)")
    parser.add_argument("--plan", action="store_true", help="print every external command, run nothing")
    parser.add_argument("--from-nothing", action="store_true",
                        help="with --plan: plan as though nothing were installed yet")
    parser.add_argument("--platform", choices=_PLATFORMS, default=None,
                        help="with --plan: another platform's plan (default: this one)")
    parser.add_argument("--ext-deps", type=Path, default=DEFAULT_EXT_DEPS, help="the folder the install goes into")
    parser.add_argument("--write-pip-lock", action="store_true",
                        help="write curobo_env's pip lock from what is installed in it now, install nothing")
    args = parser.parse_args(argv)

    here = "win-64" if sys.platform == "win32" else "linux-64"
    platform = args.platform or here
    if platform != here and not args.plan:
        parser.error(f"--platform {platform} plans another platform's install; it runs only with --plan")
    if args.from_nothing and not args.plan:
        parser.error("--from-nothing describes a plan; an install always starts from what is there")
    ext_deps = Path(args.ext_deps).resolve()
    layout = Layout(platform, DEFAULT_EXT_DEPS if ext_deps == DEFAULT_EXT_DEPS.resolve() else ext_deps,
                    fresh=args.from_nothing)
    if args.component == "main" and layout.windows:
        parser.error("--component main makes the Linux main environment; on Windows .venv is not the installer's")
    if args.write_pip_lock:
        try:
            print(write_pip_lock(layout))
        except InstallRefused as exc:
            print(f"REFUSED: {exc}", file=sys.stderr, flush=True)
            return 1
        return 0
    steps = plan(layout, component=args.component, clean=args.clean, skip_compiled=args.skip_compiled_backend,
                 ssl_no_revoke=args.ssl_no_revoke)
    if args.plan:
        for step in steps:
            line = render(step, layout)
            if line is not None:
                print(line)
        return 0
    try:
        outcome = execute(steps, layout)
    except InstallRefused as exc:
        print(f"\nREFUSED: {exc}", file=sys.stderr, flush=True)
        return 1
    if outcome.doctor_ran:
        print(f"\n{_doctor_said(outcome.code)}", flush=True)
    return outcome.code


if __name__ == "__main__":
    sys.exit(main())
