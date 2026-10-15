"""The interpreter a printed command names, as a person types it from the repository root, per platform (P0h h2).

A refusal that ends in the command that fixes it is only as good as that command. Three interpreters are named in
this repository's texts: the main environment's, the cuRobo environment's and Isaac Sim's launcher. On Windows they
are ``.venv/Scripts/python.exe`` (a venv), ``ext_deps/curobo_env/python.exe`` (a conda environment keeps its
interpreter in its root) and ``python.bat``; everywhere else ``.venv/bin/python``, ``ext_deps/curobo_env/bin/python``
and ``python.sh``. Every text that prints one asks here, so on Windows each text is what it was, byte for byte, and a
Linux box is told a command it can run. The platform is ``sys.platform`` read when a text is written, so a test names
another platform by patching it or by passing ``platform=``.

The conda layout is written down once, in :func:`conda_interpreter_parts`, and the path the cuRobo sidecar is started
with (``src.robot.safety.planning.environment.env_python``) is built from the same parts, so the command a
refusal prints and the interpreter the planner starts cannot drift apart. Standard library only.
"""

from __future__ import annotations

import sys
from enum import StrEnum

__all__ = ["Interpreter", "conda_interpreter_parts", "installer_command", "interpreter_command", "is_windows"]


class Interpreter(StrEnum):
    """Which interpreter a printed command runs under."""

    #: The repository's main environment, ``.venv``: a venv on Windows, a micromamba environment elsewhere.
    MAIN = "main"
    #: The cuRobo environment under ``ext_deps``, a conda environment on every platform.
    CUROBO = "curobo"
    #: Isaac Sim's own launcher, run from the Isaac install.
    ISAAC = "isaac"


def is_windows(platform: str | None = None) -> bool:
    """Whether ``platform`` (``sys.platform`` read now when not given) is Windows."""
    return (sys.platform if platform is None else platform) == "win32"


def conda_interpreter_parts(platform: str | None = None) -> tuple[str, ...]:
    """Where a conda or micromamba environment keeps its interpreter, as path parts below the environment's root."""
    return ("python.exe",) if is_windows(platform) else ("bin", "python")


def interpreter_command(which: Interpreter, *, platform: str | None = None) -> str:
    """The interpreter ``which`` as a command names it, relative to the repository root and with ``/`` separators."""
    if which is Interpreter.MAIN:
        return ".venv/Scripts/python.exe" if is_windows(platform) else ".venv/bin/python"
    if which is Interpreter.CUROBO:
        return "/".join(("ext_deps", "curobo_env", *conda_interpreter_parts(platform)))
    return "python.bat" if is_windows(platform) else "python.sh"


def installer_command(*, platform: str | None = None) -> str:
    """The command that installs ext_deps, as a person types it from the repository root (P0h h3).

    On Windows ``scripts/ext_deps/install.ps1``, the shim every text named before; elsewhere the installer itself,
    ``python3 scripts/ext_deps/install.py``, which the shim runs on Windows too.
    """
    return "scripts/ext_deps/install.ps1" if is_windows(platform) else "python3 scripts/ext_deps/install.py"
