"""cuRobo motion planning, the process-isolated GPU planner.

cuRobo runs collision-aware global trajectory optimisation on the GPU. It cannot share
the Isaac process, because its ``warp`` version clashes with Isaac, so it runs as a
process-isolated stdio sidecar. :class:`CuroboPlanClient` lives in this environment,
uses the standard library alone and imports neither cuRobo nor Isaac. It spawns
``curobo_planner_server.py`` with the python of the cuRobo environment and speaks
newline-delimited JSON over the pipe.

It is a firm dependency for real collision-aware motion, used by both the Isaac sim
driver in :mod:`src.robot.drivers.sim` and the real-UR execution path in
:mod:`src.robot.drivers.ur.curobo_motion`. It still stays an out-of-process service,
never imported in-process, and it cannot become a pip line: it needs python 3.10 and a
CUDA build, it clashes with ``warp``, and there is no wheel. ``WILLY_CUROBO_PYTHON``
locates the environment, defaulting to ``ext_deps/curobo_env/python.exe``, and the
reference robot and gripper descriptor lives under :mod:`.robot` at
``WILLY_CUROBO_ROBOT``. ``ext_deps/README.md`` covers the install.

The fail-closed contract: where the cuRobo environment is unavailable, the sim and mock
path may fall back to blind IK, which leaves behaviour unchanged, but a real-hardware
caller fails closed with no blind motion and surfaces
:class:`CuroboUnavailableError`. Planner collision-awareness is simulation-grade and
not a certified functional-safety stop, so a real cell still needs the vendor
safety-rated stop.

This package is also the anchor for the external engines the motion stack builds on.
:mod:`.environment` owns every environment-variable name, default path and
availability probe for cuRobo and for the Coal and python-fcl exact-mesh collision
engine, and ``python -m src.robot.safety.planning --check`` reports what is wired on
this box.
"""

from __future__ import annotations

from .curobo_client import CuroboPlanClient, CuroboUnavailableError, curobo_env_available
from .environment import (
    CollisionEngineStatus,
    CuroboStatus,
    PlanningEnvironment,
    probe_collision_engine,
    probe_curobo,
    probe_planning_environment,
)

from .stack import ModelSource, MotionStack, MotionStackReport

__all__ = [
    "CuroboPlanClient",
    "ModelSource",
    "MotionStack",
    "MotionStackReport",
    "CuroboUnavailableError",
    "curobo_env_available",
    "CollisionEngineStatus",
    "CuroboStatus",
    "PlanningEnvironment",
    "probe_collision_engine",
    "probe_curobo",
    "probe_planning_environment",
]
