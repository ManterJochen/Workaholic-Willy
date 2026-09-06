"""The lifecycle wrapper around the Isaac Sim application.

It owns the lazy SDK import and the simulator app handle, so the rest of the driver
stays free of Isaac-specific symbols. On a host without Isaac the constructor stays
inert and :meth:`start` raises :class:`IsaacNotAvailableError`, never at module import,
so such a host keeps importing this module under ``mock_mode=True``.

The real Isaac bring-up lives here: boot a headless ``SimulationApp``, open the
configured scene or a default empty stage, create the ``World`` and step it. It is
validated on the Windows RTX 5080 against Isaac Sim 5.1.0, which section 14 of
``docs/ISAAC_VALIDATION_PLATFORM.md`` records. ``mock_mode=True`` bypasses Isaac
entirely for the pure-Python kinematic mock used off that workstation.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from ...core import IsaacNotAvailableError
from .config import SimRobotConfig

__all__ = ["IsaacSimSession"]


class IsaacSimSession:
    """Manage the Isaac Sim application for one :class:`IsaacRobotArm`.

    It owns the config, the ``SimulationApp`` handle and the ``World``, and centralises
    the SDK import, so :class:`IsaacRobotArm` never imports Isaac directly.
    ``SimulationApp`` is a process singleton, so exactly one session is live per
    process.
    """

    def __init__(self, config: SimRobotConfig) -> None:
        self._config = config
        self._app: Any | None = None
        self._world: Any | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def config(self) -> SimRobotConfig:
        return self._config

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def app(self) -> Any | None:
        """The live ``SimulationApp`` handle. It is ``None`` before :meth:`start` and in mock mode."""
        return self._app

    @property
    def world(self) -> Any | None:
        """The live Isaac ``World``. It is ``None`` before :meth:`start` and in mock mode."""
        return self._world

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Boot the Isaac application. Idempotent.

        The behaviour depends on :attr:`SimRobotConfig.mock_mode`:

        * With ``mock_mode=True``, the default for any host without Isaac, no SDK import
          is attempted: the session marks itself running and returns. Together with the
          mock motion path of :class:`IsaacRobotArm` that is a deterministic pure-Python
          driver satisfying the typed motion contract.
        * With ``mock_mode=False``, a real headless ``SimulationApp`` boots, the
          configured scene or a default empty stage opens, and a ``World`` is created
          and reset.

        Raises
        ------
        IsaacNotAvailableError
            Where ``mock_mode=False`` and the Isaac SDK does not import on this host.
        """
        if self._running:
            return  # idempotent

        if self._config.mock_mode:
            # The pure-Python path, which deliberately never touches Isaac.
            self._running = True
            return

        # ``OMNI_KIT_ACCEPT_EULA`` is set before the first isaacsim import, because
        # otherwise the headless boot can block waiting for the EULA to be accepted.
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        try:
            # The lazy SDK import, in the Isaac Sim 5.1 namespace, since
            # ``omni.isaac.kit`` was removed in 4.5. It sits inside the method so module
            # import stays safe on a host without Isaac.
            from isaacsim import SimulationApp  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover (covered on non-Isaac hosts)
            raise IsaacNotAvailableError(
                "Isaac Sim SDK is not importable on this host. The RobotVendor.SIM "
                "driver's non-mock path requires NVIDIA Isaac Sim 5.1 (run with its "
                "bundled python). Set SimRobotConfig(mock_mode=True) for the pure-Python "
                "mock kinematics path. See docs/ISAAC_VALIDATION_PLATFORM.md."
            ) from exc

        # ``SimulationApp`` forwards ``sys.argv`` to the Omniverse Kit parser, which
        # rejects a foreign flag and then crashes. So argv is stripped to the program
        # name across the boot and restored afterwards.
        saved_argv, sys.argv = sys.argv, sys.argv[:1]
        try:
            self._app = SimulationApp({"headless": self._config.headless})
        finally:
            sys.argv = saved_argv

        # Deferred imports, valid only once the app has loaded its extensions.
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import create_new_stage, open_stage

        if self._config.scene:
            open_stage(self._config.scene)
        else:
            create_new_stage()

        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=self._config.step_dt_s,
            rendering_dt=self._config.step_dt_s,
        )
        self._world.reset()
        self._running = True

    def step(self, dt_s: float | None = None, *, render: bool = False) -> None:
        """Advance the simulation by one step.

        The physics and rendering ``dt`` is fixed at :attr:`SimRobotConfig.step_dt_s`,
        set on the ``World`` at :meth:`start`, and ``dt_s`` is accepted for interface
        symmetry and ignored. It does nothing in mock mode or before :meth:`start`.
        """
        if not self._running or self._world is None:
            return
        self._world.step(render=render)

    def step_n(self, count: int, *, render: bool = False) -> None:
        """Drive a fixed number of :meth:`step` calls. Most callers want this one."""
        for _ in range(max(0, count)):
            self.step(render=render)

    def stop(self) -> None:
        """Shut down the Isaac application where one is running. Safe to call repeatedly.

        The headless ``SimulationApp.close()`` can segfault on shutdown, which is a known
        upstream issue, so a caller that needs a reliable result captures it before
        stopping.
        """
        if self._app is not None:
            self._app.close()
        self._app = None
        self._world = None
        self._running = False
