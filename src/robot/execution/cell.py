"""A robot cell as one noun, with the four steps in the order that makes them safe.

The pieces are public on their own: `load_robot_config`, `run_config_preflight`,
`build_real_cell`, `SafetyAttestation.of`, `cell_lock_key`, `ConnectedCell`, `service.pick()`.
This module publishes the order they run in. Preflight before build, because a blocking config
item is decidable at a desk. Attestation before motion, because an arm that gates nothing must
say so while there is still time to stop. Lock before connect, because a UR controller accepts
one control script. Arm before gripper, because a vacuum cup's connect drives digital I/O
immediately.

Narration is the caller's, not this module's: the steps are separate methods, so a caller prints
whatever it likes between two calls and this file never authors text it cannot see.

The order is enforced rather than documented. `connected()` on a cell that was never built raises
a typed refusal rather than an `AttributeError` three frames down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from src.contracts import UNSET, Maybe, chosen
from src.robot.execution.cell_lock import CellLock, cell_lock_key
from src.robot.execution.lifecycle import ConnectedCell, ConnectStage
from src.robot.execution.real_cell.preflight import (
    PreflightReport,
    run_config_preflight,
)
from src.robot.safety import SafetyAttestation

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

__all__ = ["Cell", "CellNotBuilt"]

#: What a rehearsal drives. The vendor is overridden on the operator's own config rather than a
#: separate config tree being loaded, so a rehearsal exercises the cell they actually have.
_REHEARSAL_VENDOR = "dummy"


class CellNotBuilt(RuntimeError):
    """A step was asked for before the step it depends on.

    Typed rather than an `AttributeError` on `None`, because the fix is a different call rather than
    a bug in this module, and the message should say which one.
    """


@dataclass
class Cell:
    """The cell this configuration describes, and the four steps that act on it.

        from src.config import load_robot_config
        from src.robot.execution.cell import Cell

        cell = Cell.from_robot_config(load_robot_config(), prompt="a red cube")
        print(cell.preflight().render())        # 1. decidable at a desk, no hardware
        cell.build()                            # 2. drivers, perception, grasp stack
        print(cell.safety().render())           # 3. what this arm will refuse, before it moves
        with cell.connected() as live:          # 4. lock, arm, then gripper
            report = live.service.pick()
        print(report.render())

    Each step is independent and useful alone: `preflight()` needs no hardware, `safety()` needs a
    build but no motion, and `connected()` is the only one that touches a cell.
    """

    robot_config: "RobotConfig"
    #: What the vision front end looks for. Ignored by a rehearsal, which uses a synthetic scene.
    #:
    #: Not defaulted here. A default in a wrapper is a second declaration of what the wrapped call
    #: looks for, and `build_real_cell` already declares it. `UNSET` forwards nothing, so that one
    #: declaration stays the only one.
    prompt: "Maybe[str]" = UNSET
    #: A dummy arm and a synthetic scene: the whole path at a desk, no camera, no robot.
    is_rehearsal: bool = False
    _service: Any = field(default=None, repr=False)

    # --- factories ---------------------------------------------------------------------------

    @classmethod
    def from_robot_config(
        cls, robot_config: "RobotConfig", *, prompt: "Maybe[str]" = UNSET
    ) -> "Cell":
        """The cell the configuration describes, as configured.

        Omitting ``prompt`` lets `build_real_cell` supply its own, which is the only declaration
        of it. The CLI passes its argparse default explicitly.
        """
        return cls(robot_config=robot_config, prompt=prompt)

    @classmethod
    def rehearsal(cls, robot_config: "RobotConfig") -> "Cell":
        """The same path with a dummy arm and a synthetic scene.

        The operator's own config with the vendor changed, not a separate tree, so the profile
        chain, the gripper branch and the grasping block are the ones they run.

        It is not a safety demonstration. The dummy arm carries no preflight and says so
        (`SafetyAttestation.of` reports ungated), so a rehearsal proves the wiring and nothing
        about what would refuse a bad command.
        """
        return cls(
            robot_config=robot_config.model_copy(update={"vendor": _REHEARSAL_VENDOR}),
            is_rehearsal=True,
        )

    # --- the four steps ----------------------------------------------------------------------

    def preflight(self) -> PreflightReport:
        """Every stop-the-cell condition that is decidable at a desk, each with its fix.

        Touches no hardware and needs no build, which is what makes it the first step.
        """
        return run_config_preflight(self.robot_config)

    def build(self) -> Any:
        """Construct the service: drivers, perception, the grasp stack. Idempotent.

        A second build would open a second camera on a device that has only one, so a repeat call
        returns the service already built.
        """
        if self._service is None:
            from src.robot.execution.autonomous_grasp import (  # noqa: PLC0415
                build_real_cell,
                build_rehearsal_cell,
            )

            # One branch, in one place. Both factories above produce a `Cell`; only this line
            # knows there are two ways to build the service behind it.
            if self.is_rehearsal:
                self._service = build_rehearsal_cell(self.robot_config)
            else:
                # Forwarded only when chosen, so an unspecified prompt reaches the callee's own
                # default rather than a copy of it made here.
                extra = {"prompt": self.prompt} if chosen(self.prompt) else {}
                self._service = build_real_cell(self.robot_config, **extra)
        return self._service

    def safety(self) -> SafetyAttestation:
        """What this cell's arm will actually refuse. Requires a build; commands nothing.

        Asked of the built arm, not of the config that asked for it: a config can request a
        preflight the driver does not carry.
        """
        return SafetyAttestation.of(self.arm)

    def connected(
        self, *, announce: "Callable[[ConnectStage], None] | None" = None
    ) -> ConnectedCell:
        """The cell, connected, for the duration of a ``with`` block.

        Takes the cross-process lock first when this cell owns a controller, then connects arm before
        gripper, and on the way out takes the gripper down first, then the arm, then the cameras, and
        gives the lock back. A simulated or dummy cell owns no controller and yields no lock key, so
        two rehearsals can run at once.
        """
        service = self._require_built("connected()")
        key = cell_lock_key(self.robot_config)
        lock = CellLock(key, owner="Cell") if key else None
        return ConnectedCell(service, lock=lock, announce=announce)

    # --- what was built ----------------------------------------------------------------------

    @property
    def service(self) -> Any:
        return self._require_built("service")

    @property
    def arm(self) -> Any:
        return getattr(self._orchestrator, "arm", None)

    @property
    def gripper(self) -> Any:
        return getattr(self._orchestrator, "gripper", None)

    @property
    def vendor(self) -> str:
        raw = getattr(self.robot_config, "vendor", "")
        return str(getattr(raw, "value", raw))

    @property
    def record_log_path(self) -> str:
        """Where attempts are logged, or the empty string when nothing is recorded."""
        return str(getattr(self.robot_config.grasping, "record_log_path", "") or "")

    @property
    def _orchestrator(self) -> Any:
        service = self._require_built("arm/gripper")
        return getattr(getattr(service, "runtime", None), "orchestrator", None)

    def _require_built(self, what: str) -> Any:
        if self._service is None:
            raise CellNotBuilt(
                f"{what} needs a built cell; call build() first. It is a separate step because "
                f"preflight() deliberately runs before anything is constructed."
            )
        return self._service
