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
    from pathlib import Path

    from src.config.schema import AppConfig
    from src.config.schema.robot import RobotConfig
    from src.config.tree import LoadedTree
    from src.robot.execution.planner_start import PlannerStartReport
    from src.robot.execution.robot import Robot
    from src.robot.execution.autonomous_grasp.config import GraspMode
    from src.robot.grasping.motion.grasp_motion import GraspMotion

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
    #: The tree ``robot_config`` came out of, for the half of a cell that is not the robot.
    #:
    #: A cell has two halves and they used to come from different trees. `robot_config` is resolved
    #: by the caller, who has a `--profile` and a `--data-dir`; the camera half was then read
    #: downstream with a bare `load_config()`, which falls back to `WILLY_PROFILE` and to the
    #: checkout's own tree. Measured 2026-09-10: the same chain through the flag and through the
    #: environment produced two different refusals. `UNSET` forwards nothing and the downstream
    #: default stands, so a caller who says nothing is unchanged.
    app_config: "Maybe[AppConfig]" = UNSET
    #: A dummy arm and a synthetic scene: the whole path at a desk, no camera, no robot.
    is_rehearsal: bool = False
    #: The config tree ``robot_config`` came from. ``None`` is the repository's tree, which is what a
    #: caller loading the default tree gets. The hand resolves from the repository's registry either
    #: way; a tree whose own ``grippers/`` describes the named hand differently is refused at the desk
    #: and at the build.
    data_dir: "str | Path | None" = None
    #: How the pick moves (standoff, squeeze, retreat), built by the pick service into the one policy it
    #: drives, on the arm and hand it resolves and with every guard. ``UNSET`` keeps the service's own.
    motion: "Maybe[GraspMotion]" = UNSET
    #: Which `GraspMode` the service is BUILT in: `easy`, `auto`, `dense_clutter`, `closed_loop` or
    #: `dense_autonomous`. ``UNSET`` keeps the service's own default, which is `auto`.
    #:
    #: Build-time, not per pick, and that is the whole reason it is here. `pick(mode=...)` changes
    #: the behaviour profile of one attempt and never the sampler the service was built with, so a
    #: service built in `auto` refuses `dense_clutter` with `MODE_NOT_AVAILABLE` rather than
    #: quietly sampling the other way. Without this field the only door to the two dense modes --
    #: the ones a bin needs -- was to bypass `Cell` and call `build_real_cell` by hand.
    mode: "Maybe[GraspMode | str]" = UNSET
    _service: Any = field(default=None, repr=False)

    # --- factories ---------------------------------------------------------------------------

    @classmethod
    def from_tree(
        cls, tree: "LoadedTree", *, prompt: "Maybe[str]" = UNSET, motion: "Maybe[GraspMotion]" = UNSET,
        mode: "Maybe[GraspMode | str]" = UNSET,
    ) -> "Cell":
        """The cell a loaded tree describes: both halves, and the directory, from one load.

            cell = Cell.from_tree(ConfigTree.from_directory(profile="ur5e,hande").load())
            cell = Cell.from_tree(ConfigTree.from_directory(root="D:/cells/line3", profile=None).load())

        Calls :meth:`from_robot_config` with the tree's robot section, its camera half and its root, so
        the preflight, the planner start and the build all read the tree that was loaded, values given in
        memory included. A tree that did not load is refused with its own refusal (``ConfigError``);
        anything but a ``LoadedTree`` is a ``TypeError``.
        """
        from src.config.loader import ConfigError  # noqa: PLC0415
        from src.config.tree import LoadedTree  # noqa: PLC0415

        if not isinstance(tree, LoadedTree):
            raise TypeError(
                f"Cell.from_tree takes a loaded tree, not {type(tree).__name__}: pass "
                f"ConfigTree.from_directory(...).load()")
        if not tree.ok:
            raise ConfigError(f"the tree did not load, so it describes no cell:\n{tree.error}")
        return cls.from_robot_config(tree.robot, prompt=prompt, app_config=tree.app_config, data_dir=tree.root,
                                     motion=motion, mode=mode)

    @classmethod
    def from_robot_config(
        cls, robot_config: "RobotConfig", *, prompt: "Maybe[str]" = UNSET,
        app_config: "Maybe[AppConfig]" = UNSET, data_dir: "str | Path | None" = None,
        motion: "Maybe[GraspMotion]" = UNSET, mode: "Maybe[GraspMode | str]" = UNSET,
    ) -> "Cell":
        """The cell the configuration describes, as configured.

        Omitting ``prompt`` lets `build_real_cell` supply its own, which is the only declaration
        of it. The CLI passes its argparse default explicitly.

        ``app_config`` is the tree ``robot_config`` came from, and a caller who resolved one should
        pass it: see the field for what happened while it could not be said.
        """
        return cls(robot_config=robot_config, prompt=prompt, app_config=app_config, data_dir=data_dir,
                   motion=motion, mode=mode)

    @classmethod
    def rehearsal(
        cls, robot_config: "RobotConfig", *, data_dir: "str | Path | None" = None,
        motion: "Maybe[GraspMotion]" = UNSET, mode: "Maybe[GraspMode | str]" = UNSET,
    ) -> "Cell":
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
            data_dir=data_dir,
            motion=motion,
            mode=mode,
        )

    # --- the four steps ----------------------------------------------------------------------

    def preflight(self) -> PreflightReport:
        """Every stop-the-cell condition that is decidable at a desk, each with its fix.

        Touches no hardware and needs no build, which is what makes it the first step.
        """
        # The camera half of the same tree, because a cell's CAMERA to BASE is declared on its
        # primary rig. With no tree supplied it is the tree at `data_dir`, and the default tree, the
        # one `build` would open, when that is `None`.
        from src.config import load_config  # noqa: PLC0415

        app_config = self.app_config if chosen(self.app_config) else load_config(self.data_dir)
        return run_config_preflight(self.robot_config, camera=app_config.camera, data_dir=self.data_dir)

    def start_planner(self) -> "PlannerStartReport":
        """Start this cell's planner through its driver, with every refusal its first planned move meets, and stop it.

        Builds the arm alone, opens no camera and asks no controller, so it belongs beside the
        preflight rather than after the build: a customer's combination is proven to start at a desk.
        """
        from src.config import load_config  # noqa: PLC0415
        from src.robot.execution.planner_start import PlannerStart  # noqa: PLC0415

        # The camera half of the same tree, as the preflight reads it: its rigs say which wrist
        # cameras the arm carries, and the planner starts with them.
        app_config = self.app_config if chosen(self.app_config) else load_config(self.data_dir)
        return PlannerStart.from_robot_config(self.robot_config, data_dir=self.data_dir,
                                              camera=app_config.camera).run()

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
            # The motion is forwarded only when chosen, as the prompt is: unset keeps the service's own.
            motion: dict[str, Any] = {"motion": self.motion} if chosen(self.motion) else {}
            if chosen(self.mode):
                motion["mode"] = self.mode
            if self.is_rehearsal:
                self._service = build_rehearsal_cell(self.robot_config, data_dir=self.data_dir, **motion)
            else:
                # Forwarded only when chosen, so an unspecified prompt reaches the callee's own
                # default rather than a copy of it made here.
                #
                # Annotated because the two keys hold different types, and an inferred
                # `dict[str, str]` from the first one makes the second an error rather than a value.
                extra: dict[str, Any] = {"prompt": self.prompt} if chosen(self.prompt) else {}
                if chosen(self.app_config):
                    extra["app_config"] = self.app_config
                self._service = build_real_cell(self.robot_config, data_dir=self.data_dir, **extra, **motion)
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
    def robot(self) -> "Robot":
        """This cell's built arm and hand as a :class:`Robot`, for the verbs a pick does not run. Requires a build.

            cell.build()
            with cell.connected():
                print(cell.robot.home().render())
                print(cell.robot.move(pose).render())

        The same handles the pick service drives, so inside ``with cell.connected():`` every verb goes
        through this arm's planner, its guard and the camera world the build wired onto it; nothing is
        built, opened or locked a second time. Outside that block the verbs refuse a link that is not
        open, as on any robot. Take the lock through ``cell.connected()``, not
        ``cell.robot.connected()``, while the cell is connected: both take the same lock, and the second
        is refused as busy.
        """
        from src.robot.execution.robot import Robot  # noqa: PLC0415

        self._require_built("robot")
        return Robot.from_parts(arm=self.arm, gripper=self.gripper, robot_config=self.robot_config)

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
