"""Start a cell's planner at a desk, through the path its first planned move takes, and stop it again.

``PlannerStart`` builds the arm alone, from config, and calls the driver's own
:meth:`~src.robot.drivers.ur.arm.URRobotArm.start_planner`, so every refusal that move meets is met
here with the driver's own sentence, and no controller is asked. A cell whose hand body, sphere map,
retract rows and evidence were written with the scripts learns here whether its planner starts,
without a connected arm on a bench. The planner is stopped before the report returns.

The CLI is ``python -m src.robot.execution.real_cell --start-planner``; ``Cell.start_planner()`` is
the step on a cell. A real UR arm on ``motion_planner: curobo`` is the one cell this starts; every
other cell is refused saying why.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.contracts import UNSET, Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from pathlib import Path

    from src.config.schema.robot import RobotConfig

__all__ = ["PlannerStart", "PlannerStartReport"]


@dataclass(frozen=True)
class PlannerStartReport:
    """Whether a cell's planner started, what it loaded and what admitted it, or the refusal it met."""

    arm: str
    hand: str
    started: bool
    #: The refusal, verbatim; empty when the planner started.
    refusal: str = ""
    #: What the started sidecar said it loaded (:meth:`SidecarIdentity.render`); empty when nothing started.
    loaded: str = ""
    #: The committed evidence file that admits the combination, by name; empty when nothing started.
    evidence: str = ""
    seconds: float = 0.0
    #: Which wrist cameras the planner carried (``WristBodies.line``), or why none were asked about.
    wrist_bodies: str = ""

    @property
    def exit_code(self) -> int:
        """0 when the planner started, 1 when it was refused: a refusal here is a config fact, as at the desk."""
        return 0 if self.started else 1

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        head = f"planner  {self.arm} with {self.hand or 'no hand'}"
        if not self.started:
            return f"{head}: REFUSED after {self.seconds:.1f} s\n  {self.refusal}"
        lines = [f"{head}: STARTED and stopped, {self.seconds:.1f} s"]
        if self.evidence:
            lines.append(f"  admitted by {self.evidence}")
        if self.wrist_bodies:
            lines.append(f"  {self.wrist_bodies}")
        if self.loaded:
            lines.extend(f"  {line}" for line in self.loaded.splitlines())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "hand": self.hand, "started": self.started, "refusal": self.refusal,
                "loaded": self.loaded, "evidence": self.evidence, "seconds": self.seconds,
                "wrist_bodies": self.wrist_bodies}


@dataclass(frozen=True)
class PlannerStart:
    """One cell's planner, to be started and stopped at a desk."""

    robot_config: "RobotConfig"
    #: The config tree the cell came from, ``None`` for the repository's. A tree whose own registry
    #: describes the hand differently is refused before anything is built.
    data_dir: "str | Path | None" = None
    #: The camera section of the same tree, whose rigs say which wrist cameras the arm carries. Unset
    #: asks nothing about cameras, and the report says so.
    camera: "Maybe[Any]" = UNSET

    @classmethod
    def from_robot_config(cls, robot_config: "RobotConfig", *, data_dir: "str | Path | None" = None,
                          camera: "Maybe[Any]" = UNSET) -> "PlannerStart":
        return cls(robot_config=robot_config, data_dir=data_dir, camera=camera)

    def run(self) -> PlannerStartReport:
        """Build the arm, start its planner through the driver, stop it, and report. Never raises for a refusal."""
        cfg = self.robot_config
        arm_model = str(getattr(getattr(cfg, "ur", None), "model", "") or "")
        hand = str(cfg.gripper.model or "")
        began = time.perf_counter()

        def refused(sentence: str) -> PlannerStartReport:
            return PlannerStartReport(arm=arm_model, hand=hand, started=False, refusal=sentence,
                                      seconds=time.perf_counter() - began)

        vendor = str(getattr(cfg.vendor, "value", cfg.vendor)).lower()
        if vendor != "ur":
            return refused(
                f"robot.vendor is {vendor!r}, and this door starts a real UR arm's planner from its config: a sim "
                f"cell's planner starts with Isaac, and a dummy arm plans nothing"
            )
        from src.config.loader import ConfigError
        from src.robot.drivers import create_arm
        from src.robot.safety.planning import CuroboUnavailableError
        from src.robot.safety.planning.hand import planner_hand

        try:
            planner_hand(cfg, data_dir=self.data_dir)
        except ConfigError as exc:
            return refused(str(exc))
        from src.robot.execution.wrist_bodies import WristBodies, WristBodyRequired

        wrist: "WristBodies | None" = None
        if chosen(self.camera):
            try:
                wrist = WristBodies.from_config(cfg, self.camera, data_dir=self.data_dir)
            except WristBodyRequired as exc:
                return refused(str(exc))
        try:
            # The arm alone and not connected. The SDK readiness gate is not asked: the planner never
            # touches the controller's SDK, and a desk without ur_rtde is where this question is asked.
            # `Any`, because starting a planner at a desk is the UR driver's own verb, not the
            # vendor-neutral protocol's.
            arm: Any = create_arm("ur", config=cfg)
        except Exception as exc:  # noqa: BLE001 (a build refusal is the designed outcome, reported as it reads)
            return refused(f"{type(exc).__name__}: {exc}")
        try:
            if wrist is not None:
                wrist.hand_to(arm)
            identity = arm.start_planner()
        except CuroboUnavailableError as exc:
            return refused(str(exc))
        except Exception as exc:  # noqa: BLE001 (a sidecar that fails to start is a refusal to report, not a crash)
            return refused(f"{type(exc).__name__}: {exc}")
        finally:
            arm.stop_planner()

        from src.robot.safety.planning.evidence import desk_evidence_refusal

        _, evidence = desk_evidence_refusal(cfg, data_dir=self.data_dir)
        return PlannerStartReport(
            arm=arm_model, hand=hand, started=True, loaded=identity.render(),
            evidence=evidence.path.name if evidence is not None else "",
            seconds=time.perf_counter() - began,
            wrist_bodies=(wrist.line() if wrist is not None
                          else "wrist cameras  not asked: no camera section was handed in"),
        )
