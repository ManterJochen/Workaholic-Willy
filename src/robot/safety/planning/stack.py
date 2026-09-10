"""Which external motion engines this cell will use, and for which robot.

The defect this closes is a false green. `probe_planning_environment()` called with no
arguments returns a `fully_anchored` reading computed against ur5e descriptors and the
ur5e mesh bundle, and the object it returns carries neither the model nor any hint that
a model was chosen. On a UR3e cell that is a green light for a robot nobody configured.

The model and its source are therefore part of the reading rather than decoration on
it. A `fully_anchored` with no model attached cannot be acted on, because the mesh
bundle ships as ``{model}_collision_meshes.npz`` and the cuRobo descriptor as
``{model}.yml``, so a present `ur5e` bundle says nothing about a UR3e cell.
`MotionStackReport` carries both and `render()` prints them, which is what makes
`render()` return a whole object rather than a fragment its caller completes.

One resolution lives here and both callers read it, the terminal in
`planning/__main__.py` and the console in `api/routers/diagnostics.py`. Two answers to
which key decided which robot the box planned for is exactly the question a provenance
field exists to settle.

What `available` still does not mean: the cuRobo half reports that the interpreter of
the sidecar exists on disk. It does not mean the descriptor named beside it was built,
because that descriptor lives inside a separate environment this process deliberately
does not spawn. The caveat travels in `to_dict()` where a consumer can reach it, and
the gap it names is a bench step in `docs/runbooks/real_cell_first_pick.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from src.contracts import UNSET, Maybe, chosen

from .environment import PlanningEnvironment, probe_planning_environment

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

__all__ = ["ModelSource", "MotionStack", "MotionStackReport"]

#: What the probe falls back to where nothing declares a robot. It is a fallback rather
#: than a default, and the difference is that `ModelSource.FALLBACK` is recorded on the
#: report and printed, so a reading about `ur5e` on a cell that configures no robot
#: cannot be mistaken for a reading about that cell.
FALLBACK_MODEL = "ur5e"


class ModelSource(StrEnum):
    """Which lever decided the robot this reading is about.

    The values are the config keys themselves, verbatim, because a reader of a
    provenance string is looking for the line to edit. A friendlier label would be one
    more thing to translate.
    """

    #: The self-collision guard's own model. Authoritative when set, because it is the value the
    #: guard keys its mesh bundle on.
    SELF_COLLISION = "robot.safety.self_collision.kinematics_model"
    #: The vendor block's model, the fallback when the guard declares none.
    VENDOR_BLOCK = "robot.ur.model"
    #: The caller said so. It is named for the CLI flag, because that is the same lever
    #: seen from the terminal and a reader comparing a transcript against a script should
    #: not have to translate.
    CALLER = "--model"
    #: Nothing declared one. `detail` says which of the three ways that happened.
    FALLBACK = "default"


@dataclass(frozen=True, slots=True)
class MotionStackReport:
    """Both external engines, and the robot they were probed for."""

    #: The stack itself rather than a copy of its three fields. Copying `model`,
    #: `source` and `detail` onto the report would put the provenance derivation in two
    #: places, which is the defect this module closes one layer down.
    stack: "MotionStack"
    environment: PlanningEnvironment

    @property
    def model(self) -> str:
        return self.stack.model

    @property
    def source(self) -> ModelSource:
        return self.stack.source

    @property
    def detail(self) -> str:
        return self.stack.detail

    @property
    def model_source(self) -> str:
        """The provenance as a person reads it: a config key, a flag, or ``default (why)``."""
        return self.stack.model_source

    @property
    def fully_anchored(self) -> bool:
        """True iff both the cuRobo env and an exact-mesh engine with its bundle are present.

        It reads what is installed for :attr:`model` and nothing about the cell, which is
        why the verdict an operator acts on is :attr:`exit_code` and not this: where the
        config did not load, both engines can be present for a robot nobody configured.
        """
        return self.environment.fully_anchored

    @property
    def exit_code(self) -> int:
        """The ready-or-not verdict, on the report rather than in a caller.

        A caller that re-derives ``0 if fully_anchored else 1`` is a second derivation,
        and a second derivation is how a terminal and a service start disagreeing about
        whether the same box is ready.

        It answered 0 for a cell nothing was known about. Measured 2026-09-10:
        ``--check --data <a directory with no config in it>`` printed "=> fully anchored"
        and exited 0, because `for_this_box` had fallen back to ur5e and this rule read
        only the engine probe. The fallback is correct and deliberate, and reporting
        success about it is not, because the exit code is what a bring-up script branches
        on and it said "this cell is anchored" about a config that never loaded. A tree
        that did not load is a partially anchored answer at best, which is exactly what 1
        means here.
        """
        if self.stack.config_error:
            return 1
        return 0 if self.fully_anchored else 1

    def render(self) -> str:
        """The whole reading. ASCII, no trailing newline, no arguments.

        The model line is part of it, because a caller that appends the model itself
        leaves `PlanningEnvironment.render()` a fragment claiming to be a whole object,
        with the one fact that makes the reading actionable outside it.

        The retraction is the last line, where a terminal leaves it. The environment's own
        "=> fully anchored" is a true statement about the engines and stays. What it must
        not be allowed to be is the last word where the reading is not about the operator's
        cell.
        """
        lines = [self.environment.render(), f"  (model: {self.model}, from {self.model_source})"]
        if self.stack.config_error:
            lines.append(
                f"  !! not a reading about this cell: {self.stack.config_error}. The engines above "
                f"were probed for the {self.model} fallback, so the verdict says nothing about the "
                f"robot you are bringing up. Fix the config tree and run this again."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, safe for `json.dumps`. It is a view: nothing here is computed twice."""
        curobo = self.environment.curobo
        collision = self.environment.collision
        return {
            "model": self.model,
            "model_source": self.model_source,
            "fully_anchored": self.fully_anchored,
            # A consumer reading `fully_anchored` alone is reading about the fallback robot
            # where this is non-empty. Empty is the normal case and means the tree loaded.
            "config_error": self.stack.config_error,
            "curobo": {
                "available": curobo.available,
                "python_path": curobo.python_path,
                "robot_config": curobo.robot_config,
                # In the payload rather than the docs alone. A consumer that reads
                # `available` as a promise that the planner will work is wrong, and the
                # sentence saying so has to be somewhere a consumer can reach.
                "available_means": "the cuRobo environment's Python interpreter exists on disk; "
                                   "the robot descriptor inside it is not verified from here",
            },
            "collision": {
                "available": collision.available,
                "engine": collision.engine,
                "mesh_bundle_present": collision.mesh_bundle_present,
                "coal_prefix": collision.coal_prefix,
            },
        }


@dataclass(frozen=True, slots=True)
class MotionStack:
    """The motion stack of one cell: which robot, and which engines are anchored for it.

        from src.robot.safety.planning.stack import MotionStack

        report = MotionStack.for_this_box().probe()
        print(report.render())
        raise SystemExit(report.exit_code)

    It reads. It spawns nothing, plans nothing and moves nothing.
    """

    model: str
    source: ModelSource
    #: Why the fallback fired. It is empty for every other source, and empty is legal for
    #: `FALLBACK` too, because `api/routers/diagnostics.py` says a bare "default" and its
    #: wire field is public.
    detail: str = ""
    #: The refusal, in full, where the config tree this box would load did not load. Empty
    #: otherwise, including for a tree that loads and declares no robot: that is a config
    #: answer, this is a config fault, and only the fault means the reading is about a robot
    #: nobody chose. It is what makes `MotionStackReport.exit_code` fail closed, while
    #: `detail` stays the short provenance parenthesis a person reads at the end of the line.
    config_error: str = ""

    @property
    def model_source(self) -> str:
        """The provenance as a person reads it: a config key, a flag, or ``default (why)``.

        It is derived here, where the three fields live, and the report reads through to
        it. The CLI needs it before probing, because the `--doctor` branch prints it
        without touching either engine, so it cannot live on the reading alone.
        """
        if self.source is ModelSource.FALLBACK and self.detail:
            return f"{self.source.value} ({self.detail})"
        return self.source.value

    # --- three doors, of which the outer two call the inner one --------------------------------

    @classmethod
    def from_model(
        cls, *, model: str, source: ModelSource = ModelSource.CALLER, detail: str = "",
        config_error: str = "",
    ) -> "MotionStack":
        """A named robot. The plain-Python door, and the only one that constructs.

        This is the one builder. Both doors below resolve a model and then call it, so a
        reading taken from config and a reading taken from an argument cannot be
        assembled differently.
        """
        return cls(model=model, source=source, detail=detail, config_error=config_error)

    @classmethod
    def from_robot_config(
        cls, robot_config: "RobotConfig", *, model: "Maybe[str]" = UNSET
    ) -> "MotionStack":
        """The robot this validated config describes.

        The ladder runs in the order the guard itself uses: the `kinematics_model` of
        the self-collision guard where it is set explicitly, because that is the value
        the guard keys its mesh bundle on; then the model in the vendor block; then the
        fallback, which says so.

        It takes a validated schema object and never a path it loads. A factory that
        loaded would give this class a second way to fail and would hide the profile
        chain from the caller.
        """
        if chosen(model):
            return cls.from_model(model=model)
        declared = getattr(getattr(robot_config.safety, "self_collision", None), "kinematics_model", None)
        if declared:
            return cls.from_model(model=str(declared), source=ModelSource.SELF_COLLISION)
        vendor_model = getattr(getattr(robot_config, "ur", None), "model", None)
        if vendor_model:
            return cls.from_model(model=str(vendor_model), source=ModelSource.VENDOR_BLOCK)
        return cls.from_model(
            model=FALLBACK_MODEL,
            source=ModelSource.FALLBACK,
            detail="no model declared in this config",
        )

    @classmethod
    def for_this_box(
        cls,
        *,
        profile: "Maybe[str | None]" = UNSET,
        data_dir: "Maybe[str | None]" = UNSET,
        model: "Maybe[str]" = UNSET,
    ) -> "MotionStack":
        """Whatever config this box would load, tolerating a tree that does not load at all.

        Refusing to fail is the point. A motion-stack check still has to answer on a
        machine with a broken or absent config tree, because a config that does not load
        is one of the things an operator is standing there to diagnose. What it must
        never do is answer for the wrong robot in silence, so every way of not knowing
        produces its own `detail` and that string is printed on the reading.
        """
        if chosen(model):
            return cls.from_model(model=model)

        from src.config.loader import ConfigError, load_config  # noqa: PLC0415

        try:
            config = load_config(
                data_dir if chosen(data_dir) else None,
                **({"profile": profile} if chosen(profile) else {}),
            )
        except (ConfigError, OSError, ValueError) as exc:
            # The refusal is kept rather than summarised. Reduced to a `detail` parenthesis
            # it was printed after a green verdict and dropped from the exit code entirely,
            # so a script pointed at a broken tree was told the cell was anchored. Kept, it
            # is the one field that makes this reading fail closed.
            return cls.from_model(
                model=FALLBACK_MODEL,
                source=ModelSource.FALLBACK,
                detail=f"the config did not load: {type(exc).__name__}",
                config_error=f"{type(exc).__name__}: {exc}",
            )
        robot = getattr(config, "robot", None)
        if robot is None:
            return cls.from_model(
                model=FALLBACK_MODEL,
                source=ModelSource.FALLBACK,
                detail="this config tree configures no robot",
            )
        return cls.from_robot_config(robot)

    # --- the verb ------------------------------------------------------------------------------

    def probe(self) -> MotionStackReport:
        """Ask both engines whether they are present for this model. Spawn-free."""
        return MotionStackReport(
            stack=self,
            environment=probe_planning_environment(
                robot_config=f"{self.model}.yml", kinematics_model=self.model
            ),
        )
