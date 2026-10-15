"""Build one arm: every stage of ``scripts/arms/`` in order, each under the interpreter it runs under.

    .venv/Scripts/python.exe scripts/arms/build_arm.py ur3e                          # verify, write nothing
    .venv/Scripts/python.exe scripts/arms/build_arm.py ur3e --write                  # bake, fit, install the descriptor
    .venv/Scripts/python.exe scripts/arms/build_arm.py ur3e --profile ur3e,hande             # plan and read the cell

The stages need three environments and nothing else knew which: ``fetch``, ``descriptor`` and ``check`` run under the
cuRobo sidecar's interpreter, which cannot import this repository, and ``bake``, ``fit`` and the planning doctor under
the project's own. This script runs under the project's interpreter, finds the cuRobo one where the planner finds it
(``environment.curobo_python_path``), and starts each stage as its own process, so a stage runs exactly as its own
command line would run it. The first stage that exits non-zero stops the build, and the report says which.

Without ``--write`` each stage runs in the form that writes nothing: ``fetch --check``, a bake that compares, a fit
that reports. The descriptor has no such form (it installs into the cuRobo content directory), so it runs with
``--write`` only. With a hand the descriptor is loaded into cuRobo with it and made to plan (``check``). With
``--profile`` the planning doctor reads that cell: its retract and evidence rows name the whole ``retract.py`` and
``gate.py`` commands the combination needs, built by the functions the planner's own refusals use. Those two stages
measure a combination with Coal and a live sidecar and take the cell's plates and placement, so this script names
them rather than running them.

The doctor runs with the cell's profile or not at all. Handed a model and a hand it loads no cell
(``MotionStack.for_this_box``), and then answers "missing" for every arm, healthy or not: measured in aurora on
2026-09-23, the first run of this script stopped a healthy ur5e there. Its rows read the cell's own hand, plates and
placement, so the hand comes from the cell where the profile names one, a ``--hand`` that contradicts it is refused,
and so is a profile that describes another arm: the doctor would read that arm's cell under this arm's name. An arm
profile names no hand, and the doctor then says what the cell must declare.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.contracts import UNSET, Maybe, chosen  # noqa: E402

HERE = Path(__file__).resolve().parent
#: The order the stages depend on each other in: the meshes before the bake, the bundle before the fit, the fit before
#: the descriptor, the descriptor before a planner is asked anything.
STAGES = ("fetch", "bake", "fit", "descriptor", "check", "doctor")


@dataclass(frozen=True, slots=True)
class StageRun:
    """One stage: what it was started as, and how it ended. ``skipped`` is why it did not run, when it did not."""

    name: str
    interpreter: str
    argv: tuple[str, ...]
    exit_code: "int | None" = None
    skipped: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "interpreter": self.interpreter, "argv": list(self.argv),
                "exit_code": self.exit_code, "skipped": self.skipped}


@dataclass(frozen=True, slots=True)
class ArmBuildReport:
    """What one build ran, in order, and the one stage that stopped it, if any."""

    model: str
    write: bool
    stages: tuple[StageRun, ...]
    refused: str = ""

    @property
    def failed(self) -> "StageRun | None":
        return next((stage for stage in self.stages if stage.exit_code not in (None, 0)), None)

    @property
    def exit_code(self) -> int:
        return 1 if self.refused or self.failed is not None else 0

    def render(self) -> str:
        lines = [f"build {self.model} ({'write' if self.write else 'verify, nothing written'})"]
        if self.refused:
            lines.append(f"  refused: {self.refused}")
        for stage in self.stages:
            if stage.skipped:
                lines.append(f"  [skip] {stage.name:<10} {stage.skipped}")
            elif stage.exit_code is None:
                lines.append(f"  [ -- ] {stage.name:<10} not started: an earlier stage stopped the build")
            else:
                mark = "ok  " if stage.exit_code == 0 else "FAIL"
                lines.append(f"  [{mark}] {stage.name:<10} exit {stage.exit_code}: {' '.join(stage.argv)}")
        failed = self.failed
        if failed is not None:
            lines.append(f"stopped at {failed.name}, exit {failed.exit_code}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "write": self.write, "refused": self.refused, "exit_code": self.exit_code,
                "stages": [stage.to_dict() for stage in self.stages]}


@dataclass(frozen=True, slots=True)
class ArmBuild:
    """One arm's build, before it runs. :meth:`from_model` resolves the interpreters; :meth:`run` starts the stages."""

    model: str
    write: bool
    hand: "Maybe[str]"
    profile: "Maybe[str]"
    python: str
    curobo_python: str
    descriptor: str

    @classmethod
    def from_model(cls, model: str, *, write: bool = False, hand: "Maybe[str]" = UNSET,
                   profile: "Maybe[str]" = UNSET, python: "Maybe[str]" = UNSET,
                   curobo_python: "Maybe[str]" = UNSET) -> "ArmBuild":
        """The build for ``model``. ``ConfigError`` for an arm no description names, or for a profile that describes
        another arm, before anything runs."""
        from src.config import ConfigError
        from src.config.arms import curobo_descriptor_name
        from src.config.loader import load_robot_config
        from src.robot.safety.planning.environment import curobo_python_path

        descriptor = curobo_descriptor_name(model)
        if chosen(profile):
            cell = load_robot_config(None, profile=profile)
            described = str(cell.arm.model)
            if described != model.strip().lower():
                raise ConfigError(f"the profile {profile!r} describes a {described}, and this build is for a "
                                  f"{model.strip().lower()}: the doctor would read the {described}'s cell under "
                                  "this arm's name")
            # The doctor's retract and evidence rows read the CELL's hand, so the check plans with the same one.
            named = cell.gripper.model
            if named is not None and chosen(hand) and str(named) != hand:
                raise ConfigError(f"the profile {profile!r} names the hand {named}, and --hand says {hand}: the "
                                  "check would plan with one and the doctor read the cell's other")
            if named is not None and not chosen(hand):
                hand = str(named)
        return cls(model=model.strip().lower(), write=bool(write), hand=hand, profile=profile,
                   python=python if chosen(python) else sys.executable,
                   curobo_python=curobo_python if chosen(curobo_python) else curobo_python_path(),
                   descriptor=descriptor.removesuffix(".yml"))

    def stages(self) -> tuple[StageRun, ...]:
        """Every stage as it would be started, the ones this build does not run carrying why."""
        cu, py, model = self.curobo_python, self.python, self.model
        write = ("--write",) if self.write else ()

        def script(name: str) -> str:
            return str(HERE / f"{name}.py")

        runs = [
            StageRun("fetch", "curobo", (cu, script("fetch"), *(() if self.write else ("--check",)))),
            StageRun("bake", "project", (py, script("bake"), model, *write)),
            StageRun("fit", "project", (py, script("fit"), "--arm", model, *write)),
            StageRun("descriptor", "curobo", (cu, script("descriptor"), model),
                     skipped="" if self.write else "it installs into the cuRobo content directory; runs with --write"),
        ]
        if chosen(self.hand):
            runs.append(StageRun("check", "curobo", (cu, script("check"), self.descriptor, "--hand", self.hand)))
        else:
            runs.append(StageRun("check", "curobo", (),
                                 skipped="names no hand to plan with; pass --hand, or a --profile whose cell names one"))
        if chosen(self.profile):
            # No --model: with a model and a hand the doctor loads no cell (MotionStack.for_this_box). The profile's
            # arm is this build's and its hand, where it names one, is the check's: both settled in from_model.
            hand = ("--hand", self.hand) if chosen(self.hand) else ()
            runs.append(StageRun("doctor", "project", (py, "-m", "src.robot.safety.planning", "--doctor",
                                                       "--profile", self.profile, *hand)))
        else:
            runs.append(StageRun("doctor", "project", (),
                                 skipped="names no cell: the retract row and the evidence file a planner start looks "
                                         "up are the cell's; pass --profile with the cell's profile chain"))
        return tuple(runs)

    def run(self, *, runner: "Callable[..., Any]" = subprocess.run) -> ArmBuildReport:
        """Start each stage in order, and stop at the first that exits non-zero."""
        from src.utility.interpreters import installer_command  # noqa: PLC0415

        planned = self.stages()
        needed = {stage.interpreter for stage in planned if not stage.skipped}
        if "curobo" in needed and not Path(self.curobo_python).is_file():
            return ArmBuildReport(model=self.model, write=self.write, stages=planned,
                                  refused=f"the cuRobo interpreter {self.curobo_python} is not there, and fetch, "
                                          f"descriptor and check run under it: install it with "
                                          f"{installer_command()}, or name it in WILLY_CUROBO_PYTHON")
        done: list[StageRun] = []
        stopped = False
        for stage in planned:
            if stage.skipped or stopped:
                done.append(stage)
                continue
            print(f"[build_arm] {stage.name}: {' '.join(stage.argv)}", flush=True)
            code = int(runner(list(stage.argv), cwd=str(REPO)).returncode)
            done.append(StageRun(stage.name, stage.interpreter, stage.argv, exit_code=code))
            stopped = code != 0
        return ArmBuildReport(model=self.model, write=self.write, stages=tuple(done))


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="scripts/arms/build_arm.py", description=__doc__.split("\n")[0])
    parser.add_argument("model", help="the arm's registry name, e.g. ur5e")
    parser.add_argument("--write", action="store_true", help="bake, fit and install the descriptor")
    parser.add_argument("--hand", default=None, help="plan with this hand, then read the cell with the doctor")
    parser.add_argument("--profile", default=None,
                        help="the cell's profile, for the doctor (with --hand); it must describe this arm")
    args = parser.parse_args(argv)
    from src.config import ConfigError

    try:
        build = ArmBuild.from_model(args.model, write=args.write, hand=args.hand if args.hand else UNSET,
                                    profile=args.profile if args.profile else UNSET)
    except ConfigError as exc:
        print(f"build {args.model} refused: {exc}", file=sys.stderr)
        return 1
    report = build.run()
    print(report.render())
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
