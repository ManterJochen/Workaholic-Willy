"""42: the `grasping.*` blocks that ship off, and the mode gate that keeps most of them off.

    python scripts/examples/grasping/42_the_thirteen_switches.py
    python scripts/examples/grasping/42_the_thirteen_switches.py --profile sim
    python scripts/examples/grasping/42_the_thirteen_switches.py --profile decision

Thirteen blocks under `robot.grasping` carry an `enabled` flag and every one of them ships false.
That is a deliberate default and not an oversight: the pick a cell runs out of the box is
open-loop, and each of these adds a stage that has to be measured on the cell it runs on before it
is worth its cost. The decision this file covers is which of them to turn on, and the first thing
to know is that turning one on frequently changes nothing at all.

`robot.grasping.default_mode` is the gate in front of them. Each mode-gated block declares its own
`apply_modes`, and `build_effective_config` enforces that filter, so a block outside the running
mode's list collapses to its off value even when the YAML says true. `easy` appears in the
`apply_modes` of exactly one block. Four lists exclude `closed_loop` as well, which is easy to miss
when `closed_loop` is the mode you reached for precisely because you wanted more behaviour. This
file switches each block on in turn, resolves it in four modes, and prints which ones read back as
on, so the gate is measured here rather than quoted.

Reading on is still not the same as acting. Three blocks are built in every mode and fire in almost
none, and four ship the value that makes them matter set to zero, so `enabled: true` alone computes
a signal and multiplies it by nothing. The feasibility block below needs all three conditions at
once, its flag, a weight above zero and a mode in its list, before the orchestrator is given a
carrier, and that is why the config snapshot and the carrier are both printed: the snapshot is what
the cell reports about itself, and it can say on while nothing was installed.

There is a fourth shape, and the schema refuses it. A switch whose value reaches the snapshot and
is then read back by nothing would give an operator a cell that boots, runs, reports the capability
and behaves exactly as before. `RobotGraspingConfig.UNWIRED_SWITCHES` names those, and a model
validator refuses to load a config that sets one. This file sets one and prints the refusal.

Nothing here connects to anything. The cells built below are rehearsal cells, on the dummy vendor,
constructed and inspected and never connected, so no controller is reached and nothing can move.

Read `docs/grasping-config-reference.md` next; sections 2 and 3 are what this file executes.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  every claim this file makes about the gate and the schema was checked and held
  1  one did not: a mode gate that did not gate, or an unwired switch the schema accepted
  2  the config tree has no `robot` block, so there are no blocks to switch
"""

#: The modes the gate is measured in. `easy` and `auto` are what the shipped trees run (`auto` in
#: `config/robot/robot.yaml`, `easy` in the simulation runners); `closed_loop` is the one operators
#: reach for wanting more behaviour and lose four blocks by doing so; `dense_clutter` is the one
#: most blocks were written for.
_MODES: tuple[str, ...] = ("easy", "auto", "closed_loop", "dense_clutter")

#: The four blocks that ship their operative value at zero or empty, and the key that has to be
#: set alongside `enabled`. Their shipped values are read from the tree rather than written here.
_ZERO_WEIGHT: dict[str, str] = {
    "feasibility": "weight",
    "ordering": "unlock_weight",
    "uncertainty": "ranking_penalty_weight",
    "recovery": "allowed_actions",
}

#: The block the trap and the carrier are worked through. Feasibility is the one that needs all
#: three conditions at once, its flag, a weight above zero and a mode in its list, before the
#: orchestrator is handed anything, so it is where the three silences are visible together.
_WORKED = "feasibility"

#: How its keys are named in the flat snapshot, and what its carrier is called on the orchestrator.
#: Neither is derivable from the block name: `recovery` becomes `recovery_orchestrator_*` in the
#: snapshot and `ordering` installs a carrier called `target_ordering`.
_WORKED_PREFIX = "feasibility_"
_WORKED_CARRIER = "feasibility_config"


def _snapshot(block: Any, mode: Any) -> dict[str, Any]:
    """The flat telemetry snapshot a cell would report for ``block`` resolved in ``mode``.

    `build_effective_config` answers `None` only when it is handed no grasping block at all, which
    is the `mode=` override path and not this one. The check is here so a type checker is told
    that, rather than a case this file cannot produce being papered over at each call site.
    """
    from src.robot.execution.autonomous_grasp.builders import build_effective_config

    effective = build_effective_config(block, resolved_mode=mode, resolved_max_attempts=5)
    if effective is None:  # pragma: no cover (unreachable with a real grasping block)
        raise RuntimeError("build_effective_config returned no snapshot for a real block")
    return effective.to_dict()


def _carrier(cell: Any, name: str) -> Any:
    """What the orchestrator actually holds, which no config snapshot can report.

    Reached through the built service rather than asked of the config, because the question is
    whether a carrier was installed. `apply_orchestrator_overlays` also logs the names it installed
    for the resolved mode, and that log line is the same fact from the builder's side.
    """
    runtime = getattr(cell.service, "runtime", None)
    return getattr(getattr(runtime, "orchestrator", None), name, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__ or "", epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--profile", default=None,
        help="WILLY_PROFILE chain to load. `decision` is a shipped layer that turns exactly one "
             "of these blocks on. Default: the environment's.")
    args = parser.parse_args(argv)

    # After the parse, so `--help` answers without importing the stack.
    from pydantic import ValidationError

    from src.config import default_data_dir, load_config
    from src.config.explain import explain_in
    from src.config.schema.robot.grasping_schema import RobotGraspingConfig
    from src.robot.execution.autonomous_grasp import resolve_grasp_mode
    from src.robot.execution.cell import Cell

    with Example("42 the thirteen switches", "the default-off blocks and the mode gate",
                 hardware=False) as run:
        with run.step("the blocks that carry a switch, from the schema") as report:
            tree = load_config(profile=args.profile) if args.profile else load_config()
            robot = tree.robot
            if robot is None:
                report("no `robot` block in this tree")
            else:
                grasping = robot.grasping
                # Read off the model rather than written down here, so a block added to the schema
                # appears in this output without anyone remembering to update a list.
                blocks = [name for name in type(grasping).model_fields
                          if hasattr(getattr(grasping, name, None), "enabled")]
                on = [name for name in blocks if bool(getattr(grasping, name).enabled)]
                report(f"{len(blocks)} block(s) with an `enabled` flag, "
                       f"{len(on) or 'none'} on in this tree"
                       + (f": {', '.join(on)}" if on else ""))
        if robot is None:
            return not_ready("this config tree has no `robot` block",
                             "point the loader at a tree that has one; `config/robot/robot.yaml` "
                             "in this repository is one")

        with run.step("the mode this tree runs") as report:
            mode = resolve_grasp_mode(grasping.default_mode)
            report(f"default_mode {grasping.default_mode!r} resolves to the {mode.value} profile")
        run.note("`occlusion` is a fourteenth block and is not in the count: it has")
        run.note("`directional_enabled` and `hard_reject_enabled` instead of one `enabled`, and")
        run.note("ships both false.")
        run.note("")

        # Which mode reads which block as on. One block is switched on at a time, the snapshot is
        # rebuilt in four modes, and the answer is whether any key moved against an all-default
        # snapshot in the same mode. Nothing is asserted from the documentation.
        # The tree's own mode is always one of the columns, so a cell running `dense_autonomous`
        # gets a table that answers for it rather than for four modes it does not use.
        modes = _MODES if mode.value in _MODES else (*_MODES, mode.value)
        baseline = {name: _snapshot(grasping, resolve_grasp_mode(name)) for name in modes}
        table: dict[str, tuple[bool, ...]] = {}
        with run.step("switch each block on, and read the snapshot back") as report:
            for block in blocks:
                switched = grasping.model_copy(update={
                    block: getattr(grasping, block).model_copy(update={"enabled": True})})
                reads: list[bool] = []
                for name in modes:
                    moved = _snapshot(switched, resolve_grasp_mode(name))
                    reads.append(any(moved[k] != baseline[name][k] for k in moved))
                table[block] = tuple(reads)
            here = modes.index(mode.value)
            silent = [name for name, hits in table.items() if not hits[here]]
            report(f"{len(silent)} of {len(blocks)} change nothing in `{mode.value}`, "
                   f"the mode this tree runs")
        run.note(f"{'block':<20} " + "  ".join(f"{name:<13}" for name in modes))
        for block, hits in table.items():
            run.note(f"{block:<20} "
                     + "  ".join(f"{'reads on' if hit else 'no change':<13}" for hit in hits))
        run.note("")
        run.note("`deep_ranker` changes no column because it has no key in the snapshot at all: it")
        run.note("is carried on the orchestrator instead. A row of `no change` is not proof a")
        run.note("block is dead, only that the cell's own telemetry will not mention it. A block")
        run.note("the loaded tree already turned on reads `no change` for a third reason, that")
        run.note("switching it on again moves nothing; the first step names those.")
        run.note("")

        # The zero-weight trap, on one block. Its flag goes on, its operative value is left where
        # the tree ships it, and the mode is the one this tree runs.
        operative = _ZERO_WEIGHT[_WORKED]
        with run.step(f"`{_WORKED}.enabled: true`, and nothing else") as report:
            switched = grasping.model_copy(update={
                _WORKED: getattr(grasping, _WORKED).model_copy(update={"enabled": True})})
            snapshot = _snapshot(switched, mode)
            shipped = getattr(getattr(grasping, _WORKED), operative)
            keys = {k: v for k, v in snapshot.items() if k.startswith(_WORKED_PREFIX)}
            report(f"{_WORKED}.{operative} ships as {shipped!r}, so the signal is computed and "
                   f"then multiplied by nothing")
        for key, value in keys.items():
            run.note(f"  {key} = {value!r}")
        run.note("")
        run.note("The snapshot is the cell's own telemetry, and there it reads enabled with a")
        run.note("weight of zero. A run measured against a baseline would show no difference, and")
        run.note("the record would say the block was on. Four blocks ship this way:")
        for block, key in sorted(_ZERO_WEIGHT.items()):
            run.note(f"  {block}.{key} ships as {getattr(getattr(grasping, block), key)!r}")
        run.note("")

        # The carrier, which the snapshot cannot see. Three rehearsal cells, differing in one
        # thing each, built and inspected and never connected.
        run.note("Three cells are built next, so the builder's own notices appear three times.")
        run.note("The line to read is `orchestrator overlays for mode=...`, which names what was")
        run.note("installed; on this tree it says `none` twice and `feasibility_config` once.")
        run.note("")
        with run.step("what the orchestrator was actually given") as report:
            installed: list[tuple[str, float, bool]] = []
            for mode_name, weight in ((_MODES[0], 0.6), (mode.value, 0.0), (mode.value, 0.6)):
                asked = robot.model_copy(update={"grasping": grasping.model_copy(update={
                    "default_mode": mode_name,
                    _WORKED: getattr(grasping, _WORKED).model_copy(
                        update={"enabled": True, operative: weight}),
                })})
                cell = Cell.rehearsal(asked)
                cell.build()
                installed.append(
                    (mode_name, weight, _carrier(cell, _WORKED_CARRIER) is not None))
            report("; ".join(f"{name} at {operative} {weight}: "
                             f"{'carrier installed' if held else 'nothing installed'}"
                             for name, weight, held in installed))
        run.note(f"All three set `{_WORKED}.enabled: true`. The first is refused by the mode, the")
        run.note("second by the weight, and only the third installs anything. That is the whole")
        run.note("trap: three conditions behind one flag, and the two that are missing are silent.")
        run.note("")

        # What the schema refuses outright. Same validator `load_config` runs, so a YAML that sets
        # one of these is refused at load rather than accepted and ignored.
        refusal = ""
        unwired = dict(RobotGraspingConfig.UNWIRED_SWITCHES)
        with run.step("a switch the schema will not accept") as report:
            path = next(iter(unwired))
            raw = grasping.model_dump()
            node: Any = raw
            *parents, leaf = path.split(".")
            for part in parents:
                node = node[part]
            node[leaf] = True
            try:
                RobotGraspingConfig.model_validate(raw)
                report(f"{path} was accepted, which it should not have been")
            except ValidationError as error:
                report(f"{path} refused at validation, as declared")
                refusal = str(error)
        if refusal:
            for line in textwrap.wrap(refusal.replace("\n", " "), 88):
                run.note(f"  {line}")
        run.note("")
        run.note(f"declared unwired, read from the code: {sorted(unwired)}")
        run.note("It refuses rather than warns because the alternative is a cell that boots, runs,")
        run.note("reports the flag as on and behaves exactly as before. That is not a crash; it is")
        run.note("a false sense of a capability. When one is wired its entry is deleted.")
        run.note("")

        # A shipped profile that turns exactly one block on, by name rather than by remembering
        # which line to toggle. `--profile decision` layers it and this reports where it landed.
        run.note("The way to turn one on for a measurement is a profile layer, not an edit to the")
        run.note("base tree: `--profile decision` stacks `config/robot/robot.decision.yaml`, which")
        run.note("sets one key. Below is where that key stands in the tree you loaded.")
        run.note("")
        layers = tuple(part for part in (args.profile or "").split(",") if part)
        for line in explain_in(tree, "robot.grasping.decision.enabled",
                               default_data_dir(), layers).render().splitlines():
            run.note(line)
        run.note("")
        run.note("Three more traps this file cannot execute, and the reference explains:")
        run.note("  the constructor wins. `build_subpolicies` prefers a hand-built policy over the")
        run.note("  config block, and the simulation runners hand one in, so toggling the key")
        run.note("  changes nothing there unless `--config-subpolicies` is passed.")
        run.note("  two things are called mode: the camera set and the grasp mode.")
        run.note("  two blocks are called recovery: `recovery` is when, `dense_recovery` is what.")
        run.note("")
        run.note("To measure one: one block per run, against a baseline in the same mode with the")
        run.note("same flags, and read the reason codes rather than the pick count. A fail-closed")
        run.note("gate is supposed to cost picks; a gate that never refuses is not a gate.")

        # This example is its own check. Both claims are load-bearing: a mode gate that stopped
        # gating, or a declared-unwired switch the schema began accepting, would leave every
        # sentence above wrong and the reader with a cell that lies about itself.
        gated = not table.get(_WORKED, (True,) * len(_MODES))[_MODES.index("easy")]
        if not gated:
            run.finding("the mode gate", f"`{_WORKED}` read as on in `easy`, which its "
                                         f"`apply_modes` excludes")
        if not refusal:
            run.finding("the declared-unwired list", "the schema accepted a switch it declares "
                                                     "unwired")
        return EXIT_FAILED if (not gated or not refusal) else run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
