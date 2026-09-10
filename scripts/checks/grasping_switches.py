"""Does the mode gate still gate, and does the schema still refuse a switch it declares unwired?

    python scripts/checks/grasping_switches.py

⛔ **NOTHING ELSE IN THIS REPOSITORY PUBLISHES THIS.** No CLI and no library noun renders an
`EffectiveGraspingConfig` or a mode-gate table, so "which `robot.grasping.*` block is reachable in
which grasp mode" is a question only a sweep answers, and the pytest suite answers it for fixtures
rather than for the tree THIS box loads. A block outside the running mode's `apply_modes` collapses
to its off value even when the YAML says true, which is a config decision no fixture can see.

Each block that carries an `enabled` flag is switched on one at a time, two instruments are rebuilt
in every mode `GraspMode` declares, and the row is the diff against an all-default reading in the
same mode. The expectation for each row is read off that block's own `apply_modes` rather than
written down here, so a cell that widens a list is measured against its own new list and this check
cannot go stale against a config it has never seen.

The second instrument exists because one of them measured a block as inert that is not. A
`grasping.*` block reaches a running cell through the effective-config snapshot or through
`apply_orchestrator_overlays`, and `deep_ranker` reaches only the second: switching it on loads the
fitted ranker under `grasping.deep_ranker.artifact_dir` onto the pick loop, which then scores every
candidate and stamps `deep_ranker_*` telemetry. With the snapshot alone this sweep printed "no
change" in all five modes and closed with "13 block(s) swept". A row that cannot move is not
evidence of a switch that does nothing, and the two were spelled identically.

A third state is printed for the same reason. On a fresh checkout the ranker trees are not in the
repository (`assets/models/grasp_ranker/**/*.json` is git-ignored; the cards beside them are not),
so the overlay fail-safes to `None` and the block moves nothing here either. That is a fact about
this box and not a claim about the config, so it is reported as `unloaded`, quoting the sentence the
shipped code itself logged, rather than counted as a switch that does nothing.

The traps the sweep cannot execute (the constructor that wins over the config block, the two things
called mode, the two blocks called recovery) are in `docs/grasping-config-reference.md` sections 2
and 3.

Exit codes: 0 every block moved something under at least one instrument or said why it could not,
every mode-specific one stayed inside its own `apply_modes`, and every declared-unwired switch was
refused; 1 one of those did not; 2 this tree has no grasping block to sweep.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import ValidationError  # noqa: E402

from src.config import ConfigError, load_robot_config  # noqa: E402
from src.config.schema.robot.grasping_schema import RobotGraspingConfig  # noqa: E402
from src.robot.execution.autonomous_grasp import (  # noqa: E402
    GraspMode,
    resolve_grasp_mode,
)
from src.robot.execution.autonomous_grasp.builders import (  # noqa: E402
    apply_orchestrator_overlays,
    build_effective_config,
)

EXIT_OK, EXIT_FAILED, EXIT_NOT_READY = 0, 1, 2


def _not_ready(what: str, fix: str) -> int:
    print(f"NOT READY: {what}\n  fix: {fix}")
    return EXIT_NOT_READY


def _snapshot(grasping: RobotGraspingConfig, mode: GraspMode) -> dict[str, Any]:
    """The flat telemetry snapshot a cell would report for ``grasping`` resolved in ``mode``.

    ``build_effective_config`` answers ``None`` only when handed no grasping block at all, which is
    the legacy ``mode=`` override path and not this one.
    """
    effective = build_effective_config(grasping, resolved_mode=mode, resolved_max_attempts=5)
    if effective is None:  # pragma: no cover (unreachable with a real grasping block)
        raise RuntimeError("build_effective_config returned no snapshot for a real block")
    return effective.to_dict()


class _RecordedSlots:
    """Every attribute the overlay pass sets on an orchestrator, and nothing else.

    A stand-in for the real thing. A ``RuntimePickService`` needs an arm, a camera and a calculator,
    and this check touches no hardware, so the sweep hands ``apply_orchestrator_overlays`` an object
    that records what it writes and answers ``None`` to every read. The overlays that read a slot
    back (``frame_resolver``) then take the branch a cell with no resolver takes anyway, which is
    the one this box would take.

    What is measured is therefore the shipped overlay pass itself, not a restatement of it: a block
    that stops reaching the orchestrator stops moving a slot here on the same day.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "written", {})

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.written.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self.written[name] = value


class _CapturedWarnings:
    """Take every WARNING the shipped code emits during the sweep, and print none of them.

    The sweep calls the overlay pass once per block per mode, and that pass says one true thing per
    call about this cell's camera count, plus one per ranker artifact it tries to load. Printed,
    that is a hundred-odd lines around a thirteen-row table. Discarded, so is the only account the
    shipped code gives of why a block could not move anything, which is the distinction this check
    exists to draw. So they are kept instead.

    The records are taken at ``Logger.handle``, the one seam every logger reaches whatever handlers
    it carries and whether or not it propagates, and are not passed on. The seam is put back on the
    way out of the ``with``, before anything else prints, so a refusal below is still visible.
    """

    def __init__(self) -> None:
        self.seen: list[str] = []
        self._original = logging.Logger.handle

    def __enter__(self) -> "_CapturedWarnings":
        def handle(logger: logging.Logger, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                self.seen.append(record.getMessage())

        logging.Logger.handle = handle  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: object) -> None:
        logging.Logger.handle = self._original  # type: ignore[method-assign]


def _orchestrator(grasping: RobotGraspingConfig, mode: GraspMode) -> dict[str, str]:
    """The orchestrator slots a cell would carry for ``grasping`` resolved in ``mode``.

    ``repr`` rather than the objects: the question is only whether a slot moved, the carriers are
    frozen config dataclasses with no ``__eq__`` worth relying on, and one of them is a loaded
    gradient-boosted ranker whose equality is not a thing to ask for.
    """
    runtime = SimpleNamespace(orchestrator=_RecordedSlots())
    apply_orchestrator_overlays(runtime, grasping, resolved_mode=mode)  # type: ignore[arg-type]
    return {name: repr(value) for name, value in runtime.orchestrator.written.items()}


def main() -> int:
    try:
        config = load_robot_config()
    except ConfigError as error:
        return _not_ready(f"the config tree has no cell to sweep ({error})",
                          "select a profile that has a `robot` block: WILLY_PROFILE=ur5e")

    grasping = config.grasping
    # Read off the model rather than written down here, so a block added to the schema is swept
    # without anyone remembering to update a list. `occlusion` is deliberately absent: it carries
    # `directional_enabled` and `hard_reject_enabled` instead of one `enabled`.
    blocks = [name for name in type(grasping).model_fields
              if hasattr(getattr(grasping, name, None), "enabled")]
    if not blocks:
        return _not_ready("no block under `robot.grasping` carries an `enabled` flag in this tree",
                          "point the loader at a tree that has one; `config/robot/robot.yaml` in "
                          "this repository is one")

    here = resolve_grasp_mode(grasping.default_mode)
    modes = tuple(GraspMode)

    reads_on: dict[str, tuple[GraspMode, ...]] = {}
    reaches_runtime: dict[str, tuple[GraspMode, ...]] = {}
    #: block -> what the shipped code logged the first time switching it on explained itself.
    unloaded: dict[str, str] = {}
    with _CapturedWarnings() as captured:
        baseline = {mode: _snapshot(grasping, mode) for mode in modes}
        mark = len(captured.seen)
        baseline_slots = {mode: _orchestrator(grasping, mode) for mode in modes}
        # What the overlay pass says about this cell before any switch is touched. Subtracted
        # below, so a line that appears whatever the sweep does is not attributed to a block.
        always = set(captured.seen[mark:])

        # The sweep: one block on at a time, every mode, diffed against the all-default reading in
        # the same mode. Nothing is asserted from the documentation; the table is measured. Both
        # instruments are read, because a block reaches a running cell through either of them and a
        # block invisible to one of them is not a block that does nothing.
        for block in blocks:
            switched = grasping.model_copy(update={
                block: getattr(grasping, block).model_copy(update={"enabled": True})})
            hits, runtime_hits = [], []
            for mode in modes:
                moved = _snapshot(switched, mode)
                if any(moved[key] != baseline[mode][key] for key in moved):
                    hits.append(mode)
                mark = len(captured.seen)
                slots = _orchestrator(switched, mode)
                if any(slots.get(key) != baseline_slots[mode].get(key)
                       for key in set(slots) | set(baseline_slots[mode])):
                    runtime_hits.append(mode)
                said = [line for line in captured.seen[mark:] if line not in always]
                if said and block not in unloaded:
                    unloaded[block] = said[0]
            reads_on[block] = tuple(hits)
            reaches_runtime[block] = tuple(runtime_hits)
    # A block that moved something needs no excuse, and one that said something while moving a slot
    # was talking about something else.
    unloaded = {block: line for block, line in unloaded.items()
                if not reads_on[block] and not reaches_runtime[block]}

    def declared(block: str) -> tuple[str, ...] | None:
        """The block's own `apply_modes`, which is what declares it mode-specific."""
        modes_declared = getattr(getattr(grasping, block), "apply_modes", None)
        return None if modes_declared is None else tuple(modes_declared)

    def gate(block: str) -> str:
        """Where a mode-specific block's filter is enforced, read off the snapshot.

        Two shapes ship. Either `build_effective_config` enforces the filter and the block collapses
        to its off value outside its list (`gated`), or the snapshot carries the list on as
        `<block>_apply_modes` for the consumer to enforce (`deferred`), and then the flag reading
        through in every mode is the design rather than a hole. Only `gated` blocks are held to the
        list below, because only they claim to enforce it here.
        """
        if declared(block) is None:
            return "ungated"
        return "deferred" if f"{block}_apply_modes" in baseline[here] else "gated"

    def cell(block: str, mode: GraspMode) -> str:
        """What one switch did to one mode, in the words of whichever instrument saw it.

        `runtime` is the answer the snapshot alone could not give, and it is not a weaker `reads
        on`: it is the block reaching the built cell through the orchestrator instead of the
        telemetry contract. Spelling both `no change` is what made a loaded ranker read as an inert
        flag. `unloaded` is the third one: the block reaches the orchestrator and its carrier is not
        on this box, which the shipped code said and this check now repeats under the table.
        """
        if mode in reads_on[block]:
            return "reads on"
        if mode in reaches_runtime[block]:
            return "runtime"
        return "unloaded" if block in unloaded else "no change"

    print(f"switch each block on, rebuild both readings, diff them "
          f"(`*` = the {here.value} mode this tree runs)\n")
    widths = {mode: max(len(mode.value) + 2, 11) for mode in modes}
    head = "".join(f"{mode.value + ('*' if mode is here else ''):<{widths[mode]}}" for mode in modes)
    print(f"  {'block':<21}{head}gate")
    for block in blocks:
        row = "".join(f"{cell(block, mode):<{widths[mode]}}" for mode in modes)
        print(f"  {block:<21}{row}{gate(block)}")
    print("\n  reads on = a key of the effective-config snapshot moved.  runtime = only an "
          "orchestrator\n  slot moved, which is the other place a block reaches and the one the "
          "snapshot cannot see.")
    if unloaded:
        print("  unloaded = the block reaches the orchestrator and its carrier is not on this box,"
              "\n  in the words of the shipped code:")
        for block, line in unloaded.items():
            print(f"    {block}: {line}")

    wrong: list[str] = []
    for block in blocks:
        # A block that moved nothing anywhere was never measured, and the row printed `no change`
        # about it, which reads as a switch with no work to do. That is the shape `deep_ranker` had
        # until the orchestrator half existed. Checked for every block rather than only the gated
        # ones, because being outside an instrument has nothing to do with having an apply_modes.
        # A block the shipped code explained is reported above instead: an absent artifact is a
        # fact about this box, not a claim this cell's config failed to keep.
        if (not reads_on[block] and not reaches_runtime[block]
                and not bool(getattr(grasping, block).enabled)
                and block not in unloaded):
            wrong.append(f"{block} is off in this tree, and switching it on moved neither a "
                         f"snapshot key nor an orchestrator slot in any mode, and said nothing "
                         f"about why, so its row measures nothing and this sweep cannot speak "
                         f"for it")
            continue
        allowed = declared(block)
        if allowed is None or gate(block) != "gated":
            continue
        outside = [mode.value for mode in reads_on[block] if mode.value not in allowed]
        if outside:
            # The snapshot half only. `build_effective_config` is where a `gated` block claims to
            # enforce its list; the overlay half carries the list on for its consumer instead (the
            # `deferred` shape), so holding it to the same rule would report a design as a defect.
            wrong.append(f"{block} reads on in {', '.join(outside)}, which its own apply_modes "
                         f"({', '.join(allowed)}) excludes")

    # The second claim, and the same validator `load_config` runs: a switch that would reach the
    # snapshot and be read back by nothing is refused at load, not accepted and ignored.
    raw = grasping.model_dump()
    try:
        RobotGraspingConfig.model_validate(raw)
    except ValidationError as error:
        return _not_ready(f"this tree's own grasping block does not survive a round trip through "
                          f"its schema ({str(error).splitlines()[0]})",
                          "the tamper below could not tell its own refusal from that one; fix the "
                          "tree or the schema first")

    unwired = dict(RobotGraspingConfig.UNWIRED_SWITCHES)
    print(f"\nset each declared-unwired switch and re-validate ({len(unwired) or 'none'} declared)\n")
    for path in unwired:
        tampered = grasping.model_dump()
        node: Any = tampered
        *parents, leaf = path.split(".")
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict) or leaf not in node:
            wrong.append(f"{path} is declared unwired but no longer exists in the schema")
            print(f"  {path:<38} GONE      <- declared unwired, not in the model")
            continue
        node[leaf] = True
        try:
            RobotGraspingConfig.model_validate(tampered)
        except ValidationError as error:
            if path in str(error):
                print(f"  {path:<38} REFUSED   -> {unwired[path]} never reaches the pick path")
            else:
                wrong.append(f"{path} was refused, but the refusal does not name it")
                print(f"  {path:<38} REFUSED   <- for some other reason than this switch")
        else:
            wrong.append(f"{path} was accepted, and it is declared unwired")
            print(f"  {path:<38} ACCEPTED  <- a cell would report a capability it does not have")

    if wrong:
        print(f"\nFAILED: {len(wrong)} claim(s) this cell's own config makes did not hold")
        for line in wrong:
            print(f"  {line}")
        return EXIT_FAILED
    print(f"\nOK: {len(blocks)} block(s) swept across {len(modes)} modes on both instruments, "
          f"{len(blocks) - len(unloaded)} of them moved something somewhere and {len(unloaded)} "
          f"said why not, every mode-specific one stayed inside its own apply_modes, "
          f"{len(unwired)} declared-unwired switch(es) refused")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
