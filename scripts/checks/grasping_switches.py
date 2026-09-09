"""Does the mode gate still gate, and does the schema still refuse a switch it declares unwired?

    python scripts/checks/grasping_switches.py

⛔ **NOTHING ELSE IN THIS REPOSITORY PUBLISHES THIS.** No CLI and no library noun renders an
`EffectiveGraspingConfig` or a mode-gate table, so "which `robot.grasping.*` block is reachable in
which grasp mode" is a question only a sweep answers, and the pytest suite answers it for fixtures
rather than for the tree THIS box loads. A block outside the running mode's `apply_modes` collapses
to its off value even when the YAML says true, which is a config decision no fixture can see.

Each block that carries an `enabled` flag is switched on one at a time, the effective-config
snapshot is rebuilt in every mode `GraspMode` declares, and the row is the diff against an
all-default snapshot in the same mode. The expectation for each row is read off that block's own
`apply_modes` rather than written down here, so a cell that widens a list is measured against its
own new list and this check cannot go stale against a config it has never seen.

The traps the sweep cannot execute (the constructor that wins over the config block, the two things
called mode, the two blocks called recovery) are in `docs/grasping-config-reference.md` sections 2
and 3.

Exit codes: 0 every mode-specific block stayed inside its own `apply_modes` and every
declared-unwired switch was refused, 1 one did not, 2 this tree has no grasping block to sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path
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
    baseline = {mode: _snapshot(grasping, mode) for mode in modes}

    # The sweep: one block on at a time, every mode, diffed against the all-default snapshot in the
    # same mode. Nothing is asserted from the documentation; the table is measured.
    reads_on: dict[str, tuple[GraspMode, ...]] = {}
    for block in blocks:
        switched = grasping.model_copy(update={
            block: getattr(grasping, block).model_copy(update={"enabled": True})})
        hits = []
        for mode in modes:
            moved = _snapshot(switched, mode)
            if any(moved[key] != baseline[mode][key] for key in moved):
                hits.append(mode)
        reads_on[block] = tuple(hits)

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

    print(f"switch each block on, rebuild the snapshot, diff it "
          f"(`*` = the {here.value} mode this tree runs)\n")
    widths = {mode: max(len(mode.value) + 2, 11) for mode in modes}
    head = "".join(f"{mode.value + ('*' if mode is here else ''):<{widths[mode]}}" for mode in modes)
    print(f"  {'block':<21}{head}gate")
    for block in blocks:
        row = "".join(f"{('reads on' if mode in reads_on[block] else 'no change'):<{widths[mode]}}"
                      for mode in modes)
        print(f"  {block:<21}{row}{gate(block)}")

    wrong: list[str] = []
    for block in blocks:
        allowed = declared(block)
        if allowed is None or gate(block) != "gated":
            continue
        outside = [mode.value for mode in reads_on[block] if mode.value not in allowed]
        if outside:
            wrong.append(f"{block} reads on in {', '.join(outside)}, which its own apply_modes "
                         f"({', '.join(allowed)}) excludes")
        elif not reads_on[block] and not bool(getattr(grasping, block).enabled):
            # Off in the tree, switched on, and no column moved: either the gate now refuses every
            # mode or the block left the snapshot. Both make its row above measure nothing.
            wrong.append(f"{block} is off in this tree and switching it on moved no key in any of "
                         f"{', '.join(allowed)}")

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
    print(f"\nOK: {len(blocks)} block(s) swept across {len(modes)} modes, every mode-specific one "
          f"stayed inside its own apply_modes, {len(unwired)} declared-unwired switch(es) refused")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
