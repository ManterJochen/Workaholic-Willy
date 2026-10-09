"""Does SFE build a closing line's grasps at once, on THIS machine, exactly as it builds them one at a time?

    python scripts/checks/batched_builds.py

``robot.grasping.batched_builds`` makes the support-footprint stage build each closing line's grasps at once, in one
numpy pass per check (``support_footprint._build_many``), with the same candidates, refusal counts and stages to the
bit. That holds where numpy hands each item of a stacked product to the BLAS call it hands one product to, and every
number a verdict reads over many builds stands more than 1e-6 off its threshold or is left to ``_build`` alone. It was
proved on the desk (2026-10-09); another machine's BLAS may take other paths, so a cell switches the key on only once
this check passes on its own PC. It runs on the CPU only: it opens no device and moves nothing, and it silences the
calculator's log lines below a warning, so that a cell's log keeps only what its picks said.

What it compares, every grasp field by field (a signed zero included), the refusal counts with their order, and the
stages:

1. numpy's stacked products against one at a time, on the shapes the hand of the tree this box loads stacks;
2. the five scenes of ``tests/test_sfe_units_in_any_order_give_the_search_its_answer.py`` (an open cube, a bar boxed in
   with the camera's boxes and the floor the guard holds, a cylinder beside a wall seen from a camera that misses one
   side, a thin bar with its own low fragment and a declared wall, a round footprint) under every switch the search
   has, with their own Hand-E and with this tree's hand;
3. the owner's two recorded Zollstock looks of 2026-10-01 through the calculator the shipped ``hande`` tree builds, the
   key off and on: candidates, reasons, counts and every telemetry key;
4. 20,000 builds jittered off the scenes' lines, each against ``_build`` alone.

And how long each search takes here, one at a time and at once.

Exit codes: 0 every answer the same: ``batched_builds: true`` may go into this cell's tree; 1 one differs: keep it off
and send this output; 2 this checkout cannot run the check (its ``tests`` folder, whose scenes and recorded looks it
reads, is missing).
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from src.config import load_robot_config  # noqa: E402
from src.robot.grasping.generation import support_footprint as sf  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_NOT_READY = 0, 1, 2
#: How many builds the per-build comparison makes, about.
_BUILDS = 20_000


def _bits(value: Any) -> Any:
    """``value`` as plain data, every float and array by its bits: equal means the same bits in the same order."""
    if isinstance(value, np.ndarray):
        return ["array", value.dtype.str, list(value.shape), value.tobytes().hex()]
    if isinstance(value, (float, np.floating)):
        return ["float", np.float64(value).tobytes().hex()]
    if isinstance(value, dict):
        return ["dict", [[_bits(k), _bits(v)] for k, v in value.items()]]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_bits(v) for v in value]]
    if dataclasses.is_dataclass(value):
        return [type(value).__name__, [[f.name, _bits(getattr(value, f.name))] for f in dataclasses.fields(value)]]
    return repr(value)


def _search(cloud: np.ndarray, keywords: dict[str, Any], *, batched: bool) -> tuple[Any, float]:
    """One search: its grasps, counts and stages as bits, and how long it took, seconds."""
    counts: dict[str, int] = {}
    stages: dict[str, str] = {}
    started = time.perf_counter()
    found = sf.generate_support_footprint_grasps(cloud, refusals=counts, stages=stages, batched=batched, **keywords)
    took = time.perf_counter() - started
    return _bits([found, counts, list(counts), stages]), took


def _switches(keywords: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every switch the search has that changes something for a scene with ``keywords``."""
    out: dict[str, dict[str, Any]] = {"as it is": {}, "palm aware": {"palm_aware": True},
                                      "inflated 2 mm": {"inflate_mm": 2.0},
                                      "the fine pass left for later": {"fine_pass": False}}
    out["side approaches " + ("off" if keywords.get("side_approaches") else "on")] = {
        "side_approaches": not keywords.get("side_approaches", False)}
    for key, what in (("corridor_seen", "no seen test"), ("seen_envelope", "no camera boxes"),
                      ("hand_floor", "no floor")):
        if keywords.get(key) is not None:
            out[what] = {key: None}
    return out


def main() -> int:
    logging.disable(logging.INFO)
    try:
        from tests.test_a_lines_builds_made_at_once_answer_to_the_bit import zollstock
        from tests.test_a_pool_of_workers_grasps_as_one_process_does import _result
        from tests.test_sfe_units_in_any_order_give_the_search_its_answer import scenes
    except ImportError as error:
        print(f"NOT READY: this checkout has no tests folder to read the scenes from ({error})\n"
              "  fix: run it from a full checkout of the repository, tests/ and tests/data/ included")
        return EXIT_NOT_READY

    wrong: list[str] = []
    cfg = load_robot_config()
    try:
        jaw: "sf.SupportFootprintJaw | None" = sf.SupportFootprintJaw.from_robot_config(cfg)
    except ValueError as error:
        print(f"this tree's hand is not a parallel jaw ({error}): the scenes are searched with the Hand-E only")
        jaw = None

    print("1. numpy's stacked products against one at a time")
    why = sf.batched_builds_hold(jaw)
    print(f"  {'the same bits' if not why else why}")
    if why:
        wrong.append(why)

    print("\n2. five scenes under every switch: one at a time / at once, seconds")
    alone_s, at_once_s, searches = 0.0, 0.0, 0
    for name, (cloud, keywords) in scenes().items():
        hands: list[tuple[str, dict[str, Any]]] = [("Hand-E", {})]
        if jaw is not None:
            hands.append(("this tree's hand", {"jaw": jaw}))
        for hand, own in hands:
            for switch, more in _switches(keywords).items():
                asked = {**keywords, **own, **more}
                expected, took_alone = _search(cloud, asked, batched=False)
                got, took_at_once = _search(cloud, asked, batched=True)
                same = expected == got
                searches += 1
                alone_s += took_alone
                at_once_s += took_at_once
                if not same:
                    wrong.append(f"{name}, {hand}, {switch}: the grasps, counts or stages differ")
                print(f"  {name:32s} {hand:16s} {switch:30s} {took_alone:7.3f} / {took_at_once:7.3f}"
                      f"{'' if same else '   DIFFERS'}")

    print("\n3. the recorded Zollstock looks through the cell's calculator: key off / on, seconds")
    zollstock("P1", batched=False)   # once untimed: the first calculator imports what every later one finds loaded
    for name in ("P1", "P5"):
        started = time.perf_counter()
        off, _ = zollstock(name, batched=False)
        took_off = time.perf_counter() - started
        started = time.perf_counter()
        on, calculator = zollstock(name, batched=True)
        took_on = time.perf_counter() - started
        same = _result(off) == _result(on) and bool(calculator.sfe_batched)
        if not same:
            wrong.append(f"the recorded look {name}: the calculator's result differs, or the key did not reach it")
        print(f"  {name}  {took_off:7.3f} / {took_on:7.3f}{'' if same else '   DIFFERS'}")

    print(f"\n4. about {_BUILDS} builds jittered off the scenes' lines, each against _build alone")
    differ, builds = _every_build(scenes())
    print(f"  {builds} builds, {differ} differ")
    if differ:
        wrong.append(f"{differ} of {builds} builds differ from _build alone")

    if wrong:
        print(f"\nFAIL: {len(wrong)} answer(s) differ on this machine: keep robot.grasping.batched_builds off, "
              "and send this output")
        for line in wrong:
            print(f"  {line}")
        return EXIT_FAILED
    print(f"\nPASS: {searches} searches, 2 recorded looks and {builds} builds answer to the bit at once as one at a "
          f"time on this machine ({alone_s:.2f} s one at a time, {at_once_s:.2f} s at once): "
          "robot.grasping.batched_builds may be switched on here")
    return EXIT_OK


def _every_build(cases: dict[str, tuple[np.ndarray, dict[str, Any]]]) -> tuple[int, int]:
    """How many of about :data:`_BUILDS` builds jittered off every search's lines differ from ``_build`` alone, and how
    many were made: their verdicts, causes and numbers."""
    rng = np.random.default_rng(20261012)
    differ = builds = 0
    rounds = 0
    while builds < _BUILDS and rounds < 20:
        rounds += 1
        for cloud, keywords in cases.values():
            inputs = sf.SfeInputs(cloud, counting=True, batched=True, **keywords)
            plan = sf.plan_support_footprint(inputs)
            if plan is None:
                continue
            obstacles, side, _every_tilt = plan.grids()
            for search in (sf.COARSE, sf.FINE, sf.ROLLED):
                for axis_index in range(len(plan.searches[search][0])):
                    for frac_index in range(len(sf._FRACS)):
                        line = plan.line(search, axis_index, frac_index)
                        if line is None:
                            continue
                        ladder = plan.ladder(search, axis_index, line.axis)
                        anchor = (np.array([line.mid[0], line.mid[1], rng.uniform(plan.prism.z0 + 1.0, plan.prism.z1)])
                                  + rng.normal(scale=4.0, size=3))
                        anchors = np.repeat(anchor[None, :], ladder.size, axis=0)
                        many, causes = sf._build_many(plan, anchors, ladder, np.arange(ladder.size), counting=True)
                        for row in range(ladder.size):
                            counted: dict[str, int] = {}
                            alone = sf._build(plan.prism, anchors[row].copy(), ladder.axis[row],
                                              ladder.approach[row], plan.jaw, obstacles, inputs.support_height_mm,
                                              palm_aware=inputs.palm_aware, score_weights=inputs.score_weights,
                                              refusals=counted, side=side, tilt_deg=ladder.off_vertical[row],
                                              seen=inputs.seen_envelope, floor=inputs.hand_floor)
                            builds += 1
                            if (_bits(alone) != _bits(many[row])
                                    or tuple(k for k, n in counted.items() for _ in range(n)) != causes[row]):
                                differ += 1
    return differ, builds


if __name__ == "__main__":
    raise SystemExit(main())
