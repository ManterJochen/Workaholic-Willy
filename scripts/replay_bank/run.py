"""Judge the bank's recorded picks again and say what today's code makes of them (``bank.py``); nothing moves.

    python scripts/replay_bank/run.py                                   # every pick of every day
    python scripts/replay_bank/run.py --days 2026-10-12 --limit 10      # a day, its first ten picks
    python scripts/replay_bank/run.py --picks 114555Z 115142Z           # picks by a part of their name
    python scripts/replay_bank/run.py --out logs/replay/after.jsonl --compare logs/replay/before.jsonl
    python scripts/replay_bank/run.py --off ground --out logs/replay/without_ground.jsonl   # a lever set aside
    python scripts/replay_bank/run.py --plan --out logs/replay/routes.jsonl  # the routes to the standoffs, cuRobo asked
    python scripts/replay_bank/run.py --push --views                    # where no grasp is admitted: push, view

Each pick is one JSON line in ``--out``: the grasps the calculator offered, how many the guard admits, the first
admitted and what refused the others. ``--compare`` sets a run before beside it, pick by pick, and exits 1 where a pick
has fewer grasps admitted than before: the bank's gate for a lever or a port.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bank  # noqa: E402


def _summary(records: "list[dict]") -> str:
    judged = [r for r in records if "skipped" not in r]
    with_grasp = [r for r in judged if r.get("admitted", 0) > 0]
    offered = [r for r in judged if r.get("offered", 0) > 0]
    causes: dict[str, int] = {}
    for record in judged:
        for kind, count in (record.get("refused") or {}).items():
            causes[kind] = causes.get(kind, 0) + int(count)
    admitted = sorted(int(r.get("admitted", 0)) for r in judged)
    lines = [f"{len(judged)} picks judged ({len(records) - len(judged)} without a recorded target): "
             f"{len(offered)} with a grasp offered, {len(with_grasp)} with one the guard admits; "
             f"grasps admitted a pick: median {admitted[len(admitted) // 2] if admitted else 0}, "
             f"total {sum(admitted)} of {sum(int(r.get('offered', 0)) for r in judged)} offered"]
    for kind, count in sorted(causes.items(), key=lambda kv: -kv[1])[:8]:
        lines.append(f"  refused {count:4d}  {kind}")
    routes = [route for r in judged for route in r.get("routes", ())]
    if routes:
        kinds: dict[str, int] = {}
        whys: dict[str, int] = {}
        for route in routes:
            kinds[route["route"]] = kinds.get(route["route"], 0) + 1
            if route["route"] not in ("straight", "planned"):
                why = bank._cause(route.get("route_why", route["route"]))
                whys[why] = whys.get(why, 0) + 1
        seconds = sorted(float(route.get("route_s", 0.0)) for route in routes)
        routed = [r for r in judged if r.get("routed", 0) > 0]
        lines.append(f"routes to the standoff: {len(routes)} asked, "
                     + ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))
                     + f"; {len(routed)} picks with a route; seconds a route: median "
                     f"{seconds[len(seconds) // 2]:.2f}, max {seconds[-1]:.2f}")
        for kind, count in sorted(whys.items(), key=lambda kv: -kv[1])[:6]:
            lines.append(f"  route refused {count:4d}  {kind}")
    plans = [plan for r in judged for plan in r.get("plans", ())]
    if plans:
        kinds = {}
        for plan in plans:
            kinds[plan["plan"]] = kinds.get(plan["plan"], 0) + 1
        timed = sorted(float(plan.get("plan_s", 0.0)) for plan in plans if "plan_s" in plan)
        lengths = sorted(float(plan["plan_rad"]) for plan in plans if "plan_rad" in plan)
        lines.append(f"plans from home: {len(plans)} asked, " + ", ".join(
            f"{count} {kind}" for kind, count in sorted(kinds.items()))
            + (f"; seconds a plan: median {timed[len(timed) // 2]:.2f}, max {timed[-1]:.2f}" if timed else "")
            + (f"; joint path: median {lengths[len(lengths) // 2]:.2f} rad" if lengths else ""))
        refused_by: dict[str, int] = {}
        for plan in plans:
            if plan["plan"] == "refused by the guard":
                why = bank._cause(plan.get("plan_why", ""))
                refused_by[why] = refused_by.get(why, 0) + 1
        for kind, count in sorted(refused_by.items(), key=lambda kv: -kv[1])[:6]:
            lines.append(f"  plan refused by the guard {count:4d}  {kind}")
    return "\n".join(lines)


def _compare(records: "list[dict]", before_path: Path) -> int:
    before = {r["pick"]: r for r in (json.loads(line) for line in before_path.read_text(encoding="utf-8").splitlines()
                                     if line.strip())}
    worse = better = 0
    for record in records:
        old = before.get(record["pick"])
        if old is None or "skipped" in record:
            continue
        now, then = int(record.get("admitted", 0)), int(old.get("admitted", 0))
        if now != then:
            print(f"  {record['pick']}: {then} -> {now} admitted")
        worse += now < then
        better += now > then
        if "routed" in record and "routed" in old and record["routed"] != old["routed"]:
            print(f"  {record['pick']}: {old['routed']} -> {record['routed']} routed")
        if "planned" in record and "planned" in old and record["planned"] != old["planned"]:
            print(f"  {record['pick']}: {old['planned']} -> {record['planned']} planned from home and admitted")
    print(f"== against {before_path.name}: {better} pick(s) with more grasps admitted, {worse} with fewer")
    return 1 if worse else 0


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bank", type=Path, default=bank.BANK)
    parser.add_argument("--config", type=Path, default=None, help="the cell's config tree (<bank>/robominds_config)")
    parser.add_argument("--days", nargs="*", default=())
    parser.add_argument("--picks", nargs="*", default=(), help="parts of pick names to keep")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("logs/replay_bank/latest.jsonl"))
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--off", nargs="*", default=(), choices=bank.LEVERS,
                        help="levers to run without, to set them beside the code with them")
    parser.add_argument("--plan", action="store_true",
                        help="route the first admitted grasps to their standoffs with the cell's planner (GPU)")
    parser.add_argument("--plan-from-home", action="store_true",
                        help="plan them with cuRobo from the cell's home too, no straight line, the guard judging (GPU)")
    parser.add_argument("--push", action="store_true",
                        help="where no grasp is admitted, the push of the part and of the scene the pick would plan")
    parser.add_argument("--views", action="store_true",
                        help="where no grasp is admitted, the view the pick would generate (the instant controller)")
    parser.add_argument("--blockers", action="store_true",
                        help="where no grasp is admitted, the neighbour the pick would set aside, its grasp and the "
                             "part's once it is gone, judged by the guard")
    parser.add_argument("--set", nargs="*", default=(), metavar="KEY=VALUE",
                        help="config values in memory for this run (LoadedTree.with_values), a value read as YAML")
    args = parser.parse_args(argv)
    import yaml  # noqa: PLC0415 (a value as the tree's own files would write it)

    values = {key: yaml.safe_load(value) for key, _, value in (item.partition("=") for item in args.set)}

    logging.basicConfig(level=logging.ERROR)
    paths = bank.recorded_picks(args.bank, tuple(args.days))
    if args.picks:
        paths = [p for p in paths if any(part in p.name for part in args.picks)]
    if args.limit:
        paths = paths[:args.limit]
    cell = bank.ReplayCell(args.config or args.bank / "robominds_config" / "config", off=tuple(args.off),
                           plan=args.plan, plan_from_home=args.plan_from_home, values=values, push=args.push,
                           views=args.views, blockers=args.blockers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()
    with args.out.open("w", encoding="utf-8") as out:
        for index, path in enumerate(paths, 1):
            try:
                record = bank.replay(cell, path)
            except Exception as exc:  # noqa: BLE001 (one recording the bank cannot read is said, the run goes on)
                record = {"pick": path.stem, "day": path.parent.parent.name, "skipped": f"{type(exc).__name__}: {exc}"}
            records.append(record)
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"[{index}/{len(paths)}] {record['pick'][-32:]}: "
                  + (record["skipped"] if "skipped" in record else
                     f"{record.get('admitted', 0)}/{record.get('offered', 0)} admitted"
                     + (f", beside {record.get('parts_beside')}" if record.get("parts_beside") else "")
                     + (f" ({record.get('no_grasp')})" if not record.get("offered") else "")
                     + (f", routed {record['routed']}/{len(record['routes'])}" if "routes" in record else "")
                     + (f", planned {record['planned']}/{len(record['plans'])}" if "plans" in record else "")),
                  flush=True)
    cell.close()
    print(f"== {time.perf_counter() - started:.0f} s; {args.out}")
    print(_summary(records))
    return _compare(records, args.compare) if args.compare is not None else 0


if __name__ == "__main__":
    sys.exit(main())
