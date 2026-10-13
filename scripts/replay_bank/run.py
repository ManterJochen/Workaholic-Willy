"""Judge the bank's recorded picks again and say what today's code makes of them (``bank.py``); nothing moves.

    python scripts/replay_bank/run.py                                   # every pick of every day
    python scripts/replay_bank/run.py --days 2026-10-12 --limit 10      # a day, its first ten picks
    python scripts/replay_bank/run.py --picks 114555Z 115142Z           # picks by a part of their name
    python scripts/replay_bank/run.py --out logs/replay/after.jsonl --compare logs/replay/before.jsonl
    python scripts/replay_bank/run.py --off ground --out logs/replay/without_ground.jsonl   # a lever set aside

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
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.ERROR)
    paths = bank.recorded_picks(args.bank, tuple(args.days))
    if args.picks:
        paths = [p for p in paths if any(part in p.name for part in args.picks)]
    if args.limit:
        paths = paths[:args.limit]
    cell = bank.ReplayCell(args.config or args.bank / "robominds_config" / "config", off=tuple(args.off))
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
                     + (f" ({record.get('no_grasp')})" if not record.get("offered") else "")), flush=True)
    print(f"== {time.perf_counter() - started:.0f} s; {args.out}")
    print(_summary(records))
    return _compare(records, args.compare) if args.compare is not None else 0


if __name__ == "__main__":
    sys.exit(main())
