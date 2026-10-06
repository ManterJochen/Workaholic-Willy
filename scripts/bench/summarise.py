"""One line per scene of a bench run: what it came to and the first reason it gives, for reading failures fast.

    python scripts/bench/summarise.py logs/bench/<run>/bench_result.json [--fails]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _first_reason(row: dict[str, Any]) -> str:
    for attempt in row.get("attempts", []) or []:
        for t in attempt.get("tries", []) or []:
            message = t.get("motion_message") or ""
            if message:
                return f"try: {message}"[:220]
        if attempt.get("motion_message"):
            return f"motion: {attempt['motion_message']}"[:220]
    said = (row.get("calculator") or {}).get("no_grasp_said")
    if said:
        return f"calculator: {said}"[:220]
    return str(row.get("why") or "")[:220]


def _pushes(row: dict[str, Any]) -> str:
    return ",".join(str(p.get("code")) for p in row.get("pushes", []) or [])


def _blockers(row: dict[str, Any]) -> str:
    out = []
    for b in row.get("blockers", []) or []:
        out.append(str(b.get("code")) if isinstance(b, dict) else str(b)[:40])
    return ",".join(out)


def main(argv: list[str]) -> int:
    path = Path(argv[0])
    fails_only = "--fails" in argv
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = list(data.get("scenes", {}).values())
    possible = [r for r in rows if r.get("possible", True)]
    print(f"{sum(1 for r in possible if r.get('success'))}/{len(possible)} possible scenes picked "
          f"({len(rows)} run)")
    for r in rows:
        if fails_only and r.get("success"):
            continue
        actions = "/".join(str(a.get("action")) for a in r.get("attempts", []) or [])
        if r.get("mode") == "clear":
            print(f"{r['scene']:<28} {r.get('outcome')!s:<16} {r.get('seconds')}s")
            for i, p in enumerate(r.get("picks", []) or []):
                if p.get("held"):
                    continue
                print(f"    pick {i}: {p.get('outcome')} | {_first_reason(p)} | push {_pushes(p)} | "
                      f"blocker {_blockers(p)}")
            continue
        state = "OK " if r.get("success") else ("-- " if not r.get("possible", True) else "XX ")
        print(f"{state}{r['scene']:<26} {r.get('outcome')!s:<20} {actions:<34} push[{_pushes(r)}] "
              f"blk[{_blockers(r)}] {r.get('seconds')}s")
        if not r.get("success"):
            print(f"      {_first_reason(r)}")
            if r.get("raised"):
                print(f"      raised: {r['raised']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
