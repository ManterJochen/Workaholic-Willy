"""Roll a campaign's record log up into KPIs, and read what the numbers do and do not rest on.

The other half of `real_robot/13_pick_campaign.py`. A campaign writes one record per attempt;
`RecordLog` reads that file back and computes the rates over it. It opens no cell, loads no policy
and never writes, so the same call answers for a log captured months ago at a customer's site.

The rehearsal below is what makes this runnable with nothing attached: the records are real records,
written by the real pick service, and the picks behind them are a dummy arm on a synthetic box.
"""

import tempfile
from pathlib import Path

from willy import Cell, PassRule, PickRun, RecordLog, Recording, load_tree

cell = Cell.rehearsal(load_tree("console_dummy").robot)
cell.build()

with tempfile.TemporaryDirectory() as work:
    log = str(Path(work) / "picks.jsonl")
    # At a cell this is `Recording.to_file("picks.jsonl")` on a `PickRun.from_cell(...)` over the
    # profile's own cell; nothing about the log or the rollup below differs there.
    campaign = PickRun.from_cell(cell, runs=5, recording=Recording.to_file(log),
                                 rule=PassRule(fraction=0.8))
    print(campaign.execute())

    rollup = RecordLog.from_jsonl(log).kpis()
    print(rollup.render())

    # The verdict is about the EVIDENCE, not about the cell. A rate whose denominator is empty
    # computes to a structural 0.0 and reads as a measurement of zero, so a rate these records
    # cannot measure is named as unmeasurable instead of printed. `sound` is the one thing to
    # branch on: false means the numbers are unfounded, which is different from wrong.
    print(f"\nsound: {rollup.sound}; {len(rollup.unmeasurable)} rate(s) these records cannot "
          f"measure, {len(rollup.offenders)} record(s) missing a field a KPI reads")

    # A rehearsal has no secondary verifier and no perception, so the rates that need one are
    # absent by construction. At a real cell they are present, and their absence there would be a
    # finding about the cell's telemetry rather than about the arm.
    for name in sorted(rollup.unmeasurable):
        print(f"  {name:32} {rollup.unmeasurable[name]}")

    # An unreadable log is a verdict too, not an exception: a reporting job over a path that turns
    # out to be missing says so and carries on rather than ending in a traceback.
    print(RecordLog.from_jsonl(Path(work) / "not_written.jsonl").kpis().render())
