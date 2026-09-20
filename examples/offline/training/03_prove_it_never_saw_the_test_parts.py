"""Prove a trained generator never saw the parts it is measured on: the denominator a held-out claim rests on.

held_out_assets asks a different question than counting files on disk: only what the mesh bank can
actually place counts, and only the assets whose fold group never appeared in the training corpus
count as unseen. A generator "tested on held-out objects" whose held-out set turns out empty was
tested on nothing.
"""

import tempfile
from pathlib import Path

from willy import DatasetBuild
from datagen.heldout import format_report, held_out_assets

with tempfile.TemporaryDirectory() as work:
    corpus = Path(work) / "clouds"
    # The generated-shapes-only default trains on no named mesh at all, so every gso/ycb asset on
    # this machine reads as unseen: the honest answer for a corpus that placed none of them.
    dataset = DatasetBuild.from_file(name="parts", scenes=8, seed=0, engine="none", out_root=work)
    print(dataset.run(corpus))

    report = held_out_assets(corpus)
    print(format_report(report))

    # The warning this module exists to raise: a held-out set of zero is not evidence of anything,
    # and a report that looked healthy otherwise would hide exactly that.
    if report.total_unseen == 0:
        print("\nno unseen assets on this machine: fetch gso or ycb meshes before claiming a "
              "held-out number (python -m datagen.assets.fetch --list)")
