"""Check a written dataset against itself: the gate that stops a labelling bug reaching a model.

Everything the check reads is already on disk, so it runs on a laptop with no GPU and no engine. It
is what a labelling bug looks like before it reaches a model: a mask that disagrees with its own
JSON, a box paired with the wrong object, a blank frame recorded as rendered.

`build.verify()` checks the dataset this build wrote. `verify_dataset(path)` is the same check
pointed at a directory, for one built months ago by a run nobody here started.
"""

import tempfile
from pathlib import Path

from willy import DatasetBuild, verify_dataset

with tempfile.TemporaryDirectory() as work:
    build = DatasetBuild.from_file(name="checked", scenes=8, seed=0, engine="none", out_root=work)
    build.render()

    report = build.verify()
    print(report)
    if not report.ok:
        raise SystemExit("a written dataset disagreed with itself; see the problems above")

    # The checks are ordered by how quietly they fail. The last one, "the colour image is an
    # image", is the one the other three cannot make: a scene with perfect labels and a fully black
    # picture passes every geometric check there is. It applies only where a picture was promised,
    # and `none` renders none, so nothing here is checked and the dataset's own provenance stamp is
    # what says so -- not the absence of the files, which is what a lost picture looks like too.
    print(f"{report.images_checked} colour image(s) checked, {report.images_blank} blank")

    # The same check, by path. This is the call for a dataset on a disk somewhere, and it is what
    # `python -m datagen verify --name <dataset>` runs.
    print(verify_dataset(Path(work) / "checked"))
