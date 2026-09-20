"""Check a written dataset against itself: the gate that stops a labelling bug reaching a model.

Everything verify_dataset reads is already on disk, so it runs on a laptop with no GPU and no
engine, and can be pointed at a dataset built months ago. It is what a labelling bug looks like
before it reaches a model: a mask that disagrees with its own JSON, a box paired with the wrong
object, a blank frame recorded as rendered.
"""

import tempfile
from pathlib import Path

from willy import DatasetBuild
from datagen.verify import verify_dataset

with tempfile.TemporaryDirectory() as work:
    build = DatasetBuild.from_file(name="checked", scenes=8, seed=0, engine="none", out_root=work)
    build.render()

    report = verify_dataset(Path(work) / "checked")
    print(report.summary())
    if not report.ok:
        raise SystemExit("a written dataset disagreed with itself; see the problems above")

    # The checks are ordered by how quietly they fail. The last one, "the colour image is an
    # image", is the one the other three cannot make: a scene with perfect labels and a fully black
    # picture passes every geometric check there is.
    print(f"{report.images_checked} colour image(s) checked, {report.images_blank} blank")
