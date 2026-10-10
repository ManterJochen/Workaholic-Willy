"""Train the closed-set RT-DETR detector on a labelled dataset at the smoke tier: about a minute on a GPU, and proof
that the chain closes on your machine rather than a model to deploy.

Your own dataset is a COCO or YOLO folder, as CVAT, Label Studio or Roboflow export it: point `dataset=` at it, and
leave out `tier="smoke"` for the model you deploy. `epochs=`, `batch=` and `image_size=` set the run, `classes=` trains
some of the dataset's classes only, the objects of the others left as background, and `DetectorPlanOverrides` holds
every other setting. With no dataset at hand, a small set of coloured shapes stands in. It is written to a temporary
directory with everything else; your run names its `out_dir`, where the model and its report stay: `report.json`,
`curves.png`, drawn again after every epoch, and `report.html`, the whole run on one page.
"""

import shutil
import tempfile
from pathlib import Path

from willy import DetectorTraining

if shutil.which("nvidia-smi") is None:
    print("no NVIDIA GPU driver on this machine, and training the detector wants one.")
    raise SystemExit

with tempfile.TemporaryDirectory() as work:
    dataset = DetectorTraining.shapes_dataset(Path(work) / "shapes", images=48)
    run = DetectorTraining.from_dataset(dataset=dataset, tier="smoke", out_dir=Path(work) / "model", epochs=2,
                                        classes=["red_block", "green_disc"])
    print(run.describe())
    # What the dataset holds per class and split, and every image or box left out of it, before anything trains.
    print(run.probe())
    report = run.train()
    # The best epoch by validation mAP and where it went. A cell loads that folder through models.rtdetr.model_path,
    # with models.detector: "rtdetr"; the class names are the dataset's.
    print(report)
    written = run.write_report(report)
    print("report written to", written, "and", written.with_name("report.html"))
