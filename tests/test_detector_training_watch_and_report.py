"""A detector training run a person can set, watch and read (the owner, testing it on his chess pieces, 2026-10-10).

- ``epochs=``, ``batch=``, ``image_size=`` and ``classes=`` are arguments of ``DetectorTraining.from_dataset``; every
  other setting, the learning rate among them, stays in ``DetectorPlanOverrides``, and one given both ways is refused;
- ``classes=`` keeps the classes it names, the objects of the others left in the images as background;
- the multi-scale sizes follow the image size;
- ``curves.png`` is drawn again after every epoch, and ``write_report`` writes ``report.html`` beside ``report.json``:
  the curves embedded, AP per class at the best epoch, every epoch as a table;
- an image larger than training needs is read shrunk, its boxes scaled with it, and the widest zoom-out canvas stays
  under PIL's decompression-bomb limit (a 12-megapixel photo laid on it whole reached 195 megapixels, and PIL refuses
  to crop that);
- loader processes start without running the training script again, so a script without a ``__main__`` guard trains
  once; without them, on a GPU, threads read, and the batches are exactly those of a single process;
- the base model loads without its load report or the ``num_labels`` warning, and every level is as it was after.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.models.detection.closed_set.training import loading, trainer
from src.models.detection.closed_set.training.api import DetectorTraining
from src.models.detection.closed_set.training.augment import build_transform
from src.models.detection.closed_set.training.datasets import LabelledBox, LabelledImage, load_dataset
from src.models.detection.closed_set.training.loading import (
    MAX_WORKING_SIDE,
    ThreadedBatches,
    WorkerBatches,
    main_hidden,
    read_image,
    working_side,
)
from src.models.detection.closed_set.training.plan import MULTISCALE_SIZES, DetectorPlan, DetectorPlanOverrides
from src.utility.progress import progress_bar
from tests._detector_training_fixtures import CLASSES, draw_image, tiny_factory, write_coco

TINY = DetectorPlan(image_size=64, epochs=3, batch=4, multiscale=False, patience=0, ema_warmup=10, seed=1)
_PNG = b"\x89PNG\r\n\x1a\n"
#: PIL's decompression-bomb limit as PIL ships it (``Image.MAX_IMAGE_PIXELS``, which a caller may change).
_BOMB_PIXELS = 89_478_485
_REPO = Path(__file__).resolve().parents[1]


class TheSettingsAPersonSetsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name) / "data"
        write_coco(self.root, 4)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_epochs_batch_and_image_size_are_arguments_and_outrank_the_tier(self) -> None:
        run = DetectorTraining.from_dataset(dataset=self.root, epochs=80, batch=4, image_size=800)
        self.assertEqual((80, 4, 800), (run.plan.epochs, run.plan.batch, run.plan.image_size))
        self.assertEqual(15, run.plan.patience)
        smoke = DetectorTraining.from_dataset(dataset=self.root, tier="smoke", epochs=5)
        self.assertEqual((5, 64), (smoke.plan.epochs, smoke.plan.max_train_images))

    def test_the_learning_rate_and_the_rest_still_come_through_overrides_beside_them(self) -> None:
        run = DetectorTraining.from_dataset(dataset=self.root, epochs=12,
                                            overrides=DetectorPlanOverrides(learning_rate=5e-5, patience=4))
        self.assertEqual((12, 5e-5, 4), (run.plan.epochs, run.plan.learning_rate, run.plan.patience))

    def test_a_setting_given_twice_is_refused_rather_than_one_of_them_winning(self) -> None:
        with self.assertRaisesRegex(ValueError, "epochs given twice"):
            DetectorTraining.from_dataset(dataset=self.root, epochs=12, overrides=DetectorPlanOverrides(epochs=30))
        with self.assertRaisesRegex(ValueError, "batch, image_size given twice"):
            DetectorTraining.from_dataset(dataset=self.root, batch=2, image_size=320,
                                          overrides=DetectorPlanOverrides(batch=4, image_size=640))

    def test_a_setting_no_run_can_take_is_refused_as_before(self) -> None:
        with self.assertRaisesRegex(ValueError, "image_size must be a multiple of 32"):
            DetectorTraining.from_dataset(dataset=self.root, image_size=650)

    def test_the_multiscale_sizes_follow_the_image_size(self) -> None:
        self.assertEqual(MULTISCALE_SIZES, DetectorPlan().multiscale_sizes())
        sizes = DetectorPlan(image_size=1024).multiscale_sizes()
        self.assertEqual((768, 1280), (min(sizes), max(sizes)))
        self.assertTrue(all(size % 32 == 0 for size in sizes))
        self.assertIn("multi-scale 768-1280 px",
                      DetectorTraining.from_dataset(dataset=self.root, image_size=1024).describe())


class ChosenClassesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmp.name) / "data"
        write_coco(cls.root, 20)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_only_the_classes_named_are_kept_and_the_others_stay_in_the_images_as_background(self) -> None:
        every = load_dataset(self.root)
        disc = load_dataset(self.root, classes=["disc"])
        self.assertEqual(("disc",), disc.classes)
        self.assertEqual([i.key for i in every.train], [i.key for i in disc.train])
        self.assertEqual([i.key for i in every.val], [i.key for i in disc.val])
        for whole, kept in zip((*every.train, *every.val), (*disc.train, *disc.val)):
            self.assertEqual([box.xyxy for box in whole.boxes if box.label == 1], [box.xyxy for box in kept.boxes])
            self.assertTrue(all(box.label == 0 for box in kept.boxes))
        self.assertTrue(any("not chosen (cube)" in line and "background" in line for line in disc.skipped))
        self.assertNotEqual(every.fingerprint, disc.fingerprint)

    def test_the_classes_keep_the_dataset_order_and_all_of_them_is_no_choice_at_all(self) -> None:
        both = load_dataset(self.root, classes=["disc", "cube"])
        self.assertEqual(CLASSES, both.classes)
        self.assertEqual(load_dataset(self.root).fingerprint, both.fingerprint)

    def test_a_name_the_dataset_does_not_have_is_refused_with_the_names_it_has(self) -> None:
        with self.assertRaisesRegex(ValueError, "no class 'sphere'.*'cube', 'disc'"):
            load_dataset(self.root, classes=["sphere"])
        with self.assertRaisesRegex(ValueError, "names no class"):
            load_dataset(self.root, classes=[])

    def test_one_name_alone_is_one_class_and_the_run_says_so_before_it_trains(self) -> None:
        run = DetectorTraining.from_dataset(dataset=self.root, classes="disc")
        self.assertEqual(("disc",), run.context.classes)
        self.assertEqual(("disc",), run.dataset().classes)
        self.assertEqual(("disc",), run.probe().classes)
        self.assertIn("1 chosen class(es): disc", run.describe())


class TheRunCanBeWatchedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmp.name)
        write_coco(cls.root / "data", 12)
        cls.run_ = DetectorTraining.from_dataset(
            dataset=cls.root / "data", tier="smoke", out_dir=cls.root / "model", device="cpu", image_size=64,
            batch=4, overrides=DetectorPlanOverrides(multiscale=True))
        cls.curves_after_each_epoch: list[bool] = []
        cls.run_.attach_progress_listener(
            lambda row: cls.curves_after_each_epoch.append((cls.root / "model" / "curves.png").is_file()))
        with mock.patch.object(trainer, "pretrained_factory", tiny_factory):
            cls.report = cls.run_.train()
        cls.json_path = cls.run_.write_report(cls.report)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_curves_png_is_there_after_every_epoch(self) -> None:
        self.assertEqual([True] * len(self.report.epochs), self.curves_after_each_epoch)
        self.assertEqual(_PNG, (self.root / "model" / "curves.png").read_bytes()[:8])

    def test_report_html_beside_report_json_carries_the_curves_the_classes_and_every_epoch(self) -> None:
        self.assertEqual("completed", json.loads(self.json_path.read_text(encoding="utf-8"))["outcome"])
        page = (self.root / "model" / "report.html").read_text(encoding="utf-8")
        self.assertIn("data:image/png;base64,", page)
        for name in CLASSES:
            self.assertIn(name, page)
        epochs = page[page.index("<h2>Every epoch</h2>"):page.index("<h2>Dataset</h2>")]
        self.assertEqual(len(self.report.epochs), epochs.count("<tr><td>") + epochs.count('<tr class="best"><td>'))
        self.assertEqual(1, epochs.count('<tr class="best">'))

    def test_the_printed_report_names_its_files_and_the_ap_of_every_class(self) -> None:
        text = self.report.render()
        text.encode("ascii")
        self.assertIn("curves", text)
        self.assertIn("report.html", text)
        self.assertTrue(self.report.validated)
        self.assertIn("AP per class at the best epoch", text)

    def test_every_epoch_row_is_typed_with_its_learning_rate_and_its_ap_per_class(self) -> None:
        for row in self.report.epochs:
            self.assertGreater(row.lr, 0.0)
            self.assertEqual(set(CLASSES), set(row.per_class_ap))
        self.assertEqual(1, sum(1 for row in self.report.epochs if row.epoch == self.report.best_epoch))


class AnImageLargerThanTrainingNeedsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        _, self.processor = tiny_factory(TINY, dict(enumerate(CLASSES)))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _photo(self, name: str, size: tuple[int, int]) -> LabelledImage:
        width, height = size
        box = (width * 0.25, height * 0.30, width * 0.60, height * 0.70)
        path = self.root / name
        draw_image(path, [(0, box)], size=size, seed=3)
        return LabelledImage(path=path, width=width, height=height, boxes=(LabelledBox(0, box),), key=name)

    def test_a_12_megapixel_jpeg_is_read_at_the_working_side_and_its_boxes_scale_with_it(self) -> None:
        entry = self._photo("photo.jpg", (4032, 3024))
        image, (file_width, file_height) = read_image(entry.path, max_side=working_side(640))
        self.assertEqual((4032, 3024), (file_width, file_height))
        self.assertEqual(working_side(640), max(image.size))
        shrunk = trainer.DetectionSamples([entry], self.processor, augment=False, max_side=working_side(640))[0]
        whole = trainer.DetectionSamples([entry], self.processor, augment=False, max_side=10_000)[0]
        np.testing.assert_allclose(shrunk["labels"]["boxes"].numpy(), whole["labels"]["boxes"].numpy(), atol=2e-3)

    def test_the_widest_zoom_out_of_a_12_megapixel_photo_is_cropped_without_a_decompression_bomb(self) -> None:
        from PIL import Image
        from torchvision.transforms import v2 as T

        entry = self._photo("photo.png", (4032, 3024))
        image, _ = trainer.DetectionSamples([entry], self.processor, augment=True).decode(0)
        zoom = next(op for op in build_transform(strong=True, crop=True).transforms if isinstance(op, T.RandomZoomOut))
        self.assertEqual(float(loading._WIDEST_ZOOM_OUT), zoom.side_range[1])
        widest = T.RandomZoomOut(fill=0, side_range=(zoom.side_range[1], zoom.side_range[1]), p=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            canvas = widest(image)
            # RandomIoUCrop's largest crop; PIL checks the size of every crop. The photo whole gives 16128 x 12096.
            canvas.crop((0, 0, canvas.width, canvas.height))
        self.assertEqual(4 * working_side(640), max(canvas.size))

    def test_no_training_size_reads_an_image_whose_widest_canvas_passes_the_limit(self) -> None:
        for size in range(64, 4097, 32):
            side = working_side(size)
            self.assertLessEqual((loading._WIDEST_ZOOM_OUT * side) ** 2, _BOMB_PIXELS)
            self.assertEqual(min(MAX_WORKING_SIDE, max(1333, 2 * size)), side)

    def test_a_small_image_is_read_as_it_is(self) -> None:
        entry = self._photo("small.png", (96, 64))
        image, size = read_image(entry.path, max_side=working_side(640))
        self.assertEqual((96, 64), image.size)
        self.assertEqual((96, 64), size)


_UNGUARDED = """import multiprocessing
import sys

sys.path.insert(0, {repo!r})
print("the script runs", flush=True)
multiprocessing.set_start_method("spawn", force=True)

from transformers import RTDetrImageProcessor

from src.models.detection.closed_set.training import trainer
from src.models.detection.closed_set.training.datasets import load_dataset
from src.models.detection.closed_set.training.plan import DetectorPlan

plan = DetectorPlan(image_size=64, batch=2)
data = load_dataset({data!r})
samples = trainer.DetectionSamples(data.train, RTDetrImageProcessor(size={{"height": 64, "width": 64}}), augment=True)
loader = trainer._loader(samples, plan, shuffle=True, workers=2, seed=1)
print("batches", type(loader).__name__, sum(1 for _ in loader), len(loader), flush=True)
"""


class LoaderProcessesTests(unittest.TestCase):
    def test_a_script_without_a_main_guard_trains_once_while_spawned_processes_read_for_it(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            write_coco(Path(tmp) / "data", 8)
            script = Path(tmp) / "train_unguarded.py"
            script.write_text(_UNGUARDED.format(repo=str(_REPO), data=str(Path(tmp) / "data")), encoding="utf-8")
            done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=600,
                                  cwd=tmp)
        self.assertEqual(0, done.returncode, done.stderr[-3000:])
        self.assertEqual(1, done.stdout.count("the script runs"), done.stdout)
        kind, counted, length = done.stdout.split("batches ", 1)[1].split()[:3]
        self.assertEqual("WorkerBatches", kind)
        self.assertEqual(counted, length)
        self.assertGreater(int(counted), 0)

    def test_main_loses_its_file_and_its_module_name_only_while_the_processes_start(self) -> None:
        main = sys.modules["__main__"]
        before = (getattr(main, "__file__", None), main.__spec__)
        with main_hidden():
            self.assertFalse(hasattr(main, "__file__"))
            self.assertIsNone(main.__spec__)
        self.assertEqual(before, (getattr(main, "__file__", None), main.__spec__))


class ThreadedBatchesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(cls._tmp.name)
        write_coco(root / "data", 14)
        cls.dataset = load_dataset(root / "data")
        _, cls.processor = tiny_factory(TINY, cls.dataset.id2label)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _epochs(self, loader: object, epochs: int = 2) -> list[list[tuple[list[int], np.ndarray]]]:
        out = []
        for _ in range(epochs):
            out.append([(batch["index"], batch["pixel_values"].numpy().copy()) for batch in loader])  # type: ignore
        return out

    def test_the_threads_give_exactly_the_batches_of_a_single_process_augmented_and_shuffled(self) -> None:
        samples = trainer.DetectionSamples(self.dataset.train, self.processor, augment=True)
        torch.manual_seed(5)
        single = self._epochs(trainer._loader(samples, TINY, shuffle=True, workers=0, seed=7))
        torch.manual_seed(5)
        threaded = self._epochs(ThreadedBatches(samples, batch=TINY.batch, shuffle=True, seed=7, threads=3,
                                                collate=trainer.collate))
        self.assertEqual([[i for i, _ in epoch] for epoch in single], [[i for i, _ in epoch] for epoch in threaded])
        for left, right in zip(single, threaded):
            for (_, a), (_, b) in zip(left, right):
                np.testing.assert_array_equal(a, b)

    def test_a_loop_that_stops_early_lets_the_threads_go(self) -> None:
        samples = trainer.DetectionSamples(self.dataset.train, self.processor, augment=False)
        for _batch in ThreadedBatches(samples, batch=2, shuffle=False, seed=1, threads=2, collate=trainer.collate):
            break
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and any(t.name == "detector-batches" for t in threading.enumerate()):
            time.sleep(0.05)
        self.assertFalse(any(t.name == "detector-batches" for t in threading.enumerate()))

    def test_a_read_that_fails_is_raised_in_the_training_loop(self) -> None:
        samples = trainer.DetectionSamples(self.dataset.train, self.processor, augment=False)
        with mock.patch.object(samples, "decode", side_effect=OSError("the file is gone")):
            with self.assertRaisesRegex(OSError, "the file is gone"):
                for _batch in ThreadedBatches(samples, batch=2, shuffle=False, seed=1, threads=2,
                                              collate=trainer.collate):
                    pass

    def test_processes_read_where_there_are_some_threads_on_a_gpu_without_and_a_cpu_reads_itself(self) -> None:
        samples = trainer.DetectionSamples(self.dataset.train, self.processor, augment=False)
        cuda, cpu = torch.device("cuda"), torch.device("cpu")
        self.assertIsInstance(trainer._loader(samples, TINY, shuffle=False, workers=2, seed=1, device=cuda),
                              WorkerBatches)
        self.assertIsInstance(trainer._loader(samples, TINY, shuffle=False, workers=0, seed=1, device=cuda),
                              ThreadedBatches)
        plain = trainer._loader(samples, TINY, shuffle=False, workers=0, seed=1, device=cpu)
        self.assertNotIsInstance(plain, (ThreadedBatches, WorkerBatches))


class TheBaseModelLoadsQuietlyTests(unittest.TestCase):
    def test_the_classes_come_as_id2label_alone_and_every_level_is_restored(self) -> None:
        from huggingface_hub.utils import logging as hub_logging
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_warning()
        hub_logging.set_verbosity_warning()
        seen: dict[str, object] = {}

        def model_from(source: str, **kwargs: object) -> object:
            seen.update(kwargs)
            seen["levels"] = (hf_logging.get_verbosity(), hub_logging.get_verbosity())
            return mock.MagicMock()

        with mock.patch("transformers.AutoModelForObjectDetection.from_pretrained", side_effect=model_from), \
                mock.patch("transformers.AutoImageProcessor.from_pretrained", return_value=mock.MagicMock()), \
                mock.patch.object(trainer, "resolve_base_model", return_value=("some/base", False)):
            trainer.pretrained_factory(DetectorPlan(), {0: "white pawn", 1: "black king"})
        self.assertNotIn("num_labels", seen)
        self.assertEqual({0: "white pawn", 1: "black king"}, seen["id2label"])
        self.assertTrue(seen["ignore_mismatched_sizes"])
        self.assertEqual((hf_logging.ERROR, hub_logging.ERROR), seen["levels"])
        self.assertEqual((hf_logging.WARNING, hub_logging.WARNING),
                         (hf_logging.get_verbosity(), hub_logging.get_verbosity()))


class TheBarTests(unittest.TestCase):
    def test_a_bar_nobody_watches_writes_nothing(self) -> None:
        import io

        stream = io.StringIO()
        with mock.patch("sys.stderr", stream), progress_bar(10, desc="epoch 1/2", enabled=False) as bar:
            bar.update(1)
            bar.set_postfix_str("loss 1.0")
        self.assertEqual("", stream.getvalue())

    def test_a_bar_somebody_watches_is_ascii(self) -> None:
        import io

        stream = io.StringIO()
        with mock.patch("sys.stderr", stream), progress_bar(4, desc="epoch 1/2", enabled=True) as bar:
            for _ in range(4):
                bar.update(1)
            bar.set_postfix_str("loss 0.500  12.0 img/s")
        self.assertIn("epoch 1/2", stream.getvalue())
        stream.getvalue().encode("ascii")


if __name__ == "__main__":
    unittest.main()
