"""Reading the training images fast: shrunk where they are larger than training needs, and read beside the
training step rather than between two of its steps.

The network sees 480 to 800 px at the default size of 640. A phone photo of 3024 x 4032 px decoded at full size costs
most of a training step, and ``RandomZoomOut`` lays it on a canvas up to four times as wide and as high:
``RandomIoUCrop`` cutting that canvas reached 111 megapixels on the owner's chess images (2026-10-10), past PIL's
decompression-bomb limit, hundreds of megabytes a sample for a picture shrunk to 640 px right after. :func:`read_image`
reads every image at most :func:`working_side` px on its long side, a JPEG decoded straight at a reduced scale
(``Image.draft``), and that side keeps the largest canvas under the limit; the trainer scales the boxes with it. The
files on disk are never touched.

Loader processes read best: preparing a sample (the augmentation and the image processor, 42 ms on the chess images)
holds the GIL, so threads share one core for it. Measured on the owner's RTX 5080, four processes fed 28.9 images/s,
as fast as the GPU trains, and threads 20.7. A process started by *spawn* (Windows, macOS) runs the script that started
the training once more, and a script without an ``if __name__ == "__main__":`` guard would start the training again in
every process; :class:`WorkerBatches` starts them with ``__main__`` hidden (:func:`main_hidden`), so it does not.

Where no process reads (``workers=0``, or a dataset too small to pay for their start), :class:`ThreadedBatches` reads
in threads, ahead of the step that needs them: the decoding in a pool, and everything that draws a random number, the
augmentation, in one thread in the order of the samples. The order of the samples is drawn by torch's own
``DataLoader`` from the same generator, so a run draws exactly the numbers a single-process loader draws, in the same
order, and trains on the same batches.
"""

from __future__ import annotations

import math
import os
import queue
import sys
import threading
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = ["MAX_WORKING_SIDE", "MIN_WORKING_SIDE", "ThreadedBatches", "WorkerBatches", "decode_threads",
           "main_hidden", "read_image", "working_side"]

#: The long side, in pixels, below which an image is never shrunk: the size detectors keep COCO's images at.
MIN_WORKING_SIDE = 1333
#: PIL's decompression-bomb limit, ``Image.MAX_IMAGE_PIXELS`` as PIL ships it: ``Image.crop`` warns past it, and
#: refuses past twice it.
_BOMB_PIXELS = 89_478_485
#: How much wider and higher than the image ``RandomZoomOut`` lays its canvas at most (torchvision's ``side_range``).
_WIDEST_ZOOM_OUT = 4
#: The long side whose widest zoom-out canvas, and so any crop of it, stays under that limit: 2364 px.
MAX_WORKING_SIDE = math.isqrt(_BOMB_PIXELS) // _WIDEST_ZOOM_OUT
#: How many batches the threads prepare ahead of the step that needs them.
_AHEAD = 3


def working_side(image_size: int) -> int:
    """The long side, in pixels, an image is read at for training at ``image_size``: twice that, never under
    :data:`MIN_WORKING_SIDE` and never over :data:`MAX_WORKING_SIDE`. A crop of a tenth of the image still has a
    training size's pixels to give, and no zoom-out canvas passes PIL's decompression-bomb limit.

    Args:
        image_size (int): The plan's training image size in pixels.

    Returns:
        int: The long side in pixels.
    """
    return min(MAX_WORKING_SIDE, max(MIN_WORKING_SIDE, 2 * int(image_size)))


def read_image(path: str | Path, *, max_side: int) -> tuple[Any, tuple[int, int]]:
    """The image at ``path`` in RGB, at most ``max_side`` px on its long side, and the size of the file itself.

    A JPEG is decoded at the largest of JPEG's own reduced scales that still has ``max_side`` (``Image.draft``), so a
    12-megapixel photo costs a quarter of its decode; anything still larger is shrunk with a bilinear filter.

    Args:
        path (str | Path): The image file.
        max_side (int): The longest side to keep, in pixels.

    Returns:
        tuple[Any, tuple[int, int]]: The PIL image, and the file's own ``(width, height)``; the image's size over that
            is the factor the boxes scale by.
    """
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as raw:
        width, height = raw.size
        scale = max_side / float(max(width, height))
        if scale < 1.0 and raw.format == "JPEG":
            raw.draft("RGB", (max(1, int(width * scale)), max(1, int(height * scale))))
        image = raw.convert("RGB")
    if max(image.size) > max_side:
        factor = max_side / float(max(image.size))
        image = image.resize((max(1, round(image.width * factor)), max(1, round(image.height * factor))),
                             Image.Resampling.BILINEAR, reducing_gap=2.0)
    return image, (int(width), int(height))


def decode_threads() -> int:
    """How many threads decode the images: the machine's cores less two (the training step and the augmentation),
    between 1 and 8.

    Returns:
        int: The thread count.
    """
    return max(1, min(8, (os.cpu_count() or 2) - 2))


class ThreadedBatches:
    """The batches of ``samples``, read in threads while the GPU trains on the batch before.

    Iterates like a ``DataLoader`` (``len()`` is its batch count, and each epoch is one ``for`` loop over it). The
    order of the samples comes from a ``DataLoader`` over their indices with the same batch size, shuffle and generator,
    kept across epochs as a loader is; ``samples.decode(index)`` runs in ``threads`` threads, and
    ``samples.prepare(index, decoded)`` and ``collate`` run in one thread in that order.

    Args:
        samples (Any): Has ``decode(index)``, ``prepare(index, decoded)`` and ``len()``.
        batch (int): Samples per batch.
        shuffle (bool): Draw a new order every epoch.
        seed (int): The generator the order is drawn from.
        threads (int): Decoding threads.
        collate (Callable[[Sequence[Any]], Any]): Makes a batch of prepared samples.
        pin (bool): Pin each batch's ``pixel_values`` for a faster copy to the GPU (default: False).
    """

    def __init__(self, samples: Any, *, batch: int, shuffle: bool, seed: int, threads: int,
                 collate: Callable[[Sequence[Any]], Any], pin: bool = False) -> None:
        import torch  # noqa: PLC0415
        from torch.utils.data import DataLoader  # noqa: PLC0415

        generator = torch.Generator()
        generator.manual_seed(seed)
        self.samples = samples
        self.threads = max(1, int(threads))
        self._collate = collate
        self._pin = pin
        # A range is all a DataLoader needs of a dataset (len and indexing); its stubs ask for a Dataset.
        indices: Any = range(len(samples))
        self._indices: Any = DataLoader(indices, batch_size=batch, shuffle=shuffle, generator=generator, num_workers=0,
                                        collate_fn=list)

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[Any]:
        stop = threading.Event()
        ready: queue.Queue = queue.Queue(maxsize=_AHEAD)
        worker = threading.Thread(target=self._produce, args=(iter(self._indices), ready, stop), daemon=True,
                                  name="detector-batches")
        worker.start()
        try:
            while True:
                kind, value = ready.get()
                if kind == "batch":
                    yield value
                elif kind == "end":
                    return
                else:
                    raise value
        finally:
            # A loop that stops early (patience, divergence) lets the producer go: it sees the flag, and a put it is
            # blocked in returns once the queue is drained.
            stop.set()
            while worker.is_alive():
                try:
                    ready.get_nowait()
                except queue.Empty:
                    pass
                worker.join(timeout=0.05)

    def _produce(self, index_batches: Iterator[list[int]], ready: queue.Queue, stop: threading.Event) -> None:
        pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="detector-decode")
        try:
            pending: deque[tuple[list[int], list[Future]]] = deque()

            def submit_next() -> None:
                indices = next(index_batches, None)
                if indices is not None:
                    pending.append((indices, [pool.submit(self.samples.decode, i) for i in indices]))

            for _ in range(_AHEAD + 1):
                submit_next()
            while pending and not stop.is_set():
                indices, decoding = pending.popleft()
                submit_next()
                batch = self._collate([self.samples.prepare(i, job.result()) for i, job in zip(indices, decoding)])
                if self._pin:
                    batch["pixel_values"] = batch["pixel_values"].pin_memory()
                self._put(ready, stop, ("batch", batch))
            self._put(ready, stop, ("end", None))
        except BaseException as exc:  # noqa: BLE001 (handed to the training loop, which raises it)
            self._put(ready, stop, ("error", exc))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _put(ready: queue.Queue, stop: threading.Event, item: tuple[str, Any]) -> None:
        while not stop.is_set():
            try:
                ready.put(item, timeout=0.1)
                return
            except queue.Full:
                continue


@contextmanager
def main_hidden() -> Iterator[None]:
    """``__main__`` without its file and its module name, for the moment loader processes start.

    *spawn* hands a new process the script that started the training, by path or by module name, and the process runs
    it once more (as ``__mp_main__``) to find what was pickled from it. Nothing a loader process needs comes from that
    script: the samples, the collate function and the seeding are this package's. With neither name to go by, spawn
    leaves the script alone, and a script without a ``__main__`` guard trains once, not once more in every process.
    Both names are back as soon as the processes have started.
    """
    main = sys.modules.get("__main__")
    if main is None:
        yield
        return
    names = vars(main)
    saved = {key: names[key] for key in ("__file__", "__spec__") if key in names}
    names.pop("__file__", None)
    names["__spec__"] = None
    try:
        yield
    finally:
        for key in ("__file__", "__spec__"):
            if key in saved:
                names[key] = saved[key]
            else:
                names.pop(key, None)


class WorkerBatches:
    """A ``DataLoader`` whose worker processes start with ``__main__`` hidden (:func:`main_hidden`).

    Iterates as the loader does, ``len()`` included; persistent workers start on the first ``for`` loop over it and
    stay for the next.

    Args:
        loader (Any): The ``DataLoader``.
    """

    def __init__(self, loader: Any) -> None:
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Any]:
        with main_hidden():
            return iter(self.loader)
