"""One SAM2 pass over every box of an image cuts what one pass per box cuts, through the real transformers SAM2.

``ObjectDetector`` hands SAM2's processor and model every box of the image at once, where ``Sam2Segmenter.segment``,
the cell's path, hands them one. The stand-ins of ``tests/test_an_object_detector_cuts_every_box_in_one_sam2_pass.py``
answer as this file's author expects SAM2 to answer; this file asks SAM2 itself. Its model is the real ``Sam2Model`` at
toy size (about 90k parameters, random weights, a 128 px input), saved and loaded through ``save_pretrained`` and
``from_pretrained`` as a fetched checkpoint is, so the processor's box format, the shapes of the masks and of the
predicted IoUs, and the scaling of a slice of the masks are transformers' own. The two paths have to agree box for box:
the same mask, and the same predicted IoU for it. With random weights a mask means nothing, so the cleaning both paths
share is switched off, which leaves the pass itself to compare. RT-DETR is the stand-in, since only SAM2 is asked here.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.models.detection.object_detector import ObjectDetector
from tests._object_detector_fakes import CHESS, detector_folder, pin_cpu, rtdetr

#: What undoes the CPU pin of this module, once ``setUpModule`` has set it.
_UNPIN: list = []


def setUpModule() -> None:  # noqa: N802 (unittest's name)
    _UNPIN.append(pin_cpu())


def tearDownModule() -> None:  # noqa: N802 (unittest's name)
    while _UNPIN:
        _UNPIN.pop()()


HEIGHT, WIDTH = 60, 90
#: Three boxes away from the image's edges, where the two paths read a box alike: the cell's truncates a box to whole
#: pixels and clips it to the last pixel, this one clips it to the image's edge.
SCENE = [
    ((10.0, 10.0, 30.0, 40.0), CHESS.index("white_pawn"), 0.95),
    ((40.0, 5.0, 70.0, 45.0), CHESS.index("black_king"), 0.85),
    ((60.0, 20.0, 85.0, 55.0), CHESS.index("white_queen"), 0.75),
]


def _toy_sam2(folder: Path) -> Path:
    """The real SAM2 architecture at toy size, and its processor at a 128 px input, written as a checkpoint is."""
    from transformers import (
        Sam2Config,
        Sam2HieraDetConfig,
        Sam2ImageProcessor,
        Sam2MaskDecoderConfig,
        Sam2Model,
        Sam2Processor,
        Sam2PromptEncoderConfig,
        Sam2VisionConfig,
    )

    torch.manual_seed(0)
    # Four Hiera stages of 32, 16, 8 and 4 tokens a side; the global block is no stage's first, as in SAM2's own.
    backbone = Sam2HieraDetConfig(hidden_size=8, num_attention_heads=1, image_size=[128, 128],
                                  blocks_per_stage=[1, 1, 2, 1], embed_dim_per_stage=[8, 16, 32, 64],
                                  num_attention_heads_per_stage=[1, 1, 1, 1], window_size_per_stage=[4, 4, 4, 2],
                                  global_attention_blocks=[3], window_positional_embedding_background_size=[2, 2])
    vision = Sam2VisionConfig(backbone_config=backbone, backbone_channel_list=[64, 32, 16, 8],
                              backbone_feature_sizes=[[32, 32], [16, 16], [8, 8]], fpn_hidden_size=16)
    config = Sam2Config(
        vision_config=vision,
        prompt_encoder_config=Sam2PromptEncoderConfig(hidden_size=16, image_size=128, patch_size=16,
                                                      mask_input_channels=4),
        mask_decoder_config=Sam2MaskDecoderConfig(hidden_size=16, mlp_dim=32, num_attention_heads=2,
                                                  iou_head_hidden_dim=16))
    Sam2Model(config).eval().save_pretrained(str(folder))
    processor = Sam2Processor(Sam2ImageProcessor(size={"height": 128, "width": 128},
                                                 mask_size={"height": 32, "width": 32}))
    processor.save_pretrained(str(folder))
    return folder


class OnePassAgainstOnePassPerBoxTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(cls._tmp.name)
        sam2 = _toy_sam2(root / "sam2")
        with rtdetr(SCENE):
            cls.detector = ObjectDetector.from_weights(detector_folder(root / "chess"), segmenter=sam2, local=True,
                                                       device="cpu")
        rng = np.random.default_rng(1)
        cls.image = rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_every_box_gets_the_mask_and_the_score_its_own_pass_gives_it(self) -> None:
        segmenter = self.detector._segmenter_built()  # noqa: SLF001 (the SAM2 the batch is cut with)
        # Both paths clean a mask with these two settings; switched off, what is left to compare is the pass itself.
        segmenter.keep_largest_component = False
        segmenter.morph_kernel_size = 1
        found = self.detector.detect(self.image, segment=True)
        self.assertEqual(3, len(found))
        for obj in found:
            with self.subTest(obj.label):
                x0, y0, x1, y1 = (int(value) for value in obj.box)
                alone = segmenter.segment(image_rgb=self.image[:, :, ::-1].copy(), bbox_xyxy=(x0, y0, x1, y1),
                                          label=obj.label)
                differing = int(np.count_nonzero(alone.mask.astype(bool) != obj.mask))
                self.assertLessEqual(differing, 3, f"{differing} of {HEIGHT * WIDTH} pixels differ")
                self.assertAlmostEqual(alone.metadata["predicted_iou"][0], obj.mask_score, places=4)

    def test_the_batch_is_the_image_size_and_carries_a_score_per_box(self) -> None:
        found = self.detector.detect(self.image, segment=True)
        for obj in found:
            self.assertEqual((HEIGHT, WIDTH), obj.mask.shape)
            self.assertIsInstance(obj.mask_score, float)
        self.assertEqual(3, len({obj.mask_score for obj in found}), "each box carries its own score")


if __name__ == "__main__":
    unittest.main()
