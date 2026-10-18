"""One call answers with every class a closed-set detector knows, and a misspelt class is refused, never answered empty.

The owner, 2026-10-09: the chess program takes the closed-set detector and always wants every class at once, and that
was hard, since ``RtDetrObjectDetector`` kept its classes to itself and read a prompt as a substring of a class name.
``ObjectDetector`` reads them out (``classes``, in id order), answers with all of them unless ``classes=`` narrows the
call, matches a name exactly but for case and blanks, and refuses a name the model does not know, listing the ones it
does. RT-DETR is a stand-in here (``tests/_object_detector_fakes.py``): the wrapper runs, and only ``from_pretrained``
answers, with a model that sees the scene this file chose.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np

import willy
from src.models.detection.closed_set.detector import RtDetrObjectDetector
from src.models.detection.object_detector import (
    DetectedObject,
    Detections,
    ObjectDetector,
    decode_mask,
    encode_mask,
)
from tests._object_detector_fakes import CHESS, detector_folder, pin_cpu, rtdetr, sam2, sam2_folder

#: What undoes the CPU pin of this module, once ``setUpModule`` has set it.
_UNPIN: list = []


def setUpModule() -> None:  # noqa: N802 (unittest's name)
    _UNPIN.append(pin_cpu())


def tearDownModule() -> None:  # noqa: N802 (unittest's name)
    while _UNPIN:
        _UNPIN.pop()()


#: The image the scene is in: 160 px wide, 120 px high.
HEIGHT, WIDTH = 120, 160
#: Five objects above 0.5 and one below it; the black pawn reaches past the image and the white rook lies outside it.
SCENE = [
    ((10.0, 10.0, 30.0, 40.0), CHESS.index("white_pawn"), 0.95),
    ((50.0, 20.0, 80.0, 60.0), CHESS.index("black_king"), 0.8),
    ((100.0, 50.0, 140.0, 100.0), CHESS.index("white_queen"), 0.6),
    ((150.0, 90.0, 175.0, 130.0), CHESS.index("black_pawn"), 0.7),
    ((170.0, 10.0, 190.0, 30.0), CHESS.index("white_rook"), 0.9),
    ((60.0, 70.0, 90.0, 100.0), CHESS.index("white_knight"), 0.3),
]


def _image() -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    image[10:40, 10:30] = (240, 240, 240)
    return image


class _Built(unittest.TestCase):
    """A detector built from a folder shaped as a DetectorTraining out_dir, over the stand-in RT-DETR."""

    scene = SCENE

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        with rtdetr(self.scene) as fake:
            self.detector = ObjectDetector.from_weights(detector_folder(self.root / "chess"),
                                                        segmenter=sam2_folder(self.root / "sam2"))
        self.fake = fake


class TheClassesTests(_Built):

    def test_the_classes_are_the_id2label_names_in_id_order(self) -> None:
        self.assertEqual(CHESS, self.detector.classes)
        with rtdetr([], classes={2: "rook", 0: "pawn", 1: "knight"}):
            built = ObjectDetector.from_weights(detector_folder(self.root / "other"))
        self.assertEqual(("pawn", "knight", "rook"), built.classes)
        self.assertEqual(("pawn", "knight", "rook"), built._detector.classes)  # noqa: SLF001 (the wrapper's own)
        self.assertIsInstance(built._detector, RtDetrObjectDetector)  # noqa: SLF001

    def test_every_class_above_the_threshold_comes_back_by_default(self) -> None:
        found = self.detector.detect(_image())
        self.assertEqual(CHESS, found.classes)
        self.assertEqual({"white_pawn", "black_king", "white_queen", "black_pawn"}, set(found.labels))
        self.assertEqual([0.5], self.fake.processor.thresholds)

    def test_classes_narrows_the_answer_to_exact_names_case_and_blanks_aside(self) -> None:
        found = self.detector.detect(_image(), classes=["  Black_King ", "WHITE_PAWN"])
        self.assertEqual(("white_pawn", "black_king"), found.classes)
        self.assertEqual(("white_pawn", "black_king"), found.labels)
        self.assertEqual(("white_queen",), self.detector.detect(_image(), classes="white_queen").labels)

    def test_a_part_of_a_class_name_is_no_class_name(self) -> None:
        # The wrapper's prompt reads "pawn" as a substring of both pawns; a class list must name a class.
        with self.assertRaisesRegex(ValueError, "knows no class 'pawn'"):
            self.detector.detect(_image(), classes=["pawn"])

    def test_a_class_the_model_does_not_know_is_refused_naming_every_class_it_knows(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.detector.detect(_image(), classes=["white_pawn", "white_pawm"])
        message = str(caught.exception)
        self.assertIn("'white_pawm'", message)
        self.assertNotIn("'white_pawn'", message)
        for name in CHESS:
            self.assertIn(name, message)

    def test_an_empty_list_of_classes_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "names no class"):
            self.detector.detect(_image(), classes=[])


class TheBoxesTests(_Built):

    def test_a_box_past_the_edge_is_clipped_and_one_outside_the_image_is_dropped(self) -> None:
        found = self.detector.detect(_image())
        (pawn,) = found.of("black_pawn")
        self.assertEqual((150.0, 90.0, 160.0, 120.0), pawn.box)
        self.assertEqual((155.0, 105.0), pawn.centre)
        self.assertNotIn("white_rook", found.labels)
        for obj in found:
            x0, y0, x1, y1 = obj.box
            self.assertTrue(0.0 <= x0 < x1 <= WIDTH and 0.0 <= y0 < y1 <= HEIGHT, obj)

    def test_the_highest_score_comes_first(self) -> None:
        scores = [obj.score for obj in self.detector.detect(_image())]
        self.assertEqual(sorted(scores, reverse=True), scores)
        self.assertAlmostEqual(0.95, scores[0], places=6)

    def test_a_threshold_for_one_call_stands_in_for_the_detectors_own(self) -> None:
        low = self.detector.detect(_image(), threshold=0.2)
        self.assertIn("white_knight", low.labels)
        self.assertEqual(0.2, low.threshold)
        high = self.detector.detect(_image(), threshold=0.85)
        self.assertEqual(("white_pawn",), high.labels)
        self.assertEqual(0.5, self.detector.detect(_image()).threshold)
        self.assertEqual([0.2, 0.85, 0.5], self.fake.processor.thresholds)

    def test_a_threshold_outside_0_to_1_is_refused(self) -> None:
        for bad in (-0.1, 1.5, float("nan"), True):
            with self.subTest(threshold=bad):
                with self.assertRaisesRegex(ValueError, "score from 0 to 1"):
                    self.detector.detect(_image(), threshold=bad)
                with self.assertRaisesRegex(ValueError, "score from 0 to 1"):
                    ObjectDetector.from_weights(self.root / "chess", threshold=bad)

    def test_segment_false_asks_sam2_nothing_and_gives_no_mask(self) -> None:
        with sam2() as fake:
            found = self.detector.detect(_image())
        self.assertEqual(0, fake.loads.call_count)
        self.assertEqual([], fake.processor.calls)
        self.assertFalse(found.segmented)
        self.assertTrue(all(obj.mask is None and obj.mask_score is None for obj in found))

    def test_an_image_can_be_named_by_its_path_and_anything_else_is_refused(self) -> None:
        path = self.root / "board.png"
        cv2.imwrite(str(path), _image())
        self.assertEqual(self.detector.detect(_image()).labels, self.detector.detect(path).labels)
        with self.assertRaisesRegex(FileNotFoundError, "no image at"):
            self.detector.detect(self.root / "missing.png")
        for bad in (np.zeros((HEIGHT, WIDTH), np.uint8), np.zeros((HEIGHT, WIDTH, 4), np.uint8),
                    np.zeros((HEIGHT, WIDTH, 3), np.float32)):
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                with self.assertRaisesRegex(ValueError, "height x width x 3 uint8"):
                    self.detector.detect(bad)


class AnEmptySceneTests(_Built):
    scene: list = []

    def test_an_empty_scene_is_an_empty_detections_that_says_what_it_looked_for(self) -> None:
        found = self.detector.detect(_image())
        self.assertEqual(0, len(found))
        self.assertFalse(found)
        self.assertEqual(CHESS, found.classes)
        self.assertEqual(dict.fromkeys(CHESS, 0), found.counts)
        self.assertEqual((), found.of("white_king"))
        text = found.render()
        self.assertIn("0 object(s) above 0.50 in a 160 x 120 px image, of 12 class(es)", text)
        self.assertIn("none of", text)


class OneObjectUnderTwoClassesTests(_Built):
    """RT-DETR takes the best query-class pairs, so one query can clear the threshold as a knight and as a bishop, its
    box gathered from the same row: on a board that read two pieces on one square."""

    scene = [
        ((20.0, 20.0, 50.0, 60.0), CHESS.index("white_knight"), 0.8),
        ((20.0, 20.0, 50.0, 60.0), CHESS.index("white_bishop"), 0.55),
        ((90.0, 30.0, 120.0, 70.0), CHESS.index("black_pawn"), 0.7),
        ((90.0, 30.0, 120.0, 70.5), CHESS.index("black_rook"), 0.6),
    ]

    def test_one_box_comes_back_once_under_its_best_class(self) -> None:
        found = self.detector.detect(_image())
        self.assertEqual(("white_knight", "black_pawn", "black_rook"), found.labels)
        self.assertEqual(1, found.counts["white_knight"])
        self.assertEqual(0, found.counts["white_bishop"])

    def test_classes_picks_from_the_best_classes_alone(self) -> None:
        # The model takes the piece for a knight, so a call for bishops finds none rather than the knight again.
        self.assertEqual((), self.detector.detect(_image(), classes=["white_bishop"]).labels)
        self.assertEqual(("white_knight",), self.detector.detect(_image(), classes=["white_knight"]).labels)

    def test_boxes_that_differ_are_two_objects(self) -> None:
        # Two queries are two objects, even half a pixel apart: only one query's own two classes are one object.
        self.assertEqual(("black_pawn", "black_rook"),
                         self.detector.detect(_image(), classes=["black_pawn", "black_rook"]).labels)


def _found() -> Detections:
    """Two pieces with masks, as ``segment=True`` leaves them, of three classes looked for."""
    pawn = np.zeros((HEIGHT, WIDTH), bool)
    pawn[12:38, 12:28] = True
    king = np.zeros((HEIGHT, WIDTH), bool)
    king[22:58, 52:78] = True
    return Detections(objects=(DetectedObject("white_pawn", 0.95, (10, 10, 30, 40), mask=pawn, mask_score=0.93),
                               DetectedObject("black_king", 0.8, (50, 20, 80, 60), mask=king, mask_score=0.88)),
                      classes=("white_pawn", "white_queen", "black_king"), threshold=0.5, image_hw=(HEIGHT, WIDTH),
                      segmented=True)


class WhatItAnswersWithTests(unittest.TestCase):

    def test_it_is_a_frozen_sequence_of_its_objects(self) -> None:
        found = _found()
        self.assertEqual(2, len(found))
        self.assertEqual("white_pawn", found[0].label)
        self.assertEqual(("white_pawn", "black_king"), tuple(obj.label for obj in found))
        self.assertEqual((found[1],), found[1:])
        with self.assertRaises(AttributeError):
            found.threshold = 0.1  # type: ignore[misc]
        with self.assertRaises(ValueError):
            found[0].mask[0, 0] = True  # type: ignore[index]

    def test_counts_and_of_name_every_class_looked_for(self) -> None:
        found = _found()
        self.assertEqual({"white_pawn": 1, "white_queen": 0, "black_king": 1}, found.counts)
        self.assertEqual((found[1],), found.of(" Black_King"))
        with self.assertRaisesRegex(ValueError, "not a class this detection looked for"):
            found.of("black_pawn")

    def test_render_is_a_line_per_object_and_how_many_of_each_in_ascii(self) -> None:
        text = _found().render()
        self.assertEqual(text, str(_found()))
        self.assertFalse(text.endswith("\n"))
        text.encode("ascii")
        self.assertIn("white_pawn  0.95  box (10, 10) to (30, 40)  mask 416 px, SAM2 IoU 0.93", text)
        self.assertIn("counted    white_pawn 1, black_king 1", text)
        self.assertIn("none of    white_queen", text)
        self.assertIn("L\\xe4ufer", Detections((DetectedObject("Läufer", 0.6, (0, 0, 4, 4)),), ("Läufer",),
                                                0.5, (HEIGHT, WIDTH)).render())

    def test_draw_gives_a_new_image_and_leaves_the_given_one_alone(self) -> None:
        image = _image()
        before = image.copy()
        drawn = _found().draw(image)
        np.testing.assert_array_equal(before, image)
        self.assertEqual(image.shape, drawn.shape)
        self.assertFalse(np.array_equal(drawn[25, 20], image[25, 20]), "the pawn's mask is tinted")
        np.testing.assert_array_equal(image[110, 120], drawn[110, 120])
        with self.assertRaisesRegex(ValueError, "draw them on the image they were found in"):
            _found().draw(np.zeros((HEIGHT + 1, WIDTH, 3), np.uint8))

    def test_a_drawing_is_written_in_the_format_its_suffix_names(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = _found().write_drawing(_image(), Path(tmp) / "out" / "board.png")
            self.assertEqual(b"\x89PNG", path.read_bytes()[:4])
            with self.assertRaisesRegex(ValueError, "names no image format"):
                _found().write_drawing(_image(), Path(tmp) / "board")

    def test_to_dict_is_json_and_reads_back_with_its_masks(self) -> None:
        found = _found()
        data = json.loads(json.dumps(found.to_dict()))
        self.assertEqual("coco_rle", data["mask_encoding"])
        self.assertEqual([HEIGHT, WIDTH], data["objects"][0]["mask"]["size"])
        again = Detections.from_dict(data)
        self.assertEqual(found, again)
        for one, other in zip(found, again):
            np.testing.assert_array_equal(one.mask, other.mask)
        bare = found.to_dict(masks=False)
        self.assertIsNone(bare["mask_encoding"])
        self.assertTrue(all(item["mask"] is None for item in bare["objects"]))
        self.assertEqual([20.0, 25.0], bare["objects"][0]["centre"])

    def test_the_run_length_encoding_is_cocos_column_by_column_background_first(self) -> None:
        # Column by column the pixels read 0 0 | 1 1 | 1 0: two background, three object, one background.
        mask = np.array([[0, 1, 1], [0, 1, 0]], dtype=bool)
        self.assertEqual({"size": [2, 3], "counts": [2, 3, 1]}, encode_mask(mask))
        # A mask whose first pixel is the object starts with no background: 1 0 | 0 1 | 0 0.
        self.assertEqual([0, 1, 2, 1, 2], encode_mask(np.eye(2, 3, dtype=bool))["counts"])
        rng = np.random.default_rng(3)
        for _ in range(5):
            random = rng.random((17, 23)) > 0.6
            np.testing.assert_array_equal(random, decode_mask(encode_mask(random)))
        with self.assertRaisesRegex(ValueError, "add up to 6 pixels"):
            decode_mask({"size": [2, 3], "counts": [2, 3]})
        with self.assertRaisesRegex(ValueError, "compressed form"):
            decode_mask({"size": [2, 3], "counts": "abc"})

    def test_an_object_keeps_its_contract(self) -> None:
        for bad in ({"box": (5, 5, 5, 9)}, {"box": (0, 0, float("inf"), 1)}, {"score": 1.2}, {"label": ""},
                    {"mask": np.zeros(4, bool)}):
            with self.subTest(**{key: repr(value) for key, value in bad.items()}):
                fields = {"label": "white_pawn", "score": 0.9, "box": (0, 0, 4, 4)} | bad
                with self.assertRaises(ValueError):
                    DetectedObject(**fields)
        with self.assertRaisesRegex(ValueError, "the image's own size"):
            Detections((DetectedObject("white_pawn", 0.9, (0, 0, 4, 4), mask=np.zeros((3, 3), bool)),),
                       ("white_pawn",), 0.5, (HEIGHT, WIDTH), True)


class TheDoorTests(_Built):

    def test_the_willy_names_are_the_objects_their_home_module_defines(self) -> None:
        home = import_module("src.models.detection.object_detector")
        for name in ("ObjectDetector", "Detections", "DetectedObject"):
            with self.subTest(name):
                self.assertIs(getattr(home, name), getattr(willy, name))
                self.assertIs(getattr(home, name), getattr(import_module("src.models.detection"), name))
        self.assertIn("ObjectDetector", willy.__all__)

    def test_a_detector_prints_as_its_render_naming_its_weights_classes_and_masks(self) -> None:
        text = self.detector.render()
        self.assertEqual(text, str(self.detector))
        self.assertIn((self.root / "chess").resolve().as_posix(), text)
        self.assertIn("12: white_pawn, white_knight", text)
        self.assertIn("loaded by the first segment=True with a box to cut", text)
        self.assertEqual("cpu", self.detector.device)


if __name__ == "__main__":
    unittest.main()
