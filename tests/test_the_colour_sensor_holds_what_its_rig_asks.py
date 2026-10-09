"""The colour sensor holds what its rig asks: ``realsense.color``, written as the stream opens and read back.

The owner, 2026-10-09 ("Config-Block im Code"): the D415's colour camera ran on full auto, and the same parts came out
1 to 1.8 stops brighter or darker with what else was in view at each look, orange clipping in red and reading as yellow,
grey cubes drifting over the colour check's grey/white limit (the image review of 31 looks of the owner's cell). The
block fixes exposure, gain and white balance at the values auto settles on at look 0; every key null, as shipped, is
today's full auto and writes nothing.

What is pinned, against a stand-in SDK (``tests/_realsense_fakes.py``) whose names are held against the real one in
``test_a_rig_recording_for_research_keeps_its_infrared_images.py``:

  * the order: each auto mode goes off before its value is written, because librealsense writes the exposure's default
    as auto exposure goes off, so an exposure written first would be overwritten;
  * the read-back once the warm-up frames have run, logged, also with every key null, and a warning for a value the
    sensor holds otherwise (Windows holds a colour exposure in powers of two of a second);
  * the refusals, each before anything is written: a value outside the sensor's range, an option it does not offer, a
    camera whose colour comes off its depth imagers;
  * the schema: a fixed value needs its auto mode written off beside it, and the reference documents the defaults.
"""

from __future__ import annotations

import logging
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.camera.setup.image_taking.rgbd import RealSenseRGBDStreamer
from src.config.schema.camera import RealSenseColorConfig, RealSenseConfig
from tests._realsense_fakes import OPTION, ColourSensor, Device, Pipeline, rig, sdk

_LOGGER = "src.camera.setup.image_taking.rgbd"
_ROOT = Path(__file__).resolve().parents[1]

#: The block the owner's Monday block writes: auto off, then the values auto settled on at look 0.
_FIXED = {"auto_exposure": False, "exposure": 312.0, "gain": 32.0, "auto_white_balance": False, "white_balance": 4000.0}


class _Opened(unittest.TestCase):

    def _open(self, colour: ColourSensor | None, **block: object) -> tuple[RealSenseRGBDStreamer, Pipeline, list]:
        pipeline = Pipeline(Device(colour=colour))
        streamer = RealSenseRGBDStreamer(rig(color=dict(block)), rs_module=sdk(pipeline))
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            streamer.open()
        self.addCleanup(streamer.release)
        return streamer, pipeline, logs.records

    def _refused(self, colour: ColourSensor | None, **block: object) -> tuple[str, Pipeline]:
        pipeline = Pipeline(Device(colour=colour))
        streamer = RealSenseRGBDStreamer(rig(color=dict(block)), rs_module=sdk(pipeline))
        with self.assertRaises(RuntimeError) as caught:
            streamer.open()
        self.assertTrue(pipeline.stopped, "a refused open left the pipeline running")
        self.assertFalse(streamer.is_opened())
        return str(caught.exception), pipeline


class EveryKeyNullTests(_Opened):

    def test_writes_nothing_and_says_what_auto_holds(self) -> None:
        colour = ColourSensor()
        streamer, _, records = self._open(colour)
        self.assertEqual([], colour.written, "a block of nulls wrote to the colour sensor")
        self.assertEqual({"auto_exposure": 1.0, "exposure": 156.0, "gain": 64.0, "auto_white_balance": 1.0,
                          "white_balance": 4600.0}, streamer.color_held)
        said = next(r.getMessage() for r in records if "colour sensor holds" in r.getMessage())
        for part in ("auto exposure on", "exposure 156 (15.6 ms)", "gain 64", "auto white balance on",
                     "white balance 4600 K", "wrote nothing"):
            with self.subTest(part=part):
                self.assertIn(part, said)
        self.assertEqual([], [r.getMessage() for r in records if r.levelno >= logging.WARNING])

    def test_a_camera_without_a_colour_sensor_of_its_own_opens_and_says_so(self) -> None:
        for colour in (None, ColourSensor(is_depth=True)):
            with self.subTest(colour="none named" if colour is None else "the depth sensor"):
                streamer, _, records = self._open(colour)
                self.assertTrue(streamer.is_opened())
                self.assertTrue(any("no colour sensor of its own" in r.getMessage() for r in records))
                self.assertEqual(set(streamer.color_held.values()), {None})
                if colour is not None:
                    self.assertEqual([], colour.written)


class TheBlockIsWrittenTests(_Opened):

    def test_each_auto_mode_goes_off_before_its_value_and_the_sensor_holds_the_values(self) -> None:
        colour = ColourSensor()
        streamer, _, records = self._open(colour, **_FIXED)
        self.assertEqual([("enable_auto_exposure", 0.0), ("exposure", 312.0), ("gain", 32.0),
                          ("enable_auto_white_balance", 0.0), ("white_balance", 4000.0)], colour.written)
        self.assertEqual({"auto_exposure": 0.0, "exposure": 312.0, "gain": 32.0, "auto_white_balance": 0.0,
                          "white_balance": 4000.0}, streamer.color_held)
        said = next(r.getMessage() for r in records if "colour sensor holds" in r.getMessage())
        self.assertIn("realsense.color wrote auto exposure off, exposure 312 (31.2 ms), gain 32", said)
        self.assertEqual([], [r.getMessage() for r in records if r.levelno >= logging.WARNING])

    def test_the_exposure_written_after_auto_goes_off_is_the_one_held(self) -> None:
        """librealsense writes the exposure's default as auto exposure goes off: written before, 312 would be lost."""
        colour = ColourSensor()
        self._open(colour, auto_exposure=False, exposure=312.0)
        self.assertEqual(312.0, colour.held["exposure"])
        before = ColourSensor()
        before.set_option(OPTION.exposure, 312.0)
        before.set_option(OPTION.enable_auto_exposure, 0.0)
        self.assertEqual(166.0, before.held["exposure"], "the double no longer writes the default as auto goes off")

    def test_a_block_that_sets_only_the_auto_modes_writes_only_them(self) -> None:
        colour = ColourSensor()
        self._open(colour, auto_exposure=True, auto_white_balance=True)
        self.assertEqual([("enable_auto_exposure", 1.0), ("enable_auto_white_balance", 1.0)], colour.written)

    def test_a_value_the_sensor_holds_otherwise_is_warned_about_with_what_it_holds(self) -> None:
        colour = ColourSensor(powers_of_two=True)
        streamer, _, records = self._open(colour, auto_exposure=False, exposure=166.0)
        self.assertEqual(156.0, streamer.color_held["exposure"])
        warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
        self.assertEqual(1, len(warnings), warnings)
        for part in ("holds exposure 156 (15.6 ms) after exposure 166 (16.6 ms) was written", "realsense.color.exposure",
                     "powers of two"):
            with self.subTest(part=part):
                self.assertIn(part, warnings[0])

    def test_the_values_are_written_before_the_warm_up_frames_and_read_after_them(self) -> None:
        colour = ColourSensor()
        pipeline = Pipeline(Device(colour=colour))
        waits_at_write: list[int] = []
        original = colour.set_option

        def set_option(option: object, value: float) -> None:
            waits_at_write.append(pipeline.waits)
            original(option, value)

        colour.set_option = set_option  # type: ignore[method-assign]
        streamer = RealSenseRGBDStreamer(rig(color=dict(_FIXED), warmup=3), rs_module=sdk(pipeline))
        streamer.open()
        self.addCleanup(streamer.release)
        self.assertEqual([0] * 5, waits_at_write)
        self.assertEqual(3, pipeline.waits)


class TheBlockIsRefusedTests(_Opened):

    def test_a_value_outside_the_sensors_range_refuses_and_writes_nothing(self) -> None:
        colour = ColourSensor()
        text, _ = self._refused(colour, auto_exposure=False, exposure=20000.0)
        for part in ("camera.cameras.rigs['wrist'].realsense.color.exposure is 20000", "outside the 1 to 10000",
                     "Intel RealSense D415"):
            with self.subTest(part=part):
                self.assertIn(part, text)
        self.assertEqual([], colour.written, "a refused block wrote to the camera, which keeps it until power-off")

    def test_a_white_balance_outside_the_range_refuses_after_nothing_was_written(self) -> None:
        colour = ColourSensor()
        text, _ = self._refused(colour, **{**_FIXED, "white_balance": 9000.0})
        self.assertIn("white_balance is 9000, outside the 2800 to 6500", text)
        self.assertEqual([], colour.written)

    def test_an_option_the_colour_sensor_does_not_offer_refuses_naming_it_and_the_camera(self) -> None:
        colour = ColourSensor()
        del colour.ranges["white_balance"]
        text, _ = self._refused(colour, auto_white_balance=False, white_balance=4000.0)
        for part in ("realsense.color.white_balance is 4000.0", "does not offer white_balance",
                     "Intel RealSense D415"):
            with self.subTest(part=part):
                self.assertIn(part, text)
        self.assertEqual([], colour.written)

    def test_a_camera_whose_colour_comes_off_its_depth_imagers_refuses_a_block(self) -> None:
        for colour in (None, ColourSensor(is_depth=True)):
            with self.subTest(colour="none named" if colour is None else "the depth sensor"):
                text, _ = self._refused(colour, **_FIXED)
                self.assertIn("has no colour sensor of its own", text)
                self.assertIn("auto_exposure, exposure, gain, auto_white_balance, white_balance", text)
                if colour is not None:
                    self.assertEqual([], colour.written)


class TheSchemaTests(unittest.TestCase):

    def test_every_key_is_null_by_default(self) -> None:
        self.assertEqual({"auto_exposure": None, "exposure": None, "gain": None, "auto_white_balance": None,
                          "white_balance": None}, RealSenseConfig().color.model_dump())

    def test_a_fixed_value_needs_its_auto_mode_written_off_beside_it(self) -> None:
        for block, auto in (({"exposure": 156.0}, "auto_exposure"), ({"gain": 64.0, "auto_exposure": True}, "auto_exposure"),
                            ({"white_balance": 4600.0}, "auto_white_balance")):
            with self.subTest(block=block):
                with self.assertRaises(ValidationError) as caught:
                    RealSenseColorConfig(**block)
                self.assertIn(f"`{auto}: false`", str(caught.exception))
        RealSenseColorConfig(**_FIXED)

    def test_an_exposure_or_a_white_balance_of_nothing_is_refused(self) -> None:
        for block in ({"auto_exposure": False, "exposure": 0.0}, {"auto_white_balance": False, "white_balance": 0.0},
                      {"auto_exposure": False, "gain": -1.0}):
            with self.subTest(block=block), self.assertRaises(ValidationError):
                RealSenseColorConfig(**block)

    def test_the_reference_documents_the_realsense_block_with_its_defaults(self) -> None:
        data = yaml.safe_load((_ROOT / "config" / "all_keys" / "camera" / "cam.yaml").read_text(encoding="utf-8"))
        (written,) = [entry["realsense"] for entry in data["cameras"]["rigs"] if "realsense" in entry]
        self.assertEqual(RealSenseConfig().model_dump(), written)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
