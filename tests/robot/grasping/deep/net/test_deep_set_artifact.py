"""Writing and reading a trained set generator.

⚠ THE FAILURES HERE ARE ALL SILENT ONES. An artifact that loads but decodes under different
assumptions than it was trained with does not raise; it produces grasps that read as a bad model.
This repository has paid for exactly that once: an artifact that did not carry its sample spec had
the calculator rebuild one from DEFAULTS, so a net trained on 16,384 points and a 15 mm graspability
radius was served 8,192 and 10 mm.

The conditioned head adds a new one. A net fitted to the 2F-85 and served a 140 mm jaw's vector at
inference gets an input it never saw, and nothing anywhere raises.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.robot.grasping.deep.corpus.sample import SampleSpec
from src.robot.grasping.deep.net.slot_head import SlotHeadConfig
from src.robot.grasping.deep.net.serialized_backbone import SerializedConfig
from src.robot.grasping.deep.net.set_generator import SetGenerator, SetGeneratorConfig
from src.robot.grasping.deep.set_artifact import (
    SET_ARTIFACT_KIND,
    load_set_generator,
    write_set_generator,
)
from src.robot.grasping.deep.train.step import SetStepConfig


def _net() -> SetGenerator:
    return SetGenerator(SetGeneratorConfig(
        backbone=SerializedConfig(width=32, depth=2, heads=4, window=64),
        head=SlotHeadConfig(slots=3, width=64, gripper_width=8, query_width=8)))


def _write(directory: Path, gripper: str = "2f85", **kwargs: object) -> dict[str, Path]:
    return write_set_generator(directory, _net(), sample=SampleSpec(grasp_set=True, points=512),
                               step=SetStepConfig(seeds=16), gripper=gripper,
                               **kwargs)  # type: ignore[arg-type]


class RoundTripTests(unittest.TestCase):

    def test_the_weights_come_back_identical(self) -> None:
        with TemporaryDirectory() as tmp:
            net = _net()
            paths = write_set_generator(Path(tmp), net, sample=SampleSpec(grasp_set=True),
                                        step=SetStepConfig(), gripper="2f85")
            loaded = load_set_generator(paths["weights"])
            for (name, original), (_, restored) in zip(net.state_dict().items(),
                                                       loaded.net.state_dict().items(),
                                                       strict=True):
                with self.subTest(name):
                    self.assertTrue(torch.equal(original.cpu(), restored.cpu()))

    def test_every_decision_the_decoder_needs_survives(self) -> None:
        """⛔ The sample spec is the one this repository lost once: a net trained on a 15 mm
        graspability radius served at 10 mm looks like a bad model, not like a wrong setting."""
        with TemporaryDirectory() as tmp:
            paths = _write(Path(tmp))
            loaded = load_set_generator(paths["weights"])
            self.assertEqual(loaded.sample.points, 512)
            self.assertTrue(loaded.sample.grasp_set)
            self.assertEqual(loaded.step.seeds, 16)
            self.assertEqual(loaded.gripper, "2f85")
            self.assertEqual(loaded.config.head.slots, 3)

    def test_the_loaded_net_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            loaded = load_set_generator(_write(Path(tmp))["weights"])
            encoded = loaded.net.encode(torch.rand(1, 64, 3) * 0.5, torch.randn(1, 64, 5))
            self.assertEqual(encoded.shape[-1], loaded.net.backbone.out_features)
            self.assertFalse(loaded.net.training, "a loaded artifact must be in eval mode")

    def test_the_card_names_the_gripper_and_the_digest(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = _write(Path(tmp), report={"epochs": []})
            card = json.loads(paths["card"].read_text(encoding="utf-8"))
            self.assertEqual(card["kind"], SET_ARTIFACT_KIND)
            self.assertEqual(card["gripper"], "2f85")
            self.assertEqual(len(card["sha256"]), 64)
            self.assertGreater(card["parameters"], 0)
            self.assertIn("report", card)

    def test_the_digest_matches_the_file(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = _write(Path(tmp))
            card = json.loads(paths["card"].read_text(encoding="utf-8"))
            self.assertEqual(load_set_generator(paths["weights"]).sha256, card["sha256"])


class RefusalTests(unittest.TestCase):
    """Four refusals, each standing for a way this fails silently."""

    def test_a_binned_artifact_is_refused_and_the_RETIREMENT_is_named(self) -> None:
        """⛔ Two heads that share a file extension and not a format is how a model reaches the wrong
        decoder.

        The message matters as much as the refusal. Until 2026-09-04 both families were loadable and
        this said only that they "do not share a decoder", which was true and no longer useful: the
        binned family was retired that day, so the person holding the file needs to be told the
        architecture is gone and what to run instead. A refusal that merely reports an unexpected
        kind sends them looking for a loader that was deleted on purpose.
        """
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            torch.save({"kind": "grasp_generator", "state_dict": {}}, path)
            with self.assertRaises(ValueError) as raised:
                load_set_generator(path)
        message = str(raised.exception)
        # The family is named in prose, not shouted: the migration lowered the shout and kept every
        # element of the refusal (commit note `.commits/robot/08-grasping-deep.md`, the prose
        # convention). The phrase is pinned in full rather than one word of it.
        self.assertIn("binned grasp generator", message, "the refusal no longer names the family")
        self.assertIn("retired on 2026-09-04", message, "the retirement is no longer dated")
        self.assertIn("train-set", message, "the refusal has to say what to run instead")

    def test_a_binned_CHECKPOINT_is_refused_by_the_same_branch(self) -> None:
        """The trainer stamped a second kind, and a half-finished run is the file most likely to be
        lying around. Both retired kinds go through the one branch, so neither can drift."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.ckpt.pt"
            torch.save({"kind": "grasp_generator_checkpoint", "state_dict": {}}, path)
            with self.assertRaises(ValueError) as raised:
                load_set_generator(path)
        message = str(raised.exception)
        self.assertIn("binned grasp generator ('grasp_generator_checkpoint')", message,
                      "the checkpoint kind does not reach the one refusal branch")
        self.assertIn("retired on 2026-09-04", message, "the retirement is no longer dated")

    def test_the_LIVE_kind_is_not_caught_by_the_retirement_branch(self) -> None:
        """⚠ THE SUBSTRING TRAP, CONTROLLED FOR. 'grasp_generator' IS a substring of
        'set_grasp_generator', so a membership test written with `in` would refuse the family that
        ships. This project has already shipped that exact shape once, in the licence audit, where a
        permissive substring matched the forbidden superstring. A real artifact loads here, which is
        the only proof that the branch discriminates rather than merely matching.
        """
        with TemporaryDirectory() as tmp:
            paths = _write(Path(tmp))
            loaded = load_set_generator(paths["weights"])
        self.assertIsNotNone(loaded.net)

    def test_a_newer_artifact_version_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = _write(Path(tmp))
            payload = torch.load(paths["weights"], weights_only=False)
            payload["artifact_version"] = 99
            torch.save(payload, paths["weights"])
            with self.assertRaises(ValueError) as raised:
                load_set_generator(paths["weights"])
            self.assertIn("artifact version 99", str(raised.exception))

    def test_an_unknown_gripper_is_refused_at_write_and_at_read(self) -> None:
        """⭐ THE NEW ONE. A conditioned head served a hand it never saw produces grasps that read as
        a bad model rather than as a wrong input, and nothing raises anywhere else."""
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _write(Path(tmp), gripper="a_hand_we_never_trained_on")

            paths = _write(Path(tmp))
            payload = torch.load(paths["weights"], weights_only=False)
            payload["gripper"] = "a_hand_we_never_trained_on"
            torch.save(payload, paths["weights"])
            with self.assertRaises(ValueError) as raised:
                load_set_generator(paths["weights"])
            self.assertIn("cannot", str(raised.exception).lower())

    def test_a_config_that_does_not_fit_the_weights_is_refused(self) -> None:
        """A silently mismatched shape would load a subset of the weights and leave the rest at their
        initialisation, which is a model that half-works and looks trained."""
        with TemporaryDirectory() as tmp:
            paths = _write(Path(tmp))
            payload = torch.load(paths["weights"], weights_only=False)
            payload["config"]["head"]["slots"] = 7
            torch.save(payload, paths["weights"])
            with self.assertRaises(RuntimeError):
                load_set_generator(paths["weights"])

    def test_a_renamed_config_field_fails_loudly(self) -> None:
        """The rebuild is explicit rather than `**raw`, so a renamed field raises here instead of
        loading weights into shapes that no longer match what produced them."""
        with TemporaryDirectory() as tmp:
            paths = _write(Path(tmp))
            payload = torch.load(paths["weights"], weights_only=False)
            payload["config"]["head"].pop("slots")
            torch.save(payload, paths["weights"])
            with self.assertRaises(KeyError):
                load_set_generator(paths["weights"])


class TheHeadArchitectureRoundTripsTests(unittest.TestCase):
    """⛔⛔ THE READER REBUILT EIGHT OF `SlotHeadConfig`'s ELEVEN FIELDS.

    `asdict(net.config)` wrote all eleven from the start, so the data was never missing: the reader
    ignored `offset_from_width`, `slot_mixing` and `axis_mode`. The two failure modes are opposite,
    and BOTH matter, which is why this class exists rather than one more assertion above.

    * `slot_mixing` changes the PARAMETERS. MEASURED: affine 13 state-dict keys, `film` 14 (it adds
      `head.film`), `mlp` 15 with two of affine's replaced. A strict load of a `film` state dict into
      an affine-rebuilt net raises `Unexpected key(s) in state_dict: "film"`.
    * `axis_mode` is SHAPE-NEUTRAL BY DESIGN. `local_head.py` keeps all fourteen outputs under both
      modes, so a `vector`-trained net loaded into a `director`-rebuilt one raises NOTHING and has
      six numbers decoded by the wrong rule. It reads as a bad model, which is the worse outcome.

    ⚠ WHAT MADE THIS EXPENSIVE: `docs/runbooks/train_your_own_generator.md` put
    `--slot-mixing film` in the ONE training command it gives a customer. The documented path trained
    for hours and produced a file that `propose` and `calculator: deep` both refused to open.

    ⚠ AND NOTHING WAS LOST WHEN IT WAS FIXED. The defect was entirely on the READ side, so every
    artifact already on disk carries its three fields and became loadable the moment the reader was
    corrected.
    """

    def _round_trip(self, directory: Path, mixing: str, axis: str) -> SlotHeadConfig:
        net = SetGenerator(SetGeneratorConfig(
            backbone=SerializedConfig(width=32, depth=2, heads=4, window=64),
            head=SlotHeadConfig(slots=3, width=64, gripper_width=8, query_width=8,
                                 slot_mixing=mixing, axis_mode=axis)))
        name = f"head_{mixing}_{axis}"
        write_set_generator(directory, net, sample=SampleSpec(grasp_set=True, points=512),
                            step=SetStepConfig(seeds=16), gripper="2f85", name=name)
        return load_set_generator(directory / f"{name}.pt").net.config.head

    def test_every_slot_mixing_and_axis_mode_survives_the_artifact(self) -> None:
        """⭐ The whole grid, because the two fields fail differently and a customer can set both."""
        with TemporaryDirectory() as name:
            for mixing in ("affine", "film", "mlp"):
                for axis in ("director", "vector"):
                    with self.subTest(slot_mixing=mixing, axis_mode=axis):
                        head = self._round_trip(Path(name), mixing, axis)
                        self.assertEqual(head.slot_mixing, mixing)
                        self.assertEqual(head.axis_mode, axis)

    def test_offset_from_width_survives_too(self) -> None:
        """The third dropped field. Off is the non-default, so it is the one that proves the read."""
        with TemporaryDirectory() as name:
            directory = Path(name)
            net = SetGenerator(SetGeneratorConfig(
                backbone=SerializedConfig(width=32, depth=2, heads=4, window=64),
                head=SlotHeadConfig(slots=3, width=64, gripper_width=8, query_width=8,
                                     offset_from_width=False)))
            write_set_generator(directory, net, sample=SampleSpec(grasp_set=True, points=512),
                                step=SetStepConfig(seeds=16), gripper="2f85", name="off")
            head = load_set_generator(directory / "off.pt").net.config.head

        self.assertFalse(head.offset_from_width)

    def test_a_FILM_state_dict_really_would_not_fit_an_AFFINE_net(self) -> None:
        """The control that makes the round-trip mean something. If the two nets were
        interchangeable, the fix above would be decoration rather than a repair."""
        from src.robot.grasping.deep.net.slot_head import SlotGraspHead

        affine = SlotGraspHead(64, SlotHeadConfig(slot_mixing="affine"))
        film = SlotGraspHead(64, SlotHeadConfig(slot_mixing="film"))

        self.assertIn("film", set(film.state_dict()) - set(affine.state_dict()))
        with self.assertRaises(RuntimeError) as caught:
            affine.load_state_dict(film.state_dict())
        self.assertIn("film", str(caught.exception))

    def test_an_older_artifact_without_the_fields_still_loads(self) -> None:
        """⚠ ABSENT MUST MEAN THE BEHAVIOUR IT WAS TRAINED WITH, which is the default. Reading them
        as required keys would refuse every artifact written before the fields existed, turning a
        read-side repair into a data loss."""
        with TemporaryDirectory() as name:
            directory = Path(name)
            paths = _write(directory)
            payload = torch.load(paths["weights"], map_location="cpu", weights_only=False)
            for gone in ("offset_from_width", "slot_mixing", "axis_mode"):
                payload["config"]["head"].pop(gone, None)
            torch.save(payload, paths["weights"])
            head = load_set_generator(paths["weights"]).net.config.head

        self.assertEqual(head.slot_mixing, "affine")
        self.assertEqual(head.axis_mode, "director")
        self.assertTrue(head.offset_from_width)


if __name__ == "__main__":
    unittest.main()
