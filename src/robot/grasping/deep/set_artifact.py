"""Writing and reading the trained set generator, so a cell can load what a run produced.

The seam between training and the runtime. It exists as its own module because the file the runtime
opens has to be self-describing: the training card carries the whole plan, but a cell has weights and
nothing else, and a decoder that had to guess what it was decoding would guess wrong quietly.

Every decision the decoder needs travels in the file. An artifact that does not carry its sample
spec makes the calculator rebuild one from defaults, so a net trained on 16,384 points and a 15 mm
graspability radius is served 8,192 and 10 mm, silently, and the skew looks like a bad model.

For this head, because it is conditioned, that list is:

    model config       what shapes to build before the weights will load at all
    sample spec        how the cloud a cell hands it must be built
    step config        how many seeds to propose and what counts as a hit
    gripper            which hand the weights were fitted to

The gripper is the one that fails silently. A net trained on the 2F-85 and served a 140 mm jaw's
vector at inference gets a conditioning input it never saw; nothing raises, the grasps just come out
wrong in a way that reads as a bad model. The name is recorded and `load_set_generator` refuses an
artifact whose gripper this build does not know.

A different `kind` from the binned generator, deliberately. The binned calculator refuses a payload
whose `kind` it does not recognise, so giving this one its own name means that decoder cannot
half-read a set artifact, and this one cannot half-read a binned one. Two heads that share a file
extension and not a format is how a model gets loaded into the wrong decoder.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from src.robot.grasping.deep.corpus.sample import SampleSpec
from src.robot.grasping.deep.net.gripper import JAW_GEOMETRY
from src.robot.grasping.deep.net.generative_head import GenerativeHeadConfig
from src.robot.grasping.deep.net.local_crop import LocalCropConfig
from src.robot.grasping.deep.net.slot_head import SlotHeadConfig
from src.robot.grasping.deep.net.serialized_backbone import SerializedConfig
from src.robot.grasping.deep.net.set_generator import SetGenerator, SetGeneratorConfig
from src.robot.grasping.deep.train.step import SetStepConfig

__all__ = [
    "SET_ARTIFACT_KIND",
    "SET_ARTIFACT_VERSION",
    "LoadedSetGenerator",
    "load_set_generator",
    "write_set_generator",
]

#: Not `grasp_generator`. See the module docstring: a shared name is how a model reaches the wrong
#: decoder, and the binned calculator already refuses a payload whose kind it does not know.
SET_ARTIFACT_KIND = "set_grasp_generator"
SET_ARTIFACT_VERSION = 1

#: What the retired binned generator stamped into its files, kept as literals so that a person
#: holding one gets a straight answer instead of a puzzled one about an unexpected kind. Literals
#: rather than an import: the module that defined them is gone with the family, and importing it
#: back is what this constant exists to avoid.
RETIRED_KINDS = frozenset({"grasp_generator", "grasp_generator_checkpoint"})


class LoadedSetGenerator:
    """A trained net plus every decision its decoder needs, read back from one file."""

    __slots__ = ("net", "sample", "step", "gripper", "grippers", "config", "sha256")

    def __init__(self, net: SetGenerator, sample: SampleSpec, step: SetStepConfig,
                 gripper: str, config: SetGeneratorConfig, sha256: str,
                 grippers: "tuple[str, ...] | None" = None) -> None:
        self.net = net
        self.sample = sample
        self.step = step
        self.gripper = gripper
        self.config = config
        self.sha256 = sha256
        #: Which hands this model saw, as against the one it plans for. `gripper` is the stamp: it
        #: names the hand the artifact was written for. A model fitted across a multi-gripper corpus
        #: saw more, and without this list that fitted generalisation is unreachable: a cell with a
        #: different hand gets proposals for the stamp, silently. An artifact carrying no list reads
        #: back as the stamp alone, which is exactly what it is.
        self.grippers = tuple(grippers) if grippers else (gripper,)


def _model_config(raw: dict[str, Any]) -> SetGeneratorConfig:
    """Rebuild the nested config. Explicit rather than generic, so a renamed field fails here.

    A generic `**raw` would swallow an unknown key or silently drop a renamed one, and the weights
    would then load into shapes that no longer match what produced them. `torch.load` would either
    refuse loudly or, worse, accept a subset.
    """
    backbone = raw["backbone"]
    head = raw["head"]
    return SetGeneratorConfig(
        backbone=SerializedConfig(
            in_features=int(backbone["in_features"]), width=int(backbone["width"]),
            depth=int(backbone["depth"]), heads=int(backbone["heads"]),
            window=int(backbone["window"]), mlp_ratio=float(backbone["mlp_ratio"]),
            dropout=float(backbone["dropout"]), position=str(backbone["position"]),
            axis_orders=tuple(tuple(int(a) for a in order)      # type: ignore[misc]
                              for order in backbone["axis_orders"])),
        # `asdict(net.config)` writes every field of `SlotHeadConfig`, so a field this call skips is
        # a field the reader ignores, and the two ways of ignoring one are opposite:
        #
        # * `slot_mixing` changes the parameters. `film` adds `head.film` and `mlp` replaces the
        #   output Linear with a Sequential, so a strict load raises `Unexpected key(s) ... "film"`.
        #   Measured: affine 13 keys, film 14, mlp 15 with 2 of affine's missing. A file trained
        #   with `--slot-mixing film` and rebuilt without it opens nowhere.
        # * `axis_mode` is shape-neutral by design (slot_head.py keeps all fourteen outputs under
        #   both modes), so a `vector`-trained net loads into a `director`-rebuilt one with no error
        #   and has six numbers decoded by the wrong rule. That one reads as a bad model.
        #
        # An enumeration of `SetGeneratorConfig`'s own fields cannot see inside the nested config:
        # `head=` appears here whatever the head config is missing, so the enumeration has to
        # recurse.
        #
        # `.get` with the field default, not a required key: an artifact written before a field
        # existed does not carry it, and absent must mean the behaviour that artifact was trained
        # with, which is the default.
        head=SlotHeadConfig(
            slots=int(head["slots"]), width=int(head["width"]),
            gripper_width=int(head["gripper_width"]), query_width=int(head["query_width"]),
            dropout=float(head["dropout"]), offset_scale_m=float(head["offset_scale_m"]),
            width_centre_m=float(head["width_centre_m"]),
            width_scale_m=float(head["width_scale_m"]),
            offset_from_width=bool(head.get("offset_from_width", True)),
            slot_mixing=str(head.get("slot_mixing", "affine")),
            axis_mode=str(head.get("axis_mode", "director")),
            # An affordance net carries `head.affordance.weight` and `head.affordance.bias`, which a
            # role-less rebuild has no slot for, so a strict load raises exactly as `film` does.
            part_roles=tuple(str(role) for role in head.get("part_roles", ()))),
        # Stage 3 round-trips or it is undeployable. A crop-trained net writes a state dict with six
        # extra tensors and a wider head trunk, and a config that rebuilds neither makes
        # `load_state_dict` refuse with a size mismatch: trainable but unloadable. `None` is off,
        # which is what an artifact carrying no `crop` was trained with.
        crop=(LocalCropConfig(
            radius_mm=float(raw["crop"]["radius_mm"]),
            neighbours=int(raw["crop"]["neighbours"]),
            width=int(raw["crop"]["width"])) if raw.get("crop") else None),
        # Stage 4b round-trips too. `_model_config` is explicit by design, so it has to be extended
        # for every field added to the config; a field this function does not rebuild is a net that
        # trains and then cannot be loaded.
        generative=(GenerativeHeadConfig(
            width=int(raw["generative"]["width"]),
            depth=int(raw["generative"]["depth"]),
            gripper_width=int(raw["generative"]["gripper_width"]),
            time_width=int(raw["generative"]["time_width"]),
            steps=int(raw["generative"]["steps"]),
            samples=int(raw["generative"]["samples"])) if raw.get("generative") else None),
        graspability_width=int(raw["graspability_width"]))


def write_set_generator(directory: str | Path, net: SetGenerator, *, sample: SampleSpec,
                        step: SetStepConfig, gripper: str,
                        trained_grippers: "Sequence[str] | None" = None,
                        report: dict[str, Any] | None = None,
                        name: str = "set_grasp_generator_v1") -> dict[str, Path]:
    """Write the weights and a card beside them. Returns both paths.

    The gripper is a required argument rather than a default. A 2F-85 default is wrong, and wrong
    silently, for the first run on a varied corpus.
    """
    if gripper not in JAW_GEOMETRY:
        raise ValueError(f"unknown gripper {gripper!r}; known: {', '.join(sorted(JAW_GEOMETRY))}")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    weights = out / f"{name}.pt"
    payload = {
        "kind": SET_ARTIFACT_KIND,
        "artifact_version": SET_ARTIFACT_VERSION,
        "config": asdict(net.config),
        "sample": asdict(sample),
        "step": asdict(step),
        "gripper": gripper,
        # The hands this run actually trained across, beside the one it plans for. Without this an
        # artifact cannot say it saw more than one, and the runtime conditions on the stamp whatever
        # hand the cell has bolted on. A model fitted across `narrow_55`, `slim_pad` and `wide_140`
        # and stamped `slim_pad` can still be conditioned on the cell's own hand; without the list
        # that fitted generalisation is unreachable.
        "trained_grippers": list(trained_grippers or (gripper,)),
        "state_dict": {key: value.cpu() for key, value in net.state_dict().items()},
    }
    torch.save(payload, weights)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    card = out / f"{name}.card.json"
    card.write_text(json.dumps({
        "kind": SET_ARTIFACT_KIND, "artifact_version": SET_ARTIFACT_VERSION,
        "sha256": digest, "gripper": gripper,
        "trained_grippers": list(trained_grippers or (gripper,)),
        "parameters": net.parameter_count,
        "config": asdict(net.config), "sample": asdict(sample), "step": asdict(step),
        "report": report or {},
    }, indent=2, sort_keys=True), encoding="utf-8")
    return {"weights": weights, "card": card}


def load_set_generator(path: str | Path, *, device: str = "cpu") -> LoadedSetGenerator:
    """Read an artifact back, refusing anything it cannot decode correctly.

    Four refusals, each for a way a load would otherwise go wrong silently: the wrong `kind` (a
    binned artifact in a set decoder), a newer `artifact_version` than this build knows, a gripper
    this build cannot resolve to a vector, and a state dict that does not fit the config it
    travelled with.
    """
    file = Path(path)
    # `weights_only=True`. `False` hands `pickle` the whole file and executes whatever is in it,
    # and the package's own docstring says "the artifact is a `state_dict`, not a pickle". Every
    # artifact reader in the package comes through here, so this one call covers all of them; the
    # only other `torch.load` calls read a run's own training checkpoints in `train/trainer.py`.
    #
    # True is enough for what this module writes, and a payload carrying anything exotic would start
    # refusing: `logs/dl/models/chain_proof/set_grasp_generator_v1.pt` loads under True and yields
    # all seven keys (kind, artifact_version, config, sample, step, gripper, state_dict). They are
    # plain containers, ints, floats and strings.
    payload = torch.load(file, map_location=device, weights_only=True)
    kind = payload.get("kind")
    if kind in RETIRED_KINDS:
        # The reader is where this belongs, because it is the only place that knows the kind. A
        # re-wrap in `calculator.py` would have to recover the kind by splitting the message this
        # branch produces.
        #
        # Membership, not a substring: 'grasp_generator' is a substring of 'set_grasp_generator',
        # so a substring check here would refuse the family that ships.
        raise ValueError(
            f"{file.name} was written by the binned grasp generator ({kind!r}), an architecture "
            f"this project retired on 2026-09-04. Nothing decodes it any more, deliberately: its "
            f"numbers came from a different model and are not comparable with anything this build "
            f"reports. Train a replacement with `python -m src.robot.grasping.deep "
            f"train-set`.")
    if kind != SET_ARTIFACT_KIND:
        raise ValueError(f"{file.name} is a {kind!r} artifact, not {SET_ARTIFACT_KIND!r}. A fold "
                         f"checkpoint from a run in progress carries no kind at all; point at the "
                         f"artifact a finished run writes.")
    version = int(payload.get("artifact_version", 0))
    if version > SET_ARTIFACT_VERSION:
        raise ValueError(f"{file.name} is artifact version {version}, this build knows "
                         f"{SET_ARTIFACT_VERSION}. Refusing rather than reading a newer format "
                         f"through an older decoder.")
    gripper = str(payload.get("gripper", ""))
    if gripper not in JAW_GEOMETRY:
        raise ValueError(f"{file.name} was trained for gripper {gripper!r}, which this build cannot "
                         f"resolve. A wrong conditioning vector produces grasps that read as a bad "
                         f"model rather than as a wrong hand.")
    config = _model_config(payload["config"])
    net = SetGenerator(config).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return LoadedSetGenerator(
        net=net, sample=SampleSpec(**payload["sample"]),
        step=SetStepConfig(**{k: v for k, v in payload["step"].items()
                              if k not in ("shares", "weights")}),
        gripper=gripper, config=config,
        # An artifact with no list reads back as its stamp alone, which is the truthful answer for
        # it: nothing recorded what it saw, so the only hand it can claim is the one it was written
        # for.
        grippers=tuple(str(g) for g in payload.get("trained_grippers") or (gripper,)),
        sha256=hashlib.sha256(file.read_bytes()).hexdigest())
