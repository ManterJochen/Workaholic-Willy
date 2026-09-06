"""Walk a written dataset and emit ``prompts.jsonl``: no GPU, no Isaac, no re-render.

Runs against what is on disk, which is the point: rendering a dataset costs hours, and the prompt
layer must be re-runnable in seconds when a template or a colour rule changes.

Object extents are not in ``scene.json``, so they are recovered by rebuilding the asset manifest
from the config stored in ``provenance.json``, the same pure-numpy draw the build used. That is
only trustworthy if it really is the same draw, so the rebuilt manifest's hash is checked against
the ``asset_manifest_sha256`` the build recorded, and a mismatch refuses rather than guesses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, PROMPTS_BUILD_LOG_FILE
from datagen.prompts.expressions import (
    Prompt,
    attribute_prompts,
    instruction_prompts,
    scene_facts,
    spatial_prompts,
)
from datagen.render.camera import project_to_pixels

__all__ = ["PromptReport", "build_prompts"]

logger = create_logger("datagen.prompts.build", PROMPTS_BUILD_LOG_FILE, log_dir=DATAGEN_LOG_DIR)


@dataclass(frozen=True, slots=True)
class PromptReport:
    scenes: int
    prompts: int
    by_class: dict[str, int]
    by_language: dict[str, int]
    objects_without_prompt: int
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems and self.prompts > 0

    def summary(self) -> str:
        head = (f"{self.prompts} prompt(s) over {self.scenes} scene(s); "
                f"{dict(sorted(self.by_class.items()))}, {dict(sorted(self.by_language.items()))}; "
                f"{self.objects_without_prompt} object(s) unreachable by any expression")
        if self.ok:
            return f"OK: {head}"
        return f"FAILED: {head}\n  " + "\n  ".join(self.problems[:20])


def manifest_for(root: Path) -> tuple[dict[str, float], list[str]]:
    """``{asset_id: longest extent mm}`` rebuilt from the dataset's own provenance."""
    from datagen.build import build_manifest  # noqa: PLC0415 (only this path needs it)
    from datagen.config import DatagenConfig  # noqa: PLC0415
    from datagen.provenance import sha256_of  # noqa: PLC0415, SLF001

    stamp = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    config = DatagenConfig.from_stamp(stamp["config"])
    manifest = build_manifest(config)
    rows = [record.as_row() for record in manifest]
    problems: list[str] = []
    if sha256_of(rows) != stamp["asset_manifest_sha256"]:
        problems.append(
            "the asset manifest rebuilt from provenance does not hash to the one the build recorded; "
            "the assets have changed under this dataset, so object sizes cannot be trusted"
        )
    if problems:
        # Returned, not raised, and quiet by design: the prompts are still written. This is the
        # one line that says the sizes behind every "small"/"large" in them may be describing
        # other assets.
        logger.error("%s: %s", root, "; ".join(problems))
    return ({record.asset_id: max(record.extent_mm) for record in manifest}, problems)


def build_prompts(root: Path) -> PromptReport:
    """Write ``root/prompts.jsonl``. Returns what was produced and anything that refused."""
    root = Path(root)
    extents, problems = manifest_for(root)
    scenes = prompts_written = objects_without = 0
    by_class: dict[str, int] = {}
    by_language: dict[str, int] = {}
    rows: list[dict] = []

    for scene_dir in sorted((root / "scenes").iterdir()):
        scene_json = scene_dir / "scene.json"
        if not scene_json.exists():
            continue
        payload = json.loads(scene_json.read_text(encoding="utf-8"))
        scenes += 1
        facts = scene_facts(payload)
        if extents:
            facts = [
                type(fact)(fact.index, fact.kind, fact.colour,
                           float(extents.get(payload["spec"]["objects"][fact.index]["asset_id"], 0.0)),
                           fact.position_mm)
                for fact in facts
            ]
        scene_prompts: list[Prompt] = attribute_prompts(scene_dir.name, facts)
        covered = {prompt.target for prompt in scene_prompts}

        if len(covered) < len(facts):
            for view in payload["views"]:
                if str(view.get("outcome")) not in ("rendered", "rendered_after_resample"):
                    continue
                camera_to_base = np.asarray(view["camera_to_base_mm"], dtype=np.float64)
                intrinsics = np.asarray(view["intrinsics"], dtype=np.float64)
                pixels = project_to_pixels(
                    np.asarray([fact.position_mm for fact in facts], dtype=np.float64),
                    camera_to_base, intrinsics,
                )
                scene_prompts.extend(
                    spatial_prompts(scene_dir.name, facts, view["name"], pixels)
                )
            covered = {prompt.target for prompt in scene_prompts}
        objects_without += len(facts) - len(covered)
        # Instructions wrap the finished expressions, so they inherit their uniqueness unchanged.
        scene_prompts.extend(instruction_prompts(scene_prompts, facts))

        for prompt in scene_prompts:
            rows.append(prompt.as_row())
            by_class[prompt.prompt_class] = by_class.get(prompt.prompt_class, 0) + 1
            by_language[prompt.language] = by_language.get(prompt.language, 0) + 1
            prompts_written += 1

    with (root / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    logger.info("%d prompt(s) over %d scene(s) -> %s; by class %s, by language %s",
                prompts_written, scenes, root / "prompts.jsonl", dict(sorted(by_class.items())),
                dict(sorted(by_language.items())))
    if objects_without:
        # Not a failure: some objects genuinely cannot be singled out. It is still the number that
        # decides what a recall measured on this file can mean, so it is said rather than counted.
        logger.warning("%d object(s) got no unique expression in any view", objects_without)
    return PromptReport(scenes, prompts_written, by_class, by_language, objects_without,
                        tuple(problems))


def add_paraphrases(root: Path, *, limit: int | None = None) -> dict:
    """Run the English paraphrase pass over a written ``prompts.jsonl`` and append what survives.

    Separate from :func:`build_prompts` and re-runnable on its own, because the template layer
    costs seconds while this one costs GPU minutes: rebuilding templates must never mean reloading
    a 7B model, and re-running the model must never mean rewriting the templates.

    Only English noun phrases are fed in. Instructions are already a wrapper around those phrases,
    so paraphrasing them would vary two things at once and the class would measure neither.
    """
    from datagen.prompts.colours import name_colour  # noqa: PLC0415
    from datagen.prompts.expressions import KIND_WORDS  # noqa: PLC0415
    from datagen.prompts.paraphrase import paraphrase_rows  # noqa: PLC0415

    root = Path(root)
    rows = [json.loads(line) for line in (root / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    scenes = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "scenes").glob("*/scene.json"))
    }

    rivals: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for scene_id, payload in scenes.items():
        dropped = set(payload.get("dropped_objects", []))
        kinds, colours = [], []
        for index, obj in enumerate(payload["spec"]["objects"]):
            if index in dropped:
                continue
            kinds.append(KIND_WORDS.get(obj["asset_id"].rsplit("_", 1)[-1], ("", ""))[0])
            named = name_colour(tuple(obj["color_rgb"]))
            if named:
                colours.append(named.en)
        rivals[scene_id] = (tuple(kinds), tuple(colours))

    candidates = []
    for row in rows:
        if row["language"] != "en" or row["class"] not in ("attribute", "spatial"):
            continue
        obj = scenes[row["scene_id"]]["spec"]["objects"][row["target"]]
        named = name_colour(tuple(obj["color_rgb"]))
        candidates.append({
            **row,
            "_kind": KIND_WORDS.get(obj["asset_id"].rsplit("_", 1)[-1], ("", ""))[0],
            # The colour counts as discriminating only if the template used it: an expression that
            # never said "red" cannot lose a word it never carried.
            "_colour": named.en if (named and named.en in row["text"]) else None,
            "_size": "large" if "large" in row["text"] else ("small" if "small" in row["text"] else None),
        })

    run = paraphrase_rows(candidates, rivals, limit=limit)
    produced = [
        {**{k: v for k, v in candidates[position].items() if not k.startswith("_")},
         "text": text, "class": "paraphrase", "source": "mistral-7b-instruct-v0.3/4bit/greedy"}
        for position, text in sorted(run.texts.items())
    ]
    with (root / "prompts.jsonl").open("a", encoding="utf-8") as handle:
        for row in produced:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    # The rejection rate by reason is the measurement of what this class is worth, so it is written
    # beside the data rather than printed and lost.
    report = {"attempted": run.attempted, "accepted": run.accepted, "verdicts": run.verdicts}
    (root / "paraphrase_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
