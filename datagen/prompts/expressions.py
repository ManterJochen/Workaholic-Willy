"""Referring expressions that are unique by construction, in German and English.

The contract is one sentence: the target id never comes from language. It comes from the scene
facts, and an expression is emitted only after checking that exactly one object in the scene
matches it. That ordering is what makes the output ground truth rather than a plausible caption,
and it is what lets a paraphrase step run later without endangering the labels: the paraphrase may
change wording, and the same uniqueness check runs again on the result.

Two prompt classes, differing in what they are valid for:

* attribute: colour, shape, size. True from every camera, so one expression serves all views.
* spatial: "the can left of the blue carton". Left-of is an image-space relation, so an expression
  is valid for exactly one view and records which. That cost buys the objects attributes cannot
  single out.

Colour is used only where :mod:`datagen.prompts.colours` will defend the name; shape is always
available, because the asset was authored as that kind: the English word is true by construction,
and the German is a faithful translation of it rather than a second judgement about the geometry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from datagen.prompts.colours import ColourName, name_colour

__all__ = [
    "KIND_WORDS",
    "ObjectFacts",
    "Prompt",
    "attribute_prompts",
    "scene_facts",
    "spatial_prompts",
]

#: English kind -> (English noun, German noun). The English side is the authored kind, so it is
#: true by construction; the German side is a translation of that word and nothing more. A wrong
#: entry here silently poisons every German prompt, so this is data rather than string formatting,
#: and a new entry needs a native speaker's check.
KIND_WORDS: dict[str, tuple[str, str]] = {
    "box": ("box", "Box"),
    "cylinder": ("cylinder", "Zylinder"),
    "sphere": ("sphere", "Kugel"),
    "capsule": ("capsule", "Kapsel"),
    "tube": ("tube", "Rohr"),
    "bracket": ("bracket", "Winkel"),
    "flange": ("flange", "Flansch"),
    "plate": ("plate", "Platte"),
    "fastener": ("fastener", "Schraube"),
    "carton": ("carton", "Karton"),
    "blister": ("blister pack", "Blisterpackung"),
    "pouch": ("pouch", "Beutel"),
    "bottle": ("bottle", "Flasche"),
    "cup": ("cup", "Becher"),
    "can": ("can", "Dose"),
    "bowl": ("bowl", "Schale"),
}
#: German articles, because "die Dose" and "der Becher" are not interchangeable and a wrong one
#: makes the whole prompt read as machine output, which would itself bias a language benchmark.
_GERMAN_ARTICLE: dict[str, str] = {
    "Box": "die", "Zylinder": "der", "Kugel": "die", "Kapsel": "die", "Rohr": "das",
    "Winkel": "der", "Flansch": "der", "Platte": "die", "Schraube": "die", "Karton": "der",
    "Blisterpackung": "die", "Beutel": "der", "Flasche": "die", "Becher": "der", "Dose": "die",
    "Schale": "die",
}
#: How much larger (by longest extent) an object must be than every rival before "the large X" is
#: fair. A ratio, not millimetres: "large" is a comparison, and 1.4x is a difference a person sees.
_SIZE_RATIO = 1.4
#: Pixels an anchor must be separated by before "left of" is defensible. Two objects 5 px apart are
#: not left and right of each other, they are next to each other.
_SPATIAL_SEPARATION_PX = 40.0
#: Verbs for the instruction class: several, mixed, so the class measures robustness to the verb
#: as well as to the sentence form. German separable verbs carry their particle as the third
#: element ("Nimm ... auf"), which is why this is a triple and not a string.
_INSTRUCTION_VERBS: tuple[tuple[str, str, str], ...] = (
    ("Pick up", "Nimm", "auf"),
    ("Grasp", "Greife", ""),
    ("Get", "Hol", ""),
)
#: The verb is chosen from the scene id and the target, not from a live generator. "Randomly mixed"
#: still has to mean the same thing on the second run: this whole package is reproducible from a seed,
#: and a prompt file that differs between two runs of the same dataset is not a dataset.
_VERB_STREAM = 20260813


@dataclass(frozen=True, slots=True)
class ObjectFacts:
    """Everything about one object that an expression may be built from."""

    index: int
    kind: str
    colour: ColourName | None
    longest_mm: float
    position_mm: tuple[float, float, float]

    @property
    def words(self) -> tuple[str, str]:
        return KIND_WORDS.get(self.kind, (self.kind, self.kind))


@dataclass(frozen=True, slots=True)
class Prompt:
    """One referring expression, with the evidence that it is unique."""

    scene_id: str
    target: int
    language: str
    text: str
    prompt_class: str
    #: ``None`` for attribute prompts (true from every camera); the view name for spatial ones.
    view: str | None
    #: What made it unique, kept so a dataset can be re-audited without re-deriving the reasoning.
    basis: str

    def as_row(self) -> dict:
        return {
            "scene_id": self.scene_id, "target": self.target, "language": self.language,
            "text": self.text, "class": self.prompt_class, "view": self.view,
            "basis": self.basis, "source": "template",
        }


def scene_facts(payload: dict) -> list[ObjectFacts]:
    """The facts an expression may draw on, for the objects still in the scene.

    Dropped objects are excluded here rather than filtered later: an expression naming an object
    that left the table would be unanswerable, and the exclusion belongs where the facts are
    assembled.
    """
    dropped = {int(index) for index in payload.get("dropped_objects", [])}
    facts: list[ObjectFacts] = []
    for index, obj in enumerate(payload["spec"]["objects"]):
        if index in dropped:
            continue
        settled = payload.get("settled_poses_mm_xyzw", {}).get(str(index))
        position = tuple(settled[0]) if settled else tuple(obj["position_mm"])
        facts.append(ObjectFacts(
            index=index,
            kind=obj["asset_id"].rsplit("_", 1)[-1],
            colour=name_colour(tuple(obj["color_rgb"])),
            longest_mm=0.0,
            position_mm=(float(position[0]), float(position[1]), float(position[2])),
        ))
    return facts


#: Nominative article -> dative. ``von`` governs the dative, so every spatial anchor needs it:
#: "links von die rote Box" is not German, and a grounding benchmark written in broken German
#: measures a model's tolerance for broken German, not its grounding.
_GERMAN_DATIVE: dict[str, str] = {"der": "dem", "die": "der", "das": "dem"}
#: ...and the accusative, which the imperative needs: "Nimm DEN blauen Becher auf".
_GERMAN_ACCUSATIVE: dict[str, str] = {"der": "den", "die": "die", "das": "das"}
#: Weak adjective endings after a definite article. Nominative and dative are gender-independent
#: (-e and -en); the accusative is not: masculine takes -en while feminine and neuter keep -e.
#: That is why this table is keyed by the nominative article, the only gender marker available
#: here: "den rotEN Becher" but "die rotE Dose".
_ADJECTIVE_ENDING: dict[str, dict[str, str]] = {
    "nom": {"der": "e", "die": "e", "das": "e"},
    "dat": {"der": "en", "die": "en", "das": "en"},
    "acc": {"der": "en", "die": "e", "das": "e"},
}


def _inflect(adjective: str, case: str, article: str) -> str:
    """A German colour or size adjective in the weak declension, for this case and gender."""
    stem = adjective[:-1] if adjective.endswith("e") else adjective
    return stem + _ADJECTIVE_ENDING[case][article]


def _phrase(
    fact: ObjectFacts, language: str, *, with_colour: bool, size: str = "", case: str = "nom",
) -> str:
    english, german = fact.words
    if language == "en":
        parts = ["the"]
        if size:
            parts.append(size)
        if with_colour and fact.colour is not None:
            parts.append(fact.colour.en)
        parts.append(english)
        return " ".join(parts)
    article = _GERMAN_ARTICLE.get(german, "die")
    declined = {"nom": article, "dat": _GERMAN_DATIVE[article],
                "acc": _GERMAN_ACCUSATIVE[article]}[case]
    parts = [declined]
    if size:
        parts.append(_inflect({"large": "große", "small": "kleine"}[size], case, article))
    if with_colour and fact.colour is not None:
        parts.append(_inflect(fact.colour.de, case, article))
    parts.append(german)
    return " ".join(parts)


def attribute_prompts(scene_id: str, facts: Sequence[ObjectFacts]) -> list[Prompt]:
    """One expression per object that colour/shape/size can single out. Uniqueness is checked.

    Tried shortest first (bare shape, then colour+shape, then size+shape), because the shortest
    true description is the one a person would actually say, and a benchmark built from
    over-specified prompts measures something easier than the task.
    """
    prompts: list[Prompt] = []
    kinds = [fact.kind for fact in facts]
    colours = [fact.colour.en if fact.colour else None for fact in facts]
    longest = [fact.longest_mm for fact in facts]

    for position, fact in enumerate(facts):
        basis = ""
        with_colour = False
        size = ""
        if kinds.count(fact.kind) == 1:
            basis = "shape"
        elif fact.colour is not None and sum(
            1 for other, kind in zip(colours, kinds)
            if other == fact.colour.en and kind == fact.kind
        ) == 1:
            basis, with_colour = "colour+shape", True
        elif fact.colour is not None and colours.count(fact.colour.en) == 1:
            basis, with_colour = "colour", True
        else:
            mine = longest[position]
            # Rivals of the same kind and a different size. Same-size rivals are excluded here
            # rather than compared against: "the large can" cannot separate two cans of equal
            # size, and an empty rival list must mean no expression to offer rather than a
            # ValueError out of max().
            rivals = [
                size_mm for other, size_mm in zip(kinds, longest)
                if other == fact.kind and size_mm != mine
            ]
            if mine and rivals and mine >= _SIZE_RATIO * max(rivals):
                basis, size = "size+shape", "large"
            elif mine and rivals and mine * _SIZE_RATIO <= min(rivals):
                basis, size = "size+shape", "small"
        if not basis:
            continue
        for language in ("en", "de"):
            prompts.append(Prompt(
                scene_id=scene_id, target=fact.index, language=language,
                text=_phrase(fact, language, with_colour=with_colour, size=size),
                prompt_class="attribute", view=None, basis=basis,
            ))
    return prompts


def instruction_prompts(prompts: Sequence[Prompt], facts: Sequence[ObjectFacts]) -> list[Prompt]:
    """Wrap referring expressions as spoken instructions: a separate class, not a replacement.

    A noun phrase is what a detector is built to take, an instruction is what an operator actually
    says, and the two are different inputs. Keeping them as separate classes means the difference
    is measurable instead of being a confound baked into one prompt set.

    Built from the template expressions rather than generated, so it inherits their uniqueness
    unchanged: the wrapper adds a verb and never touches what identifies the object.
    """
    by_index = {fact.index: fact for fact in facts}
    out: list[Prompt] = []
    for prompt in prompts:
        fact = by_index.get(prompt.target)
        if fact is None or prompt.prompt_class not in ("attribute", "spatial"):
            continue
        stream = np.random.default_rng([_VERB_STREAM, abs(hash(prompt.scene_id)) % 2**31,
                                        prompt.target])
        english, german, particle = _INSTRUCTION_VERBS[int(stream.integers(len(_INSTRUCTION_VERBS)))]
        if prompt.language == "en":
            text = f"{english} {prompt.text}"
        else:
            # German needs the accusative, which the stored nominative text does not carry, so the
            # phrase is rebuilt rather than string-patched. Rebuilding is also what keeps a spatial
            # prompt's anchor in the dative where it belongs.
            accusative = _rephrase_accusative(prompt, fact, facts)
            text = f"{german} {accusative} {particle}".strip()
        out.append(Prompt(
            scene_id=prompt.scene_id, target=prompt.target, language=prompt.language,
            text=text, prompt_class="instruction", view=prompt.view,
            basis=f"{prompt.basis}+verb",
        ))
    return out


def _rephrase_accusative(
    prompt: Prompt, fact: ObjectFacts, facts: Sequence[ObjectFacts],
) -> str:
    """The German expression again, with the target in the accusative and any anchor still dative."""
    with_colour = fact.colour is not None and fact.colour.de in prompt.text
    size = "large" if "große" in prompt.text else ("small" if "kleine" in prompt.text else "")
    target = _phrase(fact, "de", with_colour=with_colour, size=size, case="acc")
    if prompt.prompt_class == "attribute":
        return target
    side, _, anchor_index = prompt.basis.partition("-of-")
    anchor = next((f for f in facts if f.index == int(anchor_index)), None)
    if anchor is None:
        return target
    anchor_phrase = _phrase(anchor, "de", with_colour=anchor.colour is not None, case="dat")
    relation = "links von" if side == "left" else "rechts von"
    return f"{target} {relation} {anchor_phrase}"


def spatial_prompts(
    scene_id: str, facts: Sequence[ObjectFacts], view_name: str, pixels: np.ndarray,
) -> list[Prompt]:
    """Expressions for the objects attributes cannot reach, anchored to one that they can.

    ``pixels`` is the projection of ``facts`` into this view, in the same order. The relation is
    image-space and so is the expression's validity: a prompt from here names its view and means
    nothing without it.
    """
    prompts: list[Prompt] = []
    kinds = [fact.kind for fact in facts]
    colours = [fact.colour.en if fact.colour else None for fact in facts]

    def uniquely_named(position: int) -> bool:
        fact = facts[position]
        if kinds.count(fact.kind) == 1:
            return True
        return fact.colour is not None and sum(
            1 for other, kind in zip(colours, kinds)
            if other == fact.colour.en and kind == fact.kind
        ) == 1

    for position, fact in enumerate(facts):
        if uniquely_named(position) or not np.all(np.isfinite(pixels[position])):
            continue
        for anchor_position, anchor in enumerate(facts):
            if anchor_position == position or not uniquely_named(anchor_position):
                continue
            if not np.all(np.isfinite(pixels[anchor_position])):
                continue
            delta = float(pixels[position][0] - pixels[anchor_position][0])
            if abs(delta) < _SPATIAL_SEPARATION_PX:
                continue
            side = "right" if delta > 0 else "left"
            # The check that makes it ground truth: exactly one object of this kind may lie on that
            # side of this anchor. Without it, "the can left of the carton" is a guess whenever two
            # cans do.
            matching = [
                other for other, other_fact in enumerate(facts)
                if other_fact.kind == fact.kind and np.all(np.isfinite(pixels[other]))
                and (pixels[other][0] - pixels[anchor_position][0] > _SPATIAL_SEPARATION_PX
                     if side == "right"
                     else pixels[other][0] - pixels[anchor_position][0] < -_SPATIAL_SEPARATION_PX)
            ]
            if matching != [position]:
                continue
            anchor_en = _phrase(anchor, "en", with_colour=anchor.colour is not None)
            # Dative: "von" governs it, and the anchor is its object.
            anchor_de = _phrase(anchor, "de", with_colour=anchor.colour is not None, case="dat")
            target_en, target_de = fact.words
            word = {"left": ("left of", "links von"), "right": ("right of", "rechts von")}[side]
            article = _GERMAN_ARTICLE.get(target_de, "die")
            prompts.append(Prompt(
                scene_id=scene_id, target=fact.index, language="en",
                text=f"the {target_en} {word[0]} {anchor_en}",
                prompt_class="spatial", view=view_name, basis=f"{side}-of-{anchor.index}",
            ))
            prompts.append(Prompt(
                scene_id=scene_id, target=fact.index, language="de",
                text=f"{article} {target_de} {word[1]} {anchor_de}",
                prompt_class="spatial", view=view_name, basis=f"{side}-of-{anchor.index}",
            ))
            break
    return prompts
