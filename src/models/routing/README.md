# Which model answers a prompt (`src/models/routing`)

Decides, from the prompt text alone and before any weights load, whether a prompt goes to the fast phrase
grounder (GroundingDINO then SAM2) or to the vision-language model. No model, no image, no network: the
same prompt always gets the same route, the reason, and the words that decided it.

```python
from willy import RuleBasedRouter

decision = RuleBasedRouter().route("greif den kaputten Wuerfel")
print(decision.describe())    # vlm (non_english)
decision.route                # Route.VLM
decision.reason               # RouteReason.NON_ENGLISH
decision.to_dict()            # JSON-safe: the perception event and the attempt record carry it
```

The decision travels with the pick, so an operator who sees a slow one can find which word chose the
expensive route. The console answers `GET /v1/diagnostics/route?prompt=...` the same way, loading
nothing. [route_hard_prompts.py](../../../examples/offline/perception/route_hard_prompts.py) routes a
list of prompts and shows the stack a config would build for them.

## Why a rule and not a cascade

The phrase grounder is fast and reliable on short noun phrases. On anything else it fails confidently: a
high-scoring box on the wrong object, never an error. "Try the cheap one and fall back if the result
looks bad" needs the cheap stage to fail loudly, and it does not; the failure is a robot grasping the
wrong thing while the record says it succeeded. So the choice is made from the prompt, before either
model runs. [`vlm/`](../vlm/README.md) has the measured case.

## The rules

The first property that matches wins, most specific first, and its name becomes the reason:

| Reason | Fires on | Why the phrase grounder cannot do it |
| --- | --- | --- |
| `non_english` | German vocabulary, or any non-ASCII letter | its text encoder is English-trained |
| `relative_clause` | "that", "which", "welche" | reference, not description |
| `negation` | "not", "without", "nicht", "ohne" | a caption has no NOT |
| `state_word` | "broken", "empty", "kaputt" | a judgement about condition |
| `comparative` | "largest", "leftmost", "groesste" | it needs candidates compared |
| `spatial_relation` | "behind", "next to", "hinter" | it relates two objects |
| `quantifier` | "every", "all", "alle" | scope over a set |
| `conjunction` | "and", "or", "und" | two targets in one phrase |
| `multi_attribute` | `MULTI_ATTRIBUTE_THRESHOLD` attributes or more, 3 today | binding becomes the hard part |
| `too_long` | more than `MAX_SIMPLE_WORDS` words, 6 today | it stopped being a noun phrase |
| `plain_noun_phrase` | none of the above | the simple route |

An empty prompt routes simple with the reason `empty_prompt`: routing is not validation, and sending a
caller's bug to the expensive route would hide it. Umlauts and eszett fold to ASCII digraphs first, so
"Wuerfel" and the umlaut spelling take the same route.

## Where it runs

Only inside `models.pipeline`, and only with `zero_shot.backend: vlm`: routing with nowhere better to
send a hard prompt is a decision nothing can act on, and the schema refuses that combination by name.
[`config/models/object.yaml`](../../../config/models/object.yaml) ships `router.enabled: false`, so the
shipped stack grounds every prompt with GroundingDINO. The schema field defaults to `true`; the file is
what a build reads.

## Traps

**The thresholds are judgement, not measurement.** Where GroundingDINO's accuracy falls off on a given
cell's scenes is unmeasured. `MAX_SIMPLE_WORDS` and `MULTI_ATTRIBUTE_THRESHOLD` are exported so they can
be argued with in one place.

**Only English and German are recognised.** Another language is caught only by its non-ASCII letters,
so an accentless prompt in a third language routes simple and grounds badly.

**A false `non_english` is the costly mistake**, since it slows every English pick. The German vocabulary
holds only words with no English reading: "die", "in", "an", "so", "war", "hat" and "leg" are left out
on purpose.

**`models.pipeline.router.route_non_english_to_vlm` reaches nothing.** The router takes no configuration
and the builder passes it none, so the non-English rule applies whatever that key says.

## Replacing the rules

`PromptRouter` is a Protocol because the rules are a starting point. Whatever replaces them must stay
small, since it runs on every pick to decide whether to load a model of billions of parameters: a
distilled encoder or a linear model over embeddings, at millisecond scale. Uncertainty must fall to
`VLM`, because a wrong `SIMPLE` grasps the wrong object while a wrong `VLM` costs seconds. Label such a
judge from measured grounding accuracy, not from an opinion of which prompts look hard.

## Files

| File | Holds |
| --- | --- |
| [`decision.py`](decision.py) | `Route`, `RouteReason`, `PromptSignals`, `RouteDecision` and the `PromptRouter` seam |
| [`rules.py`](rules.py) | the vocabularies, `analyse()`, the thresholds and `RuleBasedRouter` |
| [`normalize.py`](normalize.py) | `normalize_simple_prompt()`: lower case, one space, one trailing period, for the simple route only |

## Details

- [`../vlm/`](../vlm/README.md), the expensive route; [`../README.md`](../README.md), the perception layer
- Tests: `tests/test_prompt_router.py`, `tests/test_route_visibility.py`, `tests/test_api_perception_route.py`
