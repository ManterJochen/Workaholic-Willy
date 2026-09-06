# Prompt routing

Decide which perception stack a prompt needs, from the prompt alone, before any weights load.

```python
from src.models.routing import route

decision = route("greif den kaputten Wuerfel")
decision.route        # Route.VLM
decision.reason       # RouteReason.NON_ENGLISH
decision.describe()   # 'vlm (non_english)'
decision.to_dict()    # JSON-safe, for the perception event and GraspAttemptRecord.extra
```

## What this package guarantees

No model, no image, no network. `route(prompt)` is a pure function of a string: the same prompt
always yields the same route, the reason that decided it, and the signals it was read off. The
decision travels with the pick, so an operator seeing a slow one can find out that their wording
chose the expensive route, and which word did it.

## Why two routes, and why not a cascade

The simple route (GroundingDINO then SAM2) is fast and reliable on short attributive noun phrases.
On anything else it fails confidently: it returns a high-scoring box for the wrong object rather
than admitting defeat.

That is why "try the cheap one first and fall back if the result looks bad" is not what this does.
A cascade needs the cheap stage to fail loudly, and there is no reliable signal to fall back on;
the failure mode is a robot grasping the wrong thing while the record says it succeeded. So the
choice is made from the prompt, deterministically, before either model runs.

## The rules

The first matching property wins, most specific first, and its name becomes the reason:

| Reason | Fires on | Why the phrase grounder cannot do it |
| --- | --- | --- |
| `non_english` | German vocabulary, or any non-ASCII letter | the text encoder is English-trained |
| `relative_clause` | "that", "which", "welche" | reference, not description |
| `negation` | "not", "without", "nicht", "ohne" | there is no NOT in a caption |
| `state_word` | "broken", "empty", "kaputt" | a judgement about condition |
| `comparative` | "largest", "leftmost", "groesste" | requires comparing candidates |
| `spatial_relation` | "behind", "next to", "hinter" | relates two objects |
| `quantifier` | "every", "all", "alle" | scope over a set |
| `conjunction` | "and", "or", "und" | two targets in one phrase |
| `multi_attribute` | `MULTI_ATTRIBUTE_THRESHOLD` attributes or more, currently 3 | binding, not description, becomes the hard part |
| `too_long` | more than `MAX_SIMPLE_WORDS` words, currently 6 | stopped being a noun phrase |
| `plain_noun_phrase` | none of the above | the simple route |

An empty prompt routes simple, with the reason `empty_prompt`. Routing is not validation: sending a
caller's bug to the expensive route would hide it.

German umlauts and eszett are folded to ASCII digraphs before matching, so a prompt typed with
"Wuerfel" and one typed with the umlaut take the same route.

## Where it runs

Only inside the `models.pipeline` block, and only when `zero_shot.backend` is `vlm`: routing with
nowhere better to send a hard prompt is a decision nothing can act on, and the schema refuses that
combination by name. In [`config/models/object.yaml`](../../../config/models/object.yaml) the
router ships `enabled: false`, so the shipped stack grounds every prompt with GroundingDINO. Note
that the schema field defaults to `true` while that file sets `false`; the file is what a build
reads.

`GET /v1/diagnostics/route?prompt=...` reports which route a prompt would take, loading nothing.

## Traps

**The thresholds are judgement, not measurement.** Where GroundingDINO's grounding accuracy
actually falls off on a given cell's scenes is unmeasured. `MAX_SIMPLE_WORDS` and
`MULTI_ATTRIBUTE_THRESHOLD` are exported so they can be argued with in one place.

**Language detection covers English and German only.** Any other language is caught only if it
carries non-ASCII letters. An accentless prompt in a third language routes simple and grounds
badly. This is a known limitation, not a discovery waiting to happen.

**A false `non_english` is the expensive mistake**, since it doubles the cost of every English
pick, so the German vocabulary holds only tokens with no English reading at all: "die", "in", "an",
"so", "war", "hat" and "leg" are all deliberately absent.

**`models.pipeline.router.route_non_english_to_vlm` reaches nothing.** The rule-based router takes
no configuration, and `build_perception` constructs `RoutedPerceptionBackend` without one, so the
non-English rule is unconditional whatever that key says.

## The seam is meant to be used

`PromptRouter` is a Protocol because the rules are a starting point. Two constraints for whoever
replaces them with a learned judge:

- **Small, not generative.** This runs on every pick to decide whether to load a multi-billion
  parameter model. A generative model in that position is the latency it was meant to avoid. Use a
  distilled encoder or a linear model over embeddings, at millisecond scale.
- **Uncertainty must fall to `VLM`.** The costs are asymmetric: a wrong `SIMPLE` grasps the wrong
  object, a wrong `VLM` costs seconds. Do not let a symmetric training loss decide that threshold.

Labels for such a judge should come from measured grounding accuracy, not from anyone's opinion of
which prompts look hard.

## Contents

| File | Role |
| --- | --- |
| [`decision.py`](decision.py) | `Route`, `RouteReason`, `PromptSignals`, `RouteDecision`, and the `PromptRouter` seam. Data, not behaviour. |
| [`rules.py`](rules.py) | The vocabularies, `analyse()`, the thresholds, and the default `RuleBasedRouter`. |
| [`normalize.py`](normalize.py) | `normalize_simple_prompt()`: lowercase, collapse whitespace, one trailing period. GroundingDINO's caption convention, applied to the simple route only, since the VLM is asked to reason about the operator's own phrasing. Empty input returns `""` rather than a bare period. |

## See also

- [`src/models/vlm/`](../vlm/README.md), the expensive route
- [`src/models/`](../README.md), the perception model layer and its factory
- [`config/models/object.yaml`](../../../config/models/object.yaml), the `models.pipeline` block
  that switches this on
