# What the command reader reads off a phrase (`src/models/routing`)

Two readings of a phrase, from its words alone: is it English, and does it carry a quantifier. No model, no
image, no network: the same phrase always reads the same way, and a wrong reading names the word that caused it.

```python
from src.models.routing import analyse

analyse("den kaputten Wuerfel").non_english    # True: German words
analyse("all screws").has_quantifier           # True: "all" belongs in a task's scope
analyse("red cube").to_dict()                  # {'non_english': False, 'has_quantifier': False}
```

The command reader ([`../vlm/command.py`](../vlm/command.py)) asks both of every phrase it would hand the
detector: the object, where the parts lie, the one part a sentence singles out, and a place the camera finds. A
phrase that is not English, or that carries a quantifier, is asked again once, saying what was wrong, because the
detector is asked in English and "all", "every" and "both" belong in the task's scope.

## The vocabularies

| Reading | Fires on |
| --- | --- |
| `non_english` | a German word with no English reading ("der", "nicht", "kaputte", "wuerfel", ...), or any letter outside ASCII |
| `has_quantifier` | "all", "every", "each", "both", "any", "several", "many", "few", "none", and the German "alle", "jeder", "beide", "mehrere", ... |

Umlauts and eszett fold to ASCII digraphs first, so "Wuerfel" and the umlaut spelling read the same. The reader
adds the cell's own German words that read as English here (`GERMAN_WORDS` in `command.py`: "schrauben", "mutter").

## Traps

**Only English and German are recognised.** Another language is caught only by its letters outside ASCII, so an
accentless phrase in a third language reads as English.

**A false `non_english` costs a second question**, so the German vocabulary holds only words with no English
reading: "die", "in", "an", "so", "war", "hat" and "leg" are left out on purpose.

## Where this came from

These are the vocabularies of the prompt router, which sent a plain English noun phrase to GroundingDINO and
everything else to the VLM. Both left on 2026-10-17 (the owner's cleanup): the VLM grounds every phrase, and only
the two readings the command reader asks are kept, word for word.

## Files

| File | Holds |
| --- | --- |
| [`rules.py`](rules.py) | the vocabularies, `PromptSignals` and `analyse()` |

## Details

- [`../vlm/`](../vlm/README.md), the command reader and the detector; [`../README.md`](../README.md), the
  perception layer
- Tests: `tests/models/routing/test_a_phrase_reads_as_english_or_not_and_with_a_quantifier_or_not.py`, `tests/models/vlm/test_vlm_command.py`
