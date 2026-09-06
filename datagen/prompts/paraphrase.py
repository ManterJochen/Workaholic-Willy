"""The English paraphrase class: a model rewording what the templates already proved unique.

English only. German paraphrase is unusable at both 4-bit and fp16 ("der Flansch" -> "der
Kessel"), so German stays template-only, and this is a separate prompt class rather than a
replacement: the dataset exists to compare English against German, and diversity on one side alone
would make that comparison measure the diversity instead of the language.

Retries vary the instruction, never the sampling. Decoding is greedy, so the same prompt yields
the same sentence every time. That keeps the dataset reproducible, and it also means a retry with
an identical prompt would return the identical rejected output, so three differently-worded asks
is the only form a retry can take here.

Every output goes through :mod:`datagen.prompts.check` and a rejection costs nothing: the template
prompt for that object already exists and stands. The rejection rate, broken down by reason, is
recorded rather than hidden; it is the measurement of how much this class is worth.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, PARAPHRASE_LOG_FILE
from datagen.prompts.check import ParaphraseVerdict, check_paraphrase

__all__ = ["ParaphraseRun", "paraphrase_rows"]

logger = create_logger("datagen.prompts.paraphrase", PARAPHRASE_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Shown rather than described: examples pin the output format, which a zero-shot instruction does
#: not, since that answers with several lines and a translation. Each demonstrates the one
#: transformation allowed: reword around the object, keep the noun and the colour.
_SHOTS: tuple[tuple[str, str], ...] = (
    ("the green bottle", "the green glass bottle"),
    ("the yellow carton", "the yellow cardboard carton"),
    ("the small bowl", "the small shallow bowl"),
)
#: Three differently-worded asks. Greedy decoding makes a repeat of the same ask pointless, so the
#: variation has to live here.
_INSTRUCTIONS: tuple[str, ...] = (
    "Reword this noun phrase. Keep the object noun and the colour word exactly as they are. "
    "Answer with the phrase only.",
    "Say the same thing differently, in one short noun phrase. The object noun and the colour must "
    "stay. Answer with the phrase only.",
    "Add one descriptive word to this phrase without changing the object or its colour. "
    "Answer with the phrase only.",
)
_MAX_NEW_TOKENS = 24


@dataclass(frozen=True, slots=True)
class ParaphraseRun:
    """What the pass produced, and what it refused."""

    attempted: int
    accepted: int
    verdicts: dict[str, int]
    texts: dict[int, str]

    def summary(self) -> str:
        rate = self.accepted / self.attempted * 100.0 if self.attempted else 0.0
        rejected = {k: v for k, v in sorted(self.verdicts.items()) if k != "accepted"}
        return (f"{self.accepted}/{self.attempted} accepted ({rate:.1f}%); rejected {rejected}")


def paraphrase_rows(
    rows: Sequence[dict],
    facts_by_scene: Mapping[str, tuple[Sequence[str], Sequence[str]]],
    *,
    model=None,  # noqa: ANN001 (transformers types are heavy; injected so this stays testable)
    tokenizer=None,  # noqa: ANN001
    limit: int | None = None,
) -> ParaphraseRun:
    """Paraphrase the English noun-phrase rows. ``facts_by_scene`` gives the scene's rival words.

    ``model``/``tokenizer`` are injected rather than loaded here so the acceptance logic can run
    without a GPU: what matters is which outputs are kept, not how they were produced.
    """
    if model is None or tokenizer is None:
        from datagen.prompts.model import load_paraphraser  # noqa: PLC0415

        model, tokenizer = load_paraphraser()

    def generate(instruction: str, phrase: str) -> str:
        messages = [{"role": "user", "content": instruction},
                    {"role": "assistant", "content": "Understood."}]
        for source, target in _SHOTS:
            messages += [{"role": "user", "content": source},
                         {"role": "assistant", "content": target}]
        messages.append({"role": "user", "content": phrase})
        encoded = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True,
        ).to(model.device)
        produced = model.generate(**encoded, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
        start = encoded["input_ids"].shape[1]
        return tokenizer.decode(produced[0][start:], skip_special_tokens=True).strip()

    verdicts: dict[str, int] = {}
    texts: dict[int, str] = {}
    attempted = accepted = generated = 0
    started = time.perf_counter()
    for position, row in enumerate(rows):
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        kinds, colours = facts_by_scene.get(row["scene_id"], ((), ()))
        last = ParaphraseVerdict("empty")
        for instruction in _INSTRUCTIONS:
            produced = generate(instruction, row["text"])
            generated += 1
            last = check_paraphrase(
                produced,
                kind_word=row["_kind"], colour=row.get("_colour"), size=row.get("_size"),
                other_kind_words=kinds, other_colours=colours,
            )
            if last.accepted:
                texts[position] = produced
                accepted += 1
                break
        verdicts[last.verdict] = verdicts.get(last.verdict, 0) + 1
    # Once for the run, never per row: every row is a GPU generation and up to three of them. The
    # rejection breakdown is the measurement of what this prompt class is worth, and `generated`
    # against `attempted` is how much of the run went into retries.
    logger.info("paraphrased %d row(s) in %.1f s: %d accepted (%.1f%%), %d generation(s) "
                "including retries; verdicts %s",
                attempted, time.perf_counter() - started, accepted,
                accepted / attempted * 100.0 if attempted else 0.0, generated,
                dict(sorted(verdicts.items())))
    return ParaphraseRun(attempted, accepted, verdicts, texts)
