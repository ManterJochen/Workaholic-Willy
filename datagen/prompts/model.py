"""Loading the paraphrase model, without an account and without pretending otherwise.

Two product constraints shape every line here:

* No HuggingFace account may ever be needed. Every request goes out with ``token=False`` and the
  implicit-token lookup is disabled. That is deliberately stricter than it needs to be on a
  developer machine: a load that succeeds only because whoever ran it happens to have a token
  would fail at the first customer. If it works here, it works there.
* Weights live in the repo's own tree, under ``assets/models/hf``, gitignored. A clone lands in
  the same layout, which is what makes "it works on my machine" mean something. The parent
  ``assets/models/`` stays tracked because the small committed models live there too.

The model is ``mistralai/Mistral-7B-Instruct-v0.3``: Apache-2.0, first-party and ungated, so an
anonymous download of its config and its weights succeeds.
"""

from __future__ import annotations

import time

from src.utility.log_cfg import create_logger
from src.utility.paths import fence_model_downloads, weights_root

from datagen.constants import DATAGEN_LOG_DIR, PARAPHRASE_MODEL_LOG_FILE

__all__ = ["DOWNLOAD_IGNORE", "MODEL_ID", "WEIGHTS_ROOT", "load_paraphraser", "weights_present"]

logger = create_logger("datagen.prompts.model", PARAPHRASE_MODEL_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
#: The model repository ships two copies of the same weights: ``consolidated.safetensors``
#: (Mistral's own single-file format, 14.5 GB) and ``model-0000N-of-00003.safetensors`` (the
#: sharded transformers format, 14.5 GB). ``transformers`` reads only what
#: ``model.safetensors.index.json`` names, which is the sharded set, so fetching both doubles a
#: 15 GB download for nothing.
DOWNLOAD_IGNORE = ("consolidated*", "*.pth", "*.bin")
#: The one root every downloaded model lives under. Resolved from the project root rather than the
#: working directory: `python -m datagen` may be run from anywhere and the weights must not land in
#: two places. This module used to compute it from its own depth, which is one more anchor to keep
#: correct through a package move; `weights_root` is the single definition.
WEIGHTS_ROOT = weights_root()


def _anonymous_environment() -> None:
    """Point the hub at :data:`WEIGHTS_ROOT` and make an account impossible to use.

    The token variables are removed rather than ignored, so the no-account path is the only one ever
    exercised: a download that depends on who is logged in is not reproducible.
    """
    fence_model_downloads()


def weights_present() -> bool:
    """True when the weights are already on disk, so a caller can warn before a 15 GB download."""
    _anonymous_environment()
    slug = "models--" + MODEL_ID.replace("/", "--")
    return (WEIGHTS_ROOT / "hub" / slug).exists()


def load_paraphraser(*, four_bit: bool = True):  # noqa: ANN201 (the transformers types are heavy)
    """``(model, tokenizer)``, quantised by default. Downloads on first use, into the repo tree.

    4-bit because the paraphrase is a rewording task, not a knowledge one: about 4.5 GB instead of
    about 14.5 GB leaves the GPU free for everything else, and every output is checked for
    uniqueness anyway, so a weaker sentence becomes a rejection rather than a bad label.
    """
    _anonymous_environment()
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    # Said before the load, because when the weights are absent the next thing that happens is a
    # ~15 GB anonymous download into the repo tree and the process looks hung for a quarter of an hour.
    present = weights_present()
    logger.info("loading %s (%s, weights %s under %s)",
                MODEL_ID, "4-bit" if four_bit else "fp16",
                "present" if present else "ABSENT; downloading", WEIGHTS_ROOT)
    started = time.perf_counter()

    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415

    quantisation = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) if four_bit else None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=False,
        quantization_config=quantisation,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    logger.info("%s ready in %.1f s on %s", MODEL_ID, time.perf_counter() - started, model.device)
    return model, tokenizer
