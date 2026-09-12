"""Whisper on transformers: the main engine behind `SpeechEngine`, built once and loaded on first use.

This is the main engine, running openai/whisper-large-v3-turbo on the torch this interpreter already
has. faster-whisper on CTranslate2 stays a later bake-off, because Smart App Control refuses
CTranslate2's DLL on this workstation and its wheel carries no kernels for the RTX 5080.

What it does on purpose:

* Transcribe, never translate. `generate` is asked through `task` and `language`, the arguments
  transformers 5.5.4 supports. `forced_decoder_ids` is deprecated there and ignored once either is
  passed (`generation_whisper.py:1491-1505`), and the checkpoint's own forced tokens are cleared at load.
* German or English, from one encoder pass. Under `auto` the encoder runs once, and one decoder step
  from the start token scores the language tokens, the step transformers' own `detect_language` runs
  (`generation_whisper.py:1610-1673`). Only `<|de|>` and `<|en|>` are compared, and the winner goes to
  `generate` together with the same encoder output, so detection costs one decoder step and no second
  encoder pass. Left to itself, `generate` would detect among every language the checkpoint knows.
* Read the language back from the decoder prompt. With `return_dict_in_generate=True` a recording of
  up to 30 s comes back with the prompt at the front of the sequence (`generation_whisper.py:913-934`),
  so the report says the language token the decoder actually ran with.
* One window. A recording longer than 30 s is refused with `RecordingTooLong`, rather than cut into
  windows whose words and languages this engine would not stitch together.
* Nothing heavy at import. torch, tokenizers, safetensors and transformers are imported in `load()`.
  A machine that cannot import them gets `SpeechStackUnavailable`, which names either the requirements
  file that installs the package or the DLL Windows refused.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src.config.schema.models.models_schema import (
    InferenceOptimization,
    SpeechLanguage,
    SpeechToTextConfig,
)
from src.contracts import UNSET, Maybe, chosen, resolve
from src.models.constants import MODELS_LOG_DIR, WHISPER_LOG_FILE
from src.models.speech.capture import to_mono_at_rate
from src.models.speech.engine import (
    WHISPER_WINDOW_S,
    RecordingTooLong,
    SpeechModelMissing,
    SpeechStackUnavailable,
)
from src.models.speech.transcript import LanguageSource, Transcript
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover, typing only
    import numpy as np

__all__ = ["WhisperTransformersEngine"]

#: The only task this stack runs (`models.stt.task`).
_TASK: Final[str] = "transcribe"
#: How far into a returned sequence the decoder prompt reaches: start, language, task, no timestamps.
_PROMPT_TOKENS: Final[int] = 4
#: The languages `auto` chooses between, as Whisper language codes. A tie goes to the first.
_AUTO_LANGUAGES: Final[tuple[str, str]] = ("de", "en")
#: What `from_parts` uses for an argument nobody chose.
_DEFAULT_LANGUAGE: Final[SpeechLanguage] = "auto"
_DEFAULT_SAMPLERATE: Final[int] = 16000


def _import_stack() -> tuple[Any, Any]:
    """torch and transformers, with the two compiled extensions transformers loads lazily; or the refusal.

    transformers imports tokenizers and safetensors the first time a processor or a checkpoint is read.
    Imported here, a missing package or a refused DLL in either is named as that package at `load()`.
    Left to transformers, either one surfaces from inside `from_pretrained` as an error about the model,
    `OSError: Can't load feature extractor for 'org/whisper'`, which names the wrong thing.
    """
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise SpeechStackUnavailable(package="torch", cause=exc) from exc
    try:
        import tokenizers  # noqa: F401
    except (ImportError, OSError) as exc:
        raise SpeechStackUnavailable(package="tokenizers", cause=exc) from exc
    try:
        import safetensors  # noqa: F401
    except (ImportError, OSError) as exc:
        raise SpeechStackUnavailable(package="safetensors", cause=exc) from exc
    try:
        import transformers
    except (ImportError, OSError) as exc:
        raise SpeechStackUnavailable(package="transformers", cause=exc) from exc
    return torch, transformers


class WhisperTransformersEngine:
    """Whisper through transformers, behind `SpeechEngine`. Build it with `from_config` or `from_parts`."""

    def __init__(
        self,
        *,
        source: str,
        language: SpeechLanguage,
        samplerate: int,
        local_files_only: bool,
        optim: InferenceOptimization | None,
        device: Maybe[str],
        processor: Maybe[Any],
        model: Maybe[Any],
    ) -> None:
        self._source = source
        self._language: SpeechLanguage = language
        self._samplerate = samplerate
        self._local_files_only = local_files_only
        self._optim = optim
        self._device_choice = device
        self._processor: Any = processor if chosen(processor) else None
        self._model: Any = model if chosen(model) else None
        self._device: Any = None
        self._model_dtype: Any = None
        self._languages: dict[int, str] = {}
        #: The `<|de|>` and `<|en|>` token ids, in the order of `_AUTO_LANGUAGES`.
        self._pair: tuple[int, ...] = ()
        self._start_token = 0
        self._ready = False
        self._lock = threading.RLock()
        self._logger = create_logger("Whisper", log_file=WHISPER_LOG_FILE, log_dir=MODELS_LOG_DIR)

    @classmethod
    def from_config(
        cls, *, config: SpeechToTextConfig, device: Maybe[str] = UNSET
    ) -> WhisperTransformersEngine:
        """The YAML door: `models.stt` resolved into the arguments `from_parts` takes. Loads nothing."""
        return cls.from_parts(
            source=config.model_path if config.local else config.model_id,
            language=config.language,
            samplerate=config.samplerate,
            local_files_only=config.local,
            optim=config.optim,
            device=device,
        )

    @classmethod
    def from_parts(
        cls,
        *,
        source: str,
        language: Maybe[SpeechLanguage] = UNSET,
        samplerate: Maybe[int] = UNSET,
        local_files_only: Maybe[bool] = UNSET,
        optim: Maybe[InferenceOptimization | None] = UNSET,
        device: Maybe[str] = UNSET,
        processor: Maybe[Any] = UNSET,
        model: Maybe[Any] = UNSET,
    ) -> WhisperTransformersEngine:
        """The Python door. Loads nothing.

        ``source`` is a local directory or a Hub id. ``processor`` and ``model`` hand in parts that are
        already built, together or not at all, and ``source`` then only names them in the report. Left
        UNSET: ``language`` is auto, ``samplerate`` 16000, ``local_files_only`` True (a local directory
        that is not there is refused rather than read as a Hub id), ``optim`` none (transformers' own
        load defaults), and ``device`` resolves through `get_device` at load: ``WILLY_DEVICE``, then
        cuda, mps, cpu.
        """
        if chosen(processor) != chosen(model):
            raise ValueError(
                f"from_parts takes processor and model together or neither: one alone would load the "
                f"other from '{source}' and pair two parts nobody chose together."
            )
        return cls(
            source=source,
            language=resolve("language", language, _DEFAULT_LANGUAGE),
            samplerate=resolve("samplerate", samplerate, _DEFAULT_SAMPLERATE),
            local_files_only=resolve("local_files_only", local_files_only, True),
            optim=resolve("optim", optim, None),
            device=device,
            processor=processor,
            model=model,
        )

    @property
    def name(self) -> str:
        """The engine's name as a `Transcript` reports it."""
        return "whisper-transformers"

    def load(self) -> None:
        """Load the weights now, or adopt the parts handed in, and prepare the decoder. Loads once."""
        with self._lock:
            if self._ready:
                return
            torch, transformers = _import_stack()
            from src.utility.device import get_device

            device = torch.device(self._device_choice) if chosen(self._device_choice) else get_device()
            started = time.perf_counter()
            if self._model is None:
                self._refuse_a_missing_directory()
                self._load_weights(transformers, device)
            self._prepare_decoder()
            self._device = device
            self._ready = True
            self._logger.info(
                "Whisper ready from '%s' on %s in %.1f s (language %s).",
                self._source, device, time.perf_counter() - started, self._language,
            )
            if device.type == "cpu":
                self._logger.warning("Whisper runs on the CPU; a transcription will be slow.")

    def _refuse_a_missing_directory(self) -> None:
        """A local directory that is not there, refused in these words.

        Left to `from_pretrained`, a missing directory is read as a Hub repo id and refused as one, which
        sends an operator looking for a Hub problem instead of a fetch that never ran.
        """
        if self._local_files_only and not Path(self._source).is_dir():
            raise SpeechModelMissing(
                key="models.stt.model_path", path=self._source, kind="directory",
                fetch=(
                    "fetch the checkpoint with `python scripts/model_weights/fetch.py` (`--list` names "
                    "each entry), or set models.stt.local: false to download models.stt.model_id from "
                    "the Hub on first use"
                ),
            )

    def _load_weights(self, transformers: Any, device: Any) -> None:
        from src.models._inference import build_load_kwargs, finalize_model, weight_load_errors

        base = {"local_files_only": True} if self._local_files_only else None
        model_kwargs = build_load_kwargs(self._optim, device, base)
        self._model_dtype = model_kwargs.get("torch_dtype")
        self._logger.info(
            "Loading Whisper from '%s' on %s (dtype %s).", self._source, device, self._model_dtype
        )
        try:
            processor = transformers.WhisperProcessor.from_pretrained(self._source, **(base or {}))
            model = transformers.WhisperForConditionalGeneration.from_pretrained(
                self._source, **model_kwargs
            ).to(device)
        # The three ways weights are unusable, plus a CUDA out-of-memory on the trailing `.to(device)`.
        # A defect in this repository's own code is not among them and passes unlabelled.
        except weight_load_errors():
            self._logger.error("Failed to load Whisper model '%s'", self._source)
            raise
        self._processor = processor
        self._model = finalize_model(model, device, self._optim)

    def _prepare_decoder(self) -> None:
        generation: Any = getattr(self._model, "generation_config", None)
        table = getattr(generation, "lang_to_id", None)
        if not table:
            raise ValueError(
                f"the Whisper checkpoint '{self._source}' carries no language table "
                f"(generation_config.lang_to_id), so it can neither detect nor keep a spoken language. "
                f"An English-only checkpoint cannot serve this stack; use a multilingual one."
            )
        missing = [f"<|{code}|>" for code in _AUTO_LANGUAGES if f"<|{code}|>" not in table]
        if missing:
            raise ValueError(
                f"the Whisper checkpoint '{self._source}' has no {' and no '.join(missing)} in its language "
                f"table (generation_config.lang_to_id), and this stack transcribes German and English. "
                f"Use a multilingual checkpoint that carries both."
            )
        # The checkpoint's own forced prompt would otherwise decide what `auto` leaves open.
        generation.forced_decoder_ids = None
        if self._language == "auto":
            generation.language = None
        self._languages = {int(token_id): str(token).strip("<|>") for token, token_id in table.items()}
        self._pair = tuple(int(table[f"<|{code}|>"]) for code in _AUTO_LANGUAGES)
        self._start_token = int(generation.decoder_start_token_id)

    def transcribe(self, samples: np.ndarray, *, samplerate: int) -> Transcript:
        """The most likely text for one recording of at most 30 s. Loads the weights first when nothing
        has yet."""
        import numpy as np

        if samplerate <= 0:
            raise ValueError(f"a recording needs a positive sample rate, not {samplerate} Hz.")
        array = np.asarray(samples)
        frames = int(array.shape[0]) if array.ndim else 0
        if frames == 0:
            raise ValueError("there is nothing to transcribe: the recording holds zero frames.")
        duration_s = frames / float(samplerate)
        if duration_s > WHISPER_WINDOW_S:
            raise RecordingTooLong(duration_s=duration_s)
        self.load()
        torch, _ = _import_stack()
        from src.models._inference import autocast_ctx

        with self._lock:
            started = time.perf_counter()
            audio = self._mono_at_model_rate(array, samplerate)
            inputs = self._processor(audio, sampling_rate=self._samplerate, return_tensors="pt")
            features = inputs.input_features.to(self._device)
            if self._model_dtype is not None and self._device.type == "cuda":
                features = features.to(self._model_dtype)
            with torch.inference_mode(), autocast_ctx(self._device, dtype=self._model_dtype):
                encoded = self._model.get_encoder()(features)
                language = self._pick_language(torch, encoded) if self._language == "auto" else self._language
                output = self._model.generate(
                    encoder_outputs=encoded,
                    task=_TASK,
                    language=language,
                    return_dict_in_generate=True,
                )
            sequences = output.sequences
            text = str(self._processor.batch_decode(sequences, skip_special_tokens=True)[0]).strip()
            reported = self._language_in(sequences[0])
            latency_ms = (time.perf_counter() - started) * 1000.0
        return Transcript(
            text=text,
            language=reported,
            language_source=(
                LanguageSource.DETECTED if self._language == "auto" else LanguageSource.CONFIGURED
            ),
            duration_s=duration_s,
            engine=self.name,
            model=self._source,
            device=str(self._device.type),
            latency_ms=latency_ms,
        )

    def _pick_language(self, torch: Any, encoded: Any) -> str:
        """German or English: whichever of the two language tokens the first decoder step scores higher.

        Every other language token is left out of the comparison, so a recording that sounds French is
        still decoded as German or English. The step reads the encoder output `generate` reuses.
        """
        start = torch.full((1, 1), self._start_token, dtype=torch.long, device=self._device)
        logits = self._model(encoder_outputs=encoded, decoder_input_ids=start, use_cache=False).logits[0, -1]
        scores = logits[list(self._pair)].float()
        return _AUTO_LANGUAGES[int(scores.argmax())]

    def _mono_at_model_rate(self, samples: np.ndarray, samplerate: int) -> np.ndarray:
        """Mono float32 at the model's rate, scaled down when its peak lies beyond [-1, 1]."""
        import numpy as np

        audio = to_mono_at_rate(samples, samplerate=samplerate, target=self._samplerate)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        return audio

    def _language_in(self, sequence: Any) -> str | None:
        """The language code in the decoder prompt at the front of ``sequence``, or ``None``."""
        for token in sequence[:_PROMPT_TOKENS].tolist():
            code = self._languages.get(int(token))
            if code is not None:
                return code
        return None
