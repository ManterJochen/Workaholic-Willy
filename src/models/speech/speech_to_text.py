import time

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config.schema.models.models_schema import SpeechToTextConfig
from src.models._inference import (
    autocast_ctx,
    build_load_kwargs,
    finalize_model,
    weight_load_errors,
)
from src.models.constants import MODELS_LOG_DIR, WHISPER_LOG_FILE
from src.models.speech._poll import poll_until_text
from src.utility.device import get_device, move_inputs_to_device
from src.utility.log_cfg import create_logger


class WhisperSpeechToText:

    def __init__(self, config: SpeechToTextConfig):
        self.device = get_device()
        self.samplerate = config.samplerate
        self.block_size = config.blocksize
        self.channels = config.channels
        self.dtype = config.dtype
        self.chunk_duration = config.chunk_duration
        self.language = config.language
        self.task = config.task
        self.optim = config.optim

        self.audio_buffer: list[float] = []
        self.samples_per_chunk = int(self.samplerate * self.chunk_duration)

        self.logger = create_logger("Whisper", log_file=WHISPER_LOG_FILE, log_dir=MODELS_LOG_DIR)

        source = config.model_path if config.local else config.model_id
        proc_kwargs = {"local_files_only": True} if config.local else {}
        model_kwargs = build_load_kwargs(
            self.optim,
            self.device,
            {"local_files_only": True} if config.local else None,
        )
        self._model_dtype = model_kwargs.get("torch_dtype")

        self.logger.info(
            "Initializing Whisper: source='%s', device=%s, dtype=%s",
            source, self.device, self._model_dtype,
        )

        try:
            self.processor = WhisperProcessor.from_pretrained(source, **proc_kwargs)  # type: ignore[arg-type]  # transformers stub mistypes from_pretrained **kwargs
            self.model = WhisperForConditionalGeneration.from_pretrained(
                source, **model_kwargs
            ).to(self.device)  # type: ignore[arg-type]  # _Wrapped decorator mistypes the chained .to()
        # RuntimeError caught none of the three ways weights are actually unusable
        # (ValueError, OSError, safetensors.SafetensorError; measured 2026-09-10). See
        # weight_load_errors().
        except weight_load_errors():
            self.logger.error("Failed to load Whisper model '%s'", source)
            raise

        self.model = finalize_model(self.model, self.device, self.optim)

        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language, task=self.task
        )

        if self.device.type == "cpu":
            self.logger.warning("Whisper loaded on CPU. Inference may be slow.")
        else:
            self.logger.info("Whisper loaded on %s.", self.device.type.upper())

    def _prepare_audio(self, audio: np.ndarray, source_samplerate: int) -> np.ndarray:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32)

        if source_samplerate != self.samplerate:
            audio = resample_poly(audio, self.samplerate, source_samplerate).astype(np.float32)

        max_abs = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
        if max_abs > 1.0:
            audio = audio / max_abs

        return audio

    @torch.inference_mode()
    def transcribe_array(self, audio: np.ndarray, source_samplerate: int) -> str:
        prepared_audio = self._prepare_audio(audio, source_samplerate)
        inputs = self.processor(prepared_audio, sampling_rate=self.samplerate, return_tensors="pt")
        moved = move_inputs_to_device({"input_features": inputs.input_features}, self.device)
        input_features = moved["input_features"]
        if self._model_dtype is not None and self.device.type == "cuda":
            input_features = input_features.to(self._model_dtype)
        with autocast_ctx(self.device, dtype=self._model_dtype):
            predicted_ids = self.model.generate(
                input_features, forced_decoder_ids=self.forced_decoder_ids
            )
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()

    @torch.inference_mode()
    def start_translation(self, audio: np.ndarray) -> str:
        return self.transcribe_array(audio, self.samplerate)

    def _callback(self, indata, frames, time_info, status):
        if status:
            self.logger.warning("Audio stream status: %s", status)

        audio_chunk = indata.flatten().astype(np.float32) / 32768.0
        self.audio_buffer.extend(audio_chunk)

    def _drain_chunk(self) -> str | None:
        """Transcribe one chunk, or ``None`` while the buffer is short of one."""
        if len(self.audio_buffer) < self.samples_per_chunk:
            return None
        audio_np = np.array(self.audio_buffer[:self.samples_per_chunk])
        self.audio_buffer = self.audio_buffer[self.samples_per_chunk:]
        text = self.transcribe_array(audio_np, self.samplerate).lower().strip()
        return text or None

    def listen_once(
        self,
        *,
        timeout_s: float | None = 30.0,
        max_attempts: int | None = None,
    ) -> str | None:
        """Record from the microphone until one non-empty transcription arrives.

        Returns that text, or ``None`` once the bound (``timeout_s`` or
        ``max_attempts``) elapses. The bound is what keeps silence, a mic
        failure or persistently blank output from looping forever.
        """
        self.audio_buffer.clear()
        with sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=self.block_size,
            dtype=self.dtype,
            channels=self.channels,
            callback=self._callback,
        ):
            text = poll_until_text(
                drain=self._drain_chunk,
                sleep=sd.sleep,
                now=time.monotonic,
                timeout_s=timeout_s,
                max_attempts=max_attempts,
            )
        if text:
            self.logger.info("Recognized text: '%s'", text)
        else:
            self.logger.warning(
                "listen_once: no speech recognised within bound "
                "(timeout_s=%s, max_attempts=%s).",
                timeout_s,
                max_attempts,
            )
        return text
