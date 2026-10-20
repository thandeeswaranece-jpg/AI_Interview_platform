"""
services/speech_service.py

Local speech-to-text using faster-whisper. The model is loaded once
(lazily) and reused for every session - reloading it per answer would
be far too slow on CPU.
"""

import asyncio
import logging
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger("interview.speech")

_model: WhisperModel | None = None
_model_lock = asyncio.Lock()

WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"


class WhisperUnavailableError(Exception):
    pass


def _load_model() -> WhisperModel:
    global _model
    if _model is None:
        logger.info(f"Loading faster-whisper '{WHISPER_MODEL_SIZE}' ({WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE})...")
        _model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        logger.info("faster-whisper model loaded.")
    return _model


async def ensure_model_loaded():
    """Call once at FastAPI startup so the candidate's first answer isn't
    slowed down by a multi-second model load happening mid-interview."""
    async with _model_lock:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _load_model)


def pcm16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return audio_i16.astype(np.float32) / 32768.0


def _transcribe_blocking(audio_f32: np.ndarray) -> str:
    try:
        model = _load_model()
    except Exception as e:
        raise WhisperUnavailableError(str(e))

    segments, _info = model.transcribe(
        audio_f32,
        language="en",
        vad_filter=False,  # we already did our own endpointing in vad_service
        beam_size=1,        # fastest practical setting for real-time CPU use
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


async def transcribe_pcm16(pcm_bytes: bytes) -> str:
    if len(pcm_bytes) < 320:  # under 10ms of audio - nothing usable
        return ""
    audio_f32 = pcm16_bytes_to_float32(pcm_bytes)
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _transcribe_blocking, audio_f32)
    except WhisperUnavailableError:
        logger.error("faster-whisper could not be loaded/run.")
        raise