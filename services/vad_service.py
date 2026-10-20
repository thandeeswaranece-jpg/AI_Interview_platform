"""
services/vad_service.py

Frame slicing + WebRTC VAD wrapper + a pause-tolerant endpoint detector.
No WebSocket/FastAPI knowledge here - it only turns a stream of raw
PCM16 mono 16kHz audio into speech events.

NOTE: pip package is 'webrtcvad-wheels' (see requirements.txt) but the
import name is still 'webrtcvad' - that's intentional, not a typo.
"""

import time
import webrtcvad

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = int(SAMPLE_RATE * (FRAME_MS / 1000.0)) * 2  # 16-bit mono -> 640 bytes/frame

SPEECH_CONFIRM_FRAMES = 3        # ~60ms of speech to confirm "speaking started" (avoids noise blips)
TRAILING_SILENCE_MS = 1200       # sustained silence needed to decide the answer is actually over
TRAILING_SILENCE_FRAMES = TRAILING_SILENCE_MS // FRAME_MS
PREBUFFER_FRAMES = 15            # ~300ms kept before speech is confirmed, so the first word isn't clipped
MAX_ANSWER_SECONDS = 120         # safety cap so a VAD glitch can never hang the interview forever


def make_vad(aggressiveness: int = 2):
    return webrtcvad.Vad(aggressiveness)


class FrameSlicer:
    """Buffers arbitrary-sized incoming byte chunks into fixed 20ms VAD frames."""

    def __init__(self):
        self._buf = bytearray()

    def push(self, chunk: bytes) -> list[bytes]:
        self._buf.extend(chunk)
        frames = []
        while len(self._buf) >= FRAME_BYTES:
            frames.append(bytes(self._buf[:FRAME_BYTES]))
            del self._buf[:FRAME_BYTES]
        return frames

    def reset(self):
        self._buf.clear()


class SpeechEndpointer:
    """
    Tracks ONE candidate turn (one answer) and decides:
      - when speech has genuinely started
      - when a pause is just a natural pause (tolerated, not an ending)
      - when the answer is actually finished (sustained trailing silence)

    Call process_frame(is_speech) once per 20ms frame, in order.
    """

    def __init__(self):
        self.has_spoken = False
        self.speech_started_at: float | None = None
        self.answer_completed_at: float | None = None
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._in_pause = False
        self._pause_started_at: float | None = None
        self.pause_durations: list[float] = []
        self.frames_seen = 0

    def process_frame(self, is_speech: bool) -> str:
        """Returns one of: 'no_change', 'speech_started', 'speech_paused', 'speech_resumed', 'answer_complete'."""
        self.frames_seen += 1
        elapsed = self.frames_seen * (FRAME_MS / 1000.0)

        if is_speech:
            self._consecutive_speech += 1
            self._consecutive_silence = 0

            if not self.has_spoken and self._consecutive_speech >= SPEECH_CONFIRM_FRAMES:
                self.has_spoken = True
                self.speech_started_at = time.time()
                return "speech_started"

            if self.has_spoken and self._in_pause:
                self._in_pause = False
                if self._pause_started_at is not None:
                    self.pause_durations.append(time.time() - self._pause_started_at)
                self._pause_started_at = None
                return "speech_resumed"

            return "no_change"

        # silent frame
        self._consecutive_speech = 0
        self._consecutive_silence += 1

        if not self.has_spoken:
            return "no_change"  # still waiting for the candidate to start talking at all

        if not self._in_pause:
            self._in_pause = True
            self._pause_started_at = time.time()
            if self._consecutive_silence == 5:  # small grace period before announcing a pause
                return "speech_paused"

        if (self._consecutive_silence >= TRAILING_SILENCE_FRAMES) or (elapsed >= MAX_ANSWER_SECONDS):
            self.answer_completed_at = time.time()
            return "answer_complete"

        return "no_change"

    def answer_latency_seconds(self, listening_started_at: float | None) -> float | None:
        if self.speech_started_at is None or listening_started_at is None:
            return None
        return max(0.0, self.speech_started_at - listening_started_at)

    def audio_duration_seconds(self) -> float | None:
        if self.speech_started_at is None or self.answer_completed_at is None:
            return None
        return max(0.0, self.answer_completed_at - self.speech_started_at)