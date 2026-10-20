"""
services/interview_service.py

Per-session orchestrator: owns the InterviewSession (state machine),
the audio VAD/endpointing pipeline, and the vision extractor, and
calls llm_service / speech_service / analytics_service in the right
order. main.py's WebSocket handler talks ONLY to this class.
"""

import asyncio
import logging

from services.interview_state import InterviewSession, InterviewState
from services import vad_service, speech_service, vision_service, llm_service, analytics_service

logger = logging.getLogger("interview.orchestrator")


class InterviewOrchestrator:
    def __init__(self, session_id: str, resume_text: str, resume_summary: dict):
        self.session = InterviewSession(session_id, resume_text, resume_summary)
        self.slicer = vad_service.FrameSlicer()
        self.vad = vad_service.make_vad(aggressiveness=2)
        self.endpointer: vad_service.SpeechEndpointer | None = None
        self.audio_buffer = bytearray()
        self._prebuffer: list[bytes] = []
        self.face_extractor = vision_service.FaceMetricsExtractor()
        self.vision_enabled = vision_service.is_available()
        self._processing_lock = asyncio.Lock()

    # ---------------------------------------------------------------- start
    async def start(self) -> dict:
        if not self.session.transition(InterviewState.AI_SPEAKING):
            return self._invalid("start_interview")

        q = await llm_service.generate_next_question(
            self.session.resume_summary, history=[], current_difficulty="easy", topics_covered=[]
        )
        record = self.session.set_next_question(q["question"], q["difficulty"], q["topic"])
        return {
            "event": "ai_speak",
            "state": self.session.state,
            "question": record.question,
            "question_number": self.session.current_question_index + 1,
            "difficulty": record.difficulty,
            "topic": record.topic,
            "question_source": q.get("source"),
        }

    # ------------------------------------------------------- turn lifecycle
    def ai_finished_speaking(self) -> dict:
        if not self.session.ai_finished_speaking():
            return self._invalid("ai_done_speaking")
        self.endpointer = vad_service.SpeechEndpointer()
        self.slicer.reset()
        self.audio_buffer = bytearray()
        self._prebuffer = []
        return {"event": "listening", "state": self.session.state}

    async def handle_audio_chunk(self, chunk: bytes) -> list[dict]:
        if self.session.state not in (InterviewState.LISTENING, InterviewState.CANDIDATE_SPEAKING):
            return []
        if self.endpointer is None:
            return []

        events = []
        frames = self.slicer.push(chunk)

        for frame in frames:
            try:
                is_speech = self.vad.is_speech(frame, vad_service.SAMPLE_RATE)
            except Exception as e:
                logger.warning(f"VAD error on a frame, treating as silence: {e}")
                is_speech = False

            if not self.endpointer.has_spoken:
                self._prebuffer.append(frame)
                if len(self._prebuffer) > vad_service.PREBUFFER_FRAMES:
                    self._prebuffer.pop(0)
            else:
                self.audio_buffer.extend(frame)

            result = self.endpointer.process_frame(is_speech)

            if result == "speech_started":
                for f in self._prebuffer:
                    self.audio_buffer.extend(f)
                self._prebuffer = []
                if self.session.candidate_started_speaking():
                    events.append({"event": "speech_started", "state": self.session.state})

            elif result == "speech_resumed":
                if self.session.state == InterviewState.LISTENING:
                    self.session.candidate_started_speaking()
                events.append({"event": "speech_resumed", "state": self.session.state})

            elif result == "speech_paused":
                if self.session.state == InterviewState.CANDIDATE_SPEAKING:
                    self.session.candidate_paused()
                events.append({"event": "speech_paused", "state": self.session.state})

            elif result == "answer_complete":
                events.extend(await self._finalize_answer())
                break  # this turn is over - ignore any remaining frames in this chunk

        return events

    async def _finalize_answer(self) -> list[dict]:
        async with self._processing_lock:
            if not self.session.finalize_answer():
                return [self._invalid("answer_endpoint")]

            events = [{"event": "transcribing", "state": self.session.state}]

            transcript = ""
            try:
                transcript = await speech_service.transcribe_pcm16(bytes(self.audio_buffer))
            except speech_service.WhisperUnavailableError:
                events.append({
                    "event": "error",
                    "message": "Local speech-to-text is unavailable (faster-whisper failed to load or run). "
                               "Continuing with an empty transcript for this answer.",
                })

            self.session.attach_transcript(transcript)

            listening_started_at = self.session.current_record.listening_started_at
            latency = self.endpointer.answer_latency_seconds(listening_started_at)
            duration = self.endpointer.audio_duration_seconds()
            speech_metrics = analytics_service.compute_speech_metrics(
                transcript, duration, self.endpointer.pause_durations, latency
            )
            self.session.attach_speech_metrics(speech_metrics)

            if self.vision_enabled:
                self.session.attach_vision_summary(self.face_extractor.summary())
                self.face_extractor = vision_service.FaceMetricsExtractor()

            events.append({
                "event": "answer_complete",
                "state": self.session.state,
                "transcript": transcript,
                "speech_metrics": speech_metrics,
            })

            self.session.start_analysis()
            events.append({"event": "answer_analysis", "state": self.session.state})

            evaluation = await llm_service.evaluate_answer(
                self.session.current_record.question, transcript, self.session.resume_summary
            )
            self.session.attach_evaluation(evaluation)
            events.append({"event": "evaluation_ready", "evaluation": evaluation})

            self.session.commit_current_record()
            self.session.analysis_done()

            if self.session.should_finish():
                self.session.transition(InterviewState.FINISHED)
                events.append(await self._build_final_report_event())
                return events

            if not self.session.transition(InterviewState.AI_SPEAKING):
                events.append(await self._build_final_report_event())
                return events

            history_for_llm = [
                {"question": r.question, "transcript": r.transcript, "evaluation": r.evaluation}
                for r in self.session.history
            ]
            topics_covered = [r.topic for r in self.session.history]
            q = await llm_service.generate_next_question(
                self.session.resume_summary, history_for_llm, self.session.current_difficulty, topics_covered
            )
            record = self.session.set_next_question(q["question"], q["difficulty"], q["topic"])
            events.append({
                "event": "ai_speak",
                "state": self.session.state,
                "question": record.question,
                "question_number": self.session.current_question_index + 1,
                "difficulty": record.difficulty,
                "topic": record.topic,
                "question_source": q.get("source"),
            })
            return events

    async def handle_video_frame(self, jpeg_bytes: bytes) -> dict | None:
        if not self.vision_enabled:
            return None
        if self.session.state not in (InterviewState.LISTENING, InterviewState.CANDIDATE_SPEAKING):
            return None
        metrics = await self.face_extractor.process_frame(jpeg_bytes)
        return {"event": "vision_update", "state": self.session.state, "metrics": metrics}

    async def end_interview(self) -> dict:
        self.session.force_finish()
        return await self._build_final_report_event()

    async def _build_final_report_event(self) -> dict:
        context = analytics_service.build_report_context(self.session)
        summary = await llm_service.generate_final_summary(context)
        report = {**context, "qualitative_summary": summary}
        self.session.final_report = report
        return {"event": "interview_finished", "state": self.session.state, "report": report}

    def _invalid(self, action: str) -> dict:
        return {
            "event": "invalid_transition",
            "state": self.session.state,
            "message": f"'{action}' isn't valid from state {self.session.state}.",
        }