"""
services/interview_state.py

Core interview state machine + session data model. This module is
transport-agnostic: no WebSocket, mic, or camera code lives here.
It only tracks valid transitions and stores interview data.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import time


class InterviewState(str, Enum):
    IDLE = "IDLE"
    AI_SPEAKING = "AI_SPEAKING"
    LISTENING = "LISTENING"
    CANDIDATE_SPEAKING = "CANDIDATE_SPEAKING"
    ANSWER_COMPLETE = "ANSWER_COMPLETE"
    ANSWER_ANALYSIS = "ANSWER_ANALYSIS"
    NEXT_QUESTION = "NEXT_QUESTION"
    FINISHED = "FINISHED"


# Explicit allow-list of transitions. Anything not listed is rejected.
# The backend is authoritative - the frontend never sets state directly.
VALID_TRANSITIONS = {
    InterviewState.IDLE: {InterviewState.AI_SPEAKING},
    InterviewState.AI_SPEAKING: {InterviewState.LISTENING},
    InterviewState.LISTENING: {
        InterviewState.CANDIDATE_SPEAKING,
        InterviewState.ANSWER_COMPLETE,   # sustained silence can end a turn straight from LISTENING
        InterviewState.FINISHED,
    },
    InterviewState.CANDIDATE_SPEAKING: {
        InterviewState.LISTENING,          # natural pause mid-answer
        InterviewState.ANSWER_COMPLETE,
        InterviewState.FINISHED,
    },
    InterviewState.ANSWER_COMPLETE: {InterviewState.ANSWER_ANALYSIS, InterviewState.FINISHED},
    InterviewState.ANSWER_ANALYSIS: {InterviewState.NEXT_QUESTION, InterviewState.FINISHED},
    InterviewState.NEXT_QUESTION: {InterviewState.AI_SPEAKING, InterviewState.FINISHED},
    InterviewState.FINISHED: set(),
}

# Fixed interview length. Within your 8-12 target range; kept as a single
# constant instead of "adaptive length" logic, which would be much harder
# to make reliable for a beginner-run project.
TARGET_QUESTION_COUNT = 10


@dataclass
class QuestionRecord:
    index: int
    question: str
    difficulty: str = "medium"
    topic: str = "general"
    transcript: str = ""
    evaluation: Optional[dict] = None
    speech_metrics: Optional[dict] = None
    vision_summary: Optional[dict] = None
    question_asked_at: float = field(default_factory=time.time)
    listening_started_at: Optional[float] = None
    answer_completed_at: Optional[float] = None


class InterviewSession:
    def __init__(self, session_id: str, resume_text: str, resume_summary: dict):
        self.session_id = session_id
        self.state: InterviewState = InterviewState.IDLE
        self.resume_text = resume_text
        self.resume_summary = resume_summary
        self.history: list[QuestionRecord] = []
        self.current_record: Optional[QuestionRecord] = None
        self.current_question_index: int = -1
        self.current_difficulty: str = "easy"
        self.created_at = time.time()
        self.final_report: Optional[dict] = None

    # ---- core transition guard -----------------------------------------
    def transition(self, new_state: InterviewState) -> bool:
        allowed = VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            return False
        self.state = new_state
        return True

    def force_finish(self):
        """Explicit user-initiated 'End Interview' - must work from ANY
        state, so it bypasses the normal transition graph on purpose."""
        self.state = InterviewState.FINISHED

    # ---- turn lifecycle --------------------------------------------------
    def set_next_question(self, question: str, difficulty: str, topic: str) -> QuestionRecord:
        self.current_question_index += 1
        record = QuestionRecord(
            index=self.current_question_index,
            question=question,
            difficulty=difficulty,
            topic=topic,
        )
        self.current_record = record
        self.current_difficulty = difficulty
        return record

    def ai_finished_speaking(self) -> bool:
        ok = self.transition(InterviewState.LISTENING)
        if ok and self.current_record:
            self.current_record.listening_started_at = time.time()
        return ok

    def candidate_started_speaking(self) -> bool:
        return self.transition(InterviewState.CANDIDATE_SPEAKING)

    def candidate_paused(self) -> bool:
        return self.transition(InterviewState.LISTENING)

    def finalize_answer(self) -> bool:
        ok = self.transition(InterviewState.ANSWER_COMPLETE)
        if ok and self.current_record:
            self.current_record.answer_completed_at = time.time()
        return ok

    def attach_transcript(self, transcript: str):
        if self.current_record:
            self.current_record.transcript = transcript

    def attach_evaluation(self, evaluation: dict):
        if self.current_record:
            self.current_record.evaluation = evaluation

    def attach_speech_metrics(self, metrics: dict):
        if self.current_record:
            self.current_record.speech_metrics = metrics

    def attach_vision_summary(self, summary: dict):
        if self.current_record:
            self.current_record.vision_summary = summary

    def commit_current_record(self):
        if self.current_record:
            self.history.append(self.current_record)

    def start_analysis(self) -> bool:
        return self.transition(InterviewState.ANSWER_ANALYSIS)

    def analysis_done(self) -> bool:
        return self.transition(InterviewState.NEXT_QUESTION)

    def should_finish(self) -> bool:
        return len(self.history) >= TARGET_QUESTION_COUNT

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "current_question_index": self.current_question_index,
            "questions_answered": len(self.history),
            "current_question": self.current_record.question if self.current_record else None,
        }