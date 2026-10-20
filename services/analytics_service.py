"""
services/analytics_service.py

Deterministic, documented formulas turning raw collected signals
(speech timing, transcript, vision metrics, LLM evaluations) into
speech-analytics numbers and behavioral indicators.

Nothing here is random. The "behavioral confidence/nervousness
indicator" formulas are simple weighted combinations of OBSERVABLE
signals only - explicitly not a claim about real psychological state.
"""

import re

FILLER_WORDS = ["um", "uh", "like", "actually", "basically", "you know"]
FILLER_REGEX = re.compile(r"\b(" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b", re.IGNORECASE)


def compute_speech_metrics(transcript: str, audio_duration_sec, pause_durations: list[float], latency_sec) -> dict:
    words = transcript.split()
    word_count = len(words)
    duration = audio_duration_sec or 0.0
    wpm = (word_count / (duration / 60.0)) if duration > 0 else 0.0

    filler_count = len(FILLER_REGEX.findall(transcript))
    long_pauses = [p for p in pause_durations if p >= 1.0]

    return {
        "word_count": word_count,
        "audio_duration_sec": round(duration, 2),
        "words_per_minute": round(wpm, 1),
        "filler_count": filler_count,
        "filler_ratio": round(filler_count / word_count, 3) if word_count else 0.0,
        "pause_count": len(pause_durations),
        "long_pause_count": len(long_pauses),
        "total_pause_time_sec": round(sum(pause_durations), 2),
        "answer_latency_sec": round(latency_sec, 2) if latency_sec is not None else None,
    }


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_behavioral_indicators(speech_metrics_list: list[dict], vision_summaries: list[dict],
                                   evaluations: list[dict]) -> dict:
    if not speech_metrics_list:
        return {
            "confidence_indicator": None,
            "nervousness_indicator": None,
            "note": "Not enough speech data collected to compute behavioral indicators.",
        }

    avg_completeness = (
        sum(e.get("completeness", 0) for e in evaluations) / (10.0 * len(evaluations))
        if evaluations else 0.5
    )

    avg_filler_ratio = sum(m["filler_ratio"] for m in speech_metrics_list) / len(speech_metrics_list)
    fluency_score = _clip01(1.0 - avg_filler_ratio * 4)

    avg_pauses = sum(m["pause_count"] for m in speech_metrics_list) / len(speech_metrics_list)
    pause_score = _clip01(1.0 - (avg_pauses / 6.0))

    avg_face_visibility = None
    movement_stability = None
    if vision_summaries:
        avg_face_visibility = sum(v["face_visibility_ratio"] for v in vision_summaries) / len(vision_summaries)
        movement_values = [v["movement_variance"] for v in vision_summaries]
        max_movement = max(movement_values) if movement_values else 0.0
        avg_movement = sum(movement_values) / len(movement_values) if movement_values else 0.0
        movement_stability = _clip01(1.0 - (avg_movement / max_movement)) if max_movement > 0 else 1.0

    confidence_components = [avg_completeness, fluency_score, pause_score]
    if avg_face_visibility is not None:
        confidence_components.append(avg_face_visibility)
    if movement_stability is not None:
        confidence_components.append(movement_stability)
    confidence_indicator = round(100 * sum(confidence_components) / len(confidence_components), 1)

    nervousness_components = [1 - pause_score, _clip01(avg_filler_ratio * 4)]
    if movement_stability is not None:
        nervousness_components.append(1 - movement_stability)
    if avg_face_visibility is not None:
        nervousness_components.append(1 - avg_face_visibility)
    nervousness_indicator = round(100 * _clip01(sum(nervousness_components) / len(nervousness_components)), 1)

    return {
        "confidence_indicator": confidence_indicator,
        "nervousness_indicator": nervousness_indicator,
        "components": {
            "avg_completeness": round(avg_completeness, 3),
            "fluency_score": round(fluency_score, 3),
            "pause_score": round(pause_score, 3),
            "avg_face_visibility": round(avg_face_visibility, 3) if avg_face_visibility is not None else None,
            "movement_stability": round(movement_stability, 3) if movement_stability is not None else None,
        },
        "note": "Estimates from observable speech/vision signals only - not a measurement "
                "of the candidate's actual psychological state.",
    }


def build_report_context(session) -> dict:
    """Pure numeric aggregation - used both for the dashboard AND as the
    factual context handed to Gemma for the qualitative summary, so the
    LLM interprets real numbers instead of inventing them."""
    records = session.history
    evaluations = [r.evaluation for r in records if r.evaluation]

    def avg(key):
        vals = [e[key] for e in evaluations if key in e]
        return round(sum(vals) / len(vals), 2) if vals else None

    speech_metrics_list = [r.speech_metrics for r in records if r.speech_metrics]
    vision_summaries = [r.vision_summary for r in records if r.vision_summary]
    topics_covered = list({r.topic for r in records if r.topic})

    resume_skills = set(s.lower() for s in session.resume_summary.get("skills", []))
    covered_lower = set(t.lower() for t in topics_covered)
    skills_touched_on = {s for s in resume_skills if any(s in t for t in covered_lower)}

    strong_topics = list({
        r.topic for r in records
        if r.evaluation and r.topic and r.evaluation.get("technical_correctness", 0) >= 7
    })
    weak_topics = list({
        r.topic for r in records
        if r.evaluation and r.topic and r.evaluation.get("technical_correctness", 0) <= 4
    })

    return {
        "questions_asked": len(records),
        "questions_answered": len([r for r in records if r.transcript]),
        "average_scores": {
            "technical_correctness": avg("technical_correctness"),
            "completeness": avg("completeness"),
            "problem_solving": avg("problem_solving"),
            "communication": avg("communication"),
        },
        "topics_covered": topics_covered,
        "strong_topics": strong_topics,
        "weak_topics": weak_topics,
        "resume_skills": list(resume_skills),
        "resume_skills_touched_on": list(skills_touched_on),
        "behavioral_indicators": compute_behavioral_indicators(speech_metrics_list, vision_summaries, evaluations),
        "per_question": [
            {
                "index": r.index,
                "question": r.question,
                "topic": r.topic,
                "difficulty": r.difficulty,
                "transcript": r.transcript,
                "evaluation": r.evaluation,
                "speech_metrics": r.speech_metrics,
            }
            for r in records
        ],
    }