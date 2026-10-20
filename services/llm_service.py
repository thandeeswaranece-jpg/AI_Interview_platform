"""
services/llm_service.py

Every Ollama / Gemma 3 4B call lives here. Each function:
  1. builds a structured prompt
  2. asks Ollama for JSON output (format="json")
  3. validates the parsed result
  4. falls back to a deterministic, HONESTLY-LABELED result if Gemma
     is unavailable or returns unusable JSON - never invents a score.

Every returned dict includes a "source" field: "gemma" or
"fallback_heuristic" (or "no_transcript"), so nothing pretends to be a
real Gemma evaluation when it isn't.
"""

import json
import re
import logging
import asyncio
from typing import Optional

import ollama

logger = logging.getLogger("interview.llm")

MODEL_NAME = "gemma3:4b"

# Used only if Gemma is completely unavailable when picking the very
# next question - keeps the interview moving instead of stalling.
FALLBACK_TOPIC_POOL = [
    ("General background", "Tell me about your overall technical background and the area you enjoy working in most."),
    ("Problem solving", "Describe a time you had to debug a difficult technical problem. How did you approach it?"),
    ("Fundamentals", "Explain the difference between a stack and a queue, and give an example of when you'd use each."),
    ("Projects", "Pick a project you've worked on and walk me through the biggest technical decision you made in it."),
]


def _run_chat(messages: list[dict]) -> Optional[dict]:
    """Blocking Ollama call - always run this inside a thread executor."""
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            format="json",
            options={"temperature": 0.4},
        )
    except Exception as e:
        logger.warning(f"Ollama call failed (is 'ollama serve' running and is gemma3:4b pulled?): {e}")
        return None

    raw = response.get("message", {}).get("content", "")
    return _extract_json(raw)


def _extract_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


async def _run_chat_async(messages: list[dict]) -> Optional[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_chat, messages)


# ---------------------------------------------------------------- resume ----

async def analyze_resume(resume_text: str) -> dict:
    prompt = f"""You are a technical interviewer preparing for an interview.
Read the resume text below and extract a structured summary.

Return ONLY valid JSON with this exact shape:
{{
  "summary": "2-3 sentence summary of the candidate",
  "skills": ["skill1", "skill2"],
  "projects": ["short project description"],
  "experience_level": "student" | "junior" | "mid" | "senior"
}}

Only include skills and projects that are ACTUALLY present in the resume text below.
Do not invent anything not present in the text.

Resume text:
---
{resume_text[:6000]}
---
"""
    result = await _run_chat_async([{"role": "user", "content": prompt}])

    if result and isinstance(result.get("skills"), list) and isinstance(result.get("summary"), str):
        result["source"] = "gemma"
        result.setdefault("projects", [])
        result.setdefault("experience_level", "unknown")
        return result

    logger.warning("Resume analysis fell back to a raw text snippet (Gemma unavailable/unparseable).")
    return {
        "summary": resume_text[:300].replace("\n", " "),
        "skills": [],
        "projects": [],
        "experience_level": "unknown",
        "source": "fallback_heuristic",
    }


# ------------------------------------------------------------- questions ----

async def generate_next_question(
    resume_summary: dict,
    history: list[dict],
    current_difficulty: str,
    topics_covered: list[str],
) -> dict:
    history_text = "\n".join(
        f"- Q: {h['question']}\n  A: {h['transcript']}\n  Eval: {json.dumps(h.get('evaluation') or {})}"
        for h in history
    ) or "(no questions asked yet - this is the first question)"

    prompt = f"""You are a senior technical interviewer conducting a live, adaptive interview.

Candidate resume summary: {resume_summary.get('summary', '')}
Candidate resume skills: {', '.join(resume_summary.get('skills', [])) or 'unknown'}
Candidate resume projects: {'; '.join(resume_summary.get('projects', [])) or 'none listed'}

Interview so far:
{history_text}

Current difficulty level: {current_difficulty}
Topics already covered: {', '.join(topics_covered) or 'none'}

Decide the single best next interview question, following these rules:
- Base it on the resume and/or the candidate's previous answers.
- If the last answer was strong/correct, increase difficulty or go deeper on that same topic.
- If the last answer was weak/incomplete, ask a simpler clarifying/fundamentals question on the
  same topic, OR pivot to a different resume topic.
- If a resume project hasn't been explored yet, consider asking about it.
- Do not repeat a topic that has already been thoroughly covered, unless following up on a weakness.
- Do not invent skills or projects that are not in the resume.
- Keep the question concise (1-3 sentences).

Return ONLY valid JSON with this exact shape:
{{
  "question": "the next interview question",
  "difficulty": "easy" | "medium" | "hard",
  "topic": "short topic label, e.g. 'Java collections' or 'Project: X'",
  "is_followup": true
}}
"""
    result = await _run_chat_async([{"role": "user", "content": prompt}])

    if result and isinstance(result.get("question"), str) and result["question"].strip():
        result.setdefault("difficulty", current_difficulty)
        result.setdefault("topic", "general")
        result.setdefault("is_followup", False)
        result["source"] = "gemma"
        return result

    logger.warning("Adaptive question generation fell back to a generic pool question.")
    idx = len(history) % len(FALLBACK_TOPIC_POOL)
    topic, question = FALLBACK_TOPIC_POOL[idx]
    return {
        "question": question,
        "difficulty": current_difficulty,
        "topic": topic,
        "is_followup": False,
        "source": "fallback_heuristic",
    }


# ------------------------------------------------------------- evaluation --

async def evaluate_answer(question: str, transcript: str, resume_summary: dict) -> dict:
    if not transcript or len(transcript.strip()) < 2:
        return {
            "technical_correctness": 0,
            "completeness": 0,
            "problem_solving": 0,
            "communication": 0,
            "concepts_demonstrated": [],
            "concepts_missing": [],
            "strengths": [],
            "weaknesses": ["No answer was captured (silence or transcription failure)."],
            "follow_up_suggested": False,
            "source": "no_transcript",
        }

    prompt = f"""You are grading one answer in a technical interview.

Question asked: {question}
Candidate's spoken answer (auto-transcribed, may contain minor errors): {transcript}
Candidate resume skills: {', '.join(resume_summary.get('skills', [])) or 'unknown'}

Grade the answer honestly using integers 0-10 for each of:
technical_correctness, completeness, problem_solving, communication.

Also identify: concepts_demonstrated, concepts_missing, strengths, weaknesses (short string lists),
and follow_up_suggested (true if this answer deserves a deeper follow-up).

Return ONLY valid JSON with this exact shape:
{{
  "technical_correctness": 0,
  "completeness": 0,
  "problem_solving": 0,
  "communication": 0,
  "concepts_demonstrated": [],
  "concepts_missing": [],
  "strengths": [],
  "weaknesses": [],
  "follow_up_suggested": false
}}
"""
    result = await _run_chat_async([{"role": "user", "content": prompt}])

    if result and all(
        isinstance(result.get(k), (int, float))
        for k in ("technical_correctness", "completeness", "problem_solving", "communication")
    ):
        for k in ("technical_correctness", "completeness", "problem_solving", "communication"):
            result[k] = max(0, min(10, round(float(result[k]))))
        result.setdefault("concepts_demonstrated", [])
        result.setdefault("concepts_missing", [])
        result.setdefault("strengths", [])
        result.setdefault("weaknesses", [])
        result.setdefault("follow_up_suggested", False)
        result["source"] = "gemma"
        return result

    logger.warning("Answer evaluation fell back to a deterministic heuristic rubric (Gemma unavailable/unparseable).")
    return _heuristic_answer_evaluation(transcript, resume_summary)


def _heuristic_answer_evaluation(transcript: str, resume_summary: dict) -> dict:
    """
    Deterministic, transparent fallback used ONLY when Gemma's JSON could
    not be parsed or Ollama is unreachable. This is a rough proxy based on
    answer length and resume-keyword overlap - NOT a real evaluation, and
    it is always tagged 'fallback_heuristic' so it's never mistaken for one.
    """
    words = transcript.split()
    word_count = len(words)

    completeness = min(10, round(word_count / 12))

    skills = [s.lower() for s in resume_summary.get("skills", [])]
    transcript_lower = transcript.lower()
    overlap = sum(1 for s in skills if s and s in transcript_lower)
    technical_correctness = min(10, overlap * 3) if skills else min(10, round(word_count / 20))

    communication = min(10, round(word_count / 15))
    problem_solving = min(10, round((completeness + technical_correctness) / 3))

    return {
        "technical_correctness": technical_correctness,
        "completeness": completeness,
        "problem_solving": problem_solving,
        "communication": communication,
        "concepts_demonstrated": [],
        "concepts_missing": [],
        "strengths": [],
        "weaknesses": [],
        "follow_up_suggested": False,
        "source": "fallback_heuristic",
    }


# --------------------------------------------------------------- summary ----

async def generate_final_summary(report_context: dict) -> dict:
    prompt = f"""You are writing the final qualitative summary of a technical interview report.

Aggregated, ALREADY-COMPUTED data (interpret it, do not recompute or invent numbers):
{json.dumps(report_context, indent=2)[:6000]}

Return ONLY valid JSON with this exact shape:
{{
  "overall_summary": "2-4 sentences",
  "top_strengths": ["...", "..."],
  "top_weaknesses": ["...", "..."],
  "recommended_preparation_areas": ["...", "..."]
}}
"""
    result = await _run_chat_async([{"role": "user", "content": prompt}])

    if result and isinstance(result.get("overall_summary"), str):
        result.setdefault("top_strengths", [])
        result.setdefault("top_weaknesses", [])
        result.setdefault("recommended_preparation_areas", [])
        result["source"] = "gemma"
        return result

    logger.warning("Final summary fell back to a templated message (Gemma unavailable/unparseable).")
    return {
        "overall_summary": "Automated qualitative summary unavailable (Gemma did not return usable output "
                            "for this section). The numeric scores and per-question breakdown below are "
                            "still real, computed data.",
        "top_strengths": [],
        "top_weaknesses": [],
        "recommended_preparation_areas": [],
        "source": "fallback_heuristic",
    }