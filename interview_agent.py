"""
Interview Preparation Agent
----------------------------
BUG FIX vs. original interview_agent.py:
app.py's /interview endpoint needs two distinct behaviors:
  1. action == "question"  -> generate ONE interview question for a role
  2. action == "evaluate"  -> give feedback on a specific answer
The original file only had a single generate_interview(data) function
that always asked the model for technical + project + HR + sample
answers together, which doesn't match either use case and wastes
tokens. This file replaces it with two focused, fast functions.
"""

import ollama

MODEL = "llama3.2"


def generate_interview_question(role: str) -> str:
    """Return ONE realistic interview question for the given role."""
    role = (role or "Software Developer").strip()

    prompt = f"""You are a technical interviewer for a {role} position.

Ask exactly ONE interview question (technical, project-based, or HR —
vary it each time). Output ONLY the question text, no numbering,
no preamble, no explanation."""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.7,
            "num_predict": 80,
        },
    )

    return response["message"]["content"].strip()


def generate_interview_feedback(question: str, answer: str) -> str:
    """Return short, structured feedback on a candidate's answer."""
    question = (question or "")[:500]
    answer = (answer or "")[:1500]

    prompt = f"""You are an interview coach. Evaluate this answer concisely.

Question: {question}
Candidate answer: {answer}

Respond in this exact format:
Score: <x>/10
Strengths: <one short line>
Improve: <one short line>
Better answer (2-3 sentences):"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.3,
            "num_predict": 220,
        },
    )

    return response["message"]["content"].strip()


def generate_interview_set(role_or_data) -> str:
    """
    Kept for backward compatibility: generates a fuller question set
    (technical / project / HR / sample answers) if some other part of
    the app wants a full prep sheet instead of one-question-at-a-time.
    """
    prompt = f"""You are an AI Interview Preparation Agent.

Based on: {role_or_data}

Give a concise set:
1. 2 technical questions
2. 1 project question
3. 1 HR question
4. One short sample answer for the first question

Keep the whole response under 200 words."""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.3,
            "num_predict": 300,
        },
    )

    return response["message"]["content"].strip()
