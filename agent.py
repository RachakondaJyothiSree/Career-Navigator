"""
Career Mentor Agent
---------------------
Handles free-form chat on the Career Mentor page. Personalizes the
response with the user's stored resume (if one has been uploaded)
so advice references their actual background instead of being generic.
"""

import ollama
from memory import load_memory

MODEL = "llama3.2"


def career_agent(message: str) -> str:
    message = (message or "").strip()
    if not message:
        return "Please type a question first."

    memory = load_memory()
    resume_snippet = (memory.get("resume") or "")[:800]

    context = f"\nCandidate background (from their uploaded resume):\n{resume_snippet}\n" if resume_snippet else ""

    prompt = f"""You are an AI Career Mentor for software/tech careers.
{context}
User question:
{message}

Give a short, direct, structured answer (max ~150 words):
- Key advice
- 1-2 next steps"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.4,
            "num_predict": 260,
        },
    )

    return response["message"]["content"].strip()
