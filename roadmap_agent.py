"""
Learning Roadmap Agent
------------------------
Generates a 4-week learning plan for a chosen career goal/role.
"""

import ollama

MODEL = "llama3.2"


def generate_roadmap(goal: str) -> str:
    goal = (goal or "Software Developer").strip()

    prompt = f"""Create a concise 4-week learning roadmap for becoming a {goal}.

Respond in this exact structure:
## Week 1
## Week 2
## Week 3
## Week 4
## Projects (2 max)
## Resources (3 max)

Keep every bullet to one short line."""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.4,
            "num_predict": 400,
        },
    )

    return response["message"]["content"].strip()
