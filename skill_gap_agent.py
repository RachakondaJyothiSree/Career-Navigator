"""
Skill Gap Analyzer Agent
-------------------------
Breaks down exactly which skills a candidate has vs. what a job needs,
prioritized so the user knows what to learn first.
"""

import ollama

MODEL = "llama3.2"


def analyze_skill_gap(resume: str, job: str) -> str:
    resume = (resume or "")[:3000]
    job = (job or "")[:1500]

    if not resume.strip() or not job.strip():
        return "Please provide both a resume and a job description."

    prompt = f"""You are an AI Skill Gap Analyzer. Compare the resume with the job description.

Respond in this exact structure, keep each section short:
## Overall Match
## Existing Skills
## Missing Skills
## High Priority to Learn
## Recommended Resources

Resume:
{resume}

Job Description:
{job}"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.3,
            "num_predict": 350,
        },
    )

    return response["message"]["content"].strip()
