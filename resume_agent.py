import json
import ollama

MODEL = "llama3.2"
MAX_RESUME_CHARS = 3000


def analyze_resume(resume_text: str) -> dict:
    trimmed = resume_text[:MAX_RESUME_CHARS]

    prompt = f"""You are an AI Resume Analyzer. Analyze the resume and respond with ONLY valid JSON (no markdown, no commentary) in exactly this shape:

{{
  "summary": "2-3 sentence overview of the candidate",
  "skills": ["..."],
  "experience": ["one line per role/project with impact"],
  "education": ["one line per degree/certification"],
  "strengths": ["..."],
  "weaknesses": ["..."],
  "ats_score": 0,
  "suggestions": ["..."]
}}

Rules:
- "ats_score" is an integer 0-100 estimating ATS (resume-parser) compatibility.
- Keep each list to at most 6 short items.
- "summary" must be plain text, no lists.

Resume:
{trimmed}"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.2,
            "num_predict": 500,
            "num_ctx": 2048,
        },
        keep_alive="30m",
    )

    raw = response["message"]["content"].strip()
    return _safe_parse_json(raw)


def _safe_parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("json", "", 1)

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        return {
            "summary": "",
            "skills": [],
            "experience": [],
            "education": [],
            "strengths": [],
            "weaknesses": [],
            "ats_score": None,
            "suggestions": [],
            "raw_text": raw,
        }

    return {
        "summary": parsed.get("summary", ""),
        "skills": parsed.get("skills", []),
        "experience": parsed.get("experience", []),
        "education": parsed.get("education", []),
        "strengths": parsed.get("strengths", []),
        "weaknesses": parsed.get("weaknesses", []),
        "ats_score": parsed.get("ats_score"),
        "suggestions": parsed.get("suggestions", []),
    }
