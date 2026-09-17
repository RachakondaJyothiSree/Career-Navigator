"""
Orchestrator Agent
--------------------
Single entry point that routes a task to the correct specialist agent.

BUG FIX vs. original orchestrator.py:
app.py was calling this with task names like "job_match" and
"skill_gap" and actions "interview_question" / "interview_feedback",
but this file only recognized "match", "skillgap", and "interview".
Every job-match, skill-gap, and interview request was silently
falling through to "Invalid task". The task names below are now the
single source of truth, and app.py has been updated to match them
exactly.
"""

from agent import career_agent
from job_agent import analyze_job
from skill_gap_agent import analyze_skill_gap
from roadmap_agent import generate_roadmap
from interview_agent import generate_interview_question, generate_interview_feedback

# Central registry: task name -> handler function.
# Using a dict instead of a long if/elif chain makes it obvious at a
# glance which tasks exist and keeps app.py and orchestrator.py in sync.
#
# NOTE: there is intentionally no "resume" task here anymore. Resume
# analysis now has its own dedicated, cached, async pipeline
# (see resume_cache.py + the /resume/* routes in app.py) instead of
# going through this generic synchronous orchestrator - that's what
# makes it possible to return instantly on a cache hit and to poll
# progress on a cache miss instead of blocking one long request.
_TASKS = {
    "career": lambda data: career_agent(data),
    "match": lambda data: analyze_job(data["resume"], data["job"]),
    "skillgap": lambda data: analyze_skill_gap(data["resume"], data["job"]),
    "roadmap": lambda data: generate_roadmap(data),
    "interview_question": lambda data: generate_interview_question(data),
    "interview_feedback": lambda data: generate_interview_feedback(
        data["question"], data["answer"]
    ),
}


def career_orchestrator(task: str, data):
    handler = _TASKS.get(task)

    if handler is None:
        return f"Invalid task: '{task}'"

    try:
        return handler(data)
    except KeyError as e:
        return f"Missing required field: {e}"
    except Exception as e:
        # Agents call a local Ollama model - most failures here are
        # "model not running" / "model not pulled" style errors.
        return f"Agent error: {e}"
