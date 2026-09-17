import threading

from flask import Flask, request, jsonify
from flask_cors import CORS

from orchestrator import career_orchestrator
from memory import save_memory, get_memory_value, delete_memory_key
from text_extractor import validate_file, extract_text, ExtractionError
from resume_agent import analyze_resume
import resume_cache

app = Flask(__name__)
CORS(app)
app.url_map.strict_slashes = False


def bad_request(msg):
    return jsonify({"error": msg}), 400


def api_error(message, http_status=400):
    return jsonify({"status": "error", "progress": "failed", "data": None, "error": message}), http_status


def api_success(data, progress="done"):
    return jsonify({"status": "success", "progress": progress, "data": data})


def _resolve_resume_text(data: dict):
    """
    Single source of truth for 'which resume text do we use for this
    request', in priority order:
      1. Caller passed the raw text directly (data['resume']) - old
         behavior, still supported so nothing breaks.
      2. Caller passed a resume_id (data['resume_id']) - looked up
         from the in-memory cache, no re-extraction needed.
      3. Nothing passed - fall back to whatever resume is currently
         active for this session (memory.json 'current_resume_id'),
         so Job Match / Skill Gap / Career Mentor can all reuse the
         one resume the user already uploaded without resending it.
    Returns None if no resume can be found anywhere.
    """
    resume = (data.get("resume") or "").strip()
    if resume:
        return resume

    resume_id = data.get("resume_id") or get_memory_value("current_resume_id")
    if resume_id:
        record = resume_cache.get_resume(resume_id)
        if record:
            return record["text"]

    return None


# ---------------------------------------------------------------------------
# Generic agents (unchanged from before - career mentor, job match, etc.)
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return "AI Career Navigator API is running"


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = data.get("message")

    if not user_message:
        return bad_request("'message' is required")

    save_memory({"last_question": user_message})
    reply = career_orchestrator("career", user_message)

    return jsonify({"reply": reply})


@app.route("/match", methods=["POST"])
def match():
    data = request.json or {}
    job = (data.get("job_description") or data.get("job") or "").strip()

    if not job:
        return api_error("'job_description' is required", http_status=400)

    resume = _resolve_resume_text(data)
    if not resume:
        return api_error(
            "No resume on file. Upload one on the Resume Analyzer page first.",
            http_status=400,
        )

    save_memory({"last_job": job})
    result = career_orchestrator("match", {"resume": resume, "job": job})

    if isinstance(result, dict):
        return api_success({"match": result}, progress="done")

    # Orchestrator-level failure (e.g. Ollama unreachable) - a string
    # error message, not a match result.
    return api_error(str(result), http_status=500)


@app.route("/skill-gap", methods=["POST"])
def skill_gap():
    data = request.json or {}
    job = data.get("job")

    if not job:
        return bad_request("'job' is required")

    resume = _resolve_resume_text(data)
    if not resume:
        return bad_request("No resume on file. Upload one on the Resume Analyzer page first.")

    result = career_orchestrator("skillgap", {"resume": resume, "job": job})

    return jsonify({"result": result})


@app.route("/roadmap", methods=["POST"])
def roadmap():
    data = request.json or {}
    goal = data.get("goal")

    if not goal:
        return bad_request("'goal' is required")

    result = career_orchestrator("roadmap", goal)

    return jsonify({"roadmap": result})


@app.route("/interview", methods=["POST"])
def interview():
    data = request.json or {}
    action = data.get("action")

    if action == "question":
        role = data.get("role")
        if not role:
            return bad_request("'role' is required for action=question")

        question = career_orchestrator("interview_question", role)
        return jsonify({"question": question})

    elif action == "evaluate":
        question = data.get("question")
        answer = data.get("answer")
        if not question or not answer:
            return bad_request("'question' and 'answer' are both required for action=evaluate")

        feedback = career_orchestrator(
            "interview_feedback",
            {"question": question, "answer": answer}
        )
        return jsonify({"feedback": feedback})

    return bad_request("'action' must be 'question' or 'evaluate'")


# ---------------------------------------------------------------------------
# Resume pipeline - split into 4 explicit stages:
#   1. Upload + validate           -> /resume/upload
#   2. Extract text (inside step1, but isolated in text_extractor.py)
#   3. AI analysis (background)    -> /resume/analyze
#   4. Result response (polling)   -> /resume/analyze/status/<job_id>
# ---------------------------------------------------------------------------

@app.route("/resume/upload", methods=["POST"])
def resume_upload():
    """
    FAST endpoint. Validates the file and extracts text only - no LLM
    call happens here, which is why this responds in well under a
    second even for a multi-page resume. This is what lets the
    frontend show "uploaded" state immediately.
    """
    if "resume" not in request.files:
        return api_error("No file uploaded under field 'resume'")

    file = request.files["resume"]
    if file.filename == "":
        return api_error("No file selected")

    try:
        ext, size = validate_file(file)
        text = extract_text(file, ext)
    except ExtractionError as e:
        return api_error(str(e))
    except Exception as e:
        return api_error(f"Unexpected error reading file: {e}", http_status=500)

    resume_id, content_hash, was_duplicate = resume_cache.store_resume(
        text=text, filename=file.filename, size=size
    )

    # This resume is now the active one for the whole session - every
    # other module (chat, match, skill-gap) will use it automatically
    # from this point on, even before "Analyze" is clicked.
    save_memory({"current_resume_id": resume_id})

    # If we've already analyzed this exact resume before, tell the
    # frontend up front so it can skip straight to "Analyze" with a
    # note that this will be instant.
    cached_analysis = resume_cache.get_cached_analysis(content_hash)

    return api_success({
        "resume_id": resume_id,
        "filename": file.filename,
        "size_bytes": size,
        "word_count": len(text.split()),
        "preview": text[:220],
        "duplicate_of_existing": was_duplicate,
        "analysis_cached": cached_analysis is not None,
        # returned so the frontend can render the cached analysis
        # immediately instead of making the user click "Analyze" again
        "cached_analysis": cached_analysis,
    }, progress="uploaded")


@app.route("/resume/analyze", methods=["POST"])
def resume_analyze():
    """
    Kicks off analysis for a previously uploaded resume_id.
    Returns immediately - either with a cached result (status=success,
    instant) or with a job_id to poll (status=processing).
    """
    data = request.json or {}
    resume_id = data.get("resume_id")

    if not resume_id:
        return api_error("'resume_id' is required")

    resume = resume_cache.get_resume(resume_id)
    if not resume:
        return api_error("Unknown resume_id. Please re-upload the file.", http_status=404)

    content_hash = resume["hash"]
    save_memory({"current_resume_id": resume_id})

    # --- Cache hit: no LLM call needed at all ---
    cached = resume_cache.get_cached_analysis(content_hash)
    if cached:
        save_memory({"resume": resume["text"]})
        return api_success({
            "analysis": cached,
            "resume_text": resume["text"],
            "cached": True,
        }, progress="done")

    # --- Cache miss: run analysis in a background thread ---
    job_id = resume_cache.create_job(content_hash)

    def _run():
        resume_cache.update_job(job_id, progress="analyzing")
        try:
            result = analyze_resume(resume["text"])
            resume_cache.set_cached_analysis(content_hash, result)
            save_memory({"resume": resume["text"]})
            resume_cache.update_job(
                job_id,
                status="success",
                progress="done",
                data={"analysis": result, "resume_text": resume["text"], "cached": False},
            )
        except Exception as e:
            resume_cache.update_job(
                job_id, status="error", progress="failed", error=str(e)
            )

    threading.Thread(target=_run, daemon=True).start()

    return jsonify({
        "status": "processing",
        "progress": "queued",
        "data": {"job_id": job_id},
    })


@app.route("/resume/analyze/status/<job_id>", methods=["GET"])
def resume_analyze_status(job_id):
    job = resume_cache.get_job(job_id)
    if not job:
        return api_error("Unknown job_id", http_status=404)

    if job["status"] == "error":
        return api_error(job["error"] or "Analysis failed", http_status=500)

    return jsonify({
        "status": job["status"],       # "processing" | "success"
        "progress": job["progress"],   # "queued" | "analyzing" | "done"
        "data": job["data"],
    })


@app.route("/resume/current", methods=["GET"])
def resume_current():
    """
    Lets the frontend rehydrate the full resume+analysis state on page
    load/refresh instead of relying on the browser tab never having
    unloaded. If a resume is active for this session, returns the same
    shape /resume/analyze returns; otherwise returns data: null so the
    UI knows to show the empty upload screen.
    """
    resume_id = get_memory_value("current_resume_id")
    if not resume_id:
        return api_success(None, progress="empty")

    profile = resume_cache.get_full_profile(resume_id)
    if not profile:
        # Server restarted since upload - in-memory cache is gone.
        # Don't keep pointing at a dead resume_id.
        delete_memory_key("current_resume_id")
        return api_success(None, progress="empty")

    return api_success({
        "resume_id": profile["resume_id"],
        "filename": profile["filename"],
        "size_bytes": profile["size"],
        "word_count": len(profile["text"].split()),
        "analysis": profile["analysis"],   # None if not analyzed yet
        "resume_text": profile["text"],
        "cached": profile["analysis"] is not None,
    }, progress="done" if profile["analysis"] else "uploaded")


@app.route("/resume/current", methods=["DELETE"])
def resume_remove():
    """
    'Remove Resume'. Clears the active-resume pointer (so /chat,
    /match, /skill-gap stop auto-reusing it) without touching the
    analysis cache - if the user re-uploads the same file later it's
    still an instant cache hit instead of a fresh LLM call.
    """
    delete_memory_key("current_resume_id")
    delete_memory_key("resume")
    return api_success(None, progress="removed")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
