"""
Resume Cache & Job Store
--------------------------
Three separate in-memory stores, each solving a specific problem from
the brief:

1. `_resumes`      resume_id -> extracted text + metadata.
                    Lets /resume/upload return instantly and
                    /resume/analyze reuse the text without re-parsing
                    the file.

2. `_hash_to_id`    content hash -> resume_id.
                    If the SAME resume content is uploaded twice
                    (even under a different filename), we reuse the
                    existing resume_id instead of storing it again.
                    This is what "avoid saving duplicate files" and
                    "same resume processed multiple times" actually
                    mean in an in-memory-only design: there is nothing
                    to duplicate because we recognize it up front.

3. `_analysis_cache`  content hash -> AI analysis result.
                    If we've already analyzed this exact resume text
                    before (same person re-uploading, or clicking
                    Analyze twice), we return the cached result
                    immediately with ZERO extra LLM calls.

4. `_jobs`          job_id -> background analysis job status.
                    The LLM call happens in a background thread so
                    /resume/analyze returns immediately with a job_id,
                    and the frontend polls /resume/analyze/status/<id>
                    for progress instead of holding one long blocking
                    HTTP request open (which is what caused the UI to
                    look "stuck" / fall back to the upload screen).

None of this needs a database for a project this size - it's a single
Flask process, so plain dicts + a lock are enough. If you outgrow a
single process, swap these for Redis with the same function signatures.
"""

import hashlib
import threading
import time
import uuid

_lock = threading.Lock()

_resumes = {}          # resume_id -> {text, filename, size, hash, created_at}
_hash_to_id = {}        # content_hash -> resume_id
_analysis_cache = {}    # content_hash -> analysis dict
_jobs = {}              # job_id -> {status, progress, data, error, hash}


def hash_text(text: str) -> str:
    normalized = " ".join(text.split())  # collapse whitespace so trivial formatting diffs don't miss the cache
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def store_resume(text: str, filename: str, size: int):
    """Register extracted resume text. Reuses an existing id if this exact content was seen before."""
    content_hash = hash_text(text)

    with _lock:
        existing_id = _hash_to_id.get(content_hash)
        if existing_id:
            # Same content already known - just refresh metadata, don't duplicate storage.
            _resumes[existing_id]["filename"] = filename
            _resumes[existing_id]["last_seen"] = time.time()
            return existing_id, content_hash, True  # True = was a duplicate

        resume_id = str(uuid.uuid4())
        _resumes[resume_id] = {
            "text": text,
            "filename": filename,
            "size": size,
            "hash": content_hash,
            "created_at": time.time(),
        }
        _hash_to_id[content_hash] = resume_id
        return resume_id, content_hash, False


def get_resume(resume_id: str):
    with _lock:
        return _resumes.get(resume_id)


def get_cached_analysis(content_hash: str):
    with _lock:
        return _analysis_cache.get(content_hash)


def set_cached_analysis(content_hash: str, analysis: dict):
    with _lock:
        _analysis_cache[content_hash] = analysis


def create_job(content_hash: str) -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "status": "processing",
            "progress": "queued",
            "data": None,
            "error": None,
            "hash": content_hash,
        }
    return job_id


def update_job(job_id: str, **fields):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def get_job(job_id: str):
    with _lock:
        return _jobs.get(job_id)


def get_full_profile(resume_id: str):
    """
    Single lookup used by every downstream agent (job match, skill gap,
    career chat, roadmap). Returns everything a module needs without
    re-extracting the file or re-calling the LLM:
        {resume_id, filename, text, analysis (or None), size}
    Returns None if resume_id is unknown (e.g. server restarted since
    upload - resumes live in memory only, see module docstring).
    """
    resume = get_resume(resume_id)
    if not resume:
        return None
    analysis = get_cached_analysis(resume["hash"])
    return {
        "resume_id": resume_id,
        "filename": resume["filename"],
        "text": resume["text"],
        "size": resume["size"],
        "analysis": analysis,
    }
