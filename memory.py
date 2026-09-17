"""
Memory Layer
------------
Stores lightweight user context (last resume, last question, last job
description, etc.) in a local JSON file so agents can give more
personalized answers across requests.

BUG FIX vs. original memory.py:
The original `save_memory()` did `json.dump(data, f)` directly, which
REPLACED the entire file with whatever small dict was passed in. That
meant uploading a resume would erase the previously saved chat history,
and every /chat call would erase the previously saved resume. This
version loads what's already there, merges the new keys in, and only
then writes the file back to disk.
"""

import json
import os
import threading

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "user_memory.json")

# A simple lock avoids two Flask worker threads corrupting the file if
# requests happen to land at the same instant.
_lock = threading.Lock()


def load_memory():
    """Return the full memory dict, or {} if nothing has been saved yet."""
    if not os.path.exists(MEMORY_FILE):
        return {}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupted or empty file shouldn't crash the app.
        return {}


def save_memory(update: dict):
    """
    Merge `update` into existing memory and persist it.
    Example: save_memory({"resume": text}) keeps any previously
    stored "last_question", "last_job", etc.
    """
    with _lock:
        current = load_memory()
        current.update(update)

        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)

        return current


def get_memory_value(key, default=None):
    return load_memory().get(key, default)


def delete_memory_key(key):
    """
    Remove a single key (e.g. 'current_resume_id') without wiping the
    whole memory file. Used by 'Remove Resume' so last_question/last_job
    etc. survive, but the active resume pointer is cleared.
    """
    with _lock:
        current = load_memory()
        if key in current:
            del current[key]
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2)
        return current


def clear_memory():
    with _lock:
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
