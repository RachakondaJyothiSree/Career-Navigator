"""
Text Extraction Layer
-----------------------
Responsible for ONE thing: turning an uploaded PDF/DOCX into plain
text, as fast as possible.

Why this is its own module (and not part of resume_agent.py anymore):
the previous version mixed "extract text" and "call the LLM" in the
same file, which made it impossible to cache/reuse extracted text
without also re-running (or accidentally re-triggering) the AI step.
Splitting them lets /resume/upload return instantly (no LLM call)
while /resume/analyze can reuse already-extracted text.

Performance note: files are read straight from the in-memory upload
stream (BytesIO) and are NEVER written to disk. The original app wrote
nothing to disk either, but this makes that explicit and guarantees
no duplicate files ever accumulate on the server - there's simply
nothing to de-duplicate.
"""

import io
from pypdf import PdfReader
from docx import Document

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


class ExtractionError(Exception):
    """Raised for any problem that should be surfaced to the user as a 4xx error."""
    pass


def validate_file(file_storage):
    """Cheap, fast checks BEFORE we spend any time reading the file."""
    filename = (file_storage.filename or "").lower()

    if "." not in filename:
        raise ExtractionError("File has no extension. Please upload a .pdf or .docx file.")

    ext = "." + filename.rsplit(".", 1)[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise ExtractionError("Unsupported file type. Please upload a .pdf or .docx file.")

    # Determine size without reading the whole stream into memory twice.
    file_storage.stream.seek(0, io.SEEK_END)
    size = file_storage.stream.tell()
    file_storage.stream.seek(0)

    if size == 0:
        raise ExtractionError("The uploaded file is empty.")

    if size > MAX_FILE_SIZE_BYTES:
        raise ExtractionError("File is too large. Please upload a resume under 5MB.")

    return ext, size


def extract_text(file_storage, ext: str) -> str:
    """Extract raw text. Assumes validate_file() has already run."""

    if ext == ".pdf":
        try:
            reader = PdfReader(file_storage)
            parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    parts.append(page_text)
            text = "\n".join(parts).strip()
        except Exception as e:
            raise ExtractionError(f"Could not read PDF: {e}")

    elif ext == ".docx":
        try:
            doc = Document(file_storage)
            text = "\n".join(p.text for p in doc.paragraphs).strip()
        except Exception as e:
            raise ExtractionError(f"Could not read DOCX: {e}")

    else:
        raise ExtractionError("Unsupported file type.")

    if not text:
        raise ExtractionError(
            "No readable text found in this file. It may be a scanned/image-only PDF."
        )

    return text
