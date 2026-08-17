"""Local document ingestion (PDF / Word / PowerPoint / Excel -> Markdown).

This module adds a *new* capability on top of an existing MIT-licensed permissive
open-source dependency, Microsoft's MarkItDown (https://github.com/microsoft/markitdown).
It does not vendor or modify MarkItDown's source - it is used as-is, as a regular
pip dependency (see requirements.txt). Only the ingestion/adapter logic below,
which wires a converted document into this project's study-evidence pipeline,
is original code written for this project.

Why this exists: before this module, the only ways to feed learning evidence into
the app were screenshots (interpreted by a vision LLM) and URLs (routed through
tools/extractors/*.py). There was no path for a user to upload the original lecture
material itself - a PDF slide deck, a Word write-up, a PPTX, or an Excel practice
sheet. This module fills that gap.

Design notes:
- Fails safe: any conversion problem returns ``ok=False`` with a human-readable
  ``error`` instead of raising, so callers can show a fallback message instead of
  crashing a request (consistent with this project's existing fallback philosophy
  for source collection / LLM generation failures).
- Security: only a small extension allowlist is accepted, uploads are capped in
  size, and the temporary file written to disk for MarkItDown to read is deleted
  immediately after conversion (or on failure) - only the extracted text is kept.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB - generous for lecture slides/notes, small enough to stay responsive
MAX_STORED_CHARS = 20000  # keep evidence text bounded so one huge file can't dominate a run


def convert_document_to_markdown(*, file_bytes: bytes, filename: str, tmp_dir: Path) -> dict[str, Any]:
    """Convert an uploaded document to Markdown text via MarkItDown.

    Returns a dict with keys: ok, text, char_count, ext, error, converter.
    Never raises - conversion failures are reported through ``error``.
    """
    ext = Path(filename or "").suffix.lower()
    result: dict[str, Any] = {
        "ok": False,
        "text": "",
        "char_count": 0,
        "ext": ext,
        "error": None,
        "converter": "markitdown",
    }

    if ext not in ALLOWED_EXTENSIONS:
        result["error"] = f"unsupported file type: {ext or 'unknown'} (allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))})"
        return result
    if not file_bytes:
        result["error"] = "empty file"
        return result
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        result["error"] = f"file too large ({len(file_bytes)} bytes, limit is {MAX_UPLOAD_BYTES} bytes)"
        return result

    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        result["error"] = f"markitdown package is not installed: {exc}"
        return result

    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"upload_{uuid.uuid4().hex[:12]}{ext}"
    try:
        tmp_path.write_bytes(file_bytes)
        converter = MarkItDown(enable_plugins=False)
        conversion = converter.convert(str(tmp_path))
        text = (getattr(conversion, "text_content", "") or "").strip()
        if not text:
            result["error"] = "no extractable text found in document (it may be a scanned/image-only file)"
            return result
        result["ok"] = True
        result["char_count"] = len(text)
        result["text"] = text[:MAX_STORED_CHARS]
    except Exception as exc:  # noqa: BLE001 - any converter failure becomes a safe fallback, not a crash
        result["error"] = f"markitdown conversion failed: {exc}"
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
    return result
