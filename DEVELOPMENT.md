# Development Notes

This file contains local setup, runtime configuration, API route summary, and implementation notes for AI Study Documentation Agent.

The main README is kept as a portfolio overview. Development details are separated here so the repository can stay readable for reviewers while still remaining reproducible for local testing.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

python -m app.main
```

Open:

```text
http://127.0.0.1:7870
```

The local port is controlled by the `PORT` environment variable. Hugging Face Spaces can run the same app with the platform-provided port.

## Environment Variables

```text
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GROQ_VISION_CHUNK_SIZE=3
GROQ_VISION_MAX_TOKENS=1200
PORT=7870
```

Do not commit real API keys.

## Runtime Data

The app stores local runtime artifacts under the repository's `data/` directory during local use.

```text
data/
├── captures/      # uploaded screenshots
├── documents/     # Markdown text converted from uploaded PDF/Word/PPT/Excel files
├── tmp_uploads/   # short-lived temp files used during document conversion (deleted after use)
├── runs/          # generation run artifacts
├── notes.jsonl    # capture-based study notes
└── sessions.json  # session timeline, Q&A records, and document metadata
```

Only the converted Markdown text is kept in `data/documents/`; the original uploaded binary
(PDF/DOCX/PPTX/XLSX) is never persisted - it exists only as an in-memory upload and a
short-lived temp file in `data/tmp_uploads/` during conversion.

These files are runtime data, not source code. They should usually stay out of commits unless a small sample fixture is intentionally added.

## API Route Summary

### Notes and Search

```text
GET  /api/notes
POST /api/captures
POST /api/search
POST /api/blog
POST /api/direct-blog
POST /api/debug-collect-url
```

### LLM Health

```text
GET /api/health/llm
```

### Sessions

```text
GET    /api/sessions
POST   /api/sessions
GET    /api/sessions/{session_id}
GET    /api/sessions/{session_id}/captures
GET    /api/sessions/{session_id}/qa
POST   /api/sessions/{session_id}/captures
POST   /api/sessions/{session_id}/qa
POST   /api/sessions/{session_id}/ask
POST   /api/sessions/{session_id}/generate-article
DELETE /api/sessions/{session_id}/captures/{capture_id}
GET    /api/sessions/{session_id}/documents
POST   /api/sessions/{session_id}/documents
DELETE /api/sessions/{session_id}/documents/{document_id}
```

`POST /api/sessions/{session_id}/documents` accepts one or more files under the `document`
form field (PDF, DOCX, PPTX, XLSX, XLS, or CSV; 25MB limit per file). Each file is converted
to Markdown via MarkItDown and folded into the same evidence pool used by
`generate-article` alongside screenshots and Q&A logs. Unsupported file types or conversion
failures return a per-file `status: "error"` with a human-readable `error` message instead of
failing the whole request.

### Capture Files

```text
GET /captures/{filename}
```

## Implementation Notes

The app is currently implemented as a Python standard-library HTTP server using `BaseHTTPRequestHandler` and `ThreadingHTTPServer`.

The main implementation lives in `app/main.py`. The current file contains:

- single-page UI template
- capture upload handling
- local note and session persistence
- URL and YouTube source collection
- screenshot evidence extraction
- Q&A log handling
- tutor-style answer generation
- Medium-style draft generation
- article policy and validation checks
- fallback behavior for unavailable sources or providers

A future refactor can split the backend into smaller modules:

```text
app/
├── server.py
├── routes/
├── storage/
├── source_collection/
├── evidence/
├── generation/
└── validators/
```

That refactor is intentionally listed as future work rather than shown as the current project structure.

## Hugging Face Spaces

The repository can be deployed as a Docker Space.

Recommended notes:

- keep API keys in Space secrets
- do not commit `.env`
- set `PORT` according to the Space runtime
- keep generated capture data out of source control

## Security Notes

Do not commit:

- real `GROQ_API_KEY`
- `.env`
- private course content
- generated source packs containing paid/protected lecture material
- screenshots with personal information
