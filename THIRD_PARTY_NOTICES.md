# Third-Party Notices

This project is released under the [MIT License](./LICENSE). It depends on the following
third-party open-source packages (see `requirements.txt` for exact pinned/required versions).
All of them are permissively licensed; none are modified or vendored into this repository -
they are used as regular dependencies.

| Package | License | Project |
| --- | --- | --- |
| groq | Apache-2.0 | https://github.com/groq/groq-python |
| python-dotenv | BSD-3-Clause | https://github.com/theskumar/python-dotenv |
| markitdown | MIT | https://github.com/microsoft/markitdown |

## Notes on how these dependencies are used

- `groq` is used for both text generation and vision (screenshot interpretation) LLM calls.
- `python-dotenv` loads local environment variables from `.env` during development (with a
  minimal built-in fallback loader in `app/main.py` if the package is not installed).
- `markitdown` is used unmodified in `tools/document_ingest.py` to convert uploaded PDF/Word/
  PowerPoint/Excel/CSV files to Markdown text. Only the adapter code that wires MarkItDown's
  output into this project's session/evidence model is original.

No third-party source code is copied or forked in this repository; all of the above are used
strictly as installed dependencies through `requirements.txt`.
