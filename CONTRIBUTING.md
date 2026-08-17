# Contributing

Thanks for considering a contribution to AI Study Documentation Agent. This project is a
portfolio-stage prototype, but it follows normal open-source contribution practices so anyone
can read, run, and extend it.

## Before you start

- Read [README.md](./README.md) for what the project does and its current scope.
- Read [DEVELOPMENT.md](./DEVELOPMENT.md) for local setup, environment variables, and runtime
  data layout.
- Check open issues before starting new work, to avoid duplicate effort.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m app.main
```

## Making a change

1. Fork the repository and create a branch from `main` (`feat/...`, `fix/...`, `docs/...`).
2. Keep changes scoped - one logical change per pull request.
3. If you add a new third-party dependency, add it to `requirements.txt` with a version
   constraint and add a row to `THIRD_PARTY_NOTICES.md` with its license. Only permissive
   licenses (MIT, BSD, Apache-2.0, Unlicense, or similarly permissive) are accepted - do not
   add copyleft (GPL/AGPL) dependencies without discussing it in an issue first.
4. Do not commit real API keys, `.env`, private/paid course content, or screenshots/documents
   containing personal information.

## Commit and PR messages

Use a short, conventional prefix so history stays scannable: `feat:`, `fix:`, `docs:`,
`chore:`, `refactor:`, `test:`. Explain *why* the change is needed, not just what changed, in
the commit body when it's not obvious from the summary line alone.

Pull requests should describe: what problem the change solves, how it was tested (a manual
smoke test with the steps you ran is fine for this project's current stage), and any
follow-up work that's intentionally out of scope.

## AI-assisted contributions

This project is built with AI-assisted tooling as part of its normal workflow, and that's fine
for contributions too. What matters is that you understand and can explain the change you're
submitting, and that it's been tested. Please don't submit unreviewed, unexplained AI output.

## Code of conduct

Be respectful and assume good faith. Disagreements about design should stay focused on the
technical tradeoffs.
