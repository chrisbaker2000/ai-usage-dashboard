# Project guidance

This is a local macOS dashboard: Python/FastAPI, SQLite, launchd, and plain JavaScript/CSS.
The page has no frontend build or runtime npm dependencies. Node tools are for development only.

Read `CLAUDE.md` for the architecture and existing behavioral contracts, and
`docs/known-limitations.md` before changing deliberately accepted behavior.

## Development

```bash
uv sync
npm ci
.venv/bin/python -m pytest --cov=usagedash --cov-branch
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy
npm test
npm run format:check
npx playwright install chromium
npm run test:browser
```

Use synthetic identities and credentials in tests. Never read a live account to populate a test or
public screenshot. Browser fixtures live in `tests/browser/demo.js`; `npm run screenshot` captures
the actual dashboard with those fixtures into `docs/images/`.

## Changes

- Preserve loopback-only serving, signed and sealed page requests, and owner-only private files.
- Keep account IDs and token counts as decimal text at the API boundary.
- Use the existing safe DOM helpers; display external strings with `textContent`.
- Preserve keyboard focus across redraws and support both appearance modes and narrow screens.
- Add regression coverage for changed behavior. Keep UI styling in `src/usagedash/static/styles.css`.
- Pin GitHub Actions to full commit SHAs, with a version comment. Dependabot keeps them current.
- Work on a branch and open a PR. See `CONTRIBUTING.md` for checks and `SECURITY.md` for private reports.
