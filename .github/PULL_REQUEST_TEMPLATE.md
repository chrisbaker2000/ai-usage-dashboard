## Why

<!-- What problem this change solves. -->

## What changed

<!-- The behavior a user or a test can see. -->

## Checks

- [ ] `pytest --cov=usagedash --cov-branch`
- [ ] `ruff check` and `ruff format --check`
- [ ] `mypy`
- [ ] `npm test`
- [ ] `npm run format:check`
- [ ] `npm run test:browser` (for page changes)

Do not include a database, a config from a Mac, `open-dashboard.html`, a token, or a standing login home.
