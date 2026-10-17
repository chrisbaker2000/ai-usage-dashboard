# Contributing

Issues and pull requests are welcome. The dashboard is a program for one person's Mac.

## Set up

```bash
git clone https://github.com/chrisbaker2000/ai-usage-dashboard.git
cd ai-usage-dashboard
uv sync
```

You need macOS, [uv](https://docs.astral.sh/uv/), and Python 3.13 or newer. Everyday use is in
[README.md](README.md).

## Checks

```bash
.venv/bin/python -m pytest --cov=usagedash --cov-branch
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy
npm test
prettier --check src/usagedash/static tests/js package.json
```

`npm test` runs the page scripts with `node --test`. The page has no build step and no runtime
dependencies. Formatting for the page and its tests uses Prettier 3 with `.prettierrc.json`.

After a Python change, run `.venv/bin/usagedash agent install` so a loaded agent picks it up. After
a static-file change, run `.venv/bin/usagedash open` again.

## What not to commit

Do not commit a database, a config file from a Mac, `open-dashboard.html`, an access token, a
standing login home, or a `HANDOFF.md` note. Those can name an account or hold a credential. Tests
use synthetic identities such as `user-a@example.test`.

## Pull requests

Open a pull request against `main` and say why the change is needed. The technical contract for the
code is [CLAUDE.md](CLAUDE.md). Behavior that is deliberate and should stay is in
[docs/known-limitations.md](docs/known-limitations.md).

The same checks run on a macOS runner for each pull request.
