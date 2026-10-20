# Contributing

Issues and pull requests are welcome. The dashboard is a program for one person's Mac.

## Set up

```bash
git clone https://github.com/chrisbaker2000/ai-usage-dashboard.git
cd ai-usage-dashboard
uv sync
npm ci
npx playwright install chromium
```

You need macOS, [uv](https://docs.astral.sh/uv/), Python 3.13 or newer, and Node.js 22 or newer. Everyday use is in
[README.md](README.md).

## Checks

```bash
.venv/bin/python -m pytest --cov=usagedash --cov-branch
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy
npm test
npm run format:check
npm run test:browser
```

`npm test` runs the page scripts with `node --test`. The page has no build step and no runtime
dependencies. Formatting for the page and its tests uses the locked Prettier version with `.prettierrc.json`.
Playwright tests cover desktop and mobile layouts in light and dark mode, keyboard navigation,
renaming, filtering, and refresh. The browser tests serve static files on a separate loopback port
and supply synthetic accounts through a signed demo API. They do not start the collector.

Run `npm run screenshot` to regenerate `docs/images/dashboard-dark.png` and
`docs/images/dashboard-light.png` from that same demo, plus the 1280×640 social-preview card.
The social image must also be uploaded in GitHub's repository settings; see [Maintaining](docs/maintaining.md).
Never use a live account screenshot in the README or social preview.

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

Python and page unit checks run on a macOS runner for each pull request; browser tests run on Linux.
Sign-in tests resolve `claude`,
`codex`, and `grok` on `PATH` before they mock the process. A machine that has those commands
installed already satisfies that. The runner has none, so the workflow puts an inert stand-in for
each name on `PATH`. The stand-in is not used as a real CLI.

The maintainer reviews external workflow runs before they start. Changes to `main` require a pull
request and passing `test` and `browser` checks. Dependency updates arrive through Dependabot.
