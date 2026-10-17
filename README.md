# AI usage dashboard

[![tests](https://github.com/chrisbaker2000/ai-usage-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/chrisbaker2000/ai-usage-dashboard/actions/workflows/ci.yml)

A local Mac dashboard for the Claude, Codex, and Grok subscription accounts you sign in on this
computer. It shows quota windows and reset times, token use from this Mac's session files, and what
those tokens would have cost at the vendors' published short-prompt API rates.

The page is served on this Mac only, at <http://127.0.0.1:8787>, by a launchd agent that keeps
collecting while you are logged in. It is not a shared or hosted service. Each person runs their own
copy and connects their own accounts. Nothing in this repository is an account, a token, or a
reading. The dashboard does not send readings or credentials to the person who published it.

## Install

You need:

- macOS. The agent is a launchd service, and Claude's working login is read from the macOS keychain.
- [uv](https://docs.astral.sh/uv/), and Python 3.13 or newer. `uv sync` creates `.venv` from the
  lockfile. If it cannot find Python 3.13, install that version and run it again.
- The vendor CLIs for any standing logins you add, and the Grok CLI for working-login quota reads.
  Check that each command you need (`claude`, `codex`, or `grok`) runs in this terminal. The page
  shows all three products even when a CLI is absent.

```bash
git clone https://github.com/chrisbaker2000/ai-usage-dashboard.git
cd ai-usage-dashboard
uv sync
.venv/bin/usagedash agent install
.venv/bin/usagedash open
```

Run `agent install` from a normal terminal, not a restricted environment. The agent records this
shell's home directory, the CLIs it found, and a checked PATH. It starts now and again at each
macOS login. Install
warnings about directories left off PATH are expected when those directories are missing or another
account could change them. The install succeeded when the command exits 0.

`open` writes a private page and opens it in your browser. That file is the dashboard, and it is
how this browser is allowed to read the server. If it reports that nothing answered, the agent is
not up yet: run
`.venv/bin/usagedash agent status`, then `agent install` again, then `open` again. A stopped agent
does not write a page.

The accounts already signed in as your working Claude (`~/.claude`), Codex (`~/.codex`), and Grok
(`~/.grok`) logins are tracked with no extra step. The header says "Waiting for the first collection"
until the first cycle finishes. Collection then repeats about every five minutes. **Refresh now**
asks the running agent to collect again.

## Read the page

**Quota** is one card per account. A window is **OK**, **Near limit** (75%), **Critical** (90%), or
**Limit reached** (100%). Those thresholds are the defaults and can be changed in the config file.
The chip on a provider is the advice for that product:

| Chip    | Meaning                                                                                                                                                                                                 |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stay    | Every blocking window on the account in use is under the critical threshold.                                                                                                                            |
| Switch  | Another account that is being read is under the critical threshold.                                                                                                                                     |
| Wait    | The account in use is at or above the critical threshold, and no other account is known to be under it. The advice line names a reset only when that account is out of quota and a reset time is known. |
| Verify  | Another account might be under the critical threshold, and it is not being read right now.                                                                                                              |
| No data | No working account is signed in, no blocking limit has been read, or the account in use has no current reading.                                                                                         |

**Rename** on a card changes the label the page shows. It does not change the login.

An empty card, **Account not seen yet**, means the page is waiting for another account of that
product. The default is two accounts per product. See [Add an account](#add-an-account).

**Token usage** is this Mac's session files for the last 24 hours, 7 days, or 30 days. The chart, the
account table, and **By model** are the same stored hours. The count menu chooses all tokens, all
tokens except cache reads, or output only. Usage the dashboard cannot match to a signed-in account,
including use from before it knew who was logged in and use while nobody was logged in, is listed as
**Unattributed (before tracking)**.

**Retail API cost** prices those tokens at each vendor's published pay-as-you-go rate for a short
prompt, rounded to the nearest dollar. The all-accounts row is the sum of those rounded figures. A
model with no published rate for a component that was used stays in the table, under **Unpriced**,
and is left out of the dollar total. The figure is not the subscription charge. The rate assumptions
are in [docs/known-limitations.md](docs/known-limitations.md).

**Data sources** lists each login the collector asked, and any waiting state or error. A source that
is logged out, and has nothing to report yet, is not a failure.

Open the dashboard again later with `.venv/bin/usagedash open`, from this repository. Use that
command after an update. A browser reload keeps the previous copy of the page, and the address
`localhost` can miss the server. The page talks only to `127.0.0.1`.

## Add an account

A working login is the account currently signed in to `claude`, `codex`, or `grok` in your normal
home directory. An extra account is a **standing login**: its own home under the dashboard's data
directory, read on the same schedule, without switching the account your terminal uses.

Run one login at a time, in a terminal you can see. The vendor CLI waits there until its own sign-in
finishes. The dashboard does not ask for a password of its own.

```bash
.venv/bin/usagedash login claude --name work
.venv/bin/usagedash login codex --name personal
.venv/bin/usagedash login grok --name work
```

`--name` identifies the standing home. Use 1–32 characters from `a-z`, `0-9`, and `-`, starting with
a letter or a digit. Omit it and the name is `standing`, so the home is `homes/claude-standing` (or
`codex-` or `grok-`). Reusing a name signs that home in again. A new name creates a separate home.
The vendor sign-in decides which account that home tracks, so two homes signed in to the same
account still produce one card. Each home is
`~/Library/Application Support/ai-usage-dashboard/homes/<tool>-<name>`.

When the sign-in succeeds, the login is written to `config.toml` and an agent that serves this same
data directory is restarted. The account appears after the next collection. A failed sign-in adds no
login. If the command tells you to run `usagedash agent install`, do that, then `open` again. The
working login is left as it is.

Claude Code keeps each sign-in in the keychain under your macOS account name. The dashboard asks for
that item by that name, and runs every Claude sign-in and read with it, whatever `USER` your shell
has. A Claude standing login that shows "Claude Code reported no plan limits for this login" was
probably signed in under another name by an earlier version: sign that home in again with the same
`--name`.

The dashboard reads a working credential and does not copy it or refresh it. Refreshing a working
credential from outside the vendor CLI can sign that CLI out. A standing login lets the vendor CLI
refresh its own isolated session. Quota reads do not spend model quota.

To show more than one empty card per product, set `expected_accounts_per_provider` in the config
file. The setting only draws placeholders, from 1 to 20. Accounts you have already connected stay on
the page either way.

## Everyday commands

Run these from the repository directory.

| Command                                   | What it does                                                             |
| ----------------------------------------- | ------------------------------------------------------------------------ |
| `.venv/bin/usagedash open`                | Open the private page in your browser.                                   |
| `.venv/bin/usagedash status`              | Print the latest stored quota readings.                                  |
| `.venv/bin/usagedash collect`             | Run one collection now and print the outcome.                            |
| `.venv/bin/usagedash agent status`        | Say whether launchd has the agent loaded.                                |
| `.venv/bin/usagedash agent install`       | Install or replace the agent and start it.                               |
| `.venv/bin/usagedash agent uninstall`     | Stop the agent and remove its property list. Data and logs stay.         |
| `.venv/bin/usagedash serve`               | Run the server in the foreground. The agent is the normal way to run it. |
| `.venv/bin/usagedash login <tool> --name` | Sign in an extra account. `<tool>` is `claude`, `codex`, or `grok`.      |

`collect` exits 0 when every source is fine, including a source that is logged out and has nothing
to report. It exits 1 when a source would show as waiting or as an error. Ctrl-C exits 130.

## What the numbers mean

Quota is the account's subscription window and includes use on every device. Token totals are the
session files on this Mac. Claude and Grok files do not name the account, so those tokens follow the
login timeline recorded here. A Codex home copied whole into another configured home is counted in
both.

A reset time the vendor reports as already earlier than the reading clears none of the measured
usage, so that window keeps its usage and has no known reset time. A reset still ahead of the
reading is what the pace and the **Wait** advice use, and the window is treated as empty once that
reset has passed. An account that is not being read can be marked **Verify**. It is not offered as
a switch, because it may have been used since the last reading.

Grok can report 0% used when the billing response leaves the percentage out and still states a
complete period. The card says that 0% was not a stated value. A Grok working token that has expired,
or has five minutes or less left, stays unread until you use Grok again. A standing Grok login keeps
its own session.

Codex code-review and per-feature pools are shown and do not, by themselves, mark the account out of
quota. A limit Codex reports as reached does.

The first collection reads session files modified within the last 35 days (`backfill_days`). That
cutoff stays put, so a file modified later is still read. Token rows are kept for 400 days. Quota
history is kept for 90 days.

## Configuration

No config file is required. `usagedash login` creates
`~/Library/Application Support/ai-usage-dashboard/config.toml` when it registers the first extra
account. Defaults, then that file, then `USAGEDASH_*` environment variables: an environment value
wins for the settings that have one. The variables are listed in [`.env.example`](.env.example).
None of them are secrets.

```toml
port = 8787
poll_interval_seconds = 300 # minimum 60, maximum 86400
backfill_days = 35          # 0 through 3650; the first collection's file-age cutoff
warn_percent = 75
critical_percent = 90       # 0 < warn_percent < critical_percent <= 100
expected_accounts_per_provider = 2

[binaries]
codex = "/opt/homebrew/bin/codex"

[[claude]]
config_dir = "~/Library/Application Support/ai-usage-dashboard/homes/claude-work"
mode = "cli"
```

Paths in the file are absolute or start with `~/`. `port`, `poll_interval_seconds`, and the data
directory can also be set with `USAGEDASH_PORT`, `USAGEDASH_POLL_INTERVAL_SECONDS`, and
`USAGEDASH_DATA_DIR`. `backfill_days`, the two percentages, and
`expected_accounts_per_provider` are file settings only.

After editing the config file or a `USAGEDASH_*` override, run `.venv/bin/usagedash agent install`
and then `.venv/bin/usagedash open`. The running agent keeps the settings it loaded at startup,
including the port.

`[[claude]]`, `[[codex]]`, and `[[grok]]` add logins. Prefer `usagedash login` to writing those
tables by hand. If a file already lists a product's logins on one line, the command cannot append a
table and prints the entry to add to that list. Claude tables use `config_dir`. Codex and Grok
tables use `home`. `mode` is `cli` for a standing login, which lets the vendor CLI refresh its own
session. In `direct` mode the dashboard reads the home's current credential and does not refresh it.
Grok still launches its CLI for quota reads while that access token has more than five minutes left.
The three default homes are in direct mode.

`[binaries]` overrides where `claude`, `codex`, and `grok` are found. The environment variables
`USAGEDASH_CLAUDE_BINARY`, `USAGEDASH_CODEX_BINARY`, and `USAGEDASH_GROK_BINARY` override the file.
After you move a CLI or change one of these, run `agent install` again from a terminal that can see
the new binary. The agent keeps the paths it recorded at install time. A recorded file that has
been removed is looked up by its file name on the agent's checked PATH. When that name is the
release rather than the command, as with Claude's version file and Grok's download, the read uses
`claude`, `codex`, or `grok` on that same PATH. The replacement is checked the same way as a path
that is still present. Run `agent install` anyway so the recorded path and the agent's PATH match
the binary you intend.

The server listens on `127.0.0.1` only. There is no setting to bind another address.

## Files on this Mac

| Path                                                                   | What it holds                                                                 |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `~/Library/Application Support/ai-usage-dashboard/`                    | Database `usage.db`, `config.toml`, `homes/`, the page, and the access token. |
| `~/Library/Application Support/ai-usage-dashboard/open-dashboard.html` | The private page `usagedash open` writes. It contains the access token.       |
| `~/Library/Logs/ai-usage-dashboard/`                                   | The agent log, rotated at 1 MB with three backups.                            |
| `~/Library/LaunchAgents/com.<user>.ai-usage-dashboard.plist`           | The launchd agent. `<user>` is the macOS account name.                        |

The page file and the token are owner-only. Do not copy `open-dashboard.html` to another machine or
another account. Deleting `access-token`, then running `agent install` and `open`, issues a new
token. Programs running as you can read the token and the database.

## Update or remove

From the repository directory, after pulling a new version:

```bash
git pull
uv sync
.venv/bin/usagedash agent install
.venv/bin/usagedash open
```

`agent install` replaces the running agent. `open` rewrites the page so the browser gets the new
copy.

To stop collection:

```bash
.venv/bin/usagedash agent uninstall
```

That removes the launchd service and keeps the data directory and the logs. Installing again
continues the same database.

## Security

Readings, labels, and the database stay under
`~/Library/Application Support/ai-usage-dashboard` on this Mac. This project has no account of its
own and no server that receives them.

The server accepts connections on `127.0.0.1` only. The page proves each request with the access
token and checks the dashboard's signature on each answer. The token stays in the private page file.
It is not put in the browser's storage, a cookie, a URL, or a command argument, and the page does
not send it on the wire. Account credentials stay in the vendor CLIs. The dashboard does not copy a
working credential into its own store.

A config file or data directory that another macOS account could replace is refused. A standing home
that another account could change is refused, because the vendor CLI reads its configuration from
there.

The design of the page token, and the rate assumptions behind the retail cost, are recorded in
[docs/known-limitations.md](docs/known-limitations.md). Notes for people changing the code are in
[CLAUDE.md](CLAUDE.md).

## Development

```bash
uv sync
.venv/bin/python -m pytest --cov=usagedash --cov-branch
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy
npm test
prettier --check src/usagedash/static tests/js package.json
```

`npm test` runs the page scripts with `node --test`. The page has no build step and no runtime
dependencies. After a Python change, run `agent install` so the loaded agent picks it up. After a
static-file change, run `open` again.

The automated tests do not complete a live vendor sign-in. `usagedash login` is still the way to add
an account.

## License

[MIT](LICENSE). To report a vulnerability, see [SECURITY.md](SECURITY.md). To propose a change, see
[CONTRIBUTING.md](CONTRIBUTING.md).
