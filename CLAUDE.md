# AI usage dashboard

Local dashboard for quota windows, reset times, and token usage for the Claude, Codex, and Grok
subscription accounts signed in on this Mac. Python 3.13 package `usagedash` + SQLite + a static page,
run by a launchd agent on `127.0.0.1:8787`. Install and everyday use: `README.md`. Accepted behavior:
`docs/known-limitations.md`.

## Commands

```bash
uv sync
.venv/bin/python -m pytest --cov=usagedash --cov-branch
.venv/bin/ruff check src tests && .venv/bin/ruff format src tests
.venv/bin/mypy                                             # strict
npm test                                                   # page scripts (node --test, no dependencies)
prettier --write src/usagedash/static tests/js package.json
.venv/bin/usagedash collect
.venv/bin/usagedash status
.venv/bin/usagedash serve
.venv/bin/usagedash open
.venv/bin/usagedash login claude --name work
.venv/bin/usagedash agent install
.venv/bin/usagedash agent uninstall
.venv/bin/usagedash agent status
```

After changing Python code, restart the running agent: `.venv/bin/usagedash agent install`.
After changing static files, run `.venv/bin/usagedash open` again: the page file it writes carries the
stylesheet and the modules inline, so a browser reload alone keeps the old copies.

## Layout

| Module                                 | Responsibility                                                                                      |
| -------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `config.py`                            | Settings from defaults ← TOML ← env. Default homes are always tracked; `[[tool]]` tables add logins |
| `providers/{claude,codex,grok}.py`     | Direct-mode sources: read the CLI's live credential, never modify it; Grok usage via its CLI reader |
| `providers/{claude,codex,grok}_cli.py` | CLI-mode sources for standing logins: ask the vendor CLI over its stdio protocol                    |
| `rpc.py`                               | JSON-lines child-process client used by CLI mode                                                    |
| `scanners/`                            | Token usage from local session files (incremental, idempotent per file)                             |
| `collector.py`                         | One cycle: read quota → record who is logged in → scan tokens. Per-source isolation, 429 backoff    |
| `projection.py`                        | Pure logic: window state, reset roll-forward, pace, stay/switch/wait advice                         |
| `tokens.py`                            | Hourly rows → chart buckets in the viewer's time zone                                               |
| `pricing.py`                           | Published short-prompt API rates and the microdollar cost of a model row                            |
| `api.py`                               | FastAPI app, security checks, background collection loop                                            |
| `access.py`                            | The access token, challenge-bound request and answer signatures, the body seal, and the page file   |
| `validation.py`                        | Shared strict pydantic base, field types, and safe provider-schema error translation                |
| `private.py`                           | The one definition of owner-only directories and files; everything in the data directory uses it    |
| `errors.py`                            | Typed errors. Anything a user can cause surfaces as a `UsageDashError`, never a raw exception       |
| `standing.py`, `agent.py`, `cli.py`    | Standing-login setup, launchd agent, command line                                                   |
| `executables.py`                       | Shared no-open executable trust checks and child PATH/cwd resolution                                |
| `static/`                              | Vanilla JS page, ES modules, no build step; render-only, all decisions are made server-side         |
| `static/{format,access,latest}.js`     | The page's pure logic (chart measures included), unit tested in `tests/js/`; `app.js` renders       |
| `static/{failures,rename}.js`          | Also pure and unit tested: one failure per source; the rename as one operation at a time            |

## Decisions (do not re-litigate without new information)

- **Never refresh or copy a CLI's credential.** Refresh tokens rotate; refreshing from outside logs the
  CLI out. Direct mode reads the live access token read-only; CLI mode lets the vendor CLI authenticate.
- **Standing logins use CLI mode**, each in its own home under the data directory. All three vendor
  CLIs expose a no-model-call read: Claude `get_usage` control request, Codex `account/rateLimits/read`,
  Grok `_x.ai/billing` (verified 2026-10-03 on claude 2.1.288, codex 0.160.0, grok 1.0.46).
- **A Claude answer without its `limits` list is read from the fixed fields and `model_scoped`**
  (`claude._windows_from_legacy_fields`). Both paths use `claude._weekly_scoped`. A `model_scoped`
  entry is dropped only when a fixed field states the same limit; entries that share a key are all
  kept and suffixed. An answer that says limits are available and sends none is a
  `ProviderRequestError`, not a schema error.
- **Working logins stay in direct mode.** CLI mode in `~/.claude` runs every user hook on each poll.
  Grok's direct source takes usage only from its CLI reader (`grok.GrokSource`), and only while the
  working token passes `validation.AccessToken` and is unexpired (`grok._token_is_valid`), so that
  CLI has nothing to refresh. An expired token is `CredentialExpiredError` until Grok is next used.
- **One collection at a time, across processes.** `collector.lock` in the data directory is held with
  `flock` for the whole cycle; the agent and a manual `usagedash collect` would otherwise both add the
  same tokens, because scanners read positions and store usage in separate transactions.
- **One initialization at a time, across processes.** `Database.initialize` holds an exclusive `flock`
  on the data directory itself (no lock file) until the journal mode is set. `PRAGMA journal_mode = WAL`
  on a new database upgrades a read lock, and SQLite does not apply the busy timeout to that: with
  another starter inside its schema transaction it failed at once with "database is locked", in about
  one of twenty concurrent first starts. Do not move the pragma out from under the lock.
- **A reading older than the account's newest stored reading is dropped whole**, windows and facts.
- **Advice makes definite claims only from current readings.** An account that is not being read may
  have been used since, so it can earn `VERIFY` but never `SWITCH`, and an active account without a
  current reading gets `UNKNOWN`, not a headroom figure (`AccountView.is_confirmed`).
- **`usagedash login` registers after a successful sign-in**, never before, so a failed sign-in
  adds no registered login and does not restart the agent. The prepared home and lock file may
  remain; a retry still ends with the login in the config. The running agent is
  restarted only when it serves this same data directory and config file (`cli._same_dashboard`,
  both through `paths.same_directory`, so another spelling of one directory matches). The port is
  not part of that match and stays the one the agent was installed with. The look at what is
  installed and the restart are one step under the installation lock (`cli._login` holds
  `agent.holding_installation_lock` and calls `agent.kickstart_if_loaded`, which does not take the
  lock: a second flock on the same file can deadlock). An agent installed for another data
  directory or config file is left as it is; the command says to run `usagedash agent install`
  again before that agent will read the login. A service launchd still lists after its property
  list has gone is reported, not restarted: there is nothing to compare it with, and it may be any
  dashboard.
  A directory another account can replace above a standing home, the config file, or the data
  directory is refused (`private.ensure_no_ancestor_lets_another_account_replace`), and so is a standing
  home that already exists and another account could write to, or holding an entry another account
  could change, left as it is (`standing.prepare_home`, `private.what_lets_another_account_change`):
  the vendor CLI reads its configuration, hooks included, from there. `usagedash open`
  asks `GET /healthz` before it writes the page. The walk, that probe's limits, and the checked
  config read are in Gotchas. Registration makes a file that check accepts owner-only
  (`standing._config_content`).
  Registration (`standing.register`) holds `config.toml.lock`, reloads, and replaces the file whole.
  An interrupt removes the new file and leaves the config as it was. The replacement must contain the
  login. A one-line list of logins cannot take a `[[tool]]` table (`standing.ensure_registrable`).
  A home already listed in direct mode is refused (`standing.is_registered`). A listed path is this
  home when `standing._stored_as_this_home` says so (`paths.directory_identity`, then
  `paths.same_directory`, including a missing component or a file). The hand-added entry is split on
  line feeds only, not with `str.splitlines`. `standing.sign_in_and_register` holds that lock from a
  fresh read through `_register_locked`. A missing file holds nothing: the reload then comes from the
  environment alone (`load_settings(file_may_be_absent=True)`), so a login or a `[binaries]` override
  from a file since removed is not acted on. A lookup that cannot examine the file is a ConfigError
  (`standing._config_file_is_present`, `os.lstat`, not `os.path.lexists`).
  The check under the lock runs before the vendor CLI. Claude's keychain item is the listed spelling
  (`standing.login_environment`). A listed `account_file` is found where polling finds it and must
  be that home's `.claude.json` as stored (`standing._refuse_claude_account_file_the_sign_in_will_not_write`).
  A listed spelling of the home, for any tool, and the directory of a listed `account_file` are
  refused when another account could make them lead elsewhere, by an entry on the way it could
  replace or retarget, and when they pass through a link this account owns; the refusal names the
  path to list (`private.ensure_no_other_account_can_redirect`, from `standing.is_registered`). A
  link as the listed `account_file`, wherever it is, is refused.
- **Reauthentication preserves verified vendor output links** (`standing._checked_vendor_output_link`).
  The link must belong to this account. Claude's `debug/latest` may resolve only to a regular file
  directly inside its private debug directory; a safely missing log is accepted. The complete link
  chain, target ancestry, ownership and ACLs are checked. Codex's `tmp/arg0/*/apply_patch`, `applypatch`
  and `codex-execve-wrapper` may target only checked executable regular files; execute-only binaries
  use the shared no-open checks. All other links, including credential/configuration redirects,
  remain refused. Unsafe or unexaminable targets stop sign-in before config changes. These layouts
  were verified through metadata only; interactive isolated sign-in is still unverified.
- **A source and a scan store one resolved directory** (`paths.directory_identity`). Codex and Grok
  keep that path as the home, and each scanner keeps it as the directory it reads. The stored name
  is the directory entry's, so the next run configured as `home/../home`, through a symbolic link,
  or with only the letter case changed keeps the source id, the cursors, and the login. A second
  listing that reaches that home through a missing component or a file is refused. Claude's source
  id is the resolved `config_dir`; the stored config is not rewritten, so
  `ClaudeSourceConfig.is_default_home` still keeps the default `~/.claude` from being passed as
  `CLAUDE_CONFIG_DIR`.
- **A vendor CLI child inherits none of its own vendor's variables** (`rpc.child_environment`):
  `CLAUDE*`/`ANTHROPIC_*`, `CODEX_*`/`OPENAI_*`, `GROK_*`/`XAI_*`, dropped by prefix for reads and for
  the interactive login alike. Each CLI takes a credential or another home from such a variable ahead
  of the login in the home it is pointed at, and the identity is read from that home.
- **A Claude keychain item is named by service and account** (`credentials.KeychainItem`,
  `claude.keychain_account`). Claude Code reads its login with
  `security find-generic-password -a "$USER" -s <service>` (2.1.294, 2.1.295), and `security` given
  the service alone returns whichever matching item it finds first. The account is the user database's name, never `USER`, and every
  Claude child, reads and `usagedash login` alike, runs with `USER` set to it
  (`claude_cli.cli_environment`). On 2026-10-08 a sign-in from a shell with `USER=unknown` filed a
  standing login under `acct="unknown"`, where the agent's reads never looked, and left a second
  `Claude Code-credentials` item that the service-only lookup then returned in place of the real one.
  Do not drop `-a` or let an inherited `USER` reach a Claude child.
- **`agent install` prepares before it stops anything, and puts back what it replaced.** The plist is
  written first. A failed or interrupted stop or start restores it (`agent._put_back`). If launchd was
  not asked (`agent._Replaced.launchd_was_asked`), only the file is touched. If it was, whatever it
  lists under the label is removed and waited for, and only then is the previous agent started. The
  interrupt is raised again afterwards.
  Install and uninstall share `agent._stop`, which checks a nonzero `bootout` result immediately: an absent service permits installation
  to continue; a loaded service or failed inspection raises into rollback. Only an accepted stop
  enters the unload polling loop. This avoids repeated 45-second waits after a rejected request.
- **`agent uninstall` waits out `bootout`, then deletes the plist** (`agent.uninstall`). A stop that
  does not finish deletes nothing. `AgentStillLoadedError` means launchd still lists the service
  after the unload deadline, so it is left running. A failed stop whose outcome is unknown uses the
  same reconciliation as an interrupt: inspect the loaded state and restart only a trusted plist. A plist that cannot then be deleted is started again only from the
  owner-only copy read first (`agent._plist_safe_to_start`, then `agent._start_again`). A copy another
  account could have changed, or a restore that cannot be written, stays stopped. An interrupt uses
  that same choice (`agent._restore_interrupted_uninstall`). A missing plist is not proof launchd has
  dropped the service. `agent._path_exists` uses `lstat`: missing is absent, a dangling link is
  present, and any other failure is an `AgentError`. `Path.exists` is not that check. The interrupt
  still exits 130. `agent status` and `agent uninstall` do not read settings first
  (`cli._manage_installed_agent`). Uninstall names the data directory from the property list it
  removed, read under its lock (`agent.uninstall` returns that environment), never from the settings.
- **One command at a time changes the agent** (`agent._one_change_at_a_time`): install, uninstall, and
  the restart after a login hold an owner-only lock file beside the plist from before they look at
  what is installed until after anything has been put back. Without it a failed install undid a
  concurrent install or uninstall. A new command that changes the installation must take it too.
- **The property list is written as an owner-only file, whatever the umask** (`agent._write_plist`,
  through `open_private_file`), and so is one put back. launchd refuses mode 666 (exit 5 on macOS
  27.0.1) and accepts 664, 644, and 600. A missing `~/Library/LaunchAgents` is created by
  `ensure_private_directory` (`agent._prepare_plist_directory`), after the ancestor walk
  (`private.ensure_no_ancestor_lets_another_account_replace`). A level that already exists is never
  changed, and is refused when another account can change it (`private.ensure_only_owner_can_change`;
  `everyone deny delete` passes).
- **Validate every vendor process at launch**, including direct `collect` and manual `serve`:
  `executables.checked_command` resolves bare and relative commands against the child's actual PATH
  and working directory, applies `executables.canonical_executable`, then passes the canonical target
  to process creation. A recorded path that is no longer on disk is looked up by its file name on
  that checked PATH. When that name is absent, a quota read or `usagedash login` looks up the vendor
  command (`claude`, `codex`, or `grok`) on the same PATH: Claude's version file and Grok's download
  are named for the release, not the command. The file name still wins when it is present. A
  replacement is accepted only when the same canonical check passes. The replacement warning
  is recorded only after that acceptance (`executables._note_accepted_pin_replacement`). The recorded path wins
  while any entry is there, including a dangling link. An omitted child-PATH directory is logged
  once per process (`executables._leave_out_of_child_path`). `rpc.JsonLineProcess` applies it for each collection child and reports unsafe
  executables as `ProviderRequestError`, preserving source isolation. `standing.run_login` applies it
  after the locked settings reload, so a changed binary override cannot bypass the check. Missing bare
  commands fail before spawning. The shared check inspects ownership, ancestry, mode bits, and ACLs
  without opening binaries; agent pinning uses the same check. `executables.checked_link_target` checks
  the original selection ancestry and every link owner before resolving, including aliases to trusted
  dispatchers such as env. Agent helpers share that validator. Relative and empty PATH components
  are interpreted from the child working directory, and an absent PATH uses the system default.
  Resolve each original entry directly; reserializing expanded directories into PATH would turn
  literal colons in a working-directory name into unintended search locations.
- **Check every vendor child's PATH**, including shebang interpreters and commands launched by a trusted
  CLI. `executables.execution_environment` resolves original entries from the actual child cwd,
  canonicalizes and inspects directories, regular files, and retained links, and omits unsafe or
  colon-containing entries with reasons. An empty result uses the non-directory `/dev/null`, since an
  empty PATH string would implicitly search cwd. Filter before resolving a bare vendor name, and use
  that same environment for lookup and process creation: a rejected directory must neither supply
  the CLI nor prevent selection of a safe alternate. `rpc.JsonLineProcess` checks immediately before each
  spawn; `standing.run_login` checks the same way and prepares required macOS helpers under the dashboard
  data directory. Wiring passes the dashboard data directory to each CLI source; RPC prepares helpers
  only when that source actually launches a child, using the resolved source home as cwd. Preparation
  errors become per-source failures, leaving other sources and serve available; logged-out direct Grok
  does not prepare helpers at all. Helper files never go into vendor homes. Shared
  `executables.with_required_helpers` validates each required executable and the complete fallback
  directory; agent installation delegates to it too. Real-process tests cover unsafe node directories
  and link targets, trusted alternates, no safe interpreter, and helper fallback in both launch modes.
- **The installed property list is trusted input, so it is read only if no other account can have
  changed it** (`agent._installed_plist` through `private.read_file_only_owner_can_change`): a failed
  install puts it back and starts it, so a link to elsewhere or a file another account can write
  could run any program as its owner. It is opened without following a link, checked, and read
  through one descriptor, before anything is changed.
- **The log directory's ancestors are checked, not only the directory** (`agent.prepare_log_directory`,
  used by `agent install` and `serve`): an account that can change any directory above the log
  directory, including one above the home, could move the private log directory aside and put its own
  in its place. The same full ancestry walk runs before the directory is created. A level that exists
  is refused like the plist's and never changed.
- **The agent runs `python -P -m usagedash serve`.** Before any installation write or launchd call,
  `agent._checked_python_executable` checks the interpreter's retained spelling, link hops, final
  target, and ancestry, and requires an executable regular target. Executable validation alone permits trusted user-owned directory links
  (`private.ensure_no_ancestor_lets_another_account_replace`, `follow_owned_links=True`), checking
  their target and container ancestry; private-data paths keep the strict default. This supports
  Homebrew Python's directory aliases. Keep the virtual-environment spelling: resolving it to the base interpreter
  changes Python's environment selection. Its working directory is the data directory, and
  with `-m` Python looks there first: a `usagedash.py` left in it while others could write to it would
  be run in place of the dashboard. `-P` keeps the working directory off the import path.
- **A property list is only used if it reads back as what was written** (`agent._as_property_list`).
  `plistlib` writes U+FFFE and U+FFFF, which no XML parser accepts, and a carriage return, which every
  parser reads back as a line feed, so a successful write proves nothing about what launchd will read.
- **Nothing in the log directory is opened through a symbolic link.** The server's log is a
  `cli.PrivateRotatingFileHandler`, which opens it with `open_private_file` at start and after every
  rotation; `agent install` does the same for the two files launchd opens itself. Making the directory
  owner-only does not remove a link put there earlier. Two processes can share the log: each record
  reopens after another process's rotation. `usagedash.log.lock` stays held from the inode check
  through the flush, with a depth counter for nested rollover calls, so a second process cannot
  rotate between the check and the write. Closing the handler holds its thread lock across both
  stream and rotation-descriptor cleanup, including errors, so a concurrent record cannot acquire
  a descriptor about to close.
- **`serve` runs `uvicorn.Server` itself, not `uvicorn.run`**, which swallows the interrupt Uvicorn
  raises again after a clean shutdown on Ctrl-C. A server that cannot start exits by itself
  (`SystemExit`) after logging why.
- **FastAPI's native telemetry is off, every switch** (`api.TELEMETRY_OFF`). FastAPI 0.142 records
  requests through whatever OpenTelemetry providers the process has and adds exporters from inherited
  `OTEL_*` variables, all on by default. `tests/test_api.py` fails if a later FastAPI adds a switch
  the constant does not name.
- **Loopback only, with signed and sealed exchanges for the page.** Every `/api` route needs proof of the
  owner-only `access-token`, and the page never sends the token: it asks `GET /challenge` for a
  single-use value this process issued, adds a nonce of its own, and signs method, path and query,
  body digest, challenge, and nonce with the token
  (`Authorization: Usagedash <challenge>.<nonce>.<digest>`, `access.RequestVerifier`). The answer is
  signed in turn over the challenge, the nonce, the status, and the body (`X-Usagedash-Response`,
  `api.seal_answer`), and `static/access.js` uses an answer only when that signature is the
  dashboard's own: the server's challenge keeps requests fresh, the page's nonce keeps answers fresh,
  so a recorded answer fits no later request. Nothing in this reads the wall clock. The signatures say
  who sent and who answered;
  every body is sealed as well (`access.BodySealer`: AES-256-GCM under a key both sides derive from
  the token with HKDF-SHA256, `access.seal_key`). The page seals a request body for its method, target,
  challenge, and nonce before it signs the sealed bytes. After token and origin checks accept a signed
  request, the dashboard seals and signs its answer for the challenge, nonce, and status, including
  route errors. Refusals before that acceptance are neither sealed nor signed. The guard hands the
  route the opened JSON (`api.with_body`), and refuses (400) a signed request with a nonempty body
  that is in clear or does not open for it
  (`api.open_signed_body`). A process holding the port, answering the page itself or relaying the
  exchange, sees sealed bytes. `Authorization: Bearer <token>` is accepted, with plain bodies,
  for clients that are not a page (the tests). The page shell and static files served over HTTP hold
  no account data and are public. State-changing routes also require the custom header and either the
  page's own `Origin` or `Origin: null` with a signature (the file page). The cross-origin layer allows
  the `null` origin only and records every preflight it turns away (`api.RecordedCorsGuard`).
- **Both checks run in the `/api` router's route class** (`api.guarded_route_class`), not as
  dependencies: FastAPI decodes a JSON body before it runs a dependency, so a malformed body was
  answered 422 with no token and from any origin. The route class reads the body itself, bounded
  (`api.read_signed_body`), checks first, and hands the route the opened bytes for a sealed body
  or the original bytes otherwise (`api.open_signed_body`, `api.with_body`), so a new `/api` route
  needs the proof, and one that changes something (any method outside `api.SAFE_METHODS`) needs the
  page, by construction. Do not move the checks back into `Depends`.
- **The page never sends the token on the wire or puts it in a cookie, argument, or request URL.**
  `usagedash open` writes `open-dashboard.html`, which is the dashboard page itself
  (`access.write_bootstrap_page`): the served shell with the stylesheet and the modules inlined as
  `data:` URLs, the token and the origin as JSON-encoded globals, and a hash-pinned CSP. A browser
  refuses a module loaded from `file:`, and a module served by the listener would be the listener's
  code. The page addresses `127.0.0.1` only, never `localhost`, which may resolve to `::1`. An
  installed agent is the address (`agent.installed_environment`, then `cli._settings_for_open`): this
  shell's port and settings are not. A plist that cannot be read, or settings that are not valid,
  opens nothing and does not restart the agent. The page is written only when `GET /healthz` answers
  at that address (`cli._dashboard_is_listening`), so a stopped agent is reported instead of opened;
  that probe is a liveness check, not what protects the token. The served page removes the
  `localStorage` item an earlier version of the page kept.
- **The token file is created by linking a fully written temporary file into place** (`access.py`), so
  the server and `usagedash open` starting together always end up with the same token.
- **"Current windows" is decided by a per-account reading sequence, not by timestamps.**
  `accounts.reading_seq` and `quota_snapshots.confirmed_seq` make "the windows of the latest reading"
  exact even for two readings in one second.
- **A quota step's two times are whole microseconds since the epoch** (`observed_us`, `confirmed_us`);
  every other stored time is ISO text to the second, except a recorded login boundary, which keeps
  the fraction of its original instant (`recorded_login_boundaries`). History and retention compare
  the times of one step with a cutoff that can fall anywhere inside a second, and two steps observed
  in that second must still be one before it and one after.
- **Who is logged in comes from the reading, or is asked of the source after a read that gave none.**
  It is never asked before the read: a read can discover a logout or a different login (CLI-mode Codex
  only learns identity from its reads), and an identity taken before it would record the earlier
  account as current.
- **A login is dated from when it was seen** (`collector._read_and_identify`, `seen_at`): the clock
  after the read and the identity lookup returned, not the attempt's start, which dates only the
  reading and the attempt's status. A read can take most of a minute, and an interval dated from its
  start would take what the earlier account used meanwhile. The timeline is stored to the second, so
  a new interval starts at the first whole second at or after that moment
  (`timeutil.whole_second_not_before` in `db.observe_active_account`), never before it.
- **The current login interval is the one recorded last** (`db._current_interval`), not the one with
  the latest time, and no time in the timeline goes back: a new interval starts no earlier than the
  previous one was last observed, and a last observation only moves forward. Ordered by time, a clock
  that stepped back left an earlier login current after a later one was seen.
- **A login that cannot be told is recorded as nobody's** (`collector._who_is_logged_in`). If asking
  the source fails, an interval that belongs to no account is recorded before the token scan runs.
  Leaving the earlier interval open credits that account with usage the scan then moves past for good.
- **The login is committed in its own transaction, before the reading** (`collector._record_login`).
  A login that cannot be stored as it is, is stored as nobody's. If that also fails, the scan is held
  back (`TokenScanError`) and the gap is later recorded as nobody's (`_login_unrecorded_since`). The
  original instant is stored in that same transaction (`db.acknowledge_recorded_login_boundary`).
- **`usagedash collect` exits nonzero when the dashboard would list a source as a warning or an error**
  (`cli.FAILED_SOURCE_SEVERITIES`), judged from its own cycle's report via `api.severity_of`. Do not
  read the stored statuses back. Logged out, and nothing to report yet, exit 0. The account line is
  the label from `collector._read_and_identify`. "not logged in" is only a source that recorded nobody.
- **A limit Codex reports as reached becomes a full blocking window** (`reported_limit`, no reset time)
  when no listed window is already full, so the account reads as out of quota instead of available.
- **An extra Codex limit is keyed by its meter id, never by its display name**
  (`codex.extra_limit_prefixes`, both sources). The id is `metered_feature` or `limitId`. Both sources
  must produce the same key, because one account can be read both ways in one cycle.
- **An extra Codex limit that cannot be read is left out and counted on the card**
  (`codex.unreadable_limits_fact`), in both sources. A limit with no window is counted the same way.
  The main limit's windows are required first (`codex.require_main_limit`).
- **Payload models use strict types with explicit conversions.** `validation.LenientModel` rejects
  implicit text-to-number and boolean-to-number coercion; an integer is valid for a float field.
  `Instant` reads timestamp text, `DecimalAmount` accepts decimal text, and `OptionalIdentifier`
  normalizes blank text to `None`. Infinity and NaN are refused.
  Percentages use `UsedPercent` and epoch times `EpochSeconds`. `schema_boundary` translates parsing
  failures to `ProviderSchemaError` for callers that use it. Codex identity maps `ValidationError`
  directly to `ProviderSchemaError`; Grok session-file validation reaches `scanners.base.isolated`
  and becomes a failed-file `ScanReport`. Preserve those distinct error boundaries.
- **Errors never carry credential material, and neither do their causes.** Validation errors omit
  their input. A decoding error, a non-JSON body, and a transport failure are raised `from None`.
  Proxy and certificate settings the client cannot use are a `ConfigError` by type, with no cause
  (`http.build_client`).
- **Every token total comes from one query** (`db.token_usage`), filtered once in `tokens.py`, so the
  chart series, per-model totals, and per-provider totals cannot disagree.
- **Retail API cost is the published short-prompt standard rate** (`pricing.py`), attached in
  `api._token_report_json` and not inside `build_token_report`. Costs are integer microdollars,
  rounded half up once per model row; an account total is the sum of those rows. The page then shows
  each figure to the nearest dollar, and the all-accounts row adds those rounded figures. A used
  component with no published rate leaves the whole model out of the dollar total. Reasoning is part
  of output and is not priced again. The assumptions are in `docs/known-limitations.md`.
- **The limit on token counts belongs to the store, not to `TokenCounts`.** One stored row holds at
  most `MAX_TOKEN_COUNT` per count (a `CHECK` in the schema, and checked before a value reaches
  SQLite). `db.token_usage` adds rows up in Python because SQLite's `SUM` fails past 64 bits, and
  `TokenCounts` has no upper limit, so whatever was committed can always be read and added. Scanners
  bound each record at `MAX_TOKENS_PER_RECORD`.
- **One session file never stops the files after it** (`scanners.base.isolated`): unreadable,
  uninterpretable, or unstorable, it is counted as failed by error type, its transaction is rolled
  back, and its cursor does not move.
- **An unexpected exception's message goes nowhere.** The store and the page get the type only
  (`collector._reportable`); the log gets the stack and the types in the chain
  (`collector.describe_without_messages`). Never use `logger.exception` or `exc_info`. The API answers
  with its own plain 500. A handler that fails says so by type only (`cli.MessageFreeHandlerErrors`).
  A SQLite failure after setup is one line naming the type (`cli.main`). A command error or an
  interrupt is one line (`cli._one_line` collapses only `\r` and `\n`). Uvicorn gets `log_config=None`.
  A new handler must derive from that class.
- **The HTTP boundary contains ordinary exceptions and strips chains from cancellations and stops**
  (`api.SealedResponses`, the outermost ASGI layer, not a `@app.middleware` function). It logs ordinary
  failures by type and location, then sends a plain 500 only if the response has not started. That
  answer is sent after the `except` block has ended. A `BaseException` that still leaves the HTTP
  boundary has its `__context__` and `__cause__` cleared first; non-HTTP scopes are delegated unchanged.
- **A log line names a request by its route template** (`api.route_template`), never by its path,
  which holds the account id. A refused request (no valid token, not from the page) is logged at
  warning level the same way (`api._log_refusal`), and an accepted one at info level by its credential
  kind (`api._log_acceptance`), so a copied credential leaves a trail. The server's access log stays off
  (`cli._serve`).
- **Validation errors do not echo rejected input.** FastAPI's default validation answer quotes the rejected
  value, so `api._refuse_invalid_request` answers with the part of the request and pydantic's name
  for the kind of failure.
- **Time helpers refuse naive datetimes**, which Python would convert using this machine's time zone.
- **Owner-only is the permission bits and the macOS access-control list, set every time** and not only
  at creation (`private.py`): the data directory, standing homes, the log directory, the database, the
  lock, the config file, and the token. An inherited "allow" entry makes a `0700` directory readable
  by whoever it names, so the list is removed through the open descriptor (`restrict_to_owner`). A
  symbolic link is refused where one of these is expected.
- **The login timeline has logged-out intervals** (`db.observe_active_account` with `None`), so usage
  after a logout is unattributed and a re-login starts a new interval.
- **Grok usage is never read from its log.** A record does not say whose it is. Do not bring back a
  log parser, a freshness rule, or a bound taken from the login timeline. The log is read only for
  the plan tier, and only the record the dashboard's own exchange wrote (`grok_cli`).
- **A Grok billing answer without `creditUsagePercent` is 0% used, if it states a complete current
  period** (`grok.CreditsConfig`). The answer is proto3 JSON, which omits a number that is zero; seen
  live 2026-10-03 on an account two days into its week. The reading carries
  `grok.UNSTATED_USAGE_FACT`, so the card shows the 0% was not a stated value. Without a complete
  period it is a `ProviderSchemaError`. Do not make the field required again: the account with the
  most room is the one that leaves it out.
- **A Grok login that does not say when its credential was created is not read**
  (`grok.credential_key` raises `ProviderSchemaError`). Two undated credentials of one account have
  the same key, so a switch away and back during a read would pass as no change.
- **A Claude response is counted once, at the largest value of each count, under its earliest line**
  (`db.add_message_usage`, the only writer). It is stored under `db.CLAUDE_USAGE_FILE_KEY` and cannot
  be added, cleared, or replaced by file. A key lives in the same hour and both expire after
  `db.USAGE_RETENTION` (400 days). Before a call changes anything, every row it touches must hold the
  sum of all keys stored in it (`db._verify_claude_usage`). An expired response's key is kept
  (`claude_expired_messages`); the one residual is in `docs/known-limitations.md`. A test seeds Claude
  usage only through `add_message_usage`.
- **A quota step holds until the next one of its account window is observed**, not until it was last
  confirmed. So a snapshot history starts with the step in effect at its start (`db.snapshot_history`),
  `db.prune` keeps the step in effect at the retention cutoff, and the page's sparkline starts at that
  level (`static/format.js` `stepsInEffectFrom`).
- **A source that reads identity and usage separately reads the login again afterwards**
  (`providers.base.ensure_same_login`). A login that changed in between is no data for that cycle.
  The reads are compared whole. Direct Grok compares `grok.credential_key`. The Grok CLI source
  compares `grok.login_identity`, because its own run refreshes the token. Two members of one Claude
  organization are two logins.
- **A window is rolled forward to empty only by a reset later than its reading**
  (`projection._reset_still_ahead_of`). A reset time earlier than the reading that reports it cleared
  none of the measured usage, so the window keeps its usage and has no known reset time. Pace follows
  the same rule: such a reset bounds nothing, and a snapshot last confirmed before it is not a
  baseline (`projection._current_period`). A reading is live only when its age is between zero and
  the staleness allowance; one dated after now is not.
- **A typed error's message holds no text a provider, a CLI, or a session file supplied.** It is stored
  and shown as it is (`collector._reportable`). A CLI's rejection is reported by its numeric code, a
  schema error by declared field names with any other location part masked
  (`validation.summarize_validation_error`), and a schema error is raised with no cause.
- **A Codex rollout is counted once per home, from the largest file of its name**
  (`scanners.codex._plan`). Position and usage are per home and file name. The largest copy is read,
  onward when it is the copy read before and afresh otherwise. A replacement is kept only when its
  read reaches `_replacement_floor` (what was read, or what a cut-short copy still holds). The saved
  position is the end of what was read, not the saved size. While a directory cannot be listed, a
  rollout whose read copy is missing fails that cycle (`CopyOutOfViewError`) and keeps its state.
- **A saved scan position is resumed only in the file it was taken in**
  (`scanners.base.OpenSessionFile.resume`). Claude and Codex cursors carry a digest of the 4 KB
  before the position (`anchor_of`); a file that no longer holds those bytes there was rewritten in
  place and is read afresh, however long it is now. A position outside the file is put right at
  once (`needs_scan`). The whole prefix is not hashed: rollouts reach hundreds of megabytes, so a
  rewrite between scans that leaves those 4 KB alone is missed.
- **One scan of a session file goes through one open file** (`scanners.base.OpenSessionFile`): the
  saved position, the lines, and the cursor. Do not open the path a second time. A rewrite in place
  is caught by reading those bytes again (`pread`) and fails as `SessionFileChangedError`. Whether
  to open a file is decided from its path's size and time.
- **Two configured Codex homes share no scan state.** One position per file name shared across homes,
  with arbitration between their copies, was tried and produced a new ordering or timing defect in
  every review round. Do not bring it back; a home copied whole into another is counted in both, as
  the README says.
- **Session files are found with `scanners.base.session_files`, not `glob`/`rglob`**, which skip a
  directory they cannot list without saying so. An unlistable directory is a failed file.
- **Grok's plan tier comes only from the log record its own billing exchange wrote** (`grok_cli`): the
  log's length is marked before the exchange, and an earlier record can be another login's.
- **Refresh spacing is measured from when the last attempt returned** (`api.CollectionLoop`), not from
  the collector's last success, and a cycle counts as in progress from the moment it is handed to its
  worker thread. Shutdown waits for a cycle in flight, however often the waiter is cancelled
  (`api._until_done`).
- **The outcome of a worker or a background task is retrieved by a done callback**, never only by
  whoever awaits it (`CollectionLoop._cycle_returned`, `api._log_loop_end`). An exception nobody
  retrieves is logged by asyncio itself, message included.
- **A refresh is refused once the collection loop has stopped** (`CollectionLoop.has_stopped`):
  `/api/refresh` answers 503 instead of "scheduled", because nothing is left to run the cycle.
- **Every account id leaves the API as decimal text** (`api._text_id`). An id is an SQLite row id, any
  64-bit integer, and a browser holds numbers exactly only up to 2**53. The page treats an id as an
  opaque key. Do not add a response that writes an id as a JSON number.
- **A chart measure adds up what it shows; it never subtracts from the total** (`MEASURES` in
  `static/format.js`). Counts arrive as decimal text (`api._counts_text`) and become exact `bigint`
  values through `static/format.js` `parseTokenReport`; totals above 2**53 retain every integer part.
- **On the page, every load and action owns its failure** (`static/failures.js`), so one succeeding
  never hides another's failure.
- **A rename is one operation at a time** (`static/rename.js`): idle, editing, or saving, and while a
  save is in flight nothing else can start. Do not add a second path that saves a name; every race in
  the earlier code needed two saves under way. The draft lives in that state, not in the input
  element, so `renderProviders` rebuilds the cards on every load, form included.
- **Saved Codex scan state must be exactly what the scanner writes** (`scanners.codex._SavedState`). A
  missing total is unknown, not zero. Refused state, or state that is not JSON, costs that one file a
  fresh read. The same holds for the running totals in a rollout: the four counters every Codex
  version writes are required, and the newer cache-write counter may be left out only while the file
  has counted none. A request's own usage is not a baseline and is read leniently.
- **A Grok usage file must state its `turns`.** A changed file replaces what the session contributed,
  so a document without its history is refused instead of being read as a session with no usage.
- **API reads go through `Database.read()`** and never wait for a collection cycle's write lock.
- **The 429 backoff is in memory.** After a restart, or in a manual `usagedash collect`, a backed-off
  source is asked once more. Persisting it was judged not worth a schema column.
- **Only account-wide limits and Claude's model-scoped limits are `blocking`.** Codex code-review and
  per-feature pools are shown but never mark the account out of quota.
- **Quota snapshots are a step function**: a row per change, `confirmed_at` advanced on an unchanged
  re-read. "Current" windows are those confirmed by the account's most recent reading. A step
  continues only while everything measured is unchanged (percentage, reset, blocking, window
  length); only the label is updated in place.
- **What the store keeps as JSON goes through `db.decode_json` / `db.encode_json`**, which refuse
  `NaN` and `Infinity`: Python's own decoder reads them and its encoder writes them, though neither
  is JSON. Hour and day keys are formatted with a four-digit year, not `strftime("%Y")`, which glibc
  leaves unpadded before the year 1000. A `TokenCounts` count must be an `int`.
- **A token counter stated as `null` is malformed, not zero** (`scanners.base.token_count` takes the
  record and the field, so a field left out, which is zero, cannot look like a stated null).
- **A pace forecast is shown only while it is still to come** (`projection._forecast_still_ahead`):
  a reading stays live for minutes, and fast usage reaches its forecast within them.
- **A vendor CLI's line has a length limit** (`rpc.MAX_LINE_BYTES`, 8 MiB): the buffer of what is
  not yet a line is bounded, and consumed line by line rather than copied on every read.
- **Token usage is stored per session file** so a re-read replaces rather than double counts. Claude
  additionally counts each response once by its response id (`message.id`), because transcripts repeat
  usage on every content block and copy history into resumed sessions. The key is that id alone: a
  line's `uuid` is its own, and `requestId` is optional on a line, so neither may be part of it
  (`scanners.claude.parse_transcript_line`). A line with no response id is malformed.
- **At one instant, a Claude line credited to an account outranks one credited to nobody**
  (`db._CountedMessage.order`). A transcript copied into another home is read with that home's login
  timeline, which may not reach back to the line, and the copy must not take the usage away.
- **Claude and Grok tokens are attributed by login timeline**; their session files do not name the account.
- **Local only for now.** A GCP-hosted view would need this Mac to push readings; not built.

## Gotchas

- `CLAUDE_CONFIG_DIR` must **not** be set when targeting the default `~/.claude`: setting it changes the
  keychain item Claude Code looks up. `ClaudeSourceConfig.is_default_home` guards this.
- A child `claude` must not inherit `CLAUDE*` / `ANTHROPIC_*` variables, or it authenticates as the
  enclosing session or an API key instead of the standing login (`claude_cli.cli_environment`).
- A standing Claude login that reads "Claude Code reported no plan limits for this login" while it
  still has an account file usually means its keychain item is not under `claude.keychain_account()`.
  `security dump-keychain` shows each item's `acct`; signing that home in again with the same `--name`
  files it correctly.
- Signing in again to a Claude home the config already lists keeps that listed spelling
  for the keychain name only (`standing._registered_claude_config_dir`). `CLAUDE_CONFIG_DIR`
  is the prepared home, not a fresh resolution of the listed spelling.
  `CLAUDE_SECURESTORAGE_CONFIG_DIR` is the listed spelling when the two differ
  (`standing.login_environment`). A listed spelling that another account could make
  lead elsewhere, or that passes through a link this account owns, is refused before the vendor CLI
  runs (`private.ensure_no_other_account_can_redirect`: the ancestor walk with the last entry
  examined too). Claude hashes the spelling into the keychain
  item and does not resolve it, so a login stored under another spelling of the same directory
  is a different item. `usagedash login` reads that spelling under `config.toml.lock` and holds
  the lock until registration has decided (`standing.sign_in_and_register`). Separate open
  descriptors serialize callers across both threads and processes.
- launchd gives agents a minimal environment. `agent install` pins each resolved vendor CLI as
  `USAGEDASH_<TOOL>_BINARY` (`cli._pinned_executables`, ahead of the config via `config._binary`),
  the port, the data directory, a config file the environment named, and this shell's `HOME`
  (`cli._install_agent`). A missing CLI is pinned as its own bare name, which `config._binary`
  interprets as PATH-only lookup: later default-directory or config-file discovery cannot bypass the
  install-time PATH checks. Other relative environment values remain invalid. Reinstall to discover
  a later CLI outside that checked PATH. An unnamed config stays unnamed, so a missing default file
  is not required. Reinstall after moving a CLI, including a Homebrew cask upgrade that deletes the
  version directory the pin names. Until that reinstall, a start whose pinned file is gone uses the
  checked-PATH file of the same name, or the vendor command when that name is the release
  (`executables.checked_command`, `fallback_name`). Each pinned CLI is checked and stored as its canonical path
  (`executables.canonical_executable`): the file and its ancestors must be
  owned by this account or root and not writable by another account or through an ACL. A missing
  CLI is allowed only under an existing parent that passes `ensure_only_owner_can_change`.
  The file check uses `what_lets_another_account_change_by_path`, so native binaries need execute
  permission but not read permission.
  The agent's PATH (`agent.agent_search_path`) keeps shell order, then CLI and system directories,
  deduplicated by canonical path. Empty PATH entries, missing directories, paths whose spelling or canonical
  form holds `:`, and directories another account could change are left out. Validation failures,
  including missing, unsafe, or unexaminable directories and canonical paths containing `:`, produce
  warnings with reasons separated by semicolons. Empty entries and candidate paths
  containing `:` before canonicalization are skipped silently.
  `executables.canonical_directory` checks the canonical directory and its
  ancestry, then `executables._refuse_an_executable_entry_another_account_could_change` scans its entries.
  Every regular file and link target must pass the path-based check, including non-executable files:
  ownership or ACL rights such as `writesecurity` and `chown` can enable execution later
  (`private.what_lets_another_account_change_by_path`, `os.lstat` plus `acl_get_link_np`), so a
  root-owned execute-only binary need not be opened. Every retained link hop is checked, not just
  the final resolved file: PATH lookup follows those links again. Each hop must belong to this account
  or root, including a leaf under a root-owned sticky directory whose owner could retarget it.
  A dangling link passes only when
  its existing parent is safe; a missing entry under a sticky directory is not safe to run later.
  System directories get the same checks; an unexaminable or unsafe helper omits the whole directory.
  If that loses the required `security` command, `agent._with_required_helpers` validates
  `/usr/bin/security` through `executables.checked_executable` and exposes a canonical link in a checked
  owner-only `agent-bin` under the data directory. Claude invokes bare `security` for keychain access.
  Unsafe targets, unsafe fallback contents, conflicting links, and PATH separators in the fallback
  directory fail before launchd changes; reinstall reuses a matching safe link.
  Extend `executables.REQUIRED_PATH_HELPERS` when a vendor adds another required system command.
  Only programs get the `administrators_may_write=True` allowance for the `admin` group's write bit
  (`private.ADMINISTRATORS_GROUP_ID`), as on Homebrew's prefix. Administrators can become root;
  other group and other-write bits and ACL write grants still refuse. Private dashboard directories
  retain the strict default. Do not restore an unchecked PATH directory to silence an install warning.
- `launchctl bootout` returns before the old process has exited; `agent.install` and `agent.uninstall`
  wait until launchd no longer lists the service. Install otherwise fails to re-register with
  "Bootstrap failed: 5". Uninstall otherwise deletes the plist while the old process is still listed.
- `launchctl print` exits 113 for a service launchd does not list. Only that means "not loaded"
  (`agent.is_loaded`); any other failure is an `AgentError`, or a reinstall would take a running agent
  for one that was never there.
- The config file is read through `read_file_only_owner_can_change` before executable settings are
  parsed (`config._read_toml`). A file another account can write, or a symbolic link, is refused and
  left unchanged. A file others can only read is made owner-only on that read. `serve`, `collect`,
  `login`, and `agent install` check the config directory's ancestors first. Those commands, plus
  `status` and `open`, also refuse a data directory another account can replace. `open` then requires
  a 200 from `GET /healthz` (`cli._dashboard_is_listening`): nothing on the request, no redirect, no
  proxy, only the status is read. The exchange runs on a daemon thread waited for one deadline, so a
  listener stalled in connection setup, a header, or informational responses means nothing is
  listening; an answer the probe timed after the deadline is ignored, however late this thread reads
  it. Closing the client could not bound connection setup.
  The plist and log directories get the same ancestor check. A command error, an interrupt, a plist
  that cannot be parsed, and a config that cannot be examined are one line (`cli._one_line`,
  `agent._path_exists`, `standing._config_file_is_present`). An interrupt in uninstall recovery still
  exits 130.
- Lenient pydantic models all derive from `validation.LenientModel`; scanner skip rules live in
  `scanners/base.py`. Add to those, do not re-declare them per provider.
- The HTTP client is `httpx2` (the maintained successor Starlette's test client now requires), not `httpx`.
- Tests that spawn a child process see the outer environment; assert on the keys you set, not on the
  whole environment.
- The vendor payload shapes are undocumented. When one changes, the source reports `ProviderSchemaError`
  under "Data sources"; fix the pydantic model in the matching provider module and add a test.
- `SCHEMA_VERSION` is 2. A database stamped at version 1 gains `recorded_login_boundaries` in the
  same transaction as the version stamp (`Database._migrate`). The next schema change adds another
  migration instead of editing `_SCHEMA_STATEMENTS` for a database that is already stamped. A
  database created by a build from before the initial merge has an older table layout under version
  1; move it aside and let the dashboard create a new one.
- A new payload field must be declared with the type the provider actually sends. The models are
  strict, so `float` accepts a JSON integer but `int` does not accept `300.0`, and a number sent as
  text needs `DecimalAmount`.
- Do not build a path or a launchd label from `USER`, `LOGNAME`, or `HOME`. `agent.agent_paths` takes
  the account name, and the home when none is passed, from the user database (`agent.account_home`).
  A home that is not absolute is an `AgentError`.
- A scanner must validate a record completely before it changes carried-over state
  (`scanners/codex._apply_line`); advancing running totals for a record that is then rejected loses usage.
- The page's loads go through `latestOnly` (`static/latest.js`), so an older request answered late can
  never replace what a newer request shows. Use it for any new load that can overlap itself.

## Contributing

Install and everyday use are in `README.md`. How to propose a change is in `CONTRIBUTING.md`.
Accepted behavior is in `docs/known-limitations.md`. Changes land through a pull request.

Page development tools are locked in `package-lock.json`. `npm run format:check` checks the page,
tests, and workflow formatting. `npm run test:browser` runs Playwright on the actual page with the
signed demo API in `tests/browser/demo.js`, including both appearance modes and narrow screens.
`npm run screenshot` generates the README images with synthetic accounts; never substitute live
account data. `AGENTS.md` lists the complete development checks.
