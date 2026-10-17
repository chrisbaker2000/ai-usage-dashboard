"""Command-line entry point: run the dashboard, collect once, print status, or manage the launchd agent."""

import argparse
import contextlib
import fcntl
import logging
import os
import sqlite3
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from io import TextIOWrapper
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

import httpx2
import uvicorn

from usagedash import agent, standing
from usagedash.access import ensure_access_token, write_bootstrap_page
from usagedash.api import HEALTH_PATH, build_state, create_app, severity_of
from usagedash.config import (
    CONFIG_ENV,
    DATA_DIR_ENV,
    PORT_ENV,
    SETTINGS_ENV_PREFIX,
    Settings,
    binary_env,
    config_file,
    load_settings,
)
from usagedash.db import Database
from usagedash.errors import ConfigError, UsageDashError
from usagedash.executables import canonical_executable as _canonical_executable_only_this_account_can_change
from usagedash.http import build_client
from usagedash.models import Provider
from usagedash.paths import same_directory
from usagedash.private import (
    ensure_no_ancestor_lets_another_account_replace,
    open_private_file,
)
from usagedash.timeutil import utc_now
from usagedash.wiring import build_collector

DEFAULT_LOGIN_NAME = "standing"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_FILE_NAME = "usagedash.log"
LOG_FILE_MAX_BYTES = 1_000_000
LOG_FILE_BACKUPS = 3

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_INTERRUPTED = 130
LIVENESS_TIMEOUT_SECONDS = 2.0
"""How long `usagedash open` waits for the dashboard to answer that it is running."""
FAILED_SOURCE_SEVERITIES = frozenset({"warning", "error"})


class MessageFreeHandlerErrors(logging.Handler):
    """Report handler failures by type only; the standard handler leaks exceptions, record text and credentials."""

    def handleError(self, record: logging.LogRecord) -> None:
        error = sys.exc_info()[1]
        # Why suppressed: this runs when a handler has already failed, and standard error, one of
        # those handlers, may be what failed. Nothing else is left to report it to.
        with contextlib.suppress(OSError, ValueError):
            sys.stderr.write(f"usagedash: a log record could not be written: {type(error).__name__}\n")


class MessageFreeStreamHandler(MessageFreeHandlerErrors, logging.StreamHandler[TextIO]):
    """Standard error, reporting its own failures as ``MessageFreeHandlerErrors`` does."""


class PrivateRotatingFileHandler(MessageFreeHandlerErrors, RotatingFileHandler):
    """Share private logs across processes: reopen stale inodes, then hold a depth-counted flock through flush.

    If another process rotated, only reopen. Every reopen refuses links. Hold the thread lock across both
    stream and rotation-descriptor closure, including errors, to exclude concurrent handle().
    """

    def __init__(self, filename: Path, *, maxBytes: int, backupCount: int) -> None:
        """Open the rotation lock file, then the log.

        Raises:
            ConfigError: if either cannot be opened as a private file.
            OSError: if the log cannot be switched to appending.
        """
        self._rotation_lock: int | None = open_private_file(filename.with_name(f"{filename.name}.lock"))
        self._lock_depth = 0
        try:
            super().__init__(filename, maxBytes=maxBytes, backupCount=backupCount)
        except BaseException:
            self._close_rotation_lock()
            raise

    def close(self) -> None:
        self.acquire()
        try:
            try:
                super().close()
            finally:
                self._close_rotation_lock()
        finally:
            self.release()

    def _close_rotation_lock(self) -> None:
        if self._rotation_lock is not None:
            os.close(self._rotation_lock)
            self._rotation_lock = None

    def emit(self, record: logging.LogRecord) -> None:
        """Write the record under the rotation lock, from the check of the file at its name through the flush."""
        try:
            self._take_the_rotation_lock()
        except (OSError, ConfigError):
            self.handleError(record)
            return
        try:
            self._reopen_if_rotated_elsewhere()
            super().emit(record)
        except (OSError, ConfigError):
            self.handleError(record)
        finally:
            self._release_the_rotation_lock()

    def doRollover(self) -> None:
        """Rotate, or only reopen when another process rotated first, under the rotation lock.

        Raises:
            ConfigError: if the log file, or a lock file closed with the handler, cannot be opened as a
                private file.
            OSError: if the lock cannot be taken or the rotation fails.
        """
        self._take_the_rotation_lock()
        try:
            stream = self.stream
            if stream is None or self._holds_the_file_at_its_name(stream):
                super().doRollover()
            else:
                self._reopen(stream)
        finally:
            self._release_the_rotation_lock()

    def _take_the_rotation_lock(self) -> None:
        """Take the lock for the outermost holder on this handler; ``emit`` holds it around a rollover."""
        descriptor = self._rotation_lock_descriptor()
        if self._lock_depth == 0:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        self._lock_depth += 1

    def _release_the_rotation_lock(self) -> None:
        self._lock_depth -= 1
        if self._lock_depth == 0:
            fcntl.flock(self._rotation_lock_descriptor(), fcntl.LOCK_UN)

    def _rotation_lock_descriptor(self) -> int:
        """Return the lock file's descriptor, opened again after the handler was closed."""
        if self._rotation_lock is None:
            log_file = Path(self.baseFilename)
            self._rotation_lock = open_private_file(log_file.with_name(f"{log_file.name}.lock"))
        return self._rotation_lock

    def _open(self) -> TextIOWrapper:
        """Open the log file for appending.

        Raises:
            ConfigError: if a symbolic link or a directory is in its place, or it cannot be opened or
                made owner-only.
            OSError: if the open file cannot be switched to appending.
        """
        descriptor = open_private_file(Path(self.baseFilename))
        try:
            # Why: writes that always go to the end are what keeps two processes logging to one file,
            # the agent and a server started by hand, from overwriting each other's lines.
            fcntl.fcntl(descriptor, fcntl.F_SETFL, fcntl.fcntl(descriptor, fcntl.F_GETFL) | os.O_APPEND)
        except OSError:
            os.close(descriptor)
            raise
        return os.fdopen(descriptor, "a", encoding=self.encoding, errors=self.errors)

    def _reopen_if_rotated_elsewhere(self) -> None:
        stream = self.stream
        if stream is not None and not self._holds_the_file_at_its_name(stream):
            self._reopen(stream)

    def _reopen(self, stream: TextIOWrapper) -> None:
        stream.close()
        self.stream = self._open()

    def _holds_the_file_at_its_name(self, stream: TextIOWrapper) -> bool:
        """Report whether ``stream`` is the file now at the log's name; a closed stream or no file there is not."""
        try:
            named = os.stat(self.baseFilename)
            held = os.fstat(stream.fileno())
        except (FileNotFoundError, ValueError):
            return False
        return (named.st_dev, named.st_ino) == (held.st_dev, held.st_ino)


def _configure_logging(log_to_file: bool) -> None:
    """Log to stderr and, for serve, a private rotating file; all handlers report their own failures by type.

    Raises:
        ConfigError: if the log directory cannot be made private or the log file cannot be opened.
    """
    handlers: list[logging.Handler] = [MessageFreeStreamHandler(sys.stderr)]
    if log_to_file:
        log_dir = agent.agent_paths().log_dir
        agent.prepare_log_directory(log_dir)
        try:
            handlers.append(
                PrivateRotatingFileHandler(
                    log_dir / LOG_FILE_NAME, maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_FILE_BACKUPS
                )
            )
        except OSError as error:
            raise ConfigError(f"cannot open the log file in {log_dir}: {type(error).__name__}") from error
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=handlers)
    # Why: request logging at INFO prints every polled URL; request failures still surface at WARNING.
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("httpcore2").setLevel(logging.WARNING)


def _open_database(settings: Settings) -> Database:
    database = Database(settings.db_path)
    database.initialize()
    return database


def _serve(settings: Settings) -> int:
    _configure_logging(log_to_file=True)
    database = _open_database(settings)
    access_token = ensure_access_token(settings.data_dir)
    with build_client() as client:
        collector = build_collector(settings, database, client, utc_now)
        app = create_app(settings, database, collector, utc_now, access_token=access_token)
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=settings.host,
                port=settings.port,
                log_level="warning",
                access_log=False,
                log_config=None,
                workers=1,
            )
        )
        # Why workers=1: left out, Uvicorn reads WEB_CONCURRENCY, and a non-number there, inherited
        # from an unrelated tool, raised ValueError as a traceback.
        # Why log_config=None: Uvicorn's default gives its loggers a stderr handler whose handleError
        # prints tracebacks; without it, its records reach the handlers configured above.
        # Why not uvicorn.run: after a clean Ctrl-C shutdown Uvicorn raises the interrupt again, and
        # uvicorn.run swallows it; run this way it reaches the command line's own handler. A server
        # that cannot start logs why and exits with a failure status itself either way.
        server.run()
    return EXIT_OK


def _settings_the_installed_agent_serves() -> Settings | None:
    """Overlay the installed plist's USAGEDASH settings on this shell, or return None when absent.

    Raises:
        AgentError: if a property list is present and cannot be read.
        ConfigError: if the settings it names are not valid.
    """
    installed = agent.installed_environment(agent.agent_paths())
    if installed is None:
        return None
    shell = {name: value for name, value in os.environ.items() if not name.startswith(SETTINGS_ENV_PREFIX)}
    recorded_home = installed.get("HOME")
    home = Path(recorded_home) if recorded_home is not None else None
    return load_settings({**shell, **installed}, home=home)


def _same_dashboard(installed: Settings, invoked: Settings) -> bool:
    """Compare stored data/config directories by filesystem identity, excluding port; restart preserves that port."""
    return same_directory(installed.data_dir, invoked.data_dir) and same_directory(
        installed.config_path, invoked.config_path
    )


def _settings_for_open() -> Settings:
    """Return the installed agent's settings, or this shell's when nothing is installed.

    This shell's settings are not read when an agent is installed. An invalid override here must not
    stop the page from addressing the agent that is running.
    """
    served = _settings_the_installed_agent_serves()
    if served is not None:
        return served
    return load_settings()


def _dashboard_is_listening(dashboard_url: str, monotonic: Callable[[], float] = time.monotonic) -> bool:
    """Accept a 200 healthz answer timestamped before one monotonic deadline; this proves liveness only.

    No credentials, redirects, proxies or body reads. A daemon probe cannot keep the command alive.
    """
    if LIVENESS_TIMEOUT_SECONDS <= 0:
        return False
    url = f"{dashboard_url.rstrip('/')}{HEALTH_PATH}"
    answered: list[tuple[float, bool]] = []

    def probe() -> None:
        try:
            with (
                httpx2.Client(timeout=LIVENESS_TIMEOUT_SECONDS, follow_redirects=False, trust_env=False) as client,
                client.stream("GET", url) as response,
            ):
                answered.append((monotonic(), response.status_code == 200))
        except (httpx2.HTTPError, httpx2.StreamClosed):
            answered.append((monotonic(), False))

    deadline = monotonic() + LIVENESS_TIMEOUT_SECONDS
    worker = threading.Thread(target=probe, name="usagedash-liveness", daemon=True)
    worker.start()
    worker.join(LIVENESS_TIMEOUT_SECONDS)
    return bool(answered) and answered[0][0] <= deadline and answered[0][1]


def _open(settings: Settings) -> int:
    """After a successful liveness probe, write/open the private page with embedded code and token.

    Use installed-agent settings ahead of shell settings. The token never enters arguments, URLs or the
    network; challenge-bound requests carry signatures.
    """
    served = _settings_the_installed_agent_serves()
    if served is not None:
        settings = served
    # Why this directory too: the config file can sit somewhere this account controls while the data
    # directory is under one another account can replace. The token and the page would then be written there.
    ensure_no_ancestor_lets_another_account_replace(settings.data_dir)
    dashboard_url = f"http://{settings.host}:{settings.port}/"
    if not _dashboard_is_listening(dashboard_url):
        return _nothing_is_listening(dashboard_url)
    token = ensure_access_token(settings.data_dir)
    page = write_bootstrap_page(settings.data_dir, dashboard_url, token)
    try:
        opened = webbrowser.open(page.as_uri())
    except (webbrowser.Error, OSError, ValueError, IndexError) as error:
        # Why these four: the standard library reads the BROWSER variable when it is asked to open
        # something, and a value it cannot split into a command raises ValueError (a quote left open)
        # or IndexError (nothing but spaces). The other two are a browser that could not be started.
        raise ConfigError(f"could not open a browser: {type(error).__name__}") from error
    if not opened:
        print("error: could not open a browser", file=sys.stderr)
        return EXIT_FAILURE
    print(f"Opened the dashboard at {dashboard_url}")
    return EXIT_OK


def _nothing_is_listening(dashboard_url: str) -> int:
    print(
        f"error: nothing answered at {dashboard_url}; is the dashboard running? "
        "Check `usagedash agent status`. Nothing was opened",
        file=sys.stderr,
    )
    return EXIT_FAILURE


def _collect(settings: Settings) -> int:
    _configure_logging(log_to_file=False)
    database = _open_database(settings)
    with build_client() as client:
        report = build_collector(settings, database, client, utc_now).run_cycle()
    for outcome in report.outcomes:
        print(f"{outcome.source_id}")
        print(f"  account: {outcome.account_label or 'not logged in'}")
        print(f"  quota:   {outcome.quota_error or 'ok'}")
        print(f"  tokens:  {outcome.token_error or f'ok ({outcome.files_read} file(s) read)'}")
    print(f"cycle took {(report.finished_at - report.started_at).total_seconds():.1f}s")
    # Why: the next cycle can overwrite stored statuses after our lock is released. Classify this
    # report with the page's rules; no login or no readings yet are expected, not command failures.
    failed = any(
        severity_of(error_type) in FAILED_SOURCE_SEVERITIES
        for outcome in report.outcomes
        for error_type in (outcome.quota_error_type, outcome.token_error_type)
    )
    return EXIT_FAILURE if failed else EXIT_OK


def _status(settings: Settings) -> int:
    database = _open_database(settings)
    state = build_state(database, settings, None, utc_now())
    for provider in state["providers"]:
        print(f"{provider['display_name']}: {provider['advice']['summary']}")
        for account in provider["accounts"]:
            marker = "active" if account["is_active"] else "logged in" if account["is_live"] else "logged out"
            print(f"  {account['label']} [{marker}] {account['plan'] or ''}".rstrip())
            for window in account["windows"]:
                reset = window["resets_at"] or "no reset time"
                print(
                    f"    {window['label']:<28} {window['effective_percent']:5.1f}% used  "
                    f"resets {reset}  ({window['freshness']}, as of {window['as_of']})"
                )
    for source in state["sources"]:
        if source["severity"] != "ok":
            print(f"[{source['severity']}] {source['source_id']} {source['kind']}: {source['message']}")
    return EXIT_OK


def _pinned_executables(executables: Mapping[str, str]) -> dict[str, str]:
    """Pin checked canonical vendor paths so launchd cannot select a different installation.

    Missing CLIs use their own bare name as a checked-PATH-only override; no late config/default
    discovery. Reinstall to discover a CLI outside that PATH.
    """
    return {binary_env(name): path for name, path in executables.items()}


def _install_agent(settings: Settings) -> int:
    # Why not Path.home() alone: an empty HOME becomes "/" and a relative one follows the working directory.
    recorded_home = os.environ.get("HOME")
    home = Path(recorded_home) if recorded_home is not None else Path.home()
    if not home.is_absolute():
        raise ConfigError("HOME must name an absolute directory")
    paths = agent.agent_paths()
    binaries = settings.binaries
    executables = {"claude": binaries.claude, "codex": binaries.codex, "grok": binaries.grok}
    for name, executable in executables.items():
        if Path(executable).is_absolute():
            executables[name] = str(_canonical_executable_only_this_account_can_change(Path(executable)))
    overrides = {name: value for name, value in os.environ.items() if name.startswith(SETTINGS_ENV_PREFIX)}
    # A named config is pinned absolutely. An unnamed one stays unnamed, so a missing default is not required.
    overrides[PORT_ENV] = str(settings.port)
    overrides[DATA_DIR_ENV] = str(settings.data_dir)
    if os.environ.get(CONFIG_ENV):
        overrides[CONFIG_ENV] = str(settings.config_path)
    overrides["HOME"] = str(home)
    overrides |= _pinned_executables(executables)
    agent.install(paths, settings.data_dir, list(executables.values()), overrides)
    print(f"Installed {paths.label} on http://{settings.host}:{settings.port}; run `usagedash open` to view it")
    return EXIT_OK


def _manage_installed_agent(action: str) -> int:
    """Status/uninstall bypass current settings. Report the data directory from the removed plist read under
    its lock, or leave it unnamed if the trusted plist does not supply one.
    """
    paths = agent.agent_paths()
    if action == "status":
        print(f"{paths.label}: {'running' if agent.is_loaded(paths) else 'not loaded'}")
        return EXIT_OK
    removed = agent.uninstall(paths) or {}
    data_dir = removed.get(DATA_DIR_ENV)
    if data_dir is None:
        print(f"Removed {paths.label}; its data was kept")
        return EXIT_OK
    print(f"Removed {paths.label}; data in {data_dir} was kept")
    return EXIT_OK


def _settings_with_config(config_file: Path) -> Settings:
    """Load the settings as this command did, with ``config_file`` as the config; one not there holds nothing."""
    return load_settings({**os.environ, CONFIG_ENV: str(config_file)}, file_may_be_absent=True)


def _login(settings: Settings, provider: Provider, name: str) -> int:
    home = standing.standing_home(settings, provider, name)
    standing.prepare_home(settings, home)
    # Why: reject unregistrable configs before asking for an interactive sign-in.
    standing.ensure_registrable(settings, provider, home, _settings_with_config)
    print(f"Standing {provider.display_name} login: {home}")
    print(f"Sign in to the {provider.display_name} account this login should track.")
    # Why one lock: concurrent registration must not change Claude's keychain spelling during sign-in.
    exit_status = standing.sign_in_and_register(settings, provider, home, _settings_with_config)
    if exit_status != 0:
        print(f"error: {provider.display_name} login exited with status {exit_status}", file=sys.stderr)
        return EXIT_FAILURE
    # Why: kickstart uses the installed plist, so restart only a dashboard whose settings match.
    paths = agent.agent_paths()
    # Why: keep installation stable between inspection and restart.
    with agent.holding_installation_lock(paths):
        installed = _settings_the_installed_agent_serves()
        if installed is None:
            if agent.is_loaded(paths):
                print(
                    "Registered. A dashboard is loaded, but its property list is gone, so which settings it "
                    "reads cannot be told; run `usagedash agent install` so it reads this login."
                )
                return EXIT_OK
            restarted = False
        elif not _same_dashboard(installed, settings):
            if agent.is_loaded(paths):
                print(
                    "Registered. The running dashboard was installed with different settings and will not "
                    "read this login until `usagedash agent install` is run again."
                )
            else:
                print(_START_THE_DASHBOARD)
            return EXIT_OK
        else:
            restarted = agent.kickstart_if_loaded(paths)
    if restarted:
        print("Dashboard restarted; the account appears after the next collection.")
    else:
        print(_START_THE_DASHBOARD)
    return EXIT_OK


_START_THE_DASHBOARD = "Registered. Start the dashboard with `usagedash agent install` to begin tracking it."


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="usagedash", description="Usage and quota dashboard for AI coding accounts")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="run the dashboard and collect on a schedule")
    commands.add_parser("collect", help="run one collection cycle and print the outcome")
    commands.add_parser("status", help="print the latest stored quota state")
    commands.add_parser("open", help="open the dashboard in your browser and sign the browser in")
    agent_parser = commands.add_parser("agent", help="manage the launchd agent that keeps the dashboard running")
    agent_parser.add_argument("action", choices=["install", "uninstall", "status"])
    login_parser = commands.add_parser(
        "login", help="create a standing login so an account is tracked without switching your working login"
    )
    login_parser.add_argument("provider", choices=[provider.value for provider in Provider])
    login_parser.add_argument("--name", default=DEFAULT_LOGIN_NAME, help="short label for this login (a-z, 0-9, -)")
    return parser


def _runs_a_configured_program(arguments: argparse.Namespace) -> bool:
    """Select serve, collect, login and agent install for config executable checks; status/open do not run them.
    Agent status/uninstall run before settings load.
    """
    if arguments.command in {"serve", "collect", "login"}:
        return True
    return bool(arguments.command == "agent" and arguments.action == "install")


def _one_line(text: str) -> str:
    """Collapse line feeds and carriage returns so a command error stays one line.

    Launchctl's message and a path can contain either. Only those two are collapsed: a space stays a
    space, and U+2028 stays, because both are ordinary characters in a path.
    """
    flattened = text.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(part.strip() for part in flattened.split("\n") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.command == "agent" and arguments.action != "install":
            return _manage_installed_agent(arguments.action)
        if arguments.command == "open":
            return _open(_settings_for_open())
        # Why before the file is parsed: the directory check does not look at the file's own mode, and
        # a path read from a file another account can replace is the path that then runs.
        if _runs_a_configured_program(arguments):
            standing._refuse_replaceable_config_directory(config_file())
        settings = load_settings()
        # Why independently of the config file: a safe custom config path does not cover the data
        # directory. Collection opens the database by that path, so a replaced directory would take
        # the account history.
        ensure_no_ancestor_lets_another_account_replace(settings.data_dir)
        if arguments.command == "serve":
            return _serve(settings)
        if arguments.command == "collect":
            return _collect(settings)
        if arguments.command == "status":
            return _status(settings)
        if arguments.command == "login":
            return _login(settings, Provider(arguments.provider), arguments.name)
        return _install_agent(settings)
    except UsageDashError as error:
        print(f"error: {_one_line(str(error))}", file=sys.stderr)
        return EXIT_FAILURE
    except sqlite3.Error as error:
        # Why the type alone: SQLite puts the database path in the message.
        print(f"error: database operation failed: {type(error).__name__}", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt as interrupt:
        # Why the interrupt can carry text: a command interrupted part-way through a change says what
        # state it left things in. A line break in that text is still one line.
        detail = _one_line(str(interrupt)) if interrupt.args else ""
        print(f"interrupted; {detail}" if detail else "interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
