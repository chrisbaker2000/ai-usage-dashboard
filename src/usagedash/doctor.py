"""Read-only diagnosis of an installation: what each check found and, when it is wrong, the next command.

Nothing here starts, stops, installs, or removes anything, and no file or folder is created. Settings and the
installed property list are read the way every command reads them, and that reader makes an owner-owned file
that others can read owner-only (``private.read_file_only_owner_can_change``): the one change this module can
cause. Every check reports through a ``Finding`` and keeps going, so one failure never hides the ones after
it. Messages carry only what this package or launchd produced: the text of an error this package raised on
purpose, a launchd field, a file's own last lines. An unexpected exception is reported by its type and
nothing else, like everywhere else in this package.
"""

import contextlib
import logging
import os
import re
import socket
import stat
import subprocess
import sys
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx2

from usagedash import agent
from usagedash.agent import AgentPaths, ServiceState
from usagedash.api import HEALTH_PATH
from usagedash.config import LOOPBACK_HOST, MAX_PORT, MIN_PORT, PORT_ENV, Settings
from usagedash.errors import ConfigError, UsageDashError
from usagedash.executables import checked_command, checked_executable
from usagedash.private import ensure_no_ancestor_lets_another_account_replace, what_lets_another_account_change

HTTP_OK = 200
LSOF_BINARY = "/usr/sbin/lsof"
LSOF_TIMEOUT_SECONDS = 10
HEALTH_TIMEOUT_SECONDS = 2.0
LOG_TAIL_BYTES = 8192
LOG_TAIL_LINES = 15
MAX_LOG_LINE_CHARACTERS = 240
MAX_LISTED_OMISSIONS = 10
PORT_SUGGESTION_SPAN = 20
ISSUE_HINT = (
    "Still stuck? Open an issue with the bug template and paste this output. Your home folder is shown as ~; "
    "read it first, and leave out anything you would not post."
)


class Level(StrEnum):
    """How much a finding matters: ``FAIL`` stops the dashboard from working, ``WARN`` may limit it."""

    OK = "ok"
    INFO = "info"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class Finding:
    """One result: a title, optional lines under it, and the next step.

    Raises:
        ValueError: if a ``FAIL`` names no next step. A report that stops the user without saying what to do
            is the failure this command exists to end, so the type refuses to build one.
    """

    level: Level
    title: str
    details: tuple[str, ...] = ()
    fix: str | None = None

    def __post_init__(self) -> None:
        if self.level is Level.FAIL and not self.fix:
            raise ValueError(f"a FAIL must name the next step: {self.title}")


SettingsOutcome = Settings | UsageDashError | None
"""Settings that loaded, the error that stopped them loading, or None when there is nothing to load."""

_RECOVER_THE_PROPERTY_LIST = (
    "Run `usagedash agent uninstall`, which removes a property list it cannot trust without starting it, then "
    "`usagedash agent install`. When the file is unsafe and not just unreadable, `agent install` alone reads it "
    "through the same trust check and stops at the same error."
)

_CORRECT_THE_SETTING = (
    "Correct the setting the message names, in the config file or in a USAGEDASH_* variable, then run "
    "`usagedash agent install`: it records the settings this shell loads in place of the ones that no longer load."
)


class HealthKind(StrEnum):
    ANSWERED = "answered"
    REFUSED = "refused"
    TIMED_OUT = "timed-out"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class HealthResult:
    kind: HealthKind
    status: int | None = None


@dataclass(frozen=True, slots=True)
class PortHolder:
    pid: int
    command: str


def probe_health(url: str) -> HealthResult:
    """Ask ``url`` for its status once, without credentials, redirects, a proxy, or reading the body."""
    try:
        with (
            httpx2.Client(timeout=HEALTH_TIMEOUT_SECONDS, follow_redirects=False, trust_env=False) as client,
            client.stream("GET", url) as response,
        ):
            return HealthResult(HealthKind.ANSWERED, response.status_code)
    except httpx2.ConnectError:
        return HealthResult(HealthKind.REFUSED)
    except httpx2.TimeoutException:
        return HealthResult(HealthKind.TIMED_OUT)
    except (httpx2.HTTPError, httpx2.StreamClosed):
        return HealthResult(HealthKind.FAILED)


def port_holder(port: int) -> PortHolder | None:
    """Return the process listening on ``port`` that this account can see, or None when none is found.

    ``lsof`` lists only this account's processes without administrator rights, so None means "none that
    this account can see", not "free".
    """
    try:
        result = subprocess.run(
            [LSOF_BINARY, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpc"],
            capture_output=True,
            text=True,
            timeout=LSOF_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _first_holder(result.stdout)


def _first_holder(output: str) -> PortHolder | None:
    pid: int | None = None
    for line in output.splitlines():
        if line.startswith("p") and line[1:].isascii() and line[1:].isdecimal():
            pid = int(line[1:])
        elif line.startswith("c") and pid is not None:
            return PortHolder(pid, _printable(line[1:]))
    return None


def _printable(text: str) -> str:
    """Make text safe to print: a control character in a log or a command name must not drive the terminal."""
    shown = "".join(character if character.isprintable() else " " if character == "\t" else "?" for character in text)
    return shown if len(shown) <= MAX_LOG_LINE_CHARACTERS else f"{shown[:MAX_LOG_LINE_CHARACTERS]}…"


def tail_lines(path: Path, *, max_lines: int = LOG_TAIL_LINES, max_bytes: int = LOG_TAIL_BYTES) -> list[str] | None:
    """Return the last lines of a log this account owns, or None when there is no such readable file.

    The file is opened without following a link and without blocking, and must be a regular file of this
    account's, so a link or a pipe put in the log directory is not read.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_uid != os.geteuid():
            return None
        start = max(0, status.st_size - max_bytes)
        # Why one byte early: the byte before the window says whether the first line in it is whole.
        before = max(0, start - 1)
        data = os.pread(descriptor, status.st_size - before, before)
    except OSError:
        return None
    finally:
        os.close(descriptor)
    begins_a_line = start == 0 or data[:1] == b"\n"
    lines = data[start - before :].decode("utf-8", errors="replace").splitlines()
    if not begins_a_line and lines:
        lines = lines[1:]
    return [_printable(line) for line in lines[-max_lines:]]


def system_findings() -> list[Finding]:
    """Check the platform and the interpreter ``agent install`` would run the dashboard with."""
    findings: list[Finding] = []
    if sys.platform != "darwin":
        findings.append(
            Finding(
                Level.FAIL,
                f"This is {sys.platform}, not macOS",
                ("The agent is a launchd service and Claude's working login is read from the macOS keychain.",),
                "Run usagedash on a Mac.",
            )
        )
        return findings
    version = sys.version_info
    findings.append(Finding(Level.INFO, f"Python {version.major}.{version.minor}.{version.micro} at {sys.executable}"))
    try:
        checked_executable(Path(sys.executable).absolute(), "Python executable")
    except ConfigError as error:
        findings.append(
            Finding(
                Level.FAIL,
                f"`agent install` would refuse this Python: {error}",
                fix="Create the environment again with `uv sync` from a directory only you can write to.",
            )
        )
    return findings


def settings_findings(shell: SettingsOutcome, served: SettingsOutcome) -> list[Finding]:
    """Report the settings this shell loads and the ones the installed agent was installed with."""
    findings: list[Finding] = []
    if isinstance(shell, UsageDashError):
        findings.append(
            Finding(
                Level.FAIL,
                f"The settings cannot be loaded: {shell}",
                fix=(
                    "Read the message. If it names a setting, correct it in the config file or unset that "
                    "USAGEDASH_* variable. If it says a file's permissions are unsafe, make that file writable "
                    "by you alone (`chmod go-w <file>`). Then run `usagedash doctor` again."
                ),
            )
        )
    elif shell is not None:
        findings.append(
            Finding(
                Level.OK,
                f"Settings load: port {shell.port}, data in {shell.data_dir}",
                (f"config file: {shell.config_path}",),
            )
        )
    if served is None:
        return findings
    if isinstance(served, UsageDashError):
        findings.append(
            Finding(
                Level.FAIL,
                f"The installed agent's settings cannot be read: {served}",
                fix=_RECOVER_THE_PROPERTY_LIST if isinstance(served, agent.AgentError) else _CORRECT_THE_SETTING,
            )
        )
    elif isinstance(shell, Settings) and served.port != shell.port:
        findings.append(
            Finding(
                Level.INFO,
                f"The installed agent serves port {served.port}; this shell would use {shell.port}",
                ("`usagedash open` follows the agent.",),
            )
        )
    return findings


def data_directory_findings(settings: Settings) -> list[Finding]:
    """Check what ``agent install`` and the server require of the data directory, without creating it.

    The ancestors must not be replaceable by another account, and the directory itself must be a real
    folder this account owns that no other account can change: ``private.ensure_private_directory`` refuses a
    symbolic link or a file there and cannot make another account's folder private, and it tightens an owned
    folder's modes and access-control list only when it runs, not before. The ancestry walk does not look at
    that last entry, so "safe to use" is claimed only after ``what_lets_another_account_change`` finds nothing.
    """
    data_dir = settings.data_dir
    try:
        ensure_no_ancestor_lets_another_account_replace(data_dir)
    except ConfigError as error:
        return [
            Finding(
                Level.FAIL,
                f"The data directory cannot be trusted: {error}",
                fix="Make the directory it names writable by you alone, or point USAGEDASH_DATA_DIR elsewhere.",
            )
        ]
    cannot_examine = "Check the permissions of that folder and the ones above it, then run `usagedash doctor` again."
    try:
        status = os.lstat(data_dir)
    except FileNotFoundError:
        return [Finding(Level.INFO, f"The data directory {data_dir} does not exist yet; the first start makes it")]
    except OSError as error:
        return [
            Finding(
                Level.FAIL,
                f"The data directory {data_dir} cannot be examined: {type(error).__name__}",
                fix=cannot_examine,
            )
        ]
    in_the_way = "Move it aside or point USAGEDASH_DATA_DIR at another folder, then run `usagedash agent install`."
    if stat.S_ISLNK(status.st_mode):
        return [
            Finding(Level.FAIL, f"The data directory {data_dir} is a symbolic link, which is refused", fix=in_the_way)
        ]
    if not stat.S_ISDIR(status.st_mode):
        return [Finding(Level.FAIL, f"The data directory {data_dir} is a file, not a folder", fix=in_the_way)]
    if status.st_uid != os.geteuid():
        return [Finding(Level.FAIL, f"The data directory {data_dir} belongs to another account", fix=in_the_way)]
    try:
        problem = what_lets_another_account_change(data_dir)
    except ConfigError as error:
        return [Finding(Level.FAIL, f"The data directory cannot be examined: {error}", fix=cannot_examine)]
    if problem is not None:
        return [
            Finding(
                Level.FAIL,
                f"Another account can change the data directory {data_dir}: {problem}",
                ("It holds the database and the access token, so another account could replace them.",),
                "Run `usagedash agent install`, which makes it owner-only again.",
            )
        ]
    return [Finding(Level.OK, f"The data directory {data_dir} is safe to use")]


@contextlib.contextmanager
def _logging_silenced() -> Generator[None]:
    """Keep the log records the resolution code writes out of the report; ``doctor`` words its own findings."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def vendor_findings(settings: Settings, agent_path: str) -> list[Finding]:
    """Report how the agent would find each vendor CLI, by the resolution it uses when it starts one.

    ``agent_path`` is the PATH the agent runs with. A missing CLI is information: its cards just stay empty.
    """
    binaries = settings.binaries
    return [
        _vendor_finding(name, executable, agent_path, settings.data_dir)
        for name, executable in (("claude", binaries.claude), ("codex", binaries.codex), ("grok", binaries.grok))
    ]


def _vendor_finding(name: str, executable: str, agent_path: str, working_directory: Path) -> Finding:
    """Resolve one CLI with ``executables.checked_command``, then require what that check does not.

    That is the call the collector makes before it starts a CLI, so a bare name found on the agent's PATH and
    a pin whose file has been removed (a Homebrew cask upgrade does that) are judged as the agent will judge
    them, and not as a missing file. The check accepts a missing target and any file mode, so a dangling link
    or a file without the execute bit would pass it and still fail when the CLI is started.
    """
    pinned_file_is_gone = os.sep in executable and not os.path.lexists(executable)
    try:
        with _logging_silenced():
            command = checked_command([executable], {"PATH": agent_path}, working_directory, fallback_name=name)
    except FileNotFoundError:
        return Finding(
            Level.INFO,
            f"{name}: not found on PATH",
            (f"Its accounts show no data until {name} is installed. Ignore this if you do not use it.",),
        )
    except ConfigError as error:
        if pinned_file_is_gone:
            return Finding(
                Level.WARN,
                f"{name}: {error}",
                fix=f"Install {name} again, then run `usagedash agent install`.",
            )
        return Finding(Level.WARN, f"{name}: {error}")
    resolved = Path(command[0])
    if not os.path.isfile(resolved):
        return Finding(
            Level.WARN,
            f"{name}: {resolved} is not a file",
            ("A link that points at nothing, or a folder, cannot be run.",),
            f"Install {name} again, then run `usagedash agent install`.",
        )
    if not os.access(resolved, os.X_OK):
        return Finding(
            Level.WARN,
            f"{name}: {resolved} is not executable",
            fix=f"Run `chmod u+x {resolved}`, then `usagedash agent install`.",
        )
    if pinned_file_is_gone:
        return Finding(
            Level.WARN,
            f"{name}: the recorded {executable} is gone; the agent uses {resolved}",
            ("It found that file by name on its own PATH, so it still works.",),
            "Run `usagedash agent install` to record the current file.",
        )
    return Finding(Level.OK, f"{name}: {resolved}")


def path_findings(settings: Settings) -> list[Finding]:
    """Report the PATH directories ``agent install`` would leave off, and flag one that holds a vendor CLI."""
    binaries = settings.binaries
    try:
        omissions = agent.agent_path_omissions([binaries.claude, binaries.codex, binaries.grok])
    except OSError as error:
        return [Finding(Level.WARN, f"This shell's PATH could not be examined: {type(error).__name__}")]
    findings = [
        Finding(
            Level.WARN,
            f"The directory of a vendor CLI would be left off the agent's PATH: {omission.directory}",
            (
                omission.reason,
                "The CLI starts from its recorded path, but anything it starts by name from that directory (a "
                "Node-based CLI looks for `node`) will not be found.",
            ),
            "Make that directory and what is in it writable by you alone, or move the CLI to one that is, "
            "then run `usagedash agent install`.",
        )
        for omission in omissions
        if omission.holds_vendor_cli
    ]
    others = [omission for omission in omissions if not omission.holds_vendor_cli]
    if others:
        listed = tuple(f"{omission.directory}: {omission.reason}" for omission in others[:MAX_LISTED_OMISSIONS])
        hidden = len(others) - len(listed)
        details = listed if hidden == 0 else (*listed, f"... and {hidden} more")
        findings.append(
            Finding(
                Level.INFO,
                f"{len(others)} PATH director{'y' if len(others) == 1 else 'ies'} would be left off the agent's PATH",
                (
                    "Missing, or another account could change them. The install warning about them is not an error.",
                    *details,
                ),
            )
        )
    return findings


def _port_is_free(port: int) -> bool:
    """Whether this account can bind ``port`` on the loopback address right now; the socket is closed at once."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((LOOPBACK_HOST, port))
        except OSError:
            return False
    return True


def suggest_port(port: int, is_free: Callable[[int], bool] | None = None) -> int | None:
    """Return a free port close to ``port`` inside the range the settings accept, or None when none is found.

    Ports above come first, then below, each up to ``PORT_SUGGESTION_SPAN`` away. ``port + 1`` alone would be
    out of range at the maximum, and may be taken as well.
    """
    free = _port_is_free if is_free is None else is_free
    above = range(port + 1, min(port + PORT_SUGGESTION_SPAN, MAX_PORT) + 1)
    below = range(port - 1, max(port - PORT_SUGGESTION_SPAN, MIN_PORT) - 1, -1)
    return next((candidate for candidate in (*above, *below) if free(candidate)), None)


def _move_fix(port: int) -> str:
    """Say how to move the dashboard off ``port`` so that the next ``agent install`` keeps the new port.

    The environment variable outranks the config file, so when this shell sets it, naming the file would
    leave the old port in force. The port is written to the config file otherwise because the agent records
    the port of the shell that installs it: a port given only in one shell's environment reverts to the
    file's, or to 8787, the next time `agent install` runs from another.
    """
    suggested = suggest_port(port)
    value = "a free port from 1024 to 65535" if suggested is None else f"{suggested} (free now)"
    if os.environ.get(PORT_ENV):
        setting = f"this shell sets {PORT_ENV}, which outranks the config file, so change it to {value}"
    else:
        setting = f"set `port` in the config file to {value}"
    return (
        f"Quit that program, or move the dashboard: {setting}, then run `usagedash agent install` and `usagedash open`."
    )


@dataclass(frozen=True, slots=True)
class AgentView:
    """What the agent checks learned: their findings, launchd's state, and the environment the plist records."""

    findings: list[Finding]
    state: ServiceState | None
    environment: dict[str, str] | None


def _agent_findings(paths: AgentPaths) -> AgentView:
    """Say whether the agent is installed, loaded, and running, and keep what later checks need to know."""
    findings: list[Finding] = []
    installed: dict[str, str] | None = None
    plist_untrusted = False
    try:
        installed = agent.installed_environment(paths)
    except UsageDashError as error:
        plist_untrusted = True
        findings.append(
            Finding(
                Level.FAIL,
                f"The installed property list cannot be trusted: {error}",
                fix=_RECOVER_THE_PROPERTY_LIST,
            )
        )
    try:
        state = agent.service_state(paths)
    except UsageDashError as error:
        findings.append(
            Finding(
                Level.FAIL,
                f"launchd cannot be asked about the agent: {error}",
                fix="Check that `/bin/launchctl` runs in this terminal, then run `usagedash doctor` again.",
            )
        )
        return AgentView(findings, None, installed)
    install_fix = "Run `usagedash agent install` from the repository directory."
    if state is None:
        if plist_untrusted:
            return AgentView(findings, None, installed)
        if installed is None:
            return AgentView([Finding(Level.FAIL, "The agent is not installed", fix=install_fix)], None, None)
        not_loaded = Finding(
            Level.FAIL,
            "The agent is installed but launchd has not loaded it",
            (f"property list: {paths.plist}",),
            install_fix,
        )
        return AgentView([not_loaded], None, installed)
    if installed is None and not plist_untrusted:
        findings.append(Finding(Level.WARN, "launchd lists an agent whose property list is gone", fix=install_fix))
    findings.append(_running_finding(state))
    return AgentView(findings, state, installed)


def _running_finding(state: ServiceState) -> Finding:
    facts = tuple(
        fact
        for fact in (
            f"state = {state.state}" if state.state is not None else None,
            f"runs = {state.runs}" if state.runs is not None else None,
            f"last exit = {state.last_exit}" if state.last_exit is not None else None,
        )
        if fact is not None
    )
    if state.pid is not None:
        return Finding(Level.OK, f"launchd is running the agent (pid {state.pid})", facts)
    if state.last_run_failed:
        return Finding(
            Level.FAIL,
            f"The agent started and stopped ({state.last_exit_description})",
            (
                *facts,
                f"launchd starts it again after {agent.RESTART_THROTTLE_SECONDS} seconds, and it stops the same "
                "way until the cause is fixed.",
            ),
            "Read the log lines below, fix what they name, then run `usagedash agent install`.",
        )
    return Finding(Level.WARN, "launchd lists the agent but is not running it", facts)


def _holder_detail(port: int, holder: PortHolder | None) -> tuple[str, ...]:
    if holder is None:
        return ()
    return (f"Port {port} is held by {holder.command} (pid {holder.pid}).",)


def _answer_findings(port: int, state: ServiceState | None) -> list[Finding]:
    """Ask the dashboard's address whether it answers, and name whoever else holds the port when it does not."""
    health = probe_health(f"http://{LOOPBACK_HOST}:{port}{HEALTH_PATH}")
    holder = port_holder(port)
    agent_pid = None if state is None else state.pid
    if health.kind is HealthKind.ANSWERED and health.status == HTTP_OK:
        if agent_pid is None:
            return [_served_by_someone_else(port, holder, "the agent is not running, so something else is serving it")]
        if holder is not None and holder.pid != agent_pid:
            return [_served_by_someone_else(port, holder, f"the agent (pid {agent_pid}) is not the one holding it")]
        return [Finding(Level.OK, f"The dashboard answers on port {port}")]
    if health.kind is HealthKind.REFUSED and agent_pid is not None:
        return [
            Finding(
                Level.FAIL,
                f"The agent is running but nothing is listening on port {port} yet",
                ("It may still be starting.",),
                "Wait ten seconds, then run `usagedash doctor` again. If it stays like this, the log lines below "
                "say why it could not start.",
            )
        ]
    if health.kind is HealthKind.REFUSED:
        return [
            Finding(
                Level.FAIL,
                f"Nothing is listening on port {port}",
                _holder_detail(port, holder),
                "Fix the agent problem above first.",
            )
        ]
    if holder is not None and agent_pid is not None and holder.pid == agent_pid:
        return [
            Finding(
                Level.FAIL,
                _agent_trouble_title(agent_pid, port, health),
                ("No other program is in the way, so a different port will not help.",),
                "Restart it with `usagedash agent install`. If this comes back, read the log lines below.",
            )
        ]
    if holder is None and agent_pid is not None:
        return [_unattributed_trouble(port, health)]
    return [Finding(Level.FAIL, _port_trouble_title(port, health), _holder_detail(port, holder), _move_fix(port))]


def _unattributed_trouble(port: int, health: HealthResult) -> Finding:
    """The agent runs and the port misbehaves, but nobody could be seen holding it, so neither can be blamed.

    ``lsof`` shows only this account's processes and can fail, so no holder is not evidence of another program.
    The agent is restarted first because that is the cheaper step and is right if the agent is at fault.
    """
    if health.kind is HealthKind.ANSWERED:
        title = f"Port {port} answers HTTP {health.status}, not the dashboard's 200"
    else:
        title = _port_trouble_title(port, health)
    return Finding(
        Level.FAIL,
        title,
        ("Doctor could not see which process holds the port, so it cannot tell the agent from another program.",),
        "Restart the agent with `usagedash agent install`, then run `usagedash doctor` again. If it fails the same "
        f"way, another program may hold port {port}. {_move_fix(port)} The log lines below may say more.",
    )


def _served_by_someone_else(port: int, holder: PortHolder | None, why: str) -> Finding:
    """A healthy answer that does not come from the agent is a failure: the installed setup is not serving.

    Whoever answers may be doing it well, but the agent is what keeps the dashboard up across logins, so
    a report that ends in "working" would be wrong while it is not the one serving.
    """
    return Finding(
        Level.FAIL,
        f"The dashboard answers on port {port}, but {why}",
        (
            *_holder_detail(port, holder),
            "This is usually a `usagedash serve` started by hand, or an older agent still running.",
        ),
        "Quit that process, then run `usagedash agent install`.",
    )


def _port_trouble_title(port: int, health: HealthResult) -> str:
    """Title for a port that answers wrongly, stalls, or fails, when the agent is not what holds it."""
    if health.kind is HealthKind.ANSWERED:
        return f"Something other than the dashboard answers on port {port} (HTTP {health.status})"
    if health.kind is HealthKind.TIMED_OUT:
        return f"Port {port} accepted a connection but did not answer within {HEALTH_TIMEOUT_SECONDS:.0f} seconds"
    return f"The request to port {port} failed before an answer"


def _agent_trouble_title(agent_pid: int, port: int, health: HealthResult) -> str:
    """Title for the same three results when the agent's own process holds the port."""
    held = f"The agent (pid {agent_pid}) holds port {port}, but"
    if health.kind is HealthKind.ANSWERED:
        return f"{held} its health check answers HTTP {health.status}"
    if health.kind is HealthKind.TIMED_OUT:
        return f"{held} it did not answer within {HEALTH_TIMEOUT_SECONDS:.0f} seconds"
    return f"{held} the health request failed before an answer"


def log_findings(paths: AgentPaths) -> list[Finding]:
    """Show the last lines of the two logs the agent writes, when they hold anything.

    The server writes its records to standard error as well as to its own file, so launchd's capture of
    standard error usually repeats them: a tail identical to the one just shown is not shown again.
    """
    findings: list[Finding] = []
    shown: list[str] | None = None
    for log in (paths.launchd_logs[1], paths.server_log):
        lines = tail_lines(log)
        if not lines or lines == shown:
            continue
        findings.append(Finding(Level.INFO, f"Last lines of {log}", tuple(lines)))
        shown = lines
    return findings


def diagnose_agent(paths: AgentPaths, port: int | None) -> list[Finding]:
    """Check the agent, whether it answers, and its logs. Used alone after an install or an open that failed."""
    view = _agent_findings(paths)
    findings = list(view.findings)
    if port is not None:
        findings.extend(_answer_findings(port, view.state))
    findings.extend(log_findings(paths))
    return findings


def diagnose(paths: AgentPaths, *, shell: SettingsOutcome, served: SettingsOutcome) -> list[Finding]:
    """Run every check and return the findings in the order they should be read.

    ``shell`` is the settings this shell loads; ``served`` is what the installed agent was installed with.
    ``served`` wins for the port, the data directory, and the vendor CLIs, because those are what is actually
    running: the agent starts the executables pinned in its property list, not the ones this shell would find.
    The PATH check stays with ``shell``: it asks what ``agent install`` would do if run from here.
    """
    findings = [*system_findings(), *settings_findings(shell, served)]
    effective = served if isinstance(served, Settings) else shell if isinstance(shell, Settings) else None
    view = _agent_findings(paths)
    if effective is not None:
        findings.extend(data_directory_findings(effective))
        findings.extend(vendor_findings(effective, _agent_path(view.environment)))
    if isinstance(shell, Settings):
        findings.extend(path_findings(shell))
    findings.extend(view.findings)
    if effective is not None:
        findings.extend(_answer_findings(effective.port, view.state))
    if has_failure(findings):
        findings.extend(log_findings(paths))
    return findings


def _agent_path(environment: dict[str, str] | None) -> str:
    """The PATH the agent runs with: the one its property list records, or this shell's when none is installed."""
    recorded = None if environment is None else environment.get("PATH")
    return recorded if recorded is not None else os.environ.get("PATH", os.defpath)


def has_failure(findings: Sequence[Finding]) -> bool:
    return any(finding.level is Level.FAIL for finding in findings)


_LEVEL_LABEL = {Level.OK: "ok  ", Level.INFO: "info", Level.WARN: "WARN", Level.FAIL: "FAIL"}


def _verdict(findings: Sequence[Finding]) -> str:
    failures = sum(finding.level is Level.FAIL for finding in findings)
    warnings = sum(finding.level is Level.WARN for finding in findings)
    if failures:
        return (
            f"Result: {failures} problem{'s' if failures != 1 else ''} found. "
            "Fix the first FAIL above, then run `usagedash doctor` again.\n" + ISSUE_HINT
        )
    checked = sum(finding.level is not Level.INFO for finding in findings)
    if warnings:
        return f"Result: working, with {warnings} warning{'s' if warnings != 1 else ''}. {checked} checks ran."
    return f"Result: all {checked} checks passed. Run `usagedash open` to view the dashboard."


def _blocks(findings: Sequence[Finding]) -> str:
    lines: list[str] = []
    for finding in findings:
        lines.append(f"[{_LEVEL_LABEL[finding.level]}] {finding.title}")
        lines.extend(f"       {detail}" for detail in finding.details)
        if finding.fix is not None:
            lines.append(f"       -> {finding.fix}")
    return "\n".join(lines)


def render(findings: Sequence[Finding], home: Path) -> str:
    """Format findings as plain text, one block each, ending in a verdict. The home folder is shown as ``~``."""
    return abbreviate_home(f"{_blocks(findings)}\n\n{_verdict(findings)}", home)


def render_agent_failure(paths: AgentPaths, port: int, home: Path) -> str:
    """Format the agent checks alone, for the message after an install that found nothing answering."""
    return abbreviate_home(_blocks(diagnose_agent(paths, port)), home)


def abbreviate_home(text: str, home: Path) -> str:
    """Replace the home folder with ``~`` where it is a whole path prefix, so pasted output names no account folder."""
    return re.sub(rf"{re.escape(str(home))}(?![\w-]|\.\w)", "~", text)
