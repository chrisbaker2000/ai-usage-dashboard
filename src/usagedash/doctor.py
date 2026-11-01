"""Read-only diagnosis of an installation: what each check found and, when it is wrong, the next command.

Nothing here starts, stops, or changes anything. Every check reports through a ``Finding`` and keeps going,
so one failure never hides the ones after it. Messages carry only what this package or launchd produced:
the text of an error this package raised on purpose, a launchd field, a file's own last lines. An
unexpected exception is reported by its type and nothing else, like everywhere else in this package.
"""

import os
import re
import stat
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx2

from usagedash import agent
from usagedash.agent import AgentPaths, ServiceState
from usagedash.api import HEALTH_PATH
from usagedash.config import LOOPBACK_HOST, Settings
from usagedash.errors import ConfigError, UsageDashError
from usagedash.executables import canonical_executable, checked_executable
from usagedash.private import ensure_no_ancestor_lets_another_account_replace

HTTP_OK = 200
LSOF_BINARY = "/usr/sbin/lsof"
LSOF_TIMEOUT_SECONDS = 10
HEALTH_TIMEOUT_SECONDS = 2.0
LOG_TAIL_BYTES = 8192
LOG_TAIL_LINES = 15
MAX_LOG_LINE_CHARACTERS = 240
MAX_LISTED_OMISSIONS = 10
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
    """One result: a title, optional lines under it, and the command that fixes it when there is one."""

    level: Level
    title: str
    details: tuple[str, ...] = ()
    fix: str | None = None


SettingsOutcome = Settings | UsageDashError | None
"""Settings that loaded, the error that stopped them loading, or None when there is nothing to load."""


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
                fix="Correct the setting it names in the config file, or unset that USAGEDASH_* variable.",
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
                fix="Run `usagedash agent install` again from the repository directory.",
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
    """Check that the data directory is not one another account could replace."""
    try:
        ensure_no_ancestor_lets_another_account_replace(settings.data_dir)
    except ConfigError as error:
        return [
            Finding(
                Level.FAIL,
                f"The data directory cannot be trusted: {error}",
                fix="Make the directory it names writable by you alone, or point USAGEDASH_DATA_DIR elsewhere.",
            )
        ]
    if not os.path.lexists(settings.data_dir):
        return [
            Finding(Level.INFO, f"The data directory {settings.data_dir} does not exist yet; the first start makes it")
        ]
    return [Finding(Level.OK, f"The data directory {settings.data_dir} is safe to use")]


def vendor_findings(settings: Settings) -> list[Finding]:
    """Report where each vendor CLI is found. A missing one is information: its cards just stay empty."""
    findings: list[Finding] = []
    binaries = settings.binaries
    for name, executable in (("claude", binaries.claude), ("codex", binaries.codex), ("grok", binaries.grok)):
        if os.sep not in executable:
            findings.append(
                Finding(
                    Level.INFO,
                    f"{name}: not found on PATH",
                    (f"Its accounts show no data until {name} is installed. Ignore this if you do not use it.",),
                )
            )
        elif not os.path.lexists(executable):
            findings.append(
                Finding(
                    Level.WARN,
                    f"{name}: {executable} does not exist",
                    fix=f"Install {name} again, or correct [binaries] in the config file, then run `agent install`.",
                )
            )
        else:
            findings.append(_checked_vendor_finding(name, Path(executable)))
    return findings


def _checked_vendor_finding(name: str, executable: Path) -> Finding:
    try:
        return Finding(Level.OK, f"{name}: {canonical_executable(executable)}")
    except ConfigError as error:
        return Finding(Level.WARN, f"{name}: {error}")


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
            f"The directory holding a vendor CLI would be left off the agent's PATH: {omission.directory}",
            (omission.reason,),
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


def _describe_exit(state: ServiceState) -> str:
    if state.last_signal not in (None, "", "(none)", "(never signalled)"):
        return f"signal {state.last_signal}"
    return f"exit code {state.last_exit}"


def _agent_findings(paths: AgentPaths) -> tuple[list[Finding], ServiceState | None]:
    """Say whether the agent is installed, loaded, and running, and return launchd's state for later checks."""
    try:
        installed = agent.installed_environment(paths)
    except UsageDashError as error:
        return [Finding(Level.FAIL, f"The installed property list cannot be trusted: {error}")], None
    try:
        state = agent.service_state(paths)
    except UsageDashError as error:
        return [Finding(Level.FAIL, f"launchd cannot be asked about the agent: {error}")], None
    install_fix = "Run `usagedash agent install` from the repository directory."
    if state is None:
        if installed is None:
            return [Finding(Level.FAIL, "The agent is not installed", fix=install_fix)], None
        return [
            Finding(
                Level.FAIL,
                "The agent is installed but launchd has not loaded it",
                (f"property list: {paths.plist}",),
                install_fix,
            )
        ], None
    findings: list[Finding] = []
    if installed is None:
        findings.append(Finding(Level.WARN, "launchd lists an agent whose property list is gone", fix=install_fix))
    findings.append(_running_finding(state))
    return findings, state


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
            f"The agent started and stopped ({_describe_exit(state)})",
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
    move_fix = (
        f"Quit that program, or move the dashboard: put `port = {port + 1}` in the config file, then run "
        "`usagedash agent install` and `usagedash open`."
    )
    if health.kind is HealthKind.ANSWERED and health.status == HTTP_OK:
        if holder is not None and agent_pid is not None and holder.pid != agent_pid:
            return [
                Finding(
                    Level.WARN,
                    f"The dashboard answers on port {port}, but the agent (pid {agent_pid}) is not the one holding it",
                    (
                        *_holder_detail(port, holder),
                        "This is usually a `usagedash serve` started by hand, or an older agent still running.",
                    ),
                    "Quit that process, then run `usagedash agent install`.",
                )
            ]
        return [Finding(Level.OK, f"The dashboard answers on port {port}")]
    if health.kind is HealthKind.ANSWERED:
        return [
            Finding(
                Level.FAIL,
                f"Something other than the dashboard answers on port {port} (HTTP {health.status})",
                _holder_detail(port, holder),
                move_fix,
            )
        ]
    if health.kind is HealthKind.REFUSED and agent_pid is not None:
        return [
            Finding(
                Level.FAIL,
                f"The agent is running but nothing is listening on port {port} yet",
                (
                    "It may still be starting: wait ten seconds and run `usagedash doctor` again.",
                    "If it stays like this, the log lines below say why it could not start.",
                ),
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
    if health.kind is HealthKind.TIMED_OUT:
        return [
            Finding(
                Level.FAIL,
                f"Port {port} accepted a connection but did not answer within {HEALTH_TIMEOUT_SECONDS:.0f} seconds",
                _holder_detail(port, holder),
                move_fix,
            )
        ]
    return [
        Finding(
            Level.FAIL,
            f"The request to port {port} failed before an answer",
            _holder_detail(port, holder),
            move_fix,
        )
    ]


def log_findings(paths: AgentPaths) -> list[Finding]:
    """Show the last lines of the two logs the agent writes, when they hold anything."""
    findings: list[Finding] = []
    for log in (paths.launchd_logs[1], paths.server_log):
        lines = tail_lines(log)
        if lines:
            findings.append(Finding(Level.INFO, f"Last lines of {log}", tuple(lines)))
    return findings


def diagnose_agent(paths: AgentPaths, port: int | None) -> list[Finding]:
    """Check the agent, whether it answers, and its logs. Used alone after an install or an open that failed."""
    findings, state = _agent_findings(paths)
    if port is not None:
        findings.extend(_answer_findings(port, state))
    findings.extend(log_findings(paths))
    return findings


def diagnose(paths: AgentPaths, *, shell: SettingsOutcome, served: SettingsOutcome) -> list[Finding]:
    """Run every check and return the findings in the order they should be read.

    ``shell`` is the settings this shell loads; ``served`` is what the installed agent was installed with,
    and wins for the port and data directory because that is what is actually running.
    """
    findings = [*system_findings(), *settings_findings(shell, served)]
    effective = served if isinstance(served, Settings) else shell if isinstance(shell, Settings) else None
    if effective is not None:
        findings.extend(data_directory_findings(effective))
    if isinstance(shell, Settings):
        findings.extend(vendor_findings(shell))
        findings.extend(path_findings(shell))
    agent_findings, state = _agent_findings(paths)
    findings.extend(agent_findings)
    if effective is not None:
        findings.extend(_answer_findings(effective.port, state))
    if any(finding.level in (Level.WARN, Level.FAIL) for finding in findings):
        findings.extend(log_findings(paths))
    return findings


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
