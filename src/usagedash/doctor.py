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
from usagedash.agent import AgentPaths, AgentStatus, ServiceState
from usagedash.api import HEALTH_PATH
from usagedash.config import LOOPBACK_HOST, MAX_PORT, MIN_PORT, PORT_ENV, Settings
from usagedash.errors import ConfigError, UsageDashError
from usagedash.executables import checked_command, checked_executable, execution_environment
from usagedash.private import (
    ensure_no_ancestor_lets_another_account_replace,
    what_lets_another_account_change,
    what_lets_another_account_look,
)

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
    "Read the message. If it names a setting, correct it in the config file or in a USAGEDASH_* variable. If it "
    "says a file or folder is unsafe, make that one writable by you alone (`chmod go-w <path>`). Then run "
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
    """Return the process listening on ``port`` for connections to the loopback address, or None.

    ``lsof`` lists only this account's processes without administrator rights, so None means "none that
    this account can see", not "free". A listener bound to another interface cannot answer the dashboard's
    address and is not a holder.

    A nonzero exit is no answer, whatever was printed before it: ``lsof`` exits 1 both for "nothing listens"
    and when it could not examine everything, and a partial list with one listener in it would name a holder
    that another process, unseen, may share the port with.
    """
    try:
        result = subprocess.run(
            [LSOF_BINARY, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpcn"],
            capture_output=True,
            text=True,
            timeout=LSOF_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return _loopback_holder(result.stdout)


LOOPBACK_LISTENER_HOSTS = (LOOPBACK_HOST, "*")
"""The hosts a listener is bound to when a connection to the loopback address reaches it: that address itself,
or every IPv4 address. ``lsof`` prints the second as ``*``."""


def _loopback_holder(output: str) -> PortHolder | None:
    """Return the one process in ``lsof -Fpcn`` output with a listener the loopback address reaches, else None.

    Two processes can both listen (one on ``127.0.0.1`` and one on every address), and a connection reaches only
    one of them. Which one is not something this output says, so more than one is "no holder can be named", the
    same as none seen, and neither can be blamed or cleared. One process listening twice is still one holder.
    """
    holders: dict[int, str] = {}
    pid: int | None = None
    command = ""
    for line in output.split("\n"):
        field, value = line[:1], line[1:]
        if field == "p":
            pid = int(value) if value.isascii() and value.isdecimal() else None
            command = ""
        elif field == "c":
            command = value
        elif field == "n" and pid is not None and value.rpartition(":")[0] in LOOPBACK_LISTENER_HOSTS:
            holders.setdefault(pid, _printable(command))
    if len(holders) != 1:
        return None
    ((only_pid, only_command),) = holders.items()
    return PortHolder(only_pid, only_command)


_EMAIL_ADDRESS = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_BEARER_VALUE = re.compile(r"(?i)\bbearer[ \t]+\S+")
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"""(?ix)
    (
        (?:authorization|(?:api|access|private)[ _-]?key|secret|pass[ _-]?(?:word|wd|phrase)|token|cookie|credential)
                                                               # a credential name, inside any longer name
        [^=:\n]*                                               # what sits before the separator, however long
        [=:]                                                   # the separator
    )
    ([ \t]*)                                                   # the spacing, kept
    .*                                                         # the rest of the line, whatever its syntax
    """
)
_URL_CREDENTIALS = re.compile(r"\b([A-Za-z][A-Za-z0-9+.-]*://)(?=[^/\s?#]*:[^/\s?#]*@)[^/\s?#]*@")
"""The user and password of ``scheme://user:password@host``: everything in the authority up to its last ``@``,
which is where the host starts, so an ``@`` inside the user or the password does not end the match early. A
colon must come before that ``@``: ``scheme://user@host:8080`` has a user and no password and is left alone."""
_TOKEN_LIKE = re.compile(r"[A-Za-z0-9_+=-]{32,}")
REDACTED = "<redacted>"


def screen_for_sharing(text: str) -> str:
    """Screen each line of ``text`` on its own, so that nothing here can merge, remove, or add a line.

    A line ending in ``password:`` must not swallow the next line, and a trailing ``Bearer`` must not take the
    verdict label after it: the report is screened as a whole, and its structure is part of what it says.
    """
    return "\n".join(_screen_line(line) for line in text.split("\n"))


def _screen_line(line: str) -> str:
    """Replace what could be an email address or a credential in a line that a user may post publicly.

    The server and the launchd capture hold no credentials by design, but a crash before logging starts can put
    anything Python printed into the capture. This is a net, not a promise: it removes email addresses, a value
    after ``Bearer``, the user and password in ``scheme://user:password@host``, any run of 32 or more token
    characters, and, after a name that holds ``authorization``, ``secret``, ``token``, ``cookie``,
    ``credential``, ``password`` (also ``passwd``, ``pass word``, ``passphrase``), or a key (``api``,
    ``access``, or ``private``, joined to ``key`` by an underscore, a hyphen, a space, or nothing) and
    anything but a separator (a prefix such as ``DB_``, a quote, ``[0]``, ``.value``, a dotted path of any
    length) and then ``=`` or ``:``, the rest of the line.

    The rest of the line goes whatever it looks like, and the name's own shape and length are not modelled
    either. A screen that models the value's or the name's syntax is bypassed by the next syntax (a scheme and
    a credential, an escaped quote, a comma inside the value, an index on the name, a name longer than the
    allowance for one), so none is modelled. What that costs is the remainder of such a line, and what it does
    not see is a value printed on a later line than its name.

    There is no exemption for a remainder that looks like an exception type: ``password: ValueError`` is as
    likely a password as anything else, and a type name cannot be told from a value by its shape. A credential
    printed under a name that holds none of the words above passes. Nothing here may be cited as a guarantee
    that the output holds no credential.
    """
    screened = _BEARER_VALUE.sub(f"Bearer {REDACTED}", line)
    screened = _CREDENTIAL_ASSIGNMENT.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", screened)
    screened = _URL_CREDENTIALS.sub(lambda match: f"{match.group(1)}{REDACTED}@", screened)
    screened = _EMAIL_ADDRESS.sub("<email>", screened)
    return _TOKEN_LIKE.sub(REDACTED, screened)


def _single_line(text: str) -> str:
    """Make text one printable line: a tab becomes a space and any other control character, newline included,
    becomes ``?``.

    A path, a command name, or an error message can carry bytes that were not written by this package. A newline
    in one could forge a line of the report, such as a verdict, and an escape byte could drive the terminal.
    """
    return "".join(character if character.isprintable() else " " if character == "\t" else "?" for character in text)


def _printable(text: str) -> str:
    """Make a log line or a command name safe to print, and cut it at ``MAX_LOG_LINE_CHARACTERS``."""
    shown = _single_line(text)
    return shown if len(shown) <= MAX_LOG_LINE_CHARACTERS else f"{shown[:MAX_LOG_LINE_CHARACTERS]}…"


def tail_lines(path: Path, *, max_lines: int = LOG_TAIL_LINES, max_bytes: int = LOG_TAIL_BYTES) -> list[str] | None:
    """Return the last lines of a log this account owns, or None when there is no such readable file.

    The file is opened without following a link and without blocking, and must be a regular file of this
    account's, so a link or a pipe put in the log directory is not read.

    Lines end at a line feed and nowhere else. ``str.splitlines`` also ends one at a vertical tab, a form feed,
    the file, group, and record separators, a next-line character, and U+2028 and U+2029, and a credential's
    name and its value on either side of one would be screened as two lines, each looking harmless alone.
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
    lines = data[start - before :].decode("utf-8", errors="replace").split("\n")
    if lines[-1] == "":
        lines.pop()
    lines = [line.removesuffix("\r") for line in lines]
    if not begins_a_line and len(lines) == 1:
        return [f"(the last line is longer than {max_bytes} bytes and is not shown; open {path} to read it)"]
    if not begins_a_line and lines:
        lines = lines[1:]
    return [_printable(screen_for_sharing(line)) for line in lines[-max_lines:]]


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


def _missing_data_directory_finding(data_dir: Path) -> Finding:
    """Say whether the first start can make the data directory that does not exist yet.

    ``private.ensure_private_directory`` makes every missing level, so what matters is the nearest level that
    exists: this account has to be able to write to it and open it. The ancestry walk asks whether another
    account could replace a level, not whether this one can create below it, so a folder that is owner-owned
    and mode 0500 passes that walk and still makes the install fail. A file in the way never gets here: a path
    below one is not "missing", ``os.lstat`` raises ``NotADirectoryError`` for it, and that is its own finding.
    """
    anchor = next(parent for parent in data_dir.parents if os.path.lexists(parent))
    if os.access(anchor, os.W_OK | os.X_OK):
        return Finding(Level.INFO, f"The data directory {data_dir} does not exist yet; the first start makes it")
    return Finding(
        Level.FAIL,
        f"The data directory {data_dir} does not exist and cannot be made: {anchor} is the nearest folder "
        "that exists, and this account cannot write to it",
        fix="Make that a folder you can write to, or point USAGEDASH_DATA_DIR at one, then run "
        "`usagedash agent install`.",
    )


def data_directory_findings(settings: Settings) -> list[Finding]:
    """Check what ``agent install`` and the server require of the data directory, without creating it.

    The ancestors must not be replaceable by another account, and the directory itself must be a real
    folder this account owns that no other account can change: ``private.ensure_private_directory`` refuses a
    symbolic link or a file there and cannot make another account's folder private, and it tightens an owned
    folder's modes and access-control list only when it runs, not before. The ancestry walk does not look at
    that last entry, so "safe to use" is claimed only after ``what_lets_another_account_change`` finds nothing
    and ``what_lets_another_account_look`` finds nothing: the server makes the directory owner-only, with no
    group or other bits and no access-control list, so a folder that others can open is not as it leaves it.
    This account must also be able to write to it and open it, as the database and its journal need: a folder
    with only its owner's read bit has nothing wrong by either question and still stops collection. That is
    asked before whether others can look, because a failure must not be reported as the milder warning.
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
        return [_missing_data_directory_finding(data_dir)]
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
        visible = None if problem is not None else what_lets_another_account_look(data_dir)
    except ConfigError as error:
        return [Finding(Level.FAIL, f"The data directory cannot be examined: {error}", fix=cannot_examine)]
    make_owner_only = "Run `usagedash agent install`, which makes it owner-only again."
    if problem is not None:
        return [
            Finding(
                Level.FAIL,
                f"Another account can change the data directory {data_dir}: {problem}",
                ("It holds the database and the access token, so another account could replace them.",),
                make_owner_only,
            )
        ]
    if not os.access(data_dir, os.W_OK | os.X_OK):
        return [
            Finding(
                Level.FAIL,
                f"This account cannot write to the data directory {data_dir}",
                ("The database, its journal, and the access token live there, so collection fails.",),
                "Run `usagedash agent install`, which makes it writable by you alone again.",
            )
        ]
    if visible is not None:
        return [
            Finding(
                Level.WARN,
                f"Other accounts can look at the data directory {data_dir}: {visible}",
                (
                    "It holds the database and the access token. Each is owner-only on its own, but the folder's "
                    "listing and the sizes and times of what is in it are visible.",
                ),
                make_owner_only,
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

    A relative PATH entry is resolved from the directory this command runs in, as ``agent install`` resolves one
    from the shell that runs it. The agent's recorded PATH holds only absolute directories, so for it the
    directory makes no difference; for this shell's PATH, when nothing is installed, it is the whole answer.
    """
    binaries = settings.binaries
    return [
        _vendor_finding(name, executable, agent_path, Path.cwd())
        for name, executable in (("claude", binaries.claude), ("codex", binaries.codex), ("grok", binaries.grok))
    ]


def agent_helper_findings(settings: Settings, agent_path: str) -> list[Finding]:
    """Say whether the installed agent's collector can prepare the helper its command line tools need.

    ``path_findings`` asks what a fresh ``agent install`` from this shell would do. The agent that is installed
    uses its own recorded PATH and its own data directory, and may differ from this shell in both, so this makes
    the call its collector makes (``executables.execution_environment`` with the data directory as the helper
    directory, as ``rpc.JsonLineProcess`` does) in the mode that creates nothing.
    """
    try:
        with _logging_silenced():
            execution_environment(
                {"PATH": agent_path}, Path.cwd(), helper_directory=settings.data_dir, change_nothing=True
            )
    except ConfigError as error:
        return [
            Finding(
                Level.WARN,
                f"The installed agent cannot prepare a helper its command line tools need: {error}",
                ("The collector refuses to start a tool with this error, so its accounts show no data.",),
                "Fix what that names, then run `usagedash agent install`.",
            )
        ]
    return []


def _vendor_finding(name: str, executable: str, agent_path: str, working_directory: Path) -> Finding:
    """Resolve one CLI as ``rpc.JsonLineProcess`` does: screen the PATH, then ``executables.checked_command``.

    Those are the two calls the collector makes before it starts a CLI. The first drops every PATH directory
    that holds an entry another account could change, so a directory that has gained such an entry since the
    agent was installed does not supply the CLI here any more than it does when the collector asks. The second
    judges a bare name found on that PATH and a pin whose file has been removed (a Homebrew cask upgrade does
    that) as the agent will judge them, and not as a missing file. The check accepts a missing target and any
    file mode, so a dangling link or a file without the execute bit would pass it and still fail when the CLI
    is started. Helpers are not prepared here: that would create a folder, and ``path_findings`` asks whether
    install could prepare them.
    """
    pinned_file_is_gone = os.sep in executable and not os.path.lexists(executable)
    try:
        with _logging_silenced():
            environment = execution_environment({"PATH": agent_path}, working_directory)
            command = checked_command([executable], environment, working_directory, fallback_name=name)
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
    except OSError as error:
        return Finding(
            Level.WARN,
            f"{name}: could not examine the executable: {type(error).__name__}",
            ("The collector would fail to start it the same way.",),
            f"Check the permissions of {name}'s folder, then run `usagedash doctor` again.",
        )
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
    vendor_commands = [binaries.claude, binaries.codex, binaries.grok]
    try:
        omissions = agent.agent_path_omissions(vendor_commands)
        helper_problems = agent.helper_problems(vendor_commands, settings.data_dir)
    except OSError as error:
        return [Finding(Level.WARN, f"This shell's PATH could not be examined: {type(error).__name__}")]
    findings = [
        Finding(
            Level.WARN,
            f"`agent install` would refuse a helper Claude needs: {problem}",
            (
                "Claude reads its login from the keychain through `security`, which this PATH does not reach, so "
                "install supplies it from a private folder and checks it first.",
            ),
            "Fix what that names, then run `usagedash agent install`.",
        )
        for problem in helper_problems
    ]
    findings += [
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
    """Whether this account can bind ``port`` on the loopback address right now; the socket is closed at once.

    A socket that cannot even be created (no file descriptors left) is "not free": the answer only chooses a
    port to suggest, and an error here must not replace the report it is part of.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
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


class Relation(StrEnum):
    """How the process seen holding the port relates to the agent, when that can be told."""

    AGENT = "agent"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class AgentView:
    """What the agent checks learned: their findings, launchd's state, its environment, and what is known."""

    findings: list[Finding]
    state: ServiceState | None
    environment: dict[str, str] | None
    status: AgentStatus

    @property
    def pid(self) -> int | None:
        return None if self.state is None else self.state.pid


def _relation(holder: PortHolder | None, view: AgentView) -> Relation:
    """Relate the process seen holding the port to the agent, only as far as the evidence goes.

    Nobody seen is unknown, not another program: ``lsof`` shows only this account's processes. With the agent's
    pid known the pids compare. With the agent known to have no process, whoever holds the port is another.
    """
    if holder is None:
        return Relation.UNKNOWN
    if view.pid is not None:
        return Relation.AGENT if holder.pid == view.pid else Relation.OTHER
    if view.status is AgentStatus.STOPPED:
        return Relation.OTHER
    return Relation.UNKNOWN


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
        return AgentView(findings, None, installed, AgentStatus.UNKNOWN)
    install_fix = "Run `usagedash agent install` from the repository directory."
    if state is None:
        if plist_untrusted:
            return AgentView(findings, None, installed, AgentStatus.STOPPED)
        if installed is None:
            not_installed = Finding(Level.FAIL, "The agent is not installed", fix=install_fix)
            return AgentView([not_installed], None, None, AgentStatus.STOPPED)
        not_loaded = Finding(
            Level.FAIL,
            "The agent is installed but launchd has not loaded it",
            (f"property list: {paths.plist}",),
            install_fix,
        )
        return AgentView([not_loaded], None, installed, AgentStatus.STOPPED)
    if installed is None and not plist_untrusted:
        findings.append(
            Finding(
                Level.FAIL,
                "launchd lists an agent whose property list is gone",
                ("Which port, data folder, and CLIs it uses cannot be known without it.",),
                install_fix,
            )
        )
    findings.append(_running_finding(state))
    return AgentView(findings, state, installed, state.status)


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
    if state.status is AgentStatus.RUNNING:
        return Finding(Level.OK, f"launchd is running the agent (pid {state.pid})", facts)
    if state.status is AgentStatus.UNKNOWN:
        return Finding(Level.WARN, "launchd lists the agent, but its listing does not say whether it is running", facts)
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


def _answer_findings(port: int, view: AgentView) -> list[Finding]:
    """Say whether the dashboard answers, and from what the agent, the port, and the answer allow, who is at fault.

    Three inputs decide it: whether the agent has a process (``AgentStatus``), how the process seen holding the
    port relates to it (``Relation``), and what the health request got. Each is claimed only as far as it is
    known; an unknown input never counts as evidence for either side. The cases, for a request that got a 200:

    * the agent has no process: something else serves it (failure);
    * the agent runs and another process holds the port: failure;
    * the agent runs and the process holding the port is the agent: fine, and only this is fine;
    * anything else, the agent's state unknown or the holder not seen: the answer cannot be attributed (warning).

    For a refusal: the agent running means it is still starting or cannot bind; anything else, nothing listens.
    Unless a holder was seen: the request runs first and the lookup after, so a refusal and a listener are
    contradictory evidence (a program began listening between them), and "nothing is listening" is not claimed.
    For a stall, a failure, or an answer that is not a 200: the agent holds the port, restart it; another
    process does, or the agent has none, move the port; otherwise nobody can be blamed, so restart first and
    say what to try after.
    """
    health = probe_health(f"http://{LOOPBACK_HOST}:{port}{HEALTH_PATH}")
    holder = port_holder(port)
    relation = _relation(holder, view)
    detail = _holder_detail(port, holder)
    if health.kind is HealthKind.ANSWERED and health.status == HTTP_OK:
        if view.status is AgentStatus.STOPPED:
            return [_served_by_someone_else(port, holder, "the agent is not running, so something else is serving it")]
        if relation is Relation.OTHER:
            return [_served_by_someone_else(port, holder, f"the agent (pid {view.pid}) is not the one holding it")]
        if relation is Relation.AGENT:
            return [Finding(Level.OK, f"The dashboard answers on port {port}")]
        return [_answer_not_attributable(port, view, detail)]
    if health.kind is HealthKind.REFUSED and holder is not None:
        return [
            Finding(
                Level.FAIL,
                f"Port {port} refused the health request, but something is listening on it now",
                (*detail, "The request ran first, so a program began listening between the two checks."),
                "Run `usagedash doctor` again. If it ends the same way, read the log lines below.",
            )
        ]
    if health.kind is HealthKind.REFUSED and view.status is AgentStatus.RUNNING:
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
                detail,
                "Run `usagedash agent install`, then `usagedash doctor` again. If nothing listens after that, the "
                "log lines below say why.",
            )
        ]
    if relation is Relation.AGENT and view.pid is not None:
        return [
            Finding(
                Level.FAIL,
                _agent_trouble_title(view.pid, port, health),
                ("No other program is in the way, so a different port will not help.",),
                "Restart it with `usagedash agent install`. If this comes back, read the log lines below.",
            )
        ]
    if relation is Relation.OTHER or view.status is AgentStatus.STOPPED:
        return [Finding(Level.FAIL, _port_trouble_title(port, health), detail, _move_fix(port))]
    return [_unattributed_trouble(port, health)]


def _answer_not_attributable(port: int, view: AgentView, detail: tuple[str, ...]) -> Finding:
    """A healthy answer that cannot be tied to the agent: neither fine nor someone else's.

    Either launchd's state for the agent is unknown, or the agent runs and nobody could be seen holding the
    port. ``lsof`` shows only this account's processes and can fail, so an answer from a process doctor
    cannot see is not evidence that the agent gave it.
    """
    if view.status is AgentStatus.UNKNOWN:
        why = "launchd's state for the agent could not be read."
        fix = "Run `usagedash agent status`. If the agent is not running, run `usagedash agent install`."
    else:
        why = "Doctor could not tell which process holds the port."
        fix = f"If this is unexpected, run `lsof -nP -iTCP:{port} -sTCP:LISTEN` yourself to see who holds it."
    return Finding(
        Level.WARN,
        f"The dashboard answers on port {port}, but doctor could not tell whether the agent is serving it",
        (why, *detail),
        fix,
    )


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
        ("Doctor could not tell which process holds the port, so it cannot tell the agent from another program.",),
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
    """Show the last lines of the three logs the agent writes, when they hold anything.

    launchd captures standard error and standard output apart (a failure reported only on standard output
    would otherwise be missing), and the server writes its records to standard error as well as to its own
    file, so those usually repeat each other: a tail identical to one already shown is not shown again.
    """
    findings: list[Finding] = []
    shown: list[list[str]] = []
    standard_output, standard_error = paths.launchd_logs
    for log in (standard_error, standard_output, paths.server_log):
        lines = tail_lines(log)
        if not lines or lines in shown:
            continue
        findings.append(Finding(Level.INFO, f"Last lines of {log}", tuple(lines)))
        shown.append(lines)
    return findings


def _guarded(what: str, check: Callable[[], list[Finding]]) -> list[Finding]:
    """Run one group of checks; an exception it does not handle becomes a failure that names only its type.

    One check that raises must not hide the rest of the report, and the report is what a user pastes into an
    issue: an unexpected error's message can carry what a provider, a CLI, or a file supplied, so it is not
    used. The type is enough to find the call; the checks that did run still print.
    """
    try:
        return check()
    except Exception as error:  # noqa: BLE001 - the boundary of a report; see the docstring
        return [_check_failure(what, error)]


def _check_failure(what: str, error: Exception) -> Finding:
    return Finding(
        Level.FAIL,
        f"The {what} check could not finish: {type(error).__name__}",
        ("Every other check ran.",),
        "Run `usagedash doctor` again, and if it ends the same way, post this report in an issue.",
    )


def _agent_view(paths: AgentPaths) -> AgentView:
    """``_agent_findings``, or a view that knows nothing when it raises what no handler there expected.

    The status is unknown, never stopped: a check that could not finish has not shown that nothing is installed,
    so this shell's settings do not stand in for the agent's and no later check claims what it cannot know.
    """
    try:
        return _agent_findings(paths)
    except Exception as error:  # noqa: BLE001 - the boundary of a report; see ``_guarded``
        return AgentView([_check_failure("agent", error)], None, None, AgentStatus.UNKNOWN)


def diagnose_agent(paths: AgentPaths, port: int | None) -> list[Finding]:
    """Check the agent, whether it answers, and its logs. Used alone after an install or an open that failed."""
    view = _agent_view(paths)
    findings = list(view.findings)
    if port is not None:
        findings.extend(_guarded("port", lambda: _answer_findings(port, view)))
    findings.extend(_guarded("log", lambda: log_findings(paths)))
    return findings


def diagnose(paths: AgentPaths, *, shell: SettingsOutcome, served: SettingsOutcome) -> list[Finding]:
    """Run every check and return the findings in the order they should be read.

    ``shell`` is the settings this shell loads; ``served`` is what the installed agent was installed with.
    ``served`` wins for the port, the data directory, and the vendor CLIs, because those are what is actually
    running: the agent starts the executables pinned in its property list, not the ones this shell would find.
    This shell's settings stand in only when nothing is installed at all: no property list, and nothing loaded
    in launchd. When the installed settings cannot be read, or an agent is loaded without a property list to
    read them from, the port, data directory, and CLIs are not checked: checking this shell's would report on
    a different installation while the agent already started may still be running.
    The PATH check stays with ``shell``: it asks what ``agent install`` would do if run from here. The helper
    the installed agent's collector prepares is asked of ``served`` and its own recorded PATH.
    """
    findings = [
        *_guarded("system", system_findings),
        *_guarded("settings", lambda: settings_findings(shell, served)),
    ]
    view = _agent_view(paths)
    nothing_is_installed = served is None and view.state is None and view.status is AgentStatus.STOPPED
    effective = (
        served
        if isinstance(served, Settings)
        else shell
        if nothing_is_installed and isinstance(shell, Settings)
        else None
    )
    if effective is not None:
        findings.extend(_guarded("data directory", lambda: data_directory_findings(effective)))
        findings.extend(
            _guarded("command line tool", lambda: vendor_findings(effective, _agent_path(view.environment)))
        )
    if isinstance(served, Settings):
        findings.extend(_guarded("helper", lambda: agent_helper_findings(served, _agent_path(view.environment))))
    if isinstance(shell, Settings):
        findings.extend(_guarded("PATH", lambda: path_findings(shell)))
    findings.extend(view.findings)
    if effective is not None:
        findings.extend(_guarded("port", lambda: _answer_findings(effective.port, view)))
    if has_failure(findings):
        findings.extend(_guarded("log", lambda: log_findings(paths)))
    return findings


def _agent_path(environment: dict[str, str] | None) -> str:
    """The PATH the agent runs with: the one its property list records, or this shell's when none is installed.

    A property list that is there and records no PATH is an installed agent, and this shell's PATH is not its:
    the collector falls back to ``os.defpath`` for a missing PATH (``executables.execution_environment``), so
    that is what is judged, and a CLI that only this shell's PATH reaches is reported as the agent will find it.
    """
    if environment is None:
        return os.environ.get("PATH", os.defpath)
    return environment.get("PATH", os.defpath)


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
    """One line per title, detail, and remedy. Each field is made a single printable line first, because a title
    or a detail can carry a path or a message that this package did not write, and a newline in it would add
    a line of the report that nothing in the report produced.
    """
    lines: list[str] = []
    for finding in findings:
        lines.append(f"[{_LEVEL_LABEL[finding.level]}] {_single_line(finding.title)}")
        lines.extend(f"       {_single_line(detail)}" for detail in finding.details)
        if finding.fix is not None:
            lines.append(f"       -> {_single_line(finding.fix)}")
    return "\n".join(lines)


def _shareable(text: str, home: Path) -> str:
    """Screen the whole report line by line, then show the home folder as ``~``: users paste all of it into issues.

    Log lines are screened as they are read, but the report also echoes what an invalid setting was set to and
    the paths it names, and either can hold an email address or something that looks like a credential. The
    screen is idempotent, so a line screened twice reads the same, and it keeps the report's lines as they are.
    """
    return abbreviate_home(screen_for_sharing(text), home)


def render(findings: Sequence[Finding], home: Path) -> str:
    """Format findings as plain text, one block each, ending in a verdict, ready to paste into an issue."""
    return _shareable(f"{_blocks(findings)}\n\n{_verdict(findings)}", home)


def render_agent_failure(paths: AgentPaths, port: int, home: Path) -> str:
    """Format the agent checks alone, for the message after an install that found nothing answering."""
    return _shareable(_blocks(diagnose_agent(paths, port)), home)


def abbreviate_home(text: str, home: Path) -> str:
    """Replace the home folder with ``~`` where it is a whole path prefix, so pasted output names no account folder."""
    return re.sub(rf"{re.escape(str(home))}(?![\w-]|\.\w)", "~", text)
