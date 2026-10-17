"""Install, remove, and inspect the launchd agent that keeps the dashboard running in the user's session."""

import contextlib
import fcntl
import logging
import os
import plistlib
import pwd
import re
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.parsers.expat import ExpatError

from usagedash.errors import ConfigError, UsageDashError
from usagedash.executables import REQUIRED_PATH_HELPERS, with_required_helpers
from usagedash.executables import canonical_directory as _canonical_directory_only_this_account_can_change
from usagedash.executables import checked_executable as _checked_executable
from usagedash.private import (
    ensure_no_ancestor_lets_another_account_replace,
    ensure_only_owner_can_change,
    ensure_private_directory,
    open_private_file,
    read_file_only_owner_can_change,
)

LAUNCHCTL_BINARY = "/bin/launchctl"
LAUNCHCTL_TIMEOUT_SECONDS = 30
LAUNCHCTL_NO_SUCH_SERVICE = 113
"""What ``launchctl print`` exits with for a service launchd does not list (seen on macOS 27.0.1)."""
UNLOAD_WAIT_SECONDS = 45.0
UNLOAD_POLL_SECONDS = 0.25
SERVICE_NAME = "ai-usage-dashboard"
RESTART_THROTTLE_SECONDS = 30
SYSTEM_PATH_DIRECTORIES = ("/usr/bin", "/bin", "/usr/sbin", "/sbin")
_LOGGER = logging.getLogger(__name__)
STANDARD_OUTPUT_LOG_NAME = "launchd-stdout.log"
STANDARD_ERROR_LOG_NAME = "launchd-stderr.log"
# Why -P: with -m, Python puts the working directory first on its import path, and the agent's working
# directory is the data directory. A file named like this package, left there while others could still
# write to it, would be run in place of the dashboard, with its owner's access to every credential.
SERVE_ARGUMENTS = ("-P", "-m", "usagedash", "serve")
TEMPORARY_SUFFIX_BYTES = 8
_LABEL_NAME = re.compile(r"[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*")


PREVIOUS_AGENT_STILL_RUNNING = "the previous agent is still running"
PLIST_AS_IT_WAS = "the property list on disk is as it was before"


class AgentError(UsageDashError):
    """The launchd agent could not be installed, removed, restarted, or inspected."""


class AgentStillLoadedError(AgentError):
    """launchd still lists the service after the unload wait; the agent runs on as it was."""


@dataclass(slots=True)
class _Replaced:
    """What an install replaces, and whether launchd has been asked to change what it runs yet."""

    plist: bytes | None
    was_loaded: bool
    launchd_was_asked: bool = False


@dataclass(frozen=True, slots=True)
class AgentPaths:
    label: str
    plist: Path
    log_dir: Path

    @property
    def launchd_logs(self) -> tuple[Path, Path]:
        """The files launchd sends the agent's standard output and standard error to."""
        return (self.log_dir / STANDARD_OUTPUT_LOG_NAME, self.log_dir / STANDARD_ERROR_LOG_NAME)

    @property
    def domain_target(self) -> str:
        return f"gui/{os.getuid()}"

    @property
    def service_target(self) -> str:
        return f"{self.domain_target}/{self.label}"


def account_home() -> Path:
    """Return the account-database home, ignoring HOME so later commands find the installed plist.

    Raises:
        AgentError: if this account has no home directory there, or that directory is not absolute.
    """
    try:
        recorded = pwd.getpwuid(os.getuid()).pw_dir
    except KeyError as error:
        raise AgentError("the account home cannot be read") from error
    home = Path(recorded)
    if not home.is_absolute():
        raise AgentError("the account home is not an absolute path")
    return home


def agent_paths(home: Path | None = None, user: str | None = None) -> AgentPaths:
    """Locate plist/logs using the account-database name and home (unless home is supplied), never USER.

    Raises:
        AgentError: if the account name cannot be part of a launchd label and a file name, or the
            account home cannot be used.
    """
    home_dir = account_home() if home is None else home
    name = pwd.getpwuid(os.getuid()).pw_name if user is None else user
    if _LABEL_NAME.fullmatch(name) is None:
        raise AgentError("the account name cannot be used in a launchd label")
    label = f"com.{name}.{SERVICE_NAME}"
    return AgentPaths(
        label=label,
        plist=home_dir / "Library" / "LaunchAgents" / f"{label}.plist",
        log_dir=home_dir / "Library" / "Logs" / SERVICE_NAME,
    )


def agent_search_path(executables: Sequence[str], shell_path: str | None = None) -> str:
    """Build canonical PATH in shell, vendor, system order. Resolve relative shell entries and deduplicate.

    Omit empty, missing, colon-containing, unsafe or unexaminable directories with reasons. Check all
    regular files and retained links without opening them; permit self/root ownership and admin writes.
    Canonical paths prevent alias retargeting. Installation separately supplies required lost helpers.
    """
    shell_path = os.environ.get("PATH", "") if shell_path is None else shell_path
    resolved = (shutil.which(executable, path=shell_path) for executable in executables)
    needed = [str(Path(found).absolute().parent) for found in resolved if found is not None]
    in_shell_order = [str(Path(entry).absolute()) for entry in shell_path.split(os.pathsep) if entry]
    ordered: list[str] = []
    left_out: list[str] = []
    seen: set[str] = set()
    for directory in (*in_shell_order, *needed, *SYSTEM_PATH_DIRECTORIES):
        if directory in seen or os.pathsep in directory:
            continue
        seen.add(directory)
        try:
            canonical = _canonical_directory_only_this_account_can_change(Path(directory))
            if os.pathsep in str(canonical):
                raise ConfigError("its canonical path holds the PATH separator")
        except ConfigError as error:
            left_out.append(f"{directory}: {error}")
            continue
        if str(canonical) not in ordered:
            ordered.append(str(canonical))
    if left_out:
        _LOGGER.warning(
            "left out of the agent's PATH, being missing or open to change by another account: %s",
            "; ".join(left_out),
        )
    return os.pathsep.join(ordered)


def _checked_python_executable() -> str:
    """Check the interpreter's retained spelling and targets, preserving virtual-environment selection."""
    return _checked_executable(Path(sys.executable).absolute(), "Python executable")


def _with_required_helpers(search_path: str, working_directory: Path) -> str:
    return with_required_helpers(search_path, working_directory, REQUIRED_PATH_HELPERS)


def build_plist(
    paths: AgentPaths,
    python_executable: str,
    working_directory: Path,
    search_path: str,
    settings_overrides: Mapping[str, str],
) -> bytes:
    """Render a restarting python -P -m usagedash serve agent with the supplied settings environment.

    Raises:
        AgentError: if the property list would not be read back as it was written; a property list
            cannot carry every string exactly.
    """
    environment = {**settings_overrides, "PATH": search_path}
    standard_output, standard_error = paths.launchd_logs
    plist = _as_property_list(
        {
            "Label": paths.label,
            "ProgramArguments": [python_executable, *SERVE_ARGUMENTS],
            "RunAtLoad": True,
            "KeepAlive": True,
            "ThrottleInterval": RESTART_THROTTLE_SECONDS,
            "ProcessType": "Background",
            "WorkingDirectory": str(working_directory),
            "EnvironmentVariables": environment,
            "StandardOutPath": str(standard_output),
            "StandardErrorPath": str(standard_error),
        }
    )
    if plist is None:
        unwritable = _unwritable_variables(environment)
        cause = f"{', '.join(unwritable)} cannot be written in one" if unwritable else "a path cannot be written in one"
        raise AgentError(f"cannot build the agent's property list: {cause}")
    return plist


def _as_property_list(value: Mapping[str, object]) -> bytes | None:
    """Serialize only values that round-trip exactly; plistlib accepts XML-invalid characters and alters CRs."""
    try:
        serialized = plistlib.dumps(value)
        return serialized if plistlib.loads(serialized) == value else None
    except (ValueError, ExpatError):
        return None


def _unwritable_variables(environment: Mapping[str, str]) -> list[str]:
    """Name the variables a property list cannot carry exactly, by trying each one alone.

    Names are shown as Python literals, because what makes one unwritable is a character that should
    not be printed as it is.
    """
    return sorted(repr(name) for name, value in environment.items() if _as_property_list({name: value}) is None)


def _launchctl(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run launchctl and return its result, whatever its exit status.

    Raises:
        AgentError: if launchctl cannot be started or does not finish in time.
    """
    try:
        return subprocess.run(
            [LAUNCHCTL_BINARY, *arguments],
            capture_output=True,
            text=True,
            timeout=LAUNCHCTL_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise AgentError(f"launchctl {arguments[0]} could not be run: {type(error).__name__}") from error


def _prepare_plist_directory(directory: Path) -> None:
    """Validate ancestry/existing levels unchanged; create missing owner-only levels, refusing links and ACLs.

    Raises:
        AgentError: if a level cannot be examined, created, made private, or trusted.
    """
    try:
        ensure_no_ancestor_lets_another_account_replace(directory)
        if not _path_exists(directory):
            ensure_private_directory(directory)
        for level in (directory, directory.parent, directory.parent.parent):
            ensure_only_owner_can_change(level)
    except ConfigError as error:
        raise AgentError(str(error)) from error


def prepare_log_directory(log_dir: Path) -> None:
    """Create private logs after checking all ancestry; validate existing parent directories unchanged.

    Raises:
        ConfigError: if a level is unsafe or the log directory cannot be made private.
    """
    ensure_no_ancestor_lets_another_account_replace(log_dir)
    ensure_private_directory(log_dir)
    for level in (log_dir.parent, log_dir.parent.parent, log_dir.parent.parent.parent):
        ensure_only_owner_can_change(level)


@contextlib.contextmanager
def _one_change_at_a_time(paths: AgentPaths) -> Iterator[None]:
    """Hold an owner-only flock beside the plist from inspection through rollback; wait and release on exit.

    Raises:
        AgentError: if the plist directory cannot be created or the lock cannot be taken.
        ConfigError: if the lock cannot be opened privately.
    """
    lock = paths.plist.with_name(f".{paths.plist.name}.lock")
    _prepare_plist_directory(lock.parent)
    descriptor = open_private_file(lock)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as error:
            raise AgentError(f"cannot lock {lock}: {type(error).__name__}") from error
        yield
    finally:
        os.close(descriptor)


def is_loaded(paths: AgentPaths) -> bool:
    """Report presence; only launchd's no-such-service result proves absence.

    Raises:
        AgentError: if launchctl cannot be run, or fails in any way other than finding no such service.
    """
    result = _launchctl("print", paths.service_target)
    if result.returncode == 0:
        return True
    if result.returncode == LAUNCHCTL_NO_SUCH_SERVICE:
        return False
    raise AgentError(f"launchctl print failed ({result.returncode}): {result.stderr.strip()}")


def install(
    paths: AgentPaths,
    working_directory: Path,
    vendor_executables: Sequence[str],
    settings_overrides: Mapping[str, str],
) -> None:
    """Validate the interpreter, private data/logs, PATH and helpers before stopping anything.

    Replace under the installation lock; failed/interrupted launchd changes restore the prior plist/service
    and report recovery. Refuse linked capture files.

    Raises:
        ConfigError: if the interpreter, helpers, data, capture files or lock cannot be trusted.
        AgentError: if logs or plist preparation, locking, launchctl or service registration fails.
        KeyboardInterrupt: if interrupted during replacement; its message describes recovery.
    """
    python_executable = _checked_python_executable()
    try:
        prepare_log_directory(paths.log_dir)
    except ConfigError as error:
        # Why the same error as the property list's directory: the levels above both are the same
        # directories, checked the same way, and which check meets one first must not change the answer.
        raise AgentError(str(error)) from error
    ensure_private_directory(working_directory)
    for launchd_log in paths.launchd_logs:
        os.close(open_private_file(launchd_log))
    search_path = _with_required_helpers(agent_search_path(vendor_executables), working_directory)
    plist = build_plist(paths, python_executable, working_directory, search_path, settings_overrides)
    with _one_change_at_a_time(paths):
        replaced = _Replaced(plist=_installed_plist(paths), was_loaded=is_loaded(paths))
        try:
            _replace_installation(paths, plist, replaced)
        except KeyboardInterrupt as interrupt:
            raise KeyboardInterrupt(_put_back(paths, replaced)) from interrupt


def _replace_installation(paths: AgentPaths, plist: bytes, replaced: _Replaced) -> None:
    """Replace the plist and service transactionally, reporting rollback outcomes.

    Raises:
        AgentError: if the plist cannot be written, in which case nothing has changed, or if the old
            agent cannot be stopped or the new one started, in which case the previous installation
            has been put back and the message says what became of it.
    """
    _write_plist(paths, plist)
    try:
        # Why noted before launchd is asked and not after: an interrupt can arrive between launchd
        # acting and this function hearing of it, and what is put back depends on whether it acted.
        replaced.launchd_was_asked = True
        if replaced.was_loaded:
            _stop(paths)
        _start(paths)
    except AgentError as error:
        raise AgentError(f"{error}; {_put_back(paths, replaced)}") from error


def installed_environment(paths: AgentPaths) -> dict[str, str] | None:
    """Return string environment settings from the installed plist, or None when absent.

    Raises:
        AgentError: if a property list is present and cannot be read, or is not a property list,
            whichever way the parser fails.
    """
    raw = _installed_plist(paths)
    return None if raw is None else _environment_in(raw, paths)


def _environment_in(raw: bytes, paths: AgentPaths) -> dict[str, str]:
    """Return the string environment variables a property list's bytes hold.

    Raises:
        AgentError: if the bytes are not a property list.
    """
    try:
        loaded = plistlib.loads(raw)
    except (ValueError, ExpatError, AttributeError, IndexError, RecursionError) as error:
        raise AgentError(f"cannot read {paths.plist}: {type(error).__name__}") from error
    variables = loaded.get("EnvironmentVariables") if isinstance(loaded, dict) else None
    if not isinstance(variables, dict):
        return {}
    kept: dict[str, str] = {}
    for key, value in variables.items():
        if isinstance(value, str):
            kept[str(key)] = value
    return kept


def _plist_safe_to_start(paths: AgentPaths) -> bytes | None:
    """Return the installed property list when it is safe to start, or None when it is not.

    Nothing installed is None. A property list another account can change is also None: uninstall may
    still stop and delete it, and must not start it again.
    """
    try:
        return _installed_plist(paths)
    except AgentError:
        return None


def _installed_plist(paths: AgentPaths) -> bytes | None:
    """Read only an owner-trusted, unlinked plist for rollback, before any mutation; return None when absent.

    Raises:
        AgentError: if there is one and it cannot be read, or another account could have changed it.
    """
    try:
        return read_file_only_owner_can_change(paths.plist)
    except FileNotFoundError:
        return None
    except ConfigError as error:
        raise AgentError(str(error)) from error


def _write_plist(paths: AgentPaths, plist: bytes) -> None:
    """Atomically replace the plist or leave it unchanged; new and restored files are owner-only despite umask.

    Raises:
        AgentError: if it cannot be written in full or moved into place.
    """
    temporary = paths.plist.with_name(f".{paths.plist.name}.{secrets.token_hex(TEMPORARY_SUFFIX_BYTES)}.tmp")
    moved_into_place = False
    try:
        _prepare_plist_directory(paths.plist.parent)
        # Why an owner-only file and not one with the caller's default permissions: launchd refuses a
        # property list that others can write to (seen on macOS 27.0.1 for mode 666), and with a
        # permissive umask that is what a plain write creates. The refusal comes after the previous
        # agent was stopped, and the property list put back would be refused the same way.
        with os.fdopen(open_private_file(temporary), "wb") as handle:
            handle.write(plist)
        os.replace(temporary, paths.plist)
        moved_into_place = True
    except ConfigError as error:
        raise AgentError(f"cannot write {paths.plist}: {error}") from error
    except OSError as error:
        raise AgentError(f"cannot write {paths.plist}: {type(error).__name__}") from error
    finally:
        if not moved_into_place:
            # Why on every way out, an interrupt included: the temporary file is nothing anyone asked
            # for. Why suppressed: what is being reported is how the write ended. A temporary file that
            # cannot be removed is left behind; launchd does not load it, because its name does not
            # end in .plist.
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)


def _delete_plist(paths: AgentPaths) -> None:
    """Delete the property list on disk if there is one.

    Raises:
        AgentError: if it cannot be deleted.
    """
    try:
        paths.plist.unlink(missing_ok=True)
    except OSError as error:
        raise AgentError(f"cannot delete {paths.plist}: {type(error).__name__}") from error


def _path_exists(path: Path) -> bool:
    """Use lstat: dangling links are present; inspection failures are errors, never absence.

    Raises:
        AgentError: if the path cannot be examined.
    """
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise AgentError(f"cannot examine {path}: {type(error).__name__}") from error
    return True


def _stop(paths: AgentPaths) -> None:
    """Stop the service, checking a failed request immediately and polling an accepted request.

    Raises:
        AgentError: if launchctl fails while the service remains loaded or its state is unknown.
        AgentStillLoadedError: if an accepted stop does not finish within the deadline.
    """
    result = _launchctl("bootout", paths.service_target)
    if result.returncode != 0:
        if is_loaded(paths):
            raise AgentError(f"launchctl bootout failed ({result.returncode}): {result.stderr.strip()}")
        return
    _wait_until_unloaded(paths)


def _start(paths: AgentPaths) -> None:
    """Register the property list on disk with launchd, which starts the agent.

    Raises:
        AgentError: if launchctl cannot be run or launchd rejects the service.
    """
    result = _launchctl("bootstrap", paths.domain_target, str(paths.plist))
    if result.returncode != 0:
        raise AgentError(f"launchctl bootstrap failed ({result.returncode}): {result.stderr.strip()}")


def _start_again(paths: AgentPaths) -> str:
    """After failed plist deletion, restart from disk and return the outcome without replacing the original error."""
    try:
        _start(paths)
    except AgentError as error:
        return f"the agent could not be started again: {error}"
    return "the agent was started again"


def _put_back(paths: AgentPaths, replaced: _Replaced) -> str:
    """Restore a failed installation and return the recovery outcome without raising.

    After any launchd mutation, remove and await whatever service remains before restarting the trusted
    previous plist; it may be the new service or the old service still exiting.
    """
    try:
        if replaced.plist is None:
            _delete_plist(paths)
        else:
            _write_plist(paths, replaced.plist)
        if not replaced.launchd_was_asked:
            return PREVIOUS_AGENT_STILL_RUNNING if replaced.was_loaded else PLIST_AS_IT_WAS
        if is_loaded(paths):
            _stop(paths)
        if not replaced.was_loaded:
            return PLIST_AS_IT_WAS
        if replaced.plist is None:
            return "the previous agent had no property list on disk to start it from again"
        _start(paths)
    except AgentError as error:
        return f"the previous installation could not be put back: {error}"
    return "the previous agent was started again"


def _wait_until_unloaded(paths: AgentPaths) -> None:
    """Await actual service removal; bootout returns early and premature registration fails.

    Raises:
        AgentStillLoadedError: if the old instance is still present after ``UNLOAD_WAIT_SECONDS``.
        AgentError: if launchd cannot say whether it is.
    """
    deadline = time.monotonic() + UNLOAD_WAIT_SECONDS
    while is_loaded(paths):
        if time.monotonic() >= deadline:
            raise AgentStillLoadedError(f"the previous agent did not stop within {UNLOAD_WAIT_SECONDS:.0f} seconds")
        time.sleep(UNLOAD_POLL_SECONDS)


@contextlib.contextmanager
def holding_installation_lock(paths: AgentPaths) -> Iterator[None]:
    """Hold the installation lock from a read of what is installed through the change that follows.

    Raises:
        AgentError: if the property list's directory cannot be created or the lock cannot be taken.
        ConfigError: if the lock file cannot be opened as an owner-only file.
    """
    with _one_change_at_a_time(paths):
        yield


def kickstart_if_loaded(paths: AgentPaths) -> bool:
    """Restart the agent if launchd is running it. The caller holds ``holding_installation_lock``.

    Raises:
        AgentError: if launchd cannot say whether the agent is loaded, or refuses the restart.
    """
    if not is_loaded(paths):
        return False
    result = _launchctl("kickstart", "-k", paths.service_target)
    if result.returncode != 0:
        raise AgentError(f"launchctl kickstart failed ({result.returncode}): {result.stderr.strip()}")
    return True


def restart_if_loaded(paths: AgentPaths) -> bool:
    """Under the installation lock, inspect and restart a loaded agent; return whether it was loaded.

    Raises:
        AgentError: if launchd cannot say whether the agent is loaded, or refuses the restart, or the
            lock cannot be taken.
        ConfigError: if the lock file cannot be opened as an owner-only file.
    """
    with holding_installation_lock(paths):
        return kickstart_if_loaded(paths)


def uninstall(paths: AgentPaths) -> dict[str, str] | None:
    """Stop then delete the plist under its lock, preserving data/logs; return its trusted environment or None.

    Wait for confirmed absence. Reconcile uncertain stops, deletion failures and interrupts from the
    trusted original; untrusted copies stay stopped. AgentStillLoadedError leaves the running service alone.

    Raises:
        AgentError: if inspection, stopping, deletion or locking fails; includes recovery outcome.
        ConfigError: if the lock cannot be opened privately.
        KeyboardInterrupt: if interrupted after stopping was requested; includes recovery outcome.
    """
    with holding_installation_lock(paths):
        trusted = _plist_safe_to_start(paths)
        removed = _environment_of_trusted(trusted, paths)
        stopped = False
        launchd_was_asked = False
        try:
            if is_loaded(paths):
                launchd_was_asked = True
                try:
                    _stop(paths)
                except AgentStillLoadedError:
                    raise
                except AgentError as error:
                    # Why: launchctl can act and then time out, and the poll can fail after the stop, so
                    # whether the agent still runs is unknown; an interrupt here is settled the same way.
                    raise AgentError(f"{error}; {_restore_interrupted_uninstall(paths, trusted)}") from error
                stopped = True
            try:
                _delete_plist(paths)
            except AgentError as error:
                if not stopped:
                    raise
                try:
                    plist_remains = _path_exists(paths.plist)
                except AgentError as examination:
                    raise AgentError(f"{error}; {examination}; the agent was not started again") from error
                if not plist_remains:
                    raise
                if trusted is None:
                    raise AgentError(f"{error}; the agent was not started again") from error
                try:
                    _write_plist(paths, trusted)
                except AgentError as restore_error:
                    raise AgentError(f"{error}; the agent could not be started again: {restore_error}") from error
                raise AgentError(f"{error}; {_start_again(paths)}") from error
        except KeyboardInterrupt as interrupt:
            if not launchd_was_asked:
                raise
            raise KeyboardInterrupt(_restore_interrupted_uninstall(paths, trusted)) from interrupt
        return removed


def _environment_of_trusted(trusted: bytes | None, paths: AgentPaths) -> dict[str, str] | None:
    """Return the environment in the trusted property list, or None when there is none or it does not parse."""
    if trusted is None:
        return None
    try:
        return _environment_in(trusted, paths)
    except AgentError:
        return None


def _restore_interrupted_uninstall(paths: AgentPaths, trusted: bytes | None) -> str:
    """Reconcile after a requested stop, returning recovery details without raising.

    An absent plist means removal only after stopping any remaining service. Otherwise restore and start
    only the trusted pre-stop copy; without it, leave the service stopped.
    """
    try:
        plist_is_present = _path_exists(paths.plist)
        if is_loaded(paths):
            _stop(paths)
        if not plist_is_present:
            return "the agent was removed"
        if trusted is None:
            return "the agent was not started again"
        _write_plist(paths, trusted)
    except AgentError as error:
        return f"the previous installation could not be put back: {error}"
    return _start_again(paths)
