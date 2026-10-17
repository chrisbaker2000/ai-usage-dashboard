"""Create and register standing logins: extra CLI home directories that only this dashboard uses.

A CLI holds one login per home directory. Giving each additional account its own home lets every account be
read continuously, without switching the login the user actually works in. The vendor CLI performs the
login itself, in the user's terminal and browser; this module only chooses the directory and records it.
"""

import contextlib
import fcntl
import os
import re
import secrets
import stat
import subprocess
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

from usagedash.config import (
    CLAUDE_ACCOUNT_FILE_NAME,
    ClaudeSourceConfig,
    Settings,
    SourceMode,
    standing_source_table,
)
from usagedash.errors import ConfigError
from usagedash.executables import checked_command, checked_executable, checked_link_target, execution_environment
from usagedash.models import Provider
from usagedash.paths import directory_identity, same_directory
from usagedash.private import (
    PRIVATE_FILE_MODE,
    ensure_no_ancestor_lets_another_account_replace,
    ensure_no_other_account_can_redirect,
    ensure_only_owner_can_change,
    ensure_private_directory,
    open_private_file,
    read_file_only_owner_can_change,
    restrict_to_owner,
    what_lets_another_account_change,
)
from usagedash.providers import claude_cli, codex_cli, grok_cli

NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
TEMPORARY_SUFFIX_BYTES = 8

_LOGIN_ARGUMENTS: dict[Provider, tuple[str, ...]] = {
    Provider.CLAUDE: ("auth", "login"),
    Provider.CODEX: ("login",),
    Provider.GROK: ("login",),
}
_CLI_ENVIRONMENTS: dict[Provider, Callable[[Mapping[str, str], Path], dict[str, str]]] = {
    Provider.CLAUDE: claude_cli.cli_environment,
    Provider.CODEX: codex_cli.cli_environment,
    Provider.GROK: grok_cli.cli_environment,
}


def standing_home(settings: Settings, provider: Provider, name: str) -> Path:
    """Return the home directory for a named standing login.

    Raises:
        ConfigError: if the name is not a short lowercase slug.
    """
    if NAME_PATTERN.fullmatch(name) is None:
        raise ConfigError(
            f"login name must be 1-32 characters of a-z, 0-9, or '-', starting with a letter or digit: {name!r}"
        )
    return settings.standing_homes_dir / f"{provider.value}-{name}"


def _stored_as_this_home(listed: Path, home: Path) -> bool:
    """Compare collector directory identities, including aliases and paths through missing components/files."""
    return same_directory(directory_identity(listed), directory_identity(home))


def is_registered(settings: Settings, provider: Provider, home: Path) -> bool:
    """Match CLI-mode homes by collector identity, including case and missing components. Refuse unsafe retained
    spellings and user-owned links, naming the safe path to list.

    Raises:
        ConfigError: if the existing source is direct-mode, its spelling is redirectable, or either
            directory cannot be identified.
    """
    if provider is Provider.CLAUDE:
        listed = [
            (source.mode, source.config_dir)
            for source in settings.claude_sources
            if _stored_as_this_home(source.config_dir, home)
        ]
    else:
        sources = settings.codex_sources if provider is Provider.CODEX else settings.grok_sources
        listed = [(source.mode, source.home) for source in sources if _stored_as_this_home(source.home, home)]
    if any(mode is not SourceMode.CLI for mode, _spelling in listed):
        raise ConfigError(
            f"{home} is already listed in the config file as a {provider.value} login read in direct mode; "
            'set that entry\'s mode to "cli", or remove it, and run the command again'
        )
    for _mode, spelling in listed:
        _refuse_a_listed_spelling_another_account_can_redirect(spelling, home)
    return bool(listed)


def _refuse_a_listed_spelling_another_account_can_redirect(listed: Path, home: Path) -> None:
    """Check every retained path entry, including the leaf; polling resolves it again. Name the safe home on refusal.

    Raises:
        ConfigError: if an entry on the way could be replaced or retargeted by another account, or
            is a link this account owns.
    """
    try:
        ensure_no_other_account_can_redirect(listed)
    except ConfigError as error:
        if listed == home:
            raise
        raise ConfigError(
            f"{error}. The config file names {home} as {listed}; name it {home} and run the command again"
        ) from error


def _is_symbolic_link(path: Path) -> bool:
    """Report whether the entry at ``path`` is a symbolic link, without following one.

    Raises:
        ConfigError: if the entry cannot be examined. ``Path.is_symlink`` reports that failure as
            absence.
    """
    try:
        status = os.lstat(path)
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot examine {path}: {type(error).__name__}") from error
    return stat.S_ISLNK(status.st_mode)


def prepare_home(settings: Settings, home: Path) -> None:
    """Privatize parents before inspecting existing homes. Refuse unsafe ancestry/content and config redirects;
    retain only checked vendor output/helper links. Create missing levels; privatize accepted ones.

    Raises:
        ConfigError: if ancestry is replaceable, an existing home or entry is unsafe, or a directory
            cannot be created or made private.
    """
    ensure_no_ancestor_lets_another_account_replace(settings.data_dir)
    for directory in (settings.data_dir, settings.standing_homes_dir):
        ensure_private_directory(directory)
    _refuse_an_existing_home_another_account_could_write(home)
    ensure_private_directory(home)


def _refuse_an_existing_home_another_account_could_write(home: Path) -> None:
    try:
        os.lstat(home)
    except FileNotFoundError:
        return
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot examine {home}: {type(error).__name__}") from error
    try:
        ensure_only_owner_can_change(home)
        changeable = _entry_another_account_could_change(home)
    except ConfigError as error:
        raise ConfigError(f"{error}; {_REMOVE_THE_HOME}") from error
    if changeable is not None:
        raise ConfigError(f"cannot use {home}: {changeable}; {_REMOVE_THE_HOME}")


_REMOVE_THE_HOME = (
    "the standing home already exists and what it holds cannot be trusted, so remove it and run the command again"
)


def _entry_another_account_could_change(home: Path) -> str | None:
    """Return the first entry below ``home`` that another account could change, and what lets it, or None.

    Raises:
        ConfigError: if an entry or a directory below the home cannot be examined.
    """

    def refuse(error: OSError) -> None:
        raise ConfigError(f"cannot examine {home}: {type(error).__name__}") from error

    for directory, names, files in os.walk(home, onerror=refuse):
        for name in names + files:
            entry = Path(directory, name)
            if _is_symbolic_link(entry) and _checked_vendor_output_link(home, entry):
                continue
            problem = what_lets_another_account_change(entry)
            if problem is not None:
                return f"{entry}: {problem}"
    return None


def _checked_vendor_output_link(home: Path, entry: Path) -> bool:
    """Accept only private Claude debug/latest or trusted Codex tmp/arg0 helpers; other links stay refused.

    Raises:
        ConfigError: if a recognized link or its target is unsafe or unexaminable.
    """
    relative = entry.relative_to(home).parts
    debug = relative == ("debug", "latest")
    helper = (
        len(relative) == 4
        and relative[:2] == ("tmp", "arg0")
        and relative[-1] in {"apply_patch", "applypatch", "codex-execve-wrapper"}
    )
    if not (debug or helper):
        return False
    try:
        if os.lstat(entry).st_uid != os.geteuid():
            raise ConfigError(f"cannot use {entry}: the output link belongs to another account")
        if helper:
            checked_executable(entry, "vendor helper")
            return True
        target = checked_link_target(entry)
        if target.parent != (home / "debug").resolve():
            raise ConfigError(f"cannot use {entry}: the output link leaves its debug directory")
        ensure_no_ancestor_lets_another_account_replace(target)
        ensure_only_owner_can_change(target.parent)
        try:
            status = os.lstat(target)
        except FileNotFoundError:
            return True
        if not stat.S_ISREG(status.st_mode):
            raise ConfigError(f"cannot use {entry}: the output target is not a regular file")
        problem = what_lets_another_account_change(target)
        if problem is not None:
            raise ConfigError(f"cannot use {entry}: {problem}")
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigError(f"cannot examine {entry}: {type(error).__name__}") from error
    return True


# The settings with that path as the config file; a file that is not there holds nothing.
SettingsLoader = Callable[[Path], Settings]
"""Loads the settings as they are when the given file is the config file."""


def _config_file_is_present(config_path: Path) -> bool:
    """Use lstat: a missing config lists no login; inspection failures must not silently replace existing config.

    Raises:
        ConfigError: if the path cannot be examined.
    """
    try:
        os.lstat(config_path)
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot examine {config_path}: {type(error).__name__}") from error
    return True


def _settings_listed_now(settings: Settings, load: SettingsLoader) -> Settings:
    """Reload current config; absence uses environment-only settings, discarding removed entries/binary overrides.
    Inspection failures remain errors.
    """
    # Why ask and drop the answer: a file that cannot be examined must be an error, not taken as absent.
    _config_file_is_present(settings.config_path)
    return load(settings.config_path)


def _register_locked(settings: Settings, provider: Provider, home: Path, load: SettingsLoader) -> bool:
    """Under the caller's registration lock, reload and append home; return False if already listed."""
    candidate = _config_with_login(settings.config_path, provider, home, load)
    if candidate is None:
        return False
    replaced = False
    try:
        _move_into_place(candidate, settings.config_path)
        replaced = True
    finally:
        if not replaced:
            _discard(candidate)
    return True


def register(settings: Settings, provider: Provider, home: Path, load: SettingsLoader) -> bool:
    """Reload under one exclusive lock; validate a complete candidate beside config, then atomically replace.
    Return False when already registered.

    Raises:
        ConfigError: if home/config ancestry, privacy, locking, I/O, or replacement validation fails.
    """
    _refuse_replaceable_config_directory(settings.config_path)
    prepare_home(settings, home)
    with _registration_lock(settings.config_path):
        return _register_locked(settings, provider, home, load)


def sign_in_and_register(settings: Settings, provider: Provider, home: Path, load: SettingsLoader) -> int:
    """Reload, validate, sign in and register under one lock; return CLI status. Preserve Claude's keychain spelling.
    Refuse unregistrable/direct homes before launch; nonzero exits register nothing.

    Raises:
        ConfigError: if configuration, home preparation, executable validation, sign-in, or registration fails.
    """
    _refuse_replaceable_config_directory(settings.config_path)
    prepare_home(settings, home)
    with _registration_lock(settings.config_path):
        candidate: Path | None = None
        try:
            current = _settings_listed_now(settings, load)
            candidate = _config_with_login(settings.config_path, provider, home, load)
            if candidate is not None:
                _discard(candidate)
                candidate = None
            exit_status = run_login(current, provider, home)
        finally:
            if candidate is not None:
                _discard(candidate)
        if exit_status != 0:
            return exit_status
        _register_locked(settings, provider, home, load)
    return 0


def ensure_registrable(settings: Settings, provider: Provider, home: Path, load: SettingsLoader) -> None:
    """Check registration without changing the config, before the interactive sign-in.

    The config directory must already exist; prepare_home creates the default.

    Raises:
        ConfigError: if ancestry, loading, locking, reading, or candidate validation fails."""
    _refuse_replaceable_config_directory(settings.config_path)
    with _registration_lock(settings.config_path):
        candidate: Path | None = None
        try:
            candidate = _config_with_login(settings.config_path, provider, home, load)
        finally:
            if candidate is not None:
                _discard(candidate)


def _config_with_login(config_path: Path, provider: Provider, home: Path, load: SettingsLoader) -> Path | None:
    """Write and load-check a candidate beside config; return None if registered. Reject incompatible one-line lists
    and discard invalid candidates; require the candidate to contain this home.

    Raises:
        ConfigError: if loading, reading, writing, or candidate validation fails.
        KeyboardInterrupt: propagated after candidate cleanup.
    """
    exists = _config_file_is_present(config_path)
    # Why read before it is loaded: reading refuses a symbolic link and a file another account can
    # change, and only then makes an accepted file owner-only. Loading follows a link and changes
    # nothing. A config that already has the login is still the config, so it must not get past those
    # checks by needing no edit.
    try:
        existing = _config_content(config_path) if exists else b""
    except FileNotFoundError:
        # Why: the file went between the look and the read, and a file that is not there holds nothing.
        exists, existing = False, b""
    if exists and is_registered(load(config_path), provider, home):
        return None
    table = standing_source_table(provider.value, home)
    try:
        encoded_table = table.encode("utf-8")
    except UnicodeEncodeError as error:
        # Why this can happen: a path can hold bytes that are not text, which Python carries as lone
        # surrogates, and a TOML file is UTF-8.
        raise ConfigError(
            f"cannot add this login to {config_path}: the path of its home cannot be written as UTF-8 text"
        ) from error
    candidate = _write_beside(config_path, existing + encoded_table)
    accepted = False
    refusal: ConfigError | None = None
    try:
        try:
            accepted = is_registered(load(candidate), provider, home)
        except ConfigError as error:
            refusal = error
        if accepted:
            return candidate
        raise ConfigError(
            f"cannot add this login to {config_path}: the file would not load with a [[{provider.value}]] table "
            f"added to its end. If it lists its {provider.value} logins on one line, add this entry to that list "
            f"and run the command again: {_as_inline_table(table)}"
        ) from refusal
    finally:
        if not accepted:
            _discard(candidate)


def _as_inline_table(table: str) -> str:
    """Render an inline login table, splitting only LF so Unicode line-separator characters remain in paths."""
    _header, *fields = table.strip().split("\n")
    return "{ " + ", ".join(fields) + " }"


@contextlib.contextmanager
def _registration_lock(config_path: Path) -> Iterator[None]:
    """Lock a separate file beside config across replacement; locking the replaced inode would not serialize writers.

    Raises:
        ConfigError: if the lock file cannot be opened or locked.
    """
    descriptor = open_private_file(config_path.with_name(f"{config_path.name}.lock"))
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as error:
            raise ConfigError(f"cannot lock {config_path}: {type(error).__name__}") from error
        yield
    finally:
        os.close(descriptor)


def _config_content(config_path: Path) -> bytes:
    """Read bytes from one checked descriptor, then privatize. Refuse links/unsafe files unchanged; chmod cannot
    make previously untrusted bytes safe.

    Raises:
        FileNotFoundError: if there is nothing there.
        ConfigError: if another account can change it, or it cannot be read or made owner-only.
    """
    content = read_file_only_owner_can_change(config_path)
    _make_config_owner_only(config_path)
    return content


def _make_config_owner_only(config_path: Path) -> None:
    """Make a config the checked read accepted accessible to its owner only.

    Raises:
        ConfigError: if it cannot be opened without following a link, or cannot be restricted.
    """
    try:
        descriptor = os.open(config_path, os.O_RDWR | os.O_NOFOLLOW)
    except (OSError, ValueError) as error:
        reason = "it is a symbolic link" if os.path.islink(config_path) else type(error).__name__
        raise ConfigError(f"cannot use {config_path}: {reason}") from error
    try:
        restrict_to_owner(descriptor, PRIVATE_FILE_MODE)
    except OSError as error:
        raise ConfigError(f"cannot use {config_path}: {type(error).__name__}") from error
    finally:
        os.close(descriptor)


def _write_beside(config_path: Path, content: bytes) -> Path:
    """Write ``content`` in full to a new owner-only file beside the config, and return that file.

    Raises:
        ConfigError: if the file cannot be created or written in full. Nothing is left behind.
        KeyboardInterrupt: the file is removed, and the interrupt is not a ConfigError.
    """
    temporary = config_path.with_name(f".{config_path.name}.{secrets.token_hex(TEMPORARY_SUFFIX_BYTES)}.tmp")
    written = False
    try:
        with os.fdopen(open_private_file(temporary), "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        written = True
    except OSError as error:
        raise ConfigError(f"cannot write {config_path}: {type(error).__name__}") from error
    finally:
        if not written:
            _discard(temporary)
    return temporary


def _move_into_place(candidate: Path, config_path: Path) -> None:
    """Make ``candidate`` the config file, or leave the config exactly as it was.

    Raises:
        ConfigError: if it cannot be moved into place. The candidate is then removed.
        KeyboardInterrupt: the candidate is removed, and the interrupt is not a ConfigError.
    """
    replaced = False
    try:
        os.replace(candidate, config_path)
        replaced = True
    except OSError as error:
        raise ConfigError(f"cannot write {config_path}: {type(error).__name__}") from error
    finally:
        if not replaced:
            _discard(candidate)


def _discard(temporary: Path) -> None:
    """Remove a file written beside the config that is not going to become it."""
    # Why suppressed: this runs after the outcome is already decided. A temporary file that cannot be
    # removed is left behind, and holds nothing the config and this login's table do not.
    with contextlib.suppress(OSError):
        temporary.unlink(missing_ok=True)


def login_command(settings: Settings, provider: Provider) -> list[str]:
    binary = {
        Provider.CLAUDE: settings.binaries.claude,
        Provider.CODEX: settings.binaries.codex,
        Provider.GROK: settings.binaries.grok,
    }[provider]
    return [binary, *_LOGIN_ARGUMENTS[provider]]


def _registered_claude_source(settings: Settings, home: Path) -> ClaudeSourceConfig | None:
    """Return the standing Claude entry already listed for this home, if there is one."""
    listed = [
        source
        for source in settings.claude_sources
        if source.mode is SourceMode.CLI and _stored_as_this_home(source.config_dir, home)
    ]
    if not listed:
        return None
    return listed[0]


def _registered_claude_config_dir(settings: Settings, home: Path) -> Path | None:
    """Return the config spelling already listed for this Claude home, if a standing login lists one."""
    source = _registered_claude_source(settings, home)
    return None if source is None else source.config_dir


def login_environment(
    provider: Provider, home: Path, base: Mapping[str, str], *, credential_dir: Path | None = None
) -> dict[str, str]:
    """Drop vendor variables and use the prepared home. Claude's retained credential_dir spelling controls only
    the keychain hash; do not resolve it again. Refuse unsafe ancestry and user-owned directory links.

    Raises:
        ConfigError: if credential_dir can be redirected.
    """
    if provider is Provider.CLAUDE and credential_dir is not None:
        _refuse_a_listed_spelling_another_account_can_redirect(credential_dir, home)
        return claude_cli.cli_environment(base, home, credential_dir)
    return _CLI_ENVIRONMENTS[provider](base, home)


def _refuse_replaceable_config_directory(config_path: Path) -> None:
    """Check config ancestry unchanged before later creation; lstat distinguishes dangling links and inspection errors.

    Raises:
        ConfigError: if ancestry or directory ownership, permissions, ACLs, or inspection fails.
    """
    directory = config_path.parent
    try:
        directory.lstat()
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot use {directory}: {type(error).__name__}") from error
    else:
        ensure_only_owner_can_change(directory, root_may_own=True)
    ensure_no_ancestor_lets_another_account_replace(directory)


def _refuse_claude_account_file_the_sign_in_will_not_write(source: ClaudeSourceConfig, home: Path) -> None:
    """Require the stored home's unlinked .claude.json. Refuse redirectable ancestry and same-name files elsewhere,
    even hard links since replacement separates them. Reuse the matched home; preserve filesystem equivalence.

    Raises:
        ConfigError: if the listed file differs, is linked, or has unsafe retained ancestry.
    """
    listed = source.account_file
    stored_in = home if listed.parent == source.config_dir else directory_identity(listed.parent)
    entry = stored_in / listed.name
    if _is_symbolic_link(entry):
        raise ConfigError(
            f"{home} lists account_file {listed}, which is a symbolic link; "
            "point account_file at that home's .claude.json itself and run the command again"
        )
    if directory_identity(entry).name != CLAUDE_ACCOUNT_FILE_NAME or not same_directory(stored_in, home):
        raise ConfigError(
            f"{home} lists account_file {listed}, which this sign-in does not write; "
            "point account_file at that home's .claude.json, or remove it, and run the command again"
        )
    _refuse_a_listed_spelling_another_account_can_redirect(listed.parent, home)


def run_login(settings: Settings, provider: Provider, home: Path) -> int:
    """Run the vendor CLI's interactive login against the standing home and return its exit status.

    Raises:
        ConfigError: if the vendor CLI cannot be started, the config directory or a directory above
            it lets another account replace the file that names the binary, a listed Claude
            spelling has an ancestor another account can replace, or a listed Claude account file
            is not the ``.claude.json`` this sign-in writes in the prepared home.
    """
    _refuse_replaceable_config_directory(settings.config_path)
    command = login_command(settings, provider)
    credential_dir = None
    if provider is Provider.CLAUDE:
        source = _registered_claude_source(settings, home)
        if source is not None:
            _refuse_claude_account_file_the_sign_in_will_not_write(source, home)
            credential_dir = source.config_dir
    environment = login_environment(provider, home, os.environ, credential_dir=credential_dir)
    try:
        cwd = Path.cwd()
        environment = execution_environment(environment, cwd, helper_directory=settings.data_dir)
        completed = subprocess.run(
            checked_command(command, environment, cwd, fallback_name=provider.value),
            env=environment,
            check=False,
        )
    except OSError as error:
        raise ConfigError(f"could not start {command[0]}: {type(error).__name__}") from error
    return completed.returncode
