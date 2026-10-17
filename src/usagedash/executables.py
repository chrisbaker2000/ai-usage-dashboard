"""Validate and freeze vendor executable selection before starting a child."""

import logging
import os
import shutil
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from usagedash.errors import ConfigError
from usagedash.private import (
    ensure_no_ancestor_lets_another_account_replace,
    ensure_only_owner_can_change,
    ensure_private_directory,
    what_lets_another_account_change_by_path,
)

REQUIRED_PATH_HELPERS = (Path("/usr/bin/security"),)
_child_path_omissions_logged: set[str] = set()
_pinned_replacements_logged: set[str] = set()


def canonical_executable(executable: Path) -> Path:
    """Validate ancestry, retained links, self/root ownership, mode and ACLs without opening; return canonical path.
    Permit admin writes for Homebrew. Missing CLIs require a safe existing parent.
    Canonicalization prevents retargeting.

    Raises:
        ConfigError: if another account could change what runs or the path cannot be examined.
    """
    try:
        real = checked_link_target(executable)
        ensure_no_ancestor_lets_another_account_replace(real, administrators_may_write=True)
        try:
            os.lstat(real)
        except FileNotFoundError:
            ensure_only_owner_can_change(real.parent, root_may_own=True, administrators_may_write=True)
            return real
        problem = what_lets_another_account_change_by_path(real, root_may_own=True, administrators_may_write=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigError(f"cannot use {executable}: {type(error).__name__}") from error
    if problem is not None:
        raise ConfigError(
            f"cannot use {executable}: {problem}; it would run with this account's credentials, "
            "so make it writable by you alone, or name another in [binaries], and run the command again"
        )
    return real


def checked_link_target(path: Path) -> Path:
    """Validate executable selection ancestry and every link owner before resolving its target.

    Raises:
        ConfigError: if selection can be redirected or the link chain is cyclic/too long.
        OSError: if a path cannot be inspected.
    """
    for _ in range(40):
        ensure_no_ancestor_lets_another_account_replace(path, administrators_may_write=True, follow_owned_links=True)
        try:
            status = os.lstat(path)
        except FileNotFoundError:
            ensure_only_owner_can_change(path.parent.resolve(), root_may_own=True, administrators_may_write=True)
            return path.resolve()
        if status.st_uid not in (os.geteuid(), 0):
            raise ConfigError("the executable link target belongs to another account")
        if not stat.S_ISLNK(status.st_mode):
            return path.resolve()
        path = path.parent / os.readlink(path)
    raise ConfigError("the executable link chain is too long or cyclic")


def checked_command(
    command: Sequence[str],
    env: Mapping[str, str],
    cwd: Path,
    *,
    fallback_name: str | None = None,
) -> list[str]:
    """Resolve against the child's PATH/cwd, validate, and freeze the executable's canonical path.

    A recorded path that is no longer on disk is looked up by its own file name on that same PATH.
    When that name is absent and ``fallback_name`` is a different bare command, that command is
    looked up on the same PATH. Claude's version file and Grok's download are named for the release,
    so the command those installs put on PATH is ``claude`` or ``grok``. The recorded path wins while
    any entry is there, including a dangling link. Either replacement goes through the same canonical
    check as a path that is still present.

    Raises:
        ConfigError: if the command is empty, its executable can be changed by another account,
            ``fallback_name`` is not one file name, or a removed pin has neither its file name nor
            that command on the checked PATH.
        OSError: if a bare command is not on the checked PATH.
    """
    if not command:
        raise ConfigError("cannot start an empty command")
    if fallback_name is not None:
        _require_one_file_name(fallback_name)
    selected = command[0]
    replaced: Path | None = None
    if os.sep not in selected:
        executable = _find_on_path(selected, env, cwd)
    elif _is_absent(cwd / selected):
        replaced = cwd / selected
        executable = _replacement_for_absent_command(replaced, env, cwd, fallback_name)
    else:
        executable = selected
    resolved = canonical_executable(cwd / executable)
    if replaced is not None:
        _note_accepted_pin_replacement(replaced, executable)
    return [str(resolved), *command[1:]]


def _find_on_path(name: str, env: Mapping[str, str], cwd: Path) -> str:
    """Return the first executable file named ``name`` on the child's PATH.

    Relative PATH entries are resolved from ``cwd``. An absolute entry is not joined onto ``cwd``.

    Raises:
        FileNotFoundError: if no PATH entry holds an executable file of that name.
    """
    # Why: expanding cwd into a PATH string would reinterpret literal colons.
    for part in env.get("PATH", os.defpath).split(os.pathsep):
        candidate = cwd / part / name
        if os.access(candidate, os.X_OK) and candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(name)


def _is_absent(path: Path) -> bool:
    """Return whether ``path`` itself is not on disk. A dangling link is present."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    return False


def _require_one_file_name(name: str) -> None:
    """Refuse a replacement command that is not a single path component.

    Raises:
        ConfigError: if ``name`` is empty, ``.``, ``..``, or contains a path separator.
    """
    if name in {"", ".", ".."} or os.sep in name or os.pathsep in name or (os.altsep is not None and os.altsep in name):
        raise ConfigError("a replacement command must be one file name")


def _replacement_for_absent_command(
    selected: Path, env: Mapping[str, str], cwd: Path, fallback_name: str | None
) -> str:
    """Return the checked-PATH stand-in for a removed pin.

    The file name is tried first. ``fallback_name`` is used only when that name is absent and it
    names a different command, so a versioned pin can start ``claude`` or ``grok``.

    Raises:
        ConfigError: if neither name is an executable on the checked PATH.
    """
    try:
        return _find_on_path(selected.name, env, cwd)
    except FileNotFoundError:
        if fallback_name is None or fallback_name == selected.name:
            raise ConfigError(
                f"cannot use {selected}: FileNotFoundError, and {selected.name} is not on the checked PATH"
            ) from None
    try:
        return _find_on_path(fallback_name, env, cwd)
    except FileNotFoundError:
        raise ConfigError(
            f"cannot use {selected}: FileNotFoundError, and neither {selected.name} nor {fallback_name} "
            "is on the checked PATH"
        ) from None


def _note_accepted_pin_replacement(selected: Path, replacement: str) -> None:
    """Log one accepted stand-in for a removed pin. A rejected stand-in leaves the slot open."""
    spelled = str(selected)
    if spelled in _pinned_replacements_logged:
        return
    _pinned_replacements_logged.add(spelled)
    logging.getLogger(__name__).warning("pinned executable %s is missing; using %s", selected, replacement)


def canonical_directory(directory: Path) -> Path:
    """Check canonical ancestry, directory, every regular file and retained link. Missing directories fail;
    admin writes support Homebrew. Canonicalization prevents later directory-alias retargeting.

    Raises:
        ConfigError: if the directory or an executable entry cannot be trusted or examined.
    """
    try:
        real = directory.resolve(strict=True)
        ensure_no_ancestor_lets_another_account_replace(real, administrators_may_write=True)
        ensure_only_owner_can_change(real, root_may_own=True, administrators_may_write=True)
        _refuse_an_executable_entry_another_account_could_change(real)
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigError(f"cannot use {directory}: {type(error).__name__}") from error
    return real


def _refuse_an_executable_entry_another_account_could_change(directory: Path) -> None:
    with os.scandir(directory) as entries:
        for entry in entries:
            path = Path(entry.path)
            try:
                status = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(status.st_mode):
                    target = checked_link_target(path)
                    ensure_no_ancestor_lets_another_account_replace(target, administrators_may_write=True)
                    try:
                        os.lstat(target)
                    except FileNotFoundError:
                        ensure_only_owner_can_change(target.parent, root_may_own=True, administrators_may_write=True)
                        continue
                elif stat.S_ISREG(status.st_mode):
                    target = path
                else:
                    continue
                problem = what_lets_another_account_change_by_path(
                    target, root_may_own=True, administrators_may_write=True
                )
                if problem is not None:
                    raise ConfigError(problem)
            except (ConfigError, OSError, RuntimeError, ValueError) as error:
                reason = str(error) if isinstance(error, ConfigError) else type(error).__name__
                raise ConfigError(f"cannot use executable entry {path}: {reason}") from error


def checked_executable(executable: Path, purpose: str) -> str:
    """Validate an executable regular target; retain its spelling. Raise ConfigError on failure."""
    try:
        target = checked_link_target(executable)
        status = os.stat(target)
        if not stat.S_ISREG(status.st_mode) or not os.access(executable, os.X_OK):
            raise ConfigError(f"cannot use {purpose} {executable}: it is not an executable regular file")
        problem = what_lets_another_account_change_by_path(target, root_may_own=True, administrators_may_write=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigError(f"cannot use {purpose} {executable}: {type(error).__name__}") from error
    if problem is not None:
        raise ConfigError(f"cannot use {purpose} {executable}: {problem}")
    return str(executable)


def with_required_helpers(search_path: str, working_directory: Path, helpers: Sequence[Path]) -> str:
    """Expose checked required helpers privately; raise ConfigError for unsafe paths or preparation failures."""
    missing = [helper for helper in helpers if shutil.which(helper.name, path=search_path) is None]
    if not missing:
        return search_path
    targets = [Path(checked_executable(helper, "required helper")).resolve() for helper in missing]
    directory = working_directory / "agent-bin"
    try:
        ensure_no_ancestor_lets_another_account_replace(directory)
        try:
            directory.lstat()
        except FileNotFoundError:
            ensure_private_directory(directory)
        else:
            ensure_only_owner_can_change(directory)
        directory = directory.resolve(strict=True)
        if os.pathsep in str(directory):
            raise ConfigError("the required helper directory holds the PATH separator")
        for helper, target in zip(missing, targets, strict=True):
            link = directory / helper.name
            try:
                link.symlink_to(target)
            except FileExistsError:
                if Path(os.readlink(link)) != target:
                    raise ConfigError(f"cannot use required helper {link}: it names an unexpected target") from None
        canonical_directory(directory)
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigError(f"cannot prepare required helpers in {directory}: {type(error).__name__}") from error
    return os.pathsep.join(part for part in (search_path, str(directory)) if part)


def _leave_out_of_child_path(directory: Path, error: ConfigError) -> None:
    """Log one omission of ``directory`` for this process. The same directory is omitted on every start."""
    spelled = str(directory)
    if spelled in _child_path_omissions_logged:
        return
    _child_path_omissions_logged.add(spelled)
    logging.getLogger(__name__).warning("left out of the child PATH: %s: %s", directory, error)


def execution_environment(
    base: Mapping[str, str],
    cwd: Path,
    *,
    helper_directory: Path | None = None,
) -> dict[str, str]:
    """Copy env with canonical checked PATH; inspect all regular files/links, omitting unsafe or colon paths.
    Resolve relative entries from child cwd; use a non-directory sentinel when empty. Optional required
    helpers live only in private dashboard data.

    Raises:
        ConfigError: if required helpers cannot be prepared safely.
    """
    admitted: list[str] = []
    for part in base.get("PATH", os.defpath).split(os.pathsep):
        directory = cwd / part
        try:
            canonical = canonical_directory(directory)
            if os.pathsep in str(canonical):
                raise ConfigError("its canonical path holds the PATH separator")
        except ConfigError as error:
            _leave_out_of_child_path(directory, error)
            continue
        if str(canonical) not in admitted:
            admitted.append(str(canonical))
    search_path = os.pathsep.join(admitted)
    if helper_directory is not None and sys.platform == "darwin":
        search_path = with_required_helpers(search_path, helper_directory, REQUIRED_PATH_HELPERS)
    return {**base, "PATH": search_path or os.devnull}
