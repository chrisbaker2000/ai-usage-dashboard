"""Owner-only files and directories.

The data directory holds account emails, usage history, the dashboard's access token, and the standing
logins' own credential files, so nothing in it may be readable or replaceable by another account on this
Mac. This is the single definition of what "owner-only" means; every module that creates something there
imports it.

Owner-only is two things on macOS. The permission bits say who may do what, and so, separately, does an
access-control list: an entry on a directory can grant another account access whatever the bits say, and
can be inherited by everything created inside it. So a file or directory made private here has its
permission bits set and its access-control list removed, every time and not only when it is created.

A directory this package relies on but does not own the contents of is checked by the same two
measures instead, and refused rather than changed (``ensure_only_owner_can_change``).
"""

import contextlib
import ctypes
import errno
import functools
import grp
import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path

from usagedash.errors import ConfigError

PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
ACL_TYPE_EXTENDED = 0x00000100
"""``ACL_TYPE_EXTENDED`` from macOS ``<sys/acl.h>``: the kind of list that ``ls -le`` shows."""
MACOS = "darwin"
_NO_ACCESS_CONTROL_LISTS_HERE = frozenset({errno.ENOTSUP, errno.EOPNOTSUPP})
_NO_ACCESS_CONTROL_LIST = _NO_ACCESS_CONTROL_LISTS_HERE | {errno.ENOENT}
WRITABLE_BY_GROUP_OR_OTHERS = 0o022
ADMINISTRATORS_GROUP = "admin"


def _group_id(name: str) -> int | None:
    try:
        return grp.getgrnam(name).gr_gid
    except KeyError:
        return None


ADMINISTRATORS_GROUP_ID = _group_id(ADMINISTRATORS_GROUP)
READ_ONLY_RIGHTS = frozenset({"read", "execute", "readattr", "readextattr", "readsecurity", "synchronize"})
"""The rights an access-control entry can allow without letting anyone change what it is on.

These are the names ``acl_to_text`` gives, which are the file names for a directory's rights too:
"read" is list, "execute" is search, "write" is add_file, "append" is add_subdirectory.
"""


def ensure_private_directory(path: Path) -> None:
    """Create ``path`` if it does not exist and make it accessible to its owner only.

    That is done every time, not just at creation: ``mkdir`` leaves an existing directory's permissions
    as they were, and a directory another account can write to would let that account remove or replace
    the files inside it whatever their own modes are.

    Every directory that was missing when this call looked is made owner-only as well, whoever ends up
    creating it. ``mkdir`` with ``parents=True`` would give those the process's default mode, and a
    parent that others can write to lets them swap the private directory itself for one of their own.
    If something else creates such a directory first, it is still made owner-only here. Directories that
    already existed above ``path`` are left exactly as they are.

    ``path`` itself, and every level this call is responsible for, must be a real directory. A symbolic
    link is refused even when it points at one: the directory it names lies somewhere this module has
    not made private, where another account may be able to replace it.

    Raises:
        ConfigError: if ``path`` is not one the system accepts, a directory cannot be examined or
            created, or it cannot be made owner-only, for example because it belongs to another user,
            or a file or a symbolic link is in the way.
    """
    try:
        missing = _missing_levels(path)
        for directory in missing:
            with contextlib.suppress(FileExistsError):
                directory.mkdir(mode=PRIVATE_DIRECTORY_MODE)
            _make_owner_only(directory)
        if not missing:
            _make_owner_only(path)
    except (OSError, ValueError) as error:
        # Why ValueError too: a path holding a NUL character is refused by the system calls with that,
        # not with an OSError.
        raise ConfigError(f"cannot make {path} a private directory: {type(error).__name__}") from error


def _make_owner_only(directory: Path) -> None:
    """Make a real directory accessible to its owner only.

    Raises:
        OSError: if ``directory`` is anything but a real directory (``NotADirectoryError`` for a file or
            a symbolic link), or it cannot be made owner-only.
    """
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise NotADirectoryError(directory)
    # Why by path first: a directory whose owner cannot read it cannot be opened, and it must be opened
    # for its access-control list to be removed.
    directory.chmod(PRIVATE_DIRECTORY_MODE)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        restrict_to_owner(descriptor, PRIVATE_DIRECTORY_MODE)
    finally:
        os.close(descriptor)


def restrict_to_owner(descriptor: int, mode: int) -> None:
    """Make an open file or directory accessible to its owner only.

    Sets its permission bits to ``mode`` and removes its access-control list.

    Raises:
        OSError: if either cannot be done, for example because the file belongs to another user.
    """
    os.fchmod(descriptor, mode)
    _remove_access_control_list(descriptor)


def ensure_only_owner_can_change(
    directory: Path,
    *,
    root_may_own: bool = False,
    root_sticky_temp: bool = False,
    administrators_may_write: bool = False,
) -> None:
    """Refuse a directory that an account other than this one can change, and leave it as it is.

    For a directory this package must rely on but does not own the contents of, such as
    ``~/Library/LaunchAgents``, which holds other programs' agents: making it owner-only would change
    what they rely on. Only its owner can change it when it is a real directory, not a symbolic link;
    it belongs to this account; its permission bits let neither its group nor others write to it; and
    its access-control list allows no other account more than to read it. Entries that deny, or that
    allow only reading or searching, are harmless.

    ``root_may_own`` is for an ancestor this account does not control, such as ``/Users``. Root may own
    that directory. Group or other write, and an access-control entry that lets another account change
    it, are still refused. A directory owned by any other account is refused either way.

    ``root_sticky_temp`` is for the ancestor walk once it has reached a temp directory owned by root
    with the sticky bit set, such as ``/private/tmp`` after ``/tmp`` has been resolved. Other accounts
    can create entries there, and the sticky bit stops them replacing an entry they do not own, so the
    write bits are not a reason to refuse it. An access-control entry that lets another account change
    it is still refused. The flag does nothing unless the open directory is actually that. A directory
    this account owns is never treated that way.
    ``administrators_may_write`` accepts a group write bit when the group is the administrators'
    (``admin``), as on Homebrew's prefix: its members can become root, so that bit adds nothing. It is
    for the programs the agent runs, never for what this package keeps private.

    Raises:
        ConfigError: if it is not such a directory, or cannot be examined.
    """
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except (OSError, ValueError) as error:
        # Why the link is looked for separately: macOS refuses a link opened this way as not a
        # directory. Whatever is there now, the open has already refused it; this only words why.
        reason = "it is a symbolic link" if os.path.islink(directory) else type(error).__name__
        raise ConfigError(f"cannot use {directory}: {reason}") from error
    try:
        status = os.fstat(descriptor)
        problem = _what_lets_others_change(
            descriptor,
            status,
            root_may_own=root_may_own,
            root_sticky_temp=root_sticky_temp,
            administrators_may_write=administrators_may_write,
        )
    except OSError as error:
        raise ConfigError(f"cannot use {directory}: {type(error).__name__}") from error
    finally:
        os.close(descriptor)
    if problem is not None:
        raise ConfigError(
            f"cannot use {directory}: {problem}, so another account could replace what is in it; "
            "make it writable by you alone and run the command again"
        )


def what_lets_another_account_change(
    path: Path, *, root_may_own: bool = False, administrators_may_write: bool = False
) -> str | None:
    """Return what lets another account change the entry at ``path``, or None when nothing does.

    The entry is not followed: a symbolic link is itself the answer, since where it leads is decided
    elsewhere. A file or directory is opened without following and examined through that descriptor
    as ``ensure_only_owner_can_change`` examines a directory: owned by this account, or by root when
    ``root_may_own``, no group or other write bit, no access-control entry allowing another account
    more than reading.

    Raises:
        ConfigError: if the entry cannot be examined.
    """
    try:
        if stat.S_ISLNK(os.lstat(path).st_mode):
            return "it is a symbolic link"
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot examine {path}: {type(error).__name__}") from error
    try:
        return _what_lets_others_change(
            descriptor,
            os.fstat(descriptor),
            root_may_own=root_may_own,
            administrators_may_write=administrators_may_write,
        )
    except OSError as error:
        raise ConfigError(f"cannot examine {path}: {type(error).__name__}") from error
    finally:
        os.close(descriptor)


def what_lets_another_account_change_by_path(
    path: Path, *, root_may_own: bool = False, administrators_may_write: bool = False
) -> str | None:
    """Examine an entry without opening it, returning what lets another account change it.

    For executables in a checked PATH directory: a root-owned binary need not be readable.
    Links are refused; callers must check their targets and ancestry separately.

    Raises:
        ConfigError: if the entry or its access-control list cannot be examined.
    """
    try:
        status = os.lstat(path)
        if stat.S_ISLNK(status.st_mode):
            return "it is a symbolic link"
        return _what_lets_others_change_entry(
            status,
            lambda: _access_control_text_of(path),
            root_may_own=root_may_own,
            administrators_may_write=administrators_may_write,
        )
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot examine {path}: {type(error).__name__}") from error


def ensure_no_ancestor_lets_another_account_replace(
    path: Path, *, administrators_may_write: bool = False, follow_owned_links: bool = False
) -> None:
    """Refuse when a directory above ``path`` lets another account replace ``path``, and leave it as it is.

    Unlink is decided by the directory that contains the entry, so every lexical ancestor is checked
    up to the filesystem root. A directory owned by root is not a place to stop: a symbolic link above
    it can still be replaced. A root-owned symbolic link is followed one step, because a login keeps
    that path and writes through the link, and the directory that contains the link is still checked.
    Another account can unlink a root-owned entry from a non-sticky directory it can write. A link this
    account owns is refused and not followed. A root-owned sticky directory, such as ``/private/tmp``,
    is accepted when the entry below it is one no other account can claim: the sticky bit stops another
    account replacing an entry it does not own, but anyone can create a missing entry there, and a file
    another account owns can be swapped by that account for a link. ``path`` itself may be missing
    below such a directory, because creating it fails if another account gets there first. A directory
    this account owns is never given that pass, sticky bit or not. An ancestor is examined with
    ``lstat``. One that is not there is skipped, and so is a level below a file, and the file itself:
    nothing exists below a file, and the collector resolves such a path lexically
    (``paths.directory_identity``). Any other failure to examine an ancestor is an error, including where
    ``Path.exists`` or ``Path.is_symlink`` would report that failure as absence.
    ``follow_owned_links`` is only for retained executable paths: directory links owned by this account
    may be followed after checking their target and containing directory, just like root-owned links.
    Private dashboard paths keep the strict default.

    Raises:
        ConfigError: if an ancestor lets another account replace ``path``, or the walk cannot finish.
    """
    try:
        _walk_replaceable_ancestors(
            path,
            remaining=ANCESTOR_WALK_LIMIT,
            followed=set(),
            origin=path,
            administrators_may_write=administrators_may_write,
            follow_owned_links=follow_owned_links,
        )
    except ConfigError:
        raise
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot use {path}: {type(error).__name__}") from error


def ensure_no_other_account_can_redirect(path: Path) -> None:
    """Refuse when another account could make ``path``, as spelled, lead somewhere else, and leave it as it is.

    For a spelling that is kept and resolved again later, such as a standing home the config lists.
    ``ensure_no_ancestor_lets_another_account_replace`` asks whether the entry at ``path`` can be
    replaced; this asks as well whether where the spelling leads can be changed, so the last entry is
    examined like an ancestor. A link this account owns is refused: where it leads lies outside what
    this package made private, and the real path can always be listed instead. A root-owned link is
    followed and where it leads is examined the same way. A directory must be one only this account
    can change. An entry that is not there yet is skipped, and its ancestors are still examined.

    Raises:
        ConfigError: if an entry on the way lets another account change where ``path`` leads, or the
            walk cannot finish.
    """
    try:
        _walk_replaceable_ancestors(
            path / _LINK_TARGET_ENTRY, remaining=ANCESTOR_WALK_LIMIT, followed=set(), origin=path
        )
    except ConfigError:
        raise
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot use {path}: {type(error).__name__}") from error


def _walk_replaceable_ancestors(
    path: Path,
    *,
    remaining: int,
    followed: set[tuple[int, int]],
    origin: Path,
    claimable_below: bool = False,
    administrators_may_write: bool = False,
    follow_owned_links: bool = False,
) -> int:
    """Check ancestors of ``path`` and return the unused step budget.

    ``remaining`` is shared with the target of a root-owned link, so that target cannot start a new
    budget. ``followed`` holds links already followed, so a link that points at itself ends.
    ``claimable_below`` says whether the entry below the first ancestor could be created or replaced by
    another account; it is carried into the target of a followed link, where that entry would live.
    The walk starts from the absolute path: a relative one would stop at ``.`` and never look at the
    working directory or what is above it.
    """
    current = path.absolute()
    while True:
        if remaining <= 0:
            raise ConfigError(f"cannot use {origin}: its ancestry could not be checked")
        remaining -= 1
        parent = current.parent
        if parent == current:
            return remaining
        try:
            status = parent.lstat()
        except (FileNotFoundError, NotADirectoryError):
            # Why: nothing is here yet, so whoever can add entries to the directory above decides what is.
            claimable_below = True
            current = parent
            continue
        if stat.S_ISLNK(status.st_mode):
            identity = (status.st_dev, status.st_ino)
            if identity not in followed:
                followed.add(identity)
                if status.st_uid != 0 and not (follow_owned_links and status.st_uid == os.geteuid()):
                    ensure_only_owner_can_change(
                        parent, root_may_own=True, administrators_may_write=administrators_may_write
                    )
                else:
                    remaining = _walk_replaceable_ancestors(
                        _symlink_target(parent) / _LINK_TARGET_ENTRY,
                        remaining=remaining,
                        followed=followed,
                        origin=origin,
                        claimable_below=claimable_below,
                        administrators_may_write=administrators_may_write,
                        follow_owned_links=follow_owned_links,
                    )
            claimable_below = False
        elif stat.S_ISDIR(status.st_mode):
            root_sticky = status.st_uid == 0 and bool(status.st_mode & stat.S_ISVTX) and not claimable_below
            ensure_only_owner_can_change(
                parent,
                root_may_own=True,
                root_sticky_temp=root_sticky,
                administrators_may_write=administrators_may_write,
            )
            claimable_below = False
        else:
            # Why: nothing exists below a file, and only its owner (or root) can swap it for a link.
            claimable_below = status.st_uid not in (os.geteuid(), 0)
        current = parent


def _symlink_target(link: Path) -> Path:
    """Return the path ``link`` names, without following a link or collapsing ``..``."""
    named = Path(os.readlink(link))
    if named.is_absolute():
        return named
    return link.parent / named


READ_CHUNK_BYTES = 65536
ANCESTOR_WALK_LIMIT = 64
_LINK_TARGET_ENTRY = "entry"
"""Appended so a directory is examined itself, not only the directories above it.

That is the directory a followed link names, and the one a kept spelling leads to
(``ensure_no_other_account_can_redirect``).
"""


def read_file_only_owner_can_change(path: Path) -> bytes:
    """Read a file this package relies on but did not just write, refusing one another account can change.

    For a file read back as trusted input, such as an installed property list that is put back and
    started after a failed install: one another account can write to, or a symbolic link to a place it
    controls, could hold whatever that account wants run. It must be a regular file, not a link,
    belonging to this account, with no group or other write bit and no access-control entry allowing
    another account more than reading (``_what_lets_others_change``). It is checked and read through
    one open descriptor, so it cannot be swapped between the two, and opened without waiting, so a pipe
    put in its place is refused rather than read from. A file that passes is made owner-only on that
    descriptor before its bytes are returned: others could still read the paths it names. One that
    fails the check is left unchanged. Making it owner-only afterwards would keep bytes another
    account had already written.

    Raises:
        FileNotFoundError: if there is nothing there.
        ConfigError: if it is not such a file, or cannot be examined, made owner-only, or read.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as error:
        reason = "it is a symbolic link" if os.path.islink(path) else type(error).__name__
        raise ConfigError(f"cannot use {path}: {reason}") from error
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise ConfigError(f"cannot use {path}: it is not a regular file")
        problem = _what_lets_others_change(descriptor, status)
        if problem is not None:
            raise ConfigError(
                f"cannot use {path}: {problem}, so another account could change what it holds; "
                "make it writable by you alone and run the command again"
            )
        content = b"".join(iter(lambda: os.read(descriptor, READ_CHUNK_BYTES), b""))
        restrict_to_owner(descriptor, PRIVATE_FILE_MODE)
        return content
    except ConfigError:
        raise
    except OSError as error:
        raise ConfigError(f"cannot use {path}: {type(error).__name__}") from error
    finally:
        os.close(descriptor)


def _what_lets_others_change(
    descriptor: int,
    status: os.stat_result,
    *,
    root_may_own: bool = False,
    root_sticky_temp: bool = False,
    administrators_may_write: bool = False,
) -> str | None:
    """Return what lets an account other than this one change an open directory or file, or None if nothing does.

    ``root_sticky_temp`` skips the write-bit refusal only for a directory owned by root with the sticky
    bit set. The open directory is what is trusted, not the flag by itself.

    Raises:
        OSError: if its access-control list cannot be read.
    """
    return _what_lets_others_change_entry(
        status,
        lambda: _access_control_text(descriptor),
        root_may_own=root_may_own,
        root_sticky_temp=root_sticky_temp,
        administrators_may_write=administrators_may_write,
    )


def _what_lets_others_change_entry(
    status: os.stat_result,
    access_control_text: Callable[[], str],
    *,
    root_may_own: bool = False,
    root_sticky_temp: bool = False,
    administrators_may_write: bool = False,
) -> str | None:
    owner_is_root = root_may_own and status.st_uid == 0
    if status.st_uid != os.geteuid() and not owner_is_root:
        return "it belongs to another account"
    sticky_root_temp = root_sticky_temp and status.st_uid == 0 and bool(status.st_mode & stat.S_ISVTX)
    writable_by_others = bool(status.st_mode & WRITABLE_BY_GROUP_OR_OTHERS) and not sticky_root_temp
    if writable_by_others and not (administrators_may_write and _only_administrators_may_write(status)):
        return f"its permissions ({stat.S_IMODE(status.st_mode):04o}) let other accounts write to it"
    if access_entries_let_others_change(access_control_text(), status.st_uid):
        return "an access-control entry lets another account change it"
    return None


def _only_administrators_may_write(status: os.stat_result) -> bool:
    """Report whether the write bits let only the administrators' group write, whose members can become root anyway."""
    return not status.st_mode & stat.S_IWOTH and status.st_gid == ADMINISTRATORS_GROUP_ID


def _access_control_text(descriptor: int) -> str:
    """Return an open file's access-control list as ``acl_to_text`` writes it, or "" when it has none.

    Nothing is read on a system other than macOS, for the reason ``_remove_access_control_list`` gives.

    Raises:
        OSError: if the list cannot be read.
    """
    if sys.platform != MACOS:
        return ""
    library = _acl_library()
    access_control_list = library.acl_get_fd_np(descriptor, ACL_TYPE_EXTENDED)
    return _access_control_list_text(library, access_control_list)


def _access_control_text_of(path: Path) -> str:
    """Read an entry's ACL without opening it or following a symbolic link; raise OSError on failure."""
    if sys.platform != MACOS:
        return ""
    library = _acl_library()
    return _access_control_list_text(library, library.acl_get_link_np(os.fsencode(path), ACL_TYPE_EXTENDED))


def _access_control_list_text(library: ctypes.CDLL, access_control_list: int | None) -> str:
    if not access_control_list:
        code = ctypes.get_errno()
        if code in _NO_ACCESS_CONTROL_LIST:
            return ""
        raise OSError(code, os.strerror(code))
    try:
        length = ctypes.c_ssize_t()
        text = library.acl_to_text(access_control_list, ctypes.byref(length))
        if not text:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        try:
            return ctypes.string_at(text, length.value).decode("utf-8", "replace")
        finally:
            library.acl_free(text)
    finally:
        library.acl_free(access_control_list)


def access_entries_let_others_change(listed: str, owner: int) -> bool:
    """Report whether an access-control list, as ``acl_to_text`` writes it, lets anyone but ``owner`` change its file.

    Each entry is a line ``tag:uuid:name:id:kind[,flags]:rights``, after a header line that starts
    with ``!#``. An entry that denies changes nothing. One that allows more than ``READ_ONLY_RIGHTS``
    does, unless it names the owner's own user id. An entry that cannot be read that way, or of a kind
    other than allow or deny, is taken to allow what it lists: what it means cannot be told, and
    passing it would trust it.
    """
    for line in listed.splitlines():
        if not line.strip() or line.startswith("!#"):
            continue
        fields = line.split(":")
        if len(fields) != 6:
            return True
        tag, _uuid, _name, qualifier_id, kind, rights = fields
        if kind.split(",")[0] == "deny":
            continue
        if tag == "user" and qualifier_id == str(owner):
            continue
        if not set(rights.split(",")) <= READ_ONLY_RIGHTS:
            return True
    return False


@functools.cache
def _acl_library() -> ctypes.CDLL:
    """Return the system library with the access-control-list functions this module calls declared."""
    library = ctypes.CDLL(None, use_errno=True)
    library.acl_init.restype = ctypes.c_void_p
    library.acl_init.argtypes = [ctypes.c_int]
    library.acl_set_fd_np.restype = ctypes.c_int
    library.acl_set_fd_np.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    library.acl_get_fd_np.restype = ctypes.c_void_p
    library.acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
    library.acl_get_link_np.restype = ctypes.c_void_p
    library.acl_get_link_np.argtypes = [ctypes.c_char_p, ctypes.c_int]
    library.acl_to_text.restype = ctypes.c_void_p
    library.acl_to_text.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ssize_t)]
    library.acl_free.restype = ctypes.c_int
    library.acl_free.argtypes = [ctypes.c_void_p]
    return library


def _remove_access_control_list(descriptor: int) -> None:
    """Replace an open file's or directory's access-control list with an empty one.

    Python has no call for this, so the system library is used directly. A file system that has no
    access-control lists says so, and has none to remove. Nothing is done on a system other than macOS:
    the lists there are a different mechanism, and this package does not otherwise run there (it relies
    on launchd and the keychain).

    Raises:
        OSError: if the list cannot be replaced.
    """
    if sys.platform != MACOS:
        return
    library = _acl_library()
    empty_list = library.acl_init(0)
    if not empty_list:
        raise OSError(ctypes.get_errno(), "cannot allocate an access-control list")
    try:
        if library.acl_set_fd_np(descriptor, empty_list, ACL_TYPE_EXTENDED) != 0:
            code = ctypes.get_errno()
            if code not in _NO_ACCESS_CONTROL_LISTS_HERE:
                raise OSError(code, os.strerror(code))
    finally:
        library.acl_free(empty_list)


def open_private_file(path: Path) -> int:
    """Open ``path`` for reading and writing, creating it if absent, accessible to its owner only.

    That is done on the open file every time, not just when it is created, for the same reason as for a
    directory: creating a file that already exists leaves its permissions as they were.

    A symbolic link at ``path`` is refused, whether or not it points at anything. Following it would
    put the file's contents wherever the link says, outside the directory that was made private, and
    whoever can write to that place could replace the file between this call and the next open.

    Returns:
        The open file descriptor. The caller closes it.

    Raises:
        ConfigError: if ``path`` is not one the system accepts, or the file cannot be opened or made
            owner-only, for example because a directory or a symbolic link is in the way or the file
            belongs to another user.
    """
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, PRIVATE_FILE_MODE)
        try:
            restrict_to_owner(descriptor, PRIVATE_FILE_MODE)
        except OSError:
            os.close(descriptor)
            raise
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot open {path} as a private file: {type(error).__name__}") from error
    return descriptor


def _missing_levels(path: Path) -> list[Path]:
    """Return ``path`` and each ancestor that does not exist yet, outermost first.

    Raises:
        OSError: if an ancestor cannot be examined.
    """
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    return list(reversed(missing))
