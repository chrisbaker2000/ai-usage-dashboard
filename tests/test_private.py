import ctypes
import errno
import os
import pwd
import re
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from usagedash import private
from usagedash.db import Database
from usagedash.errors import ConfigError
from usagedash.private import (
    PRIVATE_DIRECTORY_MODE,
    PRIVATE_FILE_MODE,
    access_entries_let_others_change,
    ensure_no_ancestor_lets_another_account_replace,
    ensure_no_other_account_can_redirect,
    ensure_only_owner_can_change,
    ensure_private_directory,
    open_private_file,
    read_file_only_owner_can_change,
)

PERMISSION_BITS = 0o777


def mode_of(path: Path) -> int:
    return path.stat().st_mode & PERMISSION_BITS


def _report_root_ownership(monkeypatch: pytest.MonkeyPatch, paths: set[Path]) -> None:
    """Report ``paths`` as owned by root. The open-inode check still sees the real owner."""
    real_lstat = Path.lstat

    def lstat(self: Path) -> Any:
        status = real_lstat(self)
        if self not in paths:
            return status
        return SimpleNamespace(st_uid=0, st_mode=status.st_mode, st_dev=status.st_dev, st_ino=status.st_ino)

    monkeypatch.setattr(Path, "lstat", lstat)


class TestReadingAFileOnlyItsOwnerCanChange:
    def test_an_owner_only_file_is_read_whole(self, tmp_path: Path) -> None:
        held = tmp_path / "held"
        held.write_bytes(b"x" * 200_000)
        held.chmod(0o600)

        assert read_file_only_owner_can_change(held) == b"x" * 200_000
        assert mode_of(held) == 0o600

    def test_a_file_others_can_read_is_made_owner_only_and_then_read(self, tmp_path: Path) -> None:
        """Mode 0644 is trustworthy, and it still publishes the paths the file names."""
        held = tmp_path / "config.toml"
        held.write_bytes(b"port = 9001\n")
        held.chmod(0o644)

        assert read_file_only_owner_can_change(held) == b"port = 9001\n"

        assert mode_of(held) == 0o600

    def test_a_file_other_accounts_can_write_is_refused_and_keeps_its_mode(self, tmp_path: Path) -> None:
        held = tmp_path / "config.toml"
        held.write_bytes(b"port = 9001\n")
        held.chmod(0o666)

        with pytest.raises(ConfigError, match="let other accounts write"):
            read_file_only_owner_can_change(held)

        assert mode_of(held) == 0o666
        assert held.read_bytes() == b"port = 9001\n"

    def test_bytes_are_not_returned_when_the_file_cannot_be_made_owner_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        held = tmp_path / "config.toml"
        held.write_bytes(b"port = 9001\n")
        held.chmod(0o644)

        def refuse(_descriptor: int, _mode: int) -> None:
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr(private.os, "fchmod", refuse)

        with pytest.raises(ConfigError, match="PermissionError"):
            read_file_only_owner_can_change(held)

        assert mode_of(held) == 0o644
        assert held.read_bytes() == b"port = 9001\n"

    def test_nothing_there_is_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_file_only_owner_can_change(tmp_path / "missing")

    def test_a_directory_or_a_pipe_is_not_a_regular_file(self, tmp_path: Path) -> None:
        pipe = tmp_path / "pipe"
        os.mkfifo(pipe)

        with pytest.raises(ConfigError, match="it is not a regular file"):
            read_file_only_owner_can_change(pipe)
        with pytest.raises(ConfigError, match="it is not a regular file"):
            read_file_only_owner_can_change(tmp_path)

    def test_a_path_that_cannot_be_opened_is_a_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="ValueError"):
            read_file_only_owner_can_change(tmp_path / "nul\x00here")

    def test_a_file_whose_access_control_list_cannot_be_read_is_a_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        held = tmp_path / "held"
        held.write_bytes(b"x")

        def unreadable(_descriptor: int) -> str:
            raise OSError("synthetic")

        monkeypatch.setattr(private, "_access_control_text", unreadable)

        with pytest.raises(ConfigError, match=r"cannot use .*held: OSError$"):
            read_file_only_owner_can_change(held)

    def test_a_file_that_cannot_be_read_is_a_config_error(self, tmp_path: Path) -> None:
        held = tmp_path / "held"
        held.write_bytes(b"x")
        held.chmod(0)
        try:
            with pytest.raises(ConfigError, match="PermissionError"):
                read_file_only_owner_can_change(held)
        finally:
            held.chmod(0o600)


class TestPrivateDirectory:
    def test_the_modes_grant_nothing_to_group_or_others(self) -> None:
        assert (PRIVATE_DIRECTORY_MODE, PRIVATE_FILE_MODE) == (0o700, 0o600)

    def test_creates_a_missing_directory_and_its_parents_owner_only(self, tmp_path: Path) -> None:
        directory = tmp_path / "parent" / "data"

        ensure_private_directory(directory)

        assert directory.is_dir()
        assert mode_of(directory) == 0o700

    @pytest.mark.parametrize("existing_mode", [0o777, 0o755, 0o770, 0o750, 0o701])
    def test_tightens_an_existing_directory_other_accounts_could_reach(
        self, tmp_path: Path, existing_mode: int
    ) -> None:
        directory = tmp_path / "data"
        directory.mkdir()
        directory.chmod(existing_mode)

        ensure_private_directory(directory)

        assert mode_of(directory) == 0o700

    def test_leaves_an_already_private_directory_and_its_contents_alone(self, tmp_path: Path) -> None:
        directory = tmp_path / "data"
        ensure_private_directory(directory)
        (directory / "usage.db").write_text("kept")

        ensure_private_directory(directory)

        assert mode_of(directory) == 0o700
        assert (directory / "usage.db").read_text() == "kept"

    def test_a_symbolic_link_to_a_directory_is_refused_and_the_directory_it_names_is_left_alone(
        self, tmp_path: Path
    ) -> None:
        """What the link names lies somewhere that was never made private, where it could be replaced."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        elsewhere.chmod(0o755)
        link = tmp_path / "data"
        link.symlink_to(elsewhere, target_is_directory=True)

        with pytest.raises(ConfigError, match=r"cannot make .* a private directory: NotADirectoryError"):
            ensure_private_directory(link)

        assert mode_of(elsewhere) == 0o755

    def test_a_dangling_symbolic_link_is_refused_and_nothing_is_created_where_it_points(self, tmp_path: Path) -> None:
        link = tmp_path / "data"
        link.symlink_to(tmp_path / "not-there", target_is_directory=True)

        with pytest.raises(ConfigError, match=r"cannot make .* a private directory"):
            ensure_private_directory(link)

        assert not (tmp_path / "not-there").exists()

    def test_a_symbolic_link_above_the_directory_is_followed_as_any_existing_ancestor_is(self, tmp_path: Path) -> None:
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        (tmp_path / "linked-parent").symlink_to(real_parent, target_is_directory=True)

        ensure_private_directory(tmp_path / "linked-parent" / "data")

        assert mode_of(real_parent / "data") == 0o700

    def test_a_file_in_the_way_is_a_config_error(self, tmp_path: Path) -> None:
        in_the_way = tmp_path / "data"
        in_the_way.write_text("not a directory")

        with pytest.raises(ConfigError, match=r"cannot make .* a private directory: NotADirectoryError"):
            ensure_private_directory(in_the_way)

    def test_every_directory_it_has_to_create_is_owner_only_whatever_the_process_default(self, tmp_path: Path) -> None:
        """With a permissive umask, ``mkdir(parents=True)`` would leave the new parents open to everyone."""
        existing = tmp_path / "existing"
        existing.mkdir()
        existing.chmod(0o755)
        leaf = existing / "created" / "also-created" / "data"
        previous = os.umask(0)
        try:
            ensure_private_directory(leaf)
        finally:
            os.umask(previous)

        assert [mode_of(path) for path in (leaf, leaf.parent, leaf.parent.parent)] == [0o700, 0o700, 0o700]
        assert mode_of(existing) == 0o755

    def test_a_file_where_a_parent_should_be_is_a_config_error(self, tmp_path: Path) -> None:
        (tmp_path / "parent").write_text("not a directory")

        with pytest.raises(ConfigError, match=r"cannot make .* a private directory"):
            ensure_private_directory(tmp_path / "parent" / "data")

    def test_callers_creating_the_same_directory_at_once_all_succeed(self, tmp_path: Path) -> None:
        callers = 8
        together = threading.Barrier(callers)
        leaf = tmp_path / "parent" / "data"

        def create(_: int) -> None:
            together.wait(timeout=5)
            ensure_private_directory(leaf)

        with ThreadPoolExecutor(max_workers=callers) as pool:
            list(pool.map(create, range(callers)))

        assert mode_of(leaf) == 0o700

    def appearing_after_the_probe(self, monkeypatch: pytest.MonkeyPatch, create: Callable[[Path], None]) -> None:
        """Make each missing directory appear, by ``create``, between the probe and this call's own mkdir."""
        probe = private._missing_levels

        def probe_then_something_else_creates_them(path: Path) -> list[Path]:
            missing = probe(path)
            for directory in missing:
                create(directory)
            return missing

        monkeypatch.setattr(private, "_missing_levels", probe_then_something_else_creates_them)

    def test_a_parent_something_else_creates_first_with_open_permissions_is_still_made_owner_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A writable parent lets another account replace the private directory inside it."""

        def create_open_to_everyone(directory: Path) -> None:
            directory.mkdir()
            directory.chmod(0o777)

        leaf = tmp_path / "parent" / "data"
        self.appearing_after_the_probe(monkeypatch, create_open_to_everyone)

        ensure_private_directory(leaf)

        assert [mode_of(leaf.parent), mode_of(leaf)] == [0o700, 0o700]

    def test_a_symbolic_link_that_appears_where_a_parent_should_be_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        elsewhere.chmod(0o755)
        leaf = tmp_path / "parent" / "data"

        def link_the_parent_elsewhere(directory: Path) -> None:
            if directory == leaf.parent:
                directory.symlink_to(elsewhere, target_is_directory=True)

        self.appearing_after_the_probe(monkeypatch, link_the_parent_elsewhere)

        with pytest.raises(ConfigError, match=r"cannot make .* a private directory: NotADirectoryError"):
            ensure_private_directory(leaf)

        assert mode_of(elsewhere) == 0o755
        assert list(elsewhere.iterdir()) == []

    def test_a_file_that_appears_where_a_parent_should_be_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        leaf = tmp_path / "parent" / "data"

        def put_a_file_at_the_parent(directory: Path) -> None:
            if directory == leaf.parent:
                directory.write_text("not a directory")

        self.appearing_after_the_probe(monkeypatch, put_a_file_at_the_parent)

        with pytest.raises(ConfigError, match=r"cannot make .* a private directory: NotADirectoryError"):
            ensure_private_directory(leaf)

    def test_an_ancestor_that_cannot_be_examined_is_a_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(path: Path) -> list[Path]:
            raise PermissionError(13, "Permission denied", str(path.parent))

        monkeypatch.setattr(private, "_missing_levels", refuse)

        with pytest.raises(ConfigError, match=r"cannot make .* a private directory: PermissionError"):
            ensure_private_directory(tmp_path / "parent" / "data")

    def test_the_search_for_missing_levels_ends_at_the_root_even_if_nothing_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "exists", lambda _path: False)

        assert private._missing_levels(Path("/a/b")) == [Path("/"), Path("/a"), Path("/a/b")]

    def test_the_missing_levels_are_listed_outermost_first_and_stop_at_what_exists(self, tmp_path: Path) -> None:
        existing = tmp_path / "existing"
        existing.mkdir()
        leaf = existing / "a" / "b" / "data"

        assert private._missing_levels(leaf) == [existing / "a", existing / "a" / "b", leaf]
        assert private._missing_levels(existing) == []

    def test_a_directory_that_cannot_be_created_is_a_config_error(self, tmp_path: Path) -> None:
        read_only = tmp_path / "read-only"
        read_only.mkdir()
        read_only.chmod(0o500)

        with pytest.raises(ConfigError, match=r"cannot make .* a private directory: PermissionError"):
            ensure_private_directory(read_only / "data")


class TestPrivateFile:
    def test_creates_a_missing_file_owner_only_whatever_the_process_default(self, tmp_path: Path) -> None:
        path = tmp_path / "usage.db"
        previous = os.umask(0)
        try:
            descriptor = open_private_file(path)
        finally:
            os.umask(previous)
        os.close(descriptor)

        assert mode_of(path) == 0o600

    @pytest.mark.parametrize("existing_mode", [0o666, 0o644, 0o640, 0o604])
    def test_tightens_an_existing_file_other_accounts_could_reach_and_keeps_its_contents(
        self, tmp_path: Path, existing_mode: int
    ) -> None:
        path = tmp_path / "access-token"
        path.write_text("kept")
        path.chmod(existing_mode)

        os.close(open_private_file(path))

        assert mode_of(path) == 0o600
        assert path.read_text() == "kept"

    def test_the_descriptor_it_returns_can_be_read_and_written(self, tmp_path: Path) -> None:
        descriptor = open_private_file(tmp_path / "collector.lock")
        try:
            os.write(descriptor, b"held")
            os.lseek(descriptor, 0, os.SEEK_SET)
            assert os.read(descriptor, 4) == b"held"
        finally:
            os.close(descriptor)

    def test_a_symbolic_link_to_a_file_is_refused_and_the_file_it_names_is_left_alone(self, tmp_path: Path) -> None:
        """Following the link would put the contents outside the directory that was made private."""
        elsewhere = tmp_path / "elsewhere.db"
        elsewhere.write_text("someone else's file")
        elsewhere.chmod(0o644)
        link = tmp_path / "usage.db"
        link.symlink_to(elsewhere)

        with pytest.raises(ConfigError, match=r"cannot open .*usage\.db as a private file: OSError"):
            open_private_file(link)

        assert mode_of(elsewhere) == 0o644
        assert elsewhere.read_text() == "someone else's file"

    def test_a_dangling_symbolic_link_is_refused_and_no_file_is_created_where_it_points(self, tmp_path: Path) -> None:
        link = tmp_path / "usage.db"
        link.symlink_to(tmp_path / "not-there.db")

        with pytest.raises(ConfigError, match=r"cannot open .*usage\.db as a private file"):
            open_private_file(link)

        assert not (tmp_path / "not-there.db").exists()

    def test_a_directory_in_the_way_is_a_config_error(self, tmp_path: Path) -> None:
        in_the_way = tmp_path / "usage.db"
        in_the_way.mkdir()

        with pytest.raises(ConfigError, match=r"cannot open .*usage\.db as a private file: IsADirectoryError"):
            open_private_file(in_the_way)

    def test_a_file_whose_permissions_cannot_be_set_is_a_config_error_and_is_not_left_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closed: list[int] = []
        real_close = os.close

        def refuse(descriptor: int, mode: int) -> None:
            raise PermissionError(1, "Operation not permitted")

        def record_close(descriptor: int) -> None:
            closed.append(descriptor)
            real_close(descriptor)

        monkeypatch.setattr(private.os, "fchmod", refuse)
        monkeypatch.setattr(private.os, "close", record_close)

        with pytest.raises(ConfigError, match=r"cannot open .* as a private file: PermissionError"):
            open_private_file(tmp_path / "usage.db")

        assert len(closed) == 1


INHERITED_GRANT = "everyone allow list,search,read,readattr,readextattr,readsecurity,file_inherit,directory_inherit"
ON_MACOS = pytest.mark.skipif(sys.platform != "darwin", reason="access-control lists of this kind are a macOS feature")


def grant_everyone_access(path: Path) -> None:
    """Give ``path`` an access-control entry that lets every account read it, inherited by what is created inside."""
    subprocess.run(["/bin/chmod", "+a", INHERITED_GRANT, str(path)], check=True)


def access_control_entries(path: Path) -> list[str]:
    """Return the entries ``ls -le`` lists under ``path``: empty when it has no access-control list."""
    listing = subprocess.run(["/bin/ls", "-lde", str(path)], capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in listing.splitlines()[1:]]


@ON_MACOS
class TestAccessControlLists:
    """Permission bits are half of "owner-only" on macOS: an access-control entry grants access separately."""

    def test_the_helpers_used_by_these_tests_see_a_grant(self, tmp_path: Path) -> None:
        directory = tmp_path / "data"
        directory.mkdir(mode=0o700)

        grant_everyone_access(directory)

        assert "everyone allow" in access_control_entries(directory)[0]

    def test_an_existing_directory_s_grant_to_other_accounts_is_removed(self, tmp_path: Path) -> None:
        directory = tmp_path / "data"
        directory.mkdir(mode=0o700)
        grant_everyone_access(directory)

        ensure_private_directory(directory)

        assert access_control_entries(directory) == []
        assert mode_of(directory) == 0o700

    def test_a_directory_created_under_a_parent_whose_grant_is_inherited_ends_up_without_it(
        self, tmp_path: Path
    ) -> None:
        parent = tmp_path / "parent"
        parent.mkdir()
        grant_everyone_access(parent)

        ensure_private_directory(parent / "created" / "data")

        assert access_control_entries(parent / "created") == []
        assert access_control_entries(parent / "created" / "data") == []
        assert "everyone allow" in access_control_entries(parent)[0]

    def test_a_file_that_inherited_a_grant_loses_it_when_opened_as_private(self, tmp_path: Path) -> None:
        directory = tmp_path / "data"
        directory.mkdir(mode=0o700)
        grant_everyone_access(directory)
        inherited = directory / "usage.db"
        inherited.write_text("kept")
        assert "inherited allow" in access_control_entries(inherited)[0]

        os.close(open_private_file(inherited))

        assert access_control_entries(inherited) == []
        assert mode_of(inherited) == 0o600
        assert inherited.read_text() == "kept"

    def test_a_file_created_in_a_directory_after_it_was_made_private_inherits_no_grant(self, tmp_path: Path) -> None:
        """SQLite's journal files and the vendor CLIs' login files are created this way."""
        directory = tmp_path / "data"
        directory.mkdir(mode=0o700)
        grant_everyone_access(directory)
        ensure_private_directory(directory)

        (directory / "usage.db-wal").write_text("written by something else")

        assert access_control_entries(directory / "usage.db-wal") == []

    def test_the_database_and_its_directory_end_up_without_a_grant(self, tmp_path: Path) -> None:
        directory = tmp_path / "data"
        directory.mkdir(mode=0o755)
        grant_everyone_access(directory)
        (directory / "usage.db").touch()

        Database(directory / "usage.db").initialize()

        assert access_control_entries(directory) == []
        assert access_control_entries(directory / "usage.db") == []


class FakeAclLibrary:
    """Stands in for the system library so each way its calls can fail is exercised."""

    def __init__(self, *, allocation: int | None = 1, result: int = 0, error_code: int = 0):
        self.allocation = allocation
        self.result = result
        self.error_code = error_code
        self.set_calls: list[tuple[int, int | None, int]] = []
        self.freed: list[int | None] = []

    def acl_init(self, count: int) -> int | None:
        if self.allocation is None:
            ctypes.set_errno(self.error_code)
        return self.allocation

    def acl_set_fd_np(self, descriptor: int, access_control_list: int | None, kind: int) -> int:
        self.set_calls.append((descriptor, access_control_list, kind))
        if self.result != 0:
            ctypes.set_errno(self.error_code)
        return self.result

    def acl_free(self, access_control_list: int | None) -> int:
        self.freed.append(access_control_list)
        return 0


class TestRemovingAnAccessControlList:
    def library(
        self, monkeypatch: pytest.MonkeyPatch, *, allocation: int | None = 1, result: int = 0, error_code: int = 0
    ) -> FakeAclLibrary:
        library = FakeAclLibrary(allocation=allocation, result=result, error_code=error_code)
        monkeypatch.setattr(private.sys, "platform", "darwin")
        monkeypatch.setattr(private, "_acl_library", lambda: library)
        return library

    def test_the_list_is_replaced_by_an_empty_one_of_the_extended_kind_and_the_empty_one_is_freed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = self.library(monkeypatch, allocation=41)

        descriptor = open_private_file(tmp_path / "usage.db")
        os.close(descriptor)

        assert library.set_calls == [(descriptor, 41, private.ACL_TYPE_EXTENDED)]
        assert library.freed == [41]

    @pytest.mark.parametrize("code", [errno.ENOTSUP, errno.EOPNOTSUPP])
    def test_a_file_system_that_has_no_access_control_lists_has_none_to_remove(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
    ) -> None:
        library = self.library(monkeypatch, result=-1, error_code=code)

        os.close(open_private_file(tmp_path / "usage.db"))
        ensure_private_directory(tmp_path / "data")

        assert len(library.freed) == len(library.set_calls) == 2
        assert mode_of(tmp_path / "data") == 0o700

    @pytest.mark.parametrize("code", [errno.EPERM, errno.EACCES, errno.EIO])
    def test_a_list_that_cannot_be_removed_is_a_config_error_and_nothing_is_left_open_or_allocated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
    ) -> None:
        """Reporting success would leave the file readable by whoever the list names."""
        library = self.library(monkeypatch, result=-1, error_code=code)
        closed: list[int] = []
        real_close = os.close

        def record_close(descriptor: int) -> None:
            closed.append(descriptor)
            real_close(descriptor)

        monkeypatch.setattr(private.os, "close", record_close)

        with pytest.raises(ConfigError, match=r"cannot open .*usage\.db as a private file: "):
            open_private_file(tmp_path / "usage.db")
        with pytest.raises(ConfigError, match=r"cannot make .*data a private directory: "):
            ensure_private_directory(tmp_path / "data")

        assert len(closed) == 2
        assert library.freed == [1, 1]

    def test_an_empty_list_that_cannot_be_allocated_is_a_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = self.library(monkeypatch, allocation=None, error_code=errno.ENOMEM)

        with pytest.raises(ConfigError, match=r"cannot open .*usage\.db as a private file: OSError"):
            open_private_file(tmp_path / "usage.db")

        assert library.set_calls == []
        assert library.freed == []

    def test_nothing_is_done_on_a_system_other_than_macos(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_library() -> object:
            raise AssertionError("the macOS library must not be loaded on another system")

        monkeypatch.setattr(private.sys, "platform", "linux")
        monkeypatch.setattr(private, "_acl_library", no_library)

        os.close(open_private_file(tmp_path / "usage.db"))
        ensure_private_directory(tmp_path / "data")

        assert (mode_of(tmp_path / "usage.db"), mode_of(tmp_path / "data")) == (0o600, 0o700)

    @ON_MACOS
    def test_the_system_library_is_loaded_once_and_declares_the_calls_this_module_makes(self) -> None:
        library = private._acl_library()
        calls = ("acl_init", "acl_set_fd_np", "acl_get_fd_np", "acl_to_text", "acl_free")

        assert library is private._acl_library()
        assert all(hasattr(library, name) for name in calls)


def user_name() -> str:
    return pwd.getpwuid(os.getuid()).pw_name


class TestADirectoryOnlyItsOwnerCanChange:
    """A directory this package relies on but must not change, such as ``~/Library/LaunchAgents``, is checked."""

    def directory(self, tmp_path: Path, mode: int = 0o700) -> Path:
        directory = tmp_path / "LaunchAgents"
        directory.mkdir()
        directory.chmod(mode)
        return directory

    @pytest.mark.parametrize("mode", [0o700, 0o755, 0o500, 0o711])
    def test_a_directory_of_this_account_that_only_it_can_write_to_passes_and_is_left_as_it_is(
        self, tmp_path: Path, mode: int
    ) -> None:
        directory = self.directory(tmp_path, mode)

        ensure_only_owner_can_change(directory)

        assert mode_of(directory) == mode

    @pytest.mark.parametrize("mode", [0o720, 0o702, 0o770, 0o777])
    def test_a_directory_its_group_or_others_can_write_to_is_refused_and_left_as_it_is(
        self, tmp_path: Path, mode: int
    ) -> None:
        directory = self.directory(tmp_path, mode)

        with pytest.raises(ConfigError) as raised:
            ensure_only_owner_can_change(directory)

        assert str(raised.value) == (
            f"cannot use {directory}: its permissions ({mode:04o}) let other accounts write to it, so another "
            "account could replace what is in it; make it writable by you alone and run the command again"
        )
        assert mode_of(directory) == mode

    def test_a_directory_of_another_account_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        directory = self.directory(tmp_path)
        monkeypatch.setattr(private.os, "geteuid", lambda: os.getuid() + 1)

        with pytest.raises(ConfigError, match=r"^cannot use .*LaunchAgents: it belongs to another account, so "):
            ensure_only_owner_can_change(directory)

    def test_a_symbolic_link_to_a_directory_only_its_owner_can_change_is_refused(self, tmp_path: Path) -> None:
        """What it points at lies somewhere the caller did not look."""
        target = self.directory(tmp_path)
        link = tmp_path / "link"
        link.symlink_to(target, target_is_directory=True)

        with pytest.raises(ConfigError, match=r"^cannot use .*link: it is a symbolic link$"):
            ensure_only_owner_can_change(link)

    @pytest.mark.parametrize(
        ("make", "reason"),
        [
            pytest.param(lambda path: None, "FileNotFoundError", id="nothing there"),
            pytest.param(lambda path: path.write_text("a file"), "NotADirectoryError", id="a file"),
        ],
    )
    def test_anything_but_a_directory_is_refused(
        self, tmp_path: Path, make: Callable[[Path], object], reason: str
    ) -> None:
        path = tmp_path / "LaunchAgents"
        make(path)

        with pytest.raises(ConfigError, match=rf"^cannot use .*LaunchAgents: {reason}$"):
            ensure_only_owner_can_change(path)

    def test_a_path_the_system_does_not_accept_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"^cannot use .*: ValueError$"):
            ensure_only_owner_can_change(tmp_path / "Launch\x00Agents")

    @ON_MACOS
    def test_a_directory_only_the_administrators_group_can_write_passes_only_when_that_is_allowed(
        self, tmp_path: Path
    ) -> None:
        """Homebrew's prefix is such a directory; administrators can become root, so the bit adds nothing."""
        administrators = private.ADMINISTRATORS_GROUP_ID
        if administrators is None or administrators not in os.getgroups() or os.getgid() == administrators:
            pytest.skip("this account is not in an administrators group apart from its own")
        prefix = tmp_path / "homebrew"
        prefix.mkdir()
        os.chown(prefix, -1, administrators)
        prefix.chmod(0o775)

        with pytest.raises(ConfigError, match=r"its permissions \(0775\) let other accounts write"):
            ensure_only_owner_can_change(prefix)
        ensure_only_owner_can_change(prefix, administrators_may_write=True)

        prefix.chmod(0o777)
        with pytest.raises(ConfigError, match=r"its permissions \(0777\) let other accounts write"):
            ensure_only_owner_can_change(prefix, administrators_may_write=True)
        os.chown(prefix, -1, os.getgid())
        prefix.chmod(0o775)
        with pytest.raises(ConfigError, match=r"its permissions \(0775\) let other accounts write"):
            ensure_only_owner_can_change(prefix, administrators_may_write=True)

    def test_a_group_that_does_not_exist_has_no_id(self) -> None:
        assert private._group_id("no-such-group-for-usagedash") is None

    def test_an_entry_that_lets_every_account_add_and_delete_files_is_refused_and_left_as_it_is(
        self, tmp_path: Path
    ) -> None:
        """The mode is 700, and the entry still lets another account replace what is in it."""
        directory = self.directory(tmp_path)
        subprocess.run(["/bin/chmod", "+a", "everyone allow add_file,delete_child", str(directory)], check=True)
        granted = access_control_entries(directory)

        with pytest.raises(ConfigError, match=r"an access-control entry lets another account change it, so "):
            ensure_only_owner_can_change(directory)

        assert access_control_entries(directory) == granted

    @ON_MACOS
    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param("everyone deny delete", id="an entry that denies, as ~/Library has"),
            pytest.param(INHERITED_GRANT, id="an entry that lets every account read"),
            pytest.param("OWNER allow add_file,delete_child", id="an entry for the owner itself"),
        ],
    )
    def test_entries_that_deny_or_let_others_only_read_pass_and_are_left_as_they_are(
        self, tmp_path: Path, entry: str
    ) -> None:
        directory = self.directory(tmp_path)
        subprocess.run(["/bin/chmod", "+a", entry.replace("OWNER", f"user:{user_name()}"), str(directory)], check=True)
        granted = access_control_entries(directory)

        ensure_only_owner_can_change(directory)

        assert access_control_entries(directory) == granted != []


class FakeAclReadingLibrary:
    """Stands in for the system library's two calls that read an access-control list as text."""

    def __init__(self, *, listed: bytes | None = None, get_error: int = 0, text_error: int = 0):
        self.listed = listed
        self.get_error = get_error
        self.text_error = text_error
        self.buffer = ctypes.create_string_buffer(listed or b"")
        self.freed: list[int | None] = []

    def acl_get_fd_np(self, descriptor: int, kind: int) -> int | None:
        assert kind == private.ACL_TYPE_EXTENDED
        if self.get_error:
            ctypes.set_errno(self.get_error)
            return None
        return 7

    def acl_to_text(self, access_control_list: int, length: Any) -> int | None:
        if self.text_error:
            ctypes.set_errno(self.text_error)
            return None
        length._obj.value = len(self.listed or b"")
        return ctypes.addressof(self.buffer)

    def acl_free(self, pointer: int | None) -> int:
        self.freed.append(pointer)
        return 0


class TestReadingADirectorysAccessControlList:
    def library(self, monkeypatch: pytest.MonkeyPatch, **options: Any) -> FakeAclReadingLibrary:
        library = FakeAclReadingLibrary(**options)
        monkeypatch.setattr(private.sys, "platform", "darwin")
        monkeypatch.setattr(private, "_acl_library", lambda: library)
        return library

    def test_the_list_is_read_as_text_and_both_the_list_and_the_text_are_freed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        listed = b"!#acl 1\ngroup:ABCDEFAB-CDEF-ABCD-EFAB-CDEF0000000C:everyone:12:allow:write\n"
        library = self.library(monkeypatch, listed=listed)
        directory = tmp_path / "LaunchAgents"
        directory.mkdir(mode=0o700)

        with pytest.raises(ConfigError, match="an access-control entry lets another account change it"):
            ensure_only_owner_can_change(directory)

        assert library.freed == [ctypes.addressof(library.buffer), 7]

    @pytest.mark.parametrize("code", [errno.ENOENT, errno.ENOTSUP, errno.EOPNOTSUPP])
    def test_no_list_or_no_lists_on_the_file_system_lets_nobody_change_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
    ) -> None:
        library = self.library(monkeypatch, get_error=code)
        directory = tmp_path / "LaunchAgents"
        directory.mkdir(mode=0o700)

        ensure_only_owner_can_change(directory)

        assert library.freed == []

    @pytest.mark.parametrize(
        ("options", "reason"),
        [
            pytest.param({"get_error": errno.EACCES}, "PermissionError", id="the list cannot be read"),
            pytest.param({"listed": b"x", "text_error": errno.ENOMEM}, "OSError", id="it cannot be written as text"),
        ],
    )
    def test_a_list_that_cannot_be_read_is_a_config_error_and_nothing_is_left_allocated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: dict[str, Any], reason: str
    ) -> None:
        """Passing a directory whose list could not be read would trust whatever the list says."""
        library = self.library(monkeypatch, **options)
        directory = tmp_path / "LaunchAgents"
        directory.mkdir(mode=0o700)

        with pytest.raises(ConfigError, match=rf"^cannot use .*LaunchAgents: {reason}$"):
            ensure_only_owner_can_change(directory)

        assert library.freed == ([] if "get_error" in options else [7])

    def test_no_list_is_read_on_a_system_other_than_macos(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_library() -> object:
            raise AssertionError("the macOS library must not be loaded on another system")

        monkeypatch.setattr(private.sys, "platform", "linux")
        monkeypatch.setattr(private, "_acl_library", no_library)
        directory = tmp_path / "LaunchAgents"
        directory.mkdir(mode=0o700)

        ensure_only_owner_can_change(directory)


OWNER = 501
UUID = "ABCDEFAB-CDEF-ABCD-EFAB-CDEF0000000C"


class TestWhatAnAccessControlEntryLets:
    """The list as ``acl_to_text`` writes it: ``tag:uuid:name:id:kind[,flags]:rights`` after a ``!#`` header."""

    @pytest.mark.parametrize(
        "listed",
        [
            pytest.param("", id="no list"),
            pytest.param("!#acl 1\n", id="a header and no entries"),
            pytest.param(f"!#acl 1\ngroup:{UUID}:everyone:12:deny:delete\n", id="an entry that denies"),
            pytest.param(
                f"group:{UUID}:everyone:12:deny,file_inherit:write,delete_child\n", id="a denial that is inherited"
            ),
            pytest.param(f"group:{UUID}:everyone:12:allow:read,execute,readattr\n", id="reading and searching"),
            pytest.param(
                f"group:{UUID}:everyone:12:allow:readextattr,readsecurity,synchronize\n", id="reading its attributes"
            ),
            pytest.param(f"user:{UUID}:persona:{OWNER}:allow:write,delete_child\n", id="the owner's own entry"),
            pytest.param(f"!#acl 1\n\ngroup:{UUID}:everyone:12:deny:delete\n\n", id="blank lines"),
        ],
    )
    def test_entries_that_let_nobody_else_change_it(self, listed: str) -> None:
        assert access_entries_let_others_change(listed, OWNER) is False

    @pytest.mark.parametrize(
        "listed",
        [
            pytest.param(f"group:{UUID}:everyone:12:allow:write\n", id="adding files"),
            pytest.param(f"group:{UUID}:everyone:12:allow:append\n", id="adding directories"),
            pytest.param(f"group:{UUID}:everyone:12:allow:delete_child\n", id="removing what is in it"),
            pytest.param(f"group:{UUID}:everyone:12:allow:writesecurity\n", id="changing its permissions"),
            pytest.param(f"group:{UUID}:everyone:12:allow:chown\n", id="taking it over"),
            pytest.param(f"group:{UUID}:everyone:12:allow,file_inherit,directory_inherit:write\n", id="inherited"),
            pytest.param(f"user:{UUID}:other:502:allow:write\n", id="another user's entry"),
            pytest.param(f"group:{UUID}:staff:{OWNER}:allow:write\n", id="a group whose id is the owner's user id"),
            pytest.param(f"group:{UUID}:everyone:12:allow:read,frobnicate\n", id="a right this code does not know"),
            pytest.param(f"group:{UUID}:everyone:12:audit:write\n", id="a kind other than allow or deny"),
            pytest.param(f"group:{UUID}:every:one:12:allow:read\n", id="a line that cannot be read"),
            pytest.param(f"group:{UUID}:everyone:12:deny:delete\ngroup:{UUID}:everyone:12:allow:write\n", id="second"),
        ],
    )
    def test_entries_that_let_another_account_change_it(self, listed: str) -> None:
        assert access_entries_let_others_change(listed, OWNER) is True


class TestPathsTheSystemDoesNotAccept:
    """A path holding a NUL character is refused by the system calls with ValueError, not OSError."""

    def test_such_a_directory_path_is_a_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="a private directory: ValueError"):
            ensure_private_directory(tmp_path / "da\x00ta")

    def test_such_a_parent_path_is_a_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="a private directory: ValueError"):
            ensure_private_directory(tmp_path / "pa\x00rent" / "data")

    def test_such_a_file_path_is_a_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="as a private file: ValueError"):
            open_private_file(tmp_path / "usage\x00.db")

    def test_a_database_at_such_a_path_fails_to_start_with_a_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="as a private file: ValueError"):
            Database(tmp_path / "usage\x00.db").initialize()


class TestAncestorsAnotherAccountCouldReplace:
    def test_a_path_under_the_temp_directory_is_not_refused_because_of_the_system_boundary(
        self, tmp_path: Path
    ) -> None:
        """``/tmp`` is a root-owned link to a sticky directory. That is not this account's parent to refuse."""
        ensure_no_ancestor_lets_another_account_replace(tmp_path / "data")

    def test_the_filesystem_root_has_no_ancestor_to_refuse(self) -> None:
        ensure_no_ancestor_lets_another_account_replace(Path("/"))

    def test_a_root_owned_temp_link_is_accepted_with_its_parents(self) -> None:
        # Why this path: it is a root-owned symbolic link whose target is a sticky directory.
        link = Path("/tmp")  # noqa: S108
        if not link.is_symlink() or link.lstat().st_uid != 0:
            pytest.skip("the temp directory is not a root-owned symbolic link on this machine")

        ensure_no_ancestor_lets_another_account_replace(link / "no-such-entry")

    def test_a_symbolic_link_this_account_owns_is_refused(self, tmp_path: Path) -> None:
        link = tmp_path / "link"
        link.symlink_to(tmp_path)

        with pytest.raises(ConfigError, match="it is a symbolic link"):
            ensure_no_ancestor_lets_another_account_replace(link / "child")

    def sticky_root_temp(self) -> Path:
        """Return ``/private/tmp`` when it is a root-owned sticky directory here, or skip."""
        sticky = Path("/private/tmp")
        try:
            status = sticky.lstat()
        except OSError:
            pytest.skip("no /private/tmp on this machine")
        if status.st_uid != 0 or not status.st_mode & stat.S_ISVTX or not stat.S_ISDIR(status.st_mode):
            pytest.skip("/private/tmp is not a root-owned sticky directory on this machine")
        return sticky

    def test_a_missing_level_under_a_sticky_directory_is_refused_unless_it_is_the_path_itself(self) -> None:
        """Anyone can create a missing entry in a sticky directory, and make it a link. The path itself is
        created by this package, which fails if another account got there first."""
        sticky = self.sticky_root_temp()
        pivot = sticky / f"usagedash-missing-{os.getpid()}"

        ensure_no_ancestor_lets_another_account_replace(pivot)
        with pytest.raises(ConfigError, match=r"its permissions \(1777\) let other accounts write"):
            ensure_no_ancestor_lets_another_account_replace(pivot / ".." / "owner-data" / "homes" / "codex-work")
        with pytest.raises(ConfigError, match=r"its permissions \(1777\) let other accounts write"):
            ensure_no_other_account_can_redirect(pivot / "data")

    def test_a_missing_level_below_a_root_owned_link_to_a_sticky_directory_is_refused_too(self) -> None:
        """The link's target is where the missing entry would be created."""
        self.sticky_root_temp()
        link = Path("/tmp")  # noqa: S108
        if not link.is_symlink() or link.lstat().st_uid != 0:
            pytest.skip("the temp directory is not a root-owned symbolic link on this machine")

        ensure_no_ancestor_lets_another_account_replace(link / "no-such-entry")
        with pytest.raises(ConfigError, match=r"its permissions \(1777\) let other accounts write"):
            ensure_no_ancestor_lets_another_account_replace(link / "no-such-entry" / "data")

    def test_a_file_under_a_sticky_directory_passes_only_when_this_account_owns_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file another account owns can be swapped by that account for a link to anywhere."""
        sticky = self.sticky_root_temp()
        pivot = sticky / f"usagedash-pivot-{os.getpid()}"
        pivot.write_text("")
        try:
            ensure_no_ancestor_lets_another_account_replace(pivot / ".." / "owner-data" / "homes" / "codex-work")

            real_lstat = Path.lstat

            def lstat(self: Path) -> Any:
                status = real_lstat(self)
                if self != pivot:
                    return status
                return SimpleNamespace(
                    st_uid=status.st_uid + 1, st_mode=status.st_mode, st_dev=status.st_dev, st_ino=status.st_ino
                )

            monkeypatch.setattr(Path, "lstat", lstat)
            with pytest.raises(ConfigError, match=r"its permissions \(1777\) let other accounts write"):
                ensure_no_ancestor_lets_another_account_replace(pivot / ".." / "owner-data" / "homes" / "codex-work")
        finally:
            pivot.unlink()

    def test_a_path_through_a_file_is_examined_from_the_directory_that_holds_the_file(self, tmp_path: Path) -> None:
        """Nothing exists below a file, and only that directory decides whether the file can become a link."""
        safe_file = tmp_path / "file"
        safe_file.write_text("")
        unsafe = tmp_path / "unsafe"
        unsafe.mkdir()
        unsafe.chmod(0o777)
        unsafe_file = unsafe / "file"
        unsafe_file.write_text("")
        try:
            ensure_no_ancestor_lets_another_account_replace(safe_file / ".." / "data")
            with pytest.raises(ConfigError, match=rf"cannot use {re.escape(str(unsafe))}: its permissions \(0777\)"):
                ensure_no_ancestor_lets_another_account_replace(unsafe_file / ".." / "data")
        finally:
            unsafe.chmod(0o700)

    def test_an_accepted_symbolic_link_is_not_followed_and_its_parent_is_checked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A link this account owns is not followed. Accepting it still checks the directory that contains it."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        link = tmp_path / "link"
        link.symlink_to(elsewhere)
        seen: list[Path] = []
        real = private.ensure_only_owner_can_change

        def record(directory: Path, **kwargs: Any) -> None:
            seen.append(directory)
            if directory == link:
                return
            real(directory, **kwargs)

        monkeypatch.setattr(private, "ensure_only_owner_can_change", record)

        ensure_no_ancestor_lets_another_account_replace(link / "child")

        assert link in seen
        assert link.parent in seen
        assert elsewhere not in seen

    def test_a_relative_path_is_walked_from_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A relative HOME gives a relative data directory, whose ancestors are the working directory's."""
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(0o777)
        (shared / "work").mkdir()
        monkeypatch.chdir(shared / "work")
        try:
            with pytest.raises(ConfigError, match=r"its permissions \(0777\)"):
                private.ensure_no_ancestor_lets_another_account_replace(Path("data"))
        finally:
            shared.chmod(0o700)

    @pytest.mark.parametrize("failure", [OSError("denied"), ValueError("bad")])
    def test_an_ancestor_that_cannot_be_examined_is_a_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ) -> None:
        """``Path.exists`` and ``Path.is_symlink`` report this failure as absence. The walk must not."""

        def lstat(self: Path) -> os.stat_result:
            raise failure

        def absent(self: Path) -> bool:
            return False

        monkeypatch.setattr(Path, "lstat", lstat)
        monkeypatch.setattr(Path, "is_symlink", absent)
        monkeypatch.setattr(Path, "exists", absent)

        examined = tmp_path / "data"
        with pytest.raises(ConfigError, match=rf"cannot use {re.escape(str(examined))}: {type(failure).__name__}$"):
            ensure_no_ancestor_lets_another_account_replace(examined)

    def test_a_missing_ancestor_is_skipped_when_path_predicates_report_every_ancestor_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing ancestor is ``FileNotFoundError`` from ``lstat``, and a later ancestor is still checked."""
        unsafe = tmp_path / "unsafe"
        unsafe.mkdir()
        unsafe.chmod(0o777)
        missing = unsafe / "missing"
        real_lstat = Path.lstat

        def lstat(self: Path) -> os.stat_result:
            if self == missing:
                raise FileNotFoundError(errno.ENOENT, "No such file or directory")
            return real_lstat(self)

        def absent(self: Path) -> bool:
            return False

        monkeypatch.setattr(Path, "lstat", lstat)
        monkeypatch.setattr(Path, "is_symlink", absent)
        monkeypatch.setattr(Path, "exists", absent)
        try:
            with pytest.raises(ConfigError, match=r"its permissions \(0777\)"):
                ensure_no_ancestor_lets_another_account_replace(missing / "data")
        finally:
            unsafe.chmod(0o700)

    def test_the_sticky_temp_directory_itself_is_still_refused(self) -> None:
        """The sticky pass is for an ancestor of a path, not for a directory this command was asked to use."""
        resolved = Path(tempfile.gettempdir()).resolve()
        status = resolved.lstat()
        if status.st_uid != 0 or not status.st_mode & stat.S_ISVTX:
            pytest.skip("this temp directory is not a root-owned sticky directory")

        with pytest.raises(ConfigError, match=r"its permissions \(1777\) let other accounts write"):
            ensure_only_owner_can_change(resolved, root_may_own=True)

    def test_an_ancestor_this_account_owns_that_others_can_write_to_is_refused_and_left_as_it_is(
        self, tmp_path: Path
    ) -> None:
        unsafe = tmp_path / "unsafe"
        unsafe.mkdir()
        unsafe.chmod(0o777)
        try:
            with pytest.raises(ConfigError, match=rf"cannot use {unsafe}: its permissions \(0777\)"):
                ensure_no_ancestor_lets_another_account_replace(unsafe / "missing" / "data")
            assert mode_of(unsafe) == 0o777
        finally:
            unsafe.chmod(0o700)

    def test_a_sticky_directory_this_account_owns_is_still_refused(self, tmp_path: Path) -> None:
        sticky = tmp_path / "sticky"
        sticky.mkdir()
        sticky.chmod(0o1777)
        try:
            with pytest.raises(ConfigError, match=r"its permissions \(1777\) let other accounts write"):
                ensure_no_ancestor_lets_another_account_replace(sticky / "data")
        finally:
            sticky.chmod(0o700)

    def test_an_ancestry_that_cannot_be_walked_to_its_end_is_a_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(private, "ANCESTOR_WALK_LIMIT", 0)

        with pytest.raises(ConfigError, match="its ancestry could not be checked"):
            ensure_no_ancestor_lets_another_account_replace(tmp_path / "data")

    def test_a_root_owned_link_under_a_writable_directory_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Another account can unlink a root-owned link from a directory it can write to."""
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(0o777)
        link = shared / "root-link"
        link.symlink_to(tmp_path, target_is_directory=True)
        _report_root_ownership(monkeypatch, {link})
        try:
            with pytest.raises(ConfigError, match=rf"cannot use {re.escape(str(shared))}: its permissions \(0777\)"):
                ensure_no_ancestor_lets_another_account_replace(link / "data")
            assert mode_of(shared) == 0o777
        finally:
            shared.chmod(0o700)

    def test_a_root_owned_directory_does_not_hide_a_link_above_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A root-owned directory below a link is not the end of the path the login keeps."""
        real = tmp_path / "real"
        real.mkdir()
        root_dir = real / "root-dir"
        root_dir.mkdir()
        link = tmp_path / "other-link"
        link.symlink_to(real, target_is_directory=True)
        lexical_root = link / "root-dir"
        _report_root_ownership(monkeypatch, {lexical_root})

        with pytest.raises(ConfigError, match="it is a symbolic link"):
            ensure_no_ancestor_lets_another_account_replace(lexical_root / "data")

    def test_a_root_owned_link_is_refused_when_its_target_can_be_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The login keeps the link and writes through it, so the target's parents count."""
        safe = tmp_path / "safe"
        safe.mkdir()
        unsafe = tmp_path / "unsafe"
        unsafe.mkdir()
        unsafe.chmod(0o777)
        bucket = unsafe / "bucket"
        bucket.mkdir()
        link = safe / "root-link"
        link.symlink_to(Path(os.path.relpath(bucket, safe)), target_is_directory=True)
        _report_root_ownership(monkeypatch, {link})
        try:
            with pytest.raises(ConfigError, match=r"its permissions \(0777\)") as raised:
                ensure_no_ancestor_lets_another_account_replace(link / "data")
            assert unsafe.name in str(raised.value)
            assert mode_of(unsafe) == 0o777
        finally:
            unsafe.chmod(0o700)

    def test_a_root_owned_link_that_points_at_itself_is_not_followed_forever(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        link = tmp_path / "root-link"
        link.symlink_to(link)
        _report_root_ownership(monkeypatch, {link})

        ensure_no_ancestor_lets_another_account_replace(link / "data")

    def test_followed_links_count_toward_the_same_walk_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The step budget is shared with a followed target, so a root-owned link cannot reset it."""
        link = tmp_path / "root-link"
        link.symlink_to(tmp_path, target_is_directory=True)
        _report_root_ownership(monkeypatch, {link})
        monkeypatch.setattr(private, "ANCESTOR_WALK_LIMIT", len((link / "data").parts))

        with pytest.raises(ConfigError, match="its ancestry could not be checked"):
            ensure_no_ancestor_lets_another_account_replace(link / "data")


class TestASpellingAnotherAccountCouldRedirect:
    """The ancestor walk with the last entry examined too, for a spelling that is kept and resolved again later."""

    def test_a_directory_this_account_owns_passes_with_its_ancestors(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir(mode=0o700)

        ensure_no_other_account_can_redirect(home)

    def test_the_last_entry_is_examined_and_left_as_it_is(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        home.chmod(0o777)
        try:
            with pytest.raises(ConfigError, match=rf"cannot use {re.escape(str(home))}: its permissions \(0777\)"):
                ensure_no_other_account_can_redirect(home)
            assert mode_of(home) == 0o777
        finally:
            home.chmod(0o700)

    def test_a_link_this_account_owns_at_the_end_is_refused_however_safe_its_directory(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir(mode=0o700)
        link = tmp_path / "link"
        link.symlink_to(home, target_is_directory=True)

        with pytest.raises(ConfigError, match=rf"cannot use {re.escape(str(link))}: it is a symbolic link"):
            ensure_no_other_account_can_redirect(link)

    def test_a_root_owned_link_at_the_end_is_followed_to_where_it_leads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        safe = tmp_path / "safe"
        safe.mkdir()
        unsafe = tmp_path / "unsafe"
        unsafe.mkdir()
        unsafe.chmod(0o777)
        bucket = unsafe / "bucket"
        bucket.mkdir()
        link = safe / "root-link"
        link.symlink_to(bucket, target_is_directory=True)
        _report_root_ownership(monkeypatch, {link})
        try:
            with pytest.raises(ConfigError, match=rf"cannot use {re.escape(str(unsafe))}: its permissions \(0777\)"):
                ensure_no_other_account_can_redirect(link)
        finally:
            unsafe.chmod(0o700)

    def test_an_entry_that_is_not_there_yet_is_skipped_and_its_ancestors_are_still_examined(
        self, tmp_path: Path
    ) -> None:
        ensure_no_other_account_can_redirect(tmp_path / "not-yet")

        unsafe = tmp_path / "unsafe"
        unsafe.mkdir()
        unsafe.chmod(0o777)
        try:
            with pytest.raises(ConfigError, match=rf"cannot use {re.escape(str(unsafe))}: its permissions \(0777\)"):
                ensure_no_other_account_can_redirect(unsafe / "not-yet")
        finally:
            unsafe.chmod(0o700)

    @pytest.mark.parametrize("failure", [OSError("denied"), ValueError("bad")])
    def test_an_entry_that_cannot_be_examined_is_a_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ) -> None:
        def lstat(self: Path) -> os.stat_result:
            raise failure

        monkeypatch.setattr(Path, "lstat", lstat)

        examined = tmp_path / "home"
        with pytest.raises(ConfigError, match=rf"cannot use {re.escape(str(examined))}: {type(failure).__name__}$"):
            ensure_no_other_account_can_redirect(examined)


class TestCheckingAnEntryWithoutOpeningIt:
    @pytest.mark.parametrize("mode", [0o100, 0o500, 0o755])
    def test_executable_need_not_be_opened(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: int) -> None:
        path = tmp_path / "program"
        path.touch()
        path.chmod(mode)

        def refuse_open(*args: object, **kwargs: object) -> int:
            raise AssertionError("entry must not be opened")

        monkeypatch.setattr(private.os, "open", refuse_open)
        assert private.what_lets_another_account_change_by_path(path) is None

    def test_symlinks_are_reported_without_following(self, tmp_path: Path) -> None:
        path = tmp_path / "link"
        path.symlink_to(tmp_path / "missing")
        assert private.what_lets_another_account_change_by_path(path) == "it is a symbolic link"

    @pytest.mark.parametrize("name", ["missing", "nul\x00name"])
    def test_unexaminable_entry_is_a_config_error(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(ConfigError, match="cannot examine"):
            private.what_lets_another_account_change_by_path(tmp_path / name)

    def test_acl_failure_is_a_config_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "program"
        path.touch()

        def unreadable(path: Path) -> str:
            raise PermissionError("synthetic")

        monkeypatch.setattr(private, "_access_control_text_of", unreadable)
        with pytest.raises(ConfigError, match="PermissionError"):
            private.what_lets_another_account_change_by_path(path)

    def test_no_list_is_read_on_a_system_other_than_macos(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(private.sys, "platform", "linux")
        assert private.what_lets_another_account_change_by_path(tmp_path) is None

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS ACLs")
    def test_path_acl_write_grant_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "program"
        path.touch()
        subprocess.run(["/bin/chmod", "+a", "everyone allow write", str(path)], check=True)
        assert (
            private.what_lets_another_account_change_by_path(path)
            == "an access-control entry lets another account change it"
        )


@pytest.mark.parametrize("problem", ["none", "nested", "container", "destination", "foreign-link"])
def test_executable_directory_links_keep_their_ancestry_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "links" / "alias"
    link.parent.mkdir(mode=0o700)
    destination = target
    if problem == "nested":
        destination = tmp_path / "nested"
        destination.symlink_to(target)
    link.symlink_to(destination)
    if problem == "container":
        link.parent.chmod(0o777)
    elif problem == "destination":
        target.chmod(0o777)
    elif problem == "foreign-link":
        original = Path.lstat

        def foreign_owner(path: Path) -> os.stat_result:
            status = original(path)
            if path == link:
                values = list(status)
                values[4] = os.geteuid() + 1
                return os.stat_result(values)
            return status

        monkeypatch.setattr(Path, "lstat", foreign_owner)
    if problem in {"none", "nested"}:
        ensure_no_ancestor_lets_another_account_replace(link / "python", follow_owned_links=True)
        with pytest.raises(ConfigError, match="symbolic link"):
            ensure_no_ancestor_lets_another_account_replace(link / "python")
    else:
        with pytest.raises(ConfigError):
            ensure_no_ancestor_lets_another_account_replace(link / "python", follow_owned_links=True)
