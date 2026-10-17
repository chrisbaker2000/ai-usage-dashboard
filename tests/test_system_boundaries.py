"""Code that touches the operating system: launchd, the vendor login commands, and the command line."""

import contextlib
import errno
import fcntl
import http.client
import json
import logging
import os
import plistlib
import pwd
import re
import runpy
import shutil
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import AsyncGenerator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import httpx2
import pytest
import uvicorn
from fastapi import FastAPI

from tests.factories import (
    make_identity,
    make_reading,
    make_window,
)
from tests.fakes.fake_run import FakeRun
from tests.payloads import grok_auth_file, write_claude_account_file
from usagedash import agent, api, cli, credentials, db, executables, private, rpc, standing, wiring
from usagedash.collector import Collector, CycleReport, SourceBundle
from usagedash.config import ClaudeSourceConfig, Settings, SourceMode, load_settings, standing_source_table
from usagedash.db import Database
from usagedash.errors import (
    ConfigError,
    CredentialExpiredError,
    CredentialUnavailableError,
    ProviderRequestError,
    ProviderSchemaError,
)
from usagedash.models import AccountIdentity, Provider, QuotaReading
from usagedash.paths import directory_identity
from usagedash.providers import claude_cli, codex_cli, grok_cli
from usagedash.providers.base import NoQuotaDataError
from usagedash.providers.claude import keychain_account
from usagedash.scanners.base import AccountResolver, ScanReport
from usagedash.timeutil import Clock
from usagedash.wiring import build_bundles

FAKE_CLI = Path(__file__).parent / "fakes" / "fake_cli.py"
NOT_LOADED = 113
SERVER_WAIT_SECONDS = 10.0
SERVER_POLL_SECONDS = 0.02
STRINGS_A_PROPERTY_LIST_DOES_NOT_CARRY_EXACTLY = [
    pytest.param("\x01", id="control character"),
    pytest.param("port\x0b9001", id="vertical tab"),
    pytest.param("\x1f", id="unit separator"),
    pytest.param("\ufffe", id="U+FFFE, which no XML parser accepts"),
    pytest.param("data\uffffdir", id="U+FFFF, which no XML parser accepts"),
    pytest.param("data\rdir", id="carriage return, read back as a line feed"),
    pytest.param("line\r\n", id="carriage return and line feed, read back as one line feed"),
    pytest.param("\udcff", id="a byte that was not text"),
]
PLIST_DATE_THAT_IS_NOT_A_DATE = b"<plist><date>not-a-date</date></plist>"
PLIST_KEY_WITH_NO_VALUE = b"<plist><key>x</key></plist>"
PLIST_XML_CUT_OFF = b"<plist><string>"


def binary_plist_nested_past_the_recursion_limit() -> bytes:
    """Return a binary property list ``plistlib.loads`` cannot finish at the current recursion limit.

    The nesting is five hundred past that limit. The parser's own frames sit on the stack, so one
    past the limit can still return. The writer is given a higher limit, and the reader's limit is
    restored before the bytes are returned.
    """
    depth = sys.getrecursionlimit() + 500
    nested: object = 0
    for _ in range(depth):
        nested = [nested]
    reader_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(20_000)
    try:
        return plistlib.dumps(nested, fmt=plistlib.FMT_BINARY)
    finally:
        sys.setrecursionlimit(reader_limit)


def write_private_bytes(path: Path, payload: bytes) -> None:
    descriptor = agent.open_private_file(path)
    try:
        os.ftruncate(descriptor, 0)
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


PLISTS_THE_PARSER_CANNOT_FINISH = [
    pytest.param(PLIST_DATE_THAT_IS_NOT_A_DATE, "AttributeError", id="a date that is not a date"),
    pytest.param(PLIST_KEY_WITH_NO_VALUE, "IndexError", id="a key with no value"),
    pytest.param(
        binary_plist_nested_past_the_recursion_limit(),
        "RecursionError",
        id="nested past the recursion limit",
    ),
    pytest.param(PLIST_XML_CUT_OFF, "ExpatError", id="xml cut off before it ends"),
]


@pytest.fixture(autouse=True)
def system_path_contents(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    system_paths = {Path(path) for path in agent.SYSTEM_PATH_DIRECTORIES}
    empty = tmp_path_factory.mktemp("system-bin-contents")
    scan = os.scandir

    def controlled_scan(path: Path) -> contextlib.AbstractContextManager[Iterator[os.DirEntry[str]]]:
        return scan(empty if Path(path) in system_paths else path)

    monkeypatch.setattr(agent.os, "scandir", controlled_scan)


class InterruptedAt(FakeRun):
    def __init__(self, subcommand: str, occurrence: int = 1) -> None:
        super().__init__()
        self._subcommand = subcommand
        self._occurrence = occurrence

    def __call__(self, command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        if command[1] == self._subcommand:
            self._occurrence -= 1
            if self._occurrence == 0:
                self.commands.append(command)
                raise KeyboardInterrupt
        return super().__call__(command, **options)


def installation_lock(paths: agent.AgentPaths) -> Path:
    return paths.plist.with_name(f".{paths.plist.name}.lock")


def lock_is_held(paths: agent.AgentPaths) -> bool:
    descriptor = os.open(installation_lock(paths), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(descriptor)
    return False


class NotingTheLock:
    def __init__(self, run: FakeRun, paths: agent.AgentPaths) -> None:
        self._run = run
        self._paths = paths
        self.lock_was_held: list[bool] = []

    def __call__(self, command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        self.lock_was_held.append(lock_is_held(self._paths))
        return self._run(command, **options)


class TestOneCommandAtATimeChangesTheAgent:
    def paths(self, tmp_path: Path) -> agent.AgentPaths:
        return agent.agent_paths(home=tmp_path, user="persona")

    def installed(self, tmp_path: Path) -> agent.AgentPaths:
        paths = self.paths(tmp_path)
        paths.plist.parent.mkdir(parents=True)
        paths.plist.write_bytes(b"previous")
        return paths

    def noting(self, monkeypatch: pytest.MonkeyPatch, run: FakeRun, paths: agent.AgentPaths) -> NotingTheLock:
        noting = NotingTheLock(run, paths)
        monkeypatch.setattr(agent.subprocess, "run", noting)
        monkeypatch.setattr(agent.time, "sleep", lambda _seconds: None)
        return noting

    def test_an_install_holds_the_lock_for_every_command_it_gives_launchd_and_lets_go_after(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths = self.installed(tmp_path)
        run = FakeRun()
        run.sequences = {"print": [0, NOT_LOADED]}
        noting = self.noting(monkeypatch, run, paths)

        agent.install(paths, tmp_path / "data", [], {})

        assert run.subcommands() == ["print", "bootout", "print", "bootstrap"]
        assert noting.lock_was_held == [True] * 4
        assert not lock_is_held(paths)

    def test_an_install_that_fails_holds_the_lock_while_it_puts_back_what_it_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths = self.installed(tmp_path)
        run = FakeRun(stderr="Input/output error")
        run.sequences = {"print": [0, NOT_LOADED, NOT_LOADED], "bootstrap": [5, 0]}
        noting = self.noting(monkeypatch, run, paths)

        with pytest.raises(agent.AgentError, match=r"the previous agent was started again$"):
            agent.install(paths, tmp_path / "data", [], {})

        assert run.subcommands() == ["print", "bootout", "print", "bootstrap", "print", "bootstrap"]
        assert noting.lock_was_held == [True] * 6
        assert not lock_is_held(paths)

    def test_an_interrupted_install_holds_the_lock_while_it_puts_back_what_it_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths = self.installed(tmp_path)
        run = InterruptedAt("bootstrap")
        run.sequences = {"print": [0, NOT_LOADED, NOT_LOADED]}
        noting = self.noting(monkeypatch, run, paths)

        with pytest.raises(KeyboardInterrupt, match="the previous agent was started again"):
            agent.install(paths, tmp_path / "data", [], {})

        assert noting.lock_was_held == [True] * len(run.commands)
        assert run.subcommands()[-1] == "bootstrap"
        assert not lock_is_held(paths)

    def test_an_uninstall_holds_the_lock(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        paths = self.installed(tmp_path)
        run = FakeRun()
        run.sequences = {"print": [0, NOT_LOADED]}
        noting = self.noting(monkeypatch, run, paths)
        real_read = agent._plist_safe_to_start
        read_under_lock: list[bool] = []

        def reading(target: agent.AgentPaths) -> bytes | None:
            read_under_lock.append(lock_is_held(target))
            return real_read(target)

        monkeypatch.setattr(agent, "_plist_safe_to_start", reading)

        agent.uninstall(paths)

        assert read_under_lock == [True]
        assert run.subcommands() == ["print", "bootout", "print"]
        assert noting.lock_was_held == [True, True, True]
        assert not paths.plist.exists()
        assert not lock_is_held(paths)

    def test_a_restart_holds_the_lock_for_its_look_and_its_restart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths = self.installed(tmp_path)
        run = FakeRun()
        noting = self.noting(monkeypatch, run, paths)

        assert agent.restart_if_loaded(paths) is True

        assert run.subcommands() == ["print", "kickstart"]
        assert noting.lock_was_held == [True, True]
        assert not lock_is_held(paths)

    @pytest.mark.parametrize("command", ["install", "uninstall", "restart"])
    def test_a_command_that_finds_the_lock_held_asks_launchd_for_nothing_until_it_is_let_go(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
    ) -> None:
        paths = self.installed(tmp_path)
        run = launchctl_not_loaded(monkeypatch)
        commands = {
            "install": lambda: agent.install(paths, tmp_path / "data", [], {}),
            "uninstall": lambda: agent.uninstall(paths),
            "restart": lambda: agent.restart_if_loaded(paths),
        }
        held_by_another_command = os.open(installation_lock(paths), os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(held_by_another_command, fcntl.LOCK_EX)
        waiting = threading.Thread(target=commands[command])
        try:
            waiting.start()
            waiting.join(timeout=0.3)
            asked_while_waiting = list(run.commands)
            still_waiting = waiting.is_alive()
        finally:
            os.close(held_by_another_command)
            waiting.join(timeout=SERVER_WAIT_SECONDS)

        assert (still_waiting, asked_while_waiting) == (True, [])
        assert not waiting.is_alive()
        assert run.subcommands()[0] == "print"

    def test_a_symbolic_link_in_place_of_the_lock_is_refused_before_launchd_is_asked_for_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths = self.installed(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.write_text("left alone")
        installation_lock(paths).symlink_to(elsewhere)
        run = FakeRun()
        monkeypatch.setattr(agent.subprocess, "run", run)

        with pytest.raises(ConfigError, match=r"cannot open .*\.plist\.lock as a private file"):
            agent.uninstall(paths)

        assert run.commands == []
        assert paths.plist.read_bytes() == b"previous"
        assert elsewhere.read_text() == "left alone"

    @pytest.mark.parametrize(
        ("in_the_way", "message"),
        [
            pytest.param(
                "Library",
                r"^cannot examine .*LaunchAgents: NotADirectoryError$",
                id="a file where a level above it would go",
            ),
            pytest.param(
                "Library/LaunchAgents",
                r"^cannot use .*LaunchAgents: NotADirectoryError$",
                id="a file where the directory would go",
            ),
        ],
    )
    def test_a_directory_for_the_lock_that_cannot_be_created_is_an_agent_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_the_way: str, message: str
    ) -> None:
        (tmp_path / in_the_way).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / in_the_way).write_text("a file where the directory would go")
        run = FakeRun()
        monkeypatch.setattr(agent.subprocess, "run", run)

        with pytest.raises(agent.AgentError, match=message):
            agent.uninstall(self.paths(tmp_path))

        assert run.commands == []

    def test_a_lock_that_cannot_be_taken_is_an_agent_error_and_its_file_is_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths = self.installed(tmp_path)
        run = FakeRun()
        monkeypatch.setattr(agent.subprocess, "run", run)
        opened: list[tuple[int, int]] = []
        closed: list[int] = []
        real_open_private_file = agent.open_private_file
        real_close = os.close

        def open_lock(path: Path) -> int:
            descriptor = real_open_private_file(path)
            opened.append((descriptor, len(closed)))
            return descriptor

        def refuse(_descriptor: int, _operation: int) -> None:
            raise OSError(errno.ENOLCK, "No locks available")

        def close(descriptor: int) -> None:
            closed.append(descriptor)
            real_close(descriptor)

        monkeypatch.setattr(agent, "open_private_file", open_lock)
        monkeypatch.setattr(agent.fcntl, "flock", refuse)
        monkeypatch.setattr(agent.os, "close", close)

        with pytest.raises(agent.AgentError, match=r"^cannot lock .*\.plist\.lock: OSError$"):
            agent.uninstall(paths)

        (lock_descriptor, closed_before_it), *others = opened
        assert run.commands == []
        assert others == []
        assert closed[closed_before_it:] == [lock_descriptor]
        assert paths.plist.read_bytes() == b"previous"


class RefusesAPlistOthersCanWrite(FakeRun):
    """Stands in for launchctl, and at ``bootstrap`` refuses a property list that others can write to.

    launchd on macOS 27.0.1 was seen (2026-10-04) to refuse one that every account can write to (mode
    666, exit 5) and to accept one that its group can (mode 664). This stand-in refuses both, as a
    stricter launchd would. A refusal uses up none of the scripted statuses.
    """

    WRITABLE_BY_OTHERS = 0o022

    def __init__(self) -> None:
        super().__init__(stderr="Bootstrap failed: 5: Input/output error")
        self.bootstrapped_modes: list[int] = []

    def __call__(self, command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        if command[1] == "bootstrap":
            mode = Path(command[3]).stat().st_mode & 0o777
            self.bootstrapped_modes.append(mode)
            if mode & self.WRITABLE_BY_OTHERS:
                self.commands.append(command)
                return subprocess.CompletedProcess(command, 5, "", self.stderr)
        return super().__call__(command, **options)


INHERITED_WRITE_GRANT = "everyone allow add_file,add_subdirectory,delete_child,file_inherit,directory_inherit"
INHERITED_READ_GRANT = "everyone allow list,search,readattr,file_inherit,directory_inherit"


def access_control_entries(path: Path) -> list[str]:
    listing = subprocess.run(["/bin/ls", "-lde", str(path)], capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in listing.splitlines()[1:]]


@pytest.fixture(params=[0o002, 0o000], ids=["umask 002", "umask 000"])
def permissive_umask(request: pytest.FixtureRequest) -> Iterator[int]:
    permissive: int = request.param
    previous = os.umask(permissive)
    try:
        yield permissive
    finally:
        os.umask(previous)


class TestThePropertyListIsItsOwnersAlone:
    def paths(self, tmp_path: Path) -> agent.AgentPaths:
        return agent.agent_paths(home=tmp_path, user="persona")

    def launchctl(self, monkeypatch: pytest.MonkeyPatch) -> RefusesAPlistOthersCanWrite:
        run = RefusesAPlistOthersCanWrite()
        monkeypatch.setattr(agent.subprocess, "run", run)
        monkeypatch.setattr(agent.time, "sleep", lambda _seconds: None)
        return run

    def mode(self, path: Path) -> int:
        return path.stat().st_mode & 0o777

    def existing_plist_directory(self, paths: agent.AgentPaths) -> None:
        paths.plist.parent.parent.mkdir(mode=0o700)
        paths.plist.parent.mkdir(mode=0o700)

    @pytest.mark.usefixtures("permissive_umask")
    def test_a_first_install_writes_an_owner_only_plist_into_a_directory_only_its_owner_can_write_to(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = self.launchctl(monkeypatch)
        run.returncodes = {"print": NOT_LOADED}
        paths = self.paths(tmp_path)

        agent.install(paths, tmp_path / "data", [], {})

        assert run.bootstrapped_modes == [0o600]
        assert self.mode(paths.plist) == 0o600
        assert [self.mode(level) for level in (paths.plist.parent, paths.plist.parent.parent)] == [0o700, 0o700]
        assert sorted(entry.name for entry in paths.plist.parent.iterdir()) == sorted(
            [paths.plist.name, installation_lock(paths).name]
        )

    @pytest.mark.usefixtures("permissive_umask")
    def test_a_reinstall_launchd_rejects_puts_back_a_plist_launchd_accepts_and_starts_the_previous_agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = self.launchctl(monkeypatch)
        run.sequences = {"print": [0, NOT_LOADED, NOT_LOADED], "bootstrap": [5, 0]}
        paths = self.paths(tmp_path)
        self.existing_plist_directory(paths)
        paths.plist.write_bytes(b"previous")
        paths.plist.chmod(0o600)

        with pytest.raises(agent.AgentError) as raised:
            agent.install(paths, tmp_path / "data", [], {})

        assert str(raised.value) == (
            "launchctl bootstrap failed (5): Bootstrap failed: 5: Input/output error; "
            "the previous agent was started again"
        )
        assert run.bootstrapped_modes == [0o600, 0o600]
        assert (paths.plist.read_bytes(), self.mode(paths.plist)) == (b"previous", 0o600)
        assert run.subcommands() == ["print", "bootout", "print", "bootstrap", "print", "bootstrap"]

    @pytest.mark.usefixtures("permissive_umask")
    def test_an_interrupted_reinstall_puts_back_a_plist_launchd_accepts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = self.launchctl(monkeypatch)
        run.sequences = {"print": [0, NOT_LOADED, NOT_LOADED]}
        paths = self.paths(tmp_path)
        self.existing_plist_directory(paths)
        paths.plist.write_bytes(b"previous")
        paths.plist.chmod(0o600)
        real_bootstrap = run.__call__

        def interrupted_at_the_first_bootstrap(command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
            if command[1] == "bootstrap" and not run.bootstrapped_modes:
                run.bootstrapped_modes.append(self.mode(Path(command[3])))
                run.commands.append(command)
                raise KeyboardInterrupt
            return real_bootstrap(command, **options)

        monkeypatch.setattr(agent.subprocess, "run", interrupted_at_the_first_bootstrap)

        with pytest.raises(KeyboardInterrupt) as raised:
            agent.install(paths, tmp_path / "data", [], {})

        assert str(raised.value) == "the previous agent was started again"
        assert run.bootstrapped_modes == [0o600, 0o600]
        assert (paths.plist.read_bytes(), self.mode(paths.plist)) == (b"previous", 0o600)

    @pytest.mark.parametrize("change", ["group-writable", "a symbolic link", "a write grant"])
    def test_an_installed_plist_another_account_could_have_changed_is_refused_before_anything_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
    ) -> None:
        if change == "a write grant" and sys.platform != "darwin":
            pytest.skip("access-control lists of this kind are a macOS feature")
        paths = self.paths(tmp_path)
        self.existing_plist_directory(paths)
        elsewhere = tmp_path / "elsewhere.plist"
        elsewhere.write_bytes(b"previous")
        elsewhere.chmod(0o600)
        if change == "a symbolic link":
            paths.plist.symlink_to(elsewhere)
        else:
            paths.plist.write_bytes(b"previous")
            paths.plist.chmod(0o664 if change == "group-writable" else 0o600)
        if change == "a write grant":
            subprocess.run(["/bin/chmod", "+a", "everyone allow write", str(paths.plist)], check=True)
        run = self.launchctl(monkeypatch)

        with pytest.raises(
            agent.AgentError, match=r"plist: (it is a symbolic link|.*so another account could change what it holds)"
        ):
            agent.install(paths, tmp_path / "data", [], {})

        assert run.subcommands() == []
        assert paths.plist.read_bytes() == b"previous"

    def test_an_installed_plist_only_its_owner_can_change_is_read_and_put_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = self.launchctl(monkeypatch)
        run.sequences = {"print": [0, NOT_LOADED, NOT_LOADED], "bootstrap": [5, 0]}
        paths = self.paths(tmp_path)
        self.existing_plist_directory(paths)
        paths.plist.write_bytes(b"previous")
        paths.plist.chmod(0o644)

        with pytest.raises(agent.AgentError, match="the previous agent was started again"):
            agent.install(paths, tmp_path / "data", [], {})

        assert paths.plist.read_bytes() == b"previous"

    @pytest.mark.parametrize("level", ["Logs", "Library", "home"])
    def test_a_level_above_the_log_directory_another_account_can_change_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, level: str
    ) -> None:
        run = self.launchctl(monkeypatch)
        paths = self.paths(tmp_path)
        self.existing_plist_directory(paths)
        paths.log_dir.parent.mkdir(mode=0o700)
        writable = {"Logs": paths.log_dir.parent, "Library": paths.log_dir.parent.parent, "home": tmp_path}[level]
        writable.chmod(0o770)
        try:
            with pytest.raises(agent.AgentError, match=r"let other accounts write to it"):
                agent.install(paths, tmp_path / "data", [], {})
        finally:
            writable.chmod(0o700)

        assert run.subcommands() == []

    def test_a_private_home_under_a_directory_another_account_can_replace_is_refused_before_anything_is_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(0o777)
        home = shared / "persona"
        home.mkdir(mode=0o700)
        paths = agent.agent_paths(home=home, user="persona")
        data = tmp_path / "data"
        run = self.launchctl(monkeypatch)
        message = (
            f"cannot use {shared}: its permissions (0777) let other accounts write to it, so another "
            "account could replace what is in it; make it writable by you alone and run the command again"
        )
        try:
            with pytest.raises(agent.AgentError) as installed:
                agent.install(paths, data, [], {})
            with pytest.raises(agent.AgentError) as prepared:
                agent._prepare_plist_directory(paths.plist.parent)
        finally:
            shared.chmod(0o700)

        assert str(installed.value) == message
        assert str(prepared.value) == message
        assert run.commands == []
        assert not paths.log_dir.exists()
        assert not paths.plist.parent.exists()
        assert not data.exists()

    def test_the_stand_in_refuses_what_a_plain_write_creates_under_either_umask(
        self, tmp_path: Path, permissive_umask: int
    ) -> None:
        plist = tmp_path / "written-without-a-stated-mode.plist"
        plist.write_bytes(b"previous")
        run = RefusesAPlistOthersCanWrite()

        refused = run(["launchctl", "bootstrap", "gui/501", str(plist)])

        assert self.mode(plist) == 0o666 & ~permissive_umask
        assert (refused.returncode, run.bootstrapped_modes) == (5, [0o666 & ~permissive_umask])

    def test_a_directory_that_already_exists_is_left_with_the_permissions_it_has(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = self.launchctl(monkeypatch)
        run.returncodes = {"print": NOT_LOADED}
        paths = self.paths(tmp_path)
        paths.plist.parent.mkdir(parents=True)
        paths.plist.parent.chmod(0o755)

        agent.install(paths, tmp_path / "data", [], {})

        assert (self.mode(paths.plist.parent), self.mode(paths.plist)) == (0o755, 0o600)

    @pytest.mark.skipif(sys.platform != "darwin", reason="access-control lists of this kind are a macOS feature")
    def test_a_directory_created_for_the_plist_inherits_no_access_from_the_directory_above_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths = self.paths(tmp_path)
        library = paths.plist.parent.parent
        library.mkdir(mode=0o700)
        subprocess.run(["/bin/chmod", "+a", INHERITED_READ_GRANT, str(library)], check=True)
        granted = access_control_entries(library)
        run = self.launchctl(monkeypatch)
        run.returncodes = {"print": NOT_LOADED}

        agent.install(paths, tmp_path / "data", [], {})

        monkeypatch.undo()
        created = (paths.plist.parent, paths.plist, installation_lock(paths))
        assert len(granted) == 1
        assert "everyone allow list" in granted[0]
        assert access_control_entries(library) == granted
        assert [access_control_entries(path) for path in created] == [[], [], []]

    def grant(self, path: Path, entry: str) -> list[str]:
        subprocess.run(["/bin/chmod", "+a", entry, str(path)], check=True)
        return access_control_entries(path)

    def refused(self, paths: agent.AgentPaths, tmp_path: Path) -> str:
        with pytest.raises(agent.AgentError) as raised:
            agent.install(paths, tmp_path / "data", [], {})
        assert not paths.plist.exists()
        assert sorted(entry.name for entry in paths.plist.parent.iterdir()) == []
        return str(raised.value)

    @pytest.mark.parametrize("level", ["LaunchAgents", "Library", "the home directory"])
    @pytest.mark.parametrize("mode", [0o777, 0o775, 0o757])
    def test_an_existing_directory_others_can_write_to_is_refused_and_left_as_it_is(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, level: str, mode: int
    ) -> None:
        paths = self.paths(tmp_path)
        self.existing_plist_directory(paths)
        unsafe = {
            "LaunchAgents": paths.plist.parent,
            "Library": paths.plist.parent.parent,
            "the home directory": tmp_path,
        }[level]
        unsafe.chmod(mode)
        run = self.launchctl(monkeypatch)
        try:
            message = self.refused(paths, tmp_path)
        finally:
            tmp_path.chmod(0o700)

        assert message == (
            f"cannot use {unsafe}: its permissions ({mode:04o}) let other accounts write to it, so another "
            "account could replace what is in it; make it writable by you alone and run the command again"
        )
        assert run.commands == []
        assert self.mode(unsafe) == (0o700 if level == "the home directory" else mode)

    @pytest.mark.skipif(sys.platform != "darwin", reason="access-control lists of this kind are a macOS feature")
    @pytest.mark.parametrize("level", ["LaunchAgents", "Library"])
    def test_an_existing_directory_whose_access_control_list_lets_others_write_is_refused_and_left_as_it_is(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, level: str
    ) -> None:
        paths = self.paths(tmp_path)
        self.existing_plist_directory(paths)
        unsafe = paths.plist.parent if level == "LaunchAgents" else paths.plist.parent.parent
        granted = self.grant(unsafe, INHERITED_WRITE_GRANT)
        self.launchctl(monkeypatch)

        message = self.refused(paths, tmp_path)

        monkeypatch.undo()
        assert message.startswith(f"cannot use {unsafe}: an access-control entry lets another account change it")
        assert access_control_entries(unsafe) == granted

    @pytest.mark.skipif(sys.platform != "darwin", reason="access-control lists of this kind are a macOS feature")
    def test_an_existing_directory_whose_access_control_list_only_denies_or_lets_others_read_is_used_as_it_is(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths = self.paths(tmp_path)
        self.existing_plist_directory(paths)
        self.grant(paths.plist.parent.parent, "everyone deny delete")
        granted = self.grant(paths.plist.parent, INHERITED_READ_GRANT)
        run = self.launchctl(monkeypatch)
        run.returncodes = {"print": NOT_LOADED}

        agent.install(paths, tmp_path / "data", [], {})

        monkeypatch.undo()
        assert self.mode(paths.plist) == 0o600
        assert access_control_entries(paths.plist.parent) == granted

    def test_an_existing_directory_that_is_a_symbolic_link_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths = self.paths(tmp_path)
        paths.plist.parent.parent.mkdir(mode=0o700)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir(mode=0o700)
        paths.plist.parent.symlink_to(elsewhere, target_is_directory=True)
        run = self.launchctl(monkeypatch)

        with pytest.raises(agent.AgentError, match=r"^cannot use .*LaunchAgents: it is a symbolic link$"):
            agent.install(paths, tmp_path / "data", [], {})

        assert list(elsewhere.iterdir()) == []
        assert run.commands == []

    def test_an_uninstall_is_refused_too_before_it_takes_the_lock_in_a_directory_others_can_write_to(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths = self.paths(tmp_path)
        self.existing_plist_directory(paths)
        paths.plist.parent.chmod(0o777)
        run = self.launchctl(monkeypatch)

        with pytest.raises(agent.AgentError, match=r"let other accounts write to it"):
            agent.uninstall(paths)

        assert list(paths.plist.parent.iterdir()) == []
        assert run.commands == []

    def appearing_before_it_is_created(
        self, monkeypatch: pytest.MonkeyPatch, directory: Path, create: Callable[[Path], None]
    ) -> None:
        make_private = agent.ensure_private_directory

        def something_else_creates_it_first(path: Path) -> None:
            if path == directory:
                create(path)
            make_private(path)

        monkeypatch.setattr(agent, "ensure_private_directory", something_else_creates_it_first)

    def test_a_directory_something_else_creates_first_open_to_everyone_is_still_made_owner_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = self.launchctl(monkeypatch)
        run.returncodes = {"print": NOT_LOADED}
        paths = self.paths(tmp_path)
        paths.plist.parent.parent.mkdir(mode=0o700)

        def open_to_everyone(directory: Path) -> None:
            directory.mkdir()
            directory.chmod(0o777)

        self.appearing_before_it_is_created(monkeypatch, paths.plist.parent, open_to_everyone)

        agent.install(paths, tmp_path / "data", [], {})

        assert (self.mode(paths.plist.parent), self.mode(paths.plist)) == (0o700, 0o600)

    def test_a_symbolic_link_that_appears_where_the_directory_would_go_is_refused_and_nothing_is_put_through_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = self.launchctl(monkeypatch)
        paths = self.paths(tmp_path)
        paths.plist.parent.parent.mkdir(mode=0o700)
        shared = tmp_path / "shared"
        shared.mkdir(mode=0o777)
        self.appearing_before_it_is_created(
            monkeypatch, paths.plist.parent, lambda directory: directory.symlink_to(shared)
        )

        with pytest.raises(
            agent.AgentError, match=r"^cannot make .*LaunchAgents a private directory: NotADirectoryError$"
        ):
            agent.install(paths, tmp_path / "data", [], {})

        assert list(shared.iterdir()) == []
        assert run.commands == []

    def test_a_plist_that_cannot_be_created_as_an_owner_only_file_is_an_agent_error_and_stops_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = self.launchctl(monkeypatch)
        paths = self.paths(tmp_path)
        paths.plist.parent.mkdir(parents=True)
        installation_lock(paths).touch()
        paths.plist.parent.chmod(0o500)

        try:
            with pytest.raises(agent.AgentError) as raised:
                agent.install(paths, tmp_path / "data", [], {})
        finally:
            paths.plist.parent.chmod(0o700)

        assert str(raised.value).startswith(f"cannot write {paths.plist}: cannot open ")
        assert str(raised.value).endswith(".tmp as a private file: PermissionError")
        assert run.subcommands() == ["print"]
        assert not paths.plist.exists()


class TestLaunchAgent:
    @pytest.mark.parametrize("installed", [False, True])
    def test_required_helper_survives_an_unrelated_unsafe_system_entry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        installed: bool,
    ) -> None:
        system = tmp_path / "system-bin"
        helper = self.executable(system, "security")
        helper.write_text("#!/bin/sh\nprintf 'synthetic-helper\\n'\n")
        foreign = self.executable(system, "foreign")
        vendor = self.executable(tmp_path / "vendor-bin", "vendor")
        vendor.write_text("#!/bin/sh\nexec security\n")
        check = executables.what_lets_another_account_change_by_path
        execute = subprocess.run

        def ownership(path: Path, **options: bool) -> str | None:
            return "it belongs to another account" if path == foreign else check(path, **options)

        monkeypatch.setattr(executables, "what_lets_another_account_change_by_path", ownership)
        monkeypatch.setattr(agent, "SYSTEM_PATH_DIRECTORIES", (str(system),))
        monkeypatch.setattr(agent, "REQUIRED_PATH_HELPERS", (helper,), raising=False)
        monkeypatch.setattr(executables, "REQUIRED_PATH_HELPERS", (helper,))
        monkeypatch.setenv("PATH", str(system))
        launchctl_not_loaded(monkeypatch)
        if installed:
            paths = self.paths(tmp_path)
            agent.install(paths, tmp_path / "data", [str(vendor)], {})
            environment = plistlib.loads(paths.plist.read_bytes())["EnvironmentVariables"]
        else:
            environment = executables.execution_environment(
                {"PATH": str(system)},
                tmp_path,
                helper_directory=tmp_path / "data",
            )
        assert str(system) not in environment["PATH"].split(os.pathsep)
        result = execute([str(vendor)], env=environment, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert result.stdout == "synthetic-helper\n"

    @pytest.mark.parametrize("existing", [False, True])
    def test_required_helper_fallback_is_private_and_reusable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
    ) -> None:
        helper = self.executable(tmp_path / "system", "security")
        monkeypatch.setattr(agent, "REQUIRED_PATH_HELPERS", (helper,))
        data = tmp_path / "data"
        data.mkdir()
        directory = data / "agent-bin"
        if existing:
            directory.mkdir(mode=0o700)
            (directory / "security").symlink_to(helper)
        search_path = agent._with_required_helpers("", data)
        assert search_path == str(directory)
        assert directory.stat().st_mode & 0o777 == 0o700
        assert (directory / "security").resolve() == helper
        assert agent._with_required_helpers(search_path, data) == search_path

    @pytest.mark.parametrize(
        "kind",
        ["wrong-link", "regular-file", "shared-directory", "linked-directory", "extra-unsafe", "colon", "write-error"],
    )
    def test_required_helper_fallback_refuses_untrusted_or_unusable_contents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
    ) -> None:
        helper = self.executable(tmp_path / "system", "security")
        monkeypatch.setattr(agent, "REQUIRED_PATH_HELPERS", (helper,))
        data = tmp_path / ("data:colon" if kind == "colon" else "data")
        data.mkdir()
        directory = data / "agent-bin"
        if kind == "linked-directory":
            directory.symlink_to(helper.parent)
        else:
            directory.mkdir()
            if kind == "wrong-link":
                (directory / "security").symlink_to(helper.parent / "other")
            elif kind == "regular-file":
                (directory / "security").touch()
            elif kind == "shared-directory":
                directory.chmod(0o777)
            elif kind == "extra-unsafe":
                self.executable(directory, "other").chmod(0o777)
            elif kind == "write-error":
                directory.chmod(0o500)
        try:
            with pytest.raises(ConfigError):
                agent._with_required_helpers("", data)
        finally:
            if kind == "write-error":
                directory.chmod(0o700)

    def test_an_unsafe_required_helper_stops_install_before_launchctl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        helper = self.executable(tmp_path / "system", "security")
        helper.chmod(0o777)
        monkeypatch.setattr(agent, "REQUIRED_PATH_HELPERS", (helper,))
        monkeypatch.setattr(agent, "SYSTEM_PATH_DIRECTORIES", ())
        monkeypatch.setenv("PATH", "")
        run = launchctl_not_loaded(monkeypatch)
        paths = self.paths(tmp_path)
        with pytest.raises(ConfigError, match="required helper"):
            agent.install(paths, tmp_path / "data", [], {})
        assert run.commands == []
        assert not paths.plist.exists()

    def paths(self, tmp_path: Path) -> agent.AgentPaths:
        return agent.agent_paths(home=tmp_path, user="persona")

    def beside_the_lock(self, paths: agent.AgentPaths) -> list[str]:
        lock = installation_lock(paths).name
        return sorted(entry.name for entry in paths.plist.parent.iterdir() if entry.name != lock)

    def test_the_agent_runs_the_server_at_login_and_restarts_it_if_it_exits(self, tmp_path: Path) -> None:
        paths = self.paths(tmp_path)

        plist = plistlib.loads(agent.build_plist(paths, "/venv/bin/python", tmp_path / "data", "/opt/bin:/usr/bin", {}))

        assert plist["Label"] == "com.persona.ai-usage-dashboard"
        assert plist["ProgramArguments"] == ["/venv/bin/python", "-P", "-m", "usagedash", "serve"]
        assert (plist["RunAtLoad"], plist["KeepAlive"]) == (True, True)
        assert plist["EnvironmentVariables"] == {"PATH": "/opt/bin:/usr/bin"}
        assert plist["StandardErrorPath"] == str(
            tmp_path / "Library" / "Logs" / "ai-usage-dashboard" / "launchd-stderr.log"
        )

    def test_the_agent_inherits_the_settings_overrides_given_at_install_but_keeps_its_own_search_path(
        self, tmp_path: Path
    ) -> None:
        overrides = {"USAGEDASH_PORT": "9001", "USAGEDASH_DATA_DIR": "/elsewhere", "PATH": "/attacker/bin"}

        plist = plistlib.loads(
            agent.build_plist(self.paths(tmp_path), "/venv/bin/python", tmp_path, "/opt/bin:/usr/bin", overrides)
        )

        assert plist["EnvironmentVariables"] == {
            "USAGEDASH_PORT": "9001",
            "USAGEDASH_DATA_DIR": "/elsewhere",
            "PATH": "/opt/bin:/usr/bin",
        }

    def executable(self, directory: Path, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        program = directory / name
        program.write_text("#!/bin/sh\n")
        program.chmod(0o755)
        return program

    def test_the_agent_path_keeps_each_directory_of_the_installing_shell_s_path_once_then_the_system_ones(
        self, tmp_path: Path
    ) -> None:
        home_bin, brew_bin, other = tmp_path / "home" / "bin", tmp_path / "brew" / "bin", tmp_path / "other"
        self.executable(home_bin, "claude")
        self.executable(brew_bin, "codex")
        self.executable(home_bin, "grok")
        other.mkdir()
        shell_path = f"{home_bin}:{other}:{brew_bin}:{home_bin}:/usr/bin:/bin"

        search_path = agent.agent_search_path(["claude", "codex", "grok", "missing-cli"], shell_path)

        assert search_path == f"{home_bin}:{other}:{brew_bin}:/usr/bin:/bin:/usr/sbin:/sbin"

    def test_a_runtime_installed_apart_from_the_cli_that_needs_it_is_on_the_agent_path(self, tmp_path: Path) -> None:
        cli_bin, runtime_bin = tmp_path / "cli" / "bin", tmp_path / "nvm" / "bin"
        self.executable(cli_bin, "codex")
        runtime = self.executable(runtime_bin, "usagedash-test-runtime")
        shell_path = f"{cli_bin}:{runtime_bin}:/usr/bin:/bin"

        search_path = agent.agent_search_path(["codex"], shell_path)

        assert shutil.which("usagedash-test-runtime", path=search_path) == str(runtime)

    def test_the_agent_path_finds_the_installation_of_each_cli_that_the_installing_shell_finds(
        self, tmp_path: Path
    ) -> None:
        preferred, local = tmp_path / "preferred" / "bin", tmp_path / "local" / "bin"
        chosen_codex = self.executable(preferred, "codex")
        self.executable(local, "claude")
        self.executable(local, "codex")
        shell_path = f"{preferred}:{local}"

        search_path = agent.agent_search_path(["claude", "codex", "grok"], shell_path)

        assert search_path.split(":")[:2] == [str(preferred), str(local)]
        assert shutil.which("codex", path=search_path) == str(chosen_codex)
        for name in ("claude", "codex"):
            assert shutil.which(name, path=search_path) == shutil.which(name, path=shell_path)

    def test_a_system_directory_keeps_the_place_the_installing_shell_gave_it(self, tmp_path: Path) -> None:
        local = tmp_path / "local" / "bin"
        self.executable(local, "claude")

        search_path = agent.agent_search_path(["claude"], f"/usr/bin:{local}:/bin")

        assert search_path == f"/usr/bin:{local}:/bin:/usr/sbin:/sbin"

    def test_a_relative_directory_on_the_installing_shell_s_path_is_made_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.executable(tmp_path / "bin", "codex")
        monkeypatch.chdir(tmp_path)

        search_path = agent.agent_search_path(["codex"], "bin::/usr/bin")

        assert search_path == f"{tmp_path / 'bin'}:/usr/bin:/bin:/usr/sbin:/sbin"
        assert all(Path(directory).is_absolute() for directory in search_path.split(":"))

    def test_the_agent_path_holds_canonical_directories_and_leaves_out_what_another_account_could_change(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        shared, under_shared, kept = tmp_path / "shared", tmp_path / "shared" / "bin", tmp_path / "kept" / "bin"
        self.executable(under_shared, "codex")
        self.executable(kept, "claude")
        shared.chmod(0o777)
        alias = tmp_path / "alias-bin"
        alias.symlink_to(kept)
        not_there = tmp_path / "not-there"
        sticky_alias = Path(os.sep, "tmp") / f"usagedash-alias-{os.getpid()}"
        sticky_alias.symlink_to(Path(os.sep, "usr", "bin"))
        shell_path = f"{shared}:{under_shared}:/tmp:/tmp/not-yet:{alias}:{kept}:{not_there}:{sticky_alias}:/usr/bin"
        try:
            with caplog.at_level(logging.WARNING, logger="usagedash.agent"):
                search_path = agent.agent_search_path(["codex", "claude"], shell_path)
        finally:
            shared.chmod(0o700)
            sticky_alias.unlink()

        assert search_path == f"{kept}:/usr/bin:/bin:/usr/sbin:/sbin"
        assert len(caplog.messages) == 1
        assert caplog.messages[0].startswith("left out of the agent's PATH,")
        for directory in (shared, under_shared, Path(os.sep, "tmp"), Path(os.sep, "tmp", "not-yet"), not_there):
            assert f"{directory}:" in caplog.messages[0]

    def test_a_path_alias_to_a_directory_holding_a_colon_is_left_out(self, tmp_path: Path) -> None:
        odd = tmp_path / "odd:dir"
        odd.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(odd)
        assert agent.agent_search_path([], str(alias)) == "/usr/bin:/bin:/usr/sbin:/sbin"

    @pytest.mark.parametrize(
        "kind", ["shared-target", "shared-file", "missing-shared-target", "safe-target", "missing-safe-target"]
    )
    def test_path_executable_entries_and_link_targets_must_be_safe(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, kind: str
    ) -> None:
        directory = tmp_path / "bin"
        directory.mkdir()
        target = self.executable(tmp_path / "target", "node")
        entry = directory / "node"
        if kind == "shared-file":
            entry = self.executable(directory, "node")
            entry.chmod(0o777)
        else:
            if kind == "shared-target":
                target.parent.chmod(0o777)
            elif kind == "missing-shared-target":
                target = Path(os.sep, "tmp") / "usagedash-missing-runtime"
            elif kind == "missing-safe-target":
                target.unlink()
            entry.symlink_to(target)
        search_path = agent.agent_search_path([], str(directory))
        safe = kind in {"safe-target", "missing-safe-target"}
        assert (str(directory) in search_path.split(os.pathsep)) == safe
        if not safe:
            assert str(entry) in caplog.text

    @pytest.mark.parametrize("chain", [False, True])
    def test_a_helper_link_cannot_retain_a_replaceable_intermediate_link(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, chain: bool
    ) -> None:
        target = self.executable(tmp_path / "safe", "node")
        shared = tmp_path / "shared"
        shared.mkdir(mode=0o777)
        shared.chmod(0o777)
        alias = shared / "alias"
        alias.symlink_to(target)
        if chain:
            intermediate = tmp_path / "intermediate"
            intermediate.symlink_to(alias)
            alias = intermediate
        directory = tmp_path / "bin"
        directory.mkdir()
        (directory / "node").symlink_to(alias)
        assert str(directory) not in agent.agent_search_path([], str(directory)).split(os.pathsep)
        assert str(directory / "node") in caplog.text

    @pytest.mark.parametrize("kind", ["plain-file", "safe-plain", "directory", "pipe", "cycle", "broken-parent"])
    def test_path_entry_kinds(self, tmp_path: Path, kind: str) -> None:
        directory = tmp_path / "bin"
        directory.mkdir()
        entry = directory / "entry"
        if kind in {"plain-file", "safe-plain"}:
            entry.touch()
            entry.chmod(0o666 if kind == "plain-file" else 0o644)
        elif kind == "directory":
            entry.mkdir()
        elif kind == "pipe":
            os.mkfifo(entry)
        elif kind == "cycle":
            entry.symlink_to(entry)
        else:
            entry.symlink_to(tmp_path / "missing" / "entry")
        assert (str(directory) in agent.agent_search_path([], str(directory)).split(os.pathsep)) == (
            kind in {"safe-plain", "directory", "pipe"}
        )

    @pytest.mark.parametrize("owner", ["self", "root", "other"])
    @pytest.mark.parametrize("right", ["writesecurity", "chown"])
    def test_non_executable_path_files_cannot_gain_execution_through_ownership_or_acls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: str, right: str
    ) -> None:
        program = self.executable(tmp_path, "node")
        program.chmod(0o600)
        values = list(program.stat())
        values[4] = {"self": os.geteuid(), "root": 0, "other": os.geteuid() + 1}[owner]
        status = os.stat_result(values)
        original = os.lstat

        class Entry:
            path = str(program)

            def stat(self, *, follow_symlinks: bool) -> os.stat_result:
                return status

        @contextlib.contextmanager
        def scan(_directory: Path) -> Iterator[Iterator[Entry]]:
            yield iter([Entry()])

        def foreign_owner(path: str | os.PathLike[str], *, dir_fd: int | None = None) -> os.stat_result:
            return status if os.fspath(path) == str(program) else original(path, dir_fd=dir_fd)

        monkeypatch.setattr(agent.os, "scandir", scan)
        monkeypatch.setattr(private.os, "lstat", foreign_owner)
        monkeypatch.setattr(
            private, "_access_control_text_of", lambda _path: f"group:synthetic:everyone:12:allow:write,{right}\n"
        )
        with pytest.raises(ConfigError, match=r"belongs to another account|access-control entry"):
            executables._refuse_an_executable_entry_another_account_could_change(tmp_path)

    def test_a_foreign_helper_link_under_a_root_sticky_directory_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "bin"
        directory.mkdir()
        target = self.executable(tmp_path / "safe", "node")
        alias = Path(os.sep, "tmp") / f"usagedash-foreign-helper-{os.getpid()}"
        alias.symlink_to(target)
        (directory / "node").symlink_to(alias)
        original = os.lstat

        def foreign_owner(path: str | os.PathLike[str], *, dir_fd: int | None = None) -> os.stat_result:
            status = original(path, dir_fd=dir_fd)
            if os.fspath(path) == str(alias):
                values = list(status)
                values[4] = os.geteuid() + 1
                return os.stat_result(values)
            return status

        monkeypatch.setattr(agent.os, "lstat", foreign_owner)
        try:
            assert str(directory) not in agent.agent_search_path([], str(directory)).split(os.pathsep)
        finally:
            alias.unlink()

    @pytest.mark.parametrize(
        "kind",
        ["shared-parent", "writable-file", "retained-link", "missing", "cycle", "nul", "directory", "not-executable"],
    )
    def test_install_refuses_an_unsafe_python_before_writing_or_stopping_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
    ) -> None:
        python = self.executable(tmp_path / "venv" / "bin", "python")
        if kind == "shared-parent":
            python.parent.chmod(0o777)
        elif kind == "writable-file":
            python.chmod(0o777)
        elif kind == "directory":
            python.unlink()
            python.mkdir()
        elif kind == "not-executable":
            python.chmod(0o600)
        elif kind == "nul":
            python = Path(f"{python}\x00")
        elif kind == "retained-link":
            target = python
            python = tmp_path / "shared" / "python"
            python.parent.mkdir()
            python.parent.chmod(0o777)
            python.symlink_to(target)
        else:
            python.unlink()
            if kind == "cycle":
                python.symlink_to(python)
        monkeypatch.setattr(agent.sys, "executable", str(python))
        run = launchctl_not_loaded(monkeypatch)
        paths = self.paths(tmp_path)
        with pytest.raises(ConfigError):
            agent.install(paths, tmp_path / "data", [], {})
        assert run.commands == []
        assert not paths.plist.exists()
        assert not paths.log_dir.exists()

    @pytest.mark.parametrize("directory_link", [False, True])
    def test_install_preserves_the_checked_virtual_environment_interpreter_spelling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory_link: bool
    ) -> None:
        base = self.executable(tmp_path / "base", "python")
        if directory_link:
            alias = tmp_path / "opt-python"
            alias.symlink_to(base.parent)
            base = alias / base.name
        python = tmp_path / "venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.symlink_to(base)
        monkeypatch.setattr(agent.sys, "executable", str(python))
        launchctl_not_loaded(monkeypatch)
        paths = self.paths(tmp_path)
        agent.install(paths, tmp_path / "data", [], {})
        assert plistlib.loads(paths.plist.read_bytes())["ProgramArguments"][0] == str(python)

    def test_a_directory_only_the_administrators_group_can_write_stays_on_the_agent_path(self, tmp_path: Path) -> None:
        administrators = private.ADMINISTRATORS_GROUP_ID
        if administrators is None or administrators not in os.getgroups():
            pytest.skip("this account is not in an administrators group")
        prefix = tmp_path / "homebrew" / "bin"
        self.executable(prefix, "codex")
        for directory in (prefix.parent, prefix):
            os.chown(directory, -1, administrators)
            directory.chmod(0o775)

        assert agent.agent_search_path(["codex"], f"{prefix}:/usr/bin") == f"{prefix}:/usr/bin:/bin:/usr/sbin:/sbin"

    def test_a_cli_configured_by_a_path_the_shell_does_not_search_has_its_directory_added_after_the_shell_s(
        self, tmp_path: Path
    ) -> None:
        on_path, elsewhere = tmp_path / "on-path" / "bin", tmp_path / "elsewhere" / "bin"
        self.executable(on_path, "claude")
        configured = self.executable(elsewhere, "grok")

        search_path = agent.agent_search_path(["claude", str(configured)], str(on_path))

        assert search_path == f"{on_path}:{elsewhere}:/usr/bin:/bin:/usr/sbin:/sbin"

    def test_a_configured_cli_in_a_directory_whose_name_holds_a_colon_adds_nothing_to_the_agent_path(
        self, tmp_path: Path
    ) -> None:
        foreign = tmp_path / "foreign"
        self.executable(foreign, "codex")
        odd = Path(f"{tmp_path / 'odd'}:{foreign}")
        configured = self.executable(odd, "claude")

        search_path = agent.agent_search_path([str(configured)], "/usr/bin")

        assert search_path == "/usr/bin:/bin:/usr/sbin:/sbin"
        assert str(foreign) not in search_path.split(os.pathsep)

    def test_the_agent_path_is_taken_from_this_process_s_environment_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        local = tmp_path / "local" / "bin"
        self.executable(local, "claude")
        monkeypatch.setenv("PATH", f"{local}:/usr/bin")
        with_a_path = agent.agent_search_path(["claude"])
        monkeypatch.delenv("PATH")

        assert with_a_path == f"{local}:/usr/bin:/bin:/usr/sbin:/sbin"
        assert agent.agent_search_path(["claude"]) == "/usr/bin:/bin:/usr/sbin:/sbin"

    def test_install_writes_the_plist_and_replaces_a_loaded_agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun()
        run.sequences = {"print": [0, 0, 0, NOT_LOADED]}
        pauses: list[float] = []
        monkeypatch.setattr(agent.subprocess, "run", run)
        monkeypatch.setattr(agent.time, "sleep", pauses.append)
        paths = self.paths(tmp_path)

        agent.install(paths, tmp_path / "data", [], {})

        assert plistlib.loads(paths.plist.read_bytes())["Label"] == paths.label
        assert run.subcommands() == ["print", "bootout", "print", "print", "print", "bootstrap"]
        assert run.commands[-1][2:] == [paths.domain_target, str(paths.plist)]
        assert len(pauses) == 2

    def test_the_agent_is_named_after_the_account_in_the_user_database_not_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("USER", "../../elsewhere")
        monkeypatch.setenv("LOGNAME", "../../elsewhere")

        paths = agent.agent_paths(home=tmp_path)

        account = pwd.getpwuid(os.getuid()).pw_name
        assert paths.label == f"com.{account}.ai-usage-dashboard"
        assert paths.plist.parent == tmp_path / "Library" / "LaunchAgents"

    def test_changing_home_does_not_move_the_plist_or_its_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        account_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        plist_dir = account_home / "Library" / "LaunchAgents"
        before = sorted(path.name for path in plist_dir.iterdir()) if plist_dir.is_dir() else None
        monkeypatch.setenv("HOME", str(tmp_path / "one"))
        monkeypatch.setenv("USER", "not-the-account")
        first = agent.agent_paths()
        monkeypatch.setenv("HOME", str(tmp_path / "two"))
        second = agent.agent_paths()

        label = f"com.{pwd.getpwuid(os.getuid()).pw_name}.ai-usage-dashboard"
        expected = plist_dir / f"{label}.plist"
        assert first.plist == second.plist == expected
        assert first.label == second.label == label
        assert first.service_target == second.service_target
        assert installation_lock(first) == installation_lock(second)
        assert installation_lock(first).parent == plist_dir
        after = sorted(path.name for path in plist_dir.iterdir()) if plist_dir.is_dir() else None
        assert after == before
        assert not (tmp_path / "one" / "Library").exists()
        assert not (tmp_path / "two" / "Library").exists()

    def test_an_account_home_that_is_not_absolute_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Record:
            pw_dir = "relative"
            pw_name = "persona"

        monkeypatch.setattr(agent.pwd, "getpwuid", lambda _uid: Record())

        with pytest.raises(agent.AgentError, match="not an absolute path"):
            agent.agent_paths()

    def test_an_account_with_no_home_directory_in_the_user_database_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing(_uid: int) -> object:
            raise KeyError(_uid)

        monkeypatch.setattr(agent.pwd, "getpwuid", missing)

        with pytest.raises(agent.AgentError, match="account home cannot be read"):
            agent.account_home()

    @pytest.mark.parametrize("user", ["", "../escape", "has space", "a/b", ".hidden", "trailing.", "two..dots", "x\ny"])
    def test_an_account_name_that_cannot_be_part_of_a_label_and_a_file_name_is_refused(
        self, tmp_path: Path, user: str
    ) -> None:
        with pytest.raises(agent.AgentError, match="cannot be used in a launchd label"):
            agent.agent_paths(home=tmp_path, user=user)

    @pytest.mark.parametrize("user", ["persona", "first.last", "user_1", "A-b"])
    def test_ordinary_account_names_including_dotted_ones_are_accepted(self, tmp_path: Path, user: str) -> None:
        assert agent.agent_paths(home=tmp_path, user=user).label == f"com.{user}.ai-usage-dashboard"

    def test_install_makes_the_log_and_data_directories_owner_only_even_if_they_existed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        launchctl_not_loaded(monkeypatch)
        paths = self.paths(tmp_path)
        paths.log_dir.mkdir(parents=True)
        paths.log_dir.chmod(0o755)
        (tmp_path / "data").mkdir(mode=0o755)

        agent.install(paths, tmp_path / "data", [], {})

        assert paths.log_dir.stat().st_mode & 0o777 == 0o700
        assert (tmp_path / "data").stat().st_mode & 0o777 == 0o700

    def test_a_plist_that_cannot_be_read_is_an_agent_error_and_the_running_agent_is_not_touched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun()
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)
        paths.plist.mkdir(parents=True)

        with pytest.raises(agent.AgentError, match=r"cannot use .*\.plist: it is not a regular file$"):
            agent.install(paths, tmp_path / "data", [], {})

        assert run.commands == []

    def test_a_plist_that_cannot_be_written_is_an_agent_error_and_the_running_agent_is_not_touched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun()
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)
        paths.plist.parent.mkdir(parents=True)
        paths.plist.write_bytes(b"previous")

        def refuse(source: object, destination: object) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(agent.os, "replace", refuse)

        with pytest.raises(agent.AgentError, match=r"cannot write .*\.plist: OSError$"):
            agent.install(paths, tmp_path / "data", [], {})

        assert run.subcommands() == ["print"]
        assert paths.plist.read_bytes() == b"previous"
        assert self.beside_the_lock(paths) == [paths.plist.name]

    @pytest.mark.parametrize("value", STRINGS_A_PROPERTY_LIST_DOES_NOT_CARRY_EXACTLY)
    def test_a_setting_a_property_list_cannot_carry_is_an_agent_error_raised_before_anything_is_touched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        run = FakeRun()
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)
        paths.plist.parent.mkdir(parents=True)
        paths.plist.write_bytes(b"previous")
        overrides = {"USAGEDASH_PORT": "9001", "USAGEDASH_UNUSED": value, "USAGEDASH_ALSO_UNUSED": value}

        with pytest.raises(agent.AgentError) as raised:
            agent.install(paths, tmp_path / "data", [], overrides)

        assert str(raised.value) == (
            "cannot build the agent's property list: "
            "'USAGEDASH_ALSO_UNUSED', 'USAGEDASH_UNUSED' cannot be written in one"
        )
        assert run.commands == []
        assert paths.plist.read_bytes() == b"previous"

    def test_a_variable_name_a_property_list_cannot_carry_is_shown_as_a_literal(self, tmp_path: Path) -> None:
        with pytest.raises(agent.AgentError, match=r"property list: 'USAGEDASH_\\x01' cannot be written in one$"):
            agent.build_plist(self.paths(tmp_path), "/venv/bin/python", tmp_path, "/usr/bin", {"USAGEDASH_\x01": "1"})

    @pytest.mark.parametrize("unwritable", STRINGS_A_PROPERTY_LIST_DOES_NOT_CARRY_EXACTLY)
    def test_a_path_a_property_list_cannot_carry_is_an_agent_error(self, tmp_path: Path, unwritable: str) -> None:
        with pytest.raises(agent.AgentError, match=r"property list: a path cannot be written in one$"):
            agent.build_plist(self.paths(tmp_path), "/venv/bin/python", Path(f"/data{unwritable}dir"), "/usr/bin", {})

    def test_every_value_in_a_property_list_that_is_built_is_read_back_exactly_as_given(self, tmp_path: Path) -> None:
        overrides = {"USAGEDASH_CONFIG": "/Users/persona/caf\u00e9 \U0001f4ca/two\nlines\tand a tab/config.toml"}
        working_directory = tmp_path / 'data with spaces & <angle> "quotes"'

        plist = plistlib.loads(
            agent.build_plist(self.paths(tmp_path), "/venv/bin/python", working_directory, "/usr/bin", overrides)
        )

        assert plist["EnvironmentVariables"] == {**overrides, "PATH": "/usr/bin"}
        assert plist["WorkingDirectory"] == str(working_directory)

    def test_install_makes_the_files_launchd_writes_to_owner_only_before_asking_launchd_for_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        launchctl_not_loaded(monkeypatch)
        paths = self.paths(tmp_path)
        paths.log_dir.mkdir(parents=True)
        standard_output, standard_error = paths.launchd_logs
        standard_error.write_text("earlier output\n")
        standard_error.chmod(0o644)

        agent.install(paths, tmp_path / "data", [], {})

        assert [(log.is_file(), log.stat().st_mode & 0o777) for log in (standard_output, standard_error)] == [
            (True, 0o600),
            (True, 0o600),
        ]
        assert standard_error.read_text() == "earlier output\n"
        plist = plistlib.loads(paths.plist.read_bytes())
        assert (plist["StandardOutPath"], plist["StandardErrorPath"]) == (str(standard_output), str(standard_error))

    @pytest.mark.parametrize("linked", [0, 1])
    def test_a_symbolic_link_where_launchd_would_write_is_refused_and_nothing_is_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked: int
    ) -> None:
        run = FakeRun()
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)
        paths.log_dir.mkdir(parents=True)
        target = tmp_path / "somewhere-else"
        target.write_text("not the dashboard's\n")
        paths.launchd_logs[linked].symlink_to(target)

        with pytest.raises(ConfigError, match=r"cannot open .*launchd-std(out|err)\.log as a private file"):
            agent.install(paths, tmp_path / "data", [], {})

        assert run.commands == []
        assert not paths.plist.exists()
        assert target.read_text() == "not the dashboard's\n"

    def reinstall(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run: FakeRun, previous: bytes | None
    ) -> tuple[agent.AgentPaths, str]:
        monkeypatch.setattr(agent.subprocess, "run", run)
        monkeypatch.setattr(agent.time, "sleep", lambda _seconds: None)
        paths = self.paths(tmp_path)
        if previous is not None:
            paths.plist.parent.mkdir(parents=True)
            paths.plist.write_bytes(previous)
        with pytest.raises(agent.AgentError) as raised:
            agent.install(paths, tmp_path / "data", [], {})
        return paths, str(raised.value)

    @pytest.mark.parametrize("planted", ["usagedash.py", "usagedash/__init__.py", "usagedash/__main__.py"])
    def test_the_agent_s_command_runs_nothing_left_in_its_working_directory(self, tmp_path: Path, planted: str) -> None:
        data_dir = tmp_path / "data"
        (data_dir / planted).parent.mkdir(parents=True, exist_ok=True)
        (data_dir / planted).write_text("raise SystemExit('ran what was left in the working directory')\n")
        plist = plistlib.loads(agent.build_plist(self.paths(tmp_path), sys.executable, data_dir, "/usr/bin", {}))
        *interpreter_and_module, served = plist["ProgramArguments"]

        result = subprocess.run(
            [*interpreter_and_module, "--help"],
            cwd=plist["WorkingDirectory"],
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(agent.__file__).parents[1])},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        assert served == "serve"
        assert (result.returncode, result.stderr) == (0, "")
        assert result.stdout.startswith("usage: usagedash")

    def interrupted_install(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run: FakeRun, previous: bytes | None
    ) -> tuple[agent.AgentPaths, str]:
        monkeypatch.setattr(agent.subprocess, "run", run)
        monkeypatch.setattr(agent.time, "sleep", lambda _seconds: None)
        paths = self.paths(tmp_path)
        if previous is not None:
            paths.plist.parent.mkdir(parents=True)
            paths.plist.write_bytes(previous)
        with pytest.raises(KeyboardInterrupt) as raised:
            agent.install(paths, tmp_path / "data", [], {})
        return paths, str(raised.value)

    def test_an_install_interrupted_after_the_old_agent_was_stopped_starts_the_old_agent_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = InterruptedAt("bootstrap")
        run.sequences = {"print": [0, NOT_LOADED, NOT_LOADED]}

        paths, message = self.interrupted_install(tmp_path, monkeypatch, run, b"previous")

        assert message == "the previous agent was started again"
        assert paths.plist.read_bytes() == b"previous"
        assert run.subcommands() == ["print", "bootout", "print", "bootstrap", "print", "bootstrap"]

    def test_an_install_interrupted_before_launchd_was_asked_for_anything_leaves_the_old_agent_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def interrupted(source: object, destination: object) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(agent.os, "replace", interrupted)
        run = FakeRun()
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)

        with pytest.raises(KeyboardInterrupt) as raised:
            agent.install(paths, tmp_path / "data", [], {})

        assert str(raised.value) == "the previous agent is still running"
        assert run.subcommands() == ["print"]
        assert self.beside_the_lock(paths) == []

    def test_an_install_interrupted_while_stopping_the_old_agent_starts_it_again_once_it_has_gone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = InterruptedAt("bootout")
        run.sequences = {"print": [0, 0, NOT_LOADED]}

        paths, message = self.interrupted_install(tmp_path, monkeypatch, run, b"previous")

        assert message == "the previous agent was started again"
        assert paths.plist.read_bytes() == b"previous"
        assert run.subcommands() == ["print", "bootout", "print", "bootout", "print", "bootstrap"]

    def test_an_install_interrupted_while_the_old_agent_is_still_going_away_waits_for_it_and_starts_it_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = InterruptedAt("print", occurrence=2)
        run.sequences = {"print": [0, 0, NOT_LOADED]}

        paths, message = self.interrupted_install(tmp_path, monkeypatch, run, b"previous")

        assert message == "the previous agent was started again"
        assert paths.plist.read_bytes() == b"previous"
        assert run.subcommands() == ["print", "bootout", "print", "print", "bootout", "print", "bootstrap"]

    def test_an_install_interrupted_after_launchd_registered_the_new_agent_removes_it_and_starts_the_old_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = InterruptedAt("bootstrap")
        run.sequences = {"print": [0, NOT_LOADED, 0, NOT_LOADED]}

        paths, message = self.interrupted_install(tmp_path, monkeypatch, run, b"previous")

        assert message == "the previous agent was started again"
        assert paths.plist.read_bytes() == b"previous"
        assert run.subcommands() == [
            "print",
            "bootout",
            "print",
            "bootstrap",
            "print",
            "bootout",
            "print",
            "bootstrap",
        ]

    def test_a_first_install_interrupted_after_launchd_registered_the_agent_removes_the_agent_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = InterruptedAt("bootstrap")
        run.sequences = {"print": [NOT_LOADED, 0, NOT_LOADED]}

        paths, message = self.interrupted_install(tmp_path, monkeypatch, run, None)

        assert message == "the property list on disk is as it was before"
        assert not paths.plist.exists()
        assert run.subcommands() == ["print", "bootstrap", "print", "bootout", "print"]

    def test_a_first_install_that_is_interrupted_leaves_no_plist_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = InterruptedAt("bootstrap")
        run.returncodes = {"print": NOT_LOADED}

        paths, message = self.interrupted_install(tmp_path, monkeypatch, run, None)

        assert message == "the property list on disk is as it was before"
        assert not paths.plist.exists()

    def test_an_install_interrupted_while_putting_back_a_failed_one_is_still_put_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        class RejectedThenInterrupted(FakeRun):
            interrupted = False

            def __call__(self, command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
                if "bootstrap" in self.subcommands() and command[1] == "print" and not self.interrupted:
                    self.interrupted = True
                    self.commands.append(command)
                    raise KeyboardInterrupt
                return super().__call__(command, **options)

        run = RejectedThenInterrupted(stderr="Bootstrap failed: 5: Input/output error")
        run.sequences = {"print": [0, NOT_LOADED, NOT_LOADED], "bootstrap": [5, 0]}

        paths, message = self.interrupted_install(tmp_path, monkeypatch, run, b"previous")

        assert message == "the previous agent was started again"
        assert paths.plist.read_bytes() == b"previous"
        assert run.subcommands() == ["print", "bootout", "print", "bootstrap", "print", "print", "bootstrap"]

    def test_a_reinstall_that_launchd_rejects_puts_the_previous_agent_back_and_starts_it_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stderr="Bootstrap failed: 5: Input/output error")
        run.sequences = {"print": [0, NOT_LOADED, NOT_LOADED], "bootstrap": [5, 0]}

        paths, message = self.reinstall(tmp_path, monkeypatch, run, b"previous")

        assert message == (
            "launchctl bootstrap failed (5): Bootstrap failed: 5: Input/output error; "
            "the previous agent was started again"
        )
        assert paths.plist.read_bytes() == b"previous"
        assert run.subcommands() == ["print", "bootout", "print", "bootstrap", "print", "bootstrap"]

    def test_a_reinstall_whose_previous_agent_cannot_be_started_again_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stderr="Bootstrap failed: 5: Input/output error")
        run.sequences = {"print": [0, NOT_LOADED, NOT_LOADED]}
        run.returncodes = {"bootstrap": 5}

        paths, message = self.reinstall(tmp_path, monkeypatch, run, b"previous")

        assert message == (
            "launchctl bootstrap failed (5): Bootstrap failed: 5: Input/output error; "
            "the previous installation could not be put back: "
            "launchctl bootstrap failed (5): Bootstrap failed: 5: Input/output error"
        )
        assert paths.plist.read_bytes() == b"previous"

    @pytest.mark.parametrize("stop_result", [0, 5])
    def test_a_reinstall_that_fails_with_the_previous_agent_still_loaded_puts_its_plist_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_result: int
    ) -> None:
        clock = iter([0.0, agent.UNLOAD_WAIT_SECONDS, 100.0, 100.0 + agent.UNLOAD_WAIT_SECONDS])
        monkeypatch.setattr(agent.time, "monotonic", lambda: next(clock))

        run = FakeRun(stderr="stop refused")
        run.returncodes = {"bootout": stop_result}
        paths, message = self.reinstall(tmp_path, monkeypatch, run, b"previous")

        failure = (
            "launchctl bootout failed (5): stop refused"
            if stop_result
            else "the previous agent did not stop within 45 seconds"
        )
        assert message == f"{failure}; the previous installation could not be put back: {failure}"
        if stop_result:
            assert next(clock) == 0.0
        assert paths.plist.read_bytes() == b"previous"
        assert "bootstrap" not in run.subcommands()

    @pytest.mark.parametrize("reported", [NOT_LOADED, 5])
    @pytest.mark.parametrize("installing", [True, False])
    def test_agent_changes_reconcile_the_state_after_a_nonzero_stop_without_waiting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reported: int, installing: bool
    ) -> None:
        run = FakeRun(stderr="stop refused")
        run.returncodes = {"bootout": 5}
        run.sequences = {"print": [0, reported, NOT_LOADED]}
        monkeypatch.setattr(agent.subprocess, "run", run)

        def no_wait() -> float:
            raise AssertionError("a failed stop must be reconciled before polling")

        monkeypatch.setattr(agent.time, "monotonic", no_wait)
        paths = self.paths(tmp_path)
        self.write_private_plist(paths, b"previous")

        def change() -> None:
            if installing:
                agent.install(paths, tmp_path / "data", [], {})
            else:
                agent.uninstall(paths)

        if reported == NOT_LOADED:
            change()
            assert paths.plist.exists() == installing
            assert run.subcommands() == ["print", "bootout", "print"] + (["bootstrap"] if installing else [])
        else:
            with pytest.raises(agent.AgentError, match=r"launchctl print failed.*agent was started again"):
                change()
            assert paths.plist.read_bytes() == b"previous"
            assert run.subcommands() == ["print", "bootout", "print", "print", "bootstrap"]

    def test_a_previous_agent_that_stops_only_after_the_wait_gave_up_is_started_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = iter([0.0, agent.UNLOAD_WAIT_SECONDS, 100.0])
        monkeypatch.setattr(agent.time, "monotonic", lambda: next(clock))
        run = FakeRun()
        run.sequences = {"print": [0, 0, 0, NOT_LOADED]}

        paths, message = self.reinstall(tmp_path, monkeypatch, run, b"previous")

        assert message == "the previous agent did not stop within 45 seconds; the previous agent was started again"
        assert paths.plist.read_bytes() == b"previous"
        assert run.subcommands() == ["print", "bootout", "print", "print", "bootout", "print", "bootstrap"]

    def test_an_install_over_a_plist_whose_agent_was_not_loaded_puts_the_plist_back_when_launchd_rejects_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stderr="Bootstrap failed: 5: Input/output error")
        run.returncodes = {"print": NOT_LOADED, "bootstrap": 5}

        paths, message = self.reinstall(tmp_path, monkeypatch, run, b"previous")

        assert message.endswith("; the property list on disk is as it was before")
        assert paths.plist.read_bytes() == b"previous"
        assert run.subcommands() == ["print", "bootstrap", "print"]

    def test_a_loaded_agent_with_no_plist_on_disk_cannot_be_started_again_and_the_failure_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stderr="Bootstrap failed: 5: Input/output error")
        run.sequences = {"print": [0, NOT_LOADED, NOT_LOADED]}
        run.returncodes = {"bootstrap": 5}

        paths, message = self.reinstall(tmp_path, monkeypatch, run, None)

        assert message.endswith("; the previous agent had no property list on disk to start it from again")
        assert not paths.plist.exists()
        assert run.subcommands() == ["print", "bootout", "print", "bootstrap", "print"]

    def test_a_failed_first_install_whose_plist_cannot_be_removed_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class LocksTheDirectoryWhenAskedToBootstrap(FakeRun):
            def __call__(self, command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
                if command[1] == "bootstrap":
                    Path(command[3]).parent.chmod(0o500)
                return super().__call__(command, **options)

        run = LocksTheDirectoryWhenAskedToBootstrap(stderr="Bootstrap failed: 5: Input/output error")
        run.returncodes = {"print": NOT_LOADED, "bootstrap": 5}
        try:
            paths, message = self.reinstall(tmp_path, monkeypatch, run, None)
        finally:
            (tmp_path / "Library" / "LaunchAgents").chmod(0o700)

        assert "; the previous installation could not be put back: cannot delete " in message
        assert message.endswith(".plist: PermissionError")
        assert paths.plist.exists()

    @pytest.mark.parametrize(
        ("failure", "name"),
        [
            (FileNotFoundError("launchctl"), "FileNotFoundError"),
            (subprocess.TimeoutExpired(["/bin/launchctl", "print"], 30), "TimeoutExpired"),
        ],
    )
    def test_a_launchctl_that_cannot_be_run_or_does_not_finish_is_an_agent_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception, name: str
    ) -> None:
        monkeypatch.setattr(agent.subprocess, "run", FakeRun(error=failure))

        with pytest.raises(agent.AgentError, match=f"launchctl print could not be run: {name}"):
            agent.is_loaded(self.paths(tmp_path))

    @pytest.mark.parametrize("status", [1, 5, 112, 126], ids=lambda status: f"exit status {status}")
    def test_only_launchd_s_no_such_service_answer_means_the_agent_is_not_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        run = FakeRun(stderr="Could not find domain for user gui: 501\n")
        run.returncodes = {"print": status}
        monkeypatch.setattr(agent.subprocess, "run", run)

        with pytest.raises(
            agent.AgentError, match=rf"^launchctl print failed \({status}\): Could not find domain for user gui: 501$"
        ):
            agent.is_loaded(self.paths(tmp_path))

    def test_the_agent_is_loaded_when_launchd_lists_it_and_not_when_launchd_finds_no_such_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        launchctl_stopping_the_agent(monkeypatch)

        assert [agent.is_loaded(self.paths(tmp_path)), agent.is_loaded(self.paths(tmp_path))] == [True, False]

    def test_an_install_that_cannot_learn_whether_an_agent_is_loaded_changes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stderr="Input/output error")
        run.sequences = {"print": [5]}
        run.returncodes = {"bootstrap": 5}

        paths, message = self.reinstall(tmp_path, monkeypatch, run, b"previous")

        assert message == "launchctl print failed (5): Input/output error"
        assert run.subcommands() == ["print"]
        assert paths.plist.read_bytes() == b"previous"
        assert self.beside_the_lock(paths) == [paths.plist.name]

    def test_an_uninstall_that_cannot_learn_whether_the_agent_is_loaded_keeps_its_property_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stderr="Input/output error")
        run.returncodes = {"print": 5}
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)
        paths.plist.parent.mkdir(parents=True)
        paths.plist.write_bytes(b"installed")

        with pytest.raises(agent.AgentError, match=r"^launchctl print failed \(5\)"):
            agent.uninstall(paths)

        assert run.subcommands() == ["print"]
        assert paths.plist.read_bytes() == b"installed"

    def test_putting_back_an_install_says_so_when_it_cannot_learn_what_launchd_is_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stderr="Input/output error")
        run.sequences = {"print": [0, NOT_LOADED, 5]}
        run.returncodes = {"bootstrap": 5}

        paths, message = self.reinstall(tmp_path, monkeypatch, run, b"previous")

        assert message == (
            "launchctl bootstrap failed (5): Input/output error; the previous installation could not be put back: "
            "launchctl print failed (5): Input/output error"
        )
        assert run.subcommands() == ["print", "bootout", "print", "bootstrap", "print"]
        assert paths.plist.read_bytes() == b"previous"

    def test_a_plist_that_cannot_be_deleted_is_an_agent_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        launchctl_not_loaded(monkeypatch)
        paths = self.paths(tmp_path)
        paths.plist.mkdir(parents=True)

        with pytest.raises(agent.AgentError, match=r"cannot delete .*\.plist: "):
            agent.uninstall(paths)

    def test_install_does_not_register_the_agent_while_the_previous_one_is_still_stopping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun()
        waits = [0.0, agent.UNLOAD_WAIT_SECONDS - 0.1, agent.UNLOAD_WAIT_SECONDS]
        clock = iter([*waits, *(100.0 + moment for moment in waits)])
        monkeypatch.setattr(agent.subprocess, "run", run)
        monkeypatch.setattr(agent.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(agent.time, "monotonic", lambda: next(clock))

        paths = self.paths(tmp_path)

        with pytest.raises(agent.AgentError, match="did not stop within 45 seconds; the previous installation could"):
            agent.install(paths, tmp_path / "data", [], {})

        assert "bootstrap" not in run.subcommands()
        assert not paths.plist.exists()

    def test_a_first_install_registers_without_stopping_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun()
        run.sequences = {"print": [NOT_LOADED]}
        monkeypatch.setattr(agent.subprocess, "run", run)

        agent.install(self.paths(tmp_path), tmp_path / "data", [], {})

        assert run.subcommands() == ["print", "bootstrap"]

    def test_install_reports_a_bootstrap_that_launchd_rejects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stderr="Bootstrap failed: 5: Input/output error")
        run.returncodes = {"print": 113, "bootstrap": 5}
        monkeypatch.setattr(agent.subprocess, "run", run)

        paths = self.paths(tmp_path)

        with pytest.raises(agent.AgentError, match=r"Bootstrap failed: 5.*; the property list on disk is as it was"):
            agent.install(paths, tmp_path / "data", [], {})

        assert [command[1] for command in run.commands] == ["print", "bootstrap", "print"]
        assert not paths.plist.exists()

    def test_uninstall_stops_the_agent_removes_the_plist_and_keeps_the_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun()
        run.sequences = {"print": [NOT_LOADED, 0, NOT_LOADED]}
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)
        data = tmp_path / "data"
        agent.install(paths, data, [], {"USAGEDASH_DATA_DIR": str(data)})
        (data / "usage.db").write_text("kept")

        removed = agent.uninstall(paths)

        assert removed is not None
        assert removed["USAGEDASH_DATA_DIR"] == str(data)
        assert not paths.plist.exists()
        assert (data / "usage.db").read_text() == "kept"
        assert "bootout" in run.subcommands()
        assert run.commands[-1][1] == "print"

    def test_uninstall_of_an_agent_that_is_not_loaded_only_removes_the_plist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(returncode=113)
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)
        paths.plist.parent.mkdir(parents=True)
        paths.plist.write_bytes(b"stale")

        assert agent.uninstall(paths) is None

        assert not paths.plist.exists()
        assert [command[1] for command in run.commands] == ["print"]

    def test_restart_reports_a_refusal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        run = FakeRun(stderr="operation not permitted")
        run.returncodes = {"kickstart": 3}
        monkeypatch.setattr(agent.subprocess, "run", run)

        with pytest.raises(agent.AgentError, match=r"^launchctl kickstart failed \(3\): operation not permitted$"):
            agent.restart_if_loaded(self.paths(tmp_path))

    def test_restart_asks_launchd_to_stop_and_start_the_service_it_is_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun()
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)

        assert agent.restart_if_loaded(paths) is True

        assert run.commands == [
            ["/bin/launchctl", "print", paths.service_target],
            ["/bin/launchctl", "kickstart", "-k", paths.service_target],
        ]

    def test_restart_does_nothing_and_says_so_when_launchd_is_not_running_the_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = launchctl_not_loaded(monkeypatch)

        assert agent.restart_if_loaded(self.paths(tmp_path)) is False

        assert run.subcommands() == ["print"]

    def test_uninstall_keeps_the_plist_when_launchd_refuses_to_stop_the_agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stderr="operation not permitted")
        run.returncodes = {"bootout": 5}
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)
        paths.plist.parent.mkdir(parents=True)
        paths.plist.write_bytes(b"installed")

        with pytest.raises(agent.AgentError, match=r"launchctl bootout failed \(5\): operation not permitted"):
            agent.uninstall(paths)

        assert paths.plist.read_bytes() == b"installed"
        assert "bootstrap" not in run.subcommands()

    def write_private_plist(self, paths: agent.AgentPaths, payload: bytes) -> None:
        paths.plist.parent.mkdir(parents=True, exist_ok=True)
        descriptor = agent.open_private_file(paths.plist)
        try:
            os.ftruncate(descriptor, 0)
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)

    def test_installed_environment_is_absent_when_nothing_is_installed(self, tmp_path: Path) -> None:
        assert agent.installed_environment(self.paths(tmp_path)) is None

    def test_installed_environment_keeps_only_the_string_settings_the_agent_will_run_with(self, tmp_path: Path) -> None:
        paths = self.paths(tmp_path)
        self.write_private_plist(
            paths,
            plistlib.dumps({"EnvironmentVariables": {"USAGEDASH_PORT": "9001", "COUNT": 3, "PATH": "/usr/bin"}}),
        )

        assert agent.installed_environment(paths) == {"USAGEDASH_PORT": "9001", "PATH": "/usr/bin"}

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param(plistlib.dumps(["not-a-dict"]), id="not a dictionary"),
            pytest.param(plistlib.dumps({"Label": "x"}), id="no environment"),
            pytest.param(plistlib.dumps({"EnvironmentVariables": "nope"}), id="environment is not a dictionary"),
        ],
    )
    def test_installed_environment_is_empty_when_the_plist_names_no_string_environment(
        self, tmp_path: Path, payload: bytes
    ) -> None:
        paths = self.paths(tmp_path)
        self.write_private_plist(paths, payload)

        assert agent.installed_environment(paths) == {}

    def test_installed_environment_refuses_a_plist_that_is_not_a_property_list(self, tmp_path: Path) -> None:
        paths = self.paths(tmp_path)
        self.write_private_plist(paths, b"not a plist")

        with pytest.raises(agent.AgentError, match=rf"^cannot read {paths.plist}: "):
            agent.installed_environment(paths)

    @pytest.mark.parametrize(("payload", "failure"), PLISTS_THE_PARSER_CANNOT_FINISH)
    def test_installed_environment_refuses_a_plist_the_parser_cannot_finish(
        self, tmp_path: Path, payload: bytes, failure: str
    ) -> None:
        paths = self.paths(tmp_path)
        self.write_private_plist(paths, payload)

        with pytest.raises(agent.AgentError) as raised:
            agent.installed_environment(paths)

        assert str(raised.value) == f"cannot read {paths.plist}: {failure}"

    def test_uninstall_does_not_delete_the_plist_when_the_agent_does_not_finish_stopping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun()
        clock = iter([0.0, agent.UNLOAD_WAIT_SECONDS])
        monkeypatch.setattr(agent.subprocess, "run", run)
        monkeypatch.setattr(agent.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(agent.time, "monotonic", lambda: next(clock))
        paths = self.paths(tmp_path)
        paths.plist.parent.mkdir(parents=True)
        paths.plist.write_bytes(b"installed")

        with pytest.raises(agent.AgentError, match="did not stop within 45 seconds"):
            agent.uninstall(paths)

        assert paths.plist.read_bytes() == b"installed"
        assert run.subcommands() == ["print", "bootout", "print"]

    def test_uninstall_starts_the_agent_again_from_the_trusted_plist_when_it_cannot_be_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = launchctl_stopping_the_agent(monkeypatch)
        paths = self.paths(tmp_path)
        payload = b"installed-and-trusted"
        self.write_private_plist(paths, payload)

        def fail_delete(target: agent.AgentPaths) -> None:
            raise agent.AgentError(f"cannot delete {target.plist}: PermissionError")

        monkeypatch.setattr(agent, "_delete_plist", fail_delete)

        with pytest.raises(agent.AgentError, match=r"cannot delete .*\.plist: .*; the agent was started again$"):
            agent.uninstall(paths)

        assert paths.plist.read_bytes() == payload
        assert run.subcommands() == ["print", "bootout", "print", "bootstrap"]

    def test_uninstall_does_not_start_an_untrusted_plist_that_cannot_be_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = launchctl_stopping_the_agent(monkeypatch)
        paths = self.paths(tmp_path)
        paths.plist.parent.mkdir(parents=True)
        paths.plist.write_bytes(b"installed")
        paths.plist.chmod(0o666)

        def fail_delete(target: agent.AgentPaths) -> None:
            raise agent.AgentError(f"cannot delete {target.plist}: PermissionError")

        monkeypatch.setattr(agent, "_delete_plist", fail_delete)

        with pytest.raises(agent.AgentError, match=r"the agent was not started again$"):
            agent.uninstall(paths)

        assert paths.plist.read_bytes() == b"installed"
        assert "bootstrap" not in run.subcommands()

    def test_uninstall_says_when_the_agent_cannot_be_started_again_after_the_plist_stays(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stderr="Bootstrap failed: 5")
        run.sequences = {"print": [0, NOT_LOADED]}
        run.returncodes = {"bootstrap": 5}
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)
        self.write_private_plist(paths, b"installed-and-trusted")

        def fail_delete(target: agent.AgentPaths) -> None:
            raise agent.AgentError(f"cannot delete {target.plist}: PermissionError")

        monkeypatch.setattr(agent, "_delete_plist", fail_delete)

        with pytest.raises(agent.AgentError, match=r"could not be started again: launchctl bootstrap failed \(5\)"):
            agent.uninstall(paths)

        assert paths.plist.is_file()
        assert run.subcommands()[-1] == "bootstrap"

    def test_uninstall_does_not_start_the_agent_when_the_trusted_plist_cannot_be_written_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = launchctl_stopping_the_agent(monkeypatch)
        paths = self.paths(tmp_path)
        payload = b"installed-and-trusted"
        self.write_private_plist(paths, payload)

        def fail_delete(target: agent.AgentPaths) -> None:
            raise agent.AgentError(f"cannot delete {target.plist}: PermissionError")

        def fail_write(target: agent.AgentPaths, _payload: bytes) -> None:
            raise agent.AgentError(f"cannot write {target.plist}: OSError")

        monkeypatch.setattr(agent, "_delete_plist", fail_delete)
        monkeypatch.setattr(agent, "_write_plist", fail_write)

        with pytest.raises(agent.AgentError, match=r"could not be started again: cannot write .*\.plist: OSError$"):
            agent.uninstall(paths)

        assert paths.plist.read_bytes() == payload
        assert "bootstrap" not in run.subcommands()

    def test_an_uninstall_interrupted_after_the_stop_was_asked_starts_the_trusted_agent_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = InterruptedAt("bootout")
        run.sequences = {"print": [0, NOT_LOADED]}
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)
        payload = b"installed-and-trusted"
        self.write_private_plist(paths, payload)

        with pytest.raises(KeyboardInterrupt, match="the agent was started again"):
            agent.uninstall(paths)

        assert paths.plist.read_bytes() == payload
        assert run.subcommands() == ["print", "bootout", "print", "bootstrap"]

    @pytest.mark.parametrize("failing", ["bootout", "the unload poll"])
    def test_an_uninstall_whose_stop_fails_with_its_outcome_unknown_starts_the_trusted_agent_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str
    ) -> None:
        run = launchctl_stopping_the_agent(monkeypatch)
        paths = self.paths(tmp_path)
        payload = b"installed-and-trusted"
        self.write_private_plist(paths, payload)
        if failing == "bootout":
            real_launchctl = agent._launchctl

            def acting_then_timing_out(*arguments: str) -> subprocess.CompletedProcess[str]:
                result = real_launchctl(*arguments)
                if arguments[0] == "bootout":
                    raise agent.AgentError("launchctl bootout timed out")
                return result

            monkeypatch.setattr(agent, "_launchctl", acting_then_timing_out)
        else:

            def failing_poll(_paths: agent.AgentPaths) -> None:
                raise agent.AgentError("launchctl print failed (1): busy")

            monkeypatch.setattr(agent, "_wait_until_unloaded", failing_poll)

        with pytest.raises(agent.AgentError, match=r"(timed out|busy); the agent was started again"):
            agent.uninstall(paths)

        assert paths.plist.read_bytes() == payload
        assert run.subcommands() == ["print", "bootout", "print", "bootstrap"]

    def test_an_uninstall_interrupted_after_the_stop_was_asked_does_not_start_an_untrusted_plist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = InterruptedAt("bootout")
        run.sequences = {"print": [0, NOT_LOADED]}
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)
        paths.plist.parent.mkdir(parents=True)
        paths.plist.write_bytes(b"installed")
        paths.plist.chmod(0o666)

        with pytest.raises(KeyboardInterrupt, match="the agent was not started again"):
            agent.uninstall(paths)

        assert paths.plist.read_bytes() == b"installed"
        assert "bootstrap" not in run.subcommands()

    def test_an_uninstall_interrupted_before_launchd_was_asked_leaves_the_agent_as_it_was(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = InterruptedAt("print")
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)
        payload = b"installed-and-trusted"
        self.write_private_plist(paths, payload)

        with pytest.raises(KeyboardInterrupt) as caught:
            agent.uninstall(paths)

        assert caught.value.args == ()
        assert paths.plist.read_bytes() == payload
        assert run.subcommands() == ["print"]

    def test_an_interrupted_uninstall_without_a_plist_still_stops_the_loaded_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = InterruptedAt("bootout")
        run.sequences = {"print": [0, 0, NOT_LOADED]}
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)

        with pytest.raises(KeyboardInterrupt, match="the agent was removed"):
            agent.uninstall(paths)

        assert not paths.plist.exists()
        assert run.subcommands() == ["print", "bootout", "print", "bootout", "print"]

    def test_an_uninstall_interrupted_after_the_plist_is_gone_reports_that_it_was_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = launchctl_stopping_the_agent(monkeypatch)
        paths = self.paths(tmp_path)
        self.write_private_plist(paths, b"installed-and-trusted")

        def remove_then_interrupt(target: agent.AgentPaths) -> None:
            target.plist.unlink()
            raise KeyboardInterrupt

        monkeypatch.setattr(agent, "_wait_until_unloaded", remove_then_interrupt)

        with pytest.raises(KeyboardInterrupt, match="the agent was removed"):
            agent.uninstall(paths)

        assert not paths.plist.exists()
        assert "bootstrap" not in run.subcommands()

    def test_an_uninstall_interrupted_while_launchd_still_lists_the_agent_stops_it_and_starts_it_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun()
        run.sequences = {"print": [0, 0, NOT_LOADED]}
        monkeypatch.setattr(agent.subprocess, "run", run)
        paths = self.paths(tmp_path)
        payload = b"installed-and-trusted"
        self.write_private_plist(paths, payload)
        waits = {"count": 0}
        real_wait = agent._wait_until_unloaded

        def interrupt_the_first_wait(target: agent.AgentPaths) -> None:
            waits["count"] += 1
            if waits["count"] == 1:
                raise KeyboardInterrupt
            real_wait(target)

        monkeypatch.setattr(agent, "_wait_until_unloaded", interrupt_the_first_wait)

        with pytest.raises(KeyboardInterrupt, match="the agent was started again"):
            agent.uninstall(paths)

        assert paths.plist.read_bytes() == payload
        assert run.subcommands() == ["print", "bootout", "print", "bootout", "print", "bootstrap"]

    def test_an_uninstall_interrupted_after_the_stop_reports_a_restore_that_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = launchctl_stopping_the_agent(monkeypatch)
        paths = self.paths(tmp_path)
        self.write_private_plist(paths, b"installed-and-trusted")

        def interrupt(_target: agent.AgentPaths) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(agent, "_wait_until_unloaded", interrupt)

        def fail_write(_target: agent.AgentPaths, _payload: bytes) -> None:
            raise agent.AgentError("cannot write the property list: OSError")

        monkeypatch.setattr(agent, "_write_plist", fail_write)

        with pytest.raises(KeyboardInterrupt, match="could not be put back: cannot write the property list: OSError"):
            agent.uninstall(paths)

        assert "bootstrap" not in run.subcommands()

    def test_uninstall_does_not_start_the_agent_when_the_plist_is_already_gone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = launchctl_stopping_the_agent(monkeypatch)

        def remove_then_fail(target: agent.AgentPaths) -> None:
            target.plist.unlink()
            raise agent.AgentError("cannot delete the property list: OSError")

        monkeypatch.setattr(agent, "_delete_plist", remove_then_fail)
        paths = self.paths(tmp_path)
        paths.plist.parent.mkdir(parents=True)
        paths.plist.write_bytes(b"installed")

        with pytest.raises(agent.AgentError, match=r"^cannot delete the property list: OSError$"):
            agent.uninstall(paths)

        assert "bootstrap" not in run.subcommands()
        assert not paths.plist.exists()

    def test_a_missing_path_is_absent(self, tmp_path: Path) -> None:
        assert agent._path_exists(tmp_path / "no-such-plist") is False

    def test_an_existence_check_that_cannot_examine_its_path_is_an_agent_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "plist"
        target.write_bytes(b"installed")
        real_lstat = Path.lstat

        def lstat(self: Path) -> os.stat_result:
            if self == target:
                raise PermissionError(errno.EACCES, "Permission denied")
            return real_lstat(self)

        monkeypatch.setattr(Path, "lstat", lstat)

        with pytest.raises(agent.AgentError) as raised:
            agent._path_exists(target)

        assert str(raised.value) == f"cannot examine {target}: PermissionError"

    def test_preparing_the_plist_directory_reports_an_examination_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths = self.paths(tmp_path)
        directory = paths.plist.parent
        directory.mkdir(parents=True)
        real_lstat = Path.lstat

        def lstat(self: Path) -> os.stat_result:
            if self == directory:
                raise OSError(errno.EIO, "Input/output error")
            return real_lstat(self)

        monkeypatch.setattr(Path, "lstat", lstat)

        with pytest.raises(agent.AgentError) as raised:
            agent._prepare_plist_directory(directory)

        assert str(raised.value) == f"cannot examine {directory}: OSError"
        assert directory.is_dir()

    def test_uninstall_reports_an_examination_failure_when_a_stopped_plist_cannot_be_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        launchctl_stopping_the_agent(monkeypatch)
        paths = self.paths(tmp_path)
        self.write_private_plist(paths, b"installed-and-trusted")

        def fail_delete(target: agent.AgentPaths) -> None:
            raise agent.AgentError(f"cannot delete {target.plist}: PermissionError")

        monkeypatch.setattr(agent, "_delete_plist", fail_delete)
        real_lstat = Path.lstat

        def lstat(self: Path) -> os.stat_result:
            if self == paths.plist:
                raise PermissionError(errno.EACCES, "Permission denied")
            return real_lstat(self)

        monkeypatch.setattr(Path, "lstat", lstat)

        with pytest.raises(agent.AgentError) as raised:
            agent.uninstall(paths)

        assert str(raised.value) == (
            f"cannot delete {paths.plist}: PermissionError; cannot examine {paths.plist}: PermissionError; "
            "the agent was not started again"
        )

    def test_uninstall_does_not_examine_the_plist_when_the_agent_was_not_stopped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        launchctl_not_loaded(monkeypatch)
        paths = self.paths(tmp_path)
        self.write_private_plist(paths, b"installed-and-trusted")
        examined: list[Path] = []
        real_lstat = Path.lstat

        def fail_delete(target: agent.AgentPaths) -> None:
            raise agent.AgentError(f"cannot delete {target.plist}: PermissionError")

        def lstat(self: Path) -> os.stat_result:
            if self == paths.plist:
                examined.append(self)
                raise PermissionError(errno.EACCES, "Permission denied")
            return real_lstat(self)

        monkeypatch.setattr(agent, "_delete_plist", fail_delete)
        monkeypatch.setattr(Path, "lstat", lstat)

        with pytest.raises(agent.AgentError) as raised:
            agent.uninstall(paths)

        assert str(raised.value) == f"cannot delete {paths.plist}: PermissionError"
        assert examined == []


class TestStandingLogins:
    @pytest.fixture
    def settings(self, tmp_path: Path) -> Settings:
        return load_settings(env={"USAGEDASH_DATA_DIR": str(tmp_path / "data")}, home=tmp_path)

    def reload(self, settings: Settings, tmp_path: Path) -> Settings:
        return load_settings(env={"USAGEDASH_DATA_DIR": str(settings.data_dir)}, home=tmp_path)

    def made_claude_home(self, settings: Settings) -> Path:
        home = standing.standing_home(settings, Provider.CLAUDE, "work")
        home.parent.mkdir(parents=True)
        home.mkdir()
        return home

    def list_claude(self, settings: Settings, home: Path, account_file: Path) -> None:
        settings.config_path.parent.mkdir(parents=True, exist_ok=True)
        settings.config_path.write_text(
            f"\n[[claude]]\nconfig_dir = {json.dumps(str(home))}\n"
            f'account_file = {json.dumps(str(account_file))}\nmode = "cli"\n'
        )

    def loading(self, settings: Settings, tmp_path: Path) -> standing.SettingsLoader:

        def load(config_file: Path) -> Settings:
            env = {"USAGEDASH_DATA_DIR": str(settings.data_dir), "USAGEDASH_CONFIG": str(config_file)}
            return load_settings(env=env, home=tmp_path, file_may_be_absent=True)

        return load

    def register(self, settings: Settings, tmp_path: Path, provider: Provider, name: str = "work") -> bool:
        home = standing.standing_home(settings, provider, name)
        return standing.register(settings, provider, home, self.loading(settings, tmp_path))

    def files_beside_the_config(self, settings: Settings) -> list[str]:
        return sorted(entry.name for entry in settings.config_path.parent.iterdir())

    @pytest.mark.parametrize("name", ["", "Work", "has space", "-leading", "a" * 33, "../escape", "semi;colon"])
    def test_rejects_names_that_are_not_short_lowercase_slugs(self, settings: Settings, name: str) -> None:
        with pytest.raises(ConfigError, match="login name must be"):
            standing.standing_home(settings, Provider.CLAUDE, name)

    def test_a_standing_home_lives_under_the_private_data_directory(self, settings: Settings) -> None:
        home = standing.standing_home(settings, Provider.CODEX, "personal-2")
        assert home == settings.data_dir / "homes" / "codex-personal-2"

    def test_registering_creates_a_private_home_and_adds_a_cli_mode_login_to_the_config(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        for provider in Provider:
            assert self.register(settings, tmp_path, provider)
            assert standing.standing_home(settings, provider, "work").stat().st_mode & 0o777 == 0o700

        reloaded = self.reload(settings, tmp_path)

        assert settings.config_path.stat().st_mode & 0o777 == 0o600
        assert [(s.config_dir.name, s.mode) for s in reloaded.claude_sources[1:]] == [("claude-work", SourceMode.CLI)]
        assert [(s.home.name, s.mode) for s in reloaded.codex_sources[1:]] == [("codex-work", SourceMode.CLI)]
        assert [(s.home.name, s.mode) for s in reloaded.grok_sources[1:]] == [("grok-work", SourceMode.CLI)]
        assert all(standing.is_registered(reloaded, p, standing.standing_home(settings, p, "work")) for p in Provider)

    def test_a_standing_home_that_already_existed_open_to_others_is_made_private(self, settings: Settings) -> None:
        home = standing.standing_home(settings, Provider.CODEX, "work")
        home.mkdir(parents=True)
        (home / "auth.json").write_text("{}")
        home.chmod(0o755)

        standing.prepare_home(settings, home)

        assert home.stat().st_mode & 0o777 == 0o700

    @pytest.mark.parametrize(
        "link", ["debug/latest", "debug/missing", "apply_patch", "applypatch", "codex-execve-wrapper"]
    )
    def test_reauthentication_preserves_checked_vendor_output_links(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        link: str,
    ) -> None:
        provider = Provider.CLAUDE if link.startswith("debug/") else Provider.CODEX
        home = standing.standing_home(settings, provider, "work")
        calls: list[Path] = []
        monkeypatch.setattr(standing, "run_login", lambda settings, provider, home: calls.append(home) or 0)
        load = self.loading(settings, tmp_path)
        assert standing.sign_in_and_register(settings, provider, home, load) == 0
        before = settings.config_path.read_bytes()
        if provider is Provider.CLAUDE:
            entry = home / "debug" / "latest"
            entry.parent.mkdir()
            target = entry.parent / "synthetic-session.txt"
            if link == "debug/latest":
                target.write_text("synthetic output")
            entry.symlink_to(target.name)
        else:
            entry = home / "tmp" / "arg0" / "synthetic-run" / link
            entry.parent.mkdir(parents=True)
            target = tmp_path / "codex"
            target.write_text("#!/bin/sh\nexit 0\n")
            target.chmod(0o111)
            entry.symlink_to(target)
        retained = os.readlink(entry)
        assert standing.sign_in_and_register(settings, provider, home, load) == 0
        assert calls == [home, home]
        assert settings.config_path.read_bytes() == before
        assert entry.is_symlink() and os.readlink(entry) == retained

    @pytest.mark.parametrize(
        "kind", ["escape", "writable", "directory", "foreign", "inspection", "helper-writable", "helper-missing"]
    )
    def test_unsafe_vendor_output_links_block_sign_in_without_changing_config(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        kind: str,
    ) -> None:
        helper = kind.startswith("helper-")
        provider = Provider.CODEX if helper else Provider.CLAUDE
        home = standing.standing_home(settings, provider, "work")
        load = self.loading(settings, tmp_path)
        standing.register(settings, provider, home, load)
        before = settings.config_path.read_bytes()
        entry = home / ("tmp/arg0/synthetic-run/apply_patch" if helper else "debug/latest")
        entry.parent.mkdir(parents=True)
        target = (tmp_path if helper or kind == "escape" else entry.parent) / "target"
        if kind == "directory":
            target.mkdir()
        elif kind != "helper-missing":
            target.touch(mode=0o755)
        if kind in {"writable", "helper-writable"}:
            target.chmod(0o777)
        entry.symlink_to(target)
        if kind == "foreign":
            original = os.lstat

            def foreign(path: str | os.PathLike[str], *, dir_fd: int | None = None) -> os.stat_result:
                status = original(path, dir_fd=dir_fd)
                if Path(path) == entry:
                    values = list(status)
                    values[4] = os.geteuid() + 1
                    return os.stat_result(values)
                return status

            monkeypatch.setattr(os, "lstat", foreign)
        if kind == "inspection":

            def unreadable(path: Path) -> Path:
                raise OSError(errno.EIO, "synthetic inspection failure")

            monkeypatch.setattr(standing, "checked_link_target", unreadable)
        called: list[Path] = []
        monkeypatch.setattr(standing, "run_login", lambda settings, provider, home: called.append(home) or 0)
        if kind == "writable":
            with pytest.raises(ConfigError, match="permissions"):
                standing._checked_vendor_output_link(home, entry)
        with pytest.raises(ConfigError, match="cannot"):
            standing.sign_in_and_register(settings, provider, home, load)
        assert called == []
        assert settings.config_path.read_bytes() == before
        assert entry.is_symlink()

    @pytest.mark.parametrize(
        ("planted", "refused_for"),
        [
            ("a file others can write", r"settings\.json: its permissions \(0666\) let other accounts write"),
            ("a symbolic link", r"settings\.json: it is a symbolic link"),
            ("a directory others can write", r"hooks: its permissions \(0777\) let other accounts write"),
        ],
    )
    def test_an_entry_in_an_existing_standing_home_that_another_account_could_change_is_refused(
        self, settings: Settings, tmp_path: Path, planted: str, refused_for: str
    ) -> None:
        home = standing.standing_home(settings, Provider.CLAUDE, "work")
        home.mkdir(parents=True)
        if planted == "a file others can write":
            (home / "settings.json").write_text("{}")
            (home / "settings.json").chmod(0o666)
        elif planted == "a symbolic link":
            (home / "settings.json").symlink_to(tmp_path / "elsewhere.json")
        else:
            (home / "hooks").mkdir()
            (home / "hooks").chmod(0o777)
        mode = home.stat().st_mode

        with pytest.raises(ConfigError, match=refused_for + ".*cannot be trusted, so remove it"):
            standing.prepare_home(settings, home)

        assert home.stat().st_mode == mode

    @pytest.mark.parametrize(
        "failure", ["the entry cannot be opened", "the home cannot be listed", "its list cannot be read"]
    )
    def test_an_existing_standing_home_whose_contents_cannot_be_examined_is_a_config_error(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch, failure: str
    ) -> None:
        home = standing.standing_home(settings, Provider.CODEX, "work")
        home.mkdir(parents=True)
        entry = home / "auth.json"
        entry.write_text("{}")
        if failure == "the entry cannot be opened":
            entry.chmod(0)
        elif failure == "the home cannot be listed":
            real_scandir = os.scandir

            def scandir(path: Any = ".", *args: Any, **kwargs: Any) -> Any:
                if Path(path) == home:
                    raise PermissionError(errno.EACCES, "Permission denied")
                return real_scandir(path, *args, **kwargs)

            monkeypatch.setattr(os, "scandir", scandir)
        else:
            real_list = private._access_control_text

            def unreadable(descriptor: int) -> str:
                if stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise OSError(errno.EIO, "Input/output error")
                return real_list(descriptor)

            monkeypatch.setattr(private, "_access_control_text", unreadable)
        try:
            with pytest.raises(ConfigError, match=r"^cannot examine .*; the standing home already exists"):
                standing.prepare_home(settings, home)
        finally:
            entry.chmod(0o600)

    def test_every_directory_above_a_standing_home_up_to_the_data_directory_is_made_private(
        self, settings: Settings
    ) -> None:
        home = standing.standing_home(settings, Provider.CODEX, "work")
        home.mkdir(parents=True)
        for directory in (settings.data_dir, settings.standing_homes_dir):
            directory.chmod(0o777)
        home.chmod(0o755)

        standing.prepare_home(settings, home)

        assert [d.stat().st_mode & 0o777 for d in (settings.data_dir, settings.standing_homes_dir, home)] == [0o700] * 3

    def test_the_directories_above_a_standing_home_are_made_private_before_the_home_is_examined(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = standing.standing_home(settings, Provider.CODEX, "work")
        home.mkdir(parents=True)
        settings.data_dir.chmod(0o755)
        settings.standing_homes_dir.chmod(0o777)
        modes_at_the_check: list[int] = []
        real_check = standing._refuse_an_existing_home_another_account_could_write

        def checking(target: Path) -> None:
            modes_at_the_check.extend(
                directory.stat().st_mode & 0o777 for directory in (settings.data_dir, settings.standing_homes_dir)
            )
            real_check(target)

        monkeypatch.setattr(standing, "_refuse_an_existing_home_another_account_could_write", checking)

        standing.prepare_home(settings, home)

        assert modes_at_the_check == [0o700, 0o700]

    @pytest.mark.parametrize("mode", [0o775, 0o707])
    def test_a_standing_home_that_already_exists_writable_by_others_is_refused_and_left_as_it_is(
        self, settings: Settings, mode: int
    ) -> None:
        home = standing.standing_home(settings, Provider.CLAUDE, "work")
        home.mkdir(parents=True)
        planted = home / "settings.json"
        planted.write_text('{"hooks": "planted"}')
        home.chmod(mode)

        with pytest.raises(ConfigError, match="cannot be trusted, so remove it and run the command again"):
            standing.prepare_home(settings, home)

        assert home.stat().st_mode & 0o777 == mode
        assert planted.read_text() == '{"hooks": "planted"}'

    def test_a_standing_home_that_cannot_be_examined_is_a_config_error(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = standing.standing_home(settings, Provider.CODEX, "work")
        self.fail_to_examine(monkeypatch, home, PermissionError(errno.EACCES, "Permission denied"))

        with pytest.raises(ConfigError, match=rf"^cannot examine {re.escape(str(home))}: PermissionError$"):
            standing.prepare_home(settings, home)

    def test_a_data_directory_under_one_others_can_write_to_is_refused_and_that_directory_is_left_as_it_is(
        self, settings: Settings
    ) -> None:
        unsafe = settings.data_dir.parent
        unsafe.chmod(0o777)
        try:
            with pytest.raises(ConfigError, match=r"its permissions \(0777\) let other accounts write"):
                standing.prepare_home(settings, standing.standing_home(settings, Provider.CODEX, "work"))
            assert unsafe.stat().st_mode & 0o777 == 0o777
            assert not settings.data_dir.exists()
        finally:
            unsafe.chmod(0o700)

    def test_an_ancestor_above_the_data_directory_that_others_can_write_to_is_refused(self, tmp_path: Path) -> None:
        unsafe = tmp_path / "shared"
        unsafe.mkdir()
        unsafe.chmod(0o777)
        try:
            nested = load_settings(env={"USAGEDASH_DATA_DIR": str(unsafe / "nested" / "data")}, home=tmp_path)
            with pytest.raises(ConfigError, match=rf"cannot use {unsafe}: its permissions \(0777\)"):
                standing.prepare_home(nested, standing.standing_home(nested, Provider.GROK, "work"))
            assert unsafe.stat().st_mode & 0o777 == 0o777
            assert not (unsafe / "nested").exists()
        finally:
            unsafe.chmod(0o700)

    def test_a_config_file_in_a_directory_the_user_chose_is_written_without_changing_that_directory(
        self, tmp_path: Path
    ) -> None:
        chosen = tmp_path / "chosen"
        chosen.mkdir()
        chosen.chmod(0o755)
        env = {"USAGEDASH_DATA_DIR": str(tmp_path / "data"), "USAGEDASH_CONFIG": str(chosen / "config.toml")}
        (chosen / "config.toml").write_text("port = 9001\n")
        settings = load_settings(env=env, home=tmp_path)
        home = standing.standing_home(settings, Provider.GROK, "work")

        assert standing.register(
            settings,
            Provider.GROK,
            home,
            lambda config_file: load_settings(env=env | {"USAGEDASH_CONFIG": str(config_file)}, home=tmp_path),
        )

        assert chosen.stat().st_mode & 0o777 == 0o755
        assert (chosen / "config.toml").stat().st_mode & 0o777 == 0o600
        assert tomllib.loads((chosen / "config.toml").read_text())["grok"][0]["mode"] == "cli"

    def test_a_config_directory_others_can_write_to_is_refused_before_the_login_command_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chosen = tmp_path / "chosen"
        chosen.mkdir()
        chosen.chmod(0o777)
        config = chosen / "config.toml"
        config.write_text("port = 9001\n")
        env = {"USAGEDASH_DATA_DIR": str(tmp_path / "data"), "USAGEDASH_CONFIG": str(config)}
        loaded = load_settings(env=env, home=tmp_path)
        run = FakeRun()
        monkeypatch.setattr(standing.subprocess, "run", run)
        try:
            with pytest.raises(ConfigError, match=r"its permissions \(0777\) let other accounts write"):
                standing.run_login(loaded, Provider.GROK, standing.standing_home(loaded, Provider.GROK, "work"))
            assert chosen.stat().st_mode & 0o777 == 0o777
            assert config.read_text() == "port = 9001\n"
            assert run.commands == []
        finally:
            chosen.chmod(0o700)

    def test_a_config_whose_parent_others_can_write_to_is_refused_and_the_file_is_left_as_it_is(
        self, tmp_path: Path
    ) -> None:
        outer = tmp_path / "outer"
        outer.mkdir()
        outer.chmod(0o777)
        chosen = outer / "chosen"
        chosen.mkdir()
        chosen.chmod(0o755)
        config = chosen / "config.toml"
        config.write_text("port = 9001\n")
        env = {"USAGEDASH_DATA_DIR": str(tmp_path / "data"), "USAGEDASH_CONFIG": str(config)}
        loaded = load_settings(env=env, home=tmp_path)
        try:
            with pytest.raises(ConfigError, match=rf"cannot use {outer}: its permissions \(0777\)"):
                standing.ensure_registrable(
                    loaded,
                    Provider.GROK,
                    standing.standing_home(loaded, Provider.GROK, "work"),
                    lambda config_file: load_settings(env=env | {"USAGEDASH_CONFIG": str(config_file)}, home=tmp_path),
                )
            assert outer.stat().st_mode & 0o777 == 0o777
            assert config.read_text() == "port = 9001\n"
            assert not (tmp_path / "data").exists()
        finally:
            outer.chmod(0o700)

    def test_the_default_config_file_is_created_in_the_data_directory_which_is_made_private(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        assert settings.config_path.parent == settings.data_dir
        assert not settings.data_dir.exists()

        self.register(settings, tmp_path, Provider.GROK)

        assert settings.data_dir.stat().st_mode & 0o777 == 0o700
        assert settings.config_path.stat().st_mode & 0o777 == 0o600

    def test_a_config_file_that_is_a_directory_is_a_config_error(self, settings: Settings, tmp_path: Path) -> None:
        settings.config_path.mkdir(parents=True)

        with pytest.raises(ConfigError, match=r"cannot use .*config\.toml: it is not a regular file"):
            self.register(settings, tmp_path, Provider.GROK)

    def test_a_config_file_that_does_not_load_is_a_config_error(self, settings: Settings, tmp_path: Path) -> None:
        settings.config_path.parent.mkdir(parents=True)
        settings.config_path.write_text("this is not TOML = = =\n")

        with pytest.raises(ConfigError, match=r"config\.toml"):
            self.register(settings, tmp_path, Provider.GROK)

        assert settings.config_path.read_text() == "this is not TOML = = =\n"

    def test_a_config_file_that_is_a_symbolic_link_is_refused_and_what_it_points_at_is_left_alone(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        target = tmp_path / "elsewhere.toml"
        target.write_text("port = 9001\n")
        settings.data_dir.mkdir(parents=True)
        settings.config_path.symlink_to(target)

        with pytest.raises(ConfigError, match=r"cannot use .*config\.toml: it is a symbolic link"):
            self.register(settings, tmp_path, Provider.GROK)

        assert target.read_text() == "port = 9001\n"
        assert settings.config_path.is_symlink()

    def test_a_config_file_that_cannot_be_read_back_is_a_config_error(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreadable(_descriptor: int, _size: int) -> bytes:
            raise OSError(errno.EIO, "Input/output error")

        settings.config_path.parent.mkdir(parents=True)
        settings.config_path.write_text("port = 9001\n")
        mode = settings.config_path.stat().st_mode
        monkeypatch.setattr("usagedash.private.os.read", unreadable)

        with pytest.raises(ConfigError, match=r"cannot use .*config\.toml: OSError$"):
            self.register(settings, tmp_path, Provider.GROK)

        assert settings.config_path.read_text() == "port = 9001\n"
        assert settings.config_path.stat().st_mode == mode

    @pytest.mark.parametrize(
        ("became_a_link", "message"),
        [
            (False, r"cannot use .*config\.toml: OSError$"),
            (True, r"cannot use .*config\.toml: it is a symbolic link$"),
        ],
    )
    def test_a_config_that_cannot_be_opened_to_make_it_owner_only_is_left_as_it_was(
        self,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        became_a_link: bool,
        message: str,
    ) -> None:
        settings.config_path.parent.mkdir(parents=True)
        settings.config_path.write_text("port = 9001\n")
        real_open = os.open
        real_islink = os.path.islink
        config_opens = 0

        def open_until_the_restriction(path: str | Path, flags: int, open_mode: int = 0o777) -> int:
            nonlocal config_opens
            if Path(path) == settings.config_path:
                config_opens += 1
                if config_opens > 1:
                    raise OSError("cannot open")
            return real_open(path, flags, open_mode)

        def islink(path: str | Path) -> bool:
            if Path(path) == settings.config_path:
                return became_a_link
            return real_islink(path)

        monkeypatch.setattr(standing.os, "open", open_until_the_restriction)
        monkeypatch.setattr(standing.os.path, "islink", islink)
        home = standing.standing_home(settings, Provider.GROK, "work")

        with pytest.raises(ConfigError, match=message):
            standing.ensure_registrable(settings, Provider.GROK, home, self.loading(settings, tmp_path))

        assert settings.config_path.read_text() == "port = 9001\n"
        assert settings.config_path.stat().st_mode & 0o777 == 0o600
        assert self.files_beside_the_config(settings) == ["config.toml", "config.toml.lock"]

    def test_a_config_whose_permissions_cannot_be_restricted_is_left_as_it_was(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings.config_path.parent.mkdir(parents=True)
        settings.config_path.write_text("port = 9001\n")
        opened: list[int] = []
        real_open = os.open

        def record_open(path: str | Path, flags: int, open_mode: int = 0o777) -> int:
            descriptor = real_open(path, flags, open_mode)
            if Path(path) == settings.config_path:
                opened.append(descriptor)
            return descriptor

        def refuse_restrict(_descriptor: int, _mode: int) -> None:
            raise OSError("cannot restrict")

        monkeypatch.setattr(standing.os, "open", record_open)
        monkeypatch.setattr(standing, "restrict_to_owner", refuse_restrict)
        home = standing.standing_home(settings, Provider.GROK, "work")

        with pytest.raises(ConfigError, match=r"cannot use .*config\.toml: OSError$"):
            standing.ensure_registrable(settings, Provider.GROK, home, self.loading(settings, tmp_path))

        assert settings.config_path.read_text() == "port = 9001\n"
        assert settings.config_path.stat().st_mode & 0o777 == 0o600
        assert len(opened) == 2
        for descriptor in opened:
            with pytest.raises(OSError, match="Bad file descriptor"):
                os.fstat(descriptor)
        assert self.files_beside_the_config(settings) == ["config.toml", "config.toml.lock"]

    @pytest.mark.parametrize("failing_step", ["fsync", "replace"])
    def test_a_write_that_cannot_be_completed_leaves_the_config_file_exactly_as_it_was(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_step: str
    ) -> None:
        settings.config_path.parent.mkdir(parents=True)
        settings.config_path.write_text("port = 9001\n")
        opened: list[int] = []
        real_open = standing.open_private_file

        def refuse(*_: object) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        def record_open(path: Path) -> int:
            opened.append(real_open(path))
            return opened[-1]

        monkeypatch.setattr(standing.os, failing_step, refuse)
        monkeypatch.setattr(standing, "open_private_file", record_open)

        with pytest.raises(ConfigError, match=r"cannot write .*config\.toml: OSError$"):
            self.register(settings, tmp_path, Provider.GROK)

        assert settings.config_path.read_text() == "port = 9001\n"
        assert sorted(entry.name for entry in settings.data_dir.iterdir()) == [
            "config.toml",
            "config.toml.lock",
            "homes",
        ]
        assert len(opened) == 2
        for descriptor in opened:
            with pytest.raises(OSError, match="Bad file descriptor"):
                os.fstat(descriptor)

    def test_a_lock_that_cannot_be_taken_is_a_config_error_and_nothing_is_written(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(descriptor: int, operation: int) -> None:
            raise OSError(errno.ENOLCK, "No locks available")

        monkeypatch.setattr(standing.fcntl, "flock", refuse)

        with pytest.raises(ConfigError, match=r"cannot lock .*config\.toml: OSError$"):
            self.register(settings, tmp_path, Provider.GROK)

        assert not settings.config_path.exists()

    def test_registering_the_same_login_again_changes_nothing(self, settings: Settings, tmp_path: Path) -> None:
        self.register(settings, tmp_path, Provider.GROK)
        before = settings.config_path.read_text()

        assert not self.register(self.reload(settings, tmp_path), tmp_path, Provider.GROK)
        assert settings.config_path.read_text() == before

    def test_a_login_registered_since_these_settings_were_loaded_is_not_registered_twice(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        assert self.register(settings, tmp_path, Provider.GROK)
        before = settings.config_path.read_text()

        assert not self.register(settings, tmp_path, Provider.GROK)

        assert settings.config_path.read_text() == before
        assert len(self.reload(settings, tmp_path).grok_sources) == 2

    def test_two_logins_that_loaded_their_settings_before_either_registered_are_both_kept(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        assert self.register(settings, tmp_path, Provider.GROK, "personal")
        assert self.register(settings, tmp_path, Provider.GROK, "work")

        homes = [source.home.name for source in self.reload(settings, tmp_path).grok_sources[1:]]
        assert homes == ["grok-personal", "grok-work"]

    def test_logins_that_finish_at_the_same_moment_are_all_registered_whole(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        names = [f"login-{number}" for number in range(8)]
        start = threading.Barrier(len(names))
        failures: list[BaseException] = []

        def register(name: str) -> None:
            try:
                start.wait(timeout=10)
                self.register(settings, tmp_path, Provider.CODEX, name)
            except BaseException as error:  # noqa: BLE001 - reported by the assertion below
                failures.append(error)

        threads = [threading.Thread(target=register, args=(name,)) for name in names]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert failures == []
        registered = sorted(source.home.name for source in self.reload(settings, tmp_path).codex_sources[1:])
        assert registered == sorted(f"codex-{name}" for name in names)

    def test_registration_waits_for_whoever_holds_the_lock(self, settings: Settings, tmp_path: Path) -> None:
        settings.data_dir.mkdir(parents=True)
        holder = os.open(settings.config_path.with_name("config.toml.lock"), os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(holder, fcntl.LOCK_EX)
        registered = threading.Event()

        def register() -> None:
            self.register(settings, tmp_path, Provider.GROK)
            registered.set()

        waiting = threading.Thread(target=register)
        waiting.start()
        try:
            assert not registered.wait(timeout=0.3)
            assert not settings.config_path.exists()
        finally:
            os.close(holder)
            waiting.join(timeout=30)

        assert registered.is_set()
        assert len(self.reload(settings, tmp_path).grok_sources) == 2

    @pytest.mark.parametrize("provider", list(Provider))
    @pytest.mark.parametrize(
        "logins_on_one_line",
        ["{section} = []", '{section} = [{{ {key} = "/synthetic/existing", mode = "cli" }}]'],
    )
    def test_a_config_that_lists_a_tool_s_logins_on_one_line_is_left_exactly_as_it_was(
        self, settings: Settings, tmp_path: Path, provider: Provider, logins_on_one_line: str
    ) -> None:
        key = "config_dir" if provider is Provider.CLAUDE else "home"
        before = f"port = 9001\n{logins_on_one_line.format(section=provider.value, key=key)}\n"
        settings.config_path.parent.mkdir(parents=True)
        settings.config_path.write_text(before)
        home = standing.standing_home(settings, provider, "work")

        with pytest.raises(ConfigError) as raised:
            self.register(settings, tmp_path, provider)

        assert str(raised.value) == (
            f"cannot add this login to {settings.config_path}: the file would not load with a "
            f"[[{provider.value}]] table added to its end. If it lists its {provider.value} logins on one line, "
            f'add this entry to that list and run the command again: {{ {key} = "{home}", mode = "cli" }}'
        )
        assert isinstance(raised.value.__cause__, ConfigError)
        assert settings.config_path.read_text() == before
        assert self.files_beside_the_config(settings) == ["config.toml", "config.toml.lock", "homes"]
        assert self.reload(settings, tmp_path).port == 9001

    @pytest.mark.parametrize("provider", list(Provider))
    def test_the_entry_the_refusal_says_to_add_registers_the_login_when_it_is_added(
        self, settings: Settings, tmp_path: Path, provider: Provider
    ) -> None:
        settings.config_path.parent.mkdir(parents=True)
        settings.config_path.write_text(f"{provider.value} = []\n")
        home = standing.standing_home(settings, provider, "work")
        with pytest.raises(ConfigError) as raised:
            self.register(settings, tmp_path, provider)
        entry = str(raised.value).partition("run the command again: ")[2]

        settings.config_path.write_text(f"{provider.value} = [{entry}]\n")

        assert standing.is_registered(self.reload(settings, tmp_path), provider, home)
        assert not self.register(settings, tmp_path, provider)
        assert settings.config_path.read_text() == f"{provider.value} = [{entry}]\n"

    @pytest.mark.parametrize(
        "line_end_to_some",
        [
            pytest.param("\u0085", id="next line"),
            pytest.param("\u2028", id="line separator"),
            pytest.param("\u2029", id="paragraph separator"),
        ],
    )
    def test_the_entry_to_add_by_hand_names_the_same_home_whatever_characters_its_path_holds(
        self, tmp_path: Path, line_end_to_some: str
    ) -> None:
        data_dir = tmp_path / f"da{line_end_to_some}ta"
        settings = load_settings(env={"USAGEDASH_DATA_DIR": str(data_dir)}, home=tmp_path)
        settings.config_path.parent.mkdir(parents=True)
        settings.config_path.write_text("codex = []\n", encoding="utf-8")
        home = standing.standing_home(settings, Provider.CODEX, "work")
        with pytest.raises(ConfigError) as raised:
            standing.register(settings, Provider.CODEX, home, self.loading(settings, tmp_path))
        entry = str(raised.value).partition("run the command again: ")[2]

        settings.config_path.write_text(f"codex = [{entry}]\n", encoding="utf-8")

        assert line_end_to_some in entry
        assert standing.is_registered(self.reload(settings, tmp_path), Provider.CODEX, home)

    def test_a_file_that_loads_without_the_login_in_it_never_becomes_the_config(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        settings.config_path.parent.mkdir(parents=True)
        settings.config_path.write_text("port = 9001\n")
        home = standing.standing_home(settings, Provider.GROK, "work")

        with pytest.raises(ConfigError, match=r"^cannot add this login to .*config\.toml") as raised:
            standing.register(settings, Provider.GROK, home, lambda _config_file: settings)

        assert raised.value.__cause__ is None
        assert settings.config_path.read_text() == "port = 9001\n"
        assert self.files_beside_the_config(settings) == ["config.toml", "config.toml.lock", "homes"]

    def test_a_login_can_be_checked_before_the_sign_in_without_creating_or_changing_the_config(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        home = prepared_codex_home(settings)

        standing.ensure_registrable(settings, Provider.CODEX, home, self.loading(settings, tmp_path))

        assert not settings.config_path.exists()
        assert self.files_beside_the_config(settings) == ["config.toml.lock", "homes"]

        settings.config_path.write_text("port = 9001\n")
        standing.ensure_registrable(settings, Provider.CODEX, home, self.loading(settings, tmp_path))

        assert settings.config_path.read_text() == "port = 9001\n"
        assert self.files_beside_the_config(settings) == ["config.toml", "config.toml.lock", "homes"]

    def test_the_check_before_the_sign_in_refuses_a_config_the_login_cannot_be_added_to(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        home = prepared_codex_home(settings)
        settings.config_path.write_text("codex = []\n")

        with pytest.raises(ConfigError, match=r"^cannot add this login to .*config\.toml"):
            standing.ensure_registrable(settings, Provider.CODEX, home, self.loading(settings, tmp_path))

        assert settings.config_path.read_text() == "codex = []\n"
        assert self.files_beside_the_config(settings) == ["config.toml", "config.toml.lock", "homes"]

    def test_the_check_before_the_sign_in_passes_a_login_that_is_already_registered(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        home = prepared_codex_home(settings)
        listed = f'codex = [{{ home = "{home}", mode = "cli" }}]\n'
        settings.config_path.write_text(listed)

        standing.ensure_registrable(settings, Provider.CODEX, home, self.loading(settings, tmp_path))

        assert settings.config_path.read_text() == listed

    @pytest.mark.parametrize("step", ["the check before the sign-in", "the registration after it"])
    @pytest.mark.parametrize(
        ("provider", "listed_as"),
        [
            pytest.param(Provider.CODEX, 'codex = [{{ home = "{home}", mode = "direct" }}]\n', id="codex, on one line"),
            pytest.param(Provider.GROK, '[[grok]]\nhome = "{home}"\nmode = "direct"\n', id="grok, in a table"),
            pytest.param(
                Provider.CLAUDE,
                '[[claude]]\nconfig_dir = "{home}"\nmode = "direct"\nkeychain_service = "Synthetic-credentials"\n',
                id="claude, in a table",
            ),
        ],
    )
    def test_a_home_the_config_lists_in_direct_mode_is_not_taken_for_a_standing_login(
        self, settings: Settings, tmp_path: Path, provider: Provider, listed_as: str, step: str
    ) -> None:
        home = standing.standing_home(settings, provider, "work")
        standing.prepare_home(settings, home)
        listed = listed_as.format(home=home)
        settings.config_path.write_text(listed)
        load = self.loading(settings, tmp_path)

        with pytest.raises(ConfigError, match=r'read in direct mode; set that entry\'s mode to "cli", or remove it'):
            if step == "the check before the sign-in":
                standing.ensure_registrable(settings, provider, home, load)
            else:
                standing.register(settings, provider, home, load)

        assert settings.config_path.read_text() == listed
        assert self.files_beside_the_config(settings) == ["config.toml", "config.toml.lock", "homes"]

    def other_spelling(self, home: Path, tmp_path: Path, spelling: str) -> Path:
        if spelling == "through ..":
            return home.parent / ".." / home.parent.name / home.name
        if spelling == "through a symbolic link":
            link = tmp_path / "linked-homes"
            link.symlink_to(home.parent, target_is_directory=True)
            return link / home.name
        assert spelling == "in another case"
        other = home.parent / home.name.upper()
        if not other.exists() or not os.path.samefile(other, home):
            pytest.skip("the file system treats the other case as a different directory")
        return other

    SPELLINGS = ("through ..", "through a symbolic link", "in another case")

    @pytest.mark.parametrize("spelling", ("through ..", "in another case"))
    def test_a_home_listed_under_another_spelling_of_its_path_is_not_registered_twice(
        self, settings: Settings, tmp_path: Path, spelling: str
    ) -> None:
        home = prepared_codex_home(settings)
        listed = f'codex = [{{ home = "{self.other_spelling(home, tmp_path, spelling)}", mode = "cli" }}]\n'
        settings.config_path.write_text(listed)

        registered = standing.register(settings, Provider.CODEX, home, self.loading(settings, tmp_path))

        assert registered is False
        assert settings.config_path.read_text() == listed

    @pytest.mark.parametrize("step", ["the check before the sign-in", "the registration after it"])
    def test_a_home_listed_through_a_link_this_account_owns_is_refused_with_the_path_to_list(
        self, settings: Settings, tmp_path: Path, step: str
    ) -> None:
        home = prepared_codex_home(settings)
        spelling = self.other_spelling(home, tmp_path, "through a symbolic link")
        listed = f'codex = [{{ home = "{spelling}", mode = "cli" }}]\n'
        settings.config_path.write_text(listed)
        load = self.loading(settings, tmp_path)

        with pytest.raises(
            ConfigError, match=rf"cannot use {re.escape(str(spelling.parent))}: it is a symbolic link"
        ) as raised:
            if step == "the check before the sign-in":
                standing.ensure_registrable(settings, Provider.CODEX, home, load)
            else:
                standing.register(settings, Provider.CODEX, home, load)

        assert f"The config file names {home} as {spelling}; name it {home} and run the command again" in str(
            raised.value
        )
        assert settings.config_path.read_text() == listed

    def test_a_home_listed_through_a_directory_another_account_can_write_to_is_refused_before_the_sign_in(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = standing.standing_home(settings, Provider.GROK, "work")
        standing.prepare_home(settings, home)
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(0o777)
        roundabout = shared / ".." / home.relative_to(tmp_path)
        listed = f'[[grok]]\nhome = "{roundabout}"\nmode = "cli"\n'
        settings.config_path.write_text(listed)
        load = self.loading(settings, tmp_path)
        started: list[list[str]] = []
        monkeypatch.setattr(standing.subprocess, "run", recording_commands(started))
        refused = rf"cannot use {re.escape(str(shared))}: its permissions \(0777\)"
        try:
            with pytest.raises(ConfigError, match=refused) as before:
                standing.ensure_registrable(settings, Provider.GROK, home, load)
            with pytest.raises(ConfigError, match=refused) as during:
                standing.sign_in_and_register(settings, Provider.GROK, home, load)
        finally:
            shared.chmod(0o700)

        assert all(f"name it {home} and run the command again" in str(raised.value) for raised in (before, during))
        assert started == []
        assert settings.config_path.read_text() == listed

    def test_the_home_s_own_path_is_never_refused_as_a_spelling_of_itself(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        home = prepared_codex_home(settings)
        home.chmod(0o777)
        try:
            with pytest.raises(
                ConfigError, match=rf"cannot use {re.escape(str(home))}: its permissions \(0777\)"
            ) as raised:
                standing._refuse_a_listed_spelling_another_account_can_redirect(home, home)
        finally:
            home.chmod(0o700)

        assert "name it" not in str(raised.value)

    @pytest.mark.parametrize("spelling", SPELLINGS)
    @pytest.mark.parametrize("step", ["the check before the sign-in", "the registration after it"])
    def test_a_home_listed_in_direct_mode_under_another_spelling_of_its_path_is_refused(
        self, settings: Settings, tmp_path: Path, spelling: str, step: str
    ) -> None:
        home = standing.standing_home(settings, Provider.GROK, "work")
        standing.prepare_home(settings, home)
        listed = f'[[grok]]\nhome = "{self.other_spelling(home, tmp_path, spelling)}"\nmode = "direct"\n'
        settings.config_path.write_text(listed)
        load = self.loading(settings, tmp_path)

        with pytest.raises(ConfigError, match=r"read in direct mode"):
            if step == "the check before the sign-in":
                standing.ensure_registrable(settings, Provider.GROK, home, load)
            else:
                standing.register(settings, Provider.GROK, home, load)

        assert settings.config_path.read_text() == listed

    def test_a_listed_home_that_does_not_exist_yet_is_matched_by_its_path_with_dot_dot_taken_out(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        home = standing.standing_home(settings, Provider.CODEX, "work")
        roundabout = settings.standing_homes_dir / "elsewhere" / ".." / home.name
        unrelated = settings.standing_homes_dir / "codex-other"
        reload = self.loading(settings, tmp_path)
        settings.config_path.parent.mkdir(parents=True)

        settings.config_path.write_text(f'codex = [{{ home = "{roundabout}", mode = "cli" }}]\n')
        listed_by_another_path = standing.is_registered(reload(settings.config_path), Provider.CODEX, home)
        settings.config_path.write_text(f'codex = [{{ home = "{unrelated}", mode = "cli" }}]\n')
        listed_elsewhere = standing.is_registered(reload(settings.config_path), Provider.CODEX, home)

        assert not home.exists()
        assert (listed_by_another_path, listed_elsewhere) == (True, False)

    def listed_source(self, provider: Provider, path: Path, mode: str) -> str:
        key = "config_dir" if provider is Provider.CLAUDE else "home"
        lines = [f"[[{provider.value}]]", f'{key} = "{path}"', f'mode = "{mode}"']
        if provider is Provider.CLAUDE and mode == "direct":
            lines.append('keychain_service = "Synthetic-credentials"')
        return "\n".join(lines) + "\n"

    def detour_to(self, settings: Settings, home: Path, through: str) -> Path:
        blocked = settings.standing_homes_dir / "not-a-directory"
        if through == "a file":
            blocked.write_text("")
        return blocked / ".." / home.name

    @pytest.mark.parametrize("provider", list(Provider))
    @pytest.mark.parametrize("mode", ["cli", "direct"])
    @pytest.mark.parametrize("through", ["a missing directory", "a file"])
    @pytest.mark.parametrize("step", ["the check before the sign-in", "the registration after it"])
    def test_a_detour_to_a_listed_home_stays_one_source_the_collector_can_build(
        self, settings: Settings, tmp_path: Path, provider: Provider, mode: str, through: str, step: str
    ) -> None:
        home = standing.standing_home(settings, provider, "work")
        standing.prepare_home(settings, home)
        listed = self.listed_source(provider, self.detour_to(settings, home, through), mode)
        settings.config_path.write_text(listed)
        load = self.loading(settings, tmp_path)
        loaded = load(settings.config_path)

        if mode == "direct":
            with pytest.raises(ConfigError, match=r"read in direct mode"):
                standing.is_registered(loaded, provider, home)
            with pytest.raises(ConfigError, match=r"read in direct mode"):
                if step == "the check before the sign-in":
                    standing.ensure_registrable(settings, provider, home, load)
                else:
                    standing.register(settings, provider, home, load)
        else:
            assert standing.is_registered(loaded, provider, home) is True
            if step == "the check before the sign-in":
                standing.ensure_registrable(settings, provider, home, load)
            else:
                assert standing.register(settings, provider, home, load) is False

        assert settings.config_path.read_text() == listed
        stored = f"{provider.value}:{directory_identity(home)}"
        with httpx2.Client() as client:
            bundles = build_bundles(load(settings.config_path), client)
        assert [bundle.source.source_id for bundle in bundles].count(stored) == 1

    def test_a_listed_home_that_exists_is_not_the_new_home_that_does_not(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        other = standing.standing_home(settings, Provider.CODEX, "other")
        standing.prepare_home(settings, other)
        settings.config_path.write_text(f'codex = [{{ home = "{other}", mode = "cli" }}]\n')
        home = standing.standing_home(settings, Provider.CODEX, "work")

        assert (
            standing.is_registered(self.loading(settings, tmp_path)(settings.config_path), Provider.CODEX, home)
            is False
        )

    def test_a_listed_home_that_cannot_be_examined_is_a_config_error(self, settings: Settings, tmp_path: Path) -> None:
        home = prepared_codex_home(settings)
        locked = tmp_path / "locked"
        locked.mkdir()
        settings.config_path.write_text(f'codex = [{{ home = "{locked / "inside"}", mode = "cli" }}]\n')
        loaded = self.loading(settings, tmp_path)(settings.config_path)
        locked.chmod(0)
        try:
            with pytest.raises(ConfigError, match=r"^cannot identify .*inside: PermissionError$"):
                standing.is_registered(loaded, Provider.CODEX, home)
        finally:
            locked.chmod(0o700)

    def test_paths_that_cannot_be_followed_to_where_they_lead_are_a_config_error(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = standing.standing_home(settings, Provider.CODEX, "work")
        settings.config_path.parent.mkdir(parents=True)
        settings.config_path.write_text(f'codex = [{{ home = "{tmp_path / "missing"}", mode = "cli" }}]\n')
        loaded = self.loading(settings, tmp_path)(settings.config_path)

        def unfollowable(self: Path, strict: bool = False) -> Path:
            raise OSError(errno.ELOOP, "Too many levels of symbolic links")

        monkeypatch.setattr(Path, "resolve", unfollowable)

        with pytest.raises(ConfigError, match=r"^cannot identify .*: OSError$"):
            standing.is_registered(loaded, Provider.CODEX, home)

    def write_port(self, settings: Settings) -> None:
        settings.config_path.parent.mkdir(parents=True)
        settings.config_path.write_text("port = 9001\n")

    def raise_interrupt(self, *_args: object) -> None:
        raise KeyboardInterrupt

    def unchanged(self, settings: Settings, *names: str) -> None:
        assert settings.config_path.read_text() == "port = 9001\n"
        assert self.files_beside_the_config(settings) == ["config.toml", *names]

    def test_ctrl_c_during_the_candidate_write_removes_it(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.write_port(settings)
        monkeypatch.setattr(os, "fsync", self.raise_interrupt)
        with pytest.raises(KeyboardInterrupt):
            standing._write_beside(settings.config_path, b"port = 9001\n")
        self.unchanged(settings)

    def test_ctrl_c_during_the_candidate_check_removes_it(self, settings: Settings, tmp_path: Path) -> None:
        self.write_port(settings)
        home = standing.standing_home(settings, Provider.CODEX, "work")
        load = self.loading(settings, tmp_path)

        def interrupt(config_file: Path) -> Settings:
            if config_file == settings.config_path:
                return load(config_file)
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            standing._config_with_login(settings.config_path, Provider.CODEX, home, interrupt)
        self.unchanged(settings)

    def test_ctrl_c_during_the_replace_removes_the_candidate(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.write_port(settings)
        candidate = standing._write_beside(settings.config_path, b"port = 9001\n[[codex]]\n")
        monkeypatch.setattr(os, "replace", self.raise_interrupt)
        with pytest.raises(KeyboardInterrupt):
            standing._move_into_place(candidate, settings.config_path)
        self.unchanged(settings)

    def test_ctrl_c_before_the_candidate_is_installed_removes_it(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.write_port(settings)
        home = standing.standing_home(settings, Provider.CODEX, "work")
        monkeypatch.setattr(standing, "_move_into_place", self.raise_interrupt)
        with standing._registration_lock(settings.config_path), pytest.raises(KeyboardInterrupt):
            standing._register_locked(settings, Provider.CODEX, home, self.loading(settings, tmp_path))
        self.unchanged(settings, "config.toml.lock")

    def test_ctrl_c_while_dropping_the_probe_removes_it_before_sign_in(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.write_port(settings)
        home = standing.standing_home(settings, Provider.CODEX, "work")
        real_discard = standing._discard
        discards = {"count": 0}

        def discard_once(temporary: Path) -> None:
            discards["count"] += 1
            if discards["count"] == 1:
                raise KeyboardInterrupt
            real_discard(temporary)

        monkeypatch.setattr(standing, "_discard", discard_once)
        monkeypatch.setattr(standing, "run_login", self.raise_interrupt)
        with pytest.raises(KeyboardInterrupt):
            standing.sign_in_and_register(settings, Provider.CODEX, home, self.loading(settings, tmp_path))
        assert discards["count"] == 2
        self.unchanged(settings, "config.toml.lock", "homes")

    @pytest.mark.parametrize("step", ["the check before the sign-in", "the registration after it"])
    def test_a_home_whose_path_utf_8_cannot_hold_is_a_config_error_and_nothing_is_written(
        self, settings: Settings, tmp_path: Path, step: str
    ) -> None:
        settings.config_path.parent.mkdir(parents=True)
        settings.config_path.write_text("port = 9001\n")
        home = settings.standing_homes_dir / "codex-\udcff"
        load = self.loading(settings, tmp_path)

        with pytest.raises(ConfigError, match=r"the path of its home cannot be written as UTF-8 text$"):
            if step == "the check before the sign-in":
                standing.ensure_registrable(settings, Provider.CODEX, home, load)
            else:
                with standing._registration_lock(settings.config_path):
                    standing._config_with_login(settings.config_path, Provider.CODEX, home, load)

        assert settings.config_path.read_text() == "port = 9001\n"
        assert self.files_beside_the_config(settings) == ["config.toml", "config.toml.lock"]

    def already_listed(self, settings: Settings) -> tuple[Path, str]:
        home = prepared_codex_home(settings)
        return home, f'codex = [{{ home = "{home}", mode = "cli" }}]\n'

    @pytest.mark.parametrize("step", ["the check before the sign-in", "the registration after it"])
    def test_a_config_that_already_lists_the_login_is_still_made_owner_only(
        self, settings: Settings, tmp_path: Path, step: str
    ) -> None:
        home, listed = self.already_listed(settings)
        settings.config_path.write_text(listed)
        settings.config_path.chmod(0o644)
        load = self.loading(settings, tmp_path)

        if step == "the check before the sign-in":
            standing.ensure_registrable(settings, Provider.CODEX, home, load)
        else:
            assert standing.register(settings, Provider.CODEX, home, load) is False

        assert settings.config_path.stat().st_mode & 0o777 == 0o600
        assert settings.config_path.read_text() == listed

    @pytest.mark.parametrize("step", ["the check before the sign-in", "the registration after it"])
    def test_a_config_other_accounts_can_write_is_refused_and_left_writable(
        self, settings: Settings, tmp_path: Path, step: str
    ) -> None:
        settings.config_path.parent.mkdir(parents=True)
        text = '[binaries]\ngrok = "/not/run"\n'
        settings.config_path.write_text(text)
        settings.config_path.chmod(0o664)
        home = standing.standing_home(settings, Provider.GROK, "work")
        load = self.loading(settings, tmp_path)

        with pytest.raises(ConfigError, match=r"its permissions \(0664\) let other accounts write"):
            if step == "the check before the sign-in":
                standing.ensure_registrable(settings, Provider.GROK, home, load)
            else:
                standing.register(settings, Provider.GROK, home, load)

        assert settings.config_path.read_text() == text
        assert settings.config_path.stat().st_mode & 0o777 == 0o664

    @pytest.mark.parametrize("step", ["the check before the sign-in", "the registration after it"])
    def test_a_symbolic_link_in_place_of_a_config_that_already_lists_the_login_is_refused(
        self, settings: Settings, tmp_path: Path, step: str
    ) -> None:
        home, listed = self.already_listed(settings)
        target = tmp_path / "elsewhere.toml"
        target.write_text(listed)
        target.chmod(0o644)
        settings.config_path.symlink_to(target)
        load = self.loading(settings, tmp_path)

        with pytest.raises(ConfigError, match=r"cannot use .*config\.toml: it is a symbolic link"):
            if step == "the check before the sign-in":
                standing.ensure_registrable(settings, Provider.CODEX, home, load)
            else:
                standing.register(settings, Provider.CODEX, home, load)

        assert settings.config_path.is_symlink()
        assert (target.read_text(), target.stat().st_mode & 0o777) == (listed, 0o644)

    def test_registering_keeps_settings_already_in_the_config_file(self, settings: Settings, tmp_path: Path) -> None:
        settings.config_path.parent.mkdir(parents=True)
        settings.config_path.write_text("port = 9001\n")

        self.register(settings, tmp_path, Provider.CODEX)

        parsed = tomllib.loads(settings.config_path.read_text())
        assert parsed["port"] == 9001
        assert parsed["codex"][0]["mode"] == "cli"

    def test_each_provider_s_login_command_targets_only_the_standing_home(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        base = {
            "PATH": "/usr/bin",
            "CLAUDECODE": "1",
            "ANTHROPIC_API_KEY": "synthetic",
            "CODEX_HOME": "/elsewhere",
            "OPENAI_API_KEY": "synthetic",
            "GROK_AUTH": "synthetic",
            "XAI_API_KEY": "synthetic",
        }
        home = tmp_path / "standing"
        others = {"CLAUDECODE", "ANTHROPIC_API_KEY", "CODEX_HOME", "OPENAI_API_KEY", "GROK_AUTH", "XAI_API_KEY"}

        assert standing.login_command(settings, Provider.CLAUDE)[1:] == ["auth", "login"]
        assert standing.login_command(settings, Provider.CODEX)[1:] == ["login"]
        assert standing.login_command(settings, Provider.GROK)[1:] == ["login"]
        for provider, own, home_variable, pinned in (
            (Provider.CLAUDE, {"CLAUDECODE", "ANTHROPIC_API_KEY"}, "CLAUDE_CONFIG_DIR", {"USER": keychain_account()}),
            (Provider.CODEX, {"CODEX_HOME", "OPENAI_API_KEY"}, "CODEX_HOME", {}),
            (Provider.GROK, {"GROK_AUTH", "XAI_API_KEY"}, "GROK_HOME", {}),
        ):
            kept = {name: base[name] for name in sorted(others - own)}
            assert standing.login_environment(provider, home, base) == {
                "PATH": "/usr/bin",
                **kept,
                **pinned,
                home_variable: str(home),
            }

    def test_a_claude_sign_in_files_its_login_under_the_account_every_read_names(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression lock: a sign-in from a shell with ``USER=unknown`` filed the login where no read looked."""
        home = self.made_claude_home(settings)
        monkeypatch.setenv("USER", "unknown")
        captured: dict[str, str] = {}
        monkeypatch.setattr(standing.subprocess, "run", recording_environment(captured))

        assert standing.run_login(settings, Provider.CLAUDE, home) == 0

        assert captured["USER"] == keychain_account()
        assert captured["USER"] != "unknown"

    def test_signing_in_to_an_already_listed_claude_home_uses_that_credential_spelling(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self.made_claude_home(settings)
        listed = home.parent / ".." / home.parent.name / home.name
        settings.config_path.parent.mkdir(parents=True, exist_ok=True)
        settings.config_path.write_text(standing_source_table("claude", listed))
        loaded = self.loading(settings, tmp_path)(settings.config_path)
        captured: dict[str, str] = {}
        monkeypatch.setattr(standing.subprocess, "run", recording_environment(captured))

        assert standing.run_login(loaded, Provider.CLAUDE, home) == 0

        assert captured["CLAUDE_CONFIG_DIR"] == str(home)
        assert captured["CLAUDE_SECURESTORAGE_CONFIG_DIR"] == str(listed)
        assert "CLAUDECODE" not in captured

    def test_a_listed_claude_spelling_is_not_resolved_again_into_the_sign_in_directory(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self.made_claude_home(settings)
        listed = home.parent / ".." / home.parent.name / home.name
        settings.config_path.parent.mkdir(parents=True, exist_ok=True)
        settings.config_path.write_text(standing_source_table("claude", listed))
        loaded = self.loading(settings, tmp_path)(settings.config_path)
        resolutions = {"listed": 0}
        real_identity = standing.directory_identity

        def identity(path: Path) -> Path:
            if path == listed:
                resolutions["listed"] += 1
                if resolutions["listed"] > 1:
                    return Path("/shared/other-user-directory")
            return real_identity(path)

        captured: dict[str, str] = {}

        def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            env = kwargs["env"]
            assert isinstance(env, dict)
            captured.update(env)
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(standing, "directory_identity", identity)
        monkeypatch.setattr(standing.subprocess, "run", run)

        assert standing.run_login(loaded, Provider.CLAUDE, home) == 0

        assert captured["CLAUDE_CONFIG_DIR"] == str(home)
        assert "/shared/other-user-directory" not in captured.values()
        assert captured["CLAUDE_SECURESTORAGE_CONFIG_DIR"] == str(listed)

    @pytest.mark.parametrize(
        "shape",
        [
            "a link in a directory another account can write to",
            "a link this account owns, leading through an alias another account can retarget",
        ],
    )
    def test_a_listed_claude_spelling_another_account_could_redirect_is_refused_before_sign_in(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
    ) -> None:
        home = self.made_claude_home(settings)
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(0o777)
        alias = shared / "alias"
        alias.symlink_to(home)
        listed = alias
        if shape.startswith("a link this account owns"):
            safe = tmp_path / "safe"
            safe.mkdir()
            listed = safe / "listed"
            listed.symlink_to(alias)
        settings.config_path.parent.mkdir(parents=True, exist_ok=True)
        settings.config_path.write_text(standing_source_table("claude", listed))
        loaded = self.loading(settings, tmp_path)(settings.config_path)
        started: list[list[str]] = []
        monkeypatch.setattr(standing.subprocess, "run", recording_commands(started))
        try:
            with pytest.raises(
                ConfigError, match=rf"cannot use {re.escape(str(listed))}: it is a symbolic link"
            ) as raised:
                standing.run_login(loaded, Provider.CLAUDE, home)
        finally:
            shared.chmod(0o700)

        assert f"name it {home} and run the command again" in str(raised.value)
        assert started == []

    @pytest.mark.parametrize(
        "spelling", ["through a directory another account can write to", "through a link this account owns"]
    )
    def test_a_listed_claude_account_file_reached_through_an_entry_another_account_could_redirect_is_refused(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
    ) -> None:
        home = self.made_claude_home(settings)
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(0o777)
        if spelling == "through a link this account owns":
            alias = tmp_path / "alias"
            alias.symlink_to(home)
            account_file = alias / ".claude.json"
            refused_for = rf"cannot use {re.escape(str(alias))}: it is a symbolic link"
        else:
            account_file = shared / ".." / home.relative_to(tmp_path) / ".claude.json"
            refused_for = rf"cannot use {re.escape(str(shared))}: its permissions \(0777\)"
        self.list_claude(settings, home, account_file)
        loaded = self.loading(settings, tmp_path)(settings.config_path)
        started: list[list[str]] = []
        monkeypatch.setattr(standing.subprocess, "run", recording_commands(started))
        try:
            with pytest.raises(ConfigError, match=refused_for) as raised:
                standing.run_login(loaded, Provider.CLAUDE, home)
        finally:
            shared.chmod(0o700)

        assert f"name it {home} and run the command again" in str(raised.value)
        assert started == []

    @pytest.mark.parametrize("component", ["missing", "a file"])
    def test_a_listed_claude_account_file_spelled_through_a_missing_or_file_component_still_signs_in(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str
    ) -> None:
        home = self.made_claude_home(settings)
        pivot = home.parent / "pivot"
        if component == "a file":
            pivot.write_text("")
        account_file = pivot / ".." / home.name / ".claude.json"
        self.list_claude(settings, home, account_file)
        loaded = self.loading(settings, tmp_path)(settings.config_path)
        assert standing.is_registered(loaded, Provider.CLAUDE, home)
        captured: dict[str, str] = {}
        monkeypatch.setattr(standing.subprocess, "run", recording_environment(captured))

        assert standing.run_login(loaded, Provider.CLAUDE, home) == 0

        assert captured["CLAUDE_CONFIG_DIR"] == str(home)

    def test_a_listed_claude_account_file_in_another_case_signs_in_when_the_filesystem_keeps_one_file(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self.made_claude_home(settings)
        (home / ".claude.json").write_text("{}")
        other_case = home / ".CLAUDE.JSON"
        if not other_case.exists() or not os.path.samefile(other_case, home / ".claude.json"):
            pytest.skip("this filesystem keeps the other case as a different file")
        self.list_claude(settings, home, other_case)
        loaded = self.loading(settings, tmp_path)(settings.config_path)
        started: list[list[str]] = []
        monkeypatch.setattr(standing.subprocess, "run", recording_commands(started))

        assert standing.run_login(loaded, Provider.CLAUDE, home) == 0

        assert len(started) == 1

    @pytest.mark.parametrize("other_name", ["account.json in the home", ".claude.json in another directory"])
    def test_a_listed_claude_account_file_under_another_name_is_refused_even_while_it_is_the_same_file(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, other_name: str
    ) -> None:
        home = self.made_claude_home(settings)
        (home / ".claude.json").write_text("{}")
        other_name = home / "account.json" if other_name.startswith("account") else tmp_path / ".claude.json"
        os.link(home / ".claude.json", other_name)
        self.list_claude(settings, home, other_name)
        loaded = self.loading(settings, tmp_path)(settings.config_path)
        monkeypatch.setattr(standing.subprocess, "run", lambda *_a, **_k: pytest.fail("the sign-in was started"))

        with pytest.raises(ConfigError, match="which this sign-in does not write"):
            standing.run_login(loaded, Provider.CLAUDE, home)

    @pytest.mark.parametrize("placed", ["in the file's own place", "elsewhere, leading to the file"])
    def test_a_link_as_the_listed_claude_account_file_is_refused_before_sign_in(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, placed: str
    ) -> None:
        home = self.made_claude_home(settings)
        elsewhere = tmp_path / "elsewhere.json"
        elsewhere.write_text("{}")
        if placed == "in the file's own place":
            account_file = home / ".claude.json"
            account_file.symlink_to(elsewhere)
        else:
            (home / ".claude.json").write_text("{}")
            account_file = tmp_path / "link.json"
            account_file.symlink_to(home / ".claude.json")
        self.list_claude(settings, home, account_file)
        loaded = self.loading(settings, tmp_path)(settings.config_path)
        monkeypatch.setattr(standing.subprocess, "run", lambda *_a, **_k: pytest.fail("the sign-in was started"))

        with pytest.raises(ConfigError, match="which is a symbolic link; point account_file at that home's"):
            standing.run_login(loaded, Provider.CLAUDE, home)

        assert account_file.is_symlink()
        assert elsewhere.read_text() == "{}"

    def test_a_listed_claude_account_file_that_cannot_be_examined_is_a_config_error(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self.made_claude_home(settings)
        account_file = home / ".claude.json"
        self.list_claude(settings, home, account_file)
        loaded = self.loading(settings, tmp_path)(settings.config_path)
        real_lstat = os.lstat

        def lstat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
            if Path(path) == account_file:
                raise PermissionError(errno.EACCES, "denied")
            return real_lstat(path, *args, **kwargs)

        monkeypatch.setattr(standing.os, "lstat", lstat)
        monkeypatch.setattr(standing.subprocess, "run", lambda *_a, **_k: pytest.fail("the sign-in was started"))

        with pytest.raises(ConfigError, match=rf"cannot examine {re.escape(str(account_file))}: PermissionError"):
            standing.run_login(loaded, Provider.CLAUDE, home)

    @pytest.mark.parametrize("account_name", [".claude.json", "account.json"])
    def test_a_listed_claude_account_file_outside_the_standing_home_is_refused_before_sign_in(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, account_name: str
    ) -> None:
        home = self.made_claude_home(settings)
        other = standing.standing_home(settings, Provider.CLAUDE, "other")
        other.parent.mkdir(parents=True, exist_ok=True)
        other.mkdir()
        external = tmp_path / account_name
        settings.config_path.parent.mkdir(parents=True, exist_ok=True)
        settings.config_path.write_text(
            "\n[[claude]]\n"
            f"config_dir = {json.dumps(str(home))}\n"
            f"account_file = {json.dumps(str(external))}\n"
            'mode = "cli"\n'
            "\n[[claude]]\n"
            f"config_dir = {json.dumps(str(other))}\n"
            'mode = "cli"\n'
        )
        loaded = self.loading(settings, tmp_path)(settings.config_path)
        started: list[list[str]] = []
        monkeypatch.setattr(standing.subprocess, "run", recording_commands(started))

        with pytest.raises(ConfigError, match="which this sign-in does not write"):
            standing.run_login(loaded, Provider.CLAUDE, home)

        assert started == []

    def test_a_listed_claude_account_file_in_the_standing_home_still_signs_in(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self.made_claude_home(settings)
        account_file = home / ".claude.json"
        self.list_claude(settings, home, account_file)
        loaded = self.loading(settings, tmp_path)(settings.config_path)
        captured: dict[str, str] = {}
        monkeypatch.setattr(standing.subprocess, "run", recording_environment(captured))

        assert standing.run_login(loaded, Provider.CLAUDE, home) == 0

        assert captured["CLAUDE_CONFIG_DIR"] == str(home)

    def test_signing_in_to_an_unlisted_claude_home_names_only_that_home(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "standing"
        captured: dict[str, str] = {}
        monkeypatch.setattr(standing.subprocess, "run", recording_environment(captured))

        assert standing.run_login(settings, Provider.CLAUDE, home) == 0

        assert captured["CLAUDE_CONFIG_DIR"] == str(home)
        assert "CLAUDE_SECURESTORAGE_CONFIG_DIR" not in captured

    def test_running_the_login_passes_through_the_vendor_cli_s_exit_status(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(returncode=7)
        monkeypatch.setattr(standing.subprocess, "run", run)

        assert standing.run_login(settings, Provider.GROK, tmp_path / "standing") == 7
        assert run.commands[0][1:] == ["login"]

    def test_a_vendor_cli_that_cannot_be_started_is_a_config_error(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(standing.subprocess, "run", FakeRun(error=FileNotFoundError("grok")))
        with pytest.raises(ConfigError, match="could not start"):
            standing.run_login(settings, Provider.GROK, tmp_path / "standing")

    def test_a_login_signs_in_with_the_spelling_listed_when_the_sign_in_starts(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self.made_claude_home(settings)
        spelling = home.parent / home.name / ".." / home.name
        assert str(spelling) != str(home)
        settings.config_path.parent.mkdir(parents=True, exist_ok=True)
        settings.config_path.write_text(standing_source_table("claude", spelling))
        assert standing._registered_claude_config_dir(settings, home) is None
        captured: dict[str, str] = {}

        def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            env = kwargs["env"]
            assert isinstance(env, dict)
            captured.update(env)
            return subprocess.CompletedProcess(command, 0)

        real_paths = agent.agent_paths

        def paths(_home: Path | None = None, _user: str | None = None) -> agent.AgentPaths:
            return real_paths(home=tmp_path, user="persona")

        monkeypatch.setattr(standing.subprocess, "run", run)
        monkeypatch.setattr(agent, "agent_paths", paths)
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: False)
        monkeypatch.setattr(agent, "kickstart_if_loaded", lambda _paths: False)

        assert cli._login(settings, Provider.CLAUDE, "work") == 0

        assert captured["CLAUDE_CONFIG_DIR"] == str(home)
        assert captured["CLAUDE_SECURESTORAGE_CONFIG_DIR"] == str(spelling)
        assert settings.config_path.read_text() == standing_source_table("claude", spelling)

    def test_a_sign_in_with_no_config_file_ignores_a_spelling_the_settings_still_remember(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = self.made_claude_home(settings)
        spelling = home.parent / home.name / ".." / home.name
        settings.config_path.parent.mkdir(parents=True, exist_ok=True)
        settings.config_path.write_text(standing_source_table("claude", spelling))
        settings.config_path.write_text(
            settings.config_path.read_text()
            + f"\n[binaries]\nclaude = {json.dumps(str(tmp_path / 'removed-claude'))}\n"
        )
        (tmp_path / "removed-claude").write_text("")
        remembered = self.loading(settings, tmp_path)(settings.config_path)
        settings.config_path.unlink()
        captured: dict[str, str] = {}
        started: list[list[str]] = []

        def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            env = kwargs["env"]
            assert isinstance(env, dict)
            captured.update(env)
            started.append(command)
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(standing.subprocess, "run", run)

        assert standing.sign_in_and_register(remembered, Provider.CLAUDE, home, self.loading(settings, tmp_path)) == 0

        assert captured["CLAUDE_CONFIG_DIR"] == str(home)
        assert "CLAUDE_SECURESTORAGE_CONFIG_DIR" not in captured
        assert started[0][0] == str(
            Path(self.loading(settings, tmp_path)(settings.config_path).binaries.claude).resolve()
        )
        assert started[0][0] != str(tmp_path / "removed-claude")
        written = tomllib.loads(settings.config_path.read_text())
        assert written["claude"] == [{"config_dir": str(home), "mode": "cli"}]
        assert "binaries" not in written

    def test_a_sign_in_holds_the_registration_lock_until_its_own_spelling_is_listed(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = standing.standing_home(settings, Provider.CLAUDE, "work")
        spelling = home.parent / home.name / ".." / home.name
        flag = tmp_path / "registrar-flag"
        source = Path(standing.__file__).resolve().parents[1]
        lock = settings.config_path.with_name(f"{settings.config_path.name}.lock")
        captured: dict[str, str] = {}
        child: subprocess.Popen[str] | None = None
        real_run = subprocess.run

        def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            nonlocal child
            env = kwargs.get("env")
            if not isinstance(env, dict):
                return real_run(command, **kwargs)
            captured.update(env)
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    REGISTRAR_WHILE_THE_LOCK_IS_HELD,
                    str(flag),
                    str(settings.data_dir),
                    str(tmp_path),
                    str(spelling),
                    str(source),
                ],
                env={**os.environ, "PYTHONPATH": str(source), "USAGEDASH_DATA_DIR": str(settings.data_dir)},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 5
            while not flag.exists():
                if time.monotonic() >= deadline or child.poll() is not None:
                    stdout, stderr = child.communicate(timeout=1)
                    raise AssertionError(f"registrar never reached the lock: {stderr or stdout}")
                time.sleep(0.02)
            time.sleep(0.2)
            assert flag.read_text() == "at-lock"
            probe = real_run(
                [sys.executable, "-c", REGISTRATION_LOCK_PROBE, str(lock)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert probe.stdout.strip() == "blocked", probe.stderr
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(standing.subprocess, "run", run)
        try:
            assert standing.sign_in_and_register(settings, Provider.CLAUDE, home, self.loading(settings, tmp_path)) == 0
            assert child is not None
            stdout, stderr = child.communicate(timeout=5)
            assert child.returncode == 0, stderr or stdout
        finally:
            if child is not None and child.poll() is None:
                child.kill()
                child.communicate(timeout=5)

        assert flag.read_text() == "already"
        assert captured["CLAUDE_CONFIG_DIR"] == str(home)
        assert "CLAUDE_SECURESTORAGE_CONFIG_DIR" not in captured
        assert tomllib.loads(settings.config_path.read_text())["claude"] == [{"config_dir": str(home), "mode": "cli"}]

    @pytest.mark.parametrize(
        "failure",
        [
            OSError(errno.EIO, "Input/output error"),
            PermissionError(errno.EACCES, "Permission denied"),
            ValueError("bad"),
        ],
    )
    def test_a_config_that_cannot_be_examined_is_not_replaced(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ) -> None:
        home = prepared_codex_home(settings)
        settings.config_path.write_text("port = 9001\n")
        self.fail_to_examine(monkeypatch, settings.config_path, failure)

        with pytest.raises(ConfigError, match=rf"^cannot examine .*config\.toml: {type(failure).__name__}$"):
            standing.register(settings, Provider.CODEX, home, self.loading(settings, tmp_path))

        assert settings.config_path.read_text() == "port = 9001\n"
        assert self.files_beside_the_config(settings) == ["config.toml", "config.toml.lock", "homes"]

    def test_a_config_that_goes_between_the_look_and_the_read_holds_nothing(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = prepared_codex_home(settings)
        settings.config_path.write_text("port = 9001\n")
        real_content = standing._config_content

        def content_of_a_file_just_removed(config_path: Path) -> bytes:
            config_path.unlink()
            return real_content(config_path)

        monkeypatch.setattr(standing, "_config_content", content_of_a_file_just_removed)

        assert standing.register(settings, Provider.CODEX, home, self.loading(settings, tmp_path)) is True

        written = tomllib.loads(settings.config_path.read_text())
        assert written["codex"] == [{"home": str(home), "mode": "cli"}]
        assert "port" not in written

    def test_a_config_lstat_that_reports_the_file_missing_still_registers_the_login(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = prepared_codex_home(settings)
        assert not settings.config_path.exists()
        self.fail_to_examine(
            monkeypatch, settings.config_path, FileNotFoundError(errno.ENOENT, "No such file or directory")
        )

        assert standing.register(settings, Provider.CODEX, home, self.loading(settings, tmp_path)) is True

        assert tomllib.loads(settings.config_path.read_text())["codex"] == [{"home": str(home), "mode": "cli"}]

    @pytest.mark.parametrize(
        "failure",
        [OSError(errno.EIO, "Input/output error"), PermissionError(errno.EACCES, "Permission denied")],
    )
    def test_a_sign_in_that_cannot_examine_the_config_does_not_forget_a_listed_spelling(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ) -> None:
        home = self.made_claude_home(settings)
        spelling = home.parent / home.name / ".." / home.name
        settings.config_path.parent.mkdir(parents=True, exist_ok=True)
        listed = standing_source_table("claude", spelling)
        settings.config_path.write_text(listed)
        remembered = self.loading(settings, tmp_path)(settings.config_path)
        called: list[Provider] = []
        self.fail_to_examine(monkeypatch, settings.config_path, failure)
        monkeypatch.setattr(standing, "run_login", lambda *_args: called.append(Provider.CLAUDE) or 0)

        with pytest.raises(ConfigError, match=rf"^cannot examine .*config\.toml: {type(failure).__name__}$"):
            standing.sign_in_and_register(remembered, Provider.CLAUDE, home, self.loading(settings, tmp_path))

        assert called == []
        assert settings.config_path.read_text() == listed

    def test_a_sign_in_rechecks_a_direct_mode_listing_before_the_vendor_cli(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = prepared_codex_home(settings)
        listed = f'codex = [{{ home = "{home}", mode = "direct" }}]\n'
        settings.config_path.write_text(listed)
        called: list[Provider] = []
        monkeypatch.setattr(standing, "run_login", lambda *_args: called.append(Provider.CODEX) or 0)

        with pytest.raises(ConfigError, match=r'read in direct mode; set that entry\'s mode to "cli", or remove it'):
            standing.sign_in_and_register(settings, Provider.CODEX, home, self.loading(settings, tmp_path))

        assert called == []
        assert settings.config_path.read_text() == listed
        assert self.files_beside_the_config(settings) == ["config.toml", "config.toml.lock", "homes"]

    def test_a_sign_in_refuses_an_incompatible_list_before_the_vendor_cli(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = prepared_codex_home(settings)
        settings.config_path.write_text("codex = []\n")
        called: list[Provider] = []
        monkeypatch.setattr(standing, "run_login", lambda *_args: called.append(Provider.CODEX) or 0)

        with pytest.raises(ConfigError, match=r"^cannot add this login to .*config\.toml"):
            standing.sign_in_and_register(settings, Provider.CODEX, home, self.loading(settings, tmp_path))

        assert called == []
        assert settings.config_path.read_text() == "codex = []\n"
        assert self.files_beside_the_config(settings) == ["config.toml", "config.toml.lock", "homes"]

    def fail_to_examine(self, monkeypatch: pytest.MonkeyPatch, target: Path, failure: Exception) -> None:
        real_lstat = os.lstat

        def lstat(path: os.PathLike[str] | str, *args: object, **kwargs: object) -> os.stat_result:
            if Path(path) == target:
                raise type(failure)(*failure.args)
            return real_lstat(path, *args, **kwargs)

        monkeypatch.setattr(os, "lstat", lstat)


REGISTRAR_WHILE_THE_LOCK_IS_HELD = """\
import os
import sys
from pathlib import Path

flag = Path(sys.argv[1])
data_dir = sys.argv[2]
home_dir = Path(sys.argv[3])
spelling = Path(sys.argv[4])
sys.path.insert(0, sys.argv[5])
os.environ["USAGEDASH_DATA_DIR"] = data_dir
os.environ.pop("USAGEDASH_CONFIG", None)

from usagedash import standing
from usagedash.config import load_settings
from usagedash.models import Provider

def load(config_file: Path):
    return load_settings(
        env={"USAGEDASH_DATA_DIR": data_dir, "USAGEDASH_CONFIG": str(config_file)},
        home=home_dir,
    )

settings = load_settings(env={"USAGEDASH_DATA_DIR": data_dir}, home=home_dir)
flag.write_text("at-lock")
try:
    added = standing.register(settings, Provider.CLAUDE, spelling, load)
except Exception as error:
    flag.write_text("error:" + type(error).__name__)
    raise
flag.write_text("added" if added else "already")
"""

REGISTRATION_LOCK_PROBE = """\
import fcntl
import os
import sys

descriptor = os.open(sys.argv[1], os.O_RDWR)
try:
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("blocked")
else:
    print("free")
finally:
    os.close(descriptor)
"""


class ServerThatReturnsAtOnce:
    created: list["ServerThatReturnsAtOnce"]

    def __init__(self, config: uvicorn.Config) -> None:
        self.config = config
        self.started = True
        self.created.append(self)

    def run(self) -> None:
        return None


LISTENER_ORIGIN = "http://127.0.0.1:9"


class MeasuredStream(httpx2.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.reads = 0

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._chunks:
            self.reads += 1
            yield chunk


def recording_environment(captured: dict[str, str]) -> Callable[..., subprocess.CompletedProcess[str]]:

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        env = kwargs["env"]
        assert isinstance(env, dict)
        captured.update(env)
        return subprocess.CompletedProcess(command, 0)

    return run


def recording_commands(started: list[list[str]]) -> Callable[..., subprocess.CompletedProcess[str]]:

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        started.append(command)
        return subprocess.CompletedProcess(command, 0)

    return run


def launchctl_not_loaded(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    run = FakeRun()
    run.returncodes = {"print": NOT_LOADED}
    monkeypatch.setattr(agent.subprocess, "run", run)
    return run


def launchctl_stopping_the_agent(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    run = FakeRun()
    run.sequences = {"print": [0, NOT_LOADED]}
    monkeypatch.setattr(agent.subprocess, "run", run)
    return run


def prepared_codex_home(settings: Settings) -> Path:
    home = standing.standing_home(settings, Provider.CODEX, "work")
    standing.prepare_home(settings, home)
    return home


def listen_with(
    monkeypatch: pytest.MonkeyPatch,
    answer: Callable[[httpx2.Request], tuple[int, list[bytes]]],
) -> tuple[list[MeasuredStream], dict[str, Any]]:
    streams: list[MeasuredStream] = []
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        status_code, chunks = answer(request)
        stream = MeasuredStream(chunks)
        streams.append(stream)
        return httpx2.Response(status_code, stream=stream)

    real_client = httpx2.Client

    def build(**kwargs: Any) -> httpx2.Client:
        seen.update(kwargs)
        kwargs["transport"] = httpx2.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr(cli.httpx2, "Client", build)
    return streams, seen


def free_loopback_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
    return port


def stall_before_the_final_headers(
    pieces: list[bytes], pause: float
) -> tuple[str, dict[str, int], socket.socket, threading.Thread]:
    """Listen on loopback and write ``pieces`` after the request, pausing between them.

    ``sent`` counts pieces that were written. A client that closes the socket stops the rest.
    """
    sent = {"sent": 0}
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2)
    port = listener.getsockname()[1]

    def serve() -> None:
        try:
            connection, _address = listener.accept()
        except OSError:
            return
        with connection:
            connection.settimeout(2)
            try:
                request = b""
                while b"\r\n\r\n" not in request and len(request) < 8192:
                    chunk = connection.recv(4096)
                    if chunk == b"":
                        return
                    request += chunk
                for piece in pieces:
                    time.sleep(pause)
                    connection.sendall(piece)
                    sent["sent"] += 1
            except OSError:
                return

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    return f"http://127.0.0.1:{port}/", sent, listener, server


class HealthListener:
    """A loopback server that answers ``GET`` requests as told and records the paths it was asked.

    ``answer`` receives the request path and returns the status, the body, and an optional ``Location``.
    """

    def __init__(self, answer: Callable[[str], tuple[int, bytes, str | None]]) -> None:
        self.answer = answer
        self.paths: list[str] = []
        handler = self._handler()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                owner.paths.append(self.path)
                status_code, payload, location = owner.answer(self.path)
                self.send_response(status_code)
                if location is not None:
                    self.send_header("Location", location)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _format: str, *_args: object) -> None:
                return None

        return Handler

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/"

    @property
    def port(self) -> int:
        port: int = self._server.server_address[1]
        return port

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def answers_health_checks(port: int) -> bool:
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        try:
            connection.request("GET", "/healthz")
            return connection.getresponse().status == 200
        finally:
            connection.close()
    except OSError:
        return False


class TestCommandLine:
    @pytest.fixture
    def data_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        data = tmp_path / "data"
        monkeypatch.setenv("USAGEDASH_DATA_DIR", str(data))
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("USAGEDASH_CONFIG", raising=False)
        monkeypatch.delenv("USAGEDASH_PORT", raising=False)
        # The plist and the log live under the account home, which these tests replace with tmp_path.
        monkeypatch.setattr(agent, "account_home", lambda: tmp_path)
        return data

    def installed_for(self, monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> None:
        monkeypatch.setattr(agent, "installed_environment", lambda _paths: {"USAGEDASH_DATA_DIR": str(data_dir)})

    @pytest.fixture
    def servers(self, monkeypatch: pytest.MonkeyPatch) -> list[ServerThatReturnsAtOnce]:
        created: list[ServerThatReturnsAtOnce] = []
        monkeypatch.setattr(ServerThatReturnsAtOnce, "created", created, raising=False)
        monkeypatch.setattr(cli.uvicorn, "Server", ServerThatReturnsAtOnce)
        return created

    @pytest.fixture
    def log_file(self, tmp_path: Path) -> Path:
        return tmp_path / "Library" / "Logs" / "ai-usage-dashboard" / "usagedash.log"

    @pytest.fixture
    def real_server(self, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[uvicorn.Server]]:
        created: list[uvicorn.Server] = []

        class Recorded(uvicorn.Server):
            def __init__(self, config: uvicorn.Config) -> None:
                super().__init__(config)
                created.append(self)

        monkeypatch.setenv("USAGEDASH_PORT", str(free_loopback_port()))
        monkeypatch.setattr(credentials.subprocess, "run", FakeRun(returncode=44))
        monkeypatch.setattr(cli.uvicorn, "Server", Recorded)
        handlers_before = list(logging.getLogger().handlers)
        yield created
        for handler in logging.getLogger().handlers:
            if handler not in handlers_before:
                logging.getLogger().removeHandler(handler)
                handler.close()

    @pytest.mark.parametrize("inherited", ["", "many", "9" * 5000])
    def test_serve_runs_one_worker_whatever_web_concurrency_an_unrelated_tool_left(
        self,
        data_dir: Path,
        servers: list[ServerThatReturnsAtOnce],
        monkeypatch: pytest.MonkeyPatch,
        inherited: str,
    ) -> None:
        monkeypatch.setenv("WEB_CONCURRENCY", inherited)

        assert cli.main(["serve"]) == 0

        (server,) = servers
        assert server.config.workers == 1

    def test_uvicorn_logs_through_the_dashboard_s_handlers_and_adds_none_of_its_own(
        self, data_dir: Path, servers: list[ServerThatReturnsAtOnce], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        uvicorn_logger = logging.getLogger("uvicorn")
        monkeypatch.setattr(uvicorn_logger, "handlers", [])
        monkeypatch.setattr(uvicorn_logger, "propagate", True)

        assert cli.main(["serve"]) == 0

        (server,) = servers
        assert server.config.log_config is None
        assert (uvicorn_logger.handlers, uvicorn_logger.propagate) == ([], True)

    def test_every_handler_the_dashboard_configures_reports_its_own_failures_without_messages(
        self, data_dir: Path, log_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(logging.getLogger(), "handlers", [])
        try:
            cli._configure_logging(log_to_file=True)
            configured = list(logging.getLogger().handlers)
        finally:
            for handler in logging.getLogger().handlers:
                handler.close()

        assert [type(handler) for handler in configured] == [
            cli.MessageFreeStreamHandler,
            cli.PrivateRotatingFileHandler,
        ]
        assert all(isinstance(handler, cli.MessageFreeHandlerErrors) for handler in configured)

    def test_status_prints_each_account_s_windows_and_any_source_problem(
        self, data_dir: Path, now: datetime, capsys: pytest.CaptureFixture[str]
    ) -> None:
        database = Database(data_dir / "usage.db")
        database.initialize()
        identity = make_identity(label="Personal")
        with database.connect() as conn:
            account_id = db.upsert_account(conn, identity, now)
            db.record_reading(
                conn, account_id, make_reading(identity, now, make_window(used_percent=42, label="5-hour session"))
            )
            db.set_source_state(conn, "claude:home", Provider.CLAUDE, 0, account_id, now)
            db.record_status(conn, "claude:home", db.STATUS_KIND_QUOTA, now, None)
            db.record_status(conn, "grok:home", db.STATUS_KIND_QUOTA, now, ValueError("log reshaped"))

        assert cli.main(["status"]) == 0

        output = capsys.readouterr().out
        assert "Personal [active] Max 20x" in output
        assert "5-hour session" in output
        assert "42.0% used" in output
        assert "[error] grok:home quota: log reshaped" in output
        assert "claude:home quota" not in output

    def test_agent_uninstall_names_the_data_directory_the_agent_was_installed_for(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        removed: list[agent.AgentPaths] = []

        def uninstall(paths: agent.AgentPaths) -> dict[str, str]:
            removed.append(paths)
            return {"USAGEDASH_DATA_DIR": str(tmp_path / "agent-data")}

        monkeypatch.setattr(agent, "uninstall", uninstall)

        assert cli.main(["agent", "uninstall"]) == 0

        assert len(removed) == 1
        output = capsys.readouterr().out
        assert f"data in {tmp_path / 'agent-data'} was kept" in output
        assert str(data_dir) not in output

    def test_agent_uninstall_leaves_the_data_directory_unnamed_when_the_property_list_did_not_say(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        removed: list[agent.AgentPaths] = []
        monkeypatch.setattr(agent, "uninstall", removed.append)

        assert cli.main(["agent", "uninstall"]) == 0

        assert len(removed) == 1
        output = capsys.readouterr().out
        assert output.strip().endswith("its data was kept")
        assert str(data_dir) not in output

    def test_the_package_can_be_run_as_a_module(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: False)
        monkeypatch.setattr(sys, "argv", ["usagedash", "agent", "status"])

        with pytest.raises(SystemExit) as exited:
            runpy.run_module("usagedash", run_name="__main__")

        assert exited.value.code == 0
        assert capsys.readouterr().out.strip().endswith("not loaded")

    def test_collect_runs_one_cycle_and_reports_each_source_even_when_nobody_is_logged_in(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(credentials.subprocess, "run", FakeRun(returncode=44))

        assert cli.main(["collect"]) == 0

        output = capsys.readouterr().out
        assert output.count("account: not logged in") == 3
        assert (
            "CredentialUnavailableError: no keychain item 'Claude Code-credentials' under this macOS account's name"
            in output
        )
        assert f"'{keychain_account()}'" not in output
        assert "CredentialUnavailableError: Grok is not signed in" in output
        assert output.count("tokens:  ok (0 file(s) read)") == 3
        assert (data_dir / "usage.db").exists()

    def test_collect_names_the_account_when_the_quota_read_fails_after_the_login_is_known(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:

        class SignedIn:
            source_id = "codex:home"
            provider = Provider.CODEX

            def current_identity(self) -> AccountIdentity:
                return make_identity(Provider.CODEX, "workspace-a", "Account A")

            def read_quota(self, now: datetime) -> QuotaReading:
                raise ProviderRequestError("backend unavailable")

        def build(settings: Settings, database: Database, _client: httpx2.Client, clock: Clock) -> Collector:
            return Collector(
                database,
                (SourceBundle(SignedIn(), None, 0),),
                clock,
                poll_interval_seconds=settings.poll_interval_seconds,
                backfill_days=settings.backfill_days,
            )

        monkeypatch.setattr(cli, "build_collector", build)

        assert cli.main(["collect"]) == 1

        output = capsys.readouterr().out
        assert "account: Account A" in output
        assert "account: not logged in" not in output
        assert "quota:   ProviderRequestError: backend unavailable" in output

    def test_an_invalid_setting_is_reported_and_exits_nonzero(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("USAGEDASH_PORT", "80")

        assert cli.main(["status"]) == 1
        assert "error: port must be between" in capsys.readouterr().err

    def test_login_registers_the_standing_home_and_runs_the_vendor_login_against_it(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        record = tmp_path / "record"
        (data_dir).mkdir()
        (data_dir / "config.toml").write_text(f'[binaries]\ngrok = "{sys.executable}"\n')
        monkeypatch.setenv("FAKE_SCENARIO", "exit_immediately")
        monkeypatch.setenv("FAKE_RECORD", str(record))
        monkeypatch.setattr(
            standing, "_LOGIN_ARGUMENTS", {**standing._LOGIN_ARGUMENTS, Provider.GROK: (str(FAKE_CLI),)}
        )
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: False)

        assert cli.main(["login", "grok", "--name", "personal"]) == 0

        standing_home = data_dir / "homes" / "grok-personal"
        assert json.loads(record.read_text().splitlines()[0])["env"]["GROK_HOME"] == str(standing_home)
        assert tomllib.loads((data_dir / "config.toml").read_text())["grok"] == [
            {"home": str(standing_home), "mode": "cli"}
        ]
        assert "usagedash agent install" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "defect",
        [
            "a directory others can write",
            "a file others can write",
            "a missing file anyone could put there",
            "a path through a file",
        ],
    )
    def test_agent_install_refuses_a_pinned_cli_another_account_could_change(
        self,
        data_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        defect: str,
    ) -> None:
        shared = tmp_path / "shared-cli"
        shared.mkdir()
        grok = shared / "grok"
        grok.write_text("#!/bin/sh\n")
        grok.chmod(0o755)
        if defect == "a directory others can write":
            shared.chmod(0o777)
            expected = f"cannot use {shared}: its permissions (0777) let other accounts write"
        elif defect == "a file others can write":
            grok.chmod(0o777)
            expected = f"cannot use {grok}: its permissions (0777) let other accounts write"
        elif defect == "a missing file anyone could put there":
            sticky = Path(os.sep, "tmp")
            grok = sticky / f"usagedash-not-there-{os.getpid()}"
            expected = f"cannot use {sticky.resolve()}: its permissions (1777) let other accounts write"
        else:
            grok = grok / "grok"
            expected = f"cannot use {grok}: NotADirectoryError"
        monkeypatch.setenv("USAGEDASH_GROK_BINARY", str(grok))
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: False)
        installed: list[object] = []
        monkeypatch.setattr(agent, "install", lambda *arguments: installed.append(arguments))
        try:
            assert cli.main(["agent", "install"]) == 1
        finally:
            shared.chmod(0o700)

        assert expected in capsys.readouterr().err
        assert installed == []

    def test_a_pinned_native_binary_need_not_be_readable(self, tmp_path: Path) -> None:
        binary = tmp_path / "native-cli"
        binary.touch()
        binary.chmod(0o111)
        assert cli._canonical_executable_only_this_account_can_change(binary) == binary

    @pytest.mark.parametrize("selection", ["absolute", "relative", "PATH", "relative PATH", "empty PATH"])
    @pytest.mark.parametrize("colon", [False, True])
    def test_child_lookup_uses_its_working_directory_and_pins_the_checked_target(
        self,
        tmp_path: Path,
        selection: str,
        colon: bool,
    ) -> None:
        if colon:
            tmp_path = tmp_path / "child:directory"
            tmp_path.mkdir()
        binary = tmp_path / "vendor"
        binary.touch(mode=0o755)
        link = tmp_path / "alias"
        link.symlink_to(binary)
        command = str(link) if selection == "absolute" else "./alias" if selection == "relative" else "alias"
        search = str(tmp_path) if selection == "PATH" else "." if selection == "relative PATH" else ""
        if colon and selection == "PATH":
            with pytest.raises(FileNotFoundError):
                executables.checked_command([command], {"PATH": search}, tmp_path)
            return
        assert executables.checked_command([command, "login"], {"PATH": search}, tmp_path) == [str(binary), "login"]

    @pytest.mark.parametrize("search", ["", "."])
    def test_child_lookup_never_splits_the_expanded_working_directory(self, tmp_path: Path, search: str) -> None:
        (tmp_path / "vendor").touch(mode=0o755)
        cwd = Path(f"{tmp_path}/empty:{tmp_path}")
        with pytest.raises(FileNotFoundError):
            executables.checked_command(["vendor"], {"PATH": search}, cwd)

    @pytest.mark.parametrize("directory", [False, True])
    def test_child_lookup_skips_non_executable_candidates(self, tmp_path: Path, directory: bool) -> None:
        first, second = tmp_path / "first", tmp_path / "second"
        first.mkdir()
        second.mkdir()
        if directory:
            (first / "vendor").mkdir()
        else:
            (first / "vendor").touch(mode=0o600)
        binary = second / "vendor"
        binary.touch(mode=0o755)
        assert executables.checked_command(["vendor"], {"PATH": f"{first}:{second}"}, tmp_path) == [str(binary)]

    def test_child_lookup_with_no_path_uses_the_system_default(self, tmp_path: Path) -> None:
        assert executables.checked_command(["sh"], {}, tmp_path) == [str(Path("/bin/sh").resolve())]

    def test_child_lookup_fails_closed_for_missing_or_empty_commands(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            executables.checked_command(["missing"], {"PATH": str(tmp_path)}, tmp_path)
        with pytest.raises(ConfigError, match="empty command"):
            executables.checked_command([], {}, tmp_path)

    def test_a_missing_pinned_executable_uses_the_checked_path_replacement(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pinned = tmp_path / "Caskroom" / "codex" / "0.161.0" / "bin" / "codex"
        skipped, current_dir = tmp_path / "skipped", tmp_path / "current"
        skipped.mkdir()
        current_dir.mkdir()
        (skipped / "codex").touch()
        (skipped / "codex").chmod(0o644)
        current = current_dir / "codex"
        current.touch()
        current.chmod(0o755)
        with caplog.at_level(logging.WARNING, logger="usagedash.executables"):
            resolved = executables.checked_command(
                [str(pinned), "app-server"], {"PATH": f"{skipped}:{current_dir}"}, tmp_path
            )
            again = executables.checked_command([str(pinned)], {"PATH": str(current_dir)}, tmp_path)
        assert resolved == [str(current.resolve()), "app-server"]
        assert again == [str(current.resolve())]
        replacements = [record.getMessage() for record in caplog.records if "pinned executable" in record.getMessage()]
        assert replacements == [f"pinned executable {pinned} is missing; using {current}"]

    def test_a_missing_relative_command_uses_its_file_name_on_the_checked_path(self, tmp_path: Path) -> None:
        current_dir = tmp_path / "current"
        current_dir.mkdir()
        current = current_dir / "codex"
        current.touch()
        current.chmod(0o755)
        work = tmp_path / "work"
        work.mkdir()
        resolved = executables.checked_command(["vendor/codex"], {"PATH": str(current_dir)}, work)
        assert resolved == [str(current.resolve())]

    def test_a_present_pin_is_not_replaced_by_another_binary_on_path(self, tmp_path: Path) -> None:
        pinned_dir, other_dir = tmp_path / "pinned", tmp_path / "other"
        pinned_dir.mkdir()
        other_dir.mkdir()
        pinned, other = pinned_dir / "codex", other_dir / "codex"
        pinned.touch()
        other.touch()
        pinned.chmod(0o755)
        other.chmod(0o755)
        assert executables.checked_command([str(pinned)], {"PATH": str(other_dir)}, tmp_path) == [str(pinned.resolve())]

    def test_a_dangling_pin_is_not_replaced_from_the_path(self, tmp_path: Path) -> None:
        link = tmp_path / "codex"
        link.symlink_to(tmp_path / "missing-target")
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        other = other_dir / "codex"
        other.touch()
        other.chmod(0o755)
        assert executables.checked_command([str(link)], {"PATH": str(other_dir)}, tmp_path) == [
            str((tmp_path / "missing-target").resolve())
        ]

    def test_a_missing_pin_with_no_replacement_names_the_pin_and_the_name(self, tmp_path: Path) -> None:
        pinned = tmp_path / "gone" / "codex"
        with pytest.raises(
            ConfigError,
            match=rf"cannot use {re.escape(str(pinned))}: FileNotFoundError, and codex is not on the checked PATH$",
        ):
            executables.checked_command([str(pinned)], {"PATH": str(tmp_path)}, tmp_path)

    def test_a_missing_versioned_pin_uses_the_vendor_command(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pinned = tmp_path / "versions" / "2.1.296"
        current_dir = tmp_path / "bin"
        current_dir.mkdir()
        current = current_dir / "claude"
        current.touch()
        current.chmod(0o755)
        with caplog.at_level(logging.WARNING, logger="usagedash.executables"):
            resolved = executables.checked_command(
                [str(pinned), "-p"],
                {"PATH": str(current_dir)},
                tmp_path,
                fallback_name="claude",
            )
        assert resolved == [str(current.resolve()), "-p"]
        replacements = [record.getMessage() for record in caplog.records if "pinned executable" in record.getMessage()]
        assert replacements == [f"pinned executable {pinned} is missing; using {current}"]

    def test_a_missing_pin_prefers_its_file_name_over_the_vendor_command(self, tmp_path: Path) -> None:
        pinned = tmp_path / "versions" / "2.1.296"
        named, command = tmp_path / "named", tmp_path / "command"
        named.mkdir()
        command.mkdir()
        by_name, by_command = named / "2.1.296", command / "claude"
        by_name.touch()
        by_command.touch()
        by_name.chmod(0o755)
        by_command.chmod(0o755)
        resolved = executables.checked_command(
            [str(pinned)],
            {"PATH": f"{named}:{command}"},
            tmp_path,
            fallback_name="claude",
        )
        assert resolved == [str(by_name.resolve())]

    def test_a_present_pin_ignores_the_vendor_command(self, tmp_path: Path) -> None:
        pinned_dir, other_dir = tmp_path / "pinned", tmp_path / "other"
        pinned_dir.mkdir()
        other_dir.mkdir()
        pinned, other = pinned_dir / "2.1.296", other_dir / "claude"
        pinned.touch()
        other.touch()
        pinned.chmod(0o755)
        other.chmod(0o755)
        assert executables.checked_command(
            [str(pinned)], {"PATH": str(other_dir)}, tmp_path, fallback_name="claude"
        ) == [str(pinned.resolve())]

    def test_a_missing_versioned_pin_names_both_absent_commands(self, tmp_path: Path) -> None:
        pinned = tmp_path / "downloads" / "grok-1.0.50-macos-aarch64"
        with pytest.raises(
            ConfigError,
            match=(
                rf"cannot use {re.escape(str(pinned))}: FileNotFoundError, "
                r"and neither grok-1.0.50-macos-aarch64 nor grok is on the checked PATH$"
            ),
        ):
            executables.checked_command([str(pinned)], {"PATH": str(tmp_path)}, tmp_path, fallback_name="grok")

    def test_a_fallback_equal_to_the_file_name_is_named_once(self, tmp_path: Path) -> None:
        pinned = tmp_path / "gone" / "codex"
        with pytest.raises(
            ConfigError,
            match=rf"cannot use {re.escape(str(pinned))}: FileNotFoundError, and codex is not on the checked PATH$",
        ):
            executables.checked_command([str(pinned)], {"PATH": str(tmp_path)}, tmp_path, fallback_name="codex")

    @pytest.mark.parametrize("name", ["", ".", "..", "a/b", "cla:ude"])
    def test_a_replacement_command_must_be_one_file_name(self, tmp_path: Path, name: str) -> None:
        pinned_dir = tmp_path / "pinned"
        pinned_dir.mkdir()
        pinned = pinned_dir / "claude"
        pinned.touch()
        pinned.chmod(0o755)
        with pytest.raises(ConfigError, match="one file name"):
            executables.checked_command([str(pinned)], {"PATH": str(tmp_path)}, tmp_path, fallback_name=name)

    def test_a_missing_versioned_pin_refuses_an_unsafe_vendor_command(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pinned = tmp_path / "versions" / "2.1.296"
        unsafe = tmp_path / "unsafe"
        unsafe.mkdir()
        replacement = unsafe / "claude"
        replacement.touch()
        replacement.chmod(0o755)
        unsafe.chmod(0o777)
        with (
            caplog.at_level(logging.WARNING, logger="usagedash.executables"),
            pytest.raises(ConfigError, match=r"permissions \(0777\)"),
        ):
            executables.checked_command([str(pinned)], {"PATH": str(unsafe)}, tmp_path, fallback_name="claude")
        assert [record for record in caplog.records if "pinned executable" in record.getMessage()] == []

    def test_a_removed_version_file_starts_the_vendor_command(self, tmp_path: Path) -> None:
        pinned = tmp_path / "versions" / "2.1.296"
        bindir = tmp_path / "bin"
        bindir.mkdir()
        binary = bindir / "claude"
        binary.write_text("#!/bin/sh\nprintf '%s\\n' '{\"ok\":true}'\n")
        binary.chmod(0o755)
        with rpc.JsonLineProcess(
            [str(pinned)],
            env={"PATH": str(bindir)},
            cwd=tmp_path,
            service="Claude Code",
            fallback_name="claude",
        ) as child:
            assert child.receive(lambda _message: True, time.monotonic() + 5) == {"ok": True}

    @pytest.mark.parametrize("provider", list(Provider))
    def test_a_quota_read_names_its_vendor_command_when_the_pin_is_gone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: Provider
    ) -> None:
        module = {Provider.CLAUDE: claude_cli, Provider.CODEX: codex_cli, Provider.GROK: grok_cli}[provider]
        captured: dict[str, object] = {}

        def refuse(command: list[str], **kwargs: object) -> None:
            captured["command"] = command
            captured["fallback_name"] = kwargs["fallback_name"]
            raise ConfigError("stopped")

        monkeypatch.setattr(module, "JsonLineProcess", refuse)
        pinned = str(tmp_path / "versions" / "removed")
        if provider is Provider.CLAUDE:
            source = claude_cli.ClaudeCliSource(
                ClaudeSourceConfig(tmp_path, tmp_path / ".claude.json", SourceMode.CLI), [pinned]
            )
            monkeypatch.setattr(
                source, "_signed_in_identity", lambda: AccountIdentity(Provider.CLAUDE, "account", "Claude")
            )
        elif provider is Provider.CODEX:
            source = codex_cli.CodexCliSource(tmp_path, [pinned])
        else:
            source = grok_cli.GrokCliSource(tmp_path, [pinned])
            monkeypatch.setattr(source, "_signed_in_login", lambda: object())
        with pytest.raises(ConfigError, match="stopped"):
            source.read_quota(datetime.now(UTC))
        assert captured == {"command": [pinned, *module.CLI_ARGUMENTS], "fallback_name": provider.value}

    def test_login_starts_the_vendor_command_when_the_recorded_file_is_gone(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = tmp_path / "record"
        bindir = tmp_path / "bin"
        bindir.mkdir()
        binary = bindir / "grok"
        binary.write_text(f"#!/bin/sh\nprintf 'ran' > '{record}'\n")
        binary.chmod(0o755)
        pinned = tmp_path / "downloads" / "grok-1.0.50-macos-aarch64"
        monkeypatch.setenv("PATH", str(bindir))
        settings = load_settings(
            env={"USAGEDASH_DATA_DIR": str(data_dir), "USAGEDASH_GROK_BINARY": str(pinned)},
            home=tmp_path,
        )
        assert standing.run_login(settings, Provider.GROK, tmp_path) == 0
        assert record.read_text() == "ran"

    def test_a_missing_pin_still_refuses_an_unsafe_replacement(self, tmp_path: Path) -> None:
        pinned = tmp_path / "gone" / "codex"
        unsafe = tmp_path / "unsafe"
        unsafe.mkdir()
        replacement = unsafe / "codex"
        replacement.touch()
        replacement.chmod(0o755)
        unsafe.chmod(0o777)
        with pytest.raises(ConfigError, match=r"permissions \(0777\)"):
            executables.checked_command([str(pinned)], {"PATH": str(unsafe)}, tmp_path)

    def test_a_rejected_replacement_does_not_consume_the_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pinned = tmp_path / "gone" / "codex"
        unsafe, safe = tmp_path / "unsafe", tmp_path / "safe"
        unsafe.mkdir()
        safe.mkdir()
        rejected, accepted = unsafe / "codex", safe / "codex"
        rejected.touch()
        accepted.touch()
        rejected.chmod(0o755)
        accepted.chmod(0o755)
        unsafe.chmod(0o777)
        with caplog.at_level(logging.WARNING, logger="usagedash.executables"):
            with pytest.raises(ConfigError, match=r"permissions \(0777\)"):
                executables.checked_command([str(pinned)], {"PATH": str(unsafe)}, tmp_path)
            resolved = executables.checked_command([str(pinned)], {"PATH": str(safe)}, tmp_path)
            again = executables.checked_command([str(pinned)], {"PATH": str(safe)}, tmp_path)
        assert resolved == [str(accepted.resolve())]
        assert again == [str(accepted.resolve())]
        replacements = [record.getMessage() for record in caplog.records if "pinned executable" in record.getMessage()]
        assert replacements == [f"pinned executable {pinned} is missing; using {accepted}"]

    def test_an_unsafe_pin_is_not_replaced_from_the_path(self, tmp_path: Path) -> None:
        unsafe, safe = tmp_path / "unsafe", tmp_path / "safe"
        unsafe.mkdir()
        safe.mkdir()
        pinned = unsafe / "codex"
        pinned.touch()
        pinned.chmod(0o755)
        safe_binary = safe / "codex"
        safe_binary.touch()
        safe_binary.chmod(0o755)
        unsafe.chmod(0o777)
        with pytest.raises(ConfigError, match=r"permissions \(0777\)"):
            executables.checked_command([str(pinned)], {"PATH": str(safe)}, tmp_path)

    def test_an_omitted_child_path_is_logged_once_per_process(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        missing, other = tmp_path / "missing-bin", tmp_path / "other-missing"
        with caplog.at_level(logging.WARNING, logger="usagedash.executables"):
            first = executables.execution_environment({"PATH": str(missing), "KEEP": "1"}, tmp_path)
            second = executables.execution_environment({"PATH": f"{missing}{os.pathsep}{other}", "KEEP": "1"}, tmp_path)
        assert first == {"PATH": os.devnull, "KEEP": "1"}
        assert second == {"PATH": os.devnull, "KEEP": "1"}
        omissions = [record.getMessage() for record in caplog.records if record.getMessage().startswith("left out")]
        assert omissions.count(next(message for message in omissions if str(missing) in message)) == 1
        assert sum(str(other) in message for message in omissions) == 1

    @pytest.mark.parametrize("kind", ["relative", "colon", "unsafe"])
    def test_child_path_is_canonical_deduplicated_and_never_an_implicit_cwd(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        kind: str,
    ) -> None:
        cwd = tmp_path / ("child:colon" if kind == "colon" else "child")
        cwd.mkdir()
        if kind == "unsafe":
            cwd.chmod(0o777)
        monkeypatch.setattr(executables.sys, "platform", "linux")
        helper_directory = tmp_path / "unused"
        base = {"PATH": ".:.", "KEEP": "value"}
        result = executables.execution_environment(base, cwd, helper_directory=helper_directory)
        assert result == {"PATH": str(cwd) if kind == "relative" else os.devnull, "KEEP": "value"}
        assert base["PATH"] == ".:."
        assert not helper_directory.exists()

    @pytest.mark.parametrize("login", [False, True])
    @pytest.mark.parametrize("linked", [False, True])
    @pytest.mark.parametrize("available", [False, True])
    def test_vendor_shebang_and_helpers_use_only_checked_path_entries(
        self,
        data_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        login: bool,
        linked: bool,
        available: bool,
    ) -> None:
        record = tmp_path / "record"
        shared, unsafe, safe = (tmp_path / name for name in ("shared", "unsafe", "safe"))
        for directory in (shared, unsafe, safe):
            directory.mkdir()
        for directory, value in ((shared, "unsafe"), (safe, "trusted")):
            node = directory / "node"
            node.write_text(f"#!/bin/sh\nprintf '{value}' > '{record}'\nprintf '{{\"ok\":true}}\\n'\n")
            node.chmod(0o755)
        shared.chmod(0o777)
        if linked:
            (unsafe / "node").symlink_to(shared / "node")
        else:
            unsafe = shared
        vendor = tmp_path / "vendor"
        vendor.write_text("#!/usr/bin/env node\n")
        vendor.chmod(0o755)
        search_path = f"{unsafe}:{safe}" if available else str(unsafe)
        monkeypatch.setenv("PATH", search_path)
        settings = load_settings(
            env={"USAGEDASH_DATA_DIR": str(data_dir), "USAGEDASH_CODEX_BINARY": str(vendor)},
            home=tmp_path,
        )
        if login:
            assert (standing.run_login(settings, Provider.CODEX, tmp_path) == 0) == available
        else:
            with rpc.JsonLineProcess(
                [str(vendor)], env={"PATH": search_path}, cwd=tmp_path, service="Synthetic"
            ) as child:
                if available:
                    assert child.receive(lambda _message: True, time.monotonic() + 5) == {"ok": True}
                else:
                    with pytest.raises(ProviderRequestError, match="exited before answering"):
                        child.receive(lambda _message: True, time.monotonic() + 5)
        assert record.read_text() == "trusted" if available else not record.exists()

    @pytest.mark.parametrize("login", [False, True])
    @pytest.mark.parametrize("neighbor", [False, True])
    @pytest.mark.parametrize("available", [False, True])
    def test_bare_vendor_lookup_uses_the_filtered_child_path(
        self,
        data_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        login: bool,
        neighbor: bool,
        available: bool,
    ) -> None:
        record = tmp_path / "record"
        first, second = tmp_path / "first", tmp_path / "second"
        for directory, value in ((first, "pruned"), (second, "trusted")):
            directory.mkdir()
            binary = directory / "codex"
            binary.write_text(f"#!/bin/sh\nprintf '{value}' > '{record}'\nprintf '{{\"ok\":true}}\\n'\n")
            binary.chmod(0o755)
        unsafe = first / ("neighbor" if neighbor else "codex")
        unsafe.touch()
        unsafe.chmod(0o777)
        search = f"{first}:{second}" if available else str(first)
        monkeypatch.setenv("PATH", search)
        settings = load_settings(
            env={"USAGEDASH_DATA_DIR": str(data_dir), "USAGEDASH_CODEX_BINARY": "codex"},
            home=tmp_path,
        )

        def run() -> None:
            if login:
                assert standing.run_login(settings, Provider.CODEX, tmp_path) == 0
            else:
                with rpc.JsonLineProcess(["codex"], env={"PATH": search}, cwd=tmp_path, service="Synthetic") as child:
                    assert child.receive(lambda _message: True, time.monotonic() + 5) == {"ok": True}

        if available:
            run()
            assert record.read_text() == "trusted"
        else:
            with pytest.raises(ConfigError if login else ProviderRequestError, match="FileNotFoundError"):
                run()
            assert not record.exists()

    @pytest.mark.parametrize("command", ["collect", "serve"])
    @pytest.mark.parametrize("source", ["logged-out", "claude", "codex", "grok", "direct-grok"])
    def test_helper_failure_is_isolated_to_the_source_that_launches_a_child(
        self,
        data_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        servers: list[ServerThatReturnsAtOnce],
        now: datetime,
        command: str,
        source: str,
    ) -> None:
        fallback = data_dir / "agent-bin"
        fallback.mkdir(parents=True)
        fallback.chmod(0o777)
        path = tmp_path / "bin"
        path.mkdir()
        monkeypatch.setenv("PATH", str(path))
        monkeypatch.setattr(cli, "utc_now", lambda: now)
        for provider, source_class in ((Provider.CLAUDE, wiring.ClaudeSource), (Provider.CODEX, wiring.CodexSource)):
            monkeypatch.setattr(
                source_class,
                "read_quota",
                lambda self, at, tool=provider: make_reading(make_identity(tool), at, make_window()),
            )
        home = tmp_path / (".grok" if source == "direct-grok" else "standing")
        home.mkdir()
        if source not in {"logged-out", "direct-grok"}:
            (data_dir / "config.toml").write_text(standing_source_table(source, home))
        if source == "claude":
            write_claude_account_file(home / ".claude.json", "synthetic-org", "Synthetic Co")
        elif source in {"grok", "direct-grok"}:
            grok_auth_file(home / "auth.json", create_time=now, expires_at=now + timedelta(days=1))

        def forbidden(*args: object, **kwargs: object) -> None:
            pytest.fail("a child started despite the unsafe helper directory")

        original_app = cli.create_app

        def collecting_app(
            settings: Settings, database: Database, collector: Collector, *args: object, **kwargs: object
        ) -> FastAPI:
            collector.run_cycle()
            return original_app(settings, database, collector, *args, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", forbidden)
        monkeypatch.setattr(cli, "create_app", collecting_app)
        assert cli.main([command]) == (1 if command == "collect" and source != "logged-out" else 0)
        assert len(servers) == (command == "serve")
        with Database(data_dir / "usage.db").connect() as connection:
            outcomes = dict(
                connection.execute(
                    "SELECT source_id, error_type || ': ' || error_message FROM source_status WHERE kind = 'quota'"
                )
            )
        assert outcomes[f"claude:{tmp_path / '.claude'}"] is None
        assert outcomes[f"codex:{tmp_path / '.codex'}"] is None
        target = f"grok:{tmp_path / '.grok'}" if source == "logged-out" else f"{source.removeprefix('direct-')}:{home}"
        assert ("CredentialUnavailableError" if source == "logged-out" else "agent-bin") in outcomes[target]
        assert stat.S_IMODE(fallback.stat().st_mode) == 0o777

    @pytest.mark.parametrize("command", ["login", "collect", "serve"])
    @pytest.mark.parametrize("selection", ["configured", "PATH", "bare"])
    @pytest.mark.parametrize("alias", [False, True])
    def test_direct_commands_refuse_unsafe_vendor_executables(
        self,
        data_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        servers: list[ServerThatReturnsAtOnce],
        command: str,
        selection: str,
        alias: bool,
    ) -> None:
        shared = tmp_path / "shared"
        shared.mkdir()
        binary = shared / "codex"
        if alias:
            binary.symlink_to("/usr/bin/env")
        else:
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
        shared.chmod(0o777)
        data_dir.mkdir()
        home = tmp_path / "standing"
        home.mkdir()
        config = data_dir / "config.toml"
        config.write_text(standing_source_table("codex", home))
        monkeypatch.setenv("PATH", str(shared))
        monkeypatch.delenv("USAGEDASH_CODEX_BINARY", raising=False)
        if selection == "configured":
            config.write_text(config.read_text() + f'\n[binaries]\ncodex = "{binary}"\n')
        elif selection == "bare":
            monkeypatch.setenv("USAGEDASH_CODEX_BINARY", "codex")

        def forbidden(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            if command[0] in {str(binary), "codex", "/usr/bin/env"}:
                pytest.fail("unsafe executable reached subprocess creation")
            return subprocess.CompletedProcess(command, 1, "", "")

        def collecting_app(
            settings: Settings, database: Database, collector: Collector, *args: object, **kwargs: object
        ) -> FastAPI:
            collector.run_cycle()
            return FastAPI()

        monkeypatch.setattr(subprocess, "Popen", forbidden)
        monkeypatch.setattr(subprocess, "run", forbidden)
        monkeypatch.setattr(cli, "create_app", collecting_app)
        try:
            assert cli.main([command, "codex"] if command == "login" else [command]) == (0 if command == "serve" else 1)
        finally:
            shared.chmod(0o700)
        if command != "login":
            database = Database(data_dir / "usage.db")
            with database.connect() as connection:
                messages = connection.execute("SELECT error_message FROM source_status").fetchall()
            expected = "FileNotFoundError" if selection == "bare" else "permissions (0777)"
            assert any(expected in (row[0] or "") for row in messages)

    @pytest.mark.parametrize("intermediate", [False, True])
    def test_vendor_selection_refuses_foreign_owned_links_to_trusted_dispatchers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        intermediate: bool,
    ) -> None:
        alias = tmp_path / "alias"
        alias.symlink_to("/usr/bin/env")
        selected = tmp_path / "codex"
        selected.symlink_to(alias)
        untrusted = alias if intermediate else selected
        original = os.lstat

        def foreign(path: str | os.PathLike[str], *, dir_fd: int | None = None) -> os.stat_result:
            status = original(path, dir_fd=dir_fd)
            if Path(path) == untrusted:
                values = list(status)
                values[4] = os.geteuid() + 1
                return os.stat_result(values)
            return status

        monkeypatch.setattr(os, "lstat", foreign)
        with pytest.raises(ConfigError, match="another account"):
            executables.canonical_executable(selected)

    def test_agent_install_pins_the_file_a_link_names_not_the_link(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = tmp_path / "lib" / "claude.js"
        real.parent.mkdir()
        real.write_text("#!/bin/sh\n")
        real.chmod(0o755)
        link = tmp_path / "bin" / "claude"
        link.parent.mkdir()
        link.symlink_to(real)
        monkeypatch.setenv("USAGEDASH_CLAUDE_BINARY", str(link))
        monkeypatch.delenv("USAGEDASH_GROK_BINARY", raising=False)
        real_which = shutil.which

        def without_grok(command: str, *args: Any, **kwargs: Any) -> str | None:
            return None if command == "grok" else real_which(command, *args, **kwargs)

        monkeypatch.setattr(shutil, "which", without_grok)
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: False)
        installed: list[tuple[object, ...]] = []
        monkeypatch.setattr(agent, "install", lambda *arguments: installed.append(arguments))

        assert cli.main(["agent", "install"]) == 0

        assert len(installed) == 1
        overrides = installed[0][3]
        assert isinstance(overrides, dict)
        assert overrides["USAGEDASH_CLAUDE_BINARY"] == str(real)
        assert str(link) not in installed[0][2]
        assert overrides["USAGEDASH_GROK_BINARY"] == "grok"
        assert "grok" in installed[0][2]
        late_cli = tmp_path / ".grok" / "bin" / "grok"
        late_cli.parent.mkdir(parents=True)
        late_cli.touch()
        late_cli.parent.chmod(0o777)
        assert load_settings(env=overrides, home=tmp_path).binaries.grok == "grok"

    def test_agent_install_pins_a_cli_in_a_directory_only_the_administrators_group_can_write(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        administrators = private.ADMINISTRATORS_GROUP_ID
        if administrators is None or administrators not in os.getgroups():
            pytest.skip("this account is not in an administrators group")
        prefix = tmp_path / "homebrew" / "bin"
        prefix.mkdir(parents=True)
        grok = prefix / "grok"
        grok.write_text("#!/bin/sh\n")
        grok.chmod(0o755)
        for directory in (prefix.parent, prefix):
            os.chown(directory, -1, administrators)
            directory.chmod(0o775)
        monkeypatch.setenv("USAGEDASH_GROK_BINARY", str(grok))
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: False)
        installed: list[tuple[object, ...]] = []
        monkeypatch.setattr(agent, "install", lambda *arguments: installed.append(arguments))

        assert cli.main(["agent", "install"]) == 0

        overrides = installed[0][3]
        assert isinstance(overrides, dict)
        assert overrides["USAGEDASH_GROK_BINARY"] == str(grok)

    def test_a_group_writable_config_is_refused_before_its_binary_runs_and_left_writable(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data_dir.mkdir()
        data_dir.chmod(0o755)
        planted = tmp_path / "planted"
        config = data_dir / "config.toml"
        config.write_text(f'[binaries]\ngrok = "{planted}"\n')
        config.chmod(0o664)
        monkeypatch.setenv("USAGEDASH_GROK_BINARY", str(tmp_path / "from-the-environment"))
        run = FakeRun()
        monkeypatch.setattr(standing.subprocess, "run", run)

        assert cli.main(["login", "grok", "--name", "personal"]) == 1

        error = capsys.readouterr().err
        assert "0664" in error
        assert "let other accounts write" in error
        assert run.commands == []
        assert config.stat().st_mode & 0o777 == 0o664
        assert config.read_text() == f'[binaries]\ngrok = "{planted}"\n'
        assert not (data_dir / "homes").exists()

    @pytest.mark.skipif(sys.platform != "darwin", reason="access-control lists of this kind are a macOS feature")
    def test_a_config_an_access_control_entry_lets_others_change_is_refused_before_its_binary_runs(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data_dir.mkdir()
        data_dir.chmod(0o755)
        planted = tmp_path / "planted"
        config = data_dir / "config.toml"
        config.write_text(f'[binaries]\ngrok = "{planted}"\n')
        config.chmod(0o644)
        subprocess.run(["/bin/chmod", "+a", "everyone allow write", str(config)], check=True)
        granted = access_control_entries(config)
        # Why a saved reference: replacing standing.subprocess.run replaces subprocess.run itself.
        list_entries = subprocess.run
        run = FakeRun()
        monkeypatch.setattr(standing.subprocess, "run", run)

        assert cli.main(["login", "grok", "--name", "personal"]) == 1

        error = capsys.readouterr().err
        listed = list_entries(["/bin/ls", "-lde", str(config)], capture_output=True, text=True, check=True).stdout
        assert "an access-control entry lets another account change it" in error
        assert run.commands == []
        assert [line.strip() for line in listed.splitlines()[1:]] == granted
        assert config.stat().st_mode & 0o777 == 0o644
        assert not (data_dir / "homes").exists()

    @pytest.mark.parametrize("argv", [["serve"], ["collect"], ["agent", "install"]])
    def test_a_config_directory_others_can_write_is_refused_before_a_configured_program_runs(
        self,
        data_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        argv: list[str],
    ) -> None:
        chosen = tmp_path / "chosen"
        chosen.mkdir()
        chosen.chmod(0o777)
        planted = tmp_path / "planted"
        config = chosen / "config.toml"
        config.write_text(f'[binaries]\ngrok = "{planted}"\n')
        monkeypatch.setenv("USAGEDASH_CONFIG", str(config))
        started: list[str] = []
        monkeypatch.setattr(standing.subprocess, "run", lambda *_args, **_options: started.append("login"))
        monkeypatch.setattr(credentials.subprocess, "run", lambda *_args, **_options: started.append("credential"))
        monkeypatch.setattr(agent, "install", lambda *_args, **_options: started.append("install"))
        monkeypatch.setattr(cli.uvicorn, "Server", lambda *_args, **_options: started.append("serve"))
        try:
            assert cli.main(argv) == 1
            assert f"cannot use {chosen}: its permissions (0777)" in capsys.readouterr().err
            assert chosen.stat().st_mode & 0o777 == 0o777
            assert config.read_text() == f'[binaries]\ngrok = "{planted}"\n'
            assert started == []
            assert not data_dir.exists()
        finally:
            chosen.chmod(0o700)

    @pytest.mark.parametrize(
        "argv",
        [["serve"], ["collect"], ["status"], ["login", "grok"], ["agent", "install"]],
    )
    def test_a_safe_config_does_not_cover_a_data_directory_another_account_can_replace(
        self,
        data_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        servers: list[ServerThatReturnsAtOnce],
        argv: list[str],
    ) -> None:
        unsafe = tmp_path / "unsafe"
        unsafe.mkdir()
        unsafe.chmod(0o777)
        data = unsafe / "data"
        safe = tmp_path / "safe"
        safe.mkdir()
        safe.chmod(0o755)
        config = safe / "config.toml"
        config.write_text("port = 8787\n")
        config.chmod(0o644)
        monkeypatch.setenv("USAGEDASH_DATA_DIR", str(data))
        monkeypatch.setenv("USAGEDASH_CONFIG", str(config))
        started: list[str] = []
        monkeypatch.setattr(standing.subprocess, "run", lambda *_args, **_options: started.append("login"))
        monkeypatch.setattr(credentials.subprocess, "run", lambda *_args, **_options: started.append("credential"))
        monkeypatch.setattr(agent, "install", lambda *_args, **_options: started.append("install"))
        try:
            assert cli.main(argv) == 1
            assert f"cannot use {unsafe}: its permissions (0777)" in capsys.readouterr().err
            assert unsafe.stat().st_mode & 0o777 == 0o777
            assert config.read_text() == "port = 8787\n"
            assert started == []
            assert servers == []
            assert not data.exists()
            assert not data_dir.exists()
        finally:
            unsafe.chmod(0o700)

    @pytest.mark.parametrize("action", ["status", "uninstall"])
    @pytest.mark.parametrize("defect", ["group-writable file", "directory others can write"])
    def test_agent_status_and_uninstall_run_when_the_config_cannot_be_trusted(
        self,
        data_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        action: str,
        defect: str,
    ) -> None:
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: False)
        removed: list[agent.AgentPaths] = []
        monkeypatch.setattr(agent, "uninstall", removed.append)
        chosen: Path | None = None
        if defect == "group-writable file":
            data_dir.mkdir()
            config = data_dir / "config.toml"
            config.write_text('[binaries]\ngrok = "/not/run"\n')
            config.chmod(0o664)
        else:
            chosen = tmp_path / "chosen"
            chosen.mkdir()
            chosen.chmod(0o777)
            config = chosen / "config.toml"
            config.write_text("port = 8787\n")
            monkeypatch.setenv("USAGEDASH_CONFIG", str(config))
        try:
            assert cli.main(["agent", action]) == 0
        finally:
            if chosen is not None:
                chosen.chmod(0o700)

        output = capsys.readouterr().out
        if action == "status":
            assert output.strip().endswith("not loaded")
            assert removed == []
            return
        assert len(removed) == 1
        assert "its data was kept" in output
        assert str(data_dir) not in output

    def test_a_failed_vendor_login_exits_nonzero(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        restarted: list[str] = []
        monkeypatch.setattr(standing, "run_login", lambda *_: 3)
        monkeypatch.setattr(agent, "kickstart_if_loaded", lambda paths: restarted.append(paths.label) is None)

        assert cli.main(["login", "codex"]) == 1

        assert "login exited with status 3" in capsys.readouterr().err
        assert restarted == []
        assert not (data_dir / "config.toml").exists()

    def test_login_restarts_a_dashboard_installed_for_this_data_directory_so_it_picks_up_the_new_login(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        restarted: list[str] = []
        self.installed_for(monkeypatch, data_dir)
        monkeypatch.setattr(standing, "run_login", lambda *_: 0)
        monkeypatch.setattr(agent, "kickstart_if_loaded", lambda paths: restarted.append(paths.label) is None)

        assert cli.main(["login", "claude", "--name", "work"]) == 0
        assert len(restarted) == 1

    def test_login_does_not_restart_a_loaded_service_whose_property_list_is_gone(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        restarted: list[str] = []
        monkeypatch.setattr(standing, "run_login", lambda *_: 0)
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: True)
        monkeypatch.setattr(agent, "kickstart_if_loaded", lambda paths: restarted.append(paths.label) or True)

        assert cli.main(["login", "claude", "--name", "work"]) == 0

        output = capsys.readouterr().out
        assert restarted == []
        assert "its property list is gone" in output
        assert "usagedash agent install" in output
        assert "appears after the next collection" not in output
        assert tomllib.loads((data_dir / "config.toml").read_text())["claude"] == [
            {"config_dir": str(data_dir / "homes" / "claude-work"), "mode": "cli"}
        ]

    def test_a_sign_in_that_fails_and_is_retried_still_ends_with_the_dashboard_reading_the_login(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        restarted: list[str] = []
        exit_statuses = iter([1, 0])
        self.installed_for(monkeypatch, data_dir)
        monkeypatch.setattr(standing, "run_login", lambda *_: next(exit_statuses))
        monkeypatch.setattr(agent, "kickstart_if_loaded", lambda paths: restarted.append(paths.label) is None)

        assert cli.main(["login", "grok", "--name", "work"]) == 1
        assert cli.main(["login", "grok", "--name", "work"]) == 0

        registered = tomllib.loads((data_dir / "config.toml").read_text())["grok"]
        assert registered == [{"home": str(data_dir / "homes" / "grok-work"), "mode": "cli"}]
        assert len(restarted) == 1

    def test_signing_in_again_to_an_already_registered_login_restarts_the_dashboard_without_registering_twice(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        restarted: list[str] = []
        self.installed_for(monkeypatch, data_dir)
        monkeypatch.setattr(standing, "run_login", lambda *_: 0)
        monkeypatch.setattr(agent, "kickstart_if_loaded", lambda paths: restarted.append(paths.label) is None)

        assert cli.main(["login", "codex", "--name", "personal"]) == 0
        assert cli.main(["login", "codex", "--name", "personal"]) == 0

        assert len(tomllib.loads((data_dir / "config.toml").read_text())["codex"]) == 1
        assert len(restarted) == 2

    def test_agent_status_reports_whether_the_agent_is_loaded(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: False)
        assert cli.main(["agent", "status"]) == 0
        assert capsys.readouterr().out.strip().endswith("not loaded")

    def test_agent_install_hands_the_agent_the_settings_overrides_in_effect_and_nothing_else(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        installed: dict[str, Any] = {}

        def record_install(paths: agent.AgentPaths, working_directory: Path, executables: Any, overrides: Any) -> None:
            installed.update(working_directory=working_directory, overrides=dict(overrides))

        monkeypatch.setenv("USAGEDASH_PORT", "9001")
        monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-copied")
        monkeypatch.setattr(agent, "install", record_install)

        assert cli.main(["agent", "install"]) == 0

        assert installed["working_directory"] == data_dir
        overrides = installed["overrides"]
        pinned = {name: value for name, value in overrides.items() if name.endswith("_BINARY")}
        assert {name: value for name, value in overrides.items() if name not in pinned} == {
            "HOME": str(data_dir.parent),
            "USAGEDASH_DATA_DIR": str(data_dir),
            "USAGEDASH_PORT": "9001",
        }
        assert set(pinned) <= {"USAGEDASH_CLAUDE_BINARY", "USAGEDASH_CODEX_BINARY", "USAGEDASH_GROK_BINARY"}
        assert "http://127.0.0.1:9001" in capsys.readouterr().out
        assert "UNRELATED_SECRET" not in overrides

    def test_install_pins_a_named_config_as_the_absolute_path_it_resolved_to(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "custom.toml"
        config.write_text("port = 9001\n")
        monkeypatch.setenv("USAGEDASH_CONFIG", "~/custom.toml")
        installed: dict[str, Any] = {}

        def record_install(paths: agent.AgentPaths, working_directory: Path, executables: Any, overrides: Any) -> None:
            installed.update(overrides=dict(overrides))

        monkeypatch.setattr(agent, "install", record_install)

        assert cli.main(["agent", "install"]) == 0

        assert installed["overrides"]["USAGEDASH_CONFIG"] == str(config)
        assert installed["overrides"]["HOME"] == str(tmp_path)
        assert installed["overrides"]["USAGEDASH_DATA_DIR"] == str(data_dir)

    def test_open_reads_the_home_recorded_for_the_agent_when_this_shell_has_another(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed_home = tmp_path / "installed-home"
        invoking_home = tmp_path / "invoking-home"
        installed_home.mkdir()
        invoking_home.mkdir()
        data_dir.mkdir()
        (data_dir / "config.toml").write_text(f'[[claude]]\nconfig_dir = "{invoking_home / ".claude"}"\nmode = "cli"\n')
        monkeypatch.setenv("HOME", str(invoking_home))
        monkeypatch.setattr(
            agent,
            "installed_environment",
            lambda _paths: {
                "HOME": str(installed_home),
                "USAGEDASH_DATA_DIR": str(data_dir),
                "USAGEDASH_PORT": "8787",
            },
        )

        served = cli._settings_the_installed_agent_serves()

        assert served is not None
        assert served.claude_sources[0].config_dir == installed_home / ".claude"
        assert served.claude_sources[1].config_dir == invoking_home / ".claude"
        assert served.codex_sources[0].home == installed_home / ".codex"
        assert served.grok_sources[0].home == installed_home / ".grok"

    @pytest.mark.parametrize("home_value", ["relative-home", ""])
    def test_install_refuses_a_relative_or_empty_home_before_it_writes_the_agent(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], home_value: str
    ) -> None:
        calls: list[object] = []
        monkeypatch.setenv("HOME", home_value)
        monkeypatch.setattr(agent, "install", lambda *_args, **_kwargs: calls.append(True))

        assert cli.main(["agent", "install"]) == cli.EXIT_FAILURE

        assert calls == []
        assert "HOME must name an absolute directory" in capsys.readouterr().err

    def test_install_pins_the_home_the_defaults_were_resolved_with(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("USAGEDASH_DATA_DIR")
        monkeypatch.delenv("USAGEDASH_CONFIG", raising=False)
        before = load_settings()
        launchctl_not_loaded(monkeypatch)

        assert cli.main(["agent", "install"]) == 0

        variables = plistlib.loads(agent.agent_paths().plist.read_bytes())["EnvironmentVariables"]
        assert variables["HOME"] == str(tmp_path)
        assert variables["USAGEDASH_DATA_DIR"] == str(before.data_dir)
        assert variables["USAGEDASH_PORT"] == str(before.port)
        assert "USAGEDASH_CONFIG" not in variables
        elsewhere = tmp_path / "launchd-home"
        monkeypatch.setenv("HOME", str(elsewhere))
        drifted = load_settings(
            env={
                "USAGEDASH_DATA_DIR": variables["USAGEDASH_DATA_DIR"],
                "USAGEDASH_PORT": variables["USAGEDASH_PORT"],
            },
            home=elsewhere,
        )
        assert drifted.data_dir == before.data_dir
        assert drifted.config_path == before.config_path
        assert drifted.claude_sources[0].config_dir == elsewhere / ".claude"
        for name in ("HOME", "USAGEDASH_DATA_DIR", "USAGEDASH_PORT"):
            monkeypatch.setenv(name, variables[name])
        monkeypatch.delenv("USAGEDASH_CONFIG", raising=False)
        restored = load_settings()

        assert restored.data_dir == before.data_dir
        assert restored.config_path == before.config_path
        assert restored.claude_sources[0].config_dir == before.claude_sources[0].config_dir
        assert restored.codex_sources[0].home == before.codex_sources[0].home
        assert restored.grok_sources[0].home == before.grok_sources[0].home
        assert not data_dir.exists()

    def test_agent_install_gives_the_agent_each_cli_this_shell_resolved_by_its_path(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed: dict[str, Any] = {}
        tools = tmp_path / "tools"
        tools.mkdir()
        for name in ("claude", "codex"):
            (tools / name).write_text("#!/bin/sh\n")
            (tools / name).chmod(0o755)
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "config.toml").write_text(
            f'[binaries]\nclaude = "{tools / "claude"}"\ncodex = "{tools / "codex"}"\ngrok = "{tools / "grok"}"\n'
        )

        def record_install(paths: agent.AgentPaths, working_directory: Path, executables: Any, overrides: Any) -> None:
            installed.update(executables=list(executables), overrides=dict(overrides))

        monkeypatch.setattr(agent, "install", record_install)

        assert cli.main(["agent", "install"]) == 0

        assert installed["executables"] == [str(tools / "claude"), str(tools / "codex"), str(tools / "grok")]
        assert {name: value for name, value in installed["overrides"].items() if name.endswith("_BINARY")} == {
            "USAGEDASH_CLAUDE_BINARY": str(tools / "claude"),
            "USAGEDASH_CODEX_BINARY": str(tools / "codex"),
            "USAGEDASH_GROK_BINARY": str(tools / "grok"),
        }

    def test_a_cli_this_shell_did_not_find_is_pinned_to_checked_path_lookup(self) -> None:
        assert cli._pinned_executables({"codex": "codex", "grok": "/opt/grok"}) == {
            "USAGEDASH_CODEX_BINARY": "codex",
            "USAGEDASH_GROK_BINARY": "/opt/grok",
        }

    def test_serve_binds_the_loopback_address_only_and_creates_the_access_token(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, servers: list[ServerThatReturnsAtOnce]
    ) -> None:
        monkeypatch.setenv("USAGEDASH_PORT", "9002")

        assert cli.main(["serve"]) == 0

        (server,) = servers
        assert (server.config.host, server.config.port) == ("127.0.0.1", 9002)
        assert server.config.access_log is False
        assert (data_dir / "access-token").stat().st_mode & 0o777 == 0o600

    def test_interrupting_the_running_server_shuts_it_down_and_exits_with_the_interrupt_status(
        self, real_server: list[uvicorn.Server], capsys: pytest.CaptureFixture[str]
    ) -> None:
        port = int(os.environ["USAGEDASH_PORT"])
        answered: list[bool] = []

        def interrupt_once_it_serves() -> None:
            deadline = time.monotonic() + SERVER_WAIT_SECONDS
            while time.monotonic() < deadline and not answered:
                if answers_health_checks(port):
                    answered.append(True)
                else:
                    time.sleep(SERVER_POLL_SECONDS)
            if answered:
                os.kill(os.getpid(), signal.SIGINT)
            else:
                # Why not a signal here: with no server to catch it, it would interrupt the test run itself.
                for server in real_server:
                    server.should_exit = True

        interrupter = threading.Thread(target=interrupt_once_it_serves)
        interrupter.start()
        try:
            exit_status = cli.main(["serve"])
        finally:
            interrupter.join(timeout=SERVER_WAIT_SECONDS)

        assert answered == [True]
        assert exit_status == 130
        assert capsys.readouterr().err.endswith("interrupted\n")
        assert not answers_health_checks(port)

    def test_an_app_that_fails_to_start_ends_the_server_with_a_failure_status(
        self, real_server: list[uvicorn.Server], monkeypatch: pytest.MonkeyPatch
    ) -> None:

        @contextlib.asynccontextmanager
        async def fails_to_start(_app: FastAPI) -> AsyncGenerator[None]:
            raise RuntimeError("synthetic startup failure")
            yield

        monkeypatch.setattr(cli, "create_app", lambda *_arguments, **_options: FastAPI(lifespan=fails_to_start))

        with pytest.raises(SystemExit) as exited:
            cli.main(["serve"])

        assert exited.value.code not in (None, 0)
        assert [server.started for server in real_server] == [False]

    def test_a_port_that_is_already_taken_ends_the_server_with_a_failure_status(
        self, real_server: list[uvicorn.Server]
    ) -> None:
        with socket.socket() as occupant:
            occupant.bind(("127.0.0.1", int(os.environ["USAGEDASH_PORT"])))
            occupant.listen()

            with pytest.raises(SystemExit) as exited:
                cli.main(["serve"])

        assert exited.value.code not in (None, 0)
        assert [server.started for server in real_server] == [False]

    def test_open_hands_the_browser_a_private_page_so_the_token_is_never_a_command_argument_or_printed(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        opened: list[str] = []
        monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)
        monkeypatch.setattr(cli, "_dashboard_is_listening", lambda *_args: True)
        monkeypatch.setenv("USAGEDASH_PORT", "9003")

        assert cli.main(["open"]) == 0

        token = (data_dir / "access-token").read_text()
        page = data_dir / "open-dashboard.html"
        assert opened == [page.as_uri()]
        assert token not in opened[0]
        assert f'window.__USAGEDASH_TOKEN = "{token}";' in page.read_text()
        assert 'window.__USAGEDASH_ORIGIN = "http://127.0.0.1:9003";' in page.read_text()
        assert page.stat().st_mode & 0o777 == 0o600
        output = capsys.readouterr()
        assert token not in output.out + output.err
        assert "http://127.0.0.1:9003/" in output.out

    def test_open_addresses_the_dashboard_the_installed_agent_serves(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed_data = tmp_path / "installed-data"
        monkeypatch.setenv("USAGEDASH_DATA_DIR", str(installed_data))
        monkeypatch.setenv("USAGEDASH_PORT", "9001")
        launchctl_not_loaded(monkeypatch)
        assert cli.main(["agent", "install"]) == 0
        monkeypatch.setenv("USAGEDASH_DATA_DIR", str(data_dir))
        monkeypatch.delenv("USAGEDASH_PORT")
        opened: list[str] = []
        monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)
        monkeypatch.setattr(cli, "_dashboard_is_listening", lambda *_args: True)

        assert cli.main(["open"]) == 0

        token = (installed_data / "access-token").read_text()
        page = installed_data / "open-dashboard.html"
        assert opened == [page.as_uri()]
        assert f'window.__USAGEDASH_TOKEN = "{token}";' in page.read_text()
        assert 'window.__USAGEDASH_ORIGIN = "http://127.0.0.1:9001";' in page.read_text()
        assert "127.0.0.1:8787" not in page.read_text()
        assert not (data_dir / "access-token").exists()
        assert not (data_dir / "open-dashboard.html").exists()

    def test_open_ignores_a_port_set_in_the_shell_after_the_agent_was_installed(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        launchctl_not_loaded(monkeypatch)
        assert cli.main(["agent", "install"]) == 0
        monkeypatch.setenv("USAGEDASH_PORT", "9003")
        monkeypatch.setattr(cli.webbrowser, "open", lambda url: True)
        monkeypatch.setattr(cli, "_dashboard_is_listening", lambda *_args: True)

        assert cli.main(["open"]) == 0

        token = (data_dir / "access-token").read_text()
        page = (data_dir / "open-dashboard.html").read_text()
        assert f'window.__USAGEDASH_TOKEN = "{token}";' in page
        assert 'window.__USAGEDASH_ORIGIN = "http://127.0.0.1:8787";' in page
        assert "9003" not in page

    def test_open_keeps_the_port_from_the_config_file_the_agent_was_installed_with(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data_dir.mkdir()
        config = data_dir / "config.toml"
        config.write_text("port = 9001\n")
        monkeypatch.setenv("USAGEDASH_CONFIG", str(config))
        launchctl_not_loaded(monkeypatch)
        assert cli.main(["agent", "install"]) == 0
        config.write_text("port = 9002\n")
        monkeypatch.setattr(cli.webbrowser, "open", lambda url: True)
        monkeypatch.setattr(cli, "_dashboard_is_listening", lambda *_args: True)

        assert cli.main(["open"]) == 0

        installed = plistlib.loads(agent.agent_paths().plist.read_bytes())
        page = (data_dir / "open-dashboard.html").read_text()
        assert installed["EnvironmentVariables"]["USAGEDASH_PORT"] == "9001"
        assert "127.0.0.1:9001" in page
        assert "9002" not in page

    def test_open_uses_the_installed_agent_when_this_shells_settings_are_not_valid(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        launchctl_not_loaded(monkeypatch)
        assert cli.main(["agent", "install"]) == 0
        monkeypatch.setenv("USAGEDASH_PORT", "80")
        monkeypatch.setattr(cli.webbrowser, "open", lambda url: True)
        monkeypatch.setattr(cli, "_dashboard_is_listening", lambda *_args: True)

        assert cli.main(["open"]) == 0

        token = (data_dir / "access-token").read_text()
        page = (data_dir / "open-dashboard.html").read_text()
        assert f'window.__USAGEDASH_TOKEN = "{token}";' in page
        assert 'window.__USAGEDASH_ORIGIN = "http://127.0.0.1:8787";' in page
        assert "port must be between" not in page

    def test_open_opens_nothing_when_the_installed_agent_cannot_be_read(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        launchctl_not_loaded(monkeypatch)
        assert cli.main(["agent", "install"]) == 0
        plist = agent.agent_paths().plist
        descriptor = agent.open_private_file(plist)
        try:
            os.ftruncate(descriptor, 0)
            os.write(descriptor, b"not a plist")
        finally:
            os.close(descriptor)
        opened: list[str] = []
        monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)

        assert cli.main(["open"]) == 1

        assert opened == []
        assert "cannot read" in capsys.readouterr().err
        assert not (data_dir / "open-dashboard.html").exists()

    @pytest.mark.parametrize(("payload", "failure"), PLISTS_THE_PARSER_CANNOT_FINISH)
    def test_open_reports_a_plist_the_parser_cannot_finish_in_one_line_and_restarts_nothing(
        self,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        payload: bytes,
        failure: str,
    ) -> None:
        run = launchctl_not_loaded(monkeypatch)
        assert cli.main(["agent", "install"]) == 0
        capsys.readouterr()
        plist = agent.agent_paths().plist
        write_private_bytes(plist, payload)
        opened: list[str] = []
        restarted: list[agent.AgentPaths] = []
        monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)
        monkeypatch.setattr(agent, "kickstart_if_loaded", lambda paths: restarted.append(paths) or False)

        assert cli.main(["open"]) == 1

        assert capsys.readouterr().err == f"error: cannot read {plist}: {failure}\n"
        assert opened == []
        assert restarted == []
        assert "kickstart" not in run.subcommands()
        assert not (data_dir / "open-dashboard.html").exists()
        assert not (data_dir / "access-token").exists()

    def test_open_opens_nothing_when_the_installed_agents_settings_are_not_valid(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        launchctl_not_loaded(monkeypatch)
        assert cli.main(["agent", "install"]) == 0
        plist = agent.agent_paths().plist
        loaded = plistlib.loads(plist.read_bytes())
        loaded["EnvironmentVariables"]["USAGEDASH_PORT"] = "80"
        descriptor = agent.open_private_file(plist)
        try:
            os.ftruncate(descriptor, 0)
            os.write(descriptor, plistlib.dumps(loaded))
        finally:
            os.close(descriptor)
        opened: list[str] = []
        monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)

        assert cli.main(["open"]) == 1

        assert opened == []
        assert "port must be between" in capsys.readouterr().err
        assert not (data_dir / "open-dashboard.html").exists()

    def test_agent_status_and_uninstall_run_when_the_port_setting_is_not_valid(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("USAGEDASH_PORT", "80")
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: False)
        removed: list[agent.AgentPaths] = []
        monkeypatch.setattr(agent, "uninstall", removed.append)

        assert cli.main(["agent", "status"]) == 0
        assert capsys.readouterr().out.strip().endswith("not loaded")
        assert cli.main(["agent", "uninstall"]) == 0

        assert len(removed) == 1
        assert capsys.readouterr().out.strip().endswith("its data was kept")
        assert cli.main(["status"]) == 1
        assert "error: port must be between" in capsys.readouterr().err

    def test_login_does_not_restart_an_agent_installed_for_another_data_directory(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        other = tmp_path / "other-data"
        monkeypatch.setenv("USAGEDASH_DATA_DIR", str(other))
        run = FakeRun()
        run.sequences = {"print": [NOT_LOADED]}
        monkeypatch.setattr(agent.subprocess, "run", run)
        assert cli.main(["agent", "install"]) == 0
        monkeypatch.setenv("USAGEDASH_DATA_DIR", str(data_dir))
        restarted: list[str] = []
        monkeypatch.setattr(standing, "run_login", lambda *_arguments: 0)
        monkeypatch.setattr(agent, "kickstart_if_loaded", lambda paths: restarted.append(paths.label) or False)

        assert cli.main(["login", "claude", "--name", "work"]) == 0

        assert restarted == []
        output = capsys.readouterr().out
        assert "appears after the next collection" not in output
        assert "different settings" in output
        assert tomllib.loads((data_dir / "config.toml").read_text())["claude"] == [
            {"config_dir": str(data_dir / "homes" / "claude-work"), "mode": "cli"}
        ]

    def test_login_does_not_restart_an_agent_that_is_not_running_for_another_data_directory(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        other = tmp_path / "other-data"
        monkeypatch.setenv("USAGEDASH_DATA_DIR", str(other))
        run = FakeRun()
        run.sequences = {"print": [NOT_LOADED, NOT_LOADED]}
        monkeypatch.setattr(agent.subprocess, "run", run)
        assert cli.main(["agent", "install"]) == 0
        monkeypatch.setenv("USAGEDASH_DATA_DIR", str(data_dir))
        restarted: list[str] = []
        monkeypatch.setattr(standing, "run_login", lambda *_arguments: 0)
        monkeypatch.setattr(agent, "kickstart_if_loaded", lambda paths: restarted.append(paths.label) or False)

        assert cli.main(["login", "claude", "--name", "work"]) == 0

        assert restarted == []
        output = capsys.readouterr().out
        assert "different settings" not in output
        assert "usagedash agent install" in output

    def test_login_does_not_restart_an_agent_installed_with_another_config_file(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        other_config = tmp_path / "other-config.toml"
        other_config.write_text("")
        monkeypatch.setenv("USAGEDASH_CONFIG", str(other_config))
        run = FakeRun()
        run.sequences = {"print": [NOT_LOADED]}
        monkeypatch.setattr(agent.subprocess, "run", run)
        assert cli.main(["agent", "install"]) == 0
        monkeypatch.delenv("USAGEDASH_CONFIG")
        restarted: list[str] = []
        monkeypatch.setattr(standing, "run_login", lambda *_arguments: 0)
        monkeypatch.setattr(agent, "kickstart_if_loaded", lambda paths: restarted.append(paths.label) or False)

        assert cli.main(["login", "codex", "--name", "work"]) == 0

        assert restarted == []
        assert "different settings" in capsys.readouterr().out

    @pytest.mark.parametrize(("payload", "failure"), PLISTS_THE_PARSER_CANNOT_FINISH)
    def test_login_reports_a_plist_the_parser_cannot_finish_in_one_line_and_restarts_nothing(
        self,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        payload: bytes,
        failure: str,
    ) -> None:
        run = launchctl_not_loaded(monkeypatch)
        assert cli.main(["agent", "install"]) == 0
        capsys.readouterr()
        plist = agent.agent_paths().plist
        write_private_bytes(plist, payload)
        restarted: list[str] = []
        monkeypatch.setattr(standing, "run_login", lambda *_arguments: 0)
        monkeypatch.setattr(agent, "kickstart_if_loaded", lambda paths: restarted.append(paths.label) or False)

        assert cli.main(["login", "claude", "--name", "work"]) == 1

        captured = capsys.readouterr()
        assert captured.err == f"error: cannot read {plist}: {failure}\n"
        assert "Dashboard restarted" not in captured.out
        assert restarted == []
        assert "kickstart" not in run.subcommands()
        assert tomllib.loads((data_dir / "config.toml").read_text())["claude"] == [
            {"config_dir": str(data_dir / "homes" / "claude-work"), "mode": "cli"}
        ]

    def test_login_restarts_an_agent_installed_for_this_data_directory_on_another_port(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("USAGEDASH_PORT", "9001")
        run = FakeRun()
        run.sequences = {"print": [NOT_LOADED]}
        monkeypatch.setattr(agent.subprocess, "run", run)
        assert cli.main(["agent", "install"]) == 0
        monkeypatch.delenv("USAGEDASH_PORT")
        restarted: list[str] = []
        monkeypatch.setattr(standing, "run_login", lambda *_arguments: 0)
        monkeypatch.setattr(agent, "kickstart_if_loaded", lambda paths: restarted.append(paths.label) or True)

        assert cli.main(["login", "grok", "--name", "work"]) == 0

        assert len(restarted) == 1
        assert "appears after the next collection" in capsys.readouterr().out

    def test_login_treats_another_case_of_the_same_directory_as_this_dashboard(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run = FakeRun()
        run.sequences = {"print": [NOT_LOADED]}
        monkeypatch.setattr(agent.subprocess, "run", run)
        assert cli.main(["agent", "install"]) == 0
        alias = data_dir.parent / data_dir.name.swapcase()
        same_directory = alias.exists() and os.path.samefile(alias, data_dir)
        if not same_directory:
            alias.mkdir()
        monkeypatch.setenv("USAGEDASH_DATA_DIR", str(alias))
        restarted: list[str] = []
        monkeypatch.setattr(standing, "run_login", lambda *_arguments: 0)
        monkeypatch.setattr(agent, "kickstart_if_loaded", lambda paths: restarted.append(paths.label) or True)

        assert cli.main(["login", "grok", "--name", "work"]) == 0

        if same_directory:
            assert len(restarted) == 1
            assert "appears after the next collection" in capsys.readouterr().out
        else:
            assert restarted == []
            assert "different settings" in capsys.readouterr().out

    def test_login_reads_and_restarts_the_installed_dashboard_under_one_lock(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []
        real_lock = agent._one_change_at_a_time

        @contextlib.contextmanager
        def tracked(paths: agent.AgentPaths) -> Iterator[None]:
            events.append("lock")
            with real_lock(paths):
                yield
            events.append("unlock")

        def read_environment(_paths: agent.AgentPaths) -> dict[str, str] | None:
            assert events[-1] == "lock"
            events.append("read")
            return {"USAGEDASH_DATA_DIR": os.environ["USAGEDASH_DATA_DIR"]}

        monkeypatch.setattr(agent, "_one_change_at_a_time", tracked)
        monkeypatch.setattr(agent, "installed_environment", read_environment)
        monkeypatch.setattr(standing, "run_login", lambda *_arguments: 0)

        def kick(paths: agent.AgentPaths) -> bool:
            assert events[-1] == "read"
            events.append("kick")
            return True

        monkeypatch.setattr(agent, "kickstart_if_loaded", kick)

        assert cli.main(["login", "claude", "--name", "work"]) == 0

        assert events == ["lock", "read", "kick", "unlock"]

    def test_the_server_s_log_directory_is_owner_only(
        self,
        data_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        servers: list[ServerThatReturnsAtOnce],
    ) -> None:
        log_dir = tmp_path / "Library" / "Logs" / "ai-usage-dashboard"
        log_dir.mkdir(parents=True)
        log_dir.chmod(0o755)
        monkeypatch.setattr(logging, "basicConfig", lambda **options: None)

        assert cli.main(["serve"]) == 0

        assert log_dir.stat().st_mode & 0o777 == 0o700
        assert len(servers) == 1

    def test_the_server_s_log_file_is_owner_only_even_if_it_existed_open_to_others(
        self,
        data_dir: Path,
        log_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        servers: list[ServerThatReturnsAtOnce],
    ) -> None:
        log_file.parent.mkdir(parents=True)
        log_file.write_text("an earlier line\n")
        log_file.chmod(0o644)
        monkeypatch.setattr(logging, "basicConfig", lambda **options: None)

        assert cli.main(["serve"]) == 0

        assert log_file.stat().st_mode & 0o777 == 0o600
        assert log_file.read_text() == "an earlier line\n"

    def test_a_log_file_that_cannot_be_opened_is_reported_and_the_server_does_not_start(
        self,
        data_dir: Path,
        log_file: Path,
        servers: list[ServerThatReturnsAtOnce],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        log_file.mkdir(parents=True)

        assert cli.main(["serve"]) == 1

        assert "usagedash.log as a private file: IsADirectoryError" in capsys.readouterr().err
        assert servers == []

    def test_a_symbolic_link_where_the_log_file_goes_is_refused_and_the_server_does_not_start(
        self,
        data_dir: Path,
        tmp_path: Path,
        log_file: Path,
        servers: list[ServerThatReturnsAtOnce],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        target = tmp_path / "somewhere-else"
        target.write_text("not the dashboard's\n")
        log_file.parent.mkdir(parents=True)
        log_file.symlink_to(target)

        assert cli.main(["serve"]) == 1

        assert "usagedash.log as a private file" in capsys.readouterr().err
        assert servers == []
        assert target.read_text() == "not the dashboard's\n"
        assert log_file.is_symlink()

    def test_a_log_file_that_cannot_be_set_to_append_is_reported_and_is_not_left_open(
        self,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        servers: list[ServerThatReturnsAtOnce],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        opened: list[int] = []
        real_open = cli.open_private_file

        def record_open(path: Path) -> int:
            opened.append(real_open(path))
            return opened[-1]

        def refuse(*_: object) -> int:
            raise OSError(errno.EBADF, "Bad file descriptor")

        monkeypatch.setattr(cli, "open_private_file", record_open)
        monkeypatch.setattr(cli.fcntl, "fcntl", refuse)

        assert cli.main(["serve"]) == 1

        assert "error: cannot open the log file in" in capsys.readouterr().err
        assert servers == []
        assert len(opened) == 2
        for descriptor in opened:
            with pytest.raises(OSError, match="Bad file descriptor"):
                os.fstat(descriptor)

    def test_a_login_the_config_cannot_take_is_refused_before_the_sign_in_is_started(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sign_ins: list[Provider] = []
        data_dir.mkdir()
        (data_dir / "config.toml").write_text("codex = []\n")
        monkeypatch.setattr(standing, "run_login", lambda _settings, provider, _home: sign_ins.append(provider) or 0)
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: False)

        assert cli.main(["login", "codex", "--name", "work"]) == 1

        home = data_dir / "homes" / "codex-work"
        assert sign_ins == []
        assert capsys.readouterr().err == (
            f"error: cannot add this login to {data_dir / 'config.toml'}: the file would not load with a "
            "[[codex]] table added to its end. If it lists its codex logins on one line, add this entry to "
            f'that list and run the command again: {{ home = "{home}", mode = "cli" }}\n'
        )
        assert (data_dir / "config.toml").read_text() == "codex = []\n"

    def test_a_login_listed_on_one_line_by_hand_is_signed_in_to_without_touching_the_config(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sign_ins: list[Provider] = []
        home = data_dir / "homes" / "codex-work"
        listed = f'codex = [{{ home = "{home}", mode = "cli" }}]\n'
        data_dir.mkdir()
        (data_dir / "config.toml").write_text(listed)
        monkeypatch.setattr(standing, "run_login", lambda _settings, provider, _home: sign_ins.append(provider) or 0)
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: False)

        assert cli.main(["login", "codex", "--name", "work"]) == 0

        assert sign_ins == [Provider.CODEX]
        assert (data_dir / "config.toml").read_text() == listed

    def test_the_settings_a_login_is_checked_against_are_the_command_s_own_with_only_the_config_file_changed(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other_config = tmp_path / "other.toml"
        other_config.write_text("port = 9100\n")
        monkeypatch.setenv("USAGEDASH_PORT", "9200")

        loaded = cli._settings_with_config(other_config)

        assert (loaded.config_path, loaded.data_dir, loaded.port) == (other_config, data_dir, 9200)

    def test_an_interrupt_that_says_what_it_left_behind_has_that_printed(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def interrupted(*_: object) -> None:
            raise KeyboardInterrupt("the previous agent was started again")

        monkeypatch.setattr(agent, "install", interrupted)

        assert cli.main(["agent", "install"]) == 130
        assert capsys.readouterr().err == "interrupted; the previous agent was started again\n"

    @pytest.mark.parametrize(
        "failure",
        [PermissionError(errno.EACCES, "Permission denied"), ValueError("bad")],
    )
    def test_a_config_directory_that_cannot_be_examined_is_one_line(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: Exception
    ) -> None:
        data_dir.mkdir()
        real_stat = os.stat

        def stat(path: os.PathLike[str] | str, *args: object, **kwargs: object) -> os.stat_result:
            if Path(path) == data_dir:
                raise failure
            return real_stat(path, *args, **kwargs)

        def absent(self: Path) -> bool:
            return False

        monkeypatch.setattr(os, "stat", stat)
        monkeypatch.setattr(Path, "exists", absent)
        monkeypatch.setattr(Path, "is_symlink", absent)

        assert cli.main(["login", "codex"]) == 1

        assert capsys.readouterr().err == f"error: cannot use {data_dir}: {type(failure).__name__}\n"

    def test_an_uninstall_interrupted_while_the_plist_cannot_be_examined_exits_130(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        paths = agent.agent_paths(home=tmp_path, user="persona")
        paths.plist.parent.mkdir(parents=True)
        write_private_bytes(paths.plist, b"installed-and-trusted")
        monkeypatch.setattr(agent, "agent_paths", lambda home=None, user=None: paths)
        monkeypatch.setattr(agent.subprocess, "run", InterruptedAt("bootout"))
        real_lstat = Path.lstat

        def lstat(self: Path) -> os.stat_result:
            if self == paths.plist:
                raise PermissionError(errno.EACCES, "Permission denied")
            return real_lstat(self)

        monkeypatch.setattr(Path, "lstat", lstat)

        assert cli.main(["agent", "uninstall"]) == 130

        assert capsys.readouterr().err == (
            "interrupted; the previous installation could not be put back: "
            f"cannot examine {paths.plist}: PermissionError\n"
        )

    def collect_from(
        self,
        monkeypatch: pytest.MonkeyPatch,
        quota: Exception,
        scan: Exception | None = None,
        another_cycle_stores: Callable[[sqlite3.Connection], None] | None = None,
    ) -> int:
        """Run ``usagedash collect`` over one source whose quota read, and token scan, end as given.

        ``another_cycle_stores`` writes what a second collection would, after the command's own cycle
        has let go of the collection lock and before the command reads anything back.
        """

        class ScriptedSource:
            source_id = "codex:scripted"
            provider = Provider.CODEX

            def current_identity(self) -> AccountIdentity | None:
                return None

            def read_quota(self, now: datetime) -> QuotaReading:
                raise quota

        class ScriptedScanner:
            provider = Provider.CODEX

            def scan(
                self, database: Database, resolve_account: AccountResolver, now: datetime, backfill_days: int
            ) -> ScanReport:
                if scan is not None:
                    raise scan
                return ScanReport()

        class FollowedByAnotherCycle(Collector):
            def run_cycle(self) -> CycleReport:
                report = super().run_cycle()
                if another_cycle_stores is not None:
                    with self._database.connect() as conn:
                        another_cycle_stores(conn)
                return report

        def build(_settings: Settings, database: Database, _client: object, clock: Any) -> Collector:
            bundles = [SourceBundle(ScriptedSource(), ScriptedScanner(), 0)]
            return FollowedByAnotherCycle(database, bundles, clock, poll_interval_seconds=3600, backfill_days=35)

        monkeypatch.setattr(cli, "build_collector", build)
        return cli.main(["collect"])

    def test_collect_exits_nonzero_for_its_own_failed_read_although_another_cycle_has_since_stored_a_success(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], now: datetime
    ) -> None:

        def a_success(conn: sqlite3.Connection) -> None:
            db.record_status(conn, "codex:scripted", "quota", now, None)

        failure = ProviderRequestError("the usage endpoint answered 500", status_code=500)

        assert self.collect_from(monkeypatch, failure, another_cycle_stores=a_success) == 1
        assert "quota:   ProviderRequestError: the usage endpoint answered 500" in capsys.readouterr().out

    def test_collect_exits_zero_for_its_own_cycle_although_another_cycle_has_since_stored_a_failure(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, now: datetime
    ) -> None:
        def a_failure(conn: sqlite3.Connection) -> None:
            db.record_status(conn, "codex:scripted", "quota", now, ProviderSchemaError("an unexpected shape"))
            db.record_status(conn, "codex:scripted", "tokens", now, OSError("the sessions directory is gone"))

        nothing_yet = NoQuotaDataError("nothing to report yet")

        assert self.collect_from(monkeypatch, nothing_yet, another_cycle_stores=a_failure) == 0

    @pytest.mark.parametrize(
        ("quota", "scan", "types"),
        [
            pytest.param(NoQuotaDataError("nothing yet"), None, ("NoQuotaDataError", None), id="nothing to report"),
            pytest.param(
                ProviderRequestError("answered 500", status_code=500),
                OSError("gone"),
                ("ProviderRequestError", "UnexpectedFailureError"),
                id="a failed read and a failed scan",
            ),
            pytest.param(RuntimeError("a defect"), None, ("UnexpectedFailureError", None), id="a defect in the source"),
        ],
    )
    def test_a_cycle_s_report_names_the_type_of_each_error_as_the_store_records_it(
        self,
        data_dir: Path,
        now: datetime,
        quota: Exception,
        scan: Exception | None,
        types: tuple[str | None, str | None],
    ) -> None:
        class ScriptedSource:
            source_id = "codex:scripted"
            provider = Provider.CODEX

            def current_identity(self) -> AccountIdentity | None:
                return None

            def read_quota(self, _now: datetime) -> QuotaReading:
                raise quota

        class ScriptedScanner:
            provider = Provider.CODEX

            def scan(self, *_: object) -> ScanReport:
                if scan is not None:
                    raise scan
                return ScanReport()

        database = cli._open_database(load_settings())
        bundles = [SourceBundle(ScriptedSource(), ScriptedScanner(), 0)]
        runner = Collector(database, bundles, lambda: now, poll_interval_seconds=3600, backfill_days=35)

        outcome = runner.run_cycle().outcomes[0]

        with database.read() as conn:
            stored = {row.kind: row.error_type for row in db.list_statuses(conn)}
        assert (outcome.quota_error_type, outcome.token_error_type) == types
        assert (stored["quota"], stored["tokens"]) == types
        assert (outcome.quota_error is None, outcome.token_error is None) == (types[0] is None, types[1] is None)

    @pytest.mark.parametrize(
        ("error_type", "severity"),
        [
            (None, "ok"),
            ("NoQuotaDataError", "info"),
            ("CredentialUnavailableError", "info"),
            ("CredentialExpiredError", "warning"),
            ("ProviderRequestError", "error"),
            ("UnexpectedFailureError", "error"),
            ("ATypeNobodyHasNamedYet", "error"),
        ],
    )
    def test_an_outcome_is_classified_by_the_name_of_its_error_s_type(
        self, error_type: str | None, severity: str
    ) -> None:
        assert api.severity_of(error_type) == severity

    @pytest.mark.parametrize(
        "expected_state",
        [CredentialUnavailableError("nobody is logged in"), NoQuotaDataError("nothing to report yet")],
    )
    def test_collect_succeeds_when_a_source_is_only_logged_out_or_has_nothing_to_report(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, expected_state: Exception
    ) -> None:
        assert self.collect_from(monkeypatch, expected_state) == 0

    @pytest.mark.parametrize(
        "failure",
        [
            ProviderRequestError("the usage endpoint answered 500", status_code=500),
            ProviderSchemaError("the usage payload has an unexpected shape"),
            CredentialExpiredError("the access token has expired"),
            RuntimeError("synthetic defect in a source"),
        ],
    )
    def test_collect_exits_nonzero_when_a_source_s_quota_could_not_be_read(
        self,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        failure: Exception,
    ) -> None:
        assert self.collect_from(monkeypatch, failure) == 1
        assert "tokens:  ok" in capsys.readouterr().out

    def test_collect_exits_nonzero_when_a_source_s_token_scan_failed(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nobody_logged_in = CredentialUnavailableError("nobody is logged in")

        assert self.collect_from(monkeypatch, nobody_logged_in, scan=OSError("the sessions directory is gone")) == 1

    def test_interrupting_a_command_exits_with_the_interrupt_status_and_no_traceback(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def interrupted(_settings: Settings) -> int:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "_collect", interrupted)

        assert cli.main(["collect"]) == 130
        assert capsys.readouterr().err == "interrupted\n"

    @pytest.mark.parametrize(
        ("detail", "printed"),
        [
            ("first\nsecond", "interrupted; first second\n"),
            ("first\r\nsecond", "interrupted; first second\n"),
            ("\n\n", "interrupted\n"),
        ],
    )
    def test_an_interrupt_whose_detail_contains_a_line_break_is_one_line(
        self,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        detail: str,
        printed: str,
    ) -> None:
        def interrupted(_settings: Settings) -> int:
            raise KeyboardInterrupt(detail)

        monkeypatch.setattr(cli, "_collect", interrupted)

        assert cli.main(["collect"]) == 130
        assert capsys.readouterr().err == printed

    @pytest.mark.parametrize(
        ("message", "printed"),
        [
            ("first\nsecond", "error: first second\n"),
            ("first\rsecond", "error: first second\n"),
            ("first\r\nsecond", "error: first second\n"),
            ("first\n\nsecond", "error: first second\n"),
            ("cannot use /tmp/my dir/file", "error: cannot use /tmp/my dir/file\n"),
            ("cannot use /tmp/my dir/file\u2028name", "error: cannot use /tmp/my dir/file\u2028name\n"),
        ],
    )
    def test_a_command_error_with_a_line_break_is_one_line(
        self,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        message: str,
        printed: str,
    ) -> None:

        def fail(_settings: Settings) -> int:
            raise ConfigError(message)

        monkeypatch.setattr(cli, "_status", fail)

        assert cli.main(["status"]) == 1
        assert capsys.readouterr().err == printed

    def test_a_launchctl_message_with_a_line_break_is_one_line(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(agent.subprocess, "run", FakeRun(returncode=1, stderr="line one\nline two"))

        assert cli.main(["agent", "status"]) == 1
        assert capsys.readouterr().err == "error: launchctl print failed (1): line one line two\n"

    @pytest.mark.parametrize(
        "failure",
        [sqlite3.OperationalError("database is locked"), sqlite3.IntegrityError("FOREIGN KEY constraint failed")],
    )
    def test_a_sqlite_error_while_reading_status_is_one_line_and_names_only_its_type(
        self,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        failure: sqlite3.Error,
    ) -> None:

        def fail(_database: Database) -> None:
            raise failure

        monkeypatch.setattr(Database, "read", fail)

        assert cli.main(["status"]) == 1
        assert capsys.readouterr().err == f"error: database operation failed: {type(failure).__name__}\n"

    def test_a_sqlite_error_while_collecting_is_one_line_and_names_only_its_type(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        connects = 0
        real_connect = Database.connect

        @contextlib.contextmanager
        def fail_after_setup(database: Database) -> Iterator[sqlite3.Connection]:
            nonlocal connects
            connects += 1
            if connects > 1:
                raise sqlite3.OperationalError("database is locked")
            with real_connect(database) as connection:
                yield connection

        monkeypatch.setattr(Database, "connect", fail_after_setup)

        assert cli.main(["collect"]) == 1
        assert capsys.readouterr().err == "error: database operation failed: OperationalError\n"

    def test_open_refuses_a_data_directory_another_account_can_replace_before_writing_a_token(
        self,
        data_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        unsafe = tmp_path / "unsafe"
        unsafe.mkdir()
        unsafe.chmod(0o777)
        data = unsafe / "data"
        safe = tmp_path / "safe"
        safe.mkdir()
        safe.chmod(0o755)
        config = safe / "config.toml"
        config.write_text("port = 8787\n")
        monkeypatch.setenv("USAGEDASH_DATA_DIR", str(data))
        monkeypatch.setenv("USAGEDASH_CONFIG", str(config))
        monkeypatch.setenv("USAGEDASH_PORT", str(free_loopback_port()))
        opened: list[str] = []
        monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)
        try:
            assert cli.main(["open"]) == 1
        finally:
            unsafe.chmod(0o700)

        assert f"cannot use {unsafe}: its permissions (0777)" in capsys.readouterr().err
        assert opened == []
        assert not (data / "access-token").exists()
        assert not (data / "open-dashboard.html").exists()
        assert not data_dir.exists()

    def test_open_writes_nothing_when_nothing_is_listening(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        port = free_loopback_port()
        monkeypatch.setenv("USAGEDASH_PORT", str(port))
        opened: list[str] = []
        monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)

        assert cli.main(["open"]) == 1

        assert capsys.readouterr().err == (
            f"error: nothing answered at http://127.0.0.1:{port}/; is the dashboard running? "
            "Check `usagedash agent status`. Nothing was opened\n"
        )
        assert opened == []
        assert not (data_dir / "open-dashboard.html").exists()
        assert not (data_dir / "access-token").exists()

    @pytest.mark.parametrize(
        ("status_code", "location"),
        [
            pytest.param(302, "/healthz?followed=1", id="a redirect"),
            pytest.param(404, None, id="not found"),
            pytest.param(500, None, id="a failure"),
            pytest.param(403, None, id="a refusal"),
        ],
    )
    def test_open_needs_a_200_from_healthz_and_follows_no_redirect(
        self,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        status_code: int,
        location: str | None,
    ) -> None:

        def answer(path: str) -> tuple[int, bytes, str | None]:
            if "followed=1" in path:
                return 200, b'{"status": "ok"}', None
            return status_code, b"", location

        listener = HealthListener(answer)
        monkeypatch.setenv("USAGEDASH_PORT", str(listener.port))
        opened: list[str] = []
        monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)
        try:
            assert cli.main(["open"]) == 1
        finally:
            listener.close()

        assert capsys.readouterr().err == (
            f"error: nothing answered at http://127.0.0.1:{listener.port}/; is the dashboard running? "
            "Check `usagedash agent status`. Nothing was opened\n"
        )
        assert listener.paths == ["/healthz"]
        assert opened == []
        assert not (data_dir / "open-dashboard.html").exists()

    def test_open_writes_the_dashboard_page_and_opens_it_when_the_dashboard_answers(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        listener = HealthListener(lambda _path: (200, b'{"status": "ok"}', None))
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("USAGEDASH_PORT", str(listener.port))
        opened: list[str] = []
        monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)
        try:
            assert cli.main(["open"]) == 0
        finally:
            listener.close()

        token = (data_dir / "access-token").read_text()
        page = data_dir / "open-dashboard.html"
        content = page.read_text()
        assert opened == [page.as_uri()]
        assert listener.paths == ["/healthz"]
        assert f'window.__USAGEDASH_TOKEN = "{token}";' in content
        assert f'window.__USAGEDASH_ORIGIN = "http://127.0.0.1:{listener.port}";' in content
        assert f"connect-src http://127.0.0.1:{listener.port};" in content
        assert "#access=" not in content
        assert content.count(token) == 1
        assert page.stat().st_mode & 0o777 == 0o600
        output = capsys.readouterr()
        assert token not in output.out + output.err
        assert f"http://127.0.0.1:{listener.port}/" in output.out

    def test_the_liveness_probe_asks_for_healthz_and_reads_the_status_but_never_the_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: list[str] = []

        def answer(request: httpx2.Request) -> tuple[int, list[bytes]]:
            asked.append(str(request.url))
            return 200, [b'{"status": "ok"}']

        streams, seen = listen_with(monkeypatch, answer)

        assert cli._dashboard_is_listening(f"{LISTENER_ORIGIN}/") is True
        assert asked == [f"{LISTENER_ORIGIN}/healthz"]
        assert streams[0].reads == 0
        assert seen["trust_env"] is False
        assert seen["follow_redirects"] is False
        assert seen["timeout"] == cli.LIVENESS_TIMEOUT_SECONDS

    @pytest.mark.parametrize("status_code", [204, 302, 401, 404, 500])
    def test_a_status_other_than_200_means_nothing_is_listening(
        self, monkeypatch: pytest.MonkeyPatch, status_code: int
    ) -> None:
        streams, _seen = listen_with(monkeypatch, lambda _request: (status_code, [b"ignored"]))

        assert cli._dashboard_is_listening(f"{LISTENER_ORIGIN}/") is False
        assert streams[0].reads == 0

    def test_a_probe_whose_deadline_has_already_passed_asks_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "LIVENESS_TIMEOUT_SECONDS", 0)
        streams, _seen = listen_with(monkeypatch, lambda _request: (200, [b""]))

        assert cli._dashboard_is_listening(f"{LISTENER_ORIGIN}/") is False
        assert streams == []

    def test_a_probe_stalled_in_connection_setup_is_refused_at_the_deadline_and_a_late_200_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli, "LIVENESS_TIMEOUT_SECONDS", 0.1)
        released = threading.Event()
        answered = threading.Event()

        def late_200(_request: httpx2.Request) -> httpx2.Response:
            answered.set()
            return httpx2.Response(200)

        class StalledConnecting(httpx2.Client):
            def stream(self, method: str, url: str, **kwargs: Any) -> Any:
                released.wait(2)
                return super().stream(method, url, **kwargs)

        def build(**kwargs: Any) -> httpx2.Client:
            kwargs["transport"] = httpx2.MockTransport(late_200)
            return StalledConnecting(**kwargs)

        monkeypatch.setattr(cli.httpx2, "Client", build)
        started = time.monotonic()

        listening = cli._dashboard_is_listening(f"{LISTENER_ORIGIN}/")

        elapsed = time.monotonic() - started
        released.set()
        assert listening is False
        assert elapsed < 1
        assert answered.wait(2) is True

    def test_a_200_the_probe_timed_after_the_deadline_is_ignored_even_when_it_was_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        streams, _seen = listen_with(monkeypatch, lambda _request: (200, [b""]))
        times = [1_000.0, 1_000.0 + cli.LIVENESS_TIMEOUT_SECONDS + 0.01]

        def monotonic() -> float:
            return times.pop(0) if len(times) > 1 else times[0]

        assert cli._dashboard_is_listening(f"{LISTENER_ORIGIN}/", monotonic=monotonic) is False
        assert len(streams) == 1

    @pytest.mark.parametrize(
        "pieces",
        [
            pytest.param(
                [bytes([byte]) for byte in b"HTTP/1.1 200"],
                id="one byte of the status line at a time",
            ),
            pytest.param(
                [b"HTTP/1.1 100 Continue\r\n\r\n"] * 12,
                id="repeated informational responses",
            ),
        ],
    )
    def test_a_listener_that_stalls_before_the_final_headers_is_refused_at_the_deadline(
        self, monkeypatch: pytest.MonkeyPatch, pieces: list[bytes]
    ) -> None:
        pause = 0.05
        monkeypatch.setattr(cli, "LIVENESS_TIMEOUT_SECONDS", 0.15)
        url, sent, listener, server = stall_before_the_final_headers(pieces, pause)
        started = time.monotonic()
        try:
            assert cli._dashboard_is_listening(url) is False
            elapsed = time.monotonic() - started
            assert elapsed < 0.5
            assert sent["sent"] < len(pieces)
        finally:
            listener.close()
            server.join(timeout=2)

    def test_a_listener_that_is_not_there_is_not_listening(self) -> None:
        assert cli._dashboard_is_listening(f"http://127.0.0.1:{free_loopback_port()}/") is False

    @pytest.mark.parametrize(
        "failure",
        [
            pytest.param(ValueError("No closing quotation"), id="a BROWSER setting with a quote left open"),
            pytest.param(IndexError("list index out of range"), id="a BROWSER setting of nothing but spaces"),
            pytest.param(cli.webbrowser.Error("could not locate runnable browser"), id="no runnable browser"),
            pytest.param(OSError("Exec format error"), id="a browser that cannot be run"),
        ],
    )
    def test_open_reports_a_browser_that_fails_to_start_in_one_line(
        self,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        failure: Exception,
    ) -> None:

        def fail(_url: str) -> bool:
            raise failure

        monkeypatch.setattr(cli, "_dashboard_is_listening", lambda *_args: True)
        monkeypatch.setattr(cli.webbrowser, "open", fail)

        assert cli.main(["open"]) == 1
        assert capsys.readouterr().err == f"error: could not open a browser: {type(failure).__name__}\n"

    @pytest.mark.parametrize("command", ["collect", "serve"])
    def test_proxy_or_certificate_settings_the_client_cannot_use_are_one_line_errors(
        self,
        data_dir: Path,
        servers: list[ServerThatReturnsAtOnce],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        command: str,
    ) -> None:
        monkeypatch.setenv("SSL_CERT_FILE", str(data_dir.parent / "missing-ca.pem"))

        assert cli.main([command]) == 1

        assert capsys.readouterr().err.endswith(
            "error: cannot set up the HTTP client from the proxy or certificate settings: FileNotFoundError\n"
        )
        assert servers == []

    def test_serve_refuses_a_log_directory_whose_parent_another_account_can_change(
        self,
        data_dir: Path,
        servers: list[ServerThatReturnsAtOnce],
        log_file: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        logs = log_file.parent.parent
        logs.mkdir(parents=True)
        logs.chmod(0o777)
        try:
            assert cli.main(["serve"]) == 1
        finally:
            logs.chmod(0o700)

        assert "let other accounts write to it" in capsys.readouterr().err
        assert servers == []

    def test_open_reports_when_no_browser_can_be_started(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "_dashboard_is_listening", lambda *_args: True)
        monkeypatch.setattr(cli.webbrowser, "open", lambda _url: False)

        assert cli.main(["open"]) == 1
        assert "could not open a browser" in capsys.readouterr().err


class FailsToWrite:
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def write(self, _text: str) -> int:
        raise OSError(self.secret)

    def flush(self) -> None:
        return None


class TestALogHandlerThatFailsToWriteARecord:
    SECRET = "synthetic-secret-from-a-failure"

    def handle_while_a_failure_is_handled(self, handler: logging.Handler) -> None:
        record = logging.LogRecord("usagedash", logging.ERROR, __file__, 1, "failed: %s", (self.SECRET,), None)
        try:
            raise RuntimeError(self.SECRET)
        except RuntimeError:
            handler.handle(record)

    def test_standard_error_s_handler_names_only_the_type_of_its_failure(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        handler = cli.MessageFreeStreamHandler(FailsToWrite(self.SECRET))

        self.handle_while_a_failure_is_handled(handler)

        assert capsys.readouterr().err == "usagedash: a log record could not be written: OSError\n"

    def test_the_log_file_s_handler_names_only_the_type_of_a_rotation_that_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        handler = cli.PrivateRotatingFileHandler(logs / "usagedash.log", maxBytes=10, backupCount=2)
        try:
            handler.handle(logging.LogRecord("usagedash", logging.INFO, __file__, 1, "x" * 20, None, None))
            logs.chmod(0o500)
            self.handle_while_a_failure_is_handled(handler)
        finally:
            logs.chmod(0o700)
            handler.close()

        assert capsys.readouterr().err == "usagedash: a log record could not be written: PermissionError\n"

    def test_a_failure_to_report_that_failure_goes_no_further(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stderr", FailsToWrite(self.SECRET))
        handler = cli.MessageFreeStreamHandler(FailsToWrite(self.SECRET))

        self.handle_while_a_failure_is_handled(handler)


class TestTheServersLogFile:
    def handler(self, log_file: Path, max_bytes: int = 0) -> cli.PrivateRotatingFileHandler:
        handler = cli.PrivateRotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=2)
        handler.setFormatter(logging.Formatter("%(message)s"))
        return handler

    def record(self, message: str) -> logging.LogRecord:
        return logging.LogRecord("usagedash", logging.INFO, __file__, 1, message, None, None)

    @pytest.mark.parametrize("cleanup_error", [False, True])
    def test_log_emission_cannot_enter_until_descriptor_cleanup_finishes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_error: bool
    ) -> None:
        path = tmp_path / "usagedash.log"
        handler = self.handler(path)
        closing, release_close = threading.Event(), threading.Event()
        cleanup = handler._close_rotation_lock

        def pause_cleanup() -> None:
            closing.set()
            assert release_close.wait(5)
            cleanup()
            if cleanup_error:
                raise OSError("synthetic cleanup failure")

        def write_after_close() -> bool:
            entered = handler.lock.acquire(blocking=False)
            if entered:
                handler.lock.release()
            release_close.set()
            handler.handle(self.record("after close"))
            return entered

        monkeypatch.setattr(handler, "_close_rotation_lock", pause_cleanup)
        try:
            with ThreadPoolExecutor(max_workers=2) as workers:
                closed = workers.submit(handler.close)
                assert closing.wait(5)
                entered = workers.submit(write_after_close).result(timeout=5)
                if cleanup_error:
                    with pytest.raises(OSError, match="synthetic cleanup failure"):
                        closed.result(timeout=5)
                else:
                    closed.result(timeout=5)
            assert not entered, "a writer entered while close was still cleaning up its descriptors"
            assert path.read_text() == "after close\n"
        finally:
            release_close.set()
            monkeypatch.setattr(handler, "_close_rotation_lock", cleanup)
            handler.close()

    def test_two_processes_logging_to_the_same_file_do_not_overwrite_each_other_s_lines(self, tmp_path: Path) -> None:
        first, second = self.handler(tmp_path / "usagedash.log"), self.handler(tmp_path / "usagedash.log")
        try:
            for handler, message in ((first, "from the agent"), (second, "from a server started by hand")):
                handler.emit(self.record(message))
            first.emit(self.record("from the agent again"))
        finally:
            first.close()
            second.close()

        assert (tmp_path / "usagedash.log").read_text().splitlines() == [
            "from the agent",
            "from a server started by hand",
            "from the agent again",
        ]

    def test_a_record_after_another_process_rotated_the_file_goes_to_the_file_at_its_name(self, tmp_path: Path) -> None:
        log_file = tmp_path / "usagedash.log"
        agent_log, by_hand = self.handler(log_file, max_bytes=40), self.handler(log_file, max_bytes=40)
        try:
            for number in range(8):
                (agent_log if number % 2 == 0 else by_hand).emit(self.record(f"line {number} " + "x" * 30))
        finally:
            agent_log.close()
            by_hand.close()

        kept = {
            name: log_file.with_name(name).read_text()
            for name in ("usagedash.log", "usagedash.log.1", "usagedash.log.2")
        }
        assert kept == {
            "usagedash.log": "line 7 " + "x" * 30 + "\n",
            "usagedash.log.1": "line 6 " + "x" * 30 + "\n",
            "usagedash.log.2": "line 5 " + "x" * 30 + "\n",
        }

    def test_a_rotation_another_process_did_first_is_not_repeated(self, tmp_path: Path) -> None:
        log_file = tmp_path / "usagedash.log"
        agent_log, by_hand = self.handler(log_file, max_bytes=40), self.handler(log_file, max_bytes=40)
        try:
            agent_log.emit(self.record("line 0 " + "x" * 30))
            by_hand.doRollover()
            agent_log.doRollover()
            agent_log.emit(self.record("after"))
        finally:
            agent_log.close()
            by_hand.close()

        assert sorted(entry.name for entry in tmp_path.iterdir()) == [
            "usagedash.log",
            "usagedash.log.1",
            "usagedash.log.lock",
        ]
        assert log_file.read_text() == "after\n"
        assert log_file.with_name("usagedash.log.1").read_text() == "line 0 " + "x" * 30 + "\n"

    def test_a_link_at_the_log_s_name_after_another_process_rotated_is_refused_record_by_record(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_file = tmp_path / "logs" / "usagedash.log"
        log_file.parent.mkdir()
        target = tmp_path / "somewhere-else"
        target.write_text("not the dashboard's\n")
        handler = self.handler(log_file)
        try:
            handler.emit(self.record("line 0"))
            os.rename(log_file, log_file.with_name("usagedash.log.1"))
            os.symlink(target, log_file)
            handler.emit(self.record("refused"))
            handler.emit(self.record("refused again, through a stream the refusal closed"))
            os.unlink(log_file)
            handler.emit(self.record("line 1"))
        finally:
            handler.close()

        assert capsys.readouterr().err.splitlines() == ["usagedash: a log record could not be written: ConfigError"] * 2
        assert target.read_text() == "not the dashboard's\n"
        assert log_file.read_text() == "line 1\n"

    def test_the_lock_is_held_from_the_check_of_the_file_through_the_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_file = tmp_path / "usagedash.log"
        handler = self.handler(log_file)
        lock_file = log_file.with_name("usagedash.log.lock")
        seen: list[str] = []

        def locked_by_someone() -> bool:
            probe = os.open(lock_file, os.O_RDWR)
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            finally:
                os.close(probe)
            return False

        real_holds = handler._holds_the_file_at_its_name
        real_write = logging.FileHandler.emit

        def checking(stream: TextIOWrapper) -> bool:
            seen.append(f"check {locked_by_someone()}")
            return real_holds(stream)

        def writing(target: logging.FileHandler, record: logging.LogRecord) -> None:
            seen.append(f"write {locked_by_someone()}")
            real_write(target, record)

        monkeypatch.setattr(handler, "_holds_the_file_at_its_name", checking)
        monkeypatch.setattr(logging.FileHandler, "emit", writing)
        try:
            handler.emit(self.record("line 0"))
        finally:
            handler.close()

        assert seen == ["check True", "write True"]
        assert not locked_by_someone()
        assert log_file.read_text() == "line 0\n"

    def test_a_link_in_the_lock_file_s_place_refuses_the_record_once_the_handler_was_closed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_file = tmp_path / "usagedash.log"
        handler = self.handler(log_file)
        handler.close()
        lock_file = log_file.with_name("usagedash.log.lock")
        lock_file.unlink()
        lock_file.symlink_to(tmp_path / "elsewhere")
        try:
            handler.emit(self.record("refused"))
        finally:
            handler.close()
            handler.close()

        assert capsys.readouterr().err == "usagedash: a log record could not be written: ConfigError\n"
        assert log_file.read_text() == ""
        assert not (tmp_path / "elsewhere").exists()

    def test_a_closed_handler_opens_its_file_again_for_the_next_rotation_and_record(self, tmp_path: Path) -> None:
        log_file = tmp_path / "usagedash.log"
        handler = self.handler(log_file)
        try:
            handler.emit(self.record("line 0"))
            handler.close()
            handler.doRollover()
            handler.close()
            handler.emit(self.record("line 1"))
        finally:
            handler.close()

        assert log_file.read_text() == "line 1\n"
        assert log_file.with_name("usagedash.log.1").read_text() == "line 0\n"

    def test_each_file_a_rotation_leaves_is_owner_only(self, tmp_path: Path) -> None:
        log_file = tmp_path / "usagedash.log"
        handler = self.handler(log_file, max_bytes=40)
        try:
            for number in range(6):
                handler.emit(self.record(f"line {number} " + "x" * 30))
        finally:
            handler.close()

        files = sorted(tmp_path.iterdir())
        assert [entry.name for entry in files] == [
            "usagedash.log",
            "usagedash.log.1",
            "usagedash.log.2",
            "usagedash.log.lock",
        ]
        assert all(entry.is_file() and not entry.is_symlink() for entry in files)
        assert {entry.stat().st_mode & 0o777 for entry in files} == {0o600}
        assert log_file.read_text() == "line 5 " + "x" * 30 + "\n"

    def test_a_rotation_replaces_a_link_left_where_a_backup_goes_and_leaves_its_target_alone(
        self, tmp_path: Path
    ) -> None:
        log_file = tmp_path / "logs" / "usagedash.log"
        log_file.parent.mkdir()
        target = tmp_path / "somewhere-else"
        target.write_text("not the dashboard's\n")
        log_file.with_name("usagedash.log.1").symlink_to(target)
        handler = self.handler(log_file, max_bytes=40)
        try:
            for number in range(2):
                handler.emit(self.record(f"line {number} " + "x" * 30))
        finally:
            handler.close()

        backup = log_file.with_name("usagedash.log.1")
        assert target.read_text() == "not the dashboard's\n"
        assert (backup.is_symlink(), backup.read_text()) == (False, "line 0 " + "x" * 30 + "\n")

    def test_a_link_put_in_the_log_s_place_during_a_rotation_is_refused_when_the_log_is_opened_again(
        self, tmp_path: Path
    ) -> None:
        log_file = tmp_path / "logs" / "usagedash.log"
        log_file.parent.mkdir()
        target = tmp_path / "somewhere-else"
        target.write_text("not the dashboard's\n")
        handler = self.handler(log_file, max_bytes=40)

        def rotate_then_plant_a_link(source: str, destination: str) -> None:
            os.rename(source, destination)
            os.symlink(target, source)

        handler.rotator = rotate_then_plant_a_link
        try:
            handler.emit(self.record("line 0 " + "x" * 30))
            with pytest.raises(ConfigError, match=r"cannot open .*usagedash\.log as a private file"):
                handler.doRollover()
        finally:
            handler.close()

        assert target.read_text() == "not the dashboard's\n"
