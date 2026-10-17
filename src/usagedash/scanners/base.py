import bisect
import fnmatch
import hashlib
import json
import os
import stat
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Protocol

from usagedash.db import Database, FileCursor
from usagedash.models import UNATTRIBUTED_ACCOUNT_ID, Provider
from usagedash.timeutil import epoch_microseconds, in_utc

AccountResolver = Callable[[datetime], int]

UNKNOWN_MODEL = "unknown"
NANOSECONDS_PER_SECOND = 1_000_000_000
NANOSECONDS_PER_MICROSECOND = 1_000
ANCHOR_BYTES = 4096
ANCHOR_DIGEST_BYTES = 16
REREAD_CHUNK_BYTES = 1 << 20
MAX_TOKENS_PER_RECORD = 10**12


@dataclass(frozen=True, slots=True)
class ScanReport:
    files_read: int = 0
    files_failed: int = 0
    malformed_records: int = 0
    first_failure: str | None = None

    def merged(self, other: "ScanReport") -> "ScanReport":
        return ScanReport(
            files_read=self.files_read + other.files_read,
            files_failed=self.files_failed + other.files_failed,
            malformed_records=self.malformed_records + other.malformed_records,
            first_failure=self.first_failure or other.first_failure,
        )


class TokenScanner(Protocol):
    @property
    def provider(self) -> Provider: ...

    def scan(
        self, database: Database, resolve_account: AccountResolver, now: datetime, backfill_days: int
    ) -> ScanReport: ...


def isolated(name: str, scan_file: Callable[[], ScanReport]) -> ScanReport:
    try:
        return scan_file()
    except (OSError, ValueError, ArithmeticError, RecursionError) as error:
        return ScanReport(files_failed=1, first_failure=f"{name}: {type(error).__name__}")


def session_files(root: Path, name_pattern: str, depth: int | None = None) -> tuple[list[Path], ScanReport]:
    try:
        if not stat.S_ISDIR(root.stat().st_mode):
            return [], ScanReport()
    except (FileNotFoundError, NotADirectoryError):
        return [], ScanReport()
    except OSError as error:
        return [], _unlistable(root, error)

    report = ScanReport()
    found: list[Path] = []
    # Visited depth-first in listing order, as a stack, so a very deep tree cannot exhaust the call stack.
    pending = [(root, 0)]
    while pending:
        directory, below_root = pending.pop()
        try:
            with os.scandir(directory) as listed:
                entries = list(listed)
        except OSError as error:
            report = report.merged(_unlistable(directory, error))
            continue
        names: list[str] = []
        subdirectories: list[str] = []
        for entry in entries:
            try:
                # Why not the default: is_dir follows links and hides an error, so a locked directory vanishes.
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError as error:
                report = report.merged(_unlistable(directory / entry.name, error))
                continue
            if is_directory:
                subdirectories.append(entry.name)
            else:
                names.append(entry.name)
        if depth is None or below_root == depth:
            # Why directories too: one named like a session file must fail the scan, not disappear.
            for name in (*names, *subdirectories):
                if fnmatch.fnmatchcase(name, name_pattern):
                    found.append(directory / name)
        if depth is not None and below_root >= depth:
            continue
        for name in reversed(subdirectories):
            pending.append((directory / name, below_root + 1))
    return sorted(found), report


def _unlistable(directory: Path, error: OSError) -> ScanReport:
    return ScanReport(files_failed=1, first_failure=f"{directory.name}: {type(error).__name__}")


def named_model(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def build_account_resolver(timeline: Sequence[tuple[datetime, int | None]]) -> AccountResolver:
    starts = [started_at for started_at, _ in timeline]
    account_ids = [UNATTRIBUTED_ACCOUNT_ID if account_id is None else account_id for _, account_id in timeline]

    def resolve(moment: datetime) -> int:
        index = bisect.bisect_right(starts, moment) - 1
        return account_ids[index] if index >= 0 else UNATTRIBUTED_ACCOUNT_ID

    return resolve


class SessionFileChangedError(OSError):
    pass


def anchor_of(read: bytes) -> str:
    return hashlib.blake2b(read[-ANCHOR_BYTES:], digest_size=ANCHOR_DIGEST_BYTES).hexdigest()


def recorded_file_was_replaced(state: object, stat: os.stat_result) -> bool:
    if not isinstance(state, dict):
        return False
    saved = (state.get("device"), state.get("inode"))
    if not all(type(value) is int for value in saved):
        return False
    return saved != (stat.st_dev, stat.st_ino)


class OpenSessionFile:
    def __init__(self, handle: BinaryIO):
        self._handle = handle
        self._stat = os.fstat(handle.fileno())
        self.size_when_opened = self._stat.st_size
        self._start_offset = 0
        self._before_start = b""
        self._tail = b""
        self.end_offset = 0

    def resume(self, cursor: FileCursor | None, saved_anchor: object) -> bool:
        self._start_offset, self._before_start = 0, b""
        if cursor is None or not 0 <= cursor.byte_offset <= self._stat.st_size:
            return False
        if recorded_file_was_replaced(cursor.state, self._stat):
            return False
        before = self._before(cursor.byte_offset)
        if saved_anchor != anchor_of(before):
            return False
        self._start_offset, self._before_start = cursor.byte_offset, before
        return True

    def lines(self) -> Iterator[bytes]:
        """Yield the complete lines from the starting position, leaving ``end_offset`` after the last of them.

        A final line without a newline is still being written, so it is left unread and ``end_offset``
        stops before it; the next scan picks it up whole.

        Raises:
            SessionFileChangedError: after the last line, if the file no longer holds what was read: the
                bytes ``resume`` checked before the starting position, and every line yielded. The file
                was then rewritten in place between that check and the end of the read, and the lines
                may have come from more than one version of it. Appending to the file changes none of
                those bytes.
            OSError: if the file cannot be read.
        """
        self._tail = self._before_start
        self.end_offset = self._start_offset
        read = hashlib.sha256(self._before_start)
        self._handle.seek(self._start_offset)
        for line in self._handle:
            if not line.endswith(b"\n"):
                break
            self.end_offset += len(line)
            self._tail = (self._tail + line)[-ANCHOR_BYTES:]
            read.update(line)
            yield line
        if self._held_now(self._start_offset - len(self._before_start), self.end_offset) != read.digest():
            raise SessionFileChangedError("the file was rewritten while it was being read")

    @property
    def anchor(self) -> str:
        return anchor_of(self._tail)

    def position(self, state: object) -> FileCursor:
        carried = dict(state) if isinstance(state, dict) else {}
        carried.update(file_identity(self._stat))
        return FileCursor(
            byte_offset=self.end_offset,
            # Why not below the position: an append during the read is kept; a cursor past the size is damage.
            size=max(self._stat.st_size, self.end_offset),
            mtime_ns=self._stat.st_mtime_ns,
            state=carried,
        )

    def _before(self, offset: int) -> bytes:
        start = max(0, offset - ANCHOR_BYTES)
        return os.pread(self._handle.fileno(), offset - start, start)

    def _held_now(self, start: int, end: int) -> bytes:
        held = hashlib.sha256()
        offset = start
        while offset < end:
            chunk = os.pread(self._handle.fileno(), min(REREAD_CHUNK_BYTES, end - offset), offset)
            if not chunk:
                break
            held.update(chunk)
            offset += len(chunk)
        return held.digest()


@contextmanager
def open_session_file(path: Path) -> Generator[OpenSessionFile]:
    with path.open("rb") as handle:
        yield OpenSessionFile(handle)


def token_count(record: Mapping[str, object], field: str) -> int:
    """Read a token count from a record, as zero when the record leaves the field out.

    Raises:
        ValueError: if the field is present but is not an integer from zero to ``MAX_TOKENS_PER_RECORD``,
            ``null`` included.
    """
    if field not in record:
        return 0
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_TOKENS_PER_RECORD:
        raise ValueError("not a usable token count")
    return value


def parse_json_line(line: bytes) -> object:
    try:
        return json.loads(line)
    except RecursionError as error:
        raise ValueError("JSON is nested too deeply to parse") from error


def backfill_floor_ns(now: datetime, backfill_days: int) -> int:
    return epoch_microseconds(in_utc(now) - timedelta(days=backfill_days)) * NANOSECONDS_PER_MICROSECOND


def stable_backfill_floor_ns(stored: list[int], now: datetime, backfill_days: int) -> int:
    """Return the first poll's cutoff, so a file written after that poll is still enrolled."""
    if not stored:
        stored.append(backfill_floor_ns(now, backfill_days))
    return stored[0]


def file_identity(stat: os.stat_result) -> dict[str, int]:
    return {"device": stat.st_dev, "inode": stat.st_ino, "changed_ns": stat.st_ctime_ns}


def same_identity(state: object, *, device: int, inode: int, changed_ns: int) -> bool:
    if not isinstance(state, dict):
        return False
    expected = {"device": device, "inode": inode, "changed_ns": changed_ns}
    return all(type(state.get(key)) is int and state[key] == value for key, value in expected.items())


def names_file(state: object, stat: os.stat_result) -> bool:
    return same_identity(state, device=stat.st_dev, inode=stat.st_ino, changed_ns=stat.st_ctime_ns)


def needs_scan(cursor: FileCursor | None, stat: os.stat_result, floor_ns: int) -> bool:
    if cursor is None:
        return stat.st_mtime_ns >= floor_ns
    if not 0 <= cursor.byte_offset <= stat.st_size:
        return True
    if not names_file(cursor.state, stat):
        return True
    return cursor.size != stat.st_size or cursor.mtime_ns != stat.st_mtime_ns
