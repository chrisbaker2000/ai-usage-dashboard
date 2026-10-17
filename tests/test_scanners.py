import json
import logging
import os
import shutil
import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import pytest

from usagedash import db
from usagedash.db import Database, FileCursor, MessageUsage
from usagedash.models import MAX_TOKEN_COUNT, UNATTRIBUTED_ACCOUNT_ID, Provider, TokenCounts, UsageKey
from usagedash.paths import same_directory
from usagedash.scanners import base as scanner_base
from usagedash.scanners import codex as codex_scanner
from usagedash.scanners.base import (
    ANCHOR_BYTES,
    MAX_TOKENS_PER_RECORD,
    AccountResolver,
    OpenSessionFile,
    ScanReport,
    SessionFileChangedError,
    anchor_of,
    build_account_resolver,
    isolated,
    named_model,
    open_session_file,
    parse_json_line,
    session_files,
    token_count,
)
from usagedash.scanners.claude import ClaudeTranscriptScanner, parse_transcript_line
from usagedash.scanners.codex import CodexSessionScanner, normalize, usage_growth
from usagedash.scanners.grok import GrokSessionScanner, usage_from_file

BACKFILL_DAYS = 35
ACCOUNT_A = 7
ACCOUNT_B = 8


def everything_to(account_id: int) -> AccountResolver:
    return lambda _moment: account_id


def cursor_keys(database: Database) -> list[str]:
    with database.connect() as conn:
        return [row["file_key"] for row in conn.execute("SELECT file_key FROM scan_cursors ORDER BY file_key")]


def usage_rows(database: Database) -> dict[tuple[int, str, str], TokenCounts]:
    with database.connect() as conn:
        rows = conn.execute(
            "SELECT account_id, bucket_hour, model, SUM(input_tokens) AS i, SUM(cache_read_tokens) AS cr, "
            "SUM(cache_write_tokens) AS cw, SUM(output_tokens) AS o, SUM(reasoning_tokens) AS r, SUM(requests) AS n "
            "FROM token_usage GROUP BY account_id, bucket_hour, model"
        ).fetchall()
    return {
        (row["account_id"], row["bucket_hour"], row["model"]): TokenCounts(
            row["i"], row["cr"], row["cw"], row["o"], row["r"], row["n"]
        )
        for row in rows
    }


def total(database: Database) -> TokenCounts:
    result = TokenCounts()
    for counts in usage_rows(database).values():
        result = result.plus(counts)
    return result


def append(path: Path, *records: Any, newline: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for record in records:
            payload = record if isinstance(record, bytes) else json.dumps(record).encode()
            handle.write(payload + (b"\n" if newline else b""))


class TestAccountResolver:
    def test_resolves_an_instant_to_the_login_interval_containing_it(self, now: datetime) -> None:
        resolve = build_account_resolver([(now, ACCOUNT_A), (now + timedelta(hours=2), ACCOUNT_B)])

        assert resolve(now) == ACCOUNT_A
        assert resolve(now + timedelta(hours=2) - timedelta(seconds=1)) == ACCOUNT_A
        assert resolve(now + timedelta(hours=2)) == ACCOUNT_B
        assert resolve(now + timedelta(days=30)) == ACCOUNT_B

    def test_anything_before_tracking_began_is_unattributed(self, now: datetime) -> None:
        resolve = build_account_resolver([(now, ACCOUNT_A)])
        assert resolve(now - timedelta(seconds=1)) == UNATTRIBUTED_ACCOUNT_ID
        assert build_account_resolver([])(now) == UNATTRIBUTED_ACCOUNT_ID


class _Unclassifiable:
    def __init__(self, name: str) -> None:
        self.name = name

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        raise PermissionError(13, "Permission denied")


class _ScandirHiding:
    def __init__(self, raw: Iterator[os.DirEntry[str]], hidden_name: str) -> None:
        self._raw = raw
        self._hidden_name = hidden_name

    def __enter__(self) -> "_ScandirHiding":
        return self

    def __exit__(self, *_exc: object) -> bool:
        close = getattr(self._raw, "close", None)
        if close is not None:
            close()
        return False

    def __iter__(self) -> "_ScandirHiding":
        return self

    def __next__(self) -> os.DirEntry[str] | _Unclassifiable:
        entry = next(self._raw)
        if entry.name == self._hidden_name:
            return _Unclassifiable(entry.name)
        return entry


def _scandir_hiding(hidden_name: str) -> Callable[[str | os.PathLike[str]], _ScandirHiding]:
    real_scandir = os.scandir

    def scandir(path: str | os.PathLike[str]) -> _ScandirHiding:
        return _ScandirHiding(real_scandir(path), hidden_name)

    return scandir


def position_at(byte_offset: int) -> FileCursor:
    return FileCursor(byte_offset=byte_offset, size=byte_offset, mtime_ns=0, state=None)


def put_in_place_of(path: Path, contents: bytes) -> None:
    finished_copy = path.with_name(f"{path.name}.new")
    finished_copy.write_bytes(contents)
    finished_copy.replace(path)


def bytes_of(*records: Any) -> bytes:
    chunks: list[bytes] = []
    for record in records:
        payload = record if isinstance(record, bytes) else json.dumps(record).encode()
        chunks.append(payload + b"\n")
    return b"".join(chunks)


def with_a_tail_past_the_anchor(old: bytes, new: bytes) -> tuple[bytes, bytes]:
    if len(old) < len(new):
        old += b"x" * (len(new) - len(old))
    elif len(new) < len(old):
        new += b"x" * (len(old) - len(new))
    tail = b"\n" + (b"y" * (ANCHOR_BYTES + 1)) + b"\n"
    return old + tail, new + tail


def put_in_place_keeping_size_and_mtime(path: Path, contents: bytes) -> None:
    before = path.stat()
    assert len(contents) == before.st_size
    put_in_place_of(path, contents)
    os.utime(path, ns=(before.st_mtime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    assert after.st_ino != before.st_ino


def rewrite_in_place_keeping_mtime(path: Path, contents: bytes) -> None:
    before = path.stat()
    assert len(contents) == before.st_size
    path.write_bytes(contents)
    os.utime(path, ns=(before.st_mtime_ns, before.st_mtime_ns))
    after = path.stat()
    assert after.st_ino == before.st_ino
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    assert after.st_ctime_ns != before.st_ctime_ns


class TestOpenSessionFile:
    def session_file(self, tmp_path: Path, contents: bytes) -> Path:
        path = tmp_path / "log.jsonl"
        path.write_bytes(contents)
        return path

    def test_leaves_a_line_still_being_written_for_the_next_read(self, tmp_path: Path) -> None:
        path = self.session_file(tmp_path, b"one\ntwo\nthr")

        with open_session_file(path) as opened:
            lines = list(opened.lines())

            assert lines == [b"one\n", b"two\n"]
            assert opened.end_offset == 8
            assert opened.anchor == anchor_of(b"one\ntwo\n")

    def test_reads_on_from_a_saved_position_before_which_the_file_holds_what_was_read(self, tmp_path: Path) -> None:
        path = self.session_file(tmp_path, b"one\ntwo\n")

        with open_session_file(path) as opened:
            resumed = opened.resume(position_at(4), anchor_of(b"one\n"))

            assert resumed is True
            assert list(opened.lines()) == [b"two\n"]
            assert opened.end_offset == 8
            assert opened.anchor == anchor_of(b"one\ntwo\n")

    def test_a_position_at_the_end_of_the_file_is_read_on_from_and_gives_nothing_new(self, tmp_path: Path) -> None:
        path = self.session_file(tmp_path, b"one\ntwo\n")

        with open_session_file(path) as opened:
            resumed = opened.resume(position_at(8), anchor_of(b"one\ntwo\n"))

            assert resumed is True
            assert list(opened.lines()) == []
            assert (opened.end_offset, opened.anchor) == (8, anchor_of(b"one\ntwo\n"))

    @pytest.mark.parametrize(
        ("cursor", "saved_anchor"),
        [
            pytest.param(None, anchor_of(b"one\n"), id="nothing was saved"),
            pytest.param(position_at(4), anchor_of(b"eno\n"), id="the file held something else before the position"),
            pytest.param(position_at(4), None, id="no anchor was saved"),
            pytest.param(position_at(9), anchor_of(b"one\ntwo\n"), id="the position is past the end of the file"),
            pytest.param(position_at(-1), anchor_of(b""), id="the position is before the start of the file"),
        ],
    )
    def test_reads_from_the_start_when_the_saved_position_is_not_one_to_read_on_from(
        self, tmp_path: Path, cursor: FileCursor | None, saved_anchor: object
    ) -> None:
        path = self.session_file(tmp_path, b"one\ntwo\n")

        with open_session_file(path) as opened:
            resumed = opened.resume(cursor, saved_anchor)

            assert resumed is False
            assert list(opened.lines()) == [b"one\n", b"two\n"]
            assert opened.end_offset == 8

    @pytest.mark.parametrize("field", ["device", "inode"])
    def test_a_saved_device_or_inode_that_is_not_this_file_is_read_from_the_start(
        self, tmp_path: Path, field: str
    ) -> None:
        path = self.session_file(tmp_path, b"one\ntwo\n")
        stat = path.stat()
        identity = {"device": stat.st_dev, "inode": stat.st_ino, "changed_ns": stat.st_ctime_ns}
        identity[field] += 1
        cursor = FileCursor(byte_offset=8, size=8, mtime_ns=stat.st_mtime_ns, state=identity)

        with open_session_file(path) as opened:
            resumed = opened.resume(cursor, anchor_of(b"one\ntwo\n"))

            assert resumed is False
            assert list(opened.lines()) == [b"one\n", b"two\n"]

    def test_the_same_device_and_inode_is_read_on_from_when_the_anchor_matches(self, tmp_path: Path) -> None:
        path = self.session_file(tmp_path, b"one\ntwo\n")
        stat = path.stat()
        cursor = FileCursor(
            byte_offset=4,
            size=8,
            mtime_ns=0,
            state={"device": stat.st_dev, "inode": stat.st_ino, "changed_ns": stat.st_ctime_ns + 1},
        )

        with open_session_file(path) as opened:
            assert opened.resume(cursor, anchor_of(b"one\n")) is True
            assert list(opened.lines()) == [b"two\n"]

    @pytest.mark.parametrize("identity", [{"device": True, "inode": True}, {"anchor": "present"}])
    def test_an_identity_that_was_not_recorded_as_integers_does_not_by_itself_refuse_the_position(
        self, tmp_path: Path, identity: dict[str, object]
    ) -> None:
        path = self.session_file(tmp_path, b"one\ntwo\n")
        stat = path.stat()
        assert stat.st_dev != 1
        cursor = FileCursor(byte_offset=4, size=8, mtime_ns=stat.st_mtime_ns, state=identity)

        with open_session_file(path) as opened:
            assert opened.resume(cursor, anchor_of(b"one\n")) is True

    def test_a_position_refused_after_one_was_accepted_leaves_the_read_at_the_start(self, tmp_path: Path) -> None:
        path = self.session_file(tmp_path, b"one\ntwo\n")

        with open_session_file(path) as opened:
            opened.resume(position_at(4), anchor_of(b"one\n"))
            opened.resume(position_at(4), anchor_of(b"eno\n"))

            assert list(opened.lines()) == [b"one\n", b"two\n"]
            assert opened.anchor == anchor_of(b"one\ntwo\n")

    def test_the_anchor_covers_what_was_read_before_the_position_read_on_from(self, tmp_path: Path) -> None:
        before, appended = b"a" * 5000 + b"\n", b"short\n"
        path = self.session_file(tmp_path, before + appended)

        with open_session_file(path) as opened:
            opened.resume(position_at(len(before)), anchor_of(before))
            list(opened.lines())

            assert opened.anchor == anchor_of(before + appended)
            assert opened.anchor != anchor_of(appended)

    def test_the_anchor_is_of_at_most_the_last_four_kilobytes_read(self, tmp_path: Path) -> None:
        first, second = b"a" * 3000 + b"\n", b"b" * 3000 + b"\n"
        path = self.session_file(tmp_path, first + second)

        with open_session_file(path) as opened:
            list(opened.lines())

            assert opened.anchor == anchor_of((first + second)[-ANCHOR_BYTES:])
            assert opened.anchor != anchor_of(second)

    @pytest.mark.parametrize("change", ["rewritten in place", "replaced by another file"])
    def test_the_anchor_is_of_the_bytes_that_were_read_whatever_is_at_the_path_afterwards(
        self, tmp_path: Path, change: str
    ) -> None:
        path = self.session_file(tmp_path, b"one\ntwo\n")

        with open_session_file(path) as opened:
            list(opened.lines())
            if change == "rewritten in place":
                path.write_bytes(b"1st\n2nd\n3rd\n")
            else:
                put_in_place_of(path, b"1st\n2nd\n3rd\n")

            assert (opened.end_offset, opened.anchor) == (8, anchor_of(b"one\ntwo\n"))

    def test_a_file_rewritten_in_place_between_the_check_and_the_read_is_refused(self, tmp_path: Path) -> None:
        path = self.session_file(tmp_path, b"one\ntwo\n")

        with open_session_file(path) as opened:
            assert opened.resume(position_at(4), anchor_of(b"one\n")) is True
            path.write_bytes(b"1st\n2nd\n3rd\n")

            with pytest.raises(SessionFileChangedError):
                list(opened.lines())

    def test_a_file_rewritten_in_place_while_it_is_read_is_refused(self, tmp_path: Path) -> None:
        path = self.session_file(tmp_path, b"one\n" + b"two\n" * 50_000)

        with open_session_file(path) as opened:
            assert opened.resume(position_at(4), anchor_of(b"one\n")) is True
            lines = opened.lines()
            assert next(lines) == b"two\n"
            path.write_bytes(b"1st\n" + b"2nd\n" * 50_000)

            with pytest.raises(SessionFileChangedError):
                list(lines)

    def test_a_file_rewritten_in_place_while_it_is_read_from_the_start_is_refused(self, tmp_path: Path) -> None:
        path = self.session_file(tmp_path, b"one\n" * 50_000)

        with open_session_file(path) as opened:
            lines = opened.lines()
            assert next(lines) == b"one\n"
            path.write_bytes(b"1st\n" * 50_000)

            with pytest.raises(SessionFileChangedError):
                list(lines)

    def test_a_file_cut_short_of_what_was_read_while_it_is_read_is_refused(self, tmp_path: Path) -> None:
        path = self.session_file(tmp_path, b"one\ntwo\n")

        with open_session_file(path) as opened:
            lines = opened.lines()
            assert next(lines) == b"one\n"
            path.write_bytes(b"one\n")

            with pytest.raises(SessionFileChangedError):
                list(lines)

    def test_what_was_read_is_read_again_piece_by_piece_and_no_further_than_the_last_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(scanner_base, "REREAD_CHUNK_BYTES", 3)
        path = self.session_file(tmp_path, b"one\ntwo\nthree\nfo")

        with open_session_file(path) as opened:
            assert opened.resume(position_at(4), anchor_of(b"one\n")) is True

            assert list(opened.lines()) == [b"two\n", b"three\n"]
            assert opened.end_offset == 14

    def test_a_file_only_appended_to_while_it_is_read_is_not_refused(self, tmp_path: Path) -> None:
        path = self.session_file(tmp_path, b"one\ntwo\n")

        with open_session_file(path) as opened:
            assert opened.resume(position_at(4), anchor_of(b"one\n")) is True
            append(path, b"three")

            assert list(opened.lines()) == [b"two\n", b"three\n"]
            assert opened.anchor == anchor_of(b"one\ntwo\nthree\n")

    def test_a_file_put_in_its_place_is_not_read_and_the_file_that_was_opened_is_read_to_its_end(
        self, tmp_path: Path
    ) -> None:
        path = self.session_file(tmp_path, b"one\ntwo\n")

        with open_session_file(path) as opened:
            assert opened.resume(position_at(4), anchor_of(b"one\n")) is True
            put_in_place_of(path, b"1st\n2nd\n3rd\n")

            assert list(opened.lines()) == [b"two\n"]
            assert (opened.end_offset, opened.anchor) == (8, anchor_of(b"one\ntwo\n"))

    def test_the_position_to_save_describes_the_file_as_it_was_when_it_was_opened(self, tmp_path: Path) -> None:
        path = self.session_file(tmp_path, b"one\ntwo\nthr")
        os.utime(path, ns=(1_700_000_000_123_456_789, 1_700_000_000_123_456_789))
        opened_as = path.stat()

        with open_session_file(path) as opened:
            list(opened.lines())
            put_in_place_of(path, b"a file of another size and time\n")

            assert opened.position({"kept": "as given"}) == FileCursor(
                byte_offset=8,
                size=11,
                mtime_ns=1_700_000_000_123_456_789,
                state={"kept": "as given", **scanner_base.file_identity(opened_as)},
            )
        assert path.stat().st_ino != opened_as.st_ino

    def test_the_saved_size_is_never_less_than_the_position(self, tmp_path: Path) -> None:
        path = self.session_file(tmp_path, b"one\n")

        with open_session_file(path) as opened:
            append(path, b"two")
            list(opened.lines())

            saved = opened.position(None)

        assert (saved.byte_offset, saved.size) == (8, 8)

    def test_the_file_is_closed_when_the_scan_of_it_is_over(self, tmp_path: Path) -> None:
        path = self.session_file(tmp_path, b"one\n")

        with open_session_file(path) as opened:
            pass

        with pytest.raises(ValueError, match="closed file"):
            list(opened.lines())

    def test_what_a_file_held_before_a_position_is_told_from_at_most_its_last_four_kilobytes(self) -> None:
        read = b"a" * 5000 + b"b" * 100
        same_last_four_kilobytes = b"z" * (5100 - ANCHOR_BYTES) + read[-ANCHOR_BYTES:]
        another_byte_in_them = b"a" * 5000 + b"c" + b"b" * 99

        assert len(anchor_of(read)) == 32
        assert anchor_of(same_last_four_kilobytes) == anchor_of(read)
        assert anchor_of(another_byte_in_them) != anchor_of(read)
        assert anchor_of(read[:-1]) != anchor_of(read)


class TestTokenCountValidation:
    def test_a_field_the_record_leaves_out_is_zero(self) -> None:
        assert token_count({"other": 5}, "input_tokens") == 0

    @pytest.mark.parametrize(("value", "expected"), [(0, 0), (17, 17), (MAX_TOKENS_PER_RECORD, MAX_TOKENS_PER_RECORD)])
    def test_accepts_integers_from_zero_to_the_largest_believable_count(self, value: object, expected: int) -> None:
        assert token_count({"input_tokens": value}, "input_tokens") == expected

    @pytest.mark.parametrize("value", [None, -1, 1.5, "12", True, MAX_TOKENS_PER_RECORD + 1, 2**63, 10**400])
    def test_rejects_anything_else_instead_of_coercing_it(self, value: object) -> None:
        with pytest.raises(ValueError, match="not a usable token count"):
            token_count({"input_tokens": value}, "input_tokens")

    def test_a_million_records_at_the_limit_still_fit_one_stored_row(self) -> None:
        assert MAX_TOKENS_PER_RECORD * 1_000_000 == MAX_TOKEN_COUNT


def claude_line(
    message_id: str,
    timestamp: datetime,
    *,
    output: int = 100,
    input_tokens: int = 10,
    cache_read: int = 1000,
    cache_write: int = 50,
    thinking: int = 0,
    model: str = "claude-model-x",
    request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "assistant",
        "uuid": f"line-{message_id}",
        "requestId": request_id or f"req-{message_id}",
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "message": {
            "id": message_id,
            "model": model,
            "usage": {
                "input_tokens": input_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write,
                "output_tokens": output,
                "output_tokens_details": {"thinking_tokens": thinking},
            },
        },
    }


class TestClaudeTranscriptScanner:
    def scan(self, database: Database, projects: Path, now: datetime, resolver: AccountResolver | None = None) -> Any:
        return ClaudeTranscriptScanner(projects).scan(
            database, resolver or everything_to(ACCOUNT_A), now, BACKFILL_DAYS
        )

    def test_counts_each_api_response_once_however_many_lines_repeat_it(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        append(
            transcript,
            {"type": "user", "message": {"content": "hi"}},
            claude_line("msg-1", now, thinking=40),
            claude_line("msg-1", now, thinking=40),
            claude_line("msg-1", now, thinking=40),
            claude_line("msg-2", now + timedelta(hours=1)),
        )

        report = self.scan(database, tmp_path / "projects", now)

        assert report.files_read == 1
        assert usage_rows(database) == {
            (ACCOUNT_A, "2026-10-03T14", "claude-model-x"): TokenCounts(10, 1000, 50, 100, 40, 1),
            (ACCOUNT_A, "2026-10-03T15", "claude-model-x"): TokenCounts(10, 1000, 50, 100, 0, 1),
        }

    @pytest.mark.parametrize(
        ("path", "field"),
        [
            pytest.param((), "input_tokens", id="input"),
            pytest.param((), "cache_read_input_tokens", id="cache read"),
            pytest.param((), "cache_creation_input_tokens", id="cache write"),
            pytest.param((), "output_tokens", id="output"),
            pytest.param(("output_tokens_details",), "thinking_tokens", id="thinking"),
        ],
    )
    def test_a_line_that_states_a_counter_as_null_is_malformed_and_not_counted_as_zero(
        self, database: Database, tmp_path: Path, now: datetime, path: tuple[str, ...], field: str
    ) -> None:
        line = claude_line("msg-1", now)
        container = line["message"]["usage"]
        for key in path:
            container = container[key]
        container[field] = None
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        append(transcript, line, claude_line("msg-2", now))

        report = self.scan(database, tmp_path / "projects", now)

        assert (report.files_read, report.malformed_records) == (1, 1)
        assert total(database).requests == 1

    def test_a_line_that_leaves_a_counter_out_counts_it_as_zero(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        line = claude_line("msg-1", now)
        del line["message"]["usage"]["cache_creation_input_tokens"]
        append(tmp_path / "projects" / "proj" / "session.jsonl", line)

        report = self.scan(database, tmp_path / "projects", now)

        assert report.malformed_records == 0
        assert total(database) == TokenCounts(10, 1000, 0, 100, 0, 1)

    def test_history_copied_into_a_resumed_session_file_is_not_counted_again(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        projects = tmp_path / "projects"
        append(projects / "proj" / "original.jsonl", claude_line("msg-1", now))
        self.scan(database, projects, now)
        append(projects / "proj" / "resumed.jsonl", claude_line("msg-1", now), claude_line("msg-2", now))

        self.scan(database, projects, now)

        assert total(database).requests == 2
        assert total(database).output_tokens == 200

    def test_a_later_line_reporting_more_output_adds_only_the_difference(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        append(transcript, claude_line("msg-1", now, output=30))
        self.scan(database, tmp_path / "projects", now)
        append(transcript, claude_line("msg-1", now, output=100))

        self.scan(database, tmp_path / "projects", now)

        assert total(database) == TokenCounts(10, 1000, 50, 100, 0, 1)

    def test_only_reads_what_was_appended_since_the_last_scan(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        append(transcript, claude_line("msg-1", now))
        assert self.scan(database, tmp_path / "projects", now).files_read == 1
        assert self.scan(database, tmp_path / "projects", now).files_read == 0

        append(transcript, claude_line("msg-2", now))

        assert self.scan(database, tmp_path / "projects", now).files_read == 1
        assert total(database).requests == 2

    def test_a_partially_written_last_line_is_picked_up_once_complete(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        whole = json.dumps(claude_line("msg-1", now)).encode()
        append(transcript, whole[:40], newline=False)
        self.scan(database, tmp_path / "projects", now)
        assert total(database).requests == 0

        append(transcript, whole[40:])

        self.scan(database, tmp_path / "projects", now)
        assert total(database).requests == 1

    def test_a_rewritten_shorter_file_is_reread_without_double_counting(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        append(transcript, claude_line("msg-1", now), claude_line("msg-2", now))
        self.scan(database, tmp_path / "projects", now)
        transcript.write_bytes(json.dumps(claude_line("msg-1", now)).encode() + b"\n")

        self.scan(database, tmp_path / "projects", now)

        assert total(database).requests == 2

    def test_attributes_each_message_to_the_account_logged_in_when_it_was_sent(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        switch = now + timedelta(hours=1)
        resolver = build_account_resolver([(now, ACCOUNT_A), (switch, ACCOUNT_B)])
        append(
            tmp_path / "projects" / "proj" / "session.jsonl",
            claude_line("before-tracking", now - timedelta(minutes=1)),
            claude_line("on-a", now + timedelta(minutes=30)),
            claude_line("on-b", switch),
        )

        self.scan(database, tmp_path / "projects", now, resolver)

        assert sorted(account for account, _, _ in usage_rows(database)) == [
            UNATTRIBUTED_ACCOUNT_ID,
            ACCOUNT_A,
            ACCOUNT_B,
        ]

    def test_includes_subagent_transcripts_in_nested_directories(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        projects = tmp_path / "projects"
        append(projects / "proj" / "session.jsonl", claude_line("main", now))
        append(projects / "proj" / "session" / "subagents" / "agent-1.jsonl", claude_line("sub", now))

        assert self.scan(database, projects, now).files_read == 2
        assert total(database).requests == 2

    def test_skips_files_untouched_since_before_the_backfill_window(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        old = tmp_path / "projects" / "proj" / "old.jsonl"
        append(old, claude_line("msg-old", now))
        stale = (now - timedelta(days=BACKFILL_DAYS + 1)).timestamp()
        os.utime(old, (stale, stale))

        assert self.scan(database, tmp_path / "projects", now).files_read == 0
        assert total(database).requests == 0

    def test_ignores_synthetic_zero_usage_messages_and_non_assistant_lines(self) -> None:
        zero = claude_line(
            "synthetic", datetime(2026, 10, 3, tzinfo=UTC), output=0, input_tokens=0, cache_read=0, cache_write=0
        )
        assert parse_transcript_line(json.dumps(zero).encode()) is None
        assert parse_transcript_line(b'{"type":"user","message":{"usage":{"input_tokens":5}}}') is None
        assert parse_transcript_line(b'{"type":"summary","summary":"no usage here"}') is None

    def test_a_response_whose_only_usage_is_reasoning_tokens_is_not_taken_for_a_zero_usage_message(
        self, now: datetime
    ) -> None:
        reasoning_only = claude_line("msg-1", now, output=0, input_tokens=0, cache_read=0, cache_write=0, thinking=40)

        message = parse_transcript_line(json.dumps(reasoning_only).encode())

        assert message is not None
        assert message.counts == TokenCounts(reasoning_tokens=40, requests=1)

    def test_counts_malformed_usage_records_without_stopping(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        negative = claude_line("bad", now)
        negative["message"]["usage"]["output_tokens"] = -5
        no_timestamp = claude_line("no-time", now)
        del no_timestamp["timestamp"]
        append(
            tmp_path / "projects" / "proj" / "session.jsonl",
            negative,
            no_timestamp,
            b'{"type":"assistant","message":{"usage": truncated',
            claude_line("good", now),
        )

        report = self.scan(database, tmp_path / "projects", now)

        assert report.malformed_records == 3
        assert total(database).requests == 1

    def test_a_missing_projects_directory_is_an_empty_scan(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        report = self.scan(database, tmp_path / "absent", now)
        assert (report.files_read, report.files_failed) == (0, 0)

    def test_scanning_the_same_projects_directory_under_another_spelling_does_not_count_it_again(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        projects = tmp_path / "home" / "projects"
        append(projects / "proj" / "session.jsonl", claude_line("msg-1", now))

        self.scan(database, projects, now)
        counted = total(database)
        self.scan(database, tmp_path / "home" / ".." / "home" / "projects", now)

        assert counted.requests == 1
        assert total(database) == counted
        assert len(cursor_keys(database)) == 1


def codex_meta(account_id: str | None) -> dict[str, Any]:
    return {"type": "session_meta", "payload": {"id": "s", "creator_account_id": account_id}}


def codex_turn(model: str) -> dict[str, Any]:
    return {"type": "turn_context", "payload": {"model": model}}


def codex_totals(input_tokens: int, cached: int, output: int, reasoning: int = 0) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": 0,
        "output_tokens": output,
        "reasoning_output_tokens": reasoning,
        "total_tokens": input_tokens + output,
    }


def codex_count(timestamp: datetime, total_usage: dict[str, int], last: dict[str, int] | None) -> dict[str, Any]:
    info: dict[str, Any] = {"total_token_usage": total_usage}
    if last is not None:
        info["last_token_usage"] = last
    return {
        "type": "event_msg",
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "payload": {"type": "token_count", "info": info, "rate_limits": {}},
    }


def without(totals: dict[str, int], field: str) -> dict[str, int]:
    del totals[field]
    return totals


def undated(record: dict[str, Any]) -> dict[str, Any]:
    del record["timestamp"]
    return record


def dated_without_an_offset(record: dict[str, Any]) -> dict[str, Any]:
    record["timestamp"] = "2026-10-03T14:00:00"
    return record


def with_a_negative_request_count(record: dict[str, Any]) -> dict[str, Any]:
    record["payload"]["info"]["last_token_usage"]["output_tokens"] = -1
    return record


class TestCodexUsageGrowth:
    def test_growth_is_the_difference_between_consecutive_running_totals(self) -> None:
        growth = usage_growth(codex_totals(150, 100, 30), codex_totals(999, 999, 999), codex_totals(100, 80, 10))
        assert (growth["input_tokens"], growth["cached_input_tokens"], growth["output_tokens"]) == (50, 20, 20)

    def test_the_first_event_in_a_file_uses_its_per_request_usage_not_the_carried_over_total(self) -> None:
        growth = usage_growth(codex_totals(5000, 4000, 900), codex_totals(40, 30, 5), None)
        assert (growth["input_tokens"], growth["output_tokens"]) == (40, 5)

    def test_totals_that_shrink_fall_back_to_per_request_usage(self) -> None:
        growth = usage_growth(codex_totals(20, 0, 5), codex_totals(20, 0, 5), codex_totals(900, 800, 300))
        assert (growth["input_tokens"], growth["output_tokens"]) == (20, 5)

    def test_without_a_baseline_or_per_request_usage_nothing_is_counted(self) -> None:
        assert set(usage_growth(codex_totals(500, 0, 5), None, None).values()) == {0}

    def test_normalizing_separates_cached_input_from_fresh_input(self) -> None:
        counts = normalize(codex_totals(150, 100, 30, reasoning=12))
        assert counts == TokenCounts(
            input_tokens=50, cache_read_tokens=100, output_tokens=30, reasoning_tokens=12, requests=1
        )
        assert normalize(codex_totals(0, 0, 0)).requests == 0


class TestCodexSessionScanner:
    def scan(self, database: Database, home: Path, now: datetime) -> Any:
        return CodexSessionScanner(home).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

    def account_ids(self, database: Database) -> dict[str, int]:
        with database.connect() as conn:
            return {account.external_id: account.id for account in db.list_accounts(conn)}

    def rollout(self, home: Path, name: str = "rollout-2026-10-03T10-00-00-aaaa.jsonl") -> Path:
        return home / "sessions" / "2026" / "10" / "03" / name

    def test_attributes_usage_to_the_workspace_named_in_the_session_and_counts_growth(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(
            self.rollout(tmp_path),
            codex_meta("workspace-a"),
            codex_turn("gpt-x"),
            codex_count(now, codex_totals(100, 80, 10), codex_totals(100, 80, 10)),
            codex_count(now, codex_totals(100, 80, 10), codex_totals(100, 80, 10)),
            codex_count(now + timedelta(hours=1), codex_totals(250, 200, 40, 7), codex_totals(150, 120, 30, 7)),
        )

        self.scan(database, tmp_path, now)

        workspace = self.account_ids(database)["workspace-a"]
        assert usage_rows(database) == {
            (workspace, "2026-10-03T14", "gpt-x"): TokenCounts(20, 80, 0, 10, 0, 1),
            (workspace, "2026-10-03T15", "gpt-x"): TokenCounts(30, 120, 0, 30, 7, 1),
        }

    def test_a_workspace_never_seen_logged_in_gets_a_placeholder_account(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(
            self.rollout(tmp_path),
            codex_meta("0123456789abcdef"),
            codex_count(now, codex_totals(5, 0, 1), codex_totals(5, 0, 1)),
        )

        self.scan(database, tmp_path, now)

        with database.connect() as conn:
            account = db.list_accounts(conn)[0]
        assert (account.provider, account.external_id, account.label) == (
            Provider.CODEX,
            "0123456789abcdef",
            "Codex workspace 01234567",
        )

    def test_a_session_without_a_workspace_is_unattributed(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(self.rollout(tmp_path), codex_meta(None), codex_count(now, codex_totals(5, 0, 1), codex_totals(5, 0, 1)))

        self.scan(database, tmp_path, now)

        assert [account for account, _, _ in usage_rows(database)] == [UNATTRIBUTED_ACCOUNT_ID]

    def test_resuming_a_scan_continues_from_the_stored_totals_model_and_workspace(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.rollout(tmp_path)
        append(
            rollout, codex_meta("workspace-a"), codex_turn("gpt-x"), codex_count(now, codex_totals(100, 0, 10), None)
        )
        self.scan(database, tmp_path, now)
        append(rollout, codex_count(now, codex_totals(160, 0, 25), codex_totals(999, 0, 999)))

        self.scan(database, tmp_path, now)

        workspace = self.account_ids(database)["workspace-a"]
        assert usage_rows(database) == {(workspace, "2026-10-03T14", "gpt-x"): TokenCounts(60, 0, 0, 15, 0, 1)}

    def test_a_session_moved_into_the_archive_is_not_counted_twice(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.rollout(tmp_path)
        append(rollout, codex_meta("workspace-a"), codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)))
        self.scan(database, tmp_path, now)
        archived = tmp_path / "archived_sessions" / rollout.name
        archived.parent.mkdir(parents=True)
        rollout.rename(archived)
        os.utime(archived, None)

        self.scan(database, tmp_path, now)

        assert total(database) == TokenCounts(100, 0, 0, 10, 0, 1)

    def test_a_rewritten_shorter_file_replaces_its_earlier_contribution(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.rollout(tmp_path)
        append(
            rollout,
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
            codex_count(now, codex_totals(300, 0, 30), codex_totals(200, 0, 20)),
        )
        self.scan(database, tmp_path, now)
        rollout.write_bytes(b"")
        append(rollout, codex_meta("workspace-a"), codex_count(now, codex_totals(40, 0, 4), codex_totals(40, 0, 4)))

        self.scan(database, tmp_path, now)

        assert total(database) == TokenCounts(40, 0, 0, 4, 0, 1)

    def test_events_without_usage_info_and_malformed_events_add_nothing(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        broken = codex_count(now, codex_totals(5, 0, 1), None)
        broken["payload"]["info"]["total_token_usage"]["input_tokens"] = "five"
        append(
            self.rollout(tmp_path),
            codex_meta("workspace-a"),
            {
                "type": "event_msg",
                "timestamp": "2026-10-03T14:00:00Z",
                "payload": {"type": "token_count", "info": None},
            },
            broken,
            b'{"type":"event_msg","payload":{"type":"token_count" truncated',
        )

        report = self.scan(database, tmp_path, now)

        assert report.malformed_records == 2
        assert total(database) == TokenCounts()

    @pytest.mark.parametrize("damage", [undated, dated_without_an_offset, with_a_negative_request_count])
    def test_usage_carried_by_a_malformed_event_is_counted_by_the_next_valid_one(
        self, database: Database, tmp_path: Path, now: datetime, damage: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> None:
        append(
            self.rollout(tmp_path),
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
            damage(codex_count(now, codex_totals(200, 0, 20), codex_totals(100, 0, 10))),
            codex_count(now, codex_totals(250, 0, 25), codex_totals(50, 0, 5)),
        )

        report = self.scan(database, tmp_path, now)

        assert report.malformed_records == 1
        assert total(database) == TokenCounts(input_tokens=250, output_tokens=25, requests=2)

    @pytest.mark.parametrize(
        "field", ["input_tokens", "cached_input_tokens", "output_tokens", "cache_write_input_tokens"]
    )
    def test_a_first_request_whose_own_usage_states_a_counter_as_null_is_malformed_and_not_counted_as_zero(
        self, database: Database, tmp_path: Path, now: datetime, field: str
    ) -> None:
        own_usage: dict[str, Any] = codex_totals(100, 0, 10)
        own_usage[field] = None
        rollout = self.rollout(tmp_path)
        append(rollout, codex_meta("workspace-a"), codex_count(now, codex_totals(100, 0, 10), own_usage))

        report = self.scan(database, tmp_path, now)

        assert report.malformed_records == 1
        assert total(database) == TokenCounts()

    def test_usage_carried_by_a_malformed_last_event_is_recovered_by_the_next_scan(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.rollout(tmp_path)
        append(
            rollout,
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
            undated(codex_count(now, codex_totals(200, 0, 20), codex_totals(100, 0, 10))),
        )
        self.scan(database, tmp_path, now)
        assert total(database) == TokenCounts(input_tokens=100, output_tokens=10, requests=1)

        append(rollout, codex_count(now, codex_totals(250, 0, 25), codex_totals(50, 0, 5)))
        self.scan(database, tmp_path, now)

        assert total(database) == TokenCounts(input_tokens=250, output_tokens=25, requests=2)

    @pytest.mark.parametrize("left_out_as", ["absent", "null"])
    @pytest.mark.parametrize(
        "field", ["input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"]
    )
    def test_running_totals_that_leave_out_a_counter_are_malformed_and_never_become_the_baseline(
        self, database: Database, tmp_path: Path, now: datetime, field: str, left_out_as: str
    ) -> None:
        incomplete: dict[str, Any] = codex_totals(200, 0, 20)
        if left_out_as == "absent":
            del incomplete[field]
        else:
            incomplete[field] = None
        append(
            self.rollout(tmp_path),
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
            codex_count(now, incomplete, codex_totals(100, 0, 10)),
            codex_count(now, codex_totals(250, 0, 25), codex_totals(50, 0, 5)),
        )

        report = self.scan(database, tmp_path, now)

        assert report.malformed_records == 1
        assert total(database) == TokenCounts(input_tokens=250, output_tokens=25, requests=2)

    def test_a_rollout_from_before_codex_counted_cache_writes_is_read_with_that_counter_as_zero(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        field = "cache_write_input_tokens"
        append(
            self.rollout(tmp_path),
            codex_meta("workspace-a"),
            codex_count(now, without(codex_totals(100, 0, 10), field), without(codex_totals(100, 0, 10), field)),
            codex_count(now, without(codex_totals(250, 0, 25), field), without(codex_totals(150, 0, 15), field)),
        )

        report = self.scan(database, tmp_path, now)

        assert report.malformed_records == 0
        assert total(database) == TokenCounts(input_tokens=250, output_tokens=25, requests=2)

    @pytest.mark.parametrize("left_out_as", ["absent", "null"])
    def test_running_totals_that_stop_stating_cache_writes_the_file_had_counted_are_malformed(
        self, database: Database, tmp_path: Path, now: datetime, left_out_as: str
    ) -> None:
        field = "cache_write_input_tokens"
        counting = codex_totals(100, 0, 10) | {field: 40}
        silent: dict[str, Any] = codex_totals(200, 0, 20)
        if left_out_as == "absent":
            del silent[field]
        else:
            silent[field] = None
        append(
            self.rollout(tmp_path),
            codex_meta("workspace-a"),
            codex_count(now, counting, counting),
            codex_count(now, silent, codex_totals(100, 0, 10)),
            codex_count(now, codex_totals(250, 0, 25) | {field: 70}, codex_totals(50, 0, 5)),
        )

        report = self.scan(database, tmp_path, now)

        assert report.malformed_records == 1
        assert total(database) == TokenCounts(input_tokens=250, cache_write_tokens=70, output_tokens=25, requests=2)

    def test_one_request_s_own_usage_may_leave_out_a_counter_which_is_then_zero(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(
            self.rollout(tmp_path),
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(5000, 4000, 900), {"input_tokens": 40, "output_tokens": 5}),
        )

        report = self.scan(database, tmp_path, now)

        assert report.malformed_records == 0
        assert total(database) == TokenCounts(input_tokens=40, output_tokens=5, requests=1)

    def test_skips_sessions_untouched_since_before_the_backfill_window(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.rollout(tmp_path)
        append(rollout, codex_meta("workspace-a"), codex_count(now, codex_totals(5, 0, 1), codex_totals(5, 0, 1)))
        stale = (now - timedelta(days=BACKFILL_DAYS + 1)).timestamp()
        os.utime(rollout, (stale, stale))

        assert self.scan(database, tmp_path, now).files_read == 0

    def test_an_unchanged_rollout_is_not_reread(self, database: Database, tmp_path: Path, now: datetime) -> None:
        append(
            self.rollout(tmp_path),
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(5, 0, 1), codex_totals(5, 0, 1)),
        )
        assert self.scan(database, tmp_path, now).files_read == 1

        assert self.scan(database, tmp_path, now).files_read == 0

    def test_scanning_the_same_home_under_another_spelling_does_not_count_it_again(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        home = tmp_path / "home"
        append(
            self.rollout(home),
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(5, 0, 1), codex_totals(5, 0, 1)),
        )

        self.scan(database, home, now)
        counted = total(database)
        self.scan(database, home / ".." / "home", now)

        assert counted.requests == 1
        assert total(database) == counted
        assert cursor_keys(database) == [str(home.resolve() / "rollout-2026-10-03T10-00-00-aaaa.jsonl")]

    def test_scanning_the_same_home_under_another_capitalization_does_not_count_it_again(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        home = tmp_path / "Home"
        other = tmp_path / "HOME"
        append(
            self.rollout(home),
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(5, 0, 1), codex_totals(5, 0, 1)),
        )
        if not same_directory(home, other):
            pytest.skip("this filesystem stores the two spellings as different directories")

        self.scan(database, home, now)
        counted = total(database)
        self.scan(database, other, now)

        assert total(database) == counted
        assert len(cursor_keys(database)) == 1


class TestAClaudeTranscriptCopiedIntoAnotherHome:
    def homes(self, tmp_path: Path, now: datetime) -> tuple[Path, Path]:
        original = tmp_path / "home-a" / "projects"
        copy = tmp_path / "home-b" / "projects"
        for projects in (original, copy):
            append(projects / "proj" / "session.jsonl", claude_line("msg-1", now, output=100))
        return original, copy

    def scan(self, database: Database, projects: Path, account: int, now: datetime) -> None:
        ClaudeTranscriptScanner(projects).scan(database, everything_to(account), now, BACKFILL_DAYS)

    @pytest.mark.parametrize("copy_is_read", ["after the original", "before the original"])
    def test_the_copy_does_not_take_the_usage_from_the_account_it_was_credited_to(
        self, database: Database, tmp_path: Path, now: datetime, copy_is_read: str
    ) -> None:
        original, copy = self.homes(tmp_path, now)
        reads = [(original, ACCOUNT_A), (copy, UNATTRIBUTED_ACCOUNT_ID)]
        if copy_is_read == "before the original":
            reads.reverse()

        credited = []
        for _cycle in range(2):
            for projects, account in reads:
                self.scan(database, projects, account, now)
            credited.append(
                {account: counts.output_tokens for (account, _hour, _model), counts in usage_rows(database).items()}
            )

        assert credited == [{ACCOUNT_A: 100}] * 2

    def test_the_copy_is_one_response_and_not_a_second_one(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        original, copy = self.homes(tmp_path, now)

        self.scan(database, original, ACCOUNT_A, now)
        self.scan(database, copy, ACCOUNT_A, now)

        assert (total(database).output_tokens, total(database).requests) == (100, 1)


class CodexCopies:
    NAME = "rollout-2026-10-03T10-00-00-aaaa.jsonl"

    def requests(self, now: datetime, first: int, last: int) -> list[dict[str, Any]]:
        return [
            codex_count(now, codex_totals(100 * number, 0, 10 * number), codex_totals(100, 0, 10))
            for number in range(first, last + 1)
        ]

    def write(self, rollout: Path, now: datetime, requests: int) -> Path:
        rollout.parent.mkdir(parents=True, exist_ok=True)
        rollout.write_bytes(b"")
        append(rollout, codex_meta("workspace-a"), *self.requests(now, 1, requests))
        return rollout

    def active(self, home: Path) -> Path:
        return home / "sessions" / self.NAME

    def archived(self, home: Path) -> Path:
        return home / "archived_sessions" / self.NAME

    def scan(self, database: Database, now: datetime, *homes: Path) -> list[ScanReport]:
        return [
            CodexSessionScanner(home).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS) for home in homes
        ]

    def counted(self, requests: int) -> TokenCounts:
        return TokenCounts(input_tokens=100 * requests, output_tokens=10 * requests, requests=requests)

    def read_up_to(self, database: Database, home: Path) -> int:
        with database.read() as conn:
            return db.load_cursors(conn, Provider.CODEX)[str(home / self.NAME)].byte_offset

    def totals_over(self, database: Database, now: datetime, cycles: int, *homes: Path) -> list[TokenCounts]:
        totals = []
        for _cycle in range(cycles):
            self.scan(database, now, *homes)
            totals.append(total(database))
        return totals


class TestACodexRewriteAfterPrune(CodexCopies):
    def test_a_same_size_rewrite_of_only_expired_requests_does_not_drop_the_one_retention_kept(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        expired = now - timedelta(days=401)
        rollout = self.active(tmp_path)
        rollout.parent.mkdir(parents=True, exist_ok=True)
        rollout.write_bytes(b"")
        append(
            rollout,
            codex_meta("workspace-a"),
            codex_count(expired, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
            codex_count(expired, codex_totals(200, 0, 20), codex_totals(100, 0, 10)),
            codex_count(now, codex_totals(300, 0, 30), codex_totals(100, 0, 10)),
        )

        self.scan(database, now, tmp_path)

        assert total(database).requests == 3
        with database.connect() as conn:
            cursor = next(iter(db.load_cursors(conn, Provider.CODEX).values()))
            assert isinstance(cursor.state, dict)
            assert cursor.state["counted_requests"] == 3
            db.prune(conn, now)
        assert total(database).requests == 1
        saved_size = rollout.stat().st_size
        rollout.write_bytes(b"")
        append(
            rollout,
            codex_meta("workspace-a"),
            codex_count(expired, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
            codex_count(expired, codex_totals(200, 0, 20), codex_totals(100, 0, 10)),
        )
        gap = saved_size - rollout.stat().st_size
        assert gap > 1
        append(rollout, b"x" * (gap - 1))
        assert rollout.stat().st_size == saved_size

        self.scan(database, now, tmp_path)

        assert total(database).requests == 1
        with database.connect() as conn:
            db.prune(conn, now)
            cursor = next(iter(db.load_cursors(conn, Provider.CODEX).values()))
        assert total(database).requests == 1
        assert isinstance(cursor.state, dict)
        assert cursor.state["counted_requests"] == 3

    def test_an_archived_same_size_rewrite_of_only_expired_requests_keeps_the_retained_request(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        expired = now - timedelta(days=401)
        rollout = self.active(tmp_path)
        rollout.parent.mkdir(parents=True, exist_ok=True)
        rollout.write_bytes(b"")
        append(
            rollout,
            codex_meta("workspace-a"),
            codex_count(expired, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
            codex_count(expired, codex_totals(200, 0, 20), codex_totals(100, 0, 10)),
            codex_count(now, codex_totals(300, 0, 30), codex_totals(100, 0, 10)),
        )
        self.scan(database, now, tmp_path)
        with database.connect() as conn:
            db.prune(conn, now)
        assert total(database).requests == 1
        saved_size = rollout.stat().st_size
        rollout.write_bytes(b"")
        append(
            rollout,
            codex_meta("workspace-a"),
            codex_count(expired, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
            codex_count(expired, codex_totals(200, 0, 20), codex_totals(100, 0, 10)),
        )
        gap = saved_size - rollout.stat().st_size
        assert gap > 1
        append(rollout, b"x" * (gap - 1))
        assert rollout.stat().st_size == saved_size
        self.archived(tmp_path).parent.mkdir(parents=True)
        rollout.rename(self.archived(tmp_path))

        self.scan(database, now, tmp_path)

        assert total(database).requests == 1
        with database.connect() as conn:
            db.prune(conn, now)
            cursor = next(iter(db.load_cursors(conn, Provider.CODEX).values()))
        assert total(database).requests == 1
        assert isinstance(cursor.state, dict)
        assert cursor.state["counted_requests"] == 3

    def test_an_archived_expired_rewrite_keeps_the_retained_request_when_a_shorter_file_reuses_the_path(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        expired = now - timedelta(days=401)
        rollout = self.active(tmp_path)
        rollout.parent.mkdir(parents=True, exist_ok=True)
        rollout.write_bytes(b"")
        append(
            rollout,
            codex_meta("workspace-a"),
            codex_count(expired, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
            codex_count(expired, codex_totals(200, 0, 20), codex_totals(100, 0, 10)),
            codex_count(now, codex_totals(300, 0, 30), codex_totals(100, 0, 10)),
        )
        self.scan(database, now, tmp_path)
        with database.connect() as conn:
            db.prune(conn, now)
        assert total(database).requests == 1
        saved_size = rollout.stat().st_size
        rollout.write_bytes(b"")
        append(
            rollout,
            codex_meta("workspace-a"),
            codex_count(expired, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
            codex_count(expired, codex_totals(200, 0, 20), codex_totals(100, 0, 10)),
        )
        gap = saved_size - rollout.stat().st_size
        assert gap > 1
        append(rollout, b"x" * (gap - 1))
        assert rollout.stat().st_size == saved_size
        self.archived(tmp_path).parent.mkdir(parents=True)
        rollout.rename(self.archived(tmp_path))
        self.write(self.active(tmp_path), now, 1)
        assert self.active(tmp_path).stat().st_size < saved_size

        self.scan(database, now, tmp_path)

        assert total(database).requests == 1
        with database.connect() as conn:
            db.prune(conn, now)
            cursor = next(iter(db.load_cursors(conn, Provider.CODEX).values()))
        assert total(database).requests == 1
        assert isinstance(cursor.state, dict)
        assert cursor.state["counted_requests"] == 3

    def _retain_one_current_request(self, database: Database, tmp_path: Path, now: datetime) -> tuple[Path, datetime]:
        expired = now - timedelta(days=401)
        rollout = self.active(tmp_path)
        rollout.parent.mkdir(parents=True, exist_ok=True)
        rollout.write_bytes(b"")
        append(
            rollout,
            codex_meta("workspace-a"),
            *self.requests(expired, 1, 2),
            *self.requests(now, 3, 3),
        )
        self.scan(database, now, tmp_path)
        with database.connect() as conn:
            db.prune(conn, now)
        assert total(database).requests == 1
        return rollout, expired

    def test_a_same_size_rewrite_of_three_expired_requests_keeps_the_retained_request(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout, expired = self._retain_one_current_request(database, tmp_path, now)
        saved_size = rollout.stat().st_size
        rollout.write_bytes(b"")
        append(rollout, codex_meta("workspace-a"), *self.requests(expired, 1, 3))
        assert rollout.stat().st_size == saved_size

        (report,) = self.scan(database, now, tmp_path)
        with database.connect() as conn:
            db.prune(conn, now)

        assert report.files_read == 0
        assert total(database).requests == 1

    def test_a_longer_rewrite_of_four_expired_requests_keeps_the_retained_request(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout, expired = self._retain_one_current_request(database, tmp_path, now)
        saved_size = rollout.stat().st_size
        rollout.write_bytes(b"")
        append(rollout, codex_meta("workspace-a"), *self.requests(expired, 1, 4))
        assert rollout.stat().st_size > saved_size

        (report,) = self.scan(database, now, tmp_path)
        with database.connect() as conn:
            db.prune(conn, now)

        assert report.files_read == 0
        assert total(database).requests == 1

    def test_a_rewrite_that_keeps_the_retained_request_is_stored(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout, expired = self._retain_one_current_request(database, tmp_path, now)
        rollout.write_bytes(b"")
        append(
            rollout,
            {"type": "note", "payload": {"text": "rewritten"}},
            codex_meta("workspace-a"),
            *self.requests(expired, 1, 2),
            codex_count(now, codex_totals(40, 0, 4), codex_totals(40, 0, 4)),
        )

        (report,) = self.scan(database, now, tmp_path)
        with database.connect() as conn:
            db.prune(conn, now)

        assert report.files_read == 1
        assert total(database) == TokenCounts(40, 0, 0, 4, 0, 1)

    def test_a_padded_copy_does_not_replace_three_requests_after_the_counted_file_is_gone(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        counted = self.write(self.active(tmp_path), now, 3)
        self.scan(database, now, tmp_path)
        saved = counted.stat().st_size
        counted.unlink()
        copy = self.write(self.archived(tmp_path), now, 1)
        append(copy, {"type": "note", "payload": {"text": "x" * 2000}})
        assert copy.stat().st_size > saved

        assert self.scan(database, now, tmp_path)[0].files_read == 0
        assert total(database) == self.counted(3)

    def _pad_a_retained_rollout(self, database: Database, tmp_path: Path, now: datetime) -> tuple[Path, datetime]:
        rollout, expired = self._retain_one_current_request(database, tmp_path, now)
        append(rollout, {"type": "note", "payload": {"text": "x" * 800}})
        self.scan(database, now, tmp_path)
        return rollout, expired

    def test_a_shorter_expired_rewrite_of_the_tracked_file_keeps_the_retained_request(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout, expired = self._pad_a_retained_rollout(database, tmp_path, now)
        saved = self.read_up_to(database, tmp_path)
        self.write(rollout, expired, 3)
        assert rollout.stat().st_size < saved

        self.scan(database, now, tmp_path)
        with database.connect() as conn:
            db.prune(conn, now)

        assert total(database).requests == 1

    def test_a_larger_expired_copy_does_not_replace_after_the_tracked_file_shrinks(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout, expired = self._pad_a_retained_rollout(database, tmp_path, now)
        self.write(rollout, now, 1)
        larger = self.write(self.archived(tmp_path), expired, 5)
        assert larger.stat().st_size > rollout.stat().st_size

        self.scan(database, now, tmp_path)
        with database.connect() as conn:
            db.prune(conn, now)

        assert total(database).requests == 1

    def test_a_shorter_copy_of_only_expired_requests_keeps_the_retained_request(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout, expired = self._retain_one_current_request(database, tmp_path, now)
        append(rollout, {"type": "note", "payload": {"text": "holds no usage; " + "padding " * 400}})
        self.scan(database, now, tmp_path)
        archived = self.archived(tmp_path)
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(b"")
        append(archived, codex_meta("workspace-a"), *self.requests(expired, 1, 4))
        assert archived.stat().st_size < self.read_up_to(database, tmp_path)

        (report,) = self.scan(database, now, tmp_path)
        with database.connect() as conn:
            db.prune(conn, now)

        assert report.files_read == 0
        assert total(database).requests == 1

    def test_growth_on_the_tracked_file_is_stored_when_a_longer_expired_copy_is_rejected(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout, expired = self._retain_one_current_request(database, tmp_path, now)
        append(
            rollout,
            {"type": "note", "payload": {"text": "no usage"}},
            codex_turn("gpt-x"),
            *self.requests(now, 4, 4),
            {
                "type": "event_msg",
                "payload": {"type": "token_count", "info": {"total_token_usage": codex_totals(1, 0, 1)}},
            },
        )
        archived = self.archived(tmp_path)
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(b"")
        append(archived, codex_meta("workspace-a"), *self.requests(expired, 1, 5))
        assert archived.stat().st_size > rollout.stat().st_size

        (report,) = self.scan(database, now, tmp_path)

        assert report.files_read == 1
        assert report.malformed_records == 1
        assert total(database) == self.counted(2)
        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2

    def _replace_an_unfinished_tail(
        self, database: Database, tmp_path: Path, now: datetime, *, match_size: bool
    ) -> ScanReport:
        rollout, expired = self._retain_one_current_request(database, tmp_path, now)
        append(rollout, b'{"type":"event_msg","payload":{"type":"token_count","info":' + b" " * 4000, newline=False)
        self.scan(database, now, tmp_path)
        with database.connect() as conn:
            cursor = db.load_cursors(conn, Provider.CODEX)[str(tmp_path / self.NAME)]
        assert cursor.size > cursor.byte_offset
        rollout.write_bytes(rollout.read_bytes()[: cursor.byte_offset])
        append(rollout, *self.requests(now, 4, 4))
        if match_size:
            rollout.write_bytes(rollout.read_bytes() + b" " * (cursor.size - rollout.stat().st_size))
            assert rollout.stat().st_size == cursor.size
        else:
            assert cursor.byte_offset < rollout.stat().st_size < cursor.size
        archived = self.archived(tmp_path)
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(b"")
        append(archived, codex_meta("workspace-a"), *self.requests(expired, 1, 5))
        return self.scan(database, now, tmp_path)[0]

    def test_growth_after_an_unfinished_tail_is_stored_when_the_file_is_shorter(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        report = self._replace_an_unfinished_tail(database, tmp_path, now, match_size=False)

        assert report.files_read == 1
        assert total(database) == self.counted(2)

    def test_growth_after_an_unfinished_tail_is_stored_when_the_file_is_the_same_size(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        report = self._replace_an_unfinished_tail(database, tmp_path, now, match_size=True)

        assert report.files_read == 1
        assert total(database) == self.counted(2)

    def test_a_rewritten_tracked_file_is_not_resumed_when_its_anchor_no_longer_matches(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout, expired = self._retain_one_current_request(database, tmp_path, now)
        saved_size = rollout.stat().st_size
        self.write(rollout, expired, 1)
        append(rollout, {"type": "note", "payload": {"text": "padding " * 400}})
        assert rollout.stat().st_size > saved_size
        archived = self.archived(tmp_path)
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(b"")
        append(archived, codex_meta("workspace-a"), *self.requests(expired, 1, 5))

        (report,) = self.scan(database, now, tmp_path)

        assert report.files_read == 0
        assert total(database).requests == 1

    def test_a_rejected_expired_copy_does_not_hide_a_shorter_tracked_rewrite(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.write(self.active(tmp_path), now, 3)
        self.scan(database, now, tmp_path)
        self.write(rollout, now, 1)
        self.write(self.archived(tmp_path), now - timedelta(days=401), 5)

        assert self.scan(database, now, tmp_path)[0].files_read == 1
        assert total(database) == self.counted(1)

    def test_a_rejected_expired_copy_does_not_hide_a_longer_tracked_rewrite(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.write(self.active(tmp_path), now, 3)
        self.scan(database, now, tmp_path)
        self.write(rollout, now, 4)
        self.write(self.archived(tmp_path), now - timedelta(days=401), 5)

        assert self.scan(database, now, tmp_path)[0].files_read == 1
        assert total(database) == self.counted(4)

    def test_a_rejected_expired_copy_does_not_hide_another_recent_copy(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.write(self.active(tmp_path), now, 3)
        self.scan(database, now, tmp_path)
        self.write(tmp_path / "sessions" / "other" / self.NAME, now, 4)
        self.write(self.archived(tmp_path), now - timedelta(days=401), 5)

        assert self.scan(database, now, tmp_path)[0].files_read == 1
        assert total(database) == self.counted(4)

    def test_a_new_inode_swapped_in_after_planning_keeps_the_counted_requests(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rollout = self.write(self.active(tmp_path), now, 3)
        self.scan(database, now, tmp_path)
        self.write(rollout, now, 1)
        real = codex_scanner._read_planned_copy

        def swap(*args: object, **kwargs: object) -> ScanReport | None:
            rollout.unlink()
            self.write(rollout, now, 1)
            return real(*args, **kwargs)

        monkeypatch.setattr(codex_scanner, "_read_planned_copy", swap)
        assert self.scan(database, now, tmp_path)[0].files_read == 0
        monkeypatch.undo()
        assert self.scan(database, now, tmp_path)[0].files_read == 0
        assert total(database) == self.counted(3)


class TestTheFullRequestCountSavedOnACursor:
    def test_a_missing_or_unusable_saved_count_falls_back_to_the_rows_still_stored(
        self, database: Database, now: datetime
    ) -> None:
        file_key = "rollout.jsonl"
        kept = UsageKey(1, "2026-10-03T14", "m")
        with database.connect() as conn:
            db.add_file_usage(conn, Provider.CODEX, file_key, {kept: TokenCounts(requests=4)})
            db.save_cursor(conn, Provider.CODEX, file_key, FileCursor(1, 1, 1, {"counted_requests": True}))
        assert codex_scanner._requests_kept(database, file_key) == 4
        assert codex_scanner._saved_counted_requests(None) is None
        assert codex_scanner._saved_counted_requests(FileCursor(0, 0, 0, "not-a-dict")) is None
        assert codex_scanner._saved_counted_requests(FileCursor(0, 0, 0, {"counted_requests": -1})) is None
        assert codex_scanner._saved_counted_requests(FileCursor(0, 0, 0, {"counted_requests": 5})) == 5
        assert codex_scanner._request_total(()) == 0

    def test_the_saved_full_count_wins_when_it_is_above_the_rows_still_stored(self, database: Database) -> None:
        file_key = "rollout.jsonl"
        with database.connect() as conn:
            db.save_cursor(conn, Provider.CODEX, file_key, FileCursor(1, 1, 1, {"counted_requests": 5}))
        assert codex_scanner._requests_kept(database, file_key) == 5
        cursor = FileCursor(1, 1, 1, {"counted_requests": 5})
        assert codex_scanner._counted_after(None, False, 3, database, file_key) == 3
        assert codex_scanner._counted_after(cursor, True, 2, database, file_key) == 2
        assert codex_scanner._counted_after(cursor, False, 2, database, file_key) == 7
        bare = FileCursor(1, 1, 1, {})
        assert codex_scanner._counted_after(bare, False, 2, database, "other.jsonl") == 2

    def test_requests_kept_inside_retention_skip_an_older_bucket(self, database: Database) -> None:
        file_key = "rollout.jsonl"
        with database.connect() as conn:
            db.add_file_usage(
                conn,
                Provider.CODEX,
                file_key,
                {
                    UsageKey(1, "2020-01-01T00", "m"): TokenCounts(requests=2),
                    UsageKey(1, "2026-10-03T14", "m"): TokenCounts(requests=1),
                },
            )
            assert db.stored_requests_kept(conn, Provider.CODEX, file_key, "2026-10-03") == 1


class TestACodexRolloutWithMoreThanOneFileInAHome(CodexCopies):
    @pytest.mark.parametrize(
        ("archived_requests", "active_requests"),
        [pytest.param(2, 1, id="the archived copy is larger"), pytest.param(1, 2, id="the active copy is larger")],
    )
    def test_of_an_archived_and_an_active_copy_the_larger_is_counted_in_every_scan(
        self, database: Database, tmp_path: Path, now: datetime, archived_requests: int, active_requests: int
    ) -> None:
        self.write(self.archived(tmp_path), now, archived_requests)
        self.write(self.active(tmp_path), now, active_requests)

        assert self.totals_over(database, now, 3, tmp_path) == [self.counted(2)] * 3

    def test_an_initial_scan_counts_the_copy_with_more_complete_requests(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        complete = self.write(self.archived(tmp_path), now, 2)
        padded = self.with_a_line_still_being_written(self.write(self.active(tmp_path), now, 1))
        assert padded.stat().st_size > complete.stat().st_size

        assert self.totals_over(database, now, 3, tmp_path) == [self.counted(2)] * 3

    def test_a_copy_with_no_complete_line_does_not_hide_a_shorter_one(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        shorter = self.write(self.archived(tmp_path), now, 1)
        longer = self.active(tmp_path)
        longer.parent.mkdir(parents=True)
        longer.write_bytes(b'{"type":"event_msg","payload":{"type":"token_count","info":' + b" " * 4000)
        assert longer.stat().st_size > shorter.stat().st_size

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(1)] * 2

    def test_a_malformed_record_is_skipped_while_the_copies_requests_are_counted(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        counted = self.write(self.archived(tmp_path), now, 2)
        append(counted, {"type": "event_msg", "payload": {"type": "token_count", "info": {}}})
        longer = self.with_a_line_still_being_written(self.write(self.active(tmp_path), now, 1))
        if longer.stat().st_size <= counted.stat().st_size:
            append(longer, b" " * (counted.stat().st_size - longer.stat().st_size + 64), newline=False)
        assert longer.stat().st_size > counted.stat().st_size
        assert longer.read_bytes().rfind(b"\n") + 1 < counted.stat().st_size

        (report,) = self.scan(database, now, tmp_path)

        assert report.malformed_records == 1
        assert total(database) == self.counted(2)

    def _longer_by_an_unfinished_line(self, home: Path, now: datetime) -> tuple[Path, Path]:
        longer = self.with_a_line_still_being_written(self.write(self.active(home), now, 2))
        shorter = self.write(self.archived(home), now, 1)
        append(shorter, {"type": "note", "payload": {"text": "holds no usage; " + "makes this copy longer " * 30}})
        assert longer.read_bytes().rfind(b"\n") + 1 <= shorter.stat().st_size < longer.stat().st_size
        return longer, shorter

    def test_a_copy_resized_while_its_requests_are_counted_is_ranked_by_complete_lines(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        longer, _shorter = self._longer_by_an_unfinished_line(tmp_path, now)
        count_requests = codex_scanner._requests_in

        def resized(copy: Any) -> int | None:
            if copy.path == longer:
                longer.write_bytes(longer.read_bytes() + b"\n")
            return count_requests(copy)

        monkeypatch.setattr(codex_scanner, "_requests_in", resized)

        (report,) = self.scan(database, now, tmp_path)

        assert (report.files_read, report.files_failed) == (1, 0)
        assert total(database) == self.counted(1)

    def test_a_copy_that_cannot_be_counted_is_ranked_by_complete_lines(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        longer, _shorter = self._longer_by_an_unfinished_line(tmp_path, now)
        count_requests = codex_scanner._requests_in

        def unreadable(copy: Any) -> int | None:
            if copy.path == longer:
                longer.chmod(0)
            return count_requests(copy)

        monkeypatch.setattr(codex_scanner, "_requests_in", unreadable)
        try:
            (report,) = self.scan(database, now, tmp_path)
        finally:
            longer.chmod(0o644)

        assert (report.files_read, report.files_failed) == (1, 0)
        assert total(database) == self.counted(1)

    def test_a_new_copy_is_counted_when_the_fuller_copy_is_outside_the_backfill_window(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        older = self.write(self.archived(tmp_path), now, 2)
        aged = (now - timedelta(days=BACKFILL_DAYS + 5)).timestamp()
        os.utime(older, (aged, aged))
        self.with_a_line_still_being_written(self.write(self.active(tmp_path), now, 1))

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(1)] * 2

    @pytest.mark.parametrize(
        ("earlier_requests", "later_requests"),
        [pytest.param(2, 1, id="the first path is larger"), pytest.param(1, 2, id="the last path is larger")],
    )
    def test_of_two_copies_under_different_dated_directories_the_larger_is_counted_in_every_scan(
        self, database: Database, tmp_path: Path, now: datetime, earlier_requests: int, later_requests: int
    ) -> None:
        sessions = tmp_path / "sessions" / "2026" / "10"
        self.write(sessions / "03" / self.NAME, now, earlier_requests)
        self.write(sessions / "04" / self.NAME, now, later_requests)

        assert self.totals_over(database, now, 3, tmp_path) == [self.counted(2)] * 3

    def test_two_copies_of_one_size_are_counted_once_and_the_one_read_stays_the_one_read(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.write(self.archived(tmp_path), now, 2)
        self.write(self.active(tmp_path), now, 2)

        (first,) = self.scan(database, now, tmp_path)
        (second,) = self.scan(database, now, tmp_path)

        assert (first.files_read, second.files_read) == (1, 0)
        assert total(database) == self.counted(2)

    def test_what_the_larger_copy_gains_is_counted_while_a_smaller_copy_exists(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        archived = self.write(self.archived(tmp_path), now, 2)
        self.write(self.active(tmp_path), now, 1)
        self.scan(database, now, tmp_path)
        append(archived, *self.requests(now, 3, 4))

        self.scan(database, now, tmp_path)

        assert total(database) == self.counted(4)

    def test_a_copy_that_outgrows_the_one_read_before_is_read_afresh_and_not_as_its_continuation(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.write(self.active(tmp_path), now, 1)
        self.scan(database, now, tmp_path)
        append(
            self.archived(tmp_path),
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(500, 0, 50), codex_totals(500, 0, 50)),
            codex_count(now, codex_totals(1000, 0, 100), codex_totals(500, 0, 50)),
        )

        self.scan(database, now, tmp_path)

        assert total(database) == TokenCounts(input_tokens=1000, output_tokens=100, requests=2)

    @pytest.mark.parametrize("cut_short", ["the active copy", "the archived copy"])
    def test_a_copy_cut_short_gives_way_to_the_larger_copy_in_the_same_scan(
        self, database: Database, tmp_path: Path, now: datetime, cut_short: str
    ) -> None:
        active, archived = self.active(tmp_path), self.archived(tmp_path)
        read_before, other = (active, archived) if cut_short == "the active copy" else (archived, active)
        self.write(read_before, now, 3)
        self.write(other, now, 2)
        self.scan(database, now, tmp_path)
        assert total(database) == self.counted(3)
        self.write(read_before, now, 1)

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2

    def test_a_copy_with_the_size_and_time_the_read_copy_had_is_read_once_the_read_copy_is_cut_short(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        active = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        archived = self.archived(tmp_path)
        archived.parent.mkdir(parents=True)
        shutil.copy2(active, archived)
        assert (archived.stat().st_size, archived.stat().st_mtime_ns) == (
            active.stat().st_size,
            active.stat().st_mtime_ns,
        )
        self.write(active, now, 1)

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2

    def test_only_the_copy_that_is_read_is_opened(self, database: Database, tmp_path: Path, now: datetime) -> None:
        smaller = self.write(self.archived(tmp_path), now, 1)
        smaller.chmod(0)
        self.write(self.active(tmp_path), now, 2)

        (report,) = self.scan(database, now, tmp_path)

        assert (report.files_read, report.files_failed) == (1, 0)
        assert total(database) == self.counted(2)

    def test_a_smaller_copy_does_not_stand_in_for_the_largest_when_that_cannot_be_read(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.write(self.active(tmp_path), now, 1)
        self.scan(database, now, tmp_path)
        largest = self.write(self.archived(tmp_path), now, 3)
        largest.chmod(0)

        (report,) = self.scan(database, now, tmp_path)

        assert (report.files_read, report.files_failed) == (0, 1)
        assert report.first_failure == f"{self.NAME}: PermissionError"
        assert total(database) == self.counted(1)

    def test_a_copy_that_cannot_be_examined_fails_the_rollout_and_changes_nothing(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.write(self.active(tmp_path), now, 1)
        self.scan(database, now, tmp_path)
        append(self.active(tmp_path), *self.requests(now, 2, 2))
        self.archived(tmp_path).parent.mkdir(parents=True)
        self.archived(tmp_path).symlink_to(tmp_path / "no-such-target.jsonl")

        (report,) = self.scan(database, now, tmp_path)

        assert (report.files_read, report.files_failed) == (0, 1)
        assert report.first_failure == f"{self.NAME}: FileNotFoundError"
        assert total(database) == self.counted(1)

    def test_a_session_that_grew_and_was_then_moved_into_the_archive_is_counted_in_full_once(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.write(self.active(tmp_path), now, 1)
        self.scan(database, now, tmp_path)
        append(rollout, *self.requests(now, 2, 2))
        self.archived(tmp_path).parent.mkdir(parents=True)
        rollout.rename(self.archived(tmp_path))

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2

    def test_an_unchanged_rollout_moved_into_the_archive_resumes_at_the_saved_position(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rollout = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        archived = self.archived(tmp_path)
        archived.parent.mkdir(parents=True)
        rollout.rename(archived)
        resumes = 0
        yielded = 0
        resume = OpenSessionFile.resume
        lines = OpenSessionFile.lines

        def count_resume(opened: OpenSessionFile, cursor: FileCursor | None, anchor: object) -> bool:
            nonlocal resumes
            resumes += 1
            return resume(opened, cursor, anchor)

        def count_lines(opened: OpenSessionFile) -> Iterator[bytes]:
            nonlocal yielded
            for line in lines(opened):
                yielded += 1
                yield line

        monkeypatch.setattr(OpenSessionFile, "resume", count_resume)
        monkeypatch.setattr(OpenSessionFile, "lines", count_lines)

        self.scan(database, now, tmp_path)

        assert (resumes, yielded) == (1, 0)
        assert total(database) == self.counted(2)
        with database.connect() as conn:
            cursor = next(iter(db.load_cursors(conn, Provider.CODEX).values()))
        assert isinstance(cursor.state, dict)
        assert cursor.state["path"] == str(archived)

    def test_a_shortened_rollout_moved_to_the_archive_replaces_the_old_total(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.write(self.active(tmp_path), now, 3)
        self.scan(database, now, tmp_path)
        self.write(rollout, now, 1)
        self.archived(tmp_path).parent.mkdir(parents=True)
        rollout.rename(self.archived(tmp_path))

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(1)] * 2

    def test_a_shortened_archived_rollout_replaces_the_old_total_when_a_longer_file_reuses_the_path(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.write(self.active(tmp_path), now, 3)
        self.scan(database, now, tmp_path)
        saved_size = rollout.stat().st_size
        self.write(rollout, now, 1)
        self.archived(tmp_path).parent.mkdir(parents=True)
        rollout.rename(self.archived(tmp_path))
        reused = self.active(tmp_path)
        reused.parent.mkdir(parents=True, exist_ok=True)
        reused.write_bytes(b"")
        append(reused, codex_meta("workspace-a"))
        gap = saved_size - reused.stat().st_size
        assert gap > 1
        append(reused, b"x" * (gap - 1))
        assert reused.stat().st_size == saved_size

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(1)] * 2

    def test_two_requests_at_the_old_path_replace_a_shortened_archived_file(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.write(self.active(tmp_path), now, 3)
        self.scan(database, now, tmp_path)
        self.write(rollout, now, 1)
        self.archived(tmp_path).parent.mkdir(parents=True)
        rollout.rename(self.archived(tmp_path))
        reused = self.write(self.active(tmp_path), now, 2)
        assert reused.stat().st_size < self.read_up_to(database, tmp_path)
        assert reused.stat().st_size > self.archived(tmp_path).stat().st_size

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2

    def test_a_shorter_new_inode_at_the_saved_path_keeps_the_counted_requests(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.write(self.active(tmp_path), now, 3)
        recorded_inode = rollout.stat().st_ino
        self.scan(database, now, tmp_path)
        replacement = rollout.with_suffix(".new")
        self.write(replacement, now, 1)
        assert replacement.stat().st_size < rollout.stat().st_size
        replacement.replace(rollout)
        assert rollout.stat().st_ino != recorded_inode

        self.scan(database, now, tmp_path)

        assert total(database) == self.counted(3)

    def test_an_out_of_window_new_inode_at_the_saved_path_keeps_the_archived_requests(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.write(self.active(tmp_path), now, 3)
        self.scan(database, now, tmp_path)
        self.archived(tmp_path).parent.mkdir(parents=True)
        rollout.rename(self.archived(tmp_path))
        expired = now - timedelta(days=401)
        reused = self.active(tmp_path)
        reused.parent.mkdir(parents=True, exist_ok=True)
        reused.write_bytes(b"")
        append(reused, codex_meta("workspace-a"), *self.requests(expired, 1, 5))
        assert reused.stat().st_ino != self.archived(tmp_path).stat().st_ino
        assert reused.stat().st_size > self.archived(tmp_path).stat().st_size
        self._age_before_the_window(reused, now)

        self.scan(database, now, tmp_path)
        assert total(database) == self.counted(3)
        with database.connect() as conn:
            db.prune(conn, now)

        assert total(database) == self.counted(3)

    def test_a_tracked_session_moved_into_the_archive_outside_the_backfill_window_is_still_read(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.write(self.active(tmp_path), now, 1)
        self.scan(database, now, tmp_path)
        append(rollout, *self.requests(now, 2, 2))
        self.archived(tmp_path).parent.mkdir(parents=True)
        rollout.rename(self.archived(tmp_path))
        self._age_before_the_window(self.archived(tmp_path), now)

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2

    def test_an_older_archive_that_is_not_the_file_already_read_stays_outside_the_window(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.write(self.active(tmp_path), now, 1)
        self.scan(database, now, tmp_path)
        older = self.write(self.archived(tmp_path), now, 2)
        self._age_before_the_window(older, now)

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(1)] * 2

    @pytest.mark.parametrize("missing", ["device", "inode"])
    def test_a_moved_session_outside_the_window_is_not_read_without_a_recorded_device_and_inode(
        self, database: Database, tmp_path: Path, now: datetime, missing: str
    ) -> None:
        rollout = self.write(self.active(tmp_path), now, 1)
        self.scan(database, now, tmp_path)
        self._forget_recorded_identity(database, tmp_path, missing)
        append(rollout, *self.requests(now, 2, 2))
        self.archived(tmp_path).parent.mkdir(parents=True)
        rollout.rename(self.archived(tmp_path))
        self._age_before_the_window(self.archived(tmp_path), now)

        (report,) = self.scan(database, now, tmp_path)

        assert report.files_read == 0
        assert total(database) == self.counted(1)

    def _age_before_the_window(self, path: Path, now: datetime) -> None:
        aged = (now - timedelta(days=BACKFILL_DAYS + 1)).timestamp()
        os.utime(path, (aged, aged))

    def _forget_recorded_identity(self, database: Database, home: Path, missing: str) -> None:
        file_key = str(home / self.NAME)
        with database.connect() as conn:
            cursor = db.load_cursors(conn, Provider.CODEX)[file_key]
            assert isinstance(cursor.state, dict)
            state = dict(cursor.state)
            del state[missing]
            db.save_cursor(
                conn,
                Provider.CODEX,
                file_key,
                FileCursor(cursor.byte_offset, cursor.size, cursor.mtime_ns, state),
            )

    def test_what_was_counted_stays_when_its_copy_has_gone_and_only_a_smaller_one_is_left(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        read_before = self.write(self.active(tmp_path), now, 2)
        self.write(self.archived(tmp_path), now, 1)
        self.scan(database, now, tmp_path)
        read_before.unlink()

        (report,) = self.scan(database, now, tmp_path)

        assert (report.files_read, report.files_failed) == (0, 0)
        assert total(database) == self.counted(2)

    def test_four_expired_requests_do_not_replace_three_kept_when_the_counted_file_is_gone(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.write(self.active(tmp_path), now, 3)
        self.scan(database, now, tmp_path)
        saved_size = rollout.stat().st_size
        rollout.unlink()
        archived = self.write(self.archived(tmp_path), now - timedelta(days=401), 4)
        assert archived.stat().st_size >= saved_size

        (report,) = self.scan(database, now, tmp_path)
        with database.connect() as conn:
            db.prune(conn, now)

        assert report.files_read == 0
        assert total(database) == self.counted(3)

    def test_what_was_read_of_a_copy_that_has_gone_counts_in_full_against_a_smaller_copy(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        read_before = self.write(self.active(tmp_path), now, 2)
        size_when_taken = read_before.stat().st_size
        third_request = self.requests(now, 3, 3)

        class AppendedToBeforeItIsRead(OpenSessionFile):
            def lines(self) -> Iterator[bytes]:
                append(read_before, *third_request)
                return super().lines()

        monkeypatch.setattr(scanner_base, "OpenSessionFile", AppendedToBeforeItIsRead)
        self.scan(database, now, tmp_path)
        monkeypatch.undo()
        assert total(database) == self.counted(3)
        smaller = self.write(self.archived(tmp_path), now, 2)
        append(
            smaller, {"type": "note", "payload": {"text": "holds no usage; makes this copy longer than two requests"}}
        )
        assert size_when_taken < smaller.stat().st_size < read_before.stat().st_size
        read_before.unlink()

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(3)] * 2

    def with_a_line_still_being_written(self, rollout: Path) -> Path:
        append(rollout, b'{"type":"event_msg","payload":{"type":"token_count","info":' + b" " * 2000, newline=False)
        return rollout

    def test_an_archived_copy_holding_what_was_read_and_more_is_counted_though_the_read_copy_was_longer(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        read_before = self.with_a_line_still_being_written(self.write(self.active(tmp_path), now, 2))
        self.scan(database, now, tmp_path)
        assert total(database) == self.counted(2)
        archived = self.write(self.archived(tmp_path), now, 3)
        assert archived.stat().st_size < read_before.stat().st_size
        read_before.unlink()

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(3)] * 2

    def test_an_archived_copy_holding_exactly_what_was_read_becomes_the_copy_that_is_read(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        read_before = self.with_a_line_still_being_written(self.write(self.active(tmp_path), now, 2))
        self.scan(database, now, tmp_path)
        archived = self.write(self.archived(tmp_path), now, 2)
        read_before.unlink()

        (moved,) = self.scan(database, now, tmp_path)
        append(archived, *self.requests(now, 3, 3))
        (grown,) = self.scan(database, now, tmp_path)

        assert (moved.files_read, grown.files_read) == (1, 1)
        assert total(database) == self.counted(3)

    def test_an_archived_copy_one_byte_short_of_what_was_read_is_left_alone(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        read_before = self.with_a_line_still_being_written(self.write(self.active(tmp_path), now, 2))
        self.scan(database, now, tmp_path)
        archived = self.write(self.archived(tmp_path), now, 2)
        archived.write_bytes(archived.read_bytes()[:-1])
        read_before.unlink()

        (report,) = self.scan(database, now, tmp_path)

        assert (report.files_read, report.files_failed) == (0, 0)
        assert total(database) == self.counted(2)

    def test_an_archived_copy_as_large_as_what_was_read_but_holding_fewer_whole_lines_is_left_alone(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        read_before = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        archived = self.with_a_line_still_being_written(self.write(self.archived(tmp_path), now, 1))
        assert archived.stat().st_size >= read_before.stat().st_size
        read_before.unlink()

        (report,) = self.scan(database, now, tmp_path)
        assert (report.files_read, report.files_failed) == (0, 0)
        assert total(database) == self.counted(2)

        self.write(archived, now, 3)
        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(3)] * 2

    def test_the_copy_read_before_cut_short_after_it_was_chosen_is_left_for_the_next_cycle_to_decide(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        active = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        self.write(self.archived(tmp_path), now, 3)
        append(active, *self.requests(now, 3, 4))
        sized = codex_scanner._copy_at

        def cut_short_once_sized(path: Path) -> Any:
            copy = sized(path)
            if path == active:
                self.write(active, now, 1)
            return copy

        monkeypatch.setattr(codex_scanner, "_copy_at", cut_short_once_sized)
        (report,) = self.scan(database, now, tmp_path)
        monkeypatch.undo()

        assert (report.files_read, report.files_failed) == (0, 0)
        assert total(database) == self.counted(2)
        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(3)] * 2

    def test_a_larger_copy_holding_fewer_whole_lines_does_not_replace_a_copy_that_is_still_there(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        read_before = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        archived = self.with_a_line_still_being_written(self.write(self.archived(tmp_path), now, 1))
        assert archived.stat().st_size > read_before.stat().st_size

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2

        self.write(archived, now, 3)
        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(3)] * 2

    def test_a_rewritten_copy_larger_only_by_an_unfinished_line_does_not_erase_the_other_copy(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        active = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        archived = self.write(self.archived(tmp_path), now, 2)
        self.with_a_line_still_being_written(self.write(active, now, 1))
        assert active.stat().st_size > archived.stat().st_size

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2

    def test_a_rewritten_copy_larger_only_by_a_complete_non_usage_line_does_not_erase_counted_requests(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        active = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        counted_size = active.stat().st_size
        self.write(active, now, 1)
        append(active, {"type": "note", "payload": {"text": "holds no usage; " + "padding " * 400}})
        assert active.stat().st_size > counted_size

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2

    def test_complete_non_usage_padding_does_not_make_a_rewritten_copy_replace_the_other(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        active = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        archived = self.write(self.archived(tmp_path), now, 2)
        self.write(active, now, 1)
        append(active, {"type": "note", "payload": {"text": "holds no usage; " + "padding " * 400}})
        assert active.stat().st_size > archived.stat().st_size

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2

    def test_an_unchanged_file_that_ends_mid_line_is_not_read_again(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.with_a_line_still_being_written(self.write(self.active(tmp_path), now, 2))

        first = self.scan(database, now, tmp_path)[0]
        second = self.scan(database, now, tmp_path)[0]

        assert (first.files_read, second.files_read) == (1, 0)
        assert total(database) == self.counted(2)

    def test_an_unchanged_raw_leader_does_not_hide_a_smaller_copy_with_more_complete_requests(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        active = self.with_a_line_still_being_written(self.write(self.active(tmp_path), now, 2))
        self.scan(database, now, tmp_path)
        assert total(database) == self.counted(2)
        archived = self.write(self.archived(tmp_path), now, 3)
        read_of_active = active.read_bytes().rfind(b"\n") + 1
        assert read_of_active < archived.stat().st_size < active.stat().st_size

        assert self.totals_over(database, now, 3, tmp_path) == [self.counted(3)] * 3

    def _pad_with_a_complete_line(self, rollout: Path) -> Path:
        append(rollout, {"type": "note", "payload": {"text": "holds no usage; " + "padding " * 400}})
        return rollout

    def test_a_shorter_copy_added_after_a_padded_copy_was_read_is_counted(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        padded = self._pad_with_a_complete_line(self.write(self.active(tmp_path), now, 2))
        self.scan(database, now, tmp_path)
        shorter = self.write(self.archived(tmp_path), now, 3)
        read_up_to = self.read_up_to(database, tmp_path)
        assert shorter.stat().st_size <= read_up_to <= padded.stat().st_size

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(3)] * 2

    def test_a_shorter_copy_is_kept_when_the_padded_copy_changes(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        padded = self._pad_with_a_complete_line(self.write(self.active(tmp_path), now, 2))
        self.scan(database, now, tmp_path)
        shorter = self.write(self.archived(tmp_path), now, 3)
        append(padded, {"type": "note", "payload": {"text": "still no usage"}})
        assert shorter.stat().st_size < self.read_up_to(database, tmp_path)

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(3)] * 2

    def test_a_shorter_copy_is_kept_when_the_padded_copy_is_gone(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        padded = self._pad_with_a_complete_line(self.write(self.active(tmp_path), now, 2))
        self.scan(database, now, tmp_path)
        shorter = self.write(self.archived(tmp_path), now, 3)
        assert shorter.stat().st_size < self.read_up_to(database, tmp_path)
        padded.unlink()

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(3)] * 2

    def test_padding_a_rewrite_to_the_saved_size_does_not_erase_counted_requests(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        active = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        saved_size = active.stat().st_size
        self.write(self.archived(tmp_path), now, 2)
        self.write(active, now, 1)
        gap = saved_size - active.stat().st_size
        assert gap > 1
        append(active, b"x" * (gap - 1))
        assert active.stat().st_size == saved_size

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2

    def test_dropping_an_unread_tail_does_not_let_fewer_requests_replace_what_was_counted(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        active = self.write(self.active(tmp_path), now, 2)
        complete = active.stat().st_size
        self.with_a_line_still_being_written(active)
        raw = active.stat().st_size
        self.scan(database, now, tmp_path)
        assert total(database) == self.counted(2)
        self.write(active, now, 1)
        while active.stat().st_size <= complete:
            append(active, {"type": "note", "payload": {"text": "holds no usage; " + "padding " * 20}})
        assert complete < active.stat().st_size < raw

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2

    def test_a_larger_padded_copy_does_not_replace_an_intact_copy_and_later_growth_is_counted(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        active = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        padded = self.write(self.archived(tmp_path), now, 1)
        append(padded, {"type": "note", "payload": {"text": "holds no usage; " + "padding " * 400}})
        assert padded.stat().st_size > active.stat().st_size

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2
        append(active, *self.requests(now, 3, 3))
        assert active.stat().st_size < padded.stat().st_size
        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(3)] * 2

    def test_a_copy_smaller_than_the_padded_one_is_counted_when_it_holds_more_requests(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        unfinished = self.with_a_line_still_being_written(self.write(self.active(tmp_path), now, 2))
        three = self.write(tmp_path / "sessions" / "older" / self.NAME, now, 3)
        padded = self.write(self.archived(tmp_path), now, 1)
        while padded.stat().st_size <= three.stat().st_size:
            append(padded, {"type": "note", "payload": {"text": "holds no usage; " + "padding " * 10}})
        assert three.stat().st_size < padded.stat().st_size < unfinished.stat().st_size

        (report,) = self.scan(database, now, tmp_path)

        assert report.files_read == 1
        assert total(database) == self.counted(3)

    def test_a_copy_as_long_as_the_best_complete_lines_is_still_ranked(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        archived = self.write(self.archived(tmp_path), now, 2)
        active = self.write(self.active(tmp_path), now, 1)
        gap = archived.stat().st_size - active.stat().st_size
        assert gap > 1
        append(active, b"y" * (gap - 1))
        assert active.stat().st_size == archived.stat().st_size
        self.with_a_line_still_being_written(active)

        (report,) = self.scan(database, now, tmp_path)

        assert report.files_read == 1
        assert total(database) == self.counted(2)

    def test_the_first_scan_keeps_the_copy_with_more_requests_when_padding_is_larger(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        active = self.write(self.active(tmp_path), now, 2)
        padded = self.write(self.archived(tmp_path), now, 1)
        append(padded, {"type": "note", "payload": {"text": "holds no usage; " + "padding " * 400}})
        assert padded.stat().st_size > active.stat().st_size

        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2
        append(active, *self.requests(now, 3, 3))
        assert active.stat().st_size < padded.stat().st_size
        assert self.totals_over(database, now, 1, tmp_path) == [self.counted(3)]

    def test_copies_of_equal_size_are_ranked_by_requests(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        active = self.write(self.active(tmp_path), now, 2)
        archived = self.write(self.archived(tmp_path), now, 1)
        gap = active.stat().st_size - archived.stat().st_size
        assert gap > 1
        append(archived, b"y" * (gap - 1))
        assert archived.stat().st_size == active.stat().st_size

        (report,) = self.scan(database, now, tmp_path)

        assert report.files_read == 1
        assert total(database) == self.counted(2)

    def test_a_raw_size_tie_does_not_keep_the_copy_with_fewer_requests(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        active = self.with_a_line_still_being_written(self.write(self.active(tmp_path), now, 2))
        archived = self.write(self.archived(tmp_path), now, 1)
        target = active.stat().st_size
        while archived.stat().st_size + 80 < target:
            append(archived, {"type": "note", "payload": {"text": "holds no usage"}})
        gap = target - archived.stat().st_size
        assert gap > 0
        append(archived, b"q" * gap, newline=False)
        assert archived.stat().st_size == target
        assert archived.read_bytes().rfind(b"\n") + 1 > active.read_bytes().rfind(b"\n") + 1

        (report,) = self.scan(database, now, tmp_path)

        assert report.files_read == 1
        assert total(database) == self.counted(2)

    def test_a_copy_padded_after_it_was_counted_does_not_replace_an_intact_copy(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        active = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        archived = self.write(self.archived(tmp_path), now, 3)
        append(archived, {"type": "note", "payload": {"text": "holds no usage; " + "padding " * 400}})
        assert archived.stat().st_size > active.stat().st_size
        count_requests = codex_scanner._requests_in
        open_copy = codex_scanner.open_session_file
        counting = {"depth": 0}
        rewrite_on_next_open = {"ready": False}

        def count(copy: Any) -> int | None:
            counting["depth"] += 1
            try:
                return count_requests(copy)
            finally:
                counting["depth"] -= 1
                if copy.path == archived and counting["depth"] == 0:
                    rewrite_on_next_open["ready"] = True

        def opening(path: Path) -> Any:
            if path == archived and rewrite_on_next_open["ready"] and counting["depth"] == 0:
                rewrite_on_next_open["ready"] = False
                size = archived.stat().st_size
                self.write(archived, now, 1)
                gap = size - archived.stat().st_size
                assert gap > 1
                append(archived, b"z" * (gap - 1))
                assert archived.stat().st_size == size
            return open_copy(path)

        monkeypatch.setattr(codex_scanner, "_requests_in", count)
        monkeypatch.setattr(codex_scanner, "open_session_file", opening)
        (report,) = self.scan(database, now, tmp_path)
        monkeypatch.undo()

        assert (report.files_read, report.files_failed) == (0, 0)
        assert total(database) == self.counted(2)
        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2
        append(active, *self.requests(now, 3, 3))
        assert active.stat().st_size < archived.stat().st_size
        assert self.totals_over(database, now, 1, tmp_path) == [self.counted(3)]

    def test_a_copy_with_fewer_stored_requests_is_not_read_while_the_original_still_reaches_them(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        active = self.write(self.active(tmp_path), now, 3)
        self.scan(database, now, tmp_path)
        saved_size = active.stat().st_size
        self.write(self.archived(tmp_path), now, 2)
        self.write(active, now, 1)
        gap = saved_size - active.stat().st_size
        assert gap > 1
        append(active, b"x" * (gap - 1))
        assert active.stat().st_size == saved_size
        count_requests = codex_scanner._requests_in
        open_copy = codex_scanner.open_session_file
        counting = {"depth": 0}
        opened: list[Path] = []

        def count(copy: Any) -> int | None:
            counting["depth"] += 1
            try:
                return count_requests(copy)
            finally:
                counting["depth"] -= 1

        def opening(path: Path) -> Any:
            if counting["depth"] == 0:
                opened.append(path)
            return open_copy(path)

        monkeypatch.setattr(codex_scanner, "_requests_in", count)
        monkeypatch.setattr(codex_scanner, "open_session_file", opening)
        (report,) = self.scan(database, now, tmp_path)
        monkeypatch.undo()

        assert opened == [active]
        assert report.files_read == 0
        assert total(database) == self.counted(3)

    def test_a_larger_copy_cut_short_before_it_is_read_does_not_replace_a_copy_that_is_still_there(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        archived = self.write(self.archived(tmp_path), now, 3)
        cut_short = self.requests(now, 1, 1)

        class CutShortBeforeItIsRead(OpenSessionFile):
            def lines(self) -> Iterator[bytes]:
                archived.write_bytes(b"")
                append(archived, codex_meta("workspace-a"), *cut_short)
                return super().lines()

        monkeypatch.setattr(scanner_base, "OpenSessionFile", CutShortBeforeItIsRead)
        (report,) = self.scan(database, now, tmp_path)
        monkeypatch.undo()

        assert (report.files_read, report.files_failed) == (0, 0)
        assert total(database) == self.counted(2)

    def test_a_copy_cut_short_after_it_was_chosen_and_before_it_was_read_is_left_alone(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        read_before = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        archived = self.write(self.archived(tmp_path), now, 3)
        read_before.unlink()
        cut_short = self.requests(now, 1, 1)

        class CutShortBeforeItIsRead(OpenSessionFile):
            def lines(self) -> Iterator[bytes]:
                archived.write_bytes(b"")
                append(archived, codex_meta("workspace-a"), *cut_short)
                return super().lines()

        monkeypatch.setattr(scanner_base, "OpenSessionFile", CutShortBeforeItIsRead)
        (report,) = self.scan(database, now, tmp_path)
        monkeypatch.undo()

        assert (report.files_read, report.files_failed) == (0, 0)
        assert total(database) == self.counted(2)
        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2

    @pytest.mark.parametrize("saved_offset", [-1, 10**9])
    def test_a_saved_position_that_is_not_one_in_the_file_it_was_saved_for_is_not_measured_against(
        self, database: Database, tmp_path: Path, now: datetime, saved_offset: int
    ) -> None:
        read_before = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        with database.connect() as conn:
            conn.execute("UPDATE scan_cursors SET byte_offset = ? WHERE provider = 'codex'", (saved_offset,))
        archived = self.archived(tmp_path)
        archived.parent.mkdir(parents=True)
        read_before.rename(archived)

        (moved,) = self.scan(database, now, tmp_path)
        append(archived, *self.requests(now, 3, 3))
        self.scan(database, now, tmp_path)

        assert moved.files_read == 1
        assert total(database) == self.counted(3)

    def test_an_uncountable_copy_does_not_discard_the_counts_that_were_read(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        padded = self._pad_with_a_complete_line(self.write(self.archived(tmp_path), now, 1))
        active = self.write(self.active(tmp_path), now, 2)
        faulty = self.write(tmp_path / "sessions" / "older" / self.NAME, now, 1)
        assert faulty.stat().st_size < active.stat().st_size < padded.stat().st_size
        calls: list[Path] = []
        count_requests = codex_scanner._requests_in

        def unreadable(copy: Any) -> int | None:
            calls.append(copy.path)
            if copy.path == faulty:
                return None
            return count_requests(copy)

        monkeypatch.setattr(codex_scanner, "_requests_in", unreadable)
        (first,) = self.scan(database, now, tmp_path)
        assert (first.files_read, first.files_failed) == (1, 0)
        assert total(database) == self.counted(2)
        assert sorted(calls) == sorted([padded, active, faulty])
        calls.clear()

        (second,) = self.scan(database, now, tmp_path)

        assert (second.files_read, second.files_failed) == (0, 0)
        assert calls == [faulty]
        assert total(database) == self.counted(2)

    def test_an_unreadable_copy_the_same_size_as_a_padded_copy_does_not_hide_more_requests(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        padded = self._pad_with_a_complete_line(self.write(self.archived(tmp_path), now, 1))
        smaller = self.write(self.active(tmp_path), now, 2)
        assert smaller.stat().st_size < padded.stat().st_size
        unreadable = tmp_path / "sessions" / "older" / self.NAME
        unreadable.parent.mkdir(parents=True)
        unreadable.write_bytes(padded.read_bytes())
        assert unreadable.stat().st_size == padded.stat().st_size
        unreadable.chmod(0)
        try:
            (report,) = self.scan(database, now, tmp_path)
            assert (report.files_read, report.files_failed) == (1, 0)
            assert total(database) == self.counted(2)
            assert self.totals_over(database, now, 1, tmp_path) == [self.counted(2)]
            append(smaller, *self.requests(now, 3, 3))
            assert smaller.stat().st_size < padded.stat().st_size
            assert self.totals_over(database, now, 1, tmp_path) == [self.counted(3)]
        finally:
            unreadable.chmod(0o644)

    def test_a_larger_unreadable_copy_is_still_a_failed_read_when_smaller_copies_can_be_counted(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.write(self.active(tmp_path), now, 1)
        self.write(tmp_path / "sessions" / "older" / self.NAME, now, 2)
        largest = self.write(self.archived(tmp_path), now, 3)
        largest.chmod(0)
        try:
            (report,) = self.scan(database, now, tmp_path)
        finally:
            largest.chmod(0o644)

        assert (report.files_read, report.files_failed) == (0, 1)
        assert total(database) == self.counted(0)

    def test_the_only_readable_copy_is_read_when_an_unreadable_copy_is_the_same_size(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        readable = self.write(self.active(tmp_path), now, 2)
        unreadable = self.archived(tmp_path)
        unreadable.parent.mkdir(parents=True)
        unreadable.write_bytes(readable.read_bytes())
        assert unreadable.stat().st_size == readable.stat().st_size
        unreadable.chmod(0)
        try:
            (report,) = self.scan(database, now, tmp_path)
            assert (report.files_read, report.files_failed) == (1, 0)
            assert total(database) == self.counted(2)
        finally:
            unreadable.chmod(0o644)

    def test_one_readable_copy_does_not_stand_in_for_a_strictly_larger_unreadable_copy(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.write(self.active(tmp_path), now, 2)
        largest = self._pad_with_a_complete_line(self.write(self.archived(tmp_path), now, 1))
        assert self.active(tmp_path).stat().st_size < largest.stat().st_size
        largest.chmod(0)
        try:
            (report,) = self.scan(database, now, tmp_path)
        finally:
            largest.chmod(0o644)

        assert (report.files_read, report.files_failed) == (0, 1)
        assert total(database) == self.counted(0)

    def test_a_copy_that_grows_after_its_size_is_taken_does_not_beat_a_stable_copy_with_more_requests(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stable = self.write(self.active(tmp_path), now, 2)
        grown = self._pad_with_a_complete_line(self.write(self.archived(tmp_path), now, 1))
        assert stable.stat().st_size < grown.stat().st_size
        sized = codex_scanner._copy_at

        def grow_once_sized(path: Path) -> Any:
            copy = sized(path)
            if path == grown:
                append(grown, b"x")
            return copy

        monkeypatch.setattr(codex_scanner, "_copy_at", grow_once_sized)
        (report,) = self.scan(database, now, tmp_path)
        monkeypatch.undo()

        assert (report.files_read, report.files_failed) == (1, 0)
        assert total(database) == self.counted(2)
        assert self.totals_over(database, now, 1, tmp_path) == [self.counted(2)]

    def test_unchanged_copies_are_not_counted_again(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.write(self.archived(tmp_path), now, 1)
        self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        calls: list[Path] = []
        count_requests = codex_scanner._requests_in

        def counting(copy: Any) -> int | None:
            calls.append(copy.path)
            return count_requests(copy)

        monkeypatch.setattr(codex_scanner, "_requests_in", counting)
        (report,) = self.scan(database, now, tmp_path)

        assert calls == []
        assert report.files_read == 0
        assert total(database) == self.counted(2)

    def test_a_changed_copy_is_counted_again_and_an_unchanged_copy_is_not(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archived = self.write(self.archived(tmp_path), now, 1)
        active = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        append(active, *self.requests(now, 3, 3))
        calls: list[Path] = []
        count_requests = codex_scanner._requests_in

        def counting(copy: Any) -> int | None:
            calls.append(copy.path)
            return count_requests(copy)

        monkeypatch.setattr(codex_scanner, "_requests_in", counting)
        (report,) = self.scan(database, now, tmp_path)

        assert calls == [active]
        assert report.files_read == 1
        assert total(database) == self.counted(3)
        assert archived.stat().st_size < active.stat().st_size

    def test_counts_taken_while_an_unchanged_file_stays_chosen_are_reused(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        active = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        archived = self.write(self.archived(tmp_path), now, 1)
        assert archived.stat().st_size < active.stat().st_size
        calls: list[Path] = []
        count_requests = codex_scanner._requests_in

        def counting(copy: Any) -> int | None:
            calls.append(copy.path)
            return count_requests(copy)

        monkeypatch.setattr(codex_scanner, "_requests_in", counting)
        (deciding,) = self.scan(database, now, tmp_path)
        assert (deciding.files_read, deciding.files_failed) == (0, 0)
        assert sorted(calls) == sorted([archived, active])
        assert total(database) == self.counted(2)
        calls.clear()

        (later,) = self.scan(database, now, tmp_path)

        assert calls == []
        assert later.files_read == 0
        assert total(database) == self.counted(2)

    def test_copies_that_cannot_be_counted_are_counted_again_on_the_next_cycle(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.write(self.archived(tmp_path), now, 1)
        self.write(self.active(tmp_path), now, 2)
        calls: list[Path] = []

        def unreadable(copy: Any) -> int | None:
            calls.append(copy.path)
            return None

        monkeypatch.setattr(codex_scanner, "_requests_in", unreadable)
        (first,) = self.scan(database, now, tmp_path)
        assert first.files_read == 1
        assert total(database) == self.counted(2)
        calls.clear()

        (second,) = self.scan(database, now, tmp_path)

        assert second.files_read == 0
        assert len(calls) == 2
        assert total(database) == self.counted(2)

    @pytest.mark.parametrize("damage", ["a repeated path", "a count below zero", "not a list"])
    def test_a_saved_request_count_that_cannot_be_trusted_is_taken_again(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch, damage: str
    ) -> None:
        archived = self.write(self.archived(tmp_path), now, 1)
        active = self.write(self.active(tmp_path), now, 2)
        self.scan(database, now, tmp_path)
        with database.read() as conn:
            row = conn.execute("SELECT state_json FROM scan_cursors WHERE provider = 'codex'").fetchone()
        state = json.loads(row["state_json"])
        counts = state["request_counts"]
        assert isinstance(counts, list) and len(counts) == 2
        if damage == "a repeated path":
            state["request_counts"] = [counts[0], {**counts[0]}]
        elif damage == "a count below zero":
            state["request_counts"] = [{**counts[0], "requests": -1}]
        else:
            state["request_counts"] = {"requests": 1}
        with database.connect() as conn:
            conn.execute("UPDATE scan_cursors SET state_json = ? WHERE provider = 'codex'", (json.dumps(state),))
        calls: list[Path] = []
        count_requests = codex_scanner._requests_in

        def counting(copy: Any) -> int | None:
            calls.append(copy.path)
            return count_requests(copy)

        monkeypatch.setattr(codex_scanner, "_requests_in", counting)
        (report,) = self.scan(database, now, tmp_path)
        assert sorted(calls) == sorted([archived, active])
        assert report.files_read == 0
        assert total(database) == self.counted(2)
        calls.clear()

        (later,) = self.scan(database, now, tmp_path)

        assert calls == []
        assert later.files_read == 0

    def test_request_counts_are_not_written_onto_a_cursor_whose_state_is_not_an_object(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.write(self.active(tmp_path), now, 1)
        self.scan(database, now, tmp_path)
        file_key = str(tmp_path / self.NAME)
        with database.connect() as conn:
            conn.execute("UPDATE scan_cursors SET state_json = '[]' WHERE provider = 'codex'")
        with database.read() as conn:
            cursor = db.load_cursors(conn, Provider.CODEX)[file_key]

        codex_scanner._remember_request_counts(database, file_key, cursor, [])

        with database.read() as conn:
            assert db.load_cursors(conn, Provider.CODEX)[file_key].state == []


class TestACodexCopyHiddenByADirectoryThatCannotBeListed(CodexCopies):
    def dated(self, home: Path) -> Path:
        return home / "sessions" / "2026" / "10" / "03"

    def read_copy_and_an_older_one_that_reaches_it(self, database: Database, home: Path, now: datetime) -> Path:
        read_before = self.write(self.dated(home) / self.NAME, now, 2)
        append(read_before, b'{"type":"event_msg","payload":{"type":"token_count","info":' + b" " * 2000, newline=False)
        older = self.write(self.archived(home), now, 1)
        append(older, {"type": "note", "payload": {"text": "holds no usage; " + "makes this copy longer " * 30}})
        self.scan(database, now, home)
        read_up_to = self.saved(database)[str(home / self.NAME)].byte_offset
        assert read_up_to <= older.stat().st_size < read_before.stat().st_size
        assert total(database) == self.counted(2)
        return older

    def saved(self, database: Database) -> dict[str, FileCursor]:
        with database.read() as conn:
            return db.load_cursors(conn, Provider.CODEX)

    def scan_with_the_dated_directory_unlistable(self, database: Database, home: Path, now: datetime) -> ScanReport:
        self.dated(home).chmod(0)
        try:
            (report,) = self.scan(database, now, home)
        finally:
            self.dated(home).chmod(0o700)
        return report

    def test_the_copy_read_before_is_not_taken_for_one_that_has_gone_and_nothing_replaces_what_it_counted(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.read_copy_and_an_older_one_that_reaches_it(database, tmp_path, now)
        saved_before = self.saved(database)

        report = self.scan_with_the_dated_directory_unlistable(database, tmp_path, now)

        assert (report.files_read, report.files_failed) == (0, 2)
        assert report.first_failure == "03: PermissionError"
        assert self.saved(database) == saved_before
        assert total(database) == self.counted(2)

    def test_a_shorter_older_file_at_the_saved_path_does_not_replace_three_stored_requests(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        read_before = self.write(self.active(tmp_path), now, 3)
        self.scan(database, now, tmp_path)
        file_key = str(tmp_path / self.NAME)
        state = self.saved(database)[file_key].state
        assert isinstance(state, dict)
        recorded = (state["device"], state["inode"])
        hidden = tmp_path / "sessions" / "hidden"
        hidden.mkdir()
        read_before.rename(hidden / self.NAME)
        assert ((hidden / self.NAME).stat().st_dev, (hidden / self.NAME).stat().st_ino) == recorded
        replacement = self.write(self.active(tmp_path), now, 1)
        aged = (now - timedelta(days=BACKFILL_DAYS + 1)).timestamp()
        os.utime(replacement, (aged, aged))
        assert (replacement.stat().st_dev, replacement.stat().st_ino) != recorded
        assert replacement.stat().st_size < self.saved(database)[file_key].byte_offset
        saved_before = self.saved(database)
        hidden.chmod(0)
        try:
            (report,) = self.scan(database, now, tmp_path)
        finally:
            hidden.chmod(0o700)

        assert (report.files_read, report.files_failed) == (0, 2)
        assert self.saved(database) == saved_before
        assert total(database) == self.counted(3)

    def test_the_same_inode_in_the_archive_is_read_when_another_directory_cannot_be_listed(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = self.write(self.active(tmp_path), now, 1)
        self.scan(database, now, tmp_path)
        file_key = str(tmp_path / self.NAME)
        state = self.saved(database)[file_key].state
        assert isinstance(state, dict)
        recorded = (state["device"], state["inode"])
        append(rollout, *self.requests(now, 2, 2))
        self.archived(tmp_path).parent.mkdir(parents=True)
        rollout.rename(self.archived(tmp_path))
        archived = self.archived(tmp_path)
        assert (archived.stat().st_dev, archived.stat().st_ino) == recorded
        hidden = tmp_path / "sessions" / "hidden"
        hidden.mkdir()
        hidden.chmod(0)
        try:
            (report,) = self.scan(database, now, tmp_path)
        finally:
            hidden.chmod(0o700)

        assert (report.files_read, report.files_failed) == (1, 1)
        assert report.first_failure == "hidden: PermissionError"
        assert total(database) == self.counted(2)

    def test_once_the_directory_can_be_listed_again_the_copy_is_read_on_from_where_it_was(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.read_copy_and_an_older_one_that_reaches_it(database, tmp_path, now)
        self.scan_with_the_dated_directory_unlistable(database, tmp_path, now)
        self.write(self.dated(tmp_path) / self.NAME, now, 3)

        (report,) = self.scan(database, now, tmp_path)

        assert (report.files_read, report.files_failed) == (1, 0)
        assert total(database) == self.counted(3)

    def test_saved_state_that_cannot_be_read_back_replaces_nothing_while_a_directory_cannot_be_listed(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.read_copy_and_an_older_one_that_reaches_it(database, tmp_path, now)
        with database.connect() as conn:
            conn.execute("UPDATE scan_cursors SET state_json = '[]' WHERE provider = 'codex'")
        saved_before = self.saved(database)

        report = self.scan_with_the_dated_directory_unlistable(database, tmp_path, now)

        assert (report.files_read, report.files_failed) == (0, 2)
        assert self.saved(database) == saved_before
        assert total(database) == self.counted(2)
        assert [self.scan(database, now, tmp_path)[0].files_read for _scan in range(2)] == [1, 0]
        assert total(database) == self.counted(2)

    def test_a_copy_that_is_in_view_is_still_read_on_while_another_directory_cannot_be_listed(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        older = self.read_copy_and_an_older_one_that_reaches_it(database, tmp_path, now)
        self.write(older, now, 2)
        assert older.stat().st_size >= self.read_up_to(database, tmp_path)
        (self.dated(tmp_path) / self.NAME).unlink()
        self.scan(database, now, tmp_path)
        state = self.saved(database)[str(tmp_path / self.NAME)].state
        assert isinstance(state, dict)
        assert state["path"] == str(older)
        append(older, *self.requests(now, 3, 3))

        report = self.scan_with_the_dated_directory_unlistable(database, tmp_path, now)

        assert (report.files_read, report.files_failed) == (1, 1)
        assert total(database) == self.counted(3)

    def test_a_rollout_never_read_before_is_read_from_the_copy_that_is_found(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.write(self.dated(tmp_path) / self.NAME, now, 2)
        self.write(self.archived(tmp_path), now, 1)

        report = self.scan_with_the_dated_directory_unlistable(database, tmp_path, now)

        assert (report.files_read, report.files_failed) == (1, 1)
        assert total(database) == self.counted(1)
        assert self.totals_over(database, now, 2, tmp_path) == [self.counted(2)] * 2

    def test_a_dated_directory_whose_kind_cannot_be_told_does_not_give_way_to_the_older_archive(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.read_copy_and_an_older_one_that_reaches_it(database, tmp_path, now)
        saved_before = self.saved(database)
        monkeypatch.setattr(scanner_base.os, "scandir", _scandir_hiding("03"))

        (report,) = self.scan(database, now, tmp_path)

        assert (report.files_read, report.files_failed) == (0, 2)
        assert report.first_failure == "03: PermissionError"
        assert self.saved(database) == saved_before
        assert total(database) == self.counted(2)


class TestCodexHomesKeepTheirOwnCounts(CodexCopies):
    @pytest.mark.parametrize("order", [("longer", "shorter"), ("shorter", "longer")], ids=" then ".join)
    def test_a_rollout_in_two_homes_is_counted_in_each_and_neither_changes_the_other_s_count(
        self, database: Database, tmp_path: Path, now: datetime, order: tuple[str, str]
    ) -> None:
        self.write(self.active(tmp_path / "longer"), now, 2)
        self.write(self.active(tmp_path / "shorter"), now, 1)
        homes = [tmp_path / name for name in order]

        assert self.totals_over(database, now, 3, *homes) == [self.counted(3)] * 3

    def test_each_home_s_position_is_kept_under_that_home(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        homes = [tmp_path / "home-a", tmp_path / "home-b"]
        for home in homes:
            self.write(self.active(home), now, 1)

        self.scan(database, now, *homes)

        with database.read() as conn:
            assert sorted(db.load_cursors(conn, Provider.CODEX)) == [str(home / self.NAME) for home in homes]

    def test_what_happens_to_one_home_s_file_changes_only_that_home_s_count(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
        self.write(self.active(home_a), now, 2)
        self.write(self.active(home_b), now, 2)
        self.scan(database, now, home_a, home_b)
        assert total(database) == self.counted(4)

        self.write(self.active(home_b), now, 1)
        self.scan(database, now, home_a, home_b)
        after_one_was_rewritten = total(database)
        self.active(home_a).unlink()
        self.scan(database, now, home_a, home_b)

        assert (after_one_was_rewritten, total(database)) == (self.counted(3), self.counted(3))

    def test_a_home_that_is_no_longer_scanned_does_not_hold_back_another_home(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        former, current = tmp_path / "former", tmp_path / "current"
        grows_unread = self.write(self.active(former), now, 1)
        self.write(self.active(current), now, 1)
        self.scan(database, now, former, current)
        append(grows_unread, *self.requests(now, 2, 5))
        append(self.active(current), *self.requests(now, 2, 2))

        self.scan(database, now, current)

        assert total(database) == self.counted(3)

    def test_sessions_with_different_names_in_two_homes_are_each_counted(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.write(self.active(tmp_path / "home-a"), now, 2)
        other = tmp_path / "home-b" / "sessions" / "rollout-2026-10-03T11-00-00-bbbb.jsonl"
        append(other, codex_meta("workspace-a"), *self.requests(now, 1, 1))

        self.scan(database, now, tmp_path / "home-a", tmp_path / "home-b")

        assert total(database) == self.counted(3)


def grok_turn(
    number: int, ended_at: datetime, input_tokens: int, cached: int, output: int, model: str = "grok-x"
) -> dict[str, Any]:
    counts = {
        "inputTokens": input_tokens,
        "outputTokens": output,
        "cachedReadTokens": cached,
        "cacheCreationTokens": 0,
        "reasoningTokens": 2,
        "totalTokens": input_tokens + output,
        "modelCalls": 3,
    }
    return {
        "turnNumber": number,
        "endedAt": ended_at.isoformat(),
        "primaryModelId": model,
        **counts,
        "modelUsage": {model: counts},
    }


class TestGrokSessionScanner:
    def usage_file(self, home: Path, session: str = "session-1") -> Path:
        path = home / "sessions" / "%2Fsome%2Fproject" / session / "usage.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def scan(self, database: Database, home: Path, now: datetime, resolver: AccountResolver | None = None) -> Any:
        return GrokSessionScanner(home).scan(database, resolver or everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

    def test_counts_every_turn_once_even_though_the_file_is_rewritten_each_turn(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        usage = self.usage_file(tmp_path)
        first = grok_turn(1, now, 1000, 900, 50)
        usage.write_text(json.dumps({"turns": [first]}))
        self.scan(database, tmp_path, now)

        usage.write_text(json.dumps({"turns": [first, grok_turn(2, now + timedelta(hours=1), 200, 150, 20)]}))
        self.scan(database, tmp_path, now)

        assert usage_rows(database) == {
            (ACCOUNT_A, "2026-10-03T14", "grok-x"): TokenCounts(100, 900, 0, 50, 2, 3),
            (ACCOUNT_A, "2026-10-03T15", "grok-x"): TokenCounts(50, 150, 0, 20, 2, 3),
        }

    def test_attributes_each_turn_by_who_was_logged_in_when_it_ended(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        switch = now + timedelta(minutes=30)
        resolver = build_account_resolver([(now, ACCOUNT_A), (switch, ACCOUNT_B)])
        self.usage_file(tmp_path).write_text(
            json.dumps({"turns": [grok_turn(1, now, 10, 0, 1), grok_turn(2, switch, 10, 0, 1)]})
        )

        self.scan(database, tmp_path, now, resolver)

        assert sorted(account for account, _, _ in usage_rows(database)) == [ACCOUNT_A, ACCOUNT_B]

    def test_a_turn_without_a_per_model_breakdown_uses_its_own_totals(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        turn = grok_turn(1, now, 100, 60, 10)
        del turn["modelUsage"]
        self.usage_file(tmp_path).write_text(json.dumps({"turns": [turn]}))

        self.scan(database, tmp_path, now)

        assert usage_rows(database) == {(ACCOUNT_A, "2026-10-03T14", "grok-x"): TokenCounts(40, 60, 0, 10, 2, 3)}

    def test_a_file_caught_mid_write_is_reported_and_retried_on_the_next_scan(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        usage = self.usage_file(tmp_path)
        whole = json.dumps({"turns": [grok_turn(1, now, 10, 0, 1)]})
        usage.write_text(whole[:25])

        report = self.scan(database, tmp_path, now)
        assert (report.files_read, report.files_failed) == (0, 1)
        assert report.first_failure is not None

        usage.write_text(whole)
        assert self.scan(database, tmp_path, now).files_read == 1
        assert total(database).requests == 3

    def test_a_file_with_an_unexpected_shape_is_reported_not_ingested(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        self.usage_file(tmp_path).write_text(json.dumps({"turns": [{"turnNumber": 1, "inputTokens": -4}]}))

        report = self.scan(database, tmp_path, now)

        assert report.files_failed == 1
        assert total(database) == TokenCounts()

    @pytest.mark.parametrize(
        "document", [{}, {"turns": None}, {"sessionId": "session-1", "updatedAt": "2026-10-03T14:00:00Z"}]
    )
    def test_a_document_without_its_turn_history_is_reported_and_the_stored_usage_is_kept(
        self, database: Database, tmp_path: Path, now: datetime, document: dict[str, Any]
    ) -> None:
        usage = self.usage_file(tmp_path)
        whole = json.dumps({"turns": [grok_turn(1, now, 1000, 900, 50)]})
        usage.write_text(whole)
        self.scan(database, tmp_path, now)
        stored = usage_rows(database)

        usage.write_text(json.dumps(document))
        report = self.scan(database, tmp_path, now)

        assert (report.files_read, report.files_failed) == (0, 1)
        assert report.first_failure == "session-1: ValidationError"
        assert (
            usage_rows(database)
            == stored
            == {(ACCOUNT_A, "2026-10-03T14", "grok-x"): TokenCounts(100, 900, 0, 50, 2, 3)}
        )

        usage.write_text(whole + " ")
        assert self.scan(database, tmp_path, now).files_read == 1
        assert usage_rows(database) == stored

    def test_a_history_that_is_stated_and_empty_is_a_session_with_no_usage(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        usage = self.usage_file(tmp_path)
        usage.write_text(json.dumps({"turns": [grok_turn(1, now, 1000, 900, 50)]}))
        self.scan(database, tmp_path, now)

        usage.write_text(json.dumps({"turns": []}))
        report = self.scan(database, tmp_path, now)

        assert (report.files_read, report.files_failed) == (1, 0)
        assert usage_rows(database) == {}

    def test_an_unchanged_file_is_not_reread(self, database: Database, tmp_path: Path, now: datetime) -> None:
        self.usage_file(tmp_path).write_text(json.dumps({"turns": [grok_turn(1, now, 10, 0, 1)]}))
        assert self.scan(database, tmp_path, now).files_read == 1
        assert self.scan(database, tmp_path, now).files_read == 0

    def test_a_usage_file_replaced_with_its_size_and_mtime_kept_is_read_again(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        usage = self.usage_file(tmp_path)
        usage.write_text(json.dumps({"turns": [grok_turn(1, now, 1000, 900, 50)]}))
        self.scan(database, tmp_path, now)
        replacement = json.dumps({"turns": [grok_turn(1, now, 2000, 800, 60)]}).encode()

        put_in_place_keeping_size_and_mtime(usage, replacement)
        report = self.scan(database, tmp_path, now)

        assert report.files_read == 1
        assert total(database) == TokenCounts(
            input_tokens=1_200, cache_read_tokens=800, output_tokens=60, reasoning_tokens=2, requests=3
        )

    def test_a_usage_file_rewritten_in_place_with_its_mtime_restored_is_read_again(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        usage = self.usage_file(tmp_path)
        usage.write_text(json.dumps({"turns": [grok_turn(1, now, 1000, 900, 50)]}))
        self.scan(database, tmp_path, now)
        replacement = json.dumps({"turns": [grok_turn(1, now, 2000, 800, 60)]}).encode()

        rewrite_in_place_keeping_mtime(usage, replacement)
        report = self.scan(database, tmp_path, now)

        assert report.files_read == 1
        assert total(database) == TokenCounts(
            input_tokens=1_200, cache_read_tokens=800, output_tokens=60, reasoning_tokens=2, requests=3
        )

    def test_a_usage_file_rewritten_in_place_while_it_is_read_keeps_what_was_stored(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        usage = self.usage_file(tmp_path)
        usage.write_text(json.dumps({"turns": [grok_turn(1, now, 1000, 900, 50)]}))
        self.scan(database, tmp_path, now)
        stored = usage_rows(database)
        with database.read() as conn:
            saved = db.load_cursors(conn, Provider.GROK)

        def document(first_input: int, second_input: int) -> bytes:
            turns = [grok_turn(1, now, first_input, 0, 1), grok_turn(2, now, second_input, 0, 1)]
            return json.dumps({"turns": turns}).encode()

        opened_version = document(200, 0)
        rewritten_version = document(400, 50)
        marker = b"}, {"
        assert opened_version.index(marker) == rewritten_version.index(marker)
        boundary = opened_version.index(marker) + len(b"}, ")
        joined = opened_version[:boundary] + rewritten_version[boundary:]

        def counted(raw: bytes) -> TokenCounts:
            result = TokenCounts()
            for counts in usage_from_file(raw, everything_to(ACCOUNT_A)).values():
                result = result.plus(counts)
            return result

        assert (counted(joined).input_tokens, counted(opened_version).input_tokens) == (250, 200)
        assert counted(rewritten_version).input_tokens == 450
        usage.write_bytes(opened_version)

        armed = True

        class ReadThatRewrites:
            def __init__(self, handle: Any) -> None:
                self._handle = handle

            def __enter__(self) -> "ReadThatRewrites":
                return self

            def __exit__(self, *_exc: object) -> None:
                self._handle.close()

            def fileno(self) -> int:
                return int(self._handle.fileno())

            def read(self, _size: int = -1) -> bytes:
                nonlocal armed
                armed = False
                descriptor = self.fileno()
                first = os.pread(descriptor, boundary, 0)
                assert first == opened_version[:boundary]
                rewrite = os.open(usage, os.O_WRONLY | os.O_TRUNC)
                try:
                    os.write(rewrite, rewritten_version)
                finally:
                    os.close(rewrite)
                rest = os.pread(descriptor, len(rewritten_version) - boundary, boundary)
                assert rest == rewritten_version[boundary:]
                return first + rest

        original_open = Path.open

        def open_and_rewrite_once(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            handle = original_open(self, mode, *args, **kwargs)
            if not armed or self.name != "usage.json":
                return handle
            return ReadThatRewrites(handle)

        monkeypatch.setattr(Path, "open", open_and_rewrite_once)
        report = self.scan(database, tmp_path, now)

        assert (report.files_read, report.files_failed) == (0, 1)
        assert report.first_failure == "session-1: SessionFileChangedError"
        assert usage_rows(database) == stored
        with database.read() as conn:
            assert db.load_cursors(conn, Provider.GROK) == saved
        monkeypatch.undo()
        assert self.scan(database, tmp_path, now).files_read == 1
        assert total(database) == TokenCounts(
            input_tokens=450, cache_read_tokens=0, cache_write_tokens=0, output_tokens=2, reasoning_tokens=4, requests=6
        )
        assert self.scan(database, tmp_path, now).files_read == 0

    def test_a_home_without_sessions_is_an_empty_scan(self, database: Database, tmp_path: Path, now: datetime) -> None:
        assert self.scan(database, tmp_path, now).files_read == 0

    def test_scanning_the_same_home_under_another_spelling_does_not_count_it_again(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        home = tmp_path / "home"
        link = tmp_path / "link"
        link.symlink_to(home)
        self.usage_file(home).write_text(json.dumps({"turns": [grok_turn(1, now, 100, 0, 10)]}))

        self.scan(database, home, now)
        counted = total(database)
        self.scan(database, home / ".." / "home", now)
        self.scan(database, link, now)

        assert counted.input_tokens == 100
        assert total(database) == counted
        assert len(cursor_keys(database)) == 1


class TestScannerIdentity:
    def test_each_scanner_names_its_provider(self, tmp_path: Path) -> None:
        scanners = [ClaudeTranscriptScanner(tmp_path), CodexSessionScanner(tmp_path), GrokSessionScanner(tmp_path)]

        assert [scanner.provider for scanner in scanners] == [Provider.CLAUDE, Provider.CODEX, Provider.GROK]


class TestUnreadableSessionFiles:
    def test_claude_reports_a_file_that_vanished_and_one_it_may_not_open(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        project = tmp_path / "projects" / "proj"
        append(project / "good.jsonl", claude_line("good", now))
        append(project / "locked.jsonl", claude_line("locked", now))
        (project / "locked.jsonl").chmod(0)
        (project / "vanished.jsonl").symlink_to(project / "no-such-target.jsonl")

        report = ClaudeTranscriptScanner(tmp_path / "projects").scan(
            database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS
        )

        assert (report.files_read, report.files_failed) == (1, 2)
        assert report.first_failure == "locked.jsonl: PermissionError"
        assert total(database).requests == 1

    def test_codex_reports_a_file_that_vanished_and_one_it_may_not_open(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        sessions = tmp_path / "sessions"
        count = codex_count(now, codex_totals(5, 0, 1), codex_totals(5, 0, 1))
        append(sessions / "rollout-good.jsonl", codex_meta("workspace-a"), count)
        append(sessions / "rollout-locked.jsonl", codex_meta("workspace-a"), count)
        (sessions / "rollout-locked.jsonl").chmod(0)
        (sessions / "rollout-vanished.jsonl").symlink_to(sessions / "no-such-target.jsonl")

        report = CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert (report.files_read, report.files_failed) == (1, 2)
        assert report.first_failure == "rollout-locked.jsonl: PermissionError"
        assert total(database).requests == 1

    def test_grok_reports_a_usage_file_it_cannot_read_and_names_its_session(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        (tmp_path / "sessions" / "project" / "session-broken" / "usage.json").mkdir(parents=True)

        report = GrokSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert (report.files_read, report.files_failed) == (0, 1)
        assert report.first_failure == "session-broken: IsADirectoryError"


class TestSessionDirectoriesThatCannotBeListed:
    def test_claude_reports_a_project_directory_it_cannot_list_and_still_reads_the_others(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        projects = tmp_path / "projects"
        append(projects / "listable" / "good.jsonl", claude_line("good", now))
        append(projects / "locked" / "hidden.jsonl", claude_line("hidden", now))
        scanner = ClaudeTranscriptScanner(projects)
        (projects / "locked").chmod(0)
        try:
            report = scanner.scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)
        finally:
            (projects / "locked").chmod(0o700)

        assert (report.files_read, report.files_failed, report.first_failure) == (1, 1, "locked: PermissionError")
        assert total(database).requests == 1

        assert scanner.scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS).files_read == 1
        assert total(database).requests == 2

    def test_codex_reports_a_day_directory_it_cannot_list_and_still_reads_the_archive(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        count = codex_count(now, codex_totals(5, 0, 1), codex_totals(5, 0, 1))
        locked = tmp_path / "sessions" / "2026" / "10" / "03"
        append(locked / "rollout-hidden.jsonl", codex_meta("workspace-a"), count)
        append(tmp_path / "archived_sessions" / "rollout-archived.jsonl", codex_meta("workspace-a"), count)
        locked.chmod(0)
        try:
            report = CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)
        finally:
            locked.chmod(0o700)

        assert (report.files_read, report.files_failed, report.first_failure) == (1, 1, "03: PermissionError")
        assert total(database).requests == 1

    def test_grok_reports_a_project_directory_it_cannot_list_and_still_reads_the_others(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        sessions = tmp_path / "sessions"
        for project in ("listable", "locked"):
            usage_file = sessions / project / "session-1" / "usage.json"
            usage_file.parent.mkdir(parents=True)
            usage_file.write_text(json.dumps({"turns": [grok_turn(1, now, 10, 0, 1)]}))
        (sessions / "locked").chmod(0)
        try:
            report = GrokSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)
        finally:
            (sessions / "locked").chmod(0o700)

        assert (report.files_read, report.files_failed, report.first_failure) == (1, 1, "locked: PermissionError")
        assert total(database).requests == 3

    def test_a_session_tree_that_cannot_be_listed_at_all_is_one_failure(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        projects = tmp_path / "projects"
        append(projects / "proj" / "hidden.jsonl", claude_line("hidden", now))
        projects.chmod(0)
        try:
            report = ClaudeTranscriptScanner(projects).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)
        finally:
            projects.chmod(0o700)

        assert (report.files_read, report.files_failed, report.first_failure) == (0, 1, "projects: PermissionError")


class TestSessionFiles:
    def tree(self, root: Path, *relative_paths: str) -> None:
        for relative in relative_paths:
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text("")

    def test_lists_matching_files_at_any_depth_in_sorted_order(self, tmp_path: Path) -> None:
        self.tree(tmp_path, "b/two.jsonl", "a/deep/er/one.jsonl", "a/notes.txt", "top.jsonl")

        found, report = session_files(tmp_path, "*.jsonl")

        assert [str(path.relative_to(tmp_path)) for path in found] == [
            "a/deep/er/one.jsonl",
            "b/two.jsonl",
            "top.jsonl",
        ]
        assert report == ScanReport()

    def test_with_a_depth_only_files_exactly_that_deep_are_listed_and_nothing_deeper_is_entered(
        self, tmp_path: Path
    ) -> None:
        self.tree(
            tmp_path,
            "usage.json",
            "project/usage.json",
            "project/session/usage.json",
            "project/session/deeper/usage.json",
            "project/session/never-entered/file",
        )
        never_entered = tmp_path / "project" / "session" / "never-entered"
        never_entered.chmod(0)
        try:
            found, report = session_files(tmp_path, "usage.json", depth=2)
        finally:
            never_entered.chmod(0o700)

        assert [str(path.relative_to(tmp_path)) for path in found] == ["project/session/usage.json"]
        assert report == ScanReport()

    def test_a_directory_named_like_a_session_file_is_listed_so_that_reading_it_fails_and_is_reported(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "project" / "session" / "usage.json").mkdir(parents=True)

        found, _ = session_files(tmp_path, "usage.json", depth=2)

        assert found == [tmp_path / "project" / "session" / "usage.json"]

    def test_a_link_to_a_directory_is_not_followed(self, tmp_path: Path) -> None:
        self.tree(tmp_path, "elsewhere/outside.jsonl", "sessions/inside.jsonl")
        (tmp_path / "sessions" / "link").symlink_to(tmp_path / "elsewhere", target_is_directory=True)

        found, _ = session_files(tmp_path / "sessions", "*.jsonl")

        assert found == [tmp_path / "sessions" / "inside.jsonl"]

    @pytest.mark.parametrize("kind", ["missing", "a file", "under a file"])
    def test_a_root_that_is_not_a_directory_holds_no_sessions_and_is_no_failure(
        self, tmp_path: Path, kind: str
    ) -> None:
        (tmp_path / "file").write_text("")
        location = {
            "missing": tmp_path / "absent",
            "a file": tmp_path / "file",
            "under a file": tmp_path / "file" / "x",
        }

        assert session_files(location[kind], "*.jsonl") == ([], ScanReport())

    def test_a_root_that_cannot_be_reached_is_a_failure(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        self.tree(home, "sessions/rollout-hidden.jsonl")
        home.chmod(0)
        try:
            found, report = session_files(home / "sessions", "rollout-*.jsonl")
        finally:
            home.chmod(0o700)

        assert found == []
        assert report == ScanReport(files_failed=1, first_failure="sessions: PermissionError")

    def test_every_directory_that_cannot_be_listed_is_counted_and_the_first_is_named(self, tmp_path: Path) -> None:
        self.tree(tmp_path, "a-locked/one.jsonl", "b-listable/two.jsonl", "c-locked/three.jsonl")
        locked = [tmp_path / "a-locked", tmp_path / "c-locked"]
        for directory in locked:
            directory.chmod(0)
        try:
            found, report = session_files(tmp_path, "*.jsonl")
        finally:
            for directory in locked:
                directory.chmod(0o700)

        assert found == [tmp_path / "b-listable" / "two.jsonl"]
        assert report.files_failed == 2
        assert report.first_failure in {"a-locked: PermissionError", "c-locked: PermissionError"}

    def test_an_entry_that_cannot_be_classified_is_a_failure_and_its_sibling_is_still_listed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.tree(tmp_path, "visible/one.jsonl", "hidden/two.jsonl")
        monkeypatch.setattr(scanner_base.os, "scandir", _scandir_hiding("hidden"))

        found, report = session_files(tmp_path, "*.jsonl")

        assert found == [tmp_path / "visible" / "one.jsonl"]
        assert report == ScanReport(files_failed=1, first_failure="hidden: PermissionError")

    def test_a_directory_error_without_a_filename_is_recorded_and_the_sibling_is_still_listed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.tree(tmp_path, "kept/one.jsonl", "unreadable/two.jsonl")
        real_scandir = os.scandir

        def scandir(path: str | os.PathLike[str]) -> object:
            if Path(path).name == "unreadable":
                raise OSError(5, "synthetic directory read failure")
            return real_scandir(path)

        monkeypatch.setattr(scanner_base.os, "scandir", scandir)

        found, report = session_files(tmp_path, "*.jsonl")

        assert found == [tmp_path / "kept" / "one.jsonl"]
        assert report == ScanReport(files_failed=1, first_failure="unreadable: OSError")


class TestRecordsThatAreNotUsage:
    @pytest.mark.parametrize(
        "line",
        [
            b'["assistant", "usage"]',
            b'{"type": "assistant", "message": "usage"}',
            b'{"type": "assistant", "message": {"usage": "none"}}',
            b'{"type": "user", "message": {"role": "assistant", "usage": {"output_tokens": 5}}}',
        ],
    )
    def test_a_claude_line_that_only_looks_like_an_assistant_response_is_ignored(self, line: bytes) -> None:
        assert parse_transcript_line(line) is None

    def test_codex_lines_that_only_mention_usage_change_nothing(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(
            tmp_path / "sessions" / "rollout-odd.jsonl",
            codex_meta("workspace-a"),
            codex_turn("gpt-x"),
            {"type": "turn_context", "payload": {"model": 5}},
            {"type": "response_item", "payload": {"type": "message"}},
            b'["token_count"]',
            {"type": "event_msg", "payload": "token_count"},
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "token_count"}},
            codex_count(now, codex_totals(5, 0, 1), codex_totals(5, 0, 1)),
        )

        report = CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert report.malformed_records == 0
        assert [(model, counts.input_tokens) for (_, _, model), counts in usage_rows(database).items()] == [
            ("gpt-x", 5)
        ]

    def test_a_codex_event_whose_totals_are_not_an_object_is_malformed(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        broken = codex_count(now, codex_totals(5, 0, 1), None)
        broken["payload"]["info"]["total_token_usage"] = [5, 0, 1]
        append(tmp_path / "sessions" / "rollout-odd.jsonl", codex_meta("workspace-a"), broken)

        report = CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert report.malformed_records == 1
        assert total(database) == TokenCounts()

    @pytest.mark.parametrize("request_usage", [[100, 0, 10], "100", 100])
    def test_a_codex_event_whose_own_request_usage_is_not_an_object_is_malformed_and_never_becomes_the_baseline(
        self, database: Database, tmp_path: Path, now: datetime, request_usage: object
    ) -> None:
        broken = codex_count(now, codex_totals(200, 0, 20), codex_totals(100, 0, 10))
        broken["payload"]["info"]["last_token_usage"] = request_usage
        append(
            tmp_path / "sessions" / "rollout-odd.jsonl",
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
            broken,
            codex_count(now, codex_totals(250, 0, 25), codex_totals(50, 0, 5)),
        )

        report = CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert report.malformed_records == 1
        assert total(database) == TokenCounts(input_tokens=250, output_tokens=25, requests=2)


class TestSessionFileLines:
    def test_a_valid_line_is_parsed(self) -> None:
        assert parse_json_line(b'{"type": "assistant"}') == {"type": "assistant"}

    @pytest.mark.parametrize("line", [b'{"type": truncated', b'{"model": "caf\xe9"}', b"[" * 200_000, b""])
    def test_a_line_that_cannot_be_parsed_for_any_reason_is_a_value_error(self, line: bytes) -> None:
        with pytest.raises(ValueError):
            parse_json_line(line)


class TestCountsTooLargeToBeReal:
    def test_claude_counts_the_record_as_malformed_and_keeps_reading(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        absurd = claude_line("absurd", now, output=MAX_TOKENS_PER_RECORD + 1)
        nested = b'{"type":"assistant","message":{"usage":' + b"[" * 200_000
        append(tmp_path / "projects" / "proj" / "session.jsonl", absurd, nested, claude_line("good", now, output=7))

        report = ClaudeTranscriptScanner(tmp_path / "projects").scan(
            database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS
        )

        assert (report.files_read, report.files_failed, report.malformed_records) == (1, 0, 2)
        assert total(database).output_tokens == 7

    def test_the_largest_believable_count_is_stored(self, database: Database, tmp_path: Path, now: datetime) -> None:
        append(tmp_path / "projects" / "proj" / "session.jsonl", claude_line("big", now, output=MAX_TOKENS_PER_RECORD))

        ClaudeTranscriptScanner(tmp_path / "projects").scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert total(database).output_tokens == MAX_TOKENS_PER_RECORD

    def test_codex_skips_the_record_and_measures_the_next_one_from_the_last_good_total(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(
            tmp_path / "sessions" / "rollout-big.jsonl",
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
            codex_count(now, codex_totals(MAX_TOKENS_PER_RECORD + 1, 0, 20), codex_totals(100, 0, 10)),
            codex_count(now, codex_totals(250, 0, 25), codex_totals(50, 0, 5)),
        )

        report = CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert report.malformed_records == 1
        assert total(database) == TokenCounts(input_tokens=250, output_tokens=25, requests=2)

    def test_grok_reports_the_file_instead_of_ingesting_it(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        usage_file = tmp_path / "sessions" / "project" / "session-1" / "usage.json"
        usage_file.parent.mkdir(parents=True)
        turn = grok_turn(1, now, 10, 0, 1)
        turn["outputTokens"] = MAX_TOKENS_PER_RECORD + 1
        usage_file.write_text(json.dumps({"turns": [turn]}))

        report = GrokSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert (report.files_read, report.files_failed) == (0, 1)
        assert total(database) == TokenCounts()


class TestTimestampsAtTheEdgeOfTheCalendar:
    EDGE = "9999-12-31T23:59:59-05:00"

    def test_claude_counts_the_record_as_malformed(self, database: Database, tmp_path: Path, now: datetime) -> None:
        undatable = claude_line("edge", now)
        undatable["timestamp"] = self.EDGE
        append(tmp_path / "projects" / "proj" / "session.jsonl", undatable, claude_line("good", now, output=7))

        report = ClaudeTranscriptScanner(tmp_path / "projects").scan(
            database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS
        )

        assert report.malformed_records == 1
        assert total(database).output_tokens == 7

    def test_codex_counts_the_record_as_malformed(self, database: Database, tmp_path: Path, now: datetime) -> None:
        undatable = codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10))
        undatable["timestamp"] = self.EDGE
        append(tmp_path / "sessions" / "rollout-edge.jsonl", codex_meta("workspace-a"), undatable)

        report = CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert report.malformed_records == 1
        assert total(database) == TokenCounts()

    def test_grok_reports_the_file(self, database: Database, tmp_path: Path, now: datetime) -> None:
        usage_file = tmp_path / "sessions" / "project" / "session-1" / "usage.json"
        usage_file.parent.mkdir(parents=True)
        turn = grok_turn(1, now, 10, 0, 1)
        turn["endedAt"] = self.EDGE
        usage_file.write_text(json.dumps({"turns": [turn]}))

        report = GrokSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert (report.files_read, report.files_failed) == (0, 1)


class TestRecordsWithNoUsableIdentity:
    def test_claude_responses_without_an_id_are_malformed_not_collapsed_into_one(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        records = []
        for output in (11, 13):
            record = claude_line("unused", now, output=output)
            del record["uuid"]
            del record["message"]["id"]
            records.append(record)
        append(tmp_path / "projects" / "proj" / "session.jsonl", *records, claude_line("good", now, output=7))

        report = ClaudeTranscriptScanner(tmp_path / "projects").scan(
            database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS
        )

        assert report.malformed_records == 2
        assert total(database).output_tokens == 7

    @pytest.mark.parametrize("response_id", [None, "", "   ", "\t", 7, ["msg"]], ids=repr)
    def test_a_claude_line_s_own_id_never_stands_in_for_a_response_id_that_is_missing_or_blank(
        self, database: Database, tmp_path: Path, now: datetime, response_id: object
    ) -> None:
        records = []
        for line_id, output in (("line-1", 11), ("line-2", 13)):
            record = claude_line("unused", now, output=output)
            record["uuid"] = line_id
            if response_id is None:
                del record["message"]["id"]
            else:
                record["message"]["id"] = response_id
            records.append(record)
        append(tmp_path / "projects" / "proj" / "session.jsonl", *records, claude_line("good", now, output=7))

        report = ClaudeTranscriptScanner(tmp_path / "projects").scan(
            database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS
        )

        assert report.malformed_records == 2
        assert (total(database).output_tokens, total(database).requests) == (7, 1)

    @pytest.mark.parametrize("repeat_has", ["no requestId", "a null requestId", "another requestId"])
    def test_a_claude_response_is_one_response_whatever_request_id_its_lines_carry(
        self, database: Database, tmp_path: Path, now: datetime, repeat_has: str
    ) -> None:
        first = claude_line("msg-1", now, output=100)
        repeat = claude_line("msg-1", now, output=100)
        if repeat_has == "no requestId":
            del repeat["requestId"]
        elif repeat_has == "a null requestId":
            repeat["requestId"] = None
        else:
            repeat["requestId"] = "req-other"
        append(tmp_path / "projects" / "proj" / "session.jsonl", first, repeat)

        ClaudeTranscriptScanner(tmp_path / "projects").scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert (total(database).output_tokens, total(database).requests) == (100, 1)

    def test_two_claude_responses_with_one_request_id_are_two_responses(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        first = claude_line("msg-1", now, output=100)
        second = claude_line("msg-2", now, output=30)
        second["requestId"] = first["requestId"]
        append(tmp_path / "projects" / "proj" / "session.jsonl", first, second)

        ClaudeTranscriptScanner(tmp_path / "projects").scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert (total(database).output_tokens, total(database).requests) == (130, 2)

    @pytest.mark.parametrize("workspace", ["", "   ", "\t"])
    def test_a_codex_session_with_a_blank_workspace_is_unattributed(
        self, database: Database, tmp_path: Path, now: datetime, workspace: str
    ) -> None:
        append(
            tmp_path / "sessions" / "rollout-blank.jsonl",
            codex_meta(workspace),
            codex_count(now, codex_totals(5, 0, 1), codex_totals(5, 0, 1)),
        )

        CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert [account for account, _, _ in usage_rows(database)] == [UNATTRIBUTED_ACCOUNT_ID]
        with database.read() as conn:
            assert db.list_accounts(conn) == []

    def test_a_codex_workspace_id_has_its_surrounding_whitespace_removed(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(
            tmp_path / "sessions" / "rollout-padded.jsonl",
            codex_meta("  workspace-a \n"),
            codex_count(now, codex_totals(5, 0, 1), codex_totals(5, 0, 1)),
        )

        CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        with database.read() as conn:
            assert [account.external_id for account in db.list_accounts(conn)] == ["workspace-a"]


class TestOneFileNeverStopsTheRest:
    @pytest.mark.parametrize(
        "error",
        [
            PermissionError("denied"),
            ValueError("quotes a prompt"),
            OverflowError("too big"),
            RecursionError("too deep"),
        ],
    )
    def test_a_file_that_cannot_be_read_or_interpreted_is_counted_and_named_by_error_type_only(
        self, error: Exception
    ) -> None:
        def scan_file() -> ScanReport:
            raise error

        assert isolated("session.jsonl", scan_file) == ScanReport(
            files_failed=1, first_failure=f"session.jsonl: {type(error).__name__}"
        )

    def test_a_scan_that_succeeds_is_returned_unchanged(self) -> None:
        assert isolated("session.jsonl", lambda: ScanReport(files_read=1, malformed_records=2)) == ScanReport(
            files_read=1, malformed_records=2
        )

    def test_a_failure_of_the_store_is_not_blamed_on_the_file(self) -> None:
        def scan_file() -> ScanReport:
            raise sqlite3.OperationalError("disk I/O error")

        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            isolated("session.jsonl", scan_file)

    def test_claude_usage_that_cannot_be_stored_fails_its_own_file_leaves_nothing_behind_and_is_retried(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        project = tmp_path / "projects" / "proj"
        append(project / "a-unattributable.jsonl", claude_line("first", now, output=5))
        append(project / "b-fine.jsonl", claude_line("second", now + timedelta(hours=1), output=7))

        def resolve(moment: datetime) -> int:
            if moment == now:
                raise ValueError("cannot attribute")
            return ACCOUNT_A

        scanner = ClaudeTranscriptScanner(tmp_path / "projects")
        report = scanner.scan(database, resolve, now, BACKFILL_DAYS)

        assert (report.files_read, report.files_failed) == (1, 1)
        assert report.first_failure == "a-unattributable.jsonl: ValueError"
        assert total(database).output_tokens == 7

        retried = scanner.scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert (retried.files_read, retried.files_failed) == (1, 0)
        assert total(database).output_tokens == 12

    def test_grok_usage_that_cannot_be_attributed_fails_its_own_session_only(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        for session, ended_at in (("session-a", now), ("session-b", now + timedelta(hours=1))):
            usage_file = tmp_path / "sessions" / "project" / session / "usage.json"
            usage_file.parent.mkdir(parents=True)
            usage_file.write_text(json.dumps({"turns": [grok_turn(1, ended_at, 10, 0, 1)]}))

        def resolve(moment: datetime) -> int:
            if moment == now:
                raise ValueError("cannot attribute")
            return ACCOUNT_A

        report = GrokSessionScanner(tmp_path).scan(database, resolve, now, BACKFILL_DAYS)

        assert (report.files_read, report.files_failed, report.first_failure) == (1, 1, "session-a: ValueError")
        assert total(database).requests == 3


class TestSavedScanPositions:
    ROLLOUT = "rollout-2026-10-03T10-00-00-aaaa.jsonl"

    def save(self, database: Database, provider: Provider, column: str, value: object) -> None:
        assert column in {"state_json", "byte_offset"}
        with database.connect() as conn:
            conn.execute(f"UPDATE scan_cursors SET {column} = ? WHERE provider = ?", (value, provider.value))  # noqa: S608 - column is one of two literals

    def scanned_codex_rollout(self, database: Database, home: Path, now: datetime) -> Path:
        rollout = home / "sessions" / self.ROLLOUT
        append(
            rollout,
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
        )
        CodexSessionScanner(home).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)
        return rollout

    @pytest.mark.parametrize(
        "saved_state",
        [
            "[]",
            '"text"',
            '{"previous_totals": "many"}',
            '{"previous_totals": {"input_tokens": -5}}',
            '{"previous_totals": {"input_tokens": 1e30}}',
            '{"previous_totals": {"input_tokens": 1000000000001}}',
        ],
    )
    def test_codex_state_that_cannot_be_read_back_makes_the_file_be_read_afresh_without_double_counting(
        self, database: Database, tmp_path: Path, now: datetime, caplog: pytest.LogCaptureFixture, saved_state: str
    ) -> None:
        rollout = self.scanned_codex_rollout(database, tmp_path, now)
        self.save(database, Provider.CODEX, "state_json", saved_state)
        append(rollout, codex_count(now, codex_totals(250, 0, 25), codex_totals(150, 0, 15)))

        with caplog.at_level(logging.WARNING, logger="usagedash.scanners.codex"):
            report = CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert (report.files_read, report.files_failed) == (1, 0)
        assert total(database) == TokenCounts(input_tokens=250, output_tokens=25, requests=2)
        assert "could not be read back; reading the file afresh" in caplog.text

    @pytest.mark.parametrize("saved_offset", [-1, 10**9])
    def test_a_saved_codex_position_outside_the_file_makes_it_be_read_afresh(
        self, database: Database, tmp_path: Path, now: datetime, saved_offset: int
    ) -> None:
        rollout = self.scanned_codex_rollout(database, tmp_path, now)
        self.save(database, Provider.CODEX, "byte_offset", saved_offset)
        append(rollout, codex_count(now, codex_totals(250, 0, 25), codex_totals(150, 0, 15)))

        report = CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert (report.files_read, report.files_failed) == (1, 0)
        assert total(database) == TokenCounts(input_tokens=250, output_tokens=25, requests=2)

    @pytest.mark.parametrize("saved_offset", [-1, 10**9])
    def test_a_saved_claude_position_outside_the_file_makes_it_be_read_from_the_start_without_double_counting(
        self, database: Database, tmp_path: Path, now: datetime, saved_offset: int
    ) -> None:
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        append(transcript, claude_line("first", now, output=5))
        scanner = ClaudeTranscriptScanner(tmp_path / "projects")
        scanner.scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)
        self.save(database, Provider.CLAUDE, "byte_offset", saved_offset)
        append(transcript, claude_line("second", now, output=7))

        report = scanner.scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert (report.files_read, report.files_failed) == (1, 0)
        assert total(database).output_tokens == 12
        assert total(database).requests == 2


class TestAFileRewrittenInPlace:
    def rewrite(self, session_file: Path, *records: Any, newline: bool = True) -> None:
        before = session_file.stat().st_mtime_ns
        session_file.write_bytes(b"")
        append(session_file, *records, newline=newline)
        os.utime(session_file, ns=(before + 1_000_000_000, before + 1_000_000_000))

    def codex_scan(self, database: Database, home: Path, now: datetime) -> ScanReport:
        return CodexSessionScanner(home).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

    def claude_scan(self, database: Database, projects: Path, now: datetime) -> ScanReport:
        return ClaudeTranscriptScanner(projects).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

    def test_a_codex_rollout_rewritten_to_the_same_size_is_read_afresh(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = tmp_path / "sessions" / "rollout-rewritten.jsonl"
        append(rollout, codex_meta("workspace-a"), codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)))
        size = rollout.stat().st_size
        self.codex_scan(database, tmp_path, now)

        self.rewrite(
            rollout, codex_meta("workspace-a"), codex_count(now, codex_totals(200, 0, 20), codex_totals(200, 0, 20))
        )
        report = self.codex_scan(database, tmp_path, now)

        assert rollout.stat().st_size == size
        assert report.files_read == 1
        assert total(database) == TokenCounts(input_tokens=200, output_tokens=20, requests=1)

    def test_a_codex_rollout_replaced_with_its_size_and_mtime_kept_is_read_afresh(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = tmp_path / "sessions" / "rollout-replaced.jsonl"
        original = (codex_meta("workspace-a"), codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)))
        append(rollout, *original)
        self.codex_scan(database, tmp_path, now)
        replacement = bytes_of(
            codex_meta("workspace-a"), codex_count(now, codex_totals(200, 0, 20), codex_totals(200, 0, 20))
        )

        put_in_place_keeping_size_and_mtime(rollout, replacement)
        report = self.codex_scan(database, tmp_path, now)

        assert report.files_read == 1
        assert total(database) == TokenCounts(input_tokens=200, output_tokens=20, requests=1)

    def test_a_codex_rollout_rewritten_in_place_with_its_mtime_restored_is_read_afresh(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = tmp_path / "sessions" / "rollout-rewritten.jsonl"
        append(rollout, codex_meta("workspace-a"), codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)))
        self.codex_scan(database, tmp_path, now)
        replacement = bytes_of(
            codex_meta("workspace-a"), codex_count(now, codex_totals(200, 0, 20), codex_totals(200, 0, 20))
        )

        rewrite_in_place_keeping_mtime(rollout, replacement)
        report = self.codex_scan(database, tmp_path, now)

        assert report.files_read == 1
        assert total(database) == TokenCounts(input_tokens=200, output_tokens=20, requests=1)

    def test_a_claude_transcript_replaced_with_its_size_and_mtime_kept_is_read_afresh(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        append(transcript, claude_line("msg-a", now, output=5), claude_line("msg-b", now, output=6))
        self.claude_scan(database, tmp_path / "projects", now)
        replacement = bytes_of(claude_line("msg-c", now, output=7), claude_line("msg-d", now, output=8))

        put_in_place_keeping_size_and_mtime(transcript, replacement)
        report = self.claude_scan(database, tmp_path / "projects", now)

        assert report.files_read == 1
        assert (total(database).output_tokens, total(database).requests) == (5 + 6 + 7 + 8, 4)

    def test_a_claude_transcript_rewritten_in_place_with_its_mtime_restored_is_read_afresh(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        append(transcript, claude_line("msg-a", now, output=5), claude_line("msg-b", now, output=6))
        self.claude_scan(database, tmp_path / "projects", now)
        replacement = bytes_of(claude_line("msg-c", now, output=7), claude_line("msg-d", now, output=8))

        rewrite_in_place_keeping_mtime(transcript, replacement)
        report = self.claude_scan(database, tmp_path / "projects", now)

        assert report.files_read == 1
        assert (total(database).output_tokens, total(database).requests) == (5 + 6 + 7 + 8, 4)

    def test_a_codex_rollout_rewritten_longer_is_read_afresh_and_not_from_the_old_position(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = tmp_path / "sessions" / "rollout-rewritten.jsonl"
        append(rollout, codex_meta("workspace-a"), codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)))
        self.codex_scan(database, tmp_path, now)

        self.rewrite(
            rollout,
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(200, 0, 20), codex_totals(200, 0, 20)),
            codex_count(now, codex_totals(400, 0, 40), codex_totals(200, 0, 20)),
            codex_count(now, codex_totals(600, 0, 60), codex_totals(200, 0, 20)),
        )
        self.codex_scan(database, tmp_path, now)

        assert total(database) == TokenCounts(input_tokens=600, output_tokens=60, requests=3)

    def test_a_codex_rollout_rewritten_shorter_than_it_was_and_no_shorter_than_what_was_read_is_read_afresh(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = tmp_path / "sessions" / "rollout-rewritten.jsonl"
        append(
            rollout,
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
            codex_count(now, codex_totals(200, 0, 20), codex_totals(100, 0, 10)),
        )
        read_up_to = rollout.stat().st_size
        append(rollout, b'{"type":"event_msg","payload":{"type":"token_count","info":' + b" " * 300, newline=False)
        self.codex_scan(database, tmp_path, now)
        assert total(database) == TokenCounts(input_tokens=200, output_tokens=20, requests=2)

        self.rewrite(
            rollout,
            codex_meta("workspace-a"),
            codex_count(now, codex_totals(500, 0, 50), codex_totals(500, 0, 50)),
            codex_count(now, codex_totals(900, 0, 90), codex_totals(400, 0, 40)),
        )
        self.codex_scan(database, tmp_path, now)

        assert rollout.stat().st_size == read_up_to
        assert total(database) == TokenCounts(input_tokens=900, output_tokens=90, requests=2)

    def test_a_long_codex_rollout_replaced_with_the_same_tail_is_read_from_the_start(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = tmp_path / "sessions" / "rollout-replaced.jsonl"
        original, replacement = with_a_tail_past_the_anchor(
            bytes_of(
                codex_meta("workspace-a"),
                codex_count(now, codex_totals(10, 0, 10), codex_totals(10, 0, 10)),
            ),
            bytes_of(
                codex_meta("workspace-b"),
                codex_count(now, codex_totals(90, 0, 90), codex_totals(90, 0, 90)),
            ),
        )
        rollout.parent.mkdir(parents=True)
        rollout.write_bytes(original)
        self.codex_scan(database, tmp_path, now)

        put_in_place_keeping_size_and_mtime(rollout, replacement)
        report = self.codex_scan(database, tmp_path, now)

        with database.connect() as conn:
            accounts = {account.external_id: account.id for account in db.list_accounts(conn)}
        assert report.files_read == 1
        assert usage_rows(database) == {
            (accounts["workspace-b"], "2026-10-03T14", "unknown"): TokenCounts(
                input_tokens=90, output_tokens=90, requests=1
            )
        }
        assert self.codex_scan(database, tmp_path, now).files_read == 0

    def test_a_long_claude_transcript_replaced_with_the_same_tail_counts_the_new_responses(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        original, replacement = with_a_tail_past_the_anchor(
            bytes_of(claude_line("msg-a", now, output=5)),
            bytes_of(claude_line("msg-c", now, output=7)),
        )
        transcript.parent.mkdir(parents=True)
        transcript.write_bytes(original)
        self.claude_scan(database, tmp_path / "projects", now)

        put_in_place_keeping_size_and_mtime(transcript, replacement)
        report = self.claude_scan(database, tmp_path / "projects", now)

        assert report.files_read == 1
        assert (total(database).output_tokens, total(database).requests) == (5 + 7, 2)
        assert self.claude_scan(database, tmp_path / "projects", now).files_read == 0

    def test_an_appended_claude_line_after_a_long_file_does_not_reread_the_prefix(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        original, _ = with_a_tail_past_the_anchor(
            bytes_of(undated(claude_line("msg-undated", now)), claude_line("msg-a", now, output=5)),
            b"",
        )
        transcript.parent.mkdir(parents=True)
        transcript.write_bytes(original)
        first = self.claude_scan(database, tmp_path / "projects", now)
        append(transcript, claude_line("msg-b", now, output=6))

        second = self.claude_scan(database, tmp_path / "projects", now)

        assert first.malformed_records >= 1
        assert (second.files_read, second.malformed_records) == (1, 0)
        assert (total(database).output_tokens, total(database).requests) == (11, 2)

    def test_a_codex_rollout_that_only_grew_is_still_read_on_from_where_it_was(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        rollout = tmp_path / "sessions" / "rollout-appended.jsonl"
        append(rollout, codex_meta("workspace-a"), codex_count(now, codex_totals(100, 0, 10), None))
        self.codex_scan(database, tmp_path, now)
        append(rollout, codex_count(now, codex_totals(160, 0, 25), codex_totals(999, 0, 999)))

        self.codex_scan(database, tmp_path, now)

        assert total(database) == TokenCounts(input_tokens=60, output_tokens=15, requests=1)

    def test_a_claude_transcript_rewritten_to_the_same_size_is_read_from_the_start(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        append(transcript, claude_line("msg-a", now, output=5), claude_line("msg-b", now, output=6))
        size = transcript.stat().st_size
        self.claude_scan(database, tmp_path / "projects", now)

        self.rewrite(transcript, claude_line("msg-c", now, output=7), claude_line("msg-d", now, output=8))
        self.claude_scan(database, tmp_path / "projects", now)

        assert transcript.stat().st_size == size
        assert (total(database).output_tokens, total(database).requests) == (26, 4)

    def test_a_claude_transcript_that_only_grew_is_read_on_from_where_it_was(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        append(transcript, undated(claude_line("msg-undated", now)), claude_line("msg-a", now, output=5))
        first = self.claude_scan(database, tmp_path / "projects", now)
        append(transcript, claude_line("msg-b", now, output=6))

        second = self.claude_scan(database, tmp_path / "projects", now)

        assert (first.malformed_records, second.malformed_records) == (1, 0)
        assert (second.files_read, total(database).output_tokens, total(database).requests) == (1, 11, 2)

    @pytest.mark.parametrize("saved_state", ["[]", "{}", '{"anchor": null}', '{"anchor": 7}', '{"anchor": ""}'])
    def test_a_claude_position_saved_without_what_the_file_held_before_it_is_not_resumed(
        self, database: Database, tmp_path: Path, now: datetime, saved_state: str
    ) -> None:
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        append(transcript, undated(claude_line("msg-undated", now)), claude_line("msg-a", now, output=5))
        self.claude_scan(database, tmp_path / "projects", now)
        with database.connect() as conn:
            conn.execute("UPDATE scan_cursors SET state_json = ? WHERE provider = 'claude'", (saved_state,))
        append(transcript, claude_line("msg-b", now, output=6))

        report = self.claude_scan(database, tmp_path / "projects", now)

        assert (report.files_read, report.malformed_records) == (1, 1)
        assert (total(database).output_tokens, total(database).requests) == (11, 2)

    @pytest.mark.parametrize("saved_offset", [-1, 10**9])
    def test_a_codex_position_outside_an_unchanged_file_is_put_right_at_once_and_later_growth_is_counted(
        self, database: Database, tmp_path: Path, now: datetime, saved_offset: int
    ) -> None:
        rollout = tmp_path / "sessions" / "rollout-damaged-position.jsonl"
        append(rollout, codex_meta("workspace-a"), codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)))
        self.codex_scan(database, tmp_path, now)
        with database.connect() as conn:
            conn.execute("UPDATE scan_cursors SET byte_offset = ? WHERE provider = 'codex'", (saved_offset,))

        report = self.codex_scan(database, tmp_path, now)

        with database.read() as conn:
            (cursor,) = db.load_cursors(conn, Provider.CODEX).values()
        assert (report.files_read, cursor.byte_offset) == (1, rollout.stat().st_size)
        assert total(database) == TokenCounts(input_tokens=100, output_tokens=10, requests=1)
        append(rollout, codex_count(now, codex_totals(200, 0, 20), codex_totals(100, 0, 10)))
        self.codex_scan(database, tmp_path, now)
        assert total(database) == TokenCounts(input_tokens=200, output_tokens=20, requests=2)

    @pytest.mark.parametrize("saved_offset", [-1, 10**9])
    def test_a_claude_position_outside_an_unchanged_file_is_put_right_at_once_and_later_growth_is_counted(
        self, database: Database, tmp_path: Path, now: datetime, saved_offset: int
    ) -> None:
        transcript = tmp_path / "projects" / "proj" / "session.jsonl"
        append(transcript, claude_line("msg-a", now, output=5))
        self.claude_scan(database, tmp_path / "projects", now)
        with database.connect() as conn:
            conn.execute("UPDATE scan_cursors SET byte_offset = ? WHERE provider = 'claude'", (saved_offset,))

        report = self.claude_scan(database, tmp_path / "projects", now)

        with database.read() as conn:
            (cursor,) = db.load_cursors(conn, Provider.CLAUDE).values()
        assert (report.files_read, cursor.byte_offset) == (1, transcript.stat().st_size)
        append(transcript, claude_line("msg-b", now, output=6))
        self.claude_scan(database, tmp_path / "projects", now)
        assert (total(database).output_tokens, total(database).requests) == (11, 2)


class TestAFileChangedWhileItIsScanned:
    ROLLOUT = "rollout-2026-10-03T10-00-00-aaaa.jsonl"
    CODEX_A_READ_ONCE = TokenCounts(input_tokens=400, output_tokens=40, requests=1)
    CODEX_A = TokenCounts(input_tokens=800, output_tokens=80, requests=2)
    CODEX_B = TokenCounts(input_tokens=300, output_tokens=30, requests=3)

    def change_the_file(self, monkeypatch: pytest.MonkeyPatch, moment: str, change: Callable[[], None]) -> None:
        assert moment in {"before the read", "during the read", "after the read"}

        class ChangedWhileOpen(OpenSessionFile):
            def lines(self) -> Iterator[bytes]:
                if moment == "before the read":
                    change()
                for number, line in enumerate(super().lines()):
                    if moment == "during the read" and number == 0:
                        change()
                    yield line
                if moment == "after the read":
                    change()

        monkeypatch.setattr(scanner_base, "OpenSessionFile", ChangedWhileOpen)

    def changed(self, how: str, path: Path, contents: bytes) -> Callable[[], None]:
        assert how in {"rewritten in place", "replaced by another file"}

        def rewrite_in_place() -> None:
            path.write_bytes(contents)

        def replace_by_another_file() -> None:
            put_in_place_of(path, contents)

        return rewrite_in_place if how == "rewritten in place" else replace_by_another_file

    def as_bytes(self, *records: dict[str, Any]) -> bytes:
        return b"".join(json.dumps(record).encode() + b"\n" for record in records)

    def codex_scan(self, database: Database, home: Path, now: datetime) -> ScanReport:
        return CodexSessionScanner(home).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

    def codex_session(self, now: datetime, each: int, requests: int) -> list[dict[str, Any]]:
        return [
            codex_meta("workspace-a"),
            *(
                codex_count(now, codex_totals(each * number, 0, each // 10 * number), codex_totals(each, 0, each // 10))
                for number in range(1, requests + 1)
            ),
        ]

    def codex_a(self, now: datetime) -> list[dict[str, Any]]:
        return self.codex_session(now, 400, 2)

    def codex_b(self, now: datetime) -> bytes:
        return self.as_bytes(*self.codex_session(now, 100, 3))

    def codex_rollout_read_once_and_then_grown(self, database: Database, home: Path, now: datetime) -> Path:
        rollout = home / "sessions" / self.ROLLOUT
        meta, first, second = self.codex_a(now)
        append(rollout, meta, first)
        self.codex_scan(database, home, now)
        append(rollout, second)
        return rollout

    @pytest.mark.parametrize("moment", ["before the read", "during the read"])
    def test_a_codex_rollout_rewritten_in_place_once_its_position_was_checked_fails_and_is_then_read_afresh(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch, moment: str
    ) -> None:
        rollout = self.codex_rollout_read_once_and_then_grown(database, tmp_path, now)
        with database.read() as conn:
            saved_before = db.load_cursors(conn, Provider.CODEX)
        self.change_the_file(monkeypatch, moment, self.changed("rewritten in place", rollout, self.codex_b(now)))

        failed = self.codex_scan(database, tmp_path, now)

        with database.read() as conn:
            assert db.load_cursors(conn, Provider.CODEX) == saved_before
        assert (failed.files_read, failed.files_failed) == (0, 1)
        assert failed.first_failure == f"{self.ROLLOUT}: SessionFileChangedError"
        assert total(database) == self.CODEX_A_READ_ONCE
        monkeypatch.undo()
        assert [self.codex_scan(database, tmp_path, now).files_read for _scan in range(2)] == [1, 0]
        assert total(database) == self.CODEX_B

    def test_a_codex_rollout_rewritten_in_place_while_it_is_first_read_fails_and_is_then_read_afresh(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rollout = tmp_path / "sessions" / self.ROLLOUT
        append(rollout, *self.codex_a(now))
        self.change_the_file(
            monkeypatch, "during the read", self.changed("rewritten in place", rollout, self.codex_b(now))
        )

        failed = self.codex_scan(database, tmp_path, now)

        with database.read() as conn:
            assert db.load_cursors(conn, Provider.CODEX) == {}
        assert (failed.files_read, failed.files_failed) == (0, 1)
        assert failed.first_failure == f"{self.ROLLOUT}: SessionFileChangedError"
        assert total(database) == TokenCounts()
        monkeypatch.undo()
        assert [self.codex_scan(database, tmp_path, now).files_read for _scan in range(2)] == [1, 0]
        assert total(database) == self.CODEX_B

    @pytest.mark.parametrize("moment", ["before the read", "during the read"])
    def test_a_codex_rollout_replaced_once_its_position_was_checked_is_read_as_it_was_and_then_afresh(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch, moment: str
    ) -> None:
        rollout = self.codex_rollout_read_once_and_then_grown(database, tmp_path, now)
        self.change_the_file(monkeypatch, moment, self.changed("replaced by another file", rollout, self.codex_b(now)))

        self.codex_scan(database, tmp_path, now)

        assert total(database) == self.CODEX_A
        monkeypatch.undo()
        assert [self.codex_scan(database, tmp_path, now).files_read for _scan in range(2)] == [1, 0]
        assert total(database) == self.CODEX_B

    @pytest.mark.parametrize("how", ["rewritten in place", "replaced by another file"])
    def test_a_codex_rollout_changed_after_it_was_read_is_saved_as_it_was_read_and_then_read_afresh(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch, how: str
    ) -> None:
        rollout = self.codex_rollout_read_once_and_then_grown(database, tmp_path, now)
        read = rollout.read_bytes()
        self.change_the_file(monkeypatch, "after the read", self.changed(how, rollout, self.codex_b(now)))

        self.codex_scan(database, tmp_path, now)

        with database.read() as conn:
            (saved,) = db.load_cursors(conn, Provider.CODEX).values()
        assert isinstance(saved.state, dict)
        assert (saved.byte_offset, saved.state["anchor"]) == (len(read), anchor_of(read))
        assert total(database) == self.CODEX_A
        monkeypatch.undo()
        assert [self.codex_scan(database, tmp_path, now).files_read for _scan in range(2)] == [1, 0]
        assert total(database) == self.CODEX_B

    def claude_scan(self, database: Database, projects: Path, now: datetime) -> ScanReport:
        return ClaudeTranscriptScanner(projects).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

    def claude_a(self, now: datetime) -> list[dict[str, Any]]:
        return [claude_line("msg-a", now, output=5), claude_line("msg-b", now, output=6)]

    def claude_transcript_read_once_and_then_grown(self, database: Database, projects: Path, now: datetime) -> Path:
        transcript = projects / "proj" / "session.jsonl"
        first, second = self.claude_a(now)
        append(transcript, first)
        self.claude_scan(database, projects, now)
        append(transcript, second)
        return transcript

    def claude_b(self, now: datetime) -> bytes:
        return self.as_bytes(
            claude_line("msg-c", now, output=7),
            claude_line("msg-d", now, output=8),
            claude_line("msg-e", now, output=9),
        )

    def claude_counted(self, database: Database) -> tuple[int, int]:
        return total(database).output_tokens, total(database).requests

    @pytest.mark.parametrize("moment", ["before the read", "during the read"])
    def test_a_claude_transcript_rewritten_in_place_once_its_position_was_checked_fails_and_is_then_read_afresh(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch, moment: str
    ) -> None:
        projects = tmp_path / "projects"
        transcript = self.claude_transcript_read_once_and_then_grown(database, projects, now)
        with database.read() as conn:
            saved_before = db.load_cursors(conn, Provider.CLAUDE)
        self.change_the_file(monkeypatch, moment, self.changed("rewritten in place", transcript, self.claude_b(now)))

        failed = self.claude_scan(database, projects, now)

        with database.read() as conn:
            assert db.load_cursors(conn, Provider.CLAUDE) == saved_before
        assert (failed.files_read, failed.files_failed) == (0, 1)
        assert failed.first_failure == "session.jsonl: SessionFileChangedError"
        assert self.claude_counted(database) == (5, 1)
        monkeypatch.undo()
        assert [self.claude_scan(database, projects, now).files_read for _scan in range(2)] == [1, 0]
        assert self.claude_counted(database) == (5 + 7 + 8 + 9, 4)

    def test_a_claude_transcript_rewritten_in_place_while_it_is_first_read_fails_and_is_then_read_afresh(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        projects = tmp_path / "projects"
        transcript = projects / "proj" / "session.jsonl"
        append(transcript, *self.claude_a(now))
        self.change_the_file(
            monkeypatch, "during the read", self.changed("rewritten in place", transcript, self.claude_b(now))
        )

        failed = self.claude_scan(database, projects, now)

        with database.read() as conn:
            assert db.load_cursors(conn, Provider.CLAUDE) == {}
        assert (failed.files_read, failed.files_failed) == (0, 1)
        assert failed.first_failure == "session.jsonl: SessionFileChangedError"
        assert self.claude_counted(database) == (0, 0)
        monkeypatch.undo()
        assert [self.claude_scan(database, projects, now).files_read for _scan in range(2)] == [1, 0]
        assert self.claude_counted(database) == (7 + 8 + 9, 3)

    @pytest.mark.parametrize("moment", ["before the read", "during the read"])
    def test_a_claude_transcript_replaced_once_its_position_was_checked_is_read_as_it_was_and_then_afresh(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch, moment: str
    ) -> None:
        projects = tmp_path / "projects"
        transcript = self.claude_transcript_read_once_and_then_grown(database, projects, now)
        self.change_the_file(
            monkeypatch, moment, self.changed("replaced by another file", transcript, self.claude_b(now))
        )

        self.claude_scan(database, projects, now)

        assert self.claude_counted(database) == (5 + 6, 2)
        monkeypatch.undo()
        assert [self.claude_scan(database, projects, now).files_read for _scan in range(2)] == [1, 0]
        assert self.claude_counted(database) == (5 + 6 + 7 + 8 + 9, 5)

    @pytest.mark.parametrize("how", ["rewritten in place", "replaced by another file"])
    def test_a_claude_transcript_changed_after_it_was_read_is_saved_as_it_was_read_and_then_read_afresh(
        self, database: Database, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch, how: str
    ) -> None:
        projects = tmp_path / "projects"
        transcript = self.claude_transcript_read_once_and_then_grown(database, projects, now)
        read = transcript.read_bytes()
        opened_as = transcript.stat()
        self.change_the_file(monkeypatch, "after the read", self.changed(how, transcript, self.claude_b(now)))

        self.claude_scan(database, projects, now)

        with database.read() as conn:
            (saved,) = db.load_cursors(conn, Provider.CLAUDE).values()
        assert (saved.byte_offset, saved.state) == (
            len(read),
            {"anchor": anchor_of(read), **scanner_base.file_identity(opened_as)},
        )
        assert self.claude_counted(database) == (5 + 6, 2)
        monkeypatch.undo()
        assert [self.claude_scan(database, projects, now).files_read for _scan in range(2)] == [1, 0]
        assert self.claude_counted(database) == (5 + 6 + 7 + 8 + 9, 5)

    def test_every_position_saved_in_the_first_contents_would_pass_for_a_place_to_read_on_from_in_the_later_ones(
        self, now: datetime
    ) -> None:
        codex_a, claude_a = self.codex_a(now), self.claude_a(now)
        saved_in_a_and_b = [
            (self.as_bytes(*codex_a[:2]), self.codex_b(now)),
            (self.as_bytes(*codex_a), self.codex_b(now)),
            (self.as_bytes(*claude_a[:1]), self.claude_b(now)),
            (self.as_bytes(*claude_a), self.claude_b(now)),
        ]

        for read, later in saved_in_a_and_b:
            assert len(later) > len(read)
            assert later[len(read) - 1 : len(read)] == b"\n"
            assert later[: len(read)] != read


class TestModelNames:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("gpt-x", "gpt-x"), ("  gpt-x\n", "gpt-x"), ("", None), ("   ", None), (None, None), (7, None), (["m"], None)],
    )
    def test_a_model_name_is_text_that_is_not_blank(self, value: object, expected: str | None) -> None:
        assert named_model(value) == expected

    @pytest.mark.parametrize("model", ["", "   ", None, 7])
    def test_a_claude_response_that_names_no_model_is_stored_as_unknown(
        self, database: Database, tmp_path: Path, now: datetime, model: object
    ) -> None:
        record = claude_line("first", now)
        record["message"]["model"] = model
        append(tmp_path / "projects" / "proj" / "session.jsonl", record)

        ClaudeTranscriptScanner(tmp_path / "projects").scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert [model for _, _, model in usage_rows(database)] == ["unknown"]

    def test_a_blank_model_in_a_codex_turn_keeps_the_model_already_in_effect(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(
            tmp_path / "sessions" / "rollout-models.jsonl",
            codex_meta("workspace-a"),
            codex_turn("  gpt-x  "),
            codex_turn("   "),
            codex_count(now, codex_totals(5, 0, 1), codex_totals(5, 0, 1)),
        )

        CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert [model for _, _, model in usage_rows(database)] == ["gpt-x"]

    def test_grok_usage_under_blank_model_names_is_stored_together_as_unknown(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        usage_file = tmp_path / "sessions" / "project" / "session-1" / "usage.json"
        usage_file.parent.mkdir(parents=True)
        turn = grok_turn(1, now, 10, 0, 1)
        counts = turn["modelUsage"]["grok-x"]
        turn["modelUsage"] = {"": counts, "   ": counts, " grok-x ": counts}
        usage_file.write_text(json.dumps({"turns": [turn]}))

        GrokSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        rows = usage_rows(database)
        assert rows[(ACCOUNT_A, "2026-10-03T14", "unknown")].requests == 6
        assert rows[(ACCOUNT_A, "2026-10-03T14", "grok-x")].requests == 3

    def test_a_grok_turn_with_no_breakdown_and_no_primary_model_is_stored_as_unknown(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        usage_file = tmp_path / "sessions" / "project" / "session-1" / "usage.json"
        usage_file.parent.mkdir(parents=True)
        turn = grok_turn(1, now, 10, 0, 1)
        del turn["modelUsage"]
        turn["primaryModelId"] = None
        usage_file.write_text(json.dumps({"turns": [turn]}))

        GrokSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert [model for _, _, model in usage_rows(database)] == ["unknown"]


class TestNeedsScan:
    def cursor_for(self, path: Path) -> FileCursor:
        stat = path.stat()
        return FileCursor(
            byte_offset=stat.st_size,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            state={"device": stat.st_dev, "inode": stat.st_ino, "changed_ns": stat.st_ctime_ns},
        )

    def test_the_same_file_with_the_saved_size_and_mtime_is_not_read(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        path.write_bytes(b"one\n")

        assert scanner_base.needs_scan(self.cursor_for(path), path.stat(), 0) is False

    def test_a_file_put_in_its_place_with_the_saved_size_and_mtime_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        path.write_bytes(b"one\n")
        cursor = self.cursor_for(path)

        put_in_place_keeping_size_and_mtime(path, b"two\n")

        assert scanner_base.needs_scan(cursor, path.stat(), 0) is True

    def test_a_file_rewritten_in_place_with_its_mtime_restored_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        path.write_bytes(b"one\n")
        cursor = self.cursor_for(path)

        rewrite_in_place_keeping_mtime(path, b"two\n")

        assert scanner_base.needs_scan(cursor, path.stat(), 0) is True

    def test_a_cursor_that_does_not_name_the_file_is_read_even_when_size_and_mtime_match(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        path.write_bytes(b"one\n")
        stat = path.stat()
        cursor = FileCursor(byte_offset=stat.st_size, size=stat.st_size, mtime_ns=stat.st_mtime_ns, state={})

        assert scanner_base.needs_scan(cursor, stat, 0) is True

    @pytest.mark.parametrize("missing", ["device", "inode", "changed_ns"])
    def test_a_cursor_missing_one_identity_field_is_read(self, tmp_path: Path, missing: str) -> None:
        path = tmp_path / "session.jsonl"
        path.write_bytes(b"one\n")
        stat = path.stat()
        state = {"device": stat.st_dev, "inode": stat.st_ino, "changed_ns": stat.st_ctime_ns}
        del state[missing]
        cursor = FileCursor(byte_offset=stat.st_size, size=stat.st_size, mtime_ns=stat.st_mtime_ns, state=state)

        assert scanner_base.needs_scan(cursor, stat, 0) is True

    def test_a_boolean_is_not_an_inode(self) -> None:
        state = {"device": True, "inode": 2, "changed_ns": 3}
        assert scanner_base.same_identity(state, device=1, inode=2, changed_ns=3) is False

    def test_a_position_outside_the_file_is_read_even_when_the_file_is_otherwise_unchanged(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "session.jsonl"
        path.write_bytes(b"one\n")
        cursor = self.cursor_for(path)
        outside = FileCursor(cursor.byte_offset + 1, cursor.size, cursor.mtime_ns, cursor.state)

        assert scanner_base.needs_scan(outside, path.stat(), 0) is True


class TestTheBackfillFloor:
    def test_a_file_modified_at_the_floor_to_the_nanosecond_is_still_read(self, tmp_path: Path) -> None:
        now = datetime(2026, 10, 3, 14, 0, 0, 997, tzinfo=UTC)
        floor_moment = now - timedelta(days=BACKFILL_DAYS)
        true_floor_ns = (floor_moment - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1) * 1000
        session = tmp_path / "session.jsonl"
        session.write_text("")
        os.utime(session, ns=(true_floor_ns, true_floor_ns))

        floor_ns = scanner_base.backfill_floor_ns(now, BACKFILL_DAYS)

        assert floor_ns == true_floor_ns
        assert scanner_base.needs_scan(None, session.stat(), floor_ns)

    @pytest.mark.parametrize(
        "now",
        [
            pytest.param(datetime(2026, 3, 10, 12, tzinfo=ZoneInfo("America/New_York")), id="clocks went forward"),
            pytest.param(datetime(2026, 11, 3, 12, tzinfo=ZoneInfo("America/New_York")), id="clocks went back"),
        ],
    )
    def test_the_floor_is_whole_days_of_elapsed_time_whatever_zone_now_is_in(self, now: datetime) -> None:
        week_ago = now.astimezone(UTC) - timedelta(days=7)
        expected_ns = (week_ago - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1) * 1000

        assert scanner_base.backfill_floor_ns(now, 7) == expected_ns

    def test_zero_backfill_still_enrolls_a_file_written_between_polls(self, database: Database, tmp_path: Path) -> None:
        first = datetime(2026, 10, 3, 14, 0, tzinfo=UTC)
        later = first + timedelta(seconds=10)
        claude_file = tmp_path / "claude" / "session.jsonl"
        codex_file = tmp_path / "codex" / "sessions" / "rollout-2026-10-03T10-00-00-aaaa.jsonl"
        grok_file = tmp_path / "grok" / "sessions" / "project" / "session" / "usage.json"
        for path, contents in (
            (claude_file, b"\n"),
            (codex_file, b"\n"),
            (grok_file, b'{"turns":[]}'),
        ):
            path.parent.mkdir(parents=True)
            path.write_bytes(contents)
        scanners = (
            ClaudeTranscriptScanner(tmp_path / "claude"),
            CodexSessionScanner(tmp_path / "codex"),
            GrokSessionScanner(tmp_path / "grok"),
        )

        def whoever(_moment: datetime) -> int:
            return 0

        def cursors(provider: Provider) -> dict[str, FileCursor]:
            with database.connect() as conn:
                return db.load_cursors(conn, provider)

        before = scanner_base.backfill_floor_ns(first, 0) - 1_000_000_000
        for path in (claude_file, codex_file, grok_file):
            os.utime(path, ns=(before, before))
        for scanner in scanners:
            scanner.scan(database, whoever, first, 0)
        assert all(cursors(scanner.provider) == {} for scanner in scanners)

        between = scanner_base.backfill_floor_ns(later, 0) - 1_000_000_000
        for path in (claude_file, codex_file, grok_file):
            os.utime(path, ns=(between, between))
        for scanner in scanners:
            scanner.scan(database, whoever, later, 0)

        assert all(cursors(scanner.provider) for scanner in scanners)


class TestUsageTheStoreCannotHold:
    ALREADY_COUNTED = b"already-counted"

    def fill_the_hour(self, database: Database, now: datetime) -> None:
        with database.connect() as conn:
            db.add_message_usage(
                conn,
                [
                    MessageUsage(
                        self.ALREADY_COUNTED,
                        now,
                        ACCOUNT_A,
                        "claude-model-x",
                        TokenCounts(output_tokens=MAX_TOKEN_COUNT),
                    )
                ],
            )

    def test_a_file_whose_usage_would_take_a_stored_row_past_its_limit_fails_alone_and_changes_nothing(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        project = tmp_path / "projects" / "proj"
        full = project / "a-full.jsonl"
        append(full, claude_line("first", now, output=7))
        append(project / "b-fine.jsonl", claude_line("second", now + timedelta(hours=1), output=9))
        self.fill_the_hour(database, now)

        report = ClaudeTranscriptScanner(tmp_path / "projects").scan(
            database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS
        )

        assert (report.files_read, report.files_failed, report.first_failure) == (1, 1, "a-full.jsonl: ValueError")
        assert total(database).output_tokens == MAX_TOKEN_COUNT + 9
        assert len(self.seen_keys(database)) == 2
        with database.read() as conn:
            assert str(full) not in db.load_cursors(conn, Provider.CLAUDE)

    def seen_keys(self, database: Database) -> list[bytes]:
        with database.read() as conn:
            return [bytes(row["key"]) for row in conn.execute("SELECT key FROM claude_seen_messages")]

    def test_the_refused_file_s_messages_are_not_remembered_so_a_later_scan_can_still_count_them(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        full = tmp_path / "projects" / "proj" / "a-full.jsonl"
        append(full, claude_line("first", now, output=7))
        self.fill_the_hour(database, now)
        scanner = ClaudeTranscriptScanner(tmp_path / "projects")

        assert scanner.scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS).files_failed == 1
        assert self.seen_keys(database) == [self.ALREADY_COUNTED]

        with database.connect() as conn:
            conn.execute("DELETE FROM token_usage")
            conn.execute("DELETE FROM claude_seen_messages")
        retried = scanner.scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert (retried.files_read, retried.files_failed) == (1, 0)
        assert total(database).output_tokens == 7


class TestGrokUsageFieldsOfTheWrongType:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("inputTokens", True),
            ("inputTokens", "10"),
            ("inputTokens", 10.0),
            ("outputTokens", False),
            ("modelCalls", "3"),
            ("endedAt", 1_791_036_000),
            ("endedAt", "1791036000"),
            ("primaryModelId", 7),
        ],
    )
    def test_the_file_is_reported_and_nothing_from_it_is_stored(
        self, database: Database, tmp_path: Path, now: datetime, field: str, value: object
    ) -> None:
        usage_file = tmp_path / "sessions" / "project" / "session-1" / "usage.json"
        usage_file.parent.mkdir(parents=True)
        turn = grok_turn(1, now, 10, 0, 1)
        turn[field] = value
        usage_file.write_text(json.dumps({"turns": [turn]}))

        report = GrokSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert (report.files_read, report.files_failed) == (0, 1)
        assert report.first_failure == "session-1: ValidationError"
        assert total(database) == TokenCounts()


class TestAClaudeResponseRevisedWhileItStreams:
    def transcript(self, tmp_path: Path) -> Path:
        return tmp_path / "projects" / "proj" / "session.jsonl"

    def scan(self, database: Database, tmp_path: Path, now: datetime, resolver: AccountResolver | None = None) -> None:
        ClaudeTranscriptScanner(tmp_path / "projects").scan(
            database, resolver or everything_to(ACCOUNT_A), now, BACKFILL_DAYS
        )

    def test_reasoning_tokens_reported_on_a_later_line_in_a_later_scan_are_added(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(self.transcript(tmp_path), claude_line("msg-1", now, output=30, thinking=0))
        self.scan(database, tmp_path, now)
        append(self.transcript(tmp_path), claude_line("msg-1", now, output=100, thinking=40))

        self.scan(database, tmp_path, now)

        assert total(database) == TokenCounts(10, 1000, 50, 100, 40, 1)

    def test_the_same_two_lines_read_in_one_scan_give_the_same_totals(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(
            self.transcript(tmp_path),
            claude_line("msg-1", now, output=30, thinking=0),
            claude_line("msg-1", now, output=100, thinking=40),
        )

        self.scan(database, tmp_path, now)

        assert total(database) == TokenCounts(10, 1000, 50, 100, 40, 1)

    def test_a_revision_of_reasoning_tokens_alone_is_added_though_the_output_count_did_not_change(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(self.transcript(tmp_path), claude_line("msg-1", now, output=100, thinking=0))
        self.scan(database, tmp_path, now)
        append(self.transcript(tmp_path), claude_line("msg-1", now, output=100, thinking=40))

        self.scan(database, tmp_path, now)

        assert total(database) == TokenCounts(10, 1000, 50, 100, 40, 1)

    def test_a_revision_of_any_other_count_adds_exactly_the_difference(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(self.transcript(tmp_path), claude_line("msg-1", now, input_tokens=4, cache_read=100, cache_write=5))
        self.scan(database, tmp_path, now)
        append(self.transcript(tmp_path), claude_line("msg-1", now, input_tokens=10, cache_read=1000, cache_write=50))

        self.scan(database, tmp_path, now)

        assert total(database) == TokenCounts(10, 1000, 50, 100, 0, 1)

    def test_a_later_line_with_smaller_counts_takes_nothing_away_and_adds_nothing(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(self.transcript(tmp_path), claude_line("msg-1", now, output=100, thinking=40))
        self.scan(database, tmp_path, now)
        append(self.transcript(tmp_path), claude_line("msg-1", now, output=30, thinking=0, cache_read=10))

        self.scan(database, tmp_path, now)

        assert total(database) == TokenCounts(10, 1000, 50, 100, 40, 1)

    def test_counts_that_peak_on_different_lines_each_take_their_largest_value(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(
            self.transcript(tmp_path),
            claude_line("msg-1", now, output=100, thinking=0, cache_read=2000),
            claude_line("msg-1", now, output=60, thinking=40, cache_read=1000),
        )

        self.scan(database, tmp_path, now)

        assert total(database) == TokenCounts(10, 2000, 50, 100, 40, 1)

    def test_a_response_is_placed_in_the_hour_of_its_first_line(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(
            self.transcript(tmp_path),
            claude_line("msg-1", now + timedelta(minutes=59), output=30),
            claude_line("msg-1", now + timedelta(minutes=61), output=100),
        )

        self.scan(database, tmp_path, now)

        assert [bucket for _, bucket, _ in usage_rows(database)] == ["2026-10-03T14"]

    def test_a_later_line_read_in_a_later_scan_is_added_to_the_hour_of_the_first_line(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(self.transcript(tmp_path), claude_line("msg-1", now + timedelta(minutes=59), output=30))
        self.scan(database, tmp_path, now)
        append(self.transcript(tmp_path), claude_line("msg-1", now + timedelta(minutes=61), output=100))

        self.scan(database, tmp_path, now)

        assert usage_rows(database) == {
            (ACCOUNT_A, "2026-10-03T14", "claude-model-x"): TokenCounts(10, 1000, 50, 100, 0, 1)
        }

    def test_a_later_line_read_after_the_login_changed_is_added_to_the_account_first_counted(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        switch = now + timedelta(minutes=30)
        resolver = build_account_resolver([(now, ACCOUNT_A), (switch, ACCOUNT_B)])
        append(self.transcript(tmp_path), claude_line("msg-1", now + timedelta(minutes=29), output=30))
        self.scan(database, tmp_path, now, resolver)
        append(self.transcript(tmp_path), claude_line("msg-1", now + timedelta(minutes=31), output=100))

        self.scan(database, tmp_path, now, resolver)

        assert usage_rows(database) == {
            (ACCOUNT_A, "2026-10-03T14", "claude-model-x"): TokenCounts(10, 1000, 50, 100, 0, 1)
        }

    def test_a_later_line_naming_another_model_is_added_under_the_model_first_named(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(self.transcript(tmp_path), claude_line("msg-1", now, output=30, model="claude-model-x"))
        self.scan(database, tmp_path, now)
        append(self.transcript(tmp_path), claude_line("msg-1", now, output=100, model="claude-model-y"))

        self.scan(database, tmp_path, now)

        assert usage_rows(database) == {
            (ACCOUNT_A, "2026-10-03T14", "claude-model-x"): TokenCounts(10, 1000, 50, 100, 0, 1)
        }

    def test_a_later_line_copied_into_a_resumed_session_file_is_added_where_the_response_was_first_counted(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        projects = tmp_path / "projects" / "proj"
        append(projects / "original.jsonl", claude_line("msg-1", now + timedelta(minutes=59), output=30))
        self.scan(database, tmp_path, now)
        append(projects / "resumed.jsonl", claude_line("msg-1", now + timedelta(minutes=61), output=100))

        self.scan(database, tmp_path, now)

        assert usage_rows(database) == {
            (ACCOUNT_A, "2026-10-03T14", "claude-model-x"): TokenCounts(10, 1000, 50, 100, 0, 1)
        }

    def test_a_later_line_carrying_reasoning_tokens_and_no_other_usage_is_added(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        append(self.transcript(tmp_path), claude_line("msg-1", now, output=100, thinking=0))
        self.scan(database, tmp_path, now)
        reasoning_only = claude_line("msg-1", now, output=0, input_tokens=0, cache_read=0, cache_write=0, thinking=40)
        append(self.transcript(tmp_path), reasoning_only)

        self.scan(database, tmp_path, now)

        assert total(database) == TokenCounts(10, 1000, 50, 100, 40, 1)


class TestAClaudeResponseIsPlacedByItsEarliestLineWhateverIsReadFirst:
    @pytest.mark.parametrize(
        ("original_name", "resumed_name"),
        [("b-original.jsonl", "a-resumed.jsonl"), ("a-original.jsonl", "b-resumed.jsonl")],
    )
    @pytest.mark.parametrize("scans", ["both in one scan", "the resumed one first", "the original first"])
    def test_file_names_and_scan_boundaries_do_not_change_where_a_response_is_counted(
        self, database: Database, tmp_path: Path, now: datetime, original_name: str, resumed_name: str, scans: str
    ) -> None:
        project = tmp_path / "projects" / "proj"
        resolver = build_account_resolver([(now, ACCOUNT_A), (now + timedelta(hours=1), ACCOUNT_B)])
        scanner = ClaudeTranscriptScanner(tmp_path / "projects")
        first_line = claude_line("msg-1", now + timedelta(minutes=59), output=30)
        later_line = claude_line("msg-1", now + timedelta(minutes=61), output=100, thinking=40, model="claude-model-y")
        writes = {
            "both in one scan": [[(original_name, first_line), (resumed_name, later_line)]],
            "the resumed one first": [[(resumed_name, later_line)], [(original_name, first_line)]],
            "the original first": [[(original_name, first_line)], [(resumed_name, later_line)]],
        }

        for batch in writes[scans]:
            for name, line in batch:
                append(project / name, line)
            scanner.scan(database, resolver, now, BACKFILL_DAYS)

        assert usage_rows(database) == {
            (ACCOUNT_A, "2026-10-03T14", "claude-model-x"): TokenCounts(10, 1000, 50, 100, 40, 1)
        }

    def test_usage_already_counted_under_a_later_line_is_moved_when_the_earlier_line_is_read(
        self, database: Database, tmp_path: Path, now: datetime
    ) -> None:
        project = tmp_path / "projects" / "proj"
        scanner = ClaudeTranscriptScanner(tmp_path / "projects")
        append(project / "a-resumed.jsonl", claude_line("msg-1", now + timedelta(minutes=61), output=100))
        scanner.scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)
        assert [bucket for _, bucket, _ in usage_rows(database)] == ["2026-10-03T15"]

        append(project / "b-original.jsonl", claude_line("msg-1", now + timedelta(minutes=59), output=30))
        scanner.scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert usage_rows(database) == {
            (ACCOUNT_A, "2026-10-03T14", "claude-model-x"): TokenCounts(10, 1000, 50, 100, 0, 1)
        }


class TestSavedCodexStateThatIsNotExactlyWhatTheScannerWrites:
    ROLLOUT = "rollout-2026-10-03T10-00-00-aaaa.jsonl"
    COMPLETE_TOTALS: ClassVar[dict[str, int]] = {
        "input_tokens": 100,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 10,
        "reasoning_output_tokens": 0,
    }

    def scanned(self, database: Database, home: Path, now: datetime) -> Path:
        rollout = home / "sessions" / self.ROLLOUT
        append(
            rollout,
            codex_meta("workspace-a"),
            codex_turn("gpt-x"),
            codex_count(now, codex_totals(100, 0, 10), codex_totals(100, 0, 10)),
        )
        CodexSessionScanner(home).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)
        return rollout

    def saved_state(self, database: Database) -> dict[str, Any]:
        with database.read() as conn:
            row = conn.execute("SELECT state_json FROM scan_cursors WHERE provider = 'codex'").fetchone()
        state: dict[str, Any] = json.loads(row["state_json"])
        return state

    def save_state(self, database: Database, state: object) -> None:
        with database.connect() as conn:
            conn.execute("UPDATE scan_cursors SET state_json = ? WHERE provider = 'codex'", (json.dumps(state),))

    def test_what_the_scanner_saves_is_complete_and_is_read_back_without_a_fresh_read(
        self, database: Database, tmp_path: Path, now: datetime, caplog: pytest.LogCaptureFixture
    ) -> None:
        rollout = self.scanned(database, tmp_path, now)
        assert self.saved_state(database) == {
            "path": str(rollout),
            "anchor": anchor_of(rollout.read_bytes()),
            "account_external_id": "workspace-a",
            "model": "gpt-x",
            "previous_totals": self.COMPLETE_TOTALS,
            "counted_requests": 1,
            **scanner_base.file_identity(rollout.stat()),
        }
        append(rollout, codex_count(now, codex_totals(250, 0, 25), codex_totals(150, 0, 15)))

        with caplog.at_level(logging.WARNING, logger="usagedash.scanners.codex"):
            CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert "reading the file afresh" not in caplog.text
        assert total(database) == TokenCounts(input_tokens=250, output_tokens=25, requests=2)

    @pytest.mark.parametrize(
        "change",
        [
            {"previous_totals": {"input_tokens": 100}},
            {"previous_totals": {**COMPLETE_TOTALS, "output_tokens": None}},
            {"previous_totals": {**COMPLETE_TOTALS, "output_tokens": "10"}},
            {"previous_totals": {**COMPLETE_TOTALS, "output_tokens": 10.0}},
            {"previous_totals": {**COMPLETE_TOTALS, "output_tokens": True}},
            {"previous_totals": {**COMPLETE_TOTALS, "added_by_someone_else": 1}},
            {"previous_totals": []},
            {"account_external_id": 7},
            {"account_external_id": ""},
            {"account_external_id": "   "},
            {"model": None},
            {"model": 7},
            {"model": ""},
            {"path": None},
            {"path": 7},
            {"path": ""},
            {"anchor": None},
            {"anchor": 7},
            {"anchor": ""},
            {"anchor": "not thirty-two hexadecimal digits"},
            {"added_by_someone_else": 1},
        ],
    )
    def test_a_changed_state_is_refused_and_the_file_is_read_afresh_without_counting_anything_twice(
        self,
        database: Database,
        tmp_path: Path,
        now: datetime,
        caplog: pytest.LogCaptureFixture,
        change: dict[str, object],
    ) -> None:
        rollout = self.scanned(database, tmp_path, now)
        self.save_state(database, self.saved_state(database) | change)
        append(rollout, codex_count(now, codex_totals(250, 0, 25), codex_totals(150, 0, 15)))

        with caplog.at_level(logging.WARNING, logger="usagedash.scanners.codex"):
            report = CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert (report.files_read, report.files_failed) == (1, 0)
        assert "could not be read back; reading the file afresh" in caplog.text
        assert total(database) == TokenCounts(input_tokens=250, output_tokens=25, requests=2)
        assert [(account, model) for account, _, model in usage_rows(database)] == [
            (self.workspace_account(database), "gpt-x")
        ]

    def workspace_account(self, database: Database) -> int:
        with database.read() as conn:
            return db.list_accounts(conn)[0].id

    @pytest.mark.parametrize("missing", ["path", "anchor", "account_external_id", "model", "previous_totals"])
    def test_a_state_with_a_field_left_out_is_refused(
        self, database: Database, tmp_path: Path, now: datetime, caplog: pytest.LogCaptureFixture, missing: str
    ) -> None:
        rollout = self.scanned(database, tmp_path, now)
        state = self.saved_state(database)
        del state[missing]
        self.save_state(database, state)
        append(rollout, codex_count(now, codex_totals(250, 0, 25), codex_totals(150, 0, 15)))

        with caplog.at_level(logging.WARNING, logger="usagedash.scanners.codex"):
            CodexSessionScanner(tmp_path).scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert "could not be read back; reading the file afresh" in caplog.text
        assert total(database) == TokenCounts(input_tokens=250, output_tokens=25, requests=2)

    def test_one_file_s_saved_state_that_is_not_json_costs_that_file_a_fresh_read_and_no_other_file_anything(
        self, database: Database, tmp_path: Path, now: datetime, caplog: pytest.LogCaptureFixture
    ) -> None:
        damaged = self.scanned(database, tmp_path, now)
        healthy = tmp_path / "sessions" / "rollout-2026-10-03T11-00-00-bbbb.jsonl"
        append(healthy, codex_meta("workspace-a"), codex_count(now, codex_totals(7, 0, 1), codex_totals(7, 0, 1)))
        scanner = CodexSessionScanner(tmp_path)
        scanner.scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)
        with database.connect() as conn:
            conn.execute("UPDATE scan_cursors SET state_json = '{' WHERE file_key = ?", (str(tmp_path / damaged.name),))
        append(damaged, codex_count(now, codex_totals(250, 0, 25), codex_totals(150, 0, 15)))
        append(healthy, codex_count(now, codex_totals(10, 0, 2), codex_totals(3, 0, 1)))

        with caplog.at_level(logging.WARNING, logger="usagedash.scanners.codex"):
            report = scanner.scan(database, everything_to(ACCOUNT_A), now, BACKFILL_DAYS)

        assert (report.files_read, report.files_failed) == (2, 0)
        assert caplog.text.count("reading the file afresh") == 1
        assert total(database) == TokenCounts(input_tokens=260, output_tokens=27, requests=4)
