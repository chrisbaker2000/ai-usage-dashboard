import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any

from pydantic import ConfigDict, Field

from usagedash import db
from usagedash.db import Database, FileCursor
from usagedash.models import UNATTRIBUTED_ACCOUNT_ID, AccountIdentity, Provider, TokenCounts, UsageKey
from usagedash.paths import directory_identity
from usagedash.scanners.base import (
    MAX_TOKENS_PER_RECORD,
    UNKNOWN_MODEL,
    AccountResolver,
    ScanReport,
    isolated,
    named_model,
    open_session_file,
    parse_json_line,
    same_identity,
    session_files,
    stable_backfill_floor_ns,
    token_count,
)
from usagedash.timeutil import day_key, hour_bucket, in_utc, parse_iso
from usagedash.validation import Identifier, LenientModel

logger = logging.getLogger(__name__)

_RELEVANT_MARKERS = (b'"token_count"', b'"session_meta"', b'"turn_context"')
_ALWAYS_WRITTEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
_CACHE_WRITE_FIELD = "cache_write_input_tokens"
_TOTAL_FIELDS = (*_ALWAYS_WRITTEN_FIELDS, _CACHE_WRITE_FIELD)
PLACEHOLDER_LABEL_ID_LENGTH = 8

RawTotals = dict[str, int]
_SavedCount = Annotated[int, Field(ge=0, le=MAX_TOKENS_PER_RECORD)]


class _SavedTotals(LenientModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: _SavedCount
    cached_input_tokens: _SavedCount
    cache_write_input_tokens: _SavedCount
    output_tokens: _SavedCount
    reasoning_output_tokens: _SavedCount


class _SavedState(LenientModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[str, Field(min_length=1)]
    anchor: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    account_external_id: Identifier | None
    model: Identifier
    previous_totals: _SavedTotals | None


@dataclass(slots=True)
class _FileState:
    path: str
    anchor: str = ""
    account_external_id: str | None = None
    model: str = UNKNOWN_MODEL
    previous_totals: RawTotals | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "anchor": self.anchor,
            "account_external_id": self.account_external_id,
            "model": self.model,
            "previous_totals": self.previous_totals,
        }

    @classmethod
    def from_json(cls, raw: object) -> "_FileState":
        if isinstance(raw, dict):
            raw = {
                key: value
                for key, value in raw.items()
                if key not in {"device", "inode", "changed_ns", "request_counts", "counted_requests"}
            }
        saved = _SavedState.model_validate(raw)
        totals = saved.previous_totals
        return cls(
            path=saved.path,
            anchor=saved.anchor,
            account_external_id=saved.account_external_id,
            model=saved.model,
            previous_totals=None if totals is None else totals.model_dump(),
        )


def _workspace_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


@dataclass(frozen=True, slots=True)
class _Copy:
    path: Path
    size: int
    mtime_ns: int
    device: int
    inode: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _Plan:
    copy: _Copy
    read_before: _FileState | None
    replaces: bool
    must_hold: int = 0
    original_still_holds: bool = False


def _copy_at(path: Path) -> _Copy:
    stat = path.stat()
    return _Copy(path, stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino, stat.st_ctime_ns)


def _saved_state(cursor: FileCursor | None) -> _FileState | None:
    if cursor is None:
        return None
    try:
        return _FileState.from_json(cursor.state)
    except ValueError:
        logger.warning("The saved scan state of a Codex rollout could not be read back; reading the file afresh")
        return None


class CopyOutOfViewError(OSError):
    pass


_TAIL_CHUNK = 65_536
_GREW_PAST_THE_SIZE_TAKEN = -1


def _size_of_complete_lines(path: Path, size: int) -> tuple[int, bool]:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        actual = handle.tell()
        if actual < size:
            return size, False
        if actual > size:
            return _GREW_PAST_THE_SIZE_TAKEN, False
        remaining = size
        while remaining > 0:
            step = min(_TAIL_CHUNK, remaining)
            remaining -= step
            handle.seek(remaining)
            newline = handle.read(step).rfind(b"\n")
            if newline != -1:
                return remaining + newline + 1, True
        return 0, True


def _complete_size(copy: _Copy) -> tuple[int, bool]:
    try:
        return _size_of_complete_lines(copy.path, copy.size)
    except OSError:
        return copy.size, False


def _requests_in(copy: _Copy) -> int | None:
    state = _FileState(path=str(copy.path))
    total = 0
    try:
        with open_session_file(copy.path) as rollout:
            if rollout.size_when_opened != copy.size:
                return None
            for line in rollout.lines():
                if not any(marker in line for marker in _RELEVANT_MARKERS):
                    continue
                try:
                    event = _apply_line(line, state)
                except ValueError:
                    continue
                if event is not None:
                    total += event[2].requests
    except OSError:
        return None
    return total


class _RememberedCount(LenientModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[str, Field(min_length=1)]
    size: Annotated[int, Field(ge=0)]
    mtime_ns: Annotated[int, Field(ge=0)]
    device: Annotated[int, Field(ge=0)]
    inode: Annotated[int, Field(ge=0)]
    changed_ns: Annotated[int, Field(ge=0)]
    requests: Annotated[int, Field(ge=0)]


def _remembered_by_path(state: object) -> dict[str, _RememberedCount]:
    if not isinstance(state, dict):
        return {}
    raw = state.get("request_counts")
    if not isinstance(raw, list) or not raw:
        return {}
    found: dict[str, _RememberedCount] = {}
    for item in raw:
        try:
            remembered = _RememberedCount.model_validate(item)
        except ValueError:
            return {}
        if remembered.path in found:
            return {}
        found[remembered.path] = remembered
    return found


def _requests_already_counted(remembered: dict[str, _RememberedCount], copy: _Copy) -> int | None:
    saved = remembered.get(str(copy.path))
    if saved is None:
        return None
    same_file = (saved.size, saved.mtime_ns, saved.device, saved.inode, saved.changed_ns) == (
        copy.size,
        copy.mtime_ns,
        copy.device,
        copy.inode,
        copy.changed_ns,
    )
    return saved.requests if same_file else None


def _every_copy_matches_a_saved_count(state: object, copies: Sequence[_Copy]) -> bool:
    remembered = _remembered_by_path(state)
    if len(remembered) != len(copies):
        return False
    return all(_requests_already_counted(remembered, copy) is not None for copy in copies)


def _count_for_ranking(copies: Sequence[_Copy], remembered: dict[str, _RememberedCount]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for copy in copies:
        count = _requests_already_counted(remembered, copy)
        if count is None:
            count = _requests_in(copy)
        if count is not None:
            counts[id(copy)] = count
    return counts


def _count_records(copies: Sequence[_Copy], counts: dict[int, int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for copy in sorted(copies, key=lambda item: str(item.path)):
        count = counts.get(id(copy))
        if count is None:
            continue
        records.append(
            {
                "path": str(copy.path),
                "size": copy.size,
                "mtime_ns": copy.mtime_ns,
                "device": copy.device,
                "inode": copy.inode,
                "changed_ns": copy.changed_ns,
                "requests": count,
            }
        )
    return records


def _with_the_most_requests(
    copies: Sequence[_Copy], complete_size: dict[int, int], counts: dict[int, int]
) -> _Copy | None:
    if not counts:
        return None
    most = max(counts.values())
    leaders = sorted(
        (copy for copy in copies if counts.get(id(copy)) == most),
        key=lambda copy: (-complete_size[id(copy)], str(copy.path)),
    )
    return leaders[0]


def _largest(
    copies: Sequence[_Copy], read_before: str | None, state: object
) -> tuple[_Copy, list[dict[str, Any]] | None, dict[int, int]]:
    measured: dict[int, tuple[int, bool]] = {}
    rest = list(copies)
    best = -1
    while rest:
        raw_best = max(copy.size for copy in rest)
        if raw_best < best:
            break
        wave = [copy for copy in rest if copy.size == raw_best]
        for copy in wave:
            measured[id(copy)] = _complete_size(copy)
            best = max(best, measured[id(copy)][0])
            rest.remove(copy)
    opened = [copy for copy in copies if id(copy) in measured]
    request_counts: list[dict[str, Any]] | None = None
    counts: dict[int, int] = {}
    if len(copies) >= 2:
        for copy in copies:
            if id(copy) not in measured:
                measured[id(copy)] = _complete_size(copy)
        ranked = [copy for copy in copies if measured[id(copy)][1]]
        unreadable_size = max(
            (measured[id(copy)][0] for copy in copies if not measured[id(copy)][1]),
            default=-1,
        )
        if len(ranked) == 1 and unreadable_size <= measured[id(ranked[0])][0]:
            return ranked[0], None, {}
        if len(ranked) >= 2 and unreadable_size <= max(measured[id(copy)][0] for copy in ranked):
            counts = _count_for_ranking(ranked, _remembered_by_path(state))
            request_counts = _count_records(ranked, counts)
            by_requests = _with_the_most_requests(ranked, {id(copy): measured[id(copy)][0] for copy in ranked}, counts)
            if by_requests is not None:
                return by_requests, request_counts, counts
    leaders = sorted(
        (copy for copy in opened if measured[id(copy)][0] == best),
        key=lambda copy: str(copy.path),
    )
    chosen = next((copy for copy in leaders if str(copy.path) == read_before), leaders[0])
    return chosen, request_counts, counts


def _unchanged_raw_leader(cursor: FileCursor, saved: _FileState, copies: Sequence[_Copy]) -> bool:
    previous = next((copy for copy in copies if str(copy.path) == saved.path), None)
    if previous is None or not 0 <= cursor.byte_offset <= cursor.size:
        return False
    if (cursor.size, cursor.mtime_ns) != (previous.size, previous.mtime_ns):
        return False
    if not same_identity(cursor.state, device=previous.device, inode=previous.inode, changed_ns=previous.changed_ns):
        return False
    if all(str(copy.path) == saved.path for copy in copies):
        return True
    return _every_copy_matches_a_saved_count(cursor.state, copies)


def _copy_already_read(copies: Sequence[_Copy], cursor: FileCursor | None, saved: _FileState | None) -> _Copy | None:
    recorded = _recorded_device_and_inode(None if cursor is None else cursor.state)
    if recorded is not None:
        return next((copy for copy in copies if (copy.device, copy.inode) == recorded), None)
    if saved is not None:
        return next((copy for copy in copies if str(copy.path) == saved.path), None)
    return None


def _recorded_device_and_inode(state: object) -> tuple[int, int] | None:
    if not isinstance(state, dict):
        return None
    device = state.get("device")
    inode = state.get("inode")
    if type(device) is not int or type(inode) is not int:
        return None
    return (device, inode)


def _copies_to_rank(
    copies: Sequence[_Copy], cursor: FileCursor | None, saved: _FileState | None, floor_ns: int
) -> list[_Copy]:
    if cursor is not None and saved is None:
        return list(copies)
    saved_path = None if saved is None else saved.path
    recorded = _recorded_device_and_inode(None if cursor is None else cursor.state)
    return [
        copy
        for copy in copies
        if copy.mtime_ns >= floor_ns
        or (recorded is None and str(copy.path) == saved_path)
        or (recorded is not None and (copy.device, copy.inode) == recorded)
    ]


def _copy_that_keeps_counted_requests(
    chosen: _Copy,
    copies: Sequence[_Copy],
    cursor: FileCursor | None,
    saved: _FileState | None,
    database: Database,
    file_key: str,
    known_counts: dict[int, int],
) -> _Copy:
    if cursor is None or saved is None or saved.path == str(chosen.path):
        return chosen
    previous = _copy_already_read(copies, cursor, saved)
    if previous is None or previous.size < cursor.byte_offset:
        return chosen
    if (chosen.device, chosen.inode) == (previous.device, previous.inode):
        return chosen
    counted = known_counts.get(id(chosen))
    if counted is None:
        counted = _requests_in(chosen)
    if counted is None:
        return chosen
    if counted < _requests_kept(database, file_key):
        return previous
    return chosen


@dataclass(frozen=True, slots=True)
class _Decision:
    plan: _Plan | None = None
    request_counts: list[dict[str, Any]] | None = None


def _plan(
    cursor: FileCursor | None,
    copies: Sequence[_Copy],
    floor_ns: int,
    home_was_listed_in_full: bool,
    database: Database,
    file_key: str,
) -> _Decision:
    saved = _saved_state(cursor)
    recorded_file = _recorded_device_and_inode(None if cursor is None else cursor.state)
    recorded_file_is_in_view = recorded_file is not None and any(
        (copy.device, copy.inode) == recorded_file for copy in copies
    )
    original = _copy_already_read(copies, cursor, saved)
    if cursor is not None and not recorded_file_is_in_view and not home_was_listed_in_full:
        raise CopyOutOfViewError("the copy that was read before may be in a directory that could not be listed")
    if cursor is not None and saved is not None and _unchanged_raw_leader(cursor, saved, copies):
        return _Decision()
    candidates = _copies_to_rank(copies, cursor, saved, floor_ns)
    if not candidates:
        return _Decision()
    chosen, request_counts, known_counts = _largest(
        candidates, None if saved is None else saved.path, None if cursor is None else cursor.state
    )
    chosen = _copy_that_keeps_counted_requests(chosen, copies, cursor, saved, database, file_key, known_counts)
    afresh = _Decision(_Plan(chosen, None, replaces=True), request_counts)
    if cursor is None:
        return _Decision(_Plan(chosen, None, replaces=False), request_counts)
    if saved is None or not 0 <= cursor.byte_offset <= cursor.size:
        return afresh
    if saved.path == str(chosen.path) and (original is None or original.path == chosen.path):
        unchanged = (cursor.size, cursor.mtime_ns) == (chosen.size, chosen.mtime_ns)
        if unchanged and same_identity(
            cursor.state, device=chosen.device, inode=chosen.inode, changed_ns=chosen.changed_ns
        ):
            return _Decision(request_counts=request_counts)
        read_copy = chosen if recorded_file == (chosen.device, chosen.inode) else None
        return _Decision(
            _Plan(chosen, saved, replaces=True, must_hold=_replacement_floor(cursor, read_copy)),
            request_counts,
        )
    if original is not None:
        carried = None
        if saved is not None and recorded_file == (chosen.device, chosen.inode):
            carried = replace(saved, path=str(chosen.path))
        return _Decision(
            _Plan(
                chosen,
                carried,
                replaces=True,
                must_hold=_replacement_floor(cursor, original),
                original_still_holds=original.size >= cursor.byte_offset,
            ),
            request_counts,
        )
    return _Decision(
        _Plan(chosen, None, replaces=True, must_hold=_replacement_floor(cursor, None)),
        request_counts,
    )


def _replacement_holds_more_requests(
    events: Sequence[tuple[datetime, str, TokenCounts]], database: Database, file_key: str
) -> bool:
    return _request_total(events) > _requests_kept(database, file_key)


def _replacement_drops_requests(
    events: Sequence[tuple[datetime, str, TokenCounts]], database: Database, file_key: str, now: datetime
) -> bool:
    if _request_total(events) < _requests_kept(database, file_key):
        return True
    first_day = day_key(in_utc(now) - db.USAGE_RETENTION)
    kept_now = sum(count.requests for occurred_at, _model, count in events if hour_bucket(occurred_at) >= first_day)
    with database.connect() as conn:
        stored_now = db.stored_requests_kept(conn, Provider.CODEX, file_key, first_day)
    return kept_now < stored_now


def _request_total(events: Sequence[tuple[datetime, str, TokenCounts]]) -> int:
    return sum(count.requests for _occurred_at, _model, count in events)


def _saved_counted_requests(cursor: FileCursor | None) -> int | None:
    if cursor is None or not isinstance(cursor.state, dict):
        return None
    raw = cursor.state.get("counted_requests")
    if type(raw) is not int or raw < 0:
        return None
    return raw


def _requests_kept(database: Database, file_key: str) -> int:
    with database.connect() as conn:
        kept = db.stored_requests(conn, Provider.CODEX, file_key)
        cursor = db.load_cursors(conn, Provider.CODEX).get(file_key)
    saved = _saved_counted_requests(cursor)
    return kept if saved is None else max(kept, saved)


def _counted_after(cursor: FileCursor | None, replaces: bool, added: int, database: Database, file_key: str) -> int:
    if cursor is None or replaces:
        return added
    previous = _saved_counted_requests(cursor)
    if previous is None:
        with database.connect() as conn:
            previous = db.stored_requests(conn, Provider.CODEX, file_key)
    return previous + added


def _replacement_erases_retained_requests(
    events: Sequence[tuple[datetime, str, TokenCounts]], database: Database, file_key: str, now: datetime
) -> bool:
    first_day = day_key(in_utc(now) - db.USAGE_RETENTION)
    if any(hour_bucket(occurred_at) >= first_day and count.requests for occurred_at, _model, count in events):
        return False
    with database.connect() as conn:
        return db.stored_requests_kept(conn, Provider.CODEX, file_key, first_day) > 0


def _read_reaches_counted_history_with_fewer_requests(
    replaces: bool,
    cursor: FileCursor | None,
    plan: _Plan,
    end_offset: int,
    events: Sequence[tuple[datetime, str, TokenCounts]],
    database: Database,
    file_key: str,
    now: datetime,
) -> bool:
    if cursor is None or not replaces:
        return False
    original_is_absent = plan.read_before is None and not plan.original_still_holds
    lowered_floor = original_is_absent and plan.must_hold < cursor.byte_offset
    if end_offset < cursor.byte_offset or lowered_floor:
        return _replacement_erases_retained_requests(events, database, file_key, now)
    return _replacement_drops_requests(events, database, file_key, now)


def _replacement_floor(cursor: FileCursor, read_copy: _Copy | None) -> int:
    return cursor.byte_offset if read_copy is None else min(cursor.byte_offset, read_copy.size)


def _usage_object(raw: object) -> dict[Any, Any]:
    if not isinstance(raw, dict):
        raise ValueError("token usage is not an object")
    return raw


def _read_request_usage(raw: object) -> RawTotals:
    usage = _usage_object(raw)
    return {field: token_count(usage, field) for field in _TOTAL_FIELDS}


def _read_running_totals(raw: object, previous_totals: RawTotals | None) -> RawTotals:
    usage = _usage_object(raw)
    if any(field not in usage for field in _ALWAYS_WRITTEN_FIELDS):
        raise ValueError("running totals leave out a counter")
    cache_write_was_counted = previous_totals is not None and previous_totals[_CACHE_WRITE_FIELD] > 0
    if _CACHE_WRITE_FIELD not in usage and cache_write_was_counted:
        raise ValueError("running totals leave out a counter the file had been counting")
    return _read_request_usage(usage)


def usage_growth(totals: RawTotals, last_request: RawTotals | None, previous_totals: RawTotals | None) -> RawTotals:
    if previous_totals is not None and all(totals[field] >= previous_totals[field] for field in _TOTAL_FIELDS):
        return {field: totals[field] - previous_totals[field] for field in _TOTAL_FIELDS}
    return dict(last_request) if last_request is not None else dict.fromkeys(_TOTAL_FIELDS, 0)


def normalize(growth: RawTotals) -> TokenCounts:
    cached = growth["cached_input_tokens"]
    counts = TokenCounts(
        input_tokens=max(0, growth["input_tokens"] - cached),
        cache_read_tokens=cached,
        cache_write_tokens=growth["cache_write_input_tokens"],
        output_tokens=growth["output_tokens"],
        reasoning_tokens=growth["reasoning_output_tokens"],
    )
    return counts.plus(TokenCounts(requests=1)) if counts.total_tokens > 0 else counts


def _read_planned_copy(
    database: Database,
    file_key: str,
    cursor: FileCursor | None,
    plan: _Plan,
    request_counts: list[dict[str, Any]] | None,
    now: datetime,
) -> ScanReport | None:
    events: list[tuple[datetime, str, TokenCounts]] = []
    malformed = 0
    with open_session_file(plan.copy.path) as rollout:
        opened = rollout._stat
        if (opened.st_dev, opened.st_ino) != (plan.copy.device, plan.copy.inode):
            return ScanReport()
        if rollout.size_when_opened < plan.copy.size:
            return ScanReport()
        state, replaces = _FileState(path=str(plan.copy.path)), plan.replaces
        if plan.read_before is not None and rollout.resume(cursor, plan.read_before.anchor):
            state, replaces = plan.read_before, False
        for line in rollout.lines():
            if not any(marker in line for marker in _RELEVANT_MARKERS):
                continue
            try:
                event = _apply_line(line, state)
            except ValueError:
                malformed += 1
                continue
            if event is not None:
                events.append(event)
        end_offset = rollout.end_offset
        if end_offset < plan.must_hold and (
            not _replacement_holds_more_requests(events, database, file_key)
            or _replacement_drops_requests(events, database, file_key, now)
        ):
            return None
        state.anchor = rollout.anchor
        carried = state.to_json()
        if request_counts is not None:
            carried["request_counts"] = request_counts
        carried["counted_requests"] = _counted_after(cursor, replaces, _request_total(events), database, file_key)
        position = rollout.position(carried)

    if _read_reaches_counted_history_with_fewer_requests(
        replaces, cursor, plan, end_offset, events, database, file_key, now
    ):
        return None
    with database.connect() as conn:
        account_id = _account_id(conn, state.account_external_id, now)
        usage: dict[UsageKey, TokenCounts] = {}
        for occurred_at, model, counts in events:
            key = UsageKey(account_id, hour_bucket(occurred_at), model)
            usage[key] = usage.get(key, TokenCounts()).plus(counts)
        if replaces:
            db.replace_file_usage(conn, Provider.CODEX, file_key, usage)
        else:
            db.add_file_usage(conn, Provider.CODEX, file_key, usage)
        db.save_cursor(conn, Provider.CODEX, file_key, position)
    return ScanReport(files_read=1, malformed_records=malformed)


def _remember_request_counts(
    database: Database,
    file_key: str,
    cursor: FileCursor | None,
    request_counts: list[dict[str, Any]] | None,
) -> None:
    if cursor is None or request_counts is None or not isinstance(cursor.state, dict):
        return
    if cursor.state.get("request_counts") == request_counts:
        return
    state = dict(cursor.state)
    state["request_counts"] = request_counts
    with database.connect() as conn:
        db.save_cursor(
            conn,
            Provider.CODEX,
            file_key,
            FileCursor(cursor.byte_offset, cursor.size, cursor.mtime_ns, state),
        )


class CodexSessionScanner:
    def __init__(self, home: Path):
        self._home = directory_identity(home)
        self._backfill_floor: list[int] = []

    @property
    def provider(self) -> Provider:
        return Provider.CODEX

    def scan(
        self, database: Database, resolve_account: AccountResolver, now: datetime, backfill_days: int
    ) -> ScanReport:
        with database.connect() as conn:
            cursors = db.load_cursors(conn, Provider.CODEX)
        floor_ns = stable_backfill_floor_ns(self._backfill_floor, now, backfill_days)
        rollouts, report = self._rollout_files()
        listed_in_full = report.files_failed == 0
        for name, paths in sorted(rollouts.items()):
            file_key = str(self._home / name)
            scan_rollout = partial(
                self._scan_rollout, database, file_key, paths, cursors.get(file_key), now, floor_ns, listed_in_full
            )
            report = report.merged(isolated(name, scan_rollout))
        return report

    def _rollout_files(self) -> tuple[dict[str, list[Path]], ScanReport]:
        files: dict[str, list[Path]] = {}
        report = ScanReport()
        for directory in (self._home / "archived_sessions", self._home / "sessions"):
            rollouts, unlistable = session_files(directory, "rollout-*.jsonl")
            report = report.merged(unlistable)
            for rollout in rollouts:
                files.setdefault(rollout.name, []).append(rollout)
        return files, report

    def _scan_rollout(
        self,
        database: Database,
        file_key: str,
        paths: Sequence[Path],
        cursor: FileCursor | None,
        now: datetime,
        floor_ns: int,
        home_was_listed_in_full: bool,
    ) -> ScanReport:
        """Raises:
        OSError: a copy cannot be examined or read, or its directory could not be listed.
        ValueError: the usage cannot be stored."""
        skipped: set[tuple[int, int]] = set()
        copies = [_copy_at(path) for path in paths]
        while True:
            decision = _plan(
                cursor,
                [copy for copy in copies if (copy.device, copy.inode) not in skipped],
                floor_ns,
                home_was_listed_in_full,
                database,
                file_key,
            )
            plan = decision.plan
            if plan is None:
                _remember_request_counts(database, file_key, cursor, decision.request_counts)
                return ScanReport()
            identity = (plan.copy.device, plan.copy.inode)
            report = _read_planned_copy(database, file_key, cursor, plan, decision.request_counts, now)
            if report is not None:
                return report
            recorded = _recorded_device_and_inode(None if cursor is None else cursor.state)
            if cursor is not None and recorded == identity and plan.copy.size >= cursor.byte_offset:
                return ScanReport()
            skipped.add(identity)


def _apply_line(line: bytes, state: _FileState) -> tuple[datetime, str, TokenCounts] | None:
    record = parse_json_line(line)
    if not isinstance(record, dict):
        return None
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    record_type = record.get("type")
    if record_type == "session_meta":
        state.account_external_id = _workspace_id(payload.get("creator_account_id"))
        return None
    if record_type == "turn_context":
        state.model = named_model(payload.get("model")) or state.model
        return None
    if record_type != "event_msg" or payload.get("type") != "token_count":
        return None
    info = payload.get("info")
    if not isinstance(info, dict):
        return None
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, str):
        raise ValueError("token_count record has no timestamp")
    occurred_at = parse_iso(timestamp)
    totals = _read_running_totals(info.get("total_token_usage"), state.previous_totals)
    last_raw = info.get("last_token_usage")
    last_request = _read_request_usage(last_raw) if last_raw is not None else None
    counts = normalize(usage_growth(totals, last_request, state.previous_totals))
    state.previous_totals = totals
    return None if counts.is_empty else (occurred_at, state.model, counts)


def _account_id(conn: sqlite3.Connection, external_id: str | None, now: datetime) -> int:
    if external_id is None:
        return UNATTRIBUTED_ACCOUNT_ID
    existing = db.find_account_id(conn, Provider.CODEX, external_id)
    if existing is not None:
        return existing
    placeholder = AccountIdentity(
        provider=Provider.CODEX,
        external_id=external_id,
        default_label=f"Codex workspace {external_id[:PLACEHOLDER_LABEL_ID_LENGTH]}",
    )
    return db.upsert_account(conn, placeholder, now)
