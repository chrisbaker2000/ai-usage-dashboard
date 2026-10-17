import os
from collections.abc import Iterator
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Annotated

from pydantic import Field

from usagedash import db
from usagedash.db import Database, FileCursor
from usagedash.models import Provider, TokenCounts, UsageKey
from usagedash.paths import directory_identity
from usagedash.scanners.base import (
    MAX_TOKENS_PER_RECORD,
    UNKNOWN_MODEL,
    AccountResolver,
    ScanReport,
    SessionFileChangedError,
    file_identity,
    isolated,
    named_model,
    needs_scan,
    parse_json_line,
    session_files,
    stable_backfill_floor_ns,
)
from usagedash.timeutil import hour_bucket
from usagedash.validation import Instant, LenientModel

TokenCount = Annotated[int, Field(ge=0, le=MAX_TOKENS_PER_RECORD)]
USAGE_FILE_DEPTH = 2


class _Counts(LenientModel):
    input_tokens: TokenCount = Field(default=0, alias="inputTokens")
    output_tokens: TokenCount = Field(default=0, alias="outputTokens")
    cached_read_tokens: TokenCount = Field(default=0, alias="cachedReadTokens")
    cache_creation_tokens: TokenCount = Field(default=0, alias="cacheCreationTokens")
    reasoning_tokens: TokenCount = Field(default=0, alias="reasoningTokens")
    model_calls: TokenCount = Field(default=0, alias="modelCalls")


class _Turn(_Counts):
    ended_at: Instant = Field(alias="endedAt")
    primary_model_id: str | None = Field(default=None, alias="primaryModelId")
    model_usage: dict[str, _Counts] | None = Field(default=None, alias="modelUsage")


class _UsageFile(LenientModel):
    turns: list[_Turn]


def normalize(counts: _Counts) -> TokenCounts:
    return TokenCounts(
        input_tokens=max(0, counts.input_tokens - counts.cached_read_tokens),
        cache_read_tokens=counts.cached_read_tokens,
        cache_write_tokens=counts.cache_creation_tokens,
        output_tokens=counts.output_tokens,
        reasoning_tokens=counts.reasoning_tokens,
        requests=counts.model_calls,
    )


def _counts_by_model(turn: _Turn) -> Iterator[tuple[str | None, _Counts]]:
    if turn.model_usage:
        yield from turn.model_usage.items()
    else:
        yield turn.primary_model_id, turn


def usage_from_file(raw: bytes, resolve_account: AccountResolver) -> dict[UsageKey, TokenCounts]:
    parsed = _UsageFile.model_validate(parse_json_line(raw))
    usage: dict[UsageKey, TokenCounts] = {}
    for turn in parsed.turns:
        account_id = resolve_account(turn.ended_at)
        bucket = hour_bucket(turn.ended_at)
        for model, counts in _counts_by_model(turn):
            key = UsageKey(account_id, bucket, named_model(model) or UNKNOWN_MODEL)
            usage[key] = usage.get(key, TokenCounts()).plus(normalize(counts))
    return usage


class GrokSessionScanner:
    def __init__(self, home: Path):
        self._home = directory_identity(home)
        self._backfill_floor: list[int] = []

    @property
    def provider(self) -> Provider:
        return Provider.GROK

    def scan(
        self, database: Database, resolve_account: AccountResolver, now: datetime, backfill_days: int
    ) -> ScanReport:
        usage_files, report = session_files(self._home / "sessions", "usage.json", depth=USAGE_FILE_DEPTH)
        with database.connect() as conn:
            cursors = db.load_cursors(conn, Provider.GROK)
        floor_ns = stable_backfill_floor_ns(self._backfill_floor, now, backfill_days)
        for path in usage_files:
            scan_file = partial(self._scan_file, database, path, cursors.get(str(path)), resolve_account, floor_ns)
            report = report.merged(isolated(path.parent.name, scan_file))
        return report

    def _scan_file(
        self,
        database: Database,
        path: Path,
        cursor: FileCursor | None,
        resolve_account: AccountResolver,
        floor_ns: int,
    ) -> ScanReport:
        stat = path.stat()
        if not needs_scan(cursor, stat, floor_ns):
            return ScanReport()
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            contents = handle.read()
            # Why: a buffered read can join two versions. A new change time means that total is neither.
            if file_identity(os.fstat(handle.fileno())) != file_identity(opened):
                raise SessionFileChangedError("the file was rewritten while it was being read")
        usage = usage_from_file(contents, resolve_account)
        with database.connect() as conn:
            db.replace_file_usage(conn, Provider.GROK, str(path), usage)
            db.save_cursor(
                conn,
                Provider.GROK,
                str(path),
                FileCursor(
                    byte_offset=opened.st_size,
                    size=opened.st_size,
                    mtime_ns=opened.st_mtime_ns,
                    state=file_identity(opened),
                ),
            )
        return ScanReport(files_read=1)
