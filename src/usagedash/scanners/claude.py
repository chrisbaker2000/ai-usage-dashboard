import hashlib
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

from usagedash import db
from usagedash.db import Database, FileCursor, MessageUsage
from usagedash.models import Provider, TokenCounts
from usagedash.paths import directory_identity
from usagedash.scanners.base import (
    UNKNOWN_MODEL,
    AccountResolver,
    ScanReport,
    isolated,
    named_model,
    needs_scan,
    open_session_file,
    parse_json_line,
    session_files,
    stable_backfill_floor_ns,
    token_count,
)
from usagedash.timeutil import parse_iso

_ASSISTANT_MARKER = b'"assistant"'
_USAGE_MARKER = b'"usage"'
MESSAGE_KEY_BYTES = 12
ANCHOR_KEY = "anchor"


@dataclass(frozen=True, slots=True)
class _ResponseLine:
    key: bytes
    occurred_at: datetime
    model: str
    counts: TokenCounts


def parse_transcript_line(line: bytes) -> _ResponseLine | None:
    if _ASSISTANT_MARKER not in line or _USAGE_MARKER not in line:
        return None
    record = parse_json_line(line)
    if not isinstance(record, dict) or record.get("type") != "assistant":
        return None
    message = record.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("usage"), dict):
        return None
    usage = message["usage"]
    details = usage.get("output_tokens_details")
    counts = TokenCounts(
        input_tokens=token_count(usage, "input_tokens"),
        cache_read_tokens=token_count(usage, "cache_read_input_tokens"),
        cache_write_tokens=token_count(usage, "cache_creation_input_tokens"),
        output_tokens=token_count(usage, "output_tokens"),
        reasoning_tokens=token_count(details, "thinking_tokens") if isinstance(details, dict) else 0,
        requests=1,
    )
    if counts.total_tokens == 0 and counts.reasoning_tokens == 0:
        return None
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, str):
        raise ValueError("assistant record has no timestamp")
    message_id = message.get("id")
    if not isinstance(message_id, str) or not message_id.strip():
        # Why malformed: the response id dedupes it. A line uuid differs per line, and a blank id merges them.
        raise ValueError("assistant record has no response id to count it once by")
    # Why the response id alone: requestId is optional, so including it counts one response twice.
    return _ResponseLine(
        key=hashlib.blake2b(message_id.encode("utf-8"), digest_size=MESSAGE_KEY_BYTES).digest(),
        occurred_at=parse_iso(timestamp),
        model=named_model(message.get("model")) or UNKNOWN_MODEL,
        counts=counts,
    )


def _saved_anchor(cursor: FileCursor | None) -> object:
    if cursor is None or not isinstance(cursor.state, dict):
        return None
    return cursor.state.get(ANCHOR_KEY)


def _as_message_usage(line: _ResponseLine, resolve_account: AccountResolver) -> MessageUsage:
    return MessageUsage(line.key, line.occurred_at, resolve_account(line.occurred_at), line.model, line.counts)


class ClaudeTranscriptScanner:
    def __init__(self, projects_dir: Path):
        self._projects_dir = directory_identity(projects_dir)
        self._backfill_floor: list[int] = []

    @property
    def provider(self) -> Provider:
        return Provider.CLAUDE

    def scan(
        self, database: Database, resolve_account: AccountResolver, now: datetime, backfill_days: int
    ) -> ScanReport:
        transcripts, report = session_files(self._projects_dir, "*.jsonl")
        with database.connect() as conn:
            cursors = db.load_cursors(conn, Provider.CLAUDE)
        floor_ns = stable_backfill_floor_ns(self._backfill_floor, now, backfill_days)
        for path in transcripts:
            scan_file = partial(self._scan_file, database, path, cursors.get(str(path)), resolve_account, floor_ns)
            report = report.merged(isolated(path.name, scan_file))
        return report

    def _scan_file(
        self,
        database: Database,
        path: Path,
        cursor: FileCursor | None,
        resolve_account: AccountResolver,
        floor_ns: int,
    ) -> ScanReport:
        if not needs_scan(cursor, path.stat(), floor_ns):
            return ScanReport()

        response_lines: list[_ResponseLine] = []
        malformed = 0
        with open_session_file(path) as transcript:
            # Why resume may restart: a rewrite is reread, and a stored response counts as nothing.
            transcript.resume(cursor, _saved_anchor(cursor))
            for line in transcript.lines():
                try:
                    response_line = parse_transcript_line(line)
                except ValueError:
                    malformed += 1
                    continue
                if response_line is not None:
                    response_lines.append(response_line)
            position = transcript.position({ANCHOR_KEY: transcript.anchor})

        with database.connect() as conn:
            db.add_message_usage(conn, (_as_message_usage(line, resolve_account) for line in response_lines))
            db.save_cursor(conn, Provider.CLAUDE, str(path), position)
        return ScanReport(files_read=1, malformed_records=malformed)
