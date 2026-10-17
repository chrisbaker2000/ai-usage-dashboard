"""SQLite store for accounts, quota snapshots, token usage, and scan state.

Quota snapshots are stored as a step function: a row is inserted only when a window's value changes, and an
unchanged re-observation just advances ``confirmed_at``. Each stored reading also takes the account's next
sequence number and stamps the rows it confirmed, so "the windows of the latest reading" is an exact match
on that number and never depends on timestamps. A step's two times are stored to the microsecond, because
history and retention compare them with a cutoff that can fall anywhere inside a second; two steps observed
within one second are still one before and one after such a cutoff. The account also keeps the observation
time of its newest accepted reading to the microsecond, which is what decides whether a later-arriving
reading is older and must be dropped. Other timestamps are stored to the second, except a recorded login
boundary, which keeps the fraction of the instant that could not be recorded. Token usage is stored per
contributing session file so that re-reading a file can never double count it.
"""

import fcntl
import json
import os
import sqlite3
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from usagedash.errors import ConfigError, StoreError
from usagedash.models import (
    MAX_TOKEN_COUNT,
    UNATTRIBUTED_ACCOUNT_ID,
    AccountIdentity,
    Fact,
    Provider,
    QuotaReading,
    TokenCounts,
    UsageKey,
)
from usagedash.private import ensure_private_directory, open_private_file
from usagedash.timeutil import (
    day_key,
    epoch_microseconds,
    from_epoch_microseconds,
    hour_bucket,
    in_utc,
    parse_iso,
    to_iso,
    whole_second_not_before,
)

SCHEMA_VERSION = 2
CATEGORICAL_SLOT_COUNT = 8
OVERFLOW_COLOR_SLOT = 0
SNAPSHOT_RETENTION = timedelta(days=90)
USAGE_RETENTION = timedelta(days=400)
"""How long token usage is kept, and with it the keys that stop a Claude response being counted twice."""
SQLITE_VARIABLE_CHUNK = 500
BUSY_TIMEOUT_SECONDS = 30.0

_STORABLE_COUNT = f"BETWEEN 0 AND {MAX_TOKEN_COUNT}"

CLAUDE_USAGE_FILE_KEY = "responses"
"""The one file key Claude usage is stored under: it is counted per response, not per transcript."""
_ONE_REQUEST = TokenCounts(requests=1)
_COUNT_FIELDS = (
    "input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
    "requests",
)

STATUS_KIND_QUOTA = "quota"
STATUS_KIND_TOKENS = "tokens"

_RECORDED_LOGIN_BOUNDARIES = """
CREATE TABLE recorded_login_boundaries (
  source_id TEXT NOT NULL,
  boundary_at TEXT NOT NULL,
  PRIMARY KEY (source_id, boundary_at)
)
"""

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE accounts (
      id INTEGER PRIMARY KEY,
      provider TEXT NOT NULL,
      external_id TEXT NOT NULL,
      label TEXT NOT NULL,
      label_is_custom INTEGER NOT NULL DEFAULT 0,
      email TEXT,
      plan TEXT,
      color_slot INTEGER NOT NULL,
      first_seen_at TEXT NOT NULL,
      last_live_at TEXT,
      facts_json TEXT NOT NULL DEFAULT '[]',
      reading_seq INTEGER NOT NULL DEFAULT 0,
      last_observed_us INTEGER,
      UNIQUE (provider, external_id)
    )
    """,
    """
    CREATE TABLE quota_snapshots (
      id INTEGER PRIMARY KEY,
      account_id INTEGER NOT NULL REFERENCES accounts(id),
      window_key TEXT NOT NULL,
      label TEXT NOT NULL,
      window_seconds INTEGER,
      used_percent REAL NOT NULL,
      resets_at TEXT,
      blocking INTEGER NOT NULL,
      source TEXT NOT NULL,
      observed_us INTEGER NOT NULL,
      confirmed_us INTEGER NOT NULL,
      confirmed_seq INTEGER NOT NULL
    )
    """,
    """
    CREATE INDEX idx_quota_account_window ON quota_snapshots(account_id, window_key, observed_us)
    """,
    """
    CREATE INDEX idx_quota_account_seq ON quota_snapshots(account_id, confirmed_seq)
    """,
    """
    CREATE TABLE active_account_log (
      id INTEGER PRIMARY KEY,
      source_id TEXT NOT NULL,
      account_id INTEGER REFERENCES accounts(id),
      started_at TEXT NOT NULL,
      last_observed_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX idx_active_source ON active_account_log(source_id)
    """,
    """
    CREATE TABLE source_state (
      source_id TEXT PRIMARY KEY,
      provider TEXT NOT NULL,
      position INTEGER NOT NULL,
      current_account_id INTEGER REFERENCES accounts(id),
      checked_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE source_status (
      source_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      last_attempt_at TEXT NOT NULL,
      last_success_at TEXT,
      error_type TEXT,
      error_message TEXT,
      PRIMARY KEY (source_id, kind)
    )
    """,
    f"""
    CREATE TABLE token_usage (
      provider TEXT NOT NULL,
      file_key TEXT NOT NULL,
      account_id INTEGER NOT NULL,
      bucket_hour TEXT NOT NULL,
      model TEXT NOT NULL,
      input_tokens INTEGER NOT NULL CHECK (input_tokens {_STORABLE_COUNT}),
      cache_read_tokens INTEGER NOT NULL CHECK (cache_read_tokens {_STORABLE_COUNT}),
      cache_write_tokens INTEGER NOT NULL CHECK (cache_write_tokens {_STORABLE_COUNT}),
      output_tokens INTEGER NOT NULL CHECK (output_tokens {_STORABLE_COUNT}),
      reasoning_tokens INTEGER NOT NULL CHECK (reasoning_tokens {_STORABLE_COUNT}),
      requests INTEGER NOT NULL CHECK (requests {_STORABLE_COUNT}),
      PRIMARY KEY (provider, file_key, account_id, bucket_hour, model)
    ) WITHOUT ROWID
    """,
    """
    CREATE INDEX idx_token_usage_bucket ON token_usage(bucket_hour)
    """,
    """
    CREATE TABLE scan_cursors (
      provider TEXT NOT NULL,
      file_key TEXT NOT NULL,
      byte_offset INTEGER NOT NULL,
      size INTEGER NOT NULL,
      mtime_ns INTEGER NOT NULL,
      state_json TEXT NOT NULL,
      PRIMARY KEY (provider, file_key)
    ) WITHOUT ROWID
    """,
    f"""
    CREATE TABLE claude_seen_messages (
      key BLOB PRIMARY KEY,
      first_line_at INTEGER NOT NULL,
      account_id INTEGER NOT NULL,
      bucket_hour TEXT NOT NULL,
      model TEXT NOT NULL,
      input_tokens INTEGER NOT NULL CHECK (input_tokens {_STORABLE_COUNT}),
      cache_read_tokens INTEGER NOT NULL CHECK (cache_read_tokens {_STORABLE_COUNT}),
      cache_write_tokens INTEGER NOT NULL CHECK (cache_write_tokens {_STORABLE_COUNT}),
      output_tokens INTEGER NOT NULL CHECK (output_tokens {_STORABLE_COUNT}),
      reasoning_tokens INTEGER NOT NULL CHECK (reasoning_tokens {_STORABLE_COUNT})
    ) WITHOUT ROWID
    """,
    """
    CREATE INDEX idx_claude_seen_place ON claude_seen_messages(bucket_hour, account_id, model)
    """,
    """
    CREATE TABLE claude_expired_messages (
      key BLOB PRIMARY KEY
    ) WITHOUT ROWID
    """,
    _RECORDED_LOGIN_BOUNDARIES,
)

# Why a migration and not an edit of the version-1 statements: databases already stamped at version 1
# never re-run those statements. The stamp moves only after this table exists.
_MIGRATIONS: dict[int, tuple[str, ...]] = {1: (_RECORDED_LOGIN_BOUNDARIES,)}


@dataclass(frozen=True, slots=True)
class AccountRow:
    id: int
    provider: Provider
    external_id: str
    label: str
    label_is_custom: bool
    email: str | None
    plan: str | None
    color_slot: int
    first_seen_at: datetime
    last_live_at: datetime | None
    facts: tuple[Fact, ...]


@dataclass(frozen=True, slots=True)
class SnapshotRow:
    account_id: int
    window_key: str
    label: str
    window_seconds: int | None
    used_percent: float
    resets_at: datetime | None
    blocking: bool
    source: str
    observed_at: datetime
    confirmed_at: datetime


@dataclass(frozen=True, slots=True)
class SourceStateRow:
    source_id: str
    provider: Provider
    position: int
    current_account_id: int | None
    checked_at: datetime


@dataclass(frozen=True, slots=True)
class SourceStatusRow:
    source_id: str
    kind: str
    last_attempt_at: datetime
    last_success_at: datetime | None
    error_type: str | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class FileCursor:
    """How far a session file has been read, plus scanner-specific carry-over state.

    ``state`` is whatever JSON value the scanner saved, or None when the stored text could not be read
    back as JSON. It comes from the store, so the scanner that owns it checks its shape before use.
    """

    byte_offset: int
    size: int
    mtime_ns: int
    state: object


@dataclass(frozen=True, slots=True)
class MessageUsage:
    """One Claude response's usage, as one transcript line states it.

    ``occurred_at``, ``account_id``, and ``model`` are the line's own: when it was written, who was
    logged in then, and the model it names. They are where the response is counted if this is its
    earliest line. ``counts`` is the usage the line reports. Its ``requests`` is not read: a response
    is one request, however many lines report it.
    """

    key: bytes
    occurred_at: datetime
    account_id: int
    model: str
    counts: TokenCounts


@dataclass(frozen=True, slots=True)
class SeenMessage:
    """What has been counted for one Claude response, and where.

    ``usage_key`` is the account, hour, and model of the earliest line of the response read so far,
    and everything counted for the response is there. ``counts`` is the largest value read for each
    count. It carries no request; a response is one request.
    """

    usage_key: UsageKey
    counts: TokenCounts


@dataclass(frozen=True, slots=True)
class _CountedMessage:
    """What the store holds for one Claude response: where it is counted, by which line, and how much."""

    place: UsageKey
    first_line_at: int
    counts: TokenCounts

    @property
    def order(self) -> tuple[int, bool, str, int]:
        """Sort key of the line that placed the response: its instant, then whether it names an account.

        The instant decides. At one instant a line credited to an account comes before one credited to
        nobody: a transcript copied into another home is read there with that home's login timeline,
        which may not reach back to the line, and the copy must not take usage away from the account
        the original was credited to. The model name and the account id only make the order total, so
        that two lines at one instant are placed the same way whichever of them is read first.
        """
        is_unattributed = self.place.account_id == UNATTRIBUTED_ACCOUNT_ID
        return (self.first_line_at, is_unattributed, self.place.model, self.place.account_id)


@dataclass(frozen=True, slots=True)
class TokenUsageRow:
    provider: Provider
    account_id: int
    bucket_hour: str
    model: str
    counts: TokenCounts


@contextmanager
def _exclusive_lock_on(directory: Path) -> Generator[None]:
    """Hold an exclusive lock on ``directory`` while the block runs, waiting for whoever holds it now.

    The lock is on the directory itself, so it needs no file of its own, and it ends when its
    descriptor is closed, however the block or the process ends.

    Raises:
        ConfigError: if the directory cannot be opened or locked.
    """
    descriptor: int | None = None
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ConfigError(f"cannot lock {directory}: {type(error).__name__}") from error
    try:
        yield
    finally:
        os.close(descriptor)


class Database:
    """Owns the database file.

    ``connect()`` is one write transaction and ``read()`` is one read-only view, each on a short-lived
    connection. Every write goes through ``connect()``, which takes the write lock before its block runs.
    A function that reads a row and then writes based on what it read (registering an account, recording
    a reading) therefore cannot interleave with another writer, in this process or any other.
    """

    def __init__(self, path: Path):
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> None:
        """Create the data directory, the database file, and the schema if absent, all owner-only.

        The schema and its version stamp are written in one transaction. A failure while creating the
        first schema leaves an empty database that the next attempt can initialize. A failure while
        migrating a database that already has a schema leaves that schema in place. The version is
        read inside that transaction, so two processes starting together cannot both try to create
        the tables.

        One process does this at a time: an exclusive lock on the data directory is held from before
        the file is touched until the journal mode is set. Switching a new database to write-ahead
        logging needs the write lock, and that is the one place SQLite does not wait for it. With
        another starter part-way through its own schema transaction, the switch failed at once with
        "database is locked", so of two commands started together on a new database one could fail.

        Raises:
            ConfigError: if the data directory cannot be made private or locked, or the database file
                cannot be made private.
            StoreError: if the file is not a database SQLite can open and set up, or was written by a
                schema version other than the one this code understands.
        """
        ensure_private_directory(self._path.parent)
        with _exclusive_lock_on(self._path.parent):
            # Why before SQLite opens it: SQLite gives its journal files the database file's mode, so
            # the file must already be owner-only when the first connection creates them.
            os.close(open_private_file(self._path))
            try:
                self._create_schema()
            except sqlite3.Error as error:
                raise StoreError(f"cannot set up the database {self._path}: {type(error).__name__}") from error

    def _create_schema(self) -> None:
        with self.connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                for statement in _SCHEMA_STATEMENTS:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            elif version < SCHEMA_VERSION:
                self._migrate(conn, version)
            elif version != SCHEMA_VERSION:
                raise StoreError(f"database schema version {version} is not supported (expected {SCHEMA_VERSION})")
        # Why its own connection: the journal mode cannot be changed inside a transaction.
        conn = self._open()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        finally:
            conn.close()

    def _migrate(self, conn: sqlite3.Connection, version: int) -> None:
        """Advance ``version`` to ``SCHEMA_VERSION`` inside the open schema transaction.

        Raises:
            StoreError: if no migration is defined for a version this code would otherwise pass through.
        """
        while version < SCHEMA_VERSION:
            statements = _MIGRATIONS.get(version)
            if statements is None:
                raise StoreError(f"database schema version {version} has no migration")
            for statement in statements:
                conn.execute(statement)
            version += 1
            conn.execute(f"PRAGMA user_version = {version}")

    def _open(self) -> sqlite3.Connection:
        """Open a connection that starts no transaction on its own; the two context managers below do."""
        conn = sqlite3.connect(self._path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection]:
        """Run the block as one write transaction: committed if it returns, rolled back if it raises.

        The write lock is taken before the block starts, waiting up to ``BUSY_TIMEOUT_SECONDS`` for
        another writer to finish, so nothing the block reads can change before it commits.
        """
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    @contextmanager
    def read(self) -> Generator[sqlite3.Connection]:
        """Run the block against one consistent, read-only view of the database.

        Every query in the block sees the same committed state, however many writes happen meanwhile,
        and the block never waits for a writer. A write attempted through this connection is refused.
        """
        conn = self._open()
        try:
            conn.execute("PRAGMA query_only = ON")
            conn.execute("BEGIN")
            try:
                yield conn
            finally:
                conn.execute("ROLLBACK")
        finally:
            conn.close()


def _optional_time(value: str | None) -> datetime | None:
    return None if value is None else parse_iso(value)


def _refuse_constant(name: str) -> object:
    raise ValueError(f"{name} is not a JSON number")


def decode_json(text: str) -> object:
    """Decode JSON the store keeps, refusing what is not JSON although Python's decoder accepts it.

    That decoder reads ``NaN``, ``Infinity``, and ``-Infinity`` as numbers. None is JSON, and a stored
    document holding one did not come from this store: it is damaged and must be refused like any other
    text that is not JSON. This is the single decoder for what the store keeps as JSON.

    Raises:
        ValueError: if ``text`` is not JSON, or holds one of those constants.
        RecursionError: if it is nested too deeply to decode.
    """
    return json.loads(text, parse_constant=_refuse_constant)


def encode_json(value: object) -> str:
    """Encode a value the store keeps as JSON, refusing a number that JSON cannot write.

    Python's encoder writes a non-finite float as ``NaN`` or ``Infinity``, which is not JSON, and which
    ``decode_json`` would then refuse to read back. This is the single encoder for what the store keeps.

    Raises:
        ValueError: if ``value`` holds a non-finite float.
        TypeError: if ``value`` holds something JSON cannot write at all.
    """
    return json.dumps(value, allow_nan=False)


def _facts_to_json(facts: Sequence[Fact]) -> str:
    return encode_json(
        [
            {"label": fact.label, "value": fact.value, "resets_at": to_iso(fact.resets_at) if fact.resets_at else None}
            for fact in facts
        ]
    )


def _facts_from_json(raw: str) -> tuple[Fact, ...]:
    """Rebuild an account's facts from what ``_facts_to_json`` stored.

    Raises:
        ValueError: if the stored text is not JSON (``decode_json``), or not a list.
    """
    items = decode_json(raw)
    if not isinstance(items, list):
        raise ValueError("an account's stored facts are not a list")
    return tuple(
        Fact(label=item["label"], value=item["value"], resets_at=_optional_time(item["resets_at"])) for item in items
    )


def _account_from_row(row: sqlite3.Row) -> AccountRow:
    return AccountRow(
        id=row["id"],
        provider=Provider(row["provider"]),
        external_id=row["external_id"],
        label=row["label"],
        label_is_custom=bool(row["label_is_custom"]),
        email=row["email"],
        plan=row["plan"],
        color_slot=row["color_slot"],
        first_seen_at=parse_iso(row["first_seen_at"]),
        last_live_at=_optional_time(row["last_live_at"]),
        facts=_facts_from_json(row["facts_json"]),
    )


def _next_color_slot(conn: sqlite3.Connection) -> int:
    """Hand out categorical slots in fixed creation order; accounts past the palette share the neutral slot."""
    count: int = conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    return count + 1 if count < CATEGORICAL_SLOT_COUNT else OVERFLOW_COLOR_SLOT


def _preferred_plan(stored: str | None, incoming: str | None) -> str | None:
    """Keep the more specific plan name when two sources describe one account.

    Why: a direct read knows the tier ("Max 20x") while a CLI-mode read only knows the family ("Max").
    Without this the stored plan would flip between the two on every cycle.
    """
    if incoming is None:
        return stored
    if stored is not None and stored.startswith(incoming):
        return stored
    return incoming


def upsert_account(conn: sqlite3.Connection, identity: AccountIdentity, now: datetime) -> int:
    """Insert the account or refresh its identity fields, preserving a label the user set. Returns its id."""
    existing = conn.execute(
        "SELECT id, label_is_custom, plan FROM accounts WHERE provider = ? AND external_id = ?",
        (identity.provider.value, identity.external_id),
    ).fetchone()
    if existing is None:
        inserted = conn.execute(
            "INSERT INTO accounts (provider, external_id, label, email, plan, color_slot, first_seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id",
            (
                identity.provider.value,
                identity.external_id,
                identity.default_label,
                identity.email,
                identity.plan,
                _next_color_slot(conn),
                to_iso(now),
            ),
        ).fetchone()
        new_id: int = inserted["id"]
        return new_id
    account_id: int = existing["id"]
    conn.execute(
        "UPDATE accounts SET email = COALESCE(?, email), plan = ? WHERE id = ?",
        (identity.email, _preferred_plan(existing["plan"], identity.plan), account_id),
    )
    if not existing["label_is_custom"]:
        conn.execute("UPDATE accounts SET label = ? WHERE id = ?", (identity.default_label, account_id))
    return account_id


def find_account_id(conn: sqlite3.Connection, provider: Provider, external_id: str) -> int | None:
    row = conn.execute(
        "SELECT id FROM accounts WHERE provider = ? AND external_id = ?", (provider.value, external_id)
    ).fetchone()
    return None if row is None else row["id"]


def mark_account_live(conn: sqlite3.Connection, account_id: int, now: datetime) -> None:
    conn.execute("UPDATE accounts SET last_live_at = ? WHERE id = ?", (to_iso(now), account_id))


def rename_account(conn: sqlite3.Connection, account_id: int, label: str) -> bool:
    """Set a user-chosen label. Returns False when no such account exists."""
    cursor = conn.execute("UPDATE accounts SET label = ?, label_is_custom = 1 WHERE id = ?", (label, account_id))
    return cursor.rowcount == 1


def list_accounts(conn: sqlite3.Connection) -> list[AccountRow]:
    return [_account_from_row(row) for row in conn.execute("SELECT * FROM accounts ORDER BY id")]


def account_label(conn: sqlite3.Connection, account_id: int) -> str | None:
    row = conn.execute("SELECT label FROM accounts WHERE id = ?", (account_id,)).fetchone()
    return None if row is None else row["label"]


def record_reading(conn: sqlite3.Connection, account_id: int, reading: QuotaReading) -> None:
    """Store a reading as step-function rows and replace the account's facts.

    A reading observed before the account's newest accepted reading is ignored as a whole, windows and
    facts alike, so an aged log entry can never overwrite a fresher measurement or its facts. The
    comparison uses the account's own watermark, to the microsecond: it does not depend on which windows
    the newest reading happened to carry (it may have carried none). A reading observed at exactly the
    watermark is accepted.

    A window's step continues while everything measured about it is unchanged: how much is used, when
    it resets, whether it blocks, and how long it is. Anything else starts a new step.

    Raises:
        ValueError: if ``account_id`` names no account, or the reading's time cannot be expressed in UTC.
    """
    # Why converted first: a time at the edge of the calendar in a zone west of UTC has an epoch value
    # but no UTC datetime, so it could be stored and then never read back.
    observed_us = epoch_microseconds(in_utc(reading.observed_at))
    accepted = conn.execute(
        "UPDATE accounts SET reading_seq = reading_seq + 1, last_observed_us = ?, facts_json = ? "
        "WHERE id = ? AND (last_observed_us IS NULL OR last_observed_us <= ?) RETURNING reading_seq",
        (observed_us, _facts_to_json(reading.facts), account_id, observed_us),
    ).fetchone()
    if accepted is None:
        if conn.execute("SELECT 1 FROM accounts WHERE id = ?", (account_id,)).fetchone() is None:
            raise ValueError(f"cannot record a reading for unknown account {account_id}")
        return
    sequence: int = accepted["reading_seq"]
    for window in reading.windows:
        resets = to_iso(window.resets_at) if window.resets_at else None
        latest = conn.execute(
            "SELECT id, used_percent, resets_at, blocking, window_seconds FROM quota_snapshots "
            "WHERE account_id = ? AND window_key = ? ORDER BY id DESC LIMIT 1",
            (account_id, window.key),
        ).fetchone()
        measured = (window.used_percent, resets, int(window.blocking), window.window_seconds)
        if latest is not None and tuple(latest)[1:] == measured:
            # Why only the label is updated in place: it is what the window is called, not something
            # measured about it. Whether it blocks, and how long it is, are part of the step, and
            # rewriting them in an earlier row would report them before they were observed.
            conn.execute(
                "UPDATE quota_snapshots SET confirmed_us = ?, confirmed_seq = ?, label = ? WHERE id = ?",
                (observed_us, sequence, window.label, latest["id"]),
            )
            continue
        conn.execute(
            "INSERT INTO quota_snapshots (account_id, window_key, label, window_seconds, used_percent, resets_at, "
            "blocking, source, observed_us, confirmed_us, confirmed_seq) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                account_id,
                window.key,
                window.label,
                window.window_seconds,
                window.used_percent,
                resets,
                int(window.blocking),
                reading.source,
                observed_us,
                observed_us,
                sequence,
            ),
        )


def _snapshot_from_row(row: sqlite3.Row) -> SnapshotRow:
    return SnapshotRow(
        account_id=row["account_id"],
        window_key=row["window_key"],
        label=row["label"],
        window_seconds=row["window_seconds"],
        used_percent=row["used_percent"],
        resets_at=_optional_time(row["resets_at"]),
        blocking=bool(row["blocking"]),
        source=row["source"],
        observed_at=from_epoch_microseconds(row["observed_us"]),
        confirmed_at=from_epoch_microseconds(row["confirmed_us"]),
    )


def latest_snapshots(conn: sqlite3.Connection) -> list[SnapshotRow]:
    """Return each account's windows as of its most recent reading.

    A reading stamps exactly one row per window it reports, so matching the account's current sequence
    number returns that reading's windows and nothing else. A window the provider stopped reporting keeps
    an older number and drops out, even when both readings were stored within the same second.
    """
    rows = conn.execute(
        "SELECT s.* FROM quota_snapshots s JOIN accounts a ON a.id = s.account_id "
        "WHERE s.confirmed_seq = a.reading_seq ORDER BY s.account_id, s.id"
    )
    return [_snapshot_from_row(row) for row in rows]


def snapshot_history(conn: sqlite3.Connection, since: datetime) -> list[SnapshotRow]:
    """Return every step in effect at or after ``since``, per account window, oldest first.

    A step is in effect from when it was observed until the next step of the same account window is
    observed, and the newest step of a window stays in effect. When it was last confirmed is not when
    it ended. So this is every step observed after ``since`` and, for each account window, the last
    one observed at or before it: the state the range starts in.

    A window's steps are stored in the order they were observed, because a reading older than the
    account's newest stored one is dropped. The last one observed by a given moment is therefore the
    one with the largest id among those observed by then.

    Both sides of the comparison are whole microseconds, so ``since`` can fall between two steps
    observed within one second and still have one on each side.

    Raises:
        ValueError: if ``since`` is naive.
    """
    moment = epoch_microseconds(since)
    rows = conn.execute(
        "SELECT * FROM quota_snapshots WHERE observed_us > ? OR id IN ("
        "SELECT MAX(id) FROM quota_snapshots WHERE observed_us <= ? GROUP BY account_id, window_key) "
        "ORDER BY account_id, window_key, observed_us, id",
        (moment, moment),
    )
    return [_snapshot_from_row(row) for row in rows]


def observe_active_account(conn: sqlite3.Connection, source_id: str, account_id: int | None, now: datetime) -> None:
    """Extend the source's current interval, or open a new one when who is logged in changed.

    ``account_id`` is None when nobody is logged in. That is an interval like any other: without it, the
    account that logged out would keep being credited with whatever happened afterwards, and logging
    back in to the same account would look like one unbroken stay.

    The current interval is the one observed last (``_current_interval``), not the one with the latest
    time. A clock that steps back would otherwise leave an earlier login current after a later one was
    seen. For the same reason no time stored here goes back: a new interval starts no earlier than the
    previous one was last observed, and an interval's last observation only moves forward, so the
    timeline in time order is the timeline in the order it was observed.

    An interval's start is stored to the second, and usage is attributed by comparing its own time, which
    has fractions, with that start. So a new interval starts at the first whole second at or after
    ``now``: cut down to the second, it would begin before the login was seen and take usage from the
    account that was logged in until then.

    Raises:
        ValueError: if ``now`` is naive or at the very end of the calendar.
    """
    start = whole_second_not_before(now)
    latest = _current_interval(conn, source_id)
    if latest is not None:
        last_observed = parse_iso(latest["last_observed_at"])
        if latest["account_id"] == account_id:
            conn.execute(
                "UPDATE active_account_log SET last_observed_at = ? WHERE id = ?",
                (to_iso(max(now, last_observed)), latest["id"]),
            )
            return
        start = max(start, last_observed)
    started_at = to_iso(start)
    conn.execute(
        "INSERT INTO active_account_log (source_id, account_id, started_at, last_observed_at) VALUES (?, ?, ?, ?)",
        (source_id, account_id, started_at, started_at),
    )


def _current_interval(conn: sqlite3.Connection, source_id: str) -> sqlite3.Row | None:
    """Return the source's interval observed last, by the order intervals were recorded in."""
    row: sqlite3.Row | None = conn.execute(
        "SELECT id, account_id, started_at, last_observed_at FROM active_account_log WHERE source_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (source_id,),
    ).fetchone()
    return row


def acknowledge_recorded_login_boundary(conn: sqlite3.Connection, source_id: str, moment: datetime) -> None:
    conn.execute(
        "INSERT INTO recorded_login_boundaries (source_id, boundary_at) VALUES (?, ?) "
        "ON CONFLICT (source_id, boundary_at) DO NOTHING",
        (source_id, in_utc(moment).isoformat()),
    )


def record_unrecorded_gap(conn: sqlite3.Connection, source_id: str, moment: datetime) -> None:
    """Open a nobody interval for a boundary whose recovery file could not be written.

    The login transactions already failed. This runs in the transaction that records that failure,
    so a later collector still sees the gap. It does not go through ``observe_active_account``.

    When nobody is already the open interval, that interval is widened to cover ``moment`` instead
    of acknowledging the boundary and leaving the earlier span on the preceding account. Its last
    observation only moves forward, so a login recorded after a backward clock step starts at the
    gap rather than inside it. The start moves earlier only while it stays after the preceding
    interval: moving onto that interval's start would make the two begin together and the later
    one would take the preceding login's whole span.

    Raises:
        ValueError: if ``moment`` is naive or at the very end of the calendar.
    """
    start = whole_second_not_before(moment)
    latest = _current_interval(conn, source_id)
    if latest is not None:
        last_observed = parse_iso(latest["last_observed_at"])
        if latest["account_id"] is None:
            _cover_gap_on_open_logout(conn, source_id, latest, start, last_observed)
            acknowledge_recorded_login_boundary(conn, source_id, moment)
            return
        start = max(start, last_observed)
    started_at = to_iso(start)
    conn.execute(
        "INSERT INTO active_account_log (source_id, account_id, started_at, last_observed_at) VALUES (?, ?, ?, ?)",
        (source_id, None, started_at, started_at),
    )
    acknowledge_recorded_login_boundary(conn, source_id, moment)


def _cover_gap_on_open_logout(
    conn: sqlite3.Connection,
    source_id: str,
    latest: sqlite3.Row,
    start: datetime,
    last_observed: datetime,
) -> None:
    conn.execute(
        "UPDATE active_account_log SET last_observed_at = ? WHERE id = ?",
        (to_iso(max(start, last_observed)), latest["id"]),
    )
    if start >= parse_iso(latest["started_at"]):
        return
    previous = _interval_before(conn, source_id, latest["id"])
    if previous is not None and start <= parse_iso(previous["started_at"]):
        return
    conn.execute(
        "UPDATE active_account_log SET started_at = ? WHERE id = ?",
        (to_iso(start), latest["id"]),
    )


def _interval_before(conn: sqlite3.Connection, source_id: str, interval_id: int) -> sqlite3.Row | None:
    row: sqlite3.Row | None = conn.execute(
        "SELECT started_at FROM active_account_log WHERE source_id = ? AND id < ? ORDER BY id DESC LIMIT 1",
        (source_id, interval_id),
    ).fetchone()
    return row


def login_boundary_was_recorded(conn: sqlite3.Connection, source_id: str, moment: datetime) -> bool:
    """Report whether a committed login already applied this exact boundary."""
    row = conn.execute(
        "SELECT 1 FROM recorded_login_boundaries WHERE source_id = ? AND boundary_at = ?",
        (source_id, in_utc(moment).isoformat()),
    ).fetchone()
    return row is not None


def active_timeline(conn: sqlite3.Connection, source_id: str) -> list[tuple[datetime, int | None]]:
    """Return ``(started_at, account_id)`` intervals for a source, oldest first; None means nobody was logged in."""
    rows = conn.execute(
        "SELECT started_at, account_id FROM active_account_log WHERE source_id = ? ORDER BY id",
        (source_id,),
    )
    return [(parse_iso(row["started_at"]), row["account_id"]) for row in rows]


def set_source_state(
    conn: sqlite3.Connection,
    source_id: str,
    provider: Provider,
    position: int,
    current_account_id: int | None,
    now: datetime,
) -> None:
    conn.execute(
        "INSERT INTO source_state (source_id, provider, position, current_account_id, checked_at) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT (source_id) DO UPDATE SET provider = excluded.provider, "
        "position = excluded.position, current_account_id = excluded.current_account_id, "
        "checked_at = excluded.checked_at",
        (source_id, provider.value, position, current_account_id, to_iso(now)),
    )


def list_source_states(conn: sqlite3.Connection) -> list[SourceStateRow]:
    rows = conn.execute("SELECT * FROM source_state ORDER BY provider, position")
    return [
        SourceStateRow(
            source_id=row["source_id"],
            provider=Provider(row["provider"]),
            position=row["position"],
            current_account_id=row["current_account_id"],
            checked_at=parse_iso(row["checked_at"]),
        )
        for row in rows
    ]


def retain_sources(conn: sqlite3.Connection, source_ids: Sequence[str]) -> None:
    """Drop state and status for sources no longer configured, so removed sources stop showing as broken."""
    placeholders = ",".join("?" for _ in source_ids)
    condition = f"source_id NOT IN ({placeholders})" if source_ids else "1 = 1"
    conn.execute(f"DELETE FROM source_state WHERE {condition}", tuple(source_ids))  # noqa: S608 - placeholders only
    conn.execute(f"DELETE FROM source_status WHERE {condition}", tuple(source_ids))  # noqa: S608 - placeholders only


def clear_status(conn: sqlite3.Connection, source_id: str, kind: str) -> None:
    """Forget one kind of status for a source that is still configured but no longer does that kind of work."""
    conn.execute("DELETE FROM source_status WHERE source_id = ? AND kind = ?", (source_id, kind))


def record_status(
    conn: sqlite3.Connection, source_id: str, kind: str, now: datetime, error: BaseException | None
) -> None:
    """Record the outcome of one collection attempt. A success clears the previous error."""
    stamp = to_iso(now)
    if error is None:
        conn.execute(
            "INSERT INTO source_status (source_id, kind, last_attempt_at, last_success_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (source_id, kind) DO UPDATE SET last_attempt_at = excluded.last_attempt_at, "
            "last_success_at = excluded.last_success_at, error_type = NULL, error_message = NULL",
            (source_id, kind, stamp, stamp),
        )
        return
    conn.execute(
        "INSERT INTO source_status (source_id, kind, last_attempt_at, error_type, error_message) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT (source_id, kind) DO UPDATE SET "
        "last_attempt_at = excluded.last_attempt_at, error_type = excluded.error_type, "
        "error_message = excluded.error_message",
        (source_id, kind, stamp, type(error).__name__, str(error)),
    )


def list_statuses(conn: sqlite3.Connection) -> list[SourceStatusRow]:
    rows = conn.execute("SELECT * FROM source_status ORDER BY source_id, kind")
    return [
        SourceStatusRow(
            source_id=row["source_id"],
            kind=row["kind"],
            last_attempt_at=parse_iso(row["last_attempt_at"]),
            last_success_at=_optional_time(row["last_success_at"]),
            error_type=row["error_type"],
            error_message=row["error_message"],
        )
        for row in rows
    ]


def load_cursors(conn: sqlite3.Connection, provider: Provider) -> dict[str, FileCursor]:
    rows = conn.execute(
        "SELECT file_key, byte_offset, size, mtime_ns, state_json FROM scan_cursors WHERE provider = ?",
        (provider.value,),
    )
    return {
        row["file_key"]: FileCursor(
            byte_offset=row["byte_offset"],
            size=row["size"],
            mtime_ns=row["mtime_ns"],
            state=_saved_state(row["state_json"]),
        )
        for row in rows
    }


def _saved_state(text: str) -> object:
    """Decode one cursor's saved state, or return None when the text is not JSON.

    Why this does not raise: cursors are loaded for a whole provider at once, so one damaged row would
    stop every file of that provider from being read. With None, only that file loses its saved state.
    """
    try:
        return decode_json(text)
    except (ValueError, RecursionError):
        return None


def stored_requests(conn: sqlite3.Connection, provider: Provider, file_key: str) -> int:
    """Return how many requests are stored for one session file.

    The rows are added here and not by SQLite. A sum of stored rows can pass 64 bits, and SQLite's
    ``SUM`` then fails.
    """
    rows = conn.execute(
        "SELECT requests FROM token_usage WHERE provider = ? AND file_key = ?",
        (provider.value, file_key),
    )
    total = 0
    for row in rows:
        total += row["requests"]
    return total


def stored_requests_kept(conn: sqlite3.Connection, provider: Provider, file_key: str, first_day: str) -> int:
    """Return stored requests whose hour bucket is on or after ``first_day``.

    The rows are added in Python. A sum of stored rows can pass 64 bits, and the SQLite SUM aggregate then fails.
    """
    rows = conn.execute(
        "SELECT bucket_hour, requests FROM token_usage WHERE provider = ? AND file_key = ?",
        (provider.value, file_key),
    )
    total = 0
    for row in rows:
        if row["bucket_hour"] >= first_day:
            total += row["requests"]
    return total


def save_cursor(conn: sqlite3.Connection, provider: Provider, file_key: str, cursor: FileCursor) -> None:
    """Store how far a session file has been read.

    Raises:
        ValueError: if the cursor's state is not JSON (``encode_json``); nothing is stored.
    """
    conn.execute(
        "INSERT INTO scan_cursors (provider, file_key, byte_offset, size, mtime_ns, state_json) "
        "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (provider, file_key) DO UPDATE SET "
        "byte_offset = excluded.byte_offset, size = excluded.size, mtime_ns = excluded.mtime_ns, "
        "state_json = excluded.state_json",
        (provider.value, file_key, cursor.byte_offset, cursor.size, cursor.mtime_ns, encode_json(cursor.state)),
    )


def _storable_count(count: int) -> int:
    """Return a count one stored row can hold.

    Raises:
        ValueError: if the count is negative or above ``MAX_TOKEN_COUNT``. Checked before the value
            reaches SQLite, which cannot take an integer wider than 64 bits at all.
    """
    if not 0 <= count <= MAX_TOKEN_COUNT:
        raise ValueError("a token count is outside what one stored row can hold")
    return count


def _usage_parameters(
    provider: Provider, file_key: str, usage: Mapping[UsageKey, TokenCounts]
) -> list[tuple[str, str, int, str, str, int, int, int, int, int, int]]:
    return [
        (
            provider.value,
            file_key,
            key.account_id,
            key.bucket_hour,
            key.model,
            _storable_count(counts.input_tokens),
            _storable_count(counts.cache_read_tokens),
            _storable_count(counts.cache_write_tokens),
            _storable_count(counts.output_tokens),
            _storable_count(counts.reasoning_tokens),
            _storable_count(counts.requests),
        )
        for key, counts in usage.items()
        if not counts.is_empty
    ]


def add_file_usage(
    conn: sqlite3.Connection, provider: Provider, file_key: str, usage: Mapping[UsageKey, TokenCounts]
) -> None:
    """Add newly read usage to what the file already contributed (for append-only files read by offset).

    Claude usage is refused: it is counted per response, and only ``add_message_usage`` keeps the keys
    that guard it. Written here, under any file key, a response read again through those keys was
    counted twice.

    Raises:
        ValueError: if the usage is Claude's, or a count, or the row total it would bring a row to, is
            outside what one stored row can hold. The caller's transaction must be rolled back, as
            ``Database.connect`` does when this propagates: rows written before the refused one are
            still part of it.
    """
    if provider is Provider.CLAUDE:
        raise ValueError("Claude usage is counted per response and cannot be added by file")
    _add_usage(conn, provider, file_key, usage)


def _add_usage(
    conn: sqlite3.Connection, provider: Provider, file_key: str, usage: Mapping[UsageKey, TokenCounts]
) -> None:
    """Add usage to a file's rows, for ``add_file_usage`` and, for Claude, ``add_message_usage`` alone.

    Raises:
        ValueError: as ``add_file_usage`` does for a count or row total past the limit.
    """
    parameters = _usage_parameters(provider, file_key, usage)
    try:
        _add_usage_rows(conn, parameters)
    except sqlite3.IntegrityError as error:
        # Why this is the only cause: the table has no foreign key, every value bound is a non-null
        # integer or text, and a repeated key is an update. What is left is a row total past the limit.
        raise ValueError("a file's token usage is more than one stored row can hold") from error


def _add_usage_rows(
    conn: sqlite3.Connection, parameters: list[tuple[str, str, int, str, str, int, int, int, int, int, int]]
) -> None:
    conn.executemany(
        "INSERT INTO token_usage (provider, file_key, account_id, bucket_hour, model, input_tokens, "
        "cache_read_tokens, cache_write_tokens, output_tokens, reasoning_tokens, requests) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (provider, file_key, account_id, bucket_hour, model) DO UPDATE SET "
        "input_tokens = input_tokens + excluded.input_tokens, "
        "cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens, "
        "cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens, "
        "output_tokens = output_tokens + excluded.output_tokens, "
        "reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens, "
        "requests = requests + excluded.requests",
        parameters,
    )


def replace_file_usage(
    conn: sqlite3.Connection, provider: Provider, file_key: str, usage: Mapping[UsageKey, TokenCounts]
) -> None:
    """Replace everything the file contributed (for files re-read from the start).

    Raises:
        ValueError: if ``provider`` is Claude (see ``clear_file_usage``), or a count is outside what one
            stored row can hold.
    """
    clear_file_usage(conn, provider, file_key)
    add_file_usage(conn, provider, file_key, usage)


def clear_file_usage(conn: sqlite3.Connection, provider: Provider, file_key: str) -> None:
    """Delete everything the file contributed.

    Raises:
        ValueError: if ``provider`` is Claude. Claude usage is counted per response, and the key that
            stops a response being counted again is not stored per file. Deleting rows would leave
            the keys of their responses behind, and reading the transcripts again would then add
            nothing back, so the usage would be lost. Claude usage leaves the store only through
            ``prune``, which removes usage and keys together.
    """
    if provider is Provider.CLAUDE:
        raise ValueError("Claude usage is counted per response and cannot be cleared or replaced by file")
    conn.execute("DELETE FROM token_usage WHERE provider = ? AND file_key = ?", (provider.value, file_key))


def seen_messages(conn: sqlite3.Connection, keys: Sequence[bytes]) -> dict[bytes, SeenMessage]:
    """Return what has already been counted, and where, for each previously seen Claude message key."""
    return {key: SeenMessage(counted.place, counted.counts) for key, counted in _counted_messages(conn, keys).items()}


def _counted_messages(conn: sqlite3.Connection, keys: Sequence[bytes]) -> dict[bytes, _CountedMessage]:
    found: dict[bytes, _CountedMessage] = {}
    for start in range(0, len(keys), SQLITE_VARIABLE_CHUNK):
        chunk = keys[start : start + SQLITE_VARIABLE_CHUNK]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            "SELECT key, first_line_at, account_id, bucket_hour, model, input_tokens, cache_read_tokens, "  # noqa: S608 - placeholders only
            "cache_write_tokens, output_tokens, reasoning_tokens FROM claude_seen_messages "
            f"WHERE key IN ({placeholders})",
            tuple(chunk),
        )
        for row in rows:
            found[bytes(row["key"])] = _CountedMessage(
                place=UsageKey(row["account_id"], row["bucket_hour"], row["model"]),
                first_line_at=row["first_line_at"],
                counts=TokenCounts(
                    input_tokens=row["input_tokens"],
                    cache_read_tokens=row["cache_read_tokens"],
                    cache_write_tokens=row["cache_write_tokens"],
                    output_tokens=row["output_tokens"],
                    reasoning_tokens=row["reasoning_tokens"],
                ),
            )
    return found


def add_message_usage(conn: sqlite3.Connection, messages: Iterable[MessageUsage]) -> None:
    """Count what each line of a Claude response adds beyond what is already counted for that response.

    A response is counted once, as one request, at the largest value any of its lines gives for each
    count, under the account, hour, and model of its earliest line. That is so whichever of its lines
    is read first, from whichever transcript, and however the lines are split between calls:

    - a response not seen before is counted under the line's own account, hour, and model;
    - a line of a response seen before adds only the amount by which each count exceeds the largest
      already counted, so a repeat adds nothing;
    - a line earlier than any read before moves everything counted for the response to that line's
      account, hour, and model. At the same instant a known account wins over nobody; model name and
      account id then settle ties, so no order of reading can decide between them;
    - a response whose usage has expired (``prune``) is never counted again, whichever of its lines a
      later scan reads: counted from a line still inside retention, it would come back under that
      line's hour.

    Claude usage is kept in one set of rows (``CLAUDE_USAGE_FILE_KEY``) and not per transcript, which
    is what lets a response's usage be moved whole. A key is stored under the same hour as all the
    usage it guards and moves with it, and ``prune`` expires both by that hour, so neither outlives
    the other.

    The lines are first combined into what is counted for each response afterwards. That is compared
    with what was stored for it before the call, every row the call takes from or adds to is checked
    to hold exactly what all the keys stored in it say (``_verify_claude_usage``), and only then is
    usage taken out of the rows it leaves and added to the rows it belongs in.

    Raises:
        ValueError: if a line's time is naive, or a count, or the row total it would bring a row to, is
            outside what one stored row can hold. The caller's transaction must be rolled back, as
            ``Database.connect`` does when this propagates: rows written before the refused one are
            still part of it.
        StoreError: if a row this call would change does not hold what was counted in it.
    """
    pending = list(messages)
    stored = _counted_messages(conn, list(dict.fromkeys(message.key for message in pending)))
    expired = _expired_messages(conn, list(dict.fromkeys(message.key for message in pending)))
    pending = [message for message in pending if message.key not in expired]
    counted = dict(stored)
    for message in pending:
        line = _CountedMessage(
            place=UsageKey(message.account_id, hour_bucket(message.occurred_at), message.model),
            first_line_at=epoch_microseconds(message.occurred_at),
            counts=replace(message.counts, requests=0),
        )
        earlier = counted.get(message.key)
        counted[message.key] = line if earlier is None else _with_line(earlier, line)

    touched: set[UsageKey] = set()
    taken: dict[UsageKey, TokenCounts] = {}
    added: dict[UsageKey, TokenCounts] = {}
    for key, now_counted in counted.items():
        before = stored.get(key)
        touched.add(now_counted.place)
        if before is None:
            _accumulate(added, now_counted.place, now_counted.counts.plus(_ONE_REQUEST))
            continue
        touched.add(before.place)
        if before.place == now_counted.place:
            _accumulate(added, before.place, _surplus(now_counted.counts, before.counts))
        else:
            _accumulate(taken, before.place, before.counts.plus(_ONE_REQUEST))
            _accumulate(added, now_counted.place, now_counted.counts.plus(_ONE_REQUEST))
    _verify_claude_usage(conn, touched)
    _take_claude_usage(conn, taken)
    _add_usage(conn, Provider.CLAUDE, CLAUDE_USAGE_FILE_KEY, added)
    _remember_messages(conn, counted.items())


def _expired_messages(conn: sqlite3.Connection, keys: Sequence[bytes]) -> set[bytes]:
    """Return which of ``keys`` belong to responses whose usage has expired."""
    expired: set[bytes] = set()
    for start in range(0, len(keys), SQLITE_VARIABLE_CHUNK):
        chunk = keys[start : start + SQLITE_VARIABLE_CHUNK]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT key FROM claude_expired_messages WHERE key IN ({placeholders})",  # noqa: S608 - placeholders only
            tuple(chunk),
        )
        expired.update(bytes(row["key"]) for row in rows)
    return expired


def _with_line(counted: _CountedMessage, line: _CountedMessage) -> _CountedMessage:
    """Return what is counted for a response once another of its lines is taken into account."""
    placed_by = line if line.order < counted.order else counted
    return _CountedMessage(placed_by.place, placed_by.first_line_at, _combined(counted.counts, line.counts, max))


def _accumulate(totals: dict[UsageKey, TokenCounts], place: UsageKey, counts: TokenCounts) -> None:
    totals[place] = totals.get(place, TokenCounts()).plus(counts)


def _combined(first: TokenCounts, second: TokenCounts, pick: Callable[[int, int], int]) -> TokenCounts:
    """Build counts by applying ``pick`` to each pair of corresponding counts."""
    return TokenCounts(**{field: pick(getattr(first, field), getattr(second, field)) for field in _COUNT_FIELDS})


def _surplus(counts: TokenCounts, other: TokenCounts) -> TokenCounts:
    """Return, for every count, the amount by which it exceeds the other's, or zero."""
    return _combined(counts, other, lambda mine, theirs: max(0, mine - theirs))


def _verify_claude_usage(conn: sqlite3.Connection, places: Iterable[UsageKey]) -> None:
    """Check that each row a call will change holds exactly what every key stored in it says was counted.

    ``places`` are the account, hour, and model of every row the call takes usage from or adds it to,
    a row that does not exist yet included. Everything counted for a response is in the row its key
    names and nothing else is, so a row holds the sum of all the keys stored in it, one request each:
    any difference means the rows were changed from outside the store. All the keys of a row are
    summed, not only those of the responses the call touches: a row short of another response's usage
    would otherwise pass, and moving or adding usage would leave that response without its own. The
    check runs before the call changes anything, so nothing the call adds can stand in for usage that
    is missing. The keys are added up here, not with SQLite's ``SUM``, which fails past 64 bits.

    Raises:
        StoreError: if a row does not hold what its keys say was counted in it.
    """
    for place in places:
        recorded = TokenCounts()
        for key_row in conn.execute(
            "SELECT input_tokens, cache_read_tokens, cache_write_tokens, output_tokens, reasoning_tokens "
            "FROM claude_seen_messages WHERE bucket_hour = ? AND account_id = ? AND model = ?",
            (place.bucket_hour, place.account_id, place.model),
        ):
            recorded = recorded.plus(_key_counts(key_row)).plus(_ONE_REQUEST)
        row = conn.execute(
            "SELECT input_tokens, cache_read_tokens, cache_write_tokens, output_tokens, reasoning_tokens, requests "
            "FROM token_usage WHERE provider = ? AND file_key = ? AND account_id = ? AND bucket_hour = ? AND model = ?",
            (Provider.CLAUDE.value, CLAUDE_USAGE_FILE_KEY, place.account_id, place.bucket_hour, place.model),
        ).fetchone()
        if (TokenCounts() if row is None else _counts_from_row(row)) != recorded:
            raise StoreError("the Claude usage rows do not hold what was counted for a response")


def _key_counts(row: sqlite3.Row) -> TokenCounts:
    return TokenCounts(
        input_tokens=row["input_tokens"],
        cache_read_tokens=row["cache_read_tokens"],
        cache_write_tokens=row["cache_write_tokens"],
        output_tokens=row["output_tokens"],
        reasoning_tokens=row["reasoning_tokens"],
    )


def _take_claude_usage(conn: sqlite3.Connection, removals: Mapping[UsageKey, TokenCounts]) -> None:
    """Subtract usage from the Claude rows it was counted in, and delete rows left holding nothing.

    The rows have been verified to hold at least what is taken (``_verify_claude_usage``).
    """
    if not removals:
        return
    conn.executemany(
        "UPDATE token_usage SET input_tokens = input_tokens - ?, cache_read_tokens = cache_read_tokens - ?, "
        "cache_write_tokens = cache_write_tokens - ?, output_tokens = output_tokens - ?, "
        "reasoning_tokens = reasoning_tokens - ?, requests = requests - ? "
        "WHERE provider = ? AND file_key = ? AND account_id = ? AND bucket_hour = ? AND model = ?",
        [
            (
                counts.input_tokens,
                counts.cache_read_tokens,
                counts.cache_write_tokens,
                counts.output_tokens,
                counts.reasoning_tokens,
                counts.requests,
                Provider.CLAUDE.value,
                CLAUDE_USAGE_FILE_KEY,
                place.account_id,
                place.bucket_hour,
                place.model,
            )
            for place, counts in removals.items()
        ],
    )
    conn.execute(
        "DELETE FROM token_usage WHERE provider = ? AND file_key = ? AND input_tokens = 0 "
        "AND cache_read_tokens = 0 AND cache_write_tokens = 0 AND output_tokens = 0 "
        "AND reasoning_tokens = 0 AND requests = 0",
        (Provider.CLAUDE.value, CLAUDE_USAGE_FILE_KEY),
    )


def _remember_messages(conn: sqlite3.Connection, messages: Iterable[tuple[bytes, _CountedMessage]]) -> None:
    """Record what has been counted for each Claude message key, where, and by which line.

    Everything replaces what was remembered; ``add_message_usage`` is the only caller and has already
    combined each line with it.

    Raises:
        ValueError: if a count is outside what one stored row can hold.
    """
    conn.executemany(
        "INSERT INTO claude_seen_messages (key, first_line_at, account_id, bucket_hour, model, input_tokens, "
        "cache_read_tokens, cache_write_tokens, output_tokens, reasoning_tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (key) DO UPDATE SET "
        "first_line_at = excluded.first_line_at, "
        "account_id = excluded.account_id, "
        "bucket_hour = excluded.bucket_hour, "
        "model = excluded.model, "
        "input_tokens = excluded.input_tokens, "
        "cache_read_tokens = excluded.cache_read_tokens, "
        "cache_write_tokens = excluded.cache_write_tokens, "
        "output_tokens = excluded.output_tokens, "
        "reasoning_tokens = excluded.reasoning_tokens",
        [
            (
                key,
                counted.first_line_at,
                counted.place.account_id,
                counted.place.bucket_hour,
                counted.place.model,
                _storable_count(counted.counts.input_tokens),
                _storable_count(counted.counts.cache_read_tokens),
                _storable_count(counted.counts.cache_write_tokens),
                _storable_count(counted.counts.output_tokens),
                _storable_count(counted.counts.reasoning_tokens),
            )
            for key, counted in messages
        ],
    )


def _counts_from_row(row: sqlite3.Row) -> TokenCounts:
    return TokenCounts(
        input_tokens=row["input_tokens"],
        cache_read_tokens=row["cache_read_tokens"],
        cache_write_tokens=row["cache_write_tokens"],
        output_tokens=row["output_tokens"],
        reasoning_tokens=row["reasoning_tokens"],
        requests=row["requests"],
    )


def token_usage(conn: sqlite3.Connection, start_bucket: str, end_bucket: str) -> list[TokenUsageRow]:
    """Return usage per provider, account, hour, and model for ``start_bucket <= bucket < end_bucket``.

    This is the only read of token usage, so every total a report shows is a sum over the same rows.

    The files' rows are added up here and not by SQLite. Each stored row is within ``MAX_TOKEN_COUNT``,
    but any number of files can contribute to one hour, and SQLite's ``SUM`` fails once a total passes
    64 bits. Python's integers do not, so whatever the store accepted can always be read back and added.
    """
    rows = conn.execute(
        "SELECT provider, account_id, bucket_hour, model, input_tokens, cache_read_tokens, cache_write_tokens, "
        "output_tokens, reasoning_tokens, requests FROM token_usage "
        "WHERE bucket_hour >= ? AND bucket_hour < ? ORDER BY bucket_hour, provider, account_id, model",
        (start_bucket, end_bucket),
    )
    totals: dict[tuple[str, str, int, str], TokenCounts] = {}
    for row in rows:
        key = (row["bucket_hour"], row["provider"], row["account_id"], row["model"])
        totals[key] = totals.get(key, TokenCounts()).plus(_counts_from_row(row))
    return [
        TokenUsageRow(
            provider=Provider(provider), account_id=account_id, bucket_hour=bucket, model=model, counts=counts
        )
        for (bucket, provider, account_id, model), counts in totals.items()
    ]


def prune(conn: sqlite3.Connection, now: datetime) -> None:
    """Delete history past retention, keeping whatever is still in effect inside it.

    A quota step is deleted once it ended at or before the retention cutoff, which is when the next
    step of its account window was observed. The step in effect at the cutoff is kept however long ago
    it was last confirmed: it is the state the retained history starts in. The newest step of a window
    never ended, so an account's most recent reading is always kept. The one exception is a window
    the account's latest reading no longer reports: its last step goes once its last confirmation is
    past retention. The cutoff and a step's times are compared to the microsecond, so a step replaced
    later in the very second the cutoff falls in was still in effect at the cutoff and is kept.

    Token usage and the keys of seen Claude messages expire on the same day. A key is what stops a
    response being counted again when a resumed session copies it into a new transcript, so a key that
    expired before the usage it guards would let that usage be counted twice. ``add_message_usage``
    stores a key under the same hour as all of its usage, and the same comparison removes both here.
    An expired key is not forgotten: it moves to ``claude_expired_messages``, and ``add_message_usage``
    never counts that response again. Forgotten, a later scan that read only a line of the response
    still inside retention counted it again under that line's hour. A key is about a dozen bytes, and
    on this Mac about 270,000 responses a year were seen (2026-10-04), so kept for good they take a few
    megabytes a year.

    Raises:
        ValueError: if ``now`` is naive.
    """
    # Why converted first: subtracting days from a time in a zone with clock changes moves its
    # wall-clock fields, which is an hour away from the same number of elapsed days.
    now_utc = in_utc(now)
    cutoff = epoch_microseconds(now_utc - SNAPSHOT_RETENTION)
    conn.execute(
        "DELETE FROM quota_snapshots WHERE EXISTS (SELECT 1 FROM quota_snapshots newer "
        "WHERE newer.account_id = quota_snapshots.account_id AND newer.window_key = quota_snapshots.window_key "
        "AND newer.id > quota_snapshots.id AND newer.observed_us <= ?)",
        (cutoff,),
    )
    conn.execute(
        "DELETE FROM quota_snapshots WHERE confirmed_us < ? AND confirmed_seq < "
        "(SELECT reading_seq FROM accounts WHERE accounts.id = quota_snapshots.account_id) "
        "AND NOT EXISTS (SELECT 1 FROM quota_snapshots newer "
        "WHERE newer.account_id = quota_snapshots.account_id AND newer.window_key = quota_snapshots.window_key "
        "AND newer.id > quota_snapshots.id)",
        (cutoff,),
    )
    # Why one comparison serves both: an hour bucket is its day followed by "T" and the hour, so it sorts
    # after the bare day. Every bucket of the first kept day is kept, and every earlier one goes.
    first_kept_day = day_key(now_utc - USAGE_RETENTION)
    conn.execute("DELETE FROM token_usage WHERE bucket_hour < ?", (first_kept_day,))
    conn.execute(
        "INSERT OR IGNORE INTO claude_expired_messages (key) "
        "SELECT key FROM claude_seen_messages WHERE bucket_hour < ?",
        (first_kept_day,),
    )
    conn.execute("DELETE FROM claude_seen_messages WHERE bucket_hour < ?", (first_kept_day,))


__all__ = [
    "CLAUDE_USAGE_FILE_KEY",
    "STATUS_KIND_QUOTA",
    "STATUS_KIND_TOKENS",
    "UNATTRIBUTED_ACCOUNT_ID",
    "AccountRow",
    "Database",
    "FileCursor",
    "MessageUsage",
    "SeenMessage",
    "SnapshotRow",
    "SourceStateRow",
    "SourceStatusRow",
    "TokenUsageRow",
]
