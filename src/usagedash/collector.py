import fcntl
import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import traceback
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from usagedash import db
from usagedash.db import STATUS_KIND_QUOTA, STATUS_KIND_TOKENS, Database
from usagedash.errors import ConfigError, ProviderRequestError, UsageDashError
from usagedash.models import AccountIdentity, QuotaReading
from usagedash.private import PRIVATE_FILE_MODE, ensure_private_directory, open_private_file, restrict_to_owner
from usagedash.providers.base import QuotaSource
from usagedash.scanners.base import ScanReport, TokenScanner, build_account_resolver
from usagedash.timeutil import Clock, in_utc, parse_iso, whole_second_not_before

logger = logging.getLogger(__name__)

RATE_LIMIT_BACKOFF_INTERVALS = 2
MAX_RATE_LIMIT_BACKOFF = timedelta(days=1)
LOCK_FILE_NAME = "collector.lock"
LOGIN_UNRECORDED_FILE_NAME = "login-unrecorded.json"


class CycleAlreadyRunningError(UsageDashError):
    pass


class TokenScanError(UsageDashError):
    pass


class UnexpectedFailureError(UsageDashError):
    pass


def _reportable(error: Exception) -> UsageDashError:
    if isinstance(error, UsageDashError):
        return error
    return UnexpectedFailureError(f"unexpected {type(error).__name__}; the log says where it was raised")


def _source_ref(source_id: str) -> str:
    return hashlib.sha256(source_id.encode()).hexdigest()[:12]


def _redact_paths(text: str) -> str:
    home = str(Path.home())
    if len(home) > 1:
        return text.replace(home, "~")
    return text


def describe_without_messages(error: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        rendered = _redact_paths("".join(traceback.format_tb(current.__traceback__)))
        parts.append(f"{type(current).__name__} raised at:\n{rendered}")
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return "which followed ".join(parts).rstrip()


@dataclass(frozen=True, slots=True)
class SourceBundle:
    source: QuotaSource
    scanner: TokenScanner | None
    position: int


@dataclass(frozen=True, slots=True)
class _LoadedLogins:
    parsed: bool
    unreadable: bool
    logins: dict[str, datetime]


def load_unrecorded_logins(path: Path) -> _LoadedLogins:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _LoadedLogins(False, False, {})
    except (OSError, UnicodeError) as error:
        logger.error(
            "Could not read the unrecorded-login file %s\n%s",
            _redact_paths(str(path)),
            describe_without_messages(error),
        )
        return _LoadedLogins(False, True, {})
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError):
        logger.error("Could not parse the unrecorded-login file %s", _redact_paths(str(path)))
        return _LoadedLogins(True, False, {})
    if not isinstance(parsed, dict):
        logger.error("Could not parse the unrecorded-login file %s", _redact_paths(str(path)))
        return _LoadedLogins(True, False, {})
    logins: dict[str, datetime] = {}
    for source_id, moment in parsed.items():
        if source_id == "" or not isinstance(moment, str):
            continue
        try:
            boundary = parse_iso(moment)
            whole_second_not_before(boundary)
            logins[source_id] = boundary
        except ValueError:
            continue
    return _LoadedLogins(True, False, logins)


def _write_all(descriptor: int, raw: bytes) -> None:
    remaining = memoryview(raw)
    while remaining:
        written = os.write(descriptor, remaining)
        if written == 0:
            raise OSError("could not write the unrecorded login boundary")
        remaining = remaining[written:]


def store_unrecorded_logins(path: Path, pending: Mapping[str, datetime]) -> None:
    if not pending:
        path.unlink(missing_ok=True)
        return
    ensure_private_directory(path.parent)
    raw = json.dumps(
        {source_id: in_utc(moment).isoformat() for source_id, moment in sorted(pending.items())},
        allow_nan=False,
    ).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, PRIVATE_FILE_MODE)
    try:
        try:
            restrict_to_owner(descriptor, PRIVATE_FILE_MODE)
            _write_all(descriptor, raw)
            os.fsync(descriptor)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class SourceOutcome:
    """What one cycle found at one source.

    Each error is given as the text to show and as the name of its type. The type is what classifies
    it, and it is the same name the store records, so whoever started the cycle can judge its outcome
    from the report alone: the stored statuses may already be another cycle's by the time they are read.
    """

    source_id: str
    account_label: str | None
    quota_error: str | None
    token_error: str | None
    files_read: int
    quota_error_type: str | None = None
    token_error_type: str | None = None


@dataclass(frozen=True, slots=True)
class CycleReport:
    started_at: datetime
    finished_at: datetime
    outcomes: tuple[SourceOutcome, ...]


class Collector:
    def __init__(
        self,
        database: Database,
        bundles: Sequence[SourceBundle],
        clock: Clock,
        *,
        poll_interval_seconds: int,
        backfill_days: int,
    ):
        self._database = database
        self._bundles = tuple(bundles)
        self._clock = clock
        self._poll_interval = timedelta(seconds=poll_interval_seconds)
        self._backfill_days = backfill_days
        self._quota_not_before: dict[str, datetime] = {}
        self._login_unrecorded_since: dict[str, datetime] = {}
        self._boundary_blocked: set[str] = set()
        self._boundary_file_unreadable = False
        self._boundary_store_failed = False
        self._cycle_lock = threading.Lock()
        self._lock_path = database.path.with_name(LOCK_FILE_NAME)
        self._last_finished_at: datetime | None = None

    @property
    def is_running(self) -> bool:
        return self._cycle_lock.locked()

    def _now(self) -> datetime:
        """Read the clock in UTC, so that a time computed from a reading is a measure of elapsed time.

        Adding to a datetime in a zone with clock changes moves its wall clock, and two datetimes that
        share a zone are compared by their wall clocks: on the night the clocks change, a rate-limit
        wait came out an hour long or short, and its deadline was shown as UTC when it was not.

        Raises:
            ValueError: if the clock returns a naive datetime.
        """
        return in_utc(self._clock())

    @property
    def last_finished_at(self) -> datetime | None:
        return self._last_finished_at

    def run_cycle(self) -> CycleReport:
        """Raises:
        CycleAlreadyRunningError: if another cycle is in progress."""
        if not self._cycle_lock.acquire(blocking=False):
            raise CycleAlreadyRunningError("a collection cycle is already running")
        try:
            with self._exclusive_across_processes():
                return self._run_locked_cycle()
        finally:
            self._cycle_lock.release()

    @contextmanager
    def _exclusive_across_processes(self) -> Generator[None]:
        """Hold the data directory's collection lock for the duration of a cycle.

        Why: the background agent and a manual ``usagedash collect`` are separate processes sharing one
        database. Each scanner reads its file positions and stores usage in separate transactions, so two
        overlapping cycles would both add the same tokens. The in-process lock cannot see the other process.

        Raises:
            CycleAlreadyRunningError: if another process holds the lock.
            ConfigError: if the data directory cannot be made private or the lock file cannot be opened.
        """
        ensure_private_directory(self._lock_path.parent)
        descriptor = open_private_file(self._lock_path)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise CycleAlreadyRunningError("another process is already collecting into this database") from error
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _run_locked_cycle(self) -> CycleReport:
        started_at = self._now()
        self._boundary_store_failed = False
        self._restore_unrecorded_logins()
        with self._database.connect() as conn:
            db.retain_sources(conn, [bundle.source.source_id for bundle in self._bundles])
        quota_results = [self._collect_quota(bundle) for bundle in self._bundles]
        token_results = [self._collect_tokens(bundle) for bundle in self._bundles]
        with self._database.connect() as conn:
            db.prune(conn, self._now())
        finished_at = self._now()
        self._last_finished_at = finished_at
        outcomes = tuple(
            SourceOutcome(
                source_id=bundle.source.source_id,
                account_label=label,
                quota_error=_shown(quota_error),
                token_error=_shown(token_error),
                files_read=files_read,
                quota_error_type=_type_name(quota_error),
                token_error_type=_type_name(token_error),
            )
            for bundle, (label, quota_error), (token_error, files_read) in zip(
                self._bundles, quota_results, token_results, strict=True
            )
        )
        return CycleReport(started_at=started_at, finished_at=finished_at, outcomes=outcomes)

    def _collect_quota(self, bundle: SourceBundle) -> tuple[str | None, UsageDashError | None]:
        """Return ``(account label, error)`` for one source's identity and quota read."""
        source = bundle.source
        now = self._now()
        try:
            return self._read_and_identify(bundle, now)
        except Exception as error:  # noqa: BLE001
            # Why: one source's failure, including a parse defect, stays on that source.
            return None, self._record_quota_failure(source.source_id, now, error)

    def _record_quota_failure(self, source_id: str, now: datetime, error: Exception) -> UsageDashError:
        if not isinstance(error, UsageDashError):
            logger.error(
                "Unexpected failure collecting quota from %s\n%s",
                _source_ref(source_id),
                describe_without_messages(error),
            )
        reportable = _reportable(error)
        with self._database.connect() as conn:
            if self._boundary_store_failed or source_id in self._boundary_blocked:
                moment = self._login_unrecorded_since.get(source_id)
                if moment is not None:
                    db.record_unrecorded_gap(conn, source_id, moment)
            db.record_status(conn, source_id, STATUS_KIND_QUOTA, now, reportable)
        return reportable

    def _read_and_identify(self, bundle: SourceBundle, now: datetime) -> tuple[str | None, UsageDashError | None]:
        source = bundle.source
        reading, read_error = self._read_quota(source, now)
        identity, identity_error = _who_is_logged_in(source, reading)
        seen_at = self._now()
        account_id = self._record_login(bundle, identity, seen_at)

        with self._database.connect() as conn:
            label = None if account_id is None else db.account_label(conn, account_id)
            if reading is not None and account_id is not None:
                db.record_reading(conn, account_id, reading)
            if read_error is None and identity_error is None:
                db.record_status(conn, source.source_id, STATUS_KIND_QUOTA, now, None)
        if identity_error is not None:
            return label, self._record_quota_failure(source.source_id, now, identity_error)
        if read_error is not None:
            return label, self._record_quota_failure(source.source_id, now, read_error)
        return label, None

    def _record_login(self, bundle: SourceBundle, identity: AccountIdentity | None, seen_at: datetime) -> int | None:
        source = bundle.source
        if source.source_id in self._boundary_blocked:
            self._keep_blocked_gap(source.source_id, identity, seen_at)
            raise UsageDashError("the unrecorded login boundary could not be read")
        try:
            with self._database.connect() as conn:
                self._record_as_nobodys_since_it_could_not_be_recorded(conn, source.source_id)
                account_id = db.upsert_account(conn, identity, seen_at) if identity is not None else None
                db.set_source_state(conn, source.source_id, source.provider, bundle.position, account_id, seen_at)
                db.observe_active_account(conn, source.source_id, account_id, seen_at)
                if account_id is not None:
                    db.mark_account_live(conn, account_id, seen_at)
        except Exception:
            self._record_login_as_nobodys(bundle, seen_at)
            raise
        self._login_unrecorded_since.pop(source.source_id, None)
        self._persist_unrecorded_logins()
        return account_id

    def _keep_blocked_gap(self, source_id: str, identity: AccountIdentity | None, seen_at: datetime) -> None:
        """Remember the first blocked switch, so the account already logged in does not keep the span."""
        if source_id in self._login_unrecorded_since or not self._open_login_would_keep_the_span(source_id, identity):
            return
        self._login_unrecorded_since[source_id] = seen_at

    def _open_login_would_keep_the_span(self, source_id: str, identity: AccountIdentity | None) -> bool:
        try:
            with self._database.connect() as conn:
                timeline = db.active_timeline(conn, source_id)
                if not timeline or timeline[-1][1] is None:
                    return False
                if identity is None:
                    return True
                return db.find_account_id(conn, identity.provider, identity.external_id) != timeline[-1][1]
        except sqlite3.Error:
            return True

    def _record_login_as_nobodys(self, bundle: SourceBundle, seen_at: datetime) -> None:
        source = bundle.source
        self._login_unrecorded_since.setdefault(source.source_id, seen_at)
        self._persist_unrecorded_logins()
        try:
            with self._database.connect() as conn:
                self._record_as_nobodys_since_it_could_not_be_recorded(conn, source.source_id)
                db.set_source_state(conn, source.source_id, source.provider, bundle.position, None, seen_at)
                db.observe_active_account(conn, source.source_id, None, seen_at)
        except Exception as error:  # noqa: BLE001
            logger.error(
                "Could not record that the login at %s is unknown; its tokens are not scanned until it can be\n%s",
                _source_ref(source.source_id),
                describe_without_messages(error),
            )
            return
        del self._login_unrecorded_since[source.source_id]
        self._persist_unrecorded_logins()

    @property
    def _unrecorded_path(self) -> Path:
        return self._database.path.with_name(LOGIN_UNRECORDED_FILE_NAME)

    def _restore_unrecorded_logins(self) -> None:
        self._boundary_blocked = set()
        self._boundary_file_unreadable = False
        loaded = load_unrecorded_logins(self._unrecorded_path)
        self._boundary_file_unreadable = loaded.unreadable
        self._drop_boundaries_the_timeline_already_has()
        if loaded.unreadable:
            self._boundary_blocked = {bundle.source.source_id for bundle in self._bundles}
            return
        if not loaded.parsed:
            return
        changed = not loaded.logins
        for source_id, moment in loaded.logins.items():
            if self._gap_is_recorded(source_id, moment):
                current = self._login_unrecorded_since.get(source_id)
                if current is None or self._gap_is_recorded(source_id, current):
                    self._login_unrecorded_since.pop(source_id, None)
                changed = True
                continue
            current = self._login_unrecorded_since.get(source_id)
            if current is None or moment < current:
                self._login_unrecorded_since[source_id] = moment
                changed = True
        if changed:
            self._persist_unrecorded_logins()

    def _drop_boundaries_the_timeline_already_has(self) -> None:
        for source_id, moment in tuple(self._login_unrecorded_since.items()):
            if self._gap_is_recorded(source_id, moment):
                self._login_unrecorded_since.pop(source_id, None)

    def _gap_is_recorded(self, source_id: str, moment: datetime) -> bool:
        start = whole_second_not_before(moment)
        try:
            with self._database.connect() as conn:
                if db.login_boundary_was_recorded(conn, source_id, moment):
                    return True
                timeline = db.active_timeline(conn, source_id)
        except sqlite3.OperationalError as error:
            logger.error(
                "Could not read the login timeline for %s while restoring an unrecorded boundary\n%s",
                _source_ref(source_id),
                describe_without_messages(error),
            )
            return False
        for started_at, account_id in reversed(timeline):
            if started_at <= start:
                return account_id is None
        return False

    def _persist_unrecorded_logins(self) -> None:
        if self._boundary_file_unreadable:
            return
        try:
            store_unrecorded_logins(self._unrecorded_path, self._login_unrecorded_since)
        except (ConfigError, OSError) as error:
            self._boundary_store_failed = True
            logger.error(
                "Could not store the unrecorded login boundary\n%s",
                describe_without_messages(error),
            )
        else:
            self._boundary_store_failed = False

    def _record_as_nobodys_since_it_could_not_be_recorded(self, conn: sqlite3.Connection, source_id: str) -> None:
        unrecorded_since = self._login_unrecorded_since.get(source_id)
        if unrecorded_since is not None:
            db.record_unrecorded_gap(conn, source_id, unrecorded_since)

    def _read_quota(self, source: QuotaSource, now: datetime) -> tuple[QuotaReading | None, UsageDashError | None]:
        not_before = self._quota_not_before.get(source.source_id)
        if not_before is not None and now < not_before:
            return None, ProviderRequestError(f"backing off after a rate-limit response until {not_before:%H:%M} UTC")
        try:
            reading = source.read_quota(now)
        except ProviderRequestError as error:
            if error.status_code == 429:
                # Why the clock, not the attempt's start: the wait begins when the provider answers.
                self._quota_not_before[source.source_id] = self._now() + self._backoff(error.retry_after_seconds)
            return None, error
        except UsageDashError as error:
            return None, error
        except Exception as error:  # noqa: BLE001
            # Why: a read defect stays on this source and still reaches the identity check.
            logger.error(
                "Unexpected failure reading quota from %s\n%s",
                _source_ref(source.source_id),
                describe_without_messages(error),
            )
            return None, _reportable(error)
        self._quota_not_before.pop(source.source_id, None)
        return reading, None

    def _backoff(self, retry_after_seconds: float | None) -> timedelta:
        minimum = min(self._poll_interval * RATE_LIMIT_BACKOFF_INTERVALS, MAX_RATE_LIMIT_BACKOFF)
        if retry_after_seconds is None or not math.isfinite(retry_after_seconds):
            return minimum
        requested_seconds = min(max(retry_after_seconds, 0.0), MAX_RATE_LIMIT_BACKOFF.total_seconds())
        return max(minimum, timedelta(seconds=requested_seconds))

    def _collect_tokens(self, bundle: SourceBundle) -> tuple[UsageDashError | None, int]:
        """Return ``(error, files read)`` for one source's token scan."""
        source_id = bundle.source.source_id
        if bundle.scanner is None:
            # Why: switching to CLI mode stops scanning, so an earlier token error is cleared.
            with self._database.connect() as conn:
                db.clear_status(conn, source_id, STATUS_KIND_TOKENS)
            return None, 0
        now = self._now()
        if source_id in self._login_unrecorded_since or source_id in self._boundary_blocked:
            # Why: an unrecorded login would be credited to the previous account and then skipped.
            held_back = TokenScanError("not scanned: who is logged in there could not be recorded")
            with self._database.connect() as conn:
                db.record_status(conn, source_id, STATUS_KIND_TOKENS, now, held_back)
            return held_back, 0
        report = ScanReport()
        error: UsageDashError | None = None
        try:
            with self._database.connect() as conn:
                resolver = build_account_resolver(db.active_timeline(conn, source_id))
            report = bundle.scanner.scan(self._database, resolver, now, self._backfill_days)
            if report.files_failed > 0:
                error = TokenScanError(
                    f"{report.files_failed} session file(s) unreadable; first: {report.first_failure}"
                )
            if report.malformed_records > 0:
                logger.warning(
                    "%s: skipped %d malformed usage record(s)", _source_ref(source_id), report.malformed_records
                )
        except Exception as caught:  # noqa: BLE001
            # Why: same isolation boundary as quota collection; a scanner defect must not stop other sources.
            if not isinstance(caught, UsageDashError):
                logger.error(
                    "Unexpected failure scanning tokens for %s\n%s",
                    _source_ref(source_id),
                    describe_without_messages(caught),
                )
            error = _reportable(caught)
        with self._database.connect() as conn:
            db.record_status(conn, source_id, STATUS_KIND_TOKENS, now, error)
        return error, report.files_read


def _shown(error: UsageDashError | None) -> str | None:
    """Return the text a report shows for an error: its type's name and its message."""
    return None if error is None else f"{type(error).__name__}: {error}"


def _type_name(error: UsageDashError | None) -> str | None:
    return None if error is None else type(error).__name__


def _who_is_logged_in(
    source: QuotaSource, reading: QuotaReading | None
) -> tuple[AccountIdentity | None, Exception | None]:
    if reading is not None:
        return reading.identity, None
    try:
        return source.current_identity(), None
    except Exception as error:  # noqa: BLE001
        return None, error


__all__ = [
    "Collector",
    "CycleAlreadyRunningError",
    "CycleReport",
    "SourceBundle",
    "SourceOutcome",
    "TokenScanError",
    "UnexpectedFailureError",
    "describe_without_messages",
]
