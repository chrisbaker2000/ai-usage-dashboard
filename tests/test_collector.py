import json
import logging
import os
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx2
import pytest

from tests.factories import make_identity, make_reading, make_window
from tests.payloads import grok_auth_file
from usagedash import db
from usagedash.collector import (
    LOGIN_UNRECORDED_FILE_NAME,
    Collector,
    CycleAlreadyRunningError,
    SourceBundle,
    _redact_paths,
    _source_ref,
    describe_without_messages,
    load_unrecorded_logins,
    store_unrecorded_logins,
)
from usagedash.config import ClaudeSourceConfig, SourceMode
from usagedash.db import Database, MessageUsage
from usagedash.errors import (
    ConfigError,
    CredentialExpiredError,
    CredentialUnavailableError,
    ProviderRequestError,
    ProviderSchemaError,
    UsageDashError,
)
from usagedash.models import AccountIdentity, Provider, QuotaReading, TokenCounts
from usagedash.providers.claude import ClaudeSource
from usagedash.providers.grok_cli import GrokCliSource
from usagedash.scanners.base import AccountResolver, ScanReport
from usagedash.timeutil import in_utc, parse_iso, whole_second_not_before

POLL_SECONDS = 300
FAKE_CLI = [sys.executable, str(Path(__file__).parent / "fakes" / "fake_cli.py")]


class FakeClock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


class FakeSource:
    def __init__(self, source_id: str, provider: Provider = Provider.CLAUDE):
        self._source_id = source_id
        self._provider = provider
        self.identity: AccountIdentity | None = None
        self.identity_error: Exception | None = None
        self.result: QuotaReading | Exception | None = None
        self.read_calls: list[datetime] = []

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def provider(self) -> Provider:
        return self._provider

    def current_identity(self) -> AccountIdentity | None:
        if self.identity_error is not None:
            raise self.identity_error
        return self.identity

    def read_quota(self, now: datetime) -> QuotaReading:
        self.read_calls.append(now)
        if isinstance(self.result, Exception):
            raise self.result
        if self.result is None:
            raise UsageDashError("no result scripted")
        return self.result


class LearningSource(FakeSource):
    def __init__(self, source_id: str, provider: Provider = Provider.CODEX):
        super().__init__(source_id, provider)
        self.learns: list[AccountIdentity | None] = []

    def read_quota(self, now: datetime) -> QuotaReading:
        self.identity = self.learns.pop(0)
        return super().read_quota(now)


class FakeScanner:
    def __init__(self, report: ScanReport | Exception | None = None):
        self.report = ScanReport(files_read=2) if report is None else report
        self.resolved: list[int] = []
        self.probe_times: list[datetime] = []
        self.scans = 0

    @property
    def provider(self) -> Provider:
        return Provider.CLAUDE

    def scan(
        self, database: Database, resolve_account: AccountResolver, now: datetime, backfill_days: int
    ) -> ScanReport:
        self.resolved = [resolve_account(moment) for moment in self.probe_times]
        if isinstance(self.report, Exception):
            raise self.report
        self.scans += 1
        response = MessageUsage(
            f"response-{self.scans}".encode(), now, resolve_account(now), "m", TokenCounts(input_tokens=1)
        )
        with database.connect() as conn:
            db.add_message_usage(conn, [response])
        return self.report


def collector(database: Database, clock: FakeClock, *bundles: SourceBundle) -> Collector:
    return Collector(database, bundles, clock, poll_interval_seconds=POLL_SECONDS, backfill_days=35)


def statuses(database: Database) -> dict[tuple[str, str], tuple[str | None, str | None]]:
    with database.connect() as conn:
        return {(row.source_id, row.kind): (row.error_type, row.error_message) for row in db.list_statuses(conn)}


class TestSuccessfulCycle:
    def test_registers_the_account_stores_its_reading_and_marks_it_logged_in(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.identity = make_identity(label="Personal")
        source.result = make_reading(source.identity, now, make_window(used_percent=42))
        scanner = FakeScanner()

        report = collector(database, clock, SourceBundle(source, scanner, 0)).run_cycle()

        with database.connect() as conn:
            accounts = db.list_accounts(conn)
            latest = db.latest_snapshots(conn)
            states = db.list_source_states(conn)
            timeline = db.active_timeline(conn, "claude:home")
        assert [(a.label, a.last_live_at) for a in accounts] == [("Personal", now)]
        assert [row.used_percent for row in latest] == [42]
        assert [(state.current_account_id, state.position) for state in states] == [(accounts[0].id, 0)]
        assert timeline == [(now, accounts[0].id)]
        assert statuses(database) == {("claude:home", "quota"): (None, None), ("claude:home", "tokens"): (None, None)}
        assert report.outcomes[0].account_label == "Personal"
        assert (report.outcomes[0].quota_error, report.outcomes[0].token_error, report.outcomes[0].files_read) == (
            None,
            None,
            2,
        )

    def test_token_usage_is_attributed_through_the_timeline_recorded_in_the_same_cycle(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.identity = make_identity()
        source.result = make_reading(source.identity, now, make_window())
        scanner = FakeScanner()
        scanner.probe_times = [now - timedelta(seconds=1), now]

        collector(database, clock, SourceBundle(source, scanner, 0)).run_cycle()

        with database.connect() as conn:
            account_id = db.list_accounts(conn)[0].id
        assert scanner.resolved == [0, account_id]

    def test_each_cycle_reads_the_source_once_at_the_cycle_s_time(self, database: Database, now: datetime) -> None:
        clock = FakeClock(now)
        source = FakeSource("grok:home", Provider.GROK)
        source.identity = make_identity(Provider.GROK)
        source.result = make_reading(source.identity, now, make_window())
        runner = collector(database, clock, SourceBundle(source, None, 0))

        runner.run_cycle()
        clock.advance(minutes=5)
        runner.run_cycle()

        assert source.read_calls == [now, now + timedelta(minutes=5)]

    def test_the_identity_in_the_reading_outranks_the_locally_read_one(self, database: Database, now: datetime) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.identity = make_identity(external_id="stale-local", label="Stale")
        source.result = make_reading(make_identity(external_id="actual", label="Actual"), now, make_window())

        collector(database, clock, SourceBundle(source, None, 0)).run_cycle()

        with database.connect() as conn:
            by_external = {account.external_id: account.id for account in db.list_accounts(conn)}
            state = db.list_source_states(conn)[0]
            snapshot_owner = db.latest_snapshots(conn)[0].account_id
        assert state.current_account_id == by_external["actual"]
        assert snapshot_owner == by_external["actual"]

    def test_a_standing_login_without_a_scanner_records_no_token_status(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:standing")
        source.result = make_reading(make_identity(), now, make_window())

        report = collector(database, clock, SourceBundle(source, None, 1)).run_cycle()

        assert list(statuses(database)) == [("claude:standing", "quota")]
        assert (report.outcomes[0].token_error, report.outcomes[0].files_read) == (None, 0)


class TestFailureIsolation:
    def test_a_failing_source_is_recorded_and_the_others_still_collect(self, database: Database, now: datetime) -> None:
        clock = FakeClock(now)
        broken = FakeSource("claude:broken")
        broken.identity_error = KeyError("defect in one provider")
        healthy = FakeSource("codex:home", Provider.CODEX)
        healthy.identity = make_identity(Provider.CODEX, label="Codex A")
        healthy.result = make_reading(healthy.identity, now, make_window(used_percent=9))

        report = collector(database, clock, SourceBundle(broken, None, 0), SourceBundle(healthy, None, 0)).run_cycle()

        assert statuses(database)[("claude:broken", "quota")] == (
            "UnexpectedFailureError",
            "unexpected KeyError; the log says where it was raised",
        )
        assert statuses(database)[("codex:home", "quota")] == (None, None)
        with database.connect() as conn:
            assert [row.used_percent for row in db.latest_snapshots(conn)] == [9]
        assert (
            report.outcomes[0].quota_error
            == "UnexpectedFailureError: unexpected KeyError; the log says where it was raised"
        )
        assert report.outcomes[1].quota_error is None

    def test_a_claude_home_that_cannot_be_identified_again_does_not_stop_the_other_sources(
        self, database: Database, now: datetime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "claude"
        home.mkdir()
        config = ClaudeSourceConfig(
            config_dir=home,
            account_file=tmp_path / "account.json",
            mode=SourceMode.DIRECT,
            keychain_service="synthetic",
        )
        with httpx2.Client() as client:
            claude = ClaudeSource(config, client, lambda _service: "{}")
            healthy = FakeSource("codex:home", Provider.CODEX)
            healthy.identity = make_identity(Provider.CODEX, label="Codex A")
            healthy.result = make_reading(healthy.identity, now, make_window(used_percent=9))

            def refuse(path: Path) -> Path:
                raise ConfigError(f"cannot identify {path}: PermissionError")

            monkeypatch.setattr("usagedash.providers.claude.directory_identity", refuse)

            report = collector(
                database, FakeClock(now), SourceBundle(claude, None, 0), SourceBundle(healthy, None, 1)
            ).run_cycle()

        assert healthy.read_calls == [now]
        assert report.outcomes[1].quota_error is None
        with database.connect() as conn:
            assert [row.used_percent for row in db.latest_snapshots(conn)] == [9]

    def test_a_deliberate_error_from_a_scanner_keeps_its_own_message_and_is_not_logged_as_a_defect(
        self, database: Database, now: datetime, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.result = make_reading(make_identity(), now, make_window())
        scanner = FakeScanner(ConfigError("the sessions directory setting is not usable"))

        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            report = collector(database, clock, SourceBundle(source, scanner, 0)).run_cycle()

        assert statuses(database)[("claude:home", "tokens")] == (
            "ConfigError",
            "the sessions directory setting is not usable",
        )
        assert report.outcomes[0].token_error == "ConfigError: the sessions directory setting is not usable"
        assert "Unexpected failure" not in caplog.text

    def test_a_failed_read_keeps_the_logged_in_account_and_its_previous_reading(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.identity = make_identity()
        source.result = make_reading(source.identity, now, make_window(used_percent=30))
        runner = collector(database, clock, SourceBundle(source, None, 0))
        runner.run_cycle()

        clock.advance(minutes=5)
        source.result = CredentialExpiredError("token expired")
        runner.run_cycle()

        with database.connect() as conn:
            account = db.list_accounts(conn)[0]
            state = db.list_source_states(conn)[0]
            latest = db.latest_snapshots(conn)
        assert statuses(database)[("claude:home", "quota")] == ("CredentialExpiredError", "token expired")
        assert state.current_account_id == account.id
        assert account.last_live_at == now + timedelta(minutes=5)
        assert [(row.used_percent, row.confirmed_at) for row in latest] == [(30, now)]

    def test_a_logged_out_source_clears_who_is_logged_in(self, database: Database, now: datetime) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.identity = make_identity()
        source.result = make_reading(source.identity, now, make_window())
        runner = collector(database, clock, SourceBundle(source, None, 0))
        runner.run_cycle()

        clock.advance(minutes=5)
        source.identity = None
        source.result = UsageDashError("not logged in")
        runner.run_cycle()

        with database.connect() as conn:
            assert db.list_source_states(conn)[0].current_account_id is None

    def test_a_logout_discovered_by_the_read_itself_takes_effect_in_that_same_cycle(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source = LearningSource("codex:standing")
        account_a = make_identity(Provider.CODEX, "workspace-a", "Account A")
        source.learns = [account_a, None]
        source.result = make_reading(account_a, now, make_window(used_percent=30))
        runner = collector(database, clock, SourceBundle(source, None, 1))
        runner.run_cycle()

        clock.advance(minutes=5)
        source.result = CredentialUnavailableError("no login here yet")
        runner.run_cycle()

        with database.connect() as conn:
            account = db.list_accounts(conn)[0]
            state = db.list_source_states(conn)[0]
        assert state.current_account_id is None
        assert account.last_live_at == now
        assert statuses(database)[("codex:standing", "quota")][0] == "CredentialUnavailableError"

    def test_a_different_login_discovered_by_a_failed_read_is_recorded_instead_of_the_earlier_one(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source = LearningSource("codex:standing")
        account_a = make_identity(Provider.CODEX, "workspace-a", "Account A")
        account_b = make_identity(Provider.CODEX, "workspace-b", "Account B")
        source.learns = [account_a, account_b]
        source.result = make_reading(account_a, now, make_window(used_percent=30))
        runner = collector(database, clock, SourceBundle(source, None, 1))
        runner.run_cycle()

        clock.advance(minutes=5)
        source.result = ProviderRequestError("backend unavailable")
        runner.run_cycle()

        later = now + timedelta(minutes=5)
        with database.connect() as conn:
            accounts = {account.external_id: account for account in db.list_accounts(conn)}
            state = db.list_source_states(conn)[0]
            timeline = db.active_timeline(conn, "codex:standing")
        assert state.current_account_id == accounts["workspace-b"].id
        assert (accounts["workspace-a"].last_live_at, accounts["workspace-b"].last_live_at) == (now, later)
        assert timeline == [(now, accounts["workspace-a"].id), (later, accounts["workspace-b"].id)]

    def test_a_first_read_that_fails_after_naming_the_login_still_records_who_is_logged_in(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source = LearningSource("codex:standing")
        source.learns = [make_identity(Provider.CODEX, "workspace-a", "Account A")]
        source.result = ProviderRequestError("backend unavailable")

        report = collector(database, clock, SourceBundle(source, None, 1)).run_cycle()

        with database.connect() as conn:
            account = db.list_accounts(conn)[0]
            state = db.list_source_states(conn)[0]
        assert (account.external_id, state.current_account_id) == ("workspace-a", account.id)
        assert report.outcomes[0].account_label == "Account A"
        assert report.outcomes[0].quota_error == "ProviderRequestError: backend unavailable"

    def test_a_vendor_cli_s_own_error_text_reaches_neither_the_stored_status_nor_the_report(
        self, database: Database, now: datetime, tmp_path: Path
    ) -> None:
        grok_auth_file(tmp_path / "auth.json")
        environment = {
            "FAKE_PROTOCOL": "grok",
            "FAKE_SCENARIO": "billing_error",
            "FAKE_RECORD": str(tmp_path / "record"),
            "FAKE_BILLING_ERROR": json.dumps({"code": -32603, "message": "synthetic-secret quoted by the CLI"}),
            "PATH": "/usr/bin:/bin",
        }
        source = GrokCliSource(tmp_path, FAKE_CLI, environment)

        report = collector(database, FakeClock(now), SourceBundle(source, None, 0)).run_cycle()

        stored = statuses(database)[(source.source_id, "quota")]
        assert stored == ("ProviderRequestError", "Grok agent rejected _x.ai/billing (error code -32603)")
        assert report.outcomes[0].quota_error == f"{stored[0]}: {stored[1]}"

    def test_unreadable_session_files_are_reported_as_a_token_error(self, database: Database, now: datetime) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.result = make_reading(make_identity(), now, make_window())
        scanner = FakeScanner(ScanReport(files_read=1, files_failed=2, first_failure="a.jsonl: PermissionError"))

        collector(database, clock, SourceBundle(source, scanner, 0)).run_cycle()

        error_type, message = statuses(database)[("claude:home", "tokens")]
        assert error_type == "TokenScanError"
        assert message == "2 session file(s) unreadable; first: a.jsonl: PermissionError"

    def test_malformed_usage_records_are_logged_but_do_not_fail_the_scan(
        self, database: Database, now: datetime, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.result = make_reading(make_identity(), now, make_window())
        scanner = FakeScanner(ScanReport(files_read=4, malformed_records=3))

        with caplog.at_level(logging.WARNING, logger="usagedash.collector"):
            report = collector(database, clock, SourceBundle(source, scanner, 0)).run_cycle()

        assert statuses(database)[("claude:home", "tokens")] == (None, None)
        assert (report.outcomes[0].token_error, report.outcomes[0].files_read) == (None, 4)
        ref = _source_ref("claude:home")
        assert f"{ref}: skipped 3 malformed usage record(s)" in caplog.messages

    def test_malformed_usage_records_are_logged_even_when_the_same_scan_could_not_read_some_files(
        self, database: Database, now: datetime, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.result = make_reading(make_identity(), now, make_window())
        scanner = FakeScanner(
            ScanReport(files_read=4, files_failed=2, first_failure="a.jsonl: PermissionError", malformed_records=3)
        )

        with caplog.at_level(logging.WARNING, logger="usagedash.collector"):
            collector(database, clock, SourceBundle(source, scanner, 0)).run_cycle()

        assert statuses(database)[("claude:home", "tokens")] == (
            "TokenScanError",
            "2 session file(s) unreadable; first: a.jsonl: PermissionError",
        )
        ref = _source_ref("claude:home")
        assert f"{ref}: skipped 3 malformed usage record(s)" in caplog.messages

    def test_a_scanner_defect_is_recorded_and_does_not_stop_the_cycle(self, database: Database, now: datetime) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.result = make_reading(make_identity(), now, make_window())

        report = collector(database, clock, SourceBundle(source, FakeScanner(TypeError("defect")), 0)).run_cycle()

        assert statuses(database)[("claude:home", "tokens")] == (
            "UnexpectedFailureError",
            "unexpected TypeError; the log says where it was raised",
        )
        assert (
            report.outcomes[0].token_error
            == "UnexpectedFailureError: unexpected TypeError; the log says where it was raised"
        )
        assert statuses(database)[("claude:home", "quota")] == (None, None)


class TestRateLimitBackoff:
    def source_limited(self, retry_after: float | None) -> FakeSource:
        source = FakeSource("claude:home")
        source.identity = make_identity()
        source.result = ProviderRequestError("limited", status_code=429, retry_after_seconds=retry_after)
        return source

    def test_stops_calling_the_provider_until_two_poll_intervals_have_passed(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source = self.source_limited(None)
        runner = collector(database, clock, SourceBundle(source, None, 0))

        runner.run_cycle()
        clock.advance(seconds=2 * POLL_SECONDS - 1)
        runner.run_cycle()
        assert len(source.read_calls) == 1
        assert "backing off" in (statuses(database)[("claude:home", "quota")][1] or "")

        clock.advance(seconds=1)
        runner.run_cycle()
        assert len(source.read_calls) == 2

    def test_honors_a_longer_wait_requested_by_the_provider(self, database: Database, now: datetime) -> None:
        clock = FakeClock(now)
        source = self.source_limited(3600)
        runner = collector(database, clock, SourceBundle(source, None, 0))

        runner.run_cycle()
        clock.advance(seconds=3599)
        runner.run_cycle()
        assert len(source.read_calls) == 1

        clock.advance(seconds=1)
        runner.run_cycle()
        assert len(source.read_calls) == 2

    @pytest.mark.parametrize("retry_after", [1e300, 86_400.5, 10.0**18])
    def test_a_wait_longer_than_a_day_is_cut_to_a_day_instead_of_overflowing(
        self, database: Database, now: datetime, retry_after: float
    ) -> None:
        clock = FakeClock(now)
        source = self.source_limited(retry_after)
        runner = collector(database, clock, SourceBundle(source, None, 0))

        report = runner.run_cycle()
        assert report.outcomes[0].quota_error == "ProviderRequestError: limited"

        clock.advance(days=1, seconds=-1)
        runner.run_cycle()
        assert len(source.read_calls) == 1

        clock.advance(seconds=1)
        runner.run_cycle()
        assert len(source.read_calls) == 2

    @pytest.mark.parametrize("retry_after", [float("nan"), float("inf"), float("-inf"), -30.0, 0.0])
    def test_a_wait_that_is_not_a_usable_number_falls_back_to_two_poll_intervals(
        self, database: Database, now: datetime, retry_after: float
    ) -> None:
        clock = FakeClock(now)
        source = self.source_limited(retry_after)
        runner = collector(database, clock, SourceBundle(source, None, 0))

        runner.run_cycle()
        clock.advance(seconds=2 * POLL_SECONDS - 1)
        runner.run_cycle()
        assert len(source.read_calls) == 1

        clock.advance(seconds=1)
        runner.run_cycle()
        assert len(source.read_calls) == 2

    @pytest.mark.parametrize(
        ("retry_after", "wait_seconds"),
        [pytest.param(600.0, 600, id="the provider's wait"), pytest.param(None, 2 * POLL_SECONDS, id="two polls")],
    )
    def test_the_wait_runs_from_when_the_answer_arrived_not_from_when_the_read_began(
        self, database: Database, now: datetime, retry_after: float | None, wait_seconds: int
    ) -> None:
        read_takes = timedelta(seconds=30)
        clock = FakeClock(now)
        source = SlowSource("claude:home", clock, read_takes)
        source.result = ProviderRequestError("limited", status_code=429, retry_after_seconds=retry_after)
        runner = collector(database, clock, SourceBundle(source, None, 0))
        runner.run_cycle()
        answered_at = now + read_takes

        clock.now = answered_at + timedelta(seconds=wait_seconds - 1)
        runner.run_cycle()
        assert len(source.read_calls) == 1

        clock.now = answered_at + timedelta(seconds=wait_seconds)
        runner.run_cycle()
        assert len(source.read_calls) == 2

    def test_the_wait_is_elapsed_time_when_the_clock_reads_in_a_zone_whose_clocks_change(
        self, database: Database
    ) -> None:
        new_york = ZoneInfo("America/New_York")
        answered_at = datetime(2026, 11, 1, 0, 30, tzinfo=new_york)
        clock = FakeClock(answered_at)
        source = self.source_limited(7200)
        runner = collector(database, clock, SourceBundle(source, None, 0))
        runner.run_cycle()

        clock.now = (answered_at.astimezone(UTC) + timedelta(seconds=7199)).astimezone(new_york)
        runner.run_cycle()
        assert len(source.read_calls) == 1
        assert "until 06:30 UTC" in (statuses(database)[("claude:home", "quota")][1] or "")

        clock.now = (answered_at.astimezone(UTC) + timedelta(seconds=7200)).astimezone(new_york)
        runner.run_cycle()
        assert len(source.read_calls) == 2

    def test_a_success_ends_the_backoff(self, database: Database, now: datetime) -> None:
        clock = FakeClock(now)
        source = self.source_limited(None)
        runner = collector(database, clock, SourceBundle(source, None, 0))
        runner.run_cycle()

        clock.advance(seconds=2 * POLL_SECONDS)
        source.result = make_reading(make_identity(), clock.now, make_window())
        runner.run_cycle()
        clock.advance(seconds=1)
        runner.run_cycle()

        assert len(source.read_calls) == 3
        assert statuses(database)[("claude:home", "quota")] == (None, None)

    def test_other_request_failures_do_not_trigger_a_backoff(self, database: Database, now: datetime) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.result = ProviderRequestError("server error", status_code=503)
        runner = collector(database, clock, SourceBundle(source, None, 0))

        runner.run_cycle()
        clock.advance(seconds=1)
        runner.run_cycle()

        assert len(source.read_calls) == 2


class TestCycleLifecycle:
    def test_a_second_cycle_is_refused_while_one_is_running(self, database: Database, now: datetime) -> None:
        clock = FakeClock(now)
        runner_holder: list[Collector] = []
        observed: dict[str, object] = {}

        class ReentrantSource(FakeSource):
            def current_identity(self) -> AccountIdentity | None:
                observed["running_during_cycle"] = runner_holder[0].is_running
                try:
                    runner_holder[0].run_cycle()
                except CycleAlreadyRunningError as error:
                    observed["second_cycle_error"] = str(error)
                return None

        source = ReentrantSource("claude:home")
        runner_holder.append(collector(database, clock, SourceBundle(source, None, 0)))

        runner_holder[0].run_cycle()

        assert observed == {
            "running_during_cycle": True,
            "second_cycle_error": "a collection cycle is already running",
        }
        assert not runner_holder[0].is_running
        assert runner_holder[0].last_finished_at == now

    def test_the_lock_file_and_its_directory_are_private_even_if_the_directory_was_not(
        self, database: Database, now: datetime
    ) -> None:
        data_dir = database.path.parent
        data_dir.chmod(0o777)

        collector(database, FakeClock(now), SourceBundle(FakeSource("claude:home"), None, 0)).run_cycle()

        assert data_dir.stat().st_mode & 0o777 == 0o700
        assert (data_dir / "collector.lock").stat().st_mode & 0o777 == 0o600

    def test_a_lock_file_that_cannot_be_opened_is_a_configuration_error_and_no_source_is_read(
        self, database: Database, now: datetime
    ) -> None:
        (database.path.parent / "collector.lock").mkdir()
        source = FakeSource("claude:home")
        runner = collector(database, FakeClock(now), SourceBundle(source, None, 0))

        with pytest.raises(ConfigError, match=r"cannot open .*collector\.lock as a private file: IsADirectoryError"):
            runner.run_cycle()

        assert source.read_calls == []
        assert not runner.is_running

    def test_a_cycle_is_refused_while_another_process_is_collecting_into_the_same_database(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        observed: dict[str, object] = {}
        other_process = collector(database, clock, SourceBundle(FakeSource("claude:home"), FakeScanner(), 0))

        class CollectsFromAnotherProcess(FakeSource):
            def current_identity(self) -> AccountIdentity | None:
                try:
                    other_process.run_cycle()
                except CycleAlreadyRunningError as error:
                    observed["error"] = str(error)
                return None

        agent_collector = collector(database, clock, SourceBundle(CollectsFromAnotherProcess("claude:home"), None, 0))

        agent_collector.run_cycle()

        assert observed == {"error": "another process is already collecting into this database"}
        with database.connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0] == 0

    def test_the_cross_process_lock_is_released_after_a_cycle_even_when_it_fails(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        first = collector(database, clock, SourceBundle(FakeSource("claude:home"), None, 0))
        second = collector(database, clock, SourceBundle(FakeSource("claude:home"), None, 0))

        first.run_cycle()
        second.run_cycle()

        assert second.last_finished_at == now
        assert (database.path.parent / "collector.lock").stat().st_mode & 0o777 == 0o600

    def test_sources_dropped_from_the_configuration_stop_being_reported(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        kept, dropped = FakeSource("claude:kept"), FakeSource("claude:dropped")
        collector(database, clock, SourceBundle(kept, None, 0), SourceBundle(dropped, None, 1)).run_cycle()
        assert {source_id for source_id, _ in statuses(database)} == {"claude:kept", "claude:dropped"}

        collector(database, clock, SourceBundle(kept, None, 0)).run_cycle()

        assert {source_id for source_id, _ in statuses(database)} == {"claude:kept"}


class TestWhatIsLoggedForAnUnexpectedFailure:
    def raised(self) -> ValueError:
        def innermost() -> None:
            raise KeyError("synthetic-inner-secret")

        def outer() -> None:
            try:
                innermost()
            except KeyError as error:
                raise ValueError("synthetic-outer-secret") from error

        with pytest.raises(ValueError, match="synthetic-outer-secret") as caught:
            outer()
        return caught.value

    def test_the_description_names_each_type_in_the_chain_and_where_it_was_raised(self) -> None:
        description = describe_without_messages(self.raised())

        assert description.startswith("ValueError raised at:\n")
        assert "which followed KeyError raised at:\n" in description
        assert "in outer" in description
        assert "in innermost" in description
        assert "test_collector.py" in description

    def test_the_description_leaves_out_the_message_of_the_error_and_of_every_cause(self) -> None:
        description = describe_without_messages(self.raised())

        assert "synthetic-outer-secret" not in description.replace('raise ValueError("synthetic-outer-secret")', "")
        assert "synthetic-inner-secret" not in description.replace('raise KeyError("synthetic-inner-secret")', "")

    def test_an_error_raised_while_handling_another_is_followed_too_unless_that_was_suppressed(self) -> None:
        def while_handling(suppress: bool) -> RuntimeError:
            try:
                try:
                    raise KeyError("first")
                except KeyError:
                    if suppress:
                        raise RuntimeError("second") from None
                    raise RuntimeError("second")  # noqa: B904 - the implicit context is what is being tested
            except RuntimeError as error:
                return error

        assert "which followed KeyError raised at:" in describe_without_messages(while_handling(suppress=False))
        assert "KeyError" not in describe_without_messages(while_handling(suppress=True))

    def test_a_chain_that_loops_back_on_itself_ends(self) -> None:
        first, second = ValueError("first"), KeyError("second")
        first.__cause__, second.__cause__ = second, first

        description = describe_without_messages(first)

        assert description.count("raised at:") == 2

    def test_an_unexpected_failure_in_a_source_is_logged_without_its_message_and_recorded_by_type(
        self, database: Database, now: datetime, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.identity_error = TypeError("cannot format synthetic-secret-value")
        scanner = FakeScanner(RuntimeError("while reading synthetic-prompt-text"))

        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            report = collector(database, clock, SourceBundle(source, scanner, 0)).run_cycle()

        assert "synthetic" not in caplog.text
        ref = _source_ref("claude:home")
        assert f"Unexpected failure collecting quota from {ref}\nTypeError raised at:" in caplog.text
        assert f"Unexpected failure scanning tokens for {ref}\nRuntimeError raised at:" in caplog.text
        assert all(record.exc_info is None for record in caplog.records)
        shown = [
            *statuses(database)[("claude:home", "quota")],
            *statuses(database)[("claude:home", "tokens")],
            report.outcomes[0].quota_error,
            report.outcomes[0].token_error,
        ]
        assert all(text is not None and "synthetic" not in text for text in shown)
        assert report.outcomes[0].token_error == (
            "UnexpectedFailureError: unexpected RuntimeError; the log says where it was raised"
        )

    def test_an_unexpected_failure_of_the_read_itself_is_logged_the_same_way(
        self, database: Database, now: datetime, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.identity = make_identity()
        source.result = TypeError("cannot parse synthetic-payload-text")

        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            report = collector(database, clock, SourceBundle(source, None, 0)).run_cycle()

        assert "synthetic" not in caplog.text
        ref = _source_ref("claude:home")
        assert f"Unexpected failure reading quota from {ref}\nTypeError raised at:" in caplog.text
        assert report.outcomes[0].quota_error == (
            "UnexpectedFailureError: unexpected TypeError; the log says where it was raised"
        )
        assert statuses(database)[("claude:home", "quota")] == (
            "UnexpectedFailureError",
            "unexpected TypeError; the log says where it was raised",
        )

    def test_a_source_id_that_holds_a_home_and_an_email_is_logged_as_a_hash(
        self, database: Database, now: datetime, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock(now)
        source_id = f"codex:{Path.home()}/account-a@example.test"
        source = FakeSource(source_id)
        source.identity_error = TypeError("cannot format synthetic-secret-value")

        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            collector(database, clock, SourceBundle(source, None, 0)).run_cycle()

        assert "account-a@example.test" not in caplog.text
        assert str(Path.home()) not in caplog.text
        assert "synthetic" not in caplog.text
        assert _source_ref(source_id) in caplog.text

    def test_a_home_of_one_character_is_left_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Path, "home", lambda: Path("/"))
        assert _redact_paths("still /there") == "still /there"


class TestIdentityAfterAReadThatFailsUnexpectedly:
    def test_a_logout_the_read_discovered_before_hitting_a_defect_still_takes_effect(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source = LearningSource("codex:standing")
        account_a = make_identity(Provider.CODEX, "workspace-a", "Account A")
        source.learns = [account_a, None]
        source.result = make_reading(account_a, now, make_window(used_percent=30))
        runner = collector(database, clock, SourceBundle(source, None, 1))
        runner.run_cycle()

        clock.advance(minutes=5)
        source.result = TypeError("defect while parsing")
        report = runner.run_cycle()

        with database.connect() as conn:
            state = db.list_source_states(conn)[0]
        assert state.current_account_id is None
        assert report.outcomes[0].quota_error == (
            "UnexpectedFailureError: unexpected TypeError; the log says where it was raised"
        )

    def test_a_different_login_the_read_discovered_before_hitting_a_defect_is_recorded(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source = LearningSource("codex:standing")
        account_a = make_identity(Provider.CODEX, "workspace-a", "Account A")
        account_b = make_identity(Provider.CODEX, "workspace-b", "Account B")
        source.learns = [account_a, account_b]
        source.result = make_reading(account_a, now, make_window(used_percent=30))
        runner = collector(database, clock, SourceBundle(source, None, 1))
        runner.run_cycle()

        clock.advance(minutes=5)
        source.result = TypeError("defect while parsing")
        runner.run_cycle()

        with database.connect() as conn:
            accounts = {account.id: account.external_id for account in db.list_accounts(conn)}
            state = db.list_source_states(conn)[0]
        assert state.current_account_id is not None
        assert accounts[state.current_account_id] == "workspace-b"


class TestTheLoginTimelineAcrossALogout:
    def source_logged_in(self, database: Database, clock: FakeClock) -> tuple[FakeSource, Collector]:
        source = FakeSource("claude:home")
        source.identity = make_identity(label="Personal")
        source.result = make_reading(source.identity, clock.now, make_window())
        runner = collector(database, clock, SourceBundle(source, FakeScanner(), 0))
        runner.run_cycle()
        return source, runner

    def test_usage_after_a_logout_is_not_credited_to_the_account_that_logged_out(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source, runner = self.source_logged_in(database, clock)
        scanner = FakeScanner()
        logged_out_at = now + timedelta(minutes=5)
        scanner.probe_times = [now + timedelta(minutes=1), logged_out_at, logged_out_at + timedelta(hours=1)]

        clock.advance(minutes=5)
        source.identity = None
        source.result = CredentialUnavailableError("not logged in")
        collector(database, clock, SourceBundle(source, scanner, 0)).run_cycle()

        with database.connect() as conn:
            account_id = db.list_accounts(conn)[0].id
            timeline = db.active_timeline(conn, "claude:home")
        assert timeline == [(now, account_id), (logged_out_at, None)]
        assert scanner.resolved == [account_id, 0, 0]
        assert runner.last_finished_at == now

    def test_logging_back_in_to_the_same_account_starts_a_new_interval(self, database: Database, now: datetime) -> None:
        clock = FakeClock(now)
        source, runner = self.source_logged_in(database, clock)
        identity = source.identity
        assert identity is not None

        clock.advance(minutes=5)
        source.identity = None
        source.result = CredentialUnavailableError("not logged in")
        runner.run_cycle()

        clock.advance(hours=2)
        back_at = clock.now
        source.identity = identity
        source.result = make_reading(identity, back_at, make_window())
        runner.run_cycle()
        clock.advance(minutes=5)
        runner.run_cycle()

        with database.connect() as conn:
            account_id = db.list_accounts(conn)[0].id
            timeline = db.active_timeline(conn, "claude:home")
        assert timeline == [(now, account_id), (now + timedelta(minutes=5), None), (back_at, account_id)]


class SlowSource(FakeSource):
    def __init__(self, source_id: str, clock: FakeClock, read_takes: timedelta):
        super().__init__(source_id)
        self._clock = clock
        self._read_takes = read_takes
        self.identity_after_read: AccountIdentity | None = None

    def read_quota(self, now: datetime) -> QuotaReading:
        self._clock.now += self._read_takes
        self.identity = self.identity_after_read
        return super().read_quota(now)


class TestALoginIsDatedFromWhenItWasSeen:
    READ_TAKES = timedelta(seconds=30)

    def logged_in_as_a(self, database: Database, clock: FakeClock) -> tuple[SlowSource, AccountIdentity]:
        account_a = make_identity(label="Account A")
        source = SlowSource("claude:home", clock, timedelta())
        source.identity_after_read = account_a
        source.result = make_reading(account_a, clock.now, make_window())
        collector(database, clock, SourceBundle(source, None, 0)).run_cycle()
        source._read_takes = self.READ_TAKES
        return source, account_a

    def cycle_with_probes(self, database: Database, clock: FakeClock, source: SlowSource) -> tuple[list[int], datetime]:
        clock.advance(minutes=5)
        began = clock.now
        ended = began + self.READ_TAKES
        scanner = FakeScanner()
        scanner.probe_times = [began, ended - timedelta(seconds=1), ended, ended + timedelta(minutes=1)]
        collector(database, clock, SourceBundle(source, scanner, 0)).run_cycle()
        return scanner.resolved, began

    def account_ids(self, database: Database) -> dict[str, int]:
        with database.connect() as conn:
            return {account.label: account.id for account in db.list_accounts(conn)}

    def test_another_login_found_by_a_read_that_failed_starts_when_the_read_ended(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source, _ = self.logged_in_as_a(database, clock)
        source.identity_after_read = make_identity(external_id="org-b", label="Account B")
        source.result = ProviderRequestError("the provider did not answer")

        resolved, began = self.cycle_with_probes(database, clock, source)

        ids = self.account_ids(database)
        assert resolved == [ids["Account A"], ids["Account A"], ids["Account B"], ids["Account B"]]
        with database.connect() as conn:
            assert db.active_timeline(conn, "claude:home") == [
                (now, ids["Account A"]),
                (began + self.READ_TAKES, ids["Account B"]),
            ]

    def test_a_logout_found_by_a_read_that_failed_starts_when_the_read_ended(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source, _ = self.logged_in_as_a(database, clock)
        source.identity_after_read = None
        source.result = CredentialUnavailableError("not logged in")

        resolved, _ = self.cycle_with_probes(database, clock, source)

        account_a = self.account_ids(database)["Account A"]
        assert resolved == [account_a, account_a, 0, 0]

    def test_another_login_stated_by_a_reading_starts_when_the_read_ended_and_the_reading_keeps_its_time(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source, _ = self.logged_in_as_a(database, clock)
        account_b = make_identity(external_id="org-b", label="Account B")
        source.identity_after_read = account_b
        attempt_began = now + timedelta(minutes=5)
        source.result = make_reading(account_b, attempt_began, make_window(used_percent=44))

        resolved, began = self.cycle_with_probes(database, clock, source)

        ids = self.account_ids(database)
        assert began == attempt_began
        assert resolved == [ids["Account A"], ids["Account A"], ids["Account B"], ids["Account B"]]
        with database.connect() as conn:
            reading_of_b = [row for row in db.latest_snapshots(conn) if row.account_id == ids["Account B"]]
            status = next(row for row in db.list_statuses(conn) if row.kind == "quota")
        assert [(row.used_percent, row.observed_at) for row in reading_of_b] == [(44, attempt_began)]
        assert status.last_attempt_at == attempt_began


class TestALoginSeenPartWayThroughASecond:
    def test_usage_from_earlier_in_that_second_stays_with_the_account_that_was_logged_in_then(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.identity = make_identity(label="Account A")
        source.result = make_reading(source.identity, now, make_window())
        collector(database, clock, SourceBundle(source, None, 0)).run_cycle()

        seen_at = now + timedelta(minutes=5, milliseconds=800)
        clock.now = seen_at
        source.identity = make_identity(external_id="org-b", label="Account B")
        source.result = make_reading(source.identity, seen_at, make_window())
        scanner = FakeScanner()
        second = seen_at.replace(microsecond=0)
        scanner.probe_times = [
            second + timedelta(milliseconds=200),
            second + timedelta(milliseconds=900),
            second + timedelta(seconds=1),
        ]
        collector(database, clock, SourceBundle(source, scanner, 0)).run_cycle()

        with database.connect() as conn:
            ids = {account.label: account.id for account in db.list_accounts(conn)}
        assert scanner.resolved == [ids["Account A"], ids["Account A"], ids["Account B"]]


class TestALoginThatCannotBeTold:
    def logged_in(self, database: Database, clock: FakeClock) -> tuple[FakeSource, int]:
        source = FakeSource("claude:home")
        source.identity = make_identity(label="Account A")
        source.result = make_reading(source.identity, clock.now, make_window())
        collector(database, clock, SourceBundle(source, None, 0)).run_cycle()
        with database.connect() as conn:
            return source, db.list_accounts(conn)[0].id

    @pytest.mark.parametrize(
        ("lookup_failure", "shown_as"),
        [
            pytest.param(
                ProviderSchemaError("the auth file has an unexpected shape"),
                ("ProviderSchemaError", "the auth file has an unexpected shape"),
                id="a failure the source reports",
            ),
            pytest.param(
                KeyError("a defect"),
                ("UnexpectedFailureError", "unexpected KeyError; the log says where it was raised"),
                id="a defect",
            ),
        ],
    )
    def test_usage_read_after_the_lookup_failed_is_nobody_s_and_stays_so_once_the_login_is_known_again(
        self, database: Database, now: datetime, lookup_failure: Exception, shown_as: tuple[str, str]
    ) -> None:
        clock = FakeClock(now)
        source, account_a = self.logged_in(database, clock)

        clock.advance(minutes=5)
        unknown_from = clock.now
        source.result = CredentialExpiredError("the token has expired")
        source.identity_error = lookup_failure
        scanner = FakeScanner()
        scanner.probe_times = [unknown_from - timedelta(seconds=1), unknown_from, unknown_from + timedelta(minutes=1)]
        collector(database, clock, SourceBundle(source, scanner, 0)).run_cycle()

        assert scanner.resolved == [account_a, 0, 0]
        assert statuses(database)[("claude:home", "quota")] == shown_as
        with database.connect() as conn:
            assert db.list_source_states(conn)[0].current_account_id is None
            assert db.active_timeline(conn, "claude:home") == [(now, account_a), (unknown_from, None)]

        clock.advance(minutes=5)
        source.identity_error = None
        source.identity = make_identity(external_id="org-b", label="Account B")
        later = FakeScanner()
        later.probe_times = [unknown_from + timedelta(minutes=1), clock.now]
        collector(database, clock, SourceBundle(source, later, 0)).run_cycle()

        with database.connect() as conn:
            account_b = next(account.id for account in db.list_accounts(conn) if account.label == "Account B")
        assert later.resolved == [0, account_b]

    def test_the_usage_the_scan_stores_in_that_cycle_is_not_the_earlier_account_s(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source, account_a = self.logged_in(database, clock)

        clock.advance(minutes=5)
        source.result = CredentialExpiredError("the token has expired")
        source.identity_error = ProviderSchemaError("the auth file has an unexpected shape")
        collector(database, clock, SourceBundle(source, FakeScanner(), 0)).run_cycle()

        with database.connect() as conn:
            credited = [row["account_id"] for row in conn.execute("SELECT account_id FROM token_usage")]
        assert credited == [0]
        assert account_a != 0

    def test_a_failed_lookup_is_what_is_reported_and_no_success_is_recorded_for_the_attempt(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source, _ = self.logged_in(database, clock)

        clock.advance(minutes=5)
        source.result = CredentialExpiredError("the token has expired")
        source.identity_error = ProviderSchemaError("the auth file has an unexpected shape")
        report = collector(database, clock, SourceBundle(source, None, 0)).run_cycle()

        assert report.outcomes[0].quota_error == "ProviderSchemaError: the auth file has an unexpected shape"
        with database.connect() as conn:
            status = next(row for row in db.list_statuses(conn) if row.kind == "quota")
        assert (status.last_attempt_at, status.last_success_at) == (now + timedelta(minutes=5), now)


class TestALoginIsRecordedBeforeTheScanThatReliesOnIt:
    SOURCE = "claude:home"

    def logged_in_as_a(self, database: Database, clock: FakeClock) -> tuple[FakeSource, FakeScanner, Collector, int]:
        source = FakeSource(self.SOURCE)
        source.identity = make_identity(label="Account A")
        source.result = make_reading(source.identity, clock.now, make_window())
        scanner = FakeScanner()
        cycles = collector(database, clock, SourceBundle(source, scanner, 0))
        cycles.run_cycle()
        with database.connect() as conn:
            conn.execute("DELETE FROM token_usage")
            return source, scanner, cycles, db.list_accounts(conn)[0].id

    def b_is_seen(self, source: FakeSource, clock: FakeClock) -> datetime:
        clock.advance(minutes=5)
        source.identity = make_identity(external_id="org-b", label="Account B")
        source.result = make_reading(source.identity, clock.now, make_window())
        return clock.now

    def refusing(self, monkeypatch: pytest.MonkeyPatch, function: str, refused: Exception) -> None:
        def refuse(*_arguments: object, **_options: object) -> None:
            raise refused

        monkeypatch.setattr(db, function, refuse)

    def account_b(self, database: Database) -> int:
        with database.connect() as conn:
            return next(account.id for account in db.list_accounts(conn) if account.label == "Account B")

    def credited(self, database: Database) -> list[int]:
        with database.connect() as conn:
            return [row["account_id"] for row in conn.execute("SELECT account_id FROM token_usage")]

    def test_a_reading_the_store_refuses_does_not_take_the_login_back_with_it(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        seen_at = self.b_is_seen(source, clock)
        scanner.probe_times = [seen_at - timedelta(seconds=1), seen_at, seen_at + timedelta(minutes=1)]
        self.refusing(monkeypatch, "record_reading", ValueError("refused: this text must not be shown"))

        report = cycles.run_cycle()

        account_b = self.account_b(database)
        assert scanner.resolved == [account_a, account_b, account_b]
        assert self.credited(database) == [account_b]
        assert report.outcomes[0].quota_error == (
            "UnexpectedFailureError: unexpected ValueError; the log says where it was raised"
        )
        assert report.outcomes[0].token_error is None
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == [(now, account_a), (seen_at, account_b)]
            assert db.list_source_states(conn)[0].current_account_id == account_b

    def test_a_login_that_cannot_be_recorded_as_it_is_is_recorded_as_nobody_s(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        seen_at = self.b_is_seen(source, clock)
        scanner.probe_times = [seen_at - timedelta(seconds=1), seen_at, seen_at + timedelta(minutes=1)]
        self.refusing(monkeypatch, "upsert_account", sqlite3.OperationalError("database is locked"))

        report = cycles.run_cycle()

        assert scanner.resolved == [account_a, 0, 0]
        assert self.credited(database) == [0]
        assert report.outcomes[0].quota_error == (
            "UnexpectedFailureError: unexpected OperationalError; the log says where it was raised"
        )
        assert report.outcomes[0].token_error is None
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == [(now, account_a), (seen_at, None)]
            assert db.list_source_states(conn)[0].current_account_id is None

        monkeypatch.undo()
        clock.advance(minutes=5)
        scanner.probe_times = [seen_at + timedelta(minutes=1), clock.now]
        cycles.run_cycle()

        assert scanner.resolved == [0, self.account_b(database)]

    def test_when_not_even_nobody_can_be_recorded_the_tokens_are_not_scanned(
        self,
        database: Database,
        now: datetime,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        seen_at = self.b_is_seen(source, clock)
        scanner.probe_times = [seen_at]
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("secret: database is locked"))

        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            report = cycles.run_cycle()

        assert scanner.resolved == []
        assert self.credited(database) == []
        assert report.outcomes[0].files_read == 0
        assert report.outcomes[0].token_error == (
            "TokenScanError: not scanned: who is logged in there could not be recorded"
        )
        assert report.outcomes[0].token_error_type == "TokenScanError"
        assert statuses(database)[(self.SOURCE, "tokens")] == (
            "TokenScanError",
            "not scanned: who is logged in there could not be recorded",
        )
        assert f"Could not record that the login at {_source_ref(self.SOURCE)} is unknown" in caplog.text
        assert "secret" not in caplog.text
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == [(now, account_a)]

    def test_what_was_written_while_the_login_was_unrecorded_is_nobody_s_once_a_login_is_recorded_again(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        unrecorded_from = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        clock.advance(minutes=5)
        cycles.run_cycle()
        monkeypatch.undo()
        clock.advance(minutes=5)
        scanner.probe_times = [
            unrecorded_from - timedelta(seconds=1),
            unrecorded_from,
            unrecorded_from + timedelta(minutes=7),
            clock.now,
        ]

        report = cycles.run_cycle()

        account_b = self.account_b(database)
        assert scanner.resolved == [account_a, 0, 0, account_b]
        assert report.outcomes[0].token_error is None
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == [
                (now, account_a),
                (unrecorded_from, None),
                (clock.now, account_b),
            ]

    def test_the_gap_is_recorded_from_its_start_when_the_next_login_can_only_be_recorded_as_nobody_s(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        unrecorded_from = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        monkeypatch.undo()
        self.refusing(monkeypatch, "upsert_account", sqlite3.OperationalError("database is locked"))
        clock.advance(minutes=5)
        scanner.probe_times = [unrecorded_from + timedelta(minutes=1), clock.now]

        report = cycles.run_cycle()

        assert scanner.resolved == [0, 0]
        assert report.outcomes[0].token_error is None
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == [(now, account_a), (unrecorded_from, None)]

    def test_a_new_collector_records_the_gap_from_the_first_failure_before_it_scans(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        unrecorded_from = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        note = self.boundary_file(database)
        assert note.stat().st_mode & 0o777 == 0o600
        monkeypatch.undo()
        clock.advance(minutes=5)
        scanner.probe_times = [
            unrecorded_from - timedelta(seconds=1),
            unrecorded_from,
            unrecorded_from + timedelta(minutes=1),
            clock.now,
        ]
        fresh = collector(database, clock, SourceBundle(source, scanner, 0))

        fresh.run_cycle()

        account_b = self.account_b(database)
        assert scanner.resolved == [account_a, 0, 0, account_b]
        assert not note.exists()
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == [
                (now, account_a),
                (unrecorded_from, None),
                (clock.now, account_b),
            ]

    @pytest.mark.parametrize(
        "store_failure",
        [
            pytest.param(OSError(1, "busy"), id="the write fails"),
            pytest.param(ConfigError("cannot make the directory private"), id="the directory cannot be made private"),
        ],
    )
    def test_a_fresh_collector_keeps_the_gap_when_the_recovery_file_cannot_be_written(
        self,
        database: Database,
        now: datetime,
        monkeypatch: pytest.MonkeyPatch,
        store_failure: OSError | ConfigError,
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        unrecorded_from = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))

        def refuse_store(_path: Path, _pending: object) -> None:
            raise store_failure

        monkeypatch.setattr("usagedash.collector.store_unrecorded_logins", refuse_store)
        cycles.run_cycle()
        note = self.boundary_file(database)
        assert not note.exists()
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == [(now, account_a), (unrecorded_from, None)]
        monkeypatch.undo()
        clock.advance(minutes=5)
        scanner.probe_times = [
            unrecorded_from - timedelta(seconds=1),
            unrecorded_from,
            unrecorded_from + timedelta(minutes=1),
            clock.now,
        ]
        fresh = collector(database, clock, SourceBundle(source, scanner, 0))

        fresh.run_cycle()

        account_b = self.account_b(database)
        assert scanner.resolved == [account_a, 0, 0, account_b]
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == [
                (now, account_a),
                (unrecorded_from, None),
                (clock.now, account_b),
            ]

    def test_a_failed_clear_of_the_boundary_file_does_not_invent_a_gap(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source = FakeSource(self.SOURCE)
        source.identity = make_identity(label="Account A")
        source.result = make_reading(source.identity, now, make_window())

        def refuse_an_empty_clear(path: Path, pending: dict[str, datetime]) -> None:
            if not pending:
                raise OSError(1, "busy")
            store_unrecorded_logins(path, pending)

        monkeypatch.setattr("usagedash.collector.store_unrecorded_logins", refuse_an_empty_clear)
        self.refusing(monkeypatch, "record_reading", ValueError("refused"))
        collector(database, clock, SourceBundle(source, FakeScanner(), 0)).run_cycle()

        with database.connect() as conn:
            timeline = db.active_timeline(conn, self.SOURCE)
        assert len(timeline) == 1
        assert timeline[0][1] is not None

    def test_an_earlier_stored_boundary_is_kept_over_a_later_note_in_memory(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, _account_a = self.logged_in_as_a(database, clock)
        unrecorded_from = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        earlier = now + timedelta(minutes=1)
        self.boundary_file(database).write_text(json.dumps({self.SOURCE: earlier.isoformat()}), encoding="utf-8")
        cycles.run_cycle()
        monkeypatch.undo()
        clock.advance(minutes=5)
        scanner.probe_times = [earlier, unrecorded_from, clock.now]

        cycles.run_cycle()

        assert scanner.resolved == [0, 0, self.account_b(database)]
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE)[1] == (earlier, None)

    def test_a_boundary_already_in_the_timeline_is_not_opened_again(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        unrecorded_from = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        monkeypatch.undo()
        clock.advance(minutes=5)
        cycles.run_cycle()
        self.boundary_file(database).write_text(
            json.dumps({self.SOURCE: unrecorded_from.isoformat()}), encoding="utf-8"
        )
        clock.advance(minutes=5)
        scanner.probe_times = [clock.now - timedelta(seconds=1), clock.now]
        fresh = collector(database, clock, SourceBundle(source, scanner, 0))

        fresh.run_cycle()

        assert scanner.resolved == [self.account_b(database), self.account_b(database)]
        with database.connect() as conn:
            timeline = db.active_timeline(conn, self.SOURCE)
        assert timeline == [
            (now, account_a),
            (unrecorded_from, None),
            (unrecorded_from + timedelta(minutes=5), self.account_b(database)),
        ]
        assert not self.boundary_file(database).exists()

    def test_an_older_nobody_interval_does_not_count_as_the_boundary_that_names_a_later_one(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, _account_a = self.logged_in_as_a(database, clock)
        self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        monkeypatch.undo()
        clock.advance(minutes=5)
        cycles.run_cycle()
        clock.advance(minutes=5)
        source.identity = None
        source.result = UsageDashError("not logged in")
        logged_out_at = clock.now
        cycles.run_cycle()
        clock.advance(minutes=5)
        source.identity = make_identity(external_id="org-b", label="Account B")
        source.result = make_reading(source.identity, clock.now, make_window())
        cycles.run_cycle()
        self.boundary_file(database).write_text(json.dumps({self.SOURCE: logged_out_at.isoformat()}), encoding="utf-8")
        with database.connect() as conn:
            before = db.active_timeline(conn, self.SOURCE)
        fresh = collector(database, clock, SourceBundle(source, scanner, 0))

        fresh.run_cycle()

        assert not self.boundary_file(database).exists()
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == before

    def test_an_unreadable_boundary_file_does_not_drop_the_note_already_in_memory(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        unrecorded_from = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        note = self.boundary_file(database)
        note.unlink()
        note.mkdir()
        monkeypatch.undo()
        clock.advance(minutes=5)
        scanner.probe_times = [unrecorded_from, clock.now]
        scans_before = scanner.scans

        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            report = cycles.run_cycle()

        assert "Could not read the unrecorded-login file" in caplog.text
        assert scanner.scans == scans_before
        assert scanner.resolved == []
        assert cycles._login_unrecorded_since[self.SOURCE] == unrecorded_from
        assert note.is_dir()
        assert report.outcomes[0].files_read == 0
        assert report.outcomes[0].quota_error == "UsageDashError: the unrecorded login boundary could not be read"
        assert report.outcomes[0].token_error == (
            "TokenScanError: not scanned: who is logged in there could not be recorded"
        )
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == [(now, account_a), (unrecorded_from, None)]

    @pytest.mark.parametrize("contents", ["{", "[]"])
    def test_a_boundary_file_that_is_not_a_login_map_is_removed(
        self, database: Database, now: datetime, contents: str
    ) -> None:
        clock = FakeClock(now)
        source, scanner, _cycles, account_a = self.logged_in_as_a(database, clock)
        note = self.boundary_file(database)
        note.write_text(contents, encoding="utf-8")
        clock.advance(minutes=5)
        seen_at = self.b_is_seen(source, clock)
        scanner.probe_times = [seen_at - timedelta(seconds=1), seen_at]
        fresh = collector(database, clock, SourceBundle(source, scanner, 0))

        fresh.run_cycle()

        assert scanner.resolved == [account_a, self.account_b(database)]
        assert not note.exists()

    def test_entries_that_are_not_a_source_and_a_timestamp_are_skipped(self, database: Database, now: datetime) -> None:
        clock = FakeClock(now)
        source, _scanner, _cycles, _account_a = self.logged_in_as_a(database, clock)
        kept = (now + timedelta(minutes=1)).isoformat()
        self.boundary_file(database).write_text(
            json.dumps(
                {
                    "": kept,
                    "claude:num": 1,
                    "claude:bad": "not-a-time",
                    "claude:other": kept,
                }
            ),
            encoding="utf-8",
        )
        clock.advance(minutes=5)
        self.b_is_seen(source, clock)
        fresh = collector(database, clock, SourceBundle(source, FakeScanner(), 0))

        fresh.run_cycle()

        assert json.loads(self.boundary_file(database).read_text(encoding="utf-8")) == {"claude:other": kept}
        with database.connect() as conn:
            assert all(account_id is not None for _started, account_id in db.active_timeline(conn, self.SOURCE))

    def test_a_boundary_that_cannot_be_stored_stays_in_memory_and_holds_the_scan_back(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, _account_a = self.logged_in_as_a(database, clock)
        self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))

        def fail_replace(*_args: object, **_kwargs: object) -> None:
            refused = OSError(28, "secret: no space")
            raise refused

        monkeypatch.setattr("usagedash.collector.os.replace", fail_replace)
        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            report = cycles.run_cycle()

        assert report.outcomes[0].files_read == 0
        assert scanner.resolved == []
        assert not self.boundary_file(database).exists()
        assert "Could not store the unrecorded login boundary" in caplog.text
        assert "secret" not in caplog.text

    def test_a_timeline_that_cannot_be_read_does_not_drop_the_stored_boundary(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        unrecorded_from = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        monkeypatch.undo()
        original = db.active_timeline

        def fail_once(*args: object, **kwargs: object) -> list[tuple[datetime, int | None]]:
            monkeypatch.setattr(db, "active_timeline", original)
            refused = sqlite3.OperationalError("secret: database is locked")
            raise refused

        monkeypatch.setattr(db, "active_timeline", fail_once)
        clock.advance(minutes=5)
        scanner.probe_times = [unrecorded_from, clock.now]
        fresh = collector(database, clock, SourceBundle(source, scanner, 0))

        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            fresh.run_cycle()

        assert "secret" not in caplog.text
        assert scanner.resolved == [0, self.account_b(database)]
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE)[:2] == [(now, account_a), (unrecorded_from, None)]

    def test_an_unreadable_boundary_file_holds_a_new_collector_until_it_can_be_read(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        unrecorded_from = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        note = self.boundary_file(database)
        original = note.read_bytes()
        scans_before = scanner.scans
        monkeypatch.undo()
        real_read_text = Path.read_text

        def refuse_boundary(path: Path, *args: object, **kwargs: object) -> str:
            if path.name == LOGIN_UNRECORDED_FILE_NAME:
                refused = PermissionError(13, "secret: denied")
                raise refused
            return real_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", refuse_boundary)
        clock.advance(minutes=5)
        fresh = collector(database, clock, SourceBundle(source, scanner, 0))

        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            report = fresh.run_cycle()

        assert report.outcomes[0].files_read == 0
        assert scanner.resolved == []
        assert scanner.scans == scans_before
        assert report.outcomes[0].quota_error == "UsageDashError: the unrecorded login boundary could not be read"
        assert report.outcomes[0].token_error == (
            "TokenScanError: not scanned: who is logged in there could not be recorded"
        )
        assert note.read_bytes() == original
        assert "Could not read the unrecorded-login file" in caplog.text
        assert "secret" not in caplog.text
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == [(now, account_a), (clock.now, None)]

        monkeypatch.undo()
        scanner.probe_times = [unrecorded_from - timedelta(seconds=1), unrecorded_from, clock.now]
        recovered = collector(database, clock, SourceBundle(source, scanner, 0))

        recovered.run_cycle()

        account_b = self.account_b(database)
        assert scanner.resolved == [account_a, 0, account_b]
        assert not note.exists()
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == [
                (now, account_a),
                (unrecorded_from, None),
                (clock.now, account_b),
            ]

    def test_a_newer_boundary_in_memory_survives_an_older_recorded_one_on_disk(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        first_gap = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        monkeypatch.undo()
        real_unlink = Path.unlink

        def fail_boundary_unlink(path: Path, missing_ok: bool = False) -> None:
            if path.name == LOGIN_UNRECORDED_FILE_NAME:
                refused = OSError(1, "secret: busy")
                raise refused
            real_unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", fail_boundary_unlink)
        clock.advance(minutes=5)
        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            cycles.run_cycle()
        note = self.boundary_file(database)
        assert note.read_text(encoding="utf-8")
        assert self.SOURCE not in cycles._login_unrecorded_since

        clock.advance(minutes=5)
        source.identity = make_identity(external_id="org-c", label="Account C")
        source.result = make_reading(source.identity, clock.now, make_window())
        later_gap = clock.now
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))

        def fail_replace(*_args: object, **_kwargs: object) -> None:
            refused = OSError(28, "secret: no space")
            raise refused

        monkeypatch.setattr("usagedash.collector.os.replace", fail_replace)
        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            held = cycles.run_cycle()

        assert held.outcomes[0].files_read == 0
        assert cycles._login_unrecorded_since[self.SOURCE] == later_gap
        assert json.loads(note.read_text(encoding="utf-8")) == {self.SOURCE: first_gap.isoformat()}
        assert "secret" not in caplog.text
        monkeypatch.undo()
        clock.advance(minutes=1)
        scanner.probe_times = [
            later_gap - timedelta(seconds=30),
            later_gap + timedelta(seconds=30),
            clock.now,
        ]
        account_b = self.account_b(database)

        cycles.run_cycle()

        with database.connect() as conn:
            account_c = next(account.id for account in db.list_accounts(conn) if account.label == "Account C")
            assert db.active_timeline(conn, self.SOURCE) == [
                (now, account_a),
                (first_gap, None),
                (first_gap + timedelta(minutes=5), account_b),
                (later_gap, None),
                (clock.now, account_c),
            ]
        assert scanner.resolved == [account_b, 0, account_c]
        assert not note.exists()

    def test_an_unrecorded_later_boundary_in_memory_is_kept_when_the_disk_note_is_already_recorded(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        _source, _scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        older = now + timedelta(minutes=1)
        account_from = now + timedelta(minutes=3)
        newer = now + timedelta(minutes=4)
        with database.connect() as conn:
            db.record_unrecorded_gap(conn, self.SOURCE, older)
            db.observe_active_account(conn, self.SOURCE, account_a, account_from)
        cycles._login_unrecorded_since[self.SOURCE] = newer
        self.boundary_file(database).write_text(json.dumps({self.SOURCE: older.isoformat()}), encoding="utf-8")

        cycles._restore_unrecorded_logins()

        assert cycles._login_unrecorded_since[self.SOURCE] == newer
        stored = json.loads(self.boundary_file(database).read_text(encoding="utf-8"))
        assert parse_iso(stored[self.SOURCE]) == newer

    def test_a_recorded_boundary_still_in_memory_is_cleared_without_opening_another(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, _account_a = self.logged_in_as_a(database, clock)
        first_gap = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        monkeypatch.undo()
        clock.advance(minutes=5)
        cycles.run_cycle()
        cycles._login_unrecorded_since[self.SOURCE] = first_gap
        note = self.boundary_file(database)
        note.write_text(json.dumps({self.SOURCE: first_gap.isoformat()}), encoding="utf-8")
        with database.connect() as conn:
            before = db.active_timeline(conn, self.SOURCE)
        clock.advance(minutes=1)
        scanner.probe_times = [clock.now]

        cycles.run_cycle()

        assert self.SOURCE not in cycles._login_unrecorded_since
        assert not note.exists()
        assert scanner.resolved == [self.account_b(database)]
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == before

    def test_a_boundary_inside_an_existing_nobody_interval_is_not_opened_again(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, _account_a = self.logged_in_as_a(database, clock)
        gap_start = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        monkeypatch.undo()
        clock.advance(minutes=5)
        cycles.run_cycle()
        inside = gap_start + timedelta(minutes=1)
        self.boundary_file(database).write_text(json.dumps({self.SOURCE: inside.isoformat()}), encoding="utf-8")
        with database.connect() as conn:
            before = db.active_timeline(conn, self.SOURCE)
        covered_until = before[-1][0]
        clock.advance(minutes=1)
        scanner.probe_times = [covered_until + timedelta(seconds=30)]
        fresh = collector(database, clock, SourceBundle(source, scanner, 0))

        fresh.run_cycle()

        assert scanner.resolved == [self.account_b(database)]
        assert not self.boundary_file(database).exists()
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == before

    def test_a_short_write_still_stores_the_whole_boundary(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, _scanner, cycles, _account_a = self.logged_in_as_a(database, clock)
        unrecorded_from = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        calls = 0
        real_write = os.write

        def one_byte(descriptor: int, data: bytes | memoryview) -> int:
            nonlocal calls
            calls += 1
            return real_write(descriptor, data[:1])

        monkeypatch.setattr("usagedash.collector.os.write", one_byte)

        cycles.run_cycle()

        note = self.boundary_file(database)
        assert json.loads(note.read_text(encoding="utf-8")) == {self.SOURCE: unrecorded_from.isoformat()}
        assert calls == note.stat().st_size
        assert calls > 1

    def test_a_zero_byte_write_leaves_no_boundary_file_and_keeps_the_note(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        unrecorded_from = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))

        def zero(_descriptor: int, _data: object) -> int:
            return 0

        monkeypatch.setattr("usagedash.collector.os.write", zero)
        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            report = cycles.run_cycle()

        note = self.boundary_file(database)
        assert report.outcomes[0].files_read == 0
        assert scanner.resolved == []
        assert not note.exists()
        assert list(note.parent.glob(f".{note.name}.*.tmp")) == []
        assert "Could not store the unrecorded login boundary" in caplog.text
        assert cycles._login_unrecorded_since[self.SOURCE] == unrecorded_from
        monkeypatch.undo()
        clock.advance(minutes=5)
        scanner.probe_times = [unrecorded_from, clock.now]

        cycles.run_cycle()

        assert scanner.resolved == [0, self.account_b(database)]
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE)[:2] == [(now, account_a), (unrecorded_from, None)]

    def test_an_unreadable_file_shared_by_two_sources_is_not_replaced_when_one_recovers(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock(now)
        claude = FakeSource("claude:home")
        claude.identity = make_identity(label="Claude A")
        claude.result = make_reading(claude.identity, clock.now, make_window())
        claude_scanner = FakeScanner()
        codex = FakeSource("codex:home", Provider.CODEX)
        codex.identity = make_identity(Provider.CODEX, "ws-a", "Codex A")
        codex.result = make_reading(codex.identity, clock.now, make_window())
        codex_scanner = FakeScanner()
        cycles = collector(
            database, clock, SourceBundle(claude, claude_scanner, 0), SourceBundle(codex, codex_scanner, 0)
        )
        cycles.run_cycle()
        codex_a = self.account_named(database, "Codex A")
        clock.advance(minutes=5)
        gap = clock.now
        claude.identity = make_identity(external_id="org-b", label="Claude B")
        claude.result = make_reading(claude.identity, gap, make_window())
        codex.identity = make_identity(Provider.CODEX, "ws-b", "Codex B")
        codex.result = make_reading(codex.identity, gap, make_window())
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        note = self.boundary_file(database)
        assert json.loads(note.read_text(encoding="utf-8")) == {
            "claude:home": gap.isoformat(),
            "codex:home": gap.isoformat(),
        }
        original = note.read_bytes()
        del cycles._login_unrecorded_since["codex:home"]
        monkeypatch.undo()
        self.refuse_boundary_reads(monkeypatch)
        clock.advance(minutes=5)
        claude.result = make_reading(claude.identity, clock.now, make_window())
        codex.result = make_reading(codex.identity, clock.now, make_window())
        claude_scanner.probe_times = [gap, clock.now]
        claude_scans = claude_scanner.scans
        codex_scans = codex_scanner.scans

        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            report = cycles.run_cycle()

        assert note.read_bytes() == original
        cycles._persist_unrecorded_logins()
        assert note.read_bytes() == original
        assert claude_scanner.scans == claude_scans
        assert claude_scanner.resolved == []
        assert report.outcomes[0].quota_error == "UsageDashError: the unrecorded login boundary could not be read"
        assert report.outcomes[0].token_error == (
            "TokenScanError: not scanned: who is logged in there could not be recorded"
        )
        assert report.outcomes[0].files_read == 0
        assert codex_scanner.scans == codex_scans
        assert codex_scanner.resolved == []
        assert report.outcomes[1].quota_error == "UsageDashError: the unrecorded login boundary could not be read"
        assert report.outcomes[1].files_read == 0
        assert "secret" not in caplog.text
        monkeypatch.undo()
        codex_scanner.probe_times = [gap - timedelta(seconds=1), gap, clock.now]
        recovered = collector(
            database, clock, SourceBundle(claude, claude_scanner, 0), SourceBundle(codex, codex_scanner, 0)
        )

        recovered.run_cycle()

        codex_b = self.account_named(database, "Codex B")
        assert codex_scanner.resolved == [codex_a, 0, codex_b]
        assert not note.exists()
        with database.connect() as conn:
            assert db.active_timeline(conn, "codex:home") == [
                (now, codex_a),
                (gap, None),
                (clock.now, codex_b),
            ]

    def test_a_gap_recorded_by_another_collector_is_dropped_when_the_file_is_gone(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock, _source, scanner, cycles, _first_gap, recorded = self.gap_recorded_by_another_collector(
            database, now, monkeypatch
        )
        clock.advance(minutes=2)
        scanner.probe_times = [clock.now - timedelta(seconds=30)]
        account_b = self.account_b(database)

        cycles.run_cycle()

        assert self.SOURCE not in cycles._login_unrecorded_since
        assert scanner.resolved == [account_b]
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == recorded

    def test_a_gap_recorded_by_another_collector_is_dropped_when_the_file_omits_it(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock, _source, scanner, cycles, first_gap, recorded = self.gap_recorded_by_another_collector(
            database, now, monkeypatch
        )
        other_moment = (first_gap + timedelta(minutes=3)).isoformat()
        self.boundary_file(database).write_text(json.dumps({"codex:other": other_moment}), encoding="utf-8")
        clock.advance(minutes=2)
        scanner.probe_times = [clock.now - timedelta(seconds=30)]
        account_b = self.account_b(database)

        cycles.run_cycle()

        assert self.SOURCE not in cycles._login_unrecorded_since
        assert scanner.resolved == [account_b]
        assert json.loads(self.boundary_file(database).read_text(encoding="utf-8")) == {"codex:other": other_moment}
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == recorded

    def test_a_recorded_note_in_memory_does_not_open_a_source_whose_file_cannot_be_read(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _clock, _source, scanner, cycles, first_gap, recorded = self.gap_recorded_by_another_collector(
            database, now, monkeypatch
        )
        payload = json.dumps({"codex:other": (first_gap + timedelta(minutes=3)).isoformat()}).encode()
        note = self.boundary_file(database)
        note.write_bytes(payload)
        self.refuse_boundary_reads(monkeypatch)
        scans_before = scanner.scans

        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            report = cycles.run_cycle()

        assert note.read_bytes() == payload
        assert report.outcomes[0].files_read == 0
        assert report.outcomes[0].quota_error == "UsageDashError: the unrecorded login boundary could not be read"
        assert scanner.scans == scans_before
        assert "secret" not in caplog.text
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == recorded

    def test_a_boundary_file_that_is_not_utf_8_holds_every_source_and_stays_in_place(
        self, database: Database, now: datetime, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock(now)
        claude = FakeSource("claude:home")
        claude.identity = make_identity(label="Claude A")
        claude.result = make_reading(claude.identity, clock.now, make_window())
        codex = FakeSource("codex:home", Provider.CODEX)
        codex.identity = make_identity(Provider.CODEX, "ws-a", "Codex A")
        codex.result = make_reading(codex.identity, clock.now, make_window())
        note = self.boundary_file(database)
        note.write_bytes(b"\xff")
        loaded = load_unrecorded_logins(note)
        fresh = collector(
            database, clock, SourceBundle(claude, FakeScanner(), 0), SourceBundle(codex, FakeScanner(), 0)
        )

        with caplog.at_level(logging.ERROR, logger="usagedash.collector"):
            report = fresh.run_cycle()

        assert (loaded.parsed, loaded.unreadable, loaded.logins) == (False, True, {})
        assert note.read_bytes() == b"\xff"
        assert [outcome.quota_error for outcome in report.outcomes] == [
            "UsageDashError: the unrecorded login boundary could not be read",
            "UsageDashError: the unrecorded login boundary could not be read",
        ]
        assert [outcome.files_read for outcome in report.outcomes] == [0, 0]
        assert "Could not read the unrecorded-login file" in caplog.text
        with database.connect() as conn:
            assert db.active_timeline(conn, "claude:home") == []
            assert db.active_timeline(conn, "codex:home") == []

    @pytest.mark.parametrize(
        "contents",
        ["1" * 5000, "[" * 20_000 + "]" * 20_000],
        ids=["oversized-integer", "excessive-nesting"],
    )
    def test_a_boundary_file_whose_json_cannot_be_parsed_is_replaced(
        self, database: Database, now: datetime, contents: str
    ) -> None:
        clock = FakeClock(now)
        source = FakeSource(self.SOURCE)
        source.identity = make_identity(label="Account A")
        source.result = make_reading(source.identity, clock.now, make_window())
        note = self.boundary_file(database)
        note.write_text(contents, encoding="utf-8")
        loaded = load_unrecorded_logins(note)

        report = collector(database, clock, SourceBundle(source, FakeScanner(), 0)).run_cycle()

        assert (loaded.parsed, loaded.unreadable, loaded.logins) == (True, False, {})
        assert report.outcomes[0].quota_error is None
        assert report.outcomes[0].files_read == 2
        assert not note.exists()
        with database.connect() as conn:
            timeline = db.active_timeline(conn, self.SOURCE)
            assert [account.label for account in db.list_accounts(conn)] == ["Account A"]
        assert timeline == [(now, timeline[0][1])]

    def test_a_timestamp_that_cannot_be_a_timeline_boundary_is_skipped(self, database: Database, now: datetime) -> None:
        clock = FakeClock(now)
        source = FakeSource(self.SOURCE)
        source.identity = make_identity(label="Account A")
        source.result = make_reading(source.identity, clock.now, make_window())
        unroundable = "9999-12-31T23:59:59.000001+00:00"
        kept = (now + timedelta(minutes=1)).isoformat()
        note = self.boundary_file(database)
        note.write_text(json.dumps({self.SOURCE: unroundable, "claude:other": kept}), encoding="utf-8")
        with pytest.raises(ValueError):
            whole_second_not_before(parse_iso(unroundable))
        loaded = load_unrecorded_logins(note)
        note.write_text(json.dumps({self.SOURCE: unroundable}), encoding="utf-8")

        report = collector(database, clock, SourceBundle(source, FakeScanner(), 0)).run_cycle()

        assert self.SOURCE not in loaded.logins
        assert list(loaded.logins) == ["claude:other"]
        assert report.outcomes[0].quota_error is None
        assert report.outcomes[0].files_read == 2
        assert not note.exists()
        with database.connect() as conn:
            timeline = db.active_timeline(conn, self.SOURCE)
        assert len(timeline) == 1
        assert timeline[0][0] == now

    def test_a_gap_whose_start_ties_the_recovered_login_stays_recorded_when_the_file_returns(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        failure_at = now + timedelta(minutes=5, microseconds=100_000)
        recovery_at = now + timedelta(minutes=5, microseconds=200_000)
        self.fail_to_record(source, clock, cycles, monkeypatch, failure_at)
        self.recover_while_the_boundary_file_stays(
            database, clock, source, cycles, monkeypatch, failure_at, recovery_at
        )
        gap_start = now + timedelta(minutes=5, seconds=1)
        account_b = self.account_b(database)
        recorded = [(now, account_a), (gap_start, None), (gap_start, account_b)]
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == recorded

        self.replay_must_keep_the_recovered_account(
            database,
            clock,
            source,
            scanner,
            now + timedelta(minutes=6),
            now + timedelta(minutes=5, seconds=30),
            recorded,
        )

    def test_a_gap_clamped_after_its_boundary_stays_recorded_when_the_file_returns(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        clock.now = now + timedelta(minutes=10)
        source.result = make_reading(source.identity, clock.now, make_window())
        cycles.run_cycle()
        failure_at = now + timedelta(minutes=5, microseconds=100_000)
        recovery_at = now + timedelta(minutes=5, microseconds=200_000)
        self.fail_to_record(source, clock, cycles, monkeypatch, failure_at)
        self.recover_while_the_boundary_file_stays(
            database, clock, source, cycles, monkeypatch, failure_at, recovery_at
        )
        clamped = now + timedelta(minutes=10)
        account_b = self.account_b(database)
        recorded = [(now, account_a), (clamped, None), (clamped, account_b)]
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == recorded

        self.replay_must_keep_the_recovered_account(
            database,
            clock,
            source,
            scanner,
            now + timedelta(minutes=11),
            now + timedelta(minutes=10, seconds=30),
            recorded,
        )

    def test_a_rolled_back_login_does_not_acknowledge_its_boundary(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, _scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        failure_at = now + timedelta(minutes=5, microseconds=100_000)
        self.fail_to_record(source, clock, cycles, monkeypatch, failure_at)
        self.refusing(monkeypatch, "set_source_state", sqlite3.OperationalError("database is locked"))
        clock.now = failure_at + timedelta(microseconds=100_000)
        source.result = make_reading(source.identity, clock.now, make_window())

        cycles.run_cycle()

        assert cycles._login_unrecorded_since[self.SOURCE] == failure_at
        assert not cycles._gap_is_recorded(self.SOURCE, failure_at)
        with database.connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM recorded_login_boundaries").fetchone()[0] == 0
            assert db.active_timeline(conn, self.SOURCE) == [(now, account_a)]

    def fail_to_record(
        self,
        source: FakeSource,
        clock: FakeClock,
        cycles: Collector,
        monkeypatch: pytest.MonkeyPatch,
        failure_at: datetime,
    ) -> None:
        clock.now = failure_at
        source.identity = make_identity(external_id="org-b", label="Account B")
        source.result = make_reading(source.identity, failure_at, make_window())
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        monkeypatch.undo()
        assert cycles._login_unrecorded_since[self.SOURCE] == failure_at
        assert json.loads(self.boundary_file(cycles._database).read_text(encoding="utf-8")) == {
            self.SOURCE: in_utc(failure_at).isoformat()
        }
        with cycles._database.connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM recorded_login_boundaries").fetchone()[0] == 0

    def recover_while_the_boundary_file_stays(
        self,
        database: Database,
        clock: FakeClock,
        source: FakeSource,
        cycles: Collector,
        monkeypatch: pytest.MonkeyPatch,
        failure_at: datetime,
        recovery_at: datetime,
    ) -> None:
        real_unlink = Path.unlink

        def fail_boundary_unlink(path: Path, missing_ok: bool = False) -> None:
            if path.name == LOGIN_UNRECORDED_FILE_NAME:
                refused = OSError(1, "busy")
                raise refused
            real_unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", fail_boundary_unlink)
        clock.now = recovery_at
        source.result = make_reading(source.identity, recovery_at, make_window())
        cycles.run_cycle()
        monkeypatch.undo()
        assert self.SOURCE not in cycles._login_unrecorded_since
        assert cycles._gap_is_recorded(self.SOURCE, failure_at)
        assert json.loads(self.boundary_file(database).read_text(encoding="utf-8")) == {
            self.SOURCE: in_utc(failure_at).isoformat()
        }
        with database.connect() as conn:
            stored = conn.execute(
                "SELECT boundary_at FROM recorded_login_boundaries WHERE source_id = ?",
                (self.SOURCE,),
            ).fetchall()
        assert [row["boundary_at"] for row in stored] == [in_utc(failure_at).isoformat()]

    def replay_must_keep_the_recovered_account(
        self,
        database: Database,
        clock: FakeClock,
        source: FakeSource,
        scanner: FakeScanner,
        replay_at: datetime,
        probe_at: datetime,
        recorded: list[tuple[datetime, int | None]],
    ) -> None:
        clock.now = replay_at
        source.result = make_reading(source.identity, replay_at, make_window())
        scanner.probe_times = [probe_at]
        account_b = self.account_b(database)

        collector(database, clock, SourceBundle(source, scanner, 0)).run_cycle()

        assert scanner.resolved == [account_b]
        assert not self.boundary_file(database).exists()
        with database.connect() as conn:
            assert db.active_timeline(conn, self.SOURCE) == recorded

    def gap_recorded_by_another_collector(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[FakeClock, FakeSource, FakeScanner, Collector, datetime, list[tuple[datetime, int | None]]]:
        clock = FakeClock(now)
        source, scanner, cycles, _account_a = self.logged_in_as_a(database, clock)
        first_gap = self.b_is_seen(source, clock)
        self.refusing(monkeypatch, "observe_active_account", sqlite3.OperationalError("database is locked"))
        cycles.run_cycle()
        monkeypatch.undo()
        clock.advance(minutes=5)
        collector(database, clock, SourceBundle(source, scanner, 0)).run_cycle()
        assert not self.boundary_file(database).exists()
        assert cycles._login_unrecorded_since[self.SOURCE] == first_gap
        with database.connect() as conn:
            recorded = db.active_timeline(conn, self.SOURCE)
        return clock, source, scanner, cycles, first_gap, recorded

    def account_named(self, database: Database, label: str) -> int:
        with database.connect() as conn:
            return next(account.id for account in db.list_accounts(conn) if account.label == label)

    def refuse_boundary_reads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_read_text = Path.read_text

        def refuse_boundary(path: Path, *args: object, **kwargs: object) -> str:
            if path.name == LOGIN_UNRECORDED_FILE_NAME:
                refused = PermissionError(13, "secret: denied")
                raise refused
            return real_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", refuse_boundary)

    def boundary_file(self, database: Database) -> Path:
        return database.path.with_name(LOGIN_UNRECORDED_FILE_NAME)

    @pytest.mark.parametrize("restart", [False, True], ids=["same collector", "restarted collector"])
    @pytest.mark.parametrize("logged_out", [False, True], ids=["switched account", "logged out"])
    def test_a_blocked_recovery_file_does_not_leave_the_span_on_the_preceding_account(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch, restart: bool, logged_out: bool
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        self.boundary_file(database).write_text("{}", encoding="utf-8")
        self.refuse_boundary_reads(monkeypatch)
        if logged_out:
            clock.advance(minutes=5)
            source.identity = None
            source.result = UsageDashError("not logged in")
            blocked_at = clock.now
        else:
            blocked_at = self.b_is_seen(source, clock)
        cycles.run_cycle()
        monkeypatch.undo()
        self.boundary_file(database).write_text(json.dumps({"other:home": now.isoformat()}), encoding="utf-8")
        clock.advance(minutes=5)
        source.identity = make_identity(external_id="org-b", label="Account B")
        source.result = make_reading(source.identity, clock.now, make_window())
        scanner.probe_times = [blocked_at + timedelta(minutes=1)]
        runner = collector(database, clock, SourceBundle(source, scanner, 0)) if restart else cycles

        runner.run_cycle()

        assert scanner.resolved == [0]
        with database.connect() as conn:
            timeline = db.active_timeline(conn, self.SOURCE)
        assert timeline[0] == (now, account_a)
        assert timeline[1] == (blocked_at, None)
        assert self.boundary_file(database).read_text(encoding="utf-8").startswith("{")

    def test_a_timeline_that_cannot_be_read_while_the_recovery_file_is_blocked_keeps_the_boundary(
        self, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = FakeClock(now)
        source, scanner, cycles, account_a = self.logged_in_as_a(database, clock)
        self.boundary_file(database).write_text("{}", encoding="utf-8")
        self.refuse_boundary_reads(monkeypatch)
        blocked_at = self.b_is_seen(source, clock)

        def fail_timeline(*_args: object, **_kwargs: object) -> list[tuple[datetime, int | None]]:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(db, "active_timeline", fail_timeline)
        report = cycles.run_cycle()
        monkeypatch.undo()

        assert report.outcomes[0].quota_error == "UsageDashError: the unrecorded login boundary could not be read"
        with database.connect() as conn:
            timeline = db.active_timeline(conn, self.SOURCE)
        assert timeline[0] == (now, account_a)
        assert timeline[1] == (blocked_at, None)
        self.boundary_file(database).write_text(json.dumps({"other:home": now.isoformat()}), encoding="utf-8")
        clock.advance(minutes=5)
        source.identity = make_identity(external_id="org-b", label="Account B")
        source.result = make_reading(source.identity, clock.now, make_window())
        scanner.probe_times = [blocked_at + timedelta(minutes=1)]

        cycles.run_cycle()

        assert scanner.resolved == [0]


class TestASourceThatStoppedScanning:
    def test_the_token_error_it_last_recorded_is_cleared_when_it_no_longer_has_a_scanner(
        self, database: Database, now: datetime
    ) -> None:
        clock = FakeClock(now)
        source = FakeSource("claude:home")
        source.result = make_reading(make_identity(), now, make_window())
        failing = FakeScanner(ScanReport(files_failed=1, first_failure="a.jsonl: PermissionError"))
        collector(database, clock, SourceBundle(source, failing, 0)).run_cycle()
        assert statuses(database)[("claude:home", "tokens")][0] == "TokenScanError"

        report = collector(database, clock, SourceBundle(source, None, 0)).run_cycle()

        assert ("claude:home", "tokens") not in statuses(database)
        assert statuses(database)[("claude:home", "quota")] == (None, None)
        assert (report.outcomes[0].token_error, report.outcomes[0].files_read) == (None, 0)

    def test_a_source_that_never_had_a_scanner_has_no_token_status(self, database: Database, now: datetime) -> None:
        source = FakeSource("claude:home")
        source.result = make_reading(make_identity(), now, make_window())

        collector(database, FakeClock(now), SourceBundle(source, None, 0)).run_cycle()

        assert set(statuses(database)) == {("claude:home", "quota")}


class TestBackoffWithALongPollInterval:
    @pytest.mark.parametrize(
        ("poll_seconds", "retry_after", "expected_seconds"),
        [
            (300, None, 600),
            (43_200, None, 86_400),
            (43_201, None, 86_400),
            (86_400, None, 86_400),
            (86_400, 120.0, 86_400),
            (86_400, 1e300, 86_400),
            (300, 86_400.0, 86_400),
            (300, 900.0, 900),
        ],
    )
    def test_the_wait_never_exceeds_a_day_whichever_of_the_two_would(
        self, database: Database, now: datetime, poll_seconds: int, retry_after: float | None, expected_seconds: int
    ) -> None:
        runner = Collector(database, [], FakeClock(now), poll_interval_seconds=poll_seconds, backfill_days=35)

        assert runner._backoff(retry_after) == timedelta(seconds=expected_seconds)
