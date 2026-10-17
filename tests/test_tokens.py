"""The token report: stored hourly usage turned into chart buckets in the viewer's time zone."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from tests.factories import make_identity
from tests.seeds import seed_accounts_and_usage
from usagedash import db
from usagedash.db import Database
from usagedash.models import UNATTRIBUTED_ACCOUNT_ID, Provider, TokenCounts, UsageKey
from usagedash.tokens import RangeKey, build_token_report


@pytest.fixture
def seeded(database: Database, now: datetime) -> dict[str, int]:
    return seed_accounts_and_usage(database, now)


class TestTokenReport:
    def test_daily_buckets_follow_the_viewer_s_calendar_day_not_the_utc_day(
        self, database: Database, seeded: dict[str, int], now: datetime
    ) -> None:
        with database.connect() as conn:
            report = build_token_report(conn, db.list_accounts(conn), "7d", ZoneInfo("America/New_York"), now)

        assert [bucket["key"] for bucket in report["buckets"]] == [
            "2026-09-27",
            "2026-09-28",
            "2026-09-29",
            "2026-09-30",
            "2026-10-01",
            "2026-10-02",
            "2026-10-03",
        ]
        assert report["buckets"][-1]["start"] == "2026-10-03T04:00:00+00:00"
        active = next(series for series in report["series"] if series["account_id"] == seeded["active"])
        assert [point["total"] for point in active["points"]] == [0, 0, 0, 0, 0, 2, 135]
        assert active["totals"] == {
            "input": 11,
            "cache_read": 100,
            "cache_write": 5,
            "output": 21,
            "reasoning": 3,
            "requests": 3,
            "total": 137,
        }

    def test_the_same_usage_lands_on_a_different_day_in_another_zone(
        self, database: Database, seeded: dict[str, int], now: datetime
    ) -> None:
        with database.connect() as conn:
            report = build_token_report(conn, db.list_accounts(conn), "7d", ZoneInfo("UTC"), now)

        active = next(series for series in report["series"] if series["account_id"] == seeded["active"])
        assert [point["total"] for point in active["points"]][-2:] == [0, 137]

    def test_hourly_buckets_cover_exactly_the_last_twenty_four_hours(
        self, database: Database, seeded: dict[str, int], now: datetime
    ) -> None:
        with database.connect() as conn:
            report = build_token_report(conn, db.list_accounts(conn), "24h", ZoneInfo("UTC"), now)

        assert report["bucket"] == "hour"
        assert len(report["buckets"]) == 24
        assert (report["buckets"][0]["key"], report["buckets"][-1]["key"]) == ("2026-10-02T15", "2026-10-03T14")
        series = {item["account_id"]: sum(point["total"] for point in item["points"]) for item in report["series"]}
        assert series == {seeded["active"]: 137}

    def test_each_series_has_buckets_of_its_own_and_rows_of_one_series_add_up_in_them(
        self, database: Database, now: datetime
    ) -> None:
        """A series' buckets are built when its first row is met; later rows and other series must not share them."""
        with database.connect() as conn:
            first = db.upsert_account(conn, make_identity(Provider.CODEX, "workspace-a"), now)
            second = db.upsert_account(conn, make_identity(Provider.CODEX, "workspace-b", "Account B"), now)
            usage = {
                UsageKey(first, "2026-10-03T13", "model-x"): TokenCounts(input_tokens=5),
                UsageKey(first, "2026-10-03T14", "model-x"): TokenCounts(input_tokens=7),
                UsageKey(first, "2026-10-03T14", "model-y"): TokenCounts(input_tokens=2),
                UsageKey(second, "2026-10-03T14", "model-x"): TokenCounts(input_tokens=11),
            }
            db.add_file_usage(conn, Provider.CODEX, "rollout", usage)
            report = build_token_report(conn, db.list_accounts(conn), "24h", ZoneInfo("UTC"), now)

        last_two_hours = {
            series["account_id"]: [point["total"] for point in series["points"]][-2:] for series in report["series"]
        }
        assert last_two_hours == {first: [5, 9], second: [0, 11]}
        assert all(sum(point["total"] for point in series["points"][:-2]) == 0 for series in report["series"])

    def test_unattributed_usage_is_its_own_series_listed_after_real_accounts(
        self, database: Database, seeded: dict[str, int], now: datetime
    ) -> None:
        with database.connect() as conn:
            report = build_token_report(conn, db.list_accounts(conn), "30d", ZoneInfo("UTC"), now)

        ordering = [(series["account_id"], series["attributed"], series["color_slot"]) for series in report["series"]]
        assert ordering == [
            (seeded["active"], True, 1),
            (seeded["other"], True, 2),
            (UNATTRIBUTED_ACCOUNT_ID, False, 0),
        ]
        assert report["series"][-1]["label"] == "Unattributed (before tracking)"
        totals = {item["provider"]: item["totals"]["total"] for item in report["provider_totals"]}
        assert totals == {"claude": 137 + 10 + 1050, "codex": 0, "grok": 0}
        assert {(row["account_id"], row["model"]) for row in report["models"]} == {
            (seeded["active"], "model-x"),
            (seeded["other"], "model-y"),
            (UNATTRIBUTED_ACCOUNT_ID, "model-x"),
        }

    def test_an_hour_straddling_local_midnight_is_counted_once_by_every_total(
        self, database: Database, now: datetime
    ) -> None:
        """India is UTC+05:30, so the first day shown starts at 18:30 UTC, in the middle of a stored hour."""
        with database.connect() as conn:
            db.add_file_usage(
                conn,
                Provider.GROK,
                "file",
                {
                    UsageKey(UNATTRIBUTED_ACCOUNT_ID, "2026-09-26T18", "model-x"): TokenCounts(input_tokens=100),
                    UsageKey(UNATTRIBUTED_ACCOUNT_ID, "2026-09-26T19", "model-x"): TokenCounts(input_tokens=7),
                },
            )
            report = build_token_report(conn, [], "7d", ZoneInfo("Asia/Kolkata"), now)

        assert report["buckets"][0] == {"key": "2026-09-27", "start": "2026-09-26T18:30:00+00:00"}
        assert [point["total"] for point in report["series"][0]["points"]] == [7, 0, 0, 0, 0, 0, 0]
        assert [row["totals"]["total"] for row in report["models"]] == [7]
        assert {item["provider"]: item["totals"]["total"] for item in report["provider_totals"]}["grok"] == 7

    @pytest.mark.parametrize("range_key", ["24h", "7d", "30d"])
    @pytest.mark.parametrize("zone", ["UTC", "America/New_York", "Asia/Kolkata", "Australia/Eucla", "Asia/Kathmandu"])
    def test_series_model_and_provider_totals_always_agree(
        self, database: Database, seeded: dict[str, int], now: datetime, range_key: RangeKey, zone: str
    ) -> None:
        with database.connect() as conn:
            report = build_token_report(conn, db.list_accounts(conn), range_key, ZoneInfo(zone), now)

        def grand_total(rows: list[dict[str, dict[str, int]]]) -> dict[str, int]:
            return {
                measure: sum(row["totals"][measure] for row in rows)
                for measure in ("input", "cache_read", "cache_write", "output", "reasoning", "requests", "total")
            }

        assert grand_total(report["series"]) == grand_total(report["models"]) == grand_total(report["provider_totals"])
        for series in report["series"]:
            assert sum(point["total"] for point in series["points"]) == series["totals"]["total"]

    def test_models_are_listed_by_provider_then_account_with_unattributed_last(
        self, database: Database, seeded: dict[str, int], now: datetime
    ) -> None:
        with database.connect() as conn:
            report = build_token_report(conn, db.list_accounts(conn), "30d", ZoneInfo("UTC"), now)

        assert [(row["account_id"], row["model"], row["totals"]["total"]) for row in report["models"]] == [
            (seeded["active"], "model-x", 137),
            (seeded["other"], "model-y", 10),
            (UNATTRIBUTED_ACCOUNT_ID, "model-x", 1050),
        ]


class TestTheMomentTheReportIsBuiltFor:
    def test_a_naive_now_is_refused_instead_of_being_read_in_this_machine_s_time_zone(self, database: Database) -> None:
        with (
            pytest.raises(ValueError, match="a naive datetime has no defined instant"),
            database.connect() as conn,
        ):
            build_token_report(conn, [], "24h", ZoneInfo("UTC"), datetime(2026, 10, 3, 14, 0))

    @pytest.mark.parametrize("range_key", ["24h", "7d", "30d"])
    def test_the_report_is_the_same_whatever_offset_now_is_written_in(
        self, database: Database, seeded: dict[str, int], now: datetime, range_key: RangeKey
    ) -> None:
        elsewhere = now.astimezone(timezone(timedelta(hours=-9, minutes=-30)))
        zone = ZoneInfo("Asia/Kolkata")
        with database.connect() as conn:
            accounts = db.list_accounts(conn)
            expected = build_token_report(conn, accounts, range_key, zone, now)
            actual = build_token_report(conn, accounts, range_key, zone, elsewhere)

        assert actual == expected

    def test_hourly_buckets_end_at_the_current_utc_hour_when_now_is_in_a_zone_with_a_clock_change(
        self, database: Database
    ) -> None:
        """The hour 01:00 to 02:00 happens twice in New York on 2026-11-01; the buckets must not skip or repeat one."""
        second_occurrence = datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=1)
        with database.connect() as conn:
            report = build_token_report(conn, [], "24h", ZoneInfo("UTC"), second_occurrence)

        keys = [bucket["key"] for bucket in report["buckets"]]
        assert keys[-1] == "2026-11-01T06"
        assert keys[0] == "2026-10-31T07"
        assert len(set(keys)) == 24
