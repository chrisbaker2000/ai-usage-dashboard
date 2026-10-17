import errno
import fcntl
import itertools
import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tests.factories import make_identity, make_reading, make_window
from usagedash import db
from usagedash.db import Database, FileCursor, MessageUsage, SeenMessage
from usagedash.errors import ConfigError, StoreError
from usagedash.models import (
    MAX_TOKEN_COUNT,
    UNATTRIBUTED_ACCOUNT_ID,
    Fact,
    Provider,
    QuotaReading,
    TokenCounts,
    UsageKey,
)
from usagedash.scanners.base import build_account_resolver
from usagedash.timeutil import to_iso


class TestAccountRegistration:
    def test_a_new_account_gets_the_next_categorical_slot_in_creation_order(
        self, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn:
            ids = [db.upsert_account(conn, make_identity(external_id=f"org-{index}"), now) for index in range(3)]
            accounts = db.list_accounts(conn)

        assert [account.id for account in accounts] == ids
        assert [account.color_slot for account in accounts] == [1, 2, 3]

    def test_the_ninth_account_shares_the_neutral_slot_instead_of_reusing_a_color(
        self, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn:
            for index in range(9):
                db.upsert_account(conn, make_identity(external_id=f"org-{index}"), now)
            slots = [account.color_slot for account in db.list_accounts(conn)]

        assert slots == [1, 2, 3, 4, 5, 6, 7, 8, 0]

    def test_seeing_the_account_again_refreshes_fields_without_creating_a_duplicate(
        self, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn:
            first = db.upsert_account(conn, make_identity(label="Old name", plan=None), now)
            second = db.upsert_account(conn, make_identity(label="New name", plan="Team"), now)
            accounts = db.list_accounts(conn)

        assert first == second
        assert len(accounts) == 1
        assert (accounts[0].label, accounts[0].plan) == ("New name", "Team")

    def test_a_label_the_user_set_survives_later_sightings(self, database: Database, now: datetime) -> None:
        with database.connect() as conn:
            account_id = db.upsert_account(conn, make_identity(label="Derived"), now)
            assert db.rename_account(conn, account_id, "Work Max")
            db.upsert_account(conn, make_identity(label="Derived again"), now)
            account = db.list_accounts(conn)[0]

        assert account.label == "Work Max"
        assert account.label_is_custom

    def test_renaming_an_unknown_account_reports_failure(self, database: Database) -> None:
        with database.connect() as conn:
            assert not db.rename_account(conn, 999, "Nobody")

    def test_an_account_s_label_can_be_read_by_id_and_an_unknown_id_has_none(
        self, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn:
            account_id = db.upsert_account(conn, make_identity(label="Account A"), now)

            assert db.account_label(conn, account_id) == "Account A"
            assert db.account_label(conn, account_id + 1) is None

    def test_an_account_is_never_live_until_marked_and_then_remembers_when(
        self, database: Database, now: datetime
    ) -> None:
        later = now + timedelta(minutes=5)
        with database.connect() as conn:
            seen = db.upsert_account(conn, make_identity(external_id="seen"), now)
            db.upsert_account(conn, make_identity(external_id="never-live"), now)
            db.mark_account_live(conn, seen, now)
            db.mark_account_live(conn, seen, later)
            accounts = {account.external_id: account.last_live_at for account in db.list_accounts(conn)}

        assert accounts == {"seen": later, "never-live": None}

    def test_the_same_external_id_under_another_provider_is_a_different_account(
        self, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn:
            claude = db.upsert_account(conn, make_identity(Provider.CLAUDE, "shared-id"), now)
            codex = db.upsert_account(conn, make_identity(Provider.CODEX, "shared-id"), now)
            assert db.find_account_id(conn, Provider.CODEX, "shared-id") == codex
            assert db.find_account_id(conn, Provider.GROK, "shared-id") is None

        assert claude != codex

    @pytest.mark.parametrize(
        ("stored", "incoming", "expected"),
        [
            ("Max 20x", "Max", "Max 20x"),
            ("Max", "Max 20x", "Max 20x"),
            ("Max 20x", None, "Max 20x"),
            ("Pro", "Team", "Team"),
            (None, "Pro", "Pro"),
        ],
    )
    def test_a_less_specific_plan_name_never_replaces_a_more_specific_one(
        self, database: Database, now: datetime, stored: str | None, incoming: str | None, expected: str
    ) -> None:
        with database.connect() as conn:
            db.upsert_account(conn, make_identity(plan=stored), now)
            db.upsert_account(conn, make_identity(plan=incoming), now)
            assert db.list_accounts(conn)[0].plan == expected


class TestQuotaSnapshots:
    def record(self, database: Database, reading: QuotaReading) -> int:
        with database.connect() as conn:
            account_id = db.upsert_account(conn, reading.identity, reading.observed_at)
            db.record_reading(conn, account_id, reading)
        return account_id

    def test_an_unchanged_value_extends_the_existing_step_instead_of_adding_a_row(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity()
        reset = now + timedelta(hours=4)
        self.record(database, make_reading(identity, now, make_window(used_percent=10, resets_at=reset)))
        self.record(
            database, make_reading(identity, now + timedelta(minutes=5), make_window(used_percent=10, resets_at=reset))
        )

        with database.connect() as conn:
            history = db.snapshot_history(conn, now - timedelta(days=1))

        assert len(history) == 1
        assert history[0].observed_at == now
        assert history[0].confirmed_at == now + timedelta(minutes=5)

    def test_a_changed_value_starts_a_new_step_and_keeps_the_old_one(self, database: Database, now: datetime) -> None:
        identity = make_identity()
        self.record(database, make_reading(identity, now, make_window(used_percent=10)))
        self.record(database, make_reading(identity, now + timedelta(minutes=5), make_window(used_percent=12)))

        with database.connect() as conn:
            history = db.snapshot_history(conn, now - timedelta(days=1))
            latest = db.latest_snapshots(conn)

        assert [row.used_percent for row in history] == [10, 12]
        assert [row.used_percent for row in latest] == [12]

    @pytest.mark.parametrize(
        ("first", "then"),
        [
            pytest.param({"blocking": True}, {"blocking": False}, id="it stops blocking"),
            pytest.param({"blocking": False}, {"blocking": True}, id="it starts blocking"),
            pytest.param({"window_seconds": 18_000}, {"window_seconds": 604_800}, id="its length changes"),
            pytest.param({"window_seconds": 18_000}, {"window_seconds": None}, id="its length is no longer stated"),
        ],
    )
    def test_a_change_in_whether_it_blocks_or_how_long_it_is_starts_a_new_step(
        self, database: Database, now: datetime, first: dict[str, Any], then: dict[str, Any]
    ) -> None:
        """Rewritten in the earlier row, the change would be reported from before it was observed."""
        identity = make_identity()
        self.record(database, make_reading(identity, now, make_window(used_percent=10, **first)))
        self.record(database, make_reading(identity, now + timedelta(minutes=5), make_window(used_percent=10, **then)))

        with database.connect() as conn:
            history = db.snapshot_history(conn, now - timedelta(days=1))

        (field,) = first
        assert [getattr(row, field) for row in history] == [first[field], then[field]]
        assert [row.observed_at for row in history] == [now, now + timedelta(minutes=5)]
        assert history[0].confirmed_at == now

    def test_a_new_label_for_an_unchanged_window_is_taken_in_place(self, database: Database, now: datetime) -> None:
        """A label is what the window is called, not something measured about it."""
        identity = make_identity()
        self.record(database, make_reading(identity, now, make_window(used_percent=10, label="5h")))
        self.record(
            database,
            make_reading(identity, now + timedelta(minutes=5), make_window(used_percent=10, label="5-hour session")),
        )

        with database.connect() as conn:
            history = db.snapshot_history(conn, now - timedelta(days=1))

        assert [(row.label, row.observed_at, row.confirmed_at) for row in history] == [
            ("5-hour session", now, now + timedelta(minutes=5))
        ]

    def test_history_that_starts_inside_a_step_includes_that_step(self, database: Database, now: datetime) -> None:
        """The step in effect when the range starts is the state it starts in, whenever it was last confirmed."""
        identity = make_identity()
        self.record(database, make_reading(identity, now, make_window(used_percent=10)))
        self.record(database, make_reading(identity, now + timedelta(minutes=5), make_window(used_percent=12)))

        with database.connect() as conn:
            history = db.snapshot_history(conn, now + timedelta(minutes=2))

        assert [row.used_percent for row in history] == [10, 12]

    def test_history_holds_one_step_from_before_it_starts_and_not_the_ones_before_that(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity()
        for minutes, percent in ((-60, 8), (0, 10), (5, 12)):
            self.record(
                database, make_reading(identity, now + timedelta(minutes=minutes), make_window(used_percent=percent))
            )

        with database.connect() as conn:
            history = db.snapshot_history(conn, now + timedelta(minutes=2))

        assert [row.used_percent for row in history] == [10, 12]

    def test_history_that_starts_after_the_only_reading_still_holds_the_step_that_reading_began(
        self, database: Database, now: datetime
    ) -> None:
        """Nothing has replaced it, so it is still what ``latest_snapshots`` reports as current."""
        self.record(database, make_reading(make_identity(), now, make_window(used_percent=10)))

        with database.connect() as conn:
            history = db.snapshot_history(conn, now + timedelta(days=3))
            latest = db.latest_snapshots(conn)

        assert [row.used_percent for row in history] == [10]
        assert history == latest

    def test_a_step_observed_at_the_moment_history_starts_is_its_first_step(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity()
        self.record(database, make_reading(identity, now, make_window(used_percent=10)))
        self.record(database, make_reading(identity, now + timedelta(minutes=5), make_window(used_percent=12)))

        with database.connect() as conn:
            history = db.snapshot_history(conn, now + timedelta(minutes=5))

        assert [row.used_percent for row in history] == [12]

    def test_history_before_the_first_reading_holds_every_step(self, database: Database, now: datetime) -> None:
        identity = make_identity()
        self.record(database, make_reading(identity, now, make_window(used_percent=10)))
        self.record(database, make_reading(identity, now + timedelta(minutes=5), make_window(used_percent=12)))

        with database.connect() as conn:
            history = db.snapshot_history(conn, now - timedelta(seconds=1))

        assert [row.used_percent for row in history] == [10, 12]

    def test_each_account_window_has_its_own_starting_step(self, database: Database, now: datetime) -> None:
        first, second = make_identity(), make_identity(external_id="org-b", label="Account B")
        first_id = self.record(
            database,
            make_reading(first, now, make_window(used_percent=10), make_window(key="weekly", used_percent=40)),
        )
        second_id = self.record(
            database, make_reading(second, now - timedelta(hours=1), make_window(key="weekly", used_percent=70))
        )
        self.record(
            database,
            make_reading(
                first,
                now + timedelta(minutes=5),
                make_window(used_percent=12),
                make_window(key="weekly", used_percent=40),
            ),
        )

        with database.connect() as conn:
            history = db.snapshot_history(conn, now + timedelta(minutes=2))

        assert [(row.account_id, row.window_key, row.used_percent) for row in history] == [
            (first_id, "five_hour", 10),
            (first_id, "five_hour", 12),
            (first_id, "weekly", 40),
            (second_id, "weekly", 70),
        ]

    def test_a_changed_reset_time_alone_starts_a_new_step(self, database: Database, now: datetime) -> None:
        identity = make_identity()
        self.record(database, make_reading(identity, now, make_window(used_percent=0, resets_at=now)))
        self.record(
            database,
            make_reading(
                identity, now + timedelta(minutes=5), make_window(used_percent=0, resets_at=now + timedelta(hours=5))
            ),
        )

        with database.connect() as conn:
            assert len(db.snapshot_history(conn, now - timedelta(days=1))) == 2

    def test_a_reading_older_than_what_is_stored_does_not_overwrite_it(self, database: Database, now: datetime) -> None:
        identity = make_identity()
        self.record(database, make_reading(identity, now, make_window(used_percent=50)))
        self.record(database, make_reading(identity, now - timedelta(hours=1), make_window(used_percent=20)))

        with database.connect() as conn:
            latest = db.latest_snapshots(conn)

        assert [(row.used_percent, row.confirmed_at) for row in latest] == [(50, now)]

    def test_a_reading_older_than_what_is_stored_does_not_replace_newer_facts_either(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity()
        newer_facts = (Fact("Prepaid balance", "500"),)
        older_facts = (Fact("Prepaid balance", "900"),)
        self.record(database, QuotaReading(identity, (make_window(used_percent=72),), now, "api", newer_facts))
        self.record(
            database,
            QuotaReading(identity, (make_window(used_percent=60),), now - timedelta(hours=3), "log", older_facts),
        )

        with database.connect() as conn:
            assert db.list_accounts(conn)[0].facts == newer_facts
            assert [row.used_percent for row in db.latest_snapshots(conn)] == [72]

    def test_an_older_reading_of_a_window_the_newest_reading_lacks_is_ignored_as_a_whole(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity()
        self.record(database, make_reading(identity, now, make_window("weekly", 40)))
        self.record(database, make_reading(identity, now - timedelta(hours=1), make_window("five_hour", 99)))

        with database.connect() as conn:
            assert [row.window_key for row in db.snapshot_history(conn, now - timedelta(days=1))] == ["weekly"]

    def test_a_reading_at_exactly_the_stored_time_is_applied(self, database: Database, now: datetime) -> None:
        identity = make_identity()
        self.record(database, make_reading(identity, now, make_window(used_percent=10)))
        self.record(database, make_reading(identity, now, make_window(used_percent=11)))

        with database.connect() as conn:
            assert [row.used_percent for row in db.latest_snapshots(conn)] == [11]

    def test_a_window_the_provider_stopped_reporting_drops_out_of_the_current_view(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity()
        self.record(
            database,
            make_reading(identity, now, make_window("five_hour", 10), make_window("weekly", 40)),
        )
        self.record(database, make_reading(identity, now + timedelta(minutes=5), make_window("weekly", 40)))

        with database.connect() as conn:
            latest = db.latest_snapshots(conn)

        assert [row.window_key for row in latest] == ["weekly"]

    @pytest.mark.parametrize("later_by", [timedelta(0), timedelta(microseconds=900_000)])
    def test_a_dropped_window_drops_out_even_when_both_readings_fall_in_the_same_second(
        self, database: Database, now: datetime, later_by: timedelta
    ) -> None:
        """Timestamps are stored to the second, so only the reading order can tell these two apart."""
        identity = make_identity()
        self.record(
            database,
            make_reading(identity, now, make_window("five_hour", 100), make_window("weekly", 10)),
        )
        self.record(database, make_reading(identity, now + later_by, make_window("weekly", 10)))

        with database.connect() as conn:
            latest = db.latest_snapshots(conn)

        assert [(row.window_key, row.used_percent) for row in latest] == [("weekly", 10)]

    def test_a_window_that_returns_in_the_same_second_is_current_again(self, database: Database, now: datetime) -> None:
        identity = make_identity()
        both = make_reading(identity, now, make_window("five_hour", 100), make_window("weekly", 10))
        self.record(database, both)
        self.record(database, make_reading(identity, now, make_window("weekly", 10)))
        self.record(database, both)

        with database.connect() as conn:
            latest = db.latest_snapshots(conn)
            history = db.snapshot_history(conn, now - timedelta(days=1))

        assert sorted(row.window_key for row in latest) == ["five_hour", "weekly"]
        assert len(history) == 2

    def test_values_that_change_back_and_forth_within_a_second_keep_only_the_last_as_current(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity()
        for percent in (10, 20, 10):
            self.record(database, make_reading(identity, now, make_window(used_percent=percent)))

        with database.connect() as conn:
            latest = db.latest_snapshots(conn)
            history = db.snapshot_history(conn, now - timedelta(days=1))

        assert [row.used_percent for row in latest] == [10]
        assert [row.used_percent for row in history] == [10, 20, 10]

    def test_a_rejected_older_reading_does_not_hide_the_current_windows(
        self, database: Database, now: datetime
    ) -> None:
        """A dropped reading must not advance the reading order, or the stored windows would stop matching it."""
        identity = make_identity()
        self.record(database, make_reading(identity, now, make_window("weekly", 40)))
        self.record(database, make_reading(identity, now - timedelta(hours=1), make_window("weekly", 5)))

        with database.connect() as conn:
            assert [(row.window_key, row.used_percent) for row in db.latest_snapshots(conn)] == [("weekly", 40)]

    def test_an_account_with_no_reading_has_no_current_windows(self, database: Database, now: datetime) -> None:
        with database.connect() as conn:
            db.upsert_account(conn, make_identity(), now)

            assert db.latest_snapshots(conn) == []

    def test_each_account_keeps_its_own_latest_reading(self, database: Database, now: datetime) -> None:
        first = self.record(database, make_reading(make_identity(external_id="a"), now, make_window(used_percent=10)))
        second = self.record(
            database,
            make_reading(make_identity(external_id="b"), now - timedelta(days=3), make_window(used_percent=90)),
        )

        with database.connect() as conn:
            latest = {row.account_id: row.used_percent for row in db.latest_snapshots(conn)}

        assert latest == {first: 10, second: 90}

    def test_facts_from_the_latest_reading_replace_the_previous_ones(self, database: Database, now: datetime) -> None:
        identity = make_identity()
        reading = QuotaReading(
            identity=identity,
            windows=(make_window(),),
            observed_at=now,
            source="api",
            facts=(Fact("Credits", "10 remaining", now + timedelta(days=1)),),
        )
        self.record(database, reading)
        self.record(database, make_reading(identity, now + timedelta(minutes=5), make_window()))

        with database.connect() as conn:
            assert db.list_accounts(conn)[0].facts == ()

        self.record(
            database,
            QuotaReading(identity, (make_window(),), now + timedelta(minutes=10), "api", reading.facts),
        )
        with database.connect() as conn:
            assert db.list_accounts(conn)[0].facts == reading.facts

    @pytest.mark.parametrize(
        ("stored", "message"),
        [
            pytest.param('{"label": "Credits"}', "an account's stored facts are not a list", id="not a list"),
            pytest.param('[{"label": NaN}]', "NaN is not a JSON number", id="a constant JSON does not have"),
        ],
    )
    def test_facts_stored_as_anything_this_store_does_not_write_are_refused(
        self, database: Database, now: datetime, stored: str, message: str
    ) -> None:
        self.record(database, make_reading(make_identity(), now, make_window()))
        with database.connect() as conn:
            conn.execute("UPDATE accounts SET facts_json = ?", (stored,))

        with pytest.raises(ValueError, match=message), database.read() as conn:
            db.list_accounts(conn)

    def test_pruning_removes_expired_history_but_never_an_accounts_last_reading(
        self, database: Database, now: datetime
    ) -> None:
        dormant = make_identity(external_id="dormant")
        active = make_identity(external_id="active")
        long_ago = now - timedelta(days=200)
        self.record(database, make_reading(dormant, long_ago, make_window(used_percent=70)))
        self.record(database, make_reading(active, long_ago - timedelta(days=100), make_window(used_percent=2)))
        self.record(database, make_reading(active, long_ago, make_window(used_percent=5)))
        self.record(database, make_reading(active, now, make_window(used_percent=30)))

        with database.connect() as conn:
            db.prune(conn, now)
            remaining = [(row.used_percent) for row in db.snapshot_history(conn, long_ago - timedelta(days=300))]

        assert sorted(remaining) == [5, 30, 70]

    def test_pruning_keeps_the_step_in_effect_at_the_retention_cutoff_so_retained_history_starts_with_it(
        self, database: Database, now: datetime
    ) -> None:
        """It was last confirmed before the cutoff, and it is still the state of everything up to the next step."""
        identity = make_identity()
        self.record(database, make_reading(identity, now - timedelta(days=100), make_window(used_percent=10)))
        self.record(database, make_reading(identity, now - timedelta(days=10), make_window(used_percent=60)))

        with database.connect() as conn:
            db.prune(conn, now)
            retained = db.snapshot_history(conn, now - timedelta(days=90))

        assert [(row.used_percent, row.observed_at) for row in retained] == [
            (10, now - timedelta(days=100)),
            (60, now - timedelta(days=10)),
        ]

    def test_pruning_deletes_every_step_that_ended_before_the_cutoff_and_none_that_ended_after_it(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity()
        for days_ago, percent in ((150, 1), (120, 2), (100, 3), (80, 4), (10, 5)):
            self.record(
                database, make_reading(identity, now - timedelta(days=days_ago), make_window(used_percent=percent))
            )

        with database.connect() as conn:
            db.prune(conn, now)
            remaining = [row.used_percent for row in db.snapshot_history(conn, now - timedelta(days=365))]

        assert remaining == [3, 4, 5]

    def test_a_step_replaced_exactly_at_the_cutoff_ended_then_and_is_deleted(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity()
        self.record(database, make_reading(identity, now - timedelta(days=100), make_window(used_percent=10)))
        self.record(database, make_reading(identity, now - timedelta(days=90), make_window(used_percent=20)))

        with database.connect() as conn:
            db.prune(conn, now)
            remaining = [row.used_percent for row in db.snapshot_history(conn, now - timedelta(days=365))]

        assert remaining == [20]

    def test_pruning_keeps_each_window_s_own_step_in_effect(self, database: Database, now: datetime) -> None:
        identity = make_identity()
        old, recent = now - timedelta(days=100), now - timedelta(days=10)
        self.record(
            database,
            make_reading(identity, old - timedelta(days=50), make_window("weekly", 5), make_window(used_percent=1)),
        )
        self.record(database, make_reading(identity, old, make_window("weekly", 5), make_window(used_percent=10)))
        self.record(database, make_reading(identity, recent, make_window("weekly", 40), make_window(used_percent=60)))

        with database.connect() as conn:
            db.prune(conn, now)
            remaining = [
                (row.window_key, row.used_percent) for row in db.snapshot_history(conn, now - timedelta(days=365))
            ]

        assert remaining == [("five_hour", 10), ("five_hour", 60), ("weekly", 5), ("weekly", 40)]

    def test_a_window_the_latest_reading_stopped_reporting_keeps_its_last_step_until_that_is_past_retention(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity()
        dropped_recently = now - timedelta(days=10)
        self.record(
            database,
            make_reading(identity, dropped_recently, make_window("weekly", 20), make_window("monthly", 3)),
        )
        self.record(database, make_reading(identity, now, make_window("weekly", 25)))

        with database.connect() as conn:
            db.prune(conn, now)
            kept = [row.window_key for row in db.snapshot_history(conn, now - timedelta(days=365))]
            db.prune(conn, now + timedelta(days=81))
            later = [row.window_key for row in db.snapshot_history(conn, now - timedelta(days=365))]

        assert kept == ["monthly", "weekly", "weekly"]
        assert later == ["weekly", "weekly"]

    def test_retention_is_measured_in_elapsed_time_whatever_zone_now_is_given_in(self, database: Database) -> None:
        """Ninety days before a New York afternoon in November is an hour off by the wall clock in August."""
        now_in_new_york = datetime(2026, 11, 15, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        account = make_identity()
        cutoff = now_in_new_york.astimezone(UTC) - timedelta(days=90)
        steps = [
            (cutoff - timedelta(hours=2), 10),
            (cutoff - timedelta(minutes=30), 20),
            (cutoff + timedelta(minutes=30), 30),
            (now_in_new_york.astimezone(UTC), 40),
        ]
        for observed_at, percent in steps:
            self.record(database, make_reading(account, observed_at, make_window(used_percent=percent)))

        with database.connect() as conn:
            db.prune(conn, now_in_new_york)
            remaining = [row.used_percent for row in db.snapshot_history(conn, cutoff - timedelta(days=1))]

        # Why exactly these: the first step ended 30 minutes before the cutoff and the second 30 minutes
        # after it. A cutoff an hour off in either direction would delete both or neither.
        assert remaining == [20, 30, 40]

    def test_pruning_refuses_a_naive_now(self, database: Database) -> None:
        with (
            pytest.raises(ValueError, match="a naive datetime has no defined instant"),
            database.connect() as conn,
        ):
            db.prune(conn, datetime(2026, 11, 15, 12, 0))

    def test_pruning_keeps_exactly_the_windows_of_an_expired_last_reading(
        self, database: Database, now: datetime
    ) -> None:
        dormant = make_identity(external_id="dormant")
        long_ago = now - timedelta(days=200)
        self.record(database, make_reading(dormant, long_ago, make_window("five_hour", 70), make_window("weekly", 20)))
        self.record(database, make_reading(dormant, long_ago, make_window("weekly", 20)))

        with database.connect() as conn:
            db.prune(conn, now)
            remaining = [row.window_key for row in db.snapshot_history(conn, long_ago - timedelta(days=1))]
            current = [row.window_key for row in db.latest_snapshots(conn)]

        assert remaining == current == ["weekly"]


class TestStepsWithinOneSecond:
    """A cutoff can fall between two steps observed in the same second, and each is on its own side of it."""

    FIRST_STEP = timedelta(microseconds=100_000)
    SECOND_STEP = timedelta(microseconds=900_000)

    def record(self, database: Database, reading: QuotaReading) -> int:
        with database.connect() as conn:
            account_id = db.upsert_account(conn, reading.identity, reading.observed_at)
            db.record_reading(conn, account_id, reading)
        return account_id

    def record_steps(self, database: Database, second: datetime) -> None:
        """5% a day before ``second``, then 10% and 20% a tenth and nine tenths of the way through it."""
        identity = make_identity()
        for observed_at, percent in (
            (second - timedelta(days=1), 5),
            (second + self.FIRST_STEP, 10),
            (second + self.SECOND_STEP, 20),
        ):
            self.record(database, make_reading(identity, observed_at, make_window(used_percent=percent)))

    @pytest.mark.parametrize(
        ("microseconds_into_the_second", "expected"),
        [
            pytest.param(50_000, [5, 10, 20], id="before both"),
            pytest.param(100_000, [10, 20], id="at the first"),
            pytest.param(500_000, [10, 20], id="between them"),
            pytest.param(899_999, [10, 20], id="a microsecond before the second"),
            pytest.param(900_000, [20], id="at the second"),
            pytest.param(950_000, [20], id="after both"),
        ],
    )
    def test_history_starts_with_the_step_in_effect_at_an_instant_inside_the_second(
        self, database: Database, now: datetime, microseconds_into_the_second: int, expected: list[int]
    ) -> None:
        self.record_steps(database, now)

        with database.connect() as conn:
            history = db.snapshot_history(conn, now + timedelta(microseconds=microseconds_into_the_second))

        assert [row.used_percent for row in history] == expected

    @pytest.mark.parametrize(
        ("microseconds_into_the_second", "expected"),
        [
            pytest.param(50_000, [5, 10, 20, 30], id="before both"),
            pytest.param(100_000, [10, 20, 30], id="at the first"),
            pytest.param(500_000, [10, 20, 30], id="between them"),
            pytest.param(899_999, [10, 20, 30], id="a microsecond before the second"),
            pytest.param(900_000, [20, 30], id="at the second"),
            pytest.param(950_000, [20, 30], id="after both"),
        ],
    )
    def test_pruning_keeps_the_step_in_effect_at_a_cutoff_inside_the_second(
        self, database: Database, now: datetime, microseconds_into_the_second: int, expected: list[int]
    ) -> None:
        second = now - db.SNAPSHOT_RETENTION
        self.record_steps(database, second)
        self.record(database, make_reading(make_identity(), now, make_window(used_percent=30)))

        with database.connect() as conn:
            db.prune(conn, now + timedelta(microseconds=microseconds_into_the_second))
            remaining = [row.used_percent for row in db.snapshot_history(conn, second - timedelta(days=365))]

        assert remaining == expected

    @pytest.mark.parametrize(
        ("microseconds_into_the_second", "expected"),
        [
            pytest.param(50_000, ["monthly", "weekly", "weekly"], id="confirmed after the cutoff"),
            pytest.param(100_000, ["monthly", "weekly", "weekly"], id="confirmed at the cutoff"),
            pytest.param(100_001, ["weekly", "weekly"], id="confirmed a microsecond before the cutoff"),
            pytest.param(500_000, ["weekly", "weekly"], id="confirmed earlier in the cutoff's second"),
        ],
    )
    def test_a_window_no_longer_reported_goes_once_its_last_confirmation_is_before_the_cutoff(
        self, database: Database, now: datetime, microseconds_into_the_second: int, expected: list[str]
    ) -> None:
        second = now - db.SNAPSHOT_RETENTION
        identity = make_identity()
        self.record(
            database,
            make_reading(identity, second + self.FIRST_STEP, make_window("weekly", 20), make_window("monthly", 3)),
        )
        self.record(database, make_reading(identity, now, make_window("weekly", 25)))

        with database.connect() as conn:
            db.prune(conn, now + timedelta(microseconds=microseconds_into_the_second))
            remaining = [row.window_key for row in db.snapshot_history(conn, second - timedelta(days=365))]

        assert remaining == expected

    def test_a_step_is_read_back_with_the_exact_times_it_was_observed_and_confirmed(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity()
        observed_at = now + timedelta(microseconds=123_456)
        confirmed_at = now + timedelta(minutes=5, microseconds=654_321)
        self.record(database, make_reading(identity, observed_at, make_window(used_percent=10)))
        self.record(database, make_reading(identity, confirmed_at, make_window(used_percent=10)))

        with database.connect() as conn:
            (step,) = db.snapshot_history(conn, now - timedelta(days=1))
            (current,) = db.latest_snapshots(conn)

        assert (step.observed_at, step.confirmed_at) == (observed_at, confirmed_at)
        assert step.observed_at.utcoffset() == step.confirmed_at.utcoffset() == timedelta(0)
        assert current == step

    def test_history_refuses_a_naive_start(self, database: Database) -> None:
        with (
            pytest.raises(ValueError, match="a naive datetime has no defined instant"),
            database.connect() as conn,
        ):
            db.snapshot_history(conn, datetime(2026, 10, 3, 14, 0))

    def test_a_reading_at_a_time_utc_cannot_express_is_refused_and_changes_nothing(
        self, database: Database, now: datetime
    ) -> None:
        """It has an epoch value and no UTC datetime, so it could be stored and then never read back."""
        identity = make_identity()
        beyond_the_calendar_in_utc = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone(timedelta(hours=-5)))
        with database.connect() as conn:
            account_id = db.upsert_account(conn, identity, now)

        with pytest.raises(ValueError, match="cannot be expressed in UTC"), database.connect() as conn:
            db.record_reading(
                conn, account_id, make_reading(identity, beyond_the_calendar_in_utc, make_window(used_percent=99))
            )
        self.record(database, make_reading(identity, now, make_window(used_percent=10)))

        with database.connect() as conn:
            assert [row.used_percent for row in db.latest_snapshots(conn)] == [10]


class TestLoginTimeline:
    def test_repeated_sightings_of_one_account_form_a_single_interval(self, database: Database, now: datetime) -> None:
        with database.connect() as conn:
            account = db.upsert_account(conn, make_identity(), now)
            db.observe_active_account(conn, "claude:home", account, now)
            db.observe_active_account(conn, "claude:home", account, now + timedelta(minutes=5))
            timeline = db.active_timeline(conn, "claude:home")

        assert timeline == [(now, account)]

    def test_a_different_account_opens_a_new_interval(self, database: Database, now: datetime) -> None:
        later = now + timedelta(hours=2)
        with database.connect() as conn:
            first = db.upsert_account(conn, make_identity(external_id="a"), now)
            second = db.upsert_account(conn, make_identity(external_id="b"), now)
            db.observe_active_account(conn, "claude:home", first, now)
            db.observe_active_account(conn, "claude:home", second, later)
            timeline = db.active_timeline(conn, "claude:home")

        assert timeline == [(now, first), (later, second)]

    def test_an_interval_starts_at_the_first_whole_second_at_or_after_the_login_was_seen(
        self, database: Database, now: datetime
    ) -> None:
        seen_at = now + timedelta(hours=1, milliseconds=800)
        with database.connect() as conn:
            first = db.upsert_account(conn, make_identity(external_id="a"), now)
            second = db.upsert_account(conn, make_identity(external_id="b"), now)
            db.observe_active_account(conn, "claude:home", first, now)
            db.observe_active_account(conn, "claude:home", second, seen_at)
            db.observe_active_account(conn, "claude:home", second, seen_at + timedelta(minutes=5))
            timeline = db.active_timeline(conn, "claude:home")

        assert timeline == [(now, first), (now + timedelta(hours=1, seconds=1), second)]

    def test_a_logout_seen_part_way_through_a_second_starts_at_the_next_one(
        self, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn:
            account = db.upsert_account(conn, make_identity(), now)
            db.observe_active_account(conn, "claude:home", account, now)
            db.observe_active_account(conn, "claude:home", None, now + timedelta(minutes=5, microseconds=1))
            timeline = db.active_timeline(conn, "claude:home")

        assert timeline == [(now, account), (now + timedelta(minutes=5, seconds=1), None)]

    def test_a_logout_closes_the_account_s_interval_and_logging_back_in_starts_a_new_one(
        self, database: Database, now: datetime
    ) -> None:
        """Without the logged-out interval the account would be credited with everything in the gap."""
        logged_out_at = now + timedelta(hours=1)
        back_at = now + timedelta(hours=3)
        with database.connect() as conn:
            account = db.upsert_account(conn, make_identity(), now)
            db.observe_active_account(conn, "claude:home", account, now)
            db.observe_active_account(conn, "claude:home", None, logged_out_at)
            db.observe_active_account(conn, "claude:home", None, logged_out_at + timedelta(minutes=5))
            db.observe_active_account(conn, "claude:home", account, back_at)
            timeline = db.active_timeline(conn, "claude:home")

        assert timeline == [(now, account), (logged_out_at, None), (back_at, account)]

    def test_a_source_first_seen_with_nobody_logged_in_starts_with_a_logged_out_interval(
        self, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn:
            db.observe_active_account(conn, "claude:home", None, now)
            db.observe_active_account(conn, "claude:home", None, now + timedelta(minutes=5))

            assert db.active_timeline(conn, "claude:home") == [(now, None)]

    def test_a_login_seen_after_another_is_current_though_the_clock_stepped_back_in_between(
        self, database: Database, now: datetime
    ) -> None:
        """Ordered by time, the earlier login stayed current and the timeline put the two the wrong way round."""
        stepped_back = now - timedelta(seconds=1)
        with database.connect() as conn:
            first = db.upsert_account(conn, make_identity(external_id="a"), now)
            second = db.upsert_account(conn, make_identity(external_id="b"), now)
            db.observe_active_account(conn, "claude:home", first, now)
            db.observe_active_account(conn, "claude:home", second, stepped_back)
            db.observe_active_account(conn, "claude:home", second, stepped_back + timedelta(seconds=30))
            timeline = db.active_timeline(conn, "claude:home")

        assert timeline == [(now, first), (now, second)]

    def test_an_interval_s_last_observation_never_moves_back_and_the_next_starts_after_it(
        self, database: Database, now: datetime
    ) -> None:
        later = now + timedelta(minutes=5)
        with database.connect() as conn:
            first = db.upsert_account(conn, make_identity(external_id="a"), now)
            second = db.upsert_account(conn, make_identity(external_id="b"), now)
            db.observe_active_account(conn, "claude:home", first, now)
            db.observe_active_account(conn, "claude:home", first, later)
            db.observe_active_account(conn, "claude:home", first, now + timedelta(minutes=1))
            last_observed = conn.execute("SELECT last_observed_at FROM active_account_log").fetchone()[0]
            db.observe_active_account(conn, "claude:home", second, now + timedelta(minutes=2))
            timeline = db.active_timeline(conn, "claude:home")

        assert last_observed == to_iso(later)
        assert timeline == [(now, first), (later, second)]

    def test_timelines_are_kept_separately_per_source(self, database: Database, now: datetime) -> None:
        with database.connect() as conn:
            account = db.upsert_account(conn, make_identity(), now)
            db.observe_active_account(conn, "claude:home", account, now)
            assert db.active_timeline(conn, "claude:standing") == []
            assert db.active_timeline(conn, "claude:home") == [(now, account)]

    def test_a_recorded_login_boundary_matches_only_that_instant(self, database: Database, now: datetime) -> None:
        later = now + timedelta(microseconds=100_000)
        with database.connect() as conn:
            db.acknowledge_recorded_login_boundary(conn, "claude:home", later)
            db.acknowledge_recorded_login_boundary(conn, "claude:home", later)
            assert db.login_boundary_was_recorded(conn, "claude:home", later)
            assert not db.login_boundary_was_recorded(conn, "claude:home", now)
            assert not db.login_boundary_was_recorded(conn, "codex:home", later)
            stored = conn.execute("SELECT COUNT(*) FROM recorded_login_boundaries").fetchone()[0]

        assert stored == 1

    def test_a_gap_is_opened_when_no_interval_exists_yet(self, database: Database, now: datetime) -> None:
        with database.connect() as conn:
            db.record_unrecorded_gap(conn, "claude:fresh", now)
            assert db.active_timeline(conn, "claude:fresh") == [(now, None)]
            assert db.login_boundary_was_recorded(conn, "claude:fresh", now)

    def test_a_gap_that_is_already_open_is_acknowledged_without_a_second_interval(
        self, database: Database, now: datetime
    ) -> None:
        later = now + timedelta(minutes=5)
        with database.connect() as conn:
            db.record_unrecorded_gap(conn, "claude:home", now)
            db.record_unrecorded_gap(conn, "claude:home", later)
            assert db.active_timeline(conn, "claude:home") == [(now, None)]
            assert db.login_boundary_was_recorded(conn, "claude:home", later)
            last_observed = conn.execute("SELECT last_observed_at FROM active_account_log").fetchone()[0]

        assert last_observed == to_iso(later)

    def test_a_gap_before_an_open_logout_is_not_left_on_the_preceding_account(
        self, database: Database, now: datetime
    ) -> None:
        """Acknowledging the boundary used to leave 10:15 on account A after a logout at 10:20."""
        gap = now + timedelta(minutes=10)
        logged_out = now + timedelta(minutes=20)
        with database.connect() as conn:
            account = db.upsert_account(conn, make_identity(), now)
            db.observe_active_account(conn, "claude:home", account, now)
            db.observe_active_account(conn, "claude:home", None, logged_out)
            db.record_unrecorded_gap(conn, "claude:home", gap)
            timeline = db.active_timeline(conn, "claude:home")
            assert db.login_boundary_was_recorded(conn, "claude:home", gap)

        resolve = build_account_resolver(timeline)
        assert timeline == [(now, account), (gap, None)]
        assert resolve(now + timedelta(minutes=5)) == account
        assert resolve(now + timedelta(minutes=15)) == UNATTRIBUTED_ACCOUNT_ID

    def test_a_later_sighting_of_the_same_account_does_not_keep_the_missed_gap(
        self, database: Database, now: datetime
    ) -> None:
        gap = now + timedelta(minutes=10)
        with database.connect() as conn:
            account = db.upsert_account(conn, make_identity(), now)
            db.observe_active_account(conn, "claude:home", account, now)
            db.observe_active_account(conn, "claude:home", account, now + timedelta(minutes=18))
            db.observe_active_account(conn, "claude:home", None, now + timedelta(minutes=20))
            db.record_unrecorded_gap(conn, "claude:home", gap)
            timeline = db.active_timeline(conn, "claude:home")

        assert timeline == [(now, account), (gap, None)]
        assert build_account_resolver(timeline)(now + timedelta(minutes=15)) == UNATTRIBUTED_ACCOUNT_ID

    def test_a_gap_before_the_only_logout_moves_that_logout_to_the_boundary(
        self, database: Database, now: datetime
    ) -> None:
        gap = now + timedelta(minutes=10)
        with database.connect() as conn:
            db.observe_active_account(conn, "claude:home", None, now + timedelta(minutes=20))
            db.record_unrecorded_gap(conn, "claude:home", gap)
            timeline = db.active_timeline(conn, "claude:home")

        assert timeline == [(gap, None)]

    def test_a_gap_from_before_the_preceding_login_does_not_take_that_login(
        self, database: Database, now: datetime
    ) -> None:
        logged_out = now + timedelta(minutes=20)
        with database.connect() as conn:
            account = db.upsert_account(conn, make_identity(), now)
            db.observe_active_account(conn, "claude:home", account, now)
            db.observe_active_account(conn, "claude:home", None, logged_out)
            db.record_unrecorded_gap(conn, "claude:home", now - timedelta(minutes=10))
            timeline = db.active_timeline(conn, "claude:home")
            assert db.login_boundary_was_recorded(conn, "claude:home", now - timedelta(minutes=10))

        assert timeline == [(now, account), (logged_out, None)]
        assert build_account_resolver(timeline)(now + timedelta(minutes=15)) == account

    def test_a_gap_at_the_preceding_login_s_start_does_not_replace_that_login(
        self, database: Database, now: datetime
    ) -> None:
        logged_out = now + timedelta(minutes=20)
        with database.connect() as conn:
            account = db.upsert_account(conn, make_identity(), now)
            db.observe_active_account(conn, "claude:home", account, now)
            db.observe_active_account(conn, "claude:home", None, logged_out)
            db.record_unrecorded_gap(conn, "claude:home", now)
            timeline = db.active_timeline(conn, "claude:home")

        assert timeline == [(now, account), (logged_out, None)]

    def test_a_later_gap_on_an_open_logout_keeps_the_next_login_out_of_that_gap(
        self, database: Database, now: datetime
    ) -> None:
        """A clock step back to 10:15 used to start the login before the failed boundary at 10:20."""
        logged_out = now + timedelta(minutes=10)
        failed = now + timedelta(minutes=20)
        stepped_back = now + timedelta(minutes=15)
        with database.connect() as conn:
            db.observe_active_account(conn, "claude:home", None, logged_out)
            db.record_unrecorded_gap(conn, "claude:home", failed)
            account = db.upsert_account(conn, make_identity(), now)
            db.observe_active_account(conn, "claude:home", account, stepped_back)
            timeline = db.active_timeline(conn, "claude:home")
            assert db.login_boundary_was_recorded(conn, "claude:home", failed)

        assert timeline == [(logged_out, None), (failed, account)]

    def test_a_gap_inside_an_open_logout_does_not_move_its_last_observation_backward(
        self, database: Database, now: datetime
    ) -> None:
        last = now + timedelta(minutes=20)
        with database.connect() as conn:
            db.observe_active_account(conn, "claude:home", None, now)
            db.observe_active_account(conn, "claude:home", None, last)
            db.record_unrecorded_gap(conn, "claude:home", now + timedelta(minutes=10))
            db.record_unrecorded_gap(conn, "claude:home", last)
            timeline = db.active_timeline(conn, "claude:home")
            last_observed = conn.execute("SELECT last_observed_at FROM active_account_log").fetchone()[0]

        assert timeline == [(now, None)]
        assert last_observed == to_iso(last)

    def test_a_fractional_gap_before_an_open_logout_starts_at_the_next_whole_second(
        self, database: Database, now: datetime
    ) -> None:
        gap = now + timedelta(minutes=10, microseconds=100_000)
        logged_out = now + timedelta(minutes=20)
        with database.connect() as conn:
            account = db.upsert_account(conn, make_identity(), now)
            db.observe_active_account(conn, "claude:home", account, now)
            db.observe_active_account(conn, "claude:home", None, logged_out)
            db.record_unrecorded_gap(conn, "claude:home", gap)
            timeline = db.active_timeline(conn, "claude:home")

        starts_at = now + timedelta(minutes=10, seconds=1)
        assert timeline == [(now, account), (starts_at, None)]
        assert build_account_resolver(timeline)(gap) == account

    def test_a_gap_after_an_account_starts_at_the_later_of_the_boundary_and_the_last_observation(
        self, database: Database, now: datetime
    ) -> None:
        account = None
        with database.connect() as conn:
            account = db.upsert_account(conn, make_identity(), now)
            db.observe_active_account(conn, "claude:home", account, now + timedelta(minutes=10))
            db.record_unrecorded_gap(conn, "claude:home", now + timedelta(minutes=5, microseconds=100_000))
            timeline = db.active_timeline(conn, "claude:home")
        assert timeline == [(now + timedelta(minutes=10), account), (now + timedelta(minutes=10), None)]

    def test_a_naive_boundary_is_refused(self, database: Database) -> None:
        with pytest.raises(ValueError, match="naive"), database.connect() as conn:
            db.record_unrecorded_gap(conn, "claude:home", datetime(2026, 10, 3, 14, 0))


class TestSourceBookkeeping:
    def test_a_success_clears_the_previous_error(self, database: Database, now: datetime) -> None:
        with database.connect() as conn:
            db.record_status(conn, "codex:home", db.STATUS_KIND_QUOTA, now, ValueError("boom"))
            failed = db.list_statuses(conn)[0]
            db.record_status(conn, "codex:home", db.STATUS_KIND_QUOTA, now + timedelta(minutes=5), None)
            recovered = db.list_statuses(conn)[0]

        assert (failed.error_type, failed.error_message, failed.last_success_at) == ("ValueError", "boom", None)
        assert (recovered.error_type, recovered.error_message) == (None, None)
        assert recovered.last_success_at == now + timedelta(minutes=5)

    def test_a_failure_keeps_the_time_of_the_last_success(self, database: Database, now: datetime) -> None:
        with database.connect() as conn:
            db.record_status(conn, "codex:home", db.STATUS_KIND_QUOTA, now, None)
            db.record_status(conn, "codex:home", db.STATUS_KIND_QUOTA, now + timedelta(minutes=5), ValueError("x"))
            status = db.list_statuses(conn)[0]

        assert status.last_success_at == now
        assert status.last_attempt_at == now + timedelta(minutes=5)

    def test_one_kind_of_status_can_be_cleared_for_a_source_that_keeps_its_other_kind(
        self, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn:
            for source_id in ("claude:home", "codex:home"):
                db.record_status(conn, source_id, db.STATUS_KIND_QUOTA, now, None)
                db.record_status(conn, source_id, db.STATUS_KIND_TOKENS, now, ValueError("unreadable"))
            db.clear_status(conn, "claude:home", db.STATUS_KIND_TOKENS)
            db.clear_status(conn, "claude:home", db.STATUS_KIND_TOKENS)
            remaining = [(status.source_id, status.kind) for status in db.list_statuses(conn)]

        assert sorted(remaining) == [("claude:home", "quota"), ("codex:home", "quota"), ("codex:home", "tokens")]

    def test_sources_removed_from_the_configuration_are_forgotten(self, database: Database, now: datetime) -> None:
        with database.connect() as conn:
            for source_id in ("claude:keep", "claude:gone"):
                db.set_source_state(conn, source_id, Provider.CLAUDE, 0, None, now)
                db.record_status(conn, source_id, db.STATUS_KIND_QUOTA, now, None)
            db.retain_sources(conn, ["claude:keep"])
            states = [state.source_id for state in db.list_source_states(conn)]
            statuses = [status.source_id for status in db.list_statuses(conn)]

        assert states == ["claude:keep"]
        assert statuses == ["claude:keep"]

    def test_source_state_records_who_is_logged_in_and_updates_in_place(
        self, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn:
            account = db.upsert_account(conn, make_identity(), now)
            db.set_source_state(conn, "claude:home", Provider.CLAUDE, 0, account, now)
            db.set_source_state(conn, "claude:home", Provider.CLAUDE, 0, None, now + timedelta(minutes=5))
            states = db.list_source_states(conn)

        assert len(states) == 1
        assert states[0].current_account_id is None


class TestTokenUsageStorage:
    KEY = UsageKey(account_id=1, bucket_hour="2026-10-03T14", model="model-x")

    def totals(self, database: Database) -> list[TokenCounts]:
        with database.connect() as conn:
            return [row.counts for row in db.token_usage(conn, "2026-10-03T00", "2026-10-04T00")]

    def test_adding_accumulates_within_a_file_and_across_files(self, database: Database) -> None:
        with database.connect() as conn:
            db.add_file_usage(conn, Provider.CODEX, "a.jsonl", {self.KEY: TokenCounts(input_tokens=5, requests=1)})
            db.add_file_usage(conn, Provider.CODEX, "a.jsonl", {self.KEY: TokenCounts(input_tokens=7, requests=1)})
            db.add_file_usage(conn, Provider.CODEX, "b.jsonl", {self.KEY: TokenCounts(output_tokens=3, requests=1)})

        assert self.totals(database) == [TokenCounts(input_tokens=12, output_tokens=3, requests=3)]

    def test_replacing_discards_only_that_files_earlier_contribution(self, database: Database) -> None:
        with database.connect() as conn:
            db.add_file_usage(conn, Provider.GROK, "one", {self.KEY: TokenCounts(input_tokens=100, requests=1)})
            db.add_file_usage(conn, Provider.GROK, "two", {self.KEY: TokenCounts(input_tokens=1, requests=1)})
            db.replace_file_usage(conn, Provider.GROK, "one", {self.KEY: TokenCounts(input_tokens=40, requests=1)})

        assert self.totals(database) == [TokenCounts(input_tokens=41, requests=2)]

    def test_empty_counts_are_not_stored(self, database: Database) -> None:
        with database.connect() as conn:
            db.add_file_usage(conn, Provider.CODEX, "f", {self.KEY: TokenCounts()})

        assert self.totals(database) == []

    def test_the_bucket_range_excludes_its_end(self, database: Database) -> None:
        with database.connect() as conn:
            db.add_file_usage(conn, Provider.CODEX, "f", {self.KEY: TokenCounts(input_tokens=1)})
            assert db.token_usage(conn, "2026-10-03T14", "2026-10-03T15") != []
            assert db.token_usage(conn, "2026-10-03T13", "2026-10-03T14") == []

    def test_usage_is_returned_per_hour_and_model_with_files_summed(self, database: Database) -> None:
        same_hour_other_model = UsageKey(1, "2026-10-03T14", "model-y")
        next_hour = UsageKey(1, "2026-10-03T15", "model-x")
        with database.connect() as conn:
            db.add_file_usage(
                conn,
                Provider.CODEX,
                "first",
                {
                    self.KEY: TokenCounts(input_tokens=1),
                    same_hour_other_model: TokenCounts(input_tokens=2),
                    next_hour: TokenCounts(input_tokens=4),
                },
            )
            db.add_file_usage(conn, Provider.CODEX, "second", {self.KEY: TokenCounts(input_tokens=8)})
            rows = db.token_usage(conn, "2026", "2027")

        assert [(row.bucket_hour, row.model, row.counts.input_tokens) for row in rows] == [
            ("2026-10-03T14", "model-x", 9),
            ("2026-10-03T14", "model-y", 2),
            ("2026-10-03T15", "model-x", 4),
        ]

    def test_cursors_round_trip_with_their_scanner_state(self, database: Database) -> None:
        cursor = FileCursor(byte_offset=120, size=150, mtime_ns=99, state={"model": "m", "previous_totals": {"a": 1}})
        with database.connect() as conn:
            db.save_cursor(conn, Provider.CODEX, "rollout.jsonl", cursor)
            db.save_cursor(conn, Provider.CLAUDE, "rollout.jsonl", FileCursor(1, 1, 1, {}))
            assert db.load_cursors(conn, Provider.CODEX) == {"rollout.jsonl": cursor}

    @pytest.mark.parametrize("state", [None, [], "text", 7, {"nested": [1, {"deep": None}]}])
    def test_a_cursor_s_state_is_whatever_json_value_was_saved(self, database: Database, state: object) -> None:
        with database.connect() as conn:
            db.save_cursor(conn, Provider.CODEX, "rollout.jsonl", FileCursor(1, 2, 3, state))
            assert db.load_cursors(conn, Provider.CODEX)["rollout.jsonl"].state == state

    @pytest.mark.parametrize(
        "state",
        [
            pytest.param({"offset": float("nan")}, id="NaN"),
            pytest.param({"totals": [float("inf")]}, id="infinity, nested"),
            pytest.param(float("-inf"), id="minus infinity"),
        ],
    )
    def test_a_cursor_whose_state_holds_a_number_json_cannot_write_is_refused_and_nothing_is_stored(
        self, database: Database, state: object
    ) -> None:
        """Written as NaN or Infinity, the state would not be JSON, and could not be read back."""
        with (
            pytest.raises(ValueError, match="Out of range float values are not JSON compliant"),
            database.connect() as conn,
        ):
            db.save_cursor(conn, Provider.CODEX, "rollout.jsonl", FileCursor(1, 2, 3, state))

        with database.connect() as conn:
            assert db.load_cursors(conn, Provider.CODEX) == {}

    @pytest.mark.parametrize(
        "stored", ["{", "", "not json", "[" * 200_000, '{"offset": NaN}', "[Infinity]", "-Infinity"]
    )
    def test_a_cursor_whose_saved_state_is_not_json_is_still_loaded_with_no_state_and_costs_no_other_file(
        self, database: Database, stored: str
    ) -> None:
        """Cursors are loaded for a whole provider at once; one damaged row must not stop the rest."""
        with database.connect() as conn:
            db.save_cursor(conn, Provider.CODEX, "damaged.jsonl", FileCursor(120, 150, 99, {"model": "m"}))
            db.save_cursor(conn, Provider.CODEX, "healthy.jsonl", FileCursor(7, 8, 9, {"model": "m"}))
            conn.execute("UPDATE scan_cursors SET state_json = ? WHERE file_key = 'damaged.jsonl'", (stored,))

            cursors = db.load_cursors(conn, Provider.CODEX)

        assert cursors["damaged.jsonl"] == FileCursor(120, 150, 99, None)
        assert cursors["healthy.jsonl"] == FileCursor(7, 8, 9, {"model": "m"})

    def test_a_revision_of_reasoning_tokens_alone_is_stored(self, database: Database) -> None:
        with database.connect() as conn:
            db.add_file_usage(conn, Provider.CODEX, "a.jsonl", {self.KEY: TokenCounts(output_tokens=100, requests=1)})
            db.add_file_usage(conn, Provider.CODEX, "a.jsonl", {self.KEY: TokenCounts(reasoning_tokens=40)})

        assert self.totals(database) == [TokenCounts(output_tokens=100, reasoning_tokens=40, requests=1)]


class TestCountingAClaudeResponseOnce:
    """``add_message_usage`` counts a response once, under its earliest line, and remembers what it counted."""

    FIRST_LINE_AT = datetime(2026, 10, 3, 14, 59, tzinfo=UTC)
    LATER_LINE_AT = datetime(2026, 10, 3, 15, 1, tzinfo=UTC)
    PLACE = UsageKey(account_id=1, bucket_hour="2026-10-03T14", model="model-x")
    LATER_PLACE = UsageKey(account_id=2, bucket_hour="2026-10-03T15", model="model-y")
    TOKEN_FIELDS = ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens")

    def line(
        self,
        key: bytes = b"key-1",
        *,
        at: datetime = FIRST_LINE_AT,
        account: int = 1,
        model: str = "model-x",
        **counts: int,
    ) -> MessageUsage:
        return MessageUsage(key, at, account, model, TokenCounts(**counts))

    def later_line(self, key: bytes = b"key-1", **counts: int) -> MessageUsage:
        return self.line(key, at=self.LATER_LINE_AT, account=2, model="model-y", **counts)

    def usage(self, database: Database) -> dict[UsageKey, TokenCounts]:
        with database.read() as conn:
            return {
                UsageKey(row.account_id, row.bucket_hour, row.model): row.counts
                for row in db.token_usage(conn, "0000", "9999")
            }

    def remembered(self, database: Database, *keys: bytes) -> dict[bytes, SeenMessage]:
        with database.read() as conn:
            return db.seen_messages(conn, list(keys))

    def test_a_response_seen_for_the_first_time_adds_all_of_its_counts_as_one_request(self, database: Database) -> None:
        counts = TokenCounts(
            input_tokens=5, cache_read_tokens=100, cache_write_tokens=7, output_tokens=30, reasoning_tokens=2
        )
        with database.connect() as conn:
            db.add_message_usage(conn, [MessageUsage(b"key-1", self.FIRST_LINE_AT, 1, "model-x", counts)])

        assert self.usage(database) == {self.PLACE: counts.plus(TokenCounts(requests=1))}
        assert self.remembered(database, b"key-1", b"key-2") == {b"key-1": SeenMessage(self.PLACE, counts)}

    def test_a_response_is_counted_in_the_utc_hour_of_its_line(self, database: Database) -> None:
        late_evening_eastern = datetime(2026, 10, 3, 23, 30, tzinfo=ZoneInfo("America/New_York"))
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(at=late_evening_eastern, output_tokens=3)])

        assert list(self.usage(database)) == [UsageKey(1, "2026-10-04T03", "model-x")]

    def test_a_line_with_a_naive_time_is_refused_and_nothing_is_counted(self, database: Database) -> None:
        naive = self.line(at=datetime(2026, 10, 3, 14, 59), output_tokens=3)

        with pytest.raises(ValueError, match="naive"), database.connect() as conn:
            db.add_message_usage(conn, [self.line(b"key-0", output_tokens=1), naive])

        assert self.usage(database) == {}
        assert self.remembered(database, b"key-0", b"key-1") == {}

    @pytest.mark.parametrize("requests", [0, 1, 5])
    def test_a_response_is_one_request_whatever_request_count_its_line_carries(
        self, database: Database, requests: int
    ) -> None:
        line = self.line(output_tokens=3, requests=requests)
        with database.connect() as conn:
            db.add_message_usage(conn, [line])
            db.add_message_usage(conn, [line])

        assert self.usage(database) == {self.PLACE: TokenCounts(output_tokens=3, requests=1)}
        assert self.remembered(database, b"key-1") == {b"key-1": SeenMessage(self.PLACE, TokenCounts(output_tokens=3))}

    def test_a_repeat_adds_nothing_however_often_it_is_read(self, database: Database) -> None:
        line = self.line(input_tokens=5, output_tokens=30)
        with database.connect() as conn:
            db.add_message_usage(conn, [line])
            db.add_message_usage(conn, [line])
            db.add_message_usage(conn, [line, line])

        assert self.usage(database) == {self.PLACE: TokenCounts(input_tokens=5, output_tokens=30, requests=1)}

    def test_a_later_line_adds_for_each_count_only_what_exceeds_the_largest_already_counted(
        self, database: Database
    ) -> None:
        """A response is written several times while it streams, and any count may be larger or smaller later."""
        largest = TokenCounts(
            input_tokens=5, cache_read_tokens=100, cache_write_tokens=7, output_tokens=100, reasoning_tokens=40
        )
        with database.connect() as conn:
            db.add_message_usage(
                conn, [self.line(input_tokens=5, cache_read_tokens=100, cache_write_tokens=7, output_tokens=30)]
            )
            db.add_message_usage(
                conn,
                [
                    self.line(
                        input_tokens=5,
                        cache_read_tokens=90,
                        cache_write_tokens=7,
                        output_tokens=100,
                        reasoning_tokens=40,
                    )
                ],
            )

        assert self.usage(database) == {self.PLACE: largest.plus(TokenCounts(requests=1))}
        assert self.remembered(database, b"key-1") == {b"key-1": SeenMessage(self.PLACE, largest)}

    @pytest.mark.parametrize("field", TOKEN_FIELDS)
    def test_a_later_line_that_revises_one_count_and_states_no_other_usage_adds_that_revision(
        self, database: Database, field: str
    ) -> None:
        first = TokenCounts(input_tokens=5, cache_read_tokens=5, cache_write_tokens=5, output_tokens=5)
        with database.connect() as conn:
            db.add_message_usage(conn, [MessageUsage(b"key-1", self.FIRST_LINE_AT, 1, "model-x", first)])
            db.add_message_usage(conn, [self.line(**{field: 40})])

        expected = first.plus(TokenCounts(**{field: 40 - getattr(first, field)})).plus(TokenCounts(requests=1))
        assert self.usage(database) == {self.PLACE: expected}

    def test_a_later_line_with_smaller_counts_takes_nothing_away_and_adds_nothing(self, database: Database) -> None:
        larger = TokenCounts(input_tokens=5, output_tokens=100, reasoning_tokens=40)
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(input_tokens=5, output_tokens=100, reasoning_tokens=40)])
            db.add_message_usage(conn, [self.line(output_tokens=30)])

        assert self.usage(database) == {self.PLACE: larger.plus(TokenCounts(requests=1))}
        assert self.remembered(database, b"key-1") == {b"key-1": SeenMessage(self.PLACE, larger)}

    def test_a_later_line_adds_where_the_earliest_line_put_the_response(self, database: Database) -> None:
        """The later line was written in another hour, under another login, and names another model."""
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(output_tokens=30)])
            db.add_message_usage(conn, [self.later_line(output_tokens=100)])

        assert self.usage(database) == {self.PLACE: TokenCounts(output_tokens=100, requests=1)}
        assert self.remembered(database, b"key-1") == {
            b"key-1": SeenMessage(self.PLACE, TokenCounts(output_tokens=100))
        }

    def test_an_earlier_line_read_afterwards_moves_everything_counted_to_its_own_place(
        self, database: Database
    ) -> None:
        """A resumed transcript can be read before the one that holds the start of the response."""
        with database.connect() as conn:
            db.add_message_usage(conn, [self.later_line(output_tokens=100, reasoning_tokens=40)])
            assert db.seen_messages(conn, [b"key-1"])[b"key-1"].usage_key == self.LATER_PLACE

            db.add_message_usage(conn, [self.line(output_tokens=30, cache_read_tokens=2000)])

        moved = TokenCounts(cache_read_tokens=2000, output_tokens=100, reasoning_tokens=40)
        assert self.usage(database) == {self.PLACE: moved.plus(TokenCounts(requests=1))}
        assert self.remembered(database, b"key-1") == {b"key-1": SeenMessage(self.PLACE, moved)}

    def test_moving_a_response_leaves_the_other_usage_of_the_hour_it_leaves(self, database: Database) -> None:
        with database.connect() as conn:
            db.add_message_usage(
                conn, [self.later_line(output_tokens=100), self.later_line(b"key-2", input_tokens=9, output_tokens=1)]
            )
            db.add_message_usage(conn, [self.line(output_tokens=30)])

        assert self.usage(database) == {
            self.PLACE: TokenCounts(output_tokens=100, requests=1),
            self.LATER_PLACE: TokenCounts(input_tokens=9, output_tokens=1, requests=1),
        }

    @pytest.mark.parametrize("one_call", [True, False])
    def test_the_same_lines_give_the_same_rows_in_whatever_order_and_grouping_they_are_read(
        self, tmp_path: Path, one_call: bool
    ) -> None:
        """Transcripts are read in file-name order, and a scan can run between any two lines."""
        lines = [
            self.line(output_tokens=30, cache_read_tokens=2000),
            self.later_line(output_tokens=100, reasoning_tokens=40),
            self.line(at=datetime(2026, 10, 3, 15, 30, tzinfo=UTC), account=2, model="model-y", output_tokens=60),
            self.later_line(b"key-2", input_tokens=9),
        ]
        outcomes = []
        for number, ordering in enumerate(itertools.permutations(lines)):
            store = Database(tmp_path / f"order-{number}.db")
            store.initialize()
            with store.connect() as conn:
                for batch in [list(ordering)] if one_call else [[line] for line in ordering]:
                    db.add_message_usage(conn, batch)
            outcomes.append((self.usage(store), self.remembered(store, b"key-1", b"key-2")))

        counted = TokenCounts(cache_read_tokens=2000, output_tokens=100, reasoning_tokens=40)
        assert len(outcomes) == 24
        assert all(outcome == outcomes[0] for outcome in outcomes)
        assert outcomes[0] == (
            {
                self.PLACE: counted.plus(TokenCounts(requests=1)),
                self.LATER_PLACE: TokenCounts(input_tokens=9, requests=1),
            },
            {
                b"key-1": SeenMessage(self.PLACE, counted),
                b"key-2": SeenMessage(self.LATER_PLACE, TokenCounts(input_tokens=9)),
            },
        )

    @pytest.mark.parametrize("first_read", ["model-a", "model-b"])
    def test_lines_written_at_the_same_instant_are_placed_by_model_name_whichever_is_read_first(
        self, database: Database, first_read: str
    ) -> None:
        models = [first_read, "model-b" if first_read == "model-a" else "model-a"]
        with database.connect() as conn:
            for model in models:
                db.add_message_usage(conn, [self.line(model=model, output_tokens=30)])

        assert self.usage(database) == {
            UsageKey(1, "2026-10-03T14", "model-a"): TokenCounts(output_tokens=30, requests=1)
        }

    @pytest.mark.parametrize("first_read", [1, 2])
    @pytest.mark.parametrize("one_call", [True, False])
    def test_lines_written_at_the_same_instant_by_the_same_model_are_placed_by_account_whichever_is_read_first(
        self, database: Database, first_read: int, one_call: bool
    ) -> None:
        lines = [self.line(account=account, output_tokens=30) for account in (first_read, 3 - first_read)]
        with database.connect() as conn:
            for batch in [lines] if one_call else [[line] for line in lines]:
                db.add_message_usage(conn, batch)

        assert self.usage(database) == {
            UsageKey(1, "2026-10-03T14", "model-x"): TokenCounts(output_tokens=30, requests=1)
        }

    def test_an_earlier_line_in_the_same_hour_from_the_same_login_and_model_changes_no_row(
        self, database: Database
    ) -> None:
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(output_tokens=30)])
            db.add_message_usage(conn, [self.line(at=datetime(2026, 10, 3, 14, 10, tzinfo=UTC), output_tokens=30)])

        assert self.usage(database) == {self.PLACE: TokenCounts(output_tokens=30, requests=1)}

    def test_no_lines_change_nothing(self, database: Database) -> None:
        with database.connect() as conn:
            db.add_message_usage(conn, [])

        assert self.usage(database) == {}

    def test_more_keys_than_one_query_allows_are_all_counted_once(self, database: Database) -> None:
        lines = [self.line(index.to_bytes(4, "big"), output_tokens=1) for index in range(1205)]
        with database.connect() as conn:
            db.add_message_usage(conn, lines)
            db.add_message_usage(conn, lines)

        assert self.usage(database) == {self.PLACE: TokenCounts(output_tokens=1205, requests=1205)}
        assert len(self.remembered(database, *(line.key for line in lines))) == 1205

    def test_claude_usage_is_stored_under_one_file_key_whatever_transcripts_it_was_read_from(
        self, database: Database
    ) -> None:
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(output_tokens=30), self.later_line(b"key-2", output_tokens=5)])
            file_keys = [row["file_key"] for row in conn.execute("SELECT DISTINCT provider, file_key FROM token_usage")]

        assert file_keys == [db.CLAUDE_USAGE_FILE_KEY]

    def test_token_usage_and_the_keys_that_guard_it_expire_on_the_same_day(
        self, database: Database, now: datetime
    ) -> None:
        """A key that expired before its usage would let a copy of that response be counted again."""
        first_kept_day = (now - timedelta(days=400)).replace(hour=0, minute=0)
        moments = [first_kept_day - timedelta(hours=24), first_kept_day - timedelta(hours=1), first_kept_day]
        moments.append(first_kept_day + timedelta(hours=23))
        with database.connect() as conn:
            for moment in moments:
                db.add_message_usage(conn, [self.line(moment.isoformat().encode(), at=moment, output_tokens=1)])

            db.prune(conn, now)

        kept = [moment.strftime("%Y-%m-%dT%H") for moment in moments[2:]]
        assert sorted(key.bucket_hour for key in self.usage(database)) == kept
        assert sorted(self.remembered(database, *(moment.isoformat().encode() for moment in moments))) == [
            moment.isoformat().encode() for moment in moments[2:]
        ]

    def test_a_response_revised_on_a_later_day_expires_whole_with_its_key(
        self, database: Database, now: datetime
    ) -> None:
        """The later line's usage is stored with the first line's, so no part of it is left without its key."""
        first_kept_day = (now - timedelta(days=400)).replace(hour=0, minute=0)
        last_expired_hour = first_kept_day - timedelta(minutes=1)
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(at=last_expired_hour, output_tokens=30)])
            db.add_message_usage(conn, [self.line(at=first_kept_day, output_tokens=100)])
            db.add_message_usage(conn, [self.line(b"key-2", at=first_kept_day, output_tokens=7)])
            before_pruning = {row.bucket_hour: row.counts for row in db.token_usage(conn, "0000", "9999")}

            db.prune(conn, now)

        assert before_pruning == {
            last_expired_hour.strftime("%Y-%m-%dT%H"): TokenCounts(output_tokens=100, requests=1),
            first_kept_day.strftime("%Y-%m-%dT%H"): TokenCounts(output_tokens=7, requests=1),
        }
        assert list(self.usage(database).values()) == [TokenCounts(output_tokens=7, requests=1)]
        assert list(self.remembered(database, b"key-1", b"key-2")) == [b"key-2"]

    def test_a_response_moved_to_an_earlier_line_takes_its_key_to_that_hour_and_expires_with_it(
        self, database: Database, now: datetime
    ) -> None:
        first_kept_day = (now - timedelta(days=400)).replace(hour=0, minute=0)
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(at=first_kept_day, output_tokens=100)])
            db.add_message_usage(conn, [self.line(at=first_kept_day - timedelta(minutes=1), output_tokens=30)])

            db.prune(conn, now)

        assert self.usage(database) == {}
        assert self.remembered(database, b"key-1") == {}

    def test_a_response_revised_on_a_later_day_is_not_counted_again_by_a_copy_read_after_pruning(
        self, database: Database, now: datetime
    ) -> None:
        """Pruning between the two days must leave the key for as long as any of the response's usage is kept."""
        last_kept_hour_of_the_first_kept_day = (now - timedelta(days=400)).replace(hour=23, minute=59)
        next_day = last_kept_hour_of_the_first_kept_day + timedelta(minutes=1)
        lines = [
            self.line(at=last_kept_hour_of_the_first_kept_day, output_tokens=30),
            self.line(at=next_day, output_tokens=100),
        ]
        with database.connect() as conn:
            db.add_message_usage(conn, lines[:1])
            db.add_message_usage(conn, lines[1:])
            db.prune(conn, now)
            db.add_message_usage(conn, lines)

        assert list(self.usage(database).values()) == [TokenCounts(output_tokens=100, requests=1)]
        assert [key.bucket_hour for key in self.usage(database)] == [
            last_kept_hour_of_the_first_kept_day.strftime("%Y-%m-%dT%H")
        ]

    def expired_response_lines(self, now: datetime) -> list[MessageUsage]:
        """Two lines of one response, half an hour either side of the start of the first day retention keeps."""
        first_kept_day = (now - timedelta(days=400)).replace(hour=0, minute=0)
        return [
            self.line(at=first_kept_day - timedelta(minutes=30), output_tokens=30),
            self.line(at=first_kept_day + timedelta(minutes=30), output_tokens=100),
        ]

    @pytest.mark.parametrize(
        "replayed",
        [
            pytest.param([[1]], id="the later line alone"),
            pytest.param([[0, 1]], id="both lines together"),
            pytest.param([[1], [0]], id="the later line, then the earlier"),
            pytest.param([[0], [1]], id="the earlier line, then the later"),
        ],
    )
    def test_a_response_whose_usage_expired_is_never_counted_again_whatever_lines_are_read(
        self, database: Database, now: datetime, replayed: list[list[int]]
    ) -> None:
        """Counted from a line still inside retention it came back under that line's hour."""
        lines = self.expired_response_lines(now)
        with database.connect() as conn:
            db.add_message_usage(conn, lines)
            db.prune(conn, now)
            for call in replayed:
                db.add_message_usage(conn, [lines[index] for index in call])
            db.prune(conn, now)

        assert self.usage(database) == {}
        assert self.remembered(database, b"key-1") == {}

    def test_a_response_that_expired_stays_expired_through_later_pruning(
        self, database: Database, now: datetime
    ) -> None:
        lines = self.expired_response_lines(now)
        with database.connect() as conn:
            db.add_message_usage(conn, lines)
            db.prune(conn, now)
            db.prune(conn, now + timedelta(days=30))
            db.add_message_usage(conn, lines[1:])

        assert self.usage(database) == {}

    def test_other_responses_are_counted_as_ever_once_one_has_expired(self, database: Database, now: datetime) -> None:
        first_kept_day = (now - timedelta(days=400)).replace(hour=0, minute=0)
        with database.connect() as conn:
            lines = self.expired_response_lines(now)
            db.add_message_usage(conn, lines)
            db.prune(conn, now)
            in_the_same_call = self.line(b"key-2", at=first_kept_day + timedelta(minutes=5), output_tokens=7)
            db.add_message_usage(conn, [lines[1], in_the_same_call])

        assert list(self.usage(database).values()) == [TokenCounts(output_tokens=7, requests=1)]

    def test_a_response_still_inside_retention_still_takes_its_lines_after_pruning(
        self, database: Database, now: datetime
    ) -> None:
        first_kept_day = (now - timedelta(days=400)).replace(hour=0, minute=0)
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(at=first_kept_day + timedelta(hours=1), output_tokens=30)])
            db.prune(conn, now)
            db.add_message_usage(conn, [self.line(at=first_kept_day + timedelta(hours=2), output_tokens=100)])

        assert list(self.usage(database).values()) == [TokenCounts(output_tokens=100, requests=1)]

    def test_usage_within_retention_is_never_pruned(self, database: Database, now: datetime) -> None:
        with database.connect() as conn:
            db.add_file_usage(conn, Provider.CODEX, "f", {self.PLACE: TokenCounts(input_tokens=5, requests=1)})
            db.add_message_usage(conn, [self.line(b"recent", output_tokens=1)])
            db.prune(conn, now)
            kept = [(row.provider, row.counts) for row in db.token_usage(conn, "0000", "9999")]

        assert kept == [
            (Provider.CLAUDE, TokenCounts(output_tokens=1, requests=1)),
            (Provider.CODEX, TokenCounts(input_tokens=5, requests=1)),
        ]
        assert list(self.remembered(database, b"recent")) == [b"recent"]

    @pytest.mark.parametrize("field", TOKEN_FIELDS)
    @pytest.mark.parametrize("count", [MAX_TOKEN_COUNT + 1, 2**63, 10**400])
    def test_a_response_with_a_count_no_stored_row_can_hold_is_refused_and_not_remembered(
        self, database: Database, field: str, count: int
    ) -> None:
        with pytest.raises(ValueError, match="outside what one stored row can hold"), database.connect() as conn:
            db.add_message_usage(conn, [self.line(**{field: count})])

        assert self.usage(database) == {}
        assert self.remembered(database, b"key-1") == {}

    def test_a_response_with_every_count_at_the_limit_is_counted_and_remembered(self, database: Database) -> None:
        at_the_limit = TokenCounts(**dict.fromkeys(self.TOKEN_FIELDS, MAX_TOKEN_COUNT))
        with database.connect() as conn:
            db.add_message_usage(conn, [MessageUsage(b"key-1", self.FIRST_LINE_AT, 1, "model-x", at_the_limit)])

        assert self.usage(database) == {self.PLACE: at_the_limit.plus(TokenCounts(requests=1))}
        assert self.remembered(database, b"key-1") == {b"key-1": SeenMessage(self.PLACE, at_the_limit)}

    def test_a_later_line_that_takes_a_count_past_the_limit_is_refused_and_changes_nothing(
        self, database: Database
    ) -> None:
        at_the_limit = TokenCounts(output_tokens=MAX_TOKEN_COUNT)
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(output_tokens=MAX_TOKEN_COUNT)])

        with pytest.raises(ValueError, match="more than one stored row can hold"), database.connect() as conn:
            db.add_message_usage(conn, [self.line(output_tokens=MAX_TOKEN_COUNT + 1)])

        assert self.usage(database) == {self.PLACE: at_the_limit.plus(TokenCounts(requests=1))}
        assert self.remembered(database, b"key-1") == {b"key-1": SeenMessage(self.PLACE, at_the_limit)}

    def test_a_response_at_the_limit_can_still_be_moved_to_its_earlier_line(self, database: Database) -> None:
        """What is moved out of a row and what is moved into it are netted, so a move adds nothing on the way."""
        at_the_limit = TokenCounts(output_tokens=MAX_TOKEN_COUNT)
        earlier_in_the_hour = self.line(at=datetime(2026, 10, 3, 14, 10, tzinfo=UTC), output_tokens=MAX_TOKEN_COUNT)
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(output_tokens=MAX_TOKEN_COUNT)])
            db.add_message_usage(conn, [earlier_in_the_hour])

        assert self.usage(database) == {self.PLACE: at_the_limit.plus(TokenCounts(requests=1))}

    TAMPERING = (
        "DELETE FROM token_usage",
        "UPDATE token_usage SET output_tokens = 1",
        "UPDATE token_usage SET requests = 0",
    )

    @pytest.mark.parametrize("tampering", TAMPERING)
    def test_usage_that_is_no_longer_in_the_row_it_would_be_moved_from_is_a_store_error_and_nothing_changes(
        self, database: Database, tampering: str
    ) -> None:
        with database.connect() as conn:
            db.add_message_usage(conn, [self.later_line(output_tokens=100)])
            conn.execute(tampering)
        before = self.usage(database)

        with pytest.raises(StoreError, match="do not hold what was counted"), database.connect() as conn:
            db.add_message_usage(conn, [self.line(output_tokens=30)])

        assert self.usage(database) == before
        assert self.remembered(database, b"key-1") == {
            b"key-1": SeenMessage(self.LATER_PLACE, TokenCounts(output_tokens=100))
        }

    @pytest.mark.parametrize("tampering", TAMPERING)
    def test_usage_the_same_call_adds_to_that_row_cannot_stand_in_for_what_is_missing_from_it(
        self, database: Database, tampering: str
    ) -> None:
        """The row is checked before anything is added to it, not after."""
        with database.connect() as conn:
            db.add_message_usage(conn, [self.later_line(output_tokens=100)])
            conn.execute(tampering)
        before = self.usage(database)
        another_response_in_that_hour = self.later_line(b"key-2", output_tokens=500)

        with pytest.raises(StoreError, match="do not hold what was counted"), database.connect() as conn:
            db.add_message_usage(conn, [self.line(output_tokens=30), another_response_in_that_hour])

        assert self.usage(database) == before
        assert self.remembered(database, b"key-2") == {}

    @pytest.mark.parametrize("tampering", TAMPERING)
    def test_a_line_that_moves_nothing_is_still_refused_when_its_response_s_usage_is_missing(
        self, database: Database, tampering: str
    ) -> None:
        """An earlier line in the same hour changes no row; the row is checked all the same."""
        earlier_in_the_hour = self.later_line(output_tokens=100)
        with database.connect() as conn:
            db.add_message_usage(conn, [self.later_line(output_tokens=100)])
            conn.execute(tampering)

        with pytest.raises(StoreError, match="do not hold what was counted"), database.connect() as conn:
            db.add_message_usage(conn, [earlier_in_the_hour])

    def test_a_row_short_of_another_response_s_usage_refuses_a_move_of_the_one_it_holds(
        self, database: Database
    ) -> None:
        """Checked against the moved response alone, 120 covered its 100, and the other was left with 20 of 40."""
        with database.connect() as conn:
            db.add_message_usage(
                conn, [self.later_line(output_tokens=100), self.later_line(b"key-2", output_tokens=40)]
            )
            conn.execute("UPDATE token_usage SET output_tokens = 120")
        before = self.usage(database)

        with pytest.raises(StoreError, match="do not hold what was counted"), database.connect() as conn:
            db.add_message_usage(conn, [self.line(output_tokens=100)])

        assert self.usage(database) == before
        assert self.remembered(database, b"key-1") == {
            b"key-1": SeenMessage(self.LATER_PLACE, TokenCounts(output_tokens=100))
        }

    def test_a_row_short_of_what_its_responses_counted_refuses_a_new_response(self, database: Database) -> None:
        """No response the call touches was in that row before, and it was not checked at all."""
        with database.connect() as conn:
            db.add_message_usage(conn, [self.later_line(output_tokens=100)])
            conn.execute("UPDATE token_usage SET output_tokens = 50")
        before = self.usage(database)

        with pytest.raises(StoreError, match="do not hold what was counted"), database.connect() as conn:
            db.add_message_usage(conn, [self.later_line(b"key-2", output_tokens=1)])

        assert self.usage(database) == before
        assert self.remembered(database, b"key-2") == {}

    def test_a_row_holding_more_than_its_responses_counted_is_refused_too(self, database: Database) -> None:
        """Only the store writes these rows, so usage no key accounts for came from somewhere else."""
        with database.connect() as conn:
            db.add_message_usage(conn, [self.later_line(output_tokens=100)])
            conn.execute("UPDATE token_usage SET requests = requests + 1")

        with pytest.raises(StoreError, match="do not hold what was counted"), database.connect() as conn:
            db.add_message_usage(conn, [self.later_line(b"key-2", output_tokens=1)])

    def test_a_row_that_holds_exactly_what_its_responses_counted_takes_a_new_response(self, database: Database) -> None:
        with database.connect() as conn:
            db.add_message_usage(
                conn, [self.later_line(output_tokens=100), self.later_line(b"key-2", output_tokens=40)]
            )
            db.add_message_usage(conn, [self.later_line(b"key-3", output_tokens=1)])

        assert self.usage(database) == {self.LATER_PLACE: TokenCounts(output_tokens=141, requests=3)}

    @pytest.mark.parametrize("file_key", [db.CLAUDE_USAGE_FILE_KEY, "a.jsonl"])
    def test_claude_usage_cannot_be_added_by_file_and_nothing_is_stored(
        self, database: Database, file_key: str
    ) -> None:
        """Written by file, a response counted again through its key was counted twice."""
        with pytest.raises(ValueError, match="cannot be added by file"), database.connect() as conn:
            db.add_file_usage(conn, Provider.CLAUDE, file_key, {self.LATER_PLACE: TokenCounts(input_tokens=5)})
        with database.connect() as conn:
            db.add_message_usage(conn, [self.later_line(input_tokens=5)])

        assert self.usage(database) == {self.LATER_PLACE: TokenCounts(input_tokens=5, requests=1)}

    @pytest.mark.parametrize("operation", ["clear", "replace"])
    @pytest.mark.parametrize("file_key", [db.CLAUDE_USAGE_FILE_KEY, "a.jsonl"])
    def test_claude_usage_cannot_be_cleared_or_replaced_by_file(
        self, database: Database, operation: str, file_key: str
    ) -> None:
        """The rows would go and the keys of their responses would stay, so reading again would add nothing."""
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(output_tokens=30)])

        with pytest.raises(ValueError, match="cannot be cleared or replaced by file"), database.connect() as conn:
            if operation == "clear":
                db.clear_file_usage(conn, Provider.CLAUDE, file_key)
            else:
                db.replace_file_usage(conn, Provider.CLAUDE, file_key, {self.PLACE: TokenCounts(output_tokens=1)})

        assert self.usage(database) == {self.PLACE: TokenCounts(output_tokens=30, requests=1)}
        assert self.remembered(database, b"key-1") == {b"key-1": SeenMessage(self.PLACE, TokenCounts(output_tokens=30))}

    def test_clearing_another_tool_s_file_of_the_same_name_leaves_claude_usage_and_its_keys_alone(
        self, database: Database
    ) -> None:
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(output_tokens=30)])
            db.add_file_usage(
                conn, Provider.CODEX, db.CLAUDE_USAGE_FILE_KEY, {self.LATER_PLACE: TokenCounts(input_tokens=5)}
            )

            db.clear_file_usage(conn, Provider.CODEX, db.CLAUDE_USAGE_FILE_KEY)

        assert self.usage(database) == {self.PLACE: TokenCounts(output_tokens=30, requests=1)}
        assert list(self.remembered(database, b"key-1")) == [b"key-1"]


PROCESS_WAIT_SECONDS = 60.0
INITIALIZE_WHEN_TOLD = """
import sys, time
from pathlib import Path
from usagedash.db import Database

database, ready, go = (Path(argument) for argument in sys.argv[1:4])
ready.touch()
deadline = time.monotonic() + float(sys.argv[4])
while not go.exists():
    if time.monotonic() > deadline:
        sys.exit("never told to start")
    time.sleep(0.0005)
Database(database).initialize()
"""


class TestALineThatNamesAnAccountOutranksOneThatNamesNone:
    def line(self, account_id: int, at: datetime, model: str = "claude-model-x") -> MessageUsage:
        return MessageUsage(b"response-1", at, account_id, model, TokenCounts(output_tokens=100, requests=1))

    def credited(self, database: Database) -> dict[tuple[int, str], int]:
        with database.read() as conn:
            rows = conn.execute(
                "SELECT account_id, model, SUM(output_tokens) AS output FROM token_usage GROUP BY account_id, model"
            ).fetchall()
        return {(row["account_id"], row["model"]): row["output"] for row in rows if row["output"]}

    @pytest.mark.parametrize("order", [(7, 0), (0, 7)], ids=["the account's line first", "the unattributed line first"])
    def test_in_one_call_or_in_two_the_account_keeps_the_response(
        self, database: Database, now: datetime, order: tuple[int, int]
    ) -> None:
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(order[0], now)])
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(order[1], now)])

        assert self.credited(database) == {(7, "claude-model-x"): 100}

    def test_it_outranks_it_whatever_the_two_models_are_called(self, database: Database, now: datetime) -> None:
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(0, now, model="a-model"), self.line(7, now, model="z-model")])

        assert self.credited(database) == {(7, "z-model"): 100}

    def test_an_earlier_unattributed_line_still_places_the_response(self, database: Database, now: datetime) -> None:
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(7, now), self.line(0, now - timedelta(seconds=1))])

        assert self.credited(database) == {(0, "claude-model-x"): 100}

    def test_two_accounts_at_one_instant_are_still_ordered_by_model_and_then_account(
        self, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn:
            db.add_message_usage(conn, [self.line(9, now), self.line(7, now)])

        assert self.credited(database) == {(7, "claude-model-x"): 100}


class TestSchemaLifecycle:
    def test_a_version_1_database_gains_the_recorded_login_boundary_table(self, tmp_path: Path, now: datetime) -> None:
        database = Database(tmp_path / "data" / "usage.db")
        database.initialize()
        with database.connect() as conn:
            db.upsert_account(conn, make_identity(), now)
            conn.execute("DROP TABLE recorded_login_boundaries")
            conn.execute("PRAGMA user_version = 1")

        database.initialize()

        with database.connect() as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
            assert len(db.list_accounts(conn)) == 1
            present = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?",
                ("table", "recorded_login_boundaries"),
            ).fetchone()
            assert present is not None

    def test_a_schema_version_with_no_migration_is_left_unchanged(
        self, database: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(db, "_MIGRATIONS", {})
        with database.connect() as conn:
            conn.execute("DROP TABLE recorded_login_boundaries")
            conn.execute("PRAGMA user_version = 1")

        with pytest.raises(StoreError, match=r"schema version 1 has no migration"):
            database.initialize()

        with database.connect() as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
            present = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?",
                ("table", "recorded_login_boundaries"),
            ).fetchone()
            assert present is None

    def test_initializing_twice_keeps_existing_data(self, database: Database, now: datetime) -> None:
        with database.connect() as conn:
            db.upsert_account(conn, make_identity(), now)
        database.initialize()
        with database.connect() as conn:
            assert len(db.list_accounts(conn)) == 1

    def test_a_newer_schema_is_refused_instead_of_misread(self, database: Database) -> None:
        with database.connect() as conn:
            conn.execute("PRAGMA user_version = 99")
        with pytest.raises(StoreError, match=rf"schema version 99 is not supported \(expected {db.SCHEMA_VERSION}\)"):
            database.initialize()

    def test_a_file_that_is_not_a_database_is_a_store_error_and_is_left_as_it_was(self, tmp_path: Path) -> None:
        path = tmp_path / "data" / "usage.db"
        path.parent.mkdir()
        path.write_bytes(b"this is not an SQLite database, it is a text file that happens to be here\n" * 40)
        before = path.read_bytes()

        with pytest.raises(StoreError, match=r"cannot set up the database .*usage\.db: DatabaseError") as raised:
            Database(path).initialize()

        assert isinstance(raised.value.__cause__, sqlite3.DatabaseError)
        assert path.read_bytes() == before

    def test_a_directory_where_the_database_file_should_be_is_a_config_error(self, tmp_path: Path) -> None:
        path = tmp_path / "data" / "usage.db"
        path.mkdir(parents=True)

        with pytest.raises(ConfigError, match=r"cannot open .*usage\.db as a private file: IsADirectoryError"):
            Database(path).initialize()

    def test_a_database_path_that_is_a_symbolic_link_is_refused_and_nothing_is_written_through_it(
        self, tmp_path: Path
    ) -> None:
        elsewhere = tmp_path / "elsewhere" / "usage.db"
        elsewhere.parent.mkdir()
        path = tmp_path / "data" / "usage.db"
        path.parent.mkdir()
        path.symlink_to(elsewhere)

        with pytest.raises(ConfigError, match=r"cannot open .*usage\.db as a private file"):
            Database(path).initialize()

        assert not elsewhere.exists()

    def test_an_existing_database_file_other_accounts_could_read_is_made_private(self, tmp_path: Path) -> None:
        path = tmp_path / "data" / "usage.db"
        Database(path).initialize()
        path.chmod(0o644)

        Database(path).initialize()

        assert path.stat().st_mode & 0o777 == 0o600

    def test_the_database_file_is_private_to_the_user(self, database: Database) -> None:
        assert database.path.stat().st_mode & 0o777 == 0o600
        assert database.path.parent.stat().st_mode & 0o777 == 0o700

    def test_a_failed_transaction_leaves_no_partial_write(self, database: Database, now: datetime) -> None:
        with pytest.raises(RuntimeError, match="abort"), database.connect() as conn:
            db.upsert_account(conn, make_identity(), now)
            raise RuntimeError("abort")
        with database.connect() as conn:
            assert db.list_accounts(conn) == []

    def test_a_schema_that_fails_part_way_leaves_an_empty_database_that_can_be_initialized_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        database = Database(tmp_path / "data" / "usage.db")
        statements = db._SCHEMA_STATEMENTS
        monkeypatch.setattr(db, "_SCHEMA_STATEMENTS", (*statements[:3], "CREATE TABLE broken ("))

        with pytest.raises(StoreError, match="cannot set up the database") as raised:
            database.initialize()

        assert isinstance(raised.value.__cause__, sqlite3.OperationalError)
        with database.connect() as conn:
            assert conn.execute("SELECT name FROM sqlite_master").fetchall() == []
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 0

        monkeypatch.setattr(db, "_SCHEMA_STATEMENTS", statements)
        database.initialize()

        with database.connect() as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
            assert db.list_accounts(conn) == []

    def test_processes_initializing_the_same_new_database_together_all_succeed(self, tmp_path: Path) -> None:
        path = tmp_path / "data" / "usage.db"
        starters = 8
        together = threading.Barrier(starters)

        def initialize(_: int) -> None:
            together.wait(timeout=5)
            Database(path).initialize()

        with ThreadPoolExecutor(max_workers=starters) as pool:
            list(pool.map(initialize, range(starters)))

        with Database(path).connect() as conn:
            tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        created = sum("CREATE TABLE" in statement for statement in db._SCHEMA_STATEMENTS)
        assert len(tables) == len(set(tables)) == created

    def test_real_processes_initializing_the_same_new_database_together_all_succeed(self, tmp_path: Path) -> None:
        """Each process has its own locks, as the agent and a command run by hand do.

        Without one initialization at a time, about one round in twenty of these failed with "database
        is locked" where the journal mode is set.
        """
        starters, rounds = 6, 3
        environment = os.environ | {"PYTHONPATH": str(Path(db.__file__).parents[1])}
        outcomes: list[tuple[int, str]] = []
        for round_number in range(rounds):
            home = tmp_path / f"round-{round_number}"
            home.mkdir()
            path, go = home / "data" / "usage.db", home / "go"
            ready = [home / f"ready-{starter}" for starter in range(starters)]
            interpreter = [sys.executable, "-c", INITIALIZE_WHEN_TOLD, str(path)]
            children = [
                subprocess.Popen(
                    [*interpreter, str(waiting), str(go), str(PROCESS_WAIT_SECONDS)],
                    env=environment,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for waiting in ready
            ]
            deadline = time.monotonic() + PROCESS_WAIT_SECONDS
            while not all(waiting.exists() for waiting in ready) and time.monotonic() < deadline:
                time.sleep(0.005)
            go.touch()
            for child in children:
                _, errors = child.communicate(timeout=PROCESS_WAIT_SECONDS)
                outcomes.append((child.returncode, errors))
            with Database(path).connect() as conn:
                assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
                assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION

        assert outcomes == [(0, "")] * (starters * rounds)

    def test_initializing_waits_for_whoever_is_initializing_and_touches_nothing_until_then(
        self, tmp_path: Path
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        holder = os.open(data_dir, os.O_RDONLY)
        fcntl.flock(holder, fcntl.LOCK_EX)
        initialized = threading.Event()

        def initialize() -> None:
            Database(data_dir / "usage.db").initialize()
            initialized.set()

        waiting = threading.Thread(target=initialize)
        waiting.start()
        try:
            assert not initialized.wait(timeout=0.3)
            assert list(data_dir.iterdir()) == []
        finally:
            os.close(holder)
            waiting.join(timeout=30)

        assert initialized.is_set()
        with Database(data_dir / "usage.db").connect() as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION

    def test_the_lock_is_released_when_initializing_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "data" / "usage.db"
        path.parent.mkdir()
        path.write_bytes(b"this is not a database, and it is longer than a database header" * 4)

        with pytest.raises(StoreError, match="cannot set up the database"):
            Database(path).initialize()

        probe = os.open(path.parent, os.O_RDONLY)
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe)

    def test_a_data_directory_that_cannot_be_locked_is_a_config_error_and_no_database_is_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        refused: list[int] = []

        def refuse(descriptor: int, operation: int) -> None:
            refused.append(descriptor)
            raise OSError(errno.ENOLCK, "No locks available")

        monkeypatch.setattr(db.fcntl, "flock", refuse)

        with pytest.raises(ConfigError, match=r"cannot lock .*data: OSError$"):
            Database(tmp_path / "data" / "usage.db").initialize()

        assert not (tmp_path / "data" / "usage.db").exists()
        assert len(refused) == 1
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(refused[0])

    def test_a_data_directory_that_is_gone_when_it_is_to_be_locked_is_a_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It was there a moment ago, when it was made private. Nothing stops it being removed since."""
        monkeypatch.setattr(db, "ensure_private_directory", lambda _directory: None)

        with pytest.raises(ConfigError, match=r"cannot lock .*data: FileNotFoundError$"):
            Database(tmp_path / "data" / "usage.db").initialize()

        assert not (tmp_path / "data").exists()

    def test_an_existing_data_directory_other_accounts_can_write_to_is_made_private(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        data_dir.chmod(0o777)

        Database(data_dir / "usage.db").initialize()

        assert data_dir.stat().st_mode & 0o777 == 0o700

    def test_an_existing_database_file_other_accounts_can_read_is_made_private(self, tmp_path: Path) -> None:
        database = Database(tmp_path / "data" / "usage.db")
        database.initialize()
        database.path.chmod(0o644)

        database.initialize()

        assert database.path.stat().st_mode & 0o777 == 0o600

    def test_the_journal_files_sqlite_creates_beside_the_database_are_private_too(
        self, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn:
            db.upsert_account(conn, make_identity(), now)
            beside = [path for path in database.path.parent.iterdir() if path != database.path]

            assert beside != []
            assert {path.stat().st_mode & 0o777 for path in beside} == {0o600}

    def test_a_data_directory_that_cannot_be_made_private_is_a_config_error(self, tmp_path: Path) -> None:
        in_the_way = tmp_path / "data"
        in_the_way.write_text("a file where the directory should be")

        with pytest.raises(ConfigError, match=r"cannot make .* a private directory"):
            Database(in_the_way / "usage.db").initialize()


class TestReadingOrder:
    """Which reading is newest is decided by the account's own watermark, to the microsecond."""

    def record(self, database: Database, reading: QuotaReading) -> int:
        with database.connect() as conn:
            account_id = db.upsert_account(conn, reading.identity, reading.observed_at)
            db.record_reading(conn, account_id, reading)
        return account_id

    def test_a_reading_with_no_windows_still_counts_as_the_newest(self, database: Database, now: datetime) -> None:
        """An empty reading stamps no snapshot row, so only the account can remember when it was taken."""
        identity = make_identity()
        newest_facts = (Fact("Status", "no limits reported"),)
        self.record(database, make_reading(identity, now, make_window("weekly", 40)))
        self.record(database, QuotaReading(identity, (), now + timedelta(hours=2), "api", newest_facts))
        self.record(database, make_reading(identity, now + timedelta(hours=1), make_window("weekly", 99)))

        with database.connect() as conn:
            assert db.latest_snapshots(conn) == []
            assert db.list_accounts(conn)[0].facts == newest_facts
            assert [row.used_percent for row in db.snapshot_history(conn, now - timedelta(days=1))] == [40]

    def test_an_older_reading_from_the_same_second_does_not_replace_the_newer_one(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity()
        late_in_the_second = now.replace(microsecond=900_000)
        early_in_the_second = now.replace(microsecond=100_000)
        self.record(database, make_reading(identity, late_in_the_second, make_window(used_percent=50)))
        self.record(database, make_reading(identity, early_in_the_second, make_window(used_percent=20)))

        with database.connect() as conn:
            assert [row.used_percent for row in db.latest_snapshots(conn)] == [50]
            assert len(db.snapshot_history(conn, now - timedelta(days=1))) == 1

    def test_a_reading_one_microsecond_newer_is_accepted_and_one_microsecond_older_is_not(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity()
        self.record(database, make_reading(identity, now.replace(microsecond=500), make_window(used_percent=10)))
        self.record(database, make_reading(identity, now.replace(microsecond=499), make_window(used_percent=20)))
        with database.connect() as conn:
            assert [row.used_percent for row in db.latest_snapshots(conn)] == [10]

        self.record(database, make_reading(identity, now.replace(microsecond=501), make_window(used_percent=30)))
        with database.connect() as conn:
            assert [row.used_percent for row in db.latest_snapshots(conn)] == [30]

    def test_each_account_has_its_own_watermark(self, database: Database, now: datetime) -> None:
        first = self.record(database, make_reading(make_identity(external_id="a"), now, make_window(used_percent=10)))
        second = self.record(
            database,
            make_reading(make_identity(external_id="b"), now - timedelta(days=3), make_window(used_percent=90)),
        )
        self.record(
            database,
            make_reading(make_identity(external_id="b"), now - timedelta(days=2), make_window(used_percent=95)),
        )

        with database.connect() as conn:
            latest = {row.account_id: row.used_percent for row in db.latest_snapshots(conn)}

        assert latest == {first: 10, second: 95}

    def test_a_reading_for_an_account_that_does_not_exist_is_an_error_not_a_silent_drop(
        self, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn, pytest.raises(ValueError, match="unknown account 999"):
            db.record_reading(conn, 999, make_reading(make_identity(), now, make_window()))


class TestWriteTransactions:
    """Every write goes through ``connect()``, which holds the write lock for the whole block."""

    WRITERS = 8

    def together(self, work: Callable[[int], None]) -> None:
        start = threading.Barrier(self.WRITERS)

        def run(index: int) -> None:
            start.wait(timeout=5)
            work(index)

        with ThreadPoolExecutor(max_workers=self.WRITERS) as pool:
            list(pool.map(run, range(self.WRITERS)))

    def test_a_second_writer_waits_until_the_first_has_finished(self, database: Database) -> None:
        entered = threading.Event()

        def second_writer() -> None:
            with database.connect():
                entered.set()

        with database.connect():
            waiting = threading.Thread(target=second_writer)
            waiting.start()
            assert not entered.wait(0.3)
        waiting.join(timeout=5)

        assert entered.is_set()

    def test_registering_one_account_from_many_writers_at_once_creates_it_once(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity(external_id="shared")

        def register(_: int) -> None:
            with database.connect() as conn:
                db.upsert_account(conn, identity, now)

        self.together(register)

        with database.read() as conn:
            assert [account.external_id for account in db.list_accounts(conn)] == ["shared"]

    def test_accounts_registered_at_once_each_get_their_own_color(self, database: Database, now: datetime) -> None:
        def register(index: int) -> None:
            with database.connect() as conn:
                db.upsert_account(conn, make_identity(external_id=f"org-{index}"), now)

        self.together(register)

        with database.read() as conn:
            slots = sorted(account.color_slot for account in db.list_accounts(conn))
        assert slots == list(range(1, self.WRITERS + 1))

    def test_readings_recorded_at_once_each_take_their_own_place_in_the_order(
        self, database: Database, now: datetime
    ) -> None:
        identity = make_identity()
        with database.connect() as conn:
            account_id = db.upsert_account(conn, identity, now)

        def record(index: int) -> None:
            with database.connect() as conn:
                db.record_reading(conn, account_id, make_reading(identity, now, make_window(used_percent=index)))

        self.together(record)

        with database.read() as conn:
            sequence = conn.execute("SELECT reading_seq FROM accounts WHERE id = ?", (account_id,)).fetchone()[0]
            current = db.latest_snapshots(conn)
        assert sequence == self.WRITERS
        assert len(current) == 1

    def test_a_block_that_raises_leaves_nothing_behind_and_releases_the_lock(
        self, database: Database, now: datetime
    ) -> None:
        with pytest.raises(RuntimeError, match="abort"), database.connect() as conn:
            db.upsert_account(conn, make_identity(), now)
            raise RuntimeError("abort")

        with database.connect() as conn:
            assert db.list_accounts(conn) == []


class TestReadOnlyViews:
    def test_a_view_sees_one_committed_state_whatever_is_written_meanwhile(
        self, database: Database, now: datetime
    ) -> None:
        with database.read() as view:
            assert db.list_accounts(view) == []
            with database.connect() as conn:
                db.upsert_account(conn, make_identity(), now)

            assert db.list_accounts(view) == []

        with database.read() as later:
            assert len(db.list_accounts(later)) == 1

    def test_a_view_does_not_wait_for_a_writer_and_does_not_see_its_uncommitted_work(
        self, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn:
            db.upsert_account(conn, make_identity(), now)
            with database.read() as view:
                assert db.list_accounts(view) == []

    def test_a_write_through_a_view_is_refused(self, database: Database, now: datetime) -> None:
        with database.read() as view, pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.upsert_account(view, make_identity(), now)

        with database.read() as view:
            assert db.list_accounts(view) == []


class TestPercentagesGivenAsIntegers:
    def record(self, database: Database, reading: QuotaReading) -> None:
        with database.connect() as conn:
            db.record_reading(conn, db.upsert_account(conn, reading.identity, reading.observed_at), reading)

    @pytest.mark.parametrize("percent", [0, 100, 2**53 + 1, 2**63, 10**30])
    def test_an_integer_percentage_is_stored_and_an_unchanged_re_read_extends_its_step(
        self, database: Database, now: datetime, percent: int
    ) -> None:
        """An integer SQLite cannot hold, or one a REAL column rounds, must behave like any other value."""
        identity = make_identity()
        self.record(database, make_reading(identity, now, make_window(used_percent=percent)))
        self.record(database, make_reading(identity, now + timedelta(minutes=5), make_window(used_percent=percent)))

        with database.read() as conn:
            history = db.snapshot_history(conn, now - timedelta(days=1))

        assert [(row.used_percent, row.confirmed_at) for row in history] == [
            (float(percent), now + timedelta(minutes=5))
        ]


class TestWhatTheStoreAcceptsCanAlwaysBeReadBack:
    """Each stored row has a limit; sums over rows do not, so no committed total can fail to read."""

    KEY = UsageKey(account_id=1, bucket_hour="2026-10-03T14", model="model-x")
    COUNT_FIELDS = (
        "input_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_tokens",
        "requests",
    )

    def read(self, database: Database) -> list[TokenCounts]:
        with database.read() as conn:
            return [row.counts for row in db.token_usage(conn, "2026-10-03T00", "2026-10-04T00")]

    def test_a_row_at_the_limit_is_stored_and_read_back_exactly(self, database: Database) -> None:
        at_the_limit = TokenCounts(**dict.fromkeys(self.COUNT_FIELDS, MAX_TOKEN_COUNT))

        with database.connect() as conn:
            db.add_file_usage(conn, Provider.CODEX, "a.jsonl", {self.KEY: at_the_limit})

        assert self.read(database) == [at_the_limit]

    @pytest.mark.parametrize("field", COUNT_FIELDS)
    @pytest.mark.parametrize("count", [MAX_TOKEN_COUNT + 1, 2**63, 10**400])
    def test_a_count_no_stored_row_can_hold_is_refused_as_a_value_error_before_reaching_sqlite(
        self, database: Database, field: str, count: int
    ) -> None:
        usage = {self.KEY: TokenCounts(**{field: count, "requests": 1} | {field: count})}

        with (
            pytest.raises(ValueError, match="outside what one stored row can hold"),
            database.connect() as conn,
        ):
            db.add_file_usage(conn, Provider.CODEX, "a.jsonl", usage)

        assert self.read(database) == []

    @pytest.mark.parametrize("field", COUNT_FIELDS)
    def test_adding_to_a_row_past_the_limit_is_refused_and_leaves_the_row_as_it_was(
        self, database: Database, field: str
    ) -> None:
        at_the_limit = TokenCounts(**{field: MAX_TOKEN_COUNT, "requests": 1} | {field: MAX_TOKEN_COUNT})
        with database.connect() as conn:
            db.add_file_usage(conn, Provider.CODEX, "a.jsonl", {self.KEY: at_the_limit})

        with (
            pytest.raises(ValueError, match="more than one stored row can hold") as raised,
            database.connect() as conn,
        ):
            # Why an output token is always added: usage with no tokens and no requests is not stored at all.
            db.add_file_usage(
                conn, Provider.CODEX, "a.jsonl", {self.KEY: TokenCounts(**{"output_tokens": 1, field: 1})}
            )

        assert isinstance(raised.value.__cause__, sqlite3.IntegrityError)
        assert self.read(database) == [at_the_limit]

    def test_a_refused_row_takes_the_rows_written_before_it_in_the_same_transaction_with_it(
        self, database: Database
    ) -> None:
        other_hour = UsageKey(account_id=1, bucket_hour="2026-10-03T15", model="model-x")
        with database.connect() as conn:
            db.add_file_usage(conn, Provider.CODEX, "a.jsonl", {other_hour: TokenCounts(input_tokens=MAX_TOKEN_COUNT)})

        with pytest.raises(ValueError, match="more than one stored row can hold"), database.connect() as conn:
            db.add_file_usage(
                conn,
                Provider.CODEX,
                "a.jsonl",
                {self.KEY: TokenCounts(input_tokens=5), other_hour: TokenCounts(input_tokens=1)},
            )

        assert self.read(database) == [TokenCounts(input_tokens=MAX_TOKEN_COUNT)]

    def test_many_files_at_the_limit_for_one_hour_add_up_exactly_past_sixty_four_bits(self, database: Database) -> None:
        """SQLite's own SUM raises "integer overflow" on these rows; a report must still be able to show them."""
        at_the_limit = TokenCounts(**dict.fromkeys(self.COUNT_FIELDS, MAX_TOKEN_COUNT))
        files = 12
        with database.connect() as conn:
            for index in range(files):
                db.add_file_usage(conn, Provider.CODEX, f"file-{index}.jsonl", {self.KEY: at_the_limit})
            with pytest.raises(sqlite3.OperationalError, match="integer overflow"):
                conn.execute("SELECT SUM(input_tokens) FROM token_usage").fetchone()

        (total,) = self.read(database)

        assert total == TokenCounts(**dict.fromkeys(self.COUNT_FIELDS, files * MAX_TOKEN_COUNT))
        assert total.input_tokens > 2**63

    def test_rows_for_the_same_hour_and_model_from_different_accounts_and_providers_stay_separate(
        self, database: Database
    ) -> None:
        other_account = UsageKey(account_id=2, bucket_hour=self.KEY.bucket_hour, model=self.KEY.model)
        with database.connect() as conn:
            db.add_file_usage(conn, Provider.GROK, "a.jsonl", {self.KEY: TokenCounts(input_tokens=1)})
            db.add_file_usage(conn, Provider.GROK, "b.jsonl", {self.KEY: TokenCounts(input_tokens=2)})
            db.add_file_usage(conn, Provider.GROK, "a.jsonl", {other_account: TokenCounts(input_tokens=4)})
            db.add_file_usage(conn, Provider.CODEX, "a.jsonl", {self.KEY: TokenCounts(input_tokens=8)})
            rows = db.token_usage(conn, "2026-10-03T00", "2026-10-04T00")

        assert [(row.provider, row.account_id, row.counts.input_tokens) for row in rows] == [
            (Provider.CODEX, 1, 8),
            (Provider.GROK, 1, 3),
            (Provider.GROK, 2, 4),
        ]

    def test_replacing_a_file_s_usage_is_held_to_the_same_limit(self, database: Database) -> None:
        with database.connect() as conn:
            db.add_file_usage(conn, Provider.GROK, "usage.json", {self.KEY: TokenCounts(input_tokens=5)})

        with pytest.raises(ValueError, match="outside what one stored row can hold"), database.connect() as conn:
            db.replace_file_usage(
                conn, Provider.GROK, "usage.json", {self.KEY: TokenCounts(input_tokens=MAX_TOKEN_COUNT + 1)}
            )

        assert self.read(database) == [TokenCounts(input_tokens=5)]

    @pytest.mark.parametrize("value", [-1, MAX_TOKEN_COUNT + 1])
    def test_the_schema_itself_refuses_a_count_outside_the_limit_however_it_is_written(
        self, database: Database, value: int
    ) -> None:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"), database.connect() as conn:
            conn.execute(
                "INSERT INTO token_usage (provider, file_key, account_id, bucket_hour, model, input_tokens, "
                "cache_read_tokens, cache_write_tokens, output_tokens, reasoning_tokens, requests) "
                "VALUES ('claude', 'a.jsonl', 1, '2026-10-03T14', 'model-x', ?, 0, 0, 0, 0, 0)",
                (value,),
            )

    def test_two_values_at_the_limit_can_be_added_inside_a_signed_64_bit_integer(self) -> None:
        """So SQLite computes the sum as an integer and the CHECK sees it, instead of a rounded REAL."""
        assert 2 * MAX_TOKEN_COUNT < 2**63
