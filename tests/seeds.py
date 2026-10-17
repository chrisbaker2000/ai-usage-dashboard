"""Seed data shared by the tests of the views built on the store."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from tests.factories import make_identity, make_reading, make_window
from usagedash import db
from usagedash.db import Database, MessageUsage
from usagedash.models import UNATTRIBUTED_ACCOUNT_ID, Provider, TokenCounts, UsageKey


def seed_accounts_and_usage(database: Database, now: datetime) -> dict[str, int]:
    """Two Claude accounts (one active with quota, one logged out) and token usage across three days."""
    active = make_identity(external_id="org-a", label="Personal")
    other = make_identity(external_id="org-b", label="Work team", plan="Team")
    with database.connect() as conn:
        active_id = db.upsert_account(conn, active, now)
        other_id = db.upsert_account(conn, other, now)
        db.record_reading(
            conn,
            active_id,
            make_reading(active, now, make_window("five_hour", 92, now + timedelta(hours=2), label="5-hour session")),
        )
        db.record_reading(
            conn,
            other_id,
            make_reading(other, now - timedelta(days=1), make_window("five_hour", 100, now - timedelta(hours=3))),
        )
        db.set_source_state(conn, "claude:home", Provider.CLAUDE, 0, active_id, now)
        db.record_status(conn, "claude:home", db.STATUS_KIND_QUOTA, now, None)
        db.record_status(conn, "grok:home", db.STATUS_KIND_QUOTA, now, ValueError("log reshaped"))
        usage = {
            UsageKey(active_id, "2026-10-03T13", "model-x"): TokenCounts(10, 100, 5, 20, 3, 2),
            UsageKey(active_id, "2026-10-03T03", "model-x"): TokenCounts(1, 0, 0, 1, 0, 1),
            UsageKey(other_id, "2026-10-02T12", "model-y"): TokenCounts(7, 0, 0, 3, 0, 1),
            UsageKey(UNATTRIBUTED_ACCOUNT_ID, "2026-09-30T12", "model-x"): TokenCounts(1000, 0, 0, 50, 0, 9),
        }
        db.add_message_usage(
            conn, [response for place, counts in usage.items() for response in _responses(place, counts)]
        )
    return {"active": active_id, "other": other_id}


def _responses(place: UsageKey, counts: TokenCounts) -> list[MessageUsage]:
    """The Claude responses that leave ``counts`` in the row for ``place``: one per request, the first with every token.

    Claude usage enters the store only as counted responses, so a seeded row is built from them.
    """
    at = datetime.strptime(place.bucket_hour, "%Y-%m-%dT%H").replace(minute=30, tzinfo=UTC)
    tokens = replace(counts, requests=0)
    return [
        MessageUsage(
            f"{place.account_id}/{place.bucket_hour}/{place.model}/{number}".encode(),
            at,
            place.account_id,
            place.model,
            tokens if number == 0 else TokenCounts(),
        )
        for number in range(counts.requests)
    ]
