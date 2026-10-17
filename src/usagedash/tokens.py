"""Aggregate stored hourly token usage into the buckets a chart draws, in the viewer's time zone."""

import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, get_args
from zoneinfo import ZoneInfo

from usagedash import db
from usagedash.db import AccountRow
from usagedash.models import UNATTRIBUTED_ACCOUNT_ID, Provider, TokenCounts
from usagedash.timeutil import hour_bucket, in_utc, parse_hour_bucket, to_iso

RangeKey = Literal["24h", "7d", "30d"]
BucketUnit = Literal["hour", "day"]

RANGE_KEYS: tuple[RangeKey, ...] = get_args(RangeKey)
_RANGE_DAYS: dict[RangeKey, int] = {"7d": 7, "30d": 30}
HOURS_IN_DAY_RANGE = 24
UNATTRIBUTED_LABEL = "Unattributed (before tracking)"
UNATTRIBUTED_COLOR_SLOT = 0

SeriesKey = tuple[Provider, int]
ModelKey = tuple[Provider, int, str]


@dataclass(frozen=True, slots=True)
class _Bucket:
    key: str
    start: datetime


def _hour_buckets(now_utc: datetime) -> list[_Bucket]:
    current_hour = now_utc.replace(minute=0, second=0, microsecond=0)
    starts = [current_hour - timedelta(hours=offset) for offset in range(HOURS_IN_DAY_RANGE - 1, -1, -1)]
    return [_Bucket(key=hour_bucket(start), start=start) for start in starts]


def _day_buckets(now_utc: datetime, days: int, zone: ZoneInfo) -> list[_Bucket]:
    today = now_utc.astimezone(zone).date()
    dates = [today - timedelta(days=offset) for offset in range(days - 1, -1, -1)]
    return [
        _Bucket(key=day.isoformat(), start=in_utc(datetime(day.year, day.month, day.day, tzinfo=zone))) for day in dates
    ]


def _counts_json(counts: TokenCounts) -> dict[str, int]:
    return {
        "input": counts.input_tokens,
        "cache_read": counts.cache_read_tokens,
        "cache_write": counts.cache_write_tokens,
        "output": counts.output_tokens,
        "reasoning": counts.reasoning_tokens,
        "requests": counts.requests,
        "total": counts.total_tokens,
    }


def _series_order(item: tuple[SeriesKey, list[TokenCounts]]) -> tuple[str, bool, int]:
    """Order series by provider, then real accounts by id, with the unattributed bucket last."""
    (provider, account_id), _ = item
    return provider.value, account_id == UNATTRIBUTED_ACCOUNT_ID, account_id


def _model_order(item: tuple[ModelKey, TokenCounts]) -> tuple[str, bool, int, str]:
    (provider, account_id, model), _ = item
    return provider.value, account_id == UNATTRIBUTED_ACCOUNT_ID, account_id, model


def build_token_report(
    conn: sqlite3.Connection, accounts: Sequence[AccountRow], range_key: RangeKey, zone: ZoneInfo, now: datetime
) -> dict[str, Any]:
    """Return per-account series over the requested range, plus per-model and per-provider totals.

    Hourly buckets are UTC-aligned hours; daily buckets are calendar days in ``zone``, so "today" matches
    the viewer's day and not the UTC day. Usage is stored per UTC hour, and an hour belongs to the day
    its first instant falls in. In a zone offset by a fraction of an hour, the hour that straddles local
    midnight is therefore counted whole in the day it starts in.

    Every figure in the report is a sum over the same stored hours: the ones that fall in a bucket shown.
    The series, the per-model totals, and the per-provider totals cannot disagree with each other.

    Raises:
        ValueError: if ``now`` is naive. It would otherwise be read in this machine's time zone.
    """
    # Why converted once, here: every bucket boundary below is arithmetic on this value, and arithmetic on
    # a time in a zone with clock changes is done on its wall-clock fields.
    now_utc = in_utc(now)
    unit: BucketUnit
    if range_key == "24h":
        unit, buckets = "hour", _hour_buckets(now_utc)
    else:
        unit, buckets = "day", _day_buckets(now_utc, _RANGE_DAYS[range_key], zone)
    index_by_key = {bucket.key: index for index, bucket in enumerate(buckets)}
    start_bucket = hour_bucket(buckets[0].start)
    end_bucket = hour_bucket(now_utc + timedelta(hours=1))

    # Why a default factory: a series' buckets are built once, when its first row is met, and not
    # once per row to be thrown away. The empty counts are immutable, so one object fills every bucket.
    empty_buckets = [TokenCounts()] * len(buckets)
    points: defaultdict[SeriesKey, list[TokenCounts]] = defaultdict(empty_buckets.copy)
    by_model: dict[ModelKey, TokenCounts] = {}
    for row in db.token_usage(conn, start_bucket, end_bucket):
        instant = parse_hour_bucket(row.bucket_hour)
        key = row.bucket_hour if unit == "hour" else instant.astimezone(zone).date().isoformat()
        index = index_by_key.get(key)
        if index is None:
            continue
        series = points[(row.provider, row.account_id)]
        series[index] = series[index].plus(row.counts)
        model_key = (row.provider, row.account_id, row.model)
        by_model[model_key] = by_model.get(model_key, TokenCounts()).plus(row.counts)

    account_by_id = {account.id: account for account in accounts}
    provider_totals: dict[Provider, TokenCounts] = {provider: TokenCounts() for provider in Provider}
    series_json = []
    for (provider, account_id), series in sorted(points.items(), key=_series_order):
        total = TokenCounts()
        for counts in series:
            total = total.plus(counts)
        provider_totals[provider] = provider_totals[provider].plus(total)
        account = account_by_id.get(account_id)
        series_json.append(
            {
                "provider": provider.value,
                "account_id": account_id,
                "label": account.label if account else UNATTRIBUTED_LABEL,
                "color_slot": account.color_slot if account else UNATTRIBUTED_COLOR_SLOT,
                "attributed": account_id != UNATTRIBUTED_ACCOUNT_ID,
                "points": [_counts_json(counts) for counts in series],
                "totals": _counts_json(total),
            }
        )

    models_json = [
        {"provider": provider.value, "account_id": account_id, "model": model, "totals": _counts_json(counts)}
        for (provider, account_id, model), counts in sorted(by_model.items(), key=_model_order)
    ]

    return {
        "range": range_key,
        "bucket": unit,
        "timezone": zone.key,
        "buckets": [{"key": bucket.key, "start": to_iso(bucket.start)} for bucket in buckets],
        "series": series_json,
        "models": models_json,
        "provider_totals": [
            {"provider": provider.value, "display_name": provider.display_name, "totals": _counts_json(total)}
            for provider, total in provider_totals.items()
        ],
    }
