"""Provider-neutral domain types shared by sources, the store, and the API."""

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Literal

UNATTRIBUTED_ACCOUNT_ID = 0
FULL_PERCENT = 100.0
"""The usage percentage at which a window is exhausted."""
MAX_WINDOW_SECONDS = 10 * 365 * 24 * 60 * 60
"""Ten years. No rate-limit window is longer, and a length within this fits every later calculation."""
MAX_TOKEN_COUNT = 10**18
"""The most one stored usage row may hold in any count: one file's usage for one account, hour, and model.

It leaves room in a signed 64-bit integer for two such values to be added before the store checks
the result. It is a limit of the store, not of ``TokenCounts``, whose sums have no upper limit.
"""
"""Far above any real total, and below the largest integer the store can hold."""

ReadingSource = Literal["api", "log"]


class Provider(StrEnum):
    CLAUDE = "claude"
    CODEX = "codex"
    GROK = "grok"

    @property
    def display_name(self) -> str:
        return {"claude": "Claude", "codex": "Codex", "grok": "Grok"}[self.value]


@dataclass(frozen=True, slots=True)
class AccountIdentity:
    """Who a credential belongs to. ``external_id`` is the provider's stable key for the billed account.

    Raises:
        ValueError: if ``external_id`` is empty or only whitespace. The store keys an account on its
            provider and external id, so every login without a usable id would otherwise become one
            account and their usage would be mixed.
    """

    provider: Provider
    external_id: str
    default_label: str
    email: str | None = None
    plan: str | None = None

    def __post_init__(self) -> None:
        if self.external_id.strip() == "":
            raise ValueError("an account identity needs a non-blank external id")


def _is_percentage(value: float) -> bool:
    """Report whether ``value`` is a finite number that is not negative.

    An integer too large for a float is not finite for this purpose; testing it raises ``OverflowError``.
    """
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


@dataclass(frozen=True, slots=True)
class QuotaWindow:
    """One rate-limit window. ``blocking`` is False for informational pools that do not stop usage when full.

    ``used_percent`` is always stored as a float, whatever number it was built from, so a value compares
    equal to itself after a round trip through the store.

    Raises:
        ValueError: if ``used_percent`` is negative or not a finite number, or ``window_seconds`` is
            negative or longer than ``MAX_WINDOW_SECONDS``. Sources validate provider payloads first, so
            this only fires on a defect in a source.
    """

    key: str
    label: str
    used_percent: float
    resets_at: datetime | None
    window_seconds: int | None
    blocking: bool = True

    def __post_init__(self) -> None:
        if not _is_percentage(self.used_percent):
            raise ValueError(f"window {self.key!r} has an impossible used_percent")
        if self.window_seconds is not None and not 0 <= self.window_seconds <= MAX_WINDOW_SECONDS:
            raise ValueError(f"window {self.key!r} has an impossible length")
        # Why object.__setattr__: the dataclass is frozen, and this is the one place the value is set.
        object.__setattr__(self, "used_percent", float(self.used_percent))


@dataclass(frozen=True, slots=True)
class Fact:
    """A provider-specific detail shown beside the quota windows (credit balance, overage setting)."""

    label: str
    value: str
    resets_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class QuotaReading:
    """A point-in-time quota measurement. ``observed_at`` is when the provider measured it, not when we read it.

    Raises:
        ValueError: if two windows share a key, because the store keeps one current row per window key.
    """

    identity: AccountIdentity
    windows: tuple[QuotaWindow, ...]
    observed_at: datetime
    source: ReadingSource
    facts: tuple[Fact, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        keys = [window.key for window in self.windows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"reading has repeated window keys: {sorted(keys)}")


@dataclass(frozen=True, slots=True)
class TokenCounts:
    """Normalized token counts. ``input_tokens`` excludes cache reads and writes; ``reasoning_tokens`` is the
    portion of ``output_tokens`` spent on reasoning, reported for information and never added to the total.

    There is no upper limit: these are Python integers, so adding any number of them is exact and
    cannot fail. What the store can hold in one row is checked by the store when it writes.

    Raises:
        ValueError: if a count is not a whole number (an ``int``, and not a ``bool``), or is negative.
            A fractional count would be stored as a real number and come back as a fractional total.
    """

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0

    def __post_init__(self) -> None:
        counts = (
            self.input_tokens,
            self.cache_read_tokens,
            self.cache_write_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.requests,
        )
        if any(isinstance(count, bool) or not isinstance(count, int) for count in counts):
            raise ValueError("a token count is not a whole number")
        if any(count < 0 for count in counts):
            raise ValueError("a token count is negative")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens + self.output_tokens

    def plus(self, other: "TokenCounts") -> "TokenCounts":
        return TokenCounts(
            input_tokens=self.input_tokens + other.input_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            requests=self.requests + other.requests,
        )

    @property
    def is_empty(self) -> bool:
        """Report whether every count is zero, reasoning tokens included.

        Reasoning tokens are left out of ``total_tokens`` because they are part of the output tokens,
        but a record that revises only the reasoning count still carries something to store.
        """
        return self.total_tokens == 0 and self.reasoning_tokens == 0 and self.requests == 0


@dataclass(frozen=True, slots=True)
class UsageKey:
    """Aggregation key for token usage contributed by one session file."""

    account_id: int
    bucket_hour: str
    model: str
