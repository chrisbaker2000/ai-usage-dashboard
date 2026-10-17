"""The contract every account source implements, plus naming helpers the sources share."""

import re
from dataclasses import replace
from datetime import datetime
from typing import Protocol

from usagedash.errors import UsageDashError
from usagedash.models import AccountIdentity, Provider, QuotaReading, QuotaWindow

FIVE_HOURS_SECONDS = 5 * 60 * 60
DAY_SECONDS = 24 * 60 * 60
WEEK_SECONDS = 7 * DAY_SECONDS
FIRST_REPEAT_ORDINAL = 2

_KNOWN_DURATIONS: dict[int, tuple[str, str]] = {
    FIVE_HOURS_SECONDS: ("five_hour", "5-hour"),
    DAY_SECONDS: ("daily", "Daily"),
    WEEK_SECONDS: ("weekly", "Weekly"),
}


class NoQuotaDataError(UsageDashError):
    """The source is healthy but has nothing to report yet (for example, no billing entry logged so far)."""


def ensure_same_login(before: object, after: object, service: str) -> None:
    """Refuse a reading whose login changed while it was being read.

    A source that learns who is logged in from one place and what was used from another reads the two
    at different moments. If the login changed in between, the usage is the new login's and the
    identity the earlier one's. ``before`` and ``after`` are the login as the source read it before
    and after the usage was read: everything it reads there about whose login it is, not only the key
    the account is stored under. Two people can share that key, and the reading names one of them.

    Raises:
        NoQuotaDataError: if they differ. Nothing is wrong with the source; the next collection reads
            the login that is there now.
    """
    if before != after:
        raise NoQuotaDataError(
            f"the {service} login changed while it was being read; the next collection reads the new one"
        )


class QuotaSource(Protocol):
    """One place a CLI keeps a login. Reads credentials; never writes or refreshes them."""

    @property
    def source_id(self) -> str: ...

    @property
    def provider(self) -> Provider: ...

    def current_identity(self) -> AccountIdentity | None:
        """Return who is logged in according to local files only, or None when logged out.

        Raises:
            ProviderSchemaError: if a local file exists but cannot be understood.
        """
        ...

    def read_quota(self, now: datetime) -> QuotaReading:
        """Return the logged-in account's current quota windows.

        Raises:
            CredentialUnavailableError: if nobody is logged in.
            CredentialExpiredError: if the stored token is expired or rejected.
            ProviderRequestError: if the provider endpoint fails.
            ProviderSchemaError: if a payload or local file has an unexpected shape.
            NoQuotaDataError: if there is nothing to report yet.
        """
        ...


def slugify(text: str) -> str:
    """Lowercase ``text`` into a key fragment containing only ``a-z``, ``0-9``, and underscores."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "unnamed"


def humanize(identifier: str) -> str:
    """Turn ``snake_case`` or ``kebab-case`` into a capitalized phrase."""
    words = re.sub(r"[_\-]+", " ", identifier).strip()
    return words[:1].upper() + words[1:] if words else identifier


def duration_key_and_label(window_seconds: int | None) -> tuple[str, str]:
    """Name a window by its length so the same duration reads the same across providers.

    Only a positive length is named in days or hours: zero divides by both, and a zero-second window is
    not a zero-day one.
    """
    if window_seconds is None:
        return "window", "Limit"
    known = _KNOWN_DURATIONS.get(window_seconds)
    if known is not None:
        return known
    if window_seconds > 0 and window_seconds % DAY_SECONDS == 0:
        days = window_seconds // DAY_SECONDS
        return f"window_{days}d", f"{days}-day"
    if window_seconds > 0 and window_seconds % 3600 == 0:
        hours = window_seconds // 3600
        return f"window_{hours}h", f"{hours}-hour"
    return f"window_{window_seconds}s", f"{window_seconds}-second"


def with_unique_keys(windows: list[QuotaWindow]) -> tuple[QuotaWindow, ...]:
    """Suffix repeated keys so two limits the provider did not distinguish never overwrite each other.

    The first window to use a key keeps it. A repeat takes the lowest numbered suffix that no window in
    the reading uses, including a key the provider itself spelled with that suffix.
    """
    taken = {window.key for window in windows}
    claimed: set[str] = set()
    unique: list[QuotaWindow] = []
    for window in windows:
        if window.key not in claimed:
            claimed.add(window.key)
            unique.append(window)
            continue
        ordinal = FIRST_REPEAT_ORDINAL
        while f"{window.key}#{ordinal}" in taken:
            ordinal += 1
        key = f"{window.key}#{ordinal}"
        taken.add(key)
        claimed.add(key)
        unique.append(replace(window, key=key, label=f"{window.label} ({ordinal})"))
    return tuple(unique)
