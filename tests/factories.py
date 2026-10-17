"""Factory functions for synthetic domain objects. Every persona, id, and value is made up."""

from datetime import datetime

from usagedash.models import AccountIdentity, Provider, QuotaReading, QuotaWindow

PERSONA_EMAIL = "user-a@example.test"
OTHER_PERSONA_EMAIL = "user-b@example.test"
FIVE_HOURS = 5 * 60 * 60
WEEK = 7 * 24 * 60 * 60


def make_identity(
    provider: Provider = Provider.CLAUDE,
    external_id: str = "org-a",
    label: str = "Account A",
    email: str | None = PERSONA_EMAIL,
    plan: str | None = "Max 20x",
) -> AccountIdentity:
    return AccountIdentity(provider=provider, external_id=external_id, default_label=label, email=email, plan=plan)


def make_window(
    key: str = "five_hour",
    used_percent: float = 10.0,
    resets_at: datetime | None = None,
    window_seconds: int | None = FIVE_HOURS,
    label: str | None = None,
    blocking: bool = True,
) -> QuotaWindow:
    return QuotaWindow(
        key=key,
        label=label or key,
        used_percent=used_percent,
        resets_at=resets_at,
        window_seconds=window_seconds,
        blocking=blocking,
    )


def make_reading(
    identity: AccountIdentity,
    observed_at: datetime,
    *windows: QuotaWindow,
    source: str = "api",
) -> QuotaReading:
    return QuotaReading(identity=identity, windows=tuple(windows), observed_at=observed_at, source=source)
