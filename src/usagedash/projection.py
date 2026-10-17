"""Turn stored snapshots into what the dashboard shows: current state, reset roll-forward, pace, and advice.

Everything here is a pure function of its inputs so the rules that decide "switch accounts now" are testable
without a database or a clock.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from usagedash.db import AccountRow, SnapshotRow, SourceStateRow
from usagedash.models import FULL_PERCENT, Provider

MIN_STALE_AFTER = timedelta(minutes=15)
STALE_AFTER_POLL_INTERVALS = 3
PACE_DEFAULT_LOOKBACK = timedelta(hours=1)
PACE_MIN_LOOKBACK = timedelta(minutes=30)
PACE_MAX_LOOKBACK = timedelta(hours=6)
PACE_WINDOW_FRACTION = 5
PACE_MIN_SPAN = timedelta(minutes=10)
PACE_MAX_HORIZON = timedelta(days=30)


class WindowState(StrEnum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"


class Freshness(StrEnum):
    LIVE = "live"
    STALE = "stale"
    PROJECTED = "projected"


class AdviceAction(StrEnum):
    """What to do next. ``VERIFY`` names an account that looked usable when last read but is not being read now."""

    STAY = "stay"
    SWITCH = "switch"
    VERIFY = "verify"
    HOLD = "hold"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Thresholds:
    warn_percent: float
    critical_percent: float


@dataclass(frozen=True, slots=True)
class WindowView:
    """One window as displayed. ``effective_percent`` is ``used_percent`` after any reset that has since passed."""

    key: str
    label: str
    blocking: bool
    window_seconds: int | None
    used_percent: float
    effective_percent: float
    state: WindowState
    resets_at: datetime | None
    reset_passed: bool
    as_of: datetime
    freshness: Freshness
    pace_exhausts_at: datetime | None


@dataclass(frozen=True, slots=True)
class AccountView:
    """An account as displayed. Availability fields are None when no blocking window has ever been read.

    ``is_confirmed`` is True only when every blocking window comes from a current reading. Headroom on an
    unconfirmed account is what it was when last read (or a projected reset), not what it is now.
    """

    account: AccountRow
    is_active: bool
    is_live: bool
    windows: tuple[WindowView, ...]
    available_now: bool | None
    headroom_percent: float | None
    next_available_at: datetime | None
    is_confirmed: bool = False
    last_read_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Advice:
    """A recommendation. ``as_of`` is set when it rests on a reading that is no longer current."""

    action: AdviceAction
    summary: str
    active_account_id: int | None = None
    target_account_id: int | None = None
    next_available_at: datetime | None = None
    as_of: datetime | None = None


@dataclass(frozen=True, slots=True)
class ProviderView:
    provider: Provider
    accounts: tuple[AccountView, ...]
    advice: Advice


def classify(effective_percent: float, thresholds: Thresholds) -> WindowState:
    if effective_percent >= FULL_PERCENT:
        return WindowState.EXHAUSTED
    if effective_percent >= thresholds.critical_percent:
        return WindowState.CRITICAL
    if effective_percent >= thresholds.warn_percent:
        return WindowState.WARNING
    return WindowState.OK


def pace_exhausts_at(history: Sequence[SnapshotRow], latest: SnapshotRow) -> datetime | None:
    """Estimate when a window fills if usage continues at its recent rate.

    ``history`` holds the window's snapshots oldest first. Only snapshots from the current period count
    (see ``_current_period``). Returns None when usage is flat, the span is too short to trust, or the
    window would reset before filling. A reset time the reading was taken after is not a reset the
    window can still reach, so it bounds nothing: the estimate is then limited only by the horizon.
    """
    remaining = FULL_PERCENT - latest.used_percent
    if remaining <= 0:
        return None
    measurements = _current_period(history, latest)
    if not measurements:
        return None
    target = latest.confirmed_at - _pace_lookback(latest.window_seconds)
    in_effect_at_target = [percent for measured_at, percent in measurements if measured_at <= target]
    if in_effect_at_target:
        baseline_time, baseline_percent = target, in_effect_at_target[-1]
    else:
        baseline_time, baseline_percent = measurements[0]
    span = latest.confirmed_at - baseline_time
    gained = latest.used_percent - baseline_percent
    if span < PACE_MIN_SPAN or gained <= 0:
        return None
    # Why float seconds: a provider percentage is an arbitrary float, so a gain of 1e-9 would make the
    # projected time overflow a timedelta. The horizon is checked as a plain number first, and a datetime
    # is only built once the estimate is known to be within it.
    seconds_until_full = span.total_seconds() * (remaining / gained)
    horizon = PACE_MAX_HORIZON
    resets_at = _reset_still_ahead_of(latest)
    if resets_at is not None:
        horizon = min(horizon, resets_at - latest.confirmed_at)
    if seconds_until_full >= horizon.total_seconds():
        return None
    return latest.confirmed_at + timedelta(seconds=seconds_until_full)


def _current_period(history: Sequence[SnapshotRow], latest: SnapshotRow) -> list[tuple[datetime, float]]:
    """Return what was measured in the period ``latest`` is in: when, and the percentage used, oldest first.

    A period is known by the reset time its readings report. While that reset is still ahead of the
    latest reading, every snapshot reporting it was measured before it, when it was first observed.

    Once the latest reading was taken after the reset it reports, the period being filled began at
    that reset. A snapshot last confirmed before it measured the period that ended there, and is left
    out. A snapshot first observed before it and confirmed again after it is part of this period only
    from that confirmation: its earlier time is on the other side of the reset, and a rate measured
    from there would span it.
    """
    period = [row for row in history if row.resets_at == latest.resets_at]
    began = latest.resets_at
    if began is None or _reset_still_ahead_of(latest) is not None:
        return [(row.observed_at, row.used_percent) for row in period]
    return [
        (row.observed_at if row.observed_at >= began else row.confirmed_at, row.used_percent)
        for row in period
        if row.confirmed_at >= began
    ]


def _pace_lookback(window_seconds: int | None) -> timedelta:
    """Pick how far back to measure the usage rate: a fifth of the window, kept between the two bounds."""
    if window_seconds is None:
        return PACE_DEFAULT_LOOKBACK
    bounded_seconds = min(
        PACE_MAX_LOOKBACK.total_seconds(),
        max(PACE_MIN_LOOKBACK.total_seconds(), window_seconds / PACE_WINDOW_FRACTION),
    )
    return timedelta(seconds=bounded_seconds)


def _window_view(
    snapshot: SnapshotRow,
    history: Sequence[SnapshotRow],
    *,
    is_live: bool,
    now: datetime,
    thresholds: Thresholds,
    stale_after: timedelta,
) -> WindowView:
    resets_at = _reset_still_ahead_of(snapshot)
    reset_passed = resets_at is not None and resets_at <= now
    effective = 0.0 if reset_passed else snapshot.used_percent
    if reset_passed:
        freshness = Freshness.PROJECTED
    elif is_live and timedelta() <= now - snapshot.confirmed_at <= stale_after:
        # Why the age has a lower bound too: a reading dated after now (a clock set back, a timestamp from
        # a log) has a negative age, which would otherwise count as current for as long as it stays ahead.
        freshness = Freshness.LIVE
    else:
        freshness = Freshness.STALE
    return WindowView(
        key=snapshot.window_key,
        label=snapshot.label,
        blocking=snapshot.blocking,
        window_seconds=snapshot.window_seconds,
        used_percent=snapshot.used_percent,
        effective_percent=effective,
        state=classify(effective, thresholds),
        resets_at=resets_at,
        reset_passed=reset_passed,
        as_of=snapshot.confirmed_at,
        freshness=freshness,
        pace_exhausts_at=_forecast_still_ahead(pace_exhausts_at(history, snapshot), now)
        if freshness is Freshness.LIVE
        else None,
    )


def _forecast_still_ahead(forecast: datetime | None, now: datetime) -> datetime | None:
    """Return a pace forecast only while it is still to come.

    A reading counts as live for minutes, and usage fast enough can reach its forecast within them. A
    forecast of a time already past is not shown: the window was below full when last measured, and
    whether it has filled since is not known.
    """
    return forecast if forecast is not None and forecast > now else None


def _reset_still_ahead_of(snapshot: SnapshotRow) -> datetime | None:
    """Return the reset time a reading reports, if that reset had not already happened when it was read.

    A window is rolled forward to empty once its reset time passes, because the usage measured before
    the reset no longer counts. A reset time earlier than the reading turns that around: the usage was
    measured after it, so the reset cleared none of it, and the time says nothing about when the window
    resets next. Such a reading keeps its measured usage and has no known reset time. This is also what
    an unchanged reading confirmed again after its reset time is: a newer measurement.
    """
    if snapshot.resets_at is None or snapshot.resets_at < snapshot.confirmed_at:
        return None
    return snapshot.resets_at


def _account_view(
    account: AccountRow, windows: tuple[WindowView, ...], *, is_active: bool, is_live: bool
) -> AccountView:
    blocking = [window for window in windows if window.blocking]
    if not blocking:
        return AccountView(account, is_active, is_live, windows, None, None, None)
    exhausted = [window for window in blocking if window.state is WindowState.EXHAUSTED]
    next_available_at: datetime | None = None
    if exhausted and all(window.resets_at is not None for window in exhausted):
        next_available_at = max(window.resets_at for window in exhausted if window.resets_at is not None)
    return AccountView(
        account=account,
        is_active=is_active,
        is_live=is_live,
        windows=windows,
        available_now=not exhausted,
        headroom_percent=max(0.0, FULL_PERCENT - max(window.effective_percent for window in blocking)),
        next_available_at=next_available_at,
        is_confirmed=all(window.freshness is Freshness.LIVE for window in blocking),
        last_read_at=min(window.as_of for window in blocking),
    )


def _is_comfortable(view: AccountView, thresholds: Thresholds) -> bool:
    return bool(view.available_now) and all(
        window.effective_percent < thresholds.critical_percent for window in view.windows if window.blocking
    )


def _last_read_then_headroom(view: AccountView, now: datetime) -> tuple[float, float]:
    """Rank unconfirmed accounts by how recently they were read, then by the headroom they showed.

    A reading dated after ``now`` is stale. Its timestamp is not a later read, so it ranks after every
    reading that has already happened.
    """
    read_at = view.last_read_at
    if read_at is None or read_at > now:
        return (float("-inf"), view.headroom_percent or 0.0)
    return (read_at.timestamp(), view.headroom_percent or 0.0)


def _unconfirmed_claim(view: AccountView) -> str:
    """Say what is actually known about an account that is not being read now."""
    label = view.account.label
    if any(window.reset_passed for window in view.windows if window.blocking):
        return f"{label} has reset since it was last read, so it may have room."
    return f"{label} had {view.headroom_percent:.0f}% headroom when last read."


def advise(accounts: Sequence[AccountView], thresholds: Thresholds, *, now: datetime) -> Advice:
    """Decide whether to stay on the active account, switch to another, or wait for a reset.

    Only a current reading supports a definite statement. An account that is not being read now may have
    been used since, so the most it can earn is ``VERIFY``; and when the active account itself has no
    current reading, the answer is ``UNKNOWN`` rather than a headroom figure that may be days old. For
    the same reason ``HOLD`` names the first reset among accounts with a current reading only.
    ``now`` is the dashboard clock the views were built with. A reading dated after it is not a later read.
    """
    active = next((view for view in accounts if view.is_active), None)
    if active is None:
        return Advice(AdviceAction.UNKNOWN, "No account is logged in to the working login.")
    label = active.account.label
    if active.headroom_percent is None:
        return Advice(
            AdviceAction.UNKNOWN,
            f"No limit that blocks {label} has been read yet.",
            active_account_id=active.account.id,
        )
    if not active.is_confirmed:
        return Advice(
            AdviceAction.UNKNOWN,
            f"{label} has no current reading, so its usage now is unknown.",
            active_account_id=active.account.id,
            as_of=active.last_read_at,
        )
    if _is_comfortable(active, thresholds):
        return Advice(
            AdviceAction.STAY,
            f"{label} has {active.headroom_percent:.0f}% headroom.",
            active_account_id=active.account.id,
        )
    alternatives = [
        view
        for view in accounts
        if view is not active and view.headroom_percent is not None and _is_comfortable(view, thresholds)
    ]
    constraint = "is out of quota" if not active.available_now else f"has {active.headroom_percent:.0f}% headroom left"
    confirmed = [view for view in alternatives if view.is_confirmed]
    if confirmed:
        target = max(confirmed, key=lambda view: view.headroom_percent or 0.0)
        return Advice(
            AdviceAction.SWITCH,
            f"{label} {constraint}. Switch to {target.account.label} ({target.headroom_percent:.0f}% headroom).",
            active_account_id=active.account.id,
            target_account_id=target.account.id,
        )
    if alternatives:
        target = max(alternatives, key=lambda view: _last_read_then_headroom(view, now))
        return Advice(
            AdviceAction.VERIFY,
            f"{label} {constraint}. {_unconfirmed_claim(target)} Log in to it to confirm.",
            active_account_id=active.account.id,
            target_account_id=target.account.id,
            as_of=target.last_read_at,
        )
    waits = [view.next_available_at for view in accounts if view.is_confirmed and view.next_available_at is not None]
    return Advice(
        AdviceAction.HOLD,
        f"{label} {constraint} and no other account has comfortable headroom.",
        active_account_id=active.account.id,
        next_available_at=min(waits) if waits and not active.available_now else None,
    )


def build_provider_views(
    accounts: Sequence[AccountRow],
    latest: Sequence[SnapshotRow],
    history: Sequence[SnapshotRow],
    source_states: Sequence[SourceStateRow],
    *,
    now: datetime,
    thresholds: Thresholds,
    poll_interval_seconds: int,
) -> list[ProviderView]:
    """Assemble the per-provider view for every provider, including providers with no accounts yet."""
    stale_after = max(MIN_STALE_AFTER, timedelta(seconds=poll_interval_seconds * STALE_AFTER_POLL_INTERVALS))
    live_ids = {state.current_account_id for state in source_states if state.current_account_id is not None}
    active_ids = _primary_account_ids(source_states)

    latest_by_account: dict[int, list[SnapshotRow]] = {}
    for snapshot in latest:
        latest_by_account.setdefault(snapshot.account_id, []).append(snapshot)
    history_by_window: dict[tuple[int, str], list[SnapshotRow]] = {}
    for snapshot in history:
        history_by_window.setdefault((snapshot.account_id, snapshot.window_key), []).append(snapshot)

    views: list[ProviderView] = []
    for provider in Provider:
        account_views = []
        for account in accounts:
            if account.provider is not provider:
                continue
            is_live = account.id in live_ids
            windows = tuple(
                _window_view(
                    snapshot,
                    history_by_window.get((account.id, snapshot.window_key), []),
                    is_live=is_live,
                    now=now,
                    thresholds=thresholds,
                    stale_after=stale_after,
                )
                for snapshot in sorted(latest_by_account.get(account.id, []), key=_display_order)
            )
            account_views.append(
                _account_view(account, windows, is_active=active_ids.get(provider) == account.id, is_live=is_live)
            )
        account_views.sort(key=lambda view: (not view.is_active, not view.is_live, view.account.id))
        views.append(ProviderView(provider, tuple(account_views), advise(account_views, thresholds, now=now)))
    return views


def _display_order(snapshot: SnapshotRow) -> tuple[bool, bool, int, str]:
    """Order windows shortest first, with informational pools and windows of unknown length last."""
    return (
        not snapshot.blocking,
        snapshot.window_seconds is None,
        snapshot.window_seconds or 0,
        snapshot.window_key,
    )


def _primary_account_ids(source_states: Sequence[SourceStateRow]) -> Mapping[Provider, int]:
    """Return, per provider, the account logged in at its working source.

    The working source is the one configured first (position 0). A standing source is a later
    position. When the working source has no stored state, no standing source takes its place:
    nobody is known to be logged in there, and a standing account must not be treated as the one
    in use.

    Two rows at position 0 are not a working source. Each source records its position in its own
    transaction, so a reorder can leave the previous working source there beside the new one.
    Choosing either would follow the order of the rows, so that provider stays unresolved.
    """
    primary: dict[Provider, SourceStateRow] = {}
    ambiguous: set[Provider] = set()
    for state in source_states:
        if state.position != 0:
            continue
        if state.provider in primary:
            ambiguous.add(state.provider)
        primary.setdefault(state.provider, state)
    return {
        provider: state.current_account_id
        for provider, state in primary.items()
        if provider not in ambiguous and state.current_account_id is not None
    }
