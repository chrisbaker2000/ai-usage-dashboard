from datetime import datetime, timedelta

import pytest

from tests.factories import FIVE_HOURS, WEEK
from usagedash.db import AccountRow, SnapshotRow, SourceStateRow
from usagedash.models import Provider
from usagedash.projection import (
    AccountView,
    Advice,
    AdviceAction,
    Freshness,
    Thresholds,
    WindowState,
    advise,
    build_provider_views,
    classify,
    pace_exhausts_at,
)

THRESHOLDS = Thresholds(warn_percent=75, critical_percent=90)
POLL_SECONDS = 300


def account(account_id: int, provider: Provider = Provider.CLAUDE, label: str | None = None) -> AccountRow:
    return AccountRow(
        id=account_id,
        provider=provider,
        external_id=f"ext-{account_id}",
        label=label or f"Account {account_id}",
        label_is_custom=False,
        email=None,
        plan=None,
        color_slot=account_id,
        first_seen_at=datetime(2026, 9, 1).astimezone(),
        last_live_at=None,
        facts=(),
    )


def snapshot(
    account_id: int,
    now: datetime,
    used: float,
    *,
    key: str = "five_hour",
    resets_in: timedelta | None = timedelta(hours=2),
    observed_ago: timedelta = timedelta(),
    confirmed_ago: timedelta | None = None,
    window_seconds: int | None = FIVE_HOURS,
    blocking: bool = True,
    resets_at: datetime | None = None,
) -> SnapshotRow:
    return SnapshotRow(
        account_id=account_id,
        window_key=key,
        label=key,
        window_seconds=window_seconds,
        used_percent=used,
        resets_at=resets_at if resets_at is not None else (None if resets_in is None else now + resets_in),
        blocking=blocking,
        source="api",
        observed_at=now - observed_ago,
        confirmed_at=now - (observed_ago if confirmed_ago is None else confirmed_ago),
    )


def source_state(account_id: int | None, provider: Provider = Provider.CLAUDE, position: int = 0) -> SourceStateRow:
    return SourceStateRow(
        source_id=f"{provider.value}:{position}",
        provider=provider,
        position=position,
        current_account_id=account_id,
        checked_at=datetime(2026, 10, 3).astimezone(),
    )


def views_for(
    accounts: list[AccountRow],
    latest: list[SnapshotRow],
    states: list[SourceStateRow],
    now: datetime,
    history: list[SnapshotRow] | None = None,
) -> dict[int, AccountView]:
    views = build_provider_views(
        accounts,
        latest,
        history if history is not None else latest,
        states,
        now=now,
        thresholds=THRESHOLDS,
        poll_interval_seconds=POLL_SECONDS,
    )
    return {view.account.id: view for provider in views for view in provider.accounts}


class TestWindowClassification:
    @pytest.mark.parametrize(
        ("percent", "expected"),
        [
            (0, WindowState.OK),
            (74.9, WindowState.OK),
            (75, WindowState.WARNING),
            (89.9, WindowState.WARNING),
            (90, WindowState.CRITICAL),
            (99.9, WindowState.CRITICAL),
            (100, WindowState.EXHAUSTED),
            (140, WindowState.EXHAUSTED),
        ],
    )
    def test_each_threshold_belongs_to_the_more_severe_state(self, percent: float, expected: WindowState) -> None:
        assert classify(percent, THRESHOLDS) is expected


class TestResetRollForward:
    def test_a_window_whose_reset_time_has_passed_is_shown_as_empty_and_marked_projected(self, now: datetime) -> None:
        stale = snapshot(1, now, 100, resets_in=timedelta(minutes=-1), observed_ago=timedelta(days=2))

        window = views_for([account(1)], [stale], [], now)[1].windows[0]

        assert (window.used_percent, window.effective_percent) == (100, 0)
        assert window.reset_passed
        assert window.freshness is Freshness.PROJECTED
        assert window.state is WindowState.OK

    def test_a_window_resetting_exactly_now_counts_as_reset(self, now: datetime) -> None:
        window = views_for([account(1)], [snapshot(1, now, 80, resets_in=timedelta())], [], now)[1].windows[0]
        assert window.reset_passed

    def test_a_window_with_time_left_keeps_its_measured_value(self, now: datetime) -> None:
        window = views_for([account(1)], [snapshot(1, now, 80, resets_in=timedelta(seconds=1))], [], now)[1].windows[0]
        assert (window.effective_percent, window.reset_passed) == (80, False)

    def test_a_reading_taken_after_the_reset_time_it_reports_keeps_its_measured_usage(self, now: datetime) -> None:
        """The usage was measured after that reset, so the reset cleared none of it."""
        reported_reset = now - timedelta(minutes=1)
        measured_after_it = snapshot(1, now, 100, resets_at=reported_reset)

        view = views_for([account(1)], [measured_after_it], [source_state(1)], now)[1]
        window = view.windows[0]

        assert (window.effective_percent, window.state, window.reset_passed) == (100, WindowState.EXHAUSTED, False)
        assert window.resets_at is None
        assert window.freshness is Freshness.LIVE
        assert (view.available_now, view.next_available_at) == (False, None)

    def test_an_unchanged_reading_confirmed_again_after_its_reset_time_is_a_newer_measurement(
        self, now: datetime
    ) -> None:
        reconfirmed = snapshot(
            1,
            now,
            100,
            resets_at=now - timedelta(minutes=30),
            observed_ago=timedelta(hours=2),
            confirmed_ago=timedelta(),
        )

        window = views_for([account(1)], [reconfirmed], [source_state(1)], now)[1].windows[0]

        assert (window.effective_percent, window.reset_passed, window.resets_at) == (100, False, None)

    def test_a_reading_taken_at_the_very_moment_of_its_reset_is_still_rolled_forward(self, now: datetime) -> None:
        at_the_reset = snapshot(1, now, 100, resets_at=now - timedelta(minutes=5), observed_ago=timedelta(minutes=5))

        window = views_for([account(1)], [at_the_reset], [source_state(1)], now)[1].windows[0]

        assert (window.effective_percent, window.reset_passed) == (0, True)

    def test_a_window_without_a_reset_time_is_never_rolled_forward(self, now: datetime) -> None:
        old = snapshot(1, now, 95, resets_in=None, observed_ago=timedelta(days=30))
        window = views_for([account(1)], [old], [], now)[1].windows[0]
        assert (window.effective_percent, window.state) == (95, WindowState.CRITICAL)


class TestFreshness:
    def test_a_recent_reading_of_a_logged_in_account_is_live(self, now: datetime) -> None:
        view = views_for(
            [account(1)], [snapshot(1, now, 10, observed_ago=timedelta(minutes=15))], [source_state(1)], now
        )[1]
        assert view.windows[0].freshness is Freshness.LIVE
        assert (view.is_live, view.is_active) == (True, True)

    def test_a_reading_older_than_the_staleness_allowance_is_stale_even_when_logged_in(self, now: datetime) -> None:
        old = snapshot(1, now, 10, observed_ago=timedelta(minutes=15, seconds=1))
        assert views_for([account(1)], [old], [source_state(1)], now)[1].windows[0].freshness is Freshness.STALE

    @pytest.mark.parametrize("ahead", [timedelta(seconds=1), timedelta(minutes=10), timedelta(days=30)])
    def test_a_reading_dated_after_now_is_not_current_however_far_ahead_it_is(
        self, now: datetime, ahead: timedelta
    ) -> None:
        """Its age is negative. Without a lower bound it would count as current until now caught up with it."""
        from_the_future = snapshot(1, now, 10, observed_ago=-ahead)

        view = views_for([account(1)], [from_the_future], [source_state(1)], now)[1]

        assert view.windows[0].freshness is Freshness.STALE
        assert view.is_confirmed is False

    def test_a_reading_taken_exactly_now_is_live(self, now: datetime) -> None:
        view = views_for([account(1)], [snapshot(1, now, 10)], [source_state(1)], now)[1]
        assert view.windows[0].freshness is Freshness.LIVE

    def test_a_logged_out_account_is_stale_however_recent_its_reading(self, now: datetime) -> None:
        view = views_for([account(1)], [snapshot(1, now, 10)], [source_state(None)], now)[1]
        assert view.windows[0].freshness is Freshness.STALE
        assert (view.is_live, view.is_active) == (False, False)

    def test_an_account_logged_in_only_at_a_standing_source_is_live_but_not_active(self, now: datetime) -> None:
        states = [source_state(1, position=0), source_state(2, position=1)]
        views = views_for([account(1), account(2)], [snapshot(1, now, 10), snapshot(2, now, 10)], states, now)

        assert (views[1].is_active, views[1].is_live) == (True, True)
        assert (views[2].is_active, views[2].is_live) == (False, True)
        assert views[2].windows[0].freshness is Freshness.LIVE


class TestPace:
    def test_estimates_when_the_window_fills_at_the_recent_rate(self, now: datetime) -> None:
        reset = now + timedelta(hours=4)
        history = [
            snapshot(1, now, 20, resets_at=reset, observed_ago=timedelta(hours=1)),
            snapshot(1, now, 60, resets_at=reset),
        ]

        assert pace_exhausts_at(history, history[-1]) == now + timedelta(hours=1)

    def test_measures_from_the_value_in_effect_one_lookback_ago(self, now: datetime) -> None:
        reset = now + timedelta(hours=4)
        history = [
            snapshot(1, now, 10, resets_at=reset, observed_ago=timedelta(hours=3)),
            snapshot(1, now, 50, resets_at=reset, observed_ago=timedelta(minutes=30)),
            snapshot(1, now, 60, resets_at=reset),
        ]

        assert pace_exhausts_at(history, history[-1]) == now + timedelta(minutes=48)

    def test_no_estimate_when_the_window_would_reset_first(self, now: datetime) -> None:
        reset = now + timedelta(minutes=59)
        history = [
            snapshot(1, now, 20, resets_at=reset, observed_ago=timedelta(hours=1)),
            snapshot(1, now, 60, resets_at=reset),
        ]
        assert pace_exhausts_at(history, history[-1]) is None

    def test_no_estimate_when_usage_is_flat(self, now: datetime) -> None:
        reset = now + timedelta(hours=4)
        only = snapshot(1, now, 40, resets_at=reset, observed_ago=timedelta(hours=2), confirmed_ago=timedelta())
        assert pace_exhausts_at([only], only) is None

    def test_no_estimate_from_a_span_too_short_to_trust(self, now: datetime) -> None:
        reset = now + timedelta(hours=4)
        history = [
            snapshot(1, now, 20, resets_at=reset, observed_ago=timedelta(minutes=9)),
            snapshot(1, now, 60, resets_at=reset),
        ]
        assert pace_exhausts_at(history, history[-1]) is None

    def test_usage_from_the_previous_period_is_not_a_baseline(self, now: datetime) -> None:
        reset = now + timedelta(hours=4)
        history = [
            snapshot(1, now, 95, resets_at=now - timedelta(minutes=30), observed_ago=timedelta(hours=1)),
            snapshot(1, now, 5, resets_at=reset),
        ]
        assert pace_exhausts_at(history, history[-1]) is None

    def test_a_reading_taken_after_the_reset_it_reports_is_paced_from_what_was_measured_since(
        self, now: datetime
    ) -> None:
        """That reset is not one the window can still reach, so it does not bound the estimate."""
        reported_reset = now - timedelta(hours=2)
        history = [
            snapshot(1, now, 20, resets_at=reported_reset, observed_ago=timedelta(hours=1)),
            snapshot(1, now, 60, resets_at=reported_reset),
        ]

        assert pace_exhausts_at(history, history[-1]) == now + timedelta(hours=1)

        window = views_for([account(1)], [history[-1]], [source_state(1)], now, history)[1].windows[0]
        assert (window.resets_at, window.pace_exhausts_at) == (None, now + timedelta(hours=1))

    @pytest.mark.parametrize(
        ("read_ago", "shown"),
        [
            pytest.param(timedelta(minutes=10), None, id="forecast already passed"),
            pytest.param(timedelta(seconds=400), None, id="forecast is exactly now"),
            pytest.param(timedelta(seconds=399), timedelta(seconds=1), id="forecast a second ahead"),
        ],
    )
    def test_a_live_reading_s_forecast_is_shown_only_while_it_is_still_to_come(
        self, now: datetime, read_ago: timedelta, shown: timedelta | None
    ) -> None:
        """A reading stays live for minutes, and usage this fast reaches its forecast within them.

        A five-hour window measures its rate over the last hour: from 0% to 90%, so the last 10% takes 400 s.
        """
        reset = now + timedelta(hours=4)
        history = [
            snapshot(1, now, 0, resets_at=reset, observed_ago=read_ago + timedelta(hours=1)),
            snapshot(1, now, 90, resets_at=reset, observed_ago=read_ago),
        ]
        assert pace_exhausts_at(history, history[-1]) == now - read_ago + timedelta(seconds=400)

        window = views_for([account(1)], [history[-1]], [source_state(1)], now, history)[1].windows[0]

        assert window.freshness is Freshness.LIVE
        assert window.pace_exhausts_at == (None if shown is None else now + shown)

    def test_a_reading_taken_after_the_reset_it_reports_is_still_limited_to_thirty_days(self, now: datetime) -> None:
        reported_reset = now - timedelta(hours=2)
        history = [
            snapshot(1, now, 20.0, resets_at=reported_reset, window_seconds=None, observed_ago=timedelta(hours=1)),
            snapshot(1, now, 20.1, resets_at=reported_reset, window_seconds=None),
        ]

        assert pace_exhausts_at(history, history[-1]) is None

    def test_usage_measured_before_a_reset_that_has_since_passed_is_not_a_baseline(self, now: datetime) -> None:
        """All three report the same reset. The first was measured before it, in the period it ended."""
        reported_reset = now - timedelta(hours=2)
        history = [
            snapshot(1, now, 90, resets_at=reported_reset, window_seconds=WEEK, observed_ago=timedelta(hours=3)),
            snapshot(1, now, 20, resets_at=reported_reset, window_seconds=WEEK, observed_ago=timedelta(hours=1)),
            snapshot(1, now, 60, resets_at=reported_reset, window_seconds=WEEK),
        ]

        assert pace_exhausts_at(history, history[-1]) == now + timedelta(hours=1)

    def test_a_snapshot_confirmed_again_after_the_reset_was_measured_in_this_period(self, now: datetime) -> None:
        reported_reset = now - timedelta(hours=2)
        history = [
            snapshot(
                1,
                now,
                40,
                resets_at=reported_reset,
                observed_ago=timedelta(hours=3),
                confirmed_ago=timedelta(hours=1),
            ),
            snapshot(1, now, 60, resets_at=reported_reset),
        ]

        assert pace_exhausts_at(history, history[-1]) == now + timedelta(hours=2)

    def test_a_snapshot_that_crosses_the_reset_is_measured_from_its_confirmation_after_it(self, now: datetime) -> None:
        """Its first observation is on the other side of the reset, and a rate measured from there would span it."""
        reported_reset = now - timedelta(hours=2)
        history = [
            snapshot(
                1,
                now,
                40,
                resets_at=reported_reset,
                window_seconds=WEEK,
                observed_ago=timedelta(hours=3),
                confirmed_ago=timedelta(hours=1),
            ),
            snapshot(1, now, 60, resets_at=reported_reset, window_seconds=WEEK),
        ]

        assert pace_exhausts_at(history, history[-1]) == now + timedelta(hours=2)

    @pytest.mark.parametrize(
        ("confirmed_ago", "expected_in"),
        [
            pytest.param(timedelta(minutes=1), None, id="one minute of measurement since the reset"),
            pytest.param(timedelta(minutes=9, seconds=59), None, id="just under the shortest span trusted"),
            pytest.param(timedelta(minutes=10), timedelta(minutes=20), id="exactly the shortest span trusted"),
        ],
    )
    def test_the_span_too_short_to_trust_is_counted_from_that_confirmation_too(
        self, now: datetime, confirmed_ago: timedelta, expected_in: timedelta | None
    ) -> None:
        reported_reset = now - timedelta(hours=2)
        history = [
            snapshot(
                1,
                now,
                40,
                resets_at=reported_reset,
                window_seconds=WEEK,
                observed_ago=timedelta(hours=3),
                confirmed_ago=confirmed_ago,
            ),
            snapshot(1, now, 60, resets_at=reported_reset, window_seconds=WEEK),
        ]

        assert pace_exhausts_at(history, history[-1]) == (None if expected_in is None else now + expected_in)

    def test_a_snapshot_that_crosses_the_reset_still_holds_until_the_next_one_is_observed(self, now: datetime) -> None:
        """Confirmed after the reset and before the lookback point, it is the level in effect at that point."""
        reported_reset = now - timedelta(hours=3)
        history = [
            snapshot(
                1,
                now,
                40,
                resets_at=reported_reset,
                observed_ago=timedelta(hours=4),
                confirmed_ago=timedelta(hours=2),
            ),
            snapshot(1, now, 50, resets_at=reported_reset, observed_ago=timedelta(minutes=30)),
            snapshot(1, now, 60, resets_at=reported_reset),
        ]

        assert pace_exhausts_at(history, history[-1]) == now + timedelta(hours=2)

    def test_a_snapshot_confirmed_at_the_very_moment_of_the_reset_belongs_to_this_period(self, now: datetime) -> None:
        reported_reset = now - timedelta(hours=1)
        history = [
            snapshot(1, now, 20, resets_at=reported_reset, observed_ago=timedelta(hours=1)),
            snapshot(1, now, 60, resets_at=reported_reset),
        ]

        assert pace_exhausts_at(history, history[-1]) == now + timedelta(hours=1)

    def test_a_reading_taken_at_the_very_moment_of_its_reset_has_no_time_left_to_fill(self, now: datetime) -> None:
        history = [
            snapshot(1, now, 20, resets_at=now, observed_ago=timedelta(hours=1)),
            snapshot(1, now, 60, resets_at=now),
        ]

        assert pace_exhausts_at(history, history[-1]) is None

    def test_a_full_window_has_no_pace(self, now: datetime) -> None:
        reset = now + timedelta(hours=4)
        history = [
            snapshot(1, now, 50, resets_at=reset, observed_ago=timedelta(hours=1)),
            snapshot(1, now, 100, resets_at=reset),
        ]
        assert pace_exhausts_at(history, history[-1]) is None

    def test_a_long_window_looks_back_at_most_six_hours(self, now: datetime) -> None:
        reset = now + timedelta(days=5)
        history = [
            snapshot(1, now, 0, resets_at=reset, window_seconds=WEEK, observed_ago=timedelta(days=2)),
            snapshot(1, now, 40, resets_at=reset, window_seconds=WEEK, observed_ago=timedelta(hours=7)),
            snapshot(1, now, 52, resets_at=reset, window_seconds=WEEK),
        ]

        assert pace_exhausts_at(history, history[-1]) == now + timedelta(hours=24)

    @pytest.mark.parametrize("gain", [1e-3, 1e-6, 1e-9, 4e-15])
    @pytest.mark.parametrize("has_reset", [True, False])
    def test_a_vanishingly_small_gain_yields_no_estimate_instead_of_overflowing(
        self, now: datetime, gain: float, has_reset: bool
    ) -> None:
        resets_in = timedelta(days=5) if has_reset else None
        history = [
            snapshot(1, now, 30.0, resets_in=resets_in, window_seconds=WEEK, observed_ago=timedelta(hours=7)),
            snapshot(1, now, 30.0 + gain, resets_in=resets_in, window_seconds=WEEK),
        ]
        assert pace_exhausts_at(history, history[-1]) is None

    def test_a_window_without_a_reset_time_is_estimated_only_within_thirty_days(self, now: datetime) -> None:
        def history(gain: float) -> list[SnapshotRow]:
            return [
                snapshot(1, now, 0.0, resets_in=None, window_seconds=None, observed_ago=timedelta(hours=1)),
                snapshot(1, now, gain, resets_in=None, window_seconds=None),
            ]

        gain_reaching_full_in_exactly_thirty_days = 100 / (30 * 24 + 1)
        just_inside = history(gain_reaching_full_in_exactly_thirty_days + 0.0001)
        just_outside = history(gain_reaching_full_in_exactly_thirty_days - 0.0001)

        estimate = pace_exhausts_at(just_inside, just_inside[-1])
        assert estimate is not None
        assert timedelta(days=29, hours=23) < estimate - now < timedelta(days=30)
        assert pace_exhausts_at(just_outside, just_outside[-1]) is None

    def test_an_absurd_window_length_from_the_provider_does_not_overflow_the_lookback(self, now: datetime) -> None:
        reset = now + timedelta(hours=4)
        history = [
            snapshot(1, now, 20, resets_at=reset, window_seconds=10**18, observed_ago=timedelta(hours=6)),
            snapshot(1, now, 60, resets_at=reset, window_seconds=10**18),
        ]
        assert pace_exhausts_at(history, history[-1]) is None

    def test_pace_is_only_shown_for_live_windows(self, now: datetime) -> None:
        reset = now + timedelta(hours=4)
        history = [
            snapshot(1, now, 20, resets_at=reset, observed_ago=timedelta(hours=1)),
            snapshot(1, now, 60, resets_at=reset),
        ]

        live = views_for([account(1)], [history[-1]], [source_state(1)], now, history)[1]
        logged_out = views_for([account(1)], [history[-1]], [], now, history)[1]

        assert live.windows[0].pace_exhausts_at == now + timedelta(hours=1)
        assert logged_out.windows[0].pace_exhausts_at is None


class TestAccountAvailability:
    def test_headroom_is_set_by_the_fullest_blocking_window(self, now: datetime) -> None:
        latest = [snapshot(1, now, 30, key="five_hour"), snapshot(1, now, 82, key="weekly")]
        view = views_for([account(1)], latest, [], now)[1]
        assert (view.available_now, view.headroom_percent, view.next_available_at) == (True, 18, None)

    def test_an_exhausted_window_makes_the_account_unavailable_until_it_resets(self, now: datetime) -> None:
        latest = [
            snapshot(1, now, 100, key="five_hour", resets_in=timedelta(hours=1)),
            snapshot(1, now, 100, key="weekly", resets_in=timedelta(days=2)),
            snapshot(1, now, 40, key="other", resets_in=timedelta(days=9)),
        ]
        view = views_for([account(1)], latest, [], now)[1]
        assert (view.available_now, view.headroom_percent) == (False, 0)
        assert view.next_available_at == now + timedelta(days=2)

    def test_an_informational_window_never_blocks(self, now: datetime) -> None:
        latest = [snapshot(1, now, 20), snapshot(1, now, 100, key="credit_pool", blocking=False)]
        view = views_for([account(1)], latest, [], now)[1]
        assert (view.available_now, view.headroom_percent) == (True, 80)

    def test_a_full_window_with_no_reset_time_makes_the_account_unavailable_with_no_known_return(
        self, now: datetime
    ) -> None:
        """A limit the provider reports as reached without naming a window arrives as such a window."""
        latest = [
            snapshot(1, now, 38, key="weekly", resets_in=timedelta(days=6)),
            snapshot(1, now, 100, key="reported_limit", resets_in=None, window_seconds=None),
        ]
        view = views_for([account(1)], latest, [source_state(1)], now)[1]
        assert (view.available_now, view.headroom_percent, view.next_available_at) == (False, 0, None)

    def test_an_account_with_no_reading_has_unknown_availability(self, now: datetime) -> None:
        view = views_for([account(1)], [], [source_state(1)], now)[1]
        assert (view.available_now, view.headroom_percent, view.windows) == (None, None, ())


class TestAdvice:
    def advice_for(
        self,
        latest: list[SnapshotRow],
        active: int | None,
        now: datetime,
        accounts: int = 2,
        standing: tuple[int, ...] = (),
    ) -> Advice:
        """Advise with ``active`` on the working login and each ``standing`` account on its own standing login."""
        rows = [account(index, label=f"Account {index}") for index in range(1, accounts + 1)]
        states = [source_state(active)]
        states += [source_state(account_id, position=position) for position, account_id in enumerate(standing, 1)]
        views = views_for(rows, latest, states, now)
        return advise([views[row.id] for row in rows], THRESHOLDS, now=now)

    def test_stay_while_the_active_account_is_below_the_critical_threshold(self, now: datetime) -> None:
        advice = self.advice_for([snapshot(1, now, 89.9), snapshot(2, now, 0)], 1, now)
        assert (advice.action, advice.active_account_id, advice.target_account_id) == (AdviceAction.STAY, 1, None)
        assert advice.summary == "Account 1 has 10% headroom."
        assert advice.as_of is None

    def test_switch_at_the_critical_threshold_when_another_account_has_a_current_reading_with_room(
        self, now: datetime
    ) -> None:
        advice = self.advice_for([snapshot(1, now, 90), snapshot(2, now, 40)], 1, now, standing=(2,))
        assert (advice.action, advice.target_account_id, advice.as_of) == (AdviceAction.SWITCH, 2, None)
        assert advice.summary == "Account 1 has 10% headroom left. Switch to Account 2 (60% headroom)."

    def test_a_reading_dated_after_now_on_a_standing_login_does_not_earn_a_definite_switch(self, now: datetime) -> None:
        """Definite advice needs a current reading, and a reading from the future is not one."""
        from_the_future = snapshot(2, now, 40, observed_ago=-timedelta(days=30))

        advice = self.advice_for([snapshot(1, now, 90), from_the_future], 1, now, standing=(2,))

        assert (advice.action, advice.target_account_id) == (AdviceAction.VERIFY, 2)

    def test_an_active_account_whose_only_reading_is_dated_after_now_gets_no_headroom_claim(
        self, now: datetime
    ) -> None:
        from_the_future = snapshot(1, now, 10, observed_ago=-timedelta(minutes=10))

        advice = self.advice_for([from_the_future, snapshot(2, now, 0)], 1, now, standing=(2,))

        assert advice.action is AdviceAction.UNKNOWN

    def test_switches_to_the_currently_read_account_with_the_most_headroom(self, now: datetime) -> None:
        latest = [snapshot(1, now, 100), snapshot(2, now, 70), snapshot(3, now, 20)]
        advice = self.advice_for(latest, 1, now, accounts=3, standing=(2, 3))
        assert (advice.action, advice.target_account_id) == (AdviceAction.SWITCH, 3)
        assert advice.summary.startswith("Account 1 is out of quota.")

    def test_an_account_not_being_read_is_offered_for_verification_not_as_a_confirmed_switch(
        self, now: datetime
    ) -> None:
        last_read = timedelta(hours=2)
        advice = self.advice_for([snapshot(1, now, 100), snapshot(2, now, 40, observed_ago=last_read)], 1, now)

        assert (advice.action, advice.target_account_id) == (AdviceAction.VERIFY, 2)
        assert advice.as_of == now - last_read
        assert advice.summary == (
            "Account 1 is out of quota. Account 2 had 60% headroom when last read. Log in to it to confirm."
        )

    def test_a_reset_projected_from_a_weeks_old_reading_is_never_reported_as_confirmed_headroom(
        self, now: datetime
    ) -> None:
        long_ago = timedelta(days=21)
        reset_since = snapshot(2, now, 100, resets_in=timedelta(days=-20), observed_ago=long_ago)

        advice = self.advice_for([snapshot(1, now, 100), reset_since], 1, now)

        assert advice.action is AdviceAction.VERIFY
        assert advice.as_of == now - long_ago
        assert "100% headroom" not in advice.summary
        assert "Switch to" not in advice.summary
        assert advice.summary == (
            "Account 1 is out of quota. Account 2 has reset since it was last read, so it may have room. "
            "Log in to it to confirm."
        )

    def test_a_currently_read_account_is_preferred_over_an_unread_one_that_showed_more_headroom(
        self, now: datetime
    ) -> None:
        latest = [snapshot(1, now, 100), snapshot(2, now, 5, observed_ago=timedelta(days=3)), snapshot(3, now, 60)]
        advice = self.advice_for(latest, 1, now, accounts=3, standing=(3,))
        assert (advice.action, advice.target_account_id) == (AdviceAction.SWITCH, 3)

    def test_among_unread_accounts_the_most_recently_read_is_offered(self, now: datetime) -> None:
        latest = [
            snapshot(1, now, 100),
            snapshot(2, now, 5, observed_ago=timedelta(days=9)),
            snapshot(3, now, 50, observed_ago=timedelta(hours=1)),
        ]
        advice = self.advice_for(latest, 1, now, accounts=3)
        assert (advice.action, advice.target_account_id) == (AdviceAction.VERIFY, 3)

    def test_a_reading_dated_after_now_does_not_outrank_one_that_already_happened(self, now: datetime) -> None:
        """A timestamp ahead of the dashboard is stale. The account read an hour ago is the one to verify."""
        an_hour_ago = timedelta(hours=1)
        latest = [
            snapshot(1, now, 100),
            snapshot(2, now, 50, observed_ago=an_hour_ago),
            snapshot(3, now, 10, observed_ago=-timedelta(days=30)),
        ]

        advice = self.advice_for(latest, 1, now, accounts=3)

        assert (advice.action, advice.target_account_id) == (AdviceAction.VERIFY, 2)
        assert advice.as_of == now - an_hour_ago

    def test_two_readings_dated_after_now_are_ranked_by_the_headroom_they_showed(self, now: datetime) -> None:
        """Neither timestamp is a read that has happened, so the larger headroom is offered."""
        latest = [
            snapshot(1, now, 100),
            snapshot(2, now, 40, observed_ago=-timedelta(days=1)),
            snapshot(3, now, 80, observed_ago=-timedelta(days=30)),
        ]

        advice = self.advice_for(latest, 1, now, accounts=3)

        assert (advice.action, advice.target_account_id) == (AdviceAction.VERIFY, 2)

    def test_a_reading_taken_at_now_outranks_one_dated_after_now(self, now: datetime) -> None:
        """A reading at the dashboard clock has happened. One dated after it has not."""
        latest = [
            snapshot(1, now, 100),
            snapshot(2, now, 50),
            snapshot(3, now, 10, observed_ago=-timedelta(seconds=1)),
        ]

        advice = self.advice_for(latest, 1, now, accounts=3)

        assert (advice.action, advice.target_account_id) == (AdviceAction.VERIFY, 2)
        assert advice.as_of == now

    def test_an_active_account_without_a_current_reading_gets_no_headroom_claim(self, now: datetime) -> None:
        last_read = timedelta(hours=9)
        advice = self.advice_for([snapshot(1, now, 30, observed_ago=last_read), snapshot(2, now, 0)], 1, now)

        assert (advice.action, advice.active_account_id, advice.target_account_id) == (AdviceAction.UNKNOWN, 1, None)
        assert advice.as_of == now - last_read
        assert advice.summary == "Account 1 has no current reading, so its usage now is unknown."

    def test_an_active_account_whose_window_reset_since_its_last_reading_gets_no_headroom_claim(
        self, now: datetime
    ) -> None:
        stale = snapshot(1, now, 100, resets_in=timedelta(minutes=-5), observed_ago=timedelta(minutes=10))
        advice = self.advice_for([stale], 1, now)
        assert advice.action is AdviceAction.UNKNOWN
        assert "headroom" not in advice.summary

    def test_hold_and_name_the_first_reset_when_every_account_is_exhausted(self, now: datetime) -> None:
        latest = [
            snapshot(1, now, 100, resets_in=timedelta(hours=3)),
            snapshot(2, now, 100, resets_in=timedelta(hours=1)),
        ]
        advice = self.advice_for(latest, 1, now, standing=(2,))
        assert advice.action is AdviceAction.HOLD
        assert advice.next_available_at == now + timedelta(hours=1)
        assert advice.as_of is None

    def test_hold_does_not_name_the_reset_of_an_account_without_a_current_reading(self, now: datetime) -> None:
        """Logged out, it may have been used since it was read; its reset is no promise of room."""
        latest = [
            snapshot(1, now, 100, resets_in=timedelta(hours=3)),
            snapshot(2, now - timedelta(days=1), 100, resets_in=timedelta(hours=25)),
        ]
        advice = self.advice_for(latest, 1, now)
        assert advice.action is AdviceAction.HOLD
        assert advice.next_available_at == now + timedelta(hours=3)

    def test_hold_without_a_wait_when_the_active_account_still_works_but_no_other_is_better(
        self, now: datetime
    ) -> None:
        advice = self.advice_for([snapshot(1, now, 95), snapshot(2, now, 92)], 1, now)
        assert (advice.action, advice.next_available_at) == (AdviceAction.HOLD, None)

    def test_an_account_at_a_limit_the_provider_reported_is_told_to_switch_not_to_stay(self, now: datetime) -> None:
        latest = [
            snapshot(1, now, 38, key="weekly"),
            snapshot(1, now, 100, key="reported_limit", resets_in=None, window_seconds=None),
            snapshot(2, now, 10),
        ]
        advice = self.advice_for(latest, 1, now, standing=(2,))
        assert (advice.action, advice.target_account_id) == (AdviceAction.SWITCH, 2)
        assert "Account 1 is out of quota" in advice.summary

    def test_an_account_with_no_reading_is_never_a_switch_target(self, now: datetime) -> None:
        advice = self.advice_for([snapshot(1, now, 100, resets_in=timedelta(hours=2))], 1, now)
        assert advice.action is AdviceAction.HOLD

    def test_unknown_when_nobody_is_logged_in_or_the_active_account_has_no_reading(self, now: datetime) -> None:
        nobody = self.advice_for([snapshot(1, now, 10)], None, now, standing=(1,))
        assert (nobody.action, nobody.summary) == (
            AdviceAction.UNKNOWN,
            "No account is logged in to the working login.",
        )
        no_data = self.advice_for([snapshot(2, now, 10)], 1, now)
        assert (no_data.action, no_data.summary) == (
            AdviceAction.UNKNOWN,
            "No limit that blocks Account 1 has been read yet.",
        )

    def test_an_account_with_only_informational_windows_has_no_blocking_limit_read(self, now: datetime) -> None:
        """It has quota data on its card, so the summary must not say there is none."""
        advice = self.advice_for([snapshot(1, now, 10, key="code_review:weekly", blocking=False)], 1, now)
        assert advice.summary == "No limit that blocks Account 1 has been read yet."


class TestWindowOrdering:
    def test_windows_are_listed_shortest_first_whatever_order_they_were_stored_in(self, now: datetime) -> None:
        latest = [
            snapshot(1, now, 5, key="credit_pool", window_seconds=None, blocking=False),
            snapshot(1, now, 5, key="weekly:fable", window_seconds=WEEK),
            snapshot(1, now, 5, key="unknown_length", window_seconds=None),
            snapshot(1, now, 5, key="weekly", window_seconds=WEEK),
            snapshot(1, now, 5, key="five_hour", window_seconds=FIVE_HOURS),
        ]

        windows = views_for([account(1)], latest, [], now)[1].windows

        assert [window.key for window in windows] == [
            "five_hour",
            "weekly",
            "weekly:fable",
            "unknown_length",
            "credit_pool",
        ]


class TestProviderGrouping:
    def test_every_provider_appears_and_accounts_are_grouped_with_the_active_one_first(self, now: datetime) -> None:
        accounts = [account(1), account(2), account(3, Provider.CODEX)]
        views = build_provider_views(
            accounts,
            [snapshot(2, now, 5)],
            [],
            [source_state(2), source_state(3, Provider.CODEX)],
            now=now,
            thresholds=THRESHOLDS,
            poll_interval_seconds=POLL_SECONDS,
        )

        by_provider = {view.provider: view for view in views}
        assert list(by_provider) == [Provider.CLAUDE, Provider.CODEX, Provider.GROK]
        assert [view.account.id for view in by_provider[Provider.CLAUDE].accounts] == [2, 1]
        assert by_provider[Provider.GROK].accounts == ()
        assert by_provider[Provider.GROK].advice.action is AdviceAction.UNKNOWN

    def test_the_first_configured_source_decides_which_account_is_active(self, now: datetime) -> None:
        states = [source_state(2, position=1), source_state(1, position=0)]
        views = views_for([account(1), account(2)], [], states, now)
        assert (views[1].is_active, views[2].is_active) == (True, False)

    def test_two_sources_at_position_zero_do_not_pick_an_account_by_row_order(self, now: datetime) -> None:
        """A reorder stores each source's position on its own, so both can be position 0 at once."""
        roomy = snapshot(1, now, 20)
        tight = snapshot(2, now, 95)
        accounts = [account(1, label="Old"), account(2, label="New")]
        codex = account(3, Provider.CODEX)
        claims = (
            (("old", 1), ("new", 2)),
            (("new", 2), ("old", 1)),
            (("old", 1), ("logged-out", None)),
            (("logged-out", None), ("old", 1)),
            (("same-a", 1), ("same-b", 1)),
        )

        for left, right in claims:
            present = {account_id for _source_id, account_id in (left, right) if account_id is not None}
            states = [
                SourceStateRow(left[0], Provider.CLAUDE, 0, left[1], now),
                SourceStateRow(right[0], Provider.CLAUDE, 0, right[1], now),
                SourceStateRow("codex", Provider.CODEX, 0, 3, now),
            ]
            shown = [item for item in accounts if item.id in present] + [codex]
            latest = [item for item in (roomy, tight) if item.account_id in present]
            built = build_provider_views(
                shown, latest, [], states, now=now, thresholds=THRESHOLDS, poll_interval_seconds=POLL_SECONDS
            )
            by_provider = {view.provider: view for view in built}

            claude = by_provider[Provider.CLAUDE]
            assert [view.is_active for view in claude.accounts] == [False] * len(claude.accounts)
            assert (claude.advice.action, claude.advice.active_account_id, claude.advice.summary) == (
                AdviceAction.UNKNOWN,
                None,
                "No account is logged in to the working login.",
            )
            assert by_provider[Provider.CODEX].accounts[0].is_active is True

    def test_a_logged_out_primary_source_leaves_no_account_active(self, now: datetime) -> None:
        states = [source_state(None, position=0), source_state(2, position=1)]
        views = views_for([account(1), account(2)], [], states, now)
        assert (views[1].is_active, views[2].is_active) == (False, False)
        assert views[2].is_live

    def test_a_standing_source_is_not_active_when_the_working_source_has_no_stored_state(self, now: datetime) -> None:
        """The lowest position present was position 1, so that standing account was treated as in use."""
        states = [source_state(1, position=1)]
        views = views_for([account(1)], [snapshot(1, now, 10)], states, now)
        advice = advise([views[1]], THRESHOLDS, now=now)

        assert (views[1].is_active, views[1].is_live) == (False, True)
        assert (advice.action, advice.summary) == (
            AdviceAction.UNKNOWN,
            "No account is logged in to the working login.",
        )
