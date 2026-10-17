from datetime import datetime

import pytest

from tests.factories import make_identity, make_window
from usagedash.models import (
    FULL_PERCENT,
    MAX_TOKEN_COUNT,
    MAX_WINDOW_SECONDS,
    AccountIdentity,
    Provider,
    QuotaReading,
    QuotaWindow,
    TokenCounts,
)


class TestProvider:
    def test_each_provider_has_a_display_name(self) -> None:
        assert [(provider.value, provider.display_name) for provider in Provider] == [
            ("claude", "Claude"),
            ("codex", "Codex"),
            ("grok", "Grok"),
        ]


class TestAccountIdentityInvariant:
    @pytest.mark.parametrize("external_id", ["", " ", "\t\n"])
    def test_an_identity_without_a_usable_external_id_cannot_be_represented(self, external_id: str) -> None:
        with pytest.raises(ValueError, match="non-blank external id"):
            AccountIdentity(Provider.CODEX, external_id, "Codex account")

    def test_an_identity_needs_only_its_provider_external_id_and_label(self) -> None:
        identity = AccountIdentity(Provider.CODEX, "workspace-a", "Codex account")

        assert (identity.email, identity.plan) == (None, None)


class TestQuotaWindowInvariant:
    @pytest.mark.parametrize(
        "used_percent", [-0.001, -1, float("inf"), float("-inf"), float("nan"), 10**400, -(10**400)]
    )
    def test_an_impossible_percentage_cannot_be_represented(self, used_percent: float) -> None:
        """An integer too large for a float is included: it must be refused, not raise OverflowError."""
        with pytest.raises(ValueError, match="impossible used_percent"):
            QuotaWindow("weekly", "Weekly", used_percent, None, None)

    @pytest.mark.parametrize("used_percent", [0, 0.0, 99.9, FULL_PERCENT, 340.5])
    def test_zero_a_full_window_and_an_overrun_are_all_valid(self, used_percent: float) -> None:
        assert QuotaWindow("weekly", "Weekly", used_percent, None, None).used_percent == used_percent

    @pytest.mark.parametrize("given", [0, 42, 2**63, True])
    def test_the_percentage_is_always_held_as_a_float(self, given: float) -> None:
        window = QuotaWindow("weekly", "Weekly", given, None, None)

        assert type(window.used_percent) is float
        assert window.used_percent == float(given)

    @pytest.mark.parametrize("window_seconds", [-1, MAX_WINDOW_SECONDS + 1, 10**400])
    def test_a_window_cannot_be_negative_or_longer_than_ten_years(self, window_seconds: int) -> None:
        with pytest.raises(ValueError, match="impossible length"):
            QuotaWindow("weekly", "Weekly", 1, None, window_seconds)

    @pytest.mark.parametrize("window_seconds", [None, 0, 18_000, MAX_WINDOW_SECONDS])
    def test_a_window_may_have_no_length_or_any_length_up_to_ten_years(self, window_seconds: int | None) -> None:
        assert QuotaWindow("weekly", "Weekly", 1, None, window_seconds).window_seconds == window_seconds

    def test_ten_years_is_the_longest_window(self) -> None:
        assert MAX_WINDOW_SECONDS == 315_360_000

    def test_a_window_blocks_unless_marked_informational(self) -> None:
        assert QuotaWindow("weekly", "Weekly", 1, None, None).blocking is True
        assert QuotaWindow("pool", "Pool", 1, None, None, blocking=False).blocking is False


class TestQuotaReadingInvariant:
    def test_two_windows_with_one_key_cannot_be_represented(self, now: datetime) -> None:
        windows = (make_window("weekly", 10), make_window("five_hour", 20), make_window("weekly", 30))

        with pytest.raises(ValueError, match="repeated window keys"):
            QuotaReading(make_identity(), windows, now, "api")

    def test_distinct_keys_and_an_empty_reading_are_valid(self, now: datetime) -> None:
        windows = (make_window("weekly", 10), make_window("five_hour", 20))

        assert QuotaReading(make_identity(), windows, now, "api").windows == windows
        assert QuotaReading(make_identity(), (), now, "log").windows == ()


class TestTokenCounts:
    def test_the_total_counts_each_token_once_and_leaves_reasoning_out(self) -> None:
        counts = TokenCounts(
            input_tokens=10, cache_read_tokens=100, cache_write_tokens=5, output_tokens=20, reasoning_tokens=8
        )
        assert counts.total_tokens == 135

    def test_adding_sums_every_field(self) -> None:
        assert TokenCounts(1, 2, 3, 4, 5, 6).plus(TokenCounts(10, 20, 30, 40, 50, 60)) == TokenCounts(
            11, 22, 33, 44, 55, 66
        )

    @pytest.mark.parametrize(
        "field",
        ["input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens", "requests"],
    )
    @pytest.mark.parametrize("count", [-1, -(10**400)])
    def test_a_negative_count_cannot_be_represented(self, field: str, count: int) -> None:
        with pytest.raises(ValueError, match="a token count is negative"):
            TokenCounts(**{field: count})

    @pytest.mark.parametrize(
        "field",
        ["input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens", "requests"],
    )
    @pytest.mark.parametrize("count", [0.5, 1.0, -0.5, float("nan"), float("inf"), True, False, "3", None])
    def test_a_count_that_is_not_a_whole_number_cannot_be_represented(self, field: str, count: object) -> None:
        """A fractional count would be stored as a real number and come back as a fractional total."""
        with pytest.raises(ValueError, match="a token count is not a whole number"):
            TokenCounts(**{field: count})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field",
        ["input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens", "requests"],
    )
    @pytest.mark.parametrize("count", [0, MAX_TOKEN_COUNT, MAX_TOKEN_COUNT + 1, 2**63, 10**400])
    def test_a_count_has_no_upper_limit_because_sums_of_stored_rows_have_none(self, field: str, count: int) -> None:
        assert getattr(TokenCounts(**{field: count}), field) == count

    def test_adding_is_exact_past_what_one_stored_row_or_a_64_bit_integer_can_hold(self) -> None:
        at_the_limit = TokenCounts(input_tokens=MAX_TOKEN_COUNT, requests=MAX_TOKEN_COUNT)
        total = TokenCounts()
        for _ in range(12):
            total = total.plus(at_the_limit)

        assert total == TokenCounts(input_tokens=12 * MAX_TOKEN_COUNT, requests=12 * MAX_TOKEN_COUNT)
        assert total.total_tokens > 2**63

    def test_only_a_record_with_every_count_at_zero_is_empty(self) -> None:
        assert TokenCounts().is_empty
        assert not TokenCounts(requests=1).is_empty
        assert not TokenCounts(output_tokens=1).is_empty

    @pytest.mark.parametrize(
        "field",
        ["input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens", "requests"],
    )
    def test_a_record_with_any_one_count_set_is_not_empty(self, field: str) -> None:
        """Reasoning tokens are outside the total, but a revision of them alone is still something to store."""
        assert not TokenCounts(**{field: 1}).is_empty
