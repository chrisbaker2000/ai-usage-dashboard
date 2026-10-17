"""Published retail rates for the token report, and the cost block the API attaches."""

from typing import Any

import pytest

from usagedash.api import _token_report_json
from usagedash.pricing import COST_NOTE, _claude, _usd, price_model_rows

MILLION = 1_000_000
COUNT_FIELDS = ("input", "cache_read", "cache_write", "output", "reasoning", "requests", "total")
STORED_MODELS = (
    ("claude", "claude-fable-5"),
    ("claude", "claude-fable-5-1"),
    ("claude", "claude-haiku-4-5-20251001"),
    ("claude", "claude-opus-4-7"),
    ("claude", "claude-opus-4-8"),
    ("claude", "claude-opus-5"),
    ("claude", "claude-opus-5-5"),
    ("codex", "gpt-5.6-sol"),
    ("codex", "gpt-6-astra"),
    ("codex", "gpt-6-luna"),
    ("codex", "gpt-6-sol"),
    ("codex", "gpt-6.1-sol"),
    ("grok", "grok-4.6-build"),
    ("grok", "grok-4.7-build"),
)


def row(provider: str, account_id: int, model: str, **counts: int) -> dict[str, Any]:
    """One model row shaped as ``build_token_report`` stores it. Reasoning is not part of the total."""
    components = {
        "input": 0,
        "cache_read": 0,
        "cache_write": 0,
        "output": 0,
        "reasoning": 0,
        "requests": 0,
    }
    components.update(counts)
    components["total"] = (
        components["input"] + components["cache_read"] + components["cache_write"] + components["output"]
    )
    return {"provider": provider, "account_id": account_id, "model": model, "totals": components}


def priced(provider: str, model: str, **counts: int) -> int | None:
    return price_model_rows([row(provider, 1, model, **counts)])[0].models[0].microdollars


def _counts(**overrides: int) -> dict[str, int]:
    counts: dict[str, int] = {name: 0 for name in COUNT_FIELDS}
    counts.update(overrides)
    counts["total"] = counts["input"] + counts["cache_read"] + counts["cache_write"] + counts["output"]
    return counts


def _report(models: list[dict[str, Any]]) -> dict[str, Any]:
    account_id = models[0]["account_id"] if models else 1
    return {
        "range": "24h",
        "series": [
            {
                "provider": "claude",
                "account_id": account_id,
                "points": [_counts()],
                "totals": _counts(),
            }
        ],
        "models": models,
        "provider_totals": [{"provider": "claude", "display_name": "Claude", "totals": _counts()}],
    }


class TestRounding:
    def test_half_a_microdollar_rounds_up_and_anything_under_it_is_zero(self) -> None:
        """gpt-5-nano input is $0.05 per million tokens, so ten tokens are exactly half a microdollar."""
        assert priced("codex", "gpt-5-nano", input=9) == 0
        assert priced("codex", "gpt-5-nano", input=10) == 1
        assert priced("codex", "gpt-5-nano", input=11) == 1

    def test_a_row_is_rounded_once_not_once_per_component(self) -> None:
        """One input and four cache reads of gpt-4.1-mini are 0.8 microdollars. Each piece alone is under half."""
        assert priced("codex", "gpt-4.1-mini", input=1, cache_read=4) == 1

    def test_an_account_adds_the_rounded_rows_and_does_not_round_again(self) -> None:
        """Each row is exactly half a microdollar. Together they are one, and the account still shows both."""
        account = price_model_rows(
            [
                row("codex", 1, "gpt-5-nano", input=10),
                row("codex", 1, "gpt-4.1-nano", input=5),
            ]
        )[0]

        assert [model.microdollars for model in account.models] == [1, 1]
        assert account.priced_microdollars == 2

    def test_a_count_past_a_float_stays_an_exact_integer(self) -> None:
        assert priced("codex", "gpt-6-astra", input=10**18) == 10**19


class TestPublishedRates:
    def test_reasoning_tokens_are_not_priced_again(self) -> None:
        output_only = priced("claude", "claude-haiku-4-5", output=MILLION)
        with_reasoning = priced("claude", "claude-haiku-4-5", output=MILLION, reasoning=MILLION)

        assert output_only == with_reasoning == 5 * MILLION

    def test_fable_cache_hits_use_the_published_multiplier(self) -> None:
        assert priced("claude", "claude-fable-5-1", cache_read=MILLION) == 250_000
        assert priced("claude", "claude-fable-5", cache_read=MILLION) == MILLION

    def test_a_claude_cache_write_uses_the_five_minute_rate(self) -> None:
        assert priced("claude", "claude-opus-4-7", input=MILLION) == 5 * MILLION
        assert priced("claude", "claude-opus-4-7", cache_write=MILLION) == 6_250_000

    def test_haiku_5_5_uses_the_short_prompt_rate(self) -> None:
        assert priced("claude", "claude-haiku-5-5", input=MILLION) == 100_000

    def test_a_trailing_claude_snapshot_date_uses_its_family_only_when_that_family_is_listed(self) -> None:
        assert priced("claude", "claude-haiku-4-5-20251001", input=MILLION) == MILLION
        assert priced("claude", "claude-opus-4-20250514", input=MILLION) == 15 * MILLION
        assert priced("claude", "claude-haiku-4-5-2025100", input=MILLION) is None
        assert priced("claude", "claude-not-listed-20251001", input=MILLION) is None
        assert priced("claude", "claude-opus-4-20250514-fast", input=MILLION) is None

    def test_an_openai_snapshot_keeps_its_own_rate(self) -> None:
        assert priced("codex", "gpt-4o", input=MILLION) == 2_500_000
        assert priced("codex", "gpt-4o-2024-05-13", input=MILLION) == 5 * MILLION

    def test_a_rate_applies_only_to_the_vendor_that_published_it(self) -> None:
        assert priced("codex", "claude-opus-4-7", input=MILLION) is None
        assert priced("anthropic", "claude-opus-4-7", input=MILLION) is None

    def test_a_blank_cache_write_cell_prices_a_row_that_wrote_nothing(self) -> None:
        assert priced("codex", "gpt-5.4", input=MILLION) == 2_500_000
        assert priced("codex", "gpt-5.4", cache_read=MILLION) == 250_000

    def test_one_used_component_without_a_rate_leaves_the_whole_row_unpriced(self) -> None:
        account = price_model_rows([row("codex", 1, "gpt-5.4", input=MILLION, cache_write=1)])[0]

        assert account.models[0].microdollars is None
        assert account.models[0].unpriced_tokens == MILLION + 1
        assert account.priced_microdollars == 0
        assert account.unpriced_tokens == MILLION + 1

    def test_grok_cache_writes_have_no_published_rate(self) -> None:
        assert priced("grok", "grok-4.7", input=MILLION) == 2 * MILLION
        assert priced("grok", "grok-4.7", input=MILLION, cache_write=1) is None

    def test_a_grok_build_id_uses_the_public_model_and_keeps_the_stored_name(self) -> None:
        for stored, public in (("grok-4.7-build", "grok-4.7"), ("grok-4.6-build", "grok-4.6")):
            build = price_model_rows([row("grok", 1, stored, input=MILLION, cache_read=MILLION)])[0]
            listed = price_model_rows([row("grok", 1, public, input=MILLION, cache_read=MILLION)])[0]

            assert build.models[0].model == stored
            assert build.priced_microdollars == listed.priced_microdollars == 2_500_000

    def test_grok_fast_is_not_priced_as_the_public_model(self) -> None:
        assert priced("grok", "grok-4.7-fast", input=MILLION) is None

    @pytest.mark.parametrize(
        ("model", "component", "microdollars"),
        [
            ("gpt-4o-mini", "cache_read", 75_000),
            ("o4-mini", "cache_read", 275_000),
            ("gpt-6-luna", "cache_write", 125_000),
            ("gpt-5.6-cyber", "cache_write", 15_625_000),
        ],
    )
    def test_a_fractional_dollar_rate_is_an_exact_number_of_microdollars(
        self, model: str, component: str, microdollars: int
    ) -> None:
        assert priced("codex", model, **{component: MILLION}) == microdollars

    @pytest.mark.parametrize(("provider", "model"), STORED_MODELS)
    def test_a_model_this_dashboard_has_stored_has_a_rate(self, provider: str, model: str) -> None:
        cost = price_model_rows([row(provider, 1, model, input=MILLION)])[0].models[0]

        assert cost.microdollars is not None
        assert cost.microdollars > 0
        assert cost.unpriced_tokens == 0


class TestRateTableGuards:
    def test_a_price_that_is_not_a_non_negative_amount_is_refused(self) -> None:
        for text in ("-1", "NaN", "Infinity"):
            with pytest.raises(ValueError, match=r"^a published price is not a non-negative amount$"):
                _usd(text)

    def test_a_price_finer_than_one_microdollar_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"^a published price is finer than one microdollar$"):
            _usd("0.0000001")

    def test_a_claude_cache_price_that_is_not_a_whole_microdollar_is_refused(self) -> None:
        for input_usd, denominator in (("0.000001", 10), ("1", 3)):
            with pytest.raises(ValueError, match=r"^a Claude cache price is not a whole number of microdollars$"):
                _claude(input_usd, "1", denominator)


class TestAccountCosts:
    def test_an_unknown_model_is_unpriced_even_when_it_used_nothing(self) -> None:
        account = price_model_rows([row("claude", 1, "not-a-model")])[0]

        assert account.models[0].microdollars is None
        assert account.models[0].unpriced_tokens == 0
        assert account.priced_microdollars == 0

    def test_a_known_model_that_used_nothing_costs_nothing(self) -> None:
        cost = price_model_rows([row("claude", 1, "claude-haiku-4-5")])[0].models[0]

        assert cost.microdollars == 0
        assert cost.unpriced_tokens == 0

    def test_an_unpriced_row_is_left_out_of_the_dollar_total(self) -> None:
        account = price_model_rows(
            [
                row("claude", 1, "claude-opus-4-7", input=MILLION),
                row("claude", 1, "not-a-model", input=3),
            ]
        )[0]

        assert account.priced_microdollars == 5 * MILLION
        assert account.unpriced_tokens == 3

    def test_no_rows_is_no_accounts(self) -> None:
        assert price_model_rows([]) == ()

    def test_accounts_stay_in_the_order_the_rows_arrive(self) -> None:
        costs = price_model_rows(
            [
                row("grok", 2, "grok-4.7", input=MILLION),
                row("claude", 1, "claude-haiku-4-5", input=MILLION),
            ]
        )

        assert [(account.provider, account.account_id) for account in costs] == [("grok", 2), ("claude", 1)]

    def test_the_same_model_twice_is_priced_twice(self) -> None:
        account = price_model_rows(
            [
                row("claude", 1, "claude-haiku-4-5", input=MILLION),
                row("claude", 1, "claude-haiku-4-5", input=MILLION),
            ]
        )[0]

        assert len(account.models) == 2
        assert account.priced_microdollars == 2 * MILLION

    def test_an_unattributed_bucket_is_an_account_in_the_cost(self) -> None:
        account = price_model_rows([row("claude", 0, "claude-haiku-4-5", input=MILLION)])[0]

        assert account.account_id == 0
        assert account.priced_microdollars == MILLION

    def test_a_bool_or_a_negative_count_is_refused_without_the_count(self) -> None:
        boolean = row("claude", 1, "claude-haiku-4-5")
        boolean["totals"]["input"] = True
        negative = row("claude", 1, "claude-haiku-4-5", input=1)
        negative["totals"]["output"] = -1
        fractional = row("claude", 1, "claude-haiku-4-5")
        fractional["totals"]["cache_read"] = 1.0
        uncounted = row("claude", 1, "claude-haiku-4-5")
        uncounted["totals"] = None

        for refused in (boolean, negative, fractional, uncounted):
            with pytest.raises(ValueError, match=r"^a token count is not a whole number$"):
                price_model_rows([refused])

    def test_an_account_id_that_is_not_a_whole_number_is_refused(self) -> None:
        boolean = row("claude", 1, "claude-haiku-4-5")
        boolean["account_id"] = True
        negative = row("claude", 1, "claude-haiku-4-5")
        negative["account_id"] = -1

        for refused in (boolean, negative):
            with pytest.raises(ValueError, match=r"^an account id is not a whole number$"):
                price_model_rows([refused])

    def test_a_total_that_is_missing_or_does_not_match_its_components_is_refused(self) -> None:
        missing = row("claude", 1, "claude-haiku-4-5", input=1)
        del missing["totals"]["total"]
        mismatched = row("claude", 1, "claude-haiku-4-5", input=1)
        mismatched["totals"]["total"] = 5

        with pytest.raises(KeyError):
            price_model_rows([missing])
        with pytest.raises(ValueError, match=r"^a token total does not match its components$"):
            price_model_rows([mismatched])

    def test_a_provider_or_a_model_that_is_not_a_name_is_refused(self) -> None:
        unnamed_provider = row("claude", 1, "claude-haiku-4-5")
        unnamed_provider["provider"] = 1
        unnamed_model = row("claude", 1, "claude-haiku-4-5")
        unnamed_model["model"] = None

        for refused in (unnamed_provider, unnamed_model):
            with pytest.raises(ValueError, match=r"^a provider or a model is not a name$"):
                price_model_rows([refused])

    def test_the_note_names_the_day_and_the_limits_of_the_price(self) -> None:
        assert "2026-10-08" in COST_NOTE
        assert "short prompt" in COST_NOTE
        assert "5-minute" in COST_NOTE
        assert "nearest dollar" in COST_NOTE
        assert "not a subscription charge" in COST_NOTE
        assert "grok-4.7-build" in COST_NOTE


class TestCostJson:
    def test_ids_and_amounts_leave_as_decimal_text(self) -> None:
        account_id = 2**53 + 1
        body = _token_report_json(
            _report(
                [
                    row("claude", account_id, "claude-opus-4-7", input=MILLION),
                    row("claude", account_id, "not-a-model", input=3),
                    row("codex", account_id, "gpt-6-astra", input=10**18),
                ]
            )
        )

        assert body["series"][0]["account_id"] == str(account_id)
        assert body["series"][0]["points"][0]["input"] == "0"
        assert body["models"][0]["totals"]["input"] == str(MILLION)
        account = body["costs"]["accounts"][0]
        assert account["account_id"] == str(account_id)
        assert account["priced_microdollars"] == str(5 * MILLION)
        assert account["unpriced_tokens"] == "3"
        priced_model, unpriced_model = account["models"][0], account["models"][1]
        assert priced_model["priced"] is True
        assert priced_model["microdollars"] == str(5 * MILLION)
        assert unpriced_model["priced"] is False
        assert unpriced_model["microdollars"] is None
        assert unpriced_model["model"] == "not-a-model"
        astra = body["costs"]["accounts"][1]
        assert astra["priced_microdollars"] == str(10**19)
        assert body["costs"]["priced_on"] == "2026-10-08"
        assert body["costs"]["currency"] == "USD"
        assert body["costs"]["unit"] == "microdollar"
        assert body["costs"]["note"] == COST_NOTE

    def test_no_usage_still_carries_an_empty_cost_block(self) -> None:
        body = _token_report_json(_report([]))

        assert body["costs"]["accounts"] == []
        assert body["costs"]["note"] == COST_NOTE
