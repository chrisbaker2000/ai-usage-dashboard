"""Retail pay-as-you-go prices for the token report's model rows.

Rates are the vendors' published standard prices for a short prompt, in microdollars per
million tokens (one US dollar is 1,000,000 microdollars). They were read on 2026-10-08 from
the Claude pricing page, the OpenAI standard pricing table, and the xAI text-API table.

An hourly total does not say whether a request was long-context, batch, fast, priority, or
served from a regional endpoint, so those prices are not used. Claude cache writes are one
stored count, so they use the published 5-minute write price (1.25 times input). Cache hits
use the published multiplier: 0.1 times input, 0.05 on Opus 5.5 and Sonnet 5.5, and 0.025 on
Fable 5.1 and Mythos 5.1. Haiku 5.5's higher price above 100,000 prompt tokens is not used.
A blank cell in a vendor table is no price, not zero. A component that was used and has no
price leaves the whole model out of the dollar total. Reasoning tokens are part of output
and are not priced again.

``grok-4.7-build`` and ``grok-4.6-build`` are the subscription CLI's names for ``grok-4.7``
and ``grok-4.6``. Grok 4.7 Fast is a different product and is not priced.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from usagedash.models import Provider

PRICED_ON = "2026-10-08"
"""The day the published tables in this module were read."""

CURRENCY = "USD"
MICRODOLLAR = "microdollar"
"""The unit of every cost this module returns. One US dollar is 1,000,000 of them."""

COST_NOTE = (
    f"Published standard API rates as of {PRICED_ON}, for a short prompt. "
    "Long-context, batch, fast mode, priority, and regional surcharges are not included. "
    "Prompt-length tiers use the short-prompt rate. Claude cache writes use the 5-minute rate. "
    "grok-4.7-build and grok-4.6-build are priced as grok-4.7 and grok-4.6. "
    "A model with no published rate for a component that was used is left out of the total. "
    "Each figure is rounded to the nearest dollar. "
    "This is not a subscription charge."
)

_MICRODOLLARS_PER_DOLLAR = 1_000_000
_TOKENS_PER_MILLION = 1_000_000
_HALF_MICRODOLLAR = _TOKENS_PER_MILLION // 2
_COMPONENTS = ("input", "cache_read", "cache_write", "output")
_FIVE_MINUTE_WRITE = 5, 4
_CLAUDE_SNAPSHOT = re.compile(r"^(?P<family>.+)-\d{8}$")
# Why these two names: the CLI records them. They are the public models, not Grok 4.7 Fast.
_GROK_ALIASES = {"grok-4.7-build": "grok-4.7", "grok-4.6-build": "grok-4.6"}


def _usd(text: str) -> int:
    """Return one published USD-per-million price in microdollars per million tokens."""
    amount = Decimal(text)
    if not amount.is_finite() or amount < 0:
        raise ValueError("a published price is not a non-negative amount")
    micro = amount * _MICRODOLLARS_PER_DOLLAR
    if micro != micro.to_integral_value():
        raise ValueError("a published price is finer than one microdollar")
    return int(micro)


@dataclass(frozen=True, slots=True)
class _Rates:
    """Microdollars per million tokens. None is a published blank, not a price of zero."""

    input: int | None
    cache_read: int | None
    cache_write: int | None
    output: int | None

    def component(self, name: str) -> int | None:
        rates = {
            "input": self.input,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "output": self.output,
        }
        return rates[name]


def _listed(input_usd: str, cache_read: str | None, cache_write: str | None, output_usd: str) -> _Rates:
    return _Rates(
        _usd(input_usd),
        None if cache_read is None else _usd(cache_read),
        None if cache_write is None else _usd(cache_write),
        _usd(output_usd),
    )


def _claude(input_usd: str, output_usd: str, cache_hit_denominator: int) -> _Rates:
    """Build Claude rates. The cache hit is input divided by ``cache_hit_denominator``."""
    input_rate = _usd(input_usd)
    numerator, denominator = _FIVE_MINUTE_WRITE
    write = input_rate * numerator
    if write % denominator or input_rate % cache_hit_denominator:
        raise ValueError("a Claude cache price is not a whole number of microdollars")
    return _Rates(input_rate, input_rate // cache_hit_denominator, write // denominator, _usd(output_usd))


def _grok(input_usd: str, cache_read: str, output_usd: str) -> _Rates:
    """Build an xAI text rate. The text table publishes no cache-write price."""
    return _listed(input_usd, cache_read, None, output_usd)


# Cache-hit denominator: 10 is 0.1x input, 20 is 0.05x, 40 is 0.025x.
_CLAUDE = {
    model: _claude(input_usd, output_usd, hit)
    for model, (input_usd, output_usd, hit) in {
        "claude-fable-5-1": ("10", "50", 40),
        "claude-mythos-5-1": ("10", "50", 40),
        "claude-opus-5-5": ("4", "20", 20),
        "claude-sonnet-5-5": ("2", "10", 20),
        "claude-haiku-5-5": ("0.10", "0.50", 10),
        "claude-fable-5": ("10", "50", 10),
        "claude-mythos-5": ("10", "50", 10),
        "claude-opus-5": ("5", "25", 10),
        "claude-opus-4-8": ("5", "25", 10),
        "claude-opus-4-7": ("5", "25", 10),
        "claude-opus-4-6": ("5", "25", 10),
        "claude-opus-4-5": ("5", "25", 10),
        "claude-opus-4-1": ("15", "75", 10),
        "claude-opus-4": ("15", "75", 10),
        "claude-sonnet-5": ("2", "10", 10),
        "claude-sonnet-4-6": ("3", "15", 10),
        "claude-sonnet-4-5": ("3", "15", 10),
        "claude-sonnet-4": ("3", "15", 10),
        "claude-haiku-4-5": ("1", "5", 10),
        "claude-3-5-haiku": ("0.80", "4", 10),
    }.items()
}

# Short-context standard rates. A None cell is the vendor's "-", and the long-context columns are not used.
_CODEX = {
    model: _listed(*prices)
    for model, prices in {
        "gpt-6-astra": ("10.00", "1.00", "12.50", "50.00"),
        "gpt-6.1-sol": ("2.00", "0.10", "2.50", "10.00"),
        "gpt-6-luna": ("0.10", "0.01", "0.125", "0.50"),
        "gpt-6-sol": ("2.00", "0.20", "2.50", "10.00"),
        "gpt-5.6-sol": ("4.00", "0.40", "5.00", "20.00"),
        "gpt-5.6-terra": ("2.00", "0.20", "2.50", "12.00"),
        "gpt-5.6-luna": ("0.20", "0.02", "0.25", "1.20"),
        "gpt-5.5": ("5.00", "0.50", None, "30.00"),
        "gpt-5.5-pro": ("30.00", None, None, "180.00"),
        "gpt-5.4": ("2.50", "0.25", None, "15.00"),
        "gpt-5.4-mini": ("0.75", "0.075", None, "4.50"),
        "gpt-5.4-nano": ("0.20", "0.02", None, "1.25"),
        "gpt-5.4-pro": ("30.00", None, None, "180.00"),
        "gpt-5.2": ("1.75", "0.175", None, "14.00"),
        "gpt-5.2-pro": ("21.00", None, None, "168.00"),
        "gpt-5.1": ("1.25", "0.125", None, "10.00"),
        "gpt-5": ("1.25", "0.125", None, "10.00"),
        "gpt-5-mini": ("0.25", "0.025", None, "2.00"),
        "gpt-5-nano": ("0.05", "0.005", None, "0.40"),
        "gpt-5-pro": ("15.00", None, None, "120.00"),
        "gpt-4.1": ("2.00", "0.50", None, "8.00"),
        "gpt-4.1-mini": ("0.40", "0.10", None, "1.60"),
        "gpt-4.1-nano": ("0.10", "0.025", None, "0.40"),
        "gpt-4o": ("2.50", "1.25", None, "10.00"),
        "gpt-4o-2024-05-13": ("5.00", None, None, "15.00"),
        "gpt-4o-mini": ("0.15", "0.075", None, "0.60"),
        "o1": ("15.00", "7.50", None, "60.00"),
        "o1-pro": ("150.00", None, None, "600.00"),
        "o3-pro": ("20.00", None, None, "80.00"),
        "o3": ("2.00", "0.50", None, "8.00"),
        "o4-mini": ("1.10", "0.275", None, "4.40"),
        "o3-mini": ("1.10", "0.55", None, "4.40"),
        "gpt-4-turbo-2024-04-09": ("10.00", None, None, "30.00"),
        "gpt-4-0613": ("30.00", None, None, "60.00"),
        "gpt-3.5-turbo": ("0.50", None, None, "1.50"),
        "gpt-3.5-turbo-0125": ("0.50", None, None, "1.50"),
        "gpt-3.5-turbo-1106": ("1.00", None, None, "2.00"),
        "gpt-3.5-turbo-instruct": ("1.50", None, None, "2.00"),
        "davinci-002": ("2.00", None, None, "2.00"),
        "babbage-002": ("0.40", None, None, "0.40"),
        "gpt-5.6-cyber": ("12.50", "1.25", "15.625", "75.00"),
        "gpt-5.5-cyber": ("12.50", "1.25", None, "75.00"),
        "gpt-5.3-codex": ("1.75", "0.175", None, "14.00"),
        "chat-latest": ("5.00", "0.50", None, "30.00"),
        "gpt-rosalind-research": ("5.00", "0.50", None, "25.00"),
        "gpt-5-search-api": ("1.25", "0.125", None, "10.00"),
    }.items()
}

_GROK = {
    model: _grok(*prices)
    for model, prices in {
        "grok-4.7": ("2.00", "0.50", "6.00"),
        "grok-4.6": ("2.00", "0.50", "6.00"),
        "grok-4.5": ("2.00", "0.30", "6.00"),
        "grok-build-0.1": ("1.00", "0.20", "2.00"),
        "grok-4.3": ("1.25", "0.20", "2.50"),
        "grok-4.20-multi-agent-0309": ("1.25", "0.20", "2.50"),
        "grok-4.20-0309-reasoning": ("1.25", "0.20", "2.50"),
        "grok-4.20-0309-non-reasoning": ("1.25", "0.20", "2.50"),
    }.items()
}

_RATES = {
    **{(Provider.CLAUDE, model): rates for model, rates in _CLAUDE.items()},
    **{(Provider.CODEX, model): rates for model, rates in _CODEX.items()},
    **{(Provider.GROK, model): rates for model, rates in _GROK.items()},
}


def _published_model(provider: Provider, model: str) -> str:
    """Return the price-list id for a stored model name."""
    if provider is Provider.GROK:
        return _GROK_ALIASES.get(model, model)
    if provider is Provider.CLAUDE:
        # Why only when the family is listed: a trailing date is a snapshot of that family.
        # A dated id whose family is not listed stays unmatched instead of losing the date.
        snapshot = _CLAUDE_SNAPSHOT.fullmatch(model)
        if snapshot is not None and (provider, snapshot.group("family")) in _RATES:
            return snapshot.group("family")
    return model


def _whole(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        article = "an" if what.startswith(("a", "e", "i", "o", "u")) else "a"
        raise ValueError(f"{article} {what} is not a whole number")
    return value


def _component_counts(totals: object) -> tuple[dict[str, int], int]:
    if not isinstance(totals, Mapping):
        raise ValueError("a token count is not a whole number")
    counts = {name: _whole(totals[name], "token count") for name in _COMPONENTS}
    total = _whole(totals["total"], "token count")
    if total != sum(counts.values()):
        raise ValueError("a token total does not match its components")
    return counts, total


def _microdollars(rates: _Rates, counts: Mapping[str, int]) -> int | None:
    """Return the row cost, or None when a used component has no published price.

    Why once for the row: rounding each component would drop any piece under half a
    microdollar, and the account total is the sum of these row costs.
    """
    numerator = 0
    for name in _COMPONENTS:
        count = counts[name]
        if count == 0:
            continue
        rate = rates.component(name)
        if rate is None:
            return None
        numerator += count * rate
    return (numerator + _HALF_MICRODOLLAR) // _TOKENS_PER_MILLION


@dataclass(frozen=True, slots=True)
class ModelCost:
    """One model row. ``microdollars`` is None when the row is left out of the dollar total."""

    model: str
    microdollars: int | None
    unpriced_tokens: int


@dataclass(frozen=True, slots=True)
class AccountCost:
    """One account. ``priced_microdollars`` is the sum of its priced model rows."""

    provider: str
    account_id: int
    priced_microdollars: int
    unpriced_tokens: int
    models: tuple[ModelCost, ...]


@dataclass(slots=True)
class _AccountDraft:
    provider: str
    account_id: int
    models: list[ModelCost] = field(default_factory=list)
    priced_microdollars: int = 0
    unpriced_tokens: int = 0


def price_model_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[AccountCost, ...]:
    """Price each model row and sum the priced rows by account, in the order given.

    A model with no listed rate, and a model that used a component with no listed rate, is
    returned with ``microdollars`` None and its token total in ``unpriced_tokens``. A priced
    row contributes its rounded cost and no unpriced tokens. The account cost is the sum of
    those rounded row costs, so it matches the rows.

    Raises:
        ValueError: if a provider, model, account id, or token count is not the shape the
            token report stores. A provider name that is not one of the three is not an
            error: that row is unpriced.
        KeyError: if a row has no provider, account id, model, or totals.
    """
    drafts: dict[tuple[str, int], _AccountDraft] = {}
    order: list[tuple[str, int]] = []
    for row in rows:
        provider_name = row["provider"]
        model = row["model"]
        if not isinstance(provider_name, str) or not isinstance(model, str):
            raise ValueError("a provider or a model is not a name")
        account_id = _whole(row["account_id"], "account id")
        counts, total = _component_counts(row["totals"])
        try:
            provider = Provider(provider_name)
        except ValueError:
            provider = None
        rates = None if provider is None else _RATES.get((provider, _published_model(provider, model)))
        microdollars = None if rates is None else _microdollars(rates, counts)
        cost = ModelCost(
            model=model,
            microdollars=microdollars,
            unpriced_tokens=0 if microdollars is not None else total,
        )
        key = (provider_name, account_id)
        draft = drafts.get(key)
        if draft is None:
            draft = _AccountDraft(provider_name, account_id)
            drafts[key] = draft
            order.append(key)
        draft.models.append(cost)
        if microdollars is None:
            draft.unpriced_tokens += total
        else:
            draft.priced_microdollars += microdollars
    return tuple(
        AccountCost(
            provider=drafts[key].provider,
            account_id=drafts[key].account_id,
            priced_microdollars=drafts[key].priced_microdollars,
            unpriced_tokens=drafts[key].unpriced_tokens,
            models=tuple(drafts[key].models),
        )
        for key in order
    )
