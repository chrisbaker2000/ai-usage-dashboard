"""Rates fetched from the vendors' pricing pages: reading a page, fetching it, storing it, and pricing with it.

Every page here is a synthetic cut of the layout each vendor's pricing page had on 2026-11-10.
"""

import contextlib
import json
import subprocess
import threading
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import httpx2
import pytest
from fastapi.testclient import TestClient

from usagedash import cli, db, rate_updates
from usagedash.api import _token_report_json, create_app
from usagedash.config import Settings, load_settings
from usagedash.credentials import KeychainItem
from usagedash.db import Database, FetchedRateRow
from usagedash.errors import (
    ConfigError,
    CredentialExpiredError,
    CredentialUnavailableError,
    ProviderRequestError,
    ProviderSchemaError,
)
from usagedash.http import post_json_object
from usagedash.models import Provider
from usagedash.pricing import COST_NOTE, Rates, _usd, cost_note, price_model_rows, rate_in_use
from usagedash.rate_pages import (
    MAX_PAGE_CHARACTERS,
    RATE_PAGES,
    read_claude_rates,
    read_openai_rates,
    read_rates,
    read_xai_rates,
)
from usagedash.rate_updates import (
    FIRECRAWL_KEYCHAIN_SERVICE,
    FIRECRAWL_SCRAPE_URL,
    PageOutcome,
    RateChange,
    RateUpdateReport,
    fetched_rate_table,
    firecrawl_keychain_item,
    store_key_interactively,
    update_rates,
)

MILLION = 1_000_000
BASE_URL = "http://127.0.0.1:8787"
ACCESS_TOKEN = "synthetic-access-token-" + "y" * 20
PAGE_HEADERS = {"X-Requested-With": "usagedash", "Origin": BASE_URL}
FIRECRAWL_KEY = "fc-synthetic-key"
PAGE_TEXT_MARKER = "TEXT-ONLY-THE-PAGE-HOLDS"

CLAUDE_HEADER = (
    "| Model | Base tokens | Prompt caching |\n"
    "| --- | --- | --- |\n"
    "| Name | Input | Output | 5m writes | 1h writes | Hits and refreshes |\n"
    "| --- | --- | --- | --- | --- | --- |\n"
)
CLAUDE_BATCH_TABLE = (
    "\n| Model | Batch tokens |\n"
    "| --- | --- |\n"
    "| Name | Input | Output |\n"
    "| --- | --- | --- |\n"
    "| [Claude Fable 5.2](https://example.test/models/fable-5-2/overview) | $4/ MTok | $20/ MTok |\n"
)
CLAUDE_PAGE = (
    "# Pricing\n\n"
    + CLAUDE_HEADER
    + "| ![](https://example.test/head.svg)<br>[Claude Fable 5.2](https://example.test/models/fable-5-2/overview)"
    " For demanding reasoning | $8/ MTok | $40/ MTok | $10/ MTok | $16/ MTok | $0.20/ MTok |\n"
    "| [Claude Haiku 5.5](https://example.test/models/haiku-5-5/overview) For high-volume tasks "
    "| $0.10/ MTokfor prompts up to 100,000 tokens | $0.50/ MTok | $0.125/ MTok | $0.20/ MTok | $0.01/ MTok |\n"
    "| $0.50/ MTokfor prompts over 100,000 tokens | $2.50/ MTok | $0.625/ MTok | $1/ MTok | $0.05/ MTok |\n"
    "| Additional models |\n"
    "| Claude Opus 4.1 | $15/ MTok | $75/ MTok | $18.75/ MTok | $30/ MTok | $1.50/ MTok |\n"
    "| Claude Sonnet 4 | $3/ MTok | $15/ MTok | $3.75/ MTok | $6/ MTok | $0.30/ MTok |\n" + CLAUDE_BATCH_TABLE
)

OPENAI_CACHING_HEADER = "|  | Prompt caching |\n| --- | --- |\n"
OPENAI_COLUMNS = (
    "| Model | Input | Output | Cached input | 30m write | 12h write |\n| --- | --- | --- | --- | --- | --- |\n"
)
OPENAI_PAGE = (
    "## Flagship models\n\nStandardFlex/BatchFastUltrafast\n\nStandard\n\n"
    + OPENAI_CACHING_HEADER
    + OPENAI_COLUMNS
    + "| gpt-6-astra | $10.00 | $50.00 | $1.00 | $12.50 | $25.00 |\n"
    "| gpt-6.1-terra | $1.00 | $5.00 | $0.05 | $1.25 | $2.50 |\n"
    "\nAll models\n\nBatch\n\n"
    + OPENAI_CACHING_HEADER
    + OPENAI_COLUMNS
    + "| gpt-6-astra | $5.00 | $25.00 | $0.50 | $6.25 | $12.50 |\n"
    "\nAll models\n\nFast\n\n"
    + OPENAI_CACHING_HEADER
    + OPENAI_COLUMNS
    + "| gpt-6-astra | $20.00 | $100.00 | $2.00 | $25.00 | $50.00 |\n"
    "\n### Images\n\nStandard\n\n"
    "| Model | Modality | Input | Output | Cached input |\n| --- | --- | --- | --- | --- |\n"
    "| gpt-image-2 | Image | $8.00 | $30.00 | $2.00 |\n| Text | $5.00 | - | $1.25 |\n"
    "\n## Specialized models\n\nStandard\n\n"
    + OPENAI_CACHING_HEADER
    + "| Category | Model | Input | Output | Cached input | 30m write | 12h write |\n"
    "| --- | --- | --- | --- | --- | --- | --- |\n"
    "| CyberTrusted Access: [Daybreak](https://example.test/daybreak/) | gpt-6-cyber | $20.00 | $100.00 | $2.00 "
    "| $25.00 | $50.00 |\n"
    "| ChatGPT | chat-latest | $5.00 | $30.00 | $0.50 | - | - |\n"
    "\n## Fine-tuning\n\nStandard\n\n"
    "| Model | Training | Input | Output | Cached input |\n| --- | --- | --- | --- | --- |\n"
    "| gpt-4.1-2025-04-14 | $25.00 | $3.00 | $12.00 | $0.75 |\n"
)

XAI_HEADER = (
    "| Model | Context | Short context | Long context |\n"
    "| --- | --- | --- | --- |\n"
    "| Input | Cached | Output | Input | Cached | Output |\n"
    "| --- | --- | --- | --- | --- | --- |\n"
)
XAI_PAGE = (
    "### Text API\n\nPrices per 1M tokens\n\n"
    + XAI_HEADER
    + "| [grok-4.7](https://example.test/models/grok-4.7) <br>Long context ≥ 200k tokens | 500k | $2.00 | $0.50 "
    "| $6.00 | $4.00 | $1.00 | $12.00 |\n"
    "| [grok-4.5](https://example.test/models/grok-4.5) <br>Long context ≥ 200k tokens | 500k | $2.00 | $0.30 "
    "| $6.00 | $4.00 | $0.60 | $12.00 |\n"
    "\n### Imagine API\n\n| Model | Media Input | Resolution | Output |\n| --- | --- | --- | --- |\n"
    "| [grok-imagine-image](https://example.test/models/grok-imagine-image) | $0.002 / img | 1K | $0.02 / img |\n"
)

PAGES = {
    RATE_PAGES[Provider.CLAUDE].url: CLAUDE_PAGE,
    RATE_PAGES[Provider.CODEX].url: OPENAI_PAGE,
    RATE_PAGES[Provider.GROK].url: XAI_PAGE,
}

FABLE_5_2 = Rates(8 * MILLION, 200_000, 10 * MILLION, 40 * MILLION)


def claude_row(name: str, *cells: str) -> str:
    return CLAUDE_HEADER + f"| {name} | " + " | ".join(cells) + " |\n"


class TestClaudePage:
    def test_reads_each_model_at_its_short_prompt_five_minute_rate(self) -> None:
        assert read_claude_rates(CLAUDE_PAGE) == {
            "claude-fable-5-2": FABLE_5_2,
            "claude-haiku-5-5": Rates(100_000, 10_000, 125_000, 500_000),
            "claude-opus-4-1": Rates(15 * MILLION, 1_500_000, 18_750_000, 75 * MILLION),
            "claude-sonnet-4": Rates(3 * MILLION, 300_000, 3_750_000, 15 * MILLION),
        }

    def test_the_batch_table_is_not_read(self) -> None:
        assert read_claude_rates(CLAUDE_PAGE)["claude-fable-5-2"].input == 8 * MILLION

    def test_a_page_without_the_rate_table_is_refused(self) -> None:
        with pytest.raises(ProviderSchemaError, match=r"exactly one table with the expected rate columns"):
            read_claude_rates("# Pricing\n\n" + CLAUDE_BATCH_TABLE)

    def test_a_page_with_two_such_tables_is_refused_instead_of_choosing_one(self) -> None:
        row = claude_row("Claude Opus 5", "$5/ MTok", "$25/ MTok", "$6.25/ MTok", "$10/ MTok", "$0.50/ MTok")
        with pytest.raises(ProviderSchemaError, match=r"exactly one table"):
            read_claude_rates(row + "\nBatch\n\n" + row)

    def test_a_table_that_lists_no_model_is_refused(self) -> None:
        with pytest.raises(ProviderSchemaError, match=r"lists no model"):
            read_claude_rates(CLAUDE_HEADER + "| Additional models |\n")

    @pytest.mark.parametrize(
        ("row", "reason"),
        [
            pytest.param(
                claude_row(PAGE_TEXT_MARKER, "$5/ MTok", "$25/ MTok", "$6.25/ MTok", "$10/ MTok", "$0.50/ MTok"),
                "names no model",
                id="no model name",
            ),
            pytest.param(
                claude_row("Claude Opus 5", PAGE_TEXT_MARKER, "$25/ MTok", "$6.25/ MTok", "$10/ MTok", "$0.50/ MTok"),
                "not a dollar amount",
                id="input is not a price",
            ),
            pytest.param(
                claude_row("Claude Opus 5", "-", "$25/ MTok", "$6.25/ MTok", "$10/ MTok", "$0.50/ MTok"),
                "no input or output price",
                id="blank input",
            ),
            pytest.param(
                claude_row("Claude Opus 5", "$5/ MTok", "", "$6.25/ MTok", "$10/ MTok", "$0.50/ MTok"),
                "no input or output price",
                id="blank output",
            ),
            pytest.param(
                claude_row("Claude Opus 5", "$0/ MTok", "$25/ MTok", "$6.25/ MTok", "$10/ MTok", "$0.50/ MTok"),
                "zero or implausibly large",
                id="zero input",
            ),
            pytest.param(
                claude_row("Claude Opus 5", "$10001/ MTok", "$25/ MTok", "$6.25/ MTok", "$10/ MTok", "$0.50/ MTok"),
                "zero or implausibly large",
                id="one past the plausible limit",
            ),
            pytest.param(
                claude_row("Claude Opus 5", "$5/ MTok", "$25/ MTok", "$6.25/ MTok", "$10/ MTok", "$5.01/ MTok"),
                "cache hit above input",
                id="cache hit above input",
            ),
            pytest.param(
                claude_row("Claude Opus 5", "$5.0000001/ MTok", "$25/ MTok", "$6.25/ MTok", "$10/ MTok", "$0.5/ MTok"),
                "not a dollar amount",
                id="finer than a microdollar",
            ),
            pytest.param(
                claude_row("Claude Opus 5", "$5/ MTok", "$1,000/ MTok", "$6.25/ MTok", "$10/ MTok", "$0.5/ MTok"),
                "not a dollar amount",
                id="a thousands separator is not read as its first digits",
            ),
            pytest.param(
                claude_row("Claude Opus 5", "$5/ MTok", "$1234567/ MTok", "$6.25/ MTok", "$10/ MTok", "$0.5/ MTok"),
                "not a dollar amount",
                id="seven whole digits are not read as six",
            ),
        ],
    )
    def test_a_row_that_is_not_a_model_and_its_prices_refuses_the_page_without_quoting_it(
        self, row: str, reason: str
    ) -> None:
        with pytest.raises(ProviderSchemaError, match=reason) as refusal:
            read_claude_rates(row)

        assert PAGE_TEXT_MARKER not in str(refusal.value)
        assert refusal.value.__cause__ is None

    def test_a_price_at_the_plausible_limit_and_a_cache_hit_equal_to_input_are_read(self) -> None:
        row = claude_row("Claude Opus 9", "$10000/ MTok", "$10000/ MTok", "$10000/ MTok", "$1/ MTok", "$10000/ MTok")

        assert read_claude_rates(row) == {"claude-opus-9": Rates(10**10, 10**10, 10**10, 10**10)}

    def test_one_model_listed_twice_is_read_once_when_the_rates_agree_and_refused_when_they_differ(self) -> None:
        cells = ("$5/ MTok", "$25/ MTok", "$6.25/ MTok", "$10/ MTok", "$0.50/ MTok")
        again = "| Claude Opus 5 | " + " | ".join(cells) + " |\n"
        differing = "| Claude Opus 5 | $6/ MTok | $25/ MTok | $6.25/ MTok | $10/ MTok | $0.50/ MTok |\n"

        assert list(read_claude_rates(claude_row("Claude Opus 5", *cells) + again)) == ["claude-opus-5"]
        with pytest.raises(ProviderSchemaError, match=r"one model with two different rates"):
            read_claude_rates(claude_row("Claude Opus 5", *cells) + differing)


class TestOpenAIPage:
    def test_reads_every_standard_text_table_and_no_other(self) -> None:
        assert read_openai_rates(OPENAI_PAGE) == {
            "gpt-6-astra": Rates(10 * MILLION, MILLION, 12_500_000, 50 * MILLION),
            "gpt-6.1-terra": Rates(MILLION, 50_000, 1_250_000, 5 * MILLION),
            "gpt-6-cyber": Rates(20 * MILLION, 2 * MILLION, 25 * MILLION, 100 * MILLION),
            "chat-latest": Rates(5 * MILLION, 500_000, None, 30 * MILLION),
        }

    def test_a_page_with_no_standard_text_table_is_refused(self) -> None:
        batch_only = "Batch\n\n" + OPENAI_COLUMNS + "| gpt-6-astra | $5.00 | $25.00 | $0.50 | $6.25 | $12.50 |\n"

        with pytest.raises(ProviderSchemaError, match=r"lists no model"):
            read_openai_rates(batch_only)

    def test_a_standard_table_of_something_else_and_a_row_of_another_width_are_not_read(self) -> None:
        page = (
            "Standard\n\n| Tool | Pricing |\n| --- | --- |\n| Web search | $10.00 / 1k calls |\n"
            "\nStandard\n\n" + OPENAI_COLUMNS + "| gpt-6-luna | $0.10 | $0.50 | $0.01 | $0.125 | $0.25 |\n"
            "| All models |\n"
        )

        assert read_openai_rates(page) == {"gpt-6-luna": Rates(100_000, 10_000, 125_000, 500_000)}

    def test_a_standard_table_without_cache_columns_states_no_cache_price(self) -> None:
        page = "Standard\n\n| Model | Input | Output |\n| --- | --- | --- |\n| gpt-9 | $1.00 | $2.00 |\n"

        assert read_openai_rates(page) == {"gpt-9": Rates(MILLION, None, None, 2 * MILLION)}

    def test_a_row_whose_model_cell_is_not_an_id_refuses_the_page_without_quoting_it(self) -> None:
        page = "Standard\n\n" + OPENAI_COLUMNS + f"| {PAGE_TEXT_MARKER} | $5.00 | $25.00 | $0.50 | $6.25 | $12.50 |\n"

        with pytest.raises(ProviderSchemaError, match=r"names no model") as refusal:
            read_openai_rates(page)

        assert PAGE_TEXT_MARKER not in str(refusal.value)


class TestXaiPage:
    def test_reads_the_short_context_rates_and_no_cache_write(self) -> None:
        assert read_xai_rates(XAI_PAGE) == {
            "grok-4.7": Rates(2 * MILLION, 500_000, None, 6 * MILLION),
            "grok-4.5": Rates(2 * MILLION, 300_000, None, 6 * MILLION),
        }

    def test_a_row_of_another_width_is_not_read(self) -> None:
        page = (
            XAI_HEADER
            + "| [grok-4.7](https://example.test/models/grok-4.7) | 500k | $2.00 | $0.50 | $6.00 | $4.00 | $1.00 "
            "| $12.00 |\n| Long context applies above 200k tokens |\n"
        )

        assert read_xai_rates(page) == {"grok-4.7": Rates(2 * MILLION, 500_000, None, 6 * MILLION)}

    def test_a_page_without_the_text_table_is_refused(self) -> None:
        with pytest.raises(ProviderSchemaError, match=r"exactly one table"):
            read_xai_rates("### Imagine API\n\n| Model | Output |\n| --- | --- |\n| x | $0.02 / img |\n")

    def test_a_row_that_does_not_link_a_model_refuses_the_page_without_quoting_it(self) -> None:
        page = XAI_HEADER + f"| {PAGE_TEXT_MARKER} | 500k | $2.00 | $0.50 | $6.00 | $4.00 | $1.00 | $12.00 |\n"

        with pytest.raises(ProviderSchemaError, match=r"names no model") as refusal:
            read_xai_rates(page)

        assert PAGE_TEXT_MARKER not in str(refusal.value)


class TestPageBounds:
    def test_a_page_one_character_past_the_limit_is_refused_and_one_at_it_is_read(self) -> None:
        at_limit = CLAUDE_PAGE + " " * (MAX_PAGE_CHARACTERS - len(CLAUDE_PAGE))

        assert "claude-fable-5-2" in read_rates(Provider.CLAUDE, at_limit)
        with pytest.raises(ProviderSchemaError, match=r"longer than any pricing page"):
            read_rates(Provider.CLAUDE, at_limit + " ")

    def test_each_provider_is_read_with_its_own_reader(self) -> None:
        assert set(read_rates(Provider.CODEX, OPENAI_PAGE)) >= {"gpt-6-astra"}
        assert set(read_rates(Provider.GROK, XAI_PAGE)) == {"grok-4.7", "grok-4.5"}


class TestStoredFetchedRates:
    def row(self, model: str, input_rate: int, at: datetime, provider: str = "claude") -> FetchedRateRow:
        return FetchedRateRow(provider, model, input_rate, None, 12, 5 * input_rate, at)

    def test_rates_are_stored_and_read_back_in_provider_and_model_order(
        self, database: Database, now: datetime
    ) -> None:
        rows = [self.row("claude-b", 2, now), self.row("gpt-a", 3, now, "codex"), self.row("claude-a", 1, now)]
        with database.connect() as conn:
            db.store_fetched_rates(conn, rows)

        with database.read() as conn:
            assert db.list_fetched_rates(conn) == [rows[2], rows[0], rows[1]]

    def test_a_later_rate_replaces_the_stored_one_and_other_models_keep_theirs(
        self, database: Database, now: datetime
    ) -> None:
        later = now + timedelta(days=1)
        with database.connect() as conn:
            db.store_fetched_rates(conn, [self.row("claude-a", 1, now), self.row("claude-b", 2, now)])
            db.store_fetched_rates(conn, [self.row("claude-a", 9, later)])

        with database.read() as conn:
            assert db.list_fetched_rates(conn) == [self.row("claude-a", 9, later), self.row("claude-b", 2, now)]

    @pytest.mark.parametrize("rate", [-1, db.MAX_STORED_RATE + 1])
    def test_a_rate_outside_what_the_store_holds_stores_none_of_the_rows(
        self, database: Database, now: datetime, rate: int
    ) -> None:
        with database.connect() as conn, pytest.raises(ValueError, match=r"outside what the store holds"):
            db.store_fetched_rates(conn, [self.row("claude-a", 1, now), self.row("claude-b", rate, now)])

        with database.read() as conn:
            assert db.list_fetched_rates(conn) == []

    def test_the_largest_storable_rate_and_zero_are_stored(self, database: Database, now: datetime) -> None:
        edge = FetchedRateRow("claude", "claude-a", 0, 0, db.MAX_STORED_RATE, db.MAX_STORED_RATE, now)
        with database.connect() as conn:
            db.store_fetched_rates(conn, [edge])

        with database.read() as conn:
            assert db.list_fetched_rates(conn) == [edge]

    def test_clearing_removes_every_rate_and_says_how_many(self, database: Database, now: datetime) -> None:
        with database.connect() as conn:
            db.store_fetched_rates(conn, [self.row("claude-a", 1, now), self.row("claude-b", 2, now)])
            assert db.clear_fetched_rates(conn) == 2
            assert db.clear_fetched_rates(conn) == 0

        with database.read() as conn:
            assert db.list_fetched_rates(conn) == []

    def test_the_table_is_keyed_by_provider_and_dated_by_its_latest_rate(self, now: datetime) -> None:
        later = now + timedelta(days=2)
        table, fetched_on = fetched_rate_table(
            [self.row("claude-a", 1, now), self.row("gpt-a", 3, later, "codex"), self.row("x", 4, later, "unknown")]
        )

        assert table == {
            (Provider.CLAUDE, "claude-a"): Rates(1, None, 12, 5),
            (Provider.CODEX, "gpt-a"): Rates(3, None, 12, 15),
        }
        assert fetched_on == "2026-10-05"

    def test_no_stored_rate_is_an_empty_table_with_no_day(self) -> None:
        assert fetched_rate_table([]) == ({}, None)


def model_row(provider: str, model: str, **counts: int) -> dict[str, Any]:
    components = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "reasoning": 0, "requests": 0}
    components.update(counts)
    components["total"] = sum(components[name] for name in ("input", "cache_read", "cache_write", "output"))
    return {"provider": provider, "account_id": 1, "model": model, "totals": components}


class TestPricingWithFetchedRates:
    FETCHED: ClassVar[dict[tuple[Provider, str], Rates]] = {
        (Provider.CLAUDE, "claude-fable-5-2"): FABLE_5_2,
        (Provider.CLAUDE, "claude-opus-4-7"): Rates(4 * MILLION, 400_000, 5 * MILLION, 20 * MILLION),
    }

    def test_a_model_the_listed_table_lacks_is_priced_from_its_fetched_rate(self) -> None:
        row = model_row("claude", "claude-fable-5-2", input=MILLION, cache_read=MILLION, output=MILLION)

        unpriced = price_model_rows([row])[0]
        priced = price_model_rows([row], self.FETCHED)[0]

        assert (unpriced.priced_microdollars, unpriced.unpriced_tokens) == (0, 3 * MILLION)
        assert (priced.priced_microdollars, priced.unpriced_tokens) == (48_200_000, 0)
        assert priced.models[0].fetched is True

    def test_a_fetched_rate_replaces_the_listed_one_for_its_model_only(self) -> None:
        rows = [
            model_row("claude", "claude-opus-4-7", input=MILLION),
            model_row("claude", "claude-opus-4-8", input=MILLION),
        ]

        replaced, listed = price_model_rows(rows, self.FETCHED)[0].models

        assert (replaced.microdollars, replaced.fetched) == (4 * MILLION, True)
        assert (listed.microdollars, listed.fetched) == (5 * MILLION, False)

    def test_a_dated_snapshot_uses_a_family_that_only_a_fetch_has_priced(self) -> None:
        dated = model_row("claude", "claude-fable-5-2-20261101", input=MILLION)

        assert price_model_rows([dated])[0].models[0].microdollars is None
        assert price_model_rows([dated], self.FETCHED)[0].models[0].microdollars == 8 * MILLION

    def test_a_fetched_rate_belongs_to_the_vendor_whose_page_stated_it(self) -> None:
        other_vendor = model_row("codex", "claude-fable-5-2", input=MILLION)

        assert price_model_rows([other_vendor], self.FETCHED)[0].models[0].microdollars is None

    def test_a_model_with_no_rate_anywhere_is_not_marked_fetched(self) -> None:
        assert rate_in_use(Provider.CLAUDE, "not-a-model", self.FETCHED) == (None, False)
        assert rate_in_use(Provider.CLAUDE, "claude-fable-5-2") == (None, False)

    def test_a_fetched_blank_cell_leaves_a_row_that_used_it_unpriced(self) -> None:
        fetched = {(Provider.CODEX, "gpt-6-astra"): Rates(10 * MILLION, None, None, 50 * MILLION)}
        row = model_row("codex", "gpt-6-astra", input=MILLION, cache_read=1)

        assert price_model_rows([row])[0].models[0].microdollars == 10 * MILLION + 1
        assert price_model_rows([row], fetched)[0].models[0].microdollars is None

    def test_the_note_names_the_day_rates_were_fetched_only_when_some_were(self) -> None:
        assert cost_note(None) == COST_NOTE
        assert cost_note("2026-11-10").startswith(COST_NOTE)
        assert "marked fetched" in cost_note("2026-11-10")
        assert "2026-11-10" in cost_note("2026-11-10")

    def test_text_that_is_not_a_number_is_refused_as_a_price(self) -> None:
        with pytest.raises(ValueError, match=r"^a published price is not a non-negative amount$") as refusal:
            _usd("eight")

        assert refusal.value.__cause__ is None

    def test_the_cost_block_says_where_each_rate_came_from(self) -> None:
        report = {
            "range": "24h",
            "series": [],
            "models": [
                model_row("claude", "claude-fable-5-2", input=MILLION),
                model_row("claude", "claude-opus-4-8", input=MILLION),
                model_row("claude", "not-a-model", input=3),
            ],
            "provider_totals": [],
        }

        costs = _token_report_json(report, self.FETCHED, "2026-11-10")["costs"]

        assert costs["fetched_on"] == "2026-11-10"
        assert costs["note"] == cost_note("2026-11-10")
        assert [model["rate_source"] for model in costs["accounts"][0]["models"]] == ["fetched", "listed", None]
        assert _token_report_json(report)["costs"]["fetched_on"] is None


class Firecrawl:
    """Stands in for Firecrawl: answers each scrape with the markdown, or the response, set for its URL."""

    def __init__(self, pages: Mapping[str, str | httpx2.Response]) -> None:
        self.pages = dict(pages)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        answer = self.pages[json.loads(request.content)["url"]]
        if isinstance(answer, httpx2.Response):
            return answer
        return httpx2.Response(200, json={"success": True, "data": {"markdown": answer, "metadata": {"title": "x"}}})

    def client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self))


def stored_key(_item: KeychainItem) -> str:
    return FIRECRAWL_KEY


def stored_models(database: Database) -> set[tuple[str, str]]:
    with database.read() as conn:
        return {(row.provider, row.model) for row in db.list_fetched_rates(conn)}


class TestUpdatingRates:
    def test_every_page_is_fetched_fresh_with_the_key_and_its_rates_are_stored(
        self, database: Database, now: datetime
    ) -> None:
        firecrawl = Firecrawl(PAGES)
        asked: list[KeychainItem] = []

        def read_key(item: KeychainItem) -> str:
            asked.append(item)
            return FIRECRAWL_KEY

        report = update_rates(database, firecrawl.client(), read_key, lambda: now)

        assert [item.service for item in asked] == [FIRECRAWL_KEYCHAIN_SERVICE]
        assert [str(request.url) for request in firecrawl.requests] == [FIRECRAWL_SCRAPE_URL] * 3
        assert {request.headers["authorization"] for request in firecrawl.requests} == {f"Bearer {FIRECRAWL_KEY}"}
        assert [json.loads(request.content) for request in firecrawl.requests] == [
            {"url": url, "formats": ["markdown"], "onlyMainContent": True, "maxAge": 0} for url in PAGES
        ]
        assert report.checked_at == now
        assert report.has_failures is False
        assert [(page.provider, page.models_read) for page in report.pages] == [
            (Provider.CLAUDE, 4),
            (Provider.CODEX, 4),
            (Provider.GROK, 2),
        ]
        assert stored_models(database) >= {
            ("claude", "claude-fable-5-2"),
            ("codex", "gpt-6-cyber"),
            ("grok", "grok-4.5"),
        }
        assert len(stored_models(database)) == 10

    def test_only_a_model_whose_rate_in_use_differs_is_reported_as_changed(
        self, database: Database, now: datetime
    ) -> None:
        report = update_rates(database, Firecrawl(PAGES).client(), stored_key, lambda: now)

        claude, codex, grok = report.pages
        assert claude.changes == (RateChange("claude-fable-5-2", None, FABLE_5_2),)
        assert [(change.model, change.previous) for change in codex.changes] == [
            ("gpt-6-cyber", None),
            ("gpt-6.1-terra", None),
        ]
        assert grok.changes == ()

    def test_a_second_update_compares_with_the_fetched_rate_and_reports_a_new_price(
        self, database: Database, now: datetime
    ) -> None:
        update_rates(database, Firecrawl(PAGES).client(), stored_key, lambda: now)
        repriced = {
            **PAGES,
            RATE_PAGES[Provider.CLAUDE].url: CLAUDE_PAGE.replace(
                "| $8/ MTok | $40/ MTok |", "| $7/ MTok | $40/ MTok |"
            ),
        }
        later = now + timedelta(days=1)

        report = update_rates(database, Firecrawl(repriced).client(), stored_key, lambda: later)

        cheaper = Rates(7 * MILLION, 200_000, 10 * MILLION, 40 * MILLION)
        assert report.pages[0].changes == (RateChange("claude-fable-5-2", FABLE_5_2, cheaper),)
        assert report.pages[1].changes == ()
        with database.read() as conn:
            table, fetched_on = fetched_rate_table(db.list_fetched_rates(conn))
        assert table[(Provider.CLAUDE, "claude-fable-5-2")] == cheaper
        assert fetched_on == "2026-10-04"

    def test_a_model_the_page_no_longer_lists_keeps_its_stored_rate(self, database: Database, now: datetime) -> None:
        update_rates(database, Firecrawl(PAGES).client(), stored_key, lambda: now)
        without_terra = {
            **PAGES,
            RATE_PAGES[Provider.CODEX].url: OPENAI_PAGE.replace(
                "| gpt-6.1-terra | $1.00 | $5.00 | $0.05 | $1.25 | $2.50 |\n", ""
            ),
        }

        report = update_rates(database, Firecrawl(without_terra).client(), stored_key, lambda: now)

        assert report.pages[1].models_read == 3
        assert ("codex", "gpt-6.1-terra") in stored_models(database)

    @pytest.mark.parametrize(
        ("answer", "error_type", "message"),
        [
            pytest.param(httpx2.Response(402), "ProviderRequestError", "Firecrawl returned HTTP 402", id="no credits"),
            pytest.param(
                httpx2.Response(401),
                "CredentialExpiredError",
                "Firecrawl rejected the stored token (HTTP 401)",
                id="key rejected",
            ),
            pytest.param(
                httpx2.Response(200, json={"success": False}),
                "ProviderSchemaError",
                "Firecrawl answered without the page",
                id="reported no success",
            ),
            pytest.param(
                httpx2.Response(200, json={"success": True}),
                "ProviderSchemaError",
                "Firecrawl answered without the page",
                id="success with no page",
            ),
            pytest.param(
                httpx2.Response(200, json={"success": True, "data": {"markdown": 7}}),
                "ProviderSchemaError",
                "the Firecrawl answer has an unexpected shape: data.markdown: Input should be a valid string",
                id="markdown is not text",
            ),
            pytest.param(
                f"# Pricing\n\n{PAGE_TEXT_MARKER}\n",
                "ProviderSchemaError",
                "a pricing page does not have exactly one table with the expected rate columns",
                id="layout changed",
            ),
        ],
    )
    def test_a_page_that_cannot_be_fetched_or_read_is_reported_and_the_others_are_still_stored(
        self, database: Database, now: datetime, answer: str | httpx2.Response, error_type: str, message: str
    ) -> None:
        pages: dict[str, str | httpx2.Response] = {**PAGES, RATE_PAGES[Provider.GROK].url: answer}

        report = update_rates(database, Firecrawl(pages).client(), stored_key, lambda: now)

        assert report.has_failures is True
        assert report.pages[2] == PageOutcome(Provider.GROK, 0, (), error_type, message)
        assert [page.error for page in report.pages[:2]] == [None, None]
        assert {provider for provider, _model in stored_models(database)} == {"claude", "codex"}

    def test_a_page_that_is_refused_leaves_its_vendors_stored_rates_as_they_were(
        self, database: Database, now: datetime
    ) -> None:
        update_rates(database, Firecrawl(PAGES).client(), stored_key, lambda: now)
        before = stored_models(database)
        broken = {**PAGES, RATE_PAGES[Provider.CLAUDE].url: CLAUDE_PAGE.replace("$8/ MTok", "eight dollars")}

        report = update_rates(database, Firecrawl(broken).client(), stored_key, lambda: now)

        assert report.pages[0].error_type == "ProviderSchemaError"
        assert stored_models(database) == before

    def test_without_a_stored_key_nothing_is_requested_and_the_error_names_the_command(
        self, database: Database, now: datetime
    ) -> None:
        firecrawl = Firecrawl(PAGES)

        def no_key(item: KeychainItem) -> str:
            raise CredentialUnavailableError(f"no keychain item {item}")

        with pytest.raises(CredentialUnavailableError) as refusal:
            update_rates(database, firecrawl.client(), no_key, lambda: now)

        assert str(refusal.value) == (
            "no keychain item 'usagedash-firecrawl' under this macOS account's name. "
            "Store a Firecrawl API key with `usagedash rates set-key`."
        )
        assert refusal.value.__cause__ is None
        assert firecrawl.requests == []
        assert stored_models(database) == set()


class TestFirecrawlKey:
    def test_the_item_is_this_dashboards_service_under_the_user_databases_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("USER", "someone-else")

        item = firecrawl_keychain_item()

        assert item.service == "usagedash-firecrawl"
        assert item.account != "someone-else"
        assert item.account != ""

    def test_an_account_missing_from_the_user_database_is_a_config_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(_uid: int) -> Any:
            raise KeyError("getpwuid(): uid not found")

        monkeypatch.setattr(rate_updates.pwd, "getpwuid", missing)

        with pytest.raises(ConfigError, match=r"no entry in the user database") as refusal:
            firecrawl_keychain_item()

        assert refusal.value.__cause__ is None

    def test_the_keychain_tool_is_left_to_ask_for_the_key_so_it_is_never_an_argument(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        commands: list[list[str]] = []

        def run(command: list[str], **options: Any) -> subprocess.CompletedProcess[bytes]:
            commands.append(command)
            assert options == {"check": False}
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(rate_updates.subprocess, "run", run)

        store_key_interactively()

        item = firecrawl_keychain_item()
        assert commands == [
            ["/usr/bin/security", "add-generic-password", "-U", "-a", item.account, "-s", item.service, "-w"]
        ]

    def test_a_keychain_tool_that_fails_or_cannot_run_is_reported_by_kind_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            rate_updates.subprocess, "run", lambda command, **_options: subprocess.CompletedProcess(command, 45)
        )
        with pytest.raises(CredentialUnavailableError, match=r"^the keychain tool exited 45 without storing the key$"):
            store_key_interactively()

        def cannot_run(_command: list[str], **_options: Any) -> Any:
            raise PermissionError(f"{PAGE_TEXT_MARKER}")

        monkeypatch.setattr(rate_updates.subprocess, "run", cannot_run)
        with pytest.raises(
            CredentialUnavailableError, match=r"^cannot run the keychain tool: PermissionError$"
        ) as refusal:
            store_key_interactively()
        assert refusal.value.__cause__ is None


class TestPostingJson:
    def test_the_body_is_sent_as_json_and_the_answers_object_is_returned(self) -> None:
        seen: list[httpx2.Request] = []

        def answer(request: httpx2.Request) -> httpx2.Response:
            seen.append(request)
            return httpx2.Response(200, json={"ok": True})

        with httpx2.Client(transport=httpx2.MockTransport(answer)) as client:
            body = post_json_object(client, "https://example.test/x", {"X-Probe": "1"}, {"a": [1]}, service="Svc")

        assert body == {"ok": True}
        assert (seen[0].method, json.loads(seen[0].content), seen[0].headers["x-probe"]) == ("POST", {"a": [1]}, "1")
        assert seen[0].headers["content-type"] == "application/json"

    def test_a_rejected_key_a_failed_request_and_an_unsendable_header_are_typed_errors_without_a_cause(self) -> None:
        def fail(_request: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError(f"cannot reach {PAGE_TEXT_MARKER}")

        with httpx2.Client(transport=httpx2.MockTransport(lambda _: httpx2.Response(403))) as client:
            with pytest.raises(CredentialExpiredError, match=r"^Svc rejected the stored token \(HTTP 403\)$"):
                post_json_object(client, "https://example.test/x", {}, {}, service="Svc")
            with pytest.raises(ProviderRequestError, match=r"^Svc request not sent: Authorization holds") as unsent:
                post_json_object(client, "https://example.test/x", {"Authorization": "Bearer a\nb"}, {}, service="Svc")
        assert "a\nb" not in str(unsent.value)

        with (
            httpx2.Client(transport=httpx2.MockTransport(fail)) as client,
            pytest.raises(ProviderRequestError, match=r"^Svc request failed: ConnectError$") as failed,
        ):
            post_json_object(client, "https://example.test/x", {}, {}, service="Svc")
        assert failed.value.__cause__ is None


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return load_settings(env={"USAGEDASH_DATA_DIR": str(tmp_path / "data")}, home=tmp_path)


def a_report(now: datetime) -> RateUpdateReport:
    return RateUpdateReport(
        now,
        (
            PageOutcome(Provider.CLAUDE, 21, (RateChange("claude-fable-5-2", None, FABLE_5_2),)),
            PageOutcome(
                Provider.CODEX,
                8,
                (RateChange("gpt-6-astra", Rates(10 * MILLION, MILLION, None, 50 * MILLION), FABLE_5_2),),
            ),
            PageOutcome(Provider.GROK, 0, (), "ProviderRequestError", "Firecrawl returned HTTP 402"),
        ),
    )


class TestRateUpdateRoute:
    @contextlib.contextmanager
    def browser(self, settings: Settings, database: Database, now: datetime, updater: Any) -> Iterator[TestClient]:
        app = create_app(
            settings, database, None, lambda: now, access_token=ACCESS_TOKEN, run_collector=False, rate_updater=updater
        )
        with TestClient(app, base_url=BASE_URL, follow_redirects=False) as test_client:
            yield test_client

    @pytest.fixture
    def updates(self) -> list[int]:
        return []

    @pytest.fixture
    def client(self, settings: Settings, database: Database, now: datetime, updates: list[int]) -> Iterator[TestClient]:
        def updater() -> RateUpdateReport:
            updates.append(1)
            return a_report(now)

        with self.browser(settings, database, now, updater) as test_client:
            test_client.headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"
            yield test_client

    def test_the_update_answers_with_each_page_and_its_changes_as_decimal_text(
        self, client: TestClient, updates: list[int]
    ) -> None:
        response = client.post("/api/rates/update", headers=PAGE_HEADERS)

        assert response.status_code == 200
        assert updates == [1]
        fable = {"input": "8000000", "cache_read": "200000", "cache_write": "10000000", "output": "40000000"}
        assert response.json() == {
            "checked_at": "2026-10-03T14:00:00+00:00",
            "pages": [
                {
                    "provider": "claude",
                    "display_name": "Claude",
                    "models_read": 21,
                    "error": None,
                    "changes": [{"model": "claude-fable-5-2", "previous": None, "fetched": fable}],
                },
                {
                    "provider": "codex",
                    "display_name": "Codex",
                    "models_read": 8,
                    "error": None,
                    "changes": [
                        {
                            "model": "gpt-6-astra",
                            "previous": {
                                "input": "10000000",
                                "cache_read": "1000000",
                                "cache_write": None,
                                "output": "50000000",
                            },
                            "fetched": fable,
                        }
                    ],
                },
                {
                    "provider": "grok",
                    "display_name": "Grok",
                    "models_read": 0,
                    "error": "Firecrawl returned HTTP 402",
                    "changes": [],
                },
            ],
        }

    def test_the_update_is_refused_without_the_token_and_when_it_did_not_come_from_the_page(
        self, client: TestClient, updates: list[int]
    ) -> None:
        assert client.post("/api/rates/update").status_code == 403
        del client.headers["Authorization"]
        assert client.post("/api/rates/update", headers=PAGE_HEADERS).status_code == 401
        assert client.get("/api/rates/update").status_code == 405
        assert updates == []

    def test_a_process_that_cannot_update_rates_says_so(
        self, settings: Settings, database: Database, now: datetime
    ) -> None:
        with self.browser(settings, database, now, None) as test_client:
            response = test_client.post(
                "/api/rates/update", headers={**PAGE_HEADERS, "Authorization": f"Bearer {ACCESS_TOKEN}"}
            )

            assert (response.status_code, response.json()) == (503, {"detail": "This process cannot update rates."})

    def test_a_missing_key_is_answered_with_the_command_that_stores_one_and_the_next_update_can_run(
        self, settings: Settings, database: Database, now: datetime
    ) -> None:
        outcomes: list[Any] = [CredentialUnavailableError("no key. Store one with `usagedash rates set-key`."), None]

        def updater() -> RateUpdateReport:
            outcome = outcomes.pop(0)
            if outcome is not None:
                raise outcome
            return a_report(now)

        with self.browser(settings, database, now, updater) as test_client:
            headers = {**PAGE_HEADERS, "Authorization": f"Bearer {ACCESS_TOKEN}"}
            refused = test_client.post("/api/rates/update", headers=headers)

            assert refused.status_code == 409
            assert refused.json() == {"detail": "no key. Store one with `usagedash rates set-key`."}
            assert test_client.post("/api/rates/update", headers=headers).status_code == 200

    def test_a_second_update_is_refused_while_one_is_running(
        self, settings: Settings, database: Database, now: datetime
    ) -> None:
        started, release = threading.Event(), threading.Event()

        def updater() -> RateUpdateReport:
            started.set()
            assert release.wait(timeout=10)
            return a_report(now)

        with self.browser(settings, database, now, updater) as test_client:
            headers = {**PAGE_HEADERS, "Authorization": f"Bearer {ACCESS_TOKEN}"}
            statuses: list[int] = []
            first = threading.Thread(
                target=lambda: statuses.append(test_client.post("/api/rates/update", headers=headers).status_code)
            )
            first.start()
            try:
                assert started.wait(timeout=10)
                second = test_client.post("/api/rates/update", headers=headers)
            finally:
                release.set()
                first.join(timeout=10)

            assert (second.status_code, second.json()) == (409, {"detail": "A rate update is already running."})
            assert statuses == [200]
            assert test_client.post("/api/rates/update", headers=headers).status_code == 200

    def test_the_token_report_is_priced_with_the_stored_fetched_rates(
        self, client: TestClient, database: Database, now: datetime
    ) -> None:
        assert client.get("/api/tokens").json()["costs"]["fetched_on"] is None
        with database.connect() as conn:
            db.store_fetched_rates(conn, [FetchedRateRow("claude", "claude-fable-5-2", 8, 2, 10, 40, now)])

        costs = client.get("/api/tokens").json()["costs"]

        assert costs["fetched_on"] == "2026-10-03"
        assert costs["note"] == cost_note("2026-10-03")


class TestRatesCommand:
    @pytest.fixture
    def data_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        data = tmp_path / "data"
        monkeypatch.setenv("USAGEDASH_DATA_DIR", str(data))
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("USAGEDASH_CONFIG", raising=False)
        monkeypatch.delenv("USAGEDASH_PORT", raising=False)
        return data

    def test_update_prints_what_each_page_changed_and_succeeds_when_every_page_was_read(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report = a_report(datetime(2026, 11, 10, tzinfo=UTC))
        read_pages = RateUpdateReport(report.checked_at, report.pages[:2])
        monkeypatch.setattr(cli, "update_rates", lambda *_arguments: read_pages)

        assert cli.main(["rates", "update"]) == 0

        assert capsys.readouterr().out.splitlines() == [
            "Claude: 21 model(s) read, 1 changed",
            "  claude-fable-5-2: input $8, cache read $0.2, cache write $10, output $40",
            "    had no rate",
            "Codex: 8 model(s) read, 1 changed",
            "  gpt-6-astra: input $8, cache read $0.2, cache write $10, output $40",
            "    was input $10, cache read $1, cache write -, output $50",
        ]

    def test_update_fails_when_a_page_was_not_read_and_says_which(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "update_rates", lambda *_arguments: a_report(datetime(2026, 11, 10, tzinfo=UTC)))

        assert cli.main(["rates", "update"]) == 1

        assert capsys.readouterr().out.splitlines()[-1] == "Grok: not updated: Firecrawl returned HTTP 402"

    def test_update_without_a_stored_key_is_one_line_naming_the_command(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def no_key(_item: KeychainItem) -> str:
            raise CredentialUnavailableError("no keychain item")

        monkeypatch.setattr(cli, "read_keychain_secret", no_key)

        assert cli.main(["rates", "update"]) == 1

        assert capsys.readouterr().err == (
            "error: no keychain item. Store a Firecrawl API key with `usagedash rates set-key`.\n"
        )

    def test_reset_removes_the_fetched_rates(
        self, data_dir: Path, capsys: pytest.CaptureFixture[str], now: datetime
    ) -> None:
        database = Database(data_dir / "usage.db")
        database.initialize()
        with database.connect() as conn:
            db.store_fetched_rates(conn, [FetchedRateRow("claude", "claude-fable-5-2", 8, 2, 10, 40, now)])

        assert cli.main(["rates", "reset"]) == 0

        assert capsys.readouterr().out == (
            "Removed 1 fetched rate(s). Every model is priced from the listed table again.\n"
        )
        with database.read() as conn:
            assert db.list_fetched_rates(conn) == []

    def test_set_key_hands_over_to_the_keychain_tool_and_reports_its_failure(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        calls: list[int] = []
        monkeypatch.setattr(cli, "store_key_interactively", lambda: calls.append(1))

        assert cli.main(["rates", "set-key"]) == 0
        assert calls == [1]
        assert capsys.readouterr().out.splitlines()[-1].startswith("Stored.")

        def refuse() -> None:
            raise CredentialUnavailableError("the keychain tool exited 45 without storing the key")

        monkeypatch.setattr(cli, "store_key_interactively", refuse)

        assert cli.main(["rates", "set-key"]) == 1
        assert capsys.readouterr().err == "error: the keychain tool exited 45 without storing the key\n"
