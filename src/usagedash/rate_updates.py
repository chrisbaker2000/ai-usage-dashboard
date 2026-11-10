"""Fetch each vendor's pricing page through Firecrawl and store the rates it states.

This is the one place the dashboard asks a host that is not a vendor's own, and it does so only
when asked to: by the page's button or ``usagedash rates update``. Firecrawl returns each pricing
page as markdown; ``rate_pages`` reads the rates out of it. Nothing about an account is sent: the
request holds a public URL and the Firecrawl key.

The key is a generic-password item in the login keychain, written by ``usagedash rates set-key``
through ``security`` itself, which prompts for it. It is never in the config file, an argument, or
the environment, and it is read only for the request that sends it.

One page that cannot be fetched or read does not stop the others. A page is stored whole or not at
all, and a model a page no longer lists keeps the rate it has.
"""

import logging
import os
import pwd
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import httpx2

from usagedash import db
from usagedash.credentials import SECURITY_BINARY, KeychainItem, KeychainReader
from usagedash.db import Database, FetchedRateRow
from usagedash.errors import ConfigError, CredentialUnavailableError, ProviderSchemaError, UsageDashError
from usagedash.http import post_json_object
from usagedash.models import Provider
from usagedash.pricing import Rates, RateTable, rate_in_use
from usagedash.rate_pages import RATE_PAGES, read_rates
from usagedash.timeutil import Clock, in_utc
from usagedash.validation import LenientModel, schema_boundary

logger = logging.getLogger(__name__)

FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"
FIRECRAWL_SERVICE = "Firecrawl"
FIRECRAWL_KEYCHAIN_SERVICE = "usagedash-firecrawl"
FIRECRAWL_TIMEOUT_SECONDS = 90.0
"""How long one page may take. Firecrawl renders the page in a browser before it answers."""
SET_KEY_COMMAND = "usagedash rates set-key"


class _ScrapedPage(LenientModel):
    markdown: str


class _ScrapeAnswer(LenientModel):
    success: bool
    data: _ScrapedPage | None = None


@dataclass(frozen=True, slots=True)
class RateChange:
    """One model whose rate in use changed. ``previous`` is None for a model that had no rate."""

    model: str
    previous: Rates | None
    fetched: Rates


@dataclass(frozen=True, slots=True)
class PageOutcome:
    """What became of one vendor's page: the models read and what changed, or why it was not read."""

    provider: Provider
    models_read: int
    changes: tuple[RateChange, ...]
    error_type: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RateUpdateReport:
    checked_at: datetime
    pages: tuple[PageOutcome, ...]

    @property
    def has_failures(self) -> bool:
        return any(page.error_type is not None for page in self.pages)


def firecrawl_keychain_item() -> KeychainItem:
    """Name the keychain item that holds the Firecrawl key: this dashboard's service, this account's name.

    Raises:
        ConfigError: if the user database has no entry for this account.
    """
    try:
        account = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        raise ConfigError(
            "this account has no entry in the user database, so its keychain item cannot be named"
        ) from None
    return KeychainItem(service=FIRECRAWL_KEYCHAIN_SERVICE, account=account)


def store_key_interactively() -> None:
    """Have ``security`` ask for the Firecrawl key and store it, replacing one already stored.

    Why ``security`` asks: the key then never passes through this process, an argument list, or the
    shell's history. ``-w`` as the last argument is what makes it prompt.

    Raises:
        ConfigError: if this account has no entry in the user database.
        CredentialUnavailableError: if ``security`` cannot be run or does not store the key.
    """
    item = firecrawl_keychain_item()
    try:
        completed = subprocess.run(
            [SECURITY_BINARY, "add-generic-password", "-U", "-a", item.account, "-s", item.service, "-w"],
            check=False,
        )
    except OSError as error:
        raise CredentialUnavailableError(f"cannot run the keychain tool: {type(error).__name__}") from None
    if completed.returncode != 0:
        raise CredentialUnavailableError(f"the keychain tool exited {completed.returncode} without storing the key")


def fetched_rate_table(rows: Sequence[FetchedRateRow]) -> tuple[RateTable, str | None]:
    """Return the stored fetched rates as a rate table, and the day of the latest one, if there is any."""
    table: dict[tuple[Provider, str], Rates] = {}
    latest: datetime | None = None
    for row in rows:
        try:
            provider = Provider(row.provider)
        except ValueError:
            continue
        table[(provider, row.model)] = Rates(row.input_rate, row.cache_read_rate, row.cache_write_rate, row.output_rate)
        latest = row.fetched_at if latest is None else max(latest, row.fetched_at)
    return table, None if latest is None else in_utc(latest).date().isoformat()


def _page_markdown(client: httpx2.Client, key: str, url: str) -> str:
    """Ask Firecrawl for one page as markdown, fetched now and not from its cache.

    Raises:
        CredentialExpiredError: if Firecrawl rejects the key.
        ProviderRequestError: if the request fails or is answered with another non-success status.
        ProviderSchemaError: if the answer is not the shape a scrape returns, or reports no success.
    """
    payload = post_json_object(
        client,
        FIRECRAWL_SCRAPE_URL,
        {"Authorization": f"Bearer {key}"},
        {"url": url, "formats": ["markdown"], "onlyMainContent": True, "maxAge": 0},
        service=FIRECRAWL_SERVICE,
        timeout_seconds=FIRECRAWL_TIMEOUT_SECONDS,
    )
    with schema_boundary("the Firecrawl answer"):
        answer = _ScrapeAnswer.model_validate(payload)
    if not answer.success or answer.data is None:
        raise ProviderSchemaError("Firecrawl answered without the page")
    return answer.data.markdown


def _changes(provider: Provider, read: Mapping[str, Rates], in_use: RateTable) -> tuple[RateChange, ...]:
    changes = []
    for model in sorted(read):
        previous, _ = rate_in_use(provider, model, in_use)
        if previous != read[model]:
            changes.append(RateChange(model, previous, read[model]))
    return tuple(changes)


def _update_page(
    database: Database, client: httpx2.Client, key: str, provider: Provider, checked_at: datetime
) -> PageOutcome:
    read = read_rates(provider, _page_markdown(client, key, RATE_PAGES[provider].url))
    rows = [
        FetchedRateRow(
            provider.value, model, rates.input, rates.cache_read, rates.cache_write, rates.output, checked_at
        )
        for model, rates in read.items()
        if rates.input is not None and rates.output is not None
    ]
    with database.connect() as conn:
        in_use, _ = fetched_rate_table(db.list_fetched_rates(conn))
        changes = _changes(provider, read, in_use)
        db.store_fetched_rates(conn, rows)
    return PageOutcome(provider, len(rows), changes)


def update_rates(database: Database, client: httpx2.Client, read_key: KeychainReader, clock: Clock) -> RateUpdateReport:
    """Fetch every vendor's pricing page and store the rates each states.

    A page that cannot be fetched or read is reported in its outcome and leaves its vendor's stored
    rates as they were; the other pages are still read.

    Raises:
        ConfigError: if this account has no entry in the user database.
        CredentialUnavailableError: if no Firecrawl key is stored or the keychain cannot be read. The
            message names the command that stores one.
    """
    try:
        key = read_key(firecrawl_keychain_item())
    except CredentialUnavailableError as error:
        raise CredentialUnavailableError(f"{error}. Store a Firecrawl API key with `{SET_KEY_COMMAND}`.") from None
    checked_at = clock()
    pages = []
    for provider in RATE_PAGES:
        try:
            pages.append(_update_page(database, client, key, provider, checked_at))
        except UsageDashError as error:
            # Why the message is kept: a typed error's message holds no text the page or Firecrawl supplied.
            logger.warning("rates for %s were not updated: %s", provider.value, type(error).__name__)
            pages.append(PageOutcome(provider, 0, (), type(error).__name__, str(error)))
    return RateUpdateReport(checked_at, tuple(pages))


RateUpdater = Callable[[], RateUpdateReport]
"""Runs one update with everything it needs already bound, as the API and the command line hold it."""
