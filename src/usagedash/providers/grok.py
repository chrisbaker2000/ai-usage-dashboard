"""Grok Build source.

Who is signed in comes from the home's auth file. Usage comes from a reader handed to this source, which
asks the Grok CLI for the signed-in account's billing, so every reading is the provider's own answer for
the credential in the home.

The CLI also writes the billing it fetches to its log after every turn, and reading that would cost
nothing. It is not used. A log record does not say whose it is, and attributing one to a login by when
it was written cannot be made exact: a token refresh and a new sign-in both write a new credential, a
switch to another account and back leaves no trace between two looks, and a request begun under one
login can be logged after the next one starts.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Self

from pydantic import BeforeValidator, Field, TypeAdapter, ValidationError, model_validator

from usagedash.credentials import read_json_object
from usagedash.errors import CredentialExpiredError, CredentialUnavailableError, ProviderSchemaError
from usagedash.models import MAX_WINDOW_SECONDS, AccountIdentity, Fact, Provider, QuotaReading, QuotaWindow
from usagedash.paths import directory_identity
from usagedash.providers.base import QuotaSource, ensure_same_login
from usagedash.timeutil import parse_iso
from usagedash.validation import (
    AccessToken,
    DecimalAmount,
    LenientModel,
    OptionalIdentifier,
    UsedPercent,
)

AUTH_FILE_NAME = "auth.json"
ACCESS_TOKEN_FIELD = "key"  # noqa: S105 - the name of the auth file's token field, not a token
ENTRY_NAME_SEPARATOR = "::"
"""What a login's name in the auth file holds, between its issuer and its id (grok 1.0.46)."""
TOKEN_EXPIRY_MARGIN = timedelta(minutes=5)
_ACCESS_TOKEN = TypeAdapter(AccessToken)
_NEVER_CREATED = datetime.min.replace(tzinfo=UTC)

_PERIOD_NAMES: dict[str, tuple[str, str]] = {
    "USAGE_PERIOD_TYPE_WEEKLY": ("weekly", "Weekly credits"),
    "USAGE_PERIOD_TYPE_MONTHLY": ("monthly", "Monthly credits"),
    "USAGE_PERIOD_TYPE_DAILY": ("daily", "Daily credits"),
}
_DEFAULT_PERIOD_NAME = ("credits", "Credits")
NOTHING_USED_PERCENT = 0.0
UNSTATED_USAGE_FACT = Fact("Credits used", "Not stated by Grok, which leaves the figure out at zero")


def _exact_instant(value: object) -> datetime:
    """Parse a time only when a datetime can hold it exactly.

    Digits past a microsecond are dropped by a datetime. Two credential times a hundred nanoseconds
    apart would then be one credential, and a refresh during the read would look like no change. A
    billing period one nanosecond past the cap, or one nanosecond short of a second, would look like
    a usable length, and an answer that states no usage would be read as nothing used.
    """
    if not isinstance(value, str):
        raise ValueError("must be ISO-8601 text")
    try:
        return parse_iso(value, exact=True)
    except ValueError:
        raise ValueError("must be an ISO-8601 timestamp with an offset, expressible in UTC") from None


ExactInstant = Annotated[datetime, BeforeValidator(_exact_instant)]
CredentialInstant = ExactInstant


class AuthEntry(LenientModel):
    """One login in a Grok auth file: who it is, when its credential was made and expires, and whether it has a token.

    The access token itself is never kept, only whether there is one (``has_access_token``): an entry
    can end up in an error or a log, and the token must not go with it.
    """

    user_id: OptionalIdentifier = None
    principal_id: OptionalIdentifier = None
    email: OptionalIdentifier = None
    create_time: CredentialInstant | None = None
    expires_at: CredentialInstant | None = None
    has_access_token: bool = False

    @model_validator(mode="before")
    @classmethod
    def _note_whether_there_is_an_access_token(cls, data: object) -> object:
        """Replace the token (``key``) with whether it is one: text a request could carry as it is.

        The rules are those of every stored token (``AccessToken``). A value they refuse is no token the
        CLI could send, and the CLI would replace it using the refresh token.
        """
        if not isinstance(data, dict):
            return data
        token = data.get(ACCESS_TOKEN_FIELD)
        kept = {name: value for name, value in data.items() if name not in {ACCESS_TOKEN_FIELD, "has_access_token"}}
        return kept | {"has_access_token": _is_access_token(token)}


def _is_access_token(value: object) -> bool:
    try:
        _ACCESS_TOKEN.validate_python(value, strict=True)
    except ValidationError:
        return False
    return True


class _Period(LenientModel):
    type: str | None = None
    start: ExactInstant | None = None
    end: ExactInstant | None = None


class _Amount(LenientModel):
    val: DecimalAmount | None = None


class CreditsConfig(LenientModel):
    """Grok's billing answer for the signed-in account.

    Grok leaves ``creditUsagePercent`` out when nothing has been used in the period: its billing answer
    is proto3 JSON, which omits a number that is zero. Seen live on 2026-10-03 with grok 1.0.46, for an
    account two days into its week, in an answer that was otherwise complete. An answer without the
    figure is therefore accepted, but only when it states a complete current period, which is what
    says the answer is about a metered period at all: the period names its type and has a start and an
    end that make a usable length (see ``_is_complete``). Without either it says nothing about usage,
    and zero would present an account nothing is known about as the one with the most room.
    """

    credit_usage_percent: UsedPercent | None = Field(default=None, alias="creditUsagePercent")
    current_period: _Period | None = Field(default=None, alias="currentPeriod")
    on_demand_cap: _Amount | None = Field(default=None, alias="onDemandCap")
    on_demand_used: _Amount | None = Field(default=None, alias="onDemandUsed")
    prepaid_balance: _Amount | None = Field(default=None, alias="prepaidBalance")

    @model_validator(mode="after")
    def _states_usage_or_a_complete_period(self) -> Self:
        if self.credit_usage_percent is None and not _is_complete(self.current_period):
            raise ValueError("states neither a usage percentage nor a complete current period")
        return self


def windows_and_facts(config: CreditsConfig) -> tuple[tuple[QuotaWindow, ...], tuple[Fact, ...]]:
    """Convert a Grok billing config into its single credit window and any display facts.

    A config that leaves the usage figure out is read as nothing used (see ``CreditsConfig``), and a
    fact beside the window says the figure was not stated.
    """
    period = config.current_period
    key, label = _PERIOD_NAMES.get(period.type or "", _DEFAULT_PERIOD_NAME) if period else _DEFAULT_PERIOD_NAME
    window_seconds = _period_seconds(period)
    stated = config.credit_usage_percent
    window = QuotaWindow(
        key=key,
        label=label,
        used_percent=NOTHING_USED_PERCENT if stated is None else stated,
        resets_at=period.end if period else None,
        window_seconds=window_seconds,
    )
    facts = [UNSTATED_USAGE_FACT] if stated is None else []
    on_demand_cap = _amount(config.on_demand_cap)
    if on_demand_cap > 0:
        facts.append(Fact("On-demand usage", _on_demand_usage(config.on_demand_used, on_demand_cap)))
    prepaid = _amount(config.prepaid_balance)
    if prepaid > 0:
        facts.append(Fact("Prepaid balance", f"{prepaid:,.0f}"))
    return (window,), tuple(facts)


def _is_complete(period: _Period | None) -> bool:
    """Report whether a period names its type and has a start and an end that make a usable length.

    A blank type names nothing, and bounds that are equal, reversed, or further apart than any real
    period are no period: an answer like that is not one this code can take a zero from.
    """
    return period is not None and bool((period.type or "").strip()) and _period_seconds(period) is not None


def _period_seconds(period: _Period | None) -> int | None:
    """Return how long a billing period lasts, or None when its ends are missing or make no usable length.

    The cap is applied to the whole duration. Truncating to seconds first would accept a period up to
    one second past ``MAX_WINDOW_SECONDS``.
    """
    if period is None or period.start is None or period.end is None:
        return None
    duration = period.end - period.start
    if not timedelta(seconds=1) <= duration <= timedelta(seconds=MAX_WINDOW_SECONDS):
        return None
    return int(duration.total_seconds())


def _on_demand_usage(used: _Amount | None, cap: float) -> str:
    """Say how much of the on-demand cap is used, or that the answer does not say.

    Why an unstated amount is not read as zero: every billing answer the Grok CLI logged on this Mac
    (179 on 2026-10-04) states each amount's ``val``, zero included, so a missing one is not proto3
    leaving out a zero, and shown as 0 it would claim a figure the provider never gave.
    """
    if used is None or used.val is None:
        return f"not stated, of {cap:,.0f} cap"
    return f"{used.val:,.0f} of {cap:,.0f} cap"


def _amount(amount: _Amount | None) -> float:
    return amount.val if amount is not None and amount.val is not None else 0.0


class GrokSource:
    """The working Grok login: a home directory whose ``auth.json`` says who is signed in.

    ``reader`` reads the usage. It runs the Grok CLI in this home, and is asked only while the login's
    access token is comfortably unexpired (see ``_token_is_valid``), so that the CLI it starts has
    nothing to refresh. Once the token has expired there is no reading until Grok is next used, which
    is also when its usage next changes. It is not asked either for a credential that cannot be told
    from another of the same account (see ``credential_key``).
    """

    def __init__(self, home: Path, reader: QuotaSource):
        self._home = directory_identity(home)
        self._reader = reader

    @property
    def source_id(self) -> str:
        return f"grok:{self._home}"

    @property
    def provider(self) -> Provider:
        return Provider.GROK

    def current_identity(self) -> AccountIdentity | None:
        entry = current_auth_entry(self._home)
        return None if entry is None else identity_from_auth_entry(entry, plan=None)

    def read_quota(self, now: datetime) -> QuotaReading:
        auth = self._signed_in_login()
        if not auth.has_access_token:
            # Why the reader is not asked: the CLI it starts would get a token with the refresh token,
            # which rotates, and that would change the credential an open Grok session holds.
            raise CredentialUnavailableError(
                "Grok login has no usable access token; signing in to Grok again replaces it"
            )
        if not _token_is_valid(auth, now):
            raise CredentialExpiredError("Grok access token has expired; the Grok CLI refreshes it on its next run")
        credential = credential_key(auth)
        reading = self._reader.read_quota(now)
        # Why the login is read again: the reader was asked because this credential's token was valid,
        # and it answers for whoever is signed in when it runs. A reading is returned only if the same
        # credential was in place before and after it.
        ensure_same_login(credential, credential_key(self._signed_in_login()), "Grok")
        return reading

    def _signed_in_login(self) -> AuthEntry:
        """Return the login in the home's auth file as it is at this moment.

        Raises:
            CredentialUnavailableError: if nobody is signed in.
        """
        auth = current_auth_entry(self._home)
        if auth is None:
            raise CredentialUnavailableError("Grok is not signed in")
        return auth


def login_identity(auth: AuthEntry) -> tuple[str | None, str | None, str | None]:
    """Say whose login an auth entry is: everything the auth file holds about the person."""
    return (auth.user_id, auth.principal_id, auth.email)


def credential_key(
    auth: AuthEntry,
) -> tuple[str | None, str | None, str | None, datetime, datetime | None, bool]:
    """Identify one credential: whose it is, when it was created and expires, and whether it holds a usable token.

    Every sign-in and every token refresh writes a new credential with a new creation time (seen live
    on grok 1.0.46, where the creation time is the expiry less the token's six hours). Two reads with
    the same key are therefore the same credential. A different key is another account, a new sign-in,
    or a refresh, and the auth file does not say which. Whether the token is usable is part of the key:
    the reader is asked because it is, and a token removed or spoilt while it ran leaves the login with
    nothing a request could carry, though every other field is unchanged.

    Raises:
        ProviderSchemaError: if the entry does not say when it was created. Two credentials of one
            account would then have the same key, and a switch to another account and back would
            look like no change at all.
    """
    if auth.create_time is None:
        raise ProviderSchemaError("Grok login does not say when its credential was created")
    return (*login_identity(auth), auth.create_time, auth.expires_at, auth.has_access_token)


def _token_is_valid(auth: AuthEntry, now: datetime) -> bool:
    """Report whether the working login's access token is still comfortably unexpired.

    Why: the reader runs the Grok CLI in the working home. With an expired token that run would
    refresh it, and refresh tokens rotate, so the dashboard would be the one changing the credential an
    open Grok session holds. With a valid token the CLI has nothing to refresh. An unknown expiry is
    treated as expired.
    """
    return auth.expires_at is not None and auth.expires_at - now > TOKEN_EXPIRY_MARGIN


def current_auth_entry(home: Path) -> AuthEntry | None:
    """Return the most recently minted login in a Grok home's auth file, or None when there is none.

    Only identity fields are kept; the token fields in the file are discarded during validation. Every
    member that is an object is read as a login, and so is every member named like one
    (``ENTRY_NAME_SEPARATOR``), which must then be an object: skipped, a file whose only login was
    damaged read as nobody signed in. Any other member, such as a format version, is not a login.

    The newest login is the one in force, including one that names nobody. An older account is then
    not signed in: the newer credential replaced it. A login with no creation time cannot be ordered
    against another login. When the file holds more than one and any of them omits it, the file is
    refused: treating the omission as the oldest time would read a dated login while the undated one
    was the one in force. One undated login is still returned, and the credential check refuses to
    read it. Two logins with the same creation time are refused too: member order would otherwise
    decide which account is signed in.

    A creation time or expiry finer than a microsecond cannot be understood. A datetime would drop
    the rest, and two such times would then be one credential.

    Raises:
        ProviderSchemaError: if the file exists but cannot be understood, if more than one login is
            present and one of them does not say when it was created, or if more than one login
            shares the newest creation time.
    """
    try:
        raw = read_json_object(home / AUTH_FILE_NAME)
    except CredentialUnavailableError:
        return None
    entries = []
    for name, value in raw.items():
        if not isinstance(value, dict) and ENTRY_NAME_SEPARATOR not in name:
            continue
        try:
            entry = AuthEntry.model_validate(value)
        except ValidationError:
            # Why no cause is chained: the validation error holds the auth file entry it was given.
            raise ProviderSchemaError("Grok auth file has an unexpected shape") from None
        if _external_id(entry) is not None or ENTRY_NAME_SEPARATOR in name:
            entries.append(entry)
    if not entries:
        return None
    if len(entries) > 1 and any(entry.create_time is None for entry in entries):
        raise ProviderSchemaError("Grok login does not say when its credential was created")
    newest_at = max(_created_at(entry) for entry in entries)
    tied = [entry for entry in entries if _created_at(entry) == newest_at]
    if len(tied) != 1:
        raise ProviderSchemaError("Grok auth file does not identify one newest login")
    newest = tied[0]
    return newest if _external_id(newest) is not None else None


def _created_at(entry: AuthEntry) -> datetime:
    return entry.create_time or _NEVER_CREATED


def _external_id(entry: AuthEntry) -> str | None:
    return entry.user_id or entry.principal_id or entry.email


def identity_from_auth_entry(entry: AuthEntry, plan: str | None) -> AccountIdentity:
    """Build the account identity for a Grok login.

    Raises:
        CredentialUnavailableError: if the entry names no user.
    """
    external_id = _external_id(entry)
    if external_id is None:
        raise CredentialUnavailableError("Grok auth entry does not identify a user")
    return AccountIdentity(
        provider=Provider.GROK,
        external_id=external_id,
        default_label=entry.email or "Grok account",
        email=entry.email,
        plan=plan,
    )
