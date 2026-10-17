"""Codex (ChatGPT plan) source.

Quota comes from the ChatGPT backend usage endpoint the Codex CLI itself polls. The account is keyed on the
ChatGPT workspace id, because one login can belong to several workspaces with separate limits.
"""

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, TypeGuard

import httpx2
from pydantic import ValidationError

from usagedash.credentials import decode_jwt_claims, read_json_object
from usagedash.errors import CredentialExpiredError, CredentialUnavailableError, NotAJwtError, ProviderSchemaError
from usagedash.http import fetch_json_object
from usagedash.models import FULL_PERCENT, AccountIdentity, Fact, Provider, QuotaReading, QuotaWindow
from usagedash.paths import directory_identity
from usagedash.providers.base import duration_key_and_label, humanize, with_unique_keys
from usagedash.timeutil import from_epoch_seconds
from usagedash.validation import (
    AccessToken,
    DecimalAmount,
    EpochSeconds,
    LenientModel,
    OptionalIdentifier,
    UsedPercent,
    WindowSeconds,
    schema_boundary,
)

logger = logging.getLogger(__name__)

USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
SERVICE_NAME = "Codex usage API"
OPENAI_AUTH_CLAIM = "https://api.openai.com/auth"
AUTH_FILE_NAME = "auth.json"
REPORTED_LIMIT_KEY = "reported_limit"
REPORTED_LIMIT_LABEL = "Limit reached (reported by Codex)"


class _Tokens(LenientModel):
    access_token: AccessToken
    account_id: OptionalIdentifier = None
    id_token: str | None = None


CHATGPT_AUTH_MODE = "chatgpt"
"""The ``auth_mode`` of a Codex login that uses its ChatGPT tokens (codex 0.160.0, read live 2026-10-04)."""


class _AuthFile(LenientModel):
    """A Codex home's ``auth.json``. ``auth_mode`` says which login the CLI uses; a file in API-key mode can
    still hold the tokens of a ChatGPT login it no longer uses."""

    auth_mode: str | None = None
    tokens: _Tokens | None = None


class _Window(LenientModel):
    used_percent: UsedPercent
    limit_window_seconds: WindowSeconds | None = None
    reset_at: EpochSeconds | None = None


class _RateLimit(LenientModel):
    primary_window: _Window | None = None
    secondary_window: _Window | None = None
    allowed: bool | None = None
    limit_reached: bool | None = None

    @property
    def is_blocked(self) -> bool:
        return self.limit_reached is True or self.allowed is False


class _Credits(LenientModel):
    has_credits: bool | None = None
    unlimited: bool | None = None
    balance: DecimalAmount | None = None


class _Usage(LenientModel):
    account_id: OptionalIdentifier = None
    email: str | None = None
    plan_type: str | None = None
    rate_limit: _RateLimit | None = None
    code_review_rate_limit: _RateLimit | None = None
    additional_rate_limits: list[Any] | None = None
    credits: _Credits | None = None


def identity_from_auth(tokens: _Tokens) -> AccountIdentity:
    """Build the identity from the auth file alone.

    Raises:
        CredentialUnavailableError: if neither the token set nor its ID token names a workspace.
    """
    claims = _id_token_claims(tokens.id_token)
    auth_claim = claims.get(OPENAI_AUTH_CLAIM)
    auth_details: dict[str, Any] = auth_claim if isinstance(auth_claim, dict) else {}
    account_id = tokens.account_id or _optional_string(auth_details.get("chatgpt_account_id"))
    if account_id is None:
        raise CredentialUnavailableError("Codex auth file does not identify a ChatGPT workspace")
    return workspace_identity(
        account_id=account_id,
        email=_optional_string(claims.get("email")),
        plan_type=_optional_string(auth_details.get("chatgpt_plan_type")),
    )


def _id_token_claims(id_token: str | None) -> dict[str, Any]:
    if id_token is None:
        return {}
    try:
        return decode_jwt_claims(id_token)
    except ProviderSchemaError:
        logger.warning("Codex ID token could not be decoded; continuing without email and plan")
        return {}


def _optional_string(value: object) -> str | None:
    """Return a string with its surrounding whitespace removed, or None for anything else or a blank one."""
    if not isinstance(value, str):
        return None
    return value.strip() or None


def workspace_identity(*, account_id: str, email: str | None, plan_type: str | None) -> AccountIdentity:
    return AccountIdentity(
        provider=Provider.CODEX,
        external_id=account_id,
        default_label=email or "Codex account",
        email=email,
        plan=humanize(plan_type) if plan_type else None,
    )


def parse_usage(payload: dict[str, Any]) -> tuple[_Usage, tuple[QuotaWindow, ...], tuple[Fact, ...]]:
    """Convert a usage payload into quota windows and display facts.

    Raises:
        ProviderSchemaError: if the payload fails validation or reports no window of its main limit.
    """
    with schema_boundary(f"{SERVICE_NAME} payload"):
        usage = _Usage.model_validate(payload)
        windows = require_main_limit(
            _limited_windows(usage.rate_limit, key_prefix="", label_prefix="", blocking=True),
            SERVICE_NAME,
        )
        windows += _limited_windows(
            usage.code_review_rate_limit,
            key_prefix="code_review:",
            label_prefix="Code review · ",
            blocking=False,
        )
        unreadable = 0
        for index, entry in enumerate(usage.additional_rate_limits or []):
            extra = _additional_windows(entry, index)
            if extra is None:
                unreadable += 1
            else:
                windows += extra
        facts = _facts(usage)
        if unreadable:
            facts.append(unreadable_limits_fact(unreadable))
        return usage, with_unique_keys(windows), tuple(facts)


def require_main_limit(main_windows: list[QuotaWindow], service: str) -> list[QuotaWindow]:
    """Return the main limit's windows, refusing a reading that has none.

    The main limit is the only one that blocks the account. Without a window of it, the code-review and
    per-feature windows alone would show an account with room, whatever its main limit says. Checked
    before anything else is added, so that no other window can stand in for it.

    Raises:
        ProviderSchemaError: if there are none.
    """
    if not main_windows:
        raise ProviderSchemaError(f"{service} reported no window of its main limit")
    return main_windows


def unreadable_limits_fact(count: int) -> Fact:
    """Say on the account's card that Codex reported limits the dashboard could not read.

    Why a fact and not a failed reading: an extra limit is a per-feature pool that never blocks the
    account, and failing the whole reading over one would hide the main limit, which is what decides
    whether to switch. Why not only a log line: the card would then look complete without that limit.
    """
    noun = "limit" if count == 1 else "limits"
    return Fact("Limits not shown", f"{count} extra {noun} with an unexpected shape; the log says which")


def with_reported_limit(
    windows: list[QuotaWindow], *, limit_reached: bool, blocking: bool, key: str, label: str
) -> list[QuotaWindow]:
    """Add the window a reached flag implies when no listed window of that pool is already full.

    Codex can report a limit reached while every window it lists still shows room. A reached main limit
    blocks the account. A reached code-review or per-feature pool is shown full and does not block:
    that pool stops one feature, and a blocking window would mark ordinary use as out of quota. The
    added window has no reset time, because Codex did not say which of the pool's limits was hit.
    """
    if not limit_reached or any(window.used_percent >= FULL_PERCENT for window in windows):
        return windows
    reported = QuotaWindow(
        key=key,
        label=label,
        used_percent=FULL_PERCENT,
        resets_at=None,
        window_seconds=None,
        blocking=blocking,
    )
    return [*windows, reported]


def _limited_windows(
    rate_limit: _RateLimit | None, *, key_prefix: str, label_prefix: str, blocking: bool
) -> list[QuotaWindow]:
    """Return one pool's windows, including a non-listed full window when Codex reports the pool reached."""
    return with_reported_limit(
        _windows(rate_limit, key_prefix=key_prefix, label_prefix=label_prefix, blocking=blocking),
        limit_reached=rate_limit is not None and rate_limit.is_blocked,
        blocking=blocking,
        key=f"{key_prefix}{REPORTED_LIMIT_KEY}",
        label=f"{label_prefix}{REPORTED_LIMIT_LABEL}",
    )


def _windows(rate_limit: _RateLimit | None, *, key_prefix: str, label_prefix: str, blocking: bool) -> list[QuotaWindow]:
    """Convert one rate limit's windows. Only the main Codex limit is ``blocking``: a full code-review or
    per-feature pool stops that feature, not ordinary Codex use, so it must not mark the account as out of quota."""
    if rate_limit is None:
        return []
    windows = []
    for window in (rate_limit.primary_window, rate_limit.secondary_window):
        if window is None:
            continue
        key, label = duration_key_and_label(window.limit_window_seconds)
        windows.append(
            QuotaWindow(
                key=f"{key_prefix}{key}",
                label=f"{label_prefix}{label}",
                used_percent=window.used_percent,
                resets_at=None if window.reset_at is None else from_epoch_seconds(window.reset_at),
                window_seconds=window.limit_window_seconds,
                blocking=blocking,
            )
        )
    return windows


def names_a_meter(meter_id: object) -> TypeGuard[str]:
    """Report whether a value is a meter id. One that is blank, or is not text, names nothing."""
    return isinstance(meter_id, str) and bool(meter_id.strip())


def extra_limit_prefixes(meter_id: str, limit_name: object) -> tuple[str, str]:
    """Return the key prefix and the label prefix of one extra per-feature limit.

    The key is the limit's meter id exactly as Codex gives it: ``metered_feature`` in the usage payload,
    which the Codex CLI passes on unchanged as ``limitId`` (``backend-client/src/client.rs`` on the main
    branch of codex-rs, read 2026-10-03).
    The meter id is what identifies the limit. It stays the same when the display name changes, two
    limits may share a display name, and an account read both ways must have each limit stored under
    one key. The display name is only shown.

    An extra limit is a display detail, and both sources read one the same way: an entry that names no
    meter (``names_a_meter``), or whose windows fail validation, is logged and left out, the card says
    how many were (``unreadable_limits_fact``), and it never costs the reading its main limit.
    """
    return f"{meter_id}:", f"{humanize(_optional_string(limit_name) or meter_id)} · "


def _additional_windows(entry: object, index: int) -> list[QuotaWindow] | None:
    """Read one extra per-feature limit, or return None when it is not the shape Codex itself requires.

    An entry that is not is logged by position and left out; the caller says so on the card
    (``unreadable_limits_fact``). That includes a limit with no window this reads: every window field
    null, or only fields this does not know, leaves nothing to show for it.
    """
    if not isinstance(entry, dict) or not isinstance(entry.get("rate_limit"), dict):
        logger.warning("Leaving out additional Codex rate limit %d with unrecognized shape", index)
        return None
    meter_id = entry.get("metered_feature")
    if not names_a_meter(meter_id):
        logger.warning("Leaving out additional Codex rate limit %d that names no meter", index)
        return None
    try:
        rate_limit = _RateLimit.model_validate(entry["rate_limit"])
    except ValidationError:
        logger.warning("Leaving out additional Codex rate limit %d that failed validation", index)
        return None
    key_prefix, label_prefix = extra_limit_prefixes(meter_id, entry.get("limit_name"))
    windows = _limited_windows(
        rate_limit,
        key_prefix=key_prefix,
        label_prefix=label_prefix,
        blocking=False,
    )
    if not windows:
        logger.warning("Leaving out additional Codex rate limit %d with no window", index)
        return None
    return windows


def _facts(usage: _Usage) -> list[Fact]:
    facts = []
    credits = usage.credits
    if credits is not None:
        if credits.unlimited:
            facts.append(Fact("Credits", "Unlimited"))
        elif credits.balance is not None:
            facts.append(Fact("Credits", f"{credits.balance:,.0f} remaining"))
    return facts


class CodexSource:
    """A Codex home directory holding a ChatGPT login in ``auth.json``."""

    def __init__(self, home: Path, client: httpx2.Client):
        self._home = directory_identity(home)
        self._client = client

    @property
    def source_id(self) -> str:
        return f"codex:{self._home}"

    @property
    def home(self) -> Path:
        """The directory this source reads, identified once when the source was built."""
        return self._home

    @property
    def provider(self) -> Provider:
        return Provider.CODEX

    def current_identity(self) -> AccountIdentity | None:
        try:
            tokens = self._read_tokens()
            return identity_from_auth(tokens)
        except CredentialUnavailableError:
            return None

    def read_quota(self, now: datetime) -> QuotaReading:
        """Read the workspace's windows with the CLI's live access token, never sending one that has expired.

        Raises:
            CredentialUnavailableError: if Codex is not signed in with a usable ChatGPT token.
            CredentialExpiredError: if the access token's expiry has passed.
            ProviderSchemaError: if the token is a JWT whose expiry cannot be read, or the answer has an
                unexpected shape.
            ProviderRequestError: if the endpoint cannot be reached or refuses the request.
        """
        tokens = self._read_tokens()
        local_identity = identity_from_auth(tokens)
        if _is_expired(tokens.access_token, now):
            raise CredentialExpiredError("Codex access token has expired; the Codex CLI refreshes it on its next run")
        headers = {
            "Authorization": f"Bearer {tokens.access_token}",
            "ChatGPT-Account-Id": local_identity.external_id,
            "Accept": "application/json",
        }
        usage, windows, facts = parse_usage(fetch_json_object(self._client, USAGE_URL, headers, service=SERVICE_NAME))
        identity = workspace_identity(
            account_id=usage.account_id or local_identity.external_id,
            email=usage.email or local_identity.email,
            plan_type=usage.plan_type,
        )
        if usage.plan_type is None and local_identity.plan is not None:
            identity = AccountIdentity(
                provider=identity.provider,
                external_id=identity.external_id,
                default_label=identity.default_label,
                email=identity.email,
                plan=local_identity.plan,
            )
        return QuotaReading(identity=identity, windows=windows, observed_at=now, source="api", facts=facts)

    def _read_tokens(self) -> _Tokens:
        raw = read_json_object(self._home / AUTH_FILE_NAME)
        try:
            auth = _AuthFile.model_validate(raw)
        except ValidationError:
            # Why no cause is chained: the validation error holds the auth file it was given.
            raise CredentialUnavailableError("Codex auth file holds no usable ChatGPT token") from None
        # Why the mode is checked and not only the tokens: in another mode the CLI does not use the
        # ChatGPT tokens the file still keeps, so their workspace is not who is logged in, and reading
        # its quota would send a token the CLI has stopped using.
        is_chatgpt_login = auth.auth_mode is None or auth.auth_mode == CHATGPT_AUTH_MODE
        if auth.tokens is None or not is_chatgpt_login:
            raise CredentialUnavailableError("Codex is not signed in with a ChatGPT account")
        return auth.tokens


def _is_expired(access_token: str, now: datetime) -> bool:
    """Report whether the access token's ``exp`` claim is past, refusing a JWT whose expiry cannot be read.

    A token that is not a JWT, or a JWT with no ``exp`` claim, states no expiry, so the provider judges
    it. A JWT whose claims cannot be decoded (not base64url, not UTF-8, not a JSON object, an integer
    past Python's digit limit, nesting past the recursion limit), or whose ``exp`` is not a number, may
    be expired: it is refused rather than sent.

    Raises:
        ProviderSchemaError: if the token is a JWT whose claims or expiry cannot be read.
    """
    try:
        claims = decode_jwt_claims(access_token)
    except NotAJwtError:
        return False
    except ProviderSchemaError as error:
        raise ProviderSchemaError(f"Codex access token was not sent: its expiry cannot be read ({error})") from None
    if "exp" not in claims:
        return False
    expires: object = claims["exp"]
    is_nan = isinstance(expires, float) and math.isnan(expires)
    if isinstance(expires, bool) or not isinstance(expires, int | float) or is_nan:
        raise ProviderSchemaError("Codex access token was not sent: its expiry is not a number")
    # Why the two numbers are compared directly: building a datetime from the claim fails for a time
    # before 1970 or past year 9999, and a failure would be read as "expiry unknown". A negative expiry
    # is long past, and an infinite or impossibly large one is simply not past.
    return expires <= now.timestamp()
