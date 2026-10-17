"""Claude Code subscription source.

Quota comes from the same OAuth usage endpoint the CLI's ``/usage`` screen reads. The endpoint is not a
documented public API, so the payload is validated leniently: unknown limit kinds are still shown, and a
payload with no recognizable limits is a schema error instead of an empty success.
"""

import hashlib
import logging
import os
import pwd
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
from pydantic import Field, ValidationError

from usagedash.config import ClaudeSourceConfig
from usagedash.credentials import (
    KeychainItem,
    KeychainReader,
    parse_json_object,
    read_json_object,
    read_keychain_secret,
)
from usagedash.errors import CredentialExpiredError, CredentialUnavailableError, ProviderSchemaError
from usagedash.http import fetch_json_object
from usagedash.models import AccountIdentity, Fact, Provider, QuotaReading, QuotaWindow
from usagedash.paths import directory_identity, same_directory
from usagedash.providers.base import (
    FIVE_HOURS_SECONDS,
    WEEK_SECONDS,
    humanize,
    slugify,
    with_unique_keys,
)
from usagedash.timeutil import from_epoch_seconds, round_to_minute
from usagedash.validation import (
    MILLISECONDS_PER_SECOND,
    AccessToken,
    EpochMilliseconds,
    Identifier,
    Instant,
    LenientModel,
    UsedPercent,
    is_finite_number,
    schema_boundary,
    summarize_validation_error,
)

logger = logging.getLogger(__name__)

USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
PROFILE_URL = "https://api.anthropic.com/api/oauth/profile"
OAUTH_BETA_HEADER = "oauth-2025-04-20"
SERVICE_NAME = "Claude usage API"
MAX_CURRENCY_DECIMAL_PLACES = 6
CREDIT_POOL_AMOUNT_FIELDS = ("limit_dollars", "used_dollars")

KEYCHAIN_ACCOUNT_FALLBACK = "claude-code-user"
_KEYCHAIN_ACCOUNT_NAME = re.compile(r"[A-Za-z0-9._-]+")

PERSONAL_ORGANIZATION_TYPES = frozenset({"claude_max", "claude_pro"})
PERSONAL_ORGANIZATION_SUFFIX = "'s Organization"

_PLAN_NAMES: dict[str, str] = {
    "default_claude_max_20x": "Max 20x",
    "default_claude_max_5x": "Max 5x",
    "claude_max": "Max",
    "max": "Max",
    "claude_pro": "Pro",
    "pro": "Pro",
    "claude_team": "Team",
    "team": "Team",
    "claude_enterprise": "Enterprise",
    "enterprise": "Enterprise",
}

_LEGACY_WINDOWS: tuple[tuple[str, str, str, int], ...] = (
    ("five_hour", "five_hour", "5-hour session", FIVE_HOURS_SECONDS),
    ("seven_day", "weekly", "Weekly · all models", WEEK_SECONDS),
    ("seven_day_opus", "weekly:opus", "Weekly · Opus", WEEK_SECONDS),
    ("seven_day_sonnet", "weekly:sonnet", "Weekly · Sonnet", WEEK_SECONDS),
)


class _OAuthCredential(LenientModel):
    access_token: AccessToken = Field(alias="accessToken")
    expires_at_ms: EpochMilliseconds | None = Field(default=None, alias="expiresAt")
    subscription_type: str | None = Field(default=None, alias="subscriptionType")
    rate_limit_tier: str | None = Field(default=None, alias="rateLimitTier")


class _KeychainBlob(LenientModel):
    claude_ai_oauth: _OAuthCredential | None = Field(default=None, alias="claudeAiOauth")


class _OAuthAccount(LenientModel):
    organization_uuid: Identifier = Field(alias="organizationUuid")
    organization_name: str | None = Field(default=None, alias="organizationName")
    email_address: str | None = Field(default=None, alias="emailAddress")


class _AccountFile(LenientModel):
    oauth_account: _OAuthAccount | None = Field(default=None, alias="oauthAccount")


class _ProfileAccount(LenientModel):
    email: str | None = None


class _ProfileOrganization(LenientModel):
    uuid: Identifier
    name: str | None = None
    organization_type: str | None = None
    rate_limit_tier: str | None = None


class _Profile(LenientModel):
    account: _ProfileAccount | None = None
    organization: _ProfileOrganization


class _LimitModel(LenientModel):
    display_name: str | None = None


class _LimitScope(LenientModel):
    model: _LimitModel | None = None


class _Limit(LenientModel):
    kind: str = Field(min_length=1)
    percent: UsedPercent
    resets_at: Instant | None = None
    scope: _LimitScope | None = None


class _LegacyWindow(LenientModel):
    """One fixed-field limit.

    Absent or null means that limit was not reported. An object is a report of it, so the percentage
    is required: dropping the object would hide a limit that may already be full.
    """

    utilization: UsedPercent
    resets_at: Instant | None = None


class _ModelScopedWindow(LenientModel):
    """One model-scoped weekly limit as ``model_scoped`` lists it.

    It is the limit a ``weekly_scoped`` entry of ``limits`` describes for the same model: every answer
    Claude Code gave on this Mac on 2026-10-04 that carried both had the same percentage and reset in
    each (4 of 4). Answers often leave ``limits`` out (9 of 14 that morning) while this is there, and
    the legacy per-model fields were empty in all of them.
    """

    display_name: str | None = None
    utilization: UsedPercent
    resets_at: Instant | None = None


class _ExtraUsage(LenientModel):
    """Overage settings. Amounts are in minor units; ``decimal_places`` says how many digits are minor.

    No field has a default that stands in for what the payload left out. Whether overage is on is
    required, because "Off" is a statement about what the user can be charged; and the scale has no
    assumed value, because amounts shown at the wrong scale are wrong by a factor of a hundred.
    """

    is_enabled: bool
    monthly_limit: float | None = Field(default=None, ge=0)
    used_credits: float | None = Field(default=None, ge=0)
    decimal_places: int | None = Field(default=None, ge=0, le=MAX_CURRENCY_DECIMAL_PLACES)


class _Usage(LenientModel):
    limits: list[_Limit] | None = None
    five_hour: _LegacyWindow | None = None
    seven_day: _LegacyWindow | None = None
    seven_day_opus: _LegacyWindow | None = None
    seven_day_sonnet: _LegacyWindow | None = None
    model_scoped: list[_ModelScopedWindow] | None = None
    extra_usage: Any = None


def keychain_account() -> str:
    """Return the account name Claude Code files this macOS account's keychain items under.

    Claude Code looks its login up by service and account. It names the account ``$USER``, else the
    user database's name, and ``claude-code-user`` when that name has a character outside
    ``[A-Za-z0-9._-]`` (2.1.294 and 2.1.295). Here the user database decides and ``USER`` is ignored:
    a sign-in from a shell whose ``USER`` was something else filed the login under that name, and every
    later run under the real name found nothing (seen 2026-10-08, ``acct="unknown"``). Every Claude
    child is given this name as ``USER`` (``claude_cli.cli_environment``), so a sign-in and each later
    read name the same item.
    """
    try:
        name = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return KEYCHAIN_ACCOUNT_FALLBACK
    return name if _KEYCHAIN_ACCOUNT_NAME.fullmatch(name) else KEYCHAIN_ACCOUNT_FALLBACK


def plan_name(rate_limit_tier: str | None, organization_type: str | None, subscription_type: str | None) -> str | None:
    """Return a readable plan name from the most specific identifier available."""
    for candidate in (rate_limit_tier, organization_type, subscription_type):
        if candidate is None or candidate == "":
            continue
        return _PLAN_NAMES.get(candidate, humanize(candidate.removeprefix("default_claude_").removeprefix("claude_")))
    return None


def build_identity(
    *,
    organization_uuid: str,
    organization_name: str | None,
    organization_type: str | None,
    email: str | None,
    plan: str | None,
) -> AccountIdentity:
    """Key the account on the organization, because one email can belong to a personal and a team plan."""
    is_personal = organization_type in PERSONAL_ORGANIZATION_TYPES or (
        organization_name is not None and organization_name.endswith(PERSONAL_ORGANIZATION_SUFFIX)
    )
    label = "Personal" if is_personal else (organization_name or "Claude account")
    return AccountIdentity(
        provider=Provider.CLAUDE, external_id=organization_uuid, default_label=label, email=email, plan=plan
    )


def parse_usage(payload: dict[str, Any]) -> tuple[tuple[QuotaWindow, ...], tuple[Fact, ...]]:
    """Convert a usage payload into quota windows and display facts.

    The limits are what the reading is for, so anything wrong with them is a schema error. The facts
    are display details: one that cannot be read is left out, and never costs the reading its limits.

    Raises:
        ProviderSchemaError: if the limits cannot be interpreted or the payload contains no recognizable limit.
    """
    with schema_boundary(f"{SERVICE_NAME} payload"):
        usage = _Usage.model_validate(payload)
        windows = _windows_from_limits(usage.limits) if usage.limits else _windows_from_legacy_fields(usage)
        if not windows:
            raise ProviderSchemaError(f"{SERVICE_NAME} payload contained no recognizable limits")
        return with_unique_keys(windows), tuple(_facts(usage, payload))


def _weekly_scoped(scope_name: str | None) -> tuple[str, str, int]:
    """Key, label, and length of a model-scoped weekly limit, the same whichever part of the answer gave it."""
    scope = scope_name or "scoped"
    return f"weekly:{slugify(scope)}", f"Weekly · {scope}", WEEK_SECONDS


def _reset_time(moment: datetime | None) -> datetime | None:
    return None if moment is None else round_to_minute(moment)


def _windows_from_limits(limits: list[_Limit]) -> list[QuotaWindow]:
    windows = []
    for limit in limits:
        scope_name = limit.scope.model.display_name if limit.scope and limit.scope.model else None
        if limit.kind == "session":
            key, label, seconds = "five_hour", "5-hour session", FIVE_HOURS_SECONDS
        elif limit.kind == "weekly_all":
            key, label, seconds = "weekly", "Weekly · all models", WEEK_SECONDS
        elif limit.kind == "weekly_scoped":
            # Why this stays blocking: a full model-scoped limit stops all work on that model, which for
            # someone whose default is that model is exactly when they need to switch accounts.
            key, label, seconds = _weekly_scoped(scope_name)
        else:
            suffix = f" · {scope_name}" if scope_name else ""
            key = limit.kind if scope_name is None else f"{limit.kind}:{slugify(scope_name)}"
            label, seconds = f"{humanize(limit.kind)}{suffix}", None
        windows.append(
            QuotaWindow(
                key=key,
                label=label,
                used_percent=limit.percent,
                resets_at=_reset_time(limit.resets_at),
                window_seconds=seconds,
            )
        )
    return windows


def _windows_from_legacy_fields(usage: _Usage) -> list[QuotaWindow]:
    """Read an answer with no ``limits`` list: the fixed fields, then each model-scoped limit.

    A model-scoped limit is blocking, as it is when ``limits`` lists it, and is the one window that
    shows an account out of quota on its default model; without it such an account read as having
    room. An entry is left out only when a fixed field states that same limit: the same key, percentage,
    and reset (``seven_day_opus`` beside an Opus entry). Every other entry is kept, a repeated or
    colliding key included, and ``with_unique_keys`` tells them apart; dropping one by its key alone
    could drop the limit that is full.
    """
    windows = []
    for field_name, key, label, seconds in _LEGACY_WINDOWS:
        window: _LegacyWindow | None = getattr(usage, field_name)
        if window is None:
            continue
        windows.append(
            QuotaWindow(
                key=key,
                label=label,
                used_percent=window.utilization,
                resets_at=_reset_time(window.resets_at),
                window_seconds=seconds,
            )
        )
    stated_by_fixed_fields = {(window.key, window.used_percent, window.resets_at) for window in windows}
    for scoped in usage.model_scoped or []:
        key, label, seconds = _weekly_scoped(scoped.display_name)
        resets_at = _reset_time(scoped.resets_at)
        if (key, scoped.utilization, resets_at) in stated_by_fixed_fields:
            continue
        windows.append(
            QuotaWindow(
                key=key,
                label=label,
                used_percent=scoped.utilization,
                resets_at=resets_at,
                window_seconds=seconds,
            )
        )
    return windows


def _facts(usage: _Usage, payload: dict[str, Any]) -> list[Fact]:
    facts = []
    extra_usage = _extra_usage_fact(usage.extra_usage)
    if extra_usage is not None:
        facts.append(extra_usage)
    for name, value in payload.items():
        if not isinstance(value, dict) or not _is_credit_pool(value):
            continue
        credit_pool = _credit_pool_fact(name, value)
        if credit_pool is not None:
            facts.append(credit_pool)
    return facts


def _is_credit_pool(entry: dict[str, Any]) -> bool:
    """Report whether a payload entry states a dollar amount, which is what makes it a credit pool.

    Every limit entry the API sends carries both dollar fields. An ordinary limit, such as the 5-hour
    session, states them as null, and that is not a pool that could not be read.
    """
    return any(entry.get(field) is not None for field in CREDIT_POOL_AMOUNT_FIELDS)


def _credit_pool_fact(name: str, pool: dict[str, Any]) -> Fact | None:
    """Describe a credit pool, or None when it does not state both amounts as usable numbers.

    A pool that is left out is logged by the amount it lacks. The pool's name is the provider's text
    and is not logged.
    """
    unusable = [field for field in CREDIT_POOL_AMOUNT_FIELDS if not is_finite_number(pool.get(field))]
    if unusable:
        logger.warning("Leaving out a Claude credit pool without a usable %s", " or ".join(unusable))
        return None
    limit, used = pool["limit_dollars"], pool["used_dollars"]
    return Fact(f"Credit pool · {name}", f"${used:,.2f} of ${limit:,.2f} used", _pool_reset(pool))


def _extra_usage_fact(raw: object) -> Fact | None:
    """Describe the overage setting, or None when the payload has none or it cannot be read.

    "Off" is returned only when the payload says the setting is off. A setting that does not say, or
    that is on without stating both amounts and their scale, is left out and logged.
    """
    if raw is None:
        return None
    try:
        extra = _ExtraUsage.model_validate(raw)
    except ValidationError as error:
        logger.warning("Leaving out Claude extra usage with an unexpected shape: %s", summarize_validation_error(error))
        return None
    if not extra.is_enabled:
        return Fact("Extra usage", "Off")
    if extra.monthly_limit is None or extra.used_credits is None or extra.decimal_places is None:
        # Why this is left out and not shown as "Off": the setting is on, and saying otherwise would tell
        # the user that paid overage cannot happen.
        logger.warning("Leaving out Claude extra usage that is enabled but does not state its amounts and their scale")
        return None
    scale = 10**extra.decimal_places
    return Fact("Extra usage", f"${extra.used_credits / scale:,.2f} of ${extra.monthly_limit / scale:,.2f} used")


def _pool_reset(pool: dict[str, Any]) -> datetime | None:
    """Return when a credit pool resets, or None when the payload does not say so in a usable form."""
    raw = pool.get("resets_at")
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
        return None if parsed.tzinfo is None else round_to_minute(parsed.astimezone(UTC))
    except (ValueError, OverflowError):
        return None


def identified_claude_home(config: ClaudeSourceConfig) -> tuple[Path, Path]:
    """Return this login's stored directory and the account file kept with it.

    Raises:
        ConfigError: if the config directory cannot be identified.
    """
    home = directory_identity(config.config_dir)
    return home, _account_file_kept_with(config, home)


def _account_file_kept_with(config: ClaudeSourceConfig, home: Path) -> Path:
    account = config.account_file
    try:
        relative = account.resolve().relative_to(home.resolve())
    except (OSError, ValueError):
        if same_directory(account.parent, config.config_dir):
            return home / account.name
        return account
    return home / relative


def identity_from_account_file(account_file: Path, plan: str | None) -> AccountIdentity | None:
    """Read who is logged in from Claude Code's account file, or None when nobody is.

    Raises:
        ProviderSchemaError: if the file exists but cannot be understood.
    """
    try:
        raw = read_json_object(account_file)
    except CredentialUnavailableError:
        return None
    with schema_boundary(str(account_file)):
        account = _AccountFile.model_validate(raw).oauth_account
    if account is None:
        return None
    return build_identity(
        organization_uuid=account.organization_uuid,
        organization_name=account.organization_name,
        organization_type=None,
        email=account.email_address,
        plan=plan,
    )


class ClaudeSource:
    """A Claude Code login: the keychain OAuth credential plus the account file naming who it belongs to."""

    def __init__(
        self,
        config: ClaudeSourceConfig,
        client: httpx2.Client,
        keychain_reader: KeychainReader = read_keychain_secret,
    ):
        self._config = config
        self._home, self._account_file = identified_claude_home(config)
        self._client = client
        self._read_keychain = keychain_reader
        self._keychain_account = keychain_account()
        self._identity_by_token_digest: tuple[bytes, AccountIdentity] | None = None

    @property
    def source_id(self) -> str:
        return f"claude:{self._home}"

    @property
    def home(self) -> Path:
        """The directory this source reads, identified once when the source was built."""
        return self._home

    @property
    def provider(self) -> Provider:
        return Provider.CLAUDE

    def current_identity(self) -> AccountIdentity | None:
        return identity_from_account_file(self._account_file, self._locally_known_plan())

    def read_quota(self, now: datetime) -> QuotaReading:
        credential = self._read_credential()
        expires_at_ms = credential.expires_at_ms
        if expires_at_ms is not None and from_epoch_seconds(expires_at_ms / MILLISECONDS_PER_SECOND) <= now:
            raise CredentialExpiredError("Claude access token has expired; Claude Code refreshes it on its next run")
        headers = {
            "Authorization": f"Bearer {credential.access_token}",
            "anthropic-beta": OAUTH_BETA_HEADER,
            "Accept": "application/json",
        }
        identity = self._identity_for(credential, headers)
        windows, facts = parse_usage(fetch_json_object(self._client, USAGE_URL, headers, service=SERVICE_NAME))
        return QuotaReading(identity=identity, windows=windows, observed_at=now, source="api", facts=facts)

    def _read_credential(self) -> _OAuthCredential:
        service = self._config.keychain_service
        if service is None:
            raise CredentialUnavailableError("no keychain service is configured for this Claude login")
        item = KeychainItem(service=service, account=self._keychain_account)
        raw = parse_json_object(self._read_keychain(item), f"keychain item {item}")
        try:
            credential = _KeychainBlob.model_validate(raw).claude_ai_oauth
        except ValidationError:
            # Why no cause is chained: the validation error holds the keychain item it was given.
            raise CredentialUnavailableError("Claude Code keychain item holds no usable OAuth credential") from None
        if credential is None:
            raise CredentialUnavailableError("Claude Code is not logged in with a subscription account")
        return credential

    def _locally_known_plan(self) -> str | None:
        try:
            credential = self._read_credential()
        except (CredentialUnavailableError, ProviderSchemaError):
            return None
        return plan_name(credential.rate_limit_tier, None, credential.subscription_type)

    def _identity_for(self, credential: _OAuthCredential, headers: dict[str, str]) -> AccountIdentity:
        """Ask the API who the token belongs to, once per token.

        Why: the account file and the keychain item are written separately during ``/login``, so reading
        identity from the file could label one account's usage with the other account for a moment. The
        profile call answers for the exact token used for the usage call.
        """
        digest = hashlib.sha256(credential.access_token.encode("utf-8")).digest()
        cached = self._identity_by_token_digest
        if cached is not None and cached[0] == digest:
            return cached[1]
        payload = fetch_json_object(self._client, PROFILE_URL, headers, service="Claude profile API")
        with schema_boundary("Claude profile payload"):
            profile = _Profile.model_validate(payload)
        organization = profile.organization
        identity = build_identity(
            organization_uuid=organization.uuid,
            organization_name=organization.name,
            organization_type=organization.organization_type,
            email=profile.account.email if profile.account else None,
            plan=plan_name(organization.rate_limit_tier, organization.organization_type, credential.subscription_type),
        )
        self._identity_by_token_digest = (digest, identity)
        return identity
