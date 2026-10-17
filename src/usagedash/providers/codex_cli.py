"""A standing Codex login, read through the CLI's app-server protocol.

The CLI authenticates with its own stored session, so no credential file is opened here. If the rate-limit
read fails, the CLI is asked once to refresh its token through its normal flow and the read is retried.
"""

import logging
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from usagedash.errors import CredentialUnavailableError, ProviderRequestError, ProviderSchemaError
from usagedash.models import MAX_WINDOW_SECONDS, AccountIdentity, Fact, Provider, QuotaReading, QuotaWindow
from usagedash.paths import directory_identity
from usagedash.providers.base import duration_key_and_label, with_unique_keys
from usagedash.providers.codex import (
    REPORTED_LIMIT_KEY,
    REPORTED_LIMIT_LABEL,
    extra_limit_prefixes,
    names_a_meter,
    require_main_limit,
    unreadable_limits_fact,
    with_reported_limit,
    workspace_identity,
)
from usagedash.rpc import JsonLineProcess, child_environment, jsonrpc_result
from usagedash.timeutil import from_epoch_seconds
from usagedash.validation import (
    DecimalAmount,
    EpochSeconds,
    LenientModel,
    OptionalIdentifier,
    UsedPercent,
    schema_boundary,
)

logger = logging.getLogger(__name__)

CLI_ARGUMENTS = ("app-server",)
CLI_TIMEOUT_SECONDS = 45.0
SERVICE_NAME = "Codex app-server"
HOME_ENV = "CODEX_HOME"
INHERITED_ENV_PREFIXES_TO_DROP = ("CODEX_", "OPENAI_")
CLIENT_INFO = {"name": "usagedash", "title": "AI usage dashboard", "version": "0.1.0"}
SECONDS_PER_MINUTE = 60
CHATGPT_ACCOUNT_TYPE = "chatgpt"

INITIALIZE_REQUEST_ID = 0
ACCOUNT_REQUEST_ID = 1
RATE_LIMITS_REQUEST_ID = 2
REFRESH_REQUEST_ID = 3
RATE_LIMITS_RETRY_REQUEST_ID = 4


class _Window(LenientModel):
    used_percent: UsedPercent = Field(alias="usedPercent")
    window_duration_mins: int | None = Field(
        default=None, alias="windowDurationMins", ge=0, le=MAX_WINDOW_SECONDS // SECONDS_PER_MINUTE
    )
    resets_at: EpochSeconds | None = Field(default=None, alias="resetsAt")


class _Credits(LenientModel):
    unlimited: bool | None = None
    balance: DecimalAmount | None = None


class _Snapshot(LenientModel):
    limit_id: str | None = Field(default=None, alias="limitId")
    limit_name: str | None = Field(default=None, alias="limitName")
    primary: _Window | None = None
    secondary: _Window | None = None
    credits: _Credits | None = None
    plan_type: str | None = Field(default=None, alias="planType")
    rate_limit_reached_type: str | None = Field(default=None, alias="rateLimitReachedType")


class RateLimitsResult(LenientModel):
    account_id: OptionalIdentifier = Field(default=None, alias="accountId")
    rate_limits: _Snapshot = Field(alias="rateLimits")
    rate_limits_by_limit_id: dict[str, Any] | None = Field(default=None, alias="rateLimitsByLimitId")
    """Each entry is validated by itself (``_extra_windows``), so one that cannot be read costs only itself."""


class _Account(LenientModel):
    type: str
    email: str | None = None
    plan_type: str | None = Field(default=None, alias="planType")


class _WorkspaceRouting(LenientModel):
    chatgpt_account_id: OptionalIdentifier = Field(default=None, alias="chatgptAccountId")


class _AccountResult(LenientModel):
    account: _Account | None = None
    workspace_routing: _WorkspaceRouting | None = Field(default=None, alias="workspaceRouting")


def _snapshot_windows(snapshot: _Snapshot, *, key_prefix: str, label_prefix: str, blocking: bool) -> list[QuotaWindow]:
    """Convert one metered limit's windows. Only the main limit is ``blocking``; see ``codex._windows``."""
    windows = []
    for window in (snapshot.primary, snapshot.secondary):
        if window is None:
            continue
        seconds = None if window.window_duration_mins is None else window.window_duration_mins * SECONDS_PER_MINUTE
        key, label = duration_key_and_label(seconds)
        windows.append(
            QuotaWindow(
                key=f"{key_prefix}{key}",
                label=f"{label_prefix}{label}",
                used_percent=window.used_percent,
                resets_at=None if window.resets_at is None else from_epoch_seconds(window.resets_at),
                window_seconds=seconds,
                blocking=blocking,
            )
        )
    return windows


def _limited_snapshot(snapshot: _Snapshot, *, key_prefix: str, label_prefix: str, blocking: bool) -> list[QuotaWindow]:
    """Return one pool's windows, including a non-listed full window when the CLI reports the pool reached."""
    return with_reported_limit(
        _snapshot_windows(snapshot, key_prefix=key_prefix, label_prefix=label_prefix, blocking=blocking),
        limit_reached=snapshot.rate_limit_reached_type is not None,
        blocking=blocking,
        key=f"{key_prefix}{REPORTED_LIMIT_KEY}",
        label=f"{label_prefix}{REPORTED_LIMIT_LABEL}",
    )


def _extra_windows(limit_id: str, raw: object, position: int) -> list[QuotaWindow] | None:
    """Read one extra per-feature limit from the map of limits by meter id, or return None if it is left out.

    An entry that names no meter, or whose snapshot fails validation, is logged and left out, and the
    caller says so on the card (``codex.unreadable_limits_fact``). A pool the CLI reports as reached is
    shown full even when it lists no window, and that window does not block. A pool with neither a window
    nor a reached flag is left out. The Codex CLI builds this map from the same answer the direct source
    reads. An extra limit is a display detail (see ``codex.extra_limit_prefixes``): it must not cost the
    reading its main limit, and one that is going to be left out is not validated first.

    ``position`` is the entry's place in the map, by which it is logged: the card says the log tells which
    limits were left out, and the meter id is text the CLI supplied.
    """
    if not names_a_meter(limit_id):
        logger.warning("Leaving out Codex rate limit %d that names no meter", position)
        return None
    try:
        snapshot = _Snapshot.model_validate(raw)
    except ValidationError:
        logger.warning("Leaving out Codex rate limit %d that failed validation", position)
        return None
    key_prefix, label_prefix = extra_limit_prefixes(limit_id, snapshot.limit_name)
    windows = _limited_snapshot(snapshot, key_prefix=key_prefix, label_prefix=label_prefix, blocking=False)
    if not windows:
        logger.warning("Leaving out Codex rate limit %d with no window", position)
        return None
    return windows


def _is_the_main_meter(limit_id: str, main: _Snapshot) -> bool:
    """Report whether ``limit_id`` is the main pool's meter.

    The map is keyed by meter id, and that id is the main pool's ``limitId``. The same windows and
    the same display name are not that id: two meters can both be an unnamed weekly window at the
    same percentage. An entry that cannot be matched is kept (``docs/known-limitations.md``).
    """
    return names_a_meter(main.limit_id) and limit_id == main.limit_id


def parse_rate_limits(payload: dict[str, Any]) -> tuple[RateLimitsResult, tuple[QuotaWindow, ...], tuple[Fact, ...]]:
    """Convert an ``account/rateLimits/read`` result into quota windows and display facts.

    Raises:
        ProviderSchemaError: if the result fails validation or reports no window of its main limit.
    """
    with schema_boundary(f"{SERVICE_NAME} rate limits"):
        limits = RateLimitsResult.model_validate(payload)
        main = limits.rate_limits
        windows = require_main_limit(
            _limited_snapshot(main, key_prefix="", label_prefix="", blocking=True),
            SERVICE_NAME,
        )
        unreadable = 0
        for position, (limit_id, raw) in enumerate((limits.rate_limits_by_limit_id or {}).items()):
            if _is_the_main_meter(limit_id, main):
                continue
            extra = _extra_windows(limit_id, raw, position)
            if extra is None:
                unreadable += 1
            else:
                windows += extra
        facts = []
        if main.credits is not None:
            if main.credits.unlimited:
                facts.append(Fact("Credits", "Unlimited"))
            elif main.credits.balance is not None:
                facts.append(Fact("Credits", f"{main.credits.balance:,.0f} remaining"))
        if unreadable:
            facts.append(unreadable_limits_fact(unreadable))
        return limits, with_unique_keys(windows), tuple(facts)


def _identity(account: _Account, workspace_id: str | None, plan_type: str | None) -> AccountIdentity | None:
    """Build the identity once the workspace is named; None while only the person is known."""
    if workspace_id is None:
        return None
    return workspace_identity(account_id=workspace_id, email=account.email, plan_type=plan_type or account.plan_type)


def cli_environment(base: Mapping[str, str], home: Path) -> dict[str, str]:
    """Build the child environment for a Codex run against ``home`` (see ``rpc.child_environment``).

    Codex takes an access token and an API key from ``CODEX_*`` and ``OPENAI_*`` variables ahead of the
    login stored in its home, and this package records what it answers as that login's.
    """
    return child_environment(base, INHERITED_ENV_PREFIXES_TO_DROP, {HOME_ENV: str(home)})


class CodexCliSource:
    """One Codex home directory holding a login that only this dashboard uses.

    The login's identity is only known from the CLI's answers, so it is remembered between reads. It is
    dropped the moment an account read is attempted and set again only from an answer that names the
    workspace. A read that fails before that point (the CLI would not start) leaves it as it was; a read
    that fails at or after it can never leave an earlier login's identity in place.
    """

    def __init__(
        self,
        home: Path,
        command: Sequence[str],
        base_environment: Mapping[str, str] | None = None,
        *,
        helper_directory: Path | None = None,
    ):
        self._home = directory_identity(home)
        self._command = tuple(command)
        self._base_environment = base_environment
        self._helper_directory = helper_directory
        self._last_identity: AccountIdentity | None = None

    @property
    def source_id(self) -> str:
        return f"codex:{self._home}"

    @property
    def provider(self) -> Provider:
        return Provider.CODEX

    def current_identity(self) -> AccountIdentity | None:
        """Return who the CLI last said is logged in; this source never opens the credential file."""
        return self._last_identity

    def read_quota(self, now: datetime) -> QuotaReading:
        environment = cli_environment(
            os.environ if self._base_environment is None else self._base_environment, self._home
        )
        deadline = time.monotonic() + CLI_TIMEOUT_SECONDS
        with JsonLineProcess(
            [*self._command, *CLI_ARGUMENTS],
            env=environment,
            cwd=self._home,
            service=SERVICE_NAME,
            helper_directory=self._helper_directory,
            fallback_name=self.provider.value,
        ) as process:
            jsonrpc_result(
                process,
                {"id": INITIALIZE_REQUEST_ID, "method": "initialize", "params": {"clientInfo": CLIENT_INFO}},
                deadline,
                service=SERVICE_NAME,
            )
            process.send({"method": "initialized"})
            account, workspace_id = self._read_account(process, deadline, ACCOUNT_REQUEST_ID, refresh_token=False)
            try:
                raw_limits = jsonrpc_result(
                    process,
                    {"id": RATE_LIMITS_REQUEST_ID, "method": "account/rateLimits/read"},
                    deadline,
                    service=SERVICE_NAME,
                )
            except ProviderRequestError:
                account, workspace_id = self._read_account(process, deadline, REFRESH_REQUEST_ID, refresh_token=True)
                raw_limits = jsonrpc_result(
                    process,
                    {"id": RATE_LIMITS_RETRY_REQUEST_ID, "method": "account/rateLimits/read"},
                    deadline,
                    service=SERVICE_NAME,
                )

        limits, windows, facts = parse_rate_limits(raw_limits)
        identity = _identity(account, limits.account_id or workspace_id, limits.rate_limits.plan_type)
        if identity is None:
            raise ProviderSchemaError(f"{SERVICE_NAME} did not identify the ChatGPT workspace")
        self._last_identity = identity
        return QuotaReading(identity=identity, windows=windows, observed_at=now, source="api", facts=facts)

    def _read_account(
        self, process: JsonLineProcess, deadline: float, request_id: int, *, refresh_token: bool
    ) -> tuple[_Account, str | None]:
        """Ask who is logged in and remember the answer. Returns the account and its workspace id, if named.

        The remembered identity is dropped before asking. Whatever goes wrong from here on (no answer,
        an answer of the wrong shape, no login), who is logged in is no longer known, and an identity
        left over from an earlier read would be reported as if it were.

        Raises:
            CredentialUnavailableError: if no ChatGPT account is logged in.
            ProviderRequestError: if the CLI does not answer or answers with an error.
            ProviderSchemaError: if the answer has an unexpected shape.
        """
        self._last_identity = None
        result = jsonrpc_result(
            process,
            {"id": request_id, "method": "account/read", "params": {"refreshToken": refresh_token}},
            deadline,
            service=SERVICE_NAME,
        )
        try:
            answer = _AccountResult.model_validate(result)
        except ValidationError:
            raise ProviderSchemaError(f"{SERVICE_NAME} account has an unexpected shape") from None
        account = answer.account
        if account is None or account.type != CHATGPT_ACCOUNT_TYPE:
            raise CredentialUnavailableError("no ChatGPT login here yet; run `usagedash login codex`")
        workspace_id = answer.workspace_routing.chatgpt_account_id if answer.workspace_routing else None
        self._last_identity = _identity(account, workspace_id, plan_type=None)
        return account, workspace_id
