"""A standing Grok login, read through the CLI's stdio agent protocol.

The billing request is an extension method on the agent connection: no session is opened and no model is
called. The CLI authenticates with its own stored session and refreshes it.
"""

import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from pydantic import Field, ValidationError

from usagedash.errors import CredentialUnavailableError
from usagedash.models import AccountIdentity, Provider, QuotaReading
from usagedash.paths import directory_identity
from usagedash.providers.base import ensure_same_login
from usagedash.providers.grok import (
    AuthEntry,
    CreditsConfig,
    current_auth_entry,
    identity_from_auth_entry,
    login_identity,
    windows_and_facts,
)
from usagedash.rpc import JsonLineProcess, child_environment, jsonrpc_result
from usagedash.validation import LenientModel, schema_boundary

CLI_ARGUMENTS = ("agent", "--no-leader", "stdio")
CLI_TIMEOUT_SECONDS = 45.0
SERVICE_NAME = "Grok agent"
HOME_ENV = "GROK_HOME"
INHERITED_ENV_PREFIXES_TO_DROP = ("GROK_", "XAI_")
ACP_PROTOCOL_VERSION = 1
BILLING_METHOD = "_x.ai/billing"
LOG_RELATIVE_PATH = Path("logs") / "unified.jsonl"
BILLING_LOG_MARKER = b"billing: fetched credits config"
BILLING_LOG_MESSAGE = "billing: fetched credits config"

INITIALIZE_REQUEST = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
        "protocolVersion": ACP_PROTOCOL_VERSION,
        "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
    },
}
BILLING_REQUEST = {"jsonrpc": "2.0", "id": 1, "method": BILLING_METHOD, "params": {}}


class _BillingResult(LenientModel):
    config: CreditsConfig


class _LoggedTier(LenientModel):
    subscription_tier: str | None = Field(default=None, alias="subscriptionTier")


class _LoggedBilling(LenientModel):
    """As much of a billing log record as the plan tier needs."""

    ctx: _LoggedTier


def cli_environment(base: Mapping[str, str], home: Path) -> dict[str, str]:
    """Build the child environment for a Grok run against ``home`` (see ``rpc.child_environment``).

    Grok takes a credential, an API key, and a config path from ``GROK_*`` variables ahead of the files
    in its home. The identity this package reports is read from the home, so the billing must be too.
    """
    return child_environment(base, INHERITED_ENV_PREFIXES_TO_DROP, {HOME_ENV: str(home)})


class GrokCliSource:
    """One Grok home directory holding a login that only this dashboard uses."""

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

    @property
    def source_id(self) -> str:
        return f"grok:{self._home}"

    @property
    def home(self) -> Path:
        """The directory this source reads, identified once when the source was built."""
        return self._home

    @property
    def provider(self) -> Provider:
        return Provider.GROK

    def current_identity(self) -> AccountIdentity | None:
        entry = current_auth_entry(self._home)
        return None if entry is None else identity_from_auth_entry(entry, plan=None)

    def _signed_in_login(self) -> AuthEntry:
        """Return the login in the home's auth file as it is at this moment.

        Raises:
            CredentialUnavailableError: if nobody is signed in.
        """
        auth = current_auth_entry(self._home)
        if auth is None:
            raise CredentialUnavailableError("no Grok login here yet; run `usagedash login grok`")
        return auth

    def read_quota(self, now: datetime) -> QuotaReading:
        auth = self._signed_in_login()
        environment = cli_environment(
            os.environ if self._base_environment is None else self._base_environment, self._home
        )
        deadline = time.monotonic() + CLI_TIMEOUT_SECONDS
        log_length_before = self._log_length()
        with JsonLineProcess(
            [*self._command, *CLI_ARGUMENTS],
            env=environment,
            cwd=self._home,
            service=SERVICE_NAME,
            helper_directory=self._helper_directory,
            fallback_name=self.provider.value,
        ) as process:
            writer_pid = process.pid
            jsonrpc_result(process, INITIALIZE_REQUEST, deadline, service=SERVICE_NAME)
            raw = jsonrpc_result(process, BILLING_REQUEST, deadline, service=SERVICE_NAME)
        # Why the login is read again: the identity comes from the auth file and the billing from the
        # CLI, which answers for whoever is signed in when it runs. A login that changed in between
        # would have its usage recorded under the earlier identity.
        # Why whose login it is and not which credential: nobody else uses a standing home, so this
        # very run is what refreshes its token, and a refresh writes a new credential. The billing is
        # that account's whichever credential fetched it.
        ensure_same_login(login_identity(auth), login_identity(self._signed_in_login()), "Grok")
        with schema_boundary(f"{SERVICE_NAME} billing result"):
            windows, facts = windows_and_facts(_BillingResult.model_validate(raw).config)
        return QuotaReading(
            identity=identity_from_auth_entry(auth, plan=self._tier_logged_since(log_length_before, writer_pid)),
            windows=windows,
            observed_at=now,
            source="api",
            facts=facts,
        )

    def _log_length(self) -> int | None:
        """Return how long the CLI's log is now, or None when that cannot be told.

        Taken before a billing exchange, it marks where that exchange's own log records begin. A log
        that does not exist yet has length zero.
        """
        try:
            return (self._home / LOG_RELATIVE_PATH).stat().st_size
        except FileNotFoundError:
            return 0
        except OSError:
            return None

    def _tier_logged_since(self, log_length_before: int | None, writer_pid: int) -> str | None:
        """Read the subscription tier the CLI logged during the billing exchange; the protocol result omits it.

        Only what was appended to the log during the exchange is read, and of that only what the
        exchange's own CLI process wrote (``writer_pid``). A record from before it, or from another Grok
        process in a working home, can be another login's, or an older plan's, and would be shown as
        this account's plan. The tier is
        display-only, so a log that cannot be read, or whose start could not be marked, yields None
        instead of failing the read.
        """
        if log_length_before is None:
            return None
        try:
            with (self._home / LOG_RELATIVE_PATH).open("rb") as log:
                log.seek(log_length_before)
                appended = log.read()
        except OSError:
            return None
        return tier_of_last_billing_record(appended, writer_pid)


def tier_of_last_billing_record(log_bytes: bytes, writer_pid: int) -> str | None:
    """Return the tier in the last billing record ``writer_pid`` wrote in ``log_bytes``, or None if it gives none.

    Every record names the process that wrote it (``pid``, grok 1.0.46). A record another process wrote
    is passed over: in a working home an open Grok session logs its own billing, and a request it began
    under an earlier login can be logged while this exchange runs, after the mark and with the login
    checks still passing.

    The last record decides alone. If it cannot be read, an earlier one is not a substitute for it:
    the tier is unknown. A line that mentions billing and is not valid JSON may be that last record,
    cut short, so it counts as one that cannot be read. A valid record that only mentions the billing
    text somewhere else is not a billing record and is passed over.
    """
    tier: str | None = None
    for line in log_bytes.splitlines():
        if BILLING_LOG_MARKER not in line:
            continue
        try:
            record = json.loads(line)
        except (ValueError, RecursionError):
            tier = None
            continue
        if not isinstance(record, dict) or record.get("msg") != BILLING_LOG_MESSAGE:
            continue
        written_by = record.get("pid")
        if isinstance(written_by, bool) or written_by != writer_pid:
            continue
        try:
            tier = _LoggedBilling.model_validate(record).ctx.subscription_tier
        except ValidationError:
            tier = None
    return tier
