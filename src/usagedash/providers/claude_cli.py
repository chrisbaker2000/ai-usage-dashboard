"""A standing Claude Code login, read through the CLI's headless control protocol.

Claude Code authenticates with its own stored session and refreshes it, so no credential is read here. The
request is a control message, not a prompt, so it calls no model and uses no plan quota.
"""

import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from usagedash.config import ClaudeSourceConfig
from usagedash.errors import CredentialUnavailableError, ProviderRequestError, ProviderSchemaError
from usagedash.models import AccountIdentity, Provider, QuotaReading
from usagedash.providers.base import NoQuotaDataError, ensure_same_login
from usagedash.providers.claude import (
    identified_claude_home,
    identity_from_account_file,
    keychain_account,
    parse_usage,
    plan_name,
)
from usagedash.rpc import JsonLineProcess, Message, child_environment
from usagedash.validation import LenientModel, schema_boundary

CLI_ARGUMENTS = (
    "-p",
    "--input-format",
    "stream-json",
    "--output-format",
    "stream-json",
    "--verbose",
    "--no-session-persistence",
    "--strict-mcp-config",
)
CLI_TIMEOUT_SECONDS = 60.0
SERVICE_NAME = "Claude Code"
CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"
CREDENTIAL_DIR_ENV = "CLAUDE_SECURESTORAGE_CONFIG_DIR"
KEYCHAIN_ACCOUNT_ENV = "USER"
INHERITED_ENV_PREFIXES_TO_DROP = ("CLAUDE", "ANTHROPIC_")


class _UsageAnswer(LenientModel):
    """The body of a ``get_usage`` answer.

    The flag decides between "this login has no plan limits" and a reading, so it is required and, like
    every field of these models, must be a real boolean: the text ``"false"`` is truthy, and must not be
    taken for either answer.
    """

    rate_limits_available: bool
    rate_limits: object = None
    subscription_type: str | None = None


def cli_environment(
    base: Mapping[str, str], config_dir: Path | None, credential_dir: Path | None = None
) -> dict[str, str]:
    """Build the child environment for a Claude Code run, a sign-in included (see ``rpc.child_environment``).

    ``config_dir`` is None for the default home, which must not be named: setting the variable changes
    the keychain item Claude Code looks up. ``USER`` is always ``claude.keychain_account()``, because
    Claude Code takes the keychain item's account from it: a sign-in from a shell with another ``USER``
    would file the login where no later read looks.
    """
    settings: dict[str, str] = {KEYCHAIN_ACCOUNT_ENV: keychain_account()}
    if config_dir is not None:
        settings[CONFIG_DIR_ENV] = str(config_dir)
        if credential_dir is not None and str(credential_dir) != str(config_dir):
            settings[CREDENTIAL_DIR_ENV] = str(credential_dir)
    return child_environment(base, INHERITED_ENV_PREFIXES_TO_DROP, settings)


def control_request(process: JsonLineProcess, subtype: str, deadline: float) -> Message:
    """Send one control request and return its response body.

    Raises:
        ProviderRequestError: if Claude Code rejects the request or does not answer.
        ProviderSchemaError: if the answer has no object body.
    """
    request_id = f"usagedash-{subtype}"
    process.send({"type": "control_request", "request_id": request_id, "request": {"subtype": subtype}})

    def is_answer(message: Message) -> bool:
        response = message.get("response")
        return (
            message.get("type") == "control_response"
            and isinstance(response, dict)
            and response.get("request_id") == request_id
        )

    response = process.receive(is_answer, deadline)["response"]
    if response.get("subtype") != "success":
        # Why the reason is left out: it is Claude Code's own text, and this message is stored and shown.
        raise ProviderRequestError(f"{SERVICE_NAME} rejected {subtype}")
    body = response.get("response")
    if not isinstance(body, dict):
        raise ProviderSchemaError(f"{SERVICE_NAME} answered {subtype} without a body")
    return body


class ClaudeCliSource:
    """One Claude Code config directory holding a login that only this dashboard uses."""

    def __init__(
        self,
        config: ClaudeSourceConfig,
        command: Sequence[str],
        base_environment: Mapping[str, str] | None = None,
        *,
        helper_directory: Path | None = None,
    ):
        self._config = config
        self._home, self._account_file = identified_claude_home(config)
        self._command = tuple(command)
        self._base_environment = base_environment
        self._helper_directory = helper_directory

    @property
    def source_id(self) -> str:
        return f"claude:{self._home}"

    @property
    def provider(self) -> Provider:
        return Provider.CLAUDE

    def current_identity(self) -> AccountIdentity | None:
        return identity_from_account_file(self._account_file, None)

    def _signed_in_identity(self) -> AccountIdentity:
        """Return who the account file says is logged in at this moment.

        Raises:
            CredentialUnavailableError: if nobody is.
        """
        identity = self.current_identity()
        if identity is None:
            raise CredentialUnavailableError("no Claude login here yet; run `usagedash login claude`")
        return identity

    def read_quota(self, now: datetime) -> QuotaReading:
        local_identity = self._signed_in_identity()
        named = None if self._config.is_default_home else self._home
        # Why: Claude hashes the configured spelling for its keychain item and does not resolve it.
        credential = None if self._config.is_default_home else self._config.config_dir
        environment = cli_environment(
            os.environ if self._base_environment is None else self._base_environment,
            named,
            credential,
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
            control_request(process, "initialize", deadline)
            usage = control_request(process, "get_usage", deadline)
        # Why the login is read again: the identity comes from the account file and the usage from the
        # CLI, which answers for whoever is logged in when it runs. A login that changed in between
        # would have its usage recorded under the earlier identity.
        # Why whole identities are compared: the account is keyed on the organization, and two members
        # of one organization are two logins with their own usage.
        ensure_same_login(local_identity, self._signed_in_identity(), "Claude")

        with schema_boundary(f"{SERVICE_NAME} get_usage answer"):
            answer = _UsageAnswer.model_validate(usage)
        if not answer.rate_limits_available:
            raise NoQuotaDataError("Claude Code reported no plan limits for this login")
        if answer.rate_limits is None:
            # Why not a schema error: seen live on 2026-10-04 in 1 of about 16 answers, with the next
            # exchange answering in full. It is a read that did not get its answer, and the next cycle asks
            # again; a schema error says the shape changed and the code must follow.
            raise ProviderRequestError(f"{SERVICE_NAME} said plan limits are available but sent none this time")
        if not isinstance(answer.rate_limits, dict):
            raise ProviderSchemaError(f"{SERVICE_NAME} sent plan limits that are not an object")
        windows, facts = parse_usage(answer.rate_limits)
        identity = AccountIdentity(
            provider=local_identity.provider,
            external_id=local_identity.external_id,
            default_label=local_identity.default_label,
            email=local_identity.email,
            plan=plan_name(None, None, answer.subscription_type),
        )
        return QuotaReading(identity=identity, windows=windows, observed_at=now, source="api", facts=facts)
