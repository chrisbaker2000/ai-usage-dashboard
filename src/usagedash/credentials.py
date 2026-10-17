"""Read-only access to credentials the CLIs already hold.

Nothing here writes, refreshes, or copies a credential: refresh tokens rotate on use, so refreshing one from
outside the owning CLI would invalidate the CLI's own session. Secret values are returned to the caller for
immediate use in a request header and are never logged or persisted.

No error raised here has a chained cause. The underlying error was raised while credential material was
in hand, and may hold it: a decoding error keeps the bytes it was given, and a timed-out or failed
command keeps the output it captured. A cause is printed with every traceback, so each error here names
the kind of failure and nothing else.
"""

import base64
import binascii
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from usagedash.errors import CredentialUnavailableError, NotAJwtError, ProviderSchemaError

SECURITY_BINARY = "/usr/bin/security"
KEYCHAIN_TIMEOUT_SECONDS = 10
KEYCHAIN_ITEM_NOT_FOUND_EXIT = 44
_JWT_PAYLOAD_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


@dataclass(frozen=True)
class KeychainItem:
    """A generic-password item, named the way its owner looks it up: by service and account.

    Both are needed. Two items can share a service under different accounts, and ``security`` given the
    service alone returns whichever it finds first, which need not be the one the owner reads.

    The account is a login name, which can carry a person's name, so neither ``str`` nor ``repr`` shows
    it: an error built from either is stored, shown on the page, and printed by ``usagedash collect``.
    """

    service: str
    account: str = field(repr=False)

    def __str__(self) -> str:
        return f"{self.service!r} under this macOS account's name"


KeychainReader = Callable[[KeychainItem], str]


def read_keychain_secret(item: KeychainItem) -> str:
    """Return the password stored in the login keychain under one generic-password service and account.

    Raises:
        CredentialUnavailableError: if the item does not exist, the keychain is locked, or the lookup times out.
    """
    try:
        completed = subprocess.run(
            [SECURITY_BINARY, "find-generic-password", "-a", item.account, "-s", item.service, "-w"],
            capture_output=True,
            text=True,
            timeout=KEYCHAIN_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError) as error:
        raise CredentialUnavailableError(f"keychain lookup for {item} failed: {type(error).__name__}") from None
    if completed.returncode == KEYCHAIN_ITEM_NOT_FOUND_EXIT:
        raise CredentialUnavailableError(f"no keychain item {item}")
    if completed.returncode != 0:
        raise CredentialUnavailableError(f"keychain lookup for {item} exited {completed.returncode}")
    secret = completed.stdout.strip()
    if secret == "":
        raise CredentialUnavailableError(f"keychain item {item} is empty")
    return secret


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON file whose top level must be an object.

    Raises:
        CredentialUnavailableError: if the file is missing or unreadable.
        ProviderSchemaError: if the content is not UTF-8 or not a JSON object.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CredentialUnavailableError(f"{path} does not exist") from None
    except UnicodeDecodeError:
        raise ProviderSchemaError(f"{path} is not UTF-8") from None
    except OSError as error:
        raise CredentialUnavailableError(f"cannot read {path}: {type(error).__name__}") from None
    return parse_json_object(raw, str(path))


def parse_json_object(raw: str, origin: str) -> dict[str, Any]:
    """Parse text that must be a JSON object. ``origin`` names the source in the error, never the content.

    Raises:
        ProviderSchemaError: if the text is not valid JSON or its top level is not an object.
    """
    try:
        parsed = json.loads(raw)
    except (ValueError, RecursionError):
        raise ProviderSchemaError(f"{origin} is not valid JSON") from None
    if not isinstance(parsed, dict):
        raise ProviderSchemaError(f"{origin} is not a JSON object")
    return parsed


def decode_jwt_claims(token: str) -> dict[str, Any]:
    """Return the claims of a JWT without verifying its signature.

    Why: the token comes from the CLI's own credential file and is only read for display fields and expiry;
    the provider verifies it on every request, so local verification would add nothing.

    Raises:
        NotAJwtError: if the value is not three dot-separated parts, so not a JWT at all.
        ProviderSchemaError: if it is, but its payload is not a base64url, UTF-8 JSON object.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise NotAJwtError("token is not a JWT")
    payload = parts[1]
    if not payload or any(character not in _JWT_PAYLOAD_ALPHABET for character in payload):
        raise ProviderSchemaError("JWT payload is not valid base64url")
    try:
        decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
    except (binascii.Error, ValueError):
        raise ProviderSchemaError("JWT payload is not valid base64url") from None
    try:
        text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        raise ProviderSchemaError("JWT payload is not UTF-8") from None
    return parse_json_object(text, "JWT payload")
