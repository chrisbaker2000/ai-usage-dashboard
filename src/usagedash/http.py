"""One GET helper that maps transport and status failures onto this package's typed errors."""

import math
from collections.abc import Mapping
from typing import Any

import httpx2

from usagedash.errors import ConfigError, CredentialExpiredError, ProviderRequestError, ProviderSchemaError

REQUEST_TIMEOUT_SECONDS = 20.0
USER_AGENT = "usagedash/0.1"
MAX_RETRY_AFTER_SECONDS = 86_400.0


def build_client() -> httpx2.Client:
    """Create the shared HTTP client. Redirects stay off so a bearer token is never replayed to another host.

    The client takes its proxies and certificate files from the environment, and each can be unusable:
    a malformed proxy address (``InvalidURL``), an unsupported proxy scheme (``ValueError``), a SOCKS
    proxy without its optional package (``ImportError``), or a certificate file that is missing or is not
    one (``OSError``), each seen on httpx2 1.0.

    Raises:
        ConfigError: if those settings cannot be used. It names the error's type only and chains no
            cause: a proxy address can hold a password, and a cause is printed with every traceback.
    """
    try:
        return httpx2.Client(
            timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False, headers={"User-Agent": USER_AGENT}
        )
    except (httpx2.InvalidURL, ImportError, OSError, ValueError) as error:
        raise ConfigError(
            f"cannot set up the HTTP client from the proxy or certificate settings: {type(error).__name__}"
        ) from None


def fetch_json_object(client: httpx2.Client, url: str, headers: Mapping[str, str], *, service: str) -> dict[str, Any]:
    """GET a URL and return its JSON-object body.

    Raises:
        CredentialExpiredError: on 401/403, meaning the token is expired or revoked.
        ProviderRequestError: if a header value cannot be sent, on a transport failure, or on any other
            non-200 status (429 carries ``retry_after_seconds``).
        ProviderSchemaError: if the body is not a JSON object.
    """
    unsendable = sorted(name for name, value in headers.items() if not _is_header_text(value))
    if unsendable:
        raise ProviderRequestError(
            f"{service} request not sent: {', '.join(unsendable)} holds characters a header cannot carry"
        )
    try:
        response = client.get(url, headers=dict(headers))
    except httpx2.HTTPError as error:
        # Why no cause is chained: the transport's own message can quote a header it was sending, and
        # the Authorization header is the bearer token. A chained cause is printed with every traceback.
        raise ProviderRequestError(f"{service} request failed: {type(error).__name__}") from None
    status = response.status_code
    if status in (401, 403):
        raise CredentialExpiredError(f"{service} rejected the stored token (HTTP {status})")
    if status == 429:
        raise ProviderRequestError(
            f"{service} rate-limited the request (HTTP 429)",
            status_code=status,
            retry_after_seconds=_retry_after_seconds(response),
        )
    if status != 200:
        raise ProviderRequestError(f"{service} returned HTTP {status}", status_code=status)
    try:
        payload = response.json()
    except (ValueError, RecursionError):
        # Why no cause is chained: a JSON decoding error carries the whole body it was given.
        raise ProviderSchemaError(f"{service} response is not JSON") from None
    if not isinstance(payload, dict):
        raise ProviderSchemaError(f"{service} response is not a JSON object")
    return payload


def _is_header_text(value: str) -> bool:
    """Report whether an HTTP header can carry ``value`` as it is: printable ASCII with no space at either end.

    Why this is checked here and not left to the transport: the transport refuses such a value with an
    error that quotes it in full, or fails to encode it with an error this package does not type.
    """
    return value == value.strip() and all(" " <= character <= "~" for character in value)


def _retry_after_seconds(response: httpx2.Response) -> float | None:
    """Return the wait a 429 asked for, in seconds, capped at a day; None if it gave no usable number."""
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_SECONDS)
