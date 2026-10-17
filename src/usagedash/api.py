"""HTTP API and static dashboard, served on loopback only.

Every ``/api`` request must prove it holds the access token from the data directory, which only this
user can read; the ``Host`` header must be a loopback name; state-changing requests must also come from
the page. Refusals and acceptances are recorded by method, route template, and reason or credential
kind, never by what was sent.

The page proves it holds the token without sending it: it asks ``GET /challenge`` for a single-use
value this process made up, adds a nonce of its own, and signs method, path and query, body digest,
challenge, and nonce with the token (``Authorization: Usagedash <challenge>.<nonce>.<digest>``,
``access.RequestVerifier``). Every body travels sealed under a key derived from the token
(``access.BodySealer``), and the answer carries ``X-Usagedash-Response`` over the challenge, the nonce,
the status, and the sealed body, so a process holding the port can neither read, forge, nor replay an
exchange. ``Authorization: Bearer <token>`` is accepted, with plain bodies, from a client that is not a
page. The page is an owner-only local file (``access.write_bootstrap_page``) whose ``Origin`` is
``null``, allowed for a change only with the page's header and a valid signature; the shell served here
holds no token, sends nothing, and shows how to sign in.
"""

import asyncio
import contextlib
import enum
import logging
import time
from collections.abc import AsyncGenerator, Callable, Coroutine
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, NamedTuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, Response, status
from fastapi import Path as PathParameter
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from fastapi.telemetry import TelemetryConfig
from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from usagedash import db
from usagedash.access import (
    RESPONSE_SIGNATURE_HEADER,
    SEALED_CONTENT_TYPE,
    SIGNATURE_SCHEME,
    BodySealer,
    RequestVerifier,
    SignatureOutcome,
    SignedAuthorization,
    answer_seal_context,
    body_digest,
    is_well_formed_token,
    request_seal_context,
    response_signature,
    token_matches,
)
from usagedash.collector import Collector, CycleAlreadyRunningError, CycleReport, describe_without_messages
from usagedash.config import Settings
from usagedash.db import Database, SnapshotRow, SourceStatusRow
from usagedash.errors import SealedBodyError, UsageDashError
from usagedash.pricing import COST_NOTE, CURRENCY, MICRODOLLAR, PRICED_ON, price_model_rows
from usagedash.projection import AccountView, Advice, ProviderView, Thresholds, WindowView, build_provider_views
from usagedash.timeutil import Clock, in_utc, to_iso
from usagedash.tokens import RangeKey, build_token_report

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
CSRF_HEADER_VALUE = "usagedash"
AUTHORIZATION_SCHEME = "Bearer"
"""The scheme that carries the token itself. The page never uses it; see ``SIGNATURE_SCHEME``."""
ACCEPTED_SCHEMES = f"{SIGNATURE_SCHEME}, {AUTHORIZATION_SCHEME}"
"""The ``WWW-Authenticate`` challenge: both schemes a request may present."""
FILE_PAGE_ORIGIN = "null"
"""What a browser sends as ``Origin`` for a page opened from a local file. No website can send it."""
CORS_ALLOWED_METHODS = ["GET", "POST", "PATCH"]
CORS_ALLOWED_HEADERS = ["Authorization", "X-Requested-With"]
"""The request headers the page adds. ``Content-Type`` is allowed by default."""
MAX_SIGNED_BODY_BYTES = 64 * 1024
"""The largest body an ``/api`` request may carry. The only body the API takes is a rename of at most 60 characters."""
BODY_TOO_LARGE_MESSAGE = "The request body is larger than this API accepts."
API_PREFIX = "/api"
HEALTH_PATH = "/healthz"
CHALLENGE_PATH = "/challenge"
NOT_SIGNED_IN_MESSAGE = "Not signed in to the dashboard. Run `usagedash open` to open it in your browser."
INTERNAL_ERROR_MESSAGE = "The dashboard failed to answer this request. Its log says where."
UNSEALED_BODY_MESSAGE = "The body of a signed request must be sealed."
SEALED_BODY_NOT_OPENED_MESSAGE = "The sealed body could not be opened."
JSON_CONTENT_TYPE = "application/json"
INVALID_REQUEST_MESSAGE = "The request is not one this API accepts."
UNMATCHED_ROUTE = "(no route)"
HOST_NOT_ALLOWED = "the host is not allowed"
"""The reason recorded when the host guard refuses a request. It names no header and no path."""
PREFLIGHT_NOT_ALLOWED = "the cross-origin preflight is not allowed"
"""The reason recorded when the cross-origin layer turns a preflight away. It names no origin and no path."""
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
"""The methods that change nothing. A request by any other method must have come from the dashboard's page."""
REQUEST_PARTS = frozenset({"query", "path", "body", "header", "cookie"})
"""The parts of a request FastAPI names first in the location of a validation failure."""
HISTORY_WINDOW = timedelta(days=7)
MIN_REFRESH_SPACING = timedelta(seconds=20)
MAX_LABEL_LENGTH = 60
MAX_TIME_ZONE_NAME_LENGTH = 64
MAX_ROW_ID = 2**63 - 1
"""The largest id an SQLite row can have; a larger number in a URL cannot name one."""

TELEMETRY_OFF: TelemetryConfig = {
    "tracing": False,
    "metrics": False,
    "logs": False,
    "operation_spans": False,
    "auto_configure": False,
}
"""Every switch of FastAPI's own OpenTelemetry support, off.

FastAPI records each request as spans, metrics, and logs through whatever OpenTelemetry providers the
process has, and at startup adds exporters named by ``OTEL_*`` variables it inherits. All of that is on
unless it is turned off. A request's path holds an account id, its log records hold exception messages
and stack traces, and this dashboard sends nothing about its accounts anywhere. Turning off the
exporters alone would leave a provider that something else configured still recording.
"""

INFORMATIONAL_ERROR_TYPES = frozenset({"NoQuotaDataError", "CredentialUnavailableError"})
WARNING_ERROR_TYPES = frozenset({"CredentialExpiredError"})

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def _printable(label: str) -> str:
    """Refuse a label holding a line break, a control character, or anything else that does not print.

    A label is shown in the page and inside the advice sentences, where such a character could split a
    line or hide part of the text.
    """
    if not label.isprintable():
        raise ValueError("a label may only contain printable characters")
    return label


class RenameRequest(BaseModel):
    """The body of a rename. Nothing is converted from another type, and no other field is accepted."""

    model_config = ConfigDict(strict=True, extra="forbid")

    label: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_LABEL_LENGTH),
        AfterValidator(_printable),
    ]


def _time_zone(name: str) -> ZoneInfo:
    """Look up an IANA time zone by name.

    Raises:
        HTTPException: 422 if there is no such zone. The lookup reads the system's zone files by that
            name, so a name that is not a zone can fail as a missing key, a malformed key, or a file
            error (a directory, for instance); all three mean the same thing to the caller.
    """
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "unknown time zone") from None


def _optional_iso(moment: datetime | None) -> str | None:
    return None if moment is None else to_iso(moment)


def _window_json(window: WindowView) -> dict[str, Any]:
    return {
        "key": window.key,
        "label": window.label,
        "blocking": window.blocking,
        "window_seconds": window.window_seconds,
        "used_percent": window.used_percent,
        "effective_percent": window.effective_percent,
        "state": window.state.value,
        "resets_at": _optional_iso(window.resets_at),
        "reset_passed": window.reset_passed,
        "as_of": to_iso(window.as_of),
        "freshness": window.freshness.value,
        "pace_exhausts_at": _optional_iso(window.pace_exhausts_at),
    }


def _text_id(account_id: int) -> str:
    """Write an account id as the page gets it: decimal text.

    Why not a number: an id is an SQLite row id, which can be any 64-bit integer, and a browser holds
    numbers exactly only up to 2**53. Two ids above that could reach the page as the same number, and a
    rename meant for one account would then address the other. Text is exact for every id.
    """
    return str(account_id)


def _optional_text_id(account_id: int | None) -> str | None:
    return None if account_id is None else _text_id(account_id)


def _account_json(view: AccountView) -> dict[str, Any]:
    account = view.account
    return {
        "id": _text_id(account.id),
        "label": account.label,
        "email": account.email,
        "plan": account.plan,
        "color_slot": account.color_slot,
        "is_active": view.is_active,
        "is_live": view.is_live,
        "last_live_at": _optional_iso(account.last_live_at),
        "available_now": view.available_now,
        "headroom_percent": view.headroom_percent,
        "next_available_at": _optional_iso(view.next_available_at),
        "windows": [_window_json(window) for window in view.windows],
        "facts": [
            {"label": fact.label, "value": fact.value, "resets_at": _optional_iso(fact.resets_at)}
            for fact in account.facts
        ],
    }


def _advice_json(advice: Advice) -> dict[str, Any]:
    return {
        "action": advice.action.value,
        "summary": advice.summary,
        "active_account_id": _optional_text_id(advice.active_account_id),
        "target_account_id": _optional_text_id(advice.target_account_id),
        "next_available_at": _optional_iso(advice.next_available_at),
        "as_of": _optional_iso(advice.as_of),
    }


def _provider_json(view: ProviderView) -> dict[str, Any]:
    return {
        "provider": view.provider.value,
        "display_name": view.provider.display_name,
        "advice": _advice_json(view.advice),
        "accounts": [_account_json(account) for account in view.accounts],
    }


def severity_of(error_type: str | None) -> str:
    """Classify a source's outcome by the name of its error's type, or None for no error.

    This is the one classification: the page's list of data sources and the exit status of
    ``usagedash collect`` both come from it.
    """
    if error_type is None:
        return "ok"
    if error_type in INFORMATIONAL_ERROR_TYPES:
        return "info"
    if error_type in WARNING_ERROR_TYPES:
        return "warning"
    return "error"


def _status_json(row: SourceStatusRow) -> dict[str, Any]:
    return {
        "source_id": row.source_id,
        "kind": row.kind,
        "severity": severity_of(row.error_type),
        "message": row.error_message,
        "last_attempt_at": to_iso(row.last_attempt_at),
        "last_success_at": _optional_iso(row.last_success_at),
    }


def _history_json(rows: list[SnapshotRow], since: datetime) -> list[dict[str, Any]]:
    """Group step-function snapshots into one series per account window.

    A step is reported from where it enters the range to when it was last confirmed; the step in effect
    when the range starts is reported as starting and ending there, which is all that is known of it
    inside the range.
    """
    series: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        start = max(row.observed_at, since)
        series.setdefault((row.account_id, row.window_key), []).append(
            {
                "from": to_iso(start),
                "to": to_iso(max(row.confirmed_at, start)),
                "used_percent": row.used_percent,
                "resets_at": _optional_iso(row.resets_at),
            }
        )
    return [
        {"account_id": _text_id(account_id), "window_key": window_key, "steps": steps}
        for (account_id, window_key), steps in series.items()
    ]


_COUNT_FIELDS = ("input", "cache_read", "cache_write", "output", "reasoning", "requests", "total")


def _counts_text(counts: dict[str, int]) -> dict[str, str]:
    """Write token counts as decimal text.

    Why not numbers: a stored count can be ``10**18``, and a browser holds numbers exactly only up to
    ``2**53``. A count past that would reach the page as a smaller number, and the chart and the table
    would show that instead of what was stored.
    """
    return {field: str(counts[field]) for field in _COUNT_FIELDS}


def _costs_json(models: list[dict[str, Any]]) -> dict[str, Any]:
    """Price the report's model rows at the published short-prompt rates.

    Why not inside the token report: the chart, the model table, and the provider tiles are one sum of
    token counts, and a price is not a token count.
    """
    return {
        "priced_on": PRICED_ON,
        "currency": CURRENCY,
        "unit": MICRODOLLAR,
        "note": COST_NOTE,
        "accounts": [
            {
                "provider": account.provider,
                "account_id": _text_id(account.account_id),
                "priced_microdollars": str(account.priced_microdollars),
                "unpriced_tokens": str(account.unpriced_tokens),
                "models": [
                    {
                        "model": model.model,
                        "priced": model.microdollars is not None,
                        "microdollars": None if model.microdollars is None else str(model.microdollars),
                        "unpriced_tokens": str(model.unpriced_tokens),
                    }
                    for model in account.models
                ],
            }
            for account in price_model_rows(models)
        ],
    }


def _token_report_json(report: dict[str, Any]) -> dict[str, Any]:
    """Return a token report as the page gets it.

    Each account id is text (see ``_text_id``) and each token count is decimal text (see ``_counts_text``).
    The retail cost is attached from the same model rows, before those counts are turned into text.
    """
    return {
        **report,
        "series": [
            {
                **series,
                "account_id": _text_id(series["account_id"]),
                "points": [_counts_text(point) for point in series["points"]],
                "totals": _counts_text(series["totals"]),
            }
            for series in report["series"]
        ],
        "models": [
            {**model, "account_id": _text_id(model["account_id"]), "totals": _counts_text(model["totals"])}
            for model in report["models"]
        ],
        "provider_totals": [{**item, "totals": _counts_text(item["totals"])} for item in report["provider_totals"]],
        "costs": _costs_json(report["models"]),
    }


def build_state(database: Database, settings: Settings, collector: Collector | None, now: datetime) -> dict[str, Any]:
    """Assemble everything the dashboard's quota view needs in one response."""
    # Why: subtracting a week from a zone with clock changes uses its wall time, so the range would
    # start an hour early or late across a daylight-saving change.
    now = in_utc(now)
    thresholds = Thresholds(settings.warn_percent, settings.critical_percent)
    since = now - HISTORY_WINDOW
    with database.read() as conn:
        history = db.snapshot_history(conn, since)
        views = build_provider_views(
            db.list_accounts(conn),
            db.latest_snapshots(conn),
            history,
            db.list_source_states(conn),
            now=now,
            thresholds=thresholds,
            poll_interval_seconds=settings.poll_interval_seconds,
        )
        statuses = db.list_statuses(conn)
    return {
        "generated_at": to_iso(now),
        "poll_interval_seconds": settings.poll_interval_seconds,
        "expected_accounts_per_provider": settings.expected_accounts_per_provider,
        "thresholds": {"warn_percent": settings.warn_percent, "critical_percent": settings.critical_percent},
        "collector": {
            "running": collector.is_running if collector else False,
            "last_finished_at": _optional_iso(collector.last_finished_at) if collector else None,
        },
        "providers": [_provider_json(view) for view in views],
        "history": _history_json(history, since),
        "sources": [_status_json(row) for row in statuses],
    }


def presented_token(authorization: str | None) -> str:
    """Return the token from an ``Authorization: Bearer`` header, or an empty string when there is none."""
    scheme, _, value = (authorization or "").partition(" ")
    return value.strip() if scheme.lower() == AUTHORIZATION_SCHEME.lower() else ""


def signed_target(request: Request) -> str:
    """Return the path and query as the request line carried them (the raw path): what a signature covers."""
    raw_path = request.scope.get("raw_path")
    path = raw_path.decode("ascii", errors="replace") if isinstance(raw_path, bytes) else request.url.path
    query = request.url.query
    return f"{path}?{query}" if query else path


class Credential(enum.Enum):
    """How a request proved it holds the access token."""

    SIGNATURE = "signature"
    """A per-request signature keyed by the token: the dashboard page."""
    BEARER = "bearer"
    """The token itself: a client that is not a browser page."""


class Proof(NamedTuple):
    """How a request proved it holds the token, and the signature it presented when it signed."""

    credential: Credential
    signed: SignedAuthorization | None


def presented_signature(authorization: str | None) -> SignedAuthorization | None:
    """Return the parts of a well-formed ``Authorization: Usagedash`` header, or None."""
    scheme, _, credentials = (authorization or "").partition(" ")
    if scheme.lower() != SIGNATURE_SCHEME.lower():
        return None
    return SignedAuthorization.parse(credentials)


def route_template(scope: Scope) -> str:
    """Name the route a request matched by its template (``/api/accounts/{account_id}``), never by the path
    asked for, which holds what the client put in it."""
    template = getattr(scope.get("route"), "path", None)
    return template if isinstance(template, str) else UNMATCHED_ROUTE


def _log_refusal(request: Request, reason: str) -> None:
    """Record a refused request by method, route template, and reason: nothing the request sent."""
    logger.warning("Refused %s %s: %s", request.method, route_template(request.scope), reason)


def _log_acceptance(request: Request, credential: Credential) -> None:
    """Record an accepted request by method, route template, and credential kind (CC7.2): nothing it sent."""
    logger.info("Accepted %s %s: %s", request.method, route_template(request.scope), credential.value)


async def read_signed_body(request: Request) -> bytes:
    """Read the body of ``request`` from the connection, no more than ``MAX_SIGNED_BODY_BYTES`` of it.

    The bytes come from the connection itself, whatever the headers declare, so the signature is checked
    against what was actually sent. Reading stops at the first byte over the limit.

    Raises:
        HTTPException: 413 once more than the limit has arrived.
    """
    received = bytearray()
    async for chunk in request.stream():
        # Why before the copy: a single chunk larger than the limit must not be held even for a moment.
        if len(received) + len(chunk) > MAX_SIGNED_BODY_BYTES:
            _log_refusal(request, "the body is larger than the API accepts")
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, BODY_TOO_LARGE_MESSAGE)
        received.extend(chunk)
    return bytes(received)


def with_body(request: Request, body: bytes, *, content_type: str | None = None) -> Request:
    """Return ``request`` as the route will read it: its body is exactly ``body``.

    What the connection sends after the body (a disconnect) is passed on. With ``content_type``, the
    route sees that type and ``body``'s length instead of what the connection declared: an opened body
    is not what was sent.
    """
    delivered = False

    async def receive() -> Message:
        nonlocal delivered
        if delivered:
            return await request.receive()
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = request.scope
    if content_type is not None:
        kept = [(name, value) for name, value in scope["headers"] if name not in (b"content-type", b"content-length")]
        declared = [
            (b"content-type", content_type.encode("ascii")),
            (b"content-length", str(len(body)).encode("ascii")),
        ]
        scope = {**scope, "headers": [*kept, *declared]}
    return Request(scope, receive)


def open_signed_body(request: Request, proof: Proof, body: bytes, sealer: BodySealer) -> bytes | None:
    """Open the sealed body of a signed request, or return None when there is nothing to open.

    A signed request's body is sealed (``SEALED_CONTENT_TYPE``) for its method, target, challenge, and
    nonce. It is opened only after the signature over the sealed bytes was accepted. A bearer request,
    and a signed request without a body, are handed on as they are.

    Raises:
        HTTPException: 400 if the body of a signed request is not sealed, or does not open for it.
    """
    if proof.signed is None or not body:
        return None
    declared = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if declared != SEALED_CONTENT_TYPE:
        _log_refusal(request, "the body of a signed request is not sealed")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, UNSEALED_BODY_MESSAGE)
    context = request_seal_context(request.method, signed_target(request), proof.signed.challenge, proof.signed.nonce)
    try:
        return sealer.unseal(body, context)
    except SealedBodyError:
        _log_refusal(request, "the sealed body could not be opened")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, SEALED_BODY_NOT_OPENED_MESSAGE) from None


def require_access_token(request: Request, access_token: str, verifier: RequestVerifier, body: bytes) -> Proof:
    """Reject a request that does not prove it holds the dashboard's access token, and say how one did.

    A ``Usagedash`` signature is checked against the request's own method, target, and ``body`` over a
    challenge this process issued, and spends it; a ``Bearer`` header must hold the token itself. Either
    acceptance is recorded by its kind (``_log_acceptance``).

    Raises:
        HTTPException: 401 if the ``Authorization`` header holds neither a valid signature nor the token.
    """
    authorization = request.headers.get("authorization")
    outcome = verifier.verify(authorization, request.method, signed_target(request), body_digest(body))
    if outcome is SignatureOutcome.ACCEPTED:
        _log_acceptance(request, Credential.SIGNATURE)
        return Proof(Credential.SIGNATURE, presented_signature(authorization))
    if outcome is SignatureOutcome.NOT_SIGNED and token_matches(presented_token(authorization), access_token):
        _log_acceptance(request, Credential.BEARER)
        return Proof(Credential.BEARER, None)
    _log_refusal(request, "no valid access token" if outcome is SignatureOutcome.NOT_SIGNED else outcome.value)
    raise HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        NOT_SIGNED_IN_MESSAGE,
        headers={"WWW-Authenticate": ACCEPTED_SCHEMES},
    )


def require_same_origin(request: Request, credential: Credential) -> None:
    """Reject state-changing requests that did not come from the dashboard's own page.

    ``Origin`` is required: browsers send it on every POST and PATCH. The file page's origin is ``null``,
    accepted only from a request that signed itself, because a website cannot send that origin and a
    signature cannot be made without the token. A page served from this origin is accepted with it.

    Raises:
        HTTPException: 403 if the page's own header is missing or the origin is not the page's.
    """
    if request.headers.get("x-requested-with") != CSRF_HEADER_VALUE:
        _log_refusal(request, "the header the dashboard page sends is missing")
        raise HTTPException(status.HTTP_403_FORBIDDEN, "missing X-Requested-With header")
    origin = request.headers.get("origin")
    if origin == FILE_PAGE_ORIGIN and credential is Credential.SIGNATURE:
        return
    if origin != f"{request.url.scheme}://{request.url.netloc}":
        _log_refusal(request, "it did not come from the dashboard page")
        raise HTTPException(status.HTTP_403_FORBIDDEN, "request did not come from the dashboard page")


def guarded_route_class(access_token: str, verifier: RequestVerifier, sealer: BodySealer) -> type[APIRoute]:
    """Return the class of route that checks who is asking before FastAPI reads anything else of a request.

    FastAPI decodes a JSON body before it runs any dependency, so as dependencies these checks let a
    malformed body be answered by a route that had no token. A route of this class runs them first:
    every route needs the token, every route that changes something (any method outside
    ``SAFE_METHODS``) also needs to come from the page, and all routes share ``verifier``, so a challenge
    spent on one is spent for all. The body is read here, bounded; a signed request's body is sealed and
    the route is handed what it opened to (``open_signed_body``, ``with_body``).

    The answer to an accepted signed request is sealed and signed in turn (``seal_answer``), whatever the
    route answered: data, a refusal, a body it could not open, or the plain 500 for a failure it did not
    expect, which is answered here because the sealing and the cross-origin layer have unwound by the
    time ``SealedResponses`` would see it. A refusal of the request itself (token, origin) is neither
    sealed nor signed: there is no accepted exchange to bind it to.
    """

    class GuardedRoute(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
            handle = super().get_route_handler()

            async def guarded(request: Request) -> Response:
                body = await read_signed_body(request)
                proof = require_access_token(request, access_token, verifier, body)
                if request.method not in SAFE_METHODS:
                    require_same_origin(request, proof.credential)
                try:
                    opened = open_signed_body(request, proof, body, sealer)
                    handed = (
                        with_body(request, body)
                        if opened is None
                        else with_body(request, opened, content_type=JSON_CONTENT_TYPE)
                    )
                    answer = await handle(handed)
                except HTTPException as refusal:
                    answer = await http_exception_handler(request, refusal)
                except RequestValidationError as invalid:
                    answer = await _refuse_invalid_request(request, invalid)
                except Exception as error:  # noqa: BLE001 - answered here so the answer is sealed, signed, and reaches the page
                    log_unexpected_failure(request.scope, error)
                    answer = plain_internal_error()
                return seal_answer(answer, access_token, proof.signed, sealer)

            return guarded

    return GuardedRoute


def seal_answer(
    answer: Response, access_token: str, signed: SignedAuthorization | None, sealer: BodySealer
) -> Response:
    """Seal ``answer``'s body for the exchange and sign it, when it answers a signed request; return it.

    The body is sealed for the challenge, the nonce, and the status, declared as ``SEALED_CONTENT_TYPE``,
    and ``response_signature`` covers the sealed bytes. A bearer answer, and one with no body of its own,
    is left as it is.
    """
    body = getattr(answer, "body", None)
    if signed is None or not isinstance(body, bytes):
        return answer
    sealed = sealer.seal(body, answer_seal_context(signed.challenge, signed.nonce, answer.status_code))
    answer.body = sealed
    answer.headers["content-type"] = SEALED_CONTENT_TYPE
    answer.headers["content-length"] = str(len(sealed))
    answer.headers[RESPONSE_SIGNATURE_HEADER] = response_signature(
        access_token, signed.challenge, signed.nonce, answer.status_code, sealed
    )
    return answer


async def _refuse_invalid_request(_: Request, error: Exception) -> JSONResponse:
    """Answer a request that failed validation without repeating any of it.

    FastAPI's own answer quotes the rejected value. Each failure is named here by the part of the
    request it was in and pydantic's name for its kind, both fixed vocabularies.
    """
    failures = error.errors() if isinstance(error, RequestValidationError) else []
    problems = sorted({f"{_request_part(failure['loc'])}: {failure['type']}" for failure in failures})
    return JSONResponse(
        {"detail": INVALID_REQUEST_MESSAGE, "problems": problems}, status.HTTP_422_UNPROCESSABLE_CONTENT
    )


def _request_part(location: tuple[object, ...]) -> str:
    part = location[0] if location else None
    return part if isinstance(part, str) and part in REQUEST_PARTS else "request"


class RecordedHostGuard:
    """Refuse a Host header that is not a loopback name, and record that refusal.

    Starlette answers a foreign Host with a 400 before any route runs, so the route-level logger never
    sees it. The warning holds the method and a fixed reason, written when the response starts so a send
    that fails still leaves it. No redirect: it would leave no record, and no allowed host is a www name.
    """

    def __init__(self, app: ASGIApp, allowed_hosts: list[str]) -> None:
        self._app = app
        self._allowed_hosts = allowed_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        reached_app = False

        async def continue_inward(inner_scope: Scope, inner_receive: Receive, inner_send: Send) -> None:
            nonlocal reached_app
            reached_app = True
            await self._app(inner_scope, inner_receive, inner_send)

        async def send_recording_refusal(message: Message) -> None:
            if scope["type"] == "http" and message["type"] == "http.response.start" and not reached_app:
                _log_host_refusal(scope)
            await send(message)

        guard = TrustedHostMiddleware(continue_inward, allowed_hosts=self._allowed_hosts, www_redirect=False)
        await guard(scope, receive, send_recording_refusal)


def _log_host_refusal(scope: Scope) -> None:
    """Record a host-guard refusal by method, the unmatched route, and a fixed reason.

    The Host header and the path are what a rebinding attempt sends. Neither is written.
    """
    logger.warning("Refused %s %s: %s", scope["method"], route_template(scope), HOST_NOT_ALLOWED)


REACHED_APPLICATION_KEY = "usagedash.reached_application"
"""Set on a request's scope once the cross-origin layer has passed it on to the application."""


class RecordedCorsGuard:
    """The cross-origin layer, with every preflight it turns away recorded.

    A preflight is answered by the middleware itself and never reaches a route; one it answers with a
    failure status is logged by method, unmatched route, and a fixed reason.
    """

    def __init__(self, app: ASGIApp, /, **options: Any) -> None:
        async def mark_and_continue(scope: Scope, receive: Receive, send: Send) -> None:
            scope[REACHED_APPLICATION_KEY] = True
            await app(scope, receive, send)

        self._cors = CORSMiddleware(mark_and_continue, **options)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._cors(scope, receive, send)
            return

        async def send_recording_refusal(message: Message) -> None:
            if (
                message["type"] == "http.response.start"
                and not scope.get(REACHED_APPLICATION_KEY)
                and message["status"] >= status.HTTP_400_BAD_REQUEST
            ):
                logger.warning("Refused %s %s: %s", scope["method"], route_template(scope), PREFLIGHT_NOT_ALLOWED)
            await send(message)

        await self._cors(scope, receive, send_recording_refusal)


def log_unexpected_failure(scope: Scope, error: Exception) -> None:
    """Log a failure nothing expected by its type and where it was raised, never by its message."""
    logger.error(
        "Unexpected failure answering %s %s\n%s",
        scope["method"],
        route_template(scope),
        describe_without_messages(error),
    )


def plain_internal_error() -> JSONResponse:
    """Return the answer to a request that failed unexpectedly: a 500 that says nothing of the failure."""
    return JSONResponse({"detail": INTERNAL_ERROR_MESSAGE}, status.HTTP_500_INTERNAL_SERVER_ERROR)


class SealedResponses:
    """The outermost layer of the application: every response gets the security headers, and no exception leaves.

    A route answers its own failure (``guarded_route_class``); what reaches this layer failed outside a
    route or after the route answered, and is logged by type and location and answered with the plain
    500. A failure after the response has begun cannot be taken back: it is logged and the server closes
    the connection. The 500 is sent once the failure has been handled, not while it is, so no exception
    chain reaches the server's log. What is not an ``Exception`` (a cancellation, a stop) goes on, with
    its chain cleared.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        try:
            await self._answer(scope, receive, send)
        except BaseException as leaving:
            leaving.__context__ = None
            leaving.__cause__ = None
            raise

    async def _answer(self, scope: Scope, receive: Receive, send: Send) -> None:
        response_has_begun = False

        async def send_with_security_headers(message: Message) -> None:
            nonlocal response_has_begun
            if message["type"] == "http.response.start":
                response_has_begun = True
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers.setdefault(name, value)
            await send(message)

        try:
            await self._app(scope, receive, send_with_security_headers)
        except Exception as error:  # noqa: BLE001 - last boundary before the server: log, then answer 500
            log_unexpected_failure(scope, error)
        else:
            return
        if response_has_begun:
            return
        answer = plain_internal_error()
        try:
            await answer(scope, receive, send_with_security_headers)
        except Exception as error:  # noqa: BLE001 - last boundary before the server: the failure is logged
            logger.error(
                "Unexpected failure sending the plain 500 for %s %s\n%s",
                scope["method"],
                route_template(scope),
                describe_without_messages(error),
            )


def _log_cycle_failure(error: BaseException | None) -> None:
    """Log how a cycle ended when it raised. The loop goes on to the next cycle whatever this was."""
    if error is None:
        return
    if isinstance(error, CycleAlreadyRunningError):
        logger.info("Skipping scheduled cycle; one is already running")
    elif isinstance(error, UsageDashError):
        logger.error("Collection cycle failed\n%s", describe_without_messages(error))
    else:
        # Why not logging's own traceback: it prints the exception's message, and an unexpected
        # exception's message can quote what the failing code was handling.
        logger.error("Collection cycle failed unexpectedly\n%s", describe_without_messages(error))


def _log_loop_end(task: asyncio.Task[None]) -> None:
    """Log a collection loop that ended in any way but by being cancelled.

    Why this is a callback on the task: an exception nobody retrieves is logged by asyncio itself,
    message included. Retrieving it here, whoever is or is not awaiting the task, means that never happens.
    """
    error = None if task.cancelled() else task.exception()
    if error is not None:
        logger.error("Collection loop stopped; no further cycle will run\n%s", describe_without_messages(error))


async def _until_done(work: asyncio.Future[Any]) -> None:
    """Wait until ``work`` is done, however many times the waiting task is cancelled meanwhile.

    Cancelling a task does not stop the worker thread it waits for, which goes on writing to the
    database and holding the collection lock.

    Raises:
        asyncio.CancelledError: after ``work`` is done, if the waiting task was cancelled during the wait.
    """
    was_cancelled = False
    while not work.done():
        try:
            await asyncio.wait([work])
        except asyncio.CancelledError:
            was_cancelled = True
    if was_cancelled:
        raise asyncio.CancelledError


class CollectionLoop:
    """Runs collection cycles for as long as the app lives. Every method must be used on the event loop thread.

    ``is_busy`` is true from the moment a cycle is handed to its thread until that thread returns.
    ``last_attempt_ended_at`` is when the latest cycle returned, however it ended, and
    ``seconds_since_last_attempt`` the spacing since, from ``monotonic``: a wall clock set back would
    refuse refreshes for as long as it was set back, and one set forward would allow one at once.
    """

    def __init__(
        self,
        collector: Collector,
        interval_seconds: int,
        clock: Clock,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._collector = collector
        self._interval_seconds = interval_seconds
        self._clock = clock
        self._monotonic = monotonic
        self._wake = asyncio.Event()
        self._is_busy = False
        self._has_stopped = False
        self._last_attempt_ended_at: datetime | None = None
        self._last_attempt_ended_mono: float | None = None

    @property
    def is_busy(self) -> bool:
        return self._is_busy

    @property
    def has_stopped(self) -> bool:
        """Whether ``run`` has ended, by being cancelled or by failing. No cycle runs after that."""
        return self._has_stopped

    @property
    def last_attempt_ended_at(self) -> datetime | None:
        return self._last_attempt_ended_at

    @property
    def seconds_since_last_attempt(self) -> float | None:
        """Seconds of the monotonic clock since the latest cycle returned, or None before any has."""
        if self._last_attempt_ended_mono is None:
            return None
        return self._monotonic() - self._last_attempt_ended_mono

    def run_soon(self) -> None:
        """End the wait between cycles, so the next one starts now."""
        self._wake.set()

    async def run(self) -> None:
        """Run cycles forever, sleeping the interval between them unless woken early.

        A cycle that raises is logged and the next one still runs. Cancelling this coroutine while a
        cycle is in flight waits for that cycle. A wake already set when the loop starts is discarded:
        the first cycle is that request, and leaving the event set would start another at once.
        """
        try:
            # Why: ``run_soon`` before this task is scheduled has already asked for a cycle. That cycle
            # is the first trip through the loop, so the event must not also skip the wait after it.
            self._wake.clear()
            while True:
                await self._run_cycle()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), timeout=self._interval_seconds)
                self._wake.clear()
        finally:
            self._has_stopped = True

    async def _run_cycle(self) -> None:
        worker = asyncio.get_running_loop().run_in_executor(None, self._collector.run_cycle)
        self._is_busy = True
        worker.add_done_callback(self._cycle_returned)
        await _until_done(worker)

    def _cycle_returned(self, worker: asyncio.Future[CycleReport]) -> None:
        """Record that a cycle's thread has returned, and log how it ended if it raised.

        Why this is a callback on the worker and not code after the wait for it: the outcome is then
        retrieved whatever became of the task that was waiting, so asyncio never logs it itself, with
        the exception's message. A worker that was cancelled never ran its cycle.
        """
        self._is_busy = False
        self._last_attempt_ended_at = self._clock()
        self._last_attempt_ended_mono = self._monotonic()
        _log_cycle_failure(None if worker.cancelled() else worker.exception())


def create_app(
    settings: Settings,
    database: Database,
    collector: Collector | None,
    clock: Clock,
    monotonic: Callable[[], float] = time.monotonic,
    *,
    access_token: str,
    run_collector: bool = True,
) -> FastAPI:
    """Build the application. With ``run_collector`` the collection loop runs for the app's lifetime.

    ``access_token`` is the secret every ``/api`` request must present.

    Raises:
        ValueError: if ``access_token`` is not well formed. An empty one would match a request that
            presents no token at all, and a short one could be guessed.
    """
    if not is_well_formed_token(access_token):
        raise ValueError("the dashboard cannot run without a well-formed access token")
    collection = (
        CollectionLoop(collector, settings.poll_interval_seconds, clock, monotonic)
        if run_collector and collector is not None
        else None
    )

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        task = None if collection is None else asyncio.create_task(collection.run())
        if task is not None:
            task.add_done_callback(_log_loop_end)
        try:
            yield
        finally:
            if task is not None:
                # Why waited for: the loop does not finish being cancelled until a cycle in flight has
                # returned, so shutdown completes with no collection still using the database. That
                # holds even when shutdown itself is cancelled while it waits.
                task.cancel()
                await _until_done(task)

    # Why no slash redirect: FastAPI answers a path the route does not declare with a 307
    # before GuardedRoute runs, and that answer copies the path into Location. The route
    # that does exist still checks the token and the page.
    app = FastAPI(
        title="AI usage dashboard",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        redirect_slashes=False,
        telemetry=TELEMETRY_OFF,
    )
    # Why the router's route class: every route added under /api is refused without proof of the token,
    # and every one that changes something unless it came from the page, by construction and before any
    # part of the request is decoded or validated (see ``guarded_route_class``).
    verifier = RequestVerifier(access_token)
    sealer = BodySealer(access_token)
    api = APIRouter(prefix=API_PREFIX, route_class=guarded_route_class(access_token, verifier, sealer))

    app.add_exception_handler(RequestValidationError, _refuse_invalid_request)
    # Why in this order: the middleware added last runs first. A request with a foreign, empty, or
    # malformed Host header is refused and recorded before any route looks at it, and before the
    # cross-origin layer answers a preflight; the layer around everything, that refusal included, is the
    # one that adds the security headers and lets no exception reach the server. The cross-origin layer
    # answers the page's preflights, recording the ones it turns away, and marks responses for the page's
    # ``null`` origin alone; it holds no credentials (the page sends none) and it does not check who is
    # asking, the route does. The page reads the answer's signature, so that header is exposed to it.
    app.add_middleware(
        RecordedCorsGuard,
        allow_origins=[FILE_PAGE_ORIGIN],
        allow_methods=CORS_ALLOWED_METHODS,
        allow_headers=CORS_ALLOWED_HEADERS,
        expose_headers=[RESPONSE_SIGNATURE_HEADER],
    )
    app.add_middleware(RecordedHostGuard, allowed_hosts=ALLOWED_HOSTS)
    app.add_middleware(SealedResponses)

    @app.get(HEALTH_PATH)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(CHALLENGE_PATH)
    def issue_challenge() -> dict[str, str]:
        """Issue the value the page's next request must be signed over.

        The route is not under ``/api``: the page has nothing to prove yet when it asks. The value is
        random, held by this process alone, and spent by the one request signed over it, so handing it
        out reveals nothing and authorises nothing by itself.
        """
        return {"challenge": verifier.issue()}

    @api.get("/state")
    def get_state() -> dict[str, Any]:
        return build_state(database, settings, collector, clock())

    @api.get("/tokens")
    def get_tokens(
        range_key: Annotated[RangeKey, Query(alias="range")] = "7d",
        tz: Annotated[str, Query(min_length=1, max_length=MAX_TIME_ZONE_NAME_LENGTH)] = "UTC",
    ) -> dict[str, Any]:
        zone = _time_zone(tz)
        with database.read() as conn:
            return _token_report_json(build_token_report(conn, db.list_accounts(conn), range_key, zone, clock()))

    @api.post("/refresh", status_code=status.HTTP_202_ACCEPTED)
    async def refresh() -> dict[str, str]:
        # Why async: the collection loop's state belongs to the event loop thread.
        if collection is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "collector is not running in this process")
        if collection.has_stopped:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "collection has stopped in this process; its log says why"
            )
        if collection.is_busy:
            # Why this and not the collector's own flag: a cycle handed to its worker has not yet said so.
            return {"status": "already_running"}
        # Why the attempt and not the last success: a cycle that failed must be spaced out like any
        # other, or every refresh after it would start another one at once.
        since = collection.seconds_since_last_attempt
        if since is not None and since < MIN_REFRESH_SPACING.total_seconds():
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "refresh attempted moments ago",
                headers={"Retry-After": str(int(MIN_REFRESH_SPACING.total_seconds()))},
            )
        collection.run_soon()
        return {"status": "scheduled"}

    @api.patch("/accounts/{account_id}")
    def rename_account(
        account_id: Annotated[int, PathParameter(ge=1, le=MAX_ROW_ID)], body: RenameRequest
    ) -> dict[str, Any]:
        with database.connect() as conn:
            if not db.rename_account(conn, account_id, body.label):
                raise HTTPException(status.HTTP_404_NOT_FOUND, "no such account")
        return {"id": _text_id(account_id), "label": body.label}

    app.include_router(api)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
