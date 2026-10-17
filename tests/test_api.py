import asyncio
import contextlib
import gc
import json
import logging
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, get_type_hints
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.telemetry import TelemetryConfig
from fastapi.testclient import TestClient
from opentelemetry import _logs as otel_logs
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace as otel_trace

from tests.factories import PERSONA_EMAIL, make_identity, make_reading, make_window
from tests.payloads import codex_usage_payload
from tests.seeds import seed_accounts_and_usage
from usagedash import access, api, db
from usagedash.access import (
    SEALED_CONTENT_TYPE,
    BodySealer,
    RequestVerifier,
    answer_seal_context,
    fresh_nonce,
    request_seal_context,
    response_signature,
    sign_request,
)
from usagedash.api import MIN_REFRESH_SPACING, CollectionLoop, create_app, presented_token, signed_target
from usagedash.collector import Collector, CycleAlreadyRunningError, SourceBundle
from usagedash.config import Settings, load_settings
from usagedash.db import Database, MessageUsage
from usagedash.errors import ConfigError, CredentialExpiredError, SealedBodyError
from usagedash.models import AccountIdentity, Provider, QuotaReading, TokenCounts
from usagedash.private import open_private_file
from usagedash.providers import codex
from usagedash.providers.base import NoQuotaDataError

BASE_URL = "http://127.0.0.1:8787"
ACCESS_TOKEN = "synthetic-access-token-" + "x" * 20
PAGE_HEADERS = {"X-Requested-With": "usagedash", "Origin": BASE_URL}
FILE_PAGE_HEADERS = {"X-Requested-With": "usagedash", "Origin": "null"}
"""What a browser sends with a change made from the page opened as a local file."""
OTHER_TOKEN = "synthetic-second-token-" + "y" * 20
RENAME_BODY = b'{"label": "Renamed"}'
"""A rename body as the page sends it. Signed bytes must be the bytes sent, so tests send these bytes."""
JSON_CONTENT = {"Content-Type": "application/json"}
WAIT_TIMEOUT_SECONDS = 5.0
NOW = datetime(2026, 10, 3, 14, 0, tzinfo=UTC)
WAIT_POLL_SECONDS = 0.01


def wait_until(condition: Callable[[], bool]) -> None:
    """Block until ``condition`` holds, failing the test if it does not within the timeout."""
    deadline = time.monotonic() + WAIT_TIMEOUT_SECONDS
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail("condition was not met in time")
        time.sleep(WAIT_POLL_SECONDS)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return load_settings(env={"USAGEDASH_DATA_DIR": str(tmp_path / "data")}, home=tmp_path)


@pytest.fixture
def seeded(database: Database, now: datetime) -> dict[str, int]:
    return seed_accounts_and_usage(database, now)


def issued_challenge(browser: TestClient) -> str:
    """Ask the dashboard ``browser`` talks to for the challenge its next request is signed over."""
    response = browser.get("/challenge")
    assert response.status_code == 200
    challenge = response.json()["challenge"]
    assert isinstance(challenge, str)
    return challenge


def signed(
    browser: TestClient,
    method: str,
    target: str,
    *,
    token: str = ACCESS_TOKEN,
    body: bytes = b"",
    challenge: str | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """The header the page sends for one request: a signature over that method, target, and body, over the
    challenge the dashboard just issued to ``browser`` (or ``challenge`` when given) and a nonce (drawn when
    not given)."""
    chosen = issued_challenge(browser) if challenge is None else challenge
    return {"Authorization": sign_request(token, method, target, chosen, body, nonce)}


def sealed(
    browser: TestClient,
    method: str,
    target: str,
    body: bytes,
    *,
    token: str = ACCESS_TOKEN,
    challenge: str | None = None,
    nonce: str | None = None,
) -> tuple[dict[str, str], bytes]:
    """The headers and the content the page sends for one request that carries ``body``: the body sealed for
    that method, target, challenge, and nonce, its content type, and a signature over the sealed bytes."""
    chosen = issued_challenge(browser) if challenge is None else challenge
    drawn = fresh_nonce() if nonce is None else nonce
    content = BodySealer(token).seal(body, request_seal_context(method, target, chosen, drawn))
    headers = {
        "Content-Type": SEALED_CONTENT_TYPE,
        **signed(browser, method, target, token=token, body=content, challenge=chosen, nonce=drawn),
    }
    return headers, content


def opened(response: Any, challenge: str, nonce: str, *, token: str = ACCESS_TOKEN) -> Any:
    """What a sealed answer to the request signed over ``challenge`` and ``nonce`` holds, decoded as JSON."""
    assert response.headers["content-type"] == SEALED_CONTENT_TYPE
    context = answer_seal_context(challenge, nonce, response.status_code)
    return json.loads(BodySealer(token).unseal(response.content, context))


def signed_in(test_client: TestClient) -> TestClient:
    test_client.headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"
    return test_client


def _state_at(settings: Settings, database: Database, moment: datetime) -> dict[str, Any]:
    """Read ``/api/state`` with the clock stopped at ``moment`` and no collector running."""
    app = create_app(settings, database, None, lambda: moment, access_token=ACCESS_TOKEN, run_collector=False)
    with TestClient(app, base_url=BASE_URL) as test_client:
        return signed_in(test_client).get("/api/state").json()


@pytest.fixture
def anonymous(settings: Settings, database: Database, now: datetime) -> Iterator[TestClient]:
    """A browser that has not been signed in."""
    app = create_app(settings, database, None, lambda: now, access_token=ACCESS_TOKEN, run_collector=False)
    with TestClient(app, base_url=BASE_URL, follow_redirects=False) as test_client:
        yield test_client


@pytest.fixture
def client(anonymous: TestClient) -> TestClient:
    """A browser signed in with the dashboard's access token."""
    return signed_in(anonymous)


class TestState:
    def test_reports_each_provider_with_accounts_windows_advice_and_history(
        self, client: TestClient, seeded: dict[str, int]
    ) -> None:
        state = client.get("/api/state").json()

        assert [provider["provider"] for provider in state["providers"]] == ["claude", "codex", "grok"]
        claude = state["providers"][0]
        assert [account["label"] for account in claude["accounts"]] == ["Personal", "Work team"]
        active, other = claude["accounts"]
        assert (active["is_active"], active["headroom_percent"]) == (True, 8)
        assert active["windows"][0] == {
            "key": "five_hour",
            "label": "5-hour session",
            "blocking": True,
            "window_seconds": 18000,
            "used_percent": 92,
            "effective_percent": 92,
            "state": "critical",
            "resets_at": "2026-10-03T16:00:00+00:00",
            "reset_passed": False,
            "as_of": "2026-10-03T14:00:00+00:00",
            "freshness": "live",
            "pace_exhausts_at": None,
        }
        assert (other["is_active"], other["windows"][0]["effective_percent"]) == (False, 0)
        assert other["windows"][0]["freshness"] == "projected"
        assert claude["advice"]["action"] == "verify"
        assert claude["advice"]["target_account_id"] == str(seeded["other"])
        assert claude["advice"]["as_of"] == "2026-10-02T14:00:00+00:00"
        assert "has reset since it was last read" in claude["advice"]["summary"]
        assert {series["account_id"] for series in state["history"]} == {str(seeded["active"]), str(seeded["other"])}
        assert state["expected_accounts_per_provider"] == 2
        assert state["collector"] == {"running": False, "last_finished_at": None}

    def test_classifies_source_problems_by_severity(
        self, client: TestClient, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn:
            db.record_status(conn, "a", "quota", now, None)
            db.record_status(conn, "b", "quota", now, NoQuotaDataError("nothing yet"))
            db.record_status(conn, "c", "quota", now, CredentialExpiredError("expired"))
            db.record_status(conn, "d", "quota", now, RuntimeError("defect"))

        sources = {source["source_id"]: source for source in client.get("/api/state").json()["sources"]}

        assert {key: value["severity"] for key, value in sources.items()} == {
            "a": "ok",
            "b": "info",
            "c": "warning",
            "d": "error",
        }
        assert sources["d"]["message"] == "defect"
        assert sources["a"]["last_success_at"] == "2026-10-03T14:00:00+00:00"

    def test_an_account_codex_reports_as_at_its_limit_is_shown_as_out_of_quota(
        self, client: TestClient, database: Database, now: datetime
    ) -> None:
        """The whole path: a payload whose windows show room but whose flag says the limit is reached."""
        payload = codex_usage_payload(now, primary_percent=38)
        payload["rate_limit"]["limit_reached"] = True
        _, windows, facts = codex.parse_usage(payload)
        identity = make_identity(Provider.CODEX, "workspace-a", "Account A")
        with database.connect() as conn:
            account_id = db.upsert_account(conn, identity, now)
            db.record_reading(conn, account_id, QuotaReading(identity, windows, now, "api", facts))
            db.set_source_state(conn, "codex:home", Provider.CODEX, 0, account_id, now)

        provider = client.get("/api/state").json()["providers"][1]

        account = provider["accounts"][0]
        assert (account["available_now"], account["headroom_percent"], account["next_available_at"]) == (False, 0, None)
        assert [(window["key"], window["state"]) for window in account["windows"]] == [
            ("weekly", "ok"),
            ("reported_limit", "exhausted"),
        ]
        assert provider["advice"]["action"] == "hold"
        assert "Account A is out of quota" in provider["advice"]["summary"]

    def test_history_starts_with_the_step_in_effect_a_week_ago_reported_at_the_start_of_the_range(
        self, client: TestClient, database: Database, now: datetime
    ) -> None:
        """Its last confirmation is before the range, so inside the range it is known at the start only."""
        identity = make_identity()
        with database.connect() as conn:
            account_id = db.upsert_account(conn, identity, now)
            for days_ago, percent in ((9, 20), (8, 35), (2, 60)):
                reading = make_reading(identity, now - timedelta(days=days_ago), make_window(used_percent=percent))
                db.record_reading(conn, account_id, reading)

        (series,) = client.get("/api/state").json()["history"]

        assert (series["account_id"], series["window_key"]) == (str(account_id), "five_hour")
        assert [(step["used_percent"], step["from"], step["to"]) for step in series["steps"]] == [
            (35, "2026-09-26T14:00:00+00:00", "2026-09-26T14:00:00+00:00"),
            (60, "2026-10-01T14:00:00+00:00", "2026-10-01T14:00:00+00:00"),
        ]

    def test_a_step_inside_the_week_stays_when_the_local_clock_springs_forward(
        self, settings: Settings, database: Database
    ) -> None:
        """Seven days back from a wall clock lands an hour late, so this step would be pushed out of the range."""
        zone = ZoneInfo("America/New_York")
        moment = datetime(2026, 3, 8, 3, 0, 5, tzinfo=zone)
        observed = datetime(2026, 3, 1, 7, 30, tzinfo=UTC)
        identity = make_identity()
        with database.connect() as conn:
            account_id = db.upsert_account(conn, identity, observed)
            db.record_reading(conn, account_id, make_reading(identity, observed, make_window(used_percent=40)))

        (series,) = _state_at(settings, database, moment)["history"]

        assert series["steps"][0]["from"] == "2026-03-01T07:30:00+00:00"
        assert series["steps"][0]["used_percent"] == 40

    def test_a_step_from_before_the_week_stays_out_when_the_local_clock_falls_back(
        self, settings: Settings, database: Database
    ) -> None:
        """Seven days back from a wall clock lands an hour early, so this step would be pulled into the range."""
        zone = ZoneInfo("America/New_York")
        moment = datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=zone)
        observed = datetime(2026, 10, 25, 6, 0, tzinfo=UTC)
        identity = make_identity()
        with database.connect() as conn:
            account_id = db.upsert_account(conn, identity, observed)
            db.record_reading(conn, account_id, make_reading(identity, observed, make_window(used_percent=15)))

        (series,) = _state_at(settings, database, moment)["history"]

        assert series["steps"][0]["from"] == "2026-10-25T06:30:00+00:00"
        assert series["steps"][0]["used_percent"] == 15

    def test_an_empty_database_still_lists_every_provider(self, client: TestClient) -> None:
        state = client.get("/api/state").json()
        assert [len(provider["accounts"]) for provider in state["providers"]] == [0, 0, 0]
        assert state["history"] == []


class TestTokenEndpoint:
    def test_the_endpoint_validates_range_and_time_zone(self, client: TestClient, seeded: dict[str, int]) -> None:
        assert client.get("/api/tokens", params={"range": "7d", "tz": "America/New_York"}).json()["timezone"] == (
            "America/New_York"
        )
        assert client.get("/api/tokens", params={"range": "90d"}).status_code == 422
        assert client.get("/api/tokens", params={"tz": "Mars/Olympus"}).status_code == 422
        assert client.get("/api/tokens", params={"tz": "../etc/passwd"}).status_code == 422


class TestTimeZoneLookup:
    @pytest.mark.parametrize(
        "name",
        ["Mars/Olympus", "../etc/passwd", "/etc/passwd", "America", "America/", "UTC/../UTC", "utc ", "x" * 64],
    )
    def test_a_name_that_is_not_a_time_zone_is_refused_without_being_echoed(
        self, client: TestClient, seeded: dict[str, int], name: str
    ) -> None:
        """A name can fail the lookup as a missing key, a malformed key, or a directory; all are one answer."""
        response = client.get("/api/tokens", params={"tz": name})

        assert response.status_code == 422
        assert response.json() == {"detail": "unknown time zone"}

    @pytest.mark.parametrize("name", ["", "x" * 65])
    def test_a_name_of_an_impossible_length_is_refused(self, client: TestClient, name: str) -> None:
        assert client.get("/api/tokens", params={"tz": name}).status_code == 422

    @pytest.mark.parametrize("name", ["UTC", "America/New_York", "Asia/Kolkata", "Australia/Lord_Howe"])
    def test_real_zones_are_accepted_including_ones_offset_by_a_fraction_of_an_hour(
        self, client: TestClient, seeded: dict[str, int], name: str
    ) -> None:
        response = client.get("/api/tokens", params={"range": "7d", "tz": name})

        assert response.status_code == 200
        assert response.json()["timezone"] == name


class TestReadsNeverWaitForAWriter:
    def test_the_quota_view_and_token_report_are_served_while_a_write_transaction_is_open(
        self, client: TestClient, database: Database, seeded: dict[str, int]
    ) -> None:
        """A collection cycle holds the write lock for each of its writes; the page must not stall behind it."""
        with database.connect() as writer:
            db.rename_account(writer, seeded["active"], "Renamed but not committed")

            state = client.get("/api/state")
            tokens = client.get("/api/tokens")

        assert (state.status_code, tokens.status_code) == (200, 200)
        labels = [account["label"] for provider in state.json()["providers"] for account in provider["accounts"]]
        assert "Renamed but not committed" not in labels


class TestAccessToken:
    @pytest.mark.parametrize("path", ["/api/state", "/api/tokens"])
    def test_account_data_is_not_served_without_the_token(
        self, anonymous: TestClient, seeded: dict[str, int], path: str
    ) -> None:
        response = anonymous.get(path)

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Usagedash, Bearer"
        assert "usagedash open" in response.json()["detail"]
        assert "Personal" not in response.text
        assert PERSONA_EMAIL not in response.text

    @pytest.mark.parametrize("path", ["/", "/static/app.js", "/static/access.js", "/static/styles.css"])
    def test_the_page_shell_is_served_without_the_token_and_holds_no_account_data(
        self, anonymous: TestClient, seeded: dict[str, int], path: str
    ) -> None:
        response = anonymous.get(path)

        assert response.status_code == 200
        assert "Personal" not in response.text
        assert PERSONA_EMAIL not in response.text
        assert ACCESS_TOKEN not in response.text

    def test_the_health_check_needs_no_token_and_reveals_nothing(self, anonymous: TestClient) -> None:
        assert anonymous.get("/healthz").json() == {"status": "ok"}

    def test_the_listener_proof_route_is_gone(self, anonymous: TestClient) -> None:
        """The page never sends the token, so it no longer asks a listener to prove itself first."""
        assert anonymous.get("/access-proof", params={"nonce": "n" * 32}).status_code == 404

    def test_a_state_changing_request_without_the_token_is_refused(
        self, anonymous: TestClient, seeded: dict[str, int], database: Database
    ) -> None:
        response = anonymous.patch(
            f"/api/accounts/{seeded['active']}", json={"label": "Hijacked"}, headers=PAGE_HEADERS
        )

        assert response.status_code == 401
        with database.connect() as conn:
            assert db.account_label(conn, seeded["active"]) == "Personal"

    def test_the_refresh_endpoint_is_refused_without_the_token(self, anonymous: TestClient) -> None:
        assert anonymous.post("/api/refresh", headers=PAGE_HEADERS).status_code == 401

    @pytest.mark.parametrize(
        "authorization",
        [
            "",
            "Bearer",
            "Bearer ",
            "Bearer wrong-token",
            f"Bearer {ACCESS_TOKEN}x",
            f"Bearer {ACCESS_TOKEN[:-1]}",
            f"Basic {ACCESS_TOKEN}",
            f"Token {ACCESS_TOKEN}",
            ACCESS_TOKEN,
        ],
    )
    def test_a_missing_wrong_or_differently_presented_token_is_refused(
        self, anonymous: TestClient, authorization: str
    ) -> None:
        assert anonymous.get("/api/state", headers={"Authorization": authorization}).status_code == 401

    @pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
    def test_the_token_is_accepted_in_a_bearer_header_whatever_the_scheme_s_case(
        self, anonymous: TestClient, scheme: str
    ) -> None:
        response = anonymous.get("/api/state", headers={"Authorization": f"{scheme} {ACCESS_TOKEN}"})
        assert response.status_code == 200

    def test_the_token_is_not_accepted_from_a_cookie(self, anonymous: TestClient) -> None:
        """Cookie sign-in was removed: a browser sends a loopback cookie to every port on the host."""
        for cookie_name in ("usagedash_session", "Authorization", "access"):
            anonymous.cookies.set(cookie_name, ACCESS_TOKEN)

        assert anonymous.get("/api/state").status_code == 401

    def test_the_token_is_not_accepted_in_a_request_url(self, anonymous: TestClient) -> None:
        """The cookie-era ``/auth?token=`` exchange was removed with it; a URL is logged and remembered."""
        assert anonymous.get("/api/state", params={"token": ACCESS_TOKEN, "access": ACCESS_TOKEN}).status_code == 401
        assert anonymous.get("/auth", params={"token": ACCESS_TOKEN}).status_code == 404

    def test_no_response_sets_a_cookie(self, client: TestClient, seeded: dict[str, int]) -> None:
        for path in ("/", "/healthz", "/api/state", "/api/tokens", "/static/app.js"):
            assert "set-cookie" not in client.get(path).headers

    @pytest.mark.parametrize("token", ["", "short", "x" * 42, ACCESS_TOKEN + " ", ACCESS_TOKEN[:-1] + "é"])
    def test_an_app_cannot_be_built_with_a_token_that_is_empty_guessable_or_malformed(
        self, settings: Settings, database: Database, now: datetime, token: str
    ) -> None:
        """An empty token would match a request that presents none at all."""
        with pytest.raises(ValueError, match="without a well-formed access token"):
            create_app(settings, database, None, lambda: now, access_token=token, run_collector=False)

    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            (None, ""),
            ("", ""),
            ("Bearer", ""),
            ("Bearer abc", "abc"),
            ("bearer   abc  ", "abc"),
            ("Basic abc", ""),
            ("abc", ""),
        ],
    )
    def test_only_a_bearer_header_yields_a_token(self, header: str | None, expected: str) -> None:
        assert presented_token(header) == expected


class TestRequestProtection:
    @pytest.mark.parametrize(
        ("host", "method"),
        [
            ("attacker.example", "GET"),
            ("", "GET"),
            ("[broken", "GET"),
            ("attacker.example", "POST"),
        ],
    )
    def test_a_foreign_empty_or_malformed_host_is_refused_and_recorded_without_the_header_or_the_path(
        self,
        settings: Settings,
        database: Database,
        now: datetime,
        caplog: pytest.LogCaptureFixture,
        host: str,
        method: str,
    ) -> None:
        """The token does not make a bad Host acceptable, and the record does not repeat what it sent."""
        app = create_app(settings, database, None, lambda: now, access_token=ACCESS_TOKEN, run_collector=False)
        with (
            TestClient(app, base_url=BASE_URL) as rebinding_client,
            caplog.at_level(logging.WARNING, logger="usagedash.api"),
        ):
            response = signed_in(rebinding_client).request(method, "/api/state", headers={"Host": host})

        assert response.status_code == 400
        assert response.text == "Invalid host header"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert [record.getMessage() for record in caplog.records] == [
            f"Refused {method} (no route): the host is not allowed"
        ]
        assert all(record.levelno == logging.WARNING for record in caplog.records)
        if host:
            assert host not in caplog.text
        assert "/api/state" not in caplog.text

    def test_localhost_is_accepted_as_a_host_name(
        self, settings: Settings, database: Database, now: datetime, caplog: pytest.LogCaptureFixture
    ) -> None:
        app = create_app(settings, database, None, lambda: now, access_token=ACCESS_TOKEN, run_collector=False)
        with (
            TestClient(app, base_url="http://localhost:8787") as local_client,
            caplog.at_level(logging.WARNING, logger="usagedash.api"),
        ):
            denied = local_client.get("/api/state")
            assert signed_in(local_client).get("/healthz").json() == {"status": "ok"}

        assert denied.status_code == 401
        assert [record.getMessage() for record in caplog.records] == ["Refused GET /api/state: no valid access token"]
        assert "the host is not allowed" not in caplog.text

    @pytest.mark.parametrize(
        "failure",
        [pytest.param(OSError("send failed"), id="oserror"), pytest.param(asyncio.CancelledError(), id="cancelled")],
    )
    def test_a_host_refusal_is_recorded_when_sending_the_response_fails_or_is_cancelled(
        self, caplog: pytest.LogCaptureFixture, failure: BaseException
    ) -> None:
        """The audit record does not depend on the 400 reaching the client."""
        self._refuse_while_sending(caplog, host="attacker.example", failure=failure, fail_on="http.response.start")

    def test_a_host_refusal_is_recorded_once_when_the_body_send_fails(self, caplog: pytest.LogCaptureFixture) -> None:
        self._refuse_while_sending(
            caplog, host="attacker.example", failure=OSError("body failed"), fail_on="http.response.body"
        )

    def test_a_failed_send_for_an_allowed_host_is_not_recorded_as_a_host_refusal(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="usagedash.api"), pytest.raises(OSError, match="send failed"):
            asyncio.run(
                self._guard()(
                    self._scope("localhost"),
                    self._receive,
                    self._failing_send(OSError("send failed"), "http.response.start"),
                )
            )
        assert "the host is not allowed" not in caplog.text

    def _refuse_while_sending(
        self, caplog: pytest.LogCaptureFixture, *, host: str, failure: BaseException, fail_on: str
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="usagedash.api"), pytest.raises(type(failure)):
            asyncio.run(self._guard()(self._scope(host), self._receive, self._failing_send(failure, fail_on)))
        assert [record.getMessage() for record in caplog.records] == ["Refused GET (no route): the host is not allowed"]
        assert host not in caplog.text
        assert "/api/state" not in caplog.text

    @staticmethod
    def _guard() -> api.RecordedHostGuard:
        async def endpoint(scope: api.Scope, receive: api.Receive, send: api.Send) -> None:
            del scope, receive
            await send({"type": "http.response.start", "status": 200, "headers": []})

        return api.RecordedHostGuard(endpoint, api.ALLOWED_HOSTS)

    @staticmethod
    def _scope(host: str) -> dict[str, object]:
        return {
            "type": "http",
            "method": "GET",
            "path": "/api/state",
            "headers": [(b"host", host.encode())],
            "scheme": "http",
            "server": ("127.0.0.1", 8787),
            "client": ("127.0.0.1", 9),
            "query_string": b"",
            "http_version": "1.1",
        }

    @staticmethod
    async def _receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    @staticmethod
    def _failing_send(failure: BaseException, fail_on: str) -> Callable[..., Any]:
        async def send(message: dict[str, object]) -> None:
            if message["type"] == fail_on:
                raise failure

        return send

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"Origin": BASE_URL},
            {"X-Requested-With": "usagedash"},
            {"X-Requested-With": "usagedash", "Origin": "http://attacker.example"},
            {"X-Requested-With": "usagedash", "Origin": "http://127.0.0.1:9999"},
            {"X-Requested-With": "other", "Origin": BASE_URL},
        ],
    )
    def test_a_state_changing_request_must_carry_the_custom_header_and_the_page_s_own_origin(
        self, client: TestClient, seeded: dict[str, int], database: Database, headers: dict[str, str]
    ) -> None:
        response = client.patch(f"/api/accounts/{seeded['active']}", json={"label": "Hijacked"}, headers=headers)

        assert response.status_code == 403
        with database.connect() as conn:
            assert db.account_label(conn, seeded["active"]) == "Personal"

    def test_every_response_carries_the_security_headers(self, client: TestClient) -> None:
        for path in ("/", "/api/state", "/static/app.js"):
            headers = client.get(path).headers
            assert "script-src 'self'" in headers["content-security-policy"]
            assert "frame-ancestors 'none'" in headers["content-security-policy"]
            assert headers["x-content-type-options"] == "nosniff"
            assert headers["cache-control"] == "no-store"

    def test_the_refusal_sent_to_an_unsigned_browser_carries_the_security_headers_too(
        self, anonymous: TestClient
    ) -> None:
        refusal = anonymous.get("/api/state")

        assert refusal.status_code == 401
        assert refusal.headers["x-content-type-options"] == "nosniff"
        assert refusal.headers["cache-control"] == "no-store"

    def test_no_interactive_api_documentation_is_exposed(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


class TestSignedRequests:
    """The page proves it holds the token by signing each request over a challenge the dashboard issued."""

    def test_the_challenge_route_needs_no_token_and_issues_a_fresh_value_each_time(self, anonymous: TestClient) -> None:
        first = anonymous.get("/challenge")
        second = anonymous.get("/challenge")

        assert first.status_code == 200
        assert list(first.json()) == ["challenge"]
        assert first.json()["challenge"] != second.json()["challenge"]
        assert len(first.json()["challenge"]) == 43
        assert ACCESS_TOKEN not in first.text
        assert "set-cookie" not in first.headers
        assert first.headers["cache-control"] == "no-store"

    def test_a_request_signed_over_an_issued_challenge_for_its_own_method_and_target_is_served(
        self, anonymous: TestClient, seeded: dict[str, int]
    ) -> None:
        challenge, nonce = issued_challenge(anonymous), fresh_nonce()
        proof = signed(anonymous, "GET", "/api/state", challenge=challenge, nonce=nonce)

        response = anonymous.get("/api/state", headers=proof)

        assert response.status_code == 200
        assert "providers" in opened(response, challenge, nonce)
        assert ACCESS_TOKEN not in response.request.headers["authorization"]

    def test_the_query_is_part_of_the_signed_target(self, anonymous: TestClient, seeded: dict[str, int]) -> None:
        target = "/api/tokens?range=24h&tz=America%2FNew_York"

        assert anonymous.get(target, headers=signed(anonymous, "GET", target)).status_code == 200
        assert anonymous.get(target, headers=signed(anonymous, "GET", "/api/tokens")).status_code == 401
        assert anonymous.get("/api/tokens?range=7d", headers=signed(anonymous, "GET", target)).status_code == 401

    def test_a_signed_change_from_the_file_page_is_accepted(
        self, anonymous: TestClient, seeded: dict[str, int], database: Database
    ) -> None:
        target = f"/api/accounts/{seeded['active']}"

        headers, content = sealed(anonymous, "PATCH", target, RENAME_BODY)

        response = anonymous.patch(target, content=content, headers={**FILE_PAGE_HEADERS, **headers})

        assert response.status_code == 200
        with database.connect() as conn:
            assert db.account_label(conn, seeded["active"]) == "Renamed"

    @pytest.mark.parametrize(
        "describe",
        [
            pytest.param(lambda browser: signed(browser, "POST", "/api/state"), id="another method"),
            pytest.param(lambda browser: signed(browser, "GET", "/api/tokens"), id="another target"),
            pytest.param(lambda browser: signed(browser, "GET", "/api/state", token=OTHER_TOKEN), id="another token"),
            pytest.param(
                lambda browser: signed(browser, "GET", "/api/state", challenge="synthetic-challenge-" + "n" * 23),
                id="a challenge never issued",
            ),
            pytest.param(lambda browser: {"Authorization": "Usagedash"}, id="no credentials"),
            pytest.param(lambda browser: {"Authorization": "Usagedash abc"}, id="not three parts"),
            pytest.param(
                lambda browser: {"Authorization": f"Usagedash {issued_challenge(browser)}.{fresh_nonce()}.{'0' * 64}"},
                id="wrong digest",
            ),
            pytest.param(
                lambda browser: {"Authorization": f"Usagedash {issued_challenge(browser)}.{'0' * 64}"},
                id="no nonce",
            ),
            pytest.param(
                lambda browser: {"Authorization": f"Usagedash {ACCESS_TOKEN}"}, id="the token where a signature goes"
            ),
        ],
    )
    def test_a_signature_that_does_not_fit_this_request_to_this_process_is_refused(
        self, anonymous: TestClient, seeded: dict[str, int], describe: Callable[[TestClient], dict[str, str]]
    ) -> None:
        response = anonymous.get("/api/state", headers=describe(anonymous))

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Usagedash, Bearer"
        assert "usagedash open" in response.json()["detail"]
        assert PERSONA_EMAIL not in response.text

    def test_a_challenge_another_process_issued_is_refused(
        self, anonymous: TestClient, settings: Settings, database: Database, now: datetime, seeded: dict[str, int]
    ) -> None:
        """A signature captured by whatever held the port fits that process and not the dashboard."""
        other_app = create_app(settings, database, None, lambda: now, access_token=ACCESS_TOKEN, run_collector=False)
        with TestClient(other_app, base_url=BASE_URL) as other_process:
            header = signed(other_process, "GET", "/api/state")
            assert other_process.get("/api/state", headers=header).status_code == 200
            header_for_here = signed(other_process, "GET", "/api/state")

        assert anonymous.get("/api/state", headers=header_for_here).status_code == 401

    def test_every_answer_to_a_signed_request_is_signed_over_its_challenge_status_and_body(
        self, anonymous: TestClient, seeded: dict[str, int]
    ) -> None:
        """Whatever the route answered, data or a refusal, the page can tell the answer is the dashboard's."""
        page = {**FILE_PAGE_HEADERS, **JSON_CONTENT}
        exchanges = [
            ("GET", "/api/state", b"", {}, 200),
            ("GET", "/api/tokens?range=24h&tz=America%2FNew_York", b"", {}, 200),
            ("GET", "/api/tokens?range=24h&tz=Nowhere%2FNone", b"", {}, 422),
            ("PATCH", f"/api/accounts/{seeded['active']}", RENAME_BODY, page, 200),
            ("PATCH", "/api/accounts/999999", RENAME_BODY, page, 404),
            ("PATCH", f"/api/accounts/{seeded['active']}", b'{"label": ""}', page, 422),
            ("POST", "/api/refresh", b"", FILE_PAGE_HEADERS, 503),
        ]

        for method, target, body, extra, expected_status in exchanges:
            challenge, nonce = issued_challenge(anonymous), fresh_nonce()
            if body:
                proof, content = sealed(anonymous, method, target, body, challenge=challenge, nonce=nonce)
            else:
                proof, content = signed(anonymous, method, target, challenge=challenge, nonce=nonce), None
            response = anonymous.request(method, target, content=content, headers={**extra, **proof})

            assert response.status_code == expected_status, (method, target)
            assert response.headers["x-usagedash-response"] == response_signature(
                ACCESS_TOKEN, challenge, nonce, response.status_code, response.content
            ), (method, target)
            assert isinstance(opened(response, challenge, nonce), dict), (method, target)

    def test_a_failure_the_route_did_not_expect_is_answered_signed_and_within_the_cross_origin_layer(
        self, anonymous: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Left to the outer layer, the 500 would carry neither the answer signature nor the allowed origin."""

        def fail(*_: object, **__: object) -> None:
            raise RuntimeError("synthetic-account-data-in-a-message")

        monkeypatch.setattr(api, "build_state", fail)
        challenge, nonce = issued_challenge(anonymous), fresh_nonce()
        headers = {**signed(anonymous, "GET", "/api/state", challenge=challenge, nonce=nonce), "Origin": "null"}

        with caplog.at_level(logging.ERROR, logger="usagedash.api"):
            response = anonymous.get("/api/state", headers=headers)

        assert response.status_code == 500
        assert opened(response, challenge, nonce) == {"detail": api.INTERNAL_ERROR_MESSAGE}
        assert response.headers["x-usagedash-response"] == response_signature(
            ACCESS_TOKEN, challenge, nonce, 500, response.content
        )
        assert response.headers["access-control-allow-origin"] == "null"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "Unexpected failure answering GET /api/state\nRuntimeError raised at:" in caplog.text
        assert "synthetic-account-data-in-a-message" not in caplog.text.replace(
            'raise RuntimeError("synthetic-account-data-in-a-message")', ""
        )

    def test_an_answer_to_a_signed_request_is_sealed_for_that_challenge_and_status(
        self, anonymous: TestClient, seeded: dict[str, int]
    ) -> None:
        challenge, nonce = issued_challenge(anonymous), fresh_nonce()
        proof = signed(anonymous, "GET", "/api/state", challenge=challenge, nonce=nonce)

        response = anonymous.get("/api/state", headers=proof)

        assert response.status_code == 200
        assert response.headers["content-type"] == SEALED_CONTENT_TYPE
        assert response.headers["content-length"] == str(len(response.content))
        assert b"providers" not in response.content
        assert PERSONA_EMAIL.encode() not in response.content
        state = opened(response, challenge, nonce)
        assert "providers" in state
        assert PERSONA_EMAIL in json.dumps(state)
        sealer = BodySealer(ACCESS_TOKEN)
        with pytest.raises(SealedBodyError):
            sealer.unseal(response.content, answer_seal_context(challenge, nonce, 201))
        with pytest.raises(SealedBodyError):
            sealer.unseal(response.content, answer_seal_context(issued_challenge(anonymous), nonce, 200))
        with pytest.raises(SealedBodyError):
            sealer.unseal(response.content, answer_seal_context(challenge, fresh_nonce(), 200))
        assert response.headers["x-usagedash-response"] != response_signature(
            ACCESS_TOKEN, challenge, fresh_nonce(), 200, response.content
        )
        with pytest.raises(SealedBodyError):
            BodySealer(OTHER_TOKEN).unseal(response.content, answer_seal_context(challenge, nonce, 200))

    def test_an_answer_to_a_bearer_client_is_plain(self, anonymous: TestClient, seeded: dict[str, int]) -> None:
        """A bearer client holds the token and checks nothing; the page never sends a bearer."""
        response = anonymous.get("/api/state", headers={"Authorization": f"Bearer {ACCESS_TOKEN}"})

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert "providers" in response.json()

    def test_a_signed_request_whose_body_is_not_sealed_is_refused_and_recorded(
        self, anonymous: TestClient, seeded: dict[str, int], database: Database, caplog: pytest.LogCaptureFixture
    ) -> None:
        target = f"/api/accounts/{seeded['active']}"
        challenge, nonce = issued_challenge(anonymous), fresh_nonce()
        proof = signed(anonymous, "PATCH", target, body=RENAME_BODY, challenge=challenge, nonce=nonce)

        with caplog.at_level(logging.WARNING, logger="usagedash.api"):
            response = anonymous.patch(
                target, content=RENAME_BODY, headers={**FILE_PAGE_HEADERS, **JSON_CONTENT, **proof}
            )

        assert response.status_code == 400
        assert opened(response, challenge, nonce) == {"detail": api.UNSEALED_BODY_MESSAGE}
        assert [record.getMessage() for record in caplog.records] == [
            "Refused PATCH /api/accounts/{account_id}: the body of a signed request is not sealed"
        ]
        with database.connect() as conn:
            assert db.account_label(conn, seeded["active"]) == "Personal"

    @pytest.mark.parametrize(
        "sealed_for",
        ["another challenge", "another nonce", "another target", "another token", "nothing: bytes that are no seal"],
    )
    def test_a_sealed_body_that_does_not_open_for_this_request_is_refused_and_recorded(
        self,
        anonymous: TestClient,
        seeded: dict[str, int],
        database: Database,
        caplog: pytest.LogCaptureFixture,
        sealed_for: str,
    ) -> None:
        target = f"/api/accounts/{seeded['active']}"
        challenge, nonce = issued_challenge(anonymous), fresh_nonce()
        sealer = BodySealer(ACCESS_TOKEN)
        if sealed_for == "another challenge":
            context = request_seal_context("PATCH", target, issued_challenge(anonymous), nonce)
            content = sealer.seal(RENAME_BODY, context)
        elif sealed_for == "another nonce":
            content = sealer.seal(RENAME_BODY, request_seal_context("PATCH", target, challenge, fresh_nonce()))
        elif sealed_for == "another target":
            content = sealer.seal(RENAME_BODY, request_seal_context("PATCH", "/api/accounts/999999", challenge, nonce))
        elif sealed_for == "another token":
            content = BodySealer(OTHER_TOKEN).seal(RENAME_BODY, request_seal_context("PATCH", target, challenge, nonce))
        else:
            content = b"short"
        proof = signed(anonymous, "PATCH", target, body=content, challenge=challenge, nonce=nonce)
        headers = {**FILE_PAGE_HEADERS, "Content-Type": SEALED_CONTENT_TYPE, **proof}

        with caplog.at_level(logging.WARNING, logger="usagedash.api"):
            response = anonymous.patch(target, content=content, headers=headers)

        assert response.status_code == 400
        assert opened(response, challenge, nonce) == {"detail": api.SEALED_BODY_NOT_OPENED_MESSAGE}
        assert [record.getMessage() for record in caplog.records] == [
            "Refused PATCH /api/accounts/{account_id}: the sealed body could not be opened"
        ]
        with database.connect() as conn:
            assert db.account_label(conn, seeded["active"]) == "Personal"

    def test_a_refusal_of_the_request_itself_and_an_answer_to_a_bearer_carry_no_answer_signature(
        self, anonymous: TestClient, seeded: dict[str, int]
    ) -> None:
        """There is no accepted exchange to bind a refusal to, and a bearer client checks nothing."""
        target = f"/api/accounts/{seeded['active']}"
        answers = [
            anonymous.get("/api/state"),
            anonymous.get("/api/state", headers=signed(anonymous, "GET", "/api/tokens")),
            anonymous.patch(
                target,
                content=RENAME_BODY,
                headers={
                    **JSON_CONTENT,
                    "X-Requested-With": "usagedash",
                    **signed(anonymous, "PATCH", target, body=RENAME_BODY),
                },
            ),
            anonymous.get("/api/state", headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}),
        ]

        assert [answer.status_code for answer in answers] == [401, 401, 403, 200]
        assert all("x-usagedash-response" not in answer.headers for answer in answers)

    def test_the_answer_signature_is_exposed_to_the_file_page(
        self, anonymous: TestClient, seeded: dict[str, int]
    ) -> None:
        served = anonymous.get("/api/state", headers={**signed(anonymous, "GET", "/api/state"), "Origin": "null"})

        assert served.status_code == 200
        exposed = {name.strip().lower() for name in served.headers["access-control-expose-headers"].split(",")}
        assert "x-usagedash-response" in exposed

    def test_a_signature_is_accepted_once(self, anonymous: TestClient, seeded: dict[str, int]) -> None:
        headers = signed(anonymous, "GET", "/api/state")

        assert anonymous.get("/api/state", headers=headers).status_code == 200
        assert anonymous.get("/api/state", headers=headers).status_code == 401

    def test_a_challenge_spent_on_one_route_is_spent_for_all(
        self, anonymous: TestClient, seeded: dict[str, int]
    ) -> None:
        challenge = issued_challenge(anonymous)

        assert (
            anonymous.get("/api/state", headers=signed(anonymous, "GET", "/api/state", challenge=challenge)).status_code
            == 200
        )
        assert (
            anonymous.get(
                "/api/tokens", headers=signed(anonymous, "GET", "/api/tokens", challenge=challenge)
            ).status_code
            == 401
        )
        assert anonymous.get("/api/tokens", headers=signed(anonymous, "GET", "/api/tokens")).status_code == 200

    def test_a_signature_covers_the_body_so_an_altered_or_added_body_is_refused(
        self, anonymous: TestClient, seeded: dict[str, int], database: Database
    ) -> None:
        target = f"/api/accounts/{seeded['active']}"
        page = {**FILE_PAGE_HEADERS, **JSON_CONTENT}

        altered = anonymous.patch(
            target,
            content=b'{"label": "Hijacked"}',
            headers={**page, **signed(anonymous, "PATCH", target, body=RENAME_BODY)},
        )
        unsigned_body = anonymous.patch(
            target, content=RENAME_BODY, headers={**page, **signed(anonymous, "PATCH", target)}
        )

        assert altered.status_code == 401
        assert unsigned_body.status_code == 401
        with database.connect() as conn:
            assert db.account_label(conn, seeded["active"]) == "Personal"

    @pytest.mark.parametrize("authorization", [None, "bearer", "signature"])
    def test_a_body_over_the_limit_is_refused_before_anything_else_is_checked(
        self, anonymous: TestClient, seeded: dict[str, int], caplog: pytest.LogCaptureFixture, authorization: str | None
    ) -> None:
        target = f"/api/accounts/{seeded['active']}"
        body = b'{"label": "' + b"x" * api.MAX_SIGNED_BODY_BYTES + b'"}'
        headers = {**FILE_PAGE_HEADERS, **JSON_CONTENT}
        if authorization == "bearer":
            headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"
        elif authorization == "signature":
            headers.update(signed(anonymous, "PATCH", target, body=body))

        with caplog.at_level(logging.WARNING, logger="usagedash.api"):
            response = anonymous.patch(target, content=body, headers=headers)

        assert response.status_code == 413
        assert response.json() == {"detail": "The request body is larger than this API accepts."}
        assert [record.getMessage() for record in caplog.records] == [
            "Refused PATCH /api/accounts/{account_id}: the body is larger than the API accepts"
        ]
        assert "xxxx" not in caplog.text

    def test_a_body_at_the_limit_is_read_and_then_judged_like_any_other(
        self, anonymous: TestClient, seeded: dict[str, int]
    ) -> None:
        target = f"/api/accounts/{seeded['active']}"
        overhead = access.SEAL_NONCE_BYTES + access.SEAL_TAG_BYTES
        body = b'{"label": "' + b"x" * (api.MAX_SIGNED_BODY_BYTES - overhead - 13) + b'"}'
        headers, content = sealed(anonymous, "PATCH", target, body)
        assert len(content) == api.MAX_SIGNED_BODY_BYTES

        response = anonymous.patch(target, content=content, headers={**FILE_PAGE_HEADERS, **headers})

        assert response.status_code == 422

    def test_the_body_is_read_from_the_connection_bounded_and_handed_on_exactly(self) -> None:
        """The headers do not matter: the bytes that arrive are what is hashed and what the route gets."""

        def scope() -> dict[str, Any]:
            return {
                "type": "http",
                "method": "PATCH",
                "scheme": "http",
                "server": ("127.0.0.1", 8787),
                "path": "/api/accounts/1",
                "query_string": b"",
                "headers": [(b"content-length", b"5")],
            }

        def receiving(*frames: bytes) -> tuple[Callable[[], Any], list[str]]:
            pending = list(frames)
            delivered: list[str] = []

            async def receive() -> dict[str, Any]:
                frame = pending.pop(0)
                delivered.append("frame")
                return {"type": "http.request", "body": frame, "more_body": bool(pending)}

            return receive, delivered

        async def read_then_hand_on(*frames: bytes) -> tuple[bytes, bytes, list[str]]:
            receive, delivered = receiving(*frames)
            request = api.Request(scope(), receive)
            body = await api.read_signed_body(request)
            handed_on = await api.with_body(request, body).body()
            return body, handed_on, delivered

        body, handed_on, delivered = asyncio.run(read_then_hand_on(RENAME_BODY[:5], RENAME_BODY[5:]))
        assert body == RENAME_BODY
        assert handed_on == RENAME_BODY
        assert delivered == ["frame", "frame"]

        body, handed_on, _ = asyncio.run(read_then_hand_on(b""))
        assert (body, handed_on) == (b"", b"")

        async def read_too_much() -> list[str]:
            half = b"x" * (api.MAX_SIGNED_BODY_BYTES // 2 + 1)
            receive, delivered = receiving(half, half, b"never read")
            with pytest.raises(api.HTTPException) as refused:
                await api.read_signed_body(api.Request(scope(), receive))
            assert refused.value.status_code == 413
            return delivered

        assert asyncio.run(read_too_much()) == ["frame", "frame"]

        async def read_one_oversized_frame() -> list[str]:
            receive, delivered = receiving(b"x" * (api.MAX_SIGNED_BODY_BYTES + 1), b"never read")
            with pytest.raises(api.HTTPException) as refused:
                await api.read_signed_body(api.Request(scope(), receive))
            assert refused.value.status_code == 413
            return delivered

        assert asyncio.run(read_one_oversized_frame()) == ["frame"]

        async def read_exactly_the_limit() -> int:
            receive, _delivered = receiving(b"x" * api.MAX_SIGNED_BODY_BYTES)
            return len(await api.read_signed_body(api.Request(scope(), receive)))

        assert asyncio.run(read_exactly_the_limit()) == api.MAX_SIGNED_BODY_BYTES

    def test_what_the_connection_sends_after_the_body_still_reaches_the_route(self) -> None:
        async def disconnect_after_body() -> list[str]:
            messages = [
                {"type": "http.request", "body": RENAME_BODY, "more_body": False},
                {"type": "http.disconnect"},
            ]

            async def receive() -> dict[str, Any]:
                return messages.pop(0)

            request = api.Request(
                {
                    "type": "http",
                    "method": "PATCH",
                    "scheme": "http",
                    "server": ("127.0.0.1", 8787),
                    "path": "/",
                    "query_string": b"",
                    "headers": [],
                },
                receive,
            )
            handed = api.with_body(request, await api.read_signed_body(request))
            return [(await handed.receive())["type"], (await handed.receive())["type"]]

        assert asyncio.run(disconnect_after_body()) == ["http.request", "http.disconnect"]

    def test_a_refused_signature_is_logged_by_its_reason_and_by_nothing_it_sent(
        self, anonymous: TestClient, seeded: dict[str, int], caplog: pytest.LogCaptureFixture
    ) -> None:
        spent = signed(anonymous, "GET", "/api/state")
        wrong = signed(anonymous, "GET", "/api/tokens")
        with caplog.at_level(logging.WARNING, logger="usagedash.api"):
            assert anonymous.get("/api/state", headers=spent).status_code == 200
            anonymous.get("/api/state", headers=spent)
            anonymous.get("/api/state", headers=wrong)
            anonymous.get("/api/state", headers={"Authorization": "Usagedash not.a.signature"})
            anonymous.get("/api/state", headers={"Authorization": f"Bearer {OTHER_TOKEN}"})

        assert [record.getMessage() for record in caplog.records] == [
            "Refused GET /api/state: the challenge was not issued by this process, or has been used",
            "Refused GET /api/state: the signature does not match",
            "Refused GET /api/state: the signature is malformed",
            "Refused GET /api/state: no valid access token",
        ]
        for header in (spent, wrong):
            assert header["Authorization"].split(" ", 1)[1] not in caplog.text
        assert OTHER_TOKEN not in caplog.text

    def test_a_signed_request_is_also_refused_for_a_bad_host(
        self, settings: Settings, database: Database, now: datetime
    ) -> None:
        app = create_app(settings, database, None, lambda: now, access_token=ACCESS_TOKEN, run_collector=False)

        with TestClient(app, base_url=BASE_URL) as rebinding_client:
            header = signed(rebinding_client, "GET", "/api/state")
            response = rebinding_client.get("/api/state", headers={**header, "Host": "attacker.example"})

        assert response.status_code == 400

    def test_the_parts_of_a_header_are_read_only_from_a_well_formed_usagedash_header(self) -> None:
        challenge = "synthetic-challenge-" + "c" * 23
        nonce = "synthetic-nonce-" + "n" * 27
        header = sign_request(ACCESS_TOKEN, "GET", "/api/state", challenge, nonce=nonce)

        parsed = api.presented_signature(header)
        assert parsed is not None
        assert (parsed.challenge, parsed.nonce) == (challenge, nonce)
        assert api.presented_signature(header.replace("Usagedash", "usagedash", 1)) == parsed
        assert api.presented_signature(None) is None
        assert api.presented_signature(f"Bearer {ACCESS_TOKEN}") is None
        assert api.presented_signature("Usagedash not.a.signature") is None
        assert api.presented_signature(f"Usagedash {challenge}.{'0' * 64}") is None

    def test_the_signed_target_is_the_raw_path_and_query_when_the_server_gives_one(self) -> None:
        scope: dict[str, Any] = {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "server": ("127.0.0.1", 8787),
            "headers": [],
            "path": "/api/accounts/12",
            "raw_path": b"/api/accounts/1%32",
            "query_string": b"range=24h&tz=America%2FNew_York",
        }

        assert signed_target(api.Request(scope)) == "/api/accounts/1%32?range=24h&tz=America%2FNew_York"
        del scope["raw_path"]
        assert signed_target(api.Request(scope)) == "/api/accounts/12?range=24h&tz=America%2FNew_York"
        scope["query_string"] = b""
        assert signed_target(api.Request(scope)) == "/api/accounts/12"


class TestTheFilePageOrigin:
    """The dashboard page is a local file, so a browser sends ``Origin: null`` with its requests."""

    @pytest.mark.parametrize(
        "headers",
        [
            pytest.param(FILE_PAGE_HEADERS, id="a bearer token from the null origin"),
            pytest.param({"Origin": "null"}, id="the null origin without the page header"),
        ],
    )
    def test_a_change_from_the_null_origin_needs_a_signature_and_the_page_header(
        self, client: TestClient, seeded: dict[str, int], database: Database, headers: dict[str, str]
    ) -> None:
        response = client.patch(f"/api/accounts/{seeded['active']}", json={"label": "Hijacked"}, headers=headers)

        assert response.status_code == 403
        with database.connect() as conn:
            assert db.account_label(conn, seeded["active"]) == "Personal"

    @pytest.mark.parametrize(
        "page_headers",
        [
            pytest.param({"X-Requested-With": "usagedash"}, id="no origin"),
            pytest.param({"X-Requested-With": "usagedash", "Origin": "http://attacker.example"}, id="another origin"),
            pytest.param({"X-Requested-With": "usagedash", "Origin": "NULL"}, id="not the lower-case null"),
            pytest.param({"Origin": "null"}, id="null without the page header"),
        ],
    )
    def test_a_signed_change_still_needs_the_page_header_and_a_page_origin(
        self, anonymous: TestClient, seeded: dict[str, int], database: Database, page_headers: dict[str, str]
    ) -> None:
        target = f"/api/accounts/{seeded['active']}"

        headers, content = sealed(anonymous, "PATCH", target, RENAME_BODY)

        response = anonymous.patch(target, content=content, headers={**page_headers, **headers})

        assert response.status_code == 403
        with database.connect() as conn:
            assert db.account_label(conn, seeded["active"]) == "Personal"

    def test_a_signed_change_from_the_dashboard_s_own_origin_is_accepted_too(
        self, anonymous: TestClient, seeded: dict[str, int]
    ) -> None:
        target = f"/api/accounts/{seeded['active']}"

        headers, content = sealed(anonymous, "PATCH", target, RENAME_BODY)

        response = anonymous.patch(target, content=content, headers={**PAGE_HEADERS, **headers})

        assert response.status_code == 200

    def test_a_preflight_from_the_file_page_is_answered_for_that_origin(self, anonymous: TestClient) -> None:
        response = anonymous.options(
            "/api/accounts/1",
            headers={
                "Origin": "null",
                "Access-Control-Request-Method": "PATCH",
                "Access-Control-Request-Headers": "authorization,content-type,x-requested-with",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "null"
        allowed_methods = {method.strip() for method in response.headers["access-control-allow-methods"].split(",")}
        assert {"GET", "POST", "PATCH"} <= allowed_methods
        allowed_headers = {header.strip() for header in response.headers["access-control-allow-headers"].split(",")}
        assert {"Authorization", "Content-Type", "X-Requested-With"} <= allowed_headers
        assert "access-control-allow-credentials" not in response.headers
        assert "set-cookie" not in response.headers
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-store"

    @pytest.mark.parametrize("origin", ["http://attacker.example", "http://127.0.0.1:9999", "NULL", BASE_URL])
    def test_a_preflight_from_another_origin_is_refused_without_an_allowance(
        self, anonymous: TestClient, origin: str
    ) -> None:
        response = anonymous.options("/api/state", headers={"Origin": origin, "Access-Control-Request-Method": "GET"})

        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers

    def test_a_refused_preflight_is_recorded_without_the_origin_or_the_path(
        self, anonymous: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The cross-origin layer answers a preflight itself, so its refusals are recorded by the layer around it."""
        with caplog.at_level(logging.WARNING, logger="usagedash.api"):
            foreign = anonymous.options(
                "/api/accounts/1",
                headers={"Origin": "http://attacker.example", "Access-Control-Request-Method": "PATCH"},
            )
            method = anonymous.options(
                "/api/state", headers={"Origin": "null", "Access-Control-Request-Method": "DELETE"}
            )
            allowed = anonymous.options(
                "/api/state", headers={"Origin": "null", "Access-Control-Request-Method": "GET"}
            )

        assert (foreign.status_code, method.status_code, allowed.status_code) == (400, 400, 200)
        assert [record.getMessage() for record in caplog.records] == [
            "Refused OPTIONS (no route): the cross-origin preflight is not allowed",
            "Refused OPTIONS (no route): the cross-origin preflight is not allowed",
        ]
        assert "attacker" not in caplog.text
        assert "/api/accounts/1" not in caplog.text
        assert "DELETE" not in caplog.text

    def test_a_preflight_for_a_method_the_api_does_not_use_is_refused(self, anonymous: TestClient) -> None:
        response = anonymous.options(
            "/api/state", headers={"Origin": "null", "Access-Control-Request-Method": "DELETE"}
        )

        assert response.status_code == 400
        assert "Disallowed CORS method" in response.text

    def test_a_preflight_with_a_foreign_host_is_refused_by_the_host_guard_first(
        self, settings: Settings, database: Database, now: datetime, caplog: pytest.LogCaptureFixture
    ) -> None:
        app = create_app(settings, database, None, lambda: now, access_token=ACCESS_TOKEN, run_collector=False)
        with (
            TestClient(app, base_url=BASE_URL) as rebinding_client,
            caplog.at_level(logging.WARNING, logger="usagedash.api"),
        ):
            response = rebinding_client.options(
                "/api/state",
                headers={"Host": "attacker.example", "Origin": "null", "Access-Control-Request-Method": "GET"},
            )

        assert response.status_code == 400
        assert response.text == "Invalid host header"
        assert "access-control-allow-origin" not in response.headers
        assert [record.getMessage() for record in caplog.records] == [
            "Refused OPTIONS (no route): the host is not allowed"
        ]

    def test_an_answer_to_the_file_page_is_marked_for_its_origin_and_keeps_the_security_headers(
        self, anonymous: TestClient, seeded: dict[str, int]
    ) -> None:
        challenge = anonymous.get("/challenge", headers={"Origin": "null"})
        served = anonymous.get("/api/state", headers={**signed(anonymous, "GET", "/api/state"), "Origin": "null"})
        refused = anonymous.get("/api/state", headers={"Origin": "null"})

        assert challenge.status_code == 200
        assert served.status_code == 200
        assert refused.status_code == 401
        for response in (challenge, served, refused):
            assert response.headers["access-control-allow-origin"] == "null"
            assert "origin" in response.headers["vary"].lower()
            assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
            assert response.headers["cache-control"] == "no-store"

    def test_an_answer_to_another_origin_carries_no_allowance(self, client: TestClient, seeded: dict[str, int]) -> None:
        for origin in ("http://attacker.example", "http://127.0.0.1:9999", BASE_URL):
            response = client.get("/api/state", headers={"Origin": origin})
            assert response.status_code == 200
            assert "access-control-allow-origin" not in response.headers


class TestNothingIsRecordedForTelemetry:
    """FastAPI records requests through OpenTelemetry and adds exporters from the environment unless told not to."""

    @pytest.fixture
    def provider_lookups(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Record every time anything asks OpenTelemetry for the provider it would record through."""
        lookups: list[str] = []

        def recording(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
            def lookup(*arguments: Any, **options: Any) -> Any:
                lookups.append(name)
                return original(*arguments, **options)

            return lookup

        for module, name in (
            (otel_trace, "get_tracer_provider"),
            (otel_metrics, "get_meter_provider"),
            (otel_logs, "get_logger_provider"),
        ):
            monkeypatch.setattr(module, name, recording(name, getattr(module, name)))
        return lookups

    def stock_app(self) -> FastAPI:
        """An application as FastAPI makes one when nothing is said about telemetry."""
        app = FastAPI()

        @app.get("/")
        def index() -> dict[str, bool]:
            return {"ok": True}

        return app

    def test_fastapi_looks_up_a_provider_for_a_request_unless_it_is_told_not_to(
        self, provider_lookups: list[str]
    ) -> None:
        """Shows the recording above can see what it is there to see."""
        with TestClient(self.stock_app()) as stock:
            assert stock.get("/").status_code == 200

        assert "get_tracer_provider" in provider_lookups

    def test_no_request_to_the_dashboard_reaches_a_telemetry_provider(
        self,
        settings: Settings,
        database: Database,
        now: datetime,
        seeded: dict[str, int],
        provider_lookups: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A successful read, a refused one, a rename, a validation failure, and an unexpected failure."""
        app = create_app(settings, database, None, lambda: now, access_token=ACCESS_TOKEN, run_collector=False)
        with TestClient(app, base_url=BASE_URL, follow_redirects=False, raise_server_exceptions=False) as dashboard:
            assert dashboard.get("/api/state").status_code == 401
            signed_in(dashboard)
            assert dashboard.get("/api/state").status_code == 200
            renamed = dashboard.patch(
                f"/api/accounts/{seeded['active']}", json={"label": "Renamed"}, headers=PAGE_HEADERS
            )
            assert renamed.status_code == 200
            assert dashboard.get("/api/tokens", params={"range": "90d"}).status_code == 422
            monkeypatch.setattr(api, "build_state", self.failing)
            assert dashboard.get("/api/state").status_code == 500

        assert provider_lookups == []

    def failing(self, *arguments: object, **options: object) -> None:
        raise RuntimeError("the message of an unexpected failure")

    def test_fastapi_reads_exporters_from_the_environment_at_startup_unless_it_is_told_not_to(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Shows the next test's environment is one FastAPI acts on. An exporter it does not support is refused."""
        self.name_an_exporter(monkeypatch)

        with caplog.at_level(logging.WARNING), TestClient(self.stock_app()):
            pass

        assert "FastAPI automatic telemetry configuration failed" in caplog.text

    def test_the_dashboard_starts_without_reading_exporters_from_the_environment(
        self,
        settings: Settings,
        database: Database,
        now: datetime,
        provider_lookups: list[str],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self.name_an_exporter(monkeypatch)
        app = create_app(settings, database, None, lambda: now, access_token=ACCESS_TOKEN, run_collector=False)

        with caplog.at_level(logging.WARNING), TestClient(app, base_url=BASE_URL) as dashboard:
            assert dashboard.get("/healthz").status_code == 200

        assert "telemetry" not in caplog.text
        assert provider_lookups == []

    def name_an_exporter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set what a shell with telemetry configured would hand down, naming an exporter FastAPI refuses."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
        monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console")

    def test_every_switch_fastapi_has_is_off(self) -> None:
        """A switch a later FastAPI adds would be on by default, and this is what says so."""
        assert api.TELEMETRY_OFF == {
            "tracing": False,
            "metrics": False,
            "logs": False,
            "operation_spans": False,
            "auto_configure": False,
        }
        switches = {name for name, kind in get_type_hints(TelemetryConfig).items() if kind is bool}
        assert switches == set(api.TELEMETRY_OFF)


class TestAnUnexpectedFailureWhileAnswering:
    """The server would log an unhandled exception with its message, which can quote account data."""

    @pytest.fixture
    def failing(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
        def fail(*_: object, **__: object) -> None:
            raise RuntimeError("synthetic-account-data-in-a-message")

        monkeypatch.setattr(api, "build_state", fail)
        return client

    def test_the_request_is_answered_with_a_plain_500_that_carries_the_security_headers(
        self, failing: TestClient
    ) -> None:
        response = failing.get("/api/state")

        assert response.status_code == 500
        assert response.json() == {"detail": "The dashboard failed to answer this request. Its log says where."}
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-store"

    def test_the_log_says_where_it_was_raised_and_never_what_the_message_was(
        self, failing: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR, logger="usagedash.api"):
            failing.get("/api/state")

        assert "Unexpected failure answering GET /api/state\nRuntimeError raised at:" in caplog.text
        assert "synthetic-account-data-in-a-message" not in caplog.text.replace(
            'raise RuntimeError("synthetic-account-data-in-a-message")', ""
        )
        assert all(record.exc_info is None for record in caplog.records)

    def test_the_next_request_is_still_served(self, failing: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        assert failing.get("/api/state").status_code == 500
        monkeypatch.undo()

        assert failing.get("/healthz").status_code == 200


class TestAFailureWhileSendingThePlain500:
    """Raised while the first failure was being handled, a second one carried it to the server, message and all."""

    SECRET = "synthetic-account-data-in-a-message"
    SCOPE: ClassVar[dict[str, Any]] = {"type": "http", "method": "GET", "path": "/api/state", "headers": []}

    def answer(self, send_fails_with: BaseException | None) -> list[dict[str, Any]]:
        """Run a request whose endpoint fails, with a send that fails as told; return what reached the server."""
        sent: list[dict[str, Any]] = []
        secret = self.SECRET

        async def endpoint(scope: Any, receive: Any, send: Any) -> None:
            raise RuntimeError(secret)

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def server_send(message: dict[str, Any]) -> None:
            sent.append(message)
            if send_fails_with is not None:
                raise send_fails_with

        asyncio.run(api.SealedResponses(endpoint)(dict(self.SCOPE), receive, server_send))
        return sent

    @pytest.mark.parametrize(
        "failure",
        [
            pytest.param(OSError(32, "Broken pipe"), id="the connection is gone"),
            pytest.param(RuntimeError("a defect while sending"), id="a defect"),
        ],
    )
    def test_a_failure_to_send_it_goes_no_further_and_is_logged_without_either_message(
        self, caplog: pytest.LogCaptureFixture, failure: Exception
    ) -> None:
        with caplog.at_level(logging.ERROR, logger="usagedash.api"):
            sent = self.answer(failure)

        assert [message["type"] for message in sent] == ["http.response.start"]
        assert sent[0]["status"] == 500
        assert "Unexpected failure answering GET (no route)\nRuntimeError raised at:" in caplog.text
        assert f"Unexpected failure sending the plain 500 for GET (no route)\n{type(failure).__name__}" in caplog.text
        assert self.SECRET not in caplog.text
        assert str(failure) not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)

    def test_a_failure_from_outside_a_route_is_still_answered_by_this_layer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A route answers its own failure; one raised before any route is reached is answered here."""
        with caplog.at_level(logging.ERROR, logger="usagedash.api"):
            sent = self.answer(None)

        assert [message["type"] for message in sent] == ["http.response.start", "http.response.body"]
        assert sent[0]["status"] == 500
        assert dict(sent[0]["headers"])[b"x-content-type-options"] == b"nosniff"
        assert sent[1]["body"] == api.plain_internal_error().body
        assert "Unexpected failure answering GET (no route)\nRuntimeError raised at:" in caplog.text
        assert self.SECRET not in caplog.text

    def test_a_cancellation_while_sending_it_goes_on_and_carries_nothing_of_the_first_failure(self) -> None:
        """Swallowed, a cancellation would keep the server from stopping the request."""
        with pytest.raises(asyncio.CancelledError) as raised:
            self.answer(asyncio.CancelledError())

        assert (raised.value.__context__, raised.value.__cause__) == (None, None)
        assert self.SECRET not in "".join(traceback.format_exception(raised.value))


class TestNothingLeavesTheApplicationCarryingAnotherException:
    """Starlette sends the answer to a request that failed validation while it handles the validation error."""

    REJECTED = "Synthetic/" + "Z" * 60

    @pytest.mark.parametrize("cancelled_at", ["http.response.start", "http.response.body"])
    def test_a_cancellation_while_a_validation_answer_is_sent_leaves_without_the_rejected_request(
        self, settings: Settings, database: Database, now: datetime, cancelled_at: str
    ) -> None:
        """The validation error quotes what the request sent, and the server logs the chain an exception carries."""
        app = create_app(settings, database, None, lambda: now, access_token=ACCESS_TOKEN, run_collector=False)
        query = urlencode({"tz": self.REJECTED}).encode()
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/tokens",
            "raw_path": b"/api/tokens",
            "query_string": query,
            "root_path": "",
            "headers": [(b"host", b"127.0.0.1:8787"), (b"authorization", f"Bearer {ACCESS_TOKEN}".encode())],
            "server": ("127.0.0.1", 8787),
            "client": ("127.0.0.1", 50000),
        }
        sent: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def server_send(message: dict[str, Any]) -> None:
            sent.append(message)
            if message["type"] == cancelled_at:
                raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError) as raised:
            asyncio.run(app(scope, receive, server_send))

        assert sent[0]["status"] == 422
        assert (raised.value.__context__, raised.value.__cause__) == (None, None)
        assert "Synthetic" not in "".join(traceback.format_exception(raised.value))


class TestAFailureAfterTheResponseHasBegun:
    """A file response opens its file after its status and headers have gone, past any check of a finished response."""

    SECRET_NAME = "synthetic-directory-name"

    @pytest.fixture
    def browser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: Settings, database: Database, now: datetime
    ) -> Iterator[TestClient]:
        """A dashboard whose page and script exist and cannot be opened."""
        static = tmp_path / self.SECRET_NAME
        static.mkdir()
        for name in ("index.html", "app.js"):
            (static / name).write_text("unreadable")
            (static / name).chmod(0)
        monkeypatch.setattr(api, "STATIC_DIR", static)
        app = create_app(settings, database, None, lambda: now, access_token=ACCESS_TOKEN, run_collector=False)
        with TestClient(app, base_url=BASE_URL) as test_client:
            yield test_client

    @pytest.mark.parametrize(
        ("path", "logged_as"),
        [
            pytest.param("/", "GET /\n", id="the page, by its route"),
            pytest.param("/static/app.js", "GET (no route)\n", id="a static file, which has no route template"),
        ],
    )
    def test_the_failure_does_not_leave_the_application_and_is_logged_without_its_message(
        self, browser: TestClient, caplog: pytest.LogCaptureFixture, path: str, logged_as: str
    ) -> None:
        """The test client raises whatever the application lets out, as the server would log it."""
        with caplog.at_level(logging.ERROR):
            response = browser.get(path)

        assert (response.status_code, response.content) == (200, b"")
        assert response.headers["x-content-type-options"] == "nosniff"
        assert f"Unexpected failure answering {logged_as}PermissionError raised at:" in caplog.text
        assert self.SECRET_NAME not in caplog.text
        assert "app.js" not in caplog.text
        assert "Permission denied" not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)

    def test_the_next_request_is_still_served(self, browser: TestClient) -> None:
        browser.get("/")

        assert browser.get("/healthz").json() == {"status": "ok"}


class TestWhatALogLineSaysAboutARequest:
    """A path holds what the client put in it. The log names the route by its template."""

    def test_a_failure_while_renaming_is_logged_without_the_account_s_id(
        self,
        client: TestClient,
        seeded: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def fail(*_: object, **__: object) -> bool:
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr(db, "rename_account", fail)

        with caplog.at_level(logging.ERROR, logger="usagedash.api"):
            response = client.patch(f"/api/accounts/{seeded['active']}", json={"label": "Work"}, headers=PAGE_HEADERS)

        assert response.status_code == 500
        assert "Unexpected failure answering PATCH /api/accounts/{account_id}\nRuntimeError raised at:" in caplog.text
        assert f"/api/accounts/{seeded['active']}" not in caplog.text

    def test_a_route_template_is_what_names_a_matched_route_and_a_fixed_text_names_none(self) -> None:
        class Route:
            path = "/api/accounts/{account_id}"

        assert api.route_template({"route": Route()}) == "/api/accounts/{account_id}"
        assert api.route_template({}) == "(no route)"
        assert api.route_template({"route": object()}) == "(no route)"


class TestAnAcceptedRequestIsRecorded:
    def test_an_accepted_signature_and_an_accepted_bearer_are_logged_by_kind_and_by_nothing_they_sent(
        self, anonymous: TestClient, seeded: dict[str, int], caplog: pytest.LogCaptureFixture
    ) -> None:
        challenge, nonce = issued_challenge(anonymous), fresh_nonce()
        proof = signed(anonymous, "GET", "/api/state", challenge=challenge, nonce=nonce)

        with caplog.at_level(logging.INFO, logger="usagedash.api"):
            anonymous.get("/api/state", headers=proof)
            anonymous.get(
                "/api/tokens?range=24h&tz=America%2FNew_York", headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}
            )

        assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
            (logging.INFO, "Accepted GET /api/state: signature"),
            (logging.INFO, "Accepted GET /api/tokens: bearer"),
        ]
        assert all(secret not in caplog.text for secret in (ACCESS_TOKEN, challenge, nonce, "America"))


class TestARefusedRequestIsRecorded:
    """A request for account data that is turned away leaves a record, and the record holds nothing it sent."""

    def test_a_request_without_a_valid_token_is_logged_by_method_and_route(
        self, anonymous: TestClient, seeded: dict[str, int], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="usagedash.api"):
            anonymous.get("/api/state")
            anonymous.patch(
                f"/api/accounts/{seeded['active']}",
                json={"label": "synthetic-label"},
                headers={**PAGE_HEADERS, "Authorization": "Bearer synthetic-wrong-token"},
            )

        assert [record.getMessage() for record in caplog.records] == [
            "Refused GET /api/state: no valid access token",
            "Refused PATCH /api/accounts/{account_id}: no valid access token",
        ]
        assert all(record.levelno == logging.WARNING for record in caplog.records)

    @pytest.mark.parametrize(
        ("headers", "reason"),
        [
            pytest.param({"Origin": BASE_URL}, "the header the dashboard page sends is missing", id="no page header"),
            pytest.param(
                {"X-Requested-With": "usagedash", "Origin": "http://attacker.example"},
                "it did not come from the dashboard page",
                id="another origin",
            ),
        ],
    )
    def test_a_change_that_did_not_come_from_the_page_is_logged_without_what_it_sent(
        self,
        client: TestClient,
        seeded: dict[str, int],
        caplog: pytest.LogCaptureFixture,
        headers: dict[str, str],
        reason: str,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="usagedash.api"):
            response = client.patch(
                f"/api/accounts/{seeded['active']}", json={"label": "synthetic-label"}, headers=headers
            )

        assert response.status_code == 403
        assert [record.getMessage() for record in caplog.records] == [
            f"Refused PATCH /api/accounts/{{account_id}}: {reason}"
        ]
        assert "attacker" not in caplog.text
        assert "synthetic-label" not in caplog.text

    def test_a_request_that_is_served_logs_no_refusal(
        self, client: TestClient, seeded: dict[str, int], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="usagedash.api"):
            assert client.get("/api/state").status_code == 200

        assert caplog.records == []


class TestWhoIsAskingIsCheckedBeforeAnythingElseOfTheRequest:
    """FastAPI decodes a JSON body before it runs a dependency, so the checks are not dependencies."""

    NOT_JSON = b"{synthetic-not-json"

    def rename(self, browser: TestClient, account_id: object, headers: dict[str, str]) -> Any:
        return browser.patch(
            f"/api/accounts/{account_id}",
            content=self.NOT_JSON,
            headers={"Content-Type": "application/json", **headers},
        )

    def test_a_malformed_body_without_a_token_is_refused_for_the_token_and_the_refusal_recorded(
        self, anonymous: TestClient, seeded: dict[str, int], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="usagedash.api"):
            response = self.rename(anonymous, seeded["active"], PAGE_HEADERS)

        assert response.status_code == 401
        assert [record.getMessage() for record in caplog.records] == [
            "Refused PATCH /api/accounts/{account_id}: no valid access token"
        ]

    @pytest.mark.parametrize(
        ("headers", "reason"),
        [
            pytest.param({"Origin": BASE_URL}, "the header the dashboard page sends is missing", id="no page header"),
            pytest.param(
                {"X-Requested-With": "usagedash", "Origin": "http://attacker.example"},
                "it did not come from the dashboard page",
                id="another origin",
            ),
            pytest.param({"X-Requested-With": "usagedash"}, "it did not come from the dashboard page", id="no origin"),
        ],
    )
    def test_a_malformed_body_that_did_not_come_from_the_page_is_refused_for_that_and_the_refusal_recorded(
        self,
        client: TestClient,
        seeded: dict[str, int],
        caplog: pytest.LogCaptureFixture,
        headers: dict[str, str],
        reason: str,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="usagedash.api"):
            response = self.rename(client, seeded["active"], headers)

        assert response.status_code == 403
        assert [record.getMessage() for record in caplog.records] == [
            f"Refused PATCH /api/accounts/{{account_id}}: {reason}"
        ]

    @pytest.mark.parametrize(
        ("method", "path", "options"),
        [
            pytest.param(
                "PATCH", "/api/accounts/synthetic-id", {"json": {"label": "Work"}}, id="an id that is not one"
            ),
            pytest.param("PATCH", "/api/accounts/1", {"json": {"label": ""}}, id="a label that is not one"),
            pytest.param("GET", "/api/tokens", {"params": {"range": "never"}}, id="a range that is not one"),
            pytest.param("GET", "/api/tokens", {"params": {"tz": "Z" * 65}}, id="an overlong time zone"),
        ],
    )
    def test_no_validation_answer_is_given_to_a_request_without_a_token(
        self, anonymous: TestClient, method: str, path: str, options: dict[str, Any]
    ) -> None:
        response = anonymous.request(method, path, headers=PAGE_HEADERS, **options)

        assert response.status_code == 401
        assert "problems" not in response.json()

    def test_a_route_added_to_a_guarded_router_is_guarded_without_saying_so(self) -> None:
        """A route that changes something needs the token and the page, and one that only reads needs the token."""
        verifier = RequestVerifier(ACCESS_TOKEN)
        router = APIRouter(route_class=api.guarded_route_class(ACCESS_TOKEN, verifier, BodySealer(ACCESS_TOKEN)))

        @router.post("/added-later")
        def change(body: dict[str, int]) -> dict[str, int]:
            return body

        @router.get("/added-later")
        def read() -> dict[str, bool]:
            return {"read": True}

        app = FastAPI()
        app.include_router(router)
        bearer = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
        once = {"Authorization": sign_request(ACCESS_TOKEN, "GET", "/added-later", verifier.issue())}
        change_challenge, change_nonce = verifier.issue(), fresh_nonce()
        sealed_change = BodySealer(ACCESS_TOKEN).seal(
            b'{"n": 1}', request_seal_context("POST", "/added-later", change_challenge, change_nonce)
        )
        signed_change = {
            "Content-Type": SEALED_CONTENT_TYPE,
            "Authorization": sign_request(
                ACCESS_TOKEN, "POST", "/added-later", change_challenge, sealed_change, change_nonce
            ),
        }
        with TestClient(app, base_url=BASE_URL) as browser:
            answers = [
                browser.post("/added-later", content=self.NOT_JSON).status_code,
                browser.post("/added-later", content=self.NOT_JSON, headers=bearer).status_code,
                browser.post("/added-later", json={"n": 1}, headers={**bearer, **PAGE_HEADERS}).status_code,
                browser.get("/added-later").status_code,
                browser.get("/added-later", headers=bearer).status_code,
                browser.get("/added-later", headers=once).status_code,
                browser.get("/added-later", headers=once).status_code,
                browser.post(
                    "/added-later", content=sealed_change, headers={**FILE_PAGE_HEADERS, **signed_change}
                ).status_code,
            ]

        assert answers == [401, 403, 200, 401, 200, 200, 401, 200]

    def test_an_extra_slash_is_not_a_redirect_and_repeats_nothing(
        self, anonymous: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A path the route does not declare must not be redirected before the guard runs."""
        with caplog.at_level(logging.WARNING, logger="usagedash.api"):
            response = anonymous.patch(
                "/api/accounts/synthetic-id/",
                params={"tz": "synthetic-query"},
                content=self.NOT_JSON,
                headers={"Content-Type": "application/json", **PAGE_HEADERS},
            )

        assert response.status_code == 404
        assert response.headers.get("location") is None
        assert "synthetic-query" not in response.text
        assert "synthetic-id" not in response.text
        assert caplog.records == []

    def test_an_extra_slash_does_not_skip_the_origin_check(
        self, client: TestClient, seeded: dict[str, int], caplog: pytest.LogCaptureFixture
    ) -> None:
        headers = {"X-Requested-With": "usagedash", "Origin": "http://attacker.example"}
        with caplog.at_level(logging.WARNING, logger="usagedash.api"):
            slashed = self.rename(client, f"{seeded['active']}/", headers)

        assert slashed.status_code == 404
        assert slashed.headers.get("location") is None
        assert caplog.records == []

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="usagedash.api"):
            canonical = self.rename(client, seeded["active"], headers)

        assert canonical.status_code == 403
        assert [record.getMessage() for record in caplog.records] == [
            "Refused PATCH /api/accounts/{account_id}: it did not come from the dashboard page"
        ]

    def test_a_signed_read_with_an_extra_slash_is_not_redirected(self, client: TestClient) -> None:
        slashed = client.get("/api/state/")

        assert slashed.status_code == 404
        assert slashed.headers.get("location") is None
        assert client.get("/api/state").status_code == 200


class TestAnAnswerNeverRepeatsWhatARequestSent:
    """FastAPI's own answer to a request that fails validation quotes the rejected value."""

    def test_an_overlong_time_zone_name_is_not_echoed(self, client: TestClient) -> None:
        name = "Synthetic/" + "Z" * 60

        response = client.get("/api/tokens", params={"tz": name})

        assert response.status_code == 422
        assert response.json() == {
            "detail": "The request is not one this API accepts.",
            "problems": ["query: string_too_long"],
        }
        assert "Synthetic" not in response.text

    def test_a_rename_body_is_not_echoed_nor_a_field_name_it_made_up(
        self, client: TestClient, seeded: dict[str, int]
    ) -> None:
        body = {"label": "x" * 61, "synthetic-field-name": "synthetic-value"}

        response = client.patch(f"/api/accounts/{seeded['active']}", json=body, headers=PAGE_HEADERS)

        assert response.status_code == 422
        assert response.json()["problems"] == ["body: extra_forbidden", "body: string_too_long"]
        assert "synthetic" not in response.text
        assert "xxx" not in response.text

    def test_an_id_that_is_not_one_is_named_by_where_it_was_and_not_quoted(self, client: TestClient) -> None:
        response = client.patch("/api/accounts/synthetic-id", json={"label": "Nobody"}, headers=PAGE_HEADERS)

        assert response.status_code == 422
        assert response.json()["problems"] == ["path: int_parsing"]
        assert "synthetic" not in response.text

    def test_a_body_that_is_not_json_is_not_echoed(self, client: TestClient, seeded: dict[str, int]) -> None:
        response = client.patch(
            f"/api/accounts/{seeded['active']}",
            content=b"{synthetic-not-json",
            headers={**PAGE_HEADERS, "Content-Type": "application/json"},
        )

        assert response.status_code == 422
        assert response.json()["problems"] == ["body: json_invalid"]
        assert "synthetic" not in response.text

    def test_a_failure_in_no_known_part_of_a_request_is_named_as_the_request_s(self) -> None:
        assert api._request_part(()) == "request"
        assert api._request_part(("synthetic-part", "tz")) == "request"
        assert api._request_part((0,)) == "request"
        assert api._request_part(("query", "tz")) == "query"

    def test_an_error_of_another_kind_given_to_the_handler_is_answered_without_details(self) -> None:
        response = asyncio.run(api._refuse_invalid_request(None, RuntimeError("synthetic")))  # type: ignore[arg-type]

        assert response.status_code == 422
        assert json.loads(bytes(response.body)) == {
            "detail": "The request is not one this API accepts.",
            "problems": [],
        }


class TestRename:
    def test_renames_the_account_and_trims_surrounding_whitespace(
        self, client: TestClient, seeded: dict[str, int], database: Database
    ) -> None:
        response = client.patch(
            f"/api/accounts/{seeded['active']}", json={"label": "  Home Max  "}, headers=PAGE_HEADERS
        )

        assert response.json() == {"id": str(seeded["active"]), "label": "Home Max"}
        with database.connect() as conn:
            assert db.account_label(conn, seeded["active"]) == "Home Max"

    @pytest.mark.parametrize("label", ["", "   ", "x" * 61])
    def test_rejects_an_empty_or_overlong_label(self, client: TestClient, seeded: dict[str, int], label: str) -> None:
        response = client.patch(f"/api/accounts/{seeded['active']}", json={"label": label}, headers=PAGE_HEADERS)
        assert response.status_code == 422

    def test_accepts_a_label_at_the_length_limit(self, client: TestClient, seeded: dict[str, int]) -> None:
        response = client.patch(f"/api/accounts/{seeded['active']}", json={"label": "x" * 60}, headers=PAGE_HEADERS)
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "label",
        ["two\nlines", "tab\there", "nul\x00byte", "bell\x07", "right-to-left\u202eoverride", "zero\u200bwidth"],
    )
    def test_rejects_a_label_with_a_character_that_does_not_print(
        self, client: TestClient, database: Database, seeded: dict[str, int], label: str
    ) -> None:
        response = client.patch(f"/api/accounts/{seeded['active']}", json={"label": label}, headers=PAGE_HEADERS)

        assert response.status_code == 422
        with database.read() as conn:
            assert db.account_label(conn, seeded["active"]) == "Personal"

    @pytest.mark.parametrize("label", ["Travail équipe", "仕事", "Work · team (EU)"])
    def test_accepts_printable_text_in_any_script(self, client: TestClient, seeded: dict[str, int], label: str) -> None:
        response = client.patch(f"/api/accounts/{seeded['active']}", json={"label": label}, headers=PAGE_HEADERS)

        assert response.json() == {"id": str(seeded["active"]), "label": label}

    @pytest.mark.parametrize(
        "body", [{"label": 5}, {"label": True}, {"label": ["Work"]}, {"label": "Work", "id": 2}, {}, []]
    )
    def test_rejects_a_body_that_is_not_exactly_one_text_label(
        self, client: TestClient, seeded: dict[str, int], body: object
    ) -> None:
        response = client.patch(f"/api/accounts/{seeded['active']}", json=body, headers=PAGE_HEADERS)

        assert response.status_code == 422

    @pytest.mark.parametrize("account_id", ["0", "-1", "9223372036854775808", "1" + "0" * 30, "1.5", "one"])
    def test_an_id_no_stored_row_could_have_is_refused_before_it_reaches_the_store(
        self, client: TestClient, account_id: str
    ) -> None:
        response = client.patch(f"/api/accounts/{account_id}", json={"label": "Nobody"}, headers=PAGE_HEADERS)

        assert response.status_code == 422

    def test_the_largest_possible_row_id_is_looked_up_and_not_found(self, client: TestClient) -> None:
        response = client.patch("/api/accounts/9223372036854775807", json={"label": "Nobody"}, headers=PAGE_HEADERS)

        assert response.status_code == 404

    def test_an_unknown_account_is_not_found(self, client: TestClient) -> None:
        assert client.patch("/api/accounts/999", json={"label": "Nobody"}, headers=PAGE_HEADERS).status_code == 404


class AdjustableClock:
    """A wall clock the tests move, and the monotonic clock beside it, which only ``advance`` moves."""

    def __init__(self, start: datetime):
        self.now = start
        self.elapsed = 0.0

    def __call__(self) -> datetime:
        return self.now

    def monotonic(self) -> float:
        return self.elapsed

    def advance(self, by: timedelta) -> None:
        """Let ``by`` pass: both clocks move."""
        self.now += by
        self.elapsed += by.total_seconds()


class CountingSource:
    """A logged-out login that counts how many cycles have reached it and can hold a cycle open."""

    def __init__(self) -> None:
        self.cycles = 0
        self.hold = threading.Event()
        self.hold.set()

    @property
    def source_id(self) -> str:
        return "claude:home"

    @property
    def provider(self) -> Provider:
        return Provider.CLAUDE

    def current_identity(self) -> AccountIdentity | None:
        return None

    def read_quota(self, now: datetime) -> QuotaReading:
        self.cycles += 1
        self.hold.wait(WAIT_TIMEOUT_SECONDS)
        raise NoQuotaDataError("not logged in")


@contextlib.contextmanager
def _collecting_at(
    settings: Settings, database: Database, clock: AdjustableClock
) -> Iterator[tuple[TestClient, CountingSource, AdjustableClock]]:
    """A signed-in browser whose first collection has finished and whose clock is still at that finish."""
    source = CountingSource()
    collector = Collector(
        database, [SourceBundle(source, None, 0)], clock, poll_interval_seconds=3600, backfill_days=35
    )
    app = create_app(settings, database, collector, clock, clock.monotonic, access_token=ACCESS_TOKEN)
    with TestClient(app, base_url=BASE_URL) as live_client:
        browser = signed_in(live_client)
        wait_until(lambda: browser.post("/api/refresh", headers=PAGE_HEADERS).status_code == 429)
        yield browser, source, clock
        source.hold.set()


class TestAccountIdsABrowserCannotHoldAsNumbers:
    """An id is an SQLite row id. A browser's numbers are exact only up to 2**53, so ids travel as text."""

    FIRST, SECOND = 2**53, 2**53 + 1

    @pytest.fixture
    def neighbours(self, database: Database, now: datetime) -> None:
        """Two accounts whose ids are the same number to a browser, each with a reading and usage."""
        with database.connect() as conn:
            for external_id, label, large_id in (
                ("org-a", "Account A", self.FIRST),
                ("org-b", "Account B", self.SECOND),
            ):
                identity = make_identity(external_id=external_id, label=label)
                assigned = db.upsert_account(conn, identity, now)
                conn.execute("UPDATE accounts SET id = ? WHERE id = ?", (large_id, assigned))
                db.record_reading(conn, large_id, make_reading(identity, now, make_window(used_percent=10)))
                response = MessageUsage(
                    f"response-{external_id}".encode(),
                    datetime(2026, 10, 3, 13, 30, tzinfo=UTC),
                    large_id,
                    "model-x",
                    TokenCounts(input_tokens=1, output_tokens=1),
                )
                db.add_message_usage(conn, [response])
            db.set_source_state(conn, "claude:home", Provider.CLAUDE, 0, self.FIRST, now)

    def test_the_two_ids_are_one_number_to_a_browser(self) -> None:
        assert float(self.FIRST) == float(self.SECOND)

    def test_every_id_in_the_state_is_exact_text(self, client: TestClient, neighbours: None) -> None:
        state = client.get("/api/state").json()
        claude = next(provider for provider in state["providers"] if provider["provider"] == "claude")

        assert [account["id"] for account in claude["accounts"]] == [str(self.FIRST), str(self.SECOND)]
        assert claude["advice"]["active_account_id"] == str(self.FIRST)
        assert sorted(series["account_id"] for series in state["history"]) == [str(self.FIRST), str(self.SECOND)]

    def test_every_id_in_the_token_report_is_exact_text(self, client: TestClient, neighbours: None) -> None:
        report = client.get("/api/tokens", params={"range": "24h"}).json()

        assert sorted(series["account_id"] for series in report["series"]) == [str(self.FIRST), str(self.SECOND)]
        assert sorted(model["account_id"] for model in report["models"]) == [str(self.FIRST), str(self.SECOND)]
        assert sorted(account["account_id"] for account in report["costs"]["accounts"]) == [
            str(self.FIRST),
            str(self.SECOND),
        ]
        assert all(isinstance(account["priced_microdollars"], str) for account in report["costs"]["accounts"])

    def test_a_token_count_past_a_browsers_exact_integers_is_decimal_text(
        self, client: TestClient, database: Database, now: datetime
    ) -> None:
        """A JSON number would already be 2**53 by the time the page read 2**53 + 1."""
        exact = 2**53 + 1
        with database.connect() as conn:
            account_id = db.upsert_account(conn, make_identity(external_id="org-big", label="Account A"), now)
            db.add_message_usage(
                conn,
                [
                    MessageUsage(
                        b"response-big",
                        datetime(2026, 10, 3, 13, 30, tzinfo=UTC),
                        account_id,
                        "model-x",
                        TokenCounts(input_tokens=exact, output_tokens=exact),
                    )
                ],
            )

        report = client.get("/api/tokens", params={"range": "24h"}).json()
        counted = next(point for series in report["series"] for point in series["points"] if point["input"] != "0")

        assert counted["input"] == str(exact)
        assert counted["output"] == str(exact)
        assert counted["total"] == str(exact * 2)
        assert report["series"][0]["totals"]["input"] == str(exact)
        assert report["models"][0]["totals"]["total"] == str(exact * 2)
        assert report["provider_totals"][0]["totals"]["total"] == str(exact * 2)
        cost = report["costs"]["accounts"][0]
        assert cost["account_id"] == report["series"][0]["account_id"]
        assert isinstance(cost["priced_microdollars"], str)
        assert cost["unpriced_tokens"] == str(exact * 2)
        assert cost["models"][0]["microdollars"] is None

    def test_a_rename_reaches_the_account_it_names_and_not_its_neighbour(
        self, client: TestClient, neighbours: None
    ) -> None:
        response = client.patch(f"/api/accounts/{self.SECOND}", json={"label": "Work"}, headers=PAGE_HEADERS)

        claude = next(p for p in client.get("/api/state").json()["providers"] if p["provider"] == "claude")
        assert response.json() == {"id": str(self.SECOND), "label": "Work"}
        assert [(account["id"], account["label"]) for account in claude["accounts"]] == [
            (str(self.FIRST), "Account A"),
            (str(self.SECOND), "Work"),
        ]


class TestRetailCost:
    """The token report prices each model at the published short-prompt rate and leaves the rest out."""

    def test_a_priced_model_and_an_unpriced_model_are_both_on_the_account(
        self, client: TestClient, database: Database, now: datetime
    ) -> None:
        with database.connect() as conn:
            account_id = db.upsert_account(conn, make_identity(external_id="org-cost", label="Work"), now)
            db.add_message_usage(
                conn,
                [
                    MessageUsage(
                        b"response-priced",
                        datetime(2026, 10, 3, 13, 30, tzinfo=UTC),
                        account_id,
                        "claude-opus-4-7",
                        TokenCounts(input_tokens=1_000_000),
                    ),
                    MessageUsage(
                        b"response-unpriced",
                        datetime(2026, 10, 3, 13, 30, tzinfo=UTC),
                        account_id,
                        "not-a-model",
                        TokenCounts(input_tokens=3),
                    ),
                ],
            )

        response = client.get("/api/tokens", params={"range": "24h"})
        report = response.json()
        account = report["costs"]["accounts"][0]

        assert response.status_code == 200
        assert b'"priced_microdollars"' in response.content
        assert b'"unpriced_tokens"' in response.content
        assert b'"priced_microdollars"' != b'"unpriced_tokens"'
        assert account["priced_microdollars"] == "5000000"
        assert account["unpriced_tokens"] == "3"
        assert [model["model"] for model in account["models"]] == ["claude-opus-4-7", "not-a-model"]
        assert account["models"][0]["microdollars"] == "5000000"
        assert account["models"][0]["priced"] is True
        assert account["models"][1]["microdollars"] is None
        assert account["models"][1]["unpriced_tokens"] == "3"
        assert report["models"][0]["totals"]["input"] == "1000000"
        assert report["models"][1]["totals"]["total"] == "3"
        assert "not a subscription charge" in report["costs"]["note"]


class TestRefresh:
    @pytest.fixture
    def running(
        self, settings: Settings, database: Database
    ) -> Iterator[tuple[TestClient, CountingSource, AdjustableClock]]:
        """A signed-in browser against an app whose collection loop is running."""
        clock = AdjustableClock(datetime(2026, 10, 3, 14, 0, tzinfo=UTC))
        source = CountingSource()
        collector = Collector(
            database, [SourceBundle(source, None, 0)], clock, poll_interval_seconds=3600, backfill_days=35
        )
        app = create_app(settings, database, collector, clock, clock.monotonic, access_token=ACCESS_TOKEN)
        with TestClient(app, base_url=BASE_URL) as live_client:
            browser = signed_in(live_client)
            # Why a refresh refused for spacing: only that shows the loop has recorded the first cycle's
            # return; the collector's own finish time is set in the worker thread, before the loop's callback.
            wait_until(lambda: browser.post("/api/refresh", headers=PAGE_HEADERS).status_code == 429)
            yield browser, source, clock
            source.hold.set()

    def test_is_unavailable_when_this_process_runs_no_collector(self, client: TestClient) -> None:
        assert client.post("/api/refresh", headers=PAGE_HEADERS).status_code == 503

    def test_requires_the_custom_header_and_origin(self, client: TestClient) -> None:
        assert client.post("/api/refresh").status_code == 403

    def test_a_collection_runs_at_startup_without_being_asked(
        self, running: tuple[TestClient, CountingSource, AdjustableClock]
    ) -> None:
        live_client, source, _ = running
        assert source.cycles == 1
        assert live_client.get("/api/state").json()["collector"]["last_finished_at"] == "2026-10-03T14:00:00+00:00"

    def test_starts_another_collection_once_the_minimum_spacing_has_passed(
        self, running: tuple[TestClient, CountingSource, AdjustableClock]
    ) -> None:
        live_client, source, clock = running
        clock.advance(MIN_REFRESH_SPACING)

        response = live_client.post("/api/refresh", headers=PAGE_HEADERS)

        assert (response.status_code, response.json()) == (202, {"status": "scheduled"})
        wait_until(lambda: source.cycles == 2)

    def test_a_refresh_is_refused_once_the_collection_loop_has_stopped(
        self, settings: Settings, database: Database, now: datetime
    ) -> None:
        """Answering "scheduled" would promise a cycle that nothing is left to run."""
        app = create_app(settings, database, NoCycleToRun(), lambda: now, access_token=ACCESS_TOKEN)
        with TestClient(app, base_url=BASE_URL) as live_client:
            signed_in(live_client)
            wait_until(lambda: live_client.post("/api/refresh", headers=PAGE_HEADERS).status_code == 503)

            response = live_client.post("/api/refresh", headers=PAGE_HEADERS)

        assert (response.status_code, response.json()) == (
            503,
            {"detail": "collection has stopped in this process; its log says why"},
        )

    def test_spacing_follows_time_that_passed_whatever_the_wall_clock_is_set_to(
        self, running: tuple[TestClient, CountingSource, AdjustableClock]
    ) -> None:
        """A clock set back an hour must not refuse for an hour; one set forward must not allow at once."""
        live_client, source, clock = running
        clock.now += timedelta(hours=1)
        set_forward = live_client.post("/api/refresh", headers=PAGE_HEADERS)
        clock.now -= timedelta(hours=2)
        clock.advance(MIN_REFRESH_SPACING)

        set_back = live_client.post("/api/refresh", headers=PAGE_HEADERS)

        assert set_forward.status_code == 429
        assert (set_back.status_code, set_back.json()) == (202, {"status": "scheduled"})
        wait_until(lambda: source.cycles == 2)

    def test_spacing_is_unknown_before_any_cycle_has_returned(self) -> None:
        assert CollectionLoop(NoCycleToRun(), 3600, lambda: NOW).seconds_since_last_attempt is None

    def test_refuses_a_refresh_moments_after_the_last_collection(
        self, running: tuple[TestClient, CountingSource, AdjustableClock]
    ) -> None:
        live_client, source, clock = running
        clock.advance(MIN_REFRESH_SPACING - timedelta(seconds=1))

        response = live_client.post("/api/refresh", headers=PAGE_HEADERS)

        assert response.status_code == 429
        assert response.headers["retry-after"] == "20"
        assert response.json()["detail"] == "refresh attempted moments ago"
        assert source.cycles == 1

    def test_reports_a_collection_already_in_progress_instead_of_queueing_another(
        self, running: tuple[TestClient, CountingSource, AdjustableClock]
    ) -> None:
        live_client, source, clock = running
        clock.advance(MIN_REFRESH_SPACING)
        source.hold.clear()
        live_client.post("/api/refresh", headers=PAGE_HEADERS)
        wait_until(lambda: source.cycles == 2)

        response = live_client.post("/api/refresh", headers=PAGE_HEADERS)

        assert (response.status_code, response.json()) == (202, {"status": "already_running"})
        assert live_client.get("/api/state").json()["collector"]["running"] is True
        source.hold.set()
        wait_until(lambda: live_client.get("/api/state").json()["collector"]["running"] is False)
        assert source.cycles == 2

    def test_a_cycle_handed_to_its_worker_is_in_progress_before_the_collector_says_it_is_running(
        self, settings: Settings, database: Database, now: datetime
    ) -> None:
        """Waking the loop for work that is already on its way would start another cycle as soon as it finished."""
        collector = UnannouncedCollector()
        app = create_app(settings, database, collector, lambda: now, access_token=ACCESS_TOKEN)
        with TestClient(app, base_url=BASE_URL) as live_client:
            wait_until(lambda: collector.started == 1)

            response = signed_in(live_client).post("/api/refresh", headers=PAGE_HEADERS)

            collector.release.set()
            wait_until(lambda: collector.finished == 1)
        assert (response.status_code, response.json()) == (202, {"status": "already_running"})
        assert collector.started == 1

    def test_a_cycle_that_failed_is_spaced_out_like_one_that_succeeded(
        self, settings: Settings, database: Database
    ) -> None:
        """The collector's own finish time only moves on success, so it cannot be what spaces attempts out."""
        clock = AdjustableClock(datetime(2026, 10, 3, 14, 0, tzinfo=UTC))
        collector = AlwaysFailingCollector()
        app = create_app(settings, database, collector, clock, clock.monotonic, access_token=ACCESS_TOKEN)
        with TestClient(app, base_url=BASE_URL) as live_client:
            signed_in(live_client)
            wait_until(lambda: collector.attempts == 1 and not self.is_collecting(live_client))

            too_soon = [live_client.post("/api/refresh", headers=PAGE_HEADERS) for _ in range(3)]
            clock.advance(MIN_REFRESH_SPACING)
            spaced = live_client.post("/api/refresh", headers=PAGE_HEADERS)
            wait_until(lambda: collector.attempts == 2)

        assert [response.status_code for response in too_soon] == [429, 429, 429]
        assert too_soon[0].json()["detail"] == "refresh attempted moments ago"
        assert (spaced.status_code, spaced.json()) == (202, {"status": "scheduled"})
        assert collector.last_finished_at is None

    def test_ten_elapsed_seconds_across_a_spring_forward_are_still_too_soon(
        self, settings: Settings, database: Database
    ) -> None:
        zone = ZoneInfo("America/New_York")
        clock = AdjustableClock(datetime(2026, 3, 8, 1, 59, 55, tzinfo=zone))
        with _collecting_at(settings, database, clock) as (live_client, source, clock):
            clock.now = datetime(2026, 3, 8, 3, 0, 5, tzinfo=zone)
            clock.elapsed += 10

            response = live_client.post("/api/refresh", headers=PAGE_HEADERS)

        assert response.status_code == 429
        assert response.json()["detail"] == "refresh attempted moments ago"
        assert source.cycles == 1

    def test_thirty_elapsed_seconds_across_a_fall_back_are_long_enough(
        self, settings: Settings, database: Database
    ) -> None:
        zone = ZoneInfo("America/New_York")
        clock = AdjustableClock(datetime(2026, 11, 1, 1, 59, 50, fold=0, tzinfo=zone))
        with _collecting_at(settings, database, clock) as (live_client, source, clock):
            clock.now = datetime(2026, 11, 1, 1, 0, 20, fold=1, tzinfo=zone)
            clock.elapsed += 30

            response = live_client.post("/api/refresh", headers=PAGE_HEADERS)

            assert (response.status_code, response.json()) == (202, {"status": "scheduled"})
            wait_until(lambda: source.cycles == 2)

    def is_collecting(self, live_client: TestClient) -> bool:
        return live_client.post("/api/refresh", headers=PAGE_HEADERS).json().get("status") == "already_running"


class AlwaysFailingCollector:
    """A collector none of whose cycles reaches its end, so it never records a finish time."""

    is_running = False
    last_finished_at = None

    def __init__(self) -> None:
        self.attempts = 0

    def run_cycle(self) -> None:
        self.attempts += 1
        raise ConfigError("cannot open the lock as a private file: PermissionError")


class UnannouncedCollector:
    """A collector whose cycle is under way without it saying so, as a real one is until its thread has the lock."""

    is_running = False
    last_finished_at = None

    def __init__(self) -> None:
        self.started = 0
        self.finished = 0
        self.release = threading.Event()

    def run_cycle(self) -> None:
        self.started += 1
        self.release.wait(WAIT_TIMEOUT_SECONDS)
        self.finished += 1


class FailsWhenReleased(UnannouncedCollector):
    """A collector whose cycle is under way until released, and then fails with a defect."""

    def run_cycle(self) -> None:
        super().run_cycle()
        raise RuntimeError("synthetic-secret-in-a-late-defect")


class NoCycleToRun:
    """A collector so broken that a cycle cannot even be handed to a worker."""

    @property
    def run_cycle(self) -> Callable[[], None]:
        raise RuntimeError("synthetic-secret-in-a-loop-defect")


def without_quoted_source(log_text: str) -> str:
    """Remove the source lines a stack quotes, which hold the synthetic messages as written in this file."""
    for message in ("synthetic-secret-in-a-late-defect", "synthetic-secret-in-a-loop-defect"):
        log_text = log_text.replace(f'raise RuntimeError("{message}")', "")
    return log_text


def logged_by_asyncio(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the problems asyncio itself logged, which is where an outcome nobody retrieved ends up."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "asyncio" and record.levelno >= logging.WARNING
    ]


class FlakyCollector:
    """Fails the first cycle with a defect and the second because one is already running, then succeeds."""

    def __init__(self) -> None:
        self.calls = 0

    def run_cycle(self) -> None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("synthetic-secret-in-a-defect")
        if self.calls == 2:
            raise CycleAlreadyRunningError("busy")


class CountingCollector:
    """A collector whose cycles return at once and count how many ran."""

    is_running = False
    last_finished_at = None

    def __init__(self) -> None:
        self.calls = 0

    def run_cycle(self) -> None:
        self.calls += 1


class TestCollectionLoop:
    def test_a_wake_set_before_the_loop_starts_does_not_start_a_second_cycle(
        self,
    ) -> None:
        """The first cycle is that wake. Leaving the event set would skip the spacing after it."""
        collector = CountingCollector()
        stalled_at: list[int] = []

        async def drive() -> None:
            collection = CollectionLoop(collector, 3600, lambda: NOW)
            collection.run_soon()
            task = asyncio.create_task(collection.run())
            while collector.calls < 1 or collection.is_busy:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            deadline = time.monotonic() + 0.4
            while time.monotonic() < deadline and collector.calls < 2:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            stalled_at.append(collector.calls)
            collection.run_soon()
            while collector.calls < 2 or collection.is_busy:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))

        assert stalled_at == [1]
        assert collector.calls == 2

    def test_a_wake_during_the_first_cycle_still_starts_the_next_one(self) -> None:
        """A wake that arrives after the loop has started is the request for the following cycle."""
        collector = UnannouncedCollector()

        async def drive() -> None:
            collection = CollectionLoop(collector, 3600, lambda: NOW)
            task = asyncio.create_task(collection.run())
            while collector.started < 1:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            collection.run_soon()
            collector.release.set()
            while collector.finished < 2:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))

        assert collector.started == 2
        assert collector.finished == 2

    def test_keeps_collecting_after_a_cycle_raises(self, caplog: pytest.LogCaptureFixture) -> None:
        collector = FlakyCollector()

        async def drive() -> None:
            collection = CollectionLoop(collector, 3600, lambda: NOW)
            task = asyncio.create_task(collection.run())
            for completed in (1, 2, 3):
                while collector.calls < completed or collection.is_busy:
                    await asyncio.sleep(WAIT_POLL_SECONDS)
                collection.run_soon()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        with caplog.at_level(logging.INFO, logger="usagedash.api"):
            asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))

        assert collector.calls >= 3
        messages = [record.getMessage() for record in caplog.records]
        assert "Skipping scheduled cycle; one is already running" in messages
        failure = next(message for message in messages if message.startswith("Collection cycle failed unexpectedly"))
        assert "RuntimeError raised at:" in failure
        assert "in run_cycle" in failure
        assert "synthetic-secret-in-a-defect" not in caplog.text.replace(
            'raise RuntimeError("synthetic-secret-in-a-defect")', ""
        )
        assert all(record.exc_info is None for record in caplog.records)

    def test_a_cycle_that_fails_deliberately_is_logged_without_its_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class RefusedCollector:
            calls = 0

            def run_cycle(self) -> None:
                self.calls += 1
                raise ConfigError("cannot open the lock as a private file: PermissionError")

        collector = RefusedCollector()

        async def drive() -> None:
            collection = CollectionLoop(collector, 3600, lambda: NOW)
            task = asyncio.create_task(collection.run())
            while collector.calls < 1 or collection.is_busy:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        with caplog.at_level(logging.INFO, logger="usagedash.api"):
            asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))

        failure = next(
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("Collection cycle failed")
        )
        assert "ConfigError raised at:" in failure
        without_the_source_line = failure.replace(
            'raise ConfigError("cannot open the lock as a private file: PermissionError")', ""
        )
        assert "cannot open the lock as a private file" not in without_the_source_line

    def test_a_typed_cycle_failure_does_not_log_an_email_bearing_configured_path(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ConfigError message can hold a configured path. The cycle log must not."""
        local = "account-a" + "@" + "example.invalid"
        path = Path("/synthetic") / local / "collector.lock"

        def refuse_to_open(*_args: object, **_kwargs: object) -> int:
            raise PermissionError("synthetic failure")

        monkeypatch.setattr("usagedash.private.os.open", refuse_to_open)
        with pytest.raises(ConfigError) as caught:
            open_private_file(path)

        with caplog.at_level(logging.ERROR, logger="usagedash.api"):
            api._log_cycle_failure(caught.value)

        assert local not in caplog.text
        assert str(path) not in caplog.text
        assert "ConfigError raised at:" in caplog.text

    def test_is_busy_from_the_moment_a_cycle_is_handed_to_its_worker_until_the_worker_returns(self) -> None:
        collector = UnannouncedCollector()
        observed: list[bool] = []

        async def drive() -> None:
            collection = CollectionLoop(collector, 3600, lambda: NOW)
            observed.append(collection.is_busy)
            task = asyncio.create_task(collection.run())
            await asyncio.sleep(0)
            observed.append(collection.is_busy)
            collector.release.set()
            while collector.finished < 1 or collection.is_busy:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            observed.append(collection.is_busy)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))

        assert observed == [False, True, False]
        assert collector.is_running is False

    @pytest.mark.parametrize("collector", [UnannouncedCollector(), AlwaysFailingCollector()])
    def test_records_when_each_attempt_ended_whether_it_succeeded_or_failed(self, collector: Any) -> None:
        clock = AdjustableClock(NOW)
        ended: list[datetime | None] = []

        async def drive() -> None:
            collection = CollectionLoop(collector, 3600, clock)
            ended.append(collection.last_attempt_ended_at)
            task = asyncio.create_task(collection.run())
            await asyncio.sleep(0)
            clock.now += timedelta(seconds=7)
            getattr(collector, "release", threading.Event()).set()
            while collection.is_busy:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            ended.append(collection.last_attempt_ended_at)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))

        assert ended[0] is None
        assert ended[1] in {NOW, NOW + timedelta(seconds=7)}
        assert collector.last_finished_at is None

    def test_cancelling_the_loop_waits_for_the_cycle_in_flight_to_return(self) -> None:
        """Cancelling does not stop the worker thread, so shutdown must not finish while it is still collecting."""
        collector = UnannouncedCollector()
        finished_when_cancellation_completed: list[int] = []

        async def drive() -> None:
            collection = CollectionLoop(collector, 3600, lambda: NOW)
            task = asyncio.create_task(collection.run())
            while collector.started < 1:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            task.cancel()
            asyncio.get_running_loop().call_later(10 * WAIT_POLL_SECONDS, collector.release.set)
            with contextlib.suppress(asyncio.CancelledError):
                await task
            finished_when_cancellation_completed.append(collector.finished)
            assert collection.is_busy is False

        asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))

        assert finished_when_cancellation_completed == [1]
        assert collector.started == 1

    def test_a_cycle_that_fails_while_shutdown_waits_for_it_is_still_logged_without_its_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        collector = FailsWhenReleased()

        async def drive() -> None:
            task = asyncio.create_task(CollectionLoop(collector, 3600, lambda: NOW).run())
            while collector.started < 1:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            task.cancel()
            collector.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        with caplog.at_level(logging.INFO, logger="usagedash.api"):
            asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))

        messages = [record.getMessage() for record in caplog.records]
        assert any(message.startswith("Collection cycle failed unexpectedly") for message in messages)
        assert "synthetic-secret-in-a-late-defect" not in without_quoted_source(caplog.text)

    def test_cancelling_again_does_not_shorten_the_wait_for_the_cycle_in_flight(self) -> None:
        """The worker is still collecting after any number of cancellations, so the wait must still be on."""
        collector = UnannouncedCollector()
        observed: list[tuple[bool, bool, int]] = []

        async def drive() -> None:
            collection = CollectionLoop(collector, 3600, lambda: NOW)
            task = asyncio.create_task(collection.run())
            while collector.started < 1:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(WAIT_POLL_SECONDS)
            observed.append((task.done(), collection.is_busy, collector.finished))
            collector.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            observed.append((task.done(), collection.is_busy, collector.finished))

        asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))

        assert observed == [(False, True, 0), (True, False, 1)]
        assert collector.started == 1

    def test_a_cycle_that_fails_after_repeated_cancellation_is_logged_by_this_code_and_not_by_asyncio(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An outcome nobody retrieved is logged by asyncio when it is collected, with the exception's message."""
        collector = FailsWhenReleased()

        async def drive() -> None:
            task = asyncio.create_task(CollectionLoop(collector, 3600, lambda: NOW).run())
            while collector.started < 1:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(WAIT_POLL_SECONDS)
            collector.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        with caplog.at_level(logging.DEBUG):
            asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))
            gc.collect()

        failures = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("Collection cycle failed unexpectedly")
        ]
        assert len(failures) == 1
        assert "RuntimeError raised at:" in failures[0]
        assert logged_by_asyncio(caplog) == []
        assert "synthetic-secret-in-a-late-defect" not in without_quoted_source(caplog.text)

    def test_a_cycle_whose_worker_was_cancelled_before_it_ran_ends_the_attempt_without_a_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An executor that is shut down cancels the work still queued in it."""
        collector = UnannouncedCollector()
        only_thread_is_taken = threading.Event()
        observed: list[object] = []

        async def drive() -> None:
            executor = ThreadPoolExecutor(max_workers=1)
            asyncio.get_running_loop().set_default_executor(executor)
            executor.submit(only_thread_is_taken.wait, WAIT_TIMEOUT_SECONDS)
            collection = CollectionLoop(collector, 3600, lambda: NOW)
            task = asyncio.create_task(collection.run())
            await asyncio.sleep(0)
            observed.append(collection.is_busy)
            executor.shutdown(wait=False, cancel_futures=True)
            only_thread_is_taken.set()
            while collection.is_busy:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            observed.append(collection.last_attempt_ended_at)
            await asyncio.sleep(WAIT_POLL_SECONDS)
            observed.append(task.done())
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        with caplog.at_level(logging.DEBUG):
            asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))
            gc.collect()

        assert observed == [True, NOW, False]
        assert collector.started == 0
        assert [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING] == []

    def test_a_cycle_that_cannot_be_handed_to_a_worker_does_not_leave_the_loop_marked_busy(self) -> None:
        busy_after_the_loop_ended: list[bool] = []

        async def drive() -> None:
            collection = CollectionLoop(NoCycleToRun(), 3600, lambda: NOW)
            with pytest.raises(RuntimeError):
                await asyncio.create_task(collection.run())
            busy_after_the_loop_ended.append(collection.is_busy)

        asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))

        assert busy_after_the_loop_ended == [False]


class TestCollectionAcrossTheAppsLifetime:
    """The app starts the collection loop when it starts and waits for it when it shuts down."""

    def test_shutdown_cancelled_while_it_waits_still_waits_for_the_cycle_in_flight(
        self, settings: Settings, database: Database
    ) -> None:
        """Shutdown that returned now would close the database under a worker that is still writing to it."""
        collector = UnannouncedCollector()
        app = create_app(settings, database, collector, lambda: NOW, access_token=ACCESS_TOKEN)
        observed: list[tuple[bool, int]] = []

        async def drive() -> None:
            lifespan = app.router.lifespan_context(app)
            await lifespan.__aenter__()
            while collector.started < 1:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
            for _ in range(2):
                await asyncio.sleep(WAIT_POLL_SECONDS)
                shutdown.cancel()
            await asyncio.sleep(WAIT_POLL_SECONDS)
            observed.append((shutdown.done(), collector.finished))
            collector.release.set()
            with pytest.raises(asyncio.CancelledError):
                await shutdown
            observed.append((shutdown.done(), collector.finished))

        asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))

        assert observed == [(False, 0), (True, 1)]

    def test_a_cycle_that_fails_after_shutdown_was_cancelled_is_logged_without_its_message(
        self, settings: Settings, database: Database, caplog: pytest.LogCaptureFixture
    ) -> None:
        collector = FailsWhenReleased()
        app = create_app(settings, database, collector, lambda: NOW, access_token=ACCESS_TOKEN)

        async def drive() -> None:
            lifespan = app.router.lifespan_context(app)
            await lifespan.__aenter__()
            while collector.started < 1:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await asyncio.sleep(WAIT_POLL_SECONDS)
            shutdown.cancel()
            await asyncio.sleep(WAIT_POLL_SECONDS)
            collector.release.set()
            with pytest.raises(asyncio.CancelledError):
                await shutdown

        with caplog.at_level(logging.DEBUG):
            asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))
            gc.collect()

        messages = [record.getMessage() for record in caplog.records]
        assert len([message for message in messages if message.startswith("Collection cycle failed unexpectedly")]) == 1
        assert logged_by_asyncio(caplog) == []
        assert "synthetic-secret-in-a-late-defect" not in without_quoted_source(caplog.text)

    def test_shutdown_that_is_not_cancelled_completes_once_the_cycle_in_flight_has_returned(
        self, settings: Settings, database: Database
    ) -> None:
        collector = UnannouncedCollector()
        app = create_app(settings, database, collector, lambda: NOW, access_token=ACCESS_TOKEN)
        finished_when_shutdown_completed: list[int] = []

        async def drive() -> None:
            lifespan = app.router.lifespan_context(app)
            await lifespan.__aenter__()
            while collector.started < 1:
                await asyncio.sleep(WAIT_POLL_SECONDS)
            asyncio.get_running_loop().call_later(10 * WAIT_POLL_SECONDS, collector.release.set)
            await lifespan.__aexit__(None, None, None)
            finished_when_shutdown_completed.append(collector.finished)

        asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))

        assert finished_when_shutdown_completed == [1]

    def test_a_loop_that_stops_by_itself_is_logged_when_it_stops_and_without_its_message(
        self, settings: Settings, database: Database, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Nothing awaits the loop until shutdown, so without this it would stop collecting in silence."""
        app = create_app(settings, database, NoCycleToRun(), lambda: NOW, access_token=ACCESS_TOKEN)
        logged_before_shutdown: list[str] = []

        async def drive() -> None:
            lifespan = app.router.lifespan_context(app)
            await lifespan.__aenter__()
            await asyncio.sleep(WAIT_POLL_SECONDS)
            logged_before_shutdown.extend(record.getMessage() for record in caplog.records)
            await lifespan.__aexit__(None, None, None)

        with caplog.at_level(logging.DEBUG):
            asyncio.run(asyncio.wait_for(drive(), WAIT_TIMEOUT_SECONDS))
            gc.collect()

        stopped = [message for message in logged_before_shutdown if message.startswith("Collection loop stopped")]
        assert len(stopped) == 1
        assert stopped[0].startswith("Collection loop stopped; no further cycle will run\nRuntimeError raised at:")
        assert logged_by_asyncio(caplog) == []
        assert "synthetic-secret-in-a-loop-defect" not in without_quoted_source(caplog.text)


class TestStaticDashboard:
    def test_serves_the_page_and_its_assets(self, client: TestClient) -> None:
        page = client.get("/")
        assert page.status_code == 200
        assert '<script type="module" src="/static/app.js"></script>' in page.text
        assert client.get("/static/styles.css").status_code == 200

    def test_the_page_has_no_inline_script_or_style_that_the_content_policy_would_block(
        self, client: TestClient
    ) -> None:
        page = client.get("/").text
        assert "<style" not in page
        assert " style=" not in page
        assert "onclick=" not in page
        assert page.count("<script") == 1
