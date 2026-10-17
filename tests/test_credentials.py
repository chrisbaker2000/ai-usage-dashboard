"""Reading credentials and calling provider endpoints: the keychain, credential files, and HTTP."""

import subprocess
import traceback
from pathlib import Path

import httpx2
import pytest

from tests.fakes.fake_run import FakeRun
from tests.payloads import make_jwt
from usagedash import credentials
from usagedash.credentials import KeychainItem
from usagedash.errors import (
    ConfigError,
    CredentialExpiredError,
    CredentialUnavailableError,
    NotAJwtError,
    ProviderRequestError,
    ProviderSchemaError,
)
from usagedash.http import build_client, fetch_json_object

SYNTHETIC_ITEM = KeychainItem(service="Synthetic-service", account="synthetic-account")


class TestKeychainReader:
    def test_returns_the_stored_secret_using_a_fixed_argument_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = FakeRun(stdout='{"claudeAiOauth": {}}\n')
        monkeypatch.setattr(credentials.subprocess, "run", run)

        assert credentials.read_keychain_secret(SYNTHETIC_ITEM) == '{"claudeAiOauth": {}}'
        assert run.commands == [
            ["/usr/bin/security", "find-generic-password", "-a", "synthetic-account", "-s", "Synthetic-service", "-w"]
        ]

    def test_the_lookup_always_names_the_account_so_another_item_under_the_same_service_is_never_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression lock: with ``-s`` alone, ``security`` returned an ``acct="unknown"`` item (2026-10-08)."""
        run = FakeRun(stdout="secret\n")
        monkeypatch.setattr(credentials.subprocess, "run", run)

        credentials.read_keychain_secret(SYNTHETIC_ITEM)

        command = run.commands[0]
        assert command[command.index("-a") + 1] == "synthetic-account"
        assert command[command.index("-s") + 1] == "Synthetic-service"

    @pytest.mark.parametrize(
        ("run", "message"),
        [
            (FakeRun(returncode=44), "no keychain item 'Synthetic-service' under this macOS account's name"),
            (FakeRun(returncode=36), "exited 36"),
            (FakeRun(stdout="  \n"), "is empty"),
            (FakeRun(error=subprocess.TimeoutExpired("security", 10)), "TimeoutExpired"),
            (FakeRun(error=FileNotFoundError("security")), "FileNotFoundError"),
        ],
    )
    def test_every_failure_is_reported_as_an_unavailable_credential_that_does_not_name_the_account(
        self, monkeypatch: pytest.MonkeyPatch, run: FakeRun, message: str
    ) -> None:
        """The account is a login name, and the message is stored, shown on the page, and printed."""
        monkeypatch.setattr(credentials.subprocess, "run", run)
        with pytest.raises(CredentialUnavailableError, match=message) as raised:
            credentials.read_keychain_secret(SYNTHETIC_ITEM)

        assert "synthetic-account" not in "".join(traceback.format_exception(raised.value))

    def test_the_item_never_shows_its_account_however_it_is_printed(self) -> None:
        assert "synthetic-account" not in str(SYNTHETIC_ITEM)
        assert "synthetic-account" not in repr(SYNTHETIC_ITEM)
        assert "Synthetic-service" in str(SYNTHETIC_ITEM)


class TestCredentialFiles:
    def test_a_missing_file_is_an_unavailable_credential(self, tmp_path: Path) -> None:
        with pytest.raises(CredentialUnavailableError, match="does not exist"):
            credentials.read_json_object(tmp_path / "auth.json")

    @pytest.mark.parametrize("content", ["not json", "[1, 2]", '"text"'])
    def test_content_that_is_not_a_json_object_is_a_schema_error_that_does_not_echo_it(
        self, tmp_path: Path, content: str
    ) -> None:
        path = tmp_path / "auth.json"
        path.write_text(content)
        with pytest.raises(ProviderSchemaError) as raised:
            credentials.read_json_object(path)
        assert content not in str(raised.value)

    def test_reads_jwt_claims_without_needing_padding(self) -> None:
        assert credentials.decode_jwt_claims(make_jwt({"exp": 1, "email": "user-a@example.test"})) == {
            "exp": 1,
            "email": "user-a@example.test",
        }

    @pytest.mark.parametrize("token", ["opaque", "a.b", "a.!!!.c", "a.W10.c"])
    def test_rejects_values_that_are_not_a_jwt_with_an_object_payload(self, token: str) -> None:
        with pytest.raises(ProviderSchemaError):
            credentials.decode_jwt_claims(token)

    @pytest.mark.parametrize(
        ("token", "reason"),
        [
            ("header.a.signature", "not valid base64url"),
            ("header.__4.signature", "not UTF-8"),
            ("one.two.three.four", "not a JWT"),
        ],
    )
    def test_says_why_a_token_could_not_be_read_without_echoing_it(self, token: str, reason: str) -> None:
        with pytest.raises(ProviderSchemaError, match=reason) as raised:
            credentials.decode_jwt_claims(token)
        assert token not in str(raised.value)

    @pytest.mark.parametrize("token", ["opaque", "a.b", "one.two.three.four"])
    def test_a_value_that_is_not_three_parts_is_not_a_jwt_at_all(self, token: str) -> None:
        with pytest.raises(NotAJwtError):
            credentials.decode_jwt_claims(token)

    @pytest.mark.parametrize("token", ["header.a.signature", "header.__4.signature", "a.W10.c"])
    def test_a_jwt_whose_payload_cannot_be_read_is_not_mistaken_for_a_value_that_is_no_jwt(self, token: str) -> None:
        """A caller that lets a non-JWT through must still refuse a JWT it cannot read."""
        with pytest.raises(ProviderSchemaError) as raised:
            credentials.decode_jwt_claims(token)
        assert not isinstance(raised.value, NotAJwtError)

    def test_a_path_that_cannot_be_read_as_a_file_is_an_unavailable_credential(self, tmp_path: Path) -> None:
        directory = tmp_path / "auth.json"
        directory.mkdir()

        with pytest.raises(CredentialUnavailableError, match=r"cannot read .*: IsADirectoryError"):
            credentials.read_json_object(directory)


class TestHttpHelper:
    def client(self, response: httpx2.Response) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(lambda _: response))

    def test_returns_a_json_object_body(self) -> None:
        client = self.client(httpx2.Response(200, json={"ok": True}))
        assert fetch_json_object(client, "https://example.test/x", {}, service="Svc") == {"ok": True}

    @pytest.mark.parametrize("status", [401, 403])
    def test_an_auth_failure_means_the_stored_token_is_no_longer_good(self, status: int) -> None:
        with pytest.raises(CredentialExpiredError, match=f"HTTP {status}"):
            fetch_json_object(self.client(httpx2.Response(status)), "https://example.test/x", {}, service="Svc")

    @pytest.mark.parametrize(("header", "expected"), [("30", 30.0), ("soon", None), ("-5", None), (None, None)])
    def test_a_rate_limit_carries_a_usable_retry_delay_only(self, header: str | None, expected: float | None) -> None:
        headers = {} if header is None else {"Retry-After": header}
        with pytest.raises(ProviderRequestError) as raised:
            fetch_json_object(
                self.client(httpx2.Response(429, headers=headers)), "https://example.test/x", {}, service="Svc"
            )
        assert (raised.value.status_code, raised.value.retry_after_seconds) == (429, expected)

    @pytest.mark.parametrize("response", [httpx2.Response(200, text="<html>"), httpx2.Response(200, json=[1])])
    def test_a_body_that_is_not_a_json_object_is_a_schema_error(self, response: httpx2.Response) -> None:
        with pytest.raises(ProviderSchemaError):
            fetch_json_object(self.client(response), "https://example.test/x", {}, service="Svc")

    @pytest.mark.parametrize(
        "value",
        [
            "Bearer a\nb",
            "Bearer a\rb",
            "Bearer a\x00b",
            "Bearer tökén",
            " Bearer a",
            "Bearer a ",
            "Bearer\ta",
            "a\x7fb",
        ],
    )
    def test_a_header_value_a_header_cannot_carry_is_never_handed_to_the_transport(self, value: str) -> None:
        def refuse(request: httpx2.Request) -> httpx2.Response:
            raise AssertionError("the request must not be sent")

        client = httpx2.Client(transport=httpx2.MockTransport(refuse))
        headers = {"Authorization": value, "Accept": "application/json", "X-Workspace": value}

        with pytest.raises(ProviderRequestError) as raised:
            fetch_json_object(client, "https://example.test/x", headers, service="Svc")

        assert str(raised.value) == (
            "Svc request not sent: Authorization, X-Workspace holds characters a header cannot carry"
        )
        assert raised.value.status_code is None

    @pytest.mark.parametrize("value", ["", "Bearer abc.def-ghi_jkl~", "text with inner spaces", "!~"])
    def test_printable_ascii_header_values_are_sent_unchanged(self, value: str) -> None:
        seen: list[str] = []

        def record(request: httpx2.Request) -> httpx2.Response:
            seen.append(request.headers["x-probe"])
            return httpx2.Response(200, json={})

        client = httpx2.Client(transport=httpx2.MockTransport(record))

        assert fetch_json_object(client, "https://example.test/x", {"X-Probe": value}, service="Svc") == {}
        assert seen == [value]

    def test_a_transport_failure_does_not_keep_the_transport_s_own_error_which_can_quote_a_header(self) -> None:
        def fail(request: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError(f"failed while sending {request.headers['authorization']}", request=request)

        client = httpx2.Client(transport=httpx2.MockTransport(fail))
        # Why the token is assembled: a traceback prints the source line of each frame, so a literal
        # written on the calling line would appear in it whatever the code under test did.
        token = "-".join(["synthetic", "bearer", "value"])
        headers = {"Authorization": f"Bearer {token}"}

        with pytest.raises(ProviderRequestError) as raised:
            fetch_json_object(client, "https://example.test/x", headers, service="Svc")

        assert str(raised.value) == "Svc request failed: ConnectError"
        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__
        assert token not in "".join(traceback.format_exception(raised.value))

    def test_the_shared_client_never_follows_redirects_with_a_bearer_token(self) -> None:
        with build_client() as client:
            assert client.follow_redirects is False
            assert client.headers["user-agent"].startswith("usagedash/")


class TestAClientTheEnvironmentCannotConfigure:
    """Proxy and certificate settings come from the environment; one that cannot be used is a ConfigError."""

    @pytest.mark.parametrize(
        ("variable", "value", "error_type"),
        [
            pytest.param(
                "HTTPS_PROXY", "http://user:synthetic-proxy-secret@::bad::", "InvalidURL", id="a malformed proxy"
            ),
            pytest.param(
                "HTTPS_PROXY", "ftp://user:synthetic-proxy-secret@host:1", "ValueError", id="an unsupported proxy"
            ),
            pytest.param(
                "HTTPS_PROXY", "socks5://user:synthetic-proxy-secret@localhost:1", "ImportError", id="a socks proxy"
            ),
            pytest.param(
                "SSL_CERT_FILE", "/nonexistent/synthetic-proxy-secret.pem", "FileNotFoundError", id="no ca file"
            ),
        ],
    )
    def test_it_is_a_config_error_naming_the_error_type_and_nothing_from_the_setting(
        self, monkeypatch: pytest.MonkeyPatch, variable: str, value: str, error_type: str
    ) -> None:
        monkeypatch.setenv(variable, value)

        with pytest.raises(ConfigError) as raised:
            build_client()

        assert str(raised.value) == (
            f"cannot set up the HTTP client from the proxy or certificate settings: {error_type}"
        )
        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__
        assert "synthetic-proxy-secret" not in "".join(traceback.format_exception(raised.value))

    def test_a_certificate_file_that_is_not_one_is_a_config_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        not_a_certificate = tmp_path / "ca.pem"
        not_a_certificate.write_text("not a certificate\n")
        monkeypatch.setenv("SSL_CERT_FILE", str(not_a_certificate))

        with pytest.raises(ConfigError, match="certificate settings: SSLError"):
            build_client()


class TestMalformedCredentialMaterial:
    """Input that cannot be decoded is a typed error, and the error never carries the material itself."""

    SECRET = "synthetic-refresh-token-value"

    def test_a_credential_file_that_is_not_utf_8_is_a_schema_error(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        path.write_bytes(b'{"tokens": {"access_token": "caf\xe9"}}')

        with pytest.raises(ProviderSchemaError, match="is not UTF-8") as raised:
            credentials.read_json_object(path)

        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__

    def test_an_invalid_json_error_does_not_keep_the_document_it_failed_on(self) -> None:
        with pytest.raises(ProviderSchemaError, match="keychain item is not valid JSON") as raised:
            credentials.parse_json_object(f'{{"refreshToken": "{self.SECRET}", truncated', "keychain item")

        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__
        assert self.SECRET not in "".join(traceback.format_exception(raised.value))

    def test_json_nested_too_deeply_to_parse_is_a_schema_error(self) -> None:
        with pytest.raises(ProviderSchemaError, match="is not valid JSON"):
            credentials.parse_json_object("[" * 200_000, "auth file")

    def test_a_jwt_payload_that_is_not_utf_8_does_not_keep_its_bytes(self) -> None:
        with pytest.raises(ProviderSchemaError, match="not UTF-8") as raised:
            credentials.decode_jwt_claims("header.__4.signature")

        assert raised.value.__cause__ is None

    def test_keychain_output_that_is_not_text_does_not_keep_the_bytes_it_could_not_decode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A decoding error holds everything the keychain printed, tokens included, and shows it in its repr."""
        output = b'{"claudeAiOauth": {"refreshToken": "' + self.SECRET.encode() + b'\xff"}}'
        undecodable = UnicodeDecodeError("utf-8", output, len(output) - 4, len(output) - 3, "invalid start byte")
        monkeypatch.setattr(credentials.subprocess, "run", FakeRun(error=undecodable))

        with pytest.raises(CredentialUnavailableError, match="UnicodeDecodeError") as raised:
            credentials.read_keychain_secret(SYNTHETIC_ITEM)

        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__
        assert self.SECRET not in "".join(traceback.format_exception(raised.value))

    def test_a_keychain_lookup_that_times_out_does_not_keep_the_output_it_had_captured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        timed_out = subprocess.TimeoutExpired(["/usr/bin/security"], 10, output=self.SECRET)
        monkeypatch.setattr(credentials.subprocess, "run", FakeRun(error=timed_out))

        with pytest.raises(CredentialUnavailableError, match="TimeoutExpired") as raised:
            credentials.read_keychain_secret(SYNTHETIC_ITEM)

        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__

    @pytest.mark.parametrize("problem", ["missing", "directory"])
    def test_a_credential_file_that_cannot_be_read_has_no_chained_cause_either(
        self, tmp_path: Path, problem: str
    ) -> None:
        path = tmp_path / "auth.json"
        if problem == "directory":
            path.mkdir()

        with pytest.raises(CredentialUnavailableError) as raised:
            credentials.read_json_object(path)

        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__

    def test_a_jwt_payload_that_is_not_base64_does_not_keep_it(self) -> None:
        with pytest.raises(ProviderSchemaError, match="not valid base64url") as raised:
            credentials.decode_jwt_claims("header.a.signature")

        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__


class TestMalformedResponses:
    def client(self, response: httpx2.Response) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(lambda _: response))

    @pytest.mark.parametrize("header", ["Infinity", "inf", "1e309", "nan", "-inf"])
    def test_a_retry_delay_that_is_not_a_finite_number_is_not_passed_on(self, header: str) -> None:
        response = httpx2.Response(429, headers={"Retry-After": header})

        with pytest.raises(ProviderRequestError) as raised:
            fetch_json_object(self.client(response), "https://example.test/x", {}, service="Svc")

        assert raised.value.retry_after_seconds is None

    @pytest.mark.parametrize("header", ["0", "0.5", "86400"])
    def test_a_finite_non_negative_retry_delay_is_passed_on(self, header: str) -> None:
        response = httpx2.Response(429, headers={"Retry-After": header})

        with pytest.raises(ProviderRequestError) as raised:
            fetch_json_object(self.client(response), "https://example.test/x", {}, service="Svc")

        assert raised.value.retry_after_seconds == float(header)

    @pytest.mark.parametrize("header", ["86400.5", "86401", "1e18", "1e300"])
    def test_a_retry_delay_longer_than_a_day_is_cut_to_a_day(self, header: str) -> None:
        response = httpx2.Response(429, headers={"Retry-After": header})

        with pytest.raises(ProviderRequestError) as raised:
            fetch_json_object(self.client(response), "https://example.test/x", {}, service="Svc")

        assert raised.value.retry_after_seconds == 86_400.0

    @pytest.mark.parametrize(
        "body", [b"[" * 200_000, b'{"email": "caf\xe9"}', b'{"email": "user-a@example.test", truncated']
    )
    def test_a_body_that_cannot_be_parsed_is_a_schema_error_that_does_not_keep_the_body(self, body: bytes) -> None:
        response = httpx2.Response(200, content=body, headers={"Content-Type": "application/json"})

        with pytest.raises(ProviderSchemaError, match="Svc response is not JSON") as raised:
            fetch_json_object(self.client(response), "https://example.test/x", {}, service="Svc")

        assert raised.value.__cause__ is None
        assert "user-a@example.test" not in "".join(traceback.format_exception(raised.value))
