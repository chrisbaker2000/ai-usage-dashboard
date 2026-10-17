"""CLI-mode sources, exercised against a real child process speaking each vendor's protocol over real pipes."""

import json
import logging
import os
import re
import signal
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tests.factories import (
    PERSONA_EMAIL,
    WEEK,
)
from tests.payloads import (
    codex_usage_payload,
    grok_auth_file,
    grok_billing_log_line,
    grok_credits_config,
    write_claude_account_file,
)
from usagedash import rpc
from usagedash.config import ClaudeSourceConfig, SourceMode
from usagedash.errors import CredentialUnavailableError, ProviderRequestError, ProviderSchemaError
from usagedash.models import MAX_WINDOW_SECONDS, Fact, Provider
from usagedash.providers import claude_cli, codex, codex_cli, grok, grok_cli
from usagedash.providers.base import NoQuotaDataError
from usagedash.providers.claude import identified_claude_home, identity_from_account_file, keychain_account
from usagedash.providers.claude_cli import ClaudeCliSource, cli_environment
from usagedash.providers.codex_cli import CodexCliSource
from usagedash.providers.grok import CreditsConfig
from usagedash.providers.grok_cli import GrokCliSource
from usagedash.rpc import JsonLineProcess, child_environment, jsonrpc_result

FAKE_CLI = [sys.executable, str(Path(__file__).parent / "fakes" / "fake_cli.py")]
SECOND_PERSONA_EMAIL = "second.persona@example.test"
CLAUDE_RATE_LIMITS: dict[str, object] = {
    "limits": [
        {"kind": "session", "percent": 6, "resets_at": "2026-10-03T18:20:00+00:00", "scope": None},
        {"kind": "weekly_all", "percent": 54, "resets_at": "2026-10-06T15:00:00+00:00", "scope": None},
    ]
}


def fake_environment(protocol: str, scenario: str, record: Path) -> dict[str, str]:
    return {"FAKE_PROTOCOL": protocol, "FAKE_SCENARIO": scenario, "FAKE_RECORD": str(record), "PATH": "/usr/bin:/bin"}


def recorded(record: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in record.read_text().splitlines()]


def received_messages(record: Path) -> list[dict[str, Any]]:
    return [entry["received"] for entry in recorded(record) if "received" in entry]


class TestALoginThatChangesDuringTheExchange:
    """The identity is read from a file before the CLI runs, and the CLI answers for whoever is logged in then."""

    def claude_source(self, tmp_path: Path, **change: str) -> ClaudeCliSource:
        config_dir = tmp_path / "claude-work"
        config_dir.mkdir()
        account_file = config_dir / ".claude.json"
        write_claude_account_file(account_file, "org-work", "Synthetic Co")
        config = ClaudeSourceConfig(config_dir=config_dir, account_file=account_file, mode=SourceMode.CLI)
        environment = fake_environment("claude", "ok", tmp_path / "record") | {"FAKE_CHANGE_FILE": str(account_file)}
        return ClaudeCliSource(config, FAKE_CLI, environment | change)

    def grok_source(self, tmp_path: Path, now: datetime, **change: str) -> GrokCliSource:
        home = tmp_path / "grok-home"
        home.mkdir()
        grok_auth_file(home / "auth.json", user_id="grok-user-a", create_time=now - timedelta(days=1))
        environment = fake_environment("grok", "ok", tmp_path / "record") | {
            "FAKE_CHANGE_FILE": str(home / "auth.json")
        }
        return GrokCliSource(home, FAKE_CLI, environment | change)

    def claude_login(self, tmp_path: Path, organization_uuid: str, organization_name: str, email: str) -> str:
        write_claude_account_file(tmp_path / "another-login.json", organization_uuid, organization_name, email)
        return (tmp_path / "another-login.json").read_text()

    def grok_login(self, tmp_path: Path, user_id: str, create_time: datetime, email: str = PERSONA_EMAIL) -> str:
        grok_auth_file(tmp_path / "another-login.json", user_id=user_id, email=email, create_time=create_time)
        return (tmp_path / "another-login.json").read_text()

    @pytest.mark.parametrize(
        ("organization_uuid", "organization_name", "email"),
        [
            pytest.param("org-other", "Other Co", PERSONA_EMAIL, id="another organization"),
            pytest.param("org-work", "Synthetic Co", SECOND_PERSONA_EMAIL, id="another member of the organization"),
            pytest.param("org-work", "Renamed Co", PERSONA_EMAIL, id="the organization under another name"),
        ],
    )
    def test_claude_usage_is_refused_when_the_login_differs_in_anything_the_reading_reports(
        self, tmp_path: Path, now: datetime, organization_uuid: str, organization_name: str, email: str
    ) -> None:
        changed = self.claude_login(tmp_path, organization_uuid, organization_name, email)
        source = self.claude_source(tmp_path, FAKE_CHANGE_CONTENT=changed)

        with pytest.raises(NoQuotaDataError, match=r"^the Claude login changed while it was being read"):
            source.read_quota(now)

    def test_a_claude_login_rewritten_with_the_same_account_is_read_as_usual(
        self, tmp_path: Path, now: datetime
    ) -> None:
        unchanged = self.claude_login(tmp_path, "org-work", "Synthetic Co", PERSONA_EMAIL)

        reading = self.claude_source(tmp_path, FAKE_CHANGE_CONTENT=unchanged).read_quota(now)

        assert (reading.identity.external_id, reading.identity.email) == ("org-work", PERSONA_EMAIL)

    def test_a_claude_logout_meanwhile_is_reported_as_nobody_logged_in(self, tmp_path: Path, now: datetime) -> None:
        with pytest.raises(CredentialUnavailableError, match="no Claude login here yet"):
            self.claude_source(tmp_path).read_quota(now)

    def test_grok_billing_is_refused_when_another_account_signed_in_meanwhile(
        self, tmp_path: Path, now: datetime
    ) -> None:
        source = self.grok_source(tmp_path, now, FAKE_CHANGE_CONTENT=self.grok_login(tmp_path, "grok-user-b", now))

        with pytest.raises(NoQuotaDataError, match=r"^the Grok login changed while it was being read"):
            source.read_quota(now)

    def test_a_grok_token_refreshed_by_the_read_itself_is_still_the_same_login(
        self, tmp_path: Path, now: datetime
    ) -> None:
        """Nobody else uses a standing home, so the CLI this source runs is what refreshes its token.

        A refresh writes a new credential with a new creation time. Refusing that would cost a standing
        login one reading every time its token expired.
        """
        refreshed = self.grok_login(tmp_path, "grok-user-a", now)

        reading = self.grok_source(tmp_path, now, FAKE_CHANGE_CONTENT=refreshed).read_quota(now)

        assert (reading.identity.external_id, reading.identity.email) == ("grok-user-a", PERSONA_EMAIL)

    def test_a_grok_login_whose_email_changed_meanwhile_is_a_changed_login_too(
        self, tmp_path: Path, now: datetime
    ) -> None:
        """The reading reports the email, so the email first read must still be the one there."""
        changed = self.grok_login(tmp_path, "grok-user-a", now - timedelta(days=1), email=SECOND_PERSONA_EMAIL)

        with pytest.raises(NoQuotaDataError, match=r"^the Grok login changed while it was being read"):
            self.grok_source(tmp_path, now, FAKE_CHANGE_CONTENT=changed).read_quota(now)

    def test_a_grok_logout_meanwhile_is_reported_as_nobody_signed_in(self, tmp_path: Path, now: datetime) -> None:
        with pytest.raises(CredentialUnavailableError, match="no Grok login here yet"):
            self.grok_source(tmp_path, now).read_quota(now)

    def test_a_grok_login_rewritten_with_the_same_sign_in_is_read_as_usual(self, tmp_path: Path, now: datetime) -> None:
        unchanged = self.grok_login(tmp_path, "grok-user-a", now - timedelta(days=1))

        reading = self.grok_source(tmp_path, now, FAKE_CHANGE_CONTENT=unchanged).read_quota(now)

        assert reading.identity.external_id == "grok-user-a"


class TestTierOfTheLastBillingRecord:
    WRITER = 4242

    def record(self, tier: object = "SuperGrok", pid: object = WRITER) -> bytes:
        return json.dumps(
            {"pid": pid, "msg": "billing: fetched credits config", "ctx": {"subscriptionTier": tier}}
        ).encode()

    def test_nothing_logged_gives_no_tier(self) -> None:
        assert grok_cli.tier_of_last_billing_record(b"", self.WRITER) is None
        assert grok_cli.tier_of_last_billing_record(b'{"msg": "session started"}\n', self.WRITER) is None

    def test_the_last_billing_record_decides(self) -> None:
        log = self.record("First") + b"\n" + self.record("Second") + b"\n"

        assert grok_cli.tier_of_last_billing_record(log, self.WRITER) == "Second"

    @pytest.mark.parametrize(
        "not_a_billing_record",
        [
            b'{"msg": "tool output", "ctx": {"text": "billing: fetched credits config"}}',
            b'"billing: fetched credits config"',
            b'["billing: fetched credits config"]',
        ],
    )
    def test_a_valid_record_that_only_mentions_the_billing_text_is_passed_over(
        self, not_a_billing_record: bytes
    ) -> None:
        log = self.record("Kept") + b"\n" + not_a_billing_record + b"\n"

        assert grok_cli.tier_of_last_billing_record(log, self.WRITER) == "Kept"

    @pytest.mark.parametrize(
        "unusable",
        [
            b'{"msg": "billing: fetched credits config", "ctx": {"subscriptionTier": "Cut',
            b'{"pid": 4242, "msg": "billing: fetched credits config", "ctx": {"subscriptionTier": 7}}',
            b'{"pid": 4242, "msg": "billing: fetched credits config", "ctx": []}',
            b"billing: fetched credits config " + b"[" * 100_000,
        ],
    )
    def test_a_last_record_that_cannot_be_used_is_not_replaced_by_an_earlier_one(self, unusable: bytes) -> None:
        assert grok_cli.tier_of_last_billing_record(self.record("Earlier") + b"\n" + unusable, self.WRITER) is None

    def test_a_usable_record_after_an_unusable_one_is_used(self) -> None:
        log = b'{"pid": 4242, "msg": "billing: fetched credits config", "ctx": []}\n' + self.record("Newest") + b"\n"

        assert grok_cli.tier_of_last_billing_record(log, self.WRITER) == "Newest"

    @pytest.mark.parametrize("other", [1, "4242", True, None], ids=["another process", "text", "a flag", "no pid"])
    def test_a_record_another_process_wrote_is_passed_over(self, other: object) -> None:
        """In a working home an open Grok session logs its own billing, which can be an earlier login's."""
        log = self.record("This Exchange") + b"\n" + self.record("Another Process", pid=other) + b"\n"

        assert grok_cli.tier_of_last_billing_record(log, self.WRITER) == "This Exchange"

    def test_a_flag_is_not_a_process_id_though_python_counts_true_as_one(self) -> None:
        assert grok_cli.tier_of_last_billing_record(self.record("Flagged", pid=True), 1) is None

    def test_a_record_that_states_no_tier_gives_none(self) -> None:
        assert grok_cli.tier_of_last_billing_record(self.record(None), self.WRITER) is None


class TestChildEnvironment:
    def test_keeps_what_is_inherited_except_the_vendor_s_own_variables(self) -> None:
        base = {"PATH": "/usr/bin", "HOME": "/Users/persona", "VENDOR_KEY": "synthetic", "OTHERVENDOR_KEY": "kept"}

        environment = child_environment(base, ("VENDOR_",), {})

        assert environment == {"PATH": "/usr/bin", "HOME": "/Users/persona", "OTHERVENDOR_KEY": "kept"}

    def test_applies_the_settings_after_dropping_so_the_home_given_is_the_one_used(self) -> None:
        base = {"PATH": "/usr/bin", "VENDOR_HOME": "/inherited", "SECOND_KEY": "synthetic"}

        environment = child_environment(base, ("VENDOR_", "SECOND_"), {"VENDOR_HOME": "/standing"})

        assert environment == {"PATH": "/usr/bin", "VENDOR_HOME": "/standing"}

    def test_leaves_the_environment_it_was_given_unchanged(self) -> None:
        base = {"PATH": "/usr/bin", "VENDOR_KEY": "synthetic"}

        child_environment(base, ("VENDOR_",), {"VENDOR_HOME": "/standing"})

        assert base == {"PATH": "/usr/bin", "VENDOR_KEY": "synthetic"}


class TestARejectionCarriesNoneOfThePeerSOwnText:
    """What is raised is stored against the source and shown, and a peer's message can quote anything."""

    def rejection(self, tmp_path: Path, error: object) -> str:
        environment = fake_environment("grok", "billing_error", tmp_path / "record")
        environment["FAKE_BILLING_ERROR"] = json.dumps(error)
        request = {"jsonrpc": "2.0", "id": 2, "method": "_x.ai/billing", "params": {}}
        with (
            JsonLineProcess(FAKE_CLI, env=environment, cwd=tmp_path, service="Fake") as process,
            pytest.raises(ProviderRequestError) as raised,
        ):
            jsonrpc_result(process, request, time.monotonic() + 5, service="Fake")
        return str(raised.value)

    def test_the_message_is_left_out_and_the_code_is_kept(self, tmp_path: Path) -> None:
        text = self.rejection(tmp_path, {"code": -32603, "message": "synthetic-secret quoted by the peer"})

        assert text == "Fake rejected _x.ai/billing (error code -32603)"

    @pytest.mark.parametrize(
        "error",
        [
            "synthetic-secret as the whole error",
            {"message": "synthetic-secret with no code"},
            {"code": "synthetic-secret as the code", "message": "m"},
            {"code": True, "message": "m"},
            {"code": 1.5, "message": "m"},
            {"code": 2**31, "message": "m"},
            {"code": -(2**31) - 1, "message": "m"},
            {"code": 10**400, "message": "m"},
            ["synthetic-secret in a list"],
        ],
    )
    def test_an_error_with_no_usable_code_is_reported_as_having_none(self, tmp_path: Path, error: object) -> None:
        assert self.rejection(tmp_path, error) == "Fake rejected _x.ai/billing (no error code given)"

    @pytest.mark.parametrize("code", [-(2**31), -32000, 0, 2**31 - 1])
    def test_any_code_a_32_bit_integer_can_hold_is_shown(self, tmp_path: Path, code: int) -> None:
        assert self.rejection(tmp_path, {"code": code}) == f"Fake rejected _x.ai/billing (error code {code})"


class TestJsonLineProcess:
    def test_reports_a_command_that_cannot_be_started(self, tmp_path: Path) -> None:
        missing = tmp_path / "no-such-binary"
        with pytest.raises(
            ProviderRequestError,
            match=rf"cannot start Missing CLI: cannot use {re.escape(str(missing))}: FileNotFoundError",
        ):
            JsonLineProcess([str(missing)], env={}, cwd=tmp_path, service="Missing CLI")

    def test_a_child_started_without_its_streams_is_stopped_and_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Popen supplies both streams when asked for pipes. If it ever did not, the child must not be left running."""

        class StreamlessChild:
            stdin = None
            stdout = None
            was_killed = False

            def kill(self) -> None:
                self.was_killed = True

        started: list[StreamlessChild] = []

        def start(*_: object, **__: object) -> StreamlessChild:
            started.append(StreamlessChild())
            return started[-1]

        monkeypatch.setattr(rpc.subprocess, "Popen", start)

        with pytest.raises(ProviderRequestError, match="Fake started without its input and output streams"):
            JsonLineProcess(FAKE_CLI, env={}, cwd=tmp_path, service="Fake")

        assert [child.was_killed for child in started] == [True]

    def test_reports_a_process_that_exits_without_answering(self, tmp_path: Path) -> None:
        environment = fake_environment("grok", "exit_immediately", tmp_path / "record")
        with (
            JsonLineProcess(FAKE_CLI, env=environment, cwd=tmp_path, service="Fake") as process,
            pytest.raises(ProviderRequestError, match="exited before answering"),
        ):
            process.receive(lambda message: True, time.monotonic() + 10)

    def test_gives_up_at_the_deadline_and_stops_the_process(self, tmp_path: Path) -> None:
        environment = fake_environment("grok", "silent", tmp_path / "record")
        started = time.monotonic()
        with (
            JsonLineProcess(FAKE_CLI, env=environment, cwd=tmp_path, service="Fake") as process,
            pytest.raises(ProviderRequestError, match="did not answer in time"),
        ):
            process.receive(lambda message: True, time.monotonic() + 0.5)

        assert time.monotonic() - started < 10

    def test_skips_output_that_is_blank_not_json_or_not_an_object(self, tmp_path: Path) -> None:
        """The fake prints a banner, blank lines, a JSON array, and a JSON string before it answers."""
        environment = fake_environment("grok", "ok", tmp_path / "record")
        with JsonLineProcess(FAKE_CLI, env=environment, cwd=tmp_path, service="Fake") as process:
            result = jsonrpc_result(process, grok_cli.INITIALIZE_REQUEST, time.monotonic() + 10, service="Fake")

        assert result == {"protocolVersion": 1}

    def test_a_request_to_a_process_that_already_exited_is_a_request_error(self, tmp_path: Path) -> None:
        environment = fake_environment("grok", "exit_immediately", tmp_path / "record")
        with JsonLineProcess(FAKE_CLI, env=environment, cwd=tmp_path, service="Fake") as process:
            process._process.wait(timeout=10)
            with pytest.raises(ProviderRequestError, match="Fake exited before accepting a request"):
                process.send({"id": 1, "method": "initialize"})

    def test_a_process_that_ignores_the_request_to_stop_is_killed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rpc, "TERMINATE_GRACE_SECONDS", 0.2)
        environment = fake_environment("grok", "ignores_termination", tmp_path / "record")

        with JsonLineProcess(FAKE_CLI, env=environment, cwd=tmp_path, service="Fake") as process:
            assert process.receive(lambda message: message.get("ready") is True, time.monotonic() + 10)
            child = process._process

        assert child.returncode == -signal.SIGKILL

    def test_a_line_that_never_ends_is_refused_once_it_passes_the_limit_and_the_process_is_stopped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Held without a bound, it would take all the memory there is before the deadline came."""
        monkeypatch.setattr(rpc, "MAX_LINE_BYTES", 1024 * 1024)
        environment = fake_environment("grok", "endless_line", tmp_path / "record")

        with JsonLineProcess(FAKE_CLI, env=environment, cwd=tmp_path, service="Fake") as process:
            with pytest.raises(ProviderRequestError, match=r"^Fake sent a line longer than 1048576 bytes$"):
                process.receive(lambda message: True, time.monotonic() + 30)
            held = len(process._buffer)
            child = process._process

        assert 1024 * 1024 < held <= 1024 * 1024 + rpc.READ_CHUNK_BYTES
        assert child.returncode is not None

    @pytest.mark.parametrize(
        ("length", "refused"),
        [pytest.param(1024, False, id="exactly the limit"), pytest.param(1025, True, id="one byte over the limit")],
    )
    def test_a_line_that_ends_is_held_to_the_limit_as_well(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, length: int, refused: bool
    ) -> None:
        """Ended by the read that took it past the limit, a line was parsed before anything was measured."""
        monkeypatch.setattr(rpc, "MAX_LINE_BYTES", 1024)
        noise = b'"' + b"x" * (length - 2) + b'"'
        environment = fake_environment("grok", "silent", tmp_path / "record")
        with JsonLineProcess(FAKE_CLI, env=environment, cwd=tmp_path, service="Fake") as process:
            process._buffer.extend(noise + b'\n{"id": 1}\n')

            if refused:
                with pytest.raises(ProviderRequestError, match=r"^Fake sent a line longer than 1024 bytes$"):
                    process.receive(lambda message: True, time.monotonic() + 1)
            else:
                assert process.receive(lambda message: True, time.monotonic() + 1) == {"id": 1}

    def test_an_answer_longer_than_the_limit_is_refused_and_not_taken(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rpc, "MAX_LINE_BYTES", 1024)
        answer = json.dumps({"id": 1, "result": {"padding": "x" * 1100}}).encode()
        environment = fake_environment("grok", "silent", tmp_path / "record")
        with JsonLineProcess(FAKE_CLI, env=environment, cwd=tmp_path, service="Fake") as process:
            process._buffer.extend(answer + b"\n")

            with pytest.raises(ProviderRequestError, match="sent a line longer than 1024 bytes"):
                process.receive(lambda message: message.get("id") == 1, time.monotonic() + 1)

    def test_lines_are_taken_from_the_buffer_one_at_a_time_and_what_follows_the_last_newline_is_kept(
        self, tmp_path: Path
    ) -> None:
        environment = fake_environment("grok", "silent", tmp_path / "record")
        with JsonLineProcess(FAKE_CLI, env=environment, cwd=tmp_path, service="Fake") as process:
            process._buffer.extend(b'{"id": 1}\nnot json\n{"id": 2}\n{"id": 3, "part')

            first = process.receive(lambda message: True, time.monotonic() + 1)
            second = process.receive(lambda message: True, time.monotonic() + 1)

            assert (first, second) == ({"id": 1}, {"id": 2})
            assert bytes(process._buffer) == b'{"id": 3, "part'

    def test_a_line_nested_too_deeply_to_parse_is_skipped_like_any_other_noise(self) -> None:
        assert rpc._parse_object(b"[" * 200_000) is None
        assert rpc._parse_object(b'{"id": 1}') == {"id": 1}

    def test_a_process_that_stops_when_asked_is_not_killed(self, tmp_path: Path) -> None:
        environment = fake_environment("grok", "silent", tmp_path / "record")

        with JsonLineProcess(FAKE_CLI, env=environment, cwd=tmp_path, service="Fake") as process:
            child = process._process

        assert child.returncode == -signal.SIGTERM

    @pytest.mark.parametrize(
        ("scenario", "message"),
        [
            ("error_without_message", r"Fake rejected _x\.ai/billing \(no error code given\)$"),
            ("null_result", r"Fake returned no result for _x\.ai/billing$"),
            ("billing_error", r"Fake rejected _x\.ai/billing \(error code -32603\)$"),
        ],
    )
    def test_an_answer_that_is_an_error_or_carries_no_result_object_is_a_request_error(
        self, tmp_path: Path, scenario: str, message: str
    ) -> None:
        environment = fake_environment("grok", scenario, tmp_path / "record")
        with (
            JsonLineProcess(FAKE_CLI, env=environment, cwd=tmp_path, service="Fake") as process,
            pytest.raises(ProviderRequestError, match=message),
        ):
            jsonrpc_result(process, grok_cli.BILLING_REQUEST, time.monotonic() + 10, service="Fake")


class TestAnEquivalentClaudeAccountFile:
    def config(self, config_dir: Path, account_file: Path) -> ClaudeSourceConfig:
        return ClaudeSourceConfig(config_dir=config_dir, account_file=account_file, mode=SourceMode.CLI)

    def test_equivalent_spellings_stay_on_the_captured_home_after_the_link_moves(self, tmp_path: Path) -> None:
        original = tmp_path / "Original"
        original.mkdir()
        write_claude_account_file(original / ".claude.json", "org-original", "Original Co")
        link = tmp_path / "link"
        link.symlink_to(original, target_is_directory=True)
        kept = [
            identified_claude_home(self.config(config_dir, account_file))[1]
            for config_dir, account_file in (
                (link, link / ".." / "link" / ".claude.json"),
                (link / ".." / "link", link / ".claude.json"),
                (link, original / ".claude.json"),
                (original, link / ".claude.json"),
            )
        ]
        other = tmp_path / "Other"
        other.mkdir()
        write_claude_account_file(other / ".claude.json", "org-other", "Other Co")
        link.unlink()
        link.symlink_to(other, target_is_directory=True)

        identities = [identity_from_account_file(path, None) for path in kept]
        assert kept == [original / ".claude.json"] * 4
        assert [identity is not None and identity.external_id == "org-original" for identity in identities] == [
            True
        ] * 4

    def test_an_account_file_outside_the_config_directory_stays_there(self, tmp_path: Path) -> None:
        original = tmp_path / "Original"
        original.mkdir()
        link = tmp_path / "link"
        link.symlink_to(original, target_is_directory=True)
        external = tmp_path / "elsewhere" / ".claude.json"
        external.parent.mkdir()
        write_claude_account_file(external, "org-external", "External Co")
        escaped = link / ".." / "elsewhere" / ".claude.json"

        assert identified_claude_home(self.config(link, external))[1] == external
        assert identified_claude_home(self.config(link, escaped))[1] == escaped
        identity = identity_from_account_file(external, None)
        assert identity is not None and identity.external_id == "org-external"

    def test_a_case_variant_is_rebased_only_when_it_is_the_same_directory(self, tmp_path: Path) -> None:
        original = tmp_path / "Original"
        original.mkdir()
        folded = tmp_path / "original"
        account = folded / ".claude.json"
        if folded.exists() and os.path.samefile(folded, original):
            _home, kept = identified_claude_home(self.config(original, account))
            assert kept == original / ".claude.json"
            return
        folded.mkdir()
        _home, kept = identified_claude_home(self.config(original, account))
        assert kept == account

    def test_a_detour_that_resolves_outside_the_home_stays_there(self, tmp_path: Path) -> None:
        roots = tmp_path / "roots"
        original = tmp_path / "data" / "Original"
        elsewhere = tmp_path / "data" / "link"
        roots.mkdir()
        original.mkdir(parents=True)
        elsewhere.mkdir()
        write_claude_account_file(original / ".claude.json", "org-original", "Original Co")
        write_claude_account_file(elsewhere / ".claude.json", "org-link", "Link Co")
        link = roots / "link"
        link.symlink_to(original, target_is_directory=True)
        account = link / ".." / "link" / ".claude.json"

        kept = identified_claude_home(self.config(link, account))[1]
        identity = identity_from_account_file(kept, None)

        assert kept == account
        assert identity is not None and identity.external_id == "org-link"


class TestClaudeCliSource:
    @pytest.fixture
    def config(self, tmp_path: Path) -> ClaudeSourceConfig:
        config_dir = tmp_path / "claude-work"
        config_dir.mkdir()
        write_claude_account_file(config_dir / ".claude.json", "org-work", "Synthetic Co")
        return ClaudeSourceConfig(config_dir=config_dir, account_file=config_dir / ".claude.json", mode=SourceMode.CLI)

    def source(self, config: ClaudeSourceConfig, scenario: str, record: Path, **extra_env: str) -> ClaudeCliSource:
        return ClaudeCliSource(config, FAKE_CLI, {**fake_environment("claude", scenario, record), **extra_env})

    def test_reads_plan_limits_through_control_requests_without_sending_a_prompt(
        self, config: ClaudeSourceConfig, tmp_path: Path, now: datetime
    ) -> None:
        record = tmp_path / "record"

        reading = self.source(config, "ok", record).read_quota(now)

        assert (reading.identity.external_id, reading.identity.default_label, reading.identity.plan) == (
            "org-work",
            "Synthetic Co",
            "Max",
        )
        assert [(w.key, w.used_percent) for w in reading.windows] == [("five_hour", 6), ("weekly", 54)]
        messages = received_messages(record)
        assert [message["request"]["subtype"] for message in messages] == ["initialize", "get_usage"]
        assert all(message["type"] == "control_request" for message in messages)

    def test_points_the_cli_at_the_standing_home_and_hides_the_enclosing_session(
        self, config: ClaudeSourceConfig, tmp_path: Path, now: datetime
    ) -> None:
        record = tmp_path / "record"
        source = self.source(
            config, "ok", record, CLAUDECODE="1", CLAUDE_CODE_ENTRYPOINT="cli", ANTHROPIC_API_KEY="synthetic-key"
        )

        source.read_quota(now)

        launch = recorded(record)[0]
        assert launch["env"] == {"CLAUDE_CONFIG_DIR": str(config.config_dir)}
        assert "--no-session-persistence" in launch["argv"]
        assert "--strict-mcp-config" in launch["argv"]

    def test_the_default_home_is_never_named_explicitly(self, tmp_path: Path) -> None:
        environment = cli_environment({"PATH": "/usr/bin", "CLAUDE_CONFIG_DIR": "/inherited"}, None)
        assert environment == {"PATH": "/usr/bin", "USER": keychain_account()}

    def test_a_different_spelling_keeps_the_configured_keychain_name(self) -> None:
        environment = cli_environment({}, Path("/p/Original"), Path("/p/link/../link"))

        assert environment == {
            "USER": keychain_account(),
            "CLAUDE_CONFIG_DIR": "/p/Original",
            "CLAUDE_SECURESTORAGE_CONFIG_DIR": "/p/link/../link",
        }

    def test_one_spelling_does_not_add_a_second_keychain_name(self) -> None:
        environment = cli_environment({}, Path("/p/Original"), Path("/p/Original"))
        pinned = {"USER": keychain_account(), "CLAUDE_CONFIG_DIR": "/p/Original"}

        assert environment == pinned
        assert cli_environment({}, Path("/p/Original")) == pinned

    @pytest.mark.parametrize("home", [None, Path("/p/Original")], ids=["default home", "standing home"])
    def test_an_inherited_user_never_reaches_the_cli_because_it_names_the_keychain_account(
        self, home: Path | None
    ) -> None:
        """Regression lock: a sign-in run with ``USER=unknown`` filed the login where no read looked (2026-10-08)."""
        environment = cli_environment({"USER": "unknown", "LOGNAME": "unknown"}, home)

        assert environment["USER"] == keychain_account()
        assert environment["USER"] != "unknown"
        assert environment["LOGNAME"] == "unknown"

    def test_the_cli_process_runs_with_the_pinned_account_name(
        self, config: ClaudeSourceConfig, tmp_path: Path, now: datetime
    ) -> None:
        record = tmp_path / "record"

        self.source(config, "ok", record, USER="unknown").read_quota(now)

        assert recorded(record)[0]["user"] == keychain_account()

    def test_a_home_nobody_has_logged_in_to_starts_no_process(self, tmp_path: Path, now: datetime) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        record = tmp_path / "record"
        config = ClaudeSourceConfig(config_dir=empty, account_file=empty / ".claude.json", mode=SourceMode.CLI)
        source = self.source(config, "ok", record)

        assert source.current_identity() is None
        with pytest.raises(CredentialUnavailableError, match="usagedash login claude"):
            source.read_quota(now)
        assert not record.exists()

    def test_a_login_without_plan_limits_is_reported_as_no_data(
        self, config: ClaudeSourceConfig, tmp_path: Path, now: datetime
    ) -> None:
        with pytest.raises(NoQuotaDataError, match="no plan limits"):
            self.source(config, "no_limits", tmp_path / "record").read_quota(now)

    def answering(self, config: ClaudeSourceConfig, tmp_path: Path, answer: dict[str, object]) -> ClaudeCliSource:
        return self.source(config, "usage_answer_from_env", tmp_path / "record", FAKE_USAGE_ANSWER=json.dumps(answer))

    @pytest.mark.parametrize(
        ("answer", "problem"),
        [
            ({"rate_limits": CLAUDE_RATE_LIMITS}, "rate_limits_available: Field required"),
            ({"rate_limits_available": None, "rate_limits": CLAUDE_RATE_LIMITS}, "rate_limits_available: "),
            ({"rate_limits_available": "false", "rate_limits": CLAUDE_RATE_LIMITS}, "rate_limits_available: "),
            ({"rate_limits_available": "true", "rate_limits": CLAUDE_RATE_LIMITS}, "rate_limits_available: "),
            ({"rate_limits_available": 1, "rate_limits": CLAUDE_RATE_LIMITS}, "rate_limits_available: "),
            (
                {"rate_limits_available": True, "rate_limits": CLAUDE_RATE_LIMITS, "subscription_type": 7},
                "subscription_type: ",
            ),
        ],
    )
    def test_an_answer_whose_fields_have_the_wrong_type_is_a_schema_error_not_a_reading(
        self, config: ClaudeSourceConfig, tmp_path: Path, now: datetime, answer: dict[str, object], problem: str
    ) -> None:
        with pytest.raises(ProviderSchemaError, match=f"get_usage answer has an unexpected shape: {problem}"):
            self.answering(config, tmp_path, answer).read_quota(now)

    @pytest.mark.parametrize(
        "answer", [{"rate_limits_available": True, "rate_limits": None}, {"rate_limits_available": True}]
    )
    def test_limits_said_to_be_available_but_not_sent_this_time_are_a_failed_read_not_an_absence(
        self, config: ClaudeSourceConfig, tmp_path: Path, now: datetime, answer: dict[str, object]
    ) -> None:
        """Seen live once in about 16 answers, the next answering in full: neither "no limits" nor a new shape."""
        with pytest.raises(ProviderRequestError, match="said plan limits are available but sent none this time"):
            self.answering(config, tmp_path, answer).read_quota(now)

    @pytest.mark.parametrize("limits", ["n/a", [], 0, False])
    def test_limits_sent_as_something_other_than_an_object_are_a_schema_error(
        self, config: ClaudeSourceConfig, tmp_path: Path, now: datetime, limits: object
    ) -> None:
        answer = {"rate_limits_available": True, "rate_limits": limits}

        with pytest.raises(ProviderSchemaError, match="sent plan limits that are not an object"):
            self.answering(config, tmp_path, answer).read_quota(now)

    def test_limits_attached_to_an_answer_that_says_none_are_available_are_not_published(
        self, config: ClaudeSourceConfig, tmp_path: Path, now: datetime
    ) -> None:
        answer = {"rate_limits_available": False, "rate_limits": CLAUDE_RATE_LIMITS}

        with pytest.raises(NoQuotaDataError, match="no plan limits"):
            self.answering(config, tmp_path, answer).read_quota(now)

    def test_a_plan_is_optional_in_an_answer_that_has_limits(
        self, config: ClaudeSourceConfig, tmp_path: Path, now: datetime
    ) -> None:
        answer = {"rate_limits_available": True, "rate_limits": CLAUDE_RATE_LIMITS}

        reading = self.answering(config, tmp_path, answer).read_quota(now)

        assert reading.identity.plan is None
        assert [window.key for window in reading.windows] == ["five_hour", "weekly"]

    def test_a_rejected_request_is_reported_without_the_cli_s_own_words(
        self, config: ClaudeSourceConfig, tmp_path: Path, now: datetime
    ) -> None:
        """The message is stored and shown, and the reason is Claude Code's text, not this package's."""
        with pytest.raises(ProviderRequestError, match=r"^Claude Code rejected get_usage$"):
            self.source(config, "usage_rejected", tmp_path / "record").read_quota(now)

    def test_a_successful_answer_without_a_body_is_a_schema_error(
        self, config: ClaudeSourceConfig, tmp_path: Path, now: datetime
    ) -> None:
        with pytest.raises(ProviderSchemaError, match="answered get_usage without a body"):
            self.source(config, "usage_without_body", tmp_path / "record").read_quota(now)

    def test_the_source_is_named_by_its_provider_and_config_directory(
        self, config: ClaudeSourceConfig, tmp_path: Path
    ) -> None:
        source = self.source(config, "ok", tmp_path / "record")

        assert (source.provider, source.source_id) == (Provider.CLAUDE, f"claude:{config.config_dir}")

    def test_a_retargeted_home_link_does_not_move_the_cli_or_the_account_it_reads(
        self, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = tmp_path / "Original"
        original.mkdir()
        write_claude_account_file(original / ".claude.json", "org-original", "Original Co")
        link = tmp_path / "link"
        link.symlink_to(original, target_is_directory=True)
        config = ClaudeSourceConfig(config_dir=link, account_file=link / ".claude.json", mode=SourceMode.CLI)
        source = self.source(config, "ok", tmp_path / "record")
        source_id = source.source_id
        other = tmp_path / "Other"
        other.mkdir()
        write_claude_account_file(other / ".claude.json", "org-other", "Other Co")
        link.unlink()
        link.symlink_to(other, target_is_directory=True)
        directories: list[Path] = []
        fallback_names: list[str | None] = []
        real_process = claude_cli.JsonLineProcess

        class RecordingProcess(real_process):
            def __init__(
                self,
                command: list[str],
                *,
                env: dict[str, str],
                cwd: Path,
                service: str,
                helper_directory: Path | None = None,
                fallback_name: str | None = None,
            ) -> None:
                directories.append(Path(cwd))
                fallback_names.append(fallback_name)
                super().__init__(
                    command,
                    env=env,
                    cwd=cwd,
                    service=service,
                    helper_directory=helper_directory,
                    fallback_name=fallback_name,
                )

        monkeypatch.setattr(claude_cli, "JsonLineProcess", RecordingProcess)

        reading = source.read_quota(now)

        assert source.source_id == source_id
        assert "Other" not in source.source_id
        assert reading.identity.external_id == "org-original"
        assert recorded(tmp_path / "record")[0]["env"] == {
            "CLAUDE_CONFIG_DIR": str(original),
            "CLAUDE_SECURESTORAGE_CONFIG_DIR": str(link),
        }
        assert directories == [original]
        assert fallback_names == ["claude"]

    def test_a_cli_that_never_answers_times_out(
        self, config: ClaudeSourceConfig, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(claude_cli, "CLI_TIMEOUT_SECONDS", 0.5)
        with pytest.raises(ProviderRequestError, match="did not answer in time"):
            self.source(config, "silent", tmp_path / "record").read_quota(now)


class TestCodexCliSource:
    def source(self, tmp_path: Path, scenario: str, record: Path, **extra_env: str) -> CodexCliSource:
        return CodexCliSource(tmp_path, FAKE_CLI, {**fake_environment("codex", scenario, record), **extra_env})

    def test_runs_the_cli_as_the_login_in_its_home_whatever_the_enclosing_shell_sets(
        self, tmp_path: Path, now: datetime
    ) -> None:
        """Codex takes a token or an API key from these ahead of the login stored in its home."""
        record = tmp_path / "record"
        source = self.source(
            tmp_path,
            "ok",
            record,
            CODEX_HOME="/another/home",
            CODEX_SQLITE_HOME="/another/home",
            CODEX_ACCESS_TOKEN="synthetic-token",
            CODEX_API_KEY="synthetic-key",
            OPENAI_API_KEY="synthetic-key",
        )

        source.read_quota(now)

        assert recorded(record)[0]["env"] == {"CODEX_HOME": str(tmp_path)}

    def test_reads_rate_limits_and_identity_from_the_app_server(self, tmp_path: Path, now: datetime) -> None:
        record = tmp_path / "record"
        source = self.source(tmp_path, "ok", record)

        assert source.current_identity() is None
        reading = source.read_quota(now)

        assert (reading.identity.external_id, reading.identity.email, reading.identity.plan) == (
            "workspace-b",
            "user-b@example.test",
            "Team",
        )
        assert [(w.key, w.used_percent, w.window_seconds) for w in reading.windows] == [
            ("weekly", 40, WEEK),
            ("five_hour", 15, 300 * 60),
        ]
        assert [(fact.label, fact.value) for fact in reading.facts] == [("Credits", "500 remaining")]
        assert source.current_identity() == reading.identity
        assert recorded(record)[0]["env"] == {"CODEX_HOME": str(tmp_path)}
        assert [message["method"] for message in received_messages(record)] == [
            "initialize",
            "initialized",
            "account/read",
            "account/rateLimits/read",
        ]

    def test_does_not_ask_for_a_token_refresh_when_the_read_succeeds(self, tmp_path: Path, now: datetime) -> None:
        record = tmp_path / "record"
        self.source(tmp_path, "ok", record).read_quota(now)

        account_reads = [m for m in received_messages(record) if m.get("method") == "account/read"]
        assert [m["params"]["refreshToken"] for m in account_reads] == [False]

    def test_asks_the_cli_to_refresh_once_and_retries_when_the_read_fails(self, tmp_path: Path, now: datetime) -> None:
        record = tmp_path / "record"

        reading = self.source(tmp_path, "stale_token", record).read_quota(now)

        assert reading.windows[0].used_percent == 40
        account_reads = [m for m in received_messages(record) if m.get("method") == "account/read"]
        assert [m["params"]["refreshToken"] for m in account_reads] == [False, True]

    def test_a_read_that_still_fails_after_the_refresh_is_an_error(self, tmp_path: Path, now: datetime) -> None:
        record = tmp_path / "record"
        with pytest.raises(ProviderRequestError, match=r"\(error code -32000\)$"):
            self.source(tmp_path, "always_failing", record).read_quota(now)

        limit_reads = [m for m in received_messages(record) if m.get("method") == "account/rateLimits/read"]
        assert len(limit_reads) == 2

    def test_a_home_nobody_has_logged_in_to_is_reported_as_no_credential(self, tmp_path: Path, now: datetime) -> None:
        with pytest.raises(CredentialUnavailableError, match="usagedash login codex"):
            self.source(tmp_path, "logged_out", tmp_path / "record").read_quota(now)

    def test_an_account_answer_with_the_wrong_shape_is_a_schema_error(self, tmp_path: Path, now: datetime) -> None:
        source = self.source(tmp_path, "reshaped_account", tmp_path / "record")

        with pytest.raises(ProviderSchemaError, match="account has an unexpected shape"):
            source.read_quota(now)

        assert source.current_identity() is None

    def test_the_source_is_named_by_its_provider_and_home(self, tmp_path: Path) -> None:
        source = self.source(tmp_path, "ok", tmp_path / "record")

        assert (source.provider, source.source_id) == (Provider.CODEX, f"codex:{tmp_path}")

    @pytest.mark.parametrize(
        ("credits", "expected"),
        [
            ({"unlimited": True, "balance": "5"}, [("Credits", "Unlimited")]),
            ({"unlimited": False, "balance": "1234.6"}, [("Credits", "1,235 remaining")]),
            ({"unlimited": False, "balance": None}, []),
            (None, []),
        ],
    )
    def test_credits_are_reported_only_when_the_cli_states_them(
        self, credits: dict[str, Any] | None, expected: list[tuple[str, str]]
    ) -> None:
        window = {"usedPercent": 5, "windowDurationMins": 10080}
        payload = {"rateLimits": {"limitId": "codex", "primary": window, "credits": credits}}

        _, _, facts = codex_cli.parse_rate_limits(payload)

        assert [(fact.label, fact.value) for fact in facts] == expected

    def test_a_login_that_has_since_been_logged_out_is_no_longer_reported_as_logged_in(
        self, tmp_path: Path, now: datetime
    ) -> None:
        record = tmp_path / "record"
        source = self.source(tmp_path, "ok", record)
        source.read_quota(now)
        assert source.current_identity() is not None

        source._base_environment = fake_environment("codex", "logged_out", record)
        with pytest.raises(CredentialUnavailableError):
            source.read_quota(now)

        assert source.current_identity() is None

    def test_a_failed_read_of_a_logged_in_home_keeps_its_identity(self, tmp_path: Path, now: datetime) -> None:
        record = tmp_path / "record"
        source = self.source(tmp_path, "ok", record)
        identity = source.read_quota(now).identity

        source._base_environment = fake_environment("codex", "always_failing", record)
        with pytest.raises(ProviderRequestError):
            source.read_quota(now)

        assert source.current_identity() == identity

    def test_a_different_login_is_reported_even_when_its_rate_limit_read_fails(
        self, tmp_path: Path, now: datetime
    ) -> None:
        """The account read names the new login before the rate-limit read fails; the earlier one must not linger."""
        record = tmp_path / "record"
        source = self.source(tmp_path, "ok", record)
        assert source.read_quota(now).identity.external_id == "workspace-b"

        source._base_environment = fake_environment("codex", "always_failing", record) | {
            "FAKE_WORKSPACE": "workspace-c"
        }
        with pytest.raises(ProviderRequestError):
            source.read_quota(now)

        identity = source.current_identity()
        assert identity is not None
        assert (identity.external_id, identity.email, identity.plan) == ("workspace-c", "user-b@example.test", "Team")

    def test_a_login_whose_workspace_is_not_named_is_unknown_not_assumed_to_be_the_earlier_one(
        self, tmp_path: Path, now: datetime
    ) -> None:
        record = tmp_path / "record"
        source = self.source(tmp_path, "ok", record)
        source.read_quota(now)

        source._base_environment = fake_environment("codex", "always_failing", record) | {"FAKE_WORKSPACE_ABSENT": "1"}
        with pytest.raises(ProviderRequestError):
            source.read_quota(now)

        assert source.current_identity() is None

    def test_a_reading_that_never_names_the_workspace_is_a_schema_error_and_leaves_no_identity(
        self, tmp_path: Path, now: datetime
    ) -> None:
        record = tmp_path / "record"
        source = CodexCliSource(
            tmp_path, FAKE_CLI, fake_environment("codex", "ok", record) | {"FAKE_WORKSPACE_ABSENT": "1"}
        )

        with pytest.raises(ProviderSchemaError, match="did not identify the ChatGPT workspace"):
            source.read_quota(now)

        assert source.current_identity() is None

    def test_a_limit_the_cli_reports_as_reached_puts_the_account_out_of_quota(
        self, tmp_path: Path, now: datetime
    ) -> None:
        reading = self.source(tmp_path, "limit_reached", tmp_path / "record").read_quota(now)

        assert [(w.key, w.used_percent, w.blocking, w.resets_at) for w in reading.windows][-1] == (
            "reported_limit",
            100,
            True,
            None,
        )
        assert [w.key for w in reading.windows] == ["weekly", "five_hour", "reported_limit"]
        assert all(fact.label != "Status" for fact in reading.facts)

    @pytest.mark.parametrize("percent", [-1, float("inf"), float("nan"), "Infinity"])
    def test_an_impossible_percentage_is_a_schema_error(self, percent: object) -> None:
        payload = {"rateLimits": {"primary": {"usedPercent": percent, "windowDurationMins": 300}}}
        with pytest.raises(ProviderSchemaError, match=r"rateLimits\.primary\.usedPercent"):
            codex_cli.parse_rate_limits(payload)

    @pytest.mark.parametrize("resets_at", [-1, 253_402_300_800, float("inf")])
    def test_a_reset_time_no_datetime_can_hold_is_a_schema_error(self, resets_at: float) -> None:
        payload = {"rateLimits": {"primary": {"usedPercent": 5, "windowDurationMins": 300, "resetsAt": resets_at}}}
        with pytest.raises(ProviderSchemaError, match=r"rateLimits\.primary\.resetsAt"):
            codex_cli.parse_rate_limits(payload)

    @pytest.mark.parametrize(
        ("scenario", "error"),
        [
            ("reshaped_account", ProviderSchemaError),
            ("account_without_result", ProviderRequestError),
            ("account_error", ProviderRequestError),
            ("reshaped_account_on_refresh", ProviderSchemaError),
        ],
    )
    def test_an_account_read_that_does_not_answer_usably_leaves_no_remembered_login(
        self, tmp_path: Path, now: datetime, scenario: str, error: type[Exception]
    ) -> None:
        """Once the account read is attempted, an identity left over from an earlier read cannot be trusted."""
        record = tmp_path / "record"
        source = self.source(tmp_path, "ok", record)
        source.read_quota(now)
        assert source.current_identity() is not None

        source._base_environment = fake_environment("codex", scenario, record)
        with pytest.raises(error):
            source.read_quota(now)

        assert source.current_identity() is None

    def test_a_cli_that_cannot_be_started_keeps_the_remembered_login(self, tmp_path: Path, now: datetime) -> None:
        """Nothing was asked, so nothing was learned: the login is as it was at the last read."""
        source = self.source(tmp_path, "ok", tmp_path / "record")
        identity = source.read_quota(now).identity

        source._command = (str(tmp_path / "no-such-binary"),)
        with pytest.raises(ProviderRequestError, match="cannot start Codex app-server"):
            source.read_quota(now)

        assert source.current_identity() == identity

    def test_a_cli_that_never_answers_the_handshake_keeps_the_remembered_login(
        self, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The read timed out before anyone was asked who is logged in."""
        monkeypatch.setattr(codex_cli, "CLI_TIMEOUT_SECONDS", 0.5)
        record = tmp_path / "record"
        source = self.source(tmp_path, "ok", record)
        identity = source.read_quota(now).identity

        source._base_environment = fake_environment("codex", "silent", record)
        with pytest.raises(ProviderRequestError, match="did not answer in time"):
            source.read_quota(now)

        assert source.current_identity() == identity

    @pytest.mark.parametrize("blank", ["", " ", "\t"])
    def test_a_blank_workspace_id_is_not_an_identity(self, tmp_path: Path, now: datetime, blank: str) -> None:
        """Every login without a usable id would otherwise become one account and mix their usage."""
        record = tmp_path / "record"
        source = CodexCliSource(tmp_path, FAKE_CLI, fake_environment("codex", "ok", record) | {"FAKE_WORKSPACE": blank})

        with pytest.raises(ProviderSchemaError, match="did not identify the ChatGPT workspace"):
            source.read_quota(now)

        assert source.current_identity() is None

    def test_a_workspace_id_has_its_surrounding_whitespace_removed(self, tmp_path: Path, now: datetime) -> None:
        record = tmp_path / "record"
        environment = fake_environment("codex", "ok", record) | {"FAKE_WORKSPACE": "  workspace-c \n"}

        reading = CodexCliSource(tmp_path, FAKE_CLI, environment).read_quota(now)

        assert reading.identity.external_id == "workspace-c"

    @pytest.mark.parametrize("blank", ["", "  ", None])
    def test_a_blank_account_id_in_the_rate_limits_is_treated_as_absent(self, blank: str | None) -> None:
        window = {"usedPercent": 5, "windowDurationMins": 10080}

        limits, _, _ = codex_cli.parse_rate_limits({"accountId": blank, "rateLimits": {"primary": window}})

        assert limits.account_id is None

    @pytest.mark.parametrize("minutes", [-1, 5_256_001, 10**400])
    def test_a_window_length_outside_zero_to_ten_years_is_a_schema_error(self, minutes: int) -> None:
        payload = {"rateLimits": {"primary": {"usedPercent": 5, "windowDurationMins": minutes}}}

        with pytest.raises(ProviderSchemaError, match=r"rateLimits\.primary\.windowDurationMins"):
            codex_cli.parse_rate_limits(payload)

    @pytest.mark.parametrize(("minutes", "seconds"), [(0, 0), (300, 18_000), (5_256_000, 315_360_000)])
    def test_a_window_length_within_that_range_is_converted_to_seconds(self, minutes: int, seconds: int) -> None:
        payload = {"rateLimits": {"primary": {"usedPercent": 5, "windowDurationMins": minutes}}}

        _, windows, _ = codex_cli.parse_rate_limits(payload)

        assert windows[0].window_seconds == seconds

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("usedPercent", True),
            ("usedPercent", "40"),
            ("windowDurationMins", True),
            ("windowDurationMins", "10080"),
            ("windowDurationMins", 10080.0),
            ("resetsAt", True),
            ("resetsAt", "1791580444"),
        ],
    )
    def test_a_window_field_of_the_wrong_type_is_a_schema_error_and_never_converted(
        self, field: str, value: object
    ) -> None:
        window = {"usedPercent": 40, "windowDurationMins": 10080, "resetsAt": 1791580444} | {field: value}

        with pytest.raises(ProviderSchemaError, match=rf"rateLimits\.primary\.{field}"):
            codex_cli.parse_rate_limits({"rateLimits": {"primary": window}})

    @pytest.mark.parametrize(
        ("credits", "field"),
        [({"unlimited": "false"}, "unlimited"), ({"unlimited": 0}, "unlimited"), ({"balance": True}, "balance")],
    )
    def test_a_credits_field_of_the_wrong_type_is_a_schema_error(self, credits: dict[str, object], field: str) -> None:
        payload = {"rateLimits": {"primary": {"usedPercent": 40}, "credits": credits}}

        with pytest.raises(ProviderSchemaError, match=rf"rateLimits\.credits\.{field}"):
            codex_cli.parse_rate_limits(payload)

    def test_a_credit_balance_sent_as_decimal_text_is_read_as_the_number(self) -> None:
        payload = {"rateLimits": {"primary": {"usedPercent": 40}, "credits": {"unlimited": False, "balance": "500.5"}}}

        limits, _, _ = codex_cli.parse_rate_limits(payload)

        assert limits.rate_limits.credits is not None
        assert limits.rate_limits.credits.balance == 500.5

    def test_other_metered_limits_are_prefixed_and_the_main_one_is_not_repeated(self) -> None:
        window = {"usedPercent": 5, "windowDurationMins": 10080, "resetsAt": None}
        payload = {
            "accountId": "workspace-b",
            "rateLimits": {"limitId": "codex", "primary": window},
            "rateLimitsByLimitId": {
                "codex": {"limitId": "codex", "primary": window},
                "code_review": {"limitId": "code_review", "limitName": "Code review", "primary": window},
            },
        }

        _, windows, _ = codex_cli.parse_rate_limits(payload)

        assert [(w.key, w.label, w.blocking) for w in windows] == [
            ("weekly", "Weekly", True),
            ("code_review:weekly", "Code review · Weekly", False),
        ]

    @pytest.mark.parametrize("limit_id", [None, ""])
    @pytest.mark.parametrize("meters", [("codex", "spark"), ("spark", "codex")])
    def test_distinct_meters_with_the_same_snapshot_are_both_kept(
        self, limit_id: str | None, meters: tuple[str, str]
    ) -> None:
        """Equal windows are not a meter id. Dropping both would hide a real pool and record nothing."""
        window = {"usedPercent": 0, "windowDurationMins": 10080, "resetsAt": None}
        main: dict[str, object] = {"primary": window}
        if limit_id is not None:
            main["limitId"] = limit_id
        payload = {"rateLimits": main, "rateLimitsByLimitId": {meter: {"primary": window} for meter in meters}}

        _, windows, facts = codex_cli.parse_rate_limits(payload)

        assert [(item.key, item.label, item.blocking) for item in windows] == [
            ("weekly", "Weekly", True),
            *[(f"{meter}:weekly", f"{meter[:1].upper()}{meter[1:]} · Weekly", False) for meter in meters],
        ]
        assert facts == ()

    def test_a_meter_with_the_main_pools_snapshot_stays_when_its_id_is_different(self) -> None:
        window = {"usedPercent": 0, "windowDurationMins": 10080}
        payload = {
            "rateLimits": {"limitId": "codex", "primary": window},
            "rateLimitsByLimitId": {"codex": {"primary": window}, "spark": {"primary": window}},
        }

        _, windows, _ = codex_cli.parse_rate_limits(payload)

        assert [(item.key, item.label, item.blocking) for item in windows] == [
            ("weekly", "Weekly", True),
            ("spark:weekly", "Spark · Weekly", False),
        ]

    def test_an_entry_that_is_not_a_copy_of_an_unnamed_main_pool_is_not_treated_as_one(self) -> None:
        window = {"usedPercent": 5, "windowDurationMins": 10080}
        payload = {
            "rateLimits": {"primary": window},
            "rateLimitsByLimitId": {
                "not-json": "codex",
                "broken": {"primary": "weekly"},
                "other-use": {"primary": {"usedPercent": 80, "windowDurationMins": 300}},
                "other-span": {"primary": window, "secondary": {"usedPercent": 1, "windowDurationMins": 300}},
                "reached": {"primary": window, "rateLimitReachedType": "primary"},
                "named": {"primary": window, "limitName": "Spark"},
            },
        }

        _, windows, facts = codex_cli.parse_rate_limits(payload)

        assert [(item.key, item.blocking) for item in windows] == [
            ("weekly", True),
            ("other-use:five_hour", False),
            ("other-span:weekly", False),
            ("other-span:five_hour", False),
            ("reached:weekly", False),
            ("reached:reported_limit", False),
            ("named:weekly", False),
        ]
        assert facts == (Fact("Limits not shown", "2 extra limits with an unexpected shape; the log says which"),)

    def test_extra_limits_cannot_stand_in_for_a_main_limit_with_no_window(self) -> None:
        """None of them blocks, so alone they would show the account with room whatever its main limit says."""
        window = {"usedPercent": 10, "windowDurationMins": 10080}
        payload = {
            "rateLimits": {"limitId": "codex"},
            "rateLimitsByLimitId": {"meter_a": {"limitName": "Spark", "primary": window}},
        }

        with pytest.raises(ProviderSchemaError, match="reported no window of its main limit"):
            codex_cli.parse_rate_limits(payload)

    def test_a_main_limit_reported_as_reached_with_no_window_is_still_a_main_limit(self) -> None:
        payload = {"rateLimits": {"limitId": "codex", "rateLimitReachedType": "primary"}}

        _, windows, _ = codex_cli.parse_rate_limits(payload)

        assert [(w.key, w.blocking) for w in windows] == [("reported_limit", True)]

    def test_a_reached_extra_limit_is_shown_full_without_blocking_the_account(self) -> None:
        window = {"usedPercent": 10, "windowDurationMins": 10080}
        payload = {
            "rateLimits": {"limitId": "codex", "primary": window},
            "rateLimitsByLimitId": {
                "codex": {"limitId": "codex", "primary": window},
                "code_review": {
                    "limitId": "code_review",
                    "limitName": "Code review",
                    "primary": window,
                    "rateLimitReachedType": "primary",
                },
            },
        }

        _, windows, facts = codex_cli.parse_rate_limits(payload)

        by_key = {item.key: item for item in windows}
        reached = by_key["code_review:reported_limit"]
        assert (reached.used_percent, reached.blocking, reached.resets_at, reached.window_seconds) == (
            100,
            False,
            None,
            None,
        )
        assert reached.label == "Code review · Limit reached (reported by Codex)"
        assert not any(item.blocking and item.used_percent >= 100 for item in windows)
        assert all(fact.label != "Limits not shown" for fact in facts)

    def test_a_reached_extra_limit_adds_nothing_when_a_listed_window_is_already_full(self) -> None:
        window = {"usedPercent": 100, "windowDurationMins": 10080}
        payload = {
            "rateLimits": {"limitId": "codex", "primary": window},
            "rateLimitsByLimitId": {
                "spark": {"limitName": "Spark", "primary": window, "rateLimitReachedType": "primary"},
            },
        }

        _, windows, _ = codex_cli.parse_rate_limits(payload)

        assert [item.key for item in windows] == ["weekly", "spark:weekly"]

    def test_a_reached_extra_limit_with_no_window_is_shown_rather_than_omitted(self) -> None:
        payload = {
            "rateLimits": {"limitId": "codex", "primary": {"usedPercent": 10, "windowDurationMins": 10080}},
            "rateLimitsByLimitId": {"spark": {"limitName": "Spark", "rateLimitReachedType": "primary"}},
        }

        _, windows, facts = codex_cli.parse_rate_limits(payload)

        assert [(item.key, item.used_percent, item.blocking) for item in windows] == [
            ("weekly", 10, True),
            ("spark:reported_limit", 100, False),
        ]
        assert all(fact.label != "Limits not shown" for fact in facts)

    def test_every_extra_limit_left_out_is_counted_in_the_one_fact_that_says_so(self) -> None:
        window = {"usedPercent": 10, "windowDurationMins": 10080}
        payload = {
            "rateLimits": {"limitId": "codex", "primary": window},
            "rateLimitsByLimitId": {
                "": {"primary": window},
                "broken": {"primary": "weekly"},
                "ok": {"primary": window},
            },
        }

        _, windows, facts = codex_cli.parse_rate_limits(payload)

        assert [w.key for w in windows] == ["weekly", "ok:weekly"]
        assert [(fact.label, fact.value) for fact in facts] == [
            ("Limits not shown", "2 extra limits with an unexpected shape; the log says which")
        ]

    def test_a_result_without_windows_or_with_the_wrong_shape_is_a_schema_error(self) -> None:
        with pytest.raises(ProviderSchemaError, match="reported no window of its main limit"):
            codex_cli.parse_rate_limits({"rateLimits": {"limitId": "codex"}})
        with pytest.raises(ProviderSchemaError, match="unexpected shape: rateLimits: Field required"):
            codex_cli.parse_rate_limits({"accountId": "workspace-b"})


Meter = tuple[str, str | None, int]


class TestAnExtraCodexLimitIsKeyedByItsMeter:
    """The meter id identifies an extra limit in both sources. Its display name is only shown.

    The usage payload calls the id ``metered_feature``, and the Codex CLI passes it on as ``limitId``.
    """

    def from_the_cli(self, meters: list[Meter]) -> dict[str, tuple[str, float]]:
        """Read ``(meter id, display name, percent used)`` limits as the app-server reports them."""
        main = {"limitId": "codex", "primary": {"usedPercent": 1, "windowDurationMins": 10080}}
        by_id: dict[str, Any] = {"codex": main}
        for meter_id, name, used in meters:
            by_id[meter_id] = {
                "limitId": meter_id,
                "limitName": name,
                "primary": {"usedPercent": used, "windowDurationMins": 10080},
            }
        _, windows, _ = codex_cli.parse_rate_limits({"rateLimits": main, "rateLimitsByLimitId": by_id})
        return {window.key: (window.label, window.used_percent) for window in windows[1:]}

    def from_the_usage_endpoint(self, now: datetime, meters: list[Meter]) -> dict[str, tuple[str, float]]:
        """Read the same limits as the usage endpoint reports them to the direct source."""
        payload = codex_usage_payload(now)
        window = payload["rate_limit"]["primary_window"]
        payload["additional_rate_limits"] = [
            {
                "metered_feature": meter_id,
                "limit_name": name,
                "rate_limit": {"primary_window": {**window, "used_percent": used}},
            }
            for meter_id, name, used in meters
        ]
        _, windows, _ = codex.parse_usage(payload)
        return {window.key: (window.label, window.used_percent) for window in windows[1:]}

    def test_two_meters_with_one_display_name_keep_their_own_readings_in_either_order(self, now: datetime) -> None:
        meters: list[Meter] = [("meter_a", "Spark", 10), ("meter_b", "Spark", 90)]
        expected = {"meter_a:weekly": ("Spark · Weekly", 10.0), "meter_b:weekly": ("Spark · Weekly", 90.0)}

        assert self.from_the_cli(meters) == expected
        assert self.from_the_cli(meters[::-1]) == expected
        assert self.from_the_usage_endpoint(now, meters) == expected
        assert self.from_the_usage_endpoint(now, meters[::-1]) == expected

    def test_a_renamed_meter_keeps_its_key(self, now: datetime) -> None:
        before: list[Meter] = [("meter_a", "Spark", 10)]
        after: list[Meter] = [("meter_a", "Spark (preview)", 10)]

        assert list(self.from_the_cli(before)) == list(self.from_the_cli(after)) == ["meter_a:weekly"]
        assert self.from_the_cli(after)["meter_a:weekly"][0] == "Spark (preview) · Weekly"
        assert list(self.from_the_usage_endpoint(now, before)) == list(self.from_the_usage_endpoint(now, after))

    def test_the_meter_id_is_the_key_exactly_as_given_so_ids_that_look_alike_stay_apart(self, now: datetime) -> None:
        meters: list[Meter] = [("Spark-v2", None, 10), ("spark_v2", None, 90)]
        expected = {"Spark-v2:weekly": ("Spark v2 · Weekly", 10.0), "spark_v2:weekly": ("Spark v2 · Weekly", 90.0)}

        assert self.from_the_cli(meters) == expected
        assert self.from_the_usage_endpoint(now, meters) == expected

    def test_an_account_read_both_ways_has_each_limit_under_one_key(self, now: datetime) -> None:
        meters: list[Meter] = [
            ("codex_spark", "GPT-Codex-Spark", 10),
            ("deep_research", None, 20),
            ("review", "", 30),
        ]

        assert self.from_the_cli(meters) == self.from_the_usage_endpoint(now, meters)
        assert self.from_the_cli(meters) == {
            "codex_spark:weekly": ("GPT Codex Spark · Weekly", 10.0),
            "deep_research:weekly": ("Deep research · Weekly", 20.0),
            "review:weekly": ("Review · Weekly", 30.0),
        }

    @pytest.mark.parametrize("meter_id", ["", "   "])
    def test_the_cli_source_skips_and_logs_a_limit_whose_meter_id_is_blank(
        self, meter_id: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="usagedash.providers.codex_cli"):
            windows = self.from_the_cli([(meter_id, "Nameless", 50), ("meter_a", "Spark", 10)])

        assert list(windows) == ["meter_a:weekly"]
        assert "Leaving out Codex rate limit 1 that names no meter" in caplog.text

    @pytest.mark.parametrize("meter_id", ["", "   "])
    def test_the_cli_source_does_not_validate_a_limit_it_leaves_out_for_naming_no_meter(
        self, meter_id: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Validated first, an entry that was going to be left out would cost the reading its main limit."""
        main = {"limitId": "codex", "primary": {"usedPercent": 40, "windowDurationMins": 10080}}
        payload = {"rateLimits": main, "rateLimitsByLimitId": {meter_id: {"primary": {"usedPercent": "bad"}}}}

        with caplog.at_level(logging.WARNING, logger="usagedash.providers.codex_cli"):
            _, windows, _ = codex_cli.parse_rate_limits(payload)

        assert [(w.key, w.used_percent) for w in windows] == [("weekly", 40.0)]
        assert "Leaving out Codex rate limit 0 that names no meter" in caplog.text
        assert "failed validation" not in caplog.text

    @pytest.mark.parametrize(
        "snapshot",
        [
            pytest.param({"primary": {"usedPercent": "bad"}}, id="a percentage that is text"),
            pytest.param({"primary": {"usedPercent": -1}}, id="a negative percentage"),
            pytest.param({"primary": "weekly"}, id="a window that is not an object"),
            pytest.param("not an object", id="text"),
            pytest.param(None, id="null"),
            pytest.param(["primary"], id="a list"),
        ],
    )
    def test_the_cli_source_leaves_out_and_logs_an_extra_limit_whose_snapshot_fails_validation(
        self, snapshot: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An extra limit is a display detail. One that cannot be read costs only itself, as in the direct source."""
        window = {"usedPercent": 40, "windowDurationMins": 10080}
        payload = {
            "rateLimits": {"limitId": "codex", "primary": window},
            "rateLimitsByLimitId": {
                "codex": {"limitId": "codex", "primary": window},
                "broken_meter": snapshot,
                "meter_a": {"limitName": "Spark", "primary": {**window, "usedPercent": 10}},
            },
        }

        with caplog.at_level(logging.WARNING, logger="usagedash.providers.codex_cli"):
            _, windows, facts = codex_cli.parse_rate_limits(payload)

        assert [(w.key, w.used_percent) for w in windows] == [("weekly", 40.0), ("meter_a:weekly", 10.0)]
        assert caplog.text.count("Leaving out Codex rate limit 1 that failed validation") == 1
        assert [(fact.label, fact.value) for fact in facts] == [
            ("Limits not shown", "1 extra limit with an unexpected shape; the log says which")
        ]

    @pytest.mark.parametrize(
        "snapshot",
        [
            pytest.param({"limitName": "Spark"}, id="no window field"),
            pytest.param({"primary": None, "secondary": None}, id="every window null"),
            pytest.param({"tertiary": {"usedPercent": 10}}, id="only a window field this does not know"),
        ],
    )
    def test_the_cli_source_counts_an_extra_limit_with_no_window_as_not_shown(
        self, snapshot: dict[str, object], caplog: pytest.LogCaptureFixture
    ) -> None:
        """The direct source counts the same limit, sent with no rate limit, as not shown."""
        window = {"usedPercent": 40, "windowDurationMins": 10080}
        payload = {
            "rateLimits": {"limitId": "codex", "primary": window},
            "rateLimitsByLimitId": {"codex": {"limitId": "codex", "primary": window}, "meter_a": snapshot},
        }

        with caplog.at_level(logging.WARNING, logger="usagedash.providers.codex_cli"):
            _, windows, facts = codex_cli.parse_rate_limits(payload)

        assert [w.key for w in windows] == ["weekly"]
        assert "Leaving out Codex rate limit 1 with no window" in caplog.text
        assert [(fact.label, fact.value) for fact in facts] == [
            ("Limits not shown", "1 extra limit with an unexpected shape; the log says which")
        ]

    def test_the_main_limit_s_own_entry_in_the_map_is_neither_repeated_nor_validated(self) -> None:
        payload = {
            "rateLimits": {"limitId": "codex", "primary": {"usedPercent": 40, "windowDurationMins": 10080}},
            "rateLimitsByLimitId": {"codex": {"primary": {"usedPercent": "bad"}}},
        }

        _, windows, _ = codex_cli.parse_rate_limits(payload)

        assert [w.key for w in windows] == ["weekly"]

    @pytest.mark.parametrize("meter_id", ["", "   ", None, 7, ["meter_a"]])
    def test_the_direct_source_skips_and_logs_a_limit_that_names_no_meter(
        self, now: datetime, meter_id: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = codex_usage_payload(now)
        window = payload["rate_limit"]["primary_window"]
        payload["additional_rate_limits"] = [
            {"metered_feature": meter_id, "limit_name": "Nameless", "rate_limit": {"primary_window": window}},
            {"limit_name": "No meter at all", "rate_limit": {"primary_window": window}},
            {"metered_feature": "meter_a", "limit_name": 7, "rate_limit": {"primary_window": window}},
        ]

        with caplog.at_level(logging.WARNING, logger="usagedash.providers.codex"):
            _, windows, _ = codex.parse_usage(payload)

        assert [(w.key, w.label) for w in windows] == [("weekly", "Weekly"), ("meter_a:weekly", "Meter a · Weekly")]
        assert caplog.text.count("that names no meter") == 2


class TestGrokCliSource:
    def source(self, tmp_path: Path, scenario: str, record: Path, **extra_env: str) -> GrokCliSource:
        return GrokCliSource(tmp_path, FAKE_CLI, {**fake_environment("grok", scenario, record), **extra_env})

    def test_runs_the_cli_as_the_login_in_its_home_whatever_the_enclosing_shell_sets(
        self, tmp_path: Path, now: datetime
    ) -> None:
        """Grok takes a credential from ``GROK_AUTH`` ahead of the file in its home, where the identity is read."""
        grok_auth_file(tmp_path / "auth.json", user_id="grok-user-b", email="user-b@example.test")
        record = tmp_path / "record"
        source = self.source(
            tmp_path,
            "ok",
            record,
            GROK_HOME="/another/home",
            GROK_AUTH='{"synthetic": "another login"}',
            GROK_AUTH_PROVIDER_ACCESS_TOKEN="synthetic-token",
            GROK_CODE_XAI_API_KEY="synthetic-key",
            GROK_CONFIG_PATH="/another/config.toml",
            XAI_API_KEY="synthetic-key",
        )

        reading = source.read_quota(now)

        assert recorded(record)[0]["env"] == {"GROK_HOME": str(tmp_path)}
        assert reading.identity.external_id == "grok-user-b"

    def test_reads_billing_through_the_agent_protocol_without_opening_a_session(
        self, tmp_path: Path, now: datetime
    ) -> None:
        grok_auth_file(tmp_path / "auth.json", user_id="grok-user-b", email="user-b@example.test")
        record = tmp_path / "record"

        reading = self.source(tmp_path, "ok", record).read_quota(now)

        assert reading.source == "api"
        assert reading.observed_at == now
        assert (reading.identity.external_id, reading.identity.plan) == ("grok-user-b", None)
        assert [(w.key, w.used_percent) for w in reading.windows] == [("weekly", 31)]
        launch = recorded(record)[0]
        assert launch["env"] == {"GROK_HOME": str(tmp_path)}
        assert launch["argv"] == ["agent", "--no-leader", "stdio"]
        assert [message["method"] for message in received_messages(record)] == ["initialize", "_x.ai/billing"]

    def logged_before(self, home: Path, *lines: bytes) -> Path:
        log = home / "logs" / "unified.jsonl"
        log.parent.mkdir(exist_ok=True)
        log.write_bytes(b"".join(lines))
        return log

    def plan_after(self, home: Path, now: datetime, *logged_during: bytes) -> str | None:
        """Read once, with the CLI appending ``logged_during`` to its log while it answers the billing request."""
        grok_auth_file(home / "auth.json")
        logging = {"FAKE_GROK_LOG": b"".join(logged_during).decode()} if logged_during else {}
        return self.source(home, "ok", home / "record", **logging).read_quota(now).identity.plan

    def test_takes_the_plan_tier_from_the_record_the_cli_logs_during_the_exchange(
        self, tmp_path: Path, now: datetime
    ) -> None:
        self.logged_before(
            tmp_path,
            grok_billing_log_line(now - timedelta(hours=5), now, tier="Stale Tier"),
            grok_billing_log_line(now - timedelta(seconds=2), now, tier="Another Login's Tier"),
        )

        assert self.plan_after(tmp_path, now, grok_billing_log_line(now, now, tier="SuperGrok")) == "SuperGrok"

    def test_a_tier_logged_before_the_exchange_is_not_used_however_recent_it_is(
        self, tmp_path: Path, now: datetime
    ) -> None:
        """After a change of login, the record from seconds ago is the previous login's."""
        self.logged_before(tmp_path, grok_billing_log_line(now - timedelta(seconds=2), now, tier="Previous Login"))

        assert self.plan_after(tmp_path, now) is None

    def test_a_record_another_grok_process_logs_during_the_exchange_is_not_this_login_s_plan(
        self, tmp_path: Path, now: datetime
    ) -> None:
        """A request an open session began under an earlier login can be logged after the mark."""
        logged = grok_billing_log_line(now, now, tier="SuperGrok") + grok_billing_log_line(
            now, now, tier="Earlier Login", pid=1
        )

        assert self.plan_after(tmp_path, now, logged) == "SuperGrok"

    def test_a_home_with_no_log_yet_takes_the_tier_from_the_log_the_exchange_creates(
        self, tmp_path: Path, now: datetime
    ) -> None:
        assert self.plan_after(tmp_path, now, grok_billing_log_line(now, now, tier="SuperGrok")) == "SuperGrok"

    @pytest.mark.parametrize(
        "newest",
        [
            b'{"msg": "billing: fetched credits config", "ctx": {"subscriptionTier": "Cut sh',
            b'{"msg": "billing: fetched credits config", "ctx": {"subscriptionTier": 7}}\n',
            b'{"msg": "billing: fetched credits config", "ctx": "reshaped"}\n',
            b'{"msg": "billing: fetched credits config"}\n',
            b'{"msg": "billing: fetched credits config", "ctx": {}}\n',
        ],
    )
    def test_a_newest_record_that_gives_no_tier_leaves_the_plan_unknown_instead_of_using_an_older_one(
        self, tmp_path: Path, now: datetime, newest: bytes
    ) -> None:
        assert self.plan_after(tmp_path, now, grok_billing_log_line(now, now, tier="Older Record"), newest) is None

    def test_a_log_that_cannot_be_opened_after_the_exchange_leaves_the_plan_unknown_and_the_read_succeeds(
        self, tmp_path: Path, now: datetime
    ) -> None:
        log = self.logged_before(tmp_path, grok_billing_log_line(now - timedelta(hours=5), now))
        log.chmod(0)
        try:
            plan = self.plan_after(tmp_path, now)
        finally:
            log.chmod(0o600)

        assert plan is None

    def test_a_log_whose_length_cannot_be_told_before_the_exchange_leaves_the_plan_unknown(
        self, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the mark, what the exchange appended cannot be told from what was there before."""
        self.logged_before(tmp_path, grok_billing_log_line(now - timedelta(seconds=2), now, tier="Previous Login"))
        real_stat = Path.stat

        def stat(path: Path, **options: Any) -> os.stat_result:
            if path.name == "unified.jsonl":
                raise PermissionError(13, "Permission denied")
            return real_stat(path, **options)

        monkeypatch.setattr(Path, "stat", stat)

        assert self.plan_after(tmp_path, now, grok_billing_log_line(now, now, tier="SuperGrok")) is None

    def test_a_home_nobody_has_logged_in_to_starts_no_process(self, tmp_path: Path, now: datetime) -> None:
        record = tmp_path / "record"
        source = self.source(tmp_path, "ok", record)

        assert source.current_identity() is None
        with pytest.raises(CredentialUnavailableError, match="usagedash login grok"):
            source.read_quota(now)
        assert not record.exists()

    def test_a_billing_error_from_the_cli_is_a_request_error(self, tmp_path: Path, now: datetime) -> None:
        grok_auth_file(tmp_path / "auth.json")
        with pytest.raises(ProviderRequestError, match=r"rejected _x\.ai/billing \(error code -32603\)$"):
            self.source(tmp_path, "billing_error", tmp_path / "record").read_quota(now)

    def test_a_reshaped_billing_result_is_a_schema_error(self, tmp_path: Path, now: datetime) -> None:
        grok_auth_file(tmp_path / "auth.json")
        with pytest.raises(
            ProviderSchemaError,
            match=r"billing result has an unexpected shape: config: Value error, states neither a usage percentage "
            r"nor a complete current period$",
        ):
            self.source(tmp_path, "reshaped", tmp_path / "record").read_quota(now)

    def test_an_account_that_has_used_nothing_is_read_as_zero_with_the_figure_marked_as_not_stated(
        self, tmp_path: Path, now: datetime
    ) -> None:
        """Grok leaves the percentage out at zero. That is the account with the most room, not an unreadable one."""
        grok_auth_file(tmp_path / "auth.json")

        reading = self.source(tmp_path, "nothing_used", tmp_path / "record").read_quota(now)

        assert [(w.key, w.label, w.used_percent, w.window_seconds) for w in reading.windows] == [
            ("weekly", "Weekly credits", 0.0, WEEK)
        ]
        assert reading.windows[0].resets_at == datetime.fromisoformat("2026-10-08T14:49:41.842876+00:00")
        assert [(fact.label, fact.value) for fact in reading.facts] == [
            ("Credits used", "Not stated by Grok, which leaves the figure out at zero"),
            ("Prepaid balance", "11,970"),
        ]

    def test_the_source_is_named_by_its_provider_and_home(self, tmp_path: Path) -> None:
        source = self.source(tmp_path, "ok", tmp_path / "record")

        assert (source.provider, source.source_id) == (Provider.GROK, f"grok:{tmp_path}")

    def test_a_log_that_cannot_be_read_leaves_the_plan_unknown_without_failing_the_read(
        self, tmp_path: Path, now: datetime
    ) -> None:
        grok_auth_file(tmp_path / "auth.json")
        (tmp_path / "logs" / "unified.jsonl").mkdir(parents=True)

        reading = self.source(tmp_path, "ok", tmp_path / "record").read_quota(now)

        assert (reading.identity.plan, reading.windows[0].used_percent) == (None, 31)

    def test_a_cli_that_never_answers_times_out(
        self, tmp_path: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        grok_auth_file(tmp_path / "auth.json")
        monkeypatch.setattr(grok_cli, "CLI_TIMEOUT_SECONDS", 0.5)
        with pytest.raises(ProviderRequestError, match="did not answer in time"):
            self.source(tmp_path, "silent", tmp_path / "record").read_quota(now)


class TestAGrokBillingAnswerWithoutAUsageFigure:
    """Left out means zero only in an answer that is otherwise about a metered period."""

    def config(self, now: datetime, **changes: Any) -> dict[str, Any]:
        raw = grok_credits_config(now)
        del raw["creditUsagePercent"]
        return raw | changes

    @pytest.mark.parametrize("stated_as", ["left out", "null"])
    def test_with_a_complete_period_it_is_nothing_used_and_a_fact_says_it_was_not_stated(
        self, now: datetime, stated_as: str
    ) -> None:
        raw = self.config(now) if stated_as == "left out" else self.config(now, creditUsagePercent=None)

        windows, facts = grok.windows_and_facts(CreditsConfig.model_validate(raw))

        assert [(w.key, w.used_percent, w.resets_at) for w in windows] == [("weekly", 0.0, now + timedelta(days=1))]
        assert facts == (grok.UNSTATED_USAGE_FACT,)

    def test_a_stated_zero_is_zero_without_that_fact(self, now: datetime) -> None:
        windows, facts = grok.windows_and_facts(CreditsConfig.model_validate(grok_credits_config(now, percent=0)))

        assert windows[0].used_percent == 0.0
        assert facts == ()

    @pytest.mark.parametrize(
        "period",
        [
            pytest.param(None, id="no period"),
            pytest.param({}, id="an empty period"),
            pytest.param({"start": "2026-10-01T00:00:00Z", "end": "2026-10-08T00:00:00Z"}, id="no period type"),
            pytest.param({"type": "USAGE_PERIOD_TYPE_WEEKLY", "end": "2026-10-08T00:00:00Z"}, id="no start"),
            pytest.param({"type": "USAGE_PERIOD_TYPE_WEEKLY", "start": "2026-10-01T00:00:00Z"}, id="no end"),
            pytest.param(
                {"type": "", "start": "2026-10-01T00:00:00Z", "end": "2026-10-08T00:00:00Z"}, id="a blank type"
            ),
            pytest.param(
                {"type": "   ", "start": "2026-10-01T00:00:00Z", "end": "2026-10-08T00:00:00Z"}, id="a type of spaces"
            ),
            pytest.param(
                {"type": "USAGE_PERIOD_TYPE_WEEKLY", "start": "2026-10-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"},
                id="a period of no length",
            ),
            pytest.param(
                {"type": "USAGE_PERIOD_TYPE_WEEKLY", "start": "2026-10-08T00:00:00Z", "end": "2026-10-01T00:00:00Z"},
                id="a period that ends before it starts",
            ),
            pytest.param(
                {"type": "USAGE_PERIOD_TYPE_WEEKLY", "start": "2000-01-01T00:00:00Z", "end": "2010-01-01T00:00:01Z"},
                id="a period longer than any real one",
            ),
        ],
    )
    def test_without_a_complete_period_it_is_a_schema_error(self, now: datetime, period: dict[str, Any] | None) -> None:
        """Such an answer says nothing about usage, and zero would show an unknown account as the one to use."""
        with pytest.raises(ValidationError, match="states neither a usage percentage nor a complete current period"):
            CreditsConfig.model_validate(self.config(now, currentPeriod=period))

    def test_a_period_as_long_as_the_cap_is_complete(self, now: datetime) -> None:
        start = datetime(2000, 1, 1, tzinfo=UTC)
        raw = self.config(now, currentPeriod=self.period(start, start + timedelta(seconds=MAX_WINDOW_SECONDS)))

        windows, facts = grok.windows_and_facts(CreditsConfig.model_validate(raw))

        assert (windows[0].used_percent, windows[0].window_seconds) == (0.0, MAX_WINDOW_SECONDS)
        assert facts == (grok.UNSTATED_USAGE_FACT,)

    def test_a_period_one_microsecond_past_the_cap_is_not_complete(self, now: datetime) -> None:
        """Truncating to whole seconds first would accept a period the cap says is too long."""
        start = datetime(2000, 1, 1, tzinfo=UTC)
        end = start + timedelta(seconds=MAX_WINDOW_SECONDS, microseconds=1)

        with pytest.raises(ValidationError, match="states neither a usage percentage nor a complete current period"):
            CreditsConfig.model_validate(self.config(now, currentPeriod=self.period(start, end)))

    def test_a_period_one_nanosecond_past_the_cap_is_not_read_as_the_cap(self, now: datetime) -> None:
        """A datetime drops digits past a microsecond, so the extra nanosecond would vanish before the cap check."""
        period = {
            "type": "USAGE_PERIOD_TYPE_WEEKLY",
            "start": "2000-01-01T00:00:00Z",
            "end": "2009-12-29T00:00:00.000000001Z",
        }

        with pytest.raises(ValidationError, match=r"currentPeriod\.end"):
            CreditsConfig.model_validate(self.config(now, currentPeriod=period))

    def test_a_period_one_nanosecond_short_of_a_second_is_not_read_as_one_second(self, now: datetime) -> None:
        """The same dropped digit would turn a period shorter than a second into a usable one."""
        period = {
            "type": "USAGE_PERIOD_TYPE_WEEKLY",
            "start": "2026-10-01T00:00:00.000000001Z",
            "end": "2026-10-01T00:00:01Z",
        }

        with pytest.raises(ValidationError, match=r"currentPeriod\.start"):
            CreditsConfig.model_validate(self.config(now, currentPeriod=period))

    def test_zeros_past_a_microsecond_do_not_change_a_period_that_is_exactly_the_cap(self, now: datetime) -> None:
        raw = self.config(
            now,
            currentPeriod={
                "type": "USAGE_PERIOD_TYPE_WEEKLY",
                "start": "2000-01-01T00:00:00.000000000Z",
                "end": "2009-12-29T00:00:00.000000000Z",
            },
        )

        windows, facts = grok.windows_and_facts(CreditsConfig.model_validate(raw))

        assert (windows[0].used_percent, windows[0].window_seconds) == (0.0, MAX_WINDOW_SECONDS)
        assert facts == (grok.UNSTATED_USAGE_FACT,)

    def test_a_stated_figure_does_not_keep_a_bound_finer_than_a_microsecond(self, now: datetime) -> None:
        raw = grok_credits_config(now)
        raw["currentPeriod"]["end"] = "2026-10-08T00:00:00.000000001Z"

        with pytest.raises(ValidationError, match=r"currentPeriod\.end"):
            CreditsConfig.model_validate(raw)

    def period(self, start: datetime, end: datetime) -> dict[str, str]:
        def stamp(moment: datetime) -> str:
            return moment.isoformat().replace("+00:00", "Z")

        return {"type": "USAGE_PERIOD_TYPE_WEEKLY", "start": stamp(start), "end": stamp(end)}

    def test_a_period_of_a_kind_this_code_has_no_name_for_still_makes_the_answer_complete(self, now: datetime) -> None:
        """What the type is called is not what makes the period one: it has a type, a start, and an end."""
        raw = self.config(now)
        raw["currentPeriod"]["type"] = "USAGE_PERIOD_TYPE_FUTURE"

        windows, facts = grok.windows_and_facts(CreditsConfig.model_validate(raw))

        assert [(w.key, w.used_percent, w.window_seconds) for w in windows] == [("credits", 0.0, WEEK)]
        assert facts == (grok.UNSTATED_USAGE_FACT,)

    def test_a_config_with_no_period_at_all_and_no_figure_is_refused(self, now: datetime) -> None:
        raw = self.config(now)
        del raw["currentPeriod"]

        with pytest.raises(ValidationError, match="states neither a usage percentage nor a complete current period"):
            CreditsConfig.model_validate(raw)
