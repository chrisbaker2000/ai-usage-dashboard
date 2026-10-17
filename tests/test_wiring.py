import json
from datetime import datetime, timedelta
from pathlib import Path

import httpx2
import pytest

from tests.factories import make_identity
from tests.fakes.fake_run import FakeRun
from tests.payloads import write_claude_account_file
from tests.test_scanners import (
    ACCOUNT_A,
    append,
    claude_line,
    codex_count,
    codex_meta,
    codex_totals,
    everything_to,
    grok_turn,
    usage_rows,
)
from usagedash import credentials, db
from usagedash.collector import SourceBundle
from usagedash.config import ClaudeSourceConfig, Settings, SourceMode, load_settings
from usagedash.db import Database
from usagedash.errors import ConfigError
from usagedash.models import Provider, TokenCounts
from usagedash.providers.claude import ClaudeSource
from usagedash.providers.codex import CodexSource
from usagedash.providers.grok import GrokSource
from usagedash.providers.grok_cli import GrokCliSource
from usagedash.scanners.base import build_account_resolver
from usagedash.wiring import build_bundles, build_collector

KEYCHAIN_ITEM_NOT_FOUND = 44


class TestWiring:
    def test_direct_logins_get_a_session_scanner_and_standing_logins_do_not(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text(
            '[[claude]]\nconfig_dir = "~/homes/claude-work"\n[[codex]]\nhome = "~/homes/codex-work"\n'
            '[[grok]]\nhome = "~/homes/grok-work"\n'
        )
        settings = load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

        with httpx2.Client() as client:
            bundles = build_bundles(settings, client)

        described = [
            (b.source.provider.value, b.position, type(b.source).__name__, b.scanner is not None) for b in bundles
        ]
        assert described == [
            ("claude", 0, "ClaudeSource", True),
            ("claude", 1, "ClaudeCliSource", False),
            ("codex", 0, "CodexSource", True),
            ("codex", 1, "CodexCliSource", False),
            ("grok", 0, "GrokSource", True),
            ("grok", 1, "GrokCliSource", False),
        ]
        assert len({bundle.source.source_id for bundle in bundles}) == 6

    @pytest.mark.parametrize(("section", "field"), [("claude", "config_dir"), ("codex", "home"), ("grok", "home")])
    def test_one_home_listed_twice_under_two_spellings_is_refused(
        self, tmp_path: Path, section: str, field: str
    ) -> None:
        (tmp_path / "homes" / "work").mkdir(parents=True)
        config = tmp_path / "config.toml"
        config.write_text(
            f'[[{section}]]\n{field} = "~/homes/work"\nmode = "cli"\n'
            f'[[{section}]]\n{field} = "~/homes/work/../work"\nmode = "cli"\n'
        )
        settings = load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

        with httpx2.Client() as client, pytest.raises(ConfigError, match="which are one directory; list it once"):
            build_bundles(settings, client)

    @pytest.mark.parametrize("through", ["a missing directory", "a file"])
    def test_a_home_and_a_detour_that_resolves_to_it_are_refused(self, tmp_path: Path, through: str) -> None:
        home = tmp_path / "homes" / "work"
        home.mkdir(parents=True)
        if through == "a file":
            (home / "file").write_text("")
            other = home / "file" / ".."
        else:
            other = home / "missing" / ".."
        config = tmp_path / "config.toml"
        config.write_text(f'[[codex]]\nhome = "{home}"\nmode = "cli"\n[[codex]]\nhome = "{other}"\nmode = "cli"\n')
        settings = load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

        with httpx2.Client() as client, pytest.raises(ConfigError, match="which are one directory; list it once"):
            build_bundles(settings, client)

    def test_the_collector_runs_over_every_configured_login_and_reports_homes_nobody_is_logged_in_to(
        self, tmp_path: Path, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(credentials.subprocess, "run", FakeRun(returncode=KEYCHAIN_ITEM_NOT_FOUND))
        settings = load_settings(env={"USAGEDASH_DATA_DIR": str(tmp_path / "data")}, home=tmp_path)

        def refuse(request: httpx2.Request) -> httpx2.Response:
            raise AssertionError(f"unexpected request to {request.url}")

        with httpx2.Client(transport=httpx2.MockTransport(refuse)) as client:
            report = build_collector(settings, database, client, lambda: now).run_cycle()

        assert [(outcome.source_id, outcome.account_label) for outcome in report.outcomes] == [
            (f"claude:{tmp_path / '.claude'}", None),
            (f"codex:{tmp_path / '.codex'}", None),
            (f"grok:{tmp_path / '.grok'}", None),
        ]
        assert all((outcome.quota_error or "").startswith("CredentialUnavailableError") for outcome in report.outcomes)
        assert [(outcome.token_error, outcome.files_read) for outcome in report.outcomes] == [(None, 0)] * 3

    def test_another_spelling_of_a_home_keeps_the_source_and_its_login_timeline(
        self, tmp_path: Path, database: Database, now: datetime
    ) -> None:
        home = tmp_path / "homes" / "work"
        home.mkdir(parents=True)
        spelled = home / ".." / "work"
        usage = home / "sessions" / "%2Fproj" / "session-1" / "usage.json"
        usage.parent.mkdir(parents=True)
        usage.write_text(json.dumps({"turns": [grok_turn(1, now + timedelta(hours=1), 100, 0, 10)]}))

        with httpx2.Client() as client:
            first = self.custom_sources(self.settings_for(tmp_path, home), client)
            second = self.custom_sources(self.settings_for(tmp_path, spelled), client)

        assert [first[name].source.source_id for name in ("claude", "codex", "grok")] == [
            second[name].source.source_id for name in ("claude", "codex", "grok")
        ]
        assert all(".." not in bundle.source.source_id for bundle in first.values())
        source_id = first["grok"].source.source_id
        with database.connect() as conn:
            account_id = db.upsert_account(conn, make_identity(Provider.GROK, "work-login"), now)
            db.observe_active_account(conn, source_id, account_id, now)
            timeline = db.active_timeline(conn, second["grok"].source.source_id)
        assert second["grok"].scanner is not None
        second["grok"].scanner.scan(database, build_account_resolver(timeline), now, 35)

        assert usage_rows(database) == {(account_id, "2026-10-03T15", "grok-x"): TokenCounts(100, 0, 0, 10, 2, 3)}

    def test_a_direct_reader_and_a_cli_reader_store_one_identity_for_every_spelling(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        link = tmp_path / "link"
        link.symlink_to(home)
        spellings = (home, home / ".." / "home", link)

        def claude_id(path: Path, client: httpx2.Client) -> str:
            config = ClaudeSourceConfig(
                config_dir=path,
                account_file=path / ".claude.json",
                mode=SourceMode.DIRECT,
                keychain_service="test.keychain",
            )
            return ClaudeSource(config, client).source_id

        with httpx2.Client() as client:
            identities = (
                [claude_id(path, client) for path in spellings],
                [CodexSource(path, client).source_id for path in spellings],
                [GrokCliSource(path, ["grok"]).source_id for path in spellings],
            )

        for stored in identities:
            assert stored[0] == stored[1] == stored[2]
            assert ".." not in stored[0]
            assert not stored[0].endswith("/link")

    def test_retargeting_the_home_link_keeps_the_source_the_account_file_and_the_scanner(
        self, tmp_path: Path, database: Database, now: datetime
    ) -> None:
        original = tmp_path / "Original"
        transcript = original / "projects" / "proj" / "one.jsonl"
        transcript.parent.mkdir(parents=True)
        transcript.write_text(json.dumps(claude_line("original", now, output=200)) + "\n")
        write_claude_account_file(original / ".claude.json", "org-original", "Original Co")
        link = tmp_path / "link"
        link.symlink_to(original, target_is_directory=True)
        config_file = tmp_path / "config.toml"
        config_file.write_text(f'[[claude]]\nconfig_dir = "{link}"\nmode = "direct"\nkeychain_service = "synthetic"\n')
        settings = load_settings(env={"USAGEDASH_CONFIG": str(config_file)}, home=tmp_path)
        with httpx2.Client() as client:
            bundles = build_bundles(settings, client)
            source = ClaudeSource(
                ClaudeSourceConfig(
                    config_dir=link,
                    account_file=link / ".claude.json",
                    mode=SourceMode.DIRECT,
                    keychain_service="synthetic",
                ),
                client,
                lambda _service: "{}",
            )
        bundle = next(candidate for candidate in bundles if candidate.source.source_id.endswith("/Original"))
        assert bundle.scanner is not None
        scanner = bundle.scanner
        source_id = bundle.source.source_id
        assert source.source_id == source_id
        other = tmp_path / "Other"
        other_transcript = other / "projects" / "proj" / "two.jsonl"
        other_transcript.parent.mkdir(parents=True)
        other_transcript.write_text(json.dumps(claude_line("other", now, output=100)) + "\n")
        write_claude_account_file(other / ".claude.json", "org-other", "Other Co")
        link.unlink()
        link.symlink_to(other, target_is_directory=True)

        identity = source.current_identity()
        scanner.scan(database, everything_to(ACCOUNT_A), now, 35)

        assert bundle.source.source_id == source_id
        assert source.source_id == source_id
        assert "Other" not in source.source_id
        assert identity is not None
        assert identity.external_id == "org-original"
        assert sum(counts.output_tokens for counts in usage_rows(database).values()) == 200

    def test_retargeting_the_link_after_the_claude_source_is_built_keeps_the_scanner_there(
        self, tmp_path: Path, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = self.bundle_split_by_retarget(tmp_path, monkeypatch, now, "claude", ClaudeSource)
        assert bundle.scanner is not None

        report = bundle.scanner.scan(database, everything_to(ACCOUNT_A), now, 35)

        assert Path(bundle.source.source_id.removeprefix("claude:")).name == "Original"
        assert report.files_read == 0

    def test_retargeting_the_link_after_the_codex_source_is_built_keeps_the_scanner_there(
        self, tmp_path: Path, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = self.bundle_split_by_retarget(tmp_path, monkeypatch, now, "codex", CodexSource)
        assert bundle.scanner is not None

        report = bundle.scanner.scan(database, everything_to(ACCOUNT_A), now, 35)

        assert Path(bundle.source.source_id.removeprefix("codex:")).name == "Original"
        assert report.files_read == 0

    def test_retargeting_the_link_after_the_grok_reader_is_built_keeps_the_source_and_the_scanner_there(
        self, tmp_path: Path, database: Database, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = self.bundle_split_by_retarget(tmp_path, monkeypatch, now, "grok", GrokCliSource)
        assert isinstance(bundle.source, GrokSource)
        assert bundle.scanner is not None

        report = bundle.scanner.scan(database, everything_to(ACCOUNT_A), now, 35)

        assert Path(bundle.source.source_id.removeprefix("grok:")).name == "Original"
        assert bundle.source.source_id == bundle.source._reader.source_id
        assert report.files_read == 0

    def bundle_split_by_retarget(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        now: datetime,
        tool: str,
        source_type: type[object],
    ) -> SourceBundle:
        original = tmp_path / "Original"
        original.mkdir()
        other = tmp_path / "Other"
        other.mkdir()
        self.sessions_only_in(other, tool, now)
        link = tmp_path / "link"
        link.symlink_to(original, target_is_directory=True)
        config = tmp_path / "config.toml"
        if tool == "claude":
            config.write_text(f'[[claude]]\nconfig_dir = "{link}"\nmode = "direct"\nkeychain_service = "synthetic"\n')
        else:
            config.write_text(f'[[{tool}]]\nhome = "{link}"\nmode = "direct"\n')
        settings = load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)
        real_init = source_type.__init__

        def init(owner: object, configured: object, *args: object, **kwargs: object) -> None:
            real_init(owner, configured, *args, **kwargs)
            home = configured.config_dir if tool == "claude" else configured
            if home == link:
                link.unlink()
                link.symlink_to(other, target_is_directory=True)

        monkeypatch.setattr(source_type, "__init__", init)
        with httpx2.Client() as client:
            bundles = build_bundles(settings, client)
        matched = [
            bundle
            for bundle in bundles
            if bundle.source.source_id.endswith("/Original") or bundle.source.source_id.endswith("/Other")
        ]
        assert len(matched) == 1
        return matched[0]

    def sessions_only_in(self, home: Path, tool: str, now: datetime) -> None:
        if tool == "claude":
            append(home / "projects" / "proj" / "two.jsonl", claude_line("other", now))
            return
        if tool == "codex":
            append(
                home / "sessions" / "rollout-other.jsonl",
                codex_meta("workspace-other"),
                codex_count(now, codex_totals(9, 0, 1), codex_totals(9, 0, 1)),
            )
            return
        usage = home / "sessions" / "project" / "session-1" / "usage.json"
        usage.parent.mkdir(parents=True)
        usage.write_text(json.dumps({"turns": [grok_turn(1, now, 9, 0, 1)]}))

    def settings_for(self, root: Path, home: Path) -> Settings:
        config = root / "config.toml"
        config.write_text(
            f'[[claude]]\nconfig_dir = "{home}"\nmode = "cli"\n'
            f'[[codex]]\nhome = "{home}"\nmode = "cli"\n'
            f'[[grok]]\nhome = "{home}"\nmode = "direct"\n'
        )
        return load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=root)

    def custom_sources(self, settings: Settings, client: httpx2.Client) -> dict[str, SourceBundle]:
        bundles = {
            bundle.source.provider.value: bundle
            for bundle in build_bundles(settings, client)
            if "work" in bundle.source.source_id
        }
        assert set(bundles) == {"claude", "codex", "grok"}
        return bundles
