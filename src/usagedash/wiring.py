from collections.abc import Sequence
from pathlib import Path

import httpx2

from usagedash.collector import Collector, SourceBundle
from usagedash.config import Settings, SourceMode
from usagedash.db import Database
from usagedash.errors import ConfigError
from usagedash.paths import directory_identity, same_directory
from usagedash.providers.claude import ClaudeSource
from usagedash.providers.claude_cli import ClaudeCliSource
from usagedash.providers.codex import CodexSource
from usagedash.providers.codex_cli import CodexCliSource
from usagedash.providers.grok import GrokSource
from usagedash.providers.grok_cli import GrokCliSource
from usagedash.scanners.claude import ClaudeTranscriptScanner
from usagedash.scanners.codex import CodexSessionScanner
from usagedash.scanners.grok import GrokSessionScanner
from usagedash.timeutil import Clock

CLAUDE_PROJECTS_DIR_NAME = "projects"


def build_bundles(settings: Settings, client: httpx2.Client) -> list[SourceBundle]:
    _refuse_one_home_listed_twice("claude", [claude.config_dir for claude in settings.claude_sources])
    _refuse_one_home_listed_twice("codex", [codex.home for codex in settings.codex_sources])
    _refuse_one_home_listed_twice("grok", [grok.home for grok in settings.grok_sources])
    bundles: list[SourceBundle] = []
    for position, claude in enumerate(settings.claude_sources):
        if claude.mode is SourceMode.DIRECT:
            # The scanner reads the directory the source just identified, not the configured path
            # again: a link retargeted between the two would put the tokens on another source's timeline.
            source = ClaudeSource(claude, client)
            bundles.append(
                SourceBundle(
                    source=source,
                    scanner=ClaudeTranscriptScanner(source.home / CLAUDE_PROJECTS_DIR_NAME),
                    position=position,
                )
            )
        else:
            bundles.append(
                SourceBundle(
                    source=ClaudeCliSource(claude, [settings.binaries.claude], helper_directory=settings.data_dir),
                    scanner=None,
                    position=position,
                )
            )
    for position, codex in enumerate(settings.codex_sources):
        if codex.mode is SourceMode.DIRECT:
            codex_source = CodexSource(codex.home, client)
            bundles.append(
                SourceBundle(source=codex_source, scanner=CodexSessionScanner(codex_source.home), position=position)
            )
        else:
            bundles.append(
                SourceBundle(
                    source=CodexCliSource(codex.home, [settings.binaries.codex], helper_directory=settings.data_dir),
                    scanner=None,
                    position=position,
                )
            )
    for position, grok in enumerate(settings.grok_sources):
        if grok.mode is SourceMode.DIRECT:
            # Why not a direct CLI read: it would refresh a token an open session holds. Asked only while valid.
            reader = GrokCliSource(grok.home, [settings.binaries.grok], helper_directory=settings.data_dir)
            bundles.append(
                SourceBundle(
                    source=GrokSource(reader.home, reader),
                    scanner=GrokSessionScanner(reader.home),
                    position=position,
                )
            )
        else:
            bundles.append(
                SourceBundle(
                    source=GrokCliSource(grok.home, [settings.binaries.grok], helper_directory=settings.data_dir),
                    scanner=None,
                    position=position,
                )
            )
    return bundles


def _refuse_one_home_listed_twice(tool: str, homes: Sequence[Path]) -> None:
    identified = [directory_identity(home) for home in homes]
    for index, home in enumerate(homes):
        for earlier_index, earlier in enumerate(homes[:index]):
            if same_directory(identified[earlier_index], identified[index]):
                raise ConfigError(f"{tool} lists {earlier} and {home}, which are one directory; list it once")


def build_collector(settings: Settings, database: Database, client: httpx2.Client, clock: Clock) -> Collector:
    return Collector(
        database,
        build_bundles(settings, client),
        clock,
        poll_interval_seconds=settings.poll_interval_seconds,
        backfill_days=settings.backfill_days,
    )
