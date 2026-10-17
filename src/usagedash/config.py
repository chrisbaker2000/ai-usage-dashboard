"""Settings: built-in defaults, overridden by an optional TOML file, overridden by ``USAGEDASH_*`` env vars.

Each CLI's own home directory is always tracked. ``[[claude]]``, ``[[codex]]``, and ``[[grok]]`` tables in
the TOML file add further logins, typically standing logins created with ``usagedash login``.
"""

import io
import json
import os
import shutil
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from usagedash.errors import ConfigError
from usagedash.models import FULL_PERCENT
from usagedash.private import read_file_only_owner_can_change

# Why: the API serves account emails and usage history over plain HTTP, protected by a token meant for
# this Mac's own browser. It must never listen on a routable interface; remote viewing belongs behind a
# proxy that terminates TLS and authenticates, not a bind change.
LOOPBACK_HOST = "127.0.0.1"

DEFAULT_PORT = 8787
MIN_PORT = 1024
MAX_PORT = 65535
DEFAULT_POLL_INTERVAL_SECONDS = 300
MIN_POLL_INTERVAL_SECONDS = 60
MAX_POLL_INTERVAL_SECONDS = 24 * 60 * 60
DEFAULT_BACKFILL_DAYS = 35
MAX_BACKFILL_DAYS = 3650
MAX_EXPECTED_ACCOUNTS = 20
MAX_SHOWN_VALUE_LENGTH = 40
DEFAULT_WARN_PERCENT = 75.0
DEFAULT_CRITICAL_PERCENT = 90.0
DEFAULT_EXPECTED_ACCOUNTS = 2
DEFAULT_CLAUDE_KEYCHAIN_SERVICE = "Claude Code-credentials"
CLAUDE_ACCOUNT_FILE_NAME = ".claude.json"
CONFIG_FILE_NAME = "config.toml"
STANDING_HOMES_DIR_NAME = "homes"

SETTINGS_ENV_PREFIX = "USAGEDASH_"
CONFIG_ENV = f"{SETTINGS_ENV_PREFIX}CONFIG"
DATA_DIR_ENV = f"{SETTINGS_ENV_PREFIX}DATA_DIR"
PORT_ENV = f"{SETTINGS_ENV_PREFIX}PORT"
POLL_INTERVAL_ENV = f"{SETTINGS_ENV_PREFIX}POLL_INTERVAL_SECONDS"


class SourceMode(StrEnum):
    """How a login is read.

    ``DIRECT`` reads the CLI's live credential or log without modifying anything. ``CLI`` asks the vendor
    CLI itself, in its headless protocol mode, so the CLI keeps its own session refreshed.
    """

    DIRECT = "direct"
    CLI = "cli"


@dataclass(frozen=True, slots=True)
class ClaudeSourceConfig:
    """One Claude Code login location. ``keychain_service`` is only needed in direct mode."""

    config_dir: Path
    account_file: Path
    mode: SourceMode
    keychain_service: str | None = None
    is_default_home: bool = False


@dataclass(frozen=True, slots=True)
class HomeSourceConfig:
    """One Codex or Grok home directory holding a login."""

    home: Path
    mode: SourceMode


@dataclass(frozen=True, slots=True)
class Binaries:
    """Executables used for CLI-mode reads and for ``usagedash login``."""

    claude: str
    codex: str
    grok: str


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    config_path: Path
    port: int
    poll_interval_seconds: int
    backfill_days: int
    warn_percent: float
    critical_percent: float
    expected_accounts_per_provider: int
    claude_sources: tuple[ClaudeSourceConfig, ...]
    codex_sources: tuple[HomeSourceConfig, ...]
    grok_sources: tuple[HomeSourceConfig, ...]
    binaries: Binaries

    @property
    def db_path(self) -> Path:
        return self.data_dir / "usage.db"

    @property
    def standing_homes_dir(self) -> Path:
        return self.data_dir / STANDING_HOMES_DIR_NAME

    @property
    def host(self) -> str:
        return LOOPBACK_HOST


def default_data_dir(home: Path) -> Path:
    return home / "Library" / "Application Support" / "ai-usage-dashboard"


def config_file(env: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    """Return the config file ``load_settings`` would read, without reading it.

    Raises:
        ConfigError: if a configured path is relative or contains a NUL character.
    """
    environment = os.environ if env is None else env
    home_dir = Path.home() if home is None else home
    _data_dir, path, _named = _config_location(environment, home_dir)
    return path


def _config_location(environment: Mapping[str, str], home_dir: Path) -> tuple[Path, Path, bool]:
    """Return the data directory, the config path, and whether the environment named that file."""
    data_dir = _expand(environment.get(DATA_DIR_ENV), home_dir) or default_data_dir(home_dir)
    named = _expand(environment.get(CONFIG_ENV), home_dir)
    if named is None:
        return data_dir, data_dir / CONFIG_FILE_NAME, False
    return data_dir, named, True


def load_settings(
    env: Mapping[str, str] | None = None, home: Path | None = None, *, file_may_be_absent: bool = False
) -> Settings:
    """Build settings from defaults, the TOML file (if present), and environment overrides.

    A file the environment names must exist, unless ``file_may_be_absent``: then one that is not there
    holds nothing, and the settings come from defaults and the environment alone.

    A config file can hold anything, and Python has limits a file can exceed: an integer too long to
    print, a number too large for a float, nesting too deep to parse. Each known case is handled where
    it arises, with a message that names the setting. This is the boundary behind them: whatever
    interpreting the file raises, the caller gets a ``ConfigError``. That includes a filesystem error
    while looking for a CLI in its usual location, which is part of working out the settings.

    Raises:
        ConfigError: if the TOML file is unreadable or any value is the wrong type or out of range.
    """
    try:
        return _settings_from(
            os.environ if env is None else env, Path.home() if home is None else home, file_may_be_absent
        )
    except (OSError, ValueError, ArithmeticError, RecursionError) as error:
        raise ConfigError(f"the configuration could not be interpreted: {type(error).__name__}") from error


def _settings_from(environment: Mapping[str, str], home_dir: Path, file_may_be_absent: bool = False) -> Settings:
    data_dir, config_path, named = _config_location(environment, home_dir)
    # Why: only a file the environment names must exist. An empty variable names none, like an unset one.
    file_values = _read_toml(config_path, required=named and not file_may_be_absent)

    port = _int_setting(
        environment.get(PORT_ENV), file_values.get("port"), DEFAULT_PORT, "port", minimum=MIN_PORT, maximum=MAX_PORT
    )
    poll_interval = _int_setting(
        environment.get(POLL_INTERVAL_ENV),
        file_values.get("poll_interval_seconds"),
        DEFAULT_POLL_INTERVAL_SECONDS,
        "poll_interval_seconds",
        minimum=MIN_POLL_INTERVAL_SECONDS,
        maximum=MAX_POLL_INTERVAL_SECONDS,
    )
    backfill_days = _int_setting(
        None,
        file_values.get("backfill_days"),
        DEFAULT_BACKFILL_DAYS,
        "backfill_days",
        minimum=0,
        maximum=MAX_BACKFILL_DAYS,
    )
    expected_accounts = _int_setting(
        None,
        file_values.get("expected_accounts_per_provider"),
        DEFAULT_EXPECTED_ACCOUNTS,
        "expected_accounts_per_provider",
        minimum=1,
        maximum=MAX_EXPECTED_ACCOUNTS,
    )
    warn_percent = _float_setting(file_values.get("warn_percent"), DEFAULT_WARN_PERCENT, "warn_percent")
    critical_percent = _float_setting(file_values.get("critical_percent"), DEFAULT_CRITICAL_PERCENT, "critical_percent")
    # Why this also covers NaN and infinity, which TOML allows: NaN fails every comparison and infinity
    # is above 100.
    if not 0 < warn_percent < critical_percent <= FULL_PERCENT:
        raise ConfigError(
            f"thresholds must satisfy 0 < warn_percent < critical_percent <= 100, "
            f"got warn={warn_percent} critical={critical_percent}"
        )

    return Settings(
        data_dir=data_dir,
        config_path=config_path,
        port=port,
        poll_interval_seconds=poll_interval,
        backfill_days=backfill_days,
        warn_percent=warn_percent,
        critical_percent=critical_percent,
        expected_accounts_per_provider=expected_accounts,
        claude_sources=_claude_sources(file_values.get("claude"), home_dir),
        codex_sources=_home_sources(file_values.get("codex"), home_dir / ".codex", home_dir, "codex"),
        grok_sources=_home_sources(file_values.get("grok"), home_dir / ".grok", home_dir, "grok"),
        binaries=_binaries(file_values.get("binaries"), home_dir, environment),
    )


def _expand(raw: str | None, home: Path) -> Path | None:
    """Turn a configured path into an absolute one, with a leading ``~`` meaning the home directory.

    Raises:
        ConfigError: if the path is relative, or holds a NUL character. The background agent and a
            command run by hand start in different directories, so a relative path would name a
            different place for each of them. No file system accepts a NUL, and the calls that would
            find that out raise something other than the errors a bad path normally causes.
    """
    if raw is None or raw == "":
        return None
    if "\x00" in raw:
        raise ConfigError("a configured path must not contain a NUL character")
    if raw == "~":
        return home
    if raw.startswith("~/"):
        # Why the extra slashes are removed: joining a path that starts with one discards everything
        # before it, so "~//x" would name "/x" and not a place under the home directory.
        return home / raw[2:].lstrip("/")
    path = Path(raw)
    if not path.is_absolute():
        raise ConfigError(f"a configured path must be absolute or start with ~/, got {_shown(raw)}")
    return path


def _read_toml(path: Path, *, required: bool) -> dict[str, Any]:
    """Read a config file another account cannot change, and parse those bytes.

    The check and the read share one descriptor. A file another account can write, a symbolic
    link, or anything that is not a regular file is refused and left unchanged: the bytes are
    what later code runs, so changing the mode afterwards would not make them trustworthy. A file
    this account owns that others can only read is made owner-only on that descriptor before the
    bytes are used.

    Raises:
        ConfigError: if a required file is missing, or the file cannot be used or parsed.
    """
    try:
        content = read_file_only_owner_can_change(path)
    except FileNotFoundError as error:
        if required:
            raise ConfigError(f"config file not found: {path}") from error
        return {}
    try:
        return tomllib.load(io.BytesIO(content))
    except (OSError, ValueError, RecursionError) as error:
        # Why ValueError: it covers TOMLDecodeError, a file that is not UTF-8 (UnicodeDecodeError), and a
        # number with more digits than Python will convert. RecursionError is a file nested too deeply to
        # parse. All of them mean the file cannot be used.
        raise ConfigError(f"cannot read config file {path}: {type(error).__name__}: {error}") from error


def _shown(value: object) -> str:
    """Describe a rejected value for an error message without printing an unbounded amount of it."""
    try:
        text = repr(value)
    except ValueError:
        # Why: Python refuses to print an integer of more than a few thousand digits. TOML's decimal
        # integers are stopped by the same limit when the file is parsed, but hexadecimal, octal, and
        # binary ones are not, so such a value can reach here.
        return f"a {type(value).__name__} too large to print"
    return text if len(text) <= MAX_SHOWN_VALUE_LENGTH else f"{text[:MAX_SHOWN_VALUE_LENGTH]}… ({len(text)} characters)"


def _int_setting(
    env_value: str | None, file_value: object, default: int, name: str, *, minimum: int, maximum: int
) -> int:
    """Read an integer setting, preferring the environment, and require it to lie within its range.

    Every integer setting has an upper bound as well as a lower one, so no later arithmetic on it (a
    sleep, a date offset) can overflow.
    """
    if env_value is not None and env_value != "":
        try:
            value = int(env_value)
        except ValueError as error:
            raise ConfigError(f"{name} must be an integer, got {_shown(env_value)}") from error
    elif file_value is None:
        value = default
    elif isinstance(file_value, bool) or not isinstance(file_value, int):
        raise ConfigError(f"{name} must be an integer, got {_shown(file_value)}")
    else:
        value = file_value
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}, got {_shown(value)}")
    return value


def _float_setting(file_value: object, default: float, name: str) -> float:
    if file_value is None:
        return default
    if isinstance(file_value, bool) or not isinstance(file_value, int | float):
        raise ConfigError(f"{name} must be a number, got {_shown(file_value)}")
    try:
        return float(file_value)
    except OverflowError as error:
        raise ConfigError(f"{name} is too large to be a number, got {_shown(file_value)}") from error


def _string_field(table: Mapping[str, Any], key: str, section: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise ConfigError(f"[{section}] {key} must be a non-empty string, got {_shown(value)}")
    return value


def _tables(raw: object, section: str) -> list[Mapping[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ConfigError(f"[[{section}]] must be an array of tables")
    return raw


def _mode(table: Mapping[str, Any], section: str) -> SourceMode:
    raw = _string_field(table, "mode", section)
    if raw is None:
        return SourceMode.CLI
    try:
        return SourceMode(raw)
    except ValueError as error:
        raise ConfigError(f'[[{section}]] mode must be "direct" or "cli", got {_shown(raw)}') from error


def _claude_sources(raw: object, home: Path) -> tuple[ClaudeSourceConfig, ...]:
    sources = [
        ClaudeSourceConfig(
            config_dir=home / ".claude",
            account_file=home / CLAUDE_ACCOUNT_FILE_NAME,
            mode=SourceMode.DIRECT,
            keychain_service=DEFAULT_CLAUDE_KEYCHAIN_SERVICE,
            is_default_home=True,
        )
    ]
    for table in _tables(raw, "claude"):
        config_dir = _expand(_string_field(table, "config_dir", "claude"), home)
        if config_dir is None:
            raise ConfigError("[[claude]] requires config_dir")
        mode = _mode(table, "claude")
        keychain_service = _string_field(table, "keychain_service", "claude")
        if mode is SourceMode.DIRECT and keychain_service is None:
            raise ConfigError('[[claude]] with mode = "direct" requires keychain_service')
        account_file = _expand(_string_field(table, "account_file", "claude"), home)
        sources.append(
            ClaudeSourceConfig(
                config_dir=config_dir,
                account_file=account_file or config_dir / CLAUDE_ACCOUNT_FILE_NAME,
                mode=mode,
                keychain_service=keychain_service,
            )
        )
    _reject_duplicates([source.config_dir for source in sources], "claude")
    return tuple(sources)


def _home_sources(raw: object, default: Path, home: Path, section: str) -> tuple[HomeSourceConfig, ...]:
    sources = [HomeSourceConfig(home=default, mode=SourceMode.DIRECT)]
    for table in _tables(raw, section):
        path = _expand(_string_field(table, "home", section), home)
        if path is None:
            raise ConfigError(f"[[{section}]] requires home")
        sources.append(HomeSourceConfig(home=path, mode=_mode(table, section)))
    _reject_duplicates([source.home for source in sources], section)
    return tuple(sources)


def _reject_duplicates(paths: list[Path], section: str) -> None:
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            raise ConfigError(f"[[{section}]] lists {path} more than once")
        seen.add(path)


def binary_env(name: str) -> str:
    """Return the environment variable that names a vendor CLI's executable, such as ``USAGEDASH_CODEX_BINARY``.

    ``agent install`` sets these in the agent's environment to the executables it resolved, so that the
    agent runs the same installations the installing shell found instead of looking them up again by
    name in another PATH.
    """
    return f"{SETTINGS_ENV_PREFIX}{name.upper()}_BINARY"


def _binaries(raw: object, home: Path, environment: Mapping[str, str]) -> Binaries:
    if raw is not None and not isinstance(raw, dict):
        raise ConfigError("[binaries] must be a table")
    table: Mapping[str, Any] = raw or {}
    return Binaries(
        claude=_binary(table, "claude", home, home / ".local" / "bin" / "claude", environment),
        codex=_binary(table, "codex", home, None, environment),
        grok=_binary(table, "grok", home, home / ".grok" / "bin" / "grok", environment),
    )


def _binary(
    table: Mapping[str, Any], name: str, home: Path, usual_location: Path | None, environment: Mapping[str, str]
) -> str:
    """Resolve an executable: the environment's (``binary_env``), the configured path, its usual install
    location, or whatever PATH finds, in that order.

    What PATH finds is made absolute: through an empty or relative entry it is found relative to the
    working directory, and ``agent install`` pins only an absolute path, so it would leave that CLI to
    be looked up again by the agent, in another directory. A name PATH does not find is returned as
    it is. An environment override equal to this tool's bare name requests PATH-only lookup, without
    later default-directory or config-file discovery; agent install records that when the CLI is absent.

    Raises:
        ConfigError: if an override is not text, or is relative other than that exact environment sentinel.
            Configured paths always follow the same absolute-path rule as other paths.
    """
    if environment.get(binary_env(name)) == name:
        return name
    from_environment = _expand(environment.get(binary_env(name)), home)
    if from_environment is not None:
        return str(from_environment)
    configured = _expand(_string_field(table, name, "binaries"), home)
    if configured is not None:
        return str(configured)
    if usual_location is not None and usual_location.exists():
        return str(usual_location)
    found = shutil.which(name)
    return name if found is None else str(Path(found).absolute())


def standing_source_table(provider_section: str, standing_home: Path) -> str:
    """Render the TOML table that registers a standing login for a provider.

    The path is written as a TOML basic string. A JSON string is one: it escapes quotes, backslashes,
    and control characters the same way, except for DEL, which JSON allows and TOML does not.
    """
    key = "config_dir" if provider_section == "claude" else "home"
    quoted = json.dumps(str(standing_home), ensure_ascii=False).replace("\x7f", "\\u007f")
    return f'\n[[{provider_section}]]\n{key} = {quoted}\nmode = "cli"\n'
