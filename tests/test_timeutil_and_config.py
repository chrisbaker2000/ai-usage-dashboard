import math
import os
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from usagedash import config, timeutil
from usagedash.config import SourceMode, load_settings, standing_source_table
from usagedash.errors import ConfigError
from usagedash.timeutil import (
    MAX_EPOCH_SECONDS,
    day_key,
    epoch_microseconds,
    from_epoch_microseconds,
    from_epoch_seconds,
    hour_bucket,
    in_utc,
    parse_hour_bucket,
    parse_iso,
    round_to_minute,
    to_iso,
    utc_now,
    whole_second_not_before,
)


class TestTimestampSerialization:
    def test_serializes_any_offset_as_second_precision_utc(self) -> None:
        eastern = datetime(2026, 10, 3, 10, 0, 0, 999999, tzinfo=timezone(timedelta(hours=-4)))
        assert to_iso(eastern) == "2026-10-03T14:00:00+00:00"

    @pytest.mark.parametrize(
        ("moment", "expected"),
        [
            pytest.param(
                datetime(2026, 10, 3, 14, 0, 30, tzinfo=UTC), datetime(2026, 10, 3, 14, 0, 30, tzinfo=UTC), id="whole"
            ),
            pytest.param(
                datetime(2026, 10, 3, 14, 0, 30, 1, tzinfo=UTC),
                datetime(2026, 10, 3, 14, 0, 31, tzinfo=UTC),
                id="a microsecond past",
            ),
            pytest.param(
                datetime(2026, 10, 3, 14, 0, 59, 999_999, tzinfo=UTC),
                datetime(2026, 10, 3, 14, 1, 0, tzinfo=UTC),
                id="a microsecond short of the next minute",
            ),
            pytest.param(
                datetime(2026, 10, 3, 10, 0, 30, 800_000, tzinfo=timezone(timedelta(hours=-4))),
                datetime(2026, 10, 3, 14, 0, 31, tzinfo=UTC),
                id="written in another offset",
            ),
        ],
    )
    def test_the_first_whole_second_at_or_after_a_time_is_never_before_it(
        self, moment: datetime, expected: datetime
    ) -> None:
        result = whole_second_not_before(moment)

        assert result == expected
        assert result.utcoffset() == timedelta(0)
        assert timedelta(0) <= result - moment < timedelta(seconds=1)

    @pytest.mark.parametrize(
        "helper", [to_iso, hour_bucket, day_key, round_to_minute, epoch_microseconds, whole_second_not_before]
    )
    def test_every_helper_refuses_a_naive_datetime_instead_of_assuming_this_machine_s_time_zone(
        self, helper: Callable[[datetime], object]
    ) -> None:
        """Python converts a naive value using the local zone, so 23:00 would be a different day elsewhere."""
        with pytest.raises(ValueError, match="a naive datetime has no defined instant"):
            helper(datetime(2026, 10, 3, 23, 0))

    def test_a_datetime_whose_time_zone_gives_no_offset_counts_as_naive(self) -> None:
        class NoOffset(tzinfo):
            def utcoffset(self, moment: datetime | None) -> timedelta | None:
                return None

            def dst(self, moment: datetime | None) -> timedelta | None:
                return None

            def tzname(self, moment: datetime | None) -> str | None:
                return None

        with pytest.raises(ValueError, match="a naive datetime has no defined instant"):
            hour_bucket(datetime(2026, 10, 3, 23, 0, tzinfo=NoOffset()))

    def test_parses_a_z_suffix_and_normalizes_to_utc(self) -> None:
        assert parse_iso("2026-10-03T10:00:00-04:00") == parse_iso("2026-10-03T14:00:00Z")

    def test_rejects_a_timestamp_without_an_offset(self) -> None:
        with pytest.raises(ValueError, match="no UTC offset"):
            parse_iso("2026-10-03T14:00:00")

    @pytest.mark.parametrize(
        "timestamp",
        [
            "2026-10-03T14:00:00.000000100Z",
            "2026-10-03T14:00:00.000000200Z",
            "2026-10-03T14:00:00,000000100Z",
        ],
    )
    def test_exact_parsing_rejects_a_fraction_a_datetime_would_drop(self, timestamp: str) -> None:
        with pytest.raises(ValueError, match="finer than a microsecond"):
            parse_iso(timestamp, exact=True)

    def test_exact_parsing_keeps_a_fraction_that_is_only_zero_past_a_microsecond(self) -> None:
        assert parse_iso("2026-10-03T14:00:00.123456000Z", exact=True) == parse_iso("2026-10-03T14:00:00.123456Z")

    @pytest.mark.parametrize(
        "timestamp",
        [
            "2026-10-03T14:00:00.000000000+01:00:00.000000100",
            "2026-10-03T14:00:00.000000000+01:00:00.000000200",
        ],
    )
    def test_exact_parsing_rejects_a_finer_fraction_hidden_in_the_offset(self, timestamp: str) -> None:
        """The first fraction is all zeros. A later one in the offset is still finer than a microsecond."""
        with pytest.raises(ValueError, match="finer than a microsecond"):
            parse_iso(timestamp, exact=True)

    def test_exact_parsing_keeps_an_offset_fraction_that_is_only_zeros(self) -> None:
        assert parse_iso("2026-10-03T14:00:00.000000000+00:00:00.000000000", exact=True) == parse_iso(
            "2026-10-03T14:00:00+00:00"
        )

    def test_parsing_without_exact_still_accepts_a_finer_fraction(self) -> None:
        assert parse_iso("2026-10-03T14:00:00.000000100Z") == datetime(2026, 10, 3, 14, 0, tzinfo=UTC)

    @pytest.mark.parametrize(
        ("second", "expected_minute"),
        [(0, 19), (29, 19), (30, 20), (59, 20)],
    )
    def test_rounds_to_the_nearest_minute_with_thirty_seconds_rounding_up(
        self, second: int, expected_minute: int
    ) -> None:
        moment = datetime(2026, 10, 3, 18, 19, second, 887958, tzinfo=UTC)
        assert round_to_minute(moment) == datetime(2026, 10, 3, 18, expected_minute, tzinfo=UTC)

    def test_rounding_gives_the_same_instant_whatever_offset_the_time_was_written_in(self) -> None:
        eastern = datetime(2026, 10, 3, 14, 19, 45, tzinfo=timezone(timedelta(hours=-4)))

        rounded = round_to_minute(eastern)

        assert rounded == datetime(2026, 10, 3, 18, 20, tzinfo=UTC)
        assert rounded.utcoffset() == timedelta(0)

    @pytest.mark.parametrize(
        ("fold", "expected"),
        [(0, datetime(2026, 11, 1, 5, 31, tzinfo=UTC)), (1, datetime(2026, 11, 1, 6, 31, tzinfo=UTC))],
    )
    def test_rounding_up_inside_the_hour_a_clock_change_repeats_keeps_the_right_occurrence(
        self, fold: int, expected: datetime
    ) -> None:
        """01:30:45 happens twice in New York on 2026-11-01; arithmetic on the local time forgets which."""
        local = datetime(2026, 11, 1, 1, 30, 45, tzinfo=ZoneInfo("America/New_York"), fold=fold)

        assert round_to_minute(local) == expected
        assert round_to_minute(local) - in_utc(local) == timedelta(seconds=15)

    @pytest.mark.parametrize(
        ("local", "expected"),
        [
            (datetime(2026, 3, 8, 1, 59, 45), datetime(2026, 3, 8, 7, 0, tzinfo=UTC)),
            (datetime(2026, 11, 1, 1, 59, 45), datetime(2026, 11, 1, 6, 0, tzinfo=UTC)),
        ],
    )
    def test_rounding_up_across_a_clock_change_moves_the_instant_by_seconds_not_an_hour(
        self, local: datetime, expected: datetime
    ) -> None:
        moment = local.replace(tzinfo=ZoneInfo("America/New_York"))

        assert round_to_minute(moment) == expected
        assert round_to_minute(moment) - in_utc(moment) == timedelta(seconds=15)

    def test_conversion_to_utc_keeps_the_instant_and_refuses_a_naive_value(self) -> None:
        eastern = datetime(2026, 10, 3, 10, 0, tzinfo=timezone(timedelta(hours=-4)))

        assert in_utc(eastern) == eastern
        assert in_utc(eastern).utcoffset() == timedelta(0)
        with pytest.raises(ValueError, match="a naive datetime has no defined instant"):
            in_utc(datetime(2026, 10, 3, 10, 0))

    def test_hour_bucket_round_trips_to_the_start_of_the_utc_hour(self) -> None:
        moment = datetime(2026, 10, 3, 9, 59, 59, tzinfo=timezone(timedelta(hours=-5)))
        assert hour_bucket(moment) == "2026-10-03T14"
        assert parse_hour_bucket("2026-10-03T14") == datetime(2026, 10, 3, 14, tzinfo=UTC)

    @pytest.mark.parametrize(
        ("moment", "hour", "day"),
        [
            pytest.param(datetime(1, 1, 1, 5, tzinfo=UTC), "0001-01-01T05", "0001-01-01", id="year 1"),
            pytest.param(datetime(999, 12, 31, 23, tzinfo=UTC), "0999-12-31T23", "0999-12-31", id="year 999"),
            pytest.param(datetime(1000, 1, 1, 0, tzinfo=UTC), "1000-01-01T00", "1000-01-01", id="year 1000"),
            pytest.param(datetime(9999, 12, 31, 23, tzinfo=UTC), "9999-12-31T23", "9999-12-31", id="year 9999"),
        ],
    )
    def test_keys_have_a_four_digit_year_whatever_the_platform_s_strftime_writes(
        self, moment: datetime, hour: str, day: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """glibc writes a year before 1000 with fewer digits, which does not parse back or sort in order."""

        class UnpaddedStrftime(datetime):
            def strftime(self, format: str) -> str:
                return super().strftime(format.replace("%Y", str(self.year)))

        real_in_utc = timeutil.in_utc

        def in_utc_as_glibc_formats_it(value: datetime) -> datetime:
            utc = real_in_utc(value)
            return UnpaddedStrftime.combine(utc.date(), utc.timetz())

        monkeypatch.setattr(timeutil, "in_utc", in_utc_as_glibc_formats_it)

        assert (hour_bucket(moment), day_key(moment)) == (hour, day)
        assert parse_hour_bucket(hour_bucket(moment)) == moment
        assert sorted([hour_bucket(datetime(1000, 1, 1, tzinfo=UTC)), hour_bucket(moment)]) == sorted(
            [hour_bucket(datetime(1000, 1, 1, tzinfo=UTC)), hour]
        )


class TestTheEdgesOfTheCalendar:
    """A timestamp from a payload or a session file may be anything; these raise ValueError and nothing else."""

    LAST_HOURS_WEST_OF_UTC = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone(timedelta(hours=-5)))
    FIRST_HOURS_EAST_OF_UTC = datetime(1, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=5)))

    @pytest.mark.parametrize("moment", [LAST_HOURS_WEST_OF_UTC, FIRST_HOURS_EAST_OF_UTC])
    def test_a_datetime_that_cannot_be_expressed_in_utc_is_a_value_error_everywhere(self, moment: datetime) -> None:
        for convert in (to_iso, hour_bucket, day_key):
            with pytest.raises(ValueError, match="cannot be expressed in UTC"):
                convert(moment)

    @pytest.mark.parametrize("text", ["9999-12-31T23:59:59-05:00", "0001-01-01T00:00:00+05:00"])
    def test_a_timestamp_that_cannot_be_expressed_in_utc_is_a_value_error(self, text: str) -> None:
        with pytest.raises(ValueError, match="cannot be expressed in UTC"):
            parse_iso(text)

    def test_rounding_up_past_the_last_minute_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="too close to the end of the calendar"):
            round_to_minute(datetime(9999, 12, 31, 23, 59, 30, tzinfo=UTC))

    def test_the_whole_second_after_the_last_one_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="too close to the end of the calendar"):
            whole_second_not_before(datetime(9999, 12, 31, 23, 59, 59, 1, tzinfo=UTC))

    def test_the_last_whole_second_is_itself(self) -> None:
        last_second = datetime(9999, 12, 31, 23, 59, 59, tzinfo=UTC)

        assert whole_second_not_before(last_second) == last_second

    def test_rounding_down_in_the_last_minute_still_works(self) -> None:
        last_minute = datetime(9999, 12, 31, 23, 59, 29, tzinfo=UTC)

        assert round_to_minute(last_minute) == datetime(9999, 12, 31, 23, 59, tzinfo=UTC)

    def test_the_last_instant_is_ordered_exactly_whatever_its_offset(self) -> None:
        assert epoch_microseconds(self.LAST_HOURS_WEST_OF_UTC) == (MAX_EPOCH_SECONDS + 5 * 3600) * 1_000_000


class TestClock:
    def test_the_current_time_is_an_aware_utc_datetime(self) -> None:
        before = datetime.now(UTC)
        moment = utc_now()

        assert moment.utcoffset() == timedelta(0)
        assert before <= moment <= datetime.now(UTC)
        assert to_iso(moment).endswith("+00:00")


class TestEpochConversion:
    def test_converts_epoch_seconds_to_utc(self) -> None:
        assert from_epoch_seconds(1_791_036_000) == datetime(2026, 10, 3, 14, 0, tzinfo=UTC)
        assert from_epoch_seconds(0) == datetime(1970, 1, 1, tzinfo=UTC)

    def test_the_last_second_a_datetime_can_hold_is_accepted(self) -> None:
        assert from_epoch_seconds(MAX_EPOCH_SECONDS) == datetime(9999, 12, 31, 23, 59, 59, tzinfo=UTC)

    def test_every_fraction_of_that_last_second_is_accepted_too(self) -> None:
        last_instant_a_float_can_name = math.nextafter(MAX_EPOCH_SECONDS + 1, 0)

        assert from_epoch_seconds(MAX_EPOCH_SECONDS + 0.5) == datetime(9999, 12, 31, 23, 59, 59, 500_000, tzinfo=UTC)
        assert from_epoch_seconds(last_instant_a_float_can_name).replace(microsecond=0) == datetime(
            9999, 12, 31, 23, 59, 59, tzinfo=UTC
        )

    @pytest.mark.parametrize(
        "seconds",
        [
            -1,
            -0.5,
            MAX_EPOCH_SECONDS + 1,
            1e300,
            float("inf"),
            float("-inf"),
            float("nan"),
            10**400,
            -(10**400),
        ],
    )
    def test_refuses_a_time_no_datetime_can_hold_with_one_error_type(self, seconds: float) -> None:
        """An integer too large for a float is included: testing it for finiteness would raise OverflowError."""
        with pytest.raises(ValueError, match="epoch seconds out of range"):
            from_epoch_seconds(seconds)


class TestExactOrdering:
    def test_counts_whole_microseconds_since_the_epoch_exactly(self) -> None:
        assert epoch_microseconds(datetime(1970, 1, 1, tzinfo=UTC)) == 0
        assert epoch_microseconds(datetime(1970, 1, 1, 0, 0, 1, 1, tzinfo=UTC)) == 1_000_001
        assert epoch_microseconds(datetime(2026, 10, 3, 14, 0, 0, 900_000, tzinfo=UTC)) == 1_791_036_000_900_000
        assert epoch_microseconds(datetime(9999, 12, 31, 23, 59, 59, 999_999, tzinfo=UTC)) == 253_402_300_799_999_999

    def test_orders_two_instants_that_serialize_to_the_same_second(self) -> None:
        earlier = datetime(2026, 10, 3, 14, 0, 0, 100_000, tzinfo=UTC)
        later = datetime(2026, 10, 3, 14, 0, 0, 900_000, tzinfo=UTC)

        assert to_iso(earlier) == to_iso(later)
        assert epoch_microseconds(earlier) < epoch_microseconds(later)

    def test_the_same_instant_in_another_offset_has_the_same_value(self) -> None:
        eastern = datetime(2026, 10, 3, 10, 0, tzinfo=timezone(timedelta(hours=-4)))

        assert epoch_microseconds(eastern) == epoch_microseconds(datetime(2026, 10, 3, 14, 0, tzinfo=UTC))

    @pytest.mark.parametrize(
        "moment",
        [
            datetime(1970, 1, 1, tzinfo=UTC),
            datetime(1969, 12, 31, 23, 59, 59, 999_999, tzinfo=UTC),
            datetime(2026, 10, 3, 14, 0, 0, 900_000, tzinfo=UTC),
            datetime(1, 1, 1, tzinfo=UTC),
            datetime(9999, 12, 31, 23, 59, 59, 999_999, tzinfo=UTC),
        ],
    )
    def test_microseconds_convert_back_to_the_same_instant_in_utc(self, moment: datetime) -> None:
        restored = from_epoch_microseconds(epoch_microseconds(moment))

        assert restored == moment
        assert restored.utcoffset() == timedelta(0)

    def test_an_instant_given_in_another_offset_comes_back_as_that_instant_in_utc(self) -> None:
        eastern = datetime(2026, 10, 3, 10, 0, 0, 1, tzinfo=timezone(timedelta(hours=-4)))

        restored = from_epoch_microseconds(epoch_microseconds(eastern))

        assert restored == datetime(2026, 10, 3, 14, 0, 0, 1, tzinfo=UTC)
        assert restored.utcoffset() == timedelta(0)

    @pytest.mark.parametrize(
        "microseconds",
        [
            (MAX_EPOCH_SECONDS + 1) * 1_000_000,
            epoch_microseconds(datetime(1, 1, 1, tzinfo=UTC)) - 1,
            10**30,
            -(10**30),
        ],
    )
    def test_refuses_microseconds_no_datetime_can_hold_with_one_error_type(self, microseconds: int) -> None:
        with pytest.raises(ValueError, match="epoch microseconds out of range"):
            from_epoch_microseconds(microseconds)

    def test_the_day_and_hour_of_an_instant_do_not_depend_on_the_offset_it_is_written_in(self) -> None:
        late_evening_eastern = datetime(2026, 10, 3, 23, 0, tzinfo=timezone(timedelta(hours=-4)))

        assert (hour_bucket(late_evening_eastern), day_key(late_evening_eastern)) == ("2026-10-04T03", "2026-10-04")


class TestDefaultSettings:
    def test_tracks_each_cli_home_directly_when_no_config_file_exists(self, tmp_path: Path) -> None:
        settings = load_settings(env={}, home=tmp_path)

        assert settings.port == 8787
        assert settings.host == "127.0.0.1"
        assert settings.poll_interval_seconds == 300
        assert settings.db_path == tmp_path / "Library" / "Application Support" / "ai-usage-dashboard" / "usage.db"
        assert [(source.config_dir, source.mode) for source in settings.claude_sources] == [
            (tmp_path / ".claude", SourceMode.DIRECT)
        ]
        assert settings.claude_sources[0].keychain_service == "Claude Code-credentials"
        assert settings.claude_sources[0].is_default_home
        assert [(source.home, source.mode) for source in settings.codex_sources] == [
            (tmp_path / ".codex", SourceMode.DIRECT)
        ]
        assert [(source.home, source.mode) for source in settings.grok_sources] == [
            (tmp_path / ".grok", SourceMode.DIRECT)
        ]


class TestConfiguredSources:
    def write_config(self, tmp_path: Path, text: str) -> dict[str, str]:
        config = tmp_path / "config.toml"
        config.write_text(text)
        return {"USAGEDASH_CONFIG": str(config), "USAGEDASH_DATA_DIR": str(tmp_path / "data")}

    def test_tables_add_logins_after_the_default_home_and_default_to_cli_mode(self, tmp_path: Path) -> None:
        env = self.write_config(
            tmp_path,
            '[[claude]]\nconfig_dir = "~/homes/claude-work"\n\n[[codex]]\nhome = "~/homes/codex-personal"\n\n'
            '[[grok]]\nhome = "/abs/grok-work"\nmode = "direct"\n',
        )

        settings = load_settings(env=env, home=tmp_path)

        claude_standing = settings.claude_sources[1]
        assert len(settings.claude_sources) == 2
        assert claude_standing.config_dir == tmp_path / "homes" / "claude-work"
        assert claude_standing.account_file == tmp_path / "homes" / "claude-work" / ".claude.json"
        assert claude_standing.mode is SourceMode.CLI
        assert not claude_standing.is_default_home
        assert settings.codex_sources[1].home == tmp_path / "homes" / "codex-personal"
        assert settings.codex_sources[1].mode is SourceMode.CLI
        assert settings.grok_sources[1].home == Path("/abs/grok-work")
        assert settings.grok_sources[1].mode is SourceMode.DIRECT

    def test_rendered_standing_table_is_read_back_as_a_cli_mode_login(self, tmp_path: Path) -> None:
        standing = tmp_path / 'odd "name"' / "claude-work"
        table = standing_source_table("claude", standing)

        assert tomllib.loads(table) == {"claude": [{"config_dir": str(standing), "mode": "cli"}]}
        assert tomllib.loads(standing_source_table("grok", standing)) == {
            "grok": [{"home": str(standing), "mode": "cli"}]
        }

    @pytest.mark.parametrize(
        "name",
        [
            "plain",
            'quote " and backslash \\',
            "line\nbreak",
            "carriage\rreturn",
            "tab\tand bell \x07",
            "delete \x7f",
            "unit separator \x1f",
            "caf\u00e9 \u6771\u4eac",
        ],
    )
    def test_any_path_a_standing_login_can_have_is_rendered_as_toml_that_reads_back_exactly(
        self, tmp_path: Path, name: str
    ) -> None:
        standing = tmp_path / name / "codex-work"

        parsed = tomllib.loads(standing_source_table("codex", standing))

        assert parsed == {"codex": [{"home": str(standing), "mode": "cli"}]}

    @pytest.mark.parametrize(
        "text",
        [
            '[[codex]]\nhome = "relative/codex"\n',
            '[[grok]]\nhome = "~other/grok"\n',
            '[[claude]]\nconfig_dir = "./claude"\n',
            '[[claude]]\nconfig_dir = "~/claude"\naccount_file = "account.json"\n',
        ],
    )
    def test_rejects_a_relative_path_which_would_mean_a_different_place_for_the_agent(
        self, tmp_path: Path, text: str
    ) -> None:
        env = self.write_config(tmp_path, text)
        with pytest.raises(ConfigError, match="must be absolute or start with ~/"):
            load_settings(env=env, home=tmp_path)

    @pytest.mark.parametrize(
        ("written", "meant"),
        [
            ("~/homes/codex", "homes/codex"),
            ("~//homes/codex", "homes/codex"),
            ("~///homes//codex", "homes/codex"),
            ("~/", ""),
            ("~//", ""),
            ("~", ""),
        ],
    )
    def test_a_path_under_the_home_directory_stays_under_it_however_many_slashes_follow_the_tilde(
        self, tmp_path: Path, written: str, meant: str
    ) -> None:
        """Joining a path that starts with a slash discards the home directory it was joined to."""
        home = tmp_path / "home"
        env = self.write_config(tmp_path, f'[[codex]]\nhome = "{written}"\nmode = "cli"\n')

        settings = load_settings(env=env, home=home)

        assert settings.codex_sources[-1].home == home / meant

    def test_a_data_directory_given_with_a_repeated_slash_stays_under_the_home_directory(self, tmp_path: Path) -> None:
        settings = load_settings(env={"USAGEDASH_DATA_DIR": "~//dashboard-data"}, home=tmp_path)

        assert settings.data_dir == tmp_path / "dashboard-data"

    @pytest.mark.parametrize("configured", ["./bin/codex", "bin/codex", "codex", "~other/bin/codex"])
    def test_rejects_a_relative_executable_which_would_mean_a_different_program_for_the_agent(
        self, tmp_path: Path, configured: str
    ) -> None:
        env = self.write_config(tmp_path, f'[binaries]\ncodex = "{configured}"\n')
        with pytest.raises(ConfigError, match="must be absolute or start with ~/"):
            load_settings(env=env, home=tmp_path)

    @pytest.mark.parametrize("written", ["~/bin/codex", "~//bin/codex"])
    def test_a_configured_executable_under_the_home_directory_is_expanded(self, tmp_path: Path, written: str) -> None:
        env = self.write_config(tmp_path, f'[binaries]\ncodex = "{written}"\n')

        assert load_settings(env=env, home=tmp_path).binaries.codex == str(tmp_path / "bin" / "codex")

    def test_rejects_an_executable_that_is_not_text(self, tmp_path: Path) -> None:
        env = self.write_config(tmp_path, "[binaries]\ncodex = 7\n")
        with pytest.raises(ConfigError):
            load_settings(env=env, home=tmp_path)

    @pytest.mark.parametrize(
        "text",
        [
            '[[codex]]\nhome = "/homes/co\\u0000dex"\n',
            '[[claude]]\nconfig_dir = "~/cla\\u0000ude"\n',
            '[[claude]]\nconfig_dir = "~/claude"\naccount_file = "/acc\\u0000ount.json"\n',
            '[binaries]\ngrok = "/opt/bin/gr\\u0000ok"\n',
        ],
    )
    def test_rejects_a_path_holding_a_nul_character_which_no_file_system_accepts(
        self, tmp_path: Path, text: str
    ) -> None:
        env = self.write_config(tmp_path, text)
        with pytest.raises(ConfigError, match="must not contain a NUL character"):
            load_settings(env=env, home=tmp_path)

    @pytest.mark.parametrize("literal", ["0x" + "F" * 4000, "0o" + "7" * 6000, "0b" + "1" * 16000])
    @pytest.mark.parametrize(
        "setting", ["port", "poll_interval_seconds", "backfill_days", "expected_accounts_per_provider"]
    )
    def test_rejects_an_integer_too_long_to_print_without_trying_to_print_it(
        self, tmp_path: Path, literal: str, setting: str
    ) -> None:
        """TOML parses a hexadecimal, octal, or binary integer of any length; Python cannot render one this long."""
        env = self.write_config(tmp_path, f"{setting} = {literal}\n")

        with pytest.raises(ConfigError, match=f"{setting} must be between .* got a int too large to print"):
            load_settings(env=env, home=tmp_path)

    @pytest.mark.parametrize("setting", ["warn_percent", "critical_percent"])
    def test_rejects_a_threshold_given_as_an_integer_too_long_to_print(self, tmp_path: Path, setting: str) -> None:
        env = self.write_config(tmp_path, f"{setting} = 0x{'F' * 4000}\n")

        with pytest.raises(ConfigError, match=f"{setting} is too large to be a number, got a int too large to print"):
            load_settings(env=env, home=tmp_path)

    @pytest.mark.parametrize("text", ["extra = " + "[" * 2000 + "0" + "]" * 2000, "extra = " + "{a = " * 2000])
    def test_rejects_a_file_nested_too_deeply_to_parse(self, tmp_path: Path, text: str) -> None:
        env = self.write_config(tmp_path, text + "\n")

        with pytest.raises(ConfigError, match="cannot read config file"):
            load_settings(env=env, home=tmp_path)

    @pytest.mark.parametrize(
        "failure",
        [
            ValueError("unforeseen"),
            OverflowError("unforeseen"),
            RecursionError("unforeseen"),
            PermissionError("unforeseen"),
            OSError("unforeseen"),
        ],
    )
    def test_whatever_else_interpreting_the_file_raises_is_still_a_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ) -> None:
        """The cases that were thought of have their own messages; this is the boundary behind them."""

        def fail(*_: object, **__: object) -> None:
            raise failure

        monkeypatch.setattr(config, "_binaries", fail)

        with pytest.raises(ConfigError, match=f"could not be interpreted: {type(failure).__name__}") as raised:
            load_settings(env={}, home=tmp_path)

        assert raised.value.__cause__ is failure

    def test_a_config_error_raised_inside_passes_through_with_its_own_message(self, tmp_path: Path) -> None:
        env = self.write_config(tmp_path, "port = 70000\n")

        with pytest.raises(ConfigError, match="port must be between"):
            load_settings(env=env, home=tmp_path)

    def test_an_empty_config_variable_is_like_an_unset_one_and_needs_no_file(self, tmp_path: Path) -> None:
        """It names no file, so the default place is read if there is one, and nothing is required."""
        env = {"USAGEDASH_CONFIG": "", "USAGEDASH_DATA_DIR": str(tmp_path / "data")}

        settings = load_settings(env=env, home=tmp_path)

        assert settings.config_path == tmp_path / "data" / "config.toml"

    @pytest.mark.parametrize("variable", ["USAGEDASH_DATA_DIR", "USAGEDASH_CONFIG"])
    def test_rejects_a_relative_path_in_the_environment(self, tmp_path: Path, variable: str) -> None:
        with pytest.raises(ConfigError, match="must be absolute or start with ~/"):
            load_settings(env={variable: "relative/place"}, home=tmp_path)

    def test_rejects_the_same_home_listed_twice(self, tmp_path: Path) -> None:
        env = self.write_config(tmp_path, '[[codex]]\nhome = "~/.codex"\n')
        with pytest.raises(ConfigError, match="more than once"):
            load_settings(env=env, home=tmp_path)

    def test_direct_mode_claude_login_requires_a_keychain_service(self, tmp_path: Path) -> None:
        env = self.write_config(tmp_path, '[[claude]]\nconfig_dir = "~/other"\nmode = "direct"\n')
        with pytest.raises(ConfigError, match="keychain_service"):
            load_settings(env=env, home=tmp_path)

    def test_rejects_an_unknown_mode(self, tmp_path: Path) -> None:
        env = self.write_config(tmp_path, '[[grok]]\nhome = "~/g"\nmode = "scrape"\n')
        with pytest.raises(ConfigError, match='"direct" or "cli"'):
            load_settings(env=env, home=tmp_path)

    def test_rejects_a_table_without_its_directory(self, tmp_path: Path) -> None:
        env = self.write_config(tmp_path, '[[claude]]\nmode = "cli"\n')
        with pytest.raises(ConfigError, match="requires config_dir"):
            load_settings(env=env, home=tmp_path)

    def test_configured_binaries_override_discovery(self, tmp_path: Path) -> None:
        env = self.write_config(tmp_path, '[binaries]\ncodex = "/custom/codex"\n')
        assert load_settings(env=env, home=tmp_path).binaries.codex == "/custom/codex"

    @pytest.mark.parametrize("name", ["claude", "codex", "grok"])
    def test_the_environment_s_executable_outranks_the_config_file_s(self, tmp_path: Path, name: str) -> None:
        """It is how agent install hands the agent each CLI the installing shell resolved."""
        env = self.write_config(tmp_path, f'[binaries]\n{name} = "/configured/{name}"\n')
        env[f"USAGEDASH_{name.upper()}_BINARY"] = f"/pinned/{name}"

        assert getattr(load_settings(env=env, home=tmp_path).binaries, name) == f"/pinned/{name}"

    @pytest.mark.parametrize("name", ["claude", "codex", "grok"])
    def test_a_pinned_bare_name_prevents_late_config_and_default_discovery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        env = self.write_config(tmp_path, f'[binaries]\n{name} = "/late/{name}"\n')
        env[f"USAGEDASH_{name.upper()}_BINARY"] = name
        for directory in (tmp_path / ".local" / "bin", tmp_path / ".grok" / "bin"):
            directory.mkdir(parents=True)
            (directory / name).touch()

        def which(command: str) -> str | None:
            assert command != name
            return None

        monkeypatch.setattr("usagedash.config.shutil.which", which)
        assert getattr(load_settings(env=env, home=tmp_path).binaries, name) == name
        env[f"USAGEDASH_{name.upper()}_BINARY"] = "another-tool"
        with pytest.raises(ConfigError, match="must be absolute"):
            load_settings(env=env, home=tmp_path)

    def test_the_environment_s_executable_follows_the_rule_for_every_configured_path(self, tmp_path: Path) -> None:
        """Relative, it would name another place for the agent than for a command run by hand."""
        with pytest.raises(ConfigError, match="must be absolute"):
            load_settings(env={"USAGEDASH_CODEX_BINARY": "bin/codex"}, home=tmp_path)
        assert load_settings(env={"USAGEDASH_CODEX_BINARY": "~/bin/codex"}, home=tmp_path).binaries.codex == str(
            tmp_path / "bin" / "codex"
        )

    @pytest.mark.parametrize("path_entry", ["", ".", "bin"], ids=["an empty entry", "dot", "a relative entry"])
    def test_a_cli_found_through_a_relative_path_entry_is_resolved_to_its_absolute_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_entry: str
    ) -> None:
        """Left relative, agent install took it for a name it could not resolve, and did not pin it."""
        directory = tmp_path / "bin" if path_entry == "bin" else tmp_path
        directory.mkdir(exist_ok=True)
        codex = directory / "codex"
        codex.write_text("#!/bin/sh\n")
        codex.chmod(0o755)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", f"{path_entry}{os.pathsep}/nonexistent")

        assert load_settings(env={}, home=tmp_path).binaries.codex == str(codex)

    def test_a_cli_that_path_does_not_find_is_left_as_its_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))

        assert load_settings(env={}, home=tmp_path).binaries.codex == "codex"

    def test_a_cli_at_its_usual_install_location_is_found_without_configuration(self, tmp_path: Path) -> None:
        usual = tmp_path / ".grok" / "bin" / "grok"
        usual.parent.mkdir(parents=True)
        usual.touch()

        assert load_settings(env={}, home=tmp_path).binaries.grok == str(usual)

    def test_rejects_binaries_that_are_not_a_table(self, tmp_path: Path) -> None:
        env = self.write_config(tmp_path, 'binaries = "codex"\n')
        with pytest.raises(ConfigError, match=r"\[binaries\] must be a table"):
            load_settings(env=env, home=tmp_path)

    @pytest.mark.parametrize("section", ["codex", "grok"])
    def test_rejects_a_home_table_without_its_directory(self, tmp_path: Path, section: str) -> None:
        env = self.write_config(tmp_path, f'[[{section}]]\nmode = "cli"\n')
        with pytest.raises(ConfigError, match=rf"\[\[{section}\]\] requires home"):
            load_settings(env=env, home=tmp_path)

    @pytest.mark.parametrize("text", ['claude = "not tables"\n', "codex = [1, 2]\n", '[grok]\nhome = "~/g"\n'])
    def test_rejects_a_login_section_that_is_not_an_array_of_tables(self, tmp_path: Path, text: str) -> None:
        env = self.write_config(tmp_path, text)
        with pytest.raises(ConfigError, match="must be an array of tables"):
            load_settings(env=env, home=tmp_path)

    @pytest.mark.parametrize("value", ["5", '""', "true"])
    def test_rejects_a_directory_that_is_not_a_non_empty_string(self, tmp_path: Path, value: str) -> None:
        env = self.write_config(tmp_path, f"[[codex]]\nhome = {value}\n")
        with pytest.raises(ConfigError, match=r"\[codex\] home must be a non-empty string"):
            load_settings(env=env, home=tmp_path)

    def test_a_bare_tilde_means_the_home_directory(self, tmp_path: Path) -> None:
        env = self.write_config(tmp_path, '[[claude]]\nconfig_dir = "~"\n')

        assert load_settings(env=env, home=tmp_path).claude_sources[1].config_dir == tmp_path

    def test_standing_homes_live_under_the_data_directory(self, tmp_path: Path) -> None:
        settings = load_settings(env={"USAGEDASH_DATA_DIR": str(tmp_path / "data")}, home=tmp_path)

        assert settings.standing_homes_dir == tmp_path / "data" / "homes"


class TestSettingValidation:
    def test_environment_overrides_the_file(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("port = 9000\npoll_interval_seconds = 120\n")
        env = {"USAGEDASH_CONFIG": str(config), "USAGEDASH_PORT": "9100"}

        settings = load_settings(env=env, home=tmp_path)

        assert settings.port == 9100
        assert settings.poll_interval_seconds == 120

    @pytest.mark.parametrize("port", ["1023", "65536", "0"])
    def test_rejects_a_port_outside_the_unprivileged_range(self, tmp_path: Path, port: str) -> None:
        with pytest.raises(ConfigError, match="port must be between"):
            load_settings(env={"USAGEDASH_PORT": port}, home=tmp_path)

    @pytest.mark.parametrize("port", ["1024", "65535"])
    def test_accepts_the_edges_of_the_port_range(self, tmp_path: Path, port: str) -> None:
        assert load_settings(env={"USAGEDASH_PORT": port}, home=tmp_path).port == int(port)

    def test_rejects_a_non_numeric_port(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="must be an integer"):
            load_settings(env={"USAGEDASH_PORT": "eighty"}, home=tmp_path)

    @pytest.mark.parametrize("seconds", ["59", "86401", "0", "-300"])
    def test_rejects_polling_faster_than_once_a_minute_or_slower_than_once_a_day(
        self, tmp_path: Path, seconds: str
    ) -> None:
        with pytest.raises(ConfigError, match="poll_interval_seconds must be between 60 and 86400"):
            load_settings(env={"USAGEDASH_POLL_INTERVAL_SECONDS": seconds}, home=tmp_path)

    @pytest.mark.parametrize("seconds", ["60", "86400"])
    def test_accepts_the_edges_of_the_polling_range(self, tmp_path: Path, seconds: str) -> None:
        settings = load_settings(env={"USAGEDASH_POLL_INTERVAL_SECONDS": seconds}, home=tmp_path)

        assert settings.poll_interval_seconds == int(seconds)

    @pytest.mark.parametrize(
        "text",
        ["warn_percent = 90\ncritical_percent = 90\n", "warn_percent = 0\n", "critical_percent = 101\n"],
    )
    def test_rejects_thresholds_that_are_not_strictly_ordered_within_range(self, tmp_path: Path, text: str) -> None:
        config = tmp_path / "config.toml"
        config.write_text(text)
        with pytest.raises(ConfigError, match="thresholds must satisfy"):
            load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

    def test_rejects_a_boolean_where_a_number_is_required(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("backfill_days = true\n")
        with pytest.raises(ConfigError, match="backfill_days must be an integer"):
            load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

    @pytest.mark.parametrize("days", [-1, 3651])
    def test_rejects_a_backfill_window_outside_zero_to_ten_years(self, tmp_path: Path, days: int) -> None:
        config = tmp_path / "config.toml"
        config.write_text(f"backfill_days = {days}\n")
        with pytest.raises(ConfigError, match="backfill_days must be between 0 and 3650"):
            load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

    @pytest.mark.parametrize("days", [0, 3650])
    def test_accepts_the_edges_of_the_backfill_range(self, tmp_path: Path, days: int) -> None:
        config = tmp_path / "config.toml"
        config.write_text(f"backfill_days = {days}\n")

        assert load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path).backfill_days == days

    def test_a_config_others_can_read_is_owner_only_after_it_loads(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("port = 9001\n")
        config.chmod(0o644)

        settings = load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

        assert settings.port == 9001
        assert config.stat().st_mode & 0o777 == 0o600

    def test_a_config_other_accounts_can_write_is_refused_and_keeps_its_mode(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("port = 9001\n")
        config.chmod(0o666)

        with pytest.raises(ConfigError, match="let other accounts write"):
            load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

        assert config.stat().st_mode & 0o777 == 0o666
        assert config.read_text() == "port = 9001\n"

    @pytest.mark.parametrize("accounts", [0, -1, 21])
    def test_rejects_an_expected_account_count_outside_one_to_twenty(self, tmp_path: Path, accounts: int) -> None:
        config = tmp_path / "config.toml"
        config.write_text(f"expected_accounts_per_provider = {accounts}\n")
        with pytest.raises(ConfigError, match="expected_accounts_per_provider must be between 1 and 20"):
            load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

    @pytest.mark.parametrize("accounts", [1, 20])
    def test_accepts_the_edges_of_the_expected_account_range(self, tmp_path: Path, accounts: int) -> None:
        config = tmp_path / "config.toml"
        config.write_text(f"expected_accounts_per_provider = {accounts}\n")
        settings = load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

        assert settings.expected_accounts_per_provider == accounts

    @pytest.mark.parametrize(
        "name", ["port", "poll_interval_seconds", "backfill_days", "expected_accounts_per_provider"]
    )
    def test_an_absurdly_large_integer_is_a_config_error_that_does_not_print_all_of_it(
        self, tmp_path: Path, name: str
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_text(f"{name} = {'9' * 400}\n")

        with pytest.raises(ConfigError, match=f"{name} must be between") as raised:
            load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

        assert "(400 characters)" in str(raised.value)
        assert "9" * 60 not in str(raised.value)

    @pytest.mark.parametrize("name", ["warn_percent", "critical_percent"])
    def test_a_threshold_too_large_for_a_number_is_a_config_error(self, tmp_path: Path, name: str) -> None:
        config = tmp_path / "config.toml"
        config.write_text(f"{name} = {'9' * 400}\n")

        with pytest.raises(ConfigError, match=f"{name} is too large to be a number"):
            load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

    @pytest.mark.parametrize(
        "text",
        ["warn_percent = nan\n", "critical_percent = nan\n", "critical_percent = inf\n", "warn_percent = -inf\n"],
    )
    def test_a_threshold_that_is_not_a_finite_number_is_rejected(self, tmp_path: Path, text: str) -> None:
        config = tmp_path / "config.toml"
        config.write_text(text)
        with pytest.raises(ConfigError, match="thresholds must satisfy"):
            load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

    def test_an_environment_value_with_too_many_digits_to_convert_is_a_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="port must be an integer") as raised:
            load_settings(env={"USAGEDASH_PORT": "9" * 5000}, home=tmp_path)

        assert len(str(raised.value)) < 200

    @pytest.mark.parametrize("value", ['"high"', "true"])
    def test_rejects_a_threshold_that_is_not_a_number(self, tmp_path: Path, value: str) -> None:
        config = tmp_path / "config.toml"
        config.write_text(f"warn_percent = {value}\n")
        with pytest.raises(ConfigError, match="warn_percent must be a number"):
            load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

    def test_an_explicitly_named_config_file_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="config file not found"):
            load_settings(env={"USAGEDASH_CONFIG": str(tmp_path / "missing.toml")}, home=tmp_path)

    @pytest.mark.parametrize(
        ("content", "reason"),
        [
            (b"port = = 1\n", "TOMLDecodeError"),
            (b"port = 8787 # caf\xe9\n", "UnicodeDecodeError"),
            (b"port = " + b"9" * 5000 + b"\n", "ValueError"),
        ],
    )
    def test_a_config_file_that_cannot_be_parsed_is_a_config_error_naming_why(
        self, tmp_path: Path, content: bytes, reason: str
    ) -> None:
        config = tmp_path / "config.toml"
        config.write_bytes(content)
        with pytest.raises(ConfigError, match=f"cannot read config file .*: {reason}"):
            load_settings(env={"USAGEDASH_CONFIG": str(config)}, home=tmp_path)

    def test_a_config_path_that_is_not_a_file_is_a_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"cannot use .*: it is not a regular file"):
            load_settings(env={"USAGEDASH_CONFIG": str(tmp_path)}, home=tmp_path)
