"""Direct-mode sources: reading the CLIs' live credentials and quotas without modifying them."""

import json
import logging
import math
import traceback
import types
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from tests.factories import (
    FIVE_HOURS,
    PERSONA_EMAIL,
    WEEK,
)
from tests.payloads import (
    claude_keychain_blob,
    claude_profile_payload,
    claude_usage_payload,
    codex_auth_file,
    codex_usage_payload,
    grok_auth_file,
    grok_credits_config,
    jwt_with_raw_payload,
    make_jwt,
    write_claude_account_file,
)
from usagedash.config import ClaudeSourceConfig, SourceMode
from usagedash.credentials import KeychainItem
from usagedash.errors import (
    CredentialExpiredError,
    CredentialUnavailableError,
    ProviderRequestError,
    ProviderSchemaError,
)
from usagedash.models import AccountIdentity, Provider, QuotaReading, QuotaWindow
from usagedash.providers import claude, codex, grok
from usagedash.providers.base import (
    NoQuotaDataError,
    duration_key_and_label,
    ensure_same_login,
    with_unique_keys,
)
from usagedash.providers.claude import KEYCHAIN_ACCOUNT_FALLBACK, ClaudeSource, keychain_account
from usagedash.providers.codex import CodexSource
from usagedash.providers.grok import CreditsConfig, GrokSource


class RecordingTransport:
    """A mock transport that answers by URL path and remembers every request it saw."""

    def __init__(self, responses: dict[str, httpx2.Response]):
        self.responses = responses
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.responses[request.url.path]

    def client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self))

    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]


@pytest.fixture
def no_requests_client() -> Iterator[httpx2.Client]:
    """A client for tests that must stay off the network: any request fails the test, and it is closed after."""

    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    with httpx2.Client(transport=httpx2.MockTransport(refuse)) as client:
        yield client


def json_response(payload: Any, status: int = 200, headers: dict[str, str] | None = None) -> httpx2.Response:
    return httpx2.Response(status, json=payload, headers=headers)


class TestDurationNaming:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (FIVE_HOURS, ("five_hour", "5-hour")),
            (WEEK, ("weekly", "Weekly")),
            (86400, ("daily", "Daily")),
            (3 * 3600, ("window_3h", "3-hour")),
            (30 * 86400, ("window_30d", "30-day")),
            (90, ("window_90s", "90-second")),
            (0, ("window_0s", "0-second")),
            (1, ("window_1s", "1-second")),
            (3600, ("window_1h", "1-hour")),
            (None, ("window", "Limit")),
        ],
    )
    def test_names_a_window_by_its_length(self, seconds: int | None, expected: tuple[str, str]) -> None:
        assert duration_key_and_label(seconds) == expected


class TestUniqueWindowKeys:
    def keys(self, *keys: str) -> list[tuple[str, str]]:
        windows = [QuotaWindow(key, key.title(), 1, None, None) for key in keys]
        return [(window.key, window.label) for window in with_unique_keys(windows)]

    def test_distinct_keys_are_left_alone(self) -> None:
        assert self.keys("weekly", "five_hour") == [("weekly", "Weekly"), ("five_hour", "Five_Hour")]

    def test_each_repeat_takes_the_next_number(self) -> None:
        assert self.keys("pool", "pool", "pool") == [("pool", "Pool"), ("pool#2", "Pool (2)"), ("pool#3", "Pool (3)")]

    def test_a_numbered_key_from_the_provider_is_kept_wherever_it_appears(self) -> None:
        assert [key for key, _ in self.keys("pool#2", "pool", "pool")] == ["pool#2", "pool", "pool#3"]
        assert [key for key, _ in self.keys("pool", "pool", "pool#2", "pool#3")] == [
            "pool",
            "pool#4",
            "pool#2",
            "pool#3",
        ]

    def test_a_repeated_numbered_key_is_itself_numbered(self) -> None:
        assert [key for key, _ in self.keys("pool#2", "pool#2")] == ["pool#2", "pool#2#2"]

    @pytest.mark.parametrize(
        "keys",
        [
            ("a", "a", "a#2", "a#2", "a#3"),
            ("a#2#2", "a#2", "a#2", "a", "a"),
            ("x", "x", "x", "x#2", "x#3", "x#4"),
        ],
    )
    def test_the_result_never_repeats_a_key_and_can_be_stored_as_a_reading(
        self, keys: tuple[str, ...], now: datetime
    ) -> None:
        unique = with_unique_keys([QuotaWindow(key, key, 1, None, None) for key in keys])

        assert len({window.key for window in unique}) == len(keys)
        reading = QuotaReading(AccountIdentity(Provider.CLAUDE, "org-a", "Account A"), unique, now, "api")
        assert reading.windows == unique


NUMBERS_NO_PERCENTAGE_CAN_BE = [-1, -0.001, float("inf"), float("-inf"), float("nan"), "Infinity", "NaN", "1e999"]
TIMES_NO_DATETIME_CAN_HOLD = [-1, 253_402_300_800, 1e300, float("inf"), float("nan")]


class TestImpossibleNumbersAreRefusedAtTheBoundary:
    """Python's JSON parser accepts Infinity and NaN, and a stored one would break every later comparison."""

    @pytest.mark.parametrize("percent", NUMBERS_NO_PERCENTAGE_CAN_BE)
    def test_a_claude_limit_percentage(self, percent: object) -> None:
        with pytest.raises(ProviderSchemaError, match=r"limits\.0\.percent"):
            claude.parse_usage({"limits": [{"kind": "session", "percent": percent}]})

    @pytest.mark.parametrize("percent", NUMBERS_NO_PERCENTAGE_CAN_BE)
    def test_a_claude_legacy_utilization(self, percent: object) -> None:
        with pytest.raises(ProviderSchemaError, match=r"five_hour\.utilization"):
            claude.parse_usage({"five_hour": {"utilization": percent}})

    @pytest.mark.parametrize("percent", NUMBERS_NO_PERCENTAGE_CAN_BE)
    def test_a_codex_window_percentage(self, now: datetime, percent: object) -> None:
        payload = codex_usage_payload(now)
        payload["rate_limit"]["primary_window"]["used_percent"] = percent
        with pytest.raises(ProviderSchemaError, match=r"rate_limit\.primary_window\.used_percent"):
            codex.parse_usage(payload)

    @pytest.mark.parametrize("reset_at", TIMES_NO_DATETIME_CAN_HOLD)
    def test_a_codex_reset_time(self, now: datetime, reset_at: float) -> None:
        payload = codex_usage_payload(now)
        payload["rate_limit"]["primary_window"]["reset_at"] = reset_at
        with pytest.raises(ProviderSchemaError, match=r"rate_limit\.primary_window\.reset_at"):
            codex.parse_usage(payload)

    @pytest.mark.parametrize("seconds", [-WEEK, -1, 315_360_001, 10**400])
    def test_a_codex_window_length_outside_zero_to_ten_years(self, now: datetime, seconds: int) -> None:
        payload = codex_usage_payload(now, primary_seconds=seconds)
        with pytest.raises(ProviderSchemaError, match=r"rate_limit\.primary_window\.limit_window_seconds"):
            codex.parse_usage(payload)

    @pytest.mark.parametrize("seconds", [0, 315_360_000])
    def test_the_edges_of_the_codex_window_length_range_are_accepted(self, now: datetime, seconds: int) -> None:
        _, windows, _ = codex.parse_usage(codex_usage_payload(now, primary_seconds=seconds))

        assert windows[0].window_seconds == seconds

    @pytest.mark.parametrize("balance", [float("inf"), float("nan"), "Infinity"])
    def test_a_codex_credit_balance(self, now: datetime, balance: object) -> None:
        payload = codex_usage_payload(now)
        payload["credits"]["balance"] = balance
        with pytest.raises(ProviderSchemaError, match=r"credits\.balance"):
            codex.parse_usage(payload)

    @pytest.mark.parametrize("percent", NUMBERS_NO_PERCENTAGE_CAN_BE)
    def test_a_grok_credit_percentage(self, now: datetime, percent: object) -> None:
        with pytest.raises(ValidationError, match="creditUsagePercent"):
            CreditsConfig.model_validate(grok_credits_config(now) | {"creditUsagePercent": percent})

    @pytest.mark.parametrize("percent", [0, 0.0, 99.999, 100, 250.5])
    def test_zero_and_anything_above_it_is_a_valid_percentage_including_an_overrun(
        self, now: datetime, percent: float
    ) -> None:
        windows, _ = claude.parse_usage({"limits": [{"kind": "session", "percent": percent}]})
        _, codex_windows, _ = codex.parse_usage(codex_usage_payload(now, primary_percent=percent))
        config = CreditsConfig.model_validate(grok_credits_config(now, percent=percent))

        assert windows[0].used_percent == codex_windows[0].used_percent == config.credit_usage_percent == percent

    @pytest.mark.parametrize("dollars", [float("inf"), float("nan")])
    def test_a_claude_credit_pool_with_a_non_finite_amount_is_not_shown(self, now: datetime, dollars: float) -> None:
        payload = claude_usage_payload(now)
        payload["promo_pool"]["used_dollars"] = dollars

        _, facts = claude.parse_usage(payload)

        assert [fact.label for fact in facts] == ["Extra usage"]


class TestClaudeUsageParsing:
    def test_maps_each_reported_limit_to_a_window_with_minute_rounded_resets(self, now: datetime) -> None:
        windows, _ = claude.parse_usage(claude_usage_payload(now))

        assert [(w.key, w.label, w.used_percent, w.window_seconds) for w in windows] == [
            ("five_hour", "5-hour session", 12, FIVE_HOURS),
            ("weekly", "Weekly · all models", 52, WEEK),
            ("weekly:fable", "Weekly · Fable", 42, WEEK),
        ]
        assert windows[0].resets_at == now + timedelta(hours=4, minutes=1)

    def test_falls_back_to_the_fixed_fields_and_model_scoped_limits_when_no_limits_list_is_sent(
        self, now: datetime
    ) -> None:
        """Most answers on 2026-10-04 had no list; read from the fixed fields alone, Fable's limit was lost."""
        payload = claude_usage_payload(now)
        with_list, _ = claude.parse_usage(payload)
        del payload["limits"]

        windows, _ = claude.parse_usage(payload)

        assert [(w.key, w.label, w.used_percent, w.window_seconds, w.blocking) for w in windows] == [
            ("five_hour", "5-hour session", 12, FIVE_HOURS, True),
            ("weekly", "Weekly · all models", 52, WEEK, True),
            ("weekly:fable", "Weekly · Fable", 42, WEEK, True),
        ]
        assert [(w.key, w.label, w.resets_at) for w in windows] == [(w.key, w.label, w.resets_at) for w in with_list]

    def test_a_model_scoped_limit_is_not_read_beside_the_limits_list_that_also_gives_it(self, now: datetime) -> None:
        windows, _ = claude.parse_usage(claude_usage_payload(now))

        assert [w.key for w in windows] == ["five_hour", "weekly", "weekly:fable"]

    def test_a_model_scoped_limit_a_fixed_field_already_gave_is_not_read_twice(self, now: datetime) -> None:
        payload = claude_usage_payload(now)
        del payload["limits"]
        payload["seven_day_opus"] = {"utilization": 30, "resets_at": None}
        payload["model_scoped"] = [{"display_name": "Opus", "utilization": 30}, {"utilization": 7}]

        windows, _ = claude.parse_usage(payload)

        assert [(w.key, w.label, w.used_percent, w.window_seconds) for w in windows] == [
            ("five_hour", "5-hour session", 12, FIVE_HOURS),
            ("weekly", "Weekly · all models", 52, WEEK),
            ("weekly:opus", "Weekly · Opus", 30, WEEK),
            ("weekly:scoped", "Weekly · scoped", 7, WEEK),
        ]

    def test_a_model_scoped_limit_differing_from_the_fixed_field_of_its_key_is_kept(self, now: datetime) -> None:
        payload = claude_usage_payload(now)
        del payload["limits"]
        payload["seven_day_opus"] = {"utilization": 30, "resets_at": None}
        payload["model_scoped"] = [{"display_name": "Opus", "utilization": 100}]

        windows, _ = claude.parse_usage(payload)

        assert [(w.key, w.used_percent) for w in windows][2:] == [("weekly:opus", 30), ("weekly:opus#2", 100)]

    @pytest.mark.parametrize(
        ("names", "keys"),
        [
            pytest.param(["Fable", "Fable"], ["weekly:fable", "weekly:fable#2"], id="one name twice"),
            pytest.param([None, None], ["weekly:scoped", "weekly:scoped#2"], id="two without a name"),
            pytest.param(["Fable", "fable"], ["weekly:fable", "weekly:fable#2"], id="two names of one slug"),
        ],
    )
    def test_model_scoped_limits_with_one_key_are_all_kept(
        self, now: datetime, names: list[str | None], keys: list[str]
    ) -> None:
        """Kept by key alone, the second was dropped, and it may be the one that is full."""
        payload = claude_usage_payload(now)
        del payload["limits"]
        payload["model_scoped"] = [
            {"display_name": name, "utilization": percent} for name, percent in zip(names, (10, 100), strict=True)
        ]

        windows, _ = claude.parse_usage(payload)

        assert [(w.key, w.used_percent, w.blocking) for w in windows][2:] == [
            (keys[0], 10, True),
            (keys[1], 100, True),
        ]

    @pytest.mark.parametrize("entry", [{"display_name": "Fable", "utilization": "full"}, {"display_name": "Fable"}])
    def test_a_model_scoped_limit_that_cannot_be_read_is_a_schema_error(
        self, now: datetime, entry: dict[str, object]
    ) -> None:
        """A limit is what the reading is for; left out, an account out of quota on one model reads as having room."""
        payload = claude_usage_payload(now)
        del payload["limits"]
        payload["model_scoped"] = [entry]

        with pytest.raises(ProviderSchemaError, match="model_scoped"):
            claude.parse_usage(payload)

    @pytest.mark.parametrize("field_name", ["five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet"])
    @pytest.mark.parametrize(
        "window",
        [
            pytest.param({}, id="empty"),
            pytest.param({"resets_at": "2026-10-03T18:00:00Z"}, id="a reset and no percentage"),
            pytest.param({"utilization": None}, id="a null percentage"),
            pytest.param({"used_percent": 100}, id="a percentage under another name"),
        ],
    )
    def test_a_populated_fixed_field_without_a_usable_percentage_is_not_dropped(
        self, field_name: str, window: dict[str, object]
    ) -> None:
        """Left out, a session limit at 100 percent reads as the weekly limit alone."""
        sibling = "seven_day" if field_name == "five_hour" else "five_hour"
        payload = {field_name: window, sibling: {"utilization": 10}}

        with pytest.raises(ProviderSchemaError, match=field_name):
            claude.parse_usage(payload)

    @pytest.mark.parametrize("five_hour", [None, "omit"], ids=["null", "absent"])
    def test_an_absent_fixed_field_is_not_a_limit_and_the_other_one_is_read(self, five_hour: object) -> None:
        payload: dict[str, object] = {"seven_day": {"utilization": 10}}
        if five_hour is None:
            payload["five_hour"] = None

        windows, _ = claude.parse_usage(payload)

        assert [(window.key, window.used_percent) for window in windows] == [("weekly", 10)]

    def test_an_unknown_limit_kind_is_still_shown(self, now: datetime) -> None:
        payload = {
            "limits": [
                {"kind": "daily_burst", "percent": 5, "resets_at": None, "scope": None},
                {"kind": "daily_burst", "percent": 9, "scope": {"model": {"display_name": "Opus"}}},
            ]
        }

        windows, _ = claude.parse_usage(payload)

        assert [(w.key, w.label, w.window_seconds) for w in windows] == [
            ("daily_burst", "Daily burst", None),
            ("daily_burst:opus", "Daily burst · Opus", None),
        ]

    def test_two_limits_the_provider_did_not_distinguish_get_distinct_keys(self) -> None:
        payload = {"limits": [{"kind": "weekly_scoped", "percent": 1}, {"kind": "weekly_scoped", "percent": 2}]}

        windows, _ = claude.parse_usage(payload)

        assert [w.key for w in windows] == ["weekly:scoped", "weekly:scoped#2"]
        assert [w.used_percent for w in windows] == [1, 2]

    def test_a_repeated_limit_never_takes_a_key_the_provider_itself_used(self) -> None:
        payload = {
            "limits": [
                {"kind": "daily_burst", "percent": 10},
                {"kind": "daily_burst", "percent": 20},
                {"kind": "daily_burst#2", "percent": 30},
            ]
        }

        windows, _ = claude.parse_usage(payload)

        assert [(w.key, w.used_percent) for w in windows] == [
            ("daily_burst", 10),
            ("daily_burst#3", 20),
            ("daily_burst#2", 30),
        ]
        assert windows[1].label == "Daily burst (3)"

    @pytest.mark.parametrize("payload", [{}, {"limits": []}, {"five_hour": None, "unrelated": 1}])
    def test_a_payload_with_no_recognizable_limit_is_a_schema_error_not_an_empty_success(
        self, payload: dict[str, Any]
    ) -> None:
        with pytest.raises(ProviderSchemaError, match="no recognizable limits"):
            claude.parse_usage(payload)

    def test_a_limit_with_the_wrong_type_is_a_schema_error_naming_the_field(self) -> None:
        with pytest.raises(ProviderSchemaError, match=r"limits\.0\.percent"):
            claude.parse_usage({"limits": [{"kind": "session", "percent": "lots"}]})

    def test_a_reset_time_without_an_offset_is_rejected(self) -> None:
        with pytest.raises(ProviderSchemaError, match="resets_at"):
            claude.parse_usage({"limits": [{"kind": "session", "percent": 1, "resets_at": "2026-10-03T18:00:00"}]})

    def test_reports_disabled_extra_usage_and_dollar_pools_as_facts(self, now: datetime) -> None:
        _, facts = claude.parse_usage(claude_usage_payload(now))

        assert [(fact.label, fact.value) for fact in facts] == [
            ("Extra usage", "Off"),
            ("Credit pool · promo_pool", "$12.50 of $250.00 used"),
        ]
        assert facts[1].resets_at == now + timedelta(days=30)

    def test_reports_enabled_extra_usage_in_currency_units(self, now: datetime) -> None:
        payload = claude_usage_payload(now)
        payload["extra_usage"] = {"is_enabled": True, "monthly_limit": 10000, "used_credits": 2550, "decimal_places": 2}

        _, facts = claude.parse_usage(payload)

        assert (facts[0].label, facts[0].value) == ("Extra usage", "$25.50 of $100.00 used")

    @pytest.mark.parametrize(
        "extra_usage",
        [
            {"is_enabled": True},
            {"is_enabled": True, "monthly_limit": 10000, "decimal_places": 2},
            {"is_enabled": True, "used_credits": 2550, "decimal_places": 2},
            {"is_enabled": True, "monthly_limit": None, "used_credits": None, "decimal_places": 2},
            {"is_enabled": True, "monthly_limit": 10000, "used_credits": 2550},
            {"is_enabled": True, "monthly_limit": 10000, "used_credits": 2550, "decimal_places": None},
        ],
    )
    def test_enabled_extra_usage_that_does_not_state_both_amounts_and_their_scale_is_left_out_never_shown_as_off(
        self, now: datetime, extra_usage: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        """ "Off" would tell the user that paid overage cannot happen, when the setting says it can."""
        payload = claude_usage_payload(now)
        payload["extra_usage"] = extra_usage

        with caplog.at_level(logging.WARNING, logger="usagedash.providers.claude"):
            windows, facts = claude.parse_usage(payload)

        assert [w.key for w in windows] == ["five_hour", "weekly", "weekly:fable"]
        assert [fact.label for fact in facts] == ["Credit pool · promo_pool"]
        assert "enabled but does not state its amounts and their scale" in caplog.text

    @pytest.mark.parametrize(
        "extra_usage",
        [{}, {"monthly_limit": 10000, "used_credits": 2550, "decimal_places": 2}, {"is_enabled": None}],
    )
    def test_extra_usage_that_does_not_say_whether_it_is_on_is_left_out_and_never_shown_as_off(
        self, now: datetime, extra_usage: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing flag is not "off": the payload has said nothing about what the user can be charged."""
        payload = claude_usage_payload(now)
        payload["extra_usage"] = extra_usage

        with caplog.at_level(logging.WARNING, logger="usagedash.providers.claude"):
            windows, facts = claude.parse_usage(payload)

        assert [w.key for w in windows] == ["five_hour", "weekly", "weekly:fable"]
        assert [fact.label for fact in facts] == ["Credit pool · promo_pool"]
        assert "Leaving out Claude extra usage with an unexpected shape: is_enabled" in caplog.text

    @pytest.mark.parametrize(
        "extra_usage",
        [{"is_enabled": False}, {"is_enabled": False, "monthly_limit": 10000, "used_credits": 2550}],
    )
    def test_extra_usage_that_says_it_is_off_is_shown_as_off_whatever_amounts_it_states(
        self, now: datetime, extra_usage: dict[str, Any]
    ) -> None:
        payload = claude_usage_payload(now)
        payload["extra_usage"] = extra_usage

        _, facts = claude.parse_usage(payload)

        assert (facts[0].label, facts[0].value) == ("Extra usage", "Off")


class TestClaudeIdentity:
    @pytest.mark.parametrize(
        ("tier", "organization_type", "subscription", "expected"),
        [
            ("default_claude_max_20x", "claude_max", "max", "Max 20x"),
            (None, "claude_team", None, "Team"),
            (None, None, "pro", "Pro"),
            ("default_claude_ultra", None, None, "Ultra"),
            (None, None, None, None),
        ],
    )
    def test_names_the_plan_from_the_most_specific_identifier(
        self, tier: str | None, organization_type: str | None, subscription: str | None, expected: str | None
    ) -> None:
        assert claude.plan_name(tier, organization_type, subscription) == expected

    def test_a_personal_subscription_is_labeled_personal_and_keyed_on_the_organization(self) -> None:
        identity = claude.build_identity(
            organization_uuid="org-a",
            organization_name=f"{PERSONA_EMAIL}'s Organization",
            organization_type=None,
            email=PERSONA_EMAIL,
            plan="Max 20x",
        )
        assert (identity.external_id, identity.default_label) == ("org-a", "Personal")

    def test_a_team_organization_keeps_its_own_name(self) -> None:
        identity = claude.build_identity(
            organization_uuid="org-b",
            organization_name="Synthetic Co",
            organization_type="claude_team",
            email=PERSONA_EMAIL,
            plan="Team",
        )
        assert identity.default_label == "Synthetic Co"


class TestClaudeSource:
    @pytest.fixture
    def config(self, tmp_path: Path) -> ClaudeSourceConfig:
        account_file = tmp_path / ".claude.json"
        write_claude_account_file(account_file)
        return ClaudeSourceConfig(
            config_dir=tmp_path / ".claude",
            account_file=account_file,
            mode=SourceMode.DIRECT,
            keychain_service="Synthetic-credentials",
        )

    def transport(self, now: datetime, **overrides: httpx2.Response) -> RecordingTransport:
        responses = {
            "/api/oauth/profile": json_response(claude_profile_payload()),
            "/api/oauth/usage": json_response(claude_usage_payload(now)),
        }
        responses.update({f"/api/oauth/{name}": response for name, response in overrides.items()})
        return RecordingTransport(responses)

    def test_reads_quota_for_the_account_the_token_belongs_to(self, config: ClaudeSourceConfig, now: datetime) -> None:
        transport = self.transport(now)
        source = ClaudeSource(
            config, transport.client(), lambda _: claude_keychain_blob(expires_at=now + timedelta(hours=1))
        )

        reading = source.read_quota(now)

        assert reading.identity.external_id == "org-a"
        assert reading.identity.plan == "Max 20x"
        assert reading.observed_at == now
        assert [w.key for w in reading.windows] == ["five_hour", "weekly", "weekly:fable"]
        usage_request = transport.requests[-1]
        assert usage_request.headers["authorization"] == "Bearer synthetic-claude-token"
        assert usage_request.headers["anthropic-beta"] == "oauth-2025-04-20"

    def test_asks_the_keychain_for_the_item_claude_code_reads_by_service_and_account(
        self, config: ClaudeSourceConfig, now: datetime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression lock: by service alone, the keychain answered with another account's item (2026-10-08)."""
        monkeypatch.setenv("USER", "unknown")
        asked: list[KeychainItem] = []

        def reader(item: KeychainItem) -> str:
            asked.append(item)
            return claude_keychain_blob()

        ClaudeSource(config, self.transport(now).client(), reader).read_quota(now)

        assert asked == [KeychainItem(service="Synthetic-credentials", account=keychain_account())]
        assert asked[0].account != "unknown"

    def test_the_provider_s_statement_of_identity_outranks_the_local_account_file(
        self, config: ClaudeSourceConfig, now: datetime
    ) -> None:
        transport = self.transport(
            now, profile=json_response(claude_profile_payload("org-b", "Synthetic Co", "claude_team", "claude_team"))
        )
        source = ClaudeSource(config, transport.client(), lambda _: claude_keychain_blob())

        assert source.current_identity().external_id == "org-a"
        assert source.read_quota(now).identity.external_id == "org-b"

    def test_asks_who_the_token_belongs_to_once_per_token(self, config: ClaudeSourceConfig, now: datetime) -> None:
        transport = self.transport(now)
        blobs = iter(
            [claude_keychain_blob("token-1"), claude_keychain_blob("token-1"), claude_keychain_blob("token-2")]
        )
        source = ClaudeSource(config, transport.client(), lambda _: next(blobs))

        for _ in range(3):
            source.read_quota(now)

        assert transport.paths().count("/api/oauth/profile") == 2
        assert transport.paths().count("/api/oauth/usage") == 3

    def test_an_expired_token_is_never_sent(self, config: ClaudeSourceConfig, now: datetime) -> None:
        transport = self.transport(now)
        source = ClaudeSource(config, transport.client(), lambda _: claude_keychain_blob(expires_at=now))

        with pytest.raises(CredentialExpiredError, match="refreshes it on its next run"):
            source.read_quota(now)

        assert transport.requests == []

    def test_a_token_expiring_one_second_from_now_is_still_used(
        self, config: ClaudeSourceConfig, now: datetime
    ) -> None:
        source = ClaudeSource(
            config,
            self.transport(now).client(),
            lambda _: claude_keychain_blob(expires_at=now + timedelta(seconds=1)),
        )
        assert source.read_quota(now).windows

    @pytest.mark.parametrize("expires_at", [-1, 1e300, "Infinity", "NaN"])
    def test_a_credential_with_an_impossible_expiry_is_not_used(
        self, config: ClaudeSourceConfig, now: datetime, expires_at: object
    ) -> None:
        blob = json.loads(claude_keychain_blob())
        blob["claudeAiOauth"]["expiresAt"] = expires_at
        transport = self.transport(now)
        source = ClaudeSource(config, transport.client(), lambda _: json.dumps(blob))

        with pytest.raises(CredentialUnavailableError, match="no usable OAuth credential"):
            source.read_quota(now)

        assert transport.requests == []

    @pytest.mark.parametrize("status", [401, 403])
    def test_a_rejected_token_is_reported_as_expired(
        self, config: ClaudeSourceConfig, now: datetime, status: int
    ) -> None:
        transport = self.transport(now, usage=json_response({"error": "nope"}, status))
        source = ClaudeSource(config, transport.client(), lambda _: claude_keychain_blob())

        with pytest.raises(CredentialExpiredError, match=f"HTTP {status}"):
            source.read_quota(now)

    def test_a_rate_limit_response_carries_the_wait_the_provider_asked_for(
        self, config: ClaudeSourceConfig, now: datetime
    ) -> None:
        transport = self.transport(now, usage=json_response({}, 429, {"Retry-After": "120"}))
        source = ClaudeSource(config, transport.client(), lambda _: claude_keychain_blob())

        with pytest.raises(ProviderRequestError) as raised:
            source.read_quota(now)

        assert raised.value.status_code == 429
        assert raised.value.retry_after_seconds == 120

    def test_a_server_error_is_a_request_error_with_its_status(self, config: ClaudeSourceConfig, now: datetime) -> None:
        transport = self.transport(now, usage=json_response({}, 503))
        source = ClaudeSource(config, transport.client(), lambda _: claude_keychain_blob())

        with pytest.raises(ProviderRequestError) as raised:
            source.read_quota(now)

        assert raised.value.status_code == 503
        assert raised.value.retry_after_seconds is None

    def test_a_network_failure_is_a_request_error_that_does_not_leak_the_token(
        self, config: ClaudeSourceConfig, now: datetime
    ) -> None:
        def refuse(request: httpx2.Request) -> httpx2.Response:
            """Fail the way a real transport can: with a message quoting the header it was sending."""
            raise httpx2.LocalProtocolError(f"Illegal header value {request.headers['authorization']!r}")

        client = httpx2.Client(transport=httpx2.MockTransport(refuse))
        source = ClaudeSource(config, client, lambda _: claude_keychain_blob())

        with pytest.raises(ProviderRequestError) as raised:
            source.read_quota(now)

        assert str(raised.value) == "Claude profile API request failed: LocalProtocolError"
        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__
        assert "synthetic-claude-token" not in "".join(traceback.format_exception(raised.value))

    def test_a_logged_out_keychain_item_means_no_credential(
        self, no_requests_client: httpx2.Client, config: ClaudeSourceConfig, now: datetime
    ) -> None:
        source = ClaudeSource(config, no_requests_client, lambda _: json.dumps({"mcpOAuth": {}}))

        with pytest.raises(CredentialUnavailableError, match="not logged in"):
            source.read_quota(now)

    def test_an_empty_access_token_means_no_credential(
        self, no_requests_client: httpx2.Client, config: ClaudeSourceConfig, now: datetime
    ) -> None:
        source = ClaudeSource(config, no_requests_client, lambda _: claude_keychain_blob(access_token=""))

        with pytest.raises(CredentialUnavailableError, match="no usable OAuth credential"):
            source.read_quota(now)

    def test_a_keychain_item_that_fails_validation_is_not_kept_by_the_error_raised_for_it(
        self, no_requests_client: httpx2.Client, config: ClaudeSourceConfig, now: datetime
    ) -> None:
        """The validation error holds the whole item it was given, refresh token included."""
        source = ClaudeSource(config, no_requests_client, lambda _: claude_keychain_blob(access_token=""))

        with pytest.raises(CredentialUnavailableError) as raised:
            source.read_quota(now)

        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__

    def test_local_identity_comes_from_the_account_file_with_the_plan_from_the_keychain(
        self, no_requests_client: httpx2.Client, config: ClaudeSourceConfig
    ) -> None:
        source = ClaudeSource(config, no_requests_client, lambda _: claude_keychain_blob())

        identity = source.current_identity()

        assert identity is not None
        assert (identity.external_id, identity.default_label, identity.plan) == ("org-a", "Personal", "Max 20x")

    def test_local_identity_survives_an_unreadable_keychain(
        self, no_requests_client: httpx2.Client, config: ClaudeSourceConfig
    ) -> None:
        def locked(_: KeychainItem) -> str:
            raise CredentialUnavailableError("keychain locked")

        identity = ClaudeSource(config, no_requests_client, locked).current_identity()

        assert identity is not None
        assert identity.plan is None

    def test_no_account_file_or_no_login_in_it_means_nobody_is_logged_in(
        self, no_requests_client: httpx2.Client, config: ClaudeSourceConfig
    ) -> None:
        source = ClaudeSource(config, no_requests_client, lambda _: claude_keychain_blob())
        config.account_file.write_text(json.dumps({"userID": "x"}))
        assert source.current_identity() is None
        config.account_file.unlink()
        assert source.current_identity() is None

    def test_an_account_file_with_the_wrong_shape_is_a_schema_error(
        self, no_requests_client: httpx2.Client, config: ClaudeSourceConfig
    ) -> None:
        config.account_file.write_text(json.dumps({"oauthAccount": {"organizationUuid": ""}}))
        with pytest.raises(ProviderSchemaError, match="unexpected shape"):
            ClaudeSource(config, no_requests_client, lambda _: claude_keychain_blob()).current_identity()


class TestClaudeKeychainAccount:
    """The account name Claude Code files its keychain items under, as this dashboard must name it."""

    @staticmethod
    def user_database_name(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        monkeypatch.setattr(claude.pwd, "getpwuid", lambda _uid: types.SimpleNamespace(pw_name=name))

    def test_is_the_user_database_name_whatever_the_shell_s_user_says(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.user_database_name(monkeypatch, "synthetic.user-1_A")
        monkeypatch.setenv("USER", "unknown")

        assert keychain_account() == "synthetic.user-1_A"

    @pytest.mark.parametrize("name", ["a", "Z", "0", "a.b_c-d"])
    def test_a_name_inside_claude_code_s_character_set_is_used_as_it_is(
        self, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        self.user_database_name(monkeypatch, name)

        assert keychain_account() == name

    @pytest.mark.parametrize("name", ["", "first last", "jürgen", "a/b", "name\n"])
    def test_a_name_claude_code_would_not_use_becomes_its_fallback(
        self, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        self.user_database_name(monkeypatch, name)

        assert keychain_account() == KEYCHAIN_ACCOUNT_FALLBACK == "claude-code-user"

    def test_an_account_missing_from_the_user_database_uses_the_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(uid: int) -> object:
            raise KeyError(uid)

        monkeypatch.setattr(claude.pwd, "getpwuid", missing)

        assert keychain_account() == KEYCHAIN_ACCOUNT_FALLBACK


class TestCodexUsageParsing:
    def test_names_windows_by_duration_and_reports_credits(self, now: datetime) -> None:
        secondary = {"used_percent": 61, "limit_window_seconds": FIVE_HOURS, "reset_at": int(now.timestamp()) + 600}

        _, windows, facts = codex.parse_usage(codex_usage_payload(now, secondary=secondary))

        assert [(w.key, w.label, w.used_percent) for w in windows] == [
            ("weekly", "Weekly", 38),
            ("five_hour", "5-hour", 61),
        ]
        assert windows[1].resets_at == now + timedelta(minutes=10)
        assert [(fact.label, fact.value) for fact in facts] == [("Credits", "61,180 remaining")]

    def test_a_missing_secondary_window_is_simply_absent(self, now: datetime) -> None:
        _, windows, _ = codex.parse_usage(codex_usage_payload(now))
        assert [w.key for w in windows] == ["weekly"]

    def test_code_review_and_extra_limits_are_prefixed_so_they_cannot_collide(self, now: datetime) -> None:
        payload = codex_usage_payload(now)
        window = payload["rate_limit"]["primary_window"]
        payload["code_review_rate_limit"] = {"primary_window": window}
        payload["additional_rate_limits"] = [
            {
                "limit_name": "Deep research",
                "metered_feature": "research_meter",
                "rate_limit": {"primary_window": window},
            },
            {"unexpected": "shape"},
            "not even an object",
        ]

        _, windows, _ = codex.parse_usage(payload)

        assert [w.key for w in windows] == ["weekly", "code_review:weekly", "research_meter:weekly"]
        assert windows[2].label == "Deep research · Weekly"

    def test_only_the_main_limit_can_put_the_account_out_of_quota(self, now: datetime) -> None:
        payload = codex_usage_payload(now)
        window = payload["rate_limit"]["primary_window"]
        payload["code_review_rate_limit"] = {"primary_window": {**window, "used_percent": 100}}
        payload["additional_rate_limits"] = [
            {
                "limit_name": "Deep research",
                "metered_feature": "deep_research",
                "rate_limit": {"primary_window": {**window, "used_percent": 100}},
            }
        ]

        _, windows, _ = codex.parse_usage(payload)

        assert {w.key: w.blocking for w in windows} == {
            "weekly": True,
            "code_review:weekly": False,
            "deep_research:weekly": False,
        }

    def test_reports_unlimited_credits(self, now: datetime) -> None:
        payload = codex_usage_payload(now)
        payload["credits"] = {"unlimited": True}

        _, _, facts = codex.parse_usage(payload)

        assert [(fact.label, fact.value) for fact in facts] == [("Credits", "Unlimited")]

    @pytest.mark.parametrize(
        "flags",
        [{"limit_reached": True}, {"allowed": False}, {"limit_reached": True, "allowed": False}],
    )
    def test_a_limit_codex_reports_as_reached_puts_the_account_out_of_quota_though_its_windows_show_room(
        self, now: datetime, flags: dict[str, bool]
    ) -> None:
        payload = codex_usage_payload(now, primary_percent=38)
        payload["rate_limit"].update(flags)

        _, windows, facts = codex.parse_usage(payload)

        assert [(w.key, w.used_percent, w.blocking) for w in windows] == [
            ("weekly", 38, True),
            ("reported_limit", 100, True),
        ]
        assert (windows[1].label, windows[1].resets_at, windows[1].window_seconds) == (
            "Limit reached (reported by Codex)",
            None,
            None,
        )
        assert all(fact.label != "Status" for fact in facts)

    def test_a_reported_limit_adds_nothing_when_a_listed_window_is_already_full(self, now: datetime) -> None:
        payload = codex_usage_payload(now, primary_percent=100)
        payload["rate_limit"]["limit_reached"] = True

        _, windows, _ = codex.parse_usage(payload)

        assert [w.key for w in windows] == ["weekly"]

    @pytest.mark.parametrize(
        "flags",
        [{"limit_reached": False, "allowed": True}, {"limit_reached": None, "allowed": None}],
    )
    def test_an_account_codex_does_not_report_as_limited_gets_no_extra_window(
        self, now: datetime, flags: dict[str, bool | None]
    ) -> None:
        payload = codex_usage_payload(now)
        payload["rate_limit"].update(flags)

        assert [w.key for w in codex.parse_usage(payload)[1]] == ["weekly"]

    def test_a_rate_limit_that_omits_both_flags_gets_no_extra_window(self, now: datetime) -> None:
        payload = codex_usage_payload(now)
        del payload["rate_limit"]["limit_reached"]
        del payload["rate_limit"]["allowed"]

        assert [w.key for w in codex.parse_usage(payload)[1]] == ["weekly"]

    def test_a_reached_code_review_limit_does_not_put_the_account_out_of_quota(self, now: datetime) -> None:
        payload = codex_usage_payload(now)
        window = payload["rate_limit"]["primary_window"]
        payload["code_review_rate_limit"] = {"primary_window": window, "limit_reached": True, "allowed": False}

        _, windows, _ = codex.parse_usage(payload)

        assert [(w.key, w.used_percent, w.blocking) for w in windows] == [
            ("weekly", 38, True),
            ("code_review:weekly", 38, False),
            ("code_review:reported_limit", 100, False),
        ]

    @pytest.mark.parametrize(
        "flags",
        [{"limit_reached": True}, {"allowed": False}],
    )
    def test_a_reached_feature_pool_is_shown_full_without_blocking_the_account(
        self, now: datetime, flags: dict[str, bool]
    ) -> None:
        """Codex can report a pool reached while every window it lists still shows room."""
        payload = codex_usage_payload(now, primary_percent=38)
        window = payload["rate_limit"]["primary_window"]
        payload["code_review_rate_limit"] = {"primary_window": window, **flags}
        payload["additional_rate_limits"] = [
            {
                "limit_name": "Deep research",
                "metered_feature": "deep_research",
                "rate_limit": {"primary_window": {**window, "used_percent": 10}, **flags},
            }
        ]

        _, windows, facts = codex.parse_usage(payload)

        by_key = {window.key: window for window in windows}
        assert (by_key["code_review:reported_limit"].used_percent, by_key["code_review:reported_limit"].blocking) == (
            100,
            False,
        )
        assert by_key["code_review:reported_limit"].label == "Code review · Limit reached (reported by Codex)"
        assert (
            by_key["code_review:reported_limit"].resets_at,
            by_key["code_review:reported_limit"].window_seconds,
        ) == (None, None)
        research = by_key["deep_research:reported_limit"]
        assert (research.used_percent, research.blocking) == (100, False)
        assert research.label == "Deep research · Limit reached (reported by Codex)"
        assert not any(window.blocking and window.used_percent >= 100 for window in windows)
        assert all(fact.label != "Limits not shown" for fact in facts)

    def test_a_reached_feature_pool_adds_nothing_when_a_listed_window_is_already_full(self, now: datetime) -> None:
        payload = codex_usage_payload(now)
        window = payload["rate_limit"]["primary_window"]
        payload["code_review_rate_limit"] = {
            "primary_window": {**window, "used_percent": 100},
            "limit_reached": True,
        }
        payload["additional_rate_limits"] = [
            {
                "limit_name": "Deep research",
                "metered_feature": "deep_research",
                "rate_limit": {"primary_window": {**window, "used_percent": 100}, "allowed": False},
            }
        ]

        _, windows, _ = codex.parse_usage(payload)

        assert [window.key for window in windows] == ["weekly", "code_review:weekly", "deep_research:weekly"]

    def test_a_reached_feature_pool_with_no_window_is_shown_rather_than_omitted(self, now: datetime) -> None:
        payload = codex_usage_payload(now)
        payload["additional_rate_limits"] = [
            {"metered_feature": "deep_research", "rate_limit": {"limit_reached": True}}
        ]

        _, windows, facts = codex.parse_usage(payload)

        assert [(window.key, window.used_percent, window.blocking) for window in windows] == [
            ("weekly", 38, True),
            ("deep_research:reported_limit", 100, False),
        ]
        assert all(fact.label != "Limits not shown" for fact in facts)

    def test_a_payload_without_any_window_is_a_schema_error(self, now: datetime) -> None:
        payload = codex_usage_payload(now)
        payload["rate_limit"] = None
        with pytest.raises(ProviderSchemaError, match="reported no window of its main limit"):
            codex.parse_usage(payload)

    @pytest.mark.parametrize(
        "main",
        [
            pytest.param(None, id="no main limit"),
            pytest.param({"primary_window": None, "secondary_window": None}, id="a main limit with no window"),
        ],
    )
    def test_code_review_and_extra_windows_cannot_stand_in_for_a_missing_main_limit(
        self, now: datetime, main: object
    ) -> None:
        """None of them blocks, so alone they would show the account with room whatever its main limit says."""
        payload = codex_usage_payload(now)
        window = payload["rate_limit"]["primary_window"]
        payload["rate_limit"] = main
        payload["code_review_rate_limit"] = {"primary_window": window}
        payload["additional_rate_limits"] = [
            {"metered_feature": "deep_research", "rate_limit": {"primary_window": window}}
        ]

        with pytest.raises(ProviderSchemaError, match="reported no window of its main limit"):
            codex.parse_usage(payload)

    def test_a_main_limit_reported_as_reached_with_no_window_is_still_a_main_limit(self, now: datetime) -> None:
        """The window added for a reported limit is the main limit's, and it blocks."""
        payload = codex_usage_payload(now)
        payload["rate_limit"] = {"primary_window": None, "secondary_window": None, "limit_reached": True}

        _, windows, _ = codex.parse_usage(payload)

        assert [(w.key, w.blocking) for w in windows] == [(codex.REPORTED_LIMIT_KEY, True)]

    def test_a_window_missing_its_percentage_is_a_schema_error_naming_the_field(self, now: datetime) -> None:
        payload = codex_usage_payload(now)
        del payload["rate_limit"]["primary_window"]["used_percent"]
        with pytest.raises(ProviderSchemaError, match=r"rate_limit\.primary_window\.used_percent"):
            codex.parse_usage(payload)


class TestCodexSource:
    def transport(self, now: datetime, response: httpx2.Response | None = None) -> RecordingTransport:
        return RecordingTransport({"/backend-api/wham/usage": response or json_response(codex_usage_payload(now))})

    def test_reads_quota_for_the_workspace_named_in_the_auth_file(self, tmp_path: Path, now: datetime) -> None:
        codex_auth_file(tmp_path / "auth.json", access_expires_at=now + timedelta(days=5))
        transport = self.transport(now)

        reading = CodexSource(tmp_path, transport.client()).read_quota(now)

        assert (reading.identity.external_id, reading.identity.email, reading.identity.plan) == (
            "workspace-a",
            PERSONA_EMAIL,
            "Pro",
        )
        request = transport.requests[0]
        assert request.headers["chatgpt-account-id"] == "workspace-a"
        assert request.headers["authorization"].startswith("Bearer ")

    def test_the_workspace_the_provider_answered_for_outranks_the_auth_file(
        self, tmp_path: Path, now: datetime
    ) -> None:
        codex_auth_file(tmp_path / "auth.json")
        transport = self.transport(now, json_response(codex_usage_payload(now, account_id="workspace-b")))

        assert CodexSource(tmp_path, transport.client()).read_quota(now).identity.external_id == "workspace-b"

    def test_an_expired_token_is_never_sent(self, tmp_path: Path, now: datetime) -> None:
        codex_auth_file(tmp_path / "auth.json", access_expires_at=now)
        transport = self.transport(now)

        with pytest.raises(CredentialExpiredError, match="refreshes it on its next run"):
            CodexSource(tmp_path, transport.client()).read_quota(now)

        assert transport.requests == []

    @pytest.mark.parametrize(
        "access_token",
        [
            "opaque-token-not-a-jwt",
            "one.two.three.four",
            make_jwt({"email": "user-a@example.test"}),
            make_jwt({"exp": 1e300}),
            make_jwt({"exp": float("inf")}),
            make_jwt({"exp": 10**400}),
        ],
        ids=["opaque", "four parts", "no exp claim", "far future float", "infinite", "far future integer"],
    )
    def test_a_token_that_states_no_expiry_or_a_future_one_is_sent(
        self, tmp_path: Path, now: datetime, access_token: str
    ) -> None:
        codex_auth_file(tmp_path / "auth.json", access_token=access_token)
        transport = self.transport(now)

        assert CodexSource(tmp_path, transport.client()).read_quota(now).windows
        assert len(transport.requests) == 1

    @pytest.mark.parametrize(
        "access_token",
        [
            make_jwt({"exp": "tomorrow"}),
            make_jwt({"exp": True}),
            make_jwt({"exp": None}),
            make_jwt({"exp": float("nan")}),
            make_jwt({"exp": [1]}),
        ],
        ids=["text", "a flag", "null", "NaN", "a list"],
    )
    def test_a_jwt_whose_expiry_is_not_a_number_is_refused_and_never_sent(
        self, tmp_path: Path, now: datetime, access_token: str
    ) -> None:
        """It may be expired; a guard that cannot read the expiry must not let the token through."""
        codex_auth_file(tmp_path / "auth.json", access_token=access_token)
        transport = self.transport(now)

        with pytest.raises(ProviderSchemaError, match="was not sent: its expiry is not a number"):
            CodexSource(tmp_path, transport.client()).read_quota(now)

        assert transport.requests == []

    @pytest.mark.parametrize(
        "access_token",
        [
            "header.a.signature",
            "header.__4.signature",
            "header.W10.signature",
            "header.eyJleHAiOjQxMDI0NDQ4MDB9!.signature",
            jwt_with_raw_payload('{"exp": -1, "padding": ' + "9" * 4301 + "}"),
            jwt_with_raw_payload('{"exp": -' + "9" * 4301 + "}"),
            jwt_with_raw_payload('{"exp": -1, "nested": ' + "[" * 100_000 + "]" * 100_000 + "}"),
        ],
        ids=[
            "not base64url",
            "not UTF-8",
            "not an object",
            "invalid base64url character after a future expiry",
            "an unrelated integer one past the digit limit",
            "an expiry one past the digit limit",
            "nesting past the recursion limit",
        ],
    )
    def test_a_jwt_whose_claims_cannot_be_decoded_is_refused_and_never_sent(
        self, tmp_path: Path, now: datetime, access_token: str
    ) -> None:
        """Any of these could hide an expired ``exp``; reading them as "expiry unknown" sent the token."""
        codex_auth_file(tmp_path / "auth.json", access_token=access_token)
        transport = self.transport(now)

        with pytest.raises(ProviderSchemaError, match="was not sent: its expiry cannot be read") as raised:
            CodexSource(tmp_path, transport.client()).read_quota(now)

        assert transport.requests == []
        assert access_token not in str(raised.value)

    def test_an_expiry_at_the_digit_limit_is_still_read(self, tmp_path: Path, now: datetime) -> None:
        codex_auth_file(tmp_path / "auth.json", access_token=jwt_with_raw_payload('{"exp": -' + "9" * 4300 + "}"))
        transport = self.transport(now)

        with pytest.raises(CredentialExpiredError, match="refreshes it on its next run"):
            CodexSource(tmp_path, transport.client()).read_quota(now)

        assert transport.requests == []

    @pytest.mark.parametrize("expires", [-(10**400), -1e300, float("-inf"), -5, -1, 0, 1, 1.5])
    def test_a_token_whose_expiry_is_at_or_before_the_start_of_1970_is_expired_and_never_sent(
        self, tmp_path: Path, now: datetime, expires: float
    ) -> None:
        """Such a time cannot be built as a datetime; that must not be read as "expiry unknown"."""
        codex_auth_file(tmp_path / "auth.json", access_token=make_jwt({"exp": expires}))
        transport = self.transport(now)

        with pytest.raises(CredentialExpiredError, match="refreshes it on its next run"):
            CodexSource(tmp_path, transport.client()).read_quota(now)

        assert transport.requests == []

    def test_a_token_expiring_one_second_from_now_is_still_sent_and_one_expiring_now_is_not(
        self, tmp_path: Path, now: datetime
    ) -> None:
        transport = self.transport(now)
        codex_auth_file(tmp_path / "auth.json", access_token=make_jwt({"exp": int(now.timestamp()) + 1}))
        assert CodexSource(tmp_path, transport.client()).read_quota(now).windows

        codex_auth_file(tmp_path / "auth.json", access_token=make_jwt({"exp": int(now.timestamp())}))
        with pytest.raises(CredentialExpiredError):
            CodexSource(tmp_path, transport.client()).read_quota(now)

        assert len(transport.requests) == 1

    def test_local_identity_reads_email_and_plan_from_the_id_token(
        self, no_requests_client: httpx2.Client, tmp_path: Path
    ) -> None:
        codex_auth_file(tmp_path / "auth.json", account_id="workspace-z", plan="team")

        identity = CodexSource(tmp_path, no_requests_client).current_identity()

        assert identity is not None
        assert (identity.external_id, identity.default_label, identity.plan) == ("workspace-z", PERSONA_EMAIL, "Team")

    def test_an_undecodable_id_token_still_yields_the_workspace(
        self, no_requests_client: httpx2.Client, tmp_path: Path
    ) -> None:
        (tmp_path / "auth.json").write_text(
            json.dumps({"tokens": {"access_token": make_jwt({}), "id_token": "garbage", "account_id": "workspace-a"}})
        )

        identity = CodexSource(tmp_path, no_requests_client).current_identity()

        assert identity is not None
        assert (identity.external_id, identity.email, identity.default_label) == ("workspace-a", None, "Codex account")

    def test_no_auth_file_or_an_api_key_login_means_nobody_is_logged_in(
        self, no_requests_client: httpx2.Client, tmp_path: Path, now: datetime
    ) -> None:
        source = CodexSource(tmp_path, no_requests_client)
        assert source.current_identity() is None
        (tmp_path / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": "synthetic", "tokens": None}))
        assert source.current_identity() is None
        with pytest.raises(CredentialUnavailableError, match="not signed in with a ChatGPT account"):
            source.read_quota(now)

    @pytest.mark.parametrize("auth_mode", ["apikey", "something-newer"])
    def test_a_login_in_another_mode_is_nobody_logged_in_though_chatgpt_tokens_remain(
        self, no_requests_client: httpx2.Client, tmp_path: Path, now: datetime, auth_mode: str
    ) -> None:
        """The CLI does not use those tokens; their workspace would be shown as logged in and its quota read."""
        codex_auth_file(tmp_path / "auth.json")
        document = json.loads((tmp_path / "auth.json").read_text())
        document["auth_mode"] = auth_mode
        (tmp_path / "auth.json").write_text(json.dumps(document))
        source = CodexSource(tmp_path, no_requests_client)

        assert source.current_identity() is None
        with pytest.raises(CredentialUnavailableError, match="not signed in with a ChatGPT account"):
            source.read_quota(now)

    def test_a_login_whose_file_does_not_state_its_mode_is_read_from_its_tokens(
        self, no_requests_client: httpx2.Client, tmp_path: Path
    ) -> None:
        codex_auth_file(tmp_path / "auth.json")
        document = json.loads((tmp_path / "auth.json").read_text())
        del document["auth_mode"]
        (tmp_path / "auth.json").write_text(json.dumps(document))

        identity = CodexSource(tmp_path, no_requests_client).current_identity()

        assert identity is not None
        assert identity.external_id == "workspace-a"

    def test_an_auth_file_that_is_not_json_is_a_schema_error(
        self, no_requests_client: httpx2.Client, tmp_path: Path
    ) -> None:
        (tmp_path / "auth.json").write_text("{not json")
        with pytest.raises(ProviderSchemaError, match="not valid JSON"):
            CodexSource(tmp_path, no_requests_client).current_identity()


class TestGrokCreditsConfig:
    @pytest.mark.parametrize(
        ("start", "end"),
        [
            ("2026-10-03T14:00:00Z", "2026-10-03T14:00:00Z"),
            ("2000-01-01T00:00:00Z", "2010-01-01T00:00:01Z"),
            ("0001-01-01T00:00:00Z", "9999-12-31T23:59:59Z"),
        ],
    )
    def test_a_period_with_no_length_or_one_longer_than_ten_years_has_no_usable_length(
        self, now: datetime, start: str, end: str
    ) -> None:
        raw = grok_credits_config(now)
        raw["currentPeriod"].update({"start": start, "end": end})

        windows, _ = grok.windows_and_facts(CreditsConfig.model_validate(raw))

        assert windows[0].window_seconds is None
        assert windows[0].used_percent == 85

    @pytest.mark.parametrize("period", [None, {}, {"type": "USAGE_PERIOD_TYPE_WEEKLY"}])
    def test_a_config_that_names_no_period_ends_still_yields_its_percentage(
        self, now: datetime, period: dict[str, Any] | None
    ) -> None:
        raw = grok_credits_config(now, percent=61)
        raw["currentPeriod"] = period

        windows, _ = grok.windows_and_facts(CreditsConfig.model_validate(raw))

        assert (windows[0].used_percent, windows[0].window_seconds, windows[0].resets_at) == (61, None, None)

    def test_a_period_that_ends_before_it_starts_has_no_length(self, now: datetime) -> None:
        raw = grok_credits_config(now)
        raw["currentPeriod"]["start"], raw["currentPeriod"]["end"] = (
            raw["currentPeriod"]["end"],
            raw["currentPeriod"]["start"],
        )

        windows, _ = grok.windows_and_facts(CreditsConfig.model_validate(raw))

        assert windows[0].window_seconds is None

    @pytest.mark.parametrize(
        ("period_type", "expected"),
        [
            ("USAGE_PERIOD_TYPE_WEEKLY", ("weekly", "Weekly credits")),
            ("USAGE_PERIOD_TYPE_MONTHLY", ("monthly", "Monthly credits")),
            ("USAGE_PERIOD_TYPE_FUTURE", ("credits", "Credits")),
        ],
    )
    def test_the_credit_window_is_named_after_its_period(
        self, now: datetime, period_type: str, expected: tuple[str, str]
    ) -> None:
        config = CreditsConfig.model_validate(grok_credits_config(now, period_type=period_type))

        windows, facts = grok.windows_and_facts(config)

        assert (windows[0].key, windows[0].label) == expected
        assert windows[0].resets_at == now + timedelta(days=1)
        assert windows[0].window_seconds == WEEK
        assert facts == ()

    def test_on_demand_and_prepaid_amounts_appear_only_when_nonzero(self, now: datetime) -> None:
        raw = grok_credits_config(now)
        raw.update({"onDemandCap": {"val": 5000}, "onDemandUsed": {"val": 1250}, "prepaidBalance": {"val": 300}})

        _, facts = grok.windows_and_facts(CreditsConfig.model_validate(raw))

        assert [(fact.label, fact.value) for fact in facts] == [
            ("On-demand usage", "1,250 of 5,000 cap"),
            ("Prepaid balance", "300"),
        ]

    @pytest.mark.parametrize(
        "used",
        [
            pytest.param("leave out", id="no used amount"),
            pytest.param(None, id="a null used amount"),
            pytest.param({}, id="a used amount with no value"),
            pytest.param({"val": None}, id="a used amount whose value is null"),
        ],
    )
    def test_an_on_demand_amount_used_that_is_not_stated_is_said_to_be_so_and_not_shown_as_zero(
        self, now: datetime, used: object
    ) -> None:
        """Every answer the CLI logged states each amount, zero included; a missing one is no zero."""
        raw = grok_credits_config(now)
        raw["onDemandCap"] = {"val": 5000}
        if used == "leave out":
            del raw["onDemandUsed"]
        else:
            raw["onDemandUsed"] = used

        _, facts = grok.windows_and_facts(CreditsConfig.model_validate(raw))

        assert [(fact.label, fact.value) for fact in facts] == [("On-demand usage", "not stated, of 5,000 cap")]

    def test_an_on_demand_amount_used_stated_as_zero_is_shown_as_zero(self, now: datetime) -> None:
        raw = grok_credits_config(now)
        raw.update({"onDemandCap": {"val": 5000}, "onDemandUsed": {"val": 0}})

        _, facts = grok.windows_and_facts(CreditsConfig.model_validate(raw))

        assert [(fact.label, fact.value) for fact in facts] == [("On-demand usage", "0 of 5,000 cap")]


class FakeReader:
    """Stands in for the reader that asks the Grok CLI for the signed-in account's billing."""

    def __init__(self, result: QuotaReading | Exception, while_reading: Callable[[], None] = lambda: None):
        self.result = result
        self.calls = 0
        self._while_reading = while_reading

    @property
    def source_id(self) -> str:
        return "grok:live"

    @property
    def provider(self) -> Provider:
        return Provider.GROK

    def current_identity(self) -> AccountIdentity | None:
        return None

    def read_quota(self, now: datetime) -> QuotaReading:
        self.calls += 1
        self._while_reading()
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def never_asked() -> FakeReader:
    """A reader for a source that is only asked who is signed in. Being asked for usage fails the test."""
    return FakeReader(AssertionError("the reader was asked for usage"))


class TestGrokSourceReadsThroughItsReader:
    """Who is signed in comes from the auth file, and the usage from the reader, asked while the token is valid."""

    def home(
        self,
        tmp_path: Path,
        now: datetime,
        expires_in: timedelta | None = timedelta(hours=3),
        created_ago: timedelta | None = timedelta(hours=3),
        **entry: Any,
    ) -> Path:
        expires_at = None if expires_in is None else now + expires_in
        create_time = None if created_ago is None else now - created_ago
        grok_auth_file(tmp_path / "auth.json", create_time=create_time, expires_at=expires_at, **entry)
        return tmp_path

    def state_no_creation_time(self, home: Path) -> None:
        """Rewrite the home's login so that it states a creation time of ``null``."""
        document = json.loads((home / "auth.json").read_text())
        for entry in document.values():
            entry["create_time"] = None
        (home / "auth.json").write_text(json.dumps(document))

    def reader(self, now: datetime, while_reading: Callable[[], None] = lambda: None) -> FakeReader:
        return FakeReader(make_live_reading(now), while_reading)

    def test_the_reader_s_reading_is_returned_as_it_is(self, tmp_path: Path, now: datetime) -> None:
        reader = self.reader(now)

        reading = GrokSource(self.home(tmp_path, now), reader).read_quota(now)

        assert reading is reader.result
        assert reader.calls == 1

    @pytest.mark.parametrize(
        "expires_in",
        [
            pytest.param(timedelta(hours=-1), id="expired an hour ago"),
            pytest.param(timedelta(), id="expires now"),
            pytest.param(grok.TOKEN_EXPIRY_MARGIN, id="expires exactly at the margin"),
            pytest.param(None, id="does not say when it expires"),
        ],
    )
    def test_the_reader_is_not_asked_once_the_token_has_expired_or_is_about_to(
        self, tmp_path: Path, now: datetime, expires_in: timedelta | None
    ) -> None:
        """The reader runs the Grok CLI, which would refresh an expired token an open session still holds."""
        reader = self.reader(now)

        with pytest.raises(CredentialExpiredError, match="Grok access token has expired"):
            GrokSource(self.home(tmp_path, now, expires_in), reader).read_quota(now)

        assert reader.calls == 0

    def test_a_token_valid_for_just_over_the_margin_is_read(self, tmp_path: Path, now: datetime) -> None:
        reader = self.reader(now)
        home = self.home(tmp_path, now, grok.TOKEN_EXPIRY_MARGIN + timedelta(seconds=1))

        assert GrokSource(home, reader).read_quota(now) is reader.result

    @pytest.mark.parametrize(
        "token",
        [
            pytest.param("leave out", id="no token"),
            pytest.param("", id="an empty token"),
            pytest.param("   ", id="a blank token"),
            pytest.param(None, id="a null token"),
            pytest.param(7, id="a token that is not text"),
            pytest.param("synthetic grok token", id="a token with a space in it"),
            pytest.param("synthetic-grok-token\n", id="a token ending in a line feed"),
            pytest.param("synthetic-grok\x00token", id="a token holding a NUL"),
            pytest.param("synthetic-gr\u00f6k-token", id="a token that is not ASCII"),
        ],
    )
    def test_the_reader_is_not_asked_for_a_login_with_no_usable_access_token(
        self, tmp_path: Path, now: datetime, token: object
    ) -> None:
        """Its CLI would get a token with the refresh token, which rotates under an open Grok session."""
        home = self.home(tmp_path, now)
        document = json.loads((home / "auth.json").read_text())
        for entry in document.values():
            if token == "leave out":
                del entry[grok.ACCESS_TOKEN_FIELD]
            else:
                entry[grok.ACCESS_TOKEN_FIELD] = token
        (home / "auth.json").write_text(json.dumps(document))
        reader = self.reader(now)

        with pytest.raises(CredentialUnavailableError, match="Grok login has no usable access token"):
            GrokSource(home, reader).read_quota(now)

        assert reader.calls == 0

    @pytest.mark.parametrize("entry", ["not an object", ["a", "list"], None])
    def test_an_entry_that_is_not_an_object_is_refused_like_any_other_wrong_shape(self, entry: object) -> None:
        with pytest.raises(ValidationError):
            grok.AuthEntry.model_validate(entry)

    @pytest.mark.parametrize(
        ("token", "usable"),
        [
            pytest.param("!", True, id="one visible character"),
            pytest.param("synthetic-grok-token~", True, id="visible ASCII"),
            pytest.param(" synthetic-grok-token", False, id="a leading space"),
            pytest.param("synthetic-grok-token\x7f", False, id="a DEL"),
        ],
    )
    def test_a_token_is_usable_by_the_rules_of_every_stored_token(self, token: str, usable: bool) -> None:
        entry = grok.AuthEntry.model_validate({grok.ACCESS_TOKEN_FIELD: token})

        assert entry.has_access_token is usable

    def test_whether_there_is_a_token_is_kept_and_never_the_token(self, tmp_path: Path, now: datetime) -> None:
        """An entry can end up in an error or a log, and the token must not go with it."""
        entry = grok.current_auth_entry(self.home(tmp_path, now))

        assert entry is not None
        assert entry.has_access_token is True
        assert "synthetic-grok-token" not in repr(entry)
        assert grok.ACCESS_TOKEN_FIELD not in entry.model_dump()

    def test_a_logged_out_home_is_reported_as_nobody_signed_in_and_the_reader_is_not_asked(
        self, tmp_path: Path, now: datetime
    ) -> None:
        reader = self.reader(now)

        with pytest.raises(CredentialUnavailableError, match="not signed in"):
            GrokSource(tmp_path, reader).read_quota(now)

        assert reader.calls == 0

    def test_a_failure_of_the_reader_is_the_source_s_failure(self, tmp_path: Path, now: datetime) -> None:
        failure = ProviderRequestError("the Grok agent did not answer")

        with pytest.raises(ProviderRequestError) as raised:
            GrokSource(self.home(tmp_path, now), FakeReader(failure)).read_quota(now)

        assert raised.value is failure

    def test_a_logout_while_the_reader_reads_is_reported_as_nobody_signed_in(
        self, tmp_path: Path, now: datetime
    ) -> None:
        """The reader answers for whoever is signed in when it runs. That is nobody's reading now."""
        home = self.home(tmp_path, now)
        reader = self.reader(now, (home / "auth.json").unlink)

        with pytest.raises(CredentialUnavailableError, match="not signed in"):
            GrokSource(home, reader).read_quota(now)

        assert reader.calls == 1

    @pytest.mark.parametrize(
        "written_meanwhile",
        [
            pytest.param({"user_id": "grok-user-b"}, id="another account signed in"),
            pytest.param({"user_id": "grok-user-a"}, id="the same account signed in again, or its token refreshed"),
            pytest.param({"email": "second.persona@example.test"}, id="the same user id under another email"),
        ],
    )
    def test_a_new_credential_written_while_the_reader_reads_refuses_the_reading(
        self, tmp_path: Path, now: datetime, written_meanwhile: dict[str, str]
    ) -> None:
        """The reader was asked because the first credential's token was valid. It was not the one in use."""
        home = self.home(tmp_path, now)

        def write_a_new_credential() -> None:
            grok_auth_file(
                home / "auth.json", create_time=now, expires_at=now + timedelta(hours=6), **written_meanwhile
            )

        with pytest.raises(NoQuotaDataError, match=r"^the Grok login changed while it was being read"):
            GrokSource(home, self.reader(now, write_a_new_credential)).read_quota(now)

    @pytest.mark.parametrize("token", ["leave out", "synthetic grok token"], ids=["token removed", "token spoilt"])
    def test_a_credential_that_lost_its_usable_token_while_the_reader_reads_refuses_the_reading(
        self, tmp_path: Path, now: datetime, token: str
    ) -> None:
        """Every other field was unchanged, so the login compared as the same one and its usage was returned."""
        home = self.home(tmp_path, now)

        def spoil_the_token() -> None:
            document = json.loads((home / "auth.json").read_text())
            for entry in document.values():
                if token == "leave out":
                    del entry[grok.ACCESS_TOKEN_FIELD]
                else:
                    entry[grok.ACCESS_TOKEN_FIELD] = token
            (home / "auth.json").write_text(json.dumps(document))

        with pytest.raises(NoQuotaDataError, match=r"^the Grok login changed while it was being read"):
            GrokSource(home, self.reader(now, spoil_the_token)).read_quota(now)

    def test_a_creation_time_change_below_a_microsecond_does_not_keep_the_reading(
        self, tmp_path: Path, now: datetime
    ) -> None:
        """The two times collapse to one datetime, so the new token's usage was returned as the old credential's."""
        home = self.home(tmp_path, now)

        def rewrite_below_a_microsecond() -> None:
            document = json.loads((home / "auth.json").read_text())
            for entry in document.values():
                entry["create_time"] = "2026-10-03T11:00:00.000000100Z"
                entry[grok.ACCESS_TOKEN_FIELD] = "synthetic-grok-token-2"
            (home / "auth.json").write_text(json.dumps(document))

        reader = self.reader(now, rewrite_below_a_microsecond)

        with pytest.raises(ProviderSchemaError):
            GrokSource(home, reader).read_quota(now)

        assert reader.calls == 1

    def test_an_expiry_change_below_a_microsecond_does_not_keep_the_reading(
        self, tmp_path: Path, now: datetime
    ) -> None:
        home = self.home(tmp_path, now)

        def rewrite_the_expiry() -> None:
            document = json.loads((home / "auth.json").read_text())
            for entry in document.values():
                entry["expires_at"] = "2026-10-03T17:00:00.000000100Z"
            (home / "auth.json").write_text(json.dumps(document))

        reader = self.reader(now, rewrite_the_expiry)

        with pytest.raises(ProviderSchemaError):
            GrokSource(home, reader).read_quota(now)

        assert reader.calls == 1

    def test_zero_nanoseconds_on_a_credential_time_are_the_same_instant(self, tmp_path: Path, now: datetime) -> None:
        home = self.home(tmp_path, now)

        def pad_with_zero_nanoseconds() -> None:
            document = json.loads((home / "auth.json").read_text())
            for entry in document.values():
                entry["create_time"] = "2026-10-03T11:00:00.000000000Z"
                entry["expires_at"] = "2026-10-03T17:00:00.000000000Z"
            (home / "auth.json").write_text(json.dumps(document))

        reader = self.reader(now, pad_with_zero_nanoseconds)

        assert GrokSource(home, reader).read_quota(now) is reader.result

    def test_the_same_credential_written_again_while_the_reader_reads_is_the_same_login(
        self, tmp_path: Path, now: datetime
    ) -> None:
        home = self.home(tmp_path, now)
        reader = self.reader(now, lambda: self.home(tmp_path, now))

        assert GrokSource(home, reader).read_quota(now) is reader.result

    def test_a_credential_of_the_same_login_and_creation_time_with_another_expiry_refuses_the_reading(
        self, tmp_path: Path, now: datetime
    ) -> None:
        home = self.home(tmp_path, now, timedelta(hours=3))
        reader = self.reader(now, lambda: self.home(tmp_path, now, timedelta(hours=4)))

        with pytest.raises(NoQuotaDataError, match=r"^the Grok login changed while it was being read"):
            GrokSource(home, reader).read_quota(now)

    @pytest.mark.parametrize("states_null", [False, True], ids=["no creation time", "a creation time of null"])
    def test_the_reader_is_not_asked_for_a_credential_that_does_not_say_when_it_was_created(
        self, tmp_path: Path, now: datetime, states_null: bool
    ) -> None:
        """Two such credentials of one account are alike, so a switch away and back would go unseen."""
        home = self.home(tmp_path, now, created_ago=None)
        if states_null:
            self.state_no_creation_time(home)
        reader = self.reader(now)
        identity = GrokSource(home, reader).current_identity()

        assert identity is not None
        assert identity.external_id == "grok-user-a"
        with pytest.raises(ProviderSchemaError, match=r"^Grok login does not say when its credential was created$"):
            GrokSource(home, reader).read_quota(now)

        assert reader.calls == 0

    def test_a_switch_to_another_account_and_back_between_undated_credentials_never_reaches_the_reader(
        self, tmp_path: Path, now: datetime
    ) -> None:
        """Both surrounding reads would name the first account while the reader answered for the second."""
        home = self.home(tmp_path, now, created_ago=None, user_id="grok-user-a")

        def switch_away_and_back() -> None:
            self.home(tmp_path, now, created_ago=None, user_id="grok-user-b")
            self.home(tmp_path, now, created_ago=None, user_id="grok-user-a")

        reader = self.reader(now, switch_away_and_back)

        with pytest.raises(ProviderSchemaError, match="does not say when its credential was created"):
            GrokSource(home, reader).read_quota(now)

        assert reader.calls == 0

    def test_a_credential_that_is_undated_after_the_reader_read_refuses_the_reading(
        self, tmp_path: Path, now: datetime
    ) -> None:
        home = self.home(tmp_path, now)
        reader = self.reader(now, lambda: self.home(tmp_path, now, created_ago=None))

        with pytest.raises(ProviderSchemaError, match="does not say when its credential was created"):
            GrokSource(home, reader).read_quota(now)

        assert reader.calls == 1

    def test_a_credential_is_identified_by_its_login_its_times_and_whether_its_token_is_usable(
        self, tmp_path: Path, now: datetime
    ) -> None:
        entry = grok.current_auth_entry(self.home(tmp_path, now, timedelta(hours=3), timedelta(hours=2)))
        assert entry is not None

        assert grok.credential_key(entry) == (
            "grok-user-a",
            "grok-user-a",
            PERSONA_EMAIL,
            now - timedelta(hours=2),
            now + timedelta(hours=3),
            True,
        )

    def test_the_check_names_the_service_and_passes_equal_logins(self) -> None:
        ensure_same_login(("user-a", 1), ("user-a", 1), "Synthetic")

        with pytest.raises(NoQuotaDataError, match=r"^the Synthetic login changed while it was being read; the next"):
            ensure_same_login(("user-a", 1), ("user-b", 1), "Synthetic")


class TestGrokSource:
    def test_not_signed_in_means_no_credential_and_no_identity(self, tmp_path: Path, now: datetime) -> None:
        source = GrokSource(tmp_path, never_asked())
        assert source.current_identity() is None
        with pytest.raises(CredentialUnavailableError, match="not signed in"):
            source.read_quota(now)

    def test_a_newer_login_that_names_nobody_means_nobody_is_signed_in(self, tmp_path: Path, now: datetime) -> None:
        """The newest credential is the one in force. An older account must not stay signed in past it."""
        source = GrokSource(tmp_path, never_asked())
        (tmp_path / "auth.json").write_text(
            json.dumps(
                {
                    "issuer::old": {
                        "user_id": "old-user",
                        "email": "old@example.test",
                        "create_time": "2026-09-01T00:00:00Z",
                    },
                    "issuer::new": {
                        "user_id": "new-user",
                        "email": "new@example.test",
                        "create_time": "2026-10-01T00:00:00Z",
                    },
                    "issuer::anonymous": {"create_time": "2026-10-02T00:00:00Z"},
                    "version": 3,
                }
            )
        )

        assert source.current_identity() is None
        with pytest.raises(CredentialUnavailableError, match="not signed in"):
            source.read_quota(now)

    def test_an_older_login_that_names_nobody_does_not_hide_the_newer_account(self, tmp_path: Path) -> None:
        (tmp_path / "auth.json").write_text(
            json.dumps(
                {
                    "issuer::anonymous": {"create_time": "2026-09-01T00:00:00Z"},
                    "issuer::new": {
                        "user_id": "new-user",
                        "email": "new@example.test",
                        "create_time": "2026-10-01T00:00:00Z",
                    },
                }
            )
        )

        identity = GrokSource(tmp_path, never_asked()).current_identity()

        assert identity is not None
        assert identity.external_id == "new-user"

    def test_an_auth_file_with_no_login_means_nobody_is_signed_in(self, tmp_path: Path) -> None:
        (tmp_path / "auth.json").write_text(json.dumps({"version": {"schema": 3}}))

        assert GrokSource(tmp_path, never_asked()).current_identity() is None

    def test_a_newer_member_that_is_not_a_login_does_not_hide_the_account(self, tmp_path: Path) -> None:
        (tmp_path / "auth.json").write_text(
            json.dumps(
                {
                    "version": {"schema": 3, "create_time": "2026-10-03T00:00:00Z"},
                    "issuer::x": {"user_id": "the-user", "create_time": "2026-10-01T00:00:00Z"},
                }
            )
        )

        identity = GrokSource(tmp_path, never_asked()).current_identity()

        assert identity is not None
        assert identity.external_id == "the-user"

    @pytest.mark.parametrize("older_first", [True, False], ids=["older listed first", "newer listed first"])
    def test_of_two_logins_created_a_microsecond_apart_the_later_wins_whatever_their_order(
        self, tmp_path: Path, older_first: bool
    ) -> None:
        """Far enough ahead, a timestamp as a float holds no microseconds, and the two compared equal."""
        older = (
            "issuer::older",
            {"user_id": "older-user", "create_time": "2256-01-01T00:00:00.000001Z"},
        )
        newer = ("issuer::newer", {"user_id": "newer-user", "create_time": "2256-01-01T00:00:00.000002Z"})
        entries = [older, newer] if older_first else [newer, older]
        (tmp_path / "auth.json").write_text(json.dumps(dict(entries)))

        identity = GrokSource(tmp_path, never_asked()).current_identity()

        assert identity is not None
        assert identity.external_id == "newer-user"

    @pytest.mark.parametrize("dated_first", [True, False], ids=["dated listed first", "undated listed first"])
    @pytest.mark.parametrize(
        "undated",
        [{"user_id": "undated-user"}, {"user_id": "undated-user", "create_time": None}],
        ids=["creation time omitted", "creation time null"],
    )
    def test_a_dated_login_does_not_hide_an_undated_login(
        self, tmp_path: Path, now: datetime, dated_first: bool, undated: dict[str, str | None]
    ) -> None:
        """Neither login can be called newer, so neither one's quota may be read."""
        dated = ("issuer::dated", {"user_id": "dated-user", "create_time": "0001-01-01T00:00:01Z"})
        other = ("issuer::undated", undated)
        entries = [dated, other] if dated_first else [other, dated]
        (tmp_path / "auth.json").write_text(json.dumps(dict(entries)))
        source = GrokSource(tmp_path, never_asked())

        with pytest.raises(ProviderSchemaError, match="does not say when its credential was created"):
            source.current_identity()
        with pytest.raises(ProviderSchemaError, match="does not say when its credential was created"):
            source.read_quota(now)

    @pytest.mark.parametrize("valid_first", [True, False], ids=["valid listed first", "expired listed first"])
    def test_two_logins_created_at_one_time_are_refused_whatever_their_order(
        self, tmp_path: Path, now: datetime, valid_first: bool
    ) -> None:
        """The same file in the other order would otherwise name the other account and still ask the reader."""
        moment = "2026-10-01T00:00:00.000001Z"
        valid = (
            "issuer::valid",
            {
                "user_id": "valid-user",
                "create_time": moment,
                "expires_at": (now + timedelta(hours=3)).isoformat(),
                grok.ACCESS_TOKEN_FIELD: "synthetic-valid-token",
            },
        )
        expired = (
            "issuer::expired",
            {
                "user_id": "expired-user",
                "create_time": moment,
                "expires_at": (now - timedelta(hours=1)).isoformat(),
                grok.ACCESS_TOKEN_FIELD: "synthetic-expired-token",
            },
        )
        entries = [valid, expired] if valid_first else [expired, valid]
        (tmp_path / "auth.json").write_text(json.dumps(dict(entries)))
        reader = FakeReader(make_live_reading(now))
        source = GrokSource(tmp_path, reader)

        with pytest.raises(ProviderSchemaError, match="does not identify one newest login"):
            source.current_identity()
        with pytest.raises(ProviderSchemaError, match="does not identify one newest login"):
            source.read_quota(now)

        assert reader.calls == 0

    @pytest.mark.parametrize(
        "timestamp",
        [
            pytest.param("2026-10-03T14:00:00.000000100Z", id="a hundred nanoseconds"),
            pytest.param("2026-10-03T14:00:00.000000200Z", id="two hundred nanoseconds"),
            pytest.param("2026-10-03T14:00:00,000000100Z", id="a comma and a hundred nanoseconds"),
        ],
    )
    def test_a_creation_time_finer_than_a_microsecond_is_not_a_credential_time(self, timestamp: str) -> None:
        with pytest.raises(ValidationError):
            grok.AuthEntry.model_validate(
                {"user_id": "user-a", "create_time": timestamp, grok.ACCESS_TOKEN_FIELD: "synthetic-grok-token"}
            )

    def test_a_creation_time_that_is_not_text_is_not_a_credential_time(self) -> None:
        with pytest.raises(ValidationError):
            grok.AuthEntry.model_validate({"user_id": "user-a", "create_time": 1})

    def test_a_finer_fraction_in_the_offset_is_not_a_credential_time(self) -> None:
        with pytest.raises(ValidationError):
            grok.AuthEntry.model_validate(
                {
                    "user_id": "user-a",
                    "create_time": "2026-10-03T14:00:00.000000000+01:00:00.000000100",
                    grok.ACCESS_TOKEN_FIELD: "synthetic-grok-token",
                }
            )

    def test_an_expiry_finer_than_a_microsecond_is_not_a_credential_time(self) -> None:
        with pytest.raises(ValidationError):
            grok.AuthEntry.model_validate(
                {
                    "user_id": "user-a",
                    "create_time": "2026-10-03T14:00:00Z",
                    "expires_at": "2026-10-03T20:00:00.000000100Z",
                    grok.ACCESS_TOKEN_FIELD: "synthetic-grok-token",
                }
            )

    def test_zero_nanoseconds_are_the_same_credential_as_a_whole_second(self) -> None:
        plain = grok.AuthEntry.model_validate(
            {
                "user_id": "user-a",
                "create_time": "2026-10-03T14:00:00Z",
                "expires_at": "2026-10-03T20:00:00Z",
                grok.ACCESS_TOKEN_FIELD: "synthetic-grok-token",
            }
        )
        padded = grok.AuthEntry.model_validate(
            {
                "user_id": "user-a",
                "create_time": "2026-10-03T14:00:00.000000000Z",
                "expires_at": "2026-10-03T20:00:00.000000000Z",
                grok.ACCESS_TOKEN_FIELD: "synthetic-grok-token",
            }
        )

        assert grok.credential_key(plain) == grok.credential_key(padded)

    @pytest.mark.parametrize("entry", ["corrupt", ["a", "list"], None, 7])
    def test_a_login_that_is_not_an_object_is_a_schema_error_and_not_a_logout(
        self, tmp_path: Path, entry: object
    ) -> None:
        """Skipped, a file whose only login was damaged read as nobody signed in."""
        (tmp_path / "auth.json").write_text(json.dumps({"issuer::x": entry, "version": 3}))

        with pytest.raises(ProviderSchemaError, match="Grok auth file has an unexpected shape"):
            GrokSource(tmp_path, never_asked()).current_identity()

    def test_a_member_that_is_not_a_login_is_not_read_as_one(self, tmp_path: Path) -> None:
        (tmp_path / "auth.json").write_text(
            json.dumps(
                {"version": {"schema": 3}, "issuer::x": {"user_id": "the-user", "create_time": "2026-10-01T00:00:00Z"}}
            )
        )

        identity = GrokSource(tmp_path, never_asked()).current_identity()

        assert identity is not None
        assert identity.external_id == "the-user"

    def test_an_auth_entry_with_the_wrong_shape_is_a_schema_error(self, tmp_path: Path) -> None:
        (tmp_path / "auth.json").write_text(json.dumps({"issuer::x": {"user_id": 12345}}))
        with pytest.raises(ProviderSchemaError, match="unexpected shape"):
            GrokSource(tmp_path, never_asked()).current_identity()


class TestSourceIdentification:
    def test_each_source_is_named_by_its_provider_and_the_directory_it_reads(
        self, no_requests_client: httpx2.Client, tmp_path: Path
    ) -> None:
        claude_config = ClaudeSourceConfig(
            config_dir=tmp_path / ".claude",
            account_file=tmp_path / ".claude.json",
            mode=SourceMode.DIRECT,
            keychain_service="Synthetic-credentials",
        )
        sources = [
            ClaudeSource(claude_config, no_requests_client, lambda _: claude_keychain_blob()),
            CodexSource(tmp_path / ".codex", no_requests_client),
            GrokSource(tmp_path / ".grok", never_asked()),
        ]

        assert [(source.provider, source.source_id) for source in sources] == [
            (Provider.CLAUDE, f"claude:{tmp_path / '.claude'}"),
            (Provider.CODEX, f"codex:{tmp_path / '.codex'}"),
            (Provider.GROK, f"grok:{tmp_path / '.grok'}"),
        ]


class TestClaudeEdgeCases:
    @pytest.mark.parametrize("resets_at", [None, 1791036000, "next month", "2026-11-02T14:00:00"])
    def test_a_credit_pool_whose_reset_time_cannot_be_read_is_shown_without_one(
        self, now: datetime, resets_at: object
    ) -> None:
        payload = claude_usage_payload(now)
        payload["promo_pool"]["resets_at"] = resets_at

        _, facts = claude.parse_usage(payload)

        assert (facts[1].label, facts[1].value, facts[1].resets_at) == (
            "Credit pool · promo_pool",
            "$12.50 of $250.00 used",
            None,
        )

    def test_a_login_with_no_keychain_service_configured_has_no_credential(
        self, no_requests_client: httpx2.Client, tmp_path: Path, now: datetime
    ) -> None:
        config = ClaudeSourceConfig(
            config_dir=tmp_path / "standing",
            account_file=tmp_path / "standing" / ".claude.json",
            mode=SourceMode.CLI,
            keychain_service=None,
        )
        source = ClaudeSource(config, no_requests_client, lambda _: claude_keychain_blob())

        with pytest.raises(CredentialUnavailableError, match="no keychain service is configured"):
            source.read_quota(now)

    def test_a_profile_answer_with_the_wrong_shape_is_a_schema_error_and_no_usage_is_requested(
        self, tmp_path: Path, now: datetime
    ) -> None:
        config = ClaudeSourceConfig(
            config_dir=tmp_path / ".claude",
            account_file=tmp_path / ".claude.json",
            mode=SourceMode.DIRECT,
            keychain_service="Synthetic-credentials",
        )
        transport = RecordingTransport(
            {
                "/api/oauth/profile": json_response({"account": {"email": PERSONA_EMAIL}}),
                "/api/oauth/usage": json_response(claude_usage_payload(now)),
            }
        )
        source = ClaudeSource(config, transport.client(), lambda _: claude_keychain_blob())

        with pytest.raises(ProviderSchemaError, match="Claude profile payload has an unexpected shape: organization"):
            source.read_quota(now)

        assert transport.paths() == ["/api/oauth/profile"]


class TestCodexEdgeCases:
    def write_auth(self, home: Path, tokens: dict[str, Any]) -> None:
        (home / "auth.json").write_text(json.dumps({"auth_mode": "chatgpt", "tokens": tokens}))

    def test_a_login_that_names_no_workspace_is_treated_as_logged_out(
        self, no_requests_client: httpx2.Client, tmp_path: Path, now: datetime
    ) -> None:
        self.write_auth(tmp_path, {"access_token": make_jwt({}), "id_token": make_jwt({"email": PERSONA_EMAIL})})
        source = CodexSource(tmp_path, no_requests_client)

        assert source.current_identity() is None
        with pytest.raises(CredentialUnavailableError, match="does not identify a ChatGPT workspace"):
            source.read_quota(now)

    def test_the_workspace_can_come_from_the_id_token_when_the_file_does_not_name_it(
        self, no_requests_client: httpx2.Client, tmp_path: Path
    ) -> None:
        id_token = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "workspace-from-claim"}})
        self.write_auth(tmp_path, {"access_token": make_jwt({}), "id_token": id_token})

        identity = CodexSource(tmp_path, no_requests_client).current_identity()

        assert identity is not None
        assert identity.external_id == "workspace-from-claim"

    def test_a_login_without_an_id_token_is_identified_by_its_workspace_alone(
        self, no_requests_client: httpx2.Client, tmp_path: Path
    ) -> None:
        self.write_auth(tmp_path, {"access_token": make_jwt({}), "account_id": "workspace-a"})

        identity = CodexSource(tmp_path, no_requests_client).current_identity()

        assert identity is not None
        assert (identity.external_id, identity.email, identity.plan) == ("workspace-a", None, None)

    @pytest.mark.parametrize("tokens", [{"access_token": ""}, {"account_id": "workspace-a"}, {"access_token": 5}])
    def test_a_token_set_without_a_usable_access_token_is_no_credential(
        self, no_requests_client: httpx2.Client, tmp_path: Path, now: datetime, tokens: dict[str, Any]
    ) -> None:
        self.write_auth(tmp_path, tokens)
        source = CodexSource(tmp_path, no_requests_client)

        assert source.current_identity() is None
        with pytest.raises(CredentialUnavailableError, match="holds no usable ChatGPT token"):
            source.read_quota(now)

    def test_the_plan_in_the_auth_file_is_kept_when_the_provider_does_not_state_one(
        self, tmp_path: Path, now: datetime
    ) -> None:
        codex_auth_file(tmp_path / "auth.json", plan="team")
        payload = codex_usage_payload(now)
        payload["plan_type"] = None
        transport = RecordingTransport({"/backend-api/wham/usage": json_response(payload)})

        assert CodexSource(tmp_path, transport.client()).read_quota(now).identity.plan == "Team"

    def test_an_extra_limit_that_fails_validation_is_left_out_said_so_on_the_card_and_the_rest_is_kept(
        self, now: datetime
    ) -> None:
        """Logged only, the card looked complete without it; failing the reading would hide the main limit."""
        payload = codex_usage_payload(now)
        payload["credits"] = None
        window = payload["rate_limit"]["primary_window"]
        broken = {"primary_window": {**window, "used_percent": "lots"}}
        payload["additional_rate_limits"] = [
            {"limit_name": "Broken", "metered_feature": "broken", "rate_limit": broken},
            {"metered_feature": "deep_research", "rate_limit": {"primary_window": window}},
        ]

        _, windows, facts = codex.parse_usage(payload)

        assert [(w.key, w.label) for w in windows] == [
            ("weekly", "Weekly"),
            ("deep_research:weekly", "Deep research · Weekly"),
        ]
        assert [(fact.label, fact.value) for fact in facts] == [
            ("Limits not shown", "1 extra limit with an unexpected shape; the log says which")
        ]

    def test_every_extra_limit_left_out_is_counted_in_the_one_fact_that_says_so(self, now: datetime) -> None:
        payload = codex_usage_payload(now)
        payload["credits"] = None
        window = payload["rate_limit"]["primary_window"]
        payload["additional_rate_limits"] = [
            {"unexpected": "shape"},
            "not even an object",
            {"rate_limit": []},
            {"limit_name": "Names no meter", "rate_limit": {"primary_window": window}},
        ]

        _, windows, facts = codex.parse_usage(payload)

        assert [w.key for w in windows] == ["weekly"]
        assert [(fact.label, fact.value) for fact in facts] == [
            ("Limits not shown", "4 extra limits with an unexpected shape; the log says which")
        ]

    @pytest.mark.parametrize(
        "rate_limit",
        [
            pytest.param({}, id="no window field"),
            pytest.param({"primary_window": None, "secondary_window": None}, id="every window null"),
            pytest.param({"tertiary_window": {"used_percent": 50}}, id="only a window field this does not know"),
        ],
    )
    def test_an_extra_limit_with_no_window_is_counted_as_not_shown(
        self, now: datetime, rate_limit: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        """It validates, and converted to no window at all it left the card looking complete."""
        payload = codex_usage_payload(now)
        payload["credits"] = None
        payload["additional_rate_limits"] = [{"metered_feature": "deep_research", "rate_limit": rate_limit}]

        with caplog.at_level(logging.WARNING):
            _, windows, facts = codex.parse_usage(payload)

        assert [w.key for w in windows] == ["weekly"]
        assert [(fact.label, fact.value) for fact in facts] == [
            ("Limits not shown", "1 extra limit with an unexpected shape; the log says which")
        ]
        assert "Leaving out additional Codex rate limit 0 with no window" in caplog.text

    def test_no_fact_about_left_out_limits_is_shown_when_every_extra_limit_is_read(self, now: datetime) -> None:
        payload = codex_usage_payload(now)
        payload["credits"] = None
        payload["additional_rate_limits"] = [
            {
                "metered_feature": "deep_research",
                "rate_limit": {"primary_window": payload["rate_limit"]["primary_window"]},
            }
        ]

        assert codex.parse_usage(payload)[2] == ()

    @pytest.mark.parametrize("credits", [None, {"has_credits": False}, {"unlimited": False, "balance": None}])
    def test_no_credit_fact_is_shown_without_a_balance(self, now: datetime, credits: dict[str, Any] | None) -> None:
        payload = codex_usage_payload(now)
        payload["credits"] = credits

        assert codex.parse_usage(payload)[2] == ()


class TestGrokEdgeCases:
    def test_an_auth_file_whose_entries_name_no_user_means_nobody_is_logged_in(self, tmp_path: Path) -> None:
        (tmp_path / "auth.json").write_text(
            json.dumps({"issuer::anonymous": {"create_time": "2026-10-02T00:00:00Z"}, "version": 3})
        )

        assert GrokSource(tmp_path, never_asked()).current_identity() is None

    def test_an_entry_that_names_no_user_has_no_identity(self) -> None:
        with pytest.raises(CredentialUnavailableError, match="does not identify a user"):
            grok.identity_from_auth_entry(grok.AuthEntry(), plan=None)

    @pytest.mark.parametrize(
        ("fields", "expected_id", "expected_label"),
        [
            ({"user_id": "u-1", "principal_id": "p-1", "email": PERSONA_EMAIL}, "u-1", PERSONA_EMAIL),
            ({"principal_id": "p-1", "email": PERSONA_EMAIL}, "p-1", PERSONA_EMAIL),
            ({"email": PERSONA_EMAIL}, PERSONA_EMAIL, PERSONA_EMAIL),
            ({"user_id": "u-1"}, "u-1", "Grok account"),
        ],
    )
    def test_the_account_is_keyed_on_the_most_stable_identifier_the_entry_has(
        self, fields: dict[str, str], expected_id: str, expected_label: str
    ) -> None:
        identity = grok.identity_from_auth_entry(grok.AuthEntry.model_validate(fields), plan="SuperGrok")

        assert (identity.external_id, identity.default_label, identity.plan) == (
            expected_id,
            expected_label,
            "SuperGrok",
        )


def make_live_reading(now: datetime) -> QuotaReading:
    identity = AccountIdentity(Provider.GROK, "grok-user-a", "live")
    window = QuotaWindow("weekly", "Weekly credits", 72.0, now + timedelta(days=1), WEEK)
    return QuotaReading(identity=identity, windows=(window,), observed_at=now, source="api")


REFRESH_SECRET = "synthetic-refresh-token-value"


def rendered(error: BaseException) -> str:
    """Everything a log or a traceback would print for ``error``, including the errors it was raised from."""
    return "".join(traceback.format_exception(error))


class TestCredentialsNeverReachATraceback:
    def test_a_claude_keychain_item_missing_its_access_token_does_not_expose_its_other_tokens(
        self, no_requests_client: httpx2.Client, tmp_path: Path, now: datetime
    ) -> None:
        config = ClaudeSourceConfig(
            config_dir=tmp_path / ".claude",
            account_file=tmp_path / ".claude.json",
            mode=SourceMode.DIRECT,
            keychain_service="Synthetic-credentials",
        )
        blob = json.dumps({"claudeAiOauth": {"refreshToken": REFRESH_SECRET, "subscriptionType": "max"}})
        source = ClaudeSource(config, no_requests_client, lambda _: blob)

        with pytest.raises(CredentialUnavailableError, match="no usable OAuth credential") as raised:
            source.read_quota(now)

        assert REFRESH_SECRET not in rendered(raised.value)

    def test_a_claude_keychain_item_that_is_not_json_does_not_expose_its_content(
        self, no_requests_client: httpx2.Client, tmp_path: Path, now: datetime
    ) -> None:
        config = ClaudeSourceConfig(
            config_dir=tmp_path / ".claude",
            account_file=tmp_path / ".claude.json",
            mode=SourceMode.DIRECT,
            keychain_service="Synthetic-credentials",
        )
        source = ClaudeSource(config, no_requests_client, lambda _: f'{{"refreshToken": "{REFRESH_SECRET}" truncated')

        with pytest.raises(ProviderSchemaError, match="is not valid JSON") as raised:
            source.read_quota(now)

        assert REFRESH_SECRET not in rendered(raised.value)

    def test_a_codex_auth_file_missing_its_access_token_does_not_expose_its_other_tokens(
        self, no_requests_client: httpx2.Client, tmp_path: Path, now: datetime
    ) -> None:
        (tmp_path / "auth.json").write_text(
            json.dumps({"tokens": {"refresh_token": REFRESH_SECRET, "account_id": "workspace-a"}})
        )

        with pytest.raises(CredentialUnavailableError, match="no usable ChatGPT token") as raised:
            CodexSource(tmp_path, no_requests_client).read_quota(now)

        assert REFRESH_SECRET not in rendered(raised.value)

    def test_a_grok_auth_entry_with_the_wrong_shape_does_not_expose_its_tokens(self, tmp_path: Path) -> None:
        (tmp_path / "auth.json").write_text(
            json.dumps({"issuer::x": {"user_id": 12345, "key": REFRESH_SECRET, "refresh_token": REFRESH_SECRET}})
        )

        with pytest.raises(ProviderSchemaError, match="unexpected shape") as raised:
            GrokSource(tmp_path, never_asked()).current_identity()

        assert REFRESH_SECRET not in rendered(raised.value)


class TestClaudeDisplayDetailsNeverCostTheLimits:
    @pytest.mark.parametrize("decimal_places", [309, -324, -323, -1, 7, 2.5, "two"])
    def test_an_overage_setting_with_an_unusable_scale_is_left_out_and_the_limits_are_kept(
        self, now: datetime, decimal_places: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = claude_usage_payload(now)
        payload["extra_usage"] = {
            "is_enabled": True,
            "monthly_limit": 10000,
            "used_credits": 2550,
            "decimal_places": decimal_places,
        }

        with caplog.at_level(logging.WARNING, logger="usagedash.providers.claude"):
            windows, facts = claude.parse_usage(payload)

        assert [w.key for w in windows] == ["five_hour", "weekly", "weekly:fable"]
        assert [fact.label for fact in facts] == ["Credit pool · promo_pool"]
        assert "Leaving out Claude extra usage with an unexpected shape: decimal_places" in caplog.text

    @pytest.mark.parametrize(
        ("decimal_places", "expected"),
        [(0, "$2,550.00 of $10,000.00 used"), (2, "$25.50 of $100.00 used"), (6, "$0.00 of $0.01 used")],
    )
    def test_the_overage_amounts_are_scaled_by_the_stated_number_of_minor_digits(
        self, now: datetime, decimal_places: int, expected: str
    ) -> None:
        payload = claude_usage_payload(now)
        payload["extra_usage"] = {
            "is_enabled": True,
            "monthly_limit": 10000,
            "used_credits": 2550,
            "decimal_places": decimal_places,
        }

        _, facts = claude.parse_usage(payload)

        assert facts[0].value == expected

    @pytest.mark.parametrize("extra_usage", ["on", ["enabled"], {"is_enabled": True, "used_credits": -1}, 5])
    def test_an_overage_setting_with_any_other_unknown_shape_is_left_out(
        self, now: datetime, extra_usage: object
    ) -> None:
        payload = claude_usage_payload(now)
        payload["extra_usage"] = extra_usage

        windows, facts = claude.parse_usage(payload)

        assert len(windows) == 3
        assert [fact.label for fact in facts] == ["Credit pool · promo_pool"]

    @pytest.mark.parametrize("amount", [10**400, -(10**400), True, "12.50", None, math.inf, math.nan])
    def test_a_credit_pool_with_an_amount_that_is_not_a_usable_number_is_left_out_and_logged(
        self, now: datetime, amount: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = claude_usage_payload(now)
        payload["promo_pool"]["limit_dollars"] = amount

        with caplog.at_level(logging.WARNING, logger="usagedash.providers.claude"):
            windows, facts = claude.parse_usage(payload)

        assert len(windows) == 3
        assert [fact.label for fact in facts] == ["Extra usage"]
        assert caplog.messages == ["Leaving out a Claude credit pool without a usable limit_dollars"]

    @pytest.mark.parametrize(
        ("stated", "expected"),
        [
            ({"limit_dollars": 250}, "used_dollars"),
            ({"used_dollars": 12.5}, "limit_dollars"),
            ({"limit_dollars": 250, "used_dollars": None}, "used_dollars"),
            ({"limit_dollars": None, "used_dollars": 0}, "limit_dollars"),
            ({"limit_dollars": "250", "used_dollars": None}, "limit_dollars or used_dollars"),
        ],
    )
    def test_a_credit_pool_that_does_not_state_both_amounts_is_left_out_and_logged_by_what_it_lacks(
        self, now: datetime, stated: dict[str, object], expected: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = claude_usage_payload(now)
        payload["promo_pool"] = stated

        with caplog.at_level(logging.WARNING, logger="usagedash.providers.claude"):
            windows, facts = claude.parse_usage(payload)

        assert len(windows) == 3
        assert [fact.label for fact in facts] == ["Extra usage"]
        assert caplog.messages == [f"Leaving out a Claude credit pool without a usable {expected}"]

    def test_a_credit_pool_that_is_left_out_is_logged_without_the_name_the_provider_gave_it(
        self, now: datetime, caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = claude_usage_payload(now)
        payload["a name only the provider chose"] = {"limit_dollars": "unreadable", "used_dollars": 1}

        with caplog.at_level(logging.WARNING, logger="usagedash.providers.claude"):
            _, facts = claude.parse_usage(payload)

        assert [fact.label for fact in facts] == ["Extra usage", "Credit pool · promo_pool"]
        assert len(caplog.messages) == 1
        assert "a name only the provider chose" not in caplog.text
        assert "unreadable" not in caplog.text

    def test_one_unreadable_credit_pool_does_not_cost_the_pools_beside_it(
        self, now: datetime, caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = claude_usage_payload(now)
        payload["broken_pool"] = {"limit_dollars": 50}
        payload["second_pool"] = {"limit_dollars": 40, "used_dollars": 10}

        with caplog.at_level(logging.WARNING, logger="usagedash.providers.claude"):
            _, facts = claude.parse_usage(payload)

        assert [(fact.label, fact.value) for fact in facts[1:]] == [
            ("Credit pool · promo_pool", "$12.50 of $250.00 used"),
            ("Credit pool · second_pool", "$10.00 of $40.00 used"),
        ]
        assert caplog.messages == ["Leaving out a Claude credit pool without a usable used_dollars"]

    def test_the_payload_as_the_api_sends_it_logs_nothing(
        self, now: datetime, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Its ordinary limits carry the dollar fields too, as null. A warning for those would repeat every read."""
        payload = claude_usage_payload(now)
        assert payload["five_hour"]["limit_dollars"] is None

        with caplog.at_level(logging.WARNING, logger="usagedash.providers.claude"):
            _, facts = claude.parse_usage(payload)

        assert [fact.label for fact in facts] == ["Extra usage", "Credit pool · promo_pool"]
        assert caplog.messages == []

    @pytest.mark.parametrize(
        "entry",
        [
            {},
            {"utilization": 40.0},
            {"note": "limit_dollars"},
            {"dollars": 5},
            {"utilization": 40.0, "limit_dollars": None, "used_dollars": None},
            {"limit_dollars": None},
            {"used_dollars": None},
        ],
    )
    def test_an_entry_that_states_no_dollar_amount_is_not_a_credit_pool_and_is_not_logged(
        self, now: datetime, entry: dict[str, object], caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = claude_usage_payload(now)
        payload["some_other_entry"] = entry

        with caplog.at_level(logging.WARNING, logger="usagedash.providers.claude"):
            _, facts = claude.parse_usage(payload)

        assert [fact.label for fact in facts] == ["Extra usage", "Credit pool · promo_pool"]
        assert caplog.messages == []

    @pytest.mark.parametrize("resets_at", ["9999-12-31T23:59:30+00:00", "9999-12-31T23:59:59-05:00"])
    def test_a_credit_pool_reset_at_the_edge_of_the_calendar_is_shown_without_a_reset_time(
        self, now: datetime, resets_at: str
    ) -> None:
        payload = claude_usage_payload(now)
        payload["promo_pool"]["resets_at"] = resets_at

        _, facts = claude.parse_usage(payload)

        assert (facts[1].label, facts[1].resets_at) == ("Credit pool · promo_pool", None)

    def test_a_credit_pool_reset_is_normalized_to_utc(self, now: datetime) -> None:
        payload = claude_usage_payload(now)
        payload["promo_pool"]["resets_at"] = "2026-11-02T10:00:20-04:00"

        _, facts = claude.parse_usage(payload)

        assert facts[1].resets_at == datetime(2026, 11, 2, 14, 0, tzinfo=UTC)


class TestClaudeLimitsThatCannotBeInterpreted:
    def test_a_reset_time_that_cannot_be_rounded_within_the_calendar_is_a_schema_error(self) -> None:
        payload = {"limits": [{"kind": "session", "percent": 1, "resets_at": "9999-12-31T23:59:30+00:00"}]}

        with pytest.raises(ProviderSchemaError, match="Claude usage API payload could not be interpreted: ValueError"):
            claude.parse_usage(payload)

    def test_a_reset_time_that_cannot_be_expressed_in_utc_is_a_schema_error_naming_the_field(self) -> None:
        payload = {"limits": [{"kind": "session", "percent": 1, "resets_at": "9999-12-31T23:59:59-05:00"}]}

        with pytest.raises(ProviderSchemaError, match=r"unexpected shape: limits\.0\.resets_at"):
            claude.parse_usage(payload)

    def test_a_reset_time_is_normalized_to_utc(self) -> None:
        payload = {"limits": [{"kind": "session", "percent": 1, "resets_at": "2026-10-03T14:20:00-04:00"}]}

        windows, _ = claude.parse_usage(payload)

        assert windows[0].resets_at == datetime(2026, 10, 3, 18, 20, tzinfo=UTC)


class TestBlankIdentifiers:
    """A blank id must never become an account: every login without a usable id would share one."""

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_a_claude_account_file_with_a_blank_organization_is_a_schema_error(
        self, no_requests_client: httpx2.Client, tmp_path: Path, blank: str
    ) -> None:
        account_file = tmp_path / ".claude.json"
        account_file.write_text(json.dumps({"oauthAccount": {"organizationUuid": blank}}))

        with pytest.raises(ProviderSchemaError, match=r"unexpected shape: oauthAccount\.organizationUuid"):
            claude.identity_from_account_file(account_file, None)

    def test_a_claude_organization_id_has_its_surrounding_whitespace_removed(self, tmp_path: Path) -> None:
        account_file = tmp_path / ".claude.json"
        account_file.write_text(json.dumps({"oauthAccount": {"organizationUuid": "  org-a \n"}}))

        identity = claude.identity_from_account_file(account_file, None)

        assert identity is not None
        assert identity.external_id == "org-a"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_codex_workspace_in_the_auth_file_falls_back_to_the_id_token(
        self, no_requests_client: httpx2.Client, tmp_path: Path, blank: str
    ) -> None:
        id_token = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": " workspace-from-claim "}})
        (tmp_path / "auth.json").write_text(
            json.dumps({"tokens": {"access_token": make_jwt({}), "id_token": id_token, "account_id": blank}})
        )

        identity = CodexSource(tmp_path, no_requests_client).current_identity()

        assert identity is not None
        assert identity.external_id == "workspace-from-claim"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_codex_login_whose_every_workspace_id_is_blank_is_treated_as_logged_out(
        self, no_requests_client: httpx2.Client, tmp_path: Path, blank: str
    ) -> None:
        id_token = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": blank}})
        (tmp_path / "auth.json").write_text(
            json.dumps({"tokens": {"access_token": make_jwt({}), "id_token": id_token, "account_id": blank}})
        )

        assert CodexSource(tmp_path, no_requests_client).current_identity() is None

    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_a_blank_workspace_in_the_codex_answer_falls_back_to_the_auth_file(
        self, tmp_path: Path, now: datetime, blank: str | None
    ) -> None:
        codex_auth_file(tmp_path / "auth.json", account_id="workspace-a")
        payload = codex_usage_payload(now)
        payload["account_id"] = blank
        transport = RecordingTransport({"/backend-api/wham/usage": json_response(payload)})

        assert CodexSource(tmp_path, transport.client()).read_quota(now).identity.external_id == "workspace-a"

    def test_a_grok_entry_falls_through_blank_ids_to_the_first_usable_one(self, tmp_path: Path) -> None:
        (tmp_path / "auth.json").write_text(
            json.dumps({"issuer::x": {"user_id": "  ", "principal_id": "", "email": " user-a@example.test "}})
        )

        identity = GrokSource(tmp_path, never_asked()).current_identity()

        assert identity is not None
        assert (identity.external_id, identity.email) == ("user-a@example.test", "user-a@example.test")

    def test_a_grok_entry_whose_every_id_is_blank_means_nobody_is_logged_in(self, tmp_path: Path) -> None:
        (tmp_path / "auth.json").write_text(
            json.dumps({"issuer::x": {"user_id": " ", "principal_id": "", "email": "\t"}})
        )

        assert GrokSource(tmp_path, never_asked()).current_identity() is None


class TestValuesOfTheWrongTypeAreNeverTurnedIntoReadings:
    """pydantic's default mode would read ``true`` as 1 percent and ``12345`` as a date in 1970."""

    @pytest.mark.parametrize("percent", [True, False, "50"])
    def test_a_claude_limit_percentage(self, percent: object) -> None:
        with pytest.raises(ProviderSchemaError, match=r"limits\.0\.percent"):
            claude.parse_usage({"limits": [{"kind": "session", "percent": percent}]})

    @pytest.mark.parametrize("percent", [True, False, "50"])
    def test_a_claude_legacy_utilization(self, percent: object) -> None:
        with pytest.raises(ProviderSchemaError, match=r"five_hour\.utilization"):
            claude.parse_usage({"five_hour": {"utilization": percent}})

    @pytest.mark.parametrize("resets_at", [1_791_036_000, 1_791_036_000.5, "1791036000", True])
    def test_a_claude_reset_time_that_is_not_iso_text(self, resets_at: object) -> None:
        with pytest.raises(ProviderSchemaError, match=r"limits\.0\.resets_at") as raised:
            claude.parse_usage({"limits": [{"kind": "session", "percent": 1, "resets_at": resets_at}]})

        assert "1791036000" not in str(raised.value)

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (("rate_limit", "primary_window", "used_percent"), True),
            (("rate_limit", "primary_window", "used_percent"), "38"),
            (("rate_limit", "primary_window", "reset_at"), True),
            (("rate_limit", "primary_window", "reset_at"), "1791036000"),
            (("rate_limit", "primary_window", "limit_window_seconds"), True),
            (("rate_limit", "primary_window", "limit_window_seconds"), "604800"),
            (("rate_limit", "primary_window", "limit_window_seconds"), 604800.0),
            (("rate_limit", "allowed"), "false"),
            (("rate_limit", "allowed"), 0),
            (("rate_limit", "limit_reached"), "true"),
            (("rate_limit", "limit_reached"), 1),
            (("credits", "has_credits"), "yes"),
            (("credits", "unlimited"), 1),
            (("credits", "balance"), True),
        ],
    )
    def test_a_codex_usage_field(self, now: datetime, path: tuple[str, ...], value: object) -> None:
        payload = codex_usage_payload(now)
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

        with pytest.raises(ProviderSchemaError, match=r"\.".join(path)):
            codex.parse_usage(payload)

    def test_text_saying_a_codex_limit_is_not_reached_does_not_read_as_an_open_account(self, now: datetime) -> None:
        """``"false"`` is truthy text; it must not be taken for either answer."""
        payload = codex_usage_payload(now)
        payload["rate_limit"]["limit_reached"] = "false"

        with pytest.raises(ProviderSchemaError, match=r"rate_limit\.limit_reached"):
            codex.parse_usage(payload)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("creditUsagePercent", True), ("creditUsagePercent", "5"), ("onDemandCap", {"val": True})],
    )
    def test_a_grok_credits_field(self, now: datetime, field: str, value: object) -> None:
        with pytest.raises(ValidationError, match=field):
            CreditsConfig.model_validate(grok_credits_config(now) | {field: value})

    @pytest.mark.parametrize("start", [1_791_036_000, "1791036000", True])
    def test_a_grok_period_boundary_that_is_not_iso_text(self, now: datetime, start: object) -> None:
        config = grok_credits_config(now)
        config["currentPeriod"]["start"] = start

        with pytest.raises(ValidationError, match=r"currentPeriod\.start"):
            CreditsConfig.model_validate(config)

    def test_a_grok_amount_sent_as_decimal_text_is_read_as_the_number(self, now: datetime) -> None:
        config = CreditsConfig.model_validate(grok_credits_config(now) | {"prepaidBalance": {"val": "300"}})

        assert config.prepaid_balance is not None
        assert config.prepaid_balance.val == 300.0

    @pytest.mark.parametrize("is_enabled", ["true", "false", 1, 0, None])
    def test_a_claude_overage_flag_that_is_not_a_boolean_leaves_the_overage_out_and_keeps_the_limits(
        self, now: datetime, is_enabled: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = claude_usage_payload(now)
        payload["extra_usage"]["is_enabled"] = is_enabled

        with caplog.at_level(logging.WARNING, logger="usagedash.providers.claude"):
            windows, facts = claude.parse_usage(payload)

        assert [w.key for w in windows] == ["five_hour", "weekly", "weekly:fable"]
        assert [fact.label for fact in facts] == ["Credit pool · promo_pool"]
        assert "Leaving out Claude extra usage with an unexpected shape: is_enabled" in caplog.text


TOKENS_A_HEADER_CANNOT_CARRY = (
    "zq9marker\ntoken",
    "zq9marker\rtoken",
    "zq9marker\x00token",
    "zq9marker token",
    "zq9marker-é",
)


class TestCredentialsAHeaderCannotCarry:
    """A stored token is sent as a header. One a header cannot carry is no credential, and is never sent."""

    @pytest.mark.parametrize("access_token", TOKENS_A_HEADER_CANNOT_CARRY)
    def test_claude_reports_no_usable_credential_and_sends_nothing(
        self, no_requests_client: httpx2.Client, tmp_path: Path, now: datetime, access_token: str
    ) -> None:
        config = ClaudeSourceConfig(
            config_dir=tmp_path,
            account_file=tmp_path / ".claude.json",
            mode=SourceMode.DIRECT,
            keychain_service="Synthetic-credentials",
        )
        write_claude_account_file(config.account_file)
        source = ClaudeSource(config, no_requests_client, lambda _: claude_keychain_blob(access_token=access_token))

        with pytest.raises(CredentialUnavailableError, match="no usable OAuth credential") as raised:
            source.read_quota(now)

        assert "zq9marker" not in "".join(traceback.format_exception(raised.value))

    @pytest.mark.parametrize("access_token", TOKENS_A_HEADER_CANNOT_CARRY)
    def test_codex_reports_no_usable_token_and_sends_nothing(
        self, no_requests_client: httpx2.Client, tmp_path: Path, now: datetime, access_token: str
    ) -> None:
        codex_auth_file(tmp_path / "auth.json", access_token=access_token)

        with pytest.raises(CredentialUnavailableError, match="no usable ChatGPT token") as raised:
            CodexSource(tmp_path, no_requests_client).read_quota(now)

        assert "zq9marker" not in "".join(traceback.format_exception(raised.value))

    @pytest.mark.parametrize("workspace", ["wörkspace", "work\nspace", "work\x00space"])
    def test_a_codex_workspace_id_a_header_cannot_carry_is_a_request_error_naming_only_the_header(
        self, no_requests_client: httpx2.Client, tmp_path: Path, now: datetime, workspace: str
    ) -> None:
        codex_auth_file(tmp_path / "auth.json", account_id=workspace)

        with pytest.raises(ProviderRequestError) as raised:
            CodexSource(tmp_path, no_requests_client).read_quota(now)

        assert str(raised.value) == (
            "Codex usage API request not sent: ChatGPT-Account-Id holds characters a header cannot carry"
        )


class TestCredentialFilesThatFailValidationAreNotKeptByTheError:
    def test_a_codex_auth_file(self, no_requests_client: httpx2.Client, tmp_path: Path, now: datetime) -> None:
        codex_auth_file(tmp_path / "auth.json", access_token="")

        with pytest.raises(CredentialUnavailableError, match="no usable ChatGPT token") as raised:
            CodexSource(tmp_path, no_requests_client).read_quota(now)

        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__

    def test_a_grok_auth_file(self, tmp_path: Path) -> None:
        (tmp_path / "auth.json").write_text(json.dumps({"entry": {"user_id": 7, "key": "synthetic-grok-token"}}))

        with pytest.raises(ProviderSchemaError, match="Grok auth file has an unexpected shape") as raised:
            GrokSource(tmp_path, never_asked()).current_identity()

        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__
