import traceback
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import Field, ValidationError

from usagedash.errors import ProviderSchemaError
from usagedash.timeutil import MAX_EPOCH_SECONDS
from usagedash.validation import (
    AccessToken,
    DecimalAmount,
    EpochMilliseconds,
    EpochSeconds,
    Identifier,
    Instant,
    LenientModel,
    OptionalIdentifier,
    UsedPercent,
    WindowSeconds,
    is_finite_number,
    schema_boundary,
    summarize_validation_error,
)


class _Payload(LenientModel):
    used_percent: UsedPercent = Field(alias="usedPercent")
    resets_at: EpochSeconds | None = Field(default=None, alias="resetsAt")
    expires_at_ms: EpochMilliseconds | None = None
    balance: float | None = None
    window_seconds: WindowSeconds | None = None
    is_enabled: bool = False
    label: str | None = None
    seen_at: Instant | None = None
    amount: DecimalAmount | None = None


class _Envelope(LenientModel):
    items: list[_Payload]


def error_for(model: type[LenientModel], payload: object) -> ValidationError:
    with pytest.raises(ValidationError) as raised:
        model.model_validate(payload)
    return raised.value


class TestLenientModel:
    def test_fields_the_model_does_not_declare_are_ignored(self) -> None:
        parsed = _Payload.model_validate({"usedPercent": 12, "addedInALaterRelease": {"anything": True}})

        assert parsed.used_percent == 12
        assert not hasattr(parsed, "addedInALaterRelease")

    def test_a_field_is_accepted_under_its_alias_or_its_own_name(self) -> None:
        assert _Payload.model_validate({"usedPercent": 5}).used_percent == 5
        assert _Payload.model_validate({"used_percent": 5}).used_percent == 5

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), "Infinity", "-Infinity", "NaN"])
    def test_a_plain_number_field_refuses_infinity_and_nan(self, value: object) -> None:
        error = error_for(_Payload, {"usedPercent": 1, "balance": value})

        assert error.errors()[0]["loc"] == ("balance",)

    def test_a_plain_number_field_accepts_a_negative_number_and_refuses_text_holding_one(self) -> None:
        assert _Payload.model_validate({"usedPercent": 1, "balance": -3.5}).balance == -3.5
        assert error_for(_Payload, {"usedPercent": 1, "balance": "61180.08"}).errors()[0]["loc"] == ("balance",)


class TestUsedPercent:
    @pytest.mark.parametrize("value", [0, 0.0, 42.5, 100, 250])
    def test_accepts_zero_and_anything_above_it(self, value: float) -> None:
        assert _Payload.model_validate({"usedPercent": value}).used_percent == value

    @pytest.mark.parametrize("value", [-0.0001, -1, float("inf"), float("nan"), "1e999", "lots", None])
    def test_refuses_a_negative_non_finite_or_non_numeric_percentage(self, value: object) -> None:
        assert error_for(_Payload, {"usedPercent": value}).errors()[0]["loc"] == ("usedPercent",)


class TestEpochTimes:
    @pytest.mark.parametrize("value", [0, 1_791_036_000, 1_791_036_000.5, MAX_EPOCH_SECONDS])
    def test_seconds_within_what_a_datetime_can_hold_are_accepted(self, value: float) -> None:
        assert _Payload.model_validate({"usedPercent": 1, "resetsAt": value}).resets_at == value

    @pytest.mark.parametrize("value", [-1, MAX_EPOCH_SECONDS + 1, 1e300, float("inf"), float("nan")])
    def test_seconds_outside_that_range_are_refused(self, value: float) -> None:
        assert error_for(_Payload, {"usedPercent": 1, "resetsAt": value}).errors()[0]["loc"] == ("resetsAt",)

    def test_milliseconds_use_the_same_range_scaled_by_a_thousand(self) -> None:
        last = MAX_EPOCH_SECONDS * 1000

        assert _Payload.model_validate({"usedPercent": 1, "expires_at_ms": last}).expires_at_ms == last
        for value in (-1, last + 1, float("inf")):
            error = error_for(_Payload, {"usedPercent": 1, "expires_at_ms": value})
            assert error.errors()[0]["loc"] == ("expires_at_ms",)


class TestErrorSummary:
    def test_names_the_failing_field_by_its_path_and_the_reason(self) -> None:
        error = error_for(_Envelope, {"items": [{"usedPercent": 1}, {"usedPercent": "lots"}]})

        assert summarize_validation_error(error).startswith("items.1.usedPercent: Input should be a valid number")

    def test_does_not_echo_the_value_that_failed(self) -> None:
        error = error_for(_Payload, {"usedPercent": "synthetic-secret-value"})

        assert "synthetic-secret-value" not in summarize_validation_error(error)

    def test_reports_a_missing_field(self) -> None:
        assert summarize_validation_error(error_for(_Payload, {})) == "usedPercent: Field required"


class _ByProviderKey(LenientModel):
    limits: dict[str, _Payload]
    ceilings: dict[str, int] | None = None


class TestALocationHoldsNoTextThePayloadSupplied:
    """A summary is stored and shown, and a key of a dictionary in a payload is the provider's text."""

    def test_a_key_of_a_dictionary_in_the_payload_is_masked(self) -> None:
        error = error_for(_ByProviderKey, {"limits": {"synthetic-key-from-the-provider": {"usedPercent": "lots"}}})

        summary = summarize_validation_error(error)

        assert summary.startswith("limits.*.usedPercent: Input should be a valid number")
        assert "synthetic-key-from-the-provider" not in summary

    def test_a_key_whose_own_value_failed_is_masked(self) -> None:
        error = error_for(_ByProviderKey, {"limits": {}, "ceilings": {"synthetic key\nwith a line break": "many"}})

        assert summarize_validation_error(error) == "ceilings.*: Input should be a valid integer"

    def test_a_key_that_is_also_a_declared_field_name_is_shown_because_the_word_is_ours(self) -> None:
        error = error_for(_ByProviderKey, {"limits": {"usedPercent": {"usedPercent": "lots"}}})

        assert summarize_validation_error(error).startswith("limits.usedPercent.usedPercent: ")

    def test_an_index_and_a_field_named_by_its_alias_or_by_its_own_name_are_shown(self) -> None:
        by_alias = error_for(_Envelope, {"items": [{"usedPercent": 1}, {"usedPercent": 2, "resetsAt": "soon"}]})
        by_name = error_for(_Envelope, {"items": [{"usedPercent": 1, "window_seconds": "long"}]})

        assert summarize_validation_error(by_alias).startswith("items.1.resetsAt: ")
        assert summarize_validation_error(by_name).startswith("items.0.window_seconds: ")

    def test_a_model_declared_after_an_earlier_summary_has_its_fields_recognized(self) -> None:
        summarize_validation_error(error_for(_Payload, {}))

        class _DeclaredLate(LenientModel):
            newly_declared_count: int

        assert summarize_validation_error(error_for(_DeclaredLate, {})) == "newly_declared_count: Field required"


class _Login(LenientModel):
    access_token: AccessToken
    account_id: OptionalIdentifier = None
    organization: Identifier = "org"
    expires: Instant | None = None


class TestErrorsNeverCarryTheInput:
    """These models read credential files; an error that quoted its input would carry tokens into a traceback."""

    SECRET = "synthetic-refresh-token-value"

    def test_the_error_text_does_not_quote_the_object_a_field_was_missing_from(self) -> None:
        error = error_for(_Login, {"refresh_token": self.SECRET, "accessToken": 5})

        rendered = "".join(traceback.format_exception(error)) + repr(error) + str(error)
        assert "access_token" in rendered
        assert self.SECRET not in rendered

    def test_the_error_text_does_not_quote_a_value_of_the_wrong_type(self) -> None:
        error = error_for(_Login, {"access_token": {"nested": self.SECRET}})

        assert self.SECRET not in "".join(traceback.format_exception(error))

    def test_the_summary_does_not_quote_it_either(self) -> None:
        error = error_for(_Login, {"access_token": {"nested": self.SECRET}})

        assert summarize_validation_error(error) == "access_token: Input should be a valid string"


class TestIdentifiers:
    def test_surrounding_whitespace_is_removed(self) -> None:
        parsed = _Login.model_validate({"access_token": "t", "organization": "  org-a \n", "account_id": " ws-a "})

        assert (parsed.organization, parsed.account_id) == ("org-a", "ws-a")

    @pytest.mark.parametrize("blank", ["", " ", "\t\n"])
    def test_a_required_identifier_cannot_be_blank(self, blank: str) -> None:
        error = error_for(_Login, {"access_token": "t", "organization": blank})

        assert error.errors(include_input=False)[0]["loc"] == ("organization",)

    @pytest.mark.parametrize("absent", [None, "", " ", "\t\n"])
    def test_an_optional_identifier_that_is_blank_counts_as_absent(self, absent: str | None) -> None:
        assert _Login.model_validate({"access_token": "t", "account_id": absent}).account_id is None

    def test_an_optional_identifier_that_is_not_text_is_refused(self) -> None:
        error = error_for(_Login, {"access_token": "t", "account_id": 12345})

        assert error.errors(include_input=False)[0]["loc"] == ("account_id",)


class TestInstants:
    def test_an_offset_timestamp_is_normalized_to_utc(self) -> None:
        parsed = _Login.model_validate({"access_token": "t", "expires": "2026-10-03T10:00:00-04:00"})

        assert parsed.expires == datetime(2026, 10, 3, 14, 0, tzinfo=UTC)
        assert parsed.expires is not None
        assert parsed.expires.utcoffset() == timedelta(0)

    def test_a_timestamp_without_an_offset_is_refused(self) -> None:
        error = error_for(_Login, {"access_token": "t", "expires": "2026-10-03T10:00:00"})

        assert error.errors(include_input=False)[0]["loc"] == ("expires",)

    def test_a_timestamp_that_cannot_be_expressed_in_utc_is_refused_not_an_overflow(self) -> None:
        error = error_for(_Login, {"access_token": "t", "expires": "9999-12-31T23:59:59-05:00"})

        assert error.errors(include_input=False)[0]["msg"] == (
            "Value error, must be an ISO-8601 timestamp with an offset, expressible in UTC"
        )

    @pytest.mark.parametrize("value", [1_791_036_000, 1_791_036_000.5, True, ["2026-10-03T10:00:00Z"]])
    def test_a_timestamp_that_is_not_text_is_refused_not_read_as_seconds_since_1970(self, value: object) -> None:
        error = error_for(_Login, {"access_token": "t", "expires": value})

        assert error.errors(include_input=False)[0]["msg"] == "Value error, must be ISO-8601 text"

    @pytest.mark.parametrize("value", ["1791036000", "yesterday", "", "2026-10-03", "2026-13-45T00:00:00Z"])
    def test_text_that_is_not_an_iso_timestamp_is_refused_without_being_quoted(self, value: str) -> None:
        error = error_for(_Login, {"access_token": "t", "expires": value})

        assert summarize_validation_error(error) == (
            "expires: Value error, must be an ISO-8601 timestamp with an offset, expressible in UTC"
        )

    @pytest.mark.parametrize(
        "value",
        ["2026-10-03T14:00:00Z", "2026-10-03T14:00:00+00:00", "2026-10-03T14:00:00.123456789Z", "2026-10-03 14:00:00Z"],
    )
    def test_the_iso_forms_the_clis_and_providers_write_are_all_read(self, value: str) -> None:
        parsed = _Login.model_validate({"access_token": "t", "expires": value}).expires

        assert parsed is not None
        assert parsed.replace(microsecond=0) == datetime(2026, 10, 3, 14, 0, tzinfo=UTC)


class TestFiniteNumbers:
    @pytest.mark.parametrize("value", [0, -3, 1.5, 10**18, 1e308])
    def test_ordinary_numbers_are_finite(self, value: float) -> None:
        assert is_finite_number(value)

    @pytest.mark.parametrize(
        "value", [True, False, None, "1", [1], float("inf"), float("-inf"), float("nan"), 10**400, -(10**400)]
    )
    def test_everything_else_is_not_including_an_integer_too_large_for_a_float(self, value: object) -> None:
        assert not is_finite_number(value)

    def test_a_percentage_given_as_an_integer_too_large_for_a_float_is_refused(self) -> None:
        assert error_for(_Payload, {"usedPercent": 10**400}).errors(include_input=False)[0]["loc"] == ("usedPercent",)


class TestSchemaBoundary:
    def test_a_block_that_succeeds_is_untouched(self) -> None:
        with schema_boundary("Synthetic payload"):
            value = _Payload.model_validate({"usedPercent": 5}).used_percent

        assert value == 5

    def test_a_validation_failure_names_the_payload_and_the_field(self) -> None:
        with (
            pytest.raises(ProviderSchemaError, match=r"Synthetic payload has an unexpected shape: usedPercent: "),
            schema_boundary("Synthetic payload"),
        ):
            _Payload.model_validate({"usedPercent": "lots"})

    @pytest.mark.parametrize(
        "failure",
        [OverflowError("too large"), ZeroDivisionError("by zero"), ValueError("bad value"), RecursionError("too deep")],
    )
    def test_anything_else_a_payload_can_make_arithmetic_raise_is_a_schema_error_naming_the_kind(
        self, failure: Exception
    ) -> None:
        with pytest.raises(ProviderSchemaError) as raised, schema_boundary("Synthetic payload"):
            raise failure

        assert str(raised.value) == f"Synthetic payload could not be interpreted: {type(failure).__name__}"
        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__

    def test_a_traceback_of_the_schema_error_holds_no_key_the_payload_supplied(self) -> None:
        """The validation error underneath prints its whole location, so it must not be chained."""
        key = "-".join(["synthetic", "key", "from", "the", "provider"])

        with pytest.raises(ProviderSchemaError) as raised, schema_boundary("Synthetic payload"):
            _ByProviderKey.model_validate({"limits": {key: {"usedPercent": "lots"}}})

        assert str(raised.value).startswith("Synthetic payload has an unexpected shape: limits.*.usedPercent: ")
        assert key not in "".join(traceback.format_exception(raised.value))
        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__

    def test_a_traceback_of_the_schema_error_holds_no_message_of_the_error_underneath(self) -> None:
        message = " ".join(["synthetic", "text", "quoted", "from", "the", "payload"])

        with pytest.raises(ProviderSchemaError) as raised, schema_boundary("Synthetic payload"):
            raise ValueError(message)

        assert message not in "".join(traceback.format_exception(raised.value))

    def test_a_schema_error_raised_inside_passes_through_unchanged(self) -> None:
        original = ProviderSchemaError("Synthetic payload contained no windows")

        with pytest.raises(ProviderSchemaError) as raised, schema_boundary("Synthetic payload"):
            raise original

        assert raised.value is original

    @pytest.mark.parametrize("defect", [KeyError("missing"), TypeError("wrong type"), AttributeError("no attribute")])
    def test_a_programming_error_is_not_disguised_as_a_bad_payload(self, defect: Exception) -> None:
        with pytest.raises(type(defect)), schema_boundary("Synthetic payload"):
            raise defect


class TestNothingIsConvertedFromAnotherType:
    """A value of the wrong type means the payload's shape changed, and must be an error, never a reading."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("usedPercent", True),
            ("usedPercent", False),
            ("usedPercent", "50"),
            ("resetsAt", True),
            ("resetsAt", "1791036000"),
            ("expires_at_ms", False),
            ("expires_at_ms", "1791036000000"),
            ("balance", True),
            ("balance", "12"),
            ("window_seconds", True),
            ("window_seconds", "300"),
            ("window_seconds", 300.0),
            ("is_enabled", "false"),
            ("is_enabled", "true"),
            ("is_enabled", "yes"),
            ("is_enabled", 1),
            ("is_enabled", 0),
            ("label", 5),
            ("label", True),
            ("seen_at", 1791036000),
            ("seen_at", 1791036000.5),
            ("seen_at", "1791036000"),
            ("seen_at", True),
            ("amount", True),
        ],
    )
    def test_a_value_of_the_wrong_type_is_refused(self, field: str, value: object) -> None:
        error = error_for(_Payload, {"usedPercent": 1} | {field: value})

        assert error.errors(include_input=False)[0]["loc"] == (field,)

    def test_values_of_the_declared_types_are_accepted_as_they_are(self) -> None:
        parsed = _Payload.model_validate(
            {
                "usedPercent": 12.5,
                "resetsAt": 1_791_036_000,
                "window_seconds": 300,
                "is_enabled": True,
                "label": "Weekly",
                "seen_at": "2026-10-03T14:00:00Z",
            }
        )

        assert (parsed.used_percent, parsed.resets_at, parsed.window_seconds) == (12.5, 1_791_036_000, 300)
        assert (parsed.is_enabled, parsed.label) == (True, "Weekly")
        assert parsed.seen_at == datetime(2026, 10, 3, 14, 0, tzinfo=UTC)

    def test_an_integer_is_a_valid_value_for_a_number_that_may_have_a_fraction(self) -> None:
        parsed = _Payload.model_validate({"usedPercent": 12, "balance": 3})

        assert (parsed.used_percent, parsed.balance) == (12.0, 3.0)
        assert isinstance(parsed.used_percent, float)


class TestDecimalAmounts:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(12, 12.0), (-3.5, -3.5), ("61180.0843600000", 61180.08436), ("-3.5", -3.5), ("12", 12.0), ("0", 0.0)],
    )
    def test_a_number_or_plain_decimal_text_is_read_as_the_number(self, value: object, expected: float) -> None:
        assert _Payload.model_validate({"usedPercent": 1, "amount": value}).amount == expected

    @pytest.mark.parametrize(
        "value",
        [
            "Infinity",
            "NaN",
            "1e5",
            " 12",
            "12 ",
            "12\n",
            "",
            ".5",
            "1.",
            "+1",
            "0x10",
            "1_000",
            "1,000",
            "١٢",
            "1" * 31,
            True,
            [12],
            float("inf"),
        ],
    )
    def test_text_that_is_anything_other_than_a_plain_decimal_number_is_refused(self, value: object) -> None:
        error = error_for(_Payload, {"usedPercent": 1, "amount": value})

        assert error.errors(include_input=False)[0]["loc"] == ("amount",)


class TestAccessTokens:
    @pytest.mark.parametrize(
        "token", ["t", "sk-synthetic_token.with~every!visible#mark", "eyJhbGciOiJub25lIn0.e30.sig"]
    )
    def test_visible_ascii_text_is_a_usable_token(self, token: str) -> None:
        assert _Login.model_validate({"access_token": token}).access_token == token

    @pytest.mark.parametrize(
        "token",
        ["", "tok en", " token", "token ", "tok\nen", "tok\ren", "tok\ten", "tok\x00en", "tok\x7fen", "tökén", "токен"],
    )
    def test_a_token_a_header_could_not_carry_unchanged_is_refused(self, token: str) -> None:
        error = error_for(_Login, {"access_token": token})

        assert error.errors(include_input=False)[0]["loc"] == ("access_token",)

    @pytest.mark.parametrize("token", [12345, True, None, ["token"]])
    def test_a_token_that_is_not_text_is_refused(self, token: object) -> None:
        assert error_for(_Login, {"access_token": token}).errors(include_input=False)[0]["loc"] == ("access_token",)

    def test_the_refusal_never_quotes_the_token(self) -> None:
        error = error_for(_Login, {"access_token": "synthetic-token-with-a\nnewline"})

        rendered = "".join(traceback.format_exception(error)) + summarize_validation_error(error)
        assert "synthetic-token" not in rendered
        assert summarize_validation_error(error) == (
            "access_token: Value error, must contain only visible ASCII characters"
        )
