import pytest

from usagedash.errors import (
    ConfigError,
    CredentialExpiredError,
    CredentialUnavailableError,
    ProviderRequestError,
    ProviderSchemaError,
    StoreError,
    UsageDashError,
)


class TestErrorTypes:
    @pytest.mark.parametrize(
        "error_type",
        [
            ConfigError,
            CredentialExpiredError,
            CredentialUnavailableError,
            ProviderRequestError,
            ProviderSchemaError,
            StoreError,
        ],
    )
    def test_every_deliberate_error_can_be_caught_as_the_package_s_base_error(
        self, error_type: type[UsageDashError]
    ) -> None:
        with pytest.raises(UsageDashError, match="what went wrong"):
            raise error_type("what went wrong")

    def test_the_error_types_are_distinct_so_a_caller_can_tell_them_apart(self) -> None:
        with pytest.raises(CredentialExpiredError):
            raise CredentialExpiredError("expired")
        assert not issubclass(CredentialExpiredError, CredentialUnavailableError)
        assert not issubclass(ProviderSchemaError, ProviderRequestError)

    def test_a_request_error_carries_the_status_and_the_wait_the_provider_asked_for(self) -> None:
        error = ProviderRequestError("rate limited", status_code=429, retry_after_seconds=120.0)

        assert (str(error), error.status_code, error.retry_after_seconds) == ("rate limited", 429, 120.0)

    def test_a_request_error_without_a_response_has_no_status_or_wait(self) -> None:
        error = ProviderRequestError("connection refused")

        assert (error.status_code, error.retry_after_seconds) == (None, None)
