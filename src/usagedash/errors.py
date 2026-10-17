"""Typed errors so callers can distinguish failure classes without string matching."""


class UsageDashError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigError(UsageDashError):
    """Configuration is missing, malformed, or out of range."""


class StoreError(UsageDashError):
    """The database file cannot be used: it is not a database this code can open, or it was written by a
    schema version this code does not understand."""


class PageAssemblyError(UsageDashError):
    """The dashboard page cannot be assembled: a static file this package ships is not in the shape the
    assembler reads, so the installed package is damaged or was changed without the assembler."""


class SealedBodyError(UsageDashError):
    """A sealed body does not open: it is too short to be one, was sealed under another key or for another
    request or answer, or has been altered."""


class CredentialUnavailableError(UsageDashError):
    """No usable credential exists locally (logged out, file missing, keychain locked)."""


class CredentialExpiredError(UsageDashError):
    """A credential exists but is expired or rejected; the owning CLI must refresh it."""


class ProviderRequestError(UsageDashError):
    """A provider endpoint could not be reached or answered with a non-success status."""

    def __init__(self, message: str, *, status_code: int | None = None, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class ProviderSchemaError(UsageDashError):
    """A provider payload or local file did not match the shape this package understands."""


class NotAJwtError(ProviderSchemaError):
    """A token is not a three-part JWT at all, as opposed to a JWT whose claims cannot be read."""
