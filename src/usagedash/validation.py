"""Shared pydantic base, field types, and error boundary for payloads this package reads but does not control.

Vendor payloads and CLI files gain fields in any release, so unknown fields are ignored and only the
fields actually used are declared. This is the single definition every provider and scanner imports.
"""

import math
import re
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
)

from usagedash.errors import ProviderSchemaError
from usagedash.models import MAX_WINDOW_SECONDS
from usagedash.timeutil import MAX_EPOCH_SECONDS, parse_iso

MILLISECONDS_PER_SECOND = 1000
FIRST_VISIBLE_ASCII = "!"
LAST_VISIBLE_ASCII = "~"
MASKED_LOCATION_PART = "*"
"""Stands in an error's location for a part that is not a declared field name, such as a payload's own key."""
_DECIMAL_TEXT = re.compile(r"-?[0-9]{1,30}(\.[0-9]{1,30})?")


def _blank_to_none(value: object) -> object:
    return None if isinstance(value, str) and value.strip() == "" else value


def _instant_from_text(value: object) -> datetime:
    """Read ISO-8601 text that carries an offset as a UTC datetime.

    Why pydantic's own datetime parsing is not used: it also reads a bare number, and text holding one,
    as seconds since 1970, so a payload whose shape changed would become a date instead of an error.
    The messages are fixed text because the standard parser's own message quotes what it was given.
    """
    if not isinstance(value, str):
        raise ValueError("must be ISO-8601 text")
    try:
        return parse_iso(value)
    except ValueError:
        raise ValueError("must be an ISO-8601 timestamp with an offset, expressible in UTC") from None


def _decimal_text_as_number(value: object) -> object:
    """Convert text that is a plain decimal number; leave anything else for the number check to refuse."""
    if isinstance(value, str) and _DECIMAL_TEXT.fullmatch(value):
        return float(value)
    return value


def _visible_ascii(value: str) -> str:
    if not all(FIRST_VISIBLE_ASCII <= character <= LAST_VISIBLE_ASCII for character in value):
        raise ValueError("must contain only visible ASCII characters")
    return value


UsedPercent = Annotated[float, Field(ge=0)]
"""A usage percentage. Providers may report more than 100 for an overrun, never less than zero."""

EpochSeconds = Annotated[float, Field(ge=0, le=MAX_EPOCH_SECONDS)]
"""Unix seconds within the range a datetime can represent."""

EpochMilliseconds = Annotated[float, Field(ge=0, le=MAX_EPOCH_SECONDS * MILLISECONDS_PER_SECOND)]
"""Unix milliseconds within the range a datetime can represent."""

WindowSeconds = Annotated[int, Field(ge=0, le=MAX_WINDOW_SECONDS)]
"""How long a rate-limit window lasts, within what a quota window can hold."""

DecimalAmount = Annotated[float, BeforeValidator(_decimal_text_as_number)]
"""An amount a provider may send as a number or as plain decimal text, as Codex sends its credit balance.

This is the one place text is read as a number, and only text that is nothing but digits, a sign, and
one decimal point.
"""

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""A provider's id for an account or person: surrounding whitespace removed, and never blank."""

OptionalIdentifier = Annotated[Identifier | None, BeforeValidator(_blank_to_none)]
"""An id that may be absent. A blank string counts as absent, so a fallback id can be used instead."""

AccessToken = Annotated[str, StringConstraints(min_length=1), AfterValidator(_visible_ascii)]
"""A bearer token: at least one character, all visible ASCII, so it can be sent in a header exactly as stored."""

Instant = Annotated[datetime, BeforeValidator(_instant_from_text)]
"""A timestamp given as ISO-8601 text with an offset, normalized to UTC so later conversion cannot overflow."""


class LenientModel(BaseModel):
    """A model that is lenient about what it does not use and strict about what it does.

    Unknown fields are ignored, and a field is accepted under its name or its alias.

    A declared field is never converted from another type. A number must be a number, a flag a real
    boolean, text real text. pydantic's default mode would read ``true`` as 1 percent, ``"yes"`` as a
    set flag, and ``12345`` as a date in 1970; each of those is a payload whose shape changed, and has
    to be reported as that instead of being turned into a reading. An integer is still a valid value
    for a ``float`` field. The two deliberate exceptions are their own types: ``Instant`` reads
    ISO-8601 text, and ``DecimalAmount`` also reads plain decimal text.

    Infinity and NaN are refused for every number: Python's JSON parser accepts them, and one stored
    non-finite percentage would break every later comparison and the API's JSON output.

    Validation errors never include the input. These models read credential files and keychain items,
    and an error that quoted the surrounding object would carry its tokens into any traceback.
    """

    model_config = ConfigDict(
        extra="ignore", populate_by_name=True, strict=True, allow_inf_nan=False, hide_input_in_errors=True
    )


def summarize_validation_error(error: ValidationError) -> str:
    """Name the first failing field and why, in this package's own words.

    The summary goes into error messages that are stored and shown, so nothing in it may be text the
    payload supplied. The value that failed is never echoed. The location is the other place such text
    can come from: for a dictionary in the payload it holds the key, which is the provider's and not
    ours. So a part of the location is shown only if it is an index or the name or alias of a field
    declared on a ``LenientModel``; any other part is shown as ``MASKED_LOCATION_PART``.
    """
    first = error.errors(include_input=False)[0]
    declared = _declared_field_names()
    location = ".".join(
        str(part) if isinstance(part, int) or part in declared else MASKED_LOCATION_PART for part in first["loc"]
    )
    return f"{location}: {first['msg']}"


def _declared_field_names() -> frozenset[str]:
    """Return every field name and alias declared on ``LenientModel`` and the models derived from it.

    Collected on each call, because models are declared by whichever modules have been imported so far.
    """
    names: set[str] = set()
    pending: list[type[LenientModel]] = [LenientModel]
    while pending:
        model = pending.pop()
        pending.extend(model.__subclasses__())
        for name, field in model.model_fields.items():
            names.add(name)
            if field.alias is not None:
                names.add(field.alias)
    return frozenset(names)


def is_finite_number(value: object) -> bool:
    """Report whether ``value`` is an int or float (not a bool) that a float can hold and is not NaN.

    An integer too large for a float makes ``math.isfinite`` raise ``OverflowError``; that is not a
    usable number either.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@contextmanager
def schema_boundary(what: str) -> Generator[None]:
    """Turn whatever interpreting a vendor payload raises into ``ProviderSchemaError``.

    A payload is untrusted input, and Python's JSON parser hands over numbers no arithmetic can use (an
    integer of 400 digits, a date in year 9999). Field validation catches the cases that were thought
    of; this catches the rest, so a payload can only ever fail as a schema error and never as an
    unexpected exception. ``what`` names the payload in the message.

    The error is raised with no cause. The message is built to hold no text the payload supplied, and
    the underlying error is not: a validation error prints its whole location, payload keys included,
    and a value error's message is whatever its raiser wrote. A cause is printed with every traceback,
    so chaining one would put that text back.

    Raises:
        ProviderSchemaError: if the block raises a validation, value, arithmetic, or recursion error.
    """
    try:
        yield
    except ValidationError as error:
        raise ProviderSchemaError(f"{what} has an unexpected shape: {summarize_validation_error(error)}") from None
    except (ArithmeticError, ValueError, RecursionError) as error:
        raise ProviderSchemaError(f"{what} could not be interpreted: {type(error).__name__}") from None
