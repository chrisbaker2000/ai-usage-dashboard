"""UTC timestamp helpers. Every stored timestamp uses one format so SQL string comparison orders correctly.

Timestamps come from provider payloads and session files, so a helper given one it cannot handle raises
``ValueError`` and nothing else. A date at the edge of the calendar makes datetime arithmetic raise
``OverflowError``; that is converted here, once, so no caller has to know about it.
"""

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

Clock = Callable[[], datetime]

HOUR_BUCKET_FORMAT = "%Y-%m-%dT%H"
MAX_EPOCH_SECONDS = 253_402_300_799
"""The last second of year 9999 in UTC, the latest instant a datetime can hold."""
SECONDS_THAT_ROUND_UP = 30
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_EXCESS_FRACTION = re.compile(r"[.,](\d{7,})")


def utc_now() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(UTC)


def to_iso(moment: datetime) -> str:
    """Serialize an aware datetime as second-precision UTC ISO-8601.

    Raises:
        ValueError: if ``moment`` is naive or cannot be expressed in UTC.
    """
    return in_utc(moment).isoformat(timespec="seconds")


def _require_aware(moment: datetime) -> None:
    """Refuse a naive datetime.

    A naive value has no defined instant. Python would convert one using this machine's time zone, so
    the same value would land in a different hour, or a different day, on a different machine.

    Raises:
        ValueError: if ``moment`` carries no UTC offset.
    """
    if moment.utcoffset() is None:
        raise ValueError("a naive datetime has no defined instant")


def in_utc(moment: datetime) -> datetime:
    """Convert an aware datetime to UTC.

    Every helper here that does arithmetic on a datetime converts it first. Arithmetic on a datetime in
    a zone with clock changes is done on its wall-clock fields, so adding a minute across a change, or
    to the second occurrence of a repeated hour, lands on a different instant than intended.

    Raises:
        ValueError: if ``moment`` is naive, or its instant falls outside the years a datetime can
            represent once converted.
    """
    _require_aware(moment)
    try:
        return moment.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("datetime cannot be expressed in UTC") from error


def _fraction_fits_in_a_microsecond(value: str) -> bool:
    """Report whether every fractional second past a microsecond is absent or all zero.

    A datetime drops those digits, including a fraction carried on the offset. A non-zero remainder
    would make two different times compare equal. The first fraction can be zeros while a later one
    is not, so every fraction is checked.
    """
    return all(int(match.group(1)[6:]) == 0 for match in _EXCESS_FRACTION.finditer(value))


def parse_iso(value: str, *, exact: bool = False) -> datetime:
    """Parse an ISO-8601 timestamp that carries an offset and normalize it to UTC.

    ``exact`` refuses a fraction a datetime cannot hold. Without it, those digits are dropped, which
    is harmless for a reset time and wrong for a credential identity.

    Raises:
        ValueError: if the text is not ISO-8601, carries no offset, or cannot be expressed in UTC.
            With ``exact``, also if the fraction is finer than a microsecond and not all zeros past it.
    """
    if exact and not _fraction_fits_in_a_microsecond(value):
        raise ValueError("timestamp is finer than a microsecond")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no UTC offset")
    return in_utc(parsed)


def from_epoch_seconds(seconds: float) -> datetime:
    """Convert Unix epoch seconds to an aware UTC datetime.

    Raises:
        ValueError: if ``seconds`` is not finite or is outside the years a datetime can represent.
    """
    # Why only a range check: NaN fails every comparison and infinity is outside the range, so both are
    # refused here. Testing finiteness first would convert an integer too large for a float and raise
    # OverflowError instead. The upper bound is the start of the second after the last one, so that
    # every fraction of the last second is inside.
    if not 0 <= seconds < MAX_EPOCH_SECONDS + 1:
        raise ValueError("epoch seconds out of range")
    return datetime.fromtimestamp(seconds, UTC)


def epoch_microseconds(moment: datetime) -> int:
    """Return an aware datetime as whole microseconds since the Unix epoch, exactly.

    Most stored timestamps are ISO text to the second. This is the full-precision form, for times that
    are compared with an instant that can fall anywhere inside a second.

    Raises:
        ValueError: if ``moment`` is naive.
    """
    _require_aware(moment)
    return (moment - _EPOCH) // timedelta(microseconds=1)


def from_epoch_microseconds(microseconds: int) -> datetime:
    """Convert whole microseconds since the Unix epoch to an aware UTC datetime, exactly.

    The inverse of ``epoch_microseconds`` for every instant a UTC datetime can hold.

    Raises:
        ValueError: if the instant is outside the years a datetime can represent.
    """
    try:
        return _EPOCH + timedelta(microseconds=microseconds)
    except OverflowError as error:
        raise ValueError("epoch microseconds out of range") from error


def round_to_minute(moment: datetime) -> datetime:
    """Round to the nearest minute so sub-second jitter in a reported reset time does not look like a change.

    The result is in UTC whatever offset ``moment`` was given in.

    Raises:
        ValueError: if ``moment`` is naive, cannot be expressed in UTC, or rounding up would pass the
            last minute a datetime can represent.
    """
    moment = in_utc(moment)
    floored = moment.replace(second=0, microsecond=0)
    if moment.second < SECONDS_THAT_ROUND_UP:
        return floored
    try:
        return floored + timedelta(minutes=1)
    except OverflowError as error:
        raise ValueError("datetime is too close to the end of the calendar to round up") from error


def whole_second_not_before(moment: datetime) -> datetime:
    """Return ``moment`` if it falls on a whole second, and otherwise the next whole second, in UTC.

    For a time that is stored to the second and marks when something began to hold: cut down to the
    second, it would say the thing held before it did.

    Raises:
        ValueError: if ``moment`` is naive, cannot be expressed in UTC, or the next whole second is past
            the last instant a datetime can represent.
    """
    moment = in_utc(moment)
    if moment.microsecond == 0:
        return moment
    try:
        return moment.replace(microsecond=0) + timedelta(seconds=1)
    except OverflowError as error:
        raise ValueError("datetime is too close to the end of the calendar to round up") from error


def hour_bucket(moment: datetime) -> str:
    """Return the UTC hour bucket key (``YYYY-MM-DDTHH``) containing ``moment``.

    The year is always four digits. ``strftime("%Y")`` leaves it to the C library, and glibc writes a
    year before 1000 with fewer, which ``parse_hour_bucket`` refuses and which sorts out of order.

    Raises:
        ValueError: if ``moment`` is naive or cannot be expressed in UTC.
    """
    utc = in_utc(moment)
    return f"{day_key(utc)}T{utc.hour:02d}"


def parse_hour_bucket(bucket: str) -> datetime:
    """Return the UTC start instant of an hour bucket key."""
    return datetime.strptime(bucket, HOUR_BUCKET_FORMAT).replace(tzinfo=UTC)


def day_key(moment: datetime) -> str:
    """Return the UTC calendar day (``YYYY-MM-DD``) containing ``moment``; these keys sort chronologically.

    The year is always four digits, whatever the platform's ``strftime`` would write (see ``hour_bucket``).

    Raises:
        ValueError: if ``moment`` is naive or cannot be expressed in UTC.
    """
    utc = in_utc(moment)
    return f"{utc.year:04d}-{utc.month:02d}-{utc.day:02d}"
