"""Read the standard short-prompt rates out of each vendor's pricing page, given as markdown.

The page text is untrusted input. Each reader finds one table by its column headings, reads only
cells of the shape it expects, and refuses the page otherwise: a page whose layout changed is a
``ProviderSchemaError``, never a guess. No error here quotes the page.

What each reader takes, as seen on 2026-11-10:

- Claude: the table headed ``Name | Input | Output | 5m writes | 1h writes | Hits and refreshes``.
  The cache write is the 5-minute price. A row that continues the model above it (Haiku 5.5's
  price above 100,000 prompt tokens) starts with a price and is not read. A model is named
  ``Claude Fable 5.2`` and stored as ``claude-fable-5-2``.
- OpenAI: every table under the tab label ``Standard`` that has a ``Model`` column and no
  ``Modality`` or ``Training`` column; one of those without ``Input`` and ``Output`` refuses the
  page. The cache write is the 30-minute price. Batch, Fast, and Ultrafast tables carry their own
  labels and are not read.
- xAI: the text table headed ``Input | Cached | Output | Input | Cached | Output``. The first
  three prices are the short-context ones. It publishes no cache-write price.

Only two kinds of row are passed over: Claude's ``Additional models`` label, and its longer-prompt
row directly under a model. Every other row must have one cell for each column and name a model, a
heading may not repeat, and a table line must be delimited at both ends, or the page is refused:
passing over anything else would leave a model's rate as it was while the page was reported as read.
What no reader can tell is a model the page leaves out altogether; that is a model the vendor no
longer lists, and it keeps the rate it has.
"""

import re
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from usagedash.errors import ProviderSchemaError
from usagedash.models import Provider
from usagedash.pricing import Rates, rates_from_usd

MAX_PAGE_CHARACTERS = 2_000_000
"""The longest page read. The three pages are under 40,000 characters each."""
MAX_PLAUSIBLE_RATE = 10_000 * 1_000_000
"""Ten thousand dollars a million tokens, in microdollars: no listed model is within a tenth of it."""

_CELL_BREAK = re.compile(r"(?<!\\)\|")
_SEPARATOR_CELL = re.compile(r":?-{3,}:?")
# Why the amount may not run on into a digit, a point, or a comma: "$5.0000001" and "$1,000" would
# otherwise be read as their first part. Six decimals is one microdollar, the finest price stored.
_DOLLARS = re.compile(r"\$([0-9]{1,6}(?:\.[0-9]{1,6})?)(?![0-9.,])")
_BLANK_PRICE = frozenset({"", "-"})
_LINK_TEXT = re.compile(r"\[([^\]]+)\]\(")
_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9.-]{0,62}")
_CLAUDE_NAME = re.compile(r"Claude ([A-Z][a-z]+) ([0-9]{1,2}(?:\.[0-9]{1,2})?)(?![0-9.])")

_CLAUDE_HEADER = ("Name", "Input", "Output", "5m writes", "1h writes", "Hits and refreshes")
_CLAUDE_SECTION_LABELS = frozenset({"Additional models"})
_PRIVATE_USE = "Co"
"""The Unicode category of a character a font assigns its own glyph to, as an icon font does."""
_CLAUDE_LONG_PROMPT = re.compile(r"\$[0-9.]+\s*/\s*MTok\s*for prompts over ")
_XAI_HEADER = ("Input", "Cached", "Output", "Input", "Cached", "Output")
_XAI_ROW_CELLS = 8
_OPENAI_LABEL = "Standard"
_OPENAI_MODEL_COLUMN = "Model"
_OPENAI_REQUIRED = frozenset({_OPENAI_MODEL_COLUMN, "Input", "Output"})
_OPENAI_OTHER_PRODUCTS = frozenset({"Modality", "Training"})
_OPENAI_CACHE_READ = "Cached input"
_OPENAI_CACHE_WRITE = "30m write"


@dataclass(frozen=True, slots=True)
class _Table:
    """One markdown table: the line of text before it, and its rows without the separator rows."""

    label: str
    rows: tuple[tuple[str, ...], ...]


def _cells(line: str) -> tuple[str, ...] | None:
    """Return a table line's cells, or None when the line is not a table line."""
    text = line.strip()
    if len(text) < 2 or not text.startswith("|") or not text.endswith("|"):
        return None
    return tuple(cell.strip() for cell in _CELL_BREAK.split(text[1:-1]))


def _is_broken_table_line(text: str, follows_a_table_row: bool) -> bool:
    """Report whether a line that is not a table line looks like one that lost part of itself.

    That is a line delimited at one end only, or a line holding a cell break directly under a table
    row. Read as ordinary text it would end the table there, and the rows from it on would be left
    out of a page that was then reported as read.
    """
    if text.startswith("|") or (text.endswith("|") and not text.endswith("\\|")):
        return True
    return follows_a_table_row and _CELL_BREAK.search(text) is not None


def _tables(markdown: str) -> Iterator[_Table]:
    """Yield each table of the page.

    Raises:
        ProviderSchemaError: if a line is a table line that lost a delimiter (``_is_broken_table_line``).
    """
    label = ""
    rows: list[tuple[str, ...]] = []
    for line in [*markdown.split("\n"), ""]:
        cells = _cells(line)
        if cells is not None:
            if not all(_SEPARATOR_CELL.fullmatch(cell) for cell in cells):
                rows.append(cells)
            continue
        if _is_broken_table_line(line.strip(), bool(rows)):
            raise ProviderSchemaError("a pricing page has a table row that is not delimited at both ends")
        if rows:
            yield _Table(label, tuple(rows))
            rows = []
        if line.strip():
            label = line.strip()


def _price(cell: str) -> str | None:
    """Return the dollar amount a price cell starts with, or None for the vendor's blank.

    Raises:
        ProviderSchemaError: if the cell is neither a blank nor text that starts with a dollar amount.
    """
    if cell in _BLANK_PRICE:
        return None
    amount = _DOLLARS.match(cell)
    if amount is None:
        raise ProviderSchemaError("a pricing page has a price cell that is not a dollar amount")
    return amount.group(1)


def _rates(input_cell: str, cache_read_cell: str, cache_write_cell: str, output_cell: str) -> Rates:
    """Build one model's rates from its cells and refuse a rate no vendor would publish.

    Raises:
        ProviderSchemaError: if input or output is blank, a cell is not a price, a price is zero or
            above ``MAX_PLAUSIBLE_RATE``, or a cache hit costs more than input.
    """
    input_usd, output_usd = _price(input_cell), _price(output_cell)
    if input_usd is None or output_usd is None:
        raise ProviderSchemaError("a pricing page lists a model with no input or output price")
    # Why this cannot be refused as a price: ``_price`` passes only a non-negative amount of at most
    # six decimals, which is a whole number of microdollars.
    rates = rates_from_usd(input_usd, _price(cache_read_cell), _price(cache_write_cell), output_usd)
    stated = [rate for rate in (rates.input, rates.cache_read, rates.cache_write, rates.output) if rate is not None]
    if any(not 0 < rate <= MAX_PLAUSIBLE_RATE for rate in stated):
        raise ProviderSchemaError("a pricing page has a price that is zero or implausibly large")
    if rates.cache_read is not None and rates.input is not None and rates.cache_read > rates.input:
        raise ProviderSchemaError("a pricing page prices a cache hit above input")
    return rates


def _collected(models: Sequence[tuple[str, Rates]]) -> dict[str, Rates]:
    """Return the models by id.

    Raises:
        ProviderSchemaError: if none was read, or one model is listed with two different rates.
    """
    rates_by_model: dict[str, Rates] = {}
    for model, rates in models:
        if rates_by_model.setdefault(model, rates) != rates:
            raise ProviderSchemaError("a pricing page lists one model with two different rates")
    if not rates_by_model:
        raise ProviderSchemaError("a pricing page's rate table lists no model")
    return rates_by_model


def _is_section_label(row: tuple[str, ...], labels: frozenset[str]) -> bool:
    """Report whether a row is one of the one-cell labels this vendor's table is known to hold.

    Why by its text and not by its width alone: a model row that lost every price cell is one cell too.
    An icon drawn from a font's private-use characters is not part of the text: the live Claude label
    is U+F003 followed by ``Additional models`` (seen 2026-11-10).
    """
    if len(row) != 1:
        return False
    text = "".join(character for character in row[0] if unicodedata.category(character) != _PRIVATE_USE)
    return text.strip() in labels


def _require_width(row: tuple[str, ...], width: int) -> None:
    """Refuse a model row that does not have one cell per column.

    Why refused and not skipped: a row that lost or gained a cell is a model whose rate would
    silently stay as it was while the page was reported as read.

    Raises:
        ProviderSchemaError: if the row has another number of cells.
    """
    if len(row) != width:
        raise ProviderSchemaError("a pricing page has a rate row without one cell for each column")


def _rows_after(table: _Table, header: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return table.rows[table.rows.index(header) + 1 :]


def _only_table_with(markdown: str, header: tuple[str, ...]) -> _Table:
    """Return the one table that has ``header`` as a row.

    Raises:
        ProviderSchemaError: if no table has it, or more than one does.
    """
    found = [table for table in _tables(markdown) if header in table.rows]
    if len(found) != 1:
        raise ProviderSchemaError("a pricing page does not have exactly one table with the expected rate columns")
    return found[0]


def read_claude_rates(markdown: str) -> dict[str, Rates]:
    """Read Claude's standard rates. See the module docstring for what is taken.

    Raises:
        ProviderSchemaError: if the page does not have the expected table, or a row of it does not
            name a model or state its prices.
    """
    models: list[tuple[str, Rates]] = []
    follows_a_model = False
    for row in _rows_after(_only_table_with(markdown, _CLAUDE_HEADER), _CLAUDE_HEADER):
        if _is_section_label(row, _CLAUDE_SECTION_LABELS):
            follows_a_model = False
            continue
        # Why only under a model, at one cell fewer, and saying so: a model row that lost its name
        # also starts with a price, and must be refused, not taken for the longer-prompt row.
        if follows_a_model and len(row) == len(_CLAUDE_HEADER) - 1 and _CLAUDE_LONG_PROMPT.match(row[0]):
            follows_a_model = False
            continue
        follows_a_model = True
        _require_width(row, len(_CLAUDE_HEADER))
        name = _CLAUDE_NAME.search(row[0])
        if name is None:
            raise ProviderSchemaError("the Claude pricing page has a rate row that names no model")
        model = f"claude-{name.group(1).lower()}-{name.group(2).replace('.', '-')}"
        models.append((model, _rates(row[1], row[5], row[3], row[2])))
    return _collected(models)


def _openai_header(table: _Table) -> tuple[str, ...] | None:
    """Return the column-heading row of a standard text-token table, or None for any other table.

    A table under the standard label is a model table when a row of it has a ``Model`` column. One
    for another product (images, audio, fine-tuning) is not read.

    Raises:
        ProviderSchemaError: if a standard model table lacks an input or output column, or repeats a
            heading. Skipping it
            would store the other tables' rates and report the page as read.
    """
    if table.label != _OPENAI_LABEL:
        return None
    for row in table.rows:
        if _OPENAI_MODEL_COLUMN not in row:
            continue
        if _OPENAI_OTHER_PRODUCTS & set(row):
            return None
        # Why a repeated heading is refused: the later column would be read under the earlier one's name.
        if len(set(row)) != len(row) or not set(row) >= _OPENAI_REQUIRED:
            raise ProviderSchemaError("the OpenAI pricing page has a standard rate table without the expected columns")
        return row
    return None


def read_openai_rates(markdown: str) -> dict[str, Rates]:
    """Read OpenAI's standard rates. See the module docstring for what is taken.

    Raises:
        ProviderSchemaError: if the page has no standard text-token table, or a row of one does not
            name a model or state its prices.
    """
    models: list[tuple[str, Rates]] = []
    for table in _tables(markdown):
        header = _openai_header(table)
        if header is None:
            continue
        column = {name: index for index, name in enumerate(header)}
        for row in _rows_after(table, header):
            _require_width(row, len(header))
            model = row[column["Model"]]
            if _MODEL_ID.fullmatch(model) is None:
                raise ProviderSchemaError("the OpenAI pricing page has a rate row that names no model")
            cache_read = row[column[_OPENAI_CACHE_READ]] if _OPENAI_CACHE_READ in column else ""
            cache_write = row[column[_OPENAI_CACHE_WRITE]] if _OPENAI_CACHE_WRITE in column else ""
            models.append((model, _rates(row[column["Input"]], cache_read, cache_write, row[column["Output"]])))
    return _collected(models)


def read_xai_rates(markdown: str) -> dict[str, Rates]:
    """Read xAI's short-context text rates. See the module docstring for what is taken.

    Raises:
        ProviderSchemaError: if the page does not have the expected table, or a row of it does not
            name a model or state its prices.
    """
    models: list[tuple[str, Rates]] = []
    for row in _rows_after(_only_table_with(markdown, _XAI_HEADER), _XAI_HEADER):
        _require_width(row, _XAI_ROW_CELLS)
        link = _LINK_TEXT.match(row[0])
        if link is None or _MODEL_ID.fullmatch(link.group(1)) is None:
            raise ProviderSchemaError("the xAI pricing page has a rate row that names no model")
        models.append((link.group(1), _rates(row[2], row[3], "", row[4])))
    return _collected(models)


@dataclass(frozen=True, slots=True)
class RatePage:
    """Where one vendor publishes its rates, and the reader for that page."""

    url: str
    read: Callable[[str], dict[str, Rates]]


RATE_PAGES: Mapping[Provider, RatePage] = {
    Provider.CLAUDE: RatePage("https://platform.claude.com/docs/en/about-claude/pricing", read_claude_rates),
    Provider.CODEX: RatePage("https://developers.openai.com/api/docs/pricing", read_openai_rates),
    Provider.GROK: RatePage("https://docs.x.ai/developers/pricing", read_xai_rates),
}


def read_rates(provider: Provider, markdown: str) -> dict[str, Rates]:
    """Read one vendor's rates from its pricing page.

    Raises:
        ProviderSchemaError: if the page is longer than ``MAX_PAGE_CHARACTERS`` or its reader refuses it.
    """
    if len(markdown) > MAX_PAGE_CHARACTERS:
        raise ProviderSchemaError("a pricing page is longer than any pricing page read so far")
    return RATE_PAGES[provider].read(markdown)
