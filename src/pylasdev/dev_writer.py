"""DEV (deviation survey) file writer.

Serializes a :class:`~pylasdev.models.DevFile` back to a DEV ASCII file —
the symmetric counterpart of :mod:`pylasdev.dev_reader`.  There is no
normative DEV specification, so the emitted bytes are defined by what the
reader accepts (and by the documented industry conventions summarised in
the module's ``write_dev_file`` docstring):

- exactly ONE column-name header line, in ``column_order`` order;
- one whitespace/delimiter-separated data row per station;
- no title line, no column-count line, no ``sep=`` directive, no
  terminator line, no comment channel.

Every DEV consumer that turns a bare (non-``#``) line into content — this
library's reader, and the DUG/Petrel import paths it models — would read
any of those extra lines as a header or as data.  ``#``-prefixed lines are
the only inert preamble, and :class:`DevFile` carries no field from which
an honest one could be built, so the writer emits none.
"""

from __future__ import annotations

import math
import numbers
import os
import re
import tempfile
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._writer_base import (
    _apply_file_mode,
    _default_new_file_mode,
    _format_index_value,
    _format_number,
    _fsync_directory,
    _validate_precision,
)
from .dev_reader import (
    _DEV_DELETE_CHARS_RE,
    _DEV_NUMERIC_SENTINELS,
    _is_float_token,
    _normalize_dev_column,
)
from .exceptions import DEVWriteError
from .models import DevFile

__all__ = ["write_dev_file"]

# A rendered token that is a bare integer ("12", "-3", "+0") is re-read
# through the reader's comma-thousands recombination and through the
# DUG/headerless detectors as a COUNT or as a leading thousands group.
# Appending ".0" makes every token a decimal number: "12" + "345.6" can
# no longer merge into "12345.6" in a comma-delimited file, and a
# one-column file's first station can no longer be consumed as a column
# count.
_BARE_INTEGER_RE = re.compile(r"^[+-]?\d+$")

# A bare precision specifier (".5") is accepted by the LAS validator but
# raises on integral values ("Precision not allowed in integer format
# specifier"); the LAS writer normalizes it inline.  Same rule here.
_BARE_PRECISION_RE = re.compile(r"^\.(\d+)$")

# Excel's exporter directive ("sep=;") — the reader deletes the first
# content line that matches this shape, so a single-column file named
# "sep=;" would lose its header.
_SEP_DIRECTIVE_RE = re.compile(r"^sep=(.)$", re.IGNORECASE)

# A bare separator rule ("=====", "-----") is non-content to the reader,
# but as the ONLY column name it makes the whole header line vanish.
_SEPARATOR_RULE_RE = re.compile(r"^[=\-_*~]{3,}$")

# Characters the reader treats as a field separator regardless of the
# delimiter the writer emitted (comma/semicolon delimiter detection and
# the tab branch), plus the tab itself.
_SEPARATOR_CHARS: frozenset[str] = frozenset({",", ";", "\t"})

# The DEV format documents neither a terminator nor an encoding; every
# real export observed is CRLF, the reference emulation (petekIO) writes
# LF, and the reader's splitlines() accepts both.  LF is chosen so the
# bytes are identical on every platform — ``os.fdopen(..., newline="")``
# disables the platform's newline translation, exactly as in the LAS path.
_LINE_TERMINATOR = "\n"


@dataclass(frozen=True)
class _EmissionDiagnostics:
    """Warnings collected while the file text is assembled.

    ``write_dev_file`` emits them after assembly so the text (and every
    diagnostic it justifies) is complete before any byte reaches disk.
    """

    # (column name, first colliding value, its emitted token, count)
    sentinel_collisions: tuple[tuple[str, float, str, int], ...]
    # (emitted name, name the reader rewrites it to)
    renamed_columns: tuple[tuple[str, str], ...]
    # columns holding at least one NaN cell
    nan_columns: tuple[str, ...]
    # (index column name, emitted token, source value) — one entry per index
    # column whose emitted token no longer reads back as the model value
    inexact_index: tuple[tuple[str, str, float], ...]


def write_dev_file(
    file_path: str | Path,
    dev_data: DevFile | dict[str, Any],
    encoding: str = "utf-8",
    precision: str = ".8g",
    delimiter: str = " ",
    null_value: float = -999.25,
) -> None:
    """Write DEV (deviation survey) data to a file.

    The output is a *simple*-layout DEV file: exactly one column-name
    header line in ``column_order`` order, then one data row per station.
    It carries no title line, no column-count line, no ``sep=`` directive,
    no terminator line and no comment channel — every non-``#`` line is
    content to a DEV reader, and this library's reader classifies a
    DUG/Petrel/``sep=`` preamble as a header or as data.

    Comment/banner lines (``# WELL NAME:``, ``# WELL KB:``, unit and datum
    statements) are NOT represented in :class:`~pylasdev.models.DevFile`
    and are NOT preserved: a Petrel export's banner does not survive a
    read/write cycle.  This function never fabricates one — a derived
    ``# WELL KB:`` or ``-UNITS`` line is inert to this reader but is acted
    on by external consumers, so a wrong derived datum is worse than an
    absent one.

    The header and every data row are assembled from ``column_order``
    only; the ``columns`` dict's insertion order is never the emission
    authority.  Names are emitted verbatim (no alias normalization), so
    ``AZIM``/``INCL``/``TVDSS`` keep their spelling in the file and re-read
    as the canonical ``AZI``/``INC``/``TVD``; each such column is named in
    a ``UserWarning``.  Read with ``normalize_aliases=False`` to keep the
    emitted spelling on re-read.

    A name the header grammar cannot represent is refused with
    ``DEVWriteError`` naming the column and the reason: an empty or
    whitespace-only name, leading or trailing whitespace, embedded
    whitespace or a control character, any of ``,`` ``;`` tab or the ACTIVE
    ``delimiter`` character, a leading double quote, a header that parses
    entirely as numbers, and — positionally — a ``#``/``*COLUMNS`` prefix in
    the FIRST column or a ``sep=<char>``/separator-rule shape as the ONLY
    column.

    An existing target is REPLACED with :func:`os.replace`; it is never
    opened for writing.  Its own mode is preserved even when it is
    write-protected (a ``0444`` file is replaced and stays ``0444``) —
    write-protect the CONTAINING DIRECTORY when the destination must be
    immutable.  ACLs, extended attributes and the owner/group are NOT
    preserved (the replace installs a new inode).

    Numeric emission: every cell is formatted with ``precision`` (validated
    exactly as :func:`write_las_file` validates it).  ``NaN`` and ``Inf``
    cells are emitted as ``null_value``, because the DEV format has no
    declared-null mechanism and a non-numeric token is not portable (TNO's
    documented reader raises on any row that is not all numeric).  When
    ``null_value`` is one of the reader's numeric sentinels (``-999``,
    ``-999.25``, ``-9999``, ``-9999.25``) those cells re-read as *missing
    data*; any other finite ``null_value`` is emitted verbatim and re-reads
    as that literal NUMBER, indistinguishable from a measurement.  ``Inf``
    cells follow the same rule.  The index column — the first column whose
    alias-normalised name is ``MD``, or position 0 when no column qualifies
    — is formatted with the LAS writer's index formatter and is therefore
    exact at the default ``".8g"`` precision; at an explicitly requested
    precision it follows that precision like every other column, and an
    emitted depth that no longer reads back as the model value produces one
    ``UserWarning`` naming the column, the value and the token.  A finite
    value whose EMITTED TOKEN is one of the reader's numeric sentinels
    (``-999``, ``-999.25``, ``-9999``, ``-9999.25`` — including a value
    rounded into one by the requested precision) cannot be encoded: it is
    emitted as that token, warned about once per column, and reads back as
    missing.  A column that is more than 50 % missing re-reads with the
    reader's "Possible delimiter mismatch" density warning — a data-quality
    diagnostic, not a format error.

    Args:
        file_path: Output file path.  Missing parent directories are
            created.
        dev_data: A :class:`~pylasdev.models.DevFile`, or a legacy ``dict``
            converted with ``DevFile.from_dict`` (alias normalization on).
        encoding: Output text encoding (default UTF-8).
        precision: Numeric format specifier (default ``".8g"``).  A bare
            ``".N"`` is normalized to ``".Ng"``; the ``n`` and ``%``
            format codes are rejected.
        delimiter: Single-character field separator (default a space).  The
            reader auto-detects only space, tab, comma and semicolon, so a
            file written with any other separator must be read with the same
            ``delimiter``.
        null_value: Finite real number emitted for missing cells
            (default ``-999.25``).  Declare one of the reader's numeric
            sentinels (``-999``, ``-999.25``, ``-9999``, ``-9999.25``): the
            DEV format has no declared-null mechanism, so a null outside
            that set is emitted verbatim and re-reads as that NUMBER, not
            as missing data.

    Raises:
        DEVWriteError: For every failure mode — a ``dev_data`` that is
            neither ``DevFile`` nor ``dict``, an invalid ``precision``, a
            ``null_value`` that is not a finite real, an invalid
            ``delimiter``, a column name the DEV header grammar cannot
            represent, a ``column_order`` that does not match ``columns``,
            a degenerate model (no columns, or zero-row columns), or an
            I/O failure.  The destination file is left untouched on
            failure.
    """
    try:
        _write_dev_file(file_path, dev_data, encoding, precision, delimiter, null_value)
    except DEVWriteError:
        raise
    except Exception as exc:  # every failure of the public API is a DEVWriteError
        raise DEVWriteError(f"Failed to write DEV file {file_path!r}: {exc}") from exc


def _write_dev_file(
    file_path: str | Path,
    dev_data: DevFile | dict[str, Any],
    encoding: str,
    precision: str,
    delimiter: str,
    null_value: float,
) -> None:
    """Validation bundle, content assembly, diagnostics and atomic write."""
    # Stacklevel note for warnings emitted HERE: the frame stack is
    # user -> write_dev_file -> this function -> warnings.warn, so
    # stacklevel=3 points at the user's call.  A warning emitted one
    # function deeper (in _emit_diagnostics) needs stacklevel=4.
    dev = _coerce_dev_file(dev_data)
    resolved_precision = _resolve_precision(precision)
    resolved_null = _resolve_null_value(null_value)
    resolved_delimiter = _resolve_delimiter(delimiter)

    names = _resolve_names(dev)
    _validate_emitted_names(names, resolved_delimiter)
    _validate_name_collisions(names)

    # --- Degenerate models (warn, then refuse: no silently unreadable file)
    # The DEV reader raises DEVReadError("No data lines found") for a file
    # with a header and no data row, so writing one would produce an
    # artifact this library cannot read back.
    if not dev.columns:
        _warn_degenerate(
            "the DevFile has no columns, so the output would be a header-less "
            "file with no data lines and the DEV reader rejects it"
        )
        raise DEVWriteError(
            "Cannot write a DEV file from a DevFile with no columns: the DEV "
            "reader rejects a file with no data lines ('No data lines found "
            "in DEV file'), so the artifact would be unreadable."
        )

    column_lengths = {len(dev.columns[name]) for name in names}
    if column_lengths == {0}:
        _warn_degenerate(
            f"every column of the DevFile has 0 rows (columns: {names}), so "
            f"the output would carry a header and no data line and the DEV "
            f"reader rejects it"
        )
        raise DEVWriteError(
            f"Cannot write a DEV file from a DevFile whose columns are all "
            f"empty ({names}): the DEV reader rejects a file with no data "
            f"lines ('No data lines found in DEV file')."
        )
    if len(column_lengths) != 1:
        # Reachable only when a column array was resized in place after
        # construction (the equal-length invariant is enforced at every
        # container mutation entry point but not on the stored buffers).
        lengths = {name: len(dev.columns[name]) for name in names}
        raise DEVWriteError(
            f"Cannot write a DEV file from a DevFile with inconsistent column "
            f"lengths: {lengths}.  A DEV file has one value per column per "
            f"station; repair the arrays first."
        )
    row_count = column_lengths.pop()

    text, diagnostics = _build_dev_text(
        dev, names, row_count, resolved_precision, resolved_delimiter, resolved_null
    )
    _emit_diagnostics(dev, diagnostics, resolved_null)
    # C-4 / amended AC-3: an index token that no longer reads back as the model
    # value is a warned condition, never a silent rounding.  The warning is
    # raised HERE, not in _emit_diagnostics, because _warn_inexact_index_dev's
    # stacklevel=4 counts user -> write_dev_file -> _write_dev_file -> helper;
    # one frame deeper it would attribute the warning to this module's own
    # public wrapper.
    for column, token, value in diagnostics.inexact_index:
        _warn_inexact_index_dev(column, token, value, resolved_precision)
    _write_text_atomically(file_path, text, encoding)


def _coerce_dev_file(dev_data: DevFile | dict[str, Any]) -> DevFile:
    """Accept a ``DevFile`` or convert a legacy ``dict`` (AC-1)."""
    if isinstance(dev_data, DevFile):
        return dev_data
    if isinstance(dev_data, dict):
        try:
            return DevFile.from_dict(dev_data)
        except Exception as exc:  # every conversion failure is a write error
            raise DEVWriteError(f"Cannot create DevFile from dict: {exc}") from exc
    raise DEVWriteError(f"write_dev_file expects a DevFile or dict, got {type(dev_data).__name__}")


def _resolve_precision(precision: object) -> str:
    """Validate ``precision`` exactly as the LAS writer does (AC-5).

    The LAS path splits this contract between ``_validate_precision`` (the
    format-code regex plus a WARNING for ``n``/``%``) and an inline
    caller-side normalize/reject.  Calling the helper alone would only warn
    for ``%``/``n`` and would accept a bare ``".5"`` that later raises on an
    integral value, so the caller-side half is replicated here: the hard
    rejection happens BEFORE the helper runs, so no warning is emitted for
    a value that is about to be refused.
    """
    if not isinstance(precision, str):
        raise DEVWriteError(
            f"precision must be a string format specifier such as '.8g', got "
            f"{type(precision).__name__} ({precision!r})"
        )
    if precision[-1:] in ("n", "%"):
        raise DEVWriteError(
            f"Precision format code {precision[-1]!r} in {precision!r} is not "
            f"supported for DEV output.  The 'n' code produces locale-dependent "
            f"decimal separators and grouping; the '%' code multiplies values "
            f"by 100 and appends '%'.  Both corrupt numeric data on re-read.  "
            f"Use 'g', 'f', or 'e' instead."
        )
    try:
        _validate_precision(precision)
    except ValueError as exc:
        raise DEVWriteError(f"Invalid precision format: {exc}") from exc
    return _BARE_PRECISION_RE.sub(r".\1g", precision)


def _resolve_null_value(null_value: object) -> float:
    """Require a finite real ``null_value`` (AC-1 / AC-5).

    ``_format_number`` turns a non-finite null into literal text (``nan``,
    ``inf``), which the DEV format cannot express — the reader maps those
    spellings back to missing data and TNO's documented reader rejects a
    row that is not four numeric values.  A non-finite or non-numeric null
    is therefore refused at the entry point instead of being emitted.
    """
    if isinstance(null_value, bool) or not isinstance(null_value, numbers.Real):
        raise DEVWriteError(
            f"null_value must be a finite real number (e.g. -999.25), got "
            f"{type(null_value).__name__} ({null_value!r})"
        )
    resolved = float(null_value)
    if not math.isfinite(resolved):
        raise DEVWriteError(
            f"null_value must be finite, got {null_value!r}.  A non-finite "
            f"null is emitted as the literal text 'nan'/'inf', which is not a "
            f"numeric DEV value: this library's reader would interpret it as "
            f"missing data and strict consumers reject the row."
        )
    return resolved


def _resolve_delimiter(delimiter: object) -> str:
    """Require a single, non-newline separator the reader does not delete.

    The character class is IMPORTED from the reader (``_DEV_DELETE_CHARS_RE``)
    rather than re-expressed here: the reader deletes those control characters
    from the file before it tokenizes anything, so a file delimited by one of
    them fuses its header and every row into a single token and re-reads as one
    all-NaN column (C-2).  Deriving the rule from the consumer's own predicate
    is what keeps the two sides from drifting — the same reason the sentinel
    set and the alias map are imported.  A ``Cc`` category test would refuse
    the tab (a supported, tested delimiter) while accepting nothing extra:
    ``0x0B``/``0x0C``/``0x1C``-``0x1E`` are ``Cc`` too and re-read cleanly.
    """
    if not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in ("\n", "\r"):
        raise DEVWriteError(
            f"delimiter must be a single character other than a newline "
            f"(e.g. ' ' or ','), got {delimiter!r}"
        )
    if _DEV_DELETE_CHARS_RE.fullmatch(delimiter):
        raise DEVWriteError(
            f"delimiter {delimiter!r} (U+{ord(delimiter):04X}) cannot be used: the "
            f"DEV reader deletes that control character from the file before "
            f"parsing, so the header and every data row fuse into one token and "
            f"the file re-reads as a single all-NaN column.  Use a separator the "
            f"reader keeps, such as ' ', ',', ';' or a tab."
        )
    return delimiter


def _resolve_names(dev: DevFile) -> list[str]:
    """Return the single emission authority: a copy of ``column_order``.

    ``column_order`` is the ordered list; ``columns`` is the container.  A
    DEV writer that iterated ``columns`` would emit the container's
    insertion order and silently permute the file, so the list is copied
    ONCE here and used for both the header and every row.  ``column_order``
    itself is never mutated (no ``sort``/``reverse``/``*=``/slice
    assignment).

    Two reachable desyncs are refused instead of emitted: duplicated
    entries (the state an in-place ``column_order *= 2`` leaves behind
    after its re-wrap raise) and an order/column set mismatch in EITHER
    direction (``DevFile.validate`` reports an orphan order entry but is
    silent about a column that no order entry names).
    """
    names = list(dev.column_order)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise DEVWriteError(
            f"DevFile column_order contains duplicate entries: {duplicates}.  "
            f"Each column may only appear once; the header would name the "
            f"same column twice and the re-read would rename the second to "
            f"'<name>_2'."
        )
    order_set = set(names)
    column_set = set(dev.columns)
    if order_set != column_set:
        problems: list[str] = []
        orphans = sorted(order_set - column_set)
        unordered = sorted(column_set - order_set)
        if orphans:
            problems.append(f"column_order entries with no column: {orphans}")
        if unordered:
            problems.append(f"columns with no column_order entry: {unordered}")
        raise DEVWriteError(
            "DevFile column_order and columns keys do not match ("
            + "; ".join(problems)
            + ").  column_order is the emission authority, so every entry must "
            "name an existing column and every column must have a position."
        )
    return names


def _validate_emitted_names(names: list[str], delimiter: str) -> None:
    """Refuse a column name the DEV header grammar cannot represent (AC-8).

    The test runs on the EMITTED name — the exact string written between
    two delimiters — because that is what the reader tokenizes.  The
    rejection set is the reader's own detection set: blank/whitespace,
    embedded whitespace or control characters (split or deleted by the
    reader), any of ``,`` ``;`` tab (scanned as a delimiter regardless of
    the emitted delimiter), the ACTIVE ``delimiter`` character, a leading
    double quote (the reader's quoted-field path), ``#``/``*COLUMNS`` in the
    FIRST position (the whole header becomes a comment or a keyword header),
    a ``sep=``/separator-rule shape as the ONLY name, and a header that
    parses entirely as numbers (the file is then read as headerless and the
    header line becomes a station).

    The delimiter test is POSITIONAL: it depends on the ``delimiter``
    argument in effect, so ``A|B`` is refused under ``delimiter="|"`` and
    accepted under the default space.

    Alias-bearing and lowercase spellings (``AZIM``, ``INCL``, ``TVDSS``,
    ``MDKB``, ``DEPT``, ``md``) are deliberately NOT rejected: they are
    legitimate DEV vocabulary, they round-trip as values, and their rename
    is reported by :func:`_emit_diagnostics` instead.
    """
    only = len(names) == 1
    for index, name in enumerate(names):
        reason = _column_name_defect(name, is_first=index == 0, is_only=only, delimiter=delimiter)
        if reason is not None:
            raise DEVWriteError(
                f"DEV column {name!r} cannot be emitted unambiguously: {reason}.  "
                f"The DEV reader's header grammar cannot represent it, so the "
                f"emitted file would corrupt or lose this column on re-read."
            )
    if names and all(_is_float_token(name) for name in names):
        raise DEVWriteError(
            f"DEV header {names!r} parses entirely as numbers: the DEV reader "
            f"classifies such a line as headerless DATA, so the header would "
            f"become the first station and every column name would be lost.  "
            f"Rename at least one column to a non-numeric mnemonic."
        )


def _validate_name_collisions(names: list[str]) -> None:
    """Refuse two columns the reader's alias map collapses onto one name.

    The DEV reader normalizes every header token (``strip().upper()`` plus
    a 29-entry alias map) and then de-duplicates with an ``_N`` suffix, so
    a model holding both ``AZI`` and ``AZIM`` is written as two columns and
    read back as ``AZI`` + ``AZI_2``: the second column changes identity
    and ``column_order`` no longer round-trips.  ``DevFile`` deliberately
    allows the pair (it is what ``normalize_aliases=False`` produces), so
    the writer is the only layer that can refuse it.
    """
    canonical_owner: dict[str, str] = {}
    for name in names:
        canonical = _normalize_dev_column(name)
        owner = canonical_owner.get(canonical)
        if owner is not None and owner != name:
            raise DEVWriteError(
                f"DEV columns {owner!r} and {name!r} both normalize to "
                f"{canonical!r}: the DEV reader applies its alias map and case "
                f"normalization to every header token, so the second column "
                f"would re-read as {f'{canonical}_2'!r} and column_order would "
                f"not round-trip.  Rename one of the two columns."
            )
        canonical_owner[canonical] = name


def _column_name_defect(
    name: object, *, is_first: bool, is_only: bool, delimiter: str
) -> str | None:
    """Return why ``name`` cannot be an emitted DEV column name, or None."""
    if not isinstance(name, str):
        return f"the name is not a string but {type(name).__name__}"
    if not name.strip():
        return "the name is empty or whitespace-only"
    if name != name.strip():
        return "the name has leading or trailing whitespace (the reader strips header tokens)"
    for char in name:
        if char in _SEPARATOR_CHARS:
            return (
                f"the name contains {char!r}, which the DEV reader scans as a "
                f"field separator regardless of the emitted delimiter"
            )
    for char in name:
        if char.isspace():
            return (
                f"the name contains whitespace ({char!r}), so the header line "
                f"would split it into two phantom columns and shift every "
                f"value on each row"
            )
    for char in name:
        if unicodedata.category(char) == "Cc":
            return (
                f"the name contains the control character {char!r}, which the "
                f"DEV reader deletes from the file before parsing"
            )
    if delimiter in name:
        return (
            f"the name contains {delimiter!r}, the active field delimiter, so "
            f"the header line would split it into phantom columns and shift "
            f"every value on each row"
        )
    if name.startswith('"'):
        return "the name begins with a double quote, which starts a quoted field the reader merges"
    if is_first and name.startswith("#"):
        return (
            "the first column name begins with '#', so the whole header line re-reads as a comment"
        )
    if is_first and name.upper().startswith("*COLUMNS"):
        return (
            "the first column name begins with '*COLUMNS', so the header "
            "re-reads as a keyword header and the names are stripped"
        )
    if is_only and _SEP_DIRECTIVE_RE.match(name) is not None:
        return (
            "the only column name matches the Excel 'sep=<char>' directive, "
            "which the reader deletes as exporter metadata"
        )
    if is_only and _SEPARATOR_RULE_RE.match(name) is not None:
        return (
            "the only column name is a bare separator rule ('===', '---'), "
            "which the reader treats as a non-content line"
        )
    return None


def _resolve_index_position(names: list[str]) -> int:
    """Return the position of the depth (index) column (AC-3 as amended).

    The DEV consumer resolves its depth column NOMINALLY, not positionally
    (``dev_reader.py:2205-2209``: the column named ``MD`` after alias
    normalization), so the writer must protect the same column: with a
    permuted ``["X", "MD", "TVD"]`` model the depth sat at position 1 and was
    formatted on the generic path, silently rounding it while the position-0
    column exceeded the requested precision (C-3).  ``_normalize_dev_column``
    is the reader's own normalizer, so an emitted ``DEPT``/``MDKB`` — which
    re-reads as ``MD`` — is treated as depth, and
    :func:`_validate_name_collisions` guarantees at most one column
    qualifies.  A model with no ``MD``-family column keeps the historical
    position-0 behaviour.
    """
    for position, name in enumerate(names):
        if _normalize_dev_column(name) == "MD":
            return position
    return 0


def _build_dev_text(
    dev: DevFile,
    names: list[str],
    row_count: int,
    precision: str,
    delimiter: str,
    null_value: float,
) -> tuple[str, _EmissionDiagnostics]:
    """Assemble the whole file as one ``str`` and collect its diagnostics."""
    arrays: list[NDArray[np.float64]] = [dev.columns[name] for name in names]
    collision_first: dict[int, tuple[float, str]] = {}
    collision_counts: dict[int, int] = {}
    nan_positions: set[int] = set()
    inexact_index: dict[int, tuple[float, str]] = {}
    index_position = _resolve_index_position(names)

    header = delimiter.join(names)
    _check_line_shape(header, len(names), delimiter, "header")
    lines: list[str] = [header]
    for row in range(row_count):
        values = [float(array[row]) for array in arrays]
        tokens = [
            _format_token(value, position, index_position, precision, null_value)
            for position, value in enumerate(values)
        ]
        for position, token in enumerate(tokens):
            value = values[position]
            if math.isnan(value):
                nan_positions.add(position)
            elif math.isfinite(value):
                # The collision gate is the EMITTED TOKEN, not the value:
                # a value rounded onto a sentinel by the requested
                # precision (TVDSS=-999.001 at '.2f' -> '-999.00' -> -999.0)
                # is read back as missing data exactly like an exact hit.
                try:
                    parsed = float(token)
                except ValueError:  # pragma: no cover - numeric tokens only
                    continue
                if parsed in _DEV_NUMERIC_SENTINELS:
                    collision_counts[position] = collision_counts.get(position, 0) + 1
                    collision_first.setdefault(position, (value, token))
                # Only a FINITE index cell can be "rounded": a non-finite one
                # is routed to the null sentinel above, and comparing the
                # sentinel token against NaN would invent a rounding warning
                # for a cell that was deliberately declared missing.
                if position == index_position and position not in inexact_index:
                    if parsed != value:
                        inexact_index[position] = (value, token)
        line = delimiter.join(tokens)
        _check_line_shape(line, len(names), delimiter, "row")
        lines.append(line)

    renamed = tuple(
        (name, _normalize_dev_column(name)) for name in names if name != _normalize_dev_column(name)
    )
    diagnostics = _EmissionDiagnostics(
        sentinel_collisions=tuple(
            (
                names[position],
                collision_first[position][0],
                collision_first[position][1],
                count,
            )
            for position, count in sorted(collision_counts.items())
        ),
        renamed_columns=renamed,
        nan_columns=tuple(names[position] for position in sorted(nan_positions)),
        inexact_index=tuple(
            (names[position], token, value)
            for position, (value, token) in sorted(inexact_index.items())
        ),
    )
    return _LINE_TERMINATOR.join(lines) + _LINE_TERMINATOR, diagnostics


def _format_token(
    value: float, position: int, index_position: int, precision: str, null_value: float
) -> str:
    """Render one cell; never an empty string, never a bare integer.

    ``_format_index_value`` is used for the index column — the ``MD``-family
    column resolved by :func:`_resolve_index_position` — so a depth needing
    more than 8 significant digits is not rounded AT THE DEFAULT PRECISION,
    which is the only precision that helper repairs; at an explicitly
    requested precision the index follows that precision like every other
    column and an inexact token is warned about (C-4).  The helper is used
    for a FINITE value only: its exactness fallback is ``repr(value)``,
    which renders a missing index cell as the literal text ``nan``/``inf``
    and would hand the reader an alphabetic null token (the token class that
    promotes an all-missing row to column names).  Non-finite cells go
    through the shared null formatter instead, so the sentinel is always
    numeric.
    """
    if position == index_position and math.isfinite(value):
        token = _format_index_value(value, precision, null_value)
    else:
        token = _format_number(value, precision, null_value)
    if _BARE_INTEGER_RE.match(token) is not None:
        token += ".0"
    return token


def _check_line_shape(line: str, name_count: int, delimiter: str, what: str) -> None:
    """Assert one emitted line carries exactly one non-empty token per column.

    Applied to EVERY emitted line — the header and each data row — because
    the invariant is about the bytes on disk, not about rows: a name that
    contains the active delimiter splits the HEADER into phantom columns and
    shifts every value on each row, which is invisible to a row-only check
    (C-1).  The grammar in :func:`_column_name_defect` refuses that class
    first and carries the user-facing message; this check is the emission
    backstop that keeps an unrepresentable line off the disk whatever the
    grammar missed.

    A short row is tail-filled with NaN and assigned positionally on
    re-read (shifting every later value one column left); an empty field
    makes a space-delimited row reduce to a blank line that the reader
    silently DROPS as a station.  Neither shape is reachable through the
    validation above — this is the emission-time invariant that keeps it
    that way.
    """
    parts = line.split(delimiter)
    if len(parts) != name_count or any(part == "" for part in parts):
        raise DEVWriteError(
            f"Internal emission error: {what} {line!r} does not carry "
            f"{name_count} non-empty token(s) separated by {delimiter!r}."
        )
    if delimiter == " " and len(line.split()) != name_count:
        raise DEVWriteError(
            f"Internal emission error: whitespace-split {what} {line!r} yields "
            f"{len(line.split())} token(s), expected {name_count}."
        )


def _warn_inexact_index_dev(
    curve_name: str,
    token: str,
    value: float,
    precision: str,
) -> bool:
    """C-4: warn (once per index column) when the emitted index is inexact.

    The DEV counterpart of ``_writer_base._warn_inexact_index`` — same
    trigger, same one-warning-per-column count — but with a message written
    for a DEV file.  The LAS helper's message asserts that CWLS LAS 2.0 §5.4
    forbids rounding depth values; that authority does not govern a DEV file
    (the format has no specification of its own), so emitting its sentence
    here claimed a conformance requirement the artifact cannot carry.  The
    obligation the shared helper documents — name the column, the source
    value and the emitted token — is what transfers between formats, not its
    wording.

    ``_format_index_value`` repairs the DEFAULT precision, so this fires only
    for an explicitly requested precision (or a non-float64 scalar) that
    rounds the index — the remaining silent-loss path.  A rounded index can
    make two distinct samples share one depth on re-read (ADV-26 probe: three
    depths, one emitted token).  Returns True when it warned.
    """
    try:
        exact = float(token) == float(value)
    except ValueError:
        return False
    if exact:
        return False

    warnings.warn(
        f"Index curve {curve_name!r}: the emitted depth {token!r} does not "
        f"read back as the model value {float(value)!r} (precision "
        f"{precision!r}).  The DEV format defines no depth-precision rule of "
        f"its own, but the emitted index is the station depth a DEV consumer "
        f"reads, so distinct depths can collapse into one depth row on "
        f"re-read.  Use the default precision to keep the index exact.",
        UserWarning,
        stacklevel=4,
    )
    return True


def _warn_degenerate(detail: str) -> None:
    """Warn (then the caller raises) about a model that cannot be read back."""
    warnings.warn(
        f"write_dev_file: {detail}.  The file is not written — raising "
        f"DEVWriteError instead of emitting an unreadable DEV file.",
        UserWarning,
        stacklevel=4,
    )


def _emit_diagnostics(dev: DevFile, diagnostics: _EmissionDiagnostics, null_value: float) -> None:
    """Emit the collected ``UserWarning`` diagnostics (never raise)."""
    for column, value, token, count in diagnostics.sentinel_collisions:
        times = "once" if count == 1 else f"{count} times"
        warnings.warn(
            f"DEV column {column!r}: value {value!r} is emitted as {token!r}, "
            f"which is one of the DEV numeric null sentinels "
            f"({sorted(_DEV_NUMERIC_SENTINELS)}).  Every DEV consumer — this "
            f"library's reader included — reads that token as NO-DATA, so the "
            f"value cannot be encoded ({times} in this column).  Declare a "
            f"null_value that no real measurement can take.",
            UserWarning,
            stacklevel=4,
        )
    for name, canonical in diagnostics.renamed_columns:
        warnings.warn(
            f"DEV column {name!r} will be renamed to {canonical!r} when the "
            f"file is re-read: the DEV reader normalizes aliases and case.  "
            f"Pass normalize_aliases=False to read_dev_file to preserve the "
            f"original spelling.",
            UserWarning,
            stacklevel=4,
        )
    if not dev._designed_nan and diagnostics.nan_columns:
        # C-5: the re-read outcome is CONDITIONAL, so the sentence is branched
        # on the same predicate that decides it.  "Declared missing data" is
        # true only for a null_value the reader maps to NaN
        # (_DEV_NUMERIC_SENTINELS); any other finite null is emitted verbatim
        # and re-reads as that literal NUMBER, which no consumer can tell from
        # a measurement.  Claiming the first outcome for the second case
        # fabricated a guarantee the file does not carry.
        if null_value in _DEV_NUMERIC_SENTINELS:
            outcome = (
                "They are emitted as the null sentinel and re-read as DECLARED "
                "MISSING DATA — the DEV format cannot distinguish a designed gap "
                "from a corrupt value, so the file records no evidence of the "
                "original NaN."
            )
        else:
            outcome = (
                f"They are emitted as the literal value {null_value!r} and re-read "
                f"as that NUMBER, not as missing data: {null_value!r} is not one of "
                f"the DEV numeric null sentinels ({sorted(_DEV_NUMERIC_SENTINELS)}), "
                f"so no DEV consumer can tell these cells from a measurement.  Pass "
                f"one of those sentinels as null_value to record the gap as missing "
                f"data."
            )
        warnings.warn(
            f"write_dev_file: {len(diagnostics.nan_columns)} column(s) contain "
            f"NaN cells ({list(diagnostics.nan_columns)}).  {outcome}",
            UserWarning,
            stacklevel=4,
        )


def _write_text_atomically(file_path: str | Path, text: str, encoding: str) -> None:
    """Write ``text`` through a temp file in the target directory (AC-7).

    The destination is never opened for writing; :func:`os.replace` is its
    only mutation, so a failure at any point leaves a pre-existing target
    byte-identical.  The sequence mirrors the LAS path except for the mode:
    there the temp file is chmod'ed BEFORE the data is written, which lets
    the write clear setuid/setgid bits; here the mode is applied after the
    write and fsync so an existing target's bits survive verbatim.
    """
    target = Path(os.path.realpath(file_path))
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DEVWriteError(f"Cannot create output directory {target.parent}: {exc}") from exc

    try:
        existing_mode: int | None = os.stat(target).st_mode & 0o7777
    except FileNotFoundError:
        existing_mode = None
    except OSError as exc:
        raise DEVWriteError(f"Cannot inspect {file_path}: {exc}") from exc
    mode = existing_mode if existing_mode is not None else _default_new_file_mode()

    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp_", suffix=target.name)
    except OSError as exc:
        raise DEVWriteError(f"Cannot write to {file_path}: {exc}") from exc

    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            _apply_file_mode(stream.fileno(), tmp_path, mode)
        os.replace(tmp_path, target)
        _fsync_directory(str(target.parent))
    except DEVWriteError:
        _unlink_quietly(tmp_path)
        raise
    except Exception as exc:  # every I/O failure is a write error
        _unlink_quietly(tmp_path)
        raise DEVWriteError(f"Cannot write to {file_path}: {exc}") from exc


def _unlink_quietly(path: str) -> None:
    """Remove the temp file, ignoring the failure (the write already failed)."""
    try:
        os.unlink(path)
    except OSError:  # pragma: no cover - filesystem dependent
        pass
