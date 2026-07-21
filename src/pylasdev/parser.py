"""Regex-based LAS file parser replacing PLY.

The LAS format is line-based with a simple structure:
  MNEMONIC.UNIT  VALUE : DESCRIPTION

PLY (lex/yacc) is overkill for this. Regex reduces ~450 lines to ~150
while maintaining the same parsing capability.

Supports LAS 1.2, 2.0, and 3.0 formats.
"""

from __future__ import annotations

import logging
import re
import warnings
from itertools import groupby
from typing import ClassVar

import numpy as np

from . import data_reader as _data_reader
from ._las30_data import AsciiDataContext, process_ascii_data
from ._version_spec import _LASVersionSpec
from .data_reader import (
    _parse_float_with_d_notation,
)
from .exceptions import LASParseError
from .mnem_base import resolve_mnemonic
from .models import (
    _VALID_DATA_FORMATS,
    ArrayElementInfo,
    CurveDefinition,
    LASFile,
    ParameterEntry,
    ParameterZone,
)

logger = logging.getLogger(__name__)

# F-26: Global aggregate limit on data sections to prevent multi-section DoS.
# Each section passes per-section bounds (_data_reader.MAX_DATA_LINES, _data_reader.MAX_CURVES,
# _data_reader.MAX_TOTAL_ELEMENTS) but no global cap existed — an attacker could craft N
# data sections cumulatively exhausting memory.  Overridable at module level.
MAX_DATA_SECTIONS = 1_000

# F-29: Maximum parameter entries per file.  Curves have _data_reader.MAX_CURVES (100K)
# checked in 3 locations; parameters had zero protection anywhere.
MAX_PARAMETERS = 100_000

# F-M-02: Maximum other-section lines.  All other accumulators have explicit
# MAX_* constants; _other_lines had no bound, enabling unbounded memory growth
# from malformed files.  Overridable at module level.
MAX_OTHER_LINES = 1_000_000

# F-212: When reading files NOT produced by pylasdev's writer, the
# _desanitize_las_value transformation should not be applied — _# in
# external data is genuine content, not a writer escape.  Defaults to
# True (preserves existing roundtrip behavior).  Set to False before
# reading external files to prevent data corruption.
_DESANITIZE_ENABLED: bool = True

# F-35: Maximum deferred well entries.  Every other accumulator in parser.py
# has a MAX_* guard (_ascii_data_lines, las_file.curves, las_file.parameters,
# _other_lines, _current_data_section_idx).  _deferred_well_entries had no
# bound, enabling unbounded memory growth from malicious ~W-before-~V files
# without a ~V section.  Well sections are inherently small (20-80 entries
# in practice); this guard provides defense-in-depth against attacks.
# Overridable at module level.
MAX_DEFERRED_WELL_ENTRIES = MAX_PARAMETERS

# F-006: Maximum section sequence entries.  All other accumulators have explicit
# MAX_* guards (_ascii_data_lines, las_file.curves, las_file.parameters,
# _other_lines, _deferred_well_entries); _section_sequence had no bound,
# enabling unbounded growth from repeated unknown section headers.
# Overridable at module level.
MAX_SECTION_SEQUENCE = 200

# F-I2-M06: Maximum length of a single line in bytes/characters.  max_file_size
# (500MB) bounds total file but not individual line length — a crafted file
# could allocate a 500MB string for a single line.  Real-world LAS lines are
# typically under 1 KB; 50 KB allows generous descriptions while preventing
# absurd single-line allocations.  Overridable at module level.
MAX_LINE_LENGTH = 50_000

# F-I2-M07: Maximum length of a captured field (value, description) after
# regex matching.  The DATA_LINE_PATTERN uses unbounded .*? groups; a crafted
# 500MB value/description could be allocated.  This guards each captured group
# independently.  Overridable at module level.
MAX_FIELD_LENGTH = 100_000

# I2F-01: Maximum line length for safe regex matching without risk of
# catastrophic backtracking.  Lines exceeding this threshold bypass the
# regex entirely and use a manual scan for the colon separator.  This
# provides defense-in-depth against regex DoS — the regex itself is
# already fixed (non-backtracking lookahead), but a manual scan is
# guaranteed O(n) regardless of regex engine edge cases.
# Overridable at module level.
_SAFE_REGEX_LINE_LENGTH = 2_000

# M-01: Hoist regex and frozenset to module level — previously re-allocated
# per call in _validate_curve_data_format (hot path, up to
# _data_reader.MAX_CURVES=100K calls per file).
_KNOWN_CURVE_FORMATS: frozenset[str] = frozenset({"F", "E", "D", "S", "A", "I"})
_FORMAT_SPEC_RE = re.compile(
    r"^(?:[FEDI](?:\d+(?:\.\d+)?(?:[ED][+-]?\d+)?)?|[SA]\w*(?:;[\w.]+)*)$"
)

# F-32 + G-17 + F-I2-M04 + F-I2-M05: Control characters that appear in file
# content and must be stripped before splitlines() to prevent section-header
# injection and silent data corruption.  The writer's _CONTROL_CHARS_RE strips
# these; this makes the read path symmetric.
# F-I2-M04: Added \x00 (null byte) — previously bypassed SECTION_PATTERN,
#   routing ~\x00VERSION to _other_lines and corrupting float parsing.
# F-I2-M05: Added remaining control characters (\x00-\x08, \x0E-\x1F, \x7F)
#   that the writer strips but the parser/reader previously passed through.
# Characters: \x00-\x08, \x0B (VT), \x0C (FF), \x0E-\x1F (FS/GS/RS/etc.),
# \x7F (DEL), \x85 (NEL), \u2028 (LINE SEPARATOR), \u2029 (PARAGRAPH SEPARATOR).
# F-001: Also include the 13 Unicode whitespace characters that the writer's
# _CONTROL_CHARS_RE strips (\u00A0, \u2000-\u200A, \u202F, \u205F, \u3000)
# so the write→read roundtrip is symmetric.
_SPLITLINES_CHARS_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x85"
    r"\u2028\u2029"
    r"\u00A0\u2000-\u200A\u202F\u205F\u3000]"
)

# Section header: line starting with ~, followed by section keyword and trailing content.
# Captures the full section word (e.g., "VERSION", "Core_Definition", "Core[1]")
# and any trailing content (e.g., "| Core_Definition", "INFORMATION").
# F-01/F-08: Previous pattern ^~([A-Za-z])(.*) captured only the first letter,
# causing ~Core_Data and ~Core_Definition to both route to the curve handler (letter C).
# G-05: Narrowed to require a letter after ~, matching data_reader._is_section_header.
# Prevents divergence where _pre_scan treated non-alphabetic headers like ~3D_DATA
# as section boundaries but _read_normal did not (causing array overrun, G-04).
SECTION_PATTERN = re.compile(r"^~([A-Za-z]\S*)(.*)")

# Data line pattern: MNEMONIC.UNIT  VALUE : DESCRIPTION
# Uses \w which matches Unicode (including Cyrillic) in Python 3
# Note: LAS files commonly have spaces between mnemonic and dot (e.g., "DT  .US/M")
#
# The colon separator requires whitespace on at least one side of the colon.
# This prevents false matches on bare colons in values (timestamps like
# "12:34:56") and LAS 3.0 format specifiers ({A:0}), while still correctly
# separating value from description in standard "VALUE : DESCRIPTION" lines
# and handling empty-value lines like "MNEM.UNIT       : DESCRIPTION".
#
# Per the CWLS LAS 2.0 specification, the LAST structurally-valid colon
# is the delimiter.  The value group uses greedy (.*) matching so the
# regex engine backtracks to the rightmost colon that meets the separator
# criteria.  This correctly handles header lines with embedded colons in
# description text (e.g., "MNEM.UNIT VAL : DESC : MORE" → desc="DESC : MORE").
#
# I2F-01: The previous alternation (\s+:\s*|\s*:\s+|:\s*$) had overlapping
# alternatives (both \s+:\s* and \s*:\s+ match when whitespace exists on
# both sides of the colon) that caused catastrophic O(n^3) regex
# backtracking on long lines with many spaces and colons.
#
# The fix uses non-overlapping alternatives:
#   (1) \s+:\s*  — at least one whitespace BEFORE colon, optional after
#   (2) :(?=\s)  — colon followed by whitespace (matched when (1) fails
#                  because there's no whitespace before the colon)
#   (3) :\s*$    — colon at end of line (partial-data-line detection)
#
# Alternatives (1) and (2) are mutually exclusive at any given position:
# (1) requires whitespace before colon; (2) uses a lookahead (inherently
# atomic — no backtracking) to check for whitespace after colon, and is
# only tried when (1) fails.  This eliminates the backtracking explosion
# while preserving the original matching semantics — the separator does
# NOT match at a colon that has no surrounding whitespace.
#
# For additional defense-in-depth, _match_data_line uses a manual scan
# fallback for lines >2000 chars to bypass regex entirely.
DATA_LINE_PATTERN = re.compile(
    r"^\s*"
    r"(?P<mnemonic>[\w\-]+(?:\[\d+\])?)"  # mnemonic: word chars + hyphen + optional [N] array index
    r"\s*"  # optional whitespace before dot (common in LAS files)
    r"\."  # literal dot separator
    r"(?P<unit>[\w\-/]*)"  # unit: optional, can include /
    r"\s+"  # whitespace separator
    r"(?P<value>.*)"  # value: greedy to find LAST structurally-valid colon (CWLS spec)
    r"(\s+:\s*|:(?=\s)|:\s*$)"  # colon separator (see I2F-01 comment above)
    r"(?P<description>.*?)"  # description: rest of line
    r"\s*$"
)

# Simpler pattern for lines without description (value-only)
VALUE_ONLY_PATTERN = re.compile(
    r"^\s*"
    r"(?P<mnemonic>[\w\-]+(?:\[\d+\])?)"  # mnemonic with optional [N] array index
    r"\s*"  # optional whitespace before dot
    r"\."
    r"(?P<unit>[\w\-/]*)"
    r"\s+"
    r"(?P<value>.+?)"
    r"\s*$"
)

# LAS 3.0: Array notation pattern (e.g., NMR[1], RUN[2])
ARRAY_MNEMONIC_PATTERN = re.compile(r"^(?P<base>[\w\-]+)\[(?P<index>\d+)\]$")

# LAS 3.0: Format specifier in braces (e.g., {F}, {E}, {S}, {A:0},
# {D}, {DEG}, {DD/MM/YYYY}, {E0.00E+00}).
# F-M-05: Expanded from [FESAI] to match multi-character format codes
# (D, DEG, date templates, extended exponent notation).  Format group
# captures one or more non-brace, non-colon characters; colon and
# optional offset follow.
# F-I2-M08: Exclude whitespace (\s) from the format body so that
# brace-enclosed text containing spaces (e.g. "{well A12}") is NOT
# matched as a format specifier.  LAS 3.0 format specifiers do not
# contain spaces; matching them caused description corruption via
# sub("") stripping legitimate brace-enclosed text.
FORMAT_SPEC_PATTERN = re.compile(r"\{(?P<format>[A-Za-z][^}:\s]*?)(?::(?P<offset>[-\d.]*))?\s*\}")

# LAS 3.0: Zone association via pipe (e.g., | Run[1], | Zone[2]).
# F-M16: Support zone names containing spaces (e.g., "| Main Zone").
# Matches one or more word-character groups separated by whitespace.
ZONE_ASSOC_PATTERN = re.compile(
    r"\|\s*(?P<zone>[\w\-]+(?:\s+[\w\-]+)*)(?:\[(?P<index>\d+)\])?$"
)

COMMENT_PATTERN = re.compile(r"^\s*#")
EMPTY_PATTERN = re.compile(r"^\s*$")

# LAS 3.0 section type keywords for structured data-type sections.
# These are data sections (contain rows of values) — NOT definition sections.
# Definition sections (Core_Definition, Drilling_Definition, etc.) are
# routed to _parse_curve since they define curves for their data type.
_DATA_SECTION_WORDS = {
    "A",
    "ASCII",  # Standard log data
    "CORE",  # Core data section
    "CORE_DATA",  # Core data section (written form)
    "DRILLING",  # Drilling data section
    "DRILLING_DATA",  # Drilling data section (written form)
    "FORMATION",  # Formation data section (LAS 3.0 spec) — F-M26
    "FORMATION_DATA",  # Formation data section (written form)
    "INCLINOMETRY",  # Inclinometry data section
    "INCLINOMETRY_DATA",  # Inclinometry data section (written form)
    "LOG",  # LAS 3.0 shorthand ~Log alias for ~Log_Data / ~Ascii
    "LOG_DATA",  # Explicit log data section
    "MUD",  # Mud data section (LAS 3.0 spec) — F-M26
    "MUD_DATA",  # Mud data section (written form)
    "PERFORATIONS",  # Perforations data section
    "PERFORATIONS_DATA",  # Perforations data section (written form)
    "RISK",  # Risk data section (LAS 3.0 spec) — F-M26
    "RISK_DATA",  # Risk data section (written form)
    "STRUCTURE",  # Structure data section (LAS 3.0 spec) — F-M26
    "STRUCTURE_DATA",  # Structure data section (written form)
    "TEST",  # Test data section
    "TEST_DATA",  # Test data section (written form)
    "TOPS",  # Tops data section
    "TOPS_DATA",  # Tops data section (written form)
}

# LAS 3.0 data section types that support index notation (e.g., ~Core[1]).
# Used to match bracketed sections like ~Inclinometry[1], ~Drilling[2], etc.
_INDEXED_DATA_TYPES = frozenset(
    {
        "CORE",
        "CORE_DATA",
        "DRILLING",
        "DRILLING_DATA",
        "FORMATION",
        "FORMATION_DATA",
        "INCLINOMETRY",
        "INCLINOMETRY_DATA",
        "LOG",
        "LOG_DATA",
        "MUD",
        "MUD_DATA",
        "PERFORATIONS",
        "PERFORATIONS_DATA",
        "RISK",
        "RISK_DATA",
        "STRUCTURE",
        "STRUCTURE_DATA",
        "TEST",
        "TEST_DATA",
        "TOPS",
        "TOPS_DATA",
    }
)

# LAS 3.0 section type to canonical DataSection.section_type mapping.
_SECTION_TYPE_MAP: dict[str, str] = {
    "A": "LOG_DATA",
    "ASCII": "LOG_DATA",
    "CORE": "CORE_DATA",
    "CORE_DATA": "CORE_DATA",
    "DRILLING": "DRILLING_DATA",
    "DRILLING_DATA": "DRILLING_DATA",
    "FORMATION": "FORMATION_DATA",
    "FORMATION_DATA": "FORMATION_DATA",
    "INCLINOMETRY": "INCLINOMETRY_DATA",
    "INCLINOMETRY_DATA": "INCLINOMETRY_DATA",
    "LOG": "LOG_DATA",
    "LOG_DATA": "LOG_DATA",
    "MUD": "MUD_DATA",
    "MUD_DATA": "MUD_DATA",
    "PERFORATIONS": "PERFORATIONS_DATA",
    "PERFORATIONS_DATA": "PERFORATIONS_DATA",
    "RISK": "RISK_DATA",
    "RISK_DATA": "RISK_DATA",
    "STRUCTURE": "STRUCTURE_DATA",
    "STRUCTURE_DATA": "STRUCTURE_DATA",
    "TEST": "TEST_DATA",
    "TEST_DATA": "TEST_DATA",
    "TOPS": "TOPS_DATA",
    "TOPS_DATA": "TOPS_DATA",
}


def _is_indexed_data_section(section_word: str) -> bool:
    """Check if a section word is an indexed data section (e.g., Core[1], Inclinometry[2]).

    Matches any known data section type with bracket index notation.
    """
    bracket_idx = section_word.find("[")
    if bracket_idx < 0:
        return False
    # Verify the part after [ is digits followed by ]
    rest = section_word[bracket_idx + 1 :]
    if not rest.endswith("]"):
        return False
    index_str = rest[:-1]
    if not index_str.isdigit():
        return False
    base = section_word[:bracket_idx]
    return base in _INDEXED_DATA_TYPES


def _unescape_colons_for_las_value(value: str) -> str:
    """Reverse the ``_escape_colons_for_las_value`` transformation.

    The writer applies a two-step colon escape to prevent the parser from
    misinterpreting embedded colons as structural separators:

    1. Insert ``_`` between whitespace and colon: ``" :"`` → ``" _:"``
    2. Insert ``_`` after colon followed by whitespace or end: ``": "`` → ``":_ "``

    The combined effect on ``" : "`` produces ``" _:_ "``.

    This function reverses both steps in the opposite order (step 2 first,
    then step 1), restoring the original colon-separated text.  It is
    applied during parsing to all values and descriptions that may have
    been escaped by the writer — well entries, parameter values, curve
    descriptions, and curve API codes.

    .. note::

       Legitimate underscore characters that happen to form the escape
       pattern (e.g., ``tag_:`` in original data) will be incorrectly
       unescaped.  This is the same trade-off acknowledged by the writer's
       docstring — the roundtrip loss is limited to the contrived case
       where user data naturally contains the escape-artifact patterns.
    """
    # Undo step 2 first: remove ``_`` after colon when followed by
    # whitespace or end-of-string (``:_ `` → ``: ``, ``:_$`` → ``:$``).
    value = re.sub(r":_(?=\s|$)", ":", value)
    # Undo step 1: remove ``_`` between whitespace and colon
    # (`` _:`` → `` :``, ``\t_:`` → ``\t:``).
    value = re.sub(r"(\s+)_:", r"\1:", value)
    return value


def _desanitize_las_value(value: str) -> str:
    """Reverse the _sanitize_las_value ``#``-prefix escape.

    The writer prefixes ``#``-starting values with ``_`` to prevent the
    parser from interpreting them as comment lines.  This function strips
    that prefix, restoring the original ``#``-prefixed value.

    Two cases (matching writer's ``_sanitize_las_value``):

    1. ``value.startswith("#")`` → writer prepends ``_`` → ``"_#..."``
       → reverse: strip the leading ``_``.
    2. ``value.lstrip().startswith("#")`` → writer inserts ``_`` after
       leading whitespace → ``" _#..."`` → reverse: remove the ``_``
       between whitespace and ``#``.
    """
    if not _DESANITIZE_ENABLED:
        return value
    if value.startswith("_#"):
        return value[1:]
    # Case 2: whitespace-prefixed value with sanitized _# (e.g., " _#comment")
    idx = value.find("_#")
    if idx > 0 and value[idx - 1].isspace():
        return value[:idx] + value[idx + 1 :]
    return value


# F-088 / F-102: Shared format specifier validation — extracted from
# _process_ascii_data so _parse_curve can also invoke it.  This closes
# the deferred-validation gap where metadata-only LAS 3.0 files (no ~A
# data section) bypassed format checking because the validator sat inside
# data-processing code.  Also used by _process_ascii_data (replaces the
# inlined check) so both call sites share the same validator.
def _validate_curve_data_format(data_format: str, mnemonic: str) -> None:
    """Validate a curve data_format against known LAS format specifiers.

    Accepts single-letter codes (F, E, D, S, A) and extended Fortran-style
    format specifiers (e.g., "F8.3", "E10.2", "E0.00E+00").  Rejects
    non-numeric templates such as "DEG", "DD/MM/YYYY" which are metadata
    strings, not LAS data format specifiers.

    Args:
        data_format: The format specifier string (already uppercased).
        mnemonic: The curve mnemonic for error messages.

    Raises:
        LASParseError: If the format specifier is not a recognised single-letter
            code or valid extended format.
    """
    if not data_format:
        return
    if data_format in _KNOWN_CURVE_FORMATS or _FORMAT_SPEC_RE.match(data_format):
        return
    raise LASParseError(
        f"Curve '{mnemonic}' has unsupported format "
        f"specifier '{{{data_format}}}'. Non-numeric format types "
        f"(e.g., {{DEG}}, date templates) are not valid LAS "
        f"data format specifiers. "
        f"Valid values: {', '.join(sorted(_VALID_DATA_FORMATS))}"
    )


class LASParser:
    """Regex-based LAS file parser.

    Encapsulates all parsing state in the instance (no global variables).
    Each instance maintains its own state (instance-level isolation).
    Not thread-safe for concurrent parse() calls on the same instance.

    Supports LAS 1.2, 2.0, and 3.0 formats.
    """

    SECTION_HANDLERS: ClassVar[dict[str, str]] = {
        "V": "_parse_version",
        "W": "_parse_well",
        "C": "_parse_curve",
        "P": "_parse_parameter",
        "O": "_parse_other",
        "A": "_parse_ascii_data",
    }

    def __init__(self, mnem_base: dict[str, str] | None = None, well_format: str = "auto") -> None:
        """Initialize parser with optional mnemonic base.

        Args:
            mnem_base: Optional mnemonic-to-canonical-name mapping for
                curve/parameter normalization.
            well_format: LAS 1.2 well section format convention:
                ``"auto"`` (default) heuristically detects CWLS vs lasio
                convention per field; ``"cwls"`` forces CWLS convention
                (``MNEM.UNIT VALUE : DESCRIPTION``) for all non-numeric
                fields; ``"lasio"`` forces lasio convention
                (``MNEM.UNIT DESCRIPTION : VALUE``).
        """
        self.mnem_base = mnem_base or {}
        self._well_format = well_format
        if well_format is not None and well_format not in ("cwls", "lasio", "auto"):
            warnings.warn(
                f"Unrecognized well_format={well_format!r} — "
                f"falling back to auto-mode. Valid values: 'cwls', 'lasio', 'auto', None",
                UserWarning,
                stacklevel=2,
            )
        # Build uppercased lookup with multi-step chain resolution.
        # resolve_mnemonic walks chains like BK-3 → BK → BFV to reach
        # the terminal canonical name. Single .get() only resolves one hop.
        #
        # F-H-01: Use first-wins semantics — canonical uppercase entries
        # (e.g. "BK": "BFV") appear first in insertion order.  Later
        # lowercased aliases (e.g. "bk": "BK") that collide on uppercased
        # key must not overwrite.  Dict comprehension gives last-wins,
        # which breaks chain resolution.
        _raw_upper: dict[str, str] = {}
        # Sort so canonical uppercase entries come first, ensuring
        # first-wins semantics are invariant regardless of input dict
        # ordering.  Without this sort, a dict where lowercase aliases
        # (e.g. "bk") precede canonical entries ("BK") would break
        # chain resolution because the lowercase alias would "win"
        # and overwrite when uppercased.
        sorted_items = sorted(
            self.mnem_base.items(),
            key=lambda item: (not item[0].isupper(), item[0]),
        )
        for k, v in sorted_items:
            key = k.upper()
            if key not in _raw_upper:
                _raw_upper[key] = v
        self._mnem_base_upper: dict[str, str] = {}
        for k in _raw_upper:
            self._mnem_base_upper[k] = resolve_mnemonic(_raw_upper, k)
        self.source_file: str = ""
        self._reset()

    def _reset(self) -> None:
        """Reset parser state for a new file."""
        self.las_file = LASFile()
        self._current_section: str | None = None
        self._current_section_name: str = ""
        self._current_data_section_type: str = "LOG_DATA"
        self._data_line_count = 0
        self._ascii_data_lines: list[str] = []
        self._current_data_section_idx: int = 0
        # F2-006: Cumulative cross-section element counter — tracks
        # total elements (curves x lines) across ALL data sections
        # to prevent multi-section files from exhausting memory when
        # each individual section passes the per-section bound.
        self._cumulative_elements: int = 0
        # F-09: Cumulative cross-section raw data line counter — tracks
        # total lines across ALL sections to catch multi-section files
        # where each individual section passes per-section bounds.
        self._cumulative_data_lines: int = 0
        self._cumulative_data_lines_warned: bool = False
        self._version_found = False  # flag for required ~V section validation
        # F-3: Accumulate other-section lines in a list to avoid O(n^2)
        # string concatenation (self.las_file.other += ... per line).
        self._other_lines: list[str] = []
        # F1: Track per-section curve boundaries for LAS 3.0 files where
        # different ~C blocks define different curve sets before each ~A section.
        self._section_curve_start_idx: int = 0
        self._section_curve_end_idx: int | None = None
        # F-01/F-08: Track the end of the main (non-_Definition) curve block.
        # When a data section has a pipe "| CURVE" association, the per-section
        # curves revert to using the main block (curves[0:_main_curve_end]).
        self._main_curve_end: int = -1
        # G-02: Per-_Definition curve range storage.  Consecutive _Definition
        # sections overwrote _section_curve_start_idx, making later data
        # sections without pipe associations use the wrong curve range.
        # Maps definition name (e.g. "CORE_DEFINITION") → (start_idx, end_idx).
        self._definition_curve_ranges: dict[str, tuple[int, int]] = {}
        self._current_definition_name: str | None = None
        # M-PA4: Track whether LAS 3.0 typed data sections (~Core, ~Drilling,
        # etc.) were encountered during parsing for cross-validation against
        # the VERS header value.
        self._las30_sections_seen: bool = False
        # F-P06: Raw well entries are buffered when ~W appears before ~V so
        # they can be re-processed with the correct version after ~V is parsed.
        self._deferred_well_entries: list[dict[str, str]] = []
        # F-M12: Raw ASCII data lines are buffered when data sections (~A,
        # ~ASCII, etc.) appear before ~V so they can be re-processed after
        # the version (and hence is_las30) is known.
        # F-H01 / I2-D2-01: Each entry is a (section_type, section_name,
        # section_idx, raw_line, curve_start_idx, curve_end_idx) tuple to
        # preserve per-section boundaries AND pipe-target curve scoping
        # across the deferred replay.  section_idx distinguishes
        # consecutive sections that happen to have the same section_type
        # and section_name (e.g., two bare ~A sections).
        self._deferred_ascii_data_lines: list[
            tuple[str, str, int, str, int, int | None]
        ] = []
        # F-34: Track section headers encountered (in order) for cross-section
        # consistency validation: duplicate detection and LAS 3.0 ordering check.
        self._section_sequence: list[str] = []
        # F-048/F-103: Parallel list tracking semantic section type codes
        # ("V", "W", "C", "P", "O", "A") for duplicate detection normalization.
        # Label-based counting (above) misses semantically-duplicate sections
        # that differ only by section_name (e.g., ~V = "V:" vs ~VERSION = "VERSION:").
        self._section_type_sequence: list[str] = []

    @property
    def data_line_count(self) -> int:
        """Public accessor for pre-scanned data line count."""
        return self._data_line_count

    def parse(self, content: str, lines: list[str] | None = None) -> LASFile:
        """Parse LAS file content string.

        Args:
            content: Raw file content string. Used only if `lines` is not
                provided, for backward compatibility.
            lines: Pre-split lines list (PERF-01 optimization). When
                provided, content is not split again, eliminating the
                double splitlines() between parser and data_reader.
        """
        self._reset()

        # Three-pass parsing design:
        #   Pass 1 (_pre_scan): count ASCII data lines to enable pre-allocation
        #     of numpy arrays at the exact size, avoiding O(n^2) incremental
        #     reallocation during data fill.
        #   Pass 2 (_parse_line looping): extract metadata (version, well, curve
        #     definitions, parameters) and collect ASCII data lines.
        #   Pass 3 (_process_ascii_data): fill pre-allocated numpy arrays with
        #     parsed numeric/string values.
        # Three separate passes are necessary because Pass 2 performs array
        # pre-allocation (using the count from Pass 1) BEFORE Pass 3 fills them.
        if lines is None:
            lines = _SPLITLINES_CHARS_RE.sub(" ", content).splitlines()
        else:
            # I2-XPD-01: When lines are provided externally (e.g., from
            # reader.py's PERF-01 optimization), apply _SPLITLINES_CHARS_RE
            # sanitization to each line to prevent null-byte-prefixed
            # section-header injection.  The content-level substitution
            # was already applied by the caller, but per-line cleaning
            # ensures defense-in-depth against stale or unsanitized
            # externally-supplied line lists.
            lines = [_SPLITLINES_CHARS_RE.sub(" ", ln) for ln in lines]
        self._pre_scan(lines)

        for line in lines:
            self._parse_line(line)

        # F-P06: Re-process well entries parsed before ~V was known.
        # If the version turns out to be LAS 1.2, overwrite buffered
        # entries with the correct value/description swap.
        self._replay_deferred_well()

        # Validation: a valid LAS file must have a ~V section
        if not self._version_found and content.strip():
            raise LASParseError(
                "Content does not appear to be a valid LAS file: "
                "missing required ~V (Version Information) section."
            )

        # Warn when empty/whitespace-only content is parsed without a ~V section.
        # An empty file produces a default LASFile with version "2.0" — this is
        # intentional for robustness, but callers should know about it.
        if not self._version_found and not content.strip():
            warnings.warn(
                "Empty or whitespace-only LAS content — returning default empty LASFile",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "Empty or whitespace-only content parsed without a ~V section; "
                "returning default LASFile with version '2.0'."
            )

        # Finalize accumulated ~O section text (F-3: O(n) join vs O(n^2) concat)
        if self._other_lines:
            self.las_file.other = "\n".join(self._other_lines) + "\n"

        # Process collected ASCII data only for LAS 3.0
        # For LAS 1.2/2.0, data_reader handles ASCII data with proper wrap mode support
        if self.las_file.version.is_las30:
            ctx = AsciiDataContext(
                las_file=self.las_file,
                ascii_data_lines=self._ascii_data_lines,
                section_curve_start_idx=self._section_curve_start_idx,
                section_curve_end_idx=self._section_curve_end_idx,
                current_section_name=self._current_section_name,
                current_data_section_type=self._current_data_section_type,
                current_data_section_idx=self._current_data_section_idx,
                cumulative_elements=self._cumulative_elements,
            )
            process_ascii_data(ctx)
            self._cumulative_elements = ctx.cumulative_elements

        # Validate mandatory well fields (STRT, STOP, STEP, NULL).
        # LAS 1.2 and 2.0 both require these fields; LAS 3.0 inherits the
        # same mandatory well-field requirements from LAS 2.0.  Missing
        # fields are a spec compliance gap.  The library handles missing
        # fields gracefully (using defaults), so this is a warning, not an
        # error.  F-M24: Previous check excluded LAS 1.2 despite the LAS
        # 1.2 spec requiring STRT/STOP/STEP/NULL.
        # F-014: Previous version gate only checked startswith("2."),
        # silently skipping mandatory-field validation for LAS 3.0 files.
        is_las12_or_later = _LASVersionSpec(
            self.las_file.version.vers
        ).is_las12_or_later
        if is_las12_or_later and self._version_found:
            # M-03: All 8 LAS 1.2 mandatory well fields (was 4).
            # WELL, LOC, SRVC, and UWI are commonly missing in real-world
            # files — we warn but do NOT raise an error.
            _mandatory_fields = [
                "STRT", "STOP", "STEP", "NULL",
                "WELL", "LOC", "SRVC", "UWI",
            ]
            for field in _mandatory_fields:
                if field not in self.las_file.well.entries:
                    warnings.warn(
                        f"LAS {self.las_file.version.vers} file missing "
                        f"mandatory well field: {field}",
                        stacklevel=2,
                    )

        # M-PA4: Cross-validate VERS against encountered section types.
        # When LAS 3.0 typed data sections (~Core, ~Drilling, etc.) are
        # found but is_las30 is False (VERS does not start with "3"), the
        # parser silently discards data in those sections.  Warn the user
        # and advise correcting the VERS header.
        if self._las30_sections_seen and not self.las_file.version.is_las30:
            logger.warning(
                "LAS 3.0 structured data sections (~Core, ~Drilling, "
                "~Inclinometry, etc.) found but VERS is '%s' (not a 3.x "
                "version). LAS 3.0 data handling is DISABLED — data in "
                "typed sections has been silently discarded. Set VERS to "
                "a 3.x value to enable LAS 3.0 processing.",
                self.las_file.version.vers,
            )

        # F-34: Cross-section consistency validation.
        self._validate_cross_section_consistency()

        # F-I2-M17: Cross-validate data_sections ↔ is_las30 consistency.
        # The parser correctly skips data processing for non-LAS 3.0 files
        # (data_reader handles LAS 1.2/2.0 data), so data_sections should
        # always be empty when is_las30 is False.  This check catches
        # inconsistencies that may have been introduced via from_dict()
        # or other construction paths — it serves as a defensive invariant
        # guard at the parser's public API boundary.
        if self.las_file.data_sections and not self.las_file.version.is_las30:
            logger.warning(
                "LASFile has %d data section(s) but is_las30 is False. "
                "Data sections are only valid for LAS 3.0 files. "
                "Data in these sections may be silently ignored by "
                "LAS 1.2/2.0 processing paths.",
                len(self.las_file.data_sections),
            )

        # F-07: Re-run validations skipped during incremental construction.
        # The parser populates curves_order and curves after __post_init__,
        # so index-curve validation (and other __post_init__ guards) never
        # saw the fully-populated state.  validate(complete=True) re-checks now.
        self.las_file.validate(complete=True)

        return self.las_file

    def _pre_scan(self, lines: list[str]) -> None:
        """Pre-scan to count ASCII data lines.

        Only counts lines in ~A / ~ASCII sections, matching the data reader's
        behavior: _read_normal breaks on any non-~A section header.  Counting
        lines in non-~A sections (e.g. ~Core, ~Drilling) would inflate the
        pre-allocation estimate.

        F-I2-M16: Track data line counts per contiguous ~A block rather
        than cumulatively across ALL ~A sections.  The data reader's
        _read_normal breaks at the FIRST non-~A section header, so a
        file with ``~A (data1) ~O (metadata) ~A (data2)`` only reads
        data1.  Cumulative counting across both sections produces an
        inflated pre-allocation estimate and diverges from the data
        reader's actual processing.
        """
        in_ascii = False
        count = 0
        per_block_counts: list[int] = []

        for line in lines:
            # M-02: Guard against absurdly long lines before any regex
            # processing in _pre_scan.  _parse_line has MAX_LINE_LENGTH
            # protection but _pre_scan runs first — a crafted 500MB
            # single-line file would crash here before _parse_line ever
            # sees it.  Skip the line; pre-scan is for estimation only.
            if len(line) > MAX_LINE_LENGTH:
                continue
            stripped = line.strip()
            match = SECTION_PATTERN.match(stripped)
            if match:
                section_word = match.group(1).upper()
                # Skip unrecognized section-like patterns so that control-
                # character noise does not break the ASCII-block count
                # (matching _read_normal's _is_recognized_section_word
                # behavior).  Recognized types include standard section
                # words and suffix-based types (_DEFINITION, _PARAMETER,
                # _PARAMETERS, _DATA).
                _base = section_word.split("[", 1)[0] if "[" in section_word else section_word
                if _base not in {"A", "ASCII", "V", "VERSION", "W", "WELL",
                                  "C", "CURVE", "P", "PARAMETER", "PARAMETERS",
                                  "O", "OTHER",
                                  # F-005: Section words recognized by
                                  # data_reader._KNOWN_SECTION_WORDS but
                                  # missing from _pre_scan — divergence
                                  # causes stale line-count estimates.
                                  "D", "DEFINITION",
                                  "CORE", "DRILLING", "FORMATION",
                                  "INCLINOMETRY", "LOG", "MUD",
                                  "PERFORATIONS", "RISK", "STRUCTURE",
                                  "TEST", "TOPS"}:
                    if re.search(r"_DEFINITION(_\d+)?$", _base):
                        pass
                    elif _base.endswith("_PARAMETER") or _base.endswith("_PARAMETERS") or _base.endswith("_DATA"):
                        pass
                    else:
                        continue
                is_ascii = section_word in {"A", "ASCII"}
                # F-I2-M10: Always save the count for the contiguous ~A block
                    # that just ended — zero-count blocks must be preserved so
                    # per_block_counts[0] correctly reflects the first block's
                    # actual count.  Skipping zero blocks caused
                    # per_block_counts[0] to pick up a later block's count,
                    # producing a data-size mismatch in _read_normal.
                if not is_ascii and in_ascii:
                    per_block_counts.append(count)
                    count = 0
                in_ascii = is_ascii
                continue
            if (
                in_ascii
                and not COMMENT_PATTERN.match(stripped)
                and not EMPTY_PATTERN.match(stripped)
            ):
                count += 1

        # F-I2-M10: Always append final block count — even zero.
        if in_ascii:
            per_block_counts.append(count)

        # Use the count from the first contiguous ~A block, matching
        # _read_normal's behavior (processes until first non-~A header).
        self._data_line_count = per_block_counts[0] if per_block_counts else 0

    def _parse_line(self, line: str) -> None:
        """Route a single line to the appropriate section handler."""
        # F-I2-M06: Guard against absurdly long lines before any regex
        # processing.  max_file_size bounds total file but crafted files
        # can place a 500MB payload in a single line, causing unbounded
        # regex backtracking and string allocation.
        if len(line) > MAX_LINE_LENGTH:
            raise LASParseError(
                f"Line length ({len(line)}) exceeds maximum allowed "
                f"({MAX_LINE_LENGTH}). The file may be malformed or corrupt."
            )
        # F-M-07: Strip leading whitespace before matching SECTION_PATTERN,
        # matching _pre_scan behavior.  Leading spaces would otherwise break
        # section header detection in the main parse pass.
        section_match = SECTION_PATTERN.match(line.strip())
        if section_match:
            section_word = section_match.group(1).upper()
            section_rest = section_match.group(2).strip()

            # Parse pipe-delimited definition association (LAS 3.0).
            # e.g., "~Core[1] | Core_Definition" or "~ASCII | CURVE".
            pipe_target: str | None = None
            pipe_idx = section_rest.find("|")
            if pipe_idx >= 0:
                # Split section name at pipe; pipe target follows.
                pipe_target = section_rest[pipe_idx + 1 :].strip()
                section_rest = section_rest[:pipe_idx].strip()
            elif "|" in section_word:
                # Edge case: no space before pipe (e.g., "~Core|Definition")
                pipe_idx_w = section_word.find("|")
                pipe_target = (section_word[pipe_idx_w + 1 :] + " " + section_rest).strip()
                section_word = section_word[:pipe_idx_w].strip()
                section_rest = ""

            # M-03: Save the previous data section type before the new-section
            # detection block overwrites it.  When two consecutive data sections
            # appear (e.g., ~A then ~Core[1]), this saved value is used to give
            # the PREVIOUS section's DataSection its correct section_type.
            _prev_data_section_type: str | None = None

            # Save curve indices of the PREVIOUS section before the new section's
            # pipe handler (below) may overwrite them for the new section.
            # _process_ascii_data for the previous section must use ITS indices,
            # not the new section's.
            _prev_curve_start = self._section_curve_start_idx
            _prev_curve_end = self._section_curve_end_idx

            # H-03: Save the previous _Definition's curve range BEFORE the
            # data section classification block so pipe target lookups
            # (line 567) can find the freshly-saved entry.  Without this,
            # the lookup runs before the save at line 665, producing false
            # "unrecognized pipe target" warnings and silent data corruption
            # when ~C (or _Definition) → data section transitions occur.
            # H-01: Also save non-_Definition ~C section ranges under a
            # sentinel key so consecutive non-_Definition ~C sections don't
            # silently lose curve scoping.
            _prev_def_name = self._current_definition_name
            if self._current_section == "C":
                start = self._section_curve_start_idx
                end = (
                    self._section_curve_end_idx
                    if self._section_curve_end_idx is not None
                    else len(self.las_file.curves)
                )
                if _prev_def_name is not None:
                    self._definition_curve_ranges[_prev_def_name] = (start, end)
                else:
                    # H-01: Non-_Definition ~C section — save under sentinel.
                    self._definition_curve_ranges["__MAIN__"] = (start, end)

            # Classify section_word for dispatch.
            # Standard sections (V, W, C, P, O, A) — also accept full
            # section-word names (VERSION, WELL, CURVE, PARAMETER, OTHER, ASCII).
            # LAS 3.0 structured definition sections → route to curve handler.
            # LAS 3.0 data-type sections → route to data handler.
            if section_word in {"V", "VERSION"}:
                new_section = "V"
                section_name = section_rest or ""
            elif section_word in {"W", "WELL"}:
                new_section = "W"
                section_name = section_rest or ""
            elif section_word in {"C", "CURVE"} or re.search(r"_DEFINITION(_\d+)?$", section_word):
                new_section = "C"
                if re.search(r"_DEFINITION(_\d+)?$", section_word):
                    section_name = f"{section_word} {section_rest}".strip()
                    # F-01: When the first _Definition section is encountered,
                    # freeze the main curve block endpoint so pipe "| CURVE"
                    # associations can reference it later.
                    if self._main_curve_end == -1:
                        self._main_curve_end = len(self.las_file.curves)
                    # G-02: Track which _Definition is active so curve ranges
                    # can be saved per-definition type (prevents overwrite
                    # by consecutive _Definition sections).
                    self._current_definition_name = section_word.upper()
                else:
                    section_name = section_rest or ""
                    # G-02: Regular ~C or ~CURVE section — no definition name.
                    self._current_definition_name = None
            elif section_word in {"P", "PARAMETER", "PARAMETERS"} or section_word.endswith(
                "_PARAMETER"
            ) or section_word.endswith("_PARAMETERS"):
                # F-M-01: LAS 3.0 typed parameter sections (e.g.,
                # ~Core_Parameter, ~Drilling_Parameter) route to the
                # parameter parser like standard ~P/~Parameter sections.
                new_section = "P"
                section_name = section_rest or ""
            elif section_word in {"O", "OTHER"}:
                new_section = "O"
                section_name = section_rest or ""
                # F-05: ~Other is DEPRECATED and NOT ALLOWED in LAS 3.0.
                if self.las_file.version.is_las30:
                    _other_line = line[:80].strip()
                    raise LASParseError(
                        f"~Other section at line '{_other_line}' — ~Other is "
                        f"NOT ALLOWED in LAS 3.0.  Migrate content to "
                        f"user-defined Parameter or Column Data sections."
                    )
            elif (
                section_word in _DATA_SECTION_WORDS
                or section_word.endswith("_DATA")
                or _is_indexed_data_section(section_word)
            ):
                # LAS 3.0 structured data sections and standard ~A/ASCII.
                # Also matches: user-defined ~{Root}_Data sections (F-03)
                # and indexed sections like ~Core[1], ~Inclinometry[2] (F-03).
                new_section = "A"
                # M-PA4: Track LAS 3.0 typed data sections for
                # cross-validation against VERS.  Only non-standard
                # (non-~A/~ASCII) data sections indicate LAS 3.0 intent.
                if section_word not in ("A", "ASCII"):
                    self._las30_sections_seen = True
                # For standard 'A' or 'ASCII' sections and written-form *_DATA
                # sections (e.g., ~DRILLING_DATA DRILLING), use only the rest
                # as the section name (backward-compatible).
                if section_word in {"A", "ASCII"} or section_word.endswith("_DATA"):
                    section_name = section_rest
                else:
                    section_name = (
                        f"{section_word} {section_rest}".strip() if section_rest else section_word
                    )
                # M-03: Save the previous data section type BEFORE overwriting.
                # When two consecutive data sections appear (e.g., ~A then
                # ~Core[1]),_process_ascii_data() for the PREVIOUS section
                # must use ITS type, not the newly-set one.
                _prev_data_section_type = self._current_data_section_type
                # Derive section_type from keyword, falling back to LOG_DATA.
                # F-03: Handle indexed sections (e.g., CORE[1] → CORE_DATA).
                # F-M-03: Warn when section_word is not a recognized type
                # before defaulting to LOG_DATA.
                bracket_idx = section_word.find("[")
                if bracket_idx >= 0:
                    base_type = section_word[:bracket_idx]
                    base_with_data = f"{base_type}_DATA"
                    stype = _SECTION_TYPE_MAP.get(base_type)
                    if stype is None:
                        stype = _SECTION_TYPE_MAP.get(base_with_data)
                    if stype is None:
                        warnings.warn(
                            f"Unrecognized data section type '~{section_word}', "
                            f"defaulting to LOG_DATA.",
                            UserWarning,
                            stacklevel=2,
                        )
                        stype = "LOG_DATA"
                    self._current_data_section_type = stype
                else:
                    stype = _SECTION_TYPE_MAP.get(section_word)
                    if stype is None:
                        # User-defined _DATA-suffixed sections (e.g.,
                        # ~Custom_Data) are structurally valid LAS 3.0 data
                        # sections.  Preserve the original section word as
                        # the type so the writer can reconstruct the correct
                        # header prefix on roundtrip.
                        if section_word.endswith("_DATA"):
                            stype = section_word
                            warnings.warn(
                                f"Unrecognized data section type '~{section_word}', "
                                f"preserving as '{stype}'.",
                                UserWarning,
                                stacklevel=2,
                            )
                        else:
                            warnings.warn(
                                f"Unrecognized data section type '~{section_word}', "
                                f"defaulting to LOG_DATA.",
                                UserWarning,
                                stacklevel=2,
                            )
                            stype = "LOG_DATA"
                    self._current_data_section_type = stype
                # F-08: Handle pipe-delimited definition association.
                # "| CURVE" means use the main curve block (before
                # _Definition sections). "| X_Definition" means use
                # the per-section curves from that definition block.
                if pipe_target:
                    pipe_target_upper = pipe_target.upper()
                    if pipe_target_upper in {"CURVE", "C"}:
                        # Pipe "| CURVE" → use main curve block only.
                        self._section_curve_start_idx = 0
                        self._section_curve_end_idx = (
                            self._main_curve_end if self._main_curve_end >= 0 else None
                        )
                    elif pipe_target_upper in self._definition_curve_ranges:
                        # G-02: Explicit pipe to a known _Definition —
                        # look up the saved (start, end) range.
                        start, end = self._definition_curve_ranges[pipe_target_upper]
                        self._section_curve_start_idx = start
                        self._section_curve_end_idx = end
                    else:
                        # Unrecognized pipe target (not "CURVE"/"C" and
                        # not a known _Definition).  Reset curve indices
                        # to safe defaults so subsequent data routing
                        # doesn't use stale values from a previous section.
                        source_info = f" in {self.source_file}" if self.source_file else ""
                        logger.warning(
                            "Unrecognized pipe target '| %s' for section '~%s'%s — "
                            "curve indices reset to defaults.",
                            pipe_target,
                            section_word,
                            source_info,
                        )
                        self._section_curve_start_idx = 0
                        self._section_curve_end_idx = None  # F-051: None → all curves (0 → empty slice)
                elif self._current_data_section_type != "LOG_DATA":
                    # G-02: No pipe — try to match data section type
                    # to a _Definition (e.g., CORE_DATA → CORE_DEFINITION).
                    def_prefix = (
                        self._current_data_section_type.replace("_DATA", "") + "_DEFINITION"
                    )
                    if def_prefix in self._definition_curve_ranges:
                        start, end = self._definition_curve_ranges[def_prefix]
                        self._section_curve_start_idx = start
                        self._section_curve_end_idx = end
                    elif "__MAIN__" in self._definition_curve_ranges:
                        # H-01: No matching _Definition found — fall back
                        # to the main non-_Definition curve range.
                        start, end = self._definition_curve_ranges["__MAIN__"]
                        self._section_curve_start_idx = start
                        self._section_curve_end_idx = end
                elif "__MAIN__" in self._definition_curve_ranges:
                    # H-01: LOG_DATA section with no pipe — fall back to
                    # the main non-_Definition curve range to avoid
                    # silently losing curve scoping when the previous
                    # section was a typed data section with different
                    # curve indices.
                    start, end = self._definition_curve_ranges["__MAIN__"]
                    self._section_curve_start_idx = start
                    self._section_curve_end_idx = end
            else:
                # Unknown section type — accumulate lines as free-form text (like ~O).
                new_section = None
                # F-M01: Warn when a pipe target is specified on an
                # unknown/non-data section.  Pipe annotations are only
                # meaningful for data sections (~A, ~ASCII, etc.).
                if pipe_target:
                    source_info = f" in {self.source_file}" if self.source_file else ""
                    logger.warning(
                        "Pipe target '| %s' on non-data section '~%s'%s — "
                        "pipe annotations are only meaningful for data "
                        "sections. The pipe target will be ignored.",
                        pipe_target,
                        section_word,
                        source_info,
                    )
                section_name = f"{section_word} {section_rest}".strip()
                self._append_other_line(line)
                source_info = f" in {self.source_file}" if self.source_file else ""
                logger.warning(
                    "Unknown section type '~%s' at line%s — lines accumulated as other section: %s",
                    section_word,
                    source_info,
                    line[:80],
                )
                # F-02: Process any pending ASCII data from the current section
                # before resetting, to avoid leaving orphaned data lines and to
                # prevent LAS 3.0 contamination (F-08).
                if self._current_section == "A" and self._ascii_data_lines:
                    # F-I2-H01: Replay deferred well entries before processing
                    # data so the correct NULL value is available.  When ~W
                    # appears before ~V, well entries are buffered and not
                    # stored until _replay_deferred_well() is called.
                    if self._version_found:
                        self._replay_deferred_well()
                    try:
                        ctx = AsciiDataContext(
                            las_file=self.las_file,
                            ascii_data_lines=self._ascii_data_lines,
                            section_curve_start_idx=self._section_curve_start_idx,
                            section_curve_end_idx=self._section_curve_end_idx,
                            current_section_name=self._current_section_name,
                            current_data_section_type=self._current_data_section_type,
                            current_data_section_idx=self._current_data_section_idx,
                            cumulative_elements=self._cumulative_elements,
                        )
                        process_ascii_data(ctx)
                        self._cumulative_elements = ctx.cumulative_elements
                    finally:
                        self._ascii_data_lines = []
                        self._current_data_section_idx += 1
                # F-02: Reset current section so data lines in unknown sections
                # aren't misrouted to the previous section's handler.
                self._current_section = None
                return

            # F-M01: Warn when a pipe target is specified on a known
            # non-data section (V, W, C, P, O).  Pipe annotations are
            # only meaningful for data sections and are silently ignored
            # here — the pipe_target variable is only consumed in the
            # data-section branch (lines 684-737).
            if pipe_target and new_section not in ("A", None):
                source_info = f" in {self.source_file}" if self.source_file else ""
                logger.warning(
                    "Pipe target '| %s' on non-data section '~%s'%s — "
                    "pipe annotations are only meaningful for data "
                    "sections. The pipe target will be ignored.",
                    pipe_target,
                    section_word,
                    source_info,
                )

            # F1: When leaving a data section for a non-data section,
            # process pending data from the previous section.
            if new_section is not None:
                if self._current_section == "A" and new_section != "A":
                    # F-I2-H01: Replay deferred well entries before processing
                    # data so the correct NULL value is available.
                    if self._version_found:
                        self._replay_deferred_well()
                    try:
                        ctx = AsciiDataContext(
                            las_file=self.las_file,
                            ascii_data_lines=self._ascii_data_lines,
                            section_curve_start_idx=self._section_curve_start_idx,
                            section_curve_end_idx=self._section_curve_end_idx,
                            current_section_name=self._current_section_name,
                            current_data_section_type=self._current_data_section_type,
                            current_data_section_idx=self._current_data_section_idx,
                            cumulative_elements=self._cumulative_elements,
                        )
                        process_ascii_data(ctx)
                        self._cumulative_elements = ctx.cumulative_elements
                    finally:
                        self._ascii_data_lines = []
                        self._current_data_section_idx += 1
                elif new_section == "A" and self._current_section == "A":
                    # Consecutive data sections — process previous first.
                    # M-03: _current_data_section_type was already overwritten
                    # for the newly-detected section (lines 337-345).  Swap
                    # back to the saved previous-section type while processing
                    # the previous section's accumulated data lines, then
                    # restore the new-section type for the rest of parsing.
                    #
                    # Also swap in the PREVIOUS section's curve indices so
                    # _process_ascii_data scopes curves correctly.  The new
                    # section's pipe handler may have already overwritten
                    # _section_curve_start_idx/_end_idx for the new section.
                    _new_curve_start = self._section_curve_start_idx
                    _new_curve_end = self._section_curve_end_idx
                    self._section_curve_start_idx = _prev_curve_start
                    self._section_curve_end_idx = _prev_curve_end
                    try:
                        if _prev_data_section_type is not None:
                            _new_type = self._current_data_section_type
                            self._current_data_section_type = _prev_data_section_type
                            # F-I2-H01: Replay deferred well entries before processing
                            # data so the correct NULL value is available.
                            if self._version_found:
                                self._replay_deferred_well()
                            ctx = AsciiDataContext(
                                las_file=self.las_file,
                                ascii_data_lines=self._ascii_data_lines,
                                section_curve_start_idx=self._section_curve_start_idx,
                                section_curve_end_idx=self._section_curve_end_idx,
                                current_section_name=self._current_section_name,
                                current_data_section_type=self._current_data_section_type,
                                current_data_section_idx=self._current_data_section_idx,
                                cumulative_elements=self._cumulative_elements,
                            )
                            process_ascii_data(ctx)
                            self._cumulative_elements = ctx.cumulative_elements
                            self._current_data_section_type = _new_type
                        else:
                            # F-I2-H01: Same replay guard for the else branch.
                            if self._version_found:
                                self._replay_deferred_well()
                            ctx = AsciiDataContext(
                                las_file=self.las_file,
                                ascii_data_lines=self._ascii_data_lines,
                                section_curve_start_idx=self._section_curve_start_idx,
                                section_curve_end_idx=self._section_curve_end_idx,
                                current_section_name=self._current_section_name,
                                current_data_section_type=self._current_data_section_type,
                                current_data_section_idx=self._current_data_section_idx,
                                cumulative_elements=self._cumulative_elements,
                            )
                            process_ascii_data(ctx)
                            self._cumulative_elements = ctx.cumulative_elements
                    finally:
                        self._section_curve_start_idx = _new_curve_start
                        self._section_curve_end_idx = _new_curve_end
                        self._ascii_data_lines = []
                        self._current_data_section_idx += 1

                # F1: Track per-section curve boundaries for LAS 3.0.
                # When entering ~C (including _Definition sections), mark the
                # current curve list position for per-section curve scoping.

                # G-02/H-03: Before overwriting _section_curve_start_idx for
                # the new ~C section, save the previous _Definition's curve
                # range so later data sections can look up the correct range.
                # Uses _prev_def_name (captured BEFORE classification) to
                # avoid using a freshly-overwritten _current_definition_name
                # when consecutive _Definition sections are parsed.
                if self._current_section == "C" and _prev_def_name is not None:
                    # Guard: verify the definition name hasn't been changed
                    # by a consecutive _Definition overwrite during
                    # classification.
                    if (
                        self._current_definition_name is not None
                        and self._current_definition_name != _prev_def_name
                    ):
                        logger.warning(
                            "Definition name mismatch during curve range save: "
                            "previous='%s', current='%s'. Using previous for "
                            "range storage.",
                            _prev_def_name,
                            self._current_definition_name,
                        )
                    # F-01: Use _prev_curve_start/_prev_curve_end instead of
                    # self._section_curve_start_idx/_end_idx, which may have
                    # been overwritten by pipe handling (lines 602-657) for
                    # the new section.  The prev locals were captured at
                    # lines 456-457 BEFORE pipe handler ran.
                    self._definition_curve_ranges[_prev_def_name] = (
                        _prev_curve_start,
                        _prev_curve_end
                        if _prev_curve_end is not None
                        else len(self.las_file.curves),
                    )
                elif self._current_section == "C" and _prev_def_name is None:
                    # H-01: Non-_Definition ~C section — save under sentinel
                    # key so consecutive non-_Definition ~C sections don't
                    # silently lose curve scoping.
                    # F-01: Same as above — use preserved locals.
                    self._definition_curve_ranges["__MAIN__"] = (
                        _prev_curve_start,
                        _prev_curve_end
                        if _prev_curve_end is not None
                        else len(self.las_file.curves),
                    )

                if new_section == "C":
                    self._section_curve_start_idx = len(self.las_file.curves)
                    self._section_curve_end_idx = None

                self._current_section = new_section
                # F-M27: For parameter sections, _current_section_name must
                # preserve the section_word (e.g., CORE_PARAMETERS) for type
                # derivation in _parse_parameter_entry.  section_rest is
                # annotation text, not the type identifier.
                if new_section == "P" and (
                    section_word.endswith("_PARAMETER") or section_word.endswith("_PARAMETERS")
                ):
                    self._current_section_name = section_word
                else:
                    self._current_section_name = section_name.strip() if section_name else section_word
                # F-34: Track section sequence for cross-section validation.
                # F-I2-M11: Use section_word (e.g. "CURVE", "VERSION")
                # instead of just new_section (single letter "C", "V")
                # so that ~C and ~CURVE produce distinct labels instead
                # of both collapsing to "C".
                section_label = (
                    f"{section_word}:{section_name}" if section_name.strip()
                    else section_word
                )
                # F-006: Guard against unbounded section sequence growth
                # from repeated unknown section headers.
                if len(self._section_sequence) >= MAX_SECTION_SEQUENCE:
                    raise LASParseError(
                        f"Section sequence length ({len(self._section_sequence) + 1}) "
                        f"exceeds maximum allowed ({MAX_SECTION_SEQUENCE}). "
                        f"The file may be malformed or corrupt."
                    )
                self._section_sequence.append(section_label)
                # F-048/F-103: Track semantic section type for duplicate detection.
                # Labels differ for same type (e.g., "V:" vs "VERSION:INFORMATION")
                # but new_section is the normalized type code ("V").
                self._section_type_sequence.append(new_section)
                return

        if COMMENT_PATTERN.match(line) or EMPTY_PATTERN.match(line):
            return

        # F-83: Lines starting with ~ that bypassed SECTION_PATTERN
        # (e.g., ~., ~#, ~/ — tilde followed by non-letter) are not
        # valid data lines.  Route them to _other_lines so they don't
        # produce corrupt rows in data sections.
        if line.strip().startswith("~"):
            self._append_other_line(line)
            return

        if self._current_section:
            handler_name = self.SECTION_HANDLERS.get(self._current_section)
            if handler_name:
                getattr(self, handler_name)(line)
            else:
                # Fallback for unknown section types — accumulate in _other_lines.
                self._append_other_line(line)
        else:
            # F-02/F-08: Data lines without an active section (e.g., body lines
            # of an unknown section where _current_section was reset to None).
            # Accumulate as free-form text (like ~O).
            self._append_other_line(line)

    def _append_other_line(self, line: str) -> None:
        """Append a line to _other_lines with a bounds check (F-M-02)."""
        if len(self._other_lines) >= MAX_OTHER_LINES:
            raise LASParseError(
                f"Other section line count ({len(self._other_lines) + 1}) exceeds "
                f"maximum allowed ({MAX_OTHER_LINES}). "
                f"The file may be malformed or corrupt."
            )
        self._other_lines.append(line)

    @staticmethod
    def _manual_colon_scan(line: str) -> dict[str, str] | None:
        """Manual scan for colon separator on long lines (I2F-01 defense).

        Scans the line right-to-left for the LAST colon that has
        whitespace on at least one side (or is at end of line).  Using
        the LAST structurally-valid colon matches the CWLS LAS 2.0
        specification and correctly handles header lines with embedded
        colons in description fields (e.g., "MNEM.UNIT VAL : DESC : MORE"
        → desc = "DESC : MORE").  O(n) with no backtracking — guaranteed
        safe regardless of input.  Returns a dict with 'mnemonic',
        'unit', 'value', 'description' keys, or None if the line doesn't
        match the data-line pattern.

        Intended as a fallback for lines too long for safe regex matching
        (>_SAFE_REGEX_LINE_LENGTH).
        """
        # Find the last colon that has whitespace on at least one side.
        colon_idx = -1
        stripped = line.rstrip()
        for i in range(len(stripped) - 1, -1, -1):
            ch = stripped[i]
            if ch == ":":
                has_ws_before = i > 0 and stripped[i - 1].isspace()
                has_ws_after = i + 1 < len(stripped) and stripped[i + 1].isspace()
                # Also accept colon at end of line with optional trailing space.
                rest_after = stripped[i + 1 :].strip()
                if has_ws_before or has_ws_after or not rest_after:
                    colon_idx = i
                    break
        if colon_idx < 0:
            return None

        # Split at the colon separator.
        prefix = stripped[:colon_idx].rstrip()
        description = stripped[colon_idx + 1 :].strip()

        # Parse mnemonic and unit from the prefix.
        # Format: [whitespace] MNEMONIC [whitespace] . UNIT [whitespace] VALUE
        dot_idx = prefix.find(".")
        if dot_idx < 0:
            return None

        mnemonic = prefix[:dot_idx].strip()
        # PXM-03: A line starting with a dot produces an empty mnemonic
        # which would cause CurveDefinition.__post_init__ to raise an
        # uncaught ValueError escaping the parser boundary.
        if not mnemonic:
            return None
        after_dot = prefix[dot_idx + 1 :].strip()

        # Unit is the contiguous word chars + hyphens + slashes before the
        # first value whitespace.  Value is everything after that whitespace.
        # PD1-01: Require letter start so a purely numeric value string
        # (e.g. ``123.45``) is not greedily consumed as a unit name.
        unit_match_result = re.match(r"([a-zA-Z][\w\-/]*)", after_dot)
        if unit_match_result:
            unit = unit_match_result.group(1)
            value = after_dot[len(unit) :].strip()
        else:
            unit = ""
            value = after_dot.strip()

        return {
            "mnemonic": mnemonic,
            "unit": unit,
            "value": value,
            "description": description,
        }

    def _match_data_line(self, line: str) -> re.Match[str] | None:
        """Try to match a header data line with colon, then without.

        F-I2-M07: Validates captured group lengths after a successful
        match to prevent unbounded string allocation from crafted files
        with 500MB values or descriptions.

        I2F-01: Lines exceeding _SAFE_REGEX_LINE_LENGTH bypass regex
        matching and use a manual O(n) colon scan to prevent
        catastrophic backtracking.
        """
        if len(line) > _SAFE_REGEX_LINE_LENGTH:
            return self._match_data_line_manual(line)
        match = DATA_LINE_PATTERN.match(line)
        if match:
            self._validate_data_line_fields(match)
            return match
        match = VALUE_ONLY_PATTERN.match(line)
        if match:
            self._validate_data_line_fields(match)
            return match
        return None

    def _match_data_line_manual(self, line: str) -> re.Match[str] | None:
        """Match a data line using manual scan (long-line fallback, I2F-01)."""
        colon_result = self._manual_colon_scan(line)
        if colon_result is not None:
            # Wrap the result in a dict-like object that quacks like a
            # regex match for the groupdict()/group() protocol used by
            # callers in _parse_version, _parse_well, _parse_curve,
            # and _parse_parameter.
            class _ManualMatch:
                """Minimal regex-match duck-type for manual scan results."""
                __slots__ = ("_data",)

                def __init__(self, data: dict[str, str]) -> None:
                    self._data = data

                def group(self, name: str) -> str:
                    return self._data.get(name, "")

                def groupdict(self) -> dict[str, str]:
                    return dict(self._data)

                def __getitem__(self, name: str) -> str:
                    return self._data[name]

            match: re.Match[str] = _ManualMatch(colon_result)  # type: ignore[assignment]
            self._validate_data_line_fields(match)
            return match

        # Try without colon (VALUE_ONLY_PATTERN equivalent).
        # Match mnemonic.unit value format.
        m = re.match(
            r"^\s*(?P<mnemonic>[\w\-]+(?:\[\d+\])?)\s*\.(?P<unit>[\w\-/]*)\s+(?P<value>.+?)\s*$",
            line,
        )
        if m:
            self._validate_data_line_fields(m)
            return m
        return None

    def _validate_data_line_fields(self, match: re.Match[str]) -> None:
        """Validate captured group lengths in a data-line match (F-I2-M07).

        DATA_LINE_PATTERN uses unbounded .*? groups for value and
        description; a crafted file can cause multi-megabyte allocations
        in a single captured group.  This guard checks each named group
        independently.

        Raises:
            LASParseError: If any captured group exceeds MAX_FIELD_LENGTH.
        """
        groupdict = match.groupdict()
        for group_name in ("value", "description", "mnemonic", "unit"):
            val = groupdict.get(group_name)
            if val and len(val) > MAX_FIELD_LENGTH:
                raise LASParseError(
                    f"Field '{group_name}' length ({len(val)}) exceeds "
                    f"maximum allowed ({MAX_FIELD_LENGTH}). "
                    f"The file may be malformed or corrupt."
                )

    def _parse_version(self, line: str) -> None:
        """Parse ~V (version) section line."""
        match = self._match_data_line(line)
        if not match:
            logger.warning(
                "Non-matching ~V line in %s: %s",
                self.source_file or "<unknown>",
                line.strip()[:120],
            )
            return

        # F-10: Extract mnemonic and value before any flag-setting.
        # M-05: _version_found is set only for VERS (below), not for
        # any ~V data line (e.g., VERT, VMIN, VDATA).
        mnemonic = match.group("mnemonic").upper().strip()
        value = match.group("value").strip()

        if mnemonic == "VERS":
            # M-05: Only set _version_found for VERS, not other ~V data.
            self._version_found = True

            # F-005: Validate VERS against known LAS versions.
            # Non-standard values (e.g., "1,2" with comma, "2,0")
            # silently fail all startswith() checks, causing the
            # parser to use LAS 1.2 well-field conventions for LAS 2.0
            # files and discard LAS 3.0 data sections entirely.
            # Accept known versions ("1.2", "2.0", "3.0") silently.
            # Accept version-like strings (digit.digit pattern, e.g.,
            # "1.20", "4.0") with a warning — the reader will later
            # emit its own "not officially supported" warning for
            # unrecognized versions.
            # Non-numeric values (e.g., "CWLS LOG ASCII STANDARD" from
            # mis-formatted version lines) are preserved as-is for
            # backward compatibility.
            # For completely non-standard values (commas, no dot, etc.),
            # warn and default to "2.0".
            vers_normalized = value.strip()
            # F-03: Normalize three-segment versions (e.g., "1.2.0" → "1.2",
            # "2.0.1" → "2.0") by stripping the third segment before the
            # regex check.  Only three-dot-segment strings are affected;
            # two-segment versions like "1.2" or "1.20" are unchanged.
            vers_normalized = re.sub(
                r"^(\d+\.\d+)\.\d+$", r"\1", vers_normalized
            )
            if vers_normalized in {"1.2", "2.0", "3.0"}:
                self.las_file.version.vers = vers_normalized
            elif vers_normalized.startswith("3."):
                # I2F-02: LAS 3.x draft versions (e.g., "3.1beta",
                # "3.0-draft").  The is_las30 property (models.py:300)
                # explicitly documents acceptance of any string starting
                # with '3' as LAS 3.0 to support draft/development
                # versions.  Without this branch, values like "3.1beta"
                # failed the ^\d+\.\d+$ regex (beta is non-numeric) and
                # silently defaulted to "2.0", completely disabling
                # LAS 3.0 processing.
                warnings.warn(
                    f"Non-standard VERS value '{value}'. "
                    f"Expected '3.0'. Accepting as LAS 3.x for "
                    f"compatibility with draft/development versions.",
                    UserWarning,
                    stacklevel=2,
                )
                self.las_file.version.vers = vers_normalized
            elif re.match(r"^\d+\.\d+$", vers_normalized):
                # Version-like but non-standard (e.g., "1.20", "4.0") —
                # warn but preserve the value.  The reader will emit its
                # own "not officially supported" warning for unrecognized
                # versions.
                warnings.warn(
                    f"Non-standard VERS value '{value}'. "
                    f"Expected 1.2, 2.0, or 3.0. "
                    f"Preserving as-is for backward compatibility.",
                    UserWarning,
                    stacklevel=2,
                )
                self.las_file.version.vers = vers_normalized
            elif vers_normalized and not vers_normalized[0].isdigit():
                # Non-numeric VERS value (e.g., description text parsed
                # as VERS value due to malformed version line).  Preserve
                # as-is — the is_las30/is_las12 false-negative is the
                # intended fallback.
                self.las_file.version.vers = vers_normalized
            else:
                warnings.warn(
                    f"Unknown VERS value '{value}'. Expected 1.2, 2.0, or 3.0. "
                    f"Defaulting to 2.0.",
                    UserWarning,
                    stacklevel=2,
                )
                self.las_file.version.vers = "2.0"
            # F-020 pt2: Deferred DLM re-check.  If DLM was parsed BEFORE
            # VERS in non-standard ~V ordering, the DLM version guard at
            # L1501 uses the default "2.0" and incorrectly allows
            # non-SPACE DLM.  Now that the true version is known, re-validate.
            if _LASVersionSpec(
                self.las_file.version.vers
            ).is_las12 and self.las_file.version.dlm != "SPACE":
                _prev_dlm = self.las_file.version.dlm
                self.las_file.version.dlm = "SPACE"
                warnings.warn(
                    f"DLM value '{_prev_dlm}' was set before VERS was parsed "
                    f"and is not supported in LAS 1.2. "
                    f"Resetting DLM to SPACE.",
                    UserWarning,
                    stacklevel=2,
                )
        elif mnemonic == "WRAP":
            wrap_upper = value.upper()
            if wrap_upper in {"YES", "NO"}:
                self.las_file.version.wrap = wrap_upper
            else:
                warnings.warn(
                    f"Unknown WRAP value '{value}'. Expected YES or NO. "
                    f"Defaulting to NO.",
                    UserWarning,
                    stacklevel=2,
                )
                self.las_file.version.wrap = "NO"
        elif mnemonic == "DLM":
            dlm_upper = value.upper()
            if dlm_upper in {"SPACE", "TAB", "COMMA"}:
                if (
                    not _LASVersionSpec(
                        self.las_file.version.vers
                    ).is_las12
                    or dlm_upper == "SPACE"
                ):
                    self.las_file.version.dlm = dlm_upper
                else:
                    warnings.warn(
                        f"DLM value '{value}' is not supported in LAS 1.2 "
                        f"(only SPACE is allowed). DLM will be ignored "
                        f"for this file.",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    f"Unknown DLM value '{value}'. Expected SPACE, TAB, or COMMA. "
                    f"Defaulting to SPACE.",
                    UserWarning,
                    stacklevel=2,
                )
                self.las_file.version.dlm = "SPACE"
        else:
            # F-M07: Warn about unknown ~V fields (PROG, PROD, LIC, etc.).
            # Extended ~V mnemonics beyond VERS/WRAP/DLM are silently discarded
            # by the parser — there is no extensible storage on VersionSection
            # for custom fields.  Warnings alert users that roundtrip fidelity
            # will be lost for non-standard ~V entries.
            warnings.warn(
                f"Unknown ~V field '{mnemonic}={value}' — only VERS, WRAP, and DLM "
                f"are recognized. Extended ~V fields (PROG, PROD, LIC, etc.) are "
                f"not stored and will be lost on roundtrip.",
                UserWarning,
                stacklevel=2,
            )

    def _store_well_entry(
        self, mnemonic: str, unit: str, value: str, description: str | None, is_las12: bool
    ) -> None:
        """Store a well entry with version-appropriate value/description handling.

        Extracted from _parse_well to support deferred well processing when
        ~W appears before ~V (the version check is deferred until ~V is parsed).
        """
        # F-022: Unescape colon artifacts inserted by the writer's
        # _escape_colons_for_las_value BEFORE the CWLS/lasio swap logic.
        # Escaped patterns like " _:_ " would confuse the space/digit
        # heuristics used in auto-mode detection.
        value = _desanitize_las_value(_unescape_colons_for_las_value(value))
        if description is not None:
            description = _desanitize_las_value(_unescape_colons_for_las_value(description))

        # F-P06 / R7F-05: Bare-colon CWLS detection moved from _parse_well into
        # _store_well_entry so deferred well entries (parsed before ~V is known)
        # always receive correct handling.  _store_well_entry receives the
        # authoritative `is_las12` flag regardless of when the version is
        # resolved.  When description is None and value contains a bare colon,
        # this may be a CWLS/lasio "DESCRIPTION : VALUE" format with no
        # whitespace around the colon.  Split at the first colon so the CWLS
        # swap logic below (which requires `description is not None`) can
        # trigger.  Timestamp/datetime values like "12:34:56" or
        # "2026-07-19T12:34:56" must NOT be split.
        if is_las12 and description is None and ":" in value:
            _is_timestamp = (
                ":" in value
                and "T" in value
                and bool(re.search(r"T\d{2}:", value))
            )
            if not _is_timestamp:
                # F-003: Check the full value for bare hh:mm[:ss]
                # timestamp patterns BEFORE the partial post-colon
                # check.  A value like "12:34" has only one colon —
                # the post-colon portion "34" won't match
                # \b\d{1,2}:\d{2}\b, causing the bare-colon split to
                # corrupt the value (stored as "34" instead of "12:34").
                _is_timestamp = bool(
                    re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", value)
                )
            if not _is_timestamp:
                # F-002: Scope the timestamp regex to the portion after
                # the first colon to avoid false positives like "15:01"
                # inside a date such as "15/01/2001" in CWLS well entries.
                _time_match = re.search(
                    r"\b\d{1,2}:\d{2}(:\d{2})?\b",
                    value[value.index(":") + 1 :],
                )
                _is_timestamp = _time_match is not None

            if not _is_timestamp:
                colon_idx = value.index(":")
                description = value[colon_idx + 1 :].strip()
                value = value[:colon_idx].strip()

        # F-37-upgrade: Guard against unbounded well entry accumulation.
        # models.py:from_dict() has 3 MAX_WELL_ENTRIES checks (in
        # _validate_single_section, _validate_top_level, and per-section
        # string_data); the parser path had zero protection.
        if len(self.las_file.well.entries) >= MAX_PARAMETERS:
            raise LASParseError(
                f"Well entry count ({len(self.las_file.well.entries) + 1}) exceeds "
                f"maximum allowed ({MAX_PARAMETERS}). "
                f"The file may be malformed or corrupt."
            )

        # F-07: Use `description is not None` instead of truthiness check.
        # A colon-bearing line with empty post-colon text produces
        # description="" (falsy but not None), which must still trigger
        # the LAS 1.2 CWLS/lasio swap.  VALUE_ONLY_PATTERN (no colon)
        # produces description=None, correctly skipping the swap.
        if is_las12 and description is not None:
            # F-004/F-005 structural fix: hoist _well_format above the
            # numeric/non-numeric split.  Previously the cwls/lasio
            # branches were duplicated identically in both blocks (lines
            # 901-904 and 917-920), creating a regression surface where
            # a fix in one block could be missed in the other.
            if self._well_format == "cwls":
                if mnemonic in {"STRT", "STOP", "STEP", "NULL"}:
                    # Mandatory numeric fields: VALUE before colon in CWLS.
                    actual_value = value
                    self.las_file.well.descriptions[mnemonic] = description
                else:
                    # F-004: Non-mandatory CWLS fields: VALUE after colon,
                    # DESCRIPTION before colon.  In CWLS format, non-mandatory
                    # fields like DATE, LOC, etc. place the value after the
                    # colon (e.g. "DATE.   .LOG DATE :15/01/2001").  The
                    # previous code stored pre-colon text as the value for
                    # ALL CWLS fields — correct for mandatory but wrong for
                    # non-mandatory where the value is post-colon.
                    actual_value = description
                    self.las_file.well.descriptions[mnemonic] = value
            elif self._well_format == "lasio":
                # lasio convention: VALUE after colon (same as LAS 2.0+).
                # STRT/STOP/STEP/NULL are mandatory numeric fields where
                # lasio reverses to value:descr order (matching CWLS).
                if mnemonic in {"STRT", "STOP", "STEP", "NULL"}:
                    actual_value = value
                    self.las_file.well.descriptions[mnemonic] = description
                else:
                    actual_value = description
                    self.las_file.well.descriptions[mnemonic] = value
            else:
                # Auto-mode: detect CWLS vs lasio convention heuristically.
                if mnemonic in {"STRT", "STOP", "STEP", "NULL"}:
                    # Float-based numeric detection for mandatory fields.
                    # If pre-colon text parses as float → CWLS (value in
                    # correct position); otherwise → lasio (swap).
                    try:
                        _parse_float_with_d_notation(value)
                    except ValueError:
                        actual_value = description
                        self.las_file.well.descriptions[mnemonic] = value
                    else:
                        actual_value = value
                        self.las_file.well.descriptions[mnemonic] = description
                else:
                    # F-005: Improved auto-mode heuristic for non-mandatory
                    # fields.  The old heuristic (" " in value) only checked
                    # pre-colon text for spaces — both CWLS descriptions
                    # (e.g. "ANY OIL COMPANY") and lasio values can contain
                    # spaces, causing false positives in both directions.
                    # Better: check both sides for data-like patterns:
                    #   - CWLS non-mandatory: "DESCRIPTION : VALUE"
                    #     → post-colon text (VALUE) has digits, no spaces
                    #   - lasio: "VALUE : DESCRIPTION"
                    #     → pre-colon text (VALUE) has digits, no spaces
                    value_has_spaces = " " in value
                    value_has_digits = any(c.isdigit() for c in value)
                    desc_has_spaces = " " in description
                    desc_has_digits = any(c.isdigit() for c in description)

                    if desc_has_digits and not value_has_digits and not desc_has_spaces:
                        # Post-colon looks like data (has digits, no spaces)
                        # = CWLS format with value after colon.
                        # Swap to match explicit CWLS non-mandatory branch
                        # (lines 918-919): post-colon text = VALUE,
                        # pre-colon text = DESCRIPTION.
                        actual_value = description
                        self.las_file.well.descriptions[mnemonic] = value
                    elif value_has_digits and not desc_has_digits and not value_has_spaces:
                        # Pre-colon looks like data (has digits, no spaces)
                        # = lasio format with value before colon.
                        actual_value = description
                        self.las_file.well.descriptions[mnemonic] = value
                    elif value_has_spaces and not desc_has_spaces:
                        # Multi-word pre-colon, single-word post-colon
                        # = likely CWLS description before colon (VALUE after).
                        # Swap to match explicit CWLS non-mandatory branch
                        # (lines 918-919): post-colon text = VALUE,
                        # pre-colon text = DESCRIPTION.
                        actual_value = description
                        self.las_file.well.descriptions[mnemonic] = value
                    else:
                        # Ambiguous — default to lasio convention and warn.
                        # Include description in warning for debugging.
                        logger.warning(
                            "Cannot distinguish CWLS from lasio convention for "
                            "well field '%s' with pre-colon value '%s' "
                            "and post-colon description '%s'. "
                            "Defaulting to lasio (swapped) interpretation. "
                            "If data appears wrong, set well_format='cwls' to "
                            "force CWLS convention.",
                            mnemonic,
                            value,
                            description,
                        )
                        actual_value = description
                        self.las_file.well.descriptions[mnemonic] = value
        else:
            # LAS 2.0+: MNEM.UNIT VALUE : DESCRIPTION (unambiguous — no
            # CWLS/lasio swap needed).  Also handles LAS 1.2 entries
            # without a description field (no colon in the line).
            actual_value = value
            # F-I2-M07: Use `is not None` to preserve empty-string ""
            # descriptions (falsy but intentional), matching the LAS 1.2
            # branch at line 963 above.
            if description is not None:
                self.las_file.well.descriptions[mnemonic] = description

        # F-I2-M10: Detect duplicate well entry mnemonics.  Curves have
        # _deduplicate_curves() (60+ lines); parameters use list.append().
        # Well entries use dict assignment with zero duplicate detection —
        # duplicate mnemonic silently overwrites prior value.
        if mnemonic in self.las_file.well.entries:
            warnings.warn(
                f"Duplicate well entry mnemonic '{mnemonic}' encountered. "
                f"Previous value '{self.las_file.well.entries[mnemonic]}' "
                f"will be overwritten by '{actual_value}'.",
                UserWarning,
                stacklevel=2,
            )
        # F-022: Unescape colon artifacts inserted by the writer's
        # _escape_colons_for_las_value (e.g., " _:_ " → " : ").
        actual_value = _desanitize_las_value(_unescape_colons_for_las_value(actual_value))
        self.las_file.well[mnemonic] = actual_value
        # F-008: Use `is not None` instead of truthiness check.  An empty
        # string is a semantically meaningful unit (e.g., unitless well
        # fields) and should not be silently dropped.
        if unit is not None:
            self.las_file.well.units[mnemonic] = unit

    def _replay_deferred_well(self) -> None:
        """Re-process well entries (and data lines) that were parsed before ~V was known.

        When ~W appears before ~V, entries are buffered without being stored.
        Once ~V is parsed, all deferred entries are re-processed with the
        correct version-based swap logic (LAS 1.2 vs 2.0+).

        F-M12: When data sections appear before ~V, data lines are buffered
        and replayed here after the version is known.
        """
        if self._deferred_well_entries:
            # F-M-006: Combined-count guard before replay.  well.entries
            # may already contain entries + _deferred_well_entries may
            # independently have up to MAX_DEFERRED_WELL_ENTRIES (100K),
            # creating a transient memory pool of up to ~200K dict entries.
            combined = len(self.las_file.well.entries) + len(self._deferred_well_entries)
            if combined > 2 * MAX_DEFERRED_WELL_ENTRIES:
                raise LASParseError(
                    f"Combined well entry count ({combined}) exceeds maximum "
                    f"allowed ({2 * MAX_DEFERRED_WELL_ENTRIES}). "
                    f"The file may be malformed or corrupt."
                )
            is_las12 = _LASVersionSpec(
                self.las_file.version.vers
            ).is_las12
            for entry in self._deferred_well_entries:
                self._store_well_entry(
                    mnemonic=entry["mnemonic"],
                    unit=entry["unit"],
                    value=entry["value"],
                    description=entry["description"],
                    is_las12=is_las12,
                )
            self._deferred_well_entries.clear()

        # F-M12: Replay deferred data lines if the file is LAS 3.0.
        # F-H01 / I2-D2-01: Each deferred entry is a (section_type,
        # section_name, section_idx, line, curve_start, curve_end)
        # tuple.  Group by (section_type, section_name, section_idx)
        # to create one DataSection per original pre-~V data section.
        # curve_start/curve_end preserve pipe-target scoping through
        # the deferred replay so that |CURVE and |X_Definition pipe
        # associations are not lost.
        if self._deferred_ascii_data_lines:
            if self.las_file.version.is_las30:
                # Build groups from per-line tuple storage.
                # itertools.groupby groups consecutive lines with the same
                # key — correct since deferred lines are appended in file
                # order and section boundaries are naturally contiguous.
                groups: list[
                    tuple[tuple[str, str, int], list[str], int, int | None]
                ] = []
                for key, group_iter in groupby(
                    self._deferred_ascii_data_lines,
                    key=lambda t: (t[0], t[1], t[2]),
                ):
                    rows = list(group_iter)
                    groups.append((
                        key,
                        [r[3] for r in rows],
                        rows[0][4],       # curve_start_idx
                        rows[0][5],       # curve_end_idx
                    ))

                # Save current state before processing deferred groups.
                saved_lines = self._ascii_data_lines
                saved_curve_start = self._section_curve_start_idx
                saved_curve_end = self._section_curve_end_idx
                saved_section_type = self._current_data_section_type
                saved_section_name = self._current_section_name

                try:
                    # Process each deferred group as its own DataSection.
                    for (section_type, _section_name, _section_idx), raw_lines, curve_start, curve_end in groups:
                        # I2-D2-01: Restore pipe-target curve scoping stored
                        # at defer time, preserving |CURVE and
                        # |X_Definition associations.
                        self._section_curve_start_idx = curve_start
                        self._section_curve_end_idx = curve_end
                        # Pre-~V section type may be stale — use stored value.
                        self._current_data_section_type = section_type or "LOG_DATA"
                        # Preserve user-provided section names when available.
                        # Bare section keywords (e.g., "A" from ~A,
                        # "Core[1]" from ~Core[1]) are blanked so
                        # auto-generation produces unique Section_N names.
                        # Real user-provided names (e.g., "Main Log" from
                        # "~A Main Log") are preserved across replay.
                        _is_bare_keyword = (
                            _section_name in _DATA_SECTION_WORDS
                            or _is_indexed_data_section(_section_name)
                        )
                        if _section_name and not _is_bare_keyword:
                            self._current_section_name = _section_name
                        else:
                            self._current_section_name = ""
                        self._ascii_data_lines = raw_lines
                        ctx = AsciiDataContext(
                            las_file=self.las_file,
                            ascii_data_lines=self._ascii_data_lines,
                            section_curve_start_idx=self._section_curve_start_idx,
                            section_curve_end_idx=self._section_curve_end_idx,
                            current_section_name=self._current_section_name,
                            current_data_section_type=self._current_data_section_type,
                            current_data_section_idx=self._current_data_section_idx,
                            cumulative_elements=self._cumulative_elements,
                        )
                        process_ascii_data(ctx)
                        self._cumulative_elements = ctx.cumulative_elements
                        self._current_data_section_idx += 1
                finally:
                    # Restore state.
                    if saved_lines:
                        self._ascii_data_lines = saved_lines
                    else:
                        self._ascii_data_lines = []
                    self._section_curve_start_idx = saved_curve_start
                    self._section_curve_end_idx = saved_curve_end
                    self._current_data_section_type = saved_section_type
                    # F-I2-M01: Reset section name (defense-in-depth).
                    self._current_section_name = saved_section_name or ""
            self._deferred_ascii_data_lines.clear()

    def _parse_well(self, line: str) -> None:
        """Parse ~W (well information) section line.

        LAS 1.2 uses format ``MNEM.UNIT DESCRIPTION :VALUE``.
        LAS 2.0+ uses ``MNEM.UNIT VALUE :DESCRIPTION``.
        This method applies version-based dispatch to swap the
        value/description fields for LAS 1.2.

        The unit field (e.g. ``'.M'`` from ``'STRT.M'``) is preserved
        in ``self.las_file.well.units`` for roundtrip fidelity.
        """
        match = self._match_data_line(line)
        if not match:
            logger.warning(
                "Non-matching ~W line in %s: %s",
                self.source_file or "<unknown>",
                line.strip()[:120],
            )
            return

        # F-I2-M14: Apply mnemonic base normalization to well entries,
        # matching curves (line 1281) and parameters (line 1391).
        # Without this, if mnem_base maps "COMP" → "COMPANY",
        # curves get COMPANY.GAPI but well gets COMP — a contract
        # violation (mnem_base documents resolution for all sections).
        raw_mnemonic = match.group("mnemonic").upper().strip()
        mnemonic = self._mnem_base_upper.get(raw_mnemonic, raw_mnemonic)
        unit = match.group("unit") or ""
        value = match.group("value").strip()
        # F-07: Distinguish "no colon in line" (VALUE_ONLY_PATTERN) from
        # "colon present but post-colon text is empty" (DATA_LINE_PATTERN
        # with empty description group).  Both previously produced `""`,
        # making the `if is_las12 and description:` gate at line 923
        # skip LAS 1.2 swap logic for colon-bearing empty-description
        # lines, incorrectly storing pre-colon text as the value.
        if "description" in match.groupdict():
            description = match.group("description").strip() or ""
        else:
            description = None  # VALUE_ONLY_PATTERN — no colon in line

        # F-H03: Bare-colon CWLS/lasio lines (e.g., "DATE. LOG DATE:15/01/2001")
        # where the colon lacks surrounding whitespace fall through to
        # VALUE_ONLY_PATTERN with description=None.  The bare-colon detection
        # has been moved into _store_well_entry (see F-P06 / R7F-05 fix) so
        # that deferred well entries (parsed before ~V is known) always use
        # the correct `is_las12` flag.  _store_well_entry splits bare-colon
        # values when is_las12=True and description is None, with timestamp/
        # datetime guards to avoid corrupting "12:34:56" or ISO datetimes.
        is_las12 = _LASVersionSpec(
            self.las_file.version.vers
        ).is_las12

        # F-P06: When ~W appears before ~V, the version defaults to "2.0" and
        # is_las12 is False, skipping the LAS 1.2 convention swap.  Buffer raw
        # entries so they can be re-processed with the correct version after
        # ~V is parsed.
        if not self._version_found:
            # F-35: Guard against unbounded deferred-well-entry accumulation.
            # Every other accumulator in parser.py has a MAX_* guard; this was
            # the sole unguarded buffer.  Malicious ~W-before-~V files without
            # a ~V section could grow this list without bound.
            if len(self._deferred_well_entries) >= MAX_DEFERRED_WELL_ENTRIES:
                raise LASParseError(
                    f"Deferred well entry count ({len(self._deferred_well_entries) + 1}) "
                    f"exceeds maximum allowed ({MAX_DEFERRED_WELL_ENTRIES}). "
                    f"The file may be malformed or corrupt."
                )
            self._deferred_well_entries.append({
                "mnemonic": mnemonic,
                "unit": unit,
                "value": value,
                "description": description,  # type: ignore[dict-item]
            })
            if len(self._deferred_well_entries) == 1:
                warnings.warn(
                    "~W (Well) section encountered before ~V (Version) "
                    "section. Well data interpretation may be incorrect "
                    "for LAS 1.2 files. Entries will be re-evaluated "
                    "once the version is known.",
                    UserWarning,
                    stacklevel=2,
                )
            # F-003: Deferred entries are stored only via _replay_deferred_well
            # after ~V is known.  Do NOT call _store_well_entry here — the prior
            # fix (commit b47eea6) added the buffer but left the unconditional
            # store below, causing entries to be stored twice (once with wrong
            # is_las12, once with correct in replay).
            return

        self._store_well_entry(mnemonic, unit, value, description, is_las12)

    def _parse_curve(self, line: str) -> None:
        """Parse ~C (curve information) section line.

        Supports LAS 3.0 features:
        - Array notation: NMR[1], NMR[2], etc.
        - Format specifiers: {F}, {E}, {S}, {A:0}
        """
        match = self._match_data_line(line)
        if not match:
            logger.warning(
                "Non-matching ~C line in %s: %s",
                self.source_file or "<unknown>",
                line.strip()[:120],
            )
            return

        # F-01: Preserve original casing before uppercasing.
        # raw_mnemonic is used for mnem_base lookup (case-insensitive);
        # _original_cased stores the pre-uppercased value so
        # CurveDefinition.original_mnemonic reflects the file's casing.
        _original_cased = match.group("mnemonic").strip()
        raw_mnemonic = _original_cased.upper()
        unit = match.group("unit") or ""
        api_code = match.group("value").strip() if match.group("value") else ""
        description = (
            match.group("description").strip()
            if "description" in match.groupdict() and match.group("description")
            else ""
        )

        # F-022: Unescape colon artifacts inserted by the writer's
        # _escape_colons_for_las_value.  Curve descriptions and API codes
        # may contain escaped colons from a prior write→read roundtrip.
        api_code = _desanitize_las_value(_unescape_colons_for_las_value(api_code))
        description = _desanitize_las_value(_unescape_colons_for_las_value(description))

        # LAS 3.0: Extract format specifier from description
        data_format = ""
        array_time_offset: float | None = None
        # F-M18: Use findall() to capture ALL format specifiers.  The old
        # search() only found the first, but sub() removed them all —
        # creating an asymmetry where extra format specifiers were silently
        # discarded without warning.
        format_matches = FORMAT_SPEC_PATTERN.findall(description)
        if format_matches:
            first_fmt, first_offset = format_matches[0]
            # F2-001: Normalize to uppercase so all downstream case-sensitive
            # comparisons (string_curves at L1485, _KNOWN_CURVE_FORMATS at L1498,
            # and array-time-offset check at L1172) work regardless of input case.
            data_format = first_fmt.upper()
            # Normalize extended Fortran-style format specifiers (e.g., F8.3,
            # E10.2, E0.00E+00, D0.00E+00) to single-letter codes (F, E, D)
            # for roundtrip compatibility.  from_dict's _VALID_DATA_FORMATS
            # only accepts single-letter codes; storing the extended form
            # would cause parse→to_dict→from_dict ValueError (F-H01).
            # F-REV-01: Validate the full format string BEFORE truncation
            # so that non-format brace text (e.g. {Density}) is caught
            # by _FORMAT_SPEC_RE rather than silently normalizing to a
            # valid single-letter code (DENSITY → D).
            # I2-XWP-02: Multi-character patterns that do NOT match
            # _FORMAT_SPEC_RE (e.g., {DD/MM/YYYY}, {DEG}) are metadata
            # templates and pass through unchanged — same behavior
            # as the parameter path (line ~2077).
            try:
                _validate_curve_data_format(data_format, raw_mnemonic)
            except LASParseError:
                # Not a valid data format — clear it so CurveDefinition
                # __post_init__ does not reject the multi-char string.
                # The non-format text is preserved in the description
                # via _keep_non_format below.
                warnings.warn(
                    f"Invalid data format '{data_format}' in curve "
                    f"'{raw_mnemonic}' — clearing to empty string",
                    UserWarning,
                    stacklevel=2,
                )
                data_format = ""
            if len(data_format) > 1:
                data_format = data_format[0]
            if data_format == "A" and first_offset:
                try:
                    array_time_offset = float(first_offset)
                except ValueError as exc:
                    raise LASParseError(
                        f"Invalid format specifier offset: "
                        f"'{first_offset}' is not a valid number "
                        f"in curve description '{description}'"
                    ) from exc
                if not np.isfinite(array_time_offset):
                    raise LASParseError(
                        f"Format specifier offset overflow: "
                        f"'{first_offset}' produced "
                        f"{array_time_offset} in curve description '{description}'"
                    )
            if len(format_matches) > 1:
                extra_formats = [f[0] for f in format_matches[1:]]
                logger.warning(
                    "Multiple format specifiers found in curve '%s' "
                    "description: %s. Only the first (%s) is used; "
                    "extra specifiers %s are discarded.",
                    raw_mnemonic,
                    [f[0] for f in format_matches],
                    data_format,
                    extra_formats,
                )
            # F-REV-01: Only strip validated format specifiers from
            # description.  FORMAT_SPEC_PATTERN.sub("") blindly strips
            # ALL brace-enclosed text — including non-format metadata
            # like {Density}, {GAPI}, {Well}.  Use a callback-based sub
            # so each match is individually validated before stripping.
            def _keep_non_format(m: re.Match[str]) -> str:
                try:
                    _validate_curve_data_format(
                        m.group("format").upper(), raw_mnemonic
                    )
                    return ""  # Valid format specifier → strip it
                except LASParseError:
                    return m.group(0)  # Non-format text → keep it

            description = FORMAT_SPEC_PATTERN.sub(
                _keep_non_format, description
            ).strip()

        # LAS 3.0: Check for array notation in mnemonic
        array_info: ArrayElementInfo | None = None
        array_match = ARRAY_MNEMONIC_PATTERN.match(raw_mnemonic)
        if array_match:
            base_name = array_match.group("base").upper()
            try:
                index = int(array_match.group("index"))
            except ValueError as exc:
                raise LASParseError(
                    f"Invalid array index '{array_match.group('index')}' in "
                    f"curve mnemonic '{raw_mnemonic}'"
                ) from exc
            array_info = ArrayElementInfo(
                base_name=base_name,
                index=index,
                time_offset=array_time_offset,
            )
        elif "[" in raw_mnemonic:
            # F-M-007: Warn when mnemonic contains "[" but doesn't match
            # ARRAY_MNEMONIC_PATTERN (e.g., NMR[-1], NMR[abc], NMR[]).
            logger.warning(
                "Mnemonic %r contains '[' but does not match array notation "
                "pattern; treated as standalone curve.",
                raw_mnemonic,
            )

        # Apply mnemonic normalization from mnem_base
        normalized = self._mnem_base_upper.get(raw_mnemonic, raw_mnemonic)

        # F-M-026: Wrap CurveDefinition construction to catch ValueError
        # from __post_init__ validation (e.g., empty mnemonic after
        # mnem_base normalization) and re-raise as LASParseError.
        try:
            curve = CurveDefinition(
                mnemonic=normalized,
                unit=unit,
                api_code=api_code,
                description=description,
                original_mnemonic=_original_cased if _original_cased.upper() != normalized else "",
                data_format=data_format,
                array_info=array_info,
            )
        except ValueError as e:
            raise LASParseError(
                f"Invalid curve definition for mnemonic {raw_mnemonic!r}: {e}"
            ) from e
        # F-28: Guard against unbounded curve accumulation during ~C parsing.
        # Without this check, a metadata-only LAS 3.0 file can accumulate
        # unlimited CurveDefinition objects without triggering any bounds
        # check (_data_reader.MAX_CURVES was only checked later in _process_ascii_data,
        # which early-returns when no data section exists).
        if len(self.las_file.curves) >= _data_reader.MAX_CURVES:
            raise LASParseError(
                f"Curve count ({len(self.las_file.curves) + 1}) exceeds maximum "
                f"allowed ({_data_reader.MAX_CURVES}). The file may be malformed or corrupt."
            )
        self.las_file.curves.append(curve)
        self.las_file.curves_order.append(normalized)

    def _parse_parameter(self, line: str) -> None:
        """Parse ~P (parameter) section line.

        Supports LAS 3.0 features:
        - Array notation: RUN[1], RUN[2], etc.
        - Zone association via pipe: | Run[1], | Zone[2]
        """
        match = self._match_data_line(line)
        if not match:
            logger.warning(
                "Non-matching ~P line in %s: %s",
                self.source_file or "<unknown>",
                line.strip()[:120],
            )
            return

        # F-01: Preserve original casing (same pattern as _parse_curve).
        _original_cased = match.group("mnemonic").strip()
        raw_mnemonic = _original_cased.upper()
        unit = match.group("unit") or ""
        # F-116: Unescape colon artifacts in parameter units.
        # The writer escapes colons (writer.py:498-499) but the parser
        # did not unescape — the sole one-way path across 7 colon-escaped
        # fields.  All 6 other fields (value, description across well,
        # curve, parameter) have paired unescape.  Adding this closes the
        # seventh and final gap, ensuring "kg : m" survives write→re-parse.
        unit = _desanitize_las_value(_unescape_colons_for_las_value(unit))
        value = match.group("value").strip()
        description = (
            match.group("description").strip()
            if "description" in match.groupdict() and match.group("description")
            else ""
        )

        # F-022: Unescape colon artifacts inserted by the writer's
        # _escape_colons_for_las_value BEFORE format-specifier extraction.
        # Escaped patterns like " _:_ " in parameter values and descriptions
        # should be restored to their original colon-separated form.
        value = _desanitize_las_value(_unescape_colons_for_las_value(value))
        description = _desanitize_las_value(_unescape_colons_for_las_value(description))

        # M-PB2: Strip LAS 3.0 format specifiers from parameter
        # descriptions, mirroring _parse_curve logic (lines 877-899).
        # F-M15: Extract format specifiers BEFORE stripping and store
        # them for roundtrip fidelity.  CurveDefinition has data_format;
        # ParameterEntry does not yet, so store as an instance attribute
        # (forward-compatible — assignment works before and after the
        # field is added to the dataclass).
        param_data_format = ""
        param_format_matches = FORMAT_SPEC_PATTERN.findall(description)
        if param_format_matches:
            param_data_format = param_format_matches[0][0].upper()
            if len(param_format_matches) > 1:
                extra_formats = [f[0] for f in param_format_matches[1:]]
                logger.warning(
                    "Multiple format specifiers found in parameter '%s' "
                    "description: %s. Only the first (%s) is used; "
                    "extra specifiers %s are discarded.",
                    raw_mnemonic,
                    [f[0] for f in param_format_matches],
                    param_data_format,
                    extra_formats,
                )
            # F-101: Validate single-character parameter data_format codes at
            # parse time — mirroring from_dict's check at models.py:107.
            if param_data_format:
                # I2-XWP-02: Truncate extended Fortran-style format
                # specifiers (F8.3, E10.2, D0.00E+00) to single-letter
                # codes for roundtrip compatibility with from_dict's
                # _VALID_DATA_FORMATS.  This matches the curve path's
                # behavior (line ~1861).  Multi-character patterns that
                # do NOT match _FORMAT_SPEC_RE (e.g., {DD/MM/YYYY},
                # {DEG}) are metadata templates and pass through
                # unchanged — same as the existing documented behavior.
                if len(param_data_format) > 1:
                    try:
                        _validate_curve_data_format(param_data_format, raw_mnemonic)
                    except LASParseError:
                        warnings.warn(
                            f"Parameter '{raw_mnemonic}' has unrecognized data "
                            f"format specifier '{{{param_data_format}}}'. "
                            f"Valid LAS data format codes are single-letter "
                            f"F, E, D, S, A or Fortran-style F8.3/E10.2. "
                            f"Clearing to empty string.",
                            UserWarning,
                            stacklevel=2,
                        )
                        param_data_format = ""  # Not a valid data format — clear to prevent accumulation
                    else:
                        param_data_format = param_data_format[0]
                # Single-char non-FEDAS codes ({X}, {G}) are warned and
                # cleared (matching curve path at line 2003-2016) to
                # avoid crashing the parse — clearing to "" preserves
                # roundtrip compatibility.
                if len(param_data_format) == 1 and param_data_format not in _VALID_DATA_FORMATS:
                    warnings.warn(
                        f"Parameter '{raw_mnemonic}' has unsupported data "
                        f"format specifier '{{{param_data_format}}}'. "
                        f"Valid LAS data format codes: "
                        f"{', '.join(sorted(_VALID_DATA_FORMATS))}. "
                        f"Clearing to empty string.",
                        UserWarning,
                        stacklevel=2,
                    )
                    param_data_format = ""
        # F-REV-01: Only strip validated format specifiers from
        # description — matching the curve path.  FORMAT_SPEC_PATTERN.sub("")
        # blindly strips ALL brace-enclosed text including non-format
        # metadata like {Density}, {GAPI}, {Note}.  Use a callback-based
        # sub so each match is individually validated.
        def _keep_non_format_param(m: re.Match[str]) -> str:
            try:
                _validate_curve_data_format(
                    m.group("format").upper(), raw_mnemonic
                )
                return ""  # Valid format specifier → strip it
            except LASParseError:
                return m.group(0)  # Non-format text → keep it

        description = FORMAT_SPEC_PATTERN.sub(
            _keep_non_format_param, description
        ).strip()

        # LAS 3.0: Check for zone association in description
        zone: ParameterZone | None = None
        zone_match = ZONE_ASSOC_PATTERN.search(description)
        if zone_match:
            zone_index: int | None = None
            if zone_match.group("index"):
                try:
                    zone_index = int(zone_match.group("index"))
                except ValueError as exc:
                    raise LASParseError(
                        f"Invalid zone index '{zone_match.group('index')}' in "
                        f"parameter '{raw_mnemonic}'"
                    ) from exc
            # F-01: Preserve original zone name casing
            _orig_zone = zone_match.group("zone")
            zone = ParameterZone(
                zone_name=_orig_zone.upper(),
                zone_index=zone_index,
            )
            # Remove zone association from description
            description = ZONE_ASSOC_PATTERN.sub("", description).strip()

        # LAS 3.0: Check for array notation in mnemonic
        array_index: int | None = None
        array_match = ARRAY_MNEMONIC_PATTERN.match(raw_mnemonic)
        if array_match:
            try:
                array_index = int(array_match.group("index"))
            except ValueError as exc:
                raise LASParseError(
                    f"Invalid array index '{array_match.group('index')}' in "
                    f"parameter mnemonic '{raw_mnemonic}'"
                ) from exc
        elif "[" in raw_mnemonic:
            # F-M-007: Warn when mnemonic contains "[" but doesn't match
            # ARRAY_MNEMONIC_PATTERN (e.g., NMR[-1], NMR[abc], NMR[]).
            logger.warning(
                "Mnemonic %r contains '[' but does not match array notation "
                "pattern; treated as standalone parameter.",
                raw_mnemonic,
            )

        # Apply mnemonic normalization from mnem_base (same as curve handling)
        normalized = self._mnem_base_upper.get(raw_mnemonic, raw_mnemonic)

        # F-053: Determine section_type for per-section parameter grouping.
        # The section dispatcher sets _current_section_name from the section
        # header: standard ~P/~Parameter → "P" or "PARAMETER"; typed
        # sections (e.g., ~Core_Parameter) → "CORE_PARAMETER".
        # Derive section_type by stripping the _PARAMETER/_PARAMETERS suffix.
        _section_type: str | None = None
        _sect_name = (self._current_section_name or "").upper()
        if _sect_name.endswith("_PARAMETER"):
            _section_type = _sect_name[: -len("_PARAMETER")]
        elif _sect_name.endswith("_PARAMETERS"):
            _section_type = _sect_name[: -len("_PARAMETERS")]
        # I2-XPM-01: Sanitize section_type — tildes can appear in
        # section_type when a section header like ~C~ORE_PARAMETER is
        # parsed (SECTION_PATTERN captures C~ORE_PARAMETER as the word).
        # Tildes are the LAS section marker and must not leak into model
        # metadata (ParameterEntry.section_type, writer section_type).
        if _section_type is not None:
            _section_type = _section_type.replace("~", "")

        # F-M-026: Wrap ParameterEntry construction to catch ValueError
        # from __post_init__ validation (e.g., empty mnemonic after
        # mnem_base normalization) and re-raise as LASParseError.
        try:
            param = ParameterEntry(
                mnemonic=normalized,
                unit=unit,
                value=value,
                description=description,
                data_format=param_data_format,
                array_index=array_index,
                zone=zone,
                section_type=_section_type,
            )
        except ValueError as e:
            raise LASParseError(
                f"Invalid parameter entry for mnemonic {raw_mnemonic!r}: {e}"
            ) from e
        # F-M15: Store data_format as an instance attribute for roundtrip
        # fidelity.  Forward-compatible — works before and after the field
        # is added to the ParameterEntry dataclass.
        if param_data_format:
            param.data_format = param_data_format
        # F-29: Guard against unbounded parameter accumulation.
        # Curves have _data_reader.MAX_CURVES checked in 3 locations; parameters had zero
        # protection anywhere despite following the same append pattern.
        if len(self.las_file.parameters) >= MAX_PARAMETERS:
            raise LASParseError(
                f"Parameter count ({len(self.las_file.parameters) + 1}) exceeds "
                f"maximum allowed ({MAX_PARAMETERS}). "
                f"The file may be malformed or corrupt."
            )
        self.las_file.parameters.append(param)

    def _parse_other(self, line: str) -> None:
        """Parse ~O (other) section — free-form text, accumulated."""
        self._append_other_line(line)

    def _parse_ascii_data(self, line: str) -> None:
        """Collect ASCII data lines for later processing.

        In LAS 3.0, data can be delimited by SPACE, TAB, or COMMA.
        Data is collected and processed after all lines are parsed.

        For LAS 1.2/2.0, ASCII data is handled by data_reader, so no
        collection is needed here.
        """
        if not self.las_file.version.is_las30:
            # F-M12: When ~V hasn't been parsed yet, we can't know whether
            # is_las30 is True.  Buffer data lines so they aren't silently
            # discarded.  After ~V is parsed, _replay_deferred_well replays
            # them if the file is LAS 3.0; if not, they are discarded.
            if not self._version_found:
                if len(self._deferred_ascii_data_lines) > _data_reader.MAX_DATA_LINES:
                    raise LASParseError(
                        f"Deferred ASCII data line count exceeds maximum "
                        f"allowed ({_data_reader.MAX_DATA_LINES}). "
                        f"The file may be malformed or corrupt."
                    )
                # F-H01: Store per-line (section_type, section_name,
                # section_idx, curve_start, curve_end, line) so
                # _replay_deferred_well can reconstruct per-section
                # grouping.  section_idx disambiguates consecutive bare
                # sections.  curve_start/curve_end preserve pipe-target
                # scoping across the deferred replay (I2-D2-01).
                self._deferred_ascii_data_lines.append(
                    (
                        self._current_data_section_type,
                        self._current_section_name,
                        self._current_data_section_idx,
                        line,
                        self._section_curve_start_idx,
                        self._section_curve_end_idx,
                    )
                )
            return
        # F-27: Early bounds check during accumulation — reject before the
        # list grows unbounded.  The main check in _process_ascii_data runs
        # AFTER all lines are collected, offering no protection during the
        # accumulation phase itself.
        if len(self._ascii_data_lines) > _data_reader.MAX_DATA_LINES:
            raise LASParseError(
                f"ASCII data line count exceeds maximum allowed "
                f"({_data_reader.MAX_DATA_LINES}) during accumulation. "
                f"The file may be malformed or corrupt."
            )
        self._ascii_data_lines.append(line)
        # F-09: Cumulative cross-section data line counter — defense-in-depth
        # against multi-section files where each section passes the per-section
        # MAX_DATA_LINES bound individually.
        self._cumulative_data_lines += 1
        if (
            self._cumulative_data_lines > _data_reader.MAX_DATA_LINES * 10
            and not self._cumulative_data_lines_warned
        ):
            self._cumulative_data_lines_warned = True
            warnings.warn(
                f"Cumulative data line count ({self._cumulative_data_lines}) "
                f"across {self._current_data_section_idx + 1} sections "
                f"is unusually high.  The file may be malformed or corrupt.",
                UserWarning,
                stacklevel=2,
            )

    def _validate_cross_section_consistency(self) -> None:
        """Validate cross-section consistency (F-34).

        Three dimensions checked:
        (1) Curve count vs data column count for each data section.
        (2) LAS 3.0 section ordering — data sections before curve
            definitions have no curves to reference.
        (3) Duplicate section headers.
        """
        # (1) Curve count vs data column count per data section.
        for ds in self.las_file.data_sections:
            declared = len(ds.section_curves)
            actual_cols = len(ds.data) + len(ds.string_data)
            if actual_cols != declared and declared > 0:
                logger.warning(
                    "Data section '%s': section has %d data columns "
                    "but %d curves declared. Data count mismatch may "
                    "indicate corrupt or misaligned data.",
                    ds.name,
                    actual_cols,
                    declared,
                )

        # (2) LAS 3.0 section ordering — data sections before curves.
        # F-013: Per-type ordering check.  The previous code used a single
        # boolean flag (curve_seen) that only caught the case where ANY
        # data section appeared before ANY curve definition.  It did not
        # catch per-type ordering violations where, e.g., ~Core_Data
        # appears before ~Core_Definition but ~Drilling_Definition
        # appeared earlier covering a different type.  The fix tracks
        # which definition types have been seen using _definition_curve_ranges
        # and warns on the FIRST occurrence of a data section whose
        # corresponding definition type hasn't been seen.
        if self.las_file.version.is_las30:
            # Track which definition types have been encountered.
            # Keys: definition type names (e.g., "CORE_DEFINITION",
            # "DRILLING_DEFINITION") or "__MAIN__" for global curves.
            _defs_seen: set[str] = set()
            # M-08: Track definition types that have corresponding data
            # sections — used for forward validation (Definition→Data).
            _data_types_seen: set[str] = set()
            per_type_data_before_def: list[str] = []

            # F-004: Track parameter sections per type group, parallel to
            # _defs_seen.  LAS 3.0 requires Parameter→Definition→Data
            # order.  Warn when a Parameter section follows its Definition
            # (out-of-order group).
            per_type_param_after_def: list[str] = []

            for label in self._section_sequence:
                section_word = label.split(":")[0]
                is_data = (
                    section_word in _DATA_SECTION_WORDS
                    or section_word.endswith("_DATA")
                    or _is_indexed_data_section(section_word)
                )
                is_curve = (
                    section_word in {"C", "CURVE"}
                    or section_word.endswith("_DEFINITION")
                )
                # F-004: Identify parameter sections (both LAS 1.2/2.0 ~P
                # and LAS 3.0 type-prefixed ~Core_Parameter etc.)
                is_param = (
                    section_word == "P"
                    or section_word.endswith("_PARAMETER")
                    or section_word.endswith("_PARAMETERS")
                )

                if is_data:
                    # Normalize indexed section words (e.g., "CORE[1]" → "CORE")
                    # before resolving definition type.  _SECTION_TYPE_MAP only
                    # contains unindexed keys, and the endswith("_DATA") check
                    # cannot match bracketed forms.
                    _type_word = section_word
                    if _is_indexed_data_section(section_word):
                        _type_word = section_word[: section_word.find("[")].upper()

                    # Determine expected definition type for this data section.
                    if _type_word.endswith("_DATA"):
                        _def_type = _type_word.replace("_DATA", "_DEFINITION")
                    elif _type_word in _SECTION_TYPE_MAP:
                        _canonical = _SECTION_TYPE_MAP[_type_word]
                        if _canonical.endswith("_DATA"):
                            _def_type = _canonical.replace("_DATA", "_DEFINITION")
                        else:
                            _def_type = "__MAIN__"
                    else:
                        _def_type = "__MAIN__"

                    if _def_type not in _defs_seen:
                        _def_display = (
                            f"~{_def_type}"
                            if _def_type != "__MAIN__"
                            else "the main curve definition (~C or ~CURVE)"
                        )
                        per_type_data_before_def.append(
                            f"~{section_word} before {_def_display}"
                        )

                    # M-08: Track which definition types have data sections
                    # for forward validation (Definition→Data).
                    _data_types_seen.add(_def_type)

                if is_curve:
                    # Mark this definition type as seen.
                    if section_word.endswith("_DEFINITION"):
                        _defs_seen.add(section_word)
                    else:
                        # ~C or ~CURVE — covers all global definitions.
                        _defs_seen.add("__MAIN__")

                # F-004: Track parameter sections and detect Parameter-after-
                # Definition ordering.  When a parameter section appears in
                # the sequence AFTER its corresponding definition section,
                # the LAS 3.0 Parameter→Definition→Data order is violated.
                if is_param:
                    # Derive the expected definition type for this parameter
                    # section (e.g., CORE_PARAMETER → CORE_DEFINITION).
                    if section_word.endswith("_PARAMETERS"):
                        _def_type = section_word.replace("_PARAMETERS", "_DEFINITION")
                    elif section_word.endswith("_PARAMETER"):
                        _def_type = section_word.replace("_PARAMETER", "_DEFINITION")
                    else:
                        # ~P — global parameter section, covers all
                        _def_type = "__MAIN__"
                    if _def_type in _defs_seen:
                        per_type_param_after_def.append(
                            f"~{section_word} after ~{_def_type}"
                        )

            # Emit per-type data-before-definition warnings.
            for msg in per_type_data_before_def:
                logger.warning(
                    "LAS 3.0 data section %s. "
                    "Data sections without preceding curve definitions "
                    "will have no curves to reference and may produce "
                    "empty or truncated output.",
                    msg,
                )

            # F-004: Emit parameter-after-definition warnings.
            for msg in per_type_param_after_def:
                logger.warning(
                    "LAS 3.0 parameter section %s. "
                    "Parameter sections should precede their definition "
                    "sections per the LAS 3.0 specification. "
                    "Definitions seen before parameters may cause "
                    "metadata loss.",
                    msg,
                )

            # M-08: Forward check — definition types without corresponding
            # data sections.  The reverse check (above) catches Data→Definition
            # gaps; this catches Definition→Data gaps where curves are
            # declared but never populated.
            for _def_type in _defs_seen:
                if _def_type not in _data_types_seen:
                    _def_display = (
                        f"~{_def_type}"
                        if _def_type != "__MAIN__"
                        else "the main curve definition (~C or ~CURVE)"
                    )
                    logger.warning(
                        "LAS 3.0 curve definition %s has no corresponding "
                        "data section. Curves defined without data sections "
                        "will produce empty output.",
                        _def_display,
                    )

        # (3) Duplicate section headers — detect by semantic section TYPE,
        # not by label string.  Label-based counting (prior implementation)
        # missed semantically-duplicate sections that differ only by
        # section_name: ~V (label "V:") and ~VERSION (label "VERSION:")
        # both dispatch to _parse_version but produced different labels.
        # Similarly, ~V and ~V INFORMATION both route to "V" but produce
        # different labels ("V:" vs "V:INFORMATION").
        #
        # Single-occurrence reserved sections:
        #   V (VERSION), W (WELL) — always single occurrence per spec.
        #   O (OTHER) — always single occurrence.
        #   C (CURVE), P (PARAMETER), A (ASCII) — only in non-LAS 3.0.
        #   In LAS 3.0, multiple ~C (per Definition), ~P (per typed group),
        #   and ~A (per data section) are valid.
        _is_las30 = self.las_file.version.is_las30
        _reserved_single: set[str] = {"V", "W", "O"}
        if not _is_las30:
            _reserved_single.update({"C", "P", "A"})

        _type_counts: dict[str, int] = {}
        for stype in self._section_type_sequence:
            _type_counts[stype] = _type_counts.get(stype, 0) + 1

        for stype, count in _type_counts.items():
            if count > 1 and stype in _reserved_single:
                _duplicates = [
                    lbl for lbl, st in zip(
                        self._section_sequence, self._section_type_sequence, strict=True
                    ) if st == stype
                ]
                _dup_labels = ", ".join(f"'~{d}'" for d in _duplicates)
                logger.warning(
                    "Duplicate reserved section type '~%s' encountered "
                    "%d times: %s. Repeated reserved sections may "
                    "indicate a malformed file or cause data from earlier "
                    "instances to be overwritten.",
                    stype,
                    count,
                    _dup_labels,
                )
