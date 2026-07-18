"""Regex-based LAS file parser replacing PLY.

The LAS format is line-based with a simple structure:
  MNEMONIC.UNIT  VALUE : DESCRIPTION

PLY (lex/yacc) is overkill for this. Regex reduces ~450 lines to ~150
while maintaining the same parsing capability.

Supports LAS 1.2, 2.0, and 3.0 formats.
"""

from __future__ import annotations

import csv
import logging
import re
import warnings
from itertools import groupby
from typing import ClassVar

import numpy as np

from .data_reader import (
    MAX_CURVES,
    MAX_DATA_LINES,
    MAX_TOKENS_PER_LINE,
    MAX_TOTAL_ELEMENTS,
    _get_null_value,
    _to_finite_float,
)
from .exceptions import LASParseError
from .mnem_base import resolve_mnemonic
from .models import (
    ArrayElementInfo,
    CurveDefinition,
    DataSection,
    LASFile,
    ParameterEntry,
    ParameterZone,
)

logger = logging.getLogger(__name__)

# F-26: Global aggregate limit on data sections to prevent multi-section DoS.
# Each section passes per-section bounds (MAX_DATA_LINES, MAX_CURVES,
# MAX_TOTAL_ELEMENTS) but no global cap existed — an attacker could craft N
# data sections cumulatively exhausting memory.  Overridable at module level.
MAX_DATA_SECTIONS = 1_000

# F-29: Maximum parameter entries per file.  Curves have MAX_CURVES (100K)
# checked in 3 locations; parameters had zero protection anywhere.
MAX_PARAMETERS = 100_000

# F-M-02: Maximum other-section lines.  All other accumulators have explicit
# MAX_* constants; _other_lines had no bound, enabling unbounded memory growth
# from malformed files.  Overridable at module level.
MAX_OTHER_LINES = 1_000_000

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
_SPLITLINES_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x85\u2028\u2029]")

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
# The colon separator uses (\s+:\s*|\s*:\s+) which requires whitespace on
# at least one side of the colon.  This prevents false matches on bare
# colons in values (timestamps like "12:34:56") and LAS 3.0 format
# specifiers ({A:0}), while still correctly separating value from
# description in standard "VALUE : DESCRIPTION" lines and handling
# empty-value lines like "MNEM.UNIT       : DESCRIPTION".
DATA_LINE_PATTERN = re.compile(
    r"^\s*"
    r"(?P<mnemonic>[\w\-]+(?:\[\d+\])?)"  # mnemonic: word chars + hyphen + optional [N] array index
    r"\s*"  # optional whitespace before dot (common in LAS files)
    r"\."  # literal dot separator
    r"(?P<unit>[\w\-/]*)"  # unit: optional, can include /
    r"\s+"  # whitespace separator
    r"(?P<value>.*?)"  # value: everything up to the colon separator
    r"(\s+:\s*|\s*:\s+|:\s*$)"  # colon separator (see detailed comment above)
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
FORMAT_SPEC_PATTERN = re.compile(r"\{(?P<format>[A-Za-z][^}:\s]*?)(?::(?P<offset>[\d.]*))?\s*\}")

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
    "INCLINOMETRY",  # Inclinometry data section
    "INCLINOMETRY_DATA",  # Inclinometry data section (written form)
    "TOPS",  # Tops data section
    "TOPS_DATA",  # Tops data section (written form)
    "TEST",  # Test data section
    "TEST_DATA",  # Test data section (written form)
    "PERFORATIONS",  # Perforations data section
    "PERFORATIONS_DATA",  # Perforations data section (written form)
    "LOG",  # LAS 3.0 shorthand ~Log alias for ~Log_Data / ~Ascii
    "LOG_DATA",  # Explicit log data section
}

# LAS 3.0 data section types that support index notation (e.g., ~Core[1]).
# Used to match bracketed sections like ~Inclinometry[1], ~Drilling[2], etc.
_INDEXED_DATA_TYPES = frozenset(
    {
        "CORE",
        "CORE_DATA",
        "DRILLING",
        "DRILLING_DATA",
        "INCLINOMETRY",
        "INCLINOMETRY_DATA",
        "TOPS",
        "TOPS_DATA",
        "TEST",
        "TEST_DATA",
        "PERFORATIONS",
        "PERFORATIONS_DATA",
        "LOG",
        "LOG_DATA",
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
    "INCLINOMETRY": "INCLINOMETRY_DATA",
    "INCLINOMETRY_DATA": "INCLINOMETRY_DATA",
    "TOPS": "TOPS_DATA",
    "TOPS_DATA": "TOPS_DATA",
    "TEST": "TEST_DATA",
    "TEST_DATA": "TEST_DATA",
    "PERFORATIONS": "PERFORATIONS_DATA",
    "PERFORATIONS_DATA": "PERFORATIONS_DATA",
    "LOG": "LOG_DATA",
    "LOG_DATA": "LOG_DATA",
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
        # F-H01: Each entry is a (section_type, section_name, section_idx,
        # raw_line) tuple to preserve per-section boundaries across the
        # deferred replay.  section_idx distinguishes consecutive sections
        # that happen to have the same section_type and section_name
        # (e.g., two bare ~A sections).
        self._deferred_ascii_data_lines: list[tuple[str, str, int, str]] = []
        # F-34: Track section headers encountered (in order) for cross-section
        # consistency validation: duplicate detection and LAS 3.0 ordering check.
        self._section_sequence: list[str] = []

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
            self._process_ascii_data()

        # Validate mandatory LAS 2.0 well fields (STRT, STOP, STEP, NULL).
        # LAS 2.0 requires these fields; missing fields are a spec compliance
        # gap.  The library handles missing fields gracefully (using defaults),
        # so this is a warning, not an error.
        is_las20 = self.las_file.version.vers.startswith("2.")
        if is_las20 and self._version_found:
            _mandatory_fields = ["STRT", "STOP", "STEP", "NULL"]
            for field in _mandatory_fields:
                if field not in self.las_file.well.entries:
                    warnings.warn(
                        f"LAS 2.0 file missing mandatory well field: {field}",
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
            stripped = line.strip()
            match = SECTION_PATTERN.match(stripped)
            if match:
                section_word = match.group(1).upper()
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
            elif section_word in {"C", "CURVE"} or section_word.endswith("_DEFINITION"):
                new_section = "C"
                if section_word.endswith("_DEFINITION"):
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
                    logger.warning(
                        "~Other section at line '%s' — ~Other is deprecated "
                        "in LAS 3.0. Content is preserved but should be "
                        "migrated to user-defined Parameter or Column Data sections.",
                        line[:80].strip(),
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
                    self._process_ascii_data()
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
                    self._process_ascii_data()
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
                    if _prev_data_section_type is not None:
                        _new_type = self._current_data_section_type
                        self._current_data_section_type = _prev_data_section_type
                        # F-I2-H01: Replay deferred well entries before processing
                        # data so the correct NULL value is available.
                        if self._version_found:
                            self._replay_deferred_well()
                        self._process_ascii_data()
                        self._current_data_section_type = _new_type
                    else:
                        # F-I2-H01: Same replay guard for the else branch.
                        if self._version_found:
                            self._replay_deferred_well()
                        self._process_ascii_data()
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

    def _match_data_line(self, line: str) -> re.Match[str] | None:
        """Try to match a header data line with colon, then without.

        F-I2-M07: Validates captured group lengths after a successful
        match to prevent unbounded string allocation from crafted files
        with 500MB values or descriptions.
        """
        match = DATA_LINE_PATTERN.match(line)
        if match:
            self._validate_data_line_fields(match)
            return match
        match = VALUE_ONLY_PATTERN.match(line)
        if match:
            self._validate_data_line_fields(match)
            return match
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

        # F-10: Set _version_found only after a valid data line match,
        # not unconditionally before validation.  Setting it before the
        # match caused spurious "missing mandatory well field" warnings
        # for non-matching lines in the ~V section.
        self._version_found = True

        mnemonic = match.group("mnemonic").upper().strip()
        value = match.group("value").strip()

        if mnemonic == "VERS":
            self.las_file.version.vers = value
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
                self.las_file.version.dlm = dlm_upper
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
                actual_value = description
                self.las_file.well.descriptions[mnemonic] = value
            else:
                # Auto-mode: detect CWLS vs lasio convention heuristically.
                if mnemonic in {"STRT", "STOP", "STEP", "NULL"}:
                    # Float-based numeric detection for mandatory fields.
                    # If pre-colon text parses as float → CWLS (value in
                    # correct position); otherwise → lasio (swap).
                    try:
                        float(value.replace("D", "E").replace("d", "e"))
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
        self.las_file.well[mnemonic] = actual_value
        if unit:
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
            is_las12 = self.las_file.version.vers.startswith("1.")
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
        # F-H01: Each deferred entry is a (section_type, section_name,
        # section_idx, line) tuple.  Group by (section_type, section_name,
        # section_idx) to create one DataSection per original pre-~V data
        # section.  section_idx disambiguates consecutive bare sections
        # that share the same type and name.
        if self._deferred_ascii_data_lines:
            if self.las_file.version.is_las30:
                # Build groups from per-line tuple storage.
                # itertools.groupby groups consecutive lines with the same
                # key — correct since deferred lines are appended in file
                # order and section boundaries are naturally contiguous.
                groups: list[tuple[tuple[str, str, int], list[str]]] = []
                for key, group_iter in groupby(
                    self._deferred_ascii_data_lines,
                    key=lambda t: (t[0], t[1], t[2]),
                ):
                    groups.append((key, [line for _, _, _, line in group_iter]))

                # Save current state before processing deferred groups.
                saved_lines = self._ascii_data_lines
                saved_curve_start = self._section_curve_start_idx
                saved_curve_end = self._section_curve_end_idx
                saved_section_type = self._current_data_section_type
                saved_section_name = self._current_section_name

                # Process each deferred group as its own DataSection.
                for (section_type, _section_name, _section_idx), raw_lines in groups:
                    # Use default curve indices (full range).
                    self._section_curve_start_idx = 0
                    self._section_curve_end_idx = None
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
                    self._process_ascii_data()
                    self._current_data_section_idx += 1

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

        # F-H03: Secondary colon-split fallback when VALUE_ONLY_PATTERN matched
        # and the value contains a bare colon (no surrounding whitespace).
        # CWLS/lasio files use "DESCRIPTION : VALUE" format where the colon
        # separates description from value, but DATA_LINE_PATTERN requires
        # whitespace on at least one side of the colon.  Bare-colon lines
        # like "DATE. LOG DATE:15/01/2001" fall through to VALUE_ONLY_PATTERN,
        # placing the entire string in *value* and setting *description* to
        # None — bypassing the CWLS/lasio swap logic entirely.
        if description is None and ":" in value:
            colon_idx = value.index(":")
            description = value[colon_idx + 1 :].strip()
            value = value[:colon_idx].strip()

        is_las12 = self.las_file.version.vers.startswith("1.")

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
            # Remove all format specifiers from description
            description = FORMAT_SPEC_PATTERN.sub("", description).strip()

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

        # Apply mnemonic normalization from mnem_base
        normalized = self._mnem_base_upper.get(raw_mnemonic, raw_mnemonic)

        curve = CurveDefinition(
            mnemonic=normalized,
            unit=unit,
            api_code=api_code,
            description=description,
            original_mnemonic=_original_cased if _original_cased.upper() != normalized else "",
            data_format=data_format,
            array_info=array_info,
        )
        # F-28: Guard against unbounded curve accumulation during ~C parsing.
        # Without this check, a metadata-only LAS 3.0 file can accumulate
        # unlimited CurveDefinition objects without triggering any bounds
        # check (MAX_CURVES was only checked later in _process_ascii_data,
        # which early-returns when no data section exists).
        if len(self.las_file.curves) >= MAX_CURVES:
            raise LASParseError(
                f"Curve count ({len(self.las_file.curves) + 1}) exceeds maximum "
                f"allowed ({MAX_CURVES}). The file may be malformed or corrupt."
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
        value = match.group("value").strip()
        description = (
            match.group("description").strip()
            if "description" in match.groupdict() and match.group("description")
            else ""
        )

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
        description = FORMAT_SPEC_PATTERN.sub("", description).strip()

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

        # Apply mnemonic normalization from mnem_base (same as curve handling)
        normalized = self._mnem_base_upper.get(raw_mnemonic, raw_mnemonic)

        param = ParameterEntry(
            mnemonic=normalized,
            unit=unit,
            value=value,
            description=description,
            array_index=array_index,
            zone=zone,
        )
        # F-M15: Store data_format as an instance attribute for roundtrip
        # fidelity.  Forward-compatible — works before and after the field
        # is added to the ParameterEntry dataclass.
        if param_data_format:
            param.data_format = param_data_format
        # F-29: Guard against unbounded parameter accumulation.
        # Curves have MAX_CURVES checked in 3 locations; parameters had zero
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
                if len(self._deferred_ascii_data_lines) >= MAX_DATA_LINES:
                    raise LASParseError(
                        f"Deferred ASCII data line count exceeds maximum "
                        f"allowed ({MAX_DATA_LINES}). "
                        f"The file may be malformed or corrupt."
                    )
                # F-H01: Store per-line (section_type, section_name,
                # section_idx, line) so _replay_deferred_well can
                # reconstruct per-section grouping.  section_idx
                # disambiguates consecutive bare sections.
                self._deferred_ascii_data_lines.append(
                    (
                        self._current_data_section_type,
                        self._current_section_name,
                        self._current_data_section_idx,
                        line,
                    )
                )
            return
        # F-27: Early bounds check during accumulation — reject before the
        # list grows unbounded.  The main check in _process_ascii_data runs
        # AFTER all lines are collected, offering no protection during the
        # accumulation phase itself.
        if len(self._ascii_data_lines) >= MAX_DATA_LINES:
            raise LASParseError(
                f"ASCII data line count exceeds maximum allowed "
                f"({MAX_DATA_LINES}) during accumulation. "
                f"The file may be malformed or corrupt."
            )
        self._ascii_data_lines.append(line)

    def _deduplicate_curves(
        self, section_curves: list[CurveDefinition], is_first_section: bool
    ) -> list[str]:
        """Deduplicate curve names and return the deduplicated order.

        F-M19: Extracted from _process_ascii_data to share between parser.py,
        data_reader.py, and dev_reader.py.  Each domain agent refactors
        its own copy; the algorithmic core is identical across all three.

        When curve mnemonics collide (duplicates within the same section),
        suffixes ``_2``, ``_3``, etc. are appended.  ``original_mnemonic``
        is preserved on the renamed copy so writers can reconstruct the
        original name.

        For the first data section, renamed mnemonics are written back to
        the global ``curves``/``curves_order`` lists so ``to_dict()`` and
        the writer see consistent names.

        Args:
            section_curves: Per-section curve definitions (modified in-place).
            is_first_section: True when this is the first data section
                processed for this file (triggers global writeback).

        Returns:
            Deduplicated curve mnemonic order list.
        """
        seen: dict[str, int] = {}
        deduped_order: list[str] = []
        output_names: set[str] = set()
        for i, curve in enumerate(section_curves):
            name = curve.mnemonic
            if name in seen:
                seen[name] += 1
                suffix = seen[name]
                new_name = f"{name}_{suffix}"
                while new_name in output_names:
                    suffix += 1
                    new_name = f"{name}_{suffix}"
                seen[name] = suffix
                section_curves[i] = CurveDefinition(
                    mnemonic=new_name,
                    unit=curve.unit,
                    api_code=curve.api_code,
                    description=curve.description,
                    original_mnemonic=name
                    if not curve.original_mnemonic
                    else curve.original_mnemonic,
                    data_format=curve.data_format,
                    array_info=curve.array_info,
                )
                deduped_order.append(new_name)
                output_names.add(new_name)
                if is_first_section:
                    global_idx = self._section_curve_start_idx + i
                    if not self.las_file.curves[global_idx].original_mnemonic:
                        self.las_file.curves[global_idx].original_mnemonic = name
                    self.las_file.curves[global_idx].mnemonic = new_name
                    self.las_file.curves_order[global_idx] = new_name
            elif name in output_names:
                suffix = 2
                new_name = f"{name}_{suffix}"
                while new_name in output_names:
                    suffix += 1
                    new_name = f"{name}_{suffix}"
                seen[name] = suffix
                section_curves[i] = CurveDefinition(
                    mnemonic=new_name,
                    unit=curve.unit,
                    api_code=curve.api_code,
                    description=curve.description,
                    original_mnemonic=name
                    if not curve.original_mnemonic
                    else curve.original_mnemonic,
                    data_format=curve.data_format,
                    array_info=curve.array_info,
                )
                deduped_order.append(new_name)
                output_names.add(new_name)
                if is_first_section:
                    global_idx = self._section_curve_start_idx + i
                    if not self.las_file.curves[global_idx].original_mnemonic:
                        self.las_file.curves[global_idx].original_mnemonic = name
                    self.las_file.curves[global_idx].mnemonic = new_name
                    self.las_file.curves_order[global_idx] = new_name
            else:
                seen[name] = 1
                deduped_order.append(name)
                output_names.add(name)
        return deduped_order

    def _process_ascii_data(self) -> None:
        """Process collected ASCII data lines into numpy arrays.

        Handles LAS 3.0 delimiters and string data formats.
        Uses per-section curves (F1) to support LAS 3.0 files where
        different ~C blocks define different curve sets before each ~A.
        """
        if not self._ascii_data_lines:
            return

        # F-003: LAS 3.0 WRAP=YES is unsupported — wrapped-mode data
        # processing is not implemented.  The previous logger.warning was
        # insufficient: it acknowledged the gap but allowed corrupt data
        # to be parsed, producing phantom rows and misaligned values.
        # Raising LASParseError prevents silent data corruption and makes
        # the limitation explicit.  Users must convert wrapped files to
        # unwrapped format (one line per depth step) or set WRAP=NO.
        if self.las_file.version.wrap.upper() == "YES":
            raise LASParseError(
                "LAS 3.0 WRAP=YES is not supported by pylasdev.  "
                "Convert the file to unwrapped format (one line per "
                "depth step) before parsing, or set WRAP to NO."
            )

        # Get delimiter character
        delimiter = self.las_file.version.delimiter_char

        # F1: Get per-section curves — only curves defined since the
        # most recent ~C block.  For the first data section (no ~C
        # encountered or _section_curve_start_idx == 0), this is the
        # full curve list (backward-compatible).
        # When a pipe "| CURVE" association set _section_curve_end_idx,
        # cap to the main curve block so LOG_DATA sections don't pick up
        # per-section curves from later Definition sections.
        if self._section_curve_end_idx is not None:
            section_curves = list(
                self.las_file.curves[self._section_curve_start_idx : self._section_curve_end_idx]
            )
        else:
            section_curves = list(self.las_file.curves[self._section_curve_start_idx :])
        if not section_curves:
            # F32: Warn when data is present but no curves are defined
            # for this section, then return early.
            warnings.warn(
                "ASCII data present but no curves defined for this section. "
                "Data has been discarded.",
                UserWarning,
                stacklevel=2,
            )
            return

        # F1: Local dedup on per-section curves.  For the first data section,
        # renamed mnemonics are also written back to global curves/curves_order
        # so that to_dict() and the writer see consistent, unique mnemonics
        # (M1: LAS 3.0 global curve dedup regression fix).
        is_first_section = not self.las_file.data_sections
        deduped_order = self._deduplicate_curves(section_curves, is_first_section)

        # Determine which curves are string type.
        # F-001: Previous one-liner (c.data_format in ("S", "A")) routed ALL
        # "A"-format curves as strings, including array elements with {A:N}
        # format specifiers where N is numeric.  Array-element curves (those
        # with array_info set) contain numeric data and must be routed as
        # numeric, not stored as np.str_.
        string_curves = {
            i: c.data_format in ("S",)
            or (c.data_format in ("A",) and c.array_info is None)
            for i, c in enumerate(section_curves)
        }

        # F2-05: Validate curve format types — unrecognized formats silently
        # produce null data when routed through _to_finite_float().  Known
        # numeric format types are F (float), E (exponential), D (Fortran
        # double), and S (string).  The empty string (no format specifier)
        # defaults to numeric.  Non-numeric formats ({DEG}, date templates
        # like {DD/MM/YYYY}) route through float() and produce null_value
        # for every data point without any warning.
        #
        # F-M03: Validate the FULL format specifier, not just the first
        # character.  The prior `fmt[0]` check passed both "D" (valid
        # Fortran double) and "DEG"/"DD/MM/YYYY" (non-numeric templates)
        # since all start with "D".  Now we require the format to either
        # be a single known type letter, or a known letter followed by a
        # valid width.precision pattern (e.g., "F8.3", "E10.2").
        _KNOWN_CURVE_FORMATS: frozenset[str] = frozenset({"F", "E", "D", "S", "A"})
        # F-M03: Match full numeric format specifiers — single letter or
        # letter + width[.precision][±exponent] (e.g., "F8.3", "E0.00E+00").
        # Non-numeric templates like "DEG" and "DD/MM/YYYY" fail because
        # the second character is a letter or slash, not a digit.
        # F-M03 / R9-018: S and A are bare format specifiers with no
        # numeric width suffix (LAS spec).  The previous character class
        # [FEDS] incorrectly accepted "S10" as valid, causing string
        # curves to be classified as numeric and their values to become
        # NaN.  Split into: [FED] with optional width, or [SA] bare.
        _FORMAT_SPEC_RE = re.compile(
            r"^(?:[FED](?:\d+(?:\.\d+)?(?:[ED][+-]?\d+)?)?|[SA])$"
        )
        for curve in section_curves:
            fmt = curve.data_format
            if fmt and not (
                fmt in _KNOWN_CURVE_FORMATS or _FORMAT_SPEC_RE.match(fmt)
            ):
                logger.warning(
                    "Curve '%s' has unsupported format specifier '{%s}'. "
                    "Non-numeric format types (e.g., {DEG}, date "
                    "templates) cannot be converted to float in the LAS "
                    "3.0 parser and will produce null values for every "
                    "data point.",
                    curve.mnemonic,
                    fmt,
                )

        # Get null value (shared utility, used by parser, data_reader, writer)
        null_value = _get_null_value(self.las_file.well)

        # Create data section with per-section curves.
        data_section = DataSection(
            name=self._current_section_name or f"Section_{self._current_data_section_idx}",
            section_type=self._current_data_section_type,
            curves_order=deduped_order,
            section_curves=list(section_curves),
        )

        num_curves = len(section_curves)

        # Count actual data lines (excluding comments and blank lines) for array sizing.
        actual_count = sum(
            1
            for line in self._ascii_data_lines
            if not COMMENT_PATTERN.match(line) and not EMPTY_PATTERN.match(line)
        )

        # F-I2-M09: Guard against zero data rows.  When actual_count is 0
        # (all lines are comments/blanks), num_curves * 0 = 0 always passes
        # MAX_TOTAL_ELEMENTS check, but np.zeros(0) still allocates an empty
        # ndarray per curve — up to MAX_CURVES per section and MAX_DATA_SECTIONS
        # sections can produce GB-scale allocations within guard limits.
        # A section with zero data rows has nothing to process; return early.
        if actual_count == 0:
            warnings.warn(
                f"Data section '{self._current_section_name or 'ASCII'}' has "
                f"{len(self._ascii_data_lines)} raw line(s) but 0 data rows "
                f"(all lines are comments or blanks). No data will be stored "
                f"for this section.",
                UserWarning,
                stacklevel=2,
            )
            return

        # F-26: Global aggregate limit across ALL data sections.
        # Each section passes per-section bounds (MAX_DATA_LINES, MAX_CURVES,
        # MAX_TOTAL_ELEMENTS) individually, but an attacker can craft N
        # sections (each just under the limits) to cumulatively exhaust
        # memory.  This caps the total number of data sections processed.
        if self._current_data_section_idx >= MAX_DATA_SECTIONS:
            raise LASParseError(
                f"Data section count ({self._current_data_section_idx + 1}) exceeds "
                f"maximum allowed ({MAX_DATA_SECTIONS}). "
                f"The file may be malformed or corrupt."
            )

        if actual_count > MAX_DATA_LINES:
            raise LASParseError(
                f"ASCII data line count ({actual_count}) exceeds maximum allowed "
                f"({MAX_DATA_LINES}). The file may be malformed or corrupt."
            )
        if num_curves > MAX_CURVES:
            raise LASParseError(
                f"Curve count ({num_curves}) exceeds maximum allowed "
                f"({MAX_CURVES}). The file may be malformed or corrupt."
            )

        # Combined bound: protect against combination attacks where individual
        # curve_count and data_line_count checks pass but product exhausts memory.
        if num_curves * actual_count > MAX_TOTAL_ELEMENTS:
            raise LASParseError(
                f"Total allocation ({num_curves} curves x {actual_count} lines = "
                f"{num_curves * actual_count} elements) exceeds maximum allowed "
                f"({MAX_TOTAL_ELEMENTS}). The file may be malformed or corrupt."
            )

        # PERF-03: Pre-allocate numpy arrays for numeric curves.
        string_data_lists: dict[int, list[str]] = {}
        for i, curve in enumerate(section_curves):
            if string_curves.get(i, False):
                string_data_lists[i] = []
            else:
                arr = np.zeros(actual_count, dtype=np.float64)
                data_section.data[curve.mnemonic] = arr
                # F2-014: las_file.logs assigned via defensive copy after data
                # fill to avoid shared mutable ndarray between LASFile.logs
                # and DataSection.data — in-place mutations were silently
                # corrupting both views.

        # Fill arrays by index (no list accumulation overhead for numerics)
        numeric_arrays = [
            data_section.data[c.mnemonic] if not string_curves.get(i, False) else None
            for i, c in enumerate(section_curves)
        ]
        idx = 0
        warned_extra = False  # Track extra-column warning per section
        warned_short = False  # F-11: Track short-row warning per section
        for line in self._ascii_data_lines:
            # Skip comment lines and blank/whitespace-only lines.
            # F-32: EMPTY_PATTERN was defined at module level but never
            # used in this loop — blank lines split to [''] and produce
            # a full row of null_value entries, silently inflating data.
            if COMMENT_PATTERN.match(line) or EMPTY_PATTERN.match(line):
                continue

            # Split by delimiter.
            # F-I2-M01: Strip the line before splitting with TAB/COMMA
            # delimiters to avoid leading whitespace producing an empty
            # first token and column shift.  SPACE mode is unaffected
            # (str.split(None) strips implicitly).  Consistent with
            # data_reader.py which strips before all delimiter splits.
            # F2-015: Use csv.reader for TAB/COMMA delimiters so values
            # containing the delimiter inside double-quotes are NOT
            # incorrectly split (e.g., "Run 1, Tool A" stays as one token
            # with COMMA delimiter).  csv.QUOTE_MINIMAL handles CSV-style
            # quoting: fields are quoted only when they contain the
            # delimiter, quotechar, or line terminator.
            if delimiter == " ":
                values = line.split(maxsplit=MAX_TOKENS_PER_LINE)
            else:
                # F-I2-M12: Wrap csv.reader in try/except csv.Error.
                # csv.Error (e.g. unclosed quotes, field-size overflow)
                # is NOT a LASParseError subclass — without this guard
                # it escapes the public API unwrapped.
                try:
                    reader = csv.reader(
                        [line.strip()], delimiter=delimiter, quoting=csv.QUOTE_MINIMAL
                    )
                    row = next(reader)
                except csv.Error as exc:
                    raise LASParseError(
                        f"CSV parsing error in data section "
                        f"'{self._current_section_name or 'ASCII'}': {exc}"
                    ) from exc
                # Safety cap: prevent unbounded token count from malformed
                # input, matching the maxsplit behavior of str.split.
                values = row[: MAX_TOKENS_PER_LINE + 1]

            # Warn about extra columns being silently discarded
            if len(values) > num_curves and not warned_extra:
                warned_extra = True
                logger.warning(
                    "Data line in section '%s' has %d values but only "
                    "%d curves declared. Extra columns are discarded.",
                    self._current_section_name or "ASCII",
                    len(values),
                    num_curves,
                )

            # F-11: Warn when non-wrapped data lines have fewer values than
            # declared curves.  Short rows in wrapped mode are expected
            # (values span multiple lines), so this warning only fires in
            # non-wrapped (WRAP=NO) mode.
            if len(values) < num_curves and not warned_short:
                is_not_wrapped = self.las_file.version.wrap.upper() != "YES"
                if is_not_wrapped:
                    warned_short = True
                    logger.warning(
                        "Data line in section '%s' has %d values but %d "
                        "curves declared. Missing values are filled with "
                        "the null value (%s).",
                        self._current_section_name or "ASCII",
                        len(values),
                        num_curves,
                        null_value,
                    )

            # Pad with null values if needed
            while len(values) < num_curves:
                values.append(str(null_value))

            for i in range(num_curves):
                val_str = values[i].strip()
                if string_curves.get(i, False):
                    string_data_lists[i].append(val_str)
                else:
                    val = _to_finite_float(val_str, null_value)
                    arr = numeric_arrays[i]  # type: ignore[assignment]
                    if arr is None:
                        raise RuntimeError(
                            f"Internal error: numeric array '{i}' was not pre-allocated"
                        )
                    arr[idx] = val

            idx += 1

        # F2-014: Copy filled numeric arrays from data_section.data to
        # las_file.logs for independent views.  The allocation above only
        # assigned to data_section.data; the fill loop wrote values via
        # numeric_arrays (which references data_section.data).  Now copy
        # the fully-populated arrays so in-place mutations on one view do
        # not silently corrupt the other.
        if is_first_section:
            for curve in section_curves:
                if curve.mnemonic in data_section.data:
                    self.las_file.logs[curve.mnemonic] = (
                        data_section.data[curve.mnemonic].copy()
                    )

        # Convert string data lists to numpy arrays
        for i, curve in enumerate(section_curves):
            if i in string_data_lists:
                string_arr = np.array(string_data_lists[i], dtype=np.str_)
                data_section.string_data[curve.mnemonic] = string_arr
                if is_first_section:
                    # F2-014: Defensive copy — prevents shared-reference
                    # mutation between LASFile.string_data and
                    # DataSection.string_data.
                    self.las_file.string_data[curve.mnemonic] = string_arr.copy()

        # Store data section (LAS 3.0)
        self.las_file.data_sections.append(data_section)

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
        if self.las_file.version.is_las30:
            data_before_curves = False
            curve_seen = False
            for label in self._section_sequence:
                # F-M02: Extract section_word from the label (which may
                # include a ":section_name" suffix) instead of using
                # label[0].  The prior code assumed single-letter codes
                # (e.g., "A" for data, "C" for curves), but commit 6156a03
                # changed _section_sequence to store full section words
                # (e.g., "CORE_DATA", "CORE_DEFINITION").  CORE_DATA[0]="C"
                # was miscategorized as a curve section, producing
                # false-negative data-before-curves warnings.
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
                if is_data and not curve_seen:
                    data_before_curves = True
                    break
                if is_curve:
                    curve_seen = True
            if data_before_curves:
                logger.warning(
                    "LAS 3.0 file contains data sections before curve "
                    "definition sections. Data sections without preceding "
                    "curve definitions will have no curves to reference "
                    "and may produce empty or truncated output."
                )

        # (3) Duplicate section headers.
        name_counts: dict[str, int] = {}
        for label in self._section_sequence:
            name_counts[label] = name_counts.get(label, 0) + 1
        for label, count in name_counts.items():
            if count > 1:
                logger.warning(
                    "Duplicate section header '~%s' encountered %d times. "
                    "Repeated sections may indicate a malformed file or "
                    "cause data from earlier instances to be overwritten.",
                    label,
                    count,
                )
