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
from typing import ClassVar

import numpy as np

from .data_reader import (
    MAX_CURVES,
    MAX_DATA_LINES,
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

# Section header: line starting with ~, followed by section keyword and trailing content.
# Captures the full section word (e.g., "VERSION", "Core_Definition", "Core[1]")
# and any trailing content (e.g., "| Core_Definition", "INFORMATION").
# F-01/F-08: Previous pattern ^~([A-Za-z])(.*) captured only the first letter,
# causing ~Core_Data and ~Core_Definition to both route to the curve handler (letter C).
SECTION_PATTERN = re.compile(r"^~(\S+)(.*)")

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

# LAS 3.0: Format specifier in braces (e.g., {F}, {E}, {S}, {A:0})
FORMAT_SPEC_PATTERN = re.compile(r"\{(?P<format>[FESA]):?(?P<offset>[\d.]*)\s*\}")

# LAS 3.0: Zone association via pipe (e.g., | Run[1], | Zone[2])
ZONE_ASSOC_PATTERN = re.compile(r"\|\s*(?P<zone>[\w\-]+)(?:\[(?P<index>\d+)\])?$")

COMMENT_PATTERN = re.compile(r"^\s*#")
EMPTY_PATTERN = re.compile(r"^\s*$")

# LAS 3.0 section type keywords for structured data-type sections.
# These are data sections (contain rows of values) — NOT definition sections.
# Definition sections (Core_Definition, Drilling_Definition, etc.) are
# routed to _parse_curve since they define curves for their data type.
_DATA_SECTION_WORDS = {
    "A", "ASCII",            # Standard log data
    "CORE",                   # Core data section
    "CORE_DATA",              # Core data section (written form)
    "DRILLING",               # Drilling data section
    "DRILLING_DATA",          # Drilling data section (written form)
    "INCLINOMETRY",           # Inclinometry data section
    "INCLINOMETRY_DATA",      # Inclinometry data section (written form)
    "TOPS",                   # Tops data section
    "TOPS_DATA",              # Tops data section (written form)
    "TEST",                   # Test data section
    "TEST_DATA",              # Test data section (written form)
    "PERFORATIONS",           # Perforations data section
    "PERFORATIONS_DATA",      # Perforations data section (written form)
    "LOG_DATA",               # Explicit log data section
}

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
    "LOG_DATA": "LOG_DATA",
}


class LASParser:
    """Regex-based LAS file parser.

    Encapsulates all parsing state in the instance (no global variables).
    Thread-safe: each instance maintains its own state.

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

    def __init__(self, mnem_base: dict[str, str] | None = None) -> None:
        """Initialize parser with optional mnemonic base."""
        self.mnem_base = mnem_base or {}
        # Build uppercased lookup with multi-step chain resolution.
        # resolve_mnemonic walks chains like BK-3 → BK → BFV to reach
        # the terminal canonical name. Single .get() only resolves one hop.
        _raw_upper = {k.upper(): v for k, v in self.mnem_base.items()}
        self._mnem_base_upper: dict[str, str] = {}
        for k in _raw_upper:
            self._mnem_base_upper[k] = resolve_mnemonic(_raw_upper, k)
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
        self.source_file: str = ""
        # F1: Track per-section curve boundaries for LAS 3.0 files where
        # different ~C blocks define different curve sets before each ~A section.
        self._section_curve_start_idx: int = 0
        # F-01/F-08: Track the end of the main (non-_Definition) curve block.
        # When a data section has a pipe "| CURVE" association, the per-section
        # curves revert to using the main block (curves[0:_main_curve_end]).
        self._main_curve_end: int = 0

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
            lines = content.splitlines()
        self._pre_scan(lines)

        for line in lines:
            self._parse_line(line)

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

        return self.las_file

    def _pre_scan(self, lines: list[str]) -> None:
        """Pre-scan to count ASCII data lines."""
        in_ascii = False
        count = 0

        for line in lines:
            stripped = line.strip()
            match = SECTION_PATTERN.match(stripped)
            if match:
                section_word = match.group(1).upper()
                in_ascii = section_word in _DATA_SECTION_WORDS
                continue
            if (
                in_ascii
                and not COMMENT_PATTERN.match(stripped)
                and not EMPTY_PATTERN.match(stripped)
            ):
                count += 1

        self._data_line_count = count

    def _parse_line(self, line: str) -> None:
        """Route a single line to the appropriate section handler."""
        section_match = SECTION_PATTERN.match(line)
        if section_match:
            section_word = section_match.group(1).upper()
            section_rest = section_match.group(2).strip()

            # Parse pipe-delimited definition association (LAS 3.0).
            # e.g., "~Core[1] | Core_Definition" or "~ASCII | CURVE".
            pipe_target: str | None = None
            pipe_idx = section_rest.find("|")
            if pipe_idx >= 0:
                # Split section name at pipe; pipe target follows.
                pipe_target = section_rest[pipe_idx + 1:].strip()
                section_rest = section_rest[:pipe_idx].strip()
            elif "|" in section_word:
                # Edge case: no space before pipe (e.g., "~Core|Definition")
                pipe_idx_w = section_word.find("|")
                pipe_target = (section_word[pipe_idx_w + 1:] + " " + section_rest).strip()
                section_word = section_word[:pipe_idx_w].strip()
                section_rest = ""

            # M-03: Save the previous data section type before the new-section
            # detection block overwrites it.  When two consecutive data sections
            # appear (e.g., ~A then ~Core[1]), this saved value is used to give
            # the PREVIOUS section's DataSection its correct section_type.
            _prev_data_section_type: str | None = None

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
                    if self._main_curve_end == 0:
                        self._main_curve_end = len(self.las_file.curves)
                else:
                    section_name = section_rest or ""
            elif section_word in {"P", "PARAMETER", "PARAMETERS"}:
                new_section = "P"
                section_name = section_rest or ""
            elif section_word in {"O", "OTHER"}:
                new_section = "O"
                section_name = section_rest or ""
            elif section_word in _DATA_SECTION_WORDS or section_word.startswith("CORE["):
                # LAS 3.0 structured data sections and standard ~A/ASCII.
                new_section = "A"
                # For standard 'A' or 'ASCII' sections and written-form *_DATA
                # sections (e.g., ~DRILLING_DATA DRILLING), use only the rest
                # as the section name (backward-compatible).
                if section_word in {"A", "ASCII"} or section_word.endswith("_DATA"):
                    section_name = section_rest
                else:
                    section_name = f"{section_word} {section_rest}".strip() if section_rest else section_word
                # M-03: Save the previous data section type BEFORE overwriting.
                # When two consecutive data sections appear (e.g., ~A then
                # ~Core[1]),_process_ascii_data() for the PREVIOUS section
                # must use ITS type, not the newly-set one.
                _prev_data_section_type = self._current_data_section_type
                # Derive section_type from keyword, falling back to LOG_DATA.
                if section_word.startswith("CORE["):
                    self._current_data_section_type = "CORE_DATA"
                else:
                    self._current_data_section_type = _SECTION_TYPE_MAP.get(
                        section_word, "LOG_DATA"
                    )
                # F-08: Handle pipe-delimited definition association.
                # "| CURVE" means use the main curve block (before
                # _Definition sections). "| X_Definition" means use
                # the per-section curves from that definition block.
                if pipe_target:
                    pipe_target_upper = pipe_target.upper()
                    if pipe_target_upper in {"CURVE", "C"}:
                        # Pipe "| CURVE" → use main curve block.
                        self._section_curve_start_idx = 0
                    # For specific definition targets (e.g., "| Core_Definition"),
                    # the per-section curve tracking already positions correctly.
                    # _section_curve_start_idx was set when entering that
                    # _Definition's ~C block.
            else:
                # Unknown section type — accumulate lines as free-form text (like ~O).
                # F-01 fix: Actually accumulate lines instead of just warning.
                new_section = None
                section_name = f"{section_word} {section_rest}".strip()
                self._other_lines.append(line)
                source_info = f" in {self.source_file}" if self.source_file else ""
                logger.warning(
                    "Unknown section type '~%s' at line%s — lines accumulated as other section: %s",
                    section_word,
                    source_info,
                    line[:80],
                )
                return

            # F1: When leaving a data section for a non-data section,
            # process pending data from the previous section.
            if new_section is not None:
                if self._current_section == "A" and new_section != "A":
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
                    if _prev_data_section_type is not None:
                        _new_type = self._current_data_section_type
                        self._current_data_section_type = _prev_data_section_type
                        self._process_ascii_data()
                        self._current_data_section_type = _new_type
                    else:
                        self._process_ascii_data()
                    self._ascii_data_lines = []
                    self._current_data_section_idx += 1

                # F1: Track per-section curve boundaries for LAS 3.0.
                # When entering ~C (including _Definition sections), mark the
                # current curve list position for per-section curve scoping.
                if new_section == "C":
                    self._section_curve_start_idx = len(self.las_file.curves)

                self._current_section = new_section
                self._current_section_name = section_name.strip() if section_name else section_word
                return

        if COMMENT_PATTERN.match(line) or EMPTY_PATTERN.match(line):
            return

        if self._current_section:
            handler_name = self.SECTION_HANDLERS.get(self._current_section)
            if handler_name:
                getattr(self, handler_name)(line)
            else:
                # Fallback for unknown section types — accumulate in _other_lines.
                self._other_lines.append(line)

    def _match_data_line(self, line: str) -> re.Match[str] | None:
        """Try to match a header data line with colon, then without."""
        match = DATA_LINE_PATTERN.match(line)
        if match:
            return match
        return VALUE_ONLY_PATTERN.match(line)

    def _parse_version(self, line: str) -> None:
        """Parse ~V (version) section line."""
        self._version_found = True
        match = self._match_data_line(line)
        if not match:
            return

        mnemonic = match.group("mnemonic").upper().strip()
        value = match.group("value").strip()

        if mnemonic == "VERS":
            self.las_file.version.vers = value
        elif mnemonic == "WRAP":
            self.las_file.version.wrap = value.upper()
        elif mnemonic == "DLM":
            self.las_file.version.dlm = value

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
            return

        mnemonic = match.group("mnemonic").upper().strip()
        unit = match.group("unit") or ""
        value = match.group("value").strip()
        description = (
            match.group("description").strip()
            if "description" in match.groupdict() and match.group("description")
            else ""
        )

        # LAS 1.2: the actual well data is AFTER the colon (in the description
        # group), not before it.  Swap value and description accordingly.
        is_las12 = self.las_file.version.vers.startswith("1.")
        if is_las12 and description:
            # LAS 1.2: MNEM.UNIT DESCRIPTION :VALUE
            actual_value = description
        else:
            # LAS 2.0+: MNEM.UNIT VALUE :DESCRIPTION
            actual_value = value

        self.las_file.well[mnemonic] = actual_value
        if unit:
            self.las_file.well.units[mnemonic] = unit

    def _parse_curve(self, line: str) -> None:
        """Parse ~C (curve information) section line.

        Supports LAS 3.0 features:
        - Array notation: NMR[1], NMR[2], etc.
        - Format specifiers: {F}, {E}, {S}, {A:0}
        """
        match = self._match_data_line(line)
        if not match:
            return

        raw_mnemonic = match.group("mnemonic").upper().strip()
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
        format_match = FORMAT_SPEC_PATTERN.search(description)
        if format_match:
            data_format = format_match.group("format")
            if data_format == "A" and format_match.group("offset"):
                try:
                    array_time_offset = float(format_match.group("offset"))
                except ValueError as exc:
                    raise LASParseError(
                        f"Invalid format specifier offset: "
                        f"'{format_match.group('offset')}' is not a valid number "
                        f"in curve description '{description}'"
                    ) from exc
                if not np.isfinite(array_time_offset):
                    raise LASParseError(
                        f"Format specifier offset overflow: "
                        f"'{format_match.group('offset')}' produced "
                        f"{array_time_offset} in curve description '{description}'"
                    )
            # Remove format specifier from description
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
            original_mnemonic=raw_mnemonic if raw_mnemonic != normalized else "",
            data_format=data_format,
            array_info=array_info,
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
            return

        raw_mnemonic = match.group("mnemonic").upper().strip()
        unit = match.group("unit") or ""
        value = match.group("value").strip()
        description = (
            match.group("description").strip()
            if "description" in match.groupdict() and match.group("description")
            else ""
        )

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
            zone = ParameterZone(
                zone_name=zone_match.group("zone").upper(),
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
        self.las_file.parameters.append(param)

    def _parse_other(self, line: str) -> None:
        """Parse ~O (other) section — free-form text, accumulated."""
        self._other_lines.append(line)

    def _parse_ascii_data(self, line: str) -> None:
        """Collect ASCII data lines for later processing.

        In LAS 3.0, data can be delimited by SPACE, TAB, or COMMA.
        Data is collected and processed after all lines are parsed.

        For LAS 1.2/2.0, ASCII data is handled by data_reader, so no
        collection is needed here.
        """
        if not self.las_file.version.is_las30:
            return
        self._ascii_data_lines.append(line)

    def _process_ascii_data(self) -> None:
        """Process collected ASCII data lines into numpy arrays.

        Handles LAS 3.0 delimiters and string data formats.
        Uses per-section curves (F1) to support LAS 3.0 files where
        different ~C blocks define different curve sets before each ~A.
        """
        if not self._ascii_data_lines:
            return

        # Get delimiter character
        delimiter = self.las_file.version.delimiter_char

        # F1: Get per-section curves — only curves defined since the
        # most recent ~C block.  For the first data section (no ~C
        # encountered or _section_curve_start_idx == 0), this is the
        # full curve list (backward-compatible).
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
        seen: dict[str, int] = {}
        deduped_order: list[str] = []
        output_names: set[str] = set()  # F12: dynamic set for cross-base collision detection
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
                # Create a renamed copy
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
                # M1: For the first data section, write renamed mnemonic
                # back to global curves/curves_order so the global metadata
                # stays consistent with the locally-deduped data.
                if is_first_section:
                    global_idx = self._section_curve_start_idx + i
                    self.las_file.curves[global_idx].mnemonic = new_name
                    self.las_file.curves_order[global_idx] = new_name
            elif name in output_names:
                # F12: Cross-base collision — an original name matches a
                # previously generated _N suffix, or a previously renamed
                # curve.  Rename to avoid duplicate keys.
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
                    self.las_file.curves[global_idx].mnemonic = new_name
                    self.las_file.curves_order[global_idx] = new_name
            else:
                seen[name] = 1
                deduped_order.append(name)
                output_names.add(name)

        # Determine which curves are string type
        string_curves = {i: c.data_format == "S" for i, c in enumerate(section_curves)}

        # Get null value (shared utility, used by parser, data_reader, writer)
        null_value = _get_null_value(self.las_file.well)

        # Create data section with per-section curves.
        data_section = DataSection(
            name=self._current_section_name or f"Section_{self._current_data_section_idx}",
            section_type=self._current_data_section_type,
            curves_order=deduped_order,
        )

        num_curves = len(section_curves)

        # Count actual data lines (excluding comments) for array sizing.
        actual_count = sum(1 for line in self._ascii_data_lines if not COMMENT_PATTERN.match(line))

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
                if is_first_section:
                    self.las_file.logs[curve.mnemonic] = arr
                data_section.data[curve.mnemonic] = arr

        # Fill arrays by index (no list accumulation overhead for numerics)
        numeric_arrays = [
            data_section.data[c.mnemonic] if not string_curves.get(i, False) else None
            for i, c in enumerate(section_curves)
        ]
        idx = 0
        warned_extra = False  # Track extra-column warning per section
        warned_short = False  # F-11: Track short-row warning per section
        for line in self._ascii_data_lines:
            # Skip comment lines
            if COMMENT_PATTERN.match(line):
                continue

            # Split by delimiter
            if delimiter == " ":
                values = line.split()
            else:
                values = line.split(delimiter)

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
                        raise LASParseError(
                            f"Internal error: numeric array '{i}' was not pre-allocated"
                        )
                    arr[idx] = val

            idx += 1

        # Convert string data lists to numpy arrays
        for i, curve in enumerate(section_curves):
            if i in string_data_lists:
                string_arr = np.array(string_data_lists[i], dtype=np.str_)
                data_section.string_data[curve.mnemonic] = string_arr
                if is_first_section:
                    self.las_file.string_data[curve.mnemonic] = string_arr

        # Store data section (LAS 3.0)
        self.las_file.data_sections.append(data_section)
