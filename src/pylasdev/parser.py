"""Regex-based LAS file parser replacing PLY.

The LAS format is line-based with a simple structure:
  MNEMONIC.UNIT  VALUE : DESCRIPTION

PLY (lex/yacc) is overkill for this. Regex reduces ~450 lines to ~150
while maintaining the same parsing capability.

Supports LAS 1.2, 2.0, and 3.0 formats.
"""

from __future__ import annotations

import re
from typing import ClassVar

import numpy as np

from .models import (
    ArrayElementInfo,
    CurveDefinition,
    DataSection,
    LASFile,
    ParameterEntry,
    ParameterZone,
)

# Section header: line starting with ~, followed by section letter or name
SECTION_PATTERN = re.compile(r"^~([A-Za-z])(.*)")

# Data line pattern: MNEMONIC.UNIT  VALUE : DESCRIPTION
# Uses \w which matches Unicode (including Cyrillic) in Python 3
# Note: LAS files commonly have spaces between mnemonic and dot (e.g., "DT  .US/M")
DATA_LINE_PATTERN = re.compile(
    r"^\s*"
    r"(?P<mnemonic>[\w\-]+(?:\[\d+\])?)"  # mnemonic: word chars + hyphen + optional [N] array index
    r"\s*"  # optional whitespace before dot (common in LAS files)
    r"\."  # literal dot separator
    r"(?P<unit>[\w\-/]*)"  # unit: optional, can include /
    r"\s+"  # whitespace separator
    r"(?P<value>[^:]*?)"  # value: everything up to colon
    r"\s*:\s*"  # colon separator
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
        # Build uppercased lookup for case-insensitive matching
        self._mnem_base_upper = {k.upper(): v for k, v in self.mnem_base.items()}
        self._reset()

    def _reset(self) -> None:
        """Reset parser state for a new file."""
        self.las_file = LASFile()
        self._current_section: str | None = None
        self._current_section_name: str = ""
        self._line_number = 0
        self._wrap_mode = False
        self._data_line_count = 0
        self._ascii_data_lines: list[str] = []
        self._current_data_section_idx: int = 0

    def parse(self, content: str) -> LASFile:
        """Parse LAS file content string."""
        self._reset()

        lines = content.splitlines()
        self._pre_scan(lines)

        for i, line in enumerate(lines, 1):
            self._line_number = i
            self._parse_line(line)

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
            match = SECTION_PATTERN.match(line)
            if match:
                in_ascii = match.group(1).upper() == "A"
                continue
            if in_ascii and not COMMENT_PATTERN.match(line) and not EMPTY_PATTERN.match(line):
                count += 1

        self._data_line_count = count

    def _parse_line(self, line: str) -> None:
        """Route a single line to the appropriate section handler."""
        section_match = SECTION_PATTERN.match(line)
        if section_match:
            new_section = section_match.group(1).upper()
            # If we're switching to a new ~A section, process previous data first
            if new_section == "A" and self._current_section == "A":
                self._process_ascii_data()
                self._ascii_data_lines = []
                self._current_data_section_idx += 1
            self._current_section = new_section
            self._current_section_name = section_match.group(2).strip()
            return

        if COMMENT_PATTERN.match(line) or EMPTY_PATTERN.match(line):
            return

        if self._current_section:
            handler_name = self.SECTION_HANDLERS.get(self._current_section)
            if handler_name:
                getattr(self, handler_name)(line)

    def _match_data_line(self, line: str) -> re.Match[str] | None:
        """Try to match a header data line with colon, then without."""
        match = DATA_LINE_PATTERN.match(line)
        if match:
            return match
        return VALUE_ONLY_PATTERN.match(line)

    def _parse_version(self, line: str) -> None:
        """Parse ~V (version) section line."""
        match = self._match_data_line(line)
        if not match:
            return

        mnemonic = match.group("mnemonic").upper().strip()
        value = match.group("value").strip()

        if mnemonic == "VERS":
            self.las_file.version.vers = value
        elif mnemonic == "WRAP":
            self.las_file.version.wrap = value.upper()
            self._wrap_mode = value.upper() == "YES"
        elif mnemonic == "DLM":
            self.las_file.version.dlm = value

    def _parse_well(self, line: str) -> None:
        """Parse ~W (well information) section line."""
        match = self._match_data_line(line)
        if not match:
            return

        mnemonic = match.group("mnemonic").upper().strip()
        value = match.group("value").strip()

        self.las_file.well[mnemonic] = value

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
                array_time_offset = float(format_match.group("offset"))
            # Remove format specifier from description
            description = FORMAT_SPEC_PATTERN.sub("", description).strip()

        # LAS 3.0: Check for array notation in mnemonic
        array_info: ArrayElementInfo | None = None
        array_match = ARRAY_MNEMONIC_PATTERN.match(raw_mnemonic)
        if array_match:
            base_name = array_match.group("base").upper()
            index = int(array_match.group("index"))
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
            zone = ParameterZone(
                zone_name=zone_match.group("zone").upper(),
                zone_index=(int(zone_match.group("index")) if zone_match.group("index") else None),
            )
            # Remove zone association from description
            description = ZONE_ASSOC_PATTERN.sub("", description).strip()

        # LAS 3.0: Check for array notation in mnemonic
        array_index: int | None = None
        array_match = ARRAY_MNEMONIC_PATTERN.match(raw_mnemonic)
        if array_match:
            array_index = int(array_match.group("index"))

        param = ParameterEntry(
            mnemonic=raw_mnemonic,
            unit=unit,
            value=value,
            description=description,
            array_index=array_index,
            zone=zone,
        )
        self.las_file.parameters.append(param)

    def _parse_other(self, line: str) -> None:
        """Parse ~O (other) section — free-form text, accumulated."""
        self.las_file.other += line + "\n"

    def _parse_ascii_data(self, line: str) -> None:
        """Collect ASCII data lines for later processing.

        In LAS 3.0, data can be delimited by SPACE, TAB, or COMMA.
        Data is collected and processed after all lines are parsed.
        """
        self._ascii_data_lines.append(line)

    def _process_ascii_data(self) -> None:
        """Process collected ASCII data lines into numpy arrays.

        Handles LAS 3.0 delimiters and string data formats.
        """
        if not self._ascii_data_lines:
            return

        # Get delimiter character
        delimiter = self.las_file.version.delimiter_char

        # Get curve information
        curves = self.las_file.curves
        if not curves:
            return

        # Determine which curves are string type
        string_curves = {i: c.data_format == "S" for i, c in enumerate(curves)}

        # Get null value
        null_value = float(self.las_file.well.get("NULL", "-999.25"))

        # Create data section
        data_section = DataSection(
            name=self._current_section_name or f"Section_{self._current_data_section_idx}",
            curves_order=[c.mnemonic for c in curves],
        )

        # Parse data lines
        num_curves = len(curves)
        data_arrays: list[list[float | str]] = [[] for _ in range(num_curves)]

        for line in self._ascii_data_lines:
            # Skip comment lines
            if COMMENT_PATTERN.match(line):
                continue

            # Split by delimiter
            if delimiter == " ":
                # For space delimiter, split on any whitespace
                values = line.split()
            else:
                values = line.split(delimiter)

            # Pad with null values if needed
            while len(values) < num_curves:
                values.append(str(null_value))

            for i in range(num_curves):
                val_str = values[i].strip() if i < len(values) else str(null_value)
                try:
                    if string_curves.get(i, False):
                        data_arrays[i].append(val_str)
                    else:
                        val = float(val_str) if val_str else null_value
                        data_arrays[i].append(val)
                except ValueError:
                    if string_curves.get(i, False):
                        data_arrays[i].append(val_str)
                    else:
                        data_arrays[i].append(null_value)

        # Convert to numpy arrays
        for i, curve in enumerate(curves):
            if string_curves.get(i, False):
                self.las_file.string_data[curve.mnemonic] = np.array(data_arrays[i], dtype=np.str_)
                data_section.data[curve.mnemonic] = np.array(
                    [0.0] * len(data_arrays[i]), dtype=np.float64
                )
            else:
                arr = np.array(data_arrays[i], dtype=np.float64)
                self.las_file.logs[curve.mnemonic] = arr
                data_section.data[curve.mnemonic] = arr

        # Store data section (LAS 3.0)
        self.las_file.data_sections.append(data_section)
