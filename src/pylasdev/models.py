"""Data models for LAS file structures.

Supports LAS 1.2, 2.0, and 3.0 formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class DelimiterType(Enum):
    """Delimiter types for LAS 3.0 data sections."""

    SPACE = "SPACE"
    TAB = "TAB"
    COMMA = "COMMA"


class DataFormatType(Enum):
    """Data format types for LAS 3.0 curve definitions.

    F: Float
    E: Scientific notation (0.00E00)
    S: String
    A: Array (with time offset)
    """

    FLOAT = "F"
    SCIENTIFIC = "E"
    STRING = "S"
    ARRAY = "A"


@dataclass
class VersionSection:
    """LAS Version Information section (~V).

    Supports LAS 1.2, 2.0, and 3.0 formats.
    """

    vers: str = "2.0"
    wrap: str = "NO"
    dlm: str = "SPACE"  # LAS 3.0: SPACE, TAB, or COMMA

    def to_dict(self) -> dict[str, str]:
        """Convert to legacy dict format for backward compatibility."""
        return {
            "VERS": self.vers,
            "WRAP": self.wrap,
            "DLM": self.dlm,
        }

    @property
    def is_las30(self) -> bool:
        """Check if this is a LAS 3.0 file."""
        return self.vers.startswith("3")

    @property
    def delimiter_char(self) -> str:
        """Get the actual delimiter character for data parsing."""
        delimiter_map = {
            "SPACE": " ",
            "TAB": "\t",
            "COMMA": ",",
        }
        return delimiter_map.get(self.dlm.upper(), " ")


@dataclass
class WellSection:
    """LAS Well Information section (~W).

    All values are stored as strings to match original pylasdev behavior.
    """

    entries: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        """Convert to legacy dict format."""
        return dict(self.entries)

    def __getitem__(self, key: str) -> str:
        return self.entries[key]

    def __setitem__(self, key: str, value: str) -> None:
        self.entries[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def get(self, key: str, default: str = "") -> str:
        return self.entries.get(key, default)


@dataclass
class ArrayElementInfo:
    """LAS 3.0 array element metadata for curves.

    Captures information like {A:0}, {A:5}, etc. from LAS 3.0 curves.
    """

    base_name: str = ""  # Base mnemonic without index (e.g., "NMR")
    index: int = 0  # Array index (e.g., 1, 2, 3)
    time_offset: float | None = None  # Time offset from first element (e.g., 0, 5, 10 ms)


@dataclass
class CurveDefinition:
    """Single curve definition from ~C section.

    Supports LAS 1.2, 2.0, and 3.0 formats including array notation.
    """

    mnemonic: str
    unit: str = ""
    api_code: str = ""
    description: str = ""
    original_mnemonic: str = ""  # Pre-normalization name

    # LAS 3.0 specific fields
    data_format: str = ""  # F, E, S, or A (from {F}, {E}, {S}, {A:x})
    array_info: ArrayElementInfo | None = None  # For array curves like NMR[1]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mnemonic": self.mnemonic,
            "unit": self.unit,
            "api_code": self.api_code,
            "description": self.description,
        }
        if self.data_format:
            result["data_format"] = self.data_format
        if self.array_info:
            result["array_info"] = {
                "base_name": self.array_info.base_name,
                "index": self.array_info.index,
                "time_offset": self.array_info.time_offset,
            }
        return result

    @property
    def is_array_element(self) -> bool:
        """Check if this curve is part of an array."""
        return self.array_info is not None

    @property
    def base_mnemonic(self) -> str:
        """Get base mnemonic for array curves, or regular mnemonic otherwise."""
        if self.array_info:
            return self.array_info.base_name
        return self.mnemonic


@dataclass
class ParameterZone:
    """LAS 3.0 zone association for parameters.

    Parameters can be associated with zones via pipe notation: | Zone[1]
    """

    zone_name: str = ""
    zone_index: int | None = None


@dataclass
class ParameterEntry:
    """Single parameter entry from ~P section.

    Supports LAS 1.2, 2.0, and 3.0 formats including array notation and zones.
    """

    mnemonic: str
    unit: str = ""
    value: str = ""
    description: str = ""

    # LAS 3.0 specific fields
    array_index: int | None = None  # For RUN[1], RUN[2], etc.
    zone: ParameterZone | None = None  # Zone association via pipe notation

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mnemonic": self.mnemonic,
            "unit": self.unit,
            "value": self.value,
            "description": self.description,
        }
        if self.array_index is not None:
            result["array_index"] = self.array_index
        if self.zone:
            result["zone"] = {
                "zone_name": self.zone.zone_name,
                "zone_index": self.zone.zone_index,
            }
        return result

    @property
    def base_mnemonic(self) -> str:
        """Get base mnemonic without array index."""
        if self.array_index is not None and "[" in self.mnemonic:
            return self.mnemonic.split("[")[0]
        return self.mnemonic


@dataclass
class DataSection:
    """LAS 3.0 data section (~A).

    LAS 3.0 can have multiple data sections, each potentially with different
    curve sets or depth ranges.
    """

    name: str = ""  # Section name from ~A line (e.g., "ASCII" or custom name)
    curves_order: list[str] = field(default_factory=list)
    data: dict[str, NDArray[np.float64]] = field(default_factory=dict)


@dataclass
class LASFile:
    """Complete LAS file data structure.

    Supports LAS 1.2, 2.0, and 3.0 formats.
    """

    version: VersionSection = field(default_factory=VersionSection)
    well: WellSection = field(default_factory=WellSection)
    curves: list[CurveDefinition] = field(default_factory=list)
    parameters: list[ParameterEntry] = field(default_factory=list)
    other: str = ""
    logs: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    curves_order: list[str] = field(default_factory=list)
    source_file: str = ""
    encoding: str = "utf-8"

    # LAS 3.0 specific fields
    data_sections: list[DataSection] = field(default_factory=list)
    string_data: dict[str, NDArray[np.str_]] = field(default_factory=dict)  # For {S} format curves

    def to_dict(self) -> dict[str, Any]:
        """Convert to legacy dict format for backward compatibility."""
        params_dict: dict[str, str] = {}
        for p in self.parameters:
            params_dict[p.mnemonic] = p.value

        return {
            "version": self.version.to_dict(),
            "well": self.well.to_dict(),
            "parameters": params_dict,
            "logs": {k: v.copy() for k, v in self.logs.items()},
            "curves_order": list(self.curves_order),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LASFile:
        """Create LASFile from legacy dict format."""
        las_file = cls()

        version = data.get("version", {})
        las_file.version = VersionSection(
            vers=str(version.get("VERS", "2.0")),
            wrap=str(version.get("WRAP", "NO")),
            dlm=str(version.get("DLM", "SPACE")),
        )

        well = data.get("well", {})
        for key, value in well.items():
            las_file.well[key] = str(value)

        curves_order = data.get("curves_order", [])
        las_file.curves_order = list(curves_order)
        for curve_name in curves_order:
            las_file.curves.append(CurveDefinition(mnemonic=curve_name))

        params = data.get("parameters", {})
        for mnemonic, value in params.items():
            las_file.parameters.append(
                ParameterEntry(
                    mnemonic=mnemonic,
                    value=str(value),
                )
            )

        logs = data.get("logs", {})
        for name, arr in logs.items():
            las_file.logs[name] = np.array(arr, dtype=np.float64)

        return las_file

    @property
    def is_las30(self) -> bool:
        """Check if this is a LAS 3.0 file."""
        return self.version.is_las30

    def get_curve_by_mnemonic(self, mnemonic: str) -> CurveDefinition | None:
        """Get curve definition by mnemonic (supports base name for arrays)."""
        for curve in self.curves:
            if curve.mnemonic == mnemonic or curve.base_mnemonic == mnemonic:
                return curve
        return None

    def get_parameters_by_zone(self, zone_name: str) -> list[ParameterEntry]:
        """Get all parameters associated with a zone (LAS 3.0)."""
        return [p for p in self.parameters if p.zone and p.zone.zone_name == zone_name]

    def get_array_curves(self, base_name: str) -> list[CurveDefinition]:
        """Get all array elements for a base curve name (LAS 3.0)."""
        return [c for c in self.curves if c.array_info and c.array_info.base_name == base_name]


@dataclass
class DevFile:
    """DEV (deviation survey) file data structure."""

    columns: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    column_order: list[str] = field(default_factory=list)
    source_file: str = ""
    encoding: str = "utf-8"

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        """Convert to legacy dict format."""
        return {k: v.copy() for k, v in self.columns.items()}
