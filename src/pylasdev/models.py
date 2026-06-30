"""Data models for LAS file structures.

Supports LAS 1.2, 2.0, and 3.0 formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _safe_str(value: Any, default: str = "") -> str:
    """Convert value to str, returning *default* when *value* is None.

    Prevents ``str(None)`` → ``"None"`` in dict roundtrip paths.
    """
    if value is None:
        return default
    return str(value)


def _create_parameter_entry(param_dict: dict[str, Any]) -> ParameterEntry:
    """Create a ParameterEntry from a dictionary, handling optional zone info.

    Extracted from LASFile.from_dict to avoid duplicated construction logic
    across the parameter_details and params list branches.
    """
    zone = None
    if "zone" in param_dict:
        zone = ParameterZone(
            zone_name=param_dict["zone"].get("zone_name", ""),
            zone_index=param_dict["zone"].get("zone_index"),
        )
    return ParameterEntry(
        mnemonic=_safe_str(param_dict.get("mnemonic", "")),
        unit=_safe_str(param_dict.get("unit", "")),
        value=_safe_str(param_dict.get("value", "")),
        description=_safe_str(param_dict.get("description", "")),
        array_index=param_dict.get("array_index"),
        zone=zone,
    )


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
        """Check if this is a LAS 3.0 file.

        Uses string prefix matching ('3' prefix on the version string).
        This is a deliberate design choice: LAS 3.x versions (3.0, 3.1,
        etc.) all share the same structural features. A version string
        like '3.1beta' or '3.0-draft' will match.
        """
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
    units: dict[str, str] = field(default_factory=dict)

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
        if self.original_mnemonic:
            result["original_mnemonic"] = self.original_mnemonic
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


@dataclass(eq=False)
class DataSection:
    """LAS 3.0 data section (~A).

    LAS 3.0 can have multiple data sections, each potentially with different
    curve sets or depth ranges.
    """

    name: str = ""  # Section name from ~A line (e.g., "ASCII" or custom name)
    section_type: str = "LOG_DATA"  # Section type: LOG_DATA, CORE_DATA, DRILLING_DATA, etc.
    curves_order: list[str] = field(default_factory=list)
    data: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    string_data: dict[str, NDArray[np.str_]] = field(default_factory=dict)  # For {S} format curves
    section_curves: list[CurveDefinition] = field(default_factory=list)  # Per-section curve definitions

    def to_dict(self) -> dict[str, Any]:
        """Convert DataSection to dict for serialization."""
        return {
            "name": self.name,
            "section_type": self.section_type,
            "curves_order": list(self.curves_order),
            "data": {k: v.copy() for k, v in self.data.items()},
            "string_data": {k: v.copy() for k, v in self.string_data.items()},
            "section_curves": [c.to_dict() for c in self.section_curves],
        }


@dataclass(eq=False)
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
        """Convert to legacy dict format for backward compatibility.

        Returns a dict with both legacy ``parameters`` (``{mnemonic: value}``
        dict) and ``parameter_details`` (list of full ParameterEntry dicts)
        to preserve backward compatibility while exposing LAS 3.0 metadata.
        ``logs`` arrays are defensively copied.
        """
        params_dict: dict[str, str] = {}
        for p in self.parameters:
            params_dict[p.mnemonic] = p.value

        return {
            "version": self.version.to_dict(),
            "well": self.well.to_dict(),
            "well_units": dict(self.well.units) if self.well.units else {},
            "parameters": params_dict,
            "parameter_details": [p.to_dict() for p in self.parameters],
            "curves": [c.to_dict() for c in self.curves],
            # Defensive copy prevents callers from mutating internal arrays
            # through the returned dict (numpy arrays are mutable views).
            "logs": {k: v.copy() for k, v in self.logs.items()},
            "curves_order": list(self.curves_order),
            "other": self.other,
            "data_sections": [ds.to_dict() for ds in self.data_sections],
            "string_data": {
                k: v.copy() for k, v in self.string_data.items()
            },  # same defensive copy as logs above
            "encoding": self.encoding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LASFile:
        """Create LASFile from dict format.

        Handles multiple format variants inherently:
        - Legacy flat dict (curves as string lists, params as {name: value} dict)
        - Detailed dict with CurveDefinition metadata (unit, api_code, description)
        - LAS 3.0 dict with array_info, data_format, data_sections, string_data
        - Mixed formats from roundtrip serialization

        The method is naturally long due to covering all these variants in a
        single backwards-compatible code path.
        """
        las_file = cls()

        version = data.get("version") or {}
        las_file.version = VersionSection(
            vers=_safe_str(version.get("VERS"), "2.0"),
            wrap=_safe_str(version.get("WRAP"), "NO"),
            dlm=_safe_str(version.get("DLM"), "SPACE"),
        )

        well = data.get("well") or {}
        for key, value in well.items():
            las_file.well[key] = _safe_str(value)
        # Restore well units if present (from v1.7+ roundtrip data)
        well_units = data.get("well_units") or {}
        for key, unit in well_units.items():
            las_file.well.units[key] = _safe_str(unit)

        curves_order = data.get("curves_order", [])
        las_file.curves_order = list(curves_order)

        # Restore curve metadata if available (new format), otherwise create minimal CurveDefinition
        curves_data = data.get("curves", [])
        if curves_data and isinstance(curves_data, list) and isinstance(curves_data[0], dict):
            for curve_dict in curves_data:
                array_info = None
                if "array_info" in curve_dict:
                    ai = curve_dict["array_info"]
                    array_info = ArrayElementInfo(
                        base_name=ai.get("base_name", ""),
                        index=ai.get("index", 0),
                        time_offset=ai.get("time_offset"),
                    )
                las_file.curves.append(
                    CurveDefinition(
                        mnemonic=curve_dict.get("mnemonic", ""),
                        unit=curve_dict.get("unit", ""),
                        api_code=curve_dict.get("api_code", ""),
                        description=curve_dict.get("description", ""),
                        original_mnemonic=curve_dict.get("original_mnemonic", ""),
                        data_format=curve_dict.get("data_format", ""),
                        array_info=array_info,
                    )
                )
        else:
            # Legacy format: only curve names available
            for curve_name in curves_order:
                las_file.curves.append(CurveDefinition(mnemonic=curve_name))

        params = data.get("parameters") or []
        if isinstance(params, dict):
            # Legacy format: {mnemonic: value}
            # Check for parameter_details first to preserve full metadata
            # on roundtrip (e.g. array_index, zone, unit, description).
            param_details = data.get("parameter_details")
            if param_details and isinstance(param_details, list):
                for param_dict in param_details:
                    las_file.parameters.append(_create_parameter_entry(param_dict))
            else:
                # Pure legacy: only params dict, no details available
                for mnemonic, value in params.items():
                    las_file.parameters.append(
                        ParameterEntry(mnemonic=mnemonic, value=_safe_str(value))
                    )
        elif isinstance(params, list):
            # New format: [{"mnemonic": ..., "value": ..., ...}, ...]
            for param_dict in params:
                las_file.parameters.append(_create_parameter_entry(param_dict))

        las_file.other = _safe_str(data.get("other"), "")
        las_file.encoding = _safe_str(data.get("encoding"), "utf-8")
        las_file.source_file = _safe_str(data.get("source_file"), "")

        # Restore LAS 3.0 data sections
        ds_data = data.get("data_sections", [])
        for ds_dict in ds_data:
            ds_string_data = {}
            for name, arr in ds_dict.get("string_data", {}).items():
                ds_string_data[name] = np.array(arr, dtype=np.str_)
            ds_section_curves = []
            for sc_dict in ds_dict.get("section_curves", []):
                sc_array_info = None
                if "array_info" in sc_dict:
                    ai = sc_dict["array_info"]
                    sc_array_info = ArrayElementInfo(
                        base_name=ai.get("base_name", ""),
                        index=ai.get("index", 0),
                        time_offset=ai.get("time_offset"),
                    )
                ds_section_curves.append(
                    CurveDefinition(
                        mnemonic=sc_dict.get("mnemonic", ""),
                        unit=sc_dict.get("unit", ""),
                        api_code=sc_dict.get("api_code", ""),
                        description=sc_dict.get("description", ""),
                        original_mnemonic=sc_dict.get("original_mnemonic", ""),
                        data_format=sc_dict.get("data_format", ""),
                        array_info=sc_array_info,
                    )
                )
            ds = DataSection(
                name=ds_dict.get("name", ""),
                section_type=ds_dict.get("section_type", "LOG_DATA"),
                curves_order=list(ds_dict.get("curves_order", [])),
                data={k: np.array(v, dtype=np.float64) for k, v in ds_dict.get("data", {}).items()},
                string_data=ds_string_data,
                section_curves=ds_section_curves,
            )
            las_file.data_sections.append(ds)

        # Restore LAS 3.0 string data (top-level, backward compat
        # with data serialized before string_data was moved to
        # per-section DataSection objects).
        sd = data.get("string_data", {})
        for name, arr in sd.items():
            las_file.string_data[name] = np.array(arr, dtype=np.str_)

        logs = data.get("logs", {})
        for name, arr in logs.items():
            try:
                las_file.logs[name] = np.array(arr, dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert log data for curve '{name}' to numeric array: {e}"
                ) from e

        return las_file

    @property
    def is_las30(self) -> bool:
        """Check if this is a LAS 3.0 file."""
        return self.version.is_las30

    def get_curve_by_mnemonic(self, mnemonic: str) -> CurveDefinition | None:
        """Get curve definition by mnemonic name.

        Searches the curve list for an exact mnemonic match. For LAS 3.0
        array curves (e.g., ``NMR[1]``), also matches on the base mnemonic
        (e.g., ``"NMR"``) so that callers can find array elements without
        knowing the exact index.

        Args:
            mnemonic: Curve mnemonic to look up (e.g., ``"GR"``, ``"NMR"``).

        Returns:
            Matching ``CurveDefinition`` if found, or ``None`` if no curve
            matches the given mnemonic.
        """
        for curve in self.curves:
            if curve.mnemonic == mnemonic or curve.base_mnemonic == mnemonic:
                return curve
        return None

    def get_array_curves(self, base_name: str) -> list[CurveDefinition]:
        """Get all array-element curves for a base name (LAS 3.0).

        LAS 3.0 files use array notation (e.g., ``NMR[1]``, ``NMR[2]``) to
        represent multi-element curves. This method collects all curve
        definitions that belong to the same logical array group.

        Args:
            base_name: Base mnemonic of the array (e.g., ``"NMR"``).

        Returns:
            List of ``CurveDefinition`` objects whose ``array_info.base_name``
            matches *base_name*. Returns an empty list if no array curves match.
        """
        return [c for c in self.curves if c.array_info and c.array_info.base_name == base_name]


@dataclass(eq=False)
class DevFile:
    """DEV (deviation survey) file data structure."""

    columns: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    column_order: list[str] = field(default_factory=list)
    source_file: str = ""
    encoding: str = "utf-8"

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        """Convert to legacy dict format."""
        return {k: v.copy() for k, v in self.columns.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DevFile:
        """Create DevFile from dict (reverse of to_dict).

        Args:
            data: Flat dict mapping column names to array-like values.

        Returns:
            DevFile with columns populated from the dict.
        """
        dev = cls()
        # Separate column arrays from metadata keys
        metadata_keys = {"encoding", "source_file", "column_order"}
        for key, value in data.items():
            if key in metadata_keys:
                if key == "encoding":
                    dev.encoding = _safe_str(value, "utf-8")
                elif key == "source_file":
                    dev.source_file = _safe_str(value)
                elif key == "column_order":
                    dev.column_order = list(value)
            else:
                dev.columns[key] = np.array(value, dtype=np.float64)
        # If column_order wasn't in the dict, infer from Python 3.7+ dict order
        if not dev.column_order:
            dev.column_order = list(dev.columns.keys())
        return dev
