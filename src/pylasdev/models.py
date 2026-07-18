"""Data models for LAS file structures.

Supports LAS 1.2, 2.0, and 3.0 formats.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
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
    if "zone" in param_dict and isinstance(param_dict["zone"], dict):
        _zone_index = param_dict["zone"].get("zone_index")
        # F-026: Type validation for zone_index — raw dict values
        # pass through without type checking, violating the int | None
        # contract on ParameterZone.zone_index.
        if _zone_index is not None and not isinstance(_zone_index, int):
            raise TypeError(
                f"zone_index: expected int or None, "
                f"got {type(_zone_index).__name__}"
            )
        zone = ParameterZone(
            zone_name=_safe_str(param_dict["zone"].get("zone_name")),
            zone_index=_zone_index,
        )
    elif "zone" in param_dict:
        # F-I2-M05: Non-dict "zone" value silently ignored before.
        # Emit a warning so callers know the zone metadata was dropped.
        warnings.warn(
            f"Ignoring non-dict 'zone' value of type "
            f"{type(param_dict['zone']).__name__} "
            f"for parameter '{param_dict.get('mnemonic', '?')}'",
            stacklevel=2,
        )
    _array_index = param_dict.get("array_index")
    # F-025: Type validation for array_index — raw dict values
    # pass through without type checking, violating the int | None
    # contract on ParameterEntry.array_index.
    if _array_index is not None and not isinstance(_array_index, int):
        raise TypeError(
            f"array_index: expected int or None, "
            f"got {type(_array_index).__name__}"
        )
    return ParameterEntry(
        mnemonic=_safe_str(param_dict.get("mnemonic", "")),
        unit=_safe_str(param_dict.get("unit", "")),
        value=_safe_str(param_dict.get("value", "")),
        description=_safe_str(param_dict.get("description", "")),
        data_format=_safe_str(param_dict.get("data_format"), ""),
        array_index=_array_index,
        zone=zone,
    )


def _validate_iterable_of_dicts(
    items: Any,
    context_name: str,
) -> list[dict[str, Any]]:
    """Validate that *items* is a list and every element is a dict.

    Used by ``LASFile.from_dict`` to consolidate list-of-dicts validation
    across multiple processing paths that previously used inconsistent
    isinstance/error patterns (F-012, F-013, F-058).
    """
    if not isinstance(items, list):
        raise TypeError(
            f"{context_name} must be a list, got {type(items).__name__}"
        )
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(
                f"{context_name}[{i}] must be a dict, got {type(item).__name__}"
            )
    return items


def _resolve_dict_entry(
    data: dict[str, Any],
    key: str,
    expected_type: type[Any] | tuple[type[Any], ...],
    default_factory: Callable[[], Any],
) -> Any:
    """Extract *key* from *data* with type validation.

    Returns *default_factory()* when *key* is missing or its value is ``None``.
    Raises ``TypeError`` when the value is present but not an instance of
    *expected_type*.  Replaces the ``data.get(key) or default`` pattern used
    previously, which confused falsy values with missing keys.

    This helper eliminates the truthiness-based dispatch class of bugs that
    prior fix rounds repeatedly attempted (and failed) to eliminate.
    """
    value = data.get(key)
    if value is None:
        return default_factory()
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{key}: expected {expected_type}, got {type(value).__name__}"
        )
    return value


@dataclass
class VersionSection:
    """LAS Version Information section (~V).

    Supports LAS 1.2, 2.0, and 3.0 formats.
    """

    vers: str = "2.0"
    wrap: str = "NO"
    dlm: str = "SPACE"  # LAS 2.0+: SPACE, TAB, or COMMA

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
    descriptions: dict[str, str] = field(
        default_factory=dict
    )  # CWLS description text for well fields

    def to_dict(self) -> dict[str, Any]:
        """Convert to legacy dict format, including non-empty units and descriptions."""
        result: dict[str, Any] = dict(self.entries)
        if self.units:
            result["units"] = dict(self.units)
        if self.descriptions:
            result["descriptions"] = dict(self.descriptions)
        return result

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
    data_format: str = ""  # Format specifier ({F}, {E}, etc.) from parser

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
        if self.data_format:
            result["data_format"] = self.data_format
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
    """~A data section (LAS 1.2/2.0) and LAS 3.0 typed data sections.

    In LAS 1.2/2.0 files, the data section is a single ``~A`` block.
    LAS 3.0 can have multiple typed data sections (``~ASCII``, ``~Log_Data``,
    ``~Core_Data``, etc.), each potentially with different curve sets or
    depth ranges.
    """

    name: str = ""  # Section name from ~A line (e.g., "ASCII" or custom name)
    section_type: str = "LOG_DATA"  # Section type: LOG_DATA, CORE_DATA, DRILLING_DATA, etc.
    curves_order: list[str] = field(default_factory=list)
    data: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    string_data: dict[str, NDArray[np.str_]] = field(default_factory=dict)  # For {S} format curves
    section_curves: list[CurveDefinition] = field(
        default_factory=list
    )  # Per-section curve definitions

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

        Note: The defensive array copies (``v.copy()`` on every log/string
        array) allocate new arrays.  In low-memory environments this can
        raise ``MemoryError``.  Consider using the ``LASFile`` dataclass
        directly to avoid the copy overhead.
        """
        params_dict: dict[str, str] = {}
        for p in self.parameters:
            params_dict[p.mnemonic] = p.value

        return {
            "version": self.version.to_dict(),
            "well": dict(self.well.entries),
            "well_units": dict(self.well.units) if self.well.units else {},
            "well_descriptions": dict(self.well.descriptions) if self.well.descriptions else {},
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
            "source_file": self.source_file,
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
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")

        # F-06: Deferred imports to avoid circular dependencies
        # (models.py ← parser.py/data_reader.py which import from models.py).
        from .data_reader import MAX_CURVES, MAX_DATA_LINES, MAX_TOTAL_ELEMENTS
        from .exceptions import LASDataError
        from .parser import MAX_DATA_SECTIONS, MAX_OTHER_LINES, MAX_PARAMETERS

        # F-057: Bound well-related iterables to match the parser's
        # MAX_DEFERRED_WELL_ENTRIES guard on the well re-processing path.
        MAX_WELL_ENTRIES = MAX_PARAMETERS

        try:
            las_file = cls()

            version = _resolve_dict_entry(data, "version", dict, dict)
            las_file.version = VersionSection(
                vers=_safe_str(version.get("VERS"), "2.0"),
                wrap=_safe_str(version.get("WRAP"), "NO"),
                dlm=_safe_str(version.get("DLM"), "SPACE"),
            )

            well = _resolve_dict_entry(data, "well", dict, dict)
            # F-057: Bound check — parser has MAX_DEFERRED_WELL_ENTRIES on its
            # well re-processing path; from_dict previously had no bound at all.
            if len(well) > MAX_WELL_ENTRIES:
                raise ValueError(
                    f"Number of well entries ({len(well)}) exceeds maximum "
                    f"allowed ({MAX_WELL_ENTRIES})"
                )
            for key, value in well.items():
                las_file.well[key] = _safe_str(value)
            # Restore well units if present (from v1.7+ roundtrip data)
            well_units = _resolve_dict_entry(data, "well_units", dict, dict)
            if len(well_units) > MAX_WELL_ENTRIES:
                raise ValueError(
                    f"Number of well unit entries ({len(well_units)}) exceeds "
                    f"maximum allowed ({MAX_WELL_ENTRIES})"
                )
            for key, unit in well_units.items():
                las_file.well.units[key] = _safe_str(unit)

            # Restore well descriptions if present (from v1.8+ roundtrip data)
            well_descriptions = _resolve_dict_entry(data, "well_descriptions", dict, dict)
            if len(well_descriptions) > MAX_WELL_ENTRIES:
                raise ValueError(
                    f"Number of well description entries ({len(well_descriptions)}) "
                    f"exceeds maximum allowed ({MAX_WELL_ENTRIES})"
                )
            for key, desc in well_descriptions.items():
                las_file.well.descriptions[key] = _safe_str(desc)

            curves_order = data.get("curves_order", [])
            # F-21: Guard against non-list iterables.  list(string) silently
            # creates a list of single characters (e.g. "DEPT,DT,GR" → 10
            # single-char mnemonics), passing all downstream cross-validation
            # because both curves_order and curves get corrupted the same way.
            if curves_order is None:
                curves_order = []
            elif isinstance(curves_order, str):
                raise ValueError(
                    f"curves_order must be a list, got str: "
                    f"{curves_order!r}"
                )
            elif not isinstance(curves_order, Iterable):
                # F-M02: Non-iterable non-str types (int, float, bool) crash
                # at list() with TypeError.  Provide a clear error instead.
                raise TypeError(
                    f"curves_order must be an iterable, "
                    f"got {type(curves_order).__name__}"
                )
            las_file.curves_order = list(curves_order)

            # Restore curve metadata if available (new format), otherwise create minimal CurveDefinition
            # F-16: Use _resolve_dict_entry — data.get("curves", []) returns
            # None when "curves" exists with value None, bypassing the default
            # and crashing at len(None).  Same pattern at 6 other sites below.
            curves_data = _resolve_dict_entry(data, "curves", list, list)
            # F-06: Resource-exhaustion guard — match parser's MAX_CURVES check.
            if len(curves_data) > MAX_CURVES:
                raise ValueError(
                    f"Number of curves ({len(curves_data)}) exceeds maximum "
                    f"allowed ({MAX_CURVES})"
                )
            if isinstance(curves_data, list):
                if curves_data:
                    # F-012: Validate every element is a dict, not just curves_data[0].
                    # Non-dict elements crash at curve_dict.get() downstream.
                    _validate_iterable_of_dicts(curves_data, "curves")
                    for curve_dict in curves_data:
                        array_info = None
                        if "array_info" in curve_dict and isinstance(curve_dict["array_info"], dict):
                            ai = curve_dict["array_info"]
                            array_info = ArrayElementInfo(
                                base_name=_safe_str(ai.get("base_name")),
                                index=_resolve_dict_entry(ai, "index", int, lambda: 0),
                                # F2-002: Validate time_offset — int(offset) in
                                # writer.py crashes on non-numeric values.
                                time_offset=_resolve_dict_entry(ai, "time_offset", (int, float), lambda: None),
                            )
                        las_file.curves.append(
                            CurveDefinition(
                                mnemonic=_safe_str(curve_dict.get("mnemonic", "")),
                                unit=_safe_str(curve_dict.get("unit", "")),
                                api_code=_safe_str(curve_dict.get("api_code", "")),
                                description=_safe_str(curve_dict.get("description", "")),
                                original_mnemonic=_safe_str(curve_dict.get("original_mnemonic", "")),
                                data_format=_safe_str(curve_dict.get("data_format", "")),
                                array_info=array_info,
                            )
                        )
            elif curves_data:
                # F-I2-M01: This branch is provably unreachable with the current
                # _resolve_dict_entry(data, "curves", list, list) call above —
                # it always returns a list or raises TypeError.  Retained as
                # defensive design in case _resolve_dict_entry is ever relaxed.
                raise TypeError(
                    f"curves must be a list, got {type(curves_data).__name__}"
                )

            # Legacy format: only curve names available
            # (reached when curves_data is empty list or falsy)
            if not las_file.curves:
                # F-06: Resource-exhaustion guard for legacy curves_order path.
                if len(curves_order) > MAX_CURVES:
                    raise ValueError(
                        f"Number of curves ({len(curves_order)}) exceeds maximum "
                        f"allowed ({MAX_CURVES})"
                    )
                for curve_name in curves_order:
                    las_file.curves.append(CurveDefinition(mnemonic=curve_name))

            # F-02: Cross-validate curves_order and curves for consistency.
            # Both are built from separate dict keys independently; a mismatched
            # input dict can produce silently inconsistent state.  from_dict is
            # called on untrusted data via write_las_file (public API).
            _curve_count = len(las_file.curves)
            if len(las_file.curves_order) != _curve_count:
                raise ValueError(
                    f"curves_order length ({len(las_file.curves_order)}) does not "
                    f"match curves length ({_curve_count})"
                )
            for _i, (_order_name, _curve) in enumerate(
                zip(las_file.curves_order, las_file.curves, strict=True)
            ):
                if _order_name != _curve.mnemonic:
                    raise ValueError(
                        f"curves_order[{_i}] = {_order_name!r} does not match "
                        f"curves[{_i}].mnemonic = {_curve.mnemonic!r}"
                    )

            params = _resolve_dict_entry(data, "parameters", (dict, list), list)
            # F-06: Resource-exhaustion guard for parameters.
            _param_count = len(params)
            if _param_count > MAX_PARAMETERS:
                raise ValueError(
                    f"Number of parameters ({_param_count}) exceeds maximum "
                    f"allowed ({MAX_PARAMETERS})"
                )
            if isinstance(params, dict):
                # Legacy format: {mnemonic: value}
                # Check for parameter_details first to preserve full metadata
                # on roundtrip (e.g. array_index, zone, unit, description).
                param_details = data.get("parameter_details")
                if param_details:
                    if len(param_details) > MAX_PARAMETERS:
                        raise ValueError(
                            f"Number of parameter details ({len(param_details)}) exceeds maximum "
                            f"allowed ({MAX_PARAMETERS})"
                        )
                    # F2-25 + F-058 consistency: Validate as list-of-dicts via
                    # shared helper (same pattern as curves_data, section_curves,
                    # and data_sections).  Raises TypeError for non-dict elements.
                    param_details = _validate_iterable_of_dicts(param_details, "parameter_details")
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
                # F2-25 consistency: Shared helper validates every element is a dict
                # (previously checked inline, same as param_details and data_sections).
                _validate_iterable_of_dicts(params, "parameters")
                for param_dict in params:
                    las_file.parameters.append(_create_parameter_entry(param_dict))
            else:
                # F-I2-M02: This branch is provably unreachable with the current
                # _resolve_dict_entry(data, "parameters", (dict, list), list) call
                # above — it always returns a dict or list, or raises TypeError.
                # Retained as defensive design in case _resolve_dict_entry is ever relaxed.
                raise TypeError(
                    f"parameters must be a dict or list, got {type(params).__name__}"
                )

            # F-06: Resource-exhaustion guard for other section content.
            _other_raw = str(data.get("other", ""))
            if len(_other_raw) > MAX_OTHER_LINES:
                raise ValueError(
                    f"Other section length ({len(_other_raw)} chars) exceeds "
                    f"maximum allowed ({MAX_OTHER_LINES})"
                )
            las_file.other = _safe_str(data.get("other"), "")
            las_file.encoding = _safe_str(data.get("encoding"), "utf-8")
            las_file.source_file = _safe_str(data.get("source_file"), "")

            # Restore LAS 3.0 data sections
            ds_data = _resolve_dict_entry(data, "data_sections", list, list)
            # F-06: Resource-exhaustion guard for data sections.
            if len(ds_data) > MAX_DATA_SECTIONS:
                raise ValueError(
                    f"Number of data sections ({len(ds_data)}) exceeds maximum "
                    f"allowed ({MAX_DATA_SECTIONS})"
                )
            # F-058: Validate every element is a dict.  Previously non-dict
            # elements were silently skipped with ``continue`` while two sibling
            # paths (parameter_details and params) both raised TypeError.
            # Using the shared helper also adds a missing list-type check.
            ds_data = _validate_iterable_of_dicts(ds_data, "data_sections")
            for ds_dict in ds_data:
                ds_string_data = {}
                _ds_string_raw = _resolve_dict_entry(ds_dict, "string_data", dict, dict)
                # F-24: Per-section string_data entry count guard.  Every other
                # iterable dict in from_dict() has a count guard (curves_data →
                # MAX_CURVES, ds_data → MAX_CURVES, logs → MAX_CURVES, etc.);
                # string_data was the sole unguarded iterable.
                if len(_ds_string_raw) > MAX_CURVES:
                    ds_name = ds_dict.get("name", "<unknown>")
                    raise ValueError(
                        f"Number of string data curves ({len(_ds_string_raw)}) "
                        f"in section '{ds_name}' exceeds maximum allowed ({MAX_CURVES})"
                    )
                for name, arr in _ds_string_raw.items():
                    # F-22: Guard against None — np.array(None, dtype=np.str_)
                    # creates a 0-d array containing the string "None", which
                    # then passes the downstream len() check as a silent bug.
                    if arr is None:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"String data for curve '{name}' in section "
                            f"'{ds_name}' is None"
                        )
                    ds_string_data[name] = np.atleast_1d(np.array(arr, dtype=np.str_))
                    # F-03: Per-array size guard for data section string_data arrays.
                    if len(ds_string_data[name]) > MAX_DATA_LINES:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"String data array length ({len(ds_string_data[name])}) "
                            f"for '{name}' in section '{ds_name}' exceeds maximum "
                            f"allowed ({MAX_DATA_LINES})"
                        )
                # F-91: Per-section string_data total element count guard.
                # ds_data has this check (below) and top-level las_file.logs
                # has it; string_data was the sole unguarded path.  Many
                # short per-array entries can pass the entry-count and
                # per-array length guards without triggering them — only a
                # product check catches this class of allocation DoS.
                if ds_string_data:
                    _sds_rows = max(len(arr) for arr in ds_string_data.values())
                    _sds_total = len(ds_string_data) * _sds_rows
                    if _sds_total > MAX_TOTAL_ELEMENTS:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Total string data allocation in section '{ds_name}' "
                            f"({len(ds_string_data)} curves x {_sds_rows} rows = "
                            f"{_sds_total} elements) exceeds maximum allowed "
                            f"({MAX_TOTAL_ELEMENTS})"
                        )
                ds_section_curves = []
                _sc_raw = _resolve_dict_entry(ds_dict, "section_curves", list, list)
                # F-19: Resource-exhaustion guard for section_curves — match
                # the same MAX_CURVES check used for top-level curves_data and
                # curves_order.  The parser path has this guard (parser.py:1344);
                # from_dict() was the sole unguarded path.
                if len(_sc_raw) > MAX_CURVES:
                    ds_name = ds_dict.get("name", "<unknown>")
                    raise ValueError(
                        f"Number of section curves ({len(_sc_raw)}) in section "
                        f"'{ds_name}' exceeds maximum allowed ({MAX_CURVES})"
                    )
                # F-013: Validate every element is a dict.  Zero isinstance
                # check on _sc_raw elements previously — non-dict elements
                # crash at "array_info" in sc_dict / sc_dict.get().
                _sc_raw = _validate_iterable_of_dicts(_sc_raw, "section_curves")
                for sc_dict in _sc_raw:
                    sc_array_info = None
                    if "array_info" in sc_dict and isinstance(sc_dict["array_info"], dict):
                        ai = sc_dict["array_info"]
                        sc_array_info = ArrayElementInfo(
                            base_name=_safe_str(ai.get("base_name")),
                            index=_resolve_dict_entry(ai, "index", int, lambda: 0),
                            # F2-002: Validate time_offset — int(offset) in
                            # writer.py crashes on non-numeric values.
                            time_offset=_resolve_dict_entry(ai, "time_offset", (int, float), lambda: None),
                        )
                    ds_section_curves.append(
                        CurveDefinition(
                            mnemonic=_safe_str(sc_dict.get("mnemonic", "")),
                            unit=_safe_str(sc_dict.get("unit", "")),
                            api_code=_safe_str(sc_dict.get("api_code", "")),
                            description=_safe_str(sc_dict.get("description", "")),
                            original_mnemonic=_safe_str(sc_dict.get("original_mnemonic", "")),
                            data_format=_safe_str(sc_dict.get("data_format", "")),
                            array_info=sc_array_info,
                        )
                    )
                ds_data_raw = _resolve_dict_entry(ds_dict, "data", dict, dict)
                # F2-21: Per-section entry count guard.  Outer MAX_DATA_SECTIONS
                # guards section count; per-array MAX_DATA_LINES guards element
                # count.  Per-section curve entry count was unguarded — 1 section
                # x 200K single-element arrays passes all existing guards.
                if len(ds_data_raw) > MAX_CURVES:
                    ds_name = ds_dict.get("name", "<unknown>")
                    raise ValueError(
                        f"Number of data curves ({len(ds_data_raw)}) in section "
                        f"'{ds_name}' exceeds maximum allowed ({MAX_CURVES})"
                    )
                ds_data = {}
                for k, v in ds_data_raw.items():
                    # F-I2-M02: None guard — np.array(None, dtype=np.float64)
                    # silently produces nan.  String paths already reject None
                    # (lines 603-608, 785-788); numeric paths lacked this guard.
                    if v is None:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Numeric data for curve '{k}' in section "
                            f"'{ds_name}' is None"
                        )
                    try:
                        ds_data[k] = np.atleast_1d(np.array(v, dtype=np.float64))
                    except (ValueError, TypeError) as e:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Cannot convert data for section '{ds_name}', curve '{k}': {e}"
                        ) from e
                    # F-M02: Per-array size guard for data section arrays.
                    if len(ds_data[k]) > MAX_DATA_LINES:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Array length ({len(ds_data[k])}) for curve '{k}' in "
                            f"section '{ds_name}' exceeds maximum allowed "
                            f"({MAX_DATA_LINES})"
                        )
                # F-20: Per-section total element count guard — only top-level
                # las_file.logs had this check (below); data_section arrays were
                # unguarded.  Attack: 1,000 sections x 1 curve x 10M lines =
                # 80 GB, passing all per-array and per-section count guards.
                if ds_data:
                    _ds_rows = max(len(arr) for arr in ds_data.values())
                    _ds_total = len(ds_data) * _ds_rows
                    if _ds_total > MAX_TOTAL_ELEMENTS:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Total allocation in section '{ds_name}' "
                            f"({len(ds_data)} curves x {_ds_rows} rows = "
                            f"{_ds_total} elements) exceeds maximum allowed "
                            f"({MAX_TOTAL_ELEMENTS})"
                        )
                _ds_curves_order = ds_dict.get("curves_order", [])
                # F2-22: Guard against non-list iterables for per-section
                # curves_order — same bug as top-level F-21.
                if _ds_curves_order is None:
                    _ds_curves_order = []
                elif isinstance(_ds_curves_order, str):
                    ds_name = ds_dict.get("name", "<unknown>")
                    raise ValueError(
                        f"curves_order in section '{ds_name}' must be a list, "
                        f"got str: {_ds_curves_order!r}"
                    )
                # F-23: Cross-validate per-section curves_order with section_curves,
                # matching the top-level cross-validation pattern.  from_dict builds
                # these from independent dict keys; mismatched input would produce
                # silently inconsistent DataSection state.
                # Only validate when section_curves is non-empty — empty
                # section_curves means the section inherits curve definitions
                # from the top-level LASFile.curves (valid LAS 3.0 pattern).
                if ds_section_curves:
                    _sc_count = len(ds_section_curves)
                    if len(_ds_curves_order) != _sc_count:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"curves_order length ({len(_ds_curves_order)}) in section "
                            f"'{ds_name}' does not match section_curves length ({_sc_count})"
                        )
                    for _i, (_order_name, _sc) in enumerate(
                        zip(_ds_curves_order, ds_section_curves, strict=True)
                    ):
                        if _order_name != _sc.mnemonic:
                            ds_name = ds_dict.get("name", "<unknown>")
                            raise ValueError(
                                f"curves_order[{_i}] = {_order_name!r} in section "
                                f"'{ds_name}' does not match section_curves[{_i}].mnemonic "
                                f"= {_sc.mnemonic!r}"
                            )
                # F-14: Per-section curves_order bound.  When section_curves is
                # empty the length cross-validation above gates out, leaving
                # _ds_curves_order unbounded.  Every other iterable in from_dict
                # has a count guard — this was the sole unguarded path.
                if len(_ds_curves_order) > MAX_CURVES:
                    ds_name = ds_dict.get("name", "<unknown>")
                    raise ValueError(
                        f"Number of curves_order entries ({len(_ds_curves_order)}) "
                        f"in section '{ds_name}' exceeds maximum allowed "
                        f"({MAX_CURVES})"
                    )
                ds = DataSection(
                    name=_safe_str(ds_dict.get("name"), ""),
                    section_type=_safe_str(ds_dict.get("section_type"), "LOG_DATA"),
                    curves_order=list(_ds_curves_order),
                    data=ds_data,
                    string_data=ds_string_data,
                    section_curves=ds_section_curves,
                )
                las_file.data_sections.append(ds)
                # F-25: Cross-array length validation within this data section.
                # Inconsistent-length arrays produce silently corrupted output if
                # accepted — different curves with different sample counts in the
                # same data section represent invalid LAS data (see F-25).
                if len(ds.data) > 1:
                    _ds_len = {name: len(arr) for name, arr in ds.data.items()}
                    if len(set(_ds_len.values())) > 1:
                        raise ValueError(
                            f"Data section '{ds.name}' has inconsistent array "
                            f"lengths: {_ds_len}"
                        )
                # F-004: Cross-array length validation for string_data within
                # this data section (PRIOR_FIX_ATTEMPT in b47eea6/24c4f5c only
                # covered numeric data — string_data was missed on both levels).
                if len(ds.string_data) > 1:
                    _sds_len = {name: len(arr) for name, arr in ds.string_data.items()}
                    if len(set(_sds_len.values())) > 1:
                        raise ValueError(
                            f"Data section '{ds.name}' has inconsistent "
                            f"string_data array lengths: {_sds_len}"
                        )

            # F-I2-M17: Cross-validate data_sections with LAS version.
            # data_sections are a LAS 3.0 feature; non-LAS-3.0 files with
            # data_sections cause silent roundtrip data loss (writer emits
            # multi-section format, parser skips all data for non-LAS-3.0).
            if las_file.data_sections and not las_file.version.is_las30:
                warnings.warn(
                    f"data_sections are present but version is "
                    f"{las_file.version.vers!r} (not LAS 3.0). "
                    f"Data sections are only valid for LAS 3.0 files.",
                    stacklevel=2,
                )

            # Restore LAS 3.0 string data (top-level, backward compat
            # with data serialized before string_data was moved to
            # per-section DataSection objects).
            sd = _resolve_dict_entry(data, "string_data", dict, dict)
            # F-24: Top-level string_data entry count guard — same gap as
            # the per-section path fixed above.
            if len(sd) > MAX_CURVES:
                raise ValueError(
                    f"Number of string data curves ({len(sd)}) exceeds "
                    f"maximum allowed ({MAX_CURVES})"
                )
            for name, arr in sd.items():
                # F-22: Guard against None — same bug as per-section
                # string_data above.
                if arr is None:
                    raise ValueError(
                        f"String data for curve '{name}' is None"
                    )
                las_file.string_data[name] = np.atleast_1d(np.array(arr, dtype=np.str_))
                # F-M02: Per-array size guard for string_data arrays.
                if len(las_file.string_data[name]) > MAX_DATA_LINES:
                    raise ValueError(
                        f"Array length ({len(las_file.string_data[name])}) for "
                        f"string curve '{name}' exceeds maximum allowed "
                        f"({MAX_DATA_LINES})"
                    )
            # F-13: Top-level string_data total element count guard.
            # logs has this check (below); string_data was the sole unguarded
            # top-level path.  As with F-91 (per-section), many short arrays
            # can pass per-array and entry-count guards without a product check.
            if las_file.string_data:
                _str_rows = max(len(arr) for arr in las_file.string_data.values())
                _str_total = len(las_file.string_data) * _str_rows
                if _str_total > MAX_TOTAL_ELEMENTS:
                    raise ValueError(
                        f"Total string data allocation "
                        f"({len(las_file.string_data)} curves x {_str_rows} rows = "
                        f"{_str_total} elements) exceeds maximum allowed "
                        f"({MAX_TOTAL_ELEMENTS})"
                    )

            logs = _resolve_dict_entry(data, "logs", dict, dict)
            # F-06: Resource-exhaustion guard for logs.
            if len(logs) > MAX_CURVES:
                raise ValueError(
                    f"Number of log curves ({len(logs)}) exceeds maximum "
                    f"allowed ({MAX_CURVES})"
                )
            for name, arr in logs.items():
                # F-I2-M02: None guard — np.array(None, dtype=np.float64)
                # silently produces nan, consistent with string data guards.
                if arr is None:
                    raise ValueError(
                        f"Log data for curve '{name}' is None"
                    )
                try:
                    las_file.logs[name] = np.atleast_1d(np.array(arr, dtype=np.float64))
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Cannot convert log data for curve '{name}' to numeric array: {e}"
                    ) from e
                # F-M02: Per-array size guard for log arrays.
                if len(las_file.logs[name]) > MAX_DATA_LINES:
                    raise ValueError(
                        f"Array length ({len(las_file.logs[name])}) for log "
                        f"'{name}' exceeds maximum allowed ({MAX_DATA_LINES})"
                    )

            # F-011: Validate that log curve keys match curves_order exactly.
            # The length check above ensures count matches; this catches phantom
            # keys (extra curves in logs not in curves_order) and missing keys.
            # Only valid for legacy LAS 1.2/2.0 files where all curve data lives
            # in the logs dict.  LAS 3.0 files distribute curve data across
            # data_sections and string_data, so curves_order typically includes
            # curves whose data is in those sections, not in logs.
            if las_file.logs and not las_file.data_sections:
                _log_keys = set(las_file.logs.keys())
                _order_keys = set(las_file.curves_order)
                if _log_keys != _order_keys:
                    raise ValueError(
                        f"Log curve keys do not match curves_order. "
                        f"Extra keys: {_log_keys - _order_keys}, "
                        f"Missing keys: {_order_keys - _log_keys}"
                    )

            # F-25: Cross-array length validation for top-level log arrays.
            # Inconsistent-length arrays produce silently corrupted output if
            # accepted — different curves with different sample counts represent
            # invalid data (see F-25).
            if len(las_file.logs) > 1:
                _log_len = {name: len(arr) for name, arr in las_file.logs.items()}
                if len(set(_log_len.values())) > 1:
                    raise ValueError(
                        f"Log arrays have inconsistent lengths: {_log_len}"
                    )

            # F-004: Cross-array length validation for top-level string_data
            # arrays.  Prior fix rounds (b47eea6, 24c4f5c) added this check
            # for numeric logs but missed string_data at both the per-section
            # and top-level paths.
            if len(las_file.string_data) > 1:
                _str_len = {
                    name: len(arr) for name, arr in las_file.string_data.items()
                }
                if len(set(_str_len.values())) > 1:
                    raise ValueError(
                        f"String data arrays have inconsistent lengths: {_str_len}"
                    )

            # F-M02: Total element count guard across all log curves.
            if las_file.logs:
                _log_rows = max(len(arr) for arr in las_file.logs.values())
                _log_total = len(las_file.logs) * _log_rows
                if _log_total > MAX_TOTAL_ELEMENTS:
                    raise ValueError(
                        f"Total log allocation ({len(las_file.logs)} curves x "
                        f"{_log_rows} rows = {_log_total} elements) exceeds maximum "
                        f"allowed ({MAX_TOTAL_ELEMENTS})"
                    )

            return las_file
        except ValueError as e:
            raise LASDataError(str(e)) from e

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

    def to_dict(self) -> dict[str, Any]:
        """Convert to legacy dict format, including file metadata."""
        result: dict[str, Any] = {k: v.copy() for k, v in self.columns.items()}
        result["source_file"] = self.source_file
        result["encoding"] = self.encoding
        result["column_order"] = list(self.column_order)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DevFile:
        """Create DevFile from dict (reverse of to_dict).

        Args:
            data: Flat dict mapping column names to array-like values.

        Returns:
            DevFile with columns populated from the dict.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")

        from .data_reader import MAX_CURVES, MAX_DATA_LINES, MAX_TOTAL_ELEMENTS
        from .exceptions import LASDataError

        try:
            dev = cls()
            # Separate column arrays from metadata keys
            metadata_keys = {"encoding", "source_file", "column_order"}

            # F-M01: Resource-exhaustion guard — bound column count.
            _column_keys = [k for k in data if k not in metadata_keys]
            if len(_column_keys) > MAX_CURVES:
                raise ValueError(
                    f"Number of columns ({len(_column_keys)}) exceeds maximum "
                    f"allowed ({MAX_CURVES})"
                )

            for key, value in data.items():
                if key in metadata_keys:
                    if key == "encoding":
                        dev.encoding = _safe_str(value, "utf-8")
                    elif key == "source_file":
                        dev.source_file = _safe_str(value)
                    elif key == "column_order":
                        if value is None:
                            dev.column_order = []
                        elif isinstance(value, str):
                            dev.column_order = [value]
                        else:
                            dev.column_order = list(value)
                else:
                    # F-I2-M02: None guard — np.array(None, dtype=np.float64)
                    # silently produces nan, consistent with string data guards.
                    if value is None:
                        raise ValueError(
                            f"Numeric data for column '{key}' is None"
                        )
                    try:
                        dev.columns[key] = np.atleast_1d(np.array(value, dtype=np.float64))
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            f"Cannot convert data for column '{key}' to numeric array: {e}"
                        ) from e
                    # F-M01: Per-array size guard for DevFile columns.
                    if len(dev.columns[key]) > MAX_DATA_LINES:
                        raise ValueError(
                            f"Column '{key}' length ({len(dev.columns[key])}) exceeds "
                            f"maximum allowed ({MAX_DATA_LINES})"
                        )

            # F-M01: Total element count guard across all columns.
            if dev.columns:
                _dev_rows = max(len(arr) for arr in dev.columns.values())
                _dev_total = len(dev.columns) * _dev_rows
                if _dev_total > MAX_TOTAL_ELEMENTS:
                    raise ValueError(
                        f"Total allocation ({len(dev.columns)} columns x "
                        f"{_dev_rows} rows = {_dev_total} elements) exceeds "
                        f"maximum allowed ({MAX_TOTAL_ELEMENTS})"
                    )

            # F-006: Cross-array length validation for DevFile columns.
            # Inconsistent-length columns produce silently corrupted output
            # if accepted — different columns with different row counts
            # represent invalid DEV survey data.
            if len(dev.columns) > 1:
                _col_len = {name: len(arr) for name, arr in dev.columns.items()}
                if len(set(_col_len.values())) > 1:
                    raise ValueError(
                        f"DevFile columns have inconsistent lengths: {_col_len}"
                    )

            # If column_order wasn't in the dict, infer from Python 3.7+ dict order
            if not dev.column_order:
                dev.column_order = list(dev.columns.keys())
            return dev
        except (ValueError, TypeError) as e:
            raise LASDataError(str(e)) from e
