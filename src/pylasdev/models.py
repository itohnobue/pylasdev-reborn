"""Data models for LAS file structures.

Supports LAS 1.2, 2.0, and 3.0 formats.
"""

from __future__ import annotations

import copy
import math
import re
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._version_spec import _LASVersionSpec

# I2F-09: Maximum field length for string values — matches parser's limit
# (parser.py:86).  Without this, from_dict paths accept arbitrarily-long
# strings that bypass all item-count and element-count guards.
MAX_FIELD_LENGTH = 100_000


def _safe_str(
    value: Any, default: str = "", max_length: int | None = MAX_FIELD_LENGTH
) -> str:
    """Convert value to str, returning *default* when *value* is None.

    Prevents ``str(None)`` → ``"None"`` in dict roundtrip paths.
    When *max_length* is not ``None``, raises ``ValueError`` if the result
    exceeds the limit (I2F-09: unbounded string lengths).
    """
    if value is None:
        return default
    # F-I2-MD3-03: str(float('nan')) → "nan", str(float('inf')) → "inf".
    # Non-finite float values produce corrupted string representations
    # that propagate silently through from_dict roundtrip paths.
    # F-003: np.float32 (and other numpy floating types) do NOT
    # inherit from Python's builtin ``float``.  ``isinstance(value, float)``
    # misses numpy float types, allowing np.float32(np.nan) and
    # np.float32(np.inf) to bypass the non-finite guard.
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError(
            f"Cannot safely convert non-finite float {value} to string"
        )
    if isinstance(value, bytes):
        raise TypeError(
            "Decode to str first: value.decode('utf-8')"
        )
    result = str(value)
    # F-22: Strip control characters (except common whitespace: \t, \n, \r).
    # Control chars like \x00, \x1b, \x7f pass through str() unchanged and
    # can corrupt LAS output or downstream consumers reading dataclass fields.
    result = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x85\u2028\u2029]", "", result)
    if max_length is not None and len(result) > max_length:
        raise ValueError(
            f"String value length {len(result)} exceeds maximum allowed "
            f"({max_length})"
        )
    return result


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
        if _zone_index is not None and type(_zone_index) is not int:
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
    if _array_index is not None and type(_array_index) is not int:
        raise TypeError(
            f"array_index: expected int or None, "
            f"got {type(_array_index).__name__}"
        )
    # F-053: Restore per-section parameter grouping on roundtrip.
    # section_type is optional; None means the standard ~P section.
    _section_type = param_dict.get("section_type")
    if _section_type is not None and not isinstance(_section_type, str):
        _section_type = _safe_str(_section_type)
    # I2F-08 / F-M09: Empty-string and whitespace-only section_type is
    # not normalized to None.  ``""`` passes both guards (is not None,
    # isinstance str), and the writer silently drops parameters with
    # falsy section_type (line 532: ``if not section_type: continue``).
    # ``"  "`` (whitespace-only) passes all guards and survives
    # roundtrip, producing unparseable ``~  _Parameter`` headers.
    # Normalize both to None so that empty-string, whitespace-only,
    # and absent are treated identically.
    if _section_type is not None and _section_type.strip() == "":
        _section_type = None
    # F-118: Reject section_type values that could inject headers when
    # written.  The writer applies ``_sanitize_las_value`` which converts
    # ``\n`` → space, then ``_LEADING_SECTION_RE`` only strips tilde at
    # the STRING START — embedded ``~VERSION`` survives.  The emitted
    # header ``~CORE ~VERSION_Parameter`` is misrouted as a DATA section
    # by the parser.  Reject these values at construction time.
    if _section_type is not None:
        _sec_str = str(_section_type)
        if '\n' in _sec_str or '\r' in _sec_str or '~' in _sec_str:
            raise ValueError(
                f"section_type contains invalid characters "
                f"(newline or tilde): {_sec_str!r}.  "
                f"section_type must be a LAS identifier "
                f"(alphanumeric + underscore)."
            )
    # I2F-12: Validate parameter data_format against the same valid set
    # used for curve data_format.  Curve validation runs in two places
    # (_validate_from_dict_input lines 227-237 and 241-257).  Parameter
    # data_format had zero validation — any string was silently accepted
    # and propagated through writer as a {XYZ} format specifier.
    #
    # Only validate single-character values — multi-character strings
    # like "DD/MM/YYYY" are metadata (date format descriptors, etc.)
    # stored under the data_format key by real-world LAS files, not
    # LAS format specifiers.  Single-character non-format values like
    # "X" or "G" would be propagated as {X} / {G} format specifiers
    # and produce corrupted LAS output.
    _data_format = _safe_str(param_dict.get("data_format"), "")
    if _data_format and len(_data_format) == 1 and _data_format not in _VALID_DATA_FORMATS:
        raise ValueError(
            f"Invalid data_format '{_data_format}' for parameter "
            f"'{param_dict.get('mnemonic', '?')}'. "
            f"Valid values: {', '.join(sorted(_VALID_DATA_FORMATS))}"
        )
    return ParameterEntry(
        mnemonic=_safe_str(param_dict.get("mnemonic", "")),
        unit=_safe_str(param_dict.get("unit", "")),
        value=_safe_str(param_dict.get("value", "")),
        description=_safe_str(param_dict.get("description", "")),
        data_format=_data_format,
        array_index=_array_index,
        zone=zone,
        section_type=_section_type,
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
    # F-9-002: Reject bool when int is expected — bool subclasses int so
    # isinstance(True, int) is True, but bool is semantically not an int.
    # Consistent with type() is not int guards at lines 62, 84, 717.
    if (
        value is not None
        and isinstance(value, bool)
        and (
            expected_type is int
            or (isinstance(expected_type, tuple) and int in expected_type)
        )
    ):
        raise TypeError(
            f"{key}: expected {expected_type}, got bool"
        )
    # I2F-13: Non-finite floats (inf, nan) slip past the isinstance
    # gate — isinstance(float('inf'), (int, float)) is True, but the
    # writer crashes at int(float('inf')) → OverflowError.  The parser
    # has np.isfinite protection; from_dict lacked it.
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        raise ValueError(
            f"{key}: non-finite float values (inf, -inf, nan) are "
            f"not allowed"
        )
    return value


# Valid data_format values for LAS curves (LAS 3.0 spec).
_VALID_DATA_FORMATS = frozenset({"F", "E", "D", "A", "S", "I"})


def _validate_from_dict_input(data: dict[str, Any]) -> None:
    """Validate from_dict input before construction (F-017, F-018, IF-015,
    IF-026, F-019).

    Raises ``ValueError`` or ``TypeError`` for invalid input.
    Called at the start of ``LASFile.from_dict`` before any construction,
    closing Pattern #7 (from_dict validation gaps) structurally.
    """
    # --- F-017: Mandatory well fields (STRT, STOP, STEP, NULL) ---
    # F-028: Truthiness-based asymmetry — ``and well`` skipped the check
    # for ``well={}`` (empty dict is falsy) while ``well={"STRT": "0"}``
    # (populated but incomplete) triggered it.  Both are equally invalid.
    well = data.get("well")
    if isinstance(well, dict):
        # M-24: All 8 LAS 1.2 mandatory well fields (was 4: STRT, STOP,
        # STEP, NULL).  WELL, LOC, SRVC, and UWI are commonly missing in
        # real-world files — we warn but do NOT raise an error, matching
        # parser.py behavior.
        _mandatory = {"STRT", "STOP", "STEP", "NULL", "WELL", "LOC", "SRVC", "UWI"}
        _well_keys = {k.upper() for k in well if isinstance(k, str)}
        _missing = _mandatory - _well_keys
        if _missing:
            # F-017: Warn about missing mandatory well fields at construction
            # time.  Consistent with parser.py:436-442 and writer.py:333-343
            # which also warn rather than error.  The writer will produce valid
            # LAS output with defaults for missing fields.
            warnings.warn(
                f"Mandatory well field(s) missing: "
                f"{', '.join(sorted(_missing))}",
                stacklevel=3,
            )

    # --- I2F-11: VERS presence check ---
    # The parser raises LASParseError for missing VERS (parser.py:406-409).
    # from_dict previously silently defaulted to "2.0" via _safe_str(),
    # manufacturing version metadata on roundtrip gaps.
    version = data.get("version")
    if isinstance(version, dict):
        if "VERS" not in version:
            raise ValueError(
                "Missing required VERS field in version section. "
                "VERS must be present (e.g. '1.2', '2.0', '3.0')."
            )
        # H-04: Validate VERS value against known LAS versions.
        # M-04: VERS warning removed from here — duplicate of
        # VersionSection.__post_init__ (line 577).  The post_init
        # check covers all construction paths (direct, parser,
        # from_dict) as the single authoritative source.

    # --- F-018: DLM validation ---
    if isinstance(version, dict):
        dlm_raw = version.get("DLM")
        if dlm_raw is not None and dlm_raw != "":
            dlm_upper = str(dlm_raw).upper()
            # F-I2-MD3-01: Validate DLM value first, then check LAS
            # version compatibility.  The previous if/elif structure
            # warned for LAS 1.2 with non-SPACE DLM but skipped the
            # validity rejection when the DLM was completely invalid
            # (e.g. "FOO").  Flipping the order guarantees that
            # invalid DLMs are always rejected regardless of version.
            if dlm_upper not in {"SPACE", "TAB", "COMMA"}:
                raise ValueError(
                    f"Invalid DLM value '{dlm_raw}'. "
                    f"Expected SPACE, TAB, or COMMA."
                )
            # M-04: DLM LAS 1.2 warning removed from here — duplicate
            # of VersionSection.__post_init__ (line 617).  The post_init
            # check covers all construction paths (direct, parser,
            # from_dict) as the single authoritative source.

        # F-006: WRAP not validated against {YES, NO} in from_dict.
        # The parser enforces WRAP ∈ {YES, NO} (parser.py:1097-1108).
        # from_dict previously accepted any string, matching the
        # parser's strictness to prevent silently-corrupted WRAP
        # values propagating through roundtrip.
        wrap_raw = version.get("WRAP")
        if wrap_raw is not None:
            wrap_str = str(wrap_raw).upper()
            if wrap_str not in {"YES", "NO"}:
                raise ValueError(
                    f"Invalid WRAP value '{wrap_raw}'. "
                    f"Expected YES or NO."
                )

    # --- IF-015: data_format validation ---
    curves = data.get("curves")
    if isinstance(curves, list):
        for i, cd in enumerate(curves):
            if isinstance(cd, dict) and "data_format" in cd:
                _raw = cd.get("data_format")
                if _raw is None or isinstance(_raw, bool):
                    continue  # None means "not set"; bool is not a data format
                # F-17: Use _safe_str() instead of str() to reject
                # non-finite floats (inf, nan) — str(float('inf')) →
                # "INF" → "I" ∈ _VALID_DATA_FORMATS, bypassing validation.
                df = _safe_str(_raw).upper()
                if df:
                    df = df[0]  # F-M-012: truncate extended format codes (parser accepts F8.3, etc.)
                    cd["data_format"] = df  # F-017: truncate in-place so from_dict passes single-char to constructor
                if df and df not in _VALID_DATA_FORMATS:
                    raise ValueError(
                        f"curves[{i}]: invalid data_format '{cd['data_format']}'. "
                        f"Valid values: {', '.join(sorted(_VALID_DATA_FORMATS))}"
                    )
    # Per-section curves
    data_sections = data.get("data_sections")
    if isinstance(data_sections, list):
        for si, ds in enumerate(data_sections):
            if not isinstance(ds, dict):
                continue
            scs = ds.get("section_curves")
            if isinstance(scs, list):
                for ci, sc in enumerate(scs):
                    if isinstance(sc, dict) and "data_format" in sc:
                        _raw = sc.get("data_format")
                        if _raw is None or isinstance(_raw, bool):
                            continue
                        # F-17: Use _safe_str() to reject non-finite floats
                        # (same fix as top-level curves path above).
                        df = _safe_str(_raw).upper()
                        if df:
                            df = df[0]  # F-M-012: truncate extended format codes (parser accepts F8.3, etc.)
                            sc["data_format"] = df  # F-017: truncate in-place for from_dict construction
                        if df and df not in _VALID_DATA_FORMATS:
                            raise ValueError(
                                f"data_sections[{si}].section_curves[{ci}]: "
                                f"invalid data_format '{sc['data_format']}'. "
                                f"Valid values: {', '.join(sorted(_VALID_DATA_FORMATS))}"
                            )

    # --- IF-026: Cross-validate data_format vs data placement ---
    _check_df_vs_placement(curves, data, "top-level")
    # Also validate per-section curves against their section-level data/string_data
    if isinstance(data_sections, list):
        for si, ds in enumerate(data_sections):
            if isinstance(ds, dict):
                scs = ds.get("section_curves")
                if isinstance(scs, list):
                    _check_df_vs_placement(
                        scs, ds, f"data_sections[{si}]"
                    )

    # --- F-019: Non-3.0 multi-section data_sections ---
    vers_raw = version.get("VERS") if isinstance(version, dict) else None
    vers = str(vers_raw).strip() if vers_raw else ""
    if isinstance(data_sections, list) and len(data_sections) > 1:
        if not vers.startswith("3"):
            raise ValueError(
                f"Multiple data_sections ({len(data_sections)}) are only "
                f"supported for LAS 3.0 files, but version is {vers!r}. "
                f"Use a single section for LAS 1.2/2.0."
            )

    # F-115: Detect ambiguous curve placement — when the same curve name
    # appears in both ``logs`` and ``data_sections[*]``, the writer
    # checks data_sections first (writer.py:743) and uses the LAS 3.0
    # path, silently ignoring the logs value.  Reject the ambiguous
    # input so callers get a clear error instead of silent data discard.
    # LAS 3.0 roundtrip legitimately produces both keys when logs
    # (top-level curves) and data_sections (section-level curves)
    # contain different curve names — that is not ambiguous.
    if isinstance(data_sections, list) and data_sections:
        # Collect all curve names referenced in data_sections.
        # F-M-015: Apply case normalization (upper()) before overlap
        # check.  mnem_base maps semantically-equal keys with different
        # cases to the same canonical name — raw-key intersection
        # produces false negatives when mnem_base normalization is active.
        # Collected once at the top so both F-115 (logs overlap) and
        # E-F-022 (string_data overlap) can use the same set.
        _ds_curve_names: set[str] = set()
        for _ds in data_sections:
            if isinstance(_ds, dict):
                _ds_co = _ds.get("curves_order")
                if isinstance(_ds_co, list):
                    _ds_curve_names.update(
                        str(s).upper() for s in _ds_co
                    )
                _ds_data = _ds.get("data")
                if isinstance(_ds_data, dict):
                    _ds_curve_names.update(
                        str(k).upper() for k in _ds_data.keys()
                    )
                _ds_str = _ds.get("string_data")
                if isinstance(_ds_str, dict):
                    _ds_curve_names.update(
                        str(k).upper() for k in _ds_str.keys()
                    )

        # F-115: Detect ambiguous curve placement — logs vs
        # data_sections.
        _logs_raw = data.get("logs")
        if isinstance(_logs_raw, dict) and _logs_raw:
            _log_curve_names = {str(k).upper() for k in _logs_raw.keys()}
            _overlap = _log_curve_names & _ds_curve_names
            if _overlap:
                # F-115: Warn rather than reject — the parser roundtrip
                # legitimately produces both logs and data_sections with
                # overlapping curve names (same data, different storage).
                # Rejection would break LAS 3.0 roundtrips.  The warning
                # alerts users to potential silent data discard when
                # manually constructed dicts conflict.
                warnings.warn(
                    f"Curve(s) {sorted(_overlap)} appear in both "
                    f"'logs' and 'data_sections'.  The writer uses "
                    f"data_sections when both are present, ignoring "
                    f"the logs values for these curves.  Place each "
                    f"curve in exactly one location to avoid data "
                    f"loss.",
                    stacklevel=2,
                )

        # E-F-022: Detect overlap between top-level string_data and
        # data_sections[*].string_data.  The writer checks data_sections
        # first (writer.py:743) — when the same curve name appears in
        # both top-level string_data and data_sections, the top-level
        # value is silently ignored.  Warn so callers know about the
        # data discard.
        _top_str = data.get("string_data")
        if isinstance(_top_str, dict) and _top_str:
            _top_str_names = {str(k).upper() for k in _top_str.keys()}
            _ds_str_overlap = _top_str_names & _ds_curve_names
            if _ds_str_overlap:
                warnings.warn(
                    f"Curve(s) {sorted(_ds_str_overlap)} appear in "
                    f"both top-level 'string_data' and "
                    f"'data_sections'.  The writer uses "
                    f"data_sections when both are present, ignoring "
                    f"the top-level string_data values for these "
                    f"curves.  Place each curve in exactly one "
                    f"location to avoid data loss.",
                    stacklevel=2,
                )


def _check_df_vs_placement(
    curves: list[dict[str, Any]] | None,
    data: dict[str, Any],
    context: str,
) -> None:
    """Cross-validate curve data_format against data placement (IF-026)."""
    if not curves:
        return
    # Build sets of curve names in numeric data vs string data.
    # "logs" key for top-level LASFile dicts, "data" key for per-section DataSection dicts.
    logs = data.get("logs") or data.get("data") or {}
    string_data = data.get("string_data", {})
    logs_keys: set[str] = set()
    string_data_keys: set[str] = set()
    if isinstance(logs, dict):
        logs_keys = {str(k) for k in logs}
    if isinstance(string_data, dict):
        string_data_keys = {str(k) for k in string_data}

    for i, cd in enumerate(curves):
        if not isinstance(cd, dict):
            continue
        _raw_df = cd.get("data_format")
        if _raw_df is None or isinstance(_raw_df, bool):
            continue  # None means "not set"; bool is not a data format
        df = str(_raw_df).upper()
        mnemonic = _safe_str(cd.get("mnemonic"))
        if not df or not mnemonic:
            continue
        # I2F-37 broadened string-format exemption to {A} in the
        # string_data direction (line 426) but missed the logs direction.
        # {A}-format curves WITHOUT array_info are string-format and
        # must be rejected from numeric logs.  {A}-format curves WITH
        # array_info (e.g. NMR[1]) are genuinely numeric (float64) and
        # belong in logs/data.  (F-I2-MD4-02)
        if mnemonic in logs_keys:
            if df == "S" or (df == "A" and not cd.get("array_info")):
                raise ValueError(
                    f"{context} curve '{mnemonic}' (index {i}) has "
                    f"data_format='{df}' but is in logs (numeric data). "
                    f"String-format curves must be in string_data."
                )
        # I2F-37: Broaden the string-format exemption from "S" only to
        # "S" and "A".  The parser (parser.py:1905-1909) routes both
        # {S} and {A} (without array_info) format curves to string_data.
        # The previous guard rejected {A}-format curves in string_data
        # because "A" != "S", breaking the roundtrip path.  Numeric
        # formats (F, E, D) remain correctly rejected from string_data.
        if df not in ("S", "A") and mnemonic in string_data_keys:
            raise ValueError(
                f"{context} curve '{mnemonic}' (index {i}) has "
                f"data_format='{df}' but is in string_data. "
                f"Numeric-format curves must be in logs."
            )


@dataclass
class VersionSection:
    """LAS Version Information section (~V).

    Supports LAS 1.2, 2.0, and 3.0 formats.
    """

    vers: str = "2.0"
    wrap: str = "NO"
    dlm: str = "SPACE"  # LAS 2.0+: SPACE, TAB, or COMMA

    def validate(self, complete: bool = False) -> list[str]:
        """Validate version section fields.

        Args:
            complete: If True, also run deferred/semantic checks
                suitable for pre-write or post-construction validation.

        Returns:
            List of issue strings (empty = no issues found).
        """
        issues: list[str] = []
        # VERS: warn about unrecognized version values (H-02).
        # I2F-09: Prevent double VERS unrecognized warning.
        # __post_init__ calls validate(complete=False) (1st emission),
        # LASFile.validate(complete=True) calls version.validate(complete=True)
        # which would emit the same warning again.  Use a setattr-based
        # flag to suppress the second emission.
        if self.vers and self.vers not in {"1.2", "2.0", "3.0"}:
            if not getattr(self, '_vers_warned', False):
                object.__setattr__(self, '_vers_warned', True)
                issues.append(
                    f"VersionSection: Unrecognized VERS value "
                    f"{self.vers!r}. Expected 1.2, 2.0, or 3.0."
                )
        # DLM: warn about non-SPACE DLM on LAS 1.2.
        if complete and self.dlm:
            _dlm = self.dlm.upper()
            _spec = _LASVersionSpec(self.vers)
            if _spec.is_las12 and _dlm != "SPACE":
                issues.append(
                    f"DLM '{self.dlm}' is not valid for LAS 1.2 "
                    f"(spec requires SPACE).  The file will use "
                    f"SPACE delimiter on write."
                )
        return issues

    def __post_init__(self) -> None:
        """Validate VERS, WRAP, and DLM after construction.

        from_dict and the parser apply these checks before construction;
        direct ``VersionSection()`` construction previously bypassed them.
        Empty string values are tolerated (treated as "not specified"
        — the dataclass defaults apply).
        """
        # VERS: reject None before any string operations.
        if self.vers is None:
            raise ValueError("VersionSection: VERS cannot be None")
        # F-18: Strip whitespace from version string before any version checks.
        # "  3.0" causes version misidentification — is_las30 uses
        # str.startswith("3") which fails on leading whitespace.
        if self.vers:
            self.vers = self.vers.strip()
        # WRAP: reject None before any string operations.
        if self.wrap is None:
            raise ValueError("VersionSection: WRAP cannot be None")
        # WRAP: validate non-empty values against YES/NO
        # (matching parser.py:1097-1108).  Empty string = not specified.
        if self.wrap:
            _wrap = self.wrap.upper()
            self.wrap = _wrap
            if _wrap not in {"YES", "NO"}:
                raise ValueError(
                    f"VersionSection: invalid WRAP value "
                    f"{self.wrap!r}.  Expected YES or NO."
                )
        # DLM: validate non-empty values against SPACE/TAB/COMMA
        # (matching _validate_from_dict_input lines 246-269).
        # Empty string = not specified.
        if self.dlm is None:
            raise ValueError("VersionSection: DLM cannot be None")
        if self.dlm:
            _dlm = self.dlm.upper()
            # F-I2-MD3-01: Validate DLM value first, then check LAS
            # version compatibility.  The previous if/elif structure
            # warned for LAS 1.2 with non-SPACE DLM but skipped the
            # validity rejection when the DLM was completely invalid
            # (e.g. "FOO").  Ensuring the validity check runs
            # unconditionally before the version-specific warning
            # guarantees that invalid DLMs are always rejected.
            # Mirroring the fix from _validate_from_dict_input
            # (lines 265-282).
            if _dlm not in {"SPACE", "TAB", "COMMA"}:
                raise ValueError(
                    f"VersionSection: invalid DLM value "
                    f"{self.dlm!r}.  Expected SPACE, TAB, or COMMA."
                )
        # Run warning-producing checks via validate().
        for issue in self.validate(complete=False):
            warnings.warn(issue, stacklevel=2)

    def to_dict(self) -> dict[str, str]:
        """Convert to legacy dict format for backward compatibility."""
        return {
            "VERS": self.vers,
            # F2-26: Emit "NO" when wrap is None — the writer's
            # write-time override sets wrap to "NO" on disk but prior
            # to first write the field may be None (e.g. from direct
            # construction).  Returning None violates the dict[str, str]
            # type contract.
            "WRAP": self.wrap if self.wrap is not None else "NO",
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

    def validate(self, complete: bool = False) -> list[str]:
        """Validate well section fields.

        Args:
            complete: If True, also run deferred/semantic checks
                (STEP=0, NULL empty, STRT==STOP).

        Returns:
            List of issue strings (empty = no issues found).
            Mandatory field presence is checked separately
            by _validate_from_dict_input (from_dict path) and
            the parser (parse path) with version-aware messages.
        """
        issues: list[str] = []
        if not complete:
            return issues

        # STEP=0 check.
        _step_val = None
        for key, value in self.entries.items():
            if key.upper() == "STEP":
                _step_val = value
                break
        if _step_val is not None:
            try:
                if float(_step_val) == 0.0:
                    issues.append(
                        "STEP is zero — depth increment is invalid."
                    )
            except (TypeError, ValueError):
                pass

        # NULL empty check.
        _null_val = self.entries.get("NULL", "")
        if isinstance(_null_val, str) and not _null_val:
            issues.append(
                "NULL is an empty string — null value is ambiguous."
            )

        # STRT==STOP check.
        _strt_raw = self.entries.get("STRT")
        _stop_raw = self.entries.get("STOP")
        if _strt_raw is not None and _stop_raw is not None:
            try:
                if float(_strt_raw) == float(_stop_raw):
                    issues.append(
                        f"STRT equals STOP ({_strt_raw}) — well has "
                        f"zero depth range."
                    )
            except (TypeError, ValueError):
                pass

        return issues

    def __post_init__(self) -> None:
        """Validate entries dict contains only string keys (F-M-008).

        Direct construction bypasses parser and from_dict paths which
        already validate key types.  Non-string keys cause the writer
        to crash on ``key.upper()``.
        """
        if not isinstance(self.entries, dict):
            raise TypeError(
                f"WellSection: entries must be a dict, "
                f"got {type(self.entries).__name__}"
            )
        for _key in self.entries:
            if not isinstance(_key, str):
                raise TypeError(
                    f"WellSection: all entry keys must be str, "
                    f"got {type(_key).__name__} ({_key!r})"
                )
        # M-14: coerce non-str values to str with a warning.
        for _key, _val in self.entries.items():
            if not isinstance(_val, str):
                warnings.warn(
                    f"WellSection: coercing non-str value for key "
                    f"{_key!r} from {type(_val).__name__} to str",
                    stacklevel=2,
                )
                self.entries[_key] = _safe_str(_val)
            # F-001: MAX_FIELD_LENGTH bypass for already-string values.
            # __setitem__ guards length on mutation (L804-808), but direct
            # construction with a long string skips both _safe_str() and
            # __setitem__ — the isinstance(…, str) gate above short-circuits.
            elif len(_val) > MAX_FIELD_LENGTH:
                raise ValueError(
                    f"WellSection: value length {len(_val)} for key "
                    f"{_key!r} exceeds maximum allowed ({MAX_FIELD_LENGTH})"
                )
        # I2F-05: Validate units and descriptions dicts.
        # WellSection.__post_init__ validates entries key/value types but
        # does NOT validate units dict or descriptions dict.  Non-string
        # VALUES cause writer crash via _sanitize_las_value() →
        # AttributeError on .replace().  Non-string KEYS are silently ignored.
        for _dict_name, _dict_ref in (("units", self.units), ("descriptions", self.descriptions)):
            if not isinstance(_dict_ref, dict):
                raise TypeError(
                    f"WellSection: {_dict_name} must be a dict, "
                    f"got {type(_dict_ref).__name__}"
                )
            for _dk, _dv in _dict_ref.items():
                if not isinstance(_dk, str):
                    raise TypeError(
                        f"WellSection: {_dict_name} key must be str, "
                        f"got {type(_dk).__name__} ({_dk!r})"
                    )
                if not isinstance(_dv, str):
                    raise TypeError(
                        f"WellSection: {_dict_name} value for key "
                        f"{_dk!r} must be str, "
                        f"got {type(_dv).__name__} ({_dv!r})"
                    )

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
        # F-06: Validate value length matches __post_init__ guard.
        # Direct API users and post-transform length changes bypass
        # parser's _validate_data_line_fields() mitigation.
        if len(value) > MAX_FIELD_LENGTH:
            raise ValueError(
                f"WellSection value length {len(value)} exceeds "
                f"maximum allowed ({MAX_FIELD_LENGTH})"
            )
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

    def validate(self, complete: bool = False) -> list[str]:
        """Validate array element info fields.

        Returns:
            List of issue strings (empty = no issues found).
            All structural checks raise during construction
            and are not duplicated here.
        """
        return []

    def __post_init__(self) -> None:
        """Validate base_name, index, and time_offset.

        Direct construction bypasses parser and from_dict validation.
        Non-finite time_offset values (NaN/Inf) silently propagate to
        LAS output.
        """
        if not self.base_name or not self.base_name.strip():
            raise ValueError(
                f"ArrayElementInfo: base_name must not be empty or "
                f"whitespace-only, got {self.base_name!r}"
            )
        if type(self.index) is not int:
            raise TypeError(
                f"ArrayElementInfo: index must be int, got "
                f"{type(self.index).__name__} ({self.index!r})"
            )
        if self.index < 0:
            raise ValueError(
                f"ArrayElementInfo: index must be >= 0, got {self.index!r}"
            )
        if self.time_offset is not None and (
            not isinstance(self.time_offset, (int, float))
            or (isinstance(self.time_offset, float) and not math.isfinite(self.time_offset))
        ):
            raise ValueError(
                f"ArrayElementInfo: time_offset must be a finite number "
                f"or None, got {self.time_offset!r}"
            )
        if self.time_offset is not None and self.time_offset < 0:
            raise ValueError(
                f"ArrayElementInfo: time_offset must be >= 0, "
                f"got {self.time_offset!r}"
            )


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

    def validate(self, complete: bool = False) -> list[str]:
        """Validate curve definition fields.

        Returns:
            List of issue strings (empty = no issues found).
            All structural checks raise during construction
            and are not duplicated here.
        """
        return []

    def __post_init__(self) -> None:
        """Reject empty-or-whitespace mnemonic and invalid data_format.

        Empty mnemonics produce malformed LAS output (`` .UNIT  : DESC``).
        An invalid data_format propagates through the writer as a
        ``{INVALID}`` format specifier, producing corrupted LAS output.
        """
        # F-I2-MD3-07: Reject mnemonics with embedded spaces.
        # strip() catches leading/trailing whitespace but "GR 1"
        # passes — embedded spaces survive to produce corrupted
        # LAS output (space is the field delimiter).
        # F-004: Reject \n and \r in mnemonics.  The writer replaces
        # them with spaces, producing corrupted output where field
        # boundaries shift.  strip() catches leading/trailing newlines
        # but embedded \n/\r (e.g. "GR\nCAL") pass the space/tab checks.
        # F-030: Reject dots in mnemonics.  The writer uses dot as a
        # structural separator in LAS output; the parser splits on the
        # first dot.  A mnemonic like "GR.CO" would be written as
        # "GR.CO.M/FT" and parsed back as mnemonic="GR", unit="CO.M/FT"
        # — causing roundtrip corruption.
        if (not self.mnemonic or not self.mnemonic.strip()
                or self.mnemonic != self.mnemonic.strip()
                or ' ' in self.mnemonic.strip()
                or '\t' in self.mnemonic.strip()
                or '\n' in self.mnemonic
                or '\r' in self.mnemonic
                or '.' in self.mnemonic):
            raise ValueError(
                f"CurveDefinition: mnemonic must not be empty, "
                f"whitespace-only, or contain spaces/tabs/newlines/dots, "
                f"got {self.mnemonic!r}"
            )
        if self.data_format and self.data_format not in _VALID_DATA_FORMATS:
            raise ValueError(
                f"CurveDefinition: invalid data_format "
                f"'{self.data_format}' for curve '{self.mnemonic}'. "
                f"Valid values: {', '.join(sorted(_VALID_DATA_FORMATS))}"
            )
        # M-15: array_info must be ArrayElementInfo or None.
        if self.array_info is not None and not isinstance(
            self.array_info, ArrayElementInfo
        ):
            raise TypeError(
                f"CurveDefinition: array_info must be ArrayElementInfo "
                f"or None, got {type(self.array_info).__name__}"
            )

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

    def validate(self, complete: bool = False) -> list[str]:
        """Validate parameter zone fields.

        Returns:
            List of issue strings (empty = no issues found).
            Basic type/value checks raise during construction.
        """
        issues: list[str] = []
        # zone_name: warn if None or empty.
        if self.zone_name is None:
            issues.append(
                "ParameterZone: zone_name is None.  Zone names "
                "must be non-empty strings for valid LAS output."
            )
        elif isinstance(self.zone_name, str) and not self.zone_name.strip():
            issues.append(
                f"ParameterZone: zone_name {self.zone_name!r} is "
                f"empty or whitespace-only.  Zone names must be "
                f"non-empty strings for valid LAS output."
            )
        return issues

    def __post_init__(self) -> None:
        """Validate zone_index is int or None (F-M-010).

        Direct construction bypasses parser and from_dict paths which
        already validate the zone_index type.  A non-int value would
        pass silently and produce incorrect output.
        """
        if self.zone_index is not None and type(self.zone_index) is not int:
            raise TypeError(
                f"ParameterZone: zone_index must be int or None, "
                f"got {type(self.zone_index).__name__} "
                f"({self.zone_index!r})"
            )
        if self.zone_index is not None and self.zone_index < 0:
            raise ValueError(
                f"ParameterZone: zone_index must be >= 0, "
                f"got {self.zone_index!r}"
            )
        # Run warning-producing checks via validate().
        for issue in self.validate(complete=False):
            warnings.warn(issue, stacklevel=2)


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
    # F-053: Per-section parameter grouping.  When parameters are parsed
    # from a LAS 3.0 typed section (e.g., ~Core_Parameter), section_type
    # records the originating section type (e.g., "CORE") so the writer
    # can reconstruct per-section parameter sections on roundtrip.
    # Parameters from a standard ~P/~Parameter section have section_type=None.
    section_type: str | None = None

    def validate(self, complete: bool = False) -> list[str]:
        """Validate parameter entry fields.

        Returns:
            List of issue strings (empty = no issues found).
            Mnemonic, data_format, array_index, zone, and section_type
            checks raise during construction and are not duplicated here.
            Unit/value/description coercion warnings fire during
            __post_init__ before coercion.
        """
        return []

    def __post_init__(self) -> None:
        """Reject empty-or-whitespace mnemonic (F-M10).

        Empty mnemonics produce malformed LAS output.
        """
        # F-I2-MD3-07: Reject mnemonics with embedded spaces.
        # strip() catches leading/trailing whitespace but "GR 1"
        # passes — embedded spaces survive to produce corrupted
        # LAS output (space is the field delimiter).
        # F-004: Reject \n and \r in mnemonics — same gap as
        # CurveDefinition.__post_init__ above.
        # F-030: Reject dots in mnemonics — same roundtrip corruption
        # as CurveDefinition (writer uses dot as structural separator).
        if (not self.mnemonic or not self.mnemonic.strip()
                or self.mnemonic != self.mnemonic.strip()
                or ' ' in self.mnemonic.strip()
                or '\t' in self.mnemonic.strip()
                or '\n' in self.mnemonic
                or '\r' in self.mnemonic
                or '.' in self.mnemonic):
            raise ValueError(
                f"ParameterEntry: mnemonic must not be empty, "
                f"whitespace-only, or contain spaces/tabs/newlines/dots, "
                f"got {self.mnemonic!r}"
            )
        # F-21: Validate that unit, value, and description are strings.
        # Non-str values (int, float, None) pass silently at construction
        # and crash downstream on .upper() or .strip() calls.  Warn and
        # coerce via str() to maintain backward compatibility.
        for _attr_name, _attr_val in (
            ("unit", self.unit),
            ("value", self.value),
            ("description", self.description),
        ):
            if not isinstance(_attr_val, str):
                warnings.warn(
                    f"ParameterEntry '{self.mnemonic}': coercing "
                    f"non-str {_attr_name} from "
                    f"{type(_attr_val).__name__} to str",
                    stacklevel=2,
                )
                setattr(self, _attr_name, _safe_str(_attr_val))
        # F-M-009: Validate data_format when provided, mirroring
        # CurveDefinition.__post_init__ (lines 619-624) and the
        # from_dict parameter path (lines 125-137).  Only validate
        # single-character values — multi-character strings like
        # "DD/MM/YYYY" are metadata descriptors, not LAS format
        # specifiers.  Single-character non-format values like "X"
        # would be propagated as {X} format specifiers and produce
        # corrupted LAS output.
        if (self.data_format
                and len(self.data_format) == 1
                and self.data_format not in _VALID_DATA_FORMATS):
            raise ValueError(
                f"ParameterEntry: invalid data_format "
                f"'{self.data_format}' for parameter "
                f"'{self.mnemonic}'.  Valid values: "
                f"{', '.join(sorted(_VALID_DATA_FORMATS))}"
            )
        # F-005: Validate array_index type (int | None).
        if self.array_index is not None and type(self.array_index) is not int:
            raise TypeError(
                f"ParameterEntry.array_index must be int or None, "
                f"got {type(self.array_index).__name__}"
            )
        # F-005: Validate zone type (ParameterZone | None).
        if self.zone is not None and not isinstance(self.zone, ParameterZone):
            raise TypeError(
                f"ParameterEntry.zone must be ParameterZone or None, "
                f"got {type(self.zone).__name__}"
            )
        # F-064: Validate section_type — reject newline/carriage-return/tilde,
        # normalize whitespace-only to None.  Matches _create_parameter_entry
        # validation (lines 91-118).
        if self.section_type is not None:
            if not isinstance(self.section_type, str):
                self.section_type = _safe_str(self.section_type)
            _stripped = self.section_type.strip()
            if not _stripped:
                self.section_type = None
            else:
                self.section_type = _stripped
                _sec_str = str(self.section_type)
                if '\n' in _sec_str or '\r' in _sec_str or '~' in _sec_str:
                    raise ValueError(
                        f"ParameterEntry.section_type contains invalid "
                        f"characters (newline or tilde): {_sec_str!r}"
                    )

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
        if self.section_type is not None:
            result["section_type"] = self.section_type
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
    # F-36: dtype=object used at all construction sites; np.str_ annotation
    # was misleading and never matched actual usage.
    string_data: dict[str, NDArray[np.object_]] = field(default_factory=dict)  # For {S} format curves
    section_curves: list[CurveDefinition] = field(
        default_factory=list
    )  # Per-section curve definitions

    # M-05: from_dict re-validation control flag.  When True,
    # __post_init__ suppresses the NaN/Inf warning — the from_dict
    # path has already validated input; NaN is standard missing-data
    # representation in LAS.  Consistent with LASFile._from_dict.
    _from_dict: bool = field(default=False, repr=False)

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

    def validate(self, complete: bool = False) -> list[str]:
        """Validate data section fields.

        .. note::

            This method **mutates** ``self.data`` and ``self.string_data``
            in-place: any non-``numpy.ndarray`` values are coerced via
            ``np.asarray()`` so that dtype checks are reliable.  This
            conversion is lossless — lists and array-likes are wrapped
            into equivalent numpy arrays without changing data values.

        Args:
            complete: If True, also run deferred/semantic checks
                (NaN/Inf in numeric data, dtype validation,
                uncovered curves).

        Returns:
            List of issue strings (empty = no issues found).
            Basic structural checks (type, keys, lengths, data_format)
            raise during construction and are not duplicated here.
        """
        issues: list[str] = []
        # --- dtype validation for data arrays ---
        for _k, _arr in self.data.items():
            if not isinstance(_arr, np.ndarray):
                _arr = self.data[_k] = np.asarray(_arr)
            if not np.issubdtype(_arr.dtype, np.number):
                issues.append(
                    f"DataSection '{self.name}': curve '{_k}' in 'data' "
                    f"has non-numeric dtype ({_arr.dtype}).  'data' "
                    f"arrays must be numeric."
                )

        # --- dtype validation for string_data arrays ---
        for _sk, _sarr in self.string_data.items():
            if not isinstance(_sarr, np.ndarray):
                _sarr = self.string_data[_sk] = np.asarray(_sarr)
            if np.issubdtype(_sarr.dtype, np.number):
                issues.append(
                    f"DataSection '{self.name}': curve '{_sk}' in "
                    f"'string_data' has numeric dtype ({_sarr.dtype}).  "
                    f"'string_data' arrays must be non-numeric."
                )

        # --- NaN/Inf validation for numeric data arrays ---
        for _k, _arr in self.data.items():
            if isinstance(_arr, np.ndarray) and _arr.dtype.kind in ('f', 'c'):
                if not np.all(np.isfinite(_arr)):
                    issues.append(
                        f"DataSection '{self.name}': curve '{_k}' in "
                        f"'data' contains non-finite values (NaN/Inf)."
                    )

        # --- uncovered curves ---
        curve_set = set(self.curves_order)
        data_keys = set(self.data.keys())
        string_keys = set(self.string_data.keys())
        if self.data or self.string_data:
            uncovered = curve_set - data_keys - string_keys
            if uncovered:
                issues.append(
                    f"DataSection '{self.name}': curve(s) "
                    f"{sorted(uncovered)} appear in curves_order but "
                    f"have no data in 'data' or 'string_data'.  The "
                    f"writer will pad these curves with null_value."
                )

        # I2F-06: Validate data_format vs data/string_data placement.
        # __post_init__ checks this but validate() — which runs on
        # post-construction mutations — does not.  Adding an S-format
        # curve to data or an F-format curve to string_data after
        # construction would pass validate() silently.
        for _sc in self.section_curves:
            _df = _sc.data_format
            _mnem = _sc.mnemonic
            if not _df:
                continue
            if _mnem in data_keys and (
                _df == "S" or (_df == "A" and not _sc.is_array_element)
            ):
                issues.append(
                    f"DataSection '{self.name}': curve '{_mnem}' has "
                    f"data_format='{_df}' but is in data (numeric). "
                    f"String-format curves must be in string_data."
                )
            if _df not in ("S", "A") and _mnem in string_keys:
                issues.append(
                    f"DataSection '{self.name}': curve '{_mnem}' has "
                    f"data_format='{_df}' but is in string_data. "
                    f"Numeric-format curves must be in data."
                )

        return issues

    def __post_init__(self) -> None:
        """Validate invariants after construction (F-001).

        DataSection is a public API type — direct construction bypasses
        the validation in ``LASFile.from_dict``.  This __post_init__
        catches invalid state that would cause silent data loss on write.
        """
        # Deferred import to avoid circular dependencies.
        from .exceptions import LASDataError

        # M-25: Reject bytes for curves_order — matches top-level
        # guard in from_dict at line 1768.  ``set(b"GR")`` produces
        # ``{71, 82}`` (integers), corrupting all downstream checks.
        # ``str(b"GR")`` produces ``"b'GR'"`` which corrupts column
        # headers.  Reject with the same error type as the top-level.
        if isinstance(self.curves_order, bytes):
            raise TypeError(
                "DataSection curves_order must be a list of strings, "
                "got bytes.  Decode to str first: "
                "curves_order.decode('utf-8')"
            )

        curve_set = set(self.curves_order)

        # F-105: Reject duplicate curve names in curves_order.
        # The from_dict path has explicit dedup checks (lines 1171-1181),
        # but direct DataSection construction bypasses those entirely.
        # ``DataSection(curves_order=["GR","GR","DT"])`` passes silently
        # and the writer produces duplicate columns in output.
        if len(self.curves_order) != len(curve_set):
            _seen: set[str] = set()
            _dups: list[str] = []
            for c in self.curves_order:
                if c in _seen:
                    _dups.append(c)
                else:
                    _seen.add(c)
            raise LASDataError(
                f"DataSection '{self.name}': duplicate curve names in "
                f"curves_order: {_dups}"
            )

        # F-M06 / F-M09: Validate and normalize section_type.
        # _create_parameter_entry validates section_type for ParameterEntry
        # (rejects newline/tilde) and normalizes whitespace-only to None;
        # DataSection previously skipped both checks.  Mirror that
        # validation here.
        if self.section_type is not None:
            _stripped = self.section_type.strip()
            if not _stripped:
                # Whitespace-only → normalize to empty string.
                # _safe_str ensures non-None; empty is a valid
                # "no particular section type" value.
                self.section_type = ""
            elif '\n' in _stripped or '\r' in _stripped or '~' in _stripped:
                raise LASDataError(
                    f"DataSection '{self.name}': section_type contains "
                    f"invalid characters (newline or tilde): "
                    f"{self.section_type!r}.  section_type must be a "
                    f"LAS identifier (alphanumeric + underscore)."
                )
            else:
                self.section_type = _stripped

        # data keys ⊆ curves_order
        data_keys = set(self.data.keys())
        orphaned_data = data_keys - curve_set
        if orphaned_data:
            raise LASDataError(
                f"DataSection '{self.name}': data keys not in curves_order: "
                f"{sorted(orphaned_data)}"
            )

        # string_data keys ⊆ curves_order
        string_keys = set(self.string_data.keys())
        orphaned_string = string_keys - curve_set
        if orphaned_string:
            raise LASDataError(
                f"DataSection '{self.name}': string_data keys not in "
                f"curves_order: {sorted(orphaned_string)}"
            )

        # section_curves length must match curves_order (when specified)
        if self.section_curves and len(self.section_curves) != len(self.curves_order):
            raise LASDataError(
                f"DataSection '{self.name}': section_curves length "
                f"({len(self.section_curves)}) does not match curves_order "
                f"length ({len(self.curves_order)})"
            )

        # F-M04 / F-M21: Validate positional mnemonic alignment between
        # section_curves and curves_order.  The from_dict path checks
        # this (L1351-1360); direct DataSection construction previously
        # bypassed.  Swapped mnemonics produce silently corrupted output.
        if self.section_curves:
            for _i, (_order_name, _sc) in enumerate(
                zip(self.curves_order, self.section_curves, strict=True)
            ):
                if _order_name != _sc.mnemonic:
                    raise LASDataError(
                        f"DataSection '{self.name}': "
                        f"curves_order[{_i}] = {_order_name!r} "
                        f"does not match "
                        f"section_curves[{_i}].mnemonic = "
                        f"{_sc.mnemonic!r}"
                    )

        # data and string_data must be disjoint — a curve name cannot
        # appear in both collections (writer silently picks one).
        colliding = data_keys & string_keys
        if colliding:
            raise LASDataError(
                f"DataSection '{self.name}': curve(s) {sorted(colliding)} "
                f"appear in both data and string_data.  Each curve must "
                f"be in exactly one collection."
            )

        # F-027 / F-079: Validate within-group array lengths.
        # The from_dict path has cross-array length checks (lines 1292-1308),
        # but direct DataSection construction bypasses those entirely.
        # Arrays of different lengths pass silently — the writer pads
        # shorter arrays with null_value, producing semantically incorrect
        # output.
        if self.data:
            # M-18: Guard against 0-d numpy arrays — len(np.array(5.0))
            # raises TypeError: len() of unsized object.  0-d arrays
            # are treated as single-element arrays.
            _data_lengths = {
                name: (1 if isinstance(arr, np.ndarray) and arr.ndim == 0
                       else len(arr))
                for name, arr in self.data.items()
            }
            if len(set(_data_lengths.values())) > 1:
                raise LASDataError(
                    f"DataSection '{self.name}' has inconsistent "
                    f"array lengths: {_data_lengths}"
                )
        if self.string_data:
            _string_lengths = {
                name: (1 if isinstance(arr, np.ndarray) and arr.ndim == 0
                       else len(arr))
                for name, arr in self.string_data.items()
            }
            if len(set(_string_lengths.values())) > 1:
                raise LASDataError(
                    f"DataSection '{self.name}' has inconsistent "
                    f"string_data array lengths: {_string_lengths}"
                )

        # F-M22: Cross-group row-count validation.
        # The from_dict path validates this (L1437-1445); direct
        # DataSection construction previously bypassed.  The writer's
        # _format_data_rows uses max() across all arrays — mismatched
        # row counts between data and string_data produce semantically
        # incorrect null-padded output.
        if self.data and self.string_data:
            # M-18: Guard against 0-d numpy arrays.
            _data_rows = max(
                (1 if isinstance(arr, np.ndarray) and arr.ndim == 0
                 else len(arr))
                for arr in self.data.values()
            )
            _string_rows = max(
                (1 if isinstance(arr, np.ndarray) and arr.ndim == 0
                 else len(arr))
                for arr in self.string_data.values()
            )
            if _data_rows != _string_rows:
                raise LASDataError(
                    f"DataSection '{self.name}': data row count "
                    f"({_data_rows}) does not match string_data "
                    f"row count ({_string_rows})"
                )

        # Cross-validate data_format against numeric/string data placement
        # (mirrors _check_df_vs_placement in from_dict path, IF-026).
        # Direct construction with S-format curve in data or F-format in
        # string_data passes silently without this check.
        # I2F-03: Validate ALL section types, not just LOG_DATA.
        # from_dict path validates all sections unconditionally.
        # Direct construction with section_type="CORE_DATA" + S-format
        # curve in numeric data previously bypassed this check entirely.
        for _sc in self.section_curves:
            _df = _sc.data_format
            _mnem = _sc.mnemonic
            if not _df:
                continue
            # I2F-37: {A} without array_info is string-format, not numeric.
            if _mnem in data_keys and (
                _df == "S" or (_df == "A" and not _sc.is_array_element)
            ):
                raise LASDataError(
                    f"DataSection '{self.name}': curve '{_mnem}' has "
                    f"data_format='{_df}' but is in data (numeric). "
                    f"String-format curves must be in string_data."
                )
            if _df not in ("S", "A") and _mnem in string_keys:
                raise LASDataError(
                    f"DataSection '{self.name}': curve '{_mnem}' has "
                    f"data_format='{_df}' but is in string_data. "
                    f"Numeric-format curves must be in data."
                )

        # Run warning-producing checks via validate() (gated by _from_dict).
        if not self._from_dict:
            for issue in self.validate(complete=False):
                warnings.warn(issue, stacklevel=2)


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
    # F-36: dtype=object used at all construction sites; np.str_ annotation
    # was misleading and never matched actual usage.
    string_data: dict[str, NDArray[np.object_]] = field(default_factory=dict)  # For {S} format curves

    # I2F-13: from_dict re-validation control flag.  When True,
    # __post_init__ suppresses warnings (not errors) — the from_dict
    # path has already validated everything during construction.
    _from_dict: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate critical invariants after construction (F-M05).

        ``from_dict`` validates 800+ lines of invariants; direct
        ``LASFile`` construction previously bypassed all validation.
        This catches the most egregious invalid states while allowing
        incremental construction (empty collections skip validation).
        """
        # Deferred import to avoid circular dependencies.
        from .exceptions import LASDataError

        # F-13: Validate curves_order element types for direct construction.
        # from_dict validates per-element types at lines 2100-2111;
        # direct LASFile() construction bypasses that.  Non-str elements
        # (int, None, float) pass silently — list(range(5)) produces
        # [0,1,2,3,4] as curve names, crashing downstream string operations.
        for _i, _name in enumerate(self.curves_order):
            if not isinstance(_name, str):
                raise TypeError(
                    f"LASFile: curves_order[{_i}] must be str, "
                    f"got {type(_name).__name__}: {_name!r}"
                )

        # --- curves_order / curves consistency ---
        # Every name in curves_order must have a matching CurveDefinition
        # at the same position.  Extra definitions (curves beyond
        # curves_order length) are tolerated — they may be LAS 3.0
        # per-section definitions also registered at the top level.
        if self.curves_order and self.curves:
            if len(self.curves) < len(self.curves_order):
                raise LASDataError(
                    f"LASFile: curves_order has "
                    f"{len(self.curves_order)} entries but only "
                    f"{len(self.curves)} curve definitions found"
                )
            for _i in range(len(self.curves_order)):
                if self.curves_order[_i] != self.curves[_i].mnemonic:
                    raise LASDataError(
                        f"LASFile: curves_order[{_i}] = "
                        f"{self.curves_order[_i]!r} does not match "
                        f"curves[{_i}].mnemonic = "
                        f"{self.curves[_i].mnemonic!r}"
                    )

        _order_set = set(self.curves_order) if self.curves_order else set()

        # M-19: LAS 2.0 first-curve-must-be-index constraint.
        # Now validated by validate() below (called at end of __post_init__).

        # --- duplicate curve name detection (F-MD4-06) ---
        # DataSection.__post_init__ has this check for per-section orders.
        # from_dict has it at lines 1755-1770 for legacy top-level orders.
        # LASFile.__post_init__ previously lacked it entirely.
        # For LAS 3.0 files with data_sections, skip the top-level
        # duplicate check — the same curve name legitimately appears in
        # each section's curves_order, producing duplicates at top level.
        if self.curves_order and not self.data_sections:
            _seen: set[str] = set()
            for _n in self.curves_order:
                if _n in _seen:
                    raise LASDataError(
                        f"LASFile: duplicate curve name {_n!r} in "
                        f"curves_order.  Curve names must be unique."
                    )
                _seen.add(_n)

        # --- logs validation (skip when empty — allows incremental
        # construction, e.g. LASFile() then attribute assignment) ---
        if self.logs:
            _log_keys = set(self.logs.keys())
            _orphaned_logs = _log_keys - _order_set
            if _orphaned_logs:
                raise LASDataError(
                    f"LASFile: logs contain keys not in "
                    f"curves_order: {sorted(_orphaned_logs)}.  "
                    f"Each log key must correspond to a curve "
                    f"mnemonic."
                )
            # F-I2-MD4-04: Detect missing keys — curves in curves_order
            # that have no corresponding log entry.  Skip when
            # data_sections is non-empty (LAS 3.0 files store curve data
            # per-section, not in the top-level logs dict).  Skip when
            # string_data covers the missing keys (the curve lives in
            # string_data, not logs).
            if not self.data_sections and self.curves_order:
                _missing_logs = _order_set - _log_keys
                _str_keys_for_missing = (
                    set(self.string_data.keys()) if self.string_data
                    else set()
                )
                _missing_logs -= _str_keys_for_missing
                if _missing_logs:
                    raise LASDataError(
                        f"LASFile: curves_order has keys not found in "
                        f"logs: {sorted(_missing_logs)}.  Each curve "
                        f"mnemonic in curves_order must have a "
                        f"corresponding log entry (or string_data entry "
                        f"for string-format curves)."
                    )
            if len(self.logs) > 1:
                _log_len = {name: len(arr) for name, arr in self.logs.items()}
                if len(set(_log_len.values())) > 1:
                    raise LASDataError(
                        f"LASFile: logs have inconsistent array "
                        f"lengths: {_log_len}"
                    )

        # --- string_data validation (same skip-when-empty rule) ---
        if self.string_data:
            _str_keys = set(self.string_data.keys())
            _orphaned_str = _str_keys - _order_set
            if _orphaned_str:
                raise LASDataError(
                    f"LASFile: string_data contain keys not in "
                    f"curves_order: {sorted(_orphaned_str)}.  "
                    f"Each string_data key must correspond to a "
                    f"curve mnemonic."
                )
            # F-I2-MD4-04: Detect missing keys — curves in curves_order
            # that have no corresponding string_data entry.  Guarded
            # the same way as logs above.
            if not self.data_sections and self.curves_order:
                _missing_str = _order_set - _str_keys
                _log_keys_for_missing = (
                    set(self.logs.keys()) if self.logs else set()
                )
                _missing_str -= _log_keys_for_missing
                if _missing_str:
                    raise LASDataError(
                        f"LASFile: curves_order has keys not found in "
                        f"string_data: {sorted(_missing_str)}.  Each "
                        f"curve mnemonic in curves_order must have a "
                        f"corresponding entry."
                    )
            if len(self.string_data) > 1:
                _str_len = {
                    name: len(arr) for name, arr in self.string_data.items()
                }
                if len(set(_str_len.values())) > 1:
                    raise LASDataError(
                        f"LASFile: string_data have inconsistent "
                        f"array lengths: {_str_len}"
                    )

        # F-I2-MD4-03: Detect key overlap between logs and string_data.
        # A curve name appearing in both would silently have data
        # corrupted — one array overwrites the other's meaning.
        # Per-section DataSection has this guard; top-level lacked it.
        if self.logs and self.string_data:
            _overlap = set(self.logs.keys()) & set(self.string_data.keys())
            if _overlap:
                raise LASDataError(
                    f"LASFile: curves {sorted(_overlap)} appear in "
                    f"both logs and string_data.  Each curve may "
                    f"only be stored in one location."
                )
            # F-M-031: Cross-group row-count validation.
            # The within-group checks above verify that all arrays in
            # 'logs' have the same length, and all arrays in
            # 'string_data' have the same length — but no cross-check
            # verifies that logs rows and string_data rows match.
            # A LASFile with 100-row numeric logs and 50-row string_data
            # passes all existing validation.  The writer's
            # ``_format_data_rows`` uses ``max()`` across all arrays,
            # so one group's shorter arrays get padded — producing
            # semantically incorrect output.
            _log_row_count = len(next(iter(self.logs.values())))
            _str_row_count = len(next(iter(self.string_data.values())))
            if _log_row_count != _str_row_count:
                raise LASDataError(
                    f"LASFile: logs row count ({_log_row_count}) does "
                    f"not match string_data row count "
                    f"({_str_row_count}).  Both must have the same "
                    f"number of rows."
                )

        # F-26: Validate dtypes for logs (numeric) and string_data (object).
        # DataSection.__post_init__ has these checks; LASFile omitted them.
        # F-20: Also validate NaN/Inf in numeric data arrays.
        # These are now handled by validate() below.

        # --- data_sections validation (F-MD4-01) ---
        # from_dict has ~200 lines of data_sections validation; __post_init__
        # previously had zero.  This validates the most critical invariants
        # for direct-construction scenarios while allowing incremental
        # construction (empty data_sections skips validation).
        if self.data_sections:
            # F-41: data_sections requires LAS 3.0 version.
            if not self.version.is_las30:
                raise LASDataError(
                    "data_sections requires LAS 3.0 version"
                )
            # F-14: Validate data_sections name uniqueness.
            # Duplicate section names produce ambiguous LAS 3.0 output —
            # the writer and parser both use section names to identify
            # sections.  The from_dict path has this check; direct
            # LASFile() construction previously bypassed it.
            _ds_names: list[str] = []
            for _ds in self.data_sections:
                _ds_names.append(_ds.name or "<unnamed>")
            _seen_ds: set[str] = set()
            for _ds_name in _ds_names:
                if _ds_name in _seen_ds:
                    raise LASDataError(
                        f"LASFile: duplicate data section name "
                        f"{_ds_name!r}.  Data section names must "
                        f"be unique."
                    )
                _seen_ds.add(_ds_name)
            _all_section_curve_names: set[str] = set()
            for _ds in self.data_sections:
                # Detect duplicate curve names within each section's
                # curves_order (matching DataSection.__post_init__).
                _ds_curves = _ds.curves_order
                if _ds_curves:
                    _ds_seen: set[str] = set()
                    for _cn in _ds_curves:
                        if _cn in _ds_seen:
                            raise LASDataError(
                                f"LASFile: duplicate curve name {_cn!r} "
                                f"in data section '{_ds.name}' "
                                f"curves_order."
                            )
                        _ds_seen.add(_cn)
                # Validate per-section data/string_data keys against
                # curves_order (orphaned key detection, matching from_dict
                # lines 1472-1491).
                _ds_order_set = set(_ds_curves) if _ds_curves else set()
                if _ds.string_data:
                    _ds_str_keys = set(_ds.string_data.keys())
                    _orphaned_str = _ds_str_keys - _ds_order_set
                    if _orphaned_str:
                        raise LASDataError(
                            f"LASFile: string_data in section "
                            f"'{_ds.name}' contains keys not in "
                            f"curves_order: {sorted(_orphaned_str)}."
                        )
                if _ds.data:
                    _ds_data_keys = set(_ds.data.keys())
                    _orphaned_data = _ds_data_keys - _ds_order_set
                    if _orphaned_data:
                        raise LASDataError(
                            f"LASFile: data in section '{_ds.name}' "
                            f"contains keys not in curves_order: "
                            f"{sorted(_orphaned_data)}."
                        )
                # Validate array length consistency within each section.
                if len(_ds.data) > 1:
                    _ds_len = {
                        name: len(arr) for name, arr in _ds.data.items()
                    }
                    if len(set(_ds_len.values())) > 1:
                        raise LASDataError(
                            f"LASFile: data in section '{_ds.name}' "
                            f"has inconsistent array lengths: {_ds_len}"
                        )
                if len(_ds.string_data) > 1:
                    _sds_len = {
                        name: len(arr)
                        for name, arr in _ds.string_data.items()
                    }
                    if len(set(_sds_len.values())) > 1:
                        raise LASDataError(
                            f"LASFile: string_data in section "
                            f"'{_ds.name}' has inconsistent array "
                            f"lengths: {_sds_len}"
                        )
                # Cross-group row-count consistency.
                if _ds.data and _ds.string_data:
                    _data_rows = max(len(arr) for arr in _ds.data.values())
                    _str_rows = max(
                        len(arr) for arr in _ds.string_data.values()
                    )
                    if _data_rows != _str_rows:
                        raise LASDataError(
                            f"LASFile: section '{_ds.name}': "
                            f"data row count ({_data_rows}) does not "
                            f"match string_data row count "
                            f"({_str_rows})"
                        )

            # F-10: Cross-curve array continuity validation per LAS 3.0 spec.
            # "Channels that are members of a 3D array must occur
            # sequentially from [1] to [n], with no other channels
            # intermixed." (LAS 3.0 Spec, Page 27)
            _ARRAY_MNEMONIC_RE = re.compile(
                r"^(?P<base>[\w\-]+)\[(?P<index>\d+)\]$"
            )
            for _ds in self.data_sections:
                if not _ds.curves_order:
                    continue
                # Group array curves by base name, tracking position
                # and index to validate contiguity and sequential order.
                _base_groups: dict[str, list[tuple[int, int]]] = {}
                for _pos, _name in enumerate(_ds.curves_order):
                    _m = _ARRAY_MNEMONIC_RE.match(_name)
                    if _m is None:
                        continue
                    _base = _m.group("base")
                    _idx = int(_m.group("index"))
                    _base_groups.setdefault(_base, []).append((_pos, _idx))
                for _base, _entries in _base_groups.items():
                    # Must have at least 2 entries to be an "array."
                    if len(_entries) < 2:
                        continue
                    # Check contiguity: positions must be consecutive
                    # (no intermixing).
                    _positions = [p for p, _ in _entries]
                    if _positions != list(
                        range(_positions[0], _positions[0] + len(_positions))
                    ):
                        raise LASDataError(
                            f"LASFile: array '{_base}' curves are not "
                            f"contiguous in section '{_ds.name}'. "
                            f"Array channels must appear sequentially "
                            f"with no other channels intermixed."
                        )
                    # Check sequential indices [1]→[n], no gaps.
                    _indices = [i for _, i in _entries]
                    if _indices != list(range(1, len(_indices) + 1)):
                        raise LASDataError(
                            f"LASFile: array '{_base}' has non-sequential "
                            f"indices {_indices} in section '{_ds.name}'. "
                            f"Expected [1]→[{len(_indices)}]."
                        )

        # F-12p2: Validate parameters list — check every entry is a
        # ParameterEntry and detect duplicate mnemonics.  Direct
        # LASFile() construction bypasses from_dict's validate_before-
        # construction checks.  Non-ParameterEntry objects pass silently.
        for _p in self.parameters:
            if not isinstance(_p, ParameterEntry):
                raise TypeError(
                    f"LASFile: all parameters must be ParameterEntry "
                    f"instances, got {type(_p).__name__}"
                )
        _pmn_seen: set[str] = set()
        for _p in self.parameters:
            if _p.mnemonic in _pmn_seen:
                warnings.warn(
                    f"LASFile: duplicate parameter mnemonic "
                    f"{_p.mnemonic!r} — to_dict() will keep only "
                    f"the last value.",
                    stacklevel=2,
                )
            _pmn_seen.add(_p.mnemonic)

        # F-059: Cross-validate curve data_format against actual data
        # placement.  Now validated by validate() below.

        # I2F-08: Raise LASDataError for format-vs-placement mismatches
        # in direct construction, matching from_dict and DataSection.
        # validate() only WARNS — direct construction MUST raise to
        # prevent silent data corruption.  from_dict raises via
        # _check_df_vs_placement; DataSection.__post_init__ raises.
        if self.curves and (self.logs or self.string_data):
            _log_keys = set(self.logs.keys()) if self.logs else set()
            _str_keys = set(self.string_data.keys()) if self.string_data else set()
            for _sc in self.curves:
                _df = _sc.data_format
                _mnem = _sc.mnemonic
                if not _df:
                    continue
                if _mnem in _log_keys and (
                    _df == "S" or (_df == "A" and not _sc.is_array_element)
                ):
                    raise LASDataError(
                        f"LASFile: curve '{_mnem}' has "
                        f"data_format='{_df}' (string-format) but is "
                        f"in logs (numeric).  String-format curves "
                        f"must be in string_data."
                    )
                if _df not in ("S", "A") and _mnem in _str_keys:
                    raise LASDataError(
                        f"LASFile: curve '{_mnem}' has "
                        f"data_format='{_df}' (numeric-format) but "
                        f"is in string_data.  Numeric-format curves "
                        f"must be in logs."
                    )

        # Run warning-producing checks via validate() (gated by _from_dict).
        if not self._from_dict:
            for issue in self.validate(complete=False):
                warnings.warn(issue, stacklevel=2)

    def validate(self, complete: bool = False) -> list[str]:
        """Validate LASFile state.

        .. note::

            This method **mutates** ``self.logs`` and ``self.string_data``
            in-place: any non-``numpy.ndarray`` values are coerced via
            ``np.asarray()`` so that dtype checks are reliable.  This
            conversion is lossless — lists and array-likes are wrapped
            into equivalent numpy arrays without changing data values.

        Args:
            complete: If True, also run deferred cross-field checks
                including children delegation, cross-section consistency,
                mandatory well fields, and semantic well checks.

        Returns:
            List of issue strings (empty = no issues found).
            Basic structural checks (type, keys, lengths) raise during
            construction and are not duplicated here.
        """
        issues: list[str] = []

        # --- index curve check (LAS 2.0) ---
        _INDEX_CURVE_ALIASES = frozenset({"DEPT", "DEPTH", "TIME", "INDEX"})
        if (
            self.curves_order
            and _LASVersionSpec(self.version.vers).is_las20
            and self.curves_order[0].upper() not in _INDEX_CURVE_ALIASES
        ):
            issues.append(
                f"LAS 2.0 spec requires the first curve to be "
                f"DEPT, DEPTH, TIME, or INDEX, but got "
                f"{self.curves_order[0]!r}.  Many real-world files "
                f"use alternative index curve names."
            )

        # --- dtype and NaN/Inf for logs ---
        for _k, _arr in self.logs.items():
            if not isinstance(_arr, np.ndarray):
                _arr = self.logs[_k] = np.asarray(_arr)
            if not np.issubdtype(_arr.dtype, np.number):
                issues.append(
                    f"LASFile: curve '{_k}' in 'logs' has non-numeric "
                    f"dtype ({_arr.dtype}).  'logs' arrays must be "
                    f"numeric."
                )
            if _arr.dtype.kind in ('f', 'c') and not np.all(np.isfinite(_arr)):
                issues.append(
                    f"LASFile: curve '{_k}' in 'logs' contains "
                    f"non-finite values (NaN/Inf)."
                )

        # --- dtype for string_data ---
        for _sk, _sarr in self.string_data.items():
            if not isinstance(_sarr, np.ndarray):
                _sarr = self.string_data[_sk] = np.asarray(_sarr)
            if np.issubdtype(_sarr.dtype, np.number):
                issues.append(
                    f"LASFile: curve '{_sk}' in 'string_data' has "
                    f"numeric dtype ({_sarr.dtype}).  'string_data' "
                    f"arrays must be non-numeric."
                )

        # --- data_format vs placement ---
        if self.curves and (self.logs or self.string_data):
            for _sc in self.curves:
                _df = _sc.data_format
                _mnem = _sc.mnemonic
                if not _df:
                    continue
                if _df == "S" or (_df == "A" and not _sc.is_array_element):
                    if _mnem in self.logs:
                        issues.append(
                            f"LASFile: curve '{_mnem}' has "
                            f"data_format='{_df}' (string-format) but "
                            f"is in logs (numeric).  String-format "
                            f"curves should be in string_data."
                        )
                else:
                    if _mnem in self.string_data:
                        issues.append(
                            f"LASFile: curve '{_mnem}' has "
                            f"data_format='{_df}' (numeric-format) "
                            f"but is in string_data.  Numeric-format "
                            f"curves should be in logs."
                        )

        # --- deferred checks (complete=True) ---
        if complete:
            # Delegate to children.
            issues.extend(self.version.validate(complete=True))
            issues.extend(self.well.validate(complete=True))
            for _cd in self.curves:
                issues.extend(_cd.validate(complete=True))
            for _pe in self.parameters:
                issues.extend(_pe.validate(complete=True))
            for _ds in self.data_sections:
                issues.extend(_ds.validate(complete=True))

            # data_sections requires LAS 3.0.
            if self.data_sections and not self.version.is_las30:
                issues.append(
                    "data_sections requires LAS 3.0 version"
                )

            # F-012: Cross-section data_sections name dedup.
            # __post_init__ (L1730-1741) checks for duplicate data section
            # names, but validate(complete=True) did not — post-construction
            # mutation followed by validate() would pass with duplicates.
            # Mirror the __post_init__ logic here as a warning-producing check.
            if len(self.data_sections) > 1:
                _ds_names: list[str] = []
                for _ds in self.data_sections:
                    _ds_names.append(_ds.name or "<unnamed>")
                _seen_ds: set[str] = set()
                for _ds_name in _ds_names:
                    if _ds_name in _seen_ds:
                        issues.append(
                            f"LASFile: duplicate data section name "
                            f"{_ds_name!r}.  Data section names must "
                            f"be unique."
                        )
                    _seen_ds.add(_ds_name)

        return issues

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
            # F-19: Warn when a parameter mnemonic was already seen —
            # dict assignment is last-wins, so earlier values are
            # silently discarded.  to_dict() previously had no duplicate
            # detection; from_dict and from_dict validation now both
            # warn about duplicates, tracking the consistent approach.
            if p.mnemonic in params_dict:
                warnings.warn(
                    f"LASFile.to_dict(): duplicate parameter mnemonic "
                    f"{p.mnemonic!r} — dict result will contain only "
                    f"the last value.  Consider deduplicating "
                    f"parameter_entry list before serialization.",
                    stacklevel=2,
                )
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
    def from_dict(
        cls, data: dict[str, Any],
        mnem_base: dict[str, str] | None = None,
    ) -> LASFile:
        """Create LASFile from dict format.

        Handles multiple format variants inherently:
        - Legacy flat dict (curves as string lists, params as {name: value} dict)
        - Detailed dict with CurveDefinition metadata (unit, api_code, description)
        - LAS 3.0 dict with array_info, data_format, data_sections, string_data
        - Mixed formats from roundtrip serialization

        Args:
            data: Dict representation of a LAS file (as produced by
                ``LASFile.to_dict()``).
            mnem_base: Optional mnemonic-to-canonical-name mapping for
                curve name normalization (same as parser's *mnem_base*).
                When provided, mnemonic names extracted from *data* are
                resolved through this mapping using case-insensitive
                lookup (matching the parser's ``_mnem_base_upper``
                pattern).  Default ``None`` means no normalization —
                mnemonics are used as-is (backward-compatible).

        The method is naturally long due to covering all these variants in a
        single backwards-compatible code path.
        """
        # F-08: Pre-try errors escape PylasdevError wrapping.  The
        # isinstance check and _validate_from_dict_input were previously
        # outside the try block — TypeErrors from those calls escaped
        # without being wrapped in LASDataError, violating the method's
        # documented contract.  Move them inside so ALL validation errors
        # get the LASDataError wrapper.
        # F-06: Deferred imports to avoid circular dependencies
        # (models.py ← parser.py/data_reader.py which import from models.py).
        from .data_reader import MAX_CURVES, MAX_DATA_LINES, MAX_TOTAL_ELEMENTS
        from .exceptions import LASDataError
        from .parser import (
            MAX_DATA_SECTIONS,
            MAX_OTHER_LINES,
            MAX_PARAMETERS,
            _validate_data_section_column_counts,
        )

        # F-057: Bound well-related iterables to match the parser's
        # MAX_DEFERRED_WELL_ENTRIES guard on the well re-processing path.
        MAX_WELL_ENTRIES = MAX_PARAMETERS

        try:
            # F-008: isinstance and validation inside try block so
            # TypeError/ValueError get wrapped in LASDataError.
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict, got {type(data).__name__}")
            # F-40: Protect caller's dict from mutation.
            data = copy.deepcopy(data)

            # F-017/F-018/IF-015/IF-026/F-019: Pre-construction validation layer.
            # Closes Pattern #7 (from_dict validation gaps) structurally.
            _validate_from_dict_input(data)

            # F-MD4-03: Build mnemonic normalization lookup when mnem_base
            # is provided.  Matches the parser's _mnem_base_upper pattern
            # (parser.py:440-450) for case-insensitive lookup + chain
            # resolution.  When mnem_base is None, _norm_mnem is an
            # identity function — backward-compatible.
            from .mnem_base import build_mnemonic_lookup

            _mnem_base_upper: dict[str, str] | None = None

            def _norm_mnem(raw: str, /) -> str:
                """Normalize a mnemonic through mnem_base lookup.

                Returns the resolved canonical name when *mnem_base* was
                provided and the uppercased *raw* string matches an entry;
                returns *raw* unchanged otherwise (identity).
                """
                if _mnem_base_upper is None:
                    return raw
                return _mnem_base_upper.get(raw.upper(), raw)

            # F-019: Use shared build_mnemonic_lookup() from mnem_base.py
            # instead of the 25-LOC inlined algorithm that was duplicated
            # between models.py and mnem_base.py.  The shared function
            # provides identical deterministic first-wins semantics with
            # chain resolution via resolve_mnemonic().
            _mnem_base_upper = build_mnemonic_lookup(mnem_base) if mnem_base else None

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
            if len(well) >= MAX_WELL_ENTRIES:
                raise ValueError(
                    f"Number of well entries ({len(well)}) exceeds maximum "
                    f"allowed ({MAX_WELL_ENTRIES})"
                )
            for key, value in well.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Well dict key must be str, got {type(key).__name__}: {key!r}"
                    )
                las_file.well[key] = _safe_str(value)
            # Restore well units if present (from v1.7+ roundtrip data)
            well_units = _resolve_dict_entry(data, "well_units", dict, dict)
            if len(well_units) >= MAX_WELL_ENTRIES:
                raise ValueError(
                    f"Number of well unit entries ({len(well_units)}) exceeds "
                    f"maximum allowed ({MAX_WELL_ENTRIES})"
                )
            for key, unit in well_units.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Well unit dict key must be str, "
                        f"got {type(key).__name__}: {key!r}"
                    )
                las_file.well.units[key] = _safe_str(unit)

            # Restore well descriptions if present (from v1.8+ roundtrip data)
            well_descriptions = _resolve_dict_entry(data, "well_descriptions", dict, dict)
            if len(well_descriptions) >= MAX_WELL_ENTRIES:
                raise ValueError(
                    f"Number of well description entries ({len(well_descriptions)}) "
                    f"exceeds maximum allowed ({MAX_WELL_ENTRIES})"
                )
            for key, desc in well_descriptions.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Well description dict key must be str, "
                        f"got {type(key).__name__}: {key!r}"
                    )
                las_file.well.descriptions[key] = _safe_str(desc)

            curves_order = data.get("curves_order", [])
            # F-21: Guard against non-list iterables.  list(string) silently
            # creates a list of single characters (e.g. "DEPT,DT,GR" → 10
            # single-char mnemonics), passing all downstream cross-validation
            # because both curves_order and curves get corrupted the same way.
            if curves_order is None:
                curves_order = []
            elif isinstance(curves_order, (str, bytes)):
                # F-110: ``isinstance(bytes, str)`` is False in Python 3,
                # so bytes bypassed the str guard.  ``list(b"GR")`` →
                # ``[71, 82]`` (integers), which propagate as curve names.
                if isinstance(curves_order, bytes):
                    raise TypeError(
                        "curves_order must be a list of strings, "
                        "got bytes.  Decode to str first: "
                        "curves_order.decode('utf-8')"
                    )
                raise ValueError(
                    f"curves_order must be a list, got str: "
                    f"{curves_order!r}"
                )
            elif isinstance(curves_order, dict):
                raise TypeError(
                    "curves_order must be a list of strings, "
                    "got dict.  Use list(curves_order) to extract "
                    "keys, or provide a list directly."
                )
            elif not isinstance(curves_order, Iterable):
                # F-M02: Non-iterable non-str types (int, float, bool) crash
                # at list() with TypeError.  Provide a clear error instead.
                raise TypeError(
                    f"curves_order must be an iterable, "
                    f"got {type(curves_order).__name__}"
                )
            # F-029: Materialize iterable to prevent generator exhaustion
            # when iterated twice (validation loop + list comprehension).
            curves_order = list(curves_order)
            # F-M-014 / F-M-016: Validate per-element types in curves_order.
            # Non-string elements (int, None) crash _norm_mnem().upper()
            # when mnem_base is active, and silently produce integer curve
            # names when mnem_base is None (e.g. curves_order=range(5)
            # passes the iterable guard but produces [0,1,2,3,4]).
            for _i, _name in enumerate(curves_order):
                if isinstance(_name, bytes):
                    raise TypeError(
                        f"curves_order[{_i}] must be str, "
                        f"got bytes.  Decode to str first: "
                        f"curves_order[{_i}].decode('utf-8')"
                    )
                if not isinstance(_name, str):
                    raise TypeError(
                        f"curves_order[{_i}] must be str, "
                        f"got {type(_name).__name__}: {_name!r}"
                    )
            las_file.curves_order = [
                _norm_mnem(name) for name in curves_order
            ]

            # Restore curve metadata if available (new format), otherwise create minimal CurveDefinition
            # F-16: Use _resolve_dict_entry — data.get("curves", []) returns
            # None when "curves" exists with value None, bypassing the default
            # and crashing at len(None).  Same pattern at 6 other sites below.
            curves_data = _resolve_dict_entry(data, "curves", list, list)
            # F-06: Resource-exhaustion guard — match parser's MAX_CURVES check.
            if len(curves_data) >= MAX_CURVES:
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
                            _base_name = _safe_str(ai.get("base_name")).upper()
                            if _base_name:
                                array_info = ArrayElementInfo(
                                    base_name=_base_name,
                                    index=_resolve_dict_entry(ai, "index", int, lambda: 0),
                                    # F2-002: Validate time_offset — int(offset) in
                                    # writer.py crashes on non-numeric values.
                                    time_offset=_resolve_dict_entry(ai, "time_offset", (int, float), lambda: None),
                                )
                        _raw_mnem = _safe_str(curve_dict.get("mnemonic", ""))
                        las_file.curves.append(
                            CurveDefinition(
                                mnemonic=_norm_mnem(_raw_mnem),
                                unit=_safe_str(curve_dict.get("unit", "")),
                                api_code=_safe_str(curve_dict.get("api_code", "")),
                                description=_safe_str(curve_dict.get("description", "")),
                                original_mnemonic=_safe_str(curve_dict.get("original_mnemonic", "")),
                                data_format=_safe_str(curve_dict.get("data_format", "")),
                                array_info=array_info,
                            )
                        )
                        # F-17: Cross-check bracket-notation mnemonic against
                        # array_info.base_name.  Only validate when the mnemonic
                        # uses array bracket notation (e.g., "NMR[1]") —
                        # non-bracket-notation curves like "CORET" with
                        # base_name="CORE" are intentionally skipped.
                        if array_info and '[' in _raw_mnem:
                            if _raw_mnem.split('[')[0] != array_info.base_name:
                                warnings.warn(
                                    f"from_dict warning: mnemonic "
                                    f"{_raw_mnem!r} uses array notation but "
                                    f"array_info.base_name is "
                                    f"{array_info.base_name!r}. "
                                    f"Cross-check mismatch may indicate "
                                    f"malformed input.",
                                    stacklevel=3,
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
                if len(curves_order) >= MAX_CURVES:
                    raise ValueError(
                        f"Number of curves ({len(curves_order)}) exceeds maximum "
                        f"allowed ({MAX_CURVES})"
                    )
                for curve_name in curves_order:
                    las_file.curves.append(CurveDefinition(mnemonic=_norm_mnem(curve_name)))

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
            if _param_count >= MAX_PARAMETERS:
                raise ValueError(
                    f"Number of parameters ({_param_count}) exceeds maximum "
                    f"allowed ({MAX_PARAMETERS})"
                )
            if isinstance(params, dict):
                # Legacy format: {mnemonic: value}
                # Check for parameter_details first to preserve full metadata
                # on roundtrip (e.g. array_index, zone, unit, description).
                param_details = data.get("parameter_details")
                if param_details is not None:
                    if len(param_details) >= MAX_PARAMETERS:
                        raise ValueError(
                            f"Number of parameter details ({len(param_details)}) exceeds maximum "
                            f"allowed ({MAX_PARAMETERS})"
                        )
                    # F2-25 + F-058 consistency: Validate as list-of-dicts via
                    # shared helper (same pattern as curves_data, section_curves,
                    # and data_sections).  Raises TypeError for non-dict elements.
                    param_details = _validate_iterable_of_dicts(param_details, "parameter_details")
                    for param_dict in param_details:
                        # F-MD4-03: Normalize mnemonic through mnem_base
                        # before construction (matching parser.py:2023).
                        param_dict["mnemonic"] = _norm_mnem(
                            _safe_str(param_dict.get("mnemonic"))
                        )
                        las_file.parameters.append(_create_parameter_entry(param_dict))
                else:
                    # Pure legacy: only params dict, no details available
                    for mnemonic, value in params.items():
                        las_file.parameters.append(
                            ParameterEntry(
                                mnemonic=_norm_mnem(mnemonic),
                                value=_safe_str(value),
                            )
                        )
            elif isinstance(params, list):
                # New format: [{"mnemonic": ..., "value": ..., ...}, ...]
                # F2-25 consistency: Shared helper validates every element is a dict
                # (previously checked inline, same as param_details and data_sections).
                _validate_iterable_of_dicts(params, "parameters")
                for param_dict in params:
                    # F-MD4-03: Normalize mnemonic through mnem_base
                    # before construction (matching parser.py:2023).
                    param_dict["mnemonic"] = _norm_mnem(
                        _safe_str(param_dict.get("mnemonic"))
                    )
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
            # F-I2-M33: Count lines not characters (matching parser behavior).
            # The parser counts len(list[str]) — individual lines appended to
            # _other_lines.  from_dict previously counted len(str) — characters
            # in the joined other-section string.  Using splitlines() aligns
            # the guard: a file accepted by parse() is accepted by from_dict().
            _other_str = _safe_str(data.get("other"), "")
            _other_line_count = len(_other_str.splitlines())
            if _other_line_count >= MAX_OTHER_LINES:
                raise ValueError(
                    f"Other section line count ({_other_line_count}) exceeds "
                    f"maximum allowed ({MAX_OTHER_LINES})"
                )
            las_file.other = _safe_str(data.get("other"), "")
            las_file.encoding = _safe_str(data.get("encoding"), "utf-8")
            las_file.source_file = _safe_str(data.get("source_file"), "")

            # Restore LAS 3.0 data sections
            ds_data = _resolve_dict_entry(data, "data_sections", list, list)
            # F-06: Resource-exhaustion guard for data sections.
            if len(ds_data) >= MAX_DATA_SECTIONS:
                raise ValueError(
                    f"Number of data sections ({len(ds_data)}) exceeds maximum "
                    f"allowed ({MAX_DATA_SECTIONS})"
                )
            # F-058: Validate every element is a dict.  Previously non-dict
            # elements were silently skipped with ``continue`` while two sibling
            # paths (parameter_details and params) both raised TypeError.
            # Using the shared helper also adds a missing list-type check.
            ds_data = _validate_iterable_of_dicts(ds_data, "data_sections")
            # I2F-15: Cross-section cumulative allocation counter.
            # Each section independently passes MAX_TOTAL_ELEMENTS checks,
            # but there is no running total across ALL sections.  1,000
            # sections * 10M elements = 10B elements (~80 GB) passes all
            # per-section guards.  Track cumulative elements to prevent
            # multi-section allocation DoS.
            _cumulative_elements = 0
            for ds_dict in ds_data:
                ds_string_data = {}
                _ds_string_raw = _resolve_dict_entry(ds_dict, "string_data", dict, dict)
                # F-24: Per-section string_data entry count guard.  Every other
                # iterable dict in from_dict() has a count guard (curves_data →
                # MAX_CURVES, ds_data → MAX_CURVES, logs → MAX_CURVES, etc.);
                # string_data was the sole unguarded iterable.
                if len(_ds_string_raw) >= MAX_CURVES:
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
                    # F-MD4-02: Pre-allocation size check (same pattern as
                    # ds_data numeric path above).  np.array() allocates
                    # before the downstream len() guard — a huge list
                    # triggers MemoryError before the guard catches it.
                    if hasattr(arr, '__len__') and len(arr) > MAX_DATA_LINES:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"String data array length ({len(arr)}) for "
                            f"'{name}' in section '{ds_name}' exceeds "
                            f"maximum allowed ({MAX_DATA_LINES})"
                        )
                    # F-I2-MD4-01: Wrap in try/except MemoryError — the
                    # numeric ds_data path (line ~1616) has this; the
                    # string_data path was missing it.  A huge list can
                    # trigger MemoryError from np.array() before the
                    # downstream len() guard fires.
                    try:
                        name = _norm_mnem(name)
                        ds_string_data[name] = np.atleast_1d(np.array(arr, dtype=object))
                    except (ValueError, TypeError, MemoryError, OverflowError) as e:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Cannot convert string data for section "
                            f"'{ds_name}', curve '{name}': {e}"
                        ) from e
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
                if len(_sc_raw) >= MAX_CURVES:
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
                        _sc_base_name = _safe_str(ai.get("base_name")).upper()
                        if _sc_base_name:
                            sc_array_info = ArrayElementInfo(
                                base_name=_sc_base_name,
                                index=_resolve_dict_entry(ai, "index", int, lambda: 0),
                                # F2-002: Validate time_offset — int(offset) in
                                # writer.py crashes on non-numeric values.
                                time_offset=_resolve_dict_entry(ai, "time_offset", (int, float), lambda: None),
                            )
                    _sc_raw_mnem = _safe_str(sc_dict.get("mnemonic", ""))
                    ds_section_curves.append(
                        CurveDefinition(
                            mnemonic=_norm_mnem(_sc_raw_mnem),
                            unit=_safe_str(sc_dict.get("unit", "")),
                            api_code=_safe_str(sc_dict.get("api_code", "")),
                            description=_safe_str(sc_dict.get("description", "")),
                            original_mnemonic=_safe_str(sc_dict.get("original_mnemonic", "")),
                            data_format=_safe_str(sc_dict.get("data_format", "")),
                            array_info=sc_array_info,
                        )
                    )
                    # F-17: Cross-check bracket-notation mnemonic against
                    # array_info.base_name (per-section curves).  Same logic
                    # as top-level curves above.
                    if sc_array_info and '[' in _sc_raw_mnem:
                        if _sc_raw_mnem.split('[')[0] != sc_array_info.base_name:
                            warnings.warn(
                                f"from_dict warning: mnemonic "
                                f"{_sc_raw_mnem!r} uses array notation but "
                                f"array_info.base_name is "
                                f"{sc_array_info.base_name!r}. "
                                f"Cross-check mismatch may indicate "
                                f"malformed input.",
                                stacklevel=3,
                            )
                ds_data_raw = _resolve_dict_entry(ds_dict, "data", dict, dict)
                # F2-21: Per-section entry count guard.  Outer MAX_DATA_SECTIONS
                # guards section count; per-array MAX_DATA_LINES guards element
                # count.  Per-section curve entry count was unguarded — 1 section
                # x 200K single-element arrays passes all existing guards.
                if len(ds_data_raw) >= MAX_CURVES:
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
                    # F-012: Pre-allocation size check.  np.array() allocates
                    # BEFORE the downstream len() guard — a huge list triggers
                    # MemoryError (or system OOM) before the guard catches it.
                    # Check len() before allocation when possible.
                    if hasattr(v, '__len__') and len(v) > MAX_DATA_LINES:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Array length ({len(v)}) for curve '{k}' in "
                            f"section '{ds_name}' exceeds maximum allowed "
                            f"({MAX_DATA_LINES})"
                        )
                    try:
                        k = _norm_mnem(k)
                        ds_data[k] = np.atleast_1d(np.array(v, dtype=np.float64))
                    except (ValueError, TypeError, MemoryError, OverflowError) as e:
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
                # F-I2-M20: Cross-validate data and string_data key sets within
                # the same DataSection.  When a curve name appears in both dicts,
                # the writer silently discards one (string_data wins — writer.py
                # line 703 checks string_data first).  Either way, data is lost.
                # Reject the input so callers get a clear error instead of silent
                # data discard on the roundtrip path.
                if ds_data and ds_string_data:
                    _colliding = set(ds_data.keys()) & set(ds_string_data.keys())
                    if _colliding:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Section '{ds_name}': curve(s) {sorted(_colliding)} appear "
                            f"in both 'data' and 'string_data'.  Each curve must appear "
                            f"in exactly one collection."
                        )
                _ds_curves_order = ds_dict.get("curves_order", [])
                # F2-22: Guard against non-list iterables for per-section
                # curves_order — same bug as top-level F-21.
                if _ds_curves_order is None:
                    _ds_curves_order = []
                elif isinstance(_ds_curves_order, (str, bytes)):
                    # F-110: Same bytes bypass as top-level curves_order
                    # guard (line 808).  ``list(b"GR")`` → ``[71, 82]``
                    # produces integer curve names in a data section.
                    ds_name = ds_dict.get("name", "<unknown>")
                    if isinstance(_ds_curves_order, bytes):
                        raise TypeError(
                            f"curves_order in section '{ds_name}' must be "
                            f"a list of strings, got bytes.  Decode to "
                            f"str first: curves_order.decode('utf-8')"
                        )
                    raise ValueError(
                        f"curves_order in section '{ds_name}' must be a list, "
                        f"got str: {_ds_curves_order!r}"
                    )
                # F-I2E-05: Validate per-element types in per-section
                # curves_order.  Container type is guarded (None / str / bytes
                # above), but individual elements are not.  Non-string values
                # (int, None, float) survive when section_curves is empty
                # because the mnemonic cross-validation gate (L1496) is
                # inactive.  str(123) → "123" becomes a column header on
                # write; on re-read it looks like a genuine curve.
                # F-029: Materialize to prevent generator exhaustion
                # when iterated twice (validation loop + list comprehension).
                _ds_curves_order = list(_ds_curves_order)
                for _i, _item in enumerate(_ds_curves_order):
                    if not isinstance(_item, (str, bytes)):
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise TypeError(
                            f"curves_order[{_i}] in section '{ds_name}' "
                            f"must be str, got {type(_item).__name__}: "
                            f"{_item!r}"
                        )
                # F-MD4-03: Normalize curves_order entries through
                # mnem_base (matching parser's per-section normalization).
                _ds_curves_order = [
                    _norm_mnem(_item) for _item in _ds_curves_order
                ]
                # F-I2-M19: Detect duplicate curve names in section.  The parser
                # calls _deduplicate_curves() at data-read time to rename
                # collisions (append _2, _3 suffixes).  from_dict had zero dedup
                # calls, so duplicates passed silently — the pairwise zip
                # cross-validation (F-23 below) naturally pairs identical names
                # at matching positions.  Raise a clear error instead of silently
                # accepting duplicate curves.
                if _ds_curves_order:
                    _seen_n: set[str] = set()
                    for _n in _ds_curves_order:
                        if _n in _seen_n:
                            ds_name = ds_dict.get("name", "<unknown>")
                            raise ValueError(
                                f"Duplicate curve name {_n!r} in section '{ds_name}' "
                                f"curves_order.  Curve names must be unique within "
                                f"a data section."
                            )
                        _seen_n.add(_n)
                # Cross-validate string_data and data keys against
                # curves_order.  Keys in either dict that do not appear
                # in curves_order are orphaned — the writer silently
                # drops them, producing data loss on roundtrip.
                _curve_k = set(_ds_curves_order) if _ds_curves_order else set()
                if ds_string_data:
                    _str_orphaned = set(ds_string_data.keys()) - _curve_k
                    if _str_orphaned:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"string_data in section '{ds_name}' contains "
                            f"keys not in curves_order: {sorted(_str_orphaned)}. "
                            f"Each string_data key must correspond to a curve "
                            f"mnemonic."
                        )
                if ds_data:
                    _num_orphaned = set(ds_data.keys()) - _curve_k
                    if _num_orphaned:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"data in section '{ds_name}' contains "
                            f"keys not in curves_order: {sorted(_num_orphaned)}. "
                            f"Each data key must correspond to a curve mnemonic."
                        )
                # F-M-036: Detect uncovered curves in from_dict path.
                # The above checks detect orphaned data (data_keys - curve_set)
                # but not the reverse (curve_set - data_keys - string_keys).
                # Uncovered curves are recoverable (writer pads null_value)
                # so emit a warning rather than raising.
                _num_keys = set(ds_data.keys()) if ds_data else set()
                _str_keys = set(ds_string_data.keys()) if ds_string_data else set()
                if _curve_k:
                    _uncovered = _curve_k - _num_keys - _str_keys
                    if _uncovered:
                        ds_name = ds_dict.get("name", "<unknown>")
                        warnings.warn(
                            f"Section '{ds_name}': curve(s) "
                            f"{sorted(_uncovered)} appear in curves_order "
                            f"but have no data in 'data' or "
                            f"'string_data'.  The writer will pad these "
                            f"curves with null_value.",
                            stacklevel=2,
                        )
                if ds_section_curves:
                    _sc_mnemonics = [sc.mnemonic for sc in ds_section_curves]
                    _seen_m: set[str] = set()
                    for _m in _sc_mnemonics:
                        if _m in _seen_m:
                            ds_name = ds_dict.get("name", "<unknown>")
                            raise ValueError(
                                f"Duplicate mnemonic {_m!r} in section '{ds_name}' "
                                f"section_curves.  Curve mnemonics must be unique "
                                f"within a data section."
                            )
                        _seen_m.add(_m)
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
                if len(_ds_curves_order) >= MAX_CURVES:
                    ds_name = ds_dict.get("name", "<unknown>")
                    raise ValueError(
                        f"Number of curves_order entries ({len(_ds_curves_order)}) "
                        f"in section '{ds_name}' exceeds maximum allowed "
                        f"({MAX_CURVES})"
                    )
                # I2F-15: Cumulative cross-section allocation check.
                # Each section's per-section total is validated above, but
                # multiple sections can collectively exceed MAX_TOTAL_ELEMENTS.
                # Compute this section's element total and check against the
                # running cumulative count BEFORE allocating the DataSection.
                _section_total = 0
                if ds_data:
                    _section_total += (
                        len(ds_data) * max(len(arr) for arr in ds_data.values())
                    )
                if ds_string_data:
                    _section_total += (
                        len(ds_string_data)
                        * max(len(arr) for arr in ds_string_data.values())
                    )
                if _cumulative_elements + _section_total > MAX_TOTAL_ELEMENTS:
                    ds_name = ds_dict.get("name", "<unknown>")
                    raise ValueError(
                        f"Cumulative cross-section allocation "
                        f"({_cumulative_elements}+{_section_total} = "
                        f"{_cumulative_elements + _section_total} elements) "
                        f"in section '{ds_name}' exceeds maximum allowed "
                        f"({MAX_TOTAL_ELEMENTS})"
                    )
                _cumulative_elements += _section_total
                ds = DataSection(
                    name=_safe_str(ds_dict.get("name"), ""),
                    section_type=_safe_str(ds_dict.get("section_type"), "LOG_DATA"),
                    curves_order=list(_ds_curves_order),
                    data=ds_data,
                    string_data=ds_string_data,
                    section_curves=ds_section_curves,
                    _from_dict=True,
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
                # F-037: Cross-group row-count consistency.
                # The within-group checks above validate that all arrays
                # in 'data' have the same length, and all arrays in
                # 'string_data' have the same length — but no cross-check
                # verifies that data rows and string_data rows match.
                # A DataSection with 100-row numeric data and 50-row
                # string_data passes all existing validation.  The writer's
                # ``_format_data_rows`` uses ``max()`` across all arrays,
                # so one group's shorter arrays get padded — producing
                # semantically incorrect output.
                if ds.data and ds.string_data:
                    _data_rows = max(len(arr) for arr in ds.data.values())
                    _string_rows = max(len(arr) for arr in ds.string_data.values())
                    if _data_rows != _string_rows:
                        raise ValueError(
                            f"DataSection '{ds.name}': data row count "
                            f"({_data_rows}) does not match string_data "
                            f"row count ({_string_rows})"
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

            # F-020: Wire _validate_data_section_column_counts() into the
            # from_dict path.  The function was extracted from the parser
            # specifically so that from_dict could also call it (per its
            # docstring), but the import and call were never added.
            if las_file.data_sections:
                _validate_data_section_column_counts(las_file.data_sections)

            # Restore LAS 3.0 string data (top-level, backward compat
            # with data serialized before string_data was moved to
            # per-section DataSection objects).
            sd = _resolve_dict_entry(data, "string_data", dict, dict)
            # F-24: Top-level string_data entry count guard — same gap as
            # the per-section path fixed above.
            if len(sd) >= MAX_CURVES:
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
                # F-MD4-02: Pre-allocation size check (same pattern as
                # numeric logs path).  np.array() allocates before the
                # downstream len() guard — a huge list triggers MemoryError
                # before the guard catches it.
                if hasattr(arr, '__len__') and len(arr) > MAX_DATA_LINES:
                    raise ValueError(
                        f"String data array length ({len(arr)}) for "
                        f"'{name}' exceeds maximum allowed "
                        f"({MAX_DATA_LINES})"
                    )
                # F-I2-MD4-01: Wrap in try/except MemoryError — the
                # numeric logs path (line ~1952) has this; string_data
                # was missing it.  A huge list can trigger MemoryError
                # from np.array() before the downstream len() guard fires.
                try:
                    name = _norm_mnem(name)
                    las_file.string_data[name] = np.atleast_1d(np.array(arr, dtype=object))
                except (ValueError, TypeError, MemoryError, OverflowError) as e:
                    raise ValueError(
                        f"Cannot convert string data for curve "
                        f"'{name}' to string array: {e}"
                    ) from e
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

            # F2-11: Validate string_data keys against curves_order
            # (matching per-section guard at lines 1995-2005).
            # Keys in string_data not in curves_order are orphaned.
            if las_file.string_data and las_file.curves_order:
                _order_s = set(las_file.curves_order)
                _str_orphaned = set(las_file.string_data.keys()) - _order_s
                if _str_orphaned:
                    raise ValueError(
                        f"string_data contains keys not in "
                        f"curves_order: {sorted(_str_orphaned)}. "
                        f"Each string_data key must correspond to a "
                        f"curve mnemonic."
                    )

            logs = _resolve_dict_entry(data, "logs", dict, dict)
            # F-06: Resource-exhaustion guard for logs.
            if len(logs) >= MAX_CURVES:
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
                # F-012: Pre-allocation size check (same pattern as ds_data
                # above).  np.array() allocates before the downstream len()
                # guard catches oversized inputs.
                if hasattr(arr, '__len__') and len(arr) > MAX_DATA_LINES:
                    raise ValueError(
                        f"Array length ({len(arr)}) for log "
                        f"'{name}' exceeds maximum allowed "
                        f"({MAX_DATA_LINES})"
                    )
                try:
                    name = _norm_mnem(name)
                    las_file.logs[name] = np.atleast_1d(np.array(arr, dtype=np.float64))
                except (ValueError, TypeError, MemoryError, OverflowError) as e:
                    raise ValueError(
                        f"Cannot convert log data for curve '{name}' to numeric array: {e}"
                    ) from e
                # F-M02: Per-array size guard for log arrays.
                if len(las_file.logs[name]) > MAX_DATA_LINES:
                    raise ValueError(
                        f"Array length ({len(las_file.logs[name])}) for log "
                        f"'{name}' exceeds maximum allowed ({MAX_DATA_LINES})"
                    )

            # F-11: Detect key overlap between logs and string_data.
            # A curve name appearing in both would silently have data
            # corrupted — one array overwrites the other's meaning.
            # (matches __post_init__ guard at lines 1209-1216)
            if las_file.logs and las_file.string_data:
                _overlap = set(las_file.logs.keys()) & set(las_file.string_data.keys())
                if _overlap:
                    raise ValueError(
                        f"Curves {sorted(_overlap)} appear in "
                        f"both logs and string_data.  Each curve may "
                        f"only be stored in one location."
                    )

            # F-011: Validate that log curve keys match curves_order exactly.
            # The length check above ensures count matches; this catches phantom
            # keys (extra curves in logs not in curves_order) and missing keys.
            # Only valid for legacy LAS 1.2/2.0 files where all curve data lives
            # in the logs dict.  LAS 3.0 files distribute curve data across
            # data_sections and string_data, so curves_order typically includes
            # curves whose data is in those sections, not in logs.
            if las_file.logs and not las_file.data_sections:
                # G-015: Validate all log keys are strings before
                # _norm_mnem normalisation, mirroring the well section
                # key validation pattern (lines 1428-1433).
                for _lk in las_file.logs:
                    if not isinstance(_lk, str):
                        raise TypeError(
                            f"Log dict key must be str, "
                            f"got {type(_lk).__name__}: {_lk!r}"
                        )
                # F-M-013: Normalize log keys through mnem_base before
                # comparison.  curves_order is already normalized at
                # construction (line ~1458), but logs dict uses raw keys.
                # Without normalization, mnem_base-active environments
                # produce false "Extra keys / Missing keys" errors when
                # semantically-equivalent keys differ in case.
                _log_keys = {_norm_mnem(k) for k in las_file.logs.keys()}
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
            # F-I2-M19: Detect duplicate curve names at the top level
            # for legacy (non-LAS-3.0) files.  LAS 3.0 files distribute
            # curve data across multiple data_sections — the same curve
            # name like "DEPT" legitimately appears in every section's
            # curves_order, producing duplicates in the top-level list.
            # The parser calls _deduplicate_curves() before data allocation
            # for legacy files; from_dict rejects duplicates for these.
            if las_file.curves_order and not las_file.data_sections:
                _top_seen: set[str] = set()
                for _n in las_file.curves_order:
                    if _n in _top_seen:
                        raise ValueError(
                            f"Duplicate curve name {_n!r} in top-level "
                            f"curves_order.  Curve names must be unique."
                        )
                    _top_seen.add(_n)

            # G-008: Cross-group row count validation between logs and
            # string_data.  Within-group row checks exist (above); this
            # catches mismatched row counts across the two groups.  The
            # per-section equivalent is in DataSection.__post_init__
            # (lines 1015-1023) — same pattern.
            if las_file.logs and las_file.string_data:
                _max_log_rows = max(len(arr) for arr in las_file.logs.values())
                _max_str_rows = max(len(arr) for arr in las_file.string_data.values())
                if _max_log_rows != _max_str_rows:
                    raise ValueError(
                        f"Logs row count ({_max_log_rows}) does not match "
                        f"string_data row count ({_max_str_rows})"
                    )

            # I2F-13: Re-run all __post_init__ validations on the
            # fully-populated object.  __post_init__ skips checks
            # when collections are empty (incremental construction),
            # so from_dict must re-validate after populating everything.
            # The _from_dict flag suppresses warnings that were
            # appropriate at direct-construction time but are noise
            # during from_dict re-validation.
            las_file._from_dict = True
            try:
                las_file.__post_init__()
                # Run deferred validate(complete=True) while _from_dict
                # suppresses redundant warnings.
                _complete_issues = las_file.validate(complete=True)
                for issue in _complete_issues:
                    warnings.warn(issue, stacklevel=2)
            finally:
                las_file._from_dict = False

            return las_file
        except (ValueError, TypeError, OverflowError) as e:
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


# F-015: Validating dict wrapper for DevFile.columns.  Direct mutation
# like ``dev.columns["NEW"] = arr`` bypasses all validation when columns
# is a plain dict — no __setitem__ guard, no column_order sync, no
# length consistency check.  This wrapper intercepts __setitem__ and
# __delitem__ to validate, sync column_order, and enforce resource limits.
class _DevColumns(dict[str, NDArray[np.float64]]):
    """Validating dict for DevFile columns that intercepts mutation."""

    __slots__ = ('_dev',)

    def __init__(self, dev: DevFile, mapping: dict[str, NDArray[np.float64]] | None = None, /, **kwargs: Any) -> None:
        self._dev = dev
        super().__init__(mapping or {}, **kwargs)

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError(
                f"DevFile column keys must be str, "
                f"got {type(key).__name__}"
            )
        # Convert to numpy array matching from_dict behaviour.
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        # Validate length consistency with existing columns.
        if self:
            existing_len = len(next(iter(self.values())))
            if len(arr) != existing_len:
                raise ValueError(
                    f"DevFile: column '{key}' has length {len(arr)} "
                    f"but existing columns have length {existing_len}"
                )
        # F-016: Per-column size guard mirrors from_dict guards.
        from .data_reader import MAX_DATA_LINES
        if len(arr) > MAX_DATA_LINES:
            raise ValueError(
                f"DevFile: column '{key}' length ({len(arr)}) "
                f"exceeds maximum allowed ({MAX_DATA_LINES})"
            )
        super().__setitem__(key, arr)
        # Sync column_order — add key if not already present.
        if key not in self._dev.column_order:
            self._dev.column_order.append(key)

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        if key in self._dev.column_order:
            self._dev.column_order.remove(key)


@dataclass(eq=False)
class DevFile:
    """DEV (deviation survey) file data structure."""

    columns: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    column_order: list[str] = field(default_factory=list)
    source_file: str = ""
    encoding: str = "utf-8"

    # I2F-01: from_dict re-validation control flag.  When True,
    # __post_init__ suppresses the validate(complete=True) call —
    # the from_dict path calls validate() explicitly after full
    # population.  Consistent with LASFile._from_dict and
    # DataSection._from_dict.
    _from_dict: bool = field(default=False, repr=False)

    def validate(self, complete: bool = False) -> list[str]:
        """Validate DEV file data integrity.

        Args:
            complete: If True, run deferred data-quality checks
                (NaN/Inf in numeric columns, MD monotonicity,
                AZI/INC range validation).

        Returns:
            List of issue strings (empty = no issues found).
            Basic structural checks (key matching, consistent lengths)
            raise during construction and are not duplicated here.
        """
        issues: list[str] = []
        if not complete:
            return issues

        # NaN/Inf check for numeric column arrays.
        for _col_name, _col_data in self.columns.items():
            if isinstance(_col_data, np.ndarray) and _col_data.dtype.kind in ('f', 'c'):
                if not np.all(np.isfinite(_col_data)):
                    issues.append(
                        f"DevFile: column '{_col_name}' contains "
                        f"non-finite values (NaN/Inf)."
                    )

        # Data-quality validation (MD monotonicity, AZI/INC range).
        for _col_name, _col_data in self.columns.items():
            if _col_data is None or len(_col_data) == 0:
                continue
            _col_upper = _col_name.upper()
            # MD: monotonicity
            if _col_upper == "MD":
                _finite = _col_data[np.isfinite(_col_data)]
                if len(_finite) >= 2:
                    _diffs = np.diff(_finite)
                    if np.any(_diffs < 0):
                        _n_bad = int(np.sum(_diffs < 0))
                        issues.append(
                            f"MD values are not monotonically "
                            f"increasing: {_n_bad} decrease(s) "
                            f"found.  Unsorted MD values can "
                            f"cause inaccurate trajectory "
                            f"calculations."
                        )
            # AZI: range [0, 360]
            if _col_upper in ("AZI", "AZIM", "AZ", "AZM", "AZIMUTH"):
                _finite = _col_data[np.isfinite(_col_data)]
                if len(_finite) > 0:
                    _oor = (_finite < 0) | (_finite > 360)
                    if np.any(_oor):
                        _n_bad = int(np.sum(_oor))
                        _bad_vals = _finite[_oor][:3]
                        _extra = "..." if _n_bad > 3 else ""
                        issues.append(
                            f"Azimuth column '{_col_name}' has "
                            f"{_n_bad} value(s) outside [0, 360]: "
                            f"{list(_bad_vals)}{_extra}. "
                            f"Azimuth values outside [0, 360] "
                            f"can cause inaccurate trajectory "
                            f"calculations."
                        )
            # INC: range [0, 180]
            if _col_upper in ("INC", "INCL", "DEVI", "DIP"):
                _finite = _col_data[np.isfinite(_col_data)]
                if len(_finite) > 0:
                    _oor = (_finite < 0) | (_finite > 180)
                    if np.any(_oor):
                        _n_bad = int(np.sum(_oor))
                        _bad_vals = _finite[_oor][:3]
                        _extra = "..." if _n_bad > 3 else ""
                        issues.append(
                            f"Inclination column '{_col_name}' "
                            f"has {_n_bad} value(s) outside "
                            f"[0, 180]: {list(_bad_vals)}{_extra}. "
                            f"Inclination values outside [0, 180] "
                            f"can cause inaccurate trajectory "
                            f"calculations."
                        )

        return issues

    def __post_init__(self) -> None:
        """Validate critical invariants after construction (E-F-026).

        Direct DevFile construction previously bypassed all validation.
        This catches invalid state that would cause silent data loss
        downstream.  Empty construction is allowed for incremental
        population.
        """
        # F-015: Wrap columns in _DevColumns if not already wrapped.
        # This intercepts `dev.columns["KEY"] = arr` mutations so they
        # go through validation, column_order sync, and length checks.
        # Re-wrapping is idempotent — if columns is already a
        # _DevColumns, the isinstance guard skips re-initialisation.
        if not isinstance(self.columns, _DevColumns):
            self.columns = _DevColumns(self, self.columns)

        if not self.columns:
            return

        from .exceptions import LASDataError

        # column_order must match columns keys exactly
        # I2F-024: Reject duplicate entries in column_order.
        # The set() comparison below cannot detect duplicates
        # (set(["MD","MD","TVD"]) == {"MD","TVD"} is True).
        # Duplicates indicate a caller bug and would cause
        # double-emission downstream.
        if len(self.column_order) != len(set(self.column_order)):
            _dupes = sorted(
                c for c in set(self.column_order)
                if self.column_order.count(c) > 1
            )
            raise LASDataError(
                f"DevFile: column_order contains duplicate entries: "
                f"{_dupes}.  Each column may only appear once."
            )
        _col_keys = set(self.columns.keys())
        _ord_keys = set(self.column_order)
        if _col_keys != _ord_keys:
            raise LASDataError(
                f"DevFile: column_order and columns keys do not "
                f"match.  column_order has {sorted(_ord_keys)}, "
                f"columns has {sorted(_col_keys)}."
            )

        # All columns must have the same array length
        if len(self.columns) > 1:
            _col_lens = {
                name: len(arr) for name, arr in self.columns.items()
            }
            if len(set(_col_lens.values())) > 1:
                raise LASDataError(
                    f"DevFile: columns have inconsistent array "
                    f"lengths: {_col_lens}"
                )

        # F-016: Resource-exhaustion guards for direct construction.
        # The from_dict path has MAX_CURVES, MAX_DATA_LINES, and
        # MAX_TOTAL_ELEMENTS guards (imported from .data_reader);
        # __post_init__ previously had none — direct construction with
        # 200+ columns or multi-MB arrays bypassed all limits.
        from .data_reader import MAX_CURVES, MAX_DATA_LINES, MAX_TOTAL_ELEMENTS

        if len(self.columns) >= MAX_CURVES:
            raise LASDataError(
                f"DevFile: number of columns ({len(self.columns)}) "
                f"exceeds maximum allowed ({MAX_CURVES})"
            )

        _max_dev_len = max(len(arr) for arr in self.columns.values())
        if _max_dev_len > MAX_DATA_LINES:
            raise LASDataError(
                f"DevFile: maximum column length ({_max_dev_len}) "
                f"exceeds maximum allowed ({MAX_DATA_LINES})"
            )

        _dev_total = len(self.columns) * _max_dev_len
        if _dev_total > MAX_TOTAL_ELEMENTS:
            raise LASDataError(
                f"DevFile: total elements ({len(self.columns)} columns x "
                f"{_max_dev_len} rows = {_dev_total}) exceeds maximum "
                f"allowed ({MAX_TOTAL_ELEMENTS})"
            )

        # I2F-01: Run data-quality validation checks from __post_init__
        # when constructed directly (not via from_dict).  Matches the
        # pattern used by DataSection and LASFile.  from_dict sets
        # _from_dict=True and calls validate(complete=True) explicitly
        # after full population, avoiding double validation.
        if not self._from_dict:
            for issue in self.validate(complete=True):
                warnings.warn(issue, stacklevel=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to legacy dict format, including file metadata.

        Metadata keys (source_file, encoding, column_order) are stored in
        the flat dict alongside column data.  When a column name collides
        with a metadata key, the metadata is stored under a ``_meta_``
        prefix to avoid silently overwriting column data (I2F-28).
        """
        result: dict[str, Any] = {k: v.copy() for k, v in self.columns.items()}
        # I2F-28: Detect column name collisions with metadata keys.
        # Without this check, a column named "source_file", "encoding", or
        # "column_order" is silently overwritten by metadata — and on
        # roundtrip through read_dev_file() the key is stripped entirely
        # (dev_reader.py:522-524), producing double data loss.
        _metadata_assignments = {
            "source_file": self.source_file,
            "encoding": self.encoding,
            "column_order": list(self.column_order),
        }
        for mk, mv in _metadata_assignments.items():
            if mk in self.columns:
                warnings.warn(
                    f"DevFile column name '{mk}' collides with metadata key — "
                    f"storing metadata as '_meta_{mk}' to avoid data loss. "
                    f"Column data is preserved unchanged.",
                    stacklevel=2,
                )
                result[f"_meta_{mk}"] = mv
            else:
                result[mk] = mv
        return result

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], normalize_aliases: bool = True,
    ) -> DevFile:
        """Create DevFile from dict (reverse of to_dict).

        Args:
            data: Flat dict mapping column names to array-like values.
            normalize_aliases: If True (default), normalize column names
                through ``_DEV_ALIASES``.  Set to False to preserve
                column names exactly as provided.

        Returns:
            DevFile with columns populated from the dict.
        """
        # F-008: isinstance check moved inside try block so TypeError
        # gets wrapped in LASDataError, matching LASFile.from_dict's
        # documented contract.
        from .data_reader import MAX_CURVES, MAX_DATA_LINES, MAX_TOTAL_ELEMENTS
        from .exceptions import LASDataError

        try:
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict, got {type(data).__name__}")
            # F2-12: Protect caller's dict from mutation.
            data = copy.deepcopy(data)

            # I2F-01: Suppress double validation — __post_init__ gates
            # validate(complete=True) behind _from_dict=False.  from_dict
            # calls validate() explicitly after full population.
            dev = cls(_from_dict=True)
            # Separate column arrays from metadata keys
            metadata_keys = {"encoding", "source_file", "column_order"}

            # G-007/G-016: Validate all dict keys are strings before
            # .startswith() is called in the list comprehension below.
            # Non-str keys (int, tuple) raise AttributeError which
            # escapes the except (ValueError, TypeError) wrapper.
            for _k in data:
                if not isinstance(_k, str):
                    raise LASDataError(
                        f"DevFile.from_dict: column keys must be "
                        f"strings, got {type(_k).__name__}"
                    )
            # F-20: Normalize column names through _DEV_ALIASES before
            # storage.  Controlled by normalize_aliases parameter.
            if normalize_aliases:
                from .dev_reader import _normalize_dev_column
                # I2F-011: Build a normalization map first, then
                # rebuild the data dict in original key order so
                # renamed columns stay in their original positions.
                # The old pop+append pattern (data[norm]=data.pop(raw))
                # moved renamed columns to the dict end, corrupting
                # column_order inference.
                _norm_map: dict[str, str] = {}
                for _raw_key in data:
                    # F-M-G03: Don't normalize metadata keys — they are
                    # checked case-sensitively against metadata_keys below.
                    # Normalizing e.g. "encoding" → "ENCODING" would cause
                    # the metadata check to fail and the value to be
                    # treated as column data.
                    if _raw_key in metadata_keys:
                        continue
                    # Also skip _meta_ prefixed keys — normalization
                    # uppercases them (e.g. _meta_source_file → _META_SOURCE_FILE)
                    # and the startswith("_meta_") check below is case-sensitive.
                    if _raw_key.startswith("_meta_"):
                        continue
                    _norm_key = _normalize_dev_column(_raw_key)
                    if _norm_key != _raw_key:
                        # Collision: two raw keys normalise to the same value
                        if _norm_key in _norm_map.values():
                            raise ValueError(
                                f"DevFile.from_dict: column name collision: "
                                f"'{_raw_key}' and another key normalise "
                                f"to the same canonical column name '{_norm_key}'"
                            )
                        # Collision: normalised key already exists as a
                        # raw key that itself does not normalise away
                        if _norm_key in data and _norm_key not in _norm_map:
                            raise ValueError(
                                f"DevFile.from_dict: column name collision: "
                                f"'{_raw_key}' and '{_norm_key}' normalize "
                                f"to the same canonical column name"
                            )
                        _norm_map[_raw_key] = _norm_key
                # Rebuild data dict with normalized keys, preserving
                # original insertion order (the order of the caller's
                # input dict).
                if _norm_map:
                    data = {
                        _norm_map.get(k, k): v
                        for k, v in data.items()
                    }

            # F-M01: Resource-exhaustion guard — bound column count.
            # Exclude _meta_-prefixed keys (metadata stored under prefix
            # to avoid column-name collisions — I2F-28); they are not
            # columns and must not count toward the column limit.
            # R7F-01-gap: When both a bare metadata key (e.g. "source_file")
            # and its _meta_-prefixed counterpart exist, the bare key is
            # column data (the _meta_ key carries the real metadata).
            # Count the bare key as a column in the collision case.
            _column_keys = [
                k for k in data
                if not k.startswith("_meta_")
                and (k not in metadata_keys or f"_meta_{k}" in data)
            ]
            if len(_column_keys) >= MAX_CURVES:
                raise ValueError(
                    f"Number of columns ({len(_column_keys)}) exceeds maximum "
                    f"allowed ({MAX_CURVES})"
                )

            for key, value in data.items():
                # R7F-01: _meta_ prefix roundtrip.  When to_dict detects a
                # column name collision with a metadata key (I2F-28), it
                # stores metadata under ``_meta_``-prefixed keys (e.g.,
                # ``_meta_source_file``).  from_dict must recognise and
                # reverse that prefix so the roundtrip contract is preserved.
                if key.startswith("_meta_"):
                    real_key = key[6:]  # strip ``_meta_`` prefix
                    _recognised = True
                    if real_key == "encoding":
                        dev.encoding = _safe_str(value, "utf-8")
                    elif real_key == "source_file":
                        dev.source_file = _safe_str(value)
                    elif real_key == "column_order":
                        if value is None:
                            dev.column_order = []
                        elif isinstance(value, (str, bytes)):
                            # F-110: bytes bypasses the str guard in Python 3.
                            if isinstance(value, bytes):
                                raise TypeError(
                                    "column_order must be a list of strings, "
                                    "got bytes.  Decode to str first: "
                                    "value.decode('utf-8')"
                                )
                            dev.column_order = [value]
                            # F2-31: Normalize column_order entries to match
                            # normalized column names in dev.columns.
                            if normalize_aliases:
                                dev.column_order = [
                                    _normalize_dev_column(c)
                                    for c in dev.column_order
                                ]
                        else:
                            _col_order = list(value)
                            # F2-31: Normalize column_order entries.
                            if normalize_aliases:
                                _col_order = [
                                    _normalize_dev_column(c)
                                    for c in _col_order
                                ]
                            if len(_col_order) >= MAX_CURVES:
                                raise ValueError(
                                    f"column_order has {len(_col_order)} entries, "
                                    f"maximum allowed is {MAX_CURVES - 1}."
                                )
                            dev.column_order = _col_order
                    else:
                        _recognised = False
                    if _recognised:
                        # Known metadata key — handled above, skip
                        # column processing.
                        continue
                    # F-MD4-05: Unrecognised _meta_ key is user data
                    # (not metadata).  Warn and fall through to store
                    # as a column under the stripped name so data is
                    # not silently lost.
                    warnings.warn(
                        f"Unrecognized _meta_ prefix key '{key}' — "
                        f"treating as column '{real_key}' rather "
                        f"than dropping it.",
                        stacklevel=2,
                    )
                    key = real_key
                    # Fall through to column processing below — do not
                    # skip this entry.
                # R7F-01-gap: When to_dict detects a column-name collision
                # with a metadata key (e.g. column named "source_file"),
                # it stores both the bare key (column data) and a
                # _meta_-prefixed key (real metadata).  Check whether the
                # corresponding _meta_ key exists — if so, the bare key
                # is column data, not metadata.
                _is_collision = f"_meta_{key}" in data
                if key in metadata_keys and not _is_collision:
                    if key == "encoding":
                        dev.encoding = _safe_str(value, "utf-8")
                    elif key == "source_file":
                        dev.source_file = _safe_str(value)
                    elif key == "column_order":
                        if value is None:
                            dev.column_order = []
                        elif isinstance(value, (str, bytes)):
                            # F-110: bytes bypasses the str guard in Python 3.
                            if isinstance(value, bytes):
                                raise TypeError(
                                    "column_order must be a list of strings, "
                                    "got bytes.  Decode to str first: "
                                    "value.decode('utf-8')"
                                )
                            dev.column_order = [value]
                            # F2-31: Normalize column_order entries to match
                            # normalized column names in dev.columns.
                            if normalize_aliases:
                                dev.column_order = [
                                    _normalize_dev_column(c)
                                    for c in dev.column_order
                                ]
                        else:
                            _col_order = list(value)
                            # F2-31: Normalize column_order entries.
                            if normalize_aliases:
                                _col_order = [
                                    _normalize_dev_column(c)
                                    for c in _col_order
                                ]
                            if len(_col_order) >= MAX_CURVES:
                                raise ValueError(
                                    f"column_order has {len(_col_order)} entries, "
                                    f"maximum allowed is {MAX_CURVES - 1}."
                                )
                            dev.column_order = _col_order
                else:
                    # F-I2-M02: None guard — np.array(None, dtype=np.float64)
                    # silently produces nan, consistent with string data guards.
                    if value is None:
                        raise ValueError(
                            f"Numeric data for column '{key}' is None"
                        )
                    # I2F-14: Pre-allocation size check.  np.array() allocates
                    # BEFORE the downstream len() guard — a huge list triggers
                    # MemoryError before the guard catches it.  Check len()
                    # first when the input supports it (matches ds_data pattern
                    # at lines 1057-1063).
                    if hasattr(value, '__len__') and len(value) > MAX_DATA_LINES:
                        raise ValueError(
                            f"Column '{key}' length ({len(value)}) exceeds "
                            f"maximum allowed ({MAX_DATA_LINES})"
                        )
                    try:
                        dev.columns[key] = np.atleast_1d(np.asarray(value, dtype=np.float64))
                    except (ValueError, TypeError, MemoryError, OverflowError) as e:
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
            # F2-30: Reject empty DevFile — no columns found.
            if not dev.columns:
                raise LASDataError("No columns found in input dict")
            # I2F-011: Sync column_order entries to match actual
            # case-sensitive keys in dev.columns.  The from_dict
            # normalization loop uppercases column_order entries via
            # _normalize_dev_column, but metadata-key columns (e.g.
            # "encoding" when a collision produces _meta_encoding) are
            # stored with their original case.  Map each column_order
            # entry back to the matching dev.columns key.
            if dev.column_order and normalize_aliases:
                _actual_keys = {k.upper(): k for k in dev.columns}
                dev.column_order = [
                    _actual_keys.get(c.upper(), c) for c in dev.column_order
                ]
            # F2-32: Cross-validate column_order entries against
            # columns.keys().  Orphaned entries (e.g. ["INC", "AZI"]
            # when only "MD" exists) produce silently broken output.
            _orphaned = [c for c in dev.column_order if c not in dev.columns]
            if _orphaned:
                raise LASDataError(
                    f"column_order contains entries not in columns: "
                    f"{_orphaned}"
                )
            # F-15: Re-invoke __post_init__ after full population.
            # During `dev = cls(_from_dict=True)`, __post_init__ returned
            # early (columns was empty).  Now that all columns are
            # populated, re-run the structural checks to validate
            # column_order/columns consistency and consistent lengths.
            dev.__post_init__()
            # F-43: Minimum data-quality validation matching reader's
            # _validate_dev_data checks.  Now delegated to validate(complete=True).
            for issue in dev.validate(complete=True):
                warnings.warn(issue, stacklevel=2)

            return dev
        except (ValueError, TypeError, OverflowError) as e:
            raise LASDataError(str(e)) from e
