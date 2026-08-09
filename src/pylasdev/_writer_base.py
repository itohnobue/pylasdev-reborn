"""Base writer infrastructure — constants, utilities, and shared writer class.

Contains all version-independent code shared by version-specific writer modules:
- Module-level constants and compiled regexes
- Module-level utility functions (sanitization, formatting)
- ``_WriterMutationGuard`` context manager
- ``_WriterBase`` abstract base class with template method
- ``write_las_file`` public API with version dispatch
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._sanitize import (
    _LEADING_SECTION_RE as _LEADING_SECTION_RE,
)
from ._sanitize import (
    _escape_braces_for_las_value as _escape_braces_for_las_value,
)
from ._sanitize import (
    _escape_colons_for_las_value as _escape_colons_for_las_value,
)
from ._sanitize import (
    _escape_pipes_for_las_value as _escape_pipes_for_las_value,
)
from ._sanitize import (
    _sanitize_las_value as _sanitize_las_value,
)
from ._version_spec import _LASVersionSpec
from .data_reader import _get_null_value, _get_well_entry_ci
from .exceptions import LASDataError, LASWriteError, PylasdevError
from .models import (
    _MNEMONIC_PATTERN,
    CurveDefinition,
    LASFile,
    ParameterEntry,
    _case_key,
    _GuardedDict,
    _GuardedList,
)

# _mnem_key is the writer's alias for the shared case-normalization
# primitive (models._case_key): uppercased MATCHING key, original-case
# storage/emission.  Every writer-side mnemonic comparison/lookup routes
# through it so the writer agrees with the container's case-insensitive
# resolution (N2b family).
_mnem_key = _case_key

# ── Module-level constants & compiled regexes ────────────────────────────

# _CONTROL_CHARS_RE / _UNICODE_WS_RE / _LEADING_SECTION_RE /
# _COLON_PRECEDED_BY_WS_RE / _COLON_FOLLOWED_BY_WS_OR_END_RE live in
# _sanitize.py (single source of truth for the escape helpers).  The
# aliases above keep every existing import path working (II-22/II-25);
# _LEADING_SECTION_RE is additionally used at :721 (M-28 first-column
# escape) and :1311 (~O tilde warning) outside the moved functions.

# The 256-character line-length limit applies to:
#   - LAS 1.2 (all modes) per the LAS 1.2 specification.
#   - LAS 2.0 WRAP=NO per the CWLS specification.
MAX_LINE_LENGTH_LAS12: int = 256

# Map DataSection.section_type values to the LAS 3.0 section header prefix.
_SECTION_TYPE_TO_PREFIX: dict[str, str] = {
    "LOG_DATA": "A",
    "CORE_DATA": "CORE_DATA",
    "DRILLING_DATA": "DRILLING_DATA",
    "INCLINOMETRY_DATA": "INCLINOMETRY_DATA",
    "TOPS_DATA": "TOPS_DATA",
    "TEST_DATA": "TEST_DATA",
    "PERFORATIONS_DATA": "PERFORATIONS_DATA",
}

# Map DataSection.section_type values to the Definition section header prefix.
_SECTION_TYPE_TO_DEFINITION_PREFIX: dict[str, str] = {
    "CORE_DATA": "Core",
    "DRILLING_DATA": "Drilling",
    "INCLINOMETRY_DATA": "Inclinometry",
    "TOPS_DATA": "Tops",
    "TEST_DATA": "Test",
    "PERFORATIONS_DATA": "Perforations",
}

# E-36: the parser's ~W unit grammar — DATA_LINE_PATTERN's unit group
# (parser.py:318 ``[\w\-/.%°:]*``).  A well unit containing any other
# character cannot round-trip: the ~W line truncates at the first
# out-of-class character and the entry VALUE is absorbed into the
# description (destroyed on write→read).  The writer rejects such units
# at emission instead of silently corrupting the file.
_WELL_UNIT_PATTERN = re.compile(r"^[\w\-/.%°:]*$")


# ── Module-level utility functions ──────────────────────────────────────

# _sanitize_las_value / _escape_colons_for_las_value / _escape_pipes_for_las_value
# moved to _sanitize.py (shared with the read side); re-exported above.


def _validate_precision(precision: str) -> None:
    """Validate the precision format specifier for numeric output."""
    if not re.match(r"^\.\d+([eEfFgGn%])?$", precision):
        raise ValueError(
            f"Invalid precision format specifier: '{precision}'. "
            f"Expected a format like '.8g', '.6f', or '.10e'. "
            f"Non-numeric format codes (x, o, b, c, d) are not supported "
            f"for LAS numeric data output."
        )
    if precision[-1] in ("n", "%"):
        import warnings

        warnings.warn(
            f"Precision format code '{precision[-1]}' in '{precision}' "
            f"is not safe for LAS output.  The 'n' format code produces "
            f"locale-dependent output (unparseable comma/grouping). "
            f"The '%' format code multiplies by 100 and appends '%' "
            f"(unparseable suffix).  Consider using 'g', 'f', or 'e' instead.",
            UserWarning,
            stacklevel=2,
        )


def _section_type_to_prefix(section_type: str) -> str:
    """Convert a DataSection.section_type to the LAS header prefix."""
    section_type = section_type.upper()
    known = _SECTION_TYPE_TO_PREFIX.get(section_type)
    if known is not None:
        return known
    if section_type.endswith("_DATA"):
        return _sanitize_las_value(section_type).replace("|", "")
    # M-84: Bare known section types (e.g. "CORE") are accepted by the
    # model and by the parser's _SECTION_TYPE_MAP (which maps both "CORE"
    # and "CORE_DATA" → "CORE_DATA"), but the header prefix map only knows
    # the *_DATA forms.  Without this fallback a bare "CORE" falls back to
    # "A" + a misdiagnosing warning, and the re-read section_type silently
    # becomes LOG_DATA.  Try the canonical *_DATA form so the header stays
    # roundtrippable (defense-in-depth; the model also normalizes at
    # construction).
    _data_form = section_type + "_DATA"
    if _data_form in _SECTION_TYPE_TO_PREFIX:
        return _SECTION_TYPE_TO_PREFIX[_data_form]
    import warnings

    warnings.warn(
        f"Unknown section type '{section_type}'. "
        f"Falling back to ASCII data section header 'A'. "
        f"Known types: {', '.join(sorted(_SECTION_TYPE_TO_PREFIX.keys()))}. "
        f"Custom types must end with '_DATA'.",
        stacklevel=3,
    )
    return "A"


def _emitted_mnemonic(curve: CurveDefinition, is_las30: bool = True) -> str:
    """The mnemonic as written to the ~C / Definition line.

    Mirrors the emission logic in ``_format_curve_line``:
    - M-59 (F-16): when ``curve.original_mnemonic`` is set and differs
      from ``curve.mnemonic``, ``_format_curve_line`` emits the
      VENDOR-standard original name (reconstructing e.g. ``LLD`` from the
      reader-renamed ``BFV``).  The dedup/identity keys MUST use the same
      emitted name, or two curves that collide in the output (LLD + BFV
      with original_mnemonic='LLD') are seen as DISTINCT by dedup and BOTH
      are emitted — duplicate LLD lines in ~C, structurally invalid file,
      silent BFV identity loss on re-read (M-64 dedup-key divergence).
    - W-09: a curve with ``array_info`` but no ``[N]`` in its mnemonic is
      emitted with the bracket form (``NMR`` + index=1 → ``NMR[1]``).
      Using the EMITTED mnemonic for dedup/scoping keeps directly-
      constructed models (base mnemonic + array_info) consistent with
      parsed models (bracket mnemonic) — without it, NMR[1]/NMR[2]
      collide (M-64).

    F-27 (W-10): the ``[N]`` bracket is appended ONLY when ``is_las30``,
    mirroring the ``is_las30`` gate in ``_format_curve_line``.  On LAS
    1.2/2.0 ``_format_curve_line`` never emits the bracket (M-27 drops
    array_info), so a dedup key that appended ``[N]`` unconditionally
    diverged from the emitted name: dedup saw ``DEPT`` vs ``DEPT[1]`` as
    DISTINCT while the emitter wrote ``DEPT`` twice — two identical ~C
    lines, structurally invalid, silently written.  ``is_las30`` defaults
    to True so the LAS 3.0 call sites (which DO emit the bracket) are
    unchanged.
    """
    mnemonic = (
        curve.original_mnemonic
        if curve.original_mnemonic and curve.original_mnemonic != curve.mnemonic
        else curve.mnemonic
    )
    if curve.array_info is not None and "[" not in mnemonic and is_las30:
        mnemonic = f"{mnemonic}[{curve.array_info.index}]"
    return mnemonic


def _dedup_by_emitted_mnemonic(curves: list[CurveDefinition]) -> list[CurveDefinition]:
    """Dedup a curve list by the EMITTED mnemonic (first definition wins).

    F-16 (M-59 ↔ M-64): ``_format_curve_line`` emits
    ``original_mnemonic`` when it differs from ``curve.mnemonic``, so two
    distinct curves can emit the SAME name (e.g. a real ``LLD`` and a
    ``BFV`` with ``original_mnemonic='LLD'``).  The main ~C block's dedup
    loop keys on ``_emitted_mnemonic``; the LOG_DATA scoping comparison
    must use the SAME (deduped) curve set or a section whose curves
    collide on the emitted name would get a per-section Definition that
    re-emits the duplicate (structurally invalid file).  First definition
    wins, mirroring the ~C block's W-01 dedup.
    """
    seen: set[str] = set()
    deduped: list[CurveDefinition] = []
    for curve in curves:
        # N2b-2: dedup keys on the UPPER-CASED emitted mnemonic — the
        # parser's re-read identity is case-insensitive (it uppercases
        # mnemonics at read), so two emitted names differing only by case
        # ('DEPT' vs 'dept') WOULD collide on re-read and be renamed
        # (DEPT_2), silently altering the model identity.  First wins,
        # mirroring the ~C block's W-01 dedup for exact duplicates.
        emitted = _emitted_mnemonic(curve)
        if _mnem_key(emitted) in seen:
            continue
        seen.add(_mnem_key(emitted))
        deduped.append(curve)
    return deduped


def _emission_plan(
    curves: list[CurveDefinition],
    is_las30: bool = True,
) -> tuple[list[tuple[CurveDefinition, str | None]], list[CurveDefinition]]:
    """Compute collision-free emission names for a ~C / Definition block.

    W-10/M-59: two distinct ``CurveDefinition`` objects can emit the same
    name — a reader-renamed duplicate (``IK_2`` with
    ``original_mnemonic='IK'`` alongside a real ``IK``), a mnem_base
    vendor-rename collision (``LLD`` + ``BFV`` with
    ``original_mnemonic='LLD'``), or genuine duplicate mnemonics
    (``DEPT(M)``/``DEPT(FT)``).  The first occurrence emits normally.
    A later curve whose M-59 reconstruction would collide falls back to
    its OWN mnemonic when that is free (preserving the distinct column
    and keeping the write→read roundtrip stable); when the own mnemonic
    is ALSO taken the definition is a metadata-only duplicate and is
    dropped (returned in ``dropped`` for the caller to warn about).

    F-27 (W-10): ``is_las30`` is forwarded to ``_emitted_mnemonic`` so
    the collision candidate matches the name ``_format_curve_line`` will
    actually emit on every version (the ``[N]`` bracket is only emitted
    for LAS 3.0).  The LAS 1.2/2.0 ~C block passes ``is_las30=False``;
    the LAS 3.0 paths use the default.

    Returns ``(pairs, dropped)`` where ``pairs`` is an ordered list of
    ``(curve, mnemonic_override)`` — ``mnemonic_override`` is ``None``
    for a normal M-59 emission and ``curve.mnemonic`` for a collision
    fallback — and ``dropped`` lists the metadata-only duplicates.
    """
    seen: set[str] = set()
    pairs: list[tuple[CurveDefinition, str | None]] = []
    dropped: list[CurveDefinition] = []
    for curve in curves:
        candidate = _emitted_mnemonic(curve, is_las30)
        # N2b-2: dedup on the UPPER-CASED emitted mnemonic — the parser's
        # re-read identity is case-insensitive, so a case-variant pair
        # ('DEPT' + 'dept') must be treated as the duplicate it is (both
        # would emit distinct ~C lines that re-read renames to DEPT_2).
        cand_key = _mnem_key(candidate)
        if cand_key in seen:
            own_key = _mnem_key(curve.mnemonic)
            if own_key not in seen:
                pairs.append((curve, curve.mnemonic))
                seen.add(own_key)
            else:
                dropped.append(curve)
            continue
        seen.add(cand_key)
        pairs.append((curve, None))
    return pairs, dropped


def _format_curve_line(
    curve: CurveDefinition,
    is_las30: bool,
    string_mnemonics: frozenset[str] | None = None,
    mnemonic_override: str | None = None,
    string_union_mnemonics: frozenset[str] | None = None,
) -> str:
    """Format a single CurveDefinition as a LAS curve line.

    Args:
        string_mnemonics: UPPER-CASED mnemonics (via ``_mnem_key``) of
            curves whose DATA lives in a string_data container for the
            emitted scope.  M-77: a LAS 3.0 string curve with an empty (or
            non-'S') data_format would be emitted WITHOUT the {S} marker —
            the parser's ONLY string signal — and re-read as numeric,
            silently destroying the values.  Callers with string_data
            context pass this set so the writer forces the {S} marker.
            The membership tests the curve's EMITTED name (M-59
            original_mnemonic reconstruction) uppercased AND the curve's
            OWN storage mnemonic (``curve.mnemonic``): a case-variant
            string_data key ('dept_str' vs curve 'DEPT_STR') resolves via
            the upper-cased emitted name, and a renamed curve whose
            string_data is keyed by its STORAGE name ('BFV' storage key,
            emitted 'LLD') resolves via the storage key (N2b-1/II-7 +
            M-35).  All 4 call sites build the set upper-cased.
        mnemonic_override: When set, emit THIS mnemonic instead of the
            M-59 original_mnemonic reconstruction.  W-10: a
            reader-renamed duplicate (IK_2 with original_mnemonic='IK')
            or a mnem_base vendor-rename (BFV with original_mnemonic='LLD')
            would otherwise emit a name already taken by another curve in
            the same block, producing duplicate ~C/Definition lines.  The
            collision-free emission plan falls back to the curve's own
            mnemonic to keep the block valid and the roundtrip stable.
        string_union_mnemonics: UPPER-CASED mnemonics of curves whose DATA
            lives in a string_data container in ANY scope (top-level
            string_data plus every data section), WITHOUT the
            numeric-placement exclusion that ``string_mnemonics`` applies.
            F-01: when the same mnemonic is placed string in one scope and
            numeric in another, the main ~C line must not carry the
            curve's explicit numeric format token for the numeric
            placement — the parser's format-vs-placement check
            (parser.py:1393-1398) rejects a numeric-format curve whose
            first occurrence lands in string_data, making the written file
            self-unreadable.  A curve is "mixed" when it appears in this
            union (string somewhere) but NOT in ``string_mnemonics`` (the
            union minus numeric placements — i.e. not string in the
            emitted scope).  Purely-numeric curves never appear in the
            union and keep their format token.  Only the main ~C call
            sites pass this; per-section Definitions keep the section's
            own format so the numeric section's metadata survives re-read.
    """
    unit = _sanitize_las_value(curve.unit) if curve.unit else ""
    unit = _escape_colons_for_las_value(unit) if unit else ""
    desc = curve.description if curve.description else ""
    # N-09: escape literal braces in the user description BEFORE any
    # format token is appended.  The parser strips a VALID format token
    # ({S}, {F}, {A:N}...) from the trailing position of a curve line and
    # fabricates data_format from it — user text like "Gamma {S} ray" was
    # destroyed and the numeric column re-routed to string_data on
    # write→read.  Escaped braces (\{) cannot match FORMAT_SPEC_PATTERN;
    # the writer's OWN appended token below stays unescaped so the parser
    # still recognizes it.  The parser reverses the escape at the
    # format-strip site (parser.py:3355) with
    # _unescape_braces_for_las_value.
    desc = _escape_braces_for_las_value(desc)

    # M-59: Emit the vendor-standard mnemonic when the model records one.
    # The reader's mnem_base rename (e.g. LLD→BFV) preserves the original
    # name in CurveDefinition.original_mnemonic, but the writer previously
    # emitted curve.mnemonic ONLY — so a read→write roundtrip
    # permanently canonicalized the colliding vendor name in the output
    # file (re-read without mnem_base never recovers it).  When
    # original_mnemonic is set and differs, emit it so the write
    # reconstructs the vendor-standard name.  Data columns stay
    # positional, so values remain aligned (verified: the parser routes
    # data by ~C order, not by the ~A header).  Computed FIRST so the
    # {S}-marker membership below tests the EMITTED name (II-7: a curve
    # BFV with original_mnemonic='LLD' emits 'LLD', and the marker
    # decision must match what is actually written).
    _emit_mnem = (
        mnemonic_override
        if mnemonic_override is not None
        else (
            curve.original_mnemonic
            if curve.original_mnemonic and curve.original_mnemonic != curve.mnemonic
            else curve.mnemonic
        )
    )

    # N2b-1/II-7: the {S} membership must test the EMITTED mnemonic
    # (_emit_mnem), NOT curve.mnemonic — a curve with original_mnemonic=
    # 'LLD' emits markerless with NO case variance when only
    # curve.mnemonic ('BFV') is tested, destroying the string values on
    # write→read.  Both sides compare via _mnem_key: the string_mnemonics
    # sets are built UPPER-CASED at every call site (all 4 builders), so
    # a case-variant string_data key ('dept_str' vs curve 'DEPT_STR')
    # still resolves the marker.
    # M-35: the sets are built from the string_data CONTAINER KEYS (the
    # STORAGE names).  A renamed curve (storage key 'BFV',
    # original_mnemonic='LLD') emits 'LLD' and would MISS a set keyed
    # only by the storage name — the {S} marker is not forced and the
    # string values are destroyed on write→read.  Test the curve's OWN
    # storage mnemonic as well so the renamed curve resolves via the key
    # its data is actually stored under.
    is_string_curve = string_mnemonics is not None and (
        _mnem_key(_emit_mnem) in string_mnemonics
        or _mnem_key(curve.mnemonic) in string_mnemonics
    )
    # H-02: suppress the spurious {S} for a curve that DECLARES
    # data_format='S' but whose mnemonic is NOT in this scope's
    # string_data set — the section places the mnemonic NUMERICALLY.
    # The pre-fix code emitted {S} from the curve OBJECT's data_format
    # unconditionally, so a numeric column was re-read as STRING on
    # write→read (silent type corruption, zero warnings; the LAS 3.0
    # no-data_sections and no-section_curves shapes bypass every
    # construction guard).  Placement context exists only when
    # string_mnemonics is not None (all 4 call sites pass a set,
    # possibly empty); without context the old behavior is preserved.
    # F-01: extend the suppression to ANY explicit numeric format
    # ('F'/'E'/'A'/'I'...), not just 'S'.  When the same mnemonic is
    # placed string in one scope and numeric in another, the main ~C
    # must not carry the curve's top-level data_format token for the
    # numeric placement — the parser's format-vs-placement check
    # (parser.py:1393-1398: `_df not in ("S","A")` + mnemonic in
    # string_data → LASParseError) would reject the numeric-format
    # first occurrence that lands in the string_data mirror, making the
    # written file SELF-UNREADABLE (the H-01 explicit-format sub-variant;
    # the H-01 consumer-side `if not _df: continue` rescue only covers
    # the empty-format case).  The mixed-placement signal is
    # string_union_mnemonics: the mnemonic appears in SOME scope's
    # string_data (string somewhere) while not being in THIS scope's
    # string set (numeric here).  Purely-numeric curves (never string
    # anywhere) are absent from the union and keep their format token
    # (the format metadata must survive roundtrip for them — pinned by
    # the all-numeric explicit-'F'/'E' tests).
    _suppress_s_marker = (
        is_las30
        and string_mnemonics is not None
        and not is_string_curve
        and (
            (curve.data_format or "").upper() == "S"
            or (
                curve.data_format
                and string_union_mnemonics is not None
                and (
                    _mnem_key(_emit_mnem) in string_union_mnemonics
                    or _mnem_key(curve.mnemonic) in string_union_mnemonics
                )
            )
        )
    )
    if is_las30 and is_string_curve and (curve.data_format or "").upper() != "S":
        # M-77: a string-data curve without data_format='S' would be
        # emitted markerless; the parser classifies columns by the {S}
        # marker ONLY, so the values are re-read as numeric nulls.  Force
        # the {S} marker.  When a conflicting non-empty format is declared
        # the marker swap is a real contract change — warn loudly.
        if curve.data_format:
            import warnings

            warnings.warn(
                f"Curve '{curve.mnemonic}' is placed in string_data but "
                f"declares data_format={curve.data_format!r} (not 'S').  "
                f"Emitting the {{S}} marker so the parser recognizes the "
                f"values as strings; the declared format is not "
                f"representable for string data.",
                UserWarning,
                stacklevel=3,
            )
        format_str = "{S}"
        desc = f"{desc}  {format_str}"
    elif curve.data_format and (is_las30 or curve.data_format == "I") and not _suppress_s_marker:
        # EXT-04: the braced {I} marker is emitted for integer-format
        # curves on ALL versions.  LAS 1.2/2.0 have no format-specifier
        # convention, but without the marker a >2^53 {I} value (e.g.
        # 9007199254740993) is re-read as float64 and silently rounded —
        # the marker is the only way the data reader restores integer
        # parsing on write→read roundtrip.  Other formats (F/E/S/A) remain
        # unmarked on LAS 1.2/2.0 to preserve existing output (string
        # curves are lossy on LAS 2.0 by design — see M-29).
        format_str = f"{{{curve.data_format}"
        if (
            curve.data_format == "A"
            and curve.array_info
            and curve.array_info.time_offset is not None
        ):
            offset = curve.array_info.time_offset
            if math.isfinite(offset):
                if offset == int(offset):
                    # E-40: the emitted offset field must fit the parser's
                    # 64-character offset group (FORMAT_SPEC_PATTERN
                    # ``[-\d.]{0,64}``).  An integral offset >= 1e64
                    # formats to a 65+ digit decimal — the whole {A:N}
                    # spec then fails to parse and data_format AND
                    # time_offset are lost (the curve is re-read with a
                    # different format).  Reject loudly: the offset cannot
                    # be represented in the field at all (even rounded, a
                    # 64-char decimal of 1e64 would be meaningless).
                    _int_str = str(int(offset))
                    if len(_int_str) > 64:
                        raise LASWriteError(
                            f"Curve '{curve.mnemonic}' time_offset "
                            f"{offset!r} cannot be represented in the "
                            f"{{A:N}} format specifier: the decimal field "
                            f"({len(_int_str)} characters) exceeds the "
                            f"parser's 64-character offset-group cap, and "
                            f"the spec (and the curve's data_format) would "
                            f"be lost on write→read roundtrip."
                        )
                    # IEEE 754 negative zero: int(-0.0) == 0 loses the sign.
                    # Use float formatting to preserve "-0" in the output,
                    # matching the copysign guard in _format_number.
                    if offset == 0 and math.copysign(1.0, offset) < 0:
                        format_str += f":{offset}"
                    else:
                        format_str += f":{_int_str}"
                else:
                    format_str += f":{_format_offset_plain(offset)}"
        format_str += "}"
        desc = f"{desc}  {format_str}"
    elif curve.data_format and not _suppress_s_marker:
        # M-27: non-LAS-3.0 output cannot represent a non-{I} format
        # specifier — the metadata is silently dropped on write→read.
        # The H-02-suppressed case (data_format='S' placed numerically on
        # LAS 3.0) is excluded: the column IS numeric, so no format is
        # emitted and the DATA round-trips — not a "dropped on write→read"
        # loss, and the message is 1.2/2.0-specific.
        import warnings

        warnings.warn(
            f"Curve '{curve.mnemonic}' data_format "
            f"{curve.data_format!r} cannot be represented in LAS "
            f"1.2/2.0 output — it is dropped on write→read roundtrip.",
            UserWarning,
            stacklevel=3,
        )
    if curve.array_info is not None and not is_las30:
        # M-27: array_info (and the bracket mnemonic that carries it) is
        # only emitted for LAS 3.0; on 1.2/2.0 the metadata is silently
        # dropped on write→read.
        import warnings

        warnings.warn(
            f"Curve '{curve.mnemonic}' array_info is not representable "
            f"in LAS 1.2/2.0 output — it is dropped on write→read "
            f"roundtrip.",
            UserWarning,
            stacklevel=3,
        )

    api_code = (
        _sanitize_las_value(curve.api_code, preserve_leading_tilde=True) if curve.api_code else ""
    )
    api_code = _escape_colons_for_las_value(api_code)
    api = f"  {api_code}" if api_code else ""
    desc = _sanitize_las_value(desc, preserve_leading_tilde=True)
    desc = _escape_colons_for_las_value(desc)

    mnemonic = _sanitize_las_value(_emit_mnem)
    if is_las30 and curve.array_info is not None and "[" not in mnemonic:
        # W-09: The parser reconstructs CurveDefinition.array_info ONLY
        # from bracket mnemonics (ARRAY_MNEMONIC_PATTERN).  A curve whose
        # mnemonic lacks "[N]" loses its array_info on roundtrip — an
        # {A:N} format curve with array_info but no bracket is treated as
        # string-format on re-read and its numeric data is reclassified
        # into string_data.  Emit the bracket form so array_info survives
        # and the data stays numeric.  The "[" guard avoids
        # double-bracketing when the mnemonic already carries "[N]".
        mnemonic = f"{mnemonic}[{curve.array_info.index}]"

    return f" {mnemonic}.{unit}{api}  : {desc}"


def _format_parameter_line(param: ParameterEntry, is_las30: bool) -> str:
    """Format a single ParameterEntry as a LAS parameter line."""
    unit = _sanitize_las_value(param.unit) if param.unit else ""
    unit = _escape_colons_for_las_value(unit) if unit else ""
    desc = param.description if param.description else ""
    # N-I-02(b): Escape literal pipes in the description BEFORE any zone
    # association is appended.  A genuine pipe in user text would otherwise
    # be misparsed as a LAS 3.0 zone association (| Zone) on re-read —
    # truncating the description and attaching a bogus ParameterZone.  Real
    # zone associations appended below remain unescaped so the parser's
    # ZONE_ASSOC_PATTERN still recognizes them.
    desc = _escape_pipes_for_las_value(desc)
    # N-09: escape literal braces in the user description BEFORE any
    # format token is appended (mirrors _format_curve_line).  The parser
    # strips a valid format token from the trailing position of a ~P line
    # and fabricates param_data_format from user text; escaped braces
    # (\{) cannot match FORMAT_SPEC_PATTERN.  The parser reverses the
    # escape at the format-strip site (parser.py:3610) with
    # _unescape_braces_for_las_value.
    desc = _escape_braces_for_las_value(desc)

    if is_las30 and param.data_format:
        # N-I-21: Always emit the braced {…} form.  Previously
        # multi-character values (e.g. "DD/MM/YYYY") were emitted
        # UNBRACED, which on re-read merged the format text into the
        # description and lost the data_format field entirely.  The
        # braced form is the valid LAS 3.0 construct; the parser
        # recognizes it (extracting or clearing the format while
        # keeping the description stable), so the roundtrip is
        # deterministic across construction paths.
        desc = f"{desc}  {{{param.data_format}}}"
    elif param.data_format:
        # M-27: LAS 1.2/2.0 output cannot represent a braced format
        # specifier — the metadata is silently dropped on write→read.
        import warnings

        warnings.warn(
            f"Parameter '{param.mnemonic}' data_format "
            f"{param.data_format!r} cannot be represented in LAS "
            f"1.2/2.0 output — it is dropped on write→read roundtrip.",
            UserWarning,
            stacklevel=3,
        )

    if is_las30 and param.zone:
        # M-12: escape literal pipes in the ZONE NAME itself.  A genuine
        # pipe inside zone_name (e.g. "Zone|X") would otherwise be
        # re-parsed as the LAST '|' fragment of the zone association —
        # ZONE_ASSOC_PATTERN's (?<!\\)\| matches any non-escaped pipe, so
        # "| Zone|X[2]" re-reads zone="X" and leaks "| Zone" into the
        # description.  Escaping the zone_name pipe keeps it inside the
        # zone text; the parser unescapes it after extraction.
        zone_name = _escape_pipes_for_las_value(param.zone.zone_name)
        zone_str = f" | {zone_name}"
        if param.zone.zone_index is not None:
            zone_str += f"[{param.zone.zone_index}]"
        desc = f"{desc}{zone_str}"
    elif param.zone:
        # M-27: LAS 1.2/2.0 output cannot represent a zone association —
        # the metadata is silently dropped on write→read.
        import warnings

        warnings.warn(
            f"Parameter '{param.mnemonic}' zone association is not "
            f"representable in LAS 1.2/2.0 output — it is dropped on "
            f"write→read roundtrip.",
            UserWarning,
            stacklevel=3,
        )

    value = _sanitize_las_value(param.value, preserve_leading_tilde=True)
    desc = _sanitize_las_value(desc, preserve_leading_tilde=True)
    value = _escape_colons_for_las_value(value)
    desc = _escape_colons_for_las_value(desc)

    mnemonic = _sanitize_las_value(param.mnemonic)
    if is_las30 and param.array_index is not None and "[" not in mnemonic:
        # W-08: The parser reconstructs ParameterEntry.array_index ONLY
        # from bracket mnemonics (ARRAY_MNEMONIC_PATTERN).  A parameter
        # whose mnemonic lacks "[N]" loses its array_index on roundtrip
        # (RUN with array_index=1 → array_index=None).  Emit the bracket
        # form so the index survives.  The "[" guard avoids
        # double-bracketing when the mnemonic already carries "[N]".
        mnemonic = f"{mnemonic}[{param.array_index}]"

    return f" {mnemonic}.{unit}  {value}  : {desc}"


def _lookup_data_array(
    name: str,
    data: dict[str, NDArray[np.float64]],
    string_data: dict[str, NDArray[np.object_]],
) -> tuple[NDArray[np.float64] | NDArray[np.object_] | None, bool]:
    """Resolve a curve name to its data array, case-insensitively.

    F-32 (I2-22 consistency): the ~C definition resolution
    (``_curves_in_live_order``, ``_effective_section_curves``) is
    case-insensitive — a lowercase ``'dept'`` in ``curves_order``
    resolves to the ``DEPT`` definition.  The data-key lookup must use
    the SAME resolution or a case-variant ``curves_order`` entry
    resolves the definition but fails the data lookup, null-filling the
    written column (LAS 3.0 ``_format_data_rows`` path) or skipping the
    ~A section entirely (legacy Path C gate).  Exact-case matches win
    (unambiguous); the case-insensitive fallback mirrors the
    definition-resolution behavior.

    Returns ``(array, is_string)``; ``array`` is None when no key
    matches (the caller pads the column with the null sentinel).
    """
    if name in string_data:
        return string_data[name], True
    if name in data:
        return data[name], False
    upper = name.upper()
    for key, arr in string_data.items():
        if key.upper() == upper:
            return arr, True
    for key, num_arr in data.items():
        if key.upper() == upper:
            return num_arr, False
    return None, False


def _format_data_rows(
    curve_names: list[str],
    data: dict[str, NDArray[np.float64]],
    string_data: dict[str, NDArray[np.object_]],
    null_value: float,
    delimiter: str,
    precision: str = ".8g",
    is_las12: bool = False,
) -> list[str]:
    """Format data rows for a section — handles both legacy and LAS 3.0 sections."""
    lines: list[str] = []
    if not curve_names:
        return lines

    curve_arrays: list[tuple[NDArray[np.float64] | NDArray[np.object_] | None, bool]] = []
    for name in curve_names:
        # F-32: case-insensitive lookup — the definition resolution is
        # case-insensitive (I2-22), so the data-key lookup must be too.
        # Exact-case matches win; the fallback resolves a case-variant
        # curves_order entry ('dept') to the data stored under 'DEPT'.
        _arr, _is_string = _lookup_data_array(name, data, string_data)
        # E-16: a 0-d numpy array (accepted by _check_column_array_like —
        # the M-18 convention treats it as a single-element value, see
        # models._GuardedDict._value_len) has no len() and is not
        # indexable by position; both crashed the write with an opaque
        # TypeError wrapped into LASWriteError.  Normalize to a 1-element
        # 1-D view so the row-count and per-row access below work.
        if isinstance(_arr, np.ndarray) and _arr.ndim == 0:
            _arr = _arr.reshape(1)
        curve_arrays.append((_arr, _is_string))

    num_rows = max(
        (len(arr) for arr, _ in curve_arrays if arr is not None),
        default=0,
    )
    if num_rows == 0:
        return lines

    warned_long = False
    warned_delim_str = False
    warned_empty_str = False
    warned_tilde_str = False
    for i in range(num_rows):
        row_values: list[str] = []
        for arr, is_string in curve_arrays:
            if arr is None or i >= len(arr):
                if is_string:
                    # M-78: a short (ragged) string curve was padded with
                    # the NUMERIC null sentinel, which on re-read becomes
                    # a fabricated "-999.25" STRING value.  Route missing
                    # string values through the string-branch missing-value
                    # routing (the '-' sentinel) instead.
                    if not warned_empty_str:
                        import warnings

                        warnings.warn(
                            "Missing string curve value (short array) "
                            "padded with '-' sentinel — roundtrip "
                            "fidelity is lost: parser cannot distinguish "
                            "original '-' from the missing value.",
                            stacklevel=4,
                        )
                        warned_empty_str = True
                    row_values.append("-")
                else:
                    row_values.append(_format_number(null_value, precision, null_value))
            elif is_string:
                _raw = arr[i]
                # N-I-17: A None/NaN/Inf value in a string-data array was
                # written as the literal string "None"/"nan" (via str()),
                # fabricating data on re-read — the numeric branch routes
                # non-finite values to the null sentinel, but the string
                # branch had no guard.  Route missing values to the same
                # '-' sentinel used for empty strings.
                if _raw is None or (
                    isinstance(_raw, (float, np.floating)) and not math.isfinite(_raw)
                ):
                    if not warned_empty_str:
                        import warnings

                        warnings.warn(
                            "Missing string curve value (None/NaN/Inf) "
                            "replaced by '-' sentinel — roundtrip "
                            "fidelity is lost: parser cannot distinguish "
                            "original '-' from the missing value.",
                            stacklevel=4,
                        )
                        warned_empty_str = True
                    row_values.append("-")
                    continue
                raw_val = str(_raw)
                raw_has_delim = delimiter in raw_val
                # M-28: preserve a leading '~' in string DATA values —
                # stripping it silently corrupts the model value.  Mid-row
                # cells can never be confused with a section header.
                val = _sanitize_las_value(raw_val, preserve_leading_tilde=True)
                if delimiter == " ":
                    if re.search(r"\s", val):
                        if not warned_delim_str:
                            import warnings

                            warnings.warn(
                                "String curve data contains whitespace "
                                "characters while using SPACE delimiter. "
                                "Internal whitespace (including Unicode "
                                "whitespace such as non-breaking spaces) "
                                "will be replaced with underscores to "
                                "prevent data corruption on re-read. "
                                "Consider switching to COMMA or TAB "
                                "delimiter for files with string curves.",
                                stacklevel=4,
                            )
                        warned_delim_str = True
                        val = re.sub(r"\s", "_", val)
                elif raw_has_delim:
                    if not warned_delim_str:
                        import warnings

                        delim_name = "COMMA" if delimiter == "," else "TAB"
                        replacement = ";" if delimiter == "," else " "
                        warnings.warn(
                            f"String curve data contains the active "
                            f"delimiter character ({delim_name}). The "
                            f"delimiter will be replaced with "
                            f"{'semicolons' if delimiter == ',' else 'spaces'} "
                            f"to prevent data corruption on re-read.",
                            stacklevel=4,
                        )
                        warned_delim_str = True
                    replacement = ";" if delimiter == "," else " "
                    val = val.replace(delimiter, replacement)
                if not val and delimiter == " ":
                    if not warned_empty_str:
                        import warnings

                        warnings.warn(
                            "Empty string curve value replaced by "
                            "'-' sentinel — roundtrip fidelity is "
                            "lost: parser cannot distinguish original "
                            "'-' from originally-empty value.",
                            stacklevel=4,
                        )
                        warned_empty_str = True
                    val = "-"
                if not row_values and re.match(r"\s*~[A-Za-z]", val):
                    # M-28: this value lands in the FIRST column, so the
                    # emitted data row would start with '~'+letter and the
                    # reader would treat it as a LAS section header (the
                    # writer's own reader skips such lines) — dropping the
                    # row and breaking the file structure.  The value is
                    # preserved everywhere else, but here the leading '~'
                    # MUST be stripped for the file to stay valid — warn
                    # loudly that the value was altered.
                    val = _LEADING_SECTION_RE.sub(r"\1", val, count=1)
                    if not warned_tilde_str:
                        import warnings

                        warnings.warn(
                            "String curve value in the first data column "
                            "begins with '~' followed by a letter; the "
                            "emitted row would be misread as a LAS section "
                            "header on re-read.  The leading '~' was "
                            "removed, so the written value differs from "
                            "the model.",
                            stacklevel=4,
                        )
                        warned_tilde_str = True
                elif not row_values and re.match(r"\s*~", val):
                    # M-85: value in the FIRST data column starts with '~'
                    # + NON-letter (e.g. '~3D', '~.', bare '~').  The
                    # M-28 guard above only matches '~'+letter; these
                    # survive _sanitize_las_value, so the emitted data row
                    # would BEGIN with '~' — and the parser/reader skip
                    # '~'-prefixed lines as section-header noise
                    # (parser.py F-83 / _data_section_reader.py
                    # _iter_ascii_data_lines), silently dropping the
                    # ENTIRE row with zero warnings.  Escape the leading
                    # '~' as '_~' (mirroring the existing '#'-prefix
                    # escape) so the line never starts with '~'; the read
                    # side restores '_~' → '~' for the first-column token
                    # (restore_tilde on both the LAS 3.0 path
                    # _las30_data.py:1195 and the LAS 1.2/2.0 path
                    # data_reader._read_normal/_read_wrapped — E-19).
                    val = "_" + val.lstrip()
                    if not warned_tilde_str:
                        import warnings

                        if is_las12:
                            # E-19: on LAS 1.2/2.0 the escape IS restored
                            # for a first-column string token on re-read,
                            # but string curves are lossy there by design
                            # (M-29 — no {S} marker, the column re-reads
                            # as numeric and null-fills).  The ROW (and
                            # the other columns' values) survive; the
                            # string VALUE itself does not round-trip.
                            warnings.warn(
                                "String curve value in the first data "
                                "column begins with '~' followed by a "
                                "non-letter; the emitted row would be "
                                "misread as a LAS section header and "
                                "silently dropped on re-read.  The "
                                "leading '~' was escaped as '_~' so the "
                                "row stays in the file; on LAS 1.2/2.0 "
                                "the escape is restored for a first-column "
                                "string token but string values do not "
                                "round-trip by design (M-29).",
                                stacklevel=4,
                            )
                        else:
                            warnings.warn(
                                "String curve value in the first data "
                                "column begins with '~' followed by a "
                                "non-letter; the emitted row would be "
                                "misread as a LAS section header and "
                                "silently dropped on re-read.  The "
                                "leading '~' was escaped as '_~' and is "
                                "restored on re-read for the first-column "
                                "token, so the value round-trips.",
                                stacklevel=4,
                            )
                        warned_tilde_str = True
                row_values.append(val)
            else:
                val = arr[i]
                # IT3-F-02: math.isfinite is ~15x faster than the numpy
                # scalar isfinite/isnan/isinf chain for per-value checks and
                # is semantically identical for Python/numpy float scalars
                # (verified: no NaN-propagation divergence).  Array-vectorized
                # numpy uses elsewhere are untouched.
                if not math.isfinite(val):
                    row_values.append(_format_number(null_value, precision, null_value))
                else:
                    row_values.append(_format_number(val, precision, null_value))
        line = delimiter.join(row_values)
        # M-29: the M-78 empty-value guard above covers SPACE only.  For
        # COMMA/TAB an entirely-empty (or whitespace-only) row — a
        # single-column string section whose value is empty/blank — emits
        # a BLANK line the reader skips silently (data_reader / _data_
        # section_reader: ``not stripped: continue``), dropping the row
        # on write→read (2→1).  A multi-column row with an empty CELL is
        # deliberately preserved (',1000' re-reads DESC='' correctly), so
        # the per-value replacement cannot fire here — route the blank
        # ROW through the same '-' sentinel the M-78 guard uses.
        if delimiter != " " and not line.strip():
            if not warned_empty_str:
                import warnings

                warnings.warn(
                    "Empty string curve value replaced by "
                    "'-' sentinel — roundtrip fidelity is "
                    "lost: parser cannot distinguish original "
                    "'-' from originally-empty value.",
                    stacklevel=4,
                )
                warned_empty_str = True
            row_values = ["-" if not v.strip() else v for v in row_values]
            line = delimiter.join(row_values)
        if is_las12 and len(line) > MAX_LINE_LENGTH_LAS12:
            if not warned_long:
                import warnings

                warnings.warn(
                    f"Data line exceeds 256-character limit "
                    f"(length: {len(line)}).  Lines are NOT truncated "
                    f"to avoid data loss.  Subsequent violations in this "
                    f"section will not be reported.",
                    stacklevel=4,
                )
                warned_long = True
        lines.append(line)
    return lines


def _format_number(value: float, precision: str = ".8g", null_value: float | None = None) -> str:
    """Format a numeric value with configurable precision."""
    # IT3-F-02: math.isfinite (~15x faster than np.isnan/np.isinf for
    # Python float scalars, semantically identical for scalars).
    if not math.isfinite(value):
        if null_value is not None:
            return _format_null_sentinel(null_value, precision)
        return format(float(value), precision)
    if null_value is not None and value == null_value:
        return _format_null_sentinel(null_value, precision)
    if isinstance(value, (int, np.integer)):
        # EXT-04: integer-typed values (exact Python ints from object-dtype
        # {I} arrays, np.int64 from int64 arrays) must be formatted via
        # integer formatting.  `format(int(value), precision)` converts
        # through float64 internally whenever the .Ng result needs
        # scientific notation, silently rounding values above 2^53
        # (9007199254740993 → '9007199254740992.00000000').  str(int())
        # preserves the exact decimal.
        return str(int(value))
    if value == int(value):
        # IEEE 754 negative zero (-0.0): int(-0.0) == 0 loses the sign,
        # producing "0" instead of "-0".  Use float formatting for the
        # negative-zero case so that Python's native format(-0.0, ...)
        # correctly emits "-0".
        if value == 0 and math.copysign(1.0, value) < 0:
            result = format(float(value), precision)
        else:
            result = format(int(value), precision)
    else:
        result = format(float(value), precision)
    if "e" in result.lower():
        result = _format_fixed_precision(value, precision)
    return result


def _format_null_sentinel(null_value: float, user_precision: str) -> str:
    """Format a null-value sentinel preserving its exact float identity."""
    result = repr(null_value)
    if "e" in result.lower():
        result = _format_fixed_precision(null_value, user_precision)
    return result


def _format_fixed_precision(value: float, precision: str) -> str:
    """Convert a value to fixed-point notation with magnitude-aware precision.

    W-01: a value whose first significant digit lands BEYOND the
    100-decimal cap (e.g. ``1e-150`` with a default ``.8g`` precision)
    would be silently zeroed by the fixed-point rewrite —
    ``format(1e-150, '.100f')`` == ``'0.000...0'`` — destroying the value
    on write→read.  When the capped fixed-point width cannot reach the
    first significant digit, fall back to scientific notation (readers
    parse ``e`` notation via ``float()``) so the value survives.
    """
    m = re.match(r"\.(\d+)", precision)
    sig_digits = min(int(m.group(1)), 100) if m else 8

    if value == 0:
        return format(value, f".{sig_digits}f")

    magnitude = math.floor(math.log10(abs(value)))
    decimal_places = sig_digits + max(0, (-magnitude) - 1)
    decimal_places = min(decimal_places, 100)
    decimal_places = max(decimal_places, sig_digits)

    # W-01: the first significant digit sits at decimal position
    # (-magnitude).  When the capped fixed-point width cannot reach it the
    # output would be all zeros — silently losing the value.  Emit
    # scientific notation instead, preserving all significant digits.
    if decimal_places < -magnitude:
        return format(value, f".{sig_digits}e")

    result = format(value, f".{decimal_places}f")
    return result


# F-15: The parser's FORMAT_SPEC_PATTERN offset group is bounded at 64
# characters ([-\d.]{0,64}, parser.py) to keep the ReDoS-bounded quantifier
# linear.  The writer must never emit an offset field longer than that or
# the whole {A:N} spec fails to parse (the M-56 defect class).  '0.' +
# decimal_places fits in 64 chars up to 62 decimals; use 61 to leave room
# for a leading '-' sign on negative offsets.
_MAX_OFFSET_FIXED_DECIMALS: int = 61


def _format_offset_plain(offset: float) -> str:
    """Format a float WITHOUT scientific notation for ``{A:N}`` offsets.

    M-56: Python's default ``str()`` formats values in (0, 1e-4) as
    scientific notation (``9e-05``), which the parser's FORMAT_SPEC_PATTERN
    offset group ``[-\\d.]*`` cannot parse — the entire ``{A:N}`` spec is
    then treated as description text, losing data_format and time_offset.
    Values >= 1e-4 already format as fixed-point (str() has no exponent),
    so only the scientific case is rewritten.

    F-08: the fixed-point rewrite preserves ALL significant digits of the
    offset (counted from the shortest-repr mantissa), using the same
    magnitude-aware formula as ``_format_fixed_precision`` — the earlier
    ``-magnitude + 2`` heuristic produced only ~2-3 significant digits for
    offsets in (0, 1e-4) (1.2345e-05 → '0.0000123', 0.36% error).

    F-15: the emitted field is CLAMPED to ``_MAX_OFFSET_FIXED_DECIMALS`` so
    it never exceeds the parser's 64-character offset-group cap.  Offsets
    too small to represent exactly within the cap are rounded and a LOUD
    warning is emitted — data_format and the {A:N} spec survive the
    roundtrip (unlike the pre-fix >32-char fields which failed to parse and
    leaked the literal spec into the description).
    """
    s = str(offset)
    if "e" not in s.lower():
        return s
    if offset == 0:
        return "0"
    # Significant digits in the shortest repr — the minimum needed to
    # round-trip the float64 exactly.
    mantissa = s.split("e")[0]
    sig_digits = len(mantissa.replace(".", "").replace("-", ""))
    magnitude = math.floor(math.log10(abs(offset)))
    decimal_places = sig_digits + max(0, (-magnitude) - 1)
    # E-40: the fixed-point field's TOTAL length must fit the parser's
    # 64-character offset group (FORMAT_SPEC_PATTERN ``[-\d.]{0,64}``).
    # The decimal_places clamp below bounds only the FRACTION — a large
    # non-integral offset (e.g. 1.5e100) still emits a 100+ character
    # field (101 integer digits), the whole {A:N} spec fails to parse,
    # and data_format + time_offset are lost on re-read.  Bound the
    # integer part first: sign (1) + integer digits + dot (1) +
    # decimal_places must fit in 64.
    _int_digits = len(str(int(abs(offset))))
    if _int_digits > 63:
        # Sign + integer digits alone already exceed the 64-char cap.
        # Even a fully rounded value cannot be represented — reject
        # loudly instead of emitting a spec the parser will drop.
        raise LASWriteError(
            f"time_offset {offset!r} cannot be represented in the "
            f"{{A:N}} offset field: the integer part ({_int_digits} "
            f"digits) exceeds the parser's 64-character offset-group "
            f"cap, and the spec (and the curve's data_format) would be "
            f"lost on write→read roundtrip."
        )
    _max_decimals_for_total = 63 - _int_digits
    if decimal_places > _MAX_OFFSET_FIXED_DECIMALS or decimal_places > _max_decimals_for_total:
        import warnings

        warnings.warn(
            f"time_offset {offset!r} cannot be represented exactly in the "
            f"{{A:N}} offset field (needs {decimal_places} decimal places, "
            f"capped at {_MAX_OFFSET_FIXED_DECIMALS} to stay within the "
            f"parser's 64-character offset group).  The emitted value is "
            f"rounded and may not round-trip exactly.",
            UserWarning,
            stacklevel=3,
        )
        decimal_places = min(
            decimal_places, _MAX_OFFSET_FIXED_DECIMALS, _max_decimals_for_total
        )
    return format(offset, f".{decimal_places}f")


def _warn_long_header_lines(lines: list[str], max_length: int) -> None:
    """Warn if any header section line exceeds the LAS length limit.

    The LAS 1.2 CWLS specification limits ALL lines (including header
    lines) to 256 characters; the LAS 2.0 specification applies the same
    256-char limit to WRAP=NO files.  This check covers header-section
    lines (version, well, curve, parameter, other) that were not
    previously validated — data rows are checked separately in
    ``_format_data_rows``.  N-I-16: previously gated on ``is_las12``
    only, so LAS 2.0 WRAP=NO header lines were never checked.
    """
    warned = False
    for line in lines:
        if len(line) > max_length:
            if not warned:
                import warnings

                warnings.warn(
                    f"LAS header line exceeds {max_length}-character "
                    f"limit: {line[:80]!r}... ({len(line)} chars). "
                    f"The LAS 1.2/2.0 (WRAP=NO) specification limits all "
                    f"lines (including header lines) to {max_length} "
                    f"characters.  Subsequent violations in this file "
                    f"will not be reported.",
                    stacklevel=4,
                )
                warned = True


# ── _WriterMutationGuard ────────────────────────────────────────────────


class _WriterMutationGuard:
    """Context manager that runs deferred validation after write.

    Saves a snapshot of the LASFile state that is affected by the write
    pass (wrap/dlm flags, logs/string_data/curves containers).  On the
    SUCCESS path the model is intentionally NOT restored — the model must
    honestly reflect what was written to disk (documented G-018 intent:
    e.g. ``WRAP=YES`` is written as ``NO``).  On the FAILURE path the
    saved state IS restored so the caller's model is not left partially
    mutated by an aborted write.

    The version-specific writers' ``finally`` blocks restore the data
    containers from plain ``dict``/``list`` snapshots, which permanently
    strips the ``_GuardedDict``/``_GuardedList`` mutation guards.  This
    guard re-wraps the containers so invalid mutations are still caught
    after a write (success or failure).
    """

    def __init__(
        self,
        las_file: LASFile,
        *,
        suppress_validate: bool = False,
    ) -> None:
        self._las_file = las_file
        # M-27: write() runs validate(complete=True) itself (and warns on
        # every issue); re-running it in __exit__ double-warned every
        # issue on every write.  write_las_file passes suppress_validate
        # so the __exit__ re-validation is skipped (the pre-write
        # validation already covered the same model state — the writers
        # restore their container snapshots, so the post-write state
        # cannot introduce NEW validate issues beyond the wrap/dlm
        # normalization the write itself performed).
        self._suppress_validate = suppress_validate
        self._saved_wrap: str = las_file.version.wrap
        self._saved_dlm: str = las_file.version.dlm
        # M-30: None containers are a documented-valid state for
        # directly-constructed files (models.py __setattr__ accepts
        # logs=None / string_data=None post-construction).  The raw
        # ``dict(las_file.logs)`` calls below ran BEFORE the write try
        # block, leaking a bare TypeError instead of the documented
        # LASWriteError from write_las_file's "Raises" contract.  Guard
        # the snapshot like the sibling curves/curves_order fields do;
        # _restore_saved_state / _rewrap_guards already handle None.
        self._saved_logs = dict(las_file.logs) if las_file.logs is not None else None
        self._saved_string_data = (
            dict(las_file.string_data) if las_file.string_data is not None else None
        )
        self._saved_curves_order = (
            list(las_file.curves_order) if las_file.curves_order is not None else None
        )
        self._saved_curves = list(las_file.curves) if las_file.curves is not None else None

    def __enter__(self) -> _WriterMutationGuard:
        return self

    def _restore_saved_state(self) -> None:
        """Restore the pre-write snapshot on the failure path."""
        las_file = self._las_file
        las_file.version.wrap = self._saved_wrap
        las_file.version.dlm = self._saved_dlm
        las_file.logs = self._saved_logs
        las_file.string_data = self._saved_string_data
        las_file.curves_order = self._saved_curves_order
        las_file.curves = self._saved_curves

    def _rewrap_guards(self) -> None:
        """Re-wrap data containers in guarded dict/list after a write.

        W-06: the version-specific writers restore logs/string_data/
        curves from plain dict/list snapshots in their ``finally`` blocks,
        permanently stripping the mutation guards installed at LASFile
        construction.  Re-install the guards so subsequent invalid
        mutations are still rejected.  None containers (a valid state for
        directly-constructed files) are left untouched.
        """
        las_file = self._las_file
        if las_file.logs is not None and not isinstance(las_file.logs, _GuardedDict):
            las_file.logs = _GuardedDict(las_file.logs, _container_name="LASFile.logs")
        if las_file.string_data is not None and not isinstance(las_file.string_data, _GuardedDict):
            las_file.string_data = _GuardedDict(
                las_file.string_data, _container_name="LASFile.string_data"
            )
        if las_file.curves is not None and not isinstance(las_file.curves, _GuardedList):
            las_file.curves = _GuardedList(
                las_file.curves,
                _container_name="LASFile.curves",
                _expected_type=CurveDefinition,
            )

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        import warnings

        if exc_type is not None:
            # W-07: failure path — restore the saved state so the caller's
            # model is not left partially mutated by the failed write.
            self._restore_saved_state()
        elif not self._suppress_validate:
            # M-27: skipped when write() already ran
            # validate(complete=True) — see __init__.
            try:
                issues = self._las_file.validate(complete=True)
                for msg in issues:
                    warnings.warn(msg, UserWarning, stacklevel=2)
            except Exception:
                pass

        # W-06: re-install mutation guards stripped by the writers' finally
        # blocks (runs on both success and failure paths).
        self._rewrap_guards()

        return False  # type: ignore[return-value]


# ── _WriterBase ─────────────────────────────────────────────────────────


class _WriterBase:
    """Abstract base class for version-specific LAS writers.

    Provides the template method ``write()`` that calls section writers in
    ordered sequence, plus shared utility methods and default implementations
    for sections common to some versions.

    Subclasses override version-specific section writers.
    """

    def __init__(self, las_file: LASFile, precision: str) -> None:
        self._las_file = las_file
        self._precision = precision
        self._spec = _LASVersionSpec(las_file.version.vers)

    def write(self) -> str:
        """Generate complete LAS content string (template method)."""
        import warnings

        for issue in self._las_file.validate(complete=True):
            warnings.warn(issue, stacklevel=2)
        if self._spec.is_las30:
            self._warn_string_curves_without_s_marker()

        # PF-22 (I2-13 completion): the legacy ~A data pass emits its
        # columns from the LIVE curves_order while ~C historically emitted
        # from the `curves` list's cached order.  A post-construction
        # curves_order mutation therefore desynced the ~C column order
        # from the data rows — the written file re-read with silently
        # swapped columns and no writer-side signal (the models-side
        # validate warning is suppressible).  For the legacy emission
        # paths (LAS 1.2/2.0 always; LAS 3.0 only when it falls back to
        # ~A without data_sections — the data_sections path already emits
        # in live order per I2-13 and is driven by section_curves, not
        # this list) temporarily point `curves` at the live-ordered list
        # so EVERY ~C emitter (base AND the LAS 3.0 no-data_sections
        # override) agrees with the data rows at write time.  The model is
        # restored afterwards — the write does not silently reorder the
        # user's curves list.
        _original_curves = self._las_file.curves
        _reordered_curves: list[CurveDefinition] | None = None
        if not self._spec.is_las30 or not self._las_file.data_sections:
            _reordered_curves = self._curves_in_live_order()
            if _reordered_curves is not None:
                self._las_file.curves = _reordered_curves

        try:
            lines: list[str] = []
            lines.extend(self._write_version_section())
            lines.extend(self._write_well_section())
            lines.extend(self._write_curve_section())
            lines.extend(self._write_parameter_section())
            lines.extend(self._write_other_section())
            # N-I-16: The header-line length check was gated on `is_las12` only,
            # so LAS 2.0 WRAP=NO header lines (also subject to the CWLS 256-char
            # limit per `line_length_limit_for_wrap`) were never checked.  Use
            # the effective wrap — the writers ALWAYS emit non-wrapped output
            # (WRAP=YES is overridden to NO), so the effective limit for LAS
            # 1.2/2.0 output is always 256, matching the data-row check.
            effective_wrap = (self._las_file.version.wrap or "NO").upper()
            if effective_wrap == "YES":
                effective_wrap = "NO"
            header_limit = self._spec.line_length_limit_for_wrap(effective_wrap)
            if header_limit is not None:
                _warn_long_header_lines(lines, header_limit)
            lines.extend(self._write_ascii_sections())
            return "\n".join(lines) + "\n"
        finally:
            if _reordered_curves is not None:
                self._las_file.curves = _original_curves

    # ── Section writers (overridable) ───────────────────────────────

    def _write_version_section(self) -> list[str]:
        raise NotImplementedError

    def _write_well_section(self) -> list[str]:
        """Write ~W Well section — LAS 2.0/3.0 format (VALUE before colon, desc after)."""
        lines: list[str] = []
        lines.append("~WELL INFORMATION")

        for key in self._las_file.well.entries:
            if not isinstance(key, str):
                raise TypeError(
                    f"WellSection entry key must be str, got {type(key).__name__}: {key!r}"
                )

        # N-I-19: Defensive well-key CONTENT validation.  WellSection.
        # __post_init__ rejects non-roundtrippable keys at construction,
        # but entries can still be mutated afterwards (well.entries is a
        # plain dict).  A key containing dots/spaces/colons is emitted and
        # then silently DROPPED on re-read — the parser's ~W regex
        # (DATA_LINE_PATTERN mnemonic group) cannot match it.  Reject here
        # rather than emit metadata that cannot survive a roundtrip.
        for key in self._las_file.well.entries:
            if not _MNEMONIC_PATTERN.fullmatch(key):
                raise ValueError(
                    f"WellSection entry key {key!r} contains characters "
                    f"the LAS parser cannot roundtrip.  Well keys must "
                    f"match {_MNEMONIC_PATTERN.pattern!r}."
                )

        mandatory_order = ["STRT", "STOP", "STEP", "NULL"]
        ordered_keys: list[str] = []
        _seen_upper: set[str] = set()
        for mandatory in mandatory_order:
            for key in self._las_file.well.entries:
                if key.upper() == mandatory and key.upper() not in _seen_upper:
                    ordered_keys.append(key)
                    _seen_upper.add(key.upper())
                    break
        for key in self._las_file.well.entries:
            if key in ordered_keys:
                # Already emitted by the mandatory-order loop above — not
                # a duplicate, skip.
                continue
            if key.upper() not in _seen_upper:
                ordered_keys.append(key)
                _seen_upper.add(key.upper())
                continue
            # E-31: case-variant duplicate well key.  The parser's re-read
            # identity is case-insensitive (well mnemonics are uppercased
            # at read, parser.py:3018), so emitting BOTH variants writes
            # two ~W lines for the same logical key and the re-read
            # last-wins — one value is silently lost, and a NULL-variant
            # pair (e.g. "NULL"=-999.25 plus "null"=0) can even break the
            # fill sentinel (data_reader._get_null_value is
            # case-insensitive).  Dedup at emission like the curve block
            # does (_emission_plan): refuse loudly when the values differ,
            # warn when identical.
            _kept = next(_k for _k in ordered_keys if _k.upper() == key.upper())
            if self._las_file.well.entries[_kept] != self._las_file.well.entries[key]:
                raise LASWriteError(
                    f"Well entry keys {_kept!r} and {key!r} differ only in "
                    f"case but hold DIFFERENT values "
                    f"({self._las_file.well.entries[_kept]!r} vs "
                    f"{self._las_file.well.entries[key]!r}).  The LAS "
                    f"parser treats well mnemonics case-insensitively — "
                    f"only one would survive a write→read roundtrip, "
                    f"silently losing the other.  Rename or remove one "
                    f"of the entries."
                )
            import warnings

            warnings.warn(
                f"Well entry keys {_kept!r} and {key!r} differ only in case "
                f"and hold the same value; emitting {_kept!r} only — the "
                f"case-variant duplicate is dropped.",
                UserWarning,
                stacklevel=3,
            )

        for key in ordered_keys:
            value = self._las_file.well.entries[key]
            # II-20: well.entries keys and units/descriptions keys can
            # differ in case (from_dict mnem_base=None / direct
            # construction store them verbatim), so an exact-case
            # .get(key) silently dropped the unit/description from the
            # emitted ~W line.  Use the codebase's CI well lookup
            # (data_reader._get_well_entry_ci), matching the mandatory-
            # field ordering's key.upper() convention.
            unit = _sanitize_las_value(_get_well_entry_ci(self._las_file.well.units or {}, key, ""))
            # E-36: validate the emitted unit against the parser's ~W
            # unit grammar (DATA_LINE_PATTERN unit group,
            # parser.py:318: ``[\w\-/.%°:]*``).  A unit containing any
            # other character (e.g. the whitespace in "kg : m") truncates
            # at the first out-of-class character on re-read — the unit
            # is silently shortened AND the entry VALUE is absorbed into
            # the description, destroying it on write→read.  Reject
            # loudly rather than emit metadata that cannot round-trip.
            if unit and not _WELL_UNIT_PATTERN.fullmatch(unit):
                raise LASWriteError(
                    f"Well entry '{key}' unit {unit!r} cannot be "
                    f"represented in the ~W section: the LAS parser's "
                    f"unit grammar accepts only word characters, '-', "
                    f"'/', '.', '%', '°', and ':' — any other character "
                    f"(including whitespace) truncates the unit and "
                    f"destroys the entry value on write→read roundtrip."
                )
            unit_dot = f".{unit}" if unit else "."
            # M-28: well values/descriptions are emitted mid-line (never at
            # line start) so a leading '~' must be preserved, not stripped.
            val = _sanitize_las_value(value, preserve_leading_tilde=True)
            desc = _sanitize_las_value(
                _get_well_entry_ci(self._las_file.well.descriptions or {}, key, ""),
                preserve_leading_tilde=True,
            )
            val = _escape_colons_for_las_value(val)
            desc = _escape_colons_for_las_value(desc)
            desc_str = f"  {desc}" if desc else ""
            # LAS 2.0+: MNEM.UNIT VALUE  :
            lines.append(f" {_sanitize_las_value(key)}{unit_dot}   {val}  :{desc_str}")
        lines.append("")
        return lines

    def _write_curve_section(self) -> list[str]:
        """Write ~C Curve section — LAS 1.2/2.0 simple loop."""
        lines: list[str] = []
        lines.append("~CURVE INFORMATION")

        curves = self._las_file.curves
        if (
            not curves
            and not self._spec.is_las30
            and len(self._las_file.data_sections) == 1
            and self._las_file.data_sections[0].section_curves
        ):
            # W-05: The single-section copy-back (Path A in
            # _write_ascii_legacy) runs during the ASCII data pass, which
            # happens AFTER this section is emitted.  Consult the section's
            # curve definitions directly so ~C is not emitted EMPTY while
            # ~A carries the data columns — otherwise curve metadata
            # (units, descriptions, API codes) is silently lost from the
            # output and the data is discarded on re-read.
            curves = self._las_file.data_sections[0].section_curves

        # PF-22 (I2-13 completion): the ~A data pass emits from the LIVE
        # curves_order.  When the ~C definitions come from a single-
        # section copy-back (W-05 above, top-level curves empty), that
        # live order is the top-level curves_order when set, else the
        # section's curves_order (Path A copies it to the top level during
        # the ASCII pass).  Keep ~C consistent with it so a post-
        # construction curves_order mutation (top-level or section) cannot
        # swap columns on write→read.
        if curves is not self._las_file.curves:
            _live = self._curves_in_live_order(
                curves,
                order=(
                    self._las_file.curves_order
                    if self._las_file.curves_order
                    else self._las_file.data_sections[0].curves_order
                ),
            )
            if _live is not None:
                curves = _live

        if not self._spec.is_las30:
            # W-10: dedup duplicate EMITTED mnemonics in the ~C block,
            # mirroring the LAS 3.0 path.  Two distinct CurveDefinitions
            # can emit the same name (duplicate mnemonics like DEPT(M) /
            # DEPT(FT), or a vendor rename via original_mnemonic).  The
            # parser renames the second duplicate on re-read (DEPT →
            # DEPT_2), silently altering the model identity.  A curve
            # whose M-59 reconstruction would collide falls back to its
            # OWN mnemonic (preserving distinct columns like a
            # reader-renamed IK_2); only a metadata-only duplicate (own
            # mnemonic also taken) is dropped — warn so the drop is
            # visible.
            # F-27 (W-10): is_las30=False keeps the dedup candidate free
            # of the [N] bracket — _format_curve_line never emits the
            # bracket on LAS 1.2/2.0, so the dedup key must match the
            # emitted name or a reader-renamed array curve (IK_2 with
            # original_mnemonic='IK' + array_info) would dedup against
            # 'IK[1]' instead of 'IK' and emit a duplicate IK line.
            _curve_pairs, _dropped = _emission_plan(curves, is_las30=False)
            for _curve in _dropped:
                import warnings

                # X-1 (N2b-2): the dedup warning must be ACCURATE post-fix.
                # The old parenthetical ("a re-read would rename it and
                # silently alter the model identity") described the PRE-fix
                # consequence — after the case-insensitive dedup the second
                # curve is NOT emitted to ~C, so re-read does not rename it;
                # if it carried distinct data the legacy ~A pass refuses the
                # write (LASWriteError) rather than discard the values on
                # re-read (see _write_ascii_legacy), and if its data is
                # shared with the kept definition (or absent) nothing is lost.
                warnings.warn(
                    f"Duplicate curve mnemonic '{_emitted_mnemonic(_curve)}' "
                    f"in ~C.  Keeping the first definition; the second "
                    f"curve is not re-emitted.  If it carried distinct "
                    f"data the write refuses below rather than discard "
                    f"the values on re-read; otherwise no data is lost.",
                    UserWarning,
                    stacklevel=3,
                )
            curves = [c for c, _ in _curve_pairs]
            _overrides = {id(c): o for c, o in _curve_pairs}
        else:
            _overrides = {}

        for curve in curves:
            # M-77: pass the string_data mnemonic set so a string curve
            # without data_format='S' still gets the {S} marker.  On
            # 1.2/2.0 the is_las30 gate inside _format_curve_line keeps
            # the marker off (string curves are lossy there by design and
            # validate(complete=True) already warns).
            # N2b-1: the set is built UPPER-CASED (matching
            # _format_curve_line's _mnem_key membership) so a case-variant
            # string_data key resolves the marker.
            lines.append(
                _format_curve_line(
                    curve,
                    self._spec.is_las30,
                    frozenset(_mnem_key(k) for k in self._las_file.string_data.keys()),
                    mnemonic_override=_overrides.get(id(curve)),
                )
            )

        lines.append("")
        return lines

    def _write_parameter_section(self) -> list[str]:
        """Write ~P Parameter section — LAS 1.2/2.0 flat format."""
        if not self._las_file.parameters:
            return []

        lines: list[str] = []
        lines.append("~PARAMETER INFORMATION")
        for param in self._las_file.parameters:
            lines.append(_format_parameter_line(param, self._spec.is_las30))
        lines.append("")
        return lines

    def _write_other_section(self) -> list[str]:
        """Write ~O Other section — LAS 1.2/2.0 (emit).

        W-08: ~O content is free-form text, but the two transformations
        below alter the model value on write→read, so they are reported:
        - a line starting with ``~[A-Za-z]`` has its leading ``~``
          stripped — otherwise the parser misreads the line as a NEW
          section header, terminating the ~O block and misrouting the
          remaining lines;
        - tab characters are replaced with spaces (``_sanitize_las_value``
          treats a tab as layout whitespace).
        ``#``-prefixed lines KEEP the ``_#`` escape: the parser's
        COMMENT_PATTERN drops raw ``#`` lines before the ~O accumulator
        sees them, so an unescaped ``#comment`` is lost on re-read.  The
        reverse ``_#`` → ``#`` restore on the ~O READ path is a parser
        concern (see W-08 coordination note in the fix report).
        """
        lines: list[str] = []
        if not self._las_file.other or not self._las_file.other.strip():
            return lines
        lines.append("~OTHER")
        warned_tilde = False
        warned_tab = False
        for line in self._las_file.other.splitlines():
            if not warned_tilde and _LEADING_SECTION_RE.match(line):
                import warnings

                warnings.warn(
                    "~Other line starts with '~' followed by a letter; "
                    "the leading '~' is stripped so the line is not "
                    "misread as a new LAS section header on re-read.  "
                    "The written content differs from the model value.",
                    UserWarning,
                    stacklevel=3,
                )
                warned_tilde = True
            if not warned_tab and "\t" in line:
                import warnings

                warnings.warn(
                    "~Other line contains tab characters; tabs are "
                    "replaced with spaces when written.  The written "
                    "content differs from the model value.",
                    UserWarning,
                    stacklevel=3,
                )
                warned_tab = True
            sanitized = _sanitize_las_value(line)
            if sanitized.strip():
                lines.append(sanitized)
        lines.append("")
        return lines

    def _write_ascii_sections(self) -> list[str]:
        raise NotImplementedError

    def _warn_string_curves_without_s_marker(self) -> None:
        """M-77: warn when a LAS 3.0 string curve lacks the {S} marker.

        The parser classifies a column as string ONLY from the {S} marker
        in its ~C/Definition line.  A string-data curve with an empty (or
        non-'S') data_format is emitted markerless and its values are
        re-read as numeric nulls — silent destruction.  This check covers
        BOTH the top-level (no data_sections) and per-section paths from
        the base template ``write()``; callers that pass string_data
        context into ``_format_curve_line`` also emit {S} directly.

        F-09: the M-77 {S}-forcing branch in ``_format_curve_line`` emits
        the {S} marker for any string-data curve whose mnemonic is in the
        string_mnemonics set passed for its emitted scope — the main ~C
        block passes the UNION of every scope's string_data keys and
        per-section Definitions pass the section's own keys.  When the
        curve IS in that set the values round-trip intact, so warning
        here would misdiagnose the exact scenario the fix prevents.
        Only warn when the marker is genuinely absent for the emitted
        scope (the curve's mnemonic is NOT in the union string_mnemonics
        set) — i.e. string data would actually be lost.
        """
        import warnings

        las_file = self._las_file
        # N2b-1/II-7: all 7 comparison sites here are case-insensitive —
        # the dicts are keyed by _mnem_key(mnemonic), the emitted string
        # set holds _mnem_key keys, and lookups pass _mnem_key(mnem).  A
        # case-variant string_data key ('dept_str' vs curve 'DEPT_STR')
        # must resolve so the warning fires (and _format_curve_line forces
        # {S}) instead of silently destroying the values.
        top_curves = {_mnem_key(c.mnemonic): c for c in las_file.curves or []}
        warned: set[str] = set()
        # Union of EVERY string_data mnemonic across all scopes — the set
        # the main ~C block passes to _format_curve_line
        # (_Las30Writer._all_string_mnemonics), which is a superset of
        # every per-section Definition set.  Membership here means {S} is
        # forced at emission.
        emitted_str_mnems: set[str] = {_mnem_key(k) for k in (las_file.string_data or {})}
        for ds in las_file.data_sections:
            emitted_str_mnems.update(_mnem_key(k) for k in (ds.string_data or {}))

        def _warn_for(curve: CurveDefinition, mnem: str) -> None:
            if (curve.data_format or "").upper() == "S" or mnem in warned:
                return
            # F-09/II-7: {S} is forced only when the EMITTED name (M-59
            # original_mnemonic reconstruction — the same name
            # _format_curve_line tests) is in the string_mnemonics set.  A
            # curve BFV with original_mnemonic='LLD' whose string_data is
            # keyed 'BFV' emits 'LLD' markerless — testing curve.mnemonic
            # ('BFV') here would falsely conclude "no loss" and suppress
            # the warning exactly when the values are destroyed.
            _emit_name = (
                curve.original_mnemonic
                if curve.original_mnemonic and curve.original_mnemonic != curve.mnemonic
                else curve.mnemonic
            )
            # M-35: the {S}-forcing membership in _format_curve_line
            # tests BOTH the emitted name and the curve's storage
            # mnemonic (the string_mnemonics sets are built from the
            # string_data CONTAINER KEYS).  This warning must agree: a
            # renamed curve (storage key 'BFV', emitted 'LLD') HAS its
            # marker forced via the storage key, so warning here would
            # falsely claim "the string values are lost" exactly when
            # they round-trip intact.
            if (
                _mnem_key(_emit_name) in emitted_str_mnems
                or _mnem_key(curve.mnemonic) in emitted_str_mnems
            ):
                return
            warnings.warn(
                f"LAS 3.0 string curve '{mnem}' has "
                f"data_format={(curve.data_format or '')!r} (not 'S').  "
                f"Without the {{S}} marker the parser reads this column "
                f"as numeric and the string values are lost on "
                f"write→read roundtrip.",
                UserWarning,
                stacklevel=2,
            )
            warned.add(mnem)

        # Top-level string_data scope (no-data_sections path).
        for mnem in las_file.string_data or {}:
            cd = top_curves.get(_mnem_key(mnem))
            if cd is not None:
                _warn_for(cd, mnem)
        # Per-section scopes — definitions may live in the section itself
        # or fall back to the top-level curves list.
        for ds in las_file.data_sections:
            sec_curves = {_mnem_key(c.mnemonic): c for c in ds.section_curves or []}
            for mnem in ds.string_data or {}:
                cd = sec_curves.get(_mnem_key(mnem)) or top_curves.get(_mnem_key(mnem))
                if cd is not None:
                    _warn_for(cd, mnem)

    # ── Shared helpers for version-specific writers ──────────────────

    def _curves_in_live_order(
        self,
        curves: list[CurveDefinition] | None = None,
        order: list[str] | None = None,
    ) -> list[CurveDefinition] | None:
        """Resolve a curve list into the LIVE ``curves_order`` order.

        PF-22 (I2-13 completion): the legacy ~A data pass emits its
        column header and data rows from the LIVE ``curves_order`` while
        ~C historically emitted from the ``curves`` list's cached order.
        A post-construction ``curves_order`` mutation (reorder/insert/
        reverse) therefore desynced the ~C column order from the data
        rows — the written file re-read with silently swapped columns and
        no writer-side signal (the models-side validate warning is
        suppressible).  Callers that need ~C to agree with the data rows
        resolve the emission source through this helper.

        ``curves`` defaults to ``self._las_file.curves``; ``order``
        defaults to ``self._las_file.curves_order`` (the W-05 copy-back
        path passes the section's ``curves_order`` — that is the order the
        ~A pass will use via Path A copy-back).

        Mnemonic resolution is case-insensitive (I2-22), matching the
        LAS 3.0 section paths.  Curves absent from ``order`` are appended
        at the end so metadata-only definitions still survive ~C.  An
        entry in ``order`` with no resolvable definition is skipped (the
        ~A data pass handles unresolvable columns separately).  Dedup uses
        object IDENTITY — distinct-but-equal definitions (e.g. duplicate
        mnemonics like I2-20's two ``LLD`` curves) are preserved so the
        downstream emission-dedup (``_emission_plan``) still emits its
        duplicate warning instead of silently dropping one here.

        Returns the reordered list, or None when no reorder is needed
        (empty source/order, or the source already matches the live
        order) — callers then keep their existing emission path so
        aligned models are untouched.
        """
        source = self._las_file.curves if curves is None else curves
        live_order = self._las_file.curves_order if order is None else order
        if not source or not live_order:
            return None
        by_mnem: dict[str, CurveDefinition] = {}
        for c in source:
            by_mnem.setdefault(c.mnemonic, c)
            by_mnem.setdefault(c.mnemonic.upper(), c)
        resolved: list[CurveDefinition] = []
        seen_ids: set[int] = set()
        for mnem in live_order:
            cdef = by_mnem.get(mnem) or by_mnem.get(mnem.upper())
            if cdef is not None and id(cdef) not in seen_ids:
                seen_ids.add(id(cdef))
                resolved.append(cdef)
        for c in source:
            if id(c) not in seen_ids:
                seen_ids.add(id(c))
                resolved.append(c)
        if len(resolved) == len(source) and all(
            a is b for a, b in zip(resolved, source, strict=True)
        ):
            return None
        return resolved

    def _write_ascii_legacy(self, delimiter: str, check_line_limit: bool) -> list[str]:
        """Legacy ~A data path for LAS 1.2/2.0.

        Handles data_sections copy-back (Path A) and legacy single ~A
        section output (Path C) from the original ``_write_ascii_sections``.
        """
        lines: list[str] = []
        import warnings

        # Path A: non-LAS-3.0 data_sections copy-back
        if self._las_file.data_sections:
            if len(self._las_file.data_sections) > 1:
                raise LASWriteError(
                    f"Multiple data_sections ({len(self._las_file.data_sections)}) "
                    f"are only supported for LAS 3.0 files, but version is "
                    f"{self._las_file.version.vers!r}. Cannot safely write multi-section "
                    f"data for non-LAS-3.0 format."
                )
            _ds = self._las_file.data_sections[0]
            # W-04: The copy-back below only fills EMPTY top-level
            # containers.  When a top-level container is already
            # populated, the corresponding section content is dropped.
            # Warn honestly about the actual copy-back outcome instead
            # of always claiming "Single-section data will be preserved."
            _dropped: list[str] = []
            if _ds.data and self._las_file.logs:
                _dropped.append("data")
            if _ds.string_data and self._las_file.string_data:
                _dropped.append("string data")
            if _ds.section_curves and self._las_file.curves:
                _dropped.append("curve definitions")
            if _dropped:
                warnings.warn(
                    "data_sections are only supported for LAS 3.0 files. "
                    "Falling back to single-section ~A format.  Section "
                    f"content will NOT be preserved because the "
                    f"corresponding top-level container is already "
                    f"populated: {', '.join(_dropped)}.",
                    stacklevel=3,
                )
            else:
                warnings.warn(
                    "data_sections are only supported for LAS 3.0 files. "
                    "Falling back to single-section ~A format. "
                    "Single-section data will be preserved.",
                    stacklevel=3,
                )
            if not self._las_file.logs and _ds.data:
                self._las_file.logs.update(_ds.data)
            if not self._las_file.string_data and _ds.string_data:
                self._las_file.string_data.update(_ds.string_data)
            if not self._las_file.curves_order and _ds.curves_order:
                self._las_file.curves_order = list(_ds.curves_order)
            if not self._las_file.curves and _ds.section_curves:
                self._las_file.curves = list(_ds.section_curves)

            if (
                self._las_file.curves_order
                and _ds.curves_order
                and (self._las_file.logs or self._las_file.string_data)
            ):
                existing = set(self._las_file.curves_order)
                for k in _ds.curves_order:
                    if (
                        k in self._las_file.logs or k in self._las_file.string_data
                    ) and k not in existing:
                        self._las_file.curves_order.append(k)

            if (
                self._las_file.curves
                and self._las_file.curves_order
                and len(self._las_file.curves) != len(self._las_file.curves_order)
            ):
                raise LASDataError(
                    f"curves count ({len(self._las_file.curves)}) does not match "
                    f"curves_order count ({len(self._las_file.curves_order)}) "
                    f"after copy-back. This indicates inconsistent "
                    f"LASFile construction."
                )

            if self._las_file.curves_order and (self._las_file.logs or self._las_file.string_data):
                _log_upper = (
                    {k.upper() for k in self._las_file.logs.keys()}
                    if self._las_file.logs
                    else set()
                )
                _str_upper = (
                    {k.upper() for k in self._las_file.string_data.keys()}
                    if self._las_file.string_data
                    else set()
                )
                # F-32 (I2-22 consistency): the data-key lookup is
                # case-insensitive; compare upper-cased so a case-variant
                # curves_order entry ('dept' vs data key 'DEPT') is not
                # falsely reported as uncovered (it IS emitted, not padded).
                _uncovered = [
                    _k
                    for _k in self._las_file.curves_order
                    if _k.upper() not in _log_upper and _k.upper() not in _str_upper
                ]
                if _uncovered:
                    warnings.warn(
                        f"Curve(s) {sorted(_uncovered)} appear in "
                        f"curves_order but have no data in 'logs' or "
                        f"'string_data' after copy-back.  The writer will "
                        f"pad these curves with null_value.",
                        stacklevel=3,
                    )

        # Path C: Legacy single data section (~A)
        curve_names = self._las_file.curves_order

        # F-32 (I2-22 consistency): the ~A data-key lookup
        # (_format_data_rows) and the ~C definition resolution are
        # case-insensitive.  The gate below must use the SAME resolution
        # or a case-variant curves_order entry (e.g. 'dept' while the
        # data is keyed 'DEPT') falsely trips the "none have data"
        # warning and skips the ~A block entirely.
        _data_curves = [
            n
            for n in curve_names
            if _lookup_data_array(n, self._las_file.logs, self._las_file.string_data)[0] is not None
        ]
        if curve_names and not _data_curves:
            warnings.warn(
                f"curves_order contains {len(curve_names)} curve(s) "
                f"but none have data in logs or string_data. "
                f"No data will be emitted.",
                stacklevel=3,
            )
        elif curve_names and len(_data_curves) < len(curve_names):
            # E-32: SOME curves have data but others do not (e.g. a
            # post-construction `del las.logs['GR']` — _GuardedDict has
            # no deletion overrides, so the curves_order entry survives
            # without data).  The previous code only warned when NONE
            # had data; the partial case was silently null-padded and
            # the re-read FABRICATED a -999.25 column that never existed
            # in the model.  Warn loudly at emission time so the
            # fabrication is visible.
            _missing = sorted(
                n
                for n in curve_names
                if _lookup_data_array(n, self._las_file.logs, self._las_file.string_data)[0]
                is None
            )
            warnings.warn(
                f"curves_order entries {_missing} have no data in "
                f"'logs' or 'string_data'.  The writer will null-pad "
                f"these columns with null_value; on re-read the padded "
                f"columns fabricate null rows that do not exist in the "
                f"model.",
                stacklevel=3,
            )
        if not curve_names and (self._las_file.logs or self._las_file.string_data):
            # F-36: an EMPTY curves_order with populated logs/string_data
            # is a user-inconsistent state (direct construction only —
            # from_dict rejects it, and __post_init__'s consistency
            # checks skip when curves_order is empty).  The empty-list
            # short-circuit (`any([])` is False) previously suppressed
            # the "no data" warning AND skipped the ~A block, so the
            # data was silently lost with zero diagnostics.  Fire the
            # warning explicitly so the loss is visible at write time.
            _data_keys = sorted(
                set(self._las_file.logs.keys()) | set(self._las_file.string_data.keys())
            )
            warnings.warn(
                f"curves_order is empty but logs/string_data contain "
                f"data for curve(s) {_data_keys}.  The written file will "
                f"have no ~A data section and this data will NOT be "
                f"emitted.  Add the curves to curves_order (and to "
                f"curves) to write the data.",
                stacklevel=3,
            )
        if _data_curves:
            # M-59: Keep the ~A column-header line consistent with the
            # ~C curve lines.  The ~C section now emits
            # CurveDefinition.original_mnemonic (the vendor-standard name)
            # when it differs from curve.mnemonic; the ~A header must use
            # the SAME emitted names or an external parser sees a
            # column-header/curve mismatch (pylasdev routes data
            # positionally by ~C order, so its own roundtrip is
            # unaffected — this is for file-level consistency).
            # F-32: resolve the definition case-insensitively (matching
            # the ~C emission) so a case-variant curves_order entry
            # ('dept') emits the SAME header name as the ~C block.
            _by_mnem: dict[str, CurveDefinition] = {}
            for _c in self._las_file.curves or []:
                _by_mnem.setdefault(_c.mnemonic, _c)
                _by_mnem.setdefault(_c.mnemonic.upper(), _c)

            # W-10: mirror the ~C emission's collision-free naming.  A
            # reader-renamed duplicate (IK_2 with original_mnemonic='IK')
            # falls back to its OWN mnemonic so the ~A header stays
            # collision-free and matches the ~C block.
            _header_seen: set[str] = set()

            def _header_name(name: str) -> str:
                c = _by_mnem.get(name) or _by_mnem.get(name.upper())
                candidate = (
                    c.original_mnemonic
                    if c is not None and c.original_mnemonic and c.original_mnemonic != c.mnemonic
                    else (c.mnemonic if c is not None else name)
                )
                if candidate in _header_seen and c is not None:
                    candidate = c.mnemonic
                _header_seen.add(candidate)
                return candidate

            # W-10: two DISTINCT data columns that emit the same mnemonic
            # even after the collision fallback (two curves with the same
            # own mnemonic) would produce duplicate ~A header names — on
            # re-read the parser renames the duplicate (LLD_2) and the
            # model identity is silently altered.  The legacy single-block
            # format cannot represent duplicate columns: fail loudly.
            _header_names = [_header_name(name) for name in curve_names]
            if len(set(_header_names)) != len(_header_names):
                _dups = sorted({n for n in _header_names if _header_names.count(n) > 1})
                raise LASWriteError(
                    f"curves_order contains curve(s) that would emit "
                    f"duplicate column name(s) {_dups} in the ~A header. "
                    f"LAS 1.2/2.0 cannot represent duplicate columns — "
                    f"re-read would silently rename them and alter the "
                    f"model identity."
                )

            # F-26: warn when the ~C block declares more curves than ~A
            # will have data columns.  A metadata-only curve (present in
            # `curves`, absent from `curves_order` — legal per
            # models.py:2858-2862) is emitted to ~C but gets NO ~A
            # column; on re-read the reader pads the missing column with
            # null sentinels, fabricating a data column that did not
            # exist in the model.  The check is COUNT-based (the ~C
            # emission set vs the ~A header count) — the names can
            # legitimately differ for array-info curves (a ~C curve
            # ``NMR[5]`` backed by the ``NMR`` curves_order column), but
            # a count divergence always fabricates columns on re-read.
            _c_pairs, _c_dropped = _emission_plan(
                self._las_file.curves or [], is_las30=self._spec.is_las30
            )
            _c_emitted = [
                (_o if _o is not None else _emitted_mnemonic(_c, self._spec.is_las30))
                for _c, _o in _c_pairs
            ]

            # X-1 (N2b-2 data-discard regression): the case-insensitive ~C
            # dedup drops a case-variant duplicate ('dept' next to 'DEPT')
            # from ~C, but curves_order can still carry BOTH names — the
            # ~A header then emits more columns than ~C declares and
            # re-read DISCARDS the undeclared column's data ("Extra columns
            # are discarded").  Refuse when the dropped curve carries a
            # DISTINCT data array; a dropped curve whose data is SHARED
            # with a surviving pair (case-variant alias of the same array)
            # or absent loses nothing and only warns.  The ~C-side W-01
            # warning above is accurate for that shared/absent branch.
            # M-28: the refusal previously fired ONLY on LAS 1.2/2.0 — the
            # LAS 3.0 top-level (no-data_sections) path fell through to
            # ~A and wrote a file whose re-read discards the duplicate's
            # distinct data (warned at write time, but the write succeeds
            # and the data is gone).  The LAS 3.0 per-section path already
            # enforces this same refusal, so the top-level path must match:
            # the refusal now applies on every version.  The emitted-
            # mnemonic lookups pass self._spec.is_las30 so the ~C emission
            # (which appends the [N] bracket for LAS 3.0 array curves)
            # stays the dedup identity.
            if _c_dropped:
                for _dc in _c_dropped:
                    _lost_arr, _ = _lookup_data_array(
                        _emitted_mnemonic(_dc, self._spec.is_las30),
                        self._las_file.logs or {},
                        self._las_file.string_data or {},
                    )
                    if _lost_arr is None:
                        continue
                    _shared = False
                    for _kept_c, _kept_o in _c_pairs:
                        _kept_arr, _ = _lookup_data_array(
                            _kept_o
                            or _emitted_mnemonic(_kept_c, self._spec.is_las30),
                            self._las_file.logs or {},
                            self._las_file.string_data or {},
                        )
                        if _kept_arr is _lost_arr:
                            _shared = True
                            break
                    if not _shared:
                        raise LASWriteError(
                            f"Curve '{_emitted_mnemonic(_dc, self._spec.is_las30)}' "
                            f"emits the same mnemonic as another curve in ~C "
                            f"and has data.  The single-block ~A format "
                            f"cannot represent both columns: ~A would emit "
                            f"{len(_header_names)} data column(s) but ~C "
                            f"declares only {len(_c_emitted)} curve(s), and "
                            f"re-read discards the undeclared column, "
                            f"losing the data.  Rename one of the "
                            f"colliding curves or remove its data."
                        )

            if len(_c_emitted) > len(_header_names):
                _header_upper = {h.upper() for h in _header_names}
                _no_column = [m for m in _c_emitted if m.upper() not in _header_upper]
                warnings.warn(
                    f"~C declares {len(_c_emitted)} curve(s) but ~A will "
                    f"emit only {len(_header_names)} data column(s) from "
                    f"curves_order.  Curve(s) {sorted(set(_no_column))} "
                    f"have no data column; on re-read the reader pads "
                    f"them with null values, fabricating data columns "
                    f"that do not exist in the model.  Add them to "
                    f"curves_order to give them a data column, or remove "
                    f"them from curves.",
                    UserWarning,
                    stacklevel=3,
                )

            header_line = "~A  " + "  ".join(_sanitize_las_value(n) for n in _header_names)
            # N-I-16: The ~A column-header line is appended AFTER the
            # header-section length check in `write()` (which runs before
            # `_write_ascii_sections`), so it was NEVER length-checked for
            # any version.  Data rows ARE checked (`_format_data_rows`), so
            # a long ~A header slipped through with 0 warnings while the
            # data rows below it warned.  Apply the same limit here when
            # `check_line_limit` is active (LAS 1.2 all modes, LAS 2.0
            # WRAP=NO).
            if check_line_limit and len(header_line) > MAX_LINE_LENGTH_LAS12:
                import warnings

                warnings.warn(
                    f"~A column-header line exceeds 256-character limit "
                    f"(length: {len(header_line)}).  The LAS 1.2/2.0 "
                    f"specification limits all lines (including column "
                    f"headers) to 256 characters.  Lines are NOT truncated "
                    f"to avoid data loss.",
                    stacklevel=4,
                )
            lines.append(header_line)
            lines.extend(
                _format_data_rows(
                    curve_names,
                    self._las_file.logs,
                    self._las_file.string_data,
                    _get_null_value(self._las_file.well),
                    delimiter,
                    self._precision,
                    is_las12=check_line_limit,
                )
            )

        return lines


# ── Public API: write_las_file ──────────────────────────────────────────


def write_las_file(
    file_path: str | Path,
    las_data: dict[str, Any] | LASFile,
    encoding: str = "utf-8",
    precision: str = ".8g",
) -> None:
    """Write LAS data to file.

    Args:
        file_path: Output file path.
        las_data: LAS data as dict (legacy format) or LASFile object.
        encoding: Output file encoding (default: utf-8).
        precision: Format specifier for numeric data values (default: '.8g').

    Raises:
        LASWriteError: If file cannot be written.
    """
    try:
        _validate_precision(precision)
    except ValueError as e:
        raise LASWriteError(f"Invalid precision format: {e}") from e

    # W-02: A bare precision specifier (e.g. ".5") is accepted by
    # _validate_precision but format(int(v), ".5") raises ValueError
    # ("Precision not allowed in integer format specifier") whenever a
    # numeric value is integral — depths are commonly integral, so the
    # write crashes mid-output exactly when real data exists.  Normalize
    # bare ".N" to ".Ng" so integral values format without crashing.
    precision = re.sub(r"^\.(\d+)$", r".\1g", precision)

    if precision and precision[-1] in ("n", "%"):
        raise LASWriteError(
            f"Precision format code '{precision[-1]}' in '{precision}' "
            f"is not supported for LAS output.  The 'n' format code "
            f"produces locale-dependent decimal separators and grouping. "
            f"The '%' format code multiplies values by 100 and appends "
            f"'%'.  Both corrupt numeric data on re-read.  Use 'g', 'f', "
            f"or 'e' instead."
        )

    file_path = Path(file_path)

    if isinstance(las_data, dict):
        try:
            las_file = LASFile.from_dict(las_data)
        except (ValueError, TypeError, AttributeError, PylasdevError) as e:
            raise LASWriteError(f"Cannot create LASFile from dict: {e}") from e
    elif isinstance(las_data, LASFile):
        las_file = las_data
    else:
        raise LASWriteError(
            f"write_las_file expects a dict or LASFile, got {type(las_data).__name__}"
        )

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise LASWriteError(f"Cannot create output directory {file_path.parent}: {e}") from e

    # Version dispatch: choose the correct writer class.
    spec = _LASVersionSpec(las_file.version.vers)
    if spec.is_las12:
        from ._writer_las12 import _Las12Writer

        writer: _WriterBase = _Las12Writer(las_file, precision)
    elif spec.is_las20:
        from ._writer_las20 import _Las20Writer

        writer = _Las20Writer(las_file, precision)
    elif spec.is_las30:
        from ._writer_las30 import _Las30Writer

        writer = _Las30Writer(las_file, precision)
    else:
        raise LASWriteError(
            f"Unsupported LAS version: {las_file.version.vers!r}. "
            f"Supported versions are LAS 1.2, 2.0, and 3.0."
        )

    with _WriterMutationGuard(las_file, suppress_validate=True):
        try:
            # F-152: reconcile data_sections desynced by the LAS 3.0
            # parser's F2-07 dedup writeback BEFORE the writer resolves
            # per-section emission pairs.  Two _Definition blocks
            # declaring the same mnemonic (e.g. DEPTH) rename the shared
            # global curve object DEPTH→DEPTH_2 AFTER an earlier
            # DataSection was built; that section's curves_order/data
            # keys keep the pre-rename name while its section_curves
            # carries the renamed mnemonic — a desync that made the
            # writer raise LASWriteError (and to_dict→from_dict raise
            # LASDataError).  _WriterBase.write() also runs
            # validate(complete=True), which heals the same state via
            # DataSection.validate(); the explicit loop here guarantees
            # the reconcile even for write paths that skip validation.
            for _ds in las_file.data_sections:
                _ds._reconcile_dedup_renamed_curves()
            content = writer.write()
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, PylasdevError) as e:
            raise LASWriteError(f"Failed to generate LAS file content: {e}") from e

        try:
            target_dir = str(file_path.parent)
            fd, tmp_path = tempfile.mkstemp(dir=target_dir, prefix=".tmp_", suffix=file_path.name)
            try:
                with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
                    f.write(content)
                os.replace(tmp_path, file_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, UnicodeError, LookupError) as e:
            raise LASWriteError(f"Cannot write to {file_path}: {e}") from e
