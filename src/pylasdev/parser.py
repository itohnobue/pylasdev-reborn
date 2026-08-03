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
import sys
import threading
import types
import warnings
from collections.abc import Sequence
from itertools import groupby
from typing import Any, ClassVar

import numpy as np

from . import data_reader as _data_reader
from ._las30_data import (
    AsciiDataContext,
    _reconcile_null_sentinels,
    process_ascii_data,
)
from ._parser_state import _ParserState
from ._section_transition import _SectionTransitionHandler
from ._version_spec import _LASVersionSpec
from .data_reader import (
    _parse_float_with_d_notation,
)
from .exceptions import LASDataError, LASParseError
from .mnem_base import build_mnemonic_lookup
from .models import (
    _VALID_DATA_FORMATS,
    ArrayElementInfo,
    CurveDefinition,
    LASFile,
    ParameterEntry,
    ParameterZone,
    _GuardedList,
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
# F-21: Thread-local storage to prevent race conditions when concurrent
# callers use different desanitize values.  Module-level attribute
# setting (e.g. data_reader's ``_DESANITIZE_ENABLED = desanitize``)
# is intercepted via ``_DesanitizeModule.__setattr__`` and routed to
# per-thread storage.  Truthiness checks via ``_DesanitizeProxy.__bool__``
# also route to per-thread storage.
_desanitize_storage = threading.local()
_desanitize_storage.enabled = True  # Default on the importing thread.


def _is_desanitize_enabled() -> bool:
    """Return True if desanitization is enabled (thread-local, default True)."""
    return getattr(_desanitize_storage, "enabled", True)


def _set_desanitize_enabled(value: bool) -> None:
    """Set the desanitization flag for the current thread."""
    _desanitize_storage.enabled = value


class _DesanitizeModule(types.ModuleType):
    """Module subclass that routes ``_DESANITIZE_ENABLED`` reads and writes
    to thread-local storage.  This intercepts the ``= desanitize`` assignment
    in data_reader.py and ``if not _DESANITIZE_ENABLED:`` truthiness checks
    in this module and data_reader, all without modifying those modules.
    """

    def __getattr__(self, name: str) -> object:
        if name == "_DESANITIZE_ENABLED":
            return _is_desanitize_enabled()
        raise AttributeError(f"module '{__name__}' has no attribute {name!r}")

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_DESANITIZE_ENABLED":
            _set_desanitize_enabled(bool(value))
        else:
            super().__setattr__(name, value)


# Install the custom module class so that ``_DESANITIZE_ENABLED``
# access is intercepted regardless of which module performs it.
_sys_mod = sys.modules[__name__]
_sys_mod.__class__ = _DesanitizeModule
# Remove _DESANITIZE_ENABLED from the module's __dict__ so that reads
# and writes fall through to __getattr__ / __setattr__.  The existing
# proxy instance would otherwise shadow our custom behaviour.
_sys_mod.__dict__.pop("_DESANITIZE_ENABLED", None)
del _sys_mod

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
# I2-05: The data reader advertises MAX_CURVES=100K columns per data line
# (_resolve_max_tokens_per_line → MAX_CURVES), so a legitimately wide data
# row (e.g. ~9,500 columns ≈ 57KB) exceeded the old 50KB cap and was
# hard-rejected by _parse_line while data_reader would have accepted it —
# the advertised capability was unreachable.  Raise the cap to cover
# MAX_CURVES times a per-token budget (24 chars: covers "-999.25000000",
# 19-char {I} integers, and realistic string tokens) so a data row carrying
# the advertised column count parses without error.  The regex itself is
# still bypassed above _SAFE_REGEX_LINE_LENGTH (2000) by the manual scan,
# and MAX_FIELD_LENGTH (100K) still bounds each captured group — the
# raised line cap only bounds the raw line allocation, which max_file_size
# already bounds at the file level.
MAX_LINE_LENGTH = max(50_000, _data_reader.MAX_CURVES * 24)

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

# M-69: Sentinel stored in a deferred (pre-~V) data section's curve_end slot
# when a "| CURVE" pipe association was seen but the main curve scope was not
# yet known (main_curve_end == -1 — curves not yet parsed).  _replay_deferred_well
# re-resolves the sentinel against the now-known main curve block via
# _resolve_main_curve_scope().  Without this, the stored None ("all curves from
# start") would be interpreted against the FINAL curve list, producing phantom
# columns from later _Definition sections.  -2 is safe: -1 already means "unset"
# for main_curve_end and negative indices never occur in valid curve slices.
# The sentinel only ever lives inside the deferred tuple — it is resolved before
# any AsciiDataContext is built, so no consumer ever sees it.
_DEFERRED_MAIN_CURVE_SCOPE = -2

# PARS-06: Sentinel stored in a deferred (pre-~V) data section's curve_end
# slot when the section's pipe target is a _Definition that has NOT been
# parsed yet (forward pipe — e.g. "~ASCII | Core_Definition" appears before
# "~Core_Definition").  At defer time the classification code cannot resolve
# the scope (definition_curve_ranges lacks the target), so it falls to the
# unrecognized-pipe reset (0, None) — and at replay the bare None scope maps
# data against the FINAL curve list, producing phantom columns from later
# _Definition sections.  The pipe target is recorded per deferred group in
# LASParser._deferred_pipe_targets; _replay_deferred_well resolves the sentinel
# against the now-known definition range once the definition has been parsed.
# -3 is safe: -1 means "unset" for main_curve_end, -2 is the main-curve-scope
# sentinel, and negative indices never occur in valid curve slices.
_DEFERRED_PIPE_SCOPE = -3

# M-01: Hoist regex and frozenset to module level — previously re-allocated
# per call in _validate_curve_data_format (hot path, up to
# _data_reader.MAX_CURVES=100K calls per file).
_KNOWN_CURVE_FORMATS: frozenset[str] = frozenset({"F", "E", "D", "S", "A", "I"})
# PARS-01: The S/A branch previously used `[SA]\w*`, which over-accepted
# ANY word starting with S or A (e.g. {API}, {SAND}, {SLATE}) as a valid
# LAS data format.  Only the single-letter codes S and A — optionally with
# semicolon-separated suffix groups (e.g. "S;LITH") — are legitimate LAS
# format specifiers.  `[SA](?:;[\w.]+)*` accepts "S", "A", "S;LITH",
# "A;X" and rejects "API"/"SAND"/"SLATE".
_FORMAT_SPEC_RE = re.compile(r"^(?:[FEDI](?:\d+(?:\.\d+)?(?:[ED][+-]?\d+)?)?|[SA](?:;[\w.]+)*)$")

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
# The delimiter colon is the FIRST structurally-valid colon (the colon
# separating VALUE from DESCRIPTION).  Descriptions may legitimately contain
# ": " (e.g., "GR.API 45.5 : Gamma Ray : API" → desc="Gamma Ray : API").
# The value group uses non-greedy (.*?) matching so the regex engine stops
# at the first colon that meets the separator criteria instead of
# backtracking to the last colon and silently truncating the description.
#
# I2F-01: The previous alternation (\s+:\s*|\s*:\s+|:\s*$) had overlapping
# alternatives (both \s+:\s* and \s*:\s+ match when whitespace exists on
# both sides of the colon) that caused catastrophic O(n^3) regex
# backtracking on long lines with many spaces and colons.
#
# The fix uses non-overlapping alternatives:
#   (1) \s++:\s*+  — at least one whitespace BEFORE colon, optional after
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
# PARS-11 (pre-fix audit F-01): The remaining O(n^3) blowup comes from the
# GREEDY unit→value `\s+` (line 256) backtracking against the lazy
# `(?P<value>.*?)` — for a no-colon whitespace-run line the `\s+` gives
# back one space at a time while the value re-expands and the separator's
# own `\s+` re-scans the remaining run, giving Σ O(n)·O(n) = O(n³).
# Possessive quantifiers (`\s++`, `\s*+`) never give back matched
# whitespace, making the whole pattern linear.  Possessive differs from
# bounded (`\s{1,64}`) in being semantics-preserving: it only differs
# where backtracking would have produced a *different* match — which
# never happens on colon-bearing lines (verified: identical groupdicts on
# 14 realistic line shapes).  The bounded alternative would silently
# misparse >64-char whitespace runs (column-aligned headers with large
# gaps) → rejected.
#
# DEVIATION FROM AUDIT F-01 (documented for the second-opinion reviewer):
# The audit recommended `\s++:\s*+` as the ONLY separator whitespace
# alternative.  That alone breaks a REAL test-data shape — the
# empty-value leading-colon line `` BLOWD.  :BLOW DESCRIPTION {S}``
# (sample_las3.0_spec.las ~Test_Definition): after the possessive
# unit→value `\s++` consumes the space run, the separator's own `\s+`
# finds no whitespace left before the colon and the match FAILS, so the
# line falls to VALUE_ONLY_PATTERN and the curve loses its description.
# The ORIGINAL greedy pattern matched it by letting the unit→value `\s+`
# give back ONE space so the separator `\s+:\s*` could consume `` :``.
# To preserve that shape with linear runtime, a THIRD separator
# alternative was added: `(?<=\s):(?!.*:(?=\s|$))\s*+` — a zero-width
# lookbehind sees the whitespace already consumed by `\s++`, and the
# negative lookahead `(?!.*:(?=\s|$))` prevents it from firing at a
# leading colon when a LATER valid colon exists (preserving the old
# first-structurally-valid-colon semantics, e.g. ``WELL.  :Oil Well #1 :
# desc`` → value=':Oil Well #1').  Validated: identical groupdicts to the
# pre-fix pattern on 17 shapes (including BLOWD and I2-11's leading-colon
# shape), and O(n) on 2000/2001/20000/100000-char no-colon and
# colon-dense adversarial lines (0.005ms vs 3658ms pre-fix at 2000).
#
# For additional defense-in-depth, _match_data_line uses a manual scan
# fallback for lines >2000 chars to bypass regex entirely.
DATA_LINE_PATTERN = re.compile(
    r"^\s*"
    r"(?P<mnemonic>[\w\-]+(?:\[\d+\])?)"  # mnemonic: word chars + hyphen + optional [N] array index
    r"\s*"  # optional whitespace before dot (common in LAS files)
    r"\."  # literal dot separator
    # N-I-22: Unit class widened from [\w\-/]* to also accept %, degree
    # sign, and '.' so common real-world LAS units survive roundtrip:
    # "PHIT.%", "TEMP.°C", "RT.ohm.m" (previously the whole curve line
    # failed to match and the curve + data column were silently dropped).
    # F-23 (b): also accept ':' so colon-in-unit lines parse (lasio
    # parity: "TIME.hh:mm 23:15 21-JAN-2001 : Time Logger" → unit
    # 'hh:mm', value '23:15 21-JAN-2001').  The unit→value `\s++` is
    # possessive, so a ':' in the unit cannot be misread as the colon
    # separator (the separator requires whitespace before the colon).
    r"(?P<unit>[\w\-/.%°:]*)"  # unit: optional; letters/digits, -, /, ., %, °, :
    r"\s++"  # whitespace separator (PARS-11: possessive — no backtracking)
    r"(?P<value>.*?)"  # value: non-greedy to find FIRST structurally-valid colon (P-04)
    r"(\s++:\s*+|:(?=\s)|(?<=\s):(?!.*:(?=\s|$))\s*+|:\s*$)"  # colon separator (PARS-11)
    r"(?P<description>.*?)"  # description: rest of line
    r"\s*+$"  # PARS-11: possessive — trailing whitespace is atomic
)

# Simpler pattern for lines without description (value-only)
VALUE_ONLY_PATTERN = re.compile(
    r"^\s*"
    r"(?P<mnemonic>[\w\-]+(?:\[\d+\])?)"  # mnemonic with optional [N] array index
    r"\s*"  # optional whitespace before dot
    r"\."
    # N-I-22: widened unit class (same as DATA_LINE_PATTERN) — accepts %,
    # degree sign, and '.' in units.
    r"(?P<unit>[\w\-/.%°:]*)"
    r"\s+"
    r"(?P<value>.+?)"
    r"\s*$"
)

# F-23 (a): lasio missing-period fallback (lasio name_missing_period_re).
# Some real-world files omit the dot between mnemonic and colon
# ("HOLE DIA : 85.7").  lasio's documented behavior: "take everything left
# of the first colon as the mnemonic and everything right as the value,
# with empty unit and description" — whitespace inside the mnemonic is
# preserved ("HOLE DIA").  Only reached after the dot-anchored patterns
# fail, so well-formed ``MNEM.UNIT VALUE : DESC`` lines are unaffected.
# N-I-22/colon-in-unit: the unit class in the dot-anchored patterns accepts
# ':' so ``TIME.hh:mm`` units parse (unit='hh:mm', lasio parity) instead of
# dropping the whole line.
NO_PERIOD_PATTERN = re.compile(
    r"^\s*"
    r"(?P<mnemonic>[\w\-]+(?:\s+[\w\-]+)*)"  # name may contain spaces (HOLE DIA)
    r"(?P<unit>)"  # no unit in the missing-period form — empty for caller parity
    r"\s*:\s*"
    r"(?P<value>.*?)"  # everything right of the first colon
    r"\s*$"
)

# F-23 (a): The mnemonic grammar the CurveDefinition/ParameterEntry models
# can represent (mirrors DATA_LINE_PATTERN's mnemonic group and
# models._MNEMONIC_PATTERN).  A missing-period line whose mnemonic fails
# this (e.g. "HOLE DIA" with a space) is kept out of those models — they
# reject embedded spaces at construction.
_MNEMONIC_LINE_RE = re.compile(r"^[\w\-]+(?:\[\d+\])?$")

# I2-10/I2-11 (pre-fix audit F-06): Prefix portion of DATA_LINE_PATTERN
# used by _manual_colon_scan to locate the unit→value whitespace run.
# Matches [ws] MNEMONIC [ws] . UNIT — the same prefix grammar as the
# primary pattern (mnemonic class, optional array index, unit class).
# After this match the unit→value `\s+` run is consumed greedily to the
# value-start position, mirroring the primary pattern's possessive `\s++`.
_MANUAL_PREFIX_RE = re.compile(
    r"^\s*"
    r"(?P<mnemonic>[\w\-]+(?:\[\d+\])?)"  # mnemonic with optional [N] array index
    r"\s*"  # optional whitespace before dot
    r"\."
    r"(?P<unit>[\w\-/.%°:]*)"  # unit class (same as DATA_LINE_PATTERN, incl. ':')
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
# H-04 (M-60/M-61 family): The lazy [^}:\s]*? + anchored \s*\} backtracks
# quadratically when the tail cannot match (every '{' is a potential start
# that expands one char at a time to end-of-string).  Bound the quantifiers
# so each start position is capped: [^}:\s]{0,64}? / [-\d.]{0,64} / \s{0,64}
# → O(64·n) linear.  Real LAS format specifiers are <= 11 chars
# ({DD/MM/YYYY} = 10, {F8.3:5.5} = 8); 64 preserves every realistic case.
# A >64-char brace body is not a valid LAS format (validation rejects it
# and F-REV-01 preserves it as text), so model behavior is unchanged.
# F-15 (M-56): the OFFSET group was {0,32} but the writer's fixed-point
# re-emission of tiny {A:N} offsets (e.g. 1e-30 → 32-char "0.000…" field)
# exceeded it, so the whole {A:…} spec failed to parse and the literal
# leaked into the description.  The offset group is widened to {0,64} —
# still bounded, so the linear-time guarantee holds — and the writer
# clamps its output to <=64 chars (_MAX_OFFSET_FIXED_DECIMALS in
# _writer_base.py), so every emitted offset round-trips.
FORMAT_SPEC_PATTERN = re.compile(
    r"\{(?P<format>[A-Za-z][^}:\s]{0,64}?)(?::(?P<offset>[-\d.]{0,64}))?\s{0,64}\}"
)

# LAS 3.0: Zone association via pipe (e.g., | Run[1], | Zone[2]).
# F-M16: Support zone names containing spaces (e.g., "| Main Zone").
# N-I-02(b): Negative lookbehind (?<!\\) skips backslash-escaped pipes
# (\|) so a genuine pipe in a parameter description written by the writer
# (which escapes literal pipes) is not misparsed as a zone association.
# M-12: The zone group also accepts escaped pipes (\\\|) so a writer
# escaped pipe INSIDE the zone name (e.g. "| Zone\|X[2]" from zone_name
# "Zone|X") stays within the zone text; the parser unescapes it after
# extraction.
# M-65: The zone group is widened to the M-65 punctuation class
# (colon/semicolon/dot/slash) so zone names like "Run:1" roundtrip instead
# of making the ENTIRE association unparseable (zone dropped + raw text
# leaked into the description).  Brackets '['/']' are deliberately NOT in
# the class — they conflict with the zone-index notation.
ZONE_ASSOC_PATTERN = re.compile(
    r"(?<!\\)\|\s*(?P<zone>(?:[\w\-.:;/]|\\\|)+(?:\s+(?:[\w\-.:;/]|\\\|)+)*)(?:\[(?P<index>\d+)\])?$"
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
    # M-61: `(\s+)_:` is quadratic on long whitespace runs NOT followed by
    # `_:` — `\s+` greedily consumes the run, `_:` fails, `\s+` backtracks
    # one char at a time → O(k²).  A fixed-width lookbehind `(?<=\s)_:` has
    # no quantifier to backtrack → linear.  Semantics identical: the
    # single-char lookbehind preserves the whitespace (zero-width) and the
    # matched `_:` is replaced by `:`, exactly like `(\s+)_:` → `\1:`.
    value = re.sub(r"(?<=\s)_:", ":", value)
    return value


def _unescape_pipes_for_las_value(value: str) -> str:
    """Reverse the ``_escape_pipes_for_las_value`` transformation.

    The writer escapes literal pipes in parameter descriptions
    (``|`` → ``\\|``) so genuine description text containing a pipe is
    not misparsed as a LAS 3.0 zone association (``| Zone``) on re-read.
    This function restores the original pipe.

    .. note::

       Legitimate backslash-pipe text in original data (e.g., a literal
       ``\\|`` in a description) will be incorrectly unescaped.  This is
       the same trade-off acknowledged by the colon-escape helpers — the
       roundtrip loss is limited to the contrived case where user data
       naturally contains the escape-artifact pattern.
    """
    return value.replace("\\|", "|")


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

    F-25: Case 2 applies ONLY when the ``_#`` is the first non-whitespace
    content (preceded exclusively by leading whitespace) — the writer's
    actual escape scope.  Internal ``" _#"`` content the writer never
    escapes (e.g. ``"ACME _#Oil Corp"``) is preserved unchanged.
    """
    if not _is_desanitize_enabled():
        return value
    if value.startswith("_#"):
        return value[1:]
    # M-85: The writer escapes a FIRST-column string value starting with
    # '~'+non-letter (or bare '~') as '_~' so the emitted data line never
    # starts with '~' (which the reader skips as a section-header
    # heuristic — dropping the whole row).  Restore the leading '~'.
    # The escape is position-independent on the read side (any string
    # value that legitimately begins '_~' is a contrived case, same
    # trade-off as '_#').
    if value.startswith("_~"):
        return value[1:]
    # Case 2: whitespace-prefixed value with sanitized _# (e.g., " _#comment")
    # F-25: Restrict to the writer's ACTUAL escape scope.  The writer
    # (_sanitize_las_value, _writer_base.py:122-127) escapes '#' ONLY at
    # value start (startswith("#")) or after LEADING whitespace
    # (lstrip().startswith("#")).  The previous blanket
    # ``value.find("_#") + whitespace-before`` unescaped INTERNAL " _#
    # content the writer never escapes (e.g. "ACME _#Oil Corp" → "ACME
    # #Oil Corp") — a silent write→read corruption.  Only an "_#" preceded
    # exclusively by leading whitespace (the FIRST non-whitespace content)
    # is a writer escape artifact; mid-value "_#" is preserved, mirroring
    # _desanitize_other_line's scoped restore.
    stripped = value.lstrip()
    if len(stripped) < len(value) and stripped.startswith("_#"):
        leading = value[: len(value) - len(stripped)]
        return leading + stripped[1:]
    return value


def _desanitize_other_line(line: str) -> str:
    """Scoped W-08 restore for ~O (other) lines — reverse ONLY the escapes
    the ~O writer actually emits.

    PF-02 (regression fix): ``_parse_other`` previously ran every ~O line
    through the blanket ``_desanitize_las_value``, which also reversed the
    data-row ``_~`` escape and ANY whitespace-adjacent ``_#`` — escapes the
    ~O writer (``_sanitize_las_value``, _writer_base.py:96-130) NEVER emits.
    Genuine ``_~``-prefixed lines and mid-line ``_#`` content were silently
    altered on write→read.

    The ~O writer's actual escape scope is narrow: a line whose content
    begins with ``#`` (at the very start, or after leading whitespace) is
    prefixed with ``_`` so the parser's COMMENT_PATTERN does not drop it.
    This restores exactly those two positions:

    1. ``line.startswith("_#")`` → strip the ``_`` (``_#comment`` → ``#comment``).
    2. ``_#`` preceded ONLY by leading whitespace → strip the ``_``
       (`` _#comment`` → `` #comment``), mirroring the writer's
       whitespace-preserving escape.

    Everything else — ``_~`` anywhere (the ~O writer never emits it), and
    mid-line ``_#`` — is preserved unchanged.

    .. note::

       A line-start ``_#...`` is byte-identical whether it is a writer
       escape for ``#...`` or genuine ``_#...`` content, so the W-08
       convention (restore writer escapes) wins at position 0.  Mid-line and
       ``_~`` content carry no such ambiguity and are preserved.
    """
    if not _is_desanitize_enabled():
        return line
    if line.startswith("_#"):
        return line[1:]
    stripped = line.lstrip()
    if len(stripped) < len(line) and stripped.startswith("_#"):
        leading = line[: len(line) - len(stripped)]
        return leading + stripped[1:]
    return line


# F-27: Reusable cross-section curve count validation — extracted from
# _validate_cross_section_consistency so the from_dict path (models.py)
# can also call it.  Takes a list of DataSection objects and warns when
# the number of data columns (data + string_data) does not match the
# number of declared section_curves.
def _validate_data_section_column_counts(data_sections: Sequence[Any]) -> None:
    """Validate curve-count vs data-column-count per data section."""
    for ds in data_sections:
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


# F-088 / F-102: Shared format specifier validation — extracted from
# _process_ascii_data so _parse_curve can also invoke it.  This closes
# the deferred-validation gap where metadata-only LAS 3.0 files (no ~A
# data section) bypassed format checking because the validator sat inside
# data-processing code.  Also used by _process_ascii_data (replaces the
# inlined check) so both call sites share the same validator.
def _validate_curve_data_format(data_format: str, mnemonic: str, line_no: int = 0) -> None:
    """Validate a curve data_format against known LAS format specifiers.

    Accepts single-letter codes (F, E, D, S, A) and extended Fortran-style
    format specifiers (e.g., "F8.3", "E10.2", "E0.00E+00").  Rejects
    non-numeric templates such as "DEG", "DD/MM/YYYY" which are metadata
    strings, not LAS data format specifiers.

    Args:
        data_format: The format specifier string (already uppercased).
        mnemonic: The curve mnemonic for error messages.
        line_no: Optional line number for error messages (0 = omitted).

    Raises:
        LASParseError: If the format specifier is not a recognised single-letter
            code or valid extended format.
    """
    if not data_format:
        return
    if data_format in _KNOWN_CURVE_FORMATS or _FORMAT_SPEC_RE.match(data_format):
        return
    prefix = f"Line {line_no}: " if line_no else ""
    raise LASParseError(
        f"{prefix}curve '{mnemonic}' has unsupported format "
        f"specifier '{{{data_format}}}'. Non-numeric format types "
        f"(e.g., {{DEG}}, date templates) are not valid LAS "
        f"data format specifiers. "
        f"Valid values: {', '.join(sorted(_VALID_DATA_FORMATS))}"
    )


def _is_string_data_curve(curve: CurveDefinition) -> bool:
    """Return True when the curve stores string (non-numeric) data.

    Mirrors ``_data_section_reader._detect_string_curves`` semantics:
    data_format "S", or "A" without array_info.  LAS 3.0 string sections
    ({S} / plain {A}) hold string data rows whose values may legitimately
    equal a curve mnemonic — the standalone-mnemonic-header detector must
    not fire for them (F-19).
    """
    _df = (curve.data_format or "").upper()
    return _df == "S" or (_df == "A" and curve.array_info is None)


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
            well_format: LAS 1.2 well section format convention.
                Accepted for API compatibility: ``"auto"`` (default)
                heuristically detects the per-field convention,
                ``"cwls"`` and ``"lasio"`` are handled with the same
                extraction logic (mandatory numeric fields use
                value-before-colon; non-mandatory fields use the CWLS
                1989 label-left/value-right layout with the value after
                the colon, matching lasio's behavior on all layouts).
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
        # build_mnemonic_lookup walks chains like BK-3 → BK → BFV to reach
        # the terminal canonical name.  Uses first-wins semantics so
        # canonical uppercase entries take priority over lowercase aliases.
        self._mnem_base_upper = build_mnemonic_lookup(self.mnem_base)
        self.source_file: str = ""
        self._state = _ParserState()
        self._transition_handler = _SectionTransitionHandler(self)
        # PXM-01: Initialize the well-collision tracking here (not only in
        # _reset) so tests that monkeypatch _reset() — e.g.
        # TestValidateCrossSectionConsistency's _inject_mismatch — still
        # get the attributes.  Same pattern as _section_pipe_targets below.
        self._resolved_well_names: dict[str, str] = {}
        self._warned_well_collisions: set[tuple[str, str]] = set()
        self._reset()

    def _reset(self) -> None:
        """Reset parser state for a new file."""
        self.las_file = LASFile()
        self._state.reset()
        self._line_no = 0  # F-049: Track current line number for error messages
        # M-36: N-I-30 mnem_base resolution-collision tracking for curve
        # names — reset per file.  Keeps the ORIGINAL mnemonic for a curve
        # whose raw name resolves to a canonical already taken by another
        # curve (e.g. LLD/LLS → BFV), mirroring models.from_dict.
        self._resolved_curve_names: dict[str, str] = {}
        self._warned_collisions: set[tuple[str, str]] = set()
        # PXM-01: N-I-30 mnem_base resolution-collision tracking for WELL
        # names — reset per file.  Mirrors models.from_dict _norm_well_mnem
        # (M-44): two distinct raw well mnemonics resolving to the same
        # canonical (e.g. LLD/LLS → BFV in a dual-laterolog file) previously
        # last-won with only a warning (the LLD value was silently dropped).
        # The collision-aware normalization keeps the ORIGINAL mnemonic for
        # the colliding entry (and re-keys an earlier alias when the
        # canonical name itself arrives later), so BOTH values survive.
        # Annotated only in __init__ (mypy no-redef); this is the per-file
        # reset assignment.
        self._resolved_well_names = {}
        self._warned_well_collisions = set()
        # M-69: Pipe target of the CURRENT data section ("" | CURVE", "| C",
        # "| X_Definition", or None when no pipe).  Set during section
        # classification; consumed by _parse_ascii_data's pre-~V deferral so
        # a "| CURVE" scope that is unresolvable at defer time can be
        # re-resolved against the main curve block at replay.
        self._current_pipe_target: str | None = None
        # PARS-06: Pipe target of each DEFERRED (pre-~V) data section that
        # used a forward "| X_Definition" pipe (the target definition was
        # not yet parsed at defer time — definition_curve_ranges lacked it,
        # so classification reset the scope to (0, None)).  Keyed by the
        # deferred group key (section_type, section_name, section_idx) so
        # _replay_deferred_well can resolve _DEFERRED_PIPE_SCOPE to the
        # definition's range once it has been parsed.  The deferred tuple
        # itself only stores int|None in curve_end, so the target name is
        # recorded here, parallel to the tuple.
        self._deferred_pipe_targets: dict[tuple[str, str, int], str] = {}
        # PARS-02: Parallel-to-_state.section_sequence pipe targets.  The
        # section label loses the pipe target (e.g. "~ASCII | CURVE" is
        # labelled "ASCII"), so the cross-section consistency checker could
        # not resolve "| CURVE"/"| C" to __MAIN__ and fired spurious
        # "~ASCII before ~LOG_DEFINITION" / "main curve definition has no
        # corresponding data section" warnings on canonical LAS 3.0 files.
        # One entry per section_sequence entry (appended in _parse_line
        # right after enter_new_section).  None when the section had no pipe.
        # Initialized here (not only in _reset) so tests that monkeypatch
        # _reset() still get the attribute.
        self._section_pipe_targets: list[str | None] = []
        # DR-M2 (Stage 10): per-section cache of the standalone-mnemonic-
        # header match set, keyed by (section_curve_start_idx,
        # section_curve_end_idx, len(las_file.curves)).  The M12/DR-M2
        # 2..section_count partial-header clause lets short rows through the
        # count gate, so the O(section_count) slice + mnemonic-set build must
        # happen ONCE per section, not per data line — otherwise the F-22
        # count-reject-fast property regresses (a 10-token row against a
        # 100K-curve scope would pay O(100K) per row → CPU-exhaustion DoS
        # returns).
        self._mnemonic_header_scope_cache: dict[
            tuple[int, int | None, int], tuple[set[str], bool]
        ] = {}

    @property
    def data_line_count(self) -> int:
        """Public accessor for pre-scanned data line count."""
        return self._state.data_line_count

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

        for line_no, line in enumerate(lines, 1):
            self._line_no = line_no  # F-049: Track line number for error messages
            self._parse_line(line)

        # PARS-06/I2-09: The LAST section is never "left" —
        # capture_current_state / _save_c_curve_range only run when a NEW
        # section header appears, so a trailing ~C block's curve range —
        # the definition name for a _Definition, or __MAIN__/__MAIN_ALL__
        # for a plain ~C — is never recorded in definition_curve_ranges.
        # Deferred data replayed below (and _resolve_main_curve_scope)
        # needs that range to resolve a bare (I2-09) or forward-piped
        # (PARS-06) scope.  Mirror _save_c_curve_range for the final
        # section here, after all lines (and thus all curves) are parsed.
        if self._state.current_section == "C":
            _final_start = self._state.section_curve_start_idx
            _final_end = len(self.las_file.curves)
            if self._state.current_definition_name is not None:
                self._state.definition_curve_ranges[self._state.current_definition_name] = (
                    _final_start,
                    _final_end,
                )
            else:
                _prev_all = self._state.definition_curve_ranges.get("__MAIN_ALL__")
                if _prev_all is not None:
                    _all_start = min(_prev_all[0], _final_start)
                    _all_end = max(_prev_all[1], _final_end)
                else:
                    _all_start, _all_end = _final_start, _final_end
                self._state.definition_curve_ranges["__MAIN_ALL__"] = (_all_start, _all_end)
                self._state.definition_curve_ranges["__MAIN__"] = (_final_start, _final_end)

        # F-P06: Re-process well entries parsed before ~V was known.
        # If the version turns out to be LAS 1.2, overwrite buffered
        # entries with the correct value/description swap.
        # F-21: final=True — every section (including a trailing ~C block,
        # whose range was recorded above) has been processed, so a deferred
        # bare LOG_DATA group re-queued by a mid-parse flush can now be
        # resolved against the complete curve list.
        self._replay_deferred_well(final=True)

        # N-I-01: The missing-~V validation must be gated on the ACTUAL
        # parsed source, not `content.strip()`.  When callers pass `lines=`
        # (PERF-01 optimization) with empty `content`, the old check saw an
        # empty string and silently bypassed the LASParseError — returning
        # half-validated data with a misleading "empty content" warning,
        # while the same input via `content=` raised.  `has_content` is
        # derived from the effective line list (the same source the parser
        # actually consumed), so both argument paths behave identically.
        has_content = any(ln.strip() for ln in lines)

        # Validation: a valid LAS file must have a ~V section
        if not self._state.version_found and has_content:
            raise LASParseError(
                "Content does not appear to be a valid LAS file: "
                "missing required ~V (Version Information) section."
            )

        # Warn when empty/whitespace-only content is parsed without a ~V section.
        # An empty file produces a default LASFile with version "2.0" — this is
        # intentional for robustness, but callers should know about it.
        if not self._state.version_found and not has_content:
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
        if self._state.other_lines:
            self.las_file.other = "\n".join(self._state.other_lines) + "\n"

        # Process collected ASCII data only for LAS 3.0
        # For LAS 1.2/2.0, data_reader handles ASCII data with proper wrap mode support
        if self.las_file.version.is_las30:
            # I2F-020: Only process data when there are accumulated lines.
            # _flush_ascii_data() (L2470) applies the same guard — its
            # finally-block cleanup (ascii_data_lines = [],
            # current_data_section_idx += 1) only fires after actual
            # processing, never for an empty section.
            if self._state.ascii_data_lines:
                ctx = AsciiDataContext(
                    las_file=self.las_file,
                    ascii_data_lines=self._state.ascii_data_lines,
                    section_curve_start_idx=self._state.section_curve_start_idx,
                    section_curve_end_idx=self._state.section_curve_end_idx,
                    current_section_name=self._state.current_section_name,
                    current_data_section_type=self._state.current_data_section_type,
                    current_data_section_idx=self._state.current_data_section_idx,
                    cumulative_elements=self._state.cumulative_elements,
                )
                try:
                    process_ascii_data(ctx)
                    self._state.cumulative_elements = ctx.cumulative_elements
                finally:
                    # Mirror _flush_ascii_data() L2495-2496: clear
                    # accumulated lines and advance the section counter
                    # so _check_data_section_idx_consistency does not
                    # fire a false-positive "dangling data" warning.
                    self._state.ascii_data_lines = []
                    self._state.current_data_section_idx += 1
            # EXT-02: N-I-31 reconcile must also run when no data lines
            # remain buffered.  A single data section processed before
            # ~Well (e.g. ``~A ... ~WELL ... EOF``) has its null fill
            # cells baked with the default -999.25 sentinel, and
            # process_ascii_data only reconciles at the START of a LATER
            # data-processing call — a trailing well section never
            # triggers it.  This call is a no-op when no fill cells are
            # tracked (process_ascii_data already reconciled, or nothing
            # to reconcile).
            _reconcile_null_sentinels(self.las_file)

        # Validate mandatory well fields (STRT, STOP, STEP, NULL).
        # LAS 1.2 and 2.0 both require these fields; LAS 3.0 inherits the
        # same mandatory well-field requirements from LAS 2.0.  Missing
        # fields are a spec compliance gap.  The library handles missing
        # fields gracefully (using defaults), so this is a warning, not an
        # error.  F-M24: Previous check excluded LAS 1.2 despite the LAS
        # 1.2 spec requiring STRT/STOP/STEP/NULL.
        # F-014: Previous version gate only checked startswith("2."),
        # silently skipping mandatory-field validation for LAS 3.0 files.
        is_las12_or_later = _LASVersionSpec(self.las_file.version.vers).is_las12_or_later
        if is_las12_or_later and self._state.version_found:
            # F-25: Use version-specific mandatory well fields from
            # _LASVersionSpec.  M-11: LAS 2.0 requires the lascheck
            # 10-field set (STRT, STOP, STEP, NULL, COMP, WELL, FLD, LOC,
            # SRVC, DATE); LAS 1.2 and 3.0 require STRT, STOP, STEP, NULL
            # (I2-07 — the lascheck 10-field set is a 2.0-era requirement;
            # lascheck itself documents "supports checking against LAS 2.0
            # standard only", so applying it to 1.2 produced 6 spurious
            # warnings on minimal-but-valid 1.2 files).  UWI is optional
            # (present but not required) — it previously caused
            # false-positive "missing: UWI" warnings on common files.  We
            # warn but do NOT raise an error.
            _version_spec = _LASVersionSpec(self.las_file.version.vers)
            _mandatory_fields = list(_version_spec.mandatory_well_fields)
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
        if self._state.las30_sections_seen and not self.las_file.version.is_las30:
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

        # F-07/F-036/F-037/F-039: Re-run validations skipped during
        # incremental construction.  The parser populates curves_order
        # and curves after __post_init__, so __post_init__ guards never
        # saw the fully-populated state.
        #
        # F-037: We cannot safely re-invoke __post_init__() here
        # because it is not idempotent for real-world files.  The
        # duplicate-curve-name check (models.py:1608-1616) raises
        # LASDataError for legitimate duplicate curves that the parser
        # previously handled silently.  Instead, we replicate only the
        # format-vs-placement check from __post_init__
        # (models.py:1907-1930) — the check that raises for data-integrity
        # issues — then capture validate(complete=True) return value
        # and log all remaining warning-level issues.
        #
        # Format-vs-placement: string-format curves must be in string_data,
        # not logs; numeric-format curves must be in logs, not string_data.
        # P-11: Normalize the LASDataError escape to LASParseError at the
        # parser boundary (LASDataError IS a ValueError subclass, so
        # ``except ValueError`` catches it) — read_las_file only wraps
        # LASParseError; an undocumented LASDataError escaping parse()
        # would propagate raw to public-API callers.
        if self.las_file.curves and (self.las_file.logs or self.las_file.string_data):
            _log_keys = set(self.las_file.logs.keys()) if self.las_file.logs else set()
            _str_keys = (
                set(self.las_file.string_data.keys()) if self.las_file.string_data else set()
            )
            for _sc in self.las_file.curves:
                _df = _sc.data_format
                _mnem = _sc.mnemonic
                if not _df:
                    continue
                try:
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
                except ValueError as _exc:
                    raise LASParseError(f"Line {self._line_no}: {_exc}") from _exc
        # F-036/F-039: Capture validate(complete=True) return value and
        # log each issue.  All remaining issues (STEP=0, NULL empty,
        # STRT==STOP, DLM-on-LAS-1.2, cross-section consistency, etc.)
        # are spec-compliance warnings — not data-integrity errors.
        _complete_issues = self.las_file.validate(complete=True)
        for _issue in _complete_issues:
            logger.warning("LASFile validation issue: %s", _issue)

        # Cross-validate _ParserState against las_file.
        _state_issues = self._state.validate(self.las_file)
        for issue in _state_issues:
            logger.warning("Parser state inconsistency: %s", issue)

        # EXT-03: Dedupe top-level curves for LAS 3.0 files with data
        # sections.  The N-I-20 writer emits per-section _Definition
        # sections that re-declare curves already in the main ~C block
        # (e.g. ~Log_Definition re-declares DEPT/GR); _parse_curve appends
        # every Definition curve to the global curves/curves_order with no
        # cross-section dedup, inflating the top-level model (3 → 7) on
        # re-read.  Collapse to unique mnemonics (first-occurrence
        # definitions).  Per-section scoping is unaffected: DataSection
        # objects hold their own section_curves slices, and this runs
        # AFTER _state.validate() so definition_curve_ranges are checked
        # against the untrimmed curve list.
        if self.las_file.version.is_las30 and self.las_file.data_sections:
            _dedup_order: list[str] = []
            _dedup_curves: list[CurveDefinition] = []
            _seen_mnems: set[str] = set()
            for _i, _name in enumerate(self.las_file.curves_order):
                if _name in _seen_mnems:
                    continue
                _seen_mnems.add(_name)
                _dedup_order.append(_name)
                if _i < len(self.las_file.curves):
                    _dedup_curves.append(self.las_file.curves[_i])
            if len(_dedup_order) != len(self.las_file.curves_order):
                self.las_file.curves_order = _dedup_order
                # F-29: LASFile.__setattr__ re-wraps logs/string_data/
                # curves_order but NOT curves (models.py:3006-3057), so a
                # plain-list assignment here permanently strips the
                # _GuardedList mutation guard installed by __post_init__
                # (models.py:3506-3510).  Post-parse
                # las_file.curves.append("bad") would silently succeed and
                # the writer would fail later with a confusing AttributeError.
                # Re-wrap through _GuardedList (mirroring _WriterMutationGuard.
                # _rewrap_guards) so the mutation guard survives the dedup.
                self.las_file.curves = _GuardedList(
                    _dedup_curves,
                    _container_name="LASFile.curves",
                    _expected_type=CurveDefinition,
                )
        elif not self.las_file.version.is_las30:
            # F-30: Dedupe duplicate curve mnemonics during DIRECT parse for
            # LAS 1.2/2.0.  _parse_curve appends every ~C line with no
            # duplicate detection; the full reader path masks this because
            # read_ascii_data() calls data_reader._deduplicate_curves()
            # (rename-based: IK → IK_2) before reading data.  The semi-public
            # LASParser.parse() API therefore produced a model with duplicate
            # curves_order entries — validate(complete=True) reported 0 issues
            # (its order checks don't detect duplicates) yet to_dict()→
            # from_dict() raised LASDataError("Duplicate curve name 'IK'").
            # Call the SAME rename-based dedup so the direct-parse model is
            # from_dict-compatible and matches the full-reader model exactly.
            # Idempotent: the reader's later call is a no-op when curves are
            # already unique (data_reader.py:416-417).  Runs AFTER
            # _state.validate() like the EXT-03 block above, so parser-state
            # checks see the untrimmed curve list.  (LAS 3.0 top-level curves
            # are handled by EXT-03's collapse semantics; the reader never
            # renames LAS 3.0 top-level curves either — reader.py only calls
            # read_ascii_data for non-3.0.)
            _data_reader._deduplicate_curves(self.las_file, _stacklevel=3)

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

        FIX-CONV-1 (F-01): A standalone mnemonic header row directly
        below ~A (``~A\\nDEPT GR\\n1000.0 50.0\\n...``) is a column
        header, not a data row — data_reader._read_normal skips it via
        _is_mnemonic_header_row.  Counting it here inflates
        data_line_count by 1, triggering the F36 trim and a spurious
        "Pre-scan overcount" warning (and previously crashed via the
        M-43 length guard).  Curve mnemonics are parsed in Pass 2,
        AFTER this pre-scan, so the ~C section is scanned here for the
        declared mnemonics (mirroring _normalize_curve_mnemonic's
        mnem_base collision handling so the match set equals the
        reader's curves_order set).
        """
        in_ascii = False
        count = 0
        per_block_counts: list[int] = []
        in_curve = False
        curve_mnems: set[str] = set()
        # F-24 (pre-fix audit A1/A2): dedicated distinct-curve counter —
        # one per ~C curve-definition line, mirroring the reader's
        # curve_count (len(las_file.curves_order), data_reader.py:897/912).
        # NOT len(curve_mnems): that set includes original_mnemonic aliases
        # (FIX-CONV-2 adds both _resolved and _raw at :1316/:1322), so it
        # over-counts on mnem_base files (LLD→BFV) and a count guard built
        # on it would re-introduce the F-24 phantom row.
        curve_def_count = 0
        # F-24 (pre-fix audit A4): string-curve counter — ~C lines declaring
        # {S} or bare {A} (string data), mirroring the reader's all-string
        # exclusion (len(string_curve_indices) == curve_count, :894-895).
        # Version-independent per M-35: {S}/{A} classify string curves on
        # both LAS 1.2/2.0 and 3.0 via _detect_string_curves.
        string_curve_count = 0
        # FIX-CONV-2 (§4a): dict of first-seen raw mnemonic per RESOLVED
        # name — exact parity with _normalize_curve_mnemonic's
        # _resolved_curve_names (dict-based, not set-based).  The previous
        # set-based check (`_resolved in curve_mnems`) approximated the
        # collision semantics but MISSED the first normalized curve's
        # original_mnemonic: for LLD→BFV the reader's match set contains
        # both "BFV" and "LLD" (curves[1].original_mnemonic="LLD"), while
        # the pre-scan only had "BFV".  Without the mirror, a raw-vendor
        # header row ("DEPT LLD LLS") is counted by the pre-scan but
        # skipped by the reader → spurious overcount warning + trim on
        # every mnem_base header file (regressing F-01's zero-warning
        # outcome).
        _pre_scan_resolved_raws: dict[str, str] = {}

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
                # P-05: Handle no-space pipe section headers (e.g.
                # "~ASCII|CURVE") the same way _parse_line does — strip
                # the pipe target so the known-set check and the
                # is_ascii determination see the bare section word.
                # Without this, data lines under such a header were not
                # counted, producing a stale data_line_count and diverging
                # from the data reader's section detection.
                if "|" in section_word:
                    _pipe_idx = section_word.find("|")
                    section_word = section_word[:_pipe_idx].strip()
                # Skip unrecognized section-like patterns so that control-
                # character noise does not break the ASCII-block count
                # (matching _read_normal's _is_recognized_section_word
                # behavior).  Recognized types include standard section
                # words and suffix-based types (_DEFINITION, _PARAMETER,
                # _PARAMETERS, _DATA).
                _base = section_word.split("[", 1)[0] if "[" in section_word else section_word
                _recognized = True
                if _base not in {
                    "A",
                    "ASCII",
                    "V",
                    "VERSION",
                    "W",
                    "WELL",
                    "C",
                    "CURVE",
                    "P",
                    "PARAMETER",
                    "PARAMETERS",
                    "O",
                    "OTHER",
                    # F-005: Section words recognized by
                    # data_reader._KNOWN_SECTION_WORDS but
                    # missing from _pre_scan — divergence
                    # causes stale line-count estimates.
                    "D",
                    "DEFINITION",
                    "CORE",
                    "DRILLING",
                    "FORMATION",
                    "INCLINOMETRY",
                    "LOG",
                    "MUD",
                    "PERFORATIONS",
                    "RISK",
                    "STRUCTURE",
                    "TEST",
                    "TOPS",
                }:
                    if re.search(r"_DEFINITION(_\d+)?$", _base):
                        pass
                    elif (
                        _base.endswith("_PARAMETER")
                        or _base.endswith("_PARAMETERS")
                        or _base.endswith("_DATA")
                    ):
                        pass
                    else:
                        # P-16: Unrecognized genuine section word (e.g.
                        # ~CUSTOMSECT) is a section boundary — _parse_line
                        # routes it to other_lines and the data reader
                        # (_iter_ascii_data_lines) stops reading at it.
                        # Treat it like any other non-~A header: flush the
                        # current ~A block and leave ASCII mode so its body
                        # is NOT counted as data.  Control-character noise
                        # never reaches here — SECTION_PATTERN requires a
                        # letter after ~.
                        _recognized = False
                is_ascii = _recognized and section_word in {"A", "ASCII"}
                is_curve = _recognized and section_word in {"C", "CURVE"}
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
                in_curve = is_curve
                continue
            if in_curve:
                # FIX-CONV-1 (F-01): collect the mnemonic from each
                # curve-definition line in the ~C section.  The mnemonic is
                # the first field, delimited by whitespace or comma (both
                # layouts occur in the wild); LAS 3.0 trailing-dot
                # mnemonics ("DEPT.") and array suffixes ("NMR[1]") are
                # cleaned.  mnem_base resolution mirrors
                # _normalize_curve_mnemonic: first-seen resolved name wins;
                # a colliding alias (N-I-30) keeps its raw mnemonic so the
                # match set equals the reader's curves_order set.
                if (
                    not COMMENT_PATTERN.match(stripped)
                    and not EMPTY_PATTERN.match(stripped)
                    and not stripped.startswith("~")
                ):
                    _curve_tokens = re.split(r"[\s,]+", stripped, maxsplit=2)
                    if _curve_tokens and _curve_tokens[0]:
                        # First field is the mnemonic; strip the unit dot
                        # ("DEPT.M" → "DEPT"), LAS 3.0 trailing dot
                        # ("DEPT."), and array suffix ("NMR[1]").
                        _raw = _curve_tokens[0].split(".", 1)[0].split("[", 1)[0].strip().upper()
                        if _raw:
                            # M9 (F-24 regression): the parser DROPS some
                            # ~C lines — multi-word no-period mnemonics
                            # ("HOLE DIA : hole diameter") fail
                            # _MNEMONIC_LINE_RE in _parse_curve.  Counting
                            # a dropped line inflates curve_def_count, so
                            # the F-24 count-equality header-skip predicate
                            # below fails to skip a genuine full header row
                            # → spurious "Pre-scan overcount" warning + F36
                            # trim on valid files.  Mirror _parse_curve's
                            # acceptance exactly: count only lines that
                            # survive parsing.
                            if not self._is_curve_line_accepted(stripped):
                                continue
                            # F-24: one distinct curve per definition line.
                            curve_def_count += 1
                            # F-24 (pre-fix audit A4): detect string curves
                            # ({S} / bare {A}) for the all-string exclusion
                            # via the same description-format extraction the
                            # parser uses (DATA_LINE_PATTERN + FORMAT_SPEC_PATTERN).
                            _desc_match = DATA_LINE_PATTERN.match(stripped)
                            if _desc_match:
                                _desc_text = _desc_match.group("description") or ""
                                for _fmt, _off in FORMAT_SPEC_PATTERN.findall(_desc_text):
                                    if _fmt.upper() == "S" or (_fmt.upper() == "A" and not _off):
                                        string_curve_count += 1
                                        break
                            _resolved = self._mnem_base_upper.get(_raw, _raw)
                            _prev_raw = _pre_scan_resolved_raws.get(_resolved)
                            if _prev_raw is not None and _prev_raw != _raw:
                                # N-I-30 collision: the resolved name is
                                # already claimed by a different raw
                                # mnemonic → this curve keeps its raw
                                # mnemonic (matches _normalize_curve_mnemonic).
                                curve_mnems.add(_raw)
                            else:
                                _pre_scan_resolved_raws[_resolved] = _raw
                                curve_mnems.add(_resolved)
                                if _resolved != _raw:
                                    # First-normalized curve keeps the
                                    # original_mnemonic (parser.py:2635);
                                    # mirror it so the match set equals the
                                    # reader's declared set.
                                    curve_mnems.add(_raw)
                continue
            if (
                in_ascii
                and not COMMENT_PATTERN.match(stripped)
                and not EMPTY_PATTERN.match(stripped)
            ):
                # P-16: ~-prefixed lines that are NOT section headers (e.g.
                # ~3D, ~., ~#) are not data rows — the parser routes them
                # to other_lines and the data reader skips them.  Exclude
                # them from the ASCII-block count so data_line_count matches
                # the reader's consumption.
                if not stripped.startswith("~"):
                    # FIX-CONV-1 (F-01): skip a standalone mnemonic header
                    # row (every token is a declared curve mnemonic) so it
                    # is NOT counted as a data line — mirroring
                    # data_reader._is_mnemonic_header_row.
                    # F-24 (pre-fix audit) + DR-M1 (Stage 10, M12 residual):
                    # mirror the reader's clauses VERBATIM so the two
                    # predicates can never disagree:
                    #   (a) token-count BOUNDS (reader :929) — since M12 the
                    #       reader skips a PARTIAL mnemonic header ("DEPT GR"
                    #       with 3 declared curves, 2..curve_count tokens).
                    #       The pre-scan mirror must use the same
                    #       min(2, curve_def_count)..curve_def_count range;
                    #       strict equality counted the partial header as a
                    #       data line → spurious "Pre-scan overcount" warning
                    #       + F36 trim on valid files (the documented
                    #       verbatim-mirror contract above was factually
                    #       false post-M12).  F-01 (Stage 12, PSR-1
                    #       residual): the lower bound is min(2, ...), NOT a
                    #       flat 2 — a SINGLE-curve section's 1-token
                    #       standalone mnemonic header row ("~A\nDEPT\n...")
                    #       is a header on the reader side (min(2, 1) = 1)
                    #       but a flat >= 2 gate counted it as data here,
                    #       overcounting data_line_count by one → spurious
                    #       "Pre-scan overcount" warning on every valid
                    #       single-curve LAS 1.2/2.0 read.
                    #   (b) first-line-of-section restriction (DR-M3: the
                    #       reader gates its header-skip at current_line == 0).
                    #       count == 0 here is the per-block analog of the
                    #       reader's current_line == 0 (both increment only
                    #       for processed data rows), so a mid-section
                    #       all-mnemonic row is DATA on both sides — skipping
                    #       it would silently lose data / shift columns.
                    #   (c) all-string exclusion (reader :931-932) — rows
                    #       in an all-string section are never headers.
                    #       Count uses the dedicated distinct-curve counter
                    #       (curve_def_count), never len(curve_mnems) —
                    #       aliases would over-count on mnem_base files.
                    if curve_mnems:
                        _tokens = re.split(r"[\s,]+", stripped)
                        while _tokens and _tokens[-1] == "":
                            _tokens.pop()
                        if (
                            count == 0
                            and _tokens
                            and len(_tokens) >= min(2, curve_def_count)
                            and len(_tokens) <= curve_def_count
                            and string_curve_count != curve_def_count
                            and all(_tok.upper() in curve_mnems for _tok in _tokens)
                        ):
                            continue
                    count += 1

        # F-I2-M10: Always append final block count — even zero.
        if in_ascii:
            per_block_counts.append(count)

        # Use the count from the first contiguous ~A block, matching
        # _read_normal's behavior (processes until first non-~A header).
        self._state.data_line_count = per_block_counts[0] if per_block_counts else 0

    def _is_curve_line_accepted(self, line: str) -> bool:
        """M9: whether ``_parse_curve`` would accept this ``~C`` line.

        Mirrors the F-23 (a) no-period fallback at ``_parse_curve``'s entry
        exactly: a dot-anchored match is accepted; a no-period match is
        accepted only when its mnemonic can be represented by
        ``CurveDefinition`` (``_MNEMONIC_LINE_RE`` — multi-word mnemonics
        like "HOLE DIA" are dropped with a warning).  The pre-scan uses
        this so its ``curve_def_count`` equals the number of curves the
        parser actually produces — counting dropped lines inflated the
        count and broke the F-24 header-skip parity (spurious
        "Pre-scan overcount" warning + F36 trim on valid files).
        """
        match = self._match_data_line(line)
        if match:
            return True
        match = self._match_data_line_no_period(line)
        if match and not _MNEMONIC_LINE_RE.fullmatch(match.group("mnemonic")):
            return False
        return match is not None

    def _parse_line(self, line: str) -> None:
        """Route a single line to the appropriate section handler."""
        # F-I2-M06: Guard against absurdly long lines before any regex
        # processing.  max_file_size bounds total file but crafted files
        # can place a 500MB payload in a single line, causing unbounded
        # regex backtracking and string allocation.
        if len(line) > MAX_LINE_LENGTH:
            raise LASParseError(
                f"Line {self._line_no}: line length ({len(line)}) exceeds "
                f"maximum allowed ({MAX_LINE_LENGTH}). "
                f"The file may be malformed or corrupt."
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

            # Capture previous section's state BEFORE classification runs.
            # The handler snapshots all transition-relevant state and saves
            # C curve ranges to _definition_curve_ranges so pipe-target
            # lookups during classification can find them (H-03/H-01).
            captured = self._transition_handler.capture_current_state()

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
                    # M-67: Do NOT freeze when len(curves) == 0 — a _Definition
                    # that PRECEDES the main ~C block would capture 0 and a
                    # later "~A | CURVE" would scope to curves[0:0] (empty
                    # slice) silently discarding the entire LOG_DATA section.
                    # Keep -1 so _resolve_main_curve_scope() falls through to
                    # the recorded __MAIN__ range / all-curves.
                    if self._state.main_curve_end == -1 and len(self.las_file.curves) > 0:
                        self._state.main_curve_end = len(self.las_file.curves)
                    # G-02: Track which _Definition is active so curve ranges
                    # can be saved per-definition type (prevents overwrite
                    # by consecutive _Definition sections).
                    self._state.current_definition_name = section_word.upper()
                else:
                    section_name = section_rest or ""
                    # G-02: Regular ~C or ~CURVE section — no definition name.
                    self._state.current_definition_name = None
            elif (
                section_word in {"P", "PARAMETER", "PARAMETERS"}
                or section_word.endswith("_PARAMETER")
                or section_word.endswith("_PARAMETERS")
            ):
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
                        f"Line {self._line_no}: ~Other section "
                        f"('{_other_line}') — ~Other is NOT ALLOWED in "
                        f"LAS 3.0.  Migrate content to user-defined "
                        f"Parameter or Column Data sections."
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
                    self._state.las30_sections_seen = True
                # For standard 'A' or 'ASCII' sections and written-form *_DATA
                # sections (e.g., ~DRILLING_DATA DRILLING), use only the rest
                # as the section name (backward-compatible).
                if section_word in {"A", "ASCII"} or section_word.endswith("_DATA"):
                    section_name = section_rest
                else:
                    section_name = (
                        f"{section_word} {section_rest}".strip() if section_rest else section_word
                    )
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
                    self._state.current_data_section_type = stype
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
                    self._state.current_data_section_type = stype
                # F-08: Handle pipe-delimited definition association.
                # "| CURVE" means use the main curve block (before
                # _Definition sections). "| X_Definition" means use
                # the per-section curves from that definition block.
                # M-69: Record the pipe target of the current data
                # section so pre-~V deferred data lines can re-resolve a
                # "| CURVE" scope at replay time (main_curve_end is not
                # yet known at defer time).  Reset per section below.
                self._current_pipe_target = None
                if pipe_target:
                    pipe_target_upper = pipe_target.upper()
                    self._current_pipe_target = pipe_target_upper
                    if pipe_target_upper in {"CURVE", "C"}:
                        # Pipe "| CURVE" → use main curve block only.
                        # M-67: Resolve against the recorded plain-~C range
                        # (__MAIN__) when available — main_curve_end is
                        # frozen at the FIRST _Definition and captures 0 when
                        # a _Definition precedes the main ~C block, silently
                        # discarding LOG_DATA scoped to curves[0:0].
                        main_start, main_end = self._resolve_main_curve_scope()
                        self._state.section_curve_start_idx = main_start
                        self._state.section_curve_end_idx = main_end
                    elif pipe_target_upper in self._state.definition_curve_ranges:
                        # G-02: Explicit pipe to a known _Definition —
                        # look up the saved (start, end) range.
                        start, end = self._state.definition_curve_ranges[pipe_target_upper]
                        self._state.section_curve_start_idx = start
                        self._state.section_curve_end_idx = end
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
                        self._state.section_curve_start_idx = 0
                        self._state.section_curve_end_idx = (
                            None  # F-051: None → all curves (0 → empty slice)
                        )
                else:
                    # G-02/H-01: No pipe — try to match the data section
                    # type to its _Definition (e.g., CORE_DATA →
                    # CORE_DEFINITION, LOG_DATA → LOG_DEFINITION).
                    # H-01: This branch previously excluded LOG_DATA
                    # (``elif current_data_section_type != "LOG_DATA"``),
                    # so a bare ~A/~LOG_DATA section never consulted the
                    # saved LOG_DEFINITION range.  In _Definition-only
                    # files (no bare ~C, no __MAIN__ sentinel) LOG_DATA
                    # fell to the else-reset (start=0, end=None → ALL
                    # curves), and the LAS 3.0 consumer sliced every
                    # curve — mapping data columns positionally into
                    # wrong curve names.  Resolve LOG_DEFINITION the same
                    # way every other type resolves its _DEFINITION.
                    def_prefix = (
                        self._state.current_data_section_type.replace("_DATA", "") + "_DEFINITION"
                    )
                    if def_prefix in self._state.definition_curve_ranges:
                        start, end = self._state.definition_curve_ranges[def_prefix]
                        self._state.section_curve_start_idx = start
                        self._state.section_curve_end_idx = end
                    elif "__MAIN__" in self._state.definition_curve_ranges:
                        # H-01: No matching _Definition found — fall back
                        # to the main non-_Definition curve range.
                        start, end = self._state.definition_curve_ranges["__MAIN__"]
                        self._state.section_curve_start_idx = start
                        self._state.section_curve_end_idx = end
                    else:
                        # N-I-04: Typed data section with neither a
                        # matching _DEFINITION nor a __MAIN__ fallback —
                        # reset to ALL curves.  The previous code had no
                        # else-branch here, so the section inherited the
                        # PREVIOUS section's curve scope and its data was
                        # silently stored under the wrong curves (e.g.
                        # CORE data labeled with the MUD section's
                        # curves).  Mirror the LOG_DATA else-branch below
                        # (start=0, end=None → all).
                        self._state.section_curve_start_idx = 0
                        self._state.section_curve_end_idx = None
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
                if self._state.current_section == "A" and self._state.ascii_data_lines:
                    # F-30: Use shared _flush_ascii_data helper to avoid
                    # duplicating the AsciiDataContext construction and
                    # finally-cleanup logic with _process_ascii_section.
                    self._flush_ascii_data(
                        data_lines=self._state.ascii_data_lines,
                        section_curve_start_idx=self._state.section_curve_start_idx,
                        section_curve_end_idx=self._state.section_curve_end_idx,
                        current_section_name=self._state.current_section_name,
                        current_data_section_type=self._state.current_data_section_type,
                        current_data_section_idx=self._state.current_data_section_idx,
                        cumulative_elements=self._state.cumulative_elements,
                        version_found=self._state.version_found,
                    )
                # F-02: Reset current section so data lines in unknown sections
                # aren't misrouted to the previous section's handler.
                self._state.current_section = None
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

            # Process the previous section and enter the new section.
            # The handler enforces capture→process→enter ordering at the type
            # level: process_previous_section REQUIRES a _CapturedState.
            self._state.cumulative_elements = self._transition_handler.process_previous_section(
                captured, new_section
            )

            # F-34: Compute section label for tracking.
            section_label = (
                f"{section_word}:{section_name}" if section_name.strip() else section_word
            )

            # Enter the new section — set parser state for incoming section.
            # Unknown sections (new_section is None) return early above.
            assert new_section is not None
            self._transition_handler.enter_new_section(
                new_section, section_label, section_word, section_name
            )
            # PARS-02: Record this section's pipe target parallel to
            # _state.section_sequence (appended inside enter_new_section),
            # so the cross-section consistency checker can resolve
            # "| CURVE"/"| C" to __MAIN__.  getattr guard: tests that
            # monkeypatch _reset() (R-005 format-vs-placement path) do not
            # initialize this attribute; a missing list is equivalent to
            # "no pipes recorded".
            _pipe_targets = getattr(self, "_section_pipe_targets", None)
            if _pipe_targets is not None:
                _pipe_targets.append(pipe_target.upper() if pipe_target else None)

            # F-040: Pre-~V data sections need distinct section indices.
            # When bare ~A sections appear before ~V is parsed, each ~A
            # section should receive a unique current_data_section_idx so
            # that _replay_deferred_well can group deferred lines by
            # (section_type, section_name, section_idx) without merging
            # distinct sections that happen to share the same type and name.
            # P-03: Widened from A→A-only to EVERY pre-~V data-section
            # entry — a non-A section between two ~A sections (A→W→A) made
            # the W→A transition fail to increment, so the second section
            # collided with the first and groupby merged them into one
            # DataSection (silent data corruption).  The deferred-lines
            # guard keeps the FIRST pre-~V section at idx 0 while ensuring
            # every subsequent pre-~V data section gets a fresh idx (no
            # double-count, MAX_DATA_SECTIONS not bypassed).
            # Post-~V sections are handled by _flush_ascii_data's finally
            # block; this only applies to pre-~V transitions.
            if (
                new_section == "A"
                and not captured.version_found
                and self._state.deferred_ascii_data_lines
            ):
                self._state.current_data_section_idx += 1

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

        if self._state.current_section:
            handler_name = self.SECTION_HANDLERS.get(self._state.current_section)
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
        if len(self._state.other_lines) >= MAX_OTHER_LINES:
            raise LASParseError(
                f"Line {self._line_no}: other section line count "
                f"({len(self._state.other_lines) + 1}) exceeds "
                f"maximum allowed ({MAX_OTHER_LINES}). "
                f"The file may be malformed or corrupt."
            )
        self._state.other_lines.append(line)

    @staticmethod
    def _manual_colon_scan(line: str) -> dict[str, str] | None:
        """Manual scan for colon separator on long lines (I2F-01 defense).

        Reproduces the delimiter-colon semantics of DATA_LINE_PATTERN
        (P-04/PARS-11) with a value-start-relative scan:

          1. Parse the prefix ([ws] MNEMONIC [ws] . UNIT) with the same
             grammar as the primary pattern.
          2. Consume the unit→value whitespace run greedily to the
             value-start position — mirroring the primary pattern's
             possessive ``\\s++``, the separator is never positioned
             inside that run (I2-11).
          3. Scan forward from value-start for the FIRST colon matching
             any separator alternative, in pattern order:
               (a) ``\\s+:\\s*``  — colon at the end of a whitespace run
               (b) ``:(?=\\s)``   — colon followed by whitespace
               (c) ``(?<=\\s):(?!.*:(?=\\s|$))\\s*+`` — colon at
                   value-start whose preceding whitespace was consumed by
                   the unit→value run, with no LATER valid colon
               (d) ``:\\s*$``     — colon at end of line

        This is O(n), single pass, with no backtracking — guaranteed safe
        regardless of input.  Returns a dict with 'mnemonic', 'unit',
        'value', 'description' keys, or None if the line doesn't match
        the data-line pattern.

        Intended as a fallback for lines too long for safe regex matching
        (>_SAFE_REGEX_LINE_LENGTH).
        """
        stripped = line.rstrip()
        m = _MANUAL_PREFIX_RE.match(stripped)
        if not m:
            return None
        # value_start: position right after the unit→value whitespace run.
        value_start = m.end()
        while value_start < len(stripped) and stripped[value_start].isspace():
            value_start += 1

        # Scan forward from value-start for the first separator colon.
        colon_idx = -1
        i = value_start
        while i < len(stripped):
            if stripped[i] != ":":
                i += 1
                continue
            # (b) :(?=\s) — colon followed by whitespace.
            if i + 1 < len(stripped) and stripped[i + 1].isspace():
                colon_idx = i
                break
            # (d) :\s*$ — colon at end of line (rstrip removed trailing ws,
            # so the colon IS the last char).
            if i + 1 >= len(stripped):
                colon_idx = i
                break
            # (a) \s+:\s* — colon terminates a whitespace run that the lazy
            # value group reached (run starts at/after value-start).
            if i > 0 and stripped[i - 1].isspace() and i - 1 >= value_start:
                colon_idx = i
                break
            # (c) (?<=\s):(?!.*:(?=\s|$))\s*+ — colon at value-start whose
            # preceding whitespace was consumed by the unit→value run, with
            # no LATER colon followed by whitespace or at end of line.
            if i == value_start and i > 0 and stripped[i - 1].isspace():
                later_valid = False
                for j in range(i + 1, len(stripped)):
                    if stripped[j] == ":" and (j + 1 >= len(stripped) or stripped[j + 1].isspace()):
                        later_valid = True
                        break
                if not later_valid:
                    colon_idx = i
                    break
            i += 1
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
        # I2-10 (pre-fix audit F-06a): parse the unit/value from the
        # NON-rstripped slice up to the colon.  M-70's unit regex
        # ``([\w\-/.%°]*)(\s+)`` needs the separating whitespace to remain
        # visible; ``prefix.rstrip()`` removed it on empty-value lines
        # (``DEPT.FT : <2100 D's>`` → after_dot='FT', ``\s+`` fails, the
        # UNIT text became the VALUE).  The value is already ``.strip()``-ed
        # below, so leading/trailing whitespace in the slice is safe.
        after_dot = stripped[dot_idx + 1 : colon_idx]

        # Unit is the contiguous unit chars before the first value
        # whitespace.  Value is everything after that whitespace.
        # M-70: Match the unit EXACTLY like the primary DATA_LINE_PATTERN
        # (unit class `[\w\-/.%°:]*` followed by `\s+`) against the
        # UN-STRIPPED tail.  The primary regex sees the space after the dot
        # ("WELL. WELLNAME : d" → unit matches empty, `\s+` eats the space,
        # value = "WELLNAME").  The old letter-start requirement
        # `[a-zA-Z%°][\w\-/.%°]*` ran on the STRIPPED tail and consumed
        # alphabetic values (well names, api codes) as the unit, emptying
        # the value on >2000-char lines.  PD1-01's numeric guard is
        # preserved: a purely numeric tail ("123.45") leaves no whitespace
        # after the unit class, so `\s+` fails and the value falls through.
        unit_match_result = re.match(r"([\w\-/.%°:]*)(\s+)", after_dot)
        if unit_match_result:
            unit = unit_match_result.group(1)
            value = after_dot[unit_match_result.end() :].strip()
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

    def _match_data_line_no_period(self, line: str) -> re.Match[str] | None:
        """F-23 (a): lasio missing-period fallback for header lines.

        Matches ``MNEM : VALUE`` (no dot) when the dot-anchored patterns
        failed.  lasio's documented missing-period convention takes
        everything left of the first colon as the mnemonic (spaces
        preserved, e.g. ``HOLE DIA``) and everything right as the value,
        with empty unit and description.

        Only the WELL path uses this unconditionally — ``WellSection``
        accepts multi-word keys via ``__setitem__``.  The CURVE and
        PARAMETER paths call this and then drop the match when the
        resulting mnemonic cannot be represented by their models (which
        reject embedded spaces); dropping preserves the pre-fix behavior
        instead of raising LASParseError for the whole file.
        """
        if len(line) > _SAFE_REGEX_LINE_LENGTH:
            return None
        match = NO_PERIOD_PATTERN.match(line)
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
        # N-I-22: widened unit class (same as DATA_LINE_PATTERN).
        # M-60: `(?P<value>.+?)\s*$` was quadratic — for each `.+?` expansion
        # inside a long whitespace run, `\s*$` fails then backtracks char-by-char
        # through the run → O(m·k).  Match on `line.rstrip()` (the line has no
        # trailing whitespace, so `$` anchors immediately) and use greedy
        # `.+$` (consumes to end in one pass) → O(n).  Values are `.strip()`-ed
        # by all callers, so trailing whitespace never matters.  Degenerate edge:
        # a trailing-whitespace-only line (`'D.M   '`) no longer matches (no
        # value) → the caller emits the "Non-matching ~C line" warning instead
        # of creating a curve with an empty api_code (audit-flagged, acceptable).
        m = re.match(
            r"^\s*(?P<mnemonic>[\w\-]+(?:\[\d+\])?)\s*\.(?P<unit>[\w\-/.%°:]*)\s+(?P<value>.+)$",
            line.rstrip(),
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
                    f"Line {self._line_no}: field '{group_name}' length "
                    f"({len(val)}) exceeds maximum allowed "
                    f"({MAX_FIELD_LENGTH}). "
                    f"The file may be malformed or corrupt."
                )

    def _resolve_main_curve_scope(self) -> tuple[int, int | None]:
        """Resolve the main (non-``_Definition``) curve block for a ``| CURVE`` pipe.

        Returns a ``(start, end)`` pair where ``end`` may be ``None``
        (all curves from ``start``).

        M-67/M-69: ``main_curve_end`` is frozen ONCE at the first
        ``_Definition`` section; when a ``_Definition`` PRECEDES the plain
        ``~C`` block the freeze captures 0 and a later ``~A | CURVE`` scopes
        to ``curves[0:0]`` (empty) — silently discarding the entire data
        section.  The recorded plain-``~C`` range (``definition_curve_ranges
        ["__MAIN__"]``, captured when the plain ``~C`` block is LEFT) is
        authoritative whenever it exists: it is correct regardless of section
        order.  Fall back to the ``main_curve_end`` freeze (files that defer
        data before any ``~C`` have no ``__MAIN__`` yet), then to an
        unbounded scope (data-before-curves files).
        """
        # PARS-09: Prefer the UNION of repeated plain-~C blocks
        # (__MAIN_ALL__) over the last-writer-wins __MAIN__.  A "| CURVE"
        # pipe must see every plain-~C curve (``~C(DEPT,GR) ~C(RHOB)
        # ~A|CURVE`` → DEPT,GR,RHOB), not just the last block's range.
        # __MAIN_ALL__ is accumulated in _section_transition.py; __MAIN__
        # stays last-writer-wins for the bare-~A fallback (F-S9-02).
        main_range = self._state.definition_curve_ranges.get("__MAIN_ALL__")
        if main_range is not None:
            return main_range
        main_range = self._state.definition_curve_ranges.get("__MAIN__")
        if main_range is not None:
            return main_range
        if self._state.main_curve_end >= 0:
            return (0, self._state.main_curve_end)
        return (0, None)

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
            vers_normalized = re.sub(r"^(\d+\.\d+)\.\d+$", r"\1", vers_normalized)
            # M-39: VERS re-entry guard.  A second ~V section declaring a
            # DIFFERENT version leaves the model in a half-upgraded or
            # half-downgraded state: a 3.0→2.0 change retains the
            # already-built LAS 3.0 data_sections on a non-3.0 model
            # (breaking to_dict→from_dict roundtrip — models.py F-41
            # raises "data_sections requires LAS 3.0 version"), while a
            # 2.0→3.0 change silently drops the pre-change ~A data block
            # (its lines were already discarded by _parse_ascii_data).
            # The LAS spec makes ~V a single-occurrence section, so a
            # version CONFLICT is a malformed file: keep the FIRST
            # declared version (the one the data was parsed under) and
            # warn, so the returned model is always internally consistent.
            if self._state.version_found and vers_normalized != self.las_file.version.vers:
                warnings.warn(
                    f"VERS re-declared as '{value}' after version "
                    f"'{self.las_file.version.vers}' was already "
                    f"established by an earlier ~VERSION section.  "
                    f"Keeping the first declared version; the conflicting "
                    f"re-declaration is ignored to avoid leaving the "
                    f"model in an inconsistent state.",
                    UserWarning,
                    stacklevel=2,
                )
                return
            # M-05: Only set _version_found for VERS, not other ~V data.
            self._state.version_found = True

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
                    f"Unknown VERS value '{value}'. Expected 1.2, 2.0, or 3.0. Defaulting to 2.0.",
                    UserWarning,
                    stacklevel=2,
                )
                self.las_file.version.vers = "2.0"
            # F-020 pt2: Deferred DLM re-check.  If DLM was parsed BEFORE
            # VERS in non-standard ~V ordering, the DLM version guard at
            # L1501 uses the default "2.0" and incorrectly allows
            # non-SPACE DLM.  Now that the true version is known, re-validate.
            if (
                _LASVersionSpec(self.las_file.version.vers).is_las12
                and self.las_file.version.dlm != "SPACE"
            ):
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
                # F-051: WRAP is a LAS 1.2/2.0 concept — it is not valid
                # in LAS 3.0 where data is handled in structured sections.
                # Non-wrapped data (full rows) works fine; the downstream
                # check in _las30_data.py detects actual wrap mode and
                # raises for genuinely wrapped data.  This warning covers
                # the gap where metadata-only LAS 3.0 files bypass the
                # data-section check entirely.
                # P-02: Do NOT reset WRAP to NO here — the reset made the
                # downstream content-based detection dead code for the
                # standard VERS-before-WRAP ordering, so genuinely wrapped
                # data was silently parsed as non-wrapped (corruption).
                # Keeping the declared value lets _las30_data.py:213-244
                # detect actual wrap mode from the data content.
                if wrap_upper == "YES" and _LASVersionSpec(self.las_file.version.vers).is_las30:
                    warnings.warn(
                        "WRAP=YES is not supported in LAS 3.0. "
                        "LAS 3.0 uses structured data sections "
                        "instead of line wrapping. "
                        "Actual wrap mode is detected from the "
                        "data content; genuinely wrapped data "
                        "will be rejected.",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    f"Unknown WRAP value '{value}'. Expected YES or NO. Defaulting to NO.",
                    UserWarning,
                    stacklevel=2,
                )
                self.las_file.version.wrap = "NO"
        elif mnemonic == "DLM":
            dlm_upper = value.upper()
            if dlm_upper in {"SPACE", "TAB", "COMMA"}:
                if not _LASVersionSpec(self.las_file.version.vers).is_las12 or dlm_upper == "SPACE":
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

    def _normalize_well_mnemonic(self, raw_mnemonic: str) -> str:
        """Collision-aware well mnemonic resolution (PXM-01 parity with
        models.from_dict _norm_well_mnem / M-44).

        Two distinct raw well mnemonics resolving to the same canonical
        (e.g. LLD/LLS → BFV in a dual-laterolog file) previously stored
        last-wins with only a duplicate warning (parser.py:2345-2352) —
        the first value was silently lost while models.from_dict preserved
        both via re-keying.  This mirrors the from_dict contract:

        - the FIRST raw mnemonic to claim a resolved slot keeps the
          resolved (canonical) name;
        - a LATER raw mnemonic resolving to the same canonical keeps its
          ORIGINAL mnemonic (identity preserved) and warns;
        - when the CANONICAL name itself arrives after an alias claimed
          its slot (e.g. LLD then BFV with LLD→BFV), the canonical wins
          its own slot and the earlier alias's stored value is re-keyed
          to the alias's ORIGINAL mnemonic — both values survive
          (raw == resolved branch, mirroring models.py:3795-3805).

        Returns the storage mnemonic (canonical or original).
        """
        resolved = self._mnem_base_upper.get(raw_mnemonic, raw_mnemonic)
        _prev_raw = self._resolved_well_names.get(resolved)
        if _prev_raw is not None and _prev_raw != raw_mnemonic:
            if raw_mnemonic == resolved:
                # The canonical collides with an alias that claimed the
                # slot earlier: move the alias's stored value back under
                # its ORIGINAL mnemonic so the canonical's store below
                # cannot overwrite it.
                for _container in (
                    self.las_file.well.entries,
                    self.las_file.well.units,
                    self.las_file.well.descriptions,
                ):
                    if resolved in _container:
                        _container[_prev_raw] = _container[resolved]
                        del _container[resolved]
                if (raw_mnemonic, resolved) not in self._warned_well_collisions:
                    self._warned_well_collisions.add((raw_mnemonic, resolved))
                    warnings.warn(
                        f"parser: well mnemonic '{raw_mnemonic}' resolves "
                        f"to '{resolved}' via mnem_base, but '{resolved}' "
                        f"is already used by well entry '{_prev_raw}'.  "
                        f"Preserving both: the value for '{_prev_raw}' is "
                        f"re-keyed to its original mnemonic and "
                        f"'{raw_mnemonic}' is stored under '{resolved}'.",
                        UserWarning,
                        stacklevel=2,
                    )
                self._resolved_well_names[_prev_raw] = _prev_raw
                self._resolved_well_names[raw_mnemonic] = raw_mnemonic
                return resolved
            # Warn once per (raw, resolved) pair.
            if (raw_mnemonic, resolved) not in self._warned_well_collisions:
                self._warned_well_collisions.add((raw_mnemonic, resolved))
                warnings.warn(
                    f"parser: well mnemonic '{raw_mnemonic}' resolves to "
                    f"'{resolved}' via mnem_base, but '{resolved}' is "
                    f"already used by well entry '{_prev_raw}'.  Keeping "
                    f"the original mnemonic '{raw_mnemonic}' to preserve "
                    f"well entry identity.",
                    UserWarning,
                    stacklevel=2,
                )
            self._resolved_well_names[raw_mnemonic] = raw_mnemonic
            return raw_mnemonic
        self._resolved_well_names[resolved] = raw_mnemonic
        return resolved

    def _store_well_entry(
        self,
        mnemonic: str,
        unit: str,
        value: str,
        description: str | None,
        is_las12: bool,
        raw_mnemonic: str | None = None,
        no_period: bool = False,
    ) -> None:
        """Store a well entry with version-appropriate value/description handling.

        Extracted from _parse_well to support deferred well processing when
        ~W appears before ~V (the version check is deferred until ~V is parsed).

        *raw_mnemonic* is the file's mnemonic BEFORE mnem_base resolution
        (used for PXM-01 collision-aware re-keying).  When None (older
        callers), the resolved *mnemonic* is used as the raw name, so
        collision detection degrades to plain duplicate detection.

        *no_period* marks a line matched by the F-23 (a) missing-period
        fallback (``MNEM : VALUE``).  In that form the FIRST colon is the
        MNEM:VALUE separator, so the value must not be re-split by the
        LAS 1.2 bare-colon CWLS logic below (L24 — a colon inside the
        value like ``LOC : ACME:OIL`` must stay ``ACME:OIL``).
        """
        # PXM-01: Apply collision-aware well mnemonic resolution BEFORE any
        # storage so entries/units/descriptions all use the same key.  Two
        # distinct raw well mnemonics resolving to the same canonical
        # (e.g. LLD/LLS → BFV) previously last-won here with only a warning
        # (parser.py duplicate check below) — one value was silently lost,
        # while models.from_dict (M-44 _norm_well_mnem) preserved both.
        if raw_mnemonic is None:
            raw_mnemonic = mnemonic
        mnemonic = self._normalize_well_mnemonic(raw_mnemonic)
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
        # L24 (M10 region): a NO-PERIOD match ("LOC : ACME:OIL") already
        # consumed its first colon as the MNEM:VALUE separator — the value is
        # everything right of it and must not be re-split here.
        if is_las12 and description is None and ":" in value and not no_period:
            _is_timestamp = ":" in value and "T" in value and bool(re.search(r"T\d{2}:", value))
            if not _is_timestamp:
                # F-003: Check the full value for bare hh:mm[:ss]
                # timestamp patterns BEFORE the partial post-colon
                # check.  A value like "12:34" has only one colon —
                # the post-colon portion "34" won't match
                # \b\d{1,2}:\d{2}\b, causing the bare-colon split to
                # corrupt the value (stored as "34" instead of "12:34").
                _is_timestamp = bool(re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", value))
            if not _is_timestamp:
                # F-002 / I2F-12: Scope the timestamp regex to the portion
                # after the first colon, but ONLY when the pre-colon portion
                # does not contain alphabetic text.  When the pre-colon part
                # has text (e.g., "LOG DATE:12:34:56"), the colon is a CWLS
                # description:value separator, not part of a timestamp.
                _colon_idx = value.index(":")
                _pre_colon = value[:_colon_idx]
                if not re.search(r"[a-zA-Z]", _pre_colon):
                    _time_match = re.search(
                        r"\b\d{1,2}:\d{2}(:\d{2})?\b",
                        value[_colon_idx + 1 :],
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
                f"Line {self._line_no}: well entry count "
                f"({len(self.las_file.well.entries) + 1}) exceeds "
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
                        # P-01: The value is the PRE-colon text in this
                        # layout — previously the code stored post-colon
                        # text as the value, silently inverting
                        # value/description for lasio-style files
                        # (e.g. "DATE. 15/01/2001 : LOG DATE" → value
                        # must be "15/01/2001", not "LOG DATE").
                        actual_value = value
                        self.las_file.well.descriptions[mnemonic] = description
                    elif value_has_spaces and not desc_has_spaces:
                        # Multi-word pre-colon, single-word post-colon
                        # = likely CWLS description before colon (VALUE after).
                        # Swap to match explicit CWLS non-mandatory branch
                        # (lines 918-919): post-colon text = VALUE,
                        # pre-colon text = DESCRIPTION.
                        actual_value = description
                        self.las_file.well.descriptions[mnemonic] = value
                    else:
                        # Ambiguous — default to the spec layout
                        # interpretation (value after colon) and warn.
                        # P-01: The previous remedy ("set well_format='cwls'")
                        # was bogus — cwls and lasio are handled with the
                        # same value/description extraction per the CWLS
                        # 1989 label-left/value-right layout; the parameter
                        # is retained for API compatibility only.
                        logger.warning(
                            "Cannot distinguish CWLS from lasio convention for "
                            "well field '%s' with pre-colon value '%s' "
                            "and post-colon description '%s'. "
                            "Defaulting to the spec-layout interpretation "
                            "(value after colon).",
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

    def _replay_deferred_well(self, final: bool = False) -> None:
        """Re-process well entries (and data lines) that were parsed before ~V was known.

        When ~W appears before ~V, entries are buffered without being stored.
        Once ~V is parsed, all deferred entries are re-processed with the
        correct version-based swap logic (LAS 1.2 vs 2.0+).

        F-M12: When data sections appear before ~V, data lines are buffered
        and replayed here after the version is known.

        Args:
            final: True when called at the END of parse() after every section
                (including a trailing ~C block) has been processed.  When
                False (a mid-parse flush, e.g. from _flush_ascii_data), a
                deferred bare LOG_DATA section whose main-curve scope is not
                yet recorded is RE-QUEUED instead of processed (F-21) — the
                main ~C block may still appear later in the file.
        """
        if self._state.deferred_well_entries:
            is_las12 = _LASVersionSpec(self.las_file.version.vers).is_las12
            for entry in self._state.deferred_well_entries:
                self._store_well_entry(
                    mnemonic=entry["mnemonic"],
                    unit=entry["unit"],
                    value=entry["value"],
                    description=entry["description"],
                    is_las12=is_las12,
                    raw_mnemonic=entry["raw_mnemonic"],
                    no_period=entry.get("no_period", "False") == "True",
                )
            self._state.deferred_well_entries.clear()

        # F-M12: Replay deferred data lines if the file is LAS 3.0.
        # F-H01 / I2-D2-01: Each deferred entry is a (section_type,
        # section_name, section_idx, line, curve_start, curve_end)
        # tuple.  Group by (section_type, section_name, section_idx)
        # to create one DataSection per original pre-~V data section.
        # curve_start/curve_end preserve pipe-target scoping through
        # the deferred replay so that |CURVE and |X_Definition pipe
        # associations are not lost.
        if self._state.deferred_ascii_data_lines:
            if self.las_file.version.is_las30:
                # Build groups from per-line tuple storage.
                # itertools.groupby groups consecutive lines with the same
                # key — correct since deferred lines are appended in file
                # order and section boundaries are naturally contiguous.
                groups: list[tuple[tuple[str, str, int], list[str], int, int | None]] = []
                for key, group_iter in groupby(
                    self._state.deferred_ascii_data_lines,
                    key=lambda t: (t[0], t[1], t[2]),
                ):
                    rows = list(group_iter)
                    groups.append(
                        (
                            key,
                            [r[3] for r in rows],
                            rows[0][4],  # curve_start_idx
                            rows[0][5],  # curve_end_idx
                        )
                    )

                # Save current state before processing deferred groups.
                # F-A2: Also save current_data_section_idx and
                # cumulative_elements — they are mutated in the loop body
                # (lines 1767, 1772) and must roll back on exception so
                # the intermediate state is not observable.
                saved_lines = self._state.ascii_data_lines
                saved_curve_start = self._state.section_curve_start_idx
                saved_curve_end = self._state.section_curve_end_idx
                saved_section_type = self._state.current_data_section_type
                saved_section_name = self._state.current_section_name
                saved_data_section_idx = self._state.current_data_section_idx
                saved_cumulative_elements = self._state.cumulative_elements

                # F-A1: Track whether replay completed successfully.
                # On exception, preserve the deferred buffer so the
                # caller can diagnose what failed (permanent data-loss
                # path when clear() was unconditional in finally).
                replay_successful = False
                # F-21: Lines re-queued for a LATER replay.  A deferred bare
                # LOG_DATA section whose main-curve scope is not yet recorded
                # cannot be resolved against the current partial curve list;
                # its tuples are moved here so the final replay (parse() →
                # _replay_deferred_well(final=True), after the trailing ~C
                # range is recorded) binds them to the COMPLETE curve list.
                requeued_lines: list[tuple[str, str, int, str, int, int | None]] = []
                try:
                    # Process each deferred group as its own DataSection.
                    # N-I-03/P-03: Use the per-group STORED section_idx
                    # (t[2]) for the AsciiDataContext instead of the live
                    # counter.  The live counter may have advanced past the
                    # stored values (e.g. via post-~V flushes), which
                    # misattributes MAX_DATA_SECTIONS and produces
                    # Section_1/Section_2 names instead of Section_0/
                    # Section_1.  The live counter is still advanced per
                    # group below (F-M3) so post-replay sections continue
                    # from a consistent position.
                    for (
                        section_type,
                        _section_name,
                        section_idx,
                    ), raw_lines, curve_start, curve_end in groups:
                        # I2-D2-01: Restore pipe-target curve scoping stored
                        # at defer time, preserving |CURVE and
                        # |X_Definition associations.
                        # M-69: A deferred "| CURVE" scope stored as the
                        # sentinel was unresolvable at defer time (pre-~V,
                        # curves not yet parsed).  Re-resolve it against the
                        # now-known main curve block (M-67/M-69 coherent fix:
                        # the same _resolve_main_curve_scope() handles the
                        # frozen-at-0 M-67 direction and the unfrozen→None
                        # M-69 direction).  Without re-resolution the None
                        # scope would map data against the FINAL curve list,
                        # producing phantom DEPT_2/GR_2 null columns.
                        # PARS-06: A forward "| X_Definition" pipe (the
                        # target definition was parsed LATER) stored
                        # _DEFERRED_PIPE_SCOPE; resolve it to the
                        # definition's range now that it exists.
                        # I2-09: A BARE (no-pipe) deferred section stored
                        # curve_end=None (unbounded) — the classification
                        # fallback.  At replay the unbounded scope maps data
                        # against the FINAL curve list, picking up phantom
                        # columns AND polluting the top-level model with
                        # fabricated DEPT_2/GR_2 names (F2-07 writeback).
                        # Mirror the non-deferred bare-section resolution
                        # (parser.py classification: def_prefix → matching
                        # _Definition, then __MAIN__, then unbounded) so the
                        # deferred path scopes identically to its
                        # non-deferred twin.
                        if curve_end == _DEFERRED_MAIN_CURVE_SCOPE:
                            # M7 (F-21 incomplete fix): the "| CURVE" scope
                            # sentinel means the pipe target was unresolved
                            # at defer time (main curve block not yet
                            # parsed).  At a MID-PARSE flush the main ~C
                            # block may STILL not be parsed — resolving NOW
                            # binds the group to the current PARTIAL curve
                            # list (e.g. only the CORE_DEFINITION curves),
                            # silently discarding the section's extra
                            # columns.  Re-queue like the bare-~A variant
                            # below; the final replay resolves the sentinel
                            # against the COMPLETE main curve block once the
                            # trailing ~C range is recorded.
                            if (
                                not final
                                and "__MAIN__" not in self._state.definition_curve_ranges
                                and "__MAIN_ALL__" not in self._state.definition_curve_ranges
                            ):
                                for _ln in raw_lines:
                                    requeued_lines.append(
                                        (
                                            section_type,
                                            _section_name,
                                            section_idx,
                                            _ln,
                                            curve_start,
                                            curve_end,
                                        )
                                    )
                                continue
                            curve_start, curve_end = self._resolve_main_curve_scope()
                        elif curve_end == _DEFERRED_PIPE_SCOPE:
                            _pipe_target = self._deferred_pipe_targets.get(
                                (section_type, _section_name, section_idx)
                            )
                            if _pipe_target and _pipe_target in self._state.definition_curve_ranges:
                                curve_start, curve_end = self._state.definition_curve_ranges[
                                    _pipe_target
                                ]
                            else:
                                curve_start, curve_end = self._resolve_main_curve_scope()
                        elif curve_end is None:
                            _def_prefix = (section_type or "LOG_DATA").replace(
                                "_DATA", ""
                            ) + "_DEFINITION"
                            if _def_prefix in self._state.definition_curve_ranges:
                                curve_start, curve_end = self._state.definition_curve_ranges[
                                    _def_prefix
                                ]
                            elif "__MAIN__" in self._state.definition_curve_ranges:
                                curve_start, curve_end = self._state.definition_curve_ranges[
                                    "__MAIN__"
                                ]
                            elif (
                                not final
                                and "__MAIN__" not in self._state.definition_curve_ranges
                                and "__MAIN_ALL__" not in self._state.definition_curve_ranges
                            ):
                                # F-21: The main ~C block has NOT been parsed
                                # yet — this is a mid-parse flush (e.g. a
                                # typed data section that appeared between
                                # ~V and the main ~C).  Resolving the bare
                                # LOG_DATA scope NOW would bind it to the
                                # current PARTIAL curve list (e.g. only the
                                # CORE_DEFINITION curves), silently discarding
                                # the section's extra columns.  Re-queue the
                                # group; the final replay (parse() →
                                # _replay_deferred_well(final=True)) resolves
                                # it against the complete curve list once the
                                # trailing ~C range is recorded.
                                for _ln in raw_lines:
                                    requeued_lines.append(
                                        (
                                            section_type,
                                            _section_name,
                                            section_idx,
                                            _ln,
                                            curve_start,
                                            curve_end,
                                        )
                                    )
                                continue
                            else:
                                curve_start, curve_end = self._resolve_main_curve_scope()
                        self._state.section_curve_start_idx = curve_start
                        self._state.section_curve_end_idx = curve_end
                        # Pre-~V section type may be stale — use stored value.
                        self._state.current_data_section_type = section_type or "LOG_DATA"
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
                            self._state.current_section_name = _section_name
                        else:
                            self._state.current_section_name = ""
                        # PARS-04: A deferred (pre-~V) data section bypassed
                        # the M-40 standalone-mnemonic-header skip at
                        # accumulation time (curves were not yet parsed, so
                        # _is_standalone_mnemonic_header conservatively
                        # returned False).  Apply the same header-row filter
                        # here at replay, when the curve block IS known —
                        # mirroring the non-deferred branch in
                        # _parse_ascii_data.  Without this, a mnemonic
                        # header row directly below ~A becomes a phantom
                        # all-null first row (PARS-04).
                        _filtered_lines: list[str] = []
                        for _ln in raw_lines:
                            # DR-M2/DR-M3 coordination: the same
                            # first-line-of-section restriction as the
                            # non-deferred branch — only while no data row
                            # has been accumulated can a row be a standalone
                            # mnemonic header.  A mid-section all-mnemonic
                            # row is DATA (skipping it loses values / shifts
                            # columns).
                            if not _filtered_lines and self._is_standalone_mnemonic_header(_ln):
                                warnings.warn(
                                    f"Standalone curve-mnemonic header row "
                                    f"encountered in deferred (pre-~V) data "
                                    f"section ('{_section_name or 'ASCII'}').  "
                                    f"Skipping the row — curve mnemonics "
                                    f"belong on the ~A section line per the "
                                    f"LAS specification.",
                                    UserWarning,
                                    stacklevel=2,
                                )
                                continue
                            _filtered_lines.append(_ln)
                        self._state.ascii_data_lines = _filtered_lines
                        ctx = AsciiDataContext(
                            las_file=self.las_file,
                            ascii_data_lines=self._state.ascii_data_lines,
                            section_curve_start_idx=self._state.section_curve_start_idx,
                            section_curve_end_idx=self._state.section_curve_end_idx,
                            current_section_name=self._state.current_section_name,
                            current_data_section_type=self._state.current_data_section_type,
                            current_data_section_idx=section_idx,
                            cumulative_elements=self._state.cumulative_elements,
                        )
                        process_ascii_data(ctx)
                        self._state.cumulative_elements = ctx.cumulative_elements
                        # F-M3: Per-group increment restored.  The F-54 fix
                        # consolidated this into a single finally-block += 1,
                        # but with 2+ deferred groups the counter must
                        # increment once per group, not once total.
                        self._state.current_data_section_idx += 1
                    replay_successful = True
                finally:
                    # Restore state.
                    if saved_lines:
                        self._state.ascii_data_lines = saved_lines
                    else:
                        self._state.ascii_data_lines = []
                    self._state.section_curve_start_idx = saved_curve_start
                    self._state.section_curve_end_idx = saved_curve_end
                    self._state.current_data_section_type = saved_section_type
                    # F-I2-M01: Reset section name (defense-in-depth).
                    self._state.current_section_name = saved_section_name or ""
                    # F-A1/F-A2: On exception, roll back current_data_section_idx
                    # and cumulative_elements to pre-replay values so
                    # intermediate state from partial replay is not observable.
                    if not replay_successful:
                        self._state.current_data_section_idx = saved_data_section_idx
                        self._state.cumulative_elements = saved_cumulative_elements
                    # F-M3: Per-group current_data_section_idx increment
                    # moved back inside the for loop (one per deferred group).
                    # F-A1: Only clear on success — preserve buffer on
                    # exception so callers can diagnose what failed.
                    # F-21: Re-queued groups (whose main-curve scope was not
                    # yet known) survive the successful replay; processed
                    # groups are dropped.
                    if replay_successful:
                        self._state.deferred_ascii_data_lines = requeued_lines
            else:
                # F-021: Non-LAS 3.0 files — clear deferred lines to
                # prevent false-positive _ParserState.validate() warnings
                # about "dangling data."  Without this, the deferred buffer
                # persists across file boundaries (only _reset() clears it).
                self._state.deferred_ascii_data_lines.clear()

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
        no_period_match = False
        if not match:
            # F-23 (a): lasio missing-period lines ("LOC : ACME:OIL") have
            # no dot; parse them (mnemonic left of first colon, value right,
            # empty unit/description) instead of silently dropping the entry.
            # M10 (F-23(a) regression): mirror the curve/param grammar guard
            # — a multi-word mnemonic ("HOLE DIA") cannot be represented by
            # WellSection (the writer's N-I-19 key validation rejects
            # embedded spaces and the parser's ~W regex cannot roundtrip
            # them), so keep the pre-fix drop-with-warning for those instead
            # of storing a key the writer would crash on.
            match = self._match_data_line_no_period(line)
            if match and not _MNEMONIC_LINE_RE.fullmatch(match.group("mnemonic")):
                match = None
            elif match:
                # L24 (M10 region): the no-period form already consumed the
                # FIRST colon as the MNEM:VALUE separator — everything right
                # of it is the value.  The LAS 1.2 bare-colon CWLS split in
                # _store_well_entry (designed for the DOT-anchored
                # "MNEM. DESC:VALUE" form) must NOT re-split a colon inside
                # a no-period value ("LOC : ACME:OIL" → value must stay
                # "ACME:OIL", not corrupt to "OIL").
                no_period_match = True
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
        is_las12 = _LASVersionSpec(self.las_file.version.vers).is_las12

        # F-P06: When ~W appears before ~V, the version defaults to "2.0" and
        # is_las12 is False, skipping the LAS 1.2 convention swap.  Buffer raw
        # entries so they can be re-processed with the correct version after
        # ~V is parsed.
        if not self._state.version_found:
            # F-35: Guard against unbounded deferred-well-entry accumulation.
            # Every other accumulator in parser.py has a MAX_* guard; this was
            # the sole unguarded buffer.  Malicious ~W-before-~V files without
            # a ~V section could grow this list without bound.
            if len(self._state.deferred_well_entries) >= MAX_DEFERRED_WELL_ENTRIES:
                raise LASParseError(
                    f"Line {self._line_no}: deferred well entry count "
                    f"({len(self._state.deferred_well_entries) + 1}) "
                    f"exceeds maximum allowed ({MAX_DEFERRED_WELL_ENTRIES}). "
                    f"The file may be malformed or corrupt."
                )
            self._state.deferred_well_entries.append(
                {
                    "mnemonic": mnemonic,
                    "raw_mnemonic": raw_mnemonic,
                    "unit": unit,
                    "value": value,
                    "description": description,  # type: ignore[dict-item]
                    # Stored as a string ("True"/"False") because
                    # _ParserState.deferred_well_entries is a dict[str, str].
                    "no_period": str(no_period_match),
                }
            )
            if len(self._state.deferred_well_entries) == 1:
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

        self._store_well_entry(
            mnemonic, unit, value, description, is_las12, raw_mnemonic, no_period_match
        )

    def _normalize_curve_mnemonic(self, raw_mnemonic: str) -> str:
        """Normalize a curve mnemonic through mnem_base, preserving identity
        on resolution collisions (N-I-30 parity with models.from_dict).

        MNEM_BASE maps distinct vendor mnemonics to the same canonical
        (e.g. ``"LLD"→"BFV"`` AND ``"LLS"→"BFV"`` — a standard
        dual-laterolog file carries both).  Plain
        ``self._mnem_base_upper.get(raw, raw)`` turns both into ``"BFV"``,
        producing a duplicate ``curves_order`` entry; the resulting model
        then fails the to_dict→from_dict / write_las_file roundtrip with a
        "duplicate curve name" error (M-36).  Detect the collision during
        normalization: keep the ORIGINAL mnemonic for the colliding curve
        (preserving identity) and warn.  Genuine duplicates (the SAME raw
        name twice) still resolve identically and are caught by the
        existing duplicate-name checks.
        """
        resolved = self._mnem_base_upper.get(raw_mnemonic, raw_mnemonic)
        _prev_raw = self._resolved_curve_names.get(resolved)
        if _prev_raw is not None and _prev_raw != raw_mnemonic:
            if raw_mnemonic == resolved:
                # PXM-06: The canonical name itself collides with an alias
                # that claimed the resolved slot earlier (e.g. GR then GK
                # with GR→GK: GR normalized to GK first, then the canonical
                # GK arrives).  The canonical wins its own slot; the earlier
                # alias entry is re-keyed to its ORIGINAL mnemonic in
                # curves_order and its CurveDefinition so both distinct
                # curves keep distinct identities.  Mirror of
                # models.from_dict _norm_curve_mnem raw==resolved branch
                # (models.py:3707-3740).
                if (raw_mnemonic, resolved) not in self._warned_collisions:
                    self._warned_collisions.add((raw_mnemonic, resolved))
                    warnings.warn(
                        f"parser: canonical curve mnemonic '{resolved}' "
                        f"collides with alias '{_prev_raw}' (which resolves "
                        f"to '{resolved}') earlier in the curve list.  "
                        f"Preserving both: '{_prev_raw}' keeps its original "
                        f"mnemonic and '{resolved}' keeps the canonical name.",
                        UserWarning,
                        stacklevel=2,
                    )
                # Re-key the earlier alias's curves_order entry and
                # CurveDefinition to its ORIGINAL mnemonic.  The earlier
                # alias's CurveDefinition keeps original_mnemonic == _prev_raw
                # (the file's casing), so it is identified by that, not by
                # position — a later canonical is appended after this call.
                for _i, _n in enumerate(self.las_file.curves_order):
                    if _n == resolved:
                        self.las_file.curves_order[_i] = _prev_raw
                        break
                for _c in self.las_file.curves:
                    if _c.mnemonic == resolved and _c.original_mnemonic == _prev_raw:
                        _c.mnemonic = _prev_raw
                        break
                self._resolved_curve_names[_prev_raw] = _prev_raw
                self._resolved_curve_names[raw_mnemonic] = raw_mnemonic
                return resolved
            # Warn once per (raw, resolved) pair.
            if (raw_mnemonic, resolved) not in self._warned_collisions:
                self._warned_collisions.add((raw_mnemonic, resolved))
                warnings.warn(
                    f"parser: mnemonic '{raw_mnemonic}' resolves to "
                    f"'{resolved}' via mnem_base, but '{resolved}' is "
                    f"already used by curve '{_prev_raw}'.  Keeping the "
                    f"original mnemonic '{raw_mnemonic}' to preserve "
                    f"curve identity.",
                    UserWarning,
                    stacklevel=2,
                )
            self._resolved_curve_names[raw_mnemonic] = raw_mnemonic
            return raw_mnemonic
        self._resolved_curve_names[resolved] = raw_mnemonic
        return resolved

    def _parse_curve(self, line: str) -> None:
        """Parse ~C (curve information) section line.

        Supports LAS 3.0 features:
        - Array notation: NMR[1], NMR[2], etc.
        - Format specifiers: {F}, {E}, {S}, {A:0}
        """
        match = self._match_data_line(line)
        if not match:
            # F-23 (a): lasio missing-period lines ("HOLE DIA : 85.7").
            # A single-word mnemonic ("GR : GAMMA") can be represented by
            # CurveDefinition; a multi-word mnemonic ("HOLE DIA") cannot
            # (models.py rejects embedded spaces) — keep the pre-fix
            # drop-with-warning for those instead of raising LASParseError.
            match = self._match_data_line_no_period(line)
            if match and not _MNEMONIC_LINE_RE.fullmatch(match.group("mnemonic")):
                match = None
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
        # F-23 (b): The parser's unit class now accepts ':' so
        # colon-in-unit lines ("TIME.hh:mm 23:15 ...") parse instead of
        # being dropped.  CurveDefinition's unit grammar (models.py
        # _UNIT_PATTERN) still rejects ':' — the writer never escapes a
        # mid-unit colon, so a stored ':' unit would crash the model on
        # write→read.  Strip the colon from the stored unit (the line's
        # value/description — the data that matters — are preserved).
        unit = unit.replace(":", "")
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
        if not self.las_file.version.is_las30:
            # M-35: On LAS 1.2/2.0 the writer re-emits only {I} markers
            # (EXT-04); {S}/{A} markers are ALSO functionally meaningful
            # here because data_reader._detect_string_curves classifies
            # string curves purely by data_format ("S", or "A" without
            # array_info).  Other brace format tokens ({F}, {E}, {D})
            # are user description text on non-3.0 files (they classify
            # as numeric either way) and must be preserved.  Filter the
            # extraction candidates to the functional markers only.
            format_matches = [m for m in format_matches if m[0].upper() in ("I", "S", "A")]
        if format_matches:
            # N-I-18: Prefer the TRAILING (last) format specifier.  The
            # writer appends the curve's real data_format at the END of the
            # description (``desc  {F}``), so for writer-produced files the
            # trailing match is the authoritative format.  Using the FIRST
            # match mis-extracted a brace token from user description text
            # (e.g. "Gamma {F} log" + real format E → data_format="F") or
            # lost the real format entirely (e.g. "Bulk {Density}" + real
            # format F → "DENSITY" fails validation → data_format="").
            # For standard single-format descriptions first==trailing, so
            # documented behavior is unchanged.
            first_fmt, first_offset = format_matches[-1]
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
                _validate_curve_data_format(data_format, raw_mnemonic, line_no=self._line_no)
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
                        f"Line {self._line_no}: invalid format specifier "
                        f"offset: '{first_offset}' is not a valid number "
                        f"in curve description '{description}'"
                    ) from exc
                if not np.isfinite(array_time_offset):
                    raise LASParseError(
                        f"Line {self._line_no}: format specifier offset "
                        f"overflow: '{first_offset}' produced "
                        f"{array_time_offset} in curve description "
                        f"'{description}'"
                    )
            if len(format_matches) > 1:
                # N-I-18: The trailing (writer-appended) format is the
                # authoritative one; earlier brace tokens are treated as
                # user description text / non-authoritative formats.
                extra_formats = [f[0] for f in format_matches[:-1]]
                logger.warning(
                    "Multiple format specifiers found in curve '%s' "
                    "description: %s. Only the trailing (%s) is used; "
                    "earlier specifiers %s are discarded.",
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
            # M-35: On LAS 1.2/2.0, only the functional markers {I}
            # (integer — writer re-emits on every version) and {S}/{A}
            # (string classification — data_reader needs data_format) are
            # stripped; {F}/{E}/{D}-style brace tokens are preserved as
            # user text because the non-3.0 writer cannot restore them.
            def _keep_non_format(m: re.Match[str]) -> str:
                fmt = m.group("format").upper()
                if not self.las_file.version.is_las30 and fmt not in ("I", "S", "A"):
                    return m.group(0)  # M-35: user text on non-3.0 — keep it
                try:
                    _validate_curve_data_format(
                        fmt,
                        raw_mnemonic,
                        line_no=self._line_no,
                    )
                    # L30-01 (coordination with
                    # _las30_data._build_spec_form_array_info): LAS 3.0
                    # spec-form array channels (plain repeated mnemonics +
                    # {A:N} offset markers) need the {A:N} marker to SURVIVE
                    # so the las30 synthesis layer can extract time_offset.
                    # Bracket mnemonics carry the offset in array_info
                    # (parser.py:2901-2915) and keep the stripped
                    # description; plain-mnemonic A-format curves with an
                    # offset are spec-form candidates and must keep the
                    # marker.  The las30 side strips the marker from the
                    # description after extraction
                    # (_las30_data.py:577-578), so the writer re-emits
                    # {A:N} exactly once (from array_info.time_offset).
                    if (
                        fmt == "A"
                        and self.las_file.version.is_las30
                        and "[" not in raw_mnemonic
                        and m.group("offset")
                    ):
                        return m.group(0)
                    return ""  # Valid format specifier → strip it
                except LASParseError:
                    return m.group(0)  # Non-format text → keep it

            description = FORMAT_SPEC_PATTERN.sub(_keep_non_format, description).strip()

        # LAS 3.0: Check for array notation in mnemonic
        array_info: ArrayElementInfo | None = None
        array_match = ARRAY_MNEMONIC_PATTERN.match(raw_mnemonic)
        if array_match:
            base_name = array_match.group("base").upper()
            try:
                index = int(array_match.group("index"))
            except ValueError as exc:
                raise LASParseError(
                    f"Line {self._line_no}: invalid array index "
                    f"'{array_match.group('index')}' in curve "
                    f"mnemonic '{raw_mnemonic}'"
                ) from exc
            # P-11/M-26: ArrayElementInfo.__post_init__ raises bare
            # ValueError (e.g. negative {A:-5} time_offset) which must
            # not escape parse() — normalize to LASParseError.
            try:
                array_info = ArrayElementInfo(
                    base_name=base_name,
                    index=index,
                    time_offset=array_time_offset,
                )
            except ValueError as exc:
                raise LASParseError(
                    f"Line {self._line_no}: invalid array element info "
                    f"for curve '{raw_mnemonic}': {exc}"
                ) from exc
        elif "[" in raw_mnemonic:
            # F-M-007: Warn when mnemonic contains "[" but doesn't match
            # ARRAY_MNEMONIC_PATTERN (e.g., NMR[-1], NMR[abc], NMR[]).
            logger.warning(
                "Mnemonic %r contains '[' but does not match array notation "
                "pattern; treated as standalone curve.",
                raw_mnemonic,
            )

        # Apply mnemonic normalization from mnem_base.
        # M-36: Collision-aware — when two distinct raw mnemonics resolve
        # to the same canonical (e.g. LLD/LLS → BFV), keep the ORIGINAL
        # mnemonic for the colliding curve so the model stays roundtrip-able.
        normalized = self._normalize_curve_mnemonic(raw_mnemonic)

        # F-M-026: Wrap CurveDefinition construction to catch ValueError
        # from __post_init__ validation (e.g., empty mnemonic after
        # mnem_base normalization) and re-raise as LASParseError.
        # F-34: Preserve case-only original mnemonics.  ``normalized`` is
        # always uppercase (raw_mnemonic = _original_cased.upper(), and
        # mnem_base resolution is case-insensitive), so comparing the
        # file's original casing DIRECTLY against it keeps original_mnemonic
        # whenever the file casing differs in ANY way — mnem_base
        # normalization ("AK"→"DT") OR case alone ('dept'→'DEPT').  The
        # previous ``_original_cased.upper() != normalized`` guard erased
        # the case signal and cleared original_mnemonic for case-only
        # differences, so the writer re-emitted the canonical casing
        # instead of the file's (contradicting the F-01 comment above:
        # "CurveDefinition.original_mnemonic reflects the file's casing").
        # original_mnemonic is cleared only when it is byte-identical to
        # the canonical mnemonic.
        try:
            curve = CurveDefinition(
                mnemonic=normalized,
                unit=unit,
                api_code=api_code,
                description=description,
                original_mnemonic=_original_cased if _original_cased != normalized else "",
                data_format=data_format,
                array_info=array_info,
            )
        except ValueError as e:
            raise LASParseError(
                f"Line {self._line_no}: invalid curve definition for mnemonic {raw_mnemonic!r}: {e}"
            ) from e
        # F-28: Guard against unbounded curve accumulation during ~C parsing.
        # Without this check, a metadata-only LAS 3.0 file can accumulate
        # unlimited CurveDefinition objects without triggering any bounds
        # check (_data_reader.MAX_CURVES was only checked later in _process_ascii_data,
        # which early-returns when no data section exists).
        if len(self.las_file.curves) >= _data_reader.MAX_CURVES:
            raise LASParseError(
                f"Line {self._line_no}: curve count "
                f"({len(self.las_file.curves) + 1}) exceeds maximum "
                f"allowed ({_data_reader.MAX_CURVES}). "
                f"The file may be malformed or corrupt."
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
            # F-23 (a): lasio missing-period lines ("MUD WT : 10").  Same
            # grammar guard as _parse_curve — ParameterEntry also rejects
            # embedded-space mnemonics.
            match = self._match_data_line_no_period(line)
            if match and not _MNEMONIC_LINE_RE.fullmatch(match.group("mnemonic")):
                match = None
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
        # F-23 (b): strip ':' from the stored parameter unit — same model
        # constraint as _parse_curve (ParameterEntry._UNIT_PATTERN rejects
        # ':' and the writer never escapes a mid-unit colon).
        unit = unit.replace(":", "")
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
        if not self.las_file.version.is_las30:
            # M-35: On LAS 1.2/2.0 the parameter writer re-emits NO
            # format specifier (only ``is_las30 and param.data_format``
            # in _writer_base._format_parameter_line — there is no {I}
            # exception for parameters like curves have).  Every brace
            # format token in a non-3.0 parameter description is user
            # text and must be preserved.
            param_format_matches = []
        if param_format_matches:
            # M-03: Prefer the TRAILING (last) format specifier, matching
            # the curve path (N-I-18 in _parse_curve).  The writer appends
            # the parameter's real data_format at the END of the
            # description (``desc  {fmt}``), so for writer-produced files
            # the trailing match is authoritative.  The previous
            # ``[0]`` (FIRST) mis-extracted a brace token from user text
            # (e.g. "Mud type {S} in hole" + real format E → data_format
            # 'S') and silently replaced the real format on roundtrip.
            param_data_format = param_format_matches[-1][0].upper()
            if len(param_format_matches) > 1:
                extra_formats = [f[0] for f in param_format_matches[:-1]]
                logger.warning(
                    "Multiple format specifiers found in parameter '%s' "
                    "description: %s. Only the trailing (%s) is used; "
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
                        _validate_curve_data_format(
                            param_data_format,
                            raw_mnemonic,
                            line_no=self._line_no,
                        )
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
                        param_data_format = (
                            ""  # Not a valid data format — clear to prevent accumulation
                        )
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
            # M-35: On LAS 1.2/2.0, no parameter format specifier is re-emitted
            # by the writer — every brace format token ({F}, {E}, {S}, {A},
            # {I}) is user text and must be preserved.
            # H-04: The sub is gated on `if param_format_matches:` — it
            # previously ran UNCONDITIONALLY (indent-8, outside the if) on
            # every ~P line, doubling the ~P cost (findall + sub) even with
            # zero matches (44s @48KB observed).  Gating is behavior-identical:
            # a zero-match sub returns the description unchanged, and the
            # non-3.0 M-35 branch empties the list so user brace text is kept.
            def _keep_non_format_param(m: re.Match[str]) -> str:
                if not self.las_file.version.is_las30:
                    return m.group(0)  # M-35: user text on non-3.0 — keep it
                try:
                    _validate_curve_data_format(
                        m.group("format").upper(),
                        raw_mnemonic,
                        line_no=self._line_no,
                    )
                    return ""  # Valid format specifier → strip it
                except LASParseError:
                    return m.group(0)  # Non-format text → keep it

            description = FORMAT_SPEC_PATTERN.sub(_keep_non_format_param, description).strip()

        # N-I-02: Zone association (| Zone) is a LAS 3.0 feature.  The
        # previous code ran ZONE_ASSOC_PATTERN UNCONDITIONALLY — for
        # LAS 1.2/2.0 files a pipe-suffix in a description (e.g. "Run
        # number | Main Zone") was truncated, a bogus ParameterZone was
        # attached, and since the writer never re-emits zones for
        # non-3.0 files the pipe text was permanently lost on roundtrip.
        # Gating on is_las30 preserves the pipe text in the description
        # for LAS 1.2/2.0.
        zone: ParameterZone | None = None
        if self.las_file.version.is_las30:
            zone_match = ZONE_ASSOC_PATTERN.search(description)
            if zone_match:
                zone_index: int | None = None
                if zone_match.group("index"):
                    try:
                        zone_index = int(zone_match.group("index"))
                    except ValueError as exc:
                        raise LASParseError(
                            f"Line {self._line_no}: invalid zone index "
                            f"'{zone_match.group('index')}' in "
                            f"parameter '{raw_mnemonic}'"
                        ) from exc
                # F-01: Preserve original zone name casing
                _orig_zone = zone_match.group("zone")
                # M-12: the writer escapes literal pipes in zone names
                # (\|) so they do not split the zone association on
                # re-read; restore them here so "Zone|X" roundtrips.
                _orig_zone = _unescape_pipes_for_las_value(_orig_zone)
                zone = ParameterZone(
                    zone_name=_orig_zone.upper(),
                    zone_index=zone_index,
                )
                # Remove zone association from description
                description = ZONE_ASSOC_PATTERN.sub("", description).strip()
        # N-I-02(b): The writer escapes literal pipes in parameter
        # descriptions (| → \|) so genuine description text containing a
        # pipe is not misparsed as a zone association on re-read.
        # Unescape the escape-artifact here (applies to both LAS 3.0 —
        # where a real zone was already extracted above — and earlier
        # versions, where the pipe text was preserved in the description).
        description = _unescape_pipes_for_las_value(description)

        # LAS 3.0: Check for array notation in mnemonic
        array_index: int | None = None
        array_match = ARRAY_MNEMONIC_PATTERN.match(raw_mnemonic)
        if array_match:
            try:
                array_index = int(array_match.group("index"))
            except ValueError as exc:
                raise LASParseError(
                    f"Line {self._line_no}: invalid array index "
                    f"'{array_match.group('index')}' in "
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
        _sect_name = (self._state.current_section_name or "").upper()
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
                f"Line {self._line_no}: invalid parameter entry for mnemonic {raw_mnemonic!r}: {e}"
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
                f"Line {self._line_no}: parameter count "
                f"({len(self.las_file.parameters) + 1}) exceeds "
                f"maximum allowed ({MAX_PARAMETERS}). "
                f"The file may be malformed or corrupt."
            )
        self.las_file.parameters.append(param)

    def _parse_other(self, line: str) -> None:
        """Parse ~O (other) section — free-form text, accumulated.

        W-08: The writer's ``_sanitize_las_value`` escapes ``#``-prefixed
        lines as ``_#comment`` so the parser's COMMENT_PATTERN does not
        drop them before the ~O accumulator sees them.  Restore the
        original value here (``_#`` → ``#``) so a ``#comment`` in the model
        round-trips through write→read.  The thread-local desanitize flag
        is honored inside the scoped helper (desanitize=False reads keep
        the raw text, matching the data-row contract).

        PF-02: The restore is scoped to the writer's actual escape
        positions (``_desanitize_other_line``) — line-start ``_#`` only —
        so genuine ``_~``-prefixed and mid-line ``_#`` content survives
        write→read unchanged.
        """
        self._append_other_line(_desanitize_other_line(line))

    def _is_standalone_mnemonic_header(self, line: str) -> bool:
        """Detect a standalone mnemonic header row inside a LAS 3.0 ~A section.

        Some LAS 3.0 files repeat the curve mnemonics on a header line
        directly below ~A (e.g. ``~A`` followed by ``DEPT,GR``).  The LAS
        spec places the mnemonics on the ~A header line itself, so such a
        row is NOT data — consuming it produces a phantom all-null first
        row and shifts every value by one position (M-40).

        Detection accepts FULL and PARTIAL header rows: the delimiter-split
        token count must be between min(2, section_count) and the number of
        curves in the section's scope (min(2, section_count)..section_count
        — a partial header like "DEPT,GR" with three declared curves is
        still a header, mirroring the LAS 1.2/2.0 reader's min(2,
        curve_count)..curve_count clause in
        data_reader._is_mnemonic_header_row; the 1-token minimum applies
        only to single-curve sections, PSR-1) AND every token must match one
        of those curves' mnemonics (mnem_base-normalized or the file's
        original casing).  Numeric data rows fail the mnemonic match; a
        genuine duplicate of the header row is indistinguishable from a
        header row by design (skipping it is the correct reading).  The
        callers restrict this predicate to the section's first line(s), so
        a mid-section all-mnemonic data row is never skipped (DR-M3
        coordination).

        F-19: An all-string section (every curve is {S} or plain {A}
        string data) is excluded entirely — there every data row is a
        string, so a value may legitimately coincide with a curve mnemonic
        (e.g. LITH=['LITH','SHALE']) and a mnemonic-coincident row is
        indistinguishable from a header row.  String data rows are never
        dropped.  Only a section with at least one numeric curve is
        structurally unambiguous: there a numeric data row can never be
        all-mnemonic, so mnemonic-coincidence remains a reliable header
        signal.

        Returns:
            True when the line should be treated as a mnemonic header and
            skipped, False when it should be accumulated as a data row.
        """
        if not self.las_file.curves:
            # Curves not parsed yet (e.g. data-before-~C) — cannot verify
            # mnemonics; treat the line as data (conservative).
            return False
        delimiter = self.las_file.version.delimiter_char
        stripped = line.strip()
        if delimiter == " ":
            tokens = stripped.split()
        else:
            tokens = stripped.split(delimiter)
            # Mirror _las30_data.process_ascii_data: strip trailing empty
            # fields (trailing delimiter produces phantom columns).
            while tokens and tokens[-1] == "":
                tokens.pop()
        if not tokens:
            return False
        start = self._state.section_curve_start_idx
        end = self._state.section_curve_end_idx
        # F-22: Check the token count against the curve-range SIZE before
        # slicing.  Slicing first allocated a full CurveDefinition-ref list
        # per data line (up to 100K refs at MAX_CURVES) for every line whose
        # token count would reject it — an attacker-controlled 10M-line file
        # burned ~47 min of CPU before MAX_TOTAL_ELEMENTS fired.  The
        # count-only check rejects most rows without allocating anything.
        if end is None:
            section_count = len(self.las_file.curves) - start
        else:
            section_count = end - start
        # DR-M2 (Stage 10): a PARTIAL mnemonic header (2..section_count
        # tokens — e.g. "DEPT,GR" with 3 declared curves) is still a header,
        # mirroring the LAS 1.2/2.0 reader's 2..curve_count clause
        # (data_reader._is_mnemonic_header_row).  Strict equality consumed
        # the partial header as a data row → phantom all-null first row +
        # value shift on the LAS 3.0 path.
        # PSR-1 (Stage 11): the DR-M2 2-token minimum regressed a section
        # with EXACTLY ONE curve — a standalone 1-token mnemonic header row
        # ('DEPT') failed `len(tokens) < 2` and was consumed as data →
        # phantom all-null first row + value shift (HEAD's strict equality
        # 1==1 handled it correctly).  The lower bound is min(2,
        # section_count): a single-curve section still recognizes its
        # 1-token header, while multi-curve sections keep the 2-token
        # minimum (M-02 single-token string-data protection — the F-19
        # all-string exclusion handles the single-curve STRING section, and
        # a numeric single-curve mnemonic row is unambiguously a header).
        if len(tokens) < min(2, section_count) or len(tokens) > section_count:
            return False
        # DR-M2: cache the section's mnemonic match set per scope — the
        # 2..section_count clause lets short rows through the count gate, so
        # the O(section_count) slice + set build must happen ONCE per section,
        # not per data line (preserves the F-22 count-reject-fast property
        # for 100K-curve scopes: a 10-token row would otherwise pay
        # O(100K) per row → the CPU-exhaustion DoS F-22 fixed returns).
        _scope_key = (start, end, len(self.las_file.curves))
        _cached = self._mnemonic_header_scope_cache.get(_scope_key)
        if _cached is None:
            if end is None:
                section_curves = self.las_file.curves[start:]
            else:
                section_curves = self.las_file.curves[start:end]
            # F-19: Never treat a row as a mnemonic header in an all-string
            # section.  Every data value there is a string, so a value may
            # legitimately coincide with a curve mnemonic (e.g. LITH data rows
            # 'LITH'/'SHALE' in a {S} section).  A mnemonic-coincident row is
            # indistinguishable from a header row by content alone, and dropping
            # it destroys genuine data (M-40 regression).  Only a section with at
            # least one NUMERIC curve is structurally unambiguous — there a
            # numeric data row can never be all-mnemonic, so mnemonic-coincidence
            # remains a reliable header signal.
            _all_string = all(_is_string_data_curve(curve) for curve in section_curves)
            _expected: set[str] = set()
            for curve in section_curves:
                _expected.add(curve.mnemonic.upper())
                if curve.original_mnemonic:
                    _expected.add(curve.original_mnemonic.upper())
            _cached = (_expected, _all_string)
            self._mnemonic_header_scope_cache[_scope_key] = _cached
        expected, all_string = _cached
        if all_string:
            return False
        return all(tok.upper() in expected for tok in tokens)

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
            if not self._state.version_found:
                if len(self._state.deferred_ascii_data_lines) > _data_reader.MAX_DATA_LINES:
                    raise LASParseError(
                        f"Line {self._line_no}: deferred ASCII data line "
                        f"count exceeds maximum allowed "
                        f"({_data_reader.MAX_DATA_LINES}). "
                        f"The file may be malformed or corrupt."
                    )
                # F-H01: Store per-line (section_type, section_name,
                # section_idx, curve_start, curve_end, line) so
                # _replay_deferred_well can reconstruct per-section
                # grouping.  section_idx disambiguates consecutive bare
                # sections.  curve_start/curve_end preserve pipe-target
                # scoping across the deferred replay (I2-D2-01).
                # M-69: A "| CURVE" scope that could not be resolved at
                # defer time (main_curve_end == -1 — curves not yet parsed)
                # is stored as _DEFERRED_MAIN_CURVE_SCOPE so replay
                # re-resolves it against the now-known main curve block.
                # Without this, the stored None ("all curves from start")
                # is interpreted against the FINAL curve list and picks up
                # phantom columns from later _Definition sections.
                # PARS-06: A "| X_Definition" FORWARD pipe (the target
                # definition appears LATER in the file — not in
                # definition_curve_ranges at defer time) has the same
                # unresolvable-now problem but is NOT covered by the
                # |CURVE sentinel.  Store _DEFERRED_PIPE_SCOPE and record
                # the pipe target keyed by the deferred group so replay can
                # resolve the scope against the definition's range once it
                # has been parsed.
                _deferred_curve_end = self._state.section_curve_end_idx
                if self._current_pipe_target in {"CURVE", "C"} and _deferred_curve_end is None:
                    _deferred_curve_end = _DEFERRED_MAIN_CURVE_SCOPE
                elif (
                    self._current_pipe_target
                    and self._current_pipe_target not in {"CURVE", "C"}
                    and self._current_pipe_target not in self._state.definition_curve_ranges
                ):
                    _deferred_curve_end = _DEFERRED_PIPE_SCOPE
                    self._deferred_pipe_targets[
                        (
                            self._state.current_data_section_type,
                            self._state.current_section_name,
                            self._state.current_data_section_idx,
                        )
                    ] = self._current_pipe_target
                self._state.deferred_ascii_data_lines.append(
                    (
                        self._state.current_data_section_type,
                        self._state.current_section_name,
                        self._state.current_data_section_idx,
                        line,
                        self._state.section_curve_start_idx,
                        _deferred_curve_end,
                    )
                )
            return
        # F-27: Early bounds check during accumulation — reject before the
        # list grows unbounded.  The main check in _process_ascii_data runs
        # AFTER all lines are collected, offering no protection during the
        # accumulation phase itself.
        if len(self._state.ascii_data_lines) > _data_reader.MAX_DATA_LINES:
            raise LASParseError(
                f"Line {self._line_no}: ASCII data line count exceeds "
                f"maximum allowed ({_data_reader.MAX_DATA_LINES}) "
                f"during accumulation. "
                f"The file may be malformed or corrupt."
            )
        # M-40: Skip a standalone mnemonic header row directly below ~A
        # (e.g. "DEPT,GR") — the LAS 3.0 spec puts mnemonics on the ~A
        # line, so this row is a header, not data.  Consuming it would
        # produce a phantom all-null first row and shift every value.
        # Skipping at accumulation time keeps the LAS 3.0 array sizing
        # (actual_count in _las30_data) correct with no pre-scan impact
        # (data_line_count is only consumed by the LAS 1.2/2.0 reader).
        # DR-M2/DR-M3 coordination: the header-skip applies only to the
        # section's FIRST line(s) — the accumulated list is empty exactly
        # while no data row has been consumed, mirroring the LAS 1.2/2.0
        # reader's current_line == 0 gate (data_reader DR-M3).  A
        # mid-section all-mnemonic row is DATA; skipping it silently
        # drops values / shifts columns.
        if not self._state.ascii_data_lines and self._is_standalone_mnemonic_header(line):
            warnings.warn(
                f"Line {self._line_no}: standalone curve-mnemonic header "
                f"row encountered inside ~A data section "
                f"('{self._state.current_section_name or 'ASCII'}').  "
                f"Skipping the row — curve mnemonics belong on the ~A "
                f"section line per the LAS specification.",
                UserWarning,
                stacklevel=2,
            )
            return
        self._state.ascii_data_lines.append(line)
        # F-09: Cumulative cross-section data line counter — defense-in-depth
        # against multi-section files where each section passes the per-section
        # MAX_DATA_LINES bound individually.
        self._state.cumulative_data_lines += 1
        if (
            self._state.cumulative_data_lines > _data_reader.MAX_DATA_LINES * 10
            and not self._state.cumulative_data_lines_warned
        ):
            self._state.cumulative_data_lines_warned = True
            warnings.warn(
                f"Cumulative data line count ({self._state.cumulative_data_lines}) "
                f"across {self._state.current_data_section_idx + 1} sections "
                f"is unusually high.  The file may be malformed or corrupt.",
                UserWarning,
                stacklevel=2,
            )

    def _flush_ascii_data(
        self,
        data_lines: list[str],
        section_curve_start_idx: int,
        section_curve_end_idx: int | None,
        current_section_name: str,
        current_data_section_type: str,
        current_data_section_idx: int,
        cumulative_elements: int,
        version_found: bool,
        las_file: LASFile | None = None,
    ) -> None:
        """Flush accumulated ASCII data lines into structured data arrays.

        Shared helper used by both the unknown-section handler and
        ``_SectionTransitionHandler._process_ascii_section`` to eliminate
        duplicated ``AsciiDataContext`` construction and ``finally`` cleanup
        logic (F-30).

        Args:
            data_lines: Accumulated ASCII data lines for the current section.
            section_curve_start_idx: Start index into ``las_file.curves``.
            section_curve_end_idx: End index (exclusive), or ``None``.
            current_section_name: Human-readable section name.
            current_data_section_type: LAS 3.0 data section type string.
            current_data_section_idx: Zero-based data section counter.
            cumulative_elements: Running total of elements across sections.
            version_found: Whether ``~V`` has been parsed (controls deferred
                well replay).
            las_file: The ``LASFile`` to write into.  Defaults to
                ``self.las_file``.
        """
        if not data_lines:
            return
        if version_found:
            # I2-08: When a data section is flushed BEFORE ~C is parsed
            # (curves still empty), replaying deferred pre-~V data now and
            # processing this section now both hit the "no curves defined"
            # discard in process_ascii_data (_las30_data.py:653-662) and
            # silently lose data — for the mixed layout ``~A(data1) ~V
            # ~A(data2) ~C`` BOTH blocks were discarded (ADV-M1 repro).
            # Re-queue this section's lines into the deferred buffer so the
            # final replay (parse() → _replay_deferred_well after ALL
            # sections, including ~C, are parsed) processes them against the
            # now-known curve block.  Give the re-queued section a FRESH
            # section_idx so groupby keeps it separate from earlier pre-~V
            # deferred sections.
            # PARS-05: The guard fires whenever curves are empty — NOT just
            # when pre-existing deferred lines exist.  A lone non-deferred
            # data section before ~C (e.g. ``~V ~A(data) ~CURVE``) previously
            # hit the "data before definition is discarded" path (F-013
            # ordering test documented that discard as expected); the
            # "data-before-curves supported" claim (M-67/M-69) held only via
            # deferral.  Re-queueing in the no-deferred-lines case makes the
            # non-deferred path behave like the deferred path: the data is
            # buffered and attached once curves parse.
            if not self.las_file.curves:
                if (
                    len(self._state.deferred_ascii_data_lines) + len(data_lines)
                    > _data_reader.MAX_DATA_LINES
                ):
                    raise LASParseError(
                        f"Deferred ASCII data line count exceeds maximum "
                        f"allowed ({_data_reader.MAX_DATA_LINES}). "
                        f"The file may be malformed or corrupt."
                    )
                self._state.current_data_section_idx += 1
                _defer_idx = self._state.current_data_section_idx
                _deferred_curve_end = section_curve_end_idx
                if self._current_pipe_target in {"CURVE", "C"} and _deferred_curve_end is None:
                    _deferred_curve_end = _DEFERRED_MAIN_CURVE_SCOPE
                elif (
                    self._current_pipe_target
                    and self._current_pipe_target not in {"CURVE", "C"}
                    and self._current_pipe_target not in self._state.definition_curve_ranges
                ):
                    # PARS-06: The re-queued section carries a forward
                    # "| X_Definition" pipe whose target definition is not
                    # yet parsed.  Record the target keyed by the re-queued
                    # group so replay can resolve the scope once the
                    # definition exists.
                    _deferred_curve_end = _DEFERRED_PIPE_SCOPE
                    self._deferred_pipe_targets[
                        (
                            current_data_section_type,
                            current_section_name,
                            _defer_idx,
                        )
                    ] = self._current_pipe_target
                for _ln in data_lines:
                    self._state.deferred_ascii_data_lines.append(
                        (
                            current_data_section_type,
                            current_section_name,
                            _defer_idx,
                            _ln,
                            section_curve_start_idx,
                            _deferred_curve_end,
                        )
                    )
                self._state.ascii_data_lines = []
                return
            self._replay_deferred_well()
            # F-M2: _replay_deferred_well() mutates self._state.current_data_section_idx
            # and self._state.cumulative_elements.  The parameters were evaluated at the
            # call site BEFORE replay ran, so they are stale.  Read the live post-replay
            # values from self._state to construct AsciiDataContext correctly.
            current_data_section_idx = self._state.current_data_section_idx
            cumulative_elements = self._state.cumulative_elements
        _las = las_file if las_file is not None else self.las_file
        try:
            ctx = AsciiDataContext(
                las_file=_las,
                ascii_data_lines=data_lines,
                section_curve_start_idx=section_curve_start_idx,
                section_curve_end_idx=section_curve_end_idx,
                current_section_name=current_section_name,
                current_data_section_type=current_data_section_type,
                current_data_section_idx=current_data_section_idx,
                cumulative_elements=cumulative_elements,
            )
            process_ascii_data(ctx)
            self._state.cumulative_elements = ctx.cumulative_elements
        finally:
            self._state.ascii_data_lines = []
            self._state.current_data_section_idx += 1

    def _validate_cross_section_consistency(self) -> None:
        """Validate cross-section consistency (F-34).

        Four dimensions checked:
        (1) Curve count vs data column count for each data section.
        (2) LAS 3.0 section ordering — data sections before curve
            definitions have no curves to reference.
        (3) Duplicate section headers.
        (4) Per-section data_format x placement validation (F-28).
        """
        # (1) Curve count vs data column count per data section.
        # F-27: Extracted into a reusable module-level function so the
        # from_dict path (models.py) can also call it.
        _validate_data_section_column_counts(self.las_file.data_sections)

        # (4) F-28: Per-section data_format x placement check. Every
        # S-format (or non-array A-format) curve must be in string_data,
        # not data; every numeric-format curve must be in data, not
        # string_data.  This mirrors DataSection.__post_init__ (which runs
        # during construction) and LASFile.validate() (which only checks
        # top-level curves).
        for ds in self.las_file.data_sections:
            _data_keys = set(ds.data.keys())
            _str_keys = set(ds.string_data.keys())
            for _sc in ds.section_curves:
                _df = _sc.data_format
                _mnem = _sc.mnemonic
                if not _df:
                    continue
                if _mnem in _data_keys and (
                    _df == "S" or (_df == "A" and not _sc.is_array_element)
                ):
                    logger.warning(
                        "Data section '%s': curve '%s' has data_format='%s' "
                        "(string-format) but is in data (numeric). "
                        "String-format curves should be in string_data.",
                        ds.name,
                        _mnem,
                        _df,
                    )
                elif _df not in ("S", "A") and _mnem in _str_keys:
                    logger.warning(
                        "Data section '%s': curve '%s' has data_format='%s' "
                        "(numeric-format) but is in string_data. "
                        "Numeric-format curves should be in data.",
                        ds.name,
                        _mnem,
                        _df,
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

            def _def_type_display(_def_type: str) -> str:
                if _def_type == "__MAIN__":
                    return "the main curve definition (~C or ~CURVE)"
                return f"~{_def_type}"

            for _seq_idx, label in enumerate(self._state.section_sequence):
                section_word = label.split(":")[0]
                is_data = (
                    section_word in _DATA_SECTION_WORDS
                    or section_word.endswith("_DATA")
                    or _is_indexed_data_section(section_word)
                )
                is_curve = section_word in {"C", "CURVE"} or section_word.endswith("_DEFINITION")
                # F-004: Identify parameter sections (both LAS 1.2/2.0 ~P
                # and LAS 3.0 type-prefixed ~Core_Parameter etc.)
                is_param = (
                    section_word == "P"
                    or section_word.endswith("_PARAMETER")
                    or section_word.endswith("_PARAMETERS")
                )

                if is_data:
                    # PARS-02: Resolve the section's pipe target (recorded
                    # parallel to section_sequence in _parse_line).  Dispatch
                    # (parser.py:1322-1331) resolves "| CURVE"/"| C" to the
                    # main curve block (__MAIN__), and "| X_Definition" to
                    # that definition's range.  The checker must mirror this:
                    # a canonical ``~ASCII | CURVE`` data section maps to
                    # __MAIN__ — NOT to LOG_DEFINITION derived from the bare
                    # section word — or it fires spurious "~ASCII before
                    # ~LOG_DEFINITION" / "main curve definition has no
                    # corresponding data section" warnings on valid files.
                    _pipe_target = None
                    _pipe_targets = getattr(self, "_section_pipe_targets", None)
                    if _pipe_targets is not None and _seq_idx < len(_pipe_targets):
                        _pipe_target = _pipe_targets[_seq_idx]

                    # Normalize indexed section words (e.g., "CORE[1]" → "CORE")
                    # before resolving definition type.  _SECTION_TYPE_MAP only
                    # contains unindexed keys, and the endswith("_DATA") check
                    # cannot match bracketed forms.
                    _type_word = section_word
                    if _is_indexed_data_section(section_word):
                        _type_word = section_word[: section_word.find("[")].upper()

                    # Determine expected definition type for this data section.
                    if _pipe_target in {"CURVE", "C"}:
                        # PARS-02: explicit pipe to the main curve block.
                        _def_type = "__MAIN__"
                    elif (
                        _pipe_target is not None
                        and _pipe_target in self._state.definition_curve_ranges
                    ):
                        # PARS-02: explicit pipe to a known _Definition.
                        _def_type = _pipe_target
                    elif _type_word.endswith("_DATA"):
                        _def_type = _type_word.replace("_DATA", "_DEFINITION")
                    elif _type_word in _SECTION_TYPE_MAP:
                        _canonical = _SECTION_TYPE_MAP[_type_word]
                        if _canonical.endswith("_DATA"):
                            _def_type = _canonical.replace("_DATA", "_DEFINITION")
                        else:
                            _def_type = "__MAIN__"
                    else:
                        _def_type = "__MAIN__"

                    # F-20: Mirror the resolver's __MAIN__ fallback for
                    # BARE (no-pipe) sections.  The resolver (parser.py
                    # :1579-1591) resolves a bare ~A/~ASCII to
                    # LOG_DEFINITION only when that _Definition exists;
                    # otherwise it falls back to __MAIN__ (the H-01 fix).
                    # The checker previously derived LOG_DEFINITION from
                    # _SECTION_TYPE_MAP and fired two spurious warnings on
                    # every valid bare-~A + ~C file ("~A before
                    # ~LOG_DEFINITION" + "main curve definition has no
                    # corresponding data section") — the same bare section
                    # the resolver had already scoped to __MAIN__.  Only
                    # flag data-before-definition when NEITHER the derived
                    # definition NOR the main block is available yet (the
                    # resolver's unbounded fallback).
                    if _def_type not in _defs_seen and _def_type != "__MAIN__":
                        if "__MAIN__" in _defs_seen:
                            _def_type = "__MAIN__"
                        else:
                            per_type_data_before_def.append(
                                f"~{section_word} before {_def_type_display(_def_type)}"
                            )
                    elif _def_type not in _defs_seen:
                        per_type_data_before_def.append(
                            f"~{section_word} before {_def_type_display(_def_type)}"
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
                        per_type_param_after_def.append(f"~{section_word} after ~{_def_type}")

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
        for stype in self._state.section_type_sequence:
            _type_counts[stype] = _type_counts.get(stype, 0) + 1

        for stype, count in _type_counts.items():
            if count > 1 and stype in _reserved_single:
                _duplicates = [
                    lbl
                    for lbl, st in zip(
                        self._state.section_sequence, self._state.section_type_sequence, strict=True
                    )
                    if st == stype
                ]
                _dup_labels = ", ".join(f"'~{d}'" for d in _duplicates)
                # I2-06: For duplicate ~A data sections in LAS 1.2/2.0, the
                # data reader ingests ONLY the FIRST contiguous ~A block;
                # later ~A blocks after a non-~A section are silently DROPPED
                # (first-block-only semantics, F-EX-02).  The old generic
                # logger.warning below (a) is invisible to the warnings-API
                # (catch_warnings captures nothing) and (b) misstates the
                # direction — it says "data from earlier instances to be
                # overwritten", but the EARLIER block is preserved and the
                # LATER is dropped.  Emit a warnings-API warning with the
                # corrected direction.  LAS 3.0 is unaffected: multiple ~A
                # blocks are valid data sections there (A not in
                # _reserved_single for is_las30).
                if stype == "A" and not _is_las30:
                    warnings.warn(
                        f"Multiple ~A data sections encountered "
                        f"{count} times: {_dup_labels}. "
                        f"Only the FIRST ~A block's data is ingested; "
                        f"data in later ~A blocks after a non-~A section "
                        f"is DROPPED.",
                        UserWarning,
                        stacklevel=2,
                    )
                logger.warning(
                    "Duplicate reserved section type '~%s' encountered "
                    "%d times: %s. Repeated reserved sections may "
                    "indicate a malformed file or cause data from earlier "
                    "instances to be overwritten.",
                    stype,
                    count,
                    _dup_labels,
                )
