"""Shared sanitize/desanitize helpers — leaf module (stdlib-only).

Single source of truth for the write-side LAS escapes and their read-side
inverses, plus the thread-local ``_DESANITIZE_ENABLED`` storage:

- Write side (moved from ``_writer_base.py``): ``_sanitize_las_value``,
  ``_escape_colons_for_las_value``, ``_escape_pipes_for_las_value`` and
  the regexes they use.
- Read side (moved from ``parser.py`` / ``data_reader.py``):
  ``desanitize_las_value`` (the unified ``_#`` / ``_~`` restore) and
  ``desanitize_other_line`` (the ``~O``-scoped ``_#``-only restore), plus
  ``_unescape_colons_for_las_value`` / ``_unescape_pipes_for_las_value``.
- Thread-local flag: ``_is_desanitize_enabled`` / ``_set_desanitize_enabled``
  backed by ``threading.local`` storage, with the ``_DesanitizeModule``
  module-class proxy installed on this module so ``_DESANITIZE_ENABLED``
  attribute reads/writes route to per-thread storage.

This module imports only the standard library, so every consumer (parser,
data_reader, _writer_base, _las30_data, reader, version writers) can import
it without circular-import risk.  ``parser`` keeps a thin delegating shim
for ``_DESANITIZE_ENABLED`` so existing test references
(``pylasdev.parser._DESANITIZE_ENABLED``) keep working (II-9).
"""

from __future__ import annotations

import re
import sys
import threading
import types

# ── Module-level constants & compiled regexes ────────────────────────────

# Control characters except space and tab (which are valid LAS whitespace).
# Tab (\x09) is handled separately in _sanitize_las_value — it is replaced
# with a space to prevent mis-tokenization on re-read.  A tab inside an
# identifier acts as a field separator for str.split(), corrupting the
# parsed structure.
_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x85"
    r"\u2028\u2029]"
)

# Unicode whitespace characters that should be replaced with an ASCII
# space, not silently deleted.  These are layout/presentation characters
# (non-breaking space, en/em quads, thin spaces, ideographic space) that
# act as visual word separators.
_UNICODE_WS_RE = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")

# Previous pattern ^~([A-Za-z]) only matched a leading tilde.
# Values like "\t~Version" or "  ~Curve" bypassed section-header
# sanitization because leading whitespace prevented the regex match.
_LEADING_SECTION_RE = re.compile(r"^\s*~([A-Za-z])")

# Pattern matching whitespace-before-colon (\s+:).
_COLON_PRECEDED_BY_WS_RE = re.compile(r"(\s+):")

# Pattern matching colon-followed-by-whitespace-or-end (:\s|\s*$).
_COLON_FOLLOWED_BY_WS_OR_END_RE = re.compile(r":(?=\s|$)")


# ── Write-side escapes (moved from _writer_base.py) ─────────────────────


def _sanitize_las_value(value: str, *, preserve_leading_tilde: bool = False) -> str:
    """Sanitize a string for safe inclusion in LAS output.

    Args:
        preserve_leading_tilde: If True, a leading ``~`` (and any preceding
            whitespace) is NOT stripped.  The default strips a line-start
            ``~[A-Za-z]`` pattern so a value never mimics a LAS section
            header (``~CURVE``, ``~WELL``...).  That strip is only required
            for text emitted at the START of an output line.  Values emitted
            mid-line (well values, parameter values, descriptions,
            non-first-column data cells) can never be confused with a section
            header, and stripping them silently corrupts the model value on
            write→read (M-28).  Pass True for such mid-line content so the
            value survives roundtrip unchanged.
    """
    value = (
        value.replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\u2028", " ")
        .replace("\u2029", " ")
        .replace("\x85", " ")
    )
    value = value.replace("\t", " ")
    value = _CONTROL_CHARS_RE.sub("", value)
    value = _UNICODE_WS_RE.sub(" ", value)
    if not preserve_leading_tilde:
        value = _LEADING_SECTION_RE.sub(r"\1", value, count=1)
    if value.startswith("#"):
        value = "_" + value
    elif value and value.lstrip().startswith("#"):
        stripped = value.lstrip()
        leading = value[: len(value) - len(stripped)]
        value = leading + "_" + stripped
    return value


def _escape_colons_for_las_value(value: str) -> str:
    """Escape colons in a LAS value to prevent parser misinterpretation."""
    value = _COLON_PRECEDED_BY_WS_RE.sub(r"\1_:", value)
    value = _COLON_FOLLOWED_BY_WS_OR_END_RE.sub(r"\g<0>_", value)
    return value


def _escape_pipes_for_las_value(value: str) -> str:
    """Escape literal pipes in a LAS description (``|`` → ``\\|``).

    N-I-02: The parser treats a pipe at the END of a parameter description
    as a LAS 3.0 zone association (``| Zone``) and strips it.  Genuine
    description text that happens to contain a pipe would therefore be
    truncated and misinterpreted on re-read.  Escaping literal pipes keeps
    them out of ZONE_ASSOC_PATTERN's reach while real zone associations
    (appended separately by the writer, unescaped) still round-trip.
    The parser reverses this with ``_unescape_pipes_for_las_value``.
    """
    return value.replace("|", "\\|")


# ── Read-side inverses (moved from parser.py) ───────────────────────────


def _unescape_colons_for_las_value(value: str) -> str:
    """Reverse the ``_escape_colons_for_las_value`` transformation.

    The writer applies a two-step colon escape to prevent the parser from
    misinterpreting embedded colons as structural separators:

    1. Insert ``_`` between whitespace and colon: ``" :"`` → ``" _:"``
    2. Insert ``_`` after colon followed by whitespace or end: ``": "`` → ``":_ "``

    The combined effect on ``" : "`` produces ``" _:_ "``.

    This function reverses both steps in the opposite order (step 2 first,
    then step 1), restoring the original colon-separated text.

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
       ``\\|`` in a description) will be incorrectly unescaped.
    """
    return value.replace("\\|", "|")


def desanitize_las_value(
    value: str,
    *,
    restore_tilde: bool = False,
    _enabled: bool | None = None,
) -> str:
    """Reverse the writer's ``_``-prefix escapes on a LAS value.

    The writer prefixes ``#``-starting values with ``_`` (``_sanitize_las_value``)
    and escapes a FIRST-column string value starting ``~``+non-letter as ``_~``
    (M-85) so emitted data rows never begin with ``#``/``~``.  This function
    strips those prefixes, restoring the original value.

    Two ``_#`` cases (matching writer's ``_sanitize_las_value``):

    1. ``value.startswith("#")`` → writer prepends ``_`` → ``"_#..."``
       → reverse: strip the leading ``_``.
    2. ``value.lstrip().startswith("#")`` → writer inserts ``_`` after
       leading whitespace → ``" _#..."`` → reverse: remove the ``_``
       between whitespace and ``#``.

    F-25 (M11): Case 2 applies ONLY when the ``_#`` is the first
    non-whitespace content (preceded exclusively by leading whitespace) —
    the writer's actual escape scope.  Internal ``" _#"`` content the
    writer never escapes (e.g. ``"ACME _#Oil Corp"``) is preserved
    unchanged.

    ``_~`` restore (M-85) is gated on *restore_tilde*: the LAS 3.0 data
    path passes ``True`` (its writer emits ``_~``); the LAS 1.2/2.0 data
    path and all header call sites pass the default ``False`` (ADV-M3
    adjudication — the 1.2/2.0 writer never emits ``_~``, so a genuine
    external ``_~`` value must be preserved; II-13).

    Args:
        value: The raw value to desanitize.
        restore_tilde: When True, restore ``_~`` → ``~`` (LAS 3.0 data
            path only; default False is fail-safe).
        _enabled: Hoisted ``_DESANITIZE_ENABLED`` flag (IT3-F-01 perf —
            cached once per read).  None → look up the thread-local flag.
    """
    if _enabled is None:
        _enabled = _is_desanitize_enabled()
    if not _enabled:
        return value
    if value.startswith("_#"):
        return value[1:]
    # M-85: restore the leading '~' escaped by the writer for a first-column
    # string value starting with '~'+non-letter (or bare '~').
    if restore_tilde and value.startswith("_~"):
        return value[1:]
    # Case 2: whitespace-prefixed value with sanitized _# (e.g., " _#comment").
    # F-25: Restrict to the writer's ACTUAL escape scope — only an "_#"
    # preceded exclusively by leading whitespace (the FIRST non-whitespace
    # content) is a writer escape artifact; mid-value "_#" is preserved.
    stripped = value.lstrip()
    if len(stripped) < len(value) and stripped.startswith("_#"):
        leading = value[: len(value) - len(stripped)]
        return leading + stripped[1:]
    return value


def desanitize_other_line(line: str, *, _enabled: bool | None = None) -> str:
    """Scoped W-08 restore for ~O (other) lines — reverse ONLY the escapes
    the ~O writer actually emits.

    PF-02 (regression fix): ``_parse_other`` previously ran every ~O line
    through the blanket ``desanitize_las_value``, which also reversed the
    data-row ``_~`` escape and ANY whitespace-adjacent ``_#`` — escapes the
    ~O writer (``_sanitize_las_value``) NEVER emits.  Genuine ``_~``-prefixed
    lines and mid-line ``_#`` content were silently altered on write→read.

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

    Args:
        line: The ~O line to desanitize.
        _enabled: Hoisted ``_DESANITIZE_ENABLED`` flag; None → thread-local.
    """
    if _enabled is None:
        _enabled = _is_desanitize_enabled()
    if not _enabled:
        return line
    if line.startswith("_#"):
        return line[1:]
    stripped = line.lstrip()
    if len(stripped) < len(line) and stripped.startswith("_#"):
        leading = line[: len(line) - len(stripped)]
        return leading + stripped[1:]
    return line


# ── Thread-local _DESANITIZE_ENABLED storage (moved from parser.py) ──────

# F-212: When reading files NOT produced by pylasdev's writer, the
# desanitize_las_value transformation should not be applied — _# in
# external data is genuine content, not a writer escape.  Defaults to
# True (preserves existing roundtrip behavior).  Set to False before
# reading external files to prevent data corruption.
# F-21: Thread-local storage to prevent race conditions when concurrent
# callers use different desanitize values.  Module-level attribute
# setting (e.g. data_reader's ``_DESANITIZE_ENABLED = desanitize``)
# is intercepted via ``_DesanitizeModule.__setattr__`` and routed to
# per-thread storage.  Truthiness checks via ``_DesanitizeModule.__getattr__``
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
    in parser.py and data_reader, all without modifying those modules.
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
