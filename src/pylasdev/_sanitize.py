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
- Content normalization shared by EVERY entry point (LD-08):
  ``normalize_nbsp_family`` — the one non-breaking-space-family rule the
  writer, ``parser``, ``reader`` and ``dev_reader`` all apply.  The rule is
  position-based, so it returns the same answer for the same input string
  wherever it is called; the entry points differ only in WHEN it runs
  relative to their own C0-control deletion (L16 — see
  ``normalize_nbsp_family``'s docstring for the exact invariant and the one
  input class where the orders disagree).

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
import warnings

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
# (en/em quads, thin spaces, ideographic space) that act as visual word
# separators.  The non-breaking space family inside this class is
# normalized FIRST by normalize_nbsp_family (see below), which deletes the
# one position where a space is wrong; the family members are still listed
# here so this class remains self-describing on its own.
_UNICODE_WS_RE = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")

# ── NBSP-family normalization (LD-08: ONE rule for every entry point) ────
#
# The non-breaking space family — NBSP U+00A0, THIN SPACE U+2009, NARROW
# NBSP U+202F — is the set of code points a locale thousands group
# separator is written as.  They are NOT plain whitespace, and neither
# blanket treatment is correct:
#   * a space inside a numeric token splits one value into two columns and
#     shifts every later value on the row ('1\xa0000.5' read as 1.0 plus a
#     discarded '000.5') — F-225;
#   * a delete everywhere fuses two genuinely separate values into an
#     unconvertible one ('1.0\xa02.0' -> '1.02.0') — F-231.
# The POSITION decides — but the position alone is not enough.  A thousands
# separator can only PRODUCE a three-digit group; the converse does not
# hold, so "digit, family character, three digits" also matches a genuine
# column separator whose right-hand value starts with three digits
# ('1000\xa0500.0' is 1000 and 500.0, not 1000500.0 — M9).  The rule
# therefore also requires the digits on the LEFT to look like the start of
# one numeric token (a 1-3 digit group with no digit, word character or
# decimal point immediately before it), which is the only left-hand shape a
# thousands separator can have: 1-3 digits, then a group of exactly three.
#
# Thousands grouping separator: a family character INSIDE one numeric token
# — preceded by a token-leading 1-3 digit group and followed by exactly
# three digits, the separator+group unit repeating while every group is
# complete.  ASCII digits only — LAS/DEV numerics are ASCII, and a Unicode
# Nd neighbour (e.g. Arabic-Indic) is not a thousands group in this format.
# The trailing three digits (and the optional decimal part / exponent that
# follows them) are CONSUMED and re-emitted unchanged, so the fused token can
# be rebuilt from the capture groups when the deletion is reported (M9(b)).
# The two lookbehinds pin the LEFT side to a token start: `[\w.]` rejects a
# longer digit run ('1000\xa0500.0') and a non-numeric left token
# ('A1\xa0000.5'); `[eE][+-]` rejects a complete exponent-terminated token
# ('1E-3\xa0000.5' — a genuine separator), while a sign that starts the
# token ('-1\xa0000.5') still matches.
_THOUSANDS_GROUP_SEP_RE = re.compile(
    r"(?<![\w.])"
    r"(?<![eE][+-])"
    r"([0-9]{1,3})"
    r"((?:[\u00A0\u2009\u202F][0-9]{3})+)"
    r"((?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    r"(?![0-9])"
)
# Any remaining family character is plain whitespace and becomes one space.
_NBSP_FAMILY_RE = re.compile(r"[\u00A0\u2009\u202F]")


def normalize_nbsp_family(
    value: str,
    *,
    warn_context: str | None = None,
    stacklevel: int = 2,
) -> str:
    """Normalize the non-breaking space family (NBSP / thin / narrow NBSP).

    ONE rule for every entry point: the writer (``_sanitize_las_value``),
    ``parser.LASParser.parse``, ``reader._sanitize_control_chars`` and
    ``dev_reader.read_dev_file_as_object`` all call this, so the same input
    string normalizes the same way no matter which path reads it (LD-08).

    * **thousands grouping separator** — a family character INTERNAL to one
      numeric token: preceded by a token-leading group of 1-3 digits (no
      digit, word character or decimal point immediately before it) and
      followed by exactly three digits, the separator+three-digits unit
      repeating while each group is complete — is DELETED:
      ``'1\\xa0000.5'`` -> ``'1000.5'``,
      ``'12\\xa0345\\xa0678.9'`` -> ``'12345678.9'``.  Substituting a space
      there splits one numeric value into two columns and shifts the rest
      of the row (F-225).
    * **whitespace** — every other occurrence is replaced with one
      ordinary space: ``'1.0\\xa02.0'`` -> ``'1.0 2.0'``,
      ``'ACME\\xa0BETA'`` -> ``'ACME BETA'``.  Deleting it would fuse two
      genuinely separate values into an unconvertible one (F-231).

    The delete side is deliberately narrow, in two directions (M9):

    * the LEFT value must be a 1-3 digit token-leading group, because a
      thousands separator can only sit inside ``d{1,3} <sep> d{3}``.  A
      4+-digit left value (``'1000\\xa0500.0'``) is a genuine column
      separator between two complete numeric values and keeps the space
      treatment, as do a non-numeric left token (``'A1\\xa0000.5'``) and a
      complete exponent-terminated token (``'1E-3\\xa0000.5'``);
    * the trailing group must be complete: a two-digit group
      (``'1000\\xa000.5'``) is NOT the unambiguous thousands shape, so it
      also keeps the space treatment rather than being guessed at.

    A deletion is never silent: the caller supplies ``warn_context`` and
    the counted ``UserWarning`` names every fused token it produced, so the
    operator can see which value was joined.  DEV, the LAS reader and the
    writer all pass a context; ``None`` is kept only as the default for a
    caller that has nothing to name.

    Invariant scope (LD-08 / L16): this rule is a pure function of the
    string it is handed — for the SAME input string every caller gets the
    same output.  It is NOT a claim that the callers hand it the same
    string: the LAS reader and the parser normalize the family BEFORE
    deleting C0 controls, while DEV and the writer delete the controls
    first.  A C0 control sitting inside a grouping shape
    (``'1\\x00\\xa0000.5'``) therefore reaches the rule differently and the
    two orders disagree — LAS ``'1 000.5'`` (two tokens), DEV ``'1000.5'``
    (one token).  Both outcomes are loud (the control-deletion warning
    fires on both paths); the divergence is documented rather than unified
    because the order belongs to each reader's own control-character
    policy (``reader._DELETE_CONTROL_CHARS_RE`` vs F-083's DEV deletion).

    No other character class is touched here — true line separators, the
    other Unicode spaces and the C0 controls are handled by the callers'
    existing classes (``reader._SPACE_CHARS_RE`` /
    ``parser._SPLITLINES_CHARS_RE`` / ``dev_reader._DEV_SPACE_CHARS_RE`` and
    the delete classes beside them).

    Args:
        value: Text to normalize.
        warn_context: When given and at least one grouping separator was
            removed, emit ONE counted ``UserWarning`` naming this context
            and the fused token(s).  All in-tree callers pass one: the DEV
            reader and the LAS reader pass their file path, the writer
            names the value's origin.  ``None`` stays silent and exists only
            for a caller with nothing to name.
        stacklevel: ``stacklevel`` forwarded to :func:`warnings.warn` so a
            warning still points at the caller of the site function.

    Returns:
        The normalized string — identical output for an identical input
        string regardless of which entry point calls this.
    """
    fused: list[str] = []
    n_grouped = 0

    def _join_groups(match: re.Match[str]) -> str:
        """Delete the separators of one match and return the joined token."""
        nonlocal n_grouped
        n_grouped += len(_NBSP_FAMILY_RE.findall(match.group(2)))
        # group(3) is the token's decimal part / exponent — re-emitted
        # unchanged, so the replacement is exactly the matched text minus the
        # separators, and the same string names the fused token in the
        # warning (M9(b)).
        joined = match.group(1) + _NBSP_FAMILY_RE.sub("", match.group(2)) + match.group(3)
        fused.append(joined)
        return joined

    value = _THOUSANDS_GROUP_SEP_RE.sub(_join_groups, value)
    if n_grouped and warn_context is not None:
        named = ", ".join(repr(token) for token in fused[:3])
        if len(fused) > 3:
            named += f", and {len(fused) - 3} more"
        warnings.warn(
            f"Removed {n_grouped} non-breaking space(s) used as thousands "
            f"padding inside numeric token(s) {named} in {warn_context}; the "
            f"digits were joined.  A non-breaking space between two complete "
            f"numeric values is never deleted — it is read as a token "
            f"separator.  Verify the file is not corrupt.",
            UserWarning,
            stacklevel=stacklevel,
        )
    return _NBSP_FAMILY_RE.sub(" ", value)


# Previous pattern ^~([A-Za-z]) only matched a leading tilde.
# Values like "\t~Version" or "  ~Curve" bypassed section-header
# sanitization because leading whitespace prevented the regex match.
_LEADING_SECTION_RE = re.compile(r"^\s*~([A-Za-z])")

# Pattern matching the colon-escape insertion point: the zero-width position
# between a whitespace character and a following colon.
# F-016: the previous form `(\s+):` is quadratic on a whitespace run that is
# NOT followed by a colon — `\s+` consumes the whole run, `:` fails, `\s+`
# gives back one character and the engine re-scans, Σ O(k) per start position
# → O(k²).  Measured 4 times slower per doubling on the write path (32 KB run
# → 2.59 s).
# The read-side twin was already made linear under M-61
# (_unescape_colons_for_las_value); this is its write-side twin.  The
# zero-width form inserts the `_` at exactly the position `(\s+):` →
# `\1_:` did (between the last whitespace character and the colon) with no
# quantifier to backtrack, so the emitted bytes are unchanged.
_COLON_PRECEDED_BY_WS_RE = re.compile(r"(?<=\s)(?=:)")

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

    Warns:
        UserWarning: When an NBSP-family character used as thousands padding
            is deleted inside a numeric token (M9).  The warning is counted
            and names the fused token, so a written value the sanitizer
            altered is never silent.
        UserWarning: When the value's first non-whitespace content is
            already ``_#`` (RE-7) — the read-side restore cannot tell it
            from the writer's escape of a ``#``-leading value, so the
            underscore is lost on write→read.  The emitted bytes are the
            pinned wire form and do not change; only the loss is reported.
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
    # LD-08: the write path applies the SAME NBSP-family rule as the read
    # side.  Without it, the space substitution below destroys the
    # thousands shape on the wire ('1\xa0000.5' was written '1 000.5'), so
    # no read-side rule could ever rejoin the value — the write itself was
    # the corruption point.  M9(b): the deletion is NOT silent.  This is the
    # only site that knows which value the fused token came from, so it
    # passes a context and the warning names the token; an identical token
    # in a later field is deduplicated by Python's default warning filter.
    value = normalize_nbsp_family(value, warn_context="a written LAS value")
    value = _UNICODE_WS_RE.sub(" ", value)
    if not preserve_leading_tilde:
        value = _LEADING_SECTION_RE.sub(r"\1", value, count=1)
    # RE-7: the writer's '_#' escape below is not injective.  A value whose
    # first non-whitespace content is ALREADY '_#' is byte-identical to the
    # escape this function emits for a '#'-leading value, so the read-side
    # desanitize_las_value strips the underscore on re-read.  The escape
    # form is the documented wire convention and is NOT changed here
    # (tests pin the emitted bytes); like the F-018 colon family
    # (_escape_colons_for_las_value) the loss is computed from the ACTUAL
    # inverse and reported as a counted UserWarning naming the value class.
    _starts_with_escaped_hash = value.lstrip().startswith("_#")
    if value.startswith("#"):
        value = "_" + value
    elif value and value.lstrip().startswith("#"):
        stripped = value.lstrip()
        leading = value[: len(value) - len(stripped)]
        value = leading + "_" + stripped
    if _starts_with_escaped_hash:
        restored = desanitize_las_value(value, _enabled=True)
        if restored != value:
            _lost_underscores = len(value) - len(restored)
            warnings.warn(
                f"LAS '_#' escape is not invertible for this value class: "
                f"the value's first non-whitespace content is already '_#', "
                f"which is indistinguishable from the writer's escape of a "
                f"'#'-leading value.  Re-reading the file will drop "
                f"{max(_lost_underscores, 1)} literal underscore "
                f"character(s): {value!r} -> {restored!r}",
                UserWarning,
                stacklevel=2,
            )
    return value


def _escape_colons_for_las_value(value: str) -> str:
    """Escape colons in a LAS value to prevent parser misinterpretation.

    F-018: the escape is not injective.  A *genuine* literal ``_`` that
    already sits in one of the two escape positions (``_`` after whitespace
    and before a colon, or ``_`` after a colon and before whitespace/end) is
    byte-identical to the ``_`` this function inserts, so
    :func:`_unescape_colons_for_las_value` removes it on re-read — a lossy
    shared-layer transform that used to mutate a field with no diagnostic at
    all (``"tag _: value"`` → written ``"tag _:_ value"`` → read back
    ``"tag : value"``).

    The escape form itself is unchanged: it is the documented wire
    convention, the read-side inverse matches it position-for-position, and
    the emitted bytes are pinned (``tests/test_writer.py:1797-1826``).
    What changed is the silence: the inverse of the produced value is
    computed and, when it does not restore the input, a **counted**
    ``UserWarning`` names the value class (a literal underscore in an
    escape-artifact position) and the number of characters that will be
    lost.  See :func:`_unescape_colons_for_las_value` for the documented
    trade.
    """
    escaped = _COLON_PRECEDED_BY_WS_RE.sub("_", value)
    escaped = _COLON_FOLLOWED_BY_WS_OR_END_RE.sub(r"\g<0>_", escaped)
    if "_" in value:
        # Only values carrying an underscore can hit the non-injective
        # class; the check is the ACTUAL inverse, not a re-guessed pattern.
        restored = _unescape_colons_for_las_value(escaped, _enabled=True)
        if restored != value:
            _lost = len(value) - len(restored)
            warnings.warn(
                f"LAS colon escape is not invertible for this value class: "
                f"the value contains a literal '_' in an escape position "
                f"(after whitespace before a ':', or after a ':' before "
                f"whitespace/end), which is indistinguishable from the "
                f"writer's inserted escape '_'.  Re-reading the file will "
                f"drop {max(_lost, 1)} literal underscore character(s): "
                f"{value!r} -> {restored!r}",
                UserWarning,
                stacklevel=3,
            )
    return escaped


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


def _escape_braces_for_las_value(value: str) -> str:
    """Escape literal braces in a LAS description (``{`` → ``\\{``, ``}`` → ``\\}``).

    N-09: the parser treats a braced token at the end of a curve/parameter
    description as a LAS data-format specifier (``{S}``, ``{F}``,
    ``{A:N}``) and strips it, fabricating a ``data_format`` from user
    text.  A user description containing a valid format token in braces
    (e.g. ``"Gamma {S} ray"``) was silently destroyed on write→read, and
    the trailing token was re-parsed as the curve's format — re-routing a
    numeric column to string_data.  Escaping literal braces keeps them
    out of FORMAT_SPEC_PATTERN's reach; the writer's OWN appended format
    token (``desc  {S}``) is appended AFTER the escape and stays
    unescaped so the parser still recognizes it.  The parser reverses
    this with ``_unescape_braces_for_las_value`` at the format-strip
    sites (parser.py:3355/3610).  Applied on ALL versions so the
    non-3.0 writers' ``{I}`` marker handling is symmetric.
    """
    return value.replace("{", "\\{").replace("}", "\\}")


def _unescape_braces_for_las_value(value: str, *, _enabled: bool | None = None) -> str:
    """Reverse the ``_escape_braces_for_las_value`` transformation.

    Restores ``\\{`` → ``{`` and ``\\}`` → ``}`` in a description after the
    parser has stripped the writer's appended (unescaped) format token.
    Mirrors the pipe-escape trade (``_unescape_pipes_for_las_value``): a
    genuine external ``\\{`` sequence in a description is restored
    unconditionally.

    F-019: ``_enabled`` gates the whole transform on the ``desanitize`` flag
    (thread-local ``_DESANITIZE_ENABLED`` when None), like
    :func:`desanitize_las_value`.  ``desanitize=False`` means "this file
    was NOT produced by pylasdev's writer, do not rewrite its content", so
    every escape family must honour it — previously only the ``_#``/``_~``
    families did, and colon/pipe/brace un-escaping still rewrote external
    text with no opt-out.

    .. note::

       Legitimate backslash-brace text in original data (e.g. a literal
       ``\\{S}`` in a description) will be incorrectly unescaped.  This is
       the same trade-off documented for the pipe escape.
    """
    if _enabled is None:
        _enabled = _is_desanitize_enabled()
    if not _enabled:
        return value
    return value.replace("\\{", "{").replace("\\}", "}")


# ── Read-side inverses (moved from parser.py) ───────────────────────────


def _unescape_colons_for_las_value(value: str, *, _enabled: bool | None = None) -> str:
    """Reverse the ``_escape_colons_for_las_value`` transformation.

    The writer applies a two-step colon escape to prevent the parser from
    misinterpreting embedded colons as structural separators:

    1. Insert ``_`` between whitespace and colon: ``" :"`` → ``" _:"``
    2. Insert ``_`` after colon followed by whitespace or end: ``": "`` → ``":_ "``

    The combined effect on ``" : "`` produces ``" _:_ "``.

    This function reverses both steps in the opposite order (step 2 first,
    then step 1), restoring the original colon-separated text.

    F-019: ``_enabled`` gates the transform on the ``desanitize`` flag
    (thread-local ``_DESANITIZE_ENABLED`` when None), mirroring
    :func:`desanitize_las_value` — ``desanitize=False`` (a file NOT produced
    by pylasdev's writer) must leave ``_:`` sequences untouched instead of
    rewriting external content.

    .. note::

       Legitimate underscore characters that happen to form the escape
       pattern (e.g., ``tag_:`` in original data) will be incorrectly
       unescaped.  This is the same trade-off acknowledged by the writer's
       docstring — the roundtrip loss is limited to the contrived case
       where user data naturally contains the escape-artifact patterns.
       F-018: the writer now warns (counted) when its escape is not
       invertible for a value, so the loss is no longer silent.
    """
    if _enabled is None:
        _enabled = _is_desanitize_enabled()
    if not _enabled:
        return value
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


def _unescape_pipes_for_las_value(value: str, *, _enabled: bool | None = None) -> str:
    """Reverse the ``_escape_pipes_for_las_value`` transformation.

    The writer escapes literal pipes in parameter descriptions
    (``|`` → ``\\|``) so genuine description text containing a pipe is
    not misparsed as a LAS 3.0 zone association (``| Zone``) on re-read.
    This function restores the original pipe.

    F-019: ``_enabled`` gates the transform on the ``desanitize`` flag
    (thread-local ``_DESANITIZE_ENABLED`` when None), like
    :func:`desanitize_las_value` — ``desanitize=False`` (an external file)
    must leave ``\\|`` sequences untouched.

    .. note::

       Legitimate backslash-pipe text in original data (e.g., a literal
       ``\\|`` in a description) will be incorrectly unescaped.
    """
    if _enabled is None:
        _enabled = _is_desanitize_enabled()
    if not _enabled:
        return value
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
    path passes ``True`` for the FIRST-column token only; the LAS 1.2/2.0
    data path (data_reader._read_normal / _read_wrapped) passes ``True``
    for the first-column STRING token only (E-19 — the 1.2/2.0 writer
    emits the same M-85 ``_~`` escape to keep the row parseable, so the
    read side restores it position-scoped exactly like the LAS 3.0 path).
    Header call sites and non-first-column tokens pass the default
    ``False``: a genuine external ``_~`` value in any other position is
    preserved (II-13).

    E-01: even when *restore_tilde* is True, the ``_~`` restore is scoped
    to the writer's ACTUAL escape output.  The M-85 escape
    (_writer_base.py:710-737) fires only for a FIRST-column string value
    starting ``~``+NON-letter (or bare ``~``) — a ``~``+letter start is
    handled by the M-28 tilde-strip, never escaped.  So ``_~`` followed
    by an ASCII letter can never be writer output and is preserved (a
    genuine first-column ``_~DEPT`` value survives).  Residual: a
    first-column genuine ``_~3D`` is byte-identical to the writer's
    escape of a genuine ``~3D`` (the ESCAPE-COLLISION case — a section
    holding both writes two identical ``_~3D`` rows and the restore
    fires on both, restoring the escaped value correctly and the genuine
    value to ``~3D``).  This is the same irreducible trade the ``_#``
    restore documents below: the escape prefix is a valid data character,
    so the ambiguous byte pattern cannot be separated on read.

    Args:
        value: The raw value to desanitize.
        restore_tilde: When True, restore ``_~`` → ``~`` for tokens
            matching the M-85 escape predicate (LAS 3.0 first-column
            data path only; default False is fail-safe).
        _enabled: Hoisted ``_DESANITIZE_ENABLED`` flag (IT3-F-01 perf —
            cached once per read).  None → look up the thread-local flag.
    """
    if _enabled is None:
        _enabled = _is_desanitize_enabled()
    if not _enabled:
        return value
    if value.startswith("_#"):
        return value[1:]
    # M-85/E-01: restore the leading '~' escaped by the writer for a
    # first-column string value starting with '~'+non-letter (or bare
    # '~').  The restore is scoped to the writer's ACTUAL escape output:
    # M-85 fires only for '~'+NON-letter starts (a '~'+letter start is
    # handled by the M-28 tilde-strip, never escaped), so '_~'+letter can
    # never be writer output and must be preserved.  The caller (LAS 3.0
    # data path) further restricts restore_tilde to the FIRST column, so
    # a genuine '_~' value in a non-first column is never touched.
    if restore_tilde and value.startswith("_~"):
        rest = value[2:]
        if not rest or not (rest[0].isascii() and rest[0].isalpha()):
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
