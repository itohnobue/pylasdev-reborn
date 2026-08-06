"""Encoding detection utilities for LAS/DEV files.

Geoscience files commonly use:
- UTF-8 (modern files)
- CP1252 / Latin-1 (Western European)
- CP1251 (Russian Windows encoding)
- CP866 (Russian DOS encoding)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .exceptions import LASEncodingError

try:
    import chardet

    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

logger = logging.getLogger(__name__)

# F-ITER2-SEC-H01 + F-ITER2-SEC-M05: Default maximum file size (500 MB) to
# protect against unbounded memory consumption from Path.read_bytes() and
# content.splitlines() (each doubles peak memory — up to 2x the file size).
# Geoscience LAS files larger than 500 MB are virtually nonexistent in
# practice; the largest known real-world LAS file is ~50 MB.
DEFAULT_MAX_FILE_SIZE = 524_288_000  # 500 MB

# Ordered by likelihood in Russian geoscience context.
# F-ITER2-D4b-M09: cp1251 (Windows Cyrillic) is tried before cp866 (DOS
# Cyrillic) because cp1251 is the dominant encoding for modern Russian
# geoscience files.  Both are single-byte encodings that never raise
# UnicodeDecodeError, so fallback ordering determines which encoding wins
# for files that decode successfully under both.  cp866 was previously
# first, which caused cp1251 files to decode as mojibake — cp866 maps
# Cyrillic bytes to different code points, producing non-\\w characters
# that cause DATA_LINE_PATTERN failures and silently dropped curves.
# Reversed order: cp1251 first correctly handles modern Windows-encoded
# files, and legacy cp866 files also decode under cp1251 (as Latin-1
# mojibake rather than Cyrillic mojibake).  For truly ambiguous files,
# chardet detection (if available) runs first.
FALLBACK_ENCODINGS = ["utf-8", "cp1251", "cp866", "cp1252", "latin-1"]

# F-ITER2-D4b-M09: Minimum proportion of word characters (\w) required for
# a decoded content to pass the mojibake detection heuristic.  Both cp866
# and cp1251 decode any byte sequence without error, so the fallback chain
# cannot distinguish them by UnicodeDecodeError alone.  Instead, we compare
# candidate decodings and select the one with the highest proportion of word
# characters (alphanumeric + underscore).  Real LAS files have ~50-80% word
# characters (mnemonics, values, section headers).  Garbled single-byte
# decode to the wrong encoding typically produces significantly fewer \w
# chars (e.g. cp866 content decoded as cp1251: 33% vs 100% correctly).
# Sampling the first _MIN_VALIDATION_CHARS avoids scanning multi-GB files.
_MIN_VALIDATION_CHARS = 65_536  # Sample first 64K chars/bytes for analysis

# E-06 + M-81: Near-tie margin for the Cyrillic-vs-Western selection.  Byte
# 0xB9 is "№" in cp1251 (not alnum) but "¹" in cp1252 (alnum), so any "№"
# in the sample gives cp1252 a small but strict ratio advantage (~2.9%)
# over cp1251.  The ratio-primary sort (load-bearing — see F-24 and the
# UTF-8 preference below) would then select cp1252 and decode the whole
# file as mojibake.  When the content is judged Cyrillic and the best
# Western candidate beats the best Cyrillic candidate by only this
# near-tie margin, prefer the Cyrillic candidate.  Genuine encoding
# mismatches produce ratio gaps >10%, so this never overrides a clear
# winner.  (M-81: the exact № contribution is subtracted from the gap in
# _decode_best_quality, so the margin no longer needs to bound №-density.)
_NEAR_TIE_CYRILLIC_MARGIN = 0.04  # 4% relative ratio gap

# M-31: Cyrillic detection scans the WHOLE file — there is deliberately no
# fixed sample window.  A fixed window (64K, widened to 1 MiB by E-07) left
# files whose Cyrillic content starts beyond the window (large ASCII
# headers, large numeric ~A blocks before string curves) invisible to the
# detector, so they were misdecoded as cp1252/latin-1 → silent mojibake of
# all header strings.  The whole-file scan is cheap: the byte-class regexes
# below run at C speed and _has_confirmed_cyrillic_run() early-exits at the
# first qualifying run.  (F-88's word-char ratio sample stays bounded at
# _MIN_VALIDATION_CHARS — only the Cyrillic detector scans the whole file.)

# M-57/M-82: A "Cyrillic byte" is a byte that decodes to a Cyrillic code
# point (U+0400-U+04FF) under cp1251.  Bytes 0xC0-0xFF are ambiguous — under
# cp1252 they are accented Latin letters — so 3 such bytes in a row
# ("Ñáñez" → "Сбс") is NOT a reliable Cyrillic signal (M-82).  A run is  # noqa: RUF003
# confirmed as genuine Cyrillic when it is _CYRILLIC_RUN_CONFIRM+ bytes long
# (real words are 4+ Cyrillic letters; 4+ consecutive cp1252 accents are
# essentially nonexistent), OR when it contains a "strong" byte — a byte
# that is Cyrillic under cp1251 but NOT an alnum letter under cp1252
# (control chars/symbols that Western text cannot plausibly contain).  The
# strong-byte rule also catches cp866-encoded Cyrillic, which maps to
# 0x80-0x9F under cp1251 (runs of 3, e.g. "ЋђЋ").  A 3-byte run is also
# confirmed when a "№" marker (0xB9) is ADJACENT to it: № is a Russian
# typographic convention (well names "СКВ №1", "ПЛАСТ №2"), and genuine  # noqa: RUF003
# cp1251-encoded № is the raw byte 0xB9, so the rule fires only on
# single-byte Cyrillic files (M-81; UTF-8-encoded "№" is 0xE2 0x84 0x96).
# F-18: the №-confirmation is scoped to a small window AROUND the run —
# the 0xB9 byte must be within _NUMERO_ADJACENCY_WINDOW bytes before the
# run start or after the run end.  It must NOT be a whole-file membership
# test: a lone Western "¹" (also byte 0xB9 in cp1252) in a footnote
# hundreds of KB away from an accented run ("Ñáñez") would otherwise flip
# a genuine cp1252 file to cp1251 → mojibake (M-82 defect class re-opened).
# Byte-frequency alone (F-24) is NOT used: its common-letter byte set is
# byte-identical to common cp1252 accents, so ñ-dense Spanish/Portuguese
# text exceeds any realistic frequency threshold (M-57); run confirmation
# subsumes it (any file dense enough to exceed a frequency threshold
# contains 4+ char words).
_CYRILLIC_RUN_CONFIRM = 4
_STRONG_RUN_MIN = 3
# ENC-M1 (M4 regression, FIX pass 2): minimum length of a Cyrillic run in
# the winning NON-Western candidate's own decode that counts as a "word".
# A genuine cp866 Cyrillic word (УАЗ, ГАЗ, СКВ, ПЛАСТ, ПРИВЕТ) is at least  # noqa: RUF003
# 3 letters.  The cp1251 run detector (_has_confirmed_cyrillic_run) cannot
# see this class: under cp1251 the УАЗ bytes (0x93 0x80 0x87) decode to  # noqa: RUF003
# "“Ђ‡" — a single Cyrillic code point, no run — so a genuine cp866 file
# whose short Cyrillic word is mixed into ASCII prose is judged
# non-Cyrillic and the Western near-tie rescue fires, flipping it to a
# cp1252 mojibake decode (a NEW regression vs HEAD 82cadce).  The
# discriminator reads the Cyrillic evidence from the candidate that IS
# correct — the winning non-Western decode — instead of from cp1251.
_WORD_LIKE_CYRILLIC_RUN_MIN = 3
# F-02 (ENC-1 regression, FIX pass 4): bytes that decode to a Cyrillic
# letter (U+0400-U+04FF) under cp1251.  Within the ambiguous 0x80-0x9F
# range — the ONLY range rule (2) of _is_genuine_word_run examines, because
# rule (1) already accepts any byte >= 0xA0 as unambiguous — this set is
# {0x80, 0x81, 0x83, 0x8A, 0x8C, 0x8D, 0x8E, 0x8F, 0x90, 0x9A, 0x9C, 0x9D,
# 0x9E, 0x9F}: bytes that cp1252 renders as € ƒ Š Œ Ž š œ ž Ÿ or leaves
# UNDEFINED.  The Western smart-punctuation class (… † ‡ ‰ ‚ „ – — ' ")  # noqa: RUF003
# uses bytes that are NOT Cyrillic under cp1251, so a run containing a
# cp1251-Cyrillic byte is a genuine Cyrillic word rather than a misdecoded
# symbol cluster.  Derived from the codec table itself (the same source of
# truth as _alnum_class / _build_cyrillic_run_regexes), not a hand-enumerated
# list.
_CP1251_CYRILLIC_BYTES: frozenset[int] = frozenset(
    b
    for b in range(0x80, 0x100)
    if 0x0400 <= ord(bytes([b]).decode("cp1251", errors="replace")) <= 0x04FF
)
# E-01 (FIX pass 5): LAS-context window for the context discriminator in
# _run_has_las_context.  The window is anchored at the first non-whitespace /
# non-punctuation byte after the all-ambiguous Cyrillic run (parameter *k* of
# _run_has_las_context), so its EFFECTIVE reach is the skipped whitespace /
# punctuation plus this many bytes (L-2: the pass-5 comment said "after the
# run" but the code anchors at k).  Within the window, an ASCII digit whose
# immediately-preceding token is a full-uppercase ASCII word marks a genuine
# LAS label ('THE WELL ТЕСТ FIELD 1000.0' places '1000.0' ~8 bytes after the  # noqa: RUF003
# word, preceded by the uppercase mnemonic FIELD); Western prose digits
# follow lowercase/mixed tokens ('and 3 more', 'dated 2024') and do NOT mark
# a label (M-1 pass-6 refinement).  The value was tuned against the full
# pinned matrix (24/24), the encoding test file, and the full suite; 16-40
# was the validated range.
_LAS_DIGIT_CONTEXT_WINDOW = 24
# F-18: adjacency window for the "№" (0xB9) confirmation rule — a genuine
# "СКВ №1" places the marker within 1-3 bytes of the run (space/punctuation  # noqa: RUF003
# + 0xB9), so 8 bytes on either side comfortably covers real-world labels
# while keeping a far-away footnote marker out of scope.
_NUMERO_ADJACENCY_WINDOW = 8


# ENC-02: Codec-derived "inflator" byte classes for the Western near-tie
# rescue (ENC-02(b) below).  A byte is an inflator for the pair
# (non_western, western) when it decodes to an ALPHANUMERIC under the
# non-Western encoding but NOT under the Western one — e.g. 0x96 is 'Ц'
# (alnum) under cp866 but '–' (non-alnum) under cp1252.  Such bytes  # noqa: RUF003
# inflate the non-Western word-char ratio above a genuine Western page
# whenever the file contains them — the root of the Western→cp866/cp1251
# flip.  The class is derived from the codec tables themselves (the same
# source of truth _build_cyrillic_run_regexes uses), NOT from a
# hand-enumerated list: the previous 7-byte _SMART_PUNCT_BYTES set
# (0x91-0x97) could never converge — every sibling byte an audit
# discovered (0x80 €, the 0x82/0x84-0x87/0x89/0x8B control range, the
# 0xA1-0xBF/0xD7/0xF7 symbol class) was another hole (F-01/F-09).  The
# table is per-pair because cp866-over-cp1252 ≠ cp866-over-latin-1 ≠
# cp1251-over-*: the artifact must be exact for the actual candidate pair.
def _alnum_class(enc: str) -> frozenset[int]:
    """Return the single bytes 0x80-0xFF that decode to an alphanumeric
    under *enc* — the codec's own tables define what counts as a word char."""
    return frozenset(
        b for b in range(0x80, 0x100) if bytes([b]).decode(enc, errors="replace").isalnum()
    )


_INFLATORS: dict[tuple[str, str], frozenset[int]] = {
    (n, w): _alnum_class(n) - _alnum_class(w)
    for n in ("cp866", "cp1251")
    for w in ("cp1252", "latin-1")
}


def _printable_inflators(pair: tuple[str, str], inflators: frozenset[int]) -> frozenset[int]:
    """Return the subset of *inflators* that decode to PRINTABLE characters
    under the Western member of *pair*.

    M4: the codec-derived table treats every byte that is alnum under the
    non-Western encoding but NOT under the Western one as a "Western symbol
    inflator" — but the cp866 Cyrillic uppercase range (0x80-0x9F) decodes to
    C1 control chars under latin-1 (and 0x8F/0x90/0x9D are UNDEFINED under
    cp1252).  Such bytes are NOT Western symbols: real Western text never
    contains C1 controls, so a genuine cp866 Cyrillic word like ПРИВЕТ must  # noqa: RUF002
    not be "explained away" as Western punctuation.  Only bytes that the
    Western codec itself renders as printable characters (Euro sign, low
    quotes, ellipsis, daggers, inverted marks, the multiplication/division
    signs, the 0xA1-0xBF symbol class, ...) qualify as Western-symbol
    evidence.
    """
    return frozenset(b for b in inflators if _is_printable_byte(pair[1], b))


def _is_printable_byte(enc: str, b: int) -> bool:
    """Return True if byte *b* decodes to a printable character under *enc*.

    Uses an explicit UnicodeDecodeError catch rather than ``errors="replace"``
    because U+FFFD (REPLACEMENT CHARACTER) reports ``isprintable() == True``,
    which would wrongly count bytes that are UNDEFINED in the codec (e.g.
    0x8F/0x90 in cp1252) as Western-symbol evidence.
    """
    try:
        return bytes([b]).decode(enc).isprintable()
    except UnicodeDecodeError:
        return False


# M-19: the Western-symbol strong bytes — bytes that are Cyrillic under
# cp1251, NOT alnum under cp1252 (hence "strong"), AND printable under
# cp1252 — i.e. bytes a real Western file can plausibly contain as symbols:
# € (0x80), ¡ ¢ £ ¥ ¨ ¯ ´ ¸ ¿ (0xA1-0xA3/0xA5/0xA8/0xAF/0xB4/0xB8/0xBF) and  # noqa: RUF003
# × ÷ (0xD7/0xF7).  3+ consecutive such bytes fired the strong-byte Cyrillic  # noqa: RUF003
# rule and flipped whole Western files to cp1251 mojibake (M-19); the
# genuine-Cyrillic mirror runs (cp1251 ЎЎЎ/ЈЈЈ/ҐҐҐ etc.) are essentially  # noqa: RUF003
# nonexistent in real Russian text — an ASYMMETRIC trade — so they are
# carved out of the strong class.  The 0x80-0x9F cp866-only strong bytes
# (0x81/0x8D/0x8F/0x90/0x9D — C1 controls / undefined under cp1252) stay:
# they are the load-bearing evidence for cp866-encoded Cyrillic.
_WESTERN_STRONG_SYMBOL_BYTES: frozenset[int] = frozenset(
    b
    for b in range(0x80, 0x100)
    if 0x0400 <= ord(bytes([b]).decode("cp1251", errors="replace")) <= 0x04FF
    and not bytes([b]).decode("cp1252", errors="replace").isalnum()
    and _is_printable_byte("cp1252", b)
)

# E-02: the strict whole-file evidence set — cp1251-Cyrillic bytes in the
# 0x80-0x9F range (the cp866 UPPERCASE Cyrillic zone) MINUS the M-19-carved
# Western symbols (0x80 € — byte-identical to cp866 А) MINUS the M-07-carved  # noqa: RUF003
# Western LETTERS.  This is the only byte evidence the whole-file word-like
# check may trust: bytes >= 0xA0 are M-82/N-10-ambiguous (cp866 lowercase /
# Western symbols / accented Latin), and 0xC0-0xFF are accented-Latin-ambiguous
# (M-82) — neither may alone flip a Western winner.  M-07: the 8 alnum-under-
# cp1252 letters (ƒ Š Œ Ž š œ ž Ÿ = 0x83/0x8A/0x8C/0x8E/0x9A/0x9C/0x9E/0x9F)
# are ALSO carved — a genuine cp1252 'the ŠŠŠ field' (0x8A×3) passed the  # noqa: RUF003
# strict set-byte evidence and flipped whole Western files to cp1251 mojibake.
# What remains is the C1-control/undefined-under-cp1252 class (0x81/0x8D/0x8F/
# 0x90/0x9D) — bytes real Western text cannot contain.  Genuine cp866
# uppercase words (СКВАЖИНА = 0x91 0x8A 0x82 0x80 0x86 0x88 0x8D 0x80) still
# carry 0x8D-class set bytes.
_STRICT_CYRILLIC_EVIDENCE_BYTES: frozenset[int] = frozenset(
    b
    for b in _CP1251_CYRILLIC_BYTES
    if b < 0xA0
    and b not in _WESTERN_STRONG_SYMBOL_BYTES
    and not bytes([b]).decode("cp1252", errors="replace").isalnum()
)


# M4 (F-01/F-09 regression): the Western near-tie rescue may only switch to
# a Western candidate when the sample contains Western-SYMBOL bytes that are
# genuinely printable under that candidate (see _printable_inflators) — NOT
# cp866 Cyrillic letters that merely decode to C1 controls/symbols.
_PRINTABLE_INFLATORS: dict[tuple[str, str], frozenset[int]] = {
    pair: _printable_inflators(pair, inflators) for pair, inflators in _INFLATORS.items()
}

# ENC-02 (a): material-margin for the detected-encoding priority.  Real
# encoding mismatches produce ratio gaps >10%; a <5% gap is within the
# near-tie band and the high-confidence chardet answer should be honored.
_DETECTED_ENCODING_MARGIN = 0.05

# ENC-02 (b): plausibility floor for the Western near-tie rescue.  The
# rescue may only switch to a Western candidate whose OWN decode looks
# like plausible text — real files have ~50-80% word characters (see the
# module docstring), so a Western decode with a word-char ratio below 0.5
# is garbage.  Without this floor, a genuine short cp866/cp1251 file whose
# every byte is a codec inflator (e.g. "ПРИВЕТ" in cp866: 0x8F-0x92 all
# decode to Cyrillic alnum under cp866 but control chars/symbols under
# latin-1/cp1252) would have its 100% ratio gap fully "explained" by the
# artifact and be wrongly rescued to a Western mojibake decode.
_WESTERN_RATIO_FLOOR = 0.5

# M4 (F-01/F-09 regression): minimum ASCII-letter evidence the winning
# Western candidate must contain for the ENC-02(b) rescue to fire.  Genuine
# cp866 files whose short Cyrillic words map to PRINTABLE symbols under
# cp1252 (e.g. "УАЗ" -> 0x93 0x80 0x87 -> "“€‡") pass the printable-inflator  # noqa: RUF003
# filter, so the artifact fully "explains" the ratio gap even though the
# file is real Cyrillic data.  Such files are digit-heavy LAS data (mnemonic
# + numeric blocks) with an isolated Cyrillic well name — their Western
# decode contains almost no ASCII letters.  Genuine Western files carrying
# €/smart-punct/symbols are prose or description text with real ASCII words
# ("Well: 1234 € 5678 prix 50€ total 123€" = 13 letters).  Requiring 8 ASCII
# letters separates the two classes with margin while preserving every
# documented F-01/F-09 rescue case (all ≥ 13 letters).
_WESTERN_MIN_ASCII_LETTERS = 8


def _build_cyrillic_run_regexes() -> tuple[re.Pattern[bytes], re.Pattern[bytes], re.Pattern[bytes]]:
    """Build the C-speed byte regexes used by :func:`_has_confirmed_cyrillic_run`.

    Returns:
        (runs of ``_CYRILLIC_RUN_CONFIRM``+ Cyrillic bytes,
         runs of 3+ Cyrillic bytes,
         runs of Cyrillic bytes containing at least one strong byte).
    """
    cyrillic_bytes = bytes(
        b
        for b in range(0x80, 0x100)
        if 0x0400 <= ord(bytes([b]).decode("cp1251", errors="replace")) <= 0x04FF
    )
    strong_bytes = bytes(
        b for b in cyrillic_bytes if not bytes([b]).decode("cp1252", errors="replace").isalnum()
    )
    # M-19: carve the Western-symbol strong bytes out of the strong class.
    # A "strong byte" is justified as a byte "Western text cannot plausibly
    # contain" — but the 0x80 € / 0xA1-0xA3 ¡¢£ / 0xA5 ¥ / 0xA8 ¨ / 0xAF ¯ /
    # 0xB4 ´ / 0xB8 ¸ / 0xBF ¿ / 0xD7 × / 0xF7 ÷ class IS plausibly  # noqa: RUF003
    # contained in Western text (currency, inverted punctuation, math
    # signs), so 3+ consecutive such bytes fired the strong rule and
    # flipped whole Western files to cp1251 mojibake.  The genuine-Cyrillic
    # mirror (cp1251 ЎЎЎ/ЈЈЈ/ҐҐҐ runs of the same bytes) is essentially  # noqa: RUF003
    # nonexistent in real Russian text — an ASYMMETRIC trade — so the
    # printable-under-cp1252 strong bytes are carved out (the set is
    # computed once in _WESTERN_STRONG_SYMBOL_BYTES, above).  The cp866-only
    # strong bytes (0x81/0x8D/0x8F/0x90/0x9D — C1 controls/undefined under
    # cp1252) stay: they are the load-bearing evidence for cp866-encoded
    # Cyrillic (МЕСТОРОЖДЕНИЕ's 0x8E 0x90 0x8E triple, E-07).
    strong_bytes = bytes(b for b in strong_bytes if b not in _WESTERN_STRONG_SYMBOL_BYTES)
    cyrillic_class = b"[" + cyrillic_bytes + b"]"
    strong_class = b"[" + strong_bytes + b"]"
    return (
        re.compile(cyrillic_class + b"{" + str(_CYRILLIC_RUN_CONFIRM).encode("ascii") + b",}"),
        re.compile(cyrillic_class + b"{3,}"),
        re.compile(cyrillic_class + b"*" + strong_class + cyrillic_class + b"*"),
    )


_CYRILLIC_RUN_GE_RE, _CYRILLIC_RUN_3_RE, _STRONG_CYRILLIC_RUN_RE = _build_cyrillic_run_regexes()

# E-02: C-speed pre-gate for the whole-file word-like Cyrillic evidence
# check in _decode_best_quality.  A decoded string without a run of 3+
# Cyrillic code points cannot contain a word-like Cyrillic word, so the
# Python-level _has_word_like_cyrillic_run scan is only entered when this
# regex (which runs at C speed over the whole decode) matches — keeping
# pure-ASCII and Western-only files on the cheap path.
_CYRILLIC_WORD_PRE_RE = re.compile("[\u0400-\u04FF]{3}")


def _window_has_numero_prefix(window: bytes) -> bool:
    """Return True if *window* contains a 0xB9 byte that is a Cyrillic "№".

    Byte 0xB9 is "№" in cp1251 but "¹" (superscript one) in cp1252 — a
    Western footnote "¹" adjacent to an accented run ("Nota¹ Ñáñez") must
    NOT confirm Cyrillic (ENC-01).  The Russian typographic convention
    always places a NUMBER after the № (well labels like "№1", "№ 2"), so
    a 0xB9 counts as "№" only when followed — after optional whitespace —
    by an ASCII digit.  A Western "¹" is an ordinal/footnote marker, not a
    "№ <number>" prefix, so the digit-follow constraint disambiguates.

    E-26: the digit-follow constraint alone is NOT sufficient — a Western
    superscript-one is commonly followed by a digit too ("Nota¹1 Ñáñez",
    footnote lists "1¹ 2²"), which flipped genuine cp1252 files to cp1251
    mojibake.  The Russian convention places № immediately AFTER a Cyrillic
    word ("СКВ №1", "ПЛАСТ №2"), so the marker must also be PRECEDED —  # noqa: RUF002
    after optional whitespace — by a byte that is a Cyrillic letter under
    cp1251 (_CP1251_CYRILLIC_BYTES).  Western superscripts follow ASCII
    words ("Nota¹1", "señal¹2", "Values ¹ 2") or stand alone, so the
    preceding-byte constraint disambiguates the two classes.  Residual: a
    superscript directly following an ACCENTED Latin letter ("café¹2" —
    é decodes to Cyrillic й under cp1251) still fires; that byte is
    M-82-ambiguous by design.

    M-08: the preceding-byte requirement alone is NOT sufficient either —
    the Russian convention ALSO places № BEFORE the word ("№ 1 СКВ",
    "№1 СКВ" — well labels at line start), where no preceding Cyrillic
    byte exists.  The marker is additionally confirmed at the start of the
    window/data or at a line start (prev < 0, or a newline after optional
    whitespace).  A preceding ASCII LETTER is deliberately NOT accepted:
    "Nota¹1 Ñáñez" / "Values ¹ 2 Ñáñ" (Western superscript-one + digit +
    accent run) must not look Russian (E-26), and the "SKV № 1 СКВ"
    after-ASCII class is byte-structurally identical to them at the marker
    level — no rule can separate the two without misclassifying one side.
    """  # noqa: RUF002
    idx = 0
    while True:
        idx = window.find(0xB9, idx)
        if idx == -1:
            return False
        nxt = idx + 1
        while nxt < len(window) and window[nxt] in (0x20, 0x09):  # space/tab
            nxt += 1
        if nxt < len(window) and 0x30 <= window[nxt] <= 0x39:
            prev = idx - 1
            while prev >= 0 and window[prev] in (0x20, 0x09):  # space/tab
                prev -= 1
            # M-08: line start / data start confirms the №-before-word
            # convention ("№ 1 СКВ", "№1 СКВ" — possibly after leading  # noqa: RUF003
            # whitespace on the line).
            if prev < 0:
                return True
            if window[prev] in (0x0A, 0x0D):
                return True
            if window[prev] in _CP1251_CYRILLIC_BYTES:
                return True
        idx += 1


def _has_confirmed_cyrillic_run(data: bytes) -> bool:
    """Return True if *data* contains a run of bytes confirming Cyrillic text.

    A run of ``_CYRILLIC_RUN_CONFIRM`` or more consecutive Cyrillic bytes, a
    run of 3+ such bytes together with a "№" marker (0xB9) ADJACENT to the
    run (within ``_NUMERO_ADJACENCY_WINDOW`` bytes of it — see the module
    constants, F-18), or a run of ``_STRONG_RUN_MIN`` or more that contains a
    strong Cyrillic byte (a byte that is Cyrillic under cp1251 but not an
    alnum letter under cp1252), strongly indicates genuine cp1251/cp866
    content — see the module constants for the rationale
    (M-31/M-57/M-81/M-82/F-18).
    """
    if _CYRILLIC_RUN_GE_RE.search(data):
        return True
    # F-18: The №-confirmation rule is ADJACENCY-scoped, not whole-file.
    # The previous `b"\xb9" in data and _CYRILLIC_RUN_3_RE.search(data)`
    # fired when the 0xB9 byte and the 3-byte run were anywhere in the
    # file (probes: 700KB/1MB/10MB apart all fired), so a genuine cp1252
    # Western file with a single "¹" (0xB9 in cp1252) plus an accented
    # run ("Ñáñez") was flipped to cp1251 → mojibake with zero warnings.
    # Only a 0xB9 within a small window of the run itself confirms the
    # Russian "№" convention.
    # ENC-01: even ADJACENT, a Western "¹" must not confirm Cyrillic —
    # the marker counts only when followed by a digit ("СКВ №1").  # noqa: RUF003
    if b"\xb9" in data:
        for match in _CYRILLIC_RUN_3_RE.finditer(data):
            window_start = max(0, match.start() - _NUMERO_ADJACENCY_WINDOW)
            window_end = min(len(data), match.end() + _NUMERO_ADJACENCY_WINDOW)
            if _window_has_numero_prefix(data[window_start:window_end]):
                return True
    for match in _STRONG_CYRILLIC_RUN_RE.finditer(data):
        if len(match.group()) >= _STRONG_RUN_MIN:
            return True
    return False


def _has_word_like_cyrillic_run(
    decoded: str,
    raw: bytes | None = None,
    sample_truncated: bool = False,
    *,
    allow_rule1: bool = True,
) -> bool:
    """Return True if *decoded* contains a run of Cyrillic letters that reads
    as a GENUINE Cyrillic word.

    A run of ``_WORD_LIKE_CYRILLIC_RUN_MIN`` or more consecutive Cyrillic
    code points (U+0400-U+04FF) is "word-like" when it is a genuine Cyrillic
    word rather than a single-byte misdecode of Western typographic
    punctuation.  Two evidence rules distinguish the classes when *raw* (the
    byte string corresponding to *decoded*, for single-byte decodes) is
    supplied:

    (1) Unambiguous-alphabet rule: a run containing a character whose source
        byte is NOT in the ambiguous 0x80-0x9F range (lowercase Cyrillic
        0xA0-0xAF/0xE0-0xEF, Ё/ё and the 0xF0-0xF7 extras) is unambiguously
        genuine — Western smart punctuation occupies only 0x80-0x9F, which
        decodes to UPPERCASE А-Я under cp866.  # noqa: RUF003
    (2) Boundary rule: an all-ambiguous run is word-like unless it is
        Western prose punctuation.  The byte after the run (after optional
        whitespace / ASCII punctuation) decides: digits/hyphens/№ or the  # noqa: RUF003
        end of the sample mark a genuine LAS Cyrillic label (УАЗ-469,  # noqa: RUF003
        СКВ №1); an ASCII letter is ambiguous — genuine cp866 words are  # noqa: RUF003
        ALSO followed by a space + ASCII word (УАЗ FIELD), so the run is  # noqa: RUF003
        then judged by its byte content (F-02: a byte that is Cyrillic
        under cp1251 marks a genuine word, see _is_genuine_word_run).

    M-20 (sample-end sub-mechanism): *sample_truncated* tells rule (2) that
    the sample ends at a 64K boundary, not at the end of the file.  A
    truncated sample-end run carries no boundary evidence — the true file
    continues beyond the sample, so "end of sample" must NOT mark the run
    genuine (a Western smart-punct cluster sitting exactly at the 64K
    boundary would otherwise block the ENC-02(b) rescue → cp866 mojibake).
    At a TRUE end-of-file (sample_truncated=False) the end-of-sample rule
    still marks a genuine LAS Cyrillic label (УАЗ at EOF).

    ENC-M1 (M4 regression, FIX pass 2): this is the discriminator that
    distinguishes genuine cp866 Cyrillic (0x80-0x9F range, all bytes
    printable under cp1252) from genuine Western symbols.  The cp1251 run
    detector cannot see the УАЗ class (under cp1251 those bytes decode to  # noqa: RUF003
    "“Ђ‡" — no Cyrillic run), so the Western near-tie rescue fired on it
    and flipped genuine cp866 files to cp1252 mojibake.  Reading the run
    evidence from the winning NON-Western decode (where the Cyrillic word
    IS visible) restores HEAD parity for the prose+УАЗ class while keeping
    the F-09 Western symbol rescues (their cp866 misreads are embedded, so
    the rescue still fires).

    ENC-1 (FIX pass 3): the pass-2 discriminator classified STANDALONE
    typographic-symbol clusters (…†‡, †‡‰, ‚„…, –—… — space/punct-bounded  # noqa: RUF003
    runs of 3+ Cyrillic code points in the cp866 decode) as word-like, so  # noqa: RUF003
    genuine cp1252 files carrying them decoded cp866 mojibake (a NEW
    regression vs the v2.0.3 release).  Rule (2) now distinguishes them:
    prose punctuation is followed by ASCII words, while genuine LAS Cyrillic
    labels are followed by digits/hyphens.
    F-02 (FIX pass 4): rule (2)'s "followed by an ASCII word" test over-
    corrected — a genuine cp866 Cyrillic word is ALSO followed by a space +
    ASCII word (the most natural prose form, 'THE WELL УАЗ FIELD 1000.0'),  # noqa: RUF003
    so the pass-3 rule flipped the space-separated genuine class to cp1252
    mojibake.  The ASCII-letter-follows case is now decided by the run's
    own bytes: a byte that is Cyrillic under cp1251 (_CP1251_CYRILLIC_BYTES)
    marks a genuine word; the Western smart-punctuation clusters use bytes
    that are not Cyrillic under cp1251 and stay prose punctuation.

    When *raw* is None (multi-byte decode, e.g. utf-8), every Cyrillic run
    is genuine and the original standalone-token rule applies: the run is
    word-like when it is NOT embedded between ASCII letters.
    """  # noqa: RUF002
    i = 0
    n = len(decoded)
    while i < n:
        if 0x0400 <= ord(decoded[i]) <= 0x04FF:
            j = i + 1
            while j < n and 0x0400 <= ord(decoded[j]) <= 0x04FF:
                j += 1
            if j - i >= _WORD_LIKE_CYRILLIC_RUN_MIN:
                if raw is not None:
                    if _is_genuine_word_run(
                        decoded, raw, i, j, sample_truncated, allow_rule1=allow_rule1
                    ):
                        return True
                else:
                    before = decoded[i - 1] if i > 0 else ""
                    after = decoded[j] if j < n else ""
                    if not (
                        (before and before.isascii() and before.isalpha())
                        or (after and after.isascii() and after.isalpha())
                    ):
                        return True
            i = j
        else:
            i += 1
    return False


def _is_genuine_word_run(
    decoded: str,
    raw: bytes,
    i: int,
    j: int,
    sample_truncated: bool = False,
    *,
    allow_rule1: bool = True,
) -> bool:
    """Return True when the single-byte Cyrillic run ``decoded[i:j]`` (with
    corresponding raw bytes ``raw[i:j]``) reads as a genuine Cyrillic word
    rather than a cp866/cp1251 misdecode of Western typographic punctuation.

    Applies the two evidence rules documented in
    :func:`_has_word_like_cyrillic_run`: rule (1) the unambiguous-alphabet
    evidence, and rule (2) the boundary evidence.  *sample_truncated* is
    forwarded to rule (2)'s end-of-sample branch (see the caller's
    docstring, M-20).  *allow_rule1=False* (E-02 whole-file evidence) skips
    rule (1): bytes >= 0xA0 (cp866 lowercase, Western symbols, accented
    Latin) are ambiguous beyond the ratio window and must not alone mark a
    run genuine — only the 0x80-0x9F uppercase class, decided by rule (2)'s
    set-byte / boundary / LAS-context evidence, counts as whole-file
    Cyrillic evidence.
    """
    # Rule (1): a character whose source byte is NOT in the ambiguous
    # 0x80-0x9F smart-punctuation range (lowercase Cyrillic 0xA0-0xAF /
    # 0xE0-0xEF, the 0xF0-0xF7 extras) is unambiguously genuine Cyrillic.
    # M-19: the carved Western-symbol bytes (0xA1-0xA3/0xA5/0xA8/0xAF/0xB4/
    # 0xB8/0xBF/0xD7/0xF7) are NOT unambiguous — they are exactly the bytes
    # whose Western-symbol runs must not confirm Cyrillic.
    if allow_rule1:
        for k in range(i, j):
            if (
                k >= len(raw)
                or (raw[k] >= 0xA0 and raw[k] not in _WESTERN_STRONG_SYMBOL_BYTES)
            ):
                return True
    # Rule (2): for an all-ambiguous run, examine the byte after the run
    # (after optional whitespace / ASCII punctuation).
    k = j
    while k < len(raw):
        b = raw[k]
        if b in (0x09, 0x0A, 0x0D, 0x20):
            k += 1
        elif 0x21 <= b <= 0x2F or 0x3A <= b <= 0x40 or 0x5B <= b <= 0x60 or 0x7B <= b <= 0x7E:
            k += 1
        else:
            break
    if k >= len(raw):
        # End of the sample — a genuine LAS Cyrillic label (УАЗ at EOF,  # noqa: RUF003
        # СКВ №1 tail).  M-20: only when the sample is NOT truncated — a  # noqa: RUF003
        # 64K-boundary cut carries no boundary evidence (the file continues
        # beyond the sample), so a truncated sample-end run must not be
        # judged genuine.
        if sample_truncated:
            return False
        # M-07 (E-02 whole-file evidence, allow_rule1=False): the true-EOF
        # boundary alone must not flip a Western winner either — an
        # accented-Latin run at EOF ('well name áéí' — 0xE1 0xE9 0xED,
        # M-82-ambiguous bytes >= 0xA0) carries no class information.
        # Only a run holding a strict 0x80-0x9F evidence byte
        # (СКВАЖИНА-class) may be judged genuine at EOF.
        if not allow_rule1:
            return any(raw[t] in _STRICT_CYRILLIC_EVIDENCE_BYTES for t in range(i, j))
        return True
    b = raw[k]
    if 0x41 <= b <= 0x5A or 0x61 <= b <= 0x7A:
        # An ASCII letter follows (…†‡ and more; УАЗ FIELD).  F-02: a  # noqa: RUF003
        # genuine cp866 Cyrillic word is ALSO followed by a space + ASCII
        # word — the most natural prose form ('THE WELL УАЗ FIELD 1000.0')  # noqa: RUF003
        # — so "ASCII letter follows" alone is NOT conclusive Western-prose
        # evidence.  First distinguish by the run's byte content: a byte that
        # is a Cyrillic letter under cp1251 (_CP1251_CYRILLIC_BYTES — e.g. УАЗ  # noqa: RUF003
        # = 0x93 0x80 0x87 contains 0x80 = Ђ) marks a genuine Cyrillic
        # word, while the Western smart-punctuation class (…†‡‰‚„–—'")  # noqa: RUF003
        # uses bytes that are NOT Cyrillic under cp1251 and stays Western
        # prose punctuation.  E-02 (allow_rule1=False, whole-file evidence):
        # the trusted set shrinks to _STRICT_CYRILLIC_EVIDENCE_BYTES — the
        # 0x80-0x9F cp866-uppercase zone minus the M-19-carved Western
        # symbols — so accented-Latin bytes (0xC0-0xFF, M-82) and cp866
        # lowercase / Western-symbol bytes (>= 0xA0, N-10) never alone flip
        # a Western winner.
        _set_bytes = _CP1251_CYRILLIC_BYTES if allow_rule1 else _STRICT_CYRILLIC_EVIDENCE_BYTES
        # M-19 guard: a run consisting ENTIRELY of carved Western-symbol
        # bytes (€€€ = 0x80×3, £££ = 0xA3×3, ××× = 0xD7×3 — the cp1251  # noqa: RUF003
        # mirrors ЎЎЎ/ЈЈЈ/ҐҐҐ are essentially nonexistent in real Russian  # noqa: RUF003
        # text) must NOT pass the set-byte evidence — its bytes are Western
        # symbols, not Cyrillic words.  Mixed runs (УАЗ = 0x93 0x80 0x87)  # noqa: RUF003
        # keep the set-byte fast path (the F-02 load-bearing gate).
        if not all(raw[t] in _WESTERN_STRONG_SYMBOL_BYTES for t in range(i, j)):
            if any(raw[t] in _set_bytes for t in range(i, j)):
                return True
        # E-01 (FIX pass 5): the byte-content set above recognizes only 14
        # of the 32 cp866 uppercase letters in the ambiguous 0x80-0x9F
        # range — the 18-letter complement (В Д Е Ж З И Й Л С Т У Ф Х Ц Ч  # noqa: RUF003
        # Ш Щ Ы — e.g. ТЕСТ/ЗИЛ/ВДЕ) carries NO set byte and is  # noqa: RUF003
        # byte-identical under cp866 to the Western punctuation clusters
        # (ВДЕ == ‚„…), so the run's own bytes cannot separate the  # noqa: RUF003
        # classes.  Decide the no-set-byte ASCII-follows case by
        # LAS-CONTEXT evidence instead: an ASCII digit within
        # _LAS_DIGIT_CONTEXT_WINDOW bytes of the first non-ws/non-punct
        # byte whose immediately-preceding token is a full-uppercase ASCII
        # word (LAS mnemonic + value: 'THE WELL ТЕСТ FIELD 1000.0'), or an  # noqa: RUF003
        # UPPERCASE ASCII word following the run (LAS mnemonics are
        # uppercase: FIELD, DEPT, GR), marks a genuine LAS label; Western
        # prose clusters are followed by lowercase words in digit-free
        # prose (M-1 pass-6 refinement: the digit must follow an uppercase
        # token — 'and 3 more', 'dated 2024' are ordinary Western prose,
        # not LAS labels).
        # E3 (fix3 convergence pass): the LAS-context evidence above is
        # context-only — for allow_rule1=False (E-02 whole-file evidence)
        # it must not flip a Western winner either.  The M-07 byte-class
        # gates on the EOF (above) and digit (below) branches are mirrored
        # here: a run holding no strict 0x80-0x9F evidence byte (Western
        # accents áéí = 0xE1 0xE9 0xED, M-07-carved letters ŠŠŠ = 0x8A×3)  # noqa: RUF003
        # may not count as Cyrillic evidence even when followed by an
        # uppercase LAS mnemonic or a digit after an uppercase token —
        # the C1-control strict class (0x81/0x8D/0x8F/0x90/0x9D) stays
        # load-bearing via the set-byte fast path above.
        if not allow_rule1:
            return any(raw[t] in _STRICT_CYRILLIC_EVIDENCE_BYTES for t in range(i, j))
        return _run_has_las_context(raw, k)
    # Any other non-letter byte (digit, hyphen + digit tail, №, symbol)
    # marks a genuine LAS Cyrillic label (УАЗ-469, СКВ №1).  Note the  # noqa: RUF003
    # hyphen itself is ASCII punctuation and was skipped above — what
    # matters is the digit that follows it (УАЗ-469), not the hyphen.  # noqa: RUF003
    # M-07 (E-02 whole-file evidence, allow_rule1=False): the digit
    # boundary alone must not flip a Western winner either — 'áéí 2024'
    # (0xE1 0xE9 0xED, M-82-ambiguous accents) needs a strict 0x80-0x9F
    # evidence byte in the run to count as a genuine label.
    if not allow_rule1:
        return any(raw[t] in _STRICT_CYRILLIC_EVIDENCE_BYTES for t in range(i, j))
    return True


def _run_has_las_context(raw: bytes, k: int) -> bool:
    """Return True when the bytes at/after *k* (the first non-whitespace /
    non-punctuation byte after an all-ambiguous Cyrillic run) show
    LAS-context evidence of a genuine label.

    E-01 (FIX pass 5): the 0x80-0x9F byte space is symmetric — a genuine
    cp866 no-set-byte word (ВДЕ = 0x82 0x84 0x85) is byte-identical to the
    Western smart-punctuation cluster with the same bytes (‚„…), so the
    run's own bytes carry zero class information.  Two LAS-structural
    signals separate the pinned genuine harness shapes from Western prose:
    - an ASCII digit within _LAS_DIGIT_CONTEXT_WINDOW bytes of *k* whose
      immediately-preceding token is a full-uppercase ASCII word (LAS
      mnemonic + value: 'THE WELL ТЕСТ FIELD 1000.0' puts '1000.0' ~8
      bytes after the mnemonic FIELD).  M-1 (FIX pass 6): the pass-5
      "any digit in the window" form was sufficient-but-not-necessary —
      Western prose prices/counts/dates also put digits in the window and
      flipped whole files to cp866 mojibake — so the digit is evidence
      only when the token immediately before it is full-uppercase (a LAS
      mnemonic), not lowercase/mixed prose ('and 3', 'dated 2024',
      'over 100').
    - an UPPERCASE ASCII word following the run (LAS mnemonics are
      uppercase: FIELD, DEPT, GR); Western prose clusters are followed by
      lowercase words.
    """  # noqa: RUF002
    # (a) ASCII digit within the window whose immediately-preceding token is
    # a full-uppercase ASCII word (LAS mnemonic + value: 'THE WELL ТЕСТ  # noqa: RUF003
    # FIELD 1000.0' — the '1' of '1000.0' is preceded by the uppercase
    # mnemonic FIELD).  M-1 (FIX pass 6): the pass-5 "any digit in the
    # window" signal was sufficient-but-not-necessary — Western prose
    # prices/counts/dates ('…†‡ and 3 more', '…†‡ dated 2024', '‚„… over  # noqa: RUF003
    # 100') also put a digit in the window, flipping whole files to cp866
    # mojibake.  Digits preceded by lowercase/mixed tokens are ordinary
    # Western prose and do NOT mark a LAS label.  This is still context
    # evidence (not a byte-content rule), so the 0x80-0x9F symmetry root
    # cause stays addressed.
    # M-20 (LAS-context sub-mechanism): a digit following a `~`-prefixed
    # SECTION MARKER ('~A\n1234.5', '~C\n1 2 3') is NOT LAS label evidence
    # either — the single uppercase letter of a section header is a
    # structural marker, not a data mnemonic + value, so a Western
    # smart-punct cluster followed by a section header must not be judged
    # genuine Cyrillic ('…†‡\n~A\n1234.5' → cp866 mojibake).
    end = min(len(raw), k + _LAS_DIGIT_CONTEXT_WINDOW)
    for pos in range(k, end):
        if not (0x30 <= raw[pos] <= 0x39):
            continue
        t = pos - 1
        while t >= k and (
            raw[t] in (0x09, 0x0A, 0x0D, 0x20)
            or 0x21 <= raw[t] <= 0x2F
            or 0x3A <= raw[t] <= 0x40
            or 0x5B <= raw[t] <= 0x60
            or 0x7B <= raw[t] <= 0x7E
        ):
            t -= 1
        s = t
        while s >= k and (0x41 <= raw[s] <= 0x5A or 0x61 <= raw[s] <= 0x7A):
            s -= 1
        if (
            s < t
            and all(0x41 <= raw[i] <= 0x5A for i in range(s + 1, t + 1))
            and not _is_las_section_marker(raw, s + 1)
        ):
            return True
    # (b) UPPERCASE ASCII word follows the run — the FULL word is uppercase
    # (FIELD, DEPT, GR), NOT a sentence-initial capital in Western prose
    # ('And more' — 'A' followed by lowercase — stays Western).  M-20: a
    # `~`-prefixed section marker ('~A', '~C') is a structural header, not
    # a data mnemonic, so it does NOT mark a LAS label either.
    t = k
    while t < len(raw) and 0x41 <= raw[t] <= 0x5A:
        t += 1
    if t == k:
        return False
    if _is_las_section_marker(raw, k):
        return False
    return t >= len(raw) or not (0x61 <= raw[t] <= 0x7A)


def _is_las_section_marker(raw: bytes, token_start: int) -> bool:
    """Return True when the uppercase ASCII token starting at *token_start*
    belongs to a LAS section header — the LINE it sits on starts with
    ``~`` (e.g. the 'A' of '~A', the 'CURVE'/'INFORMATION' of
    '~CURVE INFORMATION').

    M-20 (LAS-context sub-mechanism): section markers are structural
    headers, not data mnemonics — '~A\\n1234.5' and '~CURVE INFORMATION'
    must not count as Cyrillic-evidence LAS context (a Western smart-punct
    cluster followed by a section header would otherwise block the
    ENC-02(b) rescue → cp866 mojibake).  The check scans back from the
    token to the start of its line: if the line's first non-whitespace
    byte is 0x7E (tilde), the token is part of a section header.  Data
    labels (FIELD, DEPT, GR — preceded by whitespace or prose) never sit
    on a ``~``-prefixed line.
    """
    line_start = raw.rfind(b"\n", 0, token_start) + 1
    p = line_start
    while p < token_start and raw[p] in (0x09, 0x0A, 0x0D, 0x20):
        p += 1
    return p < token_start and raw[p] == 0x7E


def detect_encoding(file_path: Path) -> str:
    """Detect file encoding using chardet (if available).

    Reads the first 50 KB of the file (with UTF-8 BOM stripped) and
    uses chardet to detect the encoding.  When chardet is unavailable,
    returns ``"utf-8"`` unconditionally — this function does not
    perform a multi-encoding fallback chain.  The fallback chain with
    quality-based selection is provided by :func:`read_with_encoding`
    and :func:`_decode_best_quality`.

    Args:
        file_path: Path to the file.

    Returns:
        Detected encoding name, or ``"utf-8"`` when chardet is
        not available or detects with low confidence.
    """
    if HAS_CHARDET:
        with open(file_path, "rb") as f:
            raw = f.read(50_000)
        # Strip UTF-8 BOM before passing to chardet for consistency
        # with read_with_encoding(), which also strips the BOM.
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        return _detect_encoding_from_bytes(raw)

    return "utf-8"


def _detect_encoding_from_bytes(raw_data: bytes) -> str:
    """Detect encoding from pre-read bytes using chardet.

    Internal helper that avoids redundant disk I/O when the caller
    already has the raw bytes available (e.g. ``read_with_encoding()``).

    Args:
        raw_data: Raw bytes from the file (at least the first few KB).

    Returns:
        Detected encoding name, or ``"utf-8"`` if chardet is unavailable
        or confidence is too low.
    """
    if HAS_CHARDET:
        result = chardet.detect(raw_data[:50_000])
        if result["confidence"] and result["confidence"] > 0.7:
            return result["encoding"] or "utf-8"

    return "utf-8"


def _detect_confidence_from_bytes(raw_data: bytes) -> float:
    """Return chardet's confidence for the sample (0.0 when unavailable).

    ``_detect_encoding_from_bytes`` discards the confidence after applying
    the >0.7 gate.  The detected-encoding priority (ENC-02) needs the raw
    confidence value to decide whether a high-confidence detection should
    override the ratio sort, so this helper re-reads it from chardet.
    """
    if HAS_CHARDET:
        result = chardet.detect(raw_data[:50_000])
        return result["confidence"] or 0.0
    return 0.0


def _decode_best_quality(
    raw_bytes: bytes,
    file_path: Path,
    detected: str,
    encodings: list[str],
    detected_confidence: float = 0.0,
) -> tuple[str, str]:
    """Decode raw_bytes using the best-quality encoding from *encodings*.

    When chardet fails or returns low confidence, the fallback chain
    may produce valid decodes under multiple single-byte encodings
    (e.g. both cp1251 and cp866 decode any byte sequence).  This
    function tries each candidate and selects the one whose decoded
    content has the highest proportion of word characters
    (alphanumeric + underscore), which correlates with correct
    Cyrillic decoding in LAS files.

    Args:
        raw_bytes: Raw file bytes.
        file_path: File path (for logging only).
        detected: The chardet-detected encoding (may be ``"utf-8"``
            when detection failed).  Tried first if not already in
            *encodings*.
        encodings: Encoding names to try.
        detected_confidence: chardet's raw confidence for the detection
            (0.0 when unavailable or below the >0.7 gate).  ENC-02:
            a high-confidence statistical detection is honored over the
            crude word-char ratio unless beaten by a material margin.

    Returns:
        Tuple of (selected_encoding, decoded_content).

    Raises:
        LASEncodingError: If no encoding in the list can decode the bytes.
    """
    # F-88: Store only the sample in candidates to avoid memory
    # amplification.  Storing full decoded content for up to 6
    # candidate encodings consumes up to ~7x file size peak memory.
    # After selecting the winning encoding, re-decode the full content.
    candidates: list[tuple[str, str, float]] = []

    # Try detected encoding first (may be utf-8 from failed detection)
    if detected not in encodings:
        try:
            content = raw_bytes.decode(detected)
            content = content.lstrip("\ufeff")
            sample = content[:_MIN_VALIDATION_CHARS]
            ratio = sum(1 for c in sample if c.isalnum() or c == "_") / max(len(sample), 1)
            candidates.append((detected, sample, ratio))
        except UnicodeDecodeError:
            pass

    # Try each fallback encoding
    for enc in encodings:
        try:
            content = raw_bytes.decode(enc)
            content = content.lstrip("\ufeff")
            sample = content[:_MIN_VALIDATION_CHARS]
            ratio = sum(1 for c in sample if c.isalnum() or c == "_") / max(len(sample), 1)
            candidates.append((enc, sample, ratio))
        except UnicodeDecodeError:
            continue

    if not candidates:
        raise LASEncodingError(
            f"Failed to decode {file_path} with any encoding in the fallback chain."
        )

    # Select the encoding with the highest word-character ratio.
    # F-24: When ratios are equal, use a content-based tiebreaker to
    # distinguish Cyrillic from Western European content.  Previous
    # approach (F-24) checked candidates[0]'s decoded output for
    # Cyrillic code points (U+0400-U+04FF) — but when chardet fails
    # (the default with chardet 7.x), candidates[0] is always cp1251,
    # which maps Western accented bytes to Cyrillic code points
    # (e.g. 0xE9 = é → U+0439 = й), producing false positives for
    # Western European files and causing mojibake.  The later
    # byte-frequency approach (F-24) was also unreliable: its top-10
    # common-Russian-byte set is byte-identical to common cp1252
    # accents, so ñ-dense Spanish/Portuguese text false-positives
    # (M-57).  The current detector is _has_confirmed_cyrillic_run()
    # over the whole file (M-31): runs of 4+ Cyrillic bytes, or runs of
    # 3+ containing a "strong" byte, confirm genuine Cyrillic — see the
    # module constants (M-57/M-82).  This subsumes both prior signals.
    _CYRILLIC_ENCS = frozenset({"cp1251", "cp866"})
    _WESTERN = frozenset({"cp1252", "latin-1"})
    _is_cyrillic = _has_confirmed_cyrillic_run(raw_bytes)
    _preferred = _CYRILLIC_ENCS if _is_cyrillic else _WESTERN
    candidates.sort(key=lambda x: (-x[2], 0 if x[0] in _preferred else 1))
    best_enc, _best_sample, best_ratio = candidates[0]

    # ENC-02 (a): Detected-encoding priority.  The ratio-primary sort
    # (F-24) overrides even a perfect high-confidence chardet answer
    # (ADV-H1-2 verified: detected='cp1252' passed directly still
    # selected cp866).  When chardet reports high confidence (>0.7 — the
    # same gate as _detect_encoding_from_bytes) AND the detected encoding
    # actually decodes (is a candidate — the decode-success filter that
    # protects against chardet's wrong "utf-8" answer), honor it unless
    # another candidate beats it by a MATERIAL margin: real encoding
    # mismatches produce ratio gaps >10%, so a <5% gap is within the
    # detection's statistical noise.
    if detected_confidence > 0.7 and best_enc != detected:
        _detected_candidate: tuple[str, str, float] | None = None
        for _cand in candidates:
            if _cand[0] == detected:
                _detected_candidate = _cand
                break
        if _detected_candidate is not None:
            _detected_ratio = _detected_candidate[2]
            if best_ratio - _detected_ratio < _DETECTED_ENCODING_MARGIN:
                best_enc = _detected_candidate[0]
                _best_sample = _detected_candidate[1]
                best_ratio = _detected_ratio

    # E-06: Near-tie Cyrillic preference.  The ratio-primary sort above is
    # load-bearing (a preference-primary sort regresses UTF-8 Cyrillic), so
    # we do not reorder candidates.  Instead, when the content is judged
    # Cyrillic and the winning Western candidate (cp1252/latin-1) beats the
    # best Cyrillic candidate by only the near-tie margin created by the
    # "№" (0xB9) per-char advantage, prefer the Cyrillic candidate.  This
    # leaves every other selection decision untouched.
    if _is_cyrillic and best_enc in _WESTERN:
        _best_cyrillic: tuple[str, str, float] | None = None
        for _cand in candidates:
            if _cand[0] in _CYRILLIC_ENCS:
                _best_cyrillic = _cand
                break
        if _best_cyrillic is not None:
            _cyr_enc, _cyr_sample, _cyr_ratio = _best_cyrillic
            # M-81: 0xB9 decodes to "№" (non-alnum) under cp1251 but "¹"
            # (alnum) under cp1252, so every № in the sample inflates the
            # Western ratio by exactly K/N (K = 0xB9 count in the ratio
            # window).  The fixed margin covers only ~4% №-density; at
            # higher density the rescue fails and genuine cp1251 files
            # misdecode as cp1252.  Subtract the № artifact from the ratio
            # gap before comparing to the margin, so any №-density is
            # handled without weakening the margin for other gaps.
            _ratio_window = raw_bytes[:_MIN_VALIDATION_CHARS]
            _numero_artifact = _ratio_window.count(0xB9) / max(len(_ratio_window), 1)
            if best_ratio - _cyr_ratio - _numero_artifact <= _NEAR_TIE_CYRILLIC_MARGIN * best_ratio:
                best_enc = _cyr_enc
                _best_sample = _cyr_sample
                best_ratio = _cyr_ratio

    # ENC-02 (b): Western near-tie rescue — the symmetric completion of
    # E-06.  When the content is judged NON-Cyrillic and the winning
    # candidate is NOT a Western encoding, a genuine Western file is being
    # misdecoded: bytes that are alphanumerics under the winning
    # non-Western encoding (cp866/cp1251) but punctuation under the
    # Western encodings (e.g. 0x96 → 'Ц' under cp866, '–' under cp1252)  # noqa: RUF003
    # inflate the non-Western word-char ratio above the correct Western
    # page.  Prefer the best Western candidate when the ratio gap — minus
    # the codec-derived inflator artifact for the ACTUAL (best_enc,
    # _west_enc) pair (each such byte inflates the non-Western ratio by
    # exactly K/N) — is within the near-tie margin, mirroring the M-81
    # №-artifact subtraction.  F-01/F-09: the byte class comes from
    # _INFLATORS (the codec tables), not a hand-enumerated subset — the
    # old 7-byte set missed the Euro (0x80), its control-range siblings,
    # and the symbol class → mojibake.  The rescue additionally requires
    # the Western candidate to be plausible text (_WESTERN_RATIO_FLOOR)
    # so a genuine short cp866/cp1251 file whose every byte is an
    # inflator is not wrongly rescued to a garbage Western decode.
    # M4: the codec-derived table alone OVER-explain: every cp866 Cyrillic
    # letter (0x80-0x9F) is an "inflator", so a genuine cp866 short-word
    # file mixed into ≥50% ASCII has its whole ratio gap explained and is
    # flipped to a Western mojibake decode.  Two additional evidence
    # requirements gate the rescue: (1) the artifact only counts inflator
    # bytes that are PRINTABLE under the actual Western candidate
    # (_PRINTABLE_INFLATORS) — cp866 Cyrillic bytes decode to C1 controls
    # under latin-1 and are undefined under cp1252, so they are never
    # "Western symbols"; (2) the Western candidate's decode must contain
    # real ASCII-letter evidence (_WESTERN_MIN_ASCII_LETTERS) — a genuine
    # cp866 file is digit-heavy LAS data whose Cyrillic word maps to a
    # printable symbol blob under cp1252 (e.g. "УАЗ" -> "“€‡") with almost  # noqa: RUF003
    # no ASCII letters, while genuine Western files carrying such symbols
    # are prose with real words.
    # ENC-M1 (M4 regression, FIX pass 2): (2) alone is not enough — a
    # genuine cp866 file with a short Cyrillic word whose bytes are ALL
    # printable under cp1252 (УАЗ/ГАЗ/МАЗ/СКВ-class) PLUS ASCII prose  # noqa: RUF003
    # (≥8 letters) still passes (1)+(2) and is flipped to cp1252 mojibake
    # (a NEW regression vs HEAD 82cadce).  The winning NON-Western
    # candidate's own decode carries the missing evidence: genuine cp866
    # Cyrillic words are standalone Cyrillic runs there, while Western
    # symbol clusters are embedded punctuation attached to ASCII words.
    # (3) the rescue is blocked when the winning candidate's decode
    # contains a word-like Cyrillic run (_has_word_like_cyrillic_run).
    if not _is_cyrillic and best_enc not in _WESTERN:
        _best_western: tuple[str, str, float] | None = None
        for _cand in candidates:
            if _cand[0] in _WESTERN:
                _best_western = _cand
                break
        if _best_western is not None:
            _west_enc, _west_sample, _west_ratio = _best_western
            _ratio_window = raw_bytes[:_MIN_VALIDATION_CHARS]
            _inflators = _PRINTABLE_INFLATORS.get((best_enc, _west_enc), frozenset())
            _smart_punct_artifact = sum(1 for b in _ratio_window if b in _inflators) / max(
                len(_ratio_window), 1
            )
            _west_ascii_letters = sum(1 for c in _west_sample if c.isascii() and c.isalpha())
            # ENC-1 (FIX pass 3): pass the raw bytes corresponding to the
            # winning candidate's sample so the discriminator can examine the
            # run boundaries.  Only single-byte decodes (cp866/cp1251) map
            # decoded[i] -> raw[i] positionally; for multi-byte decodes
            # (utf-8) every Cyrillic run is genuine and the simple
            # standalone-token rule applies.
            _best_sample_raw = (
                raw_bytes[: len(_best_sample)] if best_enc in ("cp866", "cp1251") else None
            )
            _sample_truncated = len(raw_bytes) > len(_best_sample)
            if (
                _west_ascii_letters >= _WESTERN_MIN_ASCII_LETTERS
                and _west_ratio >= _WESTERN_RATIO_FLOOR
                and best_ratio - _west_ratio - _smart_punct_artifact
                <= _NEAR_TIE_CYRILLIC_MARGIN * best_ratio
                and not _has_word_like_cyrillic_run(
                    _best_sample, _best_sample_raw, _sample_truncated
                )
            ):
                best_enc = _west_enc
                _best_sample = _west_sample
                best_ratio = _west_ratio

    # E-02 (HIGH): whole-file Cyrillic evidence when a Western candidate
    # wins.  The ratio sort sees only the first _MIN_VALIDATION_CHARS (64K)
    # window, so a cp866 file whose Cyrillic content lies BEYOND the window
    # (large ASCII headers, large numeric ~A blocks — the M-31 motivating
    # layout) ties the ratio with the Western candidates, and the Western
    # tie-break wins → silent latin-1/cp1252 mojibake of all header strings.
    # The whole-file run detector (_has_confirmed_cyrillic_run) cannot see
    # the common cp866 words (СКВАЖИНА/ПРИВЕТ/ТЕСТ/УАЗ/ПЛАСТ form runs of  # noqa: RUF003
    # at most 2 cp1251-class bytes — no 4-run, no strong byte, no №), and
    # the one detector that CAN see them (_has_word_like_cyrillic_run) was
    # unreachable: its only call site sat inside ENC-02(b), gated on a
    # NON-Western winner.  This check makes the word-like discriminator
    # reachable on the Western-winner path, using the FULL file decode (the
    # 64K sample would truncate the very evidence we need).  It fires only
    # when (a) the file actually has content beyond the ratio window
    # (within-window decisions stay with the pinned within-window machinery
    # — the C-3 ratio-tie trade is deliberately kept), (b) the winner is
    # Western, (c) the run detector did NOT confirm Cyrillic (E-06's
    # near-tie machinery already owns the confirmed case), and (d) a
    # non-Western candidate's full decode contains a word-like Cyrillic run
    # — judged WITHOUT rule (1) and with the strict 0x80-0x9F evidence set
    # (allow_rule1=False): bytes >= 0xA0 (cp866 lowercase / Western symbols
    # / accented Latin) and 0xC0-0xFF accents are ambiguous (M-82/N-10) and
    # must not flip a Western file, while the 0x80-0x9F uppercase class
    # (СКВАЖИНА-class) is decided by rule (2)'s set-byte / boundary /
    # LAS-context evidence.  The check is cheap: _CYRILLIC_WORD_PRE_RE is a
    # C-speed pre-gate that skips decodes without a 3+ Cyrillic run.
    if (
        len(raw_bytes) > _MIN_VALIDATION_CHARS
        and best_enc in _WESTERN
        and not _is_cyrillic
        and best_enc != "utf-8"
    ):
        for _cyr_cand in candidates:
            if _cyr_cand[0] not in _CYRILLIC_ENCS:
                continue
            _cyr_full: str
            try:
                _cyr_full = raw_bytes.decode(_cyr_cand[0])
            except UnicodeDecodeError:
                continue
            if _CYRILLIC_WORD_PRE_RE.search(_cyr_full) and _has_word_like_cyrillic_run(
                _cyr_full, raw_bytes, allow_rule1=False
            ):
                best_enc = _cyr_cand[0]
                _best_sample = _cyr_cand[1]
                best_ratio = _cyr_cand[2]
                break

    # F-88: After selecting the winning encoding, re-decode the full
    # content from raw_bytes rather than returning the stored sample.
    best_content = raw_bytes.decode(best_enc).lstrip("\ufeff")

    # UTF-8 preference: when UTF-8 decodes successfully and its word-char
    # ratio is close to the best candidate's (within 2%), prefer UTF-8.
    # This prevents single-byte encodings (e.g. cp1251) from "winning" by
    # a tiny margin on valid UTF-8 files — every byte maps to a printable
    # character in those encodings, so their word-char ratios can be
    # marginally higher even when the content is actually UTF-8 Cyrillic.
    # The 2% threshold is conservative: real encoding mismatches produce
    # much larger ratio gaps (typically >10%), so this won't override a
    # genuinely better match.
    if best_enc != "utf-8":
        for enc_candidate in candidates:
            if enc_candidate[0] == "utf-8":
                utf8_ratio = enc_candidate[2]
                if best_ratio - utf8_ratio < 0.02:
                    best_enc = enc_candidate[0]
                    best_content = raw_bytes.decode(best_enc).lstrip("\ufeff")
                    best_ratio = utf8_ratio
                break

    if len(candidates) > 1 and best_enc != candidates[1][0]:
        logger.debug(
            "Selected encoding '%s' (%.1f%% word chars) over '%s' (%.1f%%) for %s",
            best_enc,
            best_ratio * 100,
            candidates[1][0],
            candidates[1][2] * 100,
            file_path,
        )

    return best_enc, best_content


def read_with_encoding(
    file_path: Path,
    encoding: str | None = None,
    max_file_size: int | None = None,
) -> tuple[str, str]:
    """Read file content with encoding detection and fallback chain.

    Reads raw bytes once via ``file_path.read_bytes()``, then decodes
    in memory through the fallback chain. This avoids repeated full-file
    I/O when encoding detection fails (was up to 7 reads; now always 1).

    Args:
        file_path: Path to the file.
        encoding: Explicit encoding override. If None, auto-detected.
        max_file_size: Maximum file size in bytes. If the file exceeds
            this limit, a ValueError is raised.  Defaults to
            DEFAULT_MAX_FILE_SIZE (500 MB) to protect against unbounded
            memory consumption from reading and splitting large files.

    Returns:
        Tuple of (detected_encoding, file_content).

    Raises:
        LASEncodingError: If the path is not a regular file, if the explicit
            encoding parameter fails to decode the file, or if no encoding in
            the fallback chain works.
        ValueError: If file exceeds max_file_size.
    """
    # Guard against non-regular files (FIFOs, pipes, sockets, etc.).
    # Public entry points (read_las_file_as_object) also check is_file(),
    # but this module should be self-protecting when called directly.
    if not file_path.is_file():
        raise LASEncodingError(f"Cannot read {file_path}: not a regular file.")

    # F-ITER2-SEC-H01 + F-ITER2-SEC-M05: Apply default max file size when
    # the caller does not specify one.  Without this, Path.read_bytes()
    # and content.splitlines() both consume up to 2x file size in memory
    # with no bound, potentially OOM on large or malformed files.
    if max_file_size is None:
        max_file_size = DEFAULT_MAX_FILE_SIZE
    # F-95: Guard against negative or zero max_file_size which would
    # silently bypass the resource exhaustion check below.
    elif max_file_size <= 0:
        raise ValueError(f"max_file_size must be positive or None, got {max_file_size}")

    file_size = file_path.stat().st_size
    if file_size > max_file_size:
        raise ValueError(
            f"File size ({file_size} bytes) exceeds maximum allowed "
            f"({max_file_size} bytes): {file_path}"
        )

    # Read raw bytes once — all subsequent decoding happens in memory.
    raw_bytes = file_path.read_bytes()

    # Strip UTF-8 BOM from raw bytes before encoding detection and decoding.
    # The BOM (\xef\xbb\xbf) is a 3-byte sequence at the start of some UTF-8
    # files.  If left in place, decoding as cp1251 produces extra word
    # characters (п»ї) that inflate the quality score, causing cp1251 to
    # out-compete UTF-8 in the quality-based selection below.
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raw_bytes = raw_bytes[3:]

    # Strip UTF-16/32 BOM prefixes and decode immediately — these BOMs
    # definitively identify the encoding.  Without this, UTF-16/32 files
    # decode as garbage through single-byte fallbacks when chardet is
    # unavailable.  Checking 4-byte prefixes first avoids false matches
    # where the UTF-16 LE BOM (\xff\xfe) is also the start of the
    # UTF-32 LE BOM (\xff\xfe\x00\x00).
    #
    # Only run on files with enough content after BOM stripping to be
    # meaningful (UTF-16: >= 4 bytes content → >= 6 bytes total;
    # UTF-32: >= 4 bytes content → >= 8 bytes total).  This avoids
    # mistaking tiny test payloads (e.g. b"\\xff\\xfe\\x00\\x01") for
    # valid BOM-prefixed files.
    _bom = None
    if len(raw_bytes) >= 8 and raw_bytes.startswith(b"\xff\xfe\x00\x00"):
        _bom = ("utf-32-le", 4)
    elif len(raw_bytes) >= 8 and raw_bytes.startswith(b"\x00\x00\xfe\xff"):
        _bom = ("utf-32-be", 4)
    elif len(raw_bytes) >= 6 and raw_bytes.startswith(b"\xff\xfe"):
        _bom = ("utf-16-le", 2)
    elif len(raw_bytes) >= 6 and raw_bytes.startswith(b"\xfe\xff"):
        _bom = ("utf-16-be", 2)
    if _bom is not None:
        _bom_enc, _bom_len = _bom
        try:
            content = raw_bytes[_bom_len:].decode(_bom_enc)
            return _bom_enc, content.lstrip("\ufeff")
        except (UnicodeDecodeError, LookupError) as e:
            raise LASEncodingError(
                f"Failed to decode {file_path} with encoding '{_bom_enc}': {e}"
            ) from e

    if encoding is not None:
        try:
            content = raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError) as e:
            # F-94: Invalid encoding names (e.g. "nonexistent-enc")
            # raise LookupError from Python's codecs machinery, which
            # previously bypassed the pylasdev exception hierarchy.
            # Wrap in LASEncodingError for consistent error reporting.
            raise LASEncodingError(
                f"Failed to decode {file_path} with encoding '{encoding}': {e}"
            ) from e
        content = content.lstrip("\ufeff")
        return encoding, content

    # Try auto-detection from the already-read bytes
    detected = _detect_encoding_from_bytes(raw_bytes[:50_000])
    # ENC-02 (a): re-read the raw chardet confidence so a high-confidence
    # detection can be honored over the ratio sort (the name-only path
    # discards it).  When chardet is unavailable or low-confidence this
    # returns 0.0 and the priority never fires.
    detected_confidence = _detect_confidence_from_bytes(raw_bytes[:50_000])

    # F-ITER2-D4b-M09: Use quality-based selection instead of first-wins
    # fallback chain.  Single-byte Cyrillic encodings (cp1251, cp866) both
    # decode any byte sequence, so the first that succeeds may be wrong.
    # By comparing word-character ratios across all candidates, we select
    # the encoding that produces the most plausible text (highest proportion
    # of alphanumeric characters — real LAS files have ~50-80%).
    return _decode_best_quality(
        raw_bytes,
        file_path,
        detected,
        FALLBACK_ENCODINGS,
        detected_confidence=detected_confidence,
    )
