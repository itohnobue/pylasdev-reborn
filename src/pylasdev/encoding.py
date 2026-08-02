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
# F-18: adjacency window for the "№" (0xB9) confirmation rule — a genuine
# "СКВ №1" places the marker within 1-3 bytes of the run (space/punctuation  # noqa: RUF003
# + 0xB9), so 8 bytes on either side comfortably covers real-world labels
# while keeping a far-away footnote marker out of scope.
_NUMERO_ADJACENCY_WINDOW = 8

# ENC-02: Windows-1252 smart-punctuation byte class (0x91-0x97: ' " " – — …).  # noqa: RUF003
# These bytes decode to Cyrillic ALPHANUMERICS under cp866 (e.g. 0x96 → 'Ц'),
# inflating cp866's word-char ratio above a genuine Western page whenever the
# file contains typographic punctuation — the root of the Western→cp866 flip.
_SMART_PUNCT_BYTES = bytes((0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97))

# ENC-02 (a): material-margin for the detected-encoding priority.  Real
# encoding mismatches produce ratio gaps >10%; a <5% gap is within the
# near-tie band and the high-confidence chardet answer should be honored.
_DETECTED_ENCODING_MARGIN = 0.05


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
    cyrillic_class = b"[" + cyrillic_bytes + b"]"
    strong_class = b"[" + strong_bytes + b"]"
    return (
        re.compile(cyrillic_class + b"{" + str(_CYRILLIC_RUN_CONFIRM).encode("ascii") + b",}"),
        re.compile(cyrillic_class + b"{3,}"),
        re.compile(cyrillic_class + b"*" + strong_class + cyrillic_class + b"*"),
    )


_CYRILLIC_RUN_GE_RE, _CYRILLIC_RUN_3_RE, _STRONG_CYRILLIC_RUN_RE = _build_cyrillic_run_regexes()


def _window_has_numero_prefix(window: bytes) -> bool:
    """Return True if *window* contains a 0xB9 byte that is a Cyrillic "№".

    Byte 0xB9 is "№" in cp1251 but "¹" (superscript one) in cp1252 — a
    Western footnote "¹" adjacent to an accented run ("Nota¹ Ñáñez") must
    NOT confirm Cyrillic (ENC-01).  The Russian typographic convention
    always places a NUMBER after the № (well labels like "№1", "№ 2"), so
    a 0xB9 counts as "№" only when followed — after optional whitespace —
    by an ASCII digit.  A Western "¹" is an ordinal/footnote marker, not a
    "№ <number>" prefix, so the digit-follow constraint disambiguates.
    """
    idx = 0
    while True:
        idx = window.find(0xB9, idx)
        if idx == -1:
            return False
        nxt = idx + 1
        while nxt < len(window) and window[nxt] in (0x20, 0x09):  # space/tab
            nxt += 1
        if nxt < len(window) and 0x30 <= window[nxt] <= 0x39:
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
    # candidate is NOT a Western encoding, a genuine Western file with
    # Windows-1252 smart punctuation (0x91-0x97) is being misdecoded:
    # those bytes are Cyrillic alphanumerics under cp866 (e.g. 0x96 →
    # 'Ц') but punctuation under cp1252/latin-1, so cp866's word-char
    # ratio is inflated above the correct Western page.  Prefer the best
    # Western candidate when the ratio gap — minus the smart-punct
    # artifact (each such byte inflates the non-Western ratio by exactly
    # K/N) — is within the near-tie margin, mirroring the M-81 №-artifact
    # subtraction.
    if not _is_cyrillic and best_enc not in _WESTERN:
        _best_western: tuple[str, str, float] | None = None
        for _cand in candidates:
            if _cand[0] in _WESTERN:
                _best_western = _cand
                break
        if _best_western is not None:
            _west_enc, _west_sample, _west_ratio = _best_western
            _ratio_window = raw_bytes[:_MIN_VALIDATION_CHARS]
            _smart_punct_artifact = sum(_ratio_window.count(b) for b in _SMART_PUNCT_BYTES) / max(
                len(_ratio_window), 1
            )
            if (
                best_ratio - _west_ratio - _smart_punct_artifact
                <= _NEAR_TIE_CYRILLIC_MARGIN * best_ratio
            ):
                best_enc = _west_enc
                _best_sample = _west_sample
                best_ratio = _west_ratio

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
