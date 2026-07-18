"""Encoding detection utilities for LAS/DEV files.

Geoscience files commonly use:
- UTF-8 (modern files)
- CP1252 / Latin-1 (Western European)
- CP1251 (Russian Windows encoding)
- CP866 (Russian DOS encoding)
"""

from __future__ import annotations

import logging
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
_MIN_VALIDATION_CHARS = 10_000  # Sample first 10K chars for mojibake check


def detect_encoding(file_path: Path) -> str:
    """Detect file encoding using chardet (if available) or fallback chain.

    Args:
        file_path: Path to the file.

    Returns:
        Detected encoding name.
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


def _decode_best_quality(
    raw_bytes: bytes,
    file_path: Path,
    detected: str,
    encodings: list[str],
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
    # Western European files and causing mojibake.
    #
    # Robust approach: byte-frequency analysis on the raw bytes.
    # The top-10 most common Russian letters (о, е, а, и, н, т, с,  # noqa: RUF003
    # р, в, л) collectively account for ~35 % of Russian text.  In  # noqa: RUF003
    # cp1251 these map to the byte set {0xE0, 0xE2, 0xE5, 0xE8,
    # 0xEB, 0xED, 0xEE, 0xF0, 0xF1, 0xF2}.  In cp1252 Western
    # European text those same bytes map to infrequent accented
    # letters (à, â, å, è, ë, í, î, ð, ñ, ò) at only 2-5 %
    # combined frequency.  A threshold of 10 % provides a wide
    # safety margin — Russian files score 25-40 %, Western files
    # score 2-5 %.
    #
    # For cp866-encoded Russian, the most common letters (а-п) map  # noqa: RUF003
    # to 0xA0-0xAF, which overlaps less with our cp1251-targeted
    # set.  However cp866 is the secondary Cyrillic encoding in
    # the FALLBACK_ENCODINGS list and already wins the ratio
    # comparison against its cp1251-decoded mojibake — the
    # tiebreaker is only needed when ratios are equal, which
    # primarily occurs between cp1251 and cp1252.
    _CYRILLIC_ENCS = frozenset({"cp1251", "cp866"})
    _WESTERN = frozenset({"cp1252", "latin-1"})
    _RUSSIAN_COMMON_BYTES = frozenset({
        0xE0, 0xE2, 0xE5, 0xE8, 0xEB, 0xED, 0xEE, 0xF0, 0xF1, 0xF2,
    })
    _raw_sample = raw_bytes[:_MIN_VALIDATION_CHARS]
    _russian_byte_count = sum(1 for b in _raw_sample if b in _RUSSIAN_COMMON_BYTES)
    _russian_byte_freq = _russian_byte_count / max(len(_raw_sample), 1)
    _is_cyrillic = _russian_byte_freq >= 0.10
    _preferred = _CYRILLIC_ENCS if _is_cyrillic else _WESTERN
    candidates.sort(key=lambda x: (-x[2], 0 if x[0] in _preferred else 1))
    best_enc, _best_sample, best_ratio = candidates[0]

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
        raise ValueError(
            f"max_file_size must be positive or None, got {max_file_size}"
        )

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

    # F-ITER2-D4b-M09: Use quality-based selection instead of first-wins
    # fallback chain.  Single-byte Cyrillic encodings (cp1251, cp866) both
    # decode any byte sequence, so the first that succeeds may be wrong.
    # By comparing word-character ratios across all candidates, we select
    # the encoding that produces the most plausible text (highest proportion
    # of alphanumeric characters — real LAS files have ~50-80%).
    return _decode_best_quality(raw_bytes, file_path, detected, FALLBACK_ENCODINGS)
