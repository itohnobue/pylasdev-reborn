"""Encoding detection utilities for LAS/DEV files.

Geoscience files commonly use:
- UTF-8 (modern files)
- CP1252 / Latin-1 (Western European)
- CP866 (Russian DOS encoding)
- CP1251 (Russian Windows encoding)
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

# Ordered by likelihood in Russian geoscience context.
# latin-1 is used as the terminal fallback because it can decode any byte
# sequence (every byte 0x00-0xFF maps to a valid character), guaranteeing
# that read_with_encoding() always returns content even for files with
# unknown or corrupted encodings. This design choice prioritizes data
# recovery over strictness — geoscience files often have mixed encodings.
FALLBACK_ENCODINGS = ["utf-8", "cp1251", "cp1252", "cp866", "latin-1"]


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
        max_file_size: Optional maximum file size in bytes. If the file
            exceeds this limit, a ValueError is raised.

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

    if max_file_size is not None:
        file_size = file_path.stat().st_size
        if file_size > max_file_size:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum allowed "
                f"({max_file_size} bytes): {file_path}"
            )

    # Read raw bytes once — all subsequent decoding happens in memory.
    raw_bytes = file_path.read_bytes()

    if encoding is not None:
        try:
            content = raw_bytes.decode(encoding)
        except UnicodeDecodeError as e:
            raise LASEncodingError(
                f"Failed to decode {file_path} with encoding '{encoding}': {e}"
            ) from e
        content = content.lstrip("\ufeff")
        return encoding, content

    # Try auto-detection from the already-read bytes
    detected = _detect_encoding_from_bytes(raw_bytes[:50_000])
    try:
        content = raw_bytes.decode(detected)
        content = content.lstrip("\ufeff")
        return detected, content
    except UnicodeDecodeError:
        logger.debug(
            "Failed to decode %s with detected encoding %s, trying fallback chain",
            file_path,
            detected,
        )

    # Fallback chain — decode the same raw_bytes in memory.
    for enc in FALLBACK_ENCODINGS:
        try:
            content = raw_bytes.decode(enc)
            content = content.lstrip("\ufeff")
            return enc, content
        except UnicodeDecodeError:
            continue

    # The fallback chain always succeeds because latin-1 decodes any byte sequence.
    # This point should be unreachable, but if it is reached (e.g. because someone
    # removed latin-1 from the fallback chain), raise a proper error instead of
    # silently substituting replacement characters.
    raise LASEncodingError(f"Failed to decode {file_path} with any encoding in the fallback chain.")
