"""Encoding detection utilities for LAS/DEV files.

Geoscience files commonly use:
- UTF-8 (modern files)
- CP1252 / Latin-1 (Western European)
- CP866 (Russian DOS encoding)
- CP1251 (Russian Windows encoding)
"""

from __future__ import annotations

from pathlib import Path

try:
    import chardet

    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

# Ordered by likelihood in Russian geoscience context
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
        result = chardet.detect(raw)
        if result["confidence"] and result["confidence"] > 0.7:
            return result["encoding"] or "utf-8"

    return "utf-8"


def read_with_encoding(
    file_path: Path,
    encoding: str | None = None,
    max_file_size: int | None = None,
) -> tuple[str, str]:
    """Read file content with encoding detection and fallback chain.

    Args:
        file_path: Path to the file.
        encoding: Explicit encoding override. If None, auto-detected.
        max_file_size: Optional maximum file size in bytes. If the file
            exceeds this limit, a ValueError is raised.

    Returns:
        Tuple of (detected_encoding, file_content).

    Raises:
        UnicodeDecodeError: If no encoding in the fallback chain works.
        ValueError: If file exceeds max_file_size.
    """
    if max_file_size is not None:
        file_size = file_path.stat().st_size
        if file_size > max_file_size:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum allowed "
                f"({max_file_size} bytes): {file_path}"
            )

    if encoding is not None:
        content = file_path.read_text(encoding=encoding)
        return encoding, content

    # Try auto-detection first
    detected = detect_encoding(file_path)
    try:
        content = file_path.read_text(encoding=detected)
        return detected, content
    except UnicodeDecodeError:
        pass

    # Fallback chain
    for enc in FALLBACK_ENCODINGS:
        try:
            content = file_path.read_text(encoding=enc)
            return enc, content
        except UnicodeDecodeError:
            continue

    # Last resort
    content = file_path.read_text(encoding="utf-8", errors="replace")
    return "utf-8", content
