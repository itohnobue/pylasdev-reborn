"""Custom exceptions for pylasdev."""

from __future__ import annotations


class PylasdevError(Exception):
    """Base exception for all pylasdev errors."""


class LASReadError(PylasdevError):
    """Raised when a LAS file cannot be read (file not found, permissions)."""


class LASWriteError(PylasdevError):
    """Raised when a LAS file cannot be written."""


class LASParseError(PylasdevError):
    """Raised when LAS file content cannot be parsed."""


class LASVersionError(PylasdevError):
    """Raised when an unsupported LAS version is encountered.

    Note: the library itself issues ``warnings.warn()`` for LAS versions
    beyond 3.0 and continues processing.  This exception is provided so
    that user code can enforce a strict version policy when needed.
    """


class LASEncodingError(PylasdevError):
    """Raised when file encoding cannot be determined or decoded."""


class DEVReadError(PylasdevError):
    """Raised when a DEV file cannot be read or parsed."""
