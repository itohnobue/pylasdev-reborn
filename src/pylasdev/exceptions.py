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
    """Raised when an unsupported LAS version is encountered."""


class LASEncodingError(PylasdevError):
    """Raised when file encoding cannot be determined or decoded."""


class DEVReadError(PylasdevError):
    """Raised when a DEV file cannot be read or parsed."""
