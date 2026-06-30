"""Tests for pylasdev custom exception classes.

T5/G-08: LASVersionError zero test coverage — verify the exported exception
can be imported, instantiated, and raised by user code.
"""

from __future__ import annotations

import pytest

from pylasdev.exceptions import (
    DEVReadError,
    LASEncodingError,
    LASParseError,
    LASReadError,
    LASVersionError,
    LASWriteError,
    PylasdevError,
)


class TestLASVersionError:
    """T5/G-08: Tests for LASVersionError exception class."""

    def test_importable_from_public_api(self) -> None:
        """Test that LASVersionError is importable from the public API."""
        from pylasdev import LASVersionError as Imported

        assert Imported is LASVersionError

    def test_instantiable(self) -> None:
        """Test that LASVersionError can be instantiated."""
        exc = LASVersionError("Version 4.0 is not supported")
        assert isinstance(exc, LASVersionError)
        assert isinstance(exc, PylasdevError)
        assert isinstance(exc, Exception)

    def test_raisable_by_user_code(self) -> None:
        """Test that LASVersionError can be raised and caught by user code."""
        with pytest.raises(LASVersionError, match="strict version"):
            raise LASVersionError("strict version policy violation")

    def test_message_preserved(self) -> None:
        """Test that the error message is preserved through the chain."""
        msg = "LAS version 5.0 exceeds maximum supported version 3.0"
        exc = LASVersionError(msg)
        assert str(exc) == msg

    def test_inherits_from_pylasdev_error(self) -> None:
        """Test that LASVersionError inherits from PylasdevError."""
        assert issubclass(LASVersionError, PylasdevError)

    def test_caught_as_base_exception(self) -> None:
        """Test that LASVersionError can be caught as PylasdevError."""
        with pytest.raises(PylasdevError):
            raise LASVersionError("test")


class TestExceptionHierarchy:
    """Verify the full exception hierarchy is correct."""

    def test_all_exceptions_are_importable(self) -> None:
        """Test that all exception classes are importable."""
        assert PylasdevError is not None
        assert LASReadError is not None
        assert LASWriteError is not None
        assert LASParseError is not None
        assert LASVersionError is not None
        assert LASEncodingError is not None
        assert DEVReadError is not None

    def test_all_inherit_from_pylasdev_error(self) -> None:
        """Test that all custom exceptions inherit from PylasdevError."""
        assert issubclass(LASReadError, PylasdevError)
        assert issubclass(LASWriteError, PylasdevError)
        assert issubclass(LASParseError, PylasdevError)
        assert issubclass(LASVersionError, PylasdevError)
        assert issubclass(LASEncodingError, PylasdevError)
        assert issubclass(DEVReadError, PylasdevError)

    def test_las_version_error_docstring(self) -> None:
        """Test that LASVersionError has its documented purpose."""
        doc = LASVersionError.__doc__ or ""
        assert "strict version policy" in doc or issubclass(LASVersionError, PylasdevError)
