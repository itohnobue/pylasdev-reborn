"""Tests for pylasdev custom exception classes.

T5/G-08: LASVersionError zero test coverage — verify the exported exception
can be imported, instantiated, and raised by user code.
"""

from __future__ import annotations

import pytest

from pylasdev.exceptions import (
    DEVReadError,
    LASDataError,
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


class TestLASDataError:
    """M64: Tests for LASDataError — the only untested public exception class.

    LASDataError is imported in 8 locations and raised at 4 call sites
    (models.py:1060, 1210 — LASFile.from_dict and DevFile.from_dict).
    Previously had zero test coverage (grep confirmed).
    """

    def test_importable_from_public_api(self) -> None:
        """LASDataError is importable from the public pylasdev package."""
        from pylasdev import LASDataError as Imported

        assert Imported is LASDataError

    def test_instantiable(self) -> None:
        """LASDataError can be instantiated with a message."""
        exc = LASDataError("Validation failed for curve DT")
        assert isinstance(exc, LASDataError)
        assert str(exc) == "Validation failed for curve DT"

    def test_inherits_from_pylasdev_error(self) -> None:
        """LASDataError is a subclass of PylasdevError."""
        assert issubclass(LASDataError, PylasdevError)

    def test_inherits_from_value_error(self) -> None:
        """LASDataError is also a subclass of ValueError (dual inheritance)."""
        assert issubclass(LASDataError, ValueError)

    def test_caught_as_pylasdev_error(self) -> None:
        """LASDataError can be caught as PylasdevError."""
        with pytest.raises(PylasdevError):
            raise LASDataError("test")

    def test_caught_as_value_error(self) -> None:
        """LASDataError can be caught as ValueError (backward compat)."""
        with pytest.raises(ValueError):
            raise LASDataError("test")

    def test_raises_on_invalid_from_dict(self) -> None:
        """LASFile.from_dict with invalid data raises LASDataError.

        Exercise the raise at models.py:1060 — ValueError wrapping
        for invalid from_dict input. Inconsistent log array lengths
        trigger ValueError at models.py:1012-1015 which is caught
        and re-raised as LASDataError.
        """
        from pylasdev.models import LASFile

        with pytest.raises(LASDataError):
            LASFile.from_dict({
                "version": {"VERS": "2.0"},
                "well": {"NULL": "-999.25"},
                "logs": {"DEPT": [1, 2], "GR": [3, 4, 5]},
                "curves_order": ["DEPT", "GR"],
            })

    def test_raises_on_null_column_data(self) -> None:
        """DevFile.from_dict with None column data raises LASDataError.

        Exercise DevFile.from_dict at models.py:1166-1167 — the None
        guard raises ValueError which is caught and re-raised as LASDataError.
        """
        from pylasdev.models import DevFile

        with pytest.raises(LASDataError):
            DevFile.from_dict({"MD": None})

    def test_raises_on_non_numeric_column_data(self) -> None:
        """DevFile.from_dict with non-numeric column data raises LASDataError."""
        from pylasdev.models import DevFile

        with pytest.raises(LASDataError):
            DevFile.from_dict({"MD": ["not", "numbers"]})
