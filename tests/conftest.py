"""Pytest fixtures for pylasdev tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Test data at repository root
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory."""
    return TEST_DATA_DIR


@pytest.fixture
def all_las_files() -> list[Path]:
    """All LAS test files in test_data/.

    HARDEN (test-register row, F-8): returning an EMPTY list when the
    directory is missing made five corpus consumers vacuously pass
    (``for f in all_las_files`` iterates zero times).  The fixture now
    fails loudly instead, so a missing/renamed test_data directory is a
    test failure and not silent coverage loss.
    """
    files = sorted(TEST_DATA_DIR.glob("*.las")) if TEST_DATA_DIR.exists() else []
    assert files, f"no LAS fixtures found under {TEST_DATA_DIR} (corpus consumers would be vacuous)"
    return files


@pytest.fixture
def all_dev_files() -> list[Path]:
    """All DEV test files in test_data/.

    NOTE: Only one real .dev file (sample.dev) exists in test_data/.
    More DUG Insight exports and headerless DEV files are needed
    for comprehensive format coverage (F-T2-M05).

    HARDEN (test-register row, F-8): an empty list is a failure, not a
    vacuous pass — see all_las_files.
    """
    files = sorted(TEST_DATA_DIR.glob("*.dev")) if TEST_DATA_DIR.exists() else []
    assert files, f"no DEV fixtures found under {TEST_DATA_DIR} (corpus consumers would be vacuous)"
    return files


@pytest.fixture
def sample_las_data() -> dict[str, Any]:
    """Sample LAS data dictionary for testing write/roundtrip.

    HARDEN (test-register row, F-9): the well section carries every
    mandatory LAS 2.0 slot (STRT/STOP/STEP/NULL/COMP/WELL/FLD/LOC/
    PROV/SRVC/DATE/UWI) so consumers exercise the normal write path
    rather than the missing-mandatory-field warning path.
    """
    return {
        "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
        "well": {
            "STRT": "1670.0",
            "STOP": "1660.0",
            "STEP": "-0.125",
            "NULL": "-999.25",
            "COMP": "Test Company",
            "WELL": "Test Well #1",
            "FLD": "Test Field",
            "LOC": "12-34-56-78W5",
            "PROV": "Test Province",
            "SRVC": "Test Service Co",
            "DATE": "01/01/2020",
            "UWI": "12-34-56-78W5-00",
        },
        "parameters": {"BHT": "35.5", "BS": "200.0"},
        "logs": {
            "DEPT": np.array([1670.0, 1669.875, 1669.75]),
            "DT": np.array([123.45, 123.50, 123.55]),
            "RHOB": np.array([2550.0, 2551.0, 2552.0]),
        },
        "curves_order": ["DEPT", "DT", "RHOB"],
        "curves": [
            {"mnemonic": "DEPT", "unit": "M", "description": "Depth"},
            {"mnemonic": "DT", "unit": "US/M", "description": "Sonic Transit Time"},
            {"mnemonic": "RHOB", "unit": "K/M3", "description": "Bulk Density"},
        ],
    }
