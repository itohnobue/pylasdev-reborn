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
    """All LAS test files in test_data/."""
    if TEST_DATA_DIR.exists():
        return sorted(TEST_DATA_DIR.glob("*.las"))
    return []


@pytest.fixture
def all_dev_files() -> list[Path]:
    """All DEV test files in test_data/."""
    if TEST_DATA_DIR.exists():
        return sorted(TEST_DATA_DIR.glob("*.dev"))
    return []


@pytest.fixture
def sample_las_data() -> dict[str, Any]:
    """Sample LAS data dictionary for testing write/roundtrip."""
    return {
        "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
        "well": {
            "STRT": "1670.0",
            "STOP": "1660.0",
            "STEP": "-0.125",
            "NULL": "-999.25",
            "COMP": "Test Company",
            "WELL": "Test Well #1",
        },
        "parameters": {"BHT": "35.5", "BS": "200.0"},
        "logs": {
            "DEPT": np.array([1670.0, 1669.875, 1669.75]),
            "DT": np.array([123.45, 123.50, 123.55]),
            "RHOB": np.array([2550.0, 2551.0, 2552.0]),
        },
        "curves_order": ["DEPT", "DT", "RHOB"],
    }
