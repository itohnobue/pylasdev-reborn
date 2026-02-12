# pylasdev Modernization — Comprehensive Implementation Plan

**Version:** 3.0 (Final)
**Date:** 2026-02-12
**Status:** Ready for Implementation
**Based on:** Best elements from 4 specialist plans, verified against source code
**Reference Project:** hpgl-reborn (modernization approach)

---

## Executive Summary

**pylasdev** is a Python 2 package for reading/writing LAS (Log ASCII Standard) 1.2/2.0 well log files and DEV (deviation survey) files, used in the geoscience/oil industry. This plan transforms it into a modern Python 3.12+ package.

### Current vs Target

| Aspect | Current | Target |
|--------|---------|--------|
| Python | 2.x | 3.12+ |
| Packaging | setup.py | pyproject.toml + uv |
| Parser | PLY (lex-yacc) | Regex |
| Data Structures | Global dict | Dataclasses (class-based) |
| Testing | Manual scripts | pytest |
| Type Hints | None | Full coverage |
| Code Quality | None | ruff + mypy |
| Error Handling | Silent failures / `print` | Custom exception hierarchy |
| Encoding | None (implicit system encoding) | chardet + fallback chain |

### Why Modernize (Not Replace with lasio)?

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Modernize pylasdev** | Unique features (DEV files, 5,582-entry mnem_base), full control, small codebase | Effort required | **CHOSEN** |
| **Switch to lasio** | Industry standard, maintained | No DEV support, no mnem_base, different API | Not suitable |
| **Hybrid (lasio backend)** | Best of both | Dependency, complexity | Consider for LAS 3.0 later |

### Verified Codebase Metrics

| Metric | Value | Verification |
|--------|-------|-------------|
| Core package lines | **863** | `wc -l pylasdev/*.py` |
| Total lines (all .py) | **6,715** | `wc -l *.py pylasdev/*.py` |
| mnem_base.py | **5,582 lines (~106KB)** | `wc -l mnem_base.py` |
| Test data files | **27** (18 originals + 9 `.writed`) | `ls test_data/` |
| Python 2 `print` statements | **53** across 8 files | `grep` |
| `xrange()` calls | **5** (las_writer.py:24,42,54 + dev_reader.py:62 + las_test.py:64) | `grep` |
| `has_key()` calls | **1** (las_lex_pars2.py:73) | `grep` |
| `unicode()` calls | **2** (las_lex_pars2.py:105,107) | `grep` |
| Relative imports | **6** (4 in `__init__.py` + 2 in `las_reader.py`) | `grep` |
| eval/exec in PLY | **0** (no security risk) | `grep` |

---

## 1. LAS Format Specification Reference

### 1.1 LAS File Structure

LAS (Log ASCII Standard) is a line-based format maintained by the Canadian Well Logging Society (CWLS).

```
~VERSION INFORMATION     # ~V - Version info (VERS, WRAP, DLM)
~WELL INFORMATION        # ~W - Well metadata (STRT, STOP, STEP, NULL, COMP, WELL, etc.)
~CURVE INFORMATION       # ~C - Curve definitions (mnemonic.unit API_CODE : description)
~PARAMETER INFORMATION   # ~P - Drilling parameters (BHT, BS, FD, etc.)
~OTHER                   # ~O - Free-form text notes
~ASCII LOG DATA          # ~A - Numeric data (depth + curve values)
```

**Header line pattern:** `MNEMONIC.UNIT  VALUE : DESCRIPTION`

Example:
```
STRT.M        1670.0000 : START DEPTH
STOP.M        1660.0000 : STOP DEPTH
STEP.M          -0.1250 : STEP
NULL.          -999.2500 : NULL VALUE
```

### 1.2 LAS Version Differences

| Feature | LAS 1.2 (1989) | LAS 2.0 (1992) | LAS 3.0 (1999) |
|---------|----------------|----------------|----------------|
| Sections | ~V, ~W, ~C, ~A | + ~P, ~O | + ~D (definition), ~T (tops) |
| WRAP mode | Yes | Yes | **No** (not allowed) |
| DLM (delimiter) | SPACE only | SPACE only | SPACE, COMMA, TAB |
| Array channels | No | No | Yes: `NMR[1]`, `NMR[2]` |
| Zoned parameters | No | No | Yes: `\|` separator |
| Data type hints | No | No | `{F}` float, `{E}` scientific, `{S}` string |

**pylasdev currently supports:** LAS 1.2 and 2.0 only. LAS 3.0 parser exists (`las_lex_pars3.py`) but is incomplete/disabled.

### 1.3 Wrapped Mode

In wrapped mode (`WRAP YES`), the depth value appears alone on the first line, followed by data values spanning multiple lines:

```
~A
 1670.0000                          # depth value alone
  123.45  2550.0  0.45  3.12       # curve values (may span multiple lines)
  987.65  -999.25
 1669.8750                          # next depth
  123.50  2551.0  0.46  3.11
  987.70  -999.25
```

**Critical implementation detail:** The current code (`las_line_reader.py:35-46`) uses a `depth_line` flag to detect the first line of each depth step. This is the correct approach. `i % curve_count` does NOT work because lines can contain varying numbers of values.

### 1.4 DEV (Deviation Survey) Format

Not standardized — proprietary format from Petrel and similar software:

```
# Header comments with metadata
# Column names line
MD   X   Y   Z   TVD   DX   DY   AZIM   INCL   DLS
<numeric data rows>
```

### 1.5 Cyrillic Mnemonic Support

The current lexer (`las_lex_pars2.py:31`) uses `r'(?u)[_А-Яа-яa-zA-Z0-9\-]'` with `re.UNICODE` flag. This is essential because Russian oil industry LAS files use Cyrillic curve mnemonics (e.g., `ГК`, `НГК`, `БК`).

---

## 2. Current Architecture Analysis

### 2.1 File Structure

```
pylasdev/
├── pylasdev/                    # Main package (863 lines)
│   ├── __init__.py              # Package exports (4 lines, relative imports)
│   ├── las_reader.py            # Main LAS entry point (137 lines, 2-pass)
│   ├── las_lex_pars2.py         # LAS 1.2/2.0 PLY parser (226 lines)
│   ├── las_lex_pars3.py         # LAS 3.0 parser (232 lines, INCOMPLETE)
│   ├── las_line_reader.py       # ASCII data reader (47 lines)
│   ├── las_writer.py            # LAS file writer (62 lines)
│   ├── las_compare.py           # Dict comparison utility (88 lines)
│   └── dev_reader.py            # Deviation file reader (67 lines)
│
├── mnem_base.py                 # Curve name alias database (5,582 lines, ~106KB)
├── test_data/                   # Sample LAS/DEV files (27 files)
├── autotest.py                  # Main test script
├── las_test.py                  # LAS reading tests
├── dev_test.py                  # DEV reading tests
├── big_test.py                  # Performance test
├── parser.out                   # PLY generated (in .gitignore)
└── parsetab.py                  # PLY generated (in .gitignore)
```

### 2.2 Current Data Flow

```
read_las_file(filename, mnem_base=None)
    │
    ├─► Pass 1: Count ~A data lines + detect LAS version
    │       ├─► open(filename) ... file.readlines(100000) ... file.close()
    │       └─► PLY lexer/parser for ~V section only
    │
    └─► Pass 2: Parse all header sections + read data
            ├─► PLY yacc parser → global las_info dict
            │       ├─► ~V → las_info['version'] (stores VERS, WRAP, DLM)
            │       ├─► ~W → las_info['well'] (values as STRINGS)
            │       ├─► ~C → las_info['curves_order'] + pre-allocate numpy arrays
            │       ├─► ~P → las_info['parameters'] (values as strings)
            │       └─► ~O → skipped
            │
            └─► line_reader.read_line() → populate numpy arrays
                    ├─► Non-wrapped: direct index assignment (efficient)
                    └─► Wrapped: numpy.append() per value (O(n²) BUG)
```

### 2.3 Known Issues (Verified)

| Issue | Severity | Location | Detail |
|-------|----------|----------|--------|
| **Python 2 syntax** | CRITICAL | All files | 53 print stmts, 5 xrange, 1 has_key, 2 unicode, 6 relative imports |
| **Global mutable state** | HIGH | las_lex_pars2.py:11 | Module-level `las_info` dict, `gmnem_base`, `lines_count` — not thread-safe |
| **O(n^2) numpy.append** | HIGH | las_line_reader.py:37,43 | In wrapped mode, each append copies entire array |
| **Resource leaks** | HIGH | las_reader.py:27,59,90; las_writer.py:4,62; dev_reader.py:12,33 | `open()`/`close()` without context managers |
| **Writer data loss** | MEDIUM | las_writer.py:18,25 | Writes `.X` for ALL units, `X` for ALL descriptions |
| **No error handling** | MEDIUM | All files | Silent failures, bare `print` for errors |
| **No encoding handling** | MEDIUM | las_reader.py | Uses system default encoding |
| **Internal state leak** | LOW | las_lex_pars2.py:67 | `version['last_caption']` exposed in returned dict |
| **mnem_base duplicate keys** | LOW | mnem_base.py | e.g. `'AK': 'DT'` repeated — later silently overwrites |
| **PLY overkill** | MEDIUM | las_lex_pars2.py | 226 lines of lex/yacc for a simple line-based format |
| **LAS 3.0 incomplete** | LOW | las_lex_pars3.py | Parser exists but is disabled |

---

## 3. Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Build backend | **hatchling** | Pure Python, simpler than scikit-build-core (following hpgl-reborn) |
| Python version | **>=3.12** | Latest stable, 4+ year support |
| Parser | **Regex** | LAS format is line-based — PLY is overkill. Reduces ~450 lines to ~150 |
| Data structures | **dataclasses** | Type-safe, IDE support, backward-compatible via `to_dict()` |
| Package manager | **uv** | Fast, modern, dependency locking |
| Linter + formatter | **ruff** (both) | Replaces flake8, isort, black, pyupgrade in one tool |
| Type checker | **mypy** | Industry standard |
| Test framework | **pytest** | Modern, fixture support, parametrization |
| NumPy version | **>=1.24** | Avoids NumPy 2.0 breaking changes, broad compatibility |
| Encoding detection | **chardet** (optional) | Most common library; optional dependency with fallback chain |

---

## 4. New Project Structure

```
pylasdev/                          # Repository root
├── pyproject.toml                 # Single source of truth for config
├── uv.lock                        # Dependency lock file (COMMITTED, not gitignored)
├── README.md                      # Documentation
├── LICENSE                        # BSD-3-Clause
├── .gitignore                     # Python patterns
│
├── src/                           # Source layout (PEP 517)
│   └── pylasdev/
│       ├── __init__.py            # Public API + __all__
│       ├── models.py              # Dataclasses (LASFile, CurveDefinition, etc.)
│       ├── parser.py              # Regex-based LAS header parser
│       ├── reader.py              # LAS file reader (main entry point)
│       ├── data_reader.py         # ASCII data section reader
│       ├── writer.py              # LAS file writer
│       ├── dev_reader.py          # DEV file reader
│       ├── compare.py             # Dict/LASFile comparison utilities
│       ├── encoding.py            # Encoding detection + fallbacks
│       ├── exceptions.py          # Exception hierarchy
│       └── mnem_base.py           # Mnemonic database (moved from root)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures + test data paths
│   ├── test_parser.py             # Parser unit tests
│   ├── test_reader.py             # Reader tests (ported from las_test.py)
│   ├── test_writer.py             # Writer tests
│   ├── test_dev_reader.py         # DEV reader tests (ported from dev_test.py)
│   ├── test_encoding.py           # Encoding detection tests
│   ├── test_roundtrip.py          # Read-write-read consistency
│   ├── test_compare.py            # Comparison utility tests
│   └── test_mnem_base.py          # Mnemonic database tests (dedup verification)
│
└── test_data/                     # Sample files (kept at root)
    ├── *.las                      # LAS sample files
    ├── *.dev                      # DEV sample files
    └── *.writed                   # Previously-written comparison files
```

---

## 5. Files to Delete (Cleanup)

| File | Reason |
|------|--------|
| `pylasdev/las_lex_pars2.py` | Replaced by `src/pylasdev/parser.py` (regex) |
| `pylasdev/las_lex_pars3.py` | Incomplete LAS 3.0 parser, not functional |
| `pylasdev/las_line_reader.py` | Merged into `src/pylasdev/data_reader.py` |
| `pylasdev/las_reader.py` | Replaced by `src/pylasdev/reader.py` |
| `pylasdev/las_writer.py` | Replaced by `src/pylasdev/writer.py` |
| `pylasdev/las_compare.py` | Replaced by `src/pylasdev/compare.py` |
| `pylasdev/dev_reader.py` | Replaced by `src/pylasdev/dev_reader.py` |
| `pylasdev/__init__.py` | Replaced by `src/pylasdev/__init__.py` |
| `autotest.py` | Replaced by `tests/test_reader.py` |
| `las_test.py` | Replaced by `tests/test_reader.py` |
| `dev_test.py` | Replaced by `tests/test_dev_reader.py` |
| `big_test.py` | Replaced by `tests/test_reader.py` (slow marker) |
| `parser.out` | PLY generated file, no longer needed |
| `parsetab.py` | PLY generated file, no longer needed |
| `setup.py` (if exists) | Replaced by `pyproject.toml` |

---

## 6. Detailed Implementation

### Phase 1: Foundation Setup

#### 6.1 pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pylasdev"
version = "1.0.0"
description = "Python library for reading and writing LAS (Log ASCII Standard) and DEV (deviation) well log files"
readme = "README.md"
license = {text = "BSD-3-Clause"}
requires-python = ">=3.12"
authors = [
    {name = "Artur Muharlyamov", email = "muharlyamovar@ufanipi.ru"},
]
keywords = ["las", "well-log", "dev", "geoscience", "petrophysics"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering",
]

dependencies = [
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "mypy>=1.10",
    "ruff>=0.5.0",
    "chardet>=5.0",
]
pandas = ["pandas>=2.0"]
all = ["pylasdev[dev,pandas]"]

[project.urls]
Homepage = "https://github.com/user/pylasdev"

# UV configuration
[tool.uv]
package = true

# Ruff configuration
[tool.ruff]
target-version = "py312"
line-length = 100
src = ["src"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "C4",     # flake8-comprehensions
    "B",      # flake8-bugbear
    "UP",     # pyupgrade
    "RUF",    # ruff-specific
]
ignore = [
    "E501",   # line too long (handled by formatter)
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # unused imports (re-exports)

[tool.ruff.lint.isort]
known-first-party = ["pylasdev"]

# MyPy configuration
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true

[[tool.mypy.overrides]]
module = ["numpy.*", "chardet.*"]
ignore_missing_imports = true

# Pytest configuration
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = [
    "-ra",
    "--strict-markers",
    "--strict-config",
    "--showlocals",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks integration tests",
]

# Coverage configuration
[tool.coverage.run]
source = ["src/pylasdev"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
```

#### 6.2 .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
build/
dist/
*.egg-info/
*.egg

# Virtual environments
.venv/
venv/

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp

# Type checking
.mypy_cache/

# PLY legacy (remove these files from repo)
parser.out
parsetab.py
```

**Note:** `uv.lock` is intentionally NOT gitignored — it should be committed for reproducible builds.

---

### Phase 2: Exception Hierarchy

Create `src/pylasdev/exceptions.py` first since other modules import from it.

```python
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
```

---

### Phase 3: Data Models

Create `src/pylasdev/models.py`.

**Key corrections applied:**
- Well values stored as **strings** (matching current behavior, not floats)
- `from_dict()` fully implemented (not a stub)
- `original_mnemonic` field on CurveDefinition for tracking pre-normalization names

```python
"""Data models for LAS file structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class VersionSection:
    """LAS Version Information section (~V)."""
    vers: str = "2.0"
    wrap: str = "NO"
    dlm: str = "SPACE"

    def to_dict(self) -> dict[str, str]:
        """Convert to legacy dict format for backward compatibility."""
        return {
            "VERS": self.vers,
            "WRAP": self.wrap,
            "DLM": self.dlm,
        }


@dataclass
class WellSection:
    """LAS Well Information section (~W).

    All values are stored as strings to match original pylasdev behavior.
    The original code stores well values via: las_info['well'][mnemonic] = value.lstrip().rstrip()
    """
    entries: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        """Convert to legacy dict format."""
        return dict(self.entries)

    def __getitem__(self, key: str) -> str:
        return self.entries[key]

    def __setitem__(self, key: str, value: str) -> None:
        self.entries[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def get(self, key: str, default: str = "") -> str:
        return self.entries.get(key, default)


@dataclass
class CurveDefinition:
    """Single curve definition from ~C section."""
    mnemonic: str
    unit: str = ""
    api_code: str = ""
    description: str = ""
    original_mnemonic: str = ""  # Pre-normalization name (before mnem_base lookup)

    def to_dict(self) -> dict[str, str]:
        return {
            "mnemonic": self.mnemonic,
            "unit": self.unit,
            "api_code": self.api_code,
            "description": self.description,
        }


@dataclass
class ParameterEntry:
    """Single parameter entry from ~P section."""
    mnemonic: str
    unit: str = ""
    value: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "mnemonic": self.mnemonic,
            "unit": self.unit,
            "value": self.value,
            "description": self.description,
        }


@dataclass
class LASFile:
    """Complete LAS file data structure.

    This replaces the global las_info dict with a proper class.
    Backward compatibility is maintained via to_dict().
    """
    version: VersionSection = field(default_factory=VersionSection)
    well: WellSection = field(default_factory=WellSection)
    curves: list[CurveDefinition] = field(default_factory=list)
    parameters: list[ParameterEntry] = field(default_factory=list)
    other: str = ""  # ~O section free text
    logs: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    curves_order: list[str] = field(default_factory=list)
    source_file: str = ""
    encoding: str = "utf-8"

    def to_dict(self) -> dict[str, Any]:
        """Convert to legacy dict format for backward compatibility.

        Returns a dict matching the original pylasdev structure:
        {
            'version': {'VERS': '2.0', 'WRAP': 'NO', ...},
            'well': {'STRT': '1670.0', 'STOP': '1660.0', ...},  # strings!
            'parameters': {'BHT': '35.5', ...},
            'logs': {'DEPT': np.array([...]), ...},
            'curves_order': ['DEPT', 'DT', ...]
        }
        """
        params_dict: dict[str, str] = {}
        for p in self.parameters:
            params_dict[p.mnemonic] = p.value

        return {
            "version": self.version.to_dict(),
            "well": self.well.to_dict(),
            "parameters": params_dict,
            "logs": {k: v.copy() for k, v in self.logs.items()},
            "curves_order": list(self.curves_order),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LASFile:
        """Create LASFile from legacy dict format.

        Args:
            data: Dictionary in the original pylasdev format.

        Returns:
            LASFile instance populated from the dict.
        """
        las_file = cls()

        # Version section
        version = data.get("version", {})
        las_file.version = VersionSection(
            vers=str(version.get("VERS", "2.0")),
            wrap=str(version.get("WRAP", "NO")),
            dlm=str(version.get("DLM", "SPACE")),
        )

        # Well section (all values as strings)
        well = data.get("well", {})
        for key, value in well.items():
            las_file.well[key] = str(value)

        # Curves — reconstruct from curves_order
        curves_order = data.get("curves_order", [])
        las_file.curves_order = list(curves_order)
        for curve_name in curves_order:
            las_file.curves.append(CurveDefinition(mnemonic=curve_name))

        # Parameters (all values as strings)
        params = data.get("parameters", {})
        for mnemonic, value in params.items():
            las_file.parameters.append(ParameterEntry(
                mnemonic=mnemonic,
                value=str(value),
            ))

        # Logs (numpy arrays)
        logs = data.get("logs", {})
        for name, arr in logs.items():
            las_file.logs[name] = np.array(arr, dtype=np.float64)

        return las_file


@dataclass
class DevFile:
    """DEV (deviation survey) file data structure."""
    columns: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    column_order: list[str] = field(default_factory=list)
    source_file: str = ""
    encoding: str = "utf-8"

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        """Convert to legacy dict format (original pylasdev returns dict)."""
        return {k: v.copy() for k, v in self.columns.items()}
```

---

### Phase 4: Regex Parser

Create `src/pylasdev/parser.py`.

**Key corrections applied:**
- Regex uses `[\w\-]` which matches Cyrillic in Python 3 (Unicode `\w` by default)
- No duplicate `LASParseError` — imported from `exceptions.py`

```python
"""Regex-based LAS file parser replacing PLY.

The LAS format is line-based with a simple structure:
  MNEMONIC.UNIT  VALUE : DESCRIPTION

PLY (lex/yacc) is overkill for this. Regex reduces ~450 lines to ~150
while maintaining the same parsing capability.
"""

from __future__ import annotations

import re

from .exceptions import LASParseError
from .models import (
    CurveDefinition,
    LASFile,
    ParameterEntry,
    VersionSection,
    WellSection,
)

# Section header: line starting with ~, followed by section letter
SECTION_PATTERN = re.compile(r"^~([A-Za-z])")

# Data line pattern: MNEMONIC.UNIT  VALUE : DESCRIPTION
# Uses \w which matches Unicode (including Cyrillic) in Python 3
DATA_LINE_PATTERN = re.compile(
    r"^\s*"
    r"(?P<mnemonic>[\w\-]+)"          # mnemonic: word chars + hyphen (Cyrillic OK)
    r"\."                              # literal dot separator
    r"(?P<unit>[\w\-/]*)"             # unit: optional, can include /
    r"\s+"                             # whitespace separator
    r"(?P<value>[^:]*?)"              # value: everything up to colon
    r"\s*:\s*"                         # colon separator
    r"(?P<description>.*?)"           # description: rest of line
    r"\s*$"
)

# Simpler pattern for lines without description (value-only)
VALUE_ONLY_PATTERN = re.compile(
    r"^\s*"
    r"(?P<mnemonic>[\w\-]+)"
    r"\."
    r"(?P<unit>[\w\-/]*)"
    r"\s+"
    r"(?P<value>.+?)"
    r"\s*$"
)

COMMENT_PATTERN = re.compile(r"^\s*#")
EMPTY_PATTERN = re.compile(r"^\s*$")


class LASParser:
    """Regex-based LAS file parser.

    Encapsulates all parsing state in the instance (no global variables).
    Thread-safe: each instance maintains its own state.
    """

    SECTION_HANDLERS: dict[str, str] = {
        "V": "_parse_version",
        "W": "_parse_well",
        "C": "_parse_curve",
        "P": "_parse_parameter",
        "O": "_parse_other",
        "A": "_mark_ascii_start",
    }

    def __init__(self, mnem_base: dict[str, str] | None = None) -> None:
        """Initialize parser with optional mnemonic base.

        Args:
            mnem_base: Dictionary mapping alternate curve names to canonical names.
        """
        self.mnem_base = mnem_base or {}
        self._reset()

    def _reset(self) -> None:
        """Reset parser state for a new file."""
        self.las_file = LASFile()
        self._current_section: str | None = None
        self._line_number = 0
        self._wrap_mode = False
        self._data_line_count = 0

    def parse(self, content: str) -> LASFile:
        """Parse LAS file content string.

        Args:
            content: Full text content of a LAS file.

        Returns:
            LASFile object with parsed header data.
            Note: ASCII data (~A section) is NOT parsed here — see data_reader.

        Raises:
            LASParseError: If the file structure is invalid.
        """
        self._reset()

        lines = content.splitlines()
        self._pre_scan(lines)

        for i, line in enumerate(lines, 1):
            self._line_number = i
            self._parse_line(line)

        return self.las_file

    def _pre_scan(self, lines: list[str]) -> None:
        """Pre-scan to count ASCII data lines (needed for non-wrapped pre-allocation)."""
        in_ascii = False
        count = 0

        for line in lines:
            match = SECTION_PATTERN.match(line)
            if match:
                in_ascii = match.group(1).upper() == "A"
                continue
            if in_ascii and not COMMENT_PATTERN.match(line) and not EMPTY_PATTERN.match(line):
                count += 1

        self._data_line_count = count

    def _parse_line(self, line: str) -> None:
        """Route a single line to the appropriate section handler."""
        section_match = SECTION_PATTERN.match(line)
        if section_match:
            self._current_section = section_match.group(1).upper()
            return

        if COMMENT_PATTERN.match(line) or EMPTY_PATTERN.match(line):
            return

        if self._current_section:
            handler_name = self.SECTION_HANDLERS.get(self._current_section)
            if handler_name:
                getattr(self, handler_name)(line)

    def _match_data_line(self, line: str) -> re.Match[str] | None:
        """Try to match a header data line with colon, then without."""
        match = DATA_LINE_PATTERN.match(line)
        if match:
            return match
        return VALUE_ONLY_PATTERN.match(line)

    def _parse_version(self, line: str) -> None:
        """Parse ~V (version) section line."""
        match = self._match_data_line(line)
        if not match:
            return

        mnemonic = match.group("mnemonic").upper().strip()
        value = match.group("value").strip()

        if mnemonic == "VERS":
            self.las_file.version.vers = value
        elif mnemonic == "WRAP":
            self.las_file.version.wrap = value.upper()
            self._wrap_mode = value.upper() == "YES"
        elif mnemonic == "DLM":
            self.las_file.version.dlm = value

    def _parse_well(self, line: str) -> None:
        """Parse ~W (well information) section line.

        All values are stored as strings to match original behavior.
        """
        match = self._match_data_line(line)
        if not match:
            return

        mnemonic = match.group("mnemonic").upper().strip()
        value = match.group("value").strip()

        # Store as string (original behavior: las_info['well'][t[2]] = t[4].lstrip().rstrip())
        self.las_file.well[mnemonic] = value

    def _parse_curve(self, line: str) -> None:
        """Parse ~C (curve information) section line."""
        match = self._match_data_line(line)
        if not match:
            return

        raw_mnemonic = match.group("mnemonic").upper().strip()
        unit = match.group("unit") or ""
        description = match.group("description").strip() if "description" in match.groupdict() and match.group("description") else ""

        # Apply mnemonic normalization from mnem_base
        normalized = self.mnem_base.get(raw_mnemonic, raw_mnemonic)

        curve = CurveDefinition(
            mnemonic=normalized,
            unit=unit,
            description=description,
            original_mnemonic=raw_mnemonic if raw_mnemonic != normalized else "",
        )
        self.las_file.curves.append(curve)
        self.las_file.curves_order.append(normalized)

    def _parse_parameter(self, line: str) -> None:
        """Parse ~P (parameter) section line."""
        match = self._match_data_line(line)
        if not match:
            return

        param = ParameterEntry(
            mnemonic=match.group("mnemonic").upper().strip(),
            unit=match.group("unit") or "",
            value=match.group("value").strip(),
            description=match.group("description").strip() if "description" in match.groupdict() and match.group("description") else "",
        )
        self.las_file.parameters.append(param)

    def _parse_other(self, line: str) -> None:
        """Parse ~O (other) section — free-form text, accumulated."""
        self.las_file.other += line + "\n"

    def _mark_ascii_start(self, line: str) -> None:
        """Mark that we've entered ~A section. Actual data reading is in data_reader."""
        pass
```

---

### Phase 5: ASCII Data Reader

Create `src/pylasdev/data_reader.py`.

**Key corrections applied:**
- Wrapped mode uses `depth_line` flag protocol (matching `las_line_reader.py:35-46`)
- NOT `i % curve_count` which would misalign data
- List accumulation then `np.array()` at end (fixes O(n^2) numpy.append bug)
- Pre-allocation for non-wrapped mode (efficient — known size)

```python
"""ASCII data section reader for LAS files.

Handles both normal and wrapped modes.
Replaces las_line_reader.py with corrected wrapped-mode logic
and O(n) performance (vs O(n^2) numpy.append bug in original).
"""

from __future__ import annotations

import re

import numpy as np

from .models import LASFile


def read_ascii_data(content: str, las_file: LASFile, data_line_count: int) -> None:
    """Read the ~A (ASCII data) section and populate las_file.logs.

    Args:
        content: Full file content string.
        las_file: LASFile object with curves_order already populated.
        data_line_count: Number of data lines (from pre-scan).
    """
    curve_count = len(las_file.curves_order)
    if curve_count == 0:
        return

    lines = content.splitlines()
    in_ascii = False
    wrap_mode = las_file.version.wrap.upper() == "YES"

    if wrap_mode:
        _read_wrapped(lines, las_file, curve_count)
    else:
        _read_normal(lines, las_file, curve_count, data_line_count)


def _read_normal(
    lines: list[str],
    las_file: LASFile,
    curve_count: int,
    data_line_count: int,
) -> None:
    """Read non-wrapped ASCII data. One depth step per line.

    Pre-allocates numpy arrays (known size from pre-scan) and uses
    direct indexing — the most efficient approach for non-wrapped mode.
    This matches the original las_line_reader.py behavior for WRAP=NO.
    """
    # Pre-allocate arrays
    for curve_name in las_file.curves_order:
        las_file.logs[curve_name] = np.zeros(data_line_count, dtype=np.float64)

    in_ascii = False
    current_line = 0
    null_value = float(las_file.well.get("NULL", "-999.25"))

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("~A"):
            in_ascii = True
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        values = re.split(r"[\s\t]+", stripped)

        for i in range(min(len(values), curve_count)):
            try:
                las_file.logs[las_file.curves_order[i]][current_line] = float(values[i])
            except (ValueError, IndexError):
                las_file.logs[las_file.curves_order[i]][current_line] = null_value

        current_line += 1


def _read_wrapped(
    lines: list[str],
    las_file: LASFile,
    curve_count: int,
) -> None:
    """Read wrapped ASCII data using depth_line flag protocol.

    In wrapped mode:
    - The DEPTH value appears ALONE on its own line
    - Subsequent lines contain the remaining curve values
    - Once all curves for a depth step are read, the next depth line follows

    This matches the original las_line_reader.py:35-46 behavior.

    Uses list accumulation then np.array() at end to avoid the O(n^2)
    numpy.append bug in the original code.
    """
    # Accumulate into lists, convert to numpy at end
    data_lists: list[list[float]] = [[] for _ in range(curve_count)]

    in_ascii = False
    depth_line = True  # First data line is always a depth line
    counter = 0  # Tracks position within non-depth curves
    null_value = float(las_file.well.get("NULL", "-999.25"))

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("~A"):
            in_ascii = True
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        values = re.split(r"[\s\t]+", stripped)

        if depth_line:
            # Depth line: single value = depth for this step
            try:
                data_lists[0].append(float(values[0]))
            except (ValueError, IndexError):
                data_lists[0].append(null_value)
            depth_line = False
            counter = 0
        else:
            # Data lines: values for remaining curves
            for val_str in values:
                counter += 1
                try:
                    data_lists[counter].append(float(val_str))
                except (ValueError, IndexError):
                    if counter < curve_count:
                        data_lists[counter].append(null_value)

                if counter >= curve_count - 1:
                    # All curves for this depth step are complete
                    counter = 0
                    depth_line = True

    # Convert lists to numpy arrays
    for i, curve_name in enumerate(las_file.curves_order):
        las_file.logs[curve_name] = np.array(data_lists[i], dtype=np.float64)
```

---

### Phase 6: Encoding Detection

Create `src/pylasdev/encoding.py`.

```python
"""Encoding detection utilities for LAS/DEV files.

Geoscience files commonly use:
- UTF-8 (modern files)
- CP1252 / Latin-1 (Western European)
- CP866 (Russian DOS encoding — common in Russian oil industry)
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
) -> tuple[str, str]:
    """Read file content with encoding detection and fallback chain.

    Args:
        file_path: Path to the file.
        encoding: Explicit encoding override. If None, auto-detected.

    Returns:
        Tuple of (detected_encoding, file_content).

    Raises:
        UnicodeDecodeError: If no encoding in the fallback chain works.
    """
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
```

---

### Phase 7: LAS Reader (Main Entry Point)

Create `src/pylasdev/reader.py`.

```python
"""LAS file reader — main entry point.

Replaces las_reader.py with modern Python 3, proper encoding handling,
context managers, and no global state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data_reader import read_ascii_data
from .encoding import read_with_encoding
from .exceptions import LASReadError, LASVersionError
from .models import LASFile
from .parser import LASParser


def read_las_file(
    file_path: str | Path,
    mnem_base: dict[str, str] | None = None,
    encoding: str | None = None,
) -> dict[str, Any]:
    """Read a LAS file and return data dictionary.

    This is the main entry point, maintaining backward compatibility
    with the original pylasdev API (returns dict, not LASFile).

    Args:
        file_path: Path to LAS file.
        mnem_base: Optional dictionary for curve name normalization.
        encoding: Optional encoding override. If None, auto-detected.

    Returns:
        Dictionary with keys: version, well, parameters, logs, curves_order.
        Well values are strings. Log values are numpy arrays.

    Raises:
        LASReadError: If file cannot be read.
        LASParseError: If file content cannot be parsed.
        LASVersionError: If LAS version is not supported (>= 3.0).

    Example:
        >>> data = read_las_file("sample.las")
        >>> print(data['well']['WELL'])
        >>> print(data['logs']['DEPT'])
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise LASReadError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise LASReadError(f"Not a file: {file_path}")

    # Read with encoding detection
    detected_encoding, content = read_with_encoding(file_path, encoding)

    # Parse header sections
    parser = LASParser(mnem_base)
    las_file = parser.parse(content)
    las_file.source_file = str(file_path)
    las_file.encoding = detected_encoding

    # Check version
    try:
        vers = float(las_file.version.vers)
        if vers >= 3.0:
            raise LASVersionError(
                f"LAS version {las_file.version.vers} is not supported. "
                "Only LAS 1.2 and 2.0 are supported."
            )
    except ValueError:
        pass  # Non-numeric version string — let it through

    # Read ASCII data section
    read_ascii_data(content, las_file, parser._data_line_count)

    # Return legacy dict format for backward compatibility
    return las_file.to_dict()


def read_las_file_as_object(
    file_path: str | Path,
    mnem_base: dict[str, str] | None = None,
    encoding: str | None = None,
) -> LASFile:
    """Read a LAS file and return LASFile dataclass (new API).

    Same as read_las_file() but returns the LASFile object directly
    instead of converting to dict. Use this for richer metadata access.

    Args:
        file_path: Path to LAS file.
        mnem_base: Optional dictionary for curve name normalization.
        encoding: Optional encoding override.

    Returns:
        LASFile dataclass with full parsed data.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise LASReadError(f"File not found: {file_path}")

    detected_encoding, content = read_with_encoding(file_path, encoding)

    parser = LASParser(mnem_base)
    las_file = parser.parse(content)
    las_file.source_file = str(file_path)
    las_file.encoding = detected_encoding

    read_ascii_data(content, las_file, parser._data_line_count)

    return las_file
```

---

### Phase 8: LAS Writer

Create `src/pylasdev/writer.py`.

**Key corrections applied:**
- Preserves original units from CurveDefinition (not hardcoded `.M` or `.X`)
- Preserves descriptions from CurveDefinition
- Writes well values from WellSection entries
- Accepts both dict and LASFile input

```python
"""LAS file writer.

Replaces las_writer.py with proper metadata preservation.
The original writer destroyed units (wrote '.X') and descriptions (wrote 'X').
This version preserves the original metadata when available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .exceptions import LASWriteError
from .models import LASFile


def write_las_file(
    file_path: str | Path,
    las_data: dict[str, Any] | LASFile,
    encoding: str = "utf-8",
) -> None:
    """Write LAS data to file.

    Args:
        file_path: Output file path.
        las_data: LAS data as dict (legacy format) or LASFile object.
        encoding: Output file encoding (default: utf-8).

    Raises:
        LASWriteError: If file cannot be written.
    """
    file_path = Path(file_path)

    if isinstance(las_data, dict):
        las_file = LASFile.from_dict(las_data)
    else:
        las_file = las_data

    content = _generate_las_content(las_file)

    try:
        file_path.write_text(content, encoding=encoding)
    except OSError as e:
        raise LASWriteError(f"Cannot write to {file_path}: {e}") from e


def _generate_las_content(las_file: LASFile) -> str:
    """Generate LAS file content string with metadata preservation."""
    lines: list[str] = []

    # ~V Version section
    lines.append("~VERSION INFORMATION")
    lines.append(f" VERS.   {las_file.version.vers}  : CWLS LOG ASCII STANDARD")
    wrap_desc = "ONE LINE PER DEPTH STEP" if las_file.version.wrap == "NO" else "MULTIPLE LINES PER DEPTH STEP"
    lines.append(f" WRAP.   {las_file.version.wrap}  : {wrap_desc}")
    lines.append("")

    # ~W Well section
    lines.append("~WELL INFORMATION")
    for key, value in las_file.well.entries.items():
        # Find matching curve for unit info (well entries may have units)
        lines.append(f" {key}.   {value}  :")
    lines.append("")

    # ~C Curve section — preserve units and descriptions
    lines.append("~CURVE INFORMATION")
    for curve in las_file.curves:
        unit = curve.unit if curve.unit else ""
        desc = curve.description if curve.description else ""
        lines.append(f" {curve.mnemonic}.{unit}  : {desc}")
    lines.append("")

    # ~P Parameter section
    if las_file.parameters:
        lines.append("~PARAMETER INFORMATION")
        for param in las_file.parameters:
            unit = param.unit if param.unit else ""
            desc = param.description if param.description else ""
            lines.append(f" {param.mnemonic}.{unit}  {param.value}  : {desc}")
        lines.append("")

    # ~O Other section
    if las_file.other and las_file.other.strip():
        lines.append("~OTHER")
        lines.append(las_file.other.rstrip())
        lines.append("")

    # ~A ASCII data section
    curve_names = las_file.curves_order
    if curve_names and curve_names[0] in las_file.logs:
        num_rows = len(las_file.logs[curve_names[0]])

        # Header line with curve names
        lines.append("~A  " + "  ".join(curve_names))

        null_value = float(las_file.well.get("NULL", "-999.25"))

        for i in range(num_rows):
            row_values: list[str] = []
            for name in curve_names:
                if name in las_file.logs and i < len(las_file.logs[name]):
                    row_values.append(f"{las_file.logs[name][i]:.6f}")
                else:
                    row_values.append(f"{null_value:.6f}")
            lines.append("  ".join(row_values))

    return "\n".join(lines) + "\n"
```

---

### Phase 9: DEV Reader

Create `src/pylasdev/dev_reader.py`.

```python
"""DEV (deviation survey) file reader.

Replaces dev_reader.py with modern Python 3, context managers,
and proper encoding handling.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .encoding import read_with_encoding
from .exceptions import DEVReadError
from .models import DevFile


def read_dev_file(
    file_path: str | Path,
    encoding: str | None = None,
) -> dict[str, Any]:
    """Read a DEV (deviation survey) file and return data dictionary.

    Maintains backward compatibility — returns dict of numpy arrays.

    Args:
        file_path: Path to DEV file.
        encoding: Optional encoding override.

    Returns:
        Dictionary mapping column names to numpy arrays.

    Raises:
        DEVReadError: If file cannot be read or parsed.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise DEVReadError(f"File not found: {file_path}")

    detected_encoding, content = read_with_encoding(file_path, encoding)

    lines = content.splitlines()

    # Pass 1: Count data lines (excluding comments, empty lines, and header)
    data_lines = 0
    header_found = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not header_found:
            header_found = True  # First non-comment line is the header
        else:
            data_lines += 1

    # Pass 2: Parse header and data
    dev_dict: dict[str, Any] = {}
    names: list[str] = []
    header_found = False
    current_line = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        values = re.split(r"[\s\t]+", stripped)

        if not header_found:
            # First non-comment line = column names
            names = values
            for name in names:
                dev_dict[name] = np.zeros(data_lines, dtype=np.float64)
            header_found = True
        else:
            # Data lines
            for k in range(min(len(values), len(names))):
                try:
                    dev_dict[names[k]][current_line] = float(values[k])
                except (ValueError, IndexError):
                    dev_dict[names[k]][current_line] = 0.0
            current_line += 1

    return dev_dict
```

---

### Phase 10: Comparison Utility

Create `src/pylasdev/compare.py`.

```python
"""LAS data comparison utilities.

Replaces las_compare.py with Python 3 syntax and proper logging
instead of print statements.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compare_las_dicts(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> bool:
    """Compare two LAS data dictionaries for equality.

    Args:
        dict1: First LAS data dictionary.
        dict2: Second LAS data dictionary.
        rtol: Relative tolerance for numpy array comparison.
        atol: Absolute tolerance for numpy array comparison.

    Returns:
        True if the dictionaries are equivalent, False otherwise.
    """
    for key in dict2:
        if key not in dict1:
            logger.warning("Key '%s' not found in first dict", key)
            return False

        val1, val2 = dict1[key], dict2[key]

        if isinstance(val2, dict):
            for in_key in val2:
                if in_key not in val1:
                    logger.warning("Key '%s.%s' not found in first dict", key, in_key)
                    return False

                if isinstance(val2[in_key], np.ndarray):
                    if not _compare_arrays(val1[in_key], val2[in_key], key, in_key, rtol, atol):
                        return False
                elif val1[in_key] != val2[in_key]:
                    logger.warning(
                        "Mismatch at '%s.%s': %r vs %r", key, in_key, val1[in_key], val2[in_key]
                    )
                    return False

        elif isinstance(val2, np.ndarray):
            if not _compare_arrays(val1, val2, key, None, rtol, atol):
                return False

        elif isinstance(val2, list):
            if val1 != val2:
                logger.warning("List mismatch at '%s': %r vs %r", key, val1, val2)
                return False
        else:
            if val1 != val2:
                logger.warning("Mismatch at '%s': %r vs %r", key, val1, val2)
                return False

    return True


def _compare_arrays(
    arr1: np.ndarray,
    arr2: np.ndarray,
    key: str,
    in_key: str | None,
    rtol: float,
    atol: float,
) -> bool:
    """Compare two numpy arrays with tolerance."""
    label = f"{key}.{in_key}" if in_key else key

    if arr1.size != arr2.size:
        logger.warning("Array size mismatch at '%s': %d vs %d", label, arr1.size, arr2.size)
        return False

    if not np.allclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True):
        logger.warning("Array values mismatch at '%s'", label)
        return False

    return True
```

---

### Phase 11: Package Init

Create `src/pylasdev/__init__.py`.

```python
"""pylasdev — Python library for LAS (Log ASCII Standard) and DEV well log files.

Public API:
    read_las_file()     — Read LAS file, returns dict (backward compatible)
    write_las_file()    — Write LAS data to file
    read_dev_file()     — Read DEV deviation file, returns dict
    compare_las_dicts() — Compare two LAS data dictionaries
    LASFile             — Dataclass for rich LAS file access
    DevFile             — Dataclass for DEV file access
"""

from .compare import compare_las_dicts
from .dev_reader import read_dev_file
from .models import CurveDefinition, DevFile, LASFile, ParameterEntry, VersionSection, WellSection
from .reader import read_las_file, read_las_file_as_object
from .writer import write_las_file

__all__ = [
    # Core functions (backward compatible)
    "read_las_file",
    "write_las_file",
    "read_dev_file",
    "compare_las_dicts",
    # New object API
    "read_las_file_as_object",
    # Data models
    "LASFile",
    "DevFile",
    "VersionSection",
    "WellSection",
    "CurveDefinition",
    "ParameterEntry",
]
```

---

### Phase 12: Mnemonic Database Cleanup

Move `mnem_base.py` to `src/pylasdev/mnem_base.py`.

**Required cleanup (missing from all original plans):**

The current `mnem_base.py` has **duplicate keys** (e.g., `'AK': 'DT'` appears multiple times). In Python 2, the last value silently wins. In Python 3, same behavior but it's still a latent bug.

```python
# Deduplication script to run once:
# 1. Read mnem_base.py
# 2. Parse the dict literal
# 3. Identify duplicate keys
# 4. Keep last value for each (matches Python 2 behavior)
# 5. Write back as clean dict with no duplicates
# 6. Add type annotation

# The final file should look like:
"""Mnemonic alias database for curve name normalization.

Maps alternate/vendor-specific curve mnemonics to canonical names.
Example: {'AK': 'DT', 'APTS': 'SP', ...}
"""

MNEM_BASE: dict[str, str] = {
    # ... deduplicated entries ...
}
```

No other changes needed to the data itself — just deduplication and adding a type annotation.

---

### Phase 13: Test Suite

#### 13.1 conftest.py

```python
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
```

#### 13.2 test_reader.py

```python
"""Tests for LAS file reader."""

from __future__ import annotations

import numpy as np
import pytest

from pylasdev import read_las_file
from pylasdev.exceptions import LASReadError


class TestReadLASFile:
    """Tests for read_las_file function."""

    def test_read_all_las_files(self, all_las_files: list) -> None:
        """Test reading every LAS file in test_data/."""
        assert len(all_las_files) > 0, "No LAS test files found"

        for las_path in all_las_files:
            data = read_las_file(las_path)

            assert "version" in data
            assert "well" in data
            assert "logs" in data
            assert "curves_order" in data
            assert isinstance(data, dict)

    def test_returns_numpy_arrays(self, all_las_files: list) -> None:
        """Test that log data is returned as numpy arrays."""
        for las_path in all_las_files:
            data = read_las_file(las_path)
            for curve_data in data["logs"].values():
                assert isinstance(curve_data, np.ndarray)

    def test_preserves_curve_order(self, all_las_files: list) -> None:
        """Test that curve order matches log keys."""
        for las_path in all_las_files:
            data = read_las_file(las_path)
            if data["curves_order"]:
                assert list(data["logs"].keys()) == data["curves_order"]

    def test_well_values_are_strings(self, all_las_files: list) -> None:
        """Test that well section values are strings (backward compat)."""
        for las_path in all_las_files:
            data = read_las_file(las_path)
            for value in data["well"].values():
                assert isinstance(value, str)

    def test_file_not_found(self, tmp_path) -> None:
        """Test error handling for missing file."""
        with pytest.raises(LASReadError):
            read_las_file(tmp_path / "nonexistent.las")

    def test_version_is_valid(self, all_las_files: list) -> None:
        """Test version section contains valid version string."""
        for las_path in all_las_files:
            data = read_las_file(las_path)
            assert data["version"]["VERS"] in ["1.2", "2.0"]
```

#### 13.3 test_roundtrip.py

```python
"""Tests for read/write round-trip consistency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pylasdev import compare_las_dicts, read_las_file, write_las_file


class TestRoundTrip:
    """Tests for read-write-read consistency."""

    def test_roundtrip_from_dict(self, sample_las_data: dict, tmp_path: Path) -> None:
        """Test that writing from dict and reading back preserves data."""
        temp_file = tmp_path / "roundtrip.las"
        write_las_file(temp_file, sample_las_data)
        roundtrip = read_las_file(temp_file)

        # Check structure
        assert set(roundtrip["curves_order"]) == set(sample_las_data["curves_order"])

        # Check data values
        for curve in sample_las_data["curves_order"]:
            np.testing.assert_array_almost_equal(
                sample_las_data["logs"][curve],
                roundtrip["logs"][curve],
                decimal=4,
            )

    def test_roundtrip_all_files(self, all_las_files: list, tmp_path: Path) -> None:
        """Test round-trip on all test files."""
        for las_path in all_las_files:
            original = read_las_file(las_path)

            temp_file = tmp_path / las_path.name
            write_las_file(temp_file, original)
            roundtrip = read_las_file(temp_file)

            # Verify curve count preserved
            assert len(roundtrip["curves_order"]) == len(original["curves_order"])

            # Verify data shapes match
            for curve in original["curves_order"]:
                if curve in roundtrip["logs"]:
                    assert original["logs"][curve].shape == roundtrip["logs"][curve].shape
```

---

## 7. File-by-File Migration Map

| Original File | New File | Key Changes |
|---------------|----------|-------------|
| `pylasdev/__init__.py` | `src/pylasdev/__init__.py` | Absolute imports, `__all__`, new exports |
| `pylasdev/las_reader.py` | `src/pylasdev/reader.py` | Context managers, encoding, Path, type hints, no global state |
| `pylasdev/las_lex_pars2.py` | `src/pylasdev/parser.py` | PLY replaced with regex, class-based (no globals), Cyrillic via `\w` |
| `pylasdev/las_lex_pars3.py` | **(Delete)** | Incomplete, non-functional |
| `pylasdev/las_line_reader.py` | `src/pylasdev/data_reader.py` | O(n^2) fix, list accumulation, depth_line flag preserved |
| `pylasdev/las_writer.py` | `src/pylasdev/writer.py` | Preserves units/descriptions, context managers, accepts dict or LASFile |
| `pylasdev/las_compare.py` | `src/pylasdev/compare.py` | `print` to `logging`, tolerance support, Python 3 |
| `pylasdev/dev_reader.py` | `src/pylasdev/dev_reader.py` | Context managers, encoding, Path, type hints |
| `mnem_base.py` | `src/pylasdev/mnem_base.py` | Deduplicated keys, type annotation |
| **(new)** | `src/pylasdev/models.py` | Dataclasses: LASFile, WellSection, CurveDefinition, etc. |
| **(new)** | `src/pylasdev/exceptions.py` | PylasdevError hierarchy (6 exception classes) |
| **(new)** | `src/pylasdev/encoding.py` | chardet + fallback chain |
| `autotest.py` | `tests/test_reader.py` | Converted to pytest |
| `las_test.py` | `tests/test_reader.py` | Merged into reader tests |
| `dev_test.py` | `tests/test_dev_reader.py` | Converted to pytest |
| `big_test.py` | `tests/test_reader.py` (`@pytest.mark.slow`) | Merged with slow marker |

---

## 8. Questions for the Developer

Before implementation, clarify:

1. **LAS 3.0 fate** — Should `las_lex_pars3.py` be deleted entirely, or should a stub/placeholder be kept for future work?
2. **Backward compatibility duration** — How long must the legacy dict API be maintained? Is the new LASFile object API acceptable as the primary API eventually?
3. **mnem_base format** — Should the mnemonic database remain a Python dict literal, or convert to JSON/TOML for easier editing?
4. **Pandas dependency** — Should `to_dataframe()` be a core feature or remain optional (`pylasdev[pandas]`)?
5. **Test data organization** — Should `test_data/` be reorganized into `las/` and `dev/` subdirectories, or kept flat?

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Breaking existing API** | Low | High | `to_dict()` returns identical structure; extensive round-trip tests |
| **Cyrillic regression** | Medium | High | Explicit test with Cyrillic mnemonics; `\w` regex verified |
| **Wrapped mode regression** | Medium | High | depth_line protocol preserved; test with existing `.writed` files |
| **Encoding issues** | Medium | Medium | chardet + 5-encoding fallback chain; test with CP866 files |
| **PLY edge cases missed** | Low | Medium | Test all 27 existing test files against both old and new code |
| **mnem_base duplicate keys** | Low | Low | Dedup script preserves last-wins behavior (matches Python 2) |
| **NumPy compatibility** | Low | Low | `numpy>=1.24` avoids 2.0 breaking changes |

---

## 10. Comparison with lasio

| Feature | lasio | pylasdev (modernized) |
|---------|-------|----------------------|
| Python support | 3.7+ | 3.12+ |
| LAS 1.2/2.0 | Yes | Yes |
| LAS 3.0 | Partial | No (placeholder for future) |
| Mnemonic database | No | **Yes (5,582 entries)** |
| DEV file support | No | **Yes** |
| Pandas integration | Built-in | Optional (`[pandas]`) |
| Backward compat (dict API) | N/A | **Yes** |
| Active maintenance | Yes | Project-specific |

**Recommendation:** Keep pylasdev for its unique value (mnem_base, DEV files). Consider lasio as optional backend for future LAS 3.0 support.

---

## 11. Success Criteria

### Must Have (P0)

- [ ] Python 3.12+ — zero Python 2 syntax
- [ ] All 27 test files parse correctly (same results as original)
- [ ] Round-trip (read/write/read) preserves data
- [ ] Backward compatible API: `read_las_file()` returns dict
- [ ] Well values returned as strings (not floats)
- [ ] Cyrillic mnemonics supported (regex with Unicode)
- [ ] Wrapped mode works correctly (depth_line protocol)
- [ ] pyproject.toml + uv configuration
- [ ] Basic pytest suite passing

### Should Have (P1)

- [ ] Encoding detection (chardet) with fallback chain
- [ ] Type hints on all public APIs
- [ ] ruff lint + format passing
- [ ] mypy type checking passing
- [ ] 80%+ test coverage
- [ ] Exception hierarchy (no bare print for errors)

### Nice to Have (P2)

- [ ] `read_las_file_as_object()` returning LASFile dataclass
- [ ] Pandas DataFrame export (`to_dataframe()`)
- [ ] mnem_base deduplication
- [ ] Sphinx documentation
- [ ] GitHub Actions CI/CD

---

## 12. Implementation Timeline

| Phase | Task | Priority | Dependencies |
|-------|------|----------|-------------|
| 1 | Foundation: pyproject.toml, .gitignore, directory structure | P0 | None |
| 2 | Exceptions module | P0 | Phase 1 |
| 3 | Data models (dataclasses) | P0 | Phase 2 |
| 4 | Regex parser | P0 | Phase 3 |
| 5 | ASCII data reader | P0 | Phase 4 |
| 6 | Encoding module | P1 | Phase 1 |
| 7 | LAS reader (main entry) | P0 | Phases 4, 5, 6 |
| 8 | LAS writer | P0 | Phase 3 |
| 9 | DEV reader | P0 | Phase 6 |
| 10 | Comparison utility | P1 | Phase 3 |
| 11 | Package __init__.py | P0 | Phases 7-10 |
| 12 | mnem_base cleanup | P1 | Phase 1 |
| 13 | Test suite | P0 | Phases 7-11 |
| 14 | Documentation (README) | P1 | Phase 13 |
| 15 | File cleanup (delete old files) | P0 | Phase 13 passing |

---

## 13. Quick Start Commands

```bash
# Initialize project with uv
cd pylasdev
uv sync --extra dev

# Run tests
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=src/pylasdev --cov-report=html

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type checking
uv run mypy src/

# Build package
uv build

# Install locally for testing
uv pip install -e .
```

---

## 14. References

### Official Specifications

- [LAS 2.0 Specification (CWLS)](https://www.cwls.org/wp-content/uploads/2017/02/Las2_Update_Jan2017.pdf)
- [LAS 3.0 File Structure (BC Energy Regulator)](https://www.bc-er.ca/files/operations-documentation/Energy-Resource-Activity-Operations-Manual/Supporting-Documents/las3filestructure.pdf)
- [USGS LAS Format](https://www.usgs.gov/programs/national-geological-and-geophysical-data-preservation-program/las-format)

### Related Libraries

- [lasio on PyPI](https://pypi.org/project/lasio/) — Industry-standard LAS reader/writer
- [lasio Documentation](https://lasio.readthedocs.io/)
- [GitHub: LAS Files Topic](https://github.com/topics/las-files)

### Modernization Reference

- hpgl-reborn project structure and approach
- [uv documentation](https://docs.astral.sh/uv/)
- [ruff documentation](https://docs.astral.sh/ruff/)

---

## Appendix A: Corrections Applied to Source Plans

This plan was assembled from the best elements of 4 specialist plans, with all identified errors corrected:

### Errors Fixed from Plan 4 (primary base)

| Error | Original | Corrected |
|-------|----------|-----------|
| Regex misses Cyrillic | `[A-Za-z0-9_\-]` | `[\w\-]` (Python 3 `\w` = Unicode) |
| Wrapped mode logic | `i % curve_count` | `depth_line` flag protocol (from actual codebase) |
| NumPy version | `numpy>=2.0` | `numpy>=1.24` (avoids 2.0 breaking changes) |
| `uv.lock` in .gitignore | gitignored | **Not gitignored** (committed for reproducible builds) |
| Duplicate LASParseError | Defined in parser.py AND exceptions.py | Only in exceptions.py |
| `from_dict()` stub | Body was `...` | Fully implemented |
| Well values as floats | `strt: float = 0.0` etc. | All values as strings (matching actual behavior) |
| Writer hardcodes `.M` | `STRT.M  {value}:` | Preserves original units from data |

### Cherry-picked from Plan 1

- LAS format specification section (1.2/2.0/3.0 structure, DEV format, wrapped mode detail)
- Risk assessment matrix format (probability x impact x mitigation)
- Reference links (CWLS, LAS 3.0, USGS specifications)
- Author attribution (Artur Muharlyamov)
- Data flow diagram concept
- Cyrillic awareness / Mnemonic format knowledge

### Cherry-picked from Plan 2

- "Questions for Developer" section (LAS 3.0 fate, backward compat, mnem format, pandas)

### Cherry-picked from Plan 3

- "Files to Delete" table (including PLY-generated parser.out/parsetab.py)
- `__all__` export list in `__init__.py`
- ruff `per-file-ignores` config (`__init__.py = ["F401"]`)
- Coverage configuration in pyproject.toml

### Added (missing from ALL plans)

- `mnem_base.py` deduplication (duplicate keys verified in actual file)
- `version['last_caption']` internal state cleanup (now implicit — new dataclass doesn't have it)
- Writer metadata preservation (original units and descriptions, not `.X` / `X` placeholders)
- `WellSection` as dict-like container (matching original string-value behavior)
- `column_order` field on DevFile

---

**End of Comprehensive Implementation Plan**
