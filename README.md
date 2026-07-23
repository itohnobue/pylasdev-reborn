# pylasdev Reborn

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](https://github.com/itohnobue/pylasdev-reborn/blob/main/LICENSE)

Python library for reading and writing LAS (Log ASCII Standard) and DEV (deviation) well log files.

It is "Reborn" because it was updated, fixed and refactored to work with modern tech along with fixing many bugs, adding support for LAS 3.0 files and much more.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Basic API (dict-based)](#basic-api-dict-based)
  - [Object-oriented API (new)](#object-oriented-api-new)
    - [Key Methods](#key-methods)
- [API Reference](#api-reference)
  - [Read Functions](#read-functions)
  - [Write Functions](#write-functions)
  - [Comparison](#comparison)
  - [Data Models](#data-models)
  - [Mnemonic Database](#mnemonic-database)
  - [Exceptions](#exceptions)
- [Common Use Cases](#common-use-cases)
  - [Error Handling with try/except](#error-handling-with-tryexcept)
  - [Encoding Override](#encoding-override)
  - [Reading LAS 1.2 Files](#reading-las-12-files)
- [Features](#features)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)
- [Development](#development)
- [Changelog](CHANGELOG.md)
- [License](#license)

## Installation

**Requirements:** Python >= 3.12, NumPy >= 1.26. See [Requirements](#requirements) for details.

> **Note:** This package is **not** published on PyPI. `pip install pylasdev` will
> fail with a 404 error. Install from source:

```bash
git clone https://github.com/itohnobue/pylasdev-reborn.git
cd pylasdev-reborn
pip install .
```

> `pip install .` uses the `hatchling` build backend (specified in `pyproject.toml`).
> pip automatically installs build dependencies, but if you run `python -m build`
> directly, install `hatchling>=1.21.0` first.

Or with uv:

```bash
git clone https://github.com/itohnobue/pylasdev-reborn.git
cd pylasdev-reborn
uv sync
```

## Quick Start

```python
from pylasdev import read_las_file

# Read a LAS file — returns a dict
data = read_las_file("test_data/sample.las")
print(data["well"]["WELL"])    # Well name
print(data["logs"]["DEPT"])    # Depth curve as numpy array
```

See [Usage](#usage) below for all read/write APIs, the object-oriented interface, and advanced features.

## Usage

### Basic API (dict-based)

```python
from pylasdev import read_las_file, write_las_file, read_dev_file

# Read a LAS file (returns dict for backward compatibility)
# Use a real test file from the test_data/ directory (18 sample LAS/DEV files)
data = read_las_file("test_data/sample.las")
print(data["well"]["WELL"])  # Print well name
print(data["logs"]["DEPT"])  # Access depth curve as numpy array

# Write a LAS file
write_las_file("output.las", data)

# Read a DEV file (returns dict of column name → numpy array)
dev_data = read_dev_file("test_data/sample.dev")
print(dev_data["MD"])   # Measured depth array
print(dev_data["TVD"])  # True vertical depth array
```

### Object-oriented API (new)

```python
from pylasdev import read_las_file_as_object, LASFile, read_dev_file_as_object, DevFile

# Read as typed object for richer access
las: LASFile = read_las_file_as_object("test_data/sample.las")
print(las.well["WELL"])     # Dict-like access to well info
print(las.version.vers)     # Version string ("1.2", "2.0", "3.0")
print(las.encoding)         # Detected file encoding
for curve in las.curves:
    print(f"{curve.mnemonic}: {curve.unit}")

# LAS 3.0 features
if las.version.is_las30:
    print(las.data_sections)    # Multiple data sections
    print(las.string_data)      # String-format curve data

# DEV file reading (new object API)
dev: DevFile = read_dev_file_as_object("test_data/sample.dev")
print(dev.column_order)     # ['MD', 'TVD', 'X', 'Y']
print(dev.columns["MD"])    # numpy array of measured depth values
```

#### Key Methods

```python
# Key LASFile methods:
curve = las.get_curve_by_mnemonic("GR")   # Find curve by mnemonic
arrays = las.get_array_curves("NMR")      # Get array-type curves (LAS 3.0)
d = las.to_dict()                          # Convert to dict format
las2 = LASFile.from_dict(d)               # Create from dict
```

## API Reference

### Read Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `read_las_file(file_path, mnem_base=None, encoding=None, max_file_size=None)` | `dict` | Read LAS 1.2/2.0/3.0 file, returns legacy dict format |
| `read_las_file_as_object(file_path, mnem_base=None, encoding=None, max_file_size=None)` | `LASFile` | Read LAS file, returns typed `LASFile` dataclass |
| `read_dev_file(file_path, encoding=None, max_file_size=None)` | `dict` | Read DEV deviation file, returns `{column: ndarray}` dict |
| `read_dev_file_as_object(file_path, encoding=None, max_file_size=None)` | `DevFile` | Read DEV file, returns typed `DevFile` dataclass |

### Write Functions

| Function | Description |
|----------|-------------|
| `write_las_file(file_path, las_data, encoding="utf-8", precision=".8g")` | Write LAS data (dict or `LASFile`) to a `.las` file with configurable encoding and numeric precision |

### Comparison

```python
from pylasdev import compare_las_dicts

# Compare two LAS data dictionaries for equality with tolerances.
# rtol: Relative tolerance for numpy array comparison (default 1e-7).
#       Values are considered equal if |a-b| <= atol + rtol*|b|
# atol: Absolute tolerance for numpy array comparison (default 0.0).
#       Allows small absolute differences (e.g. 1e-6).
are_equal = compare_las_dicts(dict1, dict2, rtol=1e-7, atol=0.0)
print(are_equal)  # True if equivalent, False otherwise
```

`compare_las_dicts()` performs deep comparison of LAS data dictionaries,
including numpy arrays (with tolerance), nested dicts, lists, and scalars.

**Return value:** `True` if the dictionaries are structurally and numerically
equivalent within tolerances; `False` otherwise.

**Logging:** Mismatches are logged via Python's `logging` module at `WARNING`
level. To see detailed comparison output (which keys differed, which arrays
mismatched), configure logging before calling `compare_las_dicts()`:

```python
import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
```

### Data Models

All data model types are available as imports from `pylasdev`:

| Dataclass | Purpose |
|-----------|---------|
| `LASFile` | Complete LAS file representation (version, well, curves, parameters, logs, data sections) |
| `DevFile` | DEV deviation survey file representation |
| `VersionSection` | LAS version info (VERS, WRAP, DLM) with `is_las30` and `delimiter_char` properties |
| `WellSection` | Well information with dict-like access (`well["WELL"]`, `well.get("FLD")`) |
| `CurveDefinition` | Single curve definition including mnemonic, unit, API code, LAS 3.0 format specifiers |
| `ParameterEntry` | Parameter from ~P section with optional array index and zone metadata |
| `ParameterZone` | LAS 3.0 zone association for parameters |
| `ArrayElementInfo` | LAS 3.0 array element metadata (base name, index, time offset) |
| `DataSection` | LAS data section (~A in LAS 1.2/2.0, named sections in LAS 3.0) with name, curve order, and numeric data |

#### LASFile Properties

```python
las: LASFile = read_las_file_as_object("well.las")

# Version
las.version         # VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
las.version.vers    # str — "1.2", "2.0", "3.0"
las.version.wrap    # str — "YES" or "NO"
las.version.dlm     # str — "SPACE", "TAB", or "COMMA" (LAS 2.0+)
las.is_las30        # bool — True if version string starts with "3"

# Well information (dict-like)
las.well["WELL"]    # Well name
las.well["FLD"]     # Field name
las.well["COMP"]    # Company name
las.well.get("API", "")  # Safe access with default

# Curves
las.curves           # list[CurveDefinition] — defined curves
las.curves_order     # list[str] — curve names in write order
las.get_curve_by_mnemonic("GR")    # Lookup by name
las.get_array_curves("NMR")        # All array elements (LAS 3.0)

# Curve structure
for c in las.curves:
    c.mnemonic       # str — normalized mnemonic
    c.unit           # str — e.g., "m", "API"
    c.data_format    # str — "F", "E", "S", "A:x" (LAS 3.0)
    c.is_array_element  # bool — part of array group

# Parameters
las.parameters       # list[ParameterEntry] — all parameters

# Data
las.logs             # dict[str, ndarray] — numeric curve data
las.data_sections    # list[DataSection] — LAS 3.0 sections
las.string_data      # dict[str, ndarray] — string-format data (LAS 3.0)

# Metadata
las.source_file      # str — original file path
las.encoding         # str — detected encoding (e.g., "cp1251")

# Conversion
las.to_dict()        # Convert to legacy dict
LASFile.from_dict(d) # Reconstruct from dict
```

#### DevFile Properties

```python
dev: DevFile = read_dev_file_as_object("survey.dev")

dev.columns          # dict[str, ndarray] — column name → data array
dev.column_order     # list[str] — column names in order
dev.source_file      # str — original file path
dev.encoding         # str — detected encoding

dev.to_dict()        # Convert to legacy dict
DevFile.from_dict(d) # Reconstruct from dict
```

### Mnemonic Database

`MNEM_BASE` is a dictionary of 2,020 mnemonic aliases for curve name
normalization. It maps alternative spellings and abbreviations to canonical
names. Import and pass to read functions:

```python
from pylasdev import MNEM_BASE, read_las_file

data = read_las_file("well.las", mnem_base=MNEM_BASE)
# Curve names in data["logs"] will be normalized using MNEM_BASE
```

### Exceptions

All exceptions inherit from `PylasdevError`:

| Exception | Raised When |
|-----------|-------------|
| `PylasdevError` | Base exception for all pylasdev errors |
| `LASReadError` | File not found, permission denied |
| `LASParseError` | LAS file content cannot be parsed |
| `LASVersionError` | Provided for user code to enforce strict version policies. The library itself issues ``warnings.warn()`` for versions > 3.0 and continues processing. |
| `LASEncodingError` | File encoding cannot be determined or decoded |
| `LASWriteError` | LAS file cannot be written |
| `DEVReadError` | DEV file cannot be read or parsed |

## Common Use Cases

### Error Handling with try/except

pylasdev uses a custom exception hierarchy for all error conditions.
All exceptions inherit from `PylasdevError`, making it easy to catch any
library error with a single `except` clause:

```python
from pylasdev import PylasdevError, read_las_file_as_object

try:
    las = read_las_file_as_object("well.las")
except PylasdevError as e:
    # Catches LASReadError, LASParseError, LASEncodingError, etc.
    print(f"pylasdev error: {e}")
```

For finer-grained handling, catch specific exceptions:

```python
from pylasdev import (
    LASReadError, LASParseError, LASEncodingError, PylasdevError,
    read_las_file_as_object, read_las_file,
)

try:
    las = read_las_file_as_object("well.las")
except LASReadError:
    print("Check file path and permissions")
except LASParseError:
    print("File content is not valid LAS format")
except LASEncodingError:
    print("Unknown encoding — try specifying encoding='latin-1'")
except PylasdevError:
    print("Other pylasdev error")
```

Practical examples:

```python
from pylasdev import (
    read_las_file, read_las_file_as_object,
    LASReadError, LASParseError, LASEncodingError,
)

# Example 1: Handle common read errors
try:
    data = read_las_file("well_log.las")
except LASReadError as e:
    print(f"File error: {e}")
except LASEncodingError as e:
    print(f"Encoding error: {e}")
except LASParseError as e:
    print(f"Parse error: {e}")

# Example 2: Using the object API with error handling
try:
    las = read_las_file_as_object("survey.las")
    print(f"Version: {las.version.vers}, Curves: {len(las.curves)}")
except LASReadError:
    print("Could not open file — check that the path exists")
except LASParseError as e:
    print(f"Malformed LAS file: {e}")
```

> **Important:** The `max_file_size` parameter raises `ValueError`, which is **not** a
> subclass of `PylasdevError`. A blanket `except PylasdevError` will silently miss
> file-size-limit violations. Always catch `ValueError` separately when using
> `max_file_size`:
>
> ```python
> from pylasdev import read_las_file, PylasdevError
>
> try:
>     data = read_las_file("large.las", max_file_size=10_000_000)
> except ValueError as e:
>     print(f"File exceeds size limit: {e}")
> except PylasdevError as e:
>     print(f"LAS error: {e}")
> ```

### Encoding Override

```python
from pylasdev import read_las_file, LASEncodingError

# Auto-detection (default)
data = read_las_file("well_log.las")
print(data["encoding"])  # e.g., "cp1251"

# Force specific encoding
data = read_las_file("well_log.las", encoding="utf-8")

# Handling encoding errors gracefully
try:
    data = read_las_file("unknown_encoding.las")
except LASEncodingError:
    # Fall back to latin-1 which never fails on bytes
    data = read_las_file("unknown_encoding.las", encoding="latin-1")
```

### Reading LAS 1.2 Files

```python
from pylasdev import read_las_file_as_object

# LAS 1.2 files work exactly the same as LAS 2.0
las = read_las_file_as_object("legacy_v12.las")
print(las.version.vers)          # "1.2"
print(las.well["DATE"])         # Date field from ~W section
for curve in las.curves:        # All curves from ~C section
    print(f"{curve.mnemonic}: {curve.unit}")
```

## Features

- Read and write LAS 1.2, 2.0, and 3.0 files
- LAS 3.0 support: array notation, format specifiers, multiple data sections, string data
- Read DEV (deviation survey) files
- Automatic encoding detection with chardet (supports Cyrillic: cp1251, cp866)
- Auto-detection of mislabeled WRAP headers (WRAP=YES with non-wrapped data)
- Type-safe API with full type hints and dataclass models
- Mnemonic database (2,020 entries) for curve name normalization
- Compare LAS files for equality with configurable tolerance
- Wrapped and non-wrapped data mode support

## Troubleshooting

### File won't open — encoding issues

**Symptom:** `LASEncodingError` or garbled text when reading a LAS file.

**Cause:** The file uses a non-UTF-8 encoding (common with Cyrillic content).

**Fix:** Specify the encoding explicitly or try latin-1 as fallback:

```bash
python -c "
from pylasdev import read_las_file
data = read_las_file('well.las', encoding='latin-1')
print('Success, encoding:', data['encoding'])
"
```

### WRAP mode confusion

**Symptom:** Data is not being read correctly, or values appear shifted.

**Cause:** Some LAS files claim `WRAP=YES` in the header but actually contain
non-wrapped data. pylasdev automatically detects this and handles it correctly.

**Verify:** Check the detected WRAP mode:

```bash
python -c "
from pylasdev import read_las_file_as_object
las = read_las_file_as_object('mysurvey.las')
print('WRAP in file:', las.version.wrap)
"
```

### -999.25 null value

**Symptom:** Log data contains `-999.25` values that should be treated as missing.

**Cause:** `-999.25` is the standard null value in the LAS format specification
(CWLS convention). pylasdev does not automatically mask or replace null values
— you must handle them in your own code.

**Fix:** Mask null values after reading:

```python
import numpy as np
from pylasdev import read_las_file

data = read_las_file("well.las")
depth = data["logs"]["DEPT"]
gr = data["logs"]["GR"]

# Mask -999.25 null values
valid = gr != -999.25
depth_clean = depth[valid]
gr_clean = gr[valid]
```

### General diagnostics

```bash
# Check Python version and pylasdev import
python -c "import pylasdev; print(pylasdev.__version__)"

# Check file encoding
python -c "
from pylasdev import read_las_file_as_object
las = read_las_file_as_object('well.las')
print('Detected encoding:', las.encoding)
"
```

## Requirements

- Python >= 3.12
- NumPy >= 1.26
- chardet >= 5.0 (optional, for encoding detection)

```bash
# Install with encoding support
pip install ".[encoding]"

# Install with all extras (dev tools + encoding)
pip install ".[all]"
```

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/itohnobue/pylasdev-reborn.git
cd pylasdev-reborn
uv sync --extra dev

# Run tests
uv run pytest -v

# Run linting and type checking
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

## License

BSD-3-Clause
