# pylasdev Reborn

Python library for reading and writing LAS (Log ASCII Standard) and DEV (deviation) well log files.

It is "Reborn" because it was updated, fixed and refactored to work with modern tech along with fixing many bugs, adding support for LAS 3.0 files and much more (see full list at the end of this file).

## Installation

```bash
pip install git+https://github.com/itohnobue/pylasdev-reborn.git
```

Or with uv:

```bash
uv add git+https://github.com/itohnobue/pylasdev-reborn.git
```

## Usage

```python
from pylasdev import read_las_file, write_las_file, read_dev_file

# Read a LAS file (returns dict for backward compatibility)
data = read_las_file("well_log.las")
print(data["well"]["WELL"])  # Print well name
print(data["logs"]["DEPT"])  # Access depth curve as numpy array

# Write a LAS file
write_las_file("output.las", data)

# Read a DEV file (returns dict of column name → numpy array)
dev_data = read_dev_file("deviation.dev")
print(dev_data["MD"])   # Measured depth array
print(dev_data["TVD"])  # True vertical depth array
```

### Object-oriented API (new)

```python
from pylasdev import read_las_file_as_object, LASFile, read_dev_file_as_object, DevFile

# Read as typed object for richer access
las: LASFile = read_las_file_as_object("well_log.las")
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
dev: DevFile = read_dev_file_as_object("deviation.dev")
print(dev.column_order)     # ['MD', 'TVD', 'X', 'Y']
print(dev.columns["MD"])    # numpy array of measured depth values
```

### Key Methods

```python
# Key LASFile methods:
curve = las.get_curve_by_mnemonic("GR")   # Find curve by mnemonic
arrays = las.get_array_curves("NMR")      # Get array-type curves (LAS 3.0)
d = las.to_dict()                          # Convert to dict format
las2 = LASFile.from_dict(d)               # Create from dict
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

## Requirements

- Python >= 3.12
- NumPy >= 1.24
- chardet >= 5.0 (optional, for encoding detection)

```bash
# Install with encoding support
pip install "git+https://github.com/itohnobue/pylasdev-reborn.git#egg=pylasdev[encoding]"

# Install with all extras (dev tools + encoding)
pip install "git+https://github.com/itohnobue/pylasdev-reborn.git#egg=pylasdev[all]"
```

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/itohnobue/pylasdev-reborn.git
cd pylasdev
uv sync --extra dev

# Run tests
uv run pytest -v

# Run linting and type checking
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

## Changelog

### Version 1.5.0 (2026-06-02)

Bug fixes, production hardening, and test coverage improvements.

### Version 1.0.0 (2026-02-12)

Complete rewrite from Python 2 to Python 3.12+.

#### New Features
- LAS 3.0 support: array notation, format specifiers ({F}, {E}, {S}, {A:x}), multiple data sections, delimiters
- Type hints on all public APIs
- Object-oriented API: `read_las_file_as_object()` returns typed `LASFile`
- Encoding detection with chardet + fallback chain (cp1251, cp1252, cp866, latin-1)
- Custom exception hierarchy: `LASReadError`, `LASWriteError`, `LASParseError`, `LASVersionError`, `LASEncodingError`, `DEVReadError`
- Comprehensive pytest suite (184 tests)

#### Performance Improvements
- Wrapped mode: O(n²) → O(n) (fixed `numpy.append()` bug)
- Regex parser: 450+ → 170 lines (replaces PLY)

#### Bug Fixes
- Writer preserves original units/descriptions (old writer hardcoded `.X`)
- Writer always outputs WRAP=NO (matches actual non-wrapped output format)
- Parser handles spaces between mnemonic and dot (e.g., `DT  .US/M`)
- Parser supports LAS 3.0 array notation in mnemonic names (e.g., `NMR[1].ms`)
- Auto-detection of mislabeled WRAP headers (files claiming WRAP=YES with non-wrapped data)
- Parser only processes ASCII data inline for LAS 3.0 (prevents double-parsing for 1.2/2.0)
- Roundtrip handles LAS 3.0 string curves stored separately from numeric logs
- Thread-safe parser (no global state)
- Format specifier regex handles trailing spaces (e.g., `{A:0 }` in LAS 3.0 spec files)
- Data reader stops at section boundaries (prevents reading garbage after `~A` section)

#### Mnemonic Database Cleanup
- Deduplicated: 5,577 → 2,020 unique entries

## License

BSD-3-Clause
