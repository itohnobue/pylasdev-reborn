# pylasdev Reborn

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](https://github.com/itohnobue/pylasdev-reborn/blob/main/LICENSE)

Python library for reading and writing LAS (Log ASCII Standard) and DEV (deviation) well log files.

It is "Reborn" because it was updated, fixed and refactored to work with modern tech along with fixing many bugs, adding support for LAS 3.0 files and much more.

## Features

- Read and write LAS 1.2, 2.0, and 3.0 files
- LAS 3.0 support: array notation, format specifiers, multiple data sections, string data
- Read and write DEV (deviation survey) files
- Automatic encoding detection with chardet (supports Cyrillic: cp1251, cp866)
- Auto-detection of mislabeled WRAP headers (WRAP=YES with non-wrapped data)
- Type-safe API with full type hints and dataclass models
- Mnemonic database (2,020 entries) for curve name normalization
- Compare LAS files for equality with configurable tolerance
- Wrapped and non-wrapped data mode support

## Installation

**Requirements:** Python >= 3.12, NumPy >= 1.26. See [Requirements](USAGE.md#requirements) for details.

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

For all read/write APIs, the object-oriented interface, common use cases, troubleshooting, requirements, and development setup, see [Usage](USAGE.md).

## License

BSD-3-Clause
