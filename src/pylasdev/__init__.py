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
from .exceptions import (
    DEVReadError,
    LASEncodingError,
    LASParseError,
    LASReadError,
    LASVersionError,
    LASWriteError,
    PylasdevError,
)
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
    # Exceptions
    "PylasdevError",
    "LASReadError",
    "LASWriteError",
    "LASParseError",
    "LASVersionError",
    "LASEncodingError",
    "DEVReadError",
]
