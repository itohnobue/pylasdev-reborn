"""pylasdev — Python library for LAS (Log ASCII Standard) and DEV well log files.

Public API:
    # Read/write functions
    read_las_file()            — Read LAS file, returns dict (backward compatible)
    read_las_file_as_object()  — Read LAS file, returns typed LASFile (new API)
    write_las_file()           — Write LAS data to file
    read_dev_file()            — Read DEV deviation file, returns dict
    read_dev_file_as_object()  — Read DEV file, returns typed DevFile (new API)
    compare_las_dicts()        — Compare two LAS data dictionaries

    # Data models
    LASFile                    — Dataclass for rich LAS file access
    DevFile                    — Dataclass for DEV file access
    VersionSection             — LAS version info (VERS, WRAP, DLM)
    WellSection                — Well information with dict-like access
    CurveDefinition            — Single curve definition (mnemonic, unit, format)
    ParameterEntry             — Parameter from ~P section with metadata
    ParameterZone              — LAS 3.0 zone association for parameters
    ArrayElementInfo           — LAS 3.0 array element metadata
    DataSection                — LAS 3.0 data section (~A) with curves and data

    # Mnemonic database
    MNEM_BASE                  — Mnemonic alias database for curve name normalization

    # Exceptions
    PylasdevError              — Base exception for all pylasdev errors
    LASReadError               — File not found or permission denied
    LASWriteError              — LAS file cannot be written
    LASParseError              — LAS file content cannot be parsed
    LASVersionError            — Provided for user code to enforce strict version policies
    LASEncodingError           — File encoding cannot be determined or decoded
    LASDataError               — LAS data validation fails in from_dict
    DEVReadError               — DEV file cannot be read or parsed
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pylasdev")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for editable installs during development

from .compare import compare_las_dicts
from .dev_reader import read_dev_file, read_dev_file_as_object
from .exceptions import (
    DEVReadError,
    LASDataError,
    LASEncodingError,
    LASParseError,
    LASReadError,
    LASVersionError,
    LASWriteError,
    PylasdevError,
)
from .mnem_base import MNEM_BASE
from .models import (
    ArrayElementInfo,
    CurveDefinition,
    DataSection,
    DevFile,
    LASFile,
    ParameterEntry,
    ParameterZone,
    VersionSection,
    WellSection,
)
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
    "read_dev_file_as_object",
    # Data models
    "LASFile",
    "DevFile",
    "VersionSection",
    "WellSection",
    "CurveDefinition",
    "ParameterEntry",
    "ParameterZone",
    "ArrayElementInfo",
    "DataSection",
    # Mnemonic base
    "MNEM_BASE",
    # Exceptions
    "PylasdevError",
    "LASReadError",
    "LASWriteError",
    "LASParseError",
    "LASVersionError",
    "LASEncodingError",
    "LASDataError",
    "DEVReadError",
]
