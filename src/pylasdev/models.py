"""Data models for LAS file structures.

Supports LAS 1.2, 2.0, and 3.0 formats.
"""

from __future__ import annotations

import copy
import math
import re
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from ._version_spec import _LASVersionSpec

# I2F-09: Maximum field length for string values — matches parser's limit
# (parser.py:86).  Without this, from_dict paths accept arbitrarily-long
# strings that bypass all item-count and element-count guards.
MAX_FIELD_LENGTH = 100_000


def _safe_str(value: Any, default: str = "", max_length: int | None = MAX_FIELD_LENGTH) -> str:
    """Convert value to str, returning *default* when *value* is None.

    Prevents ``str(None)`` → ``"None"`` in dict roundtrip paths.
    When *max_length* is not ``None``, raises ``ValueError`` if the result
    exceeds the limit (I2F-09: unbounded string lengths).
    """
    if value is None:
        return default
    # F-I2-MD3-03: str(float('nan')) → "nan", str(float('inf')) → "inf".
    # Non-finite float values produce corrupted string representations
    # that propagate silently through from_dict roundtrip paths.
    # F-003: np.float32 (and other numpy floating types) do NOT
    # inherit from Python's builtin ``float``.  ``isinstance(value, float)``
    # misses numpy float types, allowing np.float32(np.nan) and
    # np.float32(np.inf) to bypass the non-finite guard.
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError(f"Cannot safely convert non-finite float {value} to string")
    if isinstance(value, bytes):
        raise TypeError("Decode to str first: value.decode('utf-8')")
    result = str(value)
    # F-22: Strip control characters (except common whitespace: \t, \n, \r).
    # Control chars like \x00, \x1b, \x7f pass through str() unchanged and
    # can corrupt LAS output or downstream consumers reading dataclass fields.
    result = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x85\u2028\u2029]", "", result)
    if max_length is not None and len(result) > max_length:
        raise ValueError(
            f"String value length {len(result)} exceeds maximum allowed ({max_length})"
        )
    return result


# M-03: Mnemonic whitelist mirroring the parser's mnemonic grammar
# (parser.DATA_LINE_PATTERN mnemonic group ``[\w\-]+(?:\[\d+\])?``).
# The previous model-layer checks blacklisted only spaces/tabs/newlines/dots,
# so punctuation (``:``, ``|``, ``#``, ``;``, etc.) passed construction,
# was emitted by the writer, and then silently dropped the whole curve +
# data column on re-read because the parser cannot match such mnemonics.
_MNEMONIC_PATTERN = re.compile(r"^[\w\-]+(\[\d+\])?$")

# M-04: Unit character whitelist.  Must match the parser's WIDENED unit
# class (parser.DATA_LINE_PATTERN unit group ``[\w\-/.%°]*`` — N-I-22
# added ``%``, ``°``, ``.`` so common units like ``PHIT.%``, ``TEMP.°C``,
# ``RT.ohm.m`` survive roundtrip).  Validating with the narrower
# ``[\w\-/]*`` here would regress N-I-22 at the model layer.
_UNIT_PATTERN = re.compile(r"^[\w\-/.%°]*$")

# M-84: Map bare LAS 3.0 section-type keywords (the parser's
# _SECTION_TYPE_MAP bare forms) to their canonical *_DATA form.  The
# parser accepts both "CORE" and "CORE_DATA" (mapping both to
# "CORE_DATA"), so a model constructed with the bare form must be
# normalized or the writer's _section_type_to_prefix cannot emit a
# roundtrippable header (it falls back to "A" and the re-read
# section_type silently becomes LOG_DATA).
_BARE_SECTION_TYPE_TO_DATA: dict[str, str] = {
    "CORE": "CORE_DATA",
    "DRILLING": "DRILLING_DATA",
    "FORMATION": "FORMATION_DATA",
    "INCLINOMETRY": "INCLINOMETRY_DATA",
    "LOG": "LOG_DATA",
    "MUD": "MUD_DATA",
    "PERFORATIONS": "PERFORATIONS_DATA",
    "RISK": "RISK_DATA",
    "STRUCTURE": "STRUCTURE_DATA",
    "TEST": "TEST_DATA",
    "TOPS": "TOPS_DATA",
}

# MOD-11: The ONLY metadata keys DevFile.to_dict can emit under a
# ``_meta_`` prefix (``_metadata_assignments`` at to_dict).  The
# from_dict ``_meta_``-prefix handling is restricted to this closed set —
# any other ``_meta_*`` key is a user column, NOT encoded metadata (the
# previous opaque-prefix design hijacked user columns named ``_meta_*``
# and silently dropped them; 5 prior fix commits patched inside the
# prefix branch without ever questioning the unconditional strip).
_DEV_META_KEYS: frozenset[str] = frozenset({"encoding", "source_file", "column_order"})


def _check_column_array_like(value: Any, context: str) -> None:
    """MOD-17/MOD-23: enforce the 1-D array-like contract for data columns.

    LAS data columns (``logs``, ``string_data``, ``data``, DevFile
    ``columns``) are strictly 1-D sequences.  Rejects:

    - Python ``str``/``bytes`` — ``len()`` exists but a string is a scalar
      in LAS context; a 3-char string passed as a 3-row column corrupts
      the writer (MOD-23).
    - Python scalars (int/float/bool/complex/None) — no ``len()``; the
      old ``_value_len`` treated them as 1-row values, letting ``3.14``
      into a 1-row file where ``validate()`` reported 0 issues and the
      writer crashed with a misleading ``len() of unsized object``
      (MOD-23).
    - ndim >= 2 arrays — ``np.atleast_1d`` preserves 2-D, numeric paths
      crash with misleading LASWriteError and the string_data path
      SILENTLY corrupts data on write→read (MOD-17).

    0-d numpy arrays (``np.array(5.0)``) are ACCEPTED — the M-18
    convention treats them as single-element values (DataSection /
    LASFile length checks special-case ndim==0).
    """
    if isinstance(value, (str, bytes)):
        raise ValueError(
            f"{context}: value must be a 1-D array-like (list, tuple, or "
            f"numpy array), got {type(value).__name__}"
        )
    if isinstance(value, np.ndarray):
        if value.ndim >= 2:
            raise ValueError(
                f"{context}: value must be a 1-D array, got {value.ndim}-D "
                f"array with shape {value.shape}.  LAS data columns are "
                f"single curves; reshape or flatten the data before "
                f"assignment."
            )
        return
    if value is None or isinstance(value, (int, float, complex, bool)):
        raise ValueError(
            f"{context}: value must be a 1-D array-like (list, tuple, or "
            f"numpy array), got scalar {type(value).__name__}"
        )
    # Generic iterable (list, tuple, range, ...): verify it is not a
    # ragged 2-D sequence after conversion.
    try:
        _arr = np.asarray(value)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"{context}: value must be a 1-D array-like, could not "
            f"convert {type(value).__name__} to an array: {e}"
        ) from None
    if _arr.ndim >= 2:
        raise ValueError(
            f"{context}: value must be a 1-D array, got {_arr.ndim}-D "
            f"array with shape {_arr.shape}.  LAS data columns are single "
            f"curves; reshape or flatten the data before assignment."
        )


def _is_dev_metadata_shaped(key: str, value: Any) -> bool:
    """PF-07: True when *value* has the metadata SHAPE DevFile.to_dict emits
    for the metadata key *key*.

    ``to_dict`` stores: ``str`` for source_file and encoding, and
    ``str``/``bytes``/``list[str]`` for column_order; ``None`` is accepted.
    An ARRAY value is never metadata-shaped — in the flat dict format a
    column is an array while metadata is str/list-of-str, so a bare
    metadata key carrying an array is a USER COLUMN (used by from_dict to
    disambiguate, mirroring ``_is_encoded_dev_metadata_key``).
    """
    if value is None:
        return True
    if key == "column_order":
        return isinstance(value, (str, bytes)) or (
            isinstance(value, list) and all(isinstance(e, str) for e in value)
        )
    return isinstance(value, (str, bytes))


def _is_encoded_dev_metadata_key(key: str, value: Any) -> bool:
    """MOD-11: True when *key* is a ``_meta_<known>`` key carrying the
    metadata SHAPE DevFile.to_dict emits for that key.

    ``to_dict`` emits ``_meta_<known>`` only when a column name collides
    with a metadata key (I2F-28), storing: ``str`` for source_file and
    encoding, and ``str``/``bytes``/``list[str]`` for column_order.  A
    ``_meta_<known>`` key whose value is NOT metadata-shaped (e.g. a
    numpy array) is a USER COLUMN literally named ``_meta_<known>`` and
    must be preserved as data — the previous opaque-prefix design
    hijacked it and silently dropped the column (MOD-11, 5 prior fix
    commits).  Unknown ``_meta_*`` keys are never encoded metadata.
    """
    if not key.startswith("_meta_"):
        return False
    real = key[6:]
    if real not in _DEV_META_KEYS:
        return False
    return _is_dev_metadata_shaped(real, value)


def _coerce_numpy_scalar(value: Any) -> Any:
    """Convert numpy scalar types to Python scalar types (M-06).

    ``np.int64`` is not an instance of Python ``int`` and ``np.float32``
    is not an instance of Python ``float``, so model-layer type checks
    (``type(x) is int`` / ``isinstance(x, (int, float))``) reject finite
    numpy scalars that from_dict and direct construction legitimately
    receive.  Convert numpy scalars to their Python equivalents so the
    dataclass field contracts hold; non-numpy values pass through.
    """
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _data_is_integral(arr: Any) -> bool:
    """Return True when every element of *arr* is an integral value (M-14).

    ``np.int64`` coercion silently truncates fractional values (1.5 → 1)
    and converts NaN → 0 (dangerous: NaN is the standard missing-data
    marker).  Before coercing ``{I}``-format data to int64, verify that
    no element would be altered by the cast: integer/uint/bool dtypes
    are always safe; float/complex dtypes must be finite and equal to
    their floor; object dtypes are checked per element (exact Python
    ints are safe, float sentinels like -999.25 and NaN are not).
    Non-numeric dtypes (str, bytes, ...) are treated as non-integral so
    they route to the object-dtype branch rather than crashing.
    """
    try:
        _arr = np.asarray(arr)
    except (ValueError, TypeError):
        return False
    if _arr.dtype.kind in "iub":
        return True
    if _arr.dtype.kind in "fc":
        if not np.all(np.isfinite(_arr)):
            return False
        return bool(np.all(_arr == np.floor(_arr)))
    if _arr.dtype.kind == "O":
        for _v in _arr.flat:
            if isinstance(_v, (int, np.integer, bool)):
                continue
            if isinstance(_v, (float, np.floating)):
                if not math.isfinite(_v) or float(_v) != math.floor(float(_v)):
                    return False
                continue
            # Unrecognised scalar — only integral if float-convertible
            # to an exact integer.
            try:
                _f = float(_v)
            except (ValueError, TypeError):
                return False
            if not math.isfinite(_f) or _f != math.floor(_f):
                return False
        return True
    return False


class _GuardedDict(dict[str, Any]):
    """Dict wrapper that validates key types on mutation.

    Prevents non-string keys from being inserted into data/logs/string_data
    dicts, avoiding downstream crashes on string operations (``key.upper()``,
    ``key.strip()``, etc.).

    M-01: CPython's C-level dict methods (``update``, ``setdefault``,
    ``|=``, and ``__init__``) do NOT route through the Python-level
    ``__setitem__`` override, so the original docstring claim ("all
    delegate to ``__setitem__`` internally") was false — int/float keys
    flowed into logs/string_data and either crashed the writer or were
    silently dropped.  Each of those methods is now overridden to
    validate keys explicitly.

    MOD-14 / MOD-01 / MOD-17 / MOD-23: the value-side guard validates the
    RESULTING state, not the current first value:

    - MOD-14: an EMPTY container previously skipped the length guard
      entirely (``if self:``), so ``update``/``|=`` accepted
      inconsistent-length batches that the writer then null-padded into
      fabricated -999.25 rows.  The batch is now validated on its own
      (all incoming values must share one row count).
    - MOD-01: consistent whole-container growth was falsely rejected
      because each incoming value was compared against the current first
      value.  ``update``/``|=`` now permit growth/shrink when the
      RESULTING state is consistent (every untouched key still matches
      the batch length); ``__setitem__`` permits replacing the ONLY key
      with a different length.
    - MOD-17 / MOD-23: scalars, str/bytes, and ndim>=2 arrays are
      rejected at every entry point via ``_check_column_array_like``.
    """

    __slots__ = ("_container_name",)

    def __init__(self, *args: Any, _container_name: str = "data", **kwargs: Any) -> None:
        self._container_name = _container_name
        # M-01: dict.__init__ bypasses __setitem__.  Validate all keys
        # from positional args and keyword args before building.
        _init_items = dict(*args, **kwargs)
        for _k in _init_items:
            self._validate_key(_k)
        # MOD-17/MOD-23: construction entry point — reject scalars, str,
        # and 2-D arrays before they enter a data container (covers
        # DataSection(data=...), LASFile(logs=...), __post_init__ wraps,
        # and the writer's guard re-installation).
        for _v in _init_items.values():
            _check_column_array_like(_v, self._container_name)
        # MOD-14/MOD-01: enforce the resulting-state equal-length
        # invariant at construction (a whole-dict wrap is the same class
        # of batch as update()).
        if len(_init_items) > 1:
            _lengths = {self._value_len(v) for v in _init_items.values()}
            if len(_lengths) > 1:
                raise ValueError(
                    f"{self._container_name}: inconsistent lengths — "
                    f"values have lengths {sorted(_lengths)}.  All values "
                    f"in a data container must have the same row count."
                )
        super().__init__(_init_items)

    def _validate_key(self, key: Any) -> None:
        if not isinstance(key, str):
            raise TypeError(f"{self._container_name}: keys must be str, got {type(key).__name__}")

    @staticmethod
    def _value_len(value: Any) -> int:
        """Row-count of a stored value.

        0-d numpy arrays and scalars have no ``len()`` — treat them as
        single-element values, matching the construction-time convention in
        DataSection.__post_init__ (M-18 guard).  (Scalars are rejected by
        the array-like contract before they reach this helper.)
        """
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return 1
        try:
            return len(value)
        except TypeError:
            return 1

    def _check_value_length(self, value: Any) -> None:
        """M-43 + MOD-17/MOD-23: enforce equal row counts and the 1-D
        array-like contract on a single inserted value.

        An empty container imposes no length constraint (the first key
        defines the batch length — MOD-14); the array-like contract always
        applies (scalars, str, and ndim>=2 arrays are rejected).
        """
        _check_column_array_like(value, self._container_name)
        if self:
            _existing = self._value_len(next(iter(self.values())))
            _new = self._value_len(value)
            if _new != _existing:
                raise ValueError(
                    f"{self._container_name}: inconsistent lengths — "
                    f"existing values have length {_existing}, new value "
                    f"has length {_new}.  All values in a data container "
                    f"must have the same row count."
                )

    def __setitem__(self, key: Any, value: Any) -> None:
        self._validate_key(key)
        _check_column_array_like(value, self._container_name)
        # MOD-01: validate the RESULTING state.  Replacing the ONLY key is
        # consistent growth (trivially consistent); adding/replacing a key
        # against OTHER keys must preserve the equal-length invariant.
        _new = self._value_len(value)
        _others = [self._value_len(v) for k, v in self.items() if k != key]
        if _others and any(_t != _new for _t in _others):
            raise ValueError(
                f"{self._container_name}: inconsistent lengths — "
                f"other values have length {sorted(set(_others))}, new "
                f"value has length {_new}.  All values in a data container "
                f"must have the same row count."
            )
        super().__setitem__(key, value)

    def _validate_batch(self, items: dict[str, Any]) -> None:
        """MOD-14/MOD-01: validate a whole-container update batch against
        the RESULTING state.

        - every incoming value satisfies the 1-D array-like contract;
        - the incoming batch is internally consistent (one row count);
        - keys NOT in the batch keep their length, so a partial resize
          that would leave the container inconsistent is rejected.
        """
        for _v in items.values():
            _check_column_array_like(_v, self._container_name)
        if not items:
            return
        _new_lengths = {_k: self._value_len(_v) for _k, _v in items.items()}
        if len(set(_new_lengths.values())) > 1:
            raise ValueError(
                f"{self._container_name}: inconsistent lengths in batch — "
                f"{_new_lengths}.  All values in a data container must "
                f"have the same row count."
            )
        _batch_len = next(iter(_new_lengths.values()))
        for _k, _v in self.items():
            if _k not in items and self._value_len(_v) != _batch_len:
                raise ValueError(
                    f"{self._container_name}: inconsistent lengths — "
                    f"existing value '{_k}' has length "
                    f"{self._value_len(_v)}, new values have length "
                    f"{_batch_len}.  All values in a data container must "
                    f"have the same row count."
                )

    def update(self, *args: Any, **kwargs: Any) -> None:
        # M-01: dict.update bypasses __setitem__.
        _update_items = dict(*args, **kwargs)
        for _k in _update_items:
            self._validate_key(_k)
        self._validate_batch(_update_items)
        super().update(_update_items)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        # M-01: dict.setdefault bypasses __setitem__.
        self._validate_key(key)
        # M-43: value-side length guard (only when the key is actually
        # inserted — an existing key returns the stored value untouched).
        if key not in self:
            self._check_value_length(default)
        return super().setdefault(key, default)

    def __ior__(self, other: Any) -> _GuardedDict:  # type: ignore[misc,override]
        # M-01: dict.__ior__ bypasses __setitem__.
        _other = dict(other)
        for _k in _other:
            self._validate_key(_k)
        self._validate_batch(_other)
        return super().__ior__(_other)

    def trim_all(self, length: int) -> None:
        """Trim every stored value to *length* rows (whole-container reconcile).

        The F36 read-path reconciliation trims ALL values to the actual
        data-line count when the pre-scan over-counted (standalone
        mnemonic header rows in ~A, section-detection divergence).
        Per-key ``__setitem__`` reassignment cannot express this: the
        FIRST key would be compared against still-untrimmed siblings and
        rejected by the M-43 length guard even though the final state is
        fully consistent (F-01).  This method performs the whole-container
        trim first, then validates the equal-length invariant ONCE —
        legitimate internal reconciliation succeeds while genuinely
        inconsistent containers still raise.
        """
        if not self:
            return
        _trimmed: dict[str, Any] = {}
        for _key, _value in dict.items(self):
            if self._value_len(_value) > length:
                _trimmed[_key] = _value[:length]
            else:
                _trimmed[_key] = _value
        _lengths = {self._value_len(_v) for _v in _trimmed.values()}
        if len(_lengths) > 1:
            raise ValueError(
                f"{self._container_name}: inconsistent lengths — trimming "
                f"to {length} rows leaves values of differing lengths "
                f"{sorted(_lengths)}.  All values in a data container must "
                f"have the same row count."
            )
        for _key, _value in _trimmed.items():
            dict.__setitem__(self, _key, _value)

    def __reduce__(self) -> Any:
        # PF-09: dict subclass with __slots__ does not unpickle by default —
        # reconstruction restores items through __setitem__ BEFORE the
        # _container_name slot is set, and the unconditional
        # _check_column_array_like(value, self._container_name) in
        # __setitem__ raises AttributeError (MOD-14/17/23 regression vs
        # HEAD, where the slot was only read behind ``if self:``).
        # Reconstruct through __init__ (which re-validates keys/values and
        # sets _container_name) and restore the slot state explicitly via
        # __setstate__.  Exact sibling of the M-16 _GuardedList fix below.
        return (
            self.__class__,
            (dict(self),),
            (self._container_name,),
        )

    def __setstate__(self, state: Any) -> None:
        (self._container_name,) = state


class _GuardedList(list[Any]):
    """List wrapper that validates item types on mutation.

    Prevents non-conforming items from being appended/inserted into curves
    or parameters lists, avoiding downstream crashes on attribute access
    (``curve.mnemonic``, ``param.value``, etc.).

    M-02: ``list.__init__`` populates via C-level code that bypasses the
    Python-level validation overrides, so constructor-provided items were
    never checked.  ``__setitem__`` also rejected legitimate slice
    assignment (``lst[0:2] = [...]`` passes an iterable of items, not a
    single item).  Both are now handled.
    """

    __slots__ = ("_container_name", "_expected_type")

    def __init__(
        self,
        *args: Any,
        _container_name: str = "list",
        _expected_type: type = object,
        **kwargs: Any,
    ) -> None:
        self._container_name = _container_name
        self._expected_type = _expected_type
        # M-02: list.__init__ bypasses __setitem__/append.  Validate all
        # constructor-provided items before building.  Materialize first so
        # one-shot iterables (generators) are not consumed by the
        # validation loop and then re-iterated empty by list.__init__.
        if args:
            _init_items = list(args[0])
            for _item in _init_items:
                self._validate_item(_item)
            super().__init__(_init_items)
        else:
            super().__init__()

    def _validate_item(self, item: Any) -> None:
        if not isinstance(item, self._expected_type):
            raise TypeError(
                f"{self._container_name}: items must be "
                f"{self._expected_type.__name__}, "
                f"got {type(item).__name__}"
            )

    def append(self, item: Any) -> None:
        self._validate_item(item)
        super().append(item)

    def insert(self, index: SupportsIndex, item: Any) -> None:
        self._validate_item(item)
        super().insert(index, item)

    def extend(self, items: Iterable[Any]) -> None:
        # M-10: Materialize the iterable BEFORE validating.  A one-shot
        # iterable (generator) would be consumed by the validation loop
        # and then mutated by list.extend with the exhausted iterator —
        # silent data loss (`las.curves.extend(gen)` → []).
        _items = list(items)
        for item in _items:
            self._validate_item(item)
        super().extend(_items)

    def __setitem__(self, index: Any, item: Any) -> None:
        if isinstance(index, slice):
            # M-02: slice assignment passes an iterable of items, not a
            # single item.  Validate each element of the assigned slice.
            # M-10: materialize first — same one-shot-iterable trap as
            # extend (a generator consumed by the validation loop would
            # empty the slice assignment).
            _items = list(item)
            for _item in _items:
                self._validate_item(_item)
            item = _items
        else:
            self._validate_item(item)
        super().__setitem__(index, item)

    def __iadd__(self, other: Iterable[Any]) -> _GuardedList:  # type: ignore[misc]
        # M-10: Materialize before validation — same one-shot-iterable
        # trap as extend (list.__iadd__ would mutate the exhausted
        # iterator, silently appending nothing).
        _other = list(other)
        for item in _other:
            self._validate_item(item)
        return super().__iadd__(_other)

    def __reduce__(self) -> Any:
        # M-16: list subclasses with __slots__ do not unpickle by
        # default — reconstruction bypasses __init__ so _expected_type
        # is unset when the item-restoration path calls _validate_item,
        # raising AttributeError.  Reconstruct through __init__ (which
        # validates the items) and restore the slot state explicitly via
        # __setstate__.
        return (
            self.__class__,
            (list(self),),
            (self._container_name, self._expected_type),
        )

    def __setstate__(self, state: Any) -> None:
        self._container_name, self._expected_type = state


def _create_parameter_entry(param_dict: dict[str, Any]) -> ParameterEntry:
    """Create a ParameterEntry from a dictionary, handling optional zone info.

    Extracted from LASFile.from_dict to avoid duplicated construction logic
    across the parameter_details and params list branches.
    """
    zone = None
    if "zone" in param_dict and isinstance(param_dict["zone"], dict):
        _zone_index = param_dict["zone"].get("zone_index")
        # M-06: Accept numpy integer scalars (np.int64 is not an instance
        # of Python int) before the F-026 type check.
        if _zone_index is not None:
            _zone_index = _coerce_numpy_scalar(_zone_index)
        # F-026: Type validation for zone_index — raw dict values
        # pass through without type checking, violating the int | None
        # contract on ParameterZone.zone_index.
        if _zone_index is not None and type(_zone_index) is not int:
            raise TypeError(f"zone_index: expected int or None, got {type(_zone_index).__name__}")
        zone = ParameterZone(
            zone_name=_safe_str(param_dict["zone"].get("zone_name")),
            zone_index=_zone_index,
        )
    elif "zone" in param_dict:
        # F-I2-M05: Non-dict "zone" value silently ignored before.
        # Emit a warning so callers know the zone metadata was dropped.
        warnings.warn(
            f"Ignoring non-dict 'zone' value of type "
            f"{type(param_dict['zone']).__name__} "
            f"for parameter '{param_dict.get('mnemonic', '?')}'",
            stacklevel=2,
        )
    _array_index = param_dict.get("array_index")
    # M-06: Accept numpy integer scalars (np.int64 is not an instance
    # of Python int) before the F-025 type check.
    if _array_index is not None:
        _array_index = _coerce_numpy_scalar(_array_index)
    # F-025: Type validation for array_index — raw dict values
    # pass through without type checking, violating the int | None
    # contract on ParameterEntry.array_index.
    if _array_index is not None and type(_array_index) is not int:
        raise TypeError(f"array_index: expected int or None, got {type(_array_index).__name__}")
    # F-053: Restore per-section parameter grouping on roundtrip.
    # section_type is optional; None means the standard ~P section.
    _section_type = param_dict.get("section_type")
    if _section_type is not None and not isinstance(_section_type, str):
        _section_type = _safe_str(_section_type)
    # I2F-08 / F-M09: Empty-string and whitespace-only section_type is
    # not normalized to None.  ``""`` passes both guards (is not None,
    # isinstance str), and the writer silently drops parameters with
    # falsy section_type (line 532: ``if not section_type: continue``).
    # ``"  "`` (whitespace-only) passes all guards and survives
    # roundtrip, producing unparseable ``~  _Parameter`` headers.
    # Normalize both to None so that empty-string, whitespace-only,
    # and absent are treated identically.
    if _section_type is not None and _section_type.strip() == "":
        _section_type = None
    # F-118: Reject section_type values that could inject headers when
    # written.  The writer applies ``_sanitize_las_value`` which converts
    # ``\n`` → space, then ``_LEADING_SECTION_RE`` only strips tilde at
    # the STRING START — embedded ``~VERSION`` survives.  The emitted
    # header ``~CORE ~VERSION_Parameter`` is misrouted as a DATA section
    # by the parser.  Reject these values at construction time.
    if _section_type is not None:
        _sec_str = str(_section_type)
        # M-27: Reject whitespace and pipe in addition to newline/tilde.
        # A section_type containing spaces produces a broken header like
        # ``~MY CORE_Parameter`` that the parser's SECTION_PATTERN stops at,
        # misrouting the section to ~O and silently dropping parameters.
        # The pipe is a structural delimiter in LAS 3.0 headers.
        if (
            "\n" in _sec_str
            or "\r" in _sec_str
            or "~" in _sec_str
            or " " in _sec_str
            or "\t" in _sec_str
            or "|" in _sec_str
        ):
            raise ValueError(
                f"section_type contains invalid characters "
                f"(whitespace, newline, tilde, or pipe): {_sec_str!r}.  "
                f"section_type must be a LAS identifier "
                f"(alphanumeric + underscore)."
            )
    # I2F-12: Validate parameter data_format against the same valid set
    # used for curve data_format.  Curve validation runs in two places
    # (_validate_from_dict_input lines 227-237 and 241-257).  Parameter
    # data_format had zero validation — any string was silently accepted
    # and propagated through writer as a {XYZ} format specifier.
    #
    # Only validate single-character values — multi-character strings
    # like "DD/MM/YYYY" are metadata (date format descriptors, etc.)
    # stored under the data_format key by real-world LAS files, not
    # LAS format specifiers.  Single-character non-format values like
    # "X" or "G" would be propagated as {X} / {G} format specifiers
    # and produce corrupted LAS output.
    # MOD-02: Uppercase the raw value BEFORE validation — mirroring the
    # curve path (_validate_from_dict_input lines 928/978 ``.upper()``) and
    # the parser (parser.py:2840 ``.upper()``).  The previous code stored
    # the raw case, so from_dict RAISED on lowercase 'f' while the curve
    # path accepted/normalized it — three construction paths, three
    # outcomes on identical input.
    _data_format = _safe_str(param_dict.get("data_format"), "").upper()
    # N-I-21: Align the from_dict path with the parser (parser.py ~2402)
    # for multi-character parameter data_format.  The parser CLEARS
    # multi-char metadata templates ({DD/MM/YYYY}, {DEG}) with a warning
    # and truncates extended Fortran-style codes (F8.3/E10.2) to their
    # single-letter base; from_dict previously PRESERVED them and the
    # writer emitted them UNBRACED — the two construction paths wrote
    # different files and re-read lost the data_format.  Mirror the
    # parser's behavior (twin of M-23, which aligned the curve path).
    if _data_format and len(_data_format) > 1:
        if _EXTENDED_FORMAT_SPEC_RE.match(_data_format):
            _data_format = _data_format[0]
        else:
            warnings.warn(
                f"Ignoring multi-character data_format "
                f"'{_data_format}' for parameter "
                f"'{param_dict.get('mnemonic', '?')}'.  "
                f"Only single-letter LAS format codes or "
                f"Fortran-style extended codes are valid; "
                f"clearing to empty string.",
                UserWarning,
                stacklevel=2,
            )
            _data_format = ""
    if _data_format and len(_data_format) == 1 and _data_format not in _VALID_DATA_FORMATS:
        # MOD-02: warn-and-clear instead of raising — matching the parser
        # (parser.py:2886-2896) and the curve path (N-I-07,
        # _validate_from_dict_input lines 954-962).  Single-char non-format
        # codes like {X}/{G} occur in real-world files; the previous raise
        # made the same input fail on from_dict but warn+clear on the
        # parser path, so a parse→to_dict→from_dict roundtrip crashed.
        warnings.warn(
            f"Invalid data_format '{_data_format}' for parameter "
            f"'{param_dict.get('mnemonic', '?')}'. "
            f"Valid values: {', '.join(sorted(_VALID_DATA_FORMATS))}. "
            f"Clearing to empty string.",
            UserWarning,
            stacklevel=2,
        )
        _data_format = ""
    return ParameterEntry(
        mnemonic=_safe_str(param_dict.get("mnemonic", "")),
        unit=_safe_str(param_dict.get("unit", "")),
        value=_safe_str(param_dict.get("value", "")),
        description=_safe_str(param_dict.get("description", "")),
        data_format=_data_format,
        array_index=_array_index,
        zone=zone,
        section_type=_section_type,
    )


def _validate_iterable_of_dicts(
    items: Any,
    context_name: str,
) -> list[dict[str, Any]]:
    """Validate that *items* is a list and every element is a dict.

    Used by ``LASFile.from_dict`` to consolidate list-of-dicts validation
    across multiple processing paths that previously used inconsistent
    isinstance/error patterns (F-012, F-013, F-058).
    """
    if not isinstance(items, list):
        raise TypeError(f"{context_name} must be a list, got {type(items).__name__}")
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"{context_name}[{i}] must be a dict, got {type(item).__name__}")
    return items


def _resolve_dict_entry(
    data: dict[str, Any],
    key: str,
    expected_type: type[Any] | tuple[type[Any], ...],
    default_factory: Callable[[], Any],
) -> Any:
    """Extract *key* from *data* with type validation.

    Returns *default_factory()* when *key* is missing or its value is ``None``.
    Raises ``TypeError`` when the value is present but not an instance of
    *expected_type*.  Replaces the ``data.get(key) or default`` pattern used
    previously, which confused falsy values with missing keys.

    This helper eliminates the truthiness-based dispatch class of bugs that
    prior fix rounds repeatedly attempted (and failed) to eliminate.
    """
    value = data.get(key)
    if value is None:
        return default_factory()
    # M-06: Accept numpy scalar types (np.int64 is not an instance of
    # Python int, np.float32 is not an instance of float) — from_dict
    # legitimately receives numpy scalars for int/float fields such as
    # ArrayElementInfo.index / time_offset.  Convert to Python scalars
    # before the isinstance gate.  Values of other types (dict, list,
    # str) pass through unchanged.
    value = _coerce_numpy_scalar(value)
    if not isinstance(value, expected_type):
        raise TypeError(f"{key}: expected {expected_type}, got {type(value).__name__}")
    # F-9-002: Reject bool when int is expected — bool subclasses int so
    # isinstance(True, int) is True, but bool is semantically not an int.
    # Consistent with type() is not int guards at lines 62, 84, 717.
    if (
        value is not None
        and isinstance(value, bool)
        and (expected_type is int or (isinstance(expected_type, tuple) and int in expected_type))
    ):
        raise TypeError(f"{key}: expected {expected_type}, got bool")
    # I2F-13: Non-finite floats (inf, nan) slip past the isinstance
    # gate — isinstance(float('inf'), (int, float)) is True, but the
    # writer crashes at int(float('inf')) → OverflowError.  The parser
    # has np.isfinite protection; from_dict lacked it.
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        raise ValueError(f"{key}: non-finite float values (inf, -inf, nan) are not allowed")
    return value


# Valid data_format values for LAS curves (LAS 3.0 spec).
_VALID_DATA_FORMATS = frozenset({"F", "E", "D", "A", "S", "I"})

# M-23: Extended Fortran-style format specifiers accepted by the parser
# (parser._FORMAT_SPEC_RE), e.g. "F8.3", "E10.2", "D0.00E+00".  Multi-char
# data_format values matching this pattern are normalized to their
# single-letter code; all other multi-char values (e.g. "DD/MM/YYYY",
# "DENSITY", "DEG") are metadata templates and must be CLEARED rather than
# blindly truncated to df[0] (which fabricates a valid-looking format).
_EXTENDED_FORMAT_SPEC_RE = re.compile(
    r"^(?:[FEDI](?:\d+(?:\.\d+)?(?:[ED][+-]?\d+)?)?|[SA]\w*(?:;[\w.]+)*)$"
)


def _validate_from_dict_input(data: dict[str, Any]) -> None:
    """Validate from_dict input before construction (F-017, F-018, IF-015,
    IF-026, F-019).

    Raises ``ValueError`` or ``TypeError`` for invalid input.
    Called at the start of ``LASFile.from_dict`` before any construction,
    closing Pattern #7 (from_dict validation gaps) structurally.
    """
    # --- F-017: Mandatory well fields (STRT, STOP, STEP, NULL) ---
    # F-028: Truthiness-based asymmetry — ``and well`` skipped the check
    # for ``well={}`` (empty dict is falsy) while ``well={"STRT": "0"}``
    # (populated but incomplete) triggered it.  Both are equally invalid.
    well = data.get("well")
    if isinstance(well, dict):
        # F-033: Use version-aware mandatory well fields instead of
        # hardcoding the 8-field LAS 1.2 set for all versions.
        # _LASVersionSpec.mandatory_well_fields returns 8 fields for
        # LAS 1.2 (STRT, STOP, STEP, NULL, WELL, LOC, SRVC, UWI)
        # and 4 fields (STRT, STOP, STEP, NULL) for LAS 2.0/3.0.
        _version = data.get("version")
        _vers_str = ""
        if isinstance(_version, dict):
            _vers_str = str(_version.get("VERS", ""))
        _spec = _LASVersionSpec(_vers_str)
        _mandatory = set(_spec.mandatory_well_fields)
        _well_keys = {k.upper() for k in well if isinstance(k, str)}
        _missing = _mandatory - _well_keys
        if _missing:
            # F-017: Warn about missing mandatory well fields at construction
            # time.  Consistent with parser.py:436-442 and writer.py:333-343
            # which also warn rather than error.  The writer will produce valid
            # LAS output with defaults for missing fields.
            warnings.warn(
                f"Mandatory well field(s) missing: {', '.join(sorted(_missing))}",
                stacklevel=3,
            )

    # --- I2F-11: VERS presence check ---
    # The parser raises LASParseError for missing VERS (parser.py:406-409).
    # from_dict previously silently defaulted to "2.0" via _safe_str(),
    # manufacturing version metadata on roundtrip gaps.
    # M-73: The check fired only when the "version" key was PRESENT but VERS
    # was missing inside it.  When the entire "version" key was absent,
    # data.get("version") returned None, isinstance(None, dict) was False,
    # the check was skipped, and from_dict silently defaulted to LAS 2.0 —
    # while the parser raises LASParseError for equivalent input (content
    # without a ~V section, parser.py:697-701).  Fire the presence check for
    # BOTH states so the two APIs stay consistent.
    version = data.get("version")
    if version is None:
        raise ValueError(
            "Missing required 'version' section.  "
            "'version' must be present with a 'VERS' key "
            "(e.g. '1.2', '2.0', '3.0')."
        )
    if not isinstance(version, dict):
        raise TypeError(f"version: expected dict, got {type(version).__name__}")
    if "VERS" not in version:
        raise ValueError(
            "Missing required VERS field in version section. "
            "VERS must be present (e.g. '1.2', '2.0', '3.0')."
        )
    # H-04: Validate VERS value against known LAS versions.
    # M-04: VERS warning removed from here — duplicate of
    # VersionSection.__post_init__ (line 577).  The post_init
    # check covers all construction paths (direct, parser,
    # from_dict) as the single authoritative source.

    # --- F-018: DLM validation ---
    if isinstance(version, dict):
        dlm_raw = version.get("DLM")
        if dlm_raw is not None and dlm_raw != "":
            dlm_upper = str(dlm_raw).upper()
            # F-I2-MD3-01: Validate DLM value first, then check LAS
            # version compatibility.  The previous if/elif structure
            # warned for LAS 1.2 with non-SPACE DLM but skipped the
            # validity rejection when the DLM was completely invalid
            # (e.g. "FOO").  Flipping the order guarantees that
            # invalid DLMs are always rejected regardless of version.
            if dlm_upper not in {"SPACE", "TAB", "COMMA"}:
                raise ValueError(f"Invalid DLM value '{dlm_raw}'. Expected SPACE, TAB, or COMMA.")
            # M-04: DLM LAS 1.2 warning removed from here — duplicate
            # of VersionSection.__post_init__ (line 617).  The post_init
            # check covers all construction paths (direct, parser,
            # from_dict) as the single authoritative source.

        # F-006: WRAP not validated against {YES, NO} in from_dict.
        # The parser enforces WRAP ∈ {YES, NO} (parser.py:1097-1108).
        # from_dict previously accepted any string, matching the
        # parser's strictness to prevent silently-corrupted WRAP
        # values propagating through roundtrip.
        wrap_raw = version.get("WRAP")
        if wrap_raw is not None:
            wrap_str = str(wrap_raw).upper()
            if wrap_str not in {"YES", "NO"}:
                raise ValueError(f"Invalid WRAP value '{wrap_raw}'. Expected YES or NO.")

    # --- IF-015: data_format validation ---
    curves = data.get("curves")
    if isinstance(curves, list):
        for i, cd in enumerate(curves):
            if isinstance(cd, dict) and "data_format" in cd:
                _raw = cd.get("data_format")
                if _raw is None or isinstance(_raw, bool):
                    continue  # None means "not set"; bool is not a data format
                # F-17: Use _safe_str() instead of str() to reject
                # non-finite floats (inf, nan) — str(float('inf')) →
                # "INF" → "I" ∈ _VALID_DATA_FORMATS, bypassing validation.
                df = _safe_str(_raw).upper()
                if df:
                    if len(df) == 1:
                        # Single-letter format code — pass through.
                        cd["data_format"] = df
                    elif _EXTENDED_FORMAT_SPEC_RE.match(df):
                        # F-M-012: truncate extended format codes
                        # (parser accepts F8.3, E10.2, etc.)
                        df = df[0]
                        cd["data_format"] = (
                            df  # F-017: truncate in-place so from_dict passes single-char to constructor
                        )
                    else:
                        # M-23: Multi-char metadata templates (DD/MM/YYYY,
                        # DENSITY, DEG) are NOT valid LAS format specifiers.
                        # Clear instead of blind df[0] truncation, which
                        # fabricates a valid-looking single-letter code.
                        df = ""
                        cd["data_format"] = ""
                if df and df not in _VALID_DATA_FORMATS:
                    # N-I-07: Unknown SINGLE-CHAR format codes ({X}, {G})
                    # are cleared with a warning, matching the parser's
                    # behavior (parser.py _parse_curve warns-and-clears
                    # deliberately — real-world files contain such codes).
                    # The previous code raised LASDataError here, so the
                    # same input behaved differently on the two
                    # construction paths and a parse→to_dict→from_dict
                    # roundtrip crashed.  Align to the parser's tolerance.
                    warnings.warn(
                        f"curves[{i}]: invalid data_format "
                        f"'{cd['data_format']}'. "
                        f"Valid values: {', '.join(sorted(_VALID_DATA_FORMATS))}. "
                        f"Clearing to empty string.",
                        UserWarning,
                        stacklevel=2,
                    )
                    cd["data_format"] = ""
    # Per-section curves
    data_sections = data.get("data_sections")
    if isinstance(data_sections, list):
        for si, ds in enumerate(data_sections):
            if not isinstance(ds, dict):
                continue
            scs = ds.get("section_curves")
            if isinstance(scs, list):
                for ci, sc in enumerate(scs):
                    if isinstance(sc, dict) and "data_format" in sc:
                        _raw = sc.get("data_format")
                        if _raw is None or isinstance(_raw, bool):
                            continue
                        # F-17: Use _safe_str() to reject non-finite floats
                        # (same fix as top-level curves path above).
                        df = _safe_str(_raw).upper()
                        if df:
                            if len(df) == 1:
                                # Single-letter format code — pass through.
                                sc["data_format"] = df
                            elif _EXTENDED_FORMAT_SPEC_RE.match(df):
                                # F-M-012: truncate extended format codes
                                # (parser accepts F8.3, E10.2, etc.)
                                df = df[0]
                                sc["data_format"] = (
                                    df  # F-017: truncate in-place for from_dict construction
                                )
                            else:
                                # M-23: Multi-char metadata templates are
                                # cleared, not truncated (same as the
                                # top-level curves path above).
                                df = ""
                                sc["data_format"] = ""
                        if df and df not in _VALID_DATA_FORMATS:
                            # N-I-07: Unknown single-char format codes are
                            # cleared with a warning (parser-aligned), not
                            # raised — same reasoning as the top-level
                            # curves path above.
                            warnings.warn(
                                f"data_sections[{si}].section_curves[{ci}]: "
                                f"invalid data_format '{sc['data_format']}'. "
                                f"Valid values: {', '.join(sorted(_VALID_DATA_FORMATS))}. "
                                f"Clearing to empty string.",
                                UserWarning,
                                stacklevel=2,
                            )
                            sc["data_format"] = ""

    # --- IF-026: Cross-validate data_format vs data placement ---
    _check_df_vs_placement(curves, data, "top-level")
    # Also validate per-section curves against their section-level data/string_data
    if isinstance(data_sections, list):
        for si, ds in enumerate(data_sections):
            if isinstance(ds, dict):
                scs = ds.get("section_curves")
                if isinstance(scs, list):
                    _check_df_vs_placement(scs, ds, f"data_sections[{si}]")

    # --- F-019: Non-3.0 multi-section data_sections ---
    vers_raw = version.get("VERS") if isinstance(version, dict) else None
    vers = str(vers_raw).strip() if vers_raw else ""
    if isinstance(data_sections, list) and len(data_sections) > 1:
        if not vers.startswith("3"):
            raise ValueError(
                f"Multiple data_sections ({len(data_sections)}) are only "
                f"supported for LAS 3.0 files, but version is {vers!r}. "
                f"Use a single section for LAS 1.2/2.0."
            )

    # F-115: Detect ambiguous curve placement — when the same curve name
    # appears in both ``logs`` and ``data_sections[*]``, the writer
    # checks data_sections first (writer.py:743) and uses the LAS 3.0
    # path, silently ignoring the logs value.  Reject the ambiguous
    # input so callers get a clear error instead of silent data discard.
    # LAS 3.0 roundtrip legitimately produces both keys when logs
    # (top-level curves) and data_sections (section-level curves)
    # contain different curve names — that is not ambiguous.
    if isinstance(data_sections, list) and data_sections:
        # Collect all curve names referenced in data_sections.
        # F-M-015: Apply case normalization (upper()) before overlap
        # check.  mnem_base maps semantically-equal keys with different
        # cases to the same canonical name — raw-key intersection
        # produces false negatives when mnem_base normalization is active.
        # Collected once at the top so both F-115 (logs overlap) and
        # E-F-022 (string_data overlap) can use the same set.
        _ds_curve_names: set[str] = set()
        for _ds in data_sections:
            if isinstance(_ds, dict):
                _ds_co = _ds.get("curves_order")
                if isinstance(_ds_co, list):
                    _ds_curve_names.update(str(s).upper() for s in _ds_co)
                _ds_data = _ds.get("data")
                if isinstance(_ds_data, dict):
                    _ds_curve_names.update(str(k).upper() for k in _ds_data.keys())
                _ds_str = _ds.get("string_data")
                if isinstance(_ds_str, dict):
                    _ds_curve_names.update(str(k).upper() for k in _ds_str.keys())

        # F-115: Detect ambiguous curve placement — logs vs
        # data_sections.
        _logs_raw = data.get("logs")
        if isinstance(_logs_raw, dict) and _logs_raw:
            _log_curve_names = {str(k).upper() for k in _logs_raw.keys()}
            _overlap = _log_curve_names & _ds_curve_names
            if _overlap:
                # F-115: Warn rather than reject — the parser roundtrip
                # legitimately produces both logs and data_sections with
                # overlapping curve names (same data, different storage).
                # Rejection would break LAS 3.0 roundtrips.  The warning
                # alerts users to potential silent data discard when
                # manually constructed dicts conflict.
                warnings.warn(
                    f"Curve(s) {sorted(_overlap)} appear in both "
                    f"'logs' and 'data_sections'.  The writer uses "
                    f"data_sections when both are present, ignoring "
                    f"the logs values for these curves.  Place each "
                    f"curve in exactly one location to avoid data "
                    f"loss.",
                    stacklevel=2,
                )

        # E-F-022: Detect overlap between top-level string_data and
        # data_sections[*].string_data.  The writer checks data_sections
        # first (writer.py:743) — when the same curve name appears in
        # both top-level string_data and data_sections, the top-level
        # value is silently ignored.  Warn so callers know about the
        # data discard.
        _top_str = data.get("string_data")
        if isinstance(_top_str, dict) and _top_str:
            _top_str_names = {str(k).upper() for k in _top_str.keys()}
            _ds_str_overlap = _top_str_names & _ds_curve_names
            if _ds_str_overlap:
                warnings.warn(
                    f"Curve(s) {sorted(_ds_str_overlap)} appear in "
                    f"both top-level 'string_data' and "
                    f"'data_sections'.  The writer uses "
                    f"data_sections when both are present, ignoring "
                    f"the top-level string_data values for these "
                    f"curves.  Place each curve in exactly one "
                    f"location to avoid data loss.",
                    stacklevel=2,
                )


def _check_df_vs_placement(
    curves: list[dict[str, Any]] | None,
    data: dict[str, Any],
    context: str,
) -> None:
    """Cross-validate curve data_format against data placement (IF-026)."""
    if not curves:
        return
    # Build sets of curve names in numeric data vs string data.
    # "logs" key for top-level LASFile dicts, "data" key for per-section DataSection dicts.
    logs = data.get("logs") or data.get("data") or {}
    string_data = data.get("string_data", {})
    logs_keys: set[str] = set()
    string_data_keys: set[str] = set()
    if isinstance(logs, dict):
        logs_keys = {str(k) for k in logs}
    if isinstance(string_data, dict):
        string_data_keys = {str(k) for k in string_data}

    for i, cd in enumerate(curves):
        if not isinstance(cd, dict):
            continue
        _raw_df = cd.get("data_format")
        if _raw_df is None or isinstance(_raw_df, bool):
            continue  # None means "not set"; bool is not a data format
        df = str(_raw_df).upper()
        mnemonic = _safe_str(cd.get("mnemonic"))
        if not df or not mnemonic:
            continue
        # I2F-37 broadened string-format exemption to {A} in the
        # string_data direction (line 426) but missed the logs direction.
        # {A}-format curves WITHOUT array_info are string-format and
        # must be rejected from numeric logs.  {A}-format curves WITH
        # array_info (e.g. NMR[1]) are genuinely numeric (float64) and
        # belong in logs/data.  (F-I2-MD4-02)
        if mnemonic in logs_keys:
            if df == "S" or (df == "A" and not cd.get("array_info")):
                raise ValueError(
                    f"{context} curve '{mnemonic}' (index {i}) has "
                    f"data_format='{df}' but is in logs (numeric data). "
                    f"String-format curves must be in string_data."
                )
        # I2F-37: Broaden the string-format exemption from "S" only to
        # "S" and "A".  The parser (parser.py:1905-1909) routes both
        # {S} and {A} (without array_info) format curves to string_data.
        # The previous guard rejected {A}-format curves in string_data
        # because "A" != "S", breaking the roundtrip path.  Numeric
        # formats (F, E, D) remain correctly rejected from string_data.
        if df not in ("S", "A") and mnemonic in string_data_keys:
            raise ValueError(
                f"{context} curve '{mnemonic}' (index {i}) has "
                f"data_format='{df}' but is in string_data. "
                f"Numeric-format curves must be in logs."
            )


@dataclass
class VersionSection:
    """LAS Version Information section (~V).

    Supports LAS 1.2, 2.0, and 3.0 formats.
    """

    vers: str = "2.0"
    wrap: str = "NO"
    dlm: str = "SPACE"  # LAS 2.0+: SPACE, TAB, or COMMA

    def validate(self, complete: bool = False) -> list[str]:
        """Validate version section fields.

        Args:
            complete: If True, also run deferred/semantic checks
                suitable for pre-write or post-construction validation.

        Returns:
            List of issue strings (empty = no issues found).
        """
        issues: list[str] = []
        # VERS: warn about unrecognized version values (H-02).
        # I2F-09: Prevent double VERS unrecognized warning.
        # __post_init__ calls validate(complete=False) (1st emission),
        # LASFile.validate(complete=True) calls version.validate(complete=True)
        # which would emit the same warning again.  Use a setattr-based
        # flag to suppress the second emission.
        if self.vers and self.vers not in {"1.2", "2.0", "3.0"}:
            if not getattr(self, "_vers_warned", False):
                object.__setattr__(self, "_vers_warned", True)
                issues.append(
                    f"VersionSection: Unrecognized VERS value "
                    f"{self.vers!r}. Expected 1.2, 2.0, or 3.0."
                )
        # DLM: warn about non-SPACE DLM on LAS 1.2.
        if complete and self.dlm:
            _dlm = self.dlm.upper()
            _spec = _LASVersionSpec(self.vers)
            if _spec.is_las12 and _dlm != "SPACE":
                issues.append(
                    f"DLM '{self.dlm}' is not valid for LAS 1.2 "
                    f"(spec requires SPACE).  The file will use "
                    f"SPACE delimiter on write."
                )
        return issues

    def __post_init__(self) -> None:
        """Validate VERS, WRAP, and DLM after construction.

        from_dict and the parser apply these checks before construction;
        direct ``VersionSection()`` construction previously bypassed them.
        Empty string values are tolerated (treated as "not specified"
        — the dataclass defaults apply).
        """
        # VERS: reject None before any string operations.
        if self.vers is None:
            raise ValueError("VersionSection: VERS cannot be None")
        # F-18: Strip whitespace from version string before any version checks.
        # "  3.0" causes version misidentification — is_las30 uses
        # str.startswith("3") which fails on leading whitespace.
        if self.vers:
            self.vers = self.vers.strip()
        # F-005: Empty VERS (after stripping) silently bypassed all
        # version recognition.  The writer falls back to "2.0" for
        # unknown versions; normalise here so downstream code sees a
        # known version instead of a falsy string.
        if not self.vers:
            warnings.warn(
                "VersionSection: VERS is empty, defaulting to '2.0'.",
                stacklevel=2,
            )
            self.vers = "2.0"
        # WRAP: reject None before any string operations.
        if self.wrap is None:
            raise ValueError("VersionSection: WRAP cannot be None")
        # WRAP: validate non-empty values against YES/NO
        # (matching parser.py:1097-1108).  Empty string = not specified.
        if self.wrap:
            _wrap = self.wrap.upper()
            self.wrap = _wrap
            if _wrap not in {"YES", "NO"}:
                raise ValueError(
                    f"VersionSection: invalid WRAP value {self.wrap!r}.  Expected YES or NO."
                )
        # DLM: validate non-empty values against SPACE/TAB/COMMA
        # (matching _validate_from_dict_input lines 246-269).
        # Empty string = not specified.
        if self.dlm is None:
            raise ValueError("VersionSection: DLM cannot be None")
        if self.dlm:
            _dlm = self.dlm.upper()
            # F-I2-MD3-01: Validate DLM value first, then check LAS
            # version compatibility.  The previous if/elif structure
            # warned for LAS 1.2 with non-SPACE DLM but skipped the
            # validity rejection when the DLM was completely invalid
            # (e.g. "FOO").  Ensuring the validity check runs
            # unconditionally before the version-specific warning
            # guarantees that invalid DLMs are always rejected.
            # Mirroring the fix from _validate_from_dict_input
            # (lines 265-282).
            if _dlm not in {"SPACE", "TAB", "COMMA"}:
                raise ValueError(
                    f"VersionSection: invalid DLM value "
                    f"{self.dlm!r}.  Expected SPACE, TAB, or COMMA."
                )
        # Run warning-producing checks via validate().
        for issue in self.validate(complete=False):
            warnings.warn(issue, stacklevel=2)

    def to_dict(self) -> dict[str, str]:
        """Convert to legacy dict format for backward compatibility."""
        return {
            "VERS": self.vers,
            # F2-26: Emit "NO" when wrap is None — the writer's
            # write-time override sets wrap to "NO" on disk but prior
            # to first write the field may be None (e.g. from direct
            # construction).  Returning None violates the dict[str, str]
            # type contract.
            "WRAP": self.wrap if self.wrap is not None else "NO",
            "DLM": self.dlm,
        }

    @property
    def is_las30(self) -> bool:
        """Check if this is a LAS 3.0 file.

        Uses string prefix matching ('3' prefix on the version string).
        This is a deliberate design choice: LAS 3.x versions (3.0, 3.1,
        etc.) all share the same structural features. A version string
        like '3.1beta' or '3.0-draft' will match.
        """
        return self.vers.startswith("3")

    @property
    def delimiter_char(self) -> str:
        """Get the actual delimiter character for data parsing."""
        delimiter_map = {
            "SPACE": " ",
            "TAB": "\t",
            "COMMA": ",",
        }
        return delimiter_map.get(self.dlm.upper(), " ")


@dataclass
class WellSection:
    """LAS Well Information section (~W).

    All values are stored as strings to match original pylasdev behavior.
    """

    entries: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    descriptions: dict[str, str] = field(
        default_factory=dict
    )  # CWLS description text for well fields

    def validate(self, complete: bool = False) -> list[str]:
        """Validate well section fields.

        Args:
            complete: If True, also run deferred/semantic checks
                (STEP=0, NULL empty, STRT==STOP).

        Returns:
            List of issue strings (empty = no issues found).
            Mandatory field presence is checked separately
            by _validate_from_dict_input (from_dict path) and
            the parser (parse path) with version-aware messages.
        """
        issues: list[str] = []
        if not complete:
            return issues

        # STEP=0 check.
        _step_val = None
        for key, value in self.entries.items():
            if key.upper() == "STEP":
                _step_val = value
                break
        if _step_val is not None:
            try:
                if float(_step_val) == 0.0:
                    issues.append("STEP is zero — depth increment is invalid.")
            except (TypeError, ValueError):
                pass

        # NULL empty check — case-INSENSITIVE lookup (matching STEP above)
        # and distinguishing ABSENT from EMPTY-STRING.  N-I-09: the old
        # ``.get("NULL", "")`` conflated absent-with-empty — every NULL-less
        # file (and every write of a NULL-less model) reported a false
        # "NULL is an empty string" diagnostic — and case-variant keys
        # (``null``, reachable via from_dict with ``mnem_base=None`` or
        # direct construction) were invisible to the case-sensitive lookup.
        _null_val = None
        for key, value in self.entries.items():
            if key.upper() == "NULL":
                _null_val = value
                break
        if _null_val is not None and isinstance(_null_val, str) and not _null_val:
            issues.append("NULL is an empty string — null value is ambiguous.")

        # STRT==STOP check — case-INSENSITIVE lookup (matching STEP above).
        # N-I-09: lowercase ``strt``/``stop`` keys were invisible, silently
        # skipping the zero-depth-range validation on the from_dict
        # (mnem_base=None) and direct-construction paths.
        _strt_raw = None
        _stop_raw = None
        for key, value in self.entries.items():
            _upper = key.upper()
            if _upper == "STRT":
                _strt_raw = value
            elif _upper == "STOP":
                _stop_raw = value
        if _strt_raw is not None and _stop_raw is not None:
            try:
                if float(_strt_raw) == float(_stop_raw):
                    issues.append(f"STRT equals STOP ({_strt_raw}) — well has zero depth range.")
            except (TypeError, ValueError):
                pass

        return issues

    def __post_init__(self) -> None:
        """Validate entries dict contains only string keys (F-M-008).

        Direct construction bypasses parser and from_dict paths which
        already validate key types.  Non-string keys cause the writer
        to crash on ``key.upper()``.
        """
        if not isinstance(self.entries, dict):
            raise TypeError(
                f"WellSection: entries must be a dict, got {type(self.entries).__name__}"
            )
        for _key in self.entries:
            if not isinstance(_key, str):
                raise TypeError(
                    f"WellSection: all entry keys must be str, got {type(_key).__name__} ({_key!r})"
                )
        # N-I-19: Reject well-section entry KEYS whose content the parser
        # cannot roundtrip.  The parser's DATA_LINE_PATTERN mnemonic group
        # is ``[\w\-]+(?:\[\d+\])?`` — a key containing dots, spaces,
        # colons, or other punctuation (``GR.CO``, ``WELL NAME``, ``GR:1``)
        # is emitted by the writer and then fails the ~W line regex on
        # re-read, silently DROPPING the entry (parser logs "Non-matching
        # ~W line" and returns).  Mirror the M-03 curve/parameter whitelist
        # so well metadata is rejected at construction instead of lost.
        for _key in self.entries:
            if not _MNEMONIC_PATTERN.fullmatch(_key):
                raise ValueError(
                    f"WellSection: entry key {_key!r} contains "
                    f"characters the LAS parser cannot roundtrip.  "
                    f"Well keys must match {_MNEMONIC_PATTERN.pattern!r} "
                    f"(no dots, spaces, colons, or other punctuation)."
                )
        # M-14: coerce non-str values to str with a warning.
        for _key, _val in self.entries.items():
            if not isinstance(_val, str):
                warnings.warn(
                    f"WellSection: coercing non-str value for key "
                    f"{_key!r} from {type(_val).__name__} to str",
                    stacklevel=2,
                )
                self.entries[_key] = _safe_str(_val)
            # F-001: MAX_FIELD_LENGTH bypass for already-string values.
            # __setitem__ guards length on mutation (L804-808), but direct
            # construction with a long string skips both _safe_str() and
            # __setitem__ — the isinstance(…, str) gate above short-circuits.
            elif len(_val) > MAX_FIELD_LENGTH:
                raise ValueError(
                    f"WellSection: value length {len(_val)} for key "
                    f"{_key!r} exceeds maximum allowed ({MAX_FIELD_LENGTH})"
                )
        # I2F-05: Validate units and descriptions dicts.
        # WellSection.__post_init__ validates entries key/value types but
        # does NOT validate units dict or descriptions dict.  Non-string
        # VALUES cause writer crash via _sanitize_las_value() →
        # AttributeError on .replace().  Non-string KEYS are silently ignored.
        for _dict_name, _dict_ref in (("units", self.units), ("descriptions", self.descriptions)):
            if not isinstance(_dict_ref, dict):
                raise TypeError(
                    f"WellSection: {_dict_name} must be a dict, got {type(_dict_ref).__name__}"
                )
            for _dk, _dv in _dict_ref.items():
                if not isinstance(_dk, str):
                    raise TypeError(
                        f"WellSection: {_dict_name} key must be str, "
                        f"got {type(_dk).__name__} ({_dk!r})"
                    )
                if not isinstance(_dv, str):
                    raise TypeError(
                        f"WellSection: {_dict_name} value for key "
                        f"{_dk!r} must be str, "
                        f"got {type(_dv).__name__} ({_dv!r})"
                    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to legacy dict format, including non-empty units and descriptions."""
        result: dict[str, Any] = dict(self.entries)
        if self.units:
            result["units"] = dict(self.units)
        if self.descriptions:
            result["descriptions"] = dict(self.descriptions)
        return result

    def __getitem__(self, key: str) -> str:
        return self.entries[key]

    def __setitem__(self, key: str, value: str) -> None:
        # F-002: Guard against non-string keys — the writer calls
        # key.upper() which crashes on int/float/None keys.
        if not isinstance(key, str):
            raise TypeError(f"WellSection: keys must be str, got {type(key).__name__} ({key!r})")
        # F-006: Coerce non-str values via _safe_str(), matching
        # __post_init__ behaviour.  _safe_str() also enforces
        # MAX_FIELD_LENGTH and rejects non-finite floats / bytes,
        # so the manual length check below is redundant for
        # non-str values but preserved for already-str values
        # (which bypass _safe_str's length guard).
        if not isinstance(value, str):
            value = _safe_str(value)
        # F-06: Validate value length matches __post_init__ guard.
        # Direct API users and post-transform length changes bypass
        # parser's _validate_data_line_fields() mitigation.
        if len(value) > MAX_FIELD_LENGTH:
            raise ValueError(
                f"WellSection value length {len(value)} exceeds "
                f"maximum allowed ({MAX_FIELD_LENGTH})"
            )
        self.entries[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def get(self, key: str, default: str = "") -> str:
        return self.entries.get(key, default)


@dataclass
class ArrayElementInfo:
    """LAS 3.0 array element metadata for curves.

    Captures information like {A:0}, {A:5}, etc. from LAS 3.0 curves.
    """

    base_name: str = ""  # Base mnemonic without index (e.g., "NMR")
    index: int = 0  # Array index (e.g., 1, 2, 3)
    time_offset: float | None = None  # Time offset from first element (e.g., 0, 5, 10 ms)

    def validate(self, complete: bool = False) -> list[str]:
        """Validate array element info fields.

        Returns:
            List of issue strings (empty = no issues found).
            All structural checks raise during construction
            and are not duplicated here.
        """
        return []

    def __post_init__(self) -> None:
        """Validate base_name, index, and time_offset.

        Direct construction bypasses parser and from_dict validation.
        Non-finite time_offset values (NaN/Inf) silently propagate to
        LAS output.
        """
        if not self.base_name or not self.base_name.strip():
            raise ValueError(
                f"ArrayElementInfo: base_name must not be empty or "
                f"whitespace-only, got {self.base_name!r}"
            )
        # M-06: Accept numpy integer scalars (np.int64 is not an instance
        # of Python int) — from_dict and direct construction legitimately
        # pass numpy scalars.  Convert to Python int before the type check.
        self.index = _coerce_numpy_scalar(self.index)
        if type(self.index) is not int:
            raise TypeError(
                f"ArrayElementInfo: index must be int, got "
                f"{type(self.index).__name__} ({self.index!r})"
            )
        if self.index < 0:
            raise ValueError(f"ArrayElementInfo: index must be >= 0, got {self.index!r}")
        # M-06: Accept numpy float scalars for time_offset (np.float32 is
        # not an instance of Python float).
        if self.time_offset is not None:
            self.time_offset = _coerce_numpy_scalar(self.time_offset)
        if self.time_offset is not None and (
            not isinstance(self.time_offset, (int, float))
            or (isinstance(self.time_offset, float) and not math.isfinite(self.time_offset))
        ):
            raise ValueError(
                f"ArrayElementInfo: time_offset must be a finite number "
                f"or None, got {self.time_offset!r}"
            )
        if self.time_offset is not None and self.time_offset < 0:
            raise ValueError(
                f"ArrayElementInfo: time_offset must be >= 0, got {self.time_offset!r}"
            )


@dataclass
class CurveDefinition:
    """Single curve definition from ~C section.

    Supports LAS 1.2, 2.0, and 3.0 formats including array notation.
    """

    mnemonic: str
    unit: str = ""
    api_code: str = ""
    description: str = ""
    original_mnemonic: str = ""  # Pre-normalization name

    # LAS 3.0 specific fields
    data_format: str = ""  # F, E, S, or A (from {F}, {E}, {S}, {A:x})
    array_info: ArrayElementInfo | None = None  # For array curves like NMR[1]

    def validate(self, complete: bool = False) -> list[str]:
        """Validate curve definition fields.

        Returns:
            List of issue strings (empty = no issues found).
            All structural checks raise during construction
            and are not duplicated here.
        """
        return []

    def __post_init__(self) -> None:
        """Reject empty-or-whitespace mnemonic and invalid data_format.

        Empty mnemonics produce malformed LAS output (`` .UNIT  : DESC``).
        An invalid data_format propagates through the writer as a
        ``{INVALID}`` format specifier, producing corrupted LAS output.
        """
        # M-05: non-str mnemonic → raw AttributeError on .strip() below.
        # Guard the type first so callers get a clear contract error
        # instead of an unhandled AttributeError.
        if not isinstance(self.mnemonic, str):
            raise TypeError(
                f"CurveDefinition: mnemonic must be str, got {type(self.mnemonic).__name__}"
            )
        # F-I2-MD3-07: Reject mnemonics with embedded spaces.
        # strip() catches leading/trailing whitespace but "GR 1"
        # passes — embedded spaces survive to produce corrupted
        # LAS output (space is the field delimiter).
        # F-004: Reject \n and \r in mnemonics.  The writer replaces
        # them with spaces, producing corrupted output where field
        # boundaries shift.  strip() catches leading/trailing newlines
        # but embedded \n/\r (e.g. "GR\nCAL") pass the space/tab checks.
        # F-030: Reject dots in mnemonics.  The writer uses dot as a
        # structural separator in LAS output; the parser splits on the
        # first dot.  A mnemonic like "GR.CO" would be written as
        # "GR.CO.M/FT" and parsed back as mnemonic="GR", unit="CO.M/FT"
        # — causing roundtrip corruption.
        if (
            not self.mnemonic
            or not self.mnemonic.strip()
            or self.mnemonic != self.mnemonic.strip()
            or " " in self.mnemonic.strip()
            or "\t" in self.mnemonic.strip()
            or "\n" in self.mnemonic
            or "\r" in self.mnemonic
            or "." in self.mnemonic
        ):
            raise ValueError(
                f"CurveDefinition: mnemonic must not be empty, "
                f"whitespace-only, or contain spaces/tabs/newlines/dots, "
                f"got {self.mnemonic!r}"
            )
        # M-03: Whitelist against the parser's mnemonic grammar.  The
        # blacklist above catches spaces/tabs/newlines/dots but NOT other
        # punctuation — a colon mnemonic (``GR:1``) passes construction,
        # is emitted by the writer, and then the parser cannot match the
        # ~C line, silently dropping the curve AND its data column.
        # The parser's mnemonic group is ``[\w\-]+(?:\[\d+\])?``; use the
        # equivalent whitelist so model-accepted mnemonics always roundtrip.
        if not _MNEMONIC_PATTERN.fullmatch(self.mnemonic):
            raise ValueError(
                f"CurveDefinition: mnemonic {self.mnemonic!r} contains "
                f"characters the LAS parser cannot roundtrip.  Mnemonics "
                f"must match {_MNEMONIC_PATTERN.pattern!r}."
            )
        # M-05: Coerce non-str unit/api_code/description to str, mirroring
        # ParameterEntry.__post_init__.  Previously non-str values passed
        # construction and crashed the writer (_sanitize_las_value(123) →
        # AttributeError).  Also enforce MAX_FIELD_LENGTH for already-str
        # values (direct construction bypasses _safe_str's length guard).
        for _attr_name, _attr_val in (
            ("unit", self.unit),
            ("api_code", self.api_code),
            ("description", self.description),
        ):
            if not isinstance(_attr_val, str):
                warnings.warn(
                    f"CurveDefinition '{self.mnemonic}': coercing "
                    f"non-str {_attr_name} from "
                    f"{type(_attr_val).__name__} to str",
                    stacklevel=2,
                )
                setattr(self, _attr_name, _safe_str(_attr_val))
            elif len(_attr_val) > MAX_FIELD_LENGTH:
                raise ValueError(
                    f"CurveDefinition '{self.mnemonic}': {_attr_name} "
                    f"length {len(_attr_val)} exceeds maximum allowed "
                    f"({MAX_FIELD_LENGTH})"
                )
        # M-04: Validate unit composition.  Whitespace/colon/#/dot units
        # are truncated by the parser or produce ~C lines that cannot be
        # re-parsed — the whole curve + data column is silently dropped on
        # roundtrip.  The pattern matches the parser's WIDENED unit class
        # (N-I-22), so units like ``%``, ``°C``, ``ohm.m`` remain valid.
        if not _UNIT_PATTERN.fullmatch(self.unit):
            raise ValueError(
                f"CurveDefinition: invalid unit {self.unit!r} for curve "
                f"'{self.mnemonic}'.  Units may only contain word "
                f"characters, '-', '/', '.', '%', and '°' (matching the "
                f"parser's unit grammar)."
            )
        if self.data_format and self.data_format not in _VALID_DATA_FORMATS:
            raise ValueError(
                f"CurveDefinition: invalid data_format "
                f"'{self.data_format}' for curve '{self.mnemonic}'. "
                f"Valid values: {', '.join(sorted(_VALID_DATA_FORMATS))}"
            )
        # M-15: array_info must be ArrayElementInfo or None.
        if self.array_info is not None and not isinstance(self.array_info, ArrayElementInfo):
            raise TypeError(
                f"CurveDefinition: array_info must be ArrayElementInfo "
                f"or None, got {type(self.array_info).__name__}"
            )
        # M-17: Cross-check bracket-notation mnemonic against
        # array_info.base_name (same check the from_dict path applies at
        # lines ~3042-3052).  A mismatched array_info silently corrupts
        # output on write — the writer rebuilds the mnemonic from
        # array_info.index (W-09) and _las30_data groups array curves by
        # base_name, so a mismatched element is validated under a group
        # it will not belong to after re-parse.  Warn (mirroring
        # from_dict) rather than raise.
        if self.array_info is not None and "[" in self.mnemonic:
            _mnem_base = self.mnemonic.split("[", 1)[0]
            if _mnem_base != self.array_info.base_name:
                warnings.warn(
                    f"CurveDefinition: mnemonic {self.mnemonic!r} uses "
                    f"array notation but array_info.base_name is "
                    f"{self.array_info.base_name!r}.  Cross-check "
                    f"mismatch may indicate malformed input.",
                    UserWarning,
                    stacklevel=2,
                )
        # M-86: Reject the AMBIGUOUS state — bracketed mnemonic +
        # data_format="A" + array_info=None.  Per _detect_string_curves,
        # data_format="A" WITHOUT array_info is a STRING curve (the model
        # accepts it), but the writer emits "NMR[1]  : desc {A}" and the
        # parser UNCONDITIONALLY fabricates ArrayElementInfo from any
        # bracketed mnemonic — reclassifying the curve to NUMERIC on
        # re-read and silently destroying the string data (numeric-looking
        # values fully silent; non-numeric → generic conversion warning).
        # The writer emits IDENTICAL output with/without array_info
        # (ambiguous signal), so the file cannot distinguish a genuine
        # numeric array (array_info set) from a string curve.  Reject the
        # ambiguous state loudly: a numeric array must set array_info; a
        # string curve must use data_format="S" (which the parser
        # classifies string regardless of array_info) or a non-bracket
        # mnemonic.
        if (
            "[" in self.mnemonic
            and (self.data_format or "").upper() == "A"
            and self.array_info is None
        ):
            raise ValueError(
                f"CurveDefinition: mnemonic {self.mnemonic!r} with "
                f"data_format='A' but no array_info is ambiguous and "
                f"cannot roundtrip.  The parser fabricates a phantom "
                f"ArrayElementInfo for bracketed mnemonics, reclassifying "
                f"this string curve to numeric on re-read and destroying "
                f"its data.  Set array_info (numeric array) or use "
                f"data_format='S' (string curve)."
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mnemonic": self.mnemonic,
            "unit": self.unit,
            "api_code": self.api_code,
            "description": self.description,
        }
        if self.original_mnemonic:
            result["original_mnemonic"] = self.original_mnemonic
        if self.data_format:
            result["data_format"] = self.data_format
        if self.array_info:
            result["array_info"] = {
                "base_name": self.array_info.base_name,
                "index": self.array_info.index,
                "time_offset": self.array_info.time_offset,
            }
        return result

    @property
    def is_array_element(self) -> bool:
        """Check if this curve is part of an array."""
        return self.array_info is not None

    @property
    def base_mnemonic(self) -> str:
        """Get base mnemonic for array curves, or regular mnemonic otherwise."""
        if self.array_info:
            return self.array_info.base_name
        return self.mnemonic


@dataclass
class ParameterZone:
    """LAS 3.0 zone association for parameters.

    Parameters can be associated with zones via pipe notation: | Zone[1]
    """

    zone_name: str = ""
    zone_index: int | None = None

    def validate(self, complete: bool = False) -> list[str]:
        """Validate parameter zone fields.

        Returns:
            List of issue strings (empty = no issues found).
            Basic type/value checks raise during construction.
        """
        issues: list[str] = []
        # zone_name: warn if None or empty.
        if self.zone_name is None:
            issues.append(
                "ParameterZone: zone_name is None.  Zone names "
                "must be non-empty strings for valid LAS output."
            )
        elif isinstance(self.zone_name, str) and not self.zone_name.strip():
            issues.append(
                f"ParameterZone: zone_name {self.zone_name!r} is "
                f"empty or whitespace-only.  Zone names must be "
                f"non-empty strings for valid LAS output."
            )
        return issues

    def __post_init__(self) -> None:
        """Validate zone_index is int or None (F-M-010).

        Direct construction bypasses parser and from_dict paths which
        already validate the zone_index type.  A non-int value would
        pass silently and produce incorrect output.
        """
        # M-06: Accept numpy integer scalars (np.int64 is not an instance
        # of Python int).  Convert to Python int before the type check.
        if self.zone_index is not None:
            self.zone_index = _coerce_numpy_scalar(self.zone_index)
        if self.zone_index is not None and type(self.zone_index) is not int:
            raise TypeError(
                f"ParameterZone: zone_index must be int or None, "
                f"got {type(self.zone_index).__name__} "
                f"({self.zone_index!r})"
            )
        if self.zone_index is not None and self.zone_index < 0:
            raise ValueError(f"ParameterZone: zone_index must be >= 0, got {self.zone_index!r}")
        # M-12/M-65/L-124: Validate zone_name against the parser's zone
        # grammar so a zone association survives write→read roundtrip.
        # - A pipe ('|') in zone_name is escaped by the writer (\|) and
        #   unescaped by the parser, so it roundtrips — but warn loudly
        #   that it is non-standard.
        # - Characters outside the roundtrip grammar (colon, semicolon,
        #   dot, slash, brackets, etc.) previously made the ENTIRE
        #   '| Zone[N]' association unparseable on re-read: the zone was
        #   silently dropped and the raw '| Zone[N]' text leaked into the
        #   parameter description.  The parser's zone grammar is now
        #   tolerant of the M-65 punctuation class ([:;./] + escaped
        #   pipes), so those roundtrip; warn loudly for characters that
        #   STILL cannot roundtrip (brackets '['/']' conflict with the
        #   zone index notation, and any other stray punctuation).
        # - Trailing whitespace (L-124) breaks the '$'-anchored zone
        #   pattern; normalize it away with a warning.
        if isinstance(self.zone_name, str):
            _normalized = self.zone_name.strip()
            if _normalized != self.zone_name:
                warnings.warn(
                    f"ParameterZone: zone_name {self.zone_name!r} has "
                    f"leading/trailing whitespace.  Normalizing to "
                    f"{_normalized!r} so the zone association "
                    f"roundtrips.",
                    stacklevel=2,
                )
                self.zone_name = _normalized
            if "|" in self.zone_name:
                warnings.warn(
                    f"ParameterZone: zone_name {self.zone_name!r} contains "
                    f"a pipe ('|').  The writer escapes it and the parser "
                    f"restores it, so the zone roundtrips, but a pipe in a "
                    f"zone name is non-standard.",
                    stacklevel=2,
                )
            _non_roundtrip = re.findall(r"[^A-Za-z0-9_\-.:;/ ]", self.zone_name)
            if _non_roundtrip:
                warnings.warn(
                    f"ParameterZone: zone_name {self.zone_name!r} contains "
                    f"characters {sorted(set(_non_roundtrip))} that the "
                    f"LAS 3.0 zone grammar cannot roundtrip (brackets "
                    f"'['/']' conflict with the zone index notation).  "
                    f"The zone association may not survive a "
                    f"write→read roundtrip.",
                    stacklevel=2,
                )
        # Run warning-producing checks via validate().
        for issue in self.validate(complete=False):
            warnings.warn(issue, stacklevel=2)


@dataclass
class ParameterEntry:
    """Single parameter entry from ~P section.

    Supports LAS 1.2, 2.0, and 3.0 formats including array notation and zones.
    """

    mnemonic: str
    unit: str = ""
    value: str = ""
    description: str = ""
    data_format: str = ""  # Format specifier ({F}, {E}, etc.) from parser

    # LAS 3.0 specific fields
    array_index: int | None = None  # For RUN[1], RUN[2], etc.
    zone: ParameterZone | None = None  # Zone association via pipe notation
    # F-053: Per-section parameter grouping.  When parameters are parsed
    # from a LAS 3.0 typed section (e.g., ~Core_Parameter), section_type
    # records the originating section type (e.g., "CORE") so the writer
    # can reconstruct per-section parameter sections on roundtrip.
    # Parameters from a standard ~P/~Parameter section have section_type=None.
    section_type: str | None = None

    def validate(self, complete: bool = False) -> list[str]:
        """Validate parameter entry fields.

        Returns:
            List of issue strings (empty = no issues found).
            Mnemonic, data_format, array_index, zone, and section_type
            checks raise during construction and are not duplicated here.
            Unit/value/description coercion warnings fire during
            __post_init__ before coercion.
        """
        return []

    def __post_init__(self) -> None:
        """Reject empty-or-whitespace mnemonic (F-M10).

        Empty mnemonics produce malformed LAS output.
        """
        # M-05: non-str mnemonic → raw AttributeError on .strip() below.
        # Guard the type first so callers get a clear contract error.
        if not isinstance(self.mnemonic, str):
            raise TypeError(
                f"ParameterEntry: mnemonic must be str, got {type(self.mnemonic).__name__}"
            )
        # F-I2-MD3-07: Reject mnemonics with embedded spaces.
        # strip() catches leading/trailing whitespace but "GR 1"
        # passes — embedded spaces survive to produce corrupted
        # LAS output (space is the field delimiter).
        # F-004: Reject \n and \r in mnemonics — same gap as
        # CurveDefinition.__post_init__ above.
        # F-030: Reject dots in mnemonics — same roundtrip corruption
        # as CurveDefinition (writer uses dot as structural separator).
        if (
            not self.mnemonic
            or not self.mnemonic.strip()
            or self.mnemonic != self.mnemonic.strip()
            or " " in self.mnemonic.strip()
            or "\t" in self.mnemonic.strip()
            or "\n" in self.mnemonic
            or "\r" in self.mnemonic
            or "." in self.mnemonic
        ):
            raise ValueError(
                f"ParameterEntry: mnemonic must not be empty, "
                f"whitespace-only, or contain spaces/tabs/newlines/dots, "
                f"got {self.mnemonic!r}"
            )
        # M-03: Whitelist against the parser's mnemonic grammar (same as
        # CurveDefinition) — colon/pipe/#/etc. parameter mnemonics are
        # emitted by the writer and then cannot be re-parsed, silently
        # dropping the parameter on roundtrip.
        if not _MNEMONIC_PATTERN.fullmatch(self.mnemonic):
            raise ValueError(
                f"ParameterEntry: mnemonic {self.mnemonic!r} contains "
                f"characters the LAS parser cannot roundtrip.  Mnemonics "
                f"must match {_MNEMONIC_PATTERN.pattern!r}."
            )
        # F-21: Validate that unit, value, and description are strings.
        # Non-str values (int, float, None) pass silently at construction
        # and crash downstream on .upper() or .strip() calls.  Warn and
        # coerce via str() to maintain backward compatibility.
        for _attr_name, _attr_val in (
            ("unit", self.unit),
            ("value", self.value),
            ("description", self.description),
        ):
            if not isinstance(_attr_val, str):
                warnings.warn(
                    f"ParameterEntry '{self.mnemonic}': coercing "
                    f"non-str {_attr_name} from "
                    f"{type(_attr_val).__name__} to str",
                    stacklevel=2,
                )
                setattr(self, _attr_name, _safe_str(_attr_val))
        # M-04: Validate unit composition (after F-21 coercion above so
        # the unit is always a str here).  A parameter unit with
        # whitespace/colon/#/dot is truncated by the parser or produces a
        # ~P line that cannot be re-parsed — value pollution on roundtrip
        # (e.g. ``unit='US', value='M sand'`` written unparseable).  The
        # pattern matches the parser's WIDENED unit class (N-I-22).
        if not _UNIT_PATTERN.fullmatch(self.unit):
            raise ValueError(
                f"ParameterEntry: invalid unit {self.unit!r} for parameter "
                f"'{self.mnemonic}'.  Units may only contain word "
                f"characters, '-', '/', '.', '%', and '°' (matching the "
                f"parser's unit grammar)."
            )
        # F-M-009 / M-11: Validate data_format when provided, mirroring
        # CurveDefinition.__post_init__ (lines 619-624) and the
        # from_dict parameter path (_create_parameter_entry, lines
        # 338-352).  Single-character values are validated against the
        # valid set; multi-character strings like "DD/MM/YYYY" are
        # metadata descriptors, not LAS format specifiers.  The writer
        # emits any non-empty data_format braced ({DD/MM/YYYY}), which
        # re-parses to data_format='' with the brace text polluting the
        # description — so direct construction must apply the SAME
        # normalization as from_dict: truncate Fortran-style extended
        # codes (F8.3 → F) and warn-and-clear other multi-char values,
        # keeping roundtrips deterministic across construction paths.
        # MOD-02: Uppercase before validation — mirroring from_dict's
        # _create_parameter_entry (and the curve path), so direct
        # construction accepts lowercase 'f' the same way from_dict does
        # and extended codes like "f8.3" normalize to "F".
        self.data_format = self.data_format.upper()
        if self.data_format and len(self.data_format) > 1:
            if _EXTENDED_FORMAT_SPEC_RE.match(self.data_format):
                self.data_format = self.data_format[0]
            else:
                warnings.warn(
                    f"Ignoring multi-character data_format "
                    f"'{self.data_format}' for parameter "
                    f"'{self.mnemonic}'.  "
                    f"Only single-letter LAS format codes or "
                    f"Fortran-style extended codes are valid; "
                    f"clearing to empty string.",
                    UserWarning,
                    stacklevel=2,
                )
                self.data_format = ""
        if (
            self.data_format
            and len(self.data_format) == 1
            and self.data_format not in _VALID_DATA_FORMATS
        ):
            # MOD-02: warn-and-clear instead of raising — matching the
            # from_dict parameter path (_create_parameter_entry) and the
            # parser (parser.py:2886-2896).  A single-char non-format
            # code like 'X' previously raised on direct construction
            # while the parser warn-and-cleared the same input.
            warnings.warn(
                f"ParameterEntry: invalid data_format "
                f"'{self.data_format}' for parameter "
                f"'{self.mnemonic}'.  Valid values: "
                f"{', '.join(sorted(_VALID_DATA_FORMATS))}. "
                f"Clearing to empty string.",
                UserWarning,
                stacklevel=2,
            )
            self.data_format = ""
        # M-06: Accept numpy integer scalars for array_index (np.int64 is
        # not an instance of int).  Convert to Python int before the
        # type check so dataclass field contracts hold.
        if self.array_index is not None:
            self.array_index = _coerce_numpy_scalar(self.array_index)
        # F-005: Validate array_index type (int | None).
        if self.array_index is not None and type(self.array_index) is not int:
            raise TypeError(
                f"ParameterEntry.array_index must be int or None, "
                f"got {type(self.array_index).__name__}"
            )
        # M-41: Reject negative array_index.  The writer emits ``RUN[-1]``
        # (W-08 guard only checks "[" not in mnemonic) but the parser's
        # mnemonic grammar (``[\w\-]+(?:\[\d+\])?``) cannot match a
        # negative index — the parameter line is skipped and the parameter
        # silently vanishes on write→read.  Sibling fields
        # (ArrayElementInfo.index, ParameterZone.zone_index) reject
        # negatives; ParameterEntry was the only field without the guard.
        if self.array_index is not None and self.array_index < 0:
            raise ValueError(f"ParameterEntry.array_index must be >= 0, got {self.array_index!r}")
        # M-42: Cross-check bracket-notation mnemonic against array_index
        # (parameter twin of M-17's CurveDefinition check).  A parameter
        # like ``RUN[3]`` with array_index=5 is accepted, the writer's
        # ``"[" not in mnemonic`` guard skips the W-08 bracket append and
        # emits ``RUN[3]``, and re-parse reconstructs array_index=3 — a
        # silent field divergence.  Warn (mirroring from_dict's F-17 curve
        # warning) rather than raise, so internally-consistent pairs
        # (bracket == array_index) and non-bracket mnemonics stay silent.
        if "[" in self.mnemonic and self.array_index is not None:
            _mnem_index = self.mnemonic.split("[", 1)[1].rstrip("]")
            if _mnem_index.isdigit() and int(_mnem_index) != self.array_index:
                warnings.warn(
                    f"ParameterEntry: mnemonic {self.mnemonic!r} uses "
                    f"array notation with index {_mnem_index} but "
                    f"array_index is {self.array_index!r}.  The writer "
                    f"emits the mnemonic's bracket index; the array_index "
                    f"field will diverge on re-parse.",
                    UserWarning,
                    stacklevel=2,
                )
        # F-005: Validate zone type (ParameterZone | None).
        if self.zone is not None and not isinstance(self.zone, ParameterZone):
            raise TypeError(
                f"ParameterEntry.zone must be ParameterZone or None, got {type(self.zone).__name__}"
            )
        # F-064: Validate section_type — reject newline/carriage-return/tilde,
        # normalize whitespace-only to None.  Matches _create_parameter_entry
        # validation (lines 91-118).
        if self.section_type is not None:
            if not isinstance(self.section_type, str):
                self.section_type = _safe_str(self.section_type)
            _stripped = self.section_type.strip()
            if not _stripped:
                self.section_type = None
            else:
                self.section_type = _stripped
                _sec_str = str(self.section_type)
                # M-27: Reject whitespace and pipe in addition to
                # newline/tilde (matches _create_parameter_entry).
                if (
                    "\n" in _sec_str
                    or "\r" in _sec_str
                    or "~" in _sec_str
                    or " " in _sec_str
                    or "\t" in _sec_str
                    or "|" in _sec_str
                ):
                    raise ValueError(
                        f"ParameterEntry.section_type contains invalid "
                        f"characters (whitespace, newline, tilde, or pipe): "
                        f"{_sec_str!r}"
                    )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mnemonic": self.mnemonic,
            "unit": self.unit,
            "value": self.value,
            "description": self.description,
        }
        if self.data_format:
            result["data_format"] = self.data_format
        if self.array_index is not None:
            result["array_index"] = self.array_index
        if self.zone:
            result["zone"] = {
                "zone_name": self.zone.zone_name,
                "zone_index": self.zone.zone_index,
            }
        if self.section_type is not None:
            result["section_type"] = self.section_type
        return result

    @property
    def base_mnemonic(self) -> str:
        """Get base mnemonic without array index."""
        if self.array_index is not None and "[" in self.mnemonic:
            return self.mnemonic.split("[")[0]
        return self.mnemonic


@dataclass(eq=False)
class DataSection:
    """~A data section (LAS 1.2/2.0) and LAS 3.0 typed data sections.

    In LAS 1.2/2.0 files, the data section is a single ``~A`` block.
    LAS 3.0 can have multiple typed data sections (``~ASCII``, ``~Log_Data``,
    ``~Core_Data``, etc.), each potentially with different curve sets or
    depth ranges.
    """

    name: str = ""  # Section name from ~A line (e.g., "ASCII" or custom name)
    section_type: str = "LOG_DATA"  # Section type: LOG_DATA, CORE_DATA, DRILLING_DATA, etc.
    curves_order: list[str] = field(default_factory=list)
    data: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    # F-36: dtype=object used at all construction sites; np.str_ annotation
    # was misleading and never matched actual usage.
    string_data: dict[str, NDArray[np.object_]] = field(
        default_factory=dict
    )  # For {S} format curves
    section_curves: list[CurveDefinition] = field(
        default_factory=list
    )  # Per-section curve definitions

    # M-05: from_dict re-validation control flag.  When True,
    # __post_init__ suppresses the NaN/Inf warning — the from_dict
    # path has already validated input; NaN is standard missing-data
    # representation in LAS.  Consistent with LASFile._from_dict.
    _from_dict: bool = field(default=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert DataSection to dict for serialization."""
        return {
            "name": self.name,
            "section_type": self.section_type,
            "curves_order": list(self.curves_order),
            "data": {k: v.copy() for k, v in self.data.items()},
            "string_data": {k: v.copy() for k, v in self.string_data.items()},
            "section_curves": [c.to_dict() for c in self.section_curves],
        }

    def validate(self, complete: bool = False) -> list[str]:
        """Validate data section fields.

        .. note::

            This method **mutates** ``self.data`` and ``self.string_data``
            in-place: any non-``numpy.ndarray`` values are coerced via
            ``np.asarray()`` so that dtype checks are reliable.  This
            conversion is lossless — lists and array-likes are wrapped
            into equivalent numpy arrays without changing data values.

        Args:
            complete: If True, also run deferred/semantic checks
                (NaN/Inf in numeric data, dtype validation,
                uncovered curves).

        Returns:
            List of issue strings (empty = no issues found).
            Basic structural checks (type, keys, lengths, data_format)
            raise during construction and are not duplicated here.
        """
        issues: list[str] = []
        # --- dtype validation for data arrays ---
        # EXT-04: {I} curves with a fractional declared NULL are stored as
        # object arrays (exact Python ints for data values, float sentinel
        # for null cells) — exempt them from the numeric-dtype check,
        # mirroring the LASFile.validate exemption for the top-level logs.
        _fmt_by_mnem = {c.mnemonic: c.data_format for c in self.section_curves}
        for _k, _arr in self.data.items():
            if not isinstance(_arr, np.ndarray):
                _arr = self.data[_k] = np.asarray(_arr)
            if _arr.dtype == object and _fmt_by_mnem.get(_k) == "I":
                continue
            if not np.issubdtype(_arr.dtype, np.number):
                issues.append(
                    f"DataSection '{self.name}': curve '{_k}' in 'data' "
                    f"has non-numeric dtype ({_arr.dtype}).  'data' "
                    f"arrays must be numeric."
                )

        # --- dtype validation for string_data arrays ---
        for _sk, _sarr in self.string_data.items():
            if not isinstance(_sarr, np.ndarray):
                _sarr = self.string_data[_sk] = np.asarray(_sarr)
            if np.issubdtype(_sarr.dtype, np.number):
                issues.append(
                    f"DataSection '{self.name}': curve '{_sk}' in "
                    f"'string_data' has numeric dtype ({_sarr.dtype}).  "
                    f"'string_data' arrays must be non-numeric."
                )

        # --- NaN/Inf validation for numeric data arrays (complete only) ---
        if not complete:
            return issues

        for _k, _arr in self.data.items():
            if isinstance(_arr, np.ndarray) and _arr.dtype.kind in ("f", "c"):
                if not np.all(np.isfinite(_arr)):
                    issues.append(
                        f"DataSection '{self.name}': curve '{_k}' in "
                        f"'data' contains non-finite values (NaN/Inf)."
                    )

        # --- uncovered curves ---
        curve_set = set(self.curves_order)
        data_keys = set(self.data.keys())
        string_keys = set(self.string_data.keys())
        if self.data or self.string_data:
            uncovered = curve_set - data_keys - string_keys
            if uncovered:
                issues.append(
                    f"DataSection '{self.name}': curve(s) "
                    f"{sorted(uncovered)} appear in curves_order but "
                    f"have no data in 'data' or 'string_data'.  The "
                    f"writer will pad these curves with null_value."
                )
            # I2-13 (models side): detect ORPHANED data keys — data that
            # curves_order no longer covers after a post-construction
            # mutation (e.g. ``del ds.curves_order[0]``).  __post_init__
            # validates this at construction; validate() — which runs on
            # post-construction state — previously did not, so a curve
            # whose data survived but whose order entry was removed passed
            # silently and the writer dropped the column.
            orphaned_data = (data_keys | string_keys) - curve_set
            if orphaned_data:
                issues.append(
                    f"DataSection '{self.name}': curve(s) "
                    f"{sorted(orphaned_data)} have data in 'data' or "
                    f"'string_data' but are NOT in curves_order.  The "
                    f"writer will not emit these columns."
                )

        # I2-13 (models side): post-construction curves_order mutation
        # (reverse/insert/reorder) desyncs the order from section_curves,
        # silently swapping columns on write.  __post_init__ validates the
        # positional alignment at construction; validate(complete=True) —
        # which the writer calls before emitting — must re-check it so a
        # reordered curves_order produces a clear validation issue instead
        # of a silent column swap.
        if self.section_curves and self.curves_order:
            if len(self.section_curves) != len(self.curves_order):
                issues.append(
                    f"DataSection '{self.name}': section_curves length "
                    f"({len(self.section_curves)}) does not match "
                    f"curves_order length ({len(self.curves_order)}).  "
                    f"Post-construction curves_order mutation may have "
                    f"desynced the order from the curve definitions."
                )
            else:
                for _i, (_order_name, _sc) in enumerate(
                    zip(self.curves_order, self.section_curves, strict=False)
                ):
                    if _order_name != _sc.mnemonic:
                        issues.append(
                            f"DataSection '{self.name}': curves_order[{_i}] "
                            f"= {_order_name!r} does not match "
                            f"section_curves[{_i}].mnemonic = "
                            f"{_sc.mnemonic!r}.  Post-construction "
                            f"curves_order mutation has desynced the "
                            f"column order from the curve definitions."
                        )
                        break

        # I2F-06: Validate data_format vs data/string_data placement.
        # __post_init__ checks this but validate() — which runs on
        # post-construction mutations — does not.  Adding an S-format
        # curve to data or an F-format curve to string_data after
        # construction would pass validate() silently.
        for _sc in self.section_curves:
            _df = _sc.data_format
            _mnem = _sc.mnemonic
            if not _df:
                continue
            if _mnem in data_keys and (_df == "S" or (_df == "A" and not _sc.is_array_element)):
                issues.append(
                    f"DataSection '{self.name}': curve '{_mnem}' has "
                    f"data_format='{_df}' but is in data (numeric). "
                    f"String-format curves must be in string_data."
                )
            if _df not in ("S", "A") and _mnem in string_keys:
                issues.append(
                    f"DataSection '{self.name}': curve '{_mnem}' has "
                    f"data_format='{_df}' but is in string_data. "
                    f"Numeric-format curves must be in data."
                )

        return issues

    def __setattr__(self, name: str, value: Any) -> None:
        """MOD-22/I2-12: guarded assignment for ``data``/``string_data``
        (dicts) and ``curves_order`` (list).

        Post-construction wholesale reassignment (``ds.data = {...}``)
        previously replaced the ``_GuardedDict`` wrapper with a plain
        dict — bypassing every guard (M-43 length check, MOD-17 ndim,
        MOD-23 array-like).  validate() reported 0 issues and the writer
        silently null-padded fabricated -999.25 rows.  Re-wrap plain-dict
        assignments through ``_GuardedDict`` (which validates str keys,
        the 1-D array-like contract, and resulting-state length
        consistency), mirroring ``DevFile.__setattr__`` (M-45/M-49).
        Idempotent: re-wrapping an already-wrapped ``_GuardedDict``
        (the __post_init__ path) copies the content and re-validates.

        I2-12: ``curves_order`` is a plain list field whose __post_init__
        element-type guard is bypassed by post-construction mutation
        (``ds.curves_order.append(42)`` corrupted the writer's per-section
        column count).  Re-wrap wholesale list assignments through
        ``_GuardedList`` so element types are validated at every mutation
        entry point.
        """
        if name in ("data", "string_data"):
            if value is None:
                super().__setattr__(name, None)
                return
            if not isinstance(value, dict):
                raise TypeError(f"DataSection: {name} must be a dict, got {type(value).__name__}")
            super().__setattr__(
                name,
                _GuardedDict(value, _container_name=f"DataSection.{name}"),
            )
            return
        if name == "curves_order":
            if value is None:
                super().__setattr__(name, None)
                return
            if not isinstance(value, list):
                raise TypeError(
                    f"DataSection: curves_order must be a list, got {type(value).__name__}"
                )
            if not isinstance(value, _GuardedList):
                value = _GuardedList(
                    value,
                    _container_name="DataSection.curves_order",
                    _expected_type=str,
                )
            super().__setattr__(name, value)
            return
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        """Validate invariants after construction (F-001).

        DataSection is a public API type — direct construction bypasses
        the validation in ``LASFile.from_dict``.  This __post_init__
        catches invalid state that would cause silent data loss on write.
        """
        # Deferred import to avoid circular dependencies.
        from .exceptions import LASDataError

        # M-25: Reject bytes for curves_order — matches top-level
        # guard in from_dict at line 1768.  ``set(b"GR")`` produces
        # ``{71, 82}`` (integers), corrupting all downstream checks.
        # ``str(b"GR")`` produces ``"b'GR'"`` which corrupts column
        # headers.  Reject with the same error type as the top-level.
        if isinstance(self.curves_order, bytes):
            raise TypeError(
                "DataSection curves_order must be a list of strings, "
                "got bytes.  Decode to str first: "
                "curves_order.decode('utf-8')"
            )

        # N-I-12: Validate per-element types in curves_order for direct
        # construction.  LASFile.__post_init__ F-13 validates the top-level
        # curves_order; from_dict validates per-section elements (F-I2E-05);
        # DataSection.__post_init__ previously did not.  A non-str element
        # (int, None, float) crashes downstream with a raw TypeError at
        # _ARRAY_MNEMONIC_RE.match (LASFile.__post_init__) instead of a
        # clear validation error.
        for _i, _name in enumerate(self.curves_order):
            if not isinstance(_name, str):
                raise LASDataError(
                    f"DataSection '{self.name}': curves_order[{_i}] must "
                    f"be str, got {type(_name).__name__}: {_name!r}"
                )

        curve_set = set(self.curves_order)

        # F-105: Reject duplicate curve names in curves_order.
        # The from_dict path has explicit dedup checks (lines 1171-1181),
        # but direct DataSection construction bypasses those entirely.
        # ``DataSection(curves_order=["GR","GR","DT"])`` passes silently
        # and the writer produces duplicate columns in output.
        if len(self.curves_order) != len(curve_set):
            _seen: set[str] = set()
            _dups: list[str] = []
            for c in self.curves_order:
                if c in _seen:
                    _dups.append(c)
                else:
                    _seen.add(c)
            raise LASDataError(
                f"DataSection '{self.name}': duplicate curve names in curves_order: {_dups}"
            )

        # F-M06 / F-M09: Validate and normalize section_type.
        # _create_parameter_entry validates section_type for ParameterEntry
        # (rejects newline/tilde) and normalizes whitespace-only to None;
        # DataSection previously skipped both checks.  Mirror that
        # validation here.
        if self.section_type is not None:
            _stripped = self.section_type.strip()
            if not _stripped:
                # Whitespace-only → normalize to empty string.
                # _safe_str ensures non-None; empty is a valid
                # "no particular section type" value.
                self.section_type = ""
            elif (
                "\n" in _stripped
                or "\r" in _stripped
                or "~" in _stripped
                or " " in _stripped
                or "\t" in _stripped
                or "|" in _stripped
            ):
                # M-27: Reject whitespace and pipe in addition to
                # newline/tilde (mirrors _create_parameter_entry and
                # ParameterEntry.__post_init__).  A section_type with
                # embedded spaces produces a broken ``~MY CORE`` header
                # that the parser misroutes; the pipe is a structural
                # delimiter in LAS 3.0 headers.
                raise LASDataError(
                    f"DataSection '{self.name}': section_type contains "
                    f"invalid characters (whitespace, newline, tilde, or "
                    f"pipe): {self.section_type!r}.  section_type must be "
                    f"a LAS identifier (alphanumeric + underscore)."
                )
            else:
                self.section_type = _stripped

        # M-84: Normalize bare known section types to their canonical
        # *_DATA form.  The parser's _SECTION_TYPE_MAP accepts both
        # "CORE" and "CORE_DATA" (mapping both to "CORE_DATA"), and a
        # from_dict/direct model may carry the bare form.  The writer's
        # _section_type_to_prefix only recognizes the *_DATA-suffixed
        # forms; a bare "CORE" falls back to "A" with a warning and the
        # re-read section_type silently becomes LOG_DATA.  Normalize at
        # the model so the roundtrip preserves the declared type.
        if self.section_type:
            _bare = _BARE_SECTION_TYPE_TO_DATA.get(self.section_type)
            if _bare is not None and _bare != self.section_type:
                self.section_type = _bare

        # data keys ⊆ curves_order
        data_keys = set(self.data.keys())
        orphaned_data = data_keys - curve_set
        if orphaned_data:
            raise LASDataError(
                f"DataSection '{self.name}': data keys not in curves_order: {sorted(orphaned_data)}"
            )

        # string_data keys ⊆ curves_order
        string_keys = set(self.string_data.keys())
        orphaned_string = string_keys - curve_set
        if orphaned_string:
            raise LASDataError(
                f"DataSection '{self.name}': string_data keys not in "
                f"curves_order: {sorted(orphaned_string)}"
            )

        # section_curves length must match curves_order (when specified)
        if self.section_curves and len(self.section_curves) != len(self.curves_order):
            raise LASDataError(
                f"DataSection '{self.name}': section_curves length "
                f"({len(self.section_curves)}) does not match curves_order "
                f"length ({len(self.curves_order)})"
            )

        # F-M04 / F-M21: Validate positional mnemonic alignment between
        # section_curves and curves_order.  The from_dict path checks
        # this (L1351-1360); direct DataSection construction previously
        # bypassed.  Swapped mnemonics produce silently corrupted output.
        if self.section_curves:
            for _i, (_order_name, _sc) in enumerate(
                zip(self.curves_order, self.section_curves, strict=True)
            ):
                if _order_name != _sc.mnemonic:
                    raise LASDataError(
                        f"DataSection '{self.name}': "
                        f"curves_order[{_i}] = {_order_name!r} "
                        f"does not match "
                        f"section_curves[{_i}].mnemonic = "
                        f"{_sc.mnemonic!r}"
                    )

        # data and string_data must be disjoint — a curve name cannot
        # appear in both collections (writer silently picks one).
        colliding = data_keys & string_keys
        if colliding:
            raise LASDataError(
                f"DataSection '{self.name}': curve(s) {sorted(colliding)} "
                f"appear in both data and string_data.  Each curve must "
                f"be in exactly one collection."
            )

        # F-027 / F-079: Validate within-group array lengths.
        # The from_dict path has cross-array length checks (lines 1292-1308),
        # but direct DataSection construction bypasses those entirely.
        # Arrays of different lengths pass silently — the writer pads
        # shorter arrays with null_value, producing semantically incorrect
        # output.
        if self.data:
            # M-18: Guard against 0-d numpy arrays — len(np.array(5.0))
            # raises TypeError: len() of unsized object.  0-d arrays
            # are treated as single-element arrays.
            _data_lengths = {
                name: (1 if isinstance(arr, np.ndarray) and arr.ndim == 0 else len(arr))
                for name, arr in self.data.items()
            }
            if len(set(_data_lengths.values())) > 1:
                raise LASDataError(
                    f"DataSection '{self.name}' has inconsistent array lengths: {_data_lengths}"
                )
        if self.string_data:
            _string_lengths = {
                name: (1 if isinstance(arr, np.ndarray) and arr.ndim == 0 else len(arr))
                for name, arr in self.string_data.items()
            }
            if len(set(_string_lengths.values())) > 1:
                raise LASDataError(
                    f"DataSection '{self.name}' has inconsistent "
                    f"string_data array lengths: {_string_lengths}"
                )

        # F-M22: Cross-group row-count validation.
        # The from_dict path validates this (L1437-1445); direct
        # DataSection construction previously bypassed.  The writer's
        # _format_data_rows uses max() across all arrays — mismatched
        # row counts between data and string_data produce semantically
        # incorrect null-padded output.
        if self.data and self.string_data:
            # M-18: Guard against 0-d numpy arrays.
            _data_rows = max(
                (1 if isinstance(arr, np.ndarray) and arr.ndim == 0 else len(arr))
                for arr in self.data.values()
            )
            _string_rows = max(
                (1 if isinstance(arr, np.ndarray) and arr.ndim == 0 else len(arr))
                for arr in self.string_data.values()
            )
            if _data_rows != _string_rows:
                raise LASDataError(
                    f"DataSection '{self.name}': data row count "
                    f"({_data_rows}) does not match string_data "
                    f"row count ({_string_rows})"
                )

        # Cross-validate data_format against numeric/string data placement
        # (mirrors _check_df_vs_placement in from_dict path, IF-026).
        # Direct construction with S-format curve in data or F-format in
        # string_data passes silently without this check.
        # I2F-03: Validate ALL section types, not just LOG_DATA.
        # from_dict path validates all sections unconditionally.
        # Direct construction with section_type="CORE_DATA" + S-format
        # curve in numeric data previously bypassed this check entirely.
        for _sc in self.section_curves:
            _df = _sc.data_format
            _mnem = _sc.mnemonic
            if not _df:
                continue
            # I2F-37: {A} without array_info is string-format, not numeric.
            if _mnem in data_keys and (_df == "S" or (_df == "A" and not _sc.is_array_element)):
                raise LASDataError(
                    f"DataSection '{self.name}': curve '{_mnem}' has "
                    f"data_format='{_df}' but is in data (numeric). "
                    f"String-format curves must be in string_data."
                )
            if _df not in ("S", "A") and _mnem in string_keys:
                raise LASDataError(
                    f"DataSection '{self.name}': curve '{_mnem}' has "
                    f"data_format='{_df}' but is in string_data. "
                    f"Numeric-format curves must be in data."
                )

        # Run warning-producing checks via validate() (gated by _from_dict).
        if not self._from_dict:
            for issue in self.validate(complete=False):
                warnings.warn(issue, stacklevel=2)

        # F-008: Wrap data/string_data with guarded dicts to catch
        # invalid mutations post-construction.  __post_init__ already
        # validated the contents; the guards prevent future corruption.
        # M-15: Direct construction must not alias the caller's arrays —
        # _GuardedDict wraps via a shallow dict(*args), so array values
        # would remain SHARED with the caller (np.shares_memory True) and
        # caller-side mutation would corrupt internal data.  Deepcopy the
        # mutable input first (N-I-11 pattern — the same fix
        # LASFile.__post_init__ applies to logs/string_data).  from_dict
        # builds private arrays from already-deepcopied input, so it
        # skips this via _from_dict=True.
        if not self._from_dict:
            self.data = copy.deepcopy(self.data)
            self.string_data = copy.deepcopy(self.string_data)
        self.data = _GuardedDict(self.data, _container_name="DataSection.data")
        self.string_data = _GuardedDict(self.string_data, _container_name="DataSection.string_data")
        # I2-12: Wrap curves_order in a guarded list so post-construction
        # mutations (append/insert/extend/__setitem__) validate element
        # types.  The __post_init__ element-type guard (N-I-12 above) only
        # ran at construction; a plain list accepted
        # ``ds.curves_order.append(42)`` which the writer then emitted as
        # a corrupted per-section column count.  __setattr__ keeps the
        # guard self-healing across wholesale assignment (e.g. the
        # writer's save/restore snapshots).
        if not isinstance(self.curves_order, _GuardedList):
            self.curves_order = _GuardedList(
                self.curves_order,
                _container_name="DataSection.curves_order",
                _expected_type=str,
            )


def _validate_array_continuity(curves_order: list[str], context: str) -> None:
    """Validate LAS 3.0 array-curve contiguity within a curve order list (F-10 / N-I-13).

    "Channels that are members of a 3D array must occur sequentially from
    [1] to [n], with no other channels intermixed." (LAS 3.0 Spec, Page 27)

    Checks that array-element curves (``NMR[1]``, ``NMR[2]``, ...) of the
    same base name are positionally contiguous (no non-array curves
    interleaved) and indexed sequentially starting at [1].  Raises
    ``LASDataError`` when either invariant is violated.

    Args:
        curves_order: The ordered curve name list to validate.
        context: Human-readable description of where the list lives
            (e.g. ``"top-level curves_order"`` or ``"section 'Log1'"``)
            for the error message.

    Raises:
        LASDataError: If array curves are non-contiguous or non-sequential.
    """
    from .exceptions import LASDataError

    _ARRAY_MNEMONIC_RE = re.compile(r"^(?P<base>[\w\-]+)\[(?P<index>\d+)\]$")
    if not curves_order:
        return
    # Group array curves by base name, tracking position and index to
    # validate contiguity and sequential order.
    _base_groups: dict[str, list[tuple[int, int]]] = {}
    for _pos, _name in enumerate(curves_order):
        _m = _ARRAY_MNEMONIC_RE.match(_name)
        if _m is None:
            continue
        _base = _m.group("base")
        _idx = int(_m.group("index"))
        _base_groups.setdefault(_base, []).append((_pos, _idx))
    for _base, _entries in _base_groups.items():
        # Must have at least 2 entries to be an "array."
        if len(_entries) < 2:
            continue
        # Check contiguity: positions must be consecutive (no intermixing).
        _positions = [p for p, _ in _entries]
        if _positions != list(range(_positions[0], _positions[0] + len(_positions))):
            raise LASDataError(
                f"LASFile: array '{_base}' curves are not "
                f"contiguous in {context}. "
                f"Array channels must appear sequentially "
                f"with no other channels intermixed."
            )
        # Check sequential indices [1]→[n], no gaps.
        _indices = [i for _, i in _entries]
        if _indices != list(range(1, len(_indices) + 1)):
            raise LASDataError(
                f"LASFile: array '{_base}' has non-sequential "
                f"indices {_indices} in {context}. "
                f"Expected [1]→[{len(_indices)}]."
            )


@dataclass(eq=False)
class LASFile:
    """Complete LAS file data structure.

    Supports LAS 1.2, 2.0, and 3.0 formats.
    """

    version: VersionSection = field(default_factory=VersionSection)
    well: WellSection = field(default_factory=WellSection)
    curves: list[CurveDefinition] = field(default_factory=list)
    parameters: list[ParameterEntry] = field(default_factory=list)
    other: str = ""
    logs: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    curves_order: list[str] = field(default_factory=list)
    source_file: str = ""
    encoding: str = "utf-8"

    # LAS 3.0 specific fields
    data_sections: list[DataSection] = field(default_factory=list)
    # F-36: dtype=object used at all construction sites; np.str_ annotation
    # was misleading and never matched actual usage.
    string_data: dict[str, NDArray[np.object_]] = field(
        default_factory=dict
    )  # For {S} format curves

    # I2F-13: from_dict re-validation control flag.  When True,
    # __post_init__ suppresses warnings (not errors) — the from_dict
    # path has already validated everything during construction.
    _from_dict: bool = field(default=False, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        """MOD-22/I2-12: guarded assignment for ``logs``/``string_data``
        (dicts) and ``curves_order`` (list).

        Post-construction wholesale reassignment (``lf.logs = {...}`` /
        ``lf.string_data = {...}``) previously replaced the
        ``_GuardedDict`` wrapper with a plain dict — bypassing every
        guard (M-43 length check, MOD-17 ndim, MOD-23 array-like).
        validate() reported 0 issues and the writer silently null-padded
        fabricated -999.25 rows.  Re-wrap plain-dict assignments through
        ``_GuardedDict`` (which validates str keys, the 1-D array-like
        contract, and resulting-state length consistency), mirroring
        ``DevFile.__setattr__`` (M-45/M-49).  Idempotent: re-wrapping an
        already-wrapped ``_GuardedDict`` (the __post_init__ path and the
        writer's guard re-installation) copies the content and
        re-validates.  All other fields pass through untouched.

        I2-12: ``curves_order`` is a plain list field whose __post_init__
        element-type guard is bypassed by post-construction mutation
        (``lf.curves_order.append(42)`` made the writer crash with a raw
        AttributeError).  Re-wrap wholesale list assignments through
        ``_GuardedList`` so element types are validated at every mutation
        entry point — including the writer's save/restore snapshots
        (``_WriterMutationGuard._restore_saved_state``) and the parser/
        reader's order rewrites.
        """
        if name in ("logs", "string_data"):
            if value is None:
                super().__setattr__(name, None)
                return
            if not isinstance(value, dict):
                raise TypeError(f"LASFile: {name} must be a dict, got {type(value).__name__}")
            super().__setattr__(
                name,
                _GuardedDict(value, _container_name=f"LASFile.{name}"),
            )
            return
        if name == "curves_order":
            if value is None:
                super().__setattr__(name, None)
                return
            if not isinstance(value, list):
                raise TypeError(f"LASFile: curves_order must be a list, got {type(value).__name__}")
            if not isinstance(value, _GuardedList):
                value = _GuardedList(
                    value,
                    _container_name="LASFile.curves_order",
                    _expected_type=str,
                )
            super().__setattr__(name, value)
            return
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        """Validate critical invariants after construction (F-M05).

        ``from_dict`` validates 800+ lines of invariants; direct
        ``LASFile`` construction previously bypassed all validation.
        This catches the most egregious invalid states while allowing
        incremental construction (empty collections skip validation).
        """
        # Deferred import to avoid circular dependencies.
        from .exceptions import LASDataError

        # N-I-11: Direct construction must not mutate the caller's dicts or
        # alias its arrays.  from_dict deepcopies its input (line ~2681);
        # the direct path previously stored the caller's dict BY REFERENCE,
        # so validate()'s list→ndarray coercion (``self.logs[_k] =
        # np.asarray(...)``) mutated the caller's storage in place, and
        # caller-side array mutations corrupted internal data.  Deepcopy
        # only on the direct path — from_dict's re-validation runs with
        # ``_from_dict=True`` because its input was already copied.
        if not self._from_dict:
            if self.logs:
                self.logs = copy.deepcopy(self.logs)
            if self.string_data:
                self.string_data = copy.deepcopy(self.string_data)
            # M-15: Direct construction must not alias the caller's
            # DataSection objects (nor the caller's data_sections list).
            # ``LASFile(data_sections=[ds])`` previously stored the
            # caller's object BY REFERENCE — caller mutation of the
            # DataSection (ds.name, ds.data[...]) propagated into the
            # model.  Deepcopy the whole container (N-I-11 pattern).
            # from_dict builds private DataSection instances and skips
            # this via _from_dict=True.
            # M-19: The deepcopy must run even for an EMPTY list.  The
            # previous ``if self.data_sections:`` gate left an empty
            # caller list aliased — a later caller ``append()`` mutated
            # the model's data_sections and bypassed the __post_init__
            # validation (duplicate-name dedup, per-section sweeps,
            # continuity) that ran at construction time.
            self.data_sections = copy.deepcopy(self.data_sections)
            # M-19: curves_order is ALSO stored by reference (adversarial
            # finding: ``lf.curves_order is order`` → True; caller append
            # propagates, bypassing the same __post_init__ validation).
            # Strings are immutable so the deepcopy is cheap — it only
            # detaches the container.
            self.curves_order = copy.deepcopy(self.curves_order)

        # F-13: Validate curves_order element types for direct construction.
        # from_dict validates per-element types at lines 2100-2111;
        # direct LASFile() construction bypasses that.  Non-str elements
        # (int, None, float) pass silently — list(range(5)) produces
        # [0,1,2,3,4] as curve names, crashing downstream string operations.
        for _i, _name in enumerate(self.curves_order):
            if not isinstance(_name, str):
                raise TypeError(
                    f"LASFile: curves_order[{_i}] must be str, "
                    f"got {type(_name).__name__}: {_name!r}"
                )

        # --- curves_order / curves consistency ---
        # Every name in curves_order must have a matching CurveDefinition
        # at the same position.  Extra definitions (curves beyond
        # curves_order length) are tolerated — they may be LAS 3.0
        # per-section definitions also registered at the top level.
        if self.curves_order and self.curves:
            if len(self.curves) < len(self.curves_order):
                raise LASDataError(
                    f"LASFile: curves_order has "
                    f"{len(self.curves_order)} entries but only "
                    f"{len(self.curves)} curve definitions found"
                )
            for _i in range(len(self.curves_order)):
                if self.curves_order[_i] != self.curves[_i].mnemonic:
                    raise LASDataError(
                        f"LASFile: curves_order[{_i}] = "
                        f"{self.curves_order[_i]!r} does not match "
                        f"curves[{_i}].mnemonic = "
                        f"{self.curves[_i].mnemonic!r}"
                    )

        _order_set = set(self.curves_order) if self.curves_order else set()

        # M-19: LAS 2.0 first-curve-must-be-index constraint.
        # Now validated by validate() below (called at end of __post_init__).

        # --- duplicate curve name detection (F-MD4-06) ---
        # DataSection.__post_init__ has this check for per-section orders.
        # from_dict has it at lines 1755-1770 for legacy top-level orders.
        # LASFile.__post_init__ previously lacked it entirely.
        # For LAS 3.0 files with data_sections, skip the top-level
        # duplicate check — the same curve name legitimately appears in
        # each section's curves_order, producing duplicates at top level.
        if self.curves_order and not self.data_sections:
            _seen: set[str] = set()
            for _n in self.curves_order:
                if _n in _seen:
                    raise LASDataError(
                        f"LASFile: duplicate curve name {_n!r} in "
                        f"curves_order.  Curve names must be unique."
                    )
                _seen.add(_n)

        # --- logs validation (skip when empty — allows incremental
        # construction, e.g. LASFile() then attribute assignment) ---
        if self.logs:
            _log_keys = set(self.logs.keys())
            _orphaned_logs = _log_keys - _order_set
            if _orphaned_logs:
                raise LASDataError(
                    f"LASFile: logs contain keys not in "
                    f"curves_order: {sorted(_orphaned_logs)}.  "
                    f"Each log key must correspond to a curve "
                    f"mnemonic."
                )
            # F-I2-MD4-04: Detect missing keys — curves in curves_order
            # that have no corresponding log entry.  Skip when
            # data_sections is non-empty (LAS 3.0 files store curve data
            # per-section, not in the top-level logs dict).  Skip when
            # string_data covers the missing keys (the curve lives in
            # string_data, not logs).
            if not self.data_sections and self.curves_order:
                _missing_logs = _order_set - _log_keys
                _str_keys_for_missing = set(self.string_data.keys()) if self.string_data else set()
                _missing_logs -= _str_keys_for_missing
                if _missing_logs:
                    raise LASDataError(
                        f"LASFile: curves_order has keys not found in "
                        f"logs: {sorted(_missing_logs)}.  Each curve "
                        f"mnemonic in curves_order must have a "
                        f"corresponding log entry (or string_data entry "
                        f"for string-format curves)."
                    )
            if len(self.logs) > 1:
                _log_len = {name: len(arr) for name, arr in self.logs.items()}
                if len(set(_log_len.values())) > 1:
                    raise LASDataError(f"LASFile: logs have inconsistent array lengths: {_log_len}")

        # --- string_data validation (same skip-when-empty rule) ---
        if self.string_data:
            _str_keys = set(self.string_data.keys())
            _orphaned_str = _str_keys - _order_set
            if _orphaned_str:
                raise LASDataError(
                    f"LASFile: string_data contain keys not in "
                    f"curves_order: {sorted(_orphaned_str)}.  "
                    f"Each string_data key must correspond to a "
                    f"curve mnemonic."
                )
            # F-I2-MD4-04: Detect missing keys — curves in curves_order
            # that have no corresponding string_data entry.  Guarded
            # the same way as logs above.
            if not self.data_sections and self.curves_order:
                _missing_str = _order_set - _str_keys
                _log_keys_for_missing = set(self.logs.keys()) if self.logs else set()
                _missing_str -= _log_keys_for_missing
                if _missing_str:
                    raise LASDataError(
                        f"LASFile: curves_order has keys not found in "
                        f"string_data: {sorted(_missing_str)}.  Each "
                        f"curve mnemonic in curves_order must have a "
                        f"corresponding entry."
                    )
            if len(self.string_data) > 1:
                _str_len = {name: len(arr) for name, arr in self.string_data.items()}
                if len(set(_str_len.values())) > 1:
                    raise LASDataError(
                        f"LASFile: string_data have inconsistent array lengths: {_str_len}"
                    )

        # F-I2-MD4-03: Detect key overlap between logs and string_data.
        # A curve name appearing in both would silently have data
        # corrupted — one array overwrites the other's meaning.
        # Per-section DataSection has this guard; top-level lacked it.
        if self.logs and self.string_data:
            _overlap = set(self.logs.keys()) & set(self.string_data.keys())
            if _overlap:
                raise LASDataError(
                    f"LASFile: curves {sorted(_overlap)} appear in "
                    f"both logs and string_data.  Each curve may "
                    f"only be stored in one location."
                )
            # F-M-031: Cross-group row-count validation.
            # The within-group checks above verify that all arrays in
            # 'logs' have the same length, and all arrays in
            # 'string_data' have the same length — but no cross-check
            # verifies that logs rows and string_data rows match.
            # A LASFile with 100-row numeric logs and 50-row string_data
            # passes all existing validation.  The writer's
            # ``_format_data_rows`` uses ``max()`` across all arrays,
            # so one group's shorter arrays get padded — producing
            # semantically incorrect output.
            _log_row_count = len(next(iter(self.logs.values())))
            _str_row_count = len(next(iter(self.string_data.values())))
            if _log_row_count != _str_row_count:
                raise LASDataError(
                    f"LASFile: logs row count ({_log_row_count}) does "
                    f"not match string_data row count "
                    f"({_str_row_count}).  Both must have the same "
                    f"number of rows."
                )

        # F-26: Validate dtypes for logs (numeric) and string_data (object).
        # DataSection.__post_init__ has these checks; LASFile omitted them.
        # F-20: Also validate NaN/Inf in numeric data arrays.
        # These are now handled by validate() below.

        # --- data_sections validation (F-MD4-01) ---
        # from_dict has ~200 lines of data_sections validation; __post_init__
        # previously had zero.  This validates the most critical invariants
        # for direct-construction scenarios while allowing incremental
        # construction (empty data_sections skips validation).
        if self.data_sections:
            # F-41: data_sections requires LAS 3.0 version.
            if not self.version.is_las30:
                raise LASDataError("data_sections requires LAS 3.0 version")
            # F-14: Validate data_sections name uniqueness.
            # Duplicate section names produce ambiguous LAS 3.0 output —
            # the writer and parser both use section names to identify
            # sections.  The from_dict path has this check; direct
            # LASFile() construction previously bypassed it.
            # IT3-THR-02 (M-22 extension): Auto-name unnamed data sections
            # BEFORE the dedup check.  The parser auto-names bare/unnamed
            # sections on read (``Section_0``, ``Section_1``, ... in
            # _las30_data.py:382); from_dict with two unnamed sections is
            # valid LAS 3.0 but previously raised a false-positive
            # "duplicate data section name '<unnamed>'".  Auto-naming here
            # mirrors the parser convention and keeps writer output
            # unambiguous.
            _ds_names: list[str] = []
            for _ds_idx, _ds in enumerate(self.data_sections):
                if not _ds.name or not _ds.name.strip():
                    _ds.name = f"Section_{_ds_idx}"
                _ds_names.append(_ds.name)
            _seen_ds: set[str] = set()
            for _ds_name in _ds_names:
                if _ds_name in _seen_ds:
                    # M-34: from_dict must tolerate duplicate section names.
                    # The parser ACCEPTS them (parser.py:842-844 reports
                    # validate() issues via logger.warning only), so the
                    # library's own parse→to_dict→from_dict roundtrip
                    # previously failed with LASDataError on a file the
                    # parser accepted.  from_dict re-runs __post_init__
                    # (I2F-13) with _from_dict=True — skip the raise on
                    # that path and let the validate(complete=True) call
                    # that from_dict ALWAYS performs right after report the
                    # duplicate as a warning (single warning, no data loss).
                    # Direct construction (_from_dict=False) still raises —
                    # that is the documented F-14 contract.
                    if not self._from_dict:
                        raise LASDataError(
                            f"LASFile: duplicate data section name "
                            f"{_ds_name!r}.  Data section names must "
                            f"be unique."
                        )
                _seen_ds.add(_ds_name)
            _all_section_curve_names: set[str] = set()
            for _ds in self.data_sections:
                # Detect duplicate curve names within each section's
                # curves_order (matching DataSection.__post_init__).
                _ds_curves = _ds.curves_order
                if _ds_curves:
                    _ds_seen: set[str] = set()
                    for _cn in _ds_curves:
                        if _cn in _ds_seen:
                            raise LASDataError(
                                f"LASFile: duplicate curve name {_cn!r} "
                                f"in data section '{_ds.name}' "
                                f"curves_order."
                            )
                        _ds_seen.add(_cn)
                # Validate per-section data/string_data keys against
                # curves_order (orphaned key detection, matching from_dict
                # lines 1472-1491).
                _ds_order_set = set(_ds_curves) if _ds_curves else set()
                if _ds.string_data:
                    _ds_str_keys = set(_ds.string_data.keys())
                    _orphaned_str = _ds_str_keys - _ds_order_set
                    if _orphaned_str:
                        raise LASDataError(
                            f"LASFile: string_data in section "
                            f"'{_ds.name}' contains keys not in "
                            f"curves_order: {sorted(_orphaned_str)}."
                        )
                if _ds.data:
                    _ds_data_keys = set(_ds.data.keys())
                    _orphaned_data = _ds_data_keys - _ds_order_set
                    if _orphaned_data:
                        raise LASDataError(
                            f"LASFile: data in section '{_ds.name}' "
                            f"contains keys not in curves_order: "
                            f"{sorted(_orphaned_data)}."
                        )
                # Validate array length consistency within each section.
                if len(_ds.data) > 1:
                    _ds_len = {name: len(arr) for name, arr in _ds.data.items()}
                    if len(set(_ds_len.values())) > 1:
                        raise LASDataError(
                            f"LASFile: data in section '{_ds.name}' "
                            f"has inconsistent array lengths: {_ds_len}"
                        )
                if len(_ds.string_data) > 1:
                    _sds_len = {name: len(arr) for name, arr in _ds.string_data.items()}
                    if len(set(_sds_len.values())) > 1:
                        raise LASDataError(
                            f"LASFile: string_data in section "
                            f"'{_ds.name}' has inconsistent array "
                            f"lengths: {_sds_len}"
                        )
                # Cross-group row-count consistency.
                if _ds.data and _ds.string_data:
                    _data_rows = max(len(arr) for arr in _ds.data.values())
                    _str_rows = max(len(arr) for arr in _ds.string_data.values())
                    if _data_rows != _str_rows:
                        raise LASDataError(
                            f"LASFile: section '{_ds.name}': "
                            f"data row count ({_data_rows}) does not "
                            f"match string_data row count "
                            f"({_str_rows})"
                        )

            # F-10: Cross-curve array continuity validation per LAS 3.0 spec.
            # "Channels that are members of a 3D array must occur
            # sequentially from [1] to [n], with no other channels
            # intermixed." (LAS 3.0 Spec, Page 27)
            for _ds in self.data_sections:
                _validate_array_continuity(
                    _ds.curves_order or [],
                    f"section '{_ds.name}'",
                )

        # N-I-13: F-10 array-continuity validation for the TOP-LEVEL
        # curves_order when there are no data_sections.  The previous code
        # nested F-10 inside ``if self.data_sections:``, so top-level
        # interleaved arrays (``DEPT,NMR[1],GR,NMR[2]``) passed validation,
        # the writer emitted them, and the library's own parser raised
        # LASParseError on re-read — self-unreadable output (PFA e8378ea
        # added F-10 inside the gate, leaving the top-level case unguarded).
        if self.curves_order and not self.data_sections:
            _validate_array_continuity(self.curves_order, "top-level curves_order")

        # F-12p2: Validate parameters list — check every entry is a
        # ParameterEntry and detect duplicate mnemonics.  Direct
        # LASFile() construction bypasses from_dict's validate_before-
        # construction checks.  Non-ParameterEntry objects pass silently.
        for _p in self.parameters:
            if not isinstance(_p, ParameterEntry):
                raise TypeError(
                    f"LASFile: all parameters must be ParameterEntry "
                    f"instances, got {type(_p).__name__}"
                )
        _pmn_seen: set[str] = set()
        for _p in self.parameters:
            if _p.mnemonic in _pmn_seen:
                warnings.warn(
                    f"LASFile: duplicate parameter mnemonic "
                    f"{_p.mnemonic!r} — to_dict() will keep only "
                    f"the last value.",
                    stacklevel=2,
                )
            _pmn_seen.add(_p.mnemonic)

        # F-059: Cross-validate curve data_format against actual data
        # placement.  Now validated by validate() below.

        # I2F-08: Raise LASDataError for format-vs-placement mismatches
        # in direct construction, matching from_dict and DataSection.
        # validate() only WARNS — direct construction MUST raise to
        # prevent silent data corruption.  from_dict raises via
        # _check_df_vs_placement; DataSection.__post_init__ raises.
        if self.curves and (self.logs or self.string_data):
            _log_keys = set(self.logs.keys()) if self.logs else set()
            _str_keys = set(self.string_data.keys()) if self.string_data else set()
            for _sc in self.curves:
                _df = _sc.data_format
                _mnem = _sc.mnemonic
                if not _df:
                    continue
                if _mnem in _log_keys and (_df == "S" or (_df == "A" and not _sc.is_array_element)):
                    raise LASDataError(
                        f"LASFile: curve '{_mnem}' has "
                        f"data_format='{_df}' (string-format) but is "
                        f"in logs (numeric).  String-format curves "
                        f"must be in string_data."
                    )
                if _df not in ("S", "A") and _mnem in _str_keys:
                    raise LASDataError(
                        f"LASFile: curve '{_mnem}' has "
                        f"data_format='{_df}' (numeric-format) but "
                        f"is in string_data.  Numeric-format curves "
                        f"must be in logs."
                    )

        # Run warning-producing checks via validate() (gated by _from_dict).
        if not self._from_dict:
            for issue in self.validate(complete=False):
                warnings.warn(issue, stacklevel=2)
            # F-036: Validate mandatory well fields for direct construction.
            # from_dict and parser paths check this separately
            # (_validate_from_dict_input / _parse_well); direct LASFile()
            # construction previously had zero mandatory well field
            # validation.  Warn (not raise) — the writer produces valid
            # LAS output with defaults for missing fields, matching the
            # from_dict warning contract.
            if self.well.entries:
                _spec = _LASVersionSpec(self.version.vers)
                _mandatory = set(_spec.mandatory_well_fields)
                _well_keys = {k.upper() for k in self.well.entries}
                _missing = _mandatory - _well_keys
                if _missing:
                    warnings.warn(
                        f"Mandatory well field(s) missing: {', '.join(sorted(_missing))}",
                        stacklevel=2,
                    )

        # F-009: Wrap logs/string_data with guarded dicts to catch
        # invalid mutations post-construction.  __post_init__ already
        # validated the contents; the guards prevent future corruption.
        self.logs = _GuardedDict(self.logs, _container_name="LASFile.logs")
        self.string_data = _GuardedDict(self.string_data, _container_name="LASFile.string_data")
        # F-030: Wrap curves/parameters with guarded lists to catch
        # invalid items appended post-construction.
        self.curves = _GuardedList(
            self.curves,
            _container_name="LASFile.curves",
            _expected_type=CurveDefinition,
        )
        self.parameters = _GuardedList(
            self.parameters,
            _container_name="LASFile.parameters",
            _expected_type=ParameterEntry,
        )
        # I2-12: Wrap curves_order in a guarded list so post-construction
        # mutations validate element types (mirroring curves/parameters
        # above).  A plain list accepted ``lf.curves_order.append(42)``
        # which the writer crashed on with a raw AttributeError ('int'
        # object has no attribute 'replace').  __setattr__ keeps the guard
        # self-healing across wholesale assignment.
        if not isinstance(self.curves_order, _GuardedList):
            self.curves_order = _GuardedList(
                self.curves_order,
                _container_name="LASFile.curves_order",
                _expected_type=str,
            )

    def validate(self, complete: bool = False) -> list[str]:
        """Validate LASFile state.

        .. note::

            This method **mutates** ``self.logs`` and ``self.string_data``
            in-place: any non-``numpy.ndarray`` values are coerced via
            ``np.asarray()`` so that dtype checks are reliable.  This
            conversion is lossless — lists and array-likes are wrapped
            into equivalent numpy arrays without changing data values.

        Args:
            complete: If True, also run deferred cross-field checks
                including children delegation, cross-section consistency,
                mandatory well fields, and semantic well checks.

        Returns:
            List of issue strings (empty = no issues found).
            Basic structural checks (type, keys, lengths) raise during
            construction and are not duplicated here.
        """
        issues: list[str] = []

        # --- index curve check (LAS 2.0) ---
        _INDEX_CURVE_ALIASES = frozenset({"DEPT", "DEPTH", "TIME", "INDEX"})
        if (
            self.curves_order
            and _LASVersionSpec(self.version.vers).is_las20
            and self.curves_order[0].upper() not in _INDEX_CURVE_ALIASES
        ):
            issues.append(
                f"LAS 2.0 spec requires the first curve to be "
                f"DEPT, DEPTH, TIME, or INDEX, but got "
                f"{self.curves_order[0]!r}.  Many real-world files "
                f"use alternative index curve names."
            )

        # --- dtype and NaN/Inf for logs ---
        # EXT-04: {I} curves with a fractional declared NULL are stored as
        # object arrays (exact Python ints for data values, float sentinel
        # for null cells) — exempt them from the numeric-dtype check.
        _curve_fmt_by_mnem = {c.mnemonic: c.data_format for c in self.curves}
        for _k, _arr in self.logs.items():
            if not isinstance(_arr, np.ndarray):
                _arr = self.logs[_k] = np.asarray(_arr)
            if _arr.dtype == object and _curve_fmt_by_mnem.get(_k) == "I":
                continue
            if not np.issubdtype(_arr.dtype, np.number):
                issues.append(
                    f"LASFile: curve '{_k}' in 'logs' has non-numeric "
                    f"dtype ({_arr.dtype}).  'logs' arrays must be "
                    f"numeric."
                )
            if _arr.dtype.kind in ("f", "c") and not np.all(np.isfinite(_arr)):
                issues.append(
                    f"LASFile: curve '{_k}' in 'logs' contains non-finite values (NaN/Inf)."
                )

        # --- dtype for string_data ---
        for _sk, _sarr in self.string_data.items():
            if not isinstance(_sarr, np.ndarray):
                _sarr = self.string_data[_sk] = np.asarray(_sarr)
            if np.issubdtype(_sarr.dtype, np.number):
                issues.append(
                    f"LASFile: curve '{_sk}' in 'string_data' has "
                    f"numeric dtype ({_sarr.dtype}).  'string_data' "
                    f"arrays must be non-numeric."
                )

        # --- data_format vs placement ---
        if self.curves and (self.logs or self.string_data):
            for _sc in self.curves:
                _df = _sc.data_format
                _mnem = _sc.mnemonic
                if not _df:
                    continue
                if _df == "S" or (_df == "A" and not _sc.is_array_element):
                    if _mnem in self.logs:
                        issues.append(
                            f"LASFile: curve '{_mnem}' has "
                            f"data_format='{_df}' (string-format) but "
                            f"is in logs (numeric).  String-format "
                            f"curves should be in string_data."
                        )
                else:
                    if _mnem in self.string_data:
                        issues.append(
                            f"LASFile: curve '{_mnem}' has "
                            f"data_format='{_df}' (numeric-format) "
                            f"but is in string_data.  Numeric-format "
                            f"curves should be in logs."
                        )

        # --- deferred checks (complete=True) ---
        if complete:
            # Delegate to children.
            issues.extend(self.version.validate(complete=True))
            issues.extend(self.well.validate(complete=True))
            for _cd in self.curves:
                issues.extend(_cd.validate(complete=True))
            for _pe in self.parameters:
                issues.extend(_pe.validate(complete=True))
            for _ds in self.data_sections:
                issues.extend(_ds.validate(complete=True))

            # I2-13 (models side): post-construction curves_order mutation
            # (reverse/insert/reorder) desyncs the top-level order from
            # ``curves``, silently swapping columns on write.  __post_init__
            # validates the positional alignment at construction;
            # validate(complete=True) — which the writer calls before
            # emitting — must re-check it so a reordered curves_order
            # produces a clear validation issue instead of a silent swap.
            # Extra curve definitions (curves beyond curves_order length,
            # LAS 3.0 per-section) are tolerated — only an order entry
            # without a definition and positional mismatches are flagged.
            if self.curves_order and self.curves:
                if len(self.curves_order) > len(self.curves):
                    issues.append(
                        f"LASFile: curves_order has {len(self.curves_order)} "
                        f"entries but only {len(self.curves)} curve "
                        f"definitions.  Post-construction curves_order "
                        f"mutation may have added an order entry without a "
                        f"definition."
                    )
                else:
                    for _i, (_order_name, _curve) in enumerate(
                        zip(self.curves_order, self.curves, strict=False)
                    ):
                        if _order_name != _curve.mnemonic:
                            issues.append(
                                f"LASFile: curves_order[{_i}] = "
                                f"{_order_name!r} does not match "
                                f"curves[{_i}].mnemonic = "
                                f"{_curve.mnemonic!r}.  Post-construction "
                                f"curves_order mutation has desynced the "
                                f"column order from the curve definitions."
                            )
                            break
            # I2-13: detect data keys curves_order no longer covers after a
            # post-construction mutation (deletion/reorder).  __post_init__
            # validates this at construction; validate() must re-check on
            # post-construction state.
            if self.curves_order and (self.logs or self.string_data):
                _order_set = set(self.curves_order)
                _all_keys = set(self.logs.keys()) | set(self.string_data.keys())
                _orphaned = _all_keys - _order_set
                if _orphaned:
                    issues.append(
                        f"LASFile: curve(s) {sorted(_orphaned)} have data "
                        f"in 'logs' or 'string_data' but are NOT in "
                        f"curves_order.  Post-construction curves_order "
                        f"mutation may have removed an order entry; the "
                        f"writer will not emit these columns."
                    )

            # data_sections requires LAS 3.0.
            if self.data_sections and not self.version.is_las30:
                issues.append("data_sections requires LAS 3.0 version")

            # M-29: Warn when string_data is written to a non-LAS-3.0 file.
            # The LAS 1.2/2.0 writers cannot emit {S} string markers, so
            # string curves are written as unmarked columns; on re-read the
            # data reader routes them as numeric and the string values are
            # replaced with the null sentinel.  The values will not survive
            # a write→read roundtrip (test_writer.py documents this as
            # intentional for LAS 2.0 — warn so callers are not surprised).
            if self.string_data and not self.version.is_las30:
                issues.append(
                    f"string_data is present but version is "
                    f"{self.version.vers!r} (not LAS 3.0).  String curves "
                    f"cannot be represented in LAS 1.2/2.0 output — their "
                    f"values will not survive a write→read roundtrip."
                )

            # F-012: Cross-section data_sections name dedup.
            # __post_init__ (L1730-1741) checks for duplicate data section
            # names, but validate(complete=True) did not — post-construction
            # mutation followed by validate() would pass with duplicates.
            # Mirror the __post_init__ logic here as a warning-producing check.
            # IT3-THR-02 (M-22 extension): __post_init__ auto-names empty
            # section names (Section_N) so two unnamed sections are not
            # duplicates.  validate() must not mutate, so it simply skips
            # empty names — matching the auto-naming outcome.
            if len(self.data_sections) > 1:
                _ds_names: list[str] = []
                for _ds in self.data_sections:
                    if _ds.name and _ds.name.strip():
                        _ds_names.append(_ds.name)
                _seen_ds: set[str] = set()
                for _ds_name in _ds_names:
                    if _ds_name in _seen_ds:
                        issues.append(
                            f"LASFile: duplicate data section name "
                            f"{_ds_name!r}.  Data section names must "
                            f"be unique."
                        )
                    _seen_ds.add(_ds_name)

            # F-067: Mandatory well field presence check.
            # __post_init__ warns on direct construction; from_dict warns
            # via _validate_from_dict_input; the parser checks during
            # _parse_well.  validate(complete=True) previously delegated
            # to WellSection.validate which does NOT check mandatory well
            # field presence (only STEP=0, NULL empty, STRT==STOP).
            if self.well.entries:
                _spec = _LASVersionSpec(self.version.vers)
                _mandatory = set(_spec.mandatory_well_fields)
                _well_keys = {k.upper() for k in self.well.entries}
                _missing = _mandatory - _well_keys
                if _missing:
                    issues.append(
                        f"LASFile: mandatory well field(s) missing: {', '.join(sorted(_missing))}"
                    )

        return issues

    def to_dict(self) -> dict[str, Any]:
        """Convert to legacy dict format for backward compatibility.

        Returns a dict with both legacy ``parameters`` (``{mnemonic: value}``
        dict) and ``parameter_details`` (list of full ParameterEntry dicts)
        to preserve backward compatibility while exposing LAS 3.0 metadata.
        ``logs`` arrays are defensively copied.

        Note: The defensive array copies (``v.copy()`` on every log/string
        array) allocate new arrays.  In low-memory environments this can
        raise ``MemoryError``.  Consider using the ``LASFile`` dataclass
        directly to avoid the copy overhead.
        """
        params_dict: dict[str, str] = {}
        for p in self.parameters:
            # F-19: Warn when a parameter mnemonic was already seen —
            # dict assignment is last-wins, so earlier values are
            # silently discarded.  to_dict() previously had no duplicate
            # detection; from_dict and from_dict validation now both
            # warn about duplicates, tracking the consistent approach.
            if p.mnemonic in params_dict:
                warnings.warn(
                    f"LASFile.to_dict(): duplicate parameter mnemonic "
                    f"{p.mnemonic!r} — dict result will contain only "
                    f"the last value.  Consider deduplicating "
                    f"parameter_entry list before serialization.",
                    stacklevel=2,
                )
            params_dict[p.mnemonic] = p.value

        return {
            "version": self.version.to_dict(),
            "well": dict(self.well.entries),
            "well_units": dict(self.well.units) if self.well.units else {},
            "well_descriptions": dict(self.well.descriptions) if self.well.descriptions else {},
            "parameters": params_dict,
            "parameter_details": [p.to_dict() for p in self.parameters],
            "curves": [c.to_dict() for c in self.curves],
            # Defensive copy prevents callers from mutating internal arrays
            # through the returned dict (numpy arrays are mutable views).
            "logs": {k: v.copy() for k, v in self.logs.items()},
            "curves_order": list(self.curves_order),
            "other": self.other,
            "data_sections": [ds.to_dict() for ds in self.data_sections],
            "string_data": {
                k: v.copy() for k, v in self.string_data.items()
            },  # same defensive copy as logs above
            "encoding": self.encoding,
            "source_file": self.source_file,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        mnem_base: dict[str, str] | None = None,
    ) -> LASFile:
        """Create LASFile from dict format.

        Handles multiple format variants inherently:
        - Legacy flat dict (curves as string lists, params as {name: value} dict)
        - Detailed dict with CurveDefinition metadata (unit, api_code, description)
        - LAS 3.0 dict with array_info, data_format, data_sections, string_data
        - Mixed formats from roundtrip serialization

        Args:
            data: Dict representation of a LAS file (as produced by
                ``LASFile.to_dict()``).
            mnem_base: Optional mnemonic-to-canonical-name mapping for
                curve name normalization (same as parser's *mnem_base*).
                When provided, mnemonic names extracted from *data* are
                resolved through this mapping using case-insensitive
                lookup (matching the parser's ``_mnem_base_upper``
                pattern).  Default ``None`` means no normalization —
                mnemonics are used as-is (backward-compatible).

        The method is naturally long due to covering all these variants in a
        single backwards-compatible code path.
        """
        # F-08: Pre-try errors escape PylasdevError wrapping.  The
        # isinstance check and _validate_from_dict_input were previously
        # outside the try block — TypeErrors from those calls escaped
        # without being wrapped in LASDataError, violating the method's
        # documented contract.  Move them inside so ALL validation errors
        # get the LASDataError wrapper.
        # F-06: Deferred imports to avoid circular dependencies
        # (models.py ← parser.py/data_reader.py which import from models.py).
        from .data_reader import MAX_CURVES, MAX_DATA_LINES, MAX_TOTAL_ELEMENTS
        from .exceptions import LASDataError
        from .parser import (
            MAX_DATA_SECTIONS,
            MAX_OTHER_LINES,
            MAX_PARAMETERS,
            _validate_data_section_column_counts,
        )

        # F-057: Bound well-related iterables to match the parser's
        # MAX_DEFERRED_WELL_ENTRIES guard on the well re-processing path.
        MAX_WELL_ENTRIES = MAX_PARAMETERS

        try:
            # F-008: isinstance and validation inside try block so
            # TypeError/ValueError get wrapped in LASDataError.
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict, got {type(data).__name__}")
            # F-40: Protect caller's dict from mutation.
            data = copy.deepcopy(data)

            # F-017/F-018/IF-015/IF-026/F-019: Pre-construction validation layer.
            # Closes Pattern #7 (from_dict validation gaps) structurally.
            _validate_from_dict_input(data)

            # F-MD4-03: Build mnemonic normalization lookup when mnem_base
            # is provided.  Matches the parser's _mnem_base_upper pattern
            # (parser.py:440-450) for case-insensitive lookup + chain
            # resolution.  When mnem_base is None, _norm_mnem is an
            # identity function — backward-compatible.
            from .mnem_base import build_mnemonic_lookup

            _mnem_base_upper: dict[str, str] | None = None

            def _norm_mnem(raw: str, /) -> str:
                """Normalize a mnemonic through mnem_base lookup.

                Returns the resolved canonical name when *mnem_base* was
                provided and the uppercased *raw* string matches an entry;
                returns *raw* unchanged otherwise (identity).
                """
                if _mnem_base_upper is None:
                    return raw
                return _mnem_base_upper.get(raw.upper(), raw)

            # F-019: Use shared build_mnemonic_lookup() from mnem_base.py
            # instead of the 25-LOC inlined algorithm that was duplicated
            # between models.py and mnem_base.py.  The shared function
            # provides identical deterministic first-wins semantics with
            # chain resolution via resolve_mnemonic().
            _mnem_base_upper = build_mnemonic_lookup(mnem_base) if mnem_base else None

            # N-I-30: Resolution-collision-aware curve-name normalization.
            # MNEM_BASE maps distinct vendor mnemonics to the same canonical
            # (e.g. "LLD"→"BK"→"BFV" AND "LLS"→"BK"→"BFV" — a standard
            # dual-laterolog file carries both).  When two DIFFERENT raw
            # mnemonics resolve to the same canonical, the old code produced
            # a duplicate curves_order and __post_init__ raised a misleading
            # LASDataError on duplicate-free input; the colliding curve's
            # identity was lost.  Detect the collision during normalization:
            # keep the ORIGINAL mnemonic for the colliding curve (preserving
            # identity) and warn accurately.  Genuine duplicates (the SAME
            # raw name twice) still resolve identically and are caught by the
            # existing duplicate checks.
            _resolved_curve_names: dict[str, str] = {}
            _warned_collisions: set[tuple[str, str]] = set()

            def _norm_curve_mnem(
                raw: str,
                dest: list[str] | None = None,
                dest_dict: dict[str, Any] | None = None,
            ) -> str:
                """Normalize a curve mnemonic, preserving identity on
                resolution collisions.

                *dest* is the in-progress ``curves_order`` list when
                normalizing curve ORDER (so the canonical-priority branch
                can re-key an earlier alias entry); *dest_dict* is the
                in-progress data/string_data dict when normalizing its
                keys (per-section keys are normalized BEFORE the section's
                curves_order, so the alias's stored value is re-keyed to
                its original mnemonic the same way the well path re-keys
                its entries).  ``None`` at the other normalization sites
                (curves, section_curves, logs keys after the top-level
                order is settled) where the state is already final.
                """
                resolved = _norm_mnem(raw)
                _raw_key = raw.upper()
                _prev_raw = _resolved_curve_names.get(resolved)
                if _prev_raw is not None and _prev_raw != _raw_key:
                    if raw == resolved:
                        # PXM-06: The canonical name itself collides with an
                        # alias that claimed the resolved slot earlier
                        # (e.g. ['LLD','BFV'] with LLD→BFV).  The canonical
                        # wins its own slot; the earlier alias entry is
                        # re-keyed to its ORIGINAL mnemonic in *dest* (and
                        # in the resolution state) so both distinct curves
                        # keep distinct identities.  Mirror of the well path's
                        # raw==resolved re-keying (models.py _norm_well_mnem).
                        if dest is not None:
                            for _i, _n in enumerate(dest):
                                if _n == resolved:
                                    dest[_i] = _prev_raw
                                    break
                        elif dest_dict is not None:
                            if resolved in dest_dict:
                                dest_dict[_prev_raw] = dest_dict[resolved]
                                del dest_dict[resolved]
                        _resolved_curve_names[_prev_raw] = _prev_raw
                        _resolved_curve_names[_raw_key] = _raw_key
                        if (_raw_key, resolved) not in _warned_collisions:
                            _warned_collisions.add((_raw_key, resolved))
                            warnings.warn(
                                f"from_dict: canonical curve mnemonic "
                                f"'{resolved}' collides with alias "
                                f"'{_prev_raw}' (which resolves to "
                                f"'{resolved}') earlier in the curve list.  "
                                f"Preserving both: '{_prev_raw}' keeps its "
                                f"original mnemonic and '{resolved}' keeps "
                                f"the canonical name.",
                                UserWarning,
                                stacklevel=2,
                            )
                        return resolved
                    # Warn once per (raw, resolved) pair — the same curve
                    # name is normalized at several sites (curves_order,
                    # curves, logs keys, string_data keys).
                    if (_raw_key, resolved) not in _warned_collisions:
                        _warned_collisions.add((_raw_key, resolved))
                        warnings.warn(
                            f"from_dict: mnemonic '{raw}' resolves to "
                            f"'{resolved}' via mnem_base, but '{resolved}' is "
                            f"already used by curve '{_prev_raw}'.  Keeping "
                            f"original mnemonic '{raw}' to preserve curve "
                            f"identity.",
                            UserWarning,
                            stacklevel=2,
                        )
                    _resolved_curve_names[_raw_key] = _raw_key
                    return raw
                _resolved_curve_names[resolved] = _raw_key
                return resolved

            # M-44: Resolution-collision-aware WELL-name normalization
            # (twin of _norm_curve_mnem above for the well section).
            # The well/well_units/well_descriptions loops previously used
            # the bare _norm_mnem, so two distinct raw mnemonics resolving
            # to the same canonical (e.g. "LLD"→"BFV" AND "LLS"→"BFV" in
            # the shipped MNEM_BASE) silently last-won — one entry was
            # dropped with ZERO warnings, while the parser path warns
            # (parser.py:1908-1915) and the curve path uses
            # _norm_curve_mnem.  Keep the ORIGINAL mnemonic for the
            # colliding entry (preserving data) and warn, matching N-I-30.
            # Separate resolved-name state: well entries and curves are
            # distinct namespaces that legitimately share canonical names
            # (e.g. a "DEPT" well entry and a "DEPT" curve).
            _resolved_well_names: dict[str, str] = {}
            _warned_well_collisions: set[tuple[str, str]] = set()

            def _norm_well_mnem(raw: str, entries: dict[str, str], /) -> str:
                """Normalize a well mnemonic, preserving ALL values on
                resolution collisions (F-10: TRUE data preservation).

                *entries* is the destination dict (``las_file.well.entries``,
                ``well.units``, or ``well.descriptions``).  When the
                canonical key itself collides with a previously-stored alias
                (e.g. ``{"LLD": "100", "BFV": "200"}`` with ``LLD→BFV``),
                the M-44 collision branch previously returned ``raw ==
                resolved`` and the caller's assignment last-won — the alias
                value was silently dropped while the warning claimed
                preservation.  Here the alias's stored value is first
                re-keyed to its ORIGINAL mnemonic, so BOTH values survive
                and the warning describes what actually happens.
                """
                resolved = _norm_mnem(raw)
                _raw_key = raw.upper()
                _prev_raw = _resolved_well_names.get(resolved)
                if _prev_raw is not None and _prev_raw != _raw_key:
                    if raw == resolved:
                        # The canonical key collides with a stored alias:
                        # move the alias's value back under its original
                        # mnemonic so the caller's store of the canonical
                        # value below cannot overwrite it.  Guarded so a
                        # container where the alias was never stored (shared
                        # _resolved_well_names state across the well /
                        # well_units / well_descriptions loops) is untouched.
                        if resolved in entries:
                            entries[_prev_raw] = entries[resolved]
                            del entries[resolved]
                    # Warn once per (raw, resolved) pair.
                    if (_raw_key, resolved) not in _warned_well_collisions:
                        _warned_well_collisions.add((_raw_key, resolved))
                        if raw == resolved:
                            warnings.warn(
                                f"from_dict: well mnemonic '{raw}' resolves "
                                f"to '{resolved}' via mnem_base, but "
                                f"'{resolved}' is already used by well entry "
                                f"'{_prev_raw}'.  Preserving both: the value "
                                f"for '{_prev_raw}' is re-keyed to its "
                                f"original mnemonic and '{raw}' is stored "
                                f"under '{resolved}'.",
                                UserWarning,
                                stacklevel=2,
                            )
                        else:
                            warnings.warn(
                                f"from_dict: well mnemonic '{raw}' resolves "
                                f"to '{resolved}' via mnem_base, but "
                                f"'{resolved}' is already used by well entry "
                                f"'{_prev_raw}'.  Keeping original mnemonic "
                                f"'{raw}' to preserve well entry identity.",
                                UserWarning,
                                stacklevel=2,
                            )
                    _resolved_well_names[_raw_key] = _raw_key
                    return resolved if raw == resolved else raw
                _resolved_well_names[resolved] = _raw_key
                return resolved

            las_file = cls()

            version = _resolve_dict_entry(data, "version", dict, dict)
            # PXM-03: Parser-equivalent VERS normalization.  The parser
            # (parser.py:1853-1933) strips three-segment versions
            # ("1.2.0"→"1.2"), accepts known versions silently, accepts
            # version-like digit.digit values and 3.x drafts with a warning,
            # preserves non-numeric values, and defaults unknown values
            # like "1,2" to "2.0" with a warning.  from_dict previously
            # kept the raw value verbatim, so a VERS="1,2" model built but
            # write_las_file raised LASWriteError ("Unsupported LAS
            # version: '1,2'") while the same input parsed from a file
            # wrote fine.  Mirror the parser so both construction paths
            # produce writable models.
            _vers_raw = _safe_str(version.get("VERS"), "2.0")
            _vers_norm = _vers_raw.strip()
            _vers_norm = re.sub(r"^(\d+\.\d+)\.\d+$", r"\1", _vers_norm)
            if _vers_norm not in {"1.2", "2.0", "3.0"}:
                if _vers_norm.startswith("3."):
                    warnings.warn(
                        f"Non-standard VERS value {_vers_raw!r}. "
                        f"Expected '3.0'. Accepting as LAS 3.x for "
                        f"compatibility with draft/development versions.",
                        UserWarning,
                        stacklevel=2,
                    )
                elif re.match(r"^\d+\.\d+$", _vers_norm):
                    warnings.warn(
                        f"Non-standard VERS value {_vers_raw!r}. "
                        f"Expected 1.2, 2.0, or 3.0. "
                        f"Preserving as-is for backward compatibility.",
                        UserWarning,
                        stacklevel=2,
                    )
                elif _vers_norm and not _vers_norm[0].isdigit():
                    pass  # non-numeric — preserve as-is (parser-compatible)
                else:
                    warnings.warn(
                        f"Unknown VERS value {_vers_raw!r}. Expected 1.2, "
                        f"2.0, or 3.0. Defaulting to 2.0.",
                        UserWarning,
                        stacklevel=2,
                    )
                    _vers_norm = "2.0"
            las_file.version = VersionSection(
                vers=_vers_norm,
                wrap=_safe_str(version.get("WRAP"), "NO"),
                dlm=_safe_str(version.get("DLM"), "SPACE"),
            )

            well = _resolve_dict_entry(data, "well", dict, dict)
            # F-057: Bound check — parser has MAX_DEFERRED_WELL_ENTRIES on its
            # well re-processing path; from_dict previously had no bound at all.
            if len(well) >= MAX_WELL_ENTRIES:
                raise ValueError(
                    f"Number of well entries ({len(well)}) exceeds maximum "
                    f"allowed ({MAX_WELL_ENTRIES})"
                )
            for key, value in well.items():
                if not isinstance(key, str):
                    raise TypeError(f"Well dict key must be str, got {type(key).__name__}: {key!r}")
                # F-058: Normalize well entry keys through mnem_base,
                # matching the parser's behaviour (parser.py:1932-1938).
                # M-44: collision-aware — two raw mnemonics resolving to
                # the same canonical keep the original and warn instead of
                # silently dropping one entry.  F-10: the destination dict
                # is passed so an alias+canonical pair in the same input
                # preserves BOTH values (the alias is re-keyed to its
                # original mnemonic).
                _norm_key = _norm_well_mnem(key, las_file.well.entries)
                las_file.well[_norm_key] = _safe_str(value)
            # Restore well units if present (from v1.7+ roundtrip data)
            well_units = _resolve_dict_entry(data, "well_units", dict, dict)
            if len(well_units) >= MAX_WELL_ENTRIES:
                raise ValueError(
                    f"Number of well unit entries ({len(well_units)}) exceeds "
                    f"maximum allowed ({MAX_WELL_ENTRIES})"
                )
            for key, unit in well_units.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Well unit dict key must be str, got {type(key).__name__}: {key!r}"
                    )
                # M-44: collision-aware well_units normalization (same
                # N-I-30 pattern as the well entries loop above).  F-10:
                # pass the destination dict so alias+canonical pairs in
                # the same input preserve BOTH unit values.
                _norm_key = _norm_well_mnem(key, las_file.well.units)
                las_file.well.units[_norm_key] = _safe_str(unit)

            # Restore well descriptions if present (from v1.8+ roundtrip data)
            well_descriptions = _resolve_dict_entry(data, "well_descriptions", dict, dict)
            if len(well_descriptions) >= MAX_WELL_ENTRIES:
                raise ValueError(
                    f"Number of well description entries ({len(well_descriptions)}) "
                    f"exceeds maximum allowed ({MAX_WELL_ENTRIES})"
                )
            for key, desc in well_descriptions.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Well description dict key must be str, got {type(key).__name__}: {key!r}"
                    )
                # M-44: collision-aware well_descriptions normalization
                # (same N-I-30 pattern as the well entries loop above).
                # F-10: pass the destination dict so alias+canonical pairs
                # in the same input preserve BOTH description values.
                _norm_key = _norm_well_mnem(key, las_file.well.descriptions)
                las_file.well.descriptions[_norm_key] = _safe_str(desc)

            curves_order = data.get("curves_order", [])
            # F-21: Guard against non-list iterables.  list(string) silently
            # creates a list of single characters (e.g. "DEPT,DT,GR" → 10
            # single-char mnemonics), passing all downstream cross-validation
            # because both curves_order and curves get corrupted the same way.
            if curves_order is None:
                curves_order = []
            elif isinstance(curves_order, (str, bytes)):
                # F-110: ``isinstance(bytes, str)`` is False in Python 3,
                # so bytes bypassed the str guard.  ``list(b"GR")`` →
                # ``[71, 82]`` (integers), which propagate as curve names.
                if isinstance(curves_order, bytes):
                    raise TypeError(
                        "curves_order must be a list of strings, "
                        "got bytes.  Decode to str first: "
                        "curves_order.decode('utf-8')"
                    )
                raise ValueError(f"curves_order must be a list, got str: {curves_order!r}")
            elif isinstance(curves_order, dict):
                raise TypeError(
                    "curves_order must be a list of strings, "
                    "got dict.  Use list(curves_order) to extract "
                    "keys, or provide a list directly."
                )
            elif not isinstance(curves_order, Iterable):
                # F-M02: Non-iterable non-str types (int, float, bool) crash
                # at list() with TypeError.  Provide a clear error instead.
                raise TypeError(
                    f"curves_order must be an iterable, got {type(curves_order).__name__}"
                )
            # F-029: Materialize iterable to prevent generator exhaustion
            # when iterated twice (validation loop + list comprehension).
            curves_order = list(curves_order)
            # F-M-014 / F-M-016: Validate per-element types in curves_order.
            # Non-string elements (int, None) crash _norm_mnem().upper()
            # when mnem_base is active, and silently produce integer curve
            # names when mnem_base is None (e.g. curves_order=range(5)
            # passes the iterable guard but produces [0,1,2,3,4]).
            for _i, _name in enumerate(curves_order):
                if isinstance(_name, bytes):
                    raise TypeError(
                        f"curves_order[{_i}] must be str, "
                        f"got bytes.  Decode to str first: "
                        f"curves_order[{_i}].decode('utf-8')"
                    )
                if not isinstance(_name, str):
                    raise TypeError(
                        f"curves_order[{_i}] must be str, got {type(_name).__name__}: {_name!r}"
                    )
            # PXM-06: Normalize through an explicit loop (not a list
            # comprehension) so _norm_curve_mnem can re-key an earlier
            # alias entry when the canonical name itself arrives later
            # (alias-before-canonical collision).
            las_file.curves_order = []
            for _name in curves_order:
                las_file.curves_order.append(_norm_curve_mnem(_name, las_file.curves_order))

            # Restore curve metadata if available (new format), otherwise create minimal CurveDefinition
            # F-16: Use _resolve_dict_entry — data.get("curves", []) returns
            # None when "curves" exists with value None, bypassing the default
            # and crashing at len(None).  Same pattern at 6 other sites below.
            curves_data = _resolve_dict_entry(data, "curves", list, list)
            # F-06: Resource-exhaustion guard — match parser's MAX_CURVES check.
            if len(curves_data) >= MAX_CURVES:
                raise ValueError(
                    f"Number of curves ({len(curves_data)}) exceeds maximum allowed ({MAX_CURVES})"
                )
            if isinstance(curves_data, list):
                if curves_data:
                    # F-012: Validate every element is a dict, not just curves_data[0].
                    # Non-dict elements crash at curve_dict.get() downstream.
                    _validate_iterable_of_dicts(curves_data, "curves")
                    for curve_dict in curves_data:
                        array_info = None
                        if "array_info" in curve_dict and isinstance(
                            curve_dict["array_info"], dict
                        ):
                            ai = curve_dict["array_info"]
                            _base_name = _safe_str(ai.get("base_name")).upper()
                            if _base_name:
                                array_info = ArrayElementInfo(
                                    base_name=_base_name,
                                    index=_resolve_dict_entry(ai, "index", int, lambda: 0),
                                    # F2-002: Validate time_offset — int(offset) in
                                    # writer.py crashes on non-numeric values.
                                    time_offset=_resolve_dict_entry(
                                        ai, "time_offset", (int, float), lambda: None
                                    ),
                                )
                        _raw_mnem = _safe_str(curve_dict.get("mnemonic", ""))
                        las_file.curves.append(
                            CurveDefinition(
                                mnemonic=_norm_curve_mnem(_raw_mnem),
                                unit=_safe_str(curve_dict.get("unit", "")),
                                api_code=_safe_str(curve_dict.get("api_code", "")),
                                description=_safe_str(curve_dict.get("description", "")),
                                original_mnemonic=_safe_str(
                                    curve_dict.get("original_mnemonic", "")
                                ),
                                data_format=_safe_str(curve_dict.get("data_format", "")),
                                array_info=array_info,
                            )
                        )
                        # F-17: Cross-check bracket-notation mnemonic against
                        # array_info.base_name.  Only validate when the mnemonic
                        # uses array bracket notation (e.g., "NMR[1]") —
                        # non-bracket-notation curves like "CORET" with
                        # base_name="CORE" are intentionally skipped.
                        if array_info and "[" in _raw_mnem:
                            if _raw_mnem.split("[")[0] != array_info.base_name:
                                warnings.warn(
                                    f"from_dict warning: mnemonic "
                                    f"{_raw_mnem!r} uses array notation but "
                                    f"array_info.base_name is "
                                    f"{array_info.base_name!r}. "
                                    f"Cross-check mismatch may indicate "
                                    f"malformed input.",
                                    stacklevel=3,
                                )
            elif curves_data:
                # F-I2-M01: This branch is provably unreachable with the current
                # _resolve_dict_entry(data, "curves", list, list) call above —
                # it always returns a list or raises TypeError.  Retained as
                # defensive design in case _resolve_dict_entry is ever relaxed.
                raise TypeError(f"curves must be a list, got {type(curves_data).__name__}")

            # Legacy format: only curve names available
            # (reached when curves_data is empty list or falsy)
            if not las_file.curves:
                # F-06: Resource-exhaustion guard for legacy curves_order path.
                if len(curves_order) >= MAX_CURVES:
                    raise ValueError(
                        f"Number of curves ({len(curves_order)}) exceeds maximum "
                        f"allowed ({MAX_CURVES})"
                    )
                for curve_name in curves_order:
                    las_file.curves.append(CurveDefinition(mnemonic=_norm_curve_mnem(curve_name)))

            # F-02: Cross-validate curves_order and curves for consistency.
            # Both are built from separate dict keys independently; a mismatched
            # input dict can produce silently inconsistent state.  from_dict is
            # called on untrusted data via write_las_file (public API).
            _curve_count = len(las_file.curves)
            # MOD-12: tolerate EXTRA curve definitions (curves beyond
            # curves_order length) — matching LASFile.__post_init__
            # (models.py L2762-2768, "Extra definitions (curves beyond
            # curves_order length) are tolerated — they may be LAS 3.0
            # per-section definitions also registered at the top level").
            # The previous strict equality check made the library's OWN
            # to_dict→from_dict roundtrip of that documented-valid state
            # fail with LASDataError.  Only curves_order LONGER than
            # curves remains invalid (an order entry with no definition).
            if len(las_file.curves_order) > _curve_count:
                raise ValueError(
                    f"curves_order length ({len(las_file.curves_order)}) "
                    f"is greater than curves length ({_curve_count}).  "
                    f"Every curves_order entry must have a matching "
                    f"curve definition."
                )
            for _i, (_order_name, _curve) in enumerate(
                zip(las_file.curves_order, las_file.curves, strict=False)
            ):
                if _order_name != _curve.mnemonic:
                    raise ValueError(
                        f"curves_order[{_i}] = {_order_name!r} does not match "
                        f"curves[{_i}].mnemonic = {_curve.mnemonic!r}"
                    )

            params = _resolve_dict_entry(data, "parameters", (dict, list), list)
            # F-06: Resource-exhaustion guard for parameters.
            _param_count = len(params)
            if _param_count >= MAX_PARAMETERS:
                raise ValueError(
                    f"Number of parameters ({_param_count}) exceeds maximum "
                    f"allowed ({MAX_PARAMETERS})"
                )
            # I2-01: parameter_details is the first-class, metadata-rich key
            # (documented in reader.py).  It was previously read ONLY inside
            # the dict-form parameters branch, so a details-only dict
            # (parameters absent) or list-form parameters silently DROPPED
            # the key with no warning.  Read it unconditionally: when present
            # (even as an explicit empty list — F-203), it takes priority
            # over BOTH dict-form and list-form parameters; when absent,
            # fall back to the legacy parameters dict/list.
            param_details = data.get("parameter_details")
            if param_details is not None:
                if len(param_details) >= MAX_PARAMETERS:
                    raise ValueError(
                        f"Number of parameter details ({len(param_details)}) exceeds maximum "
                        f"allowed ({MAX_PARAMETERS})"
                    )
                # F2-25 + F-058 consistency: Validate as list-of-dicts via
                # shared helper (same pattern as curves_data, section_curves,
                # and data_sections).  Raises TypeError for non-dict elements.
                param_details = _validate_iterable_of_dicts(param_details, "parameter_details")
                for param_dict in param_details:
                    # F-MD4-03: Normalize mnemonic through mnem_base
                    # before construction (matching parser.py:2023).
                    param_dict["mnemonic"] = _norm_mnem(_safe_str(param_dict.get("mnemonic")))
                    las_file.parameters.append(_create_parameter_entry(param_dict))
            elif isinstance(params, dict):
                # Legacy format: {mnemonic: value} (no parameter_details).
                for mnemonic, value in params.items():
                    las_file.parameters.append(
                        ParameterEntry(
                            mnemonic=_norm_mnem(mnemonic),
                            value=_safe_str(value),
                        )
                    )
            elif isinstance(params, list):
                # New format: [{"mnemonic": ..., "value": ..., ...}, ...]
                # F2-25 consistency: Shared helper validates every element is a dict
                # (previously checked inline, same as param_details and data_sections).
                _validate_iterable_of_dicts(params, "parameters")
                for param_dict in params:
                    # F-MD4-03: Normalize mnemonic through mnem_base
                    # before construction (matching parser.py:2023).
                    param_dict["mnemonic"] = _norm_mnem(_safe_str(param_dict.get("mnemonic")))
                    las_file.parameters.append(_create_parameter_entry(param_dict))
            else:
                # F-I2-M02: This branch is provably unreachable with the current
                # _resolve_dict_entry(data, "parameters", (dict, list), list) call
                # above — it always returns a dict or list, or raises TypeError.
                # Retained as defensive design in case _resolve_dict_entry is ever relaxed.
                raise TypeError(f"parameters must be a dict or list, got {type(params).__name__}")

            # F-06: Resource-exhaustion guard for other section content.
            # F-I2-M33: Count lines not characters (matching parser behavior).
            # The parser counts len(list[str]) — individual lines appended to
            # _other_lines.  from_dict previously counted len(str) — characters
            # in the joined other-section string.  Using splitlines() aligns
            # the guard: a file accepted by parse() is accepted by from_dict().
            _other_str = _safe_str(data.get("other"), "")
            _other_line_count = len(_other_str.splitlines())
            if _other_line_count >= MAX_OTHER_LINES:
                raise ValueError(
                    f"Other section line count ({_other_line_count}) exceeds "
                    f"maximum allowed ({MAX_OTHER_LINES})"
                )
            las_file.other = _safe_str(data.get("other"), "")
            las_file.encoding = _safe_str(data.get("encoding"), "utf-8")
            las_file.source_file = _safe_str(data.get("source_file"), "")

            # Restore LAS 3.0 data sections
            ds_data = _resolve_dict_entry(data, "data_sections", list, list)
            # F-06: Resource-exhaustion guard for data sections.
            if len(ds_data) >= MAX_DATA_SECTIONS:
                raise ValueError(
                    f"Number of data sections ({len(ds_data)}) exceeds maximum "
                    f"allowed ({MAX_DATA_SECTIONS})"
                )
            # F-058: Validate every element is a dict.  Previously non-dict
            # elements were silently skipped with ``continue`` while two sibling
            # paths (parameter_details and params) both raised TypeError.
            # Using the shared helper also adds a missing list-type check.
            ds_data = _validate_iterable_of_dicts(ds_data, "data_sections")
            # I2F-15: Cross-section cumulative allocation counter.
            # Each section independently passes MAX_TOTAL_ELEMENTS checks,
            # but there is no running total across ALL sections.  1,000
            # sections * 10M elements = 10B elements (~80 GB) passes all
            # per-section guards.  Track cumulative elements to prevent
            # multi-section allocation DoS.
            _cumulative_elements = 0
            for ds_dict in ds_data:
                ds_string_data: dict[str, Any] = {}
                _ds_string_raw = _resolve_dict_entry(ds_dict, "string_data", dict, dict)
                # F-24: Per-section string_data entry count guard.  Every other
                # iterable dict in from_dict() has a count guard (curves_data →
                # MAX_CURVES, ds_data → MAX_CURVES, logs → MAX_CURVES, etc.);
                # string_data was the sole unguarded iterable.
                if len(_ds_string_raw) >= MAX_CURVES:
                    ds_name = ds_dict.get("name", "<unknown>")
                    raise ValueError(
                        f"Number of string data curves ({len(_ds_string_raw)}) "
                        f"in section '{ds_name}' exceeds maximum allowed ({MAX_CURVES})"
                    )
                for name, arr in _ds_string_raw.items():
                    # F-22: Guard against None — np.array(None, dtype=np.str_)
                    # creates a 0-d array containing the string "None", which
                    # then passes the downstream len() check as a silent bug.
                    if arr is None:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"String data for curve '{name}' in section '{ds_name}' is None"
                        )
                    # F-MD4-02: Pre-allocation size check (same pattern as
                    # ds_data numeric path above).  np.array() allocates
                    # before the downstream len() guard — a huge list
                    # triggers MemoryError before the guard catches it.
                    if hasattr(arr, "__len__") and len(arr) > MAX_DATA_LINES:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"String data array length ({len(arr)}) for "
                            f"'{name}' in section '{ds_name}' exceeds "
                            f"maximum allowed ({MAX_DATA_LINES})"
                        )
                    # F-I2-MD4-01: Wrap in try/except MemoryError — the
                    # numeric ds_data path (line ~1616) has this; the
                    # string_data path was missing it.  A huge list can
                    # trigger MemoryError from np.array() before the
                    # downstream len() guard fires.
                    try:
                        # PXM-06: pass dest_dict so alias-before-canonical
                        # per-section string_data keys are re-keyed (the
                        # section's curves_order is normalized LATER).
                        name = _norm_curve_mnem(name, dest_dict=ds_string_data)
                        ds_string_data[name] = np.atleast_1d(np.array(arr, dtype=object))
                    except (ValueError, TypeError, MemoryError, OverflowError) as e:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Cannot convert string data for section "
                            f"'{ds_name}', curve '{name}': {e}"
                        ) from e
                    # F-03: Per-array size guard for data section string_data arrays.
                    if len(ds_string_data[name]) > MAX_DATA_LINES:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"String data array length ({len(ds_string_data[name])}) "
                            f"for '{name}' in section '{ds_name}' exceeds maximum "
                            f"allowed ({MAX_DATA_LINES})"
                        )
                # F-91: Per-section string_data total element count guard.
                # ds_data has this check (below) and top-level las_file.logs
                # has it; string_data was the sole unguarded path.  Many
                # short per-array entries can pass the entry-count and
                # per-array length guards without triggering them — only a
                # product check catches this class of allocation DoS.
                if ds_string_data:
                    _sds_rows = max(len(arr) for arr in ds_string_data.values())
                    _sds_total = len(ds_string_data) * _sds_rows
                    if _sds_total > MAX_TOTAL_ELEMENTS:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Total string data allocation in section '{ds_name}' "
                            f"({len(ds_string_data)} curves x {_sds_rows} rows = "
                            f"{_sds_total} elements) exceeds maximum allowed "
                            f"({MAX_TOTAL_ELEMENTS})"
                        )
                ds_section_curves = []
                _sc_raw = _resolve_dict_entry(ds_dict, "section_curves", list, list)
                # F-19: Resource-exhaustion guard for section_curves — match
                # the same MAX_CURVES check used for top-level curves_data and
                # curves_order.  The parser path has this guard (parser.py:1344);
                # from_dict() was the sole unguarded path.
                if len(_sc_raw) >= MAX_CURVES:
                    ds_name = ds_dict.get("name", "<unknown>")
                    raise ValueError(
                        f"Number of section curves ({len(_sc_raw)}) in section "
                        f"'{ds_name}' exceeds maximum allowed ({MAX_CURVES})"
                    )
                # F-013: Validate every element is a dict.  Zero isinstance
                # check on _sc_raw elements previously — non-dict elements
                # crash at "array_info" in sc_dict / sc_dict.get().
                _sc_raw = _validate_iterable_of_dicts(_sc_raw, "section_curves")
                for sc_dict in _sc_raw:
                    sc_array_info = None
                    if "array_info" in sc_dict and isinstance(sc_dict["array_info"], dict):
                        ai = sc_dict["array_info"]
                        _sc_base_name = _safe_str(ai.get("base_name")).upper()
                        if _sc_base_name:
                            sc_array_info = ArrayElementInfo(
                                base_name=_sc_base_name,
                                index=_resolve_dict_entry(ai, "index", int, lambda: 0),
                                # F2-002: Validate time_offset — int(offset) in
                                # writer.py crashes on non-numeric values.
                                time_offset=_resolve_dict_entry(
                                    ai, "time_offset", (int, float), lambda: None
                                ),
                            )
                    _sc_raw_mnem = _safe_str(sc_dict.get("mnemonic", ""))
                    ds_section_curves.append(
                        CurveDefinition(
                            mnemonic=_norm_curve_mnem(_sc_raw_mnem),
                            unit=_safe_str(sc_dict.get("unit", "")),
                            api_code=_safe_str(sc_dict.get("api_code", "")),
                            description=_safe_str(sc_dict.get("description", "")),
                            original_mnemonic=_safe_str(sc_dict.get("original_mnemonic", "")),
                            data_format=_safe_str(sc_dict.get("data_format", "")),
                            array_info=sc_array_info,
                        )
                    )
                    # F-17: Cross-check bracket-notation mnemonic against
                    # array_info.base_name (per-section curves).  Same logic
                    # as top-level curves above.
                    if sc_array_info and "[" in _sc_raw_mnem:
                        if _sc_raw_mnem.split("[")[0] != sc_array_info.base_name:
                            warnings.warn(
                                f"from_dict warning: mnemonic "
                                f"{_sc_raw_mnem!r} uses array notation but "
                                f"array_info.base_name is "
                                f"{sc_array_info.base_name!r}. "
                                f"Cross-check mismatch may indicate "
                                f"malformed input.",
                                stacklevel=3,
                            )
                ds_data_raw = _resolve_dict_entry(ds_dict, "data", dict, dict)
                # F2-21: Per-section entry count guard.  Outer MAX_DATA_SECTIONS
                # guards section count; per-array MAX_DATA_LINES guards element
                # count.  Per-section curve entry count was unguarded — 1 section
                # x 200K single-element arrays passes all existing guards.
                if len(ds_data_raw) >= MAX_CURVES:
                    ds_name = ds_dict.get("name", "<unknown>")
                    raise ValueError(
                        f"Number of data curves ({len(ds_data_raw)}) in section "
                        f"'{ds_name}' exceeds maximum allowed ({MAX_CURVES})"
                    )
                ds_data = {}
                for k, v in ds_data_raw.items():
                    # F-I2-M02: None guard — np.array(None, dtype=np.float64)
                    # silently produces nan.  String paths already reject None
                    # (lines 603-608, 785-788); numeric paths lacked this guard.
                    if v is None:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Numeric data for curve '{k}' in section '{ds_name}' is None"
                        )
                    # F-012: Pre-allocation size check.  np.array() allocates
                    # BEFORE the downstream len() guard — a huge list triggers
                    # MemoryError (or system OOM) before the guard catches it.
                    # Check len() before allocation when possible.
                    if hasattr(v, "__len__") and len(v) > MAX_DATA_LINES:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Array length ({len(v)}) for curve '{k}' in "
                            f"section '{ds_name}' exceeds maximum allowed "
                            f"({MAX_DATA_LINES})"
                        )
                    try:
                        # PXM-06: pass dest_dict so alias-before-canonical
                        # per-section data keys are re-keyed (the section's
                        # curves_order is normalized LATER).
                        k = _norm_curve_mnem(k, dest_dict=ds_data)
                        # H-03: Per-section {I} branch — mirror the
                        # top-level logs path (L-03, ~line 3877).  The
                        # unconditional float64 coercion here destroyed
                        # {I} integer precision above 2^53 on the
                        # parse→to_dict→from_dict roundtrip
                        # (9007199254740993 → 9007199254740992.0,
                        # silent); efe7181 fixed the top-level sibling
                        # but missed this per-section site
                        # (asymmetric-fix regression).  Key on the
                        # per-section curve's data_format and the
                        # integrality of the declared NULL sentinel,
                        # exactly like the top-level path.  The M-14
                        # integrality guard (below) also prevents the
                        # int64 branch from silently truncating
                        # fractional/NaN data in per-section {I} curves.
                        _fmt = next(
                            (sc.data_format for sc in ds_section_curves if sc.mnemonic == k),
                            "",
                        )
                        _declared_null = las_file.well.get("NULL")
                        _null_ok = True
                        if _declared_null is not None:
                            try:
                                _null_ok = float(_declared_null).is_integer()
                            except (ValueError, TypeError):
                                _null_ok = False
                        if _fmt == "I" and _null_ok and _data_is_integral(v):
                            ds_data[k] = np.atleast_1d(np.array(v, dtype=np.int64))
                        elif _fmt == "I":
                            # EXT-04: fractional declared NULL or
                            # non-integral data — preserve exact values
                            # in an object array (never truncate).
                            ds_data[k] = np.atleast_1d(np.array(v, dtype=object))
                        else:
                            ds_data[k] = np.atleast_1d(np.array(v, dtype=np.float64))
                    except (ValueError, TypeError, MemoryError, OverflowError) as e:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Cannot convert data for section '{ds_name}', curve '{k}': {e}"
                        ) from e
                    # F-M02: Per-array size guard for data section arrays.
                    if len(ds_data[k]) > MAX_DATA_LINES:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Array length ({len(ds_data[k])}) for curve '{k}' in "
                            f"section '{ds_name}' exceeds maximum allowed "
                            f"({MAX_DATA_LINES})"
                        )
                # F-20: Per-section total element count guard — only top-level
                # las_file.logs had this check (below); data_section arrays were
                # unguarded.  Attack: 1,000 sections x 1 curve x 10M lines =
                # 80 GB, passing all per-array and per-section count guards.
                if ds_data:
                    _ds_rows = max(len(arr) for arr in ds_data.values())
                    _ds_total = len(ds_data) * _ds_rows
                    if _ds_total > MAX_TOTAL_ELEMENTS:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Total allocation in section '{ds_name}' "
                            f"({len(ds_data)} curves x {_ds_rows} rows = "
                            f"{_ds_total} elements) exceeds maximum allowed "
                            f"({MAX_TOTAL_ELEMENTS})"
                        )
                # F-I2-M20: Cross-validate data and string_data key sets within
                # the same DataSection.  When a curve name appears in both dicts,
                # the writer silently discards one (string_data wins — writer.py
                # line 703 checks string_data first).  Either way, data is lost.
                # Reject the input so callers get a clear error instead of silent
                # data discard on the roundtrip path.
                if ds_data and ds_string_data:
                    _colliding = set(ds_data.keys()) & set(ds_string_data.keys())
                    if _colliding:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"Section '{ds_name}': curve(s) {sorted(_colliding)} appear "
                            f"in both 'data' and 'string_data'.  Each curve must appear "
                            f"in exactly one collection."
                        )
                _ds_curves_order = ds_dict.get("curves_order", [])
                # F2-22: Guard against non-list iterables for per-section
                # curves_order — same bug as top-level F-21.
                if _ds_curves_order is None:
                    _ds_curves_order = []
                elif isinstance(_ds_curves_order, (str, bytes)):
                    # F-110: Same bytes bypass as top-level curves_order
                    # guard (line 808).  ``list(b"GR")`` → ``[71, 82]``
                    # produces integer curve names in a data section.
                    ds_name = ds_dict.get("name", "<unknown>")
                    if isinstance(_ds_curves_order, bytes):
                        raise TypeError(
                            f"curves_order in section '{ds_name}' must be "
                            f"a list of strings, got bytes.  Decode to "
                            f"str first: curves_order.decode('utf-8')"
                        )
                    raise ValueError(
                        f"curves_order in section '{ds_name}' must be a list, "
                        f"got str: {_ds_curves_order!r}"
                    )
                # F-I2E-05: Validate per-element types in per-section
                # curves_order.  Container type is guarded (None / str / bytes
                # above), but individual elements are not.  Non-string values
                # (int, None, float) survive when section_curves is empty
                # because the mnemonic cross-validation gate (L1496) is
                # inactive.  str(123) → "123" becomes a column header on
                # write; on re-read it looks like a genuine curve.
                # F-029: Materialize to prevent generator exhaustion
                # when iterated twice (validation loop + list comprehension).
                _ds_curves_order = list(_ds_curves_order)
                for _i, _item in enumerate(_ds_curves_order):
                    if not isinstance(_item, (str, bytes)):
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise TypeError(
                            f"curves_order[{_i}] in section '{ds_name}' "
                            f"must be str, got {type(_item).__name__}: "
                            f"{_item!r}"
                        )
                # F-MD4-03: Normalize curves_order entries through
                # mnem_base (matching parser's per-section normalization).
                # PXM-06: pass the in-progress list as dest so an
                # alias-before-canonical collision is re-keyed instead of
                # producing a duplicate canonical in the section order.
                _normed_ds_order: list[str] = []
                for _item in _ds_curves_order:
                    _normed_ds_order.append(_norm_curve_mnem(_item, _normed_ds_order))
                _ds_curves_order = _normed_ds_order
                # F-I2-M19: Detect duplicate curve names in section.  The parser
                # calls _deduplicate_curves() at data-read time to rename
                # collisions (append _2, _3 suffixes).  from_dict had zero dedup
                # calls, so duplicates passed silently — the pairwise zip
                # cross-validation (F-23 below) naturally pairs identical names
                # at matching positions.  Raise a clear error instead of silently
                # accepting duplicate curves.
                if _ds_curves_order:
                    _seen_n: set[str] = set()
                    for _n in _ds_curves_order:
                        if _n in _seen_n:
                            ds_name = ds_dict.get("name", "<unknown>")
                            raise ValueError(
                                f"Duplicate curve name {_n!r} in section '{ds_name}' "
                                f"curves_order.  Curve names must be unique within "
                                f"a data section."
                            )
                        _seen_n.add(_n)
                # Cross-validate string_data and data keys against
                # curves_order.  Keys in either dict that do not appear
                # in curves_order are orphaned — the writer silently
                # drops them, producing data loss on roundtrip.
                _curve_k = set(_ds_curves_order) if _ds_curves_order else set()
                if ds_string_data:
                    _str_orphaned = set(ds_string_data.keys()) - _curve_k
                    if _str_orphaned:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"string_data in section '{ds_name}' contains "
                            f"keys not in curves_order: {sorted(_str_orphaned)}. "
                            f"Each string_data key must correspond to a curve "
                            f"mnemonic."
                        )
                if ds_data:
                    _num_orphaned = set(ds_data.keys()) - _curve_k
                    if _num_orphaned:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"data in section '{ds_name}' contains "
                            f"keys not in curves_order: {sorted(_num_orphaned)}. "
                            f"Each data key must correspond to a curve mnemonic."
                        )
                # F-M-036: Detect uncovered curves in from_dict path.
                # The above checks detect orphaned data (data_keys - curve_set)
                # but not the reverse (curve_set - data_keys - string_keys).
                # Uncovered curves are recoverable (writer pads null_value)
                # so emit a warning rather than raising.
                _num_keys = set(ds_data.keys()) if ds_data else set()
                _str_keys = set(ds_string_data.keys()) if ds_string_data else set()
                if _curve_k:
                    _uncovered = _curve_k - _num_keys - _str_keys
                    if _uncovered:
                        ds_name = ds_dict.get("name", "<unknown>")
                        warnings.warn(
                            f"Section '{ds_name}': curve(s) "
                            f"{sorted(_uncovered)} appear in curves_order "
                            f"but have no data in 'data' or "
                            f"'string_data'.  The writer will pad these "
                            f"curves with null_value.",
                            stacklevel=2,
                        )
                if ds_section_curves:
                    _sc_mnemonics = [sc.mnemonic for sc in ds_section_curves]
                    _seen_m: set[str] = set()
                    for _m in _sc_mnemonics:
                        if _m in _seen_m:
                            ds_name = ds_dict.get("name", "<unknown>")
                            raise ValueError(
                                f"Duplicate mnemonic {_m!r} in section '{ds_name}' "
                                f"section_curves.  Curve mnemonics must be unique "
                                f"within a data section."
                            )
                        _seen_m.add(_m)
                # F-23: Cross-validate per-section curves_order with section_curves,
                # matching the top-level cross-validation pattern.  from_dict builds
                # these from independent dict keys; mismatched input would produce
                # silently inconsistent DataSection state.
                # Only validate when section_curves is non-empty — empty
                # section_curves means the section inherits curve definitions
                # from the top-level LASFile.curves (valid LAS 3.0 pattern).
                if ds_section_curves:
                    _sc_count = len(ds_section_curves)
                    if len(_ds_curves_order) != _sc_count:
                        ds_name = ds_dict.get("name", "<unknown>")
                        raise ValueError(
                            f"curves_order length ({len(_ds_curves_order)}) in section "
                            f"'{ds_name}' does not match section_curves length ({_sc_count})"
                        )
                    for _i, (_order_name, _sc) in enumerate(
                        zip(_ds_curves_order, ds_section_curves, strict=True)
                    ):
                        if _order_name != _sc.mnemonic:
                            ds_name = ds_dict.get("name", "<unknown>")
                            raise ValueError(
                                f"curves_order[{_i}] = {_order_name!r} in section "
                                f"'{ds_name}' does not match section_curves[{_i}].mnemonic "
                                f"= {_sc.mnemonic!r}"
                            )
                # F-14: Per-section curves_order bound.  When section_curves is
                # empty the length cross-validation above gates out, leaving
                # _ds_curves_order unbounded.  Every other iterable in from_dict
                # has a count guard — this was the sole unguarded path.
                if len(_ds_curves_order) >= MAX_CURVES:
                    ds_name = ds_dict.get("name", "<unknown>")
                    raise ValueError(
                        f"Number of curves_order entries ({len(_ds_curves_order)}) "
                        f"in section '{ds_name}' exceeds maximum allowed "
                        f"({MAX_CURVES})"
                    )
                # I2F-15: Cumulative cross-section allocation check.
                # Each section's per-section total is validated above, but
                # multiple sections can collectively exceed MAX_TOTAL_ELEMENTS.
                # Compute this section's element total and check against the
                # running cumulative count BEFORE allocating the DataSection.
                _section_total = 0
                if ds_data:
                    _section_total += len(ds_data) * max(len(arr) for arr in ds_data.values())
                if ds_string_data:
                    _section_total += len(ds_string_data) * max(
                        len(arr) for arr in ds_string_data.values()
                    )
                if _cumulative_elements + _section_total > MAX_TOTAL_ELEMENTS:
                    ds_name = ds_dict.get("name", "<unknown>")
                    raise ValueError(
                        f"Cumulative cross-section allocation "
                        f"({_cumulative_elements}+{_section_total} = "
                        f"{_cumulative_elements + _section_total} elements) "
                        f"in section '{ds_name}' exceeds maximum allowed "
                        f"({MAX_TOTAL_ELEMENTS})"
                    )
                _cumulative_elements += _section_total
                ds = DataSection(
                    name=_safe_str(ds_dict.get("name"), ""),
                    section_type=_safe_str(ds_dict.get("section_type"), "LOG_DATA"),
                    curves_order=list(_ds_curves_order),
                    data=ds_data,
                    string_data=ds_string_data,
                    section_curves=ds_section_curves,
                    _from_dict=True,
                )
                las_file.data_sections.append(ds)
                # F-25: Cross-array length validation within this data section.
                # Inconsistent-length arrays produce silently corrupted output if
                # accepted — different curves with different sample counts in the
                # same data section represent invalid LAS data (see F-25).
                if len(ds.data) > 1:
                    _ds_len = {name: len(arr) for name, arr in ds.data.items()}
                    if len(set(_ds_len.values())) > 1:
                        raise ValueError(
                            f"Data section '{ds.name}' has inconsistent array lengths: {_ds_len}"
                        )
                # F-004: Cross-array length validation for string_data within
                # this data section (PRIOR_FIX_ATTEMPT in b47eea6/24c4f5c only
                # covered numeric data — string_data was missed on both levels).
                if len(ds.string_data) > 1:
                    _sds_len = {name: len(arr) for name, arr in ds.string_data.items()}
                    if len(set(_sds_len.values())) > 1:
                        raise ValueError(
                            f"Data section '{ds.name}' has inconsistent "
                            f"string_data array lengths: {_sds_len}"
                        )
                # F-037: Cross-group row-count consistency.
                # The within-group checks above validate that all arrays
                # in 'data' have the same length, and all arrays in
                # 'string_data' have the same length — but no cross-check
                # verifies that data rows and string_data rows match.
                # A DataSection with 100-row numeric data and 50-row
                # string_data passes all existing validation.  The writer's
                # ``_format_data_rows`` uses ``max()`` across all arrays,
                # so one group's shorter arrays get padded — producing
                # semantically incorrect output.
                if ds.data and ds.string_data:
                    _data_rows = max(len(arr) for arr in ds.data.values())
                    _string_rows = max(len(arr) for arr in ds.string_data.values())
                    if _data_rows != _string_rows:
                        raise ValueError(
                            f"DataSection '{ds.name}': data row count "
                            f"({_data_rows}) does not match string_data "
                            f"row count ({_string_rows})"
                        )

            # F-I2-M17: Cross-validate data_sections with LAS version.
            # data_sections are a LAS 3.0 feature; non-LAS-3.0 files with
            # data_sections cause silent roundtrip data loss (writer emits
            # multi-section format, parser skips all data for non-LAS-3.0).
            if las_file.data_sections and not las_file.version.is_las30:
                warnings.warn(
                    f"data_sections are present but version is "
                    f"{las_file.version.vers!r} (not LAS 3.0). "
                    f"Data sections are only valid for LAS 3.0 files.",
                    stacklevel=2,
                )

            # F-020: Wire _validate_data_section_column_counts() into the
            # from_dict path.  The function was extracted from the parser
            # specifically so that from_dict could also call it (per its
            # docstring), but the import and call were never added.
            if las_file.data_sections:
                _validate_data_section_column_counts(las_file.data_sections)

            # Restore LAS 3.0 string data (top-level, backward compat
            # with data serialized before string_data was moved to
            # per-section DataSection objects).
            sd = _resolve_dict_entry(data, "string_data", dict, dict)
            # F-24: Top-level string_data entry count guard — same gap as
            # the per-section path fixed above.
            if len(sd) >= MAX_CURVES:
                raise ValueError(
                    f"Number of string data curves ({len(sd)}) exceeds "
                    f"maximum allowed ({MAX_CURVES})"
                )
            for name, arr in sd.items():
                # F-22: Guard against None — same bug as per-section
                # string_data above.
                if arr is None:
                    raise ValueError(f"String data for curve '{name}' is None")
                # F-MD4-02: Pre-allocation size check (same pattern as
                # numeric logs path).  np.array() allocates before the
                # downstream len() guard — a huge list triggers MemoryError
                # before the guard catches it.
                if hasattr(arr, "__len__") and len(arr) > MAX_DATA_LINES:
                    raise ValueError(
                        f"String data array length ({len(arr)}) for "
                        f"'{name}' exceeds maximum allowed "
                        f"({MAX_DATA_LINES})"
                    )
                # F-I2-MD4-01: Wrap in try/except MemoryError — the
                # numeric logs path (line ~1952) has this; string_data
                # was missing it.  A huge list can trigger MemoryError
                # from np.array() before the downstream len() guard fires.
                try:
                    name = _norm_curve_mnem(name)
                    las_file.string_data[name] = np.atleast_1d(np.array(arr, dtype=object))
                except (ValueError, TypeError, MemoryError, OverflowError) as e:
                    raise ValueError(
                        f"Cannot convert string data for curve '{name}' to string array: {e}"
                    ) from e
                # F-M02: Per-array size guard for string_data arrays.
                if len(las_file.string_data[name]) > MAX_DATA_LINES:
                    raise ValueError(
                        f"Array length ({len(las_file.string_data[name])}) for "
                        f"string curve '{name}' exceeds maximum allowed "
                        f"({MAX_DATA_LINES})"
                    )
            # F-13: Top-level string_data total element count guard.
            # logs has this check (below); string_data was the sole unguarded
            # top-level path.  As with F-91 (per-section), many short arrays
            # can pass per-array and entry-count guards without a product check.
            if las_file.string_data:
                _str_rows = max(len(arr) for arr in las_file.string_data.values())
                _str_total = len(las_file.string_data) * _str_rows
                if _str_total > MAX_TOTAL_ELEMENTS:
                    raise ValueError(
                        f"Total string data allocation "
                        f"({len(las_file.string_data)} curves x {_str_rows} rows = "
                        f"{_str_total} elements) exceeds maximum allowed "
                        f"({MAX_TOTAL_ELEMENTS})"
                    )

            # F2-11: Validate string_data keys against curves_order
            # (matching per-section guard at lines 1995-2005).
            # Keys in string_data not in curves_order are orphaned.
            if las_file.string_data and las_file.curves_order:
                _order_s = set(las_file.curves_order)
                _str_orphaned = set(las_file.string_data.keys()) - _order_s
                if _str_orphaned:
                    raise ValueError(
                        f"string_data contains keys not in "
                        f"curves_order: {sorted(_str_orphaned)}. "
                        f"Each string_data key must correspond to a "
                        f"curve mnemonic."
                    )

            logs = _resolve_dict_entry(data, "logs", dict, dict)
            # F-06: Resource-exhaustion guard for logs.
            if len(logs) >= MAX_CURVES:
                raise ValueError(
                    f"Number of log curves ({len(logs)}) exceeds maximum allowed ({MAX_CURVES})"
                )
            for name, arr in logs.items():
                # F-I2-M02: None guard — np.array(None, dtype=np.float64)
                # silently produces nan, consistent with string data guards.
                if arr is None:
                    raise ValueError(f"Log data for curve '{name}' is None")
                # F-012: Pre-allocation size check (same pattern as ds_data
                # above).  np.array() allocates before the downstream len()
                # guard catches oversized inputs.
                if hasattr(arr, "__len__") and len(arr) > MAX_DATA_LINES:
                    raise ValueError(
                        f"Array length ({len(arr)}) for log "
                        f"'{name}' exceeds maximum allowed "
                        f"({MAX_DATA_LINES})"
                    )
                try:
                    name = _norm_curve_mnem(name)
                    # L-03: Preserve int64 dtype for {I} integer-format
                    # curves.  curves (and their data_format) are parsed
                    # above, before logs — a forced float64 coercion here
                    # would silently round integer values above 2^53 (e.g.
                    # 9007199254740993 → 9007199254740992.0), destroying
                    # the precision the reader intentionally preserved.
                    # The int64 branch is gated on an integral declared
                    # NULL (mirroring the reader's `_null_is_integral`
                    # rule) — int64 would truncate a fractional NULL like
                    # -999.25, corrupting every null cell.
                    _fmt = next(
                        (c.data_format for c in las_file.curves if c.mnemonic == name),
                        "",
                    )
                    _declared_null = las_file.well.get("NULL")
                    _null_ok = True
                    if _declared_null is not None:
                        try:
                            _null_ok = float(_declared_null).is_integer()
                        except (ValueError, TypeError):
                            _null_ok = False
                    if _fmt == "I" and _null_ok and _data_is_integral(arr):
                        las_file.logs[name] = np.atleast_1d(np.array(arr, dtype=np.int64))
                    elif _fmt == "I":
                        # EXT-04: fractional declared NULL — preserve the
                        # reader's object dtype (exact Python ints for data
                        # values, float sentinel for null cells).  Coercing
                        # to float64 would silently round {I} values above
                        # 2^53 (e.g. 9007199254740993 → 9007199254740992.0).
                        # M-14: also reached when the NULL is integral but
                        # the DATA is non-integral (fractional values would
                        # be silently truncated to int64, and NaN would be
                        # converted to 0 — NaN is the standard missing-data
                        # marker, so it must never become a real zero).
                        # Routing to object dtype preserves every value
                        # exactly instead of silently altering it.
                        las_file.logs[name] = np.atleast_1d(np.array(arr, dtype=object))
                    else:
                        las_file.logs[name] = np.atleast_1d(np.array(arr, dtype=np.float64))
                except (ValueError, TypeError, MemoryError, OverflowError) as e:
                    raise ValueError(
                        f"Cannot convert log data for curve '{name}' to numeric array: {e}"
                    ) from e
                # F-M02: Per-array size guard for log arrays.
                if len(las_file.logs[name]) > MAX_DATA_LINES:
                    raise ValueError(
                        f"Array length ({len(las_file.logs[name])}) for log "
                        f"'{name}' exceeds maximum allowed ({MAX_DATA_LINES})"
                    )

            # F-11: Detect key overlap between logs and string_data.
            # A curve name appearing in both would silently have data
            # corrupted — one array overwrites the other's meaning.
            # (matches __post_init__ guard at lines 1209-1216)
            if las_file.logs and las_file.string_data:
                _overlap = set(las_file.logs.keys()) & set(las_file.string_data.keys())
                if _overlap:
                    raise ValueError(
                        f"Curves {sorted(_overlap)} appear in "
                        f"both logs and string_data.  Each curve may "
                        f"only be stored in one location."
                    )

            # F-011: Validate that log curve keys match curves_order exactly.
            # The length check above ensures count matches; this catches phantom
            # keys (extra curves in logs not in curves_order) and missing keys.
            # Only valid for legacy LAS 1.2/2.0 files where all curve data lives
            # in the logs dict.  LAS 3.0 files distribute curve data across
            # data_sections and string_data, so curves_order typically includes
            # curves whose data is in those sections, not in logs.
            if las_file.logs and not las_file.data_sections:
                # G-015: Validate all log keys are strings before
                # _norm_mnem normalisation, mirroring the well section
                # key validation pattern (lines 1428-1433).
                for _lk in las_file.logs:
                    if not isinstance(_lk, str):
                        raise TypeError(
                            f"Log dict key must be str, got {type(_lk).__name__}: {_lk!r}"
                        )
                # F-M-013: Compare log keys directly — from_dict normalizes
                # every log key through mnem_base at storage time (line
                # ~3725), so the stored keys are already canonical.  A
                # second _norm_mnem pass here would UNDO N-I-30's
                # collision-avoidance (a kept-original key like 'LLS' that
                # is itself a mnem_base entry would re-resolve to 'BFV' and
                # produce a false "Missing keys" error).
                _log_keys = set(las_file.logs.keys())
                _order_keys = set(las_file.curves_order)
                # M-28: Subtract string_data keys from the MISSING side
                # only (stored keys are already normalized — see F-M-013
                # comment above).  For LAS 1.2/2.0 files (and LAS 3.0 files
                # without data_sections), the reader routes {S} string
                # curves to string_data (data_reader.py F-WXP-01) while
                # curves_order still includes them — a string curve in
                # string_data legitimately has no log entry.  The "Extra
                # keys" direction must NOT subtract: a log key with no
                # curve definition is always an error.
                _extra_log_keys = _log_keys - _order_keys
                _str_data_keys = set(las_file.string_data.keys())
                _missing_log_keys = _order_keys - _log_keys - _str_data_keys
                if _extra_log_keys or _missing_log_keys:
                    raise ValueError(
                        f"Log curve keys do not match curves_order. "
                        f"Extra keys: {_extra_log_keys}, "
                        f"Missing keys: {_missing_log_keys}"
                    )

            # F-25: Cross-array length validation for top-level log arrays.
            # Inconsistent-length arrays produce silently corrupted output if
            # accepted — different curves with different sample counts represent
            # invalid data (see F-25).
            if len(las_file.logs) > 1:
                _log_len = {name: len(arr) for name, arr in las_file.logs.items()}
                if len(set(_log_len.values())) > 1:
                    raise ValueError(f"Log arrays have inconsistent lengths: {_log_len}")

            # F-004: Cross-array length validation for top-level string_data
            # arrays.  Prior fix rounds (b47eea6, 24c4f5c) added this check
            # for numeric logs but missed string_data at both the per-section
            # and top-level paths.
            if len(las_file.string_data) > 1:
                _str_len = {name: len(arr) for name, arr in las_file.string_data.items()}
                if len(set(_str_len.values())) > 1:
                    raise ValueError(f"String data arrays have inconsistent lengths: {_str_len}")

            # F-M02: Total element count guard across all log curves.
            if las_file.logs:
                _log_rows = max(len(arr) for arr in las_file.logs.values())
                _log_total = len(las_file.logs) * _log_rows
                if _log_total > MAX_TOTAL_ELEMENTS:
                    raise ValueError(
                        f"Total log allocation ({len(las_file.logs)} curves x "
                        f"{_log_rows} rows = {_log_total} elements) exceeds maximum "
                        f"allowed ({MAX_TOTAL_ELEMENTS})"
                    )
            # F-I2-M19: Detect duplicate curve names at the top level
            # for legacy (non-LAS-3.0) files.  LAS 3.0 files distribute
            # curve data across multiple data_sections — the same curve
            # name like "DEPT" legitimately appears in every section's
            # curves_order, producing duplicates in the top-level list.
            # The parser calls _deduplicate_curves() before data allocation
            # for legacy files; from_dict rejects duplicates for these.
            if las_file.curves_order and not las_file.data_sections:
                _top_seen: set[str] = set()
                for _n in las_file.curves_order:
                    if _n in _top_seen:
                        raise ValueError(
                            f"Duplicate curve name {_n!r} in top-level "
                            f"curves_order.  Curve names must be unique."
                        )
                    _top_seen.add(_n)

            # G-008: Cross-group row count validation between logs and
            # string_data.  Within-group row checks exist (above); this
            # catches mismatched row counts across the two groups.  The
            # per-section equivalent is in DataSection.__post_init__
            # (lines 1015-1023) — same pattern.
            if las_file.logs and las_file.string_data:
                _max_log_rows = max(len(arr) for arr in las_file.logs.values())
                _max_str_rows = max(len(arr) for arr in las_file.string_data.values())
                if _max_log_rows != _max_str_rows:
                    raise ValueError(
                        f"Logs row count ({_max_log_rows}) does not match "
                        f"string_data row count ({_max_str_rows})"
                    )

            # I2F-13: Re-run all __post_init__ validations on the
            # fully-populated object.  __post_init__ skips checks
            # when collections are empty (incremental construction),
            # so from_dict must re-validate after populating everything.
            # The _from_dict flag suppresses warnings that were
            # appropriate at direct-construction time but are noise
            # during from_dict re-validation.
            las_file._from_dict = True
            try:
                las_file.__post_init__()
                # Run deferred validate(complete=True) while _from_dict
                # suppresses redundant warnings.
                _complete_issues = las_file.validate(complete=True)
                for issue in _complete_issues:
                    warnings.warn(issue, stacklevel=2)
            finally:
                las_file._from_dict = False

            return las_file
        except (ValueError, TypeError, OverflowError) as e:
            raise LASDataError(str(e)) from e

    @property
    def is_las30(self) -> bool:
        """Check if this is a LAS 3.0 file."""
        return self.version.is_las30

    def get_curve_by_mnemonic(self, mnemonic: str) -> CurveDefinition | None:
        """Get curve definition by mnemonic name.

        Searches the curve list for an exact mnemonic match. For LAS 3.0
        array curves (e.g., ``NMR[1]``), also matches on the base mnemonic
        (e.g., ``"NMR"``) so that callers can find array elements without
        knowing the exact index.

        Args:
            mnemonic: Curve mnemonic to look up (e.g., ``"GR"``, ``"NMR"``).

        Returns:
            Matching ``CurveDefinition`` if found, or ``None`` if no curve
            matches the given mnemonic.
        """
        for curve in self.curves:
            if curve.mnemonic == mnemonic or curve.base_mnemonic == mnemonic:
                return curve
        # F-038: Also search data_sections[].section_curves for LAS 3.0.
        # Curves may be defined per-section rather than at top level.
        for ds in self.data_sections:
            for curve in ds.section_curves:
                if curve.mnemonic == mnemonic or curve.base_mnemonic == mnemonic:
                    return curve
        return None

    def get_array_curves(self, base_name: str) -> list[CurveDefinition]:
        """Get all array-element curves for a base name (LAS 3.0).

        LAS 3.0 files use array notation (e.g., ``NMR[1]``, ``NMR[2]``) to
        represent multi-element curves. This method collects all curve
        definitions that belong to the same logical array group.

        Args:
            base_name: Base mnemonic of the array (e.g., ``"NMR"``).

        Returns:
            List of ``CurveDefinition`` objects whose ``array_info.base_name``
            matches *base_name*. Returns an empty list if no array curves match.
        """
        # M-21 (coordinated with N-I-10): Dedupe by mnemonic.  For LAS 3.0
        # multi-section files the same logical array element can appear in
        # both the top-level ``curves`` list and a section's
        # ``section_curves`` — without dedup every element is returned twice.
        result: list[CurveDefinition] = []
        seen: set[str] = set()
        for c in self.curves:
            if c.array_info and c.array_info.base_name == base_name and c.mnemonic not in seen:
                seen.add(c.mnemonic)
                result.append(c)
        for ds in self.data_sections:
            for c in ds.section_curves:
                if c.array_info and c.array_info.base_name == base_name and c.mnemonic not in seen:
                    seen.add(c.mnemonic)
                    result.append(c)
        return result


# F-015: Validating dict wrapper for DevFile.columns.  Direct mutation
# like ``dev.columns["NEW"] = arr`` bypasses all validation when columns
# is a plain dict — no __setitem__ guard, no column_order sync, no
# length consistency check.  This wrapper intercepts __setitem__ and
# __delitem__ to validate, sync column_order, and enforce resource limits.
class _DevColumns(dict[str, NDArray[np.float64]]):
    """Validating dict for DevFile columns that intercepts mutation."""

    __slots__ = ("_dev",)

    def __init__(
        self, dev: DevFile, mapping: dict[str, NDArray[np.float64]] | None = None, /, **kwargs: Any
    ) -> None:
        self._dev = dev
        super().__init__(mapping or {}, **kwargs)

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError(f"DevFile column keys must be str, got {type(key).__name__}")
        # Convert to numpy array matching from_dict behaviour.
        # MOD-17: reject ndim>=2 before coercion — a 2-D column (e.g. a
        # transposed matrix) would otherwise pass the length guards and
        # crash validate() with a raw IndexError on boolean-mask indexing.
        _check_column_array_like(value, f"DevFile column '{key}'")
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        # Validate length consistency with existing columns.
        if self:
            existing_len = len(next(iter(self.values())))
            if len(arr) != existing_len:
                raise ValueError(
                    f"DevFile: column '{key}' has length {len(arr)} "
                    f"but existing columns have length {existing_len}"
                )
        # F-016: Per-column size guard mirrors from_dict guards.
        from .data_reader import MAX_DATA_LINES

        if len(arr) > MAX_DATA_LINES:
            raise ValueError(
                f"DevFile: column '{key}' length ({len(arr)}) "
                f"exceeds maximum allowed ({MAX_DATA_LINES})"
            )
        super().__setitem__(key, arr)
        # Sync column_order — add key if not already present.
        if key not in self._dev.column_order:
            self._dev.column_order.append(key)

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        if key in self._dev.column_order:
            self._dev.column_order.remove(key)

    # N-I-14: CPython's C-level dict methods (update/pop/setdefault/clear)
    # do NOT route through the Python __setitem__/__delitem__ overrides, so
    # they silently bypassed validation, length checks, and column_order
    # sync — columns/column_order desynced and the to_dict/from_dict
    # roundtrip raised LASDataError.  Each method is overridden to route
    # through the guarded operations (PFA a6096f4 added the class with only
    # __setitem__/__delitem__, incomplete from the start).
    def update(self, *args: Any, **kwargs: Any) -> None:
        # Validate/coerce every key-value pair through __setitem__ so
        # column_order sync, length consistency, and dtype coercion apply.
        if args:
            if len(args) > 1:
                raise TypeError(f"update expected at most 1 argument, got {len(args)}")
            _other = args[0]
            if hasattr(_other, "keys"):
                for _k in _other:
                    self[_k] = _other[_k]
            else:
                for _k, _v in _other:
                    self[_k] = _v
        for _k, _v in kwargs.items():
            self[_k] = _v

    def setdefault(self, key: str, default: Any = None) -> Any:
        # Route the insert through __setitem__ so the new column is
        # coerced, length-checked, and appended to column_order.
        if key in self:
            return self[key]
        self[key] = default
        return self[key]

    def pop(self, key: str, *args: Any) -> Any:
        # Route the removal through __delitem__ so column_order stays in
        # sync.  dict.pop's two-argument default semantics are preserved.
        try:
            _value = self[key]
        except KeyError:
            if args:
                return args[0]
            raise KeyError(key) from None
        del self[key]
        return _value

    def popitem(self) -> tuple[str, NDArray[np.float64]]:
        # M-18: dict.popitem (C-level) bypasses __delitem__, so it popped
        # the last column WITHOUT removing it from column_order — the
        # stale order made to_dict/from_dict raise LASDataError.  Route
        # through __delitem__ (same pattern as pop above) so column_order
        # stays in sync.  LIFO semantics match dict.popitem.
        if not self:
            raise KeyError("popitem(): dictionary is empty")
        _key = next(reversed(self))
        _value = self[_key]
        del self[_key]
        return _key, _value

    def clear(self) -> None:
        # Clear column_order with the dict — a desynced column_order that
        # still names removed columns breaks the to_dict/from_dict contract.
        super().clear()
        self._dev.column_order.clear()

    def __ior__(self, other: Any) -> _DevColumns:  # type: ignore[misc,override]
        # M-13: dict.__ior__ (|=) bypasses the Python __setitem__
        # override, so it previously skipped str-key validation, float64
        # coercion, length consistency, MAX_DATA_LINES, AND column_order
        # sync — wrong-length/int-key inserts were silently accepted and
        # the to_dict/from_dict roundtrip raised LASDataError.  Route
        # through update() (which routes through __setitem__) so every
        # guard applies.
        self.update(other)
        return self

    def __reduce__(self) -> Any:
        # M-48: dict subclasses with __slots__ do not unpickle by default —
        # reconstruction bypasses __init__, so the _dev slot is never set
        # and any access raises AttributeError (reproduced on DevFile AND
        # bare _DevColumns).  Reconstruct through __init__ (which re-binds
        # _dev) with the dict content.  Exact sibling of the Round-1 M-16
        # _GuardedList fix above.
        return (
            self.__class__,
            (self._dev, dict(self)),
        )


class _DevColumnOrder(list[str]):
    """Validating list for DevFile.column_order.

    M-46: column_order was an unguarded plain list.  Direct mutation
    (``dev.column_order.append("GHOST")``) desynced order from columns and
    broke the to_dict→from_dict roundtrip with LASDataError, and construction
    stored the caller's list BY REFERENCE (caller mutation — including a
    silent ``reverse()`` reorder — corrupted the model with zero warnings).
    The guarded list validates item type, rejects duplicates, routes every
    ADD through the same column-existence invariant ``_DevColumns`` maintains
    (column_order may only reference existing columns), and routes REMOVE/
    pop/clear through the same invariant (an entry may not be removed while
    its column still exists — use ``del dev.columns[k]`` instead), so order
    and columns cannot desync via mutation.
    """

    __slots__ = ("_dev",)

    def __init__(self, dev: DevFile, values: Iterable[str] = ()) -> None:
        self._dev = dev
        # Materialize FIRST (a one-shot iterable would be consumed by the
        # validation loop and then re-iterated empty by list.__init__).
        # list(values) also guarantees the stored list is a fresh object —
        # the caller's list is never stored by reference.
        _items = list(values)
        for _item in _items:
            self._validate_item(_item)
        # F-12: wholesale (re)assignment through this constructor must
        # validate the same invariants __post_init__ enforces for the
        # construction-time pair — duplicates and column existence.  The
        # __setattr__ intercept re-wraps every ``dev.column_order = [...]``
        # here, and __post_init__ never re-runs post-construction, so
        # ``['GHOST']``/``['MD','MD']`` were previously accepted and
        # silently desynced order from columns.  LASDataError (a ValueError
        # subclass) matches the __post_init__ E-F-026 contract.  Column
        # existence is enforced only once columns are populated — empty
        # columns are allowed for incremental population (the __post_init__
        # ``if not self.columns: return`` design).
        if _items:
            from .exceptions import LASDataError

            _seen: set[str] = set()
            # getattr: during unpickling (_DevColumnOrder.__reduce__ →
            # __init__) the _dev backref may be a partially-restored
            # DevFile whose ``columns`` attribute is not set yet — the
            # pickled snapshot was validated when it was created, so the
            # existence check can only run when columns is available.
            _columns = getattr(dev, "columns", None)
            for _item in _items:
                if _item in _seen:
                    raise LASDataError(
                        f"DevFile: column_order contains duplicate "
                        f"entries: {_item}.  Each column may only "
                        f"appear once."
                    )
                _seen.add(_item)
                if _columns and _item not in _columns:
                    raise LASDataError(
                        f"DevFile: column_order entry '{_item}' is not "
                        f"a column in dev.columns.  column_order may "
                        f"only reference existing columns."
                    )
        super().__init__(_items)

    def _validate_item(self, item: Any) -> None:
        if not isinstance(item, str):
            raise TypeError(f"DevFile column_order entries must be str, got {type(item).__name__}")

    def _check_add(self, item: Any) -> None:
        self._validate_item(item)
        if item in self:
            raise ValueError(
                f"DevFile: column_order already contains '{item}'.  "
                f"Each column may only appear once."
            )
        if item not in self._dev.columns:
            raise ValueError(
                f"DevFile: column_order entry '{item}' is not a "
                f"column in dev.columns.  column_order may only "
                f"reference existing columns."
            )

    def _check_remove(self, item: Any) -> None:
        if item in self._dev.columns:
            raise ValueError(
                f"DevFile: cannot remove column '{item}' from "
                f"column_order while it still exists in columns.  "
                f"Use 'del dev.columns[{item!r}]' to remove a column."
            )

    def append(self, item: Any) -> None:
        self._check_add(item)
        super().append(item)

    def insert(self, index: SupportsIndex, item: Any) -> None:
        self._check_add(item)
        super().insert(index, item)

    def extend(self, items: Iterable[Any]) -> None:
        _items = list(items)
        for _item in _items:
            self._check_add(_item)
        super().extend(_items)

    def __iadd__(self, other: Any) -> _DevColumnOrder:  # type: ignore[misc,override]
        _other = list(other)
        for _item in _other:
            self._check_add(_item)
        return super().__iadd__(_other)

    def __setitem__(self, index: Any, item: Any) -> None:
        if isinstance(index, slice):
            _old = list(self[index])
            _items = list(item)
            # F-11: slice assignment REPLACES/REMOVES the entries in
            # *index* — a shrinking/empty slice (e.g. ``[1:2] = []``)
            # previously deleted entries without the remove-side desync
            # guard, silently desyncing order from columns.  Route every
            # entry being dropped (not re-inserted by the new items)
            # through _check_remove — an entry may not be removed while
            # its column still exists — mirroring remove/pop/__delitem__
            # and the add-side _check_add below.
            for _old_item in _old:
                if _old_item not in _items:
                    self._check_remove(_old_item)
            for _it in _items:
                self._check_add(_it)
            item = _items
        else:
            self._check_add(item)
        super().__setitem__(index, item)

    def remove(self, value: Any) -> None:
        self._check_remove(value)
        super().remove(value)

    def pop(self, index: Any = -1) -> Any:
        # list.pop's IndexError/ValueError semantics are preserved by
        # delegating the actual removal to super().
        _value = self[index]
        self._check_remove(_value)
        return super().pop(index)

    def __delitem__(self, index: Any) -> None:
        _value = self[index]
        self._check_remove(_value)
        super().__delitem__(index)

    def clear(self) -> None:
        # Clearing the order is only consistent while columns is also
        # empty (_DevColumns.clear() empties the dict BEFORE clearing the
        # order).  A non-empty columns dict with an empty order breaks the
        # to_dict→from_dict roundtrip (LASDataError).
        if self._dev.columns:
            raise ValueError(
                f"DevFile: cannot clear column_order while "
                f"columns still contains {sorted(self._dev.columns)}.  "
                f"Use 'dev.columns.clear()' to remove all columns."
            )
        super().clear()

    def __reduce__(self) -> Any:
        # M-48 twin: list subclass with __slots__ — reconstruct through
        # __init__ so _dev is re-bound on unpickle.
        return (
            self.__class__,
            (self._dev, list(self)),
        )


@dataclass(eq=False)
class DevFile:
    """DEV (deviation survey) file data structure."""

    columns: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    column_order: list[str] = field(default_factory=list)
    source_file: str = ""
    encoding: str = "utf-8"

    # I2F-01: from_dict re-validation control flag.  When True,
    # __post_init__ suppresses the validate(complete=True) call —
    # the from_dict path calls validate() explicitly after full
    # population.  Consistent with LASFile._from_dict and
    # DataSection._from_dict.
    _from_dict: bool = field(default=False, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        """M-45/M-46/M-49: Guarded assignment for ``columns`` and
        ``column_order``.

        Post-construction ``dev.columns = <plain dict>`` wholesale
        reassignment previously replaced the ``_DevColumns`` wrapper with
        a plain dict — dropping EVERY guard (str-key validation, float64
        coercion, length consistency, MAX_DATA_LINES, column_order sync).
        Wrong-length columns were silently accepted, ``to_dict`` emitted a
        desynced model, and ``DevFile.from_dict`` raised LASDataError
        (validate() even crashed with a raw IndexError on the desynced
        state).  ``dev.columns.copy()`` hit the same bypass.

        M-49: passing a pre-wrapped ``_DevColumns`` (the natural copy idiom
        ``DevFile(columns=old_dev.columns, ...)``) previously passed through
        BY REFERENCE — ``dev2.columns is dev1.columns``, the ``_dev`` backref
        still pointed at dev1, and mutation through dev2 corrupted dev1 while
        leaving dev2's column_order stale (roundtrip LASDataError).  The
        content is now deep-copied and re-wrapped bound to THIS DevFile.

        M-46: ``column_order`` is now wrapped in ``_DevColumnOrder`` (deepcopy
        + re-wrap on every assignment), so caller lists are never stored by
        reference and post-construction mutation cannot desync order→columns.

        Intercept ALL ``columns``/``column_order`` assignments (including the
        dataclass ``__init__`` path) and re-wrap through the same validation
        ``__post_init__`` applies, so reassignment can never bypass the
        guards.
        """
        if name != "columns":
            if name == "column_order":
                # M-46: Deepcopy + re-wrap through _DevColumnOrder.  For an
                # already-wrapped input (e.g. dev2.column_order =
                # dev1.column_order), copy the CONTENT only — deepcopying the
                # wrapper itself would copy its _dev backref (the wrong
                # DevFile).  list[str] is immutable-element, so the
                # materialized copy in _DevColumnOrder.__init__ makes the
                # storage private.
                if isinstance(value, _DevColumnOrder):
                    value = list(value)
                super().__setattr__(
                    name,
                    _DevColumnOrder(self, copy.deepcopy(value)),
                )
                return
            super().__setattr__(name, value)
            return
        # N-I-11: Direct assignment must not alias the caller's dict or its
        # arrays — deepcopy makes the storage private, and coercing list
        # values to ndarray (matching _DevColumns.__setitem__) makes
        # downstream validate() safe.
        if not isinstance(value, dict):
            raise TypeError(f"DevFile: columns must be a dict, got {type(value).__name__}")
        if isinstance(value, _DevColumns):
            # M-49: Content-only copy — dict(value) drops the _dev backref
            # so deepcopy does not also copy the SOURCE DevFile; the copy is
            # re-wrapped bound to this DevFile below.
            _cols = copy.deepcopy(dict(value))
        else:
            _cols = copy.deepcopy(value)
        _coerced: dict[str, NDArray[np.float64]] = {}
        for _k, _v in _cols.items():
            if not isinstance(_k, str):
                raise TypeError(f"DevFile column keys must be str, got {type(_k).__name__}")
            # MOD-17: reject ndim>=2 columns on wholesale assignment.
            _check_column_array_like(_v, f"DevFile column '{_k}'")
            _coerced[_k] = np.atleast_1d(np.asarray(_v, dtype=np.float64))
        # Length consistency + MAX_DATA_LINES on the assigned set (mirrors
        # _DevColumns.__setitem__ and __post_init__).  LASDataError matches
        # the exception __post_init__ raises for the same whole-dict
        # checks (E-F-026 contract); it IS a ValueError subclass, so
        # callers catching ValueError still capture these errors.
        from .exceptions import LASDataError

        if len(_coerced) > 1:
            _lens = {
                _k: (1 if isinstance(_a, np.ndarray) and _a.ndim == 0 else len(_a))
                for _k, _a in _coerced.items()
            }
            if len(set(_lens.values())) > 1:
                raise LASDataError(f"DevFile: columns have inconsistent array lengths: {_lens}")
        from .data_reader import MAX_DATA_LINES

        for _k, _a in _coerced.items():
            if len(_a) > MAX_DATA_LINES:
                raise LASDataError(
                    f"DevFile: column '{_k}' length ({len(_a)}) "
                    f"exceeds maximum allowed ({MAX_DATA_LINES})"
                )
        super().__setattr__(name, _DevColumns(self, _coerced))
        # Sync column_order to the reassigned key set: preserve existing
        # order for surviving keys, drop removed keys, append new keys in
        # insertion order.  During __init__ column_order is not yet set
        # (declared after columns) — __post_init__ validates the initial
        # pair instead.
        _order = self.__dict__.get("column_order")
        if _order is not None:
            _new_keys = list(_coerced.keys())
            if set(_order) != set(_new_keys):
                _kept = [k for k in _order if k in _coerced]
                _added = [k for k in _new_keys if k not in _order]
                self.column_order = _kept + _added

    def validate(self, complete: bool = False) -> list[str]:
        """Validate DEV file data integrity.

        Args:
            complete: If True, run deferred data-quality checks
                (NaN/Inf in numeric columns, MD monotonicity,
                AZI/INC range validation).

        Returns:
            List of issue strings (empty = no issues found).
            Basic structural checks (key matching, consistent lengths)
            raise during construction and are not duplicated here.
        """
        issues: list[str] = []
        if not complete:
            return issues

        # NaN/Inf check for numeric column arrays.
        for _col_name, _col_data in self.columns.items():
            if isinstance(_col_data, np.ndarray) and _col_data.dtype.kind in ("f", "c"):
                if not np.all(np.isfinite(_col_data)):
                    issues.append(
                        f"DevFile: column '{_col_name}' contains non-finite values (NaN/Inf)."
                    )

        # Data-quality validation (MD monotonicity, AZI/INC range, TVD
        # NaN density / MD-consistency).
        # V-17: DevFile.validate previously had NO survivor handling for
        # ANY column type — a dedup survivor (MD_2, AZI_2, INC_2, TVD_2)
        # bypassed every type-specific check (N-2A).  All checks below
        # match the primary name AND any _N-suffixed survivor, mirroring
        # dev_reader._validate_dev_data's survivor blocks.
        _azi_bases = ("AZI", "AZIM", "AZ", "AZM", "AZIMUTH")
        _inc_bases = ("INC", "INCL", "DEVI", "DIP")

        def _matches_base(_name_upper: str, _bases: tuple[str, ...]) -> bool:
            """Exact primary name OR ``<base>_<digits>`` survivor."""
            if _name_upper in _bases:
                return True
            for _base in _bases:
                if (
                    _name_upper.startswith(_base)
                    and _name_upper[len(_base) :].startswith("_")
                    and _name_upper[len(_base) + 1 :].isdigit()
                ):
                    return True
            return False

        for _col_name, _col_data in self.columns.items():
            if _col_data is None or len(_col_data) == 0:
                continue
            _col_upper = _col_name.upper()
            # MD: monotonicity (primary "MD" or survivor "MD_2"/"MD_3").
            # The exact-match check was the pre-V-17 gap: a dedup survivor
            # from MD+MDKB/DEPTH aliases escaped validation entirely.
            if _col_upper == "MD" or (
                _col_upper.startswith("MD")
                and _col_upper[2:].startswith("_")
                and _col_upper[3:].isdigit()
            ):
                _finite = _col_data[np.isfinite(_col_data)]
                if len(_finite) >= 2:
                    _diffs = np.diff(_finite)
                    if np.any(_diffs < 0):
                        _n_bad = int(np.sum(_diffs < 0))
                        issues.append(
                            f"MD values are not monotonically "
                            f"increasing: {_n_bad} decrease(s) "
                            f"found.  Unsorted MD values can "
                            f"cause inaccurate trajectory "
                            f"calculations."
                        )
            # AZI: range [0, 360]
            if _matches_base(_col_upper, _azi_bases):
                _finite = _col_data[np.isfinite(_col_data)]
                if len(_finite) > 0:
                    _oor = (_finite < 0) | (_finite > 360)
                    if np.any(_oor):
                        _n_bad = int(np.sum(_oor))
                        _bad_vals = _finite[_oor][:3]
                        _extra = "..." if _n_bad > 3 else ""
                        issues.append(
                            f"Azimuth column '{_col_name}' has "
                            f"{_n_bad} value(s) outside [0, 360]: "
                            f"{list(_bad_vals)}{_extra}. "
                            f"Azimuth values outside [0, 360] "
                            f"can cause inaccurate trajectory "
                            f"calculations."
                        )
            # INC: range [0, 180]
            if _matches_base(_col_upper, _inc_bases):
                _finite = _col_data[np.isfinite(_col_data)]
                if len(_finite) > 0:
                    _oor = (_finite < 0) | (_finite > 180)
                    if np.any(_oor):
                        _n_bad = int(np.sum(_oor))
                        _bad_vals = _finite[_oor][:3]
                        _extra = "..." if _n_bad > 3 else ""
                        issues.append(
                            f"Inclination column '{_col_name}' "
                            f"has {_n_bad} value(s) outside "
                            f"[0, 180]: {list(_bad_vals)}{_extra}. "
                            f"Inclination values outside [0, 180] "
                            f"can cause inaccurate trajectory "
                            f"calculations."
                        )
            # TVD: NaN density and MD-consistency.  dev_reader validates
            # TVD (primary and survivors); models.py had NO TVD check at
            # all (V-17 model-side gap) — add it here for parity so direct
            # construction reports the same issues as the read path.
            if _col_upper == "TVD" or (
                _col_upper.startswith("TVD")
                and _col_upper[3:].startswith("_")
                and _col_upper[4:].isdigit()
            ):
                _tvd = _col_data
                _tvd_total = len(_tvd)
                _tvd_nan = int(np.isnan(_tvd).sum())
                if _tvd_total > 0 and _tvd_nan / _tvd_total > 0.5:
                    issues.append(
                        f"TVD column '{_col_name}' has "
                        f"{_tvd_nan}/{_tvd_total} "
                        f"({_tvd_nan / _tvd_total:.1%}) NaN values. "
                        f"Possible delimiter mismatch: data may have "
                        f"been parsed with the wrong separator."
                    )
                # MD-consistency: TVD should not decrease where MD
                # increases (soft check — horizontal sections may keep
                # TVD constant, but backward jumps signal corruption).
                _md_data = self.columns.get("MD")
                if _md_data is not None:
                    _both = ~np.isnan(_tvd) & ~np.isnan(_md_data)
                    if np.sum(_both) >= 2:
                        _md_f = _md_data[_both]
                        _tvd_f = _tvd[_both]
                        _md_inc = np.diff(_md_f) > 0
                        _tvd_dec = np.diff(_tvd_f) < 0
                        _viol = np.logical_and(_md_inc, _tvd_dec)
                        if np.any(_viol):
                            _n_bad = int(np.sum(_viol))
                            issues.append(
                                f"TVD column '{_col_name}' decreases at "
                                f"{_n_bad} station(s) where MD increases. "
                                f"Unexpected TVD reversals may indicate "
                                f"data corruption."
                            )

        return issues

    def __post_init__(self) -> None:
        """Validate critical invariants after construction (E-F-026).

        Direct DevFile construction previously bypassed all validation.
        This catches invalid state that would cause silent data loss
        downstream.  Empty construction is allowed for incremental
        population.
        """
        # F-015: Wrap columns in _DevColumns if not already wrapped.
        # This intercepts `dev.columns["KEY"] = arr` mutations so they
        # go through validation, column_order sync, and length checks.
        # Re-wrapping is idempotent — if columns is already a
        # _DevColumns, the isinstance guard skips re-initialisation.
        if not isinstance(self.columns, _DevColumns):
            # N-I-11: Direct construction must not alias the caller's dict
            # or its arrays.  from_dict deepcopies its input (line ~4103);
            # the direct path previously wrapped the caller's dict BY
            # REFERENCE, so (a) caller-side array mutations corrupted
            # internal data and (b) list-valued columns crashed validate()
            # with a raw TypeError (`_col_data[np.isfinite(_col_data)]`
            # indexes a list with a boolean ndarray).  Deepcopy makes the
            # storage private, and coercing list values to ndarray (matching
            # _DevColumns.__setitem__) makes validate() safe.
            _cols = copy.deepcopy(self.columns)
            _cols = {
                _k: np.atleast_1d(np.asarray(_v, dtype=np.float64)) for _k, _v in _cols.items()
            }
            self.columns = _DevColumns(self, _cols)

        # M-46: Direct construction must not alias the caller's
        # column_order list — caller-side mutation (append, or a silent
        # reverse() reorder) previously corrupted the model with zero
        # warnings.  Deepcopy and wrap through _DevColumnOrder (str-only,
        # no duplicates, entries must reference existing columns).
        # Idempotent: the __setattr__ intercept already wraps every
        # assignment — this defensive branch covers direct __dict__
        # manipulation (same pattern as the columns wrap above).
        if not isinstance(self.column_order, _DevColumnOrder):
            self.column_order = copy.deepcopy(self.column_order)

        if not self.columns:
            return

        from .exceptions import LASDataError

        # column_order must match columns keys exactly
        # I2F-024: Reject duplicate entries in column_order.
        # The set() comparison below cannot detect duplicates
        # (set(["MD","MD","TVD"]) == {"MD","TVD"} is True).
        # Duplicates indicate a caller bug and would cause
        # double-emission downstream.
        if len(self.column_order) != len(set(self.column_order)):
            _dupes = sorted(c for c in set(self.column_order) if self.column_order.count(c) > 1)
            raise LASDataError(
                f"DevFile: column_order contains duplicate entries: "
                f"{_dupes}.  Each column may only appear once."
            )
        _col_keys = set(self.columns.keys())
        _ord_keys = set(self.column_order)
        if _col_keys != _ord_keys:
            raise LASDataError(
                f"DevFile: column_order and columns keys do not "
                f"match.  column_order has {sorted(_ord_keys)}, "
                f"columns has {sorted(_col_keys)}."
            )

        # All columns must have the same array length
        if len(self.columns) > 1:
            _col_lens = {name: len(arr) for name, arr in self.columns.items()}
            if len(set(_col_lens.values())) > 1:
                raise LASDataError(f"DevFile: columns have inconsistent array lengths: {_col_lens}")

        # F-016: Resource-exhaustion guards for direct construction.
        # The from_dict path has MAX_CURVES, MAX_DATA_LINES, and
        # MAX_TOTAL_ELEMENTS guards (imported from .data_reader);
        # __post_init__ previously had none — direct construction with
        # 200+ columns or multi-MB arrays bypassed all limits.
        from .data_reader import MAX_CURVES, MAX_DATA_LINES, MAX_TOTAL_ELEMENTS

        if len(self.columns) >= MAX_CURVES:
            raise LASDataError(
                f"DevFile: number of columns ({len(self.columns)}) "
                f"exceeds maximum allowed ({MAX_CURVES})"
            )

        _max_dev_len = max(len(arr) for arr in self.columns.values())
        if _max_dev_len > MAX_DATA_LINES:
            raise LASDataError(
                f"DevFile: maximum column length ({_max_dev_len}) "
                f"exceeds maximum allowed ({MAX_DATA_LINES})"
            )

        _dev_total = len(self.columns) * _max_dev_len
        if _dev_total > MAX_TOTAL_ELEMENTS:
            raise LASDataError(
                f"DevFile: total elements ({len(self.columns)} columns x "
                f"{_max_dev_len} rows = {_dev_total}) exceeds maximum "
                f"allowed ({MAX_TOTAL_ELEMENTS})"
            )

        # I2F-01: Run data-quality validation checks from __post_init__
        # when constructed directly (not via from_dict).  Matches the
        # pattern used by DataSection and LASFile.  from_dict sets
        # _from_dict=True and calls validate(complete=True) explicitly
        # after full population, avoiding double validation.
        if not self._from_dict:
            for issue in self.validate(complete=True):
                warnings.warn(issue, stacklevel=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to legacy dict format, including file metadata.

        Metadata keys (source_file, encoding, column_order) are stored in
        the flat dict alongside column data.  When a column name collides
        with a metadata key, the metadata is stored under a ``_meta_``
        prefix to avoid silently overwriting column data (I2F-28).
        """
        result: dict[str, Any] = {k: v.copy() for k, v in self.columns.items()}
        # I2F-28: Detect column name collisions with metadata keys.
        # Without this check, a column named "source_file", "encoding", or
        # "column_order" is silently overwritten by metadata — and on
        # roundtrip through read_dev_file() the key is stripped entirely
        # (dev_reader.py:522-524), producing double data loss.
        _metadata_assignments = {
            "source_file": self.source_file,
            "encoding": self.encoding,
            "column_order": list(self.column_order),
        }
        for mk, mv in _metadata_assignments.items():
            _meta_key = f"_meta_{mk}"
            if mk in self.columns:
                if _meta_key in self.columns:
                    # PF-07: DOUBLE collision — a user column literally
                    # named ``_meta_{mk}`` occupies the disambiguated key
                    # AND a column named ``{mk}`` occupies the bare key.
                    # The flat dict format cannot represent the metadata
                    # without overwriting one of the user columns — preserve
                    # the columns and drop the metadata (loudly).
                    warnings.warn(
                        f"DevFile column names '{mk}' and '{_meta_key}' both "
                        f"collide with metadata key — metadata '{mk}' cannot "
                        f"be stored in to_dict without overwriting a column; "
                        f"metadata dropped, user columns preserved unchanged.",
                        stacklevel=2,
                    )
                    continue
                warnings.warn(
                    f"DevFile column name '{mk}' collides with metadata key — "
                    f"storing metadata as '_meta_{mk}' to avoid data loss. "
                    f"Column data is preserved unchanged.",
                    stacklevel=2,
                )
                result[_meta_key] = mv
            else:
                # No bare-name collision.  Emit the metadata under the bare
                # key.  PF-07: when a user column named ``_meta_{mk}`` exists
                # (array value), from_dict still reads the bare key as
                # metadata because the ``_meta_`` slot is NOT metadata-shaped
                # (MOD-11 closed-set rule) — column and metadata roundtrip
                # together without ambiguity.
                result[mk] = mv
        return result

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        normalize_aliases: bool = True,
    ) -> DevFile:
        """Create DevFile from dict (reverse of to_dict).

        Args:
            data: Flat dict mapping column names to array-like values.
            normalize_aliases: If True (default), normalize column names
                through ``_DEV_ALIASES``.  Set to False to preserve
                column names exactly as provided.

        Returns:
            DevFile with columns populated from the dict.
        """
        # F-008: isinstance check moved inside try block so TypeError
        # gets wrapped in LASDataError, matching LASFile.from_dict's
        # documented contract.
        from .data_reader import MAX_CURVES, MAX_DATA_LINES, MAX_TOTAL_ELEMENTS
        from .exceptions import LASDataError

        try:
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict, got {type(data).__name__}")
            # F2-12: Protect caller's dict from mutation.
            data = copy.deepcopy(data)

            # I2F-01: Suppress double validation — __post_init__ gates
            # validate(complete=True) behind _from_dict=False.  from_dict
            # calls validate() explicitly after full population.
            dev = cls(_from_dict=True)
            # Separate column arrays from metadata keys.  MOD-11: the bare
            # metadata keys are a CLOSED set (_DEV_META_KEYS); the
            # ``_meta_``-prefixed keys are only metadata when carrying a
            # metadata-shaped value (see _is_encoded_dev_metadata_key).
            metadata_keys = _DEV_META_KEYS

            # G-007/G-016: Validate all dict keys are strings before
            # .startswith() is called in the list comprehension below.
            # Non-str keys (int, tuple) raise AttributeError which
            # escapes the except (ValueError, TypeError) wrapper.
            for _k in data:
                if not isinstance(_k, str):
                    raise LASDataError(
                        f"DevFile.from_dict: column keys must be strings, got {type(_k).__name__}"
                    )
            # F-20: Normalize column names through _DEV_ALIASES before
            # storage.  Controlled by normalize_aliases parameter.
            if normalize_aliases:
                from .dev_reader import _normalize_dev_column

                # I2F-011: Build a normalization map first, then
                # rebuild the data dict in original key order so
                # renamed columns stay in their original positions.
                # The old pop+append pattern (data[norm]=data.pop(raw))
                # moved renamed columns to the dict end, corrupting
                # column_order inference.
                _norm_map: dict[str, str] = {}
                for _raw_key in data:
                    # F-M-G03: Don't normalize metadata keys — they are
                    # checked case-sensitively against metadata_keys below.
                    # Normalizing e.g. "encoding" → "ENCODING" would cause
                    # the metadata check to fail and the value to be
                    # treated as column data.
                    if _raw_key in metadata_keys:
                        continue
                    # Also skip well-known _meta_ prefixed keys —
                    # normalization uppercases them (e.g. _meta_source_file
                    # → _META_SOURCE_FILE) and the _meta_ handling below is
                    # case-sensitive.  MOD-11: only the KNOWN ``_meta_<known>``
                    # suffixes are reserved; unknown ``_meta_*`` keys are
                    # ordinary user columns and are normalized like any other
                    # column name.
                    if _raw_key.startswith("_meta_") and _raw_key[6:] in _DEV_META_KEYS:
                        continue
                    _norm_key = _normalize_dev_column(_raw_key)
                    if _norm_key != _raw_key:
                        # Collision: two raw keys normalise to the same value
                        if _norm_key in _norm_map.values():
                            raise ValueError(
                                f"DevFile.from_dict: column name collision: "
                                f"'{_raw_key}' and another key normalise "
                                f"to the same canonical column name '{_norm_key}'"
                            )
                        # Collision: normalised key already exists as a
                        # raw key that itself does not normalise away
                        if _norm_key in data and _norm_key not in _norm_map:
                            raise ValueError(
                                f"DevFile.from_dict: column name collision: "
                                f"'{_raw_key}' and '{_norm_key}' normalize "
                                f"to the same canonical column name"
                            )
                        _norm_map[_raw_key] = _norm_key
                # Rebuild data dict with normalized keys, preserving
                # original insertion order (the order of the caller's
                # input dict).
                if _norm_map:
                    data = {_norm_map.get(k, k): v for k, v in data.items()}

            # F-M01: Resource-exhaustion guard — bound column count.
            # MOD-11: only metadata-shaped ``_meta_<known>`` keys are
            # excluded from the count (they are not columns); unknown
            # ``_meta_*`` keys and array-valued ``_meta_<known>`` keys ARE
            # columns and must count.
            # R7F-01-gap + PF-07: When both a bare metadata key (e.g.
            # "source_file") and its _meta_-prefixed counterpart exist, the
            # bare key is column data ONLY when the _meta_ key carries
            # ENCODED METADATA (to_dict stored it there because of a
            # bare-name collision).  A _meta_ key with an array value is a
            # USER COLUMN, so the bare key is metadata and must not count.
            _column_keys = [
                k
                for k in data
                if not _is_encoded_dev_metadata_key(k, data[k])
                and (
                    k not in metadata_keys
                    or (
                        f"_meta_{k}" in data
                        and _is_encoded_dev_metadata_key(f"_meta_{k}", data[f"_meta_{k}"])
                    )
                )
            ]
            if len(_column_keys) >= MAX_CURVES:
                raise ValueError(
                    f"Number of columns ({len(_column_keys)}) exceeds maximum "
                    f"allowed ({MAX_CURVES})"
                )

            for key, value in data.items():
                # R7F-01 + MOD-11: _meta_ prefix roundtrip.  When to_dict
                # detects a column name collision with a metadata key
                # (I2F-28), it stores metadata under ``_meta_``-prefixed
                # keys (e.g., ``_meta_source_file``).  from_dict must
                # recognise and reverse that prefix so the roundtrip
                # contract is preserved — but ONLY for the closed set of
                # well-known metadata keys carrying a metadata-shaped
                # value.  Any other ``_meta_*`` key (unknown suffix, or a
                # known suffix with an array value) is a USER COLUMN and
                # is stored verbatim under its literal ``_meta_...`` name
                # — the previous opaque-prefix design silently dropped
                # such columns (MOD-11).
                if _is_encoded_dev_metadata_key(key, value):
                    real_key = key[6:]  # strip ``_meta_`` prefix
                    if real_key == "encoding":
                        dev.encoding = _safe_str(value, "utf-8")
                    elif real_key == "source_file":
                        dev.source_file = _safe_str(value)
                    elif real_key == "column_order":
                        if value is None:
                            dev.column_order = []
                        elif isinstance(value, (str, bytes)):
                            # F-110: bytes bypasses the str guard in Python 3.
                            if isinstance(value, bytes):
                                raise TypeError(
                                    "column_order must be a list of strings, "
                                    "got bytes.  Decode to str first: "
                                    "value.decode('utf-8')"
                                )
                            # F2-31: Normalize column_order entries to match
                            # normalized column names in dev.columns.
                            if normalize_aliases:
                                # I2F-011: map entries back to the actual
                                # case-sensitive column keys BEFORE storing
                                # so _DevColumnOrder's existence check passes
                                # for metadata-collision columns (e.g. column
                                # 'source_file' vs normalized 'SOURCE_FILE').
                                _col_order = [_normalize_dev_column(value)]
                                _actual_keys = {k.upper(): k for k in dev.columns}
                                dev.column_order = [
                                    _actual_keys.get(c.upper(), c) for c in _col_order
                                ]
                            else:
                                dev.column_order = [value]
                        else:
                            _col_order = list(value)
                            # F2-31: Normalize column_order entries.
                            if normalize_aliases:
                                _col_order = [_normalize_dev_column(c) for c in _col_order]
                                # I2F-011: map entries back to the actual
                                # case-sensitive column keys BEFORE storing
                                # so _DevColumnOrder's existence check passes
                                # for metadata-collision columns (e.g. column
                                # 'source_file' vs normalized 'SOURCE_FILE').
                                _actual_keys = {k.upper(): k for k in dev.columns}
                                _col_order = [_actual_keys.get(c.upper(), c) for c in _col_order]
                            if len(_col_order) >= MAX_CURVES:
                                raise ValueError(
                                    f"column_order has {len(_col_order)} entries, "
                                    f"maximum allowed is {MAX_CURVES - 1}."
                                )
                            dev.column_order = _col_order
                    # Known metadata key — handled above, skip column
                    # processing.
                    continue
                # R7F-01-gap + PF-07: When to_dict detects a column-name
                # collision with a metadata key (e.g. column named
                # "source_file"), it stores both the bare key (column data)
                # and a _meta_-prefixed key (real metadata).  The bare key is
                # column data ONLY when the _meta_ slot carries ENCODED
                # METADATA — a _meta_ key with an array value is a USER
                # COLUMN (MOD-11), so the bare key is metadata.  Also require
                # the bare value to be metadata-shaped: an array under a bare
                # metadata key is a column (PF-07 double-collision where
                # to_dict dropped the metadata to preserve both columns).
                _is_collision = f"_meta_{key}" in data and _is_encoded_dev_metadata_key(
                    f"_meta_{key}", data[f"_meta_{key}"]
                )
                if (
                    key in metadata_keys
                    and not _is_collision
                    and _is_dev_metadata_shaped(key, value)
                ):
                    if key == "encoding":
                        dev.encoding = _safe_str(value, "utf-8")
                    elif key == "source_file":
                        dev.source_file = _safe_str(value)
                    elif key == "column_order":
                        if value is None:
                            dev.column_order = []
                        elif isinstance(value, (str, bytes)):
                            # F-110: bytes bypasses the str guard in Python 3.
                            if isinstance(value, bytes):
                                raise TypeError(
                                    "column_order must be a list of strings, "
                                    "got bytes.  Decode to str first: "
                                    "value.decode('utf-8')"
                                )
                            # F2-31: Normalize column_order entries to match
                            # normalized column names in dev.columns.
                            if normalize_aliases:
                                # I2F-011: map entries back to the actual
                                # case-sensitive column keys BEFORE storing
                                # so _DevColumnOrder's existence check passes
                                # for metadata-collision columns (e.g. column
                                # 'source_file' vs normalized 'SOURCE_FILE').
                                _col_order = [_normalize_dev_column(value)]
                                _actual_keys = {k.upper(): k for k in dev.columns}
                                dev.column_order = [
                                    _actual_keys.get(c.upper(), c) for c in _col_order
                                ]
                            else:
                                dev.column_order = [value]
                        else:
                            _col_order = list(value)
                            # F2-31: Normalize column_order entries.
                            if normalize_aliases:
                                _col_order = [_normalize_dev_column(c) for c in _col_order]
                                # I2F-011: map entries back to the actual
                                # case-sensitive column keys BEFORE storing
                                # so _DevColumnOrder's existence check passes
                                # for metadata-collision columns (e.g. column
                                # 'source_file' vs normalized 'SOURCE_FILE').
                                _actual_keys = {k.upper(): k for k in dev.columns}
                                _col_order = [_actual_keys.get(c.upper(), c) for c in _col_order]
                            if len(_col_order) >= MAX_CURVES:
                                raise ValueError(
                                    f"column_order has {len(_col_order)} entries, "
                                    f"maximum allowed is {MAX_CURVES - 1}."
                                )
                            dev.column_order = _col_order
                else:
                    # F-I2-M02: None guard — np.array(None, dtype=np.float64)
                    # silently produces nan, consistent with string data guards.
                    if value is None:
                        raise ValueError(f"Numeric data for column '{key}' is None")
                    # I2F-14: Pre-allocation size check.  np.array() allocates
                    # BEFORE the downstream len() guard — a huge list triggers
                    # MemoryError before the guard catches it.  Check len()
                    # first when the input supports it (matches ds_data pattern
                    # at lines 1057-1063).
                    if hasattr(value, "__len__") and len(value) > MAX_DATA_LINES:
                        raise ValueError(
                            f"Column '{key}' length ({len(value)}) exceeds "
                            f"maximum allowed ({MAX_DATA_LINES})"
                        )
                    try:
                        dev.columns[key] = np.atleast_1d(np.asarray(value, dtype=np.float64))
                    except (ValueError, TypeError, MemoryError, OverflowError) as e:
                        raise ValueError(
                            f"Cannot convert data for column '{key}' to numeric array: {e}"
                        ) from e
                    # F-M01: Per-array size guard for DevFile columns.
                    if len(dev.columns[key]) > MAX_DATA_LINES:
                        raise ValueError(
                            f"Column '{key}' length ({len(dev.columns[key])}) exceeds "
                            f"maximum allowed ({MAX_DATA_LINES})"
                        )

            # F-M01: Total element count guard across all columns.
            if dev.columns:
                _dev_rows = max(len(arr) for arr in dev.columns.values())
                _dev_total = len(dev.columns) * _dev_rows
                if _dev_total > MAX_TOTAL_ELEMENTS:
                    raise ValueError(
                        f"Total allocation ({len(dev.columns)} columns x "
                        f"{_dev_rows} rows = {_dev_total} elements) exceeds "
                        f"maximum allowed ({MAX_TOTAL_ELEMENTS})"
                    )

            # F-006: Cross-array length validation for DevFile columns.
            # Inconsistent-length columns produce silently corrupted output
            # if accepted — different columns with different row counts
            # represent invalid DEV survey data.
            if len(dev.columns) > 1:
                _col_len = {name: len(arr) for name, arr in dev.columns.items()}
                if len(set(_col_len.values())) > 1:
                    raise ValueError(f"DevFile columns have inconsistent lengths: {_col_len}")

            # If column_order wasn't in the dict, infer from Python 3.7+ dict order
            if not dev.column_order:
                dev.column_order = list(dev.columns.keys())
            # F2-30: Reject empty DevFile — no columns found.
            if not dev.columns:
                raise LASDataError("No columns found in input dict")
            # I2F-011: Sync column_order entries to match actual
            # case-sensitive keys in dev.columns.  The from_dict
            # normalization loop uppercases column_order entries via
            # _normalize_dev_column, but metadata-key columns (e.g.
            # "encoding" when a collision produces _meta_encoding) are
            # stored with their original case.  Map each column_order
            # entry back to the matching dev.columns key.
            if dev.column_order and normalize_aliases:
                _actual_keys = {k.upper(): k for k in dev.columns}
                dev.column_order = [_actual_keys.get(c.upper(), c) for c in dev.column_order]
            # F2-32: Cross-validate column_order entries against
            # columns.keys().  Orphaned entries (e.g. ["INC", "AZI"]
            # when only "MD" exists) produce silently broken output.
            _orphaned = [c for c in dev.column_order if c not in dev.columns]
            if _orphaned:
                raise LASDataError(f"column_order contains entries not in columns: {_orphaned}")
            # F-15: Re-invoke __post_init__ after full population.
            # During `dev = cls(_from_dict=True)`, __post_init__ returned
            # early (columns was empty).  Now that all columns are
            # populated, re-run the structural checks to validate
            # column_order/columns consistency and consistent lengths.
            dev.__post_init__()
            # F-43: Minimum data-quality validation matching reader's
            # _validate_dev_data checks.  validate(complete=True) covers
            # NaN/Inf, MD monotonicity, AZI/INC range; _validate_dev_data
            # additionally covers negative MD, NaN density, repeated
            # stations, TVD, and dedup survivors (F-012).
            for issue in dev.validate(complete=True):
                warnings.warn(issue, stacklevel=2)
            from .dev_reader import _validate_dev_data

            _validate_dev_data(dev, _stacklevel=2)

            return dev
        except (ValueError, TypeError, OverflowError) as e:
            raise LASDataError(str(e)) from e
