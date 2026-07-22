"""Centralized LAS version specification rules.

Consolidates scattered version-check patterns (``startswith("1.")``,
``startswith("2.")``, ``is_las30``, etc.) into a single frozen dataclass
so that LAS version-specific rules have one authoritative source.

All version-dependent logic across parser.py, writer.py, and models.py
delegates to the properties and methods on this class.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _LASVersionSpec:
    """Single source of truth for LAS version-specific rules.

    Constructed from a normalized version string (e.g. ``"1.2"``, ``"2.0"``,
    ``"3.0"``).  All boolean properties are derived from *vers*, and semantic
    rules (line-length limits, delimiter restrictions, section structure)
    are expressed as named properties instead of inline prefix checks.
    """

    vers: str
    """Normalized version string (e.g. ``"1.2"``, ``"2.0"``, ``"3.0"``)."""

    # --- identity checks ---------------------------------------------------

    @property
    def is_las12(self) -> bool:
        """True when *vers* starts with ``"1."`` (LAS 1.2 or 1.x variant)."""
        return self.vers.startswith("1.")

    @property
    def is_las20(self) -> bool:
        """True when *vers* starts with ``"2."`` (LAS 2.0)."""
        return self.vers.startswith("2.")

    @property
    def is_las30(self) -> bool:
        """True when *vers* starts with ``"3"`` (LAS 3.0 or 3.x variant).

        Uses ``"3"`` prefix (no dot) to match the existing ``VersionSection.is_las30``
        behaviour, which accepts ``"3.0"``, ``"3.1"``, ``"3.1beta"``, etc.
        """
        return self.vers.startswith("3")

    @property
    def is_las12_or_later(self) -> bool:
        """True for any recognised LAS version string (``"1."``, ``"2."``, or ``"3."``).

        Replaces ``self.las_file.version.vers.startswith(("1.", "2.", "3."))``
        used for mandatory well-field validation in the parser.
        """
        return self.is_las12 or self.is_las20 or self.is_las30

    # --- delimiter rules ---------------------------------------------------
    # Deleted: allows_non_space_dlm, emits_dlm_line (F-033: zero callers)

    # --- wrap / line-length rules ------------------------------------------
    # Deleted: supports_wrap (F-033: zero callers)

    def line_length_limit_for_wrap(self, wrap: str | None) -> int | None:
        """Return the line-length limit for the given WRAP mode, or None.

        LAS 1.2: always 256 chars.
        LAS 2.0: 256 chars when WRAP=NO; unlimited for WRAP=YES.
        LAS 3.0: no explicit limit (return None).

        Replaces the two-phase ``check_line_limit`` computation in
        ``writer._write_ascii_sections``.
        """
        if self.is_las12:
            return 256
        if self.is_las20 and (wrap or "NO").upper() == "NO":
            return 256
        return None

    # --- well-section rules ------------------------------------------------
    # Deleted: well_format_swap_value_desc (F-033: zero callers)

    @property
    def mandatory_well_fields(self) -> tuple[str, ...]:
        """Mandatory well-section field mnemonics for the current version.

        LAS 1.2, 2.0, and 3.0 all require STRT, STOP, STEP, NULL.
        LAS 1.2 additionally requires WELL, LOC, SRVC, UWI (soft-required).
        """
        if self.is_las12:
            return ("STRT", "STOP", "STEP", "NULL", "WELL", "LOC", "SRVC", "UWI")
        return ("STRT", "STOP", "STEP", "NULL")

    # --- section structure rules -------------------------------------------
    # Deleted: has_structured_sections, supports_data_sections,
    # supports_other_section, supports_data_format, supports_zone_notation
    # (F-033: zero external callers across src/ and tests/)

    # --- index curve rules -------------------------------------------------
    # Deleted: requires_index_curve (F-033: zero callers)

    # --- section duplication rules -----------------------------------------
    # Deleted: duplicate_allowed_sections, single_section_types
    # (F-033: 12 dead properties — zero external callers across src/ and tests/.
    # Version-specific decisions use identity properties is_las12/is_las20/is_las30.)
