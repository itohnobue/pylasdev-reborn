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
    def mandatory_well_field_groups(self) -> tuple[tuple[str, ...], ...]:
        """Mandatory well-section field SLOTS, as alternative-spelling groups.

        A slot is satisfied when ANY of its members is present in the well
        section.  This is the authoritative CWLS rule (LAS 2.0 "Well
        Information" requires twelve entries, two of which the standard
        accepts under alternative mnemonics):

        STRT, STOP, STEP, NULL, COMP, WELL, FLD, LOC,
        ``PROV|CNTY|STAT|CTRY``, SRVC, DATE, ``UWI|API``.

        F-024: the previous model was a flat tuple (``mandatory_well_fields``)
        that could express neither alternative, so the ``PROV`` family was
        enforced nowhere and a LAS 2.0 file with no province/county/state/
        country field drew no diagnostic at all.  ``UWI`` must NOT become a
        flat requirement — M-11 documents it as optional-when-present, and
        the non-2.0 rule below keeps the four numeric fields, so a common
        LAS 1.2 file with no UWI does not warn.  For LAS 2.0 the ``UWI|API``
        group is a SLOT: a file carrying either spelling satisfies it with no
        warning, and a file carrying NEITHER warns like the rest of the CWLS
        set — the M-11 no-UWI-warn decision is superseded for LAS 2.0 by this
        group.  (M11: the LAS 1.2 M-11 test is not a witness for this group —
        it exercises the 1.2 rule only, so what it pins is the 1.2 branch
        below, not the 2.0 slot.)

        LAS 1.2 and LAS 3.0 keep the four numeric depth-control fields
        (I2-07: the 2.0-era lascheck set is not a 1.2 requirement; the 3.0
        Well section inherits the same four).

        The flat :attr:`mandatory_well_fields` property is the legacy view.
        It cannot express the CWLS alternative spellings, and after D4b no
        in-tree consumer reads it — ``models.py`` and ``parser.py`` consume
        THIS property — so validation must use this one.
        """
        if self.is_las20:
            return (
                ("STRT",),
                ("STOP",),
                ("STEP",),
                ("NULL",),
                ("COMP",),
                ("WELL",),
                ("FLD",),
                ("LOC",),
                ("PROV", "CNTY", "STAT", "CTRY"),
                ("SRVC",),
                ("DATE",),
                ("UWI", "API"),
            )
        return (
            ("STRT",),
            ("STOP",),
            ("STEP",),
            ("NULL",),
        )

    @property
    def mandatory_well_fields(self) -> tuple[str, ...]:
        """Mandatory well-section field mnemonics for the current version.

        LAS 1.2 requires the four numeric depth-control fields
        (STRT, STOP, STEP, NULL).  The lascheck 10-field set
        (STRT, STOP, STEP, NULL, COMP, WELL, FLD, LOC, SRVC, DATE) is a
        LAS 2.0-era requirement — lascheck's own documentation states it
        "currently supports checking against LAS 2.0 standard only"
        (research s1-research-las12-report.md:18).  frackoptima and GERDA
        require only the 4 numeric fields for LAS 1.2 (research:20), so
        applying the 10-field set to 1.2 produced 6 spurious "missing
        mandatory well field" warnings on minimal-but-valid 1.2 files
        (I2-07).
        LAS 3.0 inherits the LAS 2.0 mandatory-well-field requirements per
        the spec (STRT, STOP, STEP, NULL).

        M-11: UWI is NOT in lascheck's required set (it is optional when
        present) — previously LAS 1.2 wrongly required it, producing a
        false-positive "missing: UWI" warning on common files that carry no
        UWI.  COMP/FLD/DATE were previously omitted, producing false
        negatives for LAS 2.0.  This note describes the FLAT tuple only; the
        shipped LAS 2.0 rule is the group-aware one above, where ``UWI`` or
        ``API`` IS required.

        F-024: this FLAT view is retained for compatibility only.  After D4b
        every in-tree consumer (``models.py``, ``parser.py``) reads
        :attr:`mandatory_well_field_groups`, so no library code uses this
        tuple, and it cannot express the CWLS alternative spellings
        (``UWI|API``, ``PROV|CNTY|STAT|CTRY``).  Validation must consume
        :attr:`mandatory_well_field_groups`.
        """
        if self.is_las12:
            return ("STRT", "STOP", "STEP", "NULL")
        if self.is_las20:
            return (
                "STRT",
                "STOP",
                "STEP",
                "NULL",
                "COMP",
                "WELL",
                "FLD",
                "LOC",
                "SRVC",
                "DATE",
            )
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
