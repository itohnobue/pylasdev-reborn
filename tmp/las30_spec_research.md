# LAS 3.0 Specification Research Report

## Executive Summary

LAS 3.0 (Log ASCII Standard Version 3.0) is a major redesign of the LAS format, released in June 2000 by the Canadian Well Logging Society (CWLS). It maintains backward compatibility principles while significantly expanding capabilities to handle diverse wellbore data types beyond traditional log data.

---

## 1. Section Structure Differences

### LAS 2.0 vs LAS 3.0 Section Comparison

| Feature | LAS 2.0 | LAS 3.0 |
|---------|---------|---------|
| Required Sections | ~V, ~W, ~C, ~P, ~A | ~Version, ~Well + at least one data set |
| Section Naming | Single character (~V, ~W, ~C, ~P, ~A) | Full names with suffixes |
| Multiple Sections | Not allowed | Allowed with [n] index suffixes |
| ~Other Section | Allowed | **DROPPED** |
| WRAP YES | Allowed | **DROPPED** - Only WRAP NO valid |

### LAS 3.0 Section Types

**Required Sections (must be first two, in order):**
1. `~Version` - File metadata and LAS version info
2. `~Well` - Well identification and location

**Data Section Sets (optional, but at least one required):**

Each data type follows a 3-section pattern: `_Parameter` (optional), `_Definition`, `_Data`

| Data Type | Parameter Section | Definition Section | Data Section |
|-----------|-------------------|-------------------|--------------|
| **Log Data** | `~Parameter` or `~Log_Parameter` | `~Curve` or `~Log_Definition` | `~ASCII` or `~Log_Data` |
| **Core Data** | `~Core_Parameter` | `~Core_Definition` | `~Core_Data` |
| **Inclinometry** | `~Inclinometry_Parameter` | `~Inclinometry_Definition` | `~Inclinometry_Data` |
| **Drilling** | `~Drilling_Parameter` | `~Drilling_Definition` | `~Drilling_Data` |
| **Tops** | `~Tops_Parameter` | `~Tops_Definition` | `~Tops_Data` |
| **Test Data** | `~Test_Parameter` | `~Test_Definition` | `~Test_Data` |
| **User Defined** | `~User_Parameter` | `~User_Definition` | `~User_Data` |

### Key Structural Changes

1. **Section Title Arrays**: Multiple sections of same type allowed with `[n]` suffix
   ```
   ~Log_Data[1] | Log_Definition
   ~Log_Data[2] | Log_Definition
   ```

2. **Data Section Association**: Column Data sections MUST reference their Definition section
   ```
   ~Core_Data | Core_Definition
   ```

3. **User-Defined Sections**: Can create custom sections following the naming pattern

---

## 2. Mandatory and Optional Fields

### ~Version Section (Required)

| Mnemonic | Required | Value | Description |
|----------|----------|-------|-------------|
| VERS | **Yes** | 3.0 | LAS Version Identifier |
| WRAP | **Yes** | NO | Only NO is valid in LAS 3.0 |
| DLM | **Yes** | SPACE/COMMA/TAB | Column delimiter (SPACE default) |

### ~Well Section (Required)

**Always Required:**
| Mnemonic | Required | Value Must Contain Data |
|----------|----------|------------------------|
| STRT | **Yes** | **Yes** | First Index Value |
| STOP | **Yes** | **Yes** | Last Index Value |
| STEP | **Yes** | **Yes** | Step increment (0 if irregular) |
| NULL | **Yes** | **Yes** | NULL value (-999.25 common) |
| COMP | **Yes** | No | Company |
| WELL | **Yes** | No | Well name |
| FLD | **Yes** | No | Field |
| LOC | **Yes** | No | Location |
| SRVC | **Yes** | No | Service Company |
| CTRY | **Yes** | No | Country (Internet country code) |
| DATE | **Yes** | No | Service Date |

**Location Coordinates (one complete set required):**
- Either: LATI, LONG, GDAT
- Or: X, Y, GDAT, HZCS

**Country-Specific Required Fields:**

For Canada (CTRY = ca):
- PROV, UWI, LIC

For USA (CTRY = us):
- STAT, CNTY, API

### Log Data Sections

**~Parameter/~Log_Parameter (Required for log data):**
- RUNS, RUN[n], RUN_DEPTH[n] - for multiple runs
- PDAT, APD, DREF, EREF, RUN - run-related parameters

**~Curve/~Log_Definition (Required for log data):**
- DEPT must be first channel (index)
- Each channel: MNEM.UNIT LogCode : Description {Format} | Association

**~ASCII/~Log_Data (Required for log data):**
- First column must be index (depth)
- Data delimited per DLM value

---

## 3. Data Format Changes

### New Data Formats in LAS 3.0

LAS 2.0 only supported floating-point numbers. LAS 3.0 adds:

| Format | Syntax | Example |
|--------|--------|---------|
| **Floating Point** | `{Fx.y}` or `{F}` | `{F10.4}`, `{F}` |
| **Integer** | `{Ix}` or `{I}` | `{I5}`, `{I}` |
| **String** | `{Sx}` or `{S}` | `{S32}`, `{S}` |
| **Exponential** | `{E0.00E+00}` | `{E0.00E+00}` |
| **Date** | `{DD/MM/YYYY}` | `{DD/MM/YYYY}` |
| **Time** | `{hh:mm:ss}` | `{hh:mm}` |
| **Degrees** | `{F}` with deg/min/sec | `23.45 34.06' 58.19"` |
| **Array** | `{A...}` | `{AF10.4;5ms}` |

### Date Format Options

```
D     - Single digit day (1-9) or 2-digit (10-31)
DD    - Two digit day (01-31)
M     - Single digit month (1-9) or 2-digit (10-12)
MM    - Two digit month (01-12)
MMM   - Three character month (JAN, FEB, etc.)
MMMM  - Full month name
YYYY  - Four digit year (required)
```

Delimiters: `-` (dash) or `/` (forward slash)

### Delimiter Options

| DLM Value | ASCII Code | Character |
|-----------|------------|-----------|
| SPACE | 32 | Space (consecutive = single delimiter) |
| COMMA | 44 | , |
| TAB | 9 | Tab |

### 3D Array Data Support

LAS 3.0 supports three-dimensional arrays using indexed channels:

```
NMR[1] .ms : NMR Echo Array {AF;0ms}
NMR[2] .ms : NMR Echo Array {AF;5ms}
NMR[3] .ms : NMR Echo Array {AF;10ms}
NMR[4] .ms : NMR Echo Array {AF;15ms}
NMR[5] .ms : NMR Echo Array {AF;20ms}
```

Format: `{A[format];index1;index2...}`
- `A` - Array indicator
- `format` - Optional (F, I, S, E)
- `;index` - Spacing/index values (e.g., `5ms`, `10ft`)

---

## 4. Example LAS 3.0 File Structure

```
~VERSION INFORMATION
 VERS.                          3.0 : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.                           NO : ONE LINE PER DEPTH STEP
 DLM .                        COMMA : DELIMITING CHARACTER BETWEEN DATA COLUMNS

~Well Information
 STRT .M              1670.0000    : First Index Value
 STOP .M               713.2500    : Last Index Value
 STEP .M              -0.1250      : STEP
 NULL .               -999.25      : NULL VALUE
 COMP .       ANY OIL COMPANY INC. : COMPANY
 WELL .       ANY ET AL 12-34-12-34: WELL
 FLD  .       WILDCAT              : FIELD
 LOC  .       12-34-12-34W5M       : LOCATION
 PROV .       ALBERTA              : PROVINCE
 SRVC .       ANY LOGGING COMPANY INC. : SERVICE COMPANY
 DATE .       13/12/1986           : LOG DATE  {DD/MM/YYYY}
 UWI  .       100123401234W500     : UNIQUE WELL ID
 API  .       12345678             : API NUMBER
 LAT .DEG                     34.56789 : Latitude  {DEG}
 LONG.DEG                   -102.34567 : Longitude  {DEG}

~CURVE INFORMATION
 DEPT .M                            : DEPTH               {F}
 DT   .US/M           123 456 789   : SONIC TRANSIT TIME  {F}
 RHOB .K/M3           123 456 789   : BULK DENSITY        {F}
 NPHI .V/V            123 456 789   : NEUTRON POROSITY    {F}
 CDES .               123 456 789   : CORE DESCRIPTION    {S}
 NMR[1] .ms           123 456 789   : NMR Echo Array      {A:0 }
 NMR[2] .ms           123 456 789   : NMR Echo Array      {A:5 }

~PARAMETER INFORMATION
 RUNS.  2              : of Runs for this well.
 RUN[1].            1  : Run 1
 RUN[2].            2  : Run 2

 #Parameters that are zoned.
 NMAT_Depth[1].M  500,1500     : Neutron Matrix Depth interval {F}
 NMAT_Depth[2].M  1500,2500    : Neutron Matrix Depth interval {F}

 MATR .            SAND : Neutron Porosity Matrix          |  NMAT_Depth[1]
 MATR .            LIME : Neutron Porosity Matrix          |  NMAT_Depth[2]

~Core_Parameter
 C_SRS . : Core Source {S}
 C_TY  . : Core Type {S}
 C_DT  . : Recovery Date {DD/MM/YYYY}
 C_TP  .M : Core Top Depth {F}
 C_BS  .M : Core Base Depth {F}

~Core_Definition
 CORT .M  : Core top depth {F}
 CORB .M  : Core Bottom Depth {F}
 PERM .md : Core permeability {F}
 CPOR .%  : Core porosity {F}

~Core_Data | Core_Definition
 13178.00,13179.00,5.00,17.70
 13180.00,13181.00,430.00,28.70

~Drilling_Parameter
 RIG . BIG RIG : Drilling Rig name
 CONTR . DLR DRILLING : Contractor

~Drilling_Definition
 DDEP .ft    : Depth {F}
 ROP  .ft/hr : Rate of Penetration {F}
 WOB  .klb   : Weight on bit {F}

~Drilling_Data | Drilling_Definition
 322.02,24.0,3.0
 323.05,37.5,2.0

~ASCII | Curve
 1670.000,123.45,2345.6,0.15,DOLOMITE,10.0,12.0
 1669.875,124.56,2356.7,0.16,LIMESTONE,12.0,15.0
```

---

## 5. Key Implementation Notes

### Breaking Changes from LAS 2.0

1. **WRAP YES Removed**: Only single-line-per-depth supported
2. **~Other Section Removed**: Must use proper Parameter or Data sections
3. **DLM Parameter Required**: Must specify delimiter type
4. **Data Section Association Required**: `~Data | Definition` format

### New Features to Implement

1. **Multiple Data Types**: Core, Inclinometry, Drilling, Tops, Test sections
2. **Section Indexing**: `[n]` suffix for multiple sections
3. **Parameter Associations**: Link parameters to runs, zones, or other parameters
4. **Parameter Zoning**: Different parameter values over depth intervals
5. **3D Array Channels**: Multi-element channels like NMR echo arrays

### Association Syntax

```
MNEM.UNIT VALUE : DESCRIPTION {Format} | Association1,Association2
```

Examples:
```
BS .MM 8.75 : Bit Size | RUN[1]
DPHI .V/V : Density Porosity | MDEN[1],MDEN[2]
MATR . SAND : Neutron Matrix | NMAT_Depth[1]
```

### Parsing Considerations

1. **Line Parsing**:
   - First `.` (period) = end of mnemonic, start of unit
   - First space after period = end of unit
   - Last `:` (colon) = start of description
   - `{}` braces = format specification
   - `|` (bar) = start of associations

2. **Data Values**:
   - Can contain multiple items separated by DLM character
   - Items with delimiter chars must be quoted: `"value, with comma"`
   - Consecutive delimiters (except SPACE) = NULL values

3. **NULL Value Handling**:
   ```
   1000.00,,46.0985,,,
   ```
   Equivalent to:
   ```
   1000.00,-999.25,46.0985,-999.25,-999.25,-999.25
   ```

4. **Index Channel Rules**:
   - Must be first column
   - Cannot be empty/NULL
   - Must be continuously increasing or decreasing

### Character Restrictions

**Reserved Delimiters:**
| Char | ASCII | Use |
|------|-------|-----|
| ~ | 126 | Section title start |
| # | 35 | Comment line start |
| . | 46 | MNEM/Unit separator |
| : | 58 | Value/Description separator |
| ; | 59 | Format/Array spacing separator |
| { | 123 | Format start |
| } | 125 | Format end |
| [ | 91 | Index start |
| ] | 93 | Index end |
| \| | 124 | Association separator |

**Allowed in Data Sections:** Period, Colon, Semicolon, Bar, Brackets, Braces can be used in Column Data values

### Real-Time Data Support

- STOP value can be NULL for real-time acquisition
- Data can be appended to file end
- ~ASCII must be last section in this case

---

## 6. Reserved Mnemonics

### ~Version Section
- VERS, WRAP, DLM

### ~Well Section
- STRT, STOP, STEP, NULL, COMP, WELL, FLD, LOC, STAT, PROV, CTRY, CNTY, API, UWI, LIC, SRVC, DATE, X, Y, LATI, LONG, GDAT, HZCS

### ~Parameter/~Log_Parameter
- RUN, APD, DREF, EREF, PDAT, RUNS, RUN_DEPTH, RUN_DATE, NMAT_DEPTH, DMAT_DEPTH, SMAT_DEPTH, MATR, MDEN, DTMA, FR_LR

### ~Core_Parameter
- C_SRS, C_TY, C_DATE, C_TOP, C_BS, C_RC, C_FM, C_DI, C_AC, C_AD

### ~Core_Definition
- CORT, CORB, CDES

### ~Drilling_Definition
- DDEP, DIST, HRS, ROP, WOB, RPM, TQ, PUMP, TSPM, GPM, ECD, TBR, RIG, CONTR

### ~Inclinometry_Parameter
- I_DT, I_CO, I_RF, I_AT, I_DC, I_KD, I_GD, I_ONS, I_OEW, I_CP, I_CS

### ~Inclinometry_Definition
- MD, TVD, AZIM, DEVI, RB, NSDR, EWDR, CLSR, TIEMD, TIETVD, TIEDEVI

### ~Tops_Definition
- TOPT, TOPB, TOPN, TOPSRC, TOPDR

### ~Test_Definition
- TSTT, TSTB, TSTN, DDES, ISIP, FSIP, RATE, BLOWD, TESTT

---

## 7. Sample Test File Analysis

The existing test file at `test_data/sample_3.0.las` demonstrates:

1. **Version 3.0 format** with COMMA delimiter
2. **Multiple curve formats**: F (float), E (exponential), S (string)
3. **3D Array channels**: NMR[1] through NMR[5] with time spacing
4. **Parameter zoning**: Different matrix values for depth intervals
5. **Multiple runs**: RUNS, RUN[1], RUN[2] structure
6. **Run-specific parameters**: Parameters associated with specific runs
7. **Depth interval parameters**: NMAT_Depth, DMAT_Depth for zones

---

## Sources

- [LAS 3.0 File Structures (PDF) - BC Energy Regulator](https://www.bc-er.ca/files/operations-documentation/Energy-Resource-Activity-Operations-Manual/Supporting-Documents/las3filestructure.pdf)
- [Log ASCII Standard (LAS) Version 3.0 - ResearchGate](https://www.researchgate.net/publication/254509443_Log_Ascii_Standard_LAS_Version_30)
- [Log ASCII Standard (LAS) Version 3.0 - OnePetro](https://onepetro.org/petrophysics/article/170932/Log-Ascii-Standard-las-Version-3-0)
- [LAS Format - USGS](https://www.usgs.gov/programs/national-geological-and-geophysical-data-preservation-program/las-format)
- [LAS Format - Geo-log 4 Documentation](https://docs.geolog4.com/en/las-format.html)
- [LAS 3.0 Help - Kansas Geological Survey](https://www.kgs.ku.edu/software/SS/HELP/las/index.html)
- [lasio (Python) - Read the Docs](https://lasio.readthedocs.io/en/v0.30/)
