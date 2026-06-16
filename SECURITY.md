# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.6.x   | Yes |
| 1.5.x   | Security fixes only |
| < 1.5   | Unsupported |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

To report a security issue, please email the maintainer at
[muharlyamovar@ufanipi.ru](mailto:muharlyamovar@ufanipi.ru) with:

- A clear description of the vulnerability
- Steps to reproduce (including a minimal test file if applicable)
- The affected version(s)
- Any known workarounds

You can expect a response within 72 hours. We request a 90-day disclosure
window before publishing any details about the vulnerability.

## Security Considerations for pylasdev

pylasdev reads LAS and DEV well-log files, which are typically untrusted
input. The library includes the following protections:

- **File size limits:** `max_file_size` parameter on all read functions prevents
  resource exhaustion from maliciously large files
- **Curve and data line caps:** `MAX_CURVES` (100,000) and `MAX_DATA_LINES`
  (10,000,000) prevent unbounded array allocations
- **Combined allocation guard:** `MAX_TOTAL_ELEMENTS` (1,000,000,000) prevents
  OOM from the product of moderate curve and line counts
- **NaN/Inf guards:** Numeric parsing rejects NaN and Infinity values to prevent
  downstream arithmetic corruption
- **Encoding sandbox:** File content is decoded in memory with a fallback chain;
  `latin-1` is the terminal fallback that decodes any byte sequence
