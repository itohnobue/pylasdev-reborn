# Release Process

This document describes how to create a new release of pylasdev.

## Prerequisites

- Push access to the repository
- Python 3.12+ with `uv` installed
- A [PyPI](https://pypi.org) account with trusted publisher configured for this repository (or a PyPI API token)

## Steps

### 1. Ensure CI is green

All checks on the `main` branch must pass before releasing.

### 2. Update the version

The single source of truth for the version is in `pyproject.toml`:

```toml
[project]
version = "X.Y.Z"
```

Bump the version according to [Semantic Versioning](https://semver.org/):
- **Patch** (`1.6.0` → `1.6.1`): bug fixes
- **Minor** (`1.6.0` → `1.7.0`): new features, backward compatible
- **Major** (`1.6.0` → `2.0.0`): breaking changes

Commit the version bump:

```bash
git add pyproject.toml
git commit -m "Bump version to X.Y.Z"
```

### 3. Create an annotated tag

Create an annotated tag matching the version:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
```

Push the commit and tag:

```bash
git push origin main
git push origin vX.Y.Z
```

### 4. Automated release (CI)

Pushing a `v*` tag triggers two workflows:

1. **CI** (`.github/workflows/ci.yml`) — Runs the full test suite, linting, type checking, and security scans on the tagged commit to ensure everything is clean.

2. **Release** (`.github/workflows/release.yml`) — Builds the package with `uv build`, publishes to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/), and creates a GitHub Release with auto-generated release notes.

### 5. Verify the release

- Check the [GitHub Releases page](https://github.com/itohnobue/pylasdev-reborn/releases) for the new release
- Verify the package is available on [PyPI](https://pypi.org/project/pylasdev/):
  ```bash
  pip install pylasdev==X.Y.Z
  ```

## Trusted Publishing Setup (one-time)

To enable automated PyPI publishing, configure Trusted Publishing in PyPI:
1. Go to your [PyPI project settings](https://pypi.org/manage/project/pylasdev/settings/publishing/)
2. Add a new Trusted Publisher with:
   - **Owner**: `itohnobue`
   - **Repository**: `pylasdev-reborn`
   - **Workflow**: `release.yml`
   - **Environment**: (leave blank)
