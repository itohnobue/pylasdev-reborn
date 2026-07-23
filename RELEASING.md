# Release Process

This document describes how to create a new release of pylasdev.

## Prerequisites

- Push access to the repository
- Python 3.12+ with `uv` installed
- A [PyPI](https://pypi.org) account with Trusted Publishing configured for this repository (or a PyPI API token). See [Trusted Publishing Setup](#trusted-publishing-setup-one-time) below — this is a one-time setup required before the first PyPI publication.

> **Note:** The package is **not** currently published on PyPI. Step 4
> describes manual build and publishing via Trusted Publishing,
> but this requires the Trusted Publisher to be configured in PyPI project
> settings first (see [Trusted Publishing Setup](#trusted-publishing-setup-one-time)
> below). Until the Trusted Publisher is configured and the first release is
> published, `pip install pylasdev` will return a 404 error.
> For now, install from source: `git clone` + `pip install .` or `uv sync`.

## Steps

### 1. Run local checks

Run the full test suite, linting, and type checking locally before releasing:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest -v
```

### 2. Update the version

The single source of truth for the version is in `pyproject.toml`:

```toml
[project]
version = "X.Y.Z"
```

Bump the version according to [Semantic Versioning](https://semver.org/):
- **Patch** (`2.0.0` → `2.0.1`): bug fixes
- **Minor** (`2.0.0` → `2.1.0`): new features, backward compatible
- **Major** (`2.0.0` → `3.0.0`): breaking changes

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

### 4. Build and publish

Build the package and publish to PyPI:

```bash
uv build
uv publish
```

> **Note:** The package uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
> for PyPI authentication. Configure the Trusted Publisher in your PyPI project
> settings (see [Trusted Publishing Setup](#trusted-publishing-setup-one-time) below)
> before the first publication.

### 5. Verify the release

- Check the [GitHub Releases page](https://github.com/itohnobue/pylasdev-reborn/releases) for the new release
- **After first PyPI publication:** verify the package is available on [PyPI](https://pypi.org/project/pylasdev/):
  ```bash
  pip install pylasdev==X.Y.Z
  ```
- **Before first PyPI publication:** verify the release locally:
  ```bash
  git clone https://github.com/itohnobue/pylasdev-reborn.git --branch vX.Y.Z /tmp/pylasdev-test
  cd /tmp/pylasdev-test
  pip install .
  python -c "import pylasdev; print(pylasdev.__version__)"
  ```

## Trusted Publishing Setup (one-time)

To publish to PyPI with `uv publish`, configure authentication:
1. Go to your [PyPI project settings](https://pypi.org/manage/project/pylasdev/settings/publishing/)
2. Either add a PyPI API token, or set up a Trusted Publisher with:
   - **Owner**: `itohnobue`
   - **Repository**: `pylasdev-reborn`
   - **Environment**: (leave blank)
3. See `uv publish --help` for authentication options.
