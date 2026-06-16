# Contributing to pylasdev

## Development Setup

```bash
git clone https://github.com/itohnobue/pylasdev-reborn.git
cd pylasdev-reborn
uv sync --extra dev
```

This installs all runtime and development dependencies including pytest, mypy, ruff, and bandit.

## Commands

| Command | Purpose |
|---------|---------|
| `uv run pytest -v` | Run the full test suite |
| `uv run pytest -v -m "not slow"` | Run tests excluding slow tests |
| `uv run pytest --cov=src/pylasdev --cov-report=term` | Run tests with coverage report |
| `uv run ruff check src/ tests/` | Lint all source and test files |
| `uv run ruff format src/ tests/` | Auto-format code |
| `uv run mypy src/` | Run strict type checking |

## Pull Request Workflow

1. **Fork and branch:** Create a feature branch from `main`
2. **Write code:** Follow existing code style and conventions
3. **Run checks:** All linting, formatting, and type checks must pass:
   ```bash
   uv run ruff check src/ tests/ && uv run mypy src/
   ```
4. **Run tests:** All tests must pass with coverage above 85%:
   ```bash
   uv run pytest -v
   ```
5. **Submit PR:** Open a pull request against the `main` branch with a clear description
6. **CI:** GitHub Actions will run the full suite (lint, typecheck, test, security scan) automatically

## Code Style

- Python 3.12+ with full type hints (`strict = true` mypy config)
- Double quotes for strings (enforced by ruff format)
- 100 character line length
- Imports organized with isort (first-party: `pylasdev`)
- Docstrings in Google-style (the codebase uses Google-style exclusively)
- All public functions must have docstrings with Args, Returns, Raises sections

## Testing

- Tests live in `tests/` and use pytest
- Test data files are in `test_data/` (18 sample LAS/DEV files)
- New features require tests; bug fixes require regression tests
- Minimum 85% branch coverage enforced by CI

## Security

If you discover a security vulnerability, please do **not** open a public issue.
See [SECURITY.md](SECURITY.md) for reporting instructions.
