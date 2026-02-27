# Contributing to Personalab

Thank you for your interest in contributing. This document explains how to set up the project, the conventions we follow, and how to submit changes.

## Getting started

```bash
# 1. Clone the repository
git clone <repo-url>
cd influencer

# 2. Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Verify everything works
pytest tests/ -v
```

## Project layout

```
src/personalab/       # Library source (domain-based structure)
scripts/              # CLI entry points
examples/             # Runnable examples
tests/                # Unit tests (mirrors src/ domain structure)
.cursor/rules/        # Architecture rules and coding conventions
```

## Architecture rules

Please read [`.cursor/rules/architecture_rules.md`](.cursor/rules/architecture_rules.md) before contributing. Key points:

1. **Organize by domain**, not by layer. Code that changes together lives together.
2. **Dependency inversion**: high-level code depends on `LLMClient` (protocol), not SDK classes directly. Only adapters in `llm/` import provider SDKs.
3. **No global state**: use constructor injection. No `np.random.seed()` or module-level mutable state.
4. **Typed configuration**: use `ProjectConfig`, not raw dicts, in function signatures.
5. **Tests per domain**: every domain directory must have a corresponding `tests/{domain}/` with unit tests.

## Coding conventions

### Python style

- Python 3.10+ — use `X | None` instead of `Optional[X]`, `list[str]` instead of `List[str]`.
- Type-annotate all public functions and method signatures (including `-> None` for void).
- Imports at module level (no imports inside functions unless lazy-loading heavy dependencies).
- Docstrings on all public classes and methods. Private methods need docstrings if non-obvious.

### YAML templates

- All variable placeholders use `${VAR}` syntax (uppercase, shell-style).
- Use `render_prompt()` for dict templates, `render_string()` for single strings.

### Tests

- Use `pytest`. Test files go in `tests/{domain}/test_{module}.py`.
- Use the `FakeLLMClient` from `conftest.py` for generation tests — no API calls in unit tests.
- Use `tmp_path` fixture for any file I/O in tests.
- Test both happy paths and error cases (missing keys, invalid inputs, edge cases).

## Making changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write your code** following the conventions above.

3. **Add or update tests** for any changed behavior.

4. **Run the test suite**:
   ```bash
   pytest tests/ -v
   ```

5. **Submit a pull request** with a clear description of what changed and why.

## Commit messages

Use clear, concise commit messages:

- `fix: correct base64 encoding in anchor parts`
- `feat: add render_string for single-string template substitution`
- `test: add generation domain tests`
- `docs: rewrite README with usage guide`

## Reporting issues

Open a GitHub issue with:

- A clear title and description.
- Steps to reproduce (if it's a bug).
- Expected vs. actual behavior.
- Python version and OS.
