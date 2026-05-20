# Contributing

This is a master's thesis project at Politechnika Lubelska. External
contributions are not actively solicited until after the public defence,
but well-scoped bug reports and constructive feedback via GitHub issues
are very welcome.

## Development setup

```powershell
git clone https://github.com/<your-username>/drone-detector.git
cd drone-detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1                 # Windows
# source .venv/bin/activate                  # Linux / macOS
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

Confirm the install works without an audio device:

```powershell
pytest -q
ruff check src tests
```

## Coding standards

* Python ≥ 3.10. The project uses modern syntax (PEP 604 unions, etc.).
* Line length **100**, enforced by `ruff` and configured in
  `pyproject.toml`.
* Type hints are required on every public function and class.
* Docstrings are required on every public module and public class.
* `from __future__ import annotations` at the top of every module so
  forward references work consistently.

## Commit convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Meaning |
|---|---|
| `feat:`  | a new feature |
| `fix:`   | a bug fix |
| `docs:`  | documentation only |
| `chore:` | housekeeping that does not change code behaviour |
| `build:` | dependencies or packaging |
| `ci:`    | GitHub Actions workflow changes |
| `test:`  | tests only |
| `refactor:` | code change that does not change behaviour |
| `perf:`  | performance improvement |

A scope in parentheses is encouraged, e.g. `feat(live): downmix to mono
before resampling`.

## Pull-request checklist

1. `ruff check src tests` passes.
2. `pytest -q` passes.
3. New behaviour has at least one smoke test.
4. New public API has type hints and a docstring.
5. User-visible changes are reflected in `README.md` or `docs/`.

## Safety boundary

This project is **defensive early-warning research**. Pull requests must
not introduce target assignment, weapon guidance, engagement coordinate
generation, or any analogous offensive capability. The existing safety
boundary (documented in the README and the dashboard sidebar) is
non-negotiable.
