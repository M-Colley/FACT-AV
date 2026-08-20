"""Shared pytest fixtures and per-test working-directory handling.

Two jobs:

1. **Working directory.** Tests use relative paths like ``Path("data/...")`` and
   ``Path("results/...")``. Rather than mutating the process-wide working
   directory in ``pytest_configure`` (which leaks into anything else running
   in-process and can mask path bugs), an autouse fixture switches to the repo
   root per test and ``monkeypatch`` restores the original CWD afterward.

2. **One import of ``ML-approaches.py`` per session.** That module imports shap,
   catboost, xgboost, lightgbm, tabpfn and torch at module scope, which costs
   tens of seconds. Two test modules need it, and each was paying that cost
   separately. The session-scoped fixture below loads it once.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)


def load_hyphenated_module(module_name: str, filename: str):
    """Import a module whose filename is not a valid Python identifier.

    ``ML-approaches.py`` contains a hyphen, so it cannot be imported with a plain
    ``import`` statement.
    """
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def ml_approaches():
    """The ``ML-approaches.py`` module, imported once for the whole session."""
    return load_hyphenated_module("ml_approaches", "ML-approaches.py")
