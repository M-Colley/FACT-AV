"""Root conftest.py — runs each test with the repo root as the CWD.

Tests use relative paths like Path("data/...") and Path("results/...").
Rather than mutating the process-wide working directory in ``pytest_configure``
(which leaks into anything else running in-process and can mask path bugs),
an autouse fixture switches to the repo root per test and ``monkeypatch``
restores the original CWD afterward.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
