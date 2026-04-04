"""Root conftest.py — ensures pytest always runs with the repo root as the CWD.

Tests use relative paths like Path("data/...") and Path("results/...").
This file changes the working directory to the repository root before any
test is collected, so tests pass regardless of where pytest is invoked from.
"""

import os
from pathlib import Path


def pytest_configure(config):
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
