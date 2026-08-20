#!/usr/bin/env python3
"""Regenerate ``requirements.lock`` from the currently installed environment.

``requirements.txt`` states the *supported* version ranges (``>=``), which is
right for day-to-day use but means two people running the pipeline six months
apart get different numbers and neither can tell why. ``requirements.lock``
records the exact transitive closure that produced the committed results.

This walks the dependency graph from the packages named in ``requirements.txt``
and pins whatever is installed, rather than using ``pip freeze``, which would
also capture every unrelated package in the interpreter (Jupyter, linters, other
projects sharing the environment).

Run from the repo root::

    python tools/make_lock.py

Then verify a fresh environment reproduces it::

    python -m venv .venv && .venv/bin/pip install -r requirements.lock
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib.metadata import PackageNotFoundError, distribution, requires
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def canonical(name: str) -> str:
    """Strip version specifiers, extras and environment markers from a requirement."""
    return re.split(r"[<>=!~\[; ]", name.strip())[0].lower().replace("_", "-")


def read_roots(requirements_path: Path) -> list[str]:
    roots = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            roots.append(canonical(line))
    return roots


def resolve_closure(roots: list[str]) -> tuple[dict[str, str], list[str]]:
    """Depth-first walk of the installed dependency graph."""
    seen: set[str] = set()
    versions: dict[str, str] = {}
    missing: list[str] = []

    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            missing.append(name)
            continue
        versions[dist.metadata["Name"]] = dist.version
        for requirement in requires(name) or []:
            # Extras-gated dependencies (``; extra == "dev"``) are not installed
            # by a plain ``pip install pkg``, so they do not belong in the lock.
            if "extra ==" in requirement:
                continue
            stack.append(canonical(requirement))

    return versions, missing


def render(versions: dict[str, str]) -> str:
    header = f"""# Fully pinned environment for reproducing the published results.
#
# requirements.txt states the *supported* ranges; this file states the exact
# transitive closure a result was produced with. Regenerate with:
#
#     python tools/make_lock.py
#
# and install with:
#
#     pip install -r requirements.lock
#
# Direct dependencies are listed in requirements.txt; everything else here is a
# transitive dependency captured for reproducibility.
# Packages: {len(versions)}
"""
    body = "\n".join(
        f"{name}=={version}"
        for name, version in sorted(versions.items(), key=lambda item: item[0].lower())
    )
    return header + body + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requirements", type=Path, default=REPO_ROOT / "requirements.txt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "requirements.lock")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the lock file is out of date instead of rewriting it.",
    )
    args = parser.parse_args(argv)

    roots = read_roots(args.requirements)
    versions, missing = resolve_closure(roots)
    rendered = render(versions)

    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != rendered:
            print(
                f"{args.output.name} is out of date — run: python tools/make_lock.py",
                file=sys.stderr,
            )
            return 1
        print(f"{args.output.name} is up to date ({len(versions)} packages).")
        return 0

    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output} with {len(versions)} packages.")
    if missing:
        # Platform-specific wheels (CUDA/Metal variants) are absent on other
        # platforms by design; reported, not treated as an error.
        print(f"Not installed here, omitted: {sorted(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
