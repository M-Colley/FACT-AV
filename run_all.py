#!/usr/bin/env python3
"""Cross-platform runner for the FACT-AV analysis pipelines.

Replaces ``all_pysr.bat`` and ``MLP/all_mlp.bat``, which were Windows-only while
the README advertises macOS and Linux support. Those ``.bat`` files also swallowed
failures: each ``python x.py`` line ran regardless of whether the previous one
crashed, so a broken stage was invisible unless you scrolled back through the
output. This runner stops at the first failure (unless ``--keep-going``) and
prints a summary table at the end.

Examples
--------
List what is available::

    python run_all.py --list

Everything except the symbolic-regression searches (which take hours)::

    python run_all.py ml mlp stats figures

A reproducible publication run of the PySR pipelines::

    python run_all.py pysr --seed 0 --deterministic

Every stage forwards ``--seed`` where the underlying script accepts one, so a
whole-pipeline run is described by a single command line.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    script: Path
    args: list[str] = field(default_factory=list)
    cwd: Path = REPO_ROOT
    # Which shared flags this script understands.
    accepts_seed: bool = False
    accepts_pysr_flags: bool = False


STAGES: dict[str, list[Stage]] = {
    "ml": [
        Stage(
            "ml-approaches",
            "Feature importance across RF / XGBoost / LightGBM / CatBoost / TabPFN",
            REPO_ROOT / "ML-approaches.py",
        )
    ],
    "cv": [
        Stage(
            "cross-validation",
            "Repeated participant-grouped CV with a mean baseline (headline metrics)",
            REPO_ROOT / "cross_validation.py",
        )
    ],
    "mlp": [
        Stage(
            "mlp-train",
            "Train the MLP trust classifier",
            REPO_ROOT / "MLP" / "train.py",
            cwd=REPO_ROOT / "MLP",
            accepts_seed=True,
        ),
        Stage(
            "mlp-eval",
            "Evaluate the best MLP checkpoint (confusion matrix, calibration)",
            REPO_ROOT / "MLP" / "eval.py",
            cwd=REPO_ROOT / "MLP",
            accepts_seed=True,
        ),
    ],
    "stats": [
        Stage(
            "mixed-effects",
            "Linear mixed-effects baseline (ICC, moderation tests)",
            REPO_ROOT / "mixed_effects_baseline.py",
        )
    ],
    "figures": [
        Stage(
            "publication-figures",
            "Forest plot, importance heatmap, mIoU-trust panel, PDP/ICE",
            REPO_ROOT / "publication_figures.py",
        ),
        Stage(
            "explainability",
            "SHAP interactions, DiCE counterfactuals, Anchors rules",
            REPO_ROOT / "explainability_extras.py",
        ),
    ],
    "pysr": [
        Stage(
            "pysr-basic",
            "Symbolic regression per INTRODUCTION x SCENARIO cell",
            REPO_ROOT / "main_pysr_trust_calibration.py",
            accepts_pysr_flags=True,
        ),
        Stage(
            "pysr-group",
            "Symbolic regression on the equal-trust / variable-trust split",
            REPO_ROOT / "main_group_pysr_trust_calibration.py",
            accepts_pysr_flags=True,
        ),
        Stage(
            "pysr-more-predictors",
            "Symbolic regression with the full predictor set",
            REPO_ROOT / "main_group_pysr_trust_calibration_more_predictors.py",
            accepts_pysr_flags=True,
        ),
        Stage(
            "pysr-personalized",
            "Per-participant symbolic regression",
            REPO_ROOT / "main_personalized_pysr_trust_calibration.py",
            accepts_pysr_flags=True,
        ),
    ],
}

DEFAULT_GROUPS = ["ml", "cv", "mlp", "stats", "figures"]


def build_command(stage: Stage, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(stage.script), *stage.args]
    if stage.accepts_pysr_flags:
        command += ["--seed", str(args.seed)]
        if args.deterministic:
            command.append("--deterministic")
    elif stage.accepts_seed:
        command += ["--seed", str(args.seed)]
    return command


def run_stage(stage: Stage, args: argparse.Namespace) -> tuple[bool, float]:
    command = build_command(stage, args)
    print("=" * 78)
    print(f"[{stage.name}] {stage.description}")
    print(f"$ {' '.join(command)}")
    print("=" * 78, flush=True)

    if args.dry_run:
        return True, 0.0

    started = time.monotonic()
    result = subprocess.run(command, cwd=stage.cwd)
    elapsed = time.monotonic() - started
    ok = result.returncode == 0
    if not ok:
        print(f"[{stage.name}] FAILED with exit code {result.returncode}", file=sys.stderr)
    return ok, elapsed


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Groups: {', '.join(STAGES)}. Default: {' '.join(DEFAULT_GROUPS)} (pysr is opt-in).",
    )
    # Validated below rather than via ``choices=``: with ``nargs="*"`` argparse
    # also checks the *default list itself* against choices and rejects it.
    parser.add_argument(
        "groups",
        nargs="*",
        metavar="GROUP",
        default=None,
        help="Stage groups to run. 'pysr' is excluded by default because the searches take hours.",
    )
    parser.add_argument("--list", action="store_true", help="List the stages and exit.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the commands without running them."
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failing stage instead of stopping (the .bat files always did this).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed forwarded to every stage that accepts one."
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Forwarded to the PySR stages; required for reproducible equations.",
    )
    args = parser.parse_args(argv)
    if not args.groups:
        args.groups = list(DEFAULT_GROUPS)
    unknown = [group for group in args.groups if group not in STAGES]
    if unknown:
        parser.error(f"unknown group(s): {', '.join(unknown)}. Choose from: {', '.join(STAGES)}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list:
        for group, stages in STAGES.items():
            marker = "" if group in DEFAULT_GROUPS else "   (opt-in)"
            print(f"\n{group}{marker}")
            for stage in stages:
                print(f"  {stage.name:<22} {stage.description}")
        return 0

    selected = [stage for group in args.groups for stage in STAGES[group]]
    missing = [s.name for s in selected if not s.script.exists()]
    if missing:
        print(f"Missing scripts: {missing}", file=sys.stderr)
        return 2

    results = []
    for stage in selected:
        ok, elapsed = run_stage(stage, args)
        results.append((stage.name, ok, elapsed))
        if not ok and not args.keep_going:
            print("\nStopping (use --keep-going to run the remaining stages).", file=sys.stderr)
            break

    print("\n" + "=" * 78)
    print(f"{'STAGE':<24}{'RESULT':<10}{'TIME':>10}")
    print("-" * 78)
    for name, ok, elapsed in results:
        print(f"{name:<24}{'ok' if ok else 'FAILED':<10}{elapsed:>9.1f}s")
    print("=" * 78)

    failed = [name for name, ok, _ in results if not ok]
    skipped = len(selected) - len(results)
    if skipped:
        print(f"{skipped} stage(s) not run.", file=sys.stderr)
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
