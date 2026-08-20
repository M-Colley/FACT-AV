"""Shared PySR search configuration for the FACT-AV symbolic-regression scripts.

Replaces the four copy-pasted ``create_model()`` / ``write_model_info()`` bodies in
``main_pysr_trust_calibration.py``, ``main_group_pysr_trust_calibration.py``,
``main_group_pysr_trust_calibration_more_predictors.py`` and
``main_personalized_pysr_trust_calibration.py``.

Pinned version
--------------
``requirements.txt`` pins **pysr 2.0.0b2** (SymbolicRegression.jl 2.0.0-beta.8).
Every keyword used here is accepted by 1.5.10, 2.0.0-alpha.11 and 2.0.0b2 alike;
``tests/test_pysr_api_compat.py`` asserts that on every CI run so a beta bump
fails in seconds rather than three hours into a fit.

Three deliberate choices versus the original inline configs
-----------------------------------------------------------
1. ``ncycles_per_iteration`` instead of ``ncyclesperiteration``.
   The old spelling is in PySR's ``DEPRECATED_KWARGS`` and still works in 2.0,
   but it emits a ``FutureWarning`` on every construction.

2. ``batching=False`` is **explicit**. This is the one behavioural default that
   changed between the two major versions::

       pysr 1.5.10 : batching = False,  batch_size = 50
       pysr 2.0.x  : batching = "auto", batch_size = None

   Under 2.0, ``"auto"`` switches mini-batched fitness evaluation ON whenever
   ``len(X) > 1000`` (batch_size 128 for N < 5000). In this repo that silently
   affects the fits on 2600/2310 rows (``run_all_data``) and on ``all_equal_df``
   (1740 rows): the search would optimise a 128-row minibatch MSE instead of the
   full-sample MSE, so the discovered equations would no longer match the ones
   committed under ``results/PySR/``. Pinning it keeps 1.x and 2.x comparable.

3. ``random_state`` / ``deterministic`` / ``parallelism`` are wired through to a
   command-line flag on all four scripts (see :func:`add_search_args`).

Reproducibility, precisely
--------------------------
``random_state`` **alone does not make a PySR run reproducible.** PySR's search
is stochastic and multithreaded by default; the seed only fully determines the
result when the search is also serial. That is what ``--deterministic``
does, and PySR enforces the pairing (``deterministic=True`` requires
``parallelism="serial"``). A publication run should therefore be::

    python main_pysr_trust_calibration.py --seed 0 --deterministic

which is slower than the threaded default but is the only mode whose equations
can be regenerated. Running without ``--deterministic`` logs a warning saying
so, and :func:`write_model_info` records the mode in the output file, so a
committed equation always states whether it is reproducible.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import sympy
from pysr import PySRRegressor

logger = logging.getLogger(__name__)

DEFAULT_SEED = 0

# Operator set. Kept identical to the previous inline configs so this change is a
# pure refactor. Note that mIoU is on a 0-100 scale in the shipped data, which
# makes ``exp``/``^``/``cube``/``quart`` numerically marginal at precision=32 and
# lets ``tan``/``inv`` introduce poles inside the observed range -- see the audit
# notes before treating this operator list as settled.
BINARY_OPERATORS = ["+", "-", "*", "/", "^"]

UNARY_OPERATORS = [
    "sin",
    "square",
    "tan",
    "cos",
    "cube",
    "tanh",
    "sqrt",
    "abs",
    "log",
    "exp",
    "cos2(x)=cos(x)^2",
    "quart(x) = x^4",
    "inv(x) = 1/x",
]

EXTRA_SYMPY_MAPPINGS = {
    "cos2": lambda x: sympy.cos(x) ** 2,
    "inv": lambda x: 1 / x,
    "quart": lambda x: x**4,
}


def create_model(
    niterations: int = 500,
    maxsize: int = 10,
    *,
    random_state: int | None = None,
    deterministic: bool = False,
    parallelism: str | None = None,
) -> PySRRegressor:
    """Build the PySR regressor used across all FACT-AV symbolic-regression runs.

    Parameters
    ----------
    niterations
        Search iterations. ``main_pysr_trust_calibration.py`` uses 300; the other
        three scripts use 500.
    maxsize
        Maximum expression complexity (nodes in the expression tree).
    random_state, deterministic, parallelism
        Reproducibility controls. For a repeatable run pass
        ``random_state=<int>, deterministic=True`` (which forces serial search).
    """
    if deterministic and parallelism not in (None, "serial"):
        raise ValueError('deterministic=True requires parallelism="serial"')

    return PySRRegressor(
        niterations=niterations,
        binary_operators=BINARY_OPERATORS,
        unary_operators=UNARY_OPERATORS,
        extra_sympy_mappings=EXTRA_SYMPY_MAPPINGS,
        constraints={"^": (-1, 1)},
        ncycles_per_iteration=2500,
        maxsize=maxsize,
        precision=32,
        turbo=True,
        # Pin the one default that changed in PySR 2.0 (see module docstring).
        batching=False,
        random_state=random_state,
        deterministic=deterministic,
        parallelism="serial" if deterministic else parallelism,
    )


def add_search_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the shared ``--seed`` / ``--deterministic`` / ``--parallelism`` flags.

    All four PySR entry points use this, so a publication run is the same command
    line everywhere and the mode is recorded in every output file.
    """
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for the PySR search (only fully effective with --deterministic).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Make the search reproducible. Forces parallelism='serial', which is "
            "considerably slower -- use it for a final publication run."
        ),
    )
    parser.add_argument(
        "--parallelism",
        choices=("serial", "multithreading", "multiprocessing"),
        default=None,
        help="PySR parallelism mode. Ignored (forced to 'serial') with --deterministic.",
    )
    return parser


def model_factory(args: argparse.Namespace, *, niterations: int = 500, maxsize: int = 10):
    """Return a zero-argument callable that builds a fresh, seeded regressor.

    A *factory* rather than a single instance on purpose: reusing one
    ``PySRRegressor`` across fits lets Julia-side search state (the equation
    hall-of-fame) bleed from one dataset into the next.
    """
    if not getattr(args, "deterministic", False):
        logger.warning(
            "Running WITHOUT --deterministic: PySR's search is stochastic and "
            "threaded, so --seed %s will not make these equations reproducible. "
            "Use --deterministic for the publication run.",
            getattr(args, "seed", DEFAULT_SEED),
        )

    def factory() -> PySRRegressor:
        return create_model(
            niterations=niterations,
            maxsize=maxsize,
            random_state=args.seed,
            deterministic=args.deterministic,
            parallelism=args.parallelism,
        )

    return factory


def _provenance_lines(model: PySRRegressor) -> list[str]:
    """Header recording how this equation was produced.

    Without this, a committed ``model_info_*.txt`` is an equation with no way to
    tell whether it came from a seeded deterministic search or an unseeded
    threaded one -- i.e. no way to tell whether it can be regenerated at all.
    """
    import pysr

    deterministic = bool(getattr(model, "deterministic", False))
    return [
        "PROVENANCE",
        f"generated_utc: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"pysr_version: {pysr.__version__}",
        f"random_state: {getattr(model, 'random_state', None)}",
        f"deterministic: {deterministic}",
        f"parallelism: {getattr(model, 'parallelism', None)}",
        f"niterations: {getattr(model, 'niterations', None)}",
        f"maxsize: {getattr(model, 'maxsize', None)}",
        f"reproducible: {'yes' if deterministic else 'NO - stochastic threaded search'}",
        "",
    ]


def write_model_info(model: PySRRegressor, output_path: Path | str) -> None:
    """Write the SymPy / LaTeX / LaTeX-table representation of the best equation.

    ``encoding="utf-8"`` is explicit: the previous inline copies used the platform
    default, which is cp1252 on Windows and raises on some LaTeX output.
    """
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(_provenance_lines(model)))
        handle.write("SYMPY\n")
        handle.write(str(model.sympy()))
        handle.write("\n\nLATEX\n")
        handle.write(str(model.latex()))
        handle.write("\n\nLATEX TABLE\n")
        handle.write(str(model.latex_table()))
