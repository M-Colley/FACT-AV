"""Shared PySR search configuration for the FACT-AV symbolic-regression scripts.

Replaces the four copy-pasted ``create_model()`` / ``write_model_info()`` bodies in
``main_pysr_trust_calibration.py``, ``main_group_pysr_trust_calibration.py``,
``main_group_pysr_trust_calibration_more_predictors.py`` and
``main_personalized_pysr_trust_calibration.py``.

Every keyword used here is accepted by **both** pysr 1.5.10 and pysr
2.0.0-alpha.11, so this module needs no edit when 2.0 ships.

Three deliberate changes versus the previous inline configs
----------------------------------------------------------
1. ``ncycles_per_iteration`` instead of ``ncyclesperiteration``.
   The old spelling is in PySR's ``DEPRECATED_KWARGS`` and still works in 2.0,
   but it emits a ``FutureWarning`` on every construction.

2. ``batching=False`` is now **explicit**. This is the only behavioural default
   that changed between the two versions::

       pysr 1.5.10 : batching = False,  batch_size = 50
       pysr 2.0.0a : batching = "auto", batch_size = None

   Under 2.0, ``"auto"`` switches mini-batched fitness evaluation ON whenever
   ``len(X) > 1000`` (batch_size 128 for N < 5000). In this repo that silently
   affects the fits on 2600/2310 rows (``run_all_data``) and on ``all_equal_df``
   (1740 rows): the search would optimise a 128-row minibatch MSE instead of the
   full-sample MSE, so the discovered equations would no longer match the ones
   committed under ``results/PySR/``. Pinning it keeps 1.x and 2.x comparable.

3. ``random_state`` / ``deterministic`` / ``parallelism`` are exposed so a run
   can be made reproducible. PySR's search is stochastic; with the previous
   settings (``random_state=None``, ``deterministic=False``, threaded search) the
   committed equations cannot be regenerated. Note that ``deterministic=True``
   *requires* ``parallelism="serial"`` and is therefore much slower -- use it for
   a final publication run, not for exploration.
"""

from __future__ import annotations

from pathlib import Path

import sympy
from pysr import PySRRegressor

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


def write_model_info(model: PySRRegressor, output_path: Path | str) -> None:
    """Write the SymPy / LaTeX / LaTeX-table representation of the best equation.

    ``encoding="utf-8"`` is explicit: the previous inline copies used the platform
    default, which is cp1252 on Windows and raises on some LaTeX output.
    """
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("SYMPY\n")
        handle.write(str(model.sympy()))
        handle.write("\n\nLATEX\n")
        handle.write(str(model.latex()))
        handle.write("\n\nLATEX TABLE\n")
        handle.write(str(model.latex_table()))
