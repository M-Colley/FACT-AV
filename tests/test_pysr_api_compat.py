"""Guard tests for the PySR API surface this repository depends on.

Purpose: when PySR 2.0 lands and you run ``pip install -U pysr``, these tests fail
immediately and specifically instead of the change surfacing three hours into a
fitting run. They inspect signatures and defaults only -- no search is executed,
so the module runs in well under a second once Julia has been imported.

Verified against pysr 1.5.10, 2.0.0-alpha.11 and 2.0.0b2 (the pinned version).

Findings from the diff of those versions, for context:
  * No constructor parameter this repo passes was ever removed (94 -> 100 params
    across 1.5 -> 2.0, purely additive).
  * Exactly two defaults changed: ``batching`` (False -> "auto") and
    ``batch_size`` (50 -> None). ``pysr_config.create_model`` pins ``batching``.
  * ``ncyclesperiteration`` is still auto-renamed but emits a FutureWarning.
  * ``_maybe_create_inline_operators`` -- the private helper behind the custom
    ``cos2``/``quart``/``inv`` operators -- changed signature twice, so the
    bridge below dispatches on the actual parameter names rather than on a
    version number.
"""

import inspect
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pysr = pytest.importorskip("pysr", reason="PySR / Julia not available in this environment")

from pysr import PySRRegressor  # noqa: E402

from pysr_config import create_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def signature():
    return inspect.signature(PySRRegressor.__init__)


# ---------------------------------------------------------------------------
# 1. Every keyword pysr_config passes must still be accepted.
# ---------------------------------------------------------------------------

REQUIRED_KWARGS = [
    "niterations",
    "binary_operators",
    "unary_operators",
    "extra_sympy_mappings",
    "constraints",
    "ncycles_per_iteration",
    "maxsize",
    "precision",
    "turbo",
    "batching",
    "random_state",
    "deterministic",
    "parallelism",
]


@pytest.mark.parametrize("kwarg", REQUIRED_KWARGS)
def test_constructor_accepts_kwarg(signature, kwarg):
    assert kwarg in signature.parameters, (
        f"PySRRegressor no longer accepts {kwarg!r}. "
        "Check the PySR changelog and update pysr_config.create_model()."
    )


# ---------------------------------------------------------------------------
# 2. Defaults we rely on must not drift underneath us.
# ---------------------------------------------------------------------------


def test_batching_is_pinned_explicitly():
    """``batching`` differs across versions, so create_model must set it itself.

    1.5.10 defaults to False; 2.0 defaults to "auto", which enables minibatch
    fitness evaluation for N>1000 and changes which equations the search finds.
    """
    src = inspect.getsource(create_model)
    assert "batching=False" in src, (
        "create_model() must pass batching=False explicitly, otherwise PySR 2.0 "
        "silently minibatches the fits on >1000 rows."
    )


def test_batching_default_is_one_of_the_known_values(signature):
    default = signature.parameters["batching"].default
    assert default in (False, "auto"), (
        f"Unexpected `batching` default {default!r} -- a third behaviour has appeared; "
        "re-check what pysr_config should pin."
    )


@pytest.mark.parametrize(
    "kwarg,expected",
    [
        ("precision", 32),
        ("turbo", False),
        ("deterministic", False),
        ("random_state", None),
        ("model_selection", "best"),
    ],
)
def test_default_unchanged(signature, kwarg, expected):
    assert signature.parameters[kwarg].default == expected


# ---------------------------------------------------------------------------
# 3. The methods used to serialise results must still exist.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["fit", "predict", "sympy", "latex", "latex_table"])
def test_regressor_exposes_method(method):
    assert callable(getattr(PySRRegressor, method, None)), (
        f"PySRRegressor.{method}() is gone; pysr_config.write_model_info needs updating."
    )


# ---------------------------------------------------------------------------
# 4. The custom-operator + extra_sympy_mappings contract still holds.
#
# These touch a private helper deliberately: it is the mechanism the operator
# list in pysr_config relies on for cos2 / quart / inv, and a silent change there
# is exactly what we want to be told about.
# ---------------------------------------------------------------------------


def _inline_operator_kwargs(unary_operators, extra_sympy_mappings):
    """Bridge the signature changes of ``_maybe_create_inline_operators``.

    Three shapes exist in the wild, so dispatch on parameter names rather than on
    a version string::

        1.5.10   binary_operators=/unary_operators= lists, expression_spec=  -> tuple
        2.0.0a11 operators={arity: [...]},                 expression_spec=  -> dict
        2.0.0b2  operators={arity: [...]},                 supports_sympy=bool -> dict

    If a fourth shape appears this raises with the actual parameter list, rather
    than failing somewhere less obvious.
    """
    from pysr.sr import _maybe_create_inline_operators

    params = inspect.signature(_maybe_create_inline_operators).parameters
    kwargs = {"extra_sympy_mappings": extra_sympy_mappings}

    if "operators" in params:
        kwargs["operators"] = {1: list(unary_operators), 2: ["+"]}
    elif "unary_operators" in params:
        kwargs["binary_operators"] = ["+"]
        kwargs["unary_operators"] = list(unary_operators)
    else:  # pragma: no cover - only reachable on an unreleased PySR
        raise AssertionError(
            f"Unrecognised _maybe_create_inline_operators signature: {list(params)}"
        )

    # ``expression_spec=ExpressionSpec()`` was replaced by a plain
    # ``supports_sympy: bool`` in 2.0.0b2.
    if "supports_sympy" in params:
        kwargs["supports_sympy"] = True
    elif "expression_spec" in params:
        from pysr.expression_specs import ExpressionSpec

        kwargs["expression_spec"] = ExpressionSpec()

    return kwargs


def _flatten(result):
    values = result.values() if isinstance(result, dict) else result
    return [op for group in values for op in group]


def test_custom_operator_requires_sympy_mapping():
    from pysr.sr import _maybe_create_inline_operators

    with pytest.raises(ValueError, match="extra_sympy_mappings"):
        _maybe_create_inline_operators(
            **_inline_operator_kwargs(["cos2(x)=cos(x)^2"], extra_sympy_mappings=None)
        )


def test_custom_operator_accepted_with_sympy_mapping():
    import sympy
    from pysr.sr import _maybe_create_inline_operators

    result = _maybe_create_inline_operators(
        **_inline_operator_kwargs(
            ["cos2(x)=cos(x)^2"],
            extra_sympy_mappings={"cos2": lambda x: sympy.cos(x) ** 2},
        )
    )
    # The definition string is rewritten to the bare function name.
    assert "cos2" in _flatten(result)


# ---------------------------------------------------------------------------
# 5. Deprecated spelling must not be reintroduced.
# ---------------------------------------------------------------------------


def test_repo_does_not_use_deprecated_ncyclesperiteration():
    """The old spelling still works in 2.0 but warns on every construction.

    Checks keyword arguments via the AST rather than a substring search, so prose
    mentioning the deprecated name in a docstring does not trip the test.
    """
    import ast

    candidates = list(REPO_ROOT.glob("main_*pysr*.py")) + [REPO_ROOT / "pysr_config.py"]
    offenders = []
    for path in candidates:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(
                kw.arg == "ncyclesperiteration" for kw in node.keywords
            ):
                offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, (
        f"Use ncycles_per_iteration (snake_case) instead; found the deprecated "
        f"keyword in: {offenders}"
    )


def test_deprecated_kwarg_shim_still_exists():
    """Sanity check that PySR has not removed the rename shim outright."""
    from pysr.deprecated import DEPRECATED_KWARGS

    assert DEPRECATED_KWARGS.get("ncyclesperiteration") == "ncycles_per_iteration"


def test_no_deprecation_warnings_on_construction():
    """Constructing the repo's model must not emit FutureWarnings.

    This is the test that catches a 2.x rename we have not migrated yet. The PySR
    scripts no longer call ``warnings.filterwarnings("ignore")`` at import time,
    which previously would have hidden exactly this.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        create_model(niterations=1, maxsize=5)

    future = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
    assert not future, f"PySR emitted deprecation warnings: {future}"
